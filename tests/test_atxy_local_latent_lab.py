from dataclasses import replace
import hashlib
import json

import pytest
import torch

from molor.experimental.atxy_local_latent import AtxyLocalLatentCodec
from molor.experimental.atxy_local_latent_lab import (
    LocalLatentContract,
    _attention_metrics,
    _evaluate_latent_sweep,
    _gate_joint_arms,
    _load_contract,
    _project_storage,
    _source_tree_identity,
    _stack_queries,
    _summarize_records,
    _write_json_exclusive,
)
from molor.experimental.hashed_kv_lab import CapturedLayer


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gates(**overrides: float) -> dict[str, float]:
    values = {
        "min_projected_compression_ratio": 1.0,
        "max_attention_kl_mean": 0.01,
        "max_attention_kl_p99": 0.1,
        "min_attention_top1_agreement": 0.99,
        "min_attention_top_k_recall_mean": 0.98,
        "max_output_relative_l2_global": 0.05,
        "min_output_cosine_similarity_mean": 0.99,
    }
    values.update(overrides)
    return values


def _contract(tmp_path, *, gates: dict[str, float] | None = None) -> LocalLatentContract:
    return LocalLatentContract(
        path=tmp_path / "contract.json",
        sha256="0" * 64,
        payload={},
        model_path=tmp_path / "model",
        model_config_sha256="1" * 64,
        model_content_fingerprint="3" * 64,
        expected_model_config={
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 100,
        },
        dataset_manifest=tmp_path / "manifest.json",
        dataset_manifest_sha256="2" * 64,
        layers=(0,),
        calibration_tokens=8,
        heldout_tokens=8,
        heldout_query_positions=2,
        key_ranks=(2, 4),
        value_ranks=(2, 4),
        joint_rank_pairs=((2, 2), (4, 4)),
        projected_context_tokens=100,
        page_size=4,
        top_k=2,
        seed=1,
        gates=_gates() if gates is None else gates,
    )


def test_contract_loader_binds_local_model_manifest_and_fixed_matrix(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    model_config = model / "config.json"
    model_config.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "molor.atxy-local-latent-contract.v1",
        "status": "development-shadow-only",
        "model": {
            "path": "model",
            "config_sha256": _digest(model_config),
            "content_fingerprint": "a" * 64,
            "expected_config": {
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
            },
        },
        "dataset": {
            "manifest": "manifest.json",
            "manifest_sha256": _digest(manifest),
            "calibration_tokens": 4096,
            "heldout_tokens": 4096,
        },
        "layers": [0, 13, 27],
        "heldout_query_positions": 8,
        "key_ranks": [64, 128],
        "value_ranks": [96, 128],
        "joint_rank_pairs": [{"key": 64, "value": 96}, {"key": 128, "value": 128}],
        "latent_dtype": "bfloat16",
        "basis_dtype": "float32",
        "projected_context_tokens": 40960,
        "page_size": 256,
        "top_k": 32,
        "seed": 8419,
        "gate_scope": "overall_and_each_selected_layer",
        "gates": _gates(),
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    contract = _load_contract(contract_path)

    assert contract.layers == (0, 13, 27)
    assert contract.joint_rank_pairs == ((64, 96), (128, 128))
    assert contract.model_path == model.resolve()
    manifest.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset manifest sha256 mismatch"):
        _load_contract(contract_path)


def test_source_tree_identity_is_complete_and_detects_drift(tmp_path) -> None:
    package = tmp_path / "molor"
    package.mkdir()
    first = package / "a.py"
    second = package / "nested" / "b.py"
    second.parent.mkdir()
    first.write_text("A = 1\n", encoding="utf-8")
    second.write_text("B = 2\n", encoding="utf-8")

    original = _source_tree_identity(package)
    repeated = _source_tree_identity(package)

    assert original == repeated
    assert original["file_count"] == 2
    assert [item["name"] for item in original["files"]] == ["a.py", "nested/b.py"]
    second.write_text("B = 3\n", encoding="utf-8")
    assert _source_tree_identity(package)["content_fingerprint"] != original["content_fingerprint"]


def test_report_writer_exclusively_creates_lf_json(tmp_path) -> None:
    output = tmp_path / "report.json"
    _write_json_exclusive(output, {"value": 1})

    assert output.read_bytes() == b'{\n  "value": 1\n}\n'
    with pytest.raises(FileExistsError):
        _write_json_exclusive(output, {"value": 2})


def test_stack_queries_uses_layer_position_head_channel_order() -> None:
    captures = tuple(
        CapturedLayer(
            layer_index=layer,
            query_positions=(5, 6),
            queries=torch.full((2, 2, 3), float(layer + 1)),
            keys=torch.zeros(1, 7, 3),
            values=torch.zeros(1, 7, 3),
        )
        for layer in range(2)
    )

    stacked = _stack_queries(captures)

    assert stacked.shape == (2, 2, 2, 3)
    assert torch.equal(stacked[1], torch.full((2, 2, 3), 2.0))


def test_attention_metrics_identity_and_output_error_are_separable() -> None:
    weights = torch.tensor([0.1, 0.2, 0.7])
    output = torch.tensor([2.0, -1.0])
    identity = _attention_metrics(weights, weights, output, output, top_k=2)
    changed_output = _attention_metrics(weights, weights, output, output * 0.5, top_k=2)

    assert identity["attention_kl"] == pytest.approx(0.0, abs=1e-8)
    assert identity["attention_top1_agreement"] is True
    assert identity["output_relative_l2"] == 0.0
    assert changed_output["attention_kl"] == pytest.approx(0.0, abs=1e-8)
    assert changed_output["output_relative_l2"] == pytest.approx(0.5)


def test_full_rank_latent_sweep_matches_dense_attention() -> None:
    generator = torch.Generator().manual_seed(71)
    keys = torch.randn(1, 4, 1, 3, generator=generator)
    queries = torch.randn(1, 2, 2, 3, generator=generator)
    values = torch.randn(1, 4, 1, 3, generator=generator) + 2.0
    codec = AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=3)

    records, identity = _evaluate_latent_sweep(
        codec,
        queries,
        keys,
        values,
        query_positions=(2, 3),
        layer_indices=(0,),
        key_ranks=(3,),
        value_ranks=(3,),
        joint_rank_pairs=((3, 3),),
        top_k=2,
    )
    summary = _summarize_records(records, by_layer=False)
    joint = next(item for item in summary if item["variant"] == "K+V")

    assert len(records) == 12
    assert identity["shapes"]["keys"] == [1, 4, 1, 3]
    assert joint["attention_kl_mean"] == pytest.approx(0.0, abs=1e-6)
    assert joint["attention_top1_agreement"] == 1.0
    assert joint["output_relative_l2_global"] < 1e-5


def test_storage_projection_and_worst_layer_gates_are_fail_closed(tmp_path) -> None:
    generator = torch.Generator().manual_seed(81)
    keys = torch.randn(1, 8, 1, 4, generator=generator, dtype=torch.bfloat16)
    queries = torch.randn(1, 2, 2, 4, generator=generator, dtype=torch.bfloat16)
    values = torch.randn(1, 8, 1, 4, generator=generator, dtype=torch.bfloat16)
    codec = AtxyLocalLatentCodec.fit(
        keys,
        queries,
        values,
        max_rank=4,
        latent_dtype=torch.bfloat16,
    )
    contract = _contract(tmp_path)
    storage = _project_storage(codec, contract)
    perfect = {
        "arm": "joint-k2-v2",
        "variant": "K+V",
        "evaluations": 4,
        "attention_kl_mean": 0.0,
        "attention_kl_p99": 0.0,
        "attention_top1_agreement": 1.0,
        "attention_top_k_recall_mean": 1.0,
        "output_relative_l2_global": 0.0,
        "output_cosine_similarity_mean": 1.0,
    }
    full_rank = {**perfect, "arm": "joint-k4-v4"}
    by_layer = [
        {**perfect, "evaluations": 4, "layer": 0},
        {**full_rank, "evaluations": 4, "layer": 0},
    ]

    decisions, selected = _gate_joint_arms(
        contract,
        [perfect, full_rank],
        by_layer,
        storage,
    )

    assert storage["joint-k2-v2"]["compression_ratio"] > 1
    assert storage["joint-k2-v2"]["full_model_dimensional_projection"]["layer_count"] == 1
    assert selected == "joint-k2-v2"
    assert decisions[0]["all_fixed_development_gates_passed"] is True

    strict = replace(contract, gates=_gates(min_projected_compression_ratio=100.0))
    decisions, selected = _gate_joint_arms(strict, [perfect, full_rank], by_layer, storage)
    assert selected is None
    assert all(not decision["checks"]["projected_compression_ratio"] for decision in decisions)

    with pytest.raises(ValueError, match="exactly cover fixed layers"):
        _gate_joint_arms(contract, [perfect, full_rank], [], storage)


def test_storage_projection_rejects_codec_shape_outside_contract(tmp_path) -> None:
    contract = _contract(tmp_path)
    generator = torch.Generator().manual_seed(91)
    keys = torch.randn(1, 4, 1, 3, generator=generator)
    queries = torch.randn(1, 2, 2, 3, generator=generator)
    values = torch.randn(1, 4, 1, 3, generator=generator)
    codec = AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=3)

    with pytest.raises(ValueError, match="fixed contract shape"):
        _project_storage(codec, contract)
