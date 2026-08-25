import hashlib
import json

import pytest
import torch

import molor.experimental.atxy_page_oracle_lab as oracle_lab
from molor.experimental.atxy_page_oracle_lab import (
    DevelopmentCase,
    QueryCheckpoint,
    _capture_independent_prefixes,
    _evaluate_captured_layer,
    _local_model_stack_identity,
    _load_dataset_manifest,
    _seal_report,
    _summarize_budget_records,
    _summarize_minimum_mass_records,
    _summarize_physical_records,
    _token_ids,
    _verify_dataset_snapshot,
    _write_json_exclusive,
)
from molor.experimental.hashed_kv_lab import CapturedLayer


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_payload(source_sha256: str) -> dict:
    return {
        "schema_version": "molor.atxy-page-oracle-development-dataset.v1",
        "status": "development-only",
        "cases": [
            {
                "id": "mixed-development-01",
                "path": "prompt.txt",
                "sha256": source_sha256,
                "token_count": 8,
                "queries": [
                    {"token_count": 4, "support_token_spans": [[1, 2]]},
                    {"token_count": 8, "support_token_spans": [[4, 6]]},
                ],
            }
        ],
    }


class CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens
        return {"input_ids": list(range(len(text)))}


def test_manifest_loader_checks_source_identity_and_causal_queries(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("abcdefgh", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_payload(_digest(prompt))), encoding="utf-8")

    snapshot = _load_dataset_manifest(manifest)
    cases = snapshot.cases

    assert len(cases) == 1
    assert cases[0].case_id == "mixed-development-01"
    assert tuple(query.token_count for query in cases[0].queries) == (4, 8)
    assert _token_ids(CharacterTokenizer(), cases[0]).tolist() == list(range(8))
    prompt.write_text("drifted!", encoding="utf-8")
    assert _token_ids(CharacterTokenizer(), cases[0]).tolist() == list(range(8))
    with pytest.raises(RuntimeError, match="source bytes changed"):
        _verify_dataset_snapshot(snapshot)


def test_manifest_rejects_missing_support_and_nonfinal_longest_query(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("abcdefgh", encoding="utf-8")
    payload = _manifest_payload(_digest(prompt))
    payload["cases"][0]["queries"][0]["support_token_spans"] = []
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="needs support_token_spans"):
        _load_dataset_manifest(manifest)

    payload = _manifest_payload(_digest(prompt))
    payload["cases"][0]["token_count"] = 9
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="final query"):
        _load_dataset_manifest(manifest)


def test_manifest_snapshot_rejects_manifest_drift_and_duplicate_json_keys(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("abcdefgh", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_payload(_digest(prompt))), encoding="utf-8")
    snapshot = _load_dataset_manifest(manifest)

    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest bytes changed"):
        _verify_dataset_snapshot(snapshot)

    manifest.write_text(
        '{"schema_version":"a","schema_version":"b","status":"development-only","cases":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_dataset_manifest(manifest)


def test_local_model_identity_hashes_config_tokenizer_metadata_and_weights(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (model / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    weights = model / "model.safetensors"
    weights.write_bytes(b"fixed weights")

    first = _local_model_stack_identity(model)
    second = _local_model_stack_identity(model)
    names = {item["name"] for item in first["artifacts"][0]["files"]}

    assert first == second
    assert {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
    }.issubset(names)
    weights.write_bytes(b"drift weights")
    assert _local_model_stack_identity(model) != first


def test_capture_protocol_runs_one_independent_causal_prefix_per_checkpoint() -> None:
    seen: list[tuple[int, tuple[int, ...]]] = []

    def capture_fn(model, *, input_ids, layer_indices, query_positions):
        del model
        seen.append((int(input_ids.numel()), query_positions))
        return tuple(
            CapturedLayer(
                layer_index=layer,
                query_positions=query_positions,
                queries=torch.zeros(2, 1, 3),
                keys=torch.zeros(1, input_ids.numel(), 3),
                values=torch.zeros(1, input_ids.numel(), 3),
            )
            for layer in layer_indices
        )

    checkpoints = (
        QueryCheckpoint(token_count=4, support_token_spans=((0, 1),)),
        QueryCheckpoint(token_count=8, support_token_spans=((4, 5),)),
    )
    captures = list(
        _capture_independent_prefixes(
            object(),
            input_ids=torch.arange(8),
            layer_indices=(0, 1),
            checkpoints=checkpoints,
            capture_fn=capture_fn,
        )
    )

    assert seen == [(4, (3,)), (8, (7,))]
    assert [checkpoint.token_count for checkpoint, _, _ in captures] == [4, 8]
    assert all(len(layer_captures) == 2 for _, layer_captures, _ in captures)
    assert captures[0][1][0].keys.shape[1] == 4
    assert captures[1][1][0].keys.shape[1] == 8


def test_layer_evaluator_covers_every_query_head_and_builds_physical_unions() -> None:
    generator = torch.Generator().manual_seed(811)
    capture = CapturedLayer(
        layer_index=3,
        query_positions=(6,),
        queries=torch.randn(4, 1, 3, generator=generator),
        keys=torch.randn(2, 7, 3, generator=generator),
        values=torch.randn(2, 7, 3, generator=generator),
    )
    checkpoints = (QueryCheckpoint(token_count=7, support_token_spans=((4, 6),)),)

    budget, minimum, physical = _evaluate_captured_layer(
        capture,
        case_id="synthetic",
        checkpoints=checkpoints,
        page_sizes=(2,),
        candidate_fractions=(0.5,),
        sink_pages=1,
        recent_pages=1,
        top_k=2,
    )

    assert len(budget) == 4
    assert len(minimum) == 4 * 3
    assert len(physical) == 1
    assert {record["query_head"] for record in budget} == {0, 1, 2, 3}
    assert {record["kv_head"] for record in budget if record["query_head"] < 2} == {0}
    assert {record["kv_head"] for record in budget if record["query_head"] >= 2} == {1}
    assert all(len(record["kv_head_unions"]) == 2 for record in physical)
    assert all(
        address.startswith("3.")
        for record in physical
        for address in record["layer_union_atxy_addresses"]
    )
    partial = physical[0]
    assert partial["total_pages"] == 4
    assert partial["dense_layer_payload_bytes"] == 2 * 2 * 3 * 7 * 4
    assert partial["layer_union_payload_bytes"] <= partial["dense_layer_payload_bytes"]

    budget_summary = _summarize_budget_records(budget)
    minimum_summary = _summarize_minimum_mass_records(minimum)
    physical_summary = _summarize_physical_records(physical)
    assert len(budget_summary) == 1
    assert len(minimum_summary) == 3
    assert len(physical_summary) == 1
    assert budget_summary[0]["evaluations"] == 4
    assert physical_summary[0]["layer_query_evaluations"] == 1


def test_layer_evaluator_computes_dense_attention_once_per_query_head_checkpoint(
    monkeypatch,
) -> None:
    generator = torch.Generator().manual_seed(912)
    captures = tuple(
        (
            CapturedLayer(
                layer_index=1,
                query_positions=(token_count - 1,),
                queries=torch.randn(4, 1, 3, generator=generator),
                keys=torch.randn(2, token_count, 3, generator=generator),
                values=torch.randn(2, token_count, 3, generator=generator),
            ),
            QueryCheckpoint(
                token_count=token_count,
                support_token_spans=((token_count - 2, token_count - 1),),
            ),
        )
        for token_count in (4, 8)
    )
    original = oracle_lab.exact_attention_weights
    calls = 0

    def counted(query, keys):
        nonlocal calls
        calls += 1
        return original(query, keys)

    monkeypatch.setattr(oracle_lab, "exact_attention_weights", counted)

    budget = []
    for capture, checkpoint in captures:
        checkpoint_budget, _, _ = _evaluate_captured_layer(
            capture,
            case_id="attention-reuse",
            checkpoints=(checkpoint,),
            page_sizes=(1, 2, 4),
            candidate_fractions=(0.5,),
            sink_pages=1,
            recent_pages=1,
            top_k=2,
        )
        budget.extend(checkpoint_budget)

    assert calls == 2 * 4
    assert len(budget) == 2 * 4 * 3


def test_layer_evaluator_rejects_incomplete_gqa_and_query_positions() -> None:
    bad_heads = CapturedLayer(
        layer_index=0,
        query_positions=(3,),
        queries=torch.zeros(3, 1, 2),
        keys=torch.zeros(2, 4, 2),
        values=torch.zeros(2, 4, 2),
    )
    checkpoint = (QueryCheckpoint(token_count=4, support_token_spans=((0, 1),)),)

    with pytest.raises(ValueError, match="grouped-query"):
        _evaluate_captured_layer(
            bad_heads,
            case_id="bad",
            checkpoints=checkpoint,
            page_sizes=(2,),
            candidate_fractions=(0.5,),
            sink_pages=1,
            recent_pages=1,
            top_k=1,
        )

    wrong_position = CapturedLayer(
        layer_index=0,
        query_positions=(2,),
        queries=torch.zeros(2, 1, 2),
        keys=torch.zeros(1, 4, 2),
        values=torch.zeros(1, 4, 2),
    )
    with pytest.raises(ValueError, match="query positions"):
        _evaluate_captured_layer(
            wrong_position,
            case_id="bad",
            checkpoints=checkpoint,
            page_sizes=(2,),
            candidate_fractions=(0.5,),
            sink_pages=1,
            recent_pages=1,
            top_k=1,
        )


def test_development_report_seal_and_writer_are_exclusive(tmp_path) -> None:
    report = _seal_report({"schema_version": "test", "promotion_eligible": False})
    expected = dict(report)
    integrity = expected.pop("integrity")

    assert integrity["algorithm"] == "sha256"
    output = tmp_path / "report.json"
    _write_json_exclusive(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        _write_json_exclusive(output, report)


def test_tokenizer_requires_exact_manifest_length(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("short", encoding="utf-8")
    case = DevelopmentCase(
        case_id="exact-length",
        path=prompt,
        source_sha256=_digest(prompt),
        source_size_bytes=5,
        source_bytes=b"short",
        source_text="short",
        token_count=8,
        queries=(QueryCheckpoint(token_count=8, support_token_spans=((0, 1),)),),
    )

    with pytest.raises(ValueError, match="exactly 8"):
        _token_ids(CharacterTokenizer(), case)
