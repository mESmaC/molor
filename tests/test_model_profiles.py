from __future__ import annotations

import json
from pathlib import Path

from molor.model_profiles import (
    DEVELOPMENT_HUB_ID,
    DEVELOPMENT_LANE,
    PROMOTION_LANE,
    benchmark_output_root,
    architecture_contract,
    complete_local_model,
    corpus_expert_root,
    default_model,
    detect_model_lane,
    feedback_database,
    feedback_export_root,
    model_identities_match,
    router_feature_cache_root,
    router_checkpoint_matches_model,
    router_output_root,
    runtime_artifact_defaults,
)


def _write_minimal_model(directory: Path, *, include_weights: bool = True) -> None:
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
            }
        ),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    if include_weights:
        (directory / "model.safetensors").write_bytes(b"weights")


def _write_adapter(root: Path, relative: str) -> str:
    path = root / relative
    path.mkdir(parents=True)
    lane = "Qwen3-0.6B" if "qwen3-0.6b" in relative else "Qwen3-1.7B"
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_id": str(root / "models" / lane)}),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return str(path.resolve())


def _write_router(root: Path, relative: str) -> str:
    path = root / relative
    path.mkdir(parents=True)
    lane = "Qwen3-0.6B" if "qwen3-0.6b" in relative else "Qwen3-1.7B"
    (path / "router_config.json").write_text(
        json.dumps({"model": {"model_id": str(root / "models" / lane)}}),
        encoding="utf-8",
    )
    (path / "router.safetensors").write_bytes(b"weights")
    return str(path.resolve())


def test_lane_detection_accepts_hub_ids_and_windows_paths() -> None:
    assert detect_model_lane("Qwen/Qwen3-0.6B") == DEVELOPMENT_LANE
    assert detect_model_lane("C:\\models\\Qwen3-1.7B\\") == PROMOTION_LANE
    assert detect_model_lane("another/model") is None


def test_architecture_contracts_keep_router_widths_distinct_and_kv_geometry_equal() -> None:
    development = architecture_contract("Qwen/Qwen3-0.6B")
    promotion = architecture_contract("Qwen/Qwen3-1.7B")

    assert development is not None
    assert promotion is not None
    assert (development.hidden_size, promotion.hidden_size) == (1024, 2048)
    assert (
        development.layers,
        development.key_value_heads,
        development.head_dim,
    ) == (promotion.layers, promotion.key_value_heads, promotion.head_dim)


def test_model_identity_matches_local_aliases_but_not_a_hub_id(tmp_path: Path) -> None:
    model = tmp_path / "models" / "Qwen3-1.7B"
    model.mkdir(parents=True)

    assert model_identities_match(model, str(model.resolve()), tmp_path)
    assert model_identities_match("models/Qwen3-1.7B", model, tmp_path)
    assert not model_identities_match("Qwen/Qwen3-1.7B", model, tmp_path)


def test_default_model_never_selects_an_incomplete_local_snapshot(tmp_path: Path) -> None:
    _write_minimal_model(tmp_path / "models" / "Qwen3-0.6B", include_weights=False)

    assert not complete_local_model(tmp_path / "models" / "Qwen3-0.6B")
    assert default_model(tmp_path, {}) == DEVELOPMENT_HUB_ID


def test_default_model_prefers_complete_local_06_and_honors_override(tmp_path: Path) -> None:
    model = tmp_path / "models" / "Qwen3-0.6B"
    _write_minimal_model(model)

    assert complete_local_model(model)
    assert default_model(tmp_path, {}) == str(model.resolve())
    assert default_model(tmp_path, {"MOLOR_MODEL_ID": "custom/base"}) == "custom/base"


def test_default_model_rejects_wrong_architecture_under_the_06_directory(tmp_path: Path) -> None:
    model = tmp_path / "models" / "Qwen3-0.6B"
    _write_minimal_model(model)
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hidden_size"] = 2048
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert not complete_local_model(model)
    assert default_model(tmp_path, {}) == DEVELOPMENT_HUB_ID


def test_06_base_never_inherits_existing_17_artifacts(tmp_path: Path) -> None:
    _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
    )
    _write_router(tmp_path, "checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled")

    assert runtime_artifact_defaults("Qwen/Qwen3-0.6B", tmp_path).expert_a == ""
    assert runtime_artifact_defaults("Qwen/Qwen3-0.6B", tmp_path).expert_paths == ()
    assert runtime_artifact_defaults("Qwen/Qwen3-0.6B", tmp_path).router == ""


def test_incomplete_06_artifacts_are_not_auto_selected(tmp_path: Path) -> None:
    adapter = tmp_path / "checkpoints" / "qwen3-0.6b-corpus-experts" / "embedded-knowledge-v1"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    router = tmp_path / "checkpoints" / "qwen3-0.6b-corpus-router-v1"
    router.mkdir(parents=True)
    (router / "router_config.json").write_text("{}", encoding="utf-8")

    defaults = runtime_artifact_defaults("Qwen/Qwen3-0.6B", tmp_path)

    assert defaults.expert_a == ""
    assert defaults.router == ""


def test_wrong_base_artifacts_inside_06_roots_are_not_auto_selected(tmp_path: Path) -> None:
    adapter = _write_adapter(
        tmp_path,
        "checkpoints/qwen3-0.6b-corpus-experts/embedded-knowledge-v1",
    )
    router = _write_router(tmp_path, "checkpoints/qwen3-0.6b-corpus-router-v1")
    (Path(adapter) / "adapter_config.json").write_text(
        json.dumps({"base_model_id": "Qwen/Qwen3-1.7B"}),
        encoding="utf-8",
    )
    (Path(router) / "router_config.json").write_text(
        json.dumps({"model": {"model_id": "Qwen/Qwen3-1.7B"}}),
        encoding="utf-8",
    )

    defaults = runtime_artifact_defaults("Qwen/Qwen3-0.6B", tmp_path)

    assert defaults.expert_a == ""
    assert defaults.router == ""


def test_stable_17_profile_preserves_the_promoted_local_rack(tmp_path: Path) -> None:
    expert_a = _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
    )
    expert_b = _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/ocb-can-verified-v1",
    )
    architecture = _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/software-architecture-v1",
    )
    verified_code = _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/verified-embedded-code-v1",
    )
    router = _write_router(tmp_path, "checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled")

    defaults = runtime_artifact_defaults(tmp_path / "models" / "Qwen3-1.7B", tmp_path)

    assert defaults.expert_a == expert_a
    assert defaults.expert_b == expert_b
    assert defaults.expert_paths == (architecture, verified_code)
    assert defaults.router == router
    assert router_checkpoint_matches_model(
        router,
        tmp_path / "models" / "Qwen3-1.7B",
        tmp_path,
    )


def test_hub_identity_does_not_auto_select_local_17_artifacts(tmp_path: Path) -> None:
    _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
    )
    _write_router(tmp_path, "checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled")

    defaults = runtime_artifact_defaults("Qwen/Qwen3-1.7B", tmp_path)

    assert defaults.expert_a == ""
    assert defaults.router == ""


def test_development_and_promotion_namespaces_are_disjoint(tmp_path: Path) -> None:
    development = "Qwen/Qwen3-0.6B"
    promotion = "Qwen/Qwen3-1.7B"

    assert corpus_expert_root(development, tmp_path) != corpus_expert_root(promotion, tmp_path)
    assert router_output_root(development, tmp_path) != router_output_root(promotion, tmp_path)
    assert router_feature_cache_root(development, tmp_path) != router_feature_cache_root(
        promotion, tmp_path
    )
    assert feedback_database(development, tmp_path) != feedback_database(promotion, tmp_path)
    assert feedback_export_root(development, tmp_path) != feedback_export_root(promotion, tmp_path)
    assert benchmark_output_root(development, tmp_path) != benchmark_output_root(
        promotion, tmp_path
    )
