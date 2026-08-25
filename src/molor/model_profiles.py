"""Model-lane defaults that keep incompatible adapters and routers separated."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


DEVELOPMENT_LANE = "qwen3-0.6b"
PROMOTION_LANE = "qwen3-1.7b"
DEVELOPMENT_HUB_ID = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True, slots=True)
class RuntimeArtifactDefaults:
    """Compatible local runtime artifacts for one selected base model."""

    expert_a: str = ""
    expert_b: str = ""
    expert_paths: tuple[str, ...] = ()
    router: str = ""


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """The dimensions that make adapters and router features lane-specific."""

    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    key_value_heads: int
    head_dim: int
    context_length: int


_ARCHITECTURES = {
    DEVELOPMENT_LANE: ModelArchitecture(1024, 3072, 28, 16, 8, 128, 40960),
    PROMOTION_LANE: ModelArchitecture(2048, 6144, 28, 16, 8, 128, 40960),
}


def detect_model_lane(model: str | Path) -> str | None:
    """Return a known architecture lane from a hub ID or local model path."""

    normalized = str(model).strip().replace("\\", "/").rstrip("/").casefold()
    name = normalized.rsplit("/", 1)[-1]
    if name == DEVELOPMENT_LANE:
        return DEVELOPMENT_LANE
    if name == PROMOTION_LANE:
        return PROMOTION_LANE
    return None


def complete_local_model(directory: Path) -> bool:
    """Check that a local Transformers directory includes all declared weight files."""

    if not (
        _nonempty_file(directory / "config.json")
        and _nonempty_file(directory / "tokenizer_config.json")
        and _nonempty_file(directory / "tokenizer.json")
    ):
        return False
    lane = detect_model_lane(directory)
    contract = _ARCHITECTURES.get(lane)
    if contract is not None and not _config_matches_contract(directory / "config.json", contract):
        return False
    if _nonempty_file(directory / "model.safetensors"):
        return True
    if _nonempty_file(directory / "pytorch_model.bin"):
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = directory / index_name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            continue
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        raw_filenames = tuple(weight_map.values())
        if not all(isinstance(value, str) and value.strip() for value in raw_filenames):
            continue
        filenames = set(raw_filenames)
        if all(_nonempty_file(directory / filename) for filename in filenames):
            return True
    return False


def architecture_contract(model: str | Path) -> ModelArchitecture | None:
    """Return the frozen dimensions for a known model lane."""

    return _ARCHITECTURES.get(detect_model_lane(model))


def default_model(
    workspace_root: Path,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Choose the 0.6B development base, preferring a complete local snapshot."""

    if environment:
        override = environment.get("MOLOR_MODEL_ID", "").strip()
        if override:
            return override
    local = workspace_root / "models" / "Qwen3-0.6B"
    return str(local.resolve()) if complete_local_model(local) else DEVELOPMENT_HUB_ID


def runtime_artifact_defaults(
    model: str | Path,
    workspace_root: Path,
) -> RuntimeArtifactDefaults:
    """Resolve only artifacts trained for the selected model lane."""

    lane = detect_model_lane(model)
    if lane == DEVELOPMENT_LANE:
        expert_root = Path("checkpoints/qwen3-0.6b-corpus-experts")
        expert_a_candidates = (
            expert_root / "embedded-knowledge-v1",
            expert_root / "software-architecture-v1",
        )
        expert_b_candidates = (
            expert_root / "ocb-can-verified-v1",
            expert_root / "verified-embedded-code-v1",
        )
        additional_candidates = (
            expert_root / "software-architecture-v1",
            expert_root / "verified-embedded-code-v1",
        )
        router_candidates = (
            Path("checkpoints/qwen3-0.6b-corpus-router-v4-mlp-controlled"),
            Path("checkpoints/qwen3-0.6b-corpus-router-v3-mlp"),
            Path("checkpoints/qwen3-0.6b-corpus-router-v1"),
        )
    elif lane == PROMOTION_LANE:
        expert_a_candidates = tuple(
            Path(candidate)
            for candidate in (
                "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
                "checkpoints/qwen3-1.7b-corpus-experts/software-architecture-v1",
                "checkpoints/qwen3-1.7b-real-experts/financial-analysis-verified",
                "checkpoints/qwen3-1.7b-real-experts/financial-analysis",
                "checkpoints/precision",
                "checkpoints/qwen3-1.7b-two-styles/precision",
            )
        )
        expert_b_candidates = tuple(
            Path(candidate)
            for candidate in (
                "checkpoints/qwen3-1.7b-corpus-experts/ocb-can-verified-v1",
                "checkpoints/qwen3-1.7b-real-experts/local-embedded-code",
                "checkpoints/qwen3-1.7b-real-experts/terminal-cli",
                "checkpoints/creative",
                "checkpoints/qwen3-1.7b-two-styles/creative",
            )
        )
        additional_candidates = tuple(
            Path(candidate)
            for candidate in (
                "checkpoints/qwen3-1.7b-corpus-experts/software-architecture-v1",
                "checkpoints/qwen3-1.7b-corpus-experts/verified-embedded-code-v1",
            )
        )
        router_candidates = tuple(
            Path(candidate)
            for candidate in (
                "checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled",
                "checkpoints/qwen3-1.7b-corpus-router-v3-mlp",
                "checkpoints/qwen3-1.7b-corpus-router-v1",
            )
        )
    else:
        return RuntimeArtifactDefaults()

    expert_a = _first_existing_adapter(workspace_root, expert_a_candidates, lane, model)
    expert_b = _first_existing_adapter(workspace_root, expert_b_candidates, lane, model)
    occupied = {path for path in (expert_a, expert_b) if path}
    expert_paths = tuple(
        path
        for candidate in additional_candidates
        if (path := _existing_adapter(workspace_root, candidate, lane, model))
        and path not in occupied
    )
    router = _first_existing_router(workspace_root, router_candidates, lane, model)
    return RuntimeArtifactDefaults(expert_a, expert_b, expert_paths, router)


def corpus_expert_root(model: str | Path, workspace_root: Path) -> Path | None:
    """Return the lane-specific default expert root for router training."""

    lane = detect_model_lane(model)
    if lane is None:
        return None
    return workspace_root / "checkpoints" / f"{lane}-corpus-experts"


def router_output_root(model: str | Path, workspace_root: Path) -> Path | None:
    """Return the lane-specific v1 router output directory."""

    lane = detect_model_lane(model)
    if lane is None:
        return None
    return workspace_root / "checkpoints" / f"{lane}-corpus-router-v1"


def router_feature_cache_root(model: str | Path, workspace_root: Path) -> Path | None:
    """Keep pooled router features isolated because embedding widths differ by lane."""

    lane = detect_model_lane(model)
    if lane is None:
        return None
    if lane == PROMOTION_LANE:
        return workspace_root / "runs" / "router-features" / "bootstrap-v1"
    return workspace_root / "runs" / "qwen3-0.6b-development" / "router-features" / "bootstrap-v1"


def feedback_database(model: str | Path, workspace_root: Path) -> Path:
    """Keep development annotations out of the established 1.7B feedback notebook."""

    if detect_model_lane(model) == DEVELOPMENT_LANE:
        return workspace_root / "runs" / "qwen3-0.6b-development" / "feedback" / "feedback.sqlite3"
    return workspace_root / "runs" / "feedback" / "feedback.sqlite3"


def feedback_export_root(model: str | Path, workspace_root: Path) -> Path:
    """Return the model-lane-specific feedback export directory."""

    return feedback_database(model, workspace_root).parent / "exports"


def benchmark_output_root(model: str | Path, workspace_root: Path) -> Path:
    """Separate new 0.6B measurements from the established benchmark history."""

    root = workspace_root / "runs" / "benchmarks"
    if detect_model_lane(model) == DEVELOPMENT_LANE:
        return root / "qwen3-0.6b-development"
    return root


def model_identities_match(
    left: str | Path,
    right: str | Path,
    workspace_root: Path,
) -> bool:
    """Compare local paths canonically while keeping hub IDs distinct from local snapshots."""

    return _model_identity(left, workspace_root) == _model_identity(right, workspace_root)


def router_checkpoint_matches_model(
    directory: str | Path,
    model: str | Path,
    workspace_root: Path,
) -> bool:
    """Check router completeness and its exact frozen-base identity."""

    path = Path(directory).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    if not (
        _nonempty_file(path / "router_config.json") and _nonempty_file(path / "router.safetensors")
    ):
        return False
    model_id = _router_model_id(path / "router_config.json")
    return model_id is not None and model_identities_match(model_id, model, workspace_root)


def _first_existing_adapter(
    workspace_root: Path,
    candidates: tuple[Path, ...],
    expected_lane: str,
    selected_model: str | Path,
) -> str:
    return next(
        (
            path
            for candidate in candidates
            if (path := _existing_adapter(workspace_root, candidate, expected_lane, selected_model))
        ),
        "",
    )


def _first_existing_router(
    workspace_root: Path,
    candidates: tuple[Path, ...],
    expected_lane: str,
    selected_model: str | Path,
) -> str:
    return next(
        (
            path
            for candidate in candidates
            if (path := _existing_router(workspace_root, candidate, expected_lane, selected_model))
        ),
        "",
    )


def _existing_adapter(
    workspace_root: Path,
    candidate: Path,
    expected_lane: str,
    selected_model: str | Path,
) -> str:
    path = candidate if candidate.is_absolute() else workspace_root / candidate
    if not (
        _nonempty_file(path / "adapter_config.json")
        and _nonempty_file(path / "adapter_model.safetensors")
    ):
        return ""
    payload = _read_json_object(path / "adapter_config.json")
    model_id = payload.get("base_model_id") or payload.get("base_model_name_or_path")
    if (
        not isinstance(model_id, str)
        or detect_model_lane(model_id) != expected_lane
        or not model_identities_match(model_id, selected_model, workspace_root)
    ):
        return ""
    return str(path.resolve())


def _existing_router(
    workspace_root: Path,
    candidate: Path,
    expected_lane: str,
    selected_model: str | Path,
) -> str:
    path = candidate if candidate.is_absolute() else workspace_root / candidate
    if not (
        _nonempty_file(path / "router_config.json") and _nonempty_file(path / "router.safetensors")
    ):
        return ""
    model_id = _router_model_id(path / "router_config.json")
    if (
        model_id is None
        or detect_model_lane(model_id) != expected_lane
        or not model_identities_match(model_id, selected_model, workspace_root)
    ):
        return ""
    return str(path.resolve())


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _router_model_id(path: Path) -> str | None:
    payload = _read_json_object(path)
    model_payload = payload.get("model")
    model_id = (
        model_payload.get("model_id")
        if isinstance(model_payload, dict)
        else payload.get("model_id")
    )
    return model_id.strip() if isinstance(model_id, str) and model_id.strip() else None


def _model_identity(model: str | Path, workspace_root: Path) -> str:
    raw = str(model).strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    if candidate.exists():
        return str(candidate.resolve()).casefold()
    return raw.replace("\\", "/").rstrip("/").casefold()


def _config_matches_contract(path: Path, contract: ModelArchitecture) -> bool:
    payload = _read_json_object(path)
    expected = {
        "hidden_size": contract.hidden_size,
        "intermediate_size": contract.intermediate_size,
        "num_hidden_layers": contract.layers,
        "num_attention_heads": contract.attention_heads,
        "num_key_value_heads": contract.key_value_heads,
        "head_dim": contract.head_dim,
        "max_position_embeddings": contract.context_length,
    }
    return payload.get("model_type") == "qwen3" and all(
        payload.get(name) == value for name, value in expected.items()
    )


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False
