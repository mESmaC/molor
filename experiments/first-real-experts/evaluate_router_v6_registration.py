"""Sealed one-shot registration evaluation for the six-route Taskmaster router.

This evaluator deliberately separates development-only checkpoint selection from the
blind probe.  It validates all frozen training inputs, all three candidate sidecars,
and the selected checkpoint before it hashes or parses the probe.  The probe is then
opened once, under the runtime policy frozen in ``router-v6-evaluation-policy.json``.

Passing this evaluator never mutates router or expert registration.  It only writes an
exclusive, append-only evidence artifact that can authorize a separate explicit change.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import torch
from torch import Tensor
from torch.nn import functional as F

from molor.experimental.autorouter import NULL_ROUTE_ID, ROUTER_CHECKPOINT_SCHEMA
from molor.training.router import (
    RouterExample,
    RouterFeatureContract,
    extract_prompt_features,
    load_input_embedding,
    load_router_jsonl,
    load_router_tokenizer,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = EXPERIMENT_ROOT / "router-v6-evaluation-policy.json"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "Qwen3-1.7B"
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "runs" / "evaluation" / "router-v6-taskmaster-registration-one-shot.json"
)
DEFAULT_PREBLIND_RECEIPT = (
    WORKSPACE_ROOT / "runs" / "router-v6" / "taskmaster-preblind-receipt.json"
)

POLICY_SCHEMA = "molor.router-v6-evaluation-policy.v1"
REPORT_SCHEMA = "molor.router-v6-registration-report.v1"
PREBLIND_SCHEMA = "molor.router-v6-preblind-receipt.v1"
PROBE_PURPOSE = "blind_intent_probe_never_training"
TASKMASTER_ROUTE_ID = "taskmaster-coffee-api-v2-full-dialog-r16"
EXPECTED_SEEDS = (83, 8419, 27183)
EXPECTED_RUNTIME = {
    "temperature": 1.0,
    "confidence_threshold": 0.30,
    "null_bias": 0.0,
    "top_k": 1,
}
EXPECTED_ONE_SHOT_GATES = {
    "overall_exact_accuracy_min": 0.75,
    "overall_acceptable_accuracy_min": 0.75,
    "exact_correct_gain_over_baseline_min": 8,
    "acceptable_correct_gain_over_baseline_min": 8,
    "taskmaster_exact_correct_min": 10,
    "taskmaster_row_count": 12,
    "false_taskmaster_selections_on_non_taskmaster_rows_max": 1,
    "non_taskmaster_row_count": 72,
    "max_exact_correct_drop_by_protected_category": 1,
    "max_acceptable_correct_drop_by_protected_category": 1,
    "protected_categories": [
        "base/null",
        "embedded-knowledge-v1",
        "ocb-can-verified-v1",
        "software-architecture-v1",
        "verified-embedded-code-v1",
        "mixed",
    ],
    "complete_rows_required": 84,
    "finite_scores_required": True,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
RUNTIME_SCRIPT_PATHS = {
    "evaluator": "experiments/first-real-experts/evaluate_router_v6_registration.py",
    "shared_probe_evaluator": "experiments/first-real-experts/evaluate_router.py",
    "dataset_builder": "experiments/first-real-experts/build_router_v6_taskmaster_dataset.py",
}
SOURCE_TREE_HASH_POLICY = "sha256-canonical-repository-relative-path-size-content-v1"
MODEL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
EXPECTED_PREBLIND_HASH_BINDINGS = [
    "policy",
    "runtime_source_scripts",
    "molor_source_tree",
    "model_config",
    "model_safetensors_index",
    "model_embedding_shard_and_tensor_metadata",
    "model_tokenizer_artifacts",
    "dataset_train",
    "dataset_validation",
    "dataset_test",
    "dataset_report",
    "baseline_router_config",
    "baseline_router_weights",
    "expert_adapter_configs",
    "expert_adapter_weights",
    "all_candidate_router_configs",
    "all_candidate_router_weights",
    "selected_candidate_router_config",
    "selected_candidate_router_weights",
]
EXPECTED_PREBLIND_VALIDATION_RECORDS = [
    "fresh_validation_loss_by_seed",
    "fresh_validation_accuracy_by_seed",
    "fresh_taskmaster_validation_precision_by_seed",
    "fresh_taskmaster_validation_recall_by_seed",
    "fresh_legacy_route_validation_recall_by_seed",
    "sidecar_metric_diagnostics_by_seed",
    "selected_seed",
]


def _load_router_probe_module():
    """Load the existing probe implementation without making this directory a package."""

    module_name = "molor_router_v6_shared_probe_evaluator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = EXPERIMENT_ROOT / "evaluate_router.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery boundary
        raise RuntimeError(f"cannot load shared router evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROUTER_PROBE = _load_router_probe_module()


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    path: Path
    sha256: str
    size_bytes: int

    def to_dict(self, *, repository: Path = WORKSPACE_ROOT) -> dict[str, Any]:
        try:
            display_path = self.path.relative_to(repository).as_posix()
        except ValueError:
            display_path = str(self.path)
        return {
            "path": display_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    seed: int
    checkpoint: Path
    config: ArtifactSnapshot
    weights: ArtifactSnapshot
    validation_loss: float
    validation_accuracy: float
    taskmaster_precision: float
    taskmaster_recall: float
    legacy_recalls: Mapping[str, float]
    passed: bool
    gates: tuple[Mapping[str, Any], ...]
    sidecar_metric_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, repository: Path) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "checkpoint": self.checkpoint.relative_to(repository).as_posix(),
            "config": self.config.to_dict(repository=repository),
            "weights": self.weights.to_dict(repository=repository),
            "validation_metric_source": "fresh_cpu_recomputation_from_frozen_validation_split",
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "taskmaster_precision": self.taskmaster_precision,
            "taskmaster_recall": self.taskmaster_recall,
            "legacy_recalls": dict(sorted(self.legacy_recalls.items())),
            "gates": list(self.gates),
            "sidecar_metric_diagnostics": dict(self.sidecar_metric_diagnostics),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ModelRuntimeEvidence:
    directory: Path
    config: ArtifactSnapshot
    index: ArtifactSnapshot
    embedding_artifact: ArtifactSnapshot
    embedding_tensor_name: str
    embedding_shape: tuple[int, int]
    embedding_dtype: str
    tokenizer_artifacts: Mapping[str, ArtifactSnapshot]

    @property
    def snapshots(self) -> dict[str, ArtifactSnapshot]:
        return {
            "model_config": self.config,
            "model_index": self.index,
            "model_embedding_shard": self.embedding_artifact,
            **{
                f"model_tokenizer:{name}": value for name, value in self.tokenizer_artifacts.items()
            },
        }

    def to_dict(self, *, repository: Path) -> dict[str, Any]:
        return {
            "path": self.directory.relative_to(repository).as_posix(),
            "config": self.config.to_dict(repository=repository),
            "index": self.index.to_dict(repository=repository),
            "embedding": {
                "tensor_name": self.embedding_tensor_name,
                "shape": list(self.embedding_shape),
                "dtype": self.embedding_dtype,
                "artifact": self.embedding_artifact.to_dict(repository=repository),
            },
            "tokenizer_artifacts": {
                name: value.to_dict(repository=repository)
                for name, value in sorted(self.tokenizer_artifacts.items())
            },
        }


class PreblindMetricGateFailure(ValueError):
    """All candidate artifacts were complete, but at least one frozen metric gate failed."""

    def __init__(
        self,
        message: str,
        *,
        candidates: Sequence[CandidateEvidence],
        snapshots: Mapping[str, ArtifactSnapshot],
    ) -> None:
        super().__init__(message)
        self.candidates = tuple(candidates)
        self.snapshots = dict(snapshots)


def select_candidate(candidates: Sequence[CandidateEvidence]) -> CandidateEvidence:
    """Select by validation loss with the preregistered numeric-seed tie-break."""

    if {candidate.seed for candidate in candidates} != set(EXPECTED_SEEDS):
        raise ValueError("candidate seed set differs from the frozen policy")
    if len(candidates) != len(EXPECTED_SEEDS) or not all(
        candidate.passed for candidate in candidates
    ):
        raise ValueError("all three candidate seeds must pass before selection")
    return min(candidates, key=lambda candidate: (candidate.validation_loss, candidate.seed))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot(path: Path) -> ArtifactSnapshot:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return ArtifactSnapshot(
        path=resolved,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def assert_unchanged(value: ArtifactSnapshot, *, label: str) -> None:
    if not value.path.is_file():
        raise OSError(f"{label} disappeared during evaluation: {value.path}")
    if value.path.stat().st_size != value.size_bytes or sha256_file(value.path) != value.sha256:
        raise OSError(f"{label} changed during evaluation: {value.path}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _repository_path(repository: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    result = (repository / Path(raw_path)).resolve()
    try:
        result.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository: {raw_path!r}") from exc
    return result


def _descriptor_path(
    descriptor: Mapping[str, Any],
    *,
    repository: Path,
    label: str,
    expected_path: Path | None = None,
) -> ArtifactSnapshot:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} descriptor must be an object")
    allowed = {"path", "sha256", "size_bytes", "rows", "logical_name", "purpose"}
    unknown = set(descriptor).difference(allowed)
    if unknown:
        raise ValueError(f"{label} descriptor has unknown fields: {sorted(unknown)!r}")
    path = _repository_path(repository, descriptor.get("path"), label=f"{label} path")
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} path differs from the frozen path")
    digest = descriptor.get("sha256")
    if _SHA256.fullmatch(str(digest)) is None:
        raise ValueError(f"{label} sha256 is invalid")
    actual = snapshot(path)
    if actual.sha256 != digest:
        raise ValueError(f"{label} sha256 differs from frozen policy")
    size = descriptor.get("size_bytes")
    if size is not None and (
        not isinstance(size, int) or isinstance(size, bool) or size != actual.size_bytes
    ):
        raise ValueError(f"{label} size differs from frozen policy")
    return actual


def _reported_artifact(
    descriptor: Mapping[str, Any],
    *,
    repository: Path,
    label: str,
    expected_path: Path | None = None,
    count_rows: bool = False,
) -> ArtifactSnapshot:
    """Validate an artifact descriptor emitted by a trusted in-repository builder."""

    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} descriptor must be an object")
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    candidate = Path(raw_path)
    path = candidate.resolve() if candidate.is_absolute() else (repository / candidate).resolve()
    try:
        path.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the repository") from exc
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} path differs from the frozen location")
    digest = descriptor.get("sha256")
    if _SHA256.fullmatch(str(digest)) is None:
        raise ValueError(f"{label} sha256 is invalid")
    result = snapshot(path)
    if result.sha256 != digest:
        raise ValueError(f"{label} bytes differ from the builder report")
    rows = descriptor.get("rows")
    if count_rows:
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise ValueError(f"{label} row count is invalid")
        with path.open("rb") as stream:
            actual_rows = sum(bool(line.strip()) for line in stream)
        if actual_rows != rows:
            raise ValueError(f"{label} row count differs from the builder report")
    return result


def validate_runtime_sources(
    policy: Mapping[str, Any], *, repository: Path = WORKSPACE_ROOT
) -> dict[str, ArtifactSnapshot]:
    """Validate every local source file that can change router-evaluation semantics."""

    contract = policy.get("runtime_source_artifacts")
    if not isinstance(contract, Mapping) or set(contract) != {"scripts", "molor_source_tree"}:
        raise ValueError("runtime_source_artifacts must bind scripts and the full molor tree")
    descriptors = contract["scripts"]
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(RUNTIME_SCRIPT_PATHS):
        raise ValueError("runtime source script descriptor set is incomplete")
    results: dict[str, ArtifactSnapshot] = {}
    for name, relative_path in RUNTIME_SCRIPT_PATHS.items():
        expected = (repository / relative_path).resolve()
        results[f"script:{name}"] = _descriptor_path(
            descriptors[name],
            repository=repository,
            label=f"runtime source {name}",
            expected_path=expected,
        )

    tree_contract = contract["molor_source_tree"]
    expected_tree_fields = {
        "root",
        "glob",
        "hash_policy",
        "file_count",
        "content_fingerprint",
    }
    if not isinstance(tree_contract, Mapping) or set(tree_contract) != expected_tree_fields:
        raise ValueError("molor source-tree contract is incomplete")
    if (
        tree_contract.get("root") != "src/molor"
        or tree_contract.get("glob") != "**/*.py"
        or tree_contract.get("hash_policy") != SOURCE_TREE_HASH_POLICY
        or _SHA256.fullmatch(str(tree_contract.get("content_fingerprint"))) is None
    ):
        raise ValueError("molor source-tree contract is invalid")
    source_root = (repository / "src" / "molor").resolve()
    files = sorted(
        (path.resolve() for path in source_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(repository).as_posix(),
    )
    if len(files) != tree_contract.get("file_count"):
        raise ValueError("molor source-tree file count differs from frozen policy")
    entries: list[dict[str, Any]] = []
    for path in files:
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError as exc:  # pragma: no cover - defensive symlink boundary
            raise ValueError("molor source tree contains a file outside the repository") from exc
        value = snapshot(path)
        results[f"molor:{relative}"] = value
        entries.append({"path": relative, "size_bytes": value.size_bytes, "sha256": value.sha256})
    fingerprint = sha256_text(canonical_json(entries))
    if fingerprint != tree_contract["content_fingerprint"]:
        raise ValueError("molor source-tree content fingerprint differs from frozen policy")
    return results


def validate_model_runtime(
    policy: Mapping[str, Any],
    *,
    model_path: Path,
    repository: Path = WORKSPACE_ROOT,
) -> ModelRuntimeEvidence:
    """Hash-lock the model config, index, embedding shard, and tokenizer bytes."""

    contract = policy.get("model_runtime")
    if not isinstance(contract, Mapping) or set(contract) != {
        "path",
        "config",
        "safetensors_index",
        "embedding",
        "tokenizer_artifacts",
    }:
        raise ValueError("model_runtime contract is incomplete")
    frozen_model = _repository_path(repository, contract.get("path"), label="model runtime path")
    expected_model = (repository / "models" / "Qwen3-1.7B").resolve()
    if frozen_model != expected_model or model_path.expanduser().resolve() != frozen_model:
        raise ValueError("--model must equal the frozen repository-local models/Qwen3-1.7B path")
    config = _descriptor_path(
        contract["config"],
        repository=repository,
        label="model config",
        expected_path=frozen_model / "config.json",
    )
    index = _descriptor_path(
        contract["safetensors_index"],
        repository=repository,
        label="model safetensors index",
        expected_path=frozen_model / "model.safetensors.index.json",
    )
    embedding = contract["embedding"]
    if not isinstance(embedding, Mapping) or set(embedding) != {
        "tensor_name",
        "shape",
        "dtype",
        "artifact",
    }:
        raise ValueError("model embedding contract is incomplete")
    tensor_name = embedding.get("tensor_name")
    shape = embedding.get("shape")
    dtype = embedding.get("dtype")
    if tensor_name != "model.embed_tokens.weight" or shape != [151936, 2048] or dtype != "bfloat16":
        raise ValueError("model embedding tensor metadata differs from the frozen Qwen contract")
    embedding_artifact = _descriptor_path(
        embedding["artifact"],
        repository=repository,
        label="model embedding shard",
        expected_path=frozen_model / "model-00001-of-00002.safetensors",
    )
    tokenizer_contract = contract["tokenizer_artifacts"]
    if not isinstance(tokenizer_contract, Mapping) or set(tokenizer_contract) != set(
        MODEL_TOKENIZER_FILES
    ):
        raise ValueError("model tokenizer artifact set differs from the frozen sidecars")
    tokenizer_artifacts = {
        name: _descriptor_path(
            tokenizer_contract[name],
            repository=repository,
            label=f"model tokenizer {name}",
            expected_path=frozen_model / name,
        )
        for name in MODEL_TOKENIZER_FILES
    }
    return ModelRuntimeEvidence(
        directory=frozen_model,
        config=config,
        index=index,
        embedding_artifact=embedding_artifact,
        embedding_tensor_name=tensor_name,
        embedding_shape=(151936, 2048),
        embedding_dtype=dtype,
        tokenizer_artifacts=tokenizer_artifacts,
    )


def validate_model_sidecar(
    sidecar: Mapping[str, Any],
    *,
    evidence: ModelRuntimeEvidence,
    label: str,
) -> None:
    """Require a router sidecar to name the exact independently hashed model artifacts."""

    model = sidecar.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"{label} has no model identity")
    if Path(str(model.get("model_id"))).resolve() != evidence.directory:
        raise ValueError(f"{label} model path differs from frozen runtime")

    def require_artifact(
        descriptor: Any, expected: ArtifactSnapshot, *, artifact_label: str
    ) -> None:
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"{label} {artifact_label} descriptor is missing")
        if (
            Path(str(descriptor.get("path"))).resolve() != expected.path
            or descriptor.get("sha256") != expected.sha256
            or descriptor.get("size_bytes") != expected.size_bytes
        ):
            raise ValueError(f"{label} {artifact_label} differs from frozen runtime")

    require_artifact(model.get("config"), evidence.config, artifact_label="model config")
    embedding = model.get("embedding")
    if not isinstance(embedding, Mapping):
        raise ValueError(f"{label} embedding identity is missing")
    if (
        embedding.get("tensor_name") != evidence.embedding_tensor_name
        or embedding.get("shape") != list(evidence.embedding_shape)
        or embedding.get("dtype") != evidence.embedding_dtype
    ):
        raise ValueError(f"{label} embedding tensor metadata differs from frozen runtime")
    require_artifact(
        embedding.get("artifact"),
        evidence.embedding_artifact,
        artifact_label="embedding shard",
    )
    tokenizer_values = model.get("tokenizer_artifacts")
    if not isinstance(tokenizer_values, list) or len(tokenizer_values) != len(
        evidence.tokenizer_artifacts
    ):
        raise ValueError(f"{label} tokenizer artifact list is incomplete")
    by_name = {
        descriptor.get("logical_name"): descriptor
        for descriptor in tokenizer_values
        if isinstance(descriptor, Mapping)
    }
    if set(by_name) != set(evidence.tokenizer_artifacts):
        raise ValueError(f"{label} tokenizer artifact names differ from frozen runtime")
    for name, expected in evidence.tokenizer_artifacts.items():
        require_artifact(by_name[name], expected, artifact_label=f"tokenizer {name}")


def validate_loaded_embedding(
    metadata: Mapping[str, Any], *, evidence: ModelRuntimeEvidence
) -> None:
    if (
        metadata.get("tensor_name") != evidence.embedding_tensor_name
        or metadata.get("shape") != list(evidence.embedding_shape)
        or metadata.get("dtype") != evidence.embedding_dtype
    ):
        raise ValueError("loaded embedding tensor metadata differs from frozen model runtime")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping) or (
        Path(str(artifact.get("path"))).resolve() != evidence.embedding_artifact.path
        or artifact.get("sha256") != evidence.embedding_artifact.sha256
        or artifact.get("size_bytes") != evidence.embedding_artifact.size_bytes
    ):
        raise ValueError("loaded embedding shard differs from frozen model runtime")


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _gate(name: str, *, actual: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {"name": name, "actual": actual, "required": required, "passed": bool(passed)}


def _require_exact_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} fields differ from the frozen contract")
    for name, required in expected.items():
        value = actual[name]
        if isinstance(required, float):
            if _number(value, label=f"{label}.{name}") != required:
                raise ValueError(f"{label}.{name} differs from the frozen contract")
        elif value != required:
            raise ValueError(f"{label}.{name} differs from the frozen contract")


def load_policy(
    path: Path, *, repository: Path = WORKSPACE_ROOT
) -> tuple[dict[str, Any], ArtifactSnapshot]:
    """Load the frozen policy without touching the blind-probe artifact."""

    policy_snapshot = snapshot(path)
    policy = _read_json_object(policy_snapshot.path, label="router v6 policy")
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported router v6 registration policy schema")
    if policy.get("status") != "frozen_before_candidate_training_and_blind_evaluation":
        raise ValueError("router v6 policy is not frozen")

    route_ids = policy.get("route_ids")
    if (
        not isinstance(route_ids, list)
        or any(not isinstance(route_id, str) or not route_id for route_id in route_ids)
        or len(route_ids) != 6
        or len(route_ids) != len(set(route_ids))
        or route_ids[-1] != NULL_ROUTE_ID
        or TASKMASTER_ROUTE_ID not in route_ids
    ):
        raise ValueError("router v6 policy must freeze six unique routes with base/null last")

    training = policy.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("router v6 policy has no training contract")
    seeds = training.get("seeds")
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"router v6 seeds must be {list(EXPECTED_SEEDS)!r}")
    hidden_size = training.get("router_hidden_size", training.get("hidden_size"))
    if hidden_size != 256:
        raise ValueError("router v6 policy must freeze a 256-wide MLP")
    expected_training = {
        "seeds": list(EXPECTED_SEEDS),
        "hidden_size": 256,
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.002,
        "weight_decay": 0.0001,
        "label_smoothing": 0.05,
        "load_balance_weight": 0.01,
        "z_loss_weight": 0.0001,
        "patience": 4,
        "device": "cpu",
        "max_tokens": 256,
        "feature_batch_size": 64,
        "seed_is_the_only_per_candidate_variable": True,
        "training_report_sidecar_path_template": (
            "checkpoints/qwen3-1.7b-candidate-routers-quarantine/"
            "router-v6-taskmaster-seed{seed}/router_config.json"
        ),
    }
    if training != expected_training:
        raise ValueError("router v6 training contract differs from the frozen CPU curriculum")

    runtime = policy.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("router v6 policy has no runtime contract")
    runtime_normalized = dict(runtime)
    if "threshold" in runtime_normalized and "confidence_threshold" not in runtime_normalized:
        runtime_normalized["confidence_threshold"] = runtime_normalized.pop("threshold")
    _require_exact_mapping(runtime_normalized, EXPECTED_RUNTIME, label="runtime")

    gates = policy.get("one_shot_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("router v6 policy has no one-shot gates")
    _require_exact_mapping(gates, EXPECTED_ONE_SHOT_GATES, label="one_shot_gates")

    probe = policy.get("blind_probe")
    if not isinstance(probe, Mapping):
        raise ValueError("router v6 policy has no blind-probe descriptor")
    if probe.get("purpose") != PROBE_PURPOSE:
        raise ValueError("router v6 blind probe is not permanently marked evaluation-only")
    if probe.get("row_count", probe.get("prompts")) != 84:
        raise ValueError("router v6 blind probe must contain exactly 84 rows")
    if _SHA256.fullmatch(str(probe.get("sha256"))) is None:
        raise ValueError("router v6 blind-probe hash is invalid")
    # Resolve and contain the path, but intentionally do not stat, hash, or parse it yet.
    _repository_path(repository, probe.get("path"), label="blind probe path")

    baseline = policy.get("baseline")
    candidates = policy.get("candidates")
    if not isinstance(baseline, Mapping) or not isinstance(candidates, Mapping):
        raise ValueError("policy baseline/candidate checkpoint contracts are missing")
    if candidates.get("required_seeds") != list(EXPECTED_SEEDS):
        raise ValueError("candidate seed set differs from the frozen training seeds")
    if candidates.get("hidden_size") != 256 or candidates.get("selected_alias") is not None:
        raise ValueError("candidate MLP width or no-alias quarantine posture changed")
    if candidates.get("quarantine_required") is not True:
        raise ValueError("candidate quarantine is no longer mandatory")
    if candidates.get("required_artifacts") != ["router_config.json", "router.safetensors"]:
        raise ValueError("candidate checkpoint completeness contract changed")
    if (
        candidates.get("candidate_artifact_sha256_must_be_recorded_before_blind_probe_open")
        is not True
    ):
        raise ValueError("candidate hashes are no longer required before blind access")

    datasets = policy.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("policy dataset contract is missing")
    expected_purposes = {
        "train": "router_training",
        "validation": "router_validation_and_seed_selection_only",
        "test": "sealed_router_dataset_diagnostic_never_training_or_seed_selection",
    }
    for split, purpose in expected_purposes.items():
        descriptor = datasets.get(split)
        if not isinstance(descriptor, Mapping) or descriptor.get("purpose") != purpose:
            raise ValueError(f"policy dataset purpose changed for {split}")
    if datasets.get("required_route_count_per_route_by_split") != {
        "train": 936,
        "validation": 130,
        "test": 130,
    } or datasets.get("required_total_count_by_split") != {
        "train": 5616,
        "validation": 780,
        "test": 780,
    }:
        raise ValueError("policy dataset balance contract changed")

    selection = policy.get("seed_selection")
    if not isinstance(selection, Mapping) or selection != {
        "metric": "freshly_recomputed_validation_loss",
        "source_split": "validation",
        "direction": "minimize",
        "tie_breaker": "lowest_numeric_seed",
        "sidecar_metrics_forbidden": True,
        "blind_probe_metrics_forbidden": True,
        "test_split_metrics_forbidden": True,
    }:
        raise ValueError("seed-selection contract differs from validation-loss-only selection")

    decision = policy.get("decision_rule")
    required_decision_flags = {
        "promotion_requires_every_dataset_preblind_and_one_shot_gate_to_pass": True,
        "blind_probe_retraining_reselection_or_runtime_retuning_forbidden": True,
        "registration_copy_or_alias_creation_during_evaluation_forbidden": True,
        "post_pass_registration_requires_separate_audited_action": True,
    }
    if not isinstance(decision, Mapping) or any(
        decision.get(name) is not value for name, value in required_decision_flags.items()
    ):
        raise ValueError("policy no longer forbids automatic registration or blind retuning")

    one_shot = policy.get("one_shot_evaluation")
    if not isinstance(one_shot, Mapping):
        raise ValueError("policy has no one-shot evaluation contract")
    output = _repository_path(repository, one_shot.get("report_path"), label="one-shot report path")
    expected_output = (
        repository / "runs" / "evaluation" / "router-v6-taskmaster-registration-one-shot.json"
    )
    if output != expected_output.resolve():
        raise ValueError("one-shot report path differs from the append-only destination")
    if (
        one_shot.get("baseline_evaluation_count") != 1
        or one_shot.get("selected_candidate_evaluation_count") != 1
        or one_shot.get("identical_probe_and_runtime_required") is not True
        or one_shot.get("retry_or_threshold_tuning_after_open_forbidden") is not True
    ):
        raise ValueError("one-shot evaluation count/runtime contract was weakened")

    if not isinstance(policy.get("runtime_source_artifacts"), Mapping):
        raise ValueError("policy has no complete runtime source-artifact contract")
    if not isinstance(policy.get("model_runtime"), Mapping):
        raise ValueError("policy has no frozen model runtime contract")

    receipt_contract = policy.get("preblind_receipt")
    if not isinstance(receipt_contract, Mapping):
        raise ValueError("policy has no preblind receipt contract")
    receipt_path = _repository_path(
        repository, receipt_contract.get("path"), label="preblind receipt path"
    )
    expected_receipt = repository / "runs" / "router-v6" / "taskmaster-preblind-receipt.json"
    if receipt_path != expected_receipt.resolve():
        raise ValueError("preblind receipt path differs from the frozen destination")
    if (
        receipt_contract.get("write_mode") != "exclusive_create_non_overwriting"
        or receipt_contract.get("required_hash_bindings") != EXPECTED_PREBLIND_HASH_BINDINGS
        or receipt_contract.get("required_validation_records")
        != EXPECTED_PREBLIND_VALIDATION_RECORDS
        or receipt_contract.get("blind_probe_opened") is not False
        or receipt_contract.get("test_metrics_used_for_seed_selection") is not False
        or receipt_contract.get("registration_changed") is not False
    ):
        raise ValueError("preblind receipt ordering or no-mutation contract was weakened")

    assert_unchanged(policy_snapshot, label="router v6 policy")
    return policy, policy_snapshot


def _policy_training_paths(
    policy: Mapping[str, Any], *, repository: Path
) -> tuple[Path, str, Path, Mapping[str, Any], Mapping[str, Any]]:
    baseline_contract = policy["baseline"]
    baseline_path_raw = baseline_contract.get("checkpoint_path")
    baseline = _repository_path(repository, baseline_path_raw, label="baseline checkpoint")
    expected_baseline = repository / "checkpoints" / "qwen3-1.7b-corpus-router-v4-mlp-controlled"
    if baseline != expected_baseline.resolve():
        raise ValueError("policy baseline is not the promoted router v4 checkpoint")

    template = policy["candidates"].get("checkpoint_path_template")
    if not isinstance(template, str) or template.count("{seed}") != 1:
        raise ValueError("candidate_checkpoint_template must contain exactly one {seed}")
    expected_template = (
        "checkpoints/qwen3-1.7b-candidate-routers-quarantine/router-v6-taskmaster-seed{seed}"
    )
    if template.replace("\\", "/") != expected_template:
        raise ValueError("candidate checkpoint template differs from the frozen quarantine path")

    dataset = policy.get("datasets")
    if not isinstance(dataset, Mapping):
        raise ValueError("training contract has no dataset object")
    report = dataset.get("manifest_report")
    splits = {split: dataset.get(split) for split in ("train", "validation", "test")}
    if not isinstance(report, Mapping) or any(
        not isinstance(value, Mapping) for value in splits.values()
    ):
        raise ValueError("training dataset report/split descriptors are missing")
    return (
        baseline,
        template,
        _repository_path(repository, report.get("path"), label="dataset report path"),
        report,
        splits,
    )


def _normalized_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def _read_router_split(path: Path, *, route_ids: Sequence[str]) -> tuple[set[str], Counter[str]]:
    prompts: set[str] = set()
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"prompt", "route_id"}:
                    raise ValueError("row must contain exactly prompt and route_id")
                prompt = row["prompt"]
                route_id = row["route_id"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be non-empty")
                if route_id not in route_ids:
                    raise ValueError(f"unknown route {route_id!r}")
                normalized = _normalized_prompt(prompt)
                if normalized in prompts:
                    raise ValueError("normalized prompt is duplicated within the split")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            prompts.add(normalized)
            counts[route_id] += 1
    if not prompts:
        raise ValueError(f"router split is empty: {path}")
    return prompts, counts


def _overlap_contract_is_zero(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_overlap_contract_is_zero(item) for item in value.values())
    if isinstance(value, list):
        return not value
    if isinstance(value, bool):
        return value is False
    if isinstance(value, int):
        return value == 0
    return False


def validate_dataset_contract(
    policy: Mapping[str, Any], *, repository: Path = WORKSPACE_ROOT
) -> tuple[dict[str, ArtifactSnapshot], dict[str, Any]]:
    """Validate frozen dataset bytes and independently prove prompt split disjointness."""

    _baseline, _template, report_path, report_descriptor, split_descriptors = (
        _policy_training_paths(policy, repository=repository)
    )
    expected_report = (
        repository
        / "experiments"
        / "first-real-experts"
        / "data"
        / "router-bootstrap-v6-taskmaster.report.json"
    )
    if report_path != expected_report.resolve():
        raise ValueError("dataset report path differs from the frozen location")
    report_snapshot = _descriptor_path(
        report_descriptor,
        repository=repository,
        label="router v6 dataset report",
        expected_path=expected_report,
    )
    report = _read_json_object(report_path, label="router v6 dataset report")
    if report.get("schema_version") != "molor.router-bootstrap-dataset.v6":
        raise ValueError("unsupported router v6 dataset report schema")
    if report.get("route_ids") != policy["route_ids"]:
        raise ValueError("dataset report route axis differs from policy")
    if report.get("authorized_taskmaster_source_splits") != ["train", "validation"]:
        raise ValueError("dataset report used an unauthorized Taskmaster source split")
    for key in (
        "evaluation_artifacts_read_by_builder",
        "taskmaster_test_split_read_by_builder",
        "taskmaster_prediction_artifacts_read_by_builder",
    ):
        if report.get(key) is not False:
            raise ValueError(f"dataset report does not preserve evaluation exclusion: {key}")
    for key in ("normalized_prompt_overlap", "semantic_group_overlap"):
        value = report.get(key)
        if value is None or not _overlap_contract_is_zero(value):
            raise ValueError(f"dataset report does not prove zero {key}")

    snapshots: dict[str, ArtifactSnapshot] = {"report": report_snapshot}
    source_identity = report.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("dataset report source identity is missing")
    builder_path_raw = source_identity.get("builder")
    if not isinstance(builder_path_raw, str) or not builder_path_raw:
        raise ValueError("dataset report builder path is invalid")
    builder_path = Path(builder_path_raw).resolve()
    expected_builder = (
        repository / "experiments" / "first-real-experts" / "build_router_v6_taskmaster_dataset.py"
    ).resolve()
    if builder_path != expected_builder:
        raise ValueError("dataset report was not emitted by the frozen v6 builder")
    builder_snapshot = snapshot(builder_path)
    if source_identity.get("builder_sha256") != builder_snapshot.sha256:
        raise ValueError("dataset report builder hash differs from current source")
    snapshots["source:dataset_builder"] = builder_snapshot

    legacy = report.get("legacy_v4_source")
    if not isinstance(legacy, Mapping):
        raise ValueError("dataset report legacy v4 source binding is missing")
    legacy_report = legacy.get("report")
    legacy_outputs = legacy.get("outputs")
    if not isinstance(legacy_report, Mapping) or not isinstance(legacy_outputs, Mapping):
        raise ValueError("dataset report legacy v4 descriptors are incomplete")
    snapshots["source:legacy_v4_report"] = _reported_artifact(
        legacy_report,
        repository=repository,
        label="legacy v4 report",
        expected_path=(
            repository
            / "experiments"
            / "first-real-experts"
            / "data"
            / "router-bootstrap-v4-hard-negative-s307.report.json"
        ),
    )
    for split in ("train", "validation", "test"):
        snapshots[f"source:legacy_v4_{split}"] = _reported_artifact(
            legacy_outputs.get(split, {}),
            repository=repository,
            label=f"legacy v4 {split}",
            expected_path=(
                repository
                / "experiments"
                / "first-real-experts"
                / "data"
                / f"router-bootstrap-v4-hard-negative-s307.{split}.jsonl"
            ),
            count_rows=True,
        )

    taskmaster = report.get("taskmaster_source")
    taskmaster_inputs = taskmaster.get("inputs") if isinstance(taskmaster, Mapping) else None
    if (
        not isinstance(taskmaster, Mapping)
        or taskmaster.get("route_id") != TASKMASTER_ROUTE_ID
        or taskmaster.get("source_split_mapping")
        != {"train": "train", "validation": "validation", "test": "validation"}
        or not isinstance(taskmaster_inputs, Mapping)
        or set(taskmaster_inputs) != {"train", "validation"}
    ):
        raise ValueError("dataset report Taskmaster source contract is invalid")
    for source_split in ("train", "validation"):
        descriptors = taskmaster_inputs[source_split]
        if not isinstance(descriptors, Mapping) or set(descriptors) != {
            "training",
            "provenance",
        }:
            raise ValueError(f"Taskmaster {source_split} source descriptors are incomplete")
        for kind, suffix in (("training", ".jsonl"), ("provenance", ".provenance.jsonl")):
            snapshots[f"source:taskmaster_{source_split}_{kind}"] = _reported_artifact(
                descriptors[kind],
                repository=repository,
                label=f"Taskmaster {source_split} {kind}",
                expected_path=(
                    repository
                    / "experiments"
                    / "first-real-experts"
                    / "data"
                    / f"taskmaster-coffee-api-v2.{source_split}{suffix}"
                ),
                count_rows=True,
            )
    selected_overlap = taskmaster.get("selected_component_overlap")
    if not _overlap_contract_is_zero(selected_overlap):
        raise ValueError("Taskmaster components overlap across router splits")
    if taskmaster.get("train_validation_source_component_overlap") != 0:
        raise ValueError("Taskmaster source components overlap across train and validation")

    prompt_sets: dict[str, set[str]] = {}
    route_counts: dict[str, Counter[str]] = {}
    report_outputs = report.get("outputs")
    if not isinstance(report_outputs, Mapping):
        raise ValueError("dataset report outputs are missing")
    for split in ("train", "validation", "test"):
        expected_path = (
            repository
            / "experiments"
            / "first-real-experts"
            / "data"
            / f"router-bootstrap-v6-taskmaster.{split}.jsonl"
        )
        frozen_by_policy = _descriptor_path(
            split_descriptors[split],
            repository=repository,
            label=f"router v6 {split} split",
            expected_path=expected_path,
        )
        reported = report_outputs.get(split)
        if not isinstance(reported, Mapping):
            raise ValueError(f"dataset report has no {split} output")
        frozen = _reported_artifact(
            reported,
            repository=repository,
            label=f"router v6 {split} split",
            expected_path=expected_path,
            count_rows=True,
        )
        if frozen.sha256 != frozen_by_policy.sha256:
            raise ValueError(f"router v6 {split} policy/report hashes disagree")
        reported_rows = reported.get("rows")
        prompts, counts = _read_router_split(frozen.path, route_ids=policy["route_ids"])
        if reported_rows != len(prompts):
            raise ValueError(f"dataset {split} row count differs from its descriptors")
        if split_descriptors[split].get("rows") != len(prompts):
            raise ValueError(f"dataset {split} row count differs from frozen policy")
        expected_total = policy["datasets"]["required_total_count_by_split"][split]
        expected_per_route = policy["datasets"]["required_route_count_per_route_by_split"][split]
        if len(prompts) != expected_total or any(
            counts.get(route_id) != expected_per_route for route_id in policy["route_ids"]
        ):
            raise ValueError(f"dataset {split} route balance differs from policy")
        snapshots[split] = frozen
        prompt_sets[split] = prompts
        route_counts[split] = counts

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if prompt_sets[left].intersection(prompt_sets[right]):
            raise ValueError(f"normalized prompts overlap between {left} and {right}")

    balance = report.get("balance")
    if isinstance(balance, Mapping):
        actual_balance = {
            split: dict(sorted(route_counts[split].items())) for split in route_counts
        }
        normalized_balance = {
            split: dict(sorted(values.items()))
            for split, values in balance.items()
            if isinstance(values, Mapping)
        }
        if normalized_balance != actual_balance:
            raise ValueError("dataset balance differs from independently parsed splits")

    for label, value in snapshots.items():
        assert_unchanged(value, label=label)
    return snapshots, report


def _sidecar_artifact_hash(sidecar: Mapping[str, Any], *, checkpoint: Path) -> str:
    descriptor = sidecar.get("checkpoint")
    if not isinstance(descriptor, Mapping):
        raise ValueError("router sidecar has no checkpoint descriptor")
    weights = checkpoint / "router.safetensors"
    if Path(str(descriptor.get("path"))).resolve() != weights.resolve():
        raise ValueError("router sidecar checkpoint path differs from candidate path")
    digest = descriptor.get("sha256")
    if _SHA256.fullmatch(str(digest)) is None or sha256_file(weights) != digest:
        raise ValueError("router sidecar checkpoint hash differs from candidate weights")
    return str(digest)


def _adapter_map(sidecar: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = sidecar.get("adapters")
    if not isinstance(raw, list):
        raise ValueError("router sidecar adapter list is missing")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("router sidecar adapter entry must be an object")
        route_id = item.get("route_id")
        if not isinstance(route_id, str) or not route_id or route_id in result:
            raise ValueError("router sidecar adapter route IDs are invalid")
        result[route_id] = item
    return result


def _validate_adapter_entry(
    entry: Mapping[str, Any],
    *,
    route_id: str,
    expected_path: Path,
) -> tuple[ArtifactSnapshot, ArtifactSnapshot]:
    if Path(str(entry.get("path"))).resolve() != expected_path.resolve():
        raise ValueError(f"adapter path changed for {route_id}")
    results: list[ArtifactSnapshot] = []
    for key, filename in (
        ("config", "adapter_config.json"),
        ("weights", "adapter_model.safetensors"),
    ):
        descriptor = entry.get(key)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"adapter {route_id} has no {key} descriptor")
        path = expected_path / filename
        if Path(str(descriptor.get("path"))).resolve() != path.resolve():
            raise ValueError(f"adapter {route_id} {key} path changed")
        actual = snapshot(path)
        if descriptor.get("sha256") != actual.sha256:
            raise ValueError(f"adapter {route_id} {key} hash changed")
        if descriptor.get("size_bytes") != actual.size_bytes:
            raise ValueError(f"adapter {route_id} {key} size changed")
        results.append(actual)
    return results[0], results[1]


def _dataset_hashes(sidecar: Mapping[str, Any]) -> dict[str, str]:
    datasets = sidecar.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("router sidecar dataset descriptors are missing")
    if set(datasets) != {"train", "validation", "test"}:
        raise ValueError("router sidecar must bind exactly three dataset splits")
    result: dict[str, str] = {}
    for split, descriptor in datasets.items():
        if (
            not isinstance(descriptor, Mapping)
            or _SHA256.fullmatch(str(descriptor.get("sha256"))) is None
        ):
            raise ValueError(f"router sidecar {split} dataset descriptor is invalid")
        result[split] = str(descriptor["sha256"])
    return result


def _metric_gate_values(
    validation: Mapping[str, Any], *, route_ids: Sequence[str]
) -> tuple[float, float, float, float, dict[str, float]]:
    if not isinstance(validation, Mapping):
        raise ValueError("recomputed validation metrics are missing")
    loss = _number(validation.get("loss"), label="validation loss")
    accuracy = _number(validation.get("accuracy"), label="validation accuracy")
    per_route = validation.get("per_route")
    if not isinstance(per_route, Mapping) or set(per_route) != set(route_ids):
        raise ValueError("router sidecar per-route validation metrics are incomplete")

    def route_metric(route_id: str, metric: str) -> float:
        value = per_route.get(route_id)
        if not isinstance(value, Mapping):
            raise ValueError(f"validation metrics are missing route {route_id}")
        return _number(value.get(metric), label=f"{route_id} {metric}")

    taskmaster_precision = route_metric(TASKMASTER_ROUTE_ID, "precision")
    taskmaster_recall = route_metric(TASKMASTER_ROUTE_ID, "recall")
    legacy_recalls = {
        route_id: route_metric(route_id, "recall")
        for route_id in route_ids
        if route_id != TASKMASTER_ROUTE_ID
    }
    return loss, accuracy, taskmaster_precision, taskmaster_recall, legacy_recalls


def assess_recomputed_validation_metrics(
    validation: Mapping[str, Any],
    *,
    preblind: Mapping[str, Any],
    route_ids: Sequence[str],
) -> tuple[
    float,
    float,
    float,
    float,
    dict[str, float],
    tuple[Mapping[str, Any], ...],
]:
    """Apply preblind gates exclusively to freshly recomputed validation metrics."""

    loss, accuracy, tm_precision, tm_recall, legacy_recalls = _metric_gate_values(
        validation, route_ids=route_ids
    )
    gates = (
        _gate(
            "validation_accuracy",
            actual=accuracy,
            required=">=0.99",
            passed=accuracy >= preblind["validation_accuracy_min_for_every_seed"],
        ),
        _gate(
            "taskmaster_precision",
            actual=tm_precision,
            required=">=0.99",
            passed=tm_precision >= preblind["taskmaster_validation_precision_min_for_every_seed"],
        ),
        _gate(
            "taskmaster_recall",
            actual=tm_recall,
            required=">=0.99",
            passed=tm_recall >= preblind["taskmaster_validation_recall_min_for_every_seed"],
        ),
        _gate(
            "legacy_route_recall",
            actual=min(legacy_recalls.values()),
            required=">=0.98 for every legacy route",
            passed=all(
                value >= preblind["legacy_route_validation_recall_min_for_every_seed"]
                for value in legacy_recalls.values()
            ),
        ),
    )
    return loss, accuracy, tm_precision, tm_recall, legacy_recalls, gates


def recompute_validation_metrics(
    router: Any,
    features: Tensor,
    labels: Tensor,
    *,
    route_ids: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    """Freshly score one candidate on CPU without consulting saved metrics."""

    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("validation features and labels are not aligned")
    if not labels.numel():
        raise ValueError("validation split is empty")
    if batch_size < 1:
        raise ValueError("validation batch size must be positive")
    router = router.float().cpu().eval().requires_grad_(False)
    route_count = len(route_ids)
    total_loss = 0.0
    total = 0
    correct = 0
    confidence_sum = 0.0
    confusion = [[0 for _ in range(route_count)] for _ in range(route_count)]
    with torch.inference_mode():
        for start in range(0, labels.numel(), batch_size):
            batch_features = features[start : start + batch_size].float().cpu()
            batch_labels = labels[start : start + batch_size].long().cpu()
            output = router(batch_features)
            logits = output.logits.float()
            probabilities = output.probabilities.float()
            predictions = logits.argmax(dim=-1)
            total_loss += float(F.cross_entropy(logits, batch_labels, reduction="sum").item())
            total += int(batch_labels.numel())
            correct += int((predictions == batch_labels).sum().item())
            confidence_sum += float(probabilities.max(dim=-1).values.sum().item())
            for expected, predicted in zip(
                batch_labels.tolist(), predictions.tolist(), strict=True
            ):
                confusion[expected][predicted] += 1
    per_route: dict[str, dict[str, Any]] = {}
    for index, route_id in enumerate(route_ids):
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        true_positive = confusion[index][index]
        per_route[route_id] = {
            "support": support,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / support if support else 0.0,
        }
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "mean_confidence": confidence_sum / total,
        "examples": total,
        "confusion": confusion,
        "per_route": per_route,
    }


def sidecar_metric_diagnostics(
    reported: Any,
    recomputed: Mapping[str, Any],
    *,
    route_ids: Sequence[str],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare saved metrics for audit visibility; saved values never drive gates."""

    diagnostics: dict[str, Any] = {
        "role": "diagnostic_only_never_used_for_gates_or_selection",
        "tolerance": tolerance,
        "reported_metrics_well_formed": False,
        "matches_recomputed": False,
    }
    if not isinstance(reported, Mapping):
        return diagnostics
    try:
        reported_loss = _number(reported.get("loss"), label="reported validation loss")
        reported_accuracy = _number(reported.get("accuracy"), label="reported validation accuracy")
        reported_confidence = _number(
            reported.get("mean_confidence"), label="reported validation confidence"
        )
        reported_examples = reported.get("examples")
        reported_routes = reported.get("per_route")
        if (
            not isinstance(reported_examples, int)
            or isinstance(reported_examples, bool)
            or not isinstance(reported_routes, Mapping)
            or set(reported_routes) != set(route_ids)
        ):
            return diagnostics
        route_deltas: dict[str, dict[str, float]] = {}
        routes_match = True
        for route_id in route_ids:
            values = reported_routes[route_id]
            expected = recomputed["per_route"][route_id]
            if not isinstance(values, Mapping) or values.get("support") != expected["support"]:
                return diagnostics
            precision_delta = _number(
                values.get("precision"), label=f"reported {route_id} precision"
            ) - float(expected["precision"])
            recall_delta = _number(
                values.get("recall"), label=f"reported {route_id} recall"
            ) - float(expected["recall"])
            route_deltas[route_id] = {
                "precision": precision_delta,
                "recall": recall_delta,
            }
            routes_match &= abs(precision_delta) <= tolerance and abs(recall_delta) <= tolerance
        deltas = {
            "loss": reported_loss - float(recomputed["loss"]),
            "accuracy": reported_accuracy - float(recomputed["accuracy"]),
            "mean_confidence": reported_confidence - float(recomputed["mean_confidence"]),
        }
        diagnostics.update(
            {
                "reported_metrics_well_formed": True,
                "scalar_deltas_reported_minus_recomputed": deltas,
                "per_route_deltas_reported_minus_recomputed": route_deltas,
                "examples_match": reported_examples == recomputed["examples"],
                "confusion_match": reported.get("confusion") == recomputed["confusion"],
            }
        )
        diagnostics["matches_recomputed"] = (
            all(abs(value) <= tolerance for value in deltas.values())
            and routes_match
            and diagnostics["examples_match"]
            and diagnostics["confusion_match"]
        )
    except (TypeError, ValueError):
        return diagnostics
    return diagnostics


def prepare_validation_features(
    policy: Mapping[str, Any],
    *,
    validation_path: Path,
    embedding: Tensor,
    tokenizer: Any,
) -> tuple[Tensor, Tensor, RouterFeatureContract]:
    """Load and featurize only the frozen router validation split for seed selection."""

    route_ids = tuple(policy["route_ids"])
    examples = load_router_jsonl(validation_path, route_ids=route_ids)
    expected_rows = policy["datasets"]["required_total_count_by_split"]["validation"]
    if len(examples) != expected_rows:
        raise ValueError("fresh validation example count differs from frozen policy")
    contract = RouterFeatureContract(max_tokens=policy["training"]["max_tokens"])
    features, labels = extract_prompt_features(
        examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=route_ids,
        contract=contract,
        batch_size=policy["training"]["feature_batch_size"],
    )
    return features, labels, contract


def validate_candidates(
    policy: Mapping[str, Any],
    *,
    dataset_snapshots: Mapping[str, ArtifactSnapshot],
    validation_features: Tensor,
    validation_labels: Tensor,
    feature_contract: RouterFeatureContract,
    model_evidence: ModelRuntimeEvidence,
    repository: Path = WORKSPACE_ROOT,
) -> tuple[
    list[CandidateEvidence],
    dict[str, ArtifactSnapshot],
    dict[str, Any],
    dict[int, tuple[Any, Mapping[str, Any]]],
]:
    """Validate all seeds and select solely by frozen validation loss."""

    baseline, template, _report, _report_descriptor, _splits = _policy_training_paths(
        policy, repository=repository
    )
    baseline_config = snapshot(baseline / "router_config.json")
    baseline_weights = snapshot(baseline / "router.safetensors")
    baseline_pins = policy["baseline"].get("artifacts")
    if not isinstance(baseline_pins, Mapping) or baseline_pins != {
        "router_config.json": baseline_config.sha256,
        "router.safetensors": baseline_weights.sha256,
    }:
        raise ValueError("promoted router v4 artifacts differ from frozen policy")
    baseline_sidecar = _read_json_object(baseline_config.path, label="promoted router v4 sidecar")
    if baseline_sidecar.get("schema_version") != ROUTER_CHECKPOINT_SCHEMA:
        raise ValueError("promoted router v4 sidecar schema is unsupported")
    _sidecar_artifact_hash(baseline_sidecar, checkpoint=baseline)
    baseline_routes = tuple(baseline_sidecar.get("route_ids", ()))
    if (
        tuple(route for route in policy["route_ids"] if route != TASKMASTER_ROUTE_ID)
        != baseline_routes
    ):
        raise ValueError("router v6 legacy route order differs from promoted v4")
    validate_model_sidecar(
        baseline_sidecar,
        evidence=model_evidence,
        label="promoted router v4 sidecar",
    )
    baseline_adapters = _adapter_map(baseline_sidecar)

    expert_pins = policy.get("expert_artifacts")
    if not isinstance(expert_pins, Mapping) or set(expert_pins) != set(policy["route_ids"]):
        raise ValueError("policy expert-artifact contract is incomplete")
    base_pin = expert_pins.get(NULL_ROUTE_ID)
    if not isinstance(base_pin, Mapping) or base_pin != {
        "base_model_route": True,
        "path": None,
        "artifacts": {},
    }:
        raise ValueError("base/null must remain a structural route without an adapter")
    expert_snapshots: dict[str, tuple[Path, ArtifactSnapshot, ArtifactSnapshot]] = {}
    for route_id in policy["route_ids"]:
        if route_id == NULL_ROUTE_ID:
            continue
        pin = expert_pins[route_id]
        if not isinstance(pin, Mapping):
            raise ValueError(f"policy expert pin is invalid for {route_id}")
        expert_path = _repository_path(
            repository, pin.get("path"), label=f"policy expert {route_id} path"
        )
        artifacts = pin.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            raise ValueError(f"policy expert artifacts are incomplete for {route_id}")
        config_snapshot = snapshot(expert_path / "adapter_config.json")
        weights_snapshot = snapshot(expert_path / "adapter_model.safetensors")
        if artifacts != {
            "adapter_config.json": config_snapshot.sha256,
            "adapter_model.safetensors": weights_snapshot.sha256,
        }:
            raise ValueError(f"expert artifact bytes changed for {route_id}")
        expert_snapshots[route_id] = (expert_path, config_snapshot, weights_snapshot)
    taskmaster_path, taskmaster_config, taskmaster_weights = expert_snapshots[TASKMASTER_ROUTE_ID]
    expected_taskmaster = (
        repository / "checkpoints" / "qwen3-1.7b-candidate-experts-quarantine" / TASKMASTER_ROUTE_ID
    )
    if taskmaster_path != expected_taskmaster.resolve():
        raise ValueError("Taskmaster adapter path differs from the quarantined candidate")

    frozen_split_hashes = {
        split: dataset_snapshots[split].sha256 for split in ("train", "validation", "test")
    }
    preblind = policy.get("preblind_gates")
    if not isinstance(preblind, Mapping):
        raise ValueError("policy has no preblind gates")
    required_preblind_values = {
        "candidate_count": 3,
        "validation_accuracy_min_for_every_seed": 0.99,
        "taskmaster_validation_precision_min_for_every_seed": 0.99,
        "taskmaster_validation_recall_min_for_every_seed": 0.99,
        "legacy_route_validation_recall_min_for_every_seed": 0.98,
        "legacy_routes": [route for route in policy["route_ids"] if route != TASKMASTER_ROUTE_ID],
        "blind_probe_contents_opened_before_seed_selection_max": 0,
        "blind_probe_metrics_used_for_seed_selection_max": 0,
    }
    for name, required in required_preblind_values.items():
        if preblind.get(name) != required:
            raise ValueError(f"preblind_gates.{name} differs from the frozen contract")
    required_true = {
        "dataset_gates_must_all_pass",
        "seed_set_must_exactly_match_training_seeds",
        "finite_validation_loss_required_for_every_seed",
        "candidate_checkpoint_files_complete",
        "candidate_route_ids_must_exactly_equal_policy_route_ids_in_order",
        "candidate_hidden_size_must_equal_policy_hidden_size",
        "candidate_training_hyperparameters_must_exactly_equal_policy_training",
        "candidate_paths_must_remain_in_quarantine",
        "baseline_artifact_sha256_must_match",
        "expert_artifact_sha256_must_match",
        "candidate_artifact_sha256_must_be_recorded",
        "runtime_source_tree_and_scripts_must_match",
        "model_runtime_artifacts_must_match",
        "validation_metrics_must_be_freshly_recomputed_on_cpu",
        "validation_split_only_must_drive_seed_selection",
    }
    if any(preblind.get(name) is not True for name in required_true):
        raise ValueError("one or more structural preblind gates were weakened")
    if (
        preblind.get("sidecar_validation_metrics_used_for_gates_or_selection") is not False
        or preblind.get("test_metrics_used_for_seed_selection") is not False
    ):
        raise ValueError("saved or test metrics are no longer excluded from seed selection")

    evidence: list[CandidateEvidence] = []
    snapshots: dict[str, ArtifactSnapshot] = {
        "baseline_config": baseline_config,
        "baseline_weights": baseline_weights,
        "taskmaster_adapter_config": taskmaster_config,
        "taskmaster_adapter_weights": taskmaster_weights,
    }
    for route_id, (_path, config_value, weights_value) in expert_snapshots.items():
        snapshots[f"expert_{route_id}_config"] = config_value
        snapshots[f"expert_{route_id}_weights"] = weights_value
    sidecars: dict[int, dict[str, Any]] = {}
    loaded_candidates: dict[int, tuple[Any, Mapping[str, Any]]] = {}
    for seed in EXPECTED_SEEDS:
        checkpoint = _repository_path(
            repository, template.format(seed=seed), label=f"candidate seed {seed} checkpoint"
        )
        config = snapshot(checkpoint / "router_config.json")
        weights = snapshot(checkpoint / "router.safetensors")
        sidecar = _read_json_object(config.path, label=f"candidate seed {seed} sidecar")
        sidecars[seed] = sidecar
        if sidecar.get("schema_version") != ROUTER_CHECKPOINT_SCHEMA:
            raise ValueError(f"candidate seed {seed} sidecar schema is unsupported")
        if sidecar.get("route_ids") != policy["route_ids"]:
            raise ValueError(f"candidate seed {seed} route axis differs from policy")
        training_config = sidecar.get("training_config")
        router_config = sidecar.get("router_config")
        if not isinstance(training_config, Mapping) or training_config.get("seed") != seed:
            raise ValueError(f"candidate seed {seed} does not bind its training seed")
        if not isinstance(router_config, Mapping) or router_config.get("router_hidden_size") != 256:
            raise ValueError(f"candidate seed {seed} is not the frozen 256-wide MLP")
        expected_training_config = {
            "epochs": policy["training"]["epochs"],
            "batch_size": policy["training"]["batch_size"],
            "learning_rate": policy["training"]["learning_rate"],
            "weight_decay": policy["training"]["weight_decay"],
            "label_smoothing": policy["training"]["label_smoothing"],
            "load_balance_weight": policy["training"]["load_balance_weight"],
            "z_loss_weight": policy["training"]["z_loss_weight"],
            "patience": policy["training"]["patience"],
            "seed": seed,
            "device": policy["training"]["device"],
            "router_hidden_size": policy["training"]["hidden_size"],
        }
        if training_config != expected_training_config:
            raise ValueError(f"candidate seed {seed} training hyperparameters differ from policy")
        if sidecar.get("feature_contract") != feature_contract.to_dict():
            raise ValueError(f"candidate seed {seed} prompt feature contract differs from policy")
        validate_model_sidecar(
            sidecar,
            evidence=model_evidence,
            label=f"candidate seed {seed} sidecar",
        )
        if _sidecar_artifact_hash(sidecar, checkpoint=checkpoint) != weights.sha256:
            raise ValueError(f"candidate seed {seed} weights disagree with its sidecar")
        if _dataset_hashes(sidecar) != frozen_split_hashes:
            raise ValueError(f"candidate seed {seed} was not trained on frozen v6 datasets")
        for split, descriptor in sidecar["datasets"].items():
            if Path(str(descriptor.get("path"))).resolve() != dataset_snapshots[split].path:
                raise ValueError(f"candidate seed {seed} {split} dataset path changed")

        adapters = _adapter_map(sidecar)
        if set(adapters) != set(policy["route_ids"]).difference({NULL_ROUTE_ID}):
            raise ValueError(f"candidate seed {seed} adapter routes are incomplete")
        for route_id, baseline_entry in baseline_adapters.items():
            baseline_path, expected_config, expected_weights = expert_snapshots[route_id]
            baseline_config_value, baseline_weights_value = _validate_adapter_entry(
                baseline_entry, route_id=route_id, expected_path=baseline_path
            )
            if (baseline_config_value.sha256, baseline_weights_value.sha256) != (
                expected_config.sha256,
                expected_weights.sha256,
            ):
                raise ValueError(f"promoted v4 changed frozen expert {route_id}")
            found_config, found_weights = _validate_adapter_entry(
                adapters[route_id],
                route_id=route_id,
                expected_path=baseline_path,
            )
            if (found_config.sha256, found_weights.sha256) != (
                expected_config.sha256,
                expected_weights.sha256,
            ):
                raise ValueError(f"candidate seed {seed} changed legacy adapter {route_id}")
        found_tm_config, found_tm_weights = _validate_adapter_entry(
            adapters[TASKMASTER_ROUTE_ID],
            route_id=TASKMASTER_ROUTE_ID,
            expected_path=taskmaster_path,
        )
        if (found_tm_config.sha256, found_tm_weights.sha256) != (
            taskmaster_config.sha256,
            taskmaster_weights.sha256,
        ):
            raise ValueError(f"candidate seed {seed} changed the frozen Taskmaster adapter")

        router, loaded_sidecar = ROUTER_PROBE.load_checkpoint(checkpoint)
        if loaded_sidecar != sidecar:
            raise OSError(f"candidate seed {seed} sidecar changed while loading weights")
        recomputed = recompute_validation_metrics(
            router,
            validation_features,
            validation_labels,
            route_ids=policy["route_ids"],
            batch_size=policy["training"]["batch_size"],
        )
        metrics_payload = sidecar.get("metrics")
        reported_validation = (
            metrics_payload.get("validation") if isinstance(metrics_payload, Mapping) else None
        )
        diagnostics = sidecar_metric_diagnostics(
            reported_validation,
            recomputed,
            route_ids=policy["route_ids"],
        )
        loss, accuracy, tm_precision, tm_recall, legacy_recalls, gates = (
            assess_recomputed_validation_metrics(
                recomputed,
                preblind=preblind,
                route_ids=policy["route_ids"],
            )
        )
        evidence.append(
            CandidateEvidence(
                seed=seed,
                checkpoint=checkpoint,
                config=config,
                weights=weights,
                validation_loss=loss,
                validation_accuracy=accuracy,
                taskmaster_precision=tm_precision,
                taskmaster_recall=tm_recall,
                legacy_recalls=legacy_recalls,
                passed=all(bool(gate["passed"]) for gate in gates),
                gates=gates,
                sidecar_metric_diagnostics=diagnostics,
            )
        )
        loaded_candidates[seed] = (router, sidecar)
        snapshots[f"candidate_{seed}_config"] = config
        snapshots[f"candidate_{seed}_weights"] = weights

    if not all(item.passed for item in evidence):
        failed = [item.seed for item in evidence if not item.passed]
        raise PreblindMetricGateFailure(
            f"preblind candidate gates failed for seeds {failed!r}; probe remains sealed",
            candidates=evidence,
            snapshots=snapshots,
        )

    selected = select_candidate(evidence)

    for label, value in snapshots.items():
        assert_unchanged(value, label=label)
    return (
        evidence,
        snapshots,
        {
            "metric": "freshly_recomputed_validation_cross_entropy",
            "tie_break": "ascending_seed",
            "selected_seed": selected.seed,
            "selected_checkpoint": selected.checkpoint.relative_to(repository).as_posix(),
            "selected_validation_loss": selected.validation_loss,
            "blind_probe_opened_for_selection": False,
            "sidecar_metrics_used_for_selection": False,
            "test_metrics_used_for_selection": False,
        },
        loaded_candidates,
    )


def _count_probe_lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def open_frozen_probe(
    policy: Mapping[str, Any], *, repository: Path = WORKSPACE_ROOT
) -> tuple[tuple[Any, ...], ArtifactSnapshot]:
    """Verify and parse the blind probe.  Call only after candidate selection is frozen."""

    descriptor = policy["blind_probe"]
    path = _repository_path(repository, descriptor["path"], label="blind probe path")
    expected = repository / "experiments" / "first-real-experts" / "router-blind-probe-v5.jsonl"
    if path != expected.resolve():
        raise ValueError("blind probe path differs from router-blind-probe-v5.jsonl")
    value = snapshot(path)
    if value.sha256 != descriptor["sha256"]:
        raise ValueError("blind probe bytes differ from the frozen policy")
    expected_rows = descriptor.get("row_count", descriptor.get("prompts"))
    if _count_probe_lines(path) != expected_rows:
        raise ValueError("blind probe line count differs from the frozen policy")
    items = tuple(ROUTER_PROBE.load_probe(path))
    if len(items) != expected_rows:
        raise ValueError("blind probe parsed count differs from the frozen policy")
    if any(item.purpose != PROBE_PURPOSE for item in items):
        raise ValueError("blind probe contains a non-evaluation purpose marker")
    expected_strata = descriptor.get("category_counts")
    actual_strata = dict(sorted(Counter(item.category for item in items).items()))
    if not isinstance(expected_strata, Mapping) or actual_strata != dict(
        sorted(expected_strata.items())
    ):
        raise ValueError("blind probe strata differ from the frozen policy")
    taskmaster_ids = [item.probe_id for item in items if item.expected_route == TASKMASTER_ROUTE_ID]
    if taskmaster_ids != descriptor.get("taskmaster_positive_ids"):
        raise ValueError("blind probe Taskmaster-positive IDs differ from the frozen policy")
    if len(items) - len(taskmaster_ids) != descriptor.get("non_taskmaster_row_count"):
        raise ValueError("blind probe non-Taskmaster count differs from the frozen policy")
    return items, value


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def evaluate_one_shot_gates(
    *,
    items: Sequence[Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered gates to aligned baseline and candidate predictions."""

    if len(items) != 84 or len(baseline_rows) != 84 or len(candidate_rows) != 84:
        raise ValueError("one-shot report completeness requires 84 aligned rows")
    item_ids = [item.probe_id for item in items]
    if [row.get("id") for row in baseline_rows] != item_ids or [
        row.get("id") for row in candidate_rows
    ] != item_ids:
        raise ValueError("one-shot predictions are not aligned to frozen probe IDs")
    if any(item.expected_route is None for item in items):
        raise ValueError("one-shot registration requires exact labels on all probe rows")
    if not _finite_tree(baseline_rows) or not _finite_tree(candidate_rows):
        raise ValueError("one-shot predictions contain non-finite values")

    baseline_exact = sum(row.get("exact_match") is True for row in baseline_rows)
    candidate_exact = sum(row.get("exact_match") is True for row in candidate_rows)
    baseline_acceptable = sum(row.get("acceptable_match") is True for row in baseline_rows)
    candidate_acceptable = sum(row.get("acceptable_match") is True for row in candidate_rows)
    taskmaster_indices = [
        index for index, item in enumerate(items) if item.expected_route == TASKMASTER_ROUTE_ID
    ]
    non_taskmaster_indices = [
        index for index, item in enumerate(items) if item.expected_route != TASKMASTER_ROUTE_ID
    ]
    if len(taskmaster_indices) != 12 or len(non_taskmaster_indices) != 72:
        raise ValueError("one-shot probe must contain 12 Taskmaster and 72 non-Taskmaster rows")
    taskmaster_correct = sum(
        candidate_rows[index].get("primary_route") == TASKMASTER_ROUTE_ID
        for index in taskmaster_indices
    )
    taskmaster_false_positives = sum(
        candidate_rows[index].get("primary_route") == TASKMASTER_ROUTE_ID
        for index in non_taskmaster_indices
    )

    gates: list[dict[str, Any]] = [
        _gate(
            "report_completeness_and_finiteness",
            actual=84,
            required="84 aligned finite rows",
            passed=True,
        ),
        _gate(
            "minimum_overall_exact_accuracy",
            actual=candidate_exact / 84,
            required=">=0.75",
            passed=candidate_exact / 84 >= 0.75,
        ),
        _gate(
            "minimum_overall_acceptable_accuracy",
            actual=candidate_acceptable / 84,
            required=">=0.75",
            passed=candidate_acceptable / 84 >= 0.75,
        ),
        _gate(
            "exact_correct_improvement_over_v4",
            actual=candidate_exact - baseline_exact,
            required=">=8",
            passed=candidate_exact - baseline_exact >= 8,
        ),
        _gate(
            "acceptable_correct_improvement_over_v4",
            actual=candidate_acceptable - baseline_acceptable,
            required=">=8",
            passed=candidate_acceptable - baseline_acceptable >= 8,
        ),
        _gate(
            "taskmaster_correct",
            actual=taskmaster_correct,
            required=">=10 of 12",
            passed=taskmaster_correct >= 10,
        ),
        _gate(
            "taskmaster_false_positives",
            actual=taskmaster_false_positives,
            required="<=1 of 72 non-Taskmaster rows",
            passed=taskmaster_false_positives <= 1,
        ),
    ]

    by_category: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_category[item.category].append(index)
    category_evidence: dict[str, dict[str, Any]] = {}
    for category, indices in sorted(by_category.items()):
        if len(indices) != 12:
            raise ValueError(f"probe category {category!r} does not contain exactly 12 rows")
        baseline_exact_count = sum(
            baseline_rows[index].get("exact_match") is True for index in indices
        )
        candidate_exact_count = sum(
            candidate_rows[index].get("exact_match") is True for index in indices
        )
        baseline_acceptable_count = sum(
            baseline_rows[index].get("acceptable_match") is True for index in indices
        )
        candidate_acceptable_count = sum(
            candidate_rows[index].get("acceptable_match") is True for index in indices
        )
        is_taskmaster_stratum = all(index in taskmaster_indices for index in indices)
        category_evidence[category] = {
            "count": len(indices),
            "taskmaster_stratum": is_taskmaster_stratum,
            "baseline_exact_correct": baseline_exact_count,
            "candidate_exact_correct": candidate_exact_count,
            "baseline_acceptable_correct": baseline_acceptable_count,
            "candidate_acceptable_correct": candidate_acceptable_count,
            "exact_drop": baseline_exact_count - candidate_exact_count,
            "acceptable_drop": baseline_acceptable_count - candidate_acceptable_count,
        }
        if not is_taskmaster_stratum:
            exact_drop = baseline_exact_count - candidate_exact_count
            acceptable_drop = baseline_acceptable_count - candidate_acceptable_count
            gates.append(
                _gate(
                    f"protected_category:{category}",
                    actual={"exact_drop": exact_drop, "acceptable_drop": acceptable_drop},
                    required="each drop <=1 of 12",
                    passed=exact_drop <= 1 and acceptable_drop <= 1,
                )
            )

    return {
        "baseline": {
            "exact_correct": baseline_exact,
            "acceptable_correct": baseline_acceptable,
        },
        "candidate": {
            "exact_correct": candidate_exact,
            "acceptable_correct": candidate_acceptable,
            "taskmaster_correct": taskmaster_correct,
            "taskmaster_false_positives": taskmaster_false_positives,
        },
        "by_category": category_evidence,
        "gates": gates,
        "all_gates_passed": all(bool(gate["passed"]) for gate in gates),
    }


def _redact_prompt_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "prompt"} for row in rows]


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> ArtifactSnapshot:
    """Atomically publish complete evidence without replacing an existing run."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"one-shot evidence already exists: {destination}")
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"one-shot evidence already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return snapshot(destination)


def build_preblind_receipt(
    *,
    policy_snapshot: ArtifactSnapshot,
    dataset_snapshots: Mapping[str, ArtifactSnapshot],
    runtime_source_snapshots: Mapping[str, ArtifactSnapshot],
    runtime_source_contract: Mapping[str, Any],
    model_evidence: ModelRuntimeEvidence,
    checkpoint_snapshots: Mapping[str, ArtifactSnapshot],
    candidates: Sequence[CandidateEvidence],
    selection: Mapping[str, Any],
    feature_batch_size: int,
    repository: Path,
) -> dict[str, Any]:
    """Build the durable proof that seed selection completed before blind access."""

    selected_seed = selection.get("selected_seed")
    if selected_seed not in EXPECTED_SEEDS:
        raise ValueError("preblind receipt selection has an invalid seed")
    if not all(candidate.passed for candidate in candidates):
        raise ValueError("a passing preblind receipt cannot contain a failed candidate")
    selected = next(candidate for candidate in candidates if candidate.seed == selected_seed)
    receipt: dict[str, Any] = {
        "schema_version": PREBLIND_SCHEMA,
        "purpose": "freeze candidate selection and every artifact before blind probe access",
        "policy": policy_snapshot.to_dict(repository=repository),
        "dataset_artifacts": {
            label: value.to_dict(repository=repository)
            for label, value in sorted(dataset_snapshots.items())
        },
        "runtime_source_artifacts": {
            label: value.to_dict(repository=repository)
            for label, value in sorted(runtime_source_snapshots.items())
        },
        "runtime_source_contract": dict(runtime_source_contract),
        "model_runtime": model_evidence.to_dict(repository=repository),
        "checkpoint_and_expert_artifacts": {
            label: value.to_dict(repository=repository)
            for label, value in sorted(checkpoint_snapshots.items())
        },
        "candidate_validation": [
            candidate.to_dict(repository=repository) for candidate in candidates
        ],
        "selected_candidate_artifacts": {
            "seed": selected.seed,
            "checkpoint": selected.checkpoint.relative_to(repository).as_posix(),
            "router_config": selected.config.to_dict(repository=repository),
            "router_weights": selected.weights.to_dict(repository=repository),
        },
        "selection": dict(selection),
        "execution": {
            "feature_batch_size": feature_batch_size,
            "feature_batch_size_role": (
                "execution-only batching; prompt features and routing semantics are unchanged"
            ),
        },
        "checks": {
            "all_dataset_gates_passed": True,
            "all_candidate_preblind_gates_passed": True,
            "all_candidate_hashes_recorded_before_blind_access": True,
            "selected_seed": selected_seed,
            "selected_by_freshly_recomputed_validation_loss_only": True,
            "validation_metrics_freshly_recomputed_on_cpu": True,
            "sidecar_validation_metrics_used_for_gates_or_selection": False,
            "test_metrics_used_for_seed_selection": False,
            "runtime_sources_verified": True,
            "model_runtime_verified": True,
            "blind_probe_opened": False,
            "registration_changed": False,
        },
    }
    receipt["payload_sha256"] = sha256_text(canonical_json(receipt))
    return receipt


def _preblind_failure_report(
    *,
    error: PreblindMetricGateFailure,
    policy_snapshot: ArtifactSnapshot,
    dataset_snapshots: Mapping[str, ArtifactSnapshot],
    runtime_source_snapshots: Mapping[str, ArtifactSnapshot],
    model_evidence: ModelRuntimeEvidence,
    repository: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "evaluation_kind": "router_v6_preblind_metric_gate_failure",
        "policy": policy_snapshot.to_dict(repository=repository),
        "dataset": {
            label: value.to_dict(repository=repository)
            for label, value in sorted(dataset_snapshots.items())
        },
        "runtime_source_artifacts": {
            label: value.to_dict(repository=repository)
            for label, value in sorted(runtime_source_snapshots.items())
        },
        "model_runtime": model_evidence.to_dict(repository=repository),
        "preblind": {
            "all_seed_gates_passed": False,
            "failure": str(error),
            "candidates": [
                candidate.to_dict(repository=repository) for candidate in error.candidates
            ],
            "artifacts": {
                label: value.to_dict(repository=repository)
                for label, value in sorted(error.snapshots.items())
            },
        },
        "probe": {
            "opened": False,
            "parsed": False,
            "metrics_computed": False,
        },
        "decision": "rejected_preblind_metric_gate_failed_probe_remains_sealed",
        "registration_changed": False,
        "auto_registration_performed": False,
    }
    report["payload_sha256"] = sha256_text(canonical_json(report))
    return report


def _runtime(policy: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(policy["runtime"])
    if "threshold" in result and "confidence_threshold" not in result:
        result["confidence_threshold"] = result.pop("threshold")
    return result


def _prepare_routers_before_probe(
    *,
    baseline_checkpoint: Path,
    candidate_loaded: tuple[Any, Mapping[str, Any]],
    model_evidence: ModelRuntimeEvidence,
) -> tuple[Any, Mapping[str, Any], Any, Mapping[str, Any]]:
    baseline_router, baseline_sidecar = ROUTER_PROBE.load_checkpoint(baseline_checkpoint)
    candidate_router, candidate_sidecar = candidate_loaded
    if tuple(
        route for route in candidate_sidecar["route_ids"] if route != TASKMASTER_ROUTE_ID
    ) != tuple(baseline_sidecar["route_ids"]):
        raise ValueError("candidate and baseline legacy route axes differ")
    candidate_contract = ROUTER_PROBE._feature_contract(candidate_sidecar, None)
    baseline_contract = ROUTER_PROBE._feature_contract(baseline_sidecar, None)
    if candidate_contract != baseline_contract:
        raise ValueError("candidate and baseline feature contracts differ")
    validate_model_sidecar(
        baseline_sidecar,
        evidence=model_evidence,
        label="promoted router v4 sidecar",
    )
    validate_model_sidecar(
        candidate_sidecar,
        evidence=model_evidence,
        label="selected router v6 sidecar",
    )
    return baseline_router, baseline_sidecar, candidate_router, candidate_sidecar


def run_registration_evaluation(
    *,
    policy_path: Path = DEFAULT_POLICY,
    model_path: Path = DEFAULT_MODEL,
    output_path: Path = DEFAULT_OUTPUT,
    preblind_receipt_path: Path | None = None,
    repository: Path = WORKSPACE_ROOT,
    feature_batch_size: int = 64,
    probe_loader: Callable[..., tuple[tuple[Any, ...], ArtifactSnapshot]] = open_frozen_probe,
) -> dict[str, Any]:
    """Run the sealed evaluation.  The caller must handle the human acknowledgement."""

    repository = repository.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"one-shot evidence already exists: {output_path}")
    policy, policy_snapshot = load_policy(policy_path, repository=repository)
    frozen_output = _repository_path(
        repository,
        policy["one_shot_evaluation"]["report_path"],
        label="one-shot output path",
    )
    if output_path != frozen_output:
        raise ValueError("output_path differs from the frozen one-shot report path")
    frozen_receipt = _repository_path(
        repository,
        policy["preblind_receipt"]["path"],
        label="preblind receipt path",
    )
    requested_receipt = (
        frozen_receipt
        if preblind_receipt_path is None
        else preblind_receipt_path.expanduser().resolve()
    )
    if requested_receipt != frozen_receipt:
        raise ValueError("preblind_receipt_path differs from the frozen policy")
    if frozen_receipt.exists():
        raise FileExistsError(f"preblind one-shot receipt already exists: {frozen_receipt}")
    if feature_batch_size != policy["training"]["feature_batch_size"]:
        raise ValueError("feature_batch_size differs from the frozen execution contract")

    runtime_source_snapshots = validate_runtime_sources(policy, repository=repository)
    model_evidence = validate_model_runtime(
        policy,
        model_path=model_path.expanduser().resolve(),
        repository=repository,
    )
    dataset_snapshots, dataset_report = validate_dataset_contract(policy, repository=repository)
    embedding, embedding_metadata = load_input_embedding(
        model_evidence.directory,
        tensor_name=model_evidence.embedding_tensor_name,
    )
    validate_loaded_embedding(embedding_metadata, evidence=model_evidence)
    tokenizer = load_router_tokenizer(model_evidence.directory)
    validation_features, validation_labels, feature_contract = prepare_validation_features(
        policy,
        validation_path=dataset_snapshots["validation"].path,
        embedding=embedding,
        tokenizer=tokenizer,
    )
    try:
        candidates, checkpoint_snapshots, selection, loaded_candidates = validate_candidates(
            policy,
            dataset_snapshots=dataset_snapshots,
            validation_features=validation_features,
            validation_labels=validation_labels,
            feature_contract=feature_contract,
            model_evidence=model_evidence,
            repository=repository,
        )
    except PreblindMetricGateFailure as exc:
        failure = _preblind_failure_report(
            error=exc,
            policy_snapshot=policy_snapshot,
            dataset_snapshots=dataset_snapshots,
            runtime_source_snapshots=runtime_source_snapshots,
            model_evidence=model_evidence,
            repository=repository,
        )
        write_json_exclusive(output_path, failure)
        return failure
    selected = next(item for item in candidates if item.seed == selection["selected_seed"])
    baseline_path, _template, _report, _descriptor, _splits = _policy_training_paths(
        policy, repository=repository
    )

    # Materialize and validate every non-blind dependency before opening the probe.
    prepared = _prepare_routers_before_probe(
        baseline_checkpoint=baseline_path,
        candidate_loaded=loaded_candidates[selected.seed],
        model_evidence=model_evidence,
    )
    baseline_router, baseline_sidecar, candidate_router, candidate_sidecar = prepared
    assert_unchanged(policy_snapshot, label="router v6 policy")
    all_preblind_snapshots = {
        **runtime_source_snapshots,
        **model_evidence.snapshots,
        **dataset_snapshots,
        **checkpoint_snapshots,
    }
    for label, value in all_preblind_snapshots.items():
        assert_unchanged(value, label=label)

    receipt = build_preblind_receipt(
        policy_snapshot=policy_snapshot,
        dataset_snapshots=dataset_snapshots,
        runtime_source_snapshots=runtime_source_snapshots,
        runtime_source_contract=policy["runtime_source_artifacts"],
        model_evidence=model_evidence,
        checkpoint_snapshots=checkpoint_snapshots,
        candidates=candidates,
        selection=selection,
        feature_batch_size=feature_batch_size,
        repository=repository,
    )
    receipt_snapshot = write_json_exclusive(frozen_receipt, receipt)
    assert_unchanged(receipt_snapshot, label="router v6 preblind receipt")

    # This is the first semantic access to the one-shot probe.
    items, probe_snapshot = probe_loader(policy, repository=repository)
    candidate_route_ids = tuple(candidate_sidecar["route_ids"])
    unknown_labels = {
        route
        for item in items
        for route in item.acceptable_routes
        if route not in candidate_route_ids
    }
    if unknown_labels:
        raise ValueError(f"probe labels are absent from candidate: {sorted(unknown_labels)!r}")
    contract = ROUTER_PROBE._feature_contract(candidate_sidecar, None)
    examples = [RouterExample(item.prompt, candidate_route_ids[0]) for item in items]
    features, _labels = extract_prompt_features(
        examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=candidate_route_ids,
        contract=contract,
        batch_size=feature_batch_size,
    )
    del embedding

    runtime = _runtime(policy)
    baseline_metrics, baseline_rows = ROUTER_PROBE.evaluate(
        baseline_router,
        baseline_sidecar,
        items,
        features,
        temperature=runtime["temperature"],
        confidence_threshold=runtime["confidence_threshold"],
        null_bias=runtime["null_bias"],
        top_k=runtime["top_k"],
        enabled_experts=set(baseline_sidecar["route_ids"]).difference({NULL_ROUTE_ID}),
    )
    candidate_metrics, candidate_rows = ROUTER_PROBE.evaluate(
        candidate_router,
        candidate_sidecar,
        items,
        features,
        temperature=runtime["temperature"],
        confidence_threshold=runtime["confidence_threshold"],
        null_bias=runtime["null_bias"],
        top_k=runtime["top_k"],
        enabled_experts=set(candidate_route_ids).difference({NULL_ROUTE_ID}),
    )
    one_shot = evaluate_one_shot_gates(
        items=items,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
    )

    assert_unchanged(policy_snapshot, label="router v6 policy")
    assert_unchanged(probe_snapshot, label="router v6 blind probe")
    assert_unchanged(receipt_snapshot, label="router v6 preblind receipt")
    for label, value in all_preblind_snapshots.items():
        assert_unchanged(value, label=label)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "evaluation_kind": "sealed_one_shot_router_v6_registration",
        "policy": policy_snapshot.to_dict(repository=repository),
        "probe": {
            **probe_snapshot.to_dict(repository=repository),
            "purpose": PROBE_PURPOSE,
            "rows": len(items),
            "semantic_access_count_by_this_evaluator": 1,
        },
        "dataset": {
            "report_schema": dataset_report.get("schema_version"),
            "artifacts": {
                label: value.to_dict(repository=repository)
                for label, value in sorted(dataset_snapshots.items())
            },
            "split_disjointness_verified": True,
        },
        "runtime_source_artifacts": {
            label: value.to_dict(repository=repository)
            for label, value in sorted(runtime_source_snapshots.items())
        },
        "preblind": {
            "all_seed_gates_passed": True,
            "receipt": receipt_snapshot.to_dict(repository=repository),
            "candidates": [item.to_dict(repository=repository) for item in candidates],
            "selection": selection,
        },
        "model": {
            **model_evidence.to_dict(repository=repository),
            "full_causal_model_loaded": False,
        },
        "runtime": runtime,
        "baseline": {
            "checkpoint": baseline_path.relative_to(repository).as_posix(),
            "metrics": baseline_metrics,
            "examples": _redact_prompt_rows(baseline_rows),
        },
        "candidate": {
            "checkpoint": selected.checkpoint.relative_to(repository).as_posix(),
            "seed": selected.seed,
            "metrics": candidate_metrics,
            "examples": _redact_prompt_rows(candidate_rows),
        },
        "one_shot": one_shot,
        "decision": (
            "all_gates_passed_registration_requires_separate_explicit_change"
            if one_shot["all_gates_passed"]
            else "rejected_one_or_more_frozen_registration_gates_failed"
        ),
        "registration_changed": False,
        "auto_registration_performed": False,
    }
    report["payload_sha256"] = sha256_text(canonical_json(report))
    written = write_json_exclusive(output_path, report)
    report["evidence_file_sha256"] = written.sha256
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen one-shot router v6 registration probe. Passing writes evidence only; "
            "it never changes registration."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument(
        "--acknowledge-one-shot",
        action="store_true",
        help="required acknowledgement that this command permanently exposes the blind probe",
    )
    args = parser.parse_args(argv)
    if not args.acknowledge_one_shot:
        parser.error("--acknowledge-one-shot is required before the blind probe can be opened")
    if args.feature_batch_size < 1:
        parser.error("--feature-batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_registration_evaluation(
        policy_path=args.policy,
        model_path=args.model,
        output_path=args.output,
        feature_batch_size=args.feature_batch_size,
    )
    summary = report.get("one_shot", report.get("preblind"))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Decision: {report['decision']}", flush=True)
    print(f"Wrote append-only evidence: {args.output}", flush=True)
    print("Registration was not changed.", flush=True)


if __name__ == "__main__":
    main()
