"""Sealed one-shot evaluation for the hierarchical Taskmaster router v7.

Router v7 is deliberately incremental: a binary admission gate either selects the
Taskmaster expert or delegates the original prompt to the byte-pinned promoted v4
router.  Candidate selection uses only freshly recomputed metrics from the frozen
v7 validation split.  No test split is accepted by this evaluator.

The durable preblind receipt is created and verified before the first blind-probe
open.  This program never registers, copies, aliases, or changes any runtime default.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
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

from molor.experimental.autorouter import NULL_ROUTE_ID
from molor.training.router import (
    ROUTER_CHECKPOINT_SCHEMA,
    RouterExample,
    RouterFeatureContract,
    extract_prompt_features,
    load_input_embedding,
    load_router_jsonl,
    load_router_tokenizer,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = EXPERIMENT_ROOT / "router-v7-evaluation-policy.json"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "Qwen3-1.7B"
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "runs"
    / "evaluation"
    / "router-v7-taskmaster-hierarchical-registration-one-shot.json"
)
DEFAULT_PREBLIND_RECEIPT = (
    WORKSPACE_ROOT / "runs" / "router-v7" / "taskmaster-hierarchical-preblind-receipt.json"
)

POLICY_SCHEMA = "molor.router-v7-hierarchical-evaluation-policy.v1"
REPORT_SCHEMA = "molor.router-v7-hierarchical-registration-report.v1"
PREBLIND_SCHEMA = "molor.router-v7-hierarchical-preblind-receipt.v1"
PROBE_PURPOSE = "blind_intent_probe_never_training"
TASKMASTER_ROUTE_ID = "taskmaster-coffee-api-v2-full-dialog-r16"
GATE_ROUTE_IDS = (TASKMASTER_ROUTE_ID, NULL_ROUTE_ID)
V4_ROUTE_IDS = (
    "embedded-knowledge-v1",
    "ocb-can-verified-v1",
    "software-architecture-v1",
    "verified-embedded-code-v1",
    NULL_ROUTE_ID,
)
EXPECTED_SEEDS = (83, 8419, 27183)
EXPECTED_VALIDATION_SUPPORT = {TASKMASTER_ROUTE_ID: 130, NULL_ROUTE_ID: 650}
EXPECTED_TRAIN_SUPPORT = {TASKMASTER_ROUTE_ID: 936, NULL_ROUTE_ID: 4680}
EXPECTED_V4_RUNTIME = {
    "temperature": 1.0,
    "confidence_threshold": 0.30,
    "null_bias": 0.0,
    "top_k": 1,
    "enabled_experts": list(V4_ROUTE_IDS[:-1]),
}
EXPECTED_PREBLIND_GATES = {
    "finite_validation_loss_required": True,
    "validation_accuracy_min": 0.995,
    "taskmaster_precision_min": 0.99,
    "taskmaster_recall_min": 0.99,
    "false_positives_on_650_validation_negatives_max": 0,
    "all_three_seeds_must_pass": True,
}
EXPECTED_ONE_SHOT_GATES = {
    "overall_exact_accuracy_min": 0.75,
    "overall_acceptable_accuracy_min": 0.75,
    "exact_correct_gain_over_v4_min": 8,
    "acceptable_correct_gain_over_v4_min": 8,
    "taskmaster_exact_correct_min": 10,
    "false_taskmaster_selections_on_non_taskmaster_rows_max": 1,
    "max_exact_correct_drop_per_protected_category_vs_v4": 1,
    "max_acceptable_correct_drop_per_protected_category_vs_v4": 1,
    "all_gates_must_pass": True,
}
EXPECTED_PROTECTED_CATEGORIES = (
    NULL_ROUTE_ID,
    "embedded-knowledge-v1",
    "mixed",
    "ocb-can-verified-v1",
    "software-architecture-v1",
    "verified-embedded-code-v1",
)
EXPECTED_RUNTIME_SCRIPT_PATHS = {
    "dataset_builder": "experiments/first-real-experts/build_router_v7_taskmaster_gate_dataset.py",
    "training_launcher": "experiments/first-real-experts/train_router_v7_incremental_gate.py",
    "registration_evaluator": (
        "experiments/first-real-experts/evaluate_router_v7_hierarchical_registration.py"
    ),
}
EXPECTED_SHARED_RUNTIME_PATHS = ("experiments/first-real-experts/evaluate_router.py",)
MODEL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
SOURCE_TREE_HASH_POLICY = "sha256-canonical-repository-relative-path-size-content-v1"
DATASET_ID = "router-bootstrap-v7-taskmaster-gate"
DATASET_SPLITS = ("train", "validation")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _load_probe_runtime_module():
    module_name = "molor_router_v7_shared_probe_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = EXPERIMENT_ROOT / "evaluate_router.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import boundary
        raise RuntimeError(f"cannot load shared router evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROUTER_PROBE = _load_probe_runtime_module()


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    path: Path
    sha256: str
    size_bytes: int

    def to_dict(self, *, repository: Path = WORKSPACE_ROOT) -> dict[str, Any]:
        try:
            display = self.path.relative_to(repository).as_posix()
        except ValueError:
            display = str(self.path)
        return {"path": display, "sha256": self.sha256, "size_bytes": self.size_bytes}


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
            "model:config": self.config,
            "model:index": self.index,
            "model:embedding": self.embedding_artifact,
            **{
                f"model:tokenizer:{name}": value for name, value in self.tokenizer_artifacts.items()
            },
        }

    def to_dict(self, *, repository: Path) -> dict[str, Any]:
        return {
            "path": self.directory.relative_to(repository).as_posix(),
            "config": self.config.to_dict(repository=repository),
            "safetensors_index": self.index.to_dict(repository=repository),
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
    validation_false_positives: int
    taskmaster_support: int
    non_taskmaster_support: int
    gates: tuple[Mapping[str, Any], ...]
    passed: bool

    def to_dict(self, *, repository: Path) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "checkpoint": self.checkpoint.relative_to(repository).as_posix(),
            "router_config": self.config.to_dict(repository=repository),
            "router_weights": self.weights.to_dict(repository=repository),
            "metric_source": "fresh_cpu_recomputation_from_frozen_v7_validation_only",
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "taskmaster_precision": self.taskmaster_precision,
            "taskmaster_recall": self.taskmaster_recall,
            "validation_false_positives": self.validation_false_positives,
            "taskmaster_support": self.taskmaster_support,
            "non_taskmaster_support": self.non_taskmaster_support,
            "gates": list(self.gates),
            "passed": self.passed,
            "sidecar_metrics_role": "diagnostic_only_never_used_for_gates_or_selection",
        }


class PreblindMetricGateFailure(ValueError):
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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


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
    if path.is_symlink() or resolved.is_symlink():
        raise ValueError(f"artifact may not be a symbolic link: {path}")
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
    resolved = (repository / raw_path).resolve()
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    return resolved


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _gate(name: str, *, actual: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {"name": name, "actual": actual, "required": required, "passed": bool(passed)}


def _descriptor_snapshot(
    descriptor: Any,
    *,
    repository: Path,
    label: str,
    expected_path: Path | None = None,
) -> ArtifactSnapshot:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} descriptor must be an object")
    allowed = {"path", "sha256", "size_bytes", "logical_name", "rows"}
    if set(descriptor).difference(allowed):
        raise ValueError(f"{label} descriptor has unknown fields")
    path = _repository_path(repository, descriptor.get("path"), label=f"{label} path")
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} path differs from the frozen location")
    digest = descriptor.get("sha256")
    size = descriptor.get("size_bytes")
    if _SHA256.fullmatch(str(digest)) is None:
        raise ValueError(f"{label} sha256 is invalid")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{label} size is invalid")
    value = snapshot(path)
    if value.sha256 != digest or value.size_bytes != size:
        raise ValueError(f"{label} differs from the frozen identity")
    return value


def _artifact_binding_snapshot(
    binding: Any,
    *,
    repository: Path,
    label: str,
    expected_path: Path,
) -> ArtifactSnapshot:
    if not isinstance(binding, Mapping) or binding.get("status") != "frozen_before_training":
        raise ValueError(f"{label} binding is not frozen")
    descriptor = binding.get("artifact")
    return _descriptor_snapshot(
        descriptor,
        repository=repository,
        label=label,
        expected_path=expected_path,
    )


def _candidate_binding(policy: Mapping[str, Any], seed: int, *, repository: Path) -> Path:
    bindings = policy["artifact_bindings"]
    value = bindings.get(f"gate_candidate_seed_{seed}")
    if not isinstance(value, Mapping) or value.get("seed") != seed:
        raise ValueError(f"candidate seed {seed} binding is missing")
    if (
        value.get("status")
        != "path_frozen_before_training_hash_pending_generated_artifact_preblind"
    ):
        raise ValueError(f"candidate seed {seed} path is not preregistered")
    if set(value) != {"seed", "checkpoint_path", "status"}:
        raise ValueError("candidate bindings may freeze paths only; hashes belong in the receipt")
    path = _repository_path(
        repository,
        value.get("checkpoint_path"),
        label=f"candidate seed {seed} checkpoint",
    )
    expected = (
        repository
        / "checkpoints"
        / "qwen3-1.7b-candidate-routers-quarantine"
        / f"router-v7-taskmaster-gate-seed{seed}"
    ).resolve()
    if path != expected:
        raise ValueError(f"candidate seed {seed} path differs from the quarantine contract")
    return path


def select_candidate(candidates: Sequence[CandidateEvidence]) -> CandidateEvidence:
    if len(candidates) != 3 or {item.seed for item in candidates} != set(EXPECTED_SEEDS):
        raise ValueError("candidate seed set differs from the frozen contract")
    if not all(item.passed for item in candidates):
        raise ValueError("all three candidate seeds must pass before selection")
    return min(candidates, key=lambda item: (item.validation_loss, item.seed))


def require_train_validation_only_bindings(bindings: Mapping[str, Any]) -> None:
    """Reject a policy that introduces any v7 test-dataset binding."""

    forbidden = {key for key in bindings if key.casefold().startswith("dataset_test")}
    if forbidden:
        raise ValueError("router v7 policy must not contain a test-dataset binding")
    expected_dataset_keys = {"dataset_report", "dataset_train", "dataset_validation"}
    actual_dataset_keys = {
        key for key in bindings if key.startswith("dataset_") and key != "dataset_builder"
    }
    if actual_dataset_keys != expected_dataset_keys:
        raise ValueError("router v7 binds exactly report, train, and validation artifacts")


def load_policy(
    path: Path,
    *,
    repository: Path = WORKSPACE_ROOT,
) -> tuple[dict[str, Any], ArtifactSnapshot]:
    """Validate the completed policy without touching the sealed probe path."""

    policy_snapshot = snapshot(path)
    policy = _read_json_object(policy_snapshot.path, label="router v7 evaluation policy")
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported router v7 evaluation policy schema")
    if policy.get("status") != "frozen_before_candidate_training_and_blind_evaluation":
        raise ValueError("router v7 policy is not completely frozen")
    lifecycle = policy.get("policy_lifecycle")
    required_lifecycle = {
        "semantic_contract_frozen": True,
        "artifact_bindings_complete": True,
        "training_authorized": True,
        "one_shot_evaluation_authorized": True,
    }
    if not isinstance(lifecycle, Mapping) or any(
        lifecycle.get(key) is not value for key, value in required_lifecycle.items()
    ):
        raise ValueError("router v7 policy lifecycle does not authorize the frozen run")

    route = policy.get("route_contract")
    if not isinstance(route, Mapping):
        raise ValueError("router v7 route contract is missing")
    if (
        route.get("taskmaster_route_id") != TASKMASTER_ROUTE_ID
        or route.get("delegate_router_id") != "qwen3-1.7b-corpus-router-v4-mlp-controlled"
        or route.get("architecture") != "binary_taskmaster_gate_then_exact_v4_delegate"
        or route.get("binary_labels") != ["delegate_v4", TASKMASTER_ROUTE_ID]
        or route.get("gate_score") != "sigmoid probability for the Taskmaster-positive label"
        or _number(route.get("gate_threshold"), label="gate threshold") != 0.5
        or route.get("threshold_comparison")
        != "taskmaster_probability_greater_than_or_equal_to_threshold"
        or route.get("frozen_v4_runtime") != EXPECTED_V4_RUNTIME
    ):
        raise ValueError("router v7 hierarchy or fixed routing semantics changed")

    training = policy.get("training_and_selection")
    if not isinstance(training, Mapping):
        raise ValueError("router v7 training and selection contract is missing")
    if (
        training.get("candidate_seeds") != list(EXPECTED_SEEDS)
        or training.get("seed_is_only_permitted_candidate_difference") is not True
        or training.get("device") != "cpu"
        or _number(
            training.get("gate_threshold_during_validation"),
            label="validation threshold",
        )
        != 0.5
        or training.get("fresh_validation_required") is not True
        or training.get("validation_support")
        != {"taskmaster_positive": 130, "non_taskmaster_negative": 650, "total": 780}
        or training.get("candidate_preblind_gates") != EXPECTED_PREBLIND_GATES
        or training.get("selection_order")
        != [
            "lowest freshly recomputed validation loss",
            "lowest numeric seed as the sole exact-loss tie-break",
        ]
        or training.get("sidecar_metrics_are_diagnostic_only") is not True
        or training.get("dataset_has_test_split") is not False
        or training.get("test_metrics_used_for_training_or_selection") is not False
        or training.get("blind_probe_used_for_training_selection_or_thresholding") is not False
        or training.get("post_probe_retraining_reseeding_or_threshold_tuning") is not False
    ):
        raise ValueError("router v7 validation-only seed-selection contract changed")

    probe = policy.get("blind_probe")
    if not isinstance(probe, Mapping):
        raise ValueError("router v7 blind-probe descriptor is missing")
    if (
        probe.get("path") != "experiments/first-real-experts/router-blind-probe-v7.jsonl"
        or _SHA256.fullmatch(str(probe.get("sha256"))) is None
        or not isinstance(probe.get("size_bytes"), int)
        or probe.get("rows") != 84
        or probe.get("purpose") != PROBE_PURPOSE
        or probe.get("category_counts")
        != {
            NULL_ROUTE_ID: 12,
            "embedded-knowledge-v1": 12,
            "mixed": 12,
            "ocb-can-verified-v1": 12,
            "software-architecture-v1": 12,
            TASKMASTER_ROUTE_ID: 12,
            "verified-embedded-code-v1": 12,
        }
    ):
        raise ValueError("router v7 blind-probe identity or strata changed")
    # Containment only: no stat, hash, line count, or parse before the receipt.
    _repository_path(repository, probe["path"], label="blind probe path")

    one_shot = policy.get("one_shot_evaluation")
    if not isinstance(one_shot, Mapping):
        raise ValueError("router v7 one-shot contract is missing")
    expected_report = (
        repository
        / "runs"
        / "evaluation"
        / "router-v7-taskmaster-hierarchical-registration-one-shot.json"
    ).resolve()
    if (
        one_shot.get("complete_rows_required") != 84
        or one_shot.get("finite_scores_required") is not True
        or one_shot.get("taskmaster_rows") != 12
        or one_shot.get("non_taskmaster_rows") != 72
        or tuple(one_shot.get("protected_categories", ())) != EXPECTED_PROTECTED_CATEGORIES
        or one_shot.get("gates") != EXPECTED_ONE_SHOT_GATES
        or _repository_path(
            repository,
            one_shot.get("report_path"),
            label="one-shot report path",
        )
        != expected_report
    ):
        raise ValueError("router v7 one-shot gates or output path changed")

    evidence = policy.get("evidence_discipline")
    if not isinstance(evidence, Mapping):
        raise ValueError("router v7 evidence discipline is missing")
    expected_receipt = (
        repository / "runs" / "router-v7" / "taskmaster-hierarchical-preblind-receipt.json"
    ).resolve()
    if (
        _repository_path(
            repository,
            evidence.get("preblind_receipt_path"),
            label="preblind receipt path",
        )
        != expected_receipt
        or evidence.get("receipt_written_before_probe_parse_by_evaluator") is not True
        or evidence.get("receipt_write_mode") != "exclusive_non_overwriting_atomic"
        or evidence.get("report_write_mode") != "exclusive_non_overwriting_atomic"
        or evidence.get("one_shot_report_must_not_emit_probe_prompt_text") is not True
        or evidence.get("evaluator_may_register_copy_alias_or_modify_defaults") is not False
        or evidence.get("registration_requires_separate_explicit_audited_change") is not True
        or evidence.get("no_retry_after_probe_exposure") is not True
        or evidence.get("no_threshold_seed_data_or_checkpoint_tuning_after_probe_exposure")
        is not True
    ):
        raise ValueError("router v7 append-only evidence or no-mutation contract changed")

    forbidden = policy.get("forbidden_inputs")
    if (
        not isinstance(forbidden, Mapping)
        or forbidden.get("test_data_may_select_candidate") is not False
        or forbidden.get("probe_data_may_select_candidate") is not False
    ):
        raise ValueError("router v7 forbidden-input contract changed")

    bindings = policy.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("router v7 artifact bindings are missing")
    require_train_validation_only_bindings(bindings)
    expected_binding_keys = {
        "binding_state",
        "unresolved_value_marker",
        *EXPECTED_RUNTIME_SCRIPT_PATHS,
        "shared_runtime_sources",
        "molor_source_tree",
        "model_runtime",
        "tokenizer_runtime",
        "dataset_report",
        "dataset_train",
        "dataset_validation",
        "promoted_v4_delegate",
        "taskmaster_adapter",
        *(f"gate_candidate_seed_{seed}" for seed in EXPECTED_SEEDS),
    }
    if set(bindings) != expected_binding_keys:
        raise ValueError("router v7 artifact binding set is incomplete or contains extras")
    if (
        bindings.get("binding_state") != "complete_frozen_before_training"
        or bindings.get("unresolved_value_marker") is not None
    ):
        raise ValueError("router v7 artifact bindings are not complete")
    for seed in EXPECTED_SEEDS:
        _candidate_binding(policy, seed, repository=repository)
    assert_unchanged(policy_snapshot, label="router v7 policy")
    return policy, policy_snapshot


def validate_runtime_sources(
    policy: Mapping[str, Any],
    *,
    repository: Path = WORKSPACE_ROOT,
) -> dict[str, ArtifactSnapshot]:
    bindings = policy["artifact_bindings"]
    results: dict[str, ArtifactSnapshot] = {}
    for name, relative in EXPECTED_RUNTIME_SCRIPT_PATHS.items():
        results[f"script:{name}"] = _artifact_binding_snapshot(
            bindings[name],
            repository=repository,
            label=f"runtime script {name}",
            expected_path=repository / relative,
        )

    shared = bindings.get("shared_runtime_sources")
    if not isinstance(shared, Mapping) or shared.get("status") != "frozen_before_training":
        raise ValueError("shared runtime source binding is not frozen")
    artifacts = shared.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(EXPECTED_SHARED_RUNTIME_PATHS):
        raise ValueError("shared runtime source artifact set is incomplete")
    by_path: dict[Path, Mapping[str, Any]] = {}
    for descriptor in artifacts:
        if not isinstance(descriptor, Mapping):
            raise ValueError("shared runtime source descriptor must be an object")
        path = _repository_path(
            repository,
            descriptor.get("path"),
            label="shared runtime source path",
        )
        if path in by_path:
            raise ValueError("shared runtime source path is duplicated")
        by_path[path] = descriptor
    for relative in EXPECTED_SHARED_RUNTIME_PATHS:
        expected = (repository / relative).resolve()
        descriptor = by_path.get(expected)
        if descriptor is None:
            raise ValueError(f"shared runtime source is not pinned: {relative}")
        results[f"script:shared:{relative}"] = _descriptor_snapshot(
            descriptor,
            repository=repository,
            label=f"shared runtime source {relative}",
            expected_path=expected,
        )

    tree = bindings.get("molor_source_tree")
    if not isinstance(tree, Mapping) or tree.get("status") != "frozen_before_training":
        raise ValueError("molor source-tree binding is not frozen")
    if (
        tree.get("root") != "src/molor"
        or _SHA256.fullmatch(str(tree.get("content_fingerprint"))) is None
    ):
        raise ValueError("molor source-tree contract is invalid")
    source_root = (repository / "src" / "molor").resolve()
    files = sorted(
        (item.resolve() for item in source_root.rglob("*.py") if item.is_file()),
        key=lambda item: item.relative_to(repository).as_posix(),
    )
    if len(files) != tree.get("file_count"):
        raise ValueError("molor source-tree file count differs from frozen policy")
    entries: list[dict[str, Any]] = []
    for path in files:
        value = snapshot(path)
        relative = path.relative_to(repository).as_posix()
        results[f"molor:{relative}"] = value
        entries.append({"path": relative, "size_bytes": value.size_bytes, "sha256": value.sha256})
    if sha256_text(canonical_json(entries)) != tree["content_fingerprint"]:
        raise ValueError("molor source-tree content fingerprint differs from frozen policy")
    return results


def validate_model_runtime(
    policy: Mapping[str, Any],
    *,
    model_path: Path,
    repository: Path = WORKSPACE_ROOT,
) -> ModelRuntimeEvidence:
    bindings = policy["artifact_bindings"]
    contract = bindings.get("model_runtime")
    if not isinstance(contract, Mapping) or contract.get("status") != "frozen_before_training":
        raise ValueError("model runtime binding is not frozen")
    frozen_model = _repository_path(
        repository,
        contract.get("path"),
        label="model runtime path",
    )
    expected_model = (repository / "models" / "Qwen3-1.7B").resolve()
    if frozen_model != expected_model or model_path.expanduser().resolve() != frozen_model:
        raise ValueError("--model must equal the frozen repository-local Qwen3-1.7B")
    config = _descriptor_snapshot(
        contract.get("config"),
        repository=repository,
        label="model config",
        expected_path=frozen_model / "config.json",
    )
    index = _descriptor_snapshot(
        contract.get("safetensors_index"),
        repository=repository,
        label="model safetensors index",
        expected_path=frozen_model / "model.safetensors.index.json",
    )
    embedding = contract.get("embedding")
    if not isinstance(embedding, Mapping) or set(embedding) != {
        "tensor_name",
        "shape",
        "dtype",
        "artifact",
    }:
        raise ValueError("model embedding binding is incomplete")
    if (
        embedding.get("tensor_name") != "model.embed_tokens.weight"
        or embedding.get("shape") != [151936, 2048]
        or embedding.get("dtype") != "bfloat16"
    ):
        raise ValueError("model embedding tensor metadata differs from Qwen3-1.7B")
    embedding_artifact = _descriptor_snapshot(
        embedding.get("artifact"),
        repository=repository,
        label="model embedding shard",
        expected_path=frozen_model / "model-00001-of-00002.safetensors",
    )

    tokenizers = bindings.get("tokenizer_runtime")
    if not isinstance(tokenizers, Mapping) or tokenizers.get("status") != "frozen_before_training":
        raise ValueError("tokenizer runtime binding is not frozen")
    raw_artifacts = tokenizers.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(MODEL_TOKENIZER_FILES):
        raise ValueError("tokenizer runtime artifact set is incomplete")
    by_path: dict[Path, Mapping[str, Any]] = {}
    for descriptor in raw_artifacts:
        if not isinstance(descriptor, Mapping):
            raise ValueError("tokenizer descriptor must be an object")
        path = _repository_path(
            repository,
            descriptor.get("path"),
            label="tokenizer artifact path",
        )
        if path in by_path:
            raise ValueError("tokenizer artifact path is duplicated")
        by_path[path] = descriptor
    tokenizer_artifacts: dict[str, ArtifactSnapshot] = {}
    for name in MODEL_TOKENIZER_FILES:
        expected = (frozen_model / name).resolve()
        descriptor = by_path.get(expected)
        if descriptor is None:
            raise ValueError(f"tokenizer artifact is not pinned: {name}")
        tokenizer_artifacts[name] = _descriptor_snapshot(
            descriptor,
            repository=repository,
            label=f"tokenizer artifact {name}",
            expected_path=expected,
        )
    return ModelRuntimeEvidence(
        directory=frozen_model,
        config=config,
        index=index,
        embedding_artifact=embedding_artifact,
        embedding_tensor_name="model.embed_tokens.weight",
        embedding_shape=(151936, 2048),
        embedding_dtype="bfloat16",
        tokenizer_artifacts=tokenizer_artifacts,
    )


def _absolute_sidecar_path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise ValueError(f"{label} must be an absolute path recorded by training")
    return Path(raw).resolve()


def _require_sidecar_artifact(
    descriptor: Any,
    expected: ArtifactSnapshot,
    *,
    label: str,
) -> None:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} descriptor is missing")
    if (
        _absolute_sidecar_path(descriptor.get("path"), label=f"{label} path") != expected.path
        or descriptor.get("sha256") != expected.sha256
        or descriptor.get("size_bytes") != expected.size_bytes
    ):
        raise ValueError(f"{label} differs from the independently frozen artifact")


def validate_model_sidecar(
    sidecar: Mapping[str, Any],
    *,
    evidence: ModelRuntimeEvidence,
    label: str,
) -> None:
    model = sidecar.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"{label} has no model identity")
    if _absolute_sidecar_path(model.get("model_id"), label=f"{label} model") != evidence.directory:
        raise ValueError(f"{label} model directory differs from frozen runtime")
    _require_sidecar_artifact(model.get("config"), evidence.config, label=f"{label} model config")
    embedding = model.get("embedding")
    if not isinstance(embedding, Mapping):
        raise ValueError(f"{label} has no embedding identity")
    if (
        embedding.get("tensor_name") != evidence.embedding_tensor_name
        or embedding.get("shape") != list(evidence.embedding_shape)
        or embedding.get("dtype") != evidence.embedding_dtype
    ):
        raise ValueError(f"{label} embedding metadata differs from frozen runtime")
    _require_sidecar_artifact(
        embedding.get("artifact"),
        evidence.embedding_artifact,
        label=f"{label} embedding shard",
    )
    raw_tokenizers = model.get("tokenizer_artifacts")
    if not isinstance(raw_tokenizers, list):
        raise ValueError(f"{label} tokenizer artifact list is missing")
    by_name = {
        item.get("logical_name"): item for item in raw_tokenizers if isinstance(item, Mapping)
    }
    if set(by_name) != set(evidence.tokenizer_artifacts):
        raise ValueError(f"{label} tokenizer artifact set differs from frozen runtime")
    for name, expected in evidence.tokenizer_artifacts.items():
        _require_sidecar_artifact(
            by_name[name],
            expected,
            label=f"{label} tokenizer {name}",
        )


def validate_loaded_embedding(
    metadata: Mapping[str, Any],
    *,
    evidence: ModelRuntimeEvidence,
) -> None:
    if (
        metadata.get("tensor_name") != evidence.embedding_tensor_name
        or metadata.get("shape") != list(evidence.embedding_shape)
        or metadata.get("dtype") != evidence.embedding_dtype
    ):
        raise ValueError("loaded embedding metadata differs from frozen model runtime")
    _require_sidecar_artifact(
        metadata.get("artifact"),
        evidence.embedding_artifact,
        label="loaded embedding shard",
    )


def _binding_dataset_snapshot(
    binding: Any,
    *,
    repository: Path,
    label: str,
    expected_path: Path,
    expected_rows: int | None = None,
) -> ArtifactSnapshot:
    if not isinstance(binding, Mapping) or binding.get("status") != "frozen_before_training":
        raise ValueError(f"{label} binding is not frozen")
    if expected_rows is not None and binding.get("rows") != expected_rows:
        raise ValueError(f"{label} row count differs from the frozen contract")
    value = _descriptor_snapshot(
        binding.get("artifact"),
        repository=repository,
        label=label,
        expected_path=expected_path,
    )
    if expected_rows is not None:
        with value.path.open("rb") as stream:
            actual_rows = sum(bool(line.strip()) for line in stream)
        if actual_rows != expected_rows:
            raise ValueError(f"{label} physical row count differs from the frozen contract")
    return value


def validate_dataset_contract(
    policy: Mapping[str, Any],
    *,
    repository: Path = WORKSPACE_ROOT,
) -> tuple[dict[str, ArtifactSnapshot], dict[str, Any]]:
    """Hash the train/validation-only dataset without accepting a test artifact."""

    bindings = policy["artifact_bindings"]
    require_train_validation_only_bindings(bindings)
    data_root = repository / "experiments" / "first-real-experts" / "data"
    snapshots = {
        "report": _binding_dataset_snapshot(
            bindings["dataset_report"],
            repository=repository,
            label="v7 dataset report",
            expected_path=data_root / f"{DATASET_ID}.report.json",
        ),
        "train": _binding_dataset_snapshot(
            bindings["dataset_train"],
            repository=repository,
            label="v7 train dataset",
            expected_path=data_root / f"{DATASET_ID}.train.jsonl",
            expected_rows=5616,
        ),
        "validation": _binding_dataset_snapshot(
            bindings["dataset_validation"],
            repository=repository,
            label="v7 validation dataset",
            expected_path=data_root / f"{DATASET_ID}.validation.jsonl",
            expected_rows=780,
        ),
    }
    report = _read_json_object(snapshots["report"].path, label="v7 dataset report")
    if (
        report.get("schema_version") != "molor.router-admission-gate-dataset.v1"
        or report.get("dataset_id") != DATASET_ID
        or report.get("route_ids") != list(GATE_ROUTE_IDS)
        or report.get("mapped_balance")
        != {
            "train": EXPECTED_TRAIN_SUPPORT,
            "validation": EXPECTED_VALIDATION_SUPPORT,
        }
        or report.get("normalized_prompt_overlap") != {"train__validation": 0}
        or report.get("source_row_order_preserved") is not True
        or report.get("source_prompts_preserved_byte_for_byte") is not True
        or report.get("router_test_split_read_by_builder") is not False
        or report.get("taskmaster_test_split_read_by_builder") is not False
        or report.get("prediction_or_evaluation_artifacts_read_by_builder") is not False
        or report.get("blind_probe_read_by_builder") is not False
    ):
        raise ValueError("v7 dataset report violates the frozen train/validation contract")
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(DATASET_SPLITS):
        raise ValueError("v7 dataset report must describe train and validation only")
    for split in DATASET_SPLITS:
        descriptor = outputs[split]
        expected = snapshots[split]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"v7 dataset report {split} descriptor is missing")
        path = Path(str(descriptor.get("path"))).resolve()
        if (
            path != expected.path
            or descriptor.get("sha256") != expected.sha256
            or descriptor.get("size_bytes") != expected.size_bytes
            or descriptor.get("rows") != (5616 if split == "train" else 780)
        ):
            raise ValueError(f"v7 dataset report {split} identity differs from policy")
    for label, value in snapshots.items():
        assert_unchanged(value, label=f"dataset {label}")
    return snapshots, report


def _validate_checkpoint_descriptor(
    sidecar: Mapping[str, Any],
    *,
    weights: ArtifactSnapshot,
    label: str,
) -> None:
    _require_sidecar_artifact(sidecar.get("checkpoint"), weights, label=f"{label} weights")


def validate_taskmaster_adapter(
    policy: Mapping[str, Any],
    *,
    repository: Path = WORKSPACE_ROOT,
) -> tuple[Path, ArtifactSnapshot, ArtifactSnapshot]:
    binding = policy["artifact_bindings"].get("taskmaster_adapter")
    if not isinstance(binding, Mapping) or binding.get("status") != "frozen_before_training":
        raise ValueError("Taskmaster adapter binding is not frozen")
    directory = _repository_path(
        repository,
        binding.get("checkpoint_path"),
        label="Taskmaster adapter checkpoint",
    )
    expected = (
        repository / "checkpoints" / "qwen3-1.7b-candidate-experts-quarantine" / TASKMASTER_ROUTE_ID
    ).resolve()
    if directory != expected:
        raise ValueError("Taskmaster adapter path differs from the quarantine contract")
    config = _descriptor_snapshot(
        binding.get("adapter_config"),
        repository=repository,
        label="Taskmaster adapter config",
        expected_path=directory / "adapter_config.json",
    )
    weights = _descriptor_snapshot(
        binding.get("adapter_weights"),
        repository=repository,
        label="Taskmaster adapter weights",
        expected_path=directory / "adapter_model.safetensors",
    )
    return directory, config, weights


def validate_v4_delegate(
    policy: Mapping[str, Any],
    *,
    model_evidence: ModelRuntimeEvidence,
    repository: Path = WORKSPACE_ROOT,
) -> tuple[Any, Mapping[str, Any], dict[str, ArtifactSnapshot], Path]:
    binding = policy["artifact_bindings"].get("promoted_v4_delegate")
    if not isinstance(binding, Mapping) or binding.get("status") != "frozen_before_training":
        raise ValueError("promoted v4 delegate binding is not frozen")
    checkpoint = _repository_path(
        repository,
        binding.get("checkpoint_path"),
        label="promoted v4 checkpoint",
    )
    expected = (repository / "checkpoints" / "qwen3-1.7b-corpus-router-v4-mlp-controlled").resolve()
    if checkpoint != expected:
        raise ValueError("delegate checkpoint is not the promoted v4 router")
    if binding.get("runtime_policy") != EXPECTED_V4_RUNTIME:
        raise ValueError("delegate runtime differs from the frozen v4 policy")
    config = _descriptor_snapshot(
        binding.get("router_config"),
        repository=repository,
        label="promoted v4 router config",
        expected_path=checkpoint / "router_config.json",
    )
    weights = _descriptor_snapshot(
        binding.get("router_weights"),
        repository=repository,
        label="promoted v4 router weights",
        expected_path=checkpoint / "router.safetensors",
    )
    router, sidecar = ROUTER_PROBE.load_checkpoint(checkpoint)
    if tuple(sidecar.get("route_ids", ())) != V4_ROUTE_IDS:
        raise ValueError("promoted v4 route axis differs from the frozen delegate")
    validate_model_sidecar(sidecar, evidence=model_evidence, label="promoted v4 router")
    _validate_checkpoint_descriptor(sidecar, weights=weights, label="promoted v4 router")
    runtime_contract = ROUTER_PROBE._feature_contract(sidecar, None)
    if runtime_contract != RouterFeatureContract(max_tokens=256):
        raise ValueError("promoted v4 feature contract differs from the frozen runtime")
    return router, sidecar, {"v4:config": config, "v4:weights": weights}, checkpoint


def _candidate_dataset_descriptors(
    sidecar: Mapping[str, Any],
    *,
    dataset_snapshots: Mapping[str, ArtifactSnapshot],
) -> None:
    """Require candidate sidecars to bind train+validation and reject test eagerly."""

    datasets = sidecar.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(DATASET_SPLITS):
        raise ValueError("candidate sidecar must bind exactly train and validation; test forbidden")
    for split in DATASET_SPLITS:
        _require_sidecar_artifact(
            datasets[split],
            dataset_snapshots[split],
            label=f"candidate {split} dataset",
        )


def _candidate_adapter_descriptor(
    sidecar: Mapping[str, Any],
    *,
    adapter_directory: Path,
    adapter_config: ArtifactSnapshot,
    adapter_weights: ArtifactSnapshot,
) -> None:
    adapters = sidecar.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != 1:
        raise ValueError("binary gate sidecar must bind exactly the Taskmaster adapter")
    entry = adapters[0]
    if not isinstance(entry, Mapping) or entry.get("route_id") != TASKMASTER_ROUTE_ID:
        raise ValueError("binary gate adapter route differs from Taskmaster")
    if (
        _absolute_sidecar_path(entry.get("path"), label="candidate Taskmaster adapter path")
        != adapter_directory
    ):
        raise ValueError("candidate Taskmaster adapter directory changed")
    _require_sidecar_artifact(
        entry.get("config"),
        adapter_config,
        label="candidate Taskmaster adapter config",
    )
    _require_sidecar_artifact(
        entry.get("weights"),
        adapter_weights,
        label="candidate Taskmaster adapter weights",
    )


def _validate_candidate_sidecar(
    sidecar: Mapping[str, Any],
    *,
    seed: int,
    checkpoint: Path,
    weights: ArtifactSnapshot,
    dataset_snapshots: Mapping[str, ArtifactSnapshot],
    model_evidence: ModelRuntimeEvidence,
    adapter_directory: Path,
    adapter_config: ArtifactSnapshot,
    adapter_weights: ArtifactSnapshot,
) -> None:
    if sidecar.get("schema_version") != ROUTER_CHECKPOINT_SCHEMA:
        raise ValueError("candidate router sidecar schema changed")
    if tuple(sidecar.get("route_ids", ())) != GATE_ROUTE_IDS:
        raise ValueError("candidate route axis must be Taskmaster then base/null")
    if sidecar.get("route_to_index") != {TASKMASTER_ROUTE_ID: 0, NULL_ROUTE_ID: 1}:
        raise ValueError("candidate route-index mapping changed")
    expected_training = {
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.002,
        "weight_decay": 0.0001,
        "label_smoothing": 0.05,
        "load_balance_weight": 0.01,
        "z_loss_weight": 0.0001,
        "patience": 4,
        "seed": seed,
        "device": "cpu",
        "router_hidden_size": None,
    }
    if sidecar.get("training_config") != expected_training:
        raise ValueError(f"candidate seed {seed} training config differs from preregistration")
    expected_router = {
        "hidden_size": 2048,
        "num_experts": 1,
        "top_k": 1,
        "mode": "sequence",
        "router_hidden_size": None,
        "temperature": 1.0,
        "include_null_expert": True,
        "normalize_top_k": True,
        "uniform_top_k": False,
        "straight_through": False,
    }
    if sidecar.get("router_config") != expected_router:
        raise ValueError("candidate gate must remain a linear two-route sequence head")
    if sidecar.get("feature_contract") != RouterFeatureContract(max_tokens=256).to_dict():
        raise ValueError("candidate feature contract differs from frozen validation features")
    _candidate_dataset_descriptors(sidecar, dataset_snapshots=dataset_snapshots)
    validate_model_sidecar(sidecar, evidence=model_evidence, label=f"candidate seed {seed}")
    _candidate_adapter_descriptor(
        sidecar,
        adapter_directory=adapter_directory,
        adapter_config=adapter_config,
        adapter_weights=adapter_weights,
    )
    _validate_checkpoint_descriptor(sidecar, weights=weights, label=f"candidate seed {seed}")
    if checkpoint != weights.path.parent:
        raise ValueError("candidate checkpoint directory and weights path differ")
    metrics = sidecar.get("metrics")
    if not isinstance(metrics, Mapping) or "test" in metrics:
        raise ValueError("candidate sidecar must not contain test metrics")
    cache_hits = metrics.get("feature_cache_hits")
    if not isinstance(cache_hits, Mapping) or set(cache_hits) != set(DATASET_SPLITS):
        raise ValueError("candidate feature-cache record must contain train and validation only")


def recompute_validation_metrics(
    router: Any,
    features: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Fresh binary validation at the frozen >=0.5 Taskmaster threshold."""

    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("validation features and labels are not aligned")
    if labels.numel() != 780:
        raise ValueError("fresh validation requires exactly 780 examples")
    if batch_size < 1:
        raise ValueError("validation batch size must be positive")
    positive_index = GATE_ROUTE_IDS.index(TASKMASTER_ROUTE_ID)
    negative_index = GATE_ROUTE_IDS.index(NULL_ROUTE_ID)
    router = router.float().cpu().eval().requires_grad_(False)
    total_loss = 0.0
    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    with torch.inference_mode():
        for start in range(0, labels.numel(), batch_size):
            batch_features = features[start : start + batch_size].float().cpu()
            batch_labels = labels[start : start + batch_size].long().cpu()
            logits = router(batch_features).logits.float()
            if logits.shape != (batch_labels.numel(), 2):
                raise ValueError("binary gate produced a non-binary logit matrix")
            taskmaster_scores = torch.sigmoid(logits[:, positive_index] - logits[:, negative_index])
            if not bool(torch.isfinite(taskmaster_scores).all()):
                raise ValueError("binary validation produced a non-finite Taskmaster score")
            predictions = torch.where(
                taskmaster_scores >= 0.5,
                torch.full_like(batch_labels, positive_index),
                torch.full_like(batch_labels, negative_index),
            )
            total_loss += float(F.cross_entropy(logits, batch_labels, reduction="sum").item())
            total += int(batch_labels.numel())
            correct += int((predictions == batch_labels).sum().item())
            positive_labels = batch_labels == positive_index
            positive_predictions = predictions == positive_index
            true_positive += int((positive_labels & positive_predictions).sum().item())
            false_positive += int((~positive_labels & positive_predictions).sum().item())
            false_negative += int((positive_labels & ~positive_predictions).sum().item())
            true_negative += int((~positive_labels & ~positive_predictions).sum().item())
    positive_support = true_positive + false_negative
    negative_support = true_negative + false_positive
    predicted_positive = true_positive + false_positive
    loss = total_loss / total
    if not math.isfinite(loss):
        raise ValueError("fresh validation loss is non-finite")
    return {
        "loss": loss,
        "accuracy": correct / total,
        "examples": total,
        "threshold": 0.5,
        "taskmaster_precision": (true_positive / predicted_positive if predicted_positive else 0.0),
        "taskmaster_recall": true_positive / positive_support if positive_support else 0.0,
        "false_positives": false_positive,
        "true_positives": true_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "taskmaster_support": positive_support,
        "non_taskmaster_support": negative_support,
    }


def assess_validation_metrics(
    metrics: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    gates = (
        _gate(
            "validation_loss_finite",
            actual=metrics.get("loss"),
            required="finite",
            passed=isinstance(metrics.get("loss"), float) and math.isfinite(float(metrics["loss"])),
        ),
        _gate(
            "validation_support",
            actual={
                "taskmaster": metrics.get("taskmaster_support"),
                "non_taskmaster": metrics.get("non_taskmaster_support"),
            },
            required={"taskmaster": 130, "non_taskmaster": 650},
            passed=metrics.get("taskmaster_support") == 130
            and metrics.get("non_taskmaster_support") == 650,
        ),
        _gate(
            "validation_accuracy",
            actual=metrics.get("accuracy"),
            required=">=0.995",
            passed=_number(metrics.get("accuracy"), label="validation accuracy") >= 0.995,
        ),
        _gate(
            "taskmaster_precision",
            actual=metrics.get("taskmaster_precision"),
            required=">=0.99",
            passed=_number(metrics.get("taskmaster_precision"), label="Taskmaster precision")
            >= 0.99,
        ),
        _gate(
            "taskmaster_recall",
            actual=metrics.get("taskmaster_recall"),
            required=">=0.99",
            passed=_number(metrics.get("taskmaster_recall"), label="Taskmaster recall") >= 0.99,
        ),
        _gate(
            "false_positives_on_650_validation_negatives",
            actual=metrics.get("false_positives"),
            required="0 of 650",
            passed=metrics.get("false_positives") == 0,
        ),
    )
    return gates, all(bool(gate["passed"]) for gate in gates)


def prepare_validation_features(
    validation_path: Path,
    *,
    embedding: Tensor,
    tokenizer: Any,
    feature_batch_size: int,
) -> tuple[Tensor, Tensor, RouterFeatureContract]:
    examples = load_router_jsonl(validation_path, route_ids=GATE_ROUTE_IDS)
    counts = Counter(example.route_id for example in examples)
    if len(examples) != 780 or dict(counts) != EXPECTED_VALIDATION_SUPPORT:
        raise ValueError("fresh validation support differs from 130 positive / 650 negative")
    contract = RouterFeatureContract(max_tokens=256)
    features, labels = extract_prompt_features(
        examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=GATE_ROUTE_IDS,
        contract=contract,
        batch_size=feature_batch_size,
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
    adapter_directory: Path,
    adapter_config: ArtifactSnapshot,
    adapter_weights: ArtifactSnapshot,
    repository: Path = WORKSPACE_ROOT,
    feature_batch_size: int = 64,
) -> tuple[
    tuple[CandidateEvidence, ...],
    dict[str, ArtifactSnapshot],
    dict[int, tuple[Any, Mapping[str, Any]]],
    dict[str, Any],
]:
    if feature_contract != RouterFeatureContract(max_tokens=256):
        raise ValueError("fresh validation feature contract changed")
    evidence: list[CandidateEvidence] = []
    snapshots: dict[str, ArtifactSnapshot] = {}
    loaded: dict[int, tuple[Any, Mapping[str, Any]]] = {}
    for seed in EXPECTED_SEEDS:
        checkpoint = _candidate_binding(policy, seed, repository=repository)
        config = snapshot(checkpoint / "router_config.json")
        weights = snapshot(checkpoint / "router.safetensors")
        snapshots[f"candidate:{seed}:config"] = config
        snapshots[f"candidate:{seed}:weights"] = weights
        router, sidecar = ROUTER_PROBE.load_checkpoint(checkpoint)
        _validate_candidate_sidecar(
            sidecar,
            seed=seed,
            checkpoint=checkpoint,
            weights=weights,
            dataset_snapshots=dataset_snapshots,
            model_evidence=model_evidence,
            adapter_directory=adapter_directory,
            adapter_config=adapter_config,
            adapter_weights=adapter_weights,
        )
        metrics = recompute_validation_metrics(
            router,
            validation_features,
            validation_labels,
            batch_size=feature_batch_size,
        )
        gates, passed = assess_validation_metrics(metrics)
        evidence.append(
            CandidateEvidence(
                seed=seed,
                checkpoint=checkpoint,
                config=config,
                weights=weights,
                validation_loss=float(metrics["loss"]),
                validation_accuracy=float(metrics["accuracy"]),
                taskmaster_precision=float(metrics["taskmaster_precision"]),
                taskmaster_recall=float(metrics["taskmaster_recall"]),
                validation_false_positives=int(metrics["false_positives"]),
                taskmaster_support=int(metrics["taskmaster_support"]),
                non_taskmaster_support=int(metrics["non_taskmaster_support"]),
                gates=gates,
                passed=passed,
            )
        )
        loaded[seed] = (router, sidecar)
    candidates = tuple(evidence)
    if not all(item.passed for item in candidates):
        raise PreblindMetricGateFailure(
            "one or more candidate seeds failed fresh validation-only gates",
            candidates=candidates,
            snapshots=snapshots,
        )
    selected = select_candidate(candidates)
    selection = {
        "metric": "fresh_validation_cross_entropy",
        "source_split": "validation",
        "direction": "minimize",
        "tie_breaker": "lowest_numeric_seed",
        "selected_seed": selected.seed,
        "fresh_validation_loss_by_seed": {
            str(item.seed): item.validation_loss for item in candidates
        },
        "test_split_loaded": False,
        "test_metrics_used": False,
        "sidecar_metrics_used": False,
        "blind_probe_used": False,
    }
    return candidates, snapshots, loaded, selection


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> ArtifactSnapshot:
    """Publish complete evidence atomically without replacing an earlier run."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"one-shot evidence already exists: {destination}")
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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


def _with_payload_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["payload_sha256"] = sha256_text(canonical_json(result))
    return result


def _verify_payload_hash(payload: Mapping[str, Any], *, label: str) -> None:
    digest = payload.get("payload_sha256")
    if _SHA256.fullmatch(str(digest)) is None:
        raise ValueError(f"{label} payload hash is invalid")
    unhashed = dict(payload)
    unhashed.pop("payload_sha256")
    if sha256_text(canonical_json(unhashed)) != digest:
        raise ValueError(f"{label} payload hash does not verify")


def build_preblind_receipt(
    *,
    policy_snapshot: ArtifactSnapshot,
    runtime_snapshots: Mapping[str, ArtifactSnapshot],
    model_evidence: ModelRuntimeEvidence,
    dataset_snapshots: Mapping[str, ArtifactSnapshot],
    checkpoint_snapshots: Mapping[str, ArtifactSnapshot],
    candidates: Sequence[CandidateEvidence],
    selection: Mapping[str, Any],
    repository: Path,
    feature_batch_size: int,
) -> dict[str, Any]:
    selected = select_candidate(candidates)
    if selection.get("selected_seed") != selected.seed:
        raise ValueError("receipt selection differs from fresh validation selection")
    payload = {
        "schema_version": PREBLIND_SCHEMA,
        "purpose": "freeze hierarchical router v7 before the first blind-probe open",
        "policy": policy_snapshot.to_dict(repository=repository),
        "runtime_source_artifacts": {
            key: value.to_dict(repository=repository)
            for key, value in sorted(runtime_snapshots.items())
        },
        "model_runtime": model_evidence.to_dict(repository=repository),
        "dataset_artifacts": {
            key: value.to_dict(repository=repository)
            for key, value in sorted(dataset_snapshots.items())
        },
        "checkpoint_and_adapter_artifacts": {
            key: value.to_dict(repository=repository)
            for key, value in sorted(checkpoint_snapshots.items())
        },
        "candidate_validation": [item.to_dict(repository=repository) for item in candidates],
        "selection": dict(selection),
        "selected_candidate": selected.to_dict(repository=repository),
        "execution": {
            "device": "cpu",
            "feature_batch_size": feature_batch_size,
            "gate_threshold": 0.5,
            "negative_action": "delegate_exact_pinned_v4_decision",
            "positive_action": f"select_only:{TASKMASTER_ROUTE_ID}",
        },
        "checks": {
            "all_three_seed_gates_passed": True,
            "candidate_selection_uses_fresh_validation_ce_only": True,
            "candidate_sidecar_metrics_used": False,
            "test_split_exists_for_v7": False,
            "test_artifact_opened": False,
            "test_metrics_used": False,
            "blind_probe_opened": False,
            "blind_probe_parsed": False,
            "all_runtime_and_candidate_hashes_recorded": True,
            "promoted_v4_runtime_pinned": True,
            "registration_changed": False,
        },
    }
    return _with_payload_hash(payload)


def open_frozen_probe(
    policy: Mapping[str, Any],
    *,
    repository: Path = WORKSPACE_ROOT,
) -> tuple[tuple[Any, ...], ArtifactSnapshot]:
    """Hash and semantically parse the probe; call only after receipt verification."""

    descriptor = policy["blind_probe"]
    path = _repository_path(repository, descriptor["path"], label="blind probe path")
    expected = (
        repository / "experiments" / "first-real-experts" / "router-blind-probe-v7.jsonl"
    ).resolve()
    if path != expected:
        raise ValueError("blind probe path differs from the frozen v7 path")
    value = snapshot(path)
    if value.sha256 != descriptor["sha256"] or value.size_bytes != descriptor["size_bytes"]:
        raise ValueError("blind probe bytes differ from the frozen v7 policy")
    items = tuple(ROUTER_PROBE.load_probe(path))
    if len(items) != 84:
        raise ValueError("blind probe parsed row count differs from 84")
    if any(item.purpose != PROBE_PURPOSE for item in items):
        raise ValueError("blind probe contains a non-evaluation purpose marker")
    categories = dict(sorted(Counter(item.category for item in items).items()))
    if categories != dict(sorted(descriptor["category_counts"].items())):
        raise ValueError("blind probe categories differ from the frozen policy")
    taskmaster_rows = sum(item.expected_route == TASKMASTER_ROUTE_ID for item in items)
    if taskmaster_rows != 12:
        raise ValueError("blind probe Taskmaster support differs from 12")
    allowed_routes = set(V4_ROUTE_IDS) | {TASKMASTER_ROUTE_ID}
    unknown = {
        route
        for item in items
        for route in (item.expected_route, *item.acceptable_routes)
        if route is not None and route not in allowed_routes
    }
    if unknown:
        raise ValueError("blind probe contains routes outside v4 plus Taskmaster")
    return items, value


def seal_preblind_then_open_probe(
    *,
    receipt_path: Path,
    receipt_payload: Mapping[str, Any],
    policy: Mapping[str, Any],
    repository: Path,
    snapshots_to_recheck: Mapping[str, ArtifactSnapshot],
    probe_loader: Callable[..., tuple[tuple[Any, ...], ArtifactSnapshot]],
) -> tuple[tuple[Any, ...], ArtifactSnapshot, ArtifactSnapshot]:
    """Enforce receipt creation/re-read and pin recheck before the first probe load."""

    receipt_snapshot = write_json_exclusive(receipt_path, receipt_payload)
    reread = _read_json_object(receipt_snapshot.path, label="router v7 preblind receipt")
    _verify_payload_hash(reread, label="router v7 preblind receipt")
    if canonical_json(reread) != canonical_json(receipt_payload):
        raise OSError("preblind receipt bytes do not encode the requested payload")
    assert_unchanged(receipt_snapshot, label="router v7 preblind receipt")
    for label, value in snapshots_to_recheck.items():
        assert_unchanged(value, label=label)
    # The first probe hash/open/parse is strictly below this durable boundary.
    items, probe_snapshot = probe_loader(policy, repository=repository)
    return items, probe_snapshot, receipt_snapshot


def compute_taskmaster_scores(router: Any, features: Tensor) -> Tensor:
    router = router.float().cpu().eval().requires_grad_(False)
    with torch.inference_mode():
        logits = router(features.float().cpu()).logits.float()
    if logits.shape != (features.shape[0], 2):
        raise ValueError("selected binary gate produced a non-binary logit matrix")
    positive = GATE_ROUTE_IDS.index(TASKMASTER_ROUTE_ID)
    negative = GATE_ROUTE_IDS.index(NULL_ROUTE_ID)
    scores = torch.sigmoid(logits[:, positive] - logits[:, negative])
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("selected binary gate produced non-finite Taskmaster scores")
    return scores


def _binary_entropy(score: float) -> float:
    if score <= 0.0 or score >= 1.0:
        return 0.0
    return -(score * math.log(score) + (1.0 - score) * math.log(1.0 - score))


def build_hierarchical_predictions(
    *,
    items: Sequence[Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    taskmaster_scores: Sequence[float],
) -> list[dict[str, Any]]:
    """Apply the fixed binary gate while preserving every negative v4 decision field."""

    if len(items) != len(baseline_rows) or len(items) != len(taskmaster_scores):
        raise ValueError("hierarchical inputs are not aligned")
    rows: list[dict[str, Any]] = []
    decision_fields = (
        "primary_route",
        "selected_routes",
        "applied_weights",
        "probabilities",
        "confidence",
        "entropy",
        "threshold_abstained",
        "base_null_selected",
        "exact_match",
        "acceptable_match",
        "top_k_acceptable_match",
    )
    for item, baseline, raw_score in zip(
        items,
        baseline_rows,
        taskmaster_scores,
        strict=True,
    ):
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Taskmaster score must be finite and within [0, 1]")
        if baseline.get("id") != item.probe_id:
            raise ValueError("baseline prediction is not aligned to the probe")
        admitted = score >= 0.5
        if not admitted:
            # Copy the v4 prediction, then add audit-only hierarchy metadata.  Every
            # delegate decision field remains exactly the standalone v4 value.
            row = dict(baseline)
            delegate_projection = {key: baseline.get(key) for key in decision_fields}
            if {key: row.get(key) for key in decision_fields} != delegate_projection:
                raise AssertionError("gate-negative v4 decision changed during delegation")
            row["hierarchy"] = {
                "taskmaster_probability": score,
                "threshold": 0.5,
                "gate_positive": False,
                "action": "delegate_exact_v4_decision",
                "delegate_consulted_for_candidate_decision": True,
                "delegate_decision_sha256": sha256_text(canonical_json(delegate_projection)),
            }
        else:
            exact_match = item.expected_route == TASKMASTER_ROUTE_ID
            acceptable_match = TASKMASTER_ROUTE_ID in item.acceptable_routes
            row = {
                "id": item.probe_id,
                "category": item.category,
                "prompt": item.prompt,
                "expected_route": item.expected_route,
                "acceptable_routes": list(item.acceptable_routes),
                "primary_route": TASKMASTER_ROUTE_ID,
                "selected_routes": [TASKMASTER_ROUTE_ID],
                "applied_weights": {TASKMASTER_ROUTE_ID: 1.0},
                "probabilities": {
                    TASKMASTER_ROUTE_ID: score,
                    "delegate_v4": 1.0 - score,
                },
                "confidence": score,
                "entropy": _binary_entropy(score),
                "threshold_abstained": False,
                "base_null_selected": False,
                "exact_match": exact_match,
                "acceptable_match": acceptable_match,
                "top_k_acceptable_match": acceptable_match,
                "hierarchy": {
                    "taskmaster_probability": score,
                    "threshold": 0.5,
                    "gate_positive": True,
                    "action": f"select_only:{TASKMASTER_ROUTE_ID}",
                    "delegate_consulted_for_candidate_decision": False,
                },
            }
        rows.append(row)
    return rows


def summarize_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("prediction rows must not be empty")
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        category = row.get("category")
        if not isinstance(category, str):
            raise ValueError("prediction row category is missing")
        by_category[category].append(row)
    return {
        "prompt_count": len(rows),
        "exact_correct": sum(row.get("exact_match") is True for row in rows),
        "acceptable_correct": sum(row.get("acceptable_match") is True for row in rows),
        "taskmaster_selections": sum(
            row.get("primary_route") == TASKMASTER_ROUTE_ID for row in rows
        ),
        "gate_positive_count": sum(
            isinstance(row.get("hierarchy"), Mapping)
            and row["hierarchy"].get("gate_positive") is True
            for row in rows
        ),
        "by_category": {
            category: {
                "count": len(values),
                "exact_correct": sum(row.get("exact_match") is True for row in values),
                "acceptable_correct": sum(row.get("acceptable_match") is True for row in values),
                "taskmaster_selections": sum(
                    row.get("primary_route") == TASKMASTER_ROUTE_ID for row in values
                ),
            }
            for category, values in sorted(by_category.items())
        },
    }


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
    if len(items) != 84 or len(baseline_rows) != 84 or len(candidate_rows) != 84:
        raise ValueError("one-shot completeness requires 84 aligned rows")
    ids = [item.probe_id for item in items]
    if [row.get("id") for row in baseline_rows] != ids or [
        row.get("id") for row in candidate_rows
    ] != ids:
        raise ValueError("one-shot predictions are not aligned to frozen probe IDs")
    if any(item.expected_route is None for item in items):
        raise ValueError("one-shot registration requires exact labels on all rows")
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
        raise ValueError("one-shot probe must contain 12 Taskmaster and 72 other rows")
    taskmaster_correct = sum(
        candidate_rows[index].get("primary_route") == TASKMASTER_ROUTE_ID
        for index in taskmaster_indices
    )
    false_taskmaster = sum(
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
            actual=false_taskmaster,
            required="<=1 of 72 non-Taskmaster rows",
            passed=false_taskmaster <= 1,
        ),
    ]
    categories: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        categories[item.category].append(index)
    if set(categories) != set(EXPECTED_PROTECTED_CATEGORIES) | {TASKMASTER_ROUTE_ID}:
        raise ValueError("one-shot category set differs from the frozen strata")
    by_category: dict[str, dict[str, Any]] = {}
    for category, indices in sorted(categories.items()):
        if len(indices) != 12:
            raise ValueError(f"probe category {category!r} must contain exactly 12 rows")
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
        exact_drop = baseline_exact_count - candidate_exact_count
        acceptable_drop = baseline_acceptable_count - candidate_acceptable_count
        by_category[category] = {
            "count": 12,
            "baseline_exact_correct": baseline_exact_count,
            "candidate_exact_correct": candidate_exact_count,
            "baseline_acceptable_correct": baseline_acceptable_count,
            "candidate_acceptable_correct": candidate_acceptable_count,
            "exact_drop": exact_drop,
            "acceptable_drop": acceptable_drop,
        }
        if category in EXPECTED_PROTECTED_CATEGORIES:
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
            "taskmaster_false_positives": false_taskmaster,
        },
        "by_category": by_category,
        "gates": gates,
        "all_gates_passed": all(bool(gate["passed"]) for gate in gates),
    }


def _redact_prompt_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "prompt"} for row in rows]


def _preblind_failure_report(
    *,
    error: PreblindMetricGateFailure,
    policy_snapshot: ArtifactSnapshot,
    runtime_snapshots: Mapping[str, ArtifactSnapshot],
    model_evidence: ModelRuntimeEvidence,
    dataset_snapshots: Mapping[str, ArtifactSnapshot],
    repository: Path,
) -> dict[str, Any]:
    return _with_payload_hash(
        {
            "schema_version": REPORT_SCHEMA,
            "evaluation_kind": "router_v7_preblind_validation_gate_failure",
            "policy": policy_snapshot.to_dict(repository=repository),
            "runtime_source_artifacts": {
                key: value.to_dict(repository=repository)
                for key, value in sorted(runtime_snapshots.items())
            },
            "model_runtime": model_evidence.to_dict(repository=repository),
            "dataset_artifacts": {
                key: value.to_dict(repository=repository)
                for key, value in sorted(dataset_snapshots.items())
            },
            "preblind": {
                "failure": str(error),
                "candidates": [item.to_dict(repository=repository) for item in error.candidates],
                "candidate_artifacts": {
                    key: value.to_dict(repository=repository)
                    for key, value in sorted(error.snapshots.items())
                },
                "test_split_loaded": False,
                "test_metrics_used": False,
            },
            "probe": {"opened": False, "parsed": False, "metrics_computed": False},
            "decision": "rejected_preblind_validation_gate_failed_probe_remains_sealed",
            "registration_changed": False,
            "auto_registration_performed": False,
        }
    )


def run_registration_evaluation(
    *,
    policy_path: Path = DEFAULT_POLICY,
    model_path: Path = DEFAULT_MODEL,
    output_path: Path = DEFAULT_OUTPUT,
    preblind_receipt_path: Path | None = None,
    repository: Path = WORKSPACE_ROOT,
    feature_batch_size: int = 64,
) -> dict[str, Any]:
    """Run the frozen evaluation; passing still requires a separate registration change."""

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
        raise ValueError("output_path differs from the frozen one-shot destination")
    frozen_receipt = _repository_path(
        repository,
        policy["evidence_discipline"]["preblind_receipt_path"],
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
        raise FileExistsError(f"preblind receipt already exists: {frozen_receipt}")
    if feature_batch_size != 64:
        raise ValueError("feature_batch_size differs from the frozen execution contract")

    runtime_snapshots = validate_runtime_sources(policy, repository=repository)
    model_evidence = validate_model_runtime(
        policy,
        model_path=model_path,
        repository=repository,
    )
    dataset_snapshots, dataset_report = validate_dataset_contract(
        policy,
        repository=repository,
    )
    adapter_directory, adapter_config, adapter_weights = validate_taskmaster_adapter(
        policy,
        repository=repository,
    )
    embedding, embedding_metadata = load_input_embedding(
        model_evidence.directory,
        tensor_name=model_evidence.embedding_tensor_name,
    )
    validate_loaded_embedding(embedding_metadata, evidence=model_evidence)
    tokenizer = load_router_tokenizer(model_evidence.directory)
    validation_features, validation_labels, feature_contract = prepare_validation_features(
        dataset_snapshots["validation"].path,
        embedding=embedding,
        tokenizer=tokenizer,
        feature_batch_size=feature_batch_size,
    )
    try:
        candidates, candidate_snapshots, loaded_candidates, selection = validate_candidates(
            policy,
            dataset_snapshots=dataset_snapshots,
            validation_features=validation_features,
            validation_labels=validation_labels,
            feature_contract=feature_contract,
            model_evidence=model_evidence,
            adapter_directory=adapter_directory,
            adapter_config=adapter_config,
            adapter_weights=adapter_weights,
            repository=repository,
            feature_batch_size=feature_batch_size,
        )
    except PreblindMetricGateFailure as exc:
        failure = _preblind_failure_report(
            error=exc,
            policy_snapshot=policy_snapshot,
            runtime_snapshots=runtime_snapshots,
            model_evidence=model_evidence,
            dataset_snapshots=dataset_snapshots,
            repository=repository,
        )
        write_json_exclusive(output_path, failure)
        return failure

    v4_router, v4_sidecar, v4_snapshots, v4_checkpoint = validate_v4_delegate(
        policy,
        model_evidence=model_evidence,
        repository=repository,
    )
    selected = select_candidate(candidates)
    selected_router, selected_sidecar = loaded_candidates[selected.seed]
    selected_contract = ROUTER_PROBE._feature_contract(selected_sidecar, None)
    v4_contract = ROUTER_PROBE._feature_contract(v4_sidecar, None)
    if selected_contract != feature_contract or v4_contract != feature_contract:
        raise ValueError("gate, delegate, and frozen feature contracts are not identical")

    checkpoint_snapshots = {
        **candidate_snapshots,
        **v4_snapshots,
        "taskmaster_adapter:config": adapter_config,
        "taskmaster_adapter:weights": adapter_weights,
    }
    all_preblind_snapshots = {
        "policy": policy_snapshot,
        **runtime_snapshots,
        **model_evidence.snapshots,
        **dataset_snapshots,
        **checkpoint_snapshots,
    }
    for label, value in all_preblind_snapshots.items():
        assert_unchanged(value, label=label)
    receipt = build_preblind_receipt(
        policy_snapshot=policy_snapshot,
        runtime_snapshots=runtime_snapshots,
        model_evidence=model_evidence,
        dataset_snapshots=dataset_snapshots,
        checkpoint_snapshots=checkpoint_snapshots,
        candidates=candidates,
        selection=selection,
        repository=repository,
        feature_batch_size=feature_batch_size,
    )
    items, probe_snapshot, receipt_snapshot = seal_preblind_then_open_probe(
        receipt_path=frozen_receipt,
        receipt_payload=receipt,
        policy=policy,
        repository=repository,
        snapshots_to_recheck=all_preblind_snapshots,
        probe_loader=open_frozen_probe,
    )

    probe_examples = [RouterExample(item.prompt, TASKMASTER_ROUTE_ID) for item in items]
    probe_features, _ = extract_prompt_features(
        probe_examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=GATE_ROUTE_IDS,
        contract=feature_contract,
        batch_size=feature_batch_size,
    )
    del embedding
    runtime = policy["route_contract"]["frozen_v4_runtime"]
    baseline_metrics, baseline_rows = ROUTER_PROBE.evaluate(
        v4_router,
        v4_sidecar,
        items,
        probe_features,
        temperature=runtime["temperature"],
        confidence_threshold=runtime["confidence_threshold"],
        null_bias=runtime["null_bias"],
        top_k=runtime["top_k"],
        enabled_experts=set(runtime["enabled_experts"]),
    )
    taskmaster_scores = compute_taskmaster_scores(selected_router, probe_features)
    candidate_rows = build_hierarchical_predictions(
        items=items,
        baseline_rows=baseline_rows,
        taskmaster_scores=taskmaster_scores.tolist(),
    )
    candidate_metrics = summarize_prediction_rows(candidate_rows)
    one_shot = evaluate_one_shot_gates(
        items=items,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
    )

    assert_unchanged(probe_snapshot, label="router v7 blind probe")
    assert_unchanged(receipt_snapshot, label="router v7 preblind receipt")
    for label, value in all_preblind_snapshots.items():
        assert_unchanged(value, label=label)
    report = _with_payload_hash(
        {
            "schema_version": REPORT_SCHEMA,
            "evaluation_kind": "sealed_one_shot_router_v7_hierarchical_registration",
            "policy": policy_snapshot.to_dict(repository=repository),
            "probe": {
                **probe_snapshot.to_dict(repository=repository),
                "purpose": PROBE_PURPOSE,
                "rows": len(items),
                "semantic_parse_count_by_this_evaluator": 1,
            },
            "preblind": {
                "receipt": receipt_snapshot.to_dict(repository=repository),
                "all_seed_gates_passed": True,
                "candidates": [item.to_dict(repository=repository) for item in candidates],
                "selection": selection,
                "test_split_loaded": False,
                "test_metrics_used": False,
            },
            "dataset": {
                "report_schema": dataset_report.get("schema_version"),
                "available_splits": list(DATASET_SPLITS),
                "test_split_present": False,
                "artifacts": {
                    key: value.to_dict(repository=repository)
                    for key, value in sorted(dataset_snapshots.items())
                },
            },
            "model": {
                **model_evidence.to_dict(repository=repository),
                "full_causal_model_loaded": False,
            },
            "runtime_source_artifacts": {
                key: value.to_dict(repository=repository)
                for key, value in sorted(runtime_snapshots.items())
            },
            "hierarchy": {
                "threshold": 0.5,
                "positive": f"select_only:{TASKMASTER_ROUTE_ID}",
                "negative": "return_exact_pinned_v4_decision",
                "v4_checkpoint": v4_checkpoint.relative_to(repository).as_posix(),
                "v4_runtime": runtime,
            },
            "baseline": {
                "metrics": baseline_metrics,
                "examples": _redact_prompt_rows(baseline_rows),
            },
            "candidate": {
                "seed": selected.seed,
                "checkpoint": selected.checkpoint.relative_to(repository).as_posix(),
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
            "copy_alias_or_default_change_performed": False,
        }
    )
    write_json_exclusive(output_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen one-shot hierarchical router v7 evaluation. "
            "This writes evidence only and never changes registration."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument(
        "--acknowledge-one-shot",
        action="store_true",
        help="required acknowledgement that this command permanently exposes the probe",
    )
    args = parser.parse_args(argv)
    if not args.acknowledge_one_shot:
        parser.error("--acknowledge-one-shot is required before the probe can be opened")
    if args.feature_batch_size != 64:
        parser.error("--feature-batch-size is frozen at 64")
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
