"""Conservative reward optimization for the frozen MoLoRA prompt router.

This tool consumes the counterfactual teacher-evaluation matrix rather than dataset labels.  Each
training row contains a complete reward vector for the forced base/null route and every frozen LoRA
expert in the parent router's action axis.  Auto-routed rollouts are deliberately excluded from
optimization.

The learned policy maximizes the full-information contextual-bandit objective

    E[sum_a pi(a | prompt) * reward(prompt, a)] - beta * KL(pi || pi_0)

where ``pi_0`` is the immutable bootstrap-router checkpoint.  Only the tiny prompt-router network
is updated; the language model and expert adapters are never loaded or changed.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import run_teacher_eval as evaluation
from molor.config import RouterConfig
from molor.router import TopKRouter
from molor.training.router import (
    ROUTER_CHECKPOINT_SCHEMA,
    RouterExample,
    RouterFeatureContract,
    extract_prompt_features,
    load_input_embedding,
    load_router_tokenizer,
)


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")
DEFAULT_INITIAL_ROUTER = WORKSPACE / "checkpoints" / "qwen3-1.7b-corpus-router-v1"
DEFAULT_OUTPUT = WORKSPACE / "checkpoints" / "qwen3-1.7b-corpus-router-teacher-rl-v1"
REPORT_SCHEMA = "molor.reward-router-training-report.v1"
METHOD = "full_information_contextual_bandit_kl_v1"


class OptimizerError(RuntimeError):
    """An actionable data-contract, provenance, or optimization error."""


@dataclass(frozen=True, slots=True)
class RewardObservation:
    generation_id: str
    score: float
    confidence: float
    evaluator: str
    observation_id: str


@dataclass(frozen=True, slots=True)
class RewardExample:
    prompt_id: str
    prompt: str
    category: str
    rewards: tuple[float, ...]
    annotations: tuple[int, ...]
    mean_confidence: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    train_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
    method: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PreparedRewards:
    manifest: evaluation.EvalManifest
    route_ids: tuple[str, ...]
    examples: tuple[RewardExample, ...]
    split: SplitAssignment
    missing_cells: tuple[dict[str, object], ...]
    source_summary: Mapping[str, object]
    dataset_fingerprint: str


@dataclass(frozen=True, slots=True)
class ParentContract:
    directory: Path
    config_path: Path
    weights_path: Path
    sidecar: dict[str, Any]
    route_ids: tuple[str, ...]
    runtime_names: tuple[str, ...]
    router_config: RouterConfig


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, recorded_path: Path | None = None) -> dict[str, object]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "logical_name": source.name,
        "path": str((recorded_path or source).resolve()),
        "sha256": _sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OptimizerError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise OptimizerError(f"{label} must be a finite number")
    return result


def _load_parent_contract(directory: Path) -> ParentContract:
    root = directory.expanduser().resolve()
    config_path = root / "router_config.json"
    weights_path = root / "router.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise OptimizerError(f"initial router checkpoint is incomplete: {root}")
    try:
        sidecar = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizerError(f"cannot read initial router contract {config_path}: {exc}") from exc
    if not isinstance(sidecar, dict) or sidecar.get("schema_version") != ROUTER_CHECKPOINT_SCHEMA:
        raise OptimizerError("initial router uses an unsupported checkpoint schema")
    raw_routes = sidecar.get("route_ids")
    if not isinstance(raw_routes, list) or any(
        not isinstance(route_id, str) or not route_id for route_id in raw_routes
    ):
        raise OptimizerError("initial router has invalid route_ids")
    route_ids = tuple(raw_routes)
    if len(route_ids) < 2 or route_ids[-1] != "base/null" or len(set(route_ids)) != len(route_ids):
        raise OptimizerError(
            "reward optimization requires one or more unique experts followed by base/null"
        )
    expected_route_to_index = {
        route_id: index for index, route_id in enumerate(route_ids)
    }
    if sidecar.get("route_to_index") != expected_route_to_index:
        raise OptimizerError("initial router route_to_index does not match its ordered route_ids")

    raw_adapters = sidecar.get("adapters")
    if not isinstance(raw_adapters, list) or any(
        not isinstance(adapter, Mapping) for adapter in raw_adapters
    ):
        raise OptimizerError("initial router has invalid adapter metadata")
    adapter_route_ids = tuple(str(adapter.get("route_id") or "") for adapter in raw_adapters)
    if adapter_route_ids != route_ids[:-1]:
        raise OptimizerError(
            "initial router adapter metadata does not match its ordered expert routes"
        )
    runtime_names: list[str] = []
    for route_id, adapter in zip(adapter_route_ids, raw_adapters, strict=True):
        runtime_name = adapter.get("runtime_name", route_id)
        if not isinstance(runtime_name, str) or not evaluation.ID_PATTERN.fullmatch(runtime_name):
            raise OptimizerError("initial router adapter runtime_name is invalid")
        runtime_names.append(runtime_name)
    if len(runtime_names) != len(set(runtime_names)) or "base" in runtime_names:
        raise OptimizerError(
            "initial router adapter runtime names must be unique and cannot use reserved base"
        )

    checkpoint = sidecar.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise OptimizerError("initial router has no checkpoint artifact identity")
    expected_checkpoint_sha = checkpoint.get("sha256")
    expected_checkpoint_size = checkpoint.get("size_bytes")
    if expected_checkpoint_sha != _sha256_file(weights_path) or expected_checkpoint_size != (
        weights_path.stat().st_size
    ):
        raise OptimizerError("initial router weights differ from its checkpoint artifact identity")
    raw_config = sidecar.get("router_config")
    if not isinstance(raw_config, dict):
        raise OptimizerError("initial router has no router_config object")
    try:
        router_config = RouterConfig(**raw_config)
    except (TypeError, ValueError) as exc:
        raise OptimizerError(f"invalid initial router config: {exc}") from exc
    if (
        router_config.num_routes != len(route_ids)
        or not router_config.include_null_expert
        or router_config.mode != "sequence"
    ):
        raise OptimizerError("initial router action axis/configuration is incompatible")
    return ParentContract(
        root,
        config_path,
        weights_path,
        sidecar,
        route_ids,
        tuple([*runtime_names, "base"]),
        router_config,
    )


def _validate_manifest_route_contract(
    manifest: evaluation.EvalManifest,
    parent: ParentContract,
) -> None:
    routes_by_id = {route.route_id: route for route in manifest.forced_routes}
    if set(routes_by_id) != set(parent.route_ids):
        raise OptimizerError(
            "manifest forced routes must match the initial router's action IDs exactly"
        )

    expected_runtime_names = dict(zip(parent.route_ids, parent.runtime_names, strict=True))
    for route_id, runtime_name in expected_runtime_names.items():
        if routes_by_id[route_id].runtime_name != runtime_name:
            raise OptimizerError(
                "manifest runtime routes must match the parent adapter runtime names and map "
                "base/null to base"
            )

    expected_enabled_experts = set(parent.runtime_names[:-1])
    for label, policy in (
        ("manual_policy_defaults", manifest.manual_policy_defaults),
        ("auto.policy", manifest.auto_policy),
    ):
        enabled = policy.get("enabled_experts")
        if (
            not isinstance(enabled, list)
            or len(enabled) != len(expected_enabled_experts)
            or set(enabled) != expected_enabled_experts
        ):
            raise OptimizerError(
                f"manifest {label}.enabled_experts must match the parent expert bank exactly"
            )


def _parent_health_identity(parent: ParentContract) -> dict[str, object]:
    model = parent.sidecar.get("model")
    if not isinstance(model, Mapping):
        raise OptimizerError("initial router has no frozen-base model metadata")
    model_id = model.get("model_id")
    adapters = parent.sidecar.get("adapters")
    if not isinstance(adapters, list):
        raise OptimizerError("initial router has invalid adapter metadata")
    expert_paths = [adapter.get("path") for adapter in adapters]
    health = {
        "status": "ok",
        "protocol_version": "molor.teacher-api.v1",
        "model_loaded": True,
        "model_id": model_id,
        "experts": list(parent.runtime_names[:-1]),
        "configured_expert_paths": expert_paths,
        "router_checkpoint": str(parent.directory),
        "router_loaded": True,
    }
    try:
        return evaluation._student_health_identity(health)
    except evaluation.EvalError as exc:
        raise OptimizerError(f"initial router cannot identify a local student stack: {exc}") from exc


def _collection_provenance(
    events: Sequence[Mapping[str, object]],
    manifest: evaluation.EvalManifest,
    parent: ParentContract,
) -> dict[str, object]:
    """Validate rollout sessions, retaining an honest compatibility status for old reports."""

    sessions = [
        event
        for event in events
        if event.get("event") == "collection_session"
        and event.get("manifest_sha256") == manifest.fingerprint
        and event.get("run_id") == manifest.run_id
    ]
    successful = [
        event
        for event in evaluation._successful_rollouts(events).values()
        if event.get("manifest_sha256") == manifest.fingerprint
        and event.get("run_id") == manifest.run_id
    ]
    if successful and not sessions:
        raise OptimizerError("raw rollouts have no matching collection session provenance")

    legacy_selected = any(event.get("artifact_contract_sha256") is None for event in successful)
    selected_exact_contracts: set[str] = set()
    selected_legacy_capabilities: set[str] = set()
    for rollout in successful:
        contract_sha = rollout.get("artifact_contract_sha256")
        capability_sha = rollout.get("capabilities_sha256")
        if not isinstance(capability_sha, str) or not capability_sha:
            raise OptimizerError("raw rollout has no capabilities fingerprint")
        if contract_sha is None:
            selected_legacy_capabilities.add(capability_sha)
        elif not isinstance(contract_sha, str) or not contract_sha:
            raise OptimizerError("raw rollout has an invalid artifact contract fingerprint")
        else:
            selected_exact_contracts.add(contract_sha)
    if legacy_selected and selected_exact_contracts:
        raise OptimizerError(
            "successful rollout matrix mixes legacy and exact artifact provenance generations"
        )
    if len(selected_exact_contracts) > 1:
        raise OptimizerError(
            "successful rollout matrix mixes multiple exact artifact contract generations"
        )
    if len(selected_legacy_capabilities) > 1:
        raise OptimizerError(
            "successful legacy rollout matrix mixes multiple collection session identities"
        )

    expected_health = _parent_health_identity(parent)
    expected_contract: dict[str, object] | None = None
    exact_contracts: set[str] = set()
    legacy_capabilities: set[str] = set()
    legacy_session_identities: set[str] = set()
    for rollout in successful:
        capability_sha = rollout.get("capabilities_sha256")
        assert isinstance(capability_sha, str)  # validated for the complete selected set above
        contract_sha = rollout.get("artifact_contract_sha256")
        if contract_sha is None:
            matching = [
                session
                for session in sessions
                if session.get("capabilities_sha256") == capability_sha
                and session.get("student_artifact_contract") is None
            ]
            if not matching:
                raise OptimizerError("legacy rollout has no matching legacy collection session")
            for session in matching:
                if session.get("artifact_provenance_status") not in (
                    None,
                    evaluation.LEGACY_ARTIFACT_PROVENANCE,
                ):
                    raise OptimizerError("legacy collection session has an invalid provenance status")
                health = session.get("health")
                if not isinstance(health, Mapping):
                    raise OptimizerError("legacy collection session has no health identity")
                try:
                    actual_health = evaluation._student_health_identity(health)
                except evaluation.EvalError as exc:
                    raise OptimizerError(f"legacy collection health is invalid: {exc}") from exc
                if actual_health != expected_health:
                    raise OptimizerError(
                        "legacy collection health paths/names do not match the parent router stack"
                    )
                legacy_session_identities.add(
                    _fingerprint(
                        {
                            "capabilities_sha256": capability_sha,
                            "health_identity": actual_health,
                        }
                    )
                )
            legacy_capabilities.add(capability_sha)
            continue

        assert isinstance(contract_sha, str)  # validated for the complete selected set above
        matching = [
            session
            for session in sessions
            if session.get("capabilities_sha256") == capability_sha
            and isinstance(session.get("student_artifact_contract"), Mapping)
            and session["student_artifact_contract"].get("contract_sha256") == contract_sha
        ]
        if not matching:
            raise OptimizerError("exact-provenance rollout has no matching collection session")
        if expected_contract is None:
            try:
                expected_contract = evaluation._build_student_artifact_contract(
                    {
                        "status": "ok",
                        "protocol_version": "molor.teacher-api.v1",
                        **expected_health,
                    },
                    show_progress=False,
                )
            except evaluation.EvalError as exc:
                raise OptimizerError(
                    f"cannot fingerprint the parent router's current student stack: {exc}"
                ) from exc
        for session in matching:
            if session.get("artifact_provenance_status") != evaluation.EXACT_ARTIFACT_PROVENANCE:
                raise OptimizerError("exact collection session has an invalid provenance status")
            try:
                actual_contract = evaluation._validate_student_artifact_contract(
                    session.get("student_artifact_contract")
                )
            except evaluation.EvalError as exc:
                raise OptimizerError(str(exc)) from exc
            if actual_contract != expected_contract:
                raise OptimizerError(
                    "collection artifact bytes differ from the parent/current student stack"
                )
        exact_contracts.add(contract_sha)

    if len(legacy_session_identities) > 1:
        raise OptimizerError(
            "successful legacy rollout matrix mixes multiple validated session identities"
        )

    status = (
        evaluation.LEGACY_ARTIFACT_PROVENANCE
        if legacy_capabilities
        else evaluation.EXACT_ARTIFACT_PROVENANCE
    )
    return {
        "status": status,
        "successful_rollouts_validated": len(successful),
        "collection_sessions_considered": len(sessions),
        "exact_contract_sha256": sorted(exact_contracts),
        "legacy_capabilities_sha256": sorted(legacy_capabilities),
        "legacy_session_identity_sha256": sorted(legacy_session_identities),
    }


def _prompt_text(prompt: evaluation.PromptSpec) -> str:
    for message in reversed(prompt.messages):
        if message["role"] == "user" and message["content"].strip():
            return message["content"].strip()
    raise OptimizerError(f"prompt {prompt.prompt_id!r} has no current-user content")


def _prompt_category(prompt: evaluation.PromptSpec) -> str:
    category = prompt.metadata.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()
    return prompt.tags[0] if prompt.tags else "unstratified"


def _split_prompts(
    prompts: Sequence[evaluation.PromptSpec],
    *,
    seed: int,
    heldout_fraction: float,
) -> SplitAssignment:
    if not 0 < heldout_fraction < 1:
        raise OptimizerError("heldout_fraction must be strictly between zero and one")
    text_owner: dict[str, str] = {}
    for prompt in prompts:
        text_hash = _fingerprint(_prompt_text(prompt))
        previous = text_owner.setdefault(text_hash, prompt.prompt_id)
        if previous != prompt.prompt_id:
            raise OptimizerError(
                f"duplicate prompt content under {previous!r} and {prompt.prompt_id!r}; "
                "refusing train/heldout leakage"
            )

    explicit = [prompt.metadata.get("router_split") for prompt in prompts]
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit):
            raise OptimizerError("router_split must be specified for every prompt or none")
        invalid = sorted({str(value) for value in explicit if value not in {"train", "heldout"}})
        if invalid:
            raise OptimizerError(f"invalid explicit router_split values: {invalid}")
        train_ids = tuple(
            prompt.prompt_id
            for prompt, value in zip(prompts, explicit, strict=True)
            if value == "train"
        )
        heldout_ids = tuple(
            prompt.prompt_id
            for prompt, value in zip(prompts, explicit, strict=True)
            if value == "heldout"
        )
        method = "manifest.metadata.router_split"
    else:
        by_category: dict[str, list[evaluation.PromptSpec]] = {}
        for prompt in prompts:
            by_category.setdefault(_prompt_category(prompt), []).append(prompt)
        heldout: set[str] = set()
        for category, rows in sorted(by_category.items()):
            ranked = sorted(
                rows,
                key=lambda prompt: _fingerprint(
                    {
                        "seed": seed,
                        "category": category,
                        "prompt_id": prompt.prompt_id,
                        "prompt": _prompt_text(prompt),
                    }
                ),
            )
            if len(ranked) < 2:
                continue
            count = min(len(ranked) - 1, max(1, math.ceil(len(ranked) * heldout_fraction)))
            heldout.update(prompt.prompt_id for prompt in ranked[:count])
        train_ids = tuple(prompt.prompt_id for prompt in prompts if prompt.prompt_id not in heldout)
        heldout_ids = tuple(prompt.prompt_id for prompt in prompts if prompt.prompt_id in heldout)
        method = "deterministic_category_stratified_hash"

    if not train_ids or not heldout_ids:
        raise OptimizerError("train and heldout prompt sets must both be non-empty")
    overlap = set(train_ids).intersection(heldout_ids)
    if overlap or set(train_ids).union(heldout_ids) != {prompt.prompt_id for prompt in prompts}:
        raise OptimizerError("train/heldout separation invariant failed")
    payload = {
        "seed": seed,
        "heldout_fraction": heldout_fraction,
        "method": method,
        "train_ids": list(train_ids),
        "heldout_ids": list(heldout_ids),
        "prompt_hashes": {
            prompt.prompt_id: _fingerprint(_prompt_text(prompt)) for prompt in prompts
        },
    }
    return SplitAssignment(train_ids, heldout_ids, method, _fingerprint(payload))


def _load_candidates(
    manifest: evaluation.EvalManifest,
    report: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, int, str], list[dict[str, Any]]],
    int,
]:
    events = evaluation._read_events(report, "raw report")
    incompatible = [
        event
        for event in events
        if event.get("event") == "rollout"
        and (
            event.get("manifest_sha256") != manifest.fingerprint
            or event.get("run_id") != manifest.run_id
        )
    ]
    if incompatible:
        raise OptimizerError(
            f"raw report contains {len(incompatible)} rollout events from another manifest/run"
        )
    candidates = evaluation._candidate_rows(events)
    prompts = {prompt.prompt_id: prompt for prompt in manifest.prompts}
    route_runtime = {route.route_id: route.runtime_name for route in manifest.forced_routes}
    forced: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    ignored_auto = 0
    for candidate_key, row in candidates.items():
        prompt_id = row.get("prompt_id")
        route_id = row.get("route_id")
        repetition = row.get("repetition")
        if not isinstance(prompt_id, str) or prompt_id not in prompts:
            raise OptimizerError(f"candidate {candidate_key!r} has an unknown prompt_id")
        if route_id == manifest.auto_route_id:
            ignored_auto += 1
            continue
        if not isinstance(route_id, str) or route_id not in route_runtime:
            raise OptimizerError(f"candidate {candidate_key!r} has an unknown forced route")
        if row.get("routing_mode") != "forced":
            raise OptimizerError(f"candidate {candidate_key!r} is not a forced-route rollout")
        if (
            isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or not 0 <= repetition < manifest.repetitions
        ):
            raise OptimizerError(f"candidate {candidate_key!r} has an invalid repetition")
        if _canonical_json(row.get("messages")) != _canonical_json(prompts[prompt_id].messages):
            raise OptimizerError(f"candidate {candidate_key!r} messages differ from the manifest")
        decision = row.get("routing_decision")
        if not isinstance(decision, Mapping):
            raise OptimizerError(f"candidate {candidate_key!r} has no routing decision")
        applied = decision.get("applied_route")
        expected_runtime = route_runtime[route_id]
        if not isinstance(applied, Mapping) or set(applied) != {expected_runtime}:
            raise OptimizerError(
                f"candidate {candidate_key!r} did not execute only {expected_runtime!r}"
            )
        weight = _finite(applied[expected_runtime], f"{candidate_key}.applied_route")
        if not math.isclose(weight, 1.0, abs_tol=1e-6):
            raise OptimizerError(f"candidate {candidate_key!r} was not a unit-scale forced route")
        forced.setdefault((prompt_id, repetition, route_id), []).append(row)
    return candidates, forced, ignored_auto


def _local_observations(
    manifest: evaluation.EvalManifest,
    judgments_path: Path,
    candidates: Mapping[str, Mapping[str, object]],
    *,
    reward_key: str,
    min_confidence: float,
) -> tuple[dict[str, list[RewardObservation]], dict[str, object]]:
    path = judgments_path.expanduser().resolve()
    operations = evaluation._submission_operations(manifest, path, candidates)
    observations: dict[str, list[RewardObservation]] = {}
    seen_candidates: set[str] = set()
    preference_count = 0
    skipped_confidence = 0
    for operation in operations:
        if operation.get("kind") == "preference":
            preference_count += 1
            continue
        candidate_keys = operation.get("candidate_keys")
        payload = operation.get("payload")
        if not isinstance(candidate_keys, list) or len(candidate_keys) != 1:
            raise OptimizerError("local grade operation has invalid candidate identity")
        candidate_key = str(candidate_keys[0])
        if candidate_key in seen_candidates:
            raise OptimizerError(f"local judgments grade {candidate_key!r} more than once")
        seen_candidates.add(candidate_key)
        if not isinstance(payload, Mapping):
            raise OptimizerError("local grade operation has no payload")
        scores = payload.get("scores")
        if not isinstance(scores, Mapping) or reward_key not in scores:
            raise OptimizerError(
                f"local grade {candidate_key!r} has no reward dimension {reward_key!r}"
            )
        confidence = _finite(payload.get("confidence"), f"{candidate_key}.confidence")
        if confidence < min_confidence:
            skipped_confidence += 1
            continue
        generation_id = str(payload.get("generation_id") or "")
        teacher = payload.get("teacher")
        evaluator = str(teacher.get("id")) if isinstance(teacher, Mapping) else "local"
        observation_id = str(operation.get("submission_key") or candidate_key)
        observations.setdefault(generation_id, []).append(
            RewardObservation(
                generation_id=generation_id,
                score=_finite(scores[reward_key], f"{candidate_key}.{reward_key}"),
                confidence=confidence,
                evaluator=evaluator,
                observation_id=observation_id,
            )
        )
    summary = {
        "kind": "local_judgments",
        "path": str(path),
        "sha256": _sha256_file(path),
        "grade_observations": sum(len(rows) for rows in observations.values()),
        "preferences_ignored": preference_count,
        "below_confidence_ignored": skipped_confidence,
    }
    return observations, summary


def _committed_observations(
    database_path: Path,
    candidates: Mapping[str, Mapping[str, object]],
    *,
    reward_key: str,
    min_confidence: float,
) -> tuple[dict[str, list[RewardObservation]], dict[str, object]]:
    path = database_path.expanduser().resolve()
    if not path.is_file():
        raise OptimizerError(f"feedback database does not exist: {path}")
    generation_ids = sorted(
        {
            str(row["generation_id"])
            for row in candidates.values()
            if isinstance(row.get("generation_id"), str)
        }
    )
    latest: dict[tuple[str, str, str], sqlite3.Row] = {}
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        for start in range(0, len(generation_ids), 500):
            batch = generation_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                "SELECT rowid, annotation_id, generation_id, teacher_id, teacher_model, "
                "created_at, overall_score, dimensions_json, confidence "
                f"FROM teacher_annotations WHERE generation_id IN ({placeholders}) "
                "ORDER BY created_at ASC, rowid ASC",
                batch,
            ).fetchall()
            for row in rows:
                key = (
                    str(row["generation_id"]),
                    str(row["teacher_id"]),
                    str(row["teacher_model"] or ""),
                )
                latest[key] = row
    except sqlite3.Error as exc:
        raise OptimizerError(f"cannot read committed judgments from {path}: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    observations: dict[str, list[RewardObservation]] = {}
    skipped_confidence = 0
    missing_dimension = 0
    normalized_for_fingerprint: list[dict[str, object]] = []
    for row in latest.values():
        confidence = _finite(row["confidence"], "teacher annotation confidence")
        if confidence < min_confidence:
            skipped_confidence += 1
            continue
        if reward_key == "overall":
            raw_score = row["overall_score"]
        else:
            try:
                dimensions = json.loads(str(row["dimensions_json"]))
            except json.JSONDecodeError as exc:
                raise OptimizerError("committed annotation has invalid dimensions JSON") from exc
            if not isinstance(dimensions, dict) or reward_key not in dimensions:
                missing_dimension += 1
                continue
            raw_score = dimensions[reward_key]
        observation = RewardObservation(
            generation_id=str(row["generation_id"]),
            score=_finite(raw_score, f"annotation {row['annotation_id']} score"),
            confidence=confidence,
            evaluator=str(row["teacher_id"]),
            observation_id=str(row["annotation_id"]),
        )
        observations.setdefault(observation.generation_id, []).append(observation)
        normalized_for_fingerprint.append(
            {
                "generation_id": observation.generation_id,
                "score": observation.score,
                "confidence": observation.confidence,
                "evaluator": observation.evaluator,
                "observation_id": observation.observation_id,
            }
        )
    summary = {
        "kind": "committed_feedback_database",
        "path": str(path),
        "latest_per_evaluator": len(latest),
        "grade_observations": sum(len(rows) for rows in observations.values()),
        "below_confidence_ignored": skipped_confidence,
        "missing_reward_dimension_ignored": missing_dimension,
        "query_fingerprint": _fingerprint(sorted(normalized_for_fingerprint, key=_canonical_json)),
    }
    return observations, summary


def _aggregate_observations(rows: Sequence[RewardObservation]) -> tuple[float, float]:
    weights = [row.confidence for row in rows]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        weights = [1.0] * len(rows)
        weight_sum = float(len(rows))
    score = sum(row.score * weight for row, weight in zip(rows, weights, strict=True)) / weight_sum
    return score, sum(row.confidence for row in rows) / len(rows)


def _prepare_rewards(
    *,
    manifest_path: Path,
    report_path: Path | None,
    parent: ParentContract,
    judgments_path: Path | None,
    feedback_db: Path | None,
    reward_key: str,
    min_confidence: float,
    min_annotations_per_route: int,
    seed: int,
    heldout_fraction: float,
    strict: bool,
) -> PreparedRewards:
    try:
        manifest = evaluation.load_manifest(manifest_path)
    except evaluation.EvalError as exc:
        raise OptimizerError(str(exc)) from exc
    _validate_manifest_route_contract(manifest, parent)
    report = (report_path or manifest.raw_report).expanduser().resolve()
    if not report.is_file():
        raise OptimizerError(f"raw rollout report does not exist: {report}")
    try:
        events = evaluation._read_events(report, "raw report")
        artifact_provenance = _collection_provenance(events, manifest, parent)
        candidates, forced, ignored_auto_candidates = _load_candidates(manifest, report)
    except evaluation.EvalError as exc:
        raise OptimizerError(str(exc)) from exc
    if (judgments_path is None) == (feedback_db is None):
        raise OptimizerError("select exactly one of local judgments or committed feedback DB")
    try:
        if judgments_path is not None:
            observations, judgment_summary = _local_observations(
                manifest,
                judgments_path,
                candidates,
                reward_key=reward_key,
                min_confidence=min_confidence,
            )
        else:
            assert feedback_db is not None
            observations, judgment_summary = _committed_observations(
                feedback_db,
                candidates,
                reward_key=reward_key,
                min_confidence=min_confidence,
            )
    except evaluation.EvalError as exc:
        raise OptimizerError(str(exc)) from exc

    split = _split_prompts(manifest.prompts, seed=seed, heldout_fraction=heldout_fraction)
    ignored_auto_judgments = sum(
        len(observations.get(str(row.get("generation_id")), ()))
        for row in candidates.values()
        if row.get("route_id") == manifest.auto_route_id
    )
    missing: list[dict[str, object]] = []
    examples: list[RewardExample] = []
    for prompt in manifest.prompts:
        repetition_vectors: list[list[float]] = []
        annotation_totals = [0] * len(parent.route_ids)
        confidence_rows: list[list[float]] = [[] for _ in parent.route_ids]
        prompt_complete = True
        for repetition in range(manifest.repetitions):
            vector: list[float] = []
            for route_index, route_id in enumerate(parent.route_ids):
                rollout_rows = forced.get((prompt.prompt_id, repetition, route_id), [])
                if not rollout_rows:
                    missing.append(
                        {
                            "prompt_id": prompt.prompt_id,
                            "repetition": repetition,
                            "route_id": route_id,
                            "reason": "missing_forced_rollout",
                        }
                    )
                    prompt_complete = False
                    continue
                grades_by_id = {
                    observation.observation_id: observation
                    for row in rollout_rows
                    for observation in observations.get(str(row.get("generation_id")), ())
                }
                grades = list(grades_by_id.values())
                if len(grades) < min_annotations_per_route:
                    missing.append(
                        {
                            "prompt_id": prompt.prompt_id,
                            "repetition": repetition,
                            "route_id": route_id,
                            "reason": "insufficient_judgments",
                            "found": len(grades),
                            "required": min_annotations_per_route,
                        }
                    )
                    prompt_complete = False
                    continue
                score, mean_confidence = _aggregate_observations(grades)
                vector.append(score)
                annotation_totals[route_index] += len(grades)
                confidence_rows[route_index].append(mean_confidence)
            if len(vector) == len(parent.route_ids):
                repetition_vectors.append(vector)
        if prompt_complete and len(repetition_vectors) == manifest.repetitions:
            rewards = tuple(
                sum(vector[index] for vector in repetition_vectors) / len(repetition_vectors)
                for index in range(len(parent.route_ids))
            )
            mean_confidence = tuple(
                sum(rows) / len(rows) if rows else 0.0 for rows in confidence_rows
            )
            examples.append(
                RewardExample(
                    prompt_id=prompt.prompt_id,
                    prompt=_prompt_text(prompt),
                    category=_prompt_category(prompt),
                    rewards=rewards,
                    annotations=tuple(annotation_totals),
                    mean_confidence=mean_confidence,
                )
            )

    if strict and missing:
        preview = ", ".join(
            f"{row['prompt_id']}/{row['route_id']}:{row['reason']}" for row in missing[:8]
        )
        raise OptimizerError(
            f"full-information training requires every forced reward cell; "
            f"{len(missing)} are unavailable ({preview})"
        )
    if strict and {example.prompt_id for example in examples} != {
        prompt.prompt_id for prompt in manifest.prompts
    }:
        raise OptimizerError("not every manifest prompt has a complete reward vector")

    dataset_payload = {
        "manifest_sha256": manifest.fingerprint,
        "raw_report_sha256": _sha256_file(report),
        "judgments": judgment_summary,
        "reward_key": reward_key,
        "min_confidence": min_confidence,
        "min_annotations_per_route": min_annotations_per_route,
        "route_ids": list(parent.route_ids),
        "split_fingerprint": split.fingerprint,
        "examples": [
            {
                "prompt_id": row.prompt_id,
                "prompt_sha256": _fingerprint(row.prompt),
                "rewards": list(row.rewards),
                "annotations": list(row.annotations),
            }
            for row in examples
        ],
    }
    source_summary = {
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.fingerprint,
            "run_id": manifest.run_id,
        },
        "raw_report": {
            "path": str(report),
            "sha256": dataset_payload["raw_report_sha256"],
            "candidate_generations": len(candidates),
            "auto_candidates_ignored": ignored_auto_candidates,
            "artifact_provenance": artifact_provenance,
        },
        "judgments": {
            **judgment_summary,
            "auto_judgments_ignored": ignored_auto_judgments,
        },
    }
    return PreparedRewards(
        manifest=manifest,
        route_ids=parent.route_ids,
        examples=tuple(examples),
        split=split,
        missing_cells=tuple(missing),
        source_summary=source_summary,
        dataset_fingerprint=_fingerprint(dataset_payload),
    )


def _reward_summary(data: PreparedRewards) -> dict[str, object]:
    rows = data.examples
    per_route: dict[str, object] = {}
    for index, route_id in enumerate(data.route_ids):
        values = [row.rewards[index] for row in rows]
        per_route[route_id] = {
            "mean": sum(values) / len(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "oracle_wins_including_ties": sum(
                math.isclose(row.rewards[index], max(row.rewards), abs_tol=1e-9) for row in rows
            ),
            "annotations": sum(row.annotations[index] for row in rows),
        }
    return {"complete_vectors": len(rows), "per_route": per_route}


def _plan_payload(
    data: PreparedRewards,
    parent: ParentContract,
    *,
    output: Path,
    beta: float,
    seed: int,
    heldout_fraction: float,
) -> dict[str, object]:
    complete_ids = {row.prompt_id for row in data.examples}
    expected_ids = {prompt.prompt_id for prompt in data.manifest.prompts}
    resolved_output = output.expanduser().resolve()
    output_safe = (
        not resolved_output.exists()
        and resolved_output != parent.directory
        and parent.directory not in resolved_output.parents
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "mode": "plan",
        "method": METHOD,
        "ready_to_train": (
            not data.missing_cells and complete_ids == expected_ids and output_safe
        ),
        "objective": "E_pi[reward] - beta * KL(pi || pi0)",
        "beta": beta,
        "seed": seed,
        "heldout_fraction": heldout_fraction,
        "initial_router": {
            "directory": str(parent.directory),
            "config": _artifact(parent.config_path),
            "weights": _artifact(parent.weights_path),
        },
        "output": {
            "directory": str(resolved_output),
            "exists": resolved_output.exists(),
            "safe_new_destination": output_safe,
            "will_never_overwrite": True,
        },
        "route_ids": list(data.route_ids),
        "sources": data.source_summary,
        "dataset_fingerprint": data.dataset_fingerprint,
        "split": {
            "method": data.split.method,
            "fingerprint": data.split.fingerprint,
            "train_ids": list(data.split.train_ids),
            "heldout_ids": list(data.split.heldout_ids),
        },
        "rewards": _reward_summary(data),
        "missing_cells": list(data.missing_cells),
    }


def _load_parent_router(parent: ParentContract) -> TopKRouter:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise OptimizerError('install molor[hf] to train the reward router') from exc
    tensors = load_file(str(parent.weights_path), device="cpu")
    if not tensors or any(not name.startswith("network.") for name in tensors):
        raise OptimizerError("initial router weights must contain only network.* tensors")
    state = {name.removeprefix("network."): tensor for name, tensor in tensors.items()}
    router = TopKRouter(parent.router_config)
    try:
        router.network.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise OptimizerError(f"initial router weights do not match its config: {exc}") from exc
    router.requires_grad_(False)
    router.eval()
    return router


def _extract_features(
    data: PreparedRewards,
    parent: ParentContract,
    *,
    model_dir: Path | None,
    feature_batch_size: int,
) -> tuple[Tensor, dict[str, object]]:
    raw_contract = parent.sidecar.get("feature_contract")
    if not isinstance(raw_contract, dict):
        raise OptimizerError("initial router has no feature_contract")
    try:
        feature_contract = RouterFeatureContract(**raw_contract)
    except (TypeError, ValueError) as exc:
        raise OptimizerError(f"invalid feature contract: {exc}") from exc
    model_metadata = parent.sidecar.get("model")
    if not isinstance(model_metadata, Mapping):
        raise OptimizerError("initial router has no frozen-base model metadata")
    configured_model = model_dir or Path(str(model_metadata.get("model_id") or ""))
    model_root = configured_model.expanduser().resolve()
    print(f"loading frozen input embedding only: {model_root}", file=sys.stderr, flush=True)
    embedding, embedding_metadata = load_input_embedding(
        model_root,
        tensor_name=feature_contract.embedding_tensor,
    )
    expected_embedding = model_metadata.get("embedding")
    expected_artifact = (
        expected_embedding.get("artifact") if isinstance(expected_embedding, Mapping) else None
    )
    expected_sha = expected_artifact.get("sha256") if isinstance(expected_artifact, Mapping) else None
    actual_artifact = embedding_metadata.get("artifact")
    actual_sha = actual_artifact.get("sha256") if isinstance(actual_artifact, Mapping) else None
    if isinstance(expected_sha, str) and actual_sha != expected_sha:
        raise OptimizerError("frozen input embedding differs from the initial router contract")
    tokenizer = load_router_tokenizer(model_root)
    dummy_route = parent.route_ids[-1]
    router_examples = [RouterExample(row.prompt, dummy_route) for row in data.examples]
    features, _labels = extract_prompt_features(
        router_examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=parent.route_ids,
        contract=feature_contract,
        batch_size=feature_batch_size,
        progress=lambda text: print(text, file=sys.stderr, flush=True),
    )
    del embedding
    return features.float().contiguous(), {
        "model_dir": str(model_root),
        "embedding": embedding_metadata,
        "feature_contract": feature_contract.to_dict(),
    }


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise OptimizerError("CUDA was requested but is unavailable")
    return device


def _log_policy(router: TopKRouter, features: Tensor) -> Tensor:
    logits = router.network(features) / router.config.temperature
    return F.log_softmax(logits.float(), dim=-1)


@torch.inference_mode()
def _policy_metrics(
    router: TopKRouter,
    baseline: TopKRouter,
    features: Tensor,
    rewards: Tensor,
    *,
    route_ids: Sequence[str],
    beta: float,
    device: torch.device,
) -> dict[str, object]:
    router.eval()
    baseline.eval()
    feature_rows = features.to(device)
    reward_rows = rewards.to(device)
    log_policy = _log_policy(router, feature_rows)
    log_baseline = _log_policy(baseline, feature_rows)
    policy = log_policy.exp()
    baseline_policy = log_baseline.exp()
    expected = (policy * reward_rows).sum(dim=-1)
    baseline_expected = (baseline_policy * reward_rows).sum(dim=-1)
    kl = (policy * (log_policy - log_baseline)).sum(dim=-1)
    oracle = reward_rows.max(dim=-1).values
    hard_index = policy.argmax(dim=-1)
    baseline_hard_index = baseline_policy.argmax(dim=-1)
    hard_reward = reward_rows.gather(1, hard_index.unsqueeze(-1)).squeeze(-1)
    baseline_hard_reward = reward_rows.gather(
        1, baseline_hard_index.unsqueeze(-1)
    ).squeeze(-1)
    changed = hard_index.ne(baseline_hard_index).float()
    total_variation = 0.5 * (policy - baseline_policy).abs().sum(dim=-1)
    hard_usage = torch.bincount(hard_index, minlength=len(route_ids))
    return {
        "examples": int(features.shape[0]),
        "expected_reward": float(expected.mean().item()),
        "baseline_expected_reward": float(baseline_expected.mean().item()),
        "expected_reward_gain": float((expected - baseline_expected).mean().item()),
        "mean_kl_to_initial": float(kl.mean().item()),
        "regularized_objective": float((expected - beta * kl).mean().item()),
        "oracle_reward": float(oracle.mean().item()),
        "oracle_regret": float((oracle - expected).mean().item()),
        "hard_top1_reward": float(hard_reward.mean().item()),
        "baseline_hard_top1_reward": float(baseline_hard_reward.mean().item()),
        "hard_top1_regret": float((oracle - hard_reward).mean().item()),
        "route_changed_fraction": float(changed.mean().item()),
        "mean_total_variation": float(total_variation.mean().item()),
        "mean_probabilities": {
            route_id: float(policy[:, index].mean().item())
            for index, route_id in enumerate(route_ids)
        },
        "hard_route_counts": {
            route_id: int(hard_usage[index].item())
            for index, route_id in enumerate(route_ids)
        },
    }


def _fit_reward_router(
    parent_router: TopKRouter,
    train_features: Tensor,
    train_rewards: Tensor,
    heldout_features: Tensor,
    heldout_rewards: Tensor,
    *,
    route_ids: Sequence[str],
    beta: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    patience: int,
    min_delta: float,
    seed: int,
    device_name: str,
) -> tuple[TopKRouter, dict[str, object]]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = _device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    baseline = copy.deepcopy(parent_router).to(device)
    baseline.requires_grad_(False)
    baseline.eval()
    router = copy.deepcopy(parent_router).to(device)
    router.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        router.network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    with torch.inference_mode():
        train_log_baseline = _log_policy(baseline, train_features.to(device)).cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_rewards, train_log_baseline),
        batch_size=min(batch_size, len(train_features)),
        shuffle=True,
        generator=generator,
    )

    baseline_metrics = {
        "train": _policy_metrics(
            baseline,
            baseline,
            train_features,
            train_rewards,
            route_ids=route_ids,
            beta=beta,
            device=device,
        ),
        "heldout": _policy_metrics(
            baseline,
            baseline,
            heldout_features,
            heldout_rewards,
            route_ids=route_ids,
            beta=beta,
            device=device,
        ),
    }
    # Serving uses hard top-1 routing, so an RL checkpoint is not allowed to win
    # early stopping merely by rearranging probability mass while leaving (or
    # worsening) the route that will actually be served.  Epoch zero is a real
    # candidate: if no update improves held-out hard-route reward, preserve the
    # parent router byte-for-byte.
    best_hard_reward = float(baseline_metrics["heldout"]["hard_top1_reward"])
    best_objective = float(baseline_metrics["heldout"]["regularized_objective"])
    best_epoch = 0
    best_state = {
        name: tensor.detach().cpu().clone() for name, tensor in router.network.state_dict().items()
    }
    history: list[dict[str, object]] = []
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        router.train()
        objective_sum = 0.0
        reward_sum = 0.0
        kl_sum = 0.0
        seen = 0
        for batch_features, batch_rewards, batch_log_baseline in loader:
            feature_rows = batch_features.to(device)
            reward_rows = batch_rewards.to(device)
            baseline_rows = batch_log_baseline.to(device)
            log_policy = _log_policy(router, feature_rows)
            policy = log_policy.exp()
            expected_reward = (policy * reward_rows).sum(dim=-1)
            kl = (policy * (log_policy - baseline_rows)).sum(dim=-1)
            objective = expected_reward - beta * kl
            loss = -objective.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.network.parameters(), max_grad_norm)
            optimizer.step()
            count = int(feature_rows.shape[0])
            seen += count
            objective_sum += float(objective.detach().sum().item())
            reward_sum += float(expected_reward.detach().sum().item())
            kl_sum += float(kl.detach().sum().item())

        heldout = _policy_metrics(
            router,
            baseline,
            heldout_features,
            heldout_rewards,
            route_ids=route_ids,
            beta=beta,
            device=device,
        )
        epoch_row = {
            "epoch": epoch,
            "train_objective": objective_sum / max(seen, 1),
            "train_expected_reward": reward_sum / max(seen, 1),
            "train_kl": kl_sum / max(seen, 1),
            "heldout_objective": heldout["regularized_objective"],
            "heldout_expected_reward": heldout["expected_reward"],
            "heldout_kl": heldout["mean_kl_to_initial"],
            "heldout_oracle_regret": heldout["oracle_regret"],
            "heldout_hard_top1_reward": heldout["hard_top1_reward"],
            "heldout_hard_top1_regret": heldout["hard_top1_regret"],
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch}/{epochs} train objective {epoch_row['train_objective']:.5f} "
            f"heldout objective {epoch_row['heldout_objective']:.5f} "
            f"KL {epoch_row['heldout_kl']:.5f}",
            file=sys.stderr,
            flush=True,
        )
        hard_reward = float(heldout["hard_top1_reward"])
        objective = float(heldout["regularized_objective"])
        improves_hard_route = hard_reward > best_hard_reward + min_delta
        improves_soft_tiebreak = (
            best_epoch > 0
            and hard_reward >= best_hard_reward - 1e-12
            and objective > best_objective + min_delta
        )
        if improves_hard_route or improves_soft_tiebreak:
            best_hard_reward = hard_reward
            best_objective = objective
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in router.network.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    router.network.load_state_dict(best_state, strict=True)
    router.to(device)
    trained_metrics = {
        "train": _policy_metrics(
            router,
            baseline,
            train_features,
            train_rewards,
            route_ids=route_ids,
            beta=beta,
            device=device,
        ),
        "heldout": _policy_metrics(
            router,
            baseline,
            heldout_features,
            heldout_rewards,
            route_ids=route_ids,
            beta=beta,
            device=device,
        ),
    }
    if float(trained_metrics["heldout"]["hard_top1_reward"]) + 1e-8 < float(
        baseline_metrics["heldout"]["hard_top1_reward"]
    ):
        raise OptimizerError("promotion gate selected a worse heldout hard top-1 policy")
    return router.cpu(), {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stopped_early": len(history) < epochs,
        "selection_metric": "heldout.hard_top1_reward_then_regularized_objective",
        "promotion": {
            "promoted": best_epoch > 0,
            "parent_preserved": best_epoch == 0,
            "required_primary_improvement": "heldout.hard_top1_reward",
        },
        "history": history,
        "baseline": baseline_metrics,
        "trained": trained_metrics,
    }


def _save_checkpoint(
    router: TopKRouter,
    parent: ParentContract,
    data: PreparedRewards,
    report: dict[str, object],
    *,
    output: Path,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise OptimizerError('install molor[hf] to save the reward router') from exc
    destination = output.expanduser().resolve()
    if destination.exists():
        raise OptimizerError(f"refusing to overwrite existing output: {destination}")
    if destination == parent.directory or parent.directory in destination.parents:
        raise OptimizerError("output must not be the initial checkpoint or a child of it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        weights_path = temporary / "router.safetensors"
        tensor_state = {
            f"network.{name}": tensor.detach().cpu().contiguous()
            for name, tensor in router.network.state_dict().items()
        }
        save_file(tensor_state, weights_path)
        sidecar = copy.deepcopy(parent.sidecar)
        sidecar["parent_checkpoint"] = {
            "directory": str(parent.directory),
            "config": _artifact(parent.config_path),
            "weights": _artifact(parent.weights_path),
        }
        sidecar["training_config"] = report["training_config"]
        sidecar["datasets"] = report["sources"]
        sidecar["metrics"] = report["metrics"]
        sidecar["reward_training"] = {
            "method": METHOD,
            "dataset_fingerprint": data.dataset_fingerprint,
            "split": report["split"],
            "reward_vectors": report["reward_vectors"],
            "created_at": report["created_at"],
        }
        sidecar["checkpoint"] = _artifact(
            weights_path,
            recorded_path=destination / "router.safetensors",
        )
        config_path = temporary / "router_config.json"
        config_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise OptimizerError(f"output appeared during training; refusing overwrite: {destination}")
        temporary.rename(destination)
    return destination


def _training_report(
    data: PreparedRewards,
    parent: ParentContract,
    features_metadata: Mapping[str, object],
    metrics: Mapping[str, object],
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    train_set = set(data.split.train_ids)
    return {
        "schema_version": REPORT_SCHEMA,
        "created_at": _now(),
        "method": METHOD,
        "objective": "E_pi[reward] - beta * KL(pi || pi0)",
        "frozen_components": ["causal_language_model", "all_lora_experts"],
        "optimized_component": "prompt_router_only",
        "route_ids": list(data.route_ids),
        "parent_checkpoint": {
            "directory": str(parent.directory),
            "config": _artifact(parent.config_path),
            "weights": _artifact(parent.weights_path),
        },
        "output_directory": str(args.output.expanduser().resolve()),
        "sources": data.source_summary,
        "dataset_fingerprint": data.dataset_fingerprint,
        "feature_extraction": dict(features_metadata),
        "training_config": {
            "method": METHOD,
            "beta": args.beta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": args.seed,
            "device": args.device,
            "reward_key": args.reward_key,
            "min_confidence": args.min_confidence,
            "min_annotations_per_route": args.min_annotations_per_route,
        },
        "split": {
            "method": data.split.method,
            "fingerprint": data.split.fingerprint,
            "train_ids": list(data.split.train_ids),
            "heldout_ids": list(data.split.heldout_ids),
            "overlap": [],
        },
        "reward_vectors": [
            {
                "prompt_id": row.prompt_id,
                "split": "train" if row.prompt_id in train_set else "heldout",
                "category": row.category,
                "rewards": dict(zip(data.route_ids, row.rewards, strict=True)),
                "annotations": dict(zip(data.route_ids, row.annotations, strict=True)),
                "mean_confidence": dict(
                    zip(data.route_ids, row.mean_confidence, strict=True)
                ),
            }
            for row in data.examples
        ],
        "metrics": dict(metrics),
    }


def _validate_training_args(args: argparse.Namespace) -> None:
    positive_floats = ("beta", "learning_rate", "max_grad_norm")
    for name in positive_floats:
        value = _finite(getattr(args, name), name)
        if value <= 0:
            raise OptimizerError(f"{name} must be greater than zero")
    if not 0 <= args.weight_decay:
        raise OptimizerError("weight_decay must be non-negative")
    if not 0 <= args.min_delta:
        raise OptimizerError("min_delta must be non-negative")
    if not 0 <= args.min_confidence <= 1:
        raise OptimizerError("min_confidence must be from zero to one")
    for name in (
        "epochs",
        "batch_size",
        "patience",
        "feature_batch_size",
        "min_annotations_per_route",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise OptimizerError(f"{name} must be a positive integer")


def plan(args: argparse.Namespace) -> int:
    if args.beta <= 0:
        raise OptimizerError("beta must be greater than zero")
    parent = _load_parent_contract(args.initial_router)
    data = _prepare_rewards(
        manifest_path=args.manifest,
        report_path=args.report,
        parent=parent,
        judgments_path=args.judgments,
        feedback_db=args.feedback_db,
        reward_key=args.reward_key,
        min_confidence=args.min_confidence,
        min_annotations_per_route=args.min_annotations_per_route,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
        strict=False,
    )
    payload = _plan_payload(
        data,
        parent,
        output=args.output,
        beta=args.beta,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def train(args: argparse.Namespace) -> int:
    _validate_training_args(args)
    parent = _load_parent_contract(args.initial_router)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise OptimizerError(f"refusing to overwrite existing output: {output}")
    data = _prepare_rewards(
        manifest_path=args.manifest,
        report_path=args.report,
        parent=parent,
        judgments_path=args.judgments,
        feedback_db=args.feedback_db,
        reward_key=args.reward_key,
        min_confidence=args.min_confidence,
        min_annotations_per_route=args.min_annotations_per_route,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
        strict=True,
    )
    features, feature_metadata = _extract_features(
        data,
        parent,
        model_dir=args.model_dir,
        feature_batch_size=args.feature_batch_size,
    )
    rewards = torch.tensor([row.rewards for row in data.examples], dtype=torch.float32)
    train_ids = set(data.split.train_ids)
    heldout_ids = set(data.split.heldout_ids)
    train_indices = [
        index for index, row in enumerate(data.examples) if row.prompt_id in train_ids
    ]
    heldout_indices = [
        index for index, row in enumerate(data.examples) if row.prompt_id in heldout_ids
    ]
    if set(train_indices).intersection(heldout_indices):
        raise OptimizerError("feature rows leaked across train and heldout splits")
    if len(train_indices) + len(heldout_indices) != len(data.examples):
        raise OptimizerError("not every feature row belongs to exactly one split")
    parent_router = _load_parent_router(parent)
    router, metrics = _fit_reward_router(
        parent_router,
        features[train_indices],
        rewards[train_indices],
        features[heldout_indices],
        rewards[heldout_indices],
        route_ids=data.route_ids,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device_name=args.device,
    )
    report = _training_report(
        data,
        parent,
        feature_metadata,
        metrics,
        args=args,
    )
    destination = _save_checkpoint(router, parent, data, report, output=output)
    summary = {
        "output": str(destination),
        "checkpoint": str(destination / "router.safetensors"),
        "config": str(destination / "router_config.json"),
        "report": str(destination / "training_report.json"),
        "best_epoch": metrics["best_epoch"],
        "heldout": metrics["trained"]["heldout"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    common.add_argument(
        "--report",
        type=Path,
        help="Raw JSONL report; defaults to manifest.outputs.raw_report.",
    )
    common.add_argument("--initial-router", type=Path, default=DEFAULT_INITIAL_ROUTER)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    source = common.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--judgments",
        type=Path,
        help="Local molor.teacher-eval.judgments.v1 JSON before submission.",
    )
    source.add_argument(
        "--feedback-db",
        type=Path,
        help="SQLite feedback DB containing committed teacher annotations.",
    )
    common.add_argument("--reward-key", default="overall")
    common.add_argument("--min-confidence", type=float, default=0.0)
    common.add_argument("--min-annotations-per-route", type=int, default=1)
    common.add_argument("--seed", type=int, default=8419)
    common.add_argument("--heldout-fraction", type=float, default=0.2)
    common.add_argument("--beta", type=float, default=0.5)
    return common


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    subparsers.add_parser(
        "plan",
        parents=[common],
        help="Validate reward-vector coverage and print the deterministic split; no model load.",
    )
    trainer = subparsers.add_parser(
        "train",
        parents=[common],
        help="Optimize and save a new conservative reward-router checkpoint.",
    )
    trainer.add_argument("--model-dir", type=Path)
    trainer.add_argument("--epochs", type=int, default=200)
    trainer.add_argument("--batch-size", type=int, default=32)
    trainer.add_argument("--feature-batch-size", type=int, default=64)
    trainer.add_argument("--learning-rate", type=float, default=2e-4)
    trainer.add_argument("--weight-decay", type=float, default=1e-4)
    trainer.add_argument("--max-grad-norm", type=float, default=1.0)
    trainer.add_argument("--patience", type=int, default=20)
    trainer.add_argument("--min-delta", type=float, default=1e-5)
    trainer.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            if not 0 <= args.min_confidence <= 1:
                raise OptimizerError("min_confidence must be from zero to one")
            if args.min_annotations_per_route < 1:
                raise OptimizerError("min_annotations_per_route must be positive")
            return plan(args)
        if args.command == "train":
            return train(args)
        raise AssertionError("unreachable")
    except (OptimizerError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
