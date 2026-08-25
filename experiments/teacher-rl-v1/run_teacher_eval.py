"""Resumable forced-route and auto-route evaluation for the local teacher API.

This tool deliberately separates three actions:

* ``collect`` asks the local student for outputs and appends raw API responses;
* ``review-pack`` creates a portable, read-only packet for an external teacher;
* ``submit`` validates and submits only judgments explicitly present in a JSON file.

It never invents a score, correction, or preference.  ``submit`` is a preview unless
``--commit`` is supplied, and every committed operation has an append-only receipt so
the command can be resumed safely.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from molor.benchmark.lock import lock_backend_identity


SCHEMA_MANIFEST = "molor.teacher-eval.manifest.v1"
SCHEMA_RAW = "molor.teacher-eval.raw.v1"
SCHEMA_REVIEW = "molor.teacher-eval.review-pack.v1"
SCHEMA_JUDGMENTS = "molor.teacher-eval.judgments.v1"
SCHEMA_SUBMISSION = "molor.teacher-eval.submission.v1"
SCHEMA_ARTIFACT_CONTRACT = "molor.teacher-eval.student-artifact-contract.v1"
EXACT_ARTIFACT_PROVENANCE = "exact_local_artifact_bytes"
LEGACY_ARTIFACT_PROVENANCE = "legacy_path_identity_only"
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
RUBRIC_DIMENSIONS = {
    "accuracy",
    "instruction_following",
    "reasoning",
    "relevance",
    "style",
    "calibration",
}


class EvalError(RuntimeError):
    """An actionable manifest, report, API, or judgment error."""


class ApiError(EvalError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


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


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_directory(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalError(f"{label} must be a non-empty local directory path")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise EvalError(f"{label} is not an available local directory: {path}")
    return str(path)


def _string_array(value: object, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvalError(f"{label} must be an array")
    result = list(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise EvalError(f"{label} must contain only non-empty strings")
    if len(result) != len(set(result)):
        raise EvalError(f"{label} must not contain duplicates")
    return result


def _student_health_identity(health: Mapping[str, object]) -> dict[str, object]:
    """Extract the strict, path-bound runtime identity used by collection provenance."""

    if health.get("status") != "ok" or health.get("protocol_version") != "molor.teacher-api.v1":
        raise EvalError("teacher health does not advertise the expected healthy v1 protocol")
    if health.get("model_loaded") is not True:
        raise EvalError("student must be loaded before artifact provenance can be recorded")
    if health.get("router_loaded") is not True:
        raise EvalError("student router must be loaded before artifact provenance can be recorded")

    experts = _string_array(health.get("experts"), "health.experts")
    expert_paths = _string_array(
        health.get("configured_expert_paths"), "health.configured_expert_paths"
    )
    if len(experts) != len(expert_paths):
        raise EvalError("health expert names and configured paths must have the same length")
    resolved_experts = [
        _local_directory(path, f"health.configured_expert_paths[{index}]")
        for index, path in enumerate(expert_paths)
    ]
    return {
        "model_id": _local_directory(health.get("model_id"), "health.model_id"),
        "experts": experts,
        "configured_expert_paths": resolved_experts,
        "router_checkpoint": _local_directory(
            health.get("router_checkpoint"), "health.router_checkpoint"
        ),
        "model_loaded": True,
        "router_loaded": True,
    }


def _build_student_artifact_contract(
    health: Mapping[str, object],
    *,
    show_progress: bool = True,
) -> dict[str, object]:
    """Bind one loaded local student stack to the bytes used for new rollouts."""

    health_identity = _student_health_identity(health)
    progress_context = nullcontext() if show_progress else redirect_stdout(io.StringIO())
    with progress_context:
        locked = lock_backend_identity(
            health_identity,
            # The benchmark locker deliberately performs a fresh full read and ignores this path.
            cache_path=Path(__file__).with_name(".teacher-artifact-lock-cache.json"),
        )
    if locked.get("fully_content_addressed") is not True:
        unresolved = locked.get("unresolved")
        raise EvalError(f"student artifact stack is not fully content-addressed: {unresolved}")
    artifacts = locked.get("artifacts")
    if not isinstance(artifacts, list):
        raise EvalError("student artifact locker returned no artifact list")
    roles = [item.get("role") for item in artifacts if isinstance(item, Mapping)]
    if roles.count("base") != 1 or roles.count("router") != 1 or roles.count("expert") != len(
        health_identity["experts"]
    ):
        raise EvalError("student artifact stack does not match the loaded base/expert/router rack")

    payload: dict[str, object] = {
        "schema_version": SCHEMA_ARTIFACT_CONTRACT,
        "hash_policy": locked.get("hash_policy"),
        "health_identity": health_identity,
        "artifacts": artifacts,
        "stack_fingerprint": locked.get("stack_fingerprint"),
    }
    return {**payload, "contract_sha256": _fingerprint(payload)}


def _validate_student_artifact_contract(value: object) -> dict[str, object]:
    """Validate the self-fingerprint of a serialized collection contract."""

    if not isinstance(value, Mapping):
        raise EvalError("collection session has no student artifact contract")
    contract = dict(value)
    supplied = contract.pop("contract_sha256", None)
    if contract.get("schema_version") != SCHEMA_ARTIFACT_CONTRACT:
        raise EvalError("collection session uses an unsupported artifact contract schema")
    expected = _fingerprint(contract)
    if supplied != expected:
        raise EvalError("collection session artifact contract fingerprint is invalid")
    return {**contract, "contract_sha256": expected}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalError(f"cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvalError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalError(f"{label} must contain one JSON object")
    return value


def _require_keys(value: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise EvalError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise EvalError(f"{label} must match {ID_PATTERN.pattern}")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvalError(f"{label} must be a finite number")
    return result


def _positive_int(value: object, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvalError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise EvalError(f"{label} must be at most {maximum}")
    return value


def _messages(value: object, label: str) -> tuple[dict[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise EvalError(f"{label} must be a non-empty message array")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise EvalError(f"{label}[{index}] needs exactly role and content")
        role, content = item["role"], item["content"]
        if role not in MESSAGE_ROLES:
            raise EvalError(f"{label}[{index}].role is unsupported")
        if not isinstance(content, str) or not content.strip():
            raise EvalError(f"{label}[{index}].content must be non-empty")
        result.append({"role": role, "content": content})
    if result[-1]["role"] != "user":
        raise EvalError(f"{label} must end with a user message")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    messages: tuple[dict[str, str], ...]
    tags: tuple[str, ...]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class RouteSpec:
    route_id: str
    runtime_name: str
    legacy_weights: dict[str, float] | None


@dataclass(frozen=True, slots=True)
class EvalManifest:
    path: Path
    raw: dict[str, Any]
    fingerprint: str
    run_id: str
    description: str
    prompts: tuple[PromptSpec, ...]
    forced_routes: tuple[RouteSpec, ...]
    auto_enabled: bool
    auto_route_id: str
    auto_policy: dict[str, object]
    manual_policy_defaults: dict[str, object]
    settings: dict[str, object]
    base_seed: int
    repetitions: int
    candidates_per_route: int
    endpoint: str
    request_timeout_seconds: float
    raw_report: Path
    review_pack: Path
    submission_log: Path


def _resolve_path(manifest_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise EvalError(f"{label} must be a path string")
    return (manifest_path.parent / value).resolve()


def _validate_policy(raw: object, label: str, *, expected_mode: str | None) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise EvalError(f"{label} must be an object")
    allowed = {
        "mode",
        "top_k",
        "temperature",
        "confidence_threshold",
        "null_bias",
        "max_prompt_tokens",
        "enabled_experts",
        "manual_route",
    }
    _require_keys(raw, allowed, label)
    policy = dict(raw)
    mode = policy.get("mode", expected_mode)
    if mode not in {"auto", "manual"} or (expected_mode and mode != expected_mode):
        raise EvalError(f"{label}.mode must be {expected_mode or 'auto or manual'}")
    policy["mode"] = mode
    policy["top_k"] = _positive_int(policy.get("top_k", 1), f"{label}.top_k", 64)
    temperature = _finite(policy.get("temperature", 1.0), f"{label}.temperature")
    if temperature <= 0:
        raise EvalError(f"{label}.temperature must be greater than zero")
    policy["temperature"] = temperature
    confidence = _finite(policy.get("confidence_threshold", 0.0), f"{label}.confidence_threshold")
    if not 0 <= confidence <= 1:
        raise EvalError(f"{label}.confidence_threshold must be from zero to one")
    policy["confidence_threshold"] = confidence
    policy["null_bias"] = _finite(policy.get("null_bias", 0.0), f"{label}.null_bias")
    policy["max_prompt_tokens"] = _positive_int(
        policy.get("max_prompt_tokens", 256), f"{label}.max_prompt_tokens", 65536
    )
    experts = policy.get("enabled_experts", [])
    if isinstance(experts, (str, bytes)) or not isinstance(experts, Sequence):
        raise EvalError(f"{label}.enabled_experts must be an array")
    normalized_experts = [_identifier(item, f"{label}.enabled_experts") for item in experts]
    if len(normalized_experts) != len(set(normalized_experts)):
        raise EvalError(f"{label}.enabled_experts must not contain duplicates")
    policy["enabled_experts"] = normalized_experts
    manual_route = policy.get("manual_route", {})
    if not isinstance(manual_route, Mapping):
        raise EvalError(f"{label}.manual_route must be an object")
    normalized_route: dict[str, float] = {}
    for name, weight in manual_route.items():
        route_name = _identifier(name, f"{label}.manual_route key")
        route_weight = _finite(weight, f"{label}.manual_route[{name!r}]")
        if route_weight < 0:
            raise EvalError(f"{label}.manual_route weights cannot be negative")
        normalized_route[route_name] = route_weight
    policy["manual_route"] = normalized_route
    return policy


def load_manifest(path: Path) -> EvalManifest:
    manifest_path = path.expanduser().resolve()
    raw = _load_object(manifest_path, "manifest")
    _require_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "description",
            "api",
            "generation",
            "routing",
            "outputs",
            "evaluation_contract",
            "prompts",
        },
        "manifest",
    )
    if raw.get("schema_version") != SCHEMA_MANIFEST:
        raise EvalError(f"manifest.schema_version must be {SCHEMA_MANIFEST!r}")
    run_id = _identifier(raw.get("run_id"), "manifest.run_id")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise EvalError("manifest.description must be a string")

    api = raw.get("api")
    if not isinstance(api, Mapping):
        raise EvalError("manifest.api must be an object")
    _require_keys(api, {"endpoint", "request_timeout_seconds"}, "manifest.api")
    endpoint = _loopback_endpoint(str(api.get("endpoint", "")))
    timeout = _finite(api.get("request_timeout_seconds", 3600), "request_timeout_seconds")
    if not 1 <= timeout <= 86400:
        raise EvalError("request_timeout_seconds must be from 1 to 86400")

    generation = raw.get("generation")
    if not isinstance(generation, Mapping):
        raise EvalError("manifest.generation must be an object")
    _require_keys(
        generation,
        {"base_seed", "repetitions", "candidates_per_route", "settings"},
        "manifest.generation",
    )
    base_seed = generation.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise EvalError("manifest.generation.base_seed must be a non-negative integer")
    repetitions = _positive_int(generation.get("repetitions", 1), "repetitions", 100)
    candidates = _positive_int(generation.get("candidates_per_route", 1), "candidates_per_route", 8)
    settings = generation.get("settings")
    if not isinstance(settings, Mapping):
        raise EvalError("manifest.generation.settings must be an object")
    settings = dict(settings)
    if "seed" in settings:
        raise EvalError("put the seed in generation.base_seed, not generation.settings")
    _canonical_json(settings)

    routing = raw.get("routing")
    if not isinstance(routing, Mapping):
        raise EvalError("manifest.routing must be an object")
    _require_keys(
        routing,
        {"forced_routes", "manual_policy_defaults", "auto"},
        "manifest.routing",
    )
    manual_defaults = _validate_policy(
        routing.get("manual_policy_defaults", {}),
        "manifest.routing.manual_policy_defaults",
        expected_mode="manual",
    )
    if manual_defaults["manual_route"]:
        raise EvalError("manual_policy_defaults.manual_route must be empty")

    raw_routes = routing.get("forced_routes")
    if isinstance(raw_routes, (str, bytes)) or not isinstance(raw_routes, Sequence):
        raise EvalError("manifest.routing.forced_routes must be an array")
    routes: list[RouteSpec] = []
    for index, item in enumerate(raw_routes):
        label = f"manifest.routing.forced_routes[{index}]"
        if not isinstance(item, Mapping):
            raise EvalError(f"{label} must be an object")
        _require_keys(item, {"route_id", "runtime_name", "legacy_weights"}, label)
        route_id = _identifier(item.get("route_id"), f"{label}.route_id")
        runtime_name = _identifier(item.get("runtime_name"), f"{label}.runtime_name")
        raw_weights = item.get("legacy_weights")
        legacy_weights: dict[str, float] | None = None
        if raw_weights is not None:
            if not isinstance(raw_weights, Mapping) or set(raw_weights) != {
                "base",
                "expert_a",
                "expert_b",
            }:
                raise EvalError(
                    f"{label}.legacy_weights needs exactly base, expert_a, and expert_b"
                )
            legacy_weights = {}
            for name, weight in raw_weights.items():
                normalized = _finite(weight, f"{label}.legacy_weights.{name}")
                if normalized < 0:
                    raise EvalError(f"{label}.legacy_weights cannot be negative")
                legacy_weights[name] = normalized
            if sum(legacy_weights.values()) <= 0:
                raise EvalError(f"{label}.legacy_weights must have positive total weight")
        routes.append(RouteSpec(route_id, runtime_name, legacy_weights))
    route_ids = [route.route_id for route in routes]
    if not routes or len(route_ids) != len(set(route_ids)):
        raise EvalError("forced route IDs must be non-empty and unique")

    auto = routing.get("auto")
    if not isinstance(auto, Mapping):
        raise EvalError("manifest.routing.auto must be an object")
    _require_keys(auto, {"enabled", "route_id", "policy"}, "manifest.routing.auto")
    auto_enabled = auto.get("enabled", True)
    if not isinstance(auto_enabled, bool):
        raise EvalError("manifest.routing.auto.enabled must be boolean")
    auto_route_id = _identifier(auto.get("route_id", "auto"), "manifest.routing.auto.route_id")
    if auto_route_id in set(route_ids):
        raise EvalError("auto route_id must not collide with a forced route")
    auto_policy = _validate_policy(
        auto.get("policy", {}), "manifest.routing.auto.policy", expected_mode="auto"
    )

    raw_prompts = raw.get("prompts")
    if isinstance(raw_prompts, (str, bytes)) or not isinstance(raw_prompts, Sequence):
        raise EvalError("manifest.prompts must be an array")
    prompts: list[PromptSpec] = []
    for index, item in enumerate(raw_prompts):
        label = f"manifest.prompts[{index}]"
        if not isinstance(item, Mapping):
            raise EvalError(f"{label} must be an object")
        _require_keys(item, {"prompt_id", "prompt", "messages", "tags", "metadata"}, label)
        prompt_id = _identifier(item.get("prompt_id"), f"{label}.prompt_id")
        prompt = item.get("prompt")
        raw_messages = item.get("messages")
        if (prompt is None) == (raw_messages is None):
            raise EvalError(f"{label} needs exactly one of prompt or messages")
        if prompt is not None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise EvalError(f"{label}.prompt must be non-empty")
            prompt_messages = ({"role": "user", "content": prompt.strip()},)
        else:
            prompt_messages = _messages(raw_messages, f"{label}.messages")
        raw_tags = item.get("tags", [])
        if isinstance(raw_tags, (str, bytes)) or not isinstance(raw_tags, Sequence):
            raise EvalError(f"{label}.tags must be an array")
        tags = tuple(_identifier(tag, f"{label}.tags") for tag in raw_tags)
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise EvalError(f"{label}.metadata must be an object")
        normalized_metadata = dict(metadata)
        _canonical_json(normalized_metadata)
        prompts.append(PromptSpec(prompt_id, prompt_messages, tags, normalized_metadata))
    prompt_ids = [prompt.prompt_id for prompt in prompts]
    if not prompts or len(prompt_ids) != len(set(prompt_ids)):
        raise EvalError("prompt IDs must be non-empty and unique")

    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping):
        raise EvalError("manifest.outputs must be an object")
    _require_keys(outputs, {"raw_report", "review_pack", "submission_log"}, "outputs")
    return EvalManifest(
        path=manifest_path,
        raw=raw,
        fingerprint=_fingerprint(raw),
        run_id=run_id,
        description=description,
        prompts=tuple(prompts),
        forced_routes=tuple(routes),
        auto_enabled=auto_enabled,
        auto_route_id=auto_route_id,
        auto_policy=auto_policy,
        manual_policy_defaults=manual_defaults,
        settings=settings,
        base_seed=base_seed,
        repetitions=repetitions,
        candidates_per_route=candidates,
        endpoint=endpoint,
        request_timeout_seconds=timeout,
        raw_report=_resolve_path(manifest_path, outputs.get("raw_report"), "raw_report"),
        review_pack=_resolve_path(manifest_path, outputs.get("review_pack"), "review_pack"),
        submission_log=_resolve_path(
            manifest_path, outputs.get("submission_log"), "submission_log"
        ),
    )


def _loopback_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint.strip())
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise EvalError("teacher endpoint must be loopback HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise EvalError("teacher endpoint cannot contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise EvalError("teacher endpoint must not include a path")
    if parsed.port is None:
        raise EvalError("teacher endpoint needs an explicit port")
    return endpoint.rstrip("/")


def _api_json(
    endpoint: str,
    method: str,
    path: str,
    payload: Mapping[str, object] | None,
    timeout: float,
) -> dict[str, Any]:
    data = None if payload is None else _canonical_json(payload).encode("utf-8")
    request = Request(
        endpoint + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback checked
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read(65536).decode("utf-8", errors="replace")
        raise ApiError(f"HTTP {exc.code} from {path}: {body}", status=exc.code, body=body) from exc
    except URLError as exc:
        raise ApiError(f"cannot reach teacher API at {endpoint}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ApiError(f"teacher API request timed out at {path}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(f"teacher API returned non-JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ApiError(f"teacher API returned a non-object from {path}")
    return value


def _ask_student_schema(capabilities: Mapping[str, object]) -> dict[str, Any]:
    tools = capabilities.get("tools")
    if not isinstance(tools, Sequence):
        return {}
    for tool in tools:
        if isinstance(tool, Mapping) and tool.get("name") == "ask_student":
            schema = tool.get("input_schema")
            return dict(schema) if isinstance(schema, Mapping) else {}
    return {}


def _supports_dynamic_routing(capabilities: Mapping[str, object]) -> bool:
    properties = _ask_student_schema(capabilities).get("properties")
    return isinstance(properties, Mapping) and isinstance(properties.get("routing"), Mapping)


def _append_event(path: Path, event: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(event) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_events(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"invalid JSON at {label} {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EvalError(f"{label} {path}:{line_number} must be a JSON object")
            events.append(value)
    return events


@dataclass(frozen=True, slots=True)
class WorkItem:
    prompt: PromptSpec
    route_id: str
    routing_mode: str
    route: RouteSpec | None
    repetition: int


def _work_items(
    manifest: EvalManifest,
    prompt_filter: set[str],
    route_filter: set[str],
) -> list[WorkItem]:
    known_prompts = {prompt.prompt_id for prompt in manifest.prompts}
    known_routes = {route.route_id for route in manifest.forced_routes}
    if manifest.auto_enabled:
        known_routes.add(manifest.auto_route_id)
    unknown_prompts = prompt_filter.difference(known_prompts)
    unknown_routes = route_filter.difference(known_routes)
    if unknown_prompts:
        raise EvalError(f"unknown --prompt IDs: {', '.join(sorted(unknown_prompts))}")
    if unknown_routes:
        raise EvalError(f"unknown --route IDs: {', '.join(sorted(unknown_routes))}")
    items: list[WorkItem] = []
    for prompt in manifest.prompts:
        if prompt_filter and prompt.prompt_id not in prompt_filter:
            continue
        for repetition in range(manifest.repetitions):
            for route in manifest.forced_routes:
                if not route_filter or route.route_id in route_filter:
                    items.append(WorkItem(prompt, route.route_id, "forced", route, repetition))
            if manifest.auto_enabled and (
                not route_filter or manifest.auto_route_id in route_filter
            ):
                items.append(WorkItem(prompt, manifest.auto_route_id, "auto", None, repetition))
    return items


def _route_patch(
    manifest: EvalManifest,
    item: WorkItem,
    dynamic: bool,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    if item.routing_mode == "auto":
        if not dynamic:
            return None, {
                "mechanism": "unsupported",
                "reason": "ask_student schema does not expose dynamic routing",
            }
        return {"routing": dict(manifest.auto_policy)}, {
            "mechanism": "dynamic_auto_policy",
            "policy": manifest.auto_policy,
        }
    assert item.route is not None
    if dynamic:
        policy = dict(manifest.manual_policy_defaults)
        policy["manual_route"] = {item.route.runtime_name: 1.0}
        return {"routing": policy}, {
            "mechanism": "dynamic_manual_policy",
            "stable_route_id": item.route.route_id,
            "runtime_name": item.route.runtime_name,
            "policy": policy,
        }
    if item.route.legacy_weights is not None:
        return {"route": dict(item.route.legacy_weights)}, {
            "mechanism": "legacy_weights",
            "stable_route_id": item.route.route_id,
            "runtime_name": item.route.runtime_name,
            "weights": item.route.legacy_weights,
        }
    return None, {
        "mechanism": "unsupported",
        "stable_route_id": item.route.route_id,
        "reason": "route needs dynamic routing and has no legacy A/B mapping",
    }


def _seed(manifest: EvalManifest, item: WorkItem) -> int:
    material = f"{manifest.run_id}|{item.prompt.prompt_id}|{item.route_id}|{item.repetition}"
    offset = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    return (manifest.base_seed + offset) % (2**31 - 1)


def _request_payload(
    manifest: EvalManifest,
    item: WorkItem,
    route_patch: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    seed = _seed(manifest, item)
    identity = {
        "manifest_sha256": manifest.fingerprint,
        "run_id": manifest.run_id,
        "prompt_id": item.prompt.prompt_id,
        "route_id": item.route_id,
        "routing_mode": item.routing_mode,
        "repetition": item.repetition,
        "seed": seed,
        "settings": manifest.settings,
        "route_patch": route_patch,
    }
    request_key = _fingerprint(identity)
    payload: dict[str, object] = {
        "messages": list(item.prompt.messages),
        "candidates": manifest.candidates_per_route,
        "conversation_id": f"teacher-eval-{request_key[:24]}",
        "teacher_session_id": manifest.run_id,
        "seed": seed,
        "settings": dict(manifest.settings),
        "metadata": {
            "experiment": "teacher-rl-v1",
            "run_id": manifest.run_id,
            "manifest_sha256": manifest.fingerprint,
            "request_key": request_key,
            "prompt_id": item.prompt.prompt_id,
            "prompt_tags": list(item.prompt.tags),
            "prompt_metadata": item.prompt.metadata,
            "declared_route_id": item.route_id,
            "routing_mode": item.routing_mode,
            "repetition": item.repetition,
        },
        **route_patch,
    }
    return request_key, payload


def _validate_generation_response(response: Mapping[str, object]) -> None:
    candidates = response.get("candidates")
    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or not candidates
    ):
        raise ApiError("ask_student response has no candidates")
    seen: set[int] = set()
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise ApiError("ask_student candidate is not an object")
        index = raw.get("candidate_index")
        generation_id = raw.get("generation_id")
        response_text = raw.get("response")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
            raise ApiError("ask_student candidate_index is invalid or duplicated")
        if not isinstance(generation_id, str) or not generation_id:
            raise ApiError("ask_student candidate has no generation_id")
        if not isinstance(response_text, str) or not response_text.strip():
            raise ApiError("ask_student candidate has no response text")
        seen.add(index)


def collect(args: argparse.Namespace, manifest: EvalManifest) -> int:
    endpoint = _loopback_endpoint(args.endpoint or manifest.endpoint)
    report = (args.report or manifest.raw_report).expanduser().resolve()
    # Collection is going to load the student on the first generation anyway.  Load it before
    # recording the session so health includes the actual runtime expert names and router state.
    _api_json(endpoint, "POST", "/v1/student/load", {}, manifest.request_timeout_seconds)
    health = _api_json(endpoint, "GET", "/health", None, manifest.request_timeout_seconds)
    capabilities = _api_json(
        endpoint, "GET", "/v1/capabilities", None, manifest.request_timeout_seconds
    )
    dynamic = _supports_dynamic_routing(capabilities)
    capability_sha = _fingerprint(capabilities)
    artifact_contract = _build_student_artifact_contract(health)
    artifact_contract_sha = str(artifact_contract["contract_sha256"])
    existing = _read_events(report, "raw report")
    successes = {
        event.get("request_key")
        for event in existing
        if event.get("event") == "rollout" and event.get("status") == "ok"
        and event.get("artifact_contract_sha256") == artifact_contract_sha
    }
    terminal_attempts = {
        (
            event.get("request_key"),
            event.get("capabilities_sha256"),
            event.get("artifact_contract_sha256"),
            event.get("status"),
        )
        for event in existing
        if event.get("event") == "rollout"
    }
    _append_event(
        report,
        {
            "schema_version": SCHEMA_RAW,
            "event": "collection_session",
            "created_at": _now(),
            "run_id": manifest.run_id,
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.fingerprint,
            "endpoint": endpoint,
            "health": health,
            "capabilities": capabilities,
            "capabilities_sha256": capability_sha,
            "dynamic_routing_available": dynamic,
            "artifact_provenance_status": EXACT_ARTIFACT_PROVENANCE,
            "student_artifact_contract": artifact_contract,
        },
    )

    items = _work_items(manifest, set(args.prompt), set(args.route))
    if args.limit is not None:
        items = items[: args.limit]
    counts = {"ok": 0, "resumed": 0, "unsupported": 0, "error": 0}
    for ordinal, item in enumerate(items, start=1):
        patch, resolution = _route_patch(manifest, item, dynamic)
        placeholder_patch = patch or {"unsupported_route": item.route_id}
        request_key, payload = _request_payload(manifest, item, placeholder_patch)
        if request_key in successes:
            counts["resumed"] += 1
            print(f"[{ordinal}/{len(items)}] resume {item.prompt.prompt_id} / {item.route_id}")
            continue
        if patch is None:
            if (
                request_key,
                capability_sha,
                artifact_contract_sha,
                "unsupported",
            ) in terminal_attempts:
                counts["resumed"] += 1
                continue
            _append_event(
                report,
                {
                    "schema_version": SCHEMA_RAW,
                    "event": "rollout",
                    "status": "unsupported",
                    "created_at": _now(),
                    "run_id": manifest.run_id,
                    "manifest_sha256": manifest.fingerprint,
                    "capabilities_sha256": capability_sha,
                    "artifact_contract_sha256": artifact_contract_sha,
                    "request_key": request_key,
                    "prompt_id": item.prompt.prompt_id,
                    "route_id": item.route_id,
                    "routing_mode": item.routing_mode,
                    "repetition": item.repetition,
                    "route_resolution": resolution,
                },
            )
            counts["unsupported"] += 1
            print(f"[{ordinal}/{len(items)}] unsupported {item.prompt.prompt_id} / {item.route_id}")
            continue
        if not args.retry_errors and (
            request_key,
            capability_sha,
            artifact_contract_sha,
            "error",
        ) in terminal_attempts:
            counts["resumed"] += 1
            continue
        started_at = _now()
        started = time.perf_counter()
        print(f"[{ordinal}/{len(items)}] generate {item.prompt.prompt_id} / {item.route_id}")
        try:
            response = _api_json(
                endpoint,
                "POST",
                "/v1/student/generate",
                payload,
                manifest.request_timeout_seconds,
            )
            _validate_generation_response(response)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            event = {
                "schema_version": SCHEMA_RAW,
                "event": "rollout",
                "status": "ok",
                "started_at": started_at,
                "completed_at": _now(),
                "duration_ms": elapsed_ms,
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.fingerprint,
                "capabilities_sha256": capability_sha,
                "artifact_contract_sha256": artifact_contract_sha,
                "request_key": request_key,
                "prompt_id": item.prompt.prompt_id,
                "route_id": item.route_id,
                "routing_mode": item.routing_mode,
                "repetition": item.repetition,
                "route_resolution": resolution,
                "request": payload,
                "response": response,
            }
            _append_event(report, event)
            successes.add(request_key)
            counts["ok"] += 1
        except Exception as exc:  # persist enough state for an explicit retry
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            _append_event(
                report,
                {
                    "schema_version": SCHEMA_RAW,
                    "event": "rollout",
                    "status": "error",
                    "started_at": started_at,
                    "completed_at": _now(),
                    "duration_ms": elapsed_ms,
                    "run_id": manifest.run_id,
                    "manifest_sha256": manifest.fingerprint,
                    "capabilities_sha256": capability_sha,
                    "artifact_contract_sha256": artifact_contract_sha,
                    "request_key": request_key,
                    "prompt_id": item.prompt.prompt_id,
                    "route_id": item.route_id,
                    "routing_mode": item.routing_mode,
                    "repetition": item.repetition,
                    "route_resolution": resolution,
                    "request": payload,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "http_status": getattr(exc, "status", None),
                    },
                },
            )
            counts["error"] += 1
            print(f"  error: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break
    print(
        f"report={report} ok={counts['ok']} resumed={counts['resumed']} "
        f"unsupported={counts['unsupported']} error={counts['error']}"
    )
    return 1 if counts["error"] else 0


def _successful_rollouts(events: Iterable[Mapping[str, object]]) -> dict[str, dict[str, Any]]:
    successful: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "rollout" and event.get("status") == "ok":
            request_key = event.get("request_key")
            if isinstance(request_key, str):
                successful[request_key] = dict(event)
    return successful


def _candidate_rows(events: Iterable[Mapping[str, object]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for request_key, event in _successful_rollouts(events).items():
        response = event.get("response")
        request = event.get("request")
        if not isinstance(response, Mapping) or not isinstance(request, Mapping):
            continue
        candidates = response.get("candidates")
        if not isinstance(candidates, Sequence):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            index = candidate.get("candidate_index")
            generation_id = candidate.get("generation_id")
            text = candidate.get("response")
            if not isinstance(index, int) or not isinstance(generation_id, str):
                continue
            candidate_key = f"{request_key}:{index}"
            rows[candidate_key] = {
                "candidate_key": candidate_key,
                "request_key": request_key,
                "candidate_index": index,
                "generation_id": generation_id,
                "prompt_id": event.get("prompt_id"),
                "route_id": event.get("route_id"),
                "routing_mode": event.get("routing_mode"),
                "repetition": event.get("repetition"),
                "messages": request.get("messages"),
                "response": text,
                "telemetry": candidate.get("telemetry", {}),
                "routing_decision": candidate.get(
                    "routing_decision", response.get("routing_decision")
                ),
                "routing_policy": response.get("routing_policy"),
            }
    return rows


def status(args: argparse.Namespace, manifest: EvalManifest) -> int:
    report = (args.report or manifest.raw_report).expanduser().resolve()
    events = _read_events(report, "raw report")
    expected = _work_items(manifest, set(args.prompt), set(args.route))
    success = _successful_rollouts(events)
    latest: dict[str, str] = {}
    for event in events:
        if event.get("event") == "rollout" and isinstance(event.get("request_key"), str):
            latest[str(event["request_key"])] = str(event.get("status"))
    expected_keys: list[str] = []
    for item in expected:
        # Dynamic and legacy requests have different keys. Match by semantic coordinates below.
        coordinate = (item.prompt.prompt_id, item.route_id, item.repetition)
        found = any(
            event.get("prompt_id") == coordinate[0]
            and event.get("route_id") == coordinate[1]
            and event.get("repetition") == coordinate[2]
            and event.get("status") == "ok"
            for event in events
        )
        if found:
            expected_keys.append("ok")
        else:
            expected_keys.append("pending")
    status_counts: dict[str, int] = {}
    for value in latest.values():
        status_counts[value] = status_counts.get(value, 0) + 1
    print(f"run_id: {manifest.run_id}")
    print(f"manifest_sha256: {manifest.fingerprint}")
    print(f"report: {report}")
    print(f"expected matrix cells: {len(expected)}")
    print(f"completed expected cells: {expected_keys.count('ok')}")
    print(f"pending expected cells: {expected_keys.count('pending')}")
    print(f"successful request records: {len(success)}")
    print(f"candidate generations: {len(_candidate_rows(events))}")
    print(f"latest recorded statuses: {json.dumps(status_counts, sort_keys=True)}")
    return 0


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def review_pack(args: argparse.Namespace, manifest: EvalManifest) -> int:
    report = (args.report or manifest.raw_report).expanduser().resolve()
    output = (args.output or manifest.review_pack).expanduser().resolve()
    events = _read_events(report, "raw report")
    candidates = _candidate_rows(events)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates.values():
        prompt_id = str(row["prompt_id"])
        grouped.setdefault(prompt_id, []).append(row)
    prompt_map = {prompt.prompt_id: prompt for prompt in manifest.prompts}
    groups: list[dict[str, object]] = []
    for prompt_id in [prompt.prompt_id for prompt in manifest.prompts]:
        rows = grouped.get(prompt_id, [])
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{manifest.fingerprint}|blind-order|{row['candidate_key']}".encode("utf-8")
            ).hexdigest()
        )
        prompt = prompt_map[prompt_id]
        blind_metadata = {
            key: value
            for key, value in prompt.metadata.items()
            if key
            not in {
                "category",
                "expected",
                "expected_route",
                "acceptable",
                "acceptable_routes",
            }
        }
        blind_candidates = [
            {
                "candidate_key": row["candidate_key"],
                "generation_id": row["generation_id"],
                "messages": row["messages"],
                "response": row["response"],
            }
            for row in rows
        ]
        groups.append(
            {
                "prompt_key": "prompt-"
                + hashlib.sha256(
                    f"{manifest.fingerprint}|blind-prompt|{prompt_id}".encode("utf-8")
                ).hexdigest()[:12],
                "evaluation": blind_metadata,
                "messages": list(prompt.messages),
                "candidates": blind_candidates,
            }
        )
    rubric = None
    for event in reversed(events):
        capabilities = event.get("capabilities")
        if event.get("event") == "collection_session" and isinstance(capabilities, Mapping):
            rubric = capabilities.get("rubric")
            break
    payload = {
        "schema_version": SCHEMA_REVIEW,
        "created_at": _now(),
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.fingerprint,
        "raw_report": str(report),
        "raw_report_sha256": _file_fingerprint(report),
        "rubric": rubric,
        "instructions": [
            "Judge output content without using the declared route as a quality signal.",
            "Return judgments in molor.teacher-eval.judgments.v1 format.",
            "Use candidate_key values exactly; do not fabricate generation IDs.",
            "Preferences may compare only candidates in the same prompt group.",
            "Supply confidence and a specific critique or rationale for every judgment.",
        ],
        "judgments_template": {
            "schema_version": SCHEMA_JUDGMENTS,
            "run_id": manifest.run_id,
            "rubric_version": "molor.teacher-rubric.v1",
            "teacher": {"id": "REQUIRED", "model": "REQUIRED", "provider": "REQUIRED"},
            "grades": [],
            "preferences": [],
        },
        "groups": groups,
    }
    _atomic_write_json(output, payload)
    print(f"wrote {len(candidates)} candidates in {len(groups)} prompt groups to {output}")
    return 0


def _teacher(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise EvalError("judgments.teacher must be an object")
    _require_keys(value, {"id", "model", "provider"}, "judgments.teacher")
    result: dict[str, str] = {}
    for field in ("id", "model", "provider"):
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise EvalError(f"judgments.teacher.{field} is required")
        result[field] = item.strip()
    return result


def _judgment_metadata(
    manifest: EvalManifest,
    candidate: Mapping[str, object],
    source_sha: str,
    supplied: object,
) -> dict[str, object]:
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise EvalError("judgment metadata must be an object")
    return {
        **dict(supplied),
        "experiment": "teacher-rl-v1",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.fingerprint,
        "judgments_file_sha256": source_sha,
        "candidate_key": candidate["candidate_key"],
        "request_key": candidate["request_key"],
        "prompt_id": candidate["prompt_id"],
        "declared_route_id": candidate["route_id"],
        "routing_mode": candidate["routing_mode"],
    }


def _submission_operations(
    manifest: EvalManifest,
    judgments_path: Path,
    candidates: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    raw = _load_object(judgments_path, "judgments")
    _require_keys(
        raw,
        {"schema_version", "run_id", "rubric_version", "teacher", "grades", "preferences"},
        "judgments",
    )
    if raw.get("schema_version") != SCHEMA_JUDGMENTS:
        raise EvalError(f"judgments.schema_version must be {SCHEMA_JUDGMENTS!r}")
    if raw.get("run_id") != manifest.run_id:
        raise EvalError("judgments.run_id does not match the manifest")
    rubric_version = raw.get("rubric_version")
    if not isinstance(rubric_version, str) or not rubric_version.strip():
        raise EvalError("judgments.rubric_version is required")
    teacher = _teacher(raw.get("teacher"))
    source_sha = _file_fingerprint(judgments_path)
    grades = raw.get("grades")
    preferences = raw.get("preferences")
    if not isinstance(grades, list) or not isinstance(preferences, list):
        raise EvalError("judgments.grades and judgments.preferences must be arrays")
    operations: list[dict[str, object]] = []

    for index, grade in enumerate(grades):
        label = f"judgments.grades[{index}]"
        if not isinstance(grade, Mapping):
            raise EvalError(f"{label} must be an object")
        _require_keys(
            grade,
            {"candidate_key", "scores", "confidence", "critique", "corrected_answer", "metadata"},
            label,
        )
        candidate_key = grade.get("candidate_key")
        if not isinstance(candidate_key, str) or candidate_key not in candidates:
            raise EvalError(f"{label}.candidate_key is not in the raw report")
        candidate = candidates[candidate_key]
        scores = grade.get("scores")
        if not isinstance(scores, Mapping) or "overall" not in scores:
            raise EvalError(f"{label}.scores must contain overall")
        unknown_scores = set(scores).difference(RUBRIC_DIMENSIONS | {"overall"})
        if unknown_scores:
            raise EvalError(f"{label}.scores has unknown dimensions: {sorted(unknown_scores)}")
        normalized_scores: dict[str, float] = {}
        for name, score in scores.items():
            value = _finite(score, f"{label}.scores.{name}")
            if not -2 <= value <= 2:
                raise EvalError(f"{label}.scores.{name} must be from -2 to 2")
            normalized_scores[name] = value
        confidence = _finite(grade.get("confidence"), f"{label}.confidence")
        if not 0 <= confidence <= 1:
            raise EvalError(f"{label}.confidence must be from zero to one")
        critique = grade.get("critique")
        if not isinstance(critique, str) or not critique.strip():
            raise EvalError(f"{label}.critique must be a non-empty string")
        corrected = grade.get("corrected_answer")
        if corrected is not None and (not isinstance(corrected, str) or not corrected.strip()):
            raise EvalError(f"{label}.corrected_answer must be null or non-empty")
        payload = {
            "generation_id": candidate["generation_id"],
            "teacher": teacher,
            "scores": normalized_scores,
            "confidence": confidence,
            "critique": critique,
            "corrected_answer": corrected,
            "rubric_version": rubric_version,
            "metadata": _judgment_metadata(manifest, candidate, source_sha, grade.get("metadata")),
        }
        operations.append(
            {
                "kind": "grade",
                "candidate_keys": [candidate_key],
                "path": "/v1/student/grade",
                "payload": payload,
                "submission_key": _fingerprint(
                    {
                        "kind": "grade",
                        "generation_id": candidate["generation_id"],
                        "teacher": teacher,
                        "scores": normalized_scores,
                        "confidence": confidence,
                        "critique": critique,
                        "corrected_answer": corrected,
                        "rubric_version": rubric_version,
                    }
                ),
            }
        )

    for index, preference in enumerate(preferences):
        label = f"judgments.preferences[{index}]"
        if not isinstance(preference, Mapping):
            raise EvalError(f"{label} must be an object")
        _require_keys(
            preference,
            {
                "chosen_candidate_key",
                "rejected_candidate_key",
                "confidence",
                "rationale",
                "metadata",
            },
            label,
        )
        chosen_key = preference.get("chosen_candidate_key")
        rejected_key = preference.get("rejected_candidate_key")
        if not isinstance(chosen_key, str) or chosen_key not in candidates:
            raise EvalError(f"{label}.chosen_candidate_key is not in the raw report")
        if not isinstance(rejected_key, str) or rejected_key not in candidates:
            raise EvalError(f"{label}.rejected_candidate_key is not in the raw report")
        if chosen_key == rejected_key:
            raise EvalError(f"{label} cannot compare a candidate with itself")
        chosen, rejected = candidates[chosen_key], candidates[rejected_key]
        if _canonical_json(chosen["messages"]) != _canonical_json(rejected["messages"]):
            raise EvalError(f"{label} candidates do not have identical input messages")
        confidence = _finite(preference.get("confidence"), f"{label}.confidence")
        if not 0 <= confidence <= 1:
            raise EvalError(f"{label}.confidence must be from zero to one")
        rationale = preference.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise EvalError(f"{label}.rationale must be a non-empty string")
        metadata = preference.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise EvalError(f"{label}.metadata must be an object")
        payload = {
            "chosen_generation_id": chosen["generation_id"],
            "rejected_generation_id": rejected["generation_id"],
            "teacher": teacher,
            "confidence": confidence,
            "rationale": rationale,
            "metadata": {
                **dict(metadata),
                "experiment": "teacher-rl-v1",
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.fingerprint,
                "judgments_file_sha256": source_sha,
                "chosen_candidate_key": chosen_key,
                "rejected_candidate_key": rejected_key,
                "prompt_id": chosen["prompt_id"],
                "chosen_route_id": chosen["route_id"],
                "rejected_route_id": rejected["route_id"],
            },
        }
        operations.append(
            {
                "kind": "preference",
                "candidate_keys": [chosen_key, rejected_key],
                "path": "/v1/student/preference",
                "payload": payload,
                "submission_key": _fingerprint(
                    {
                        "kind": "preference",
                        "chosen_generation_id": chosen["generation_id"],
                        "rejected_generation_id": rejected["generation_id"],
                        "teacher": teacher,
                        "confidence": confidence,
                        "rationale": rationale,
                    }
                ),
            }
        )
    keys = [str(operation["submission_key"]) for operation in operations]
    if len(keys) != len(set(keys)):
        raise EvalError("judgments file contains duplicate operations")
    return operations


def submit(args: argparse.Namespace, manifest: EvalManifest) -> int:
    report = (args.report or manifest.raw_report).expanduser().resolve()
    log = (args.submission_log or manifest.submission_log).expanduser().resolve()
    judgments = args.judgments.expanduser().resolve()
    candidates = _candidate_rows(_read_events(report, "raw report"))
    operations = _submission_operations(manifest, judgments, candidates)
    receipts = _read_events(log, "submission log")
    completed = {
        event.get("submission_key")
        for event in receipts
        if event.get("event") == "submission" and event.get("status") == "ok"
    }
    pending = [op for op in operations if op["submission_key"] not in completed]
    print(
        f"validated {len(operations)} supplied judgments; "
        f"already submitted={len(operations) - len(pending)} pending={len(pending)}"
    )
    if not args.commit:
        for operation in pending:
            print(
                f"preview {operation['kind']} {operation['submission_key']} "
                f"candidates={','.join(operation['candidate_keys'])}"
            )
        print("preview only; add --commit to submit these exact payloads")
        return 0
    endpoint = _loopback_endpoint(args.endpoint or manifest.endpoint)
    failures = 0
    for index, operation in enumerate(pending, start=1):
        started = time.perf_counter()
        try:
            response = _api_json(
                endpoint,
                "POST",
                str(operation["path"]),
                operation["payload"],
                manifest.request_timeout_seconds,
            )
            receipt = {
                "schema_version": SCHEMA_SUBMISSION,
                "event": "submission",
                "status": "ok",
                "created_at": _now(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "run_id": manifest.run_id,
                "manifest_sha256": manifest.fingerprint,
                **operation,
                "response": response,
            }
            _append_event(log, receipt)
            print(f"[{index}/{len(pending)}] submitted {operation['kind']}")
        except Exception as exc:
            failures += 1
            _append_event(
                log,
                {
                    "schema_version": SCHEMA_SUBMISSION,
                    "event": "submission",
                    "status": "error",
                    "created_at": _now(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "run_id": manifest.run_id,
                    "manifest_sha256": manifest.fingerprint,
                    **operation,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            print(f"[{index}/{len(pending)}] error: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break
    print(f"submission_log={log} submitted={len(pending) - failures} errors={failures}")
    return 1 if failures else 0


def plan(args: argparse.Namespace, manifest: EvalManifest) -> int:
    items = _work_items(manifest, set(args.prompt), set(args.route))
    if args.limit is not None:
        items = items[: args.limit]
    print(f"run_id: {manifest.run_id}")
    print(f"manifest_sha256: {manifest.fingerprint}")
    print(f"prompts: {len({item.prompt.prompt_id for item in items})}")
    print(f"matrix cells: {len(items)}")
    print(f"candidates per cell: {manifest.candidates_per_route}")
    print(f"maximum new generations: {len(items) * manifest.candidates_per_route}")
    print(f"raw report: {manifest.raw_report}")
    for route_id in sorted({item.route_id for item in items}):
        count = sum(item.route_id == route_id for item in items)
        route = next((item.route for item in items if item.route_id == route_id), None)
        fallback = (
            "dynamic only" if route is None or route.legacy_weights is None else "dynamic/legacy"
        )
        print(f"  {route_id}: {count} cells ({fallback})")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect resumable forced/auto route rollouts and submit supplied judgments.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def matrix_filters(command: argparse.ArgumentParser) -> None:
        command.add_argument("--prompt", action="append", default=[], help="prompt ID; repeatable")
        command.add_argument("--route", action="append", default=[], help="route ID; repeatable")

    plan_parser = subparsers.add_parser(
        "plan", help="validate and print the offline request matrix"
    )
    matrix_filters(plan_parser)
    plan_parser.add_argument("--limit", type=int)

    collect_parser = subparsers.add_parser(
        "collect", help="call ask_student and append raw results"
    )
    matrix_filters(collect_parser)
    collect_parser.add_argument("--endpoint")
    collect_parser.add_argument("--report", type=Path)
    collect_parser.add_argument("--limit", type=int)
    collect_parser.add_argument("--retry-errors", action="store_true")
    collect_parser.add_argument("--continue-on-error", action="store_true")

    status_parser = subparsers.add_parser("status", help="summarize a report without API access")
    matrix_filters(status_parser)
    status_parser.add_argument("--report", type=Path)

    review_parser = subparsers.add_parser(
        "review-pack", help="export collected candidates for an external teacher"
    )
    review_parser.add_argument("--report", type=Path)
    review_parser.add_argument("--output", type=Path)

    submit_parser = subparsers.add_parser(
        "submit", help="validate supplied judgments; submit only with --commit"
    )
    submit_parser.add_argument("--judgments", type=Path, required=True)
    submit_parser.add_argument("--report", type=Path)
    submit_parser.add_argument("--submission-log", type=Path)
    submit_parser.add_argument("--endpoint")
    submit_parser.add_argument("--commit", action="store_true")
    submit_parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if getattr(args, "limit", None) is not None and args.limit < 1:
            raise EvalError("--limit must be a positive integer")
        operations = {
            "plan": plan,
            "collect": collect,
            "status": status,
            "review-pack": review_pack,
            "submit": submit,
        }
        return operations[args.command](args, manifest)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
