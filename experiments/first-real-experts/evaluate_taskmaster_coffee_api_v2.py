"""One-shot, three-route evaluation for the Taskmaster coffee API v2 candidate.

The real v2 test is deliberately not opened until the policy, evaluator, metric
contract, build report, base model, candidate adapter, and quarantined v1 comparator
have been content-addressed in a pre-access receipt.  A completed run exposes the test
forever; its results must never be used to tune or select another checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from build_taskmaster_coffee_api_v2 import (
    DATASET_ID,
    EVALUATION_SCHEMA,
    TOOL_CATALOG_SCHEMA,
    SourceDialogError,
    canonical_json,
    messages_for,
    sha256_file,
    sha256_text,
    strict_json_loads,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_MODEL = WORKSPACE / "models" / "Qwen3-1.7B"
DEFAULT_CANDIDATE = (
    WORKSPACE
    / "checkpoints"
    / "qwen3-1.7b-candidate-experts-quarantine"
    / "taskmaster-coffee-api-v2-full-dialog-r16"
)
DEFAULT_V1_COMPARATOR = (
    WORKSPACE
    / "checkpoints"
    / "qwen3-1.7b-candidate-experts-quarantine"
    / "taskmaster-coffee-api-v1"
)
DEFAULT_DATA = HERE / "data"
DEFAULT_EVALUATION = DEFAULT_DATA / f"{DATASET_ID}.test.evaluation.jsonl"
DEFAULT_BUILD_REPORT = DEFAULT_DATA / f"{DATASET_ID}.build-report.json"
DEFAULT_POLICY = HERE / "taskmaster-coffee-api-v2-evaluation-policy.json"
DEFAULT_METRIC_CONTRACT = HERE / "taskmaster-coffee-api-v2-metric-contract.json"
DEFAULT_PREDICTIONS = WORKSPACE / "runs" / "evaluation" / f"{DATASET_ID}.predictions.jsonl"
DEFAULT_REPORT = WORKSPACE / "runs" / "evaluation" / f"{DATASET_ID}.json"
DEFAULT_PREACCESS_RECEIPT = (
    WORKSPACE / "runs" / "evaluation" / f"{DATASET_ID}.preaccess-receipt.json"
)

PREDICTION_SCHEMA = "molor.taskmaster-coffee-api-prediction.v2"
REPORT_SCHEMA = "molor.taskmaster-coffee-api-evaluation-report.v2"
PREACCESS_SCHEMA = "molor.taskmaster-coffee-api-preaccess-receipt.v1"
TRAINING_COMPLETION_SCHEMA = "molor.taskmaster-coffee-api-training-completion.v2"
TRAINING_PREFLIGHT_SCHEMA = "molor.taskmaster-coffee-api-training-receipt.v2"
METRIC_CONTRACT_SCHEMA = "molor.taskmaster-coffee-api-metric-contract.v2"
ROUTE_ORDER = ("forced_candidate", "frozen_base_null", "quarantined_v1")
PRINCIPAL_METRICS = ("api_name_exact", "arguments_exact", "semantic_exact")
SCORED_METRICS = (
    "json_parseable",
    "schema_valid",
    "api_name_exact",
    "arguments_exact",
    "semantic_exact",
)
ONE_SHOT_WARNING = (
    "The v2 test is now exposed. Never use it for tuning, checkpoint selection, "
    "threshold adjustment, or training."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXAMPLE_ID = re.compile(r"^tm4-coffee-api-v2:[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^tm4-coffee-v2-component:[0-9a-f]{64}$")
_USER_PREFIX = "Taskmaster-4 conversation context:\n"
_SYSTEM_PREFIX = (
    "You are the API planner for a coffee-ordering agent. The user supplies a "
    "Taskmaster-4 conversation context as canonical JSON. Return exactly one JSON "
    "object with role=agent and one api call. Use only this train-derived tool catalog: "
)
_SYSTEM_SUFFIX = ". Output canonical minified JSON only: no Markdown, commentary, or thinking tags."


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    example_id: str
    group_id: str
    messages: tuple[dict[str, str], ...]
    context: list[dict[str, str]]
    expected: dict[str, Any]
    expected_canonical: str
    expected_sha256: str
    tool_catalog_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    cases: tuple[EvaluationCase, ...]
    tool_catalog: dict[str, Any]
    tool_catalog_canonical_sha256: str


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = strict_json_loads(path.read_text(encoding="utf-8"), reason=f"invalid_{label}_json")
    except OSError as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    return raw


def _validate_generation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"api", "role"}:
        raise ValueError("generation must contain exactly api and role")
    if value["role"] != "agent":
        raise ValueError("generation role must be agent")
    api = value["api"]
    if not isinstance(api, dict) or len(api) != 1:
        raise ValueError("generation api must contain exactly one call")
    name, arguments = next(iter(api.items()))
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("generation API name must be a non-empty trimmed string")
    if not isinstance(arguments, dict):
        raise ValueError("generation API arguments must be an object")
    canonical_json(value)
    return value


def _validate_context(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("conversation context must be a non-empty array")
    context: list[dict[str, str]] = []
    for index, turn in enumerate(value):
        if (
            not isinstance(turn, dict)
            or set(turn) != {"role", "text"}
            or turn.get("role") not in {"user", "agent"}
            or not isinstance(turn.get("text"), str)
            or not turn["text"].strip()
        ):
            raise ValueError(f"invalid conversation context turn {index}")
        context.append({"role": turn["role"], "text": turn["text"]})
    if context[-1]["role"] != "user":
        raise ValueError("decision context must end with a user turn")
    return context


def _validate_tool_catalog(value: Any) -> dict[str, Any]:
    required = {"schema_version", "dataset_id", "source_split", "tools"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("tool catalog fields do not match the v2 contract")
    if value["schema_version"] != TOOL_CATALOG_SCHEMA:
        raise ValueError("unsupported tool catalog schema")
    if value["dataset_id"] != DATASET_ID or value["source_split"] != "train":
        raise ValueError("tool catalog must be the v2 train-derived catalog")
    tools = value["tools"]
    if not isinstance(tools, list) or not tools:
        raise ValueError("tool catalog must contain tools")
    names: list[str] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or set(tool) != {
            "argument_key_sets",
            "argument_types",
            "name",
        }:
            raise ValueError(f"invalid tool catalog entry {index}")
        name = tool["name"]
        key_sets = tool["argument_key_sets"]
        argument_types = tool["argument_types"]
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"invalid tool name at catalog entry {index}")
        if (
            not isinstance(key_sets, list)
            or not key_sets
            or any(
                not isinstance(keys, list)
                or any(not isinstance(key, str) or not key for key in keys)
                or keys != sorted(set(keys))
                for keys in key_sets
            )
            or key_sets != sorted(key_sets)
        ):
            raise ValueError(f"invalid argument key sets for tool {name}")
        if not isinstance(argument_types, dict):
            raise ValueError(f"invalid argument types for tool {name}")
        for key, types in argument_types.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(types, list)
                or not types
                or types != sorted(set(types))
                or any(
                    value_type
                    not in {"array", "boolean", "integer", "null", "number", "object", "string"}
                    for value_type in types
                )
            ):
                raise ValueError(f"invalid argument type declaration for tool {name}")
        if sorted(argument_types) != sorted({key for keys in key_sets for key in keys}):
            raise ValueError(f"tool {name} key-set and type declarations disagree")
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("tool catalog entries must have unique sorted names")
    canonical_json(value)
    return value


def _parse_messages(
    value: Any,
    *,
    expected_canonical: str,
) -> tuple[tuple[dict[str, str], ...], list[dict[str, str]], dict[str, Any], str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(message, dict)
            or set(message) != {"content", "role"}
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            for message, role in zip(value, ("system", "user"), strict=True)
        )
    ):
        raise ValueError("evaluation messages must be one exact system/user pair")
    system = value[0]["content"]
    user = value[1]["content"]
    if not system.startswith(_SYSTEM_PREFIX) or not system.endswith(_SYSTEM_SUFFIX):
        raise ValueError("system prompt does not match the dynamic v2 contract")
    catalog_text = system[len(_SYSTEM_PREFIX) : -len(_SYSTEM_SUFFIX)]
    catalog = _validate_tool_catalog(strict_json_loads(catalog_text, reason="invalid_tool_catalog"))
    if canonical_json(catalog) != catalog_text:
        raise ValueError("embedded tool catalog is not canonical JSON")
    if not user.startswith(_USER_PREFIX):
        raise ValueError("user prompt does not contain a v2 conversation context")
    context_text = user[len(_USER_PREFIX) :]
    context = _validate_context(strict_json_loads(context_text, reason="invalid_context_json"))
    if canonical_json(context) != context_text:
        raise ValueError("embedded conversation context is not canonical JSON")
    expected_messages = messages_for(
        context_text,
        expected_canonical,
        tool_catalog_canonical=catalog_text,
    )[:-1]
    if value != expected_messages:
        raise ValueError("evaluation messages do not match the v2 builder contract")
    normalized = tuple(
        {"role": message["role"], "content": message["content"]} for message in value
    )
    return normalized, context, catalog, sha256_text(catalog_text)


def load_evaluation(path: Path) -> EvaluationSuite:
    """Strictly load a v2 evaluation file and validate every content identity."""

    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    catalog_by_hash: dict[str, dict[str, Any]] = {}
    required = {
        "schema_version",
        "example_id",
        "expected",
        "expected_canonical",
        "expected_sha256",
        "group_id",
        "messages",
    }
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not open evaluation {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                if not line.strip():
                    raise ValueError("blank evaluation rows are forbidden")
                raw = strict_json_loads(line, reason="invalid_evaluation_json")
                if not isinstance(raw, dict) or set(raw) != required:
                    raise ValueError("evaluation row has unexpected fields")
                if raw["schema_version"] != EVALUATION_SCHEMA:
                    raise ValueError("unsupported evaluation schema")
                example_id = raw["example_id"]
                group_id = raw["group_id"]
                if not isinstance(example_id, str) or _EXAMPLE_ID.fullmatch(example_id) is None:
                    raise ValueError("invalid v2 example_id")
                if example_id in seen:
                    raise ValueError("duplicate example_id")
                if not isinstance(group_id, str) or _GROUP_ID.fullmatch(group_id) is None:
                    raise ValueError("invalid v2 group_id")
                expected = _validate_generation(raw["expected"])
                expected_canonical = canonical_json(expected)
                if raw["expected_canonical"] != expected_canonical:
                    raise ValueError("expected_canonical does not match expected")
                expected_sha256 = sha256_text(expected_canonical)
                if raw["expected_sha256"] != expected_sha256:
                    raise ValueError("expected_sha256 does not match expected")
                messages, context, catalog, catalog_hash = _parse_messages(
                    raw["messages"], expected_canonical=expected_canonical
                )
                expected_api_name = next(iter(expected["api"]))
                if expected_api_name not in {tool["name"] for tool in catalog["tools"]}:
                    raise ValueError("expected API name is absent from the train-derived catalog")
                identifier_payload = {"context": context, "generation": expected}
                canonical_id = (
                    f"tm4-coffee-api-v2:{sha256_text(canonical_json(identifier_payload))}"
                )
                if example_id != canonical_id:
                    raise ValueError("example_id does not match context and expected content")
            except (KeyError, SourceDialogError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen.add(example_id)
            catalog_by_hash[catalog_hash] = catalog
            cases.append(
                EvaluationCase(
                    example_id=example_id,
                    group_id=group_id,
                    messages=messages,
                    context=context,
                    expected=expected,
                    expected_canonical=expected_canonical,
                    expected_sha256=expected_sha256,
                    tool_catalog_sha256=catalog_hash,
                )
            )
    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    if [case.example_id for case in cases] != sorted(case.example_id for case in cases):
        raise ValueError("evaluation cases must be sorted by example_id")
    if len(catalog_by_hash) != 1:
        raise ValueError("evaluation rows do not share one train-derived tool catalog")
    catalog_hash, catalog = next(iter(catalog_by_hash.items()))
    return EvaluationSuite(tuple(cases), catalog, catalog_hash)


def prediction_id(example_id: str, route: str, output: str) -> str:
    payload = {"example_id": example_id, "output": output, "route": route}
    return f"tm4-coffee-api-v2-prediction:{sha256_text(canonical_json(payload))}"


def prediction_rows(
    suite: EvaluationSuite,
    outputs_by_route: Mapping[str, Sequence[str]],
) -> list[dict[str, str]]:
    if set(outputs_by_route) != set(ROUTE_ORDER):
        raise ValueError("outputs must contain exactly the three policy routes")
    rows: list[dict[str, str]] = []
    for route in ROUTE_ORDER:
        outputs = outputs_by_route[route]
        if len(outputs) != len(suite.cases):
            raise ValueError(f"{route} output count does not match evaluation")
        for case, output in zip(suite.cases, outputs, strict=True):
            if not isinstance(output, str):
                raise TypeError("model outputs must be strings")
            rows.append(
                {
                    "schema_version": PREDICTION_SCHEMA,
                    "prediction_id": prediction_id(case.example_id, route, output),
                    "route": route,
                    "example_id": case.example_id,
                    "output": output,
                    "output_sha256": sha256_text(output),
                }
            )
    return rows


def write_predictions(
    path: Path,
    suite: EvaluationSuite,
    outputs_by_route: Mapping[str, Sequence[str]],
) -> None:
    rows = prediction_rows(suite, outputs_by_route)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row))
            stream.write("\n")


def load_predictions(path: Path, suite: EvaluationSuite) -> dict[str, list[str]]:
    required = {
        "schema_version",
        "prediction_id",
        "route",
        "example_id",
        "output",
        "output_sha256",
    }
    expected_keys = [(route, case.example_id) for route in ROUTE_ORDER for case in suite.cases]
    parsed: dict[tuple[str, str], str] = {}
    observed_keys: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                if not line.strip():
                    raise ValueError("blank prediction rows are forbidden")
                raw = strict_json_loads(line, reason="invalid_prediction_json")
                if not isinstance(raw, dict) or set(raw) != required:
                    raise ValueError("prediction row has unexpected fields")
                if raw["schema_version"] != PREDICTION_SCHEMA:
                    raise ValueError("unsupported prediction schema")
                route = raw["route"]
                example_id = raw["example_id"]
                output = raw["output"]
                if route not in ROUTE_ORDER or not isinstance(example_id, str):
                    raise ValueError("prediction route or example_id is invalid")
                if not isinstance(output, str):
                    raise ValueError("prediction output must be a string")
                key = (route, example_id)
                if key in parsed:
                    raise ValueError("duplicate route/example prediction")
                if raw["output_sha256"] != sha256_text(output):
                    raise ValueError("output_sha256 does not match output")
                if raw["prediction_id"] != prediction_id(example_id, route, output):
                    raise ValueError("prediction_id does not match prediction content")
            except (KeyError, SourceDialogError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            parsed[key] = output
            observed_keys.append(key)
    if observed_keys != expected_keys:
        missing = set(expected_keys).difference(observed_keys)
        extra = set(observed_keys).difference(expected_keys)
        if missing or extra:
            raise ValueError(
                f"prediction IDs do not match evaluation: {len(missing)} missing, "
                f"{len(extra)} extra"
            )
        raise ValueError("prediction rows are not in canonical route/example order")
    return {
        route: [parsed[(route, case.example_id)] for case in suite.cases] for route in ROUTE_ORDER
    }


def _api_parts(value: Any) -> tuple[str, dict[str, Any]] | None:
    try:
        generation = _validate_generation(value)
    except (SourceDialogError, ValueError):
        return None
    name, arguments = next(iter(generation["api"].items()))
    return name, arguments


def score_one(case: EvaluationCase, route: str, output: str) -> dict[str, Any]:
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = strict_json_loads(output, reason="invalid_model_output")
    except SourceDialogError as exc:
        parse_error = str(exc)
    expected_name, expected_arguments = next(iter(case.expected["api"].items()))
    actual_parts = _api_parts(parsed)
    actual_name, actual_arguments = actual_parts or (None, None)
    arguments_exact = actual_arguments is not None and canonical_json(
        actual_arguments
    ) == canonical_json(expected_arguments)
    semantic_exact = parse_error is None and canonical_json(parsed) == case.expected_canonical
    return {
        "prediction_id": prediction_id(case.example_id, route, output),
        "example_id": case.example_id,
        "group_id": case.group_id,
        "expected_sha256": case.expected_sha256,
        "output_sha256": sha256_text(output),
        "json_parseable": parse_error is None,
        "parse_error": parse_error,
        "schema_valid": actual_parts is not None,
        "api_name_exact": actual_name == expected_name,
        "arguments_exact": arguments_exact,
        "semantic_exact": semantic_exact,
        "full_object_exact": semantic_exact,
        "canonical_exact": output == case.expected_canonical,
        "expected_api_name": expected_name,
        "actual_api_name": actual_name,
    }


def score_route(
    suite: EvaluationSuite,
    route: str,
    outputs: Sequence[str],
) -> dict[str, Any]:
    if route not in ROUTE_ORDER:
        raise ValueError(f"unsupported route {route!r}")
    if len(outputs) != len(suite.cases):
        raise ValueError("output count does not match evaluation")
    rows = [
        score_one(case, route, output) for case, output in zip(suite.cases, outputs, strict=True)
    ]
    count = len(rows)
    summary: dict[str, Any] = {"examples": count}
    for metric in SCORED_METRICS:
        matches = sum(bool(row[metric]) for row in rows)
        summary[f"{metric}_count"] = matches
        summary[f"{metric}_rate"] = matches / count
    summary["full_object_exact_count"] = summary["semantic_exact_count"]
    summary["full_object_exact_rate"] = summary["semantic_exact_rate"]
    summary["expected_api_counts"] = dict(
        sorted(Counter(str(row["expected_api_name"]) for row in rows).items())
    )
    return {
        "summary": summary,
        "predictions_content_sha256": sha256_text(
            canonical_json([row["prediction_id"] for row in rows])
        ),
        "examples": rows,
    }


def load_policy(path: Path) -> dict[str, Any]:
    policy = _read_json_object(path, label="policy")
    if policy.get("schema_version") != "molor.expert-promotion-policy.v2":
        raise ValueError("unsupported promotion policy schema")
    if policy.get("candidate_posture") != "quarantine_until_every_gate_passes":
        raise ValueError("policy must keep the candidate quarantined")
    dataset_contract = policy.get("dataset_contract")
    evaluation = policy.get("evaluation")
    if not isinstance(dataset_contract, dict) or dataset_contract.get("dataset_id") != DATASET_ID:
        raise ValueError("policy does not bind the v2 dataset")
    if not isinstance(evaluation, dict) or evaluation.get("route") != (
        "forced candidate, frozen base/null, and quarantined v1 on the same fresh holdout"
    ):
        raise ValueError("policy does not require the three-route fresh comparison")
    generation = evaluation.get("generation")
    if generation != {
        "do_sample": False,
        "enable_thinking": False,
        "max_prompt_tokens": 2048,
        "max_new_tokens": 192,
    }:
        raise ValueError("policy generation settings do not match the v2 frozen contract")
    rates = policy.get("minimum_rates")
    improvements = policy.get("minimum_absolute_improvement_over_base")
    if not isinstance(rates, dict) or set(rates) != set(SCORED_METRICS):
        raise ValueError("policy minimum rates do not match the frozen metrics")
    if not isinstance(improvements, dict) or set(improvements) != set(PRINCIPAL_METRICS):
        raise ValueError("policy base-improvement gates do not match the principal metrics")
    for label, values in (("minimum rate", rates), ("minimum improvement", improvements)):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            for value in values.values()
        ):
            raise ValueError(f"policy has an invalid {label}")
    if "must not trail v1" not in str(policy.get("fresh_v1_comparator_gate", "")):
        raise ValueError("policy v1 comparator gate is missing")
    if "Never tune" not in str(policy.get("reuse_policy", "")):
        raise ValueError("policy one-shot reuse warning is missing")
    base = policy.get("training_artifact_contract", {}).get("base_artifact")
    if (
        not isinstance(base, dict)
        or _SHA256.fullmatch(str(base.get("content_fingerprint"))) is None
    ):
        raise ValueError("policy base artifact fingerprint is missing")
    return policy


def load_metric_contract(path: Path) -> dict[str, Any]:
    contract = _read_json_object(path, label="metric_contract")
    if contract.get("schema_version") != METRIC_CONTRACT_SCHEMA:
        raise ValueError("unsupported metric contract schema")
    if contract.get("evaluation_schema") != EVALUATION_SCHEMA:
        raise ValueError("metric contract has the wrong evaluation schema")
    if contract.get("prediction_schema") != PREDICTION_SCHEMA:
        raise ValueError("metric contract has the wrong prediction schema")
    if contract.get("routes") != list(ROUTE_ORDER):
        raise ValueError("metric contract route order is not frozen")
    if set(contract.get("metrics", {})) != set(SCORED_METRICS):
        raise ValueError("metric contract does not define every frozen metric")
    if contract.get("generation") != {
        "do_sample": False,
        "enable_thinking": False,
        "max_prompt_tokens": 2048,
        "max_new_tokens": 192,
    }:
        raise ValueError("metric contract generation settings are not frozen")
    if "one-shot" not in str(contract.get("reuse_warning", "")):
        raise ValueError("metric contract one-shot warning is missing")
    return contract


def compare_routes(
    route_scores: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    if set(route_scores) != set(ROUTE_ORDER):
        raise ValueError("route scores must contain exactly the three policy routes")
    summaries = {route: route_scores[route]["summary"] for route in ROUTE_ORDER}
    candidate = summaries["forced_candidate"]
    base = summaries["frozen_base_null"]
    v1 = summaries["quarantined_v1"]
    candidate_minus_base = {
        metric: candidate[f"{metric}_rate"] - base[f"{metric}_rate"] for metric in SCORED_METRICS
    }
    candidate_minus_v1 = {
        metric: candidate[f"{metric}_rate"] - v1[f"{metric}_rate"] for metric in SCORED_METRICS
    }
    rate_gates = {
        metric: {
            "observed": candidate[f"{metric}_rate"],
            "required_minimum": float(minimum),
            "passed": candidate[f"{metric}_rate"] >= float(minimum),
        }
        for metric, minimum in policy["minimum_rates"].items()
    }
    base_gates = {
        metric: {
            "observed_absolute_improvement": candidate_minus_base[metric],
            "required_minimum": float(minimum),
            "passed": candidate_minus_base[metric] >= float(minimum),
        }
        for metric, minimum in policy["minimum_absolute_improvement_over_base"].items()
    }
    v1_gates = {
        metric: {
            "candidate_rate": candidate[f"{metric}_rate"],
            "quarantined_v1_rate": v1[f"{metric}_rate"],
            "candidate_minus_v1": candidate_minus_v1[metric],
            "passed": candidate_minus_v1[metric] >= 0.0,
        }
        for metric in PRINCIPAL_METRICS
    }
    all_passed = all(
        gate["passed"]
        for collection in (rate_gates, base_gates, v1_gates)
        for gate in collection.values()
    )
    return {
        "candidate_minus_frozen_base_absolute_rates": candidate_minus_base,
        "candidate_minus_quarantined_v1_absolute_rates": candidate_minus_v1,
        "gates": {
            "candidate_minimum_rates": rate_gates,
            "absolute_improvement_over_base": base_gates,
            "fresh_v1_non_regression": v1_gates,
            "all_expert_policy_gates_passed": all_passed,
        },
        "decision": (
            "expert_policy_gates_passed_but_router_registration_not_authorized"
            if all_passed
            else "keep_candidate_quarantined"
        ),
    }


def evaluate_outputs(
    suite: EvaluationSuite,
    outputs_by_route: Mapping[str, Sequence[str]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    scores = {route: score_route(suite, route, outputs_by_route[route]) for route in ROUTE_ORDER}
    return {"routes": scores, **compare_routes(scores, policy)}


def _adapter_targets(directory: Path) -> tuple[str, ...]:
    config = _read_json_object(directory / "adapter_config.json", label="adapter_config")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(item, str) and item for item in targets)
    ):
        raise ValueError(f"adapter has invalid target_modules: {directory}")
    return tuple(targets)


def _artifact_stack(model: Path, candidate: Path, v1: Path) -> dict[str, Any]:
    """Hash exact supplied paths; intentionally bypass desktop adapter discovery."""

    from molor.benchmark.lock import lock_backend_identity

    stack = lock_backend_identity(
        {
            "backend": "transformers",
            "model_artifact_path": str(model),
            "configured_expert_paths": [str(candidate), str(v1)],
            "experts": ["forced_candidate", "quarantined_v1"],
        },
        cache_path=Path("unused-no-digest-cache.json"),
    )
    if not stack["fully_content_addressed"]:
        raise ValueError(f"model/adapter artifact stack is incomplete: {stack['unresolved']}")
    return stack


def _artifact_by_role(stack: Mapping[str, Any], role: str, index: int = 0) -> Mapping[str, Any]:
    matches = [item for item in stack["artifacts"] if item.get("role") == role]
    try:
        return matches[index]
    except IndexError as exc:
        raise ValueError(f"artifact stack is missing {role} index {index}") from exc


def _validate_adapter_bindings(
    *,
    model: Path,
    candidate: Path,
    v1: Path,
    policy: Mapping[str, Any],
    stack: Mapping[str, Any],
) -> None:
    def is_quarantined(path: Path) -> bool:
        return any(
            part.casefold() == "quarantine" or part.casefold().endswith("-quarantine")
            for part in path.parts
        )

    if not is_quarantined(candidate) or not is_quarantined(v1):
        raise ValueError("both policy comparators must remain under explicit quarantine paths")
    candidate_config = _read_json_object(
        candidate / "adapter_config.json", label="candidate_adapter_config"
    )
    v1_config = _read_json_object(v1 / "adapter_config.json", label="v1_adapter_config")
    if candidate_config.get("name") != policy.get("candidate"):
        raise ValueError("candidate adapter name does not match promotion policy")
    if v1_config.get("name") != "taskmaster-coffee-api-v1":
        raise ValueError("v1 comparator is not the quarantined Taskmaster v1 adapter")
    if _adapter_targets(candidate) != _adapter_targets(v1):
        raise ValueError("candidate and v1 comparator target modules differ")
    for label, config in (("candidate", candidate_config), ("v1 comparator", v1_config)):
        raw_base = config.get("base_model_id")
        if not isinstance(raw_base, str) or Path(raw_base).expanduser().resolve() != model:
            raise ValueError(f"{label} adapter does not declare the frozen base path")
    locked_base = _artifact_by_role(stack, "base")
    expected_base = policy["training_artifact_contract"]["base_artifact"]["content_fingerprint"]
    if locked_base["content_fingerprint"] != expected_base:
        raise ValueError("frozen base content fingerprint does not match policy")


def _validated_descriptor_path(
    descriptor: Any,
    *,
    repository: Path,
    label: str,
) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{label} descriptor fields are invalid")
    raw_path = descriptor["path"]
    digest = descriptor["sha256"]
    size = descriptor["size_bytes"]
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or _SHA256.fullmatch(str(digest)) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise ValueError(f"{label} descriptor values are invalid")
    path = (repository / raw_path).resolve()
    try:
        path.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"{label} descriptor escapes the repository") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} artifact is missing: {path}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f"{label} descriptor does not match artifact bytes")
    return path


def _validate_self_hash(payload: Mapping[str, Any], *, label: str) -> str:
    value = payload.get("payload_sha256")
    if _SHA256.fullmatch(str(value)) is None:
        raise ValueError(f"{label} payload_sha256 is invalid")
    without_hash = dict(payload)
    del without_hash["payload_sha256"]
    expected = sha256_text(canonical_json(without_hash))
    if value != expected:
        raise ValueError(f"{label} payload_sha256 does not match canonical content")
    return str(value)


def validate_training_completion_receipt(
    path: Path,
    *,
    repository: Path,
    candidate: Path,
    policy_path: Path,
    build_report_path: Path,
    policy: Mapping[str, Any],
    stack: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind completed training to current inputs and independently rehashed bytes."""

    repository = repository.resolve()
    if path.resolve() != (candidate / "training_completion_receipt.json").resolve():
        raise ValueError("training completion receipt must be beside the candidate adapter")
    completion = _read_json_object(path, label="training_completion_receipt")
    required = {
        "schema_version",
        "status",
        "candidate",
        "preflight_receipt",
        "preflight_payload_sha256",
        "artifacts",
        "final_training_loss",
        "test_evaluation_opened",
        "registration_changed",
        "payload_sha256",
    }
    if set(completion) != required:
        raise ValueError("training completion receipt fields do not match the v2 contract")
    if completion["schema_version"] != TRAINING_COMPLETION_SCHEMA:
        raise ValueError("unsupported training completion receipt schema")
    if completion["status"] != "completed_quarantined_candidate_not_promotion_eligible":
        raise ValueError("training completion receipt does not preserve quarantine")
    if (
        completion["candidate"] != policy.get("candidate")
        or candidate.name != completion["candidate"]
    ):
        raise ValueError("training completion candidate does not match policy and adapter path")
    if completion["test_evaluation_opened"] is not False:
        raise ValueError("training completion says the v2 test was already opened")
    if completion["registration_changed"] is not False:
        raise ValueError("training completion says registration changed")
    loss = completion["final_training_loss"]
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise ValueError("training completion final loss is invalid")
    canonical_json(loss)
    completion_payload_sha256 = _validate_self_hash(completion, label="training completion")

    descriptors = completion["artifacts"]
    expected_names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_summary.json",
    )
    if not isinstance(descriptors, list) or len(descriptors) != len(expected_names):
        raise ValueError("training completion artifact list is incomplete")
    actual_descriptors: dict[str, dict[str, Any]] = {}
    for expected_name, descriptor in zip(expected_names, descriptors, strict=True):
        artifact_path = _validated_descriptor_path(
            descriptor,
            repository=repository,
            label=f"candidate {expected_name}",
        )
        if artifact_path != (candidate / expected_name).resolve():
            raise ValueError(f"training completion does not bind candidate {expected_name}")
        actual_descriptors[expected_name] = dict(descriptor)

    locked_candidate = _artifact_by_role(stack, "expert", 0)
    locked_files = {item["name"]: item for item in locked_candidate["files"]}
    if set(locked_files) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("independent candidate artifact lock has unexpected files")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        completion_descriptor = actual_descriptors[name]
        locked = locked_files[name]
        if (
            completion_descriptor["sha256"] != locked["sha256"]
            or completion_descriptor["size_bytes"] != locked["size_bytes"]
        ):
            raise ValueError(f"training completion and independent lock disagree on {name}")

    preflight_path = _validated_descriptor_path(
        completion["preflight_receipt"],
        repository=repository,
        label="training preflight receipt",
    )
    preflight = _read_json_object(preflight_path, label="training_preflight_receipt")
    if preflight.get("schema_version") != TRAINING_PREFLIGHT_SCHEMA:
        raise ValueError("unsupported training preflight receipt schema")
    preflight_payload_sha256 = _validate_self_hash(preflight, label="training preflight")
    if completion["preflight_payload_sha256"] != preflight_payload_sha256:
        raise ValueError("completion receipt does not bind the preflight payload")
    if preflight.get("dataset_id") != DATASET_ID or preflight.get("candidate") != candidate.name:
        raise ValueError("training preflight does not bind the v2 candidate")
    if preflight.get("posture") != "sealed_for_single_quarantined_training_run_only":
        raise ValueError("training preflight does not preserve one-run quarantine")
    checks = preflight.get("checks")
    if (
        not isinstance(checks, dict)
        or checks.get("v2_test_evaluation_opened") is not False
        or checks.get("quarantine_preserved") is not True
    ):
        raise ValueError("training preflight test/quarantine checks are invalid")
    input_descriptors = preflight.get("input_descriptors")
    if not isinstance(input_descriptors, dict):
        raise ValueError("training preflight input descriptors are missing")
    required_identities = policy.get("training_artifact_contract", {}).get(
        "required_file_identities"
    )
    identities = preflight.get("identities")
    if (
        not isinstance(required_identities, list)
        or not isinstance(identities, dict)
        or set(identities) != set(required_identities)
        or any(_SHA256.fullmatch(str(value)) is None for value in identities.values())
    ):
        raise ValueError("training preflight identities do not match the current policy contract")
    run_contract = preflight.get("run_contract")
    summary = _read_json_object(
        candidate / "training_summary.json", label="candidate_training_summary"
    )
    if not isinstance(run_contract, dict):
        raise ValueError("training preflight run contract is missing")
    expected_summary = {
        "expert": run_contract.get("expert"),
        "dataset_sha256": identities.get("train_dataset_sha256"),
        "examples": run_contract.get("examples"),
        "epochs": run_contract.get("epochs"),
        "seed": run_contract.get("seed"),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("training summary does not match the sealed run and dataset")
    current_inputs = {
        "policy": (policy_path, sha256_file(policy_path)),
        "build_report": (build_report_path, sha256_file(build_report_path)),
    }
    for name, (current_path, current_hash) in current_inputs.items():
        descriptor = input_descriptors.get(name)
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"path", "sha256", "size_bytes"}
            or descriptor.get("path") != current_path.name
            or descriptor.get("sha256") != current_hash
            or descriptor.get("size_bytes") != current_path.stat().st_size
        ):
            raise ValueError(f"training preflight does not bind the current {name}")
        identity_name = f"{name}_sha256"
        if identities.get(identity_name) != current_hash:
            raise ValueError(f"training preflight identity does not bind the current {name}")
    expected_trainer = policy.get("training_artifact_contract", {}).get("trainer_source_contract")
    preflight_trainer = input_descriptors.get("trainer_source_tree")
    if not isinstance(expected_trainer, dict) or not isinstance(preflight_trainer, dict):
        raise ValueError("training source-tree contract is missing")
    expected_trainer_hash = expected_trainer.get("content_fingerprint")
    if (
        _SHA256.fullmatch(str(expected_trainer_hash)) is None
        or preflight_trainer.get("content_fingerprint") != expected_trainer_hash
        or identities.get("trainer_source_tree_sha256") != expected_trainer_hash
    ):
        raise ValueError("training preflight source tree does not match the current policy")
    locked_base = _artifact_by_role(stack, "base")
    preflight_base = preflight.get("base_artifact")
    if (
        not isinstance(preflight_base, dict)
        or preflight_base.get("content_fingerprint") != locked_base["content_fingerprint"]
    ):
        raise ValueError("training preflight and evaluator disagree on frozen base bytes")
    return {
        "completion_receipt": {
            "path": str(path),
            "sha256": sha256_file(path),
            "payload_sha256": completion_payload_sha256,
        },
        "preflight_receipt": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "payload_sha256": preflight_payload_sha256,
        },
        "candidate_artifacts": actual_descriptors,
        "final_training_loss": loss,
        "test_evaluation_opened": False,
        "registration_changed": False,
    }


def load_build_report(path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    report = _read_json_object(path, label="build_report")
    if report.get("schema_version") != "molor.taskmaster-coffee-api-build-report.v2":
        raise ValueError("unsupported v2 build report")
    if report.get("dataset_id") != DATASET_ID:
        raise ValueError("build report does not describe the v2 dataset")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("build report outputs are missing")
    evaluation = outputs.get("test_evaluation")
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("path") != f"{DATASET_ID}.test.evaluation.jsonl"
        or not isinstance(evaluation.get("rows"), int)
        or evaluation["rows"] < 1
        or _SHA256.fullmatch(str(evaluation.get("sha256"))) is None
    ):
        raise ValueError("build report has no valid v2 test-evaluation descriptor")
    declared_inputs = policy["dataset_contract"]["artifact_identity_inputs"]
    build_inputs = report.get("build_inputs")
    if not isinstance(build_inputs, dict):
        raise ValueError("build report inputs are missing")
    for policy_key, report_key in (
        ("builder_sha256", "builder"),
        ("source_manifest_sha256", "source_manifest"),
        ("recipe_sha256", "recipe"),
    ):
        descriptor = build_inputs.get(report_key)
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("sha256") != declared_inputs[policy_key]
        ):
            raise ValueError(f"build report and policy disagree on {policy_key}")
    return report


def build_preaccess_receipt(
    *,
    evaluator_path: Path,
    metric_contract_path: Path,
    policy_path: Path,
    build_report_path: Path,
    build_report: Mapping[str, Any],
    stack: Mapping[str, Any],
    training_completion_binding: Mapping[str, Any],
    candidate: Path,
    v1: Path,
    evaluation_path: Path,
    dtype_name: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    expected_evaluation = build_report["outputs"]["test_evaluation"]
    return {
        "schema_version": PREACCESS_SCHEMA,
        "purpose": "freeze every comparator and metric identity before first v2 test access",
        "test_access_state": "sealed_not_opened_by_this_evaluator_yet",
        "one_shot_warning": ONE_SHOT_WARNING,
        "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
        "metric_contract": {
            "path": str(metric_contract_path),
            "sha256": sha256_file(metric_contract_path),
        },
        "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        "build_report": {
            "path": str(build_report_path),
            "sha256": sha256_file(build_report_path),
        },
        "evaluation": {
            "path": str(evaluation_path),
            "expected_sha256_from_unopened_build_report": expected_evaluation["sha256"],
            "expected_rows": expected_evaluation["rows"],
            "expected_size_bytes": expected_evaluation["size_bytes"],
        },
        "artifact_stack": stack,
        "training_completion": training_completion_binding,
        "forced_candidate_path": str(candidate),
        "quarantined_v1_path": str(v1),
        "generation": {
            "batch_size": batch_size,
            "device": device,
            "do_sample": False,
            "dtype": dtype_name,
            "enable_thinking": False,
            "max_new_tokens": 192,
            "max_prompt_tokens": 2048,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_all_routes(
    suite: EvaluationSuite,
    *,
    model: Path,
    candidate: Path,
    v1: Path,
    device: str,
    dtype_name: str,
    batch_size: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    import torch

    from molor.experimental.mixture import BASE_ROUTE, MixtureEngine

    targets = _adapter_targets(candidate)
    engine = MixtureEngine.from_pretrained(
        str(model),
        dtype=getattr(torch, dtype_name),
        device_map={"": device},
        target_modules=targets,
        local_files_only=True,
    )
    candidate_name = engine.load_expert(candidate)
    v1_name = engine.load_expert(v1)
    routes = {
        "forced_candidate": {candidate_name: 1.0},
        "frozen_base_null": {BASE_ROUTE: 1.0},
        "quarantined_v1": {v1_name: 1.0},
    }
    tokenizer = engine.tokenizer
    previous_padding_side = tokenizer.padding_side
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model_device = engine.model.get_input_embeddings().weight.device
    outputs_by_route: dict[str, list[str]] = {}
    engine.model.eval()
    try:
        with torch.inference_mode():
            for route_label in ROUTE_ORDER:
                route_outputs: list[str] = []
                with engine.use_route(routes[route_label], normalize=False):
                    for offset in range(0, len(suite.cases), batch_size):
                        batch = suite.cases[offset : offset + batch_size]
                        prompts = [
                            tokenizer.apply_chat_template(
                                list(case.messages),
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                            for case in batch
                        ]
                        encoded = tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=2048,
                        )
                        encoded = {key: value.to(model_device) for key, value in encoded.items()}
                        prompt_width = int(encoded["input_ids"].shape[1])
                        generated = engine.model.generate(
                            **encoded,
                            max_new_tokens=192,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        route_outputs.extend(
                            tokenizer.batch_decode(
                                generated[:, prompt_width:],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                outputs_by_route[route_label] = route_outputs
    finally:
        tokenizer.padding_side = previous_padding_side
        tokenizer.truncation_side = previous_truncation_side
    return outputs_by_route, {
        "route_bindings": {
            "forced_candidate": candidate_name,
            "frozen_base_null": BASE_ROUTE,
            "quarantined_v1": v1_name,
        },
        "greedy": True,
        "thinking": False,
        "dtype": dtype_name,
        "device": device,
        "batch_size": batch_size,
        "max_prompt_tokens": 2048,
        "max_new_tokens": 192,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-shot three-route Taskmaster coffee API v2 comparison."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--candidate-adapter", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--v1-adapter", type=Path, default=DEFAULT_V1_COMPARATOR)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--build-report", type=Path, default=DEFAULT_BUILD_REPORT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--metric-contract", type=Path, default=DEFAULT_METRIC_CONTRACT)
    parser.add_argument(
        "--training-completion-receipt",
        type=Path,
        help="defaults to <candidate-adapter>/training_completion_receipt.json",
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--preaccess-receipt", type=Path, default=DEFAULT_PREACCESS_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--acknowledge-one-shot",
        action="store_true",
        help="required acknowledgement that this run permanently exposes the v2 test",
    )
    return parser


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(arguments)
    if not args.acknowledge_one_shot:
        parser.error("--acknowledge-one-shot is required before the v2 test can be opened")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    evaluator_path = Path(__file__).resolve()
    model = args.model.expanduser().resolve()
    candidate = args.candidate_adapter.expanduser().resolve()
    v1 = args.v1_adapter.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    build_report_path = args.build_report.expanduser().resolve()
    policy_path = args.policy.expanduser().resolve()
    metric_contract_path = args.metric_contract.expanduser().resolve()
    completion_path = (
        args.training_completion_receipt.expanduser().resolve()
        if args.training_completion_receipt is not None
        else candidate / "training_completion_receipt.json"
    )

    # Every operation before the receipt deliberately avoids opening evaluation_path.
    policy = load_policy(policy_path)
    metric_contract = load_metric_contract(metric_contract_path)
    build_report = load_build_report(build_report_path, policy)
    stack = _artifact_stack(model, candidate, v1)
    _validate_adapter_bindings(
        model=model,
        candidate=candidate,
        v1=v1,
        policy=policy,
        stack=stack,
    )
    training_completion_binding = validate_training_completion_receipt(
        completion_path,
        repository=WORKSPACE,
        candidate=candidate,
        policy_path=policy_path,
        build_report_path=build_report_path,
        policy=policy,
        stack=stack,
    )
    receipt = build_preaccess_receipt(
        evaluator_path=evaluator_path,
        metric_contract_path=metric_contract_path,
        policy_path=policy_path,
        build_report_path=build_report_path,
        build_report=build_report,
        stack=stack,
        training_completion_binding=training_completion_binding,
        candidate=candidate,
        v1=v1,
        evaluation_path=evaluation_path,
        dtype_name=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
    )
    receipt_path = args.preaccess_receipt.expanduser().resolve()
    _write_json(receipt_path, receipt)
    receipt_sha256 = sha256_file(receipt_path)
    print(f"SEALED ONE-SHOT WARNING: {ONE_SHOT_WARNING}", flush=True)
    print(f"Pre-access receipt: {receipt_path} ({receipt_sha256})", flush=True)

    expected_evaluation = build_report["outputs"]["test_evaluation"]
    actual_evaluation_sha256 = sha256_file(evaluation_path)
    if actual_evaluation_sha256 != expected_evaluation["sha256"]:
        raise ValueError("v2 evaluation bytes do not match the sealed build report")
    if evaluation_path.stat().st_size != expected_evaluation["size_bytes"]:
        raise ValueError("v2 evaluation size does not match the sealed build report")
    suite = load_evaluation(evaluation_path)
    if len(suite.cases) != expected_evaluation["rows"]:
        raise ValueError("v2 evaluation row count does not match the sealed build report")
    if sha256_file(evaluation_path) != actual_evaluation_sha256:
        raise OSError("v2 evaluation bytes changed while the sealed suite was loading")

    generation: dict[str, Any]
    if args.predictions is not None:
        predictions_path = args.predictions.expanduser().resolve()
        outputs_by_route = load_predictions(predictions_path, suite)
        generation = {
            "source": "precomputed_content_addressed_predictions",
            "settings_bound_by_preaccess_receipt": receipt["generation"],
        }
    else:
        outputs_by_route, generation = generate_all_routes(
            suite,
            model=model,
            candidate=candidate,
            v1=v1,
            device=args.device,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
        )
        predictions_path = args.predictions_output.expanduser().resolve()
        write_predictions(predictions_path, suite, outputs_by_route)

    evaluation_result = evaluate_outputs(suite, outputs_by_route, policy)
    report = {
        "schema_version": REPORT_SCHEMA,
        "dataset_id": DATASET_ID,
        "one_shot_test_warning": {
            "status": "exposed_by_this_run",
            "warning": ONE_SHOT_WARNING,
            "reuse_policy": policy["reuse_policy"],
        },
        "identities": {
            "evaluator_sha256": sha256_file(evaluator_path),
            "metric_contract_sha256": sha256_file(metric_contract_path),
            "policy_sha256": sha256_file(policy_path),
            "build_report_sha256": sha256_file(build_report_path),
            "preaccess_receipt_sha256": receipt_sha256,
            "training_completion_receipt_sha256": training_completion_binding["completion_receipt"][
                "sha256"
            ],
            "training_preflight_receipt_sha256": training_completion_binding["preflight_receipt"][
                "sha256"
            ],
            "evaluation_sha256": actual_evaluation_sha256,
            "predictions_sha256": sha256_file(predictions_path),
            "tool_catalog_canonical_sha256": suite.tool_catalog_canonical_sha256,
            "artifact_stack_fingerprint": stack["stack_fingerprint"],
        },
        "paths": {
            "evaluation": str(evaluation_path),
            "predictions": str(predictions_path),
            "preaccess_receipt": str(receipt_path),
            "forced_candidate_adapter": str(candidate),
            "quarantined_v1_adapter": str(v1),
            "frozen_base": str(model),
        },
        "available_examples": len(suite.cases),
        "evaluated_examples": len(suite.cases),
        "generation": generation,
        "metric_contract": metric_contract,
        "artifact_stack": stack,
        **evaluation_result,
    }
    output_path = args.output.expanduser().resolve()
    _write_json(output_path, report)
    print(
        json.dumps({route: report["routes"][route]["summary"] for route in ROUTE_ORDER}, indent=2)
    )
    print(json.dumps(report["gates"], indent=2))
    print(f"Saved exposed-test report to {output_path} ({sha256_file(output_path)})", flush=True)


if __name__ == "__main__":
    main()
