"""Run the small held-out intent probe for the experimental prompt router.

This is deliberately *not* a broad router benchmark.  It is a compact, hand-labelled
OOD diagnostic intended to expose obvious intent-routing failures, lexical traps,
over-routing, and failure to abstain to the frozen base.  The probe JSONL is evaluation
only and must never be mixed into router training or feature-cache construction.

Only the router checkpoint, tokenizer, and frozen input-embedding tensor are loaded;
the causal language model and LoRA adapters are not materialized.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from molor.config import RouterConfig
from molor.experimental.autorouter import NULL_ROUTE_ID, ROUTER_CHECKPOINT_SCHEMA
from molor.router import TopKRouter
from molor.training.router import (
    RouterExample,
    RouterFeatureContract,
    extract_prompt_features,
    load_input_embedding,
    load_router_tokenizer,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "qwen3-1.7b-corpus-router-v1"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "Qwen3-1.7B"
DEFAULT_PROBE = Path(__file__).resolve().parent / "data" / "router-intent-probe-v1.jsonl"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "runs" / "evaluation" / "router-intent-probe-v1.json"
PROBE_PURPOSE = "small_intent_probe_never_training"


@dataclass(frozen=True, slots=True)
class ProbeItem:
    probe_id: str
    prompt: str
    category: str
    acceptable_routes: tuple[str, ...]
    expected_route: str | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.probe_id, "id"),
            (self.prompt, "prompt"),
            (self.category, "category"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        routes = tuple(self.acceptable_routes)
        if not routes or any(not isinstance(route, str) or not route for route in routes):
            raise ValueError("acceptable_routes must contain route IDs")
        if len(routes) != len(set(routes)):
            raise ValueError("acceptable_routes must not contain duplicates")
        if self.expected_route is not None:
            if not isinstance(self.expected_route, str) or not self.expected_route:
                raise ValueError("expected_route must be a route ID or null")
            if self.expected_route not in routes:
                raise ValueError("expected_route must also be acceptable")
        object.__setattr__(self, "acceptable_routes", routes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_probe(path: Path) -> tuple[ProbeItem, ...]:
    allowed_fields = {
        "id",
        "prompt",
        "category",
        "expected_route",
        "acceptable_routes",
        "purpose",
    }
    items: list[ProbeItem] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("row must be an object")
                unknown = set(raw).difference(allowed_fields)
                if unknown:
                    raise ValueError(f"unknown fields: {sorted(unknown)!r}")
                if raw.get("purpose") != PROBE_PURPOSE:
                    raise ValueError(
                        f"purpose must be {PROBE_PURPOSE!r} to prevent training reuse"
                    )
                acceptable = raw.get("acceptable_routes")
                expected = raw.get("expected_route")
                if acceptable is None:
                    acceptable = [expected] if expected is not None else None
                if not isinstance(acceptable, list):
                    raise ValueError("acceptable_routes must be an array")
                item = ProbeItem(
                    probe_id=raw["id"],
                    prompt=raw["prompt"],
                    category=raw["category"],
                    expected_route=expected,
                    acceptable_routes=tuple(acceptable),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            items.append(item)
    if not items:
        raise ValueError(f"probe contains no rows: {path}")
    ids = [item.probe_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("probe IDs must be unique")
    prompts = [item.prompt for item in items]
    if len(prompts) != len(set(prompts)):
        raise ValueError("probe prompts must be unique")
    return tuple(items)


def load_checkpoint(directory: Path) -> tuple[TopKRouter, dict[str, Any]]:
    config_path = directory / "router_config.json"
    weights_path = directory / "router.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"checkpoint must contain router_config.json and router.safetensors: {directory}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != ROUTER_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported router checkpoint sidecar")
    raw_config = payload.get("router_config")
    route_ids = payload.get("route_ids")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint has no router_config object")
    if not isinstance(route_ids, list) or any(
        not isinstance(route_id, str) or not route_id for route_id in route_ids
    ):
        raise ValueError("checkpoint has invalid route_ids")
    config = RouterConfig(**raw_config)
    if config.mode != "sequence" or len(route_ids) != config.num_routes:
        raise ValueError("probe requires a sequence router with a matching route axis")
    if tuple(route_ids).count(NULL_ROUTE_ID) != 1:
        raise ValueError(f"checkpoint must contain exactly one {NULL_ROUTE_ID!r} route")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError('install molor[hf] to evaluate the router') from exc
    tensors = load_file(str(weights_path), device="cpu")
    if not tensors or any(not key.startswith("network.") for key in tensors):
        raise ValueError("router weights must contain only network.* tensors")
    router = TopKRouter(config)
    router.network.load_state_dict(
        {key.removeprefix("network."): value for key, value in tensors.items()},
        strict=True,
    )
    router.float().eval().requires_grad_(False)
    return router, payload


def _feature_contract(payload: Mapping[str, Any], max_prompt_tokens: int | None) -> RouterFeatureContract:
    raw = payload.get("feature_contract")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no feature_contract")
    fields = {
        "max_tokens": raw.get("max_tokens"),
        "source": raw.get("source"),
        "pooling": raw.get("pooling"),
        "prompt_scope": raw.get("prompt_scope"),
        "padding": raw.get("padding"),
        "truncation": raw.get("truncation"),
        "add_special_tokens": raw.get("add_special_tokens"),
        "embedding_tensor": raw.get("embedding_tensor"),
    }
    if max_prompt_tokens is not None:
        fields["max_tokens"] = max_prompt_tokens
    return RouterFeatureContract(**fields)


def _validate_embedding_identity(
    payload: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    model = payload.get("model")
    expected = model.get("embedding") if isinstance(model, Mapping) else None
    if not isinstance(expected, Mapping):
        raise ValueError("checkpoint has no frozen input-embedding identity")
    for field_name in ("tensor_name", "shape", "dtype"):
        if actual.get(field_name) != expected.get(field_name):
            raise ValueError(f"input embedding {field_name} differs from router training")
    expected_artifact = expected.get("artifact")
    actual_artifact = actual.get("artifact")
    if not isinstance(expected_artifact, Mapping) or not isinstance(actual_artifact, Mapping):
        raise ValueError("input embedding artifact identity is incomplete")
    if actual_artifact.get("sha256") != expected_artifact.get("sha256"):
        raise ValueError("input embedding shard differs from the router training artifact")


def _policy_probabilities(
    logits: Tensor,
    *,
    route_ids: tuple[str, ...],
    enabled_experts: set[str],
    temperature: float,
    null_bias: float,
) -> Tensor:
    adjusted = logits.float().clone()
    null_index = route_ids.index(NULL_ROUTE_ID)
    adjusted[null_index] += null_bias
    allowed = torch.tensor(
        [route_id == NULL_ROUTE_ID or route_id in enabled_experts for route_id in route_ids],
        dtype=torch.bool,
    )
    adjusted.masked_fill_(~allowed, float("-inf"))
    return torch.softmax(adjusted / temperature, dim=-1)


def _entropy(probabilities: Tensor) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * positive.log()).sum().item())


def evaluate(
    router: TopKRouter,
    payload: Mapping[str, Any],
    items: Sequence[ProbeItem],
    features: Tensor,
    *,
    temperature: float,
    confidence_threshold: float,
    null_bias: float,
    top_k: int,
    enabled_experts: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route_ids = tuple(payload["route_ids"])
    if features.shape != (len(items), router.config.hidden_size):
        raise ValueError("probe features do not match item count and router hidden size")
    with torch.inference_mode():
        raw_logits = router.network(features.float())

    exact_total = 0
    exact_correct = 0
    acceptable_correct = 0
    top_k_acceptable = 0
    base_selections = 0
    threshold_abstentions = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    candidate_count = 1 + len(enabled_experts)
    effective_top_k = min(top_k, candidate_count)
    for item, logits in zip(items, raw_logits, strict=True):
        probabilities = _policy_probabilities(
            logits,
            route_ids=route_ids,
            enabled_experts=enabled_experts,
            temperature=temperature,
            null_bias=null_bias,
        )
        confidence = float(probabilities.max().item())
        ranked_indices = probabilities.topk(effective_top_k).indices.tolist()
        ranked_routes = tuple(route_ids[index] for index in ranked_indices)
        threshold_abstained = confidence < confidence_threshold
        if threshold_abstained:
            selected_routes = (NULL_ROUTE_ID,)
            primary_route = NULL_ROUTE_ID
            applied_weights = {NULL_ROUTE_ID: 1.0}
            threshold_abstentions += 1
        else:
            selected_routes = ranked_routes
            primary_route = ranked_routes[0]
            chosen = probabilities[ranked_indices]
            if effective_top_k == 1:
                chosen = torch.ones_like(chosen)
            else:
                chosen = chosen / chosen.sum()
            applied_weights = {
                route_ids[index]: float(weight.item())
                for index, weight in zip(ranked_indices, chosen, strict=True)
            }
        if primary_route == NULL_ROUTE_ID:
            base_selections += 1
        acceptable_match = primary_route in item.acceptable_routes
        acceptable_correct += int(acceptable_match)
        top_k_match = any(route in item.acceptable_routes for route in selected_routes)
        top_k_acceptable += int(top_k_match)
        exact_match: bool | None = None
        if item.expected_route is not None:
            exact_total += 1
            exact_match = primary_route == item.expected_route
            exact_correct += int(exact_match)
            confusion[item.expected_route][primary_route] += 1
        row = {
            "id": item.probe_id,
            "category": item.category,
            "prompt": item.prompt,
            "expected_route": item.expected_route,
            "acceptable_routes": list(item.acceptable_routes),
            "primary_route": primary_route,
            "selected_routes": list(selected_routes),
            "applied_weights": applied_weights,
            "probabilities": {
                route_id: float(probabilities[index].item())
                for index, route_id in enumerate(route_ids)
            },
            "confidence": confidence,
            "entropy": _entropy(probabilities),
            "threshold_abstained": threshold_abstained,
            "base_null_selected": primary_route == NULL_ROUTE_ID,
            "exact_match": exact_match,
            "acceptable_match": acceptable_match,
            "top_k_acceptable_match": top_k_match,
        }
        rows.append(row)
        category_rows[item.category].append(row)

    def ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    by_category: dict[str, dict[str, Any]] = {}
    for category, values in sorted(category_rows.items()):
        category_exact = [row for row in values if row["exact_match"] is not None]
        by_category[category] = {
            "count": len(values),
            "exact_expected_accuracy": ratio(
                sum(bool(row["exact_match"]) for row in category_exact),
                len(category_exact),
            ),
            "acceptable_route_accuracy": ratio(
                sum(bool(row["acceptable_match"]) for row in values),
                len(values),
            ),
            "base_null_selection_rate": ratio(
                sum(bool(row["base_null_selected"]) for row in values),
                len(values),
            ),
            "mean_confidence": sum(float(row["confidence"]) for row in values) / len(values),
        }

    metrics = {
        "prompt_count": len(items),
        "exact_labeled_count": exact_total,
        "exact_expected_accuracy": ratio(exact_correct, exact_total),
        "acceptable_route_accuracy": ratio(acceptable_correct, len(items)),
        "top_k_acceptable_recall": ratio(top_k_acceptable, len(items)),
        "abstention_definition": "primary route is base/null, naturally or by threshold",
        "abstention_count": base_selections,
        "abstention_rate": ratio(base_selections, len(items)),
        "threshold_abstention_count": threshold_abstentions,
        "threshold_abstention_rate": ratio(threshold_abstentions, len(items)),
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / len(rows),
        "mean_entropy": sum(float(row["entropy"]) for row in rows) / len(rows),
        "confusion_matrix_exact_labels_only": {
            expected: {predicted: count for predicted, count in sorted(predictions.items())}
            for expected, predictions in sorted(confusion.items())
        },
        "by_category": by_category,
    }
    return metrics, rows


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small held-out intent probe against the lightweight prompt router. "
            "This is not a broad benchmark and its prompts must not be used for training."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=_positive_float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.40)
    parser.add_argument("--null-bias", type=_finite_float, default=0.0)
    parser.add_argument("--top-k", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument(
        "--enabled-expert",
        action="append",
        default=[],
        help="repeat to restrict non-null candidates; default enables every checkpoint expert",
    )
    parser.add_argument("--feature-batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in [0, 1]")
    if args.max_prompt_tokens is not None and args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be positive")
    if args.feature_batch_size < 1:
        parser.error("--feature-batch-size must be positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    model = args.model.expanduser().resolve()
    probe_path = args.probe.expanduser().resolve()
    output = args.output.expanduser().resolve()
    print("Small intent probe only; these prompts are excluded from training.", flush=True)
    router, sidecar = load_checkpoint(checkpoint)
    route_ids = tuple(sidecar["route_ids"])
    known_experts = set(route_ids).difference({NULL_ROUTE_ID})
    enabled = set(args.enabled_expert) if args.enabled_expert else set(known_experts)
    unknown = enabled.difference(known_experts)
    if unknown:
        parser.error(f"enabled experts are absent from checkpoint: {sorted(unknown)!r}")
    items = load_probe(probe_path)
    unknown_labels = {
        route
        for item in items
        for route in item.acceptable_routes
        if route not in route_ids
    }
    if unknown_labels:
        raise ValueError(f"probe labels are absent from checkpoint: {sorted(unknown_labels)!r}")

    contract = _feature_contract(sidecar, args.max_prompt_tokens)
    print(f"Loading input embedding only from {model}", flush=True)
    embedding, embedding_metadata = load_input_embedding(
        model,
        tensor_name=contract.embedding_tensor,
    )
    _validate_embedding_identity(sidecar, embedding_metadata)
    tokenizer = load_router_tokenizer(model)
    feature_examples = [
        RouterExample(item.prompt, route_ids[0])
        for item in items
    ]
    features, _unused_labels = extract_prompt_features(
        feature_examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=route_ids,
        contract=contract,
        batch_size=args.feature_batch_size,
        progress=lambda message: print(message, flush=True),
    )
    del embedding

    metrics, rows = evaluate(
        router,
        sidecar,
        items,
        features,
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        null_bias=args.null_bias,
        top_k=args.top_k,
        enabled_experts=enabled,
    )
    report = {
        "schema_version": "molor.router-intent-probe-report.v1",
        "evaluation_kind": "small_held_out_intent_probe",
        "scope_warning": (
            "This 40-prompt hand-labelled OOD probe diagnoses obvious intent routing behavior. "
            "It is not a broad capability benchmark and does not measure answer quality."
        ),
        "training_exclusion": (
            "Probe prompts are evaluation-only and must not enter router training, tuning, "
            "feature caches, or prompt generation."
        ),
        "probe": {
            "path": str(probe_path),
            "sha256": _sha256(probe_path),
            "purpose": PROBE_PURPOSE,
            "count": len(items),
            "categories": dict(sorted(Counter(item.category for item in items).items())),
        },
        "checkpoint": {
            "directory": str(checkpoint),
            "config_sha256": _sha256(checkpoint / "router_config.json"),
            "weights_sha256": _sha256(checkpoint / "router.safetensors"),
            "route_ids": list(route_ids),
        },
        "model": {
            "directory": str(model),
            "embedding": embedding_metadata,
            "full_causal_model_loaded": False,
        },
        "policy": {
            "temperature": args.temperature,
            "confidence_threshold": args.confidence_threshold,
            "null_bias": args.null_bias,
            "top_k": args.top_k,
            "max_prompt_tokens": contract.max_tokens,
            "enabled_experts": [route for route in route_ids if route in enabled],
        },
        "metrics": metrics,
        "examples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
