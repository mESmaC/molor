"""Audit a hierarchical router against a frozen route contract before generation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from molor.benchmark.models import load_suite
from molor.benchmark.routing import (
    BASE_ROUTE,
    evaluate_route_probabilities,
    load_route_evaluation_contract,
)
from molor.experimental.autorouter import AutoRouterPolicy
from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.experimental.hierarchical_prompt_router import HierarchicalPromptAutoRouter
from molor.identity import canonical_json_bytes, fingerprint_json
from molor.training.hierarchical_router import load_hierarchical_router_checkpoint
from molor.training.router import (
    RouterExample,
    extract_prompt_features,
    load_input_embedding,
    load_router_tokenizer,
)


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints" / "qwen3-0.6b-hierarchical-router-v1-fast-r8"
DEFAULT_SUITE = WORKSPACE / "benchmarks" / "suites" / "molor-core-v1.json"
DEFAULT_CONTRACT = (
    WORKSPACE / "experiments" / "expert-rack-v5" / "molor-core-v1-rack-v5-route-contract-v1.json"
)
AUDIT_SCHEMA = "molor.hierarchical-router-generalization-audit.v1"


def _checkpoint_paths(checkpoint_dir: Path) -> tuple[Path, Path]:
    config_path = checkpoint_dir / "router_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        model_dir = Path(payload["model"]["model_id"])
        taxonomy_path = Path(payload["taxonomy"]["artifact"]["path"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("router checkpoint has no readable pinned model/taxonomy paths") from exc
    return model_dir.expanduser().resolve(), taxonomy_path.expanduser().resolve()


def _probability_table(
    *,
    checkpoint: Any,
    features: torch.Tensor,
    case_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    with torch.inference_mode():
        probabilities = checkpoint.router.joint_probabilities(features).cpu()
    return {
        case_id: {
            route_id: float(probability)
            for route_id, probability in zip(
                checkpoint.axes.joint_routes,
                row.tolist(),
                strict=True,
            )
        }
        for case_id, row in zip(case_ids, probabilities, strict=True)
    }


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"router audit output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--route-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-overall-rate", type=float, default=None)
    parser.add_argument("--minimum-specialist-rate", type=float, default=None)
    parser.add_argument(
        "--minimum-base-contract-rate",
        type=float,
        default=0.80,
        help="minimum acceptable top-1 rate on base-only contract cases (default: 0.80)",
    )
    parser.add_argument("--maximum-base-selection-rate", type=float, default=None)
    return parser


def _optional_probability(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return value


def _base_contract_gate_failed(value: object, minimum: float) -> bool:
    """Fail closed for missing or malformed base-only contract evidence."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    rate = float(value)
    return not math.isfinite(rate) or not 0.0 <= rate <= 1.0 or rate < minimum


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.feature_batch_size < 1:
        raise ValueError("--feature-batch-size must be positive")
    minimum_overall = _optional_probability(args.minimum_overall_rate, "minimum overall rate")
    minimum_specialist = _optional_probability(
        args.minimum_specialist_rate, "minimum specialist rate"
    )
    minimum_base_contract = _optional_probability(
        args.minimum_base_contract_rate, "minimum base contract rate"
    )
    assert minimum_base_contract is not None
    maximum_base = _optional_probability(
        args.maximum_base_selection_rate, "maximum base selection rate"
    )

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    model_dir, taxonomy_path = _checkpoint_paths(checkpoint_dir)
    suite = load_suite(args.suite)
    taxonomy = load_expert_taxonomy(taxonomy_path)
    contract = load_route_evaluation_contract(
        args.route_contract,
        suite=suite,
        taxonomy=taxonomy,
    )
    embedding, embedding_metadata = load_input_embedding(model_dir)
    tokenizer = load_router_tokenizer(model_dir)
    checkpoint = load_hierarchical_router_checkpoint(
        checkpoint_dir,
        taxonomy_path=taxonomy_path,
        adapters=None,
        expected_model_id=model_dir,
        verify_adapter_artifacts=False,
        verify_model_artifacts=True,
        device=args.device,
    )
    feature_device = next(checkpoint.router.parameters()).device
    embedding_module = nn.Embedding.from_pretrained(embedding.to(feature_device), freeze=True)
    runtime = HierarchicalPromptAutoRouter(
        checkpoint=checkpoint,
        taxonomy=taxonomy,
        tokenizer=tokenizer,
        input_embeddings=embedding_module,
        available_experts=checkpoint.adapter_ids,
    )
    examples = [RouterExample(case.prompt, BASE_ROUTE) for case in suite.cases]
    features, _labels = extract_prompt_features(
        examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=(BASE_ROUTE,),
        contract=checkpoint.feature_contract,
        batch_size=args.feature_batch_size,
    )
    features = features.to(feature_device)
    case_ids = tuple(case.case_id for case in suite.cases)
    probability_rows = _probability_table(
        checkpoint=checkpoint,
        features=features,
        case_ids=case_ids,
    )
    audit = evaluate_route_probabilities(
        suite,
        contract,
        probability_rows,
        route_order=checkpoint.axes.joint_routes,
    )

    max_feature_delta = 0.0
    max_runtime_probability_delta = 0.0
    audited_predictions = {prediction["case_id"]: prediction for prediction in audit["predictions"]}
    for case, expected_feature in zip(suite.cases, features, strict=True):
        runtime_feature = runtime._prompt_features(case.prompt)[0]
        max_feature_delta = max(
            max_feature_delta,
            float((runtime_feature - expected_feature).abs().max().item()),
        )
        decision = runtime.decide(case.prompt, AutoRouterPolicy(top_k=1))
        runtime_probabilities = {
            (BASE_ROUTE if route_id == "base" else route_id): probability
            for route_id, probability in decision.probabilities
        }
        direct_probabilities = probability_rows[case.case_id]
        max_runtime_probability_delta = max(
            max_runtime_probability_delta,
            max(
                abs(runtime_probabilities[route_id] - direct_probabilities[route_id])
                for route_id in checkpoint.axes.joint_routes
            ),
        )
        runtime_selected = (
            BASE_ROUTE if decision.selected_routes[0] == "base" else decision.selected_routes[0]
        )
        if runtime_selected != audited_predictions[case.case_id]["selected_route"]:
            raise RuntimeError("runtime and direct checkpoint routing decisions diverged")

    semantic: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "content_sha256": checkpoint.sidecar["content_sha256"],
        },
        "model": {
            "model_id": str(model_dir),
            "embedding": embedding_metadata,
        },
        "suite": {
            "path": str(suite.path),
            "suite_id": suite.suite_id,
            "version": suite.version,
            "content_sha256": suite.fingerprint,
        },
        "routing_evaluation": contract.provenance(),
        "feature_contract": checkpoint.feature_contract.to_dict(),
        "parity": {
            "training_vs_runtime_feature_max_abs_delta": max_feature_delta,
            "direct_vs_runtime_probability_max_abs_delta": max_runtime_probability_delta,
            "runtime_decisions_match": True,
        },
        "audit": audit,
    }
    receipt = {**semantic, "content_sha256": fingerprint_json(semantic)}
    if args.output is not None:
        _write_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                "acceptable_top1_rate": audit["acceptable_top1_rate"],
                "specialist_acceptable_top1_rate": audit["specialist_acceptable_top1_rate"],
                "base_contract_acceptable_top1_rate": audit["base_contract_acceptable_top1_rate"],
                "base_selection_rate": audit["base_selection_rate"],
                "feature_max_abs_delta": max_feature_delta,
                "runtime_probability_max_abs_delta": max_runtime_probability_delta,
                "content_sha256": receipt["content_sha256"],
                "output": str(args.output.expanduser().resolve()) if args.output else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    failed = (
        (minimum_overall is not None and audit["acceptable_top1_rate"] < minimum_overall)
        or (
            minimum_specialist is not None
            and audit["specialist_acceptable_top1_rate"] < minimum_specialist
        )
        or _base_contract_gate_failed(
            audit["base_contract_acceptable_top1_rate"], minimum_base_contract
        )
        or (maximum_base is not None and audit["base_selection_rate"] > maximum_base)
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
