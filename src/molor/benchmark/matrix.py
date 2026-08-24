"""Base/expert/auto/top-k benchmark matrix and router-oracle report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from molor.benchmark.backends import TeacherApiBackend
from molor.benchmark.models import BenchmarkSuite
from molor.benchmark.runner import BenchmarkRunner, RunOptions


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (result or "matrix")[:80]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _oracle(
    suite: BenchmarkSuite,
    forced_summaries: Sequence[tuple[str, Mapping[str, object]]],
) -> dict[str, object]:
    best: dict[tuple[str, int], float] = {}
    route_wins: dict[tuple[str, int], str] = {}
    case_metadata = {case.case_id: (case.category, case.weight) for case in suite.cases}
    for route, summary in forced_summaries:
        result_cells = summary.get("result_cells")
        if not isinstance(result_cells, Sequence) or isinstance(result_cells, (str, bytes)):
            raise ValueError(f"forced-route summary {route} has no verified result cells")
        for record in result_cells:
            if not isinstance(record, Mapping):
                continue
            score = record.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            key = str(record.get("case_id")), int(record.get("repetition", 0))
            previous = best.get(key)
            if previous is None or float(score) > previous:
                best[key] = float(score)
                route_wins[key] = route
    by_category: dict[str, list[tuple[float, float]]] = {}
    for (case_id, _repetition), score in best.items():
        category, weight = case_metadata[case_id]
        by_category.setdefault(category, []).append((score, weight))
    category_scores = {
        category: (
            sum(score * weight for score, weight in scores)
            / sum(weight for _score, weight in scores)
        )
        for category, scores in by_category.items()
    }
    included_weight = sum(suite.category_weights[name] for name in category_scores)
    overall = (
        sum(category_scores[name] * suite.category_weights[name] for name in category_scores)
        / included_weight
        if included_weight
        else 0.0
    )
    utilization: dict[str, int] = {}
    for route in route_wins.values():
        utilization[route] = utilization.get(route, 0) + 1
    return {
        "quality_index": overall * 1000,
        "category_scores": {
            category: score * 100 for category, score in sorted(category_scores.items())
        },
        "best_forced_route_utilization": dict(sorted(utilization.items())),
        "matched_cells": len(best),
    }


def run_matrix(
    suite: BenchmarkSuite,
    *,
    endpoint: str,
    label: str,
    output_root: Path,
    profile: str,
    categories: Sequence[str],
    limit: int | None,
    repetitions: int,
    experts: Sequence[str],
    include_top2: bool,
    route_temperature: float,
    confidence_threshold: float,
    null_bias: float,
    timeout: float,
    matrix_directory: Path | None = None,
) -> Path:
    discovery = TeacherApiBackend(
        endpoint,
        route_temperature=route_temperature,
        confidence_threshold=confidence_threshold,
        null_bias=null_bias,
        timeout=timeout,
    )
    identity = discovery.identity()
    available = identity.get("experts")
    available_experts = [str(item) for item in available] if isinstance(available, list) else []
    selected_experts = list(experts) if experts else available_experts
    unknown = set(selected_experts).difference(available_experts)
    if unknown:
        raise ValueError("unknown resident experts: " + ", ".join(sorted(unknown)))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    matrix_id = f"{timestamp}-{_slug(label)}"
    matrix_root = (
        matrix_directory.expanduser().resolve()
        if matrix_directory is not None
        else output_root.expanduser().resolve() / "matrices" / matrix_id
    )
    policies_root = matrix_root / "policies"
    policies_root.mkdir(parents=True, exist_ok=True)
    policies: list[tuple[str, str, int, bool]] = [("base", "base/null", 1, True)]
    policies.extend((f"expert-{_slug(name)}", name, 1, True) for name in selected_experts)
    policies.append(("auto-top1", "auto", 1, False))
    if include_top2:
        policies.append(("auto-top2", "auto", 2, False))
    matrix_manifest = {
        "schema_version": "molor.benchmark-matrix.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "matrix_id": matrix_id,
        "label": label,
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "suite_fingerprint": suite.fingerprint,
        "profile": profile,
        "categories": list(categories),
        "limit": limit,
        "repetitions": repetitions,
        "router_policy": {
            "temperature": route_temperature,
            "confidence_threshold": confidence_threshold,
            "null_bias": null_bias,
        },
        "teacher_identity": identity,
        "policies": [
            {"policy_id": policy_id, "route": route, "top_k": top_k}
            for policy_id, route, top_k, _forced in policies
        ],
    }
    _write_json(matrix_root / "matrix.json", matrix_manifest)
    print(
        f"Route matrix {matrix_id}: {len(policies)} policies over profile {profile}",
        flush=True,
    )
    base_summary: Path | None = None
    summaries: dict[str, dict[str, Any]] = {}
    forced_summaries: list[tuple[str, Mapping[str, object]]] = []
    for policy_id, route, top_k, forced in policies:
        backend = TeacherApiBackend(
            endpoint,
            route=route,
            route_top_k=top_k,
            route_temperature=route_temperature,
            confidence_threshold=confidence_threshold,
            null_bias=null_bias,
            timeout=timeout,
        )
        run_directory = policies_root / policy_id
        runner = BenchmarkRunner(
            suite,
            backend,
            output_root=policies_root,
            options=RunOptions(
                label=f"{label}-{policy_id}",
                profile=profile,
                categories=tuple(categories),
                limit=limit,
                repetitions=repetitions,
            ),
            run_directory=run_directory,
        )
        runner.run(baseline_summary=base_summary)
        if policy_id == "base":
            base_summary = runner.summary_path
        summaries[policy_id] = _read_json(runner.summary_path)
        if forced:
            forced_summaries.append((policy_id, summaries[policy_id]))
    oracle = _oracle(suite, forced_summaries)
    base_index = float(summaries["base"]["quality_index"])
    policy_table: dict[str, dict[str, object]] = {}
    for policy_id, summary in summaries.items():
        categories_summary = summary.get("categories")
        categories_summary = categories_summary if isinstance(categories_summary, Mapping) else {}
        policy_table[policy_id] = {
            "quality_index": summary.get("quality_index"),
            "delta_vs_base": float(summary["quality_index"]) - base_index,
            "category_scores": {
                name: raw.get("score_percent")
                for name, raw in categories_summary.items()
                if isinstance(raw, Mapping)
            },
            "tokens_per_second_median": summary.get("performance", {}).get(
                "tokens_per_second_median"
            ),
            "ttft_seconds_median": summary.get("performance", {}).get("ttft_seconds_median"),
            "acceptable_route_rate": summary.get("routing", {}).get("acceptable_top1_rate"),
            "summary_path": str(policies_root / policy_id / "summary.json"),
        }
    auto_index = float(summaries["auto-top1"]["quality_index"])
    matrix_summary = {
        "schema_version": "molor.benchmark-matrix-summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "matrix_id": matrix_id,
        "label": label,
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "suite_fingerprint": suite.fingerprint,
        "evaluation_fingerprint": summaries["base"].get("evaluation_fingerprint"),
        "workload_fingerprint": summaries["base"].get("workload_fingerprint"),
        "policies": policy_table,
        "forced_route_oracle": oracle,
        "auto_top1_oracle_regret": float(oracle["quality_index"]) - auto_index,
    }
    _write_json(matrix_root / "summary.json", matrix_summary)
    lines = [
        f"# {label} — route benchmark matrix",
        "",
        f"Evaluator: `{matrix_summary['evaluation_fingerprint']}`  ",
        f"Workload: `{matrix_summary['workload_fingerprint']}`",
        "",
        "| Policy | Quality index | Delta vs base | Route accuracy | Median tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy_id, summary in policy_table.items():
        route_rate = summary["acceptable_route_rate"]
        rendered_rate = "n/a" if route_rate is None else f"{float(route_rate) * 100:.1f}%"
        lines.append(
            f"| {policy_id} | {float(summary['quality_index']):.1f} | "
            f"{float(summary['delta_vs_base']):+.1f} | {rendered_rate} | "
            f"{float(summary['tokens_per_second_median'] or 0):.1f} |"
        )
    lines.extend(
        [
            "",
            f"Best-forced-route oracle: **{float(oracle['quality_index']):.1f} / 1000**",
            "",
            f"Auto top-1 oracle regret: **{matrix_summary['auto_top1_oracle_regret']:.1f} points**",
            "",
            "Best-forced-route utilization: "
            + ", ".join(
                f"{route}={count}"
                for route, count in oracle["best_forced_route_utilization"].items()
            ),
            "",
            "## Category score vectors",
            "",
            "| Category | " + " | ".join(policy_table) + " |",
            "|---|" + "---:|" * len(policy_table),
        ]
    )
    for category in suite.category_weights:
        values = []
        for policy in policy_table.values():
            category_scores = policy.get("category_scores")
            score = category_scores.get(category) if isinstance(category_scores, Mapping) else None
            values.append("n/a" if score is None else f"{float(score):.1f}%")
        lines.append(f"| {suite.category_labels[category]} | " + " | ".join(values) + " |")
    lines.append("")
    (matrix_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Matrix report: {matrix_root / 'report.md'}", flush=True)
    return matrix_root / "summary.json"


__all__ = ["run_matrix"]
