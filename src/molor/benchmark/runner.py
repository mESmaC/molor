"""Resumable benchmark execution, aggregation, comparison, and reports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from molor.benchmark.backends import BenchmarkBackend, GenerationRequest
from molor.benchmark.evaluation import BenchmarkEvaluator, UnscorableEvaluation
from molor.benchmark.lock import lock_backend_identity
from molor.benchmark.models import BenchmarkCase, BenchmarkSuite, canonical_json
from molor.benchmark.routing import BASE_ROUTE, RouteEvaluationContract
from molor.benchmark.scoring import SCORER_VERSION, score_response


RUN_SCHEMA = "molor.benchmark-run.v2"
RESULT_SCHEMA = "molor.benchmark-result.v2"
SUMMARY_SCHEMA = "molor.benchmark-summary.v2"
COMPARISON_SCHEMA = "molor.benchmark-comparison.v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (result or "benchmark")[:80]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()


def _append_history_once(path: Path, value: Mapping[str, object]) -> None:
    run_id = value.get("run_id")
    for existing in _read_jsonl(path):
        if existing.get("run_id") == run_id:
            return
    _append_jsonl(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("molor", "torch", "transformers", "peft", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _gpu_snapshot() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    gpus: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        index, name, uuid, driver, total, used, power = fields
        try:
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "uuid": uuid,
                    "driver_version": driver,
                    "memory_total_mib": float(total),
                    "memory_used_mib": float(used),
                    "power_limit_watts": float(power),
                }
            )
        except ValueError:
            continue
    return gpus


@dataclass(frozen=True, slots=True)
class RunOptions:
    label: str
    profile: str = "full"
    categories: tuple[str, ...] = ()
    limit: int | None = None
    repetitions: int = 1
    warmup: bool = True
    continue_on_error: bool = True
    max_new_tokens_override: int | None = None
    enable_thinking_override: bool | None = None

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.max_new_tokens_override is not None and self.max_new_tokens_override < 1:
            raise ValueError("max_new_tokens_override must be positive")
        if self.enable_thinking_override is not None and not isinstance(
            self.enable_thinking_override, bool
        ):
            raise ValueError("enable_thinking_override must be boolean or None")


def _request_settings(
    suite: BenchmarkSuite,
    case: BenchmarkCase,
    repetition: int,
    *,
    max_new_tokens_override: int | None,
    enable_thinking_override: bool | None,
) -> dict[str, object]:
    generation = suite.generation
    max_new_tokens = (
        max_new_tokens_override or case.max_new_tokens or int(generation["max_new_tokens"])
    )
    seed = int.from_bytes(
        hashlib.sha256(
            f"{suite.fingerprint}:{case.case_id}:{repetition}:{int(generation['seed'])}".encode(
                "utf-8"
            )
        ).digest()[:4],
        "big",
    )
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "top_k": int(generation["top_k"]),
        "enable_thinking": (
            enable_thinking_override
            if enable_thinking_override is not None
            else bool(generation["enable_thinking"])
        ),
        "context_token_budget": generation.get("context_token_budget"),
        "cache_mode": str(generation["cache_mode"]),
        "seed": seed,
    }


def _build_workload_contract(
    suite: BenchmarkSuite,
    cases: Sequence[BenchmarkCase],
    options: RunOptions,
) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for repetition in range(options.repetitions):
        for case in cases:
            cells.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "case_weight": case.weight,
                    "repetition": repetition,
                    "request": _request_settings(
                        suite,
                        case,
                        repetition,
                        max_new_tokens_override=options.max_new_tokens_override,
                        enable_thinking_override=options.enable_thinking_override,
                    ),
                }
            )
    return {
        "suite_fingerprint": suite.fingerprint,
        "profile": options.profile,
        "categories": list(options.categories),
        "limit": options.limit,
        "case_ids": [case.case_id for case in cases],
        "repetitions": options.repetitions,
        "generation": dict(suite.generation),
        "max_new_tokens_override": options.max_new_tokens_override,
        "enable_thinking_override": options.enable_thinking_override,
        "cells": cells,
    }


def _fingerprint_workload(contract: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def _cell_id(
    *,
    suite_fingerprint: str,
    evaluation_fingerprint: str,
    workload_fingerprint: str,
    stack_fingerprint: str,
    case_id: str,
    repetition: int,
    request: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "suite_fingerprint": suite_fingerprint,
                "evaluation_fingerprint": evaluation_fingerprint,
                "workload_fingerprint": workload_fingerprint,
                "stack_fingerprint": stack_fingerprint,
                "case_id": case_id,
                "repetition": repetition,
                "request": dict(request),
            }
        ).encode("utf-8")
    ).hexdigest()


def _contract_cells(
    contract: Mapping[str, object],
) -> dict[tuple[str, int], dict[str, object]]:
    raw_cells = contract.get("cells")
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
        raise ValueError("benchmark workload contract has no cell list")
    cells: dict[tuple[str, int], dict[str, object]] = {}
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark workload contains a malformed cell")
        case_id = raw.get("case_id")
        repetition = raw.get("repetition")
        request = raw.get("request")
        if (
            not isinstance(case_id, str)
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition < 0
            or not isinstance(request, Mapping)
        ):
            raise ValueError("benchmark workload contains a malformed cell")
        key = case_id, repetition
        if key in cells:
            raise ValueError(f"duplicate benchmark workload cell {key}")
        cells[key] = dict(raw)
    return cells


def _record_matches_cell(
    record: Mapping[str, object],
    manifest: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    case_id = expected.get("case_id")
    repetition = expected.get("repetition")
    request = expected.get("request")
    if not isinstance(case_id, str) or not isinstance(repetition, int):
        return False
    if not isinstance(request, Mapping):
        return False
    expected_cell_id = _cell_id(
        suite_fingerprint=str(manifest.get("suite_fingerprint")),
        evaluation_fingerprint=str(manifest.get("evaluation_fingerprint")),
        workload_fingerprint=str(manifest.get("workload_fingerprint")),
        stack_fingerprint=str(manifest.get("stack_fingerprint")),
        case_id=case_id,
        repetition=repetition,
        request=request,
    )
    return (
        record.get("schema_version") == RESULT_SCHEMA
        and record.get("run_id") == manifest.get("run_id")
        and record.get("suite_fingerprint") == manifest.get("suite_fingerprint")
        and record.get("evaluation_fingerprint") == manifest.get("evaluation_fingerprint")
        and record.get("workload_fingerprint") == manifest.get("workload_fingerprint")
        and record.get("stack_fingerprint") == manifest.get("stack_fingerprint")
        and record.get("case_id") == case_id
        and record.get("repetition") == repetition
        and record.get("request") == request
        and record.get("cell_id") == expected_cell_id
    )


class BenchmarkRunner:
    def __init__(
        self,
        suite: BenchmarkSuite,
        backend: BenchmarkBackend,
        *,
        output_root: Path,
        options: RunOptions,
        run_directory: Path | None = None,
        evaluator: BenchmarkEvaluator | None = None,
        route_contract: RouteEvaluationContract | None = None,
    ) -> None:
        self.suite = suite
        self.backend = backend
        self.options = options
        self.evaluator = evaluator
        self.route_contract = route_contract
        if route_contract is not None:
            route_contract.validate_suite(suite)
        self.routing_evaluation = route_contract.provenance() if route_contract else None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        self.run_id = f"{timestamp}-{_slug(options.label)}"
        self.run_directory = (
            run_directory.expanduser().resolve()
            if run_directory is not None
            else (output_root.expanduser().resolve() / self.run_id)
        )
        self.results_path = self.run_directory / "results.jsonl"
        self.run_path = self.run_directory / "run.json"
        self.summary_path = self.run_directory / "summary.json"
        self.report_path = self.run_directory / "report.md"
        self.scores_path = self.run_directory / "scores.csv"
        self.stack_fingerprint = ""
        self.workload_fingerprint = ""
        evaluator_identity = evaluator.fingerprint if evaluator is not None else SCORER_VERSION
        self.evaluation_fingerprint = hashlib.sha256(
            f"{self.suite.fingerprint}:{evaluator_identity}".encode("utf-8")
        ).hexdigest()

    def run(self, *, baseline_summary: Path | None = None) -> dict[str, Any]:
        cases = self.suite.select(
            profile=self.options.profile,
            categories=self.options.categories,
            limit=self.options.limit,
        )
        if self.evaluator is not None:
            for case in cases:
                self.evaluator.validate_case(case)
        workload_contract = _build_workload_contract(self.suite, cases, self.options)
        workload_fingerprint = _fingerprint_workload(workload_contract)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        if self.run_path.exists():
            run_manifest = _read_json(self.run_path)
            if run_manifest.get("suite_fingerprint") != self.suite.fingerprint:
                raise ValueError("cannot resume a run with a different benchmark suite")
            if run_manifest.get("evaluation_fingerprint") != self.evaluation_fingerprint:
                raise ValueError(
                    "cannot resume a run after evaluator semantics changed; start a new run"
                )
            expected_evaluator = (
                self.evaluator.manifest_dict() if self.evaluator is not None else None
            )
            if run_manifest.get("evaluator") != expected_evaluator:
                raise ValueError(
                    "cannot resume after the benchmark evaluator configuration changed"
                )
            if run_manifest.get("workload_fingerprint") != workload_fingerprint:
                raise ValueError(
                    "cannot resume with a different profile, cases, repetitions, or generation "
                    "settings; start a new run"
                )
            if run_manifest.get("workload_contract") != workload_contract:
                raise ValueError("cannot resume because the workload contract changed")
            if run_manifest.get("routing_evaluation") != self.routing_evaluation:
                raise ValueError(
                    "cannot resume after the routing evaluation contract changed"
                )
            current_lock = lock_backend_identity(
                self.backend.identity(),
                cache_path=self.run_directory.parent / ".artifact-hash-cache.json",
            )
            if current_lock["stack_fingerprint"] != run_manifest.get("stack_fingerprint"):
                raise ValueError(
                    "cannot resume because the model/adapter/router bytes changed; start a new run"
                )
            _atomic_json(self.run_directory / "stack.lock.json", current_lock)
            self.run_id = str(run_manifest["run_id"])
        else:
            identity = self.backend.identity()
            stack_lock = lock_backend_identity(
                identity,
                cache_path=self.run_directory.parent / ".artifact-hash-cache.json",
            )
            _atomic_json(self.run_directory / "stack.lock.json", stack_lock)
            _atomic_json(self.run_directory / "suite.snapshot.json", self.suite.raw)
            run_manifest = {
                "schema_version": RUN_SCHEMA,
                "created_at": _now(),
                "run_id": self.run_id,
                "label": self.options.label,
                "suite_id": self.suite.suite_id,
                "suite_version": self.suite.version,
                "suite_path": str(self.suite.path),
                "suite_fingerprint": self.suite.fingerprint,
                "scorer_version": SCORER_VERSION,
                "evaluator": (
                    self.evaluator.manifest_dict() if self.evaluator is not None else None
                ),
                "evaluation_fingerprint": self.evaluation_fingerprint,
                "workload_fingerprint": workload_fingerprint,
                "workload_contract": workload_contract,
                "routing_evaluation": self.routing_evaluation,
                "profile": self.options.profile,
                "categories": list(self.options.categories),
                "case_ids": [case.case_id for case in cases],
                "repetitions": self.options.repetitions,
                "generation": dict(self.suite.generation),
                "max_new_tokens_override": self.options.max_new_tokens_override,
                "enable_thinking_override": self.options.enable_thinking_override,
                "backend": identity,
                "stack_fingerprint": stack_lock["stack_fingerprint"],
                "stack_fully_content_addressed": stack_lock["fully_content_addressed"],
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "packages": _package_versions(),
                    "gpus_before": _gpu_snapshot(),
                },
            }
            _atomic_json(self.run_path, run_manifest)
        self.stack_fingerprint = str(run_manifest["stack_fingerprint"])
        self.workload_fingerprint = str(run_manifest["workload_fingerprint"])
        expected_cells = _contract_cells(workload_contract)
        selected_case_lookup = {case.case_id: case for case in cases}
        existing = _read_jsonl(self.results_path)
        completed: set[tuple[str, int]] = set()
        for record in existing:
            case_id = record.get("case_id")
            repetition = record.get("repetition")
            if not isinstance(case_id, str) or not isinstance(repetition, int):
                continue
            key = case_id, repetition
            expected = expected_cells.get(key)
            score = record.get("score")
            case = selected_case_lookup.get(case_id)
            rescored = self._validated_stored_score(case, record) if case is not None else None
            if (
                expected is not None
                and _record_matches_cell(record, run_manifest, expected)
                and record.get("error") is None
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                and rescored is not None
                and math.isclose(float(score), rescored, abs_tol=1e-12)
            ):
                completed.add(key)
        print(
            f"MoLoRA benchmark {self.suite.suite_id} {self.suite.version} | "
            f"{len(cases)} cases x {self.options.repetitions} | run {self.run_id}",
            flush=True,
        )
        print(f"Results: {self.run_directory}", flush=True)

        pending = [
            (case, repetition)
            for repetition in range(self.options.repetitions)
            for case in cases
            if (case.case_id, repetition) not in completed
        ]
        if self.options.warmup and pending and not completed:
            self._warmup(pending[0][0])
        total = len(cases) * self.options.repetitions
        done = len(completed)
        for case, repetition in pending:
            done += 1
            result = self._run_case(case, repetition)
            _append_jsonl(self.results_path, result)
            if result.get("error"):
                print(
                    f"[{done:>3}/{total}] {case.case_id:<10} ERROR  {result['error']}",
                    flush=True,
                )
                if not self.options.continue_on_error:
                    raise RuntimeError(str(result["error"]))
            else:
                score = float(result["score"])
                print(
                    f"[{done:>3}/{total}] {case.case_id:<10} {score * 100:6.1f}  {case.category}",
                    flush=True,
                )

        records = _read_jsonl(self.results_path)
        summary = summarize_run(
            self.suite,
            _read_json(self.run_path),
            records,
            evaluator=self.evaluator,
            route_contract=self.route_contract,
        )
        summary["environment_after"] = {"gpus": _gpu_snapshot()}
        _atomic_json(self.summary_path, summary)
        _write_scores_csv(self.scores_path, summary)
        comparison: dict[str, Any] | None = None
        if baseline_summary is not None:
            comparison = compare_runs(
                baseline_summary.expanduser().resolve(),
                self.summary_path,
            )
            _atomic_json(self.run_directory / "comparison.json", comparison)
        self.report_path.write_text(
            render_markdown_report(summary, comparison=comparison),
            encoding="utf-8",
        )
        history = self.run_directory.parent / "history.jsonl"
        history_row = {
            "schema_version": "molor.benchmark-history.v1",
            "created_at": summary["created_at"],
            "run_id": summary["run_id"],
            "label": summary["label"],
            "suite_id": summary["suite_id"],
            "suite_version": summary["suite_version"],
            "suite_fingerprint": summary["suite_fingerprint"],
            "evaluation_fingerprint": summary["evaluation_fingerprint"],
            "workload_fingerprint": summary["workload_fingerprint"],
            "quality_index": summary["quality_index"],
            "coverage": summary["coverage"],
            "summary_path": str(self.summary_path),
        }
        if summary["comparable"]:
            _append_history_once(history, history_row)
        else:
            print(
                "Run is incomplete and was not added to the score history; resume it after "
                "fixing backend errors.",
                flush=True,
            )
        print_summary(summary, comparison=comparison)
        return summary

    def _validated_stored_score(
        self,
        case: BenchmarkCase,
        record: Mapping[str, object],
    ) -> float | None:
        response = record.get("response")
        if not isinstance(response, str):
            return None
        try:
            if self.evaluator is None:
                return score_response(response, case.scorer).score
            repetition = record.get("repetition")
            if not isinstance(repetition, int):
                return None
            outcome = self.evaluator.validate_evidence(
                case,
                response,
                record.get("evaluation_evidence"),
                sample_id=f"{self.run_id}:{case.case_id}:{repetition}",
            )
            return outcome.result.score
        except Exception:
            return None

    def _warmup(self, case: BenchmarkCase) -> None:
        print("Warming up model (not scored)...", flush=True)
        generation = self.suite.generation
        request = GenerationRequest(
            messages=tuple(case.messages(self.suite.default_system)),
            max_new_tokens=min(8, case.max_new_tokens or int(generation["max_new_tokens"])),
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            enable_thinking=False,
            context_token_budget=generation.get("context_token_budget"),
            cache_mode=str(generation.get("cache_mode", "dynamic")),
            seed=int(generation["seed"]) - 1,
            run_id=self.run_id,
            case_id="__warmup__",
            repetition=0,
        )
        self.backend.generate(request)

    def _run_case(self, case: BenchmarkCase, repetition: int) -> dict[str, Any]:
        settings = _request_settings(
            self.suite,
            case,
            repetition,
            max_new_tokens_override=self.options.max_new_tokens_override,
            enable_thinking_override=self.options.enable_thinking_override,
        )
        request = GenerationRequest(
            messages=tuple(case.messages(self.suite.default_system)),
            max_new_tokens=int(settings["max_new_tokens"]),
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            top_k=int(settings["top_k"]),
            enable_thinking=bool(settings["enable_thinking"]),
            context_token_budget=(
                int(settings["context_token_budget"])
                if isinstance(settings["context_token_budget"], int)
                else None
            ),
            cache_mode=str(settings["cache_mode"]),
            seed=int(settings["seed"]),
            run_id=self.run_id,
            case_id=case.case_id,
            repetition=repetition,
        )
        base: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "created_at": _now(),
            "run_id": self.run_id,
            "suite_fingerprint": self.suite.fingerprint,
            "scorer_version": SCORER_VERSION,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "stack_fingerprint": self.stack_fingerprint,
            "case_id": case.case_id,
            "category": case.category,
            "case_weight": case.weight,
            "tags": list(case.tags),
            "acceptable_routes": list(case.acceptable_routes),
            "repetition": repetition,
            "request": settings,
        }
        base["cell_id"] = _cell_id(
            suite_fingerprint=self.suite.fingerprint,
            evaluation_fingerprint=self.evaluation_fingerprint,
            workload_fingerprint=self.workload_fingerprint,
            stack_fingerprint=self.stack_fingerprint,
            case_id=case.case_id,
            repetition=repetition,
            request=settings,
        )
        try:
            response = self.backend.generate(request)
        except Exception as exc:  # preserve a resumable per-case generation failure
            return {
                **base,
                "response": None,
                "score": None,
                "passed": False,
                "extracted": None,
                "details": {},
                "telemetry": {},
                "routing_decision": None,
                "generation_id": None,
                "provider_metadata": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        generated: dict[str, Any] = {
            **base,
            "response": response.text,
            "telemetry": {
                "elapsed_seconds": response.elapsed_seconds,
                "time_to_first_token_seconds": response.time_to_first_token_seconds,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "tokens_per_second": response.tokens_per_second,
                "finish_reason": response.finish_reason,
            },
            "routing_decision": response.routing_decision,
            "generation_id": response.generation_id,
            "provider_metadata": response.provider_metadata,
        }
        try:
            if self.evaluator is None:
                score_record = score_response(response.text, case.scorer).to_dict()
            else:
                outcome = self.evaluator.evaluate(
                    case,
                    response.text,
                    sample_id=f"{self.run_id}:{case.case_id}:{repetition}",
                )
                score_record = outcome.to_record()
            return {**generated, **score_record, "evaluation_error": None, "error": None}
        except UnscorableEvaluation as exc:
            return {
                **generated,
                **exc.to_record(),
                "error": f"UnscorableEvaluation: {exc}",
            }
        except Exception as exc:
            return {
                **generated,
                "score": None,
                "passed": False,
                "extracted": None,
                "details": {},
                "evaluation_evidence": None,
                "evaluation_error": f"{type(exc).__name__}: {exc}",
                "error": f"EvaluationError: {type(exc).__name__}: {exc}",
            }


def _weighted_mean(records: Iterable[Mapping[str, object]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        score = record.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            score = 0.0
        weight = record.get("case_weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            weight = 1.0
        numerator += float(score) * float(weight)
        denominator += float(weight)
    return numerator / denominator if denominator else None


def _overall_score(
    category_scores: Mapping[str, float],
    category_weights: Mapping[str, float],
) -> float:
    included = {name: category_weights[name] for name in category_scores}
    weight_sum = sum(included.values())
    if weight_sum <= 0:
        return 0.0
    return sum(category_scores[name] * weight for name, weight in included.items()) / weight_sum


def _bootstrap_quality_ci(
    records: Sequence[Mapping[str, object]],
    suite: BenchmarkSuite,
    *,
    seed: int,
    iterations: int = 2000,
) -> dict[str, float] | None:
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        if isinstance(record.get("score"), (int, float)) and not isinstance(
            record.get("score"), bool
        ):
            by_category[str(record["category"])].append(record)
    if not by_category:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        category_scores: dict[str, float] = {}
        for category, category_records in by_category.items():
            sample = [rng.choice(category_records) for _ in category_records]
            mean = _weighted_mean(sample)
            if mean is not None:
                category_scores[category] = mean
        samples.append(_overall_score(category_scores, suite.category_weights) * 1000)
    lower = _percentile(samples, 0.025)
    upper = _percentile(samples, 0.975)
    if lower is None or upper is None:
        return None
    return {"level": 0.95, "lower": lower, "upper": upper, "method": "stratified-bootstrap"}


def summarize_run(
    suite: BenchmarkSuite,
    run_manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    evaluator: BenchmarkEvaluator | None = None,
    route_contract: RouteEvaluationContract | None = None,
) -> dict[str, Any]:
    workload_contract = run_manifest.get("workload_contract")
    workload_fingerprint = run_manifest.get("workload_fingerprint")
    if not isinstance(workload_contract, Mapping) or not isinstance(workload_fingerprint, str):
        raise ValueError("run manifest has no workload contract")
    if _fingerprint_workload(workload_contract) != workload_fingerprint:
        raise ValueError("run manifest workload fingerprint does not match its contract")
    if route_contract is not None:
        route_contract.validate_suite(suite)
    routing_evaluation = route_contract.provenance() if route_contract is not None else None
    if run_manifest.get("routing_evaluation") != routing_evaluation:
        raise ValueError("run manifest routing evaluation contract does not match this summary")
    expected_cells = _contract_cells(workload_contract)
    case_lookup = {case.case_id: case for case in suite.cases}
    for key, expected in expected_cells.items():
        case = case_lookup.get(key[0])
        request = expected.get("request")
        if case is None or not isinstance(request, Mapping):
            raise ValueError(f"workload references unknown benchmark cell {key}")
        canonical_request = _request_settings(
            suite,
            case,
            key[1],
            max_new_tokens_override=workload_contract.get("max_new_tokens_override")
            if isinstance(workload_contract.get("max_new_tokens_override"), int)
            else None,
            enable_thinking_override=workload_contract.get("enable_thinking_override")
            if isinstance(workload_contract.get("enable_thinking_override"), bool)
            else None,
        )
        if (
            expected.get("category") != case.category
            or expected.get("case_weight") != case.weight
            or dict(request) != canonical_request
        ):
            raise ValueError(f"workload cell {key} does not match the suite snapshot")
    latest: dict[tuple[str, int], Mapping[str, object]] = {}
    for record in records:
        case_id = record.get("case_id")
        repetition = record.get("repetition")
        if not isinstance(case_id, str) or not isinstance(repetition, int):
            continue
        key = case_id, repetition
        expected = expected_cells.get(key)
        if expected is None or not _record_matches_cell(record, run_manifest, expected):
            continue
        previous = latest.get(key)
        if previous is None or previous.get("error") is not None or record.get("error") is None:
            latest[key] = record
    relevant: list[dict[str, object]] = []
    for key, raw_record in latest.items():
        case = case_lookup[key[0]]
        record = dict(raw_record)
        record.update(
            {
                "category": case.category,
                "case_weight": case.weight,
                "tags": list(case.tags),
                "acceptable_routes": list(
                    route_contract.acceptable_routes(case.case_id)
                    if route_contract is not None
                    else case.acceptable_routes
                ),
            }
        )
        response = record.get("response")
        if record.get("error") is None and isinstance(response, str):
            try:
                if evaluator is None:
                    rescored = score_response(response, case.scorer)
                else:
                    rescored = evaluator.validate_evidence(
                        case,
                        response,
                        record.get("evaluation_evidence"),
                        sample_id=(
                            f"{run_manifest.get('run_id')}:{case.case_id}:{key[1]}"
                        ),
                    ).result
            except Exception as exc:
                record.update(
                    {
                        "score": None,
                        "passed": False,
                        "error": f"stored evaluation evidence failed validation: {exc}",
                    }
                )
                relevant.append(record)
                continue
            raw_score = record.get("score")
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isclose(float(raw_score), rescored.score, abs_tol=1e-12)
            ):
                record.update(
                    {
                        "score": None,
                        "passed": False,
                        "error": "stored score failed deterministic revalidation",
                    }
                )
            else:
                record.update(rescored.to_dict())
        elif record.get("error") is None:
            record.update(
                {
                    "score": None,
                    "passed": False,
                    "error": "successful result has no response text",
                }
            )
        relevant.append(record)
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in relevant:
        by_category[str(record.get("category"))].append(record)
    categories: dict[str, dict[str, object]] = {}
    category_scores: dict[str, float] = {}
    for category in suite.category_weights:
        category_records = by_category.get(category, [])
        if not category_records:
            continue
        score = _weighted_mean(category_records)
        successful = [record for record in category_records if record.get("score") is not None]
        if score is not None:
            category_scores[category] = score
        categories[category] = {
            "label": suite.category_labels[category],
            "suite_weight": suite.category_weights[category],
            "attempted": len(category_records),
            "scored": len(successful),
            "errors": sum(record.get("error") is not None for record in category_records),
            "passed": sum(record.get("passed") is True for record in successful),
            "mean_score": score,
            "score_percent": None if score is None else score * 100,
        }
    overall = _overall_score(category_scores, suite.category_weights)
    successful_records = [record for record in relevant if record.get("score") is not None]
    telemetry = [
        record.get("telemetry")
        for record in successful_records
        if isinstance(record.get("telemetry"), Mapping)
    ]
    elapsed = [
        float(item["elapsed_seconds"])
        for item in telemetry
        if isinstance(item.get("elapsed_seconds"), (int, float))
    ]
    ttft = [
        float(item["time_to_first_token_seconds"])
        for item in telemetry
        if isinstance(item.get("time_to_first_token_seconds"), (int, float))
    ]
    throughput = [
        float(item["tokens_per_second"])
        for item in telemetry
        if isinstance(item.get("tokens_per_second"), (int, float))
    ]
    completion_tokens = [
        int(item["completion_tokens"])
        for item in telemetry
        if isinstance(item.get("completion_tokens"), int)
    ]
    routing_total = 0
    routing_acceptable = 0
    route_utilization: dict[str, int] = defaultdict(int)
    for record in successful_records:
        decision = record.get("routing_decision")
        acceptable = record.get("acceptable_routes")
        if not isinstance(decision, Mapping):
            continue
        selected = decision.get("selected_routes")
        if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)) or not selected:
            continue
        top1 = selected[0]
        canonical_top1 = (
            route_contract.canonicalize_selected(top1)
            if route_contract is not None
            else (BASE_ROUTE if top1 == "base" else str(top1))
        )
        route_utilization[canonical_top1] += 1
        if (
            isinstance(acceptable, Sequence)
            and not isinstance(acceptable, (str, bytes))
            and acceptable
        ):
            routing_total += 1
            routing_acceptable += canonical_top1 in acceptable
    digest = hashlib.sha256(
        f"{suite.fingerprint}:{run_manifest.get('run_id')}".encode("utf-8")
    ).digest()
    ci = _bootstrap_quality_ci(
        successful_records,
        suite,
        seed=int.from_bytes(digest[:8], "big"),
    )
    expected_results = len(expected_cells)
    errors = sum(record.get("error") is not None for record in relevant)
    successful_keys = {
        (str(record["case_id"]), int(record["repetition"])) for record in successful_records
    }
    comparable = successful_keys == set(expected_cells) and errors == 0
    if not comparable:
        ci = None
    result_cells = [
        {
            "case_id": str(record["case_id"]),
            "repetition": int(record["repetition"]),
            "cell_id": record.get("cell_id"),
            "category": record.get("category"),
            "case_weight": record.get("case_weight"),
            "score": record.get("score"),
            "passed": record.get("passed"),
            "response_sha256": (
                hashlib.sha256(str(record["response"]).encode("utf-8")).hexdigest()
                if isinstance(record.get("response"), str)
                else None
            ),
            "evaluation_evidence_fingerprint": (
                record.get("evaluation_evidence", {}).get("evidence_fingerprint")
                if isinstance(record.get("evaluation_evidence"), Mapping)
                else None
            ),
        }
        for record in sorted(
            relevant,
            key=lambda item: (str(item.get("case_id")), int(item.get("repetition", 0))),
        )
    ]
    result_fingerprint = hashlib.sha256(canonical_json(result_cells).encode("utf-8")).hexdigest()
    return {
        "schema_version": SUMMARY_SCHEMA,
        "created_at": _now(),
        "run_id": run_manifest.get("run_id"),
        "label": run_manifest.get("label"),
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "suite_fingerprint": suite.fingerprint,
        "scorer_version": run_manifest.get("scorer_version"),
        "evaluation_fingerprint": run_manifest.get("evaluation_fingerprint"),
        "workload_fingerprint": workload_fingerprint,
        "workload_contract": dict(workload_contract),
        "profile": run_manifest.get("profile"),
        "case_ids": list(workload_contract.get("case_ids", [])),
        "repetitions": workload_contract.get("repetitions"),
        "backend": run_manifest.get("backend"),
        "stack_fingerprint": run_manifest.get("stack_fingerprint"),
        "stack_fully_content_addressed": run_manifest.get("stack_fully_content_addressed", False),
        "quality_index": overall * 1000,
        "quality_percent": overall * 100,
        "quality_index_ci": ci,
        "comparable": comparable,
        "result_cells": result_cells,
        "result_fingerprint": result_fingerprint,
        "categories": categories,
        "coverage": {
            "expected": expected_results,
            "recorded": len(relevant),
            "scored": len(successful_records),
            "errors": errors,
            "fraction": len(successful_records) / expected_results if expected_results else 0.0,
        },
        "performance": {
            "elapsed_seconds_total": sum(elapsed),
            "latency_seconds_median": _percentile(elapsed, 0.5),
            "latency_seconds_p95": _percentile(elapsed, 0.95),
            "ttft_seconds_median": _percentile(ttft, 0.5),
            "ttft_seconds_p95": _percentile(ttft, 0.95),
            "tokens_per_second_median": _percentile(throughput, 0.5),
            "tokens_per_second_p10": _percentile(throughput, 0.1),
            "completion_tokens_total": sum(completion_tokens),
            "samples_with_ttft": len(ttft),
            "samples_with_throughput": len(throughput),
        },
        "routing": {
            "evaluated": routing_total,
            "acceptable_top1": routing_acceptable,
            "acceptable_top1_rate": (routing_acceptable / routing_total if routing_total else None),
            "top1_utilization": dict(sorted(route_utilization.items())),
            "evaluation": routing_evaluation,
        },
        "artifacts": {
            "run_manifest": "run.json",
            "stack_lock": "stack.lock.json",
            "suite_snapshot": "suite.snapshot.json",
            "raw_results": "results.jsonl",
            "category_scores": "scores.csv",
            "report": "report.md",
        },
    }


def _summary_and_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = path
    if path.is_dir():
        summary_path = path / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError(f"{summary_path} is not a {SUMMARY_SCHEMA} summary")
    results = _read_jsonl(summary_path.parent / "results.jsonl")
    return summary, results


def _validated_result_map(
    summary: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    workload = summary.get("workload_contract")
    workload_fingerprint = summary.get("workload_fingerprint")
    result_cells = summary.get("result_cells")
    result_fingerprint = summary.get("result_fingerprint")
    if (
        not isinstance(workload, Mapping)
        or not isinstance(workload_fingerprint, str)
        or not workload_fingerprint
        or not isinstance(result_cells, Sequence)
        or isinstance(result_cells, (str, bytes))
        or not isinstance(result_fingerprint, str)
    ):
        raise ValueError("benchmark summary is missing strict workload/result provenance")
    if _fingerprint_workload(workload) != workload_fingerprint:
        raise ValueError("benchmark summary workload fingerprint is invalid")
    if hashlib.sha256(canonical_json(list(result_cells)).encode("utf-8")).hexdigest() != (
        result_fingerprint
    ):
        raise ValueError("benchmark summary result fingerprint is invalid")
    expected = _contract_cells(workload)
    summarized: dict[tuple[str, int], dict[str, object]] = {}
    for raw in result_cells:
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark summary contains a malformed result cell")
        case_id = raw.get("case_id")
        repetition = raw.get("repetition")
        if not isinstance(case_id, str) or not isinstance(repetition, int):
            raise ValueError("benchmark summary contains a malformed result cell")
        key = case_id, repetition
        contract_cell = expected.get(key)
        score = raw.get("score")
        if (
            contract_cell is None
            or key in summarized
            or raw.get("category") != contract_cell.get("category")
            or raw.get("case_weight") != contract_cell.get("case_weight")
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise ValueError(f"benchmark summary has an invalid result cell {key}")
        summarized[key] = dict(raw)
    if set(summarized) != set(expected):
        raise ValueError("benchmark summary does not contain every expected workload cell")

    validated: dict[tuple[str, int], dict[str, object]] = {}
    for raw in records:
        case_id = raw.get("case_id")
        repetition = raw.get("repetition")
        if not isinstance(case_id, str) or not isinstance(repetition, int):
            continue
        key = case_id, repetition
        contract_cell = expected.get(key)
        summarized_cell = summarized.get(key)
        score = raw.get("score")
        response = raw.get("response")
        if (
            contract_cell is None
            or summarized_cell is None
            or not _record_matches_cell(raw, summary, contract_cell)
            or raw.get("error") is not None
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isclose(float(score), float(summarized_cell["score"]), abs_tol=1e-12)
            or raw.get("cell_id") != summarized_cell.get("cell_id")
            or not isinstance(response, str)
            or hashlib.sha256(response.encode("utf-8")).hexdigest()
            != summarized_cell.get("response_sha256")
        ):
            continue
        validated[key] = dict(summarized_cell)
    if set(validated) != set(expected):
        raise ValueError("raw benchmark results do not prove every summarized workload cell")
    return validated


def _weighted_pairs_mean(values: Sequence[tuple[float, float]]) -> float:
    weight = sum(item[1] for item in values)
    return sum(item[0] * item[1] for item in values) / weight if weight else 0.0


def compare_runs(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline, baseline_results = _summary_and_results(baseline_path)
    candidate, candidate_results = _summary_and_results(candidate_path)
    suite_fingerprint = baseline.get("suite_fingerprint")
    evaluator_fingerprint = baseline.get("evaluation_fingerprint")
    workload_fingerprint = baseline.get("workload_fingerprint")
    if not isinstance(suite_fingerprint, str) or not suite_fingerprint:
        raise ValueError("baseline has no suite fingerprint")
    if not isinstance(evaluator_fingerprint, str) or not evaluator_fingerprint:
        raise ValueError("baseline has no evaluator fingerprint")
    if not isinstance(workload_fingerprint, str) or not workload_fingerprint:
        raise ValueError("baseline has no workload fingerprint")
    if suite_fingerprint != candidate.get("suite_fingerprint"):
        raise ValueError("benchmark comparisons require identical suite fingerprints")
    if evaluator_fingerprint != candidate.get("evaluation_fingerprint"):
        raise ValueError("benchmark comparisons require identical evaluator fingerprints")
    if workload_fingerprint != candidate.get("workload_fingerprint"):
        raise ValueError(
            "benchmark comparisons require the same profile, cases, repetitions, and generation "
            "settings"
        )
    if baseline.get("comparable") is not True or candidate.get("comparable") is not True:
        raise ValueError("cannot compare an incomplete benchmark run")
    baseline_map = _validated_result_map(baseline, baseline_results)
    candidate_map = _validated_result_map(candidate, candidate_results)
    if set(baseline_map) != set(candidate_map):
        raise ValueError("benchmark result cells differ despite a shared workload fingerprint")
    common = sorted(baseline_map)
    deltas = [
        float(candidate_map[item]["score"]) - float(baseline_map[item]["score"]) for item in common
    ]
    by_category: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item, delta in zip(common, deltas, strict=True):
        cell = candidate_map[item]
        by_category[str(cell["category"])].append((delta, float(cell["case_weight"])))
    category_weights: dict[str, float] = {}
    baseline_categories = baseline.get("categories")
    candidate_categories = candidate.get("categories")
    if not isinstance(baseline_categories, Mapping) or not isinstance(
        candidate_categories, Mapping
    ):
        raise ValueError("benchmark summaries have no category weights")
    for category in by_category:
        base_item = baseline_categories.get(category)
        candidate_item = candidate_categories.get(category)
        if not isinstance(base_item, Mapping) or not isinstance(candidate_item, Mapping):
            raise ValueError(f"benchmark summary is missing category {category}")
        base_weight = base_item.get("suite_weight")
        candidate_weight = candidate_item.get("suite_weight")
        if (
            isinstance(base_weight, bool)
            or not isinstance(base_weight, (int, float))
            or base_weight != candidate_weight
        ):
            raise ValueError(f"benchmark category weight differs for {category}")
        category_weights[category] = float(base_weight)
    rng = random.Random(8419)
    bootstrap: list[float] = []
    if deltas:
        for _ in range(5000):
            sampled_categories: dict[str, float] = {}
            for category, values in by_category.items():
                sample = [rng.choice(values) for _ in values]
                sampled_categories[category] = _weighted_pairs_mean(sample)
            bootstrap.append(_overall_score(sampled_categories, category_weights) * 1000)

    def validated_quality_index(
        records: Mapping[tuple[str, int], Mapping[str, object]],
    ) -> float:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for record in records.values():
            grouped[str(record["category"])].append(record)
        scores = {
            category: mean
            for category, values in grouped.items()
            if (mean := _weighted_mean(values)) is not None
        }
        return _overall_score(scores, category_weights) * 1000

    baseline_index = validated_quality_index(baseline_map)
    candidate_index = validated_quality_index(candidate_map)
    for name, summary, recomputed in (
        ("baseline", baseline, baseline_index),
        ("candidate", candidate, candidate_index),
    ):
        stored = summary.get("quality_index")
        if (
            isinstance(stored, bool)
            or not isinstance(stored, (int, float))
            or not math.isclose(float(stored), recomputed, abs_tol=1e-9)
        ):
            raise ValueError(f"{name} summary quality index fails result-cell revalidation")
    index_delta = candidate_index - baseline_index
    return {
        "schema_version": COMPARISON_SCHEMA,
        "created_at": _now(),
        "suite_id": candidate.get("suite_id"),
        "suite_version": candidate.get("suite_version"),
        "suite_fingerprint": candidate.get("suite_fingerprint"),
        "evaluation_fingerprint": candidate.get("evaluation_fingerprint"),
        "workload_fingerprint": candidate.get("workload_fingerprint"),
        "baseline": {
            "run_id": baseline.get("run_id"),
            "label": baseline.get("label"),
            "quality_index": baseline_index,
            "path": str(baseline_path),
        },
        "candidate": {
            "run_id": candidate.get("run_id"),
            "label": candidate.get("label"),
            "quality_index": candidate_index,
            "path": str(candidate_path),
        },
        "quality_index_delta": index_delta,
        "quality_relative_delta_percent": (
            index_delta / baseline_index * 100 if baseline_index != 0 else None
        ),
        "paired": {
            "count": len(common),
            "wins": sum(delta > 1e-12 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
            "losses": sum(delta < -1e-12 for delta in deltas),
            "mean_case_delta": statistics.fmean(deltas) if deltas else None,
            "mean_case_index_delta": statistics.fmean(deltas) * 1000 if deltas else None,
            "quality_index_delta_ci95": (
                {
                    "lower": _percentile(bootstrap, 0.025),
                    "upper": _percentile(bootstrap, 0.975),
                    "method": "paired-stratified-category-weighted-bootstrap",
                }
                if bootstrap
                else None
            ),
        },
        "category_deltas": {
            category: {
                "matched": len(values),
                "mean_score_delta": _weighted_pairs_mean(values),
                "percentage_point_delta": _weighted_pairs_mean(values) * 100,
            }
            for category, values in sorted(by_category.items())
        },
        "regression_flags": [
            category
            for category, values in sorted(by_category.items())
            if _weighted_pairs_mean(values) <= -0.05
        ],
    }


def _write_scores_csv(path: Path, summary: Mapping[str, object]) -> None:
    categories = summary.get("categories")
    categories = categories if isinstance(categories, Mapping) else {}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["category", "label", "score_percent", "passed", "scored", "errors", "weight"]
        )
        for name, raw in categories.items():
            item = raw if isinstance(raw, Mapping) else {}
            writer.writerow(
                [
                    name,
                    item.get("label"),
                    item.get("score_percent"),
                    item.get("passed"),
                    item.get("scored"),
                    item.get("errors"),
                    item.get("suite_weight"),
                ]
            )


def _format(value: object, decimals: int = 1) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def render_markdown_report(
    summary: Mapping[str, object],
    *,
    comparison: Mapping[str, object] | None = None,
) -> str:
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), Mapping) else {}
    performance = (
        summary.get("performance") if isinstance(summary.get("performance"), Mapping) else {}
    )
    routing = summary.get("routing") if isinstance(summary.get("routing"), Mapping) else {}
    lines = [
        f"# {summary.get('label')} — benchmark report",
        "",
        f"**MoLoRA Quality Index: {_format(summary.get('quality_index'))} / 1000**",
        (
            ""
            if summary.get("comparable") is True
            else "\n**INCOMPLETE — do not record or compare this score. Resume failed cells.**"
        ),
        "",
        f"Suite: `{summary.get('suite_id')}` v{summary.get('suite_version')}  ",
        f"Evaluator: `{summary.get('scorer_version')}`  ",
        f"Evaluation fingerprint: `{summary.get('evaluation_fingerprint')}`  ",
        f"Workload fingerprint: `{summary.get('workload_fingerprint')}`  ",
        f"Stack fingerprint: `{summary.get('stack_fingerprint')}`  ",
        f"Run: `{summary.get('run_id')}`  ",
        f"Coverage: {coverage.get('scored', 0)}/{coverage.get('expected', 0)} scored, "
        f"{coverage.get('errors', 0)} errors",
        "",
        "## Category scores",
        "",
        "| Category | Score | Passed | Scored | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    categories = summary.get("categories")
    if isinstance(categories, Mapping):
        for raw in categories.values():
            item = raw if isinstance(raw, Mapping) else {}
            lines.append(
                f"| {item.get('label')} | {_format(item.get('score_percent'))}% | "
                f"{item.get('passed', 0)} | {item.get('scored', 0)} | {item.get('errors', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Median throughput: {_format(performance.get('tokens_per_second_median'))} tok/s",
            f"- Median TTFT: {_format(performance.get('ttft_seconds_median'), 3)} s",
            f"- P95 TTFT: {_format(performance.get('ttft_seconds_p95'), 3)} s",
            f"- Median latency: {_format(performance.get('latency_seconds_median'), 3)} s",
            f"- Total measured generation time: {_format(performance.get('elapsed_seconds_total'), 2)} s",
            "",
            "## Routing",
            "",
            f"Acceptable top-1 route: {routing.get('acceptable_top1', 0)}/"
            f"{routing.get('evaluated', 0)} "
            f"({_format((routing.get('acceptable_top1_rate') or 0) * 100)}%)",
        ]
    )
    if comparison is not None:
        paired = comparison.get("paired") if isinstance(comparison.get("paired"), Mapping) else {}
        interval = (
            paired.get("quality_index_delta_ci95")
            if isinstance(paired.get("quality_index_delta_ci95"), Mapping)
            else {}
        )
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                f"Index delta: **{float(comparison.get('quality_index_delta', 0)):+.1f}**  ",
                f"Paired cases: {paired.get('wins', 0)} wins / {paired.get('ties', 0)} ties / "
                f"{paired.get('losses', 0)} losses  ",
                f"95% paired interval: {_format(interval.get('lower'))} to "
                f"{_format(interval.get('upper'))} index points  ",
                "Regression flags: "
                + (", ".join(comparison.get("regression_flags", [])) or "none"),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The Quality Index is a 0–1000 weighted score defined by this exact suite and evaluator "
            "version. Compare it only when the evaluation fingerprint matches. Runtime measurements "
            "are reported separately so hardware changes do not look like model-quality changes.",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(
    summary: Mapping[str, object],
    *,
    comparison: Mapping[str, object] | None = None,
) -> None:
    print("\n" + "=" * 68, flush=True)
    print(
        f"MOLOR QUALITY INDEX  {_format(summary.get('quality_index'))} / 1000",
        flush=True,
    )
    print("=" * 68, flush=True)
    categories = summary.get("categories")
    if isinstance(categories, Mapping):
        for raw in categories.values():
            item = raw if isinstance(raw, Mapping) else {}
            print(
                f"{str(item.get('label')):<34} {_format(item.get('score_percent')):>6}%  "
                f"({item.get('passed', 0)}/{item.get('scored', 0)})",
                flush=True,
            )
    performance = summary.get("performance")
    if isinstance(performance, Mapping):
        print("-" * 68, flush=True)
        print(
            f"Throughput median: {_format(performance.get('tokens_per_second_median'))} tok/s | "
            f"TTFT median: {_format(performance.get('ttft_seconds_median'), 3)} s",
            flush=True,
        )
    if comparison is not None:
        paired = comparison.get("paired") if isinstance(comparison.get("paired"), Mapping) else {}
        print(
            f"Baseline delta: {float(comparison.get('quality_index_delta', 0)):+.1f} | "
            f"{paired.get('wins', 0)}W/{paired.get('ties', 0)}T/{paired.get('losses', 0)}L",
            flush=True,
        )
    print(f"Report: {summary.get('run_id')}/report.md", flush=True)


__all__ = [
    "BenchmarkRunner",
    "COMPARISON_SCHEMA",
    "RunOptions",
    "compare_runs",
    "print_summary",
    "render_markdown_report",
    "summarize_run",
]
