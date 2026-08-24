"""Pinned lm-evaluation-harness launcher for published benchmark tasks."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from molor.benchmark.lock import lock_backend_identity
from molor.benchmark.models import fingerprint


LM_EVAL_VERSION = "0.4.12"
STANDARD_SUMMARY_SCHEMA = "molor.standard-benchmark-summary.v2"
LM_EVAL_SEEDS = {
    "python_random": 0,
    "numpy": 1234,
    "torch": 1234,
    "fewshot": 1234,
}
METRIC_SELECTION_VERSION = "molor.standard-primary-metric.v1"
PRIMARY_METRICS = (
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
    "exact_match,flexible-extract",
)
FAST_TASKS = (
    "arc_challenge",
    "gsm8k",
    "ifeval",
    "mmlu_electrical_engineering",
    "mmlu_college_computer_science",
    "mmlu_management",
    "mmlu_marketing",
)
NIGHTLY_TASKS = ("arc_challenge", "hellaswag", "mmlu", "gsm8k", "ifeval")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _revision_provenance(model_id: str, model_lock: Mapping[str, object]) -> dict[str, object]:
    if model_lock.get("fully_content_addressed") is True:
        return {
            "requested_revision": None,
            "resolved_revision": None,
            "status": "local-content-addressed",
            "reason": "the local checkpoint and tokenizer are locked by per-file SHA-256",
        }
    return {
        "requested_revision": None,
        "resolved_revision": None,
        "status": "unresolved",
        "reason": (
            f"{model_id!r} is not a complete local checkpoint and no immutable Hugging Face "
            "revision was supplied by this CLI"
        ),
    }


def _metric(result: Mapping[str, object]) -> tuple[str | None, float | None]:
    for name in PRIMARY_METRICS:
        value = result.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value)
    for name, value in result.items():
        if "stderr" not in name and isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value)
    return None, None


def _find_results(output: Path) -> Path:
    if output.is_file():
        return output
    candidates = sorted(
        output.rglob("*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and isinstance(value.get("results"), Mapping):
            return candidate
    raise FileNotFoundError(f"lm-eval did not produce a results JSON under {output}")


def run_lm_eval(
    *,
    model: str,
    tasks: Sequence[str],
    output_root: Path,
    label: str,
    batch_size: str,
    device: str,
    offline: bool,
    limit: int | None,
) -> Path:
    try:
        installed = importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "lm-evaluation-harness is not installed in this environment. Install the pinned "
            f'lane with: python -m pip install "lm_eval[hf,ifeval]=={LM_EVAL_VERSION}"'
        ) from exc
    if installed != LM_EVAL_VERSION:
        raise RuntimeError(
            f"lm-eval {installed} is installed, but this benchmark lane is pinned to "
            f"{LM_EVAL_VERSION}; use an isolated evaluation venv"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_directory = output_root.expanduser().resolve() / f"{timestamp}-{label}"
    raw_directory = run_directory / "lm-eval-raw"
    raw_directory.mkdir(parents=True, exist_ok=False)
    model_path = Path(model).expanduser()
    model_id = str(model_path.resolve()) if model_path.is_dir() else model
    model_lock = lock_backend_identity(
        {"backend": "lm-eval-hf", "model_id": model_id},
        cache_path=run_directory / ".unused-artifact-cache.json",
    )
    _write_json(run_directory / "model.lock.json", model_lock)
    revision = _revision_provenance(model_id, model_lock)
    requested_tasks = [str(task) for task in tasks]
    seed_argument = ",".join(str(value) for value in LM_EVAL_SEEDS.values())
    model_args = (
        f"pretrained={model_id},dtype=bfloat16,trust_remote_code=False,enable_thinking=False"
    )
    if offline:
        model_args += ",local_files_only=True"
    execution_settings = {
        "model_backend": "hf",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "enable_thinking": False,
        "apply_chat_template": True,
        "generation": {"do_sample": False},
        "batch_size": batch_size,
        "device": device,
        "offline": offline,
        "limit": limit,
        "check_integrity": True,
        "metric_selection": {
            "version": METRIC_SELECTION_VERSION,
            "priority": list(PRIMARY_METRICS),
            "fallback": "first-numeric-non-stderr-metric",
        },
    }
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--device",
        device,
        "--batch_size",
        batch_size,
        "--apply_chat_template",
        "--gen_kwargs",
        "do_sample=False",
        "--seed",
        seed_argument,
        "--check_integrity",
        "--show_config",
        "--log_samples",
        "--output_path",
        str(raw_directory),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    environment = os.environ.copy()
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["HF_DATASETS_OFFLINE"] = "1"
    model_provenance_complete = model_lock.get("fully_content_addressed") is True
    score_recordable = limit is None and model_provenance_complete
    _write_json(
        run_directory / "command.json",
        {
            "schema_version": "molor.standard-benchmark-command.v2",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "command": command,
            "harness": {
                "package": "lm_eval",
                "version": installed,
                "python_executable": sys.executable,
                "dependencies": _package_versions(
                    ("torch", "transformers", "datasets", "accelerate", "huggingface_hub")
                ),
            },
            "tasks": requested_tasks,
            "model": model_id,
            "tokenizer": model_id,
            "model_revision": revision,
            "model_lock_file": "model.lock.json",
            "model_stack_fingerprint": model_lock["stack_fingerprint"],
            "model_fully_content_addressed": model_lock["fully_content_addressed"],
            "seeds": LM_EVAL_SEEDS,
            "execution_settings": execution_settings,
            "batch_size": batch_size,
            "device": device,
            "offline": offline,
            "limit": limit,
            "score_recordable": score_recordable,
            "strictly_reproducible": False,
            "comparable": False,
        },
    )
    print("Launching pinned lm-evaluation-harness:", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"lm-eval exited with code {completed.returncode}")
    result_path = _find_results(raw_directory)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    results = raw.get("results") if isinstance(raw, Mapping) else None
    if not isinstance(results, Mapping):
        raise ValueError("lm-eval results JSON has no results object")
    task_scores: dict[str, dict[str, object]] = {}
    for task, raw_result in results.items():
        if not isinstance(raw_result, Mapping):
            continue
        metric, value = _metric(raw_result)
        task_scores[str(task)] = {
            "primary_metric": metric,
            "primary_score": value,
            "all_metrics": dict(raw_result),
        }
    raw_groups = raw.get("groups") if isinstance(raw, Mapping) else None
    groups = raw_groups if isinstance(raw_groups, Mapping) else {}
    benchmark_components: dict[str, dict[str, object]] = {}
    component_values: list[float] = []
    for task in requested_tasks:
        raw_component = groups.get(task)
        source = "group"
        if not isinstance(raw_component, Mapping):
            raw_component = results.get(task)
            source = "task"
        if not isinstance(raw_component, Mapping):
            raise ValueError(f"lm-eval returned no task or group aggregate for {task!r}")
        metric, value = _metric(raw_component)
        if value is None:
            raise ValueError(f"lm-eval returned no numeric primary metric for {task!r}")
        benchmark_components[task] = {
            "source": source,
            "primary_metric": metric,
            "primary_score": value,
            "all_metrics": dict(raw_component),
        }
        component_values.append(value)
    index = sum(component_values) / len(component_values) * 1000
    raw_config = raw.get("config") if isinstance(raw, Mapping) else None
    raw_task_versions = raw.get("versions") if isinstance(raw, Mapping) else None
    raw_task_configs = raw.get("configs") if isinstance(raw, Mapping) else None
    dependencies = _package_versions(
        ("torch", "transformers", "datasets", "accelerate", "huggingface_hub")
    )
    dataset_provenance = {
        "status": "unresolved",
        "reason": (
            "lm-eval task metadata does not lock the immutable Hugging Face dataset revisions "
            "used to populate the local cache"
        ),
    }
    evaluation_contract = {
        "lm_eval_version": installed,
        "dependencies": dependencies,
        "requested_tasks": requested_tasks,
        "task_versions": dict(raw_task_versions)
        if isinstance(raw_task_versions, Mapping)
        else None,
        "task_configs": dict(raw_task_configs) if isinstance(raw_task_configs, Mapping) else None,
        "seeds": LM_EVAL_SEEDS,
        "execution_settings": execution_settings,
        "dataset_provenance": dataset_provenance,
    }
    standard_evaluation_fingerprint = fingerprint(evaluation_contract)
    conventional_comparison_ready = score_recordable and len(benchmark_components) == len(
        requested_tasks
    )
    summary = {
        "schema_version": STANDARD_SUMMARY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "label": label,
        "model": model_id,
        "tokenizer": model_id,
        "model_revision": revision,
        "model_lock_file": str(run_directory / "model.lock.json"),
        "model_stack_fingerprint": model_lock["stack_fingerprint"],
        "model_fully_content_addressed": model_lock["fully_content_addressed"],
        "harness": {
            "package": "lm_eval",
            "version": installed,
            "dependencies": dependencies,
            "run_config": dict(raw_config) if isinstance(raw_config, Mapping) else None,
        },
        "lm_eval_version": installed,
        "requested_tasks": requested_tasks,
        "benchmark_components": benchmark_components,
        "tasks": task_scores,
        "task_versions": dict(raw_task_versions)
        if isinstance(raw_task_versions, Mapping)
        else None,
        "task_configs": dict(raw_task_configs) if isinstance(raw_task_configs, Mapping) else None,
        "seeds": LM_EVAL_SEEDS,
        "execution_settings": execution_settings,
        "dataset_provenance": dataset_provenance,
        "evaluation_contract": evaluation_contract,
        "evaluation_fingerprint": standard_evaluation_fingerprint,
        "task_macro_index": index,
        "score_recordable": score_recordable,
        "conventional_comparison_ready": conventional_comparison_ready,
        "strictly_reproducible": False,
        "comparable": False,
        "warning": (
            "--limit is a smoke test only and must not be recorded as a benchmark score"
            if limit is not None
            else (
                "model revision is unresolved; use a complete local checkpoint"
                if not model_provenance_complete
                else "dataset revisions are recorded only through lm-eval task metadata and are "
                "not content-locked; this is a conventional benchmark comparison, not a "
                "byte-exact regression claim"
            )
        ),
        "raw_results": str(result_path),
    }
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {label} — standard benchmark report",
        "",
        f"lm-evaluation-harness: `{installed}`",
        "",
        f"Requested-component Macro Index: **{index:.1f} / 1000**",
        "",
        "| Task | Metric | Score |",
        "|---|---|---:|",
    ]
    for task, task_result in benchmark_components.items():
        value = task_result["primary_score"]
        rendered = "n/a" if value is None else f"{float(value) * 100:.2f}%"
        lines.append(f"| {task} | {task_result['primary_metric']} | {rendered} |")
    if limit is not None:
        lines.extend(["", "**Smoke-only run: `--limit` scores are not comparable.**"])
    elif not model_provenance_complete:
        lines.extend(
            [
                "",
                "**The remote model revision is unresolved; this score is not marked comparable.**",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "**Dataset revisions are not content-locked. Treat this as a conventional public "
                "benchmark comparison, not a byte-exact regression result.**",
            ]
        )
    (run_directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Standard benchmark summary: {summary_path}", flush=True)
    return summary_path


__all__ = ["FAST_TASKS", "LM_EVAL_VERSION", "NIGHTLY_TASKS", "run_lm_eval"]
