"""Command line entry point for local and standard model benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from molor.benchmark.backends import (
    LmStudioNativeBackend,
    OpenCodeCliBackend,
    OpenAICompatibleBackend,
    TeacherApiBackend,
)
from molor.benchmark.evaluation import BenchmarkEvaluator
from molor.benchmark.matrix import run_matrix
from molor.benchmark.models import SuiteError, load_suite
from molor.benchmark.runner import BenchmarkRunner, RunOptions, compare_runs
from molor.benchmark.scoring import SCORER_VERSION, validate_score_spec
from molor.benchmark.standard import FAST_TASKS, NIGHTLY_TASKS, run_lm_eval


DEFAULT_SUITE = Path("benchmarks/suites/molor-core-v1.json")
DEFAULT_OUTPUT = Path("runs/benchmarks")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molor-benchmark",
        description="Repeatable model quality, routing, and GPU-runtime benchmarks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate and describe a suite")
    validate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)

    run = commands.add_parser("run", help="run the MoLoRA deterministic suite")
    run.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    run.add_argument("--profile", default="quick")
    run.add_argument("--category", action="append", default=[])
    run.add_argument("--limit", type=int)
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--label", required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--run-directory", type=Path, help="resume this exact run directory")
    run.add_argument("--baseline", type=Path, help="prior summary.json or run directory")
    run.add_argument("--no-warmup", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--max-new-tokens", type=int)
    thinking = run.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="enable_thinking",
        action="store_true",
        help="enable provider reasoning when the backend supports it",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="disable provider reasoning when the backend supports it",
    )
    run.set_defaults(enable_thinking=None)
    run.add_argument(
        "--backend",
        choices=("teacher", "openai", "lmstudio", "opencode"),
        default="teacher",
    )
    run.add_argument("--endpoint", default="http://127.0.0.1:8765")
    run.add_argument("--model", default="")
    run.add_argument(
        "--model-artifact",
        type=Path,
        help="local model artifact to content-hash (for example the served GGUF)",
    )
    run.add_argument("--api-key-env", default="")
    run.add_argument(
        "--verifier-config",
        type=Path,
        help="fixed-command registry required by suites with external executable scorers",
    )
    run.add_argument("--opencode-cli", default="opencode")
    run.add_argument("--opencode-variant", default="max")
    run.add_argument("--opencode-agent", default="exam")
    run.add_argument(
        "--opencode-directory",
        type=Path,
        help="isolated OpenCode project containing the tool-free exam agent config",
    )
    run.add_argument("--route", default="auto")
    run.add_argument("--route-top-k", type=int, default=1)
    run.add_argument("--route-temperature", type=float, default=1.0)
    run.add_argument("--confidence-threshold", type=float, default=0.0)
    run.add_argument("--null-bias", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=600.0)

    compare = commands.add_parser("compare", help="compare two completed suite runs")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path)

    history = commands.add_parser("history", help="print the local benchmark scoreboard")
    history.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)

    matrix = commands.add_parser(
        "matrix", help="run base, every resident expert, auto top-1, and auto top-2"
    )
    matrix.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    matrix.add_argument("--profile", default="quick")
    matrix.add_argument("--category", action="append", default=[])
    matrix.add_argument("--limit", type=int)
    matrix.add_argument("--repetitions", type=int, default=1)
    matrix.add_argument("--label", required=True)
    matrix.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    matrix.add_argument("--matrix-directory", type=Path)
    matrix.add_argument("--endpoint", default="http://127.0.0.1:8765")
    matrix.add_argument("--expert", action="append", default=[])
    matrix.add_argument("--no-top2", action="store_true")
    matrix.add_argument("--route-temperature", type=float, default=1.0)
    matrix.add_argument("--confidence-threshold", type=float, default=0.0)
    matrix.add_argument("--null-bias", type=float, default=0.0)
    matrix.add_argument("--timeout", type=float, default=600.0)

    standard = commands.add_parser(
        "standard", help="run pinned lm-eval tasks against a plain HF checkpoint"
    )
    standard.add_argument("--model", required=True)
    standard.add_argument("--preset", choices=("fast", "nightly"), default="fast")
    standard.add_argument("--tasks", help="comma-separated override")
    standard.add_argument("--label", required=True)
    standard.add_argument("--output-root", type=Path, default=Path("runs/standard-benchmarks"))
    standard.add_argument("--batch-size", default="1")
    standard.add_argument("--device", default="cuda:0")
    standard.add_argument("--offline", action="store_true")
    standard.add_argument(
        "--limit",
        type=int,
        help="smoke testing only; limited results are marked non-comparable",
    )
    return parser


def _validate(path: Path) -> int:
    suite = load_suite(path)
    for case in suite.cases:
        validate_score_spec(case.scorer)
    print(f"{suite.title} ({suite.suite_id} v{suite.version})")
    print(f"Fingerprint: {suite.fingerprint}")
    evaluation_fingerprint = hashlib.sha256(
        f"{suite.fingerprint}:{SCORER_VERSION}".encode("utf-8")
    ).hexdigest()
    print(f"Evaluator: {SCORER_VERSION} ({evaluation_fingerprint})")
    print(f"Cases: {len(suite.cases)}")
    print("Profiles: " + ", ".join(f"{name}={len(ids)}" for name, ids in suite.profiles.items()))
    print(
        "Categories: "
        + ", ".join(
            f"{suite.category_labels[name]}={weight * 100:.1f}%"
            for name, weight in suite.category_weights.items()
        )
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    if args.backend == "teacher":
        backend = TeacherApiBackend(
            args.endpoint,
            route=args.route,
            route_top_k=args.route_top_k,
            route_temperature=args.route_temperature,
            confidence_threshold=args.confidence_threshold,
            null_bias=args.null_bias,
            timeout=args.timeout,
        )
    elif args.backend == "openai":
        if not args.model:
            raise ValueError("--model is required for the OpenAI-compatible backend")
        backend = OpenAICompatibleBackend(
            args.endpoint,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
        )
    elif args.backend == "lmstudio":
        if not args.model:
            raise ValueError("--model is required for the LM Studio backend")
        backend = LmStudioNativeBackend(
            args.endpoint,
            model=args.model,
            model_artifact_path=(str(args.model_artifact.resolve()) if args.model_artifact else ""),
            timeout=args.timeout,
        )
    else:
        if not args.model:
            raise ValueError("--model is required for the OpenCode backend")
        if args.opencode_directory is None:
            raise ValueError(
                "--opencode-directory is required for the isolated OpenCode exam backend"
            )
        backend = OpenCodeCliBackend(
            args.opencode_cli,
            model=args.model,
            variant=args.opencode_variant,
            agent=args.opencode_agent,
            working_directory=str(args.opencode_directory.resolve()),
            timeout=args.timeout,
        )
    options = RunOptions(
        label=args.label,
        profile=args.profile,
        categories=tuple(args.category),
        limit=args.limit,
        repetitions=args.repetitions,
        warmup=not args.no_warmup,
        continue_on_error=not args.fail_fast,
        max_new_tokens_override=args.max_new_tokens,
        enable_thinking_override=args.enable_thinking,
    )
    selected_cases = suite.select(
        profile=options.profile,
        categories=options.categories,
        limit=options.limit,
    )
    required_verifiers = BenchmarkEvaluator.required_external_verifiers(selected_cases)
    evaluator = None
    if required_verifiers or args.verifier_config is not None:
        if args.verifier_config is None:
            raise ValueError(
                "--verifier-config is required for external scorers: "
                + ", ".join(required_verifiers)
            )
        evaluator = BenchmarkEvaluator.from_config(
            args.verifier_config,
            required=required_verifiers,
        )
        for case in selected_cases:
            evaluator.validate_case(case)
    runner = BenchmarkRunner(
        suite,
        backend,
        output_root=args.output_root,
        options=options,
        run_directory=args.run_directory,
        evaluator=evaluator,
    )
    runner.run(baseline_summary=args.baseline)
    return 0


def _history(root: Path) -> int:
    history = root.expanduser().resolve() / "history.jsonl"
    if not history.is_file():
        print(f"No benchmark history at {history}")
        return 0
    rows = []
    for line in history.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    print(f"{'RUN':<38} {'LABEL':<28} {'SCORE':>8}  {'EVAL':<10} {'WORKLOAD':<10} SUITE")
    for row in rows:
        print(
            f"{str(row.get('run_id')):<38} {str(row.get('label'))[:28]:<28} "
            f"{float(row.get('quality_index', 0)):8.1f}  "
            f"{str(row.get('evaluation_fingerprint') or 'legacy')[:10]:<10} "
            f"{str(row.get('workload_fingerprint') or 'legacy')[:10]:<10} "
            f"{row.get('suite_id')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.suite)
        if args.command == "run":
            return _run(args)
        if args.command == "compare":
            comparison = compare_runs(args.baseline.resolve(), args.candidate.resolve())
            rendered = json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
                print(args.output.resolve())
            else:
                print(rendered)
            return 0
        if args.command == "history":
            return _history(args.output_root)
        if args.command == "matrix":
            run_matrix(
                load_suite(args.suite),
                endpoint=args.endpoint,
                label=args.label,
                output_root=args.output_root,
                profile=args.profile,
                categories=args.category,
                limit=args.limit,
                repetitions=args.repetitions,
                experts=args.expert,
                include_top2=not args.no_top2,
                route_temperature=args.route_temperature,
                confidence_threshold=args.confidence_threshold,
                null_bias=args.null_bias,
                timeout=args.timeout,
                matrix_directory=args.matrix_directory,
            )
            return 0
        if args.command == "standard":
            tasks = (
                tuple(item.strip() for item in args.tasks.split(",") if item.strip())
                if args.tasks
                else (FAST_TASKS if args.preset == "fast" else NIGHTLY_TASKS)
            )
            run_lm_eval(
                model=args.model,
                tasks=tasks,
                output_root=args.output_root,
                label=args.label,
                batch_size=args.batch_size,
                device=args.device,
                offline=args.offline,
                limit=args.limit,
            )
            return 0
    except (OSError, RuntimeError, SuiteError, ValueError) as exc:
        print(f"benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
