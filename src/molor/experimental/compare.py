"""Run one prompt through a base, two LoRA experts, and their blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from molor.experimental.mixture import (
    BASE_ROUTE,
    GenerationSettings,
    GenerationStats,
    MixtureEngine,
)
from molor.model_profiles import (
    corpus_expert_root,
    default_model,
    runtime_artifact_defaults,
)


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = default_model(_WORKSPACE_ROOT)
DEFAULT_PROMPT = (
    "Explain how embedded C firmware should decode a signed CAN signal safely, including "
    "bit layout, scaling, range checks, and failure handling."
)


def _adapter_config(directory: Path) -> dict[str, Any]:
    config_path = directory / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read adapter config {config_path}: {error}") from error
    targets = payload.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        raise SystemExit(f"Adapter config {config_path} has invalid target_modules")
    return payload


def _complete_adapter(directory: Path) -> bool:
    try:
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in (
                directory / "adapter_config.json",
                directory / "adapter_model.safetensors",
            )
        )
    except OSError:
        return False


def _shared_targets(expert_a: Path, expert_b: Path) -> tuple[str, ...]:
    targets_a = tuple(_adapter_config(expert_a)["target_modules"])
    targets_b = tuple(_adapter_config(expert_b)["target_modules"])
    if targets_a != targets_b:
        raise SystemExit(
            "Adapters target different projections: "
            f"{expert_a} uses {targets_a}, while {expert_b} uses {targets_b}"
        )
    return targets_a


def _local_or_hub_model(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def _print_stats(stats: GenerationStats) -> None:
    ttft = (
        "n/a"
        if stats.time_to_first_token_seconds is None
        else f"{stats.time_to_first_token_seconds:.2f}s"
    )
    print(
        f"\n[{stats.completion_tokens} tokens | {stats.tokens_per_second:.1f} tok/s | "
        f"TTFT {ttft} | {stats.elapsed_seconds:.2f}s | {stats.finish_reason}]",
        flush=True,
    )


def _run_route(
    engine: MixtureEngine,
    *,
    label: str,
    prompt: str,
    route: dict[str, float],
    settings: GenerationSettings,
    thinking: bool,
    seed: int,
) -> None:
    print(f"\n=== {label} ===", flush=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    terminal: GenerationStats | None = None
    for update in engine.stream_chat(
        [{"role": "user", "content": prompt}],
        route=route,
        settings=settings,
        enable_thinking=thinking,
    ):
        if update.text:
            print(update.text, end="", flush=True)
        if update.finished:
            terminal = update.stats
    if terminal is not None:
        _print_stats(terminal)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two resident LoRA experts and their 50/50 mixture."
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--expert-a", type=Path, default=None)
    parser.add_argument("--expert-b", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face network access when the model is not local",
    )
    return parser


def _default_experts(model: str) -> tuple[Path | None, Path | None]:
    artifacts = runtime_artifact_defaults(model, _WORKSPACE_ROOT)
    root = corpus_expert_root(model, _WORKSPACE_ROOT)
    expert_a = Path(artifacts.expert_a) if artifacts.expert_a else None
    expert_b = Path(artifacts.expert_b) if artifacts.expert_b else None
    if root is not None:
        expert_a = expert_a or root / "embedded-knowledge-v1"
        expert_b = expert_b or root / "ocb-can-verified-v1"
    return expert_a, expert_b


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)

    default_expert_a, default_expert_b = _default_experts(args.model)
    if args.expert_a is None and default_expert_a is None:
        parser.error("unknown model lane: pass --expert-a")
    if args.expert_b is None and default_expert_b is None:
        parser.error("unknown model lane: pass --expert-b")

    expert_a = (args.expert_a or default_expert_a).resolve()
    expert_b = (args.expert_b or default_expert_b).resolve()
    for option, path in (("--expert-a", expert_a), ("--expert-b", expert_b)):
        if not _complete_adapter(path):
            parser.error(
                f"{option} has no complete compatible adapter at {path}; "
                "train the lane expert or pass an explicit checkpoint"
            )
    targets = _shared_targets(expert_a, expert_b)
    model_id = _local_or_hub_model(args.model)
    print(f"Loading {model_id} in bfloat16; projections: {', '.join(targets)}", flush=True)
    engine = MixtureEngine.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        target_modules=targets,
        local_files_only=not args.allow_download,
    )
    name_a = engine.load_expert(expert_a)
    name_b = engine.load_expert(expert_b)
    print(f"Resident experts: {name_a}, {name_b}", flush=True)

    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    comparisons = (
        ("base", {BASE_ROUTE: 1.0}),
        (name_a, {name_a: 1.0}),
        (name_b, {name_b: 1.0}),
        (f"50% {name_a} / 50% {name_b}", {name_a: 0.5, name_b: 0.5}),
    )
    for label, route in comparisons:
        _run_route(
            engine,
            label=label,
            prompt=args.prompt,
            route=route,
            settings=settings,
            thinking=args.thinking,
            seed=args.seed,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
