"""Command-line entry point for a small local expert run."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from molor.training.recipe import load_training_recipe
from molor.training.sft import TrainingProgress, train_experts


def _print_progress(update: TrainingProgress) -> None:
    print(
        f"[{update.expert}] epoch {update.epoch}/{update.epochs} "
        f"step {update.optimizer_step}/{update.total_optimizer_steps} "
        f"loss {update.loss:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train independent LoRA experts sequentially on one frozen causal LM."
    )
    parser.add_argument("recipe", type=Path, help="path to a training recipe JSON file")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="refuse model/tokenizer network access",
    )
    parser.add_argument(
        "--expert",
        action="append",
        help="train only this named expert (repeat to select several)",
    )
    args = parser.parse_args()
    recipe = load_training_recipe(args.recipe)
    if args.expert:
        selected = set(args.expert)
        unknown = selected.difference(expert.name for expert in recipe.experts)
        if unknown:
            parser.error(f"unknown expert(s): {', '.join(sorted(unknown))}")
        recipe = replace(
            recipe,
            experts=tuple(expert for expert in recipe.experts if expert.name in selected),
        )

    try:
        from molor.experimental import MixtureEngine
    except ImportError as exc:  # pragma: no cover - editable install can be stale
        raise SystemExit("molor.experimental is unavailable; reinstall the workspace") from exc

    dtype = getattr(torch, recipe.dtype)
    device_map: str | dict[str, str]
    if recipe.device == "auto":
        device_map = "auto"
    else:
        device_map = {"": recipe.device}
    print(f"Loading frozen base: {recipe.model_id}", flush=True)
    engine = MixtureEngine.from_pretrained(
        recipe.model_id,
        dtype=dtype,
        device_map=device_map,
        target_modules=recipe.target_modules,
        revision=recipe.revision,
        local_files_only=args.local_files_only,
    )
    losses = train_experts(engine, recipe, progress=_print_progress)
    print("Saved adapters:", flush=True)
    for expert in recipe.experts:
        print(
            f"  {expert.name}: {recipe.output_dir / expert.name} (loss {losses[expert.name]:.4f})"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
