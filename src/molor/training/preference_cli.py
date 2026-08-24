"""Offline manual-preference adapter training command."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from molor.experimental import MixtureEngine
from molor.training.preference import (
    PreferenceProgress,
    load_preference_recipe,
    train_preference_expert,
)


def _progress(update: PreferenceProgress) -> None:
    if update.phase == "reference":
        print(f"[reference] pair {update.step}/{update.total_steps}", flush=True)
    else:
        print(
            f"[dpo] epoch {update.epoch}/{update.epochs} "
            f"step {update.step}/{update.total_steps} loss {update.loss:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one LoRA from pairwise manual preferences.")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    recipe = load_preference_recipe(args.recipe)
    device_map: str | dict[str, str] = "auto" if recipe.device == "auto" else {"": recipe.device}
    print(f"Loading frozen base: {recipe.model_id}", flush=True)
    print(f"Preference data: {recipe.dataset}", flush=True)
    engine = MixtureEngine.from_pretrained(
        recipe.model_id,
        dtype=getattr(torch, recipe.dtype),
        device_map=device_map,
        target_modules=recipe.target_modules,
        revision=recipe.revision,
        local_files_only=args.local_files_only,
    )
    loss = train_preference_expert(engine, recipe, progress=_progress)
    destination = (recipe.output_dir / recipe.expert_name).resolve()
    print(f"Saved preference expert: {destination} (loss {loss:.4f})", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
