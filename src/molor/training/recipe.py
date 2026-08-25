"""Readable JSON configuration for a quick sequential expert run."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExpertRecipe:
    name: str
    dataset: Path
    epochs: int | None = None
    initial_adapter: Path | None = None
    start_optimizer_step: int = 0

    def __post_init__(self) -> None:
        if not self.name or any(
            not character.isalnum() and character not in "_-" for character in self.name
        ):
            raise ValueError("expert names may contain letters, digits, underscores, and hyphens")
        if self.epochs is not None and (
            not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs < 1
        ):
            raise ValueError("expert epochs must be a positive integer or null")
        if (
            not isinstance(self.start_optimizer_step, int)
            or isinstance(self.start_optimizer_step, bool)
            or self.start_optimizer_step < 0
        ):
            raise ValueError("start_optimizer_step must be a non-negative integer")
        if self.start_optimizer_step and self.initial_adapter is None:
            raise ValueError("start_optimizer_step requires initial_adapter")


@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    model_id: str
    output_dir: Path
    experts: tuple[ExpertRecipe, ...]
    target_modules: tuple[str, ...]
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    max_length: int = 512
    epochs: int = 8
    learning_rate: float = 2e-4
    lr_scheduler: str = "constant"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    autocast_adapter_compute: bool = False
    batching_strategy: str = "online"
    fused_adamw: bool = False
    checkpoint_every_steps: int | None = None
    max_grad_norm: float = 1.0
    seed: int = 42
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.experts:
            raise ValueError("at least one expert is required")
        names = [expert.name for expert in self.experts]
        if len(names) != len(set(names)):
            raise ValueError("expert names must be unique")
        if not self.target_modules or any(not name for name in self.target_modules):
            raise ValueError("target_modules must contain module suffixes")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler must be constant or cosine")
        if self.batching_strategy not in {
            "online",
            "pretokenized",
            "pretokenized_group_sorted",
        }:
            raise ValueError(
                "batching_strategy must be online, pretokenized, or pretokenized_group_sorted"
            )
        for name, value in (
            ("rank", self.rank),
            ("max_length", self.max_length),
            ("epochs", self.epochs),
            ("micro_batch_size", self.micro_batch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("alpha", self.alpha),
            ("learning_rate", self.learning_rate),
            ("max_grad_norm", self.max_grad_norm),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.warmup_ratio, (int, float)) or isinstance(self.warmup_ratio, bool):
            raise ValueError("warmup_ratio must be numeric")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.checkpoint_every_steps is not None and (
            not isinstance(self.checkpoint_every_steps, int)
            or isinstance(self.checkpoint_every_steps, bool)
            or self.checkpoint_every_steps < 1
        ):
            raise ValueError("checkpoint_every_steps must be a positive integer or null")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if not isinstance(self.autocast_adapter_compute, bool):
            raise ValueError("autocast_adapter_compute must be a boolean")
        if self.autocast_adapter_compute and self.dtype != "bfloat16":
            raise ValueError("autocast_adapter_compute currently requires dtype='bfloat16'")
        if not isinstance(self.fused_adamw, bool):
            raise ValueError("fused_adamw must be a boolean")


def load_training_recipe(path: str | Path) -> TrainingRecipe:
    """Load a recipe, resolving datasets and output relative to the recipe file."""

    recipe_path = Path(path).resolve()
    with recipe_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("training recipe must be a JSON object")

    base = recipe_path.parent
    model_id = raw.get("model_id")
    if isinstance(model_id, str):
        local_model = (base / model_id).resolve()
        if local_model.exists():
            raw["model_id"] = str(local_model)
    try:
        experts = tuple(
            ExpertRecipe(
                name=item["name"],
                dataset=(base / item["dataset"]).resolve(),
                epochs=item.get("epochs"),
                initial_adapter=(
                    (base / item["initial_adapter"]).resolve()
                    if item.get("initial_adapter") is not None
                    else None
                ),
                start_optimizer_step=item.get("start_optimizer_step", 0),
            )
            for item in raw.pop("experts")
        )
        output_dir = (base / raw.pop("output_dir")).resolve()
        target_modules = tuple(raw.pop("target_modules"))
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "recipe needs output_dir, target_modules, and expert name/dataset rows"
        ) from exc
    return TrainingRecipe(
        output_dir=output_dir,
        experts=experts,
        target_modules=target_modules,
        **raw,
    )
