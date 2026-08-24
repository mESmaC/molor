"""Minimal offline DPO for one new expert against the frozen base route."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from molor.training.data import AssistantOnlyCollator, ChatExample, ChatMessage


@dataclass(frozen=True, slots=True)
class PreferenceExample:
    messages: tuple[ChatMessage, ...]
    chosen: str
    rejected: str


def load_preference_jsonl(path: str | Path) -> list[PreferenceExample]:
    source = Path(path)
    examples: list[PreferenceExample] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                messages = tuple(
                    ChatMessage(role=message["role"], content=message["content"])
                    for message in raw["messages"]
                )
                chosen = raw["chosen"]
                rejected = raw["rejected"]
                if not messages or messages[-1].role != "user":
                    raise ValueError("messages must end with a user prompt")
                if not isinstance(chosen, str) or not chosen.strip():
                    raise ValueError("chosen must be a non-empty string")
                if not isinstance(rejected, str) or not rejected.strip():
                    raise ValueError("rejected must be a non-empty string")
                examples.append(PreferenceExample(messages, chosen, rejected))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}:{line_number}: invalid preference row: {exc}") from exc
    if not examples:
        raise ValueError(f"{source} contains no preference pairs")
    return examples


@dataclass(frozen=True, slots=True)
class PreferenceRecipe:
    model_id: str
    dataset: Path
    output_dir: Path
    expert_name: str
    target_modules: tuple[str, ...]
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    max_length: int = 512
    epochs: int = 2
    learning_rate: float = 5e-5
    beta: float = 0.1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    chat_template_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id or not self.expert_name:
            raise ValueError("model_id and expert_name are required")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16, or float32")
        for name, value in (
            ("rank", self.rank),
            ("max_length", self.max_length),
            ("epochs", self.epochs),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("alpha", self.alpha),
            ("learning_rate", self.learning_rate),
            ("beta", self.beta),
            ("max_grad_norm", self.max_grad_norm),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def load_preference_recipe(path: str | Path) -> PreferenceRecipe:
    recipe_path = Path(path).resolve()
    raw = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("preference recipe must be a JSON object")
    base = recipe_path.parent
    for field_name in ("model_id", "dataset", "output_dir"):
        raw_value = raw[field_name]
        candidate = (base / raw_value).resolve()
        if field_name != "model_id" or candidate.exists():
            raw[field_name] = candidate if field_name != "model_id" else str(candidate)
    raw["target_modules"] = tuple(raw["target_modules"])
    return PreferenceRecipe(**raw)


class PreferenceEngine(Protocol):
    model: nn.Module
    tokenizer: Any

    def add_expert(self, name: str, *, rank: int, alpha: float, dropout: float) -> object: ...

    def set_trainable_expert(self, name: str) -> object: ...

    def expert_parameters(self, name: str) -> Iterator[nn.Parameter]: ...

    def use_route(self, weights: Mapping[str, float], *, normalize: bool = True) -> Any: ...

    def save_expert(self, name: str, directory: str | Path) -> Path: ...


@dataclass(frozen=True, slots=True)
class PreferenceProgress:
    phase: str
    epoch: int
    epochs: int
    step: int
    total_steps: int
    loss: float | None = None


def _input_device(model: nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()  # type: ignore[attr-defined]
    return embeddings.weight.device


def _pair_batch(
    collator: AssistantOnlyCollator,
    example: PreferenceExample,
    device: torch.device,
) -> dict[str, Tensor]:
    chosen = ChatExample((*example.messages, ChatMessage("assistant", example.chosen)))
    rejected = ChatExample((*example.messages, ChatMessage("assistant", example.rejected)))
    return {key: value.to(device) for key, value in collator([chosen, rejected]).items()}


def _sequence_log_probabilities(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    output = model(**batch)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise RuntimeError("model forward did not return token logits")
    labels = batch["labels"][:, 1:]
    shifted_logits = logits[:, :-1].float()
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1))
    return (token_logps.squeeze(-1) * mask).sum(dim=-1)


def train_preference_expert(
    engine: PreferenceEngine,
    recipe: PreferenceRecipe,
    *,
    progress: Callable[[PreferenceProgress], None] | None = None,
) -> float:
    """Train one policy adapter with DPO while the base/null route is the reference."""

    examples = load_preference_jsonl(recipe.dataset)
    torch.manual_seed(recipe.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe.seed)
    engine.add_expert(
        recipe.expert_name,
        rank=recipe.rank,
        alpha=recipe.alpha,
        dropout=recipe.dropout,
    )
    engine.set_trainable_expert(recipe.expert_name)
    parameters = list(engine.expert_parameters(recipe.expert_name))
    model = engine.model
    device = _input_device(model)
    collator = AssistantOnlyCollator(
        engine.tokenizer,
        max_length=recipe.max_length,
        chat_template_kwargs=dict(recipe.chat_template_kwargs),
    )

    if recipe.gradient_checkpointing:
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise RuntimeError("model does not support gradient checkpointing")
        enable_checkpointing()
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    model_config = getattr(model, "config", None)
    previous_use_cache = getattr(model_config, "use_cache", None)
    if model_config is not None and previous_use_cache is not None:
        model_config.use_cache = False

    reference_ratios: list[float] = []
    model.eval()
    with torch.inference_mode(), engine.use_route({"base": 1.0}, normalize=False):
        for index, example in enumerate(examples, start=1):
            logps = _sequence_log_probabilities(model, _pair_batch(collator, example, device))
            reference_ratios.append(float(logps[0] - logps[1]))
            if progress is not None:
                progress(PreferenceProgress("reference", 0, recipe.epochs, index, len(examples)))

    optimizer = torch.optim.AdamW(parameters, lr=recipe.learning_rate)
    steps_per_epoch = math.ceil(len(examples) / recipe.gradient_accumulation_steps)
    total_steps = recipe.epochs * steps_per_epoch
    optimizer_step = 0
    final_loss = 0.0
    model.train()
    try:
        for epoch in range(1, recipe.epochs + 1):
            order = list(range(len(examples)))
            random.Random(recipe.seed + epoch).shuffle(order)
            for offset in range(0, len(order), recipe.gradient_accumulation_steps):
                group = order[offset : offset + recipe.gradient_accumulation_steps]
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for index in group:
                    policy_logps = _sequence_log_probabilities(
                        model, _pair_batch(collator, examples[index], device)
                    )
                    policy_ratio = policy_logps[0] - policy_logps[1]
                    margin = recipe.beta * (policy_ratio - reference_ratios[index])
                    loss = -F.logsigmoid(margin)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite DPO loss")
                    group_loss += float(loss.detach())
                    (loss / len(group)).backward()
                torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm)
                optimizer.step()
                optimizer_step += 1
                final_loss = group_loss / len(group)
                if progress is not None:
                    progress(
                        PreferenceProgress(
                            "train",
                            epoch,
                            recipe.epochs,
                            optimizer_step,
                            total_steps,
                            final_loss,
                        )
                    )
        engine.save_expert(recipe.expert_name, recipe.output_dir / recipe.expert_name)
    finally:
        if model_config is not None and previous_use_cache is not None:
            model_config.use_cache = previous_use_cache
    model.eval()
    return final_loss
