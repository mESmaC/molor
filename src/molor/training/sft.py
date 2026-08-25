"""A compact sequential SFT loop for independently useful LoRA experts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Protocol

import torch
from torch import Tensor, nn

from molor.training.data import AssistantOnlyCollator, ChatTemplateTokenizer, load_jsonl
from molor.training.recipe import TrainingRecipe


class ExpertTrainingEngine(Protocol):
    """The training slice of :class:`molor.experimental.MixtureEngine`."""

    model: nn.Module
    tokenizer: ChatTemplateTokenizer

    def add_expert(
        self,
        name: str,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        parameter_dtype: torch.dtype | None = None,
    ) -> object: ...

    def load_expert(
        self,
        directory: str | Path,
        *,
        name: str | None = None,
        parameter_dtype: torch.dtype | None = None,
    ) -> str: ...

    def set_trainable_expert(self, name: str) -> None: ...

    def expert_parameters(self, name: str) -> Iterator[nn.Parameter]: ...

    def save_expert(self, name: str, directory: str | Path) -> Path: ...


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    expert: str
    epoch: int
    epochs: int
    optimizer_step: int
    total_optimizer_steps: int
    loss: float


ProgressCallback = Callable[[TrainingProgress], None]


def _chunks(values: list[int], width: int) -> Iterable[list[int]]:
    for offset in range(0, len(values), width):
        yield values[offset : offset + width]


def _learning_rate_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    schedule: str,
) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _input_device(model: nn.Module) -> torch.device:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, Tensor) and weight.device.type != "meta":
            return weight.device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("could not determine the model input device")


def _loss_from_output(output: object) -> Tensor:
    loss = getattr(output, "loss", None)
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise RuntimeError("model forward did not return a scalar .loss")
    return loss


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_training_summary(
    directory: Path,
    *,
    recipe: TrainingRecipe,
    expert_name: str,
    dataset: Path,
    examples: int,
    epochs: int,
    optimizer_steps: int,
    final_loss: float,
    elapsed_seconds: float,
    initial_adapter: Path | None,
    start_optimizer_step: int,
    optimizer_steps_this_run: int,
) -> None:
    warm_start_hashes = None
    if initial_adapter is not None:
        warm_start_hashes = {
            name: _file_sha256(initial_adapter / name)
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "recovery_summary.json",
            )
            if (initial_adapter / name).is_file()
        }
    summary = {
        "expert": expert_name,
        "model_id": recipe.model_id,
        "dataset": str(dataset),
        "dataset_sha256": _file_sha256(dataset),
        "examples": examples,
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_this_run": optimizer_steps_this_run,
        "start_optimizer_step": start_optimizer_step,
        "warm_started_from_adapter": (
            str(initial_adapter) if initial_adapter is not None else None
        ),
        "warm_start_artifact_sha256": warm_start_hashes,
        "optimizer_state_resumed": False if initial_adapter is not None else None,
        "scheduler_state_resumed": False if initial_adapter is not None else None,
        "scheduler_phase_offset": start_optimizer_step,
        "skipped_group_count": start_optimizer_step,
        "final_training_loss": final_loss,
        "elapsed_seconds": elapsed_seconds,
        "dtype": recipe.dtype,
        "adapter_parameter_dtype": "float32",
        "device": recipe.device,
        "target_modules": list(recipe.target_modules),
        "rank": recipe.rank,
        "alpha": recipe.alpha,
        "dropout": recipe.dropout,
        "max_length": recipe.max_length,
        "learning_rate": recipe.learning_rate,
        "lr_scheduler": recipe.lr_scheduler,
        "warmup_ratio": recipe.warmup_ratio,
        "weight_decay": recipe.weight_decay,
        "micro_batch_size": recipe.micro_batch_size,
        "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
        "effective_batch_size": (recipe.micro_batch_size * recipe.gradient_accumulation_steps),
        "gradient_checkpointing": recipe.gradient_checkpointing,
        "autocast_adapter_compute": recipe.autocast_adapter_compute,
        "batching_strategy": recipe.batching_strategy,
        "fused_adamw": recipe.fused_adamw,
        "checkpoint_every_steps": recipe.checkpoint_every_steps,
        "max_grad_norm": recipe.max_grad_norm,
        "seed": recipe.seed,
        "chat_template_kwargs": recipe.chat_template_kwargs,
    }
    destination = directory / "training_summary.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _validate_recovery_checkpoint(
    directory: Path,
    *,
    expert_name: str,
    dataset: Path,
    start_optimizer_step: int,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: tuple[str, ...],
) -> dict[str, object] | None:
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapter_model.safetensors"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("initial_adapter needs a readable adapter_config.json") from exc
    if not weights_path.is_file():
        raise ValueError("initial_adapter is missing adapter_model.safetensors")
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain an object")
    # A completed adapter may seed a newly named successor (for example v1 -> v2).
    # Exact-name matching remains mandatory for an interrupted-run recovery because
    # its step counter and recovery sidecar belong to that specific expert identity.
    if config.get("name") != expert_name and start_optimizer_step != 0:
        raise ValueError("recovery adapter expert does not match the recipe")
    if config.get("rank") != rank or config.get("alpha") != float(alpha):
        raise ValueError("initial adapter rank/alpha do not match the recipe")
    if config.get("dropout") != float(dropout):
        raise ValueError("initial adapter dropout does not match the recipe")
    if tuple(config.get("target_modules", ())) != target_modules:
        raise ValueError("initial adapter target modules do not match the recipe")

    summary_path = directory / "recovery_summary.json"
    if start_optimizer_step == 0:
        if summary_path.exists():
            raise ValueError(
                "an adapter recovery checkpoint requires an explicit nonzero start_optimizer_step"
            )
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "a nonzero start_optimizer_step requires a readable recovery_summary.json"
        ) from exc
    expected_hash = _file_sha256(dataset)
    if not isinstance(payload, dict):
        raise ValueError("recovery_summary.json must contain an object")
    if payload.get("kind") != "adapter_recovery_checkpoint":
        raise ValueError("initial adapter is not a recognized recovery checkpoint")
    if payload.get("expert") != expert_name:
        raise ValueError("recovery checkpoint expert does not match the recipe")
    if payload.get("dataset_sha256") != expected_hash:
        raise ValueError("recovery checkpoint dataset hash does not match the recipe")
    if payload.get("optimizer_step") != start_optimizer_step:
        raise ValueError("recovery checkpoint step does not match start_optimizer_step")
    return payload


def _write_recovery_checkpoint(
    engine: ExpertTrainingEngine,
    *,
    recipe: TrainingRecipe,
    expert_name: str,
    dataset: Path,
    optimizer_step: int,
    total_optimizer_steps: int,
    epoch: int,
    loss: float,
) -> None:
    directory = engine.save_expert(
        expert_name,
        recipe.output_dir / ".checkpoints" / f"{expert_name}-step-{optimizer_step:06d}",
    )
    payload = {
        "kind": "adapter_recovery_checkpoint",
        "expert": expert_name,
        "dataset": str(dataset),
        "dataset_sha256": _file_sha256(dataset),
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "total_optimizer_steps": total_optimizer_steps,
        "loss": loss,
        "note": (
            "Adapter weights only; optimizer state is not retained, so this is a warm-restart "
            "checkpoint rather than an exact training resume."
        ),
    }
    (directory / "recovery_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def train_experts(
    engine: ExpertTrainingEngine,
    recipe: TrainingRecipe,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, float]:
    """Attach every expert once, then train and save each against the same frozen base."""

    torch.manual_seed(recipe.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe.seed)
    if engine.tokenizer is None:
        raise RuntimeError("training requires the model's chat tokenizer")
    if any(expert.start_optimizer_step for expert in recipe.experts) and len(recipe.experts) != 1:
        raise ValueError("warm continuation recipes must contain exactly one expert")

    recovery_summaries: dict[str, dict[str, object]] = {}
    for expert in recipe.experts:
        if expert.initial_adapter is None:
            engine.add_expert(
                expert.name,
                rank=recipe.rank,
                alpha=recipe.alpha,
                dropout=recipe.dropout,
                parameter_dtype=torch.float32,
            )
        else:
            recovery_summary = _validate_recovery_checkpoint(
                expert.initial_adapter,
                expert_name=expert.name,
                dataset=expert.dataset,
                start_optimizer_step=expert.start_optimizer_step,
                rank=recipe.rank,
                alpha=recipe.alpha,
                dropout=recipe.dropout,
                target_modules=recipe.target_modules,
            )
            if recovery_summary is not None:
                recovery_summaries[expert.name] = recovery_summary
            engine.load_expert(
                expert.initial_adapter,
                name=expert.name,
                parameter_dtype=torch.float32,
            )

    model = engine.model
    device = _input_device(model)
    if recipe.autocast_adapter_compute and device.type != "cuda":
        raise RuntimeError("autocast_adapter_compute requires a CUDA training device")
    if recipe.fused_adamw and device.type != "cuda":
        raise RuntimeError("fused_adamw requires a CUDA training device")
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

    final_losses: dict[str, float] = {}
    try:
        for expert_index, expert in enumerate(recipe.experts):
            expert_started = time.perf_counter()
            examples = load_jsonl(expert.dataset)
            expert_epochs = expert.epochs or recipe.epochs
            collator = AssistantOnlyCollator(
                engine.tokenizer,
                max_length=recipe.max_length,
                chat_template_kwargs=recipe.chat_template_kwargs,
            )
            encoded_examples = (
                None
                if recipe.batching_strategy == "online"
                else tuple(collator.encode(example) for example in examples)
            )
            engine.set_trainable_expert(expert.name)
            parameters = list(engine.expert_parameters(expert.name))
            if not parameters or any(not parameter.requires_grad for parameter in parameters):
                raise RuntimeError(f"expert {expert.name!r} did not expose trainable parameters")
            if any(parameter.dtype != torch.float32 for parameter in parameters):
                raise RuntimeError(f"expert {expert.name!r} parameters are not float32")
            optimizer_kwargs: dict[str, object] = {
                "lr": recipe.learning_rate,
                "weight_decay": recipe.weight_decay,
            }
            if recipe.fused_adamw:
                optimizer_kwargs["fused"] = True
            optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)
            examples_per_step = recipe.micro_batch_size * recipe.gradient_accumulation_steps
            groups_per_epoch = math.ceil(len(examples) / examples_per_step)
            total_steps = expert_epochs * groups_per_epoch
            if expert.start_optimizer_step >= total_steps:
                raise ValueError(
                    f"expert {expert.name!r} starts at step "
                    f"{expert.start_optimizer_step}, but this run has only {total_steps} steps"
                )
            recovery_summary = recovery_summaries.get(expert.name)
            if (
                recovery_summary is not None
                and recovery_summary.get("total_optimizer_steps") != total_steps
            ):
                raise ValueError("recovery checkpoint total step horizon does not match the recipe")
            warmup_steps = round(total_steps * recipe.warmup_ratio)
            optimizer_step = expert.start_optimizer_step
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: _learning_rate_multiplier(
                    step + expert.start_optimizer_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    schedule=recipe.lr_scheduler,
                ),
            )
            model.train()

            global_group_index = 0
            for epoch in range(1, expert_epochs + 1):
                order = list(range(len(examples)))
                random.Random(recipe.seed + expert_index * 10_000 + epoch).shuffle(order)
                for group in _chunks(order, examples_per_step):
                    if global_group_index < expert.start_optimizer_step:
                        global_group_index += 1
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    group_loss = 0.0
                    prepared_batches: list[tuple[dict[str, Tensor], int]] = []
                    if encoded_examples is None:
                        for micro_batch in _chunks(group, recipe.micro_batch_size):
                            batch = collator(
                                [examples[example_index] for example_index in micro_batch]
                            )
                            target_tokens = int((batch["labels"] != -100).sum())
                            prepared_batches.append((batch, target_tokens))
                    else:
                        encoded_group = [encoded_examples[example_index] for example_index in group]
                        if recipe.batching_strategy == "pretokenized_group_sorted":
                            encoded_group.sort(
                                key=lambda example: len(example.input_ids), reverse=True
                            )
                        for offset in range(0, len(encoded_group), recipe.micro_batch_size):
                            encoded_batch = encoded_group[offset : offset + recipe.micro_batch_size]
                            batch = collator.collate_encoded(encoded_batch)
                            target_tokens = sum(
                                label != -100
                                for encoded in encoded_batch
                                for label in encoded.labels
                            )
                            prepared_batches.append((batch, target_tokens))
                    group_target_tokens = sum(count for _batch, count in prepared_batches)
                    for cpu_batch, target_tokens in prepared_batches:
                        batch = {key: value.to(device) for key, value in cpu_batch.items()}
                        autocast_context = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if recipe.autocast_adapter_compute
                            else nullcontext()
                        )
                        with autocast_context:
                            loss = _loss_from_output(model(**batch))
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"non-finite loss while training {expert.name!r}")
                        micro_batch_weight = target_tokens / group_target_tokens
                        group_loss += float(loss.detach()) * micro_batch_weight
                        (loss * micro_batch_weight).backward()
                    torch.nn.utils.clip_grad_norm_(parameters, recipe.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer_step += 1
                    global_group_index += 1
                    final_losses[expert.name] = group_loss
                    if (
                        recipe.checkpoint_every_steps is not None
                        and optimizer_step % recipe.checkpoint_every_steps == 0
                        and optimizer_step < total_steps
                    ):
                        _write_recovery_checkpoint(
                            engine,
                            recipe=recipe,
                            expert_name=expert.name,
                            dataset=expert.dataset,
                            optimizer_step=optimizer_step,
                            total_optimizer_steps=total_steps,
                            epoch=epoch,
                            loss=group_loss,
                        )
                    if progress is not None:
                        progress(
                            TrainingProgress(
                                expert=expert.name,
                                epoch=epoch,
                                epochs=expert_epochs,
                                optimizer_step=optimizer_step,
                                total_optimizer_steps=total_steps,
                                loss=final_losses[expert.name],
                            )
                        )

            saved_directory = engine.save_expert(expert.name, recipe.output_dir / expert.name)
            _write_training_summary(
                saved_directory,
                recipe=recipe,
                expert_name=expert.name,
                dataset=expert.dataset,
                examples=len(examples),
                epochs=expert_epochs,
                optimizer_steps=optimizer_step,
                final_loss=final_losses[expert.name],
                elapsed_seconds=time.perf_counter() - expert_started,
                initial_adapter=expert.initial_adapter,
                start_optimizer_step=expert.start_optimizer_step,
                optimizer_steps_this_run=(optimizer_step - expert.start_optimizer_step),
            )
    finally:
        if model_config is not None and previous_use_cache is not None:
            model_config.use_cache = previous_use_cache

    model.eval()
    return final_losses
