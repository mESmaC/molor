"""Run the one sealed, quarantined Taskmaster coffee API v2 training job.

The launcher intentionally exposes no recipe overrides.  It rehashes and seals
the full training receipt immediately before loading the frozen base, refuses to
overwrite an existing candidate, and writes a content-addressed completion
receipt beside the resulting adapter.  It never reads the v2 test shard.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import torch

from molor.experimental import MixtureEngine
from molor.training.recipe import load_training_recipe
from molor.training.sft import TrainingProgress, train_experts
from seal_taskmaster_coffee_api_v2_training import (
    DATA,
    DESTINATION,
    HERE,
    RECIPE,
    REPOSITORY,
    canonical_json,
    seal_training_receipt,
    sha256_file,
)


COMPLETION_SCHEMA = "molor.taskmaster-coffee-api-training-completion.v2"


def _require_workspace_code(repository: Path) -> None:
    expected = {
        Path(inspect.getsourcefile(MixtureEngine) or "").resolve(): (
            repository / "src" / "molor" / "experimental" / "mixture.py"
        ).resolve(),
        Path(inspect.getsourcefile(train_experts) or "").resolve(): (
            repository / "src" / "molor" / "training" / "sft.py"
        ).resolve(),
        Path(inspect.getsourcefile(load_training_recipe) or "").resolve(): (
            repository / "src" / "molor" / "training" / "recipe.py"
        ).resolve(),
        Path(inspect.getsourcefile(seal_training_receipt) or "").resolve(): (
            repository
            / "experiments"
            / "first-real-experts"
            / "seal_taskmaster_coffee_api_v2_training.py"
        ).resolve(),
    }
    if any(actual != wanted for actual, wanted in expected.items()):
        raise RuntimeError("training code was not imported from the sealed workspace")


def _print_progress(update: TrainingProgress) -> None:
    print(
        f"[{update.expert}] epoch {update.epoch}/{update.epochs} "
        f"step {update.optimizer_step}/{update.total_optimizer_steps} "
        f"loss {update.loss:.4f}",
        flush=True,
    )


def _output_descriptor(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_completion_receipt(
    *,
    repository: Path,
    candidate_directory: Path,
    preflight_path: Path,
    preflight_payload_sha256: str,
    preflight_file_sha256: str,
    train_dataset_sha256: str,
    final_loss: float,
) -> Path:
    if sha256_file(preflight_path) != preflight_file_sha256:
        raise ValueError("preflight receipt changed during training")
    artifacts = []
    for name in ("adapter_config.json", "adapter_model.safetensors", "training_summary.json"):
        path = candidate_directory / name
        if not path.is_file():
            raise FileNotFoundError(f"completed training did not produce {path}")
        artifacts.append(_output_descriptor(path, root=repository))
    summary_path = candidate_directory / "training_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("completed training summary is unreadable") from exc
    if not isinstance(summary, dict) or summary.get("dataset_sha256") != train_dataset_sha256:
        raise ValueError("trained dataset hash does not match the sealed preflight")
    receipt: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "completed_quarantined_candidate_not_promotion_eligible",
        "candidate": candidate_directory.name,
        "preflight_receipt": _output_descriptor(preflight_path, root=repository),
        "preflight_payload_sha256": preflight_payload_sha256,
        "artifacts": artifacts,
        "final_training_loss": final_loss,
        "test_evaluation_opened": False,
        "registration_changed": False,
    }
    receipt["payload_sha256"] = hashlib.sha256(canonical_json(receipt).encode()).hexdigest()
    destination = candidate_directory / "training_completion_receipt.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return destination


def launch_training(
    *,
    repository: Path = REPOSITORY,
    experiment: Path = HERE,
    data: Path = DATA,
    receipt_path: Path = DESTINATION,
    preflight_only: bool = False,
    workspace_code_verifier: Callable[[Path], None] = _require_workspace_code,
) -> dict[str, Any]:
    """Revalidate the sealed contract and, unless requested otherwise, train once."""

    repository = repository.resolve()
    experiment = experiment.resolve()
    data = data.resolve()
    receipt_path = receipt_path.resolve()
    try:
        receipt_path.relative_to(experiment)
    except ValueError as exc:
        raise ValueError("training receipt must remain inside the sealed experiment") from exc
    workspace_code_verifier(repository)
    preflight = seal_training_receipt(
        repository=repository,
        experiment=experiment,
        data=data,
        destination=receipt_path,
    )
    preflight_file_sha256 = sha256_file(receipt_path)
    print(f"sealed preflight: {receipt_path} ({preflight_file_sha256})", flush=True)
    if preflight_only:
        return {"status": "preflight_only", "receipt": preflight}

    recipe_path = experiment / RECIPE.name
    recipe = load_training_recipe(recipe_path)
    if sha256_file(recipe_path) != preflight["identities"]["recipe_sha256"]:
        raise ValueError("recipe changed after the sealed preflight")
    expert = recipe.experts[0]
    if expert.name != preflight["candidate"]:
        raise ValueError("recipe candidate changed after the sealed preflight")
    approved_output = (
        repository / "checkpoints" / "qwen3-1.7b-candidate-experts-quarantine"
    ).resolve()
    if recipe.output_dir.resolve() != approved_output:
        raise ValueError("recipe output changed after the sealed preflight")
    candidate_directory = recipe.output_dir / expert.name
    if candidate_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite the sealed candidate directory: {candidate_directory}"
        )
    recovery_root = recipe.output_dir / ".checkpoints"
    if recovery_root.is_dir() and any(
        path.is_dir() and path.name.startswith(f"{expert.name}-step-")
        for path in recovery_root.iterdir()
    ):
        raise FileExistsError("refusing a second attempt after a v2 recovery checkpoint exists")
    expected_model = (repository / "models" / "Qwen3-1.7B").resolve()
    if Path(recipe.model_id).resolve() != expected_model:
        raise ValueError("sealed training requires the exact policy-bound local base model")
    train_path = data / "taskmaster-coffee-api-v2.train.jsonl"
    if expert.dataset.resolve() != train_path.resolve():
        raise ValueError("recipe dataset changed after the sealed preflight")
    if sha256_file(train_path) != preflight["identities"]["train_dataset_sha256"]:
        raise ValueError("training dataset changed after the sealed preflight")

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
        local_files_only=True,
    )
    if sha256_file(recipe_path) != preflight["identities"]["recipe_sha256"]:
        raise ValueError("recipe changed while loading the frozen base")
    if sha256_file(train_path) != preflight["identities"]["train_dataset_sha256"]:
        raise ValueError("training dataset changed while loading the frozen base")
    losses = train_experts(engine, recipe, progress=_print_progress)
    completion = _write_completion_receipt(
        repository=repository,
        candidate_directory=candidate_directory,
        preflight_path=receipt_path,
        preflight_payload_sha256=preflight["payload_sha256"],
        preflight_file_sha256=preflight_file_sha256,
        train_dataset_sha256=preflight["identities"]["train_dataset_sha256"],
        final_loss=losses[expert.name],
    )
    print(f"saved quarantined candidate: {candidate_directory}", flush=True)
    print(f"completion receipt: {completion} ({sha256_file(completion)})", flush=True)
    return {
        "status": "completed_quarantined",
        "receipt": preflight,
        "candidate_directory": str(candidate_directory),
        "completion_receipt": str(completion),
        "final_loss": losses[expert.name],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the single sealed Taskmaster coffee API v2 training job."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="seal and verify all training inputs without loading the model",
    )
    args = parser.parse_args()
    launch_training(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
