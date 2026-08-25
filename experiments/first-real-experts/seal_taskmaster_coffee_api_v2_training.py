"""Seal the immutable preflight receipt for the Taskmaster coffee API v2 run.

This command deliberately reads only training-side generated artifacts.  It does
not open the v2 test-evaluation shard or load a model.  The resulting receipt
binds the dataset builder, policy, recipe, build report, training inputs, local
base-model bytes, and exact dependency versions before the single declared run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
from typing import Any, Callable, Mapping

from molor.benchmark.lock import lock_backend_identity
from molor.training.data import load_jsonl
from molor.training.recipe import load_training_recipe


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DATASET_ID = "taskmaster-coffee-api-v2"
BUILDER = HERE / "build_taskmaster_coffee_api_v2.py"
SOURCE_MANIFEST = HERE / f"{DATASET_ID}-source.json"
RECIPE = HERE / f"{DATASET_ID}-recipe.json"
POLICY = HERE / f"{DATASET_ID}-evaluation-policy.json"
DATA = HERE / "data"
BUILD_REPORT = DATA / f"{DATASET_ID}.build-report.json"
DESTINATION = HERE / f"{DATASET_ID}-training-receipt.json"
TRAINING_LAUNCHER = HERE / "run_taskmaster_coffee_api_v2_training.py"
APPROVED_OUTPUT_ROOT = REPOSITORY / "checkpoints" / "qwen3-1.7b-candidate-experts-quarantine"
RECEIPT_SCHEMA = "molor.taskmaster-coffee-api-training-receipt.v2"
_SHA256_BUFFER = 8 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_SHA256_BUFFER), b""):
            digest.update(chunk)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise OSError(f"file changed while it was being hashed: {path}")
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_hash(actual: str, expected: object, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"{name} does not expose __version__")
        versions[name] = version
    return versions


def _base_artifact_lock(path: Path) -> dict[str, Any]:
    lock = lock_backend_identity(
        {"backend": "local", "model_artifact_path": str(path)},
        cache_path=Path("unused-content-hash-cache.json"),
    )
    if lock.get("fully_content_addressed") is not True:
        raise ValueError(f"base model could not be fully content-addressed: {lock['unresolved']}")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("base lock must contain exactly one artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or artifact.get("role") != "base":
        raise ValueError("base lock returned an invalid artifact")
    return {
        "role": "base",
        "content_fingerprint": artifact["content_fingerprint"],
        "size_bytes": artifact["size_bytes"],
        "files": artifact["files"],
        "complete": artifact["complete"],
    }


def _validate_content_tree(
    value: Mapping[str, Any],
    *,
    role: str | None = None,
) -> dict[str, Any]:
    result = dict(value)
    if role is not None and result.get("role") != role:
        raise ValueError(f"content tree must have role {role!r}")
    if role is not None and result.get("complete") is not True:
        raise ValueError("content tree is not marked complete")
    files = result.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("content tree needs non-empty file descriptors")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {"name", "size_bytes", "sha256"}:
            raise ValueError("content tree has an invalid file descriptor")
        name = raw["name"]
        size = raw["size_bytes"]
        digest = raw["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or name in names
        ):
            raise ValueError("content tree has an unsafe or duplicate file name")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("content tree has an invalid file size")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("content tree has an invalid file hash")
        names.add(name)
        normalized.append({"name": name, "size_bytes": size, "sha256": digest})
    if normalized != sorted(normalized, key=lambda item: item["name"]):
        raise ValueError("content tree file descriptors must be path-sorted")
    expected_payload: dict[str, Any] = {"files": normalized}
    if role is not None:
        expected_payload["role"] = role
        size = result.get("size_bytes")
        if size != sum(item["size_bytes"] for item in normalized):
            raise ValueError("content tree aggregate size does not reconcile")
    fingerprint = hashlib.sha256(canonical_json(expected_payload).encode()).hexdigest()
    if result.get("content_fingerprint") != fingerprint:
        raise ValueError("content tree fingerprint does not match its file descriptors")
    result["files"] = normalized
    return result


def _trainer_source_descriptor(repository: Path, experiment: Path) -> dict[str, Any]:
    source_files = {
        repository / "pyproject.toml",
        experiment / Path(__file__).name,
        experiment / TRAINING_LAUNCHER.name,
    }
    source_files.update((repository / "src" / "molor").rglob("*.py"))
    missing = [path for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"trainer source contract is missing files: {sorted(missing)}")
    files = []
    for path in sorted(source_files, key=lambda item: item.relative_to(repository).as_posix()):
        descriptor = _descriptor(path, relative_to=repository)
        files.append(
            {
                "name": descriptor["path"],
                "sha256": descriptor["sha256"],
                "size_bytes": descriptor["size_bytes"],
            }
        )
    fingerprint = hashlib.sha256(canonical_json({"files": files}).encode()).hexdigest()
    return {
        "hash_policy": "sha256-full-read-canonical-relative-path-tree-v1",
        "content_fingerprint": fingerprint,
        "files": files,
    }


def _safe_repository_path(repository: Path, raw_path: object, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} must be a repository-relative path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes the repository")
    resolved = (repository / relative).resolve()
    try:
        resolved.relative_to(repository.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    return resolved


def _descriptor(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_descriptor(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    expected_path: Path | None = None,
    count_rows: bool = False,
) -> Path:
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("artifact descriptor has no path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact descriptor path escapes its declared root")
    path = root / relative
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise ValueError(f"descriptor path mismatch: expected {expected_path}, found {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    _require_hash(sha256_file(path), descriptor.get("sha256"), label=str(path))
    if path.stat().st_size != descriptor.get("size_bytes"):
        raise ValueError(f"artifact size mismatch for {path}")
    if count_rows and _count_nonempty_lines(path) != descriptor.get("rows"):
        raise ValueError(f"artifact row-count mismatch for {path}")
    return path


def seal_training_receipt(
    *,
    repository: Path = REPOSITORY,
    experiment: Path = HERE,
    data: Path = DATA,
    destination: Path = DESTINATION,
    dependency_versions: Mapping[str, str] | None = None,
    base_artifact_locker: Callable[[Path], Mapping[str, Any]] = _base_artifact_lock,
    trainer_source_locker: Callable[[Path, Path], Mapping[str, Any]] = (_trainer_source_descriptor),
) -> dict[str, Any]:
    """Verify all declared training inputs and atomically write the receipt."""

    repository = repository.resolve()
    experiment = experiment.resolve()
    data = data.resolve()
    expected_experiment = repository / "experiments" / "first-real-experts"
    if experiment != expected_experiment.resolve():
        raise ValueError("experiment root does not match the repository contract")
    try:
        experiment.relative_to(repository)
    except ValueError as exc:
        raise ValueError("experiment root resolves outside the repository") from exc
    if data != (experiment / "data").resolve():
        raise ValueError("data root must be the experiment's generated data directory")
    try:
        data.relative_to(experiment)
    except ValueError as exc:
        raise ValueError("data root resolves outside the experiment") from exc
    try:
        destination.resolve().relative_to(experiment)
    except ValueError as exc:
        raise ValueError("training receipt destination must stay inside the experiment") from exc
    builder_path = experiment / BUILDER.name
    source_path = experiment / SOURCE_MANIFEST.name
    recipe_path = experiment / RECIPE.name
    policy_path = experiment / POLICY.name
    report_path = data / BUILD_REPORT.name

    policy = _load_object(policy_path)
    report = _load_object(report_path)
    recipe_raw = _load_object(recipe_path)
    recipe = load_training_recipe(recipe_path)

    if policy.get("candidate_posture") != "quarantine_until_every_gate_passes":
        raise ValueError("policy no longer keeps the candidate quarantined")
    if report.get("dataset_id") != DATASET_ID:
        raise ValueError("build report dataset identity mismatch")
    if report.get("intended_checkpoint_posture") != (
        "candidate_quarantine_only_never_auto_promote"
    ):
        raise ValueError("build report no longer enforces quarantine")

    builder = _descriptor(builder_path, relative_to=experiment)
    source = _descriptor(source_path, relative_to=experiment)
    recipe_descriptor = _descriptor(recipe_path, relative_to=experiment)
    policy_descriptor = _descriptor(policy_path, relative_to=experiment)
    report_descriptor = _descriptor(report_path, relative_to=data)

    pins = policy.get("dataset_contract", {}).get("artifact_identity_inputs", {})
    for label, descriptor in (
        ("builder", builder),
        ("source_manifest", source),
        ("recipe", recipe_descriptor),
    ):
        _require_hash(
            descriptor["sha256"],
            pins.get(f"{label}_sha256"),
            label=f"policy {label} pin",
        )

    report_inputs = report.get("build_inputs")
    if not isinstance(report_inputs, dict):
        raise ValueError("build report has no build_inputs object")
    for label, descriptor in (
        ("builder", builder),
        ("source_manifest", source),
        ("recipe", recipe_descriptor),
        ("policy", policy_descriptor),
    ):
        reported = report_inputs.get(label)
        if not isinstance(reported, dict):
            raise ValueError(f"build report has no {label} descriptor")
        _require_hash(descriptor["sha256"], reported.get("sha256"), label=f"build-report {label}")

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("build report has no outputs object")
    train_path = _verify_descriptor(
        outputs.get("train_training", {}),
        root=data,
        expected_path=data / f"{DATASET_ID}.train.jsonl",
        count_rows=True,
    )
    provenance_path = _verify_descriptor(
        outputs.get("train_provenance", {}),
        root=data,
        expected_path=data / f"{DATASET_ID}.train.provenance.jsonl",
        count_rows=True,
    )
    catalog_path = _verify_descriptor(
        outputs.get("tool_catalog", {}),
        root=data,
        expected_path=data / f"{DATASET_ID}.tool-catalog.json",
    )
    training_examples = load_jsonl(train_path)
    if len(training_examples) != outputs["train_training"].get("rows"):
        raise ValueError("trainer-loaded example count does not match the build report")
    _require_hash(
        sha256_file(train_path),
        outputs["train_training"]["sha256"],
        label="train dataset after trainer parsing",
    )

    if len(recipe.experts) != 1 or recipe.experts[0].name != policy.get("candidate"):
        raise ValueError("recipe candidate does not match the policy")
    raw_experts = recipe_raw.get("experts")
    if (
        not isinstance(raw_experts, list)
        or len(raw_experts) != 1
        or not isinstance(raw_experts[0], dict)
        or set(raw_experts[0]) != {"name", "dataset"}
    ):
        raise ValueError("fresh v2 recipe expert must contain exactly name and dataset")
    if recipe.experts[0].initial_adapter is not None or recipe.experts[0].start_optimizer_step != 0:
        raise ValueError("fresh v2 run cannot warm-start or skip optimizer steps")
    if recipe.experts[0].dataset.resolve() != train_path.resolve():
        raise ValueError("recipe does not point at the verified train dataset")
    selection = policy.get("selection", {})
    if selection.get("training_runs") != 1:
        raise ValueError("policy must declare exactly one training run")
    effective_epochs = recipe.experts[0].epochs or recipe.epochs
    if recipe.seed != selection.get("seed") or effective_epochs != selection.get("epochs"):
        raise ValueError("recipe seed/epochs do not match the frozen selection")
    approved_output = (
        repository / "checkpoints" / "qwen3-1.7b-candidate-experts-quarantine"
    ).resolve()
    try:
        approved_output.relative_to(repository)
    except ValueError as exc:
        raise ValueError("approved quarantine root resolves outside the repository") from exc
    if recipe.output_dir.resolve() != approved_output:
        raise ValueError("recipe output directory is not the approved quarantine root")
    if set(recipe_raw) - {
        "model_id",
        "output_dir",
        "dtype",
        "device",
        "target_modules",
        "rank",
        "alpha",
        "dropout",
        "max_length",
        "epochs",
        "learning_rate",
        "lr_scheduler",
        "warmup_ratio",
        "weight_decay",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "checkpoint_every_steps",
        "max_grad_norm",
        "seed",
        "chat_template_kwargs",
        "experts",
        "revision",
    }:
        raise ValueError("recipe contains fields the training entrypoint does not bind")

    contract = policy.get("training_artifact_contract")
    if not isinstance(contract, dict):
        raise ValueError("policy has no training_artifact_contract")
    required = set(contract.get("required_file_identities", ()))
    identities = {
        "builder_sha256": builder["sha256"],
        "source_manifest_sha256": source["sha256"],
        "recipe_sha256": recipe_descriptor["sha256"],
        "policy_sha256": policy_descriptor["sha256"],
        "build_report_sha256": report_descriptor["sha256"],
        "train_dataset_sha256": outputs["train_training"]["sha256"],
        "train_provenance_sha256": outputs["train_provenance"]["sha256"],
        "tool_catalog_sha256": outputs["tool_catalog"]["sha256"],
    }

    expected_base = contract.get("base_artifact")
    if not isinstance(expected_base, dict):
        raise ValueError("policy base artifact is missing")
    base_path = _safe_repository_path(
        repository, expected_base.get("path"), label="policy base artifact path"
    )
    if Path(recipe.model_id).resolve() != base_path.resolve():
        raise ValueError("recipe model does not match the policy base artifact")
    actual_base = _validate_content_tree(base_artifact_locker(base_path), role="base")
    if actual_base.get("content_fingerprint") != expected_base.get("content_fingerprint"):
        raise ValueError("base-model content fingerprint does not match the policy")

    expected_trainer = contract.get("trainer_source_contract")
    if not isinstance(expected_trainer, dict):
        raise ValueError("policy trainer source contract is missing")
    actual_trainer = _validate_content_tree(trainer_source_locker(repository, experiment))
    if actual_trainer.get("content_fingerprint") != expected_trainer.get("content_fingerprint"):
        raise ValueError("trainer source tree does not match the policy")
    identities["trainer_source_tree_sha256"] = actual_trainer["content_fingerprint"]
    if set(identities) != required:
        raise ValueError("receipt identity fields do not exactly match the policy contract")

    actual_dependencies = dict(dependency_versions or _dependency_versions())
    expected_dependencies = contract.get("required_dependency_versions")
    if actual_dependencies != expected_dependencies:
        raise ValueError(
            "dependency contract mismatch: "
            f"expected {expected_dependencies}, found {actual_dependencies}"
        )

    for path, descriptor, label in (
        (builder_path, builder, "builder"),
        (source_path, source, "source manifest"),
        (recipe_path, recipe_descriptor, "recipe"),
        (policy_path, policy_descriptor, "policy"),
        (report_path, report_descriptor, "build report"),
        (train_path, outputs["train_training"], "train dataset"),
        (provenance_path, outputs["train_provenance"], "train provenance"),
        (catalog_path, outputs["tool_catalog"], "tool catalog"),
    ):
        _require_hash(sha256_file(path), descriptor["sha256"], label=f"{label} final read")
    if _load_object(policy_path) != policy or _load_object(report_path) != report:
        raise OSError("policy or build report changed during receipt verification")
    if _load_object(recipe_path) != recipe_raw:
        raise OSError("recipe changed during receipt verification")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "dataset_id": DATASET_ID,
        "candidate": policy["candidate"],
        "posture": "sealed_for_single_quarantined_training_run_only",
        "identities": identities,
        "input_descriptors": {
            "builder": builder,
            "source_manifest": source,
            "recipe": recipe_descriptor,
            "policy": policy_descriptor,
            "build_report": report_descriptor,
            "train_dataset": outputs["train_training"],
            "train_provenance": outputs["train_provenance"],
            "tool_catalog": outputs["tool_catalog"],
            "trainer_source_tree": actual_trainer,
        },
        "base_artifact": actual_base,
        "dependency_versions": actual_dependencies,
        "run_contract": {
            "training_runs": 1,
            "seed": recipe.seed,
            "epochs": effective_epochs,
            "checkpoint_selection": selection.get("checkpoint_selection"),
            "output_dir": recipe.output_dir.resolve().relative_to(repository).as_posix(),
            "expert": recipe.experts[0].name,
            "examples": len(training_examples),
        },
        "checks": {
            "all_required_identities_bound": True,
            "base_artifact_verified": True,
            "dependencies_verified": True,
            "trainer_rows_loaded": True,
            "quarantine_preserved": True,
            "v2_test_evaluation_opened": False,
        },
        "test_access": (
            "still blocked until the candidate, evaluator, metric contract, and v1 comparator "
            "are sealed in a separate evaluation receipt"
        ),
    }
    receipt["payload_sha256"] = hashlib.sha256(canonical_json(receipt).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and seal the Taskmaster coffee API v2 training receipt."
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--experiment", type=Path, default=HERE)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    receipt = seal_training_receipt(
        repository=args.repository,
        experiment=args.experiment,
        data=args.data,
        destination=args.output,
    )
    print(f"sealed {args.output}: {sha256_file(args.output)}")
    print(f"payload: {receipt['payload_sha256']}")


if __name__ == "__main__":
    main()
