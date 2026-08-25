from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "first-real-experts" / "seal_taskmaster_coffee_api_v2_training.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("taskmaster_v2_training_receipt_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _descriptor(path: Path, relative: str, *, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _fixture(module, tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "repo"
    experiment = repository / "experiments" / "first-real-experts"
    data = experiment / "data"
    model = repository / "models" / "Qwen3-1.7B"
    model.mkdir(parents=True)
    candidate = "taskmaster-coffee-api-v2-full-dialog-r16"
    dependencies = {
        "python": "3.12.10",
        "torch": "2.11.0+cu128",
        "transformers": "5.15.1",
        "peft": "0.20.0",
        "accelerate": "1.14.0",
        "safetensors": "0.8.0",
    }
    base_files = [{"name": "config.json", "size_bytes": 7, "sha256": "b" * 64}]
    base_fingerprint = hashlib.sha256(
        module.canonical_json({"role": "base", "files": base_files}).encode()
    ).hexdigest()
    base_lock = {
        "role": "base",
        "content_fingerprint": base_fingerprint,
        "size_bytes": 7,
        "files": base_files,
        "complete": True,
    }
    trainer_files = [{"name": "src/molor/training/sft.py", "size_bytes": 9, "sha256": "c" * 64}]
    trainer_tree = {
        "hash_policy": "sha256-full-read-canonical-relative-path-tree-v1",
        "content_fingerprint": hashlib.sha256(
            module.canonical_json({"files": trainer_files}).encode()
        ).hexdigest(),
        "files": trainer_files,
    }

    builder = experiment / module.BUILDER.name
    source = experiment / module.SOURCE_MANIFEST.name
    recipe = experiment / module.RECIPE.name
    policy = experiment / module.POLICY.name
    report = data / module.BUILD_REPORT.name
    builder.parent.mkdir(parents=True)
    builder.write_text("# deterministic builder\n", encoding="utf-8")
    _write_json(source, {"schema_version": "fixture"})
    _write_json(
        recipe,
        {
            "model_id": "../../models/Qwen3-1.7B",
            "output_dir": "../../checkpoints/qwen3-1.7b-candidate-experts-quarantine",
            "dtype": "bfloat16",
            "device": "cuda",
            "target_modules": ["q_proj"],
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "max_length": 2048,
            "epochs": 1,
            "learning_rate": 0.00005,
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
            "checkpoint_every_steps": 100,
            "max_grad_norm": 1.0,
            "seed": 241104,
            "chat_template_kwargs": {"enable_thinking": False},
            "experts": [
                {
                    "name": candidate,
                    "dataset": "data/taskmaster-coffee-api-v2.train.jsonl",
                }
            ],
        },
    )
    pins = {
        "builder_sha256": _sha256(builder),
        "source_manifest_sha256": _sha256(source),
        "recipe_sha256": _sha256(recipe),
    }
    required = [
        "builder_sha256",
        "source_manifest_sha256",
        "recipe_sha256",
        "policy_sha256",
        "build_report_sha256",
        "train_dataset_sha256",
        "train_provenance_sha256",
        "tool_catalog_sha256",
        "trainer_source_tree_sha256",
    ]
    _write_json(
        policy,
        {
            "candidate": candidate,
            "candidate_posture": "quarantine_until_every_gate_passes",
            "selection": {
                "training_runs": 1,
                "seed": 241104,
                "epochs": 1,
                "checkpoint_selection": "final checkpoint",
            },
            "dataset_contract": {"artifact_identity_inputs": pins},
            "training_artifact_contract": {
                "base_artifact": {
                    "path": "models/Qwen3-1.7B",
                    "content_fingerprint": base_fingerprint,
                },
                "trainer_source_contract": {
                    "content_fingerprint": trainer_tree["content_fingerprint"]
                },
                "required_file_identities": required,
                "required_dependency_versions": dependencies,
            },
        },
    )

    train = data / "taskmaster-coffee-api-v2.train.jsonl"
    provenance = data / "taskmaster-coffee-api-v2.train.provenance.jsonl"
    catalog = data / "taskmaster-coffee-api-v2.tool-catalog.json"
    test_evaluation = data / "taskmaster-coffee-api-v2.test.evaluation.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "context"},
                {"role": "assistant", "content": '{"api":{"show_menu":{}}}'},
            ]
        }
    ]
    _write_jsonl(train, rows)
    _write_jsonl(provenance, [{"example_id": "fixture"}])
    _write_json(catalog, {"tools": [{"name": "show_menu"}]})
    _write_jsonl(test_evaluation, [{"sentinel": "must-not-open"}])
    outputs = {
        "train_training": _descriptor(train, train.name, rows=1),
        "train_provenance": _descriptor(provenance, provenance.name, rows=1),
        "tool_catalog": _descriptor(catalog, catalog.name),
        "test_evaluation": _descriptor(test_evaluation, test_evaluation.name, rows=1),
    }
    _write_json(
        report,
        {
            "dataset_id": module.DATASET_ID,
            "intended_checkpoint_posture": "candidate_quarantine_only_never_auto_promote",
            "build_inputs": {
                "builder": _descriptor(builder, builder.name),
                "source_manifest": _descriptor(source, source.name),
                "recipe": _descriptor(recipe, recipe.name),
                "policy": _descriptor(policy, policy.name),
            },
            "outputs": outputs,
        },
    )
    return {
        "repository": repository,
        "experiment": experiment,
        "data": data,
        "destination": experiment / "receipt.json",
        "dependencies": dependencies,
        "base_lock": base_lock,
        "trainer_tree": trainer_tree,
        "train": train,
        "policy": policy,
        "test_evaluation": test_evaluation,
    }


def _seal(module, fixture: dict[str, object]):
    return module.seal_training_receipt(
        repository=fixture["repository"],
        experiment=fixture["experiment"],
        data=fixture["data"],
        destination=fixture["destination"],
        dependency_versions=fixture["dependencies"],
        base_artifact_locker=lambda _path: fixture["base_lock"],
        trainer_source_locker=lambda _repository, _experiment: fixture["trainer_tree"],
    )


def _repin_recipe_and_policy(module, fixture: dict[str, object]) -> None:
    experiment = fixture["experiment"]
    data = fixture["data"]
    recipe = experiment / module.RECIPE.name
    policy_path = fixture["policy"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["dataset_contract"]["artifact_identity_inputs"]["recipe_sha256"] = _sha256(recipe)
    _write_json(policy_path, policy)
    report_path = data / module.BUILD_REPORT.name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["build_inputs"]["recipe"] = _descriptor(recipe, recipe.name)
    report["build_inputs"]["policy"] = _descriptor(policy_path, policy_path.name)
    _write_json(report_path, report)


def test_receipt_is_deterministic_complete_and_does_not_need_test_data(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    first = _seal(module, fixture)
    first_bytes = fixture["destination"].read_bytes()
    second = _seal(module, fixture)

    assert first == second
    assert fixture["destination"].read_bytes() == first_bytes
    assert b"\r\n" not in first_bytes
    assert set(first["identities"]) == {
        "builder_sha256",
        "source_manifest_sha256",
        "recipe_sha256",
        "policy_sha256",
        "build_report_sha256",
        "train_dataset_sha256",
        "train_provenance_sha256",
        "tool_catalog_sha256",
        "trainer_source_tree_sha256",
    }
    assert first["run_contract"]["examples"] == 1
    assert first["checks"]["trainer_rows_loaded"] is True
    assert first["checks"]["v2_test_evaluation_opened"] is False
    without_payload = dict(first)
    payload_hash = without_payload.pop("payload_sha256")
    assert (
        hashlib.sha256(module.canonical_json(without_payload).encode()).hexdigest() == payload_hash
    )


def test_receipt_never_opens_the_v2_test_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == fixture["test_evaluation"].resolve():
            raise AssertionError("training preflight opened the v2 test shard")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    receipt = _seal(module, fixture)
    assert receipt["checks"]["v2_test_evaluation_opened"] is False


@pytest.mark.parametrize("mutation", ["dataset", "policy_pin", "dependency", "base", "trainer"])
def test_receipt_hard_fails_stale_or_mismatched_inputs(tmp_path: Path, mutation: str) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    if mutation == "dataset":
        fixture["train"].write_text("{}\n", encoding="utf-8")
    elif mutation == "policy_pin":
        policy = json.loads(fixture["policy"].read_text(encoding="utf-8"))
        policy["dataset_contract"]["artifact_identity_inputs"]["builder_sha256"] = "f" * 64
        _write_json(fixture["policy"], policy)
    elif mutation == "dependency":
        fixture["dependencies"] = {**fixture["dependencies"], "torch": "wrong"}
    elif mutation == "base":
        fixture["base_lock"] = {**fixture["base_lock"], "content_fingerprint": "f" * 64}
    else:
        fixture["trainer_tree"] = {
            **fixture["trainer_tree"],
            "content_fingerprint": "f" * 64,
        }

    with pytest.raises((ValueError, FileNotFoundError)):
        _seal(module, fixture)


def test_receipt_rejects_a_recipe_the_real_loader_cannot_consume(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    recipe = fixture["experiment"] / module.RECIPE.name
    raw = json.loads(recipe.read_text(encoding="utf-8"))
    raw["unbound_metadata"] = "not accepted by TrainingRecipe"
    _write_json(recipe, raw)

    with pytest.raises(TypeError, match="unbound_metadata"):
        _seal(module, fixture)


@pytest.mark.parametrize(
    "mutation",
    ["nested_expert_key", "wrong_quarantine_root", "warm_start", "per_expert_epoch"],
)
def test_receipt_rejects_unbound_or_discoverable_recipe_state(
    tmp_path: Path, mutation: str
) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    recipe_path = fixture["experiment"] / module.RECIPE.name
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if mutation == "nested_expert_key":
        recipe["experts"][0]["ignored_metadata"] = True
    elif mutation == "wrong_quarantine_root":
        recipe["output_dir"] = "../../checkpoints/quarantine-candidate"
    elif mutation == "warm_start":
        recipe["experts"][0]["initial_adapter"] = "../../checkpoints/old-adapter"
    else:
        recipe["experts"][0]["epochs"] = 2
    _write_json(recipe_path, recipe)
    _repin_recipe_and_policy(module, fixture)

    with pytest.raises(
        ValueError, match="exactly name and dataset|approved quarantine|seed/epochs"
    ):
        _seal(module, fixture)


def test_receipt_rejects_escaped_roots_and_incomplete_injected_locks(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        module._safe_repository_path(
            fixture["repository"], "../outside-model", label="policy base artifact path"
        )

    fixture["base_lock"] = {"content_fingerprint": "a" * 64}
    with pytest.raises(ValueError, match="role"):
        _seal(module, fixture)


def test_receipt_rejects_external_data_and_experiment_roots(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    with pytest.raises(ValueError, match="data root"):
        module.seal_training_receipt(
            repository=fixture["repository"],
            experiment=fixture["experiment"],
            data=tmp_path / "outside-data",
            destination=fixture["destination"],
            dependency_versions=fixture["dependencies"],
            base_artifact_locker=lambda _path: fixture["base_lock"],
            trainer_source_locker=lambda _repository, _experiment: fixture["trainer_tree"],
        )


def test_receipt_rejects_report_output_path_traversal(tmp_path: Path) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    report_path = fixture["data"] / module.BUILD_REPORT.name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outputs"]["train_training"]["path"] = "../data/taskmaster-coffee-api-v2.train.jsonl"
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="escapes"):
        _seal(module, fixture)


def test_receipt_rejects_policy_base_path_escape_even_when_report_is_repinned(
    tmp_path: Path,
) -> None:
    module = _load_script()
    fixture = _fixture(module, tmp_path)
    policy = json.loads(fixture["policy"].read_text(encoding="utf-8"))
    policy["training_artifact_contract"]["base_artifact"]["path"] = "../outside-model"
    _write_json(fixture["policy"], policy)
    report_path = fixture["data"] / module.BUILD_REPORT.name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["build_inputs"]["policy"] = _descriptor(fixture["policy"], fixture["policy"].name)
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="escapes"):
        _seal(module, fixture)
