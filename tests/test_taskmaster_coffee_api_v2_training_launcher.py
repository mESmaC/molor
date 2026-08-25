from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "first-real-experts"
SCRIPT = EXPERIMENT / "run_taskmaster_coffee_api_v2_training.py"


def _load_launcher(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(EXPERIMENT))
    spec = importlib.util.spec_from_file_location("taskmaster_v2_training_launcher_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_recipe(experiment: Path, *, model_id: str) -> Path:
    path = experiment / "taskmaster-coffee-api-v2-recipe.json"
    value = {
        "model_id": model_id,
        "output_dir": "../../checkpoints/qwen3-1.7b-candidate-experts-quarantine",
        "target_modules": ["q_proj"],
        "epochs": 1,
        "seed": 241104,
        "experts": [
            {
                "name": "taskmaster-coffee-api-v2-full-dialog-r16",
                "dataset": "data/taskmaster-coffee-api-v2.train.jsonl",
            }
        ],
    }
    _write(path, json.dumps(value) + "\n")
    return path


def _fake_sealer(destination: Path):
    def seal(**kwargs):
        path = kwargs["destination"]
        recipe = kwargs["experiment"] / "taskmaster-coffee-api-v2-recipe.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"sealed":true}\n', encoding="utf-8")
        return {
            "payload_sha256": "a" * 64,
            "candidate": "taskmaster-coffee-api-v2-full-dialog-r16",
            "identities": {
                "recipe_sha256": (
                    hashlib.sha256(recipe.read_bytes()).hexdigest()
                    if recipe.is_file()
                    else "c" * 64
                ),
                "train_dataset_sha256": "b" * 64,
            },
        }

    return seal


def test_workspace_code_verifier_accepts_only_this_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    launcher._require_workspace_code(ROOT)
    with pytest.raises(RuntimeError, match="sealed workspace"):
        launcher._require_workspace_code(tmp_path / "different-repository")


def test_preflight_only_seals_without_loading_recipe_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    repository = tmp_path / "repo"
    experiment = repository / "experiments" / "first-real-experts"
    data = experiment / "data"
    receipt = experiment / "receipt.json"
    monkeypatch.setattr(launcher, "seal_training_receipt", _fake_sealer(receipt))
    monkeypatch.setattr(
        launcher.MixtureEngine,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("preflight loaded the model"),
    )

    result = launcher.launch_training(
        repository=repository,
        experiment=experiment,
        data=data,
        receipt_path=receipt,
        preflight_only=True,
        workspace_code_verifier=lambda _repository: None,
    )

    assert result["status"] == "preflight_only"
    assert receipt.is_file()


def test_launcher_refuses_to_overwrite_an_existing_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    repository = tmp_path / "repo"
    experiment = repository / "experiments" / "first-real-experts"
    data = experiment / "data"
    receipt = experiment / "receipt.json"
    _write_recipe(experiment, model_id="../../models/Qwen3-1.7B")
    candidate = (
        repository
        / "checkpoints"
        / "qwen3-1.7b-candidate-experts-quarantine"
        / "taskmaster-coffee-api-v2-full-dialog-r16"
    )
    candidate.mkdir(parents=True)
    monkeypatch.setattr(launcher, "seal_training_receipt", _fake_sealer(receipt))
    monkeypatch.setattr(
        launcher.MixtureEngine,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("overwrite guard loaded the model"),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        launcher.launch_training(
            repository=repository,
            experiment=experiment,
            data=data,
            receipt_path=receipt,
            workspace_code_verifier=lambda _repository: None,
        )


def test_launcher_refuses_a_second_attempt_after_a_recovery_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    repository = tmp_path / "repo"
    experiment = repository / "experiments" / "first-real-experts"
    data = experiment / "data"
    receipt = experiment / "receipt.json"
    _write_recipe(experiment, model_id="../../models/Qwen3-1.7B")
    recovery = (
        repository
        / "checkpoints"
        / "qwen3-1.7b-candidate-experts-quarantine"
        / ".checkpoints"
        / "taskmaster-coffee-api-v2-full-dialog-r16-step-000100"
    )
    recovery.mkdir(parents=True)
    monkeypatch.setattr(launcher, "seal_training_receipt", _fake_sealer(receipt))
    monkeypatch.setattr(
        launcher.MixtureEngine,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("recovery guard loaded the model"),
    )

    with pytest.raises(FileExistsError, match="second attempt"):
        launcher.launch_training(
            repository=repository,
            experiment=experiment,
            data=data,
            receipt_path=receipt,
            workspace_code_verifier=lambda _repository: None,
        )


def test_launcher_rejects_recipe_drift_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    repository = tmp_path / "repo"
    experiment = repository / "experiments" / "first-real-experts"
    data = experiment / "data"
    receipt = experiment / "receipt.json"
    _write_recipe(experiment, model_id="../../models/Qwen3-1.7B")
    sealer = _fake_sealer(receipt)

    def stale_sealer(**kwargs):
        value = sealer(**kwargs)
        value["identities"]["recipe_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(launcher, "seal_training_receipt", stale_sealer)
    monkeypatch.setattr(
        launcher.MixtureEngine,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("recipe guard loaded the model"),
    )

    with pytest.raises(ValueError, match="recipe changed"):
        launcher.launch_training(
            repository=repository,
            experiment=experiment,
            data=data,
            receipt_path=receipt,
            workspace_code_verifier=lambda _repository: None,
        )


def test_launcher_rejects_a_model_path_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    repository = tmp_path / "repo"
    experiment = repository / "experiments" / "first-real-experts"
    data = experiment / "data"
    receipt = experiment / "receipt.json"
    _write_recipe(experiment, model_id="../../../../repo-evil/models/Qwen3-1.7B")
    monkeypatch.setattr(launcher, "seal_training_receipt", _fake_sealer(receipt))
    monkeypatch.setattr(
        launcher.MixtureEngine,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("path guard loaded the model"),
    )

    with pytest.raises(ValueError, match="policy-bound local base"):
        launcher.launch_training(
            repository=repository,
            experiment=experiment,
            data=data,
            receipt_path=receipt,
            workspace_code_verifier=lambda _repository: None,
        )


def test_completion_receipt_binds_outputs_and_rejects_dataset_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher(monkeypatch)
    repository = tmp_path / "repo"
    candidate = (
        repository
        / "checkpoints"
        / "qwen3-1.7b-candidate-experts-quarantine"
        / "taskmaster-coffee-api-v2-full-dialog-r16"
    )
    preflight = repository / "experiments" / "first-real-experts" / "receipt.json"
    _write(preflight, '{"sealed":true}\n')
    _write(candidate / "adapter_config.json", "{}\n")
    _write(candidate / "adapter_model.safetensors", "weights")
    _write(candidate / "training_summary.json", json.dumps({"dataset_sha256": "b" * 64}))

    completion = launcher._write_completion_receipt(
        repository=repository,
        candidate_directory=candidate,
        preflight_path=preflight,
        preflight_payload_sha256="a" * 64,
        preflight_file_sha256=launcher.sha256_file(preflight),
        train_dataset_sha256="b" * 64,
        final_loss=0.25,
    )
    payload = json.loads(completion.read_text(encoding="utf-8"))
    assert b"\r\n" not in completion.read_bytes()
    assert payload["status"] == "completed_quarantined_candidate_not_promotion_eligible"
    assert {row["path"].split("/")[-1] for row in payload["artifacts"]} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_summary.json",
    }

    _write(candidate / "training_summary.json", json.dumps({"dataset_sha256": "c" * 64}))
    completion.unlink()
    with pytest.raises(ValueError, match="dataset hash"):
        launcher._write_completion_receipt(
            repository=repository,
            candidate_directory=candidate,
            preflight_path=preflight,
            preflight_payload_sha256="a" * 64,
            preflight_file_sha256=launcher.sha256_file(preflight),
            train_dataset_sha256="b" * 64,
            final_loss=0.25,
        )
    assert not completion.exists()
