from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "first-real-experts"
LAUNCHER = EXPERIMENT / "train_router_v7_incremental_gate.py"


def _load_launcher():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location(
            "molor_router_v7_gate_training_test", LAUNCHER
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


def test_v7_gate_training_contract_is_linear_cpu_and_train_validation_only() -> None:
    launcher = _load_launcher()

    assert launcher.ALLOWED_SEEDS == (83, 8419, 27183)
    assert set(launcher.DATASET_IDENTITIES) == {"train", "validation", "report"}
    assert set(launcher.MODEL_IDENTITIES) == {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }
    assert launcher.ROUTE_IDS == (
        "taskmaster-coffee-api-v2-full-dialog-r16",
        "base/null",
    )
    for seed in launcher.ALLOWED_SEEDS:
        config = launcher.frozen_training_config(seed)
        assert config.device == "cpu"
        assert config.router_hidden_size is None
        assert config.seed == seed


def test_v7_gate_rejects_unknown_seed() -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="seed must be one of"):
        launcher.candidate_output(7)
    with pytest.raises(ValueError, match="seed must be one of"):
        launcher.frozen_training_config(7)


def test_v7_gate_train_passes_no_test_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    seed = launcher.ALLOWED_SEEDS[0]
    destination = tmp_path / "candidate"
    captured: dict[str, object] = {}

    monkeypatch.setattr(launcher, "candidate_output", lambda _seed: destination)
    monkeypatch.setattr(launcher, "preflight", lambda **_kwargs: {})

    def fake_train_router_checkpoint(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            checkpoint_path=destination / "router.safetensors",
            config_path=destination / "router_config.json",
            metrics={"validation": {"accuracy": 1.0}},
        )

    monkeypatch.setattr(launcher, "train_router_checkpoint", fake_train_router_checkpoint)
    launcher.train(seed=seed)

    assert captured["test_path"] is None
    assert captured["route_ids"] == launcher.ROUTE_IDS
    assert captured["adapters"] == {launcher.TASKMASTER_ROUTE_ID: launcher.ADAPTER_DIR}
    assert captured["training"].router_hidden_size is None
    assert captured["training"].device == "cpu"
