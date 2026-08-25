from __future__ import annotations

import json
from pathlib import Path

from molor.desktop.adapter_catalog import discover_adapters


def _write_adapter(directory: Path, name: str) -> None:
    directory.mkdir(parents=True)
    lane = "Qwen3-0.6B" if "qwen3-0.6b" in str(directory) else "Qwen3-1.7B"
    (directory / "adapter_config.json").write_text(
        json.dumps({"name": name, "base_model_id": f"models/{lane}"}),
        encoding="utf-8",
    )
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    (directory / "training_summary.json").write_text(
        json.dumps({"expert": name, "examples": 3}),
        encoding="utf-8",
    )


def test_discovery_excludes_explicit_quarantine_roots(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    _write_adapter(
        checkpoints / "qwen3-1.7b-corpus-experts" / "promoted-expert",
        "promoted-expert",
    )
    _write_adapter(
        checkpoints / "qwen3-1.7b-candidate-experts-quarantine" / "rejected-candidate",
        "rejected-candidate",
    )
    _write_adapter(
        checkpoints / "Quarantine" / "another-rejected-candidate",
        "another-rejected-candidate",
    )

    entries = discover_adapters(tmp_path)

    assert [entry.name for entry in entries] == ["promoted-expert"]


def test_duplicate_expert_names_are_labeled_and_ordered_by_model_lane(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    _write_adapter(
        checkpoints / "qwen3-0.6b-corpus-experts" / "shared-name",
        "shared-name",
    )
    _write_adapter(
        checkpoints / "qwen3-1.7b-corpus-experts" / "shared-name",
        "shared-name",
    )

    entries = discover_adapters(tmp_path)

    assert [entry.source for entry in entries] == ["0.6B development corpus", "corpus"]
    assert [Path(entry.base_model_id).name for entry in entries] == [
        "Qwen3-0.6B",
        "Qwen3-1.7B",
    ]
    assert entries[0].picker_label != entries[1].picker_label


def test_incomplete_adapter_is_not_added_to_the_catalog(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints" / "qwen3-0.6b-corpus-experts" / "interrupted"
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"name": "interrupted", "base_model_id": "models/Qwen3-0.6B"}),
        encoding="utf-8",
    )

    assert discover_adapters(tmp_path) == ()
