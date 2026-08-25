from __future__ import annotations

import json
from pathlib import Path

from molor.desktop.adapter_catalog import discover_adapters


def _write_adapter(directory: Path, name: str) -> None:
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"name": name}),
        encoding="utf-8",
    )
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
