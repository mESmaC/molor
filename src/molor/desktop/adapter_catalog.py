"""Discover locally trained adapters for the native desktop picker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AdapterCatalogEntry:
    """One loadable adapter and the useful parts of its training provenance."""

    name: str
    path: Path
    source: str
    dataset: str | None = None
    examples: int | None = None

    @property
    def picker_label(self) -> str:
        details = [self.source]
        if self.examples is not None:
            details.insert(0, f"{self.examples:,} examples")
        return f"{self.name}  ·  {' · '.join(details)}"

    def detail(self, workspace_root: Path) -> str:
        if self.dataset:
            dataset = _display_path(Path(self.dataset).expanduser(), workspace_root)
            count = f"{self.examples:,} examples · " if self.examples is not None else ""
            return f"{count}{dataset}"
        return f"No training summary · {_display_path(self.path, workspace_root)}"

    def tooltip(self, workspace_root: Path) -> str:
        lines = [f"Adapter: {self.path}", f"Source: {self.source}"]
        if self.dataset:
            lines.append(f"Dataset: {self.dataset}")
        if self.examples is not None:
            lines.append(f"Training examples: {self.examples:,}")
        return "\n".join(lines)


def discover_adapters(workspace_root: Path) -> tuple[AdapterCatalogEntry, ...]:
    """Scan checkpoint roots, with new corpus-derived experts ordered first."""

    checkpoint_root = workspace_root / "checkpoints"
    roots = (
        (checkpoint_root / "qwen3-1.7b-corpus-experts", "corpus"),
        (checkpoint_root / "qwen3-1.7b-real-experts", "earlier expert"),
        (checkpoint_root / "qwen3-1.7b-two-styles", "style demo"),
        (checkpoint_root, "local checkpoint"),
    )
    entries: list[tuple[int, AdapterCatalogEntry]] = []
    seen: set[Path] = set()
    for priority, (root, source) in enumerate(roots):
        if not root.is_dir():
            continue
        for config_path in sorted(root.rglob("adapter_config.json")):
            directory = config_path.parent.resolve()
            try:
                checkpoint_parts = directory.relative_to(checkpoint_root.resolve()).parts
            except ValueError:
                checkpoint_parts = ()
            if any(part.startswith(".") for part in checkpoint_parts):
                continue
            if directory in seen:
                continue
            seen.add(directory)
            config = _read_json_object(config_path)
            summary = _read_json_object(directory / "training_summary.json")
            name = _first_text(summary.get("expert"), config.get("name"), directory.name)
            dataset = _optional_text(summary.get("dataset"))
            raw_examples = summary.get("examples")
            examples = (
                int(raw_examples)
                if isinstance(raw_examples, int)
                and not isinstance(raw_examples, bool)
                and raw_examples >= 0
                else None
            )
            entries.append(
                (
                    priority,
                    AdapterCatalogEntry(
                        name=name,
                        path=directory,
                        source=source,
                        dataset=dataset,
                        examples=examples,
                    ),
                )
            )
    entries.sort(key=lambda item: (item[0], item[1].name.casefold(), str(item[1].path)))
    return tuple(entry for _priority, entry in entries)


def path_key(raw_path: str | Path, workspace_root: Path) -> str:
    """Normalize a path for catalog-to-manual-field matching."""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return str(path.resolve()).casefold()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_text(*values: object) -> str:
    return next((value.strip() for value in values if isinstance(value, str) and value.strip()))


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except (OSError, ValueError):
        return str(path)
