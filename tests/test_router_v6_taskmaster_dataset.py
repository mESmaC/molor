from __future__ import annotations

from collections import Counter
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "first-real-experts"
DATA = EXPERIMENT / "data"
BUILDER = EXPERIMENT / "build_router_v6_taskmaster_dataset.py"


def _load_builder():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location("molor_router_v6_builder_test", BUILDER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_v6_defaults_pin_promoted_v4_and_reject_taskmaster_test_split() -> None:
    builder = _load_builder()
    parameters = inspect.signature(builder.build_router_v6_dataset).parameters

    assert parameters["v4_prefix"].default == (DATA / "router-bootstrap-v4-hard-negative-s307")
    assert builder.PINNED_V4_REPORT_SHA256 == (
        "b566cde269bbc4e9a2886685de5844a99dff8c6b343355800355dd72513bbddb"
    )
    assert builder.TASKMASTER_SOURCE_SPLITS == ("train", "validation")
    with pytest.raises(ValueError, match="unauthorized Taskmaster source split"):
        builder._authorized_taskmaster_path(
            DATA / builder.TASKMASTER_DATASET_ID,
            split="test",
            provenance=False,
        )


def test_v6_build_reads_only_allowlisted_taskmaster_sources_and_is_balanced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    read_paths: set[Path] = set()
    original_open = Path.open

    def tracked_open(
        self: Path,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ):
        if "r" in mode:
            read_paths.add(self.resolve())
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    report = builder.build_router_v6_dataset(
        output_prefix=tmp_path / "router-bootstrap-v6-taskmaster",
    )

    assert report["schema_version"] == "molor.router-bootstrap-dataset.v6"
    assert report["route_ids"] == list(builder.ROUTE_IDS)
    assert report["evaluation_artifacts_read_by_builder"] is False
    assert report["taskmaster_test_split_read_by_builder"] is False
    assert report["taskmaster_prediction_artifacts_read_by_builder"] is False
    assert report["authorized_taskmaster_source_splits"] == ["train", "validation"]
    assert report["balance"] == {
        split: {route_id: builder.TARGETS[split] for route_id in builder.ROUTE_IDS}
        for split in builder.SPLITS
    }
    assert not any(report["normalized_prompt_overlap"].values())
    assert not any(report["semantic_group_overlap"].values())
    assert not any(report["taskmaster_source"]["selected_component_overlap"].values())
    assert report["taskmaster_source"]["selected_rows"] == builder.TARGETS
    assert report["taskmaster_source"]["source_split_mapping"] == {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    assert report["taskmaster_source"]["routing_normalization"] == {
        "train": {
            "source_rows": 11626,
            "routing_rows": 11589,
            "normalized_prompt_duplicates_collapsed": 37,
            "source_components": 2608,
            "routing_components": 2574,
            "routing_component_unions": 33,
        },
        "validation": {
            "source_rows": 1380,
            "routing_rows": 1380,
            "normalized_prompt_duplicates_collapsed": 0,
            "source_components": 324,
            "routing_components": 324,
            "routing_component_unions": 0,
        },
    }
    taskmaster_source = report["taskmaster_source"]
    assert taskmaster_source["available_rows"] == {"train": 11626, "validation": 1380}
    assert taskmaster_source["available_routing_rows"] == {
        "train": 11589,
        "validation": 1380,
    }
    assert taskmaster_source["routing_component_unions"]["validation"] == {}
    assert all(
        group_id.startswith("tm4-coffee-v2-component:")
        for split in ("validation", "test")
        for group_id in taskmaster_source["selected_component_ids"][split]
    )

    allowed_taskmaster_inputs = {
        f"{builder.TASKMASTER_DATASET_ID}.train.jsonl",
        f"{builder.TASKMASTER_DATASET_ID}.train.provenance.jsonl",
        f"{builder.TASKMASTER_DATASET_ID}.validation.jsonl",
        f"{builder.TASKMASTER_DATASET_ID}.validation.provenance.jsonl",
    }
    taskmaster_reads = {
        path.name for path in read_paths if path.name.startswith(builder.TASKMASTER_DATASET_ID)
    }
    assert taskmaster_reads == allowed_taskmaster_inputs


def test_v6_preserves_every_v4_row_and_builds_deterministically(tmp_path: Path) -> None:
    builder = _load_builder()
    first = builder.build_router_v6_dataset(
        output_prefix=tmp_path / "first" / "router-bootstrap-v6-taskmaster",
    )
    second = builder.build_router_v6_dataset(
        output_prefix=tmp_path / "second" / "router-bootstrap-v6-taskmaster",
    )

    assert {split: first["outputs"][split]["sha256"] for split in builder.SPLITS} == {
        split: second["outputs"][split]["sha256"] for split in builder.SPLITS
    }
    for split in builder.SPLITS:
        legacy_path = DATA / f"router-bootstrap-v4-hard-negative-s307.{split}.jsonl"
        output_path = Path(first["outputs"][split]["path"])
        legacy = Counter((row["prompt"], row["route_id"]) for row in _read_jsonl(legacy_path))
        output_rows = _read_jsonl(output_path)
        output_legacy = Counter(
            (row["prompt"], row["route_id"])
            for row in output_rows
            if row["route_id"] != builder.TASKMASTER_ROUTE_ID
        )
        taskmaster_rows = [
            row for row in output_rows if row["route_id"] == builder.TASKMASTER_ROUTE_ID
        ]

        assert output_legacy == legacy
        assert len(taskmaster_rows) == builder.TARGETS[split]
        assert all(
            row["prompt"].startswith(builder.TASKMASTER_PROMPT_PREFIX) for row in taskmaster_rows
        )
        assert first["legacy_v4_preservation"][split] == {
            "rows": len(legacy),
            "exact_row_multiset_preserved": True,
        }
