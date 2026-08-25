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
BUILDER = EXPERIMENT / "build_router_v7_taskmaster_gate_dataset.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("molor_router_v7_gate_builder_test", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_v7_gate_defaults_pin_only_v6_train_and_validation() -> None:
    builder = _load_builder()
    parameters = inspect.signature(builder.build_router_v7_gate_dataset).parameters

    assert parameters["source_prefix"].default == DATA / builder.SOURCE_DATASET_ID
    assert builder.SOURCE_SPLITS == ("train", "validation")
    assert set(builder.SOURCE_IDENTITIES) == {"train", "validation"}
    with pytest.raises(ValueError, match="unauthorized v7 gate source split"):
        builder._source_path(DATA / builder.SOURCE_DATASET_ID, "test")


def test_v7_gate_route_mapping_is_strict() -> None:
    builder = _load_builder()

    assert builder._map_route(builder.TASKMASTER_ROUTE_ID) == builder.TASKMASTER_ROUTE_ID
    for route_id in builder.SOURCE_ROUTE_IDS:
        if route_id != builder.TASKMASTER_ROUTE_ID:
            assert builder._map_route(route_id) == builder.DELEGATE_ROUTE_ID
    with pytest.raises(ValueError, match="unknown router-v6 route ID"):
        builder._map_route("unregistered-expert")


def test_v7_gate_build_is_deterministic_and_never_reads_v6_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    reads: set[Path] = set()
    original_open = Path.open

    def tracked_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any):
        if "r" in mode:
            reads.add(self.resolve())
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    first = builder.build_router_v7_gate_dataset(
        output_prefix=tmp_path / "first" / builder.OUTPUT_DATASET_ID,
    )
    second = builder.build_router_v7_gate_dataset(
        output_prefix=tmp_path / "second" / builder.OUTPUT_DATASET_ID,
    )

    assert first["schema_version"] == "molor.router-admission-gate-dataset.v1"
    assert first["route_ids"] == list(builder.ROUTE_IDS)
    assert first["normalized_prompt_overlap"] == {"train__validation": 0}
    assert first["router_test_split_read_by_builder"] is False
    assert first["taskmaster_test_split_read_by_builder"] is False
    assert first["prediction_or_evaluation_artifacts_read_by_builder"] is False
    assert first["blind_probe_read_by_builder"] is False
    assert {split: first["outputs"][split]["sha256"] for split in builder.SOURCE_SPLITS} == {
        split: second["outputs"][split]["sha256"] for split in builder.SOURCE_SPLITS
    }
    assert not any(path.name.endswith(".test.jsonl") for path in reads)
    assert not any("probe" in path.name or "prediction" in path.name for path in reads)

    for split in builder.SOURCE_SPLITS:
        rows = _read_jsonl(Path(first["outputs"][split]["path"]))
        balance = Counter(row["route_id"] for row in rows)
        assert balance == Counter(first["mapped_balance"][split])
        assert set(balance) == set(builder.ROUTE_IDS)


def test_v7_gate_source_path_rejects_escape_and_symlink(tmp_path: Path) -> None:
    builder = _load_builder()

    with pytest.raises(ValueError, match="escaped the repository data directory"):
        builder._source_path(tmp_path / builder.SOURCE_DATASET_ID, "train")
