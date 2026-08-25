from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "first-real-experts"
BUILDER = EXPERIMENT / "build_router_v4_hard_negative_dataset.py"


def _load_builder():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location("molor_router_v4_builder_test", BUILDER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


def test_v4_authored_rows_are_unique_and_family_split_disjoint() -> None:
    builder = _load_builder()
    rows = builder._v4_authored_rows()

    normalized = [row.normalized_prompt for split in builder.SPLITS for row in rows[split]]
    assert len(normalized) == len(set(normalized))

    owners: dict[tuple[str, str], str] = {}
    for split in builder.SPLITS:
        for row in rows[split]:
            assert row.source == "manual-hard-negative-intent-v4"
            assert row.route_id in builder.ROUTE_IDS
            assert row.family is not None
            key = (row.route_id, row.family)
            assert owners.setdefault(key, split) == split

    counts = builder._route_counts(rows)
    assert counts == {
        "train": {
            "ocb-can-verified-v1": 96,
            "embedded-knowledge-v1": 16,
            "base/null": 16,
        },
        "validation": {
            "ocb-can-verified-v1": 24,
            "embedded-knowledge-v1": 4,
            "base/null": 4,
        },
        "test": {
            "ocb-can-verified-v1": 24,
            "embedded-knowledge-v1": 4,
            "base/null": 4,
        },
    }


def test_v4_authored_rows_do_not_copy_exact_probe_prompts() -> None:
    builder = _load_builder()
    rows = builder._v4_authored_rows()
    authored = {
        row.normalized_prompt for split in builder.SPLITS for row in rows[split]
    }
    probe_prompts: set[str] = set()
    for name in ("router-intent-probe-v1.jsonl", "router-intent-probe-v2.jsonl"):
        with (EXPERIMENT / "data" / name).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    probe_prompts.add(builder._normalize_prompt(json.loads(line)["prompt"]))

    assert authored.isdisjoint(probe_prompts)


def test_v4_build_preserves_balance_and_zero_overlap(tmp_path: Path) -> None:
    builder = _load_builder()
    report = builder.build_router_v4_dataset(
        v1_prefix=EXPERIMENT / "data" / "router-bootstrap-v1",
        output_prefix=tmp_path / "router-bootstrap-v4-hard-negative",
    )

    assert report["schema_version"] == "molor.router-bootstrap-dataset.v4"
    assert report["evaluation_artifacts_read_by_builder"] is False
    assert report["exact_probe_rows_used"] is False
    assert report["probe_gap_analysis_informed_curriculum"] is True
    assert report["probe_status_for_v4"] == "development_only_not_blind_promotion_evidence"
    assert not any(report["normalized_prompt_overlap"].values())
    assert not any(report["semantic_group_overlap"].values())
    assert report["balance"] == {
        "train": {route_id: 936 for route_id in builder.ROUTE_IDS},
        "validation": {route_id: 130 for route_id in builder.ROUTE_IDS},
        "test": {route_id: 130 for route_id in builder.ROUTE_IDS},
    }
    for output in report["outputs"].values():
        assert Path(output["path"]).is_file()
