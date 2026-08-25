from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "first-real-experts"
DATA = EXPERIMENT / "data"
BUILDER = EXPERIMENT / "build_router_v5_verified_code_contrastive_dataset.py"
V4_PREFIX = DATA / "router-bootstrap-v4-hard-negative-s307"


def _load_builder():
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location("molor_router_v5_builder_test", BUILDER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_v5_authored_rows_are_unique_split_owned_code_intents() -> None:
    builder = _load_builder()
    rows = builder._v5_authored_rows()

    normalized = [row.normalized_prompt for split in builder.SPLITS for row in rows[split]]
    assert len(normalized) == len(set(normalized))
    assert {split: len(rows[split]) for split in builder.SPLITS} == {
        "train": 96,
        "validation": 24,
        "test": 24,
    }

    owners: dict[str, str] = {}
    expected_families = {"train": 8, "validation": 2, "test": 2}
    for split in builder.SPLITS:
        families: set[str] = set()
        for row in rows[split]:
            assert row.source == "manual-embedded-code-intent-v5"
            assert row.route_id == builder.CODE_ROUTE_ID
            assert row.family is not None
            families.add(row.family)
            assert owners.setdefault(row.family, split) == split
        assert len(families) == expected_families[split]


def test_v5_authored_rows_do_not_copy_any_known_probe_prompt() -> None:
    builder = _load_builder()
    rows = builder._v5_authored_rows()
    authored = {row.normalized_prompt for split in builder.SPLITS for row in rows[split]}

    probes = (
        DATA / "router-intent-probe-v1.jsonl",
        DATA / "router-intent-probe-v2.jsonl",
        EXPERIMENT / "router-blind-probe-v3.jsonl",
        EXPERIMENT / "router-blind-probe-v4.jsonl",
    )
    probe_prompts: set[str] = set()
    for path in probes:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    probe_prompts.add(builder._normalize_prompt(json.loads(line)["prompt"]))

    assert authored.isdisjoint(probe_prompts)


def test_v5_build_changes_only_code_rows_in_place(tmp_path: Path) -> None:
    builder = _load_builder()
    output_prefix = tmp_path / "router-bootstrap-v5-verified-code-contrastive"
    report = builder.build_router_v5_dataset(
        v4_prefix=V4_PREFIX,
        output_prefix=output_prefix,
    )

    assert report["schema_version"] == "molor.router-bootstrap-dataset.v5"
    assert report["evaluation_artifacts_read_by_builder"] is False
    assert report["exact_probe_rows_used"] is False
    assert report["blind_gap_analysis_informed_curriculum"] is True
    assert report["probe_status_for_v5"] == ("development_only_after_gap_informed_curriculum")
    assert not any(report["normalized_prompt_overlap"].values())
    assert not any(report["semantic_group_overlap"].values())
    assert report["balance"] == {
        "train": {route_id: 936 for route_id in builder.ROUTE_IDS},
        "validation": {route_id: 130 for route_id in builder.ROUTE_IDS},
        "test": {route_id: 130 for route_id in builder.ROUTE_IDS},
    }

    expected_replacements = {
        "train": {"rows": 96, "groups": 48, "unchanged": 4584},
        "validation": {"rows": 24, "groups": 12, "unchanged": 626},
        "test": {"rows": 24, "groups": 12, "unchanged": 626},
    }
    for split in builder.SPLITS:
        details = report["replacement"][split]
        expected = expected_replacements[split]
        assert details["removed_rows"] == expected["rows"]
        assert details["added_rows"] == expected["rows"]
        assert details["removed_generated_semantic_groups"] == expected["groups"]
        assert details["unchanged_rows"] == expected["unchanged"]
        assert details["all_replacements_preserve_route"] is True

        before = _load_jsonl(builder._split_path(V4_PREFIX, split))
        after = _load_jsonl(builder._split_path(output_prefix, split))
        assert len(before) == len(after)
        changed = 0
        for old_row, new_row in zip(before, after, strict=True):
            if old_row == new_row:
                continue
            changed += 1
            assert old_row["route_id"] == builder.CODE_ROUTE_ID
            assert new_row["route_id"] == builder.CODE_ROUTE_ID
        assert changed == expected["rows"]


def test_v5_build_is_deterministic(tmp_path: Path) -> None:
    builder = _load_builder()
    first = builder.build_router_v5_dataset(
        v4_prefix=V4_PREFIX,
        output_prefix=tmp_path / "first",
    )
    second = builder.build_router_v5_dataset(
        v4_prefix=V4_PREFIX,
        output_prefix=tmp_path / "second",
    )

    assert {split: first["outputs"][split]["sha256"] for split in builder.SPLITS} == {
        split: second["outputs"][split]["sha256"] for split in builder.SPLITS
    }
    assert first["replacement"] == second["replacement"]


def test_v5_promotion_policy_pins_probe_and_forbids_reuse() -> None:
    policy = json.loads(
        (EXPERIMENT / "router-v5-evaluation-policy.json").read_text(encoding="utf-8")
    )

    assert policy["parent_checkpoint"].endswith("router-v4-mlp-controlled")
    assert policy["candidate_checkpoint"].endswith("router-v5-mlp-seed83")
    assert policy["blind_probe"] == {
        "path": "router-blind-probe-v4.jsonl",
        "sha256": "03d3b671ac831596a3cfff2c154e256b88c57731c04d37c59e0910912235bce2",
        "prompts": 72,
        "confidence_threshold": 0.3,
        "top_k": 1,
    }
    assert (
        hashlib.sha256((EXPERIMENT / policy["blind_probe"]["path"]).read_bytes()).hexdigest()
        == policy["blind_probe"]["sha256"]
    )
    assert len(policy["gates"]) == 4
    assert "Do not tune" in policy["reuse_policy"]
