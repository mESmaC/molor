from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.business_acquisition import (
    SnapshotSource,
    build_business_acquisition_receipt,
)


def _source_manifest(path: Path, *, license_gate: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "fixture.business-sources.v1",
                "private_training_policy": {
                    "license_or_rights_metadata_is_an_admission_gate": license_gate
                },
            }
        ),
        encoding="utf-8",
    )


def test_receipt_is_deterministic_prunes_blocked_trees_and_contains_no_excerpts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "business-sources"
    source = root / "operating-handbook"
    source.mkdir(parents=True)
    (source / "one.md").write_text("private fixture one", encoding="utf-8")
    (source / "two.json").write_text('{"fixture": 2}\n', encoding="utf-8")
    blocked = source / "security-notes"
    blocked.mkdir()
    (blocked / "never-read.txt").write_text("blocked fixture", encoding="utf-8")
    git_objects = source / ".git" / "objects"
    git_objects.mkdir(parents=True)
    (git_objects / "ignored").write_bytes(b"git fixture")
    manifest = tmp_path / "source-manifest.json"
    _source_manifest(manifest)
    spec = SnapshotSource(
        source_id="operating-handbook",
        relative_root="operating-handbook",
        origins=("https://example.invalid/handbook",),
        target_leaves=("business-core", "business-operations"),
        expected_files=2,
    )

    destinations = (tmp_path / "receipt-a.json", tmp_path / "receipt-b.json")
    sums = (tmp_path / "sums-a.txt", tmp_path / "sums-b.txt")
    for destination, sums_destination in zip(destinations, sums, strict=True):
        build_business_acquisition_receipt(
            snapshot_root=root,
            source_manifest=manifest,
            destination=destination,
            sums_destination=sums_destination,
            snapshot_at="2026-08-25T20:40:00-04:00",
            sources=(spec,),
        )

    assert destinations[0].read_bytes() == destinations[1].read_bytes()
    assert sums[0].read_bytes() == sums[1].read_bytes()
    receipt = json.loads(destinations[0].read_text(encoding="utf-8"))
    assert receipt["totals"] == {
        "bytes": 35,
        "file_count": 2,
        "known_missing": 0,
        "source_count": 1,
        "zero_silent_drops": True,
    }
    source_receipt = receipt["sources"][0]
    assert source_receipt["pruned_paths"] == [".git", "security-notes"]
    assert source_receipt["partition_policy"] == "deterministic_artifact_hash_90_5_5"
    assert [artifact["path"] for artifact in source_receipt["artifacts"]] == [
        "one.md",
        "two.json",
    ]
    assert "private fixture" not in destinations[0].read_text(encoding="utf-8")
    expected_hash = hashlib.sha256(b"private fixture one").hexdigest()
    assert expected_hash in sums[0].read_text(encoding="utf-8")


def test_receipt_rejects_license_gated_manifest_and_file_count_drift(tmp_path: Path) -> None:
    root = tmp_path / "business-sources"
    source = root / "source"
    source.mkdir(parents=True)
    (source / "one.md").write_text("fixture", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    spec = SnapshotSource(
        source_id="source",
        relative_root="source",
        origins=("https://example.invalid/source",),
        target_leaves=("business-core",),
        expected_files=2,
    )

    _source_manifest(manifest, license_gate=True)
    with pytest.raises(ValueError, match="disable license admission gating"):
        build_business_acquisition_receipt(
            snapshot_root=root,
            source_manifest=manifest,
            destination=tmp_path / "receipt.json",
            sums_destination=tmp_path / "sums.txt",
            snapshot_at="fixture",
            sources=(spec,),
        )

    _source_manifest(manifest)
    with pytest.raises(RuntimeError, match="has 1 files; expected 2"):
        build_business_acquisition_receipt(
            snapshot_root=root,
            source_manifest=manifest,
            destination=tmp_path / "receipt.json",
            sums_destination=tmp_path / "sums.txt",
            snapshot_at="fixture",
            sources=(spec,),
        )


def test_snapshot_source_and_root_reject_cyber_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside this receipt"):
        SnapshotSource(
            source_id="blocked",
            relative_root="cyber-corpus",
            origins=("https://example.invalid/blocked",),
            target_leaves=("blocked",),
        )

    root = tmp_path / "security-data"
    root.mkdir()
    manifest = tmp_path / "manifest.json"
    _source_manifest(manifest)
    with pytest.raises(ValueError, match="may not be a cyber/security path"):
        build_business_acquisition_receipt(
            snapshot_root=root,
            source_manifest=manifest,
            destination=tmp_path / "receipt.json",
            sums_destination=tmp_path / "sums.txt",
            snapshot_at="fixture",
            sources=(),
        )
