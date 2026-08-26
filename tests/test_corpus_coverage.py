from __future__ import annotations

import json
from pathlib import Path

import pytest

from molor.experimental.corpus_coverage import (
    audit_corpus_coverage,
    load_corpus_coverage_ledger,
)
from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.identity import fingerprint_json


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "expert-rack-v5"
    / "expert-taxonomy-v1.json"
)


def _ledger(path: Path) -> Path:
    value = {
        "schema_version": "molor.corpus-coverage-ledger.v1",
        "ledger_id": "fixture-ledger",
        "revision": "1",
        "dataset_root_label": "datasets",
        "entries": [
            {
                "path": "source-a",
                "kind": "directory",
                "disposition": "raw-source",
                "canonical_dataset_id": "source-a-v1",
                "target_leaves": ["formal-math"],
                "note": "Synthetic source fixture.",
            },
            {
                "path": "tool.py",
                "kind": "file",
                "disposition": "preparation-tooling",
                "canonical_dataset_id": "tooling-v1",
                "target_leaves": [],
                "note": "Synthetic preparation tool fixture.",
            },
        ],
    }
    value["content_sha256"] = fingerprint_json(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_root_audit_accounts_every_eligible_entry_and_prunes_blocked_root(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    (root / "source-a").mkdir(parents=True)
    (root / "source-a" / "content.jsonl").write_text("not inspected", encoding="utf-8")
    (root / "tool.py").write_text("not inspected", encoding="utf-8")
    blocked = root / "private-security-corpus"
    blocked.mkdir()
    (blocked / "never-open.jsonl").write_text("blocked", encoding="utf-8")
    ledger = _ledger(tmp_path / "ledger.json")

    receipt = audit_corpus_coverage(
        dataset_root=root,
        ledger_path=ledger,
        taxonomy_path=TAXONOMY_PATH,
    )

    assert receipt["eligible_root_entries"] == 2
    assert receipt["blocked_root_entries_pruned_before_traversal"] == 1
    assert receipt["zero_silent_root_omissions"] is True
    assert receipt["contains_corpus_content"] is False
    assert receipt["unlisted_eligible_entries"] == 0


def test_ledger_and_audit_fail_on_tamper_or_unlisted_source(tmp_path: Path) -> None:
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    ledger = _ledger(tmp_path / "ledger.json")
    value = json.loads(ledger.read_text(encoding="utf-8"))
    value["entries"][0]["note"] = "tampered"
    ledger.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        load_corpus_coverage_ledger(ledger, taxonomy=taxonomy)

    ledger = _ledger(tmp_path / "ledger-clean.json")
    root = tmp_path / "datasets"
    root.mkdir()
    (root / "source-a").mkdir()
    (root / "tool.py").write_text("fixture", encoding="utf-8")
    (root / "unlisted").mkdir()
    with pytest.raises(ValueError, match="coverage drift"):
        audit_corpus_coverage(
            dataset_root=root,
            ledger_path=ledger,
            taxonomy_path=TAXONOMY_PATH,
        )
