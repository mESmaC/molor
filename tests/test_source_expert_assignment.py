from __future__ import annotations

import json
from pathlib import Path

import pytest

from molor.experimental.source_expert_assignment import audit_source_expert_assignments
from molor.identity import fingerprint_json


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = ROOT / "experiments" / "expert-rack-v5" / "source-expert-assignment-v1.json"
COVERAGE = ROOT / "experiments" / "expert-rack-v5" / "corpus-coverage-v1.json"
TAXONOMY = ROOT / "experiments" / "expert-rack-v5" / "expert-taxonomy-v1.json"


def _write_matrix(path: Path, value: dict[str, object]) -> Path:
    semantic = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = fingerprint_json(semantic)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_repository_matrix_accounts_every_source_and_exposes_gaps() -> None:
    receipt = audit_source_expert_assignments(
        assignment_path=ASSIGNMENT,
        coverage_path=COVERAGE,
        taxonomy_path=TAXONOMY,
    )

    assert receipt["covered_root_entries"] == 43
    assert receipt["source_groups"] == 14
    assert receipt["effective_source_entries"] == 35
    assert receipt["administrative_or_lineage_alias_entries"] == 8
    assert receipt["assignment_rows"] == 306
    assert receipt["effective_assignment_rows"] == 170
    assert receipt["role_assignment_rows"] == {
        "alias": 136,
        "auxiliary": 4,
        "reference/pretraining": 68,
        "sft": 98,
    }
    assert receipt["materialization_status_rows"] == {
        "deferred": 4,
        "not-applicable": 136,
        "ready": 166,
    }
    assert receipt["materialization_gap_source_entries"] == 0
    assert receipt["materialization_gap_canonical_datasets"] == []
    assert receipt["materializing_source_entries"] == 0
    assert receipt["materializing_canonical_datasets"] == []
    assert receipt["canonical_taxonomy_leaves"] == 25
    assert receipt["effectively_assigned_leaves"] == 24
    assert receipt["ready_leaves"] == 24
    assert receipt["assigned_but_not_ready_leaves"] == 0
    assert receipt["unassigned_leaf_gaps"] == ["terminal-cli"]
    assert receipt["zero_silent_root_omissions"] is True
    assert receipt["contains_corpus_content"] is False

    matrix = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    business = next(
        group for group in matrix["source_groups"] if group["group_id"] == "business-sources"
    )
    assert all(policy["role"] == "sft" for policy in business["policies"])
    reconstruction_leaves = {
        leaf
        for policy in business["policies"]
        if "business-grounded-reconstruction-v1" in policy["materialization_ids"]
        for leaf in policy["target_leaves"]
    }
    assert reconstruction_leaves == {
        "business-core",
        "business-finance",
        "business-operations",
        "funding-capital",
        "growth-gtm",
        "investing-markets",
    }
    assert "not synthetic QA" in business["note"]
    assert "does not prove answer quality" in business["note"]

    embedded_groups = {
        group["group_id"]: group
        for group in matrix["source_groups"]
        if group["group_id"]
        in {
            "embedded-can-code-reference",
            "embedded-code-reference",
            "embedded-visual-reference",
        }
    }
    assert len(embedded_groups) == 3
    for group in embedded_groups.values():
        reconstruction = next(policy for policy in group["policies"] if policy["role"] == "sft")
        assert reconstruction["target_leaves"] == [
            "embedded-code-generation",
            "embedded-knowledge",
        ]
        assert reconstruction["materialization_ids"] == [
            "embedded-grounded-reconstruction-v1"
        ]
        reference = next(
            policy for policy in group["policies"] if policy["role"] == "reference/pretraining"
        )
        assert reference["materialization_ids"] == ["embedded-source-experts-v1"]
        assert "not independently chat-sampled" in group["note"]


def test_matrix_rejects_semantic_tamper(tmp_path: Path) -> None:
    value = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    value["source_groups"][0]["note"] = "changed without updating identity"
    tampered = tmp_path / "tampered-assignment.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="content_sha256"):
        audit_source_expert_assignments(
            assignment_path=tampered,
            coverage_path=COVERAGE,
            taxonomy_path=TAXONOMY,
        )


def test_matrix_rejects_a_silently_omitted_coverage_root(tmp_path: Path) -> None:
    value = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    groups = value["source_groups"]
    embedded_group = next(
        group for group in groups if group["group_id"] == "embedded-code-reference"
    )
    embedded_group["source_paths"].pop()
    incomplete = _write_matrix(tmp_path / "incomplete-assignment.json", value)

    with pytest.raises(ValueError, match="missing sources"):
        audit_source_expert_assignments(
            assignment_path=incomplete,
            coverage_path=COVERAGE,
            taxonomy_path=TAXONOMY,
        )


def test_matrix_rejects_unknown_expert_leaf(tmp_path: Path) -> None:
    value = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    value["source_groups"][0]["policies"][0]["target_leaves"].append("zz-unknown-leaf")
    invalid = _write_matrix(tmp_path / "invalid-leaf-assignment.json", value)

    with pytest.raises(ValueError, match="unknown leaves"):
        audit_source_expert_assignments(
            assignment_path=invalid,
            coverage_path=COVERAGE,
            taxonomy_path=TAXONOMY,
        )
