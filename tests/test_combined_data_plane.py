from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from molor.experimental.combined_data_plane import (
    audit_combined_data_plane_status,
    build_combined_data_plane_status,
    discover_required_materialization_ids,
    load_combined_data_plane_status,
    write_json_artifact,
)
from molor.identity import fingerprint_json


ROOT = Path(__file__).resolve().parents[1]
RACK = ROOT / "experiments" / "expert-rack-v5"
TAXONOMY = RACK / "expert-taxonomy-v1.json"
COVERAGE = RACK / "corpus-coverage-v1.json"
ASSIGNMENT = RACK / "source-expert-assignment-v1.json"
REPO_INVENTORY = RACK / "repo-local-corpus-inventory-v1.json"
REGISTRATION_FRAGMENT = (
    RACK / "embedded-grounded-reconstruction-v1" / "registration-fragment.json"
)
BUSINESS_LEAVES = {
    "business-core",
    "business-finance",
    "business-operations",
    "funding-capital",
    "growth-gtm",
    "investing-markets",
}
EMBEDDED_RECONSTRUCTION_LEAVES = {
    "embedded-code-generation",
    "embedded-knowledge",
}
def _embedded_reconstruction_fixture(tmp_path: Path) -> Path:
    routes = [
        {
            "leaf_id": leaf,
            "split": split,
            "source_rows": 1,
            "source_characters": 100,
            "emitted_chunks": 2,
            "emitted_characters": 100,
            "tiers": {"le-1024": 1, "1025-4096": 1},
        }
        for leaf in sorted(EMBEDDED_RECONSTRUCTION_LEAVES)
        for split in ("train", "validation", "test")
    ]
    accounting = {
        "selected_input_rows": 6,
        "terminal_source_statuses": {
            "emitted": 6,
            "duplicate": 0,
            "quarantined": 0,
            "rejected": 0,
            "deferred": 0,
        },
        "emitted_message_records": 12,
        "source_characters": 600,
        "emitted_characters": 600,
        "truncated_rows": 0,
        "deferred_rows": 0,
        "characters_omitted": 0,
        "characters_reordered": 0,
        "characters_duplicated": 0,
        "every_source_concatenation_exact": True,
        "fully_reconciled": True,
        "zero_silent_omissions": True,
        "by_leaf_split": routes,
    }
    outputs = []
    for leaf in sorted(EMBEDDED_RECONSTRUCTION_LEAVES):
        for split in ("train", "validation", "test"):
            for tier, token_count in (("le-1024", 800), ("1025-4096", 4000)):
                outputs.append(
                    {
                        "leaf_id": leaf,
                        "split": split,
                        "tier": tier,
                        "records": 1,
                        "minimum_rendered_tokens": token_count,
                        "maximum_rendered_tokens": token_count,
                        "messages_sha256": hashlib.sha256(
                            f"messages-{leaf}-{split}-{tier}".encode()
                        ).hexdigest(),
                        "provenance_sha256": hashlib.sha256(
                            f"provenance-{leaf}-{split}-{tier}".encode()
                        ).hexdigest(),
                    }
                )
    semantic = {
        "schema_version": "molor.embedded-grounded-reconstruction.v1",
        "task_contract": {
            "task_type": "grounded_reference_reconstruction",
            "assistant_target": "contiguous verbatim source-text chunk",
            "selected_leaves": sorted(EMBEDDED_RECONSTRUCTION_LEAVES),
            "user_prompt_source_fields": ["repo", "file_path", "leaf", "chunk_index"],
            "contains_synthetic_question_answering": False,
            "proves_source_correctness": False,
            "proves_answer_quality": False,
            "truncation": False,
            "deferred_rows_permitted": False,
            "maximum_rendered_tokens_inclusive": 4096,
            "concatenated_targets_exactly_reproduce_each_source": True,
        },
        "split_and_leakage_contract": {
            "source_splits_preserved": True,
            "source_group_ids_preserved": True,
            "exact_content_duplicate_guard": True,
            "leaf_upstream_identity_duplicate_guard": True,
            "opposite_split_exact_content_guard": True,
            "opposite_split_group_guard": True,
        },
        "accounting": accounting,
        "outputs": outputs,
    }
    manifest = {**semantic, "content_sha256": fingerprint_json(semantic)}
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    primary = tmp_path / "embedded-primary"
    verification = tmp_path / "embedded-verification"
    primary.mkdir()
    verification.mkdir()
    (primary / "manifest.json").write_bytes(rendered)
    (verification / "manifest.json").write_bytes(rendered)
    receipt_accounting = {
        key: accounting[key]
        for key in (
            "selected_input_rows",
            "emitted_message_records",
            "source_characters",
            "emitted_characters",
            "characters_omitted",
            "characters_reordered",
            "characters_duplicated",
            "truncated_rows",
            "deferred_rows",
            "every_source_concatenation_exact",
            "zero_silent_omissions",
            "by_leaf_split",
        )
    }
    manifest_sha = hashlib.sha256(rendered).hexdigest()
    receipt = {
        "schema_version": "molor.embedded-grounded-reconstruction-build-receipt.v1",
        "scope": {
            "leaves": sorted(EMBEDDED_RECONSTRUCTION_LEAVES),
            "modality": "reference_cpt",
            "splits": ["train", "validation", "test"],
        },
        "accounting": receipt_accounting,
        "artifacts": {
            "manifest": {
                "bytes": len(rendered),
                "sha256": manifest_sha,
                "content_sha256": manifest["content_sha256"],
            }
        },
        "independent_rebuild": {
            "byte_identical": True,
            "files_compared": 1,
            "primary_files": 1,
            "verification_files": 1,
            "primary_bytes": len(rendered),
            "verification_bytes": len(rendered),
            "primary_root": str(primary),
            "verification_root": str(verification),
            "verification_manifest_sha256": manifest_sha,
        },
        "tokenizer": {
            "model": "Qwen3-0.6B",
            "aggregate_sha256": "1" * 64,
            "chat_template_sha256": "2" * 64,
        },
        "training_run_started": False,
    }
    receipt_path = tmp_path / "embedded-grounded-reconstruction-v1.build-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_path


def _rewrite_embedded_fixture(
    receipt_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    primary = Path(receipt["independent_rebuild"]["primary_root"]) / "manifest.json"
    verification = (
        Path(receipt["independent_rebuild"]["verification_root"]) / "manifest.json"
    )
    manifest = json.loads(primary.read_text(encoding="utf-8"))
    mutate(manifest)
    semantic = {key: value for key, value in manifest.items() if key != "content_sha256"}
    manifest["content_sha256"] = fingerprint_json(semantic)
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    primary.write_bytes(rendered)
    verification.write_bytes(rendered)
    manifest_sha = hashlib.sha256(rendered).hexdigest()
    receipt["artifacts"]["manifest"] = {
        "bytes": len(rendered),
        "sha256": manifest_sha,
        "content_sha256": manifest["content_sha256"],
    }
    for key in receipt["accounting"]:
        receipt["accounting"][key] = manifest["accounting"][key]
    rebuild = receipt["independent_rebuild"]
    rebuild["primary_bytes"] = len(rendered)
    rebuild["verification_bytes"] = len(rendered)
    rebuild["verification_manifest_sha256"] = manifest_sha
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _materializations(tmp_path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    required = discover_required_materialization_ids(
        taxonomy_path=TAXONOMY,
        coverage_path=COVERAGE,
        assignment_path=ASSIGNMENT,
        repo_inventory_path=REPO_INVENTORY,
    )
    for materialization_id in required:
        if materialization_id == "embedded-grounded-reconstruction-v1":
            result[materialization_id] = _embedded_reconstruction_fixture(tmp_path)
            continue
        path = tmp_path / f"{materialization_id}.manifest.json"
        if materialization_id == "business-grounded-reconstruction-v1":
            route_status = {
                "emitted": 1,
                "duplicate": 0,
                "quarantined": 0,
                "rejected": 0,
                "deferred": 0,
            }
            fixture = {
                "schema_version": "molor.business-grounded-reconstruction.v1",
                "task_contract": {
                    "task_type": "grounded_reference_reconstruction",
                    "assistant_target": "verbatim canonical source passage",
                    "assistant_target_is_verbatim_source_passage": True,
                    "contains_synthetic_question_answering": False,
                    "proves_answer_quality": False,
                },
                "split_and_dedup_policy": {
                    "preserve_canonical_leaf_and_split": True,
                    "document_id_guard": True,
                    "exact_title_text_guard": True,
                    "source_family_guard": True,
                },
                "rendering_policy": {
                    "truncation": False,
                    "deferred_rows_permitted": False,
                },
                "accounting": {
                    "input_records": 18,
                    "terminal_statuses": {
                        "emitted": 18,
                        "duplicate": 0,
                        "quarantined": 0,
                        "rejected": 0,
                        "deferred": 0,
                    },
                    "by_leaf_and_split": {
                        leaf: {
                            split: dict(route_status) for split in ("train", "validation", "test")
                        }
                        for leaf in BUSINESS_LEAVES
                    },
                    "zero_silent_omissions": True,
                    "fully_reconciled": True,
                    "truncated_rows": 0,
                    "deferred_rows": 0,
                    "source_provenance_ids_missing": 0,
                },
            }
        else:
            fixture = {
                "schema_version": "fixture.aggregate-manifest.v1",
                "materialization_id": materialization_id,
            }
        path.write_text(
            json.dumps(fixture, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result[materialization_id] = path
    return result


def _build(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    materializations = _materializations(tmp_path)
    status = build_combined_data_plane_status(
        taxonomy_path=TAXONOMY,
        coverage_path=COVERAGE,
        assignment_path=ASSIGNMENT,
        repo_inventory_path=REPO_INVENTORY,
        materialization_artifacts=materializations,
    )
    return status, materializations


def test_combined_status_covers_every_leaf_and_safe_source(tmp_path: Path) -> None:
    status, _ = _build(tmp_path)

    assert status["schema_version"] == "molor.combined-data-plane-status.v1"
    assert status["accounting"]["taxonomy_leaves"] == 25
    assert status["accounting"]["external_root_entries"] == 43
    assert status["accounting"]["external_root_entries_accounted"] == 43
    assert status["accounting"]["repo_local_data_artifacts"] == 128
    assert status["accounting"]["repo_local_data_artifacts_accounted"] == 128
    assert status["accounting"]["repo_local_source_containers"] == 4
    assert status["accounting"]["repo_local_source_containers_accounted"] == 4
    assert status["accounting"]["zero_silent_safe_source_omissions"] is True
    assert status["accounting"]["zero_double_counted_canonical_materializations"] is True
    semantic = {key: value for key, value in status.items() if key != "content_sha256"}
    assert status["content_sha256"] == fingerprint_json(semantic)


def test_embedded_registration_fragment_pins_current_control_plane() -> None:
    fragment = json.loads(REGISTRATION_FRAGMENT.read_text(encoding="utf-8"))
    integration = fragment["control_plane_integration"]
    assignment = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    status_path = RACK / "combined-data-plane-status-v1.json"
    status = load_combined_data_plane_status(status_path)

    assert integration["source_assignment_file_sha256"] == hashlib.sha256(
        ASSIGNMENT.read_bytes()
    ).hexdigest()
    assert integration["source_assignment_semantic_sha256"] == assignment["content_sha256"]
    assert integration["combined_status_file_sha256"] == hashlib.sha256(
        status_path.read_bytes()
    ).hexdigest()
    assert integration["combined_status_semantic_sha256"] == status["content_sha256"]
    assert integration["independent_raw_reference_cpt_chat_sampling"] is False
    assert integration["zero_double_counted_derived_chat_views"] is True


def test_repo_canonicals_aliases_negatives_and_eval_are_explicit(tmp_path: Path) -> None:
    status, _ = _build(tmp_path)
    registrations = {item["registration_id"]: item for item in status["source_registrations"]}

    taskmaster = registrations["repo-data/taskmaster-coffee-api-v2"]
    terminal = registrations["repo-data/terminal-cli"]
    old_ocb = registrations["repo-data/ocb-can-verified"]
    negative = registrations["repo-data/legacy-router-v7"]
    evaluation = registrations["repo-data/router-intent-probe-v2"]
    assert taskmaster["roles"] == ["eval", "sft"]
    assert terminal["roles"] == ["eval", "sft"]
    assert old_ocb["roles"] == ["alias"]
    assert old_ocb["materialization_ids"] == ["embedded-source-experts-v1"]
    assert negative["roles"] == ["quarantine"]
    assert evaluation["roles"] == ["eval"]


def test_leaf_readiness_records_only_real_sft_gaps(tmp_path: Path) -> None:
    status, _ = _build(tmp_path)
    leaves = {item["leaf_id"]: item for item in status["leaves"]}

    assert set(leaves) == {
        leaf
        for family in json.loads(TAXONOMY.read_text(encoding="utf-8"))["families"]
        for leaf in (item["leaf_id"] for item in family["leaves"])
    }
    assert leaves["terminal-cli"]["readiness"] == "sft-ready"
    assert leaves["embedded-can-diagnostics"]["readiness"] == "sft-ready"
    assert leaves["agentic-planning"]["readiness"] == "sft-ready"
    assert all(leaves[leaf]["readiness"] == "sft-ready" for leaf in BUSINESS_LEAVES)
    assert all(leaves[leaf]["explicit_gaps"] == [] for leaf in BUSINESS_LEAVES)
    assert status["accounting"]["explicit_sft_gap_leaves"] == []


def test_business_reconstruction_is_pinned_as_honest_supervision(tmp_path: Path) -> None:
    status, _ = _build(tmp_path)
    pin = next(
        item
        for item in status["materializations"]
        if item["materialization_id"] == "business-grounded-reconstruction-v1"
    )
    contract = pin["supervision_contract"]
    assert contract["task_type"] == "grounded_reference_reconstruction"
    assert contract["assistant_target"] == "verbatim canonical source passage"
    assert contract["contains_synthetic_question_answering"] is False
    assert contract["proves_answer_quality"] is False
    assert contract["business_leaves"] == sorted(BUSINESS_LEAVES)
    assert contract["splits"] == ["test", "train", "validation"]
    assert contract["zero_silent_omissions"] is True

    leaves = {item["leaf_id"]: item for item in status["leaves"]}
    for leaf in BUSINESS_LEAVES:
        assert (
            "business-grounded-reconstruction-v1" in leaves[leaf]["ready_sft_materialization_ids"]
        )


def test_embedded_reconstruction_is_pinned_without_double_chat_sampling(tmp_path: Path) -> None:
    status, _ = _build(tmp_path)
    pin = next(
        item
        for item in status["materializations"]
        if item["materialization_id"] == "embedded-grounded-reconstruction-v1"
    )
    contract = pin["supervision_contract"]
    assert contract["assistant_target"] == "contiguous verbatim source-text chunk"
    assert contract["maximum_rendered_tokens_inclusive"] == 4096
    assert contract["contains_synthetic_question_answering"] is False
    assert contract["zero_failure_dispositions"] is True
    assert contract["source_rows"] == 6
    assert contract["source_characters"] == contract["emitted_characters"] == 600
    assert contract["independent_rebuild_byte_identical"] is True

    leaves = {item["leaf_id"]: item for item in status["leaves"]}
    for leaf in EMBEDDED_RECONSTRUCTION_LEAVES:
        assert "embedded-grounded-reconstruction-v1" in leaves[leaf][
            "ready_sft_materialization_ids"
        ]
        assert "embedded-source-experts-v1" in leaves[leaf][
            "ready_reference_materialization_ids"
        ]
    derived = status["sampling_contract"]["derived_supervision_views"]
    assert derived == [
        {
            "raw_materialization_id": "embedded-source-experts-v1",
            "raw_modality": "reference_cpt",
            "target_leaves": sorted(EMBEDDED_RECONSTRUCTION_LEAVES),
            "independent_chat_sampling": False,
            "supervised_materialization_id": "embedded-grounded-reconstruction-v1",
            "reason": (
                "raw references remain accounted while the single supervised view "
                "uses verbatim token-bounded reconstruction"
            ),
        }
    ]
    assert status["accounting"]["zero_double_counted_derived_chat_views"] is True


def test_embedded_reconstruction_requires_an_independent_rebuild(tmp_path: Path) -> None:
    materializations = _materializations(tmp_path)
    receipt_path = materializations["embedded-grounded-reconstruction-v1"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["independent_rebuild"]["byte_identical"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="independent rebuild pin"):
        build_combined_data_plane_status(
            taxonomy_path=TAXONOMY,
            coverage_path=COVERAGE,
            assignment_path=ASSIGNMENT,
            repo_inventory_path=REPO_INVENTORY,
            materialization_artifacts=materializations,
        )


@pytest.mark.parametrize(
    ("case", "error"),
    (
        ("token-ceiling", "task contract mismatch"),
        ("split-guard", "split/leakage guards"),
        ("failure-disposition", "failure dispositions"),
        ("character-reconciliation", "row/character totals"),
    ),
)
def test_embedded_reconstruction_contract_tamper_fails_closed(
    tmp_path: Path,
    case: str,
    error: str,
) -> None:
    materializations = _materializations(tmp_path)
    receipt_path = materializations["embedded-grounded-reconstruction-v1"]

    def mutate(manifest: dict[str, object]) -> None:
        if case == "token-ceiling":
            manifest["task_contract"]["maximum_rendered_tokens_inclusive"] = 4097
        elif case == "split-guard":
            manifest["split_and_leakage_contract"]["opposite_split_group_guard"] = False
        elif case == "failure-disposition":
            manifest["accounting"]["terminal_source_statuses"]["duplicate"] = 1
        elif case == "character-reconciliation":
            manifest["accounting"]["emitted_characters"] = 599
        else:  # pragma: no cover - parametrization is closed above
            raise AssertionError(case)

    _rewrite_embedded_fixture(receipt_path, mutate)
    with pytest.raises(ValueError, match=error):
        build_combined_data_plane_status(
            taxonomy_path=TAXONOMY,
            coverage_path=COVERAGE,
            assignment_path=ASSIGNMENT,
            repo_inventory_path=REPO_INVENTORY,
            materialization_artifacts=materializations,
        )


def test_dishonest_reconstruction_contract_is_not_supervised_ready(tmp_path: Path) -> None:
    materializations = _materializations(tmp_path)
    reconstruction = materializations["business-grounded-reconstruction-v1"]
    value = json.loads(reconstruction.read_text(encoding="utf-8"))
    value["task_contract"]["proves_answer_quality"] = True
    reconstruction.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="task contract mismatch"):
        build_combined_data_plane_status(
            taxonomy_path=TAXONOMY,
            coverage_path=COVERAGE,
            assignment_path=ASSIGNMENT,
            repo_inventory_path=REPO_INVENTORY,
            materialization_artifacts=materializations,
        )


def test_audit_detects_materialization_pin_drift(tmp_path: Path) -> None:
    status, materializations = _build(tmp_path)
    status_path = tmp_path / "combined-status.json"
    write_json_artifact(status, status_path)
    receipt = audit_combined_data_plane_status(
        status_path=status_path,
        taxonomy_path=TAXONOMY,
        coverage_path=COVERAGE,
        assignment_path=ASSIGNMENT,
        repo_inventory_path=REPO_INVENTORY,
        materialization_artifacts=materializations,
    )
    receipt_semantic = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == fingerprint_json(receipt_semantic)

    changed = materializations["terminal-cli-v1"]
    changed.write_text('{"schema_version":"fixture.changed.v1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its aggregate dependencies"):
        audit_combined_data_plane_status(
            status_path=status_path,
            taxonomy_path=TAXONOMY,
            coverage_path=COVERAGE,
            assignment_path=ASSIGNMENT,
            repo_inventory_path=REPO_INVENTORY,
            materialization_artifacts=materializations,
        )


def test_loader_rejects_semantic_tampering(tmp_path: Path) -> None:
    status, _ = _build(tmp_path)
    status["revision"] = "tampered"
    path = tmp_path / "tampered-status.json"
    path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic fingerprint drift"):
        load_combined_data_plane_status(path)
