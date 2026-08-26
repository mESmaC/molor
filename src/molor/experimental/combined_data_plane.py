"""Build and audit the aggregate-only expert-rack data-plane status.

The status joins four content-addressed control-plane artifacts and aggregate build
manifests.  It never opens a corpus shard.  External root sources and repo-local artifacts
are each registered exactly once, while aliases, historical negatives, held-out evaluations,
and quarantines remain visible without becoming additional training inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from molor.experimental.corpus_coverage import load_corpus_coverage_ledger
from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.experimental.source_expert_assignment import (
    load_source_expert_assignment_matrix,
)
from molor.identity import fingerprint_json, validate_sha256


COMBINED_DATA_PLANE_SCHEMA = "molor.combined-data-plane-status.v1"
COMBINED_DATA_PLANE_RECEIPT_SCHEMA = "molor.combined-data-plane-receipt.v1"
REPO_LOCAL_INVENTORY_SCHEMA = "molor.repo-local-corpus-inventory.v1"

_RESTRICTED_WORDS = ("cyber", "security")
_ROLES = ("alias", "eval", "quarantine", "reference", "sft")
_ACTIVE_TRAINING_ROLES = frozenset({"reference", "sft"})
_BUSINESS_GROUNDED_RECONSTRUCTION_ID = "business-grounded-reconstruction-v1"
_BUSINESS_GROUNDED_RECONSTRUCTION_SCHEMA = "molor.business-grounded-reconstruction.v1"
_EMBEDDED_GROUNDED_RECONSTRUCTION_ID = "embedded-grounded-reconstruction-v1"
_EMBEDDED_GROUNDED_RECONSTRUCTION_SCHEMA = "molor.embedded-grounded-reconstruction.v1"
_EMBEDDED_GROUNDED_RECONSTRUCTION_RECEIPT_SCHEMA = (
    "molor.embedded-grounded-reconstruction-build-receipt.v1"
)
_EMBEDDED_RECONSTRUCTION_LEAVES = frozenset(
    {"embedded-code-generation", "embedded-knowledge"}
)
_BUSINESS_LEAVES = frozenset(
    {
        "business-core",
        "business-finance",
        "business-operations",
        "funding-capital",
        "growth-gtm",
        "investing-markets",
    }
)
_CANONICAL_SPLITS = frozenset({"train", "validation", "test"})
_EXTERNAL_ROLE_MAP = {
    "alias": "alias",
    "auxiliary": "reference",
    "evaluation-only": "eval",
    "quarantine/reject": "quarantine",
    "reference/pretraining": "reference",
    "sft": "sft",
}

# These older repo-local products are represented by a newer canonical materialization.
# Keeping the relationship explicit prevents a mixture from sampling both copies.
_REPO_ALIAS_TARGETS = {
    "legacy-financial-analysis-verified": "business-finance-supervision-v1",
    "legacy-software-architecture": "software-architecture-experts-v1",
    "legacy-software-architecture-v1": "software-architecture-experts-v1",
    "ocb-can-verified": "embedded-source-experts-v1",
    "repo-finqa-source": "business-finance-supervision-v1",
    "repo-taskmaster-source": "taskmaster-coffee-api-v2",
    "repo-tat-qa-source": "business-finance-supervision-v1",
    "verified-embedded-code-union": "verified-embedded-code-v3",
    "verified-embedded-code-v1": "verified-embedded-code-v3",
    "verified-embedded-code-v2": "verified-embedded-code-v3",
}


def _restricted(value: str) -> bool:
    normalized = value.casefold()
    return any(word in normalized for word in _RESTRICTED_WORDS)


def _safe_file(path: str | Path, field: str) -> Path:
    raw = str(path)
    if _restricted(raw):
        raise ValueError(f"{field} may not name a restricted path")
    source = Path(path).expanduser().resolve()
    if _restricted(str(source)):
        raise ValueError(f"{field} may not name a restricted path")
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file")
    return source


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path, field: str) -> dict[str, Any]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != before.st_size
    ):
        raise RuntimeError(f"{field} changed while it was read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"unsupported JSON constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 180:
        raise ValueError(f"{field} must be a non-empty string of at most 180 characters")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-._/" for character in value):
        raise ValueError(f"{field} contains unsupported characters")
    if _restricted(value):
        raise ValueError(f"{field} may not name a restricted source")
    return value


def _inventory(
    path: str | Path,
    *,
    taxonomy_id: str,
    taxonomy_revision: str,
    taxonomy_semantic_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    source = _safe_file(path, "repo_inventory_path")
    value = _strict_json(source, "repo inventory")
    if value.get("schema_version") != REPO_LOCAL_INVENTORY_SCHEMA:
        raise ValueError("unsupported repo-local inventory schema")
    claimed = validate_sha256(value.get("content_sha256"), "inventory.content_sha256")
    if value.get("content_sha256") != claimed:
        raise ValueError("inventory content_sha256 must use lowercase hexadecimal")
    semantic = {key: item for key, item in value.items() if key != "content_sha256"}
    if fingerprint_json(semantic) != claimed:
        raise ValueError("repo-local inventory semantic fingerprint drift")
    contract = value.get("taxonomy")
    if not isinstance(contract, Mapping):
        raise ValueError("repo-local inventory taxonomy contract is missing")
    observed = (
        contract.get("taxonomy_id"),
        contract.get("revision"),
        contract.get("semantic_sha256"),
    )
    expected = (taxonomy_id, taxonomy_revision, taxonomy_semantic_sha256)
    if observed != expected:
        raise ValueError("repo-local inventory taxonomy contract drift")
    summary = value.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("repo-local inventory summary is missing")
    if summary.get("zero_silent_eligible_omissions") is not True:
        raise ValueError("repo-local inventory does not prove complete safe-source accounting")
    if summary.get("contains_corpus_content") is not False:
        raise ValueError("repo-local inventory must be aggregate-only")
    return source, value


def _contract(
    *,
    path: Path,
    artifact_id: str,
    revision: str,
    semantic_sha256: str,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "artifact_name": path.name,
        "file_sha256": _sha256(path),
        "revision": revision,
        "semantic_sha256": semantic_sha256,
    }


def _registration(
    *,
    registration_id: str,
    scope: str,
    source_id: str,
    source_items: Sequence[str],
    disposition: str,
    roles: Sequence[str],
    readiness: str,
    target_leaves: Sequence[str],
    materialization_ids: Sequence[str],
    note: str,
) -> dict[str, object]:
    parsed_roles = sorted(set(roles))
    if not parsed_roles or any(role not in _ROLES for role in parsed_roles):
        raise ValueError(f"registration {registration_id!r} has unsupported roles")
    items = sorted(set(source_items), key=str.casefold)
    if len(items) != len(source_items):
        raise ValueError(f"registration {registration_id!r} repeats source items")
    if any(not isinstance(item, str) or not item or _restricted(item) for item in items):
        raise ValueError(f"registration {registration_id!r} contains an unsafe source item")
    return {
        "registration_id": _identifier(registration_id, "registration_id"),
        "scope": _identifier(scope, "scope"),
        "source_id": _identifier(source_id, "source_id"),
        "source_items": items,
        "disposition": _identifier(disposition, "disposition"),
        "roles": parsed_roles,
        "readiness": _identifier(readiness, "readiness"),
        "target_leaves": sorted(set(target_leaves)),
        "materialization_ids": sorted(set(materialization_ids)),
        "note": note.strip(),
    }


def _external_registrations(matrix: Any) -> list[dict[str, object]]:
    registrations: list[dict[str, object]] = []
    for group in matrix.source_groups:
        for policy_index, policy in enumerate(group.policies):
            role = _EXTERNAL_ROLE_MAP[policy.role]
            roles = [role]
            if policy.materialization_status == "ready" and role in _ACTIVE_TRAINING_ROLES:
                roles.append("eval")
            registrations.append(
                _registration(
                    registration_id=f"external/{group.group_id}/{policy_index:02d}",
                    scope="external-root",
                    source_id=group.group_id,
                    source_items=group.source_paths,
                    disposition=policy.role.replace("/", "-"),
                    roles=roles,
                    readiness=policy.materialization_status,
                    target_leaves=policy.target_leaves,
                    materialization_ids=policy.materialization_ids,
                    note=group.note,
                )
            )
    return registrations


def _repo_group_registration(
    group_id: str,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    first = artifacts[0]
    disposition = str(first["dataset_disposition"])
    source_items = [str(item["path"]) for item in artifacts]
    target_leaves = tuple(str(item) for item in first["target_leaves"])
    dataset_id = str(first["canonical_dataset_id"])
    notes = {str(item["note"]) for item in artifacts}
    if len(notes) != 1:
        raise ValueError(f"repo-local group {group_id!r} has inconsistent notes")
    for item in artifacts:
        stable = (
            item.get("canonical_dataset_id"),
            item.get("dataset_disposition"),
            tuple(item.get("target_leaves", ())),
        )
        expected = (dataset_id, disposition, target_leaves)
        if stable != expected:
            raise ValueError(f"repo-local group {group_id!r} has inconsistent metadata")

    admitted = any(item.get("admitted_for_new_expert_training") is True for item in artifacts)
    has_eval = any(item.get("artifact_role") == "evaluation" for item in artifacts)
    if group_id in _REPO_ALIAS_TARGETS:
        roles = ["alias"]
        readiness = "alias"
        materialization_ids = [_REPO_ALIAS_TARGETS[group_id]]
    elif disposition in {"canonical", "supplemental"} and admitted:
        roles = ["sft"] + (["eval"] if has_eval else [])
        readiness = "ready"
        materialization_ids = [dataset_id]
    elif disposition == "evaluation-only":
        roles = ["eval"]
        readiness = "ready"
        materialization_ids = [dataset_id]
    elif disposition in {"historical-negative", "quarantined"}:
        roles = ["quarantine"]
        readiness = "quarantined"
        materialization_ids = [dataset_id]
    else:
        roles = ["alias"]
        readiness = "alias"
        materialization_ids = [_REPO_ALIAS_TARGETS.get(group_id, dataset_id)]
    return _registration(
        registration_id=f"repo-data/{group_id}",
        scope="repo-local-data",
        source_id=group_id,
        source_items=source_items,
        disposition=disposition,
        roles=roles,
        readiness=readiness,
        target_leaves=target_leaves,
        materialization_ids=materialization_ids,
        note=next(iter(notes)),
    )


def _repo_container_registration(container: Mapping[str, Any]) -> dict[str, object]:
    group_id = str(container["group_id"])
    disposition = str(container["dataset_disposition"])
    dataset_id = str(container["canonical_dataset_id"])
    if disposition == "quarantined":
        roles = ["quarantine"]
        readiness = "quarantined"
        ids = [dataset_id]
    else:
        roles = ["alias"]
        readiness = "alias"
        ids = [_REPO_ALIAS_TARGETS.get(group_id, dataset_id)]
    return _registration(
        registration_id=f"repo-source/{group_id}",
        scope="repo-local-source-container",
        source_id=group_id,
        source_items=[str(container["path"])],
        disposition=disposition,
        roles=roles,
        readiness=readiness,
        target_leaves=tuple(str(item) for item in container["target_leaves"]),
        materialization_ids=ids,
        note=str(container["note"]),
    )


def _repo_registrations(inventory: Mapping[str, Any]) -> list[dict[str, object]]:
    raw_artifacts = inventory.get("artifacts")
    raw_containers = inventory.get("source_containers")
    if not isinstance(raw_artifacts, list) or not isinstance(raw_containers, list):
        raise ValueError("repo-local inventory lacks artifact or source-container arrays")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, artifact in enumerate(raw_artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"repo inventory artifacts[{index}] must be an object")
        group_id = _identifier(artifact.get("group_id"), f"artifacts[{index}].group_id")
        groups[group_id].append(artifact)
    registrations = [
        _repo_group_registration(group_id, tuple(groups[group_id])) for group_id in sorted(groups)
    ]
    for index, container in enumerate(raw_containers):
        if not isinstance(container, Mapping):
            raise ValueError(f"repo inventory source_containers[{index}] must be an object")
        registrations.append(_repo_container_registration(container))
    return registrations


def _required_materialization_ids(
    registrations: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(materialization_id)
                for registration in registrations
                if registration["readiness"] == "ready"
                and _ACTIVE_TRAINING_ROLES.intersection(registration["roles"])
                for materialization_id in registration["materialization_ids"]
            }
        )
    )


def _grounded_reconstruction_contract(aggregate: Mapping[str, Any]) -> dict[str, object]:
    """Validate the honest, split-complete reconstruction supervision contract."""

    if aggregate.get("schema_version") != _BUSINESS_GROUNDED_RECONSTRUCTION_SCHEMA:
        raise ValueError("business grounded reconstruction has an unexpected schema")
    task = aggregate.get("task_contract")
    if not isinstance(task, Mapping):
        raise ValueError("business grounded reconstruction lacks a task contract")
    expected_task = {
        "task_type": "grounded_reference_reconstruction",
        "assistant_target": "verbatim canonical source passage",
        "assistant_target_is_verbatim_source_passage": True,
        "contains_synthetic_question_answering": False,
        "proves_answer_quality": False,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            raise ValueError(f"business grounded reconstruction task contract mismatch for {key!r}")
    split_policy = aggregate.get("split_and_dedup_policy")
    rendering = aggregate.get("rendering_policy")
    if not isinstance(split_policy, Mapping) or not isinstance(rendering, Mapping):
        raise ValueError("business grounded reconstruction lacks split or rendering policy")
    if split_policy.get("preserve_canonical_leaf_and_split") is not True:
        raise ValueError("business grounded reconstruction does not preserve canonical splits")
    if any(
        split_policy.get(key) is not True
        for key in ("document_id_guard", "exact_title_text_guard", "source_family_guard")
    ):
        raise ValueError("business grounded reconstruction lacks required leakage guards")
    if (
        rendering.get("truncation") is not False
        or rendering.get("deferred_rows_permitted") is not False
    ):
        raise ValueError("business grounded reconstruction permits truncation or deferred rows")

    accounting = aggregate.get("accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("business grounded reconstruction lacks accounting")
    terminal = accounting.get("terminal_statuses")
    by_leaf_split = accounting.get("by_leaf_and_split")
    if not isinstance(terminal, Mapping) or not isinstance(by_leaf_split, Mapping):
        raise ValueError("business grounded reconstruction accounting is incomplete")
    input_records = accounting.get("input_records")
    emitted = terminal.get("emitted")
    if type(input_records) is not int or input_records < 1 or emitted != input_records:
        raise ValueError("business grounded reconstruction does not emit every input record")
    for status in ("duplicate", "quarantined", "rejected", "deferred"):
        if terminal.get(status) != 0:
            raise ValueError(f"business grounded reconstruction has nonzero {status!r} accounting")
    if (
        accounting.get("zero_silent_omissions") is not True
        or accounting.get("fully_reconciled") is not True
        or accounting.get("truncated_rows") != 0
        or accounting.get("deferred_rows") != 0
        or accounting.get("source_provenance_ids_missing") != 0
    ):
        raise ValueError("business grounded reconstruction is not fully reconciled")
    if set(by_leaf_split) != _BUSINESS_LEAVES:
        raise ValueError("business grounded reconstruction does not cover all six business leaves")
    route_emissions = 0
    for leaf_id in sorted(_BUSINESS_LEAVES):
        split_rows = by_leaf_split[leaf_id]
        if not isinstance(split_rows, Mapping) or set(split_rows) != _CANONICAL_SPLITS:
            raise ValueError(
                f"business grounded reconstruction leaf {leaf_id!r} lacks all three splits"
            )
        for split in sorted(_CANONICAL_SPLITS):
            statuses = split_rows[split]
            if not isinstance(statuses, Mapping):
                raise ValueError("business grounded reconstruction route accounting is malformed")
            route_emitted = statuses.get("emitted")
            if type(route_emitted) is not int or route_emitted < 1:
                raise ValueError("business grounded reconstruction contains an empty route")
            if any(
                statuses.get(status) != 0
                for status in ("duplicate", "quarantined", "rejected", "deferred")
            ):
                raise ValueError("business grounded reconstruction route is not fully admitted")
            route_emissions += route_emitted
    if route_emissions != input_records:
        raise ValueError("business grounded reconstruction route counts do not reconcile")
    return {
        **expected_task,
        "business_leaves": sorted(_BUSINESS_LEAVES),
        "splits": sorted(_CANONICAL_SPLITS),
        "input_records": input_records,
        "emitted_records": emitted,
        "truncated_rows": 0,
        "deferred_rows": 0,
        "zero_silent_omissions": True,
        "fully_reconciled": True,
    }


def _embedded_grounded_reconstruction_contract(
    receipt: Mapping[str, Any],
) -> dict[str, object]:
    """Validate verbatim chunks, complete accounting, and independent rebuild pins."""

    if receipt.get("schema_version") != _EMBEDDED_GROUNDED_RECONSTRUCTION_RECEIPT_SCHEMA:
        raise ValueError("embedded grounded reconstruction has an unexpected receipt schema")
    if receipt.get("training_run_started") is not False:
        raise ValueError("embedded grounded reconstruction receipt claims a training run")
    tokenizer = receipt.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or tokenizer.get("model") != "Qwen3-0.6B":
        raise ValueError("embedded grounded reconstruction lacks the required tokenizer pin")
    tokenizer_aggregate_sha = validate_sha256(
        tokenizer.get("aggregate_sha256"), "embedded reconstruction tokenizer sha256"
    )
    chat_template_sha = validate_sha256(
        tokenizer.get("chat_template_sha256"),
        "embedded reconstruction chat template sha256",
    )
    scope = receipt.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("embedded grounded reconstruction receipt lacks scope")
    if (
        set(scope.get("leaves", ())) != _EMBEDDED_RECONSTRUCTION_LEAVES
        or scope.get("modality") != "reference_cpt"
        or set(scope.get("splits", ())) != _CANONICAL_SPLITS
    ):
        raise ValueError("embedded grounded reconstruction receipt scope drift")

    artifacts = receipt.get("artifacts")
    rebuild = receipt.get("independent_rebuild")
    if not isinstance(artifacts, Mapping) or not isinstance(rebuild, Mapping):
        raise ValueError("embedded grounded reconstruction lacks artifact or rebuild pins")
    manifest_pin = artifacts.get("manifest")
    if not isinstance(manifest_pin, Mapping):
        raise ValueError("embedded grounded reconstruction lacks a manifest pin")
    expected_manifest_sha = validate_sha256(
        manifest_pin.get("sha256"), "embedded reconstruction manifest sha256"
    )
    expected_content_sha = validate_sha256(
        manifest_pin.get("content_sha256"),
        "embedded reconstruction manifest content_sha256",
    )
    manifest_bytes = manifest_pin.get("bytes")
    if type(manifest_bytes) is not int or manifest_bytes < 1:
        raise ValueError("embedded grounded reconstruction manifest byte pin is invalid")
    if (
        rebuild.get("byte_identical") is not True
        or rebuild.get("files_compared") != rebuild.get("primary_files")
        or rebuild.get("files_compared") != rebuild.get("verification_files")
        or type(rebuild.get("files_compared")) is not int
        or rebuild.get("files_compared") < 1
        or rebuild.get("primary_bytes") != rebuild.get("verification_bytes")
        or type(rebuild.get("primary_bytes")) is not int
        or rebuild.get("primary_bytes") < manifest_bytes
        or rebuild.get("verification_manifest_sha256") != expected_manifest_sha
    ):
        raise ValueError("embedded grounded reconstruction independent rebuild pin is invalid")
    primary_root = rebuild.get("primary_root")
    verification_root = rebuild.get("verification_root")
    if not isinstance(primary_root, str) or not isinstance(verification_root, str):
        raise ValueError("embedded grounded reconstruction rebuild roots are invalid")
    primary_manifest = _safe_file(
        Path(primary_root) / "manifest.json", "embedded reconstruction primary manifest"
    )
    verification_manifest = _safe_file(
        Path(verification_root) / "manifest.json",
        "embedded reconstruction verification manifest",
    )
    if primary_manifest == verification_manifest:
        raise ValueError("embedded grounded reconstruction rebuild roots must be independent")
    if (
        primary_manifest.stat().st_size != manifest_bytes
        or verification_manifest.stat().st_size != manifest_bytes
        or _sha256(primary_manifest) != expected_manifest_sha
        or _sha256(verification_manifest) != expected_manifest_sha
    ):
        raise ValueError("embedded grounded reconstruction rebuild manifest pin drift")

    aggregate = _strict_json(primary_manifest, "embedded reconstruction manifest")
    if aggregate.get("schema_version") != _EMBEDDED_GROUNDED_RECONSTRUCTION_SCHEMA:
        raise ValueError("embedded grounded reconstruction has an unexpected manifest schema")
    declared = validate_sha256(
        aggregate.get("content_sha256"), "embedded reconstruction content_sha256"
    )
    semantic = {key: item for key, item in aggregate.items() if key != "content_sha256"}
    if declared != expected_content_sha or fingerprint_json(semantic) != declared:
        raise ValueError("embedded grounded reconstruction semantic manifest pin drift")

    task = aggregate.get("task_contract")
    if not isinstance(task, Mapping):
        raise ValueError("embedded grounded reconstruction lacks a task contract")
    expected_task = {
        "task_type": "grounded_reference_reconstruction",
        "assistant_target": "contiguous verbatim source-text chunk",
        "contains_synthetic_question_answering": False,
        "proves_source_correctness": False,
        "proves_answer_quality": False,
        "truncation": False,
        "deferred_rows_permitted": False,
        "maximum_rendered_tokens_inclusive": 4096,
        "concatenated_targets_exactly_reproduce_each_source": True,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            raise ValueError(
                f"embedded grounded reconstruction task contract mismatch for {key!r}"
            )
    if (
        set(task.get("selected_leaves", ())) != _EMBEDDED_RECONSTRUCTION_LEAVES
        or task.get("user_prompt_source_fields")
        != ["repo", "file_path", "leaf", "chunk_index"]
    ):
        raise ValueError("embedded grounded reconstruction prompt or leaf scope drift")
    split_contract = aggregate.get("split_and_leakage_contract")
    required_guards = (
        "source_splits_preserved",
        "source_group_ids_preserved",
        "exact_content_duplicate_guard",
        "leaf_upstream_identity_duplicate_guard",
        "opposite_split_exact_content_guard",
        "opposite_split_group_guard",
    )
    if not isinstance(split_contract, Mapping) or any(
        split_contract.get(key) is not True for key in required_guards
    ):
        raise ValueError("embedded grounded reconstruction lacks split/leakage guards")

    accounting = aggregate.get("accounting")
    receipt_accounting = receipt.get("accounting")
    if not isinstance(accounting, Mapping) or not isinstance(receipt_accounting, Mapping):
        raise ValueError("embedded grounded reconstruction accounting is missing")
    source_rows = accounting.get("selected_input_rows")
    emitted_messages = accounting.get("emitted_message_records")
    source_characters = accounting.get("source_characters")
    emitted_characters = accounting.get("emitted_characters")
    if (
        type(source_rows) is not int
        or source_rows < 1
        or type(emitted_messages) is not int
        or emitted_messages < source_rows
        or type(source_characters) is not int
        or source_characters < 1
        or source_characters != emitted_characters
    ):
        raise ValueError("embedded grounded reconstruction row/character totals do not reconcile")
    terminal = accounting.get("terminal_source_statuses")
    if not isinstance(terminal, Mapping) or terminal.get("emitted") != source_rows:
        raise ValueError("embedded grounded reconstruction terminal accounting is incomplete")
    if any(
        terminal.get(status) != 0
        for status in ("duplicate", "quarantined", "rejected", "deferred")
    ):
        raise ValueError("embedded grounded reconstruction has nonzero failure dispositions")
    required_accounting = {
        "characters_omitted": 0,
        "characters_reordered": 0,
        "characters_duplicated": 0,
        "truncated_rows": 0,
        "deferred_rows": 0,
        "every_source_concatenation_exact": True,
        "fully_reconciled": True,
        "zero_silent_omissions": True,
    }
    if any(accounting.get(key) != expected for key, expected in required_accounting.items()):
        raise ValueError("embedded grounded reconstruction is not fully reconciled")
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
    ):
        if receipt_accounting.get(key) != accounting.get(key):
            raise ValueError("embedded reconstruction receipt/manifest accounting drift")

    routes = accounting.get("by_leaf_split")
    if not isinstance(routes, list):
        raise ValueError("embedded grounded reconstruction lacks route accounting")
    observed_routes: set[tuple[str, str]] = set()
    route_rows = route_chunks = route_characters = 0
    for route in routes:
        if not isinstance(route, Mapping):
            raise ValueError("embedded reconstruction route accounting is malformed")
        leaf = route.get("leaf_id")
        split = route.get("split")
        key = (leaf, split)
        if leaf not in _EMBEDDED_RECONSTRUCTION_LEAVES or split not in _CANONICAL_SPLITS:
            raise ValueError("embedded reconstruction route is outside its declared scope")
        if key in observed_routes:
            raise ValueError("embedded reconstruction repeats a leaf/split route")
        observed_routes.add(key)  # type: ignore[arg-type]
        rows = route.get("source_rows")
        chunks = route.get("emitted_chunks")
        characters = route.get("source_characters")
        tiers = route.get("tiers")
        if (
            type(rows) is not int
            or rows < 1
            or type(chunks) is not int
            or chunks < rows
            or type(characters) is not int
            or characters < 1
            or route.get("emitted_characters") != characters
            or not isinstance(tiers, Mapping)
            or set(tiers) != {"le-1024", "1025-4096"}
            or any(type(value) is not int or value < 0 for value in tiers.values())
            or sum(tiers.values()) != chunks
        ):
            raise ValueError("embedded reconstruction route totals do not reconcile")
        route_rows += rows
        route_chunks += chunks
        route_characters += characters
    expected_routes = {
        (leaf, split)
        for leaf in _EMBEDDED_RECONSTRUCTION_LEAVES
        for split in _CANONICAL_SPLITS
    }
    if (
        observed_routes != expected_routes
        or route_rows != source_rows
        or route_chunks != emitted_messages
        or route_characters != source_characters
        or receipt_accounting.get("by_leaf_split") != routes
    ):
        raise ValueError("embedded reconstruction aggregate route accounting drift")

    outputs = aggregate.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("embedded grounded reconstruction lacks output pins")
    expected_outputs = {
        (leaf, split, tier)
        for leaf in _EMBEDDED_RECONSTRUCTION_LEAVES
        for split in _CANONICAL_SPLITS
        for tier in ("le-1024", "1025-4096")
    }
    observed_outputs: set[tuple[str, str, str]] = set()
    output_records = 0
    for output in outputs:
        if not isinstance(output, Mapping):
            raise ValueError("embedded reconstruction output pin is malformed")
        key = (output.get("leaf_id"), output.get("split"), output.get("tier"))
        if key not in expected_outputs or key in observed_outputs:
            raise ValueError("embedded reconstruction output routing drift")
        observed_outputs.add(key)  # type: ignore[arg-type]
        records = output.get("records")
        minimum = output.get("minimum_rendered_tokens")
        maximum = output.get("maximum_rendered_tokens")
        if type(records) is not int or records < 0:
            raise ValueError("embedded reconstruction output record count is invalid")
        if records:
            if (
                type(minimum) is not int
                or type(maximum) is not int
                or minimum < 1
                or minimum > maximum
                or maximum > 4096
                or (key[2] == "le-1024" and maximum > 1024)
                or (key[2] == "1025-4096" and minimum < 1025)
            ):
                raise ValueError("embedded reconstruction output token tier is invalid")
        elif minimum is not None or maximum is not None:
            raise ValueError("empty embedded reconstruction output has token bounds")
        validate_sha256(output.get("messages_sha256"), "embedded messages sha256")
        validate_sha256(output.get("provenance_sha256"), "embedded provenance sha256")
        output_records += records
    if observed_outputs != expected_outputs or output_records != emitted_messages:
        raise ValueError("embedded reconstruction output pins do not reconcile")

    return {
        **expected_task,
        "embedded_leaves": sorted(_EMBEDDED_RECONSTRUCTION_LEAVES),
        "splits": sorted(_CANONICAL_SPLITS),
        "source_rows": source_rows,
        "emitted_message_records": emitted_messages,
        "source_characters": source_characters,
        "emitted_characters": emitted_characters,
        "zero_failure_dispositions": True,
        "zero_silent_omissions": True,
        "fully_reconciled": True,
        "independent_rebuild_byte_identical": True,
        "independent_rebuild_files": rebuild["files_compared"],
        "manifest_file_sha256": expected_manifest_sha,
        "manifest_content_sha256": expected_content_sha,
        "tokenizer_aggregate_sha256": tokenizer_aggregate_sha,
        "chat_template_sha256": chat_template_sha,
    }


def _pin_materializations(
    *,
    required_ids: Sequence[str],
    materialization_artifacts: Mapping[str, str | Path],
) -> list[dict[str, object]]:
    observed_ids = {
        _identifier(item, "materialization_artifacts key") for item in materialization_artifacts
    }
    required = set(required_ids)
    if observed_ids != required:
        raise ValueError(
            "materialization artifact set drift "
            f"(missing={sorted(required - observed_ids)}, extra={sorted(observed_ids - required)})"
        )
    pins: list[dict[str, object]] = []
    for materialization_id in sorted(required):
        path = _safe_file(
            materialization_artifacts[materialization_id],
            f"materialization_artifacts[{materialization_id!r}]",
        )
        lowered = path.name.casefold()
        if path.suffix.casefold() != ".json" or not any(
            label in lowered for label in ("manifest", "receipt", "report")
        ):
            raise ValueError("materialization pins must name aggregate JSON manifests or receipts")
        aggregate = _strict_json(path, f"materialization {materialization_id!r}")
        schema = aggregate.get("schema_version", "unversioned-aggregate-json")
        if not isinstance(schema, str) or not schema:
            raise ValueError(
                "materialization schema_version must be a non-empty string when present"
            )
        declared = aggregate.get("content_sha256")
        if declared is not None:
            declared = validate_sha256(declared, "materialization content_sha256")
        pin: dict[str, object] = {
            "materialization_id": materialization_id,
            "artifact_name": path.name,
            "artifact_kind": next(
                label for label in ("manifest", "receipt", "report") if label in lowered
            ),
            "bytes": path.stat().st_size,
            "file_sha256": _sha256(path),
            "schema_version": schema,
            "declared_content_sha256": declared,
        }
        if materialization_id == _BUSINESS_GROUNDED_RECONSTRUCTION_ID:
            pin["supervision_contract"] = _grounded_reconstruction_contract(aggregate)
        if materialization_id == _EMBEDDED_GROUNDED_RECONSTRUCTION_ID:
            pin["supervision_contract"] = _embedded_grounded_reconstruction_contract(
                aggregate
            )
        pins.append(pin)
    return pins


def _leaf_statuses(
    *, taxonomy: Any, registrations: Sequence[Mapping[str, Any]]
) -> list[dict[str, object]]:
    family_by_leaf = {
        leaf.leaf_id: family.family_id for family in taxonomy.families for leaf in family.leaves
    }
    statuses: list[dict[str, object]] = []
    for leaf_id in sorted(taxonomy.leaf_ids):
        by_role: dict[str, list[str]] = {role: [] for role in _ROLES}
        ready_sft: set[str] = set()
        ready_reference: set[str] = set()
        ready_eval: set[str] = set()
        for registration in registrations:
            if leaf_id not in registration["target_leaves"]:
                continue
            registration_id = str(registration["registration_id"])
            for role in registration["roles"]:
                by_role[str(role)].append(registration_id)
            if registration["readiness"] != "ready":
                continue
            ids = {str(item) for item in registration["materialization_ids"]}
            if "sft" in registration["roles"]:
                ready_sft.update(ids)
            if "reference" in registration["roles"]:
                ready_reference.update(ids)
            if "eval" in registration["roles"]:
                ready_eval.update(ids)
        if ready_sft:
            readiness = "sft-ready"
            gaps: list[str] = []
        elif ready_reference:
            readiness = "reference-ready-sft-gap"
            gaps = ["grounded-sft-not-materialized"]
        else:
            readiness = "training-gap"
            gaps = ["no-ready-training-materialization"]
        if not ready_eval:
            gaps.append("held-out-evaluation-not-registered")
        statuses.append(
            {
                "leaf_id": leaf_id,
                "family_id": family_by_leaf[leaf_id],
                "readiness": readiness,
                "roles": {role: sorted(by_role[role]) for role in _ROLES},
                "ready_sft_materialization_ids": sorted(ready_sft),
                "ready_reference_materialization_ids": sorted(ready_reference),
                "ready_evaluation_materialization_ids": sorted(ready_eval),
                "explicit_gaps": gaps,
            }
        )
    return statuses


def build_combined_data_plane_status(
    *,
    taxonomy_path: str | Path,
    coverage_path: str | Path,
    assignment_path: str | Path,
    repo_inventory_path: str | Path,
    materialization_artifacts: Mapping[str, str | Path],
) -> dict[str, object]:
    """Build one deterministic aggregate-only data-plane status artifact."""

    taxonomy_file = _safe_file(taxonomy_path, "taxonomy_path")
    coverage_file = _safe_file(coverage_path, "coverage_path")
    assignment_file = _safe_file(assignment_path, "assignment_path")
    taxonomy = load_expert_taxonomy(taxonomy_file)
    coverage = load_corpus_coverage_ledger(coverage_file, taxonomy=taxonomy)
    matrix = load_source_expert_assignment_matrix(
        assignment_file,
        coverage=coverage,
        taxonomy=taxonomy,
    )
    inventory_file, inventory = _inventory(
        repo_inventory_path,
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_revision=taxonomy.revision,
        taxonomy_semantic_sha256=taxonomy.fingerprint,
    )

    registrations = _external_registrations(matrix) + _repo_registrations(inventory)
    registrations.sort(key=lambda item: str(item["registration_id"]))
    registration_ids = [str(item["registration_id"]) for item in registrations]
    if len(registration_ids) != len(set(registration_ids)):
        raise ValueError("combined registrations must have unique IDs")

    external_items = {
        str(item)
        for registration in registrations
        if registration["scope"] == "external-root"
        for item in registration["source_items"]
    }
    expected_external = {entry.path for entry in coverage.entries}
    if external_items != expected_external:
        raise ValueError("combined status does not account every external root exactly")
    inventory_artifacts = inventory["artifacts"]
    inventory_containers = inventory["source_containers"]
    repo_data_items = {
        str(item)
        for registration in registrations
        if registration["scope"] == "repo-local-data"
        for item in registration["source_items"]
    }
    expected_repo_data = {str(item["path"]) for item in inventory_artifacts}
    if repo_data_items != expected_repo_data:
        raise ValueError("combined status does not account every repo-local data artifact exactly")
    repo_source_items = {
        str(item)
        for registration in registrations
        if registration["scope"] == "repo-local-source-container"
        for item in registration["source_items"]
    }
    expected_repo_sources = {str(item["path"]) for item in inventory_containers}
    if repo_source_items != expected_repo_sources:
        raise ValueError("combined status does not account every repo-local source container")

    required_ids = _required_materialization_ids(registrations)
    materializations = _pin_materializations(
        required_ids=required_ids,
        materialization_artifacts=materialization_artifacts,
    )
    leaves = _leaf_statuses(taxonomy=taxonomy, registrations=registrations)
    if len(leaves) != 25 or {item["leaf_id"] for item in leaves} != set(taxonomy.leaf_ids):
        raise ValueError("combined status must cover all 25 taxonomy leaves")

    role_counts = Counter(
        str(role) for registration in registrations for role in registration["roles"]
    )
    readiness_counts = Counter(str(item["readiness"]) for item in leaves)
    payload: dict[str, object] = {
        "schema_version": COMBINED_DATA_PLANE_SCHEMA,
        "status_id": "molor-expert-rack-combined-data-plane",
        "revision": "1",
        "contracts": {
            "taxonomy": _contract(
                path=taxonomy_file,
                artifact_id=taxonomy.taxonomy_id,
                revision=taxonomy.revision,
                semantic_sha256=taxonomy.fingerprint,
            ),
            "external_coverage": _contract(
                path=coverage_file,
                artifact_id=coverage.ledger_id,
                revision=coverage.revision,
                semantic_sha256=coverage.fingerprint,
            ),
            "external_assignment": _contract(
                path=assignment_file,
                artifact_id=matrix.matrix_id,
                revision=matrix.revision,
                semantic_sha256=matrix.fingerprint,
            ),
            "repo_local_inventory": _contract(
                path=inventory_file,
                artifact_id=str(inventory["inventory_id"]),
                revision=str(inventory["revision"]),
                semantic_sha256=str(inventory["content_sha256"]),
            ),
        },
        "materializations": materializations,
        "source_registrations": registrations,
        "leaves": leaves,
        "sampling_contract": {
            "unit": "unique-canonical-materialization-id",
            "alias_role_is_trainable": False,
            "evaluation_role_is_trainable": False,
            "quarantine_role_is_trainable": False,
            "cross_source_dedup_required_before_mixture_sampling": True,
            "derived_supervision_views": [
                {
                    "raw_materialization_id": "embedded-source-experts-v1",
                    "raw_modality": "reference_cpt",
                    "target_leaves": sorted(_EMBEDDED_RECONSTRUCTION_LEAVES),
                    "independent_chat_sampling": False,
                    "supervised_materialization_id": (
                        _EMBEDDED_GROUNDED_RECONSTRUCTION_ID
                    ),
                    "reason": (
                        "raw references remain accounted while the single supervised view "
                        "uses verbatim token-bounded reconstruction"
                    ),
                }
            ],
        },
        "accounting": {
            "taxonomy_leaves": len(leaves),
            "external_root_entries": len(expected_external),
            "external_root_entries_accounted": len(external_items),
            "repo_local_data_artifacts": len(expected_repo_data),
            "repo_local_data_artifacts_accounted": len(repo_data_items),
            "repo_local_source_containers": len(expected_repo_sources),
            "repo_local_source_containers_accounted": len(repo_source_items),
            "source_registrations": len(registrations),
            "registration_roles": dict(sorted(role_counts.items())),
            "leaf_readiness": dict(sorted(readiness_counts.items())),
            "canonical_training_materializations": len(required_ids),
            "canonical_training_materialization_ids": list(required_ids),
            "explicit_sft_gap_leaves": [
                str(item["leaf_id"]) for item in leaves if item["readiness"] != "sft-ready"
            ],
            "zero_silent_safe_source_omissions": True,
            "zero_double_counted_canonical_materializations": True,
            "zero_double_counted_derived_chat_views": True,
            "contains_corpus_content": False,
        },
    }
    payload["content_sha256"] = fingerprint_json(payload)
    return payload


def discover_required_materialization_ids(
    *,
    taxonomy_path: str | Path,
    coverage_path: str | Path,
    assignment_path: str | Path,
    repo_inventory_path: str | Path,
) -> tuple[str, ...]:
    """Discover aggregate manifests required by the current control-plane contracts."""

    taxonomy_file = _safe_file(taxonomy_path, "taxonomy_path")
    coverage_file = _safe_file(coverage_path, "coverage_path")
    assignment_file = _safe_file(assignment_path, "assignment_path")
    taxonomy = load_expert_taxonomy(taxonomy_file)
    coverage = load_corpus_coverage_ledger(coverage_file, taxonomy=taxonomy)
    matrix = load_source_expert_assignment_matrix(
        assignment_file,
        coverage=coverage,
        taxonomy=taxonomy,
    )
    _, inventory = _inventory(
        repo_inventory_path,
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_revision=taxonomy.revision,
        taxonomy_semantic_sha256=taxonomy.fingerprint,
    )
    registrations = _external_registrations(matrix) + _repo_registrations(inventory)
    return _required_materialization_ids(registrations)


def load_combined_data_plane_status(path: str | Path) -> dict[str, Any]:
    """Load and verify the self-authenticating combined status payload."""

    source = _safe_file(path, "status_path")
    value = _strict_json(source, "combined data-plane status")
    if value.get("schema_version") != COMBINED_DATA_PLANE_SCHEMA:
        raise ValueError("unsupported combined data-plane status schema")
    claimed = validate_sha256(value.get("content_sha256"), "status.content_sha256")
    if value.get("content_sha256") != claimed:
        raise ValueError("status content_sha256 must use lowercase hexadecimal")
    semantic = {key: item for key, item in value.items() if key != "content_sha256"}
    if fingerprint_json(semantic) != claimed:
        raise ValueError("combined data-plane status semantic fingerprint drift")
    if value.get("accounting", {}).get("contains_corpus_content") is not False:
        raise ValueError("combined data-plane status must remain aggregate-only")
    return value


def audit_combined_data_plane_status(
    *,
    status_path: str | Path,
    taxonomy_path: str | Path,
    coverage_path: str | Path,
    assignment_path: str | Path,
    repo_inventory_path: str | Path,
    materialization_artifacts: Mapping[str, str | Path],
) -> dict[str, object]:
    """Rebuild the expected aggregate view and return a content-addressed receipt."""

    status_file = _safe_file(status_path, "status_path")
    observed = load_combined_data_plane_status(status_file)
    expected = build_combined_data_plane_status(
        taxonomy_path=taxonomy_path,
        coverage_path=coverage_path,
        assignment_path=assignment_path,
        repo_inventory_path=repo_inventory_path,
        materialization_artifacts=materialization_artifacts,
    )
    if observed != expected:
        raise ValueError("combined data-plane status differs from its aggregate dependencies")
    accounting = observed["accounting"]
    receipt: dict[str, object] = {
        "schema_version": COMBINED_DATA_PLANE_RECEIPT_SCHEMA,
        "status": {
            "status_id": observed["status_id"],
            "revision": observed["revision"],
            "file_sha256": _sha256(status_file),
            "semantic_sha256": observed["content_sha256"],
        },
        "contracts": observed["contracts"],
        "sampling_contract": observed["sampling_contract"],
        "taxonomy_leaves": accounting["taxonomy_leaves"],
        "leaf_readiness": accounting["leaf_readiness"],
        "explicit_sft_gap_leaves": accounting["explicit_sft_gap_leaves"],
        "canonical_training_materialization_ids": accounting[
            "canonical_training_materialization_ids"
        ],
        "source_accounting": {
            "external_root_entries": accounting["external_root_entries"],
            "repo_local_data_artifacts": accounting["repo_local_data_artifacts"],
            "repo_local_source_containers": accounting["repo_local_source_containers"],
            "source_registrations": accounting["source_registrations"],
            "registration_roles": accounting["registration_roles"],
        },
        "zero_silent_safe_source_omissions": accounting["zero_silent_safe_source_omissions"],
        "zero_double_counted_canonical_materializations": accounting[
            "zero_double_counted_canonical_materializations"
        ],
        "zero_double_counted_derived_chat_views": accounting[
            "zero_double_counted_derived_chat_views"
        ],
        "contains_corpus_content": False,
    }
    receipt["content_sha256"] = fingerprint_json(receipt)
    return receipt


def write_json_artifact(value: Mapping[str, object], path: str | Path) -> None:
    destination = Path(path).expanduser()
    if _restricted(str(destination)):
        raise ValueError("output may not name a restricted path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(rendered)
    os.replace(temporary, destination)


def _parse_materializations(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        materialization_id, separator, raw_path = raw.partition("=")
        if not separator or not raw_path:
            raise ValueError("--materialization must use ID=PATH")
        parsed_id = _identifier(materialization_id, "--materialization ID")
        if parsed_id in result:
            raise ValueError(f"duplicate materialization ID {parsed_id!r}")
        result[parsed_id] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--repo-inventory", type=Path, required=True)
    parser.add_argument("--materialization", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    materializations = _parse_materializations(args.materialization)
    status = build_combined_data_plane_status(
        taxonomy_path=args.taxonomy,
        coverage_path=args.coverage,
        assignment_path=args.assignment,
        repo_inventory_path=args.repo_inventory,
        materialization_artifacts=materializations,
    )
    write_json_artifact(status, args.output)
    receipt = audit_combined_data_plane_status(
        status_path=args.output,
        taxonomy_path=args.taxonomy,
        coverage_path=args.coverage,
        assignment_path=args.assignment,
        repo_inventory_path=args.repo_inventory,
        materialization_artifacts=materializations,
    )
    write_json_artifact(receipt, args.receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "COMBINED_DATA_PLANE_RECEIPT_SCHEMA",
    "COMBINED_DATA_PLANE_SCHEMA",
    "audit_combined_data_plane_status",
    "build_combined_data_plane_status",
    "discover_required_materialization_ids",
    "load_combined_data_plane_status",
    "main",
    "write_json_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
