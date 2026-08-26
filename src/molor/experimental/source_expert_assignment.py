"""Validate a content-addressed source-to-expert assignment matrix.

The matrix joins two aggregate artifacts only: the top-level corpus coverage ledger and the
expert taxonomy.  It never opens corpus roots or rows.  Every covered root must appear exactly
once, every source-to-leaf edge has one corpus role and one materialization status, and taxonomy
leaves without an effective source are recorded as explicit gaps.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

from molor.experimental.corpus_coverage import (
    CorpusCoverageLedger,
    load_corpus_coverage_ledger,
)
from molor.experimental.expert_taxonomy import ExpertTaxonomy, load_expert_taxonomy
from molor.identity import fingerprint_json, validate_sha256


SOURCE_EXPERT_ASSIGNMENT_SCHEMA = "molor.source-expert-assignment.v1"
SOURCE_EXPERT_ASSIGNMENT_RECEIPT_SCHEMA = "molor.source-expert-assignment-receipt.v1"

_BLOCKED_WORDS = ("cyber", "security")
_ROLES = frozenset(
    {
        "alias",
        "auxiliary",
        "evaluation-only",
        "quarantine/reject",
        "reference/pretraining",
        "sft",
    }
)
_MATERIALIZATION_STATUSES = frozenset(
    {"deferred", "gap", "materializing", "not-applicable", "ready"}
)
_ROLE_STATUSES = {
    "alias": frozenset({"not-applicable"}),
    "auxiliary": frozenset({"deferred", "gap", "ready"}),
    "evaluation-only": frozenset({"gap", "materializing", "ready"}),
    "quarantine/reject": frozenset({"deferred"}),
    "reference/pretraining": frozenset({"gap", "materializing", "ready"}),
    "sft": frozenset({"gap", "materializing", "ready"}),
}
_ADMINISTRATIVE_DISPOSITIONS = frozenset(
    {"canonical-container", "derived-container", "preparation-tooling"}
)
_NON_EFFECTIVE_ROLES = frozenset({"alias", "quarantine/reject"})


def _blocked(value: str) -> bool:
    normalized = value.casefold()
    return any(word in normalized for word in _BLOCKED_WORDS)


def _safe_artifact_path(value: str | Path, field: str) -> Path:
    raw = str(value)
    if _blocked(raw):
        raise ValueError(f"{field} may not name a blocked path")
    path = Path(value).expanduser()
    if _blocked(str(path)):
        raise ValueError(f"{field} may not name a blocked path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file")
    return path


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != before.st_size
    ):
        raise RuntimeError("assignment matrix changed while it was read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"unsupported JSON constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("assignment matrix must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("assignment matrix must contain one JSON object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{field} fields differ from v1 "
            f"(missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)})"
        )


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ValueError(f"{field} must be a non-empty string of at most 160 characters")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-._/" for character in value):
        raise ValueError(f"{field} contains unsupported characters")
    return value


def _strings(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    identifiers: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{field} must be an array with at least {minimum} item(s)")
    result: list[str] = []
    for index, item in enumerate(value):
        if identifiers:
            result.append(_identifier(item, f"{field}[{index}]"))
        elif not isinstance(item, str) or not item:
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        else:
            result.append(item)
    if len(result) != len(set(result)) or result != sorted(result, key=str.casefold):
        raise ValueError(f"{field} must be unique and use canonical lexical ordering")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AssignmentPolicy:
    target_leaves: tuple[str, ...]
    role: str
    materialization_status: str
    materialization_ids: tuple[str, ...]
    gap_reason: str | None


@dataclass(frozen=True, slots=True)
class SourceAssignmentGroup:
    group_id: str
    source_paths: tuple[str, ...]
    policies: tuple[AssignmentPolicy, ...]
    note: str

    @property
    def target_leaves(self) -> tuple[str, ...]:
        return tuple(
            sorted({leaf for policy in self.policies for leaf in policy.target_leaves})
        )


@dataclass(frozen=True, slots=True)
class TaxonomyLeafGap:
    leaf_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class SourceExpertAssignmentMatrix:
    matrix_id: str
    revision: str
    coverage_contract: Mapping[str, str]
    taxonomy_contract: Mapping[str, str]
    source_groups: tuple[SourceAssignmentGroup, ...]
    taxonomy_leaf_gaps: tuple[TaxonomyLeafGap, ...]
    fingerprint: str


def _parse_policy(
    raw: object,
    field: str,
    *,
    known_leaves: set[str],
) -> AssignmentPolicy:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{field} must be an object")
    _exact(
        raw,
        {
            "target_leaves",
            "role",
            "materialization_status",
            "materialization_ids",
            "gap_reason",
        },
        field,
    )
    leaves = _strings(raw["target_leaves"], f"{field}.target_leaves", minimum=1)
    unknown = sorted(set(leaves).difference(known_leaves))
    if unknown:
        raise ValueError(f"{field}.target_leaves contains unknown leaves: {unknown}")
    role = raw["role"]
    if role not in _ROLES:
        raise ValueError(f"{field}.role must be one of {sorted(_ROLES)}")
    status = raw["materialization_status"]
    if status not in _MATERIALIZATION_STATUSES:
        raise ValueError(
            f"{field}.materialization_status must be one of {sorted(_MATERIALIZATION_STATUSES)}"
        )
    if status not in _ROLE_STATUSES[str(role)]:
        raise ValueError(f"{field} has incompatible role/materialization status")
    materialization_ids = _strings(raw["materialization_ids"], f"{field}.materialization_ids")
    gap_reason = raw["gap_reason"]
    if status in {"deferred", "gap", "materializing"}:
        if not isinstance(gap_reason, str) or not gap_reason.strip():
            raise ValueError(f"{field}.gap_reason must explain non-ready materialization")
        gap_reason = gap_reason.strip()
    elif gap_reason is not None:
        raise ValueError(f"{field}.gap_reason must be null for ready/not-applicable status")
    if role == "alias" and materialization_ids:
        raise ValueError(f"{field}.materialization_ids must be empty for aliases")
    if role != "alias" and not materialization_ids:
        raise ValueError(f"{field}.materialization_ids must identify current or planned output")
    return AssignmentPolicy(
        target_leaves=leaves,
        role=str(role),
        materialization_status=str(status),
        materialization_ids=materialization_ids,
        gap_reason=gap_reason,
    )


def _parse_group(
    raw: object,
    field: str,
    *,
    known_leaves: set[str],
) -> SourceAssignmentGroup:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{field} must be an object")
    _exact(raw, {"group_id", "source_paths", "policies", "note"}, field)
    group_id = _identifier(raw["group_id"], f"{field}.group_id")
    source_paths = _strings(
        raw["source_paths"], f"{field}.source_paths", minimum=1, identifiers=False
    )
    for source_path in source_paths:
        if _blocked(source_path):
            raise ValueError(f"{field}.source_paths may not contain blocked names")
        pure = PurePath(source_path)
        if pure.is_absolute() or len(pure.parts) != 1 or pure.parts[0] in {".", ".."}:
            raise ValueError(f"{field}.source_paths must name direct root children")
    raw_policies = raw["policies"]
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError(f"{field}.policies must be a non-empty array")
    policies = tuple(
        _parse_policy(item, f"{field}.policies[{index}]", known_leaves=known_leaves)
        for index, item in enumerate(raw_policies)
    )
    policy_keys = [
        (
            policy.target_leaves[0],
            policy.role,
            policy.materialization_status,
            policy.materialization_ids,
        )
        for policy in policies
    ]
    if policy_keys != sorted(policy_keys):
        raise ValueError(f"{field}.policies must use canonical ordering")
    role_edges = [
        (leaf, policy.role) for policy in policies for leaf in policy.target_leaves
    ]
    if len(role_edges) != len(set(role_edges)):
        raise ValueError(f"{field}.policies may assign each leaf/role pair only once")
    note = raw["note"]
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{field}.note must be non-empty")
    return SourceAssignmentGroup(group_id, source_paths, policies, note.strip())


def _contract(
    raw: object,
    field: str,
    *,
    identifier_key: str,
) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{field} must be an object")
    _exact(raw, {identifier_key, "revision", "semantic_sha256"}, field)
    result = {
        identifier_key: _identifier(raw[identifier_key], f"{field}.{identifier_key}"),
        "revision": _identifier(raw["revision"], f"{field}.revision"),
        "semantic_sha256": validate_sha256(
            raw["semantic_sha256"],
            f"{field}.semantic_sha256",  # type: ignore[arg-type]
        ),
    }
    if raw["semantic_sha256"] != result["semantic_sha256"]:
        raise ValueError(f"{field}.semantic_sha256 must use lowercase hexadecimal")
    return result


def load_source_expert_assignment_matrix(
    path: str | Path,
    *,
    coverage: CorpusCoverageLedger,
    taxonomy: ExpertTaxonomy,
) -> SourceExpertAssignmentMatrix:
    """Load and validate one immutable aggregate-only assignment matrix."""

    source = _safe_artifact_path(path, "assignment_path")
    value = _strict_json(source)
    _exact(
        value,
        {
            "schema_version",
            "matrix_id",
            "revision",
            "coverage_contract",
            "taxonomy_contract",
            "source_groups",
            "taxonomy_leaf_gaps",
            "content_sha256",
        },
        "matrix",
    )
    if value["schema_version"] != SOURCE_EXPERT_ASSIGNMENT_SCHEMA:
        raise ValueError("unsupported source-expert assignment schema")
    coverage_contract = _contract(
        value["coverage_contract"], "coverage_contract", identifier_key="ledger_id"
    )
    expected_coverage = {
        "ledger_id": coverage.ledger_id,
        "revision": coverage.revision,
        "semantic_sha256": coverage.fingerprint,
    }
    if coverage_contract != expected_coverage:
        raise ValueError("coverage_contract does not pin the supplied coverage ledger")
    taxonomy_contract = _contract(
        value["taxonomy_contract"], "taxonomy_contract", identifier_key="taxonomy_id"
    )
    expected_taxonomy = {
        "taxonomy_id": taxonomy.taxonomy_id,
        "revision": taxonomy.revision,
        "semantic_sha256": taxonomy.fingerprint,
    }
    if taxonomy_contract != expected_taxonomy:
        raise ValueError("taxonomy_contract does not pin the supplied expert taxonomy")

    raw_groups = value["source_groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("source_groups must be a non-empty array")
    known_leaves = set(taxonomy.leaf_ids)
    groups = tuple(
        _parse_group(item, f"source_groups[{index}]", known_leaves=known_leaves)
        for index, item in enumerate(raw_groups)
    )
    group_ids = [group.group_id for group in groups]
    if group_ids != sorted(group_ids) or len(group_ids) != len(set(group_ids)):
        raise ValueError("source_groups must have unique, lexically ordered group IDs")
    all_paths = [path for group in groups for path in group.source_paths]
    folded_paths = [path.casefold() for path in all_paths]
    if len(folded_paths) != len(set(folded_paths)):
        raise ValueError("each covered source path must occur in exactly one source group")

    raw_gaps = value["taxonomy_leaf_gaps"]
    if not isinstance(raw_gaps, list):
        raise TypeError("taxonomy_leaf_gaps must be an array")
    leaf_gaps: list[TaxonomyLeafGap] = []
    for index, raw_gap in enumerate(raw_gaps):
        field = f"taxonomy_leaf_gaps[{index}]"
        if not isinstance(raw_gap, Mapping):
            raise TypeError(f"{field} must be an object")
        _exact(raw_gap, {"leaf_id", "reason"}, field)
        leaf_id = _identifier(raw_gap["leaf_id"], f"{field}.leaf_id")
        if leaf_id not in known_leaves:
            raise ValueError(f"{field}.leaf_id is not a canonical taxonomy leaf")
        reason = raw_gap["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{field}.reason must be non-empty")
        leaf_gaps.append(TaxonomyLeafGap(leaf_id, reason.strip()))
    gap_ids = [gap.leaf_id for gap in leaf_gaps]
    if gap_ids != sorted(gap_ids) or len(gap_ids) != len(set(gap_ids)):
        raise ValueError("taxonomy_leaf_gaps must be unique and ordered by leaf_id")

    claimed = validate_sha256(value["content_sha256"], "content_sha256")  # type: ignore[arg-type]
    if value["content_sha256"] != claimed:
        raise ValueError("content_sha256 must use lowercase hexadecimal")
    semantic = {key: item for key, item in value.items() if key != "content_sha256"}
    fingerprint = fingerprint_json(semantic)
    if fingerprint != claimed:
        raise ValueError("assignment matrix content_sha256 does not match its semantic payload")

    matrix = SourceExpertAssignmentMatrix(
        matrix_id=_identifier(value["matrix_id"], "matrix_id"),
        revision=_identifier(value["revision"], "revision"),
        coverage_contract=coverage_contract,
        taxonomy_contract=taxonomy_contract,
        source_groups=groups,
        taxonomy_leaf_gaps=tuple(leaf_gaps),
        fingerprint=fingerprint,
    )
    _validate_coverage_join(matrix, coverage=coverage, taxonomy=taxonomy)
    return matrix


def _validate_coverage_join(
    matrix: SourceExpertAssignmentMatrix,
    *,
    coverage: CorpusCoverageLedger,
    taxonomy: ExpertTaxonomy,
) -> None:
    expected = {entry.path.casefold(): entry for entry in coverage.entries}
    observed: dict[str, SourceAssignmentGroup] = {}
    for group in matrix.source_groups:
        group_leaves = set(group.target_leaves)
        for source_path in group.source_paths:
            key = source_path.casefold()
            if key not in expected:
                raise ValueError(f"assignment coverage drift: unlisted source {source_path!r}")
            entry = expected[key]
            if entry.path != source_path:
                raise ValueError(f"assignment source case differs from coverage: {source_path!r}")
            if key in observed:
                raise ValueError(f"assignment source is duplicated: {source_path!r}")
            observed[key] = group
            expected_leaves = set(entry.target_leaves)
            if expected_leaves:
                if group_leaves != expected_leaves:
                    raise ValueError(
                        f"assignment leaf drift for {source_path!r}: "
                        f"expected={sorted(expected_leaves)}, observed={sorted(group_leaves)}"
                    )
            else:
                if entry.disposition not in _ADMINISTRATIVE_DISPOSITIONS:
                    raise ValueError(
                        f"target-free coverage entry {source_path!r} is not administrative"
                    )
                if any(
                    policy.role != "alias" or policy.materialization_status != "not-applicable"
                    for policy in group.policies
                ):
                    raise ValueError(
                        f"administrative coverage entry {source_path!r} must be alias-only"
                    )
            if entry.disposition == "source-sidecar" and any(
                policy.role != "alias" for policy in group.policies
            ):
                raise ValueError(f"source sidecar {source_path!r} must be alias-only")
    missing = sorted(entry.path for key, entry in expected.items() if key not in observed)
    if missing:
        raise ValueError(f"assignment coverage drift: missing sources {missing}")

    effective_leaves = {
        leaf
        for group in matrix.source_groups
        for policy in group.policies
        if policy.role not in _NON_EFFECTIVE_ROLES
        for leaf in policy.target_leaves
    }
    expected_gaps = set(taxonomy.leaf_ids).difference(effective_leaves)
    observed_gaps = {gap.leaf_id for gap in matrix.taxonomy_leaf_gaps}
    if observed_gaps != expected_gaps:
        raise ValueError(
            "taxonomy leaf gap drift "
            f"(missing={sorted(expected_gaps - observed_gaps)}, "
            f"extra={sorted(observed_gaps - expected_gaps)})"
        )


def audit_source_expert_assignments(
    *,
    assignment_path: str | Path,
    coverage_path: str | Path,
    taxonomy_path: str | Path,
) -> dict[str, object]:
    """Return a content-free coverage, role, readiness, and materialization-gap receipt."""

    assignment_file = _safe_artifact_path(assignment_path, "assignment_path")
    coverage_file = _safe_artifact_path(coverage_path, "coverage_path")
    taxonomy_file = _safe_artifact_path(taxonomy_path, "taxonomy_path")
    taxonomy = load_expert_taxonomy(taxonomy_file)
    coverage = load_corpus_coverage_ledger(coverage_file, taxonomy=taxonomy)
    matrix = load_source_expert_assignment_matrix(
        assignment_file,
        coverage=coverage,
        taxonomy=taxonomy,
    )

    role_rows: Counter[str] = Counter()
    status_rows: Counter[str] = Counter()
    role_sources: dict[str, set[str]] = {role: set() for role in _ROLES}
    status_sources: dict[str, set[str]] = {status: set() for status in _MATERIALIZATION_STATUSES}
    materialization_ids: Counter[str] = Counter()
    effective_leaves: set[str] = set()
    ready_leaves: set[str] = set()
    for group in matrix.source_groups:
        for policy in group.policies:
            rows = len(group.source_paths) * len(policy.target_leaves)
            role_rows[policy.role] += rows
            status_rows[policy.materialization_status] += rows
            role_sources[policy.role].update(group.source_paths)
            status_sources[policy.materialization_status].update(group.source_paths)
            for materialization_id in policy.materialization_ids:
                materialization_ids[materialization_id] += rows
            if policy.role not in _NON_EFFECTIVE_ROLES:
                effective_leaves.update(policy.target_leaves)
                if policy.materialization_status == "ready":
                    ready_leaves.update(policy.target_leaves)

    coverage_by_path = {entry.path: entry for entry in coverage.entries}
    gap_sources = status_sources["gap"]
    materializing_sources = status_sources["materializing"]
    gap_datasets = sorted({coverage_by_path[path].canonical_dataset_id for path in gap_sources})
    materializing_datasets = sorted(
        {coverage_by_path[path].canonical_dataset_id for path in materializing_sources}
    )
    effective_sources = {
        source_path
        for group in matrix.source_groups
        if any(policy.role not in _NON_EFFECTIVE_ROLES for policy in group.policies)
        for source_path in group.source_paths
    }
    alias_only_sources = set(coverage_by_path).difference(effective_sources)

    return {
        "schema_version": SOURCE_EXPERT_ASSIGNMENT_RECEIPT_SCHEMA,
        "matrix": {
            "matrix_id": matrix.matrix_id,
            "revision": matrix.revision,
            "sha256": hashlib.sha256(assignment_file.read_bytes()).hexdigest(),
            "semantic_sha256": matrix.fingerprint,
        },
        "coverage": dict(matrix.coverage_contract),
        "taxonomy": dict(matrix.taxonomy_contract),
        "source_groups": len(matrix.source_groups),
        "covered_root_entries": len(coverage.entries),
        "effective_source_entries": len(effective_sources),
        "administrative_or_lineage_alias_entries": len(alias_only_sources),
        "assignment_rows": sum(role_rows.values()),
        "effective_assignment_rows": sum(
            count for role, count in role_rows.items() if role not in _NON_EFFECTIVE_ROLES
        ),
        "role_assignment_rows": dict(sorted(role_rows.items())),
        "role_source_entries": {
            role: len(paths) for role, paths in sorted(role_sources.items()) if paths
        },
        "materialization_status_rows": dict(sorted(status_rows.items())),
        "materialization_status_source_entries": {
            status: len(paths) for status, paths in sorted(status_sources.items()) if paths
        },
        "materialization_ids": dict(sorted(materialization_ids.items())),
        "materialization_gap_source_entries": len(gap_sources),
        "materialization_gap_canonical_datasets": gap_datasets,
        "materializing_source_entries": len(materializing_sources),
        "materializing_canonical_datasets": materializing_datasets,
        "canonical_taxonomy_leaves": len(taxonomy.leaf_ids),
        "effectively_assigned_leaves": len(effective_leaves),
        "ready_leaves": len(ready_leaves),
        "assigned_but_not_ready_leaves": len(effective_leaves.difference(ready_leaves)),
        "unassigned_leaf_gaps": [gap.leaf_id for gap in matrix.taxonomy_leaf_gaps],
        "zero_silent_root_omissions": True,
        "contains_corpus_content": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = audit_source_expert_assignments(
        assignment_path=args.assignment,
        coverage_path=args.coverage,
        taxonomy_path=args.taxonomy,
    )
    rendered = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.output is not None:
        destination = args.output.expanduser()
        if _blocked(str(destination)):
            raise ValueError("output may not name a blocked path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(rendered)
        os.replace(temporary, destination)
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "SOURCE_EXPERT_ASSIGNMENT_RECEIPT_SCHEMA",
    "SOURCE_EXPERT_ASSIGNMENT_SCHEMA",
    "AssignmentPolicy",
    "SourceAssignmentGroup",
    "SourceExpertAssignmentMatrix",
    "TaxonomyLeafGap",
    "audit_source_expert_assignments",
    "load_source_expert_assignment_matrix",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
