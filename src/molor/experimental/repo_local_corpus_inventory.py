"""Inventory every eligible repo-local corpus artifact without parsing corpus rows.

The external dataset-root audit and the prepared-corpus manifest do not cover the historical
artifacts under ``experiments/first-real-experts``.  This module closes that accounting gap.  It
hashes and counts safe direct corpus artifacts, records the disposition of every source container,
and fails closed when an unknown eligible artifact appears.  Restricted names are pruned before
stat, hashing, or traversal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.identity import fingerprint_json


REPO_LOCAL_CORPUS_INVENTORY_SCHEMA = "molor.repo-local-corpus-inventory.v1"
_BLOCKED_WORDS = ("cyber", "security")
_ACTIVE_DISPOSITIONS = frozenset({"canonical", "supplemental"})


@dataclass(frozen=True, slots=True)
class CorpusGroup:
    prefix: str
    group_id: str
    canonical_dataset_id: str
    disposition: str
    target_leaves: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class SourceContainer:
    name: str
    group_id: str
    canonical_dataset_id: str
    disposition: str
    target_leaves: tuple[str, ...]
    note: str


_GROUPS: tuple[CorpusGroup, ...] = (
    CorpusGroup(
        "build_report",
        "legacy-multidomain-build",
        "legacy-first-real-experts-build-v1",
        "lineage-only",
        (),
        "Aggregate receipt for an older multi-domain build; it is not a trainable shard.",
    ),
    CorpusGroup(
        "customer-gtm",
        "multidogo-customer-support",
        "multidogo-customer-support-quarantine-v1",
        "quarantined",
        ("business-operations", "growth-gtm"),
        "Customer-support dialogs are not GTM expertise and remain excluded pending a privacy and duplication rebuild.",
    ),
    CorpusGroup(
        "embedded-can",
        "legacy-embedded-can",
        "legacy-embedded-can-v1",
        "superseded",
        ("embedded-can-diagnostics",),
        "Superseded by verified OCB/CAN materializations.",
    ),
    CorpusGroup(
        "embedded-knowledge-v1.seen-recall-probe",
        "embedded-knowledge-seen-probe",
        "embedded-knowledge-seen-recall-probe-v1",
        "evaluation-only",
        ("embedded-knowledge",),
        "Seen-recall diagnostic is explicitly not a held-out training source.",
    ),
    CorpusGroup(
        "embedded-knowledge-v1",
        "embedded-knowledge",
        "embedded-knowledge-v1",
        "supplemental",
        ("embedded-knowledge",),
        "Existing file-held-out embedded knowledge SFT remains an admitted supplement.",
    ),
    CorpusGroup(
        "financial-analysis-verified",
        "legacy-financial-analysis-verified",
        "financial-analysis-verified-v1",
        "superseded",
        ("business-finance", "investing-markets"),
        "Subset of FinQA/TAT-QA superseded by the full split-preserving finance supervision build.",
    ),
    CorpusGroup(
        "financial-analysis",
        "legacy-financial-analysis",
        "financial-analysis-v1",
        "historical-negative",
        ("business-finance", "investing-markets"),
        "Unverified predecessor retained only for comparison.",
    ),
    CorpusGroup(
        "grounded-can-curriculum",
        "grounded-can-curriculum",
        "grounded-can-curriculum-v1",
        "historical-negative",
        ("embedded-can-diagnostics",),
        "Recorded negative curriculum result; excluded from new mixtures.",
    ),
    CorpusGroup(
        "grounded-can",
        "grounded-can",
        "grounded-can-v1",
        "superseded",
        ("embedded-can-diagnostics",),
        "Narrow grounded predecessor superseded by the verified OCB/CAN build.",
    ),
    CorpusGroup(
        "local-embedded-code",
        "local-embedded-code",
        "local-embedded-code-v1",
        "supplemental",
        ("embedded-code-generation",),
        "File-separated local code continuation data remains an admitted supplement.",
    ),
    CorpusGroup(
        "ocb-can-verified",
        "ocb-can-verified",
        "ocb-can-verified-v1",
        "canonical",
        ("embedded-can-diagnostics",),
        "Verified and message-group-separated OCB/CAN supervision.",
    ),
    CorpusGroup(
        "router-bootstrap-v1",
        "legacy-router-v1",
        "router-bootstrap-v1",
        "superseded",
        (),
        "Legacy flat-router bootstrap superseded by later router curricula and the new taxonomy.",
    ),
    CorpusGroup(
        "router-bootstrap-v2",
        "legacy-router-v2",
        "router-bootstrap-v2",
        "superseded",
        (),
        "Legacy flat-router bootstrap superseded by later router curricula and the new taxonomy.",
    ),
    CorpusGroup(
        "router-bootstrap-v3-contrastive",
        "legacy-router-v3",
        "router-bootstrap-v3-contrastive",
        "superseded",
        (),
        "Legacy flat-router contrastive data does not match the granular hierarchy.",
    ),
    CorpusGroup(
        "router-bootstrap-v4-hard-negative-s307",
        "legacy-router-v4-s307",
        "router-bootstrap-v4-hard-negative-s307",
        "superseded",
        (),
        "Selected legacy flat-router dataset; labels do not match the granular hierarchy.",
    ),
    CorpusGroup(
        "router-bootstrap-v4-hard-negative",
        "legacy-router-v4",
        "router-bootstrap-v4-hard-negative",
        "superseded",
        (),
        "Non-selected legacy flat-router dataset.",
    ),
    CorpusGroup(
        "router-bootstrap-v5-verified-code-contrastive",
        "legacy-router-v5",
        "router-bootstrap-v5-verified-code-contrastive",
        "superseded",
        (),
        "Legacy flat-router curriculum; source prompts may be rematerialized only under the new labels.",
    ),
    CorpusGroup(
        "router-bootstrap-v6-taskmaster",
        "legacy-router-v6",
        "router-bootstrap-v6-taskmaster",
        "historical-negative",
        (),
        "Taskmaster flat-router registration failed its declared gates.",
    ),
    CorpusGroup(
        "router-bootstrap-v7-taskmaster-gate",
        "legacy-router-v7",
        "router-bootstrap-v7-taskmaster-gate",
        "historical-negative",
        (),
        "Taskmaster admission gate failed its declared one-shot gates.",
    ),
    CorpusGroup(
        "router-intent-probe-v1",
        "router-intent-probe-v1",
        "router-intent-probe-v1",
        "evaluation-only",
        (),
        "Previously used router probe; permanently excluded from training and tuning.",
    ),
    CorpusGroup(
        "router-intent-probe-v2",
        "router-intent-probe-v2",
        "router-intent-probe-v2",
        "evaluation-only",
        (),
        "Previously used router probe; permanently excluded from training and tuning.",
    ),
    CorpusGroup(
        "software-architecture-v1",
        "legacy-software-architecture-v1",
        "software-architecture-v1",
        "superseded",
        ("software-architecture-design",),
        "Balanced subset superseded by the full granular software materialization.",
    ),
    CorpusGroup(
        "software-architecture",
        "legacy-software-architecture",
        "software-architecture-legacy-v1",
        "superseded",
        ("software-architecture-design",),
        "Older architecture subset superseded by the repaired and full granular builds.",
    ),
    CorpusGroup(
        "taskmaster-coffee-api-v1",
        "taskmaster-coffee-api-v1",
        "taskmaster-coffee-api-v1",
        "historical-negative",
        ("tool-use-execution",),
        "First narrow candidate missed promotion gates and is superseded by v2.",
    ),
    CorpusGroup(
        "taskmaster-coffee-api-v2",
        "taskmaster-coffee-api-v2",
        "taskmaster-coffee-api-v2",
        "canonical",
        ("agentic-planning", "tool-use-execution"),
        "Component-disjoint full-dialog tool-use corpus; held-out splits remain evaluation-only.",
    ),
    CorpusGroup(
        "terminal-cli",
        "terminal-cli",
        "terminal-cli-v1",
        "canonical",
        ("terminal-cli",),
        "Task-family-separated terminal execution traces.",
    ),
    CorpusGroup(
        "trusted-can",
        "trusted-can",
        "trusted-can-v1",
        "historical-negative",
        ("embedded-can-diagnostics",),
        "Invalid lookup/memorization experiment retained only for historical accounting.",
    ),
    CorpusGroup(
        "verified-embedded-code-v1-plus-v3",
        "verified-embedded-code-union",
        "verified-embedded-code-v1-plus-v3",
        "superseded",
        ("embedded-code-generation",),
        "Materialized union is an alias of versioned constituents and must not be double counted.",
    ),
    CorpusGroup(
        "verified-embedded-code-v1",
        "verified-embedded-code-v1",
        "verified-embedded-code-v1",
        "superseded",
        ("embedded-code-generation",),
        "Compiler-verified predecessor retained as lineage for v3.",
    ),
    CorpusGroup(
        "verified-embedded-code-v2",
        "verified-embedded-code-v2",
        "verified-embedded-code-v2",
        "superseded",
        ("embedded-code-generation",),
        "Compiler-verified predecessor retained as lineage for v3.",
    ),
    CorpusGroup(
        "verified-embedded-code-v3",
        "verified-embedded-code-v3",
        "verified-embedded-code-v3",
        "supplemental",
        ("embedded-code-generation",),
        "Newest compiler-verified embedded-code curriculum remains an admitted supplement.",
    ),
)


_SOURCE_CONTAINERS: tuple[SourceContainer, ...] = (
    SourceContainer(
        "finqa",
        "repo-finqa-source",
        "finqa-upstream-source",
        "superseded",
        ("business-finance", "investing-markets"),
        "Repo-local source clone aliases the pinned acquisition snapshot used by the full finance build.",
    ),
    SourceContainer(
        "multidogo",
        "repo-multidogo-source",
        "multidogo-upstream-source",
        "quarantined",
        ("business-operations", "growth-gtm"),
        "Support transcripts are excluded pending a privacy-safe task-appropriate rebuild.",
    ),
    SourceContainer(
        "taskmaster",
        "repo-taskmaster-source",
        "taskmaster-upstream-source",
        "canonical",
        ("agentic-planning", "tool-use-execution"),
        "Pinned upstream source for the component-disjoint Taskmaster v2 materialization.",
    ),
    SourceContainer(
        "tat-qa",
        "repo-tat-qa-source",
        "tat-qa-upstream-source",
        "superseded",
        ("business-finance", "investing-markets"),
        "Repo-local source clone aliases the pinned acquisition snapshot used by the full finance build.",
    ),
)


def _blocked(value: str) -> bool:
    normalized = value.casefold()
    return any(word in normalized for word in _BLOCKED_WORDS)


def _safe_root(path: str | Path, field: str) -> Path:
    raw = str(path)
    if _blocked(raw):
        raise ValueError(f"{field} may not name a restricted path")
    root = Path(path).expanduser().resolve()
    if _blocked(str(root)):
        raise ValueError(f"{field} may not name a restricted path")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{field} must be a regular non-symlink directory")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> int | None:
    if path.suffix.casefold() != ".jsonl":
        return None
    rows = 0
    last = b""
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            rows += block.count(b"\n")
            last = block[-1:]
    if path.stat().st_size and last != b"\n":
        rows += 1
    return rows


def _group_for(filename: str) -> CorpusGroup:
    lowered = filename.casefold()
    matches = [
        group
        for group in _GROUPS
        if lowered == group.prefix.casefold() or lowered.startswith(group.prefix.casefold() + ".")
    ]
    if not matches:
        raise ValueError(f"eligible repo-local artifact has no unique disposition: {filename!r}")
    matches.sort(key=lambda item: len(item.prefix), reverse=True)
    if len(matches) > 1 and len(matches[0].prefix) == len(matches[1].prefix):
        raise ValueError(f"eligible repo-local artifact has ambiguous disposition: {filename!r}")
    return matches[0]


def _artifact_role(filename: str) -> str:
    lowered = filename.casefold()
    if lowered.endswith(".quarantine.jsonl"):
        return "quarantine"
    if lowered.endswith(".provenance.jsonl"):
        return "provenance"
    if lowered.endswith((".report.json", ".build-report.json", ".manifest.json")):
        return "report"
    if lowered.endswith((".token-lengths.json", ".tool-catalog.json")):
        return "auxiliary"
    if "probe" in lowered or ".validation" in lowered or ".test" in lowered:
        return "evaluation"
    if lowered.endswith(".jsonl"):
        return "training"
    if lowered.endswith(".json"):
        return "auxiliary"
    raise ValueError(f"unsupported repo-local artifact type: {filename!r}")


def _container_metadata(root: Path) -> tuple[int, int, int]:
    file_count = 0
    total_bytes = 0
    blocked_count = 0
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                if _blocked(child.name):
                    blocked_count += 1
                    continue
                if child.is_symlink():
                    raise ValueError("eligible source-container symlinks are not permitted")
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                elif child.is_file(follow_symlinks=False):
                    stat = child.stat(follow_symlinks=False)
                    file_count += 1
                    total_bytes += stat.st_size
                else:
                    raise ValueError("unsupported source-container entry type")
    return file_count, total_bytes, blocked_count


def build_repo_local_corpus_inventory(
    *,
    data_root: str | Path,
    source_root: str | Path,
    taxonomy_path: str | Path,
) -> dict[str, object]:
    """Return a content-free, self-authenticating inventory of safe repo-local corpora."""

    data = _safe_root(data_root, "data_root")
    sources = _safe_root(source_root, "source_root")
    taxonomy_file = Path(taxonomy_path).expanduser().resolve()
    if _blocked(str(taxonomy_file)) or taxonomy_file.is_symlink() or not taxonomy_file.is_file():
        raise ValueError("taxonomy_path must be a safe regular file")
    taxonomy = load_expert_taxonomy(taxonomy_file)
    known_leaves = set(taxonomy.leaf_ids)
    declared_leaves = {leaf for group in _GROUPS for leaf in group.target_leaves} | {
        leaf for container in _SOURCE_CONTAINERS for leaf in container.target_leaves
    }
    unknown_leaves = sorted(declared_leaves.difference(known_leaves))
    if unknown_leaves:
        raise ValueError(f"inventory rules reference unknown taxonomy leaves: {unknown_leaves}")

    artifacts: list[dict[str, object]] = []
    blocked_data_entries = 0
    with os.scandir(data) as children:
        for child in children:
            if _blocked(child.name):
                blocked_data_entries += 1
                continue
            if child.is_symlink() or not child.is_file(follow_symlinks=False):
                raise ValueError(f"eligible data-root entry must be a regular file: {child.name!r}")
            path = Path(child.path)
            group = _group_for(child.name)
            role = _artifact_role(child.name)
            stat = child.stat(follow_symlinks=False)
            admitted = group.disposition in _ACTIVE_DISPOSITIONS and role == "training"
            artifacts.append(
                {
                    "path": child.name,
                    "bytes": stat.st_size,
                    "sha256": _sha256(path),
                    "jsonl_rows": _jsonl_rows(path),
                    "artifact_role": role,
                    "group_id": group.group_id,
                    "canonical_dataset_id": group.canonical_dataset_id,
                    "dataset_disposition": group.disposition,
                    "target_leaves": list(group.target_leaves),
                    "admitted_for_new_expert_training": admitted,
                    "note": group.note,
                }
            )
    artifacts.sort(key=lambda item: str(item["path"]).casefold())

    expected_sources = {item.name.casefold(): item for item in _SOURCE_CONTAINERS}
    observed_sources: dict[str, os.DirEntry[str]] = {}
    blocked_source_entries = 0
    with os.scandir(sources) as children:
        for child in children:
            if _blocked(child.name):
                blocked_source_entries += 1
                continue
            observed_sources[child.name.casefold()] = child
    missing_sources = sorted(set(expected_sources).difference(observed_sources))
    unknown_sources = sorted(set(observed_sources).difference(expected_sources))
    if missing_sources or unknown_sources:
        raise ValueError(
            "repo-local source-container drift "
            f"(missing={missing_sources}, unknown={unknown_sources})"
        )
    containers: list[dict[str, object]] = []
    nested_blocked = 0
    for key, definition in sorted(expected_sources.items()):
        child = observed_sources[key]
        if child.is_symlink() or not child.is_dir(follow_symlinks=False):
            raise ValueError(f"source container must be a regular directory: {child.name!r}")
        file_count, total_bytes, blocked = _container_metadata(Path(child.path))
        nested_blocked += blocked
        containers.append(
            {
                "path": definition.name,
                "group_id": definition.group_id,
                "canonical_dataset_id": definition.canonical_dataset_id,
                "dataset_disposition": definition.disposition,
                "target_leaves": list(definition.target_leaves),
                "eligible_nested_files": file_count,
                "eligible_nested_bytes": total_bytes,
                "note": definition.note,
            }
        )

    role_counts = Counter(str(item["artifact_role"]) for item in artifacts)
    disposition_counts = Counter(str(item["dataset_disposition"]) for item in artifacts)
    admitted_rows = sum(
        int(item["jsonl_rows"] or 0)
        for item in artifacts
        if item["admitted_for_new_expert_training"]
    )
    payload: dict[str, object] = {
        "schema_version": REPO_LOCAL_CORPUS_INVENTORY_SCHEMA,
        "inventory_id": "first-real-experts-repo-local-v1",
        "revision": "1",
        "taxonomy": {
            "taxonomy_id": taxonomy.taxonomy_id,
            "revision": taxonomy.revision,
            "semantic_sha256": taxonomy.fingerprint,
            "file_sha256": _sha256(taxonomy_file),
        },
        "roots": {"data": data.name, "sources": sources.name},
        "restricted_entry_policy": "prune-before-stat-hash-or-traversal",
        "artifacts": artifacts,
        "source_containers": containers,
        "summary": {
            "eligible_data_artifacts": len(artifacts),
            "eligible_data_bytes": sum(int(item["bytes"]) for item in artifacts),
            "eligible_jsonl_rows": sum(int(item["jsonl_rows"] or 0) for item in artifacts),
            "admitted_training_rows_before_cross-corpus-dedup": admitted_rows,
            "artifact_roles": dict(sorted(role_counts.items())),
            "dataset_dispositions": dict(sorted(disposition_counts.items())),
            "eligible_source_containers": len(containers),
            "eligible_source_nested_files": sum(
                int(item["eligible_nested_files"]) for item in containers
            ),
            "eligible_source_nested_bytes": sum(
                int(item["eligible_nested_bytes"]) for item in containers
            ),
            "restricted_data_entries_pruned": blocked_data_entries,
            "restricted_source_entries_pruned": blocked_source_entries,
            "restricted_nested_source_entries_pruned": nested_blocked,
            "unlisted_eligible_data_artifacts": 0,
            "unlisted_eligible_source_containers": 0,
            "zero_silent_eligible_omissions": True,
            "contains_corpus_content": False,
        },
    }
    payload["content_sha256"] = fingerprint_json(payload)
    return payload


def write_repo_local_corpus_inventory(
    inventory: Mapping[str, object], output_path: str | Path
) -> None:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(rendered)
    os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inventory = build_repo_local_corpus_inventory(
        data_root=args.data_root,
        source_root=args.source_root,
        taxonomy_path=args.taxonomy,
    )
    write_repo_local_corpus_inventory(inventory, args.output)
    print(json.dumps(inventory["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "REPO_LOCAL_CORPUS_INVENTORY_SCHEMA",
    "build_repo_local_corpus_inventory",
    "main",
    "write_repo_local_corpus_inventory",
]
