"""Top-level non-cyber corpus coverage ledger and read-only audit.

The audit intentionally does not recurse into datasets.  It proves that every eligible root-level
dataset artifact is named in an explicit lineage/disposition ledger while blocked roots are pruned
before any metadata or content access below them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

from molor.experimental.expert_taxonomy import ExpertTaxonomy, load_expert_taxonomy
from molor.identity import fingerprint_json, validate_sha256


CORPUS_COVERAGE_SCHEMA = "molor.corpus-coverage-ledger.v1"
CORPUS_COVERAGE_RECEIPT_SCHEMA = "molor.corpus-coverage-receipt.v1"
_BLOCKED_WORDS = ("cyber", "security")
_KINDS = frozenset({"directory", "file"})
_DISPOSITIONS = frozenset(
    {
        "canonical-container",
        "derived-container",
        "preparation-tooling",
        "raw-source",
        "reference-source",
        "source-database",
        "source-sidecar",
    }
)


def _blocked(value: str) -> bool:
    normalized = value.casefold()
    return any(word in normalized for word in _BLOCKED_WORDS)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"unsupported JSON constant {raw!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{field} fields differ from v1 "
            f"(missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)})"
        )


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{field} must be a non-empty string of at most 128 characters")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-._" for character in value):
        raise ValueError(f"{field} contains unsupported characters")
    return value


@dataclass(frozen=True, slots=True)
class CorpusCoverageEntry:
    path: str
    kind: str
    disposition: str
    canonical_dataset_id: str
    target_leaves: tuple[str, ...]
    note: str


@dataclass(frozen=True, slots=True)
class CorpusCoverageLedger:
    ledger_id: str
    revision: str
    dataset_root_label: str
    entries: tuple[CorpusCoverageEntry, ...]
    fingerprint: str


def load_corpus_coverage_ledger(
    path: str | Path,
    *,
    taxonomy: ExpertTaxonomy,
) -> CorpusCoverageLedger:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("coverage ledger must be a regular non-symlink file")
    value = _strict_json(source)
    _exact(
        value,
        {
            "schema_version",
            "ledger_id",
            "revision",
            "dataset_root_label",
            "entries",
            "content_sha256",
        },
        "ledger",
    )
    if value["schema_version"] != CORPUS_COVERAGE_SCHEMA:
        raise ValueError("unsupported corpus coverage schema")
    root_label = value["dataset_root_label"]
    if not isinstance(root_label, str) or not root_label or _blocked(root_label):
        raise ValueError("dataset_root_label must be non-empty and non-blocked")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("entries must be a non-empty array")
    known_leaves = set(taxonomy.leaf_ids)
    entries: list[CorpusCoverageEntry] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise TypeError(f"entries[{index}] must be an object")
        _exact(
            raw,
            {"path", "kind", "disposition", "canonical_dataset_id", "target_leaves", "note"},
            f"entries[{index}]",
        )
        raw_path = raw["path"]
        if not isinstance(raw_path, str) or not raw_path or _blocked(raw_path):
            raise ValueError(f"entries[{index}].path must be non-empty and non-blocked")
        pure = PurePath(raw_path)
        if pure.is_absolute() or len(pure.parts) != 1 or pure.parts[0] in {".", ".."}:
            raise ValueError(f"entries[{index}].path must name one direct child")
        kind = raw["kind"]
        disposition = raw["disposition"]
        if kind not in _KINDS:
            raise ValueError(f"entries[{index}].kind must be one of {sorted(_KINDS)}")
        if disposition not in _DISPOSITIONS:
            raise ValueError(f"entries[{index}].disposition must be one of {sorted(_DISPOSITIONS)}")
        dataset_id = _identifier(
            raw["canonical_dataset_id"], f"entries[{index}].canonical_dataset_id"
        )
        raw_leaves = raw["target_leaves"]
        if not isinstance(raw_leaves, list):
            raise TypeError(f"entries[{index}].target_leaves must be an array")
        leaves = tuple(raw_leaves)
        if (
            any(not isinstance(leaf, str) or leaf not in known_leaves for leaf in leaves)
            or len(leaves) != len(set(leaves))
            or list(leaves) != sorted(leaves)
        ):
            raise ValueError(
                f"entries[{index}].target_leaves must be unique canonical taxonomy leaves"
            )
        note = raw["note"]
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"entries[{index}].note must be non-empty")
        entries.append(
            CorpusCoverageEntry(
                path=raw_path,
                kind=str(kind),
                disposition=str(disposition),
                canonical_dataset_id=dataset_id,
                target_leaves=leaves,
                note=note.strip(),
            )
        )
    paths = [entry.path.casefold() for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("entries must have unique case-insensitive paths in lexical order")

    claimed = validate_sha256(value["content_sha256"], "content_sha256")  # type: ignore[arg-type]
    if value["content_sha256"] != claimed:
        raise ValueError("content_sha256 must use lowercase hexadecimal")
    semantic = {key: item for key, item in value.items() if key != "content_sha256"}
    fingerprint = fingerprint_json(semantic)
    if fingerprint != claimed:
        raise ValueError("coverage ledger content_sha256 does not match its semantic payload")
    return CorpusCoverageLedger(
        ledger_id=_identifier(value["ledger_id"], "ledger_id"),
        revision=_identifier(value["revision"], "revision"),
        dataset_root_label=root_label,
        entries=tuple(entries),
        fingerprint=fingerprint,
    )


def audit_corpus_coverage(
    *,
    dataset_root: str | Path,
    ledger_path: str | Path,
    taxonomy_path: str | Path,
) -> dict[str, object]:
    """Audit immediate dataset-root children without reading any corpus content."""

    root = Path(dataset_root).expanduser().resolve()
    if _blocked(str(root)):
        raise ValueError("dataset_root may not be a cyber/security path")
    if not root.is_dir():
        raise FileNotFoundError(root)
    taxonomy_file = Path(taxonomy_path).expanduser().resolve()
    ledger_file = Path(ledger_path).expanduser().resolve()
    taxonomy = load_expert_taxonomy(taxonomy_file)
    ledger = load_corpus_coverage_ledger(ledger_file, taxonomy=taxonomy)
    if root.name != ledger.dataset_root_label:
        raise ValueError("dataset root label differs from the coverage ledger")

    observed: dict[str, str] = {}
    blocked_count = 0
    symlink_count = 0
    with os.scandir(root) as children:
        for child in children:
            if _blocked(child.name):
                blocked_count += 1
                continue
            if child.is_symlink():
                symlink_count += 1
                continue
            if child.is_dir(follow_symlinks=False):
                kind = "directory"
            elif child.is_file(follow_symlinks=False):
                kind = "file"
            else:
                kind = "other"
            observed[child.name.casefold()] = kind
    if symlink_count:
        raise ValueError("eligible dataset-root symlinks are not permitted")

    expected = {entry.path.casefold(): entry for entry in ledger.entries}
    unlisted = sorted(set(observed).difference(expected))
    missing = sorted(set(expected).difference(observed))
    wrong_kind = sorted(
        expected[name].path
        for name in set(expected).intersection(observed)
        if observed[name] != expected[name].kind
    )
    if unlisted or missing or wrong_kind:
        raise ValueError(
            "corpus coverage drift "
            f"(unlisted={unlisted}, missing={missing}, wrong_kind={wrong_kind})"
        )

    dispositions = Counter(entry.disposition for entry in ledger.entries)
    canonical_datasets = Counter(entry.canonical_dataset_id for entry in ledger.entries)
    leaf_assignments = Counter(leaf for entry in ledger.entries for leaf in entry.target_leaves)
    return {
        "schema_version": CORPUS_COVERAGE_RECEIPT_SCHEMA,
        "dataset_root_label": root.name,
        "ledger": {
            "ledger_id": ledger.ledger_id,
            "revision": ledger.revision,
            "sha256": hashlib.sha256(ledger_file.read_bytes()).hexdigest(),
            "semantic_sha256": ledger.fingerprint,
        },
        "taxonomy": {
            "taxonomy_id": taxonomy.taxonomy_id,
            "revision": taxonomy.revision,
            "sha256": hashlib.sha256(taxonomy_file.read_bytes()).hexdigest(),
            "semantic_sha256": taxonomy.fingerprint,
        },
        "eligible_root_entries": len(observed),
        "blocked_root_entries_pruned_before_traversal": blocked_count,
        "dispositions": dict(sorted(dispositions.items())),
        "canonical_dataset_lineage": dict(sorted(canonical_datasets.items())),
        "target_leaf_assignments": dict(sorted(leaf_assignments.items())),
        "unlisted_eligible_entries": 0,
        "missing_ledger_entries": 0,
        "wrong_kind_entries": 0,
        "zero_silent_root_omissions": True,
        "contains_corpus_content": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = audit_corpus_coverage(
        dataset_root=args.dataset_root,
        ledger_path=args.ledger,
        taxonomy_path=args.taxonomy,
    )
    rendered = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.output is not None:
        destination = args.output.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(rendered)
        os.replace(temporary, destination)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
