"""Content-addressed receipts for private business-corpus source snapshots.

The acquisition itself may use HTTP or Git, but this module is deliberately read-only with
respect to source material.  It inventories bytes and hashes without parsing or copying source
content, and emits no excerpts.  Cyber/security-named trees are pruned before traversal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable, Sequence


SCHEMA_VERSION = "molor.business-source-snapshot.v1"
DEFAULT_RECEIPT_NAME = "acquisition-manifest.json"
DEFAULT_SUMS_NAME = "sha256sums.txt"
_BLOCKED_PATH_WORDS = ("cyber", "security")
_GENERATED_NAMES = frozenset({DEFAULT_RECEIPT_NAME, DEFAULT_SUMS_NAME})


def _blocked(parts: Iterable[str]) -> bool:
    return any(word in part.casefold() for part in parts for word in _BLOCKED_PATH_WORDS)


@dataclass(frozen=True, slots=True)
class SnapshotSource:
    """One acquired source tree and its intended business leaves."""

    source_id: str
    relative_root: str
    origins: tuple[str, ...]
    target_leaves: tuple[str, ...]
    expected_files: int | None = None
    git_commit_required: bool = False
    known_missing: tuple[str, ...] = ()
    partition_policy: str = "deterministic_artifact_hash_90_5_5"

    def __post_init__(self) -> None:
        if not self.source_id or not self.relative_root:
            raise ValueError("source_id and relative_root are required")
        relative = Path(self.relative_root)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_root must stay below the snapshot root")
        if _blocked(relative.parts):
            raise ValueError("cyber/security source trees are outside this receipt")
        if not self.origins or any(not value for value in self.origins):
            raise ValueError("origins must contain non-empty URLs")
        if not self.target_leaves or any(not value for value in self.target_leaves):
            raise ValueError("target_leaves must contain non-empty leaf IDs")
        if self.expected_files is not None and (
            not isinstance(self.expected_files, int)
            or isinstance(self.expected_files, bool)
            or self.expected_files < 1
        ):
            raise ValueError("expected_files must be a positive integer or null")
        if not self.partition_policy:
            raise ValueError("partition_policy must be non-empty")


DEFAULT_SOURCES = (
    SnapshotSource(
        source_id="fdic-sba-money-smart-small-business",
        relative_root="fdic-money-smart-small-business",
        origins=(
            "https://www.fdic.gov/consumer-resource-center/money-smart-small-business",
            "https://www.fdic.gov/resources/consumers/money-smart/"
            "money-smart-for-small-business/documents/mssb-modules-list.csv",
        ),
        target_leaves=(
            "business-core",
            "business-finance",
            "funding-capital",
            "business-operations",
        ),
        expected_files=45,
    ),
    SnapshotSource(
        source_id="gitlab-company-handbook",
        relative_root="gitlab-handbook",
        origins=("https://gitlab.com/gitlab-com/content-sites/handbook.git",),
        target_leaves=(
            "business-core",
            "business-finance",
            "growth-gtm",
            "business-operations",
        ),
        expected_files=2043,
        git_commit_required=True,
    ),
    SnapshotSource(
        source_id="onet-business-work-activities",
        relative_root="onet-31.0",
        origins=("https://www.onetcenter.org/dl_files/database/db_31_0_json.zip",),
        target_leaves=("business-core", "growth-gtm", "business-operations"),
        expected_files=1,
    ),
    SnapshotSource(
        source_id="posthog-handbook-and-manual",
        relative_root="posthog-handbook",
        origins=("https://github.com/PostHog/posthog.com.git",),
        target_leaves=("business-core", "growth-gtm", "business-operations"),
        expected_files=377,
        git_commit_required=True,
    ),
    SnapshotSource(
        source_id="sba-business-guide-and-content-api",
        relative_root="sba-business-guide",
        origins=(
            "https://www.sba.gov/business-guide",
            "https://www.sba.gov/counseling/plan-your-business/",
            "https://www.sba.gov/counseling/launch-your-business/",
            "https://www.sba.gov/counseling/manage-your-business/",
            "https://www.sba.gov/counseling/grow-your-business/",
        ),
        target_leaves=(
            "business-core",
            "business-finance",
            "growth-gtm",
            "funding-capital",
            "business-operations",
        ),
        expected_files=5,
    ),
    SnapshotSource(
        source_id="yc-startup-library",
        relative_root="yc-startup-library",
        origins=("https://www.ycombinator.com/library/sitemap.xml",),
        target_leaves=(
            "business-core",
            "growth-gtm",
            "funding-capital",
            "business-operations",
        ),
        expected_files=481,
        known_missing=(
            "https://www.ycombinator.com/library/5J-how-to-create-and-manage-a-board-part-2",
            "https://www.ycombinator.com/library/7q-post-product-market-fit-people-customers-sales",
        ),
    ),
    SnapshotSource(
        source_id="finqa",
        relative_root="finqa",
        origins=("https://github.com/czyssrs/FinQA.git",),
        target_leaves=("business-finance", "investing-markets"),
        expected_files=45,
        git_commit_required=True,
        partition_policy="preserve_upstream_train_dev_test",
    ),
    SnapshotSource(
        source_id="tat-qa",
        relative_root="tat-qa",
        origins=("https://github.com/NExTplusplus/TAT-QA.git",),
        target_leaves=("business-finance", "investing-markets"),
        expected_files=73,
        git_commit_required=True,
        partition_policy="preserve_upstream_train_dev_test",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _inventory(directory: Path) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    artifacts: list[dict[str, object]] = []
    pruned: set[str] = set()
    for current, directory_names, file_names in os.walk(directory, topdown=True):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directory_names:
            relative = (current_path / name).relative_to(directory)
            if name == ".git" or _blocked(relative.parts):
                pruned.add(relative.as_posix())
            else:
                kept_directories.append(name)
        directory_names[:] = sorted(kept_directories)
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(directory)
            if name in _GENERATED_NAMES:
                continue
            if _blocked(relative.parts):
                pruned.add(relative.as_posix())
                continue
            stat = path.stat()
            artifacts.append(
                {
                    "bytes": stat.st_size,
                    "path": relative.as_posix(),
                    "sha256": _sha256(path),
                }
            )
    return artifacts, tuple(sorted(pruned))


def _git_commit(directory: Path) -> str | None:
    if not (directory / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"invalid Git commit identity for {directory}")
    return commit


def build_business_acquisition_receipt(
    *,
    snapshot_root: Path,
    source_manifest: Path,
    destination: Path,
    sums_destination: Path,
    snapshot_at: str,
    sources: Sequence[SnapshotSource] = DEFAULT_SOURCES,
) -> dict[str, object]:
    """Hash a complete non-cyber acquisition wave and emit aggregate-only receipts."""

    root = snapshot_root.expanduser().resolve()
    if _blocked(root.parts):
        raise ValueError("snapshot_root may not be a cyber/security path")
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = source_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("business source manifest must contain a JSON object")
    private_policy = manifest.get("private_training_policy")
    if (
        not isinstance(private_policy, dict)
        or private_policy.get("license_or_rights_metadata_is_an_admission_gate") is not False
    ):
        raise ValueError(
            "business source manifest must explicitly disable license admission gating"
        )
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("snapshot source IDs must be unique")

    source_receipts: list[dict[str, object]] = []
    sums: list[str] = []
    total_files = 0
    total_bytes = 0
    for source in sources:
        directory = (root / source.relative_root).resolve()
        if directory.parent != root:
            raise ValueError("every source directory must be a direct child of snapshot_root")
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        artifacts, pruned_paths = _inventory(directory)
        if source.expected_files is not None and len(artifacts) != source.expected_files:
            raise RuntimeError(
                f"{source.source_id} has {len(artifacts)} files; expected {source.expected_files}"
            )
        commit = _git_commit(directory) if source.git_commit_required else None
        if source.git_commit_required and commit is None:
            raise RuntimeError(f"{source.source_id} requires a pinned Git commit")
        byte_count = sum(int(artifact["bytes"]) for artifact in artifacts)
        source_receipts.append(
            {
                "artifacts": artifacts,
                "bytes": byte_count,
                "file_count": len(artifacts),
                "git_commit": commit,
                "known_missing": list(source.known_missing),
                "origins": list(source.origins),
                "partition_policy": source.partition_policy,
                "pruned_paths": list(pruned_paths),
                "relative_root": source.relative_root,
                "source_id": source.source_id,
                "target_leaves": list(source.target_leaves),
            }
        )
        total_files += len(artifacts)
        total_bytes += byte_count
        sums.extend(
            f"{artifact['sha256']}  {source.relative_root}/{artifact['path']}"
            for artifact in artifacts
        )

    receipt: dict[str, object] = {
        "contains_raw_examples": False,
        "license_or_rights_metadata_used_for_admission": False,
        "schema_version": SCHEMA_VERSION,
        "snapshot_at": snapshot_at,
        "snapshot_root_label": root.name,
        "source_manifest": {
            "path": manifest_path.name,
            "schema_version": manifest.get("schema_version"),
            "sha256": _sha256(manifest_path),
        },
        "sources": source_receipts,
        "totals": {
            "bytes": total_bytes,
            "file_count": total_files,
            "known_missing": sum(len(source.known_missing) for source in sources),
            "source_count": len(sources),
            "zero_silent_drops": True,
        },
    }
    _write_atomic(destination, _canonical_json_bytes(receipt))
    _write_atomic(sums_destination, ("\n".join(sorted(sums)) + "\n").encode("utf-8"))
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sums-destination", type=Path, required=True)
    parser.add_argument("--snapshot-at", required=True)
    args = parser.parse_args(argv)
    build_business_acquisition_receipt(
        snapshot_root=args.snapshot_root,
        source_manifest=args.source_manifest,
        destination=args.destination,
        sums_destination=args.sums_destination,
        snapshot_at=args.snapshot_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
