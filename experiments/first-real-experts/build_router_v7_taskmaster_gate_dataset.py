"""Build the train/validation-only Taskmaster admission-gate corpus.

Router v7 is an incremental binary gate in front of the immutable promoted v4
router.  The positive class is the quarantined Taskmaster v2 expert; every
legacy v4 route maps to the structural ``base/null`` outcome, which means
"delegate to v4" at runtime.

The builder consumes only the hash-pinned router-v6 train and validation
splits.  It has no code path for a router test split, a Taskmaster test split,
predictions, evaluation reports, or blind probes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
SOURCE_DATASET_ID = "router-bootstrap-v6-taskmaster"
OUTPUT_DATASET_ID = "router-bootstrap-v7-taskmaster-gate"
SOURCE_SPLITS = ("train", "validation")
TASKMASTER_ROUTE_ID = "taskmaster-coffee-api-v2-full-dialog-r16"
DELEGATE_ROUTE_ID = "base/null"
ROUTE_IDS = (TASKMASTER_ROUTE_ID, DELEGATE_ROUTE_ID)
SOURCE_ROUTE_IDS = (
    "embedded-knowledge-v1",
    "ocb-can-verified-v1",
    "software-architecture-v1",
    "verified-embedded-code-v1",
    TASKMASTER_ROUTE_ID,
    DELEGATE_ROUTE_ID,
)
SOURCE_IDENTITIES = {
    "train": {
        "rows": 5616,
        "sha256": "0dd1d8fada18d0c306a12eb5cc4fbfcfa04e31290dd3112dfcc89791fae9c31f",
    },
    "validation": {
        "rows": 780,
        "sha256": "0b9203e5c5778017f6580a9139201e950fd275a6b68dc104807b46cfe0f04220",
    },
}
EXPECTED_SOURCE_BALANCE = {
    "train": {route_id: 936 for route_id in SOURCE_ROUTE_IDS},
    "validation": {route_id: 130 for route_id in SOURCE_ROUTE_IDS},
}
_WHITESPACE = re.compile(r"\s+")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_prompt(prompt: str) -> str:
    return _WHITESPACE.sub(" ", prompt).strip().casefold()


def _source_path(prefix: Path, split: str) -> Path:
    if split not in SOURCE_SPLITS:
        raise ValueError(f"unauthorized v7 gate source split: {split!r}")
    source = prefix.expanduser()
    if source.name != SOURCE_DATASET_ID:
        raise ValueError(
            f"v7 gate source prefix must end in {SOURCE_DATASET_ID!r}, got {source.name!r}"
        )
    expected_parent = DATA_ROOT.resolve()
    if source.parent.resolve() != expected_parent:
        raise ValueError("v7 gate source prefix escaped the repository data directory")
    path = source.parent / f"{source.name}.{split}.jsonl"
    if path.is_symlink():
        raise ValueError(f"v7 gate source may not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve()
    if resolved.parent != expected_parent:
        raise ValueError("resolved v7 gate source escaped the repository data directory")
    expected_hash = SOURCE_IDENTITIES[split]["sha256"]
    actual_hash = _sha256_file(resolved)
    if actual_hash != expected_hash:
        raise ValueError(f"v7 gate {split} source differs from its frozen identity: {actual_hash}")
    return resolved


def _map_route(route_id: str) -> str:
    if route_id not in SOURCE_ROUTE_IDS:
        raise ValueError(f"unknown router-v6 route ID: {route_id!r}")
    return TASKMASTER_ROUTE_ID if route_id == TASKMASTER_ROUTE_ID else DELEGATE_ROUTE_ID


def _load_and_map(path: Path, *, split: str) -> tuple[list[dict[str, str]], Counter[str]]:
    rows: list[dict[str, str]] = []
    source_balance: Counter[str] = Counter()
    normalized_prompts: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank rows are forbidden")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict) or set(raw) != {"prompt", "route_id"}:
                raise ValueError(
                    f"{path}:{line_number}: row must contain exactly prompt and route_id"
                )
            prompt = raw["prompt"]
            route_id = raw["route_id"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: prompt must be non-empty text")
            if not isinstance(route_id, str):
                raise ValueError(f"{path}:{line_number}: route_id must be text")
            normalized = _normalize_prompt(prompt)
            if normalized in normalized_prompts:
                raise ValueError(f"{path}:{line_number}: duplicate normalized prompt in {split}")
            normalized_prompts.add(normalized)
            source_balance[route_id] += 1
            rows.append({"prompt": prompt, "route_id": _map_route(route_id)})
    expected_rows = int(SOURCE_IDENTITIES[split]["rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"v7 gate {split} source has {len(rows)} rows, expected {expected_rows}")
    if dict(source_balance) != EXPECTED_SOURCE_BALANCE[split]:
        raise ValueError(f"v7 gate {split} source route balance differs from the frozen contract")
    return rows, source_balance


def _write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_canonical_json(row))
            stream.write("\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_router_v7_gate_dataset(
    *,
    source_prefix: Path = DATA_ROOT / SOURCE_DATASET_ID,
    output_prefix: Path = DATA_ROOT / OUTPUT_DATASET_ID,
) -> dict[str, Any]:
    """Create deterministic binary train/validation splits and an identity report."""

    source_paths = {split: _source_path(source_prefix, split) for split in SOURCE_SPLITS}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    source_balance: dict[str, dict[str, int]] = {}
    normalized_by_split: dict[str, set[str]] = {}
    for split, path in source_paths.items():
        rows, balance = _load_and_map(path, split=split)
        rows_by_split[split] = rows
        source_balance[split] = dict(balance)
        normalized_by_split[split] = {_normalize_prompt(row["prompt"]) for row in rows}

    overlap = normalized_by_split["train"].intersection(normalized_by_split["validation"])
    if overlap:
        raise ValueError("v7 gate train and validation prompts overlap after normalization")

    destination = output_prefix.expanduser().resolve()
    outputs: dict[str, dict[str, Any]] = {}
    mapped_balance: dict[str, dict[str, int]] = {}
    for split in SOURCE_SPLITS:
        path = destination.parent / f"{destination.name}.{split}.jsonl"
        _write_jsonl(path, rows_by_split[split])
        balance = Counter(row["route_id"] for row in rows_by_split[split])
        expected_positive = EXPECTED_SOURCE_BALANCE[split][TASKMASTER_ROUTE_ID]
        expected_delegate = int(SOURCE_IDENTITIES[split]["rows"]) - expected_positive
        expected = {
            TASKMASTER_ROUTE_ID: expected_positive,
            DELEGATE_ROUTE_ID: expected_delegate,
        }
        if dict(balance) != expected:
            raise AssertionError(f"v7 gate {split} mapped balance is inconsistent")
        mapped_balance[split] = dict(balance)
        outputs[split] = {
            "path": str(path),
            "rows": len(rows_by_split[split]),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    report = {
        "schema_version": "molor.router-admission-gate-dataset.v1",
        "dataset_id": OUTPUT_DATASET_ID,
        "semantics": {
            TASKMASTER_ROUTE_ID: "admit the quarantined Taskmaster expert",
            DELEGATE_ROUTE_ID: "delegate unchanged to the pinned promoted v4 router",
        },
        "route_ids": list(ROUTE_IDS),
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "authorized_splits": list(SOURCE_SPLITS),
            "identities": SOURCE_IDENTITIES,
            "paths": {split: str(path) for split, path in source_paths.items()},
            "route_balance": source_balance,
        },
        "mapped_balance": mapped_balance,
        "normalized_prompt_overlap": {"train__validation": len(overlap)},
        "source_row_order_preserved": True,
        "source_prompts_preserved_byte_for_byte": True,
        "router_test_split_read_by_builder": False,
        "taskmaster_test_split_read_by_builder": False,
        "prediction_or_evaluation_artifacts_read_by_builder": False,
        "blind_probe_read_by_builder": False,
        "outputs": outputs,
    }
    report_path = destination.parent / f"{destination.name}.report.json"
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the train/validation-only binary Taskmaster admission-gate corpus."
    )
    parser.add_argument(
        "--source-prefix",
        type=Path,
        default=DATA_ROOT / SOURCE_DATASET_ID,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ROOT / OUTPUT_DATASET_ID,
    )
    args = parser.parse_args()
    report = build_router_v7_gate_dataset(
        source_prefix=args.source_prefix,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
