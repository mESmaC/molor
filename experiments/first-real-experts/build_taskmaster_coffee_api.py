"""Build the quarantined Taskmaster-4 coffee API candidate curriculum.

The release is read only from the pinned local checkout.  Positive API-call reward
rows become canonical JSON targets.  Exact content is deduplicated, ambiguous
contexts are quarantined, and every trajectory-root group is assigned wholly to
one deterministic split.  Generated corpora stay under the gitignored ``data``
directory; this builder, its source pin, and the candidate recipe are the durable
reproduction contract.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DATASET_ID = "taskmaster-coffee-api-v1"
DATASET_SCHEMA = "molor.taskmaster-coffee-api-dataset.v1"
PROVENANCE_SCHEMA = "molor.taskmaster-coffee-api-provenance.v1"
EVALUATION_SCHEMA = "molor.taskmaster-coffee-api-evaluation.v1"
QUARANTINE_SCHEMA = "molor.taskmaster-coffee-api-quarantine.v1"
REPORT_SCHEMA = "molor.taskmaster-coffee-api-build-report.v1"
SOURCE_MANIFEST = HERE / "taskmaster-coffee-api-v1-source.json"
DEFAULT_SOURCE = HERE / "sources" / "taskmaster"
DEFAULT_OUTPUT = HERE / "data"
SPLITS = ("train", "validation", "test")
SYSTEM_PROMPT = (
    "You are the API planner for a coffee-ordering agent. The user supplies a "
    "Taskmaster-4 conversation context as canonical JSON. Return exactly one JSON "
    "object with role=agent and one api call. Use the source API names and arguments. "
    "Output canonical minified JSON only: no Markdown, commentary, or thinking tags."
)


class SourceRowError(ValueError):
    """A source row cannot safely become a canonical example."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SourceLocator:
    source_file: str
    source_row_index: int
    conversation_id: str
    source_record_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "source_file": self.source_file,
            "source_record_sha256": self.source_record_sha256,
            "source_row_index": self.source_row_index,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    context: list[dict[str, Any]]
    generation: dict[str, Any]
    context_canonical: str
    target_canonical: str
    locator: SourceLocator

    @property
    def context_sha256(self) -> str:
        return sha256_text(self.context_canonical)

    @property
    def target_sha256(self) -> str:
        return sha256_text(self.target_canonical)


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    group_id: str
    split: str
    context: list[dict[str, Any]]
    generation: dict[str, Any]
    messages: list[dict[str, str]]
    context_sha256: str
    target_sha256: str
    source_records: tuple[SourceLocator, ...]


def canonical_json(value: Any) -> str:
    """Return the sole JSON representation used for IDs, prompts, and targets."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SourceRowError("non_canonical_json_value", str(exc)) from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def strict_json_loads(value: str) -> Any:
    if not isinstance(value, str):
        raise SourceRowError("embedded_json_not_string", "embedded JSON must be a string")
    try:
        return json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SourceRowError("invalid_embedded_json", str(exc)) from exc


def _validate_one_key_mapping(
    value: Any,
    *,
    reason: str,
    require_object_value: bool,
) -> None:
    if not isinstance(value, dict) or len(value) != 1:
        raise SourceRowError(reason, "API mapping must contain exactly one entry")
    name, payload = next(iter(value.items()))
    if not isinstance(name, str) or not name or name != name.strip():
        raise SourceRowError(reason, "API name must be a non-empty trimmed string")
    if require_object_value and not isinstance(payload, dict):
        raise SourceRowError(reason, "API arguments must be a JSON object")
    canonical_json(value)


def validate_context(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SourceRowError("context_not_nonempty_array", "context must be a non-empty array")
    found_user = False
    for entry in value:
        if not isinstance(entry, dict):
            raise SourceRowError("invalid_context_entry", "context entry is not an object")
        keys = set(entry)
        role = entry.get("role")
        if keys == {"role", "text"}:
            if role not in {"agent", "user"}:
                raise SourceRowError("invalid_context_text", "unsupported text role")
            text = entry["text"]
            if not isinstance(text, str) or not text.strip():
                raise SourceRowError("invalid_context_text", "text must be non-empty")
            found_user = found_user or role == "user"
        elif keys == {"role", "api"}:
            if role != "agent":
                raise SourceRowError("invalid_context_api", "API caller must be agent")
            _validate_one_key_mapping(
                entry["api"],
                reason="invalid_context_api",
                require_object_value=True,
            )
        elif keys == {"role", "response"}:
            if role != "api":
                raise SourceRowError("invalid_context_response", "response role must be api")
            _validate_one_key_mapping(
                entry["response"],
                reason="invalid_context_response",
                require_object_value=False,
            )
        else:
            raise SourceRowError("invalid_context_entry", "context entry has an unsupported schema")
    if not found_user:
        raise SourceRowError("context_missing_user", "context has no user turn")
    if value[-1].get("role") not in {"user", "api"}:
        raise SourceRowError(
            "context_not_ready_for_agent", "context does not end at an agent decision point"
        )
    canonical_json(value)
    return value


def validate_api_generation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"role", "api"}:
        raise SourceRowError("generation_not_agent_api", "generation is not an agent API object")
    if value.get("role") != "agent":
        raise SourceRowError("invalid_generation_api", "generation role must be agent")
    _validate_one_key_mapping(
        value["api"],
        reason="invalid_generation_api",
        require_object_value=True,
    )
    canonical_json(value)
    return value


def is_valid_text_generation(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"role", "text"}
        and value.get("role") == "agent"
        and isinstance(value.get("text"), str)
        and bool(value["text"].strip())
    )


def trajectory_root(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the opening text exchange shared by one reward trajectory."""

    prefix: list[dict[str, Any]] = []
    for entry in context:
        if "api" in entry or entry.get("role") == "api":
            break
        prefix.append(entry)
    if not prefix or not any(entry.get("role") == "user" for entry in prefix):
        raise SourceRowError(
            "missing_trajectory_root", "context has no user turn before its first API call"
        )
    return prefix


def group_id(context: list[dict[str, Any]]) -> str:
    digest = sha256_text(canonical_json(trajectory_root(context)))
    return f"tm4-coffee-trajectory:{digest}"


def split_for_group(identifier: str) -> str:
    digest = hashlib.sha256(f"{DATASET_ID}:split:{identifier}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.10:
        return "test"
    if bucket < 0.20:
        return "validation"
    return "train"


def messages_for(context_canonical: str, target_canonical: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Taskmaster-4 conversation context:\n{context_canonical}",
        },
        {"role": "assistant", "content": target_canonical},
    ]


def _git_output(source: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not verify pinned Taskmaster checkout: {exc}") from exc
    return result.stdout.strip()


def load_source_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "molor.source-pin.v1":
        raise ValueError("unsupported Taskmaster source manifest")
    return payload


def verify_source(source: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    expected_commit = manifest.get("resolved_commit")
    actual_commit = _git_output(source, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"Taskmaster commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    dirty = _git_output(source, "status", "--porcelain", "--", manifest["release_path"])
    if dirty:
        raise RuntimeError("pinned Taskmaster-4 source has uncommitted changes")

    verified: list[dict[str, Any]] = []
    for descriptor in manifest.get("files", []):
        relative = descriptor.get("path")
        expected_hash = descriptor.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("source manifest file descriptors need path and sha256")
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned source file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned source hash mismatch for {relative}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
        verified.append(
            {"path": relative, "sha256": actual_hash, "size_bytes": path.stat().st_size}
        )
    return verified


def _source_locator(relative: str, index: int, row: Any) -> SourceLocator:
    record_hash = sha256_text(canonical_json(row))
    if not isinstance(row, dict):
        raise SourceRowError("source_row_not_object", "source row must be an object")
    if set(row) != {"conversation_id", "context", "generation", "reward"}:
        raise SourceRowError("invalid_source_row_schema", "source row fields do not match TM4")
    conversation_id = row.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise SourceRowError("invalid_conversation_id", "conversation_id must be non-empty")
    return SourceLocator(relative, index, conversation_id, record_hash)


def _quarantine_row(
    locator: SourceLocator | None,
    *,
    source_file: str,
    source_row_index: int,
    reason: str,
    detail: str,
    context_sha256: str | None = None,
    target_sha256: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": QUARANTINE_SCHEMA,
        "detail": detail,
        "disposition": "quarantined",
        "reason": reason,
        "source_file": source_file,
        "source_row_index": source_row_index,
    }
    if locator is not None:
        row.update(locator.to_dict())
    if context_sha256 is not None:
        row["context_sha256"] = context_sha256
    if target_sha256 is not None:
        row["target_sha256"] = target_sha256
    return row


def _load_candidates(
    source: Path,
    manifest: dict[str, Any],
) -> tuple[list[Candidate], list[dict[str, Any]], Counter[str], int]:
    candidates: list[Candidate] = []
    quarantine: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    total = 0
    release_path = str(manifest["release_path"])
    reward_files = [
        descriptor["path"]
        for descriptor in manifest["files"]
        if str(descriptor["path"]).startswith(f"{release_path}/reward/")
    ]
    for relative in sorted(reward_files):
        path = source / relative
        raw_rows = strict_json_loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise RuntimeError(f"{relative} is not a JSON array")
        for index, row in enumerate(raw_rows):
            total += 1
            locator: SourceLocator | None = None
            try:
                locator = _source_locator(relative, index, row)
                context = validate_context(strict_json_loads(row["context"]))
                generation_raw = strict_json_loads(row["generation"])
                reward = row["reward"]
                if (
                    not isinstance(reward, (int, float))
                    or isinstance(reward, bool)
                    or not math.isfinite(reward)
                    or reward not in {0, 1}
                ):
                    raise SourceRowError("invalid_reward", "reward must be finite 0 or 1")
                if is_valid_text_generation(generation_raw):
                    excluded["non_api_generation"] += 1
                    continue
                generation = validate_api_generation(generation_raw)
                if reward != 1:
                    excluded["non_positive_reward"] += 1
                    continue
                context_canonical = canonical_json(context)
                target_canonical = canonical_json(generation)
                candidates.append(
                    Candidate(
                        context=context,
                        generation=generation,
                        context_canonical=context_canonical,
                        target_canonical=target_canonical,
                        locator=locator,
                    )
                )
            except SourceRowError as exc:
                quarantine.append(
                    _quarantine_row(
                        locator,
                        source_file=relative,
                        source_row_index=index,
                        reason=exc.reason,
                        detail=str(exc),
                    )
                )
    return candidates, quarantine, excluded, total


def _deduplicate_and_quarantine_conflicts(
    candidates: list[Candidate],
    quarantine: list[dict[str, Any]],
) -> tuple[list[Example], dict[str, int]]:
    targets_by_context: dict[str, set[str]] = defaultdict(set)
    rows_by_context: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        targets_by_context[candidate.context_canonical].add(candidate.target_canonical)
        rows_by_context[candidate.context_canonical].append(candidate)

    conflicting_contexts = {
        context for context, targets in targets_by_context.items() if len(targets) > 1
    }
    conflicting_pairs = sum(len(targets_by_context[key]) for key in conflicting_contexts)
    conflicting_rows = 0
    for context in sorted(conflicting_contexts):
        target_count = len(targets_by_context[context])
        for candidate in rows_by_context[context]:
            conflicting_rows += 1
            quarantine.append(
                _quarantine_row(
                    candidate.locator,
                    source_file=candidate.locator.source_file,
                    source_row_index=candidate.locator.source_row_index,
                    reason="ambiguous_positive_targets",
                    detail=f"canonical context has {target_count} distinct positive API targets",
                    context_sha256=candidate.context_sha256,
                    target_sha256=candidate.target_sha256,
                )
            )

    by_content: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.context_canonical not in conflicting_contexts:
            by_content[(candidate.context_canonical, candidate.target_canonical)].append(candidate)

    examples: list[Example] = []
    for (context_canonical, target_canonical), sources in sorted(by_content.items()):
        representative = min(
            sources,
            key=lambda row: (row.locator.source_file, row.locator.source_row_index),
        )
        messages = messages_for(context_canonical, target_canonical)
        content_hash = sha256_text(canonical_json(messages))
        identifier = f"tm4-coffee-api:{content_hash}"
        trajectory = group_id(representative.context)
        examples.append(
            Example(
                example_id=identifier,
                group_id=trajectory,
                split=split_for_group(trajectory),
                context=representative.context,
                generation=representative.generation,
                messages=messages,
                context_sha256=representative.context_sha256,
                target_sha256=representative.target_sha256,
                source_records=tuple(
                    sorted(
                        (source.locator for source in sources),
                        key=lambda item: (item.source_file, item.source_row_index),
                    )
                ),
            )
        )
    return examples, {
        "conflicting_contexts": len(conflicting_contexts),
        "conflicting_positive_target_pairs": conflicting_pairs,
        "conflicting_source_rows": conflicting_rows,
        "deduplicated_source_rows": sum(len(rows) - 1 for rows in by_content.values()),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row))
            stream.write("\n")


def _output_descriptor(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_dataset(
    *,
    source: Path,
    output: Path,
    source_manifest_path: Path = SOURCE_MANIFEST,
    verify_pin: bool = True,
) -> dict[str, Any]:
    manifest = load_source_manifest(source_manifest_path)
    verified_files = verify_source(source, manifest) if verify_pin else []
    candidates, quarantine, excluded, source_rows = _load_candidates(source, manifest)
    invalid_rows = len(quarantine)
    examples, conflict_stats = _deduplicate_and_quarantine_conflicts(candidates, quarantine)
    quarantine.sort(
        key=lambda row: (
            str(row["source_file"]),
            int(row["source_row_index"]),
            str(row["reason"]),
        )
    )

    split_examples = {
        split: sorted(
            (example for example in examples if example.split == split),
            key=lambda example: example.example_id,
        )
        for split in SPLITS
    }
    group_owners: dict[str, str] = {}
    for split, rows in split_examples.items():
        for example in rows:
            previous = group_owners.setdefault(example.group_id, split)
            if previous != split:
                raise RuntimeError(f"group {example.group_id} leaked across splits")

    source_contract = {
        "source_id": manifest["source_id"],
        "repository_url": manifest["repository_url"],
        "resolved_commit": manifest["resolved_commit"],
        "release_path": manifest["release_path"],
        "collection_method": manifest["collection_method"],
        "license": manifest["license"],
        "attribution": manifest["attribution"],
        "paper": manifest["paper"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }
    outputs: dict[str, dict[str, Any]] = {}
    api_counts: dict[str, dict[str, int]] = {}
    group_counts: dict[str, int] = {}
    for split, rows in split_examples.items():
        training_path = output / f"{DATASET_ID}.{split}.jsonl"
        provenance_path = output / f"{DATASET_ID}.{split}.provenance.jsonl"
        _write_jsonl(training_path, ({"messages": row.messages} for row in rows))
        _write_jsonl(
            provenance_path,
            (
                {
                    "schema_version": PROVENANCE_SCHEMA,
                    "context_sha256": row.context_sha256,
                    "example_id": row.example_id,
                    "group_id": row.group_id,
                    "license_spdx": manifest["license"]["spdx_id"],
                    "source_commit": manifest["resolved_commit"],
                    "source_records": [record.to_dict() for record in row.source_records],
                    "split": split,
                    "target_sha256": row.target_sha256,
                    "training_messages_sha256": row.example_id.rsplit(":", 1)[1],
                }
                for row in rows
            ),
        )
        outputs[f"{split}_training"] = _output_descriptor(training_path, len(rows))
        outputs[f"{split}_provenance"] = _output_descriptor(provenance_path, len(rows))
        group_counts[split] = len({row.group_id for row in rows})
        api_counts[split] = dict(
            sorted(Counter(next(iter(row.generation["api"])) for row in rows).items())
        )

        if split != "train":
            evaluation_path = output / f"{DATASET_ID}.{split}.evaluation.jsonl"
            _write_jsonl(
                evaluation_path,
                (
                    {
                        "schema_version": EVALUATION_SCHEMA,
                        "example_id": row.example_id,
                        "expected": row.generation,
                        "expected_canonical": canonical_json(row.generation),
                        "expected_sha256": row.target_sha256,
                        "group_id": row.group_id,
                        "messages": row.messages[:-1],
                    }
                    for row in rows
                ),
            )
            outputs[f"{split}_evaluation"] = _output_descriptor(evaluation_path, len(rows))

    quarantine_path = output / f"{DATASET_ID}.quarantine.jsonl"
    _write_jsonl(quarantine_path, quarantine)
    outputs["quarantine"] = _output_descriptor(quarantine_path, len(quarantine))

    report = {
        "schema_version": REPORT_SCHEMA,
        "dataset_id": DATASET_ID,
        "source": source_contract,
        "verified_source_files": verified_files,
        "selection": {
            "accepted_reward": 1,
            "accepted_generation": "one agent API call with object arguments",
            "canonical_json": "UTF-8, sorted object keys, minified, finite values",
            "conflict_policy": (
                "quarantine every source row when one canonical context has multiple "
                "distinct positive canonical API targets"
            ),
            "deduplication": ("exact canonical context plus target; all source locators retained"),
            "grouping": "canonical text-only prefix before the first API call/response",
            "split_policy": (
                "SHA-256(dataset-id + group-id), 80% train / 10% validation / 10% test"
            ),
        },
        "counts": {
            "source_rows": source_rows,
            "positive_api_source_rows_before_conflicts": len(candidates),
            "quarantined_invalid_source_rows": invalid_rows,
            "excluded_source_rows": dict(sorted(excluded.items())),
            **conflict_stats,
            "emitted_unique_examples": len(examples),
            "emitted_groups": len(group_owners),
            "examples_by_split": {split: len(split_examples[split]) for split in SPLITS},
            "groups_by_split": group_counts,
            "api_calls_by_split": api_counts,
        },
        "integrity": {
            "group_disjoint": True,
            "example_ids_unique": len({row.example_id for row in examples}) == len(examples),
            "held_out_targets_absent_from_training_by_group": True,
        },
        "outputs": outputs,
        "intended_checkpoint_posture": "candidate_quarantine_only_never_auto_promote",
    }
    report_path = output / f"{DATASET_ID}.build-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the pinned, group-disjoint Taskmaster-4 coffee API curriculum."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_dataset(
        source=args.source.resolve(),
        output=args.output.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        verify_pin=True,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
