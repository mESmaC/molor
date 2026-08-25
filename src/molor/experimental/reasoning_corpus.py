"""Deterministically materialize a leakage-screened reasoning SFT corpus.

The builder intentionally consumes the already-prepared Complete-FABLE JSONL shards one line at
a time.  It accepts only simple one-turn records from an explicit permissive-license allowlist,
renders every candidate with the pinned local Qwen tokenizer, and never truncates a record to make
it fit.  Normalized user prompts are the split groups, and each approved upstream source belongs
to exactly one split, so neither prompt-equivalent records nor sources can cross train,
validation, and test boundaries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Protocol, Sequence
import unicodedata


SCHEMA_VERSION = "molor.reasoning-corpus.v1"
DEFAULT_SOURCE_COMMIT = "b9f20b5c2638b96311252d5894e3d010ba2a2715"
DEFAULT_SEED = "molor-qwen3-0.6b-reasoning-v1"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_SPLIT_COUNTS = {"train": 12_000, "validation": 512, "test": 512}
NEUTRAL_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Solve the user's problem and give a clear final "
    "answer."
)

ALLOWED_SOURCE_LICENSES: dict[str, str] = {
    "Roman1111111/claude-sonnet-4.6-100000X-filtered": "mit",
    "Roman1111111/claude-opus-4.6-10000x": "mit",
    "TeichAI/lordx64-claude-opus-4.7-max-cleaned": "apache-2.0",
    "TeichAI/Claude-Sonnet-4.6-Reasoning-1100x": "apache-2.0",
    "TeichAI/Claude-Opus-4.6-Reasoning-887x": "apache-2.0",
}
ALLOWED_SOURCES = tuple(ALLOWED_SOURCE_LICENSES)

_DEFAULT_DATASET_ROOT = (
    Path.home() / "Documents" / "datasets" / "prepared" / "complete-fable-5-traces"
)
DEFAULT_INPUTS = (
    _DEFAULT_DATASET_ROOT / "train.jsonl",
    _DEFAULT_DATASET_ROOT / "validation.jsonl",
)
DEFAULT_INPUT_HASHES = {
    str(DEFAULT_INPUTS[0].resolve()): (
        "70e7462373f5f6254a48c0f5a8c937195045dc1700b36cea25398e3686d83c8e"
    ),
    str(DEFAULT_INPUTS[1].resolve()): (
        "4b4ebdd4744ffb4ce4ec9da555738e7e3c5106f10ac0ec76ea7a156129f6770d"
    ),
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER = _REPOSITORY_ROOT / "models" / "Qwen3-0.6B"

_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_RESERVED_CHAT_MARKERS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")
_TEACHER_IDENTITY_RE = re.compile(
    r"(?i)(?:\bclaude\b|\banthropic\b|\bfable[\s-]*5\b|"
    r"\bsonnet[\s-]*4(?:\.\d+)?\b|\bopus[\s-]*4(?:\.\d+)?\b)"
)
_BENCHMARK_RE = re.compile(
    r"(?i)(?:\bbenchmark(?:s|ing)?\b|\bgsm8k\b|\bmmlu(?:-pro)?\b|\bhumaneval\b|"
    r"\bmbpp\b|\bifeval\b|\bgpqa\b|\baime(?:\s*\d{4})?\b|\bhellaswag\b|"
    r"\btruthfulqa\b|\barc[-\s]challenge\b|\bbig[-\s]bench\b|\bswe[-\s]bench\b|"
    r"\blivecodebench\b|\blm[-_]eval\b|\bmolor[-_](?:core|executable)[-_]v\d+\b|"
    r"\bexact[-_]context\b)"
)


class ChatTemplateTokenizer(Protocol):
    """Small tokenizer surface needed by this materializer."""

    def apply_chat_template(
        self, conversation: list[dict[str, str]], **kwargs: Any
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    source: str
    prompt_group: str
    selection_rank: str
    record_fingerprint: str
    payload: str
    token_count: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(prompt: str) -> str:
    """Return the exact, documented grouping normalization for a user prompt."""

    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(normalized.split())


def prompt_group_id(prompt: str) -> str:
    return _sha256_bytes(normalize_prompt(prompt).encode("utf-8"))


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = cleaned.strip()
    return cleaned or None


def _token_ids(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("chat template must return a non-empty flat token list")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError("chat template returned invalid token IDs")
    return list(value)


def render_token_ids(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    """Render prompt and full record with Qwen thinking enabled and verify their prefix."""

    if len(messages) != 3 or [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("reasoning records must be system/user/assistant chats")
    if not messages[-1].get("content", "").startswith("<think>\n"):
        raise ValueError("reasoning assistant target must begin with an explicit thinking block")

    common = {
        "tokenize": True,
        "enable_thinking": True,
        "truncation": False,
        "padding": False,
        "return_tensors": None,
        "return_dict": False,
    }
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, **common
        )
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, **common
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "tokenizer chat template has inconsistent thinking-enabled prompt and full prefixes"
        )
    return prompt_ids, full_ids


def _simple_messages(record: Mapping[str, Any]) -> tuple[str, str, str] | None:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        return None

    # Source system messages are deliberately discarded.  They are accepted only in the normal
    # leading position; a system/tool/developer message inside the turn is not a simple record.
    index = 0
    while index < len(raw_messages):
        message = raw_messages[index]
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        index += 1
    turn = raw_messages[index:]
    if len(turn) != 2 or any(not isinstance(message, dict) for message in turn):
        return None
    user, assistant = turn
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None

    prompt = _clean_text(user.get("content"))
    thinking = _clean_text(assistant.get("thinking"))
    answer = _clean_text(assistant.get("content"))
    if prompt is None or thinking is None or answer is None:
        return None
    return prompt, thinking, answer


def _candidate_from_record(
    record: Mapping[str, Any],
    *,
    tokenizer: ChatTemplateTokenizer,
    max_length: int,
    seed: str,
) -> tuple[Candidate | None, str | None]:
    source = record.get("upstream")
    if source not in ALLOWED_SOURCE_LICENSES:
        return None, "source_not_allowlisted"

    fields = _simple_messages(record)
    if fields is None:
        return None, "not_simple_reasoning_turn"
    prompt, thinking, answer = fields
    combined = "\n".join(fields)
    if any(marker.casefold() in combined.casefold() for marker in _RESERVED_CHAT_MARKERS):
        return None, "reserved_chat_marker"
    if _TEACHER_IDENTITY_RE.search(combined):
        return None, "teacher_identity_mention"
    if _BENCHMARK_RE.search(combined):
        return None, "benchmark_mention"

    assistant_target = f"<think>\n{thinking}\n</think>\n\n{answer}"
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_target},
    ]
    _, full_ids = render_token_ids(tokenizer, messages)
    if len(full_ids) > max_length:
        return None, "rendered_too_long"

    payload = json.dumps(
        {"messages": messages}, ensure_ascii=False, separators=(",", ":")
    )
    fingerprint_payload = json.dumps(
        {"source": source, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = _sha256_bytes(fingerprint_payload)
    group = prompt_group_id(prompt)
    selection_rank = _sha256_bytes(
        f"{seed}\0select\0{group}\0{fingerprint}".encode("utf-8")
    )
    return (
        Candidate(
            source=source,
            prompt_group=group,
            selection_rank=selection_rank,
            record_fingerprint=fingerprint,
            payload=payload,
            token_count=len(full_ids),
        ),
        None,
    )


def _create_candidate_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE candidates (
            prompt_group TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            selection_rank TEXT NOT NULL,
            record_fingerprint TEXT NOT NULL,
            payload TEXT NOT NULL,
            token_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX candidates_source_rank
            ON candidates(source, selection_rank, prompt_group);
        """
    )


def _insert_canonical(connection: sqlite3.Connection, candidate: Candidate) -> None:
    connection.execute(
        """
        INSERT INTO candidates(
            prompt_group, source, selection_rank, record_fingerprint, payload, token_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(prompt_group) DO UPDATE SET
            source=excluded.source,
            selection_rank=excluded.selection_rank,
            record_fingerprint=excluded.record_fingerprint,
            payload=excluded.payload,
            token_count=excluded.token_count
        WHERE excluded.selection_rank < candidates.selection_rank
        """,
        (
            candidate.prompt_group,
            candidate.source,
            candidate.selection_rank,
            candidate.record_fingerprint,
            candidate.payload,
            candidate.token_count,
        ),
    )


def _expected_hash_for(
    path: Path, expected_hashes: Mapping[str | Path, str]
) -> str | None:
    resolved = str(path.resolve())
    for key, expected in expected_hashes.items():
        if str(Path(key).resolve()) == resolved:
            return expected.lower()
    return None


def _stream_inputs(
    connection: sqlite3.Connection,
    *,
    input_paths: Sequence[Path],
    expected_hashes: Mapping[str | Path, str],
    require_expected_hashes: bool,
    tokenizer: ChatTemplateTokenizer,
    max_length: int,
    seed: str,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    inputs: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    eligible_before_group_dedup = 0

    for path in input_paths:
        source_path = path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"prepared input does not exist: {source_path}")
        expected = _expected_hash_for(source_path, expected_hashes)
        if require_expected_hashes and expected is None:
            raise ValueError(f"no expected SHA-256 supplied for prepared input: {source_path}")

        digest = hashlib.sha256()
        byte_count = 0
        record_count = 0
        with source_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                byte_count += len(raw_line)
                if not raw_line.strip():
                    continue
                record_count += 1
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{source_path}:{line_number}: invalid JSONL: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{source_path}:{line_number}: record must be a JSON object")
                candidate, rejection = _candidate_from_record(
                    value, tokenizer=tokenizer, max_length=max_length, seed=seed
                )
                if candidate is None:
                    assert rejection is not None
                    rejections[rejection] += 1
                    continue
                eligible_before_group_dedup += 1
                _insert_canonical(connection, candidate)

        actual = digest.hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(
                f"prepared input SHA-256 mismatch for {source_path}: "
                f"expected {expected}, got {actual}"
            )
        inputs.append(
            {
                "path": str(source_path),
                "sha256": actual,
                "expected_sha256": expected,
                "hash_verified": expected is not None and actual == expected,
                "bytes": byte_count,
                "records": record_count,
            }
        )
        connection.commit()
    return inputs, rejections, eligible_before_group_dedup


def _candidate_from_row(row: Sequence[Any]) -> Candidate:
    return Candidate(
        prompt_group=str(row[0]),
        source=str(row[1]),
        selection_rank=str(row[2]),
        record_fingerprint=str(row[3]),
        payload=str(row[4]),
        token_count=int(row[5]),
    )


_CANDIDATE_COLUMNS = (
    "c.prompt_group, c.source, c.selection_rank, c.record_fingerprint, "
    "c.payload, c.token_count"
)


_SPLIT_SOURCE_WEIGHTS: dict[str, tuple[tuple[str, int], ...]] = {
    # The default sizes yield exactly 9,000 / 3,000.
    "train": ((ALLOWED_SOURCES[0], 3), (ALLOWED_SOURCES[1], 1)),
    # This upstream source is held out from both train and final test.
    "validation": ((ALLOWED_SOURCES[2], 1),),
    # The default size yields exactly 300 / 212.
    "test": ((ALLOWED_SOURCES[3], 300), (ALLOWED_SOURCES[4], 212)),
}


def _weighted_quotas(total: int, weighted_sources: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Allocate an exact integer total by largest remainder with stable source-order ties."""

    weight_total = sum(weight for _, weight in weighted_sources)
    if weight_total < 1 or any(weight < 1 for _, weight in weighted_sources):
        raise ValueError("source allocation weights must be positive")
    quotas: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    allocated = 0
    for index, (source, weight) in enumerate(weighted_sources):
        quota, remainder = divmod(total * weight, weight_total)
        quotas[source] = quota
        allocated += quota
        remainders.append((remainder, -index, source))
    for _, _, source in sorted(remainders, reverse=True)[: total - allocated]:
        quotas[source] += 1
    return quotas


def _split_source_quotas(split_counts: Mapping[str, int]) -> dict[str, dict[str, int]]:
    expected_names = {"train", "validation", "test"}
    if set(split_counts) != expected_names:
        raise ValueError("split_counts must contain exactly train, validation, and test")
    if any(type(count) is not int or count < 1 for count in split_counts.values()):
        raise ValueError("every split count must be a positive integer")

    configured_sources = [
        source
        for weighted_sources in _SPLIT_SOURCE_WEIGHTS.values()
        for source, _ in weighted_sources
    ]
    if len(configured_sources) != len(set(configured_sources)):
        raise AssertionError("a source is assigned to more than one split")
    return {
        split: _weighted_quotas(split_counts[split], _SPLIT_SOURCE_WEIGHTS[split])
        for split in ("train", "validation", "test")
    }


def _select_candidates(
    connection: sqlite3.Connection,
    *,
    split_counts: Mapping[str, int],
) -> tuple[
    dict[str, list[Candidate]],
    dict[str, dict[str, int | str]],
    dict[str, dict[str, int]],
]:
    quotas_by_split = _split_source_quotas(split_counts)
    availability = {
        str(source): int(count)
        for source, count in connection.execute(
            "SELECT source, COUNT(*) FROM candidates GROUP BY source"
        )
    }
    splits: dict[str, list[Candidate]] = {"train": [], "validation": [], "test": []}
    per_source: dict[str, dict[str, int | str]] = {}
    for split in ("train", "validation", "test"):
        for source, quota in quotas_by_split[split].items():
            rows = connection.execute(
                f"SELECT {_CANDIDATE_COLUMNS} FROM candidates AS c "
                "WHERE c.source = ? ORDER BY c.selection_rank, c.prompt_group LIMIT ?",
                (source, quota),
            ).fetchall()
            candidates = [_candidate_from_row(row) for row in rows]
            if len(candidates) != quota:
                raise ValueError(
                    f"split {split!r} requires {quota} unique eligible prompt groups from "
                    f"{source!r}, but only {availability.get(source, 0)} are available; "
                    "cross-source fill is forbidden"
                )
            splits[split].extend(candidates)
            per_source[source] = {
                "split": split,
                "available_prompt_groups": availability.get(source, 0),
                "requested_quota": quota,
                "selected_at_quota": len(candidates),
                "shortfall": 0,
                "same_source_fill_added": 0,
                "cross_source_fill_added": 0,
                "final_selected": len(candidates),
                "fill_delta": len(candidates) - quota,
            }
        splits[split].sort(key=lambda item: (item.selection_rank, item.prompt_group))

    selected = [candidate for candidates in splits.values() for candidate in candidates]
    total = sum(split_counts.values())
    if len(selected) != total or len({item.prompt_group for item in selected}) != total:
        raise AssertionError("candidate selection did not produce exact unique prompt groups")
    return splits, per_source, quotas_by_split


def _token_stats(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty token-length collection")

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[max(0, index)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
        "total": sum(ordered),
    }


def _tokenizer_identity(path: Path) -> dict[str, Any]:
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local tokenizer directory does not exist: {root}")
    artifacts: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for name in _TOKENIZER_ARTIFACTS:
        artifact = root / name
        if not artifact.is_file():
            continue
        digest = _sha256_file(artifact)
        artifacts.append(
            {"path": name, "sha256": digest, "bytes": artifact.stat().st_size}
        )
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if not artifacts:
        raise ValueError(f"no tokenizer artifacts found in {root}")
    return {
        "path": str(root),
        "aggregate_sha256": aggregate.hexdigest(),
        "artifacts": artifacts,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _write_outputs(
    output_dir: Path,
    *,
    splits: Mapping[str, Sequence[Candidate]],
    manifest: dict[str, Any],
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        outputs: dict[str, dict[str, Any]] = {}
        for name in ("train", "validation", "test"):
            path = stage / f"{name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for candidate in splits[name]:
                    stream.write(candidate.payload)
                    stream.write("\n")
            outputs[name] = {
                "path": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "records": len(splits[name]),
            }
        manifest["outputs"] = outputs
        manifest_path = stage / "quality-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return target / "quality-manifest.json"


def materialize_reasoning_corpus(
    *,
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    tokenizer: ChatTemplateTokenizer,
    tokenizer_path: str | Path,
    source_commit: str = DEFAULT_SOURCE_COMMIT,
    expected_input_hashes: Mapping[str | Path, str] | None = None,
    require_expected_hashes: bool = True,
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: str = DEFAULT_SEED,
) -> Path:
    """Build the exact strict JSONL splits and return the quality-manifest path."""

    paths = tuple(Path(path) for path in input_paths)
    if not paths:
        raise ValueError("at least one prepared input is required")
    if len({str(path.resolve()) for path in paths}) != len(paths):
        raise ValueError("prepared input paths must be unique")
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA-1")
    total = sum(split_counts.values())
    expected_hashes = expected_input_hashes or {}
    tokenizer_root = Path(tokenizer_path)
    tokenizer_identity = _tokenizer_identity(tokenizer_root)

    with tempfile.TemporaryDirectory(prefix="molor-reasoning-corpus-") as temporary:
        database = Path(temporary) / "candidates.sqlite3"
        # sqlite3.Connection's own context manager commits or rolls back but does not close the
        # handle.  Explicit closing is required before Windows can remove the temporary database.
        with closing(sqlite3.connect(database)) as connection:
            _create_candidate_table(connection)
            inputs, rejections, eligible_before_dedup = _stream_inputs(
                connection,
                input_paths=paths,
                expected_hashes=expected_hashes,
                require_expected_hashes=require_expected_hashes,
                tokenizer=tokenizer,
                max_length=max_length,
                seed=seed,
            )
            canonical_count = int(
                connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            )
            splits, per_source, quotas_by_split = _select_candidates(
                connection, split_counts=split_counts
            )

    selected = [candidate for candidates in splits.values() for candidate in candidates]
    group_sets = {
        name: {candidate.prompt_group for candidate in candidates}
        for name, candidates in splits.items()
    }
    overlaps = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError("normalized prompt groups overlap across splits")

    source_sets = {
        name: {candidate.source for candidate in candidates}
        for name, candidates in splits.items()
    }
    source_overlaps = {
        "train_validation": len(source_sets["train"] & source_sets["validation"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "validation_test": len(source_sets["validation"] & source_sets["test"]),
    }
    if any(source_overlaps.values()):
        raise AssertionError("upstream sources overlap across splits")

    per_split_sources = {
        name: dict(sorted(Counter(candidate.source for candidate in candidates).items()))
        for name, candidates in splits.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "source_dataset": "Crownelius/Complete-FABLE.5-traces-2M",
            "source_commit": source_commit,
            "inputs": inputs,
            "allowlisted_sources": [
                {"source": source, "license": ALLOWED_SOURCE_LICENSES[source]}
                for source in ALLOWED_SOURCES
            ],
            "tokenizer": tokenizer_identity,
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
        },
        "contract": {
            "seed": seed,
            "max_rendered_tokens": max_length,
            "truncation": False,
            "assistant_target": "<think>\\n{thinking}\\n</think>\\n\\n{content}",
            "chat_template_enable_thinking": True,
            "source_system_prompts": "discarded",
            "replacement_system_prompt": NEUTRAL_SYSTEM_PROMPT,
            "replacement_system_prompt_sha256": _sha256_bytes(
                NEUTRAL_SYSTEM_PROMPT.encode("utf-8")
            ),
            "prompt_group_normalization": "Unicode NFKC, casefold, collapse whitespace",
            "source_record_shape": (
                "zero or more leading system messages, then exactly one user and one final "
                "assistant with nonempty thinking and content"
            ),
            "teacher_identity_filter": _TEACHER_IDENTITY_RE.pattern,
            "benchmark_name_filter": _BENCHMARK_RE.pattern,
        },
        "quality": {
            "input_records": sum(item["records"] for item in inputs),
            "rejections": dict(sorted(rejections.items())),
            "eligible_before_prompt_group_dedup": eligible_before_dedup,
            "unique_eligible_prompt_groups": canonical_count,
            "prompt_group_duplicates_removed": eligible_before_dedup - canonical_count,
            "selected_records": len(selected),
            "all_selected_at_or_below_max_length": all(
                candidate.token_count <= max_length for candidate in selected
            ),
            "all_targets_have_explicit_thinking_block": True,
            "prompt_group_split_overlap": {
                **overlaps,
                "zero_overlap": not any(overlaps.values()),
            },
            "source_split_overlap": {
                **source_overlaps,
                "zero_overlap": not any(source_overlaps.values()),
            },
            "unique_selected_prompt_groups": len(set().union(*group_sets.values())),
        },
        "selection": {
            "policy": (
                "source-disjoint split quotas, deterministic within-source prompt-group rank, "
                "same-source fill only, fail closed on shortfall"
            ),
            "source_order": list(ALLOWED_SOURCES),
            "target_records": total,
            "quotas_by_split": quotas_by_split,
            "per_source": per_source,
        },
        "splits": {
            name: {
                "records": len(candidates),
                "per_source": per_split_sources[name],
                "prompt_group_set_sha256": _sha256_bytes(
                    ("\n".join(sorted(group_sets[name])) + "\n").encode("ascii")
                ),
                "token_stats": _token_stats(
                    candidate.token_count for candidate in candidates
                ),
            }
            for name, candidates in splits.items()
        },
        "token_stats": _token_stats(candidate.token_count for candidate in selected),
    }
    return _write_outputs(Path(output_dir), splits=splits, manifest=manifest)


def _parse_expected_hashes(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expected input hashes must use PATH=SHA256")
        raw_path, digest = value.rsplit("=", 1)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"invalid SHA-256 for {raw_path!r}")
        result[str(Path(raw_path).resolve())] = digest.lower()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="prepared Complete-FABLE JSONL (repeatable; defaults to train and validation)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--train-count", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument(
        "--validation-count", type=int, default=DEFAULT_SPLIT_COUNTS["validation"]
    )
    parser.add_argument("--test-count", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument(
        "--expected-input-sha256",
        action="append",
        default=[],
        metavar="PATH=SHA256",
        help="expected digest for each custom input",
    )
    parser.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help="permit inputs without a preregistered SHA-256 (not for the production corpus)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = tuple(args.input) if args.input else DEFAULT_INPUTS
    try:
        supplied_hashes = _parse_expected_hashes(args.expected_input_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    expected_hashes: dict[str | Path, str] = {}
    if not args.input:
        expected_hashes.update(DEFAULT_INPUT_HASHES)
    expected_hashes.update(supplied_hashes)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without the hf extra
        parser.error(f"transformers is required: {exc}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    manifest = materialize_reasoning_corpus(
        input_paths=inputs,
        output_dir=args.output,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        source_commit=args.source_commit,
        expected_input_hashes=expected_hashes,
        require_expected_hashes=not args.allow_unverified_inputs,
        split_counts={
            "train": args.train_count,
            "validation": args.validation_count,
            "test": args.test_count,
        },
        max_length=args.max_length,
        seed=args.seed,
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
