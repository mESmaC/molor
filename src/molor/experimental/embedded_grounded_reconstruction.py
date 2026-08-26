"""Build honest grounded-reconstruction SFT from safe embedded references.

The only training target created here is a contiguous, verbatim slice of an
already-materialized ``reference_cpt`` passage.  Long passages are split at
deterministic character boundaries selected with the pinned Qwen chat
tokenizer.  Concatenating a source row's targets in ``chunk_index`` order must
reproduce its original text exactly.

This is reference reconstruction for domain adaptation.  It is not synthetic
question answering and does not establish the correctness or usefulness of the
source material.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from molor.experimental.length_tier_shards import (
    ChatTemplateTokenizer,
    render_token_count,
    tokenizer_identity,
)
from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256


SCHEMA_VERSION = "molor.embedded-grounded-reconstruction.v1"
PROVENANCE_SCHEMA_VERSION = "molor.embedded-grounded-reconstruction-provenance.v1"
ACCOUNTING_SCHEMA_VERSION = "molor.embedded-grounded-reconstruction-accounting.v1"
REPORT_SCHEMA_VERSION = "molor.embedded-grounded-reconstruction-report.v1"
SOURCE_SCHEMA_VERSION = "molor.embedded-source-experts.v1"
TASK_TYPE = "grounded_reference_reconstruction"

LEAVES = ("embedded-code-generation", "embedded-knowledge")
SPLITS = ("train", "validation", "test")
OUTPUT_TIERS = ("le-1024", "1025-4096")
MAXIMUM_RENDERED_TOKENS = 4_096
INITIAL_CHARACTER_WINDOW = 16_384

SYSTEM_PROMPT = (
    "This is grounded source reconstruction for domain adaptation, not question "
    "answering. Return the recorded source chunk verbatim. Do not add, remove, "
    "summarize, explain, or infer content. Successful reconstruction does not prove "
    "source correctness or answer quality."
)

_PROVENANCE_FIELDS = frozenset(
    {
        "content_sha256",
        "disposition",
        "file_path",
        "group_id",
        "leaf",
        "record_sha256",
        "repo",
        "source",
        "source_line",
        "source_split",
        "split",
        "upstream_id",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReconstructionPlan:
    """Pinned aggregate and tokenizer identities for one materialization."""

    source_manifest: Path
    source_manifest_sha256: str
    tokenizer_path: Path
    tokenizer_aggregate_sha256: str
    chat_template_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_manifest", self.source_manifest.resolve())
        object.__setattr__(self, "tokenizer_path", self.tokenizer_path.resolve())
        object.__setattr__(
            self,
            "source_manifest_sha256",
            _sha256(self.source_manifest_sha256, "source_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "tokenizer_aggregate_sha256",
            _sha256(self.tokenizer_aggregate_sha256, "tokenizer_aggregate_sha256"),
        )
        object.__setattr__(
            self,
            "chat_template_sha256",
            _sha256(self.chat_template_sha256, "chat_template_sha256"),
        )


@dataclass(frozen=True, slots=True)
class _FilePin:
    path: Path
    relative_path: str
    sha256: str
    bytes: int
    records: int


@dataclass(frozen=True, slots=True)
class _InputShard:
    leaf: str
    split: str
    data: _FilePin
    provenance: _FilePin


@dataclass(frozen=True, slots=True)
class _SourceRow:
    ordinal: int
    leaf: str
    split: str
    input_line: int
    text: str
    provenance: dict[str, object]


@dataclass(frozen=True, slots=True)
class _Chunk:
    source: _SourceRow
    chunk_index: int
    character_start: int
    character_end: int
    target: str
    messages: list[dict[str, str]]
    rendered_tokens: int
    tier: str


def _sha256(value: object, field_name: str) -> str:
    digest = validate_sha256(value, field_name)  # type: ignore[arg-type]
    if value != digest:
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int | None]:
    return stat.st_size, stat.st_mtime_ns, getattr(stat, "st_ino", None)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"JSON contains unsupported numeric constant {value!r}")


def _strict_json(raw: bytes, field_name: str) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be strict UTF-8 JSON") from exc


def _verified_file(pin: _FilePin | None, path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pinned input must be a regular non-symlink file: {path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after) or len(raw) != before.st_size:
        raise RuntimeError(f"pinned input changed while it was read: {path}")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"pinned input SHA-256 mismatch: {path}")
    if pin is not None and len(raw) != pin.bytes:
        raise ValueError(f"pinned input byte-count mismatch: {path}")
    return raw


def _inside_exact(root: Path, relative: object, expected: str, field_name: str) -> Path:
    if relative != expected:
        raise ValueError(f"{field_name} must be exactly {expected!r}")
    candidate = (root / expected).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{field_name} escapes the source aggregate")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field_name} must resolve to a regular non-symlink file")
    return candidate


def _parse_pin(
    value: object,
    *,
    root: Path,
    expected_path: str,
    field_name: str,
) -> _FilePin:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes", "records"}:
        raise ValueError(f"{field_name} has an invalid file-pin contract")
    byte_count = value["bytes"]
    records = value["records"]
    if type(byte_count) is not int or byte_count < 1:
        raise ValueError(f"{field_name}.bytes must be a positive integer")
    if type(records) is not int or records < 1:
        raise ValueError(f"{field_name}.records must be a positive integer")
    return _FilePin(
        path=_inside_exact(root, value["path"], expected_path, f"{field_name}.path"),
        relative_path=expected_path,
        sha256=_sha256(value["sha256"], f"{field_name}.sha256"),
        bytes=byte_count,
        records=records,
    )


def _load_source_manifest(
    plan: ReconstructionPlan,
) -> tuple[tuple[_InputShard, ...], dict[str, object]]:
    raw = _verified_file(None, plan.source_manifest, plan.source_manifest_sha256)
    value = _strict_json(raw, "source manifest")
    if not isinstance(value, dict) or value.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported source manifest schema")
    raw_outputs = value.get("outputs")
    if not isinstance(raw_outputs, list):
        raise ValueError("source manifest outputs must be an array")
    root = plan.source_manifest.parent.resolve()
    selected: dict[tuple[str, str], _InputShard] = {}
    for index, output in enumerate(raw_outputs):
        if not isinstance(output, dict):
            continue
        leaf = output.get("leaf")
        modality = output.get("modality")
        if leaf not in LEAVES or modality != "reference_cpt":
            continue
        if set(output) != {"data", "leaf", "modality", "provenance", "split"}:
            raise ValueError(f"selected source output {index} has an invalid contract")
        split = output["split"]
        if split not in SPLITS:
            raise ValueError(f"selected source output {index} has an invalid split")
        key = (leaf, split)
        if key in selected:
            raise ValueError(f"source manifest repeats selected leaf/split {key!r}")
        data_relative = f"reference/{leaf}/{split}.jsonl"
        provenance_relative = f"reference/{leaf}/{split}.provenance.jsonl"
        data = _parse_pin(
            output["data"],
            root=root,
            expected_path=data_relative,
            field_name=f"outputs[{index}].data",
        )
        provenance = _parse_pin(
            output["provenance"],
            root=root,
            expected_path=provenance_relative,
            field_name=f"outputs[{index}].provenance",
        )
        if data.records != provenance.records:
            raise ValueError(f"selected source output {index} data/provenance counts differ")
        selected[key] = _InputShard(leaf=leaf, split=split, data=data, provenance=provenance)
    expected = {(leaf, split) for leaf in LEAVES for split in SPLITS}
    if set(selected) != expected:
        raise ValueError("source manifest must pin exactly one selected shard per leaf and split")
    shards = tuple(selected[(leaf, split)] for leaf in LEAVES for split in SPLITS)
    return shards, {
        "path": str(plan.source_manifest),
        "sha256": plan.source_manifest_sha256,
        "bytes": len(raw),
        "schema_version": SOURCE_SCHEMA_VERSION,
        "pin_verified": True,
    }


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _parse_source_row(
    raw_data: bytes,
    raw_provenance: bytes,
    *,
    shard: _InputShard,
    input_line: int,
    ordinal: int,
) -> _SourceRow:
    data = _strict_json(raw_data, f"{shard.data.relative_path}:{input_line}")
    if not isinstance(data, dict) or set(data) != {"text"} or not isinstance(data["text"], str):
        raise ValueError(f"source data line {input_line} must contain only string field 'text'")
    provenance = _strict_json(
        raw_provenance,
        f"{shard.provenance.relative_path}:{input_line}",
    )
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError(f"source provenance line {input_line} has an invalid contract")
    if provenance["disposition"] != "emitted":
        raise ValueError(f"source provenance line {input_line} is not emitted")
    if provenance["leaf"] != shard.leaf or provenance["split"] != shard.split:
        raise ValueError(f"source provenance line {input_line} disagrees with its shard")
    for field in (
        "repo",
        "file_path",
        "group_id",
        "source",
        "source_split",
        "upstream_id",
    ):
        _nonempty_string(provenance[field], f"source provenance line {input_line}.{field}")
    if type(provenance["source_line"]) is not int or provenance["source_line"] < 1:
        raise ValueError(f"source provenance line {input_line}.source_line is invalid")
    content_sha = _sha256(
        provenance["content_sha256"],
        f"source provenance line {input_line}.content_sha256",
    )
    _sha256(
        provenance["record_sha256"],
        f"source provenance line {input_line}.record_sha256",
    )
    text = data["text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != content_sha:
        raise ValueError(f"source data/provenance content mismatch at line {input_line}")
    return _SourceRow(
        ordinal=ordinal,
        leaf=shard.leaf,
        split=shard.split,
        input_line=input_line,
        text=text,
        provenance=dict(provenance),
    )


def _read_sources(
    shards: Sequence[_InputShard],
) -> tuple[list[_SourceRow], list[dict[str, object]], dict[str, str]]:
    rows: list[_SourceRow] = []
    receipts: list[dict[str, object]] = []
    content_owner: dict[str, _SourceRow] = {}
    group_split: dict[str, str] = {}
    upstream_owner: dict[tuple[str, str], _SourceRow] = {}
    source_file_sums: dict[str, str] = {}
    for shard in shards:
        data_raw = _verified_file(shard.data, shard.data.path, shard.data.sha256)
        provenance_raw = _verified_file(
            shard.provenance,
            shard.provenance.path,
            shard.provenance.sha256,
        )
        data_lines = data_raw.splitlines()
        provenance_lines = provenance_raw.splitlines()
        if len(data_lines) != shard.data.records or len(provenance_lines) != shard.provenance.records:
            raise ValueError(f"record-count mismatch for selected shard {shard.leaf}/{shard.split}")
        if len(data_lines) != len(provenance_lines):
            raise ValueError(f"data/provenance rows are not aligned for {shard.leaf}/{shard.split}")
        for input_line, (raw_data, raw_provenance) in enumerate(
            zip(data_lines, provenance_lines, strict=True),
            start=1,
        ):
            row = _parse_source_row(
                raw_data,
                raw_provenance,
                shard=shard,
                input_line=input_line,
                ordinal=len(rows),
            )
            content_sha = str(row.provenance["content_sha256"])
            owner = content_owner.get(content_sha)
            if owner is not None:
                if owner.split != row.split:
                    raise ValueError("opposite-split exact-content leakage in selected sources")
                raise ValueError("duplicate exact-content identity in selected sources")
            content_owner[content_sha] = row
            group_id = str(row.provenance["group_id"])
            previous_split = group_split.get(group_id)
            if previous_split is not None and previous_split != row.split:
                raise ValueError("opposite-split source-group leakage in selected sources")
            group_split[group_id] = row.split
            upstream_key = (row.leaf, str(row.provenance["upstream_id"]))
            if upstream_key in upstream_owner:
                raise ValueError("duplicate leaf/upstream identity in selected sources")
            upstream_owner[upstream_key] = row
            rows.append(row)
        receipts.append(
            {
                "leaf_id": shard.leaf,
                "split": shard.split,
                "data": {
                    "path": shard.data.relative_path,
                    "sha256": shard.data.sha256,
                    "bytes": shard.data.bytes,
                    "records": shard.data.records,
                    "stream_hash_verified": True,
                },
                "provenance": {
                    "path": shard.provenance.relative_path,
                    "sha256": shard.provenance.sha256,
                    "bytes": shard.provenance.bytes,
                    "records": shard.provenance.records,
                    "stream_hash_verified": True,
                },
                "row_alignment_verified": True,
            }
        )
        source_file_sums[shard.data.relative_path] = shard.data.sha256
        source_file_sums[shard.provenance.relative_path] = shard.provenance.sha256
    return rows, receipts, source_file_sums


def _user_prompt(source: _SourceRow, chunk_index: int) -> str:
    provenance = source.provenance
    return (
        "Task type: grounded source reconstruction (not question answering).\n"
        f"Repository: {json.dumps(provenance['repo'], ensure_ascii=False)}\n"
        f"File path: {json.dumps(provenance['file_path'], ensure_ascii=False)}\n"
        f"Leaf: {json.dumps(source.leaf, ensure_ascii=False)}\n"
        f"Chunk index: {chunk_index}\n"
        "Reproduce the recorded source chunk verbatim. Return no commentary."
    )


def _messages(source: _SourceRow, chunk_index: int, target: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(source, chunk_index)},
        {"role": "assistant", "content": target},
    ]


def _rendered_tokens(
    tokenizer: ChatTemplateTokenizer,
    source: _SourceRow,
    chunk_index: int,
    target: str,
) -> tuple[list[dict[str, str]], int]:
    messages = _messages(source, chunk_index, target)
    count = render_token_count(tokenizer, messages, template_mode="direct")
    if count < 1:
        raise ValueError("tokenizer returned an empty rendered conversation")
    return messages, count


def _tier(token_count: int) -> str:
    if 1 <= token_count <= 1_024:
        return "le-1024"
    if 1_025 <= token_count <= MAXIMUM_RENDERED_TOKENS:
        return "1025-4096"
    raise ValueError(f"rendered chat exceeds {MAXIMUM_RENDERED_TOKENS} tokens")


def _chunk_source(source: _SourceRow, tokenizer: ChatTemplateTokenizer) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    start = 0
    text = source.text
    if not text:
        messages, count = _rendered_tokens(tokenizer, source, 0, "")
        if count > MAXIMUM_RENDERED_TOKENS:
            raise ValueError("fixed reconstruction prompt exceeds the rendered-token limit")
        chunks.append(
            _Chunk(source, 0, 0, 0, "", messages, count, _tier(count))
        )
        return chunks

    while start < len(text):
        chunk_index = len(chunks)
        probe_end = min(len(text), start + INITIAL_CHARACTER_WINDOW)
        messages, count = _rendered_tokens(
            tokenizer,
            source,
            chunk_index,
            text[start:probe_end],
        )
        if count <= MAXIMUM_RENDERED_TOKENS:
            end = probe_end
        else:
            low = start + 1
            high = probe_end - 1
            best_end: int | None = None
            while low <= high:
                middle = (low + high) // 2
                _candidate_messages, candidate_count = _rendered_tokens(
                    tokenizer,
                    source,
                    chunk_index,
                    text[start:middle],
                )
                if candidate_count <= MAXIMUM_RENDERED_TOKENS:
                    best_end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best_end is None:
                raise ValueError("one source character cannot fit under the rendered-token limit")
            end = best_end
            messages, count = _rendered_tokens(
                tokenizer,
                source,
                chunk_index,
                text[start:end],
            )
        if end <= start or count > MAXIMUM_RENDERED_TOKENS:
            raise AssertionError("chunk boundary failed the rendered-token invariant")
        target = text[start:end]
        chunks.append(
            _Chunk(
                source=source,
                chunk_index=chunk_index,
                character_start=start,
                character_end=end,
                target=target,
                messages=messages,
                rendered_tokens=count,
                tier=_tier(count),
            )
        )
        start = end
    if "".join(chunk.target for chunk in chunks) != text:
        raise AssertionError("chunk concatenation does not reproduce the source text")
    if sum(len(chunk.target) for chunk in chunks) != len(text):
        raise AssertionError("source-character accounting does not balance")
    return chunks


def _tokenizer_receipt(plan: ReconstructionPlan) -> dict[str, object]:
    receipt = tokenizer_identity(plan.tokenizer_path)
    if receipt["aggregate_sha256"] != plan.tokenizer_aggregate_sha256:
        raise ValueError("tokenizer aggregate SHA-256 mismatch")
    if receipt["chat_template"]["sha256"] != plan.chat_template_sha256:
        raise ValueError("tokenizer chat-template SHA-256 mismatch")
    return receipt


def _write_rows(path: Path, rows: Iterable[Mapping[str, object] | bytes]) -> dict[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    records = 0
    with path.open("wb") as stream:
        for row in rows:
            payload = row if isinstance(row, bytes) else canonical_json_bytes(row)
            payload += b"\n"
            stream.write(payload)
            digest.update(payload)
            byte_count += len(payload)
            records += 1
    return {"sha256": digest.hexdigest(), "bytes": byte_count, "records": records}


def _chunk_provenance(chunk: _Chunk) -> dict[str, object]:
    source = chunk.source
    upstream = source.provenance
    message_row = {"messages": chunk.messages}
    target_sha = hashlib.sha256(chunk.target.encode("utf-8")).hexdigest()
    identity = {
        "source_content_sha256": upstream["content_sha256"],
        "leaf_id": source.leaf,
        "split": source.split,
        "chunk_index": chunk.chunk_index,
        "character_start": chunk.character_start,
        "character_end": chunk.character_end,
        "assistant_target_sha256": target_sha,
    }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "task_type": TASK_TYPE,
        "reconstruction_id": fingerprint_json(identity),
        "leaf_id": source.leaf,
        "split": source.split,
        "group_id": upstream["group_id"],
        "repo": upstream["repo"],
        "file_path": upstream["file_path"],
        "source_upstream_id": upstream["upstream_id"],
        "source_content_sha256": upstream["content_sha256"],
        "source_record_sha256": upstream["record_sha256"],
        "source_input_line": source.input_line,
        "source_character_count": len(source.text),
        "chunk_index": chunk.chunk_index,
        "character_start": chunk.character_start,
        "character_end": chunk.character_end,
        "assistant_target_sha256": target_sha,
        "assistant_target_utf8_bytes": len(chunk.target.encode("utf-8")),
        "messages_sha256": fingerprint_json(message_row),
        "user_prompt_sha256": hashlib.sha256(
            chunk.messages[1]["content"].encode("utf-8")
        ).hexdigest(),
        "rendered_tokens": chunk.rendered_tokens,
        "tier": chunk.tier,
        "assistant_target_is_contiguous_verbatim_source_text": True,
        "truncated": False,
        "is_synthetic_question_answering": False,
        "source_provenance": upstream,
    }


def _accounting_row(source: _SourceRow, chunks: Sequence[_Chunk]) -> dict[str, object]:
    reconstructed = "".join(chunk.target for chunk in chunks)
    source_sha = hashlib.sha256(source.text.encode("utf-8")).hexdigest()
    reconstructed_sha = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
    return {
        "schema_version": ACCOUNTING_SCHEMA_VERSION,
        "status": "emitted",
        "reason": "verbatim_source_reconstructed_in_token_bounded_chunks",
        "leaf_id": source.leaf,
        "split": source.split,
        "group_id": source.provenance["group_id"],
        "repo": source.provenance["repo"],
        "file_path": source.provenance["file_path"],
        "source_upstream_id": source.provenance["upstream_id"],
        "source_content_sha256": source.provenance["content_sha256"],
        "source_record_sha256": source.provenance["record_sha256"],
        "source_input_line": source.input_line,
        "source_characters": len(source.text),
        "source_utf8_bytes": len(source.text.encode("utf-8")),
        "emitted_chunks": len(chunks),
        "emitted_characters": sum(len(chunk.target) for chunk in chunks),
        "reconstructed_text_sha256": reconstructed_sha,
        "source_text_sha256": source_sha,
        "chunk_indices": [chunk.chunk_index for chunk in chunks],
        "character_boundaries": [
            [chunk.character_start, chunk.character_end] for chunk in chunks
        ],
        "concatenation_exact": reconstructed == source.text,
        "characters_omitted": 0,
        "characters_reordered": 0,
        "characters_duplicated": 0,
        "truncated": False,
        "deferred": False,
    }


def _pretty_json(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def materialize_embedded_grounded_reconstruction(
    *,
    plan: ReconstructionPlan,
    output_directory: str | Path,
    tokenizer: ChatTemplateTokenizer,
) -> dict[str, object]:
    """Atomically build the exact, split-preserving <=4096-token SFT view."""

    if not isinstance(plan, ReconstructionPlan):
        raise TypeError("plan must be a ReconstructionPlan")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    tokenizer_receipt = _tokenizer_receipt(plan)
    shards, source_manifest_receipt = _load_source_manifest(plan)
    sources, input_receipts, source_file_sums = _read_sources(shards)
    chunks_by_source: list[list[_Chunk]] = []
    all_chunks: list[_Chunk] = []
    for source in sources:
        source_chunks = _chunk_source(source, tokenizer)
        chunks_by_source.append(source_chunks)
        all_chunks.extend(source_chunks)

    if len(chunks_by_source) != len(sources):
        raise AssertionError("every selected source row must have a terminal reconstruction")
    accounting_rows = [
        _accounting_row(source, chunks)
        for source, chunks in zip(sources, chunks_by_source, strict=True)
    ]
    if not all(row["concatenation_exact"] is True for row in accounting_rows):
        raise AssertionError("a source failed exact reconstruction")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        output_receipts: list[dict[str, object]] = []
        sums: dict[str, str] = {}
        grouped: dict[tuple[str, str, str], list[_Chunk]] = defaultdict(list)
        for chunk in all_chunks:
            grouped[(chunk.source.leaf, chunk.source.split, chunk.tier)].append(chunk)

        for leaf in LEAVES:
            for split in SPLITS:
                for tier in OUTPUT_TIERS:
                    members = sorted(
                        grouped.get((leaf, split, tier), ()),
                        key=lambda item: (item.source.ordinal, item.chunk_index),
                    )
                    messages_relative = Path("messages") / leaf / split / f"{tier}.jsonl"
                    provenance_relative = (
                        Path("provenance") / leaf / split / f"{tier}.jsonl"
                    )
                    (stage / messages_relative).parent.mkdir(parents=True, exist_ok=True)
                    (stage / provenance_relative).parent.mkdir(parents=True, exist_ok=True)
                    message_receipt = _write_rows(
                        stage / messages_relative,
                        ({"messages": item.messages} for item in members),
                    )
                    provenance_receipt = _write_rows(
                        stage / provenance_relative,
                        (_chunk_provenance(item) for item in members),
                    )
                    if message_receipt["records"] != provenance_receipt["records"]:
                        raise AssertionError("messages/provenance emission counts diverged")
                    rendered = [item.rendered_tokens for item in members]
                    output_receipts.append(
                        {
                            "leaf_id": leaf,
                            "split": split,
                            "tier": tier,
                            "messages_path": messages_relative.as_posix(),
                            "messages_sha256": message_receipt["sha256"],
                            "messages_bytes": message_receipt["bytes"],
                            "provenance_path": provenance_relative.as_posix(),
                            "provenance_sha256": provenance_receipt["sha256"],
                            "provenance_bytes": provenance_receipt["bytes"],
                            "records": message_receipt["records"],
                            "minimum_rendered_tokens": min(rendered) if rendered else None,
                            "maximum_rendered_tokens": max(rendered) if rendered else None,
                        }
                    )
                    sums[messages_relative.as_posix()] = str(message_receipt["sha256"])
                    sums[provenance_relative.as_posix()] = str(provenance_receipt["sha256"])

        accounting_relative = Path("accounting.jsonl")
        accounting_receipt = _write_rows(stage / accounting_relative, accounting_rows)
        sums[accounting_relative.as_posix()] = str(accounting_receipt["sha256"])

        source_counts = Counter((row.leaf, row.split) for row in sources)
        chunk_counts = Counter((item.source.leaf, item.source.split) for item in all_chunks)
        tier_counts = Counter(
            (item.source.leaf, item.source.split, item.tier) for item in all_chunks
        )
        source_characters = sum(len(row.text) for row in sources)
        emitted_characters = sum(len(item.target) for item in all_chunks)
        by_leaf_split = []
        for leaf in LEAVES:
            for split in SPLITS:
                matching_sources = [row for row in sources if row.leaf == leaf and row.split == split]
                matching_chunks = [
                    item
                    for item in all_chunks
                    if item.source.leaf == leaf and item.source.split == split
                ]
                by_leaf_split.append(
                    {
                        "leaf_id": leaf,
                        "split": split,
                        "source_rows": source_counts[(leaf, split)],
                        "source_characters": sum(len(row.text) for row in matching_sources),
                        "emitted_chunks": chunk_counts[(leaf, split)],
                        "emitted_characters": sum(len(item.target) for item in matching_chunks),
                        "tiers": {
                            tier: tier_counts[(leaf, split, tier)] for tier in OUTPUT_TIERS
                        },
                    }
                )

        accounting = {
            "selected_input_rows": len(sources),
            "terminal_source_statuses": {
                "emitted": len(sources),
                "duplicate": 0,
                "quarantined": 0,
                "rejected": 0,
                "deferred": 0,
            },
            "emitted_message_records": len(all_chunks),
            "source_characters": source_characters,
            "emitted_characters": emitted_characters,
            "truncated_rows": 0,
            "deferred_rows": 0,
            "characters_omitted": 0,
            "characters_reordered": 0,
            "characters_duplicated": 0,
            "every_source_concatenation_exact": True,
            "fully_reconciled": len(sources) == len(accounting_rows),
            "zero_silent_omissions": (
                len(sources) == len(accounting_rows)
                and source_characters == emitted_characters
                and sum(source_counts.values()) == len(sources)
            ),
            "by_leaf_split": by_leaf_split,
        }
        if not accounting["zero_silent_omissions"]:
            raise AssertionError("aggregate source/output accounting does not reconcile")

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "task_type": TASK_TYPE,
            "accounting": accounting,
            "outputs": output_receipts,
            "source_manifest_sha256": plan.source_manifest_sha256,
            "tokenizer_aggregate_sha256": plan.tokenizer_aggregate_sha256,
            "maximum_rendered_tokens": MAXIMUM_RENDERED_TOKENS,
        }
        report_relative = Path("aggregate_report.json")
        report_receipt = _pretty_json(stage / report_relative, report)
        sums[report_relative.as_posix()] = str(report_receipt["sha256"])

        semantic_manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "task_contract": {
                "task_type": TASK_TYPE,
                "description": "grounded source reconstruction for domain adaptation",
                "input_modality": "reference_cpt",
                "selected_leaves": list(LEAVES),
                "assistant_target": "contiguous verbatim source-text chunk",
                "user_prompt_source_fields": [
                    "repo",
                    "file_path",
                    "leaf",
                    "chunk_index",
                ],
                "contains_synthetic_question_answering": False,
                "proves_source_correctness": False,
                "proves_answer_quality": False,
                "truncation": False,
                "deferred_rows_permitted": False,
                "maximum_rendered_tokens_inclusive": MAXIMUM_RENDERED_TOKENS,
                "chunk_boundary_algorithm": "bounded_character_probe_binary_v1",
                "initial_character_window": INITIAL_CHARACTER_WINDOW,
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
            "provenance": {
                "source_manifest": source_manifest_receipt,
                "tokenizer": {
                    **tokenizer_receipt,
                    "expected_aggregate_sha256": plan.tokenizer_aggregate_sha256,
                    "expected_chat_template_sha256": plan.chat_template_sha256,
                    "hashes_verified": True,
                },
                "inputs": input_receipts,
                "source_file_sha256": dict(sorted(source_file_sums.items())),
            },
            "accounting": accounting,
            "accounting_ledger": {
                "path": accounting_relative.as_posix(),
                **accounting_receipt,
            },
            "aggregate_report": {
                "path": report_relative.as_posix(),
                **report_receipt,
            },
            "outputs": output_receipts,
        }
        manifest = {
            **semantic_manifest,
            "content_sha256": fingerprint_json(semantic_manifest),
        }
        manifest_relative = Path("manifest.json")
        manifest_receipt = _pretty_json(stage / manifest_relative, manifest)
        sums[manifest_relative.as_posix()] = str(manifest_receipt["sha256"])

        sums_relative = Path("sha256sums.txt")
        sums_payload = "".join(
            f"{digest}  {relative}\n" for relative, digest in sorted(sums.items())
        ).encode("utf-8")
        (stage / sums_relative).write_bytes(sums_payload)

        os.replace(stage, output)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def directory_identity(root: str | Path) -> dict[str, str]:
    """Return content hashes for every regular file in one completed output."""

    directory = Path(root).resolve()
    if not directory.is_dir():
        raise ValueError(f"materialization is not a directory: {directory}")
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"materialization contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = _sha256_file(path)
    return result


def assert_identical_materializations(left: str | Path, right: str | Path) -> dict[str, str]:
    """Fail unless two independently built output trees are byte-identical."""

    left_identity = directory_identity(left)
    right_identity = directory_identity(right)
    if left_identity != right_identity:
        left_names = set(left_identity)
        right_names = set(right_identity)
        differing = sorted(
            name
            for name in left_names & right_names
            if left_identity[name] != right_identity[name]
        )
        raise ValueError(
            "materializations differ: "
            f"left_only={sorted(left_names - right_names)!r}, "
            f"right_only={sorted(right_names - left_names)!r}, "
            f"content={differing!r}"
        )
    return left_identity


def _load_local_tokenizer(path: Path) -> ChatTemplateTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional CLI dependency
        raise RuntimeError("transformers is required to run the reconstruction CLI") from exc
    return AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-aggregate-sha256", required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args(argv)
    plan = ReconstructionPlan(
        source_manifest=args.source_manifest,
        source_manifest_sha256=args.source_manifest_sha256,
        tokenizer_path=args.tokenizer,
        tokenizer_aggregate_sha256=args.tokenizer_aggregate_sha256,
        chat_template_sha256=args.chat_template_sha256,
    )
    manifest = materialize_embedded_grounded_reconstruction(
        plan=plan,
        output_directory=args.output,
        tokenizer=_load_local_tokenizer(plan.tokenizer_path),
    )
    comparison = None
    if args.compare_to is not None:
        comparison = {
            "files": len(assert_identical_materializations(args.output, args.compare_to)),
            "byte_identical": True,
        }
    print(
        json.dumps(
            {
                "accounting": manifest["accounting"],
                "comparison": comparison,
                "content_sha256": manifest["content_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ACCOUNTING_SCHEMA_VERSION",
    "LEAVES",
    "INITIAL_CHARACTER_WINDOW",
    "MAXIMUM_RENDERED_TOKENS",
    "OUTPUT_TIERS",
    "PROVENANCE_SCHEMA_VERSION",
    "ReconstructionPlan",
    "SCHEMA_VERSION",
    "SPLITS",
    "SYSTEM_PROMPT",
    "TASK_TYPE",
    "assert_identical_materializations",
    "directory_identity",
    "main",
    "materialize_embedded_grounded_reconstruction",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
