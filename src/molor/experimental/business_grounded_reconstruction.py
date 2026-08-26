"""Materialize honest grounded-reconstruction SFT from business references.

Each assistant target is the canonical source passage verbatim.  The deterministic
user request names only the source title and business-taxonomy topic.  This is a
domain-adaptation reconstruction task, not synthetic question answering, and its
existence is not evidence that an answer is correct or useful.
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

from molor.experimental.business_reference_corpus import LEAVES, SPLITS
from molor.experimental.length_tier_shards import (
    ChatTemplateTokenizer,
    render_token_count,
    tier_for_token_count,
    tokenizer_identity,
)
from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256


SCHEMA_VERSION = "molor.business-grounded-reconstruction.v1"
PROVENANCE_SCHEMA_VERSION = "molor.business-grounded-reconstruction-provenance.v1"
ACCOUNTING_SCHEMA_VERSION = "molor.business-grounded-reconstruction-accounting.v1"
REFERENCE_SCHEMA_VERSION = "molor.business-reference-corpus.v1"
REFERENCE_PROVENANCE_SCHEMA_VERSION = "molor.business-reference-document-provenance.v1"
TASK_TYPE = "grounded_reference_reconstruction"

SYSTEM_PROMPT = (
    "This is grounded reference reconstruction, not synthetic question answering. "
    "Return the recorded source passage verbatim. Do not add, remove, summarize, or infer "
    "content. Successful reconstruction does not prove factual accuracy or answer quality."
)

TOPIC_LABELS = {
    "business-core": "general business and startups",
    "business-finance": "business finance",
    "growth-gtm": "growth and go-to-market",
    "investing-markets": "investing and markets",
    "funding-capital": "startup funding and capital",
    "business-operations": "business operations",
}

OUTPUT_TIERS = ("le-1024", "1025-4096", "4097-8192", "8193-32768")
SPLIT_PRECEDENCE = ("test", "validation", "train")
TERMINAL_STATUSES = ("emitted", "duplicate", "quarantined", "rejected", "deferred")
MAXIMUM_RECORD_BYTES = 16 * 1024 * 1024
RESERVED_CHAT_MARKERS = ("<|im_start|>", "<|im_end|>")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReconstructionPlan:
    """Pinned reference corpus and Qwen tokenizer identities."""

    reference_manifest: Path
    reference_manifest_sha256: str
    tokenizer_path: Path
    tokenizer_aggregate_sha256: str
    chat_template_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_manifest", self.reference_manifest.resolve())
        object.__setattr__(self, "tokenizer_path", self.tokenizer_path.resolve())
        object.__setattr__(
            self,
            "reference_manifest_sha256",
            _sha256(self.reference_manifest_sha256, "reference_manifest_sha256"),
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
class _InputShard:
    leaf_id: str
    leaf_revision: str
    split: str
    path: Path
    relative_path: str
    sha256: str
    records: int
    bytes: int


@dataclass(frozen=True, slots=True)
class _SourceProvenance:
    source_id: str
    artifact_path: str
    artifact_sha256: str
    family_sha256: str
    internal_reference: str
    chunk_index: int
    leaf_id: str
    split: str
    sidecar_line: int


@dataclass(slots=True)
class _Disposition:
    shard_index: int
    shard: _InputShard
    input_line: int
    raw_record_sha256: str
    status: str
    reason: str
    document_id: str | None = None
    exact_document_sha256: str | None = None
    source: _SourceProvenance | None = None
    rendered_tokens: int | None = None
    tier: str | None = None
    messages_payload: bytes | None = None
    messages_sha256: str | None = None
    user_prompt_sha256: str | None = None
    duplicate_of: dict[str, object] | None = None


def _sha256(value: object, field_name: str) -> str:
    digest = validate_sha256(value, field_name)  # type: ignore[arg-type]
    if value != digest:
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
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


def _inside(root: Path, relative: str, field_name: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError(f"{field_name} must be a non-empty relative path")
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"{field_name} must be relative to the reference corpus")
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field_name} escapes the reference corpus root")
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{field_name} must resolve to a regular non-symlink file")
    return resolved


def _verified_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pinned input must be a regular non-symlink file: {path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after) or len(raw) != before.st_size:
        raise RuntimeError(f"pinned input changed while it was read: {path}")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"pinned input SHA-256 mismatch: {path}")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError(f"pinned input byte count mismatch: {path}")
    return raw


def _load_reference_manifest(
    plan: ReconstructionPlan,
) -> tuple[dict[str, object], tuple[_InputShard, ...], dict[str, object]]:
    manifest_raw = _verified_file(
        plan.reference_manifest,
        plan.reference_manifest_sha256,
    )
    value = _strict_json(manifest_raw, "reference manifest")
    if not isinstance(value, dict) or value.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported reference manifest schema")
    claimed = _sha256(value.get("content_sha256"), "manifest.content_sha256")
    semantic = dict(value)
    semantic.pop("content_sha256")
    if fingerprint_json(semantic) != claimed:
        raise ValueError("reference manifest semantic content SHA-256 mismatch")
    root = plan.reference_manifest.parent.resolve()
    raw_shards = value.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("reference manifest shards must be an array")
    shards: list[_InputShard] = []
    for index, item in enumerate(raw_shards):
        if not isinstance(item, dict) or set(item) != {
            "leaf_id",
            "leaf_revision",
            "split",
            "path",
            "sha256",
            "bytes",
            "records",
        }:
            raise ValueError(f"reference manifest shard {index} has an invalid contract")
        leaf = item["leaf_id"]
        split = item["split"]
        revision = item["leaf_revision"]
        if leaf not in LEAVES or split not in SPLITS or not isinstance(revision, str):
            raise ValueError(f"reference manifest shard {index} has invalid routing metadata")
        records = item["records"]
        byte_count = item["bytes"]
        if type(records) is not int or records < 1 or type(byte_count) is not int or byte_count < 1:
            raise ValueError(f"reference manifest shard {index} has invalid counts")
        relative = item["path"]
        path = _inside(root, relative, f"manifest.shards[{index}].path")
        shards.append(
            _InputShard(
                leaf_id=leaf,
                leaf_revision=revision,
                split=split,
                path=path,
                relative_path=relative,
                sha256=_sha256(item["sha256"], f"manifest.shards[{index}].sha256"),
                records=records,
                bytes=byte_count,
            )
        )
    observed_pairs = {(item.leaf_id, item.split) for item in shards}
    expected_pairs = {(leaf, split) for leaf in LEAVES for split in SPLITS}
    if len(shards) != len(observed_pairs) or observed_pairs != expected_pairs:
        raise ValueError("reference manifest must contain exactly one shard per leaf and split")

    sidecars = value.get("provenance_sidecars")
    if not isinstance(sidecars, dict) or not isinstance(sidecars.get("documents"), dict):
        raise ValueError("reference manifest must pin its document provenance sidecar")
    descriptor = sidecars["documents"]
    if set(descriptor) != {"path", "sha256", "bytes", "records"}:
        raise ValueError("document provenance descriptor has an invalid contract")
    provenance_path = _inside(root, descriptor["path"], "provenance_sidecars.documents.path")
    provenance_descriptor = {
        "path": provenance_path,
        "relative_path": descriptor["path"],
        "sha256": _sha256(descriptor["sha256"], "provenance_sidecars.documents.sha256"),
        "bytes": descriptor["bytes"],
        "records": descriptor["records"],
    }
    if (
        type(provenance_descriptor["bytes"]) is not int
        or type(provenance_descriptor["records"]) is not int
    ):
        raise ValueError("document provenance descriptor counts must be integers")
    return value, tuple(shards), provenance_descriptor


def _load_source_provenance(
    descriptor: Mapping[str, object],
) -> tuple[dict[str, _SourceProvenance], dict[str, object]]:
    path = descriptor["path"]
    if not isinstance(path, Path):
        raise TypeError("resolved source-provenance path must be a Path")
    raw = _verified_file(
        path,
        descriptor["sha256"],  # type: ignore[arg-type]
        descriptor["bytes"],  # type: ignore[arg-type]
    )
    lines = raw.splitlines()
    if len(lines) != descriptor["records"]:
        raise ValueError("source-provenance record count mismatch")
    status_counts: Counter[str] = Counter()
    emitted: dict[str, _SourceProvenance] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        value = _strict_json(raw_line, f"source provenance line {line_number}")
        if not isinstance(value, dict):
            raise ValueError(f"source provenance line {line_number} must be an object")
        if value.get("schema_version") != REFERENCE_PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"source provenance line {line_number} has an invalid schema")
        status = value.get("status")
        if not isinstance(status, str):
            raise ValueError(f"source provenance line {line_number} has no status")
        status_counts[status] += 1
        if status != "emitted":
            continue
        document_id = value.get("content_sha256")
        source_id = value.get("source_id")
        artifact_path = value.get("artifact_path")
        artifact_sha256 = value.get("artifact_sha256")
        family_sha256 = value.get("family_sha256")
        internal_reference = value.get("internal_reference")
        chunk_index = value.get("chunk_index")
        leaf_id = value.get("leaf_id")
        split = value.get("split")
        if (
            not isinstance(document_id, str)
            or _SHA256_RE.fullmatch(document_id) is None
            or not isinstance(source_id, str)
            or not isinstance(artifact_path, str)
            or not isinstance(artifact_sha256, str)
            or _SHA256_RE.fullmatch(artifact_sha256) is None
            or not isinstance(family_sha256, str)
            or _SHA256_RE.fullmatch(family_sha256) is None
            or not isinstance(internal_reference, str)
            or type(chunk_index) is not int
            or leaf_id not in LEAVES
            or split not in SPLITS
        ):
            raise ValueError(f"emitted source provenance line {line_number} is malformed")
        if document_id in emitted:
            raise ValueError(f"emitted source provenance repeats document ID {document_id}")
        emitted[document_id] = _SourceProvenance(
            source_id=source_id,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            family_sha256=family_sha256,
            internal_reference=internal_reference,
            chunk_index=chunk_index,
            leaf_id=leaf_id,
            split=split,
            sidecar_line=line_number,
        )
    return emitted, {
        "path": descriptor["relative_path"],
        "sha256": descriptor["sha256"],
        "bytes": descriptor["bytes"],
        "records": len(lines),
        "status_accounting": dict(sorted(status_counts.items())),
        "emitted_index_records": len(emitted),
        "zero_silent_omissions": len(lines) == sum(status_counts.values()),
    }


def _parse_document(
    raw_line: bytes,
) -> tuple[str, str, dict[str, str] | None, str | None, str | None]:
    if len(raw_line) > MAXIMUM_RECORD_BYTES:
        return "rejected", "record_too_large", None, None, None
    try:
        value = _strict_json(raw_line, "reference document")
    except ValueError:
        return "rejected", "invalid_json", None, None, None
    if not isinstance(value, dict) or set(value) != {"document_id", "title", "text"}:
        return "rejected", "row_not_document_only", None, None, None
    document_id = value["document_id"]
    title = value["title"]
    text = value["text"]
    if not isinstance(document_id, str) or _SHA256_RE.fullmatch(document_id) is None:
        return "quarantined", "invalid_document_id", None, None, None
    if not isinstance(title, str) or not isinstance(text, str):
        return "quarantined", "invalid_document_field_type", None, document_id, None
    if not text.strip():
        return "quarantined", "empty_source_passage", None, document_id, None
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != document_id:
        return "quarantined", "document_id_text_sha256_mismatch", None, document_id, None
    if any(marker in title or marker in text for marker in RESERVED_CHAT_MARKERS):
        return "quarantined", "reserved_chat_marker_in_source", None, document_id, None
    exact = hashlib.sha256(canonical_json_bytes({"title": title, "text": text})).hexdigest()
    return "valid", "canonical_reference", value, document_id, exact


def _user_prompt(title: str, leaf_id: str) -> str:
    encoded_title = json.dumps(title, ensure_ascii=False)
    return (
        "Task type: grounded reference reconstruction (not question answering).\n"
        f"Business topic: {TOPIC_LABELS[leaf_id]}\n"
        f"Recorded source title: {encoded_title}\n"
        "Reproduce the recorded source passage verbatim as the briefing. Do not add analysis, "
        "claims, or alterations."
    )


def _messages(title: str, text: str, leaf_id: str) -> tuple[list[dict[str, str]], str]:
    user = _user_prompt(title, leaf_id)
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": text},
        ],
        user,
    )


def _duplicate_owner(disposition: _Disposition) -> dict[str, object]:
    return {
        "document_id": disposition.document_id,
        "leaf_id": disposition.shard.leaf_id,
        "split": disposition.shard.split,
        "input_path": disposition.shard.relative_path,
        "input_line": disposition.input_line,
    }


def _scan_shards(
    shards: Sequence[_InputShard],
    source_index: Mapping[str, _SourceProvenance],
    tokenizer: ChatTemplateTokenizer,
) -> tuple[list[_Disposition], set[str], list[dict[str, object]]]:
    leaf_order = {leaf: index for index, leaf in enumerate(LEAVES)}
    split_order = {split: index for index, split in enumerate(SPLIT_PRECEDENCE)}
    ordered = sorted(shards, key=lambda item: (split_order[item.split], leaf_order[item.leaf_id]))
    document_owners: dict[str, _Disposition] = {}
    exact_owners: dict[str, _Disposition] = {}
    family_owners: dict[str, _Disposition] = {}
    consumed_source_ids: set[str] = set()
    dispositions: list[_Disposition] = []
    shard_receipts: list[dict[str, object]] = []
    for shard_index, shard in enumerate(ordered):
        before = shard.path.stat()
        file_hash = hashlib.sha256()
        observed = 0
        shard_status: Counter[str] = Counter()
        with shard.path.open("rb") as stream:
            for input_line, raw_line in enumerate(stream, start=1):
                file_hash.update(raw_line)
                observed += 1
                raw_hash = hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest()
                status, reason, document, document_id, exact_sha = _parse_document(raw_line)
                disposition = _Disposition(
                    shard_index=shard_index,
                    shard=shard,
                    input_line=input_line,
                    raw_record_sha256=raw_hash,
                    status=status,
                    reason=reason,
                    document_id=document_id,
                    exact_document_sha256=exact_sha,
                )
                dispositions.append(disposition)
                if document is None:
                    shard_status[disposition.status] += 1
                    continue
                if document_id is None or exact_sha is None:
                    raise AssertionError("valid reference document has no identity")
                id_owner = document_owners.get(document_id)
                if id_owner is not None:
                    disposition.duplicate_of = _duplicate_owner(id_owner)
                    if id_owner.shard.split != shard.split:
                        disposition.status = "quarantined"
                        disposition.reason = "opposite_split_document_id_match"
                    elif id_owner.exact_document_sha256 != exact_sha:
                        disposition.status = "quarantined"
                        disposition.reason = "document_id_payload_conflict"
                    else:
                        disposition.status = "duplicate"
                        disposition.reason = "document_id_duplicate"
                    shard_status[disposition.status] += 1
                    continue
                exact_owner = exact_owners.get(exact_sha)
                if exact_owner is not None:
                    disposition.duplicate_of = _duplicate_owner(exact_owner)
                    if exact_owner.shard.split != shard.split:
                        disposition.status = "quarantined"
                        disposition.reason = "opposite_split_exact_document_match"
                    else:
                        disposition.status = "duplicate"
                        disposition.reason = "exact_document_duplicate"
                    shard_status[disposition.status] += 1
                    continue
                source = source_index.get(document_id)
                if source is None:
                    disposition.status = "quarantined"
                    disposition.reason = "missing_exact_source_provenance"
                    shard_status[disposition.status] += 1
                    continue
                consumed_source_ids.add(document_id)
                disposition.source = source
                if source.leaf_id != shard.leaf_id or source.split != shard.split:
                    disposition.status = "quarantined"
                    disposition.reason = "source_provenance_route_mismatch"
                    shard_status[disposition.status] += 1
                    continue
                family_owner = family_owners.get(source.family_sha256)
                if family_owner is not None and family_owner.shard.split != shard.split:
                    disposition.status = "quarantined"
                    disposition.reason = "opposite_split_source_family_match"
                    disposition.duplicate_of = _duplicate_owner(family_owner)
                    shard_status[disposition.status] += 1
                    continue
                messages, user = _messages(document["title"], document["text"], shard.leaf_id)
                try:
                    rendered_tokens = render_token_count(
                        tokenizer,
                        messages,
                        template_mode="direct",
                    )
                except (TypeError, ValueError):
                    disposition.status = "quarantined"
                    disposition.reason = "qwen_chat_template_render_failure"
                    shard_status[disposition.status] += 1
                    continue
                tier = tier_for_token_count(rendered_tokens)
                if tier == "deferred-gt32768":
                    disposition.status = "deferred"
                    disposition.reason = "rendered_length_gt_32768"
                    disposition.rendered_tokens = rendered_tokens
                    disposition.tier = tier
                    shard_status[disposition.status] += 1
                    continue
                messages_payload = canonical_json_bytes({"messages": messages})
                if messages[-1]["content"] != document["text"]:
                    raise AssertionError("assistant target changed from the source passage")
                disposition.status = "emitted"
                disposition.reason = "verbatim_grounded_reference_reconstruction"
                disposition.rendered_tokens = rendered_tokens
                disposition.tier = tier
                disposition.messages_payload = messages_payload
                disposition.messages_sha256 = hashlib.sha256(messages_payload).hexdigest()
                disposition.user_prompt_sha256 = hashlib.sha256(user.encode("utf-8")).hexdigest()
                document_owners[document_id] = disposition
                exact_owners[exact_sha] = disposition
                family_owners.setdefault(source.family_sha256, disposition)
                shard_status[disposition.status] += 1
        after = shard.path.stat()
        if _stat_identity(before) != _stat_identity(after):
            raise RuntimeError(f"reference shard changed while it was read: {shard.path}")
        if file_hash.hexdigest() != shard.sha256:
            raise ValueError(f"reference shard SHA-256 mismatch: {shard.relative_path}")
        if observed != shard.records or after.st_size != shard.bytes:
            raise ValueError(f"reference shard count or byte pin mismatch: {shard.relative_path}")
        shard_receipts.append(
            {
                "leaf_id": shard.leaf_id,
                "leaf_revision": shard.leaf_revision,
                "split": shard.split,
                "path": shard.relative_path,
                "sha256": shard.sha256,
                "bytes": shard.bytes,
                "records": observed,
                "accounting": _complete_status_counts(shard_status),
            }
        )
    return dispositions, consumed_source_ids, shard_receipts


def _complete_status_counts(counter: Mapping[str, int]) -> dict[str, int]:
    return {status: int(counter.get(status, 0)) for status in TERMINAL_STATUSES}


def _accounting_row(disposition: _Disposition) -> dict[str, object]:
    return {
        "schema_version": ACCOUNTING_SCHEMA_VERSION,
        "input_path": disposition.shard.relative_path,
        "input_sha256": disposition.shard.sha256,
        "input_line": disposition.input_line,
        "raw_record_sha256": disposition.raw_record_sha256,
        "leaf_id": disposition.shard.leaf_id,
        "leaf_revision": disposition.shard.leaf_revision,
        "split": disposition.shard.split,
        "document_id": disposition.document_id,
        "exact_document_sha256": disposition.exact_document_sha256,
        "status": disposition.status,
        "reason": disposition.reason,
        "rendered_tokens": disposition.rendered_tokens,
        "tier": disposition.tier,
        "duplicate_of": disposition.duplicate_of,
    }


def _provenance_row(disposition: _Disposition) -> dict[str, object]:
    source = disposition.source
    if source is None or disposition.document_id is None:
        raise AssertionError("emitted reconstruction has no exact source provenance")
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "task_type": TASK_TYPE,
        "is_synthetic_question_answering": False,
        "proves_answer_quality": False,
        "assistant_target_is_verbatim_source_passage": True,
        "source_id": source.source_id,
        "source_artifact_path": source.artifact_path,
        "source_artifact_sha256": source.artifact_sha256,
        "source_family_sha256": source.family_sha256,
        "source_internal_reference": source.internal_reference,
        "source_chunk_index": source.chunk_index,
        "source_provenance_line": source.sidecar_line,
        "document_id": disposition.document_id,
        "assistant_target_sha256": disposition.document_id,
        "leaf_id": disposition.shard.leaf_id,
        "leaf_revision": disposition.shard.leaf_revision,
        "split": disposition.shard.split,
        "input_path": disposition.shard.relative_path,
        "input_sha256": disposition.shard.sha256,
        "input_line": disposition.input_line,
        "messages_sha256": disposition.messages_sha256,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": disposition.user_prompt_sha256,
        "rendered_tokens": disposition.rendered_tokens,
        "tier": disposition.tier,
    }


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


def _tokenizer_receipt(plan: ReconstructionPlan) -> dict[str, object]:
    receipt = tokenizer_identity(plan.tokenizer_path)
    if receipt["aggregate_sha256"] != plan.tokenizer_aggregate_sha256:
        raise ValueError("tokenizer aggregate SHA-256 mismatch")
    if receipt["chat_template"]["sha256"] != plan.chat_template_sha256:
        raise ValueError("tokenizer chat-template SHA-256 mismatch")
    return receipt


def materialize_grounded_reconstruction(
    *,
    plan: ReconstructionPlan,
    output_directory: str | Path,
    tokenizer: ChatTemplateTokenizer,
) -> dict[str, object]:
    """Materialize split-safe, no-truncation Qwen length tiers atomically."""

    if not isinstance(plan, ReconstructionPlan):
        raise TypeError("plan must be a ReconstructionPlan")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    tokenizer_receipt = _tokenizer_receipt(plan)
    reference_manifest, shards, source_descriptor = _load_reference_manifest(plan)
    source_index, source_receipt = _load_source_provenance(source_descriptor)
    dispositions, consumed_source_ids, shard_receipts = _scan_shards(
        shards, source_index, tokenizer
    )
    status_counts = Counter(item.status for item in dispositions)
    accounting = _complete_status_counts(status_counts)
    if accounting["deferred"]:
        raise ValueError(
            f"{accounting['deferred']} rows exceed 32,768 rendered tokens; "
            "no deferred rows are permitted"
        )
    if len(dispositions) != sum(accounting.values()):
        raise AssertionError("input disposition accounting does not balance")
    missing_source_ids = sorted(set(source_index).difference(consumed_source_ids))
    if missing_source_ids:
        raise ValueError(
            f"{len(missing_source_ids)} emitted source-provenance IDs have no reference row"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        sums: dict[str, str] = {}
        output_receipts: list[dict[str, object]] = []
        emitted = [item for item in dispositions if item.status == "emitted"]
        grouped: dict[tuple[str, str, str], list[_Disposition]] = defaultdict(list)
        for item in emitted:
            if item.tier is None:
                raise AssertionError("emitted row has no rendered-length tier")
            grouped[(item.shard.leaf_id, item.shard.split, item.tier)].append(item)
        for leaf in LEAVES:
            for split in SPLITS:
                for tier in OUTPUT_TIERS:
                    members = sorted(
                        grouped.get((leaf, split, tier), ()),
                        key=lambda item: (item.document_id, item.input_line),
                    )
                    message_relative = Path("messages") / leaf / split / f"{tier}.jsonl"
                    provenance_relative = Path("provenance") / leaf / split / f"{tier}.jsonl"
                    (stage / message_relative).parent.mkdir(parents=True, exist_ok=True)
                    (stage / provenance_relative).parent.mkdir(parents=True, exist_ok=True)
                    message_receipt = _write_rows(
                        stage / message_relative,
                        (
                            item.messages_payload
                            for item in members
                            if item.messages_payload is not None
                        ),
                    )
                    provenance_receipt = _write_rows(
                        stage / provenance_relative,
                        (_provenance_row(item) for item in members),
                    )
                    if message_receipt["records"] != len(members) or provenance_receipt[
                        "records"
                    ] != len(members):
                        raise AssertionError("messages and provenance emission counts diverged")
                    rendered = [item.rendered_tokens for item in members]
                    output_receipts.append(
                        {
                            "leaf_id": leaf,
                            "leaf_revision": "1",
                            "split": split,
                            "tier": tier,
                            "messages_path": message_relative.as_posix(),
                            "messages_sha256": message_receipt["sha256"],
                            "provenance_path": provenance_relative.as_posix(),
                            "provenance_sha256": provenance_receipt["sha256"],
                            "bytes": message_receipt["bytes"],
                            "records": len(members),
                            "minimum_rendered_tokens": min(rendered) if rendered else None,
                            "maximum_rendered_tokens": max(rendered) if rendered else None,
                        }
                    )
                    sums[message_relative.as_posix()] = str(message_receipt["sha256"])
                    sums[provenance_relative.as_posix()] = str(provenance_receipt["sha256"])

        accounting_relative = Path("accounting.jsonl")
        accounting_receipt = _write_rows(
            stage / accounting_relative,
            (
                _accounting_row(item)
                for item in sorted(
                    dispositions,
                    key=lambda item: (item.shard_index, item.input_line),
                )
            ),
        )
        sums[accounting_relative.as_posix()] = str(accounting_receipt["sha256"])
        by_leaf_split: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        for leaf in LEAVES:
            for split in SPLITS:
                counts = Counter(
                    item.status
                    for item in dispositions
                    if item.shard.leaf_id == leaf and item.shard.split == split
                )
                by_leaf_split[leaf][split] = _complete_status_counts(counts)
        by_tier = Counter(item.tier for item in emitted)
        reasons = Counter(item.reason for item in dispositions)
        manifest_content: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "task_contract": {
                "task_type": TASK_TYPE,
                "description": "deterministic title/topic-grounded source-passage reconstruction",
                "assistant_target": "verbatim canonical source passage",
                "assistant_target_is_verbatim_source_passage": True,
                "contains_synthetic_question_answering": False,
                "proves_answer_quality": False,
                "system_prompt": SYSTEM_PROMPT,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "topic_labels": TOPIC_LABELS,
            },
            "split_and_dedup_policy": {
                "preserve_canonical_leaf_and_split": True,
                "precedence": list(SPLIT_PRECEDENCE),
                "document_id_guard": True,
                "exact_title_text_guard": True,
                "source_family_guard": True,
                "test_is_evaluation_only": True,
            },
            "rendering_policy": {
                "model_family": "Qwen3-0.6B",
                "template_mode": "direct",
                "truncation": False,
                "maximum_rendered_tokens": 32768,
                "deferred_rows_permitted": False,
                "tiers": list(OUTPUT_TIERS),
            },
            "inputs": {
                "reference_manifest": {
                    "path": str(plan.reference_manifest),
                    "sha256": plan.reference_manifest_sha256,
                    "content_sha256": reference_manifest["content_sha256"],
                },
                "source_document_provenance": source_receipt,
                "tokenizer": tokenizer_receipt,
                "builder": {
                    "path": Path(__file__).name,
                    "sha256": _sha256_file(Path(__file__).resolve()),
                },
                "shards": shard_receipts,
            },
            "accounting": {
                "input_records": len(dispositions),
                "terminal_statuses": accounting,
                "reasons": dict(sorted(reasons.items())),
                "by_leaf_and_split": dict(by_leaf_split),
                "emitted_by_tier": {tier: by_tier.get(tier, 0) for tier in OUTPUT_TIERS},
                "source_provenance_emitted_records": len(source_index),
                "source_provenance_ids_consumed": len(consumed_source_ids),
                "source_provenance_ids_missing": len(missing_source_ids),
                "deferred_rows": accounting["deferred"],
                "truncated_rows": 0,
                "zero_silent_omissions": len(dispositions) == sum(accounting.values()),
                "fully_reconciled": len(dispositions) == len(source_index),
            },
            "outputs": output_receipts,
            "accounting_artifact": {
                "path": accounting_relative.as_posix(),
                **accounting_receipt,
                "content_free": True,
            },
        }
        manifest = {**manifest_content, "content_sha256": fingerprint_json(manifest_content)}
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        (stage / "manifest.json").write_bytes(manifest_bytes)
        sums["manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
        sums_bytes = "".join(f"{digest}  {path}\n" for path, digest in sorted(sums.items())).encode(
            "utf-8"
        )
        (stage / "sha256sums.txt").write_bytes(sums_bytes)
        os.replace(stage, output)
        stage = None
        return manifest
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


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
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-aggregate-sha256", required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = ReconstructionPlan(
        reference_manifest=args.reference_manifest,
        reference_manifest_sha256=args.reference_manifest_sha256,
        tokenizer_path=args.tokenizer,
        tokenizer_aggregate_sha256=args.tokenizer_aggregate_sha256,
        chat_template_sha256=args.chat_template_sha256,
    )
    tokenizer = _load_local_tokenizer(plan.tokenizer_path)
    manifest = materialize_grounded_reconstruction(
        plan=plan,
        output_directory=args.output,
        tokenizer=tokenizer,
    )
    print(json.dumps(manifest["accounting"], sort_keys=True))
    return 0


__all__ = [
    "ACCOUNTING_SCHEMA_VERSION",
    "OUTPUT_TIERS",
    "PROVENANCE_SCHEMA_VERSION",
    "ReconstructionPlan",
    "SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "TASK_TYPE",
    "TOPIC_LABELS",
    "main",
    "materialize_grounded_reconstruction",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
