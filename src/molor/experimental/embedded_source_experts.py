"""Materialize grounded embedded expert sources with group-first evaluation splits.

This builder has an intentionally narrow input surface.  It accepts only the pinned prepared
``code-corpus`` and ``ocb-can-signal-db`` shards, the aggregate prepared manifest, the exact OCB
``training_corpus`` SQLite view, a local tokenizer, and explicitly allow-listed aggregate reports
for prior derived datasets.  It never opens payloads referenced by those prior reports.

File text and OCB table rows remain reference/continued-pretraining documents.  The only SFT
records are existing OCB QA messages, preserved verbatim.  Repository groups (code) and vehicle
families (OCB) are assigned before rows are emitted, replacing the unsafe prepared row-wise split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from molor.experimental.length_tier_shards import (
    ChatTemplateTokenizer,
    TIER_SPECS,
    render_token_count,
    sha256_file,
    tier_for_token_count,
    tokenizer_identity,
)


SCHEMA_VERSION = "molor.embedded-source-experts.v1"
PLAN_SCHEMA_VERSION = "molor.embedded-source-plan.v1"
LEAVES = (
    "embedded-can-diagnostics",
    "embedded-code-generation",
    "embedded-knowledge",
    "frontend-visual-computing",
    "terminal-cli",
)
SPLITS = ("train", "validation", "test")
SOURCE_SPLITS = ("train", "validation")
OCB_QA_FAMILIES = ("definition_decode", "encode_physical_100", "location")
EVALUATION_FRACTION = 0.05
MAX_EVAL_GROUP_SHARE = 0.50
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DENIED_PATH_WORDS = ("cyber", "security")
_OCB_FIELDS = (
    "signal_id",
    "signal_name",
    "start_bit",
    "bit_length",
    "scale",
    "offset",
    "unit",
    "byte_order",
    "is_signed",
    "training_eligible",
    "can_id",
    "message_name",
    "dlc",
    "make",
    "model",
    "year",
    "platform",
    "vehicle_tier",
    "data_trust",
)
_PRIOR_REPORT_RELATIONSHIPS = {
    "terminal-cli.build-report.json": "complementary_prior_sft_for_terminal_cli",
    "embedded-knowledge-v1.report.json": "complementary_prior_grounded_sft",
    "local-embedded-code.report.json": "complementary_prior_code_completion_sft",
    "trusted-can.report.json": "complementary_prior_high_trust_can_sft",
    "grounded-can.report.json": "complementary_prior_grounded_can_sft",
    "grounded-can-curriculum.report.json": "complementary_prior_can_curriculum",
    "ocb-can-verified.report.json": "complementary_prior_verified_can_sft",
    "verified-embedded-code-v1.manifest.json": "complementary_prior_verified_code_sft",
    "verified-embedded-code-v2.manifest.json": "complementary_prior_verified_code_sft",
    "verified-embedded-code-v3.manifest.json": "complementary_prior_verified_code_sft",
}
_SHELL_EXTENSIONS = frozenset({".sh", ".bash", ".bat", ".cmd", ".ps1", ".zsh", ".fish"})
_FRONTEND_EXTENSIONS = frozenset(
    {".qml", ".ui", ".qrc", ".css", ".html", ".htm", ".js", ".ts", ".tsx", ".jsx"}
)
_KNOWLEDGE_EXTENSIONS = frozenset(
    {
        ".md",
        ".txt",
        ".rst",
        ".rtf",
        ".csv",
        ".json",
        ".xml",
        ".yml",
        ".yaml",
        ".ini",
        ".cfg",
        ".conf",
        ".url",
        ".readme",
        ".properties",
    }
)
_CAN_REPOSITORIES = frozenset(
    {"ocb", "obd-can-mod", "aevit2-can-fd-interceptor", "aevit3_can_tool"}
)
_CAN_PATH_RE = re.compile(r"(^|[/_.-])(can|obd|dbc|uds|diagnostic)([/_.-]|$)")


@dataclass(frozen=True, slots=True)
class FileDescriptor:
    path: Path
    sha256: str
    bytes: int
    records: int | None = None


@dataclass(frozen=True, slots=True)
class PriorReportDescriptor:
    name: str
    path: Path
    sha256: str
    bytes: int
    relationship: str


@dataclass(frozen=True, slots=True)
class BuildPlan:
    path: Path
    sha256: str
    prepared_manifest: FileDescriptor
    code: Mapping[str, FileDescriptor]
    ocb: Mapping[str, FileDescriptor]
    ocb_database: Path
    ocb_view_rows: int
    ocb_view_sha256: str
    tokenizer_path: Path
    tokenizer_sha256: str
    chat_template_sha256: str
    prior_reports: tuple[PriorReportDescriptor, ...]


@dataclass(frozen=True, slots=True)
class CodeRecord:
    upstream_id: str
    repo: str
    file_path: str
    text: str
    source_split: str
    source_line: int
    record_sha256: str


@dataclass(frozen=True, slots=True)
class OCBTableRecord:
    upstream_id: str
    row: Mapping[str, Any]
    source_split: str
    source_line: int
    record_sha256: str


@dataclass(frozen=True, slots=True)
class OCBQARecord:
    upstream_id: str
    messages: tuple[Mapping[str, str], ...]
    source_split: str
    source_line: int
    record_sha256: str


@dataclass(slots=True)
class QAMatch:
    family: str | None = None
    signal_ids: set[int] | None = None
    vehicle_groups: set[str] | None = None

    def __post_init__(self) -> None:
        if self.signal_ids is None:
            self.signal_ids = set()
        if self.vehicle_groups is None:
            self.vehicle_groups = set()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_document(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _checked_sha256(value: object, *, field: str) -> str:
    digest = str(value).casefold()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return digest


def _denied_path(path: Path) -> bool:
    return any(word in part.casefold() for part in path.parts for word in _DENIED_PATH_WORDS)


def _safe_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _denied_path(resolved):
        raise ValueError(f"{label} contains a denied path component")
    return resolved


def _strict_json_object(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve_plan_path(raw: object, *, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return _safe_path(path, label=label)


def _file_descriptor(value: object, *, base: Path, label: str, records: bool) -> FileDescriptor:
    expected = {"path", "sha256", "bytes"}
    if records:
        expected.add("records")
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} descriptor must contain exactly {sorted(expected)!r}")
    byte_count = value["bytes"]
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError(f"{label} bytes must be a non-negative integer")
    record_count = value.get("records")
    if records and (type(record_count) is not int or record_count < 0):
        raise ValueError(f"{label} records must be a non-negative integer")
    return FileDescriptor(
        path=_resolve_plan_path(value["path"], base=base, label=label),
        sha256=_checked_sha256(value["sha256"], field=f"{label} sha256"),
        bytes=byte_count,
        records=record_count,
    )


def load_plan(path: str | Path) -> BuildPlan:
    """Load the strict, cryptographically pinned embedded-source plan."""

    plan_path = _safe_path(Path(path), label="plan")
    document = _strict_json_object(plan_path)
    expected = {
        "schema_version",
        "prepared_manifest",
        "code_corpus",
        "ocb_corpus",
        "ocb_database",
        "tokenizer",
        "prior_reports",
        "split_policy",
    }
    if set(document) != expected or document.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"plan must match {PLAN_SCHEMA_VERSION} with exact top-level fields")
    base = plan_path.parent
    split_policy = document["split_policy"]
    if split_policy != {
        "evaluation_fraction": EVALUATION_FRACTION,
        "max_eval_group_share": MAX_EVAL_GROUP_SHARE,
        "seed": "molor.embedded-source.v1",
    }:
        raise ValueError("plan split_policy does not match the v1 group-first contract")

    def split_descriptors(value: object, *, label: str) -> dict[str, FileDescriptor]:
        if not isinstance(value, dict) or set(value) != set(SOURCE_SPLITS):
            raise ValueError(f"{label} must contain exactly train and validation")
        return {
            split: _file_descriptor(value[split], base=base, label=f"{label}.{split}", records=True)
            for split in SOURCE_SPLITS
        }

    database = document["ocb_database"]
    if not isinstance(database, dict) or set(database) != {
        "path",
        "view",
        "rows",
        "canonical_sha256",
    }:
        raise ValueError("ocb_database descriptor is malformed")
    if database["view"] != "training_corpus":
        raise ValueError("only the OCB training_corpus view is allowed")
    if type(database["rows"]) is not int or database["rows"] < 0:
        raise ValueError("ocb_database rows must be a non-negative integer")

    tokenizer = document["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {
        "path",
        "aggregate_sha256",
        "chat_template_sha256",
    }:
        raise ValueError("tokenizer descriptor is malformed")

    raw_reports = document["prior_reports"]
    if not isinstance(raw_reports, list) or len(raw_reports) != len(_PRIOR_REPORT_RELATIONSHIPS):
        raise ValueError("plan must pin every allow-listed prior aggregate report exactly once")
    reports: list[PriorReportDescriptor] = []
    for index, raw in enumerate(raw_reports):
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "path",
            "sha256",
            "bytes",
            "relationship",
        }:
            raise ValueError(f"prior report descriptor {index} is malformed")
        name = raw["name"]
        if not isinstance(name, str) or name not in _PRIOR_REPORT_RELATIONSHIPS:
            raise ValueError(f"prior report {index} is not allow-listed")
        if raw["relationship"] != _PRIOR_REPORT_RELATIONSHIPS[name]:
            raise ValueError(f"prior report relationship mismatch for {name}")
        descriptor = _file_descriptor(
            {key: raw[key] for key in ("path", "sha256", "bytes")},
            base=base,
            label=f"prior_reports[{index}]",
            records=False,
        )
        if descriptor.path.name != name:
            raise ValueError(f"prior report basename mismatch for {name}")
        reports.append(
            PriorReportDescriptor(
                name=name,
                path=descriptor.path,
                sha256=descriptor.sha256,
                bytes=descriptor.bytes,
                relationship=raw["relationship"],
            )
        )
    if {report.name for report in reports} != set(_PRIOR_REPORT_RELATIONSHIPS):
        raise ValueError("prior report names must be unique and complete")

    prepared_descriptor = _file_descriptor(
        document["prepared_manifest"], base=base, label="prepared_manifest", records=False
    )
    if prepared_descriptor.path.name != "manifest.json":
        raise ValueError("prepared manifest must be the exact prepared/manifest.json artifact")
    code_descriptors = split_descriptors(document["code_corpus"], label="code_corpus")
    ocb_descriptors = split_descriptors(document["ocb_corpus"], label="ocb_corpus")
    prepared_root = prepared_descriptor.path.parent
    for split in SOURCE_SPLITS:
        if code_descriptors[split].path != prepared_root / "code-corpus" / f"{split}.jsonl":
            raise ValueError(f"code_corpus.{split} is outside the exact allowed prepared source")
        if ocb_descriptors[split].path != prepared_root / "ocb-can-signal-db" / f"{split}.jsonl":
            raise ValueError(f"ocb_corpus.{split} is outside the exact allowed prepared source")
    database_path = _resolve_plan_path(database["path"], base=base, label="ocb_database")
    if database_path != prepared_root.parent / "ocb_ocb.db":
        raise ValueError("OCB database must be the exact sibling ocb_ocb.db artifact")

    return BuildPlan(
        path=plan_path,
        sha256=sha256_file(plan_path),
        prepared_manifest=prepared_descriptor,
        code=code_descriptors,
        ocb=ocb_descriptors,
        ocb_database=database_path,
        ocb_view_rows=database["rows"],
        ocb_view_sha256=_checked_sha256(
            database["canonical_sha256"], field="ocb_database canonical_sha256"
        ),
        tokenizer_path=_resolve_plan_path(tokenizer["path"], base=base, label="tokenizer"),
        tokenizer_sha256=_checked_sha256(
            tokenizer["aggregate_sha256"], field="tokenizer aggregate_sha256"
        ),
        chat_template_sha256=_checked_sha256(
            tokenizer["chat_template_sha256"], field="chat template sha256"
        ),
        prior_reports=tuple(reports),
    )


def _verify_file(descriptor: FileDescriptor, *, label: str) -> dict[str, Any]:
    if not descriptor.path.is_file():
        raise FileNotFoundError(f"pinned {label} does not exist: {descriptor.path}")
    actual_bytes = descriptor.path.stat().st_size
    if actual_bytes != descriptor.bytes:
        raise RuntimeError(
            f"{label} byte-count drift: expected {descriptor.bytes}, got {actual_bytes}"
        )
    actual_sha = sha256_file(descriptor.path)
    if actual_sha != descriptor.sha256:
        raise RuntimeError(f"{label} SHA-256 drift: expected {descriptor.sha256}, got {actual_sha}")
    return {
        "path": str(descriptor.path),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        **({"records": descriptor.records} if descriptor.records is not None else {}),
        "verified": True,
    }


def _prior_identity(document: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("dataset", "schema_version", "policy_version", "purpose"):
        value = document.get(key)
        if isinstance(value, str):
            result[key] = value
    return result


def _reported_outputs(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only declared row/hash facts; never follow or open reported payload paths."""

    receipts: list[dict[str, Any]] = []

    def visit(value: object, logical_path: str) -> None:
        if isinstance(value, dict):
            count_key = next(
                (key for key in ("rows", "examples") if type(value.get(key)) is int), None
            )
            hash_key = next(
                (
                    key
                    for key in ("sha256", "data_sha256", "dataset_sha256")
                    if isinstance(value.get(key), str)
                    and _SHA256_RE.fullmatch(str(value[key]).casefold())
                ),
                None,
            )
            if count_key is not None and hash_key is not None:
                receipts.append(
                    {
                        "logical_path": logical_path or "/",
                        "records": value[count_key],
                        "reported_sha256": str(value[hash_key]).casefold(),
                    }
                )
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    visit(child, f"{logical_path}/{key}")
        elif isinstance(value, list) and len(value) <= 64:
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    visit(child, f"{logical_path}/{index}")

    visit(document, "")
    unique = {
        (item["logical_path"], item["records"], item["reported_sha256"]): item for item in receipts
    }
    return [unique[key] for key in sorted(unique)]


def _verify_plan_inputs(plan: BuildPlan) -> dict[str, Any]:
    manifest_receipt = _verify_file(plan.prepared_manifest, label="prepared manifest")
    prepared_manifest = _strict_json_object(plan.prepared_manifest.path)
    datasets = prepared_manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("prepared manifest datasets section is malformed")
    code_manifest = datasets.get("code-corpus")
    ocb_manifest = datasets.get("ocb-can-signal-db")
    if not isinstance(code_manifest, dict) or not isinstance(ocb_manifest, dict):
        raise ValueError("prepared manifest lacks the two required safe datasets")
    if code_manifest.get("train_rows") != plan.code["train"].records:
        raise ValueError("code train count disagrees with prepared manifest")
    if code_manifest.get("validation_rows") != plan.code["validation"].records:
        raise ValueError("code validation count disagrees with prepared manifest")
    expected_ocb_total = sum(descriptor.records or 0 for descriptor in plan.ocb.values())
    if ocb_manifest.get("train_rows") != plan.ocb["train"].records:
        raise ValueError("OCB train count disagrees with prepared manifest")
    if ocb_manifest.get("validation_rows") != plan.ocb["validation"].records:
        raise ValueError("OCB validation count disagrees with prepared manifest")
    if ocb_manifest.get("table_rows", 0) + ocb_manifest.get("qa_rows", 0) != expected_ocb_total:
        raise ValueError("OCB kind totals disagree with pinned physical rows")

    code_receipts = {
        split: _verify_file(plan.code[split], label=f"code_corpus.{split}")
        for split in SOURCE_SPLITS
    }
    ocb_receipts = {
        split: _verify_file(plan.ocb[split], label=f"ocb_corpus.{split}") for split in SOURCE_SPLITS
    }
    tokenizer_receipt = tokenizer_identity(plan.tokenizer_path)
    if tokenizer_receipt["aggregate_sha256"] != plan.tokenizer_sha256:
        raise RuntimeError("tokenizer aggregate SHA-256 drift")
    if tokenizer_receipt["chat_template"]["sha256"] != plan.chat_template_sha256:
        raise RuntimeError("tokenizer chat-template SHA-256 drift")

    prior_receipts: list[dict[str, Any]] = []
    for report in plan.prior_reports:
        descriptor = FileDescriptor(report.path, report.sha256, report.bytes)
        receipt = _verify_file(descriptor, label=f"prior report {report.name}")
        document = _strict_json_object(report.path)
        prior_receipts.append(
            {
                **receipt,
                "name": report.name,
                "disposition": "prior_derived_supplement",
                "relationship": report.relationship,
                "identity": _prior_identity(document),
                "reported_outputs": _reported_outputs(document),
                "payload_files_opened": False,
                "superseded_by_this_build": False,
            }
        )
    return {
        "prepared_manifest": manifest_receipt,
        "code_corpus": code_receipts,
        "ocb_corpus": ocb_receipts,
        "tokenizer": tokenizer_receipt,
        "prior_derived_reports": prior_receipts,
    }


def _raw_line_sha256(raw: bytes) -> str:
    return _sha256_bytes(raw.rstrip(b"\r\n"))


def _read_json_lines(
    descriptor: FileDescriptor,
) -> Iterator[tuple[int, bytes, object | None, str | None]]:
    if descriptor.records is None:
        raise ValueError("JSONL descriptor lacks a physical record count")
    streamed = hashlib.sha256()
    count = 0
    with descriptor.path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            streamed.update(raw)
            count += 1
            try:
                value = json.loads(raw.decode("utf-8"))
                error = None
            except UnicodeDecodeError:
                value = None
                error = "invalid_utf8"
            except json.JSONDecodeError:
                value = None
                error = "invalid_json"
            yield line_number, raw, value, error
    if count != descriptor.records:
        raise RuntimeError(
            f"record-count drift while reading {descriptor.path}: "
            f"expected {descriptor.records}, got {count}"
        )
    if streamed.hexdigest() != descriptor.sha256:
        raise RuntimeError(f"input changed while streaming: {descriptor.path}")


def _valid_messages(value: object) -> tuple[Mapping[str, str], ...] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    messages: list[Mapping[str, str]] = []
    for message in value:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            return None
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        messages.append({"role": role, "content": content})
    if messages[-2]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None
    return tuple(messages)


def _load_code_records(
    descriptors: Mapping[str, FileDescriptor],
    non_emitted: list[dict[str, Any]],
) -> list[CodeRecord]:
    records: list[CodeRecord] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for source_split in SOURCE_SPLITS:
        for line_number, raw, value, parse_error in _read_json_lines(descriptors[source_split]):
            fingerprint = _raw_line_sha256(raw)
            reason = parse_error
            if reason is None and (
                not isinstance(value, dict)
                or set(value) != {"id", "source", "repo", "file_path", "text"}
            ):
                reason = "invalid_code_record_schema"
            if reason is None and isinstance(value, dict):
                if value.get("source") != "code-corpus":
                    reason = "unexpected_code_source"
                elif not all(
                    isinstance(value.get(key), str) and value[key].strip()
                    for key in ("id", "repo", "file_path", "text")
                ):
                    reason = "empty_code_identity_or_text"
            if reason is not None or not isinstance(value, dict):
                non_emitted.append(
                    {
                        "disposition": "quarantined",
                        "reason": reason or "invalid_code_record",
                        "record_sha256": fingerprint,
                        "source": "code-corpus",
                        "source_line": line_number,
                        "source_split": source_split,
                    }
                )
                continue
            upstream_id = value["id"]
            text_hash = _sha256_bytes(value["text"].encode("utf-8"))
            if upstream_id in seen_ids:
                non_emitted.append(
                    {
                        "disposition": "duplicate",
                        "reason": "duplicate_upstream_id",
                        "record_sha256": fingerprint,
                        "source": "code-corpus",
                        "source_line": line_number,
                        "source_split": source_split,
                        "upstream_id": upstream_id,
                    }
                )
                continue
            seen_ids.add(upstream_id)
            if text_hash in seen_text:
                non_emitted.append(
                    {
                        "disposition": "duplicate",
                        "reason": "duplicate_exact_text",
                        "record_sha256": fingerprint,
                        "source": "code-corpus",
                        "source_line": line_number,
                        "source_split": source_split,
                        "upstream_id": upstream_id,
                    }
                )
                continue
            seen_text.add(text_hash)
            records.append(
                CodeRecord(
                    upstream_id=upstream_id,
                    repo=value["repo"],
                    file_path=value["file_path"],
                    text=value["text"],
                    source_split=source_split,
                    source_line=line_number,
                    record_sha256=fingerprint,
                )
            )
    return records


def _load_ocb_records(
    descriptors: Mapping[str, FileDescriptor],
    non_emitted: list[dict[str, Any]],
) -> tuple[list[OCBTableRecord], list[OCBQARecord]]:
    tables: list[OCBTableRecord] = []
    qas: list[OCBQARecord] = []
    seen_ids: set[str] = set()
    seen_signal_ids: set[int] = set()
    seen_messages: set[str] = set()
    for source_split in SOURCE_SPLITS:
        for line_number, raw, value, parse_error in _read_json_lines(descriptors[source_split]):
            fingerprint = _raw_line_sha256(raw)
            reason = parse_error
            if reason is None and (
                not isinstance(value, dict)
                or value.get("source") != "ocb-can-signal-db"
                or value.get("kind") not in {"table_row", "qa"}
                or not isinstance(value.get("id"), str)
                or not value["id"].strip()
            ):
                reason = "invalid_ocb_record_envelope"
            if reason is not None or not isinstance(value, dict):
                non_emitted.append(
                    {
                        "disposition": "quarantined",
                        "reason": reason or "invalid_ocb_record",
                        "record_sha256": fingerprint,
                        "source": "ocb-can-signal-db",
                        "source_line": line_number,
                        "source_split": source_split,
                    }
                )
                continue
            upstream_id = value["id"]
            if upstream_id in seen_ids:
                non_emitted.append(
                    {
                        "disposition": "duplicate",
                        "reason": "duplicate_upstream_id",
                        "record_sha256": fingerprint,
                        "source": "ocb-can-signal-db",
                        "source_line": line_number,
                        "source_split": source_split,
                        "upstream_id": upstream_id,
                    }
                )
                continue
            seen_ids.add(upstream_id)
            if value["kind"] == "table_row":
                row = value.get("row")
                if not isinstance(row, dict) or set(row) != set(_OCB_FIELDS):
                    non_emitted.append(
                        {
                            "disposition": "quarantined",
                            "reason": "invalid_ocb_table_schema",
                            "record_sha256": fingerprint,
                            "source": "ocb-can-signal-db",
                            "source_line": line_number,
                            "source_split": source_split,
                            "upstream_id": upstream_id,
                        }
                    )
                    continue
                signal_id = row.get("signal_id")
                if type(signal_id) is not int or signal_id in seen_signal_ids:
                    non_emitted.append(
                        {
                            "disposition": "duplicate"
                            if signal_id in seen_signal_ids
                            else "rejected",
                            "reason": "duplicate_signal_id"
                            if signal_id in seen_signal_ids
                            else "invalid_signal_id",
                            "record_sha256": fingerprint,
                            "source": "ocb-can-signal-db",
                            "source_line": line_number,
                            "source_split": source_split,
                            "upstream_id": upstream_id,
                        }
                    )
                    continue
                seen_signal_ids.add(signal_id)
                tables.append(
                    OCBTableRecord(
                        upstream_id=upstream_id,
                        row=row,
                        source_split=source_split,
                        source_line=line_number,
                        record_sha256=fingerprint,
                    )
                )
            else:
                if set(value) != {"id", "source", "kind", "messages"}:
                    messages = None
                else:
                    messages = _valid_messages(value.get("messages"))
                if messages is None:
                    non_emitted.append(
                        {
                            "disposition": "quarantined",
                            "reason": "invalid_ocb_qa_messages",
                            "record_sha256": fingerprint,
                            "source": "ocb-can-signal-db",
                            "source_line": line_number,
                            "source_split": source_split,
                            "upstream_id": upstream_id,
                        }
                    )
                    continue
                message_hash = _sha256_bytes(_compact(list(messages)).encode("utf-8"))
                if message_hash in seen_messages:
                    non_emitted.append(
                        {
                            "disposition": "duplicate",
                            "reason": "duplicate_exact_messages",
                            "record_sha256": fingerprint,
                            "source": "ocb-can-signal-db",
                            "source_line": line_number,
                            "source_split": source_split,
                            "upstream_id": upstream_id,
                        }
                    )
                    continue
                seen_messages.add(message_hash)
                qas.append(
                    OCBQARecord(
                        upstream_id=upstream_id,
                        messages=messages,
                        source_split=source_split,
                        source_line=line_number,
                        record_sha256=fingerprint,
                    )
                )
    return tables, qas


def _verify_ocb_training_view(
    *, database: Path, tables: Sequence[OCBTableRecord], expected_rows: int, expected_sha256: str
) -> dict[str, Any]:
    if database.name != "ocb_ocb.db" or not database.is_file():
        raise FileNotFoundError("the exact pinned OCB database file is unavailable")
    prepared_by_id = {int(record.row["signal_id"]): dict(record.row) for record in tables}
    if len(prepared_by_id) != len(tables):
        raise ValueError("prepared OCB table signal IDs are not unique")
    digest = hashlib.sha256()
    rows = 0
    mismatches: list[int] = []
    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = ",".join('"offset"' if field == "offset" else field for field in _OCB_FIELDS)
        query = f"SELECT {columns} FROM training_corpus ORDER BY signal_id"
        for value in connection.execute(query):
            row = {field: value[field] for field in _OCB_FIELDS}
            rows += 1
            digest.update(_compact(row).encode("utf-8"))
            digest.update(b"\n")
            signal_id = int(row["signal_id"])
            if prepared_by_id.get(signal_id) != row and len(mismatches) < 16:
                mismatches.append(signal_id)
    finally:
        connection.close()
    actual_sha256 = digest.hexdigest()
    if rows != expected_rows or actual_sha256 != expected_sha256:
        raise RuntimeError(
            "OCB training_corpus view identity drift: "
            f"expected ({expected_rows}, {expected_sha256}), got ({rows}, {actual_sha256})"
        )
    if rows != len(tables) or mismatches:
        raise RuntimeError(
            f"prepared OCB table rows disagree with the training_corpus view: {mismatches!r}"
        )
    return {
        "database_path": str(database),
        "database_file_hashed": False,
        "query_scope": "training_corpus view fields only",
        "rows": rows,
        "canonical_sha256": actual_sha256,
        "prepared_row_mismatches": 0,
        "verified": True,
    }


def _vehicle_group(row: Mapping[str, Any]) -> str:
    values = [row.get("make"), row.get("model"), row.get("year")]
    return "|".join("" if value is None else str(value) for value in values)


def _opaque_group(prefix: str, value: str) -> str:
    return f"{prefix}:{_sha256_bytes(value.encode('utf-8'))[:24]}"


def _qa_templates(row: Mapping[str, Any]) -> tuple[tuple[str, tuple[Mapping[str, str], ...]], ...]:
    signal = row["signal_name"]
    message = row["message_name"]
    can_id = row["can_id"]
    dlc = row["dlc"]
    start_bit = row["start_bit"]
    bit_length = row["bit_length"]
    scale = row["scale"]
    offset = row["offset"]
    byte_order = row["byte_order"]
    signed = "signed" if row["is_signed"] else "unsigned"
    unit = row["unit"] or "dimensionless (no unit)"
    year = row["year"] or "n/a"
    vehicle = f"{year} {row['make']} {row['model']}"
    platform = row["platform"] or "standard"
    tier = row["vehicle_tier"] or "unknown"
    trust = row["data_trust"]

    location = (
        {
            "role": "user",
            "content": (
                f"Which CAN message carries the signal '{signal}' on a {vehicle}, and where "
                "does the signal live in the frame?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"'{signal}' is carried in '{message}' (ID {can_id}, DLC {dlc}, tier {tier}). "
                f"It starts at bit {start_bit}, spans {bit_length} bits, {byte_order} byte order, "
                f"converting with physical = raw x {scale} + {offset} in {unit}. "
                f"Source trust: {trust}."
            ),
        },
    )
    decode = (
        {
            "role": "user",
            "content": (
                f"On a {vehicle} ({platform}), the CAN message '{message}' "
                f"(ID {can_id}, DLC {dlc}) carries the signal '{signal}'. How is this signal "
                "encoded and how do you decode it to physical units?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"Signal '{signal}' in message '{message}' (ID {can_id}, DLC {dlc}) starts at "
                f"bit {start_bit} and spans {bit_length} bits, {byte_order} byte order, {signed}. "
                f"Extract the raw {bit_length}-bit field at bit {start_bit} ({byte_order} "
                f"ordering), then apply physical = raw x {scale} + {offset}, giving the value "
                f"in {unit}."
            ),
        },
    )
    encode = (
        {
            "role": "user",
            "content": (
                f"You need to transmit the signal '{signal}' (unit: {unit}) in CAN message "
                f"'{message}' (ID {can_id}) on a {vehicle}. How do you encode a physical value "
                f"of 100 {unit} into the raw frame bytes?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"The field lives at bit {start_bit}, {bit_length} bits wide, {byte_order} byte "
                f"order, {signed}. Invert the conversion raw = (physical - {offset}) / {scale}. "
                f"For physical 100 {unit}: raw = round((100 - {offset}) / {scale}), then insert "
                f"that {bit_length}-bit value at bit {start_bit} of the {dlc}-byte payload using "
                f"{byte_order} ordering."
            ),
        },
    )
    return (("location", location), ("definition_decode", decode), ("encode_physical_100", encode))


def _message_hash(messages: Iterable[Mapping[str, str]]) -> str:
    return _sha256_bytes(_compact(list(messages)).encode("utf-8"))


def _match_existing_qas(
    tables: Sequence[OCBTableRecord], qas: Sequence[OCBQARecord]
) -> dict[str, QAMatch]:
    matches = {_message_hash(record.messages): QAMatch() for record in qas}
    if len(matches) != len(qas):
        raise ValueError("OCB QA messages must be exact-deduplicated before matching")
    for table in tables:
        vehicle_group = _vehicle_group(table.row)
        signal_id = int(table.row["signal_id"])
        for family, messages in _qa_templates(table.row):
            match = matches.get(_message_hash(messages))
            if match is None:
                continue
            if match.family is not None and match.family != family:
                raise ValueError("one OCB QA matched multiple global template families")
            match.family = family
            assert match.signal_ids is not None
            assert match.vehicle_groups is not None
            match.signal_ids.add(signal_id)
            match.vehicle_groups.add(vehicle_group)
    unmatched = [digest for digest, match in matches.items() if not match.signal_ids]
    ambiguous = [
        digest for digest, match in matches.items() if len(match.vehicle_groups or ()) != 1
    ]
    if unmatched or ambiguous:
        raise ValueError(
            f"OCB QA provenance is incomplete: unmatched={len(unmatched)}, "
            f"ambiguous_vehicle_group={len(ambiguous)}"
        )
    return matches


def classify_code_leaf(repo: str, file_path: str) -> str:
    """Classify one grounded file using repository/path metadata only."""

    normalized = file_path.replace("\\", "/").casefold()
    name = PurePosixPath(normalized).name
    extension = PurePosixPath(normalized).suffix
    surrounded = f"/{normalized.strip('/')} /".replace(" /", "/")
    if extension in _SHELL_EXTENSIONS or name == "makefile":
        return "terminal-cli"
    if (
        extension in _FRONTEND_EXTENSIONS
        or repo == "display_qt_app"
        or any(segment in surrounded for segment in ("/qml/", "/gui/", "/ui/", "/frontend/"))
    ):
        return "frontend-visual-computing"
    if repo in _CAN_REPOSITORIES or _CAN_PATH_RE.search(normalized):
        return "embedded-can-diagnostics"
    if extension in _KNOWLEDGE_EXTENSIONS or name.startswith("readme"):
        return "embedded-knowledge"
    return "embedded-code-generation"


def assign_group_splits(
    counts: Mapping[str, int], *, namespace: str
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign whole groups using deterministic hash order and bounded eval group size."""

    if not counts or any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError("split group counts must be positive integers")
    total = sum(counts.values())
    target = max(1, math.ceil(total * EVALUATION_FRACTION))
    cap = max(1, math.ceil(target * MAX_EVAL_GROUP_SHARE), min(counts.values()))
    remaining = set(counts)
    assignments: dict[str, str] = {}
    achieved: dict[str, int] = {}
    for split in ("test", "validation"):
        selected = 0
        candidates = sorted(
            (group for group in remaining if counts[group] <= cap),
            key=lambda group: _sha256_bytes(
                f"molor.embedded-source.v1\0{namespace}\0{split}\0{group}".encode("utf-8")
            ),
        )
        for group in candidates:
            assignments[group] = split
            remaining.remove(group)
            selected += counts[group]
            if selected >= target:
                break
        if selected == 0:
            raise ValueError(f"no eligible group could be assigned to {split}")
        achieved[split] = selected
    for group in remaining:
        assignments[group] = "train"
    split_counts = Counter()
    split_groups = Counter()
    for group, split in assignments.items():
        split_counts[split] += counts[group]
        split_groups[split] += 1
    return assignments, {
        "algorithm": "bounded_group_hash_accumulation_v1",
        "namespace": namespace,
        "seed": "molor.embedded-source.v1",
        "input_records": total,
        "evaluation_target_records": target,
        "maximum_single_eval_group_records": cap,
        "records_by_split": {split: split_counts[split] for split in SPLITS},
        "groups_by_split": {split: split_groups[split] for split in SPLITS},
        "target_reached": {
            "validation": achieved["validation"] >= target,
            "test": achieved["test"] >= target,
        },
    }


def assign_stratified_group_splits(
    counts: Mapping[str, int],
    *,
    features: Mapping[str, set[str]],
    required_features: Iterable[str],
    namespace: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign atomic groups while requiring every evaluation split to cover each feature."""

    required = set(required_features)
    if set(features) != set(counts):
        raise ValueError("feature groups must exactly match count groups")
    if not required or any(not values <= required for values in features.values()):
        raise ValueError("group features must be non-empty subsets of required features")
    total = sum(counts.values())
    target = max(1, math.ceil(total * EVALUATION_FRACTION))
    cap = max(1, math.ceil(target * MAX_EVAL_GROUP_SHARE), min(counts.values()))
    remaining = set(counts)
    assignments: dict[str, str] = {}
    achieved: dict[str, int] = {}
    feature_receipt: dict[str, list[str]] = {}
    for split in ("test", "validation"):
        selected = 0
        covered: set[str] = set()
        while covered != required:
            candidates = [group for group in remaining if counts[group] <= cap]
            if not candidates:
                raise ValueError(f"no eligible groups remain to cover {split} features")
            candidates.sort(
                key=lambda group: (
                    -len(features[group] - covered),
                    _sha256_bytes(
                        f"molor.embedded-source.v1\0{namespace}\0{split}\0coverage\0{group}".encode(
                            "utf-8"
                        )
                    ),
                )
            )
            group = candidates[0]
            if not (features[group] - covered):
                missing = sorted(required - covered)
                raise ValueError(f"eligible groups cannot cover {split} features: {missing!r}")
            assignments[group] = split
            remaining.remove(group)
            selected += counts[group]
            covered.update(features[group])
        for group in sorted(
            (item for item in remaining if counts[item] <= cap),
            key=lambda item: _sha256_bytes(
                f"molor.embedded-source.v1\0{namespace}\0{split}\0fill\0{item}".encode("utf-8")
            ),
        ):
            if selected >= target:
                break
            assignments[group] = split
            remaining.remove(group)
            selected += counts[group]
            covered.update(features[group])
        achieved[split] = selected
        feature_receipt[split] = sorted(covered)
    for group in remaining:
        assignments[group] = "train"
    train_features = set().union(*(features[group] for group in remaining))
    if train_features != required:
        raise ValueError(
            f"training split lost required features: {sorted(required - train_features)!r}"
        )
    feature_receipt["train"] = sorted(train_features)
    split_counts = Counter()
    split_groups = Counter()
    for group, split in assignments.items():
        split_counts[split] += counts[group]
        split_groups[split] += 1
    return assignments, {
        "algorithm": "bounded_stratified_group_hash_accumulation_v1",
        "namespace": namespace,
        "seed": "molor.embedded-source.v1",
        "input_records": total,
        "evaluation_target_records": target,
        "maximum_single_eval_group_records": cap,
        "records_by_split": {split: split_counts[split] for split in SPLITS},
        "groups_by_split": {split: split_groups[split] for split in SPLITS},
        "features_by_split": feature_receipt,
        "required_features": sorted(required),
        "target_reached": {
            "validation": achieved["validation"] >= target,
            "test": achieved["test"] >= target,
        },
    }


def _write_json_line(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(_compact(value))
    stream.write("\n")


def _pairwise_overlap(sets: Mapping[str, set[str] | set[int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            result[f"{left}__{right}"] = len(sets[left] & sets[right])
    return result


def _output_file_receipt(path: Path, *, relative_to: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_local_tokenizer(path: Path) -> ChatTemplateTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional CLI dependency
        raise RuntimeError("transformers is required for the embedded-source CLI") from exc
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def materialize_embedded_source_experts(
    *, plan: BuildPlan, output_dir: str | Path, tokenizer: ChatTemplateTokenizer
) -> Path:
    """Build grounded reference/CPT and verbatim OCB SFT artifacts atomically."""

    destination = _safe_path(Path(output_dir), label="output")
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_receipts = _verify_plan_inputs(plan)
    non_emitted: list[dict[str, Any]] = []
    code_records = _load_code_records(plan.code, non_emitted)
    table_records, qa_records = _load_ocb_records(plan.ocb, non_emitted)
    view_receipt = _verify_ocb_training_view(
        database=plan.ocb_database,
        tables=table_records,
        expected_rows=plan.ocb_view_rows,
        expected_sha256=plan.ocb_view_sha256,
    )
    qa_matches = _match_existing_qas(table_records, qa_records)

    rendered_tokens: dict[str, int] = {}
    accepted_qas: list[OCBQARecord] = []
    for record in qa_records:
        digest = _message_hash(record.messages)
        try:
            rendered_tokens[digest] = render_token_count(
                tokenizer, list(record.messages), template_mode="direct"
            )
        except (TypeError, ValueError, RuntimeError):
            non_emitted.append(
                {
                    "disposition": "quarantined",
                    "reason": "qwen_chat_template_render_error",
                    "record_sha256": record.record_sha256,
                    "source": "ocb-can-signal-db",
                    "source_line": record.source_line,
                    "source_split": record.source_split,
                    "upstream_id": record.upstream_id,
                }
            )
            continue
        accepted_qas.append(record)
    qa_records = accepted_qas

    code_group_counts = Counter(record.repo for record in code_records)
    code_group_features: dict[str, set[str]] = {repo: set() for repo in code_group_counts}
    for record in code_records:
        code_group_features[record.repo].add(classify_code_leaf(record.repo, record.file_path))
    ocb_group_counts: Counter[str] = Counter(_vehicle_group(record.row) for record in table_records)
    for record in qa_records:
        match = qa_matches[_message_hash(record.messages)]
        assert match.vehicle_groups is not None
        ocb_group_counts[next(iter(match.vehicle_groups))] += 1
    code_splits, code_split_receipt = assign_stratified_group_splits(
        code_group_counts,
        features=code_group_features,
        required_features=LEAVES,
        namespace="code-repository",
    )
    ocb_splits, ocb_split_receipt = assign_group_splits(
        ocb_group_counts, namespace="ocb-vehicle-family"
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    output_counts: Counter[tuple[str, ...]] = Counter()
    token_min: dict[tuple[str, str, str], int] = {}
    token_max: dict[tuple[str, str, str], int] = {}
    code_groups_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    ocb_groups_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    signal_ids_by_split: dict[str, set[int]] = {split: set() for split in SPLITS}
    prompt_hashes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    contextual_families_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    reference_hashes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    sft_hashes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    template_counts: Counter[str] = Counter()

    try:
        with ExitStack() as stack:
            reference_writers: dict[tuple[str, str, str], Any] = {}
            for leaf in LEAVES:
                for split in SPLITS:
                    root = staging / "reference" / leaf
                    root.mkdir(parents=True, exist_ok=True)
                    reference_writers[(leaf, split, "data")] = stack.enter_context(
                        (root / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
                    )
                    reference_writers[(leaf, split, "provenance")] = stack.enter_context(
                        (root / f"{split}.provenance.jsonl").open(
                            "w", encoding="utf-8", newline="\n"
                        )
                    )

            sft_writers: dict[tuple[str, str, str, str], Any] = {}
            for family in OCB_QA_FAMILIES:
                for split in SPLITS:
                    root = staging / "sft" / "embedded-can-diagnostics" / "direct" / family / split
                    root.mkdir(parents=True, exist_ok=True)
                    for tier, _, _ in TIER_SPECS:
                        sft_writers[(family, split, tier, "data")] = stack.enter_context(
                            (root / f"{tier}.jsonl").open("w", encoding="utf-8", newline="\n")
                        )
                        sft_writers[(family, split, tier, "provenance")] = stack.enter_context(
                            (root / f"{tier}.provenance.jsonl").open(
                                "w", encoding="utf-8", newline="\n"
                            )
                        )

            non_emitted_writer = stack.enter_context(
                (staging / "non_emitted_dispositions.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                )
            )

            for record in sorted(
                code_records,
                key=lambda item: (
                    item.repo.casefold(),
                    item.file_path.casefold(),
                    item.upstream_id,
                ),
            ):
                leaf = classify_code_leaf(record.repo, record.file_path)
                split = code_splits[record.repo]
                group_id = _opaque_group("code-repo", record.repo.casefold())
                text_hash = _sha256_bytes(record.text.encode("utf-8"))
                _write_json_line(reference_writers[(leaf, split, "data")], {"text": record.text})
                _write_json_line(
                    reference_writers[(leaf, split, "provenance")],
                    {
                        "content_sha256": text_hash,
                        "disposition": "emitted",
                        "file_path": record.file_path,
                        "group_id": group_id,
                        "leaf": leaf,
                        "record_sha256": record.record_sha256,
                        "repo": record.repo,
                        "source": "code-corpus",
                        "source_line": record.source_line,
                        "source_split": record.source_split,
                        "split": split,
                        "upstream_id": record.upstream_id,
                    },
                )
                output_counts[("reference", leaf, split)] += 1
                code_groups_by_split[split].add(group_id)
                reference_hashes_by_split[split].add(text_hash)

            for record in sorted(table_records, key=lambda item: int(item.row["signal_id"])):
                leaf = "embedded-can-diagnostics"
                vehicle_group = _vehicle_group(record.row)
                split = ocb_splits[vehicle_group]
                group_id = _opaque_group("ocb-vehicle", vehicle_group)
                text = _compact(record.row)
                text_hash = _sha256_bytes(text.encode("utf-8"))
                _write_json_line(reference_writers[(leaf, split, "data")], {"text": text})
                _write_json_line(
                    reference_writers[(leaf, split, "provenance")],
                    {
                        "content_sha256": text_hash,
                        "disposition": "emitted",
                        "group_id": group_id,
                        "leaf": leaf,
                        "record_sha256": record.record_sha256,
                        "signal_id": record.row["signal_id"],
                        "source": "ocb-can-signal-db",
                        "source_kind": "table_row",
                        "source_line": record.source_line,
                        "source_split": record.source_split,
                        "split": split,
                        "upstream_id": record.upstream_id,
                    },
                )
                output_counts[("reference", leaf, split)] += 1
                ocb_groups_by_split[split].add(group_id)
                signal_ids_by_split[split].add(int(record.row["signal_id"]))
                reference_hashes_by_split[split].add(text_hash)

            for record in sorted(qa_records, key=lambda item: item.upstream_id):
                message_hash = _message_hash(record.messages)
                match = qa_matches[message_hash]
                assert match.vehicle_groups is not None
                assert match.signal_ids is not None
                assert match.family is not None
                vehicle_group = next(iter(match.vehicle_groups))
                split = ocb_splits[vehicle_group]
                group_id = _opaque_group("ocb-vehicle", vehicle_group)
                token_count = rendered_tokens[message_hash]
                tier = tier_for_token_count(token_count)
                prompt = next(
                    message["content"]
                    for message in reversed(record.messages)
                    if message["role"] == "user"
                )
                prompt_hash = _sha256_bytes(" ".join(prompt.split()).casefold().encode("utf-8"))
                contextual_family = _sha256_bytes(f"{match.family}\0{group_id}".encode("utf-8"))
                _write_json_line(
                    sft_writers[(match.family, split, tier, "data")],
                    {"messages": list(record.messages)},
                )
                _write_json_line(
                    sft_writers[(match.family, split, tier, "provenance")],
                    {
                        "content_sha256": message_hash,
                        "contextual_prompt_family_sha256": contextual_family,
                        "disposition": "emitted",
                        "global_template_family": match.family,
                        "group_id": group_id,
                        "leaf": "embedded-can-diagnostics",
                        "matched_signal_ids": sorted(match.signal_ids),
                        "mode": "direct",
                        "prompt_sha256": prompt_hash,
                        "record_sha256": record.record_sha256,
                        "rendered_tokens": token_count,
                        "source": "ocb-can-signal-db",
                        "source_kind": "qa",
                        "source_line": record.source_line,
                        "source_split": record.source_split,
                        "split": split,
                        "tier": tier,
                        "upstream_id": record.upstream_id,
                    },
                )
                key = (match.family, split, tier)
                token_min[key] = min(token_min.get(key, token_count), token_count)
                token_max[key] = max(token_max.get(key, token_count), token_count)
                output_counts[
                    (
                        "sft",
                        "embedded-can-diagnostics",
                        "direct",
                        match.family,
                        split,
                        tier,
                    )
                ] += 1
                ocb_groups_by_split[split].add(group_id)
                prompt_hashes_by_split[split].add(prompt_hash)
                contextual_families_by_split[split].add(contextual_family)
                sft_hashes_by_split[split].add(message_hash)
                template_counts[match.family] += 1

            for disposition in sorted(
                non_emitted,
                key=lambda item: (
                    str(item.get("source", "")),
                    str(item.get("source_split", "")),
                    int(item.get("source_line", 0)),
                ),
            ):
                _write_json_line(non_emitted_writer, disposition)

        emitted_physical = len(code_records) + len(table_records) + len(qa_records)
        physical_inputs = sum(
            descriptor.records or 0 for descriptor in (*plan.code.values(), *plan.ocb.values())
        )
        if physical_inputs != emitted_physical + len(non_emitted):
            raise AssertionError(
                f"physical input accounting mismatch: {physical_inputs} != "
                f"{emitted_physical} + {len(non_emitted)}"
            )

        output_receipts: list[dict[str, Any]] = []
        sums: list[str] = []
        for leaf in LEAVES:
            for split in SPLITS:
                root = staging / "reference" / leaf
                records = output_counts[("reference", leaf, split)]
                data = _output_file_receipt(
                    root / f"{split}.jsonl", relative_to=staging, records=records
                )
                provenance = _output_file_receipt(
                    root / f"{split}.provenance.jsonl",
                    relative_to=staging,
                    records=records,
                )
                output_receipts.append(
                    {
                        "leaf": leaf,
                        "modality": "reference_cpt",
                        "split": split,
                        "data": data,
                        "provenance": provenance,
                    }
                )
                sums.extend(
                    (
                        f"{data['sha256']}  {data['path']}",
                        f"{provenance['sha256']}  {provenance['path']}",
                    )
                )
        for family in OCB_QA_FAMILIES:
            for split in SPLITS:
                for tier, _, _ in TIER_SPECS:
                    root = staging / "sft" / "embedded-can-diagnostics" / "direct" / family / split
                    records = output_counts[
                        (
                            "sft",
                            "embedded-can-diagnostics",
                            "direct",
                            family,
                            split,
                            tier,
                        )
                    ]
                    data = _output_file_receipt(
                        root / f"{tier}.jsonl", relative_to=staging, records=records
                    )
                    provenance = _output_file_receipt(
                        root / f"{tier}.provenance.jsonl",
                        relative_to=staging,
                        records=records,
                    )
                    output_receipts.append(
                        {
                            "leaf": "embedded-can-diagnostics",
                            "modality": "sft",
                            "mode": "direct",
                            "template_family": family,
                            "split": split,
                            "tier": tier,
                            "min_rendered_tokens": token_min.get((family, split, tier)),
                            "max_rendered_tokens": token_max.get((family, split, tier)),
                            "data": data,
                            "provenance": provenance,
                        }
                    )
                    sums.extend(
                        (
                            f"{data['sha256']}  {data['path']}",
                            f"{provenance['sha256']}  {provenance['path']}",
                        )
                    )
        non_emitted_receipt = _output_file_receipt(
            staging / "non_emitted_dispositions.jsonl",
            relative_to=staging,
            records=len(non_emitted),
        )
        sums.append(f"{non_emitted_receipt['sha256']}  {non_emitted_receipt['path']}")

        non_emitted_counts = Counter(
            (str(item["disposition"]), str(item["reason"])) for item in non_emitted
        )
        leaf_totals = {
            leaf: {split: output_counts[("reference", leaf, split)] for split in SPLITS}
            for leaf in LEAVES
        }
        sft_family_totals = {
            family: {
                split: {
                    tier: output_counts[
                        (
                            "sft",
                            "embedded-can-diagnostics",
                            "direct",
                            family,
                            split,
                            tier,
                        )
                    ]
                    for tier, _, _ in TIER_SPECS
                }
                for split in SPLITS
            }
            for family in OCB_QA_FAMILIES
        }
        sft_totals = {
            split: {
                tier: sum(sft_family_totals[family][split][tier] for family in OCB_QA_FAMILIES)
                for tier, _, _ in TIER_SPECS
            }
            for split in SPLITS
        }
        leakage = {
            "code_repository_group_overlap": _pairwise_overlap(code_groups_by_split),
            "ocb_vehicle_family_group_overlap": _pairwise_overlap(ocb_groups_by_split),
            "ocb_signal_id_overlap": _pairwise_overlap(signal_ids_by_split),
            "exact_normalized_prompt_overlap": _pairwise_overlap(prompt_hashes_by_split),
            "contextual_prompt_family_overlap": _pairwise_overlap(contextual_families_by_split),
            "exact_reference_content_overlap": _pairwise_overlap(reference_hashes_by_split),
            "exact_sft_message_overlap": _pairwise_overlap(sft_hashes_by_split),
        }
        if any(value for audit in leakage.values() for value in audit.values()):
            raise AssertionError(f"cross-split leakage detected: {leakage!r}")

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": {
                "code_file_text_policy": "verbatim_text_only_reference_cpt",
                "ocb_table_policy": "canonical_json_of_exact_training_corpus_row_reference_cpt",
                "ocb_sft_policy": "existing_prepared_qa_messages_verbatim",
                "ocb_sft_template_family_isolation": True,
                "invented_qa": False,
                "messages_only_sft": True,
                "metadata_sidecars_only": True,
                "prepared_row_splits_reused": False,
                "qwen_length_rendering": {
                    "enable_thinking": False,
                    "truncation": False,
                    "tiers": [
                        {"name": name, "minimum": lower, "maximum": upper}
                        for name, lower, upper in TIER_SPECS
                    ],
                },
            },
            "taxonomy": {
                "leaves": list(LEAVES),
                "classification": "repository_and_file_path_metadata_v1",
                "one_primary_leaf_per_physical_code_file": True,
                "ocb_leaf": "embedded-can-diagnostics",
            },
            "provenance": {
                "plan": {"path": str(plan.path), "sha256": plan.sha256},
                "builder": {
                    "path": Path(__file__).name,
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                **input_receipts,
                "ocb_training_view": view_receipt,
            },
            "split": {
                "code": code_split_receipt,
                "ocb": ocb_split_receipt,
                "code_group_key": "repository",
                "ocb_group_key": "make/model/year vehicle family",
            },
            "accounting": {
                "physical_input_records": physical_inputs,
                "emitted_physical_records": emitted_physical,
                "non_emitted_physical_records": len(non_emitted),
                "non_emitted_by_disposition_and_reason": {
                    f"{disposition}:{reason}": count
                    for (disposition, reason), count in sorted(non_emitted_counts.items())
                },
                "input_kinds": {
                    "code_file": len(code_records),
                    "ocb_table_row": len(table_records),
                    "ocb_qa": len(qa_records),
                },
                "reference_by_leaf_and_split": leaf_totals,
                "sft_by_split_and_tier": sft_totals,
                "sft_by_template_family_split_and_tier": sft_family_totals,
                "ocb_global_template_families": dict(sorted(template_counts.items())),
                "fully_reconciled": True,
                "zero_silent_drops": True,
            },
            "leakage": {
                **leakage,
                "all_enforced_overlap_counts_zero": True,
                "global_template_shapes": {
                    "count": len(template_counts),
                    "names": sorted(template_counts),
                    "shared_across_splits": True,
                    "interpretation": (
                        "Template wording is deliberately shared; contextual template family plus "
                        "vehicle group is the leakage boundary."
                    ),
                },
            },
            "prior_lineage": {
                "artifacts": input_receipts["prior_derived_reports"],
                "aggregate_reports_only": True,
                "reported_payloads_opened": False,
                "relationship": "complements_not_supersedes",
            },
            "limitations": [
                "The prepared OCB QA answers are preserved verbatim and are not promoted to the "
                "higher-trust status of the prior verified CAN subsets.",
                "The OCB training_corpus view does not expose a repository/source identifier; "
                "vehicle-family isolation is used as the strongest available grouping surrogate.",
                "Global three-shape QA template wording is shared across splits; exact prompts and "
                "contextual template families are isolated.",
                "New code-derived outputs are reference/CPT documents only; prior grounded and "
                "verified SFT remains a separately pinned mixture supplement.",
            ],
            "outputs": output_receipts,
            "non_emitted_ledger": non_emitted_receipt,
        }
        manifest_payload = _canonical_document(manifest)
        (staging / "manifest.json").write_bytes(manifest_payload)
        manifest_sha = _sha256_bytes(manifest_payload)
        sums.append(f"{manifest_sha}  manifest.json")
        (staging / "sha256sums.txt").write_text(
            "\n".join(sorted(sums)) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(staging, destination)
        return destination / "manifest.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _count_lines_and_sha256(path: Path) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            count += 1
    return count, digest.hexdigest()


def verify_materialized_embedded_source(
    *, output_dir: str | Path, tokenizer: ChatTemplateTokenizer
) -> dict[str, Any]:
    """Independently re-read every output, sidecar, tier, checksum, and split boundary."""

    root = _safe_path(Path(output_dir), label="verification output")
    manifest_path = root / "manifest.json"
    manifest = _strict_json_object(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported embedded-source output schema")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("manifest outputs must be a list")

    groups: dict[str, dict[str, set[str]]] = {
        "code": {split: set() for split in SPLITS},
        "ocb": {split: set() for split in SPLITS},
    }
    signals: dict[str, set[int]] = {split: set() for split in SPLITS}
    prompts: dict[str, set[str]] = {split: set() for split in SPLITS}
    contextual: dict[str, set[str]] = {split: set() for split in SPLITS}
    reference_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    sft_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    physical_emissions = 0
    sft_rows = 0
    files_verified = 0

    for receipt in outputs:
        if not isinstance(receipt, dict):
            raise ValueError("output receipt must be an object")
        split = receipt.get("split")
        modality = receipt.get("modality")
        if split not in SPLITS or modality not in {"reference_cpt", "sft"}:
            raise ValueError("output receipt split/modality is malformed")
        data_receipt = receipt.get("data")
        provenance_receipt = receipt.get("provenance")
        if not isinstance(data_receipt, dict) or not isinstance(provenance_receipt, dict):
            raise ValueError("output receipt lacks data/provenance identities")
        data = (root / str(data_receipt["path"])).resolve()
        provenance = (root / str(provenance_receipt["path"])).resolve()
        if root != data.parent and root not in data.parents:
            raise ValueError("data output escaped output root")
        if root != provenance.parent and root not in provenance.parents:
            raise ValueError("provenance output escaped output root")
        data_count, data_sha = _count_lines_and_sha256(data)
        provenance_count, provenance_sha = _count_lines_and_sha256(provenance)
        expected_records = data_receipt.get("records")
        if (
            data_count != expected_records
            or provenance_count != expected_records
            or data_sha != data_receipt.get("sha256")
            or provenance_sha != provenance_receipt.get("sha256")
        ):
            raise RuntimeError(f"receipt/alignment mismatch for {data}")
        files_verified += 2

        with (
            data.open("r", encoding="utf-8") as data_stream,
            provenance.open("r", encoding="utf-8") as provenance_stream,
        ):
            for line_number, (data_line, provenance_line) in enumerate(
                zip(data_stream, provenance_stream, strict=True), start=1
            ):
                row = json.loads(data_line)
                sidecar = json.loads(provenance_line)
                if not isinstance(sidecar, dict) or sidecar.get("disposition") != "emitted":
                    raise ValueError(f"invalid provenance at {provenance}:{line_number}")
                group_id = sidecar.get("group_id")
                if not isinstance(group_id, str) or sidecar.get("split") != split:
                    raise ValueError(
                        f"invalid group/split provenance at {provenance}:{line_number}"
                    )
                source = sidecar.get("source")
                group_kind = "code" if source == "code-corpus" else "ocb"
                groups[group_kind][split].add(group_id)
                physical_emissions += 1

                if modality == "reference_cpt":
                    if not isinstance(row, dict) or set(row) != {"text"}:
                        raise ValueError(
                            f"reference row is not exact text-only: {data}:{line_number}"
                        )
                    text = row["text"]
                    if not isinstance(text, str) or not text:
                        raise ValueError(f"empty reference document: {data}:{line_number}")
                    content_hash = _sha256_bytes(text.encode("utf-8"))
                    if content_hash != sidecar.get("content_sha256"):
                        raise RuntimeError(f"reference content hash mismatch: {data}:{line_number}")
                    reference_hashes[split].add(content_hash)
                    signal_id = sidecar.get("signal_id")
                    if type(signal_id) is int:
                        signals[split].add(signal_id)
                else:
                    if not isinstance(row, dict) or set(row) != {"messages"}:
                        raise ValueError(f"SFT row is not messages-only: {data}:{line_number}")
                    messages = _valid_messages(row["messages"])
                    if messages is None:
                        raise ValueError(f"invalid SFT messages: {data}:{line_number}")
                    content_hash = _message_hash(messages)
                    if content_hash != sidecar.get("content_sha256"):
                        raise RuntimeError(f"SFT content hash mismatch: {data}:{line_number}")
                    token_count = render_token_count(
                        tokenizer, list(messages), template_mode="direct"
                    )
                    if (
                        token_count != sidecar.get("rendered_tokens")
                        or tier_for_token_count(token_count) != receipt.get("tier")
                        or receipt.get("tier") != sidecar.get("tier")
                        or receipt.get("template_family") != sidecar.get("global_template_family")
                    ):
                        raise RuntimeError(f"SFT token tier mismatch: {data}:{line_number}")
                    prompts[split].add(str(sidecar["prompt_sha256"]))
                    contextual[split].add(str(sidecar["contextual_prompt_family_sha256"]))
                    sft_hashes[split].add(content_hash)
                    sft_rows += 1

    leakage = {
        "code_repository_group_overlap": _pairwise_overlap(groups["code"]),
        "ocb_vehicle_family_group_overlap": _pairwise_overlap(groups["ocb"]),
        "ocb_signal_id_overlap": _pairwise_overlap(signals),
        "exact_normalized_prompt_overlap": _pairwise_overlap(prompts),
        "contextual_prompt_family_overlap": _pairwise_overlap(contextual),
        "exact_reference_content_overlap": _pairwise_overlap(reference_hashes),
        "exact_sft_message_overlap": _pairwise_overlap(sft_hashes),
    }
    if any(value for audit in leakage.values() for value in audit.values()):
        raise RuntimeError(f"independent verification found cross-split leakage: {leakage!r}")

    ledger = manifest.get("non_emitted_ledger")
    if not isinstance(ledger, dict):
        raise ValueError("manifest lacks non-emitted ledger receipt")
    ledger_path = (root / str(ledger["path"])).resolve()
    ledger_count, ledger_sha = _count_lines_and_sha256(ledger_path)
    if ledger_count != ledger.get("records") or ledger_sha != ledger.get("sha256"):
        raise RuntimeError("non-emitted ledger receipt mismatch")
    files_verified += 1

    accounting = manifest.get("accounting")
    if not isinstance(accounting, dict):
        raise ValueError("manifest accounting is malformed")
    if physical_emissions != accounting.get("emitted_physical_records"):
        raise RuntimeError("emitted physical record count mismatch")
    if ledger_count != accounting.get("non_emitted_physical_records"):
        raise RuntimeError("non-emitted physical record count mismatch")
    if physical_emissions + ledger_count != accounting.get("physical_input_records"):
        raise RuntimeError("physical input reconciliation mismatch")

    sums_path = root / "sha256sums.txt"
    checked_sums = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = (root / relative).resolve()
        if root != target.parent and root not in target.parents:
            raise ValueError("checksum target escaped output root")
        if sha256_file(target) != digest:
            raise RuntimeError(f"checksum mismatch: {relative}")
        checked_sums += 1
    expected_sum_entries = len(outputs) * 2 + 2
    if checked_sums != expected_sum_entries:
        raise RuntimeError(
            f"checksum entry count mismatch: expected {expected_sum_entries}, got {checked_sums}"
        )
    return {
        "schema_version": "molor.embedded-source-verification.v1",
        "manifest_sha256": sha256_file(manifest_path),
        "checksum_entries_verified": checked_sums,
        "output_files_verified": files_verified,
        "physical_input_records": accounting["physical_input_records"],
        "emitted_physical_records": physical_emissions,
        "non_emitted_physical_records": ledger_count,
        "sft_rows_retokenized": sft_rows,
        "leakage": leakage,
        "fully_verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    tokenizer = _load_local_tokenizer(plan.tokenizer_path)
    if not args.verify_only:
        manifest = materialize_embedded_source_experts(
            plan=plan, output_dir=args.output, tokenizer=tokenizer
        )
    else:
        manifest = args.output.resolve() / "manifest.json"
    verification = verify_materialized_embedded_source(output_dir=args.output, tokenizer=tokenizer)
    print(
        json.dumps(
            {"manifest": str(manifest), "verification": verification},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
