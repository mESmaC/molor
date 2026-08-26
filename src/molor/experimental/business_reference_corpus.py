"""Build a grounded, deduplicated business reference corpus from pinned snapshots.

The output is reference text for a later, separately reviewed teacher-generation stage.  It
does not invent questions, answers, rationales, or chat turns.  Source provenance is kept out
of document shards and written only to sidecars.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
import html as html_module
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any
import unicodedata
from zipfile import BadZipFile, ZipFile, ZipInfo
import xml.etree.ElementTree as ElementTree

from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256


REFERENCE_SCHEMA = "molor.business-reference-corpus.v1"
ARTIFACT_PROVENANCE_SCHEMA = "molor.business-reference-artifact-provenance.v1"
DOCUMENT_PROVENANCE_SCHEMA = "molor.business-reference-document-provenance.v1"
CONTAINER_PROVENANCE_SCHEMA = "molor.business-reference-container-provenance.v1"
ACQUISITION_SCHEMA = "molor.business-source-snapshot.v1"

LEAVES = (
    "business-core",
    "business-finance",
    "growth-gtm",
    "investing-markets",
    "funding-capital",
    "business-operations",
)
SPLITS = ("train", "validation", "test")
SELECTED_SOURCE_IDS = (
    "fdic-sba-money-smart-small-business",
    "gitlab-company-handbook",
    "onet-business-work-activities",
    "posthog-handbook-and-manual",
    "sba-business-guide-and-content-api",
    "yc-startup-library",
)
PREHANDLED_SOURCE_IDS = frozenset(("finqa", "tat-qa"))
EXPECTED_SOURCE_ROOTS = {
    "fdic-sba-money-smart-small-business": "fdic-money-smart-small-business",
    "gitlab-company-handbook": "gitlab-handbook",
    "onet-business-work-activities": "onet-31.0",
    "posthog-handbook-and-manual": "posthog-handbook",
    "sba-business-guide-and-content-api": "sba-business-guide",
    "yc-startup-library": "yc-startup-library",
}
TRUSTED_BUSINESS_SOURCES = frozenset(
    (
        "fdic-sba-money-smart-small-business",
        "onet-business-work-activities",
        "sba-business-guide-and-content-api",
        "yc-startup-library",
    )
)

DEFAULT_MINIMUM_CHARACTERS = 240
DEFAULT_MAXIMUM_CHARACTERS = 6000
DEFAULT_NEAR_DUPLICATE_DISTANCE = 3
MAXIMUM_ARTIFACT_BYTES = 512 * 1024 * 1024
MAXIMUM_ZIP_MEMBER_BYTES = 256 * 1024 * 1024
MAXIMUM_SELECTED_ZIP_BYTES = 1024 * 1024 * 1024
_BLOCKED_PATH_WORDS = ("cyber", "security")

_ONET_SELECTED_MEMBERS = (
    "db_31_0_json/occupation_data.json",
    "db_31_0_json/task_statements.json",
    "db_31_0_json/work_activities.json",
    "db_31_0_json/knowledge.json",
    "db_31_0_json/essential_skills.json",
    "db_31_0_json/work_styles.json",
)
_ONET_BUSINESS_MAJOR_GROUPS = frozenset(("11", "13", "41", "43"))

_LEAF_TERMS: dict[str, dict[str, int]] = {
    "business-core": {
        "business plan": 4,
        "business model": 4,
        "entrepreneur": 4,
        "startup": 3,
        "small business": 3,
        "founder": 2,
        "company strategy": 3,
        "competitive advantage": 3,
        "incorporation": 2,
        "franchise": 2,
    },
    "business-finance": {
        "accounting": 4,
        "balance sheet": 5,
        "cash flow": 5,
        "financial statement": 5,
        "financial planning": 4,
        "budget": 3,
        "bookkeeping": 4,
        "accounts payable": 4,
        "accounts receivable": 4,
        "revenue recognition": 5,
        "profit and loss": 5,
        "tax": 2,
        "payroll": 2,
        "expense": 2,
        "margin": 2,
        "treasury": 3,
    },
    "growth-gtm": {
        "go to market": 5,
        "go-to-market": 5,
        "marketing": 3,
        "sales": 3,
        "customer acquisition": 5,
        "customer success": 4,
        "retention": 3,
        "churn": 4,
        "pricing": 4,
        "product market fit": 5,
        "product-market fit": 5,
        "conversion": 3,
        "sales pipeline": 4,
        "brand": 2,
        "market research": 3,
        "distribution channel": 3,
        "growth": 2,
    },
    "investing-markets": {
        "public markets": 5,
        "stock market": 5,
        "capital markets": 5,
        "portfolio management": 5,
        "investment analysis": 5,
        "investment banking": 4,
        "asset management": 4,
        "equity research": 5,
        "securities": 4,
        "fixed income": 5,
        "bond market": 5,
        "market valuation": 4,
        "financial analyst": 3,
        "investing": 3,
        "portfolio": 2,
    },
    "funding-capital": {
        "fundraising": 5,
        "venture capital": 5,
        "seed round": 5,
        "series a": 5,
        "term sheet": 5,
        "cap table": 5,
        "angel investor": 5,
        "pitch deck": 4,
        "raise capital": 5,
        "raising capital": 5,
        "business loan": 4,
        "debt financing": 5,
        "equity financing": 5,
        "grant funding": 4,
        "investor": 2,
        "funding": 3,
        "runway": 2,
    },
    "business-operations": {
        "business operations": 5,
        "operations management": 5,
        "people operations": 5,
        "project management": 4,
        "process improvement": 4,
        "procurement": 4,
        "supply chain": 4,
        "inventory management": 4,
        "human resources": 4,
        "hiring": 3,
        "recruiting": 3,
        "management": 2,
        "leadership": 2,
        "governance": 3,
        "compliance": 3,
        "legal": 2,
        "workflow": 2,
        "planning": 2,
        "organizational": 2,
    },
}
_LEAF_TIE_ORDER = (
    "funding-capital",
    "investing-markets",
    "business-finance",
    "growth-gtm",
    "business-operations",
    "business-core",
)

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?"
    r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.DOTALL,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|"
    r"passwd|secret)\b(\s*[:=]\s*)(?:[\"']?)[A-Za-z0-9_./+\-=]{8,}(?:[\"']?)"
)
_AWS_KEY_RE = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
_SERVICE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,})(?![A-Za-z0-9_])"
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])")
_PHONE_RE = re.compile(
    r"(?<![\w])(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s])\d{3}[-.\s]\d{4}(?![\w])"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IPV4_RE = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
)


def _blocked(parts: Iterable[str]) -> bool:
    return any(word in part.casefold() for part in parts for word in _BLOCKED_PATH_WORDS)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json(raw: bytes, label: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains unsupported numeric constant {value!r}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc


def _lowercase_sha256(value: object, field_name: str) -> str:
    digest = validate_sha256(value, field_name)  # type: ignore[arg-type]
    if value != digest:
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    return digest


def _positive_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class PinnedArtifact:
    source_id: str
    source_root: str
    relative_path: str
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        if self.source_id not in SELECTED_SOURCE_IDS:
            raise ValueError(f"source is outside the six-source reference build: {self.source_id}")
        if EXPECTED_SOURCE_ROOTS[self.source_id] != self.source_root:
            raise ValueError(f"unexpected source root for {self.source_id}")
        if not self.relative_path or "\\" in self.relative_path or "\x00" in self.relative_path:
            raise ValueError("artifact paths must be non-empty canonical POSIX paths")
        root = PurePosixPath(self.source_root)
        relative = PurePosixPath(self.relative_path)
        if (
            root.is_absolute()
            or relative.is_absolute()
            or ".." in root.parts
            or ".." in relative.parts
            or _blocked(root.parts)
            or _blocked(relative.parts)
        ):
            raise ValueError("blocked or escaping source path in acquisition receipt")
        object.__setattr__(self, "sha256", _lowercase_sha256(self.sha256, "artifact sha256"))
        if (
            not isinstance(self.byte_count, int)
            or isinstance(self.byte_count, bool)
            or self.byte_count < 0
        ):
            raise ValueError("artifact byte count must be a non-negative integer")

    @property
    def receipt_path(self) -> str:
        return f"{self.source_root}/{self.relative_path}"


@dataclass(frozen=True, slots=True)
class BusinessReceipt:
    snapshot_root_label: str
    receipt_sha256: str
    artifacts_by_source: Mapping[str, tuple[PinnedArtifact, ...]]
    excluded_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawDocument:
    title: str
    text: str
    family_key: str
    internal_reference: str
    trusted_business: bool = False


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    parser: str
    documents: tuple[RawDocument, ...]
    reason: str
    container_rows: tuple[dict[str, object], ...] = ()
    container_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BuildConfig:
    minimum_characters: int = DEFAULT_MINIMUM_CHARACTERS
    maximum_characters: int = DEFAULT_MAXIMUM_CHARACTERS
    near_duplicate_distance: int = DEFAULT_NEAR_DUPLICATE_DISTANCE

    def __post_init__(self) -> None:
        minimum = _positive_integer(self.minimum_characters, "minimum_characters")
        maximum = _positive_integer(self.maximum_characters, "maximum_characters")
        if maximum < minimum * 2:
            raise ValueError("maximum_characters must be at least twice minimum_characters")
        if (
            not isinstance(self.near_duplicate_distance, int)
            or isinstance(self.near_duplicate_distance, bool)
            or not 0 <= self.near_duplicate_distance <= 8
        ):
            raise ValueError("near_duplicate_distance must be an integer from 0 through 8")


class _ShardAccumulator:
    def __init__(self, destination: Path | None) -> None:
        self.destination = destination
        self._digest = hashlib.sha256()
        self.bytes = 0
        self.records = 0
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._stream = destination.open("wb")
        else:
            self._stream = None

    def add(self, value: Mapping[str, object]) -> None:
        payload = canonical_json_bytes(value) + b"\n"
        self._digest.update(payload)
        self.bytes += len(payload)
        self.records += 1
        if self._stream is not None:
            self._stream.write(payload)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def load_business_receipt(path: str | Path) -> BusinessReceipt:
    source = Path(path).expanduser()
    if _blocked(source.parts):
        raise ValueError("receipt path is outside the permitted business boundary")
    if source.is_symlink() or not source.is_file():
        raise ValueError("receipt must be a regular non-symlink file")
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("receipt changed while it was loaded")
    value = _strict_json(raw, "acquisition receipt")
    if not isinstance(value, Mapping) or value.get("schema_version") != ACQUISITION_SCHEMA:
        raise ValueError("unsupported acquisition receipt schema")
    if value.get("license_or_rights_metadata_used_for_admission") is not False:
        raise ValueError("receipt must confirm that license metadata was not an admission gate")
    snapshot_root_label = value.get("snapshot_root_label")
    raw_sources = value.get("sources")
    if not isinstance(snapshot_root_label, str) or not snapshot_root_label:
        raise ValueError("receipt snapshot_root_label must be non-empty")
    if not isinstance(raw_sources, list):
        raise ValueError("receipt sources must be an array")

    selected: dict[str, tuple[PinnedArtifact, ...]] = {}
    excluded: list[str] = []
    seen_source_ids: set[str] = set()
    for source_index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"receipt source {source_index} must be an object")
        source_id = raw_source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"receipt source {source_index} has no source_id")
        if source_id in seen_source_ids:
            raise ValueError(f"receipt repeats source_id {source_id!r}")
        seen_source_ids.add(source_id)
        if source_id in PREHANDLED_SOURCE_IDS:
            excluded.append(source_id)
            continue
        if source_id not in SELECTED_SOURCE_IDS:
            raise ValueError(f"unexpected acquisition source {source_id!r}")
        source_root = raw_source.get("relative_root")
        artifacts = raw_source.get("artifacts")
        file_count = raw_source.get("file_count")
        byte_count = raw_source.get("bytes")
        if not isinstance(source_root, str) or not isinstance(artifacts, list):
            raise ValueError(f"receipt source {source_id!r} has invalid artifacts")
        if source_root != EXPECTED_SOURCE_ROOTS[source_id] or not artifacts:
            raise ValueError(
                f"receipt source {source_id!r} has an invalid source root/artifact set"
            )
        pins: list[PinnedArtifact] = []
        seen_paths: set[str] = set()
        for artifact_index, raw_artifact in enumerate(artifacts):
            if not isinstance(raw_artifact, Mapping) or set(raw_artifact) != {
                "path",
                "sha256",
                "bytes",
            }:
                raise ValueError(
                    f"receipt artifact {source_id}[{artifact_index}] has an invalid descriptor"
                )
            artifact_path = raw_artifact["path"]
            if not isinstance(artifact_path, str) or artifact_path in seen_paths:
                raise ValueError(f"receipt source {source_id!r} has an invalid/repeated path")
            seen_paths.add(artifact_path)
            pins.append(
                PinnedArtifact(
                    source_id=source_id,
                    source_root=source_root,
                    relative_path=artifact_path,
                    sha256=raw_artifact["sha256"],  # type: ignore[arg-type]
                    byte_count=raw_artifact["bytes"],  # type: ignore[arg-type]
                )
            )
        if file_count != len(pins) or byte_count != sum(pin.byte_count for pin in pins):
            raise ValueError(f"receipt aggregate drift for source {source_id!r}")
        selected[source_id] = tuple(sorted(pins, key=lambda pin: pin.relative_path))

    missing = set(SELECTED_SOURCE_IDS).difference(selected)
    if missing:
        raise ValueError(f"receipt is missing required business sources: {sorted(missing)!r}")
    if not PREHANDLED_SOURCE_IDS.issubset(excluded):
        raise ValueError("receipt must contain the two explicitly prehandled supervision sources")
    return BusinessReceipt(
        snapshot_root_label=snapshot_root_label,
        receipt_sha256=hashlib.sha256(raw).hexdigest(),
        artifacts_by_source=selected,
        excluded_source_ids=tuple(sorted(excluded)),
    )


def _resolve_artifacts(
    snapshot_root: Path, receipt: BusinessReceipt
) -> dict[str, tuple[tuple[PinnedArtifact, Path], ...]]:
    root = snapshot_root.expanduser().resolve()
    if _blocked(root.parts):
        raise ValueError("snapshot root is outside the permitted business boundary")
    if not root.is_dir() or root.name != receipt.snapshot_root_label:
        raise ValueError("snapshot root does not match the acquisition receipt")
    resolved: dict[str, tuple[tuple[PinnedArtifact, Path], ...]] = {}
    for source_id in SELECTED_SOURCE_IDS:
        source_root = (root / EXPECTED_SOURCE_ROOTS[source_id]).resolve()
        if source_root.parent != root or _blocked(source_root.parts) or not source_root.is_dir():
            raise ValueError(f"invalid source root for {source_id}")
        entries: list[tuple[PinnedArtifact, Path]] = []
        for pin in receipt.artifacts_by_source[source_id]:
            relative = Path(*PurePosixPath(pin.relative_path).parts)
            candidate = source_root / relative
            if _blocked(candidate.parts):
                raise ValueError("blocked source artifact path")
            try:
                path = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"pinned artifact cannot be resolved: {pin.receipt_path}") from exc
            if _blocked(path.parts) or not path.is_file():
                raise ValueError(f"pinned artifact is not a safe regular file: {pin.receipt_path}")
            try:
                path.relative_to(source_root)
            except ValueError as exc:
                raise ValueError("pinned artifact escapes its source root") from exc
            entries.append((pin, path))
        resolved[source_id] = tuple(entries)
    return resolved


def _verified_bytes(path: Path, pin: PinnedArtifact) -> bytes:
    before = path.stat()
    if before.st_size != pin.byte_count or before.st_size > MAXIMUM_ARTIFACT_BYTES:
        raise ValueError(f"artifact size drift for {pin.receipt_path}")
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ino", None),
    ) != (
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ino", None),
    ):
        raise RuntimeError(f"artifact changed while it was loaded: {pin.receipt_path}")
    if len(raw) != pin.byte_count or hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise ValueError(f"artifact hash drift for {pin.receipt_path}")
    return raw


def _decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", html_module.unescape(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[\t ]+", " ", raw_line).strip()
        if line:
            current.append(line)
        elif current:
            paragraph = " ".join(current).strip()
            if paragraph and (not paragraphs or paragraph != paragraphs[-1]):
                paragraphs.append(paragraph)
            current = []
    if current:
        paragraph = " ".join(current).strip()
        if paragraph and (not paragraphs or paragraph != paragraphs[-1]):
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _clean_markdown(text: str) -> tuple[str, str]:
    title = ""
    if text.startswith("---"):
        match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, re.DOTALL)
        if match:
            front_matter = match.group(1)
            title_match = re.search(r"(?mi)^title:\s*[\"']?(.+?)[\"']?\s*$", front_matter)
            if title_match:
                title = title_match.group(1).strip(" \"'")
            text = text[match.end() :]
    if not title:
        heading = re.search(r"(?m)^#\s+(.+?)\s*$", text)
        if heading:
            title = heading.group(1)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", " ", text, flags=re.I | re.S)
    text = re.sub(r"(?m)^\s*(?:import|export)\s+.+$", " ", text)
    text = re.sub(r"```[^\n]*\n", "\n", text)
    text = text.replace("```", "\n")
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<https?://[^>]+>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?m)^\s*\{[.#][^}]+\}\s*$", " ", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    return _normalize_text(title), _normalize_text(text)


def _html_blocks(root: Any) -> str:
    blocks: list[str] = []
    seen: set[str] = set()
    for node in root.xpath(
        ".//h1|.//h2|.//h3|.//h4|.//h5|.//h6|.//p|.//li|.//blockquote|.//pre|.//tr"
    ):
        value = _normalize_text(" ".join(node.itertext()))
        if value and value not in seen:
            seen.add(value)
            blocks.append(value)
    if not blocks:
        return _normalize_text(" ".join(root.itertext()))
    return "\n\n".join(blocks)


def _extract_generic_html(raw: bytes, family_key: str, fallback_title: str) -> ExtractionResult:
    try:
        from lxml import html as lxml_html

        document = lxml_html.fromstring(raw)
        for node in document.xpath(
            "//script|//style|//noscript|//svg|//form|//nav|//footer|//header|//template"
        ):
            node.drop_tree()
        title = _normalize_text(document.findtext(".//title") or fallback_title)
        roots = document.xpath("//main") or document.xpath("//article") or document.xpath("//body")
        root = roots[0] if roots else document
        text = _html_blocks(root)
    except Exception as exc:
        return ExtractionResult("html-lxml-v1", (), f"parse_error:{type(exc).__name__}")
    if not text:
        return ExtractionResult("html-lxml-v1", (), "no_extractable_text")
    return ExtractionResult(
        "html-lxml-v1",
        (RawDocument(title or fallback_title, text, family_key, "document"),),
        "extracted",
    )


def _extract_yc_html(raw: bytes, family_key: str, fallback_title: str) -> ExtractionResult:
    try:
        from lxml import html as lxml_html

        document = lxml_html.fromstring(raw)
        holders = document.xpath("//*[@data-page]")
        if not holders:
            return ExtractionResult("yc-data-page-v1", (), "missing_data_page")
        encoded_page = holders[0].get("data-page")
        if not isinstance(encoded_page, str):
            return ExtractionResult("yc-data-page-v1", (), "invalid_data_page")
        page = _strict_json(encoded_page.encode("utf-8"), "YC data-page")
        if not isinstance(page, Mapping):
            return ExtractionResult("yc-data-page-v1", (), "invalid_data_page")
        article = page.get("props", {}).get("article")
        if not isinstance(article, Mapping):
            return ExtractionResult("yc-data-page-v1", (), "missing_article")
        title = article.get("title")
        if not isinstance(title, str) or not title.strip():
            title = fallback_title
        pieces: list[str] = []
        for key in ("content", "transcript", "description"):
            value = article.get(key)
            if isinstance(value, str) and value.strip() and value not in pieces:
                pieces.append(value)
        _unused, text = _clean_markdown("\n\n".join(pieces))
    except Exception as exc:
        return ExtractionResult("yc-data-page-v1", (), f"parse_error:{type(exc).__name__}")
    if not text:
        return ExtractionResult("yc-data-page-v1", (), "no_extractable_article_text")
    return ExtractionResult(
        "yc-data-page-v1",
        (RawDocument(_normalize_text(title), text, family_key, "article", True),),
        "extracted",
    )


def _extract_markdown(raw: bytes, family_key: str, fallback_title: str) -> ExtractionResult:
    try:
        title, text = _clean_markdown(_decode_text(raw))
    except (UnicodeDecodeError, ValueError) as exc:
        return ExtractionResult("markdown-grounded-v1", (), f"parse_error:{type(exc).__name__}")
    if not text:
        return ExtractionResult("markdown-grounded-v1", (), "no_extractable_text")
    return ExtractionResult(
        "markdown-grounded-v1",
        (RawDocument(title or fallback_title, text, family_key, "document"),),
        "extracted",
    )


def _extract_text(raw: bytes, family_key: str, fallback_title: str) -> ExtractionResult:
    try:
        text = _normalize_text(_decode_text(raw))
    except UnicodeDecodeError as exc:
        return ExtractionResult("plain-text-v1", (), f"parse_error:{type(exc).__name__}")
    if not text:
        return ExtractionResult("plain-text-v1", (), "no_extractable_text")
    return ExtractionResult(
        "plain-text-v1",
        (RawDocument(fallback_title, text, family_key, "document"),),
        "extracted",
    )


def _extract_csv(raw: bytes, family_key: str, fallback_title: str) -> ExtractionResult:
    try:
        stream = StringIO(_decode_text(raw))
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            return ExtractionResult("csv-grounded-v1", (), "missing_header")
        paragraphs: list[str] = []
        for row in reader:
            values = [
                f"{header.strip()}: {_normalize_text(value)}"
                for header, value in row.items()
                if header and value and _normalize_text(value)
            ]
            if values:
                paragraphs.append("; ".join(values))
    except (csv.Error, UnicodeDecodeError, ValueError) as exc:
        return ExtractionResult("csv-grounded-v1", (), f"parse_error:{type(exc).__name__}")
    text = "\n\n".join(paragraphs)
    if not text:
        return ExtractionResult("csv-grounded-v1", (), "no_extractable_rows")
    return ExtractionResult(
        "csv-grounded-v1",
        (RawDocument(fallback_title, text, family_key, "table", True),),
        "extracted",
    )


def _extract_pdf(raw: bytes, family_key: str, fallback_title: str) -> ExtractionResult:
    if raw.startswith(b"PK\x03\x04"):
        return _extract_presentation_package(raw, family_key, fallback_title)
    first_error = "no_extractable_text"
    try:
        from pypdf import PdfReader

        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(BytesIO(raw), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            first_error = "encrypted_pdf"
        else:
            metadata_title = ""
            if reader.metadata is not None and isinstance(reader.metadata.title, str):
                metadata_title = reader.metadata.title
            pages = [_normalize_text(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(page for page in pages if page)
            if text:
                return ExtractionResult(
                    "pypdf-text-v1",
                    (
                        RawDocument(
                            _normalize_text(metadata_title) or fallback_title,
                            text,
                            family_key,
                            "document",
                            True,
                        ),
                    ),
                    "extracted",
                )
    except Exception as exc:
        first_error = f"pypdf_{type(exc).__name__}"

    second_error = "no_extractable_text"
    try:
        import pdfplumber

        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        with pdfplumber.open(BytesIO(raw)) as document:
            metadata_title = document.metadata.get("Title", "") if document.metadata else ""
            pages = [_normalize_text(page.extract_text() or "") for page in document.pages]
        text = "\n\n".join(page for page in pages if page)
        if text:
            return ExtractionResult(
                "pdfplumber-fallback-v1",
                (
                    RawDocument(
                        _normalize_text(str(metadata_title)) or fallback_title,
                        text,
                        family_key,
                        "document",
                        True,
                    ),
                ),
                "extracted_fallback",
            )
    except Exception as exc:
        second_error = f"pdfplumber_{type(exc).__name__}"

    return ExtractionResult(
        "pdf-dual-extractor-v1",
        (),
        f"parse_error:{first_error}+{second_error}",
    )


def _presentation_member_order(name: str) -> tuple[int, int, str]:
    slide_match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
    if slide_match:
        return int(slide_match.group(1)), 0, name
    notes_match = re.fullmatch(r"ppt/notesSlides/notesSlide(\d+)\.xml", name)
    if notes_match:
        return int(notes_match.group(1)), 1, name
    return 2**31 - 1, 2, name


def _xml_text_runs(raw: bytes) -> str:
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("OOXML member contains a forbidden declaration")
    root = ElementTree.fromstring(raw)
    values = [
        _normalize_text(node.text)
        for node in root.iter()
        if node.tag.endswith("}t") and isinstance(node.text, str) and node.text.strip()
    ]
    return _normalize_text("\n".join(value for value in values if value))


def _extract_presentation_package(
    raw: bytes, family_key: str, fallback_title: str
) -> ExtractionResult:
    try:
        with ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            if not infos:
                return ExtractionResult("pptx-ooxml-text-v1", (), "empty_presentation_package")
            seen: set[str] = set()
            selected: list[ZipInfo] = []
            core: ZipInfo | None = None
            selected_bytes = 0
            for info in infos:
                if info.filename in seen:
                    raise ValueError("presentation package repeats a member name")
                seen.add(info.filename)
                if not _zip_member_is_safe(info):
                    raise ValueError("presentation package contains an unsafe member")
                if info.file_size > MAXIMUM_ZIP_MEMBER_BYTES:
                    raise ValueError("presentation member exceeds its size bound")
                if re.fullmatch(
                    r"ppt/(?:slides/slide|notesSlides/notesSlide)\d+\.xml", info.filename
                ):
                    selected.append(info)
                    selected_bytes += info.file_size
                elif info.filename == "docProps/core.xml":
                    core = info
                    selected_bytes += info.file_size
            if selected_bytes > MAXIMUM_SELECTED_ZIP_BYTES:
                raise ValueError("presentation text members exceed their aggregate size bound")
            title = fallback_title
            if core is not None:
                core_raw = archive.read(core)
                if b"<!DOCTYPE" in core_raw.upper() or b"<!ENTITY" in core_raw.upper():
                    raise ValueError("presentation properties contain a forbidden declaration")
                core_root = ElementTree.fromstring(core_raw)
                for node in core_root.iter():
                    if (
                        node.tag.endswith("}title")
                        and isinstance(node.text, str)
                        and node.text.strip()
                    ):
                        title = _normalize_text(node.text)
                        break
            sections: list[str] = []
            seen_text: set[str] = set()
            for info in sorted(
                selected, key=lambda item: _presentation_member_order(item.filename)
            ):
                text = _xml_text_runs(archive.read(info))
                if text and text not in seen_text:
                    seen_text.add(text)
                    sections.append(text)
    except (BadZipFile, ElementTree.ParseError, ValueError) as exc:
        return ExtractionResult("pptx-ooxml-text-v1", (), f"parse_error:{type(exc).__name__}")
    text = "\n\n".join(sections)
    if not text:
        return ExtractionResult("pptx-ooxml-text-v1", (), "no_extractable_ooxml_text")
    return ExtractionResult(
        "pptx-ooxml-text-v1",
        (RawDocument(title, text, family_key, "presentation", True),),
        "extracted_fallback",
    )


def _zip_member_is_safe(info: ZipInfo) -> bool:
    member = PurePosixPath(info.filename)
    return (
        bool(info.filename)
        and "\\" not in info.filename
        and "\x00" not in info.filename
        and not member.is_absolute()
        and ".." not in member.parts
        and not _blocked(member.parts)
    )


def _onet_rows(value: object, member: str) -> list[Mapping[str, object]]:
    if not isinstance(value, Mapping) or not isinstance(value.get("row"), list):
        raise ValueError(f"O*NET member {member!r} has an invalid table envelope")
    rows = value["row"]
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"O*NET member {member!r} contains a non-object row")
    return rows  # type: ignore[return-value]


def _extract_onet_zip(raw: bytes, artifact_family: str) -> ExtractionResult:
    container_rows: list[dict[str, object]] = []
    container_counts: Counter[str] = Counter()
    try:
        with ZipFile(BytesIO(raw)) as archive:
            infos = archive.infolist()
            names: set[str] = set()
            selected_infos: dict[str, ZipInfo] = {}
            selected_payloads: dict[str, object] = {}
            selected_uncompressed = 0
            for info in infos:
                if info.filename in names:
                    raise ValueError("ZIP contains duplicate member names")
                names.add(info.filename)
                if not _zip_member_is_safe(info):
                    container_counts["blocked_before_open"] += 1
                    continue
                if info.is_dir():
                    status = "directory"
                elif info.filename not in _ONET_SELECTED_MEMBERS:
                    status = "quarantined_unselected_table"
                else:
                    if info.file_size > MAXIMUM_ZIP_MEMBER_BYTES:
                        raise ValueError("selected ZIP member exceeds its size bound")
                    selected_uncompressed += info.file_size
                    if selected_uncompressed > MAXIMUM_SELECTED_ZIP_BYTES:
                        raise ValueError("selected ZIP members exceed their aggregate size bound")
                    selected_infos[info.filename] = info
                    status = "selected"
                container_counts[status] += 1
                container_rows.append(
                    {
                        "schema_version": CONTAINER_PROVENANCE_SCHEMA,
                        "container_family_sha256": hashlib.sha256(
                            artifact_family.encode("utf-8")
                        ).hexdigest(),
                        "member": info.filename,
                        "status": status,
                        "uncompressed_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                    }
                )
            missing = set(_ONET_SELECTED_MEMBERS).difference(selected_infos)
            if missing:
                raise ValueError(f"O*NET ZIP is missing selected tables: {sorted(missing)!r}")
            for member in _ONET_SELECTED_MEMBERS:
                selected_payloads[member] = _strict_json(
                    archive.read(selected_infos[member]), f"O*NET member {member}"
                )
    except (BadZipFile, ValueError, json.JSONDecodeError) as exc:
        return ExtractionResult(
            "onet-json-join-v1",
            (),
            f"parse_error:{type(exc).__name__}",
            tuple(container_rows),
            dict(container_counts),
        )

    occupation_rows = _onet_rows(
        selected_payloads["db_31_0_json/occupation_data.json"], "occupation_data"
    )
    occupations: dict[str, tuple[str, str]] = {}
    for row in occupation_rows:
        code = row.get("onetsoc_code")
        title = row.get("title")
        description = row.get("description")
        if (
            isinstance(code, str)
            and code[:2] in _ONET_BUSINESS_MAJOR_GROUPS
            and isinstance(title, str)
            and isinstance(description, str)
        ):
            occupations[code] = (title, description)

    grouped: dict[str, dict[str, set[str]]] = {
        code: {
            "Tasks": set(),
            "Work activities": set(),
            "Knowledge": set(),
            "Skills": set(),
            "Work styles": set(),
        }
        for code in occupations
    }
    table_fields = (
        ("db_31_0_json/task_statements.json", "Tasks", "task"),
        ("db_31_0_json/work_activities.json", "Work activities", "element_name"),
        ("db_31_0_json/knowledge.json", "Knowledge", "element_name"),
        ("db_31_0_json/essential_skills.json", "Skills", "element_name"),
        ("db_31_0_json/work_styles.json", "Work styles", "element_name"),
    )
    for member, section, field_name in table_fields:
        for row in _onet_rows(selected_payloads[member], member):
            code = row.get("onetsoc_code")
            value = row.get(field_name)
            if (
                isinstance(code, str)
                and code in grouped
                and isinstance(value, str)
                and value.strip()
            ):
                grouped[code][section].add(_normalize_text(value))

    documents: list[RawDocument] = []
    for code in sorted(occupations):
        title, description = occupations[code]
        parts = [f"Occupation: {title}", f"Description: {description}"]
        for section in ("Tasks", "Work activities", "Knowledge", "Skills", "Work styles"):
            values = sorted(grouped[code][section])
            if values:
                parts.append(section + ":\n" + "\n".join(f"- {value}" for value in values))
        documents.append(
            RawDocument(
                f"{title} — O*NET role profile",
                "\n\n".join(parts),
                f"{artifact_family}:occupation:{code}",
                f"occupation:{code}",
                True,
            )
        )
    if not documents:
        return ExtractionResult(
            "onet-json-join-v1",
            (),
            "no_business_occupations",
            tuple(container_rows),
            dict(container_counts),
        )
    return ExtractionResult(
        "onet-json-join-v1",
        tuple(documents),
        "extracted",
        tuple(container_rows),
        dict(container_counts),
    )


def _eligible_artifact(source_id: str, relative_path: str) -> tuple[bool, str]:
    suffix = PurePosixPath(relative_path).suffix.casefold()
    if source_id == "gitlab-company-handbook":
        return (
            relative_path.startswith("content/handbook/") and suffix == ".md",
            "outside_handbook_markdown",
        )
    if source_id == "posthog-handbook-and-manual":
        in_reference_tree = relative_path.startswith(
            "contents/founders/"
        ) or relative_path.startswith("contents/handbook/")
        return in_reference_tree and suffix in {".md", ".mdx"}, "outside_handbook_or_founders"
    if source_id == "fdic-sba-money-smart-small-business":
        return suffix in {".pdf", ".html", ".csv", ".txt"}, "unsupported_fdic_artifact"
    if source_id == "sba-business-guide-and-content-api":
        return suffix == ".html", "unsupported_sba_artifact"
    if source_id == "yc-startup-library":
        return suffix == ".html", "non_article_or_unsupported_yc_artifact"
    if source_id == "onet-business-work-activities":
        return relative_path == "db_31_0_json.zip", "unsupported_onet_artifact"
    raise AssertionError(source_id)


def _extract_artifact(pin: PinnedArtifact, raw: bytes) -> ExtractionResult:
    eligible, reason = _eligible_artifact(pin.source_id, pin.relative_path)
    if not eligible:
        return ExtractionResult("none", (), reason)
    family = f"{pin.source_id}:{pin.relative_path}"
    fallback_title = PurePosixPath(pin.relative_path).stem.replace("-", " ").replace("_", " ")
    suffix = PurePosixPath(pin.relative_path).suffix.casefold()
    if pin.source_id == "onet-business-work-activities":
        return _extract_onet_zip(raw, family)
    if pin.source_id == "yc-startup-library":
        return _extract_yc_html(raw, family, fallback_title)
    if suffix == ".html":
        return _extract_generic_html(raw, family, fallback_title)
    if suffix in {".md", ".mdx"}:
        return _extract_markdown(raw, family, fallback_title)
    if suffix == ".pdf":
        return _extract_pdf(raw, family, fallback_title)
    if suffix == ".csv":
        return _extract_csv(raw, family, fallback_title)
    if suffix == ".txt":
        return _extract_text(raw, family, fallback_title)
    return ExtractionResult("none", (), "unsupported_artifact")


def _apply_pattern(
    text: str,
    pattern: re.Pattern[str],
    replacement: str | Any,
) -> tuple[str, int]:
    return pattern.subn(replacement, text)


def _redact(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    patterns: tuple[tuple[str, re.Pattern[str], str | Any], ...] = (
        ("private_key", _PRIVATE_KEY_RE, "[REDACTED_SECRET]"),
        (
            "secret_assignment",
            _SECRET_ASSIGNMENT_RE,
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED_SECRET]",
        ),
        ("service_token", _SERVICE_TOKEN_RE, "[REDACTED_SECRET]"),
        ("aws_access_key", _AWS_KEY_RE, "[REDACTED_SECRET]"),
        ("jwt", _JWT_RE, "[REDACTED_SECRET]"),
        ("email", _EMAIL_RE, "[REDACTED_EMAIL]"),
        ("phone", _PHONE_RE, "[REDACTED_PHONE]"),
        ("ssn", _SSN_RE, "[REDACTED_SSN]"),
        ("ipv4", _IPV4_RE, "[REDACTED_IP]"),
    )
    for name, pattern, replacement in patterns:
        text, count = _apply_pattern(text, pattern, replacement)
        counts[name] += count
    return text, dict(sorted(counts.items()))


def _contains_unredacted_sensitive_pattern(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (
            _PRIVATE_KEY_RE,
            _SECRET_ASSIGNMENT_RE,
            _SERVICE_TOKEN_RE,
            _AWS_KEY_RE,
            _JWT_RE,
            _EMAIL_RE,
            _PHONE_RE,
            _SSN_RE,
            _IPV4_RE,
        )
    )


def _hard_split(paragraph: str, maximum: int) -> list[str]:
    chunks: list[str] = []
    remaining = paragraph
    while len(remaining) > maximum:
        split_at = remaining.rfind(" ", 0, maximum + 1)
        if split_at < maximum // 2:
            split_at = maximum
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _chunk_text(text: str, config: BuildConfig) -> tuple[str, ...]:
    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraphs.extend(_hard_split(paragraph, config.maximum_characters))
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        separator = 2 if current else 0
        if current and current_size + separator + len(paragraph) > config.maximum_characters:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0
            separator = 0
        current.append(paragraph)
        current_size += separator + len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    if len(chunks) > 1 and len(chunks[-1]) < config.minimum_characters:
        tail = chunks.pop()
        combined = chunks.pop() + "\n\n" + tail
        if len(combined) <= config.maximum_characters:
            chunks.append(combined)
        else:
            midpoint = len(combined) // 2
            split_at = combined.rfind(" ", config.minimum_characters, midpoint + 1)
            if split_at < config.minimum_characters:
                split_at = combined.find(
                    " ", midpoint, len(combined) - config.minimum_characters + 1
                )
            if split_at < config.minimum_characters:
                split_at = midpoint
            chunks.extend((combined[:split_at].strip(), combined[split_at:].strip()))
    return tuple(chunk for chunk in chunks if len(chunk) >= config.minimum_characters)


def _term_count(text: str, term: str) -> int:
    pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"(?!\w)"
    return min(len(re.findall(pattern, text, flags=re.IGNORECASE)), 5)


def _classify_leaf(title: str, text: str, path_hint: str) -> tuple[str, dict[str, int]]:
    emphasized = f"{title}\n{path_hint.replace('-', ' ').replace('_', ' ')}"
    scores: dict[str, int] = {}
    for leaf, weighted_terms in _LEAF_TERMS.items():
        score = 0
        for term, weight in weighted_terms.items():
            score += weight * (_term_count(text, term) + 3 * _term_count(emphasized, term))
        scores[leaf] = score
    maximum = max(scores.values())
    if maximum == 0:
        return "business-core", scores
    for leaf in _LEAF_TIE_ORDER:
        if scores[leaf] == maximum:
            return leaf, scores
    raise AssertionError("unreachable leaf tie")


def _business_relevant(source_id: str, scores: Mapping[str, int], title: str, text: str) -> bool:
    if source_id in TRUSTED_BUSINESS_SOURCES:
        return True
    if max(scores.values()) > 0:
        return True
    generic = (
        "company",
        "team",
        "customer",
        "employee",
        "organization",
        "strategy",
        "process",
        "product",
        "manager",
    )
    sample = f"{title}\n{text}".casefold()
    return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", sample) for term in generic)


def _family_split(source_id: str, family_key: str) -> tuple[str, str]:
    family_id = hashlib.sha256(f"{source_id}\0{family_key}".encode("utf-8")).hexdigest()
    bucket = int(family_id[:8], 16) % 10_000
    split = "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"
    return family_id, split


def _simhash64(text: str) -> int:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    counts = Counter(tokens)
    vector = [0] * 64
    for token, frequency in counts.items():
        weight = min(frequency, 3)
        token_hash = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += weight if token_hash & (1 << bit) else -weight
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def _near_bands(simhash: int) -> tuple[tuple[int, int], ...]:
    return tuple((band, (simhash >> (band * 16)) & 0xFFFF) for band in range(4))


def _near_owner(
    simhash: int,
    threshold: int,
    bands: Mapping[tuple[int, int], list[str]],
    owners: Mapping[str, int],
) -> tuple[str | None, int | None]:
    candidates: set[str] = set()
    for band in _near_bands(simhash):
        candidates.update(bands.get(band, ()))
    winner: tuple[int, str] | None = None
    for document_id in candidates:
        distance = (simhash ^ owners[document_id]).bit_count()
        if distance <= threshold:
            candidate = (distance, document_id)
            if winner is None or candidate < winner:
                winner = candidate
    return (None, None) if winner is None else (winner[1], winner[0])


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> tuple[str, int, int]:
    accumulator = _ShardAccumulator(path)
    try:
        for row in rows:
            accumulator.add(row)
    finally:
        accumulator.close()
    return accumulator.sha256, accumulator.bytes, accumulator.records


def _artifact_parser_status(extraction: ExtractionResult, emitted: int, candidates: int) -> str:
    if not extraction.documents:
        if extraction.reason.startswith("parse_error"):
            return "rejected"
        return "quarantined"
    if candidates == 0:
        return "quarantined"
    return "emitted" if emitted else "duplicate_or_quarantined_only"


def build_business_reference_corpus(
    *,
    snapshot_root: str | Path,
    receipt_path: str | Path,
    output_directory: str | Path,
    config: BuildConfig = BuildConfig(),
    dry_run: bool = False,
) -> dict[str, object]:
    """Materialize six-leaf grounded references with exact provenance and accounting."""

    if not isinstance(config, BuildConfig):
        raise TypeError("config must be a BuildConfig")
    receipt = load_business_receipt(receipt_path)
    resolved = _resolve_artifacts(Path(snapshot_root), receipt)
    output = Path(output_directory).expanduser()
    stage: Path | None = None
    if not dry_run:
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))

    shard_writers: dict[tuple[str, str], _ShardAccumulator] = {}
    for leaf in LEAVES:
        for split in SPLITS:
            destination = None if stage is None else stage / "documents" / leaf / f"{split}.jsonl"
            shard_writers[(leaf, split)] = _ShardAccumulator(destination)

    artifact_rows: list[dict[str, object]] = []
    document_rows: list[dict[str, object]] = []
    container_rows: list[dict[str, object]] = []
    artifact_statuses: Counter[str] = Counter()
    artifact_reasons: Counter[str] = Counter()
    artifact_statuses_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    parser_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    document_statuses: Counter[str] = Counter()
    document_statuses_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    document_statuses_by_leaf: dict[str, Counter[str]] = defaultdict(Counter)
    redaction_totals: Counter[str] = Counter()
    container_totals: Counter[str] = Counter()
    exact_owners: dict[str, str] = {}
    simhash_owners: dict[str, int] = {}
    near_band_index: dict[tuple[int, int], list[str]] = defaultdict(list)
    family_splits: dict[str, str] = {}
    selected_artifact_count = 0

    try:
        for source_id in SELECTED_SOURCE_IDS:
            for pin, path in resolved[source_id]:
                selected_artifact_count += 1
                raw = _verified_bytes(path, pin)
                extraction = _extract_artifact(pin, raw)
                container_rows.extend(extraction.container_rows)
                container_totals.update(extraction.container_counts)
                artifact_counts: Counter[str] = Counter()
                artifact_redactions: Counter[str] = Counter()
                raw_sections = len(extraction.documents)
                short_sections = 0
                for raw_document in extraction.documents:
                    chunks = _chunk_text(_normalize_text(raw_document.text), config)
                    if not chunks:
                        short_sections += 1
                        continue
                    family_id, split = _family_split(source_id, raw_document.family_key)
                    prior_split = family_splits.setdefault(family_id, split)
                    if prior_split != split:
                        raise AssertionError("one artifact family crossed data splits")
                    for chunk_index, unredacted_chunk in enumerate(chunks):
                        title, title_redactions = _redact(_normalize_text(raw_document.title))
                        text, text_redactions = _redact(unredacted_chunk)
                        text = _normalize_text(text)
                        redactions = Counter(title_redactions)
                        redactions.update(text_redactions)
                        redaction_totals.update(redactions)
                        artifact_redactions.update(redactions)
                        if _contains_unredacted_sensitive_pattern(title + "\n" + text):
                            raise AssertionError("redaction verification left a sensitive pattern")
                        leaf, leaf_scores = _classify_leaf(title, text, pin.relative_path)
                        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        simhash = _simhash64(text)
                        status = "candidate"
                        reason = "eligible"
                        duplicate_of = None
                        near_distance = None
                        if not _business_relevant(source_id, leaf_scores, title, text):
                            status = "quarantined"
                            reason = "business_relevance_below_threshold"
                        elif content_sha in exact_owners:
                            status = "duplicate_exact"
                            reason = "normalized_text_exact_duplicate"
                            duplicate_of = exact_owners[content_sha]
                        else:
                            duplicate_of, near_distance = _near_owner(
                                simhash,
                                config.near_duplicate_distance,
                                near_band_index,
                                simhash_owners,
                            )
                            if duplicate_of is not None:
                                status = "duplicate_near"
                                reason = "simhash_near_duplicate"
                        if status == "candidate":
                            document_id = content_sha
                            exact_owners[content_sha] = document_id
                            simhash_owners[document_id] = simhash
                            for band in _near_bands(simhash):
                                near_band_index[band].append(document_id)
                            shard_writers[(leaf, split)].add(
                                {"document_id": document_id, "text": text, "title": title}
                            )
                            status = "emitted"
                            reason = "grounded_reference"
                        artifact_counts[status] += 1
                        document_statuses[status] += 1
                        document_statuses_by_source[source_id][status] += 1
                        document_statuses_by_leaf[leaf][status] += 1
                        document_rows.append(
                            {
                                "schema_version": DOCUMENT_PROVENANCE_SCHEMA,
                                "source_id": source_id,
                                "artifact_path": pin.relative_path,
                                "artifact_sha256": pin.sha256,
                                "family_sha256": family_id,
                                "internal_reference": raw_document.internal_reference,
                                "chunk_index": chunk_index,
                                "content_sha256": content_sha,
                                "leaf_id": leaf,
                                "leaf_scores": dict(sorted(leaf_scores.items())),
                                "split": split,
                                "status": status,
                                "reason": reason,
                                "duplicate_of": duplicate_of,
                                "near_duplicate_distance": near_distance,
                                "characters": len(text),
                                "redactions": dict(sorted(redactions.items())),
                            }
                        )
                candidates = sum(artifact_counts.values())
                emitted = artifact_counts["emitted"]
                artifact_status = _artifact_parser_status(extraction, emitted, candidates)
                artifact_statuses[artifact_status] += 1
                artifact_statuses_by_source[source_id][artifact_status] += 1
                parser_statuses[extraction.parser][artifact_status] += 1
                artifact_reason = extraction.reason if not candidates else "document_accounted"
                artifact_reasons[artifact_reason] += 1
                artifact_rows.append(
                    {
                        "schema_version": ARTIFACT_PROVENANCE_SCHEMA,
                        "source_id": source_id,
                        "artifact_path": pin.relative_path,
                        "artifact_sha256": pin.sha256,
                        "bytes": pin.byte_count,
                        "parser": extraction.parser,
                        "status": artifact_status,
                        "reason": artifact_reason,
                        "raw_sections": raw_sections,
                        "short_sections": short_sections,
                        "document_candidates": candidates,
                        "document_accounting": dict(sorted(artifact_counts.items())),
                        "redactions": dict(sorted(artifact_redactions.items())),
                    }
                )
        for writer in shard_writers.values():
            writer.close()

        artifact_input = sum(
            len(receipt.artifacts_by_source[source]) for source in SELECTED_SOURCE_IDS
        )
        if (
            selected_artifact_count != artifact_input
            or sum(artifact_statuses.values()) != artifact_input
        ):
            raise AssertionError("artifact accounting is incomplete")
        document_input = len(document_rows)
        if sum(document_statuses.values()) != document_input:
            raise AssertionError("document accounting is incomplete")
        if any(value == "candidate" for value in document_statuses):
            raise AssertionError("document accounting contains a non-terminal state")

        shard_manifest: list[dict[str, object]] = []
        for leaf in LEAVES:
            for split in SPLITS:
                writer = shard_writers[(leaf, split)]
                shard_manifest.append(
                    {
                        "leaf_id": leaf,
                        "leaf_revision": "1",
                        "split": split,
                        "path": f"documents/{leaf}/{split}.jsonl",
                        "sha256": writer.sha256,
                        "bytes": writer.bytes,
                        "records": writer.records,
                    }
                )

        artifact_sidecar = _sidecar_metadata(artifact_rows, stage, "provenance/artifacts.jsonl")
        document_sidecar = _sidecar_metadata(document_rows, stage, "provenance/documents.jsonl")
        container_sidecar = _sidecar_metadata(
            container_rows, stage, "provenance/container-members.jsonl"
        )
        manifest_content: dict[str, object] = {
            "schema_version": REFERENCE_SCHEMA,
            "revision": "1",
            "source_receipt": {
                "path": Path(receipt_path).name,
                "sha256": receipt.receipt_sha256,
                "snapshot_root_label": receipt.snapshot_root_label,
            },
            "selected_source_ids": list(SELECTED_SOURCE_IDS),
            "explicitly_prehandled_source_ids": list(receipt.excluded_source_ids),
            "source_order_is_dedup_precedence": True,
            "contains_reference_text": True,
            "contains_synthetic_sft": False,
            "license_or_rights_metadata_used_for_admission": False,
            "leaves": [{"leaf_id": leaf, "leaf_revision": "1"} for leaf in LEAVES],
            "split_policy": {
                "name": "artifact_family_sha256_90_5_5",
                "train_buckets": "0000-8999",
                "validation_buckets": "9000-9499",
                "test_buckets": "9500-9999",
                "family_leakage_count": 0,
            },
            "deduplication": {
                "exact": "sha256_of_normalized_redacted_text",
                "near": "64_bit_token_frequency_simhash_four_16_bit_bands",
                "maximum_hamming_distance": config.near_duplicate_distance,
            },
            "redaction_policy": {
                "applied_before_emission_and_deduplication": True,
                "classes": [
                    "private_key",
                    "secret_assignment",
                    "service_token",
                    "aws_access_key",
                    "jwt",
                    "email",
                    "phone",
                    "ssn",
                    "ipv4",
                ],
                "replacement_counts": dict(sorted(redaction_totals.items())),
            },
            "chunking": {
                "minimum_characters": config.minimum_characters,
                "maximum_characters": config.maximum_characters,
                "overlap_characters": 0,
            },
            "artifact_accounting": {
                "input": artifact_input,
                "statuses": dict(sorted(artifact_statuses.items())),
                "reasons": dict(sorted(artifact_reasons.items())),
                "by_source": {
                    source_id: dict(sorted(artifact_statuses_by_source[source_id].items()))
                    for source_id in SELECTED_SOURCE_IDS
                },
                "by_parser": {
                    parser_name: dict(sorted(statuses.items()))
                    for parser_name, statuses in sorted(parser_statuses.items())
                },
                "zero_silent_drops": sum(artifact_statuses.values()) == artifact_input,
            },
            "document_accounting": {
                "input": document_input,
                "statuses": dict(sorted(document_statuses.items())),
                "by_source": {
                    source_id: dict(sorted(document_statuses_by_source[source_id].items()))
                    for source_id in SELECTED_SOURCE_IDS
                },
                "by_leaf": {
                    leaf: dict(sorted(document_statuses_by_leaf[leaf].items())) for leaf in LEAVES
                },
                "zero_silent_drops": sum(document_statuses.values()) == document_input,
            },
            "container_member_accounting": {
                "input": len(container_rows) + int(container_totals.get("blocked_before_open", 0)),
                "statuses": dict(sorted(container_totals.items())),
                "zero_silent_drops": sum(container_totals.values())
                == len(container_rows) + int(container_totals.get("blocked_before_open", 0)),
            },
            "shards": shard_manifest,
            "provenance_sidecars": {
                "artifacts": artifact_sidecar,
                "documents": document_sidecar,
                "container_members": container_sidecar,
            },
            "dry_run": dry_run,
        }
        manifest = {**manifest_content, "content_sha256": fingerprint_json(manifest_content)}
        if stage is not None:
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            (stage / "manifest.json").write_bytes(manifest_bytes)
            checksums = {item["path"]: item["sha256"] for item in shard_manifest}
            checksums.update(
                {
                    artifact_sidecar["path"]: artifact_sidecar["sha256"],
                    document_sidecar["path"]: document_sidecar["sha256"],
                    container_sidecar["path"]: container_sidecar["sha256"],
                    "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
                }
            )
            sums = "".join(
                f"{digest}  {path}\n" for path, digest in sorted(checksums.items())
            ).encode("utf-8")
            (stage / "sha256sums.txt").write_bytes(sums)
            os.replace(stage, output)
            stage = None
        return manifest
    finally:
        for writer in shard_writers.values():
            writer.close()
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def _sidecar_metadata(
    rows: Sequence[Mapping[str, object]], stage: Path | None, relative_path: str
) -> dict[str, object]:
    if stage is None:
        accumulator = _ShardAccumulator(None)
        for row in rows:
            accumulator.add(row)
        accumulator.close()
        sha256, byte_count, records = accumulator.sha256, accumulator.bytes, accumulator.records
    else:
        sha256, byte_count, records = _write_jsonl(stage / relative_path, rows)
    return {"path": relative_path, "sha256": sha256, "bytes": byte_count, "records": records}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-characters", type=int, default=DEFAULT_MINIMUM_CHARACTERS)
    parser.add_argument("--maximum-characters", type=int, default=DEFAULT_MAXIMUM_CHARACTERS)
    parser.add_argument(
        "--near-duplicate-distance", type=int, default=DEFAULT_NEAR_DUPLICATE_DISTANCE
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_business_reference_corpus(
        snapshot_root=args.snapshot_root,
        receipt_path=args.receipt,
        output_directory=args.output,
        config=BuildConfig(
            minimum_characters=args.minimum_characters,
            maximum_characters=args.maximum_characters,
            near_duplicate_distance=args.near_duplicate_distance,
        ),
        dry_run=args.dry_run,
    )
    aggregate = {
        "schema_version": manifest["schema_version"],
        "content_sha256": manifest["content_sha256"],
        "artifact_accounting": manifest["artifact_accounting"],
        "document_accounting": manifest["document_accounting"],
        "container_member_accounting": manifest["container_member_accounting"],
        "redaction_policy": manifest["redaction_policy"],
        "shards": manifest["shards"],
        "dry_run": manifest["dry_run"],
    }
    print(json.dumps(aggregate, sort_keys=True))
    return 0


__all__ = [
    "ACQUISITION_SCHEMA",
    "ARTIFACT_PROVENANCE_SCHEMA",
    "BuildConfig",
    "BusinessReceipt",
    "CONTAINER_PROVENANCE_SCHEMA",
    "DOCUMENT_PROVENANCE_SCHEMA",
    "LEAVES",
    "PREHANDLED_SOURCE_IDS",
    "REFERENCE_SCHEMA",
    "SELECTED_SOURCE_IDS",
    "SPLITS",
    "build_business_reference_corpus",
    "load_business_receipt",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
