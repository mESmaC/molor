"""Deterministic, content-addressed mixtures of messages-only corpus shards.

This module is a data-plane companion to :mod:`mixture_presets`.  Presets describe
inference-time LoRA composition; this module builds immutable JSONL datasets for one
training or evaluation split.  Source text never appears in reports or provenance
ledgers, and every source record receives exactly one terminal accounting status.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from typing import Any, ClassVar, Literal

from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256
from molor.training.data import ChatExample, ToolTraceError


CORPUS_MIXTURE_RECIPE_SCHEMA = "molor.corpus-mixture-recipe.v1"
CORPUS_MIXTURE_PROVENANCE_SCHEMA = "molor.corpus-mixture-provenance.v1"
CORPUS_MIXTURE_REPORT_SCHEMA = "molor.corpus-mixture-report.v1"
CORPUS_MIXTURE_CONTENT_SCHEMA = "molor.corpus-mixture-content.v1"
DEFAULT_MAXIMUM_RECIPE_BYTES = 4 * 1024 * 1024
DEFAULT_MAXIMUM_RECORD_BYTES = 16 * 1024 * 1024
TERMINAL_STATUSES = ("emitted", "duplicate", "quarantined", "rejected", "capped")

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TEMPLATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SPLITS = frozenset(("train", "validation", "test"))
_USAGES = frozenset(("training", "evaluation"))
_SHORTFALL_POLICIES = frozenset(("error", "emit_available"))


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 1-128 characters using letters, digits, '.', '_', ':', '/', "
            "or '-'"
        )
    return value


def _revision(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 1-128 characters using letters, digits, '.', '_', or '-'"
        )
    return value


def _split(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value not in _SPLITS:
        raise ValueError(f"{field_name} must be one of {sorted(_SPLITS)!r}")
    return value


def _template_mode(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _TEMPLATE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase versioned template identifier")
    return value


def _relative_or_absolute_path(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be a non-empty filesystem path")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, field_name)


def _positive_weight(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return weight


def _lowercase_sha256(value: object, field_name: str) -> str:
    digest = validate_sha256(value, field_name)  # type: ignore[arg-type]
    if value != digest:
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    return digest


@dataclass(frozen=True, slots=True)
class CorpusShardDescriptor:
    """One immutable, versioned messages-only JSONL shard."""

    shard_id: str
    shard_revision: str
    leaf_id: str
    leaf_revision: str
    split: str
    template_mode: str
    path: str
    sha256: str
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_id", _identifier(self.shard_id, "shard_id"))
        object.__setattr__(self, "shard_revision", _revision(self.shard_revision, "shard_revision"))
        object.__setattr__(self, "leaf_id", _identifier(self.leaf_id, "leaf_id"))
        object.__setattr__(self, "leaf_revision", _revision(self.leaf_revision, "leaf_revision"))
        object.__setattr__(self, "split", _split(self.split, "split"))
        object.__setattr__(
            self, "template_mode", _template_mode(self.template_mode, "template_mode")
        )
        object.__setattr__(self, "path", _relative_or_absolute_path(self.path, "path"))
        object.__setattr__(self, "sha256", _lowercase_sha256(self.sha256, "sha256"))
        object.__setattr__(
            self, "record_count", _positive_integer(self.record_count, "record_count")
        )

    @property
    def versioned_id(self) -> str:
        return f"{self.shard_id}@{self.shard_revision}"

    @property
    def versioned_leaf_id(self) -> str:
        return f"{self.leaf_id}@{self.leaf_revision}"

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "shard_revision": self.shard_revision,
            "leaf_id": self.leaf_id,
            "leaf_revision": self.leaf_revision,
            "split": self.split,
            "template_mode": self.template_mode,
            "path": self.path,
            "sha256": self.sha256,
            "record_count": self.record_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], field_name: str) -> CorpusShardDescriptor:
        expected = {
            "shard_id",
            "shard_revision",
            "leaf_id",
            "leaf_revision",
            "split",
            "template_mode",
            "path",
            "sha256",
            "record_count",
        }
        _require_exact_keys(value, expected, field_name)
        return cls(**{key: value[key] for key in expected})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CorpusMixtureShard:
    """A source shard plus its adjustable sampling weight and optional hard cap."""

    descriptor: CorpusShardDescriptor
    weight: float
    cap: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CorpusShardDescriptor):
            raise TypeError("descriptor must be a CorpusShardDescriptor")
        object.__setattr__(
            self,
            "weight",
            _positive_weight(self.weight, f"weight[{self.descriptor.shard_id!r}]"),
        )
        object.__setattr__(
            self,
            "cap",
            _optional_positive_integer(self.cap, f"cap[{self.descriptor.shard_id!r}]"),
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.descriptor.to_dict(), "weight": self.weight, "cap": self.cap}

    @classmethod
    def from_dict(cls, value: Mapping[str, object], field_name: str) -> CorpusMixtureShard:
        expected = {
            "shard_id",
            "shard_revision",
            "leaf_id",
            "leaf_revision",
            "split",
            "template_mode",
            "path",
            "sha256",
            "record_count",
            "weight",
            "cap",
        }
        _require_exact_keys(value, expected, field_name)
        descriptor_values = {key: value[key] for key in expected if key not in {"weight", "cap"}}
        return cls(
            descriptor=CorpusShardDescriptor.from_dict(
                descriptor_values, f"{field_name}.descriptor"
            ),
            weight=value["weight"],  # type: ignore[arg-type]
            cap=value["cap"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CorpusMixtureRecipe:
    """One exact data mixture for a single split and chat-template mode."""

    schema_version: ClassVar[str] = CORPUS_MIXTURE_RECIPE_SCHEMA

    recipe_id: str
    revision: str
    usage: Literal["training", "evaluation"] | str
    output_split: Literal["train", "validation", "test"] | str
    template_mode: str
    seed: str
    target_records: int
    shortfall_policy: Literal["error", "emit_available"] | str
    shards: tuple[CorpusMixtureShard, ...]
    leakage_guard_shards: tuple[CorpusShardDescriptor, ...] = ()
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        recipe_id = _identifier(self.recipe_id, "recipe_id")
        revision = _revision(self.revision, "revision")
        if self.usage not in _USAGES:
            raise ValueError(f"usage must be one of {sorted(_USAGES)!r}")
        output_split = _split(self.output_split, "output_split")
        template_mode = _template_mode(self.template_mode, "template_mode")
        if not isinstance(self.seed, str) or not self.seed or len(self.seed) > 256:
            raise ValueError("seed must be a non-empty string of at most 256 characters")
        target_records = _positive_integer(self.target_records, "target_records")
        if self.shortfall_policy not in _SHORTFALL_POLICIES:
            raise ValueError(f"shortfall_policy must be one of {sorted(_SHORTFALL_POLICIES)!r}")
        if isinstance(self.shards, (str, bytes)) or not isinstance(self.shards, Sequence):
            raise TypeError("shards must be a sequence of CorpusMixtureShard values")
        shards = tuple(self.shards)
        if not shards or any(not isinstance(shard, CorpusMixtureShard) for shard in shards):
            raise ValueError("shards must contain at least one CorpusMixtureShard")
        if isinstance(self.leakage_guard_shards, (str, bytes)) or not isinstance(
            self.leakage_guard_shards, Sequence
        ):
            raise TypeError("leakage_guard_shards must be a sequence")
        guards = tuple(self.leakage_guard_shards)
        if any(not isinstance(shard, CorpusShardDescriptor) for shard in guards):
            raise TypeError("leakage_guard_shards must contain CorpusShardDescriptor values")

        shard_ids = [shard.descriptor.shard_id for shard in shards]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("mixture shard IDs must be unique within a recipe")
        guard_ids = [shard.shard_id for shard in guards]
        if len(guard_ids) != len(set(guard_ids)):
            raise ValueError("leakage-guard shard IDs must be unique within a recipe")
        if set(shard_ids).intersection(guard_ids):
            raise ValueError("mixture and leakage-guard shard IDs must be disjoint")

        leaf_revisions: dict[str, str] = {}
        for shard in (item.descriptor for item in shards):
            prior = leaf_revisions.setdefault(shard.leaf_id, shard.leaf_revision)
            if prior != shard.leaf_revision:
                raise ValueError(f"leaf {shard.leaf_id!r} has multiple revisions in one recipe")
            if shard.split != output_split:
                raise ValueError("every mixture shard split must equal output_split")
            if shard.template_mode != template_mode:
                raise ValueError("every mixture shard template_mode must equal the recipe mode")

        if self.usage == "training":
            if output_split != "train":
                raise ValueError("training recipes must emit the train split")
            if any(shard.split == "train" for shard in guards):
                raise ValueError("training leakage guards must use validation or test shards")
        else:
            if output_split not in {"validation", "test"}:
                raise ValueError("evaluation recipes must emit validation or test")
            if not guards:
                raise ValueError("evaluation recipes require at least one pinned train guard shard")
            if any(shard.split != "train" for shard in guards):
                raise ValueError("evaluation leakage guards must all use the train split")

        canonical_shards = tuple(
            sorted(
                shards,
                key=lambda shard: (shard.descriptor.shard_id, shard.descriptor.shard_revision),
            )
        )
        canonical_guards = tuple(
            sorted(guards, key=lambda shard: (shard.shard_id, shard.shard_revision))
        )
        object.__setattr__(self, "recipe_id", recipe_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "output_split", output_split)
        object.__setattr__(self, "template_mode", template_mode)
        object.__setattr__(self, "target_records", target_records)
        object.__setattr__(self, "shards", canonical_shards)
        object.__setattr__(self, "leakage_guard_shards", canonical_guards)
        object.__setattr__(self, "_fingerprint", fingerprint_json(self.content_dict()))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "revision": self.revision,
            "usage": self.usage,
            "output_split": self.output_split,
            "template_mode": self.template_mode,
            "seed": self.seed,
            "target_records": self.target_records,
            "shortfall_policy": self.shortfall_policy,
            "shards": [shard.to_dict() for shard in self.shards],
            "leakage_guard_shards": [shard.to_dict() for shard in self.leakage_guard_shards],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_dict(), "content_sha256": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CorpusMixtureRecipe:
        expected = {
            "schema_version",
            "recipe_id",
            "revision",
            "usage",
            "output_split",
            "template_mode",
            "seed",
            "target_records",
            "shortfall_policy",
            "shards",
            "leakage_guard_shards",
            "content_sha256",
        }
        _require_exact_keys(value, expected, "recipe")
        if value["schema_version"] != CORPUS_MIXTURE_RECIPE_SCHEMA:
            raise ValueError("unsupported corpus-mixture recipe schema")
        raw_shards = value["shards"]
        raw_guards = value["leakage_guard_shards"]
        if not isinstance(raw_shards, list) or not isinstance(raw_guards, list):
            raise TypeError("shards and leakage_guard_shards must be arrays")
        recipe = cls(
            recipe_id=value["recipe_id"],  # type: ignore[arg-type]
            revision=value["revision"],  # type: ignore[arg-type]
            usage=value["usage"],  # type: ignore[arg-type]
            output_split=value["output_split"],  # type: ignore[arg-type]
            template_mode=value["template_mode"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            target_records=value["target_records"],  # type: ignore[arg-type]
            shortfall_policy=value["shortfall_policy"],  # type: ignore[arg-type]
            shards=tuple(
                CorpusMixtureShard.from_dict(item, f"shards[{index}]")
                if isinstance(item, Mapping)
                else _raise_type(f"shards[{index}] must be an object")
                for index, item in enumerate(raw_shards)
            ),
            leakage_guard_shards=tuple(
                CorpusShardDescriptor.from_dict(item, f"leakage_guard_shards[{index}]")
                if isinstance(item, Mapping)
                else _raise_type(f"leakage_guard_shards[{index}] must be an object")
                for index, item in enumerate(raw_guards)
            ),
        )
        if raw_shards != recipe.content_dict()["shards"]:
            raise ValueError("mixture shards must be ordered canonically by versioned shard ID")
        if raw_guards != recipe.content_dict()["leakage_guard_shards"]:
            raise ValueError("leakage guards must be ordered canonically by versioned shard ID")
        claimed = _lowercase_sha256(value["content_sha256"], "content_sha256")
        if claimed != recipe.fingerprint:
            raise ValueError("recipe content_sha256 does not match its semantic payload")
        return recipe


def _raise_type(message: str) -> Any:
    raise TypeError(message)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        raise ValueError(
            f"{field_name} fields differ from v1 (missing={missing}, unknown={unknown})"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"JSON contains unsupported numeric constant {value!r}")


def corpus_mixture_recipe_json_bytes(recipe: CorpusMixtureRecipe) -> bytes:
    if not isinstance(recipe, CorpusMixtureRecipe):
        raise TypeError("recipe must be a CorpusMixtureRecipe")
    return (json.dumps(recipe.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def load_corpus_mixture_recipe(
    path: str | Path, *, maximum_bytes: int = DEFAULT_MAXIMUM_RECIPE_BYTES
) -> CorpusMixtureRecipe:
    maximum_bytes = _positive_integer(maximum_bytes, "maximum_bytes")
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"recipe must be a regular non-symlink file: {source}")
    before = source.stat()
    if before.st_size < 1 or before.st_size > maximum_bytes:
        raise ValueError(f"recipe size must be between 1 and {maximum_bytes} bytes")
    raw = source.read_bytes()
    after = source.stat()
    if _stat_identity(before) != _stat_identity(after) or len(raw) != before.st_size:
        raise RuntimeError("recipe changed while it was loaded")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("recipe must be strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("recipe must contain one JSON object")
    return CorpusMixtureRecipe.from_dict(value)


def parse_weight_overrides(values: Sequence[str]) -> dict[str, float]:
    """Parse repeatable ``leaf:ID=WEIGHT`` or ``shard:ID=WEIGHT`` controls."""

    overrides: dict[str, float] = {}
    for index, raw in enumerate(values):
        if not isinstance(raw, str) or "=" not in raw:
            raise ValueError(f"weight override {index} must be KIND:ID=WEIGHT")
        selector, raw_weight = raw.rsplit("=", 1)
        if ":" not in selector:
            raise ValueError(f"weight override {index} must start with 'leaf:' or 'shard:'")
        kind, identifier = selector.split(":", 1)
        if kind not in {"leaf", "shard"} or not identifier:
            raise ValueError(f"weight override {index} must start with 'leaf:' or 'shard:'")
        _identifier(identifier, f"weight override {index} identifier")
        key = f"{kind}:{identifier}"
        if key in overrides:
            raise ValueError(f"duplicate weight override for {key!r}")
        try:
            parsed: object = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"weight override {key!r} is not numeric") from exc
        overrides[key] = _positive_weight(parsed, f"weight override {key!r}")
    return overrides


def apply_weight_overrides(
    recipe: CorpusMixtureRecipe, overrides: Mapping[str, float]
) -> CorpusMixtureRecipe:
    """Return a new recipe; the loaded canonical recipe and its file remain untouched."""

    if not overrides:
        return recipe
    known_shards = {shard.descriptor.shard_id for shard in recipe.shards}
    known_leaves = {shard.descriptor.leaf_id for shard in recipe.shards}
    parsed: dict[str, float] = {}
    for key, value in overrides.items():
        if not isinstance(key, str) or ":" not in key:
            raise ValueError("override keys must be leaf:ID or shard:ID")
        kind, identifier = key.split(":", 1)
        if kind == "leaf" and identifier not in known_leaves:
            raise KeyError(f"unknown leaf weight override: {identifier!r}")
        if kind == "shard" and identifier not in known_shards:
            raise KeyError(f"unknown shard weight override: {identifier!r}")
        if kind not in {"leaf", "shard"}:
            raise ValueError("override keys must be leaf:ID or shard:ID")
        parsed[key] = _positive_weight(value, f"weight override {key!r}")

    weights = {shard.descriptor.shard_id: Fraction(str(shard.weight)) for shard in recipe.shards}
    for key in sorted(parsed):
        kind, identifier = key.split(":", 1)
        if kind != "leaf":
            continue
        members = [shard for shard in recipe.shards if shard.descriptor.leaf_id == identifier]
        prior_total = sum(
            (weights[shard.descriptor.shard_id] for shard in members),
            start=Fraction(0, 1),
        )
        scale = Fraction(str(parsed[key])) / prior_total
        for shard in members:
            weights[shard.descriptor.shard_id] *= scale
    for key in sorted(parsed):
        kind, identifier = key.split(":", 1)
        if kind == "shard":
            weights[identifier] = Fraction(str(parsed[key]))
    return replace(
        recipe,
        shards=tuple(
            replace(shard, weight=float(weights[shard.descriptor.shard_id]))
            for shard in recipe.shards
        ),
    )


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int | None]:
    return stat.st_size, stat.st_mtime_ns, getattr(stat, "st_ino", None)


def _resolve_shard_path(recipe_directory: Path, descriptor: CorpusShardDescriptor) -> Path:
    raw = Path(descriptor.path).expanduser()
    source = raw if raw.is_absolute() else recipe_directory / raw
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            f"shard {descriptor.versioned_id!r} must resolve to a regular non-symlink file"
        )
    return source.resolve()


def _normalized_payload(value: object) -> tuple[str, str] | tuple[None, str]:
    if not isinstance(value, Mapping) or set(value) != {"messages"}:
        return None, "row_not_messages_only"
    raw_messages = value["messages"]
    if not isinstance(raw_messages, list):
        return None, "messages_not_array"
    try:
        example = ChatExample.from_dict({"messages": raw_messages})
    except ToolTraceError as exc:
        return None, exc.reason
    except (TypeError, ValueError):
        return None, "invalid_chat_contract"
    messages: list[dict[str, object]] = []
    for message in example.messages:
        normalized_message = message.to_dict()
        normalized_message["content"] = unicodedata.normalize(
            "NFC", message.content.replace("\r\n", "\n").replace("\r", "\n")
        ).strip()
        messages.append(normalized_message)
    normalized = {"messages": messages}
    payload = canonical_json_bytes(normalized).decode("utf-8")
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_jsonl_record(raw_line: bytes) -> tuple[str, str, str | None, str | None]:
    if len(raw_line) > DEFAULT_MAXIMUM_RECORD_BYTES:
        return "rejected", "record_too_large", None, None
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError:
        return "rejected", "invalid_utf8", None, None
    if not text.strip():
        return "rejected", "blank_record", None, None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError):
        return "rejected", "invalid_json", None, None
    if not isinstance(value, Mapping) or set(value) != {"messages"}:
        return "rejected", "row_not_messages_only", None, None
    payload, detail = _normalized_payload(value)
    if payload is None:
        return "quarantined", detail, None, None
    return "valid", "messages_only", payload, detail


def _create_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE guard_hashes (
            digest TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE records (
            id INTEGER PRIMARY KEY,
            shard_index INTEGER NOT NULL,
            shard_id TEXT NOT NULL,
            shard_revision TEXT NOT NULL,
            leaf_id TEXT NOT NULL,
            leaf_revision TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            digest TEXT,
            payload TEXT,
            selection_rank TEXT,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            output_index INTEGER,
            duplicate_shard_id TEXT,
            duplicate_shard_revision TEXT,
            duplicate_source_line INTEGER,
            UNIQUE (shard_index, source_line)
        );
        CREATE TABLE owners (
            digest TEXT PRIMARY KEY,
            record_id INTEGER NOT NULL REFERENCES records(id)
        ) WITHOUT ROWID;
        CREATE TABLE emissions (
            record_id INTEGER PRIMARY KEY REFERENCES records(id),
            output_index INTEGER NOT NULL UNIQUE
        );
        CREATE INDEX records_selection
            ON records(shard_index, status, selection_rank, digest, source_line);
        """
    )


def _scan_guard(
    connection: sqlite3.Connection,
    source: Path,
    descriptor: CorpusShardDescriptor,
) -> dict[str, object]:
    before = source.stat()
    digest = hashlib.sha256()
    records = 0
    duplicates = 0
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            records += 1
            digest.update(raw_line)
            status, reason, _payload, message_digest = _parse_jsonl_record(raw_line)
            if status != "valid" or message_digest is None:
                raise ValueError(
                    f"leakage guard {descriptor.versioned_id!r} line {line_number} is invalid: "
                    f"{reason}"
                )
            cursor = connection.execute(
                "INSERT OR IGNORE INTO guard_hashes(digest) VALUES (?)", (message_digest,)
            )
            duplicates += int(cursor.rowcount == 0)
    after = source.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"shard {descriptor.versioned_id!r} changed while it was read")
    observed_digest = digest.hexdigest()
    if observed_digest != descriptor.sha256:
        raise ValueError(f"shard {descriptor.versioned_id!r} SHA-256 mismatch")
    if records != descriptor.record_count:
        raise ValueError(
            f"shard {descriptor.versioned_id!r} record count mismatch: "
            f"expected {descriptor.record_count}, observed {records}"
        )
    return {
        "shard_id": descriptor.shard_id,
        "shard_revision": descriptor.shard_revision,
        "records_scanned": records,
        "duplicate_guard_records": duplicates,
    }


def _selection_rank(seed: str, descriptor: CorpusShardDescriptor, digest: str, line: int) -> str:
    value = f"{seed}\0record\0{descriptor.versioned_id}\0{digest}\0{line}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _scan_mixture_shard(
    connection: sqlite3.Connection,
    source: Path,
    descriptor: CorpusShardDescriptor,
    shard_index: int,
    seed: str,
) -> None:
    before = source.stat()
    file_digest = hashlib.sha256()
    records = 0
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            records += 1
            file_digest.update(raw_line)
            status, reason, payload, message_digest = _parse_jsonl_record(raw_line)
            common = (
                shard_index,
                descriptor.shard_id,
                descriptor.shard_revision,
                descriptor.leaf_id,
                descriptor.leaf_revision,
                line_number,
            )
            if status != "valid" or payload is None or message_digest is None:
                connection.execute(
                    """
                    INSERT INTO records(
                        shard_index, shard_id, shard_revision, leaf_id, leaf_revision,
                        source_line, status, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*common, status, reason),
                )
                continue
            leak = connection.execute(
                "SELECT 1 FROM guard_hashes WHERE digest = ?", (message_digest,)
            ).fetchone()
            if leak is not None:
                connection.execute(
                    """
                    INSERT INTO records(
                        shard_index, shard_id, shard_revision, leaf_id, leaf_revision,
                        source_line, digest, status, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'quarantined', 'train_eval_exact_match')
                    """,
                    (*common, message_digest),
                )
                continue
            owner = connection.execute(
                """
                SELECT r.shard_id, r.shard_revision, r.source_line
                FROM owners AS o JOIN records AS r ON r.id = o.record_id
                WHERE o.digest = ?
                """,
                (message_digest,),
            ).fetchone()
            if owner is not None:
                connection.execute(
                    """
                    INSERT INTO records(
                        shard_index, shard_id, shard_revision, leaf_id, leaf_revision,
                        source_line, digest, status, reason, duplicate_shard_id,
                        duplicate_shard_revision, duplicate_source_line
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'duplicate', 'normalized_messages_duplicate',
                              ?, ?, ?)
                    """,
                    (*common, message_digest, owner[0], owner[1], owner[2]),
                )
                continue
            cursor = connection.execute(
                """
                INSERT INTO records(
                    shard_index, shard_id, shard_revision, leaf_id, leaf_revision,
                    source_line, digest, payload, selection_rank, status, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', 'eligible')
                """,
                (
                    *common,
                    message_digest,
                    payload,
                    _selection_rank(seed, descriptor, message_digest, line_number),
                ),
            )
            connection.execute(
                "INSERT INTO owners(digest, record_id) VALUES (?, ?)",
                (message_digest, cursor.lastrowid),
            )
    after = source.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"shard {descriptor.versioned_id!r} changed while it was read")
    if file_digest.hexdigest() != descriptor.sha256:
        raise ValueError(f"shard {descriptor.versioned_id!r} SHA-256 mismatch")
    if records != descriptor.record_count:
        raise ValueError(
            f"shard {descriptor.versioned_id!r} record count mismatch: "
            f"expected {descriptor.record_count}, observed {records}"
        )


def _apply_caps(connection: sqlite3.Connection, recipe: CorpusMixtureRecipe) -> None:
    for shard_index, shard in enumerate(recipe.shards):
        if shard.cap is None:
            continue
        connection.execute(
            """
            UPDATE records SET status = 'capped', reason = 'shard_cap'
            WHERE id IN (
                SELECT id FROM records
                WHERE shard_index = ? AND status = 'candidate'
                ORDER BY selection_rank, digest, source_line
                LIMIT -1 OFFSET ?
            )
            """,
            (shard_index, shard.cap),
        )


def _interleave_tie(seed: str, versioned_id: str, ordinal: int) -> str:
    return hashlib.sha256(
        f"{seed}\0interleave\0{versioned_id}\0{ordinal}".encode("utf-8")
    ).hexdigest()


def _select_records(connection: sqlite3.Connection, recipe: CorpusMixtureRecipe) -> int:
    available = int(
        connection.execute("SELECT COUNT(*) FROM records WHERE status = 'candidate'").fetchone()[0]
    )
    if available < recipe.target_records and recipe.shortfall_policy == "error":
        raise ValueError(
            f"mixture has {available} eligible records but target_records is "
            f"{recipe.target_records}"
        )
    target = min(available, recipe.target_records)
    states: dict[int, tuple[sqlite3.Cursor, int, Fraction, str]] = {}
    heap: list[tuple[Fraction, str, int, int]] = []
    for shard_index, shard in enumerate(recipe.shards):
        cursor = connection.execute(
            """
            SELECT id FROM records
            WHERE shard_index = ? AND status = 'candidate'
            ORDER BY selection_rank, digest, source_line
            """,
            (shard_index,),
        )
        row = cursor.fetchone()
        if row is None:
            cursor.close()
            continue
        exact_weight = Fraction(str(shard.weight))
        versioned_id = shard.descriptor.versioned_id
        states[shard_index] = (cursor, 1, exact_weight, versioned_id)
        heapq.heappush(
            heap,
            (
                Fraction(1, 1) / exact_weight,
                _interleave_tie(recipe.seed, versioned_id, 1),
                shard_index,
                int(row[0]),
            ),
        )

    selected = 0
    while selected < target:
        _priority, _tie, shard_index, record_id = heapq.heappop(heap)
        connection.execute(
            "INSERT INTO emissions(record_id, output_index) VALUES (?, ?)",
            (record_id, selected),
        )
        selected += 1
        cursor, ordinal, exact_weight, versioned_id = states[shard_index]
        row = cursor.fetchone()
        if row is None:
            cursor.close()
            del states[shard_index]
            continue
        next_ordinal = ordinal + 1
        states[shard_index] = (cursor, next_ordinal, exact_weight, versioned_id)
        heapq.heappush(
            heap,
            (
                Fraction(next_ordinal, 1) / exact_weight,
                _interleave_tie(recipe.seed, versioned_id, next_ordinal),
                shard_index,
                int(row[0]),
            ),
        )
    for cursor, _ordinal, _weight, _versioned_id in states.values():
        cursor.close()

    connection.execute(
        """
        UPDATE records
        SET status = 'emitted', reason = 'selected',
            output_index = (SELECT output_index FROM emissions WHERE record_id = records.id)
        WHERE id IN (SELECT record_id FROM emissions)
        """
    )
    connection.execute(
        "UPDATE records SET status = 'capped', reason = 'target_budget' WHERE status = 'candidate'"
    )
    return selected


def _write_or_hash_rows(
    rows: Sequence[bytes] | Any, destination: Path | None
) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    handle = destination.open("wb") if destination is not None else None
    try:
        for row in rows:
            digest.update(row)
            byte_count += len(row)
            record_count += 1
            if handle is not None:
                handle.write(row)
    finally:
        if handle is not None:
            handle.close()
    return digest.hexdigest(), byte_count, record_count


def _message_rows(connection: sqlite3.Connection) -> Any:
    cursor = connection.execute(
        "SELECT payload FROM records WHERE status = 'emitted' ORDER BY output_index"
    )
    for (payload,) in cursor:
        yield (str(payload) + "\n").encode("utf-8")


def _provenance_rows(connection: sqlite3.Connection) -> Any:
    cursor = connection.execute(
        """
        SELECT shard_id, shard_revision, leaf_id, leaf_revision, source_line, status, reason,
               digest, output_index, duplicate_shard_id, duplicate_shard_revision,
               duplicate_source_line
        FROM records ORDER BY shard_index, source_line
        """
    )
    for row in cursor:
        duplicate_of = None
        if row[9] is not None:
            duplicate_of = {
                "shard_id": row[9],
                "shard_revision": row[10],
                "source_line": row[11],
            }
        value = {
            "schema_version": CORPUS_MIXTURE_PROVENANCE_SCHEMA,
            "shard_id": row[0],
            "shard_revision": row[1],
            "leaf_id": row[2],
            "leaf_revision": row[3],
            "source_line": row[4],
            "status": row[5],
            "reason": row[6],
            "normalized_messages_sha256": row[7],
            "output_index": row[8],
            "duplicate_of": duplicate_of,
        }
        yield canonical_json_bytes(value) + b"\n"


def _status_counts(
    connection: sqlite3.Connection, where: str = "", args: tuple = ()
) -> dict[str, int]:
    query = "SELECT status, COUNT(*) FROM records"
    if where:
        query += " WHERE " + where
    query += " GROUP BY status"
    observed = {str(status): int(count) for status, count in connection.execute(query, args)}
    return {status: observed.get(status, 0) for status in TERMINAL_STATUSES}


def _reason_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        str(reason): int(count)
        for reason, count in connection.execute(
            "SELECT reason, COUNT(*) FROM records GROUP BY reason ORDER BY reason"
        )
    }


def _build_report(
    connection: sqlite3.Connection,
    recipe: CorpusMixtureRecipe,
    *,
    canonical_recipe_fingerprint: str,
    overrides: Mapping[str, float],
    guard_reports: list[dict[str, object]],
    selected: int,
    dry_run: bool,
    message_artifact: dict[str, object],
    provenance_artifact: dict[str, object],
    resolved_recipe_artifact: dict[str, object],
) -> dict[str, object]:
    input_records = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    accounting = _status_counts(connection)
    accounted = sum(accounting.values())
    per_shard: list[dict[str, object]] = []
    for shard_index, shard in enumerate(recipe.shards):
        counts = _status_counts(connection, "shard_index = ?", (shard_index,))
        per_shard.append(
            {
                "shard_id": shard.descriptor.shard_id,
                "shard_revision": shard.descriptor.shard_revision,
                "leaf_id": shard.descriptor.leaf_id,
                "leaf_revision": shard.descriptor.leaf_revision,
                "weight": shard.weight,
                "cap": shard.cap,
                "input_records": sum(counts.values()),
                "accounting": counts,
            }
        )
    per_leaf = {
        str(leaf): int(count)
        for leaf, count in connection.execute(
            "SELECT leaf_id, COUNT(*) FROM records WHERE status = 'emitted' "
            "GROUP BY leaf_id ORDER BY leaf_id"
        )
    }
    content_identity = {
        "schema_version": CORPUS_MIXTURE_CONTENT_SCHEMA,
        "messages_jsonl_sha256": message_artifact["sha256"],
        "record_count": selected,
    }
    semantic_run = fingerprint_json(
        {
            "canonical_recipe_sha256": canonical_recipe_fingerprint,
            "resolved_recipe_sha256": recipe.fingerprint,
            "content": content_identity,
        }
    )
    return {
        "schema_version": CORPUS_MIXTURE_REPORT_SCHEMA,
        "recipe_id": recipe.recipe_id,
        "recipe_revision": recipe.revision,
        "canonical_recipe_sha256": canonical_recipe_fingerprint,
        "resolved_recipe_sha256": recipe.fingerprint,
        "semantic_run_sha256": semantic_run,
        "weight_overrides": dict(sorted(overrides.items())),
        "usage": recipe.usage,
        "output_split": recipe.output_split,
        "template_mode": recipe.template_mode,
        "seed": recipe.seed,
        "target_records": recipe.target_records,
        "emitted_records": selected,
        "shortfall_records": recipe.target_records - selected,
        "input_records": input_records,
        "input_accounting": accounting,
        "reason_accounting": _reason_counts(connection),
        "per_shard": per_shard,
        "emitted_by_leaf": per_leaf,
        "leakage_guard": {
            "shards": guard_reports,
            "records_scanned": sum(int(item["records_scanned"]) for item in guard_reports),
            "unique_normalized_messages": int(
                connection.execute("SELECT COUNT(*) FROM guard_hashes").fetchone()[0]
            ),
        },
        "content_identity": content_identity,
        "artifacts": {
            "messages.jsonl": message_artifact,
            "provenance.jsonl": provenance_artifact,
            "resolved_recipe.json": resolved_recipe_artifact,
        },
        "dry_run": dry_run,
        "zero_silent_drops": input_records == accounted,
    }


def _artifact_metadata(
    sha256: str, byte_count: int, record_count: int | None = None
) -> dict[str, object]:
    result: dict[str, object] = {"sha256": sha256, "bytes": byte_count}
    if record_count is not None:
        result["records"] = record_count
    return result


def build_corpus_mixture(
    recipe: CorpusMixtureRecipe,
    *,
    recipe_directory: str | Path,
    output_directory: str | Path,
    canonical_recipe_fingerprint: str | None = None,
    weight_overrides: Mapping[str, float] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Validate, deduplicate, select, and optionally materialize one corpus mixture."""

    if not isinstance(recipe, CorpusMixtureRecipe):
        raise TypeError("recipe must be a CorpusMixtureRecipe")
    recipe_root = Path(recipe_directory).expanduser().resolve()
    output = Path(output_directory).expanduser()
    source_fingerprint = canonical_recipe_fingerprint or recipe.fingerprint
    source_fingerprint = _lowercase_sha256(source_fingerprint, "canonical_recipe_fingerprint")
    overrides = dict(weight_overrides or {})

    descriptors = [shard.descriptor for shard in recipe.shards] + list(recipe.leakage_guard_shards)
    resolved_paths = [_resolve_shard_path(recipe_root, descriptor) for descriptor in descriptors]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("one physical shard file may appear only once in a recipe")
    mixture_paths = resolved_paths[: len(recipe.shards)]
    guard_paths = resolved_paths[len(recipe.shards) :]

    with tempfile.TemporaryDirectory(prefix="molor-corpus-mixture-") as temporary:
        database = Path(temporary) / "mixture.sqlite3"
        connection = sqlite3.connect(database)
        try:
            _create_database(connection)
            guard_reports = [
                _scan_guard(connection, path, descriptor)
                for path, descriptor in zip(guard_paths, recipe.leakage_guard_shards, strict=True)
            ]
            for shard_index, (shard, path) in enumerate(
                zip(recipe.shards, mixture_paths, strict=True)
            ):
                _scan_mixture_shard(
                    connection,
                    path,
                    shard.descriptor,
                    shard_index,
                    recipe.seed,
                )
            _apply_caps(connection, recipe)
            selected = _select_records(connection, recipe)
            connection.commit()

            stage: Path | None = None
            if not dry_run:
                if output.exists():
                    raise FileExistsError(f"output directory already exists: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
            try:
                message_hash, message_bytes, message_records = _write_or_hash_rows(
                    _message_rows(connection), None if stage is None else stage / "messages.jsonl"
                )
                provenance_hash, provenance_bytes, provenance_records = _write_or_hash_rows(
                    _provenance_rows(connection),
                    None if stage is None else stage / "provenance.jsonl",
                )
                resolved_bytes = corpus_mixture_recipe_json_bytes(recipe)
                resolved_hash = hashlib.sha256(resolved_bytes).hexdigest()
                if stage is not None:
                    (stage / "resolved_recipe.json").write_bytes(resolved_bytes)
                report = _build_report(
                    connection,
                    recipe,
                    canonical_recipe_fingerprint=source_fingerprint,
                    overrides=overrides,
                    guard_reports=guard_reports,
                    selected=selected,
                    dry_run=dry_run,
                    message_artifact=_artifact_metadata(
                        message_hash, message_bytes, message_records
                    ),
                    provenance_artifact=_artifact_metadata(
                        provenance_hash, provenance_bytes, provenance_records
                    ),
                    resolved_recipe_artifact=_artifact_metadata(resolved_hash, len(resolved_bytes)),
                )
                if not report["zero_silent_drops"]:
                    raise AssertionError(
                        "internal accounting error left source records unclassified"
                    )
                if stage is not None:
                    report_bytes = (
                        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                    ).encode("utf-8")
                    (stage / "aggregate_report.json").write_bytes(report_bytes)
                    sums = {
                        "aggregate_report.json": hashlib.sha256(report_bytes).hexdigest(),
                        "messages.jsonl": message_hash,
                        "provenance.jsonl": provenance_hash,
                        "resolved_recipe.json": resolved_hash,
                    }
                    sums_bytes = "".join(
                        f"{digest}  {name}\n" for name, digest in sorted(sums.items())
                    ).encode("utf-8")
                    (stage / "sha256sums.txt").write_bytes(sums_bytes)
                    os.replace(stage, output)
                    stage = None
                return report
            finally:
                if stage is not None and stage.exists():
                    shutil.rmtree(stage)
        finally:
            connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="KIND:ID=VALUE",
        help="repeatable leaf or shard weight override",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    canonical = load_corpus_mixture_recipe(args.recipe)
    overrides = parse_weight_overrides(args.weight)
    resolved = apply_weight_overrides(canonical, overrides)
    report = build_corpus_mixture(
        resolved,
        recipe_directory=args.recipe.parent,
        output_directory=args.output,
        canonical_recipe_fingerprint=canonical.fingerprint,
        weight_overrides=overrides,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CORPUS_MIXTURE_CONTENT_SCHEMA",
    "CORPUS_MIXTURE_PROVENANCE_SCHEMA",
    "CORPUS_MIXTURE_RECIPE_SCHEMA",
    "CORPUS_MIXTURE_REPORT_SCHEMA",
    "CorpusMixtureRecipe",
    "CorpusMixtureShard",
    "CorpusShardDescriptor",
    "apply_weight_overrides",
    "build_corpus_mixture",
    "corpus_mixture_recipe_json_bytes",
    "load_corpus_mixture_recipe",
    "main",
    "parse_weight_overrides",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
