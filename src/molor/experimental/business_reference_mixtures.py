"""Deterministic, content-addressed mixtures of grounded business references.

The input and output contract is deliberately narrow: every data row is exactly
``{document_id, title, text}``.  This module selects reference documents; it never
turns them into prompts, answers, or reasoning traces.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
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
import tempfile
from typing import Any, ClassVar, Literal

from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256


REFERENCE_MIXTURE_RECIPE_SCHEMA = "molor.business-reference-mixture-recipe.v1"
REFERENCE_MIXTURE_PROVENANCE_SCHEMA = "molor.business-reference-mixture-provenance.v1"
REFERENCE_MIXTURE_REPORT_SCHEMA = "molor.business-reference-mixture-report.v1"
REFERENCE_MIXTURE_CONTENT_SCHEMA = "molor.business-reference-mixture-content.v1"
REFERENCE_CORPUS_SCHEMA = "molor.business-reference-corpus.v1"

DEFAULT_MAXIMUM_RECIPE_BYTES = 4 * 1024 * 1024
DEFAULT_MAXIMUM_RECORD_BYTES = 16 * 1024 * 1024
TERMINAL_STATUSES = ("emitted", "duplicate", "quarantined", "rejected", "capped")
SPLITS = frozenset(("train", "validation", "test"))
USAGES = frozenset(("training", "evaluation"))
SHORTFALL_POLICIES = frozenset(("error", "emit_available"))

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
    if not isinstance(value, str) or value not in SPLITS:
        raise ValueError(f"{field_name} must be one of {sorted(SPLITS)!r}")
    return value


def _path(value: object, field_name: str) -> str:
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
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return parsed


def _lowercase_sha256(value: object, field_name: str) -> str:
    digest = validate_sha256(value, field_name)  # type: ignore[arg-type]
    if value != digest:
        raise ValueError(f"{field_name} must use lowercase hexadecimal")
    return digest


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        raise ValueError(
            f"{field_name} fields differ from v1 (missing={missing}, unknown={unknown})"
        )


def _raise_type(message: str) -> Any:
    raise TypeError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"JSON contains unsupported numeric constant {value!r}")


@dataclass(frozen=True, slots=True)
class ReferenceShardDescriptor:
    """One immutable, versioned JSONL shard of grounded documents."""

    shard_id: str
    shard_revision: str
    leaf_id: str
    leaf_revision: str
    split: str
    path: str
    sha256: str
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_id", _identifier(self.shard_id, "shard_id"))
        object.__setattr__(self, "shard_revision", _revision(self.shard_revision, "shard_revision"))
        object.__setattr__(self, "leaf_id", _identifier(self.leaf_id, "leaf_id"))
        object.__setattr__(self, "leaf_revision", _revision(self.leaf_revision, "leaf_revision"))
        object.__setattr__(self, "split", _split(self.split, "split"))
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(self, "sha256", _lowercase_sha256(self.sha256, "sha256"))
        object.__setattr__(
            self, "record_count", _positive_integer(self.record_count, "record_count")
        )

    @property
    def versioned_id(self) -> str:
        return f"{self.shard_id}@{self.shard_revision}"

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "shard_revision": self.shard_revision,
            "leaf_id": self.leaf_id,
            "leaf_revision": self.leaf_revision,
            "split": self.split,
            "path": self.path,
            "sha256": self.sha256,
            "record_count": self.record_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], field_name: str) -> ReferenceShardDescriptor:
        expected = {
            "shard_id",
            "shard_revision",
            "leaf_id",
            "leaf_revision",
            "split",
            "path",
            "sha256",
            "record_count",
        }
        _require_exact_keys(value, expected, field_name)
        return cls(**{key: value[key] for key in expected})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ReferenceMixtureShard:
    """A pinned shard plus adjustable selection controls."""

    descriptor: ReferenceShardDescriptor
    weight: float
    cap: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ReferenceShardDescriptor):
            raise TypeError("descriptor must be a ReferenceShardDescriptor")
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
    def from_dict(cls, value: Mapping[str, object], field_name: str) -> ReferenceMixtureShard:
        expected = {
            "shard_id",
            "shard_revision",
            "leaf_id",
            "leaf_revision",
            "split",
            "path",
            "sha256",
            "record_count",
            "weight",
            "cap",
        }
        _require_exact_keys(value, expected, field_name)
        descriptor = {key: value[key] for key in expected if key not in {"weight", "cap"}}
        return cls(
            descriptor=ReferenceShardDescriptor.from_dict(descriptor, f"{field_name}.descriptor"),
            weight=value["weight"],  # type: ignore[arg-type]
            cap=value["cap"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ReferenceMixtureRecipe:
    """One exact reference mixture with mandatory opposite-split guards."""

    schema_version: ClassVar[str] = REFERENCE_MIXTURE_RECIPE_SCHEMA

    recipe_id: str
    revision: str
    usage: Literal["training", "evaluation"] | str
    output_split: Literal["train", "validation", "test"] | str
    seed: str
    target_records: int
    shortfall_policy: Literal["error", "emit_available"] | str
    shards: tuple[ReferenceMixtureShard, ...]
    leakage_guard_shards: tuple[ReferenceShardDescriptor, ...]
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        recipe_id = _identifier(self.recipe_id, "recipe_id")
        revision = _revision(self.revision, "revision")
        if self.usage not in USAGES:
            raise ValueError(f"usage must be one of {sorted(USAGES)!r}")
        output_split = _split(self.output_split, "output_split")
        if self.usage == "training" and output_split != "train":
            raise ValueError("training recipes must emit the train split")
        if self.usage == "evaluation" and output_split == "train":
            raise ValueError("evaluation recipes must emit validation or test")
        if not isinstance(self.seed, str) or not self.seed or len(self.seed) > 256:
            raise ValueError("seed must be a non-empty string of at most 256 characters")
        target_records = _positive_integer(self.target_records, "target_records")
        if self.shortfall_policy not in SHORTFALL_POLICIES:
            raise ValueError(f"shortfall_policy must be one of {sorted(SHORTFALL_POLICIES)!r}")
        if isinstance(self.shards, (str, bytes)) or not isinstance(self.shards, Sequence):
            raise TypeError("shards must be a sequence")
        shards = tuple(self.shards)
        if not shards or any(not isinstance(item, ReferenceMixtureShard) for item in shards):
            raise ValueError("shards must contain at least one ReferenceMixtureShard")
        if isinstance(self.leakage_guard_shards, (str, bytes)) or not isinstance(
            self.leakage_guard_shards, Sequence
        ):
            raise TypeError("leakage_guard_shards must be a sequence")
        guards = tuple(self.leakage_guard_shards)
        if any(not isinstance(item, ReferenceShardDescriptor) for item in guards):
            raise TypeError("leakage_guard_shards must contain ReferenceShardDescriptor values")

        all_ids = [item.descriptor.shard_id for item in shards] + [item.shard_id for item in guards]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("all mixture and guard shard IDs must be unique")
        leaf_revisions: dict[str, str] = {}
        for descriptor in [item.descriptor for item in shards] + list(guards):
            prior = leaf_revisions.setdefault(descriptor.leaf_id, descriptor.leaf_revision)
            if prior != descriptor.leaf_revision:
                raise ValueError(f"leaf {descriptor.leaf_id!r} has multiple revisions")
        if any(item.descriptor.split != output_split for item in shards):
            raise ValueError("every mixture shard split must equal output_split")

        mixture_leaves = {item.descriptor.leaf_id for item in shards}
        expected_guard_pairs = {
            (leaf_id, split)
            for leaf_id in mixture_leaves
            for split in SPLITS.difference({output_split})
        }
        guard_pairs = {(item.leaf_id, item.split) for item in guards}
        if guard_pairs != expected_guard_pairs:
            missing = sorted(expected_guard_pairs.difference(guard_pairs))
            extra = sorted(guard_pairs.difference(expected_guard_pairs))
            raise ValueError(
                "leakage guards must cover every participating leaf in both opposite splits "
                f"(missing={missing}, extra={extra})"
            )
        if len(guard_pairs) != len(guards):
            raise ValueError("only one leakage guard shard is allowed per leaf and split")

        canonical_shards = tuple(
            sorted(
                shards, key=lambda item: (item.descriptor.shard_id, item.descriptor.shard_revision)
            )
        )
        canonical_guards = tuple(
            sorted(guards, key=lambda item: (item.shard_id, item.shard_revision))
        )
        object.__setattr__(self, "recipe_id", recipe_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "output_split", output_split)
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
            "seed": self.seed,
            "target_records": self.target_records,
            "shortfall_policy": self.shortfall_policy,
            "shards": [item.to_dict() for item in self.shards],
            "leakage_guard_shards": [item.to_dict() for item in self.leakage_guard_shards],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_dict(), "content_sha256": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ReferenceMixtureRecipe:
        expected = {
            "schema_version",
            "recipe_id",
            "revision",
            "usage",
            "output_split",
            "seed",
            "target_records",
            "shortfall_policy",
            "shards",
            "leakage_guard_shards",
            "content_sha256",
        }
        _require_exact_keys(value, expected, "recipe")
        if value["schema_version"] != REFERENCE_MIXTURE_RECIPE_SCHEMA:
            raise ValueError("unsupported reference-mixture recipe schema")
        raw_shards = value["shards"]
        raw_guards = value["leakage_guard_shards"]
        if not isinstance(raw_shards, list) or not isinstance(raw_guards, list):
            raise TypeError("shards and leakage_guard_shards must be arrays")
        recipe = cls(
            recipe_id=value["recipe_id"],  # type: ignore[arg-type]
            revision=value["revision"],  # type: ignore[arg-type]
            usage=value["usage"],  # type: ignore[arg-type]
            output_split=value["output_split"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            target_records=value["target_records"],  # type: ignore[arg-type]
            shortfall_policy=value["shortfall_policy"],  # type: ignore[arg-type]
            shards=tuple(
                ReferenceMixtureShard.from_dict(item, f"shards[{index}]")
                if isinstance(item, Mapping)
                else _raise_type(f"shards[{index}] must be an object")
                for index, item in enumerate(raw_shards)
            ),
            leakage_guard_shards=tuple(
                ReferenceShardDescriptor.from_dict(item, f"leakage_guard_shards[{index}]")
                if isinstance(item, Mapping)
                else _raise_type(f"leakage_guard_shards[{index}] must be an object")
                for index, item in enumerate(raw_guards)
            ),
        )
        if raw_shards != recipe.content_dict()["shards"]:
            raise ValueError("mixture shards must be ordered canonically")
        if raw_guards != recipe.content_dict()["leakage_guard_shards"]:
            raise ValueError("leakage guards must be ordered canonically")
        claimed = _lowercase_sha256(value["content_sha256"], "content_sha256")
        if claimed != recipe.fingerprint:
            raise ValueError("recipe content_sha256 does not match its semantic payload")
        return recipe


def reference_mixture_recipe_json_bytes(recipe: ReferenceMixtureRecipe) -> bytes:
    if not isinstance(recipe, ReferenceMixtureRecipe):
        raise TypeError("recipe must be a ReferenceMixtureRecipe")
    return (json.dumps(recipe.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def load_reference_mixture_recipe(
    path: str | Path, *, maximum_bytes: int = DEFAULT_MAXIMUM_RECIPE_BYTES
) -> ReferenceMixtureRecipe:
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
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("recipe must be strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("recipe must contain one JSON object")
    return ReferenceMixtureRecipe.from_dict(value)


def parse_weight_overrides(values: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, raw in enumerate(values):
        selector, raw_value = _parse_selector(raw, index, "weight")
        try:
            parsed: object = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"weight override {selector!r} is not numeric") from exc
        if selector in result:
            raise ValueError(f"duplicate weight override for {selector!r}")
        result[selector] = _positive_weight(parsed, f"weight override {selector!r}")
    return result


def parse_cap_overrides(values: Sequence[str]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for index, raw in enumerate(values):
        selector, raw_value = _parse_selector(raw, index, "cap")
        if selector in result:
            raise ValueError(f"duplicate cap override for {selector!r}")
        if raw_value == "none":
            result[selector] = None
            continue
        try:
            parsed: object = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"cap override {selector!r} is not an integer or 'none'") from exc
        result[selector] = _positive_integer(parsed, f"cap override {selector!r}")
    return result


def _parse_selector(raw: object, index: int, kind_name: str) -> tuple[str, str]:
    if not isinstance(raw, str) or "=" not in raw:
        raise ValueError(f"{kind_name} override {index} must be KIND:ID=VALUE")
    selector, raw_value = raw.rsplit("=", 1)
    if ":" not in selector:
        raise ValueError(f"{kind_name} override {index} must start with 'leaf:' or 'shard:'")
    kind, identifier = selector.split(":", 1)
    if kind not in {"leaf", "shard"} or not identifier:
        raise ValueError(f"{kind_name} override {index} must start with 'leaf:' or 'shard:'")
    _identifier(identifier, f"{kind_name} override {index} identifier")
    return f"{kind}:{identifier}", raw_value


def apply_overrides(
    recipe: ReferenceMixtureRecipe,
    *,
    weight_overrides: Mapping[str, float] | None = None,
    cap_overrides: Mapping[str, int | None] | None = None,
) -> ReferenceMixtureRecipe:
    """Return a resolved recipe without mutating the canonical recipe."""

    weights = dict(weight_overrides or {})
    caps = dict(cap_overrides or {})
    known_shards = {item.descriptor.shard_id for item in recipe.shards}
    known_leaves = {item.descriptor.leaf_id for item in recipe.shards}
    for selector in [*weights, *caps]:
        if not isinstance(selector, str) or ":" not in selector:
            raise ValueError("override keys must be leaf:ID or shard:ID")
        kind, identifier = selector.split(":", 1)
        if kind == "leaf" and identifier not in known_leaves:
            raise KeyError(f"unknown leaf override: {identifier!r}")
        if kind == "shard" and identifier not in known_shards:
            raise KeyError(f"unknown shard override: {identifier!r}")
        if kind not in {"leaf", "shard"}:
            raise ValueError("override keys must be leaf:ID or shard:ID")

    resolved_weights = {item.descriptor.shard_id: item.weight for item in recipe.shards}
    for selector in sorted(weights):
        kind, identifier = selector.split(":", 1)
        parsed = _positive_weight(weights[selector], f"weight override {selector!r}")
        if kind == "leaf":
            members = [item for item in recipe.shards if item.descriptor.leaf_id == identifier]
            prior_total = math.fsum(resolved_weights[item.descriptor.shard_id] for item in members)
            scale = parsed / prior_total
            for item in members:
                resolved_weights[item.descriptor.shard_id] *= scale
    for selector in sorted(weights):
        kind, identifier = selector.split(":", 1)
        if kind == "shard":
            resolved_weights[identifier] = _positive_weight(
                weights[selector], f"weight override {selector!r}"
            )

    resolved_caps = {item.descriptor.shard_id: item.cap for item in recipe.shards}
    for selector in sorted(caps):
        kind, identifier = selector.split(":", 1)
        cap = _optional_positive_integer(caps[selector], f"cap override {selector!r}")
        if kind == "leaf":
            for item in recipe.shards:
                if item.descriptor.leaf_id == identifier:
                    resolved_caps[item.descriptor.shard_id] = cap
    for selector in sorted(caps):
        kind, identifier = selector.split(":", 1)
        if kind == "shard":
            resolved_caps[identifier] = _optional_positive_integer(
                caps[selector], f"cap override {selector!r}"
            )

    return replace(
        recipe,
        shards=tuple(
            replace(
                item,
                weight=resolved_weights[item.descriptor.shard_id],
                cap=resolved_caps[item.descriptor.shard_id],
            )
            for item in recipe.shards
        ),
    )


@dataclass(slots=True)
class _Record:
    shard_index: int
    descriptor: ReferenceShardDescriptor
    source_line: int
    status: str
    reason: str
    document_id: str | None = None
    exact_sha256: str | None = None
    payload: str | None = None
    selection_rank: str | None = None
    output_index: int | None = None
    duplicate_of: dict[str, object] | None = None


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int | None]:
    return stat.st_size, stat.st_mtime_ns, getattr(stat, "st_ino", None)


def _resolve_shard_path(recipe_directory: Path, descriptor: ReferenceShardDescriptor) -> Path:
    raw = Path(descriptor.path).expanduser()
    source = raw if raw.is_absolute() else recipe_directory / raw
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            f"shard {descriptor.versioned_id!r} must resolve to a regular non-symlink file"
        )
    return source.resolve()


def _parse_document(
    raw_line: bytes,
) -> tuple[str, str, str | None, str | None, str | None]:
    if len(raw_line) > DEFAULT_MAXIMUM_RECORD_BYTES:
        return "rejected", "record_too_large", None, None, None
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError:
        return "rejected", "invalid_utf8", None, None, None
    if not text.strip():
        return "rejected", "blank_record", None, None, None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return "rejected", "invalid_json", None, None, None
    if not isinstance(value, Mapping) or set(value) != {"document_id", "title", "text"}:
        return "rejected", "row_not_document_only", None, None, None
    document_id = value["document_id"]
    title = value["title"]
    document_text = value["text"]
    if not isinstance(document_id, str) or _SHA256_PATTERN.fullmatch(document_id) is None:
        return "quarantined", "invalid_document_id", None, None, None
    if not isinstance(title, str) or not isinstance(document_text, str):
        return "quarantined", "invalid_document_field_type", None, None, None
    if not document_text.strip():
        return "quarantined", "empty_document_text", None, None, None
    payload_value = {"document_id": document_id, "title": title, "text": document_text}
    payload = canonical_json_bytes(payload_value).decode("utf-8")
    exact_payload = canonical_json_bytes({"title": title, "text": document_text})
    exact_sha256 = hashlib.sha256(exact_payload).hexdigest()
    return "valid", "grounded_document", payload, document_id, exact_sha256


def _scan_guard(
    source: Path,
    descriptor: ReferenceShardDescriptor,
    document_ids: dict[str, tuple[str, int]],
    exact_hashes: dict[str, tuple[str, int]],
) -> dict[str, object]:
    before = source.stat()
    file_hash = hashlib.sha256()
    records = 0
    duplicate_ids = 0
    duplicate_exact = 0
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            records += 1
            file_hash.update(raw_line)
            status, reason, _payload, document_id, exact_sha256 = _parse_document(raw_line)
            if status != "valid" or document_id is None or exact_sha256 is None:
                raise ValueError(
                    f"leakage guard {descriptor.versioned_id!r} line {line_number} is invalid: "
                    f"{reason}"
                )
            owner = (descriptor.versioned_id, line_number)
            duplicate_ids += int(document_id in document_ids)
            duplicate_exact += int(exact_sha256 in exact_hashes)
            document_ids.setdefault(document_id, owner)
            exact_hashes.setdefault(exact_sha256, owner)
    after = source.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"shard {descriptor.versioned_id!r} changed while it was read")
    if file_hash.hexdigest() != descriptor.sha256:
        raise ValueError(f"shard {descriptor.versioned_id!r} SHA-256 mismatch")
    if records != descriptor.record_count:
        raise ValueError(
            f"shard {descriptor.versioned_id!r} record count mismatch: "
            f"expected {descriptor.record_count}, observed {records}"
        )
    return {
        "shard_id": descriptor.shard_id,
        "shard_revision": descriptor.shard_revision,
        "leaf_id": descriptor.leaf_id,
        "split": descriptor.split,
        "records_scanned": records,
        "duplicate_document_ids": duplicate_ids,
        "duplicate_exact_documents": duplicate_exact,
    }


def _selection_rank(
    seed: str, descriptor: ReferenceShardDescriptor, document_id: str, line: int
) -> str:
    return hashlib.sha256(
        f"{seed}\0document\0{descriptor.versioned_id}\0{document_id}\0{line}".encode()
    ).hexdigest()


def _owner(record: _Record) -> dict[str, object]:
    return {
        "shard_id": record.descriptor.shard_id,
        "shard_revision": record.descriptor.shard_revision,
        "source_line": record.source_line,
    }


def _scan_mixture_shard(
    source: Path,
    descriptor: ReferenceShardDescriptor,
    shard_index: int,
    seed: str,
    guard_document_ids: Mapping[str, tuple[str, int]],
    guard_exact_hashes: Mapping[str, tuple[str, int]],
    document_owners: dict[str, _Record],
    exact_owners: dict[str, _Record],
) -> list[_Record]:
    before = source.stat()
    file_hash = hashlib.sha256()
    records: list[_Record] = []
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            file_hash.update(raw_line)
            status, reason, payload, document_id, exact_sha256 = _parse_document(raw_line)
            record = _Record(
                shard_index=shard_index,
                descriptor=descriptor,
                source_line=line_number,
                status=status,
                reason=reason,
                document_id=document_id,
                exact_sha256=exact_sha256,
            )
            records.append(record)
            if status != "valid" or payload is None or document_id is None or exact_sha256 is None:
                continue
            if document_id in guard_document_ids:
                record.status = "quarantined"
                record.reason = "opposite_split_document_id_match"
                continue
            if exact_sha256 in guard_exact_hashes:
                record.status = "quarantined"
                record.reason = "opposite_split_exact_match"
                continue
            id_owner = document_owners.get(document_id)
            if id_owner is not None:
                record.duplicate_of = _owner(id_owner)
                if id_owner.exact_sha256 == exact_sha256:
                    record.status = "duplicate"
                    record.reason = "document_id_duplicate"
                else:
                    record.status = "quarantined"
                    record.reason = "document_id_payload_conflict"
                continue
            exact_owner = exact_owners.get(exact_sha256)
            if exact_owner is not None:
                record.status = "duplicate"
                record.reason = "exact_document_duplicate"
                record.duplicate_of = _owner(exact_owner)
                continue
            record.status = "candidate"
            record.reason = "eligible"
            record.payload = payload
            record.selection_rank = _selection_rank(seed, descriptor, document_id, line_number)
            document_owners[document_id] = record
            exact_owners[exact_sha256] = record
    after = source.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"shard {descriptor.versioned_id!r} changed while it was read")
    if file_hash.hexdigest() != descriptor.sha256:
        raise ValueError(f"shard {descriptor.versioned_id!r} SHA-256 mismatch")
    if len(records) != descriptor.record_count:
        raise ValueError(
            f"shard {descriptor.versioned_id!r} record count mismatch: "
            f"expected {descriptor.record_count}, observed {len(records)}"
        )
    return records


def _apply_caps(records: list[_Record], recipe: ReferenceMixtureRecipe) -> None:
    for shard_index, shard in enumerate(recipe.shards):
        if shard.cap is None:
            continue
        candidates = sorted(
            (
                record
                for record in records
                if record.shard_index == shard_index and record.status == "candidate"
            ),
            key=lambda record: (
                record.selection_rank,
                record.document_id,
                record.source_line,
            ),
        )
        for record in candidates[shard.cap :]:
            record.status = "capped"
            record.reason = "shard_cap"


def _interleave_tie(seed: str, versioned_id: str, ordinal: int) -> str:
    return hashlib.sha256(f"{seed}\0interleave\0{versioned_id}\0{ordinal}".encode()).hexdigest()


def _select_records(records: list[_Record], recipe: ReferenceMixtureRecipe) -> list[_Record]:
    by_shard: dict[int, list[_Record]] = {}
    for shard_index in range(len(recipe.shards)):
        by_shard[shard_index] = sorted(
            (
                record
                for record in records
                if record.shard_index == shard_index and record.status == "candidate"
            ),
            key=lambda record: (
                record.selection_rank,
                record.document_id,
                record.source_line,
            ),
        )
    available = sum(len(items) for items in by_shard.values())
    if available < recipe.target_records and recipe.shortfall_policy == "error":
        raise ValueError(
            f"mixture has {available} eligible records but target_records is "
            f"{recipe.target_records}"
        )
    target = min(available, recipe.target_records)
    heap: list[tuple[Fraction, str, int, int]] = []
    for shard_index, items in by_shard.items():
        if not items:
            continue
        shard = recipe.shards[shard_index]
        heapq.heappush(
            heap,
            (
                Fraction(1, 1) / Fraction(str(shard.weight)),
                _interleave_tie(recipe.seed, shard.descriptor.versioned_id, 1),
                shard_index,
                0,
            ),
        )
    selected: list[_Record] = []
    while len(selected) < target:
        _priority, _tie, shard_index, item_index = heapq.heappop(heap)
        record = by_shard[shard_index][item_index]
        record.status = "emitted"
        record.reason = "selected"
        record.output_index = len(selected)
        selected.append(record)
        next_index = item_index + 1
        if next_index < len(by_shard[shard_index]):
            shard = recipe.shards[shard_index]
            ordinal = next_index + 1
            heapq.heappush(
                heap,
                (
                    Fraction(ordinal, 1) / Fraction(str(shard.weight)),
                    _interleave_tie(recipe.seed, shard.descriptor.versioned_id, ordinal),
                    shard_index,
                    next_index,
                ),
            )
    for record in records:
        if record.status == "candidate":
            record.status = "capped"
            record.reason = "target_budget"
    return selected


def _document_rows(records: Sequence[_Record]) -> Iterable[bytes]:
    for record in sorted(
        records, key=lambda item: item.output_index if item.output_index is not None else -1
    ):
        if record.payload is None:
            raise AssertionError("emitted record has no document payload")
        yield (record.payload + "\n").encode("utf-8")


def _provenance_rows(records: Sequence[_Record]) -> Iterable[bytes]:
    for record in sorted(records, key=lambda item: (item.shard_index, item.source_line)):
        value = {
            "schema_version": REFERENCE_MIXTURE_PROVENANCE_SCHEMA,
            "shard_id": record.descriptor.shard_id,
            "shard_revision": record.descriptor.shard_revision,
            "leaf_id": record.descriptor.leaf_id,
            "leaf_revision": record.descriptor.leaf_revision,
            "source_line": record.source_line,
            "document_id": record.document_id,
            "exact_document_sha256": record.exact_sha256,
            "status": record.status,
            "reason": record.reason,
            "output_index": record.output_index,
            "duplicate_of": record.duplicate_of,
        }
        yield canonical_json_bytes(value) + b"\n"


def _write_or_hash_rows(rows: Iterable[bytes], destination: Path | None) -> tuple[str, int, int]:
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


def _artifact(sha256: str, byte_count: int, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"sha256": sha256, "bytes": byte_count}
    if records is not None:
        result["records"] = records
    return result


def _accounting(records: Sequence[_Record]) -> dict[str, int]:
    observed = Counter(record.status for record in records)
    return {status: observed[status] for status in TERMINAL_STATUSES}


def _reason_accounting(records: Sequence[_Record]) -> dict[str, int]:
    observed = Counter(record.reason for record in records)
    return dict(sorted(observed.items()))


def _report(
    records: list[_Record],
    recipe: ReferenceMixtureRecipe,
    *,
    canonical_recipe_fingerprint: str,
    weight_overrides: Mapping[str, float],
    cap_overrides: Mapping[str, int | None],
    guard_reports: list[dict[str, object]],
    selected: list[_Record],
    dry_run: bool,
    document_artifact: dict[str, object],
    provenance_artifact: dict[str, object],
    resolved_recipe_artifact: dict[str, object],
) -> dict[str, object]:
    accounting = _accounting(records)
    per_shard: list[dict[str, object]] = []
    for shard_index, shard in enumerate(recipe.shards):
        member_records = [record for record in records if record.shard_index == shard_index]
        per_shard.append(
            {
                "shard_id": shard.descriptor.shard_id,
                "shard_revision": shard.descriptor.shard_revision,
                "leaf_id": shard.descriptor.leaf_id,
                "leaf_revision": shard.descriptor.leaf_revision,
                "weight": shard.weight,
                "cap": shard.cap,
                "input_records": len(member_records),
                "accounting": _accounting(member_records),
            }
        )
    emitted_by_leaf = Counter(record.descriptor.leaf_id for record in selected)
    content_identity = {
        "schema_version": REFERENCE_MIXTURE_CONTENT_SCHEMA,
        "documents_jsonl_sha256": document_artifact["sha256"],
        "record_count": len(selected),
    }
    semantic_run = fingerprint_json(
        {
            "canonical_recipe_sha256": canonical_recipe_fingerprint,
            "resolved_recipe_sha256": recipe.fingerprint,
            "content": content_identity,
        }
    )
    return {
        "schema_version": REFERENCE_MIXTURE_REPORT_SCHEMA,
        "recipe_id": recipe.recipe_id,
        "recipe_revision": recipe.revision,
        "canonical_recipe_sha256": canonical_recipe_fingerprint,
        "resolved_recipe_sha256": recipe.fingerprint,
        "semantic_run_sha256": semantic_run,
        "weight_overrides": dict(sorted(weight_overrides.items())),
        "cap_overrides": dict(sorted(cap_overrides.items())),
        "usage": recipe.usage,
        "output_split": recipe.output_split,
        "seed": recipe.seed,
        "target_records": recipe.target_records,
        "emitted_records": len(selected),
        "shortfall_records": recipe.target_records - len(selected),
        "input_records": len(records),
        "input_accounting": accounting,
        "reason_accounting": _reason_accounting(records),
        "per_shard": per_shard,
        "emitted_by_leaf": dict(sorted(emitted_by_leaf.items())),
        "leakage_guard": {
            "policy": "document_id_and_exact_payload_all_opposite_splits",
            "shards": guard_reports,
            "records_scanned": sum(int(item["records_scanned"]) for item in guard_reports),
            "unique_document_ids": None,
            "unique_exact_documents": None,
        },
        "content_identity": content_identity,
        "artifacts": {
            "documents.jsonl": document_artifact,
            "provenance.jsonl": provenance_artifact,
            "resolved_recipe.json": resolved_recipe_artifact,
        },
        "dry_run": dry_run,
        "zero_silent_drops": len(records) == sum(accounting.values()),
    }


def build_reference_mixture(
    recipe: ReferenceMixtureRecipe,
    *,
    recipe_directory: str | Path,
    output_directory: str | Path,
    canonical_recipe_fingerprint: str | None = None,
    weight_overrides: Mapping[str, float] | None = None,
    cap_overrides: Mapping[str, int | None] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Validate, deduplicate, select, and optionally materialize a reference mixture."""

    if not isinstance(recipe, ReferenceMixtureRecipe):
        raise TypeError("recipe must be a ReferenceMixtureRecipe")
    recipe_root = Path(recipe_directory).expanduser().resolve()
    output = Path(output_directory).expanduser()
    if not dry_run and output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    source_fingerprint = _lowercase_sha256(
        canonical_recipe_fingerprint or recipe.fingerprint,
        "canonical_recipe_fingerprint",
    )
    weight_values = dict(weight_overrides or {})
    cap_values = dict(cap_overrides or {})
    descriptors = [item.descriptor for item in recipe.shards] + list(recipe.leakage_guard_shards)
    resolved_paths = [_resolve_shard_path(recipe_root, descriptor) for descriptor in descriptors]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("one physical shard file may appear only once in a recipe")
    mixture_paths = resolved_paths[: len(recipe.shards)]
    guard_paths = resolved_paths[len(recipe.shards) :]

    guard_document_ids: dict[str, tuple[str, int]] = {}
    guard_exact_hashes: dict[str, tuple[str, int]] = {}
    guard_reports = [
        _scan_guard(path, descriptor, guard_document_ids, guard_exact_hashes)
        for path, descriptor in zip(guard_paths, recipe.leakage_guard_shards, strict=True)
    ]
    document_owners: dict[str, _Record] = {}
    exact_owners: dict[str, _Record] = {}
    records: list[_Record] = []
    for shard_index, (shard, path) in enumerate(zip(recipe.shards, mixture_paths, strict=True)):
        records.extend(
            _scan_mixture_shard(
                path,
                shard.descriptor,
                shard_index,
                recipe.seed,
                guard_document_ids,
                guard_exact_hashes,
                document_owners,
                exact_owners,
            )
        )
    _apply_caps(records, recipe)
    selected = _select_records(records, recipe)

    stage: Path | None = None
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        document_hash, document_bytes, document_records = _write_or_hash_rows(
            _document_rows(selected), None if stage is None else stage / "documents.jsonl"
        )
        provenance_hash, provenance_bytes, provenance_records = _write_or_hash_rows(
            _provenance_rows(records), None if stage is None else stage / "provenance.jsonl"
        )
        resolved_bytes = reference_mixture_recipe_json_bytes(recipe)
        resolved_hash = hashlib.sha256(resolved_bytes).hexdigest()
        if stage is not None:
            (stage / "resolved_recipe.json").write_bytes(resolved_bytes)
        report = _report(
            records,
            recipe,
            canonical_recipe_fingerprint=source_fingerprint,
            weight_overrides=weight_values,
            cap_overrides=cap_values,
            guard_reports=guard_reports,
            selected=selected,
            dry_run=dry_run,
            document_artifact=_artifact(document_hash, document_bytes, document_records),
            provenance_artifact=_artifact(provenance_hash, provenance_bytes, provenance_records),
            resolved_recipe_artifact=_artifact(resolved_hash, len(resolved_bytes)),
        )
        report["leakage_guard"]["unique_document_ids"] = len(guard_document_ids)  # type: ignore[index]
        report["leakage_guard"]["unique_exact_documents"] = len(guard_exact_hashes)  # type: ignore[index]
        if not report["zero_silent_drops"]:
            raise AssertionError("internal accounting error left source records unclassified")
        if stage is not None:
            report_bytes = (
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            (stage / "aggregate_report.json").write_bytes(report_bytes)
            sums = {
                "aggregate_report.json": hashlib.sha256(report_bytes).hexdigest(),
                "documents.jsonl": document_hash,
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


def recipes_from_reference_manifest(
    manifest_path: str | Path,
) -> tuple[ReferenceMixtureRecipe, ReferenceMixtureRecipe, ReferenceMixtureRecipe]:
    """Create full-corpus train/validation/test recipes from one pinned v1 manifest."""

    manifest_source = Path(manifest_path).expanduser().resolve()
    if manifest_source.is_symlink() or not manifest_source.is_file():
        raise ValueError("reference manifest must be a regular non-symlink file")
    try:
        raw = json.loads(
            manifest_source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference manifest must be strict UTF-8 JSON") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != REFERENCE_CORPUS_SCHEMA:
        raise ValueError("unsupported business reference manifest")
    shards = raw.get("shards")
    if not isinstance(shards, list):
        raise ValueError("reference manifest shards must be an array")
    descriptors: list[ReferenceShardDescriptor] = []
    for index, value in enumerate(shards):
        if not isinstance(value, Mapping):
            raise ValueError(f"reference manifest shard {index} must be an object")
        expected = {"leaf_id", "leaf_revision", "split", "path", "sha256", "records", "bytes"}
        _require_exact_keys(value, expected, f"manifest.shards[{index}]")
        leaf_id = _identifier(value["leaf_id"], f"manifest.shards[{index}].leaf_id")
        split = _split(value["split"], f"manifest.shards[{index}].split")
        relative = _path(value["path"], f"manifest.shards[{index}].path")
        absolute = (manifest_source.parent / relative).resolve()
        descriptors.append(
            ReferenceShardDescriptor(
                shard_id=f"business-reference-{leaf_id}-{split}-v1",
                shard_revision="1",
                leaf_id=leaf_id,
                leaf_revision=value["leaf_revision"],  # type: ignore[arg-type]
                split=split,
                path=str(absolute),
                sha256=value["sha256"],  # type: ignore[arg-type]
                record_count=value["records"],  # type: ignore[arg-type]
            )
        )
    pairs = [(item.leaf_id, item.split) for item in descriptors]
    if len(pairs) != len(set(pairs)):
        raise ValueError("reference manifest contains duplicate leaf/split shards")
    leaves = {item.leaf_id for item in descriptors}
    expected_pairs = {(leaf, split) for leaf in leaves for split in SPLITS}
    if set(pairs) != expected_pairs:
        raise ValueError("reference manifest must contain every leaf in all three splits")

    results: list[ReferenceMixtureRecipe] = []
    for output_split in ("train", "validation", "test"):
        mixture_descriptors = tuple(item for item in descriptors if item.split == output_split)
        guards = tuple(item for item in descriptors if item.split != output_split)
        results.append(
            ReferenceMixtureRecipe(
                recipe_id=f"business-reference-{output_split}-full-v1",
                revision="1",
                usage="training" if output_split == "train" else "evaluation",
                output_split=output_split,
                seed=f"business-reference-{output_split}-full-seed-v1",
                target_records=sum(item.record_count for item in mixture_descriptors),
                shortfall_policy="error",
                shards=tuple(
                    ReferenceMixtureShard(item, weight=1.0) for item in mixture_descriptors
                ),
                leakage_guard_shards=guards,
            )
        )
    return results[0], results[1], results[2]


def write_recipes_from_reference_manifest(
    manifest_path: str | Path, output_directory: str | Path
) -> dict[str, object]:
    """Atomically materialize canonical full-corpus recipes from a reference manifest."""

    output = Path(output_directory).expanduser()
    if output.exists():
        raise FileExistsError(f"recipe output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        recipes = recipes_from_reference_manifest(manifest_path)
        artifacts: dict[str, dict[str, object]] = {}
        for recipe in recipes:
            name = f"{recipe.output_split}.recipe.json"
            data = reference_mixture_recipe_json_bytes(recipe)
            (stage / name).write_bytes(data)
            artifacts[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "content_sha256": recipe.fingerprint,
                "target_records": recipe.target_records,
            }
        receipt_content = {
            "schema_version": "molor.business-reference-mixture-recipes-receipt.v1",
            "source_manifest": {
                "path": str(Path(manifest_path).expanduser().resolve()),
                "sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            },
            "recipes": artifacts,
        }
        receipt = {**receipt_content, "content_sha256": fingerprint_json(receipt_content)}
        receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (stage / "recipes-receipt.json").write_bytes(receipt_bytes)
        sums = {name: item["sha256"] for name, item in artifacts.items()}
        sums["recipes-receipt.json"] = hashlib.sha256(receipt_bytes).hexdigest()
        (stage / "sha256sums.txt").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
            encoding="utf-8",
        )
        os.replace(stage, output)
        stage = None
        return receipt
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def recipe_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build pinned business-reference recipes")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = write_recipes_from_reference_manifest(args.manifest, args.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0


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
    parser.add_argument(
        "--cap",
        action="append",
        default=[],
        metavar="KIND:ID=COUNT|none",
        help="repeatable leaf or shard cap override; 'none' removes a canonical cap",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    canonical = load_reference_mixture_recipe(args.recipe)
    weights = parse_weight_overrides(args.weight)
    caps = parse_cap_overrides(args.cap)
    resolved = apply_overrides(canonical, weight_overrides=weights, cap_overrides=caps)
    report = build_reference_mixture(
        resolved,
        recipe_directory=args.recipe.parent,
        output_directory=args.output,
        canonical_recipe_fingerprint=canonical.fingerprint,
        weight_overrides=weights,
        cap_overrides=caps,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "REFERENCE_MIXTURE_CONTENT_SCHEMA",
    "REFERENCE_MIXTURE_PROVENANCE_SCHEMA",
    "REFERENCE_MIXTURE_RECIPE_SCHEMA",
    "REFERENCE_MIXTURE_REPORT_SCHEMA",
    "ReferenceMixtureRecipe",
    "ReferenceMixtureShard",
    "ReferenceShardDescriptor",
    "apply_overrides",
    "build_reference_mixture",
    "load_reference_mixture_recipe",
    "main",
    "parse_cap_overrides",
    "parse_weight_overrides",
    "recipe_main",
    "recipes_from_reference_manifest",
    "reference_mixture_recipe_json_bytes",
    "write_recipes_from_reference_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
