"""Strict, content-addressed presets for reproducible LoRA mixtures.

This module is intentionally isolated from the live teacher and desktop runtimes.  It
defines an immutable artifact that future serving and benchmark integrations can resolve
without accepting request-supplied paths or ambiguous route semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, ClassVar

from molor.identity import fingerprint_json, validate_sha256


MIXTURE_PRESET_SCHEMA = "molor.mixture-preset.v1"
MIXTURE_PRESET_REGISTRY_SCHEMA = "molor.mixture-preset-registry.v1"
MIXTURE_NORMALIZATION = "leaf_weights_sum_to_one"
MIXTURE_COMPOSITION = "frozen_base_plus_weighted_lora_deltas"
DEFAULT_MAXIMUM_PRESET_BYTES = 1024 * 1024
DEFAULT_MAXIMUM_PRESETS = 1024
MAXIMUM_ACTIVE_LEAVES = 64

_PRESET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LEAF_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_RESERVED_LEAF_IDS = frozenset(("base", "base/null"))


def _preset_id(value: object, field_name: str = "preset_id") -> str:
    if not isinstance(value, str) or _PRESET_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 1-128 characters using letters, digits, '.', '_', or '-'"
        )
    return value


def _revision(value: object) -> str:
    if not isinstance(value, str) or _PRESET_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "revision must be 1-128 characters using letters, digits, '.', '_', or '-'"
        )
    return value


def _leaf_id(value: object, field_name: str = "leaf_id") -> str:
    if not isinstance(value, str) or _LEAF_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 1-128 characters using letters, digits, '.', '_', ':', '/', "
            "or '-'"
        )
    if ".." in PurePosixPath(value.replace("\\", "/")).parts:
        raise ValueError(f"{field_name} must not contain a parent-directory component")
    if value in _RESERVED_LEAF_IDS:
        raise ValueError(f"{field_name} must identify an adapter leaf, not {value!r}")
    return value


def _finite_positive_weight(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    try:
        weight = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be finite and positive") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return weight


def _positive_limit(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class MixtureLeafWeight:
    """One exact adapter-leaf coefficient in a normalized preset."""

    leaf_id: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "leaf_id", _leaf_id(self.leaf_id))
        object.__setattr__(
            self,
            "weight",
            _finite_positive_weight(self.weight, f"weight[{self.leaf_id!r}]"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"leaf_id": self.leaf_id, "weight": self.weight}


@dataclass(frozen=True, slots=True)
class MixturePreset:
    """An immutable, rack-bound mixture of independently named adapter leaves.

    The base model is always evaluated at full strength.  The listed weights apply only
    to LoRA deltas and must already sum to one; loaders never silently renormalize them.
    """

    schema_version: ClassVar[str] = MIXTURE_PRESET_SCHEMA

    preset_id: str
    revision: str
    rack_fingerprint: str
    leaves: tuple[MixtureLeafWeight, ...]
    normalization: str = MIXTURE_NORMALIZATION
    composition: str = MIXTURE_COMPOSITION
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        preset_id = _preset_id(self.preset_id)
        revision = _revision(self.revision)
        rack_fingerprint = validate_sha256(self.rack_fingerprint, "rack_fingerprint")
        if self.normalization != MIXTURE_NORMALIZATION:
            raise ValueError(f"normalization must be {MIXTURE_NORMALIZATION!r}")
        if self.composition != MIXTURE_COMPOSITION:
            raise ValueError(f"composition must be {MIXTURE_COMPOSITION!r}")
        if isinstance(self.leaves, (str, bytes)) or not isinstance(self.leaves, Sequence):
            raise TypeError("leaves must be a sequence of MixtureLeafWeight values")
        leaves = tuple(self.leaves)
        if not leaves:
            raise ValueError("a mixture preset must contain at least one adapter leaf")
        if len(leaves) > MAXIMUM_ACTIVE_LEAVES:
            raise ValueError(
                f"a mixture preset must not contain more than {MAXIMUM_ACTIVE_LEAVES} leaves"
            )
        if any(not isinstance(leaf, MixtureLeafWeight) for leaf in leaves):
            raise TypeError("leaves must contain only MixtureLeafWeight values")
        leaf_ids = [leaf.leaf_id for leaf in leaves]
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("a mixture preset must not repeat a leaf ID")
        total = math.fsum(leaf.weight for leaf in leaves)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"leaf weights must sum to one, got {total:.17g}")

        canonical_leaves = tuple(sorted(leaves, key=lambda leaf: leaf.leaf_id))
        object.__setattr__(self, "preset_id", preset_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "rack_fingerprint", rack_fingerprint)
        object.__setattr__(self, "leaves", canonical_leaves)
        object.__setattr__(self, "_fingerprint", fingerprint_json(self.content_dict()))

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the canonical semantic payload, excluding its own digest."""

        return self._fingerprint

    @property
    def route(self) -> Mapping[str, float]:
        """Return an immutable leaf-to-weight mapping suitable for later resolution."""

        return MappingProxyType({leaf.leaf_id: leaf.weight for leaf in self.leaves})

    def content_dict(self) -> dict[str, object]:
        """Return the exact payload covered by :attr:`fingerprint`."""

        return {
            "schema_version": self.schema_version,
            "preset_id": self.preset_id,
            "revision": self.revision,
            "rack_fingerprint": self.rack_fingerprint,
            "normalization": self.normalization,
            "composition": self.composition,
            "leaves": [leaf.to_dict() for leaf in self.leaves],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the self-authenticating serialized representation."""

        return {**self.content_dict(), "content_sha256": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MixturePreset:
        """Parse one exact v1 payload and verify its claimed content digest."""

        if not isinstance(value, Mapping):
            raise TypeError("mixture preset must be an object")
        expected_keys = {
            "schema_version",
            "preset_id",
            "revision",
            "rack_fingerprint",
            "normalization",
            "composition",
            "leaves",
            "content_sha256",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys.difference(value))
            unknown = sorted(set(value).difference(expected_keys))
            raise ValueError(
                f"mixture preset fields differ from v1 (missing={missing!r}, unknown={unknown!r})"
            )
        if value.get("schema_version") != MIXTURE_PRESET_SCHEMA:
            raise ValueError("unsupported mixture preset schema")
        raw_leaves = value.get("leaves")
        if not isinstance(raw_leaves, list):
            raise TypeError("mixture preset leaves must be an array")
        parsed_leaves: list[MixtureLeafWeight] = []
        for index, raw_leaf in enumerate(raw_leaves):
            if not isinstance(raw_leaf, Mapping) or set(raw_leaf) != {"leaf_id", "weight"}:
                raise ValueError(f"leaves[{index}] must contain exactly 'leaf_id' and 'weight'")
            parsed_leaves.append(
                MixtureLeafWeight(
                    leaf_id=raw_leaf["leaf_id"],
                    weight=raw_leaf["weight"],
                )
            )
        preset = cls(
            preset_id=value["preset_id"],
            revision=value["revision"],
            rack_fingerprint=value["rack_fingerprint"],
            leaves=tuple(parsed_leaves),
            normalization=value["normalization"],
            composition=value["composition"],
        )
        if raw_leaves != preset.content_dict()["leaves"]:
            raise ValueError("mixture preset leaves must be ordered canonically by leaf_id")
        claimed = validate_sha256(value["content_sha256"], "content_sha256")
        if value["content_sha256"] != claimed:
            raise ValueError("content_sha256 must use lowercase hexadecimal")
        if claimed != preset.fingerprint:
            raise ValueError("mixture preset content_sha256 does not match its semantic payload")
        return preset


@dataclass(frozen=True, slots=True)
class MixturePresetRegistry:
    """A validated set of named presets for one exact resident expert rack."""

    schema_version: ClassVar[str] = MIXTURE_PRESET_REGISTRY_SCHEMA

    rack_fingerprint: str
    leaf_ids: tuple[str, ...]
    presets: tuple[MixturePreset, ...]
    _by_id: Mapping[str, MixturePreset] = field(init=False, repr=False, compare=False)
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rack_fingerprint = validate_sha256(self.rack_fingerprint, "rack_fingerprint")
        if isinstance(self.leaf_ids, (str, bytes)) or not isinstance(self.leaf_ids, Sequence):
            raise TypeError("leaf_ids must be the exact ordered rack leaf sequence")
        leaf_ids = tuple(
            _leaf_id(leaf_id, f"leaf_ids[{index}]") for index, leaf_id in enumerate(self.leaf_ids)
        )
        if not leaf_ids:
            raise ValueError("leaf_ids must contain at least one rack leaf")
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("leaf_ids must not contain duplicates")
        if isinstance(self.presets, (str, bytes)) or not isinstance(self.presets, Sequence):
            raise TypeError("presets must be a sequence of MixturePreset values")
        presets = tuple(self.presets)
        if not presets:
            raise ValueError("a mixture preset registry must contain at least one preset")
        if len(presets) > DEFAULT_MAXIMUM_PRESETS:
            raise ValueError(
                f"a mixture preset registry must not exceed {DEFAULT_MAXIMUM_PRESETS} presets"
            )
        if any(not isinstance(preset, MixturePreset) for preset in presets):
            raise TypeError("presets must contain only MixturePreset values")
        preset_ids = [preset.preset_id for preset in presets]
        if len(preset_ids) != len(set(preset_ids)):
            raise ValueError("a registry must not contain duplicate preset IDs")
        known_leaves = set(leaf_ids)
        for preset in presets:
            if preset.rack_fingerprint != rack_fingerprint:
                raise ValueError(f"preset {preset.preset_id!r} belongs to a different expert rack")
            unknown = set(preset.route).difference(known_leaves)
            if unknown:
                raise ValueError(
                    f"preset {preset.preset_id!r} references unknown rack leaves: "
                    f"{sorted(unknown)!r}"
                )
        canonical_presets = tuple(sorted(presets, key=lambda preset: preset.preset_id))
        by_id = MappingProxyType({preset.preset_id: preset for preset in canonical_presets})
        object.__setattr__(self, "rack_fingerprint", rack_fingerprint)
        object.__setattr__(self, "leaf_ids", leaf_ids)
        object.__setattr__(self, "presets", canonical_presets)
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_fingerprint", fingerprint_json(self.content_dict()))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def resolve(
        self,
        preset_id: str,
        *,
        expected_fingerprint: str | None = None,
    ) -> MixturePreset:
        """Resolve by stable ID and optionally require one exact preset revision."""

        normalized_id = _preset_id(preset_id)
        try:
            preset = self._by_id[normalized_id]
        except KeyError as exc:
            raise KeyError(f"unknown mixture preset: {normalized_id!r}") from exc
        if expected_fingerprint is not None:
            expected = validate_sha256(expected_fingerprint, "expected_fingerprint")
            if preset.fingerprint != expected:
                raise ValueError(
                    f"mixture preset {normalized_id!r} does not match the expected fingerprint"
                )
        return preset

    def content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rack_fingerprint": self.rack_fingerprint,
            "leaf_ids": list(self.leaf_ids),
            "presets": [
                {
                    "preset_id": preset.preset_id,
                    "revision": preset.revision,
                    "content_sha256": preset.fingerprint,
                }
                for preset in self.presets
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_dict(), "content_sha256": self.fingerprint}

    @classmethod
    def load_directory(
        cls,
        directory: str | Path,
        *,
        rack_fingerprint: str,
        leaf_ids: Sequence[str],
        expected_registry_fingerprint: str | None = None,
        expected_presets: Mapping[str, str] | None = None,
        maximum_presets: int = DEFAULT_MAXIMUM_PRESETS,
        maximum_preset_bytes: int = DEFAULT_MAXIMUM_PRESET_BYTES,
    ) -> MixturePresetRegistry:
        """Load a closed directory of ``<preset_id>.json`` artifacts.

        Every entry must be a regular JSON file.  Callers may pin the entire registry
        fingerprint and/or the exact ID-to-preset fingerprint mapping.
        """

        maximum_presets = _positive_limit(maximum_presets, "maximum_presets")
        maximum_preset_bytes = _positive_limit(maximum_preset_bytes, "maximum_preset_bytes")
        source = Path(directory).expanduser()
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"mixture preset registry must be a regular directory: {source}")
        root = source.resolve()
        before = root.stat()
        entries = sorted(root.iterdir(), key=lambda path: path.name)
        after = root.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("mixture preset registry changed while it was enumerated")
        if not entries:
            raise ValueError("mixture preset registry directory is empty")
        if len(entries) > maximum_presets:
            raise ValueError(f"mixture preset registry exceeds its {maximum_presets}-file limit")
        presets: list[MixturePreset] = []
        for entry in entries:
            if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
                raise ValueError(
                    "mixture preset registry may contain only regular lowercase .json files"
                )
            preset = load_mixture_preset(entry, maximum_bytes=maximum_preset_bytes)
            if entry.name != f"{preset.preset_id}.json":
                raise ValueError(
                    f"preset filename must match its ID exactly: expected {preset.preset_id}.json"
                )
            presets.append(preset)
        registry = cls(
            rack_fingerprint=rack_fingerprint,
            leaf_ids=tuple(leaf_ids),
            presets=tuple(presets),
        )
        if expected_presets is not None:
            if not isinstance(expected_presets, Mapping):
                raise TypeError("expected_presets must map preset IDs to SHA-256 fingerprints")
            expected = {
                _preset_id(preset_id, "expected_presets key"): validate_sha256(
                    fingerprint,
                    f"expected_presets[{preset_id!r}]",
                )
                for preset_id, fingerprint in expected_presets.items()
            }
            observed = {preset.preset_id: preset.fingerprint for preset in registry.presets}
            if observed != expected:
                raise ValueError("loaded mixture presets differ from the expected registry set")
        if expected_registry_fingerprint is not None:
            expected_registry = validate_sha256(
                expected_registry_fingerprint,
                "expected_registry_fingerprint",
            )
            if registry.fingerprint != expected_registry:
                raise ValueError("mixture preset registry fingerprint mismatch")
        return registry


def mixture_preset_json_bytes(preset: MixturePreset) -> bytes:
    """Return deterministic, human-readable UTF-8 bytes for one preset artifact."""

    if not isinstance(preset, MixturePreset):
        raise TypeError("preset must be a MixturePreset")
    return (
        json.dumps(
            preset.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"JSON contains unsupported numeric constant {value!r}")


def load_mixture_preset(
    path: str | Path,
    *,
    maximum_bytes: int = DEFAULT_MAXIMUM_PRESET_BYTES,
) -> MixturePreset:
    """Load one bounded, regular, stable UTF-8 JSON artifact and verify its digest."""

    maximum_bytes = _positive_limit(maximum_bytes, "maximum_bytes")
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"mixture preset must be a regular non-symlink file: {source}")
    before = source.stat()
    if before.st_size < 1:
        raise ValueError("mixture preset must not be empty")
    if before.st_size > maximum_bytes:
        raise ValueError(f"mixture preset exceeds its {maximum_bytes}-byte limit")
    raw = source.read_bytes()
    after = source.stat()
    if (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ino", None),
    ) != (
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ino", None),
    ):
        raise RuntimeError("mixture preset changed while it was loaded")
    if len(raw) != before.st_size:
        raise RuntimeError("mixture preset size changed while it was loaded")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("mixture preset must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("mixture preset must contain one JSON object")
    return MixturePreset.from_dict(payload)


__all__ = [
    "DEFAULT_MAXIMUM_PRESET_BYTES",
    "DEFAULT_MAXIMUM_PRESETS",
    "MAXIMUM_ACTIVE_LEAVES",
    "MIXTURE_COMPOSITION",
    "MIXTURE_NORMALIZATION",
    "MIXTURE_PRESET_REGISTRY_SCHEMA",
    "MIXTURE_PRESET_SCHEMA",
    "MixtureLeafWeight",
    "MixturePreset",
    "MixturePresetRegistry",
    "load_mixture_preset",
    "mixture_preset_json_bytes",
]
