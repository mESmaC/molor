"""Canonical content identity helpers used by manifests and reward artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_json_value(value: Any, path: str = "value") -> None:
    """Reject values that cannot participate in the project's canonical JSON contract."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_json_value(child, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            validate_json_value(child, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def freeze_json(value: Any) -> Any:
    """Recursively copy JSON data into immutable mapping/tuple containers."""

    validate_json_value(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(child) for child in value)
    return value


def thaw_json(value: Any) -> Any:
    """Return ordinary dict/list containers suitable for JSON serialization."""

    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON data deterministically for content addressing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint_json(value: Any) -> str:
    """Return a lowercase SHA-256 digest of canonical JSON data."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_sha256(value: str, field_name: str = "fingerprint") -> str:
    """Validate and normalize a bare hexadecimal SHA-256 digest."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be exactly 64 hexadecimal characters")
    return normalized


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one artifact without reading the whole file into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact is not a file: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
