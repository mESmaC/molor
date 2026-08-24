"""Strict persisted JSON boundary for base-model manifests.

JSON dictionaries are untrusted input.  This module rejects duplicate keys, unknown fields,
legacy schemas, non-finite numbers, and implicit Python coercions before constructing the
already-validated immutable manifest types.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

from molor.identity import canonical_json_bytes
from molor.manifests import ArtifactDescriptor, BaseModelManifest


DEFAULT_MAXIMUM_MANIFEST_BYTES: Final[int] = 4 * 1024 * 1024

_ARTIFACT_FIELDS: Final[frozenset[str]] = frozenset(
    {"logical_name", "media_type", "sha256", "size_bytes"}
)
_BASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "model_id",
        "requested_revision",
        "resolved_revision",
        "architecture",
        "dtype",
        "quantization",
        "config",
        "weights",
        "weight_index",
        "tokenizer",
        "special_tokens",
        "chat_template",
        "custom_code",
    }
)


class ManifestSerializationError(ValueError):
    """Persisted manifest JSON is malformed or violates the exact schema."""


def _validate_maximum(maximum_bytes: int) -> int:
    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes < 1:
        raise ValueError("maximum_bytes must be a positive integer")
    return maximum_bytes


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestSerializationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ManifestSerializationError("JSON contains a non-finite number")
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ManifestSerializationError(f"JSON contains non-finite constant {value!r}")


def _decode_json(data: str | bytes, maximum_bytes: int) -> Any:
    maximum = _validate_maximum(maximum_bytes)
    if isinstance(data, bytes):
        if len(data) > maximum:
            raise ManifestSerializationError("manifest exceeds the configured byte limit")
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ManifestSerializationError("manifest is not strict UTF-8") from exc
    elif isinstance(data, str):
        try:
            encoded = data.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ManifestSerializationError("manifest is not strict UTF-8") from exc
        if len(encoded) > maximum:
            raise ManifestSerializationError("manifest exceeds the configured byte limit")
        text = data
    else:
        raise TypeError("manifest JSON must be str or bytes")
    try:
        return json.loads(
            text,
            object_pairs_hook=_without_duplicate_keys,
            parse_float=_finite_float,
            parse_constant=_reject_json_constant,
        )
    except ManifestSerializationError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ManifestSerializationError("manifest is not valid strict JSON") from exc


def _exact_object(value: object, fields: frozenset[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestSerializationError(f"{path} must be a JSON object")
    actual = set(value)
    missing = sorted(fields - actual)
    extra = sorted(actual - fields)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing!r}")
        if extra:
            details.append(f"extra={extra!r}")
        raise ManifestSerializationError(f"{path} has an invalid field set ({', '.join(details)})")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ManifestSerializationError(f"{path} must be a string")
    return value


def _artifact(value: object, path: str) -> ArtifactDescriptor:
    record = _exact_object(value, _ARTIFACT_FIELDS, path)
    size_bytes = record["size_bytes"]
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise ManifestSerializationError(f"{path}.size_bytes must be a non-negative integer")
    try:
        return ArtifactDescriptor(
            logical_name=_string(record["logical_name"], f"{path}.logical_name"),
            media_type=_string(record["media_type"], f"{path}.media_type"),
            sha256=_string(record["sha256"], f"{path}.sha256"),
            size_bytes=size_bytes,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestSerializationError(f"{path} is not a valid artifact descriptor") from exc


def _optional_artifact(value: object, path: str) -> ArtifactDescriptor | None:
    if value is None:
        return None
    return _artifact(value, path)


def _artifact_list(
    value: object,
    path: str,
) -> tuple[ArtifactDescriptor, ...]:
    if not isinstance(value, list):
        raise ManifestSerializationError(f"{path} must be a JSON array")
    return tuple(_artifact(item, f"{path}[{index}]") for index, item in enumerate(value))


def loads_base_model_manifest(
    data: str | bytes,
    *,
    maximum_bytes: int = DEFAULT_MAXIMUM_MANIFEST_BYTES,
) -> BaseModelManifest:
    """Parse exact BaseModelManifest v2 JSON from bounded UTF-8 input."""

    decoded = _decode_json(data, maximum_bytes)
    if not isinstance(decoded, dict):
        raise ManifestSerializationError("base manifest must be a JSON object")
    schema_version = decoded.get("schema_version")
    if not isinstance(schema_version, str):
        raise ManifestSerializationError("base manifest schema_version must be a string")
    if schema_version != BaseModelManifest.schema_version:
        raise ManifestSerializationError(
            f"unsupported base manifest schema_version: {schema_version!r}"
        )
    record = _exact_object(decoded, _BASE_FIELDS, "base manifest")
    try:
        return BaseModelManifest(
            model_id=_string(record["model_id"], "base manifest.model_id"),
            requested_revision=_string(
                record["requested_revision"],
                "base manifest.requested_revision",
            ),
            resolved_revision=_string(
                record["resolved_revision"],
                "base manifest.resolved_revision",
            ),
            architecture=_string(record["architecture"], "base manifest.architecture"),
            dtype=_string(record["dtype"], "base manifest.dtype"),
            quantization=_string(
                record["quantization"],
                "base manifest.quantization",
            ),
            config=_artifact(record["config"], "base manifest.config"),
            weights=_artifact_list(record["weights"], "base manifest.weights"),
            tokenizer=_artifact_list(record["tokenizer"], "base manifest.tokenizer"),
            special_tokens=_artifact(
                record["special_tokens"],
                "base manifest.special_tokens",
            ),
            weight_index=_optional_artifact(
                record["weight_index"],
                "base manifest.weight_index",
            ),
            chat_template=_optional_artifact(
                record["chat_template"],
                "base manifest.chat_template",
            ),
            custom_code=_artifact_list(
                record["custom_code"],
                "base manifest.custom_code",
            ),
        )
    except ManifestSerializationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ManifestSerializationError("base manifest violates its typed contract") from exc


def dumps_base_model_manifest(
    manifest: BaseModelManifest,
    *,
    maximum_bytes: int = DEFAULT_MAXIMUM_MANIFEST_BYTES,
) -> bytes:
    """Return the canonical UTF-8 JSON representation used by content identities."""

    if not isinstance(manifest, BaseModelManifest):
        raise TypeError("manifest must be a BaseModelManifest")
    maximum = _validate_maximum(maximum_bytes)
    serialized = canonical_json_bytes(manifest.to_dict())
    if len(serialized) > maximum:
        raise ManifestSerializationError("manifest exceeds the configured byte limit")
    return serialized


def load_base_model_manifest(
    path: str | Path,
    *,
    maximum_bytes: int = DEFAULT_MAXIMUM_MANIFEST_BYTES,
) -> BaseModelManifest:
    """Read at most the configured number of bytes and parse one strict manifest."""

    maximum = _validate_maximum(maximum_bytes)
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"base manifest is not a file: {manifest_path}")
    try:
        with manifest_path.open("rb") as handle:
            data = handle.read(maximum + 1)
    except OSError as exc:
        raise ManifestSerializationError("base manifest could not be read") from exc
    return loads_base_model_manifest(data, maximum_bytes=maximum)


def canonicalize_base_model_manifest_json(
    data: str | bytes,
    *,
    maximum_bytes: int = DEFAULT_MAXIMUM_MANIFEST_BYTES,
) -> bytes:
    """Strictly parse then emit canonical JSON, useful at persistence boundaries."""

    manifest = loads_base_model_manifest(data, maximum_bytes=maximum_bytes)
    return dumps_base_model_manifest(manifest, maximum_bytes=maximum_bytes)
