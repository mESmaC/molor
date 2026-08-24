"""Content-verified local artifact staging for model loading.

The materializer is deliberately unaware of model hubs.  It copies only manifest-declared
bytes into a fresh private directory and verifies the copied stream before any model library
is allowed to inspect it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, ClassVar, Iterable

from molor.identity import fingerprint_json, validate_sha256
from molor.manifests import ArtifactDescriptor, BaseModelManifest


class ArtifactVerificationError(RuntimeError):
    """A materialized artifact differs from its immutable descriptor."""


class WeightIndexValidationError(ArtifactVerificationError):
    """A sharded safetensors index is malformed or does not describe the manifest shards."""


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """Portable verification evidence for one copied artifact.

    Absolute source and staging paths are intentionally excluded from the identity.
    """

    schema_version: ClassVar[str] = "molor.verified-artifact.v1"

    descriptor: ArtifactDescriptor
    observed_sha256: str
    observed_size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ArtifactDescriptor):
            raise TypeError("descriptor must be an ArtifactDescriptor")
        observed_sha256 = validate_sha256(self.observed_sha256, "observed_sha256")
        if (
            not isinstance(self.observed_size_bytes, int)
            or isinstance(self.observed_size_bytes, bool)
            or self.observed_size_bytes < 0
        ):
            raise ValueError("observed_size_bytes must be a non-negative integer")
        if observed_sha256 != self.descriptor.sha256:
            raise ValueError("observed SHA-256 does not match the artifact descriptor")
        if self.observed_size_bytes != self.descriptor.size_bytes:
            raise ValueError("observed size does not match the artifact descriptor")
        object.__setattr__(self, "observed_sha256", observed_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "descriptor_fingerprint": self.descriptor.fingerprint,
            "descriptor": self.descriptor.to_dict(),
            "observed_sha256": self.observed_sha256,
            "observed_size_bytes": self.observed_size_bytes,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class VerifiedArtifactSet:
    """Path-independent identity for every verified artifact in one base manifest."""

    schema_version: ClassVar[str] = "molor.verified-artifact-set.v1"

    base_fingerprint: str
    artifacts: tuple[VerifiedArtifact, ...]
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_fingerprint",
            validate_sha256(self.base_fingerprint, "base_fingerprint"),
        )
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("artifacts must not be empty")
        if any(not isinstance(artifact, VerifiedArtifact) for artifact in artifacts):
            raise TypeError("artifacts must contain only VerifiedArtifact values")
        names = [artifact.descriptor.logical_name for artifact in artifacts]
        if len(names) != len(set(names)):
            raise ValueError("verified artifacts must have unique logical names")
        artifacts = tuple(sorted(artifacts, key=lambda item: item.descriptor.logical_name))
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "_fingerprint", fingerprint_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_fingerprint": self.base_fingerprint,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


_VERIFIED_STAGING_MARKER = object()


class StagedBaseArtifacts:
    """A private staging directory paired with immutable verification evidence.

    Instances are created only by :class:`LocalArtifactMaterializer`.  Call ``close`` after
    the loaded model no longer needs files in the staging directory.
    """

    __slots__ = ("_closed", "_evidence", "_marker", "_root")

    def __init__(
        self,
        root: Path,
        evidence: VerifiedArtifactSet,
        *,
        _marker: object,
    ) -> None:
        if _marker is not _VERIFIED_STAGING_MARKER:
            raise TypeError("StagedBaseArtifacts must be created by LocalArtifactMaterializer")
        self._root = root
        self._evidence = evidence
        self._marker = _marker
        self._closed = False

    @property
    def root(self) -> Path:
        if self._closed:
            raise RuntimeError("staged artifacts have been closed")
        return self._root

    @property
    def evidence(self) -> VerifiedArtifactSet:
        return self._evidence

    @property
    def closed(self) -> bool:
        return self._closed

    def path_for(self, logical_name: str) -> Path:
        if logical_name not in {item.descriptor.logical_name for item in self._evidence.artifacts}:
            raise KeyError(f"artifact was not staged: {logical_name!r}")
        return self.root.joinpath(*PurePosixPath(logical_name).parts)

    def close(self) -> None:
        if self._closed:
            return
        try:
            shutil.rmtree(self._root)
        except FileNotFoundError:
            pass
        self._closed = True

    def __enter__(self) -> StagedBaseArtifacts:
        if self._closed:
            raise RuntimeError("staged artifacts have been closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def base_artifact_descriptors(manifest: BaseModelManifest) -> tuple[ArtifactDescriptor, ...]:
    """Return every byte-bearing base artifact in canonical logical-name order."""

    if not isinstance(manifest, BaseModelManifest):
        raise TypeError("manifest must be a BaseModelManifest")
    descriptors = [
        manifest.config,
        *manifest.weights,
        *manifest.tokenizer,
        manifest.special_tokens,
    ]
    if manifest.weight_index is not None:
        descriptors.append(manifest.weight_index)
    if manifest.chat_template is not None:
        descriptors.append(manifest.chat_template)
    descriptors.extend(manifest.custom_code)
    return tuple(sorted(descriptors, key=lambda item: item.logical_name))


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WeightIndexValidationError(
                f"safetensors weight index contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _validate_index_shard_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise WeightIndexValidationError("weight_map values must be bounded shard names")
    if "\\" in value:
        raise WeightIndexValidationError("weight_map shard names must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise WeightIndexValidationError("weight_map contains an unsafe shard name")
    return str(path)


def validate_safetensors_weight_index(
    path: str | Path,
    *,
    expected_shards: Iterable[str],
    maximum_bytes: int = 16 * 1024 * 1024,
) -> dict[str, str]:
    """Strictly validate an HF sharded-safetensors index against declared shards."""

    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes < 1:
        raise ValueError("maximum_bytes must be a positive integer")
    index_path = Path(path)
    try:
        size = index_path.stat().st_size
    except OSError as exc:
        raise WeightIndexValidationError("weight index is not readable") from exc
    if size > maximum_bytes:
        raise WeightIndexValidationError("weight index exceeds the configured byte limit")
    try:
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            data = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    except WeightIndexValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeightIndexValidationError("weight index is not strict UTF-8 JSON") from exc
    if not isinstance(data, dict) or set(data) != {"metadata", "weight_map"}:
        raise WeightIndexValidationError(
            "weight index must contain exactly metadata and weight_map"
        )
    metadata = data["metadata"]
    if not isinstance(metadata, dict):
        raise WeightIndexValidationError("weight index metadata must be an object")
    total_size = metadata.get("total_size")
    if total_size is not None and (
        not isinstance(total_size, int) or isinstance(total_size, bool) or total_size < 0
    ):
        raise WeightIndexValidationError("metadata.total_size must be non-negative")
    weight_map = data["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise WeightIndexValidationError("weight_map must be a non-empty object")
    normalized_map: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or len(tensor_name) > 1024
            or any(ord(character) < 32 for character in tensor_name)
        ):
            raise WeightIndexValidationError(
                "weight_map tensor names must be bounded non-empty strings"
            )
        normalized_map[tensor_name] = _validate_index_shard_name(shard_name)

    expected = set(expected_shards)
    if not expected or any(not isinstance(item, str) or not item for item in expected):
        raise ValueError("expected_shards must contain non-empty strings")
    referenced = set(normalized_map.values())
    unknown = referenced - expected
    omitted = expected - referenced
    if unknown:
        raise WeightIndexValidationError(
            f"weight index references undeclared shards: {sorted(unknown)!r}"
        )
    if omitted:
        raise WeightIndexValidationError(f"weight index omits declared shards: {sorted(omitted)!r}")
    return normalized_map


@dataclass(frozen=True, slots=True)
class LocalArtifactMaterializer:
    """Copy and verify one manifest from a caller-selected local source directory."""

    chunk_size: int = 1024 * 1024
    maximum_index_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for field_name in ("chunk_size", "maximum_index_bytes"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")

    def materialize(
        self,
        manifest: BaseModelManifest,
        source_root: str | Path,
        *,
        staging_parent: str | Path | None = None,
    ) -> StagedBaseArtifacts:
        """Stage only declared artifacts, rejecting any observed byte mismatch."""

        if not isinstance(manifest, BaseModelManifest):
            raise TypeError("manifest must be a BaseModelManifest")
        source = Path(source_root)
        if not source.is_dir():
            raise FileNotFoundError(f"artifact source is not a directory: {source}")
        parent = None if staging_parent is None else Path(staging_parent)
        if parent is not None and not parent.is_dir():
            raise FileNotFoundError(f"staging parent is not a directory: {parent}")
        root = Path(
            tempfile.mkdtemp(
                prefix=f"molor-base-{manifest.fingerprint[:12]}-",
                dir=None if parent is None else str(parent),
            )
        )
        try:
            os.chmod(root, 0o700)
            verified = tuple(
                self._copy_verified(source, root, descriptor)
                for descriptor in base_artifact_descriptors(manifest)
            )
            if manifest.weight_index is not None:
                validate_safetensors_weight_index(
                    root.joinpath(*PurePosixPath(manifest.weight_index.logical_name).parts),
                    expected_shards=(item.logical_name for item in manifest.weights),
                    maximum_bytes=self.maximum_index_bytes,
                )
            evidence = VerifiedArtifactSet(manifest.fingerprint, verified)
            return StagedBaseArtifacts(
                root,
                evidence,
                _marker=_VERIFIED_STAGING_MARKER,
            )
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise

    def _copy_verified(
        self,
        source_root: Path,
        staging_root: Path,
        descriptor: ArtifactDescriptor,
    ) -> VerifiedArtifact:
        relative = PurePosixPath(descriptor.logical_name)
        source_path = source_root.joinpath(*relative.parts)
        destination = staging_root.joinpath(*relative.parts)
        if not source_path.is_file():
            raise ArtifactVerificationError(
                f"declared artifact is missing or not a file: {descriptor.logical_name!r}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        observed_size = 0
        try:
            with source_path.open("rb") as source_handle, destination.open("xb") as output:
                while chunk := source_handle.read(self.chunk_size):
                    observed_size += len(chunk)
                    if observed_size > descriptor.size_bytes:
                        raise ArtifactVerificationError(
                            f"artifact size differs from descriptor: {descriptor.logical_name!r}"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except ArtifactVerificationError:
            raise
        except OSError as exc:
            raise ArtifactVerificationError(
                f"failed to copy artifact: {descriptor.logical_name!r}"
            ) from exc
        observed_sha256 = digest.hexdigest()
        if observed_size != descriptor.size_bytes:
            raise ArtifactVerificationError(
                f"artifact size differs from descriptor: {descriptor.logical_name!r}"
            )
        if observed_sha256 != descriptor.sha256:
            raise ArtifactVerificationError(
                f"artifact SHA-256 differs from descriptor: {descriptor.logical_name!r}"
            )
        return VerifiedArtifact(descriptor, observed_sha256, observed_size)
