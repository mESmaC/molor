"""Content-addressed identities for a frozen base and whole-model LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, ClassVar, Iterable

from molor.config import LoRABankConfig, RouterConfig
from molor.identity import fingerprint_json, sha256_file, validate_sha256

SAFETENSORS_MEDIA_TYPE = "application/vnd.safetensors"
JSON_MEDIA_TYPE = "application/json"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _validate_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 1-128 characters using letters, digits, '.', '_', ':', '/', or '-'"
        )
    if ".." in PurePosixPath(value.replace("\\", "/")).parts:
        raise ValueError(f"{field_name} must not contain a parent-directory component")
    return value


def _nonempty(value: str, field_name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{field_name} must not exceed {maximum} characters")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Portable content descriptor; paths are labels and are never resolved automatically."""

    schema_version: ClassVar[str] = "molor.artifact.v1"

    logical_name: str
    media_type: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _nonempty(self.logical_name, "logical_name", maximum=256)
        normalized_name = self.logical_name.replace("\\", "/")
        path = PurePosixPath(normalized_name)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("logical_name must be a normalized relative artifact name")
        if ":" in path.parts[0]:
            raise ValueError("logical_name must not contain a drive or URI scheme")
        _nonempty(self.media_type, "media_type", maximum=128)
        if any(character.isspace() or ord(character) < 32 for character in self.media_type):
            raise ValueError("media_type must not contain whitespace or control characters")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("size_bytes must be a non-negative integer")
        object.__setattr__(self, "logical_name", str(path))
        object.__setattr__(self, "sha256", validate_sha256(self.sha256, "sha256"))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        logical_name: str,
        media_type: str,
    ) -> ArtifactDescriptor:
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(f"artifact is not a file: {artifact}")
        return cls(
            logical_name=logical_name,
            media_type=media_type,
            sha256=sha256_file(artifact),
            size_bytes=artifact.stat().st_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json({"schema_version": self.schema_version, **self.to_dict()})


def _artifact_tuple(
    values: Iterable[ArtifactDescriptor],
    field_name: str,
    *,
    required: bool = True,
) -> tuple[ArtifactDescriptor, ...]:
    artifacts = tuple(values)
    if required and not artifacts:
        raise ValueError(f"{field_name} must contain at least one artifact")
    if any(not isinstance(item, ArtifactDescriptor) for item in artifacts):
        raise TypeError(f"{field_name} must contain only ArtifactDescriptor values")
    names = [item.logical_name for item in artifacts]
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} contains duplicate logical artifact names")
    return tuple(sorted(artifacts, key=lambda item: item.logical_name))


@dataclass(frozen=True, slots=True)
class BaseModelManifest:
    """Exact checkpoint, tokenizer, template, and runtime contract shared by a bank."""

    schema_version: ClassVar[str] = "molor.base-model.v2"

    model_id: str
    requested_revision: str
    resolved_revision: str
    architecture: str
    dtype: str
    quantization: str
    config: ArtifactDescriptor
    weights: tuple[ArtifactDescriptor, ...]
    tokenizer: tuple[ArtifactDescriptor, ...]
    special_tokens: ArtifactDescriptor
    weight_index: ArtifactDescriptor | None = None
    chat_template: ArtifactDescriptor | None = None
    custom_code: tuple[ArtifactDescriptor, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.model_id, "model_id")
        _nonempty(self.requested_revision, "requested_revision")
        _nonempty(self.resolved_revision, "resolved_revision")
        normalized_revision = self.resolved_revision.lower()
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", normalized_revision) is None:
            raise ValueError("resolved_revision must be an immutable 40- or 64-hex digest")
        _nonempty(self.architecture, "architecture")
        _nonempty(self.dtype, "dtype")
        normalized_dtype = self.dtype.lower().removeprefix("torch.")
        if normalized_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        _nonempty(self.quantization, "quantization")
        normalized_quantization = self.quantization.strip().lower()
        if not isinstance(self.config, ArtifactDescriptor):
            raise TypeError("config must be an ArtifactDescriptor")
        if not isinstance(self.special_tokens, ArtifactDescriptor):
            raise TypeError("special_tokens must be an ArtifactDescriptor")
        if self.weight_index is not None and not isinstance(
            self.weight_index,
            ArtifactDescriptor,
        ):
            raise TypeError("weight_index must be an ArtifactDescriptor or None")
        if self.chat_template is not None and not isinstance(
            self.chat_template, ArtifactDescriptor
        ):
            raise TypeError("chat_template must be an ArtifactDescriptor or None")
        object.__setattr__(self, "resolved_revision", normalized_revision)
        object.__setattr__(self, "dtype", normalized_dtype)
        object.__setattr__(self, "quantization", normalized_quantization)
        object.__setattr__(self, "weights", _artifact_tuple(self.weights, "weights"))
        object.__setattr__(self, "tokenizer", _artifact_tuple(self.tokenizer, "tokenizer"))
        object.__setattr__(
            self,
            "custom_code",
            _artifact_tuple(self.custom_code, "custom_code", required=False),
        )
        if any(artifact.media_type != SAFETENSORS_MEDIA_TYPE for artifact in self.weights):
            raise ValueError("base weights must use the safetensors media type")
        if self.config.media_type != JSON_MEDIA_TYPE:
            raise ValueError("base config must use the JSON media type")
        if self.special_tokens.media_type != JSON_MEDIA_TYPE:
            raise ValueError("special_tokens must use the JSON media type")
        if self.weight_index is not None:
            if self.weight_index.media_type != JSON_MEDIA_TYPE:
                raise ValueError("weight_index must use the JSON media type")
            if not self.weight_index.logical_name.endswith(".safetensors.index.json"):
                raise ValueError("weight_index must name a safetensors index JSON file")
        if len(self.weights) > 1 and self.weight_index is None:
            raise ValueError("sharded base weights require an explicit weight_index")

        all_artifacts = (
            self.config,
            *self.weights,
            *self.tokenizer,
            self.special_tokens,
            *((self.weight_index,) if self.weight_index is not None else ()),
            *((self.chat_template,) if self.chat_template is not None else ()),
            *self.custom_code,
        )
        names = [artifact.logical_name for artifact in all_artifacts]
        if len(names) != len(set(names)):
            raise ValueError("base manifest contains duplicate logical artifact names")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "architecture": self.architecture,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "config": self.config.to_dict(),
            "weights": [artifact.to_dict() for artifact in self.weights],
            "weight_index": (None if self.weight_index is None else self.weight_index.to_dict()),
            "tokenizer": [artifact.to_dict() for artifact in self.tokenizer],
            "special_tokens": self.special_tokens.to_dict(),
            "chat_template": (None if self.chat_template is None else self.chat_template.to_dict()),
            "custom_code": [artifact.to_dict() for artifact in self.custom_code],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class TargetModuleManifest:
    """Exact adapted projection shape used to validate one runtime LoRA bank."""

    module_name: str
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        _nonempty(self.module_name, "module_name")
        for field_name in ("in_features", "out_features"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }


@dataclass(frozen=True, slots=True)
class LoRAFormatManifest:
    """Whole-adapter structural contract required for exact vectorized composition."""

    schema_version: ClassVar[str] = "molor.lora-format.v1"

    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    tensor_schema_sha256: str
    module_shapes: tuple[TargetModuleManifest, ...] = ()
    dtype: str = "bfloat16"
    bias: str = "none"
    use_rslora: bool = False
    use_dora: bool = False
    fan_in_fan_out: bool = False
    modules_to_save: tuple[str, ...] = ()
    format_name: str = "peft"
    format_version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be a real number")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a real number")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if isinstance(self.target_modules, (str, bytes)):
            raise TypeError("target_modules must be an iterable of module names")
        target_modules = tuple(self.target_modules)
        if not target_modules:
            raise ValueError("target_modules must not be empty")
        if any(not isinstance(module, str) or not module for module in target_modules):
            raise ValueError("target_modules must contain non-empty strings")
        if len(target_modules) != len(set(target_modules)):
            raise ValueError("target_modules must not contain duplicates")
        if isinstance(self.module_shapes, (str, bytes)):
            raise TypeError("module_shapes must be an iterable of TargetModuleManifest values")
        module_shapes = tuple(self.module_shapes)
        if any(not isinstance(module, TargetModuleManifest) for module in module_shapes):
            raise TypeError("module_shapes must contain only TargetModuleManifest values")
        module_names = [module.module_name for module in module_shapes]
        if len(module_names) != len(set(module_names)):
            raise ValueError("module_shapes must not contain duplicate module names")
        unmatched_shapes = [
            module.module_name
            for module in module_shapes
            if not any(
                module.module_name == target or module.module_name.endswith(f".{target}")
                for target in target_modules
            )
        ]
        if unmatched_shapes:
            raise ValueError(
                f"module_shapes contain names outside target_modules: {unmatched_shapes!r}"
            )
        if isinstance(self.modules_to_save, (str, bytes)):
            raise TypeError("modules_to_save must be an iterable of module names")
        modules_to_save = tuple(self.modules_to_save)
        if any(not isinstance(module, str) or not module for module in modules_to_save):
            raise ValueError("modules_to_save must contain non-empty strings")
        if len(modules_to_save) != len(set(modules_to_save)):
            raise ValueError("modules_to_save must not contain duplicates")
        _nonempty(self.dtype, "dtype")
        _nonempty(self.bias, "bias")
        _nonempty(self.format_name, "format_name")
        _nonempty(self.format_version, "format_version")
        for field_name in ("use_rslora", "use_dora", "fan_in_fan_out"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")
        object.__setattr__(
            self,
            "tensor_schema_sha256",
            validate_sha256(self.tensor_schema_sha256, "tensor_schema_sha256"),
        )
        normalized_dtype = self.dtype.lower().removeprefix("torch.")
        if normalized_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "dtype", normalized_dtype)
        object.__setattr__(self, "target_modules", tuple(sorted(target_modules)))
        object.__setattr__(
            self,
            "module_shapes",
            tuple(sorted(module_shapes, key=lambda module: module.module_name)),
        )
        object.__setattr__(self, "modules_to_save", tuple(sorted(modules_to_save)))

    @property
    def scaling_convention(self) -> str:
        return "alpha_over_sqrt_rank" if self.use_rslora else "alpha_over_rank"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "tensor_schema_sha256": self.tensor_schema_sha256,
            "module_shapes": [module.to_dict() for module in self.module_shapes],
            "dtype": self.dtype,
            "bias": self.bias,
            "scaling_convention": self.scaling_convention,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
            "fan_in_fan_out": self.fan_in_fan_out,
            "modules_to_save": list(self.modules_to_save),
            "format_name": self.format_name,
            "format_version": self.format_version,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    """One independently trained whole-model adapter with a stable logical ID."""

    schema_version: ClassVar[str] = "molor.adapter.v1"

    adapter_id: str
    revision: str
    base: BaseModelManifest
    lora: LoRAFormatManifest
    weights: tuple[ArtifactDescriptor, ...]
    peft_config: ArtifactDescriptor
    training_run_sha256: str
    library_versions: tuple[tuple[str, str], ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.base, BaseModelManifest):
            raise TypeError("base must be a BaseModelManifest")
        if not isinstance(self.lora, LoRAFormatManifest):
            raise TypeError("lora must be a LoRAFormatManifest")
        if not isinstance(self.peft_config, ArtifactDescriptor):
            raise TypeError("peft_config must be an ArtifactDescriptor")
        _validate_identifier(self.adapter_id, "adapter_id")
        _nonempty(self.revision, "revision")
        weights = _artifact_tuple(self.weights, "weights")
        if any(artifact.media_type != SAFETENSORS_MEDIA_TYPE for artifact in weights):
            raise ValueError("adapter weights must use the safetensors media type")
        if self.peft_config.media_type != JSON_MEDIA_TYPE:
            raise ValueError("peft_config must use the JSON media type")
        if self.peft_config.logical_name in {artifact.logical_name for artifact in weights}:
            raise ValueError("adapter artifacts must have unique logical names")
        if isinstance(self.library_versions, (str, bytes)):
            raise TypeError("library_versions must be an iterable of pairs")
        raw_versions = tuple(self.library_versions)
        if any(not isinstance(pair, (tuple, list)) for pair in raw_versions):
            raise TypeError("library_versions must contain tuple or list pairs")
        versions = tuple(tuple(pair) for pair in raw_versions)
        if any(
            len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or not isinstance(pair[1], str)
            or not pair[1]
            for pair in versions
        ):
            raise ValueError("library_versions must contain non-empty (name, version) pairs")
        version_names = [pair[0] for pair in versions]
        if len(version_names) != len(set(version_names)):
            raise ValueError("library_versions must not repeat a library name")
        if isinstance(self.tags, (str, bytes)):
            raise TypeError("tags must be an iterable of tag strings")
        tags = tuple(self.tags)
        if any(not isinstance(tag, str) or not tag for tag in tags):
            raise ValueError("tags must contain non-empty strings")
        if len(tags) != len(set(tags)):
            raise ValueError("tags must not contain duplicates")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "training_run_sha256",
            validate_sha256(self.training_run_sha256, "training_run_sha256"),
        )
        object.__setattr__(self, "library_versions", tuple(sorted(versions)))
        object.__setattr__(self, "tags", tuple(sorted(tags)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "revision": self.revision,
            "base_fingerprint": self.base.fingerprint,
            "base": self.base.to_dict(),
            "lora": self.lora.to_dict(),
            "weights": [artifact.to_dict() for artifact in self.weights],
            "peft_config": self.peft_config.to_dict(),
            "training_run_sha256": self.training_run_sha256,
            "library_versions": [list(pair) for pair in self.library_versions],
            "tags": list(self.tags),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RouteManifest:
    """Stable logical route plus its bank-local runtime index."""

    route_id: str
    route_index: int
    adapter_fingerprint: str | None
    is_null: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.route_id, "route_id")
        if (
            not isinstance(self.route_index, int)
            or isinstance(self.route_index, bool)
            or self.route_index < 0
        ):
            raise ValueError("route_index must be a non-negative integer")
        if not isinstance(self.is_null, bool):
            raise TypeError("is_null must be boolean")
        if self.is_null:
            if self.adapter_fingerprint is not None:
                raise ValueError("null route must not have an adapter fingerprint")
        elif self.adapter_fingerprint is None:
            raise ValueError("non-null route requires an adapter fingerprint")
        else:
            object.__setattr__(
                self,
                "adapter_fingerprint",
                validate_sha256(self.adapter_fingerprint, "adapter_fingerprint"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "route_index": self.route_index,
            "adapter_fingerprint": self.adapter_fingerprint,
            "is_null": self.is_null,
        }


@dataclass(frozen=True, slots=True)
class AdapterBankManifest:
    """Ordered adapter bank whose fingerprint guards every positional route index."""

    schema_version: ClassVar[str] = "molor.adapter-bank.v1"

    bank_id: str
    revision: str
    adapters: tuple[AdapterManifest, ...]
    include_null: bool = True
    null_route_id: str = "base/null"
    _routes: tuple[RouteManifest, ...] = field(init=False, repr=False, compare=False)
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_identifier(self.bank_id, "bank_id")
        _nonempty(self.revision, "revision")
        adapters = tuple(self.adapters)
        if not adapters:
            raise ValueError("adapter bank requires at least one adapter")
        if any(not isinstance(adapter, AdapterManifest) for adapter in adapters):
            raise TypeError("adapters must contain only AdapterManifest values")
        if not isinstance(self.include_null, bool):
            raise TypeError("include_null must be boolean")
        adapter_ids = [adapter.adapter_id for adapter in adapters]
        if len(adapter_ids) != len(set(adapter_ids)):
            raise ValueError("adapter IDs must be unique within a bank")
        if self.include_null:
            _validate_identifier(self.null_route_id, "null_route_id")
            if self.null_route_id in adapter_ids:
                raise ValueError("null_route_id must not collide with an adapter ID")

        base_fingerprints = {adapter.base.fingerprint for adapter in adapters}
        if len(base_fingerprints) != 1:
            raise ValueError("all adapters in a bank must target the exact same base manifest")
        lora_fingerprints = {adapter.lora.fingerprint for adapter in adapters}
        if len(lora_fingerprints) != 1:
            raise ValueError("all adapters in a v1 bank must share one LoRA format manifest")
        object.__setattr__(self, "adapters", adapters)
        routes = [
            RouteManifest(
                route_id=adapter.adapter_id,
                route_index=index,
                adapter_fingerprint=adapter.fingerprint,
            )
            for index, adapter in enumerate(adapters)
        ]
        if self.include_null:
            routes.append(
                RouteManifest(
                    route_id=self.null_route_id,
                    route_index=len(routes),
                    adapter_fingerprint=None,
                    is_null=True,
                )
            )
        object.__setattr__(self, "_routes", tuple(routes))
        object.__setattr__(self, "_fingerprint", fingerprint_json(self.to_dict()))

    @property
    def base(self) -> BaseModelManifest:
        return self.adapters[0].base

    @property
    def lora(self) -> LoRAFormatManifest:
        return self.adapters[0].lora

    @property
    def routes(self) -> tuple[RouteManifest, ...]:
        return self._routes

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(route.route_id for route in self.routes)

    def route_by_id(self, route_id: str) -> RouteManifest:
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise KeyError(f"unknown route ID for bank {self.bank_id!r}: {route_id!r}")

    def route_by_index(self, route_index: int) -> RouteManifest:
        if (
            not isinstance(route_index, int)
            or isinstance(route_index, bool)
            or not 0 <= route_index < len(self.routes)
        ):
            raise IndexError(f"route index is out of range: {route_index}")
        return self.routes[route_index]

    def validate_lora_config(self, config: LoRABankConfig) -> None:
        """Reject a runtime bank whose positional layout or math differs from this manifest."""

        if not isinstance(config, LoRABankConfig):
            raise TypeError("config must be a LoRABankConfig")
        expected = (
            len(self.adapters),
            self.lora.rank,
            self.lora.alpha,
            self.lora.dropout,
            self.include_null,
        )
        actual = (
            config.num_experts,
            config.rank,
            config.alpha,
            config.dropout,
            config.include_null_expert,
        )
        if actual != expected:
            raise ValueError(
                "LoRA runtime config does not match bank manifest "
                f"(expected {expected!r}, got {actual!r})"
            )
        unsupported = []
        if self.lora.use_rslora:
            unsupported.append("use_rslora")
        if self.lora.use_dora:
            unsupported.append("use_dora")
        if self.lora.fan_in_fan_out:
            unsupported.append("fan_in_fan_out")
        if self.lora.modules_to_save:
            unsupported.append("modules_to_save")
        if self.lora.bias != "none":
            unsupported.append("bias")
        if unsupported:
            raise ValueError(
                "current vectorized runtime does not implement: " + ", ".join(unsupported)
            )

    def validate_router_config(self, config: RouterConfig) -> None:
        """Reject a router whose output axis cannot map exactly to these routes."""

        if not isinstance(config, RouterConfig):
            raise TypeError("config must be a RouterConfig")
        if config.num_experts != len(self.adapters):
            raise ValueError("router expert count does not match adapter bank")
        if config.include_null_expert != self.include_null:
            raise ValueError("router null-route setting does not match adapter bank")

    def validate_runtime_module(
        self,
        *,
        module_name: str | None,
        in_features: int,
        out_features: int,
        dtype: str,
    ) -> None:
        """Validate one per-projection bank against the whole-adapter tensor contract."""

        if module_name is None:
            raise ValueError("manifest-bound LoRA banks require an exact module_name")
        try:
            expected = next(
                module for module in self.lora.module_shapes if module.module_name == module_name
            )
        except StopIteration as exc:
            raise ValueError(
                f"module {module_name!r} is absent from the adapter tensor contract"
            ) from exc
        if (in_features, out_features) != (
            expected.in_features,
            expected.out_features,
        ):
            raise ValueError(
                f"module {module_name!r} shape differs from manifest "
                f"(expected {expected.in_features}->{expected.out_features}, "
                f"got {in_features}->{out_features})"
            )
        normalized_dtype = dtype.lower().removeprefix("torch.")
        if normalized_dtype != self.lora.dtype:
            raise ValueError(
                f"module {module_name!r} dtype differs from manifest "
                f"(expected {self.lora.dtype!r}, got {normalized_dtype!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "revision": self.revision,
            "base_fingerprint": self.base.fingerprint,
            "lora_fingerprint": self.lora.fingerprint,
            "adapters": [adapter.to_dict() for adapter in self.adapters],
            "routes": [route.to_dict() for route in self.routes],
        }

    @property
    def fingerprint(self) -> str:
        return self._fingerprint
