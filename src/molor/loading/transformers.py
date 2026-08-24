"""Offline Transformers base-model and tokenizer loading.

The public loader accepts only a previously verified private staging bundle.  Hugging Face
libraries are imported lazily so the core research primitives remain usable without the
optional integration dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Protocol

from molor.identity import fingerprint_json, validate_sha256
from molor.manifests import BaseModelManifest
from molor.loading.artifacts import StagedBaseArtifacts

if TYPE_CHECKING:
    from molor.runtime.execution import PromptMessage


class OptionalTransformersDependencyError(RuntimeError):
    """The explicitly optional Transformers loader dependencies are unavailable."""


class TransformersLoadError(RuntimeError):
    """Verified artifacts could not be loaded under the declared runtime contract."""


class TransformersBackend(Protocol):
    """Narrow injectable seam around the optional third-party model library."""

    @property
    def library_versions(self) -> Mapping[str, str]: ...

    def load_config(
        self,
        directory: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> object: ...

    def load_tokenizer(
        self,
        directory: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        use_fast: bool,
    ) -> object: ...

    def load_causal_lm(
        self,
        directory: Path,
        *,
        config: object,
        dtype_name: str,
        local_files_only: bool,
        trust_remote_code: bool,
        use_safetensors: bool,
        output_loading_info: bool,
        placement: str,
        max_memory: Mapping[str, str],
        offload_directory: Path | None,
    ) -> tuple[object, Mapping[str, object]]: ...


class HuggingFaceTransformersBackend:
    """Lazy adapter for native Transformers classes."""

    _required_packages = ("torch", "transformers", "safetensors")

    @staticmethod
    def _package_version(name: str, *, required: bool = True) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            if not required:
                return None
            raise OptionalTransformersDependencyError(
                f"optional loader dependency {name!r} is not installed; install the hf extra"
            ) from exc

    @property
    def library_versions(self) -> Mapping[str, str]:
        versions = {name: self._package_version(name) for name in self._required_packages}
        for optional in ("tokenizers", "accelerate"):
            version = self._package_version(optional, required=False)
            if version is not None:
                versions[optional] = version
        return {name: value for name, value in versions.items() if value is not None}

    @staticmethod
    def _imports() -> tuple[object, object, object, object]:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except (ImportError, ModuleNotFoundError) as exc:
            raise OptionalTransformersDependencyError(
                "Transformers loading requires torch, transformers, and safetensors"
            ) from exc
        try:
            metadata.version("safetensors")
        except metadata.PackageNotFoundError as exc:
            raise OptionalTransformersDependencyError(
                "Transformers loading requires the safetensors package"
            ) from exc
        return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer

    def load_config(
        self,
        directory: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> object:
        _, auto_config, _, _ = self._imports()
        return auto_config.from_pretrained(
            directory,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    def load_tokenizer(
        self,
        directory: Path,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        use_fast: bool,
    ) -> object:
        _, _, _, auto_tokenizer = self._imports()
        return auto_tokenizer.from_pretrained(
            directory,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )

    def load_causal_lm(
        self,
        directory: Path,
        *,
        config: object,
        dtype_name: str,
        local_files_only: bool,
        trust_remote_code: bool,
        use_safetensors: bool,
        output_loading_info: bool,
        placement: str,
        max_memory: Mapping[str, str],
        offload_directory: Path | None,
    ) -> tuple[object, Mapping[str, object]]:
        torch, _, auto_model, _ = self._imports()
        dtype = getattr(torch, dtype_name)
        kwargs: dict[str, object] = {
            "config": config,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "use_safetensors": use_safetensors,
            "output_loading_info": output_loading_info,
        }
        transformers_version = self._package_version("transformers")
        assert transformers_version is not None
        try:
            major_version = int(transformers_version.split(".", maxsplit=1)[0])
        except ValueError as exc:
            raise OptionalTransformersDependencyError(
                "installed Transformers version is not parseable"
            ) from exc
        if major_version >= 5:
            kwargs["dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
        if placement == "auto":
            if self._package_version("accelerate", required=False) is None:
                raise OptionalTransformersDependencyError(
                    "automatic/offloaded placement requires accelerate"
                )
            kwargs["device_map"] = "auto"
            kwargs["low_cpu_mem_usage"] = True
            if max_memory:
                kwargs["max_memory"] = dict(max_memory)
            if offload_directory is not None:
                kwargs["offload_folder"] = offload_directory
        loaded = auto_model.from_pretrained(directory, **kwargs)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise TransformersLoadError("Transformers did not return model loading diagnostics")
        model, loading_info = loaded
        if not isinstance(loading_info, Mapping):
            raise TransformersLoadError("Transformers returned invalid loading diagnostics")
        return model, loading_info


@dataclass(frozen=True, slots=True)
class BaseLoadPolicy:
    """Small, fingerprinted set of loader choices; arbitrary backend kwargs are forbidden."""

    schema_version: ClassVar[str] = "molor.base-load-policy.v1"

    placement: str = "cpu"
    use_fast_tokenizer: bool = True
    context_window_cap: int | None = None
    max_memory: tuple[tuple[str, str], ...] = ()
    offload_directory: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.placement not in {"cpu", "auto"}:
            raise ValueError("placement must be 'cpu' or 'auto'")
        if not isinstance(self.use_fast_tokenizer, bool):
            raise TypeError("use_fast_tokenizer must be boolean")
        if self.context_window_cap is not None and (
            not isinstance(self.context_window_cap, int)
            or isinstance(self.context_window_cap, bool)
            or self.context_window_cap < 1
        ):
            raise ValueError("context_window_cap must be a positive integer or None")
        if isinstance(self.max_memory, (str, bytes)):
            raise TypeError("max_memory must contain (device, limit) pairs")
        pairs = tuple(tuple(pair) for pair in self.max_memory)
        if any(
            len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or not isinstance(pair[1], str)
            or not pair[1]
            for pair in pairs
        ):
            raise ValueError("max_memory must contain non-empty string pairs")
        devices = [pair[0] for pair in pairs]
        if len(devices) != len(set(devices)):
            raise ValueError("max_memory must not repeat a device")
        if self.placement == "cpu" and (pairs or self.offload_directory is not None):
            raise ValueError("max_memory and offload_directory require automatic placement")
        if self.offload_directory is not None:
            directory = Path(self.offload_directory)
            if not directory.is_dir():
                raise FileNotFoundError("offload_directory must be an existing directory")
            object.__setattr__(self, "offload_directory", directory)
        object.__setattr__(self, "max_memory", tuple(sorted(pairs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "placement": self.placement,
            "use_fast_tokenizer": self.use_fast_tokenizer,
            "context_window_cap": self.context_window_cap,
            "max_memory": [list(pair) for pair in self.max_memory],
            "offload_enabled": self.offload_directory is not None,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class BaseRuntimeBindingManifest:
    """Immutable identity for one verified base/tokenizer runtime realization."""

    schema_version: ClassVar[str] = "molor.base-runtime-binding.v1"

    base_fingerprint: str
    artifact_set_fingerprint: str
    load_policy_fingerprint: str
    architecture: str
    config_class: str
    tokenizer_class: str
    tokenizer_is_fast: bool
    effective_context_window: int
    library_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "base_fingerprint",
            "artifact_set_fingerprint",
            "load_policy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name),
            )
        for field_name in ("architecture", "config_class", "tokenizer_class"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError(f"{field_name} must be a bounded non-empty string")
        if not isinstance(self.tokenizer_is_fast, bool):
            raise TypeError("tokenizer_is_fast must be boolean")
        if (
            not isinstance(self.effective_context_window, int)
            or isinstance(self.effective_context_window, bool)
            or self.effective_context_window < 1
        ):
            raise ValueError("effective_context_window must be a positive integer")
        versions = tuple(tuple(pair) for pair in self.library_versions)
        if any(
            len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or not isinstance(pair[1], str)
            or not pair[1]
            for pair in versions
        ):
            raise ValueError("library_versions must contain non-empty string pairs")
        names = [pair[0] for pair in versions]
        if len(names) != len(set(names)):
            raise ValueError("library_versions must not repeat a package")
        object.__setattr__(self, "library_versions", tuple(sorted(versions)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_fingerprint": self.base_fingerprint,
            "artifact_set_fingerprint": self.artifact_set_fingerprint,
            "load_policy_fingerprint": self.load_policy_fingerprint,
            "architecture": self.architecture,
            "config_class": self.config_class,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_is_fast": self.tokenizer_is_fast,
            "effective_context_window": self.effective_context_window,
            "library_versions": [list(pair) for pair in self.library_versions],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


class TransformersRuntimeTokenizer:
    """Verified tokenizer behavior used for retrieval and final prompt planning."""

    __slots__ = (
        "_chat_template",
        "_context_limit",
        "_fingerprint",
        "_tokenizer",
        "_vocabulary_size",
    )

    def __init__(self, tokenizer: object, binding: BaseRuntimeBindingManifest) -> None:
        if not isinstance(binding, BaseRuntimeBindingManifest):
            raise TypeError("binding must be a BaseRuntimeBindingManifest")
        try:
            vocabulary_size = len(tokenizer)  # type: ignore[arg-type]
        except (TypeError, AttributeError) as exc:
            raise TransformersLoadError("loaded tokenizer has no finite vocabulary") from exc
        if (
            not isinstance(vocabulary_size, int)
            or isinstance(vocabulary_size, bool)
            or vocabulary_size < 1
        ):
            raise TransformersLoadError("loaded tokenizer has an invalid vocabulary size")
        self._tokenizer = tokenizer
        self._vocabulary_size = vocabulary_size
        self._context_limit = binding.effective_context_window
        self._chat_template = getattr(tokenizer, "chat_template", None)
        self._fingerprint = fingerprint_json(
            {
                "schema_version": "molor.transformers-runtime-tokenizer.v1",
                "base_runtime_binding_fingerprint": binding.fingerprint,
                "context_limit": self._context_limit,
                "text_encoding": {
                    "add_special_tokens": False,
                    "truncation": False,
                },
                "chat_encoding": {
                    "api": "apply_chat_template",
                    "tokenize": True,
                    "truncation": False,
                    "padding": False,
                    "return_tensors": None,
                    "return_dict": False,
                    "add_generation_prompt": "caller-controlled-boolean",
                },
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def context_limit(self) -> int:
        return self._context_limit

    def _validated_token_ids(
        self,
        value: object,
        operation: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)):
            raise TransformersLoadError(f"tokenizer {operation} must return a flat token list")
        token_ids = tuple(value)
        if not token_ids and not allow_empty:
            raise TransformersLoadError(f"tokenizer {operation} returned no token IDs")
        if any(
            type(token_id) is not int  # bool and integer-like coercions are forbidden
            or not 0 <= token_id < self._vocabulary_size
            for token_id in token_ids
        ):
            raise TransformersLoadError(
                f"tokenizer {operation} returned an invalid or non-flat token ID"
            )
        return token_ids

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encode = getattr(self._tokenizer, "encode", None)
        if not callable(encode):
            raise TransformersLoadError("loaded tokenizer does not provide encode")
        token_ids = encode(text, add_special_tokens=False, truncation=False)
        return len(self._validated_token_ids(token_ids, "encode", allow_empty=True))

    def encode_chat(
        self,
        messages: tuple[PromptMessage, ...],
        *,
        add_generation_prompt: bool,
    ) -> tuple[int, ...]:
        """Apply the verified chat template and return exact flat token IDs."""

        from molor.runtime.execution import PromptMessage

        if not isinstance(add_generation_prompt, bool):
            raise TypeError("add_generation_prompt must be boolean")
        if not isinstance(messages, tuple):
            raise TypeError("messages must be a tuple of PromptMessage values")
        if not messages:
            raise ValueError("messages must not be empty")
        if any(not isinstance(message, PromptMessage) for message in messages):
            raise TypeError("messages must contain only PromptMessage values")
        current_template = getattr(self._tokenizer, "chat_template", None)
        if (
            not isinstance(self._chat_template, str)
            or not self._chat_template
            or current_template != self._chat_template
        ):
            raise TransformersLoadError("loaded tokenizer has no stable verified chat template")
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise TransformersLoadError("loaded tokenizer does not provide apply_chat_template")
        encoded = apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
        )
        return self._validated_token_ids(encoded, "apply_chat_template")


# Compatibility name retained for callers that use this object only as a retrieval counter.
TransformersTokenCounter = TransformersRuntimeTokenizer


@dataclass(slots=True)
class LoadedBaseRuntime:
    """Loaded mutable library objects paired with an immutable runtime binding."""

    manifest: BaseModelManifest
    config: object
    model: object
    tokenizer: object
    binding: BaseRuntimeBindingManifest
    token_counter: TransformersRuntimeTokenizer
    staged_artifacts: StagedBaseArtifacts

    def close(self) -> None:
        self.staged_artifacts.close()

    @property
    def runtime_tokenizer(self) -> TransformersRuntimeTokenizer:
        return self.token_counter

    def __enter__(self) -> LoadedBaseRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _require_root_level_artifacts(manifest: BaseModelManifest) -> None:
    from molor.loading.artifacts import base_artifact_descriptors

    for descriptor in base_artifact_descriptors(manifest):
        if len(PurePosixPath(descriptor.logical_name).parts) != 1:
            raise TransformersLoadError(
                "Transformers loader v0 requires a flat save_pretrained artifact layout"
            )


def _validate_manifest_for_loader(manifest: BaseModelManifest) -> None:
    if manifest.quantization != "none":
        raise TransformersLoadError("Transformers loader v0 supports only unquantized frozen bases")
    if manifest.custom_code:
        raise TransformersLoadError(
            "Transformers loader v0 rejects custom code and never enables remote code"
        )
    if manifest.config.logical_name != "config.json":
        raise TransformersLoadError("base config must be staged as config.json")
    if manifest.special_tokens.logical_name != "special_tokens_map.json":
        raise TransformersLoadError("special tokens must be staged as special_tokens_map.json")
    if manifest.weight_index is not None and (
        manifest.weight_index.logical_name != "model.safetensors.index.json"
    ):
        raise TransformersLoadError("sharded weights must use model.safetensors.index.json")
    if manifest.chat_template is not None and (
        manifest.chat_template.logical_name != "chat_template.jinja"
    ):
        raise TransformersLoadError("chat template must be staged as chat_template.jinja")
    _require_root_level_artifacts(manifest)


def _loading_errors(loading_info: Mapping[str, object]) -> dict[str, object]:
    failures: dict[str, object] = {}
    for field_name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        value = loading_info.get(field_name, ())
        if value:
            failures[field_name] = value
    return failures


def _positive_integer_attribute(value: object, name: str) -> int | None:
    candidate = getattr(value, name, None)
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
        return candidate
    return None


def _effective_context_window(
    config: object,
    tokenizer: object,
    cap: int | None,
) -> int:
    candidates = [
        _positive_integer_attribute(config, field_name)
        for field_name in ("max_position_embeddings", "n_positions", "seq_length")
    ]
    tokenizer_limit = _positive_integer_attribute(tokenizer, "model_max_length")
    # Tokenizers uses enormous sentinel values to mean "unknown".
    if tokenizer_limit is not None and tokenizer_limit <= 10_000_000:
        candidates.append(tokenizer_limit)
    if cap is not None:
        candidates.append(cap)
    bounded = [candidate for candidate in candidates if candidate is not None]
    if not bounded:
        raise TransformersLoadError(
            "model/tokenizer does not expose a bounded context window; configure an explicit cap"
        )
    return min(bounded)


def _embedding_rows(model: object) -> int:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_embeddings):
        raise TransformersLoadError("loaded causal model has no input embeddings")
    embeddings = get_embeddings()
    rows = getattr(embeddings, "num_embeddings", None)
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        weight = getattr(embeddings, "weight", None)
        shape = getattr(weight, "shape", ())
        rows = shape[0] if shape else None
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise TransformersLoadError("loaded causal model has invalid input embeddings")
    return rows


def _validate_tokenizer_embeddings(tokenizer: object, model: object) -> None:
    try:
        tokenizer_size = len(tokenizer)  # type: ignore[arg-type]
    except (TypeError, AttributeError) as exc:
        raise TransformersLoadError("loaded tokenizer has no finite vocabulary") from exc
    if (
        not isinstance(tokenizer_size, int)
        or isinstance(tokenizer_size, bool)
        or tokenizer_size < 1
    ):
        raise TransformersLoadError("loaded tokenizer has an invalid vocabulary size")
    rows = _embedding_rows(model)
    if tokenizer_size > rows:
        raise TransformersLoadError(
            "tokenizer vocabulary exceeds frozen input embeddings; resizing is forbidden"
        )
    special_ids = getattr(tokenizer, "all_special_ids", ())
    if not isinstance(special_ids, (list, tuple)) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or not 0 <= token_id < rows
        for token_id in special_ids
    ):
        raise TransformersLoadError("tokenizer special-token IDs exceed frozen embeddings")


def _validate_chat_template(
    manifest: BaseModelManifest,
    staged: StagedBaseArtifacts,
    tokenizer: object,
) -> None:
    loaded_template = getattr(tokenizer, "chat_template", None)
    if manifest.chat_template is None:
        if loaded_template not in (None, ""):
            raise TransformersLoadError(
                "tokenizer exposes a chat template absent from the base manifest"
            )
        return
    try:
        expected = staged.path_for(manifest.chat_template.logical_name).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TransformersLoadError("chat template is not readable UTF-8") from exc
    if loaded_template != expected:
        raise TransformersLoadError("loaded tokenizer chat template differs from its artifact")


class TransformersBaseLoader:
    """Load one content-verified local frozen causal LM and its exact tokenizer."""

    def __init__(self, backend: TransformersBackend | None = None) -> None:
        self._backend = HuggingFaceTransformersBackend() if backend is None else backend

    def load(
        self,
        manifest: BaseModelManifest,
        staged: StagedBaseArtifacts,
        *,
        policy: BaseLoadPolicy | None = None,
    ) -> LoadedBaseRuntime:
        if not isinstance(manifest, BaseModelManifest):
            raise TypeError("manifest must be a BaseModelManifest")
        if not isinstance(staged, StagedBaseArtifacts):
            raise TypeError("staged must be StagedBaseArtifacts")
        load_policy = BaseLoadPolicy() if policy is None else policy
        if not isinstance(load_policy, BaseLoadPolicy):
            raise TypeError("policy must be a BaseLoadPolicy")
        try:
            if staged.evidence.base_fingerprint != manifest.fingerprint:
                raise TransformersLoadError("staged artifacts belong to a different base manifest")
            _validate_manifest_for_loader(manifest)
            directory = staged.root
            config = self._backend.load_config(
                directory,
                local_files_only=True,
                trust_remote_code=False,
            )
            architectures = getattr(config, "architectures", None)
            if not isinstance(architectures, (list, tuple)) or manifest.architecture not in (
                architectures
            ):
                raise TransformersLoadError(
                    "loaded config architecture differs from the base manifest"
                )
            hidden_size = _positive_integer_attribute(config, "hidden_size")
            if hidden_size is None:
                raise TransformersLoadError("loaded causal config has no positive hidden_size")
            tokenizer = self._backend.load_tokenizer(
                directory,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=load_policy.use_fast_tokenizer,
            )
            tokenizer_is_fast = getattr(tokenizer, "is_fast", None)
            if tokenizer_is_fast is not load_policy.use_fast_tokenizer:
                raise TransformersLoadError(
                    "loaded tokenizer implementation differs from the load policy"
                )
            model, loading_info = self._backend.load_causal_lm(
                directory,
                config=config,
                dtype_name=manifest.dtype,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                output_loading_info=True,
                placement=load_policy.placement,
                max_memory=dict(load_policy.max_memory),
                offload_directory=load_policy.offload_directory,
            )
            if model.__class__.__name__ != manifest.architecture:
                raise TransformersLoadError(
                    "loaded causal model class differs from the base manifest"
                )
            failures = _loading_errors(loading_info)
            if failures:
                raise TransformersLoadError(
                    f"checkpoint did not load exactly: {sorted(failures)!r}"
                )
            requires_grad = getattr(model, "requires_grad_", None)
            eval_model = getattr(model, "eval", None)
            parameters = getattr(model, "parameters", None)
            if not callable(requires_grad) or not callable(eval_model) or not callable(parameters):
                raise TransformersLoadError("loaded object is not a trainable model")
            requires_grad(False)
            eval_model()
            if any(getattr(parameter, "requires_grad", True) for parameter in parameters()):
                raise TransformersLoadError("frozen base contains trainable parameters")
            _validate_tokenizer_embeddings(tokenizer, model)
            _validate_chat_template(manifest, staged, tokenizer)
            context_window = _effective_context_window(
                config,
                tokenizer,
                load_policy.context_window_cap,
            )
            versions = self._backend.library_versions
            if not isinstance(versions, Mapping) or any(
                not isinstance(name, str) or not name or not isinstance(version, str) or not version
                for name, version in versions.items()
            ):
                raise TransformersLoadError("backend returned invalid library versions")
            binding = BaseRuntimeBindingManifest(
                base_fingerprint=manifest.fingerprint,
                artifact_set_fingerprint=staged.evidence.fingerprint,
                load_policy_fingerprint=load_policy.fingerprint,
                architecture=model.__class__.__name__,
                config_class=config.__class__.__name__,
                tokenizer_class=tokenizer.__class__.__name__,
                tokenizer_is_fast=tokenizer_is_fast,
                effective_context_window=context_window,
                library_versions=tuple(versions.items()),
            )
            return LoadedBaseRuntime(
                manifest=manifest,
                config=config,
                model=model,
                tokenizer=tokenizer,
                binding=binding,
                token_counter=TransformersRuntimeTokenizer(tokenizer, binding),
                staged_artifacts=staged,
            )
        except BaseException:
            staged.close()
            raise
