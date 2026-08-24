from __future__ import annotations

from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Mapping

import pytest

from molor.loading import (
    BaseLoadPolicy,
    LocalArtifactMaterializer,
    OptionalTransformersDependencyError,
    TransformersBaseLoader,
    TransformersLoadError,
)
from molor.loading.transformers import HuggingFaceTransformersBackend
from molor.manifests import (
    JSON_MEDIA_TYPE,
    SAFETENSORS_MEDIA_TYPE,
    ArtifactDescriptor,
    BaseModelManifest,
)


def write_artifact(
    root: Path,
    name: str,
    content: bytes,
    media_type: str,
) -> ArtifactDescriptor:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactDescriptor.from_file(path, logical_name=name, media_type=media_type)


def base_manifest(
    root: Path,
    *,
    quantization: str = "none",
    custom_code: tuple[ArtifactDescriptor, ...] = (),
) -> BaseModelManifest:
    return BaseModelManifest(
        model_id="local/Fake",
        requested_revision="main",
        resolved_revision="0123456789abcdef0123456789abcdef01234567",
        architecture="FakeForCausalLM",
        dtype="float32",
        quantization=quantization,
        config=write_artifact(root, "config.json", b"{}", JSON_MEDIA_TYPE),
        weights=(
            write_artifact(
                root,
                "model.safetensors",
                b"not-read-by-the-fake-backend",
                SAFETENSORS_MEDIA_TYPE,
            ),
        ),
        tokenizer=(
            write_artifact(root, "tokenizer.json", b"{}", JSON_MEDIA_TYPE),
            write_artifact(root, "tokenizer_config.json", b"{}", JSON_MEDIA_TYPE),
        ),
        special_tokens=write_artifact(
            root,
            "special_tokens_map.json",
            b"{}",
            JSON_MEDIA_TYPE,
        ),
        chat_template=write_artifact(
            root,
            "chat_template.jinja",
            b"{{ messages }}",
            "text/plain",
        ),
        custom_code=custom_code,
    )


class FakeConfig:
    architectures = ["FakeForCausalLM"]
    hidden_size = 16
    max_position_embeddings = 4096


class FakeTokenizer:
    is_fast = True
    model_max_length = 8192
    all_special_ids = [0, 1]
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, bool, bool]] = []

    def __len__(self) -> int:
        return 32

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> list[int]:
        self.encode_calls.append((text, add_special_tokens, truncation))
        return list(range(len(text.split())))


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class FakeEmbeddings:
    num_embeddings = 64


class FakeForCausalLM:
    def __init__(self) -> None:
        self.training = True
        self._parameters = [FakeParameter(), FakeParameter()]

    def requires_grad_(self, enabled: bool) -> FakeForCausalLM:
        for parameter in self._parameters:
            parameter.requires_grad = enabled
        return self

    def eval(self) -> FakeForCausalLM:
        self.training = False
        return self

    def parameters(self) -> tuple[FakeParameter, ...]:
        return tuple(self._parameters)

    def get_input_embeddings(self) -> FakeEmbeddings:
        return FakeEmbeddings()


class FakeBackend:
    def __init__(
        self,
        *,
        config: object | None = None,
        tokenizer: object | None = None,
        model: object | None = None,
        loading_info: Mapping[str, object] | None = None,
    ) -> None:
        self.config = FakeConfig() if config is None else config
        self.tokenizer = FakeTokenizer() if tokenizer is None else tokenizer
        self.model = FakeForCausalLM() if model is None else model
        self.loading_info = (
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
            if loading_info is None
            else loading_info
        )
        self.calls: list[tuple[str, dict[str, object]]] = []

    @property
    def library_versions(self) -> Mapping[str, str]:
        return {
            "torch": "2.test",
            "transformers": "5.test",
            "tokenizers": "1.test",
            "safetensors": "1.test",
        }

    def load_config(self, directory: Path, **kwargs: object) -> object:
        self.calls.append(("config", {"directory": directory, **kwargs}))
        return self.config

    def load_tokenizer(self, directory: Path, **kwargs: object) -> object:
        self.calls.append(("tokenizer", {"directory": directory, **kwargs}))
        return self.tokenizer

    def load_causal_lm(
        self,
        directory: Path,
        **kwargs: object,
    ) -> tuple[object, Mapping[str, object]]:
        self.calls.append(("model", {"directory": directory, **kwargs}))
        return self.model, self.loading_info


def staged_manifest(tmp_path: Path) -> tuple[BaseModelManifest, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source"
    source.mkdir()
    manifest = base_manifest(source)
    staged = LocalArtifactMaterializer().materialize(
        manifest,
        source,
        staging_parent=tmp_path,
    )
    return manifest, staged


def test_fake_backend_receives_only_forced_offline_safe_options_and_freezes_base(
    tmp_path: Path,
) -> None:
    manifest, staged = staged_manifest(tmp_path)
    backend = FakeBackend()
    runtime = TransformersBaseLoader(backend).load(
        manifest,
        staged,
        policy=BaseLoadPolicy(context_window_cap=2048),
    )
    try:
        assert [name for name, _ in backend.calls] == ["config", "tokenizer", "model"]
        config_call = backend.calls[0][1]
        tokenizer_call = backend.calls[1][1]
        model_call = backend.calls[2][1]
        assert config_call["local_files_only"] is True
        assert config_call["trust_remote_code"] is False
        assert tokenizer_call["local_files_only"] is True
        assert tokenizer_call["trust_remote_code"] is False
        assert tokenizer_call["use_fast"] is True
        assert model_call["local_files_only"] is True
        assert model_call["trust_remote_code"] is False
        assert model_call["use_safetensors"] is True
        assert model_call["output_loading_info"] is True
        assert model_call["dtype_name"] == "float32"
        assert model_call["placement"] == "cpu"
        assert model_call["max_memory"] == {}
        assert model_call["offload_directory"] is None
        assert runtime.binding.base_fingerprint == manifest.fingerprint
        assert runtime.binding.effective_context_window == 2048
        assert runtime.binding.tokenizer_is_fast
        assert not backend.model.training
        assert all(not parameter.requires_grad for parameter in backend.model.parameters())
        assert runtime.token_counter.count_tokens("one two three") == 3
        assert backend.tokenizer.encode_calls == [("one two three", False, False)]
    finally:
        root = staged.root
        runtime.close()
    assert not root.exists()


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        (
            FakeBackend(config=type("WrongConfig", (), {"architectures": ["Other"]})()),
            "architecture differs",
        ),
        (
            FakeBackend(loading_info={"missing_keys": ["model.weight"]}),
            "did not load exactly",
        ),
        (
            FakeBackend(
                tokenizer=type(
                    "SlowTokenizer",
                    (),
                    {
                        "is_fast": False,
                        "model_max_length": 1024,
                        "all_special_ids": [],
                        "chat_template": "{{ messages }}",
                        "__len__": lambda self: 32,
                    },
                )()
            ),
            "implementation differs",
        ),
        (
            FakeBackend(
                tokenizer=type(
                    "WrongTemplateTokenizer",
                    (),
                    {
                        "is_fast": True,
                        "model_max_length": 1024,
                        "all_special_ids": [],
                        "chat_template": "different",
                        "__len__": lambda self: 32,
                    },
                )()
            ),
            "chat template differs",
        ),
    ],
)
def test_loader_fails_closed_on_runtime_contract_mismatch(
    tmp_path: Path,
    backend: FakeBackend,
    message: str,
) -> None:
    manifest, staged = staged_manifest(tmp_path)
    root = staged.root

    with pytest.raises(TransformersLoadError, match=message):
        TransformersBaseLoader(backend).load(manifest, staged)
    assert staged.closed
    assert not root.exists()


def test_loader_v0_rejects_quantization_and_custom_code_before_backend_use(
    tmp_path: Path,
) -> None:
    quantized_source = tmp_path / "quantized"
    quantized_source.mkdir()
    quantized = base_manifest(quantized_source, quantization="bitsandbytes-nf4")
    quantized_stage = LocalArtifactMaterializer().materialize(
        quantized,
        quantized_source,
        staging_parent=tmp_path,
    )
    backend = FakeBackend()
    with pytest.raises(TransformersLoadError, match="only unquantized"):
        TransformersBaseLoader(backend).load(quantized, quantized_stage)
    assert backend.calls == []

    custom_source = tmp_path / "custom"
    custom_source.mkdir()
    custom_descriptor = write_artifact(
        custom_source,
        "modeling_custom.py",
        b"raise RuntimeError('must never execute')",
        "text/x-python",
    )
    custom = base_manifest(custom_source, custom_code=(custom_descriptor,))
    custom_stage = LocalArtifactMaterializer().materialize(
        custom,
        custom_source,
        staging_parent=tmp_path,
    )
    with pytest.raises(TransformersLoadError, match="rejects custom code"):
        TransformersBaseLoader(backend).load(custom, custom_stage)
    assert backend.calls == []


def test_runtime_binding_changes_with_loader_policy(tmp_path: Path) -> None:
    first_manifest, first_staged = staged_manifest(tmp_path / "first")
    first = TransformersBaseLoader(FakeBackend()).load(first_manifest, first_staged)
    first_fingerprint = first.binding.fingerprint
    first.close()

    second_manifest, second_staged = staged_manifest(tmp_path / "second")
    second = TransformersBaseLoader(FakeBackend()).load(
        second_manifest,
        second_staged,
        policy=BaseLoadPolicy(context_window_cap=1024),
    )
    try:
        assert first_manifest.fingerprint == second_manifest.fingerprint
        assert first_fingerprint != second.binding.fingerprint
    finally:
        second.close()


def test_default_backend_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def version(name: str) -> str:
        if name == "transformers":
            raise metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr("molor.loading.transformers.metadata.version", version)

    with pytest.raises(OptionalTransformersDependencyError, match="transformers"):
        _ = HuggingFaceTransformersBackend().library_versions


def test_cpu_policy_rejects_offload_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require automatic placement"):
        BaseLoadPolicy(max_memory=(("cpu", "64GiB"),))
    with pytest.raises(ValueError, match="require automatic placement"):
        BaseLoadPolicy(offload_directory=tmp_path)


def test_loader_rejects_staging_for_another_manifest(tmp_path: Path) -> None:
    manifest, staged = staged_manifest(tmp_path)
    different = replace(manifest, requested_revision="different")
    root = staged.root

    with pytest.raises(TransformersLoadError, match="different base manifest"):
        TransformersBaseLoader(FakeBackend()).load(different, staged)
    assert not root.exists()
