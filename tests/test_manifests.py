import hashlib

import pytest

from molor import LoRABankConfig, RouterConfig
from molor.manifests import (
    JSON_MEDIA_TYPE,
    SAFETENSORS_MEDIA_TYPE,
    AdapterBankManifest,
    AdapterManifest,
    ArtifactDescriptor,
    BaseModelManifest,
    LoRAFormatManifest,
    TargetModuleManifest,
)


def digest(character: str) -> str:
    return character * 64


def artifact(
    name: str,
    marker: str,
    *,
    media_type: str = "application/octet-stream",
    size: int = 100,
) -> ArtifactDescriptor:
    return ArtifactDescriptor(name, media_type, digest(marker), size)


def base(*, weights: str = "b", template: str = "e") -> BaseModelManifest:
    return BaseModelManifest(
        model_id="Qwen/Qwen3-1.7B",
        requested_revision="main",
        resolved_revision="0123456789abcdef0123456789abcdef01234567",
        architecture="Qwen3ForCausalLM",
        dtype="bfloat16",
        quantization="none",
        config=artifact("config.json", "a", media_type=JSON_MEDIA_TYPE),
        weights=(
            artifact(
                "model.safetensors",
                weights,
                media_type=SAFETENSORS_MEDIA_TYPE,
            ),
        ),
        tokenizer=(artifact("tokenizer.json", "c", media_type=JSON_MEDIA_TYPE),),
        special_tokens=artifact(
            "special_tokens_map.json",
            "d",
            media_type=JSON_MEDIA_TYPE,
        ),
        chat_template=artifact("chat_template.jinja", template, media_type="text/plain"),
    )


def lora(
    *,
    rank: int = 16,
    targets: tuple[str, ...] = ("v_proj", "q_proj"),
    schema: str = "9",
) -> LoRAFormatManifest:
    return LoRAFormatManifest(
        rank=rank,
        alpha=32,
        dropout=0.05,
        target_modules=targets,
        tensor_schema_sha256=digest(schema),
        module_shapes=tuple(TargetModuleManifest(target, 4, 3) for target in targets),
        dtype="float32",
    )


def adapter(
    adapter_id: str,
    marker: str,
    *,
    base_manifest: BaseModelManifest | None = None,
    lora_manifest: LoRAFormatManifest | None = None,
    revision: str = "1",
) -> AdapterManifest:
    return AdapterManifest(
        adapter_id=adapter_id,
        revision=revision,
        base=base_manifest or base(),
        lora=lora_manifest or lora(),
        weights=(
            artifact(
                "adapter_model.safetensors",
                marker,
                media_type=SAFETENSORS_MEDIA_TYPE,
            ),
        ),
        peft_config=artifact("adapter_config.json", "8", media_type=JSON_MEDIA_TYPE),
        training_run_sha256=digest("f"),
        library_versions=(("peft", "0.18.0"), ("transformers", "5.0.0")),
        tags=("specialist", adapter_id),
    )


def test_manifest_collections_are_canonical() -> None:
    left = lora(targets=("q_proj", "v_proj"))
    right = lora(targets=("v_proj", "q_proj"))

    assert left.target_modules == ("q_proj", "v_proj")
    assert left.fingerprint == right.fingerprint


def test_base_identity_covers_weights_tokenizer_and_template() -> None:
    original = base()

    assert original.fingerprint != base(weights="7").fingerprint
    assert original.fingerprint != base(template="6").fingerprint
    assert original.requested_revision == "main"
    assert len(original.resolved_revision) == 40
    with pytest.raises(ValueError, match="immutable"):
        BaseModelManifest(
            model_id="Qwen/Qwen3-1.7B",
            requested_revision="main",
            resolved_revision="main",
            architecture="Qwen3ForCausalLM",
            dtype="bfloat16",
            quantization="none",
            config=artifact("config.json", "a", media_type=JSON_MEDIA_TYPE),
            weights=(
                artifact(
                    "model.safetensors",
                    "b",
                    media_type=SAFETENSORS_MEDIA_TYPE,
                ),
            ),
            tokenizer=(artifact("tokenizer.json", "c", media_type=JSON_MEDIA_TYPE),),
            special_tokens=artifact(
                "special_tokens_map.json",
                "d",
                media_type=JSON_MEDIA_TYPE,
            ),
        )


def test_bank_has_stable_ids_and_explicit_last_null_route() -> None:
    bank = AdapterBankManifest(
        "skills",
        "1",
        (adapter("math", "1"), adapter("code", "2")),
    )

    assert bank.route_ids == ("math", "code", "base/null")
    assert bank.route_by_id("code").route_index == 1
    assert bank.route_by_index(2).is_null
    assert bank.route_by_index(2).adapter_fingerprint is None


def test_bank_rejects_mixed_base_or_lora_contracts() -> None:
    with pytest.raises(ValueError, match="same base"):
        AdapterBankManifest(
            "mixed-base",
            "1",
            (
                adapter("math", "1"),
                adapter("code", "2", base_manifest=base(weights="e")),
            ),
        )
    with pytest.raises(ValueError, match="LoRA format"):
        AdapterBankManifest(
            "mixed-lora",
            "1",
            (
                adapter("math", "1"),
                adapter("code", "2", lora_manifest=lora(rank=8)),
            ),
        )


def test_reorder_and_retraining_change_bank_fingerprint_without_changing_route_id() -> None:
    math = adapter("math", "1")
    code = adapter("code", "2")
    original = AdapterBankManifest("skills", "1", (math, code))
    reordered = AdapterBankManifest("skills", "2", (code, math))
    retrained_math = adapter("math", "3", revision="2")
    retrained = AdapterBankManifest("skills", "3", (retrained_math, code))

    assert original.fingerprint != reordered.fingerprint
    assert original.fingerprint != retrained.fingerprint
    assert original.route_by_id("math").route_id == retrained.route_by_id("math").route_id
    assert (
        original.route_by_id("math").adapter_fingerprint
        != retrained.route_by_id("math").adapter_fingerprint
    )


def test_manifest_rejects_bad_digests_paths_rank_and_pickle_weights() -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        ArtifactDescriptor("config.json", JSON_MEDIA_TYPE, "bad", 1)
    with pytest.raises(ValueError, match="relative artifact"):
        artifact("../adapter.safetensors", "1")
    with pytest.raises(ValueError, match="positive integer"):
        LoRAFormatManifest(1.5, 2, 0, ("q_proj",), digest("1"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="safetensors"):
        AdapterManifest(
            "unsafe",
            "1",
            base(),
            lora(),
            (artifact("adapter.bin", "1", media_type="application/x-pickle"),),
            artifact("adapter_config.json", "2", media_type=JSON_MEDIA_TYPE),
            digest("3"),
        )


def test_manifest_validates_runtime_shape_and_scaling_contract() -> None:
    bank = AdapterBankManifest(
        "skills",
        "1",
        (adapter("math", "1"), adapter("code", "2")),
    )
    bank.validate_lora_config(LoRABankConfig(num_experts=2, rank=16, alpha=32, dropout=0.05))
    bank.validate_router_config(RouterConfig(hidden_size=8, num_experts=2, top_k=1))

    with pytest.raises(ValueError, match="runtime config"):
        bank.validate_lora_config(LoRABankConfig(num_experts=2, rank=8))
    with pytest.raises(ValueError, match="expert count"):
        bank.validate_router_config(RouterConfig(hidden_size=8, num_experts=1, top_k=1))


def test_empty_generator_targets_and_runtime_bias_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        LoRAFormatManifest(16, 32, 0.05, iter(()), digest("1"))  # type: ignore[arg-type]

    biased_lora = LoRAFormatManifest(
        16,
        32,
        0.05,
        ("q_proj",),
        digest("1"),
        dtype="float32",
        bias="all",
    )
    biased_bank = AdapterBankManifest(
        "biased",
        "1",
        (
            adapter("math", "1", lora_manifest=biased_lora),
            adapter("code", "2", lora_manifest=biased_lora),
        ),
    )
    with pytest.raises(ValueError, match="bias"):
        biased_bank.validate_lora_config(
            LoRABankConfig(num_experts=2, rank=16, alpha=32, dropout=0.05)
        )


def test_artifact_descriptor_hashes_file_content(tmp_path) -> None:
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(b"adapter-bytes")

    descriptor = ArtifactDescriptor.from_file(
        path,
        logical_name="adapter.safetensors",
        media_type=SAFETENSORS_MEDIA_TYPE,
    )

    assert descriptor.sha256 == hashlib.sha256(b"adapter-bytes").hexdigest()
    assert descriptor.size_bytes == len(b"adapter-bytes")
