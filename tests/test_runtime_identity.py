import pytest
import torch

from molor import LoRABankConfig, LoRAExpertBank, RouterConfig, TopKRouter
from molor.manifests import AdapterBankManifest

from test_manifests import adapter


def manifest(*, reverse: bool = False) -> AdapterBankManifest:
    adapters = (adapter("math", "1"), adapter("code", "2"))
    if reverse:
        adapters = tuple(reversed(adapters))
    return AdapterBankManifest("skills", "1", adapters)


def lora_config() -> LoRABankConfig:
    return LoRABankConfig(
        num_experts=2,
        rank=16,
        alpha=32,
        dropout=0.05,
        include_null_expert=True,
    )


def router_config() -> RouterConfig:
    return RouterConfig(
        hidden_size=4,
        num_experts=2,
        top_k=1,
        include_null_expert=True,
    )


def confirm_artifacts(
    runtime: LoRAExpertBank,
    bank_manifest: AdapterBankManifest,
) -> None:
    runtime.confirm_loaded_artifacts(
        bank_manifest,
        artifact_sha256s={
            f"{adapter_manifest.adapter_id}:{artifact.logical_name}": artifact.sha256
            for adapter_manifest in bank_manifest.adapters
            for artifact in adapter_manifest.weights
        },
        tensor_schema_sha256=bank_manifest.lora.tensor_schema_sha256,
    )


def test_bound_router_output_runs_only_against_the_same_bank() -> None:
    first_manifest = manifest()
    second_manifest = manifest(reverse=True)
    router = TopKRouter(router_config(), bank_manifest=first_manifest)
    first_bank = LoRAExpertBank(
        4, 3, lora_config(), bank_manifest=first_manifest, module_name="q_proj"
    )
    second_bank = LoRAExpertBank(
        4, 3, lora_config(), bank_manifest=second_manifest, module_name="q_proj"
    )
    confirm_artifacts(first_bank, first_manifest)
    confirm_artifacts(second_bank, second_manifest)

    output = router(torch.randn(2, 4))

    assert output.bank_fingerprint == first_manifest.fingerprint
    assert output.route_ids == first_manifest.route_ids
    assert first_bank(torch.randn(2, 4), output).shape == (2, 3)
    with pytest.raises(ValueError, match="different banks"):
        second_bank(torch.randn(2, 4), output)


def test_bound_bank_rejects_anonymous_routing_tensor_without_explicit_identity() -> None:
    bank_manifest = manifest()
    bank = LoRAExpertBank(4, 3, lora_config(), bank_manifest=bank_manifest, module_name="q_proj")
    confirm_artifacts(bank, bank_manifest)
    weights = torch.tensor([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="different banks"):
        bank(torch.randn(1, 4), weights)

    result = bank(
        torch.randn(1, 4),
        weights,
        bank_fingerprint=bank_manifest.fingerprint,
    )
    assert result.shape == (1, 3)


def test_stable_route_ids_execute_the_intended_expert_and_last_null_delta() -> None:
    bank_manifest = manifest()
    runtime = LoRAExpertBank(
        4,
        3,
        lora_config(),
        bank_manifest=bank_manifest,
        module_name="q_proj",
    )
    runtime.eval()
    with torch.no_grad():
        for expert in runtime.experts:
            expert.lora_A.zero_()
            expert.lora_B.zero_()
        runtime.experts[0].lora_A[0, 0] = 1
        runtime.experts[0].lora_B[0, 0] = 2
        runtime.experts[1].lora_A[0, 1] = 1
        runtime.experts[1].lora_B[0, 0] = 3
    confirm_artifacts(runtime, bank_manifest)
    inputs = torch.tensor([[4.0, 5.0, 0.0, 0.0]])

    def route(route_id: str) -> torch.Tensor:
        weights = torch.zeros(1, len(bank_manifest.routes))
        weights[0, bank_manifest.route_by_id(route_id).route_index] = 1
        return runtime(
            inputs,
            weights,
            bank_fingerprint=bank_manifest.fingerprint,
        )

    assert torch.equal(route("math"), torch.tensor([[16.0, 0.0, 0.0]]))
    assert torch.equal(route("code"), torch.tensor([[30.0, 0.0, 0.0]]))
    assert torch.equal(route("base/null"), torch.zeros(1, 3))


def test_router_checkpoint_rejects_same_width_different_bank() -> None:
    source = TopKRouter(router_config(), bank_manifest=manifest())
    destination = TopKRouter(router_config(), bank_manifest=manifest(reverse=True))

    with pytest.raises(RuntimeError, match="different adapter bank"):
        destination.load_state_dict(source.state_dict())


def test_lora_checkpoint_rejects_same_shape_different_bank() -> None:
    source = LoRAExpertBank(4, 3, lora_config(), bank_manifest=manifest(), module_name="q_proj")
    destination = LoRAExpertBank(
        4,
        3,
        lora_config(),
        bank_manifest=manifest(reverse=True),
        module_name="q_proj",
    )

    with pytest.raises(RuntimeError, match="different adapter bank"):
        destination.load_state_dict(source.state_dict())


def test_bound_bank_rejects_parameter_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="dtype differs"):
        LoRAExpertBank(
            4,
            3,
            lora_config(),
            dtype=torch.float16,
            bank_manifest=manifest(),
            module_name="q_proj",
        )


def test_newly_initialized_bound_bank_is_not_artifact_verified() -> None:
    bank_manifest = manifest()
    bank = LoRAExpertBank(4, 3, lora_config(), bank_manifest=bank_manifest, module_name="q_proj")
    output = TopKRouter(router_config(), bank_manifest=bank_manifest)(torch.randn(1, 4))

    assert not bank.parameters_verified
    assert bank.verified_bank_fingerprint is None
    with pytest.raises(RuntimeError, match="not artifact-verified"):
        bank(torch.randn(1, 4), output)

    confirm_artifacts(bank, bank_manifest)
    assert bank.parameters_verified
    assert bank.verified_bank_fingerprint == bank_manifest.fingerprint


def test_unbound_bank_rejects_checkpoint_with_different_lora_math() -> None:
    source = LoRAExpertBank(4, 3, lora_config(), bank_manifest=manifest(), module_name="q_proj")
    destination = LoRAExpertBank(
        4,
        3,
        LoRABankConfig(
            num_experts=2,
            rank=16,
            alpha=16,
            dropout=0.05,
            include_null_expert=True,
        ),
        module_name="q_proj",
    )

    with pytest.raises(RuntimeError, match="runtime contract"):
        destination.load_state_dict(source.state_dict())


def test_bound_bank_rejects_unknown_module_and_wrong_projection_shape() -> None:
    bank_manifest = manifest()
    with pytest.raises(ValueError, match="absent from the adapter tensor contract"):
        LoRAExpertBank(
            4,
            3,
            lora_config(),
            bank_manifest=bank_manifest,
            module_name="k_proj",
        )
    with pytest.raises(ValueError, match="shape differs"):
        LoRAExpertBank(
            4,
            2,
            lora_config(),
            bank_manifest=bank_manifest,
            module_name="q_proj",
        )


def test_unbound_router_rejects_checkpoint_with_different_gate_semantics() -> None:
    source = TopKRouter(router_config(), bank_manifest=manifest())
    destination = TopKRouter(
        RouterConfig(
            hidden_size=4,
            num_experts=2,
            top_k=2,
            include_null_expert=True,
        )
    )

    with pytest.raises(RuntimeError, match="runtime contract"):
        destination.load_state_dict(source.state_dict())


def test_no_null_bank_binds_and_round_trips_without_an_extra_route() -> None:
    bank_manifest = AdapterBankManifest(
        "skills-no-null",
        "1",
        (adapter("math", "1"), adapter("code", "2")),
        include_null=False,
    )
    bank_config = LoRABankConfig(
        num_experts=2,
        rank=16,
        alpha=32,
        dropout=0.05,
        include_null_expert=False,
    )
    gate_config = RouterConfig(
        hidden_size=4,
        num_experts=2,
        top_k=1,
        include_null_expert=False,
    )
    router = TopKRouter(gate_config, bank_manifest=bank_manifest)
    runtime = LoRAExpertBank(
        4,
        3,
        bank_config,
        bank_manifest=bank_manifest,
        module_name="q_proj",
    )
    confirm_artifacts(runtime, bank_manifest)
    output = router(torch.randn(2, 4))

    assert bank_manifest.route_ids == ("math", "code")
    assert output.weights.shape == (2, 2)
    assert runtime(torch.randn(2, 4), output).shape == (2, 3)

    restored_router = TopKRouter(gate_config, bank_manifest=bank_manifest)
    restored_runtime = LoRAExpertBank(
        4,
        3,
        bank_config,
        bank_manifest=bank_manifest,
        module_name="q_proj",
    )
    restored_router.load_state_dict(router.state_dict())
    restored_runtime.load_state_dict(runtime.state_dict())
    assert not restored_runtime.parameters_verified
    confirm_artifacts(restored_runtime, bank_manifest)
    restored_output = restored_router(torch.randn(1, 4))
    assert restored_runtime(torch.randn(1, 4), restored_output).shape == (1, 3)
