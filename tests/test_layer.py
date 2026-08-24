import torch
from torch import nn

from molor import FactorizedMixtureLoRALinear, LoRABankConfig, MixtureLoRALinear


def test_zero_initialized_layer_matches_frozen_base() -> None:
    torch.manual_seed(1)
    base = nn.Linear(4, 3)
    inputs = torch.randn(2, 5, 4)
    expected = base(inputs).detach().clone()

    layer = MixtureLoRALinear(base, LoRABankConfig(num_experts=3))
    routes = torch.softmax(torch.randn(2, 4), dim=-1)

    assert torch.equal(layer(inputs, routes), expected)
    assert all(not parameter.requires_grad for parameter in layer.base.parameters())


def test_null_route_preserves_base_after_expert_changes() -> None:
    base = nn.Linear(2, 2)
    layer = MixtureLoRALinear(base, LoRABankConfig(num_experts=1, rank=1))
    with torch.no_grad():
        layer.expert_bank.experts[0].lora_A.fill_(1.0)
        layer.expert_bank.experts[0].lora_B.fill_(1.0)

    inputs = torch.randn(3, 2)
    null_routes = torch.tensor([[0.0, 1.0]]).expand(3, -1)

    assert torch.equal(layer(inputs, null_routes), base(inputs))


def test_capability_and_memory_deltas_are_independent_additive_axes() -> None:
    base = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    layer = FactorizedMixtureLoRALinear(
        base,
        LoRABankConfig(num_experts=1, rank=1, alpha=1),
        LoRABankConfig(num_experts=1, rank=1, alpha=1),
    )
    with torch.no_grad():
        capability = layer.capability_bank.experts[0]
        capability.lora_A.copy_(torch.tensor([[1.0, 0.0]]))
        capability.lora_B.fill_(2.0)
        memory = layer.memory_bank.experts[0]
        memory.lora_A.copy_(torch.tensor([[0.0, 1.0]]))
        memory.lora_B.fill_(3.0)

    inputs = torch.tensor([[5.0, 7.0]])
    active = torch.tensor([[1.0, 0.0]])
    null = torch.tensor([[0.0, 1.0]])

    assert torch.equal(layer(inputs, active, active), torch.tensor([[31.0]]))
    assert torch.equal(layer(inputs, active, null), torch.tensor([[10.0]]))
    assert torch.equal(layer(inputs, null, active), torch.tensor([[21.0]]))


def test_factorized_layer_requires_one_memory_adapter_and_null() -> None:
    base = nn.Linear(2, 2)

    for invalid in (
        LoRABankConfig(num_experts=2),
        LoRABankConfig(num_experts=1, include_null_expert=False),
    ):
        try:
            FactorizedMixtureLoRALinear(base, LoRABankConfig(num_experts=1), invalid)
        except ValueError as error:
            assert "one adapter and an explicit null" in str(error)
        else:
            raise AssertionError("invalid memory bank was accepted")
