import pytest
import torch

from molor import LoRABankConfig, LoRAExpertBank


def test_bank_sums_complete_expert_deltas() -> None:
    config = LoRABankConfig(num_experts=2, rank=1, alpha=1, include_null_expert=False)
    bank = LoRAExpertBank(2, 1, config)
    with torch.no_grad():
        bank.experts[0].lora_A.copy_(torch.tensor([[1.0, 0.0]]))
        bank.experts[0].lora_B.copy_(torch.tensor([[2.0]]))
        bank.experts[1].lora_A.copy_(torch.tensor([[0.0, 1.0]]))
        bank.experts[1].lora_B.copy_(torch.tensor([[3.0]]))

    inputs = torch.tensor([[4.0, 5.0]])
    weights = torch.tensor([[0.25, 0.75]])

    assert torch.allclose(bank(inputs, weights), torch.tensor([[13.25]]))


def test_sequence_routes_broadcast_over_tokens() -> None:
    config = LoRABankConfig(num_experts=1, rank=1, alpha=1, include_null_expert=False)
    bank = LoRAExpertBank(2, 1, config)
    with torch.no_grad():
        bank.experts[0].lora_A.fill_(1.0)
        bank.experts[0].lora_B.fill_(1.0)

    inputs = torch.ones(2, 3, 2)
    routes = torch.ones(2, 1)

    assert bank(inputs, routes).shape == (2, 3, 1)
    assert torch.allclose(bank(inputs, routes), torch.full((2, 3, 1), 2.0))


def test_null_route_is_exactly_zero() -> None:
    config = LoRABankConfig(num_experts=1, rank=2, include_null_expert=True)
    bank = LoRAExpertBank(3, 4, config)
    with torch.no_grad():
        bank.experts[0].lora_B.fill_(1.0)

    inputs = torch.randn(2, 3)
    null_route = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    assert torch.equal(bank(inputs, null_route), torch.zeros(2, 4))


def test_experts_can_be_gradient_isolated() -> None:
    config = LoRABankConfig(num_experts=2, rank=2, include_null_expert=False)
    bank = LoRAExpertBank(3, 2, config)
    bank.set_trainable_experts([1])

    inputs = torch.randn(4, 3)
    route_to_one = torch.tensor([[0.0, 1.0]]).expand(4, -1)
    bank(inputs, route_to_one).sum().backward()

    assert bank.experts[0].lora_A.grad is None
    assert bank.experts[0].lora_B.grad is None
    assert bank.experts[1].lora_B.grad is not None
    assert torch.count_nonzero(bank.experts[1].lora_B.grad) > 0


def test_bank_rejects_wrong_route_width() -> None:
    bank = LoRAExpertBank(2, 2, LoRABankConfig(num_experts=2))
    with pytest.raises(ValueError, match="route weights"):
        bank(torch.ones(1, 2), torch.ones(1, 2))
