import pytest
import torch

from molor import RouterConfig, TopKRouter


def test_sequence_router_returns_normalized_sparse_routes() -> None:
    router = TopKRouter(
        RouterConfig(hidden_size=6, num_experts=3, top_k=2, include_null_expert=True)
    )
    hidden = torch.randn(4, 5, 6)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )

    output = router(hidden, mask)

    assert output.weights.shape == (4, 4)
    assert output.indices.shape == (4, 2)
    assert torch.allclose(output.weights.sum(dim=-1), torch.ones(4))
    assert torch.equal((output.weights > 0).sum(dim=-1), torch.full((4,), 2))
    assert output.load_balance_loss.ndim == 0
    assert output.z_loss.ndim == 0


def test_token_router_preserves_token_dimensions() -> None:
    router = TopKRouter(
        RouterConfig(
            hidden_size=4,
            num_experts=2,
            top_k=1,
            mode="token",
            include_null_expert=False,
            normalize_top_k=False,
        )
    )

    output = router(torch.randn(3, 7, 4))

    assert output.weights.shape == (3, 7, 2)
    assert output.indices.shape == (3, 7, 1)


def test_sequence_router_honors_attention_mask() -> None:
    router = TopKRouter(
        RouterConfig(hidden_size=2, num_experts=2, top_k=2, include_null_expert=False)
    )
    hidden = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    expected_features = torch.tensor([[3.0, 5.0]])

    output = router(hidden, mask)
    expected_logits = router.network(expected_features)

    assert torch.allclose(output.logits, expected_logits)


def test_router_rejects_empty_sequences() -> None:
    router = TopKRouter(RouterConfig(hidden_size=3, num_experts=1, top_k=1))
    with pytest.raises(ValueError, match="at least one unmasked token"):
        router(torch.randn(1, 2, 3), torch.zeros(1, 2))


def test_straight_through_router_is_hard_forward_and_soft_backward() -> None:
    torch.manual_seed(7)
    router = TopKRouter(
        RouterConfig(
            hidden_size=4,
            num_experts=3,
            top_k=1,
            include_null_expert=False,
            straight_through=True,
        )
    )

    output = router(torch.randn(5, 4))
    rewards = torch.tensor([0.0, 1.0, -0.5])
    loss = -(output.weights * rewards).sum()
    loss.backward()

    assert torch.equal(output.weights.detach().sum(dim=-1), torch.ones(5))
    assert torch.equal((output.weights.detach() != 0).sum(dim=-1), torch.ones(5, dtype=torch.long))
    assert router.network[-1].weight.grad is not None
    assert torch.count_nonzero(router.network[-1].weight.grad) > 0


def test_straight_through_router_rejects_top_two() -> None:
    with pytest.raises(ValueError, match="requires top_k=1"):
        RouterConfig(
            hidden_size=4,
            num_experts=3,
            top_k=2,
            straight_through=True,
        )


def test_uniform_top_two_uses_equal_fixed_weights() -> None:
    router = TopKRouter(
        RouterConfig(
            hidden_size=4,
            num_experts=4,
            top_k=2,
            include_null_expert=False,
            uniform_top_k=True,
        )
    )

    output = router(torch.randn(3, 4))

    assert torch.equal(output.weights.sum(dim=-1), torch.ones(3))
    assert torch.equal(
        output.weights[output.weights != 0],
        torch.full((6,), 0.5),
    )
