from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from molor.config import RouterConfig
from molor.experimental.autorouter import (
    AutoRouterPolicy,
    NULL_ROUTE_ID,
    PromptAutoRouter,
)
from molor.router import TopKRouter


class _Tokenizer:
    def __call__(self, _prompt: str, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }


def test_auto_selected_routes_and_applied_weights_follow_probability_rank(tmp_path: Path) -> None:
    config = RouterConfig(
        hidden_size=2,
        num_experts=2,
        top_k=2,
        mode="sequence",
        router_hidden_size=None,
    )
    network = TopKRouter(config)
    linear = network.network[-1]
    assert isinstance(linear, nn.Linear)
    with torch.no_grad():
        linear.weight.zero_()
        # Checkpoint axis is resident-a, resident-b, null.  resident-b is the true top-1.
        linear.bias.copy_(torch.tensor([1.0, 3.0, 0.0]))

    router = PromptAutoRouter(
        router=network,
        tokenizer=_Tokenizer(),
        input_embeddings=nn.Embedding(4, 2),
        route_ids=("resident-a", "resident-b", NULL_ROUTE_ID),
        available_experts=("resident-a", "resident-b"),
        checkpoint_directory=tmp_path,
        feature_contract={
            "source": "input_embeddings",
            "pooling": "masked_mean",
            "prompt_scope": "user_content_only",
            "max_tokens": 256,
            "padding": "longest",
            "truncation": "right",
            "add_special_tokens": True,
        },
    )

    decision = router.decide("route this", AutoRouterPolicy(top_k=2))

    assert decision.selected_routes == ("resident-b", "resident-a")
    assert tuple(name for name, _weight in decision.applied_route) == decision.selected_routes
    assert decision.applied_route[0][1] > decision.applied_route[1][1]
    assert decision.probability_map["resident-b"] > decision.probability_map["resident-a"]
