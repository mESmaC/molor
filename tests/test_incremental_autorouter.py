from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from molor.experimental import (
    AutoRouteDecision,
    AutoRouterPolicy,
    IncrementalPromptAutoRouter,
    PromptAutoRouter,
)
from molor.experimental.autorouter import BASE_ROUTE, NULL_ROUTE_ID, manual_route_decision


NEW_EXPERT = "taskmaster-v2"
LEGACY_EXPERTS = ("legacy-a", "legacy-b")


class _StubPromptAutoRouter(PromptAutoRouter):
    def __init__(
        self,
        *,
        route_ids: Sequence[str],
        available_experts: Sequence[str],
        decision: AutoRouteDecision,
        feature_contract: Mapping[str, object] | None = None,
        max_prompt_tokens: int = 256,
        tokenizer: object | None = None,
        input_embeddings: object | None = None,
    ) -> None:
        self.route_ids = tuple(route_ids)
        self.available_experts = tuple(available_experts)
        self.decision = decision
        self.feature_contract = dict(feature_contract or _feature_contract(max_prompt_tokens))
        self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer = tokenizer if tokenizer is not None else object()
        self.input_embeddings = input_embeddings if input_embeddings is not None else object()
        self.calls: list[tuple[str, AutoRouterPolicy]] = []

    def decide(self, prompt: str, policy: AutoRouterPolicy) -> AutoRouteDecision:
        self.calls.append((prompt, policy))
        if policy.mode == "manual":
            return manual_route_decision(self.available_experts, policy.manual_route)
        return self.decision


def _feature_contract(max_tokens: int = 256) -> dict[str, Any]:
    return {
        "source": "input_embeddings",
        "pooling": "masked_mean",
        "prompt_scope": "user_content_only",
        "max_tokens": max_tokens,
        "padding": "longest",
        "truncation": "right",
        "add_special_tokens": True,
        "embedding_tensor": "model.embed_tokens.weight",
    }


def _decision(
    *,
    probabilities: Sequence[tuple[str, float]],
    applied_route: Sequence[tuple[str, float]],
    reason: str,
) -> AutoRouteDecision:
    return AutoRouteDecision(
        mode="auto",
        applied_route=tuple(applied_route),
        probabilities=tuple(probabilities),
        selected_routes=tuple(name for name, _weight in applied_route),
        confidence=max(weight for _name, weight in probabilities),
        reason=reason,
        checkpoint_loaded=True,
    )


def _components(
    *,
    gate_probability: float = 0.4,
    threshold: float = 0.6,
) -> tuple[
    IncrementalPromptAutoRouter,
    _StubPromptAutoRouter,
    _StubPromptAutoRouter,
]:
    tokenizer = object()
    input_embeddings = object()
    gate_applied = NEW_EXPERT if gate_probability >= 0.5 else BASE_ROUTE
    gate = _StubPromptAutoRouter(
        route_ids=(NEW_EXPERT, NULL_ROUTE_ID),
        available_experts=(NEW_EXPERT,),
        decision=_decision(
            probabilities=(
                (BASE_ROUTE, 1.0 - gate_probability),
                (NEW_EXPERT, gate_probability),
            ),
            applied_route=((gate_applied, 1.0),),
            reason="binary gate",
        ),
        tokenizer=tokenizer,
        input_embeddings=input_embeddings,
    )
    delegate = _StubPromptAutoRouter(
        route_ids=(*LEGACY_EXPERTS, NULL_ROUTE_ID),
        available_experts=LEGACY_EXPERTS,
        decision=_decision(
            probabilities=(
                (BASE_ROUTE, 0.2),
                (LEGACY_EXPERTS[0], 0.3),
                (LEGACY_EXPERTS[1], 0.5),
            ),
            applied_route=((LEGACY_EXPERTS[0], 0.25), (LEGACY_EXPERTS[1], 0.75)),
            reason="legacy decision",
        ),
        tokenizer=tokenizer,
        input_embeddings=input_embeddings,
    )
    combined = IncrementalPromptAutoRouter(
        admission_gate=gate,
        delegate=delegate,
        new_expert=NEW_EXPERT,
        admission_threshold=threshold,
    )
    return combined, gate, delegate


def test_below_threshold_returns_the_exact_delegate_decision() -> None:
    router, gate, delegate = _components(gate_probability=0.4, threshold=0.6)
    policy = AutoRouterPolicy(
        top_k=2,
        temperature=0.7,
        confidence_threshold=0.25,
        null_bias=0.1,
    )

    result = router.decide("route this", policy)

    assert result is delegate.decision
    assert gate.calls[0][1] == AutoRouterPolicy(
        top_k=1,
        max_prompt_tokens=256,
        enabled_experts=(NEW_EXPERT,),
    )
    delegated_policy = delegate.calls[0][1]
    assert delegated_policy.top_k == 2
    assert delegated_policy.temperature == pytest.approx(0.7)
    assert delegated_policy.confidence_threshold == pytest.approx(0.25)
    assert delegated_policy.null_bias == pytest.approx(0.1)


@pytest.mark.parametrize("gate_probability", [0.6, 0.8])
def test_threshold_is_inclusive_and_admits_new_expert(gate_probability: float) -> None:
    router, _gate, delegate = _components(
        gate_probability=gate_probability,
        threshold=0.6,
    )

    result = router.decide("route this", AutoRouterPolicy())

    assert result.applied_route == ((NEW_EXPERT, 1.0),)
    assert result.selected_routes == (NEW_EXPERT,)
    assert result.probability_map[NEW_EXPERT] == pytest.approx(gate_probability)
    assert sum(result.probability_map.values()) == pytest.approx(1.0)
    assert delegate.calls == []


def test_disabled_new_expert_skips_gate_and_delegates_unchanged() -> None:
    router, gate, delegate = _components(gate_probability=0.99, threshold=0.6)

    result = router.decide(
        "route this",
        AutoRouterPolicy(enabled_experts=(LEGACY_EXPERTS[1],)),
    )

    assert gate.calls == []
    assert delegate.calls[0][1].enabled_experts == (LEGACY_EXPERTS[1],)
    assert result is delegate.decision


def test_manual_policy_supports_the_full_union() -> None:
    router, gate, delegate = _components()

    result = router.decide(
        "route this",
        AutoRouterPolicy(
            mode="manual",
            manual_route={LEGACY_EXPERTS[0]: 2.0, NEW_EXPERT: 1.0},
        ),
    )

    assert router.available_experts == (*LEGACY_EXPERTS, NEW_EXPERT)
    assert result.mode == "manual"
    assert result.applied_route == (
        (LEGACY_EXPERTS[0], pytest.approx(2.0 / 3.0)),
        (NEW_EXPERT, pytest.approx(1.0 / 3.0)),
    )
    assert set(result.probability_map) == {BASE_ROUTE, *LEGACY_EXPERTS, NEW_EXPERT}
    assert gate.calls == []
    assert delegate.calls == []


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("inf"), float("nan"), True])
def test_admission_threshold_must_be_finite_and_bounded(threshold: object) -> None:
    _router, gate, delegate = _components()

    with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
        IncrementalPromptAutoRouter(
            admission_gate=gate,
            delegate=delegate,
            new_expert=NEW_EXPERT,
            admission_threshold=threshold,  # type: ignore[arg-type]
        )


def test_constructor_requires_a_binary_disjoint_gate() -> None:
    _router, gate, delegate = _components()
    gate.route_ids = (NEW_EXPERT, "unexpected", NULL_ROUTE_ID)
    with pytest.raises(ValueError, match="route axis must be exactly"):
        IncrementalPromptAutoRouter(
            admission_gate=gate,
            delegate=delegate,
            new_expert=NEW_EXPERT,
            admission_threshold=0.5,
        )

    gate.route_ids = (NEW_EXPERT, NULL_ROUTE_ID)
    delegate.route_ids = (*delegate.route_ids[:-1], NEW_EXPERT, NULL_ROUTE_ID)
    with pytest.raises(ValueError, match="expert axes must be disjoint"):
        IncrementalPromptAutoRouter(
            admission_gate=gate,
            delegate=delegate,
            new_expert=NEW_EXPERT,
            admission_threshold=0.5,
        )


def test_constructor_requires_matching_feature_contract_and_token_limit() -> None:
    _router, gate, delegate = _components()
    gate.feature_contract = {**gate.feature_contract, "add_special_tokens": False}
    with pytest.raises(ValueError, match="feature contracts must match"):
        IncrementalPromptAutoRouter(
            admission_gate=gate,
            delegate=delegate,
            new_expert=NEW_EXPERT,
            admission_threshold=0.5,
        )

    gate.feature_contract = dict(delegate.feature_contract)
    gate.max_prompt_tokens = 128
    with pytest.raises(ValueError, match="max prompt tokens must match"):
        IncrementalPromptAutoRouter(
            admission_gate=gate,
            delegate=delegate,
            new_expert=NEW_EXPERT,
            admission_threshold=0.5,
        )


def test_constructor_requires_identical_runtime_representation_objects() -> None:
    _router, gate, delegate = _components()
    gate.tokenizer = object()
    with pytest.raises(ValueError, match="share the exact tokenizer object"):
        IncrementalPromptAutoRouter(
            admission_gate=gate,
            delegate=delegate,
            new_expert=NEW_EXPERT,
            admission_threshold=0.5,
        )

    gate.tokenizer = delegate.tokenizer
    gate.input_embeddings = object()
    with pytest.raises(ValueError, match="share the exact input_embeddings module"):
        IncrementalPromptAutoRouter(
            admission_gate=gate,
            delegate=delegate,
            new_expert=NEW_EXPERT,
            admission_threshold=0.5,
        )


def test_auto_policy_rejects_unknown_expert_and_wrong_token_limit() -> None:
    router, gate, delegate = _components()
    with pytest.raises(KeyError, match="not available"):
        router.decide("route this", AutoRouterPolicy(enabled_experts=("unknown",)))
    with pytest.raises(ValueError, match="pinned by its training contracts"):
        router.decide("route this", AutoRouterPolicy(max_prompt_tokens=128))
    assert gate.calls == []
    assert delegate.calls == []
