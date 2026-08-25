"""Backward-compatible admission routing for one incremental expert.

The admission gate is a binary :class:`PromptAutoRouter` over one new expert and
``base/null``.  A rejected or disabled incremental expert delegates execution to
an existing prompt router without changing that router or its checkpoint format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from molor.experimental.autorouter import (
    BASE_ROUTE,
    NULL_ROUTE_ID,
    AutoRouteDecision,
    AutoRouterPolicy,
    PromptAutoRouter,
    manual_route_decision,
)


@dataclass(frozen=True, slots=True)
class IncrementalPromptAutoRouter:
    """Compose one binary expert-admission gate with an immutable legacy router.

    ``admission_gate`` must expose exactly ``(new_expert, base/null)``.  When the
    new expert is not admitted, ``delegate`` remains solely responsible for the
    complete decision.  Both routers must share the exact tokenizer and input-embedding
    objects, not merely equivalent serialized feature contracts.  A negative gate
    returns the delegate's decision object unchanged; a positive gate does not consult
    the delegate.
    """

    admission_gate: PromptAutoRouter
    delegate: PromptAutoRouter
    new_expert: str
    admission_threshold: float
    available_experts: tuple[str, ...] = field(init=False)
    feature_contract: Mapping[str, object] = field(init=False, repr=False)
    max_prompt_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.admission_gate, PromptAutoRouter):
            raise TypeError("admission_gate must be a PromptAutoRouter")
        if not isinstance(self.delegate, PromptAutoRouter):
            raise TypeError("delegate must be a PromptAutoRouter")
        if (
            not isinstance(self.new_expert, str)
            or not self.new_expert
            or self.new_expert in (BASE_ROUTE, NULL_ROUTE_ID)
        ):
            raise ValueError("new_expert must be a non-empty expert route name")
        threshold = _admission_threshold(self.admission_threshold)

        gate_routes = tuple(self.admission_gate.route_ids)
        if gate_routes != (self.new_expert, NULL_ROUTE_ID):
            raise ValueError("admission_gate route axis must be exactly (new_expert, 'base/null')")
        gate_experts = tuple(self.admission_gate.available_experts)
        if gate_experts != (self.new_expert,):
            raise ValueError("admission_gate must expose only the new expert")

        delegate_expert_axis = {
            *self.delegate.available_experts,
            *(route_id for route_id in self.delegate.route_ids if route_id != NULL_ROUTE_ID),
        }
        if self.new_expert in delegate_expert_axis:
            raise ValueError("admission and delegate expert axes must be disjoint")

        if self.admission_gate.max_prompt_tokens != self.delegate.max_prompt_tokens:
            raise ValueError("admission gate and delegate max prompt tokens must match")
        gate_contract = dict(self.admission_gate.feature_contract)
        delegate_contract = dict(self.delegate.feature_contract)
        if gate_contract != delegate_contract:
            raise ValueError("admission gate and delegate prompt feature contracts must match")
        if self.admission_gate.tokenizer is not self.delegate.tokenizer:
            raise ValueError("admission gate and delegate must share the exact tokenizer object")
        if self.admission_gate.input_embeddings is not self.delegate.input_embeddings:
            raise ValueError(
                "admission gate and delegate must share the exact input_embeddings module"
            )

        available_experts = (*self.delegate.available_experts, self.new_expert)
        if len(available_experts) != len(set(available_experts)):
            raise ValueError("admission and delegate expert axes must be disjoint")
        object.__setattr__(self, "admission_threshold", threshold)
        object.__setattr__(self, "available_experts", available_experts)
        object.__setattr__(
            self,
            "feature_contract",
            MappingProxyType(delegate_contract),
        )
        object.__setattr__(
            self,
            "max_prompt_tokens",
            self.delegate.max_prompt_tokens,
        )

    def decide(self, prompt: str, policy: AutoRouterPolicy) -> AutoRouteDecision:
        """Apply manual union routing or the fixed incremental admission rule."""

        if not isinstance(policy, AutoRouterPolicy):
            raise TypeError("policy must be an AutoRouterPolicy")
        if policy.mode == "manual":
            return manual_route_decision(self.available_experts, policy.manual_route)
        if policy.max_prompt_tokens != self.max_prompt_tokens:
            raise ValueError(
                "incremental router max_prompt_tokens is pinned by its training contracts at "
                f"{self.max_prompt_tokens}, not {policy.max_prompt_tokens}"
            )

        enabled = self._enabled_experts(policy)
        if self.new_expert not in enabled:
            return self._delegate_decision(prompt, policy, enabled)

        gate_policy = AutoRouterPolicy(
            mode="auto",
            top_k=1,
            temperature=1.0,
            confidence_threshold=0.0,
            null_bias=0.0,
            max_prompt_tokens=self.max_prompt_tokens,
            enabled_experts=(self.new_expert,),
        )
        gate_decision = self.admission_gate.decide(prompt, gate_policy)
        gate_probability = self._gate_probability(gate_decision)
        if gate_probability >= self.admission_threshold:
            return AutoRouteDecision(
                mode="auto",
                applied_route=((self.new_expert, 1.0),),
                probabilities=gate_decision.probabilities,
                selected_routes=(self.new_expert,),
                confidence=gate_decision.confidence,
                reason=(
                    f"Incremental admission probability {gate_probability:.3f} met the fixed "
                    f"{self.admission_threshold:.3f} threshold; selected {self.new_expert}."
                ),
                checkpoint_loaded=gate_decision.checkpoint_loaded,
            )
        return self._delegate_decision(prompt, policy, enabled)

    def _enabled_experts(self, policy: AutoRouterPolicy) -> set[str]:
        if not policy.enabled_experts:
            return set(self.available_experts)
        enabled = set(policy.enabled_experts)
        unknown = enabled.difference(self.available_experts)
        if unknown:
            raise KeyError(
                f"enabled experts are not available to the incremental router: {sorted(unknown)!r}"
            )
        return enabled

    def _delegate_decision(
        self,
        prompt: str,
        policy: AutoRouterPolicy,
        enabled: set[str],
    ) -> AutoRouteDecision:
        if not policy.enabled_experts:
            delegate_enabled: tuple[str, ...] = ()
        else:
            delegate_enabled = tuple(
                name for name in self.delegate.available_experts if name in enabled
            )
            if not delegate_enabled:
                return self.delegate.decide(
                    prompt,
                    AutoRouterPolicy(
                        mode="manual",
                        max_prompt_tokens=self.max_prompt_tokens,
                        manual_route={},
                    ),
                )
        return self.delegate.decide(
            prompt,
            AutoRouterPolicy(
                mode="auto",
                top_k=policy.top_k,
                temperature=policy.temperature,
                confidence_threshold=policy.confidence_threshold,
                null_bias=policy.null_bias,
                max_prompt_tokens=policy.max_prompt_tokens,
                enabled_experts=delegate_enabled,
            ),
        )

    def _gate_probability(self, decision: AutoRouteDecision) -> float:
        expected = {BASE_ROUTE, self.new_expert}
        observed = set(decision.probability_map)
        if observed != expected:
            raise RuntimeError(
                "binary admission gate probabilities must contain exactly base and the new expert"
            )
        return decision.probability_map[self.new_expert]


def _admission_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("admission_threshold must be a finite number in [0, 1]")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("admission_threshold must be a finite number in [0, 1]")
    return threshold


__all__ = ["IncrementalPromptAutoRouter"]
