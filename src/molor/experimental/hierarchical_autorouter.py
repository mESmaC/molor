"""Hierarchical family-to-leaf routing with calibrated posterior semantics.

This module is deliberately independent of a particular classifier checkpoint.  It combines
root-family logits and conditional leaf logits into exact joint probabilities, applies masks and
fallback gates, and keeps execution blend weights distinct from posterior probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal

from molor.experimental.expert_taxonomy import ExpertTaxonomy


BASE_ROUTE = "base/null"


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _probability(value: object, field_name: str) -> float:
    result = _finite(value, field_name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return result


def _positive(value: object, field_name: str) -> float:
    result = _finite(value, field_name)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _softmax(
    logits: Mapping[str, float],
    ordered_ids: Sequence[str],
    *,
    temperature: float,
) -> tuple[tuple[str, float], ...]:
    values = [
        _finite(logits[route_id], f"logits[{route_id!r}]") / temperature for route_id in ordered_ids
    ]
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = math.fsum(exponentials)
    return tuple(
        (route_id, exponential / total)
        for route_id, exponential in zip(ordered_ids, exponentials, strict=True)
    )


@dataclass(frozen=True, slots=True)
class HierarchicalRouterPolicy:
    """Request-time gates around fixed root and leaf classifier logits."""

    top_k: int = 1
    root_temperature: float = 1.0
    leaf_temperature: float = 1.0
    base_bias: float = 0.0
    minimum_family_probability: float = 0.0
    minimum_joint_probability: float = 0.0
    enabled_leaves: tuple[str, ...] = ()
    explicit_blend_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or not 1 <= self.top_k <= 2
        ):
            raise ValueError("top_k must be 1 or 2")
        object.__setattr__(
            self, "root_temperature", _positive(self.root_temperature, "root_temperature")
        )
        object.__setattr__(
            self, "leaf_temperature", _positive(self.leaf_temperature, "leaf_temperature")
        )
        object.__setattr__(self, "base_bias", _finite(self.base_bias, "base_bias"))
        object.__setattr__(
            self,
            "minimum_family_probability",
            _probability(self.minimum_family_probability, "minimum_family_probability"),
        )
        object.__setattr__(
            self,
            "minimum_joint_probability",
            _probability(self.minimum_joint_probability, "minimum_joint_probability"),
        )
        if isinstance(self.enabled_leaves, (str, bytes)):
            raise TypeError("enabled_leaves must be a sequence")
        leaves = tuple(self.enabled_leaves)
        if any(not isinstance(leaf, str) or not leaf for leaf in leaves):
            raise ValueError("enabled_leaves must contain non-empty IDs")
        if len(leaves) != len(set(leaves)):
            raise ValueError("enabled_leaves must not contain duplicates")
        if not isinstance(self.explicit_blend_weights, Mapping):
            raise TypeError("explicit_blend_weights must be a mapping")
        weights: dict[str, float] = {}
        for leaf, weight in self.explicit_blend_weights.items():
            if not isinstance(leaf, str) or not leaf:
                raise ValueError("explicit blend leaf IDs must be non-empty")
            weights[leaf] = _positive(weight, f"explicit_blend_weights[{leaf!r}]")
        object.__setattr__(self, "enabled_leaves", leaves)
        object.__setattr__(self, "explicit_blend_weights", MappingProxyType(weights))


@dataclass(frozen=True, slots=True)
class HierarchicalRouteDecision:
    """Observable posterior, gating, and execution result for one prompt."""

    root_probabilities: tuple[tuple[str, float], ...]
    conditional_leaf_probabilities: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    joint_probabilities: tuple[tuple[str, float], ...]
    applied_route: tuple[tuple[str, float], ...]
    selected_routes: tuple[str, ...]
    confidence: float
    reason: str
    blend_weight_source: Literal["uniform", "explicit"]

    @property
    def root_probability_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.root_probabilities))

    @property
    def joint_probability_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.joint_probabilities))

    @property
    def route(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.applied_route))


def decide_hierarchical_route(
    taxonomy: ExpertTaxonomy,
    *,
    root_logits: Mapping[str, float],
    conditional_leaf_logits: Mapping[str, Mapping[str, float]],
    policy: HierarchicalRouterPolicy = HierarchicalRouterPolicy(),
) -> HierarchicalRouteDecision:
    """Combine root and conditional logits into a probability-ranked sparse route."""

    if not isinstance(taxonomy, ExpertTaxonomy):
        raise TypeError("taxonomy must be an ExpertTaxonomy")
    if not isinstance(root_logits, Mapping) or not isinstance(conditional_leaf_logits, Mapping):
        raise TypeError("root and conditional logits must be mappings")
    if not isinstance(policy, HierarchicalRouterPolicy):
        raise TypeError("policy must be a HierarchicalRouterPolicy")

    family_ids = tuple(family.family_id for family in taxonomy.families)
    expected_root = {BASE_ROUTE, *family_ids}
    if set(root_logits) != expected_root:
        raise ValueError("root logits must exactly match base/null plus every taxonomy family")
    if set(conditional_leaf_logits) != set(family_ids):
        raise ValueError("conditional logits must contain exactly every taxonomy family")

    leaf_to_family = {
        leaf.leaf_id: family.family_id for family in taxonomy.families for leaf in family.leaves
    }
    enabled = set(policy.enabled_leaves or taxonomy.leaf_ids)
    unknown_enabled = enabled.difference(leaf_to_family)
    if unknown_enabled:
        raise KeyError(f"enabled leaves are absent from the taxonomy: {sorted(unknown_enabled)!r}")
    unknown_blends = set(policy.explicit_blend_weights).difference(leaf_to_family)
    if unknown_blends:
        raise KeyError(
            f"explicit blend leaves are absent from the taxonomy: {sorted(unknown_blends)!r}"
        )

    active_family_ids: list[str] = []
    active_leaves: dict[str, tuple[str, ...]] = {}
    for family in taxonomy.families:
        expected_leaves = tuple(leaf.leaf_id for leaf in family.leaves)
        supplied = conditional_leaf_logits[family.family_id]
        if not isinstance(supplied, Mapping) or set(supplied) != set(expected_leaves):
            raise ValueError(
                f"conditional logits for {family.family_id!r} must exactly match its leaves"
            )
        selected = tuple(leaf_id for leaf_id in expected_leaves if leaf_id in enabled)
        active_leaves[family.family_id] = selected
        if selected:
            active_family_ids.append(family.family_id)
    if not active_family_ids:
        raise ValueError("at least one expert leaf must remain enabled")

    active_root_ids = (BASE_ROUTE, *active_family_ids)
    adjusted_root_logits = dict(root_logits)
    adjusted_root_logits[BASE_ROUTE] = (
        _finite(adjusted_root_logits[BASE_ROUTE], f"root_logits[{BASE_ROUTE!r}]") + policy.base_bias
    )
    active_root = _softmax(
        adjusted_root_logits,
        active_root_ids,
        temperature=policy.root_temperature,
    )
    active_root_map = dict(active_root)
    root_probabilities = tuple(
        (route_id, active_root_map.get(route_id, 0.0)) for route_id in (BASE_ROUTE, *family_ids)
    )

    conditionals: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    joints: list[tuple[str, float]] = []
    for family in taxonomy.families:
        enabled_family_leaves = active_leaves[family.family_id]
        if enabled_family_leaves:
            conditional = _softmax(
                conditional_leaf_logits[family.family_id],
                enabled_family_leaves,
                temperature=policy.leaf_temperature,
            )
        else:
            conditional = ()
        conditional_map = dict(conditional)
        full_conditional = tuple(
            (leaf.leaf_id, conditional_map.get(leaf.leaf_id, 0.0)) for leaf in family.leaves
        )
        conditionals.append((family.family_id, full_conditional))
        family_probability = active_root_map.get(family.family_id, 0.0)
        joints.extend(
            (leaf_id, family_probability * probability) for leaf_id, probability in conditional
        )
    ranked = tuple(sorted(joints, key=lambda item: (-item[1], item[0])))
    best_leaf, best_joint = ranked[0]
    best_family = leaf_to_family[best_leaf]
    best_family_probability = active_root_map[best_family]
    base_probability = active_root_map[BASE_ROUTE]

    eligible = tuple(
        (leaf_id, probability)
        for leaf_id, probability in ranked
        if probability >= policy.minimum_joint_probability
        and active_root_map[leaf_to_family[leaf_id]] >= policy.minimum_family_probability
    )
    fallback_reason: str | None = None
    if base_probability >= best_joint:
        fallback_reason = "base/null had the highest joint posterior"
    elif best_family_probability < policy.minimum_family_probability:
        fallback_reason = "best family probability was below the configured threshold"
    elif best_joint < policy.minimum_joint_probability:
        fallback_reason = "best leaf joint probability was below the configured threshold"
    elif not eligible:  # pragma: no cover - implied by the two threshold checks above
        fallback_reason = "no expert leaf passed the configured gates"

    if fallback_reason is not None:
        return HierarchicalRouteDecision(
            root_probabilities=root_probabilities,
            conditional_leaf_probabilities=tuple(conditionals),
            joint_probabilities=ranked,
            applied_route=((BASE_ROUTE, 1.0),),
            selected_routes=(BASE_ROUTE,),
            confidence=max(base_probability, best_joint),
            reason=fallback_reason,
            blend_weight_source="uniform",
        )

    selected = eligible[: policy.top_k]
    selected_names = tuple(leaf_id for leaf_id, _probability_value in selected)
    if policy.explicit_blend_weights:
        missing = [
            leaf_id for leaf_id in selected_names if leaf_id not in policy.explicit_blend_weights
        ]
        if missing:
            raise ValueError(
                f"explicit_blend_weights must cover every selected leaf; missing={missing!r}"
            )
        raw_weights = [policy.explicit_blend_weights[leaf_id] for leaf_id in selected_names]
        blend_source: Literal["uniform", "explicit"] = "explicit"
    else:
        raw_weights = [1.0] * len(selected_names)
        blend_source = "uniform"
    total_weight = math.fsum(raw_weights)
    applied = tuple(
        (leaf_id, weight / total_weight)
        for leaf_id, weight in zip(selected_names, raw_weights, strict=True)
    )
    return HierarchicalRouteDecision(
        root_probabilities=root_probabilities,
        conditional_leaf_probabilities=tuple(conditionals),
        joint_probabilities=ranked,
        applied_route=applied,
        selected_routes=selected_names,
        confidence=best_joint,
        reason="hierarchical router selected probability-ranked expert leaves",
        blend_weight_source=blend_source,
    )


__all__ = [
    "BASE_ROUTE",
    "HierarchicalRouteDecision",
    "HierarchicalRouterPolicy",
    "decide_hierarchical_route",
]
