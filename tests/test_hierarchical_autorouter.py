from __future__ import annotations

from pathlib import Path

import pytest

from molor.experimental.expert_taxonomy import ExpertTaxonomy, load_expert_taxonomy
from molor.experimental.hierarchical_autorouter import (
    BASE_ROUTE,
    HierarchicalRouterPolicy,
    decide_hierarchical_route,
)


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "expert-rack-v5"
    / "expert-taxonomy-v1.json"
)


def _scores(taxonomy: ExpertTaxonomy) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    root = {BASE_ROUTE: 0.0, **{family.family_id: -4.0 for family in taxonomy.families}}
    conditional = {
        family.family_id: {leaf.leaf_id: 0.0 for leaf in family.leaves}
        for family in taxonomy.families
    }
    return root, conditional


def test_joint_posteriors_are_ranked_and_blend_weights_are_separate() -> None:
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    root, conditional = _scores(taxonomy)
    root["business"] = 4.0
    conditional["business"]["business-finance"] = 3.0
    conditional["business"]["investing-markets"] = 2.0

    decision = decide_hierarchical_route(
        taxonomy,
        root_logits=root,
        conditional_leaf_logits=conditional,
        policy=HierarchicalRouterPolicy(top_k=2),
    )

    assert decision.selected_routes == ("business-finance", "investing-markets")
    assert decision.applied_route == (
        ("business-finance", 0.5),
        ("investing-markets", 0.5),
    )
    assert decision.blend_weight_source == "uniform"
    assert (
        decision.joint_probability_map["business-finance"]
        > decision.joint_probability_map["investing-markets"]
    )
    assert decision.route["business-finance"] == decision.route["investing-markets"]
    assert sum(decision.joint_probability_map.values()) + decision.root_probability_map[
        BASE_ROUTE
    ] == pytest.approx(1.0)

    explicit = decide_hierarchical_route(
        taxonomy,
        root_logits=root,
        conditional_leaf_logits=conditional,
        policy=HierarchicalRouterPolicy(
            top_k=2,
            explicit_blend_weights={"business-finance": 3.0, "investing-markets": 1.0},
        ),
    )
    assert explicit.applied_route == (
        ("business-finance", 0.75),
        ("investing-markets", 0.25),
    )
    assert explicit.blend_weight_source == "explicit"


def test_base_fallback_and_enabled_leaf_mask_are_fail_closed() -> None:
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    root, conditional = _scores(taxonomy)
    root[BASE_ROUTE] = 8.0

    fallback = decide_hierarchical_route(
        taxonomy,
        root_logits=root,
        conditional_leaf_logits=conditional,
    )
    assert fallback.selected_routes == (BASE_ROUTE,)
    assert fallback.applied_route == ((BASE_ROUTE, 1.0),)
    assert "highest joint posterior" in fallback.reason

    root[BASE_ROUTE] = -2.0
    root["reasoning"] = 4.0
    masked = decide_hierarchical_route(
        taxonomy,
        root_logits=root,
        conditional_leaf_logits=conditional,
        policy=HierarchicalRouterPolicy(enabled_leaves=("formal-math",)),
    )
    assert masked.selected_routes == ("formal-math",)
    assert masked.root_probability_map["business"] == 0.0
    assert masked.joint_probability_map == {"formal-math": pytest.approx(masked.confidence)}


def test_hierarchy_requires_exact_axes_and_gates_low_confidence() -> None:
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    root, conditional = _scores(taxonomy)
    root["business"] = 3.0

    missing_root = dict(root)
    missing_root.pop("software")
    with pytest.raises(ValueError, match="root logits must exactly match"):
        decide_hierarchical_route(
            taxonomy,
            root_logits=missing_root,
            conditional_leaf_logits=conditional,
        )

    malformed = {family: dict(values) for family, values in conditional.items()}
    malformed["business"].pop("business-core")
    with pytest.raises(ValueError, match="must exactly match its leaves"):
        decide_hierarchical_route(
            taxonomy,
            root_logits=root,
            conditional_leaf_logits=malformed,
        )

    gated = decide_hierarchical_route(
        taxonomy,
        root_logits=root,
        conditional_leaf_logits=conditional,
        policy=HierarchicalRouterPolicy(minimum_joint_probability=0.9),
    )
    assert gated.selected_routes == (BASE_ROUTE,)
    assert "below the configured threshold" in gated.reason

    with pytest.raises(ValueError, match="cover every selected leaf"):
        decide_hierarchical_route(
            taxonomy,
            root_logits=root,
            conditional_leaf_logits=conditional,
            policy=HierarchicalRouterPolicy(
                top_k=2,
                explicit_blend_weights={"business-core": 1.0},
            ),
        )
