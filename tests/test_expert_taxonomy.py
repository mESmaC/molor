from __future__ import annotations

import json
from pathlib import Path

import pytest

from molor.experimental.expert_taxonomy import load_expert_taxonomy


TAXONOMY = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "expert-rack-v5"
    / "expert-taxonomy-v1.json"
)


def test_checked_in_taxonomy_is_content_addressed_granular_and_hierarchical() -> None:
    taxonomy = load_expert_taxonomy(TAXONOMY)

    assert (
        taxonomy.fingerprint == "278748ac0cc029963d640460d36bb27b2cb1b8893b14e9e29d549b17c0d36dd1"
    )
    assert len(taxonomy.families) == 5
    assert len(taxonomy.leaves) == 25
    assert len(taxonomy.leaf_ids) == len(set(taxonomy.leaf_ids))
    assert taxonomy.routing_contract["root_routes"] == [
        "base/null",
        "agentic-tools",
        "business",
        "embedded-systems",
        "reasoning",
        "software",
    ]
    assert taxonomy.routing_contract["posterior_order"] == "strict_descending_probability"
    assert (
        taxonomy.routing_contract["blend_weight_semantics"]
        == "separate_normalized_adapter_contribution"
    )
    assert taxonomy.resolve_leaf("formal_math") == "formal-math"
    assert taxonomy.resolve_leaf("ocb-can-verified-v1") == "embedded-can-diagnostics"
    assert taxonomy.resolve_leaf("business-core") == "business-core"
    tool_leaf = next(leaf for leaf in taxonomy.leaves if leaf.leaf_id == "tool-use-execution")
    assert "direct" in tool_leaf.allowed_streams
    assert taxonomy.superseded_composite_labels["mathematical_formal_reasoning"] == (
        "formal-math",
        "quantitative-math",
    )
    assert taxonomy.external_opaque_import["row_access"] == "prohibited"
    with pytest.raises(TypeError):
        taxonomy.aliases["new"] = "formal-math"  # type: ignore[index]
    with pytest.raises(KeyError, match="unknown expert leaf"):
        taxonomy.resolve_leaf("unknown")


def test_taxonomy_rejects_tampering_duplicate_leaves_and_route_drift(tmp_path: Path) -> None:
    value = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    value["families"][0]["description"] = "tampered"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        load_expert_taxonomy(tampered)

    value = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    value["families"][1]["leaves"][0]["leaf_id"] = "agentic-planning"
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="globally unique"):
        load_expert_taxonomy(duplicate)

    value = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    value["routing_contract"]["root_routes"] = ["base/null", "business"]
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="root_routes"):
        load_expert_taxonomy(routes)


def test_taxonomy_rejects_ambiguous_alias_and_opaque_row_access(tmp_path: Path) -> None:
    value = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    value["aliases"]["formal-math"] = "quantitative-math"
    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="aliases must be non-canonical"):
        load_expert_taxonomy(ambiguous)

    value = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    value["external_opaque_import"]["row_access"] = "allowed"
    opaque = tmp_path / "opaque.json"
    opaque.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="row access"):
        load_expert_taxonomy(opaque)
