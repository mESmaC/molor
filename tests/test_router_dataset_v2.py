from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.corpus_mixtures import CorpusShardDescriptor
from molor.experimental.expert_mixture_registry import (
    CollectedExpertData,
    RegisteredChatShard,
)
from molor.experimental.router_dataset_v2 import (
    ROUTER_EXAMPLES_V2_MANIFEST_SCHEMA,
    build_router_dataset_v2,
    load_router_curriculum_v2,
)
from molor.identity import canonical_json_bytes, fingerprint_json


def _write_chat(path: Path, prompts: list[str]) -> tuple[str, int]:
    payload = b"".join(
        canonical_json_bytes(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "synthetic answer"},
                ]
            }
        )
        + b"\n"
        for prompt in prompts
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(prompts)


def _collection(root: Path) -> CollectedExpertData:
    shards: list[RegisteredChatShard] = []
    families = {"leaf-a": "family-a", "leaf-b": "family-b"}
    for leaf in families:
        for split in ("train", "validation", "test"):
            path = root / "chat" / leaf / f"{split}.jsonl"
            count = 640 if split == "train" else 160
            prompts = [
                (
                    f"source {leaf} {split} prompt {index:04d}"
                    if index % 2 == 0
                    else (
                        f"source {leaf} {split} long prompt {index:04d}: "
                        + "an intentionally extended source context for length-tier retention " * 5
                    )
                )
                for index in range(count)
            ]
            sha256, records = _write_chat(path, prompts)
            descriptor = CorpusShardDescriptor(
                shard_id=f"synthetic/{leaf}/direct/{split}",
                shard_revision="source-v1",
                leaf_id=leaf,
                leaf_revision="taxonomy-v1",
                split=split,
                template_mode="direct-v1",
                path=str(path.resolve()),
                sha256=sha256,
                record_count=records,
            )
            shards.append(
                RegisteredChatShard(
                    descriptor=descriptor,
                    source_id="synthetic",
                    source_revision="source-v1",
                    semantic_stream="direct",
                    recipe_stream="direct",
                    length_tier="le-1024",
                    admission_role="primary",
                )
            )
    taxonomy_content: dict[str, object] = {
        "schema_version": "molor.expert-taxonomy.v1",
        "taxonomy_id": "synthetic",
        "revision": "1",
        "families": [],
        "aliases": {},
    }
    taxonomy = {**taxonomy_content, "content_sha256": fingerprint_json(taxonomy_content)}
    return CollectedExpertData(
        taxonomy=taxonomy,
        taxonomy_path=str((root / "taxonomy.json").resolve()),
        taxonomy_sha256="a" * 64,
        leaf_revision="taxonomy-v1",
        family_by_leaf=families,
        shards=tuple(shards),
        references=(),
        sources=(),
        dispositions=(),
    )


def _curriculum_document() -> dict[str, object]:
    content: dict[str, object] = {
        "schema_version": "molor.router-curriculum.v2",
        "dataset_id": "synthetic-router-curriculum-v2",
        "revision": "2",
        "authoring_contract": {
            "source_basis": "independently authored synthetic fixtures",
            "benchmark_material_used": False,
            "legacy_base_source_used": False,
            "intended_use": "router-training-only",
        },
        "split_weights": {"train": 4, "validation": 1, "test": 1},
        "base_negative_deny_terms": ["finance", "software"],
        "base_negative_groups": [
            {
                "group_id": "ordinary-writing-train",
                "split": "train",
                "template": "Write a friendly note using {left} and {right}.",
                "slots": {
                    "left": [f"train-left-{index:02d}" for index in range(40)],
                    "right": [f"train-right-{index:02d}" for index in range(20)],
                },
            },
            {
                "group_id": "ordinary-writing-validation",
                "split": "validation",
                "template": "Draft a gentle message using {left} and {right}.",
                "slots": {
                    "left": [f"validation-left-{index:02d}" for index in range(20)],
                    "right": [f"validation-right-{index:02d}" for index in range(10)],
                },
            },
            {
                "group_id": "ordinary-writing-test",
                "split": "test",
                "template": "Create a warm greeting using {left} and {right}.",
                "slots": {
                    "left": [f"test-left-{index:02d}" for index in range(20)],
                    "right": [f"test-right-{index:02d}" for index in range(10)],
                },
            },
        ],
        "anchor_surface_variants": [
            "{prompt}",
            "Please handle this request: {prompt}",
            "I need a clear response to this: {prompt}",
            "Work through the following task: {prompt}",
        ],
        "leaf_anchors": {
            "leaf-a": [f"Route alpha semantic request {index:02d}." for index in range(48)],
            "leaf-b": [f"Route beta semantic request {index:02d}." for index in range(48)],
        },
    }
    return {**content, "content_sha256": fingerprint_json(content)}


def _write_curriculum(path: Path, document: dict[str, object] | None = None) -> Path:
    payload = document or _curriculum_document()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_is_deterministic_disjoint_and_fully_supported(tmp_path: Path) -> None:
    collected = _collection(tmp_path)
    curriculum = _write_curriculum(tmp_path / "router-curriculum-v2.json")
    first = tmp_path / "router-v2-first"
    second = tmp_path / "router-v2-second"

    first_summary = build_router_dataset_v2(
        collected,
        curriculum_path=curriculum,
        output_directory=first,
    )
    second_summary = build_router_dataset_v2(
        collected,
        curriculum_path=curriculum,
        output_directory=second,
    )

    assert first_summary["content_sha256"] == second_summary["content_sha256"]
    manifest = json.loads((first / "router-examples-v2.manifest.json").read_bytes())
    assert manifest["schema_version"] == ROUTER_EXAMPLES_V2_MANIFEST_SCHEMA
    claimed = manifest.pop("content_sha256")
    assert claimed == fingerprint_json(manifest)
    assert manifest["benchmark_isolation"]["benchmark_material_used"] is False
    assert manifest["sources"]["true_general_negatives"]["legacy_router_base_source_used"] is False
    assert manifest["accounting"]["fully_reconciled"] is True
    assert manifest["accounting"]["zero_silent_drops"] is True
    assert manifest["accounting"]["curated_base_null_occurrences"] == 1_200
    assert manifest["accounting"]["short_anchor_occurrences"] == 384
    assert manifest["normalized_prompt_overlap"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert all(
        manifest["route_coverage"][split]["full_route_support"]
        for split in ("train", "validation", "test")
    )
    assert manifest["curriculum_balance"] == {
        "train": {"base/null": 800, "leaf-a": 128, "leaf-b": 128},
        "validation": {"base/null": 200, "leaf-a": 32, "leaf-b": 32},
        "test": {"base/null": 200, "leaf-a": 32, "leaf-b": 32},
    }
    semantic = manifest["curriculum_semantic_family_accounting"]
    assert semantic["anchor_surface_variants_per_seed"] == 4
    assert semantic["anchor_semantic_seeds"] == {
        "train": {"leaf-a": 32, "leaf-b": 32},
        "validation": {"leaf-a": 8, "leaf-b": 8},
        "test": {"leaf-a": 8, "leaf-b": 8},
    }
    assert semantic["base_group_split_owners"] == {
        "ordinary-writing-test": "test",
        "ordinary-writing-train": "train",
        "ordinary-writing-validation": "validation",
    }
    assert manifest["route_balance"] == {
        "train": {"base/null": 800, "leaf-a": 742, "leaf-b": 742},
        "validation": {"base/null": 200, "leaf-a": 185, "leaf-b": 185},
        "test": {"base/null": 200, "leaf-a": 185, "leaf-b": 185},
    }
    assert (
        manifest["evaluation_contract"]["curriculum_validation_test_use"]
        == "optimization holdouts only; not independent generalization evidence"
    )
    for split, long_minimum in (("train", 64), ("validation", 16), ("test", 16)):
        for leaf in ("leaf-a", "leaf-b"):
            accounting = manifest["length_origin_style_accounting"][split][leaf]
            assert accounting["curriculum_selected"] == (128 if split == "train" else 32)
            assert accounting["short_minimum_met"] is True
            assert accounting["long_source_minimum_met"] is True
            assert accounting["selected_long_source_rows"] >= long_minimum
            assert accounting["source_styles_selected"]["synthetic"] == (
                614 if split == "train" else 153
            )
            assert "source-style-cap-limited" in accounting["sparse_support_reasons"]

    output_prompts: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (first / f"router-examples-v2.{split}.jsonl").read_text().splitlines()
        ]
        output_prompts[split] = {row["prompt"] for row in rows}
    document = _curriculum_document()
    variants = document["anchor_surface_variants"]
    anchors = document["leaf_anchors"]
    assert isinstance(variants, list)
    assert isinstance(anchors, dict)
    seed_owners = {split: 0 for split in ("train", "validation", "test")}
    for seeds in anchors.values():
        for seed in seeds:
            expanded = {template.format(prompt=seed) for template in variants}
            owners = [
                split for split, prompts in output_prompts.items() if expanded.intersection(prompts)
            ]
            assert len(owners) == 1
            assert expanded.issubset(output_prompts[owners[0]])
            seed_owners[owners[0]] += 1
    assert seed_owners == {"train": 64, "validation": 16, "test": 16}
    for split in ("train", "validation", "test"):
        artifact = manifest["outputs"][split]
        output = first / artifact["path"]
        assert artifact["sha256"] == _sha256(output)
        assert _sha256(output) == _sha256(second / artifact["path"])


def test_v2_rejects_curriculum_collision_with_chat(tmp_path: Path) -> None:
    collected = _collection(tmp_path)
    document = _curriculum_document()
    anchors = document["leaf_anchors"]
    assert isinstance(anchors, dict)
    anchors["leaf-a"][0] = "source leaf-a train prompt 0000"
    document["content_sha256"] = fingerprint_json(
        {key: value for key, value in document.items() if key != "content_sha256"}
    )
    curriculum = _write_curriculum(tmp_path / "router-curriculum-v2.json", document)
    output = tmp_path / "router-v2"

    with pytest.raises(ValueError, match="collides with admitted chat supervision"):
        build_router_dataset_v2(
            collected,
            curriculum_path=curriculum,
            output_directory=output,
        )
    assert not output.exists()


def test_v2_rejects_evaluation_linked_input_path_before_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "molor-core-route-contract.json"
    with pytest.raises(ValueError, match="refuses benchmark-linked curriculum inputs"):
        load_router_curriculum_v2(forbidden, expected_leaf_ids=("leaf-a", "leaf-b"))
