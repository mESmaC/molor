from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from molor.experimental.corpus_mixtures import (
    CorpusShardDescriptor,
    load_corpus_mixture_recipe,
)
from molor.experimental.expert_mixture_registry import (
    EXPERT_CHAT_RECEIPT_SCHEMA,
    EXPERT_CHAT_REGISTRY_SCHEMA,
    ROUTER_EXAMPLES_MANIFEST_SCHEMA,
    CollectedExpertData,
    RegisteredChatShard,
    _resolve_router_owners,
    build_canonical_recipes,
    write_expert_mixture_bundle,
)
from molor.identity import canonical_json_bytes, fingerprint_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_BUNDLE = REPOSITORY_ROOT / "experiments" / "expert-rack-v5" / "chat-mixtures-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_chat(path: Path, prompt: str, answer: str) -> tuple[str, int]:
    row = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }
    payload = canonical_json_bytes(row) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), 1


def _synthetic_collection(root: Path) -> CollectedExpertData:
    taxonomy_content: dict[str, object] = {
        "schema_version": "molor.expert-taxonomy.v1",
        "taxonomy_id": "synthetic",
        "revision": "1",
        "families": [],
        "aliases": {},
    }
    taxonomy = {**taxonomy_content, "content_sha256": fingerprint_json(taxonomy_content)}
    shards: list[RegisteredChatShard] = []
    definitions = (
        ("leaf-a", "direct", "direct-v1"),
        ("leaf-b", "trace", "trace-v1"),
        # Semantic tool-call SFT is intentionally rendered through direct chat.
        ("leaf-tool", "tool", "direct-v1"),
    )
    for leaf, stream, template in definitions:
        for split in ("train", "validation", "test"):
            path = root / "data" / leaf / stream / f"{split}.jsonl"
            prompt = (
                "x" * 9_000
                if leaf == "leaf-a" and split == "train"
                else f"{leaf} {stream} {split} prompt"
            )
            sha256, records = _write_chat(path, prompt, f"{leaf} answer")
            descriptor = CorpusShardDescriptor(
                shard_id=f"synthetic/{leaf}/{stream}/{split}",
                shard_revision="source-v1",
                leaf_id=leaf,
                leaf_revision="taxonomy-v1",
                split=split,
                template_mode=template,
                path=str(path.resolve()),
                sha256=sha256,
                record_count=records,
            )
            shards.append(
                RegisteredChatShard(
                    descriptor=descriptor,
                    source_id="synthetic",
                    source_revision="source-v1",
                    semantic_stream=stream,
                    recipe_stream=stream,
                    length_tier="le-1024",
                    admission_role="primary",
                )
            )
    return CollectedExpertData(
        taxonomy=taxonomy,
        taxonomy_path=str((root / "taxonomy.json").resolve()),
        taxonomy_sha256="a" * 64,
        leaf_revision="taxonomy-v1",
        family_by_leaf={
            "leaf-a": "family-a",
            "leaf-b": "family-b",
            "leaf-tool": "family-tool",
        },
        shards=tuple(shards),
        references=(),
        sources=(
            {
                "source_id": "synthetic",
                "source_revision": "source-v1",
                "root": str(root.resolve()),
                "source_role": "primary",
                "manifest_pins": [],
                "declared_output_entries": 9,
                "chat_nonempty_shards": 9,
                "chat_empty_shards": 0,
                "blocked_chat_shards": 0,
                "blocked_chat_records": 0,
                "reference_shards": 0,
                "auxiliary_artifacts": {},
            },
        ),
        dispositions=(),
    )


def _write_router_base(root: Path) -> Path:
    outputs: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        path = root / f"router-base.{split}.jsonl"
        rows = [
            {"prompt": f"ordinary conversation {split}", "route_id": "base/null"},
            {"prompt": f"legacy expert {split}", "route_id": "legacy-expert"},
        ]
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        path.write_bytes(payload)
        outputs[split] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    report = {"schema_version": "synthetic-router-v6", "outputs": outputs}
    report_path = root / "router-base.report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_recipes_are_per_leaf_adjustable_and_globally_guarded(tmp_path: Path) -> None:
    collected = _synthetic_collection(tmp_path)
    recipes = build_canonical_recipes(collected)
    assert len(recipes) == 18

    train = recipes["recipes/fast/leaf-a/direct/train.json"]
    assert train.usage == "training"
    assert {item.split for item in train.leakage_guard_shards} == {"validation", "test"}
    assert len(train.leakage_guard_shards) == 6
    assert train.template_mode == "direct-v1"

    validation = recipes["recipes/full/leaf-b/trace/validation.json"]
    assert validation.usage == "evaluation"
    assert {item.split for item in validation.leakage_guard_shards} == {"train"}
    assert len(validation.leakage_guard_shards) == 3

    tool = recipes["recipes/fast/leaf-tool/tool/train.json"]
    assert tool.template_mode == "direct-v1"
    assert tool.shards[0].descriptor.leaf_id == "leaf-tool"


def test_router_owner_resolution_prefers_specific_sources_then_heldout_split() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            """
            CREATE TABLE occurrences(
                digest TEXT NOT NULL,
                normalized TEXT NOT NULL,
                prompt TEXT NOT NULL,
                route_id TEXT NOT NULL,
                split TEXT NOT NULL,
                selection_rank TEXT NOT NULL,
                origin TEXT NOT NULL,
                route_priority INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                # Curated terminal supervision owns this prompt even though the broad
                # copy appears in the higher-precedence test split.
                ("a", "same", "same", "terminal-cli", "train", "2", "curated", 0),
                ("a", "same", "same", "tool-use-execution", "test", "1", "broad", 2),
                # Equal-priority disagreement remains excluded.
                ("b", "ambiguous", "ambiguous", "leaf-a", "train", "1", "one", 1),
                ("b", "ambiguous", "ambiguous", "leaf-b", "test", "2", "two", 1),
                # Once the route agrees, held-out precedence owns duplicate prompts.
                ("c", "heldout", "heldout", "leaf-a", "train", "1", "one", 1),
                ("c", "heldout", "heldout", "leaf-a", "validation", "2", "two", 1),
            ),
        )
        accounting = _resolve_router_owners(connection)
        owners = connection.execute(
            "SELECT digest, route_id, split FROM owners ORDER BY digest"
        ).fetchall()
    finally:
        connection.close()

    assert owners == [
        ("a", "terminal-cli", "train"),
        ("c", "leaf-a", "validation"),
    ]
    assert accounting["owners"] == 2
    assert accounting["resolved_conflict_losing_occurrences"] == 1
    assert accounting["unresolved_conflict_occurrences"] == 2
    assert accounting["duplicate_or_cross_split_occurrences"] == 1


def test_bundle_and_router_rows_are_content_addressed_and_strict(tmp_path: Path) -> None:
    collected = _synthetic_collection(tmp_path)
    recipes = build_canonical_recipes(collected)
    report_path = _write_router_base(tmp_path)
    output = tmp_path / "bundle"
    router_output = tmp_path / "router-dataset"
    summary = write_expert_mixture_bundle(
        collected,
        recipes=recipes,
        router_base_report=report_path,
        output_directory=output,
        router_output_directory=router_output,
    )
    assert summary["recipes"] == 18

    registry = json.loads((output / "shard-registry.json").read_bytes())
    assert registry["schema_version"] == EXPERT_CHAT_REGISTRY_SCHEMA
    claimed = registry.pop("content_sha256")
    assert claimed == fingerprint_json(registry)

    receipt = json.loads((output / "generation-receipt.json").read_bytes())
    assert receipt["schema_version"] == EXPERT_CHAT_RECEIPT_SCHEMA
    claimed = receipt.pop("content_sha256")
    assert claimed == fingerprint_json(receipt)

    router_manifest = json.loads((router_output / "router-examples-v1.manifest.json").read_bytes())
    assert router_manifest["schema_version"] == ROUTER_EXAMPLES_MANIFEST_SCHEMA
    assert router_manifest["accounting"]["zero_silent_drops"] is True
    assert router_manifest["accounting"]["legacy_expert_rows_not_admitted"] == 3
    assert router_manifest["accounting"]["chat_prompts_length_bounded"] == 1
    assert router_manifest["evaluation_contract"]["train_and_validation"] == (
        "fail-closed-full-route-support"
    )
    assert all(
        router_manifest["route_coverage"][split]["full_expert_route_support"]
        for split in ("train", "validation", "test")
    )
    for split in ("train", "validation", "test"):
        artifact = router_manifest["outputs"][split]
        path = router_output / artifact["path"]
        assert _sha256(path) == artifact["sha256"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert all(set(row) == {"prompt", "route_id"} for row in rows)
        assert all(len(row["prompt"]) <= 8_192 for row in rows)
        assert any(row["route_id"] == "base/null" for row in rows)

    recipe = load_corpus_mixture_recipe(
        output / "recipes" / "fast" / "leaf-a" / "direct" / "train.json"
    )
    assert recipe.recipe_id == "rack-v5-leaf-a-direct-fast-train"
    assert (output / "router-examples-v1.manifest.json").is_file()
    assert not (output / "router").exists()


def test_checked_in_registry_recipes_and_router_are_pinned() -> None:
    registry_path = CHECKED_IN_BUNDLE / "shard-registry.json"
    assert registry_path.is_file()
    registry = json.loads(registry_path.read_bytes())
    claimed = registry.pop("content_sha256")
    assert claimed == fingerprint_json(registry)
    assert registry["summary"]["taxonomy_leaves"] == 25
    assert (
        registry["summary"]["chat_ready_leaves"]
        + registry["summary"]["gap_or_reference_only_leaves"]
        == 25
    )
    assert registry["summary"]["chat_ready_leaves"] == 25
    assert registry["summary"]["gap_leaf_ids"] == []
    assert registry["summary"]["chat_shards"] == 364
    assert registry["summary"]["chat_records_declared"] == 883_073
    assert registry["summary"]["recipes"] == 258
    native_tool = registry["source_semantics"]["broad-distillation-experts-v1/native-tool"]
    assert native_tool["trainer_contract"] == "strict-lossless-qwen-compatible"
    assert native_tool["recipe_streams"] == ["tool-direct", "tool-trace"]
    embedded_source = next(
        source
        for source in registry["source_registry"]
        if source["source_id"] == "embedded-grounded-reconstruction-v1"
    )
    assert embedded_source["registration_fragment"]["state"] == "supervised_ready"

    for item in registry["recipes"]:
        path = CHECKED_IN_BUNDLE / item["path"]
        assert _sha256(path) == item["sha256"]
        recipe = load_corpus_mixture_recipe(path)
        assert recipe.fingerprint == item["content_sha256"]
        if recipe.usage == "training":
            assert {guard.split for guard in recipe.leakage_guard_shards} == {
                "validation",
                "test",
            }
        else:
            assert {guard.split for guard in recipe.leakage_guard_shards} == {"train"}

    router = registry["router_examples"]
    manifest_path = CHECKED_IN_BUNDLE / router["registry_manifest_copy_path"]
    assert _sha256(manifest_path) == router["manifest_sha256"]
    router_manifest = json.loads(manifest_path.read_bytes())
    router_claimed = router_manifest.pop("content_sha256")
    assert router_claimed == fingerprint_json(router_manifest)
    assert router_manifest["route_coverage"]["train"]["full_expert_route_support"] is True
    assert router_manifest["route_coverage"]["validation"]["full_expert_route_support"] is True
    assert router_manifest["route_coverage"]["test"]["expert_routes_missing"] == ["terminal-cli"]
    assert router_manifest["evaluation_contract"]["test"]["unsupported_route_recall"] is None
