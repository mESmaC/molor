from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from molor.training.recipe import load_training_recipe


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "first-real-experts"
BUILDER_PATH = EXPERIMENT / "build_taskmaster_coffee_api.py"
EVALUATOR_PATH = EXPERIMENT / "evaluate_taskmaster_coffee_api.py"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


def _conversation(query: str) -> list[dict[str, object]]:
    return [
        {"role": "agent", "text": "What can I get for you?"},
        {"role": "user", "text": query},
    ]


def _source_row(
    identifier: str,
    context: list[dict[str, object]],
    generation: dict[str, object],
    reward: float = 1.0,
) -> dict[str, object]:
    return {
        "conversation_id": identifier,
        "context": json.dumps(context),
        "generation": json.dumps(generation),
        "reward": reward,
    }


def _fixture(builder, tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    source = tmp_path / "source"
    release = source / "TM-4-2024"
    reward_dir = release / "reward"
    reward_dir.mkdir(parents=True)
    (release / "README.md").write_text("CC BY 4.0 fixture\n", encoding="utf-8")

    contexts: dict[str, list[dict[str, object]]] = {}
    candidate_index = 0
    while set(contexts) != set(builder.SPLITS):
        context = _conversation(f"fixture coffee request {candidate_index}")
        split = builder.split_for_group(builder.group_id(context))
        contexts.setdefault(split, context)
        candidate_index += 1
    calls = {
        "train": {"get_menu_items": {"query": "latte", "limit": 2}},
        "validation": {"get_addons": {"menu_item_id": "latte-1", "query": "oat"}},
        "test": {"finish_order": {"order_id": "order-1"}},
    }
    rows = [
        _source_row(
            f"accepted-{split}",
            contexts[split],
            {"role": "agent", "api": calls[split]},
        )
        for split in builder.SPLITS
    ]
    rows.append(
        _source_row(
            "accepted-train-duplicate",
            contexts["train"],
            {"api": calls["train"], "role": "agent"},
        )
    )
    conflicting_context = _conversation("an ambiguous coffee order")
    rows.extend(
        [
            _source_row(
                "conflict-a",
                conflicting_context,
                {"role": "agent", "api": {"show_menu": {}}},
            ),
            _source_row(
                "conflict-b",
                conflicting_context,
                {"role": "agent", "api": {"get_menu_items": {"query": "coffee"}}},
            ),
            _source_row(
                "invalid-api",
                _conversation("invalid"),
                {"role": "agent", "api": {"": {}}},
            ),
            _source_row(
                "text-generation",
                _conversation("say something"),
                {"role": "agent", "text": "What size?"},
            ),
            _source_row(
                "negative-api",
                _conversation("wrong action"),
                {"role": "agent", "api": {"show_menu": {}}},
                reward=0.0,
            ),
        ]
    )
    reward_path = reward_dir / "reward_00.json"
    reward_path.write_text(json.dumps(rows), encoding="utf-8")
    manifest = {
        "schema_version": "molor.source-pin.v1",
        "source_id": "taskmaster-fixture",
        "repository_url": "https://example.invalid/taskmaster.git",
        "resolved_commit": "0" * 40,
        "release_path": "TM-4-2024",
        "collection_method": "self-dialog",
        "license": {
            "spdx_id": "CC-BY-4.0",
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "attribution": "Taskmaster fixture attribution",
        "paper": {"title": "Fixture", "url": "https://example.invalid/paper"},
        "files": [
            {"path": "TM-4-2024/README.md", "sha256": "0" * 64},
            {"path": "TM-4-2024/reward/reward_00.json", "sha256": "0" * 64},
        ],
    }
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return (
        source,
        manifest_path,
        {split: builder.group_id(context) for split, context in contexts.items()},
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builder_is_content_addressed_group_disjoint_and_auditable(tmp_path: Path) -> None:
    builder = _load_module("taskmaster_builder_test", BUILDER_PATH)
    source, manifest, expected_groups = _fixture(builder, tmp_path)
    output = tmp_path / "output"

    report = builder.build_dataset(
        source=source,
        output=output,
        source_manifest_path=manifest,
        verify_pin=False,
    )

    assert report["counts"] == {
        "source_rows": 9,
        "positive_api_source_rows_before_conflicts": 6,
        "quarantined_invalid_source_rows": 1,
        "excluded_source_rows": {
            "non_api_generation": 1,
            "non_positive_reward": 1,
        },
        "conflicting_contexts": 1,
        "conflicting_positive_target_pairs": 2,
        "conflicting_source_rows": 2,
        "deduplicated_source_rows": 1,
        "emitted_unique_examples": 3,
        "emitted_groups": 3,
        "examples_by_split": {split: 1 for split in builder.SPLITS},
        "groups_by_split": {split: 1 for split in builder.SPLITS},
        "api_calls_by_split": {
            "train": {"get_menu_items": 1},
            "validation": {"get_addons": 1},
            "test": {"finish_order": 1},
        },
    }
    owners: dict[str, str] = {}
    for split in builder.SPLITS:
        training = _jsonl(output / f"{builder.DATASET_ID}.{split}.jsonl")
        provenance = _jsonl(output / f"{builder.DATASET_ID}.{split}.provenance.jsonl")
        assert len(training) == len(provenance) == 1
        assert provenance[0]["group_id"] == expected_groups[split]
        assert owners.setdefault(str(provenance[0]["group_id"]), split) == split
        messages = training[0]["messages"]
        assert provenance[0]["training_messages_sha256"] == builder.sha256_text(
            builder.canonical_json(messages)
        )
        assistant = messages[-1]["content"]
        assert assistant == builder.canonical_json(json.loads(assistant))
    train_provenance = _jsonl(output / f"{builder.DATASET_ID}.train.provenance.jsonl")
    assert len(train_provenance[0]["source_records"]) == 2
    quarantine = _jsonl(output / f"{builder.DATASET_ID}.quarantine.jsonl")
    assert [row["reason"] for row in quarantine] == [
        "ambiguous_positive_targets",
        "ambiguous_positive_targets",
        "invalid_generation_api",
    ]
    assert report["source"]["license"]["spdx_id"] == "CC-BY-4.0"
    assert report["intended_checkpoint_posture"] == "candidate_quarantine_only_never_auto_promote"


def test_builder_bytes_and_evaluation_inputs_are_deterministic(tmp_path: Path) -> None:
    builder = _load_module("taskmaster_builder_determinism_test", BUILDER_PATH)
    source, manifest, _ = _fixture(builder, tmp_path)
    left, right = tmp_path / "left", tmp_path / "right"
    first = builder.build_dataset(
        source=source,
        output=left,
        source_manifest_path=manifest,
        verify_pin=False,
    )
    second = builder.build_dataset(
        source=source,
        output=right,
        source_manifest_path=manifest,
        verify_pin=False,
    )

    for key, descriptor in first["outputs"].items():
        assert descriptor["sha256"] == second["outputs"][key]["sha256"]
    for split in ("validation", "test"):
        evaluation = _jsonl(left / f"{builder.DATASET_ID}.{split}.evaluation.jsonl")
        assert len(evaluation) == 1
        row = evaluation[0]
        assert len(row["messages"]) == 2
        assert row["expected_canonical"] == builder.canonical_json(row["expected"])
        assert row["expected_sha256"] == builder.sha256_text(row["expected_canonical"])


def test_strict_json_rejects_duplicate_keys_and_scorer_separates_metrics(
    tmp_path: Path,
) -> None:
    builder = _load_module("taskmaster_builder_scorer_test", BUILDER_PATH)
    evaluator = _load_module("taskmaster_evaluator_test", EVALUATOR_PATH)
    with pytest.raises(builder.SourceRowError, match="duplicate JSON object key"):
        builder.strict_json_loads('{"api":{},"api":{}}')

    source, manifest, _ = _fixture(builder, tmp_path)
    output = tmp_path / "output"
    builder.build_dataset(
        source=source,
        output=output,
        source_manifest_path=manifest,
        verify_pin=False,
    )
    cases = evaluator.load_evaluation(output / f"{builder.DATASET_ID}.validation.evaluation.jsonl")
    case = cases[0]
    exact = evaluator.score_one(case, case.expected_canonical)
    pretty = evaluator.score_one(case, json.dumps(case.expected, indent=2))
    fenced = evaluator.score_one(case, f"```json\n{case.expected_canonical}\n```")

    assert exact["semantic_exact"] is True
    assert exact["canonical_exact"] is True
    assert pretty["semantic_exact"] is True
    assert pretty["canonical_exact"] is False
    assert fenced["json_parseable"] is False
    assert fenced["semantic_exact"] is False


def test_rank16_recipe_keeps_candidate_output_quarantined() -> None:
    recipe = load_training_recipe(EXPERIMENT / "taskmaster-coffee-api-v1-recipe.json")

    assert recipe.rank == 16
    assert recipe.alpha == 32
    assert recipe.chat_template_kwargs == {"enable_thinking": False}
    assert recipe.experts[0].name == "taskmaster-coffee-api-v1"
    assert recipe.output_dir.name == "qwen3-1.7b-candidate-experts-quarantine"


def test_evaluation_policy_keeps_the_v1_test_sealed_after_scoring() -> None:
    policy = json.loads(
        (EXPERIMENT / "taskmaster-coffee-api-v1-evaluation-policy.json").read_text(encoding="utf-8")
    )

    assert policy["candidate_posture"] == "quarantine_until_every_gate_passes"
    assert policy["evaluation"]["dataset_sha256"] == (
        "dbc04384ef9b0481655494bdcd8edb15979944d532fd609d0670b43472dc07c1"
    )
    assert policy["minimum_rates"] == {
        "json_parseable": 0.98,
        "schema_valid": 0.98,
        "api_name_exact": 0.85,
        "arguments_exact": 0.7,
        "semantic_exact": 0.6,
    }
    assert policy["minimum_absolute_improvement_over_base"] == 0.1
    assert "Do not tune" in policy["reuse_policy"]


def test_evaluator_base_mode_requires_model_and_is_mutually_exclusive() -> None:
    evaluator = _load_module("taskmaster_evaluator_cli_test", EVALUATOR_PATH)

    with pytest.raises(SystemExit):
        evaluator.parse_args(["--evaluation", "holdout.jsonl", "--base"])
    with pytest.raises(SystemExit):
        evaluator.parse_args(
            [
                "--evaluation",
                "holdout.jsonl",
                "--base",
                "--adapter",
                "adapter",
                "--model",
                "model",
            ]
        )

    parsed = evaluator.parse_args(["--evaluation", "holdout.jsonl", "--base", "--model", "model"])
    assert parsed.base is True
    assert parsed.adapter is None
    assert parsed.model == Path("model")
