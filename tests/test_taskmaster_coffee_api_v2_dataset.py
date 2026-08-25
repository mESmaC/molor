from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "first-real-experts"
BUILDER_PATH = EXPERIMENT / "build_taskmaster_coffee_api_v2.py"
SOURCE_MANIFEST_PATH = EXPERIMENT / "taskmaster-coffee-api-v2-source.json"
RECIPE_PATH = EXPERIMENT / "taskmaster-coffee-api-v2-recipe.json"
POLICY_PATH = EXPERIMENT / "taskmaster-coffee-api-v2-evaluation-policy.json"
SOURCE_ROOT = EXPERIMENT / "sources" / "taskmaster"


@dataclass(frozen=True)
class FixturePaths:
    source: Path
    source_manifest: Path
    recipe: Path
    policy: Path
    v1_provenance: tuple[Path, ...]
    conversation_ids: dict[str, tuple[str, ...]]


def _load_builder(name: str):
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location(name, BUILDER_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _annotation(name: str, value: str, context: str) -> dict[str, str]:
    return {"name": name, "value": value, "context": context}


def _dialogue(
    conversation_id: str,
    user_text: str,
    calls: list[dict[str, object]],
) -> dict[str, object]:
    annotations: list[dict[str, str]] = []
    for api_index, call in enumerate(calls):
        api_name = str(call["name"])
        call_context = f"api_call_{api_index}"
        response_context = f"api_response_{api_index}"
        annotations.append(_annotation("api_call", api_name, call_context))
        if not call.get("omit_request", False):
            request_value = call.get("request_json")
            if request_value is None:
                request_value = json.dumps(
                    call["arguments"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            annotations.append(_annotation("request", str(request_value), call_context))
        annotations.append(_annotation("api_response", api_name, response_context))
        response_value = call.get("response_json")
        if response_value is None:
            response_value = json.dumps(
                call["response"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        annotations.append(_annotation("response", str(response_value), response_context))
    return {
        "conversation_id": conversation_id,
        "instructions": None,
        "scenario": "Deterministic full-dialog Taskmaster coffee fixture",
        "utterances": [
            {
                "annotations": annotations,
                "index": 0,
                "segments": [],
                "speaker": "user",
                "text": user_text,
            },
            {
                "annotations": [],
                "index": 1,
                "segments": [],
                "speaker": "assistant",
                "text": "I have updated the coffee order.",
            },
        ],
        "vertical": "Coffee",
    }


def _id_for_split(builder, split: str, prefix: str) -> str:
    for index in range(10_000):
        conversation_id = f"dlg-{prefix}-{index:04d}"
        group_id = builder.group_id_for_conversations((conversation_id,))
        if builder.split_for_group(group_id) == split:
            return conversation_id
    raise AssertionError(f"could not find a deterministic {split} fixture ID")


def _pair_for_split(builder, split: str, prefix: str) -> tuple[str, str]:
    for index in range(10_000):
        pair = (f"dlg-{prefix}-{index:04d}-a", f"dlg-{prefix}-{index:04d}-b")
        group_id = builder.group_id_for_conversations(pair)
        if builder.split_for_group(group_id) == split:
            return pair
    raise AssertionError(f"could not find a deterministic {split} fixture ID pair")


def _v1_provenance_row(conversation_id: str, split: str) -> dict[str, object]:
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
    reward_file = {"train": "00", "validation": "01", "test": "02"}[split]
    return {
        "context_sha256": digest,
        "example_id": f"fixture-v1:{digest}",
        "group_id": f"fixture-v1-group:{digest}",
        "license_spdx": "CC-BY-4.0",
        "schema_version": "molor.taskmaster-coffee-api-provenance.v1",
        "source_commit": "d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
        "source_records": [
            {
                "conversation_id": conversation_id,
                "source_file": f"TM-4-2024/reward/reward_{reward_file}.json",
                "source_record_sha256": digest,
                "source_row_index": 0,
            }
        ],
        "split": split,
        "target_sha256": digest,
        "training_messages_sha256": digest,
    }


def _fixture(builder, tmp_path: Path) -> FixturePaths:
    train_pair = _pair_for_split(builder, "train", "train-shared-context")
    train_show_menu = _id_for_split(builder, "train", "train-show-menu")
    validation = _id_for_split(builder, "validation", "validation-heldout-key")
    test = _id_for_split(builder, "test", "test-heldout-key")
    invalid = "dlg-invalid-whole-dialog"

    get_menu_response = {
        "menu_items": [{"menu_item_id": "latte-1", "name": "Latte"}],
    }
    rows = [
        *[
            _dialogue(
                conversation_id,
                "shared training request for a latte",
                [
                    {
                        "name": "get_menu_items",
                        "arguments": {"query": "latte"},
                        "response": get_menu_response,
                    }
                ],
            )
            for conversation_id in train_pair
        ],
        _dialogue(
            train_show_menu,
            "show me the menu",
            [
                {
                    "name": "show_menu",
                    "arguments": {},
                    "omit_request": True,
                    "response": {"success": True},
                }
            ],
        ),
        _dialogue(
            validation,
            "validation request for a mocha",
            [
                {
                    "name": "get_menu_items",
                    "arguments": {"query": "mocha", "validation_only": "heldout"},
                    "response": get_menu_response,
                }
            ],
        ),
        _dialogue(
            test,
            "test request for tea",
            [
                {
                    "name": "get_menu_items",
                    "arguments": {"query": "tea", "test_only": "heldout"},
                    "response": get_menu_response,
                }
            ],
        ),
        _dialogue(
            invalid,
            "a malformed two-step order that must be entirely discarded",
            [
                {
                    "name": "update_order",
                    "arguments": {"order_id": "whole-dialog-first-call"},
                    "response": {"success": True},
                },
                {
                    "name": "get_order_details",
                    "arguments": {"order_id": "broken-response"},
                    "response": {},
                    "response_json": '{"order_id":',
                },
            ],
        ),
    ]

    source = tmp_path / "source"
    release = source / "TM-4-2024"
    data_file = release / "data" / "data_00.json"
    readme = release / "README.md"
    _write_json(data_file, rows)
    readme.write_text(
        "Taskmaster-4 fixture. License: Creative Commons Attribution 4.0.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "molor.source-pin.v1",
        "source_id": "google-taskmaster-tm4-2024-full-dialog",
        "repository_url": "https://github.com/google-research-datasets/Taskmaster.git",
        "resolved_commit": "d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
        "release_path": "TM-4-2024",
        "collection_method": "self-dialog with turn-level API annotations",
        "license": {
            "spdx_id": "CC-BY-4.0",
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "attribution": (
            "Taskmaster-4 is the work of Bill Byrne and Saravanan Ganesh from Google LLC."
        ),
        "paper": {
            "title": "PERL: Parameter Efficient Reinforcement Learning from Human Feedback",
            "url": "https://arxiv.org/abs/2403.10704",
        },
        "files": [
            {
                "path": "TM-4-2024/README.md",
                "sha256": _sha256_file(readme),
            },
            {
                "path": "TM-4-2024/data/data_00.json",
                "sha256": _sha256_file(data_file),
            },
        ],
    }
    manifest_path = tmp_path / "taskmaster-coffee-api-v2-source.json"
    _write_json(manifest_path, manifest)

    recipe_path = tmp_path / RECIPE_PATH.name
    policy_path = tmp_path / POLICY_PATH.name
    recipe_path.write_bytes(RECIPE_PATH.read_bytes())
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    identity_inputs = policy["dataset_contract"]["artifact_identity_inputs"]
    identity_inputs["builder_sha256"] = _sha256_file(BUILDER_PATH)
    identity_inputs["source_manifest_sha256"] = _sha256_file(manifest_path)
    identity_inputs["recipe_sha256"] = _sha256_file(recipe_path)
    _write_json(policy_path, policy)

    v1_paths: list[Path] = []
    for split in ("train", "validation", "test"):
        path = tmp_path / f"taskmaster-coffee-api-v1.{split}.provenance.jsonl"
        _write_jsonl(path, [_v1_provenance_row(f"dlg-v1-disjoint-{split}", split)])
        v1_paths.append(path)
    return FixturePaths(
        source=source,
        source_manifest=manifest_path,
        recipe=recipe_path,
        policy=policy_path,
        v1_provenance=tuple(v1_paths),
        conversation_ids={
            "train_pair": train_pair,
            "train_show_menu": (train_show_menu,),
            "validation": (validation,),
            "test": (test,),
            "invalid": (invalid,),
        },
    )


def _build(builder, fixture: FixturePaths, output: Path) -> dict[str, Any]:
    return builder.build_dataset(
        source=fixture.source,
        output=output,
        source_manifest_path=fixture.source_manifest,
        recipe_path=fixture.recipe,
        policy_path=fixture.policy,
        v1_provenance_paths=fixture.v1_provenance,
        verify_pin=False,
    )


def _target_from_training_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    assert messages[-1]["role"] == "assistant"
    target_text = messages[-1]["content"]
    target = json.loads(target_text)
    assert target_text == json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return target


def test_builder_is_byte_deterministic_group_disjoint_and_content_addressed(
    tmp_path: Path,
) -> None:
    from molor.training.data import load_jsonl

    builder = _load_builder("taskmaster_coffee_api_v2_determinism_test")
    fixture = _fixture(builder, tmp_path)
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = _build(builder, fixture, left)
    second = _build(builder, fixture, right)

    left_files = sorted(path.name for path in left.iterdir() if path.is_file())
    right_files = sorted(path.name for path in right.iterdir() if path.is_file())
    assert left_files == right_files
    assert left_files
    for name in left_files:
        assert (left / name).read_bytes() == (right / name).read_bytes(), name
    assert first == second

    owners: dict[str, str] = {}
    conversation_owners: dict[str, str] = {}
    expected_examples = {"train": 2, "validation": 1, "test": 1}
    catalog_path = left / f"{builder.DATASET_ID}.tool-catalog.json"
    identity_hashes = {
        "builder_sha256": _sha256_file(BUILDER_PATH),
        "source_manifest_sha256": _sha256_file(fixture.source_manifest),
        "recipe_sha256": _sha256_file(fixture.recipe),
        "policy_sha256": _sha256_file(fixture.policy),
        "tool_catalog_sha256": _sha256_file(catalog_path),
    }
    for split, expected_count in expected_examples.items():
        data_path = left / f"{builder.DATASET_ID}.{split}.jsonl"
        provenance_path = left / f"{builder.DATASET_ID}.{split}.provenance.jsonl"
        training = _read_jsonl(data_path)
        provenance = _read_jsonl(provenance_path)
        assert len(training) == len(provenance) == expected_count
        assert len(load_jsonl(data_path)) == expected_count
        assert all(set(row) == {"messages"} for row in training)
        for raw_line in data_path.read_text(encoding="utf-8").splitlines():
            assert raw_line == builder.canonical_json(json.loads(raw_line))
        for training_row, provenance_row in zip(training, provenance, strict=True):
            target = _target_from_training_row(training_row)
            assert target["role"] == "agent"
            assert len(target["api"]) == 1
            group_id = provenance_row["group_id"]
            assert owners.setdefault(group_id, split) == split
            for source_record in provenance_row["source_records"]:
                conversation_id = source_record["conversation_id"]
                assert conversation_owners.setdefault(conversation_id, split) == split
                assert {
                    "source_file",
                    "source_record_sha256",
                    "source_row_index",
                    "utterance_index",
                    "api_index",
                } <= source_record.keys()
            messages_sha256 = _sha256_bytes(
                builder.canonical_json(training_row["messages"]).encode("utf-8")
            )
            assert provenance_row["training_messages_sha256"] == messages_sha256
            for key, expected_hash in identity_hashes.items():
                assert provenance_row[key] == expected_hash

    pair = fixture.conversation_ids["train_pair"]
    expected_pair_group = builder.group_id_for_conversations(pair)
    train_provenance = _read_jsonl(left / f"{builder.DATASET_ID}.train.provenance.jsonl")
    duplicate = next(row for row in train_provenance if row["group_id"] == expected_pair_group)
    assert {row["conversation_id"] for row in duplicate["source_records"]} == set(pair)
    assert first["counts"]["examples_by_split"] == expected_examples
    assert first["counts"]["groups_by_split"] == expected_examples


def test_catalog_is_train_only_and_missing_show_menu_request_becomes_empty_object(
    tmp_path: Path,
) -> None:
    builder = _load_builder("taskmaster_coffee_api_v2_catalog_test")
    fixture = _fixture(builder, tmp_path)
    output = tmp_path / "output"
    report = _build(builder, fixture, output)

    catalog_path = output / f"{builder.DATASET_ID}.tool-catalog.json"
    catalog_text = catalog_path.read_text(encoding="utf-8")
    catalog = json.loads(catalog_text)
    assert "validation_only" not in catalog_text
    assert "test_only" not in catalog_text
    tools = {tool["name"]: tool for tool in catalog["tools"]}
    assert set(tools) == {"get_menu_items", "show_menu"}
    assert tools["get_menu_items"]["argument_key_sets"] == [["query"]]
    assert tools["get_menu_items"]["argument_types"] == {"query": ["string"]}
    assert tools["show_menu"]["argument_key_sets"] == [[]]
    assert tools["show_menu"]["argument_types"] == {}

    training = _read_jsonl(output / f"{builder.DATASET_ID}.train.jsonl")
    targets = [_target_from_training_row(row) for row in training]
    assert any(target["api"] == {"show_menu": {}} for target in targets)
    assert report["counts"]["implicit_empty_show_menu_requests"] == 1

    for split, heldout_key in (("validation", "validation_only"), ("test", "test_only")):
        rows = _read_jsonl(output / f"{builder.DATASET_ID}.{split}.jsonl")
        assert len(rows) == 1
        target = _target_from_training_row(rows[0])
        assert heldout_key in target["api"]["get_menu_items"]
        assert heldout_key not in rows[0]["messages"][0]["content"]
        evaluation = _read_jsonl(output / f"{builder.DATASET_ID}.{split}.evaluation.jsonl")
        assert len(evaluation) == 1
        assert evaluation[0]["expected_canonical"] == builder.canonical_json(
            evaluation[0]["expected"]
        )


def test_invalid_embedded_json_quarantines_the_entire_dialogue(tmp_path: Path) -> None:
    builder = _load_builder("taskmaster_coffee_api_v2_quarantine_test")
    fixture = _fixture(builder, tmp_path)
    output = tmp_path / "output"
    report = _build(builder, fixture, output)
    invalid_id = fixture.conversation_ids["invalid"][0]

    emitted_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output.glob(f"{builder.DATASET_ID}.*.jsonl"))
        if "quarantine" not in path.name
    )
    assert invalid_id not in emitted_text
    assert "whole-dialog-first-call" not in emitted_text

    quarantine_path = output / f"{builder.DATASET_ID}.quarantine.jsonl"
    quarantine = _read_jsonl(quarantine_path)
    matching = [row for row in quarantine if invalid_id in builder.canonical_json(row)]
    assert len(matching) == 1
    assert report["counts"]["source_dialogs"] == 6
    assert report["counts"]["source_api_calls"] == 7
    assert report["counts"]["valid_dialogs"] == 5
    assert report["counts"]["quarantined_invalid_dialogs"] == 1
    assert report["counts"]["quarantined_api_calls"] == 2
    assert report["counts"]["emitted_unique_examples"] == 4


def test_any_v1_reward_source_conversation_overlap_is_a_hard_failure(
    tmp_path: Path,
) -> None:
    builder = _load_builder("taskmaster_coffee_api_v2_overlap_test")
    fixture = _fixture(builder, tmp_path)
    overlap = fixture.conversation_ids["validation"][0]
    _write_jsonl(
        fixture.v1_provenance[0],
        [_v1_provenance_row(overlap, "train")],
    )

    with pytest.raises(ValueError, match="(?i)overlap"):
        _build(builder, fixture, tmp_path / "output")


def test_v1_overlap_audit_requires_all_three_typed_split_sidecars(tmp_path: Path) -> None:
    builder = _load_builder("taskmaster_coffee_api_v2_v1_contract_test")
    fixture = _fixture(builder, tmp_path)
    with pytest.raises(ValueError, match="exactly the train, validation, and test"):
        builder.build_dataset(
            source=fixture.source,
            output=tmp_path / "missing-sidecar-output",
            source_manifest_path=fixture.source_manifest,
            recipe_path=fixture.recipe,
            policy_path=fixture.policy,
            v1_provenance_paths=fixture.v1_provenance[:2],
            verify_pin=False,
        )

    wrong_schema = _v1_provenance_row("dlg-v1-disjoint-train", "train")
    wrong_schema["schema_version"] = "molor.taskmaster-coffee-api-provenance.v0"
    _write_jsonl(fixture.v1_provenance[0], [wrong_schema])
    with pytest.raises(ValueError, match="wrong v1 provenance schema"):
        _build(builder, fixture, tmp_path / "wrong-schema-output")


def test_v1_overlap_audit_rejects_filename_split_mismatch(tmp_path: Path) -> None:
    builder = _load_builder("taskmaster_coffee_api_v2_v1_split_test")
    fixture = _fixture(builder, tmp_path)
    wrong_split = _v1_provenance_row("dlg-v1-disjoint-train", "validation")
    _write_jsonl(fixture.v1_provenance[0], [wrong_split])
    with pytest.raises(ValueError, match="split does not match"):
        _build(builder, fixture, tmp_path / "wrong-split-output")


@pytest.mark.parametrize(
    "invalid_response",
    [
        '{"order_id":"first","order_id":"duplicate"}',
        '{"message":"raw\x01control"}',
    ],
)
def test_strict_embedded_json_rejects_duplicate_keys_and_raw_controls(
    tmp_path: Path,
    invalid_response: str,
) -> None:
    builder = _load_builder("taskmaster_coffee_api_v2_strict_json_test")
    fixture = _fixture(builder, tmp_path)
    data_path = fixture.source / "TM-4-2024" / "data" / "data_00.json"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    invalid_id = fixture.conversation_ids["invalid"][0]
    invalid_dialogue = next(row for row in rows if row["conversation_id"] == invalid_id)
    responses = [
        annotation
        for annotation in invalid_dialogue["utterances"][0]["annotations"]
        if annotation["name"] == "response"
    ]
    responses[-1]["value"] = invalid_response
    _write_json(data_path, rows)
    manifest = json.loads(fixture.source_manifest.read_text(encoding="utf-8"))
    data_descriptor = next(
        descriptor
        for descriptor in manifest["files"]
        if descriptor["path"] == "TM-4-2024/data/data_00.json"
    )
    data_descriptor["sha256"] = _sha256_file(data_path)
    _write_json(fixture.source_manifest, manifest)
    policy = json.loads(fixture.policy.read_text(encoding="utf-8"))
    policy["dataset_contract"]["artifact_identity_inputs"]["source_manifest_sha256"] = _sha256_file(
        fixture.source_manifest
    )
    _write_json(fixture.policy, policy)

    output = tmp_path / "output"
    report = _build(builder, fixture, output)
    quarantine = _read_jsonl(output / f"{builder.DATASET_ID}.quarantine.jsonl")
    matching = [row for row in quarantine if row.get("conversation_id") == invalid_id]
    assert len(matching) == 1
    assert matching[0]["reason"] == "invalid_embedded_json"
    assert report["counts"]["observed_invalid_embedded_json_fields"] == 1


def test_v2_recipe_remains_loadable_by_the_training_entrypoint() -> None:
    from molor.training.recipe import load_training_recipe

    recipe = load_training_recipe(RECIPE_PATH)
    assert recipe.rank == 16
    assert recipe.alpha == 32
    assert recipe.seed == 241104
    assert len(recipe.experts) == 1
    assert recipe.experts[0].name == "taskmaster-coffee-api-v2-full-dialog-r16"
    assert recipe.output_dir.name == "qwen3-1.7b-candidate-experts-quarantine"


def test_v2_source_recipe_and_policy_are_pinned_and_quarantined() -> None:
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    assert source_manifest["resolved_commit"] == ("d92cb6af3005f1dc09c39e75e7daf4a04905e00b")
    assert source_manifest["license"]["spdx_id"] == "CC-BY-4.0"
    assert "Bill Byrne" in source_manifest["attribution"]
    assert "Saravanan Ganesh" in source_manifest["attribution"]
    data_files = [entry for entry in source_manifest["files"] if "/data/" in entry["path"]]
    assert len(data_files) == 8
    assert all("/reward/" not in entry["path"] for entry in source_manifest["files"])
    for descriptor in source_manifest["files"]:
        assert len(descriptor["sha256"]) == 64
        assert _sha256_file(SOURCE_ROOT / descriptor["path"]) == descriptor["sha256"]

    assert recipe["rank"] == 16
    assert recipe["alpha"] == 32
    assert recipe["seed"] == 241104
    assert recipe["chat_template_kwargs"] == {"enable_thinking": False}
    assert set(recipe["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert recipe["output_dir"].endswith("candidate-experts-quarantine")
    assert recipe["experts"] == [
        {
            "name": "taskmaster-coffee-api-v2-full-dialog-r16",
            "dataset": "data/taskmaster-coffee-api-v2.train.jsonl",
        }
    ]

    assert policy["candidate_posture"] == "quarantine_until_every_gate_passes"
    contract = policy["dataset_contract"]
    assert contract["dataset_id"] == "taskmaster-coffee-api-v2"
    assert contract["v1_reward_source_conversation_overlap_required"] == 0
    assert contract["tool_catalog_source"] == "train split only"
    assert "full-dialog" in contract["source_scope"]
    assert "1,623" in contract["v1_overlap_scope"]
    identity_inputs = contract["artifact_identity_inputs"]
    assert identity_inputs["builder_sha256"] == _sha256_file(BUILDER_PATH)
    assert identity_inputs["source_manifest_sha256"] == _sha256_file(SOURCE_MANIFEST_PATH)
    assert identity_inputs["recipe_sha256"] == _sha256_file(RECIPE_PATH)
    required = set(policy["training_artifact_contract"]["required_file_identities"])
    assert {
        "builder_sha256",
        "source_manifest_sha256",
        "recipe_sha256",
        "policy_sha256",
        "build_report_sha256",
        "train_dataset_sha256",
        "train_provenance_sha256",
        "tool_catalog_sha256",
        "trainer_source_tree_sha256",
    } <= required
    trainer_source = policy["training_artifact_contract"]["trainer_source_contract"]
    assert len(trainer_source["content_fingerprint"]) == 64
    base = policy["training_artifact_contract"]["base_artifact"]
    assert base["content_fingerprint"] == (
        "772bbbfd94e3a63f48e2f315ea8f4d2318c66290c1d98d624dd9bc679c85bb7c"
    )
    assert "does not authorize router" in policy["registration_policy"]
    assert policy["evaluation"]["status"] == (
        "blocked_pending_candidate_training_and_sealed_preaccess_receipt"
    )
    assert len(policy["evaluation"]["required_before_test_access"]) == 4
