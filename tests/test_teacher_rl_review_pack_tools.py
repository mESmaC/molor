from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT = WORKSPACE / "experiments" / "teacher-rl-v1" / "review_pack_tools.py"
TEACHER = {"id": "blind-teacher-v1", "model": "teacher-model", "provider": "local"}


@pytest.fixture(scope="module")
def tools() -> ModuleType:
    spec = importlib.util.spec_from_file_location("molor_test_review_pack_tools", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _review_pack() -> dict[str, object]:
    groups = []
    cursor = 0
    for group_index, size in enumerate((3, 2, 1, 1)):
        candidates = []
        for _ in range(size):
            candidates.append(
                {
                    "candidate_key": f"candidate-{cursor}",
                    "generation_id": f"generation-{cursor}",
                    "messages": [{"role": "user", "content": f"question {group_index}"}],
                    "response": f"answer {cursor}",
                }
            )
            cursor += 1
        groups.append(
            {
                "prompt_key": f"prompt-{group_index}",
                "evaluation": {"check": f"check {group_index}"},
                "messages": [{"role": "user", "content": f"question {group_index}"}],
                "candidates": candidates,
            }
        )
    return {
        "schema_version": "molor.teacher-eval.review-pack.v1",
        "created_at": "2026-08-24T00:00:00Z",
        "run_id": "review-tools-test",
        "manifest_sha256": "a" * 64,
        "raw_report": "raw.jsonl",
        "raw_report_sha256": "b" * 64,
        "rubric": {"version": "molor.teacher-rubric.v1", "dimensions": {}},
        "instructions": ["Grade blind."],
        "judgments_template": {
            "schema_version": "molor.teacher-eval.judgments.v1",
            "run_id": "review-tools-test",
            "rubric_version": "molor.teacher-rubric.v1",
            "teacher": {"id": "REQUIRED", "model": "REQUIRED", "provider": "REQUIRED"},
            "grades": [],
            "preferences": [],
        },
        "groups": groups,
    }


def _grade(candidate_key: str) -> dict[str, object]:
    return {
        "candidate_key": candidate_key,
        "scores": {"overall": 1, "accuracy": 1},
        "confidence": 0.8,
        "critique": f"specific critique for {candidate_key}",
        "corrected_answer": None,
        "metadata": {},
    }


def _preference(chosen: str, rejected: str) -> dict[str, object]:
    return {
        "chosen_candidate_key": chosen,
        "rejected_candidate_key": rejected,
        "confidence": 0.75,
        "rationale": f"{chosen} is better than {rejected}",
        "metadata": {},
    }


def _judgments(
    candidate_keys: list[str],
    *,
    preferences: list[dict[str, object]] | None = None,
    teacher: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "molor.teacher-eval.judgments.v1",
        "run_id": "review-tools-test",
        "rubric_version": "molor.teacher-rubric.v1",
        "teacher": teacher or TEACHER,
        "grades": [_grade(key) for key in candidate_keys],
        "preferences": preferences or [],
    }


def test_shards_are_byte_deterministic_and_keep_prompt_groups_whole(
    tools: ModuleType,
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "review-pack.json", _review_pack())
    first = tools.shard_review_pack(source, tmp_path / "first", 2)
    second = tools.shard_review_pack(source, tmp_path / "second", 2)

    first_index = json.loads(Path(first["index"]).read_text(encoding="utf-8"))
    second_index = json.loads(Path(second["index"]).read_text(encoding="utf-8"))
    assert first_index == second_index
    assert first["index_sha256"] == second["index_sha256"]
    assert first_index["group_count"] == 4
    assert first_index["candidate_count"] == 7

    seen_prompts: set[str] = set()
    seen_candidates: set[str] = set()
    for record in first_index["shards"]:
        first_path = tmp_path / "first" / record["file"]
        second_path = tmp_path / "second" / record["file"]
        assert first_path.read_bytes() == second_path.read_bytes()
        shard = json.loads(first_path.read_text(encoding="utf-8"))
        assert shard["schema_version"] == "molor.teacher-eval.review-pack.v1"
        assert shard["run_id"] == "review-tools-test"
        assert shard["rubric"] == _review_pack()["rubric"]
        assert shard["judgments_template"] == _review_pack()["judgments_template"]
        assert shard["shard"]["candidate_count"] == record["candidate_count"]
        for group in shard["groups"]:
            assert group["prompt_key"] not in seen_prompts
            seen_prompts.add(group["prompt_key"])
            seen_candidates.update(item["candidate_key"] for item in group["candidates"])

    assert seen_prompts == {f"prompt-{index}" for index in range(4)}
    assert seen_candidates == {f"candidate-{index}" for index in range(7)}


def test_merge_is_input_order_independent_and_preserves_identities(
    tools: ModuleType,
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "review-pack.json", _review_pack())
    first = _write_json(
        tmp_path / "judgments-1.json",
        _judgments(
            ["candidate-0", "candidate-2", "candidate-4", "candidate-6"],
            preferences=[_preference("candidate-0", "candidate-1")],
        ),
    )
    second = _write_json(
        tmp_path / "judgments-2.json",
        _judgments(["candidate-1", "candidate-3", "candidate-5"]),
    )

    forward = tools.merge_judgments(
        source,
        [first, second],
        tmp_path / "merged-forward.json",
        require_complete=True,
    )
    reverse = tools.merge_judgments(
        source,
        [second, first],
        tmp_path / "merged-reverse.json",
        require_complete=True,
    )

    forward_path = Path(forward["output"])
    reverse_path = Path(reverse["output"])
    assert forward_path.read_bytes() == reverse_path.read_bytes()
    assert forward["output_sha256"] == reverse["output_sha256"]
    assert forward["complete"] is True
    assert forward["grade_count"] == forward["expected_candidate_count"] == 7
    assert forward["preference_count"] == 1
    merged = json.loads(forward_path.read_text(encoding="utf-8"))
    assert merged["schema_version"] == "molor.teacher-eval.judgments.v1"
    assert merged["run_id"] == "review-tools-test"
    assert merged["rubric_version"] == "molor.teacher-rubric.v1"
    assert merged["teacher"] == TEACHER
    assert [grade["candidate_key"] for grade in merged["grades"]] == [
        f"candidate-{index}" for index in range(7)
    ]


@pytest.mark.parametrize("failure", ["unknown", "duplicate", "teacher", "cross_group"])
def test_merge_rejects_unsafe_candidate_or_teacher_state(
    tools: ModuleType,
    tmp_path: Path,
    failure: str,
) -> None:
    source = _write_json(tmp_path / "review-pack.json", _review_pack())
    first_payload = _judgments(["candidate-0"])
    second_payload = _judgments(["candidate-1"])
    if failure == "unknown":
        second_payload = _judgments(["fabricated-candidate"])
    elif failure == "duplicate":
        second_payload = _judgments(["candidate-0"])
    elif failure == "teacher":
        second_payload = _judgments(
            ["candidate-1"],
            teacher={"id": "different", "model": "teacher-model", "provider": "local"},
        )
    else:
        second_payload = _judgments(
            ["candidate-1"],
            preferences=[_preference("candidate-0", "candidate-3")],
        )
    first = _write_json(tmp_path / "judgments-1.json", first_payload)
    second = _write_json(tmp_path / "judgments-2.json", second_payload)

    with pytest.raises(tools.ReviewPackError):
        tools.merge_judgments(source, [first, second], tmp_path / "merged.json")
    assert not (tmp_path / "merged.json").exists()


def test_complete_coverage_is_optional_but_enforced_when_requested(
    tools: ModuleType,
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "review-pack.json", _review_pack())
    partial = _write_json(tmp_path / "partial.json", _judgments(["candidate-0"]))

    summary = tools.merge_judgments(source, [partial], tmp_path / "partial-merged.json")
    assert summary["complete"] is False
    assert summary["missing_grade_count"] == 6
    assert len(summary["missing_candidate_keys_sha256"]) == 64

    with pytest.raises(tools.ReviewPackError, match="missing 6 of 7"):
        tools.merge_judgments(
            source,
            [partial],
            tmp_path / "complete-merged.json",
            require_complete=True,
        )
    assert not (tmp_path / "complete-merged.json").exists()


def test_duplicate_candidate_registry_is_rejected_before_writes(
    tools: ModuleType,
    tmp_path: Path,
) -> None:
    payload = _review_pack()
    payload["groups"][1]["candidates"][0]["candidate_key"] = "candidate-0"
    source = _write_json(tmp_path / "duplicate-review-pack.json", payload)

    with pytest.raises(tools.ReviewPackError, match="duplicate candidate key"):
        tools.shard_review_pack(source, tmp_path / "shards", 2)
    assert not (tmp_path / "shards").exists()
