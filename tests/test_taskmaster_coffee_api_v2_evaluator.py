from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "first-real-experts"
EVALUATOR_PATH = EXPERIMENT / "evaluate_taskmaster_coffee_api_v2.py"
POLICY_PATH = EXPERIMENT / "taskmaster-coffee-api-v2-evaluation-policy.json"
METRIC_CONTRACT_PATH = EXPERIMENT / "taskmaster-coffee-api-v2-metric-contract.json"
REAL_V2_TEST = EXPERIMENT / "data" / "taskmaster-coffee-api-v2.test.evaluation.jsonl"


def _load_evaluator(name: str):
    sys.path.insert(0, str(EXPERIMENT))
    try:
        spec = importlib.util.spec_from_file_location(name, EVALUATOR_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXPERIMENT))


@pytest.fixture(autouse=True)
def _forbid_real_v2_test_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make accidental access to the one-shot repository holdout fail every unit test."""

    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        if path.resolve() == REAL_V2_TEST.resolve():
            raise AssertionError("unit tests must never open the real v2 test")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)


def _catalog(evaluator) -> dict[str, Any]:
    return {
        "schema_version": evaluator.TOOL_CATALOG_SCHEMA,
        "dataset_id": evaluator.DATASET_ID,
        "source_split": "train",
        "tools": [
            {
                "argument_key_sets": [["query"]],
                "argument_types": {"query": ["string"]},
                "name": "get_menu_items",
            },
            {
                "argument_key_sets": [["item_id", "quantity"]],
                "argument_types": {
                    "item_id": ["string"],
                    "quantity": ["integer"],
                },
                "name": "update_order",
            },
        ],
    }


def _evaluation_rows(evaluator) -> list[dict[str, Any]]:
    catalog_canonical = evaluator.canonical_json(_catalog(evaluator))
    decisions = [
        (
            [{"role": "user", "text": "Please show me the latte choices."}],
            {"role": "agent", "api": {"get_menu_items": {"query": "latte"}}},
        ),
        (
            [
                {"role": "user", "text": "Add two cappuccinos."},
                {"role": "agent", "text": "I can add those."},
                {"role": "user", "text": "Yes, add them now."},
            ],
            {
                "role": "agent",
                "api": {"update_order": {"item_id": "cappuccino", "quantity": 2}},
            },
        ),
    ]
    rows: list[dict[str, Any]] = []
    for index, (context, expected) in enumerate(decisions):
        context_canonical = evaluator.canonical_json(context)
        expected_canonical = evaluator.canonical_json(expected)
        identifier_payload = {"context": context, "generation": expected}
        rows.append(
            {
                "schema_version": evaluator.EVALUATION_SCHEMA,
                "example_id": (
                    "tm4-coffee-api-v2:"
                    + evaluator.sha256_text(evaluator.canonical_json(identifier_payload))
                ),
                "expected": expected,
                "expected_canonical": expected_canonical,
                "expected_sha256": evaluator.sha256_text(expected_canonical),
                "group_id": (
                    "tm4-coffee-v2-component:" + evaluator.sha256_text(f"synthetic-group-{index}")
                ),
                "messages": evaluator.messages_for(
                    context_canonical,
                    expected_canonical,
                    tool_catalog_canonical=catalog_canonical,
                )[:-1],
            }
        )
    return sorted(rows, key=lambda row: row["example_id"])


def _write_jsonl(path: Path, rows: list[dict[str, Any]], evaluator) -> None:
    path.write_text(
        "".join(evaluator.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _descriptor(path: Path, repository: Path, evaluator) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(repository.resolve()).as_posix(),
        "sha256": evaluator.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _basename_descriptor(path: Path, evaluator) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": evaluator.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _with_payload_sha256(value: dict[str, Any], evaluator) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = evaluator.sha256_text(evaluator.canonical_json(result))
    return result


def _suite(evaluator, tmp_path: Path):
    path = tmp_path / "synthetic-v2.evaluation.jsonl"
    _write_jsonl(path, _evaluation_rows(evaluator), evaluator)
    return evaluator.load_evaluation(path)


def test_loads_only_synthetic_v2_rows_and_validates_dynamic_prompt_ids(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_load_test")
    suite = _suite(evaluator, tmp_path)

    assert len(suite.cases) == 2
    assert suite.tool_catalog == _catalog(evaluator)
    assert suite.tool_catalog_canonical_sha256 == evaluator.sha256_text(
        evaluator.canonical_json(_catalog(evaluator))
    )
    assert all(case.example_id.startswith("tm4-coffee-api-v2:") for case in suite.cases)
    assert all(
        case.messages[0]["content"].startswith(evaluator._SYSTEM_PREFIX) for case in suite.cases
    )


@pytest.mark.parametrize("mutation", ["id", "context", "catalog", "duplicate_key"])
def test_rejects_noncanonical_or_tampered_evaluation_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    evaluator = _load_evaluator(f"taskmaster_v2_evaluator_reject_{mutation}")
    rows = _evaluation_rows(evaluator)
    path = tmp_path / f"tampered-{mutation}.jsonl"
    if mutation == "id":
        rows[0]["example_id"] = "tm4-coffee-api-v2:" + "0" * 64
        _write_jsonl(path, rows, evaluator)
    elif mutation == "context":
        content = rows[0]["messages"][1]["content"]
        prefix, context = content.split("\n", maxsplit=1)
        rows[0]["messages"][1]["content"] = (
            prefix + "\n" + json.dumps(json.loads(context), indent=2)
        )
        _write_jsonl(path, rows, evaluator)
    elif mutation == "catalog":
        catalog = _catalog(evaluator)
        catalog["tools"][0]["name"] = "a_changed_name"
        expected = rows[0]["expected_canonical"]
        user = rows[0]["messages"][1]["content"]
        context_canonical = user[len(evaluator._USER_PREFIX) :]
        rows[0]["messages"] = evaluator.messages_for(
            context_canonical,
            expected,
            tool_catalog_canonical=evaluator.canonical_json(catalog),
        )[:-1]
        _write_jsonl(path, rows, evaluator)
    else:
        encoded = evaluator.canonical_json(rows[0])
        encoded = encoded.replace(
            '"schema_version":"molor.taskmaster-coffee-api-evaluation.v2"',
            (
                '"schema_version":"molor.taskmaster-coffee-api-evaluation.v2",'
                '"schema_version":"molor.taskmaster-coffee-api-evaluation.v2"'
            ),
            1,
        )
        path.write_text(encoded + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        evaluator.load_evaluation(path)


def test_predictions_are_complete_ordered_and_content_addressed(tmp_path: Path) -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_predictions_test")
    suite = _suite(evaluator, tmp_path)
    outputs = {
        route: [f'{case.expected_canonical[:-1]},"route_note":"{route}"}}' for case in suite.cases]
        for route in evaluator.ROUTE_ORDER
    }
    path = tmp_path / "predictions.jsonl"

    evaluator.write_predictions(path, suite, outputs)
    loaded = evaluator.load_predictions(path, suite)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert loaded == outputs
    assert [row["route"] for row in rows] == [
        route for route in evaluator.ROUTE_ORDER for _case in suite.cases
    ]
    assert all(
        row["prediction_id"]
        == evaluator.prediction_id(row["example_id"], row["route"], row["output"])
        for row in rows
    )

    rows[0]["prediction_id"] = "tm4-coffee-api-v2-prediction:" + "0" * 64
    _write_jsonl(path, rows, evaluator)
    with pytest.raises(ValueError, match="prediction_id"):
        evaluator.load_predictions(path, suite)


def test_scoring_separates_metrics_and_preserves_json_types(tmp_path: Path) -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_metrics_test")
    suite = _suite(evaluator, tmp_path)
    case = next(
        case for case in suite.cases if next(iter(case.expected["api"])) == "get_menu_items"
    )
    expected_arguments = next(iter(case.expected["api"].values()))
    wrong_name = evaluator.canonical_json(
        {"role": "agent", "api": {"update_order": expected_arguments}}
    )

    score = evaluator.score_one(case, "forced_candidate", wrong_name)
    duplicate = evaluator.score_one(
        case,
        "forced_candidate",
        '{"role":"agent","api":{},"api":{}}',
    )

    assert score["json_parseable"] is True
    assert score["schema_valid"] is True
    assert score["api_name_exact"] is False
    assert score["arguments_exact"] is True
    assert score["semantic_exact"] is False
    assert duplicate["json_parseable"] is False

    integer_expected = case.expected.copy()
    integer_expected = {"role": "agent", "api": {"get_menu_items": {"query": 1}}}
    integer_case = evaluator.EvaluationCase(
        example_id=case.example_id,
        group_id=case.group_id,
        messages=case.messages,
        context=case.context,
        expected=integer_expected,
        expected_canonical=evaluator.canonical_json(integer_expected),
        expected_sha256=evaluator.sha256_text(evaluator.canonical_json(integer_expected)),
        tool_catalog_sha256=case.tool_catalog_sha256,
    )
    boolean_output = '{"api":{"get_menu_items":{"query":true}},"role":"agent"}'
    typed = evaluator.score_one(integer_case, "forced_candidate", boolean_output)
    assert typed["arguments_exact"] is False
    assert typed["semantic_exact"] is False


def test_three_route_comparison_reports_base_improvement_and_v1_gates(tmp_path: Path) -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_comparison_test")
    suite = _suite(evaluator, tmp_path)
    exact = [case.expected_canonical for case in suite.cases]
    invalid = ["not json" for _case in suite.cases]
    v1 = [exact[0], "not json"]
    policy = evaluator.load_policy(POLICY_PATH)
    policy = dict(policy)
    policy["minimum_rates"] = {metric: 1.0 for metric in evaluator.SCORED_METRICS}
    policy["minimum_absolute_improvement_over_base"] = {
        metric: 1.0 for metric in evaluator.PRINCIPAL_METRICS
    }

    result = evaluator.evaluate_outputs(
        suite,
        {
            "forced_candidate": exact,
            "frozen_base_null": invalid,
            "quarantined_v1": v1,
        },
        policy,
    )

    assert result["routes"]["forced_candidate"]["summary"]["full_object_exact_rate"] == 1.0
    assert result["candidate_minus_frozen_base_absolute_rates"] == {
        metric: 1.0 for metric in evaluator.SCORED_METRICS
    }
    assert result["candidate_minus_quarantined_v1_absolute_rates"]["semantic_exact"] == 0.5
    assert all(gate["passed"] for gate in result["gates"]["fresh_v1_non_regression"].values())
    assert result["gates"]["all_expert_policy_gates_passed"] is True
    assert "router_registration_not_authorized" in result["decision"]

    failing = evaluator.evaluate_outputs(
        suite,
        {
            "forced_candidate": v1,
            "frozen_base_null": invalid,
            "quarantined_v1": exact,
        },
        policy,
    )
    assert failing["gates"]["fresh_v1_non_regression"]["semantic_exact"]["passed"] is False
    assert failing["decision"] == "keep_candidate_quarantined"


def test_policy_metric_contract_and_one_shot_cli_are_frozen() -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_contract_test")

    policy = evaluator.load_policy(POLICY_PATH)
    metrics = evaluator.load_metric_contract(METRIC_CONTRACT_PATH)

    assert policy["evaluation"]["generation"] == metrics["generation"]
    assert metrics["routes"] == list(evaluator.ROUTE_ORDER)
    assert "one-shot" in metrics["reuse_warning"]
    with pytest.raises(SystemExit):
        evaluator.parse_args([])
    args = evaluator.parse_args(["--acknowledge-one-shot"])
    assert args.candidate_adapter == evaluator.DEFAULT_CANDIDATE
    assert "quarantine" in str(args.candidate_adapter).lower()
    assert args.v1_adapter == evaluator.DEFAULT_V1_COMPARATOR


def test_exact_quarantine_adapter_paths_are_valid_without_catalog_discovery(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_direct_quarantine_paths_test")
    model = (tmp_path / "models" / "Qwen3-1.7B").resolve()
    model.mkdir(parents=True)
    quarantine = tmp_path / "checkpoints" / "qwen3-1.7b-candidate-experts-quarantine"
    candidate = (quarantine / "taskmaster-coffee-api-v2-full-dialog-r16").resolve()
    v1 = (quarantine / "taskmaster-coffee-api-v1").resolve()
    for directory, name in (
        (candidate, candidate.name),
        (v1, "taskmaster-coffee-api-v1"),
    ):
        _write_json(
            directory / "adapter_config.json",
            {
                "name": name,
                "base_model_id": str(model),
                "target_modules": ["q_proj", "v_proj"],
            },
        )
    stack = {
        "artifacts": [
            {"role": "base", "content_fingerprint": "a" * 64},
            {"role": "expert", "files": []},
            {"role": "expert", "files": []},
        ]
    }
    policy = {
        "candidate": candidate.name,
        "training_artifact_contract": {"base_artifact": {"content_fingerprint": "a" * 64}},
    }

    evaluator._validate_adapter_bindings(
        model=model,
        candidate=candidate,
        v1=v1,
        policy=policy,
        stack=stack,
    )

    outside_quarantine = (tmp_path / "checkpoints" / "taskmaster-coffee-api-v1").resolve()
    _write_json(
        outside_quarantine / "adapter_config.json",
        {
            "name": "taskmaster-coffee-api-v1",
            "base_model_id": str(model),
            "target_modules": ["q_proj", "v_proj"],
        },
    )
    with pytest.raises(ValueError, match="quarantine"):
        evaluator._validate_adapter_bindings(
            model=model,
            candidate=candidate,
            v1=outside_quarantine,
            policy=policy,
            stack=stack,
        )


def _training_completion_fixture(evaluator, tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "repo"
    candidate = (
        repository
        / "checkpoints"
        / "qwen3-1.7b-candidate-experts-quarantine"
        / "taskmaster-coffee-api-v2-full-dialog-r16"
    )
    candidate.mkdir(parents=True)
    policy_path = repository / "experiments" / "policy.json"
    build_report_path = repository / "experiments" / "data" / "build-report.json"
    _write_json(policy_path, {"candidate": candidate.name})
    _write_json(build_report_path, {"dataset_id": evaluator.DATASET_ID})
    adapter_config = candidate / "adapter_config.json"
    adapter_weights = candidate / "adapter_model.safetensors"
    training_summary = candidate / "training_summary.json"
    _write_json(adapter_config, {"name": candidate.name})
    adapter_weights.write_bytes(b"synthetic adapter weights")
    train_dataset_sha256 = "1" * 64
    _write_json(
        training_summary,
        {
            "expert": candidate.name,
            "dataset_sha256": train_dataset_sha256,
            "examples": 12,
            "epochs": 1,
            "seed": 241104,
            "final_training_loss": 0.125,
        },
    )
    base_lock = {
        "role": "base",
        "content_fingerprint": "a" * 64,
        "size_bytes": 7,
        "files": [{"name": "config.json", "sha256": "b" * 64, "size_bytes": 7}],
        "complete": True,
    }
    candidate_lock = {
        "role": "expert",
        "content_fingerprint": "c" * 64,
        "size_bytes": adapter_config.stat().st_size + adapter_weights.stat().st_size,
        "files": [
            {
                "name": path.name,
                "sha256": evaluator.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (adapter_config, adapter_weights)
        ],
        "complete": True,
    }
    stack = {
        "schema_version": "molor.benchmark-stack-lock.v2",
        "artifacts": [base_lock, candidate_lock],
        "fully_content_addressed": True,
        "stack_fingerprint": "d" * 64,
    }
    trainer_contract = {
        "hash_policy": "sha256-full-read-canonical-relative-path-tree-v1",
        "content_fingerprint": "e" * 64,
        "files": [
            {
                "name": "src/molor/training/sft.py",
                "sha256": "f" * 64,
                "size_bytes": 10,
            }
        ],
    }
    identities = {
        "policy_sha256": evaluator.sha256_file(policy_path),
        "build_report_sha256": evaluator.sha256_file(build_report_path),
        "trainer_source_tree_sha256": trainer_contract["content_fingerprint"],
        "train_dataset_sha256": train_dataset_sha256,
    }
    preflight = _with_payload_sha256(
        {
            "schema_version": evaluator.TRAINING_PREFLIGHT_SCHEMA,
            "dataset_id": evaluator.DATASET_ID,
            "candidate": candidate.name,
            "posture": "sealed_for_single_quarantined_training_run_only",
            "input_descriptors": {
                "policy": _basename_descriptor(policy_path, evaluator),
                "build_report": _basename_descriptor(build_report_path, evaluator),
                "trainer_source_tree": trainer_contract,
            },
            "identities": identities,
            "base_artifact": base_lock,
            "checks": {
                "v2_test_evaluation_opened": False,
                "quarantine_preserved": True,
            },
            "run_contract": {
                "expert": candidate.name,
                "examples": 12,
                "epochs": 1,
                "seed": 241104,
            },
        },
        evaluator,
    )
    preflight_path = repository / "experiments" / "training-preflight.json"
    _write_json(preflight_path, preflight)
    completion = _with_payload_sha256(
        {
            "schema_version": evaluator.TRAINING_COMPLETION_SCHEMA,
            "status": "completed_quarantined_candidate_not_promotion_eligible",
            "candidate": candidate.name,
            "preflight_receipt": _descriptor(preflight_path, repository, evaluator),
            "preflight_payload_sha256": preflight["payload_sha256"],
            "artifacts": [
                _descriptor(path, repository, evaluator)
                for path in (adapter_config, adapter_weights, training_summary)
            ],
            "final_training_loss": 0.125,
            "test_evaluation_opened": False,
            "registration_changed": False,
        },
        evaluator,
    )
    completion_path = candidate / "training_completion_receipt.json"
    _write_json(completion_path, completion)
    policy = {
        "candidate": candidate.name,
        "training_artifact_contract": {
            "base_artifact": {"content_fingerprint": base_lock["content_fingerprint"]},
            "required_file_identities": list(identities),
            "trainer_source_contract": trainer_contract,
        },
    }
    return {
        "repository": repository,
        "candidate": candidate,
        "policy_path": policy_path,
        "build_report_path": build_report_path,
        "completion_path": completion_path,
        "policy": policy,
        "stack": stack,
    }


def test_training_completion_is_rehashed_against_candidate_and_preflight(tmp_path: Path) -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_training_completion_test")
    fixture = _training_completion_fixture(evaluator, tmp_path)

    binding = evaluator.validate_training_completion_receipt(
        fixture["completion_path"],
        repository=fixture["repository"],
        candidate=fixture["candidate"],
        policy_path=fixture["policy_path"],
        build_report_path=fixture["build_report_path"],
        policy=fixture["policy"],
        stack=fixture["stack"],
    )

    assert binding["test_evaluation_opened"] is False
    assert binding["registration_changed"] is False
    assert binding["candidate_artifacts"]["adapter_model.safetensors"]["sha256"] == (
        evaluator.sha256_file(fixture["candidate"] / "adapter_model.safetensors")
    )
    assert binding["completion_receipt"]["payload_sha256"]
    assert binding["preflight_receipt"]["payload_sha256"]


@pytest.mark.parametrize("mutation", ["adapter", "test_opened", "policy", "summary_binding"])
def test_training_completion_rejects_tampered_or_stale_bindings(
    tmp_path: Path,
    mutation: str,
) -> None:
    evaluator = _load_evaluator(f"taskmaster_v2_evaluator_training_reject_{mutation}")
    fixture = _training_completion_fixture(evaluator, tmp_path)
    if mutation == "adapter":
        (fixture["candidate"] / "adapter_model.safetensors").write_bytes(b"tampered")
    elif mutation == "test_opened":
        completion = json.loads(fixture["completion_path"].read_text(encoding="utf-8"))
        completion["test_evaluation_opened"] = True
        completion.pop("payload_sha256")
        _write_json(fixture["completion_path"], _with_payload_sha256(completion, evaluator))
    elif mutation == "policy":
        _write_json(fixture["policy_path"], {"candidate": "changed-after-training"})
    else:
        summary_path = fixture["candidate"] / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["dataset_sha256"] = "9" * 64
        _write_json(summary_path, summary)
        completion = json.loads(fixture["completion_path"].read_text(encoding="utf-8"))
        completion["artifacts"][2] = _descriptor(summary_path, fixture["repository"], evaluator)
        completion.pop("payload_sha256")
        _write_json(fixture["completion_path"], _with_payload_sha256(completion, evaluator))

    with pytest.raises((FileNotFoundError, ValueError)):
        evaluator.validate_training_completion_receipt(
            fixture["completion_path"],
            repository=fixture["repository"],
            candidate=fixture["candidate"],
            policy_path=fixture["policy_path"],
            build_report_path=fixture["build_report_path"],
            policy=fixture["policy"],
            stack=fixture["stack"],
        )


def test_preaccess_receipt_binds_expected_data_without_opening_evaluation(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator("taskmaster_v2_evaluator_receipt_test")
    nonexistent_holdout = tmp_path / "unopened.evaluation.jsonl"
    build_report = {
        "outputs": {
            "test_evaluation": {
                "path": "taskmaster-coffee-api-v2.test.evaluation.jsonl",
                "rows": 2,
                "sha256": "a" * 64,
                "size_bytes": 123,
            }
        }
    }
    stack = {
        "schema_version": "synthetic-stack",
        "fully_content_addressed": True,
        "stack_fingerprint": "b" * 64,
    }

    receipt = evaluator.build_preaccess_receipt(
        evaluator_path=EVALUATOR_PATH,
        metric_contract_path=METRIC_CONTRACT_PATH,
        policy_path=POLICY_PATH,
        build_report_path=POLICY_PATH,
        build_report=build_report,
        stack=stack,
        training_completion_binding={
            "completion_receipt": {"sha256": "c" * 64},
            "preflight_receipt": {"sha256": "d" * 64},
        },
        candidate=tmp_path / "checkpoints" / "candidate-experts-quarantine" / "v2",
        v1=tmp_path / "checkpoints" / "candidate-experts-quarantine" / "v1",
        evaluation_path=nonexistent_holdout,
        dtype_name="bfloat16",
        device="cuda",
        batch_size=4,
    )

    assert not nonexistent_holdout.exists()
    assert receipt["test_access_state"] == "sealed_not_opened_by_this_evaluator_yet"
    assert receipt["evaluation"]["expected_sha256_from_unopened_build_report"] == "a" * 64
    assert receipt["artifact_stack"] is stack
    assert receipt["training_completion"]["completion_receipt"]["sha256"] == "c" * 64
    assert evaluator.ONE_SHOT_WARNING in receipt["one_shot_warning"]
