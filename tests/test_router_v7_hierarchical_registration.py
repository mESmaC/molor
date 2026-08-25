from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EVALUATOR = (
    WORKSPACE
    / "experiments"
    / "first-real-experts"
    / "evaluate_router_v7_hierarchical_registration.py"
)


def _load_evaluator():
    name = "molor_router_v7_hierarchical_registration_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _candidate(evaluator, tmp_path: Path, *, seed: int, loss: float, passed: bool = True):
    checkpoint = tmp_path / f"seed{seed}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    config = checkpoint / "router_config.json"
    weights = checkpoint / "router.safetensors"
    config.write_text("{}\n", encoding="utf-8")
    weights.write_bytes(f"weights:{seed}".encode())
    return evaluator.CandidateEvidence(
        seed=seed,
        checkpoint=checkpoint,
        config=evaluator.snapshot(config),
        weights=evaluator.snapshot(weights),
        validation_loss=loss,
        validation_accuracy=0.995,
        taskmaster_precision=1.0,
        taskmaster_recall=1.0,
        validation_false_positives=0,
        taskmaster_support=130,
        non_taskmaster_support=650,
        gates=(),
        passed=passed,
    )


def _probe_items(evaluator):
    categories = (
        "base/null",
        "embedded-knowledge-v1",
        "mixed",
        "ocb-can-verified-v1",
        "software-architecture-v1",
        evaluator.TASKMASTER_ROUTE_ID,
        "verified-embedded-code-v1",
    )
    items = []
    for category in categories:
        expected = (
            evaluator.TASKMASTER_ROUTE_ID
            if category == evaluator.TASKMASTER_ROUTE_ID
            else ("base/null" if category == "mixed" else category)
        )
        for index in range(12):
            items.append(
                evaluator.ROUTER_PROBE.ProbeItem(
                    probe_id=f"{category}:{index}",
                    prompt=f"synthetic prompt {category} {index}",
                    category=category,
                    expected_route=expected,
                    acceptable_routes=(expected,),
                    purpose=evaluator.PROBE_PURPOSE,
                )
            )
    return items


def _prediction(item, route: str) -> dict[str, Any]:
    return {
        "id": item.probe_id,
        "category": item.category,
        "primary_route": route,
        "exact_match": route == item.expected_route,
        "acceptable_match": route in item.acceptable_routes,
        "confidence": 0.9,
    }


def test_seed_selection_uses_fresh_loss_then_numeric_seed(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    candidates = [
        _candidate(evaluator, tmp_path, seed=27183, loss=0.02),
        _candidate(evaluator, tmp_path, seed=8419, loss=0.01),
        _candidate(evaluator, tmp_path, seed=83, loss=0.01),
    ]

    assert evaluator.select_candidate(candidates).seed == 83


def test_seed_selection_rejects_one_failed_seed(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    candidates = [
        _candidate(evaluator, tmp_path, seed=83, loss=0.01),
        _candidate(evaluator, tmp_path, seed=8419, loss=0.02, passed=False),
        _candidate(evaluator, tmp_path, seed=27183, loss=0.03),
    ]

    with pytest.raises(ValueError, match="all three candidate seeds must pass"):
        evaluator.select_candidate(candidates)


def test_validation_gates_require_995_accuracy_and_zero_of_650_false_positives() -> None:
    evaluator = _load_evaluator()
    passing = {
        "loss": 0.01,
        "accuracy": 0.995,
        "taskmaster_precision": 0.99,
        "taskmaster_recall": 0.99,
        "false_positives": 0,
        "taskmaster_support": 130,
        "non_taskmaster_support": 650,
    }

    gates, passed = evaluator.assess_validation_metrics(passing)

    assert passed is True
    assert all(gate["passed"] for gate in gates)

    one_false_positive = dict(passing, false_positives=1)
    gates, passed = evaluator.assess_validation_metrics(one_false_positive)
    false_positive_gate = next(
        gate for gate in gates if gate["name"] == "false_positives_on_650_validation_negatives"
    )
    assert false_positive_gate == {
        "name": "false_positives_on_650_validation_negatives",
        "actual": 1,
        "required": "0 of 650",
        "passed": False,
    }
    assert passed is False

    low_accuracy = dict(passing, accuracy=0.994999)
    _, passed = evaluator.assess_validation_metrics(low_accuracy)
    assert passed is False


def test_gate_threshold_is_inclusive_and_negative_preserves_v4_decision() -> None:
    evaluator = _load_evaluator()
    items = [
        evaluator.ROUTER_PROBE.ProbeItem(
            probe_id="negative",
            prompt="synthetic negative",
            category="base/null",
            expected_route="base/null",
            acceptable_routes=("base/null",),
            purpose=evaluator.PROBE_PURPOSE,
        ),
        evaluator.ROUTER_PROBE.ProbeItem(
            probe_id="positive",
            prompt="synthetic positive",
            category=evaluator.TASKMASTER_ROUTE_ID,
            expected_route=evaluator.TASKMASTER_ROUTE_ID,
            acceptable_routes=(evaluator.TASKMASTER_ROUTE_ID,),
            purpose=evaluator.PROBE_PURPOSE,
        ),
    ]
    baseline_rows = [
        {
            **_prediction(items[0], "base/null"),
            "prompt": items[0].prompt,
            "selected_routes": ["base/null"],
            "applied_weights": {"base/null": 1.0},
            "probabilities": {"base/null": 0.9},
            "entropy": 0.1,
            "threshold_abstained": False,
            "base_null_selected": True,
            "top_k_acceptable_match": True,
        },
        {
            **_prediction(items[1], "base/null"),
            "prompt": items[1].prompt,
            "selected_routes": ["base/null"],
            "applied_weights": {"base/null": 1.0},
            "probabilities": {"base/null": 0.9},
            "entropy": 0.1,
            "threshold_abstained": False,
            "base_null_selected": True,
            "top_k_acceptable_match": False,
        },
    ]

    rows = evaluator.build_hierarchical_predictions(
        items=items,
        baseline_rows=baseline_rows,
        taskmaster_scores=[0.499999, 0.5],
    )

    for key in (
        "primary_route",
        "selected_routes",
        "applied_weights",
        "probabilities",
        "confidence",
        "entropy",
        "threshold_abstained",
        "base_null_selected",
        "exact_match",
        "acceptable_match",
        "top_k_acceptable_match",
    ):
        assert rows[0][key] == baseline_rows[0][key]
    assert rows[0]["hierarchy"]["action"] == "delegate_exact_v4_decision"
    assert rows[1]["primary_route"] == evaluator.TASKMASTER_ROUTE_ID
    assert rows[1]["hierarchy"]["gate_positive"] is True
    assert rows[1]["hierarchy"]["delegate_consulted_for_candidate_decision"] is False


def test_one_shot_gates_pass_at_preregistered_boundaries() -> None:
    evaluator = _load_evaluator()
    items = _probe_items(evaluator)
    baseline = [
        _prediction(
            item,
            "base/null"
            if item.expected_route == evaluator.TASKMASTER_ROUTE_ID
            else item.expected_route,
        )
        for item in items
    ]
    candidate = [_prediction(item, item.expected_route) for item in items]
    taskmaster_indices = [
        index
        for index, item in enumerate(items)
        if item.expected_route == evaluator.TASKMASTER_ROUTE_ID
    ]
    for index in taskmaster_indices[-2:]:
        candidate[index] = _prediction(items[index], "base/null")
    candidate[0] = _prediction(items[0], evaluator.TASKMASTER_ROUTE_ID)

    result = evaluator.evaluate_one_shot_gates(
        items=items,
        baseline_rows=baseline,
        candidate_rows=candidate,
    )

    assert result["candidate"] == {
        "exact_correct": 81,
        "acceptable_correct": 81,
        "taskmaster_correct": 10,
        "taskmaster_false_positives": 1,
    }
    assert result["all_gates_passed"] is True


def test_preblind_receipt_is_durable_and_verified_before_probe_loader(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator()
    receipt_path = tmp_path / "runs" / "router-v7" / "receipt.json"
    pinned_file = tmp_path / "pinned.bin"
    pinned_file.write_bytes(b"pinned")
    pinned = evaluator.snapshot(pinned_file)
    synthetic_probe = tmp_path / "synthetic-probe.jsonl"
    synthetic_probe.write_text("{}\n", encoding="utf-8")
    order: list[str] = []
    payload = evaluator._with_payload_hash(
        {
            "schema_version": evaluator.PREBLIND_SCHEMA,
            "checks": {"blind_probe_opened": False},
        }
    )

    def probe_loader(_policy, *, repository):
        assert repository == tmp_path.resolve()
        assert receipt_path.is_file()
        reread = json.loads(receipt_path.read_text(encoding="utf-8"))
        evaluator._verify_payload_hash(reread, label="test receipt")
        assert reread["checks"]["blind_probe_opened"] is False
        order.append("probe")
        return (), evaluator.snapshot(synthetic_probe)

    items, _probe_snapshot, receipt_snapshot = evaluator.seal_preblind_then_open_probe(
        receipt_path=receipt_path,
        receipt_payload=payload,
        policy={},
        repository=tmp_path.resolve(),
        snapshots_to_recheck={"pinned": pinned},
        probe_loader=probe_loader,
    )

    assert items == ()
    assert receipt_snapshot.path == receipt_path.resolve()
    assert order == ["probe"]


def test_top_level_evaluator_cannot_substitute_the_frozen_probe_loader() -> None:
    evaluator = _load_evaluator()

    signature = inspect.signature(evaluator.run_registration_evaluation)

    assert "probe_loader" not in signature.parameters
    source = inspect.getsource(evaluator.run_registration_evaluation)
    assert "probe_loader=open_frozen_probe" in source


def test_v7_test_binding_is_rejected_before_any_artifact_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator()
    opens = 0

    def forbidden_open(*_args, **_kwargs):
        nonlocal opens
        opens += 1
        raise AssertionError("a test artifact must never be opened")

    monkeypatch.setattr(Path, "open", forbidden_open)
    bindings = {
        "dataset_report": {},
        "dataset_train": {},
        "dataset_validation": {},
        "dataset_test": {"path": "forbidden-taskmaster.test.jsonl"},
    }

    with pytest.raises(ValueError, match="must not contain a test-dataset binding"):
        evaluator.require_train_validation_only_bindings(bindings)
    assert opens == 0


def test_v7_builder_binding_is_not_mistaken_for_a_dataset_split() -> None:
    evaluator = _load_evaluator()

    evaluator.require_train_validation_only_bindings(
        {
            "dataset_builder": {},
            "dataset_report": {},
            "dataset_train": {},
            "dataset_validation": {},
        }
    )


def test_candidate_sidecar_rejects_test_descriptor_before_resolving_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator()
    descriptor_calls = 0

    def forbidden_descriptor(*_args, **_kwargs):
        nonlocal descriptor_calls
        descriptor_calls += 1
        raise AssertionError("test descriptor must not be resolved or opened")

    monkeypatch.setattr(evaluator, "_require_sidecar_artifact", forbidden_descriptor)
    sidecar = {
        "datasets": {
            "train": {},
            "validation": {},
            "test": {"path": "forbidden-taskmaster.test.jsonl"},
        }
    }

    with pytest.raises(ValueError, match="test forbidden"):
        evaluator._candidate_dataset_descriptors(sidecar, dataset_snapshots={})
    assert descriptor_calls == 0


def test_exclusive_writer_never_replaces_existing_evidence(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    output = tmp_path / "evidence.json"

    evaluator.write_json_exclusive(output, {"decision": "first"})

    with pytest.raises(FileExistsError, match="already exists"):
        evaluator.write_json_exclusive(output, {"decision": "replacement"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"decision": "first"}
