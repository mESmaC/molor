from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EVALUATOR = WORKSPACE / "experiments" / "first-real-experts" / "evaluate_router_v6_registration.py"


def _load_evaluator():
    name = "molor_router_v6_registration_test"
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
    config_path = checkpoint / "router_config.json"
    weights_path = checkpoint / "router.safetensors"
    config_path.write_text("{}\n", encoding="utf-8")
    weights_path.write_bytes(f"weights-{seed}".encode())
    return evaluator.CandidateEvidence(
        seed=seed,
        checkpoint=checkpoint,
        config=evaluator.snapshot(config_path),
        weights=evaluator.snapshot(weights_path),
        validation_loss=loss,
        validation_accuracy=0.995,
        taskmaster_precision=1.0,
        taskmaster_recall=1.0,
        legacy_recalls={"base/null": 1.0},
        passed=passed,
        gates=(),
    )


def _probe_items(evaluator):
    categories = (
        "base/null",
        "embedded-knowledge-v1",
        "ocb-can-verified-v1",
        "software-architecture-v1",
        "verified-embedded-code-v1",
        evaluator.TASKMASTER_ROUTE_ID,
        "mixed",
    )
    category_routes = {
        "base/null": "base/null",
        "embedded-knowledge-v1": "embedded-knowledge-v1",
        "ocb-can-verified-v1": "ocb-can-verified-v1",
        "software-architecture-v1": "software-architecture-v1",
        "verified-embedded-code-v1": "verified-embedded-code-v1",
        evaluator.TASKMASTER_ROUTE_ID: evaluator.TASKMASTER_ROUTE_ID,
        "mixed": "base/null",
    }
    items = []
    for category in categories:
        for index in range(12):
            route = category_routes[category]
            items.append(
                evaluator.ROUTER_PROBE.ProbeItem(
                    probe_id=f"{category}-{index:02d}",
                    prompt=f"synthetic {category} {index}",
                    category=category,
                    expected_route=route,
                    acceptable_routes=(route,),
                    purpose=evaluator.PROBE_PURPOSE,
                )
            )
    return items


def _prediction(item, primary_route: str) -> dict[str, Any]:
    exact = primary_route == item.expected_route
    acceptable = primary_route in item.acceptable_routes
    return {
        "id": item.probe_id,
        "primary_route": primary_route,
        "exact_match": exact,
        "acceptable_match": acceptable,
        "confidence": 0.9,
    }


def _artifact_descriptor(evaluator, path: Path, repository: Path) -> dict[str, Any]:
    value = evaluator.snapshot(path)
    return value.to_dict(repository=repository)


def _fake_model_evidence(evaluator, tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir(parents=True, exist_ok=True)
    artifact = model / "artifact.bin"
    artifact.write_bytes(b"frozen")
    value = evaluator.snapshot(artifact)
    return evaluator.ModelRuntimeEvidence(
        directory=model.resolve(),
        config=value,
        index=value,
        embedding_artifact=value,
        embedding_tensor_name="model.embed_tokens.weight",
        embedding_shape=(151936, 2048),
        embedding_dtype="bfloat16",
        tokenizer_artifacts={},
    )


def test_seed_selection_uses_loss_then_numeric_seed(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    candidates = [
        _candidate(evaluator, tmp_path, seed=27183, loss=0.02),
        _candidate(evaluator, tmp_path, seed=8419, loss=0.01),
        _candidate(evaluator, tmp_path, seed=83, loss=0.01),
    ]

    selected = evaluator.select_candidate(candidates)

    assert selected.seed == 83


def test_seed_selection_rejects_any_failed_candidate(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    candidates = [
        _candidate(evaluator, tmp_path, seed=83, loss=0.01),
        _candidate(evaluator, tmp_path, seed=8419, loss=0.02, passed=False),
        _candidate(evaluator, tmp_path, seed=27183, loss=0.03),
    ]

    with pytest.raises(ValueError, match="all three candidate seeds must pass"):
        evaluator.select_candidate(candidates)


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


def test_one_shot_gates_reject_two_false_taskmaster_routes() -> None:
    evaluator = _load_evaluator()
    items = _probe_items(evaluator)
    baseline = [_prediction(item, item.expected_route) for item in items]
    for index, item in enumerate(items):
        if item.expected_route == evaluator.TASKMASTER_ROUTE_ID:
            baseline[index] = _prediction(item, "base/null")
    candidate = [_prediction(item, item.expected_route) for item in items]
    candidate[0] = _prediction(items[0], evaluator.TASKMASTER_ROUTE_ID)
    candidate[1] = _prediction(items[1], evaluator.TASKMASTER_ROUTE_ID)

    result = evaluator.evaluate_one_shot_gates(
        items=items,
        baseline_rows=baseline,
        candidate_rows=candidate,
    )

    false_positive_gate = next(
        gate for gate in result["gates"] if gate["name"] == "taskmaster_false_positives"
    )
    assert false_positive_gate["actual"] == 2
    assert false_positive_gate["passed"] is False
    assert result["all_gates_passed"] is False


def test_exclusive_writer_never_replaces_evidence(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    output = tmp_path / "evidence.json"

    written = evaluator.write_json_exclusive(output, {"decision": "first"})

    assert written.sha256 == evaluator.sha256_file(output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"decision": "first"}
    with pytest.raises(FileExistsError, match="already exists"):
        evaluator.write_json_exclusive(output, {"decision": "replacement"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"decision": "first"}


def test_preblind_metric_failure_never_calls_probe_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load_evaluator()
    policy_file = tmp_path / "policy.json"
    policy_file.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "runs" / "evaluation" / "router-v6-taskmaster-registration-one-shot.json"
    receipt = tmp_path / "runs" / "router-v6" / "taskmaster-preblind-receipt.json"
    policy = {
        "one_shot_evaluation": {
            "report_path": "runs/evaluation/router-v6-taskmaster-registration-one-shot.json"
        },
        "preblind_receipt": {"path": "runs/router-v6/taskmaster-preblind-receipt.json"},
        "training": {"feature_batch_size": 64},
    }
    policy_snapshot = evaluator.snapshot(policy_file)
    candidate = _candidate(evaluator, tmp_path, seed=83, loss=0.1, passed=False)
    failure = evaluator.PreblindMetricGateFailure(
        "frozen validation gate failed",
        candidates=[candidate],
        snapshots={"candidate": candidate.config},
    )
    probe_calls = 0

    def forbidden_probe_loader(*_args, **_kwargs):
        nonlocal probe_calls
        probe_calls += 1
        raise AssertionError("probe must remain sealed")

    monkeypatch.setattr(
        evaluator,
        "load_policy",
        lambda *_args, **_kwargs: (policy, policy_snapshot),
    )
    monkeypatch.setattr(
        evaluator,
        "validate_dataset_contract",
        lambda *_args, **_kwargs: ({"validation": candidate.config}, {}),
    )
    model_evidence = _fake_model_evidence(evaluator, tmp_path)
    monkeypatch.setattr(
        evaluator,
        "validate_runtime_sources",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluator,
        "validate_model_runtime",
        lambda *_args, **_kwargs: model_evidence,
    )
    monkeypatch.setattr(
        evaluator,
        "load_input_embedding",
        lambda *_args, **_kwargs: (object(), {}),
    )
    monkeypatch.setattr(evaluator, "validate_loaded_embedding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluator, "load_router_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        evaluator,
        "prepare_validation_features",
        lambda *_args, **_kwargs: (object(), object(), object()),
    )

    def fail_candidates(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(evaluator, "validate_candidates", fail_candidates)

    report = evaluator.run_registration_evaluation(
        policy_path=policy_file,
        model_path=tmp_path / "model",
        output_path=output,
        preblind_receipt_path=receipt,
        repository=tmp_path,
        probe_loader=forbidden_probe_loader,
    )

    assert probe_calls == 0
    assert report["probe"]["opened"] is False
    assert report["registration_changed"] is False
    assert output.is_file()
    assert not receipt.exists()


def test_router_split_loader_rejects_normalized_duplicates(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    split = tmp_path / "train.jsonl"
    split.write_text(
        json.dumps({"prompt": "Coffee  order", "route_id": "base/null"})
        + "\n"
        + json.dumps({"prompt": " coffee order ", "route_id": "base/null"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicated within the split"):
        evaluator._read_router_split(split, route_ids=("base/null",))


def test_frozen_policy_detects_later_source_drift_without_opening_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator()
    probe_calls = 0

    def forbidden_probe_loader(*_args, **_kwargs):
        nonlocal probe_calls
        probe_calls += 1
        raise AssertionError("preblind validation must not parse the probe")

    monkeypatch.setattr(evaluator.ROUTER_PROBE, "load_probe", forbidden_probe_loader)

    raw_policy = json.loads(evaluator.DEFAULT_POLICY.read_text(encoding="utf-8"))
    if "runtime_source_artifacts" not in raw_policy:
        pytest.skip("final runtime/model policy bindings have not been resealed yet")

    policy, policy_snapshot = evaluator.load_policy(evaluator.DEFAULT_POLICY)
    with pytest.raises(ValueError, match="source-tree file count differs from frozen policy"):
        evaluator.validate_runtime_sources(policy)
    model_evidence = evaluator.validate_model_runtime(
        policy,
        model_path=evaluator.DEFAULT_MODEL,
    )
    snapshots, report = evaluator.validate_dataset_contract(policy)

    assert policy_snapshot.sha256 == (
        "31e216ddaeafce844caea51a15b662db6e6f69702adaa15428a8e0a6e3bdf869"
    )
    assert len(model_evidence.snapshots) == 9
    assert snapshots["report"].sha256 == (
        "4a47892f7aef275483a9915a56a553f510ebccef9d578336d6f42cafa40c4673"
    )
    assert report["route_ids"] == policy["route_ids"]
    assert probe_calls == 0


def test_fabricated_sidecar_metrics_cannot_override_fresh_failure() -> None:
    evaluator = _load_evaluator()
    route_ids = (
        "embedded-knowledge-v1",
        "ocb-can-verified-v1",
        "software-architecture-v1",
        "verified-embedded-code-v1",
        evaluator.TASKMASTER_ROUTE_ID,
        "base/null",
    )
    recomputed = {
        "loss": 2.0,
        "accuracy": 0.50,
        "mean_confidence": 0.60,
        "examples": 780,
        "confusion": [[0] * 6 for _ in range(6)],
        "per_route": {
            route_id: {"support": 130, "precision": 0.5, "recall": 0.5} for route_id in route_ids
        },
    }
    fabricated = {
        **recomputed,
        "loss": 0.001,
        "accuracy": 1.0,
        "mean_confidence": 1.0,
        "per_route": {
            route_id: {"support": 130, "precision": 1.0, "recall": 1.0} for route_id in route_ids
        },
    }
    preblind = {
        "validation_accuracy_min_for_every_seed": 0.99,
        "taskmaster_validation_precision_min_for_every_seed": 0.99,
        "taskmaster_validation_recall_min_for_every_seed": 0.99,
        "legacy_route_validation_recall_min_for_every_seed": 0.98,
    }

    diagnostics = evaluator.sidecar_metric_diagnostics(
        fabricated,
        recomputed,
        route_ids=route_ids,
    )
    *_, gates = evaluator.assess_recomputed_validation_metrics(
        recomputed,
        preblind=preblind,
        route_ids=route_ids,
    )

    assert diagnostics["matches_recomputed"] is False
    assert diagnostics["role"] == "diagnostic_only_never_used_for_gates_or_selection"
    assert not all(gate["passed"] for gate in gates)


def test_runtime_source_tree_drift_blocks_validation(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    for relative in evaluator.RUNTIME_SCRIPT_PATHS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    source = tmp_path / "src" / "molor" / "router.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    entries = [
        {
            "path": "src/molor/router.py",
            "size_bytes": source.stat().st_size,
            "sha256": evaluator.sha256_file(source),
        }
    ]
    policy = {
        "runtime_source_artifacts": {
            "scripts": {
                name: _artifact_descriptor(evaluator, tmp_path / relative, tmp_path)
                for name, relative in evaluator.RUNTIME_SCRIPT_PATHS.items()
            },
            "molor_source_tree": {
                "root": "src/molor",
                "glob": "**/*.py",
                "hash_policy": evaluator.SOURCE_TREE_HASH_POLICY,
                "file_count": 1,
                "content_fingerprint": evaluator.sha256_text(evaluator.canonical_json(entries)),
            },
        }
    }

    snapshots = evaluator.validate_runtime_sources(policy, repository=tmp_path)
    assert snapshots["molor:src/molor/router.py"].sha256 == entries[0]["sha256"]

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content fingerprint"):
        evaluator.validate_runtime_sources(policy, repository=tmp_path)


def test_source_drift_stops_run_before_probe_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load_evaluator()
    policy_file = tmp_path / "policy.json"
    policy_file.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "runs" / "evaluation" / "router-v6-taskmaster-registration-one-shot.json"
    receipt = tmp_path / "runs" / "router-v6" / "taskmaster-preblind-receipt.json"
    policy = {
        "one_shot_evaluation": {
            "report_path": "runs/evaluation/router-v6-taskmaster-registration-one-shot.json"
        },
        "preblind_receipt": {"path": "runs/router-v6/taskmaster-preblind-receipt.json"},
        "training": {"feature_batch_size": 64},
    }
    policy_snapshot = evaluator.snapshot(policy_file)
    probe_calls = 0

    def probe_loader(*_args, **_kwargs):
        nonlocal probe_calls
        probe_calls += 1
        raise AssertionError("probe must remain sealed")

    monkeypatch.setattr(
        evaluator,
        "load_policy",
        lambda *_args, **_kwargs: (policy, policy_snapshot),
    )

    def source_drift(*_args, **_kwargs):
        raise ValueError("molor source-tree content fingerprint differs from frozen policy")

    monkeypatch.setattr(evaluator, "validate_runtime_sources", source_drift)

    with pytest.raises(ValueError, match="source-tree content fingerprint"):
        evaluator.run_registration_evaluation(
            policy_path=policy_file,
            model_path=tmp_path / "models" / "Qwen3-1.7B",
            output_path=output,
            preblind_receipt_path=receipt,
            repository=tmp_path,
            probe_loader=probe_loader,
        )

    assert probe_calls == 0
    assert not receipt.exists()
    assert not output.exists()


def test_model_runtime_drift_blocks_before_loading(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    model = tmp_path / "models" / "Qwen3-1.7B"
    model.mkdir(parents=True)
    names = (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        *evaluator.MODEL_TOKENIZER_FILES,
    )
    for name in names:
        (model / name).write_bytes(f"frozen:{name}".encode())
    policy = {
        "model_runtime": {
            "path": "models/Qwen3-1.7B",
            "config": _artifact_descriptor(evaluator, model / "config.json", tmp_path),
            "safetensors_index": _artifact_descriptor(
                evaluator, model / "model.safetensors.index.json", tmp_path
            ),
            "embedding": {
                "tensor_name": "model.embed_tokens.weight",
                "shape": [151936, 2048],
                "dtype": "bfloat16",
                "artifact": _artifact_descriptor(
                    evaluator,
                    model / "model-00001-of-00002.safetensors",
                    tmp_path,
                ),
            },
            "tokenizer_artifacts": {
                name: _artifact_descriptor(evaluator, model / name, tmp_path)
                for name in evaluator.MODEL_TOKENIZER_FILES
            },
        }
    }

    evidence = evaluator.validate_model_runtime(
        policy,
        model_path=model,
        repository=tmp_path,
    )
    assert evidence.directory == model.resolve()

    (model / "tokenizer.json").write_bytes(b"drift")
    with pytest.raises(ValueError, match="sha256"):
        evaluator.validate_model_runtime(policy, model_path=model, repository=tmp_path)
