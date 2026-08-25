from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


EVALUATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "exact-context-v1"
    / "evaluate_acceptance.py"
)
SPEC = importlib.util.spec_from_file_location("exact_context_acceptance", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acceptance
SPEC.loader.exec_module(acceptance)


def _memory(*, rss: int, free: int, peak_cuda: int, label: str) -> dict[str, object]:
    return {
        "label": label,
        "process": {
            "current_rss_bytes": rss,
            "peak_rss_bytes": rss,
            "source": "synthetic",
        },
        "cuda": {
            "device_global_free_bytes": free,
            "device_global_total_bytes": 12 * acceptance.GIB,
            "process_allocator_allocated_bytes": peak_cuda,
            "process_allocator_reserved_bytes": peak_cuda,
            "process_allocator_peak_allocated_bytes": peak_cuda,
            "process_allocator_peak_reserved_bytes": peak_cuda,
        },
    }


def _synthetic_contract(tmp_path: Path) -> dict[str, object]:
    contract = acceptance.load_json_strict(EVALUATOR_PATH.with_name("contract.json"))
    contract["artifacts"]["model"]["path"] = "model"
    contract["identities"]["tokenizer"]["name_or_path"] = str(tmp_path / "model")
    contract["phases"]["parity"]["report_path"] = "parity.json"
    contract["phases"]["capacity"]["report_path"] = "capacity.json"
    contract["summary"]["path"] = "summary.json"
    return contract


def _model_identities(contract: dict[str, object], tmp_path: Path) -> dict[str, object]:
    frozen = contract["identities"]
    model = contract["artifacts"]["model"]
    model_path = str(tmp_path / "model")
    return {
        "config": deepcopy(frozen["config"]),
        "tokenizer": deepcopy(frozen["tokenizer"]),
        "model_artifact": {
            "schema_version": "molor.benchmark-stack-lock.v2",
            "hash_policy": "sha256-full-read-every-lock",
            "backend_identity": {
                "backend": "huggingface-local",
                "model_id": model_path,
                "model_artifact_path": model_path,
            },
            "stable_backend_identity": {"backend": "huggingface-local"},
            "required_roles": [{"role": "base", "identifier": model_path}],
            "artifacts": [
                {
                    "role": "base",
                    "path": model_path,
                    "content_fingerprint": model["content_fingerprint"],
                    "size_bytes": model["size_bytes"],
                    "files": deepcopy(model["files"]),
                    "complete": True,
                }
            ],
            "unresolved": [],
            "fully_content_addressed": True,
            "stack_fingerprint": model["runner_stack_fingerprint"],
        },
        "loaded_model": {
            "class": frozen["loaded_model"]["class"],
            "name_or_path": model_path,
            "parameter_count": frozen["loaded_model"]["parameter_count"],
            "parameter_devices": ["cuda:0"],
            "parameter_dtypes": ["torch.bfloat16"],
            "loaded_config": deepcopy(frozen["config"]),
            "attention_implementation": "sdpa",
        },
        "model_artifact_recheck": {
            "seconds": 1.0,
            "stack_fingerprint": model["runner_stack_fingerprint"],
            "matches_preload_identity": True,
        },
    }


def _synthetic_run(
    contract: dict[str, object],
    *,
    length: int,
    repeat: int,
    mode: str,
) -> dict[str, object]:
    token_ids = [11, 12, 13, 14, 15, 16, 17, 18]
    peak_cuda = 3 * acceptance.GIB if mode == "offloaded" else 5 * acceptance.GIB
    observations = [
        _memory(
            rss=2 * acceptance.GIB,
            free=3 * acceptance.GIB,
            peak_cuda=peak_cuda,
            label="before_prefill",
        ),
        _memory(
            rss=2 * acceptance.GIB,
            free=3 * acceptance.GIB,
            peak_cuda=peak_cuda,
            label="run_complete",
        ),
    ]
    steps = []
    for index, token_id in enumerate(token_ids):
        steps.append(
            {
                "completion_index": index,
                "phase": "prefill_to_first_token" if index == 0 else "single_token_decode",
                "model_input_tokens": length if index == 0 else 1,
                "cache_tokens_after_forward": length if index == 0 else length + index,
                "output_token_id": token_id,
                "seconds": 10.0 if index == 0 else 0.25,
            }
        )
    return {
        "prompt_tokens": length,
        "completion_tokens": 8,
        "completion_token_ids": token_ids,
        "completion_token_ids_sha256": acceptance._token_id_sha256(token_ids),
        "eos_stopping": False,
        "timing": {
            "prefill_seconds": 10.0,
            "ttft_seconds": 10.0,
            "ttft_scope": "prefill model call plus finite-logit validation and greedy argmax; excludes prompt host-to-device transfer, cache construction, and pre-run telemetry",
            "decode_model_step_seconds": 1.75,
            "all_model_step_seconds": 11.75,
            "run_wall_seconds": 12.0,
            "decode_tokens_per_second": 4.0,
        },
        "steps": steps,
        "memory_observations": observations,
        "final_cache_tokens": length + 7,
        "repeat": repeat,
        "prompt_token_ids_sha256": contract["identities"]["prompt_token_ids_sha256"][str(length)],
        "cache": {
            "mode": mode,
            "class": "transformers.cache_utils.DynamicCache",
            "offloading": mode == "offloaded",
            "offload_only_non_sliding": False,
            "payload": "exact model-produced K/V; no quantization or approximation",
        },
    }


def _seal(payload: dict[str, object]) -> dict[str, object]:
    sealed = deepcopy(payload)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "scope": acceptance.INTEGRITY_SCOPE,
        "sha256": acceptance._canonical_sha256(sealed),
    }
    return sealed


def _synthetic_report(
    contract: dict[str, object], tmp_path: Path, phase_name: str
) -> dict[str, object]:
    phase = contract["phases"][phase_name]
    keys = [
        (length, repeat, mode)
        for length in phase["request"]["lengths"]
        for repeat in range(3)
        for mode in phase["request"]["modes"]
    ]
    runs = [
        _synthetic_run(contract, length=length, repeat=repeat, mode=mode)
        for length, repeat, mode in keys
    ]
    return _seal(
        {
            "schema_version": acceptance.REPORT_SCHEMA,
            "status": "completed",
            "claim_scope": {
                "exact_cache_modes_only": True,
                "snapshot_or_archive": False,
                "nvme_or_disk_backed_decode": False,
                "sparse_or_compressed_cache": False,
                "context_extension_claim": False,
                "position_contract": "no request may exceed model max_position_embeddings",
            },
            "request": deepcopy(phase["request"]),
            "identities": _model_identities(contract, tmp_path),
            "attention_policy": {
                "requested": "cudnn",
                "forced": True,
                "resolved_pytorch_enum": "CUDNN_ATTENTION",
                "availability": {"selected_build_or_library_available": True},
            },
            "arithmetic_policy": {
                "seed": 8419,
                "greedy_argmax": True,
                "sampling": False,
                "eos_early_stopping": False,
                "requested_deterministic_algorithms": False,
                "torch_deterministic_algorithms": False,
            },
            "environment": {"device": {"requested_and_normalized": "cuda:0", "type": "cuda"}},
            "prompts": [
                {
                    "prompt_tokens": length,
                    "token_ids_sha256": contract["identities"]["prompt_token_ids_sha256"][
                        str(length)
                    ],
                    "construction": "repeat tokenizer output without special tokens, then exact truncate",
                }
                for length in phase["request"]["lengths"]
            ],
            "attempts": [
                {
                    "execution_ordinal": index,
                    "mode": mode,
                    "prompt_tokens": length,
                    "repeat": repeat,
                    "status": "completed",
                    "run_index": index,
                }
                for index, (length, repeat, mode) in enumerate(keys)
            ],
            "runs": runs,
            "errors": [],
            "model_load_memory": [
                _memory(
                    rss=512 * 2**20,
                    free=8 * acceptance.GIB,
                    peak_cuda=0,
                    label="before_model_load",
                ),
                _memory(
                    rss=acceptance.GIB,
                    free=4 * acceptance.GIB,
                    peak_cuda=3 * acceptance.GIB,
                    label="after_model_load",
                ),
            ],
            "token_identity": {
                "matrix": {
                    "expected_cells": phase["expected_cells"],
                    "actual_records": phase["expected_cells"],
                    "complete_and_unique": True,
                    "missing": [],
                    "duplicates": [],
                    "unexpected": [],
                },
                "all_completed_run_tokens_identical_by_prompt": True,
                "comparisons": [],
                "dynamic_vs_offloaded_matched_pairs": [],
                "failure_count": 0,
            },
        }
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def test_synthetic_complete_reports_pass_every_frozen_gate(tmp_path, monkeypatch) -> None:
    contract = _synthetic_contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    for phase_name in ("parity", "capacity"):
        _write_json(
            tmp_path / f"{phase_name}.json", _synthetic_report(contract, tmp_path, phase_name)
        )

    monkeypatch.setattr(acceptance, "REPOSITORY", tmp_path)
    monkeypatch.setattr(
        acceptance,
        "validate_preflight",
        lambda _path: {
            "contract": {"path": "contract.json", "sha256": "a" * 64, "size_bytes": 1},
            "runner": {"path": "runner.py", "sha256": "b" * 64, "size_bytes": 1},
            "evaluator": {"path": "evaluator.py", "sha256": "c" * 64, "size_bytes": 1},
        },
    )

    summary = acceptance.evaluate_reports(contract_path)

    assert summary["status"] == "accepted"
    assert summary["all_gates_passed"] is True
    assert all(gate["passed"] for gate in summary["gate_results"])
    assert summary["decision"] == "exact systems path through native 40,960 positions"


def test_valid_slow_capacity_report_is_preserved_as_a_gate_rejection(tmp_path, monkeypatch) -> None:
    contract = _synthetic_contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    parity = _synthetic_report(contract, tmp_path, "parity")
    capacity = _synthetic_report(contract, tmp_path, "capacity")
    capacity.pop("integrity")
    for run in capacity["runs"]:
        if run["prompt_tokens"] == 40952:
            run["timing"]["ttft_seconds"] = 61.0
            run["timing"]["prefill_seconds"] = 61.0
            run["timing"]["all_model_step_seconds"] = 62.75
            run["timing"]["run_wall_seconds"] = 63.0
            run["steps"][0]["seconds"] = 61.0
    capacity = _seal(capacity)
    _write_json(tmp_path / "parity.json", parity)
    _write_json(tmp_path / "capacity.json", capacity)
    monkeypatch.setattr(acceptance, "REPOSITORY", tmp_path)
    monkeypatch.setattr(
        acceptance,
        "validate_preflight",
        lambda _path: {
            "contract": {},
            "runner": {},
            "evaluator": {},
        },
    )

    summary = acceptance.evaluate_reports(contract_path)

    assert summary["status"] == "rejected"
    failed = [gate["name"] for gate in summary["gate_results"] if not gate["passed"]]
    assert failed == ["40952_each_ttft"]


@pytest.mark.parametrize(
    ("field", "contradictory_value"),
    [
        ("prefill_seconds", 1.0),
        ("ttft_seconds", 1.0),
        ("decode_model_step_seconds", 1.0),
        ("all_model_step_seconds", 1.0),
        ("decode_tokens_per_second", 999.0),
    ],
)
def test_reported_timing_must_match_sealed_step_timings(
    tmp_path, monkeypatch, field, contradictory_value
) -> None:
    contract = _synthetic_contract(tmp_path)
    report = _synthetic_report(contract, tmp_path, "capacity")
    report.pop("integrity")
    report["runs"][0]["timing"][field] = contradictory_value
    path = tmp_path / "capacity.json"
    _write_json(path, _seal(report))
    monkeypatch.setattr(acceptance, "REPOSITORY", tmp_path)

    with pytest.raises(acceptance.AcceptanceError, match="recorded step timings"):
        acceptance.validate_report(contract, phase_name="capacity", report_path=path)


def test_run_wall_time_cannot_be_shorter_than_sealed_model_steps(tmp_path, monkeypatch) -> None:
    contract = _synthetic_contract(tmp_path)
    report = _synthetic_report(contract, tmp_path, "capacity")
    report.pop("integrity")
    report["runs"][0]["timing"]["run_wall_seconds"] = 1.0
    path = tmp_path / "capacity.json"
    _write_json(path, _seal(report))
    monkeypatch.setattr(acceptance, "REPOSITORY", tmp_path)

    with pytest.raises(acceptance.AcceptanceError, match="shorter than"):
        acceptance.validate_report(contract, phase_name="capacity", report_path=path)


def test_structural_evaluation_failure_writes_exclusive_rejection_summary(
    tmp_path, monkeypatch
) -> None:
    contract = _synthetic_contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    monkeypatch.setattr(acceptance, "REPOSITORY", tmp_path)

    def reject(_path):
        raise acceptance.AcceptanceError("malformed report")

    monkeypatch.setattr(acceptance, "evaluate_reports", reject)

    assert acceptance.main(["--contract", str(contract_path)]) == 1
    summary = acceptance.load_json_strict(tmp_path / "summary.json")
    assert summary["status"] == "rejected"
    assert summary["decision"] == "not authorized"
    assert summary["all_gates_passed"] is False
    assert summary["evaluation_error"] == "malformed report"
    assert summary["evidence"]["models_run_by_evaluator"] is False

    assert acceptance.main(["--contract", str(contract_path)]) == 1


def test_strict_loader_rejects_duplicate_keys_and_all_nonfinite_forms(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"completed","status":"failed"}', encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="duplicate JSON object key"):
        acceptance.load_json_strict(duplicate)

    for index, payload in enumerate(('{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}')):
        path = tmp_path / f"nonfinite-{index}.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(acceptance.AcceptanceError, match="non-finite"):
            acceptance.load_json_strict(path)


def test_tampered_integrity_and_duplicate_run_cells_are_rejected(tmp_path, monkeypatch) -> None:
    contract = _synthetic_contract(tmp_path)
    report = _synthetic_report(contract, tmp_path, "parity")
    path = tmp_path / "parity.json"
    _write_json(path, report)
    monkeypatch.setattr(acceptance, "REPOSITORY", tmp_path)

    report["status"] = "failed"
    _write_json(path, report)
    with pytest.raises(acceptance.AcceptanceError, match="content integrity"):
        acceptance.validate_report(contract, phase_name="parity", report_path=path)

    report = _synthetic_report(contract, tmp_path, "parity")
    report.pop("integrity")
    report["runs"][1] = deepcopy(report["runs"][0])
    _write_json(path, _seal(report))
    with pytest.raises(acceptance.AcceptanceError, match="duplicate run cell"):
        acceptance.validate_report(contract, phase_name="parity", report_path=path)


def test_summary_creation_is_exclusive_and_evaluator_has_no_model_runtime_imports(
    tmp_path,
) -> None:
    path = tmp_path / "summary.json"
    acceptance._write_summary_exclusive(path, {"schema_version": acceptance.SUMMARY_SCHEMA})
    with pytest.raises(acceptance.AcceptanceError, match="refusing to overwrite"):
        acceptance._write_summary_exclusive(path, {"schema_version": acceptance.SUMMARY_SCHEMA})

    source = EVALUATOR_PATH.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import transformers" not in source
    assert "import subprocess" not in source
    assert "molor.experimental.exact_context_lab" not in source.replace(
        '"molor.experimental.exact_context_lab"', ""
    )
