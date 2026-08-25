from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


LAB_PATH = Path(__file__).parents[1] / "experiments" / "exact-context-v3" / "lab.py"
SPEC = importlib.util.spec_from_file_location("molor_exact_context_v3_lab_test", LAB_PATH)
assert SPEC is not None and SPEC.loader is not None
lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)


def valid_request(model: Path) -> lab.CompatibilityRequest:
    return lab.CompatibilityRequest(model=model)


def test_projected_native_capacity_is_exact() -> None:
    projected = lab.projected_memory_contract()

    assert projected["cpu_backing_bytes"] == 4_697_620_480
    assert projected["cpu_backing_gib"] == 4.375
    assert projected["shared_device_staging_bytes"] == 167_772_160
    assert projected["shared_device_staging_mib"] == 160
    assert projected["full_context_h2d_bytes_per_generated_token"] == 4_697_620_480


def test_implementation_artifacts_bind_lab_cache_and_frozen_contract() -> None:
    bundle = lab.implementation_artifacts()
    artifacts = bundle["artifacts"]

    assert set(artifacts) == {
        "lab",
        "shared_staging_cache",
        "exact_context_lab",
        "exact_context_v2_lab",
        "v1_contract",
    }
    assert artifacts["v1_contract"]["sha256"] == lab.FROZEN_V1_CONTRACT["file_sha256"]
    assert all(item["size_bytes"] > 0 for item in artifacts.values())
    assert lab._validate_implementation_bundle(bundle) == bundle


def test_request_is_local_explicit_cuda_and_native_capacity(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    request = valid_request(model)
    lab.validate_request(request)

    invalid = [
        replace(request, model=tmp_path / "missing"),
        replace(request, max_new_tokens=1),
        replace(request, max_cache_len=40_959),
        replace(request, length=40_953),
        replace(request, device="cuda"),
        replace(request, device="cpu"),
        replace(request, worker_timeout_seconds=0),
        replace(request, host_headroom_gib=float("nan")),
    ]
    for candidate in invalid:
        with pytest.raises(ValueError):
            lab.validate_request(candidate)


def test_loaded_attention_contract_requires_sdpa_before_inference() -> None:
    assert (
        lab.validate_loaded_attention_implementation(SimpleNamespace(_attn_implementation="sdpa"))[
            "matches"
        ]
        is True
    )
    with pytest.raises(RuntimeError, match="SDPA"):
        lab.validate_loaded_attention_implementation(SimpleNamespace(_attn_implementation="eager"))


def test_command_contract_has_two_distinct_one_mode_workers(tmp_path: Path) -> None:
    request = valid_request(tmp_path / "not-read-in-dry-contract")
    output = tmp_path / "aggregate.json"

    contract = lab.command_contract(
        request,
        final_output=output,
        python_executable="python-under-test",
        script_path=LAB_PATH,
    )

    assert contract["runs_model"] is False
    assert contract["one_cache_mode_per_process"] is True
    pinned_hash = contract["implementation_bundle"]["bundle_sha256"]
    assert [item["mode"] for item in contract["commands"]] == [
        "dynamic",
        "shared_staging",
    ]
    worker_outputs = {item["worker_output"] for item in contract["commands"]}
    assert len(worker_outputs) == 2
    for item in contract["commands"]:
        argv = item["argv"]
        assert argv[:2] == ["python-under-test", str(LAB_PATH)]
        assert argv[argv.index("--_worker-mode") + 1] == item["mode"]
        assert argv[argv.index("--_expected-implementation-bundle-sha256") + 1] == pinned_hash
        encoded = argv[argv.index("--_expected-implementation-snapshot") + 1]
        parsed = lab._parser().parse_args(argv[2:])
        assert parsed._expected_implementation_snapshot == encoded
        assert lab._decode_pinned_implementation_bundle(parsed) == contract["implementation_bundle"]
        assert "--allow-download" not in argv
        assert "--revision" not in argv


def test_emit_command_contract_does_not_require_model_or_create_output(
    tmp_path: Path, capsys
) -> None:
    missing_model = tmp_path / "missing-model"
    output = tmp_path / "never-created.json"

    assert (
        lab.main(
            [
                "--model",
                str(missing_model),
                "--output",
                str(output),
                "--emit-command-contract",
            ]
        )
        == 0
    )
    assert lab.COMMAND_SCHEMA in capsys.readouterr().out
    assert not output.exists()


def test_output_set_is_refused_as_a_group(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    paths = lab._validate_output_set_absent(output)
    assert set(paths) == {"dynamic", "shared_staging"}
    paths["shared_staging"].write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="before workers start"):
        lab._validate_output_set_absent(output)


def test_worker_refuses_existing_output_before_body(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "worker.json"
    output.write_text("preserve", encoding="utf-8")
    called = False

    def forbidden_body(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    args = lab._parser().parse_args(
        [
            "--model",
            str(model),
            "--_worker-mode",
            "dynamic",
            "--_worker-output",
            str(output),
        ]
    )
    request = lab._request_from_args(args)

    assert lab._worker_main(args, request, body=forbidden_body) == 1
    assert called is False
    assert output.read_text(encoding="utf-8") == "preserve"


def _drifted_bundle(bundle: dict) -> dict:
    drifted = deepcopy(bundle)
    drifted["artifacts"]["lab"]["sha256"] = "0" * 64
    core = {
        "schema_version": drifted["schema_version"],
        "artifacts": drifted["artifacts"],
    }
    drifted["bundle_sha256"] = lab._canonical_sha256(core)
    return drifted


def _worker_args(
    request: lab.CompatibilityRequest,
    output: Path,
    bundle: dict,
    *,
    mode: str = "dynamic",
):
    argv = lab._build_worker_argv(
        request,
        mode=mode,
        worker_output=output,
        implementation_bundle=bundle,
        python_executable="python-under-test",
        script_path=LAB_PATH,
    )
    return lab._parser().parse_args(argv[2:])


def test_worker_rejects_source_drift_before_body(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    request = valid_request(model)
    expected = lab.implementation_artifacts()
    observed = _drifted_bundle(expected)
    output = tmp_path / "preflight-drift.json"
    called = False

    def forbidden_body(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    assert (
        lab._worker_main(
            _worker_args(request, output, expected),
            request,
            body=forbidden_body,
            bundle_reader=lambda: deepcopy(observed),
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert called is False
    assert report["status"] == "failed"
    assert "changed from supervisor pin" in report["errors"][0]["message"]


def test_worker_rejects_mid_worker_source_drift_after_cleanup(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    request = valid_request(model)
    expected = lab.implementation_artifacts()
    observed = iter((deepcopy(expected), _drifted_bundle(expected)))
    output = tmp_path / "postflight-drift.json"
    called = False

    def harmless_body(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    assert (
        lab._worker_main(
            _worker_args(request, output, expected),
            request,
            body=harmless_body,
            bundle_reader=lambda: next(observed),
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert called is True
    assert report["status"] == "failed"
    assert "postflight_validation_error" in report["implementation_provenance"]
    assert "changed from supervisor pin" in report["errors"][0]["message"]


def test_atomic_exclusive_publication_never_replaces(tmp_path: Path) -> None:
    output = tmp_path / "sealed.json"
    lab._write_json_atomic_exclusive(output, {"complete": True})
    first = output.read_bytes()

    with pytest.raises(FileExistsError):
        lab._write_json_atomic_exclusive(output, {"complete": False})

    assert output.read_bytes() == first
    assert json.loads(first) == {"complete": True}
    assert list(tmp_path.glob(".*.tmp-*")) == []


def _supervisor_args(model: Path, output: Path):
    args = lab._parser().parse_args(["--model", str(model), "--output", str(output)])
    return args, lab._request_from_args(args)


def test_supervisor_gates_shared_worker_after_baseline_failure(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "aggregate.json"
    args, request = _supervisor_args(model, output)
    calls: list[list[str]] = []

    def failed_baseline(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 3, stdout="baseline failed", stderr="failure")

    assert lab._supervisor_main(args, request, runner=failed_baseline) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert calls[0][calls[0].index("--_worker-mode") + 1] == "dynamic"
    assert report["status"] == "failed"
    assert any(error["type"] == "baseline_gate" for error in report["errors"])
    assert report["implementation_postflight"] == report["implementation_bundle"]


def test_supervisor_normalizes_timeout_bytes_and_gates_shared_worker(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "aggregate-timeout.json"
    args, request = _supervisor_args(model, output)
    calls = 0

    def timed_out(argv, **kwargs):
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(
            cmd=argv,
            timeout=kwargs["timeout"],
            output=b"\xffpartial stdout",
            stderr=b"partial stderr",
        )

    assert lab._supervisor_main(args, request, runner=timed_out) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    invocation = report["worker_artifacts"][0]
    assert calls == 1
    assert invocation["timed_out"] is True
    assert invocation["stdout"]["text"] == "\ufffdpartial stdout"
    assert invocation["stderr"]["text"] == "partial stderr"
    assert any(error["type"] == "baseline_gate" for error in report["errors"])


def test_supervisor_rejects_own_source_drift_before_sealing(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "aggregate-source-drift.json"
    args, request = _supervisor_args(model, output)
    pinned = lab.implementation_artifacts()
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    def successful_worker(argv, **_kwargs):
        mode = argv[argv.index("--_worker-mode") + 1]
        worker_output = Path(argv[argv.index("--_worker-output") + 1])
        worker = _completed_worker(mode, token_ids, request=request, bundle=pinned)
        lab._write_json_atomic_exclusive(worker_output, lab._seal_report(worker))
        return subprocess.CompletedProcess(argv, 0, stdout="completed", stderr="")

    bundles = iter((deepcopy(pinned), _drifted_bundle(pinned)))
    assert (
        lab._supervisor_main(
            args,
            request,
            runner=successful_worker,
            bundle_reader=lambda: next(bundles),
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["comparison"]["exact_parity_passed"] is True
    assert report["status"] == "failed"
    assert "changed from supervisor pin" in report["implementation_postflight_error"]["message"]


def _valid_shared_telemetry(request: lab.CompatibilityRequest) -> dict:
    projected = lab.projected_memory_contract(dtype=torch.bfloat16)
    layers = lab.FROZEN_V1_CONTRACT["num_hidden_layers"]
    heads = lab.FROZEN_V1_CONTRACT["num_key_value_heads"]
    head_dim = lab.FROZEN_V1_CONTRACT["head_dim"]
    role_bytes = heads * request.max_cache_len * head_dim * projected["dtype_bytes"]
    final_length = request.length + request.max_new_tokens - 1
    device = str(torch.device(request.device))
    return {
        "schema_version": lab.SHARED_CACHE_TELEMETRY_SCHEMA,
        "geometry": {
            "num_layers": layers,
            "max_cache_len": request.max_cache_len,
            "num_key_value_heads": heads,
            "key_head_dim": head_dim,
            "value_head_dim": head_dim,
            "dtype": request.dtype,
            "staging_device": device,
            "pin_cpu_backing": True,
            "non_blocking_copies": True,
            "batch_size": 1,
            "layout": "batch_heads_sequence_head_dim",
        },
        "health": {
            "poisoned": False,
            "active_poison_reason": None,
            "poison_events": 0,
            "poison_events_by_reason": {},
            "cuda_stream_contract": "one_stable_current_stream",
            "cuda_stream_bound": True,
        },
        "sequence": {
            "layer_logical_lengths": [final_length] * layers,
            "expected_next_layer_idx": 0,
            "update_cycle_in_progress": False,
        },
        "storage": {
            "backing_device": "cpu",
            "backing_bytes": projected["cpu_backing_bytes"],
            "layer_backing": [
                {
                    "layer_idx": layer_idx,
                    "initialized": True,
                    "logical_length": final_length,
                    "key_bytes": role_bytes,
                    "value_bytes": role_bytes,
                    "total_bytes": role_bytes * 2,
                }
                for layer_idx in range(layers)
            ],
            "staging_device": device,
            "staging_allocated": True,
            "staging_key_bytes": role_bytes,
            "staging_value_bytes": role_bytes,
            "staging_bytes": projected["shared_device_staging_bytes"],
            "staging_pair_allocations": 1,
            "live_cache_bytes": projected["cpu_backing_bytes"]
            + projected["shared_device_staging_bytes"],
        },
        "operations": {
            "updates_by_layer": [request.max_new_tokens] * layers,
            "prefills_by_layer": [1] * layers,
            "decodes_by_layer": [request.max_new_tokens - 1] * layers,
            "resets": 0,
        },
        "transfers": lab._expected_shared_transfer_contract(request),
    }


def test_shared_telemetry_validation_proves_storage_and_transfers(tmp_path: Path) -> None:
    request = valid_request(tmp_path)
    telemetry = _valid_shared_telemetry(request)

    validated = lab.validate_shared_cache_telemetry(telemetry, request=request)

    assert validated["all_checks_passed"] is True
    assert validated["expected_final_cache_tokens"] == 4_103
    assert validated["decode_steps_that_append_to_cache"] == 7
    assert validated["full_history_h2d_is_still_present"] is True

    assert validated["expected_total_copies"] == 840
    assert validated["events_retained"] == 256
    assert validated["events_dropped"] == 584

    telemetry["health"]["poisoned"] = True
    with pytest.raises(RuntimeError, match="health"):
        lab.validate_shared_cache_telemetry(telemetry, request=request)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda item: item["sequence"].update(expected_next_layer_idx=1), "sequence"),
        (lambda item: item["operations"].update(resets=1), "operation"),
        (lambda item: item["transfers"].update(event_capacity=255), "ledger"),
        (lambda item: item["transfers"]["events"][0].update(ordinal=0), "ledger"),
        (lambda item: item["transfers"].update(events_dropped=583), "ledger"),
        (
            lambda item: item["transfers"]["by_direction"]["cpu_to_cuda:0"].update(bytes=1),
            "ledger",
        ),
    ],
)
def test_shared_telemetry_rejects_incomplete_or_inconsistent_evidence(
    tmp_path: Path, mutator, message: str
) -> None:
    request = valid_request(tmp_path)
    telemetry = _valid_shared_telemetry(request)
    mutator(telemetry)

    with pytest.raises(RuntimeError, match=message):
        lab.validate_shared_cache_telemetry(telemetry, request=request)


def _completed_worker(
    mode: str,
    token_ids: list[int],
    *,
    request: lab.CompatibilityRequest,
    bundle: dict,
) -> dict:
    frozen = lab.FROZEN_V1_CONTRACT
    prompt_hash = lab.FROZEN_PROMPT_HASHES[request.length]
    return {
        "schema_version": lab.WORKER_SCHEMA,
        "status": "completed",
        "mode": mode,
        "request": request.to_report_dict(),
        "implementation_provenance": {
            "expected": deepcopy(bundle),
            "preflight_observed": deepcopy(bundle),
            "postflight_observed": deepcopy(bundle),
        },
        "prompt": {"tokens": request.length, "token_ids_sha256": prompt_hash},
        "identities": {
            "model_artifact": {
                "stack_fingerprint": frozen["model_stack_fingerprint"],
                "artifacts": [
                    {
                        "role": "base",
                        "content_fingerprint": frozen["model_content_fingerprint"],
                    }
                ],
            },
            "config": {"canonical_sha256": frozen["config_canonical_sha256"]},
            "tokenizer": {
                "vocabulary_sha256": frozen["tokenizer_vocabulary_sha256"],
                "backend_json_sha256": frozen["tokenizer_backend_json_sha256"],
                "special_tokens_map_sha256": frozen["tokenizer_special_tokens_map_sha256"],
                "chat_template_sha256": frozen["tokenizer_chat_template_sha256"],
            },
        },
        "run": {
            "mode": mode,
            "prompt_tokens": request.length,
            "prompt_token_ids_sha256": prompt_hash,
            "completion_tokens": request.max_new_tokens,
            "completion_token_ids": token_ids,
            "completion_token_ids_sha256": lab._token_id_sha256(token_ids),
            "final_cache_tokens": request.length + request.max_new_tokens - 1,
        },
    }


def test_worker_comparison_requires_exact_token_ids_and_input_identities(tmp_path: Path) -> None:
    request = valid_request(tmp_path)
    bundle = lab.implementation_artifacts()
    token_ids = [3, 5, 8, 13, 21, 34, 55, 89]
    dynamic = _completed_worker("dynamic", token_ids, request=request, bundle=bundle)
    shared = _completed_worker("shared_staging", token_ids, request=request, bundle=bundle)

    comparison = lab.compare_worker_reports(
        dynamic,
        shared,
        expected_request=request,
        expected_implementation_bundle=bundle,
    )
    assert comparison["exact_parity_passed"] is True
    assert comparison["completion"]["first_mismatch_index"] is None

    changed_ids = [3, 7, 8, 13, 21, 34, 55, 89]
    shared["run"]["completion_token_ids"] = changed_ids
    shared["run"]["completion_token_ids_sha256"] = lab._token_id_sha256(changed_ids)
    comparison = lab.compare_worker_reports(
        dynamic,
        shared,
        expected_request=request,
        expected_implementation_bundle=bundle,
    )
    assert comparison["exact_parity_passed"] is False
    assert comparison["completion"]["first_mismatch_index"] == 1

    shared = _completed_worker("shared_staging", token_ids, request=request, bundle=bundle)
    shared["identities"]["config"]["canonical_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="config_canonical_sha256"):
        lab.compare_worker_reports(
            dynamic,
            shared,
            expected_request=request,
            expected_implementation_bundle=bundle,
        )


def test_worker_comparison_rejects_old_bogus_hash_pass(tmp_path: Path) -> None:
    request = valid_request(tmp_path)
    bundle = lab.implementation_artifacts()
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    dynamic = _completed_worker("dynamic", token_ids, request=request, bundle=bundle)
    shared = _completed_worker("shared_staging", token_ids, request=request, bundle=bundle)
    dynamic["run"]["completion_token_ids_sha256"] = "0" * 64
    shared["run"]["completion_token_ids_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="does not match its token IDs"):
        lab.compare_worker_reports(
            dynamic,
            shared,
            expected_request=request,
            expected_implementation_bundle=bundle,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda worker: worker["run"].update(completion_token_ids=[True] * 8),
        lambda worker: worker["run"].update(final_cache_tokens=4_104),
        lambda worker: worker["run"].update(prompt_token_ids_sha256="different"),
        lambda worker: worker["request"].update(seed=1),
    ],
)
def test_worker_comparison_rejects_malformed_worker_claims(tmp_path: Path, mutation) -> None:
    request = valid_request(tmp_path)
    bundle = lab.implementation_artifacts()
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    dynamic = _completed_worker("dynamic", token_ids, request=request, bundle=bundle)
    shared = _completed_worker("shared_staging", token_ids, request=request, bundle=bundle)
    mutation(shared)

    with pytest.raises(RuntimeError):
        lab.compare_worker_reports(
            dynamic,
            shared,
            expected_request=request,
            expected_implementation_bundle=bundle,
        )


def test_memory_summary_keeps_process_allocator_and_global_scopes_separate() -> None:
    observations = [
        {
            "process": {
                "current_rss_bytes": 10,
                "peak_rss_bytes": 12,
                "private_committed_bytes": 14,
            },
            "cuda": {
                "process_allocator_allocated_bytes": 20,
                "process_allocator_reserved_bytes": 25,
                "process_allocator_peak_allocated_bytes": 30,
                "device_global_free_bytes": 100,
            },
            "nvidia_smi": {"free_bytes_assuming_mib": 95},
            "live_cache_storage": {
                "exact_cpu_backing_bytes": 40,
                "exact_shared_staging_bytes": 5,
            },
        },
        {
            "process": {
                "current_rss_bytes": 11,
                "peak_rss_bytes": 13,
                "private_committed_bytes": 15,
            },
            "cuda": {
                "process_allocator_allocated_bytes": 21,
                "process_allocator_reserved_bytes": 26,
                "process_allocator_peak_allocated_bytes": 31,
                "device_global_free_bytes": 90,
            },
            "nvidia_smi": {"free_bytes_assuming_mib": 85},
            "live_cache_storage": {
                "exact_cpu_backing_bytes": 40,
                "exact_shared_staging_bytes": 5,
            },
        },
    ]

    summary = lab.summarize_memory(observations)

    assert summary["max_current_rss_bytes"] == 11
    assert summary["max_private_committed_bytes"] == 15
    assert summary["max_torch_cuda_peak_allocated_bytes"] == 31
    assert summary["minimum_cuda_device_global_free_bytes"] == 90
    assert summary["minimum_nvidia_smi_global_free_bytes"] == 85
    assert "different scopes" in summary["separation_note"]


def test_nvidia_smi_snapshot_is_best_effort_and_parses_mib() -> None:
    def successful_runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout="9790, 12288\n", stderr="")

    snapshot = lab._nvidia_smi_snapshot(0, runner=successful_runner)
    assert snapshot["status"] == "available"
    assert snapshot["free_bytes_assuming_mib"] == 9_790 * 2**20

    def missing_runner(*_args, **_kwargs):
        raise FileNotFoundError("not installed")

    snapshot = lab._nvidia_smi_snapshot(0, runner=missing_runner)
    assert snapshot["status"] == "unavailable"


def test_frozen_identity_validation_reports_each_exact_check() -> None:
    frozen = lab.FROZEN_V1_CONTRACT
    model = {
        "stack_fingerprint": frozen["model_stack_fingerprint"],
        "artifacts": [
            {
                "role": "base",
                "content_fingerprint": frozen["model_content_fingerprint"],
            }
        ],
    }
    config = {"canonical_sha256": frozen["config_canonical_sha256"]}
    tokenizer = {
        "vocabulary_sha256": frozen["tokenizer_vocabulary_sha256"],
        "backend_json_sha256": frozen["tokenizer_backend_json_sha256"],
        "special_tokens_map_sha256": frozen["tokenizer_special_tokens_map_sha256"],
        "chat_template_sha256": frozen["tokenizer_chat_template_sha256"],
    }

    evidence = lab.validate_frozen_identities(
        model_identity=model,
        config_identity=config,
        tokenizer_identity=tokenizer,
    )
    assert evidence["all_match"] is True

    tokenizer["vocabulary_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="tokenizer_vocabulary_sha256"):
        lab.validate_frozen_identities(
            model_identity=model,
            config_identity=config,
            tokenizer_identity=tokenizer,
        )
