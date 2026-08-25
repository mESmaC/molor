"""Pure report evaluator for the preregistered exact-context-v1 experiment.

This module intentionally uses only the Python standard library.  It never imports the
benchmark runner, PyTorch, Transformers, or a process-launching API.  Normal evaluation
parses the two frozen reports and creates one exclusive summary; ``--preflight`` checks
the frozen contract and local source/model bytes without requiring either report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time
from typing import Any


CONTRACT_SCHEMA = "molor.exact-context-acceptance-contract.v1"
REPORT_SCHEMA = "molor.exact-context-report.v1"
SUMMARY_SCHEMA = "molor.exact-context-acceptance-summary.v1"
INTEGRITY_SCOPE = "canonical JSON of this report with the integrity member omitted"
GIB = 2**30
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = Path(__file__).with_name("contract.json")


class AcceptanceError(ValueError):
    """The frozen contract or supplied evidence is invalid."""


@dataclass(frozen=True)
class PhaseEvidence:
    """Validated evidence extracted from one report."""

    name: str
    path: Path
    report_sha256: str
    run_by_key: dict[tuple[int, int, str], dict[str, Any]]
    minimum_global_cuda_free_bytes: int
    incremental_rss_bytes: int


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AcceptanceError(f"non-finite JSON constant is forbidden: {value}")


def _reject_nonfinite(value: Any, *, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AcceptanceError(f"non-finite numeric value at {location}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, location=f"{location}[{index}]")


def load_json_strict(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object while rejecting duplicate keys and non-finite values."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AcceptanceError(f"cannot read {path}: {error}") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise AcceptanceError(f"invalid strict JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AcceptanceError(f"JSON root must be an object: {path}")
    _reject_nonfinite(payload)
    return payload


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AcceptanceError(f"artifact changed while hashing: {path}")
    return digest.hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def _require_exact(actual: Any, expected: Any, location: str) -> None:
    if actual != expected:
        raise AcceptanceError(f"{location} differs from the frozen contract")


def _relative_path(raw: Any, *, field: str) -> Path:
    _require(isinstance(raw, str) and raw, f"{field} must be a non-empty path string")
    path = Path(raw)
    _require(not path.is_absolute(), f"{field} must be repository-relative")
    _require(".." not in path.parts, f"{field} must not escape the repository")
    resolved = (REPOSITORY / path).resolve()
    try:
        resolved.relative_to(REPOSITORY.resolve())
    except ValueError as error:
        raise AcceptanceError(f"{field} escapes the repository") from error
    return resolved


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(REPOSITORY).as_posix(),
        "sha256": _file_sha256(path),
        "size_bytes": stat.st_size,
    }


def _validate_file_descriptor(descriptor: Any, *, field: str) -> Path:
    _require(isinstance(descriptor, dict), f"{field} must be an object")
    _require(
        set(descriptor) == {"path", "sha256", "size_bytes"},
        f"{field} has unexpected descriptor fields",
    )
    path = _relative_path(descriptor["path"], field=f"{field}.path")
    _require(path.is_file(), f"frozen artifact is missing: {path}")
    _require(_is_int(descriptor["size_bytes"]), f"{field}.size_bytes must be an integer")
    _require(path.stat().st_size == descriptor["size_bytes"], f"{field} size differs")
    observed_hash = _file_sha256(path)
    _require(observed_hash == descriptor["sha256"], f"{field} SHA-256 differs")
    return path


def _expected_runner_command(phase: Mapping[str, Any]) -> list[str]:
    request = phase["request"]
    return [
        r".\.venv\Scripts\python.exe",
        "-m",
        "molor.experimental.exact_context_lab",
        "--model",
        request["model"],
        "--modes",
        *request["modes"],
        "--lengths",
        *(str(value) for value in request["lengths"]),
        "--max-new-tokens",
        str(request["max_new_tokens"]),
        "--repeats",
        str(request["repeats"]),
        "--seed",
        str(request["seed"]),
        "--device",
        request["device"],
        "--dtype",
        request["dtype"],
        "--attn-implementation",
        request["attention_implementation"],
        "--sdpa-backend",
        request["sdpa_backend"],
        "--no-deterministic-algorithms",
        "--minimum-free-cuda-gib",
        str(request["minimum_free_cuda_gib"]),
        "--cuda-headroom-gib",
        str(request["cuda_headroom_gib"]),
        "--output",
        phase["report_path"],
    ]


def _validate_frozen_policy(contract: dict[str, Any]) -> None:
    _require_exact(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _require_exact(contract.get("status"), "frozen_before_reports", "contract status")
    _require_exact(
        contract.get("source_commit"),
        "84f95c260420b3f07e75886e3bdfc1f27ff26f99",
        "source commit",
    )
    claims = contract.get("decision_scope")
    _require(isinstance(claims, dict), "decision_scope must be an object")
    _require_exact(
        claims.get("authorized_only_if_all_gates_pass"),
        "exact systems path through native 40,960 positions",
        "authorized decision scope",
    )
    _require_exact(
        claims.get("not_authorized"),
        [
            "semantic long-context quality",
            "positions above 40,960",
            "sparse KV",
            "ATXY KV",
            "compressed KV",
            "NVMe or disk-backed KV",
        ],
        "excluded decision scopes",
    )
    _require_exact(
        contract.get("failure_policy"),
        {
            "retries": 0,
            "rule": "Any missing, malformed, failed, or gate-failing frozen report rejects the decision; preserve both reports and the exclusive summary without rerunning or tuning this v1 contract.",
        },
        "failure policy",
    )

    phases = contract.get("phases")
    _require(isinstance(phases, dict), "phases must be an object")
    _require_exact(set(phases), {"parity", "capacity"}, "phase names")
    expected_shapes = {
        "parity": {"modes": ["offloaded", "dynamic"], "lengths": [8192, 16384]},
        "capacity": {"modes": ["offloaded"], "lengths": [32768, 40952]},
    }
    paths: set[str] = set()
    for name, shape in expected_shapes.items():
        phase = phases[name]
        _require(isinstance(phase, dict), f"phase {name} must be an object")
        request = phase.get("request")
        _require(isinstance(request, dict), f"phase {name} request must be an object")
        expected_request = {
            "model": "models/Qwen3-1.7B",
            "revision": None,
            "allow_download": False,
            **shape,
            "max_new_tokens": 8,
            "repeats": 3,
            "seed": 8419,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "sdpa_backend": "cudnn",
            "deterministic_algorithms": False,
            "logits_to_keep": 1,
            "mode_order": "declared CLI order inside each length and repeat; no warm-up",
            "minimum_free_cuda_gib": 5.0,
            "cuda_headroom_gib": 1.0,
            "validated_max_position_embeddings": 40960,
            "validated_loaded_max_position_embeddings": 40960,
        }
        _require_exact(request, expected_request, f"phase {name} request")
        expected_cells = len(shape["modes"]) * len(shape["lengths"]) * 3
        _require_exact(phase.get("expected_cells"), expected_cells, f"phase {name} cells")
        report_path = phase.get("report_path")
        _relative_path(report_path, field=f"phase {name} report_path")
        _require(report_path not in paths, "phase report paths must be exclusive")
        paths.add(report_path)
        command_request = dict(request)
        command_request.pop("validated_max_position_embeddings")
        command_request.pop("validated_loaded_max_position_embeddings")
        command_phase = {"request": command_request, "report_path": report_path}
        _require_exact(
            phase.get("command_argv"),
            _expected_runner_command(command_phase),
            f"phase {name} command",
        )

    summary = contract.get("summary")
    _require(isinstance(summary, dict), "summary must be an object")
    summary_path = summary.get("path")
    _relative_path(summary_path, field="summary.path")
    _require(summary_path not in paths, "summary path must be exclusive from report paths")
    _require_exact(
        summary.get("command_argv"),
        [
            r".\.venv\Scripts\python.exe",
            "experiments/exact-context-v1/evaluate_acceptance.py",
            "--contract",
            "experiments/exact-context-v1/contract.json",
        ],
        "summary evaluator command",
    )

    gates = contract.get("gates")
    _require(isinstance(gates, dict), "gates must be an object")
    _require_exact(
        gates,
        {
            "minimum_global_cuda_free_bytes": GIB,
            "maximum_incremental_process_rss_bytes": 6 * GIB,
            "capacity_40952": {
                "maximum_each_ttft_seconds": 60.0,
                "minimum_median_decode_tokens_per_second": 2.0,
                "maximum_nearest_rank_p95_single_token_decode_seconds": 1.0,
            },
            "parity_16384": {
                "minimum_dynamic_median_minus_maximum_offloaded_peak_cuda_allocated_bytes": int(
                    1.5 * GIB
                )
            },
        },
        "acceptance gates",
    )


def validate_preflight(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Validate the frozen contract, evaluator/runner bytes, and local model bytes."""

    contract_path = contract_path.resolve()
    contract = load_json_strict(contract_path)
    _validate_frozen_policy(contract)
    _require(
        contract_path == _relative_path(contract["contract_path"], field="contract_path"),
        "the loaded contract is not the frozen contract_path",
    )

    artifacts = contract.get("artifacts")
    _require(isinstance(artifacts, dict), "artifacts must be an object")
    _require_exact(set(artifacts), {"runner", "evaluator", "model"}, "artifact roles")
    runner_path = _validate_file_descriptor(artifacts["runner"], field="artifacts.runner")
    evaluator_path = _validate_file_descriptor(artifacts["evaluator"], field="artifacts.evaluator")
    _require(
        runner_path == (REPOSITORY / "src/molor/experimental/exact_context_lab.py").resolve(),
        "runner descriptor is not the exact-context runner",
    )
    _require(evaluator_path == Path(__file__).resolve(), "evaluator descriptor is not this file")

    model = artifacts["model"]
    _require(isinstance(model, dict), "artifacts.model must be an object")
    model_path = _relative_path(model.get("path"), field="artifacts.model.path")
    _require(model_path.is_dir(), "frozen model directory is missing")
    files = model.get("files")
    _require(isinstance(files, list) and files, "model files must be a non-empty list")
    names: list[str] = []
    total_size = 0
    for index, descriptor in enumerate(files):
        _require(isinstance(descriptor, dict), f"model file {index} must be an object")
        _require(
            set(descriptor) == {"name", "sha256", "size_bytes"},
            f"model file {index} descriptor fields differ",
        )
        name = descriptor["name"]
        _require(isinstance(name, str) and name, f"model file {index} name is invalid")
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts, "invalid model file")
        path = (model_path / relative).resolve()
        try:
            path.relative_to(model_path)
        except ValueError as error:
            raise AcceptanceError("model file escapes model directory") from error
        _require(path.is_file(), f"frozen model file is missing: {name}")
        _require(_is_int(descriptor["size_bytes"]), f"model file {name} size is invalid")
        _require(
            path.stat().st_size == descriptor["size_bytes"], f"model file size differs: {name}"
        )
        _require(_file_sha256(path) == descriptor["sha256"], f"model file hash differs: {name}")
        names.append(name)
        total_size += descriptor["size_bytes"]
    _require(len(names) == len(set(names)), "model file descriptors contain duplicates")
    _require(names == sorted(names, key=str.casefold), "model file descriptors are not sorted")
    _require_exact(total_size, model.get("size_bytes"), "model total size")
    _require_exact(
        model.get("content_fingerprint"),
        "772bbbfd94e3a63f48e2f315ea8f4d2318c66290c1d98d624dd9bc679c85bb7c",
        "model content fingerprint",
    )
    _require_exact(
        model.get("runner_stack_fingerprint"),
        "c89442b5bbc335341e634f163f41ba193e0412f9e80676cf0e8a283ad60daa91",
        "runner model stack fingerprint",
    )

    return {
        "schema_version": "molor.exact-context-preflight.v1",
        "status": "passed",
        "contract": _artifact_descriptor(contract_path),
        "runner": _artifact_descriptor(runner_path),
        "evaluator": _artifact_descriptor(evaluator_path),
        "model_content_fingerprint": model["content_fingerprint"],
        "model_size_bytes": total_size,
        "reports_required": False,
        "models_run": False,
    }


def _verify_report_integrity(report: dict[str, Any], *, phase: str) -> None:
    candidate = deepcopy(report)
    integrity = candidate.pop("integrity", None)
    _require(isinstance(integrity, dict), f"{phase} report has no integrity object")
    _require_exact(set(integrity), {"algorithm", "scope", "sha256"}, f"{phase} integrity fields")
    _require_exact(integrity.get("algorithm"), "sha256", f"{phase} integrity algorithm")
    _require_exact(integrity.get("scope"), INTEGRITY_SCOPE, f"{phase} integrity scope")
    _require_exact(
        integrity.get("sha256"), _canonical_sha256(candidate), f"{phase} content integrity"
    )


def _token_id_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"molor.token-ids.signed-int64-big-endian.v1\0")
    digest.update(struct.pack(">Q", len(token_ids)))
    for token_id in token_ids:
        _require(_is_int(token_id), "completion token IDs must be integers")
        try:
            digest.update(struct.pack(">q", token_id))
        except struct.error as error:
            raise AcceptanceError("completion token ID exceeds signed int64") from error
    return digest.hexdigest()


def _positive_number(value: Any, *, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number) and number > 0, f"{field} must be finite and positive")
    return number


def _derived_run_timings(run: Mapping[str, Any]) -> dict[str, float]:
    """Derive acceptance timings from the sealed per-forward measurements."""

    steps = run["steps"]
    prefill_seconds = float(steps[0]["seconds"])
    decode_seconds = sum(float(step["seconds"]) for step in steps[1:])
    all_step_seconds = sum(float(step["seconds"]) for step in steps)
    return {
        "prefill_seconds": prefill_seconds,
        "ttft_seconds": prefill_seconds,
        "decode_model_step_seconds": decode_seconds,
        "all_model_step_seconds": all_step_seconds,
        "decode_tokens_per_second": (len(steps) - 1) / decode_seconds,
    }


def _nonnegative_int(value: Any, *, field: str) -> int:
    _require(_is_int(value) and value >= 0, f"{field} must be a non-negative integer")
    return value


def _validate_memory_observation(observation: Any, *, field: str) -> tuple[int, int, int]:
    _require(isinstance(observation, dict), f"{field} must be an object")
    process = observation.get("process")
    cuda = observation.get("cuda")
    _require(isinstance(process, dict), f"{field}.process must be an object")
    _require(isinstance(cuda, dict), f"{field}.cuda must be an object")
    rss = _nonnegative_int(process.get("current_rss_bytes"), field=f"{field}.current_rss")
    free = _nonnegative_int(
        cuda.get("device_global_free_bytes"), field=f"{field}.device_global_free"
    )
    peak = _nonnegative_int(
        cuda.get("process_allocator_peak_allocated_bytes"),
        field=f"{field}.process_allocator_peak_allocated",
    )
    return rss, free, peak


def _validate_model_identity(
    report: dict[str, Any], contract: dict[str, Any], *, phase: str
) -> None:
    identities = report.get("identities")
    _require(isinstance(identities, dict), f"{phase} identities must be an object")
    frozen = contract["identities"]
    _require_exact(identities.get("config"), frozen["config"], f"{phase} config identity")
    _require_exact(identities.get("tokenizer"), frozen["tokenizer"], f"{phase} tokenizer identity")

    artifact = identities.get("model_artifact")
    _require(isinstance(artifact, dict), f"{phase} model artifact identity is missing")
    model = contract["artifacts"]["model"]
    expected_model_path = _relative_path(model["path"], field="artifacts.model.path")
    _require_exact(
        artifact.get("schema_version"),
        "molor.benchmark-stack-lock.v2",
        f"{phase} model lock schema",
    )
    _require_exact(
        artifact.get("hash_policy"), "sha256-full-read-every-lock", f"{phase} hash policy"
    )
    _require_exact(
        artifact.get("stable_backend_identity"),
        {"backend": "huggingface-local"},
        f"{phase} stable backend",
    )
    _require_exact(artifact.get("unresolved"), [], f"{phase} unresolved artifacts")
    _require_exact(artifact.get("fully_content_addressed"), True, f"{phase} content addressing")
    _require_exact(
        artifact.get("stack_fingerprint"),
        model["runner_stack_fingerprint"],
        f"{phase} model stack fingerprint",
    )
    backend = artifact.get("backend_identity")
    _require(isinstance(backend, dict), f"{phase} backend identity is missing")
    _require_exact(backend.get("backend"), "huggingface-local", f"{phase} backend type")
    for field in ("model_id", "model_artifact_path"):
        _require(isinstance(backend.get(field), str), f"{phase} backend {field} is missing")
        _require(
            Path(backend[field]).resolve() == expected_model_path, f"{phase} {field} path differs"
        )
    required = artifact.get("required_roles")
    _require(isinstance(required, list) and len(required) == 1, f"{phase} required roles differ")
    _require_exact(required[0].get("role"), "base", f"{phase} required base role")
    _require(
        Path(required[0].get("identifier", "")).resolve() == expected_model_path,
        f"{phase} required base path differs",
    )
    artifacts = artifact.get("artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == 1, f"{phase} base artifacts differ")
    base = artifacts[0]
    _require_exact(base.get("role"), "base", f"{phase} base role")
    _require(
        Path(base.get("path", "")).resolve() == expected_model_path, f"{phase} base path differs"
    )
    _require_exact(
        base.get("content_fingerprint"), model["content_fingerprint"], f"{phase} model fingerprint"
    )
    _require_exact(base.get("size_bytes"), model["size_bytes"], f"{phase} model size")
    _require_exact(base.get("files"), model["files"], f"{phase} model files")
    _require_exact(base.get("complete"), True, f"{phase} complete model lock")

    recheck = identities.get("model_artifact_recheck")
    _require(isinstance(recheck, dict), f"{phase} model artifact recheck is missing")
    _require_exact(recheck.get("matches_preload_identity"), True, f"{phase} model recheck")
    _require_exact(
        recheck.get("stack_fingerprint"),
        model["runner_stack_fingerprint"],
        f"{phase} rechecked model fingerprint",
    )
    _positive_number(recheck.get("seconds"), field=f"{phase}.model_artifact_recheck.seconds")

    loaded = identities.get("loaded_model")
    _require(isinstance(loaded, dict), f"{phase} loaded model identity is missing")
    _require_exact(loaded.get("class"), frozen["loaded_model"]["class"], f"{phase} model class")
    _require(
        Path(loaded.get("name_or_path", "")).resolve() == expected_model_path,
        f"{phase} loaded model path differs",
    )
    _require_exact(
        loaded.get("parameter_count"),
        frozen["loaded_model"]["parameter_count"],
        f"{phase} parameter count",
    )
    _require_exact(loaded.get("parameter_devices"), ["cuda:0"], f"{phase} parameter devices")
    _require_exact(loaded.get("parameter_dtypes"), ["torch.bfloat16"], f"{phase} parameter dtypes")
    _require_exact(loaded.get("loaded_config"), frozen["config"], f"{phase} loaded config identity")
    _require_exact(
        loaded.get("attention_implementation"), "sdpa", f"{phase} attention implementation"
    )


def _validate_execution_policy(report: dict[str, Any], *, phase: str) -> None:
    policy = report.get("attention_policy")
    _require(isinstance(policy, dict), f"{phase} attention policy is missing")
    _require_exact(policy.get("requested"), "cudnn", f"{phase} SDPA backend")
    _require_exact(policy.get("forced"), True, f"{phase} forced SDPA")
    _require_exact(policy.get("resolved_pytorch_enum"), "CUDNN_ATTENTION", f"{phase} SDPA enum")
    availability = policy.get("availability")
    _require(isinstance(availability, dict), f"{phase} SDPA availability is missing")
    _require_exact(
        availability.get("selected_build_or_library_available"),
        True,
        f"{phase} cuDNN availability",
    )
    arithmetic = report.get("arithmetic_policy")
    _require(isinstance(arithmetic, dict), f"{phase} arithmetic policy is missing")
    _require_exact(arithmetic.get("seed"), 8419, f"{phase} seed")
    _require_exact(arithmetic.get("greedy_argmax"), True, f"{phase} greedy policy")
    _require_exact(arithmetic.get("sampling"), False, f"{phase} sampling policy")
    _require_exact(arithmetic.get("eos_early_stopping"), False, f"{phase} EOS policy")
    _require_exact(
        arithmetic.get("requested_deterministic_algorithms"),
        False,
        f"{phase} requested deterministic policy",
    )
    _require_exact(
        arithmetic.get("torch_deterministic_algorithms"),
        False,
        f"{phase} actual deterministic policy",
    )
    environment = report.get("environment")
    _require(isinstance(environment, dict), f"{phase} environment is missing")
    device = environment.get("device")
    _require(isinstance(device, dict), f"{phase} device identity is missing")
    _require_exact(device.get("requested_and_normalized"), "cuda:0", f"{phase} device")
    _require_exact(device.get("type"), "cuda", f"{phase} device type")


def _validate_run(
    run: Any,
    *,
    phase: str,
    expected_length: int,
    expected_repeat: int,
    expected_mode: str,
    prompt_hash: str,
) -> None:
    label = f"{phase} run {expected_length}/{expected_repeat}/{expected_mode}"
    _require(isinstance(run, dict), f"{label} must be an object")
    _require_exact(run.get("prompt_tokens"), expected_length, f"{label} prompt tokens")
    _require_exact(run.get("completion_tokens"), 8, f"{label} completion tokens")
    _require_exact(run.get("repeat"), expected_repeat, f"{label} repeat")
    _require_exact(run.get("prompt_token_ids_sha256"), prompt_hash, f"{label} prompt identity")
    _require_exact(run.get("eos_stopping"), False, f"{label} EOS stopping")
    _require_exact(
        run.get("final_cache_tokens"), expected_length + 7, f"{label} final cache length"
    )
    cache = run.get("cache")
    _require(isinstance(cache, dict), f"{label} cache contract is missing")
    _require_exact(cache.get("mode"), expected_mode, f"{label} cache mode")
    _require_exact(
        cache.get("class"), "transformers.cache_utils.DynamicCache", f"{label} cache class"
    )
    _require_exact(cache.get("offloading"), expected_mode == "offloaded", f"{label} offloading")
    _require_exact(cache.get("offload_only_non_sliding"), False, f"{label} offload policy")
    _require_exact(
        cache.get("payload"),
        "exact model-produced K/V; no quantization or approximation",
        f"{label} cache payload",
    )

    token_ids = run.get("completion_token_ids")
    _require(isinstance(token_ids, list) and len(token_ids) == 8, f"{label} token IDs differ")
    for token_id in token_ids:
        _require(_is_int(token_id) and 0 <= token_id < 151936, f"{label} token ID is invalid")
    _require_exact(
        run.get("completion_token_ids_sha256"),
        _token_id_sha256(token_ids),
        f"{label} completion token hash",
    )

    timing = run.get("timing")
    _require(isinstance(timing, dict), f"{label} timing is missing")
    reported_timing: dict[str, float] = {}
    for field in (
        "prefill_seconds",
        "ttft_seconds",
        "decode_model_step_seconds",
        "all_model_step_seconds",
        "run_wall_seconds",
        "decode_tokens_per_second",
    ):
        reported_timing[field] = _positive_number(
            timing.get(field), field=f"{label}.timing.{field}"
        )
    _require_exact(
        timing.get("ttft_scope"),
        "prefill model call plus finite-logit validation and greedy argmax; excludes prompt host-to-device transfer, cache construction, and pre-run telemetry",
        f"{label} TTFT scope",
    )

    steps = run.get("steps")
    _require(isinstance(steps, list) and len(steps) == 8, f"{label} step count differs")
    for index, step in enumerate(steps):
        _require(isinstance(step, dict), f"{label} step {index} must be an object")
        _require_exact(step.get("completion_index"), index, f"{label} completion index")
        _require_exact(step.get("output_token_id"), token_ids[index], f"{label} step token")
        _positive_number(step.get("seconds"), field=f"{label}.steps[{index}].seconds")
        if index == 0:
            _require_exact(step.get("phase"), "prefill_to_first_token", f"{label} prefill phase")
            _require_exact(
                step.get("model_input_tokens"), expected_length, f"{label} prefill input"
            )
            _require_exact(
                step.get("cache_tokens_after_forward"), expected_length, f"{label} prefill cache"
            )
        else:
            _require_exact(step.get("phase"), "single_token_decode", f"{label} decode phase")
            _require_exact(step.get("model_input_tokens"), 1, f"{label} decode input")
            _require_exact(
                step.get("cache_tokens_after_forward"),
                expected_length + index,
                f"{label} decode cache",
            )

    derived_timing = _derived_run_timings(run)
    for field, expected in derived_timing.items():
        _require(
            reported_timing[field] == expected,
            f"{label}.timing.{field} contradicts the recorded step timings",
        )
    _require(
        reported_timing["run_wall_seconds"] >= derived_timing["all_model_step_seconds"],
        f"{label}.timing.run_wall_seconds is shorter than the recorded model steps",
    )

    observations = run.get("memory_observations")
    _require(isinstance(observations, list) and observations, f"{label} memory evidence is missing")
    for index, observation in enumerate(observations):
        _validate_memory_observation(observation, field=f"{label}.memory[{index}]")


def validate_report(
    contract: dict[str, Any],
    *,
    phase_name: str,
    report_path: Path,
) -> PhaseEvidence:
    """Strictly validate one sealed report and return its independently derived evidence."""

    phase = contract["phases"][phase_name]
    expected_path = _relative_path(phase["report_path"], field=f"{phase_name}.report_path")
    _require(report_path.resolve() == expected_path, f"{phase_name} report path differs")
    report = load_json_strict(report_path)
    _verify_report_integrity(report, phase=phase_name)
    _require_exact(report.get("schema_version"), REPORT_SCHEMA, f"{phase_name} report schema")
    _require_exact(report.get("status"), "completed", f"{phase_name} report status")
    _require_exact(report.get("errors"), [], f"{phase_name} report errors")
    _require_exact(report.get("request"), phase["request"], f"{phase_name} request")
    claim_scope = report.get("claim_scope")
    _require(isinstance(claim_scope, dict), f"{phase_name} claim scope is missing")
    for field, expected in {
        "exact_cache_modes_only": True,
        "snapshot_or_archive": False,
        "nvme_or_disk_backed_decode": False,
        "sparse_or_compressed_cache": False,
        "context_extension_claim": False,
        "position_contract": "no request may exceed model max_position_embeddings",
    }.items():
        _require_exact(claim_scope.get(field), expected, f"{phase_name} claim scope {field}")
    _validate_model_identity(report, contract, phase=phase_name)
    _validate_execution_policy(report, phase=phase_name)

    prompt_contract = contract["identities"]["prompt_token_ids_sha256"]
    expected_lengths = phase["request"]["lengths"]
    prompts = report.get("prompts")
    _require(isinstance(prompts, list), f"{phase_name} prompts must be a list")
    prompt_counts = Counter(item.get("prompt_tokens") for item in prompts if isinstance(item, dict))
    _require_exact(prompt_counts, Counter(expected_lengths), f"{phase_name} prompt matrix")
    for prompt in prompts:
        length = prompt["prompt_tokens"]
        _require_exact(
            prompt.get("token_ids_sha256"),
            prompt_contract[str(length)],
            f"{phase_name} prompt {length} identity",
        )
        _require_exact(
            prompt.get("construction"),
            "repeat tokenizer output without special tokens, then exact truncate",
            f"{phase_name} prompt construction",
        )

    expected_keys = [
        (length, repeat, mode)
        for length in expected_lengths
        for repeat in range(phase["request"]["repeats"])
        for mode in phase["request"]["modes"]
    ]
    attempts = report.get("attempts")
    runs = report.get("runs")
    _require(isinstance(attempts, list), f"{phase_name} attempts must be a list")
    _require(isinstance(runs, list), f"{phase_name} runs must be a list")
    _require_exact(len(attempts), phase["expected_cells"], f"{phase_name} attempt count")
    _require_exact(len(runs), phase["expected_cells"], f"{phase_name} run count")
    attempt_keys: list[tuple[int, int, str]] = []
    for index, attempt in enumerate(attempts):
        _require(isinstance(attempt, dict), f"{phase_name} attempt {index} must be an object")
        key = (attempt.get("prompt_tokens"), attempt.get("repeat"), attempt.get("mode"))
        attempt_keys.append(key)
        _require_exact(attempt.get("execution_ordinal"), index, f"{phase_name} ordinal")
        _require_exact(attempt.get("status"), "completed", f"{phase_name} attempt status")
        _require_exact(attempt.get("run_index"), index, f"{phase_name} run index")
        _require("error" not in attempt, f"{phase_name} completed attempt contains an error")
    _require_exact(attempt_keys, expected_keys, f"{phase_name} ordered attempt matrix")

    run_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for run in runs:
        _require(isinstance(run, dict), f"{phase_name} run must be an object")
        cache = run.get("cache")
        _require(isinstance(cache, dict), f"{phase_name} run cache is missing")
        key = (run.get("prompt_tokens"), run.get("repeat"), cache.get("mode"))
        _require(key not in run_by_key, f"{phase_name} duplicate run cell {key}")
        run_by_key[key] = run
    _require_exact(set(run_by_key), set(expected_keys), f"{phase_name} complete run matrix")
    for length, repeat, mode in expected_keys:
        _validate_run(
            run_by_key[(length, repeat, mode)],
            phase=phase_name,
            expected_length=length,
            expected_repeat=repeat,
            expected_mode=mode,
            prompt_hash=prompt_contract[str(length)],
        )

    for length in expected_lengths:
        identities = {
            run_by_key[key]["completion_token_ids_sha256"]
            for key in expected_keys
            if key[0] == length
        }
        _require(len(identities) == 1, f"{phase_name} run token identities differ at {length}")
    token_identity = report.get("token_identity")
    _require(isinstance(token_identity, dict), f"{phase_name} token identity is missing")
    _require_exact(token_identity.get("failure_count"), 0, f"{phase_name} parity failures")
    _require_exact(
        token_identity.get("all_completed_run_tokens_identical_by_prompt"),
        True,
        f"{phase_name} token parity",
    )
    matrix = token_identity.get("matrix")
    _require(isinstance(matrix, dict), f"{phase_name} identity matrix is missing")
    _require_exact(
        matrix,
        {
            "expected_cells": phase["expected_cells"],
            "actual_records": phase["expected_cells"],
            "complete_and_unique": True,
            "missing": [],
            "duplicates": [],
            "unexpected": [],
        },
        f"{phase_name} token matrix",
    )

    model_memory = report.get("model_load_memory")
    _require(isinstance(model_memory, list), f"{phase_name} model memory is missing")
    after_load = [
        item
        for item in model_memory
        if isinstance(item, dict) and item.get("label") == "after_model_load"
    ]
    _require(len(after_load) == 1, f"{phase_name} requires one after_model_load observation")
    baseline_rss, _baseline_free, _baseline_peak = _validate_memory_observation(
        after_load[0], field=f"{phase_name}.after_model_load"
    )
    observations: list[dict[str, Any]] = [item for item in model_memory if isinstance(item, dict)]
    for run in runs:
        observations.extend(run["memory_observations"])
    memory_values = [
        _validate_memory_observation(item, field=f"{phase_name}.observation[{index}]")
        for index, item in enumerate(observations)
    ]
    minimum_free = min(value[1] for value in memory_values)
    maximum_rss = max(value[0] for value in memory_values)
    incremental_rss = max(0, maximum_rss - baseline_rss)
    return PhaseEvidence(
        name=phase_name,
        path=report_path.resolve(),
        report_sha256=_file_sha256(report_path),
        run_by_key=run_by_key,
        minimum_global_cuda_free_bytes=minimum_free,
        incremental_rss_bytes=incremental_rss,
    )


def _nearest_rank(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "percentile requires at least one observation")
    _require(0 < quantile <= 1, "nearest-rank quantile must be in (0, 1]")
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _gate(name: str, passed: bool, *, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def evaluate_reports(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Validate both frozen reports and derive every acceptance gate without inference."""

    preflight = validate_preflight(contract_path)
    contract = load_json_strict(contract_path.resolve())
    evidence = {
        name: validate_report(
            contract,
            phase_name=name,
            report_path=_relative_path(phase["report_path"], field=f"{name}.report_path"),
        )
        for name, phase in contract["phases"].items()
    }
    parity = evidence["parity"]
    capacity = evidence["capacity"]
    gates = contract["gates"]
    gate_results: list[dict[str, Any]] = []

    minimum_free = min(item.minimum_global_cuda_free_bytes for item in evidence.values())
    gate_results.append(
        _gate(
            "minimum_global_cuda_free_observation",
            minimum_free >= gates["minimum_global_cuda_free_bytes"],
            observed={"bytes": minimum_free, "gib": minimum_free / GIB},
            requirement=">= 1 GiB across every recorded CUDA observation",
        )
    )
    maximum_incremental_rss = max(item.incremental_rss_bytes for item in evidence.values())
    gate_results.append(
        _gate(
            "maximum_incremental_process_rss_over_after_model_load",
            maximum_incremental_rss <= gates["maximum_incremental_process_rss_bytes"],
            observed={
                "bytes": maximum_incremental_rss,
                "gib": maximum_incremental_rss / GIB,
                "per_phase_bytes": {
                    name: item.incremental_rss_bytes for name, item in evidence.items()
                },
            },
            requirement="<= 6 GiB, using max current process RSS minus that report's after_model_load current RSS",
        )
    )

    capacity_runs = [run for key, run in capacity.run_by_key.items() if key[0] == 40952]
    derived_capacity_timings = [_derived_run_timings(run) for run in capacity_runs]
    ttfts = [timing["ttft_seconds"] for timing in derived_capacity_timings]
    decode_rates = [timing["decode_tokens_per_second"] for timing in derived_capacity_timings]
    decode_steps = [
        float(step["seconds"])
        for run in capacity_runs
        for step in run["steps"]
        if step["phase"] == "single_token_decode"
    ]
    capacity_gate = gates["capacity_40952"]
    gate_results.extend(
        [
            _gate(
                "40952_each_ttft",
                all(value <= capacity_gate["maximum_each_ttft_seconds"] for value in ttfts),
                observed={"seconds": ttfts, "maximum_seconds": max(ttfts)},
                requirement="each <= 60 seconds",
            ),
            _gate(
                "40952_median_decode_throughput",
                statistics.median(decode_rates)
                >= capacity_gate["minimum_median_decode_tokens_per_second"],
                observed={
                    "tokens_per_second": decode_rates,
                    "median_tokens_per_second": statistics.median(decode_rates),
                },
                requirement=">= 2 decoded tokens/second",
            ),
            _gate(
                "40952_p95_single_token_decode",
                _nearest_rank(decode_steps, 0.95)
                <= capacity_gate["maximum_nearest_rank_p95_single_token_decode_seconds"],
                observed={
                    "sample_count": len(decode_steps),
                    "nearest_rank_p95_seconds": _nearest_rank(decode_steps, 0.95),
                },
                requirement="nearest-rank p95 <= 1 second over the 21 post-prefill decode steps",
            ),
        ]
    )

    pair_hashes: list[dict[str, Any]] = []
    for repeat in range(3):
        offloaded = parity.run_by_key[(16384, repeat, "offloaded")]
        dynamic = parity.run_by_key[(16384, repeat, "dynamic")]
        pair_hashes.append(
            {
                "repeat": repeat,
                "offloaded": offloaded["completion_token_ids_sha256"],
                "dynamic": dynamic["completion_token_ids_sha256"],
                "identical": offloaded["completion_token_ids_sha256"]
                == dynamic["completion_token_ids_sha256"],
            }
        )
    gate_results.append(
        _gate(
            "16384_paired_offloaded_dynamic_token_identity",
            all(pair["identical"] for pair in pair_hashes),
            observed=pair_hashes,
            requirement="all three same-repeat completion token hashes identical",
        )
    )

    def run_peak(run: dict[str, Any]) -> int:
        return max(
            _validate_memory_observation(item, field="parity.16384.memory")[2]
            for item in run["memory_observations"]
        )

    offloaded_peaks = [
        run_peak(parity.run_by_key[(16384, repeat, "offloaded")]) for repeat in range(3)
    ]
    dynamic_peaks = [run_peak(parity.run_by_key[(16384, repeat, "dynamic")]) for repeat in range(3)]
    dynamic_median = int(statistics.median(dynamic_peaks))
    maximum_offloaded = max(offloaded_peaks)
    cuda_delta = dynamic_median - maximum_offloaded
    required_delta = gates["parity_16384"][
        "minimum_dynamic_median_minus_maximum_offloaded_peak_cuda_allocated_bytes"
    ]
    gate_results.append(
        _gate(
            "16384_offloaded_cuda_allocation_saving",
            cuda_delta >= required_delta,
            observed={
                "offloaded_peak_allocated_bytes": offloaded_peaks,
                "maximum_offloaded_peak_allocated_bytes": maximum_offloaded,
                "dynamic_peak_allocated_bytes": dynamic_peaks,
                "dynamic_median_peak_allocated_bytes": dynamic_median,
                "dynamic_median_minus_maximum_offloaded_bytes": cuda_delta,
                "dynamic_median_minus_maximum_offloaded_gib": cuda_delta / GIB,
            },
            requirement=">= 1.5 GiB; dynamic median peak allocation minus maximum offloaded peak allocation",
        )
    )

    passed = all(result["passed"] for result in gate_results)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "accepted" if passed else "rejected",
        "decision": (
            contract["decision_scope"]["authorized_only_if_all_gates_pass"]
            if passed
            else "not authorized"
        ),
        "decision_scope": contract["decision_scope"],
        "failure_policy": contract["failure_policy"],
        "evidence": {
            "contract": preflight["contract"],
            "runner": preflight["runner"],
            "evaluator": preflight["evaluator"],
            "reports": {
                name: {
                    "path": item.path.relative_to(REPOSITORY).as_posix(),
                    "sha256": item.report_sha256,
                    "cells": len(item.run_by_key),
                }
                for name, item in evidence.items()
            },
            "models_run_by_evaluator": False,
        },
        "gate_results": gate_results,
        "all_gates_passed": passed,
    }


def _write_summary_exclusive(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    payload["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON of this summary with the integrity member omitted",
        "sha256": _canonical_sha256(payload),
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError as error:
        raise AcceptanceError(f"refusing to overwrite existing summary: {path}") from error


def _validation_failure_summary(
    contract: Mapping[str, Any], error: AcceptanceError
) -> dict[str, Any]:
    """Create a fail-closed summary when frozen evidence cannot be structurally evaluated."""

    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "rejected",
        "decision": "not authorized",
        "decision_scope": contract.get("decision_scope"),
        "failure_policy": contract.get("failure_policy"),
        "evidence": {"models_run_by_evaluator": False},
        "evaluation_error": str(error),
        "gate_results": [],
        "all_gates_passed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate frozen contract/source/model artifacts without requiring reports",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.preflight:
            result = validate_preflight(args.contract)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        contract = load_json_strict(args.contract.resolve())
        summary_path = _relative_path(contract["summary"]["path"], field="summary.path")
        try:
            summary = evaluate_reports(args.contract)
        except AcceptanceError as error:
            summary = _validation_failure_summary(contract, error)
        _write_summary_exclusive(summary_path, summary)
    except AcceptanceError as error:
        print(f"Exact-context acceptance rejected: {error}", file=sys.stderr)
        return 1
    print(f"Wrote exclusive acceptance summary to {summary_path}")
    return 0 if summary["all_gates_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
