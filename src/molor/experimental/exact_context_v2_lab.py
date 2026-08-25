"""Development worker for exact, preallocated, CPU-offloaded StaticCache trials.

This module is deliberately not a formal acceptance experiment.  It reuses the
content-sealing, identity, prompt, decode, and arithmetic helpers from the frozen v1
lab while changing only the cache implementation and memory diagnostics.  One process
may execute exactly one cell so CPU allocator state cannot leak across trial reports.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import ctypes
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import weakref

import torch
from torch import Tensor

from molor.experimental.exact_context_lab import (
    GIB,
    SDPA_BACKENDS,
    ContextRequest,
    _config_identity,
    _configure_arithmetic_policy,
    _device_identity,
    _environment_identity,
    _error_record,
    _finite_float_or_text,
    _local_or_hub_model,
    _manual_greedy_decode,
    _model_artifact_identity,
    _nominal_kv_bytes,
    _process_memory as _v1_process_memory,
    _sdpa_backend_evidence,
    _seal_report,
    _synthetic_prompt_ids,
    _token_id_sha256,
    _tokenizer_identity,
    _torch_dtype,
    _validate_attention_policy,
    _validate_context_request,
    _validate_loaded_config_contract,
    _write_json_exclusive,
)


SCHEMA_VERSION = "molor.exact-context-v2-development-report.v1"
COMMAND_CONTRACT_SCHEMA = "molor.exact-context-v2-development-command-contract.v1"
DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/exact-context-v2")
DEFAULT_LENGTH = 40_952
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_MAX_CACHE_LEN = 40_960
DEFAULT_TRIALS = 3


@dataclass(frozen=True)
class StaticWorkerRequest:
    """The single cache cell permitted in one worker process."""

    length: int
    max_new_tokens: int
    max_cache_len: int
    trial_id: int
    cache_implementation: str = "static"
    cache_placement: str = "offloaded"
    repeats: int = 1

    def context_request(self) -> ContextRequest:
        return ContextRequest(
            modes=(self.cache_placement,),
            lengths=(self.length,),
            max_new_tokens=self.max_new_tokens,
            repeats=self.repeats,
        )


def validate_single_worker_request(
    request: StaticWorkerRequest,
    *,
    max_position_embeddings: int,
) -> None:
    """Reject any request that is not one exact static/offloaded cache cell."""

    if request.cache_implementation != "static":
        raise ValueError("v2 worker cache_implementation must be static")
    if request.cache_placement != "offloaded":
        raise ValueError("v2 worker cache_placement must be offloaded")
    if request.repeats != 1:
        raise ValueError("v2 worker must execute exactly one repeat")
    if (
        not isinstance(request.trial_id, int)
        or isinstance(request.trial_id, bool)
        or request.trial_id < 0
    ):
        raise ValueError("trial_id must be a non-negative integer")
    if (
        not isinstance(request.max_cache_len, int)
        or isinstance(request.max_cache_len, bool)
        or request.max_cache_len < 1
    ):
        raise ValueError("max_cache_len must be a positive integer")
    if request.max_cache_len != max_position_embeddings:
        raise ValueError("max_cache_len must equal the loaded native position limit")
    _validate_context_request(
        request.context_request(),
        max_position_embeddings=max_position_embeddings,
    )
    if request.length + request.max_new_tokens > request.max_cache_len:
        raise ValueError("prompt plus completion exceeds the static cache capacity")


def build_lazy_offloaded_static_cache(
    config: Any,
    *,
    max_cache_len: int,
    static_cache_class: type[Any] | None = None,
) -> Any:
    """Construct an exact StaticCache without eagerly allocating its backing tensors."""

    if static_cache_class is None:
        from transformers import StaticCache

        static_cache_class = StaticCache
    cache = static_cache_class(
        config=config,
        max_cache_len=max_cache_len,
        offloading=True,
        offload_only_non_sliding=False,
    )
    if getattr(cache, "offloading", None) is not True:
        raise RuntimeError("StaticCache did not enable offloading")
    if getattr(cache, "only_non_sliding", None) is not False:
        raise RuntimeError("StaticCache did not enable full-layer offloading")
    get_max_length = getattr(cache, "get_max_length", None)
    if not callable(get_max_length) or get_max_length() != max_cache_len:
        raise RuntimeError("StaticCache did not retain the requested maximum length")
    if getattr(cache, "is_initialized", False) is True:
        raise RuntimeError("StaticCache unexpectedly allocated storage during construction")
    return cache


StorageRecord = tuple[str, int, int]


def _tensor_storage_record(tensor: Any) -> StorageRecord | None:
    if not isinstance(tensor, Tensor):
        return None
    storage = tensor.untyped_storage()
    size_bytes = int(storage.nbytes())
    if size_bytes == 0:
        return None
    return str(tensor.device), int(storage.data_ptr()), size_bytes


def summarize_unique_storage_records(
    records: Iterable[StorageRecord],
) -> dict[str, Any]:
    """Deduplicate storage records and split their bytes by current device."""

    unique = set(records)
    by_device: dict[str, int] = {}
    for device, _pointer, size_bytes in unique:
        if size_bytes < 0:
            raise ValueError("storage size must be non-negative")
        by_device[device] = by_device.get(device, 0) + size_bytes
    cpu_bytes = sum(size for device, size in by_device.items() if device == "cpu")
    cuda_bytes = sum(size for device, size in by_device.items() if device.startswith("cuda"))
    total_bytes = sum(by_device.values())
    return {
        "unique_storage_count": len(unique),
        "cpu_bytes": cpu_bytes,
        "cuda_bytes": cuda_bytes,
        "other_device_bytes": total_bytes - cpu_bytes - cuda_bytes,
        "total_bytes": total_bytes,
        "by_device_bytes": dict(sorted(by_device.items())),
    }


def unique_live_cache_storage(cache: Any | None) -> dict[str, Any]:
    """Measure unique live K/V backing storage without counting shared views twice."""

    records: list[StorageRecord] = []
    if cache is not None:
        layers = getattr(cache, "layers", ())
        for layer in layers:
            for name in ("keys", "values"):
                record = _tensor_storage_record(getattr(layer, name, None))
                if record is not None:
                    records.append(record)
    result = summarize_unique_storage_records(records)
    result["roles_scanned"] = ["keys", "values"]
    return result


def _windows_counter_payload(counters: Any) -> dict[str, Any]:
    """Convert PROCESS_MEMORY_COUNTERS_EX values into the report schema."""

    return {
        "current_rss_bytes": int(counters.WorkingSetSize),
        "peak_rss_bytes": int(counters.PeakWorkingSetSize),
        "private_committed_bytes": int(counters.PrivateUsage),
        "pagefile_usage_bytes": int(counters.PagefileUsage),
        "peak_pagefile_usage_bytes": int(counters.PeakPagefileUsage),
        "source": "Windows GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX",
    }


def _windows_process_memory() -> dict[str, Any]:
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    succeeded = get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    if not succeeded:
        raise OSError("GetProcessMemoryInfo failed")
    return _windows_counter_payload(counters)


def process_memory_v2() -> dict[str, Any]:
    """Return RSS plus private committed bytes where Windows exposes them."""

    if os.name == "nt":
        return _windows_process_memory()
    payload = dict(_v1_process_memory())
    payload["private_committed_bytes"] = None
    payload["pagefile_usage_bytes"] = None
    payload["peak_pagefile_usage_bytes"] = None
    return payload


class CacheAwareRuntimeTelemetry:
    """Process/CUDA telemetry augmented with weakly referenced live cache storage."""

    def __init__(
        self,
        device: torch.device,
        *,
        process_memory_reader: Callable[[], dict[str, Any]] = process_memory_v2,
    ) -> None:
        self.device = device
        self._process_memory_reader = process_memory_reader
        self._cache_reference: weakref.ReferenceType[Any] | None = None

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def bind_cache(self, cache: Any) -> None:
        self._cache_reference = weakref.ref(cache)

    def clear_cache(self) -> None:
        self._cache_reference = None

    def cache_reference(self) -> weakref.ReferenceType[Any] | None:
        """Return the weak reference itself without retaining the cache object."""

        return self._cache_reference

    def reset_cuda_peaks(self) -> None:
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def observe(self, label: str) -> dict[str, Any]:
        cache = self._cache_reference() if self._cache_reference is not None else None
        observation: dict[str, Any] = {
            "label": label,
            "process": self._process_memory_reader(),
            "live_cache_storage": unique_live_cache_storage(cache),
            "cuda": None,
        }
        if self.is_cuda:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            observation["cuda"] = {
                "device_global_free_bytes": int(free_bytes),
                "device_global_total_bytes": int(total_bytes),
                "process_allocator_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "process_allocator_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                "process_allocator_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.device)
                ),
                "process_allocator_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self.device)
                ),
            }
        return observation


def observe_after_cache_release(
    telemetry: CacheAwareRuntimeTelemetry,
    *,
    release: Callable[[], None],
    collect: Callable[[], Any] = gc.collect,
    empty_cuda_cache: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Release the intended owner and prove whether the cache was actually collected."""

    cache_reference = telemetry.cache_reference()
    if cache_reference is None:
        raise RuntimeError("no cache is bound to release telemetry")
    telemetry.synchronize()
    release()
    collect()
    if empty_cuda_cache is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        empty_cuda_cache()
    telemetry.synchronize()
    cache_object_collected = cache_reference() is None
    try:
        observation = telemetry.observe("after_cache_release")
        storage_zero = observation["live_cache_storage"]["total_bytes"] == 0
        observation["cache_release"] = {
            "cache_object_collected": cache_object_collected,
            "live_cache_storage_zero": storage_zero,
            "release_succeeded": cache_object_collected and storage_zero,
        }
        return observation
    finally:
        # Keep the weak reference bound through the honest storage observation.  Clear it
        # only afterward so a surviving reference cannot manufacture a zero-byte result.
        telemetry.clear_cache()


def postflight_recheck_local_model_artifact(
    report: dict[str, Any],
    *,
    model_id: str,
    is_local: bool,
    active_error: BaseException | None,
    identity_reader: Callable[..., dict[str, Any]] = _model_artifact_identity,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Recheck local model bytes even while another worker failure is propagating."""

    if not is_local:
        return
    started = clock()
    original = report["identities"]["model_artifact"].get("stack_fingerprint")
    try:
        rechecked = identity_reader(model_id, is_local=True, revision=None)
    except (Exception, KeyboardInterrupt) as error:
        report["identities"]["model_artifact_recheck"] = {
            "status": "failed",
            "seconds": clock() - started,
            "matches_preload_identity": False,
            "error": _error_record(error),
        }
        if active_error is None:
            raise
        return

    observed = rechecked.get("stack_fingerprint")
    matches = observed == original
    report["identities"]["model_artifact_recheck"] = {
        "status": "completed",
        "seconds": clock() - started,
        "stack_fingerprint": observed,
        "matches_preload_identity": matches,
    }
    if not matches and active_error is None:
        raise RuntimeError("local model identity changed during the worker")


def single_worker_argv(
    trial_id: int,
    *,
    length: int = DEFAULT_LENGTH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
) -> list[str]:
    """Return one exact, directly executable development-worker command."""

    output = (
        DEFAULT_OUTPUT_DIRECTORY
        / f"static-offloaded-{length}-plus-{max_new_tokens}-trial-{trial_id}.json"
    )
    return [
        r".\.venv\Scripts\python.exe",
        "-m",
        "molor.experimental.exact_context_v2_lab",
        "--model",
        DEFAULT_MODEL.as_posix(),
        "--cache-implementation",
        "static",
        "--cache-placement",
        "offloaded",
        "--length",
        str(length),
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-cache-len",
        str(max_cache_len),
        "--trial-id",
        str(trial_id),
        "--seed",
        "8419",
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "sdpa",
        "--sdpa-backend",
        "cudnn",
        "--no-deterministic-algorithms",
        "--minimum-free-cuda-gib",
        "5.0",
        "--cuda-headroom-gib",
        "1.0",
        "--output",
        output.as_posix(),
    ]


def development_command_contract(*, trials: int = DEFAULT_TRIALS) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be positive")
    return {
        "schema_version": COMMAND_CONTRACT_SCHEMA,
        "status": "development_unfrozen",
        "execution": "run commands sequentially as separate OS processes",
        "one_cell_per_process": True,
        "commands": [
            {"trial_id": trial_id, "argv": single_worker_argv(trial_id)}
            for trial_id in range(trials)
        ],
        "formal_acceptance_claim": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--revision")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-implementation", choices=("static",), default="static")
    parser.add_argument("--cache-placement", choices=("offloaded",), default="offloaded")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-cache-len", type=int, default=DEFAULT_MAX_CACHE_LEN)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=8419)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--sdpa-backend", choices=SDPA_BACKENDS, default="cudnn")
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--minimum-free-cuda-gib", type=float, default=5.0)
    parser.add_argument("--cuda-headroom-gib", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--emit-command-contract",
        action="store_true",
        help="print three exact worker commands without loading a model",
    )
    return parser


def _request_from_args(args: argparse.Namespace) -> StaticWorkerRequest:
    return StaticWorkerRequest(
        length=args.length,
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        trial_id=args.trial_id,
        cache_implementation=args.cache_implementation,
        cache_placement=args.cache_placement,
    )


def _output_path(args: argparse.Namespace) -> Path:
    path = args.output
    if path is None:
        path = (
            DEFAULT_OUTPUT_DIRECTORY / f"static-offloaded-{args.length}-plus-{args.max_new_tokens}"
            f"-trial-{args.trial_id}.json"
        )
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_worker(args: argparse.Namespace, report: dict[str, Any]) -> None:
    request = _request_from_args(args)
    if args.seed < 0 or args.seed > torch.iinfo(torch.int64).max:
        raise ValueError("seed must be a non-negative signed 64-bit integer")
    if (
        not math.isfinite(args.minimum_free_cuda_gib)
        or not math.isfinite(args.cuda_headroom_gib)
        or args.minimum_free_cuda_gib < 0
        or args.cuda_headroom_gib < 0
    ):
        raise ValueError("CUDA memory guards must be finite and non-negative")

    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise ValueError("the static offloaded worker requires a CUDA model device")
    _validate_attention_policy(
        attention_implementation=args.attn_implementation,
        sdpa_backend=args.sdpa_backend,
        deterministic_algorithms=args.deterministic_algorithms,
        device_type=requested_device.type,
    )
    report["attention_policy"] = _sdpa_backend_evidence(
        args.sdpa_backend, device_type=requested_device.type
    )
    if report["attention_policy"]["availability"]["selected_build_or_library_available"] is False:
        raise RuntimeError(f"forced {args.sdpa_backend} SDPA is unavailable")

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StaticCache
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("install molor[hf] for the exact context v2 lab") from error

    model_id, is_local = _local_or_hub_model(args.model)
    if not is_local and not args.allow_download:
        raise ValueError("a remote model requires --allow-download")
    revision_kwargs = {"revision": args.revision} if args.revision else {}
    load_kwargs = {
        "local_files_only": not args.allow_download,
        "trust_remote_code": False,
        **revision_kwargs,
    }
    config = AutoConfig.from_pretrained(model_id, **load_kwargs)
    max_positions = getattr(config, "max_position_embeddings", None)
    validate_single_worker_request(request, max_position_embeddings=max_positions)
    report["request"]["validated_max_position_embeddings"] = max_positions
    report["identities"] = {"config": _config_identity(config)}

    identity_started = time.perf_counter()
    report["identities"]["model_artifact"] = _model_artifact_identity(
        model_id, is_local=is_local, revision=args.revision
    )
    report["identity_seconds"] = time.perf_counter() - identity_started
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    report["identities"]["tokenizer"] = _tokenizer_identity(tokenizer)
    prompt_ids = _synthetic_prompt_ids(tokenizer, request.length)
    report["prompt"] = {
        "prompt_tokens": request.length,
        "token_ids_sha256": _token_id_sha256(prompt_ids),
        "construction": "repeat tokenizer output without special tokens, then exact truncate",
    }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    index = (
        requested_device.index
        if requested_device.index is not None
        else torch.cuda.current_device()
    )
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested CUDA device does not support bfloat16")
    environment = _environment_identity(transformers, StaticCache)
    environment["static_cache_class"] = environment.pop("dynamic_cache_class")
    environment["static_cache_signature"] = environment.pop("dynamic_cache_signature")
    environment["device"] = _device_identity(device)
    report["environment"] = environment
    report["arithmetic_policy"] = _configure_arithmetic_policy(
        args.seed,
        deterministic_algorithms=args.deterministic_algorithms,
    )

    telemetry = CacheAwareRuntimeTelemetry(device)
    before_load = telemetry.observe("before_model_load")
    report["model_load_memory"] = [before_load]
    free_before_load = before_load["cuda"]["device_global_free_bytes"]
    required_before_load = int(args.minimum_free_cuda_gib * GIB)
    if free_before_load < required_before_load:
        raise RuntimeError("CUDA free memory is below the pre-load guard")

    dtype = _torch_dtype(args.dtype)
    model = None
    cache_holder: dict[str, Any | None] = {"cache": None}
    try:
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map={"": device},
            **load_kwargs,
        )
        model.eval()
        telemetry.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_started
        after_load = telemetry.observe("after_model_load")
        report["model_load_memory"].append(after_load)
        loaded_context = _validate_loaded_config_contract(
            config,
            model.config,
            request.context_request(),
        )
        report["request"]["validated_loaded_max_position_embeddings"] = loaded_context[
            "max_position_embeddings"
        ]
        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if parameter_devices != [str(device)]:
            raise RuntimeError("model must remain on one exact CUDA device")
        report["identities"]["loaded_model"] = {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "name_or_path": getattr(model, "name_or_path", None),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_devices": parameter_devices,
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "loaded_config": _config_identity(model.config),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
        }
        nominal_static_bytes = _nominal_kv_bytes(
            model.config,
            sequence_tokens=request.max_cache_len,
            dtype=dtype,
        )
        report["memory_contract"] = {
            "nominal_static_kv_backing_bytes": nominal_static_bytes,
            "nominal_formula": "2*K/V*layers*kv_heads*head_dim*max_cache_len*dtype_bytes",
            "post_load_cuda_headroom_bytes": int(args.cuda_headroom_gib * GIB),
            "cpu_rss_is_not_replaced_by_nominal_storage": True,
        }
        if after_load["cuda"]["device_global_free_bytes"] < int(args.cuda_headroom_gib * GIB):
            raise RuntimeError("CUDA free memory is below the post-load headroom guard")

        cache_holder["cache"] = build_lazy_offloaded_static_cache(
            model.config,
            max_cache_len=request.max_cache_len,
            static_cache_class=StaticCache,
        )
        telemetry.bind_cache(cache_holder["cache"])
        cache = cache_holder["cache"]
        cache_contract = {
            "implementation": "static",
            "placement": "offloaded",
            "class": f"{type(cache).__module__}.{type(cache).__qualname__}",
            "offloading": getattr(cache, "offloading", None),
            "offload_only_non_sliding": getattr(cache, "only_non_sliding", None),
            "max_cache_len": cache.get_max_length(),
            "lazy_initialization": True,
            "payload": "exact model-produced K/V; preallocated but unquantized",
        }
        del cache
        result: dict[str, Any] | None = None
        release_observation: dict[str, Any] | None = None
        try:
            result = _manual_greedy_decode(
                model,
                prompt_ids=prompt_ids,
                cache=cache_holder["cache"],
                max_new_tokens=request.max_new_tokens,
                device=device,
                telemetry=telemetry,
                sdpa_backend=args.sdpa_backend,
            )
        finally:

            def release() -> None:
                cache_holder["cache"] = None

            release_observation = observe_after_cache_release(
                telemetry,
                release=release,
            )
            report["cache_release_memory_observations"].append(release_observation)
        if result is None:  # pragma: no cover - the original exception is re-raised
            raise RuntimeError("static cache worker produced no result")
        if not release_observation["cache_release"]["release_succeeded"]:
            raise RuntimeError("StaticCache remained reachable after the intended release boundary")
        result.update(
            {
                "trial_id": request.trial_id,
                "prompt_token_ids_sha256": _token_id_sha256(prompt_ids),
                "cache": cache_contract,
                "post_cache_release_memory_observation": release_observation,
            }
        )
        report["runs"].append(result)
    finally:
        active_error = sys.exception()
        try:
            cache_holder["cache"] = None
            telemetry.clear_cache()
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            postflight_recheck_local_model_artifact(
                report,
                model_id=model_id,
                is_local=is_local,
                active_error=sys.exception() or active_error,
            )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.emit_command_contract:
        print(json.dumps(development_command_contract(), indent=2, allow_nan=False))
        return 0

    output = _output_path(args)
    if output.exists():
        print(f"Refusing existing output before model load: {output}", file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "development_only": True,
        "formal_acceptance_claim": False,
        "claim_scope": {
            "exact_static_cache": True,
            "cpu_offloaded": True,
            "preallocated": True,
            "compressed_or_quantized": False,
            "context_extension_claim": False,
            "allocator_trim_performed": False,
        },
        "request": {
            "model": args.model,
            "revision": args.revision,
            "allow_download": args.allow_download,
            "cache_implementation": args.cache_implementation,
            "cache_placement": args.cache_placement,
            "length": args.length,
            "max_new_tokens": args.max_new_tokens,
            "max_cache_len": args.max_cache_len,
            "trial_id": args.trial_id,
            "repeats": 1,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "sdpa_backend": args.sdpa_backend,
            "deterministic_algorithms": args.deterministic_algorithms,
            "minimum_free_cuda_gib": _finite_float_or_text(args.minimum_free_cuda_gib),
            "cuda_headroom_gib": _finite_float_or_text(args.cuda_headroom_gib),
        },
        "runs": [],
        "cache_release_memory_observations": [],
        "errors": [],
    }
    exit_code = 0
    try:
        _run_worker(args, report)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["errors"].append(_error_record(error))
        exit_code = 1
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sealed = _seal_report(report)
    try:
        _write_json_exclusive(output, sealed)
    except FileExistsError:
        print(f"Refusing to overwrite existing report: {output}", file=sys.stderr)
        return 1
    print(f"Wrote exact-context v2 development worker report to {output}")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
