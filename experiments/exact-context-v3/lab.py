"""Isolated compatibility lab for exact CPU-owned KV with shared CUDA staging.

The public process is a supervisor.  It launches one DynamicCache worker and,
only after that baseline succeeds, one SharedStagingCache worker.  Each worker
loads the frozen local checkpoint independently and writes a content-sealed,
exclusive report.  The supervisor compares token IDs and seals a third report.

This file deliberately has no package entry point and never runs at import.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping
import ctypes
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import weakref

import torch

from molor.experimental.exact_context_lab import (
    GIB,
    ContextRequest,
    _config_identity,
    _canonical_sha256,
    _configure_arithmetic_policy,
    _device_identity,
    _environment_identity,
    _error_record,
    _manual_greedy_decode,
    _model_artifact_identity,
    _nominal_kv_bytes,
    _sdpa_backend_evidence,
    _seal_report,
    _synthetic_prompt_ids,
    _token_id_sha256,
    _tokenizer_identity,
    _torch_dtype,
    _validate_attention_policy,
    _validate_context_request,
    _validate_loaded_config_contract,
    _verify_report_integrity,
)
from molor.experimental.exact_context_v2_lab import (
    observe_after_cache_release,
    postflight_recheck_local_model_artifact,
    process_memory_v2,
    unique_live_cache_storage,
)
from molor.experimental.shared_staging_cache import (
    TELEMETRY_SCHEMA as SHARED_CACHE_TELEMETRY_SCHEMA,
    SharedStagingCache,
)


REPORT_SCHEMA = "molor.exact-context-v3-compatibility-report.v1"
WORKER_SCHEMA = "molor.exact-context-v3-worker-report.v1"
COMMAND_SCHEMA = "molor.exact-context-v3-command-contract.v1"
DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/exact-context-v3")
DEFAULT_LENGTH = 4_096
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_MAX_CACHE_LEN = 40_960
DEFAULT_SEED = 8_419
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WORKER_TIMEOUT_SECONDS = 1_800.0
DEFAULT_MINIMUM_FREE_CUDA_GIB = 5.0
DEFAULT_CUDA_HEADROOM_GIB = 1.0
DEFAULT_HOST_HEADROOM_GIB = 2.0
EXPECTED_TRANSFORMERS_VERSION = "5.15.1"
SHARED_TELEMETRY_EVENT_CAPACITY = 256
WORKER_MODES = ("dynamic", "shared_staging")
MIB = 2**20

# These are inherited from the sealed v1 contract.  A v3 worker must fail before
# inference if the local model/config/tokenizer is no longer that artifact.
FROZEN_V1_CONTRACT = {
    "path": "experiments/exact-context-v1/contract.json",
    "file_sha256": "88c924dd8ca8aa17b2fa4a9cbf4cf80ebfad8642462d7edec481c4611eef3afd",
    "model_content_fingerprint": (
        "772bbbfd94e3a63f48e2f315ea8f4d2318c66290c1d98d624dd9bc679c85bb7c"
    ),
    "model_stack_fingerprint": ("c89442b5bbc335341e634f163f41ba193e0412f9e80676cf0e8a283ad60daa91"),
    "config_canonical_sha256": ("e2709ebe03b4c4e47fc99312b15b4ff9bdfeb2a28512f25ab52081cdbc577065"),
    "tokenizer_vocabulary_sha256": (
        "6e949d8fec10c504f4207a43cdeac7d7961f7fd9894d256f14a8153bb5479b63"
    ),
    "tokenizer_backend_json_sha256": (
        "41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04"
    ),
    "tokenizer_special_tokens_map_sha256": (
        "edcb2fc2acbbe77f858a9c4fe51295ffdb84711efba5703ec5906b3d67282569"
    ),
    "tokenizer_chat_template_sha256": (
        "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
    ),
    "max_position_embeddings": 40_960,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "head_dim": 128,
}

# The exact 4,096-token prompt identity was recorded by the v1 diagnostic using
# the same deterministic synthetic prompt construction imported above.
FROZEN_PROMPT_HASHES = {4_096: "f2aef9fe4bc2057f6089979e7136014a94f0085db2e20773e55e601ea5236741"}


@dataclass(frozen=True, slots=True)
class CompatibilityRequest:
    """One baseline/staged compatibility pair."""

    model: Path
    length: int = DEFAULT_LENGTH
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_cache_len: int = DEFAULT_MAX_CACHE_LEN
    seed: int = DEFAULT_SEED
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    minimum_free_cuda_gib: float = DEFAULT_MINIMUM_FREE_CUDA_GIB
    cuda_headroom_gib: float = DEFAULT_CUDA_HEADROOM_GIB
    host_headroom_gib: float = DEFAULT_HOST_HEADROOM_GIB
    worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS

    def to_report_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = str(self.model)
        payload.update(
            {
                "worker_order": list(WORKER_MODES),
                "workers_share_process": False,
                "attention_implementation": "sdpa",
                "sdpa_backend": "cudnn",
                "deterministic_algorithms": False,
                "greedy": True,
                "local_files_only": True,
                "trust_remote_code": False,
            }
        )
        return payload


def _plain_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_request(request: CompatibilityRequest, *, require_model: bool = True) -> None:
    """Validate without importing Transformers or allocating model/GPU tensors."""

    if not isinstance(request.model, Path):
        raise TypeError("model must be a pathlib.Path")
    if require_model:
        if not request.model.exists():
            raise ValueError("model must be an existing local directory; downloads are forbidden")
        if not request.model.is_dir():
            raise ValueError("model must be a local directory, not a file or Hub identifier")
    _plain_positive_int(request.length, "length")
    _plain_positive_int(request.max_new_tokens, "max_new_tokens")
    _plain_positive_int(request.max_cache_len, "max_cache_len")
    if request.max_new_tokens < 2:
        raise ValueError("max_new_tokens must be at least 2 to exercise decode staging")
    if request.length + request.max_new_tokens > request.max_cache_len:
        raise ValueError("prompt plus completion must fit max_cache_len")
    if request.max_cache_len != FROZEN_V1_CONTRACT["max_position_embeddings"]:
        raise ValueError("max_cache_len must equal the frozen native 40,960-position contract")
    if not isinstance(request.seed, int) or isinstance(request.seed, bool):
        raise ValueError("seed must be an integer")
    if request.seed < 0 or request.seed > torch.iinfo(torch.int64).max:
        raise ValueError("seed must be a non-negative signed 64-bit integer")
    device = torch.device(request.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("device must be an explicit CUDA device such as cuda:0")
    if request.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    for name in (
        "minimum_free_cuda_gib",
        "cuda_headroom_gib",
        "host_headroom_gib",
        "worker_timeout_seconds",
    ):
        value = getattr(request, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if request.worker_timeout_seconds <= 0:
        raise ValueError("worker_timeout_seconds must be positive")


def projected_memory_contract(
    *,
    max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Return the frozen Qwen geometry's exact backing/staging arithmetic."""

    layers = int(FROZEN_V1_CONTRACT["num_hidden_layers"])
    heads = int(FROZEN_V1_CONTRACT["num_key_value_heads"])
    head_dim = int(FROZEN_V1_CONTRACT["head_dim"])
    element_bytes = torch.empty((), dtype=dtype).element_size()
    cpu_backing_bytes = 2 * layers * heads * head_dim * max_cache_len * element_bytes
    staging_bytes = 2 * heads * head_dim * max_cache_len * element_bytes
    return {
        "formula": "2*K/V*layers*kv_heads*head_dim*max_cache_len*dtype_bytes",
        "max_cache_len": max_cache_len,
        "dtype": str(dtype).removeprefix("torch."),
        "dtype_bytes": element_bytes,
        "layers": layers,
        "kv_heads": heads,
        "head_dim": head_dim,
        "cpu_backing_bytes": cpu_backing_bytes,
        "cpu_backing_gib": cpu_backing_bytes / GIB,
        "shared_device_staging_bytes": staging_bytes,
        "shared_device_staging_mib": staging_bytes / MIB,
        "full_context_h2d_bytes_per_generated_token": cpu_backing_bytes,
        "full_context_h2d_gib_per_generated_token": cpu_backing_bytes / GIB,
        "bandwidth_warning": (
            "The shared pair bounds live staging allocation but still transfers the complete "
            "used history, layer by layer, for every decode token."
        ),
    }


def validate_loaded_attention_implementation(model_config: Any) -> dict[str, Any]:
    """Require the model to retain SDPA after from_pretrained resolves its config."""

    observed = getattr(model_config, "_attn_implementation", None)
    evidence = {
        "expected": "sdpa",
        "observed": observed,
        "matches": observed == "sdpa",
        "checked_before_inference": True,
    }
    if observed != "sdpa":
        raise RuntimeError("loaded model did not retain the required SDPA implementation")
    return evidence


def _model_base_content_fingerprint(model_identity: Mapping[str, Any]) -> str | None:
    artifacts = model_identity.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if isinstance(artifact, Mapping) and artifact.get("role") == "base":
            value = artifact.get("content_fingerprint")
            return value if isinstance(value, str) else None
    return None


def validate_frozen_identities(
    *,
    model_identity: Mapping[str, Any],
    config_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact v1 model/config/tokenizer identities."""

    checks = {
        "model_content_fingerprint": (
            _model_base_content_fingerprint(model_identity),
            FROZEN_V1_CONTRACT["model_content_fingerprint"],
        ),
        "model_stack_fingerprint": (
            model_identity.get("stack_fingerprint"),
            FROZEN_V1_CONTRACT["model_stack_fingerprint"],
        ),
        "config_canonical_sha256": (
            config_identity.get("canonical_sha256"),
            FROZEN_V1_CONTRACT["config_canonical_sha256"],
        ),
        "tokenizer_vocabulary_sha256": (
            tokenizer_identity.get("vocabulary_sha256"),
            FROZEN_V1_CONTRACT["tokenizer_vocabulary_sha256"],
        ),
        "tokenizer_backend_json_sha256": (
            tokenizer_identity.get("backend_json_sha256"),
            FROZEN_V1_CONTRACT["tokenizer_backend_json_sha256"],
        ),
        "tokenizer_special_tokens_map_sha256": (
            tokenizer_identity.get("special_tokens_map_sha256"),
            FROZEN_V1_CONTRACT["tokenizer_special_tokens_map_sha256"],
        ),
        "tokenizer_chat_template_sha256": (
            tokenizer_identity.get("chat_template_sha256"),
            FROZEN_V1_CONTRACT["tokenizer_chat_template_sha256"],
        ),
    }
    evidence = {
        name: {"observed": observed, "expected": expected, "matches": observed == expected}
        for name, (observed, expected) in checks.items()
    }
    failed = [name for name, item in evidence.items() if not item["matches"]]
    if failed:
        raise RuntimeError("frozen v1 identity mismatch: " + ", ".join(failed))
    return {
        "source_contract": dict(FROZEN_V1_CONTRACT),
        "all_match": True,
        "checks": evidence,
    }


def _host_memory_snapshot() -> dict[str, Any]:
    """Best-effort host physical-memory evidence without optional packages."""

    if os.name == "nt":
        from ctypes import wintypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        call = ctypes.windll.kernel32.GlobalMemoryStatusEx
        call.argtypes = [ctypes.POINTER(MemoryStatusEx)]
        call.restype = wintypes.BOOL
        if not call(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {
            "available_physical_bytes": int(status.ullAvailPhys),
            "total_physical_bytes": int(status.ullTotalPhys),
            "source": "Windows GlobalMemoryStatusEx",
        }
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="ascii").splitlines():
            key, separator, tail = line.partition(":")
            if separator and tail.strip().endswith(" kB"):
                values[key] = int(tail.strip().split()[0]) * 1_024
        return {
            "available_physical_bytes": values.get("MemAvailable"),
            "total_physical_bytes": values.get("MemTotal"),
            "source": "/proc/meminfo",
        }
    return {
        "available_physical_bytes": None,
        "total_physical_bytes": None,
        "source": "unsupported",
    }


def _nvidia_smi_snapshot(
    device_index: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Best-effort independent NVIDIA global-memory observation."""

    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "error": str(error), "command": command}
    line = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or len(line) != 1:
        return {
            "status": "unavailable",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip()[:2_000],
            "command": command,
        }
    fields = [part.strip() for part in line[0].split(",")]
    try:
        free_mib, total_mib = (int(value) for value in fields)
    except (ValueError, TypeError):
        return {"status": "unavailable", "raw": line[0][:2_000], "command": command}
    return {
        "status": "available",
        "free_reported_mib": free_mib,
        "total_reported_mib": total_mib,
        "free_bytes_assuming_mib": free_mib * MIB,
        "total_bytes_assuming_mib": total_mib * MIB,
        "source": "nvidia-smi global device memory (includes other processes)",
        "command": command,
    }


def _shared_live_storage(cache_telemetry: Mapping[str, Any]) -> dict[str, Any]:
    storage = cache_telemetry.get("storage")
    if not isinstance(storage, Mapping):
        raise RuntimeError("shared cache telemetry has no storage object")
    backing = int(storage["backing_bytes"])
    staging = int(storage["staging_bytes"])
    staging_device = str(storage["staging_device"])
    cuda_bytes = staging if staging_device.startswith("cuda") else 0
    cpu_bytes = backing + (staging if staging_device == "cpu" else 0)
    return {
        "unique_storage_count": None,
        "cpu_bytes": cpu_bytes,
        "cuda_bytes": cuda_bytes,
        "other_device_bytes": 0,
        "total_bytes": cpu_bytes + cuda_bytes,
        "by_device_bytes": {"cpu": cpu_bytes, staging_device: staging}
        if staging
        else {"cpu": cpu_bytes},
        "source": "SharedStagingCache.telemetry exact owned storage bytes",
        "exact_cpu_backing_bytes": backing,
        "exact_shared_staging_bytes": staging,
    }


class WorkerTelemetry:
    """V2 process telemetry plus exact cache placement and global GPU evidence."""

    def __init__(self, device: torch.device, mode: str) -> None:
        self.device = device
        self.mode = mode
        self._cache_reference: weakref.ReferenceType[Any] | None = None

    def bind_cache(self, cache: Any) -> None:
        self._cache_reference = weakref.ref(cache)

    def clear_cache(self) -> None:
        self._cache_reference = None

    def cache_reference(self) -> weakref.ReferenceType[Any] | None:
        return self._cache_reference

    def reset_cuda_peaks(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def observe(self, label: str) -> dict[str, Any]:
        cache = self._cache_reference() if self._cache_reference is not None else None
        if cache is not None and self.mode == "shared_staging":
            live_storage = _shared_live_storage(cache.telemetry())
        else:
            live_storage = unique_live_cache_storage(cache)
            live_storage["source"] = "deduplicated live layer K/V tensor storages"
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        index = self.device.index
        if index is None:  # pragma: no cover - request validation requires an explicit index
            raise RuntimeError("worker CUDA device lost its explicit index")
        return {
            "label": label,
            "process": process_memory_v2(),
            "host": _host_memory_snapshot(),
            "live_cache_storage": live_storage,
            "cuda": {
                "device_global_free_bytes": int(free_bytes),
                "device_global_total_bytes": int(total_bytes),
                "global_source": (
                    "torch.cuda.mem_get_info; device-global and includes other processes"
                ),
                "process_allocator_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                "process_allocator_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                "process_allocator_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.device)
                ),
                "process_allocator_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self.device)
                ),
            },
            "nvidia_smi": _nvidia_smi_snapshot(index),
        }


def summarize_memory(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize independently labelled process, allocator, and global metrics."""

    if not observations:
        raise ValueError("at least one memory observation is required")

    def numeric(path: tuple[str, ...]) -> list[int]:
        values: list[int] = []
        for observation in observations:
            item: Any = observation
            for key in path:
                item = item.get(key) if isinstance(item, Mapping) else None
            if isinstance(item, int) and not isinstance(item, bool):
                values.append(item)
        return values

    def maximum(path: tuple[str, ...]) -> int | None:
        values = numeric(path)
        return max(values) if values else None

    def minimum(path: tuple[str, ...]) -> int | None:
        values = numeric(path)
        return min(values) if values else None

    return {
        "observations": len(observations),
        "max_current_rss_bytes": maximum(("process", "current_rss_bytes")),
        "max_process_peak_rss_bytes": maximum(("process", "peak_rss_bytes")),
        "max_private_committed_bytes": maximum(("process", "private_committed_bytes")),
        "max_torch_cuda_allocated_bytes": maximum(("cuda", "process_allocator_allocated_bytes")),
        "max_torch_cuda_reserved_bytes": maximum(("cuda", "process_allocator_reserved_bytes")),
        "max_torch_cuda_peak_allocated_bytes": maximum(
            ("cuda", "process_allocator_peak_allocated_bytes")
        ),
        "minimum_cuda_device_global_free_bytes": minimum(("cuda", "device_global_free_bytes")),
        "minimum_nvidia_smi_global_free_bytes": minimum(("nvidia_smi", "free_bytes_assuming_mib")),
        "max_exact_cpu_backing_bytes": maximum(("live_cache_storage", "exact_cpu_backing_bytes")),
        "max_exact_shared_staging_bytes": maximum(
            ("live_cache_storage", "exact_shared_staging_bytes")
        ),
        "separation_note": (
            "Process RSS/private commit, PyTorch allocator bytes, and device-global free "
            "memory have different scopes and are not added together."
        ),
    }


def _expected_shared_transfer_contract(request: CompatibilityRequest) -> dict[str, Any]:
    layers = int(FROZEN_V1_CONTRACT["num_hidden_layers"])
    heads = int(FROZEN_V1_CONTRACT["num_key_value_heads"])
    head_dim = int(FROZEN_V1_CONTRACT["head_dim"])
    element_bytes = torch.empty((), dtype=_torch_dtype(request.dtype)).element_size()
    device = str(torch.device(request.device))
    events: list[dict[str, Any]] = []

    def append(
        *,
        layer_idx: int,
        phase: str,
        tensor_role: str,
        source_device: str,
        destination_device: str,
        elements: int,
    ) -> None:
        events.append(
            {
                "ordinal": len(events),
                "layer_idx": layer_idx,
                "phase": phase,
                "tensor_role": tensor_role,
                "source_device": source_device,
                "destination_device": destination_device,
                "direction": f"{source_device}_to_{destination_device}",
                "elements": elements,
                "bytes": elements * element_bytes,
                "non_blocking": True,
            }
        )

    prefill_elements = heads * request.length * head_dim
    for layer_idx in range(layers):
        for role in ("key", "value"):
            append(
                layer_idx=layer_idx,
                phase="prefill_to_backing",
                tensor_role=role,
                source_device=device,
                destination_device="cpu",
                elements=prefill_elements,
            )
    one_token_elements = heads * head_dim
    for decode_index in range(1, request.max_new_tokens):
        used_length = request.length + decode_index
        staging_elements = heads * used_length * head_dim
        for layer_idx in range(layers):
            for role in ("key", "value"):
                append(
                    layer_idx=layer_idx,
                    phase="decode_to_backing",
                    tensor_role=role,
                    source_device=device,
                    destination_device="cpu",
                    elements=one_token_elements,
                )
            for role in ("key", "value"):
                append(
                    layer_idx=layer_idx,
                    phase="backing_to_staging",
                    tensor_role=role,
                    source_device="cpu",
                    destination_device=device,
                    elements=staging_elements,
                )

    by_phase: dict[str, dict[str, int]] = {}
    by_direction: dict[str, dict[str, int]] = {}
    for event in events:
        for totals, key in (
            (by_phase, event["phase"]),
            (by_direction, event["direction"]),
        ):
            bucket = totals.setdefault(key, {"copies": 0, "bytes": 0})
            bucket["copies"] += 1
            bucket["bytes"] += event["bytes"]
    retained = min(len(events), SHARED_TELEMETRY_EVENT_CAPACITY)
    return {
        "copies": len(events),
        "bytes": sum(event["bytes"] for event in events),
        "by_direction": {key: by_direction[key] for key in sorted(by_direction)},
        "by_phase": {key: by_phase[key] for key in sorted(by_phase)},
        "event_capacity": SHARED_TELEMETRY_EVENT_CAPACITY,
        "events_retained": retained,
        "events_dropped": len(events) - retained,
        "events": events[-retained:] if retained else [],
    }


def _require_plain_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"{field} must be a plain integer >= {minimum}")
    return value


def _validate_transfer_integer_fields(transfers: Mapping[str, Any]) -> None:
    for field in ("copies", "bytes", "event_capacity", "events_retained", "events_dropped"):
        _require_plain_int(transfers.get(field), f"transfers.{field}")
    for group_name in ("by_phase", "by_direction"):
        group = transfers.get(group_name)
        if not isinstance(group, Mapping):
            raise RuntimeError(f"transfers.{group_name} must be an object")
        for name, totals in group.items():
            if not isinstance(name, str) or not isinstance(totals, Mapping):
                raise RuntimeError(f"transfers.{group_name} is malformed")
            _require_plain_int(totals.get("copies"), f"{group_name}.{name}.copies")
            _require_plain_int(totals.get("bytes"), f"{group_name}.{name}.bytes")
    events = transfers.get("events")
    if not isinstance(events, list):
        raise RuntimeError("transfers.events must be a list")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise RuntimeError(f"transfers.events[{index}] must be an object")
        for field in ("ordinal", "layer_idx", "elements", "bytes"):
            _require_plain_int(event.get(field), f"transfers.events[{index}].{field}")


def validate_shared_cache_telemetry(
    telemetry: Mapping[str, Any],
    *,
    request: CompatibilityRequest,
) -> dict[str, Any]:
    """Fail closed on cache health, geometry, operations, storage, and ledger evidence."""

    if telemetry.get("schema_version") != SHARED_CACHE_TELEMETRY_SCHEMA:
        raise RuntimeError("unexpected shared-cache telemetry schema")
    projected = projected_memory_contract(
        max_cache_len=request.max_cache_len,
        dtype=_torch_dtype(request.dtype),
    )
    layers = int(FROZEN_V1_CONTRACT["num_hidden_layers"])
    heads = int(FROZEN_V1_CONTRACT["num_key_value_heads"])
    head_dim = int(FROZEN_V1_CONTRACT["head_dim"])
    device = str(torch.device(request.device))
    final_length = request.length + request.max_new_tokens - 1
    expected_health = {
        "poisoned": False,
        "active_poison_reason": None,
        "poison_events": 0,
        "poison_events_by_reason": {},
        "cuda_stream_contract": "one_stable_current_stream",
        "cuda_stream_bound": True,
    }
    if telemetry.get("health") != expected_health:
        raise RuntimeError("shared cache health is not pristine and unpoisoned")
    expected_geometry = {
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
    }
    if telemetry.get("geometry") != expected_geometry:
        raise RuntimeError("shared cache geometry/device differs from the frozen worker request")
    expected_sequence = {
        "layer_logical_lengths": [final_length] * layers,
        "expected_next_layer_idx": 0,
        "update_cycle_in_progress": False,
    }
    if telemetry.get("sequence") != expected_sequence:
        raise RuntimeError("shared cache sequence/cycle state is incomplete")
    expected_operations = {
        "updates_by_layer": [request.max_new_tokens] * layers,
        "prefills_by_layer": [1] * layers,
        "decodes_by_layer": [request.max_new_tokens - 1] * layers,
        "resets": 0,
    }
    if telemetry.get("operations") != expected_operations:
        raise RuntimeError("shared cache per-layer operation counts are not exact")

    role_bytes = heads * request.max_cache_len * head_dim * int(projected["dtype_bytes"])
    layer_bytes = role_bytes * 2
    expected_storage = {
        "backing_device": "cpu",
        "backing_bytes": int(projected["cpu_backing_bytes"]),
        "layer_backing": [
            {
                "layer_idx": layer_idx,
                "initialized": True,
                "logical_length": final_length,
                "key_bytes": role_bytes,
                "value_bytes": role_bytes,
                "total_bytes": layer_bytes,
            }
            for layer_idx in range(layers)
        ],
        "staging_device": device,
        "staging_allocated": True,
        "staging_key_bytes": role_bytes,
        "staging_value_bytes": role_bytes,
        "staging_bytes": int(projected["shared_device_staging_bytes"]),
        "staging_pair_allocations": 1,
        "live_cache_bytes": int(projected["cpu_backing_bytes"])
        + int(projected["shared_device_staging_bytes"]),
    }
    if telemetry.get("storage") != expected_storage:
        raise RuntimeError("shared cache owned storage differs from exact geometry")

    transfers = telemetry.get("transfers")
    if not isinstance(transfers, Mapping):
        raise RuntimeError("shared cache transfer telemetry is missing")
    _validate_transfer_integer_fields(transfers)
    expected_transfers = _expected_shared_transfer_contract(request)
    if transfers != expected_transfers:
        raise RuntimeError("shared cache bounded transfer ledger is not internally exact")
    if sum(item["copies"] for item in transfers["by_phase"].values()) != transfers["copies"]:
        raise RuntimeError("transfer phase copy totals do not reconcile")
    if sum(item["bytes"] for item in transfers["by_phase"].values()) != transfers["bytes"]:
        raise RuntimeError("transfer phase byte totals do not reconcile")
    if sum(item["copies"] for item in transfers["by_direction"].values()) != transfers["copies"]:
        raise RuntimeError("transfer direction copy totals do not reconcile")
    if sum(item["bytes"] for item in transfers["by_direction"].values()) != transfers["bytes"]:
        raise RuntimeError("transfer direction byte totals do not reconcile")
    ordinals = [event["ordinal"] for event in transfers["events"]]
    expected_ordinals = list(range(transfers["events_dropped"], transfers["copies"]))
    if ordinals != expected_ordinals:
        raise RuntimeError("retained transfer events are not the contiguous ledger tail")
    return {
        "all_checks_passed": True,
        "expected_final_cache_tokens": final_length,
        "decode_steps_that_append_to_cache": request.max_new_tokens - 1,
        "expected_total_copies": expected_transfers["copies"],
        "expected_total_transfer_bytes": expected_transfers["bytes"],
        "event_capacity": SHARED_TELEMETRY_EVENT_CAPACITY,
        "events_retained": expected_transfers["events_retained"],
        "events_dropped": expected_transfers["events_dropped"],
        "full_history_h2d_is_still_present": True,
    }


def _worker_report_path(final_output: Path, mode: str) -> Path:
    if mode not in WORKER_MODES:
        raise ValueError("unknown worker mode")
    return final_output.with_name(f"{final_output.stem}.{mode}.worker.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_artifacts() -> dict[str, Any]:
    """Snapshot every imported implementation artifact into one canonical bundle."""

    repo_root = Path(__file__).resolve().parents[2]
    paths = {
        "lab": Path(__file__).resolve(),
        "shared_staging_cache": (
            repo_root / "src" / "molor" / "experimental" / "shared_staging_cache.py"
        ),
        "exact_context_lab": (
            repo_root / "src" / "molor" / "experimental" / "exact_context_lab.py"
        ),
        "exact_context_v2_lab": (
            repo_root / "src" / "molor" / "experimental" / "exact_context_v2_lab.py"
        ),
        "v1_contract": repo_root / str(FROZEN_V1_CONTRACT["path"]),
    }
    artifacts: dict[str, Any] = {}
    for role, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"required {role} artifact is missing: {path}")
        artifacts[role] = {
            "path": path.relative_to(repo_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    if artifacts["v1_contract"]["sha256"] != FROZEN_V1_CONTRACT["file_sha256"]:
        raise RuntimeError("inherited v1 contract bytes differ from the pinned contract hash")
    core = {
        "schema_version": "molor.exact-context-v3-implementation-bundle.v1",
        "artifacts": artifacts,
    }
    return {**core, "bundle_sha256": _canonical_sha256(core)}


def _validate_implementation_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bundle's shape and independently recompute its canonical hash."""

    if bundle.get("schema_version") != "molor.exact-context-v3-implementation-bundle.v1":
        raise RuntimeError("implementation bundle has the wrong schema")
    artifacts = bundle.get("artifacts")
    expected_roles = {
        "lab",
        "shared_staging_cache",
        "exact_context_lab",
        "exact_context_v2_lab",
        "v1_contract",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_roles:
        raise RuntimeError("implementation bundle has the wrong artifact roles")
    normalized: dict[str, Any] = {}
    for role in sorted(expected_roles):
        item = artifacts.get(role)
        if not isinstance(item, Mapping) or set(item) != {"path", "size_bytes", "sha256"}:
            raise RuntimeError(f"implementation artifact {role} is malformed")
        path = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise RuntimeError(f"implementation artifact {role} path is invalid")
        if type(size) is not int or size < 1:
            raise RuntimeError(f"implementation artifact {role} size is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"implementation artifact {role} SHA-256 is invalid")
        normalized[role] = {"path": path, "size_bytes": size, "sha256": digest}
    core = {
        "schema_version": "molor.exact-context-v3-implementation-bundle.v1",
        "artifacts": normalized,
    }
    recomputed = _canonical_sha256(core)
    if bundle.get("bundle_sha256") != recomputed:
        raise RuntimeError("implementation bundle canonical hash is invalid")
    return {**core, "bundle_sha256": recomputed}


def _assert_implementation_bundle(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    expected_validated = _validate_implementation_bundle(expected)
    observed_validated = _validate_implementation_bundle(observed)
    if observed_validated != expected_validated:
        changed = [
            role
            for role in sorted(expected_validated["artifacts"])
            if observed_validated["artifacts"].get(role)
            != expected_validated["artifacts"].get(role)
        ]
        raise RuntimeError(
            "implementation artifacts changed from supervisor pin: " + ", ".join(changed)
        )
    return observed_validated


def _encode_implementation_bundle(bundle: Mapping[str, Any]) -> str:
    validated = _validate_implementation_bundle(bundle)
    encoded = json.dumps(
        validated,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii")


def _decode_pinned_implementation_bundle(args: argparse.Namespace) -> dict[str, Any]:
    encoded = args._expected_implementation_snapshot
    expected_hash = args._expected_implementation_bundle_sha256
    if not isinstance(encoded, str) or not isinstance(expected_hash, str):
        raise RuntimeError("worker requires a supervisor-pinned implementation snapshot and hash")
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), altchars=b"-_", validate=True)
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("worker implementation snapshot encoding is invalid") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("worker implementation snapshot must decode to an object")
    validated = _validate_implementation_bundle(payload)
    if validated["bundle_sha256"] != expected_hash:
        raise RuntimeError("worker implementation snapshot and pinned hash disagree")
    return validated


def _write_json_atomic_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically while refusing any existing destination."""

    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "nt":
            # Windows rename is same-volume atomic and refuses an existing destination.
            os.rename(temporary, path)
        else:
            # A hard-link publication is atomic and, unlike POSIX rename, never replaces.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _bounded_text(value: str | bytes | None, *, limit: int = 8_192) -> dict[str, Any]:
    if value is None:
        value = ""
    elif isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) <= limit:
        return {"text": value, "truncated": False, "original_characters": len(value)}
    return {
        "text": value[-limit:],
        "truncated": True,
        "retained": "tail",
        "original_characters": len(value),
    }


def _request_from_args(args: argparse.Namespace) -> CompatibilityRequest:
    model = Path(args.model).expanduser()
    if model.exists():
        model = model.resolve()
    return CompatibilityRequest(
        model=model,
        length=args.length,
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        minimum_free_cuda_gib=args.minimum_free_cuda_gib,
        cuda_headroom_gib=args.cuda_headroom_gib,
        host_headroom_gib=args.host_headroom_gib,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )


def _build_worker_argv(
    request: CompatibilityRequest,
    *,
    mode: str,
    worker_output: Path,
    implementation_bundle: Mapping[str, Any],
    python_executable: str = sys.executable,
    script_path: Path | None = None,
) -> list[str]:
    if mode not in WORKER_MODES:
        raise ValueError("unknown worker mode")
    if script_path is None:
        script_path = Path(__file__).resolve()
    pinned_bundle = _validate_implementation_bundle(implementation_bundle)
    return [
        python_executable,
        str(script_path),
        "--model",
        str(request.model),
        "--length",
        str(request.length),
        "--max-new-tokens",
        str(request.max_new_tokens),
        "--max-cache-len",
        str(request.max_cache_len),
        "--seed",
        str(request.seed),
        "--device",
        request.device,
        "--dtype",
        request.dtype,
        "--minimum-free-cuda-gib",
        str(request.minimum_free_cuda_gib),
        "--cuda-headroom-gib",
        str(request.cuda_headroom_gib),
        "--host-headroom-gib",
        str(request.host_headroom_gib),
        "--worker-timeout-seconds",
        str(request.worker_timeout_seconds),
        "--_worker-mode",
        mode,
        "--_worker-output",
        str(worker_output),
        "--_expected-implementation-bundle-sha256",
        pinned_bundle["bundle_sha256"],
        "--_expected-implementation-snapshot",
        _encode_implementation_bundle(pinned_bundle),
    ]


def command_contract(
    request: CompatibilityRequest,
    *,
    final_output: Path,
    python_executable: str = sys.executable,
    script_path: Path | None = None,
) -> dict[str, Any]:
    """Return the exact two-process plan without loading model files."""

    validate_request(request, require_model=False)
    implementation_bundle = implementation_artifacts()
    commands = []
    for mode in WORKER_MODES:
        worker_output = _worker_report_path(final_output, mode)
        commands.append(
            {
                "mode": mode,
                "worker_output": str(worker_output),
                "argv": _build_worker_argv(
                    request,
                    mode=mode,
                    worker_output=worker_output,
                    implementation_bundle=implementation_bundle,
                    python_executable=python_executable,
                    script_path=script_path,
                ),
            }
        )
    return {
        "schema_version": COMMAND_SCHEMA,
        "status": "development_unfrozen",
        "runs_model": False,
        "one_cache_mode_per_process": True,
        "order": list(WORKER_MODES),
        "shared_worker_runs_only_after_baseline_success": True,
        "final_output": str(final_output),
        "request": request.to_report_dict(),
        "implementation_bundle": implementation_bundle,
        "commands": commands,
    }


def _make_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    directory = args.output_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return directory / f"compatibility-{stamp}-{os.getpid()}-{time.time_ns()}.json"


def _validate_output_set_absent(final_output: Path) -> dict[str, Path]:
    paths = {mode: _worker_report_path(final_output, mode) for mode in WORKER_MODES}
    collisions = [path for path in (final_output, *paths.values()) if path.exists()]
    if collisions:
        rendered = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"refusing existing output path(s) before workers start: {rendered}")
    return paths


def _build_cache(mode: str, model_config: Any, dtype: torch.dtype, device: torch.device) -> Any:
    if mode == "dynamic":
        from transformers import DynamicCache

        return DynamicCache(config=model_config, offloading=False)
    if mode != "shared_staging":
        raise ValueError("unknown worker mode")
    values = {
        "num_layers": getattr(model_config, "num_hidden_layers", None),
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", None),
        "head_dim": getattr(model_config, "head_dim", None),
    }
    if values != {
        "num_layers": FROZEN_V1_CONTRACT["num_hidden_layers"],
        "num_key_value_heads": FROZEN_V1_CONTRACT["num_key_value_heads"],
        "head_dim": FROZEN_V1_CONTRACT["head_dim"],
    }:
        raise RuntimeError("loaded model geometry differs from the frozen shared-cache geometry")
    return SharedStagingCache(
        num_layers=int(values["num_layers"]),
        max_cache_len=DEFAULT_MAX_CACHE_LEN,
        num_key_value_heads=int(values["num_key_value_heads"]),
        key_head_dim=int(values["head_dim"]),
        dtype=dtype,
        staging_device=device,
        pin_cpu_backing=True,
        non_blocking_copies=True,
        telemetry_event_capacity=SHARED_TELEMETRY_EVENT_CAPACITY,
    )


def _run_worker_body(
    request: CompatibilityRequest,
    *,
    mode: str,
    report: dict[str, Any],
) -> None:
    """Execute one cache mode.  Called only inside an isolated child process."""

    validate_request(request)
    if mode not in WORKER_MODES:
        raise ValueError("unknown worker mode")
    _validate_attention_policy(
        attention_implementation="sdpa",
        sdpa_backend="cudnn",
        deterministic_algorithms=False,
        device_type="cuda",
    )
    report["attention_policy"] = _sdpa_backend_evidence("cudnn", device_type="cuda")
    if report["attention_policy"]["availability"]["selected_build_or_library_available"] is False:
        raise RuntimeError("forced cuDNN SDPA is unavailable")

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("install molor[hf] for the exact-context v3 lab") from error
    report["dependency_contract"] = {
        "transformers_expected": EXPECTED_TRANSFORMERS_VERSION,
        "transformers_observed": transformers.__version__,
        "matches": transformers.__version__ == EXPECTED_TRANSFORMERS_VERSION,
    }
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "SharedStagingCache targets Transformers " + EXPECTED_TRANSFORMERS_VERSION
        )

    load_kwargs = {"local_files_only": True, "trust_remote_code": False}
    model_id = str(request.model)
    config = AutoConfig.from_pretrained(model_id, **load_kwargs)
    context_request = ContextRequest(
        modes=("dynamic",),
        lengths=(request.length,),
        max_new_tokens=request.max_new_tokens,
        repeats=1,
    )
    _validate_context_request(
        context_request,
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
    )
    if request.max_cache_len != getattr(config, "max_position_embeddings", None):
        raise RuntimeError(
            "requested backing capacity differs from the loaded native position limit"
        )
    config_identity = _config_identity(config)
    identity_started = time.perf_counter()
    model_identity = _model_artifact_identity(model_id, is_local=True, revision=None)
    report["identity_seconds"] = time.perf_counter() - identity_started
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    tokenizer_identity = _tokenizer_identity(tokenizer)
    report["identities"] = {
        "config": config_identity,
        "model_artifact": model_identity,
        "tokenizer": tokenizer_identity,
    }
    report["frozen_v1_identity_validation"] = validate_frozen_identities(
        model_identity=model_identity,
        config_identity=config_identity,
        tokenizer_identity=tokenizer_identity,
    )

    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise RuntimeError("model config has no valid vocab_size")
    prompt_ids = _synthetic_prompt_ids(tokenizer, request.length)
    if int(prompt_ids.min().item()) < 0 or int(prompt_ids.max().item()) >= vocab_size:
        raise RuntimeError("synthetic prompt token is outside the frozen vocabulary")
    prompt_hash = _token_id_sha256(prompt_ids)
    frozen_prompt_hash = FROZEN_PROMPT_HASHES.get(request.length)
    if frozen_prompt_hash is not None and prompt_hash != frozen_prompt_hash:
        raise RuntimeError("default synthetic prompt identity differs from its v1 observation")
    report["prompt"] = {
        "tokens": request.length,
        "token_ids_sha256": prompt_hash,
        "frozen_observed_sha256": frozen_prompt_hash,
        "matches_frozen_observation": (
            prompt_hash == frozen_prompt_hash if frozen_prompt_hash is not None else None
        ),
        "construction": (
            "v1 deterministic synthetic text; repeat tokenizer output without special tokens; "
            "exact truncate"
        ),
        "token_id_min": int(prompt_ids.min().item()),
        "token_id_max": int(prompt_ids.max().item()),
    }

    requested_device = torch.device(request.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    assert requested_device.index is not None
    torch.cuda.set_device(requested_device.index)
    device = torch.device("cuda", requested_device.index)
    if request.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested CUDA device does not support bfloat16")
    dtype = _torch_dtype(request.dtype)
    report["environment"] = _environment_identity(transformers, DynamicCache)
    report["environment"]["device"] = _device_identity(device)
    report["environment"]["worker_pid"] = os.getpid()
    report["arithmetic_policy"] = _configure_arithmetic_policy(
        request.seed,
        deterministic_algorithms=False,
    )
    report["projected_memory_contract"] = projected_memory_contract(
        max_cache_len=request.max_cache_len,
        dtype=dtype,
    )

    telemetry = WorkerTelemetry(device, mode)
    before_load = telemetry.observe("before_model_load")
    report["model_load_memory"] = [before_load]
    if before_load["cuda"]["device_global_free_bytes"] < int(request.minimum_free_cuda_gib * GIB):
        raise RuntimeError("CUDA global free memory is below the pre-load guard")
    if mode == "shared_staging":
        available_host = before_load["host"].get("available_physical_bytes")
        required_host = int(report["projected_memory_contract"]["cpu_backing_bytes"]) + int(
            request.host_headroom_gib * GIB
        )
        report["host_memory_guard"] = {
            "available_before_model_load_bytes": available_host,
            "required_cpu_backing_plus_headroom_bytes": required_host,
            "enforced_when_available_is_known": True,
        }
        if isinstance(available_host, int) and available_host < required_host:
            raise RuntimeError("host available memory is below CPU backing plus headroom guard")

    model = None
    cache_holder: dict[str, Any | None] = {"cache": None}
    try:
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation="sdpa",
            device_map={"": device},
            **load_kwargs,
        )
        model.eval()
        telemetry.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_started
        after_load = telemetry.observe("after_model_load")
        report["model_load_memory"].append(after_load)
        report["loaded_attention_contract"] = validate_loaded_attention_implementation(model.config)
        loaded_contract = _validate_loaded_config_contract(config, model.config, context_request)
        report["loaded_context_contract"] = loaded_contract
        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if parameter_devices != [str(device)]:
            raise RuntimeError("model parameters must stay on one exact CUDA device")
        report["identities"]["loaded_model"] = {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_devices": parameter_devices,
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "loaded_config": _config_identity(model.config),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
        }

        nominal_dynamic = _nominal_kv_bytes(
            model.config,
            sequence_tokens=request.length + request.max_new_tokens,
            dtype=dtype,
        )
        projected = report["projected_memory_contract"]
        required_cache_cuda = (
            nominal_dynamic if mode == "dynamic" else int(projected["shared_device_staging_bytes"])
        )
        if required_cache_cuda is None:
            raise RuntimeError("could not calculate the exact CUDA cache guard")
        required_after_load = int(request.cuda_headroom_gib * GIB) + int(required_cache_cuda)
        report["cuda_memory_guard"] = {
            "global_free_after_model_load_bytes": after_load["cuda"]["device_global_free_bytes"],
            "required_cache_bytes": required_cache_cuda,
            "additional_headroom_bytes": int(request.cuda_headroom_gib * GIB),
            "required_total_free_bytes": required_after_load,
        }
        if after_load["cuda"]["device_global_free_bytes"] < required_after_load:
            raise RuntimeError("CUDA global free memory is below cache plus headroom guard")

        cache_holder["cache"] = _build_cache(mode, model.config, dtype, device)
        telemetry.bind_cache(cache_holder["cache"])
        if mode == "shared_staging":
            before_backing = telemetry.observe("immediately_before_shared_backing_allocation")
            report["pre_shared_backing_memory"] = before_backing
            available_host = before_backing["host"].get("available_physical_bytes")
            required_host = int(projected["cpu_backing_bytes"]) + int(
                request.host_headroom_gib * GIB
            )
            report["post_model_load_host_memory_guard"] = {
                "observation_label": before_backing["label"],
                "available_physical_bytes": available_host,
                "required_cpu_backing_plus_headroom_bytes": required_host,
                "passed": not isinstance(available_host, int) or available_host >= required_host,
                "enforced_when_available_is_known": True,
            }
            if isinstance(available_host, int) and available_host < required_host:
                raise RuntimeError(
                    "post-model-load host memory is below CPU backing plus headroom guard"
                )
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
                sdpa_backend="cudnn",
            )
            if mode == "shared_staging":
                cache_telemetry = cache_holder["cache"].telemetry()
                report["shared_cache_telemetry"] = cache_telemetry
                report["shared_cache_telemetry_validation"] = validate_shared_cache_telemetry(
                    cache_telemetry,
                    request=request,
                )
        finally:

            def release() -> None:
                cache_holder["cache"] = None

            release_observation = observe_after_cache_release(
                telemetry,  # type: ignore[arg-type]
                release=release,
            )
            report["cache_release_memory"] = release_observation
        if result is None:  # pragma: no cover - original error is re-raised
            raise RuntimeError("worker produced no decode result")
        if not release_observation["cache_release"]["release_succeeded"]:
            raise RuntimeError("cache remained reachable after its release boundary")
        result["mode"] = mode
        result["prompt_token_ids_sha256"] = prompt_hash
        report["run"] = result
        observations = [*report["model_load_memory"], *result["memory_observations"]]
        if "pre_shared_backing_memory" in report:
            observations.append(report["pre_shared_backing_memory"])
        observations.append(release_observation)
        report["memory_summary"] = summarize_memory(observations)
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
                is_local=True,
                active_error=sys.exception() or active_error,
            )


def _initial_worker_report(request: CompatibilityRequest, mode: str) -> dict[str, Any]:
    return {
        "schema_version": WORKER_SCHEMA,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "development_only": True,
        "formal_acceptance_claim": False,
        "mode": mode,
        "request": request.to_report_dict(),
        "claim_scope": {
            "exact_unquantized_kv": True,
            "native_positions_only": True,
            "context_extension_claim": False,
            "long_context_quality_claim": False,
            "compressed_or_sparse": False,
        },
        "errors": [],
    }


def _worker_main(
    args: argparse.Namespace,
    request: CompatibilityRequest,
    *,
    body: Callable[..., None] = _run_worker_body,
    bundle_reader: Callable[[], dict[str, Any]] = implementation_artifacts,
) -> int:
    mode = args._worker_mode
    output = args._worker_output
    if mode not in WORKER_MODES or output is None:
        print("internal worker mode requires an exclusive worker output", file=sys.stderr)
        return 2
    output = output.expanduser().resolve()
    if output.exists():
        print(f"Refusing existing worker output before model load: {output}", file=sys.stderr)
        return 1
    report = _initial_worker_report(request, mode)
    expected_bundle: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        expected_bundle = _decode_pinned_implementation_bundle(args)
        report["implementation_provenance"] = {"expected": expected_bundle}
        preflight = bundle_reader()
        report["implementation_provenance"]["preflight_observed"] = preflight
        _assert_implementation_bundle(expected_bundle, preflight)
        body(request, mode=mode, report=report)
    except (Exception, KeyboardInterrupt) as error:
        failure = error
    finally:
        if expected_bundle is not None:
            try:
                postflight = bundle_reader()
                report.setdefault("implementation_provenance", {})["postflight_observed"] = (
                    postflight
                )
                _assert_implementation_bundle(expected_bundle, postflight)
            except (Exception, KeyboardInterrupt) as error:
                report.setdefault("implementation_provenance", {})[
                    "postflight_validation_error"
                ] = _error_record(error)
                if failure is None:
                    failure = error
                else:
                    report["errors"].append(_error_record(error))
    if failure is None:
        report["status"] = "completed"
        exit_code = 0
    else:
        report["status"] = "failed"
        report["errors"].insert(0, _error_record(failure))
        exit_code = 1
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        _write_json_atomic_exclusive(output, _seal_report(report))
    except FileExistsError:
        print(f"Refusing to overwrite worker report: {output}", file=sys.stderr)
        return 1
    print(f"Wrote exact-context v3 {mode} worker report to {output}")
    return exit_code


def _read_worker_report(
    path: Path,
    *,
    expected_mode: str,
    expected_implementation_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or report.get("schema_version") != WORKER_SCHEMA:
        raise RuntimeError(f"{expected_mode} worker report has the wrong schema")
    if report.get("mode") != expected_mode:
        raise RuntimeError(f"{expected_mode} worker report has the wrong mode")
    if not _verify_report_integrity(report):
        raise RuntimeError(f"{expected_mode} worker report failed its integrity seal")
    provenance = report.get("implementation_provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(f"{expected_mode} worker report has no implementation provenance")
    for phase in ("expected", "preflight_observed", "postflight_observed"):
        bundle = provenance.get(phase)
        if not isinstance(bundle, Mapping):
            raise RuntimeError(f"{expected_mode} worker report is missing {phase} provenance")
        _assert_implementation_bundle(expected_implementation_bundle, bundle)
    return report


def compare_worker_reports(
    dynamic: Mapping[str, Any],
    shared: Mapping[str, Any],
    *,
    expected_request: CompatibilityRequest,
    expected_implementation_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate each worker and compare exact token identity."""

    if dynamic.get("status") != "completed" or shared.get("status") != "completed":
        raise RuntimeError("both workers must complete before parity comparison")
    if dynamic.get("mode") != "dynamic" or shared.get("mode") != "shared_staging":
        raise RuntimeError("worker modes are not the expected baseline/staged pair")
    expected_request_payload = expected_request.to_report_dict()
    for label, worker in (("dynamic", dynamic), ("shared_staging", shared)):
        if worker.get("request") != expected_request_payload:
            raise RuntimeError(f"{label} worker request differs from the supervisor request")
        provenance = worker.get("implementation_provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError(f"{label} worker has no implementation provenance")
        for phase in ("expected", "preflight_observed", "postflight_observed"):
            observed = provenance.get(phase)
            if not isinstance(observed, Mapping):
                raise RuntimeError(f"{label} worker provenance is missing {phase}")
            _assert_implementation_bundle(expected_implementation_bundle, observed)
    dynamic_prompt = dynamic.get("prompt")
    shared_prompt = shared.get("prompt")
    dynamic_run = dynamic.get("run")
    shared_run = shared.get("run")
    if not all(
        isinstance(value, Mapping)
        for value in (dynamic_prompt, shared_prompt, dynamic_run, shared_run)
    ):
        raise RuntimeError("completed worker report is missing prompt or run evidence")
    assert isinstance(dynamic_prompt, Mapping)
    assert isinstance(shared_prompt, Mapping)
    assert isinstance(dynamic_run, Mapping)
    assert isinstance(shared_run, Mapping)

    for label, worker, prompt in (
        ("dynamic", dynamic, dynamic_prompt),
        ("shared_staging", shared, shared_prompt),
    ):
        identities = worker.get("identities")
        if not isinstance(identities, Mapping) or not all(
            isinstance(identities.get(name), Mapping)
            for name in ("model_artifact", "config", "tokenizer")
        ):
            raise RuntimeError(f"{label} worker identities are incomplete")
        validate_frozen_identities(
            model_identity=identities["model_artifact"],
            config_identity=identities["config"],
            tokenizer_identity=identities["tokenizer"],
        )
        frozen_prompt_hash = FROZEN_PROMPT_HASHES.get(expected_request.length)
        if frozen_prompt_hash is not None and prompt.get("token_ids_sha256") != frozen_prompt_hash:
            raise RuntimeError(f"{label} prompt differs from the frozen v1 observation")

    identity_fields: dict[str, tuple[Any, Any]] = {
        "prompt_token_ids_sha256": (
            dynamic_prompt.get("token_ids_sha256"),
            shared_prompt.get("token_ids_sha256"),
        ),
        "model_stack_fingerprint": (
            dynamic["identities"]["model_artifact"].get("stack_fingerprint"),
            shared["identities"]["model_artifact"].get("stack_fingerprint"),
        ),
        "config_canonical_sha256": (
            dynamic["identities"]["config"].get("canonical_sha256"),
            shared["identities"]["config"].get("canonical_sha256"),
        ),
        "tokenizer_vocabulary_sha256": (
            dynamic["identities"]["tokenizer"].get("vocabulary_sha256"),
            shared["identities"]["tokenizer"].get("vocabulary_sha256"),
        ),
        "implementation_bundle_sha256": (
            dynamic["implementation_provenance"]["postflight_observed"].get("bundle_sha256"),
            shared["implementation_provenance"]["postflight_observed"].get("bundle_sha256"),
        ),
    }
    identity_matches = {
        name: {"dynamic": left, "shared_staging": right, "identical": left == right}
        for name, (left, right) in identity_fields.items()
    }
    mismatched_identities = [
        name for name, item in identity_matches.items() if not item["identical"]
    ]
    validated_runs: dict[str, dict[str, Any]] = {}
    for label, prompt, run in (
        ("dynamic", dynamic_prompt, dynamic_run),
        ("shared_staging", shared_prompt, shared_run),
    ):
        ids = run.get("completion_token_ids")
        if not isinstance(ids, list):
            raise RuntimeError(f"{label} worker completion token IDs are missing")
        if len(ids) != expected_request.max_new_tokens:
            raise RuntimeError(f"{label} worker completion token count is not exact")
        for index, token_id in enumerate(ids):
            if type(token_id) is not int:
                raise RuntimeError(f"{label} completion token {index} is not a plain integer")
            if token_id < 0 or token_id > torch.iinfo(torch.int64).max:
                raise RuntimeError(f"{label} completion token {index} is outside int64 range")
        recomputed_hash = _token_id_sha256(ids)
        reported_hash = run.get("completion_token_ids_sha256")
        if reported_hash != recomputed_hash:
            raise RuntimeError(f"{label} completion hash does not match its token IDs")
        if (
            type(run.get("completion_tokens")) is not int
            or run.get("completion_tokens") != expected_request.max_new_tokens
        ):
            raise RuntimeError(f"{label} run completion_tokens is invalid")
        if (
            type(run.get("prompt_tokens")) is not int
            or run.get("prompt_tokens") != expected_request.length
        ):
            raise RuntimeError(f"{label} run prompt_tokens is invalid")
        expected_final = expected_request.length + expected_request.max_new_tokens - 1
        if (
            type(run.get("final_cache_tokens")) is not int
            or run.get("final_cache_tokens") != expected_final
        ):
            raise RuntimeError(f"{label} final cache length is invalid")
        if run.get("mode") != label:
            raise RuntimeError(f"{label} run mode is invalid")
        prompt_hash = prompt.get("token_ids_sha256")
        if not isinstance(prompt_hash, str) or run.get("prompt_token_ids_sha256") != prompt_hash:
            raise RuntimeError(f"{label} run/prompt token hashes are inconsistent")
        if type(prompt.get("tokens")) is not int or prompt.get("tokens") != expected_request.length:
            raise RuntimeError(f"{label} prompt token count is invalid")
        validated_runs[label] = {
            "ids": ids,
            "reported_hash": reported_hash,
            "recomputed_hash": recomputed_hash,
            "prompt_hash": prompt_hash,
            "final_cache_tokens": expected_final,
        }
    dynamic_ids = validated_runs["dynamic"]["ids"]
    shared_ids = validated_runs["shared_staging"]["ids"]
    first_mismatch: int | None = None
    for index, (left, right) in enumerate(zip(dynamic_ids, shared_ids, strict=False)):
        if left != right:
            first_mismatch = index
            break
    if first_mismatch is None and len(dynamic_ids) != len(shared_ids):
        first_mismatch = min(len(dynamic_ids), len(shared_ids))
    tokens_identical = first_mismatch is None
    completion_hashes_identical = (
        validated_runs["dynamic"]["reported_hash"]
        == validated_runs["shared_staging"]["reported_hash"]
    )
    if tokens_identical != completion_hashes_identical:
        raise RuntimeError("completion token equality and independently verified hashes disagree")
    return {
        "identity_checks": identity_matches,
        "all_input_identities_identical": not mismatched_identities,
        "mismatched_input_identities": mismatched_identities,
        "completion": {
            "dynamic_token_ids": dynamic_ids,
            "shared_staging_token_ids": shared_ids,
            "dynamic_sha256": validated_runs["dynamic"]["reported_hash"],
            "shared_staging_sha256": validated_runs["shared_staging"]["reported_hash"],
            "dynamic_recomputed_sha256": validated_runs["dynamic"]["recomputed_hash"],
            "shared_staging_recomputed_sha256": validated_runs["shared_staging"]["recomputed_hash"],
            "token_ids_identical": tokens_identical,
            "hashes_identical": completion_hashes_identical,
            "first_mismatch_index": first_mismatch,
        },
        "exact_parity_passed": not mismatched_identities and tokens_identical,
    }


def _worker_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    run = report.get("run")
    timing = run.get("timing") if isinstance(run, Mapping) else None
    return {
        "status": report.get("status"),
        "mode": report.get("mode"),
        "prompt": report.get("prompt"),
        "completion_token_ids": run.get("completion_token_ids")
        if isinstance(run, Mapping)
        else None,
        "completion_token_ids_sha256": run.get("completion_token_ids_sha256")
        if isinstance(run, Mapping)
        else None,
        "timing": timing,
        "memory_summary": report.get("memory_summary"),
        "projected_memory_contract": report.get("projected_memory_contract"),
        "shared_cache_storage": (
            report.get("shared_cache_telemetry", {}).get("storage")
            if isinstance(report.get("shared_cache_telemetry"), Mapping)
            else None
        ),
        "shared_cache_transfers": (
            report.get("shared_cache_telemetry", {}).get("transfers")
            if isinstance(report.get("shared_cache_telemetry"), Mapping)
            else None
        ),
        "shared_cache_telemetry_validation": report.get("shared_cache_telemetry_validation"),
        "cache_release_memory": report.get("cache_release_memory"),
        "errors": report.get("errors"),
    }


def _supervisor_main(
    args: argparse.Namespace,
    request: CompatibilityRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    bundle_reader: Callable[[], dict[str, Any]] = implementation_artifacts,
) -> int:
    try:
        validate_request(request)
        output = _make_output_path(args)
        worker_paths = _validate_output_set_absent(output)
        pinned_bundle = _validate_implementation_bundle(bundle_reader())
    except (Exception, KeyboardInterrupt) as error:
        print(str(error), file=sys.stderr)
        return 1
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "development_only": True,
        "formal_acceptance_claim": False,
        "request": request.to_report_dict(),
        "implementation_bundle": pinned_bundle,
        "projected_memory_contract": projected_memory_contract(
            max_cache_len=request.max_cache_len,
            dtype=_torch_dtype(request.dtype),
        ),
        "architecture": {
            "supervisor_pid": os.getpid(),
            "one_cache_mode_per_child_process": True,
            "order": list(WORKER_MODES),
            "shared_worker_requires_successful_dynamic_baseline": True,
            "shell_invocation": False,
            "worker_timeout_seconds": request.worker_timeout_seconds,
        },
        "worker_artifacts": [],
        "comparison": None,
        "errors": [],
    }
    reports: dict[str, dict[str, Any]] = {}
    successful_modes: set[str] = set()
    repo_root = Path(__file__).resolve().parents[2]
    exit_code = 0
    for mode in WORKER_MODES:
        if mode == "shared_staging" and "dynamic" not in successful_modes:
            report["errors"].append(
                {
                    "type": "baseline_gate",
                    "message": "shared staging worker was not started because baseline failed",
                }
            )
            exit_code = 1
            break
        path = worker_paths[mode]
        argv = _build_worker_argv(
            request,
            mode=mode,
            worker_output=path,
            implementation_bundle=pinned_bundle,
        )
        invocation: dict[str, Any] = {
            "mode": mode,
            "path": str(path),
            "argv": argv,
            "started": True,
        }
        try:
            completed = runner(
                argv,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=request.worker_timeout_seconds,
            )
            invocation.update(
                {
                    "returncode": completed.returncode,
                    "stdout": _bounded_text(completed.stdout),
                    "stderr": _bounded_text(completed.stderr),
                }
            )
        except subprocess.TimeoutExpired as error:
            invocation.update(
                {
                    "timed_out": True,
                    "timeout_seconds": request.worker_timeout_seconds,
                    "stdout": _bounded_text(error.stdout),
                    "stderr": _bounded_text(error.stderr),
                }
            )
            report["worker_artifacts"].append(invocation)
            report["errors"].append(_error_record(error))
            exit_code = 1
            if mode == "dynamic":
                continue
            break
        except (Exception, KeyboardInterrupt) as error:
            invocation["launch_error"] = _error_record(error)
            report["worker_artifacts"].append(invocation)
            report["errors"].append(_error_record(error))
            exit_code = 1
            if mode == "dynamic":
                continue
            break

        if path.exists():
            invocation["file_sha256"] = _file_sha256(path)
            try:
                worker_report = _read_worker_report(
                    path,
                    expected_mode=mode,
                    expected_implementation_bundle=pinned_bundle,
                )
                reports[mode] = worker_report
                invocation["report_integrity_sha256"] = worker_report["integrity"]["sha256"]
                invocation["report_summary"] = _worker_summary(worker_report)
            except (Exception, KeyboardInterrupt) as error:
                invocation["report_read_error"] = _error_record(error)
                report["errors"].append(_error_record(error))
                exit_code = 1
        else:
            invocation["report_missing"] = True
            report["errors"].append(
                {"type": "missing_worker_report", "message": f"{mode} produced no report"}
            )
            exit_code = 1
        report["worker_artifacts"].append(invocation)
        if invocation.get("returncode") != 0 or reports.get(mode, {}).get("status") != "completed":
            exit_code = 1
            if mode == "dynamic":
                continue
            break
        successful_modes.add(mode)

    if set(reports) == set(WORKER_MODES):
        try:
            report["comparison"] = compare_worker_reports(
                reports["dynamic"],
                reports["shared_staging"],
                expected_request=request,
                expected_implementation_bundle=pinned_bundle,
            )
            if not report["comparison"]["exact_parity_passed"]:
                report["errors"].append(
                    {
                        "type": "exact_parity_failure",
                        "message": "shared staging completion differs from DynamicCache baseline",
                    }
                )
                exit_code = 1
        except (Exception, KeyboardInterrupt) as error:
            report["errors"].append(_error_record(error))
            exit_code = 1
    else:
        exit_code = 1

    try:
        supervisor_postflight = bundle_reader()
        report["implementation_postflight"] = supervisor_postflight
        _assert_implementation_bundle(pinned_bundle, supervisor_postflight)
    except (Exception, KeyboardInterrupt) as error:
        report["implementation_postflight_error"] = _error_record(error)
        report["errors"].append(_error_record(error))
        exit_code = 1

    report["status"] = "completed" if exit_code == 0 else "failed"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        _write_json_atomic_exclusive(output, _seal_report(report))
    except FileExistsError:
        print(f"Refusing to overwrite aggregate report: {output}", file=sys.stderr)
        return 1
    print(f"Wrote exact-context v3 compatibility report to {output}")
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only 4K/8-token compatibility pair in isolated subprocesses: "
            "GPU DynamicCache baseline, then exact CPU-owned KV with one shared CUDA staging pair."
        ),
        epilog=(
            "cuDNN SDPA and production-like nondeterministic kernels are fixed. This is a systems "
            "compatibility test at native positions, not a context-extension or quality claim."
        ),
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="existing local model directory; Hub IDs and downloads are rejected",
    )
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=DEFAULT_MAX_CACHE_LEN,
        help="must remain the frozen native capacity of 40960",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE, help="explicit CUDA device, e.g. cuda:0"
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_DTYPE,
    )
    parser.add_argument(
        "--minimum-free-cuda-gib",
        type=float,
        default=DEFAULT_MINIMUM_FREE_CUDA_GIB,
        help="device-global pre-load free-memory guard",
    )
    parser.add_argument(
        "--cuda-headroom-gib",
        type=float,
        default=DEFAULT_CUDA_HEADROOM_GIB,
        help="post-load reserve in addition to nominal cache/staging allocation",
    )
    parser.add_argument(
        "--host-headroom-gib",
        type=float,
        default=DEFAULT_HOST_HEADROOM_GIB,
        help="shared worker host-memory reserve beyond the exact CPU backing",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        help="per-child timeout; termination, if needed, is scoped to that child",
    )
    parser.add_argument("--output", type=Path, help="exclusive aggregate JSON path")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for a unique aggregate name when --output is omitted",
    )
    parser.add_argument(
        "--emit-command-contract",
        action="store_true",
        help="print the two-worker plan without loading a model or touching CUDA",
    )
    parser.add_argument("--_worker-mode", choices=WORKER_MODES, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--_expected-implementation-bundle-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_expected-implementation-snapshot",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = _request_from_args(args)
    if args.emit_command_contract:
        if any(
            value is not None
            for value in (
                args._worker_mode,
                args._worker_output,
                args._expected_implementation_bundle_sha256,
                args._expected_implementation_snapshot,
            )
        ):
            print("command-contract mode cannot also be an internal worker", file=sys.stderr)
            return 2
        try:
            final_output = (
                args.output.expanduser().resolve()
                if args.output is not None
                else (args.output_directory / "<unique-compatibility-report>.json")
            )
            payload = command_contract(
                request,
                final_output=final_output,
                script_path=Path(__file__).resolve(),
            )
        except (Exception, KeyboardInterrupt) as error:
            print(str(error), file=sys.stderr)
            return 1
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    worker_values = (
        args._worker_mode,
        args._worker_output,
        args._expected_implementation_bundle_sha256,
        args._expected_implementation_snapshot,
    )
    if any(value is not None for value in worker_values) and not all(
        value is not None for value in worker_values
    ):
        print("all internal worker provenance arguments must be supplied together", file=sys.stderr)
        return 2
    if args._worker_mode is not None:
        return _worker_main(args, request)
    return _supervisor_main(args, request)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
