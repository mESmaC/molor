"""Dry-run-first validation of the Qwen3-0.6B paged CPU KV backing layout.

The default invocation only projects geometry and writes a sealed receipt.  A
real pinned allocation is possible only with ``--allocate-pinned`` and only
when exact Windows RSS/private-commit counters are available.  This probe does
not load a model, initialize CUDA, or allocate the shared GPU staging pair.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from molor.experimental.paged_shared_staging_cache import (  # noqa: E402
    PagedSharedStagingCache,
    default_tensor_allocator,
)


SCHEMA_VERSION = "molor.paged-backing-allocation-probe.v1"
GIB = 2**30
MAX_MEMORY_GATE_BYTES = 6 * GIB
DEFAULT_PROJECTION_OUTPUT = Path(
    "runs/atxy-staged-kv-v1/qwen3-0.6b-paged-backing-projection-v1.json"
)
DEFAULT_ALLOCATION_OUTPUT = Path(
    "runs/atxy-staged-kv-v1/qwen3-0.6b-paged-backing-allocation-v1.json"
)
FROZEN_MODEL_CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
FROZEN_CACHE_SOURCE_SHA256 = "c49c9ee9c289fa39b1c087e0123a8b7ce4e6257d8cbdd50eec0f4cf57fdf4ef7"

FROZEN_QWEN3_06B_CONTRACT: dict[str, object] = {
    "model_id": "Qwen3-0.6B",
    "architecture": "Qwen3ForCausalLM",
    "num_hidden_layers": 28,
    "max_cache_len": 40_960,
    "page_token_capacity": 8_192,
    "num_key_value_heads": 8,
    "key_head_dim": 128,
    "value_head_dim": 128,
    "dtype": "bfloat16",
    "expected_page_count_per_role_per_layer": 5,
    "expected_backing_page_allocations": 280,
    "expected_logical_backing_bytes": 4_697_620_480,
    "expected_paged_allocator_bytes": 4_697_620_480,
    "expected_paged_rounding_waste_bytes": 0,
    "expected_unpaged_allocator_bytes": 7 * GIB,
    "expected_shared_stage_bytes_not_allocated_by_probe": 160 * 2**20,
    "model_config_sha256": FROZEN_MODEL_CONFIG_SHA256,
    "cache_source_sha256": FROZEN_CACHE_SOURCE_SHA256,
}


class ProbeValidationError(RuntimeError):
    """The probe cannot safely establish its requested validation claim."""


TensorAllocator = Callable[..., torch.Tensor]
MemoryReader = Callable[[], Mapping[str, object]]
ProvenanceReader = Callable[[], Mapping[str, object]]


def _plain_positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    """Exact cache geometry and memory gate for one probe receipt."""

    model_id: str
    num_layers: int
    max_cache_len: int
    page_token_capacity: int
    num_key_value_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: torch.dtype
    memory_gate_bytes: int = MAX_MEMORY_GATE_BYTES
    pin_cpu_backing: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        for field in (
            "num_layers",
            "max_cache_len",
            "page_token_capacity",
            "num_key_value_heads",
            "key_head_dim",
            "value_head_dim",
            "memory_gate_bytes",
        ):
            _plain_positive_int(getattr(self, field), field)
        if self.memory_gate_bytes > MAX_MEMORY_GATE_BYTES:
            raise ValueError("memory_gate_bytes must not exceed 6 GiB")
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch.dtype")
        if not isinstance(self.pin_cpu_backing, bool):
            raise TypeError("pin_cpu_backing must be boolean")

    @classmethod
    def frozen_qwen3_06b(cls, *, memory_gate_bytes: int = MAX_MEMORY_GATE_BYTES) -> ProbeConfig:
        return cls(
            model_id=str(FROZEN_QWEN3_06B_CONTRACT["model_id"]),
            num_layers=int(FROZEN_QWEN3_06B_CONTRACT["num_hidden_layers"]),
            max_cache_len=int(FROZEN_QWEN3_06B_CONTRACT["max_cache_len"]),
            page_token_capacity=int(FROZEN_QWEN3_06B_CONTRACT["page_token_capacity"]),
            num_key_value_heads=int(FROZEN_QWEN3_06B_CONTRACT["num_key_value_heads"]),
            key_head_dim=int(FROZEN_QWEN3_06B_CONTRACT["key_head_dim"]),
            value_head_dim=int(FROZEN_QWEN3_06B_CONTRACT["value_head_dim"]),
            dtype=torch.bfloat16,
            memory_gate_bytes=memory_gate_bytes,
            pin_cpu_backing=True,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "max_cache_len": self.max_cache_len,
            "page_token_capacity": self.page_token_capacity,
            "num_key_value_heads": self.num_key_value_heads,
            "key_head_dim": self.key_head_dim,
            "value_head_dim": self.value_head_dim,
            "dtype": _dtype_name(self.dtype),
            "pin_cpu_backing": self.pin_cpu_backing,
            "memory_gate_bytes": self.memory_gate_bytes,
            "memory_gate_gib": self.memory_gate_bytes / GIB,
            "memory_gate_hard_ceiling_bytes": MAX_MEMORY_GATE_BYTES,
        }


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_descriptor(path: Path, *, expected_sha256: str | None = None) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    digest = _file_sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProbeValidationError(
            f"frozen source hash mismatch for {resolved.relative_to(REPOSITORY_ROOT)}"
        )
    return {
        "path": resolved.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": digest,
        "expected_sha256": expected_sha256,
        "hash_verified": expected_sha256 is None or digest == expected_sha256,
    }


def verify_frozen_provenance() -> dict[str, object]:
    """Verify the exact local model config and cache implementation before allocation."""

    model_config_path = REPOSITORY_ROOT / "models" / "Qwen3-0.6B" / "config.json"
    cache_source_path = SOURCE_ROOT / "molor" / "experimental" / "paged_shared_staging_cache.py"
    script_path = Path(__file__).resolve()
    model_descriptor = _file_descriptor(
        model_config_path,
        expected_sha256=FROZEN_MODEL_CONFIG_SHA256,
    )
    cache_descriptor = _file_descriptor(
        cache_source_path,
        expected_sha256=FROZEN_CACHE_SOURCE_SHA256,
    )
    script_descriptor = _file_descriptor(script_path)
    model_payload = json.loads(model_config_path.read_text(encoding="utf-8"))
    expected_fields = {
        "architectures": [FROZEN_QWEN3_06B_CONTRACT["architecture"]],
        "head_dim": FROZEN_QWEN3_06B_CONTRACT["key_head_dim"],
        "max_position_embeddings": FROZEN_QWEN3_06B_CONTRACT["max_cache_len"],
        "num_hidden_layers": FROZEN_QWEN3_06B_CONTRACT["num_hidden_layers"],
        "num_key_value_heads": FROZEN_QWEN3_06B_CONTRACT["num_key_value_heads"],
        "torch_dtype": FROZEN_QWEN3_06B_CONTRACT["dtype"],
    }
    mismatches = {
        key: {"expected": expected, "observed": model_payload.get(key)}
        for key, expected in expected_fields.items()
        if model_payload.get(key) != expected
    }
    if mismatches:
        raise ProbeValidationError(f"frozen Qwen3-0.6B config fields drifted: {mismatches}")
    return {
        "verified": True,
        "frozen_contract_sha256": _canonical_sha256(FROZEN_QWEN3_06B_CONTRACT),
        "files": {
            "model_config": model_descriptor,
            "paged_cache_source": cache_descriptor,
            "probe_script": script_descriptor,
        },
        "model_config_fields": expected_fields,
    }


def _windows_process_memory() -> dict[str, object]:
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
    if not get_process_memory_info(
        get_current_process(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return {
        "current_rss_bytes": int(counters.WorkingSetSize),
        "peak_rss_bytes": int(counters.PeakWorkingSetSize),
        "private_committed_bytes": int(counters.PrivateUsage),
        "pagefile_usage_bytes": int(counters.PagefileUsage),
        "peak_pagefile_usage_bytes": int(counters.PeakPagefileUsage),
        "source": "Windows GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX",
        "gate_measurement_supported": True,
        "portable_fallback": False,
        "limitations": [],
    }


def _portable_process_memory() -> dict[str, object]:
    """Best-effort RSS fallback; deliberately unsupported for the allocation gate."""

    current_rss: int | None = None
    source = "portable fallback unavailable"
    statm = Path("/proc/self/statm")
    if statm.is_file():
        fields = statm.read_text(encoding="ascii").split()
        if len(fields) >= 2:
            current_rss = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
            source = "/proc/self/statm resident pages"
    if current_rss is None:
        try:
            import resource

            maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            current_rss = maximum_rss if sys.platform == "darwin" else maximum_rss * 1024
            source = "resource.getrusage peak RSS (not current RSS)"
        except (ImportError, OSError, ValueError):
            pass
    return {
        "current_rss_bytes": current_rss,
        "peak_rss_bytes": current_rss if "peak RSS" in source else None,
        "private_committed_bytes": None,
        "pagefile_usage_bytes": None,
        "peak_pagefile_usage_bytes": None,
        "source": source,
        "gate_measurement_supported": False,
        "portable_fallback": True,
        "limitations": [
            "Windows PrivateUsage is unavailable",
            "the pinned-allocation mode refuses to run without both current RSS and private bytes",
        ],
    }


def process_memory_snapshot() -> dict[str, object]:
    """Return exact Windows counters or an explicitly unsupported portable fallback."""

    return _windows_process_memory() if os.name == "nt" else _portable_process_memory()


def _normalize_memory_snapshot(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    source = normalized.get("source")
    supported = normalized.get("gate_measurement_supported")
    if not isinstance(source, str) or not source:
        raise ProbeValidationError("memory measurement source is missing")
    if not isinstance(supported, bool):
        raise ProbeValidationError("memory measurement support flag is invalid")
    for field in ("current_rss_bytes", "private_committed_bytes"):
        value = normalized.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ProbeValidationError(f"memory measurement {field} is invalid")
    if supported and (
        normalized.get("current_rss_bytes") is None
        or normalized.get("private_committed_bytes") is None
    ):
        raise ProbeValidationError(
            "supported memory measurement omitted RSS or private committed bytes"
        )
    normalized.setdefault("portable_fallback", not supported)
    normalized.setdefault("limitations", [])
    return normalized


def _environment_payload() -> dict[str, object]:
    try:
        transformers_version: str | None = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os_name": os.name,
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_initialized": torch.cuda.is_initialized(),
        "process_id": os.getpid(),
    }


def _new_cache(config: ProbeConfig, *, allocator: TensorAllocator) -> PagedSharedStagingCache:
    return PagedSharedStagingCache(
        num_layers=config.num_layers,
        max_cache_len=config.max_cache_len,
        page_token_capacity=config.page_token_capacity,
        num_key_value_heads=config.num_key_value_heads,
        key_head_dim=config.key_head_dim,
        value_head_dim=config.value_head_dim,
        dtype=config.dtype,
        staging_device="cpu",
        pin_cpu_backing=config.pin_cpu_backing,
        non_blocking_copies=False,
        telemetry_event_capacity=0,
        allocator=allocator,
    )


def _projection_checks(
    cache: PagedSharedStagingCache,
    config: ProbeConfig,
) -> dict[str, bool]:
    geometry = cache.geometry
    checks = {
        "geometry_matches_requested_config": (
            geometry.num_layers == config.num_layers
            and geometry.max_cache_len == config.max_cache_len
            and geometry.page_token_capacity == config.page_token_capacity
            and geometry.num_key_value_heads == config.num_key_value_heads
            and geometry.key_head_dim == config.key_head_dim
            and geometry.value_head_dim == config.value_head_dim
            and geometry.dtype == config.dtype
        ),
        "logical_backing_bytes_match_page_geometry": (
            geometry.logical_backing_bytes
            == geometry.num_layers
            * geometry.page_count
            * (geometry.key_page_bytes + geometry.value_page_bytes)
        ),
        "paged_allocator_rounding_waste_is_zero": geometry.predicted_rounding_waste_bytes == 0,
        "projected_backing_fits_memory_gate": (
            geometry.predicted_rounded_backing_bytes <= config.memory_gate_bytes
        ),
        "shared_stage_is_not_allocated": cache.staging_data_ptrs() is None,
        "cuda_remains_uninitialized": not torch.cuda.is_initialized(),
    }
    if config.model_id == FROZEN_QWEN3_06B_CONTRACT["model_id"]:
        checks.update(
            {
                "frozen_page_count_matches": geometry.page_count
                == FROZEN_QWEN3_06B_CONTRACT["expected_page_count_per_role_per_layer"],
                "frozen_page_allocations_match": geometry.backing_allocation_count
                == FROZEN_QWEN3_06B_CONTRACT["expected_backing_page_allocations"],
                "frozen_logical_backing_bytes_match": geometry.logical_backing_bytes
                == FROZEN_QWEN3_06B_CONTRACT["expected_logical_backing_bytes"],
                "frozen_paged_allocator_bytes_match": (
                    geometry.predicted_rounded_backing_bytes
                    == FROZEN_QWEN3_06B_CONTRACT["expected_paged_allocator_bytes"]
                ),
                "frozen_unpaged_allocator_bytes_match": (
                    geometry.unpaged_predicted_rounded_backing_bytes
                    == FROZEN_QWEN3_06B_CONTRACT["expected_unpaged_allocator_bytes"]
                ),
            }
        )
    return checks


def _initial_population_stats(cache: PagedSharedStagingCache) -> dict[str, object]:
    return {
        "performed": False,
        "completed": False,
        "layers_completed": 0,
        "page_tensor_copies_completed": 0,
        "expected_page_tensor_copies": cache.geometry.backing_allocation_count,
        "elements_copied": 0,
        "bytes_copied": 0,
        "reusable_input_tensors": 2,
        "reusable_input_bytes": cache.geometry.key_page_bytes + cache.geometry.value_page_bytes,
        "bounded_input_contract": "one reusable unpinned CPU K page and one reusable unpinned CPU V page",
        "validation": "torch.equal immediately after every page copy",
    }


def _allocate_and_populate_layer_by_layer(
    cache: PagedSharedStagingCache,
    population: dict[str, object],
) -> None:
    """Allocate then populate each layer using one bounded reusable input pair."""

    geometry = cache.geometry
    population["performed"] = True
    reusable_keys = torch.empty(geometry.key_page_shape(), dtype=geometry.dtype, device="cpu")
    reusable_values = torch.empty(geometry.value_page_shape(), dtype=geometry.dtype, device="cpu")
    if reusable_keys.is_pinned() or reusable_values.is_pinned():
        raise ProbeValidationError("reusable population inputs must remain unpinned")
    empty_keys = torch.empty(
        (1, geometry.num_key_value_heads, 0, geometry.key_head_dim),
        dtype=geometry.dtype,
        device="cpu",
    )
    empty_values = torch.empty(
        (1, geometry.num_key_value_heads, 0, geometry.value_head_dim),
        dtype=geometry.dtype,
        device="cpu",
    )
    for layer_idx, layer in enumerate(cache.layers):
        layer.lazy_initialization(empty_keys, empty_values)
        if cache.telemetry()["health"]["poisoned"]:  # type: ignore[index]
            raise ProbeValidationError("cache poisoned during layer allocation")
        if (
            len(layer.key_pages) != geometry.page_count
            or len(layer.value_pages) != geometry.page_count
        ):
            raise ProbeValidationError("layer page allocation count is incomplete")
        for page_idx in range(geometry.page_count):
            ordinal = layer_idx * geometry.page_count + page_idx + 1
            reusable_keys.fill_(float(ordinal))
            reusable_values.fill_(float(-ordinal))
            layer.key_pages[page_idx].copy_(reusable_keys)
            if not torch.equal(layer.key_pages[page_idx], reusable_keys):
                raise ProbeValidationError("key page population verification failed")
            population["page_tensor_copies_completed"] = (
                int(population["page_tensor_copies_completed"]) + 1
            )
            population["elements_copied"] = (
                int(population["elements_copied"]) + reusable_keys.numel()
            )
            population["bytes_copied"] = (
                int(population["bytes_copied"])
                + reusable_keys.numel() * reusable_keys.element_size()
            )
            layer.value_pages[page_idx].copy_(reusable_values)
            if not torch.equal(layer.value_pages[page_idx], reusable_values):
                raise ProbeValidationError("value page population verification failed")
            population["page_tensor_copies_completed"] = (
                int(population["page_tensor_copies_completed"]) + 1
            )
            population["elements_copied"] = (
                int(population["elements_copied"]) + reusable_values.numel()
            )
            population["bytes_copied"] = (
                int(population["bytes_copied"])
                + reusable_values.numel() * reusable_values.element_size()
            )
        population["layers_completed"] = layer_idx + 1
    population["completed"] = True


def _allocation_checks(
    cache: PagedSharedStagingCache,
    config: ProbeConfig,
    population: Mapping[str, object],
) -> dict[str, bool]:
    telemetry = cache.telemetry()
    storage = telemetry["storage"]
    health = telemetry["health"]
    pointers = [
        pointer
        for layer in cache.backing_page_data_ptrs()
        for roles in layer or ()
        for pointer in roles
    ]
    pages = [page for layer in cache.layers for page in (*layer.key_pages, *layer.value_pages)]
    expected_allocations = cache.geometry.backing_allocation_count
    return {
        "cache_is_initialized": cache.is_initialized,
        "cache_is_healthy": health["poisoned"] is False,  # type: ignore[index]
        "all_layer_transactions_completed": (
            telemetry["operations"]["backing_allocation_transactions"] == config.num_layers  # type: ignore[index]
        ),
        "live_page_allocations_match": (
            storage["live_backing_page_allocations"] == expected_allocations  # type: ignore[index]
        ),
        "live_logical_backing_bytes_match": (
            storage["live_logical_backing_bytes"] == cache.geometry.logical_backing_bytes  # type: ignore[index]
        ),
        "all_page_storage_is_exclusive": (
            len(pointers) == len(set(pointers)) == expected_allocations
        ),
        "all_pages_have_requested_pin_state": (
            len(pages) == expected_allocations
            and all(page.is_pinned() is config.pin_cpu_backing for page in pages)
        ),
        "all_page_tensors_populated_and_verified": (
            population.get("completed") is True
            and population.get("page_tensor_copies_completed") == expected_allocations
        ),
        "shared_stage_remains_unallocated": (
            storage["staging_allocated"] is False and storage["staging_bytes"] == 0  # type: ignore[index]
        ),
        "logical_lengths_remain_zero": all(
            length == 0
            for length in telemetry["sequence"]["layer_logical_lengths"]  # type: ignore[index]
        ),
        "cuda_remains_uninitialized": not torch.cuda.is_initialized(),
    }


def _memory_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
    gate_bytes: int,
) -> dict[str, object]:
    rss = int(after["current_rss_bytes"]) - int(before["current_rss_bytes"])
    private = int(after["private_committed_bytes"]) - int(before["private_committed_bytes"])
    return {
        "incremental_rss_bytes": rss,
        "incremental_private_committed_bytes": private,
        "maximum_incremental_process_bytes": max(rss, private),
        "gate_bytes": gate_bytes,
        "rss_nonnegative": rss >= 0,
        "private_nonnegative": private >= 0,
        "rss_within_gate": 0 <= rss <= gate_bytes,
        "private_within_gate": 0 <= private <= gate_bytes,
        "passed": 0 <= rss <= gate_bytes and 0 <= private <= gate_bytes,
        "gate_policy": "both incremental current RSS and Windows PrivateUsage must be <= gate",
    }


def run_probe(
    config: ProbeConfig,
    *,
    allocate: bool,
    memory_reader: MemoryReader = process_memory_snapshot,
    provenance_reader: ProvenanceReader = verify_frozen_provenance,
    allocator: TensorAllocator = default_tensor_allocator,
    permit_unpinned_test_allocation: bool = False,
) -> dict[str, object]:
    """Build one receipt payload without publishing it.

    ``permit_unpinned_test_allocation`` exists only so unit tests can exercise
    the control flow with tiny ordinary CPU tensors.  The CLI never enables it.
    """

    started = time.perf_counter()
    phase = "provenance"
    errors: list[dict[str, str]] = []
    cache: PagedSharedStagingCache | None = None
    provenance: dict[str, object] | None = None
    before: dict[str, object] | None = None
    after: dict[str, object] | None = None
    delta: dict[str, object] | None = None
    projection_checks: dict[str, bool] = {}
    allocation_checks: dict[str, bool] = {}
    population: dict[str, object] | None = None
    status = "failed"
    allocation_attempted = False
    try:
        provenance = dict(provenance_reader())
        if provenance.get("verified") is not True:
            raise ProbeValidationError("provenance reader did not verify the frozen inputs")
        phase = "cache_projection"
        cache = _new_cache(config, allocator=allocator)
        population = _initial_population_stats(cache)
        projection_checks = _projection_checks(cache, config)
        if not all(projection_checks.values()):
            raise ProbeValidationError("one or more cache projection checks failed")
        phase = "memory_preflight"
        before = _normalize_memory_snapshot(memory_reader())
        if not allocate:
            status = "projected"
        else:
            if not config.pin_cpu_backing and not permit_unpinned_test_allocation:
                raise ProbeValidationError("allocation mode requires pinned CPU backing")
            if config.memory_gate_bytes > MAX_MEMORY_GATE_BYTES:
                raise ProbeValidationError("memory gate exceeds the hard 6 GiB ceiling")
            if not projection_checks["projected_backing_fits_memory_gate"]:
                raise ProbeValidationError("projected backing exceeds the configured memory gate")
            if before.get("gate_measurement_supported") is not True:
                raise ProbeValidationError(
                    "allocation refused: exact RSS and private-byte measurement is unsupported"
                )
            phase = "allocate_and_populate"
            allocation_attempted = True
            _allocate_and_populate_layer_by_layer(cache, population)
            phase = "allocation_validation"
            allocation_checks = _allocation_checks(cache, config, population)
            if not all(allocation_checks.values()):
                raise ProbeValidationError("one or more live allocation checks failed")
            phase = "memory_postflight"
            after = _normalize_memory_snapshot(memory_reader())
            if after.get("gate_measurement_supported") is not True:
                raise ProbeValidationError("post-allocation memory measurement became unsupported")
            if after.get("source") != before.get("source"):
                raise ProbeValidationError("memory measurement source changed during the probe")
            delta = _memory_delta(before, after, config.memory_gate_bytes)
            if delta["passed"] is not True:
                raise ProbeValidationError(
                    "incremental process memory exceeded the configured gate"
                )
            status = "completed"
    except Exception as exc:
        errors.append(
            {
                "phase": phase,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        status = "failed"
        if allocate and allocation_attempted and before is not None and after is None:
            try:
                after = _normalize_memory_snapshot(memory_reader())
                if (
                    before.get("gate_measurement_supported") is True
                    and after.get("gate_measurement_supported") is True
                    and before.get("source") == after.get("source")
                ):
                    delta = _memory_delta(before, after, config.memory_gate_bytes)
            except Exception as measurement_error:
                errors.append(
                    {
                        "phase": "failure_memory_postflight",
                        "type": type(measurement_error).__name__,
                        "message": str(measurement_error),
                    }
                )
    telemetry = cache.telemetry() if cache is not None else None
    health = telemetry.get("health") if isinstance(telemetry, Mapping) else None
    if isinstance(health, Mapping) and health.get("poisoned") is True and status != "failed":
        status = "failed"
        errors.append(
            {
                "phase": "final_health_check",
                "type": "ProbeValidationError",
                "message": "cache was poisoned at receipt finalization",
            }
        )
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "mode": "pinned_allocation" if allocate else "dry_run_projection",
        "config": config.to_dict(),
        "frozen_contract": dict(FROZEN_QWEN3_06B_CONTRACT),
        "provenance": provenance,
        "environment": _environment_payload(),
        "checks": {
            "projection": projection_checks,
            "allocation": allocation_checks,
            "allocation_checks_evaluated": bool(allocation_checks),
            "measurement_supported_for_allocation": (
                before.get("gate_measurement_supported") is True if before is not None else False
            ),
            "incremental_memory_gate_passed": (delta.get("passed") if delta is not None else None),
        },
        "cache_telemetry": telemetry,
        "population": population,
        "measurements": {
            "before": before,
            "after": after,
            "delta": delta,
            "allocation_attempted": allocation_attempted,
            "portable_fallback_policy": (
                "fallback RSS may be reported for dry-run diagnostics, but allocation fails closed "
                "unless Windows current RSS and PrivateUsage are both available"
            ),
        },
        "errors": errors,
        "duration_seconds": time.perf_counter() - started,
    }
    receipt["hashes"] = {
        "config_sha256": _canonical_sha256(receipt["config"]),
        "frozen_contract_sha256": _canonical_sha256(receipt["frozen_contract"]),
        "cache_telemetry_sha256": _canonical_sha256(receipt["cache_telemetry"]),
        "measurements_sha256": _canonical_sha256(receipt["measurements"]),
        "population_sha256": _canonical_sha256(receipt["population"]),
    }
    return receipt


def seal_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    sealed = dict(receipt)
    if "content_sha256" in sealed:
        raise ValueError("receipt is already sealed")
    sealed["content_sha256"] = _canonical_sha256(sealed)
    return sealed


def verify_receipt_seal(receipt: Mapping[str, object]) -> bool:
    claimed = receipt.get("content_sha256")
    if not isinstance(claimed, str):
        return False
    unsealed = dict(receipt)
    del unsealed["content_sha256"]
    return claimed == _canonical_sha256(unsealed)


def write_json_atomic_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    """Publish complete JSON atomically and never replace an existing destination."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}-{secrets.token_hex(8)}"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_gate_gib(value: str) -> int:
    try:
        gib = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("memory gate must be a decimal GiB value") from exc
    if not gib.is_finite() or gib <= 0 or gib > 6:
        raise argparse.ArgumentTypeError("memory gate must be > 0 and <= 6 GiB")
    return int(gib * GIB)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocate-pinned",
        action="store_true",
        help=(
            "opt in to allocating and populating the complete ~4.375 GiB pinned CPU backing; "
            "no model or CUDA stage is created"
        ),
    )
    parser.add_argument(
        "--max-incremental-gib",
        type=_parse_gate_gib,
        default=MAX_MEMORY_GATE_BYTES,
        metavar="GIB",
        help="RSS and private-commit gate in GiB, at most 6 (default: 6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="exclusive JSON receipt path (default is mode-specific beneath runs/atxy-staged-kv-v1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    default_output = (
        DEFAULT_ALLOCATION_OUTPUT if args.allocate_pinned else DEFAULT_PROJECTION_OUTPUT
    )
    output = (args.output if args.output is not None else default_output).resolve()
    if output.exists():
        print(f"refusing existing receipt: {output}", file=sys.stderr)
        return 2
    config = ProbeConfig.frozen_qwen3_06b(memory_gate_bytes=args.max_incremental_gib)
    receipt = run_probe(config, allocate=args.allocate_pinned)
    receipt["publication"] = {
        "requested_output": str(output),
        "atomic_publication": True,
        "existing_destination_replaced": False,
    }
    sealed = seal_receipt(receipt)
    try:
        write_json_atomic_exclusive(output, sealed)
    except FileExistsError:
        print(f"refusing concurrently created receipt: {output}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": sealed["status"],
                "mode": sealed["mode"],
                "output": str(output),
                "content_sha256": sealed["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if sealed["status"] in {"projected", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
