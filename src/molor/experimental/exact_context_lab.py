"""Exact GPU-resident versus CPU-offloaded Hugging Face KV-cache benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time
from typing import Any, Callable, Protocol

import torch
from torch import Tensor

from molor.benchmark.lock import lock_backend_identity


SCHEMA_VERSION = "molor.exact-context-report.v1"
DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/exact-context-v1")
DEFAULT_LENGTHS = (512, 2_048, 4_096)
DEFAULT_MODES = ("dynamic", "offloaded")
MIB = 2**20
GIB = 2**30

_SYNTHETIC_PROMPT = """\
Exact context systems probe. Record each item without assuming that neighboring items are related.
Embedded control: configure clocks before releasing peripherals from reset. Architecture: persist
durable state before acknowledging a transition. Finance: cash flow and accounting profit are
different measures. Logic: validity requires every permitted assignment to preserve the conclusion.
Retrieval marker ALDER-2047 belongs only to this deterministic synthetic workload.
"""


class ExactnessError(RuntimeError):
    """A completed pair of supposedly exact runs produced different token identities."""


class _Telemetry(Protocol):
    def reset_cuda_peaks(self) -> None: ...

    def synchronize(self) -> None: ...

    def observe(self, label: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ContextRequest:
    modes: tuple[str, ...]
    lengths: tuple[int, ...]
    max_new_tokens: int
    repeats: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _finite_float_or_text(value: float) -> float | str:
    """Keep valid CLI floats numeric while making rejected values evidence-safe."""

    return value if math.isfinite(value) else repr(value)


def _seal_report(payload: dict[str, Any]) -> dict[str, Any]:
    if "integrity" in payload:
        raise ValueError("unsealed report payload must not contain integrity")
    sealed = _jsonable(payload)
    if not isinstance(sealed, dict):  # pragma: no cover - fixed by the type contract
        raise TypeError("report payload must serialize to an object")
    sealed["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON of this report with the integrity member omitted",
        "sha256": _canonical_sha256(sealed),
    }
    return sealed


def _verify_report_integrity(report: dict[str, Any]) -> bool:
    candidate = dict(report)
    integrity = candidate.pop("integrity", None)
    return bool(
        isinstance(integrity, dict)
        and integrity.get("algorithm") == "sha256"
        and integrity.get("sha256") == _canonical_sha256(candidate)
    )


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create one LF JSON report and refuse to replace any existing path."""

    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _token_id_sha256(token_ids: list[int] | tuple[int, ...] | Tensor) -> str:
    values = (
        token_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
        if isinstance(token_ids, Tensor)
        else list(token_ids)
    )
    digest = hashlib.sha256()
    digest.update(b"molor.token-ids.signed-int64-big-endian.v1\0")
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("token IDs must be integers")
        digest.update(struct.pack(">q", value))
    return digest.hexdigest()


def _normalize_tokenizer_ids(raw_ids: Any) -> list[int]:
    if isinstance(raw_ids, Tensor):
        raw_ids = raw_ids.detach().to(device="cpu").tolist()
    if isinstance(raw_ids, tuple):
        raw_ids = list(raw_ids)
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("synthetic prompt seed tokenized to an empty or unsupported sequence")
    if isinstance(raw_ids[0], list):
        if len(raw_ids) != 1:
            raise ValueError("synthetic prompt tokenizer unexpectedly returned a batch")
        raw_ids = raw_ids[0]
    values: list[int] = []
    for raw_id in raw_ids:
        if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id < 0:
            raise ValueError("synthetic prompt tokenizer returned an invalid token ID")
        values.append(raw_id)
    return values


def _synthetic_prompt_ids(tokenizer: Any, token_count: int) -> Tensor:
    """Repeat deterministic tokenizer output and truncate to an exact token count."""

    if token_count < 1:
        raise ValueError("token_count must be positive")
    encoded = tokenizer(
        _SYNTHETIC_PROMPT,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    if not isinstance(encoded, dict) or "input_ids" not in encoded:
        raise ValueError("tokenizer did not return input_ids")
    seed_ids = _normalize_tokenizer_ids(encoded["input_ids"])
    repeats = math.ceil(token_count / len(seed_ids))
    result = torch.tensor((seed_ids * repeats)[:token_count], dtype=torch.long)
    if result.shape != (token_count,):  # pragma: no cover - construction invariant
        raise RuntimeError("failed to construct the exact requested prompt length")
    return result


def _validate_context_request(
    request: ContextRequest,
    *,
    max_position_embeddings: int,
) -> None:
    if not isinstance(max_position_embeddings, int) or max_position_embeddings < 1:
        raise ValueError("model max_position_embeddings must be a positive integer")
    if not request.modes or any(mode not in DEFAULT_MODES for mode in request.modes):
        raise ValueError("modes must contain only dynamic and/or offloaded")
    if len(set(request.modes)) != len(request.modes):
        raise ValueError("modes must be unique")
    if not request.lengths or any(length < 1 for length in request.lengths):
        raise ValueError("lengths must contain positive integers")
    if len(set(request.lengths)) != len(request.lengths):
        raise ValueError("lengths must be unique")
    if request.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if request.repeats < 1:
        raise ValueError("repeats must be positive")
    for length in request.lengths:
        requested_total = length + request.max_new_tokens
        if requested_total > max_position_embeddings:
            raise ValueError(
                f"prompt {length:,} + completion {request.max_new_tokens:,} = "
                f"{requested_total:,} exceeds model max_position_embeddings "
                f"{max_position_embeddings:,}"
            )


def _cache_sequence_length(cache: Any) -> int | None:
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        return None
    value = getter()
    if isinstance(value, Tensor):
        value = value.item()
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError("cache returned an invalid sequence length")
    return value


def _validate_logits(logits: Any) -> Tensor:
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise RuntimeError("model must return rank-3 logits")
    if logits.shape[0] != 1 or logits.shape[1] != 1 or logits.shape[2] < 1:
        raise RuntimeError("model must honor logits_to_keep=1 for the manual decode contract")
    if not bool(torch.isfinite(logits[:, -1, :]).all().item()):
        raise RuntimeError("model returned non-finite next-token logits")
    return logits


def _manual_greedy_decode(
    model: Any,
    *,
    prompt_ids: Tensor,
    cache: Any,
    max_new_tokens: int,
    device: torch.device,
    telemetry: _Telemetry,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run an exact-length prefill and deterministic, no-sampling manual decode."""

    if prompt_ids.ndim != 1 or prompt_ids.numel() < 1:
        raise ValueError("prompt_ids must be a non-empty rank-1 tensor")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    prompt_tokens = int(prompt_ids.numel())
    prompt = prompt_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(prompt)
    cache_position = torch.arange(prompt_tokens, device=device, dtype=torch.long)
    observations: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    completion_ids: list[int] = []

    telemetry.reset_cuda_peaks()
    telemetry.synchronize()
    observations.append(telemetry.observe("before_prefill"))
    total_started = clock()

    with torch.inference_mode():
        telemetry.synchronize()
        prefill_started = clock()
        outputs = model(
            input_ids=prompt,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = _validate_logits(outputs.logits)
        next_token = logits[:, -1, :].argmax(dim=-1)
        telemetry.synchronize()
        prefill_seconds = clock() - prefill_started
        cache = getattr(outputs, "past_key_values", None)
        if cache is None:
            raise RuntimeError("model did not return past_key_values")
        sequence_length = _cache_sequence_length(cache)
        if sequence_length is not None and sequence_length != prompt_tokens:
            raise RuntimeError(
                f"cache contains {sequence_length} tokens after a {prompt_tokens}-token prefill"
            )
        first_token_id = int(next_token.item())
        completion_ids.append(first_token_id)
        first_observation = telemetry.observe("after_prefill_and_first_token")
        observations.append(first_observation)
        steps.append(
            {
                "completion_index": 0,
                "phase": "prefill_to_first_token",
                "model_input_tokens": prompt_tokens,
                "cache_tokens_after_forward": sequence_length,
                "output_token_id": first_token_id,
                "seconds": prefill_seconds,
                "memory_observation": first_observation,
            }
        )

        for completion_index in range(1, max_new_tokens):
            position = prompt_tokens + completion_index - 1
            attention_mask = torch.ones((1, position + 1), device=device, dtype=torch.long)
            cache_position = torch.tensor([position], device=device, dtype=torch.long)
            telemetry.synchronize()
            step_started = clock()
            outputs = model(
                input_ids=next_token.reshape(1, 1),
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            logits = _validate_logits(outputs.logits)
            next_token = logits[:, -1, :].argmax(dim=-1)
            telemetry.synchronize()
            step_seconds = clock() - step_started
            cache = getattr(outputs, "past_key_values", None)
            if cache is None:
                raise RuntimeError("model did not return past_key_values")
            sequence_length = _cache_sequence_length(cache)
            expected_cache_length = prompt_tokens + completion_index
            if sequence_length is not None and sequence_length != expected_cache_length:
                raise RuntimeError(
                    f"cache contains {sequence_length} tokens; expected "
                    f"{expected_cache_length} after decode step {completion_index}"
                )
            token_id = int(next_token.item())
            completion_ids.append(token_id)
            observation = telemetry.observe(f"after_completion_{completion_index + 1}")
            observations.append(observation)
            steps.append(
                {
                    "completion_index": completion_index,
                    "phase": "single_token_decode",
                    "model_input_tokens": 1,
                    "cache_tokens_after_forward": sequence_length,
                    "output_token_id": token_id,
                    "seconds": step_seconds,
                    "memory_observation": observation,
                }
            )

    telemetry.synchronize()
    total_seconds = clock() - total_started
    final_observation = telemetry.observe("run_complete")
    observations.append(final_observation)
    decode_seconds = sum(float(step["seconds"]) for step in steps[1:])
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(completion_ids),
        "completion_token_ids": completion_ids,
        "completion_token_ids_sha256": _token_id_sha256(completion_ids),
        "eos_stopping": False,
        "timing": {
            "prefill_seconds": prefill_seconds,
            "ttft_seconds": prefill_seconds,
            "ttft_scope": (
                "prefill model call plus finite-logit validation and greedy argmax; excludes "
                "prompt host-to-device transfer, cache construction, and pre-run telemetry"
            ),
            "decode_model_step_seconds": decode_seconds,
            "all_model_step_seconds": sum(float(step["seconds"]) for step in steps),
            "run_wall_seconds": total_seconds,
            "decode_tokens_per_second": (
                (len(completion_ids) - 1) / decode_seconds
                if len(completion_ids) > 1 and decode_seconds > 0
                else None
            ),
        },
        "steps": steps,
        "memory_observations": observations,
        "final_cache_tokens": _cache_sequence_length(cache),
    }


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if left_id != right_id:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _compare_run_token_identity(
    records: list[dict[str, Any]],
    *,
    expected_request: ContextRequest | None = None,
) -> dict[str, Any]:
    """Compare repeats and matched dynamic/offloaded runs without using decoded text."""

    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    actual_keys = [
        (int(record["prompt_tokens"]), int(record["repeat"]), record["cache"]["mode"])
        for record in records
    ]
    duplicate_keys = sorted(key for key, count in Counter(actual_keys).items() if count > 1)
    expected_keys = (
        {
            (length, repeat, mode)
            for length in expected_request.lengths
            for repeat in range(expected_request.repeats)
            for mode in expected_request.modes
        }
        if expected_request is not None
        else set(actual_keys)
    )
    missing_keys = sorted(expected_keys - set(actual_keys))
    unexpected_keys = sorted(set(actual_keys) - expected_keys)
    matrix_complete = not duplicate_keys and not missing_keys and not unexpected_keys

    def render_keys(keys: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
        return [
            {"prompt_tokens": length, "repeat": repeat, "mode": mode}
            for length, repeat, mode in keys
        ]

    lengths = sorted({int(record["prompt_tokens"]) for record in records})
    for prompt_tokens in lengths:
        group = [record for record in records if record["prompt_tokens"] == prompt_tokens]
        reference = group[0]
        for candidate in group[1:]:
            mismatch = _first_mismatch(
                reference["completion_token_ids"], candidate["completion_token_ids"]
            )
            comparison = {
                "scope": "same_prompt_all_modes_and_repeats",
                "prompt_tokens": prompt_tokens,
                "reference": {
                    "mode": reference["cache"]["mode"],
                    "repeat": reference["repeat"],
                    "sha256": reference["completion_token_ids_sha256"],
                },
                "candidate": {
                    "mode": candidate["cache"]["mode"],
                    "repeat": candidate["repeat"],
                    "sha256": candidate["completion_token_ids_sha256"],
                },
                "identical": mismatch is None,
                "first_mismatch_index": mismatch,
            }
            comparisons.append(comparison)
            if mismatch is not None:
                failures.append(comparison)

    matched_pairs: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for key, record in zip(actual_keys, records, strict=True):
        by_key.setdefault(key, record)
    pair_keys = sorted({(length, repeat) for length, repeat, _mode in by_key})
    for length, repeat in pair_keys:
        dynamic = by_key.get((length, repeat, "dynamic"))
        offloaded = by_key.get((length, repeat, "offloaded"))
        if dynamic is None or offloaded is None:
            continue
        mismatch = _first_mismatch(
            dynamic["completion_token_ids"], offloaded["completion_token_ids"]
        )
        matched_pairs.append(
            {
                "prompt_tokens": length,
                "repeat": repeat,
                "dynamic_sha256": dynamic["completion_token_ids_sha256"],
                "offloaded_sha256": offloaded["completion_token_ids_sha256"],
                "identical": mismatch is None,
                "first_mismatch_index": mismatch,
            }
        )
    return {
        "matrix": {
            "expected_cells": len(expected_keys),
            "actual_records": len(records),
            "complete_and_unique": matrix_complete,
            "missing": render_keys(missing_keys),
            "duplicates": render_keys(duplicate_keys),
            "unexpected": render_keys(unexpected_keys),
        },
        "all_completed_run_tokens_identical_by_prompt": matrix_complete and not failures,
        "comparisons": comparisons,
        "dynamic_vs_offloaded_matched_pairs": matched_pairs,
        "failure_count": (
            len(failures) + len(missing_keys) + len(duplicate_keys) + len(unexpected_keys)
        ),
    }


def _process_memory() -> dict[str, Any]:
    """Return current process RSS and a process peak where the platform exposes one."""

    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
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
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        succeeded = get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        )
        if not succeeded:
            raise OSError("GetProcessMemoryInfo failed")
        return {
            "current_rss_bytes": int(counters.WorkingSetSize),
            "peak_rss_bytes": int(counters.PeakWorkingSetSize),
            "source": "Windows GetProcessMemoryInfo working set",
        }

    statm = Path("/proc/self/statm")
    if statm.is_file():
        fields = statm.read_text(encoding="ascii").split()
        if len(fields) < 2:
            raise OSError("/proc/self/statm did not contain an RSS field")
        return {
            "current_rss_bytes": int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE")),
            "peak_rss_bytes": None,
            "source": "/proc/self/statm resident pages",
        }

    raise OSError(f"current process RSS is unsupported on {platform.system()}")


class RuntimeTelemetry:
    """Process RSS plus CUDA allocator and device-global memory observations."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def reset_cuda_peaks(self) -> None:
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def observe(self, label: str) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "label": label,
            "process": _process_memory(),
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


def _config_identity(config: Any) -> dict[str, Any]:
    payload = config.to_dict()
    return {
        "class": f"{type(config).__module__}.{type(config).__qualname__}",
        "canonical_sha256": _canonical_sha256(payload),
        "model_type": getattr(config, "model_type", None),
        "architectures": _jsonable(getattr(config, "architectures", None)),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "rope_scaling": _jsonable(getattr(config, "rope_scaling", None)),
        "commit_hash": getattr(config, "_commit_hash", None),
    }


_CONTEXT_CONFIG_FIELDS = (
    "max_position_embeddings",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "sliding_window",
)


def _context_config_contract(config: Any) -> dict[str, Any]:
    return {name: _jsonable(getattr(config, name, None)) for name in _CONTEXT_CONFIG_FIELDS}


def _validate_loaded_config_contract(
    source_config: Any,
    loaded_config: Any,
    request: ContextRequest,
) -> dict[str, Any]:
    source = _context_config_contract(source_config)
    loaded = _context_config_contract(loaded_config)
    if source != loaded:
        changed = [name for name in _CONTEXT_CONFIG_FIELDS if source[name] != loaded[name]]
        raise ValueError(
            "loaded model context config differs from pre-load config: " + ", ".join(changed)
        )
    _validate_context_request(
        request,
        max_position_embeddings=getattr(loaded_config, "max_position_embeddings", None),
    )
    return loaded


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("tokenizer get_vocab() returned no identity payload")
    normalized_vocabulary = sorted(
        (str(token), int(token_id)) for token, token_id in vocabulary.items()
    )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None and hasattr(backend, "to_str") else None
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size_attribute": getattr(tokenizer, "vocab_size", None),
        "vocabulary_entries": len(normalized_vocabulary),
        "vocabulary_sha256": _canonical_sha256(normalized_vocabulary),
        "backend_json_sha256": (
            hashlib.sha256(backend_json.encode("utf-8")).hexdigest()
            if backend_json is not None
            else None
        ),
        "special_tokens_map_sha256": _canonical_sha256(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "chat_template_sha256": (
            hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            if isinstance(chat_template, str)
            else None
        ),
        "commit_hash": getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    }


def _local_or_hub_model(value: str) -> tuple[str, bool]:
    path = Path(value).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError("exact context lab requires a model directory, not a single file")
        return str(path.resolve()), True
    return value, False


def _model_artifact_identity(
    model_id: str, *, is_local: bool, revision: str | None
) -> dict[str, Any]:
    if is_local:
        identity = lock_backend_identity(
            {
                "backend": "huggingface-local",
                "model_id": model_id,
                "model_artifact_path": model_id,
            },
            cache_path=Path(".unused-exact-context-lock-cache.json"),
        )
        if identity.get("fully_content_addressed") is not True:
            raise ValueError("local model could not be fully content-addressed")
        return identity
    if (
        revision is None
        or len(revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise ValueError("a remote model requires --revision with a full 40-hex commit hash")
    return {
        "schema_version": "molor.huggingface-commit-identity.v1",
        "model_id": model_id,
        "revision": revision.lower(),
        "fully_content_addressed": True,
        "hash_policy": "Hugging Face repository full commit identity",
    }


def _torch_dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return values[name]


def _device_identity(device: torch.device) -> dict[str, Any]:
    identity: dict[str, Any] = {"requested_and_normalized": str(device), "type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        identity.update(
            {
                "index": index,
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(properties.total_memory),
                "uuid": str(getattr(properties, "uuid", "")) or None,
            }
        )
    return identity


def _environment_identity(
    transformers_module: Any, dynamic_cache_class: type[Any]
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "transformers": transformers_module.__version__,
        "dynamic_cache_class": (
            f"{dynamic_cache_class.__module__}.{dynamic_cache_class.__qualname__}"
        ),
        "dynamic_cache_signature": str(inspect.signature(dynamic_cache_class)),
    }


def _nominal_kv_bytes(config: Any, *, sequence_tokens: int, dtype: torch.dtype) -> int | None:
    values = [
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_key_value_heads", None),
        getattr(config, "head_dim", None),
    ]
    if any(not isinstance(value, int) or value < 1 for value in values):
        return None
    element_bytes = torch.empty((), dtype=dtype).element_size()
    layers, kv_heads, head_dim = values
    return 2 * layers * kv_heads * head_dim * sequence_tokens * element_bytes


def _configure_determinism(seed: int) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "seed": seed,
        "greedy_argmax": True,
        "sampling": False,
        "eos_early_stopping": False,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def _make_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    directory = args.output_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return directory / f"exact-context-{timestamp}-{os.getpid()}-{time.time_ns()}.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact manual greedy decoding with GPU-resident and CPU-offloaded "
            "Hugging Face DynamicCache."
        )
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--revision", help="required full commit hash for a remote model")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8419)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument(
        "--minimum-free-cuda-gib",
        type=float,
        default=5.0,
        help="fail before model loading when device-global free memory is below this value",
    )
    parser.add_argument(
        "--cuda-headroom-gib",
        type=float,
        default=1.0,
        help="free-memory reserve required beyond nominal dynamic-cache tensor bytes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def _error_record(error: BaseException) -> dict[str, Any]:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }


def _run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    request = ContextRequest(
        modes=tuple(args.modes),
        lengths=tuple(args.lengths),
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
    )
    if args.seed < 0 or args.seed > torch.iinfo(torch.int64).max:
        raise ValueError("seed must be a non-negative signed 64-bit integer")
    if (
        not math.isfinite(args.minimum_free_cuda_gib)
        or not math.isfinite(args.cuda_headroom_gib)
        or args.minimum_free_cuda_gib < 0
        or args.cuda_headroom_gib < 0
    ):
        raise ValueError("CUDA memory guards must be finite and non-negative")

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("install molor[hf] for the exact context lab") from error

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
    config_identity = _config_identity(config)
    max_positions = getattr(config, "max_position_embeddings", None)
    _validate_context_request(request, max_position_embeddings=max_positions)
    report["request"]["validated_max_position_embeddings"] = max_positions
    report["identities"] = {"config": config_identity}

    identity_started = time.perf_counter()
    report["identities"]["model_artifact"] = _model_artifact_identity(
        model_id, is_local=is_local, revision=args.revision
    )
    report["identity_seconds"] = time.perf_counter() - identity_started

    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    report["identities"]["tokenizer"] = _tokenizer_identity(tokenizer)
    if not is_local:
        expected_revision = args.revision.lower()
        observed_commits = {
            value.lower()
            for value in (
                config_identity.get("commit_hash"),
                report["identities"]["tokenizer"].get("commit_hash"),
            )
            if isinstance(value, str) and len(value) == 40
        }
        if observed_commits != {expected_revision}:
            raise ValueError(
                "remote config/tokenizer commit identity does not match the requested revision"
            )

    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("model config must declare a positive vocab_size")
    prompts: dict[int, Tensor] = {}
    prompt_evidence: list[dict[str, Any]] = []
    for length in request.lengths:
        prompt_ids = _synthetic_prompt_ids(tokenizer, length)
        if int(prompt_ids.min().item()) < 0 or int(prompt_ids.max().item()) >= vocab_size:
            raise ValueError("synthetic prompt contains a token outside model vocab_size")
        prompts[length] = prompt_ids
        prompt_evidence.append(
            {
                "prompt_tokens": length,
                "token_ids_sha256": _token_id_sha256(prompt_ids),
                "token_id_min": int(prompt_ids.min().item()),
                "token_id_max": int(prompt_ids.max().item()),
                "construction": "repeat tokenizer output without special tokens, then exact truncate",
            }
        )
    report["prompts"] = prompt_evidence

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if "offloaded" in request.modes and device.type != "cuda":
        raise ValueError("HF DynamicCache offloading requires a CUDA model device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("requested CUDA device does not support bfloat16")
    report["environment"] = _environment_identity(transformers, DynamicCache)
    report["environment"]["device"] = _device_identity(device)
    report["determinism"] = _configure_determinism(args.seed)

    telemetry = RuntimeTelemetry(device)
    initial_observation = telemetry.observe("before_model_load")
    report["model_load_memory"] = [initial_observation]
    if device.type == "cuda":
        free_bytes = initial_observation["cuda"]["device_global_free_bytes"]
        required_before_load = int(args.minimum_free_cuda_gib * GIB)
        if free_bytes < required_before_load:
            raise RuntimeError(
                f"CUDA device has {free_bytes / GIB:.2f} GiB free, below the "
                f"{args.minimum_free_cuda_gib:.2f} GiB pre-load guard"
            )

    dtype = _torch_dtype(args.dtype)
    print(f"Loading exact base model {model_id} on {device} as {dtype}", flush=True)
    load_started = time.perf_counter()
    model = None
    try:
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

        loaded_context_contract = _validate_loaded_config_contract(config, model.config, request)
        report["request"]["validated_loaded_max_position_embeddings"] = loaded_context_contract[
            "max_position_embeddings"
        ]

        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if parameter_devices != [str(device)]:
            raise RuntimeError(f"model must be on one exact device; observed {parameter_devices}")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        report["identities"]["loaded_model"] = {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "name_or_path": getattr(model, "name_or_path", None),
            "parameter_count": parameter_count,
            "parameter_devices": parameter_devices,
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "loaded_config": _config_identity(model.config),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
        }

        nominal_dynamic_bytes = _nominal_kv_bytes(
            model.config,
            sequence_tokens=max(request.lengths) + request.max_new_tokens,
            dtype=dtype,
        )
        if "dynamic" in request.modes and nominal_dynamic_bytes is None:
            raise ValueError(
                "dynamic-cache safety guard requires layers, KV heads, and head_dim in config"
            )
        report["memory_contract"] = {
            "nominal_dynamic_kv_tensor_bytes_at_largest_request": nominal_dynamic_bytes,
            "nominal_formula": "2*K/V*layers*kv_heads*head_dim*tokens*dtype_bytes",
            "excludes": "allocator overhead, attention workspace, logits, model weights, and other processes",
            "pre_load_minimum_global_free_bytes": int(args.minimum_free_cuda_gib * GIB),
            "post_load_cuda_headroom_bytes": int(args.cuda_headroom_gib * GIB),
        }
        if device.type == "cuda":
            required_after_load = int(args.cuda_headroom_gib * GIB)
            if "dynamic" in request.modes and nominal_dynamic_bytes is not None:
                required_after_load += nominal_dynamic_bytes
            free_after_load = after_load["cuda"]["device_global_free_bytes"]
            if free_after_load < required_after_load:
                dynamic_text = (
                    f"{nominal_dynamic_bytes / GIB:.2f} GiB nominal KV plus "
                    if "dynamic" in request.modes and nominal_dynamic_bytes is not None
                    else ""
                )
                raise RuntimeError(
                    f"largest request requires {dynamic_text}"
                    f"{args.cuda_headroom_gib:.2f} GiB guard, but only "
                    f"{free_after_load / GIB:.2f} GiB is globally free after model load"
                )

        for length in request.lengths:
            for repeat in range(request.repeats):
                for mode in request.modes:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    attempt: dict[str, Any] = {
                        "execution_ordinal": len(report["attempts"]),
                        "mode": mode,
                        "prompt_tokens": length,
                        "repeat": repeat,
                        "status": "running",
                    }
                    report["attempts"].append(attempt)
                    print(
                        f"Running {mode}: prompt={length:,}, completion="
                        f"{request.max_new_tokens}, repeat={repeat}",
                        flush=True,
                    )
                    cache = None
                    try:
                        cache = DynamicCache(
                            config=model.config,
                            offloading=mode == "offloaded",
                            offload_only_non_sliding=False,
                        )
                        expected_offloading = mode == "offloaded"
                        actual_offloading = getattr(cache, "offloading", None)
                        if actual_offloading is not expected_offloading:
                            raise RuntimeError(
                                "DynamicCache did not honor the requested offloading mode"
                            )
                        if (
                            expected_offloading
                            and getattr(cache, "only_non_sliding", None) is not False
                        ):
                            raise RuntimeError(
                                "DynamicCache did not honor offload_only_non_sliding=False"
                            )
                        cache_contract = {
                            "mode": mode,
                            "class": f"{type(cache).__module__}.{type(cache).__qualname__}",
                            "offloading": actual_offloading,
                            "offload_only_non_sliding": bool(
                                getattr(cache, "only_non_sliding", False)
                            ),
                            "payload": (
                                "exact model-produced K/V; no quantization or approximation"
                            ),
                        }
                        result = _manual_greedy_decode(
                            model,
                            prompt_ids=prompts[length],
                            cache=cache,
                            max_new_tokens=request.max_new_tokens,
                            device=device,
                            telemetry=telemetry,
                        )
                        result.update(
                            {
                                "repeat": repeat,
                                "prompt_token_ids_sha256": _token_id_sha256(prompts[length]),
                                "cache": cache_contract,
                            }
                        )
                        report["runs"].append(result)
                    except (Exception, KeyboardInterrupt) as error:
                        attempt["status"] = "failed"
                        attempt["error"] = _error_record(error)
                        raise
                    else:
                        attempt["status"] = "completed"
                        attempt["run_index"] = len(report["runs"]) - 1
                    finally:
                        if cache is not None:
                            del cache

        if is_local:
            recheck_started = time.perf_counter()
            rechecked_artifact = _model_artifact_identity(model_id, is_local=True, revision=None)
            original_fingerprint = report["identities"]["model_artifact"].get("stack_fingerprint")
            rechecked_fingerprint = rechecked_artifact.get("stack_fingerprint")
            report["identities"]["model_artifact_recheck"] = {
                "seconds": time.perf_counter() - recheck_started,
                "stack_fingerprint": rechecked_fingerprint,
                "matches_preload_identity": rechecked_fingerprint == original_fingerprint,
            }
            if rechecked_fingerprint != original_fingerprint:
                raise RuntimeError("local model artifact identity changed during the run")

        report["token_identity"] = _compare_run_token_identity(
            report["runs"], expected_request=request
        )
        if report["token_identity"]["failure_count"]:
            raise ExactnessError(
                f"{report['token_identity']['failure_count']} exact token identity comparisons failed"
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = _make_output_path(args)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": started_at,
        "finished_at": None,
        "claim_scope": {
            "exact_cache_modes_only": True,
            "live_cache_modes": ["GPU-resident DynamicCache", "CPU-offloaded DynamicCache"],
            "snapshot_or_archive": False,
            "nvme_or_disk_backed_decode": False,
            "sparse_or_compressed_cache": False,
            "context_extension_claim": False,
            "position_contract": "no request may exceed model max_position_embeddings",
        },
        "request": {
            "model": args.model,
            "revision": args.revision,
            "allow_download": args.allow_download,
            "modes": list(args.modes),
            "lengths": list(args.lengths),
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "logits_to_keep": 1,
            "mode_order": "declared CLI order inside each length and repeat; no warm-up",
            "minimum_free_cuda_gib": _finite_float_or_text(args.minimum_free_cuda_gib),
            "cuda_headroom_gib": _finite_float_or_text(args.cuda_headroom_gib),
        },
        "attempts": [],
        "runs": [],
        "errors": [],
    }
    exit_code = 0
    try:
        _run(args, report)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["errors"].append(_error_record(error))
        exit_code = 1
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sealed = _seal_report(report)
    try:
        _write_json_exclusive(output_path, sealed)
    except FileExistsError:
        print(f"Refusing to overwrite existing report: {output_path}", file=sys.stderr)
        return 1
    print(
        f"Wrote exclusively created, content-sealed exact-context evidence to {output_path}",
        flush=True,
    )
    if exit_code:
        print(
            f"Exact-context run failed: {report['errors'][-1]['message']}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
