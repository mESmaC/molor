"""Exact append-only KV cache with one staging pair shared by every layer.

This is an isolated systems prototype for batch-1 Qwen-style attention under the
Transformers 5.15 ``Cache``/``CacheLayerMixin`` contract.  Each layer owns exact
CPU K/V backing tensors, while decode copies only the used prefix into one shared
device staging pair.  Layers must update sequentially, so the next layer cannot
overwrite staging until the previous layer's attention work has been enqueued.

The module does not load or run a model and is not wired into the live runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

try:  # Transformers is an optional molor dependency.
    from transformers.cache_utils import Cache as _TransformersCache
    from transformers.cache_utils import CacheLayerMixin as _TransformersCacheLayerMixin

    TRANSFORMERS_CACHE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    TRANSFORMERS_CACHE_AVAILABLE = False

    class _TransformersCacheLayerMixin:
        is_compileable = False
        supports_early_init = True

        def __init__(self, **_kwargs: object) -> None:
            self.keys: Tensor | None = None
            self.values: Tensor | None = None
            self.is_initialized = False

    class _TransformersCache:
        def __init__(
            self,
            *,
            layers: list[Any],
            offloading: bool = False,
            **_kwargs: object,
        ) -> None:
            self.layers = layers
            self.offloading = offloading


TELEMETRY_SCHEMA = "molor.shared-staging-cache-telemetry.v1"


class SharedStagingCacheError(RuntimeError):
    """Base error for a violated exact-cache invariant."""


class UnsupportedSharedStagingOperation(SharedStagingCacheError):
    """An operation would violate append-only batch-1 semantics."""


class TensorAllocator(Protocol):
    """Injectable allocation seam used by CPU-only tests."""

    def __call__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> Tensor: ...


class TensorCopier(Protocol):
    """Injectable copy seam used to prove transfer order without CUDA."""

    def __call__(
        self,
        destination: Tensor,
        source: Tensor,
        *,
        non_blocking: bool,
    ) -> None: ...


def default_tensor_allocator(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    pin_memory: bool,
) -> Tensor:
    """Allocate one exclusive strided tensor with the requested placement."""

    if pin_memory and device.type != "cpu":
        raise ValueError("pin_memory is valid only for CPU backing")
    return torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)


def default_tensor_copier(
    destination: Tensor,
    source: Tensor,
    *,
    non_blocking: bool,
) -> None:
    """Copy on PyTorch's current stream, preserving call order."""

    destination.copy_(source, non_blocking=non_blocking)


def _plain_positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _plain_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class SharedStagingGeometry:
    """Exact tensor and capacity contract shared by every cache layer."""

    num_layers: int
    max_cache_len: int
    num_key_value_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: torch.dtype
    staging_device: torch.device
    pin_cpu_backing: bool = True
    non_blocking_copies: bool = True

    def __post_init__(self) -> None:
        for field in (
            "num_layers",
            "max_cache_len",
            "num_key_value_heads",
            "key_head_dim",
            "value_head_dim",
        ):
            _plain_positive_int(getattr(self, field), field)
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not self.dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        if not isinstance(self.staging_device, torch.device):
            raise TypeError("staging_device must be a torch.device")
        if self.staging_device.type not in {"cpu", "cuda"}:
            raise ValueError("staging_device must be CPU or CUDA")
        if self.staging_device.type == "cuda" and self.staging_device.index is None:
            raise ValueError("CUDA staging_device must include an explicit device index")
        if not isinstance(self.pin_cpu_backing, bool):
            raise TypeError("pin_cpu_backing must be boolean")
        if not isinstance(self.non_blocking_copies, bool):
            raise TypeError("non_blocking_copies must be boolean")

    def key_shape(self) -> tuple[int, int, int, int]:
        return (1, self.num_key_value_heads, self.max_cache_len, self.key_head_dim)

    def value_shape(self) -> tuple[int, int, int, int]:
        return (1, self.num_key_value_heads, self.max_cache_len, self.value_head_dim)

    def to_dict(self) -> dict[str, object]:
        return {
            "num_layers": self.num_layers,
            "max_cache_len": self.max_cache_len,
            "num_key_value_heads": self.num_key_value_heads,
            "key_head_dim": self.key_head_dim,
            "value_head_dim": self.value_head_dim,
            "dtype": _dtype_name(self.dtype),
            "staging_device": str(self.staging_device),
            "pin_cpu_backing": self.pin_cpu_backing,
            "non_blocking_copies": self.non_blocking_copies,
            "batch_size": 1,
            "layout": "batch_heads_sequence_head_dim",
        }


@dataclass(frozen=True, slots=True)
class TransferEvent:
    """One successfully submitted exact tensor copy, without timestamps or pointers."""

    ordinal: int
    layer_idx: int
    phase: str
    tensor_role: str
    source_device: str
    destination_device: str
    direction: str
    elements: int
    bytes: int
    non_blocking: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "layer_idx": self.layer_idx,
            "phase": self.phase,
            "tensor_role": self.tensor_role,
            "source_device": self.source_device,
            "destination_device": self.destination_device,
            "direction": self.direction,
            "elements": self.elements,
            "bytes": self.bytes,
            "non_blocking": self.non_blocking,
        }


class _TransferLedger:
    def __init__(self, *, event_capacity: int) -> None:
        self.event_capacity = _plain_nonnegative_int(event_capacity, "telemetry_event_capacity")
        self._events: deque[TransferEvent] = deque(maxlen=self.event_capacity)
        self._copies = 0
        self._bytes = 0
        self._totals_by_direction: dict[str, dict[str, int]] = {}
        self._totals_by_phase: dict[str, dict[str, int]] = {}

    def record(
        self,
        *,
        layer_idx: int,
        phase: str,
        tensor_role: str,
        source: Tensor,
        destination: Tensor,
        non_blocking: bool,
    ) -> None:
        source_device = str(source.device)
        destination_device = str(destination.device)
        event = TransferEvent(
            ordinal=self._copies,
            layer_idx=layer_idx,
            phase=phase,
            tensor_role=tensor_role,
            source_device=source_device,
            destination_device=destination_device,
            direction=f"{source_device}_to_{destination_device}",
            elements=source.numel(),
            bytes=source.numel() * source.element_size(),
            non_blocking=non_blocking,
        )
        self._copies += 1
        self._bytes += event.bytes
        for totals, key in (
            (self._totals_by_direction, event.direction),
            (self._totals_by_phase, event.phase),
        ):
            bucket = totals.setdefault(key, {"copies": 0, "bytes": 0})
            bucket["copies"] += 1
            bucket["bytes"] += event.bytes
        self._events.append(event)

    def snapshot(self) -> dict[str, object]:
        return {
            "copies": self._copies,
            "bytes": self._bytes,
            "by_direction": {
                key: dict(self._totals_by_direction[key])
                for key in sorted(self._totals_by_direction)
            },
            "by_phase": {
                key: dict(self._totals_by_phase[key]) for key in sorted(self._totals_by_phase)
            },
            "event_capacity": self.event_capacity,
            "events_retained": len(self._events),
            "events_dropped": self._copies - len(self._events),
            "events": [event.to_dict() for event in self._events],
        }


def _storage_identity(tensor: Tensor) -> tuple[str, int]:
    return str(tensor.device), int(tensor.untyped_storage().data_ptr())


def _tensor_storage_bytes(tensor: Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.untyped_storage().nbytes())


class _SharedWorkspace:
    def __init__(
        self,
        geometry: SharedStagingGeometry,
        *,
        allocator: TensorAllocator,
        copier: TensorCopier,
        telemetry_event_capacity: int,
    ) -> None:
        self.geometry = geometry
        self.allocator = allocator
        self.copier = copier
        self.ledger = _TransferLedger(event_capacity=telemetry_event_capacity)
        self.staging_keys: Tensor | None = None
        self.staging_values: Tensor | None = None
        self.staging_allocation_count = 0
        self._owned_storage: set[tuple[str, int]] = set()
        self._cuda_stream_identity: int | None = None
        self._poison_reason: str | None = None
        self._poison_count = 0
        self._poison_counts_by_reason: dict[str, int] = {}

    @property
    def is_poisoned(self) -> bool:
        return self._poison_reason is not None

    def poison(self, reason: str) -> None:
        if self._poison_reason is not None:
            return
        self._poison_reason = reason
        self._poison_count += 1
        self._poison_counts_by_reason[reason] = self._poison_counts_by_reason.get(reason, 0) + 1

    def require_healthy(self) -> None:
        if self.is_poisoned:
            raise SharedStagingCacheError(
                "cache is poisoned by a partial failure; call reset before reuse"
            )

    def clear_poison(self) -> None:
        self._poison_reason = None

    def health_snapshot(self) -> dict[str, object]:
        return {
            "poisoned": self.is_poisoned,
            "active_poison_reason": self._poison_reason,
            "poison_events": self._poison_count,
            "poison_events_by_reason": {
                key: self._poison_counts_by_reason[key]
                for key in sorted(self._poison_counts_by_reason)
            },
            "cuda_stream_contract": "one_stable_current_stream",
            "cuda_stream_bound": self._cuda_stream_identity is not None,
        }

    def bind_current_stream(self) -> None:
        if self.geometry.staging_device.type != "cuda":
            return
        stream_identity = int(torch.cuda.current_stream(self.geometry.staging_device).cuda_stream)
        if self._cuda_stream_identity is None:
            self._cuda_stream_identity = stream_identity
        elif self._cuda_stream_identity != stream_identity:
            raise UnsupportedSharedStagingOperation(
                "cross-stream updates are unsupported; use one stable current CUDA stream"
            )

    def _allocate(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        pin_memory: bool,
        owner: str,
    ) -> Tensor:
        tensor = self.allocator(
            shape,
            dtype=self.geometry.dtype,
            device=device,
            pin_memory=pin_memory,
        )
        if not isinstance(tensor, Tensor):
            raise SharedStagingCacheError(f"allocator returned a non-tensor for {owner}")
        if tuple(tensor.shape) != shape:
            raise SharedStagingCacheError(f"allocator returned the wrong shape for {owner}")
        if tensor.dtype != self.geometry.dtype:
            raise SharedStagingCacheError(f"allocator returned the wrong dtype for {owner}")
        if tensor.device != device:
            raise SharedStagingCacheError(f"allocator returned the wrong device for {owner}")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise SharedStagingCacheError(f"allocator must return a contiguous strided {owner}")
        if tensor.storage_offset() != 0:
            raise SharedStagingCacheError(f"allocator returned an offset view for {owner}")
        expected_bytes = tensor.numel() * tensor.element_size()
        if tensor.untyped_storage().nbytes() != expected_bytes:
            raise SharedStagingCacheError(
                f"allocator returned shared or oversized storage for {owner}"
            )
        if pin_memory and not tensor.is_pinned():
            raise SharedStagingCacheError(
                f"allocator did not pin requested CPU storage for {owner}"
            )
        identity = _storage_identity(tensor)
        if identity in self._owned_storage:
            raise SharedStagingCacheError(f"allocator aliased cache-owned storage for {owner}")
        return tensor

    def _commit_pair(self, keys: Tensor, values: Tensor, *, owner: str) -> None:
        identities = (_storage_identity(keys), _storage_identity(values))
        if identities[0] == identities[1]:
            raise SharedStagingCacheError(f"allocator aliased key/value storage for {owner}")
        if any(identity in self._owned_storage for identity in identities):
            raise SharedStagingCacheError(f"allocator aliased cache-owned storage for {owner}")
        self._owned_storage.update(identities)

    def allocate_backing(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        keys = self._allocate(
            self.geometry.key_shape(),
            device=torch.device("cpu"),
            pin_memory=self.geometry.pin_cpu_backing,
            owner=f"layer {layer_idx} key backing",
        )
        values = self._allocate(
            self.geometry.value_shape(),
            device=torch.device("cpu"),
            pin_memory=self.geometry.pin_cpu_backing,
            owner=f"layer {layer_idx} value backing",
        )
        self._commit_pair(keys, values, owner=f"layer {layer_idx} backing")
        return keys, values

    def ensure_staging(self) -> tuple[Tensor, Tensor]:
        if (self.staging_keys is None) != (self.staging_values is None):
            raise SharedStagingCacheError("shared staging pair is only partially allocated")
        if self.staging_keys is None:
            staging_keys = self._allocate(
                self.geometry.key_shape(),
                device=self.geometry.staging_device,
                pin_memory=False,
                owner="shared key staging",
            )
            staging_values = self._allocate(
                self.geometry.value_shape(),
                device=self.geometry.staging_device,
                pin_memory=False,
                owner="shared value staging",
            )
            self._commit_pair(staging_keys, staging_values, owner="shared staging")
            self.staging_keys = staging_keys
            self.staging_values = staging_values
            self.staging_allocation_count += 1
        assert self.staging_values is not None
        return self.staging_keys, self.staging_values

    def copy(
        self,
        destination: Tensor,
        source: Tensor,
        *,
        layer_idx: int,
        phase: str,
        tensor_role: str,
    ) -> None:
        if tuple(destination.shape) != tuple(source.shape):
            raise SharedStagingCacheError("copy source and destination shapes differ")
        if destination.dtype != source.dtype or destination.dtype != self.geometry.dtype:
            raise SharedStagingCacheError("copy source and destination dtypes differ")
        non_blocking = self.geometry.non_blocking_copies
        self.copier(destination, source, non_blocking=non_blocking)
        self.ledger.record(
            layer_idx=layer_idx,
            phase=phase,
            tensor_role=tensor_role,
            source=source,
            destination=destination,
            non_blocking=non_blocking,
        )

    def staging_pointers(self) -> tuple[int, int] | None:
        if self.staging_keys is None or self.staging_values is None:
            return None
        return (
            int(self.staging_keys.untyped_storage().data_ptr()),
            int(self.staging_values.untyped_storage().data_ptr()),
        )


class SharedStagingCacheLayer(_TransformersCacheLayerMixin):
    """One exact CPU-backed layer using a cache-wide staging workspace."""

    is_compileable = False
    is_sliding = False
    supports_early_init = True

    def __init__(
        self,
        *,
        geometry: SharedStagingGeometry,
        layer_idx: int,
        workspace: _SharedWorkspace,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.layer_idx = _plain_nonnegative_int(layer_idx, "layer_idx")
        if self.layer_idx >= geometry.num_layers:
            raise ValueError("layer_idx exceeds configured layer count")
        if not isinstance(workspace, _SharedWorkspace) or workspace.geometry != geometry:
            raise TypeError("workspace must carry the identical geometry contract")
        self._workspace = workspace
        self.dtype = geometry.dtype
        self.device = geometry.staging_device
        self.batch_size = 1
        self.num_heads = geometry.num_key_value_heads
        self.k_head_dim = geometry.key_head_dim
        self.v_head_dim = geometry.value_head_dim
        self.cumulative_length = 0

    def _validate_states(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *,
        allow_empty_sequence: bool,
    ) -> int:
        if not isinstance(key_states, Tensor) or not isinstance(value_states, Tensor):
            raise TypeError("key_states and value_states must be tensors")
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("K/V must have shape [batch,heads,sequence,head_dim]")
        if key_states.shape[0] != 1 or value_states.shape[0] != 1:
            raise UnsupportedSharedStagingOperation("multibatch cache updates are unsupported")
        if key_states.shape[1] != self.geometry.num_key_value_heads:
            raise ValueError("key_states has the wrong KV-head count")
        if value_states.shape[1] != self.geometry.num_key_value_heads:
            raise ValueError("value_states has the wrong KV-head count")
        if key_states.shape[-1] != self.geometry.key_head_dim:
            raise ValueError("key_states has the wrong head dimension")
        if value_states.shape[-1] != self.geometry.value_head_dim:
            raise ValueError("value_states has the wrong head dimension")
        if key_states.shape[-2] != value_states.shape[-2]:
            raise ValueError("key_states and value_states sequence lengths differ")
        sequence_length = int(key_states.shape[-2])
        if sequence_length == 0 and not allow_empty_sequence:
            raise ValueError("cache update sequence length must be positive")
        if key_states.dtype != self.geometry.dtype or value_states.dtype != self.geometry.dtype:
            raise ValueError("K/V dtype differs from the frozen cache dtype")
        if key_states.device != self.geometry.staging_device:
            raise ValueError("key_states device differs from staging_device")
        if value_states.device != self.geometry.staging_device:
            raise ValueError("value_states device differs from staging_device")
        if key_states.layout != torch.strided or value_states.layout != torch.strided:
            raise ValueError("K/V must use strided tensor layout")
        return sequence_length

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        self._workspace.require_healthy()
        if self.is_initialized:
            raise SharedStagingCacheError("cache layer is already initialized")
        self._validate_states(key_states, value_states, allow_empty_sequence=True)
        try:
            self.keys, self.values = self._workspace.allocate_backing(self.layer_idx)
        except Exception as exc:
            self._workspace.poison("backing_allocation_failure")
            raise SharedStagingCacheError(
                "cache backing allocation failed; reset or discard this cache"
            ) from exc
        self.is_initialized = True

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        self._workspace.require_healthy()
        if args or kwargs:
            raise UnsupportedSharedStagingOperation(
                "cache kwargs/positions are unsupported; this cache is append-only"
            )
        sequence_length = self._validate_states(
            key_states,
            value_states,
            allow_empty_sequence=False,
        )
        if self.cumulative_length > 0 and sequence_length != 1:
            raise UnsupportedSharedStagingOperation(
                "after prefill, only single-token append decode is supported"
            )
        new_length = self.cumulative_length + sequence_length
        if new_length > self.geometry.max_cache_len:
            raise SharedStagingCacheError("cache capacity exceeded")
        self._workspace.bind_current_stream()
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        assert self.keys is not None and self.values is not None
        destination = slice(self.cumulative_length, new_length)
        phase = "prefill_to_backing" if self.cumulative_length == 0 else "decode_to_backing"
        try:
            self._workspace.copy(
                self.keys[..., destination, :],
                key_states,
                layer_idx=self.layer_idx,
                phase=phase,
                tensor_role="key",
            )
            self._workspace.copy(
                self.values[..., destination, :],
                value_states,
                layer_idx=self.layer_idx,
                phase=phase,
                tensor_role="value",
            )
        except Exception as exc:
            self._workspace.poison("backing_copy_failure")
            raise SharedStagingCacheError(
                "cache backing copy failed; call reset before reuse"
            ) from exc

        if self.cumulative_length == 0:
            self.cumulative_length = new_length
            return key_states, value_states

        try:
            staging_keys, staging_values = self._workspace.ensure_staging()
            self._workspace.copy(
                staging_keys[..., :new_length, :],
                self.keys[..., :new_length, :],
                layer_idx=self.layer_idx,
                phase="backing_to_staging",
                tensor_role="key",
            )
            self._workspace.copy(
                staging_values[..., :new_length, :],
                self.values[..., :new_length, :],
                layer_idx=self.layer_idx,
                phase="backing_to_staging",
                tensor_role="value",
            )
        except Exception as exc:
            self._workspace.poison("staging_allocation_or_copy_failure")
            raise SharedStagingCacheError("shared staging failed; call reset before reuse") from exc
        self.cumulative_length = new_length
        return staging_keys[..., :new_length, :], staging_values[..., :new_length, :]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        self._workspace.require_healthy()
        query_length = _plain_positive_int(query_length, "query_length")
        kv_length = self.cumulative_length + query_length
        if kv_length > self.geometry.max_cache_len:
            raise SharedStagingCacheError("attention mask request exceeds cache capacity")
        return kv_length, 0

    def get_seq_length(self) -> int:
        self._workspace.require_healthy()
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.geometry.max_cache_len

    def reset(self) -> None:
        """Reset logical length while deliberately retaining stale allocated bytes."""

        self.cumulative_length = 0

    def reorder_cache(self, _beam_idx: Tensor) -> None:
        raise UnsupportedSharedStagingOperation("beam reordering is unsupported")

    def crop(self, _tokens_to_remove: int) -> None:
        raise UnsupportedSharedStagingOperation("cache cropping is unsupported")

    def batch_repeat_interleave(self, _repeats: int) -> None:
        raise UnsupportedSharedStagingOperation("batch expansion is unsupported")

    def batch_select_indices(self, _indices: Tensor) -> None:
        raise UnsupportedSharedStagingOperation("batch selection is unsupported")

    def offload(self) -> None:
        raise UnsupportedSharedStagingOperation("layer backing is permanently CPU-owned")

    def prefetch(self) -> None:
        raise UnsupportedSharedStagingOperation("prefetch is replaced by shared decode staging")


class SharedStagingCache(_TransformersCache):
    """Transformers-compatible container enforcing sequential shared-stage reuse."""

    def __init__(
        self,
        *,
        num_layers: int,
        max_cache_len: int,
        num_key_value_heads: int,
        key_head_dim: int,
        value_head_dim: int | None = None,
        dtype: torch.dtype,
        staging_device: torch.device | str,
        pin_cpu_backing: bool = True,
        non_blocking_copies: bool = True,
        telemetry_event_capacity: int = 256,
        allocator: TensorAllocator = default_tensor_allocator,
        copier: TensorCopier = default_tensor_copier,
    ) -> None:
        if value_head_dim is None:
            value_head_dim = key_head_dim
        device = torch.device(staging_device)
        self.geometry = SharedStagingGeometry(
            num_layers=num_layers,
            max_cache_len=max_cache_len,
            num_key_value_heads=num_key_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            dtype=dtype,
            staging_device=device,
            pin_cpu_backing=pin_cpu_backing,
            non_blocking_copies=non_blocking_copies,
        )
        if not callable(allocator) or not callable(copier):
            raise TypeError("allocator and copier must be callable")
        telemetry_event_capacity = _plain_nonnegative_int(
            telemetry_event_capacity,
            "telemetry_event_capacity",
        )
        self._workspace = _SharedWorkspace(
            self.geometry,
            allocator=allocator,
            copier=copier,
            telemetry_event_capacity=telemetry_event_capacity,
        )
        layers = [
            SharedStagingCacheLayer(
                geometry=self.geometry,
                layer_idx=layer_idx,
                workspace=self._workspace,
            )
            for layer_idx in range(self.geometry.num_layers)
        ]
        super().__init__(layers=layers, offloading=False)
        self._expected_layer_idx = 0
        self._cycle_start_length: int | None = None
        self._cycle_query_length: int | None = None
        self._update_counts = [0] * self.geometry.num_layers
        self._prefill_counts = [0] * self.geometry.num_layers
        self._decode_counts = [0] * self.geometry.num_layers
        self._reset_count = 0

    def _layer(self, layer_idx: int) -> SharedStagingCacheLayer:
        layer_idx = _plain_nonnegative_int(layer_idx, "layer_idx")
        if layer_idx >= len(self.layers):
            raise IndexError("layer_idx exceeds configured layer count")
        layer = self.layers[layer_idx]
        if not isinstance(layer, SharedStagingCacheLayer):
            raise SharedStagingCacheError("cache layer type changed after construction")
        return layer

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        self._workspace.require_healthy()
        partial_cycle = self._expected_layer_idx != 0
        try:
            return self._update_impl(
                key_states,
                value_states,
                layer_idx,
                *args,
                **kwargs,
            )
        except Exception:
            if partial_cycle:
                self._workspace.poison("partial_update_cycle_failure")
            elif not self._workspace.is_poisoned:
                self._cycle_start_length = None
                self._cycle_query_length = None
            raise

    def _update_impl(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        if args or kwargs:
            raise UnsupportedSharedStagingOperation(
                "cache kwargs/positions are unsupported; this cache is append-only"
            )
        layer = self._layer(layer_idx)
        if layer_idx != self._expected_layer_idx:
            raise SharedStagingCacheError(
                f"layers must update sequentially; expected {self._expected_layer_idx}, got {layer_idx}"
            )
        query_length = layer._validate_states(  # noqa: SLF001 - same-module invariant check
            key_states,
            value_states,
            allow_empty_sequence=False,
        )
        if self._expected_layer_idx == 0:
            lengths = [candidate.get_seq_length() for candidate in self.layers]
            if len(set(lengths)) != 1:
                raise SharedStagingCacheError(
                    "layer logical lengths diverged before an update cycle"
                )
            self._cycle_start_length = lengths[0]
            self._cycle_query_length = query_length
        else:
            if query_length != self._cycle_query_length:
                raise SharedStagingCacheError("query length changed within one layer update cycle")
            if layer.get_seq_length() != self._cycle_start_length:
                raise SharedStagingCacheError("layer length changed before its sequential update")

        was_prefill = layer.get_seq_length() == 0
        result = layer.update(key_states, value_states)
        self._update_counts[layer_idx] += 1
        if was_prefill:
            self._prefill_counts[layer_idx] += 1
        else:
            self._decode_counts[layer_idx] += 1

        self._expected_layer_idx = (layer_idx + 1) % self.geometry.num_layers
        if self._expected_layer_idx == 0:
            assert self._cycle_start_length is not None
            assert self._cycle_query_length is not None
            expected_length = self._cycle_start_length + self._cycle_query_length
            if any(candidate.get_seq_length() != expected_length for candidate in self.layers):
                raise SharedStagingCacheError(
                    "layer logical lengths diverged after an update cycle"
                )
            self._cycle_start_length = None
            self._cycle_query_length = None
        return result

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int | Sequence[int],
        head_dim: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        batch_size = _plain_positive_int(batch_size, "batch_size")
        if batch_size != 1:
            raise UnsupportedSharedStagingOperation(
                "multibatch early initialization is unsupported"
            )
        expected_heads = [self.geometry.num_key_value_heads] * self.geometry.num_layers
        heads = self._normalize_early_dimension(num_heads, "num_heads")
        if heads != expected_heads:
            raise ValueError("early initialization KV-head geometry differs")
        expected_dims = [self.geometry.key_head_dim] * self.geometry.num_layers
        dims = self._normalize_early_dimension(head_dim, "head_dim")
        if dims != expected_dims or self.geometry.key_head_dim != self.geometry.value_head_dim:
            raise ValueError("early initialization head geometry differs")
        if dtype != self.geometry.dtype or torch.device(device) != self.geometry.staging_device:
            raise ValueError("early initialization dtype/device differs")
        fake_keys = torch.empty(
            (1, self.geometry.num_key_value_heads, 0, self.geometry.key_head_dim),
            dtype=dtype,
            device=device,
        )
        fake_values = torch.empty(
            (1, self.geometry.num_key_value_heads, 0, self.geometry.value_head_dim),
            dtype=dtype,
            device=device,
        )
        for layer in self.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(fake_keys, fake_values)

    def _normalize_early_dimension(
        self,
        value: int | Sequence[int],
        field: str,
    ) -> list[int]:
        if isinstance(value, int):
            item = _plain_positive_int(value, field)
            return [item] * self.geometry.num_layers
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{field} must be an integer or per-layer sequence")
        result = [
            _plain_positive_int(item, f"{field}[{index}]") for index, item in enumerate(value)
        ]
        if len(result) != self.geometry.num_layers:
            raise ValueError(f"{field} per-layer sequence has the wrong length")
        return result

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._layer(layer_idx).get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self._layer(layer_idx).get_mask_sizes(query_length)

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_max_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None:
            self._layer(layer_idx)
        return self.geometry.max_cache_len

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def is_initialized(self) -> bool:
        return all(layer.is_initialized for layer in self.layers)

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * self.geometry.num_layers

    @property
    def is_linear(self) -> list[bool]:
        return [False] * self.geometry.num_layers

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self._expected_layer_idx = 0
        self._cycle_start_length = None
        self._cycle_query_length = None
        self._workspace.clear_poison()
        self._reset_count += 1

    def reorder_cache(self, _beam_idx: Tensor) -> None:
        raise UnsupportedSharedStagingOperation("beam reordering is unsupported")

    def crop(self, _tokens_to_remove: int) -> None:
        raise UnsupportedSharedStagingOperation("cache cropping is unsupported")

    def batch_repeat_interleave(self, _repeats: int) -> None:
        raise UnsupportedSharedStagingOperation("batch expansion is unsupported")

    def batch_select_indices(self, _indices: Tensor) -> None:
        raise UnsupportedSharedStagingOperation("batch selection is unsupported")

    def offload(self, _layer_idx: int, _only_non_sliding: bool = True) -> None:
        raise UnsupportedSharedStagingOperation("layer backing is permanently CPU-owned")

    def prefetch(self, _layer_idx: int, _only_non_sliding: bool = True) -> None:
        raise UnsupportedSharedStagingOperation("prefetch is replaced by shared decode staging")

    def staging_data_ptrs(self) -> tuple[int, int] | None:
        """Return staging storage pointers for explicit reuse tests, not telemetry."""

        return self._workspace.staging_pointers()

    def backing_data_ptrs(self) -> tuple[tuple[int, int] | None, ...]:
        """Return per-layer backing pointers for explicit non-alias tests."""

        result: list[tuple[int, int] | None] = []
        for layer in self.layers:
            if layer.keys is None or layer.values is None:
                result.append(None)
            else:
                result.append(
                    (
                        int(layer.keys.untyped_storage().data_ptr()),
                        int(layer.values.untyped_storage().data_ptr()),
                    )
                )
        return tuple(result)

    def telemetry(self) -> dict[str, object]:
        """Return deterministic logical/storage/transfer evidence without pointers."""

        layer_backing_bytes: list[dict[str, object]] = []
        backing_bytes = 0
        for layer in self.layers:
            key_bytes = _tensor_storage_bytes(layer.keys)
            value_bytes = _tensor_storage_bytes(layer.values)
            backing_bytes += key_bytes + value_bytes
            layer_backing_bytes.append(
                {
                    "layer_idx": layer.layer_idx,
                    "initialized": layer.is_initialized,
                    "logical_length": layer.cumulative_length,
                    "key_bytes": key_bytes,
                    "value_bytes": value_bytes,
                    "total_bytes": key_bytes + value_bytes,
                }
            )
        staging_key_bytes = _tensor_storage_bytes(self._workspace.staging_keys)
        staging_value_bytes = _tensor_storage_bytes(self._workspace.staging_values)
        staging_bytes = staging_key_bytes + staging_value_bytes
        payload: dict[str, object] = {
            "schema_version": TELEMETRY_SCHEMA,
            "geometry": self.geometry.to_dict(),
            "health": self._workspace.health_snapshot(),
            "sequence": {
                "layer_logical_lengths": [layer.cumulative_length for layer in self.layers],
                "expected_next_layer_idx": self._expected_layer_idx,
                "update_cycle_in_progress": self._cycle_start_length is not None,
            },
            "storage": {
                "backing_device": "cpu",
                "backing_bytes": backing_bytes,
                "layer_backing": layer_backing_bytes,
                "staging_device": str(self.geometry.staging_device),
                "staging_allocated": self._workspace.staging_keys is not None,
                "staging_key_bytes": staging_key_bytes,
                "staging_value_bytes": staging_value_bytes,
                "staging_bytes": staging_bytes,
                "staging_pair_allocations": self._workspace.staging_allocation_count,
                "live_cache_bytes": backing_bytes + staging_bytes,
            },
            "operations": {
                "updates_by_layer": list(self._update_counts),
                "prefills_by_layer": list(self._prefill_counts),
                "decodes_by_layer": list(self._decode_counts),
                "resets": self._reset_count,
            },
            "transfers": self._workspace.ledger.snapshot(),
        }
        return payload
