"""Exact append-only KV cache with paged CPU backing and one shared stage.

This isolated v4 prototype keeps every layer's exact K/V state in fixed-size
CPU pages.  Decode reconstructs the used prefix in a single cache-wide staging
pair.  It is deliberately batch-1, append-only, and sequential by layer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

try:
    from transformers.cache_utils import Cache as _TransformersCache
    from transformers.cache_utils import CacheLayerMixin as _TransformersCacheLayerMixin

    TRANSFORMERS_CACHE_AVAILABLE = True
except ImportError:  # pragma: no cover
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
            self, *, layers: list[Any], offloading: bool = False, **_kwargs: object
        ) -> None:
            self.layers = layers
            self.offloading = offloading


TELEMETRY_SCHEMA = "molor.paged-shared-staging-cache-telemetry.v1"


class PagedSharedStagingCacheError(RuntimeError):
    """A paged exact-cache invariant was violated."""


class UnsupportedPagedSharedStagingOperation(PagedSharedStagingCacheError):
    """The requested operation is outside the append-only batch-1 contract."""


class TensorAllocator(Protocol):
    def __call__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> Tensor: ...


class TensorCopier(Protocol):
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
    if pin_memory and device.type != "cpu":
        raise ValueError("pin_memory is valid only for CPU backing")
    return torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)


def default_tensor_copier(
    destination: Tensor,
    source: Tensor,
    *,
    non_blocking: bool,
) -> None:
    destination.copy_(source, non_blocking=non_blocking)


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def power_of_two_ceil(value: int) -> int:
    """Return the CUDACachingHostAllocator size-class projection for one request."""

    value = _positive_int(value, "value")
    return 1 << (value - 1).bit_length()


def _storage_identity(tensor: Tensor) -> tuple[str, int]:
    return str(tensor.device), int(tensor.untyped_storage().data_ptr())


def _storage_bytes(tensor: Tensor | None) -> int:
    return 0 if tensor is None else int(tensor.untyped_storage().nbytes())


@dataclass(frozen=True, slots=True)
class PagedSharedStagingGeometry:
    """Frozen physical and logical cache geometry."""

    num_layers: int
    max_cache_len: int
    page_token_capacity: int
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
            "page_token_capacity",
            "num_key_value_heads",
            "key_head_dim",
            "value_head_dim",
        ):
            _positive_int(getattr(self, field), field)
        if self.max_cache_len % self.page_token_capacity:
            raise ValueError("max_cache_len must be divisible by page_token_capacity")
        if not _is_power_of_two(self.page_token_capacity):
            raise ValueError("page_token_capacity must be a power of two")
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch.dtype")
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
        if not _is_power_of_two(self.key_page_bytes):
            raise ValueError("key page byte size must be a power of two")
        if not _is_power_of_two(self.value_page_bytes):
            raise ValueError("value page byte size must be a power of two")

    @property
    def page_count(self) -> int:
        return self.max_cache_len // self.page_token_capacity

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def key_page_bytes(self) -> int:
        return (
            self.num_key_value_heads
            * self.page_token_capacity
            * self.key_head_dim
            * self.element_size
        )

    @property
    def value_page_bytes(self) -> int:
        return (
            self.num_key_value_heads
            * self.page_token_capacity
            * self.value_head_dim
            * self.element_size
        )

    @property
    def backing_allocation_count(self) -> int:
        return self.num_layers * self.page_count * 2

    @property
    def logical_backing_bytes(self) -> int:
        return self.num_layers * self.page_count * (self.key_page_bytes + self.value_page_bytes)

    @property
    def predicted_rounded_backing_bytes(self) -> int:
        per_layer = self.page_count * (
            power_of_two_ceil(self.key_page_bytes) + power_of_two_ceil(self.value_page_bytes)
        )
        return self.num_layers * per_layer

    @property
    def predicted_rounding_waste_bytes(self) -> int:
        return self.predicted_rounded_backing_bytes - self.logical_backing_bytes

    @property
    def staging_key_bytes(self) -> int:
        return self.page_count * self.key_page_bytes

    @property
    def staging_value_bytes(self) -> int:
        return self.page_count * self.value_page_bytes

    @property
    def unpaged_predicted_rounded_backing_bytes(self) -> int:
        key_bytes = self.page_count * self.key_page_bytes
        value_bytes = self.page_count * self.value_page_bytes
        return self.num_layers * (power_of_two_ceil(key_bytes) + power_of_two_ceil(value_bytes))

    def key_page_shape(self) -> tuple[int, int, int, int]:
        return (1, self.num_key_value_heads, self.page_token_capacity, self.key_head_dim)

    def value_page_shape(self) -> tuple[int, int, int, int]:
        return (1, self.num_key_value_heads, self.page_token_capacity, self.value_head_dim)

    def staging_key_shape(self) -> tuple[int, int, int, int]:
        return (1, self.num_key_value_heads, self.max_cache_len, self.key_head_dim)

    def staging_value_shape(self) -> tuple[int, int, int, int]:
        return (1, self.num_key_value_heads, self.max_cache_len, self.value_head_dim)

    def to_dict(self) -> dict[str, object]:
        return {
            "num_layers": self.num_layers,
            "max_cache_len": self.max_cache_len,
            "page_token_capacity": self.page_token_capacity,
            "page_count_per_role_per_layer": self.page_count,
            "backing_allocation_count": self.backing_allocation_count,
            "num_key_value_heads": self.num_key_value_heads,
            "key_head_dim": self.key_head_dim,
            "value_head_dim": self.value_head_dim,
            "dtype": str(self.dtype).removeprefix("torch."),
            "staging_device": str(self.staging_device),
            "pin_cpu_backing": self.pin_cpu_backing,
            "non_blocking_copies": self.non_blocking_copies,
            "key_page_bytes": self.key_page_bytes,
            "value_page_bytes": self.value_page_bytes,
            "logical_backing_bytes": self.logical_backing_bytes,
            "predicted_caching_host_allocator_rounded_bytes": (
                self.predicted_rounded_backing_bytes
            ),
            "predicted_rounding_waste_bytes": self.predicted_rounding_waste_bytes,
            "unpaged_predicted_caching_host_allocator_rounded_bytes": (
                self.unpaged_predicted_rounded_backing_bytes
            ),
            "staging_key_bytes": self.staging_key_bytes,
            "staging_value_bytes": self.staging_value_bytes,
            "rounding_model": "PowerOf2Ceil(each_backing_allocation)",
            "layout": "batch_heads_sequence_head_dim",
            "batch_size": 1,
        }


@dataclass(frozen=True, slots=True)
class _TransferEvent:
    ordinal: int
    layer_idx: int
    page_idx: int
    phase: str
    tensor_role: str
    direction: str
    elements: int
    bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "layer_idx": self.layer_idx,
            "page_idx": self.page_idx,
            "phase": self.phase,
            "tensor_role": self.tensor_role,
            "direction": self.direction,
            "elements": self.elements,
            "bytes": self.bytes,
        }


class _TransferLedger:
    def __init__(self, capacity: int) -> None:
        self.capacity = _nonnegative_int(capacity, "telemetry_event_capacity")
        self.events: deque[_TransferEvent] = deque(maxlen=self.capacity)
        self.copies = 0
        self.bytes = 0
        self.by_phase: dict[str, dict[str, int]] = {}

    def record(
        self,
        *,
        layer_idx: int,
        page_idx: int,
        phase: str,
        tensor_role: str,
        source: Tensor,
        destination: Tensor,
    ) -> None:
        size = source.numel() * source.element_size()
        event = _TransferEvent(
            ordinal=self.copies,
            layer_idx=layer_idx,
            page_idx=page_idx,
            phase=phase,
            tensor_role=tensor_role,
            direction=f"{source.device}_to_{destination.device}",
            elements=source.numel(),
            bytes=size,
        )
        self.copies += 1
        self.bytes += size
        bucket = self.by_phase.setdefault(phase, {"copies": 0, "bytes": 0})
        bucket["copies"] += 1
        bucket["bytes"] += size
        self.events.append(event)

    def snapshot(self) -> dict[str, object]:
        return {
            "copies": self.copies,
            "bytes": self.bytes,
            "by_phase": {key: dict(self.by_phase[key]) for key in sorted(self.by_phase)},
            "event_capacity": self.capacity,
            "events_retained": len(self.events),
            "events_dropped": self.copies - len(self.events),
            "events": [event.to_dict() for event in self.events],
        }


class _PagedWorkspace:
    def __init__(
        self,
        geometry: PagedSharedStagingGeometry,
        *,
        allocator: TensorAllocator,
        copier: TensorCopier,
        telemetry_event_capacity: int,
    ) -> None:
        self.geometry = geometry
        self.allocator = allocator
        self.copier = copier
        self.ledger = _TransferLedger(telemetry_event_capacity)
        self.staging_keys: Tensor | None = None
        self.staging_values: Tensor | None = None
        self.staging_pair_allocations = 0
        self.backing_allocation_transactions = 0
        self._owned_storage: set[tuple[str, int]] = set()
        self._stream_identity: int | None = None
        self._poison_reason: str | None = None
        self._poison_events = 0

    @property
    def is_poisoned(self) -> bool:
        return self._poison_reason is not None

    def poison(self, reason: str) -> None:
        if not self.is_poisoned:
            self._poison_reason = reason
            self._poison_events += 1

    def require_healthy(self) -> None:
        if self.is_poisoned:
            raise PagedSharedStagingCacheError(
                "cache is poisoned by a partial failure; call reset before reuse"
            )

    def clear_poison(self) -> None:
        self._poison_reason = None

    def bind_current_stream(self) -> None:
        if self.geometry.staging_device.type != "cuda":
            return
        identity = int(torch.cuda.current_stream(self.geometry.staging_device).cuda_stream)
        if self._stream_identity is None:
            self._stream_identity = identity
        elif identity != self._stream_identity:
            raise UnsupportedPagedSharedStagingOperation(
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
            raise PagedSharedStagingCacheError(f"allocator returned non-tensor for {owner}")
        expected_bytes = tensor.numel() * tensor.element_size()
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != self.geometry.dtype
            or tensor.device != device
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or tensor.untyped_storage().nbytes() != expected_bytes
        ):
            raise PagedSharedStagingCacheError(f"allocator violated exact storage for {owner}")
        if pin_memory and not tensor.is_pinned():
            raise PagedSharedStagingCacheError(f"allocator did not pin {owner}")
        if _storage_identity(tensor) in self._owned_storage:
            raise PagedSharedStagingCacheError(f"allocator aliased owned storage for {owner}")
        return tensor

    def _commit_transaction(self, tensors: Sequence[Tensor], *, owner: str) -> None:
        identities = [_storage_identity(tensor) for tensor in tensors]
        if len(identities) != len(set(identities)):
            raise PagedSharedStagingCacheError(f"allocator aliased transaction storage for {owner}")
        if any(identity in self._owned_storage for identity in identities):
            raise PagedSharedStagingCacheError(f"allocator aliased owned storage for {owner}")
        self._owned_storage.update(identities)

    def allocate_backing_pages(
        self, layer_idx: int
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Allocate and commit one layer's complete K/V page set atomically."""

        keys: list[Tensor] = []
        values: list[Tensor] = []
        for page_idx in range(self.geometry.page_count):
            keys.append(
                self._allocate(
                    self.geometry.key_page_shape(),
                    device=torch.device("cpu"),
                    pin_memory=self.geometry.pin_cpu_backing,
                    owner=f"layer {layer_idx} key page {page_idx}",
                )
            )
            values.append(
                self._allocate(
                    self.geometry.value_page_shape(),
                    device=torch.device("cpu"),
                    pin_memory=self.geometry.pin_cpu_backing,
                    owner=f"layer {layer_idx} value page {page_idx}",
                )
            )
        self._commit_transaction([*keys, *values], owner=f"layer {layer_idx} page set")
        self.backing_allocation_transactions += 1
        return tuple(keys), tuple(values)

    def ensure_staging(self) -> tuple[Tensor, Tensor]:
        if (self.staging_keys is None) != (self.staging_values is None):
            raise PagedSharedStagingCacheError("shared staging pair is partially allocated")
        if self.staging_keys is None:
            keys = self._allocate(
                self.geometry.staging_key_shape(),
                device=self.geometry.staging_device,
                pin_memory=False,
                owner="shared key staging",
            )
            values = self._allocate(
                self.geometry.staging_value_shape(),
                device=self.geometry.staging_device,
                pin_memory=False,
                owner="shared value staging",
            )
            self._commit_transaction((keys, values), owner="shared staging pair")
            self.staging_keys, self.staging_values = keys, values
            self.staging_pair_allocations += 1
        assert self.staging_values is not None
        return self.staging_keys, self.staging_values

    def copy(
        self,
        destination: Tensor,
        source: Tensor,
        *,
        layer_idx: int,
        page_idx: int,
        phase: str,
        tensor_role: str,
    ) -> None:
        if tuple(destination.shape) != tuple(source.shape) or destination.dtype != source.dtype:
            raise PagedSharedStagingCacheError("copy source/destination geometry differs")
        self.copier(
            destination,
            source,
            non_blocking=self.geometry.non_blocking_copies,
        )
        self.ledger.record(
            layer_idx=layer_idx,
            page_idx=page_idx,
            phase=phase,
            tensor_role=tensor_role,
            source=source,
            destination=destination,
        )

    def health(self) -> dict[str, object]:
        return {
            "poisoned": self.is_poisoned,
            "active_poison_reason": self._poison_reason,
            "poison_events": self._poison_events,
            "cuda_stream_contract": "one_stable_current_stream",
            "cuda_stream_bound": self._stream_identity is not None,
        }


class PagedSharedStagingCacheLayer(_TransformersCacheLayerMixin):
    """One exact paged CPU-backed cache layer."""

    is_compileable = False
    is_sliding = False
    supports_early_init = True

    def __init__(
        self,
        *,
        geometry: PagedSharedStagingGeometry,
        layer_idx: int,
        workspace: _PagedWorkspace,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.layer_idx = _nonnegative_int(layer_idx, "layer_idx")
        self._workspace = workspace
        self.key_pages: tuple[Tensor, ...] = ()
        self.value_pages: tuple[Tensor, ...] = ()
        self.dtype = geometry.dtype
        self.device = geometry.staging_device
        self.batch_size = 1
        self.num_heads = geometry.num_key_value_heads
        self.k_head_dim = geometry.key_head_dim
        self.v_head_dim = geometry.value_head_dim
        self.cumulative_length = 0

    def _validate(self, keys: Tensor, values: Tensor, *, allow_empty: bool) -> int:
        if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
            raise TypeError("key_states and value_states must be tensors")
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("K/V must have shape [batch,heads,sequence,head_dim]")
        if keys.shape[0] != 1 or values.shape[0] != 1:
            raise UnsupportedPagedSharedStagingOperation("multibatch updates are unsupported")
        if keys.shape[1] != self.geometry.num_key_value_heads or values.shape[1] != keys.shape[1]:
            raise ValueError("K/V have the wrong KV-head count")
        if keys.shape[-1] != self.geometry.key_head_dim:
            raise ValueError("key_states has the wrong head dimension")
        if values.shape[-1] != self.geometry.value_head_dim:
            raise ValueError("value_states has the wrong head dimension")
        if keys.shape[-2] != values.shape[-2]:
            raise ValueError("K/V sequence lengths differ")
        if not allow_empty and keys.shape[-2] == 0:
            raise ValueError("empty cache updates are unsupported")
        if keys.dtype != self.geometry.dtype or values.dtype != self.geometry.dtype:
            raise ValueError("K/V dtype differs from cache geometry")
        if keys.device != self.geometry.staging_device or values.device != keys.device:
            raise ValueError("K/V device differs from staging device")
        return int(keys.shape[-2])

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        self._workspace.require_healthy()
        if self.is_initialized:
            raise PagedSharedStagingCacheError("cache layer is already initialized")
        self._validate(key_states, value_states, allow_empty=True)
        try:
            pages = self._workspace.allocate_backing_pages(self.layer_idx)
        except Exception as exc:
            self._workspace.poison("backing_allocation_failure")
            raise PagedSharedStagingCacheError(
                "paged backing allocation failed; call reset before reuse"
            ) from exc
        self.key_pages, self.value_pages = pages
        self.is_initialized = True

    def _write(self, keys: Tensor, values: Tensor, start: int, phase: str) -> None:
        page_size = self.geometry.page_token_capacity
        source_start = 0
        remaining = int(keys.shape[-2])
        try:
            while remaining:
                page_idx = (start + source_start) // page_size
                page_offset = (start + source_start) % page_size
                count = min(remaining, page_size - page_offset)
                source_slice = slice(source_start, source_start + count)
                page_slice = slice(page_offset, page_offset + count)
                self._workspace.copy(
                    self.key_pages[page_idx][..., page_slice, :],
                    keys[..., source_slice, :],
                    layer_idx=self.layer_idx,
                    page_idx=page_idx,
                    phase=phase,
                    tensor_role="key",
                )
                self._workspace.copy(
                    self.value_pages[page_idx][..., page_slice, :],
                    values[..., source_slice, :],
                    layer_idx=self.layer_idx,
                    page_idx=page_idx,
                    phase=phase,
                    tensor_role="value",
                )
                source_start += count
                remaining -= count
        except Exception as exc:
            self._workspace.poison("backing_copy_failure")
            raise PagedSharedStagingCacheError(
                "paged backing copy failed; call reset before reuse"
            ) from exc

    def _stage(self, length: int) -> tuple[Tensor, Tensor]:
        try:
            staging_keys, staging_values = self._workspace.ensure_staging()
            page_size = self.geometry.page_token_capacity
            for page_idx in range((length + page_size - 1) // page_size):
                start = page_idx * page_size
                count = min(page_size, length - start)
                destination = slice(start, start + count)
                source = slice(0, count)
                self._workspace.copy(
                    staging_keys[..., destination, :],
                    self.key_pages[page_idx][..., source, :],
                    layer_idx=self.layer_idx,
                    page_idx=page_idx,
                    phase="backing_page_to_staging",
                    tensor_role="key",
                )
                self._workspace.copy(
                    staging_values[..., destination, :],
                    self.value_pages[page_idx][..., source, :],
                    layer_idx=self.layer_idx,
                    page_idx=page_idx,
                    phase="backing_page_to_staging",
                    tensor_role="value",
                )
        except Exception as exc:
            self._workspace.poison("staging_allocation_or_copy_failure")
            raise PagedSharedStagingCacheError(
                "shared staging failed; call reset before reuse"
            ) from exc
        return staging_keys[..., :length, :], staging_values[..., :length, :]

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        self._workspace.require_healthy()
        if args or kwargs:
            raise UnsupportedPagedSharedStagingOperation(
                "cache kwargs/positions are unsupported; this cache is append-only"
            )
        query_length = self._validate(key_states, value_states, allow_empty=False)
        if self.cumulative_length and query_length != 1:
            raise UnsupportedPagedSharedStagingOperation(
                "after prefill, only single-token decode is supported"
            )
        new_length = self.cumulative_length + query_length
        if new_length > self.geometry.max_cache_len:
            raise PagedSharedStagingCacheError("cache capacity exceeded")
        self._workspace.bind_current_stream()
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        phase = (
            "prefill_to_backing_pages" if not self.cumulative_length else "decode_to_backing_page"
        )
        self._write(key_states, value_states, self.cumulative_length, phase)
        if not self.cumulative_length:
            self.cumulative_length = new_length
            return key_states, value_states
        staged = self._stage(new_length)
        self.cumulative_length = new_length
        return staged

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        self._workspace.require_healthy()
        length = self.cumulative_length + _positive_int(query_length, "query_length")
        if length > self.geometry.max_cache_len:
            raise PagedSharedStagingCacheError("attention mask request exceeds capacity")
        return length, 0

    def get_seq_length(self) -> int:
        self._workspace.require_healthy()
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.geometry.max_cache_len

    def reset(self) -> None:
        self.cumulative_length = 0

    def reorder_cache(self, _beam_idx: Tensor) -> None:
        raise UnsupportedPagedSharedStagingOperation("beam reordering is unsupported")

    def crop(self, _tokens_to_remove: int) -> None:
        raise UnsupportedPagedSharedStagingOperation("cache cropping is unsupported")

    def batch_repeat_interleave(self, _repeats: int) -> None:
        raise UnsupportedPagedSharedStagingOperation("batch expansion is unsupported")

    def batch_select_indices(self, _indices: Tensor) -> None:
        raise UnsupportedPagedSharedStagingOperation("batch selection is unsupported")

    def offload(self) -> None:
        raise UnsupportedPagedSharedStagingOperation("backing is permanently CPU-owned")

    def prefetch(self) -> None:
        raise UnsupportedPagedSharedStagingOperation("prefetch is replaced by staging")


class PagedSharedStagingCache(_TransformersCache):
    """Transformers-compatible container for exact paged CPU backing."""

    def __init__(
        self,
        *,
        num_layers: int,
        max_cache_len: int,
        page_token_capacity: int = 8192,
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
        self.geometry = PagedSharedStagingGeometry(
            num_layers=num_layers,
            max_cache_len=max_cache_len,
            page_token_capacity=page_token_capacity,
            num_key_value_heads=num_key_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            dtype=dtype,
            staging_device=torch.device(staging_device),
            pin_cpu_backing=pin_cpu_backing,
            non_blocking_copies=non_blocking_copies,
        )
        if not callable(allocator) or not callable(copier):
            raise TypeError("allocator and copier must be callable")
        self._workspace = _PagedWorkspace(
            self.geometry,
            allocator=allocator,
            copier=copier,
            telemetry_event_capacity=telemetry_event_capacity,
        )
        layers = [
            PagedSharedStagingCacheLayer(
                geometry=self.geometry,
                layer_idx=idx,
                workspace=self._workspace,
            )
            for idx in range(self.geometry.num_layers)
        ]
        super().__init__(layers=layers, offloading=False)
        self._expected_layer_idx = 0
        self._cycle_start_length: int | None = None
        self._cycle_query_length: int | None = None
        self._completed_cycles = 0
        self._reset_count = 0

    def _layer(self, layer_idx: int) -> PagedSharedStagingCacheLayer:
        layer_idx = _nonnegative_int(layer_idx, "layer_idx")
        if layer_idx >= len(self.layers):
            raise IndexError("layer_idx exceeds configured layer count")
        layer = self.layers[layer_idx]
        if not isinstance(layer, PagedSharedStagingCacheLayer):
            raise PagedSharedStagingCacheError("cache layer type changed")
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
            if args or kwargs:
                raise UnsupportedPagedSharedStagingOperation(
                    "cache kwargs/positions are unsupported; this cache is append-only"
                )
            layer = self._layer(layer_idx)
            if layer_idx != self._expected_layer_idx:
                raise PagedSharedStagingCacheError(
                    f"layers must update sequentially; expected {self._expected_layer_idx}, got {layer_idx}"
                )
            query_length = layer._validate(  # noqa: SLF001
                key_states, value_states, allow_empty=False
            )
            if layer_idx == 0:
                lengths = [candidate.get_seq_length() for candidate in self.layers]
                if len(set(lengths)) != 1:
                    raise PagedSharedStagingCacheError("layer logical lengths diverged")
                self._cycle_start_length = lengths[0]
                self._cycle_query_length = query_length
            elif query_length != self._cycle_query_length:
                raise PagedSharedStagingCacheError("query length changed within update cycle")
            elif layer.get_seq_length() != self._cycle_start_length:
                raise PagedSharedStagingCacheError("layer length changed before its update")
            result = layer.update(key_states, value_states)
            self._expected_layer_idx = (layer_idx + 1) % self.geometry.num_layers
            if self._expected_layer_idx == 0:
                expected = int(self._cycle_start_length) + int(self._cycle_query_length)
                if any(candidate.get_seq_length() != expected for candidate in self.layers):
                    raise PagedSharedStagingCacheError("layer logical lengths diverged after cycle")
                self._cycle_start_length = None
                self._cycle_query_length = None
                self._completed_cycles += 1
            return result
        except Exception:
            if partial_cycle:
                self._workspace.poison("partial_update_cycle_failure")
            elif not self._workspace.is_poisoned:
                self._cycle_start_length = None
                self._cycle_query_length = None
            raise

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int | Sequence[int],
        head_dim: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if _positive_int(batch_size, "batch_size") != 1:
            raise UnsupportedPagedSharedStagingOperation("multibatch initialization unsupported")
        heads = self._normalize_dimension(num_heads, "num_heads")
        dims = self._normalize_dimension(head_dim, "head_dim")
        if heads != [self.geometry.num_key_value_heads] * self.geometry.num_layers:
            raise ValueError("early initialization head geometry differs")
        if dims != [self.geometry.key_head_dim] * self.geometry.num_layers:
            raise ValueError("early initialization head dimension differs")
        if self.geometry.key_head_dim != self.geometry.value_head_dim:
            raise ValueError("early initialization requires equal K/V dimensions")
        if dtype != self.geometry.dtype or torch.device(device) != self.geometry.staging_device:
            raise ValueError("early initialization dtype/device differs")
        fake = torch.empty(
            (1, self.geometry.num_key_value_heads, 0, self.geometry.key_head_dim),
            dtype=dtype,
            device=device,
        )
        for layer in self.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(fake, fake)

    def _normalize_dimension(self, value: int | Sequence[int], field: str) -> list[int]:
        if isinstance(value, int):
            return [_positive_int(value, field)] * self.geometry.num_layers
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{field} must be an integer or sequence")
        result = [_positive_int(item, f"{field}[{idx}]") for idx, item in enumerate(value)]
        if len(result) != self.geometry.num_layers:
            raise ValueError(f"{field} sequence has wrong length")
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
        raise UnsupportedPagedSharedStagingOperation("beam reordering is unsupported")

    def crop(self, _tokens_to_remove: int) -> None:
        raise UnsupportedPagedSharedStagingOperation("cache cropping is unsupported")

    def batch_repeat_interleave(self, _repeats: int) -> None:
        raise UnsupportedPagedSharedStagingOperation("batch expansion is unsupported")

    def batch_select_indices(self, _indices: Tensor) -> None:
        raise UnsupportedPagedSharedStagingOperation("batch selection is unsupported")

    def offload(self, _layer_idx: int, _only_non_sliding: bool = True) -> None:
        raise UnsupportedPagedSharedStagingOperation("backing is permanently CPU-owned")

    def prefetch(self, _layer_idx: int, _only_non_sliding: bool = True) -> None:
        raise UnsupportedPagedSharedStagingOperation("prefetch is replaced by staging")

    def staging_data_ptrs(self) -> tuple[int, int] | None:
        if self._workspace.staging_keys is None or self._workspace.staging_values is None:
            return None
        return (
            int(self._workspace.staging_keys.untyped_storage().data_ptr()),
            int(self._workspace.staging_values.untyped_storage().data_ptr()),
        )

    def backing_page_data_ptrs(
        self,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]] | None, ...]:
        result: list[tuple[tuple[int, ...], tuple[int, ...]] | None] = []
        for layer in self.layers:
            if not layer.is_initialized:
                result.append(None)
            else:
                result.append(
                    (
                        tuple(int(page.untyped_storage().data_ptr()) for page in layer.key_pages),
                        tuple(int(page.untyped_storage().data_ptr()) for page in layer.value_pages),
                    )
                )
        return tuple(result)

    def telemetry(self) -> dict[str, object]:
        live_backing_bytes = 0
        live_page_allocations = 0
        layer_backing: list[dict[str, object]] = []
        for layer in self.layers:
            key_bytes = sum(_storage_bytes(page) for page in layer.key_pages)
            value_bytes = sum(_storage_bytes(page) for page in layer.value_pages)
            pages = len(layer.key_pages) + len(layer.value_pages)
            live_backing_bytes += key_bytes + value_bytes
            live_page_allocations += pages
            layer_backing.append(
                {
                    "layer_idx": layer.layer_idx,
                    "initialized": layer.is_initialized,
                    "logical_length": layer.cumulative_length,
                    "key_pages": len(layer.key_pages),
                    "value_pages": len(layer.value_pages),
                    "key_bytes": key_bytes,
                    "value_bytes": value_bytes,
                    "total_bytes": key_bytes + value_bytes,
                }
            )
        stage_key_bytes = _storage_bytes(self._workspace.staging_keys)
        stage_value_bytes = _storage_bytes(self._workspace.staging_values)
        return {
            "schema_version": TELEMETRY_SCHEMA,
            "geometry": self.geometry.to_dict(),
            "health": self._workspace.health(),
            "sequence": {
                "layer_logical_lengths": [layer.cumulative_length for layer in self.layers],
                "expected_next_layer_idx": self._expected_layer_idx,
                "update_cycle_in_progress": self._cycle_start_length is not None,
                "completed_update_cycles": self._completed_cycles,
            },
            "storage": {
                "backing_device": "cpu",
                "live_backing_page_allocations": live_page_allocations,
                "live_logical_backing_bytes": live_backing_bytes,
                "planned_logical_backing_bytes": self.geometry.logical_backing_bytes,
                "planned_predicted_caching_host_allocator_rounded_bytes": (
                    self.geometry.predicted_rounded_backing_bytes
                ),
                "planned_predicted_rounding_waste_bytes": (
                    self.geometry.predicted_rounding_waste_bytes
                ),
                "rounding_model": "PowerOf2Ceil(each_backing_allocation)",
                "layer_backing": layer_backing,
                "staging_device": str(self.geometry.staging_device),
                "staging_allocated": self._workspace.staging_keys is not None,
                "staging_key_bytes": stage_key_bytes,
                "staging_value_bytes": stage_value_bytes,
                "staging_bytes": stage_key_bytes + stage_value_bytes,
                "staging_pair_allocations": self._workspace.staging_pair_allocations,
                "live_cache_bytes": live_backing_bytes + stage_key_bytes + stage_value_bytes,
            },
            "operations": {
                "backing_allocation_transactions": (
                    self._workspace.backing_allocation_transactions
                ),
                "resets": self._reset_count,
            },
            "transfers": self._workspace.ledger.snapshot(),
        }
