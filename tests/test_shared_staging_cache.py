from __future__ import annotations

import json
import os
from collections.abc import Callable

import pytest
import torch

from molor.experimental.shared_staging_cache import (
    TRANSFORMERS_CACHE_AVAILABLE,
    SharedStagingCache,
    SharedStagingCacheError,
    SharedStagingCacheLayer,
    UnsupportedSharedStagingOperation,
)


class RecordingAllocator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "pin_memory": pin_memory,
            }
        )
        return torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)


class RecordingCopier:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> None:
        self.calls.append(
            {
                "destination_device": str(destination.device),
                "source_device": str(source.device),
                "shape": tuple(source.shape),
                "non_blocking": non_blocking,
            }
        )
        destination.copy_(source, non_blocking=non_blocking)


class MutateThenFailCopier:
    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def __call__(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> None:
        destination.copy_(source, non_blocking=non_blocking)
        call_index = self.call_count
        self.call_count += 1
        if call_index == self.fail_on_call:
            raise RuntimeError(f"injected copy failure {call_index}")


class FailSecondPairAllocationOnce:
    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.call_count = 0
        self.failed = False

    def __call__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> torch.Tensor:
        call_index = self.call_count
        self.call_count += 1
        if call_index == self.fail_on_call and not self.failed:
            self.failed = True
            raise RuntimeError(f"injected allocation failure {call_index}")
        return torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)


def make_cache(
    *,
    allocator: Callable[..., torch.Tensor] | None = None,
    copier: Callable[..., None] | None = None,
    pin_cpu_backing: bool = False,
    non_blocking_copies: bool = False,
    max_cache_len: int = 5,
    staging_device: str = "cpu",
    telemetry_event_capacity: int = 256,
) -> SharedStagingCache:
    kwargs: dict[str, object] = {}
    if allocator is not None:
        kwargs["allocator"] = allocator
    if copier is not None:
        kwargs["copier"] = copier
    return SharedStagingCache(
        num_layers=2,
        max_cache_len=max_cache_len,
        num_key_value_heads=2,
        key_head_dim=3,
        value_head_dim=2,
        dtype=torch.float32,
        staging_device=staging_device,
        pin_cpu_backing=pin_cpu_backing,
        non_blocking_copies=non_blocking_copies,
        telemetry_event_capacity=telemetry_event_capacity,
        **kwargs,
    )


def kv(
    sequence_length: int, *, offset: float = 0.0, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    key_elements = 2 * sequence_length * 3
    value_elements = 2 * sequence_length * 2
    keys = torch.arange(key_elements, dtype=torch.float32, device=device).reshape(
        1, 2, sequence_length, 3
    )
    values = torch.arange(value_elements, dtype=torch.float32, device=device).reshape(
        1, 2, sequence_length, 2
    )
    return keys + offset, values + offset + 1000.0


def complete_prefill(
    cache: SharedStagingCache, *, sequence_length: int = 3
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(2):
        states = kv(sequence_length, offset=layer_idx * 10_000.0)
        returned = cache.update(*states, layer_idx)
        assert returned[0] is states[0]
        assert returned[1] is states[1]
        inputs.append(states)
    return inputs


def test_prefill_returns_original_tensors_and_persists_exact_cpu_backing() -> None:
    allocator = RecordingAllocator()
    copier = RecordingCopier()
    cache = make_cache(allocator=allocator, copier=copier)

    inputs = complete_prefill(cache)

    assert cache.staging_data_ptrs() is None
    assert len(allocator.calls) == 4
    assert all(call["device"] == torch.device("cpu") for call in allocator.calls)
    assert all(call["pin_memory"] is False for call in allocator.calls)
    assert len({pointer for pair in cache.backing_data_ptrs() for pointer in pair or ()}) == 4
    for layer_idx, (keys, values) in enumerate(inputs):
        layer = cache.layers[layer_idx]
        assert layer.keys is not None and layer.values is not None
        assert layer.keys.device.type == "cpu"
        assert layer.values.device.type == "cpu"
        assert torch.equal(layer.keys[..., :3, :], keys)
        assert torch.equal(layer.values[..., :3, :], values)
        assert layer.get_seq_length() == 3
    telemetry = cache.telemetry()
    assert telemetry["sequence"]["layer_logical_lengths"] == [3, 3]  # type: ignore[index]
    assert telemetry["storage"]["staging_allocated"] is False  # type: ignore[index]
    assert telemetry["transfers"]["copies"] == 4  # type: ignore[index]
    assert [event["phase"] for event in telemetry["transfers"]["events"]] == [  # type: ignore[index]
        "prefill_to_backing",
        "prefill_to_backing",
        "prefill_to_backing",
        "prefill_to_backing",
    ]


def test_decode_returns_only_used_prefix_and_reuses_one_staging_pair() -> None:
    allocator = RecordingAllocator()
    copier = RecordingCopier()
    cache = make_cache(allocator=allocator, copier=copier)
    prefills = complete_prefill(cache)
    clones: list[tuple[torch.Tensor, torch.Tensor]] = []
    pointers: list[tuple[int, int]] = []

    for layer_idx in range(2):
        decode = kv(1, offset=50_000.0 + layer_idx * 10_000.0)
        returned = cache.update(*decode, layer_idx)
        assert returned[0].shape == (1, 2, 4, 3)
        assert returned[1].shape == (1, 2, 4, 2)
        assert returned[0].untyped_storage().nbytes() == 120
        assert returned[1].untyped_storage().nbytes() == 80
        pointers.append(
            (
                returned[0].untyped_storage().data_ptr(),
                returned[1].untyped_storage().data_ptr(),
            )
        )
        clones.append((returned[0].clone(), returned[1].clone()))
        expected_keys = torch.cat((prefills[layer_idx][0], decode[0]), dim=-2)
        expected_values = torch.cat((prefills[layer_idx][1], decode[1]), dim=-2)
        assert torch.equal(clones[-1][0], expected_keys)
        assert torch.equal(clones[-1][1], expected_values)
        layer = cache.layers[layer_idx]
        assert torch.equal(layer.keys[..., :4, :], expected_keys)
        assert torch.equal(layer.values[..., :4, :], expected_values)

    assert pointers[0] == pointers[1] == cache.staging_data_ptrs()
    assert len(allocator.calls) == 6
    telemetry = cache.telemetry()
    assert telemetry["storage"]["staging_pair_allocations"] == 1  # type: ignore[index]
    assert telemetry["storage"]["staging_bytes"] == 200  # type: ignore[index]
    assert telemetry["operations"]["decodes_by_layer"] == [1, 1]  # type: ignore[index]
    assert [event["phase"] for event in telemetry["transfers"]["events"][-8:]] == [  # type: ignore[index]
        "decode_to_backing",
        "decode_to_backing",
        "backing_to_staging",
        "backing_to_staging",
        "decode_to_backing",
        "decode_to_backing",
        "backing_to_staging",
        "backing_to_staging",
    ]


def test_mask_sizes_track_used_length_instead_of_full_capacity() -> None:
    cache = make_cache()

    assert cache.get_mask_sizes(3, 0) == (3, 0)
    complete_prefill(cache, sequence_length=3)
    assert cache.get_seq_length() == 3
    assert cache.get_query_offset() == 3
    assert cache.get_mask_sizes(1, 0) == (4, 0)
    assert cache.get_mask_sizes(2, 1) == (5, 0)
    assert cache.get_max_length() == 5
    with pytest.raises(SharedStagingCacheError, match="mask request exceeds"):
        cache.get_mask_sizes(3, 0)


def test_reset_only_changes_logical_lengths_and_retains_allocations_and_bytes() -> None:
    cache = make_cache()
    complete_prefill(cache)
    for layer_idx in range(2):
        cache.update(*kv(1, offset=30_000.0 + layer_idx), layer_idx)
    for layer in cache.layers:
        assert layer.keys is not None and layer.values is not None
        layer.keys.fill_(17.0 + layer.layer_idx)
        layer.values.fill_(29.0 + layer.layer_idx)
    before_bytes = [
        (layer.keys.clone(), layer.values.clone())  # type: ignore[union-attr]
        for layer in cache.layers
    ]
    before_backing_pointers = cache.backing_data_ptrs()
    before_staging_pointers = cache.staging_data_ptrs()
    assert before_staging_pointers is not None

    cache.reset()

    assert [layer.get_seq_length() for layer in cache.layers] == [0, 0]
    assert cache.backing_data_ptrs() == before_backing_pointers
    assert cache.staging_data_ptrs() == before_staging_pointers
    for layer_idx, layer in enumerate(cache.layers):
        assert torch.equal(layer.keys, before_bytes[layer_idx][0])
        assert torch.equal(layer.values, before_bytes[layer_idx][1])
    assert cache.telemetry()["operations"]["resets"] == 1  # type: ignore[index]

    new_prefill = kv(1, offset=90_000.0)
    returned = cache.update(*new_prefill, 0)
    assert returned[0] is new_prefill[0]
    assert torch.equal(cache.layers[0].keys[..., :1, :], new_prefill[0])


def test_layer_updates_must_form_complete_sequential_cycles() -> None:
    cache = make_cache()
    states = kv(2)
    with pytest.raises(SharedStagingCacheError, match="expected 0, got 1"):
        cache.update(*states, 1)

    cache.update(*states, 0)
    with pytest.raises(SharedStagingCacheError, match="expected 1, got 0"):
        cache.update(*states, 0)
    assert cache.telemetry()["health"]["poisoned"] is True  # type: ignore[index]
    with pytest.raises(SharedStagingCacheError, match="poisoned"):
        cache.update(*kv(1, offset=100.0), 1)
    with pytest.raises(SharedStagingCacheError, match="poisoned"):
        cache.get_mask_sizes(1, 1)

    cache.reset()
    cache.update(*states, 0)
    cache.update(*kv(2, offset=200.0), 1)
    assert [layer.get_seq_length() for layer in cache.layers] == [2, 2]


def test_query_validation_failure_midcycle_poisons_until_reset() -> None:
    cache = make_cache()
    cache.update(*kv(2), 0)

    with pytest.raises(SharedStagingCacheError, match="query length changed"):
        cache.update(*kv(1, offset=100.0), 1)

    assert cache.telemetry()["health"]["active_poison_reason"] == (  # type: ignore[index]
        "partial_update_cycle_failure"
    )
    with pytest.raises(SharedStagingCacheError, match="poisoned"):
        cache.get_seq_length()
    cache.reset()
    assert cache.get_seq_length() == 0


def test_only_initial_prefill_may_append_multiple_tokens() -> None:
    cache = make_cache()
    complete_prefill(cache, sequence_length=2)

    with pytest.raises(UnsupportedSharedStagingOperation, match="single-token"):
        cache.update(*kv(2, offset=500.0), 0)
    assert cache.get_seq_length(0) == 2


def test_capacity_failure_does_not_advance_logical_length_or_transfer() -> None:
    cache = make_cache(max_cache_len=2)

    with pytest.raises(SharedStagingCacheError, match="capacity exceeded"):
        cache.update(*kv(3), 0)

    assert cache.get_seq_length(0) == 0
    assert cache.telemetry()["transfers"]["copies"] == 0  # type: ignore[index]


@pytest.mark.parametrize(
    "mutation",
    ["batch", "key_heads", "value_heads", "key_dim", "value_dim", "sequence", "dtype"],
)
def test_update_rejects_geometry_batch_and_dtype_drift(mutation: str) -> None:
    cache = make_cache()
    keys, values = kv(2)
    if mutation == "batch":
        keys = keys.expand(2, -1, -1, -1).clone()
        values = values.expand(2, -1, -1, -1).clone()
    elif mutation == "key_heads":
        keys = torch.zeros(1, 1, 2, 3)
    elif mutation == "value_heads":
        values = torch.zeros(1, 1, 2, 2)
    elif mutation == "key_dim":
        keys = torch.zeros(1, 2, 2, 4)
    elif mutation == "value_dim":
        values = torch.zeros(1, 2, 2, 3)
    elif mutation == "sequence":
        values = torch.zeros(1, 2, 1, 2)
    else:
        keys = keys.double()

    exception = UnsupportedSharedStagingOperation if mutation == "batch" else ValueError
    with pytest.raises(exception):
        cache.update(keys, values, 0)


def test_update_rejects_device_drift_before_copy() -> None:
    cache = make_cache()
    keys = torch.empty(1, 2, 1, 3, dtype=torch.float32, device="meta")
    values = torch.empty(1, 2, 1, 2, dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="device differs"):
        cache.update(keys, values, 0)


def test_append_only_cache_rejects_positions_beams_crop_and_batch_operations() -> None:
    cache = make_cache()
    operations = [
        lambda: cache.update(*kv(1), 0, cache_position=torch.tensor([0])),
        lambda: cache.reorder_cache(torch.tensor([0])),
        lambda: cache.crop(1),
        lambda: cache.batch_repeat_interleave(2),
        lambda: cache.batch_select_indices(torch.tensor([0])),
        lambda: cache.offload(0),
        lambda: cache.prefetch(0),
    ]

    for operation in operations:
        with pytest.raises(UnsupportedSharedStagingOperation):
            operation()


def test_early_initialization_allocates_only_cpu_backing() -> None:
    allocator = RecordingAllocator()
    cache = SharedStagingCache(
        num_layers=2,
        max_cache_len=5,
        num_key_value_heads=2,
        key_head_dim=3,
        value_head_dim=3,
        dtype=torch.float32,
        staging_device="cpu",
        pin_cpu_backing=False,
        non_blocking_copies=False,
        allocator=allocator,
    )

    cache.early_initialization(
        batch_size=1,
        num_heads=2,
        head_dim=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert cache.is_initialized is True
    assert cache.staging_data_ptrs() is None
    assert len(allocator.calls) == 4
    assert [layer.get_seq_length() for layer in cache.layers] == [0, 0]


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("batch_size", True, TypeError),
        ("batch_size", 1.0, TypeError),
        ("batch_size", 2, UnsupportedSharedStagingOperation),
        ("num_heads", True, TypeError),
        ("num_heads", [2, True], TypeError),
        ("num_heads", [2], ValueError),
        ("head_dim", True, TypeError),
        ("head_dim", [3, 3.0], TypeError),
        ("head_dim", [3], ValueError),
    ],
)
def test_early_initialization_rejects_bool_float_and_bad_axis_lengths(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    cache = SharedStagingCache(
        num_layers=2,
        max_cache_len=5,
        num_key_value_heads=2,
        key_head_dim=3,
        dtype=torch.float32,
        staging_device="cpu",
        pin_cpu_backing=False,
    )
    arguments: dict[str, object] = {
        "batch_size": 1,
        "num_heads": 2,
        "head_dim": 3,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    arguments[field] = value

    with pytest.raises(exception):
        cache.early_initialization(**arguments)  # type: ignore[arg-type]

    assert cache.is_initialized is False


def test_allocator_must_honor_exclusive_storage_and_pin_request() -> None:
    shared = torch.empty(1, 2, 3, 3)

    def aliasing_allocator(
        _shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> torch.Tensor:
        assert dtype == torch.float32 and device.type == "cpu" and pin_memory is False
        return shared

    cache = SharedStagingCache(
        num_layers=1,
        max_cache_len=3,
        num_key_value_heads=2,
        key_head_dim=3,
        value_head_dim=3,
        dtype=torch.float32,
        staging_device="cpu",
        pin_cpu_backing=False,
        allocator=aliasing_allocator,
    )
    states = torch.zeros(1, 2, 1, 3)
    with pytest.raises(SharedStagingCacheError, match="backing allocation failed"):
        cache.update(states, states, 0)
    assert cache.telemetry()["health"]["poisoned"] is True  # type: ignore[index]

    def unpinned_allocator(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> torch.Tensor:
        assert pin_memory is True
        return torch.empty(shape, dtype=dtype, device=device)

    pinned_cache = SharedStagingCache(
        num_layers=1,
        max_cache_len=3,
        num_key_value_heads=2,
        key_head_dim=3,
        dtype=torch.float32,
        staging_device="cpu",
        pin_cpu_backing=True,
        allocator=unpinned_allocator,
    )
    with pytest.raises(SharedStagingCacheError, match="backing allocation failed"):
        pinned_cache.update(states, states, 0)
    assert pinned_cache.telemetry()["health"]["poisoned"] is True  # type: ignore[index]


@pytest.mark.parametrize(("phase", "fail_on_call"), [("backing", 1), ("staging", 5)])
def test_pair_allocation_is_atomic_and_transient_failure_recovers_after_reset(
    phase: str,
    fail_on_call: int,
) -> None:
    allocator = FailSecondPairAllocationOnce(fail_on_call)
    cache = make_cache(allocator=allocator)
    if phase == "staging":
        complete_prefill(cache)

    with pytest.raises(SharedStagingCacheError, match="reset|allocation"):
        cache.update(*kv(1 if phase == "staging" else 3, offset=77_000.0), 0)

    failed = cache.telemetry()
    assert failed["health"]["poisoned"] is True  # type: ignore[index]
    if phase == "backing":
        assert cache.backing_data_ptrs() == (None, None)
        assert failed["storage"]["backing_bytes"] == 0  # type: ignore[index]
    else:
        assert cache.staging_data_ptrs() is None
        assert failed["storage"]["staging_allocated"] is False  # type: ignore[index]
        assert failed["storage"]["staging_bytes"] == 0  # type: ignore[index]

    cache.reset()
    prefills = complete_prefill(cache, sequence_length=2)
    for layer_idx in range(2):
        decode = kv(1, offset=88_000.0 + layer_idx)
        returned = cache.update(*decode, layer_idx)
        assert torch.equal(returned[0], torch.cat((prefills[layer_idx][0], decode[0]), dim=-2))
        assert torch.equal(returned[1], torch.cat((prefills[layer_idx][1], decode[1]), dim=-2))

    recovered = cache.telemetry()
    assert recovered["health"]["poisoned"] is False  # type: ignore[index]
    assert recovered["storage"]["staging_pair_allocations"] == 1  # type: ignore[index]
    owned_pointers = [pointer for pair in cache.backing_data_ptrs() for pointer in pair or ()]
    assert len(owned_pointers) == len(set(owned_pointers)) == 4


@pytest.mark.parametrize(
    ("fail_on_call", "decode_phase", "expected_reason"),
    [
        (0, False, "backing_copy_failure"),
        (1, False, "backing_copy_failure"),
        (4, True, "backing_copy_failure"),
        (5, True, "backing_copy_failure"),
        (6, True, "staging_allocation_or_copy_failure"),
        (7, True, "staging_allocation_or_copy_failure"),
    ],
)
def test_any_partial_copy_failure_poisons_until_logical_reset(
    fail_on_call: int,
    decode_phase: bool,
    expected_reason: str,
) -> None:
    copier = MutateThenFailCopier(fail_on_call)
    cache = make_cache(copier=copier)
    if decode_phase:
        complete_prefill(cache)

    with pytest.raises(SharedStagingCacheError, match="reset"):
        cache.update(*kv(1 if decode_phase else 3, offset=80_000.0), 0)

    failed_pointers = cache.backing_data_ptrs()
    failed_staging_pointers = cache.staging_data_ptrs()
    telemetry = cache.telemetry()
    assert telemetry["health"] == {
        "poisoned": True,
        "active_poison_reason": expected_reason,
        "poison_events": 1,
        "poison_events_by_reason": {expected_reason: 1},
        "cuda_stream_contract": "one_stable_current_stream",
        "cuda_stream_bound": False,
    }
    with pytest.raises(SharedStagingCacheError, match="poisoned"):
        cache.update(*kv(1), 0)
    with pytest.raises(SharedStagingCacheError, match="poisoned"):
        cache.get_mask_sizes(1, 0)
    with pytest.raises(SharedStagingCacheError, match="poisoned"):
        cache.get_seq_length(0)

    cache.reset()
    recovered = cache.telemetry()
    assert recovered["health"]["poisoned"] is False  # type: ignore[index]
    assert recovered["health"]["poison_events"] == 1  # type: ignore[index]
    assert recovered["sequence"]["layer_logical_lengths"] == [0, 0]  # type: ignore[index]
    assert cache.backing_data_ptrs() == failed_pointers
    assert cache.staging_data_ptrs() == failed_staging_pointers

    copier.fail_on_call = -1
    complete_prefill(cache, sequence_length=2)
    assert [cache.get_seq_length(index) for index in range(2)] == [2, 2]


def test_noncontiguous_qwen_like_kv_views_roundtrip_exactly() -> None:
    cache = make_cache()
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(2):
        key_base = torch.arange(1 * 2 * 3 * 6, dtype=torch.float32).reshape(1, 2, 3, 6)
        value_base = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
        key_base.add_(layer_idx * 1000.0)
        value_base.add_(layer_idx * 2000.0)
        keys = key_base[..., ::2]
        values = value_base[..., ::2]
        assert not keys.is_contiguous() and not values.is_contiguous()
        returned = cache.update(keys, values, layer_idx)
        assert returned[0] is keys and returned[1] is values
        retained.append((keys, values))

    for layer_idx in range(2):
        decode_keys = torch.arange(12, dtype=torch.float32).reshape(1, 2, 1, 6)[..., ::2]
        decode_values = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)[..., ::2]
        returned = cache.update(decode_keys, decode_values, layer_idx)
        assert torch.equal(returned[0], torch.cat((retained[layer_idx][0], decode_keys), dim=-2))
        assert torch.equal(returned[1], torch.cat((retained[layer_idx][1], decode_values), dim=-2))


def test_telemetry_is_json_safe_deterministic_and_accounts_exact_live_bytes() -> None:
    cache = make_cache()
    complete_prefill(cache)
    for layer_idx in range(2):
        cache.update(*kv(1, offset=70_000.0 + layer_idx), layer_idx)

    first = cache.telemetry()
    second = cache.telemetry()

    assert first == second
    json.dumps(first, sort_keys=True, allow_nan=False)
    assert "pointer" not in json.dumps(first).lower()
    assert first["storage"] == {
        "backing_device": "cpu",
        "backing_bytes": 400,
        "layer_backing": [
            {
                "layer_idx": 0,
                "initialized": True,
                "logical_length": 4,
                "key_bytes": 120,
                "value_bytes": 80,
                "total_bytes": 200,
            },
            {
                "layer_idx": 1,
                "initialized": True,
                "logical_length": 4,
                "key_bytes": 120,
                "value_bytes": 80,
                "total_bytes": 200,
            },
        ],
        "staging_device": "cpu",
        "staging_allocated": True,
        "staging_key_bytes": 120,
        "staging_value_bytes": 80,
        "staging_bytes": 200,
        "staging_pair_allocations": 1,
        "live_cache_bytes": 600,
    }
    assert first["transfers"]["copies"] == 12  # type: ignore[index]
    assert first["transfers"]["bytes"] == 640  # type: ignore[index]
    assert [event["ordinal"] for event in first["transfers"]["events"]] == list(  # type: ignore[index]
        range(12)
    )


def test_transfer_telemetry_keeps_exact_lifetime_totals_with_a_bounded_tail() -> None:
    cache = make_cache(telemetry_event_capacity=4)
    complete_prefill(cache)
    for layer_idx in range(2):
        cache.update(*kv(1, offset=72_000.0 + layer_idx), layer_idx)

    transfers = cache.telemetry()["transfers"]

    assert transfers["copies"] == 12  # type: ignore[index]
    assert transfers["bytes"] == 640  # type: ignore[index]
    assert transfers["event_capacity"] == 4  # type: ignore[index]
    assert transfers["events_retained"] == 4  # type: ignore[index]
    assert transfers["events_dropped"] == 8  # type: ignore[index]
    assert [event["ordinal"] for event in transfers["events"]] == [8, 9, 10, 11]  # type: ignore[index]
    assert transfers["by_phase"] == {  # type: ignore[index]
        "backing_to_staging": {"copies": 4, "bytes": 320},
        "decode_to_backing": {"copies": 4, "bytes": 80},
        "prefill_to_backing": {"copies": 4, "bytes": 240},
    }


def test_transformers_cache_layer_contract_when_dependency_is_available() -> None:
    if not TRANSFORMERS_CACHE_AVAILABLE:
        pytest.skip("Transformers optional dependency is unavailable")
    from transformers.cache_utils import Cache, CacheLayerMixin

    cache = make_cache()
    assert isinstance(cache, Cache)
    assert all(isinstance(layer, CacheLayerMixin) for layer in cache.layers)
    assert all(isinstance(layer, SharedStagingCacheLayer) for layer in cache.layers)


TINY_QWEN_INTEGRATION_ENABLED = (
    os.environ.get("MOLOR_RUN_TINY_QWEN_SHARED_STAGING_TEST") == "1"
    and TRANSFORMERS_CACHE_AVAILABLE
)


@pytest.mark.skipif(
    not TINY_QWEN_INTEGRATION_ENABLED,
    reason="set MOLOR_RUN_TINY_QWEN_SHARED_STAGING_TEST=1 for opt-in CPU model parity",
)
def test_tiny_qwen_prefill_and_decode_logits_are_bit_exact_against_dynamic_cache() -> None:
    from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 5, 8, 3, 2]], dtype=torch.long)
    dynamic = DynamicCache(config=config)
    shared = SharedStagingCache(
        num_layers=2,
        max_cache_len=16,
        num_key_value_heads=2,
        key_head_dim=8,
        dtype=torch.float32,
        staging_device="cpu",
        pin_cpu_backing=False,
        non_blocking_copies=False,
    )

    with torch.inference_mode():
        dynamic_prefill = model(
            input_ids=input_ids,
            past_key_values=dynamic,
            use_cache=True,
        ).logits
        shared_prefill = model(
            input_ids=input_ids,
            past_key_values=shared,
            use_cache=True,
        ).logits
        token = dynamic_prefill[:, -1].argmax(dim=-1, keepdim=True)
        dynamic_decode = model(
            input_ids=token,
            past_key_values=dynamic,
            use_cache=True,
        ).logits
        shared_decode = model(
            input_ids=token,
            past_key_values=shared,
            use_cache=True,
        ).logits

    assert torch.equal(shared_prefill, dynamic_prefill)
    assert torch.equal(shared_decode, dynamic_decode)
    assert dynamic.get_seq_length() == shared.get_seq_length() == 6
    assert shared.telemetry()["transfers"]["copies"] == 12  # type: ignore[index]


CUDA_SHARED_STAGING_ENABLED = (
    os.environ.get("MOLOR_RUN_CUDA_SHARED_STAGING_TESTS") == "1" and torch.cuda.is_available()
)


@pytest.mark.skipif(
    not CUDA_SHARED_STAGING_ENABLED,
    reason="set MOLOR_RUN_CUDA_SHARED_STAGING_TESTS=1 to run explicit CUDA systems proof",
)
def test_cuda_shared_pointer_exact_roundtrip_and_current_stream_order() -> None:
    from torch.nn import functional as F

    device = "cuda:0"
    cache = make_cache(
        staging_device=device,
        pin_cpu_backing=True,
        non_blocking_copies=True,
    )
    stream = torch.cuda.Stream(device=device)
    prefills: list[tuple[torch.Tensor, torch.Tensor]] = []
    observed: list[tuple[torch.Tensor, torch.Tensor]] = []
    pointers: list[tuple[int, int]] = []
    decodes: list[tuple[torch.Tensor, torch.Tensor]] = []
    attention_outputs: list[torch.Tensor] = []
    attention_references: list[torch.Tensor] = []
    with torch.cuda.stream(stream):
        for layer_idx in range(2):
            states = kv(2, offset=layer_idx * 10_000.0, device=device)
            returned = cache.update(*states, layer_idx)
            assert returned[0] is states[0] and returned[1] is states[1]
            prefills.append(states)

        for layer_idx in range(2):
            decode = kv(1, offset=50_000.0 + layer_idx * 10_000.0, device=device)
            decodes.append(decode)
            returned = cache.update(*decode, layer_idx)
            pointers.append(
                (
                    returned[0].untyped_storage().data_ptr(),
                    returned[1].untyped_storage().data_ptr(),
                )
            )
            # Actual SDPA consumption and the next staging overwrite share this stream.
            query = torch.arange(6, dtype=torch.float32, device=device).reshape(1, 2, 1, 3)
            attention_outputs.append(
                F.scaled_dot_product_attention(query, returned[0], returned[1]).clone()
            )
            expected_keys = torch.cat((prefills[layer_idx][0], decode[0]), dim=-2)
            expected_values = torch.cat((prefills[layer_idx][1], decode[1]), dim=-2)
            attention_references.append(
                F.scaled_dot_product_attention(
                    query,
                    expected_keys,
                    expected_values,
                ).clone()
            )
            observed.append((returned[0].clone(), returned[1].clone()))

    stream.synchronize()
    assert pointers[0] == pointers[1] == cache.staging_data_ptrs()
    for layer_idx in range(2):
        expected_keys = torch.cat((prefills[layer_idx][0], decodes[layer_idx][0]), dim=-2)
        expected_values = torch.cat((prefills[layer_idx][1], decodes[layer_idx][1]), dim=-2)
        assert torch.equal(observed[layer_idx][0], expected_keys)
        assert torch.equal(observed[layer_idx][1], expected_values)
        assert torch.equal(attention_outputs[layer_idx], attention_references[layer_idx])
        assert torch.equal(cache.layers[layer_idx].keys[..., :3, :], expected_keys.cpu())
        assert torch.equal(cache.layers[layer_idx].values[..., :3, :], expected_values.cpu())
    assert cache.telemetry()["health"]["cuda_stream_bound"] is True  # type: ignore[index]

    other_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(other_stream):
        with pytest.raises(UnsupportedSharedStagingOperation, match="cross-stream"):
            cache.update(*kv(1, offset=99_000.0, device=device), 0)
