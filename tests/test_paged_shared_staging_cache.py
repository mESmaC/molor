from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import torch

from molor.experimental.paged_shared_staging_cache import (
    PagedSharedStagingCache,
    PagedSharedStagingCacheError,
    UnsupportedPagedSharedStagingOperation,
)


class FailAllocationOnce:
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


def make_cache(
    *,
    allocator: Callable[..., torch.Tensor] | None = None,
    copier: Callable[..., None] | None = None,
    staging_device: str = "cpu",
    pin_cpu_backing: bool = False,
    non_blocking_copies: bool = False,
    telemetry_event_capacity: int = 256,
) -> PagedSharedStagingCache:
    kwargs: dict[str, object] = {}
    if allocator is not None:
        kwargs["allocator"] = allocator
    if copier is not None:
        kwargs["copier"] = copier
    return PagedSharedStagingCache(
        num_layers=2,
        max_cache_len=8,
        page_token_capacity=4,
        num_key_value_heads=2,
        key_head_dim=2,
        value_head_dim=1,
        dtype=torch.float32,
        staging_device=staging_device,
        pin_cpu_backing=pin_cpu_backing,
        non_blocking_copies=non_blocking_copies,
        telemetry_event_capacity=telemetry_event_capacity,
        **kwargs,
    )


def kv(
    sequence_length: int,
    *,
    offset: float = 0.0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.arange(4 * sequence_length, dtype=torch.float32, device=device).reshape(
        1, 2, sequence_length, 2
    )
    values = torch.arange(2 * sequence_length, dtype=torch.float32, device=device).reshape(
        1, 2, sequence_length, 1
    )
    return keys + offset, values + offset + 1000.0


def _page_prefix(pages: tuple[torch.Tensor, ...], length: int) -> torch.Tensor:
    return torch.cat(pages, dim=-2)[..., :length, :]


def complete_prefill(
    cache: PagedSharedStagingCache,
    *,
    sequence_length: int = 2,
    offset: float = 0.0,
    device: str = "cpu",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    states_by_layer: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(2):
        states = kv(
            sequence_length,
            offset=offset + layer_idx * 10_000.0,
            device=device,
        )
        returned = cache.update(*states, layer_idx)
        assert returned[0] is states[0] and returned[1] is states[1]
        states_by_layer.append(states)
    return states_by_layer


def test_prefill_crosses_page_boundary_and_persists_exact_nonaliased_pages() -> None:
    cache = make_cache()
    expected: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_idx in range(2):
        states = kv(7, offset=layer_idx * 10_000.0)
        returned = cache.update(*states, layer_idx)
        assert returned[0] is states[0]
        assert returned[1] is states[1]
        expected.append(states)

    pointers = [
        pointer
        for layer in cache.backing_page_data_ptrs()
        for role in layer or ()
        for pointer in role
    ]
    assert len(pointers) == len(set(pointers)) == 8
    assert cache.staging_data_ptrs() is None
    for layer_idx, layer in enumerate(cache.layers):
        assert len(layer.key_pages) == len(layer.value_pages) == 2
        assert torch.equal(_page_prefix(layer.key_pages, 7), expected[layer_idx][0])
        assert torch.equal(_page_prefix(layer.value_pages, 7), expected[layer_idx][1])

    telemetry = cache.telemetry()
    assert telemetry["storage"]["live_backing_page_allocations"] == 8  # type: ignore[index]
    assert telemetry["storage"]["live_logical_backing_bytes"] == 384  # type: ignore[index]
    assert telemetry["sequence"]["completed_update_cycles"] == 1  # type: ignore[index]


def test_decode_fills_final_page_and_reconstructs_exact_shared_stage() -> None:
    cache = make_cache()
    prefills: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx in range(2):
        states = kv(7, offset=layer_idx * 10_000.0)
        cache.update(*states, layer_idx)
        prefills.append(states)

    returned_pointers: list[tuple[int, int]] = []
    for layer_idx in range(2):
        decode = kv(1, offset=50_000.0 + layer_idx * 10_000.0)
        returned = cache.update(*decode, layer_idx)
        expected_keys = torch.cat((prefills[layer_idx][0], decode[0]), dim=-2)
        expected_values = torch.cat((prefills[layer_idx][1], decode[1]), dim=-2)
        assert torch.equal(returned[0], expected_keys)
        assert torch.equal(returned[1], expected_values)
        assert torch.equal(_page_prefix(cache.layers[layer_idx].key_pages, 8), expected_keys)
        assert torch.equal(_page_prefix(cache.layers[layer_idx].value_pages, 8), expected_values)
        returned_pointers.append(
            (
                int(returned[0].untyped_storage().data_ptr()),
                int(returned[1].untyped_storage().data_ptr()),
            )
        )

    assert returned_pointers[0] == returned_pointers[1] == cache.staging_data_ptrs()
    telemetry = cache.telemetry()
    assert telemetry["storage"]["staging_pair_allocations"] == 1  # type: ignore[index]
    assert telemetry["storage"]["staging_bytes"] == 192  # type: ignore[index]
    assert telemetry["sequence"]["layer_logical_lengths"] == [8, 8]  # type: ignore[index]
    assert telemetry["sequence"]["completed_update_cycles"] == 2  # type: ignore[index]


def test_frozen_qwen_8k_page_projection_removes_v3_allocator_rounding_waste() -> None:
    cache = PagedSharedStagingCache(
        num_layers=28,
        max_cache_len=40_960,
        page_token_capacity=8_192,
        num_key_value_heads=8,
        key_head_dim=128,
        value_head_dim=128,
        dtype=torch.bfloat16,
        staging_device="cpu",
        pin_cpu_backing=False,
    )
    geometry = cache.geometry

    assert geometry.page_count == 5
    assert geometry.key_page_bytes == geometry.value_page_bytes == 16 * 1024**2
    assert geometry.backing_allocation_count == 280
    assert geometry.logical_backing_bytes == 4_697_620_480  # 4.375 GiB
    assert geometry.predicted_rounded_backing_bytes == 4_697_620_480
    assert geometry.predicted_rounding_waste_bytes == 0
    assert geometry.unpaged_predicted_rounded_backing_bytes == 7 * 1024**3
    assert geometry.staging_key_bytes + geometry.staging_value_bytes == 160 * 1024**2
    assert cache.telemetry()["storage"]["live_logical_backing_bytes"] == 0  # type: ignore[index]


@pytest.mark.parametrize("fail_on_call", range(8))
def test_every_backing_page_allocation_position_is_atomic_and_reset_recovers(
    fail_on_call: int,
) -> None:
    allocator = FailAllocationOnce(fail_on_call)
    cache = make_cache(allocator=allocator)
    failed_layer_idx = fail_on_call // 4

    if failed_layer_idx == 1:
        cache.update(*kv(7), 0)
    with pytest.raises(PagedSharedStagingCacheError, match="backing allocation failed"):
        cache.update(*kv(7, offset=failed_layer_idx * 10_000.0), failed_layer_idx)

    failed = cache.telemetry()
    assert failed["health"]["active_poison_reason"] == "backing_allocation_failure"  # type: ignore[index]
    assert failed["health"]["poison_events_by_reason"] == {  # type: ignore[index]
        "backing_allocation_failure": 1
    }
    assert cache.backing_page_data_ptrs()[failed_layer_idx] is None
    assert failed["storage"]["live_backing_page_allocations"] == failed_layer_idx * 4  # type: ignore[index]
    assert failed["operations"]["backing_allocation_transactions"] == failed_layer_idx  # type: ignore[index]
    with pytest.raises(PagedSharedStagingCacheError, match="poisoned"):
        cache.get_seq_length()

    cache.reset()
    prefills = complete_prefill(cache)
    for layer_idx in range(2):
        assert torch.equal(
            _page_prefix(cache.layers[layer_idx].key_pages, 2),
            prefills[layer_idx][0],
        )
        assert torch.equal(
            _page_prefix(cache.layers[layer_idx].value_pages, 2),
            prefills[layer_idx][1],
        )

    recovered = cache.telemetry()
    assert recovered["health"]["poisoned"] is False  # type: ignore[index]
    assert recovered["health"]["poison_events"] == 1  # type: ignore[index]
    pointers = [
        pointer
        for layer in cache.backing_page_data_ptrs()
        for role in layer or ()
        for pointer in role
    ]
    assert len(pointers) == len(set(pointers)) == 8


@pytest.mark.parametrize("fail_on_call", [8, 9])
def test_both_staging_allocation_positions_are_atomic_and_reset_recovers(
    fail_on_call: int,
) -> None:
    allocator = FailAllocationOnce(fail_on_call)
    cache = make_cache(allocator=allocator)
    complete_prefill(cache, sequence_length=7)
    backing_pointers = cache.backing_page_data_ptrs()

    with pytest.raises(PagedSharedStagingCacheError, match="shared staging failed"):
        cache.update(*kv(1, offset=80_000.0), 0)

    failed = cache.telemetry()
    assert failed["health"]["active_poison_reason"] == (  # type: ignore[index]
        "staging_allocation_or_copy_failure"
    )
    assert cache.staging_data_ptrs() is None
    assert cache.backing_page_data_ptrs() == backing_pointers
    assert failed["storage"]["staging_allocated"] is False  # type: ignore[index]
    assert failed["storage"]["staging_bytes"] == 0  # type: ignore[index]
    assert failed["storage"]["staging_pair_allocations"] == 0  # type: ignore[index]

    cache.reset()
    prefills = complete_prefill(cache)
    for layer_idx in range(2):
        decode = kv(1, offset=90_000.0 + layer_idx * 10_000.0)
        returned = cache.update(*decode, layer_idx)
        assert torch.equal(returned[0], torch.cat((prefills[layer_idx][0], decode[0]), dim=-2))
        assert torch.equal(
            returned[1],
            torch.cat((prefills[layer_idx][1], decode[1]), dim=-2),
        )

    recovered = cache.telemetry()
    assert recovered["health"]["poisoned"] is False  # type: ignore[index]
    assert recovered["health"]["poison_events_by_reason"] == {  # type: ignore[index]
        "staging_allocation_or_copy_failure": 1
    }
    assert recovered["storage"]["staging_pair_allocations"] == 1  # type: ignore[index]


@pytest.mark.parametrize("fail_on_call", range(8))
def test_every_prefill_partial_copy_failure_poisons_until_reset(
    fail_on_call: int,
) -> None:
    copier = MutateThenFailCopier(fail_on_call)
    cache = make_cache(copier=copier)
    failed_layer_idx = fail_on_call // 4

    if failed_layer_idx == 1:
        cache.update(*kv(7), 0)
    with pytest.raises(PagedSharedStagingCacheError, match="backing copy failed"):
        cache.update(*kv(7, offset=failed_layer_idx * 10_000.0), failed_layer_idx)

    failed = cache.telemetry()
    assert failed["health"]["active_poison_reason"] == "backing_copy_failure"  # type: ignore[index]
    assert failed["transfers"]["copies"] == fail_on_call  # type: ignore[index]
    assert copier.call_count == fail_on_call + 1
    with pytest.raises(PagedSharedStagingCacheError, match="poisoned"):
        cache.update(*kv(1), 0)
    with pytest.raises(PagedSharedStagingCacheError, match="poisoned"):
        cache.get_mask_sizes(1, 0)

    cache.reset()
    copier.fail_on_call = -1
    prefills = complete_prefill(cache)
    assert [cache.get_seq_length(index) for index in range(2)] == [2, 2]
    for layer_idx in range(2):
        assert torch.equal(
            _page_prefix(cache.layers[layer_idx].key_pages, 2),
            prefills[layer_idx][0],
        )
        assert torch.equal(
            _page_prefix(cache.layers[layer_idx].value_pages, 2),
            prefills[layer_idx][1],
        )


@pytest.mark.parametrize("fail_on_call", range(8, 14))
def test_every_decode_and_staging_partial_copy_failure_poisons_until_reset(
    fail_on_call: int,
) -> None:
    copier = MutateThenFailCopier(fail_on_call)
    cache = make_cache(copier=copier)
    complete_prefill(cache, sequence_length=7)
    expected_reason = (
        "backing_copy_failure" if fail_on_call < 10 else "staging_allocation_or_copy_failure"
    )

    with pytest.raises(PagedSharedStagingCacheError, match="reset"):
        cache.update(*kv(1, offset=80_000.0), 0)

    failed = cache.telemetry()
    assert failed["health"] == {
        "poisoned": True,
        "active_poison_reason": expected_reason,
        "poison_events": 1,
        "poison_events_by_reason": {expected_reason: 1},
        "cuda_stream_contract": "one_stable_current_stream",
        "cuda_stream_bound": False,
    }
    assert failed["transfers"]["copies"] == fail_on_call  # type: ignore[index]
    assert failed["sequence"]["layer_logical_lengths"] == [7, 7]  # type: ignore[index]
    assert (cache.staging_data_ptrs() is not None) is (fail_on_call >= 10)
    failed_backing_pointers = cache.backing_page_data_ptrs()
    failed_staging_pointers = cache.staging_data_ptrs()
    with pytest.raises(PagedSharedStagingCacheError, match="poisoned"):
        cache.get_seq_length()

    cache.reset()
    assert cache.backing_page_data_ptrs() == failed_backing_pointers
    assert cache.staging_data_ptrs() == failed_staging_pointers
    copier.fail_on_call = -1
    prefills = complete_prefill(cache)
    for layer_idx in range(2):
        decode = kv(1, offset=90_000.0 + layer_idx * 10_000.0)
        returned = cache.update(*decode, layer_idx)
        assert torch.equal(returned[0], torch.cat((prefills[layer_idx][0], decode[0]), dim=-2))
        assert torch.equal(
            returned[1],
            torch.cat((prefills[layer_idx][1], decode[1]), dim=-2),
        )
    assert cache.telemetry()["health"]["poisoned"] is False  # type: ignore[index]


def test_sequential_cycle_violation_poisons_only_after_partial_progress_and_reset_recovers() -> (
    None
):
    cache = make_cache()
    states = kv(2)

    with pytest.raises(PagedSharedStagingCacheError, match="expected 0, got 1"):
        cache.update(*states, 1)
    assert cache.telemetry()["health"]["poisoned"] is False  # type: ignore[index]

    cache.update(*states, 0)
    with pytest.raises(PagedSharedStagingCacheError, match="expected 1, got 0"):
        cache.update(*states, 0)
    failed = cache.telemetry()
    assert failed["health"]["active_poison_reason"] == "partial_update_cycle_failure"  # type: ignore[index]
    assert failed["sequence"]["update_cycle_in_progress"] is True  # type: ignore[index]
    with pytest.raises(PagedSharedStagingCacheError, match="poisoned"):
        cache.update(*kv(2, offset=10_000.0), 1)

    cache.reset()
    complete_prefill(cache)
    recovered = cache.telemetry()
    assert recovered["health"]["poisoned"] is False  # type: ignore[index]
    assert recovered["sequence"]["expected_next_layer_idx"] == 0  # type: ignore[index]
    assert recovered["sequence"]["update_cycle_in_progress"] is False  # type: ignore[index]
    assert recovered["sequence"]["completed_update_cycles"] == 1  # type: ignore[index]


def test_query_length_change_midcycle_poisons_and_reset_recovers() -> None:
    cache = make_cache()
    cache.update(*kv(2), 0)

    with pytest.raises(PagedSharedStagingCacheError, match="query length changed"):
        cache.update(*kv(1, offset=10_000.0), 1)

    failed = cache.telemetry()
    assert failed["health"]["active_poison_reason"] == "partial_update_cycle_failure"  # type: ignore[index]
    assert failed["sequence"]["layer_logical_lengths"] == [2, 0]  # type: ignore[index]
    cache.reset()
    complete_prefill(cache)
    assert [cache.get_seq_length(index) for index in range(2)] == [2, 2]


@pytest.mark.parametrize("event_capacity", [0, 3])
def test_transfer_ledger_retains_a_bounded_tail_and_exact_lifetime_totals(
    event_capacity: int,
) -> None:
    cache = make_cache(telemetry_event_capacity=event_capacity)
    complete_prefill(cache, sequence_length=7)
    for layer_idx in range(2):
        cache.update(*kv(1, offset=70_000.0 + layer_idx * 10_000.0), layer_idx)
    cache.reset()
    complete_prefill(cache)

    telemetry = cache.telemetry()
    transfers = telemetry["transfers"]
    assert transfers["copies"] == 24  # type: ignore[index]
    assert transfers["bytes"] == 864  # type: ignore[index]
    assert transfers["event_capacity"] == event_capacity  # type: ignore[index]
    assert transfers["events_retained"] == event_capacity  # type: ignore[index]
    assert transfers["events_dropped"] == 24 - event_capacity  # type: ignore[index]
    expected_ordinals = [] if event_capacity == 0 else [21, 22, 23]
    assert [event["ordinal"] for event in transfers["events"]] == expected_ordinals  # type: ignore[index]
    assert transfers["by_phase"] == {  # type: ignore[index]
        "backing_page_to_staging": {"copies": 8, "bytes": 384},
        "decode_to_backing_page": {"copies": 4, "bytes": 48},
        "prefill_to_backing_pages": {"copies": 12, "bytes": 432},
    }
    assert telemetry["operations"]["resets"] == 1  # type: ignore[index]


CUDA_PAGED_SHARED_STAGING_ENABLED = (
    os.environ.get("MOLOR_RUN_CUDA_PAGED_SHARED_STAGING_TESTS") == "1" and torch.cuda.is_available()
)


@pytest.mark.skipif(
    not CUDA_PAGED_SHARED_STAGING_ENABLED,
    reason="set MOLOR_RUN_CUDA_PAGED_SHARED_STAGING_TESTS=1 for tiny CUDA systems proof",
)
def test_cuda_paged_roundtrip_sdpa_and_current_stream_contract() -> None:
    from torch.nn import functional as F

    device = "cuda:0"
    cache = make_cache(
        staging_device=device,
        pin_cpu_backing=True,
        non_blocking_copies=True,
    )
    stream = torch.cuda.Stream(device=device)
    prefills: list[tuple[torch.Tensor, torch.Tensor]] = []
    decodes: list[tuple[torch.Tensor, torch.Tensor]] = []
    observed: list[tuple[torch.Tensor, torch.Tensor]] = []
    attention_outputs: list[torch.Tensor] = []
    attention_references: list[torch.Tensor] = []
    pointers: list[tuple[int, int]] = []
    with torch.cuda.stream(stream):
        prefills = complete_prefill(cache, device=device)
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
            query = torch.arange(4, dtype=torch.float32, device=device).reshape(1, 2, 1, 2)
            attention_outputs.append(
                F.scaled_dot_product_attention(query, returned[0], returned[1]).clone()
            )
            expected_keys = torch.cat((prefills[layer_idx][0], decode[0]), dim=-2)
            expected_values = torch.cat((prefills[layer_idx][1], decode[1]), dim=-2)
            attention_references.append(
                F.scaled_dot_product_attention(query, expected_keys, expected_values).clone()
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
        assert torch.equal(
            _page_prefix(cache.layers[layer_idx].key_pages, 3),
            expected_keys.cpu(),
        )
        assert torch.equal(
            _page_prefix(cache.layers[layer_idx].value_pages, 3),
            expected_values.cpu(),
        )
    assert cache.telemetry()["health"]["cuda_stream_bound"] is True  # type: ignore[index]

    other_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(other_stream):
        with pytest.raises(UnsupportedPagedSharedStagingOperation, match="cross-stream"):
            cache.update(*kv(1, offset=99_000.0, device=device), 0)
