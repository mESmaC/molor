from __future__ import annotations

import torch

from molor.experimental.paged_shared_staging_cache import PagedSharedStagingCache


def make_cache() -> PagedSharedStagingCache:
    return PagedSharedStagingCache(
        num_layers=2,
        max_cache_len=8,
        page_token_capacity=4,
        num_key_value_heads=2,
        key_head_dim=2,
        value_head_dim=1,
        dtype=torch.float32,
        staging_device="cpu",
        pin_cpu_backing=False,
        non_blocking_copies=False,
    )


def kv(sequence_length: int, *, offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.arange(4 * sequence_length, dtype=torch.float32).reshape(1, 2, sequence_length, 2)
    values = torch.arange(2 * sequence_length, dtype=torch.float32).reshape(
        1, 2, sequence_length, 1
    )
    return keys + offset, values + offset + 1000.0


def _page_prefix(pages: tuple[torch.Tensor, ...], length: int) -> torch.Tensor:
    return torch.cat(pages, dim=-2)[..., :length, :]


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
