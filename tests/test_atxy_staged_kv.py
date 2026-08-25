from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from molor.experimental.atxy_staged_kv import (
    AtxyLayerPageIndex,
    OnlineBlockwiseAttentionAccumulator,
    ProgressiveLayerSharedSelector,
    _exp_lower,
    _exp_upper,
    _subtract_lower,
    _subtract_upper,
    conservative_page_logit_upper_bounds,
)
from molor.experimental.hashed_kv import PageSpan


def _page_logits(query: Tensor, keys: Tensor) -> Tensor:
    query_value = query.to(dtype=torch.float64)
    keys_value = keys.to(dtype=torch.float64)
    query_heads, head_dim = query_value.shape
    kv_heads = keys_value.shape[0]
    groups = query_heads // kv_heads
    kv_head_ids = torch.arange(query_heads, dtype=torch.long) // groups
    return torch.einsum(
        "hd,htd->ht", query_value, keys_value.index_select(0, kv_head_ids)
    ) / math.sqrt(head_dim)


def _observe_pending(
    selector: ProgressiveLayerSharedSelector,
    index: AtxyLayerPageIndex,
    keys: Tensor,
):
    pending = selector.next_page_ids()
    page_keys = {
        page_id: keys[:, index.page(page_id).span.start : index.page(page_id).span.stop]
        for page_id in pending
    }
    return selector.observe_exact_page_keys(page_keys)


def _actual_retained_mass(
    query: Tensor,
    keys: Tensor,
    index: AtxyLayerPageIndex,
    selected_page_ids: tuple[int, ...],
) -> Tensor:
    full_logits = _page_logits(query, keys)
    selected_tokens = [
        token
        for page_id in selected_page_ids
        for token in range(index.page(page_id).span.start, index.page(page_id).span.stop)
    ]
    return torch.softmax(full_logits, dim=1)[:, selected_tokens].sum(dim=1)


def test_exact_descriptors_bound_every_logit_and_account_partial_page_bytes() -> None:
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(2, 11, 5, generator=generator, dtype=torch.float32)
    values = torch.randn(2, 11, 3, generator=generator, dtype=torch.float32)
    query = torch.randn(4, 5, generator=generator, dtype=torch.float32)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=7,
        keys=keys,
        values=values,
        page_size=4,
    )

    bounds = conservative_page_logit_upper_bounds(query, index)

    assert bounds.shape == (4, 3)
    assert tuple(page.span.length for page in index.pages) == (4, 4, 3)
    assert tuple(page.address.text for page in index.pages) == ("7.000", "7.001", "7.002")
    assert index.total_payload_nbytes == (
        keys.numel() * keys.element_size() + values.numel() * values.element_size()
    )
    for page in index.pages:
        logits = _page_logits(query, keys[:, page.span.start : page.span.stop])
        assert bool((logits.amax(dim=1) <= bounds[:, page.span.page_id]).all())

    exposed = index.pages[0].key_minimum
    exposed.zero_()
    assert bool(index.pages[0].key_minimum.any())
    assert len(index.pages[0].key_fingerprint_sha256) == 64
    assert len(index.pages[0].value_fingerprint_sha256) == 64
    changed_values = AtxyLayerPageIndex.from_kv(
        layer_index=7,
        keys=keys,
        values=values + 1,
        page_size=4,
    )
    assert changed_values.page(0).key_fingerprint_sha256 == index.page(0).key_fingerprint_sha256
    assert changed_values.page(0).value_fingerprint_sha256 != index.page(0).value_fingerprint_sha256


def test_descriptor_fingerprint_rejects_fake_keys_with_identical_minmax_bounds() -> None:
    keys = torch.tensor([[[-2.0], [-1.0], [1.0], [2.0], [-10.0], [-10.0], [-10.0], [-10.0]]])
    values = torch.ones(1, 8, 1)
    query = torch.ones(1, 1)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=0,
        keys=keys,
        values=values,
        page_size=4,
    )
    selector = ProgressiveLayerSharedSelector(
        index=index,
        query=query,
        mandatory_page_ids=(0,),
        target_mass=0.9,
        page_cap=1,
        initial_pages=1,
    )
    fake_page = torch.tensor([[[-2.0], [2.0], [2.0], [2.0]]])

    assert torch.equal(fake_page.amin(dim=1), index.page(0).key_minimum)
    assert torch.equal(fake_page.amax(dim=1), index.page(0).key_maximum)
    assert bool(
        (
            _page_logits(query, fake_page).amax(dim=1)
            <= conservative_page_logit_upper_bounds(query, index)[:, 0]
        ).all()
    )
    assert not hasattr(selector, "observe_exact_logits")
    with pytest.raises(ValueError, match="fingerprint"):
        selector.observe_exact_page_keys({0: fake_page})
    assert selector.probed_page_ids == ()
    assert selector.probed_retained_mass_lower_bounds is None

    decision = selector.observe_exact_page_keys({0: keys[:, :4]})
    assert decision is not None
    assert decision.dense_fallback is False


def test_directed_subtraction_brackets_catastrophic_cancellation_before_exp() -> None:
    left = 1.0e16
    right = 1.0e16

    lower = _subtract_lower(left, right)
    upper = _subtract_upper(left, right)

    assert lower < 0.0 < upper
    assert _exp_lower(lower) < 1.0 < _exp_upper(upper)


@pytest.mark.parametrize("seed", [0, 7, 29, 71, 97])
def test_progressive_mass_certificates_are_conservative_at_every_round(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(2, 13, 4, generator=generator)
    values = torch.randn(2, 13, 3, generator=generator)
    query = torch.randn(4, 4, generator=generator)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=2,
        keys=keys,
        values=values,
        page_size=3,
    )
    selector = ProgressiveLayerSharedSelector(
        index=index,
        query=query,
        mandatory_page_ids=(0, 4),
        target_mass=0.999,
        page_cap=5,
        initial_pages=2,
        pages_per_round=1,
    )

    while selector.decision is None:
        _observe_pending(selector, index, keys)
        lower = selector.probed_retained_mass_lower_bounds
        assert lower is not None
        if set(selector.probed_page_ids) == set(index.page_ids):
            assert lower == (1.0,) * 4
        else:
            actual = _actual_retained_mass(query, keys, index, selector.probed_page_ids)
            assert bool((torch.tensor(lower, dtype=torch.float64) <= actual).all())

    assert selector.decision is not None
    assert selector.decision.dense_fallback is False
    assert selector.decision.selected_page_ids == tuple(sorted(selector.probed_page_ids))
    assert all(lower >= 0.999 for lower in selector.decision.selected_retained_mass_lower_bounds)


def test_layer_shared_gqa_selection_is_deterministic_and_uses_one_physical_union() -> None:
    keys = torch.tensor(
        [
            [[4.0], [4.0], [0.0], [0.0], [-4.0], [-4.0], [-6.0], [-6.0]],
            [[-4.0], [-4.0], [0.0], [0.0], [4.0], [4.0], [-6.0], [-6.0]],
        ]
    )
    values = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)
    query = torch.ones(4, 1)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=3,
        keys=keys,
        values=values,
        page_size=2,
    )

    decisions = []
    for _ in range(2):
        selector = ProgressiveLayerSharedSelector(
            index=index,
            query=query,
            mandatory_page_ids=(3,),
            target_mass=0.97,
            page_cap=3,
            initial_pages=1,
            pages_per_round=1,
        )
        while selector.decision is None:
            _observe_pending(selector, index, keys)
        decisions.append(selector.decision)

    assert decisions[0] == decisions[1]
    assert decisions[0] is not None
    assert decisions[0].dense_fallback is False
    assert decisions[0].selected_page_ids == (0, 2, 3)
    assert decisions[0].probed_page_ids == (3, 0, 2)
    assert len(decisions[0].probed_retained_mass_lower_bounds) == 4
    assert decisions[0].selected_retained_mass_lower_bounds == (
        decisions[0].probed_retained_mass_lower_bounds
    )


def test_larger_initial_round_never_duplicates_a_mandatory_page() -> None:
    keys = torch.tensor([[[5.0], [5.0], [0.0], [0.0], [-5.0], [-5.0]]])
    values = torch.ones(1, 6, 1)
    query = torch.ones(2, 1)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=1,
        keys=keys,
        values=values,
        page_size=2,
    )
    selector = ProgressiveLayerSharedSelector(
        index=index,
        query=query,
        mandatory_page_ids=(0,),
        target_mass=0.9,
        page_cap=2,
        initial_pages=2,
    )

    assert selector.next_page_ids() == (0, 1)
    assert len(set(selector.next_page_ids())) == 2


def test_page_cap_triggers_explicit_dense_fallback_and_full_transfer_accounting() -> None:
    keys = torch.zeros(1, 8, 2)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    query = torch.ones(2, 2)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=0,
        keys=keys,
        values=values,
        page_size=2,
    )
    selector = ProgressiveLayerSharedSelector(
        index=index,
        query=query,
        mandatory_page_ids=(0,),
        target_mass=0.99,
        page_cap=2,
        initial_pages=1,
        pages_per_round=1,
    )

    while selector.decision is None:
        _observe_pending(selector, index, keys)

    decision = selector.decision
    assert decision is not None
    assert decision.dense_fallback is True
    assert decision.fallback_reason == "page-cap-exhausted-before-mass-certificate"
    assert decision.selected_page_ids == index.page_ids
    assert len(decision.probed_page_ids) == 2
    assert decision.probed_retained_mass_lower_bounds == pytest.approx((0.5, 0.5))
    assert decision.selected_retained_mass_lower_bounds == (1.0, 1.0)
    assert decision.total_transfer_nbytes == index.total_payload_nbytes
    assert decision.probe_transfer_nbytes + decision.fallback_transfer_nbytes == (
        decision.total_transfer_nbytes
    )


def test_target_mass_one_requires_observing_every_page() -> None:
    keys = torch.tensor([[[0.0], [-40.0]]])
    values = torch.ones(1, 2, 1)
    query = torch.ones(1, 1)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=0,
        keys=keys,
        values=values,
        page_size=1,
    )
    selector = ProgressiveLayerSharedSelector(
        index=index,
        query=query,
        mandatory_page_ids=(0,),
        target_mass=1.0,
        page_cap=2,
        initial_pages=1,
    )

    assert _observe_pending(selector, index, keys) is None
    assert selector.next_page_ids() == (1,)
    decision = _observe_pending(selector, index, keys)
    assert decision is not None
    assert decision.selected_page_ids == (0, 1)
    assert decision.probed_retained_mass_lower_bounds == (1.0,)
    assert decision.selected_retained_mass_lower_bounds == (1.0,)


def test_invalid_multi_page_observation_is_atomic_and_retryable() -> None:
    keys = torch.tensor([[[2.0], [2.0], [0.0], [0.0]]])
    values = torch.ones(1, 4, 1)
    query = torch.ones(1, 1)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=0,
        keys=keys,
        values=values,
        page_size=2,
    )
    selector = ProgressiveLayerSharedSelector(
        index=index,
        query=query,
        target_mass=0.9,
        page_cap=2,
        initial_pages=2,
    )
    pending = selector.next_page_ids()
    valid = {
        page_id: keys[:, index.page(page_id).span.start : index.page(page_id).span.stop]
        for page_id in pending
    }
    invalid = dict(valid)
    invalid[pending[-1]] = torch.zeros(1, 1, 1)

    with pytest.raises(ValueError, match="invalid shape"):
        selector.observe_exact_page_keys(invalid)
    assert selector.probed_page_ids == ()
    assert selector.probed_retained_mass_lower_bounds is None
    assert selector.next_page_ids() == pending

    decision = selector.observe_exact_page_keys(valid)
    assert decision is not None
    assert decision.probed_page_ids == pending


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_online_blockwise_attention_matches_dense_for_any_partial_page_order(
    order: tuple[int, ...],
) -> None:
    generator = torch.Generator().manual_seed(43)
    query = torch.randn(4, 5, generator=generator)
    keys = torch.randn(2, 10, 5, generator=generator)
    values = torch.randn(2, 10, 3, generator=generator)
    spans = ((0, 4), (4, 8), (8, 10))
    expected_pages = tuple(
        PageSpan(page_id, start, stop) for page_id, (start, stop) in enumerate(spans)
    )
    accumulator = OnlineBlockwiseAttentionAccumulator(query, expected_pages=expected_pages)

    for page_id in order:
        start, stop = spans[page_id]
        accumulator.update(page_id, keys[:, start:stop], values[:, start:stop])

    logits = _page_logits(query, keys)
    kv_head_ids = torch.arange(4, dtype=torch.long) // 2
    dense = torch.einsum(
        "ht,htv->hv",
        torch.softmax(logits, dim=1),
        values.to(torch.float64).index_select(0, kv_head_ids),
    )

    assert accumulator.token_count == 10
    assert torch.allclose(accumulator.result(), dense, atol=1e-12, rtol=1e-12)


def test_online_attention_is_stable_for_extreme_logits() -> None:
    query = torch.tensor([[1000.0, -1000.0]])
    keys = torch.tensor([[[1000.0, -1000.0], [-1000.0, 1000.0]]])
    values = torch.tensor([[[3.0, 4.0], [-7.0, 2.0]]])
    accumulator = OnlineBlockwiseAttentionAccumulator(
        query,
        expected_pages=(PageSpan(0, 0, 1), PageSpan(1, 1, 2)),
    )

    accumulator.update(1, keys[:, 1:], values[:, 1:])
    accumulator.update(0, keys[:, :1], values[:, :1])

    assert torch.allclose(accumulator.result(), torch.tensor([[3.0, 4.0]], dtype=torch.float64))


def test_online_attention_rejects_duplicate_and_omitted_physical_pages() -> None:
    query = torch.ones(2, 2)
    keys = torch.ones(1, 4, 2)
    values = torch.ones(1, 4, 3)
    accumulator = OnlineBlockwiseAttentionAccumulator(
        query,
        expected_pages=(PageSpan(7, 0, 2), PageSpan(9, 2, 4)),
    )

    accumulator.update(7, keys[:, :2], values[:, :2])
    with pytest.raises(RuntimeError, match="missing"):
        accumulator.result()
    with pytest.raises(ValueError, match="already accumulated"):
        accumulator.update(7, keys[:, :2], values[:, :2])
    assert accumulator.token_count == 2

    accumulator.update(9, keys[:, 2:], values[:, 2:])
    assert accumulator.result().shape == (2, 3)


def test_online_attention_rejects_invalid_gqa_and_later_block_geometry_atomically() -> None:
    query = torch.ones(4, 2)
    expected = (PageSpan(0, 0, 2), PageSpan(1, 2, 4))
    invalid_gqa = OnlineBlockwiseAttentionAccumulator(query, expected_pages=expected)
    with pytest.raises(ValueError, match="divisible"):
        invalid_gqa.update(0, torch.ones(3, 2, 2), torch.ones(3, 2, 1))
    assert invalid_gqa.token_count == 0

    accumulator = OnlineBlockwiseAttentionAccumulator(query, expected_pages=expected)
    accumulator.update(0, torch.ones(2, 2, 2), torch.ones(2, 2, 3))
    with pytest.raises(ValueError, match="same GQA and value geometry"):
        accumulator.update(1, torch.ones(1, 2, 2), torch.ones(1, 2, 4))
    assert accumulator.token_count == 2

    accumulator.update(1, torch.ones(2, 2, 2), torch.ones(2, 2, 3))
    fresh = OnlineBlockwiseAttentionAccumulator(query, expected_pages=expected)
    fresh.update(0, torch.ones(2, 2, 2), torch.ones(2, 2, 3))
    fresh.update(1, torch.ones(2, 2, 2), torch.ones(2, 2, 3))
    assert torch.equal(accumulator.result(), fresh.result())


def test_multi_token_decode_and_prefill_shapes_fail_closed() -> None:
    keys = torch.ones(1, 4, 2)
    values = torch.ones(1, 4, 2)
    index = AtxyLayerPageIndex.from_kv(
        layer_index=0,
        keys=keys,
        values=values,
        page_size=2,
    )
    multi_token_query = torch.ones(2, 3, 2)

    with pytest.raises(ValueError, match="multi-token decode"):
        conservative_page_logit_upper_bounds(multi_token_query, index)
    with pytest.raises(ValueError, match="multi-token decode"):
        ProgressiveLayerSharedSelector(
            index=index,
            query=multi_token_query,
            page_cap=1,
        )
    with pytest.raises(ValueError, match="multi-token decode"):
        OnlineBlockwiseAttentionAccumulator(
            multi_token_query,
            expected_pages=(PageSpan(0, 0, 1),),
        )
