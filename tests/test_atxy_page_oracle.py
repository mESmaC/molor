from itertools import combinations

import pytest
import torch

from molor.experimental.atxy_page_oracle import (
    build_exact_page_oracle_curve,
    build_exact_page_oracle_curve_from_weights,
)
from molor.experimental.hashed_kv import (
    build_page_spans,
    evaluate_candidate_tokens,
    exact_attention_weights,
    select_oracle_pages,
)


def test_cumulative_oracle_matches_existing_exact_reference_on_partial_pages() -> None:
    generator = torch.Generator().manual_seed(419)
    query = torch.randn(5, generator=generator)
    keys = torch.randn(11, 5, generator=generator)
    values = torch.randn(11, 3, generator=generator)
    curve = build_exact_page_oracle_curve(
        query,
        keys,
        values,
        page_size=4,
        sink_pages=1,
        recent_pages=1,
        top_k=5,
    )

    metrics = curve.metrics_for_pages(3, support_token_spans=((4, 6),))
    reference_selection = select_oracle_pages(
        exact_attention_weights(query, keys),
        build_page_spans(11, 4),
        page_budget=3,
        sink_pages=1,
        recent_pages=1,
    )
    reference_metrics = evaluate_candidate_tokens(
        query,
        keys,
        values,
        reference_selection.token_indices,
        top_k=5,
        candidate_page_count=3,
        total_page_count=3,
    )

    assert metrics.selected_page_ids == reference_selection.page_ids
    assert metrics.candidate_tokens == 11
    assert metrics.retained_attention_mass == 1.0
    assert metrics.top_k_recall == reference_metrics.top_k_recall == 1.0
    assert metrics.output_relative_l2 == pytest.approx(0.0, abs=1e-8)
    assert metrics.output_cosine_similarity == pytest.approx(1.0)
    assert metrics.support_page_recall == 1.0


def test_vectorized_page_statistics_match_manual_partial_page_reductions() -> None:
    generator = torch.Generator().manual_seed(420)
    weights = torch.softmax(torch.randn(13, generator=generator), dim=0)
    values = torch.randn(13, 7, generator=generator)

    curve = build_exact_page_oracle_curve_from_weights(
        weights,
        values,
        page_size=4,
        sink_pages=0,
        recent_pages=0,
    )

    expected_masses = []
    expected_weighted_values = []
    for page in build_page_spans(13, 4):
        page_weights = weights[page.start : page.stop]
        page_values = values[page.start : page.stop]
        expected_masses.append(page_weights.sum())
        expected_weighted_values.append(torch.matmul(page_weights, page_values))
    assert torch.allclose(curve.page_masses, torch.stack(expected_masses), atol=1e-7)
    assert torch.allclose(
        curve.page_weighted_values,
        torch.stack(expected_weighted_values),
        atol=1e-6,
    )


def test_oracle_is_mass_optimal_at_fixed_budget_with_mandatory_pages() -> None:
    query = torch.tensor([1.0])
    keys = torch.tensor([[0.0], [0.0], [1.0], [1.0], [5.0], [5.0], [-1.0]])
    values = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    curve = build_exact_page_oracle_curve(
        query,
        keys,
        values,
        page_size=2,
        sink_pages=1,
        recent_pages=1,
        top_k=3,
    )

    metrics = curve.metrics_for_pages(3, support_token_spans=((4, 5),))
    weights = exact_attention_weights(query, keys)
    pages = build_page_spans(7, 2)
    mandatory = set(curve.mandatory_page_ids)
    alternatives = []
    for extra in combinations(set(range(len(pages))) - mandatory, 1):
        selected = mandatory.union(extra)
        alternatives.append(
            float(
                sum(
                    weights[page.start : page.stop].sum()
                    for page in pages
                    if page.page_id in selected
                ).item()
            )
        )

    assert metrics.selected_page_ids == (0, 2, 3)
    assert metrics.retained_attention_mass == pytest.approx(max(alternatives))
    assert metrics.maximum_attention_token_page_recalled
    assert metrics.top_attention_page_recall_1 == 1.0
    assert metrics.support_page_recall == 1.0
    assert metrics.cold_attention_mass_recall is not None


def test_oracle_ties_use_page_id_and_mass_threshold_curve_is_monotonic() -> None:
    query = torch.zeros(2)
    keys = torch.zeros(8, 2)
    values = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    curve = build_exact_page_oracle_curve(
        query,
        keys,
        values,
        page_size=2,
        sink_pages=0,
        recent_pages=0,
        top_k=8,
    )

    assert curve.mandatory_page_ids == ()
    assert curve.oracle_ranked_page_ids == (0, 1, 2, 3)
    assert curve.metrics_for_pages(1).selected_page_ids == (0,)
    assert curve.minimum_candidate_pages_for_mass(0.25) == 1
    assert curve.minimum_candidate_pages_for_mass(0.50) == 2
    assert curve.minimum_candidate_pages_for_mass(0.99) == 4
    assert curve.candidate_pages_for_fraction(0.26) == 2


def test_oracle_reports_forced_and_cold_attribution_and_rejects_bad_inputs() -> None:
    query = torch.tensor([1.0, -0.5])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0]])
    values = torch.eye(4)
    curve = build_exact_page_oracle_curve(
        query,
        keys,
        values,
        page_size=1,
        sink_pages=1,
        recent_pages=1,
        top_k=2,
    )

    metrics = curve.metrics_for_fraction(0.75, support_token_spans=((1, 2),))

    assert metrics.candidate_pages == 3
    assert metrics.candidate_page_fraction == 0.75
    assert metrics.candidate_token_fraction == 0.75
    assert metrics.retained_attention_mass == pytest.approx(
        metrics.forced_attention_mass + metrics.incremental_attention_mass_over_forced
    )
    assert 0 <= metrics.cold_attention_mass_recall <= 1
    with pytest.raises(ValueError, match="mandatory"):
        curve.metrics_for_pages(1)
    with pytest.raises(ValueError, match="support token span"):
        curve.metrics_for_pages(3, support_token_spans=((0, 5),))
    with pytest.raises(ValueError, match="candidate fraction"):
        curve.metrics_for_fraction(0.0)
    with pytest.raises(ValueError, match="floating-point"):
        build_exact_page_oracle_curve(query, keys, torch.ones(4, 2, dtype=torch.int64), page_size=2)


def test_precomputed_attention_builder_reuses_validated_unit_mass_weights() -> None:
    query = torch.tensor([0.5, -1.0])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-0.5, 0.5]])
    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    weights = exact_attention_weights(query, keys)

    small = build_exact_page_oracle_curve_from_weights(weights, values, page_size=1)
    large = build_exact_page_oracle_curve_from_weights(weights, values, page_size=2)

    assert small.total_tokens == large.total_tokens == 4
    assert torch.equal(small.exact_output, large.exact_output)
    assert small.metrics_for_pages(4).output_relative_l2 == 0.0
    assert large.metrics_for_pages(2).output_relative_l2 == 0.0
    with pytest.raises(ValueError, match="unit mass"):
        build_exact_page_oracle_curve_from_weights(weights * 0.5, values, page_size=2)
    with pytest.raises(ValueError, match="non-negative"):
        build_exact_page_oracle_curve_from_weights(
            torch.tensor([0.5, -0.1, 0.3, 0.3]), values, page_size=2
        )


def test_long_extreme_attention_is_stably_normalized_before_page_scoring() -> None:
    token_count = 32_768
    query = torch.tensor([1.0])
    keys = torch.linspace(-40.0, 40.0, token_count).unsqueeze(1)
    values = torch.linspace(-1.0, 1.0, token_count).unsqueeze(1)

    weights = exact_attention_weights(query, keys)
    curve = build_exact_page_oracle_curve_from_weights(
        weights,
        values,
        page_size=256,
        sink_pages=1,
        recent_pages=1,
    )

    assert weights.dtype is torch.float32
    assert float(weights.sum(dtype=torch.float64)) == pytest.approx(1.0, abs=1e-7)
    assert curve.total_tokens == token_count
    assert float(curve.page_masses.sum(dtype=torch.float64)) == pytest.approx(1.0, abs=1e-6)
