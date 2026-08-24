from dataclasses import FrozenInstanceError

import pytest
import torch

from molor.experimental.hashed_kv import (
    HashedKVConfig,
    PageSpan,
    RandomHyperplaneSimHash,
    build_page_spans,
    evaluate_candidate_tokens,
    evaluate_hashed_kv,
    exact_attention_weights,
    exact_content_digest,
    exact_content_equal,
    select_hashed_pages,
    select_minmax_pages,
    select_oracle_pages,
)


def test_page_spans_and_config_are_immutable_and_validated() -> None:
    pages = build_page_spans(10, 4)
    config = HashedKVConfig(hash_bits=32, hash_tables=3)

    assert pages == (
        PageSpan(0, 0, 4),
        PageSpan(1, 4, 8),
        PageSpan(2, 8, 10),
    )
    assert pages[-1].length == 2
    assert config.routing_metadata_bits_per_token_per_head == 96
    assert config.routing_metadata_bytes_per_token_per_head == 12
    with pytest.raises(FrozenInstanceError):
        pages[0].start = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        HashedKVConfig().hash_bits = 64  # type: ignore[misc]
    with pytest.raises(ValueError, match="page stop"):
        PageSpan(0, 3, 3)
    with pytest.raises(ValueError, match="hash_bits"):
        HashedKVConfig(hash_bits=16)
    with pytest.raises(ValueError, match="candidate_pages"):
        HashedKVConfig(candidate_pages=1, sink_pages=1, recent_pages=1)


@pytest.mark.parametrize(("bits", "tables"), [(32, 1), (64, 3)])
def test_simhash_is_seeded_deterministic_and_has_explicit_bit_shape(
    bits: int, tables: int
) -> None:
    vectors = torch.tensor(
        [[1.0, 0.5, -0.25], [-0.5, 0.25, 1.5], [2.0, -1.0, 0.0]]
    )

    first = RandomHyperplaneSimHash(3, bits=bits, tables=tables, seed=19)
    second = RandomHyperplaneSimHash(3, bits=bits, tables=tables, seed=19)
    different = RandomHyperplaneSimHash(3, bits=bits, tables=tables, seed=20)

    first_codes = first.encode(vectors)
    expected_dtype = torch.int32 if bits == 32 else torch.int64
    assert first_codes.dtype == expected_dtype
    assert first_codes.device.type == "cpu"
    assert first_codes.shape == (3, tables)
    assert first_codes.element_size() * 8 == bits
    assert torch.equal(first_codes, second.encode(vectors))
    assert not torch.equal(first_codes, different.encode(vectors))
    assert torch.equal(first.hyperplanes, second.hyperplanes)

    exposed = first.hyperplanes
    exposed.zero_()
    assert bool(first.hyperplanes.any())


def test_hash_selection_uses_minimum_token_and_table_distance_with_mandatory_pages() -> None:
    pages = build_page_spans(8, 2)
    query_codes = torch.zeros(2, dtype=torch.int32)
    token_codes = torch.full((8, 2), -1, dtype=torch.int32)
    token_codes[2, 0] = -2  # distance 31: not competitive
    token_codes[4, 1] = 0  # page 2 wins because either table may retrieve it

    selection = select_hashed_pages(
        query_codes,
        token_codes,
        pages,
        page_budget=3,
        sink_pages=1,
        recent_pages=1,
    )

    assert selection.page_ids == (0, 2, 3)
    assert selection.mandatory_page_ids == (0, 3)
    assert selection.ranked_page_ids[0] == 2
    assert selection.token_indices == (0, 1, 4, 5, 6, 7)
    assert dict(selection.page_scores) == {0: 32.0, 1: 31.0, 2: 0.0, 3: 32.0}


def test_routing_collision_never_deduplicates_distinct_content() -> None:
    # Positive scalar multiples always share random-hyperplane signs, which gives
    # this test a guaranteed routing collision independent of the random seed.
    keys = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    hasher = RandomHyperplaneSimHash(2, bits=32, seed=7)
    codes = hasher.encode(keys)
    assert torch.equal(codes[0], codes[1])
    namespace = "schema=v1|model=test|route=base|position=post-rope|layer=0"
    assert exact_content_digest(keys[0], namespace=namespace) != exact_content_digest(
        keys[1], namespace=namespace
    )
    assert not exact_content_equal(keys[0], keys[1])

    selection = select_hashed_pages(
        hasher.encode(torch.tensor([1.0, 0.0])),
        codes,
        build_page_spans(2, 2),
        page_budget=1,
        sink_pages=0,
        recent_pages=0,
    )

    assert selection.token_indices == (0, 1)


def test_exact_content_digest_is_metadata_sensitive_and_equality_is_exact() -> None:
    original = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    clone = original.clone()
    reshaped = original.reshape(2)
    lower_precision = original.to(torch.float16)

    namespace = "schema=v1|model=test|route=base|position=post-rope|layer=0"
    assert exact_content_digest(original, namespace=namespace) == exact_content_digest(
        clone, namespace=namespace
    )
    assert exact_content_equal(original, clone)
    assert exact_content_digest(original, namespace=namespace) != exact_content_digest(
        reshaped, namespace=namespace
    )
    assert exact_content_digest(original, namespace=namespace) != exact_content_digest(
        lower_precision, namespace=namespace
    )
    assert exact_content_digest(original, namespace=namespace) != exact_content_digest(
        original, namespace=f"{namespace}|quantizer=int8"
    )
    with pytest.raises(ValueError, match="namespace"):
        exact_content_digest(original, namespace="")
    assert not exact_content_equal(original, reshaped)
    assert not exact_content_equal(original, lower_precision)

    positive_zero = torch.tensor([0.0])
    negative_zero = torch.tensor([-0.0])
    nan_value = torch.tensor([float("nan")])
    assert not exact_content_equal(positive_zero, negative_zero)
    assert exact_content_equal(nan_value, nan_value.clone())


def test_oracle_ranks_exact_page_attention_mass_but_keeps_sink_and_recent() -> None:
    pages = build_page_spans(8, 2)
    weights = torch.tensor([0.03, 0.02, 0.04, 0.06, 0.40, 0.40, 0.03, 0.02])

    selection = select_oracle_pages(
        weights,
        pages,
        page_budget=3,
        sink_pages=1,
        recent_pages=1,
    )

    assert selection.page_ids == (0, 2, 3)
    assert selection.mandatory_page_ids == (0, 3)
    assert selection.ranked_page_ids[0] == 2
    assert dict(selection.page_scores)[2] == pytest.approx(0.8)


def test_minmax_selector_ranks_dot_product_upper_bound_with_mandatory_pages() -> None:
    query = torch.tensor([2.0, -1.0])
    keys = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 1.0],
            [1.0, -1.0],
            [1.0, -5.0],
            [3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    pages = build_page_spans(8, 2)

    selection = select_minmax_pages(
        query,
        keys,
        pages,
        3,
        1,
        1,
    )

    assert selection.page_ids == (0, 2, 3)
    assert selection.mandatory_page_ids == (0, 3)
    assert selection.ranked_page_ids[0] == 2
    scores = dict(selection.page_scores)
    assert scores[1] == pytest.approx(5 / (2**0.5))
    assert scores[2] == pytest.approx(11 / (2**0.5))


def test_minmax_selector_uses_page_id_as_stable_score_tie_break() -> None:
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[0.0, 0.0], [2.0, 1.0], [2.0, -1.0], [0.0, 0.0]])
    pages = build_page_spans(4, 1)

    selection = select_minmax_pages(
        query,
        keys,
        pages,
        page_budget=1,
        sink_pages=0,
        recent_pages=0,
    )

    assert selection.ranked_page_ids[:2] == (1, 2)
    assert selection.page_ids == (1,)


def test_candidate_metrics_measure_quality_and_token_and_page_fractions() -> None:
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor(
        [[0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [-1.0, 0.0]]
    )
    values = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    )
    candidates = (1, 3)

    metrics = evaluate_candidate_tokens(
        query,
        keys,
        values,
        candidates,
        top_k=1,
        candidate_page_count=1,
        total_page_count=2,
    )
    exact_weights = exact_attention_weights(query, keys)

    assert metrics.top_k == 1
    assert metrics.top_k_recall == 1.0
    assert metrics.retained_attention_mass == pytest.approx(
        float(exact_weights[list(candidates)].sum().item())
    )
    assert metrics.relative_output_l2 > 0
    assert -1.0 <= metrics.cosine_similarity <= 1.0
    assert metrics.candidate_token_fraction == 0.5
    assert metrics.candidate_page_fraction == 0.5

    full = evaluate_candidate_tokens(
        query,
        keys,
        values,
        range(4),
        top_k=99,
        candidate_page_count=2,
        total_page_count=2,
    )
    assert full.top_k == 4
    assert full.top_k_recall == 1.0
    assert full.retained_attention_mass == pytest.approx(1.0)
    assert full.relative_output_l2 == pytest.approx(0.0, abs=1e-7)
    assert full.cosine_similarity == pytest.approx(1.0)
    assert full.candidate_token_fraction == 1.0
    assert full.candidate_page_fraction == 1.0


def test_shadow_evaluation_is_deterministic_and_compares_equal_budget_oracle() -> None:
    generator = torch.Generator().manual_seed(123)
    query = torch.randn(8, generator=generator)
    keys = torch.randn(24, 8, generator=generator)
    values = torch.randn(24, 6, generator=generator)
    config = HashedKVConfig(
        page_size=4,
        hash_bits=64,
        hash_tables=2,
        candidate_pages=3,
        sink_pages=1,
        recent_pages=1,
        top_k=5,
        seed=41,
    )

    first = evaluate_hashed_kv(query, keys, values, config=config)
    second = evaluate_hashed_kv(query, keys, values, config=config)

    assert first.hashed_selection == second.hashed_selection
    assert first.minmax_selection == second.minmax_selection
    assert first.oracle_selection == second.oracle_selection
    assert first.hashed_metrics == second.hashed_metrics
    assert first.minmax_metrics == second.minmax_metrics
    assert first.oracle_metrics == second.oracle_metrics
    assert first.forced_selection == second.forced_selection
    assert first.forced_metrics == second.forced_metrics
    assert first.head_dim == 8
    assert torch.equal(first.query_routing_codes, second.query_routing_codes)
    assert torch.equal(first.key_routing_codes, second.key_routing_codes)
    assert (
        len(first.hashed_selection.page_ids)
        == len(first.minmax_selection.page_ids)
        == len(first.oracle_selection.page_ids)
        == 3
    )
    assert first.oracle_metrics.retained_attention_mass >= (
        first.hashed_metrics.retained_attention_mass - 1e-7
    )
    assert first.oracle_metrics.retained_attention_mass >= (
        first.minmax_metrics.retained_attention_mass - 1e-7
    )
    assert first.hashed_metrics.candidate_token_fraction == 0.5
    assert first.hashed_metrics.candidate_page_fraction == 0.5
    assert first.minmax_metrics.candidate_token_fraction == 0.5
    assert first.minmax_metrics.candidate_page_fraction == 0.5


def test_falsy_non_config_values_are_rejected() -> None:
    query = torch.tensor([1.0, 0.0])
    keys = torch.tensor([[1.0, 0.0]])
    values = torch.tensor([[1.0, 0.0]])

    with pytest.raises(TypeError, match="HashedKVConfig"):
        evaluate_hashed_kv(query, keys, values, config=False)  # type: ignore[arg-type]
