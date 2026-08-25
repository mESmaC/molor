from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from molor.experimental.atxy_local_latent import (
    AtxyLocalLatentCodec,
    AtxyLocalStorageBytes,
    GQA_QUERY_HEADS_PER_KV_HEAD,
)


def _random_calibration(
    *,
    layers: int = 2,
    tokens: int = 9,
    query_positions: int = 5,
    kv_heads: int = 2,
    channels: int = 6,
    seed: int = 1,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.randn(
        layers,
        tokens,
        kv_heads,
        channels,
        generator=generator,
        dtype=dtype,
    )
    queries = torch.randn(
        layers,
        query_positions,
        kv_heads * GQA_QUERY_HEADS_PER_KV_HEAD,
        channels,
        generator=generator,
        dtype=dtype,
    )
    values = (
        torch.randn(
            layers,
            tokens,
            kv_heads,
            channels,
            generator=generator,
            dtype=dtype,
        )
        * 2.0
        + 3.0
    )
    return keys, queries, values


def _dense_gqa_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded_keys = keys.repeat_interleave(GQA_QUERY_HEADS_PER_KV_HEAD, dim=2)
    expanded_values = values.repeat_interleave(GQA_QUERY_HEADS_PER_KV_HEAD, dim=2)
    scores = torch.einsum("ashy,athy->asht", queries, expanded_keys)
    weights = torch.softmax(scores / math.sqrt(keys.shape[-1]), dim=-1)
    output = torch.einsum("asht,athy->ashy", weights, expanded_values)
    return weights, output


def test_fit_produces_orthonormal_local_bases_and_nested_rank_prefixes() -> None:
    keys, queries, values = _random_calibration(seed=10)
    codec = AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=5)

    for basis in (codec.key_basis, codec.value_basis):
        gram = torch.einsum("axyr,axys->axrs", basis, basis)
        expected = torch.eye(5).expand(2, 2, 5, 5)
        torch.testing.assert_close(gram, expected, rtol=1e-5, atol=2e-6)

    assert torch.equal(codec.key_basis_prefix(3), codec.key_basis_prefix(5)[..., :3])
    assert torch.equal(codec.value_basis_prefix(2), codec.value_basis_prefix(5)[..., :2])
    assert torch.equal(
        codec.encode_keys(keys, rank=3),
        codec.encode_keys(keys, rank=5)[..., :3],
    )
    assert torch.equal(
        codec.encode_values(values, rank=2),
        codec.encode_values(values, rank=5)[..., :2],
    )
    exposed = codec.key_basis
    exposed.zero_()
    assert bool(codec.key_basis.any())
    with pytest.raises(FrozenInstanceError):
        codec.max_rank = 3  # type: ignore[misc]


def test_balanced_key_query_moment_is_sample_count_independent_and_values_are_centered() -> None:
    keys = torch.tensor([[[[2.0, 0.0]]]])
    queries = torch.zeros(1, 50, 2, 2)
    queries[..., 1] = 1.0
    values = torch.tensor([[[[9.0, -4.0]]]])

    codec = AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=2)

    # Equal source weighting uses mean second moments: 50 query positions do
    # not outvote the one key vector merely by having more samples.
    torch.testing.assert_close(codec.key_basis[0, 0, :, 0].abs(), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(codec.value_center, values[:, 0])
    encoded_values = codec.encode_values(values, rank=1)
    assert torch.equal(encoded_values, torch.zeros_like(encoded_values))
    torch.testing.assert_close(codec.decode_values(encoded_values), values)


def test_fit_and_ranked_encodings_are_deterministic() -> None:
    calibration = _random_calibration(seed=20)
    first = AtxyLocalLatentCodec.fit(*calibration, max_rank=4)
    second = AtxyLocalLatentCodec.fit(*calibration, max_rank=4)

    assert torch.equal(first.key_basis, second.key_basis)
    assert torch.equal(first.value_basis, second.value_basis)
    assert torch.equal(first.value_center, second.value_center)
    assert torch.equal(
        first.encode_keys(calibration[0], rank=3),
        second.encode_keys(calibration[0], rank=3),
    )
    assert torch.equal(
        first.encode_queries(calibration[1], rank=3),
        second.encode_queries(calibration[1], rank=3),
    )


def test_rank_128_matches_native_gqa_attention_and_round_trips() -> None:
    keys, queries, values = _random_calibration(
        layers=1,
        tokens=11,
        query_positions=4,
        kv_heads=1,
        channels=128,
        seed=30,
    )
    codec = AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=128)
    key_latents = codec.encode_keys(keys, rank=128)
    query_latents = codec.encode_queries(queries, rank=128)
    value_latents = codec.encode_values(values, rank=128)

    latent = codec.latent_attention(query_latents, key_latents, value_latents)
    dense_weights, dense_output = _dense_gqa_attention(queries, keys, values)

    torch.testing.assert_close(codec.decode_keys(key_latents), keys, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(codec.decode_queries(query_latents), queries, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(codec.decode_values(value_latents), values, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(latent.weights, dense_weights, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(latent.output, dense_output, rtol=3e-5, atol=3e-5)


def test_query_projection_and_attention_preserve_two_to_one_gqa_mapping() -> None:
    keys, queries, values = _random_calibration(
        layers=1,
        tokens=3,
        query_positions=2,
        kv_heads=2,
        channels=3,
        seed=40,
    )
    codec = AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=3)
    encoded_queries = codec.encode_queries(queries, rank=2)
    grouped_basis = codec.key_basis_prefix(2).repeat_interleave(2, dim=1)
    manual_queries = torch.einsum("ashy,ahyr->ashr", queries, grouped_basis)
    torch.testing.assert_close(encoded_queries, manual_queries)

    query_latents = torch.tensor([[[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]]])
    key_latents = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
    value_latents = codec.encode_values(values[:, :2], rank=2)
    result = codec.latent_attention(query_latents, key_latents, value_latents)

    expanded_keys = key_latents.repeat_interleave(2, dim=2)
    expected_scores = torch.einsum("ashr,athr->asht", query_latents, expanded_keys)
    expected_weights = torch.softmax(expected_scores / math.sqrt(3), dim=-1)
    torch.testing.assert_close(result.weights, expected_weights)
    assert result.weights[0, 0, 0, 0] > result.weights[0, 0, 0, 1]
    assert result.weights[0, 0, 1, 1] > result.weights[0, 0, 1, 0]
    assert result.weights[0, 0, 2, 1] > result.weights[0, 0, 2, 0]
    assert result.weights[0, 0, 3, 0] > result.weights[0, 0, 3, 1]


def test_storage_accounting_uses_actual_latents_and_deployed_basis_prefixes() -> None:
    keys, queries, values = _random_calibration(
        layers=2,
        tokens=5,
        query_positions=3,
        kv_heads=2,
        channels=4,
        seed=50,
        dtype=torch.bfloat16,
    )
    codec = AtxyLocalLatentCodec.fit(
        keys,
        queries,
        values,
        max_rank=4,
        latent_dtype=torch.float16,
    )
    key_latents = codec.encode_keys(keys, rank=2)
    value_latents = codec.encode_values(values, rank=3)

    storage = codec.storage_bytes(key_latents, value_latents)
    projected = codec.projected_storage_bytes(token_count=5, key_rank=2, value_rank=3)

    assert storage == AtxyLocalStorageBytes(
        token_count=5,
        native_key_bytes=160,
        native_value_bytes=160,
        native_bytes=320,
        key_latent_bytes=80,
        value_latent_bytes=120,
        latent_bytes=200,
        key_basis_bytes=128,
        value_basis_bytes=192,
        value_center_bytes=64,
        basis_bytes=384,
        total_bytes=584,
    )
    assert projected == storage
    assert codec.resident_basis_bytes == 576
    assert storage.compression_ratio == pytest.approx(320 / 584)
    assert storage.savings_bytes == -264


def test_validation_rejects_invalid_shapes_ranks_dtypes_and_latent_pairs() -> None:
    keys, queries, values = _random_calibration(
        layers=1,
        tokens=4,
        query_positions=2,
        kv_heads=2,
        channels=4,
        seed=60,
    )
    with pytest.raises(ValueError, match="positive integer"):
        AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=0)
    with pytest.raises(ValueError, match="channel dimension"):
        AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=5)
    with pytest.raises(ValueError, match="same token count"):
        AtxyLocalLatentCodec.fit(keys, queries, values[:, :-1], max_rank=2)
    with pytest.raises(ValueError, match="must have shape"):
        AtxyLocalLatentCodec.fit(keys, queries[:, :, :-1], values, max_rank=2)
    with pytest.raises(ValueError, match="dtype"):
        AtxyLocalLatentCodec.fit(keys, queries.to(torch.bfloat16), values, max_rank=2)
    bad_keys = keys.clone()
    bad_keys[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        AtxyLocalLatentCodec.fit(bad_keys, queries, values, max_rank=2)

    codec = AtxyLocalLatentCodec.fit(keys, queries, values, max_rank=3)
    with pytest.raises(ValueError, match="max_rank"):
        codec.encode_keys(keys, rank=4)
    with pytest.raises(ValueError, match="non-time axes"):
        codec.encode_values(values[:, :, :1], rank=2)
    with pytest.raises(ValueError, match="dtype"):
        codec.encode_queries(queries.to(torch.bfloat16), rank=2)
    with pytest.raises(ValueError, match="must have shape"):
        codec.decode_keys(torch.ones(1, 4, 1, 2))

    query_latents = codec.encode_queries(queries, rank=2)
    key_latents = codec.encode_keys(keys, rank=3)
    value_latents = codec.encode_values(values, rank=2)
    with pytest.raises(ValueError, match="same rank"):
        codec.latent_attention(query_latents, key_latents, value_latents)
    with pytest.raises(ValueError, match="same token count"):
        codec.storage_bytes(key_latents, value_latents[:, :-1])
    with pytest.raises(ValueError, match="positive integer"):
        codec.projected_storage_bytes(token_count=0, key_rank=2, value_rank=2)
    with pytest.raises(ValueError, match="max_rank"):
        codec.projected_storage_bytes(token_count=8, key_rank=4, value_rank=2)


def test_storage_dataclass_rejects_inconsistent_component_totals() -> None:
    with pytest.raises(ValueError, match="native_bytes"):
        AtxyLocalStorageBytes(
            token_count=1,
            native_key_bytes=8,
            native_value_bytes=8,
            native_bytes=15,
            key_latent_bytes=4,
            value_latent_bytes=4,
            latent_bytes=8,
            key_basis_bytes=4,
            value_basis_bytes=4,
            value_center_bytes=4,
            basis_bytes=12,
            total_bytes=20,
        )
