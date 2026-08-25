from dataclasses import FrozenInstanceError

import pytest
import torch

from molor.experimental.atxy_kv import (
    AtxyRanks,
    AtxyShape,
    AtxyStorageBytes,
    AtxyTuckerCodec,
)


def _random_atxy(
    shape: AtxyShape,
    tokens: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        shape.layers,
        tokens,
        shape.heads,
        shape.channels,
        generator=generator,
        dtype=dtype,
    )


def test_identity_control_is_byte_exact_immutable_and_accounts_storage() -> None:
    shape = AtxyShape(layers=3, heads=2, channels=4)
    codec = AtxyTuckerCodec.identity(shape, dtype=torch.bfloat16)
    value = _random_atxy(shape, 5, seed=11, dtype=torch.bfloat16)

    core = codec.encode(value)
    reconstructed = codec.decode(core)
    storage = codec.storage_bytes(core)

    assert codec.is_identity
    assert codec.core_shape(5) == (5, 3, 2, 4)
    assert core.shape == (5, 3, 2, 4)
    assert core.dtype == torch.bfloat16
    assert torch.equal(reconstructed, value)
    assert storage == AtxyStorageBytes(
        token_count=5,
        native_bytes=value.numel() * value.element_size(),
        core_bytes=core.numel() * core.element_size(),
        basis_bytes=0,
        total_bytes=core.numel() * core.element_size(),
    )
    assert storage.compression_ratio == 1.0
    assert storage.savings_bytes == 0
    with pytest.raises(FrozenInstanceError):
        codec.shape = AtxyShape(1, 1, 1)  # type: ignore[misc]


def test_fit_is_deterministic_and_returns_orthonormal_defensive_basis_copies() -> None:
    shape = AtxyShape(4, 3, 5)
    calibrations = (
        _random_atxy(shape, 7, seed=1),
        _random_atxy(shape, 5, seed=2),
    )
    ranks = AtxyRanks(3, 2, 4)

    first = AtxyTuckerCodec.fit(calibrations, ranks)
    second = AtxyTuckerCodec.fit(calibrations, ranks)
    value = _random_atxy(shape, 3, seed=3)

    assert not first.is_identity
    for first_basis, second_basis, rank in (
        (first.layer_basis, second.layer_basis, ranks.layers),
        (first.head_basis, second.head_basis, ranks.heads),
        (first.channel_basis, second.channel_basis, ranks.channels),
    ):
        assert first_basis is not None
        assert second_basis is not None
        assert torch.equal(first_basis, second_basis)
        torch.testing.assert_close(
            first_basis.mT @ first_basis,
            torch.eye(rank),
            rtol=1e-5,
            atol=1e-6,
        )

    exposed = first.layer_basis
    assert exposed is not None
    exposed.zero_()
    assert bool(first.layer_basis.any())  # type: ignore[union-attr]
    assert torch.equal(first.encode(value), second.encode(value))


def test_known_multilinear_rank_reconstructs_with_small_error() -> None:
    generator = torch.Generator().manual_seed(9)
    ranks = AtxyRanks(2, 1, 2)
    layer_basis, _ = torch.linalg.qr(torch.randn(3, 2, generator=generator))
    head_basis, _ = torch.linalg.qr(torch.randn(2, 1, generator=generator))
    channel_basis, _ = torch.linalg.qr(torch.randn(4, 2, generator=generator))
    core = torch.randn(8, 2, 1, 2, generator=generator)
    calibration = torch.einsum(
        "tijk,ai,xj,yk->atxy",
        core,
        layer_basis,
        head_basis,
        channel_basis,
    )

    codec = AtxyTuckerCodec.fit(calibration, ranks, core_dtype=torch.float32)
    reconstructed = codec.decode(codec.encode(calibration))

    torch.testing.assert_close(reconstructed, calibration, rtol=2e-5, atol=2e-5)


def test_append_matches_whole_encoding_and_preserves_the_token_axis() -> None:
    shape = AtxyShape(3, 2, 4)
    calibration = _random_atxy(shape, 9, seed=20)
    codec = AtxyTuckerCodec.fit(calibration, AtxyRanks(2, 2, 3))
    first = _random_atxy(shape, 2, seed=21)
    second = _random_atxy(shape, 3, seed=22)
    whole = torch.cat((first, second), dim=1)

    appended = codec.append(None, first)
    appended = codec.append(appended, second)

    assert torch.equal(appended, codec.encode(whole))
    assert appended.shape[0] == first.shape[1] + second.shape[1]
    assert codec.decode(appended).shape == whole.shape


def test_key_and_value_payloads_use_independent_generic_codecs() -> None:
    shape = AtxyShape(3, 2, 6)
    keys = _random_atxy(shape, 10, seed=30)
    values = _random_atxy(shape, 10, seed=31) * 5.0 + 2.0
    ranks = AtxyRanks(2, 2, 4)

    key_codec = AtxyTuckerCodec.fit(keys, ranks, core_dtype=torch.bfloat16)
    value_codec = AtxyTuckerCodec.fit(values, ranks, core_dtype=torch.bfloat16)
    key_core = key_codec.encode(keys)
    value_core = value_codec.encode(values)

    assert key_core.shape == value_core.shape == (10, 2, 2, 4)
    assert key_core.dtype == value_core.dtype == torch.bfloat16
    assert not torch.equal(key_codec.layer_basis, value_codec.layer_basis)
    assert key_codec.decode(key_core).shape == keys.shape
    assert value_codec.decode(value_core).shape == values.shape


def test_storage_accounting_includes_actual_core_and_fp32_bases() -> None:
    shape = AtxyShape(4, 3, 6)
    ranks = AtxyRanks(2, 2, 3)
    calibration = _random_atxy(shape, 7, seed=40, dtype=torch.bfloat16)
    codec = AtxyTuckerCodec.fit(calibration, ranks, core_dtype=torch.bfloat16)
    core = codec.encode(calibration)

    storage = codec.storage_bytes(core)

    expected_native = 4 * 7 * 3 * 6 * 2
    expected_core = 7 * 2 * 2 * 3 * 2
    expected_bases = (4 * 2 + 3 * 2 + 6 * 3) * 4
    assert storage.native_bytes == expected_native
    assert storage.core_bytes == expected_core
    assert storage.basis_bytes == expected_bases
    assert storage.total_bytes == expected_core + expected_bases
    assert storage.compression_ratio == pytest.approx(
        expected_native / (expected_core + expected_bases)
    )


def test_validation_rejects_invalid_shapes_ranks_dtypes_and_cores() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AtxyShape(0, 2, 4)
    with pytest.raises(ValueError, match="positive integer"):
        AtxyRanks(1, False, 1)  # type: ignore[arg-type]

    shape = AtxyShape(2, 2, 4)
    calibration = _random_atxy(shape, 4, seed=50)
    with pytest.raises(ValueError, match="layer rank"):
        AtxyTuckerCodec.fit(calibration, AtxyRanks(3, 1, 1))
    with pytest.raises(ValueError, match="non-zero signal"):
        AtxyTuckerCodec.fit(torch.zeros_like(calibration), AtxyRanks(1, 1, 1))
    with pytest.raises(ValueError, match="shape"):
        AtxyTuckerCodec.fit(torch.zeros(2, 3, 4), AtxyRanks(1, 1, 1))
    with pytest.raises(ValueError, match="float16"):
        AtxyTuckerCodec.fit(torch.ones(2, 3, 2, 4, dtype=torch.int32), AtxyRanks(1, 1, 1))
    with pytest.raises(ValueError, match="non-time axes"):
        AtxyTuckerCodec.fit((calibration, torch.ones(3, 2, 2, 4)), AtxyRanks(1, 1, 1))
    with pytest.raises(ValueError, match="dtype"):
        AtxyTuckerCodec.fit((calibration, calibration.to(torch.bfloat16)), AtxyRanks(1, 1, 1))

    codec = AtxyTuckerCodec.fit(calibration, AtxyRanks(1, 1, 2))
    with pytest.raises(ValueError, match="non-time axes"):
        codec.encode(torch.ones(2, 3, 1, 4))
    with pytest.raises(ValueError, match="dtype"):
        codec.encode(calibration.to(torch.bfloat16))
    with pytest.raises(ValueError, match="core must have shape"):
        codec.decode(torch.ones(3, 1, 1, 1))
    with pytest.raises(ValueError, match="core dtype"):
        codec.decode(torch.ones(3, 1, 1, 2, dtype=torch.bfloat16))
    bad_core = codec.encode(calibration)
    bad_core[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        codec.storage_bytes(bad_core)


def test_storage_dataclass_rejects_inconsistent_totals() -> None:
    with pytest.raises(ValueError, match="must equal"):
        AtxyStorageBytes(
            token_count=1,
            native_bytes=8,
            core_bytes=4,
            basis_bytes=2,
            total_bytes=7,
        )
