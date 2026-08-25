from dataclasses import FrozenInstanceError

import pytest
import torch

from molor.experimental.atxy_vq import (
    ATXY_VQ_CODEBOOK_COUNT,
    ATXY_VQ_CODEBOOK_SIZE,
    ATXY_VQ_PACKED_CARDINALITY,
    AtxyResidualVQ,
    AtxyVQStorageBytes,
    pack_atxy_indices,
    unpack_atxy_indices,
)


def _training_fixture(*, samples: int = 144, channels: int = 7) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(71)
    cluster_centers = torch.randn(12, channels, generator=generator) * 3.0
    assignments = torch.arange(samples) % cluster_centers.shape[0]
    noise = torch.randn(samples, channels, generator=generator) * 0.2
    return cluster_centers.index_select(0, assignments) + noise


def test_pack_uses_canonical_base36_equation_and_round_trips() -> None:
    indices = torch.tensor(
        [[0, 0, 0, 0], [1, 2, 3, 4], [35, 35, 35, 35]],
        dtype=torch.uint8,
    )

    packed = pack_atxy_indices(indices)

    assert packed.dtype == torch.int32
    assert packed.tolist() == [
        0,
        (((1 * 36 + 2) * 36 + 3) * 36 + 4),
        ATXY_VQ_PACKED_CARDINALITY - 1,
    ]
    assert torch.equal(unpack_atxy_indices(packed), indices)


def test_fit_is_deterministic_cpu_float32_and_does_not_change_global_rng() -> None:
    training = _training_fixture().to(torch.float64)
    torch.manual_seed(1234)
    expected_random = torch.rand(5)
    torch.manual_seed(1234)

    first = AtxyResidualVQ.fit(training, seed=19, max_iterations=12)
    actual_random = torch.rand(5)
    second = AtxyResidualVQ.fit(training, seed=19, max_iterations=12)

    assert torch.equal(actual_random, expected_random)
    assert first.codebooks.shape == (ATXY_VQ_CODEBOOK_COUNT, ATXY_VQ_CODEBOOK_SIZE, 7)
    assert first.codebooks.dtype == torch.float32
    assert first.codebooks.device.type == "cpu"
    assert torch.equal(first.codebooks, second.codebooks)
    assert torch.equal(first.encode(training), second.encode(training))
    assert not bool(first.codebooks[:, 0, :].any())

    exposed = first.codebooks
    exposed.fill_(torch.nan)
    assert bool(torch.isfinite(first.codebooks).all())
    with pytest.raises(FrozenInstanceError):
        first.seed = 4  # type: ignore[misc]


def test_training_reconstruction_error_is_nonincreasing_at_every_depth() -> None:
    training = _training_fixture()
    original = training.clone()
    codec = AtxyResidualVQ.fit(training, seed=5, max_iterations=15)
    indices = codec.encode(training)

    errors = [
        float((training - codec.decode(indices, depth=depth)).square().sum().item())
        for depth in range(1, ATXY_VQ_CODEBOOK_COUNT + 1)
    ]

    assert torch.equal(training, original)
    assert all(later <= earlier + 1e-5 for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < errors[0]


def test_bfloat16_codebooks_decode_in_float32_and_storage_is_exact() -> None:
    training = _training_fixture(samples=50, channels=6).to(torch.bfloat16)
    codec = AtxyResidualVQ.fit(
        training,
        seed=8,
        max_iterations=8,
        codebook_dtype=torch.bfloat16,
    )
    indices = codec.encode(training)
    packed = pack_atxy_indices(indices)
    reconstruction = codec.decode(indices)
    storage = codec.storage_bytes(indices)

    expected_codebook_bytes = 4 * 36 * 6 * 2
    expected_unpacked_bytes = 50 * 4
    expected_packed_bytes = 50 * 4
    assert codec.codebook_dtype == torch.bfloat16
    assert codec.codebook_bytes == expected_codebook_bytes
    assert indices.shape == (50, 4)
    assert indices.dtype == torch.uint8
    assert packed.shape == (50,)
    assert packed.dtype == torch.int32
    assert reconstruction.shape == training.shape
    assert reconstruction.dtype == torch.float32
    assert storage == AtxyVQStorageBytes(
        sample_count=50,
        codebook_bytes=expected_codebook_bytes,
        unpacked_code_bytes=expected_unpacked_bytes,
        packed_code_bytes=expected_packed_bytes,
        total_unpacked_bytes=expected_codebook_bytes + expected_unpacked_bytes,
        total_packed_bytes=expected_codebook_bytes + expected_packed_bytes,
    )


def test_single_sample_and_zero_signal_are_supported() -> None:
    training = torch.zeros(1, 3)
    codec = AtxyResidualVQ.fit(training, seed=0, max_iterations=2)
    indices = codec.encode(training)

    assert torch.equal(indices, torch.zeros((1, 4), dtype=torch.uint8))
    assert torch.equal(codec.decode(indices), training)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: AtxyResidualVQ.fit([[1.0, 2.0]]), TypeError, "torch.Tensor"),
        (lambda: AtxyResidualVQ.fit(torch.ones(3)), ValueError, "shape"),
        (lambda: AtxyResidualVQ.fit(torch.ones(0, 3)), ValueError, "non-empty"),
        (lambda: AtxyResidualVQ.fit(torch.ones(2, 3, dtype=torch.int32)), ValueError, "floating"),
        (
            lambda: AtxyResidualVQ.fit(torch.tensor([[float("nan")]])),
            ValueError,
            "finite float32",
        ),
        (lambda: AtxyResidualVQ.fit(torch.ones(2, 3), seed=-1), ValueError, "seed"),
        (lambda: AtxyResidualVQ.fit(torch.ones(2, 3), seed=False), ValueError, "seed"),
        (
            lambda: AtxyResidualVQ.fit(torch.ones(2, 3), max_iterations=0),
            ValueError,
            "positive",
        ),
        (
            lambda: AtxyResidualVQ.fit(torch.ones(2, 3), max_iterations=10_001),
            ValueError,
            "at most",
        ),
        (
            lambda: AtxyResidualVQ.fit(torch.ones(2, 3), codebook_dtype=torch.float16),
            ValueError,
            "codebook_dtype",
        ),
    ],
)
def test_fit_validation_is_helpful(call, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        call()


def test_code_pack_decode_and_channel_validation() -> None:
    codec = AtxyResidualVQ.fit(_training_fixture(samples=40, channels=4), max_iterations=3)
    valid = torch.zeros((2, 4), dtype=torch.uint8)

    with pytest.raises(ValueError, match="4 channels"):
        codec.encode(torch.ones(2, 5))
    with pytest.raises(ValueError, match="depth"):
        codec.decode(valid, depth=0)
    with pytest.raises(ValueError, match="depth"):
        codec.decode(valid, depth=5)
    with pytest.raises(ValueError, match="depth"):
        codec.decode(valid, depth=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        codec.decode(torch.zeros(2, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="uint8"):
        codec.decode(torch.zeros(2, 4, dtype=torch.int32))

    out_of_range = valid.clone()
    out_of_range[0, 2] = 36
    with pytest.raises(ValueError, match="below 36"):
        pack_atxy_indices(out_of_range)
    with pytest.raises(ValueError, match="int32"):
        unpack_atxy_indices(torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="from 0"):
        unpack_atxy_indices(torch.tensor([-1], dtype=torch.int32))
    with pytest.raises(ValueError, match="from 0"):
        unpack_atxy_indices(torch.tensor([ATXY_VQ_PACKED_CARDINALITY], dtype=torch.int32))


def test_storage_dataclass_rejects_inconsistent_totals() -> None:
    with pytest.raises(ValueError, match="total_unpacked_bytes"):
        AtxyVQStorageBytes(
            sample_count=1,
            codebook_bytes=16,
            unpacked_code_bytes=4,
            packed_code_bytes=4,
            total_unpacked_bytes=19,
            total_packed_bytes=20,
        )
