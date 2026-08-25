from dataclasses import FrozenInstanceError

import pytest
import torch

from molor.experimental.atxy_vq_stack import AtxyResidualVQStack


def _samples(seed: int, count: int = 96, channels: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(12, channels, generator=generator)
    assignments = torch.arange(count) % centers.shape[0]
    return centers[assignments] + 0.05 * torch.randn(count, channels, generator=generator)


def test_stack_is_deterministic_progressive_and_does_not_mutate_inputs() -> None:
    calibration = _samples(1)
    heldout = _samples(2)
    original_calibration = calibration.clone()
    original_heldout = heldout.clone()

    first = AtxyResidualVQStack.fit(
        calibration,
        tuple_count=3,
        seed=17,
        max_iterations=3,
        codebook_dtype=torch.bfloat16,
    )
    second = AtxyResidualVQStack.fit(
        calibration,
        tuple_count=3,
        seed=17,
        max_iterations=3,
        codebook_dtype=torch.bfloat16,
    )
    indices = first.encode(heldout)

    assert torch.equal(calibration, original_calibration)
    assert torch.equal(heldout, original_heldout)
    assert torch.equal(first.codebooks, second.codebooks)
    assert torch.equal(indices, second.encode(heldout))
    assert indices.shape == (heldout.shape[0], 3, 4)
    errors = [
        float((heldout - first.decode(indices, tuple_depth=depth)).square().sum().item())
        for depth in range(1, 4)
    ]
    assert errors[1] <= errors[0]
    assert errors[2] <= errors[1]


def test_stack_storage_accounts_every_tuple_and_returns_defensive_codebooks() -> None:
    samples = _samples(4, count=48, channels=6)
    stack = AtxyResidualVQStack.fit(samples, tuple_count=2, seed=3, max_iterations=2)
    indices = stack.encode(samples)
    storage = stack.storage_bytes(indices)

    assert storage.unpacked_code_bytes == 48 * 2 * 4
    assert storage.packed_code_bytes == 48 * 2 * 4
    assert storage.codebook_bytes == 2 * 4 * 36 * 6 * 4
    exposed = stack.codebooks
    exposed.fill_(99)
    assert not bool((stack.codebooks == 99).all())
    with pytest.raises(FrozenInstanceError):
        stack.seed = 0  # type: ignore[misc]


def test_stack_validation_rejects_invalid_counts_shapes_and_codes() -> None:
    samples = _samples(5, count=40, channels=4)
    with pytest.raises(ValueError, match="tuple_count"):
        AtxyResidualVQStack.fit(samples, tuple_count=0)
    stack = AtxyResidualVQStack.fit(samples, tuple_count=2, max_iterations=1)
    with pytest.raises(ValueError, match="channels"):
        stack.encode(torch.ones(3, 5))
    with pytest.raises(ValueError, match="shape"):
        stack.decode(torch.zeros(3, 1, 4, dtype=torch.uint8))
    invalid = torch.zeros(3, 2, 4, dtype=torch.uint8)
    invalid[0, 0, 0] = 36
    with pytest.raises(ValueError, match="below 36"):
        stack.storage_bytes(invalid)
