"""Literal four-stage ATXY residual vector quantization primitives.

This module is an isolated payload codec for finite tensors shaped
``[samples, channels]``.  Its A, T, X, and Y fields are four ordered additive
residual-codebook indices.  They are not page addresses or routing hashes.
Fitting and distance calculations run in float32 on CPU; codebooks may be
stored in float32 or bfloat16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


ATXY_VQ_CODEBOOK_COUNT = 4
ATXY_VQ_CODEBOOK_SIZE = 36
ATXY_VQ_PACKED_CARDINALITY = ATXY_VQ_CODEBOOK_SIZE**ATXY_VQ_CODEBOOK_COUNT
ATXY_VQ_STAGE_NAMES = ("A", "T", "X", "Y")

_DEFAULT_MAX_ITERATIONS = 25
_MAX_ITERATIONS = 10_000
_MAX_SEED = 2**63 - 1
_DISTANCE_BATCH_SIZE = 4_096
_CODEBOOK_DTYPES = (torch.float32, torch.bfloat16)


def _non_negative_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _positive_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _logical_bytes(value: Tensor) -> int:
    return value.numel() * value.element_size()


def _validate_samples(
    value: object,
    name: str,
    *,
    channels: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, channels]")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} sample and channel axes must be non-empty")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a real floating-point dtype")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {value.shape[1]}")

    candidate = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError(f"{name} must contain values representable as finite float32")
    return candidate


def _validate_indices(value: object, name: str = "indices") -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != ATXY_VQ_CODEBOOK_COUNT:
        raise ValueError(f"{name} must have shape [samples, 4] in A,T,X,Y order")
    if value.shape[0] < 1:
        raise ValueError(f"{name} sample axis must be non-empty")
    if value.dtype != torch.uint8:
        raise ValueError(f"{name} must use torch.uint8")

    candidate = value.detach().to(device="cpu").contiguous()
    if bool((candidate >= ATXY_VQ_CODEBOOK_SIZE).any()):
        raise ValueError(f"{name} entries must be below {ATXY_VQ_CODEBOOK_SIZE}")
    return candidate


def _validate_packed(value: object, name: str = "packed") -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [samples]")
    if value.shape[0] < 1:
        raise ValueError(f"{name} sample axis must be non-empty")
    if value.dtype != torch.int32:
        raise ValueError(f"{name} must use torch.int32")

    candidate = value.detach().to(device="cpu").contiguous()
    if bool((candidate < 0).any()) or bool((candidate >= ATXY_VQ_PACKED_CARDINALITY).any()):
        raise ValueError(f"{name} entries must be from 0 through {ATXY_VQ_PACKED_CARDINALITY - 1}")
    return candidate


def pack_atxy_indices(indices: Tensor) -> Tensor:
    """Pack uint8 ``[A,T,X,Y]`` fields into canonical base-36 int32 values.

    The equation is ``(((A * 36 + T) * 36 + X) * 36 + Y)``.
    """

    fields = _validate_indices(indices)
    working = fields.to(torch.int32)
    packed = (
        (working[:, 0] * ATXY_VQ_CODEBOOK_SIZE + working[:, 1]) * ATXY_VQ_CODEBOOK_SIZE
        + working[:, 2]
    ) * ATXY_VQ_CODEBOOK_SIZE + working[:, 3]
    return packed.to(torch.int32).contiguous()


def unpack_atxy_indices(packed: Tensor) -> Tensor:
    """Unpack canonical base-36 int32 values to uint8 ``[A,T,X,Y]`` fields."""

    remaining = _validate_packed(packed).to(torch.int64)
    fields_reversed: list[Tensor] = []
    for _ in range(ATXY_VQ_CODEBOOK_COUNT):
        fields_reversed.append(torch.remainder(remaining, ATXY_VQ_CODEBOOK_SIZE))
        remaining = torch.div(remaining, ATXY_VQ_CODEBOOK_SIZE, rounding_mode="floor")
    return torch.stack(tuple(reversed(fields_reversed)), dim=1).to(torch.uint8).contiguous()


@dataclass(frozen=True, slots=True)
class AtxyVQStorageBytes:
    """Exact logical bytes for alternate packed and unpacked VQ payloads.

    Tensor object overhead and scalar codec configuration are intentionally not
    included.  Each total includes one copy of the codec's stored codebooks.
    """

    sample_count: int
    codebook_bytes: int
    unpacked_code_bytes: int
    packed_code_bytes: int
    total_unpacked_bytes: int
    total_packed_bytes: int

    def __post_init__(self) -> None:
        _positive_int(self.sample_count, "sample_count")
        for name in (
            "codebook_bytes",
            "unpacked_code_bytes",
            "packed_code_bytes",
            "total_unpacked_bytes",
            "total_packed_bytes",
        ):
            _positive_int(getattr(self, name), name)
        if self.total_unpacked_bytes != self.codebook_bytes + self.unpacked_code_bytes:
            raise ValueError("total_unpacked_bytes must equal codebook_bytes + unpacked_code_bytes")
        if self.total_packed_bytes != self.codebook_bytes + self.packed_code_bytes:
            raise ValueError("total_packed_bytes must equal codebook_bytes + packed_code_bytes")


def _nearest_indices(samples: Tensor, codebook: Tensor) -> Tensor:
    assignments = torch.empty(samples.shape[0], dtype=torch.int64, device="cpu")
    for start in range(0, samples.shape[0], _DISTANCE_BATCH_SIZE):
        stop = min(start + _DISTANCE_BATCH_SIZE, samples.shape[0])
        differences = samples[start:stop, None, :] - codebook[None, :, :]
        distances = differences.square().sum(dim=2)
        assignments[start:stop] = distances.argmin(dim=1)
    return assignments


def _initial_centroids(samples: Tensor, generator: torch.Generator) -> Tensor:
    centroids = torch.zeros(
        (ATXY_VQ_CODEBOOK_SIZE, samples.shape[1]),
        dtype=torch.float32,
        device="cpu",
    )
    learned_entry_count = ATXY_VQ_CODEBOOK_SIZE - 1
    permutation = torch.randperm(samples.shape[0], generator=generator, device="cpu")
    repeat_count = (learned_entry_count + samples.shape[0] - 1) // samples.shape[0]
    initial_indices = permutation.repeat(repeat_count)[:learned_entry_count]
    centroids[1:] = samples.index_select(0, initial_indices)
    return centroids


def _fit_codebook(
    residual: Tensor,
    *,
    generator: torch.Generator,
    max_iterations: int,
) -> Tensor:
    centroids = _initial_centroids(residual, generator)
    previous_assignments: Tensor | None = None

    for _ in range(max_iterations):
        assignments = _nearest_indices(residual, centroids)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assignments, residual)
        counts = torch.bincount(assignments, minlength=ATXY_VQ_CODEBOOK_SIZE)

        updated = centroids.clone()
        nonempty = counts[1:] > 0
        if bool(nonempty.any()):
            learned_indices = torch.nonzero(nonempty, as_tuple=False).flatten() + 1
            updated[learned_indices] = sums[learned_indices] / counts[learned_indices, None]
        # A fixed all-zero entry makes every progressive residual stage optional.
        updated[0].zero_()

        converged = previous_assignments is not None and torch.equal(
            assignments, previous_assignments
        )
        centroids = updated
        if converged:
            break
        previous_assignments = assignments

    return centroids.contiguous()


@dataclass(frozen=True, slots=True)
class AtxyResidualVQ:
    """A fixed A/T/X/Y additive residual vector quantizer.

    Four codebooks with 36 entries each are always used.  Entry zero in every
    stage is the zero vector, so selecting another stage can be skipped without
    increasing the residual error.  Fitting uses a local seeded CPU generator
    and never changes global random state.
    """

    _codebooks: Tensor
    seed: int
    max_iterations: int

    def __post_init__(self) -> None:
        seed = _non_negative_int(self.seed, "seed", maximum=_MAX_SEED)
        max_iterations = _positive_int(
            self.max_iterations,
            "max_iterations",
            maximum=_MAX_ITERATIONS,
        )
        if not isinstance(self._codebooks, Tensor):
            raise TypeError("codebooks must be a torch.Tensor")
        if self._codebooks.ndim != 3 or self._codebooks.shape[:2] != (
            ATXY_VQ_CODEBOOK_COUNT,
            ATXY_VQ_CODEBOOK_SIZE,
        ):
            raise ValueError("codebooks must have shape [4, 36, channels]")
        if self._codebooks.shape[2] < 1:
            raise ValueError("codebook channel axis must be non-empty")
        if self._codebooks.dtype not in _CODEBOOK_DTYPES:
            raise ValueError("codebooks must use torch.float32 or torch.bfloat16")

        candidate = self._codebooks.detach().to(device="cpu").contiguous().clone()
        if not bool(torch.isfinite(candidate.to(torch.float32)).all()):
            raise ValueError("codebooks must contain only finite values")
        if bool((candidate[:, 0, :] != 0).any()):
            raise ValueError("codebook entry zero must be the zero vector at every stage")
        object.__setattr__(self, "_codebooks", candidate)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_iterations", max_iterations)

    @classmethod
    def fit(
        cls,
        samples: Tensor,
        *,
        seed: int = 0,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        codebook_dtype: torch.dtype = torch.float32,
    ) -> AtxyResidualVQ:
        """Fit four ordered residual codebooks using bounded Lloyd iterations."""

        training = _validate_samples(samples, "samples")
        selected_seed = _non_negative_int(seed, "seed", maximum=_MAX_SEED)
        selected_iterations = _positive_int(
            max_iterations,
            "max_iterations",
            maximum=_MAX_ITERATIONS,
        )
        if codebook_dtype not in _CODEBOOK_DTYPES:
            raise ValueError("codebook_dtype must be torch.float32 or torch.bfloat16")

        generator = torch.Generator(device="cpu").manual_seed(selected_seed)
        residual = training.clone()
        stored_codebooks: list[Tensor] = []
        for _ in range(ATXY_VQ_CODEBOOK_COUNT):
            fitted = _fit_codebook(
                residual,
                generator=generator,
                max_iterations=selected_iterations,
            )
            stored = fitted.to(dtype=codebook_dtype).contiguous()
            stored_codebooks.append(stored)
            working_codebook = stored.to(torch.float32)
            assignments = _nearest_indices(residual, working_codebook)
            residual.sub_(working_codebook.index_select(0, assignments))

        return cls(
            _codebooks=torch.stack(stored_codebooks, dim=0),
            seed=selected_seed,
            max_iterations=selected_iterations,
        )

    @property
    def channels(self) -> int:
        return int(self._codebooks.shape[2])

    @property
    def codebook_dtype(self) -> torch.dtype:
        return self._codebooks.dtype

    @property
    def codebooks(self) -> Tensor:
        return self._codebooks.clone()

    @property
    def codebook_bytes(self) -> int:
        return _logical_bytes(self._codebooks)

    def encode(self, samples: Tensor) -> Tensor:
        """Encode samples to uint8 ``[A,T,X,Y]`` residual indices on CPU."""

        residual = _validate_samples(samples, "samples", channels=self.channels)
        indices = torch.empty(
            (residual.shape[0], ATXY_VQ_CODEBOOK_COUNT),
            dtype=torch.uint8,
            device="cpu",
        )
        for stage in range(ATXY_VQ_CODEBOOK_COUNT):
            codebook = self._codebooks[stage].to(torch.float32)
            selected = _nearest_indices(residual, codebook)
            indices[:, stage] = selected.to(torch.uint8)
            residual.sub_(codebook.index_select(0, selected))
        return indices.contiguous()

    def decode(self, indices: Tensor, *, depth: int = ATXY_VQ_CODEBOOK_COUNT) -> Tensor:
        """Decode the first ``depth`` additive stages to float32 on CPU."""

        selected_depth = _positive_int(depth, "depth")
        if selected_depth > ATXY_VQ_CODEBOOK_COUNT:
            raise ValueError(f"depth must be from 1 through {ATXY_VQ_CODEBOOK_COUNT}")
        fields = _validate_indices(indices)
        reconstruction = torch.zeros(
            (fields.shape[0], self.channels),
            dtype=torch.float32,
            device="cpu",
        )
        for stage in range(selected_depth):
            codebook = self._codebooks[stage].to(torch.float32)
            reconstruction.add_(codebook.index_select(0, fields[:, stage].to(torch.int64)))
        return reconstruction.contiguous()

    def storage_bytes(self, indices: Tensor) -> AtxyVQStorageBytes:
        """Account for codebooks plus packed and unpacked code alternatives."""

        fields = _validate_indices(indices)
        packed = pack_atxy_indices(fields)
        codebook_bytes = self.codebook_bytes
        unpacked_code_bytes = _logical_bytes(fields)
        packed_code_bytes = _logical_bytes(packed)
        return AtxyVQStorageBytes(
            sample_count=int(fields.shape[0]),
            codebook_bytes=codebook_bytes,
            unpacked_code_bytes=unpacked_code_bytes,
            packed_code_bytes=packed_code_bytes,
            total_unpacked_bytes=codebook_bytes + unpacked_code_bytes,
            total_packed_bytes=codebook_bytes + packed_code_bytes,
        )


__all__ = [
    "ATXY_VQ_CODEBOOK_COUNT",
    "ATXY_VQ_CODEBOOK_SIZE",
    "ATXY_VQ_PACKED_CARDINALITY",
    "ATXY_VQ_STAGE_NAMES",
    "AtxyResidualVQ",
    "AtxyVQStorageBytes",
    "pack_atxy_indices",
    "unpack_atxy_indices",
]
