"""Successive literal ATXY tuples for residual vector quantization.

One tuple is often too small for a 128-dimensional K or V vector.  This module
stacks independently fitted four-field ATXY residual quantizers while keeping
their byte cost explicit.  It remains a payload codec, not a page router.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from molor.experimental.atxy_vq import (
    ATXY_VQ_CODEBOOK_COUNT,
    ATXY_VQ_CODEBOOK_SIZE,
    AtxyResidualVQ,
    pack_atxy_indices,
)


_MAX_TUPLES = 64
_MAX_SEED = 2**63 - 1


def _positive_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _non_negative_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _samples(value: object, name: str, *, channels: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty shape [samples, channels]")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels")
    candidate = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError(f"{name} must contain only finite float32 values")
    return candidate


@dataclass(frozen=True, slots=True)
class AtxyVQStackStorageBytes:
    """Logical storage for all codebooks and alternate code layouts."""

    sample_count: int
    tuple_count: int
    codebook_bytes: int
    unpacked_code_bytes: int
    packed_code_bytes: int
    total_unpacked_bytes: int
    total_packed_bytes: int

    def __post_init__(self) -> None:
        _positive_int(self.sample_count, "sample_count")
        _positive_int(self.tuple_count, "tuple_count", maximum=_MAX_TUPLES)
        for name in ("codebook_bytes", "unpacked_code_bytes", "packed_code_bytes"):
            _positive_int(getattr(self, name), name)
        if self.total_unpacked_bytes != self.codebook_bytes + self.unpacked_code_bytes:
            raise ValueError("total_unpacked_bytes must include codebooks and unpacked codes")
        if self.total_packed_bytes != self.codebook_bytes + self.packed_code_bytes:
            raise ValueError("total_packed_bytes must include codebooks and packed codes")


@dataclass(frozen=True, slots=True)
class AtxyResidualVQStack:
    """A fixed sequence of residual ATXY tuples."""

    _codecs: tuple[AtxyResidualVQ, ...]
    seed: int

    def __post_init__(self) -> None:
        selected_seed = _non_negative_int(self.seed, "seed", maximum=_MAX_SEED)
        if not isinstance(self._codecs, tuple) or not self._codecs:
            raise ValueError("codecs must be a non-empty tuple")
        _positive_int(len(self._codecs), "tuple count", maximum=_MAX_TUPLES)
        if any(not isinstance(codec, AtxyResidualVQ) for codec in self._codecs):
            raise TypeError("codecs must contain AtxyResidualVQ values")
        channels = self._codecs[0].channels
        if any(codec.channels != channels for codec in self._codecs):
            raise ValueError("all codecs must have the same channel count")
        object.__setattr__(self, "seed", selected_seed)

    @classmethod
    def fit(
        cls,
        samples: Tensor,
        *,
        tuple_count: int,
        seed: int = 0,
        max_iterations: int = 25,
        codebook_dtype: torch.dtype = torch.float32,
    ) -> AtxyResidualVQStack:
        """Fit each tuple to the residual left by all preceding tuples."""

        residual = _samples(samples, "samples")
        selected_count = _positive_int(tuple_count, "tuple_count", maximum=_MAX_TUPLES)
        selected_seed = _non_negative_int(seed, "seed", maximum=_MAX_SEED)
        if selected_seed + selected_count - 1 > _MAX_SEED:
            raise ValueError("seed plus tuple_count must fit in signed int64")
        codecs: list[AtxyResidualVQ] = []
        for tuple_index in range(selected_count):
            codec = AtxyResidualVQ.fit(
                residual,
                seed=selected_seed + tuple_index,
                max_iterations=max_iterations,
                codebook_dtype=codebook_dtype,
            )
            fields = codec.encode(residual)
            residual.sub_(codec.decode(fields))
            codecs.append(codec)
        return cls(tuple(codecs), selected_seed)

    @property
    def tuple_count(self) -> int:
        return len(self._codecs)

    @property
    def channels(self) -> int:
        return self._codecs[0].channels

    @property
    def codebook_dtype(self) -> torch.dtype:
        return self._codecs[0].codebook_dtype

    @property
    def codebooks(self) -> Tensor:
        """Return a defensive ``[tuples,4,36,channels]`` copy."""

        return torch.stack(tuple(codec.codebooks for codec in self._codecs), dim=0)

    @property
    def codebook_bytes(self) -> int:
        return sum(codec.codebook_bytes for codec in self._codecs)

    def encode(self, samples: Tensor) -> Tensor:
        """Encode to uint8 ``[samples, tuples, A/T/X/Y]`` fields."""

        residual = _samples(samples, "samples", channels=self.channels)
        encoded: list[Tensor] = []
        for codec in self._codecs:
            fields = codec.encode(residual)
            residual.sub_(codec.decode(fields))
            encoded.append(fields)
        return torch.stack(encoded, dim=1).contiguous()

    def _indices(self, indices: object) -> Tensor:
        if not isinstance(indices, Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if indices.ndim != 3 or indices.shape[1:] != (
            self.tuple_count,
            ATXY_VQ_CODEBOOK_COUNT,
        ):
            raise ValueError(f"indices must have shape [samples, {self.tuple_count}, 4]")
        if indices.shape[0] < 1:
            raise ValueError("indices sample axis must be non-empty")
        if indices.dtype != torch.uint8:
            raise ValueError("indices must use torch.uint8")
        candidate = indices.detach().to(device="cpu").contiguous()
        if bool((candidate >= ATXY_VQ_CODEBOOK_SIZE).any()):
            raise ValueError(f"indices entries must be below {ATXY_VQ_CODEBOOK_SIZE}")
        return candidate

    def decode(self, indices: Tensor, *, tuple_depth: int | None = None) -> Tensor:
        """Decode the requested number of complete ATXY tuples."""

        fields = self._indices(indices)
        depth = (
            self.tuple_count if tuple_depth is None else _positive_int(tuple_depth, "tuple_depth")
        )
        if depth > self.tuple_count:
            raise ValueError(f"tuple_depth must not exceed {self.tuple_count}")
        reconstructed = torch.zeros(
            (fields.shape[0], self.channels), dtype=torch.float32, device="cpu"
        )
        for tuple_index in range(depth):
            reconstructed.add_(self._codecs[tuple_index].decode(fields[:, tuple_index]))
        return reconstructed.contiguous()

    def storage_bytes(self, indices: Tensor) -> AtxyVQStackStorageBytes:
        """Account for uint8 fields and canonical int32 packing per tuple."""

        fields = self._indices(indices)
        packed = torch.stack(
            tuple(pack_atxy_indices(fields[:, index]) for index in range(self.tuple_count)),
            dim=1,
        )
        unpacked_code_bytes = fields.numel() * fields.element_size()
        packed_code_bytes = packed.numel() * packed.element_size()
        return AtxyVQStackStorageBytes(
            sample_count=int(fields.shape[0]),
            tuple_count=self.tuple_count,
            codebook_bytes=self.codebook_bytes,
            unpacked_code_bytes=unpacked_code_bytes,
            packed_code_bytes=packed_code_bytes,
            total_unpacked_bytes=self.codebook_bytes + unpacked_code_bytes,
            total_packed_bytes=self.codebook_bytes + packed_code_bytes,
        )


__all__ = ["AtxyResidualVQStack", "AtxyVQStackStorageBytes"]
