"""Continuous ATXY-shaped tensor compression primitives for KV shadow tests.

This module treats one K or V payload as a tensor shaped
``[layers, tokens, kv_heads, head_channels]``.  It deliberately does not route,
page, deduplicate, or mutate a live cache.  Fit one codec for K and another for
V when their calibration distributions or ranks differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


_PAYLOAD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _payload_dtype(value: object, name: str) -> torch.dtype:
    if value not in _PAYLOAD_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _PAYLOAD_DTYPES)
        raise ValueError(f"{name} must be one of {supported}")
    return value


def _logical_bytes(value: Tensor) -> int:
    return value.numel() * value.element_size()


@dataclass(frozen=True, slots=True)
class AtxyShape:
    """Stable non-time axes for an ``[A, T, X, Y]`` payload tensor."""

    layers: int
    heads: int
    channels: int

    def __post_init__(self) -> None:
        _positive_int(self.layers, "layers")
        _positive_int(self.heads, "heads")
        _positive_int(self.channels, "channels")


@dataclass(frozen=True, slots=True)
class AtxyRanks:
    """Tucker ranks for the layer, KV-head, and head-channel axes."""

    layers: int
    heads: int
    channels: int

    def __post_init__(self) -> None:
        _positive_int(self.layers, "layer rank")
        _positive_int(self.heads, "head rank")
        _positive_int(self.channels, "channel rank")


@dataclass(frozen=True, slots=True)
class AtxyStorageBytes:
    """Exact logical byte accounting for one encoded K or V payload."""

    token_count: int
    native_bytes: int
    core_bytes: int
    basis_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        _positive_int(self.token_count, "token_count")
        for name in ("native_bytes", "core_bytes", "total_bytes"):
            _positive_int(getattr(self, name), name)
        if not isinstance(self.basis_bytes, int) or isinstance(self.basis_bytes, bool):
            raise ValueError("basis_bytes must be a non-negative integer")
        if self.basis_bytes < 0:
            raise ValueError("basis_bytes must be a non-negative integer")
        if self.total_bytes != self.core_bytes + self.basis_bytes:
            raise ValueError("total_bytes must equal core_bytes + basis_bytes")

    @property
    def compression_ratio(self) -> float:
        """Native logical bytes divided by encoded core-plus-basis bytes."""

        return self.native_bytes / self.total_bytes

    @property
    def savings_bytes(self) -> int:
        return self.native_bytes - self.total_bytes


def _validate_payload(
    value: object,
    name: str,
    *,
    shape: AtxyShape | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [layers, tokens, heads, channels]")
    if any(dimension < 1 for dimension in value.shape):
        raise ValueError(f"{name} axes must all be non-empty")
    if value.dtype not in _PAYLOAD_DTYPES:
        raise ValueError(f"{name} must use float16, bfloat16, or float32")
    if shape is not None and (
        value.shape[0] != shape.layers
        or value.shape[2] != shape.heads
        or value.shape[3] != shape.channels
    ):
        raise ValueError(
            f"{name} non-time axes must be [{shape.layers}, *, {shape.heads}, {shape.channels}]"
        )
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}, got {value.dtype}")
    cpu = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise ValueError(f"{name} must contain only finite values")
    return cpu


def _calibration_values(calibration: Tensor | Iterable[Tensor]) -> tuple[Tensor, ...]:
    if isinstance(calibration, Tensor):
        values = (calibration,)
    else:
        try:
            values = tuple(calibration)
        except TypeError as error:
            raise TypeError("calibration must be a tensor or an iterable of tensors") from error
    if not values:
        raise ValueError("calibration must contain at least one tensor")

    first = _validate_payload(values[0], "calibration[0]")
    shape = AtxyShape(first.shape[0], first.shape[2], first.shape[3])
    validated = [first]
    for index, value in enumerate(values[1:], start=1):
        validated.append(
            _validate_payload(
                value,
                f"calibration[{index}]",
                shape=shape,
                dtype=first.dtype,
            )
        )
    return tuple(validated)


def _canonical_basis(covariance: Tensor, rank: int) -> Tensor:
    covariance = (covariance + covariance.mT) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True, stable=True)
    basis = eigenvectors.index_select(1, order[:rank]).to(torch.float32).contiguous()

    # Eigenvector signs are arbitrary.  Canonical signs make repeated CPU fits
    # produce byte-identical bases when the eigenspaces themselves are stable.
    for column_index in range(rank):
        column = basis[:, column_index]
        pivot = int(column.abs().argmax().item())
        if float(column[pivot].item()) < 0:
            basis[:, column_index].neg_()
    return basis


def _fit_bases(
    calibrations: tuple[Tensor, ...], shape: AtxyShape, ranks: AtxyRanks
) -> tuple[Tensor, Tensor, Tensor]:
    layer_covariance = torch.zeros((shape.layers, shape.layers), dtype=torch.float64)
    head_covariance = torch.zeros((shape.heads, shape.heads), dtype=torch.float64)
    channel_covariance = torch.zeros((shape.channels, shape.channels), dtype=torch.float64)
    energy = 0.0
    sample_tokens = 0

    for calibration in calibrations:
        value = calibration.to(dtype=torch.float32)
        layer_covariance.add_(torch.einsum("atxy,btxy->ab", value, value).to(torch.float64))
        head_covariance.add_(torch.einsum("atxy,atzy->xz", value, value).to(torch.float64))
        channel_covariance.add_(torch.einsum("atxy,atxz->yz", value, value).to(torch.float64))
        energy += float(value.square().sum(dtype=torch.float64).item())
        sample_tokens += int(value.shape[1])

    if energy == 0.0:
        raise ValueError("calibration tensors must contain non-zero signal energy")
    # Scaling does not change eigenvectors and keeps long-context covariance
    # magnitudes bounded before the symmetric eigensolve.
    layer_covariance.div_(sample_tokens)
    head_covariance.div_(sample_tokens)
    channel_covariance.div_(sample_tokens)
    return (
        _canonical_basis(layer_covariance, ranks.layers),
        _canonical_basis(head_covariance, ranks.heads),
        _canonical_basis(channel_covariance, ranks.channels),
    )


@dataclass(frozen=True, slots=True)
class AtxyTuckerCodec:
    """Fixed appendable Tucker codec for one K or V tensor family.

    Time is retained in the encoded core.  Only layer, KV-head, and channel axes
    are projected.  Learned codecs calculate in float32 on CPU and store cores
    in ``core_dtype``.  Identity codecs use a direct permutation and therefore
    provide a byte-exact full-rank layout control.
    """

    shape: AtxyShape
    ranks: AtxyRanks
    core_dtype: torch.dtype
    reconstruction_dtype: torch.dtype
    _layer_basis: Tensor | None
    _head_basis: Tensor | None
    _channel_basis: Tensor | None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, AtxyShape):
            raise TypeError("shape must be an AtxyShape")
        if not isinstance(self.ranks, AtxyRanks):
            raise TypeError("ranks must be AtxyRanks")
        _payload_dtype(self.core_dtype, "core_dtype")
        _payload_dtype(self.reconstruction_dtype, "reconstruction_dtype")
        for rank, dimension, name in (
            (self.ranks.layers, self.shape.layers, "layer rank"),
            (self.ranks.heads, self.shape.heads, "head rank"),
            (self.ranks.channels, self.shape.channels, "channel rank"),
        ):
            if rank > dimension:
                raise ValueError(f"{name} cannot exceed its source dimension ({dimension})")

        bases = (self._layer_basis, self._head_basis, self._channel_basis)
        if all(basis is None for basis in bases):
            expected = AtxyRanks(self.shape.layers, self.shape.heads, self.shape.channels)
            if self.ranks != expected:
                raise ValueError("identity codec ranks must equal every source dimension")
            if self.core_dtype != self.reconstruction_dtype:
                raise ValueError("identity codec core and reconstruction dtypes must match")
            return
        if any(basis is None for basis in bases):
            raise ValueError("learned codecs require all three bases")

        expected_shapes = (
            (self.shape.layers, self.ranks.layers),
            (self.shape.heads, self.ranks.heads),
            (self.shape.channels, self.ranks.channels),
        )
        names = ("layer", "head", "channel")
        for field_name, name, basis, expected_shape in zip(
            ("_layer_basis", "_head_basis", "_channel_basis"),
            names,
            bases,
            expected_shapes,
            strict=True,
        ):
            assert basis is not None
            if not isinstance(basis, Tensor):
                raise TypeError(f"{name} basis must be a torch.Tensor")
            candidate = basis.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            if candidate.shape != expected_shape:
                raise ValueError(f"{name} basis must have shape {expected_shape}")
            if not bool(torch.isfinite(candidate).all()):
                raise ValueError(f"{name} basis must contain only finite values")
            identity = torch.eye(expected_shape[1], dtype=torch.float32)
            if not torch.allclose(candidate.mT @ candidate, identity, rtol=1e-5, atol=1e-6):
                raise ValueError(f"{name} basis columns must be orthonormal")
            object.__setattr__(self, field_name, candidate)

    @classmethod
    def fit(
        cls,
        calibration: Tensor | Iterable[Tensor],
        ranks: AtxyRanks,
        *,
        core_dtype: torch.dtype | None = None,
    ) -> AtxyTuckerCodec:
        """Fit fixed uncentered HOSVD bases from one or more calibration tensors."""

        if not isinstance(ranks, AtxyRanks):
            raise TypeError("ranks must be AtxyRanks")
        calibrations = _calibration_values(calibration)
        first = calibrations[0]
        shape = AtxyShape(first.shape[0], first.shape[2], first.shape[3])
        selected_core_dtype = first.dtype if core_dtype is None else core_dtype
        _payload_dtype(selected_core_dtype, "core_dtype")
        for rank, dimension, name in (
            (ranks.layers, shape.layers, "layer rank"),
            (ranks.heads, shape.heads, "head rank"),
            (ranks.channels, shape.channels, "channel rank"),
        ):
            if rank > dimension:
                raise ValueError(f"{name} cannot exceed its source dimension ({dimension})")
        layer_basis, head_basis, channel_basis = _fit_bases(calibrations, shape, ranks)
        return cls(
            shape=shape,
            ranks=ranks,
            core_dtype=selected_core_dtype,
            reconstruction_dtype=first.dtype,
            _layer_basis=layer_basis,
            _head_basis=head_basis,
            _channel_basis=channel_basis,
        )

    @classmethod
    def identity(
        cls,
        shape: AtxyShape,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> AtxyTuckerCodec:
        """Create an implicit full-rank codec with a byte-exact round trip."""

        if not isinstance(shape, AtxyShape):
            raise TypeError("shape must be an AtxyShape")
        selected_dtype = _payload_dtype(dtype, "dtype")
        return cls(
            shape=shape,
            ranks=AtxyRanks(shape.layers, shape.heads, shape.channels),
            core_dtype=selected_dtype,
            reconstruction_dtype=selected_dtype,
            _layer_basis=None,
            _head_basis=None,
            _channel_basis=None,
        )

    @property
    def is_identity(self) -> bool:
        return self._layer_basis is None

    @property
    def layer_basis(self) -> Tensor | None:
        return None if self._layer_basis is None else self._layer_basis.clone()

    @property
    def head_basis(self) -> Tensor | None:
        return None if self._head_basis is None else self._head_basis.clone()

    @property
    def channel_basis(self) -> Tensor | None:
        return None if self._channel_basis is None else self._channel_basis.clone()

    @property
    def basis_bytes(self) -> int:
        return sum(
            _logical_bytes(basis)
            for basis in (self._layer_basis, self._head_basis, self._channel_basis)
            if basis is not None
        )

    def core_shape(self, token_count: int) -> tuple[int, int, int, int]:
        tokens = _positive_int(token_count, "token_count")
        return (tokens, self.ranks.layers, self.ranks.heads, self.ranks.channels)

    def encode(self, value: Tensor) -> Tensor:
        """Encode ``[A, T, X, Y]`` to appendable ``[T, rA, rX, rY]`` cores."""

        source = _validate_payload(
            value,
            "value",
            shape=self.shape,
            dtype=self.reconstruction_dtype,
        )
        if self.is_identity:
            return source.permute(1, 0, 2, 3).contiguous()

        assert self._layer_basis is not None
        assert self._head_basis is not None
        assert self._channel_basis is not None
        working = source.to(torch.float32)
        by_layer = torch.einsum("atxy,ai->itxy", working, self._layer_basis)
        by_head = torch.einsum("itxy,xj->itjy", by_layer, self._head_basis)
        core = torch.einsum("itjy,yk->tijk", by_head, self._channel_basis)
        return core.to(dtype=self.core_dtype).contiguous()

    def _validate_core(self, core: object, name: str = "core") -> Tensor:
        if not isinstance(core, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if core.ndim != 4 or core.shape[1:] != (
            self.ranks.layers,
            self.ranks.heads,
            self.ranks.channels,
        ):
            raise ValueError(
                f"{name} must have shape "
                f"[tokens, {self.ranks.layers}, {self.ranks.heads}, {self.ranks.channels}]"
            )
        if core.shape[0] < 1:
            raise ValueError(f"{name} token axis must be non-empty")
        if core.dtype != self.core_dtype:
            raise ValueError(f"{name} dtype must be {self.core_dtype}, got {core.dtype}")
        candidate = core.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError(f"{name} must contain only finite values")
        return candidate

    def decode(self, core: Tensor) -> Tensor:
        """Reconstruct an ``[A, T, X, Y]`` tensor from encoded token cores."""

        encoded = self._validate_core(core)
        if self.is_identity:
            return encoded.permute(1, 0, 2, 3).contiguous()

        assert self._layer_basis is not None
        assert self._head_basis is not None
        assert self._channel_basis is not None
        working = encoded.to(torch.float32)
        by_channel = torch.einsum("tijk,yk->itjy", working, self._channel_basis)
        by_head = torch.einsum("itjy,xj->itxy", by_channel, self._head_basis)
        value = torch.einsum("itxy,ai->atxy", by_head, self._layer_basis)
        return value.to(dtype=self.reconstruction_dtype).contiguous()

    def append(self, core: Tensor | None, value: Tensor) -> Tensor:
        """Encode new causal positions and append them without refitting bases."""

        appended = self.encode(value)
        if core is None:
            return appended
        existing = self._validate_core(core, "existing core")
        return torch.cat((existing, appended), dim=0).contiguous()

    def storage_bytes(self, core: Tensor) -> AtxyStorageBytes:
        """Return exact logical bytes for one core plus this codec's fixed bases."""

        encoded = self._validate_core(core)
        token_count = int(encoded.shape[0])
        native_element_bytes = torch.empty(
            (), dtype=self.reconstruction_dtype, device="cpu"
        ).element_size()
        native_bytes = (
            self.shape.layers
            * token_count
            * self.shape.heads
            * self.shape.channels
            * native_element_bytes
        )
        core_bytes = _logical_bytes(encoded)
        basis_bytes = self.basis_bytes
        return AtxyStorageBytes(
            token_count=token_count,
            native_bytes=native_bytes,
            core_bytes=core_bytes,
            basis_bytes=basis_bytes,
            total_bytes=core_bytes + basis_bytes,
        )


__all__ = [
    "AtxyRanks",
    "AtxyShape",
    "AtxyStorageBytes",
    "AtxyTuckerCodec",
]
