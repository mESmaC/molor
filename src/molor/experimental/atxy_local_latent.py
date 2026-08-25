"""Per-layer, per-KV-head linear latent ceiling for ATXY KV experiments.

This module is deliberately an offline, CPU-oriented primitive.  It does not
install a cache, route pages, or mutate model state.  Native keys and values use
the ATXY layout ``[layer, token, KV head, head channel]``.  Post-RoPE queries
use ``[layer, query position, query head, head channel]`` with exactly two query
heads assigned to each KV head.

The key basis is a *linear dot-product* basis, not conventional centered PCA.
For every layer/KV-head pair it eigendecomposes an equal-weight mixture of the
per-vector uncentered K and post-RoPE-Q second moments::

    C_key = 0.5 * mean(K K.T) + 0.5 * mean(Q Q.T)

The two query heads in a GQA group are flattened into that group's Q sample
set.  Values use ordinary centered PCA and retain a local mean.  A codec is fit
once at ``max_rank``; every smaller rank is the corresponding leading prefix,
so rank sweeps are nested rather than independently refit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from molor.experimental.atxy_kv import AtxyShape


GQA_QUERY_HEADS_PER_KV_HEAD = 2
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


def _validate_native_payload(
    value: object,
    name: str,
    *,
    shape: AtxyShape | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape [layers, tokens, KV heads, channels]")
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
    candidate = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError(f"{name} must contain only finite values")
    return candidate


def _validate_queries(
    value: object,
    name: str,
    *,
    shape: AtxyShape,
    dtype: torch.dtype,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_heads = shape.heads * GQA_QUERY_HEADS_PER_KV_HEAD
    if (
        value.ndim != 4
        or value.shape[0] != shape.layers
        or value.shape[2:]
        != (
            expected_heads,
            shape.channels,
        )
    ):
        raise ValueError(
            f"{name} must have shape "
            f"[{shape.layers}, query positions, {expected_heads}, {shape.channels}]"
        )
    if value.shape[1] < 1:
        raise ValueError(f"{name} query-position axis must be non-empty")
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}, got {value.dtype}")
    candidate = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(candidate).all()):
        raise ValueError(f"{name} must contain only finite values")
    return candidate


def _canonical_basis(covariance: Tensor, rank: int) -> Tensor:
    """Return leading eigenvectors with deterministic, canonical signs."""

    symmetric = (covariance + covariance.mT) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    order = torch.argsort(eigenvalues, descending=True, stable=True)
    basis = eigenvectors.index_select(1, order[:rank]).to(torch.float32).contiguous()
    for column_index in range(rank):
        column = basis[:, column_index]
        pivot = int(column.abs().argmax().item())
        if float(column[pivot].item()) < 0:
            basis[:, column_index].neg_()
    return basis


def _fit_key_bases(keys: Tensor, queries: Tensor, shape: AtxyShape, max_rank: int) -> Tensor:
    bases = torch.empty(
        (shape.layers, shape.heads, shape.channels, max_rank),
        dtype=torch.float32,
    )
    keys64 = keys.to(torch.float64)
    queries64 = queries.to(torch.float64)
    for layer_index in range(shape.layers):
        for head_index in range(shape.heads):
            key_samples = keys64[layer_index, :, head_index, :]
            query_samples = queries64[
                layer_index,
                :,
                GQA_QUERY_HEADS_PER_KV_HEAD * head_index : GQA_QUERY_HEADS_PER_KV_HEAD
                * (head_index + 1),
                :,
            ].reshape(-1, shape.channels)
            key_moment = key_samples.mT @ key_samples / key_samples.shape[0]
            query_moment = query_samples.mT @ query_samples / query_samples.shape[0]
            bases[layer_index, head_index] = _canonical_basis(
                0.5 * key_moment + 0.5 * query_moment,
                max_rank,
            )
    return bases


def _fit_value_bases(values: Tensor, shape: AtxyShape, max_rank: int) -> tuple[Tensor, Tensor]:
    values64 = values.to(torch.float64)
    means64 = values64.mean(dim=1)
    bases = torch.empty(
        (shape.layers, shape.heads, shape.channels, max_rank),
        dtype=torch.float32,
    )
    for layer_index in range(shape.layers):
        for head_index in range(shape.heads):
            samples = values64[layer_index, :, head_index, :]
            centered = samples - means64[layer_index, head_index]
            covariance = centered.mT @ centered / samples.shape[0]
            bases[layer_index, head_index] = _canonical_basis(covariance, max_rank)
    return bases, means64.to(torch.float32).contiguous()


@dataclass(frozen=True, slots=True)
class AtxyLocalStorageBytes:
    """Exact logical bytes for a deployed K/V latent pair and its local bases.

    ``basis_bytes`` includes the selected K and V basis prefixes plus the V
    centers.  It intentionally does not charge a second query basis because Q
    uses the K basis, and queries are not persistent KV-cache payloads.
    """

    token_count: int
    native_key_bytes: int
    native_value_bytes: int
    native_bytes: int
    key_latent_bytes: int
    value_latent_bytes: int
    latent_bytes: int
    key_basis_bytes: int
    value_basis_bytes: int
    value_center_bytes: int
    basis_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        _positive_int(self.token_count, "token_count")
        fields = (
            "native_key_bytes",
            "native_value_bytes",
            "native_bytes",
            "key_latent_bytes",
            "value_latent_bytes",
            "latent_bytes",
            "key_basis_bytes",
            "value_basis_bytes",
            "value_center_bytes",
            "basis_bytes",
            "total_bytes",
        )
        for name in fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.native_bytes != self.native_key_bytes + self.native_value_bytes:
            raise ValueError("native_bytes must equal native_key_bytes + native_value_bytes")
        if self.latent_bytes != self.key_latent_bytes + self.value_latent_bytes:
            raise ValueError("latent_bytes must equal key_latent_bytes + value_latent_bytes")
        if self.basis_bytes != (
            self.key_basis_bytes + self.value_basis_bytes + self.value_center_bytes
        ):
            raise ValueError(
                "basis_bytes must equal key_basis_bytes + value_basis_bytes + value_center_bytes"
            )
        if self.total_bytes != self.latent_bytes + self.basis_bytes:
            raise ValueError("total_bytes must equal latent_bytes + basis_bytes")

    @property
    def compression_ratio(self) -> float:
        return self.native_bytes / self.total_bytes

    @property
    def savings_bytes(self) -> int:
        return self.native_bytes - self.total_bytes


@dataclass(frozen=True, slots=True)
class AtxyLatentAttention:
    """Float32 attention probabilities and reconstructed native-space outputs."""

    weights: Tensor
    output: Tensor


@dataclass(frozen=True, slots=True)
class AtxyLocalLatentCodec:
    """Fixed nested local bases for a linear dense-attention ceiling."""

    shape: AtxyShape
    max_rank: int
    latent_dtype: torch.dtype
    reconstruction_dtype: torch.dtype
    _key_basis: Tensor
    _value_basis: Tensor
    _value_center: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.shape, AtxyShape):
            raise TypeError("shape must be an AtxyShape")
        rank = _positive_int(self.max_rank, "max_rank")
        if rank > self.shape.channels:
            raise ValueError(f"max_rank cannot exceed channel dimension ({self.shape.channels})")
        _payload_dtype(self.latent_dtype, "latent_dtype")
        _payload_dtype(self.reconstruction_dtype, "reconstruction_dtype")

        expected_basis_shape = (
            self.shape.layers,
            self.shape.heads,
            self.shape.channels,
            rank,
        )
        for field_name, label in (("_key_basis", "key basis"), ("_value_basis", "value basis")):
            source = getattr(self, field_name)
            if not isinstance(source, Tensor):
                raise TypeError(f"{label} must be a torch.Tensor")
            candidate = source.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            if candidate.shape != expected_basis_shape:
                raise ValueError(f"{label} must have shape {expected_basis_shape}")
            if not bool(torch.isfinite(candidate).all()):
                raise ValueError(f"{label} must contain only finite values")
            gram = torch.einsum("axyr,axys->axrs", candidate, candidate)
            identity = torch.eye(rank, dtype=torch.float32).expand(
                self.shape.layers, self.shape.heads, rank, rank
            )
            if not torch.allclose(gram, identity, rtol=1e-5, atol=2e-6):
                raise ValueError(f"{label} columns must be orthonormal for every layer/KV head")
            object.__setattr__(self, field_name, candidate)

        if not isinstance(self._value_center, Tensor):
            raise TypeError("value center must be a torch.Tensor")
        center = (
            self._value_center.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        )
        expected_center_shape = (self.shape.layers, self.shape.heads, self.shape.channels)
        if center.shape != expected_center_shape:
            raise ValueError(f"value center must have shape {expected_center_shape}")
        if not bool(torch.isfinite(center).all()):
            raise ValueError("value center must contain only finite values")
        object.__setattr__(self, "_value_center", center)

    @classmethod
    def fit(
        cls,
        keys: Tensor,
        post_rope_queries: Tensor,
        values: Tensor,
        *,
        max_rank: int,
        latent_dtype: torch.dtype | None = None,
    ) -> AtxyLocalLatentCodec:
        """Fit local K/Q dot-product and centered-V bases from CPU calibration tensors."""

        validated_keys = _validate_native_payload(keys, "keys")
        shape = AtxyShape(
            validated_keys.shape[0],
            validated_keys.shape[2],
            validated_keys.shape[3],
        )
        validated_values = _validate_native_payload(
            values,
            "values",
            shape=shape,
            dtype=validated_keys.dtype,
        )
        if validated_values.shape[1] != validated_keys.shape[1]:
            raise ValueError("keys and values must have the same token count")
        validated_queries = _validate_queries(
            post_rope_queries,
            "post_rope_queries",
            shape=shape,
            dtype=validated_keys.dtype,
        )
        selected_rank = _positive_int(max_rank, "max_rank")
        if selected_rank > shape.channels:
            raise ValueError(f"max_rank cannot exceed channel dimension ({shape.channels})")
        selected_dtype = validated_keys.dtype if latent_dtype is None else latent_dtype
        _payload_dtype(selected_dtype, "latent_dtype")

        key_basis = _fit_key_bases(
            validated_keys,
            validated_queries,
            shape,
            selected_rank,
        )
        value_basis, value_center = _fit_value_bases(
            validated_values,
            shape,
            selected_rank,
        )
        return cls(
            shape=shape,
            max_rank=selected_rank,
            latent_dtype=selected_dtype,
            reconstruction_dtype=validated_keys.dtype,
            _key_basis=key_basis,
            _value_basis=value_basis,
            _value_center=value_center,
        )

    @property
    def query_heads(self) -> int:
        return self.shape.heads * GQA_QUERY_HEADS_PER_KV_HEAD

    @property
    def key_basis(self) -> Tensor:
        return self._key_basis.clone()

    @property
    def value_basis(self) -> Tensor:
        return self._value_basis.clone()

    @property
    def value_center(self) -> Tensor:
        return self._value_center.clone()

    @property
    def resident_basis_bytes(self) -> int:
        """Logical bytes of the full max-rank fitted state held by this object."""

        return (
            _logical_bytes(self._key_basis)
            + _logical_bytes(self._value_basis)
            + _logical_bytes(self._value_center)
        )

    def _rank(self, rank: object, name: str = "rank") -> int:
        selected = _positive_int(rank, name)
        if selected > self.max_rank:
            raise ValueError(f"{name} cannot exceed fitted max_rank ({self.max_rank})")
        return selected

    def key_basis_prefix(self, rank: int) -> Tensor:
        selected = self._rank(rank)
        return self._key_basis[..., :selected].clone()

    def value_basis_prefix(self, rank: int) -> Tensor:
        selected = self._rank(rank)
        return self._value_basis[..., :selected].clone()

    def encode_keys(self, keys: Tensor, *, rank: int) -> Tensor:
        source = _validate_native_payload(
            keys,
            "keys",
            shape=self.shape,
            dtype=self.reconstruction_dtype,
        )
        selected = self._rank(rank)
        latent = torch.einsum(
            "atxy,axyr->atxr",
            source.to(torch.float32),
            self._key_basis[..., :selected],
        )
        return latent.to(dtype=self.latent_dtype).contiguous()

    def encode_values(self, values: Tensor, *, rank: int) -> Tensor:
        source = _validate_native_payload(
            values,
            "values",
            shape=self.shape,
            dtype=self.reconstruction_dtype,
        )
        selected = self._rank(rank)
        centered = source.to(torch.float32) - self._value_center[:, None, :, :]
        latent = torch.einsum(
            "atxy,axyr->atxr",
            centered,
            self._value_basis[..., :selected],
        )
        return latent.to(dtype=self.latent_dtype).contiguous()

    def encode_queries(self, post_rope_queries: Tensor, *, rank: int) -> Tensor:
        queries = _validate_queries(
            post_rope_queries,
            "post_rope_queries",
            shape=self.shape,
            dtype=self.reconstruction_dtype,
        )
        selected = self._rank(rank)
        grouped_queries = queries.to(torch.float32).reshape(
            self.shape.layers,
            queries.shape[1],
            self.shape.heads,
            GQA_QUERY_HEADS_PER_KV_HEAD,
            self.shape.channels,
        )
        latent = torch.einsum(
            "asxgy,axyr->asxgr",
            grouped_queries,
            self._key_basis[..., :selected],
        )
        return (
            latent.reshape(
                self.shape.layers,
                queries.shape[1],
                self.query_heads,
                selected,
            )
            .to(dtype=self.latent_dtype)
            .contiguous()
        )

    def _validate_kv_latent(self, value: object, name: str) -> tuple[Tensor, int]:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            value.ndim != 4
            or value.shape[0] != self.shape.layers
            or value.shape[1] < 1
            or value.shape[2] != self.shape.heads
        ):
            raise ValueError(
                f"{name} must have shape [{self.shape.layers}, tokens, {self.shape.heads}, rank]"
            )
        rank = self._rank(value.shape[3], f"{name} rank")
        if value.dtype != self.latent_dtype:
            raise ValueError(f"{name} dtype must be {self.latent_dtype}, got {value.dtype}")
        candidate = value.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError(f"{name} must contain only finite values")
        return candidate, rank

    def _validate_query_latent(self, value: object) -> tuple[Tensor, int]:
        if not isinstance(value, Tensor):
            raise TypeError("query_latents must be a torch.Tensor")
        if (
            value.ndim != 4
            or value.shape[0] != self.shape.layers
            or value.shape[1] < 1
            or value.shape[2] != self.query_heads
        ):
            raise ValueError(
                "query_latents must have shape "
                f"[{self.shape.layers}, query positions, {self.query_heads}, rank]"
            )
        rank = self._rank(value.shape[3], "query_latents rank")
        if value.dtype != self.latent_dtype:
            raise ValueError(f"query_latents dtype must be {self.latent_dtype}, got {value.dtype}")
        candidate = value.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError("query_latents must contain only finite values")
        return candidate, rank

    def decode_keys(self, key_latents: Tensor) -> Tensor:
        encoded, rank = self._validate_kv_latent(key_latents, "key_latents")
        reconstructed = torch.einsum(
            "atxr,axyr->atxy",
            encoded.to(torch.float32),
            self._key_basis[..., :rank],
        )
        return reconstructed.to(dtype=self.reconstruction_dtype).contiguous()

    def decode_values(self, value_latents: Tensor) -> Tensor:
        encoded, rank = self._validate_kv_latent(value_latents, "value_latents")
        reconstructed = torch.einsum(
            "atxr,axyr->atxy",
            encoded.to(torch.float32),
            self._value_basis[..., :rank],
        )
        reconstructed.add_(self._value_center[:, None, :, :])
        return reconstructed.to(dtype=self.reconstruction_dtype).contiguous()

    def decode_queries(self, query_latents: Tensor) -> Tensor:
        encoded, rank = self._validate_query_latent(query_latents)
        grouped = encoded.to(torch.float32).reshape(
            self.shape.layers,
            encoded.shape[1],
            self.shape.heads,
            GQA_QUERY_HEADS_PER_KV_HEAD,
            rank,
        )
        reconstructed = torch.einsum(
            "asxgr,axyr->asxgy",
            grouped,
            self._key_basis[..., :rank],
        )
        return (
            reconstructed.reshape(
                self.shape.layers,
                encoded.shape[1],
                self.query_heads,
                self.shape.channels,
            )
            .to(dtype=self.reconstruction_dtype)
            .contiguous()
        )

    def latent_attention(
        self,
        query_latents: Tensor,
        key_latents: Tensor,
        value_latents: Tensor,
    ) -> AtxyLatentAttention:
        """Run grouped-query attention without reconstructing individual K/V vectors.

        Scores retain the native ``1 / sqrt(head_channels)`` scale.  Query
        heads ``2g`` and ``2g + 1`` both consume KV head ``g``.  Value outputs
        are accumulated in latent space and reconstructed only after attention.
        """

        queries, query_rank = self._validate_query_latent(query_latents)
        keys, key_rank = self._validate_kv_latent(key_latents, "key_latents")
        values, value_rank = self._validate_kv_latent(value_latents, "value_latents")
        if query_rank != key_rank:
            raise ValueError("query_latents and key_latents must use the same rank")
        if keys.shape[1] != values.shape[1]:
            raise ValueError("key_latents and value_latents must have the same token count")

        grouped_queries = queries.to(torch.float32).reshape(
            self.shape.layers,
            queries.shape[1],
            self.shape.heads,
            GQA_QUERY_HEADS_PER_KV_HEAD,
            query_rank,
        )
        scores = torch.einsum(
            "asxgr,atxr->asxgt",
            grouped_queries,
            keys.to(torch.float32),
        )
        scores.mul_(1.0 / math.sqrt(self.shape.channels))
        grouped_weights = torch.softmax(scores, dim=-1)

        latent_output = torch.einsum(
            "asxgt,atxr->asxgr",
            grouped_weights,
            values.to(torch.float32),
        )
        output = torch.einsum(
            "asxgr,axyr->asxgy",
            latent_output,
            self._value_basis[..., :value_rank],
        )
        output.add_(self._value_center[:, None, :, None, :])
        weights = grouped_weights.reshape(
            self.shape.layers,
            queries.shape[1],
            self.query_heads,
            keys.shape[1],
        )
        native_output = output.reshape(
            self.shape.layers,
            queries.shape[1],
            self.query_heads,
            self.shape.channels,
        )
        return AtxyLatentAttention(
            weights=weights.contiguous(),
            output=native_output.to(dtype=self.reconstruction_dtype).contiguous(),
        )

    def storage_bytes(
        self,
        key_latents: Tensor,
        value_latents: Tensor,
    ) -> AtxyLocalStorageBytes:
        """Account a deployed rank-prefix payload, including both bases and V centers."""

        keys, key_rank = self._validate_kv_latent(key_latents, "key_latents")
        values, value_rank = self._validate_kv_latent(value_latents, "value_latents")
        if keys.shape[1] != values.shape[1]:
            raise ValueError("key_latents and value_latents must have the same token count")
        storage = self.projected_storage_bytes(
            token_count=int(keys.shape[1]),
            key_rank=key_rank,
            value_rank=value_rank,
        )
        if storage.key_latent_bytes != _logical_bytes(keys):
            raise RuntimeError("projected key latent bytes do not match the encoded tensor")
        if storage.value_latent_bytes != _logical_bytes(values):
            raise RuntimeError("projected value latent bytes do not match the encoded tensor")
        return storage

    def projected_storage_bytes(
        self,
        *,
        token_count: int,
        key_rank: int,
        value_rank: int,
    ) -> AtxyLocalStorageBytes:
        """Project exact logical storage without allocating token-length latent tensors."""

        selected_tokens = _positive_int(token_count, "token_count")
        selected_key_rank = self._rank(key_rank, "key_rank")
        selected_value_rank = self._rank(value_rank, "value_rank")
        native_element_bytes = torch.empty(
            (), dtype=self.reconstruction_dtype, device="cpu"
        ).element_size()
        latent_element_bytes = torch.empty((), dtype=self.latent_dtype, device="cpu").element_size()
        native_payload_bytes = (
            self.shape.layers
            * selected_tokens
            * self.shape.heads
            * self.shape.channels
            * native_element_bytes
        )
        key_latent_bytes = (
            self.shape.layers
            * selected_tokens
            * self.shape.heads
            * selected_key_rank
            * latent_element_bytes
        )
        value_latent_bytes = (
            self.shape.layers
            * selected_tokens
            * self.shape.heads
            * selected_value_rank
            * latent_element_bytes
        )
        basis_element_bytes = self._key_basis.element_size()
        key_basis_bytes = (
            self.shape.layers
            * self.shape.heads
            * self.shape.channels
            * selected_key_rank
            * basis_element_bytes
        )
        value_basis_bytes = (
            self.shape.layers
            * self.shape.heads
            * self.shape.channels
            * selected_value_rank
            * basis_element_bytes
        )
        value_center_bytes = _logical_bytes(self._value_center)
        latent_bytes = key_latent_bytes + value_latent_bytes
        basis_bytes = key_basis_bytes + value_basis_bytes + value_center_bytes
        return AtxyLocalStorageBytes(
            token_count=selected_tokens,
            native_key_bytes=native_payload_bytes,
            native_value_bytes=native_payload_bytes,
            native_bytes=2 * native_payload_bytes,
            key_latent_bytes=key_latent_bytes,
            value_latent_bytes=value_latent_bytes,
            latent_bytes=latent_bytes,
            key_basis_bytes=key_basis_bytes,
            value_basis_bytes=value_basis_bytes,
            value_center_bytes=value_center_bytes,
            basis_bytes=basis_bytes,
            total_bytes=latent_bytes + basis_bytes,
        )


__all__ = [
    "AtxyLatentAttention",
    "AtxyLocalLatentCodec",
    "AtxyLocalStorageBytes",
    "GQA_QUERY_HEADS_PER_KV_HEAD",
]
