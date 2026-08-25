"""Correctness-first primitives for a staged exact-payload ATXY KV cache.

Phase one is a CPU reference implementation.  It builds immutable physical-page
descriptors from exact K/V, computes conservative current-query logit bounds
without reading cold keys, progressively selects one layer-shared physical page
set for every GQA head, and certifies a lower bound on retained attention mass.
Exact page K/V fingerprints bind later staged payloads to the snapshotted bytes
used to construct each descriptor; selector logits are never caller supplied.
If a configured page cap is exhausted before certification, selection fails
open to an explicit dense fallback.

The online attention accumulator consumes exact page payloads in any order and
uses a numerically stable running softmax.  Only single-token causal decode is
supported.  Prefill, speculative multi-token decode, attention bias, and live
model/cache integration are deliberately outside this phase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hmac
import math
from numbers import Real

import torch
from torch import Tensor

from molor.experimental.atxy_pages import AtxyPageAddress
from molor.experimental.hashed_kv import PageSpan, build_page_spans, exact_content_digest


_EXACT_PAYLOAD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_PAYLOAD_FINGERPRINT_SCHEMA = "molor.atxy-staged-kv-page-payload.v1"


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _fraction(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return result


def _scale(value: object | None, head_dim: int) -> float:
    if value is None:
        return 1.0 / math.sqrt(head_dim)
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("scale must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("scale must be finite and positive")
    return result


def _sum_lower(values: Sequence[float]) -> float:
    result = math.fsum(values)
    return math.nextafter(result, 0.0) if result > 0 else 0.0


def _sum_upper(values: Sequence[float]) -> float:
    return math.nextafter(math.fsum(values), math.inf)


def _product_lower(left: float, right: float) -> float:
    result = left * right
    return math.nextafter(result, 0.0) if result > 0 else 0.0


def _product_upper(left: float, right: float) -> float:
    return math.nextafter(left * right, math.inf)


def _exp_lower(value: float) -> float:
    result = math.exp(value)
    return math.nextafter(result, 0.0) if result > 0 else 0.0


def _exp_upper(value: float) -> float:
    return math.nextafter(math.exp(value), math.inf)


def _subtract_lower(left: float, right: float) -> float:
    return math.nextafter(left - right, -math.inf)


def _subtract_upper(left: float, right: float) -> float:
    return math.nextafter(left - right, math.inf)


def _payload_fingerprint(
    payload: Tensor,
    *,
    address: AtxyPageAddress,
    span: PageSpan,
    kind: str,
) -> str:
    if kind not in {"keys", "values"}:
        raise ValueError("payload fingerprint kind must be keys or values")
    namespace = (
        f"schema={_PAYLOAD_FINGERPRINT_SCHEMA}|kind={kind}|address={address.text}|"
        f"token_span={span.start}:{span.stop}"
    )
    return exact_content_digest(payload, namespace=namespace)


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_payload(value: object, name: str, *, dimensions: int = 3) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if value.device.type != "cpu":
        raise ValueError(f"phase-one {name} must be on CPU")
    if value.dtype not in _EXACT_PAYLOAD_DTYPES:
        raise ValueError(f"{name} must use float16, bfloat16, or float32")
    if any(dimension < 1 for dimension in value.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.detach().contiguous().clone()


def _decode_query(value: object, name: str = "query") -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [query_heads, head_dim]; "
            "multi-token decode and prefill are unsupported"
        )
    if value.device.type != "cpu":
        raise ValueError(f"phase-one {name} must be on CPU")
    if value.dtype not in _EXACT_PAYLOAD_DTYPES:
        raise ValueError(f"{name} must use float16, bfloat16, or float32")
    if any(dimension < 1 for dimension in value.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.detach().to(dtype=torch.float64).contiguous()


@dataclass(frozen=True, slots=True, eq=False)
class AtxyPageDescriptor:
    """Exact physical-page identity plus conservative key extrema metadata."""

    address: AtxyPageAddress
    span: PageSpan
    _key_minimum: Tensor
    _key_maximum: Tensor
    key_dtype: torch.dtype
    value_dtype: torch.dtype
    value_head_dim: int
    payload_nbytes: int
    key_fingerprint_sha256: str
    value_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.address, AtxyPageAddress):
            raise TypeError("address must be an AtxyPageAddress")
        if not isinstance(self.span, PageSpan):
            raise TypeError("span must be a PageSpan")
        if self.address.page_id != self.span.page_id:
            raise ValueError("ATXY address and page span IDs differ")
        minimum = self._key_minimum
        maximum = self._key_maximum
        if (
            not isinstance(minimum, Tensor)
            or not isinstance(maximum, Tensor)
            or minimum.device.type != "cpu"
            or maximum.device.type != "cpu"
            or minimum.dtype is not torch.float64
            or maximum.dtype is not torch.float64
            or minimum.ndim != 2
            or minimum.shape != maximum.shape
            or any(dimension < 1 for dimension in minimum.shape)
            or not bool(torch.isfinite(minimum).all())
            or not bool(torch.isfinite(maximum).all())
            or bool((minimum > maximum).any())
        ):
            raise ValueError("key extrema must be finite CPU float64 [kv_heads, head_dim] tensors")
        if self.key_dtype not in _EXACT_PAYLOAD_DTYPES:
            raise ValueError("key_dtype is unsupported")
        if self.value_dtype not in _EXACT_PAYLOAD_DTYPES:
            raise ValueError("value_dtype is unsupported")
        _positive_int(self.value_head_dim, "value_head_dim")
        _positive_int(self.payload_nbytes, "payload_nbytes")
        _validate_sha256(self.key_fingerprint_sha256, "key_fingerprint_sha256")
        _validate_sha256(self.value_fingerprint_sha256, "value_fingerprint_sha256")
        object.__setattr__(self, "_key_minimum", minimum.detach().clone().contiguous())
        object.__setattr__(self, "_key_maximum", maximum.detach().clone().contiguous())

    @property
    def key_minimum(self) -> Tensor:
        return self._key_minimum.clone()

    @property
    def key_maximum(self) -> Tensor:
        return self._key_maximum.clone()

    @property
    def kv_heads(self) -> int:
        return int(self._key_minimum.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self._key_minimum.shape[1])


@dataclass(frozen=True, slots=True)
class AtxyLayerPageIndex:
    """A contiguous immutable physical-page index for one transformer layer."""

    layer_index: int
    pages: tuple[AtxyPageDescriptor, ...]

    def __post_init__(self) -> None:
        _non_negative_int(self.layer_index, "layer_index")
        if not isinstance(self.pages, tuple) or not self.pages:
            raise ValueError("pages must be a non-empty tuple")
        if any(not isinstance(page, AtxyPageDescriptor) for page in self.pages):
            raise TypeError("pages must contain AtxyPageDescriptor values")
        first = self.pages[0]
        expected_start = 0
        for expected_id, page in enumerate(self.pages):
            if page.address.layer_index != self.layer_index:
                raise ValueError("page belongs to a different layer")
            if page.span.page_id != expected_id or page.span.start != expected_start:
                raise ValueError("pages must be ordered and form a contiguous partition")
            if (
                page.kv_heads != first.kv_heads
                or page.head_dim != first.head_dim
                or page.value_head_dim != first.value_head_dim
                or page.key_dtype != first.key_dtype
                or page.value_dtype != first.value_dtype
            ):
                raise ValueError("page descriptor geometries and dtypes must agree")
            expected_start = page.span.stop

    @classmethod
    def from_kv(
        cls,
        *,
        layer_index: int,
        keys: Tensor,
        values: Tensor,
        page_size: int,
    ) -> AtxyLayerPageIndex:
        """Describe exact ``[kv_heads, tokens, dim]`` K/V without retaining payloads."""

        layer_index = _non_negative_int(layer_index, "layer_index")
        page_size = _positive_int(page_size, "page_size")
        key_payload = _exact_payload(keys, "keys")
        value_payload = _exact_payload(values, "values")
        if key_payload.shape[:2] != value_payload.shape[:2]:
            raise ValueError("keys and values must share KV-head and token dimensions")
        descriptors: list[AtxyPageDescriptor] = []
        for span in build_page_spans(int(key_payload.shape[1]), page_size):
            page_keys = key_payload[:, span.start : span.stop]
            page_values = value_payload[:, span.start : span.stop]
            address = AtxyPageAddress.for_layer_page(layer_index, span.page_id)
            descriptors.append(
                AtxyPageDescriptor(
                    address=address,
                    span=span,
                    _key_minimum=page_keys.to(dtype=torch.float64).amin(dim=1),
                    _key_maximum=page_keys.to(dtype=torch.float64).amax(dim=1),
                    key_dtype=page_keys.dtype,
                    value_dtype=page_values.dtype,
                    value_head_dim=int(page_values.shape[2]),
                    payload_nbytes=(
                        page_keys.numel() * page_keys.element_size()
                        + page_values.numel() * page_values.element_size()
                    ),
                    key_fingerprint_sha256=_payload_fingerprint(
                        page_keys,
                        address=address,
                        span=span,
                        kind="keys",
                    ),
                    value_fingerprint_sha256=_payload_fingerprint(
                        page_values,
                        address=address,
                        span=span,
                        kind="values",
                    ),
                )
            )
        return cls(layer_index=layer_index, pages=tuple(descriptors))

    @property
    def kv_heads(self) -> int:
        return self.pages[0].kv_heads

    @property
    def head_dim(self) -> int:
        return self.pages[0].head_dim

    @property
    def total_tokens(self) -> int:
        return self.pages[-1].span.stop

    @property
    def total_payload_nbytes(self) -> int:
        return sum(page.payload_nbytes for page in self.pages)

    @property
    def page_ids(self) -> tuple[int, ...]:
        return tuple(page.span.page_id for page in self.pages)

    def page(self, page_id: int) -> AtxyPageDescriptor:
        page_id = _non_negative_int(page_id, "page_id")
        if page_id >= len(self.pages):
            raise ValueError("page_id is outside the layer index")
        return self.pages[page_id]


def conservative_page_logit_upper_bounds(
    query: Tensor,
    index: AtxyLayerPageIndex,
    *,
    scale: float | None = None,
) -> Tensor:
    """Bound every query head's maximum scaled QK logit in every page.

    Query head ``h`` maps to KV head ``h // groups``.  The returned CPU float64
    tensor has shape ``[query_heads, pages]``.  Bounds assume post-RoPE Q/K and
    no additive attention bias.
    """

    if not isinstance(index, AtxyLayerPageIndex):
        raise TypeError("index must be an AtxyLayerPageIndex")
    query_value = _decode_query(query)
    if query_value.shape[1] != index.head_dim:
        raise ValueError("query and indexed keys have different head dimensions")
    query_heads = int(query_value.shape[0])
    if query_heads % index.kv_heads:
        raise ValueError("query heads must be divisible by KV heads for grouped-query attention")
    groups = query_heads // index.kv_heads
    kv_head_ids = torch.arange(query_heads, dtype=torch.long) // groups
    logit_scale = _scale(scale, index.head_dim)
    float32_unit_roundoff = torch.finfo(torch.float32).eps / 2
    operations = index.head_dim + 2
    if operations * float32_unit_roundoff >= 1:
        raise ValueError("head_dim is too large for a conservative float32 dot-product guard")
    gamma = (operations * float32_unit_roundoff) / (1 - operations * float32_unit_roundoff)
    bounds: list[Tensor] = []
    for page in index.pages:
        minimum = page._key_minimum.index_select(0, kv_head_ids)
        maximum = page._key_maximum.index_select(0, kv_head_ids)
        products = torch.maximum(query_value * minimum, query_value * maximum)
        absolute_products = torch.maximum(
            torch.abs(query_value * minimum), torch.abs(query_value * maximum)
        )
        rounding_guard = absolute_products.sum(dim=1) * logit_scale * gamma
        bound = products.sum(dim=1) * logit_scale + rounding_guard
        bounds.append(torch.nextafter(bound, torch.full_like(bound, math.inf)))
    result = torch.stack(bounds, dim=1).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("query/key bounds produced non-finite logits")
    return result


@dataclass(frozen=True, slots=True)
class LayerSharedSelectionDecision:
    """Final progressive-selection outcome for one physical layer."""

    selected_page_ids: tuple[int, ...]
    probed_page_ids: tuple[int, ...]
    mandatory_page_ids: tuple[int, ...]
    probed_retained_mass_lower_bounds: tuple[float, ...]
    selected_retained_mass_lower_bounds: tuple[float, ...]
    target_mass: float
    dense_fallback: bool
    fallback_reason: str | None
    probe_transfer_nbytes: int
    fallback_transfer_nbytes: int
    total_transfer_nbytes: int
    total_payload_nbytes: int


class ProgressiveLayerSharedSelector:
    """Deterministic physical-page selection with conservative mass certificates."""

    def __init__(
        self,
        *,
        index: AtxyLayerPageIndex,
        query: Tensor,
        mandatory_page_ids: Sequence[int] = (),
        target_mass: float = 0.995,
        page_cap: int,
        initial_pages: int | None = None,
        pages_per_round: int = 1,
        scale: float | None = None,
    ) -> None:
        if not isinstance(index, AtxyLayerPageIndex):
            raise TypeError("index must be an AtxyLayerPageIndex")
        self._index = index
        self._query = _decode_query(query)
        self._logit_scale = _scale(scale, index.head_dim)
        self._upper_bounds = conservative_page_logit_upper_bounds(query, index, scale=scale)
        self._query_heads = int(self._upper_bounds.shape[0])
        groups = self._query_heads // index.kv_heads
        self._kv_head_ids = torch.arange(self._query_heads, dtype=torch.long) // groups
        self._target_mass = _fraction(target_mass, "target_mass")
        self._page_cap = _positive_int(page_cap, "page_cap")
        if self._page_cap > len(index.pages):
            raise ValueError("page_cap must not exceed the physical page count")
        self._pages_per_round = _positive_int(pages_per_round, "pages_per_round")

        if isinstance(mandatory_page_ids, (str, bytes)) or not isinstance(
            mandatory_page_ids, Sequence
        ):
            raise TypeError("mandatory_page_ids must be a sequence")
        mandatory_values = tuple(mandatory_page_ids)
        if any(
            not isinstance(page_id, int)
            or isinstance(page_id, bool)
            or page_id < 0
            or page_id >= len(index.pages)
            for page_id in mandatory_values
        ):
            raise ValueError("mandatory_page_ids contains an invalid page ID")
        if len(set(mandatory_values)) != len(mandatory_values):
            raise ValueError("mandatory_page_ids must be unique")
        mandatory_set = set(mandatory_values)
        self._mandatory_page_ids = tuple(
            page_id for page_id in index.page_ids if page_id in mandatory_set
        )
        if len(self._mandatory_page_ids) > self._page_cap:
            raise ValueError("mandatory pages exceed page_cap")

        minimum_initial = max(1, len(self._mandatory_page_ids))
        if initial_pages is None:
            initial_pages = min(
                self._page_cap,
                max(minimum_initial, len(self._mandatory_page_ids) + self._pages_per_round),
            )
        self._initial_pages = _positive_int(initial_pages, "initial_pages")
        if not minimum_initial <= self._initial_pages <= self._page_cap:
            raise ValueError("initial_pages must include mandatory pages and not exceed page_cap")

        page_lengths = torch.tensor([page.span.length for page in index.pages], dtype=torch.float64)
        self._page_log_upper_mass = self._upper_bounds + page_lengths.log().unsqueeze(0)
        self._observed_log_mass: dict[int, Tensor] = {}
        self._observed_mass_intervals: dict[int, tuple[Tensor, Tensor]] = {}
        self._fetch_order: list[int] = []
        self._pending: tuple[int, ...] = ()
        self._last_lower_bounds: Tensor | None = None
        self._decision: LayerSharedSelectionDecision | None = None
        self._plan_initial_round()

    @property
    def decision(self) -> LayerSharedSelectionDecision | None:
        return self._decision

    @property
    def probed_page_ids(self) -> tuple[int, ...]:
        return tuple(self._fetch_order)

    @property
    def probed_retained_mass_lower_bounds(self) -> tuple[float, ...] | None:
        if self._last_lower_bounds is None:
            return None
        return tuple(float(value) for value in self._last_lower_bounds.tolist())

    def next_page_ids(self) -> tuple[int, ...]:
        if self._decision is not None:
            raise RuntimeError("selection is already final")
        if not self._pending:
            raise RuntimeError("no pending selection round")
        return self._pending

    def _candidate_priority(self, page_id: int) -> float:
        page_upper = self._page_log_upper_mass[:, page_id]
        if not self._observed_log_mass:
            return float(torch.logsumexp(page_upper, dim=0).item())
        selected_log_mass = torch.logsumexp(
            torch.stack(tuple(self._observed_log_mass.values()), dim=0), dim=0
        )
        return float((page_upper - selected_log_mass).amax().item())

    def _rank_unselected(self) -> tuple[int, ...]:
        unavailable = set(self._observed_log_mass).union(self._pending)
        candidates = [page_id for page_id in self._index.page_ids if page_id not in unavailable]
        return tuple(
            sorted(candidates, key=lambda page_id: (-self._candidate_priority(page_id), page_id))
        )

    def _plan_initial_round(self) -> None:
        self._pending = self._mandatory_page_ids
        remaining = self._initial_pages - len(self._pending)
        if remaining:
            self._pending += self._rank_unselected()[:remaining]

    def _plan_next_round(self) -> None:
        remaining_capacity = self._page_cap - len(self._observed_log_mass)
        count = min(self._pages_per_round, remaining_capacity)
        self._pending = self._rank_unselected()[:count]
        if not self._pending:
            raise RuntimeError("selector could not plan a non-empty refinement round")

    def _mass_lower_bounds(self) -> Tensor:
        if not self._observed_log_mass:
            raise RuntimeError("mass certification requires at least one observed page")
        unobserved = [
            page_id for page_id in self._index.page_ids if page_id not in self._observed_log_mass
        ]
        if not unobserved:
            return torch.ones(self._query_heads, dtype=torch.float64)
        results: list[float] = []
        for query_head in range(self._query_heads):
            selected_maxima = [
                float(summary[0][query_head].item())
                for summary in self._observed_mass_intervals.values()
            ]
            omitted_bounds = [
                float(self._upper_bounds[query_head, page_id].item()) for page_id in unobserved
            ]
            shift = max(*selected_maxima, *omitted_bounds)
            selected_contributions = [
                _product_lower(
                    _exp_lower(_subtract_lower(float(maxima[query_head].item()), shift)),
                    float(relative_sum_lower[query_head].item()),
                )
                for maxima, relative_sum_lower in self._observed_mass_intervals.values()
            ]
            selected_lower = _sum_lower(selected_contributions)
            omitted_contributions = [
                _product_upper(
                    _exp_upper(
                        _subtract_upper(
                            float(self._upper_bounds[query_head, page_id].item()), shift
                        )
                    ),
                    float(self._index.page(page_id).span.length),
                )
                for page_id in unobserved
            ]
            omitted_upper = _sum_upper(omitted_contributions)
            denominator_upper = _sum_upper((selected_lower, omitted_upper))
            raw_lower = selected_lower / denominator_upper
            results.append(math.nextafter(raw_lower, 0.0) if raw_lower > 0 else 0.0)
        return torch.tensor(results, dtype=torch.float64)

    def _finalize(self, *, dense_fallback: bool) -> LayerSharedSelectionDecision:
        if self._last_lower_bounds is None:
            raise RuntimeError("cannot finalize without a mass certificate")
        probed = tuple(self._fetch_order)
        probed_set = set(probed)
        probe_bytes = sum(self._index.page(page_id).payload_nbytes for page_id in probed_set)
        if dense_fallback:
            selected = self._index.page_ids
            fallback_bytes = self._index.total_payload_nbytes - probe_bytes
            reason = "page-cap-exhausted-before-mass-certificate"
        else:
            selected = tuple(page_id for page_id in self._index.page_ids if page_id in probed_set)
            fallback_bytes = 0
            reason = None
        probed_mass_lower_bounds = tuple(float(value) for value in self._last_lower_bounds.tolist())
        selected_mass_lower_bounds = (
            (1.0,) * self._query_heads if dense_fallback else probed_mass_lower_bounds
        )
        decision = LayerSharedSelectionDecision(
            selected_page_ids=selected,
            probed_page_ids=probed,
            mandatory_page_ids=self._mandatory_page_ids,
            probed_retained_mass_lower_bounds=probed_mass_lower_bounds,
            selected_retained_mass_lower_bounds=selected_mass_lower_bounds,
            target_mass=self._target_mass,
            dense_fallback=dense_fallback,
            fallback_reason=reason,
            probe_transfer_nbytes=probe_bytes,
            fallback_transfer_nbytes=fallback_bytes,
            total_transfer_nbytes=probe_bytes + fallback_bytes,
            total_payload_nbytes=self._index.total_payload_nbytes,
        )
        self._decision = decision
        self._pending = ()
        return decision

    def observe_exact_page_keys(
        self, page_keys: Mapping[int, Tensor]
    ) -> LayerSharedSelectionDecision | None:
        """Verify pending exact K payloads and compute their GQA logits internally."""

        if self._decision is not None:
            raise RuntimeError("selection is already final")
        if not isinstance(page_keys, Mapping):
            raise TypeError("page_keys must be a mapping")
        if set(page_keys) != set(self._pending) or len(page_keys) != len(self._pending):
            raise ValueError("page_keys must contain exactly the pending page IDs")

        pending_log_mass: dict[int, Tensor] = {}
        pending_intervals: dict[int, tuple[Tensor, Tensor]] = {}
        for page_id in self._pending:
            page = self._index.page(page_id)
            exact_keys = _exact_payload(page_keys[page_id], "page keys")
            if exact_keys.shape != (
                self._index.kv_heads,
                page.span.length,
                self._index.head_dim,
            ):
                raise ValueError("exact page keys have an invalid shape")
            if exact_keys.dtype != page.key_dtype:
                raise ValueError("exact page keys have a different dtype than the descriptor")
            fingerprint = _payload_fingerprint(
                exact_keys,
                address=page.address,
                span=page.span,
                kind="keys",
            )
            if not hmac.compare_digest(fingerprint, page.key_fingerprint_sha256):
                raise ValueError(
                    "exact page keys do not match the immutable descriptor fingerprint"
                )
            keys_by_query = exact_keys.to(dtype=torch.float64).index_select(0, self._kv_head_ids)
            exact = torch.einsum("hd,htd->ht", self._query, keys_by_query) * self._logit_scale
            if not bool(torch.isfinite(exact).all()):
                raise ValueError("exact page logits must be finite")
            observed_maximum = exact.amax(dim=1)
            if bool((observed_maximum > self._upper_bounds[:, page_id]).any()):
                raise ValueError("exact logits violate the conservative page bound")
            relative_sum_lower = torch.tensor(
                [
                    _sum_lower(
                        tuple(
                            _exp_lower(
                                _subtract_lower(value, float(observed_maximum[query_head].item()))
                            )
                            for value in exact[query_head].tolist()
                        )
                    )
                    for query_head in range(self._query_heads)
                ],
                dtype=torch.float64,
            )
            pending_log_mass[page_id] = torch.logsumexp(exact, dim=1)
            pending_intervals[page_id] = (observed_maximum, relative_sum_lower)

        for page_id in self._pending:
            self._observed_log_mass[page_id] = pending_log_mass[page_id]
            self._observed_mass_intervals[page_id] = pending_intervals[page_id]
            self._fetch_order.append(page_id)

        self._pending = ()
        self._last_lower_bounds = self._mass_lower_bounds()
        all_pages_observed = len(self._observed_log_mass) == len(self._index.pages)
        if all_pages_observed or (
            self._target_mass < 1 and bool((self._last_lower_bounds >= self._target_mass).all())
        ):
            return self._finalize(dense_fallback=False)
        if len(self._observed_log_mass) >= self._page_cap:
            return self._finalize(dense_fallback=True)
        self._plan_next_round()
        return None


class OnlineBlockwiseAttentionAccumulator:
    """Stable single-token GQA attention over exact K/V blocks in any order."""

    def __init__(
        self,
        query: Tensor,
        *,
        expected_pages: Sequence[PageSpan],
        scale: float | None = None,
    ) -> None:
        self._query = _decode_query(query)
        self._query_heads = int(self._query.shape[0])
        self._head_dim = int(self._query.shape[1])
        self._scale = _scale(scale, self._head_dim)
        if isinstance(expected_pages, (str, bytes)) or not isinstance(expected_pages, Sequence):
            raise TypeError("expected_pages must be a sequence of PageSpan values")
        expected = tuple(expected_pages)
        if not expected or any(not isinstance(page, PageSpan) for page in expected):
            raise ValueError("expected_pages must contain at least one PageSpan")
        if len({page.page_id for page in expected}) != len(expected):
            raise ValueError("expected page IDs must be unique")
        by_position = sorted(expected, key=lambda page: (page.start, page.stop, page.page_id))
        if any(left.stop > right.start for left, right in zip(by_position, by_position[1:])):
            raise ValueError("expected physical page spans must not overlap")
        self._expected_page_lengths = {page.page_id: page.length for page in expected}
        self._observed_page_ids: set[int] = set()
        self._kv_heads: int | None = None
        self._value_head_dim: int | None = None
        self._running_maximum = torch.full((self._query_heads,), -math.inf, dtype=torch.float64)
        self._running_denominator = torch.zeros(self._query_heads, dtype=torch.float64)
        self._running_numerator: Tensor | None = None
        self._tokens = 0

    @property
    def token_count(self) -> int:
        return self._tokens

    def update(self, page_id: int, keys: Tensor, values: Tensor) -> None:
        page_id = _non_negative_int(page_id, "page_id")
        if page_id not in self._expected_page_lengths:
            raise ValueError("page_id is not part of the expected physical page set")
        if page_id in self._observed_page_ids:
            raise ValueError("physical page was already accumulated")
        key_block = _exact_payload(keys, "keys")
        value_block = _exact_payload(values, "values")
        if key_block.shape[:2] != value_block.shape[:2]:
            raise ValueError("keys and values must share KV-head and token dimensions")
        if key_block.shape[2] != self._head_dim:
            raise ValueError("query and keys have different head dimensions")
        if key_block.shape[1] != self._expected_page_lengths[page_id]:
            raise ValueError("K/V block length differs from the expected physical page")
        kv_heads = int(key_block.shape[0])
        if self._query_heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        value_head_dim = int(value_block.shape[2])
        if self._kv_heads is None:
            running_numerator = torch.zeros(
                (self._query_heads, value_head_dim), dtype=torch.float64
            )
        elif kv_heads != self._kv_heads or value_head_dim != self._value_head_dim:
            raise ValueError("all K/V blocks must use the same GQA and value geometry")
        elif self._running_numerator is None:  # pragma: no cover - internal invariant
            raise RuntimeError("attention numerator geometry is missing")
        else:
            running_numerator = self._running_numerator

        groups = self._query_heads // kv_heads
        kv_head_ids = torch.arange(self._query_heads, dtype=torch.long) // groups
        keys_by_query = key_block.to(dtype=torch.float64).index_select(0, kv_head_ids)
        values_by_query = value_block.to(dtype=torch.float64).index_select(0, kv_head_ids)
        logits = torch.einsum("hd,htd->ht", self._query, keys_by_query) * self._scale
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("QK logits are non-finite")
        block_maximum = logits.amax(dim=1)
        new_maximum = torch.maximum(self._running_maximum, block_maximum)
        prior_scale = torch.where(
            self._running_denominator > 0,
            torch.exp(self._running_maximum - new_maximum),
            torch.zeros_like(new_maximum),
        )
        block_weights = torch.exp(logits - new_maximum.unsqueeze(1))
        block_denominator = block_weights.sum(dim=1)
        block_numerator = torch.einsum("ht,htv->hv", block_weights, values_by_query)
        new_numerator = running_numerator * prior_scale.unsqueeze(1) + block_numerator
        new_denominator = self._running_denominator * prior_scale + block_denominator

        self._kv_heads = kv_heads
        self._value_head_dim = value_head_dim
        self._running_numerator = new_numerator
        self._running_denominator = new_denominator
        self._running_maximum = new_maximum
        self._tokens += int(key_block.shape[1])
        self._observed_page_ids.add(page_id)

    def result(self) -> Tensor:
        if self._running_numerator is None or self._tokens == 0:
            raise RuntimeError("attention requires at least one K/V block")
        if self._observed_page_ids != set(self._expected_page_lengths):
            raise RuntimeError("attention is missing one or more expected physical pages")
        if not bool((self._running_denominator > 0).all()):
            raise RuntimeError("attention denominator is not positive")
        return (self._running_numerator / self._running_denominator.unsqueeze(1)).clone()


__all__ = [
    "AtxyLayerPageIndex",
    "AtxyPageDescriptor",
    "LayerSharedSelectionDecision",
    "OnlineBlockwiseAttentionAccumulator",
    "ProgressiveLayerSharedSelector",
    "conservative_page_logit_upper_bounds",
]
