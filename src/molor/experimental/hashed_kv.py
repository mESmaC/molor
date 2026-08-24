"""Pure-Torch primitives for a hashed-KV shadow experiment.

The routing codes in this module are approximate SimHash addresses.  A collision
only makes two tokens equally attractive to the router: it never makes the tokens
the same cache entry.  ``exact_content_digest`` is deliberately a separate API,
and callers still need ``exact_content_equal`` before any content deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F


_SUPPORTED_HASH_BITS = (32, 64)


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    lower_bound = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower_bound:
        qualifier = "a non-negative" if allow_zero else "a positive"
        raise ValueError(f"{name} must be {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class PageSpan:
    """An immutable half-open token span belonging to one cache page."""

    page_id: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        _positive_int(self.page_id, "page_id", allow_zero=True)
        _positive_int(self.start, "start", allow_zero=True)
        _positive_int(self.stop, "stop")
        if self.stop <= self.start:
            raise ValueError("page stop must be greater than page start")

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class HashedKVConfig:
    """Configuration for a single-query, single-head shadow evaluation.

    ``candidate_pages`` is the total page budget.  Sink and recent pages are
    mandatory and consume that budget; the remaining slots are hash-routed.
    """

    page_size: int = 256
    hash_bits: int = 32
    hash_tables: int = 1
    candidate_pages: int = 8
    sink_pages: int = 1
    recent_pages: int = 1
    top_k: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.page_size, "page_size")
        if self.hash_bits not in _SUPPORTED_HASH_BITS:
            raise ValueError(f"hash_bits must be one of {_SUPPORTED_HASH_BITS}")
        _positive_int(self.hash_tables, "hash_tables")
        _positive_int(self.candidate_pages, "candidate_pages")
        _positive_int(self.sink_pages, "sink_pages", allow_zero=True)
        _positive_int(self.recent_pages, "recent_pages", allow_zero=True)
        _positive_int(self.top_k, "top_k")
        _positive_int(self.seed, "seed", allow_zero=True)
        if self.seed > torch.iinfo(torch.int64).max:
            raise ValueError("seed must fit in a signed 64-bit integer")
        if self.candidate_pages < self.sink_pages + self.recent_pages:
            raise ValueError(
                "candidate_pages must be at least sink_pages + recent_pages"
            )

    @property
    def routing_metadata_bits_per_token_per_head(self) -> int:
        """Packed routing metadata for each indexed token in each indexed head."""

        return self.hash_bits * self.hash_tables

    @property
    def routing_metadata_bytes_per_token_per_head(self) -> int:
        return self.routing_metadata_bits_per_token_per_head // 8


@dataclass(frozen=True, slots=True)
class PageSelection:
    """A deterministic page selection and its expanded token positions."""

    page_ids: tuple[int, ...]
    token_indices: tuple[int, ...]
    mandatory_page_ids: tuple[int, ...]
    ranked_page_ids: tuple[int, ...]
    page_scores: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class AttentionMetrics:
    """Quality and sparsity metrics for one candidate token set."""

    top_k: int
    top_k_recall: float
    retained_attention_mass: float
    relative_output_l2: float
    cosine_similarity: float
    candidate_tokens: int
    total_tokens: int
    candidate_token_fraction: float
    candidate_pages: int
    total_pages: int
    candidate_page_fraction: float


@dataclass(frozen=True, slots=True)
class HashedKVShadowResult:
    """Hash, min/max, and attention-oracle results at an identical page budget."""

    config: HashedKVConfig
    head_dim: int
    pages: tuple[PageSpan, ...]
    forced_selection: PageSelection | None
    hashed_selection: PageSelection
    minmax_selection: PageSelection
    oracle_selection: PageSelection
    forced_metrics: AttentionMetrics | None
    hashed_metrics: AttentionMetrics
    minmax_metrics: AttentionMetrics
    oracle_metrics: AttentionMetrics
    query_routing_codes: Tensor
    key_routing_codes: Tensor
    exact_attention_weights: Tensor


class RandomHyperplaneSimHash:
    """Deterministic random-hyperplane SimHash with independent tables.

    Codes are packed as signed ``int32`` or ``int64`` tensors shaped
    ``[..., tables]``.  Signedness is only a storage detail: Hamming distance
    treats the values as fixed-width bit patterns, including the sign bit.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        bits: int = 32,
        tables: int = 1,
        seed: int = 0,
    ) -> None:
        self.input_dim = _positive_int(input_dim, "input_dim")
        if bits not in _SUPPORTED_HASH_BITS:
            raise ValueError(f"bits must be one of {_SUPPORTED_HASH_BITS}")
        self.bits = bits
        self.tables = _positive_int(tables, "tables")
        self.seed = _positive_int(seed, "seed", allow_zero=True)
        if self.seed > torch.iinfo(torch.int64).max:
            raise ValueError("seed must fit in a signed 64-bit integer")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        planes = torch.randn(
            (self.tables, self.bits, self.input_dim),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        self._hyperplanes = F.normalize(planes, dim=-1).contiguous()

    @property
    def hyperplanes(self) -> Tensor:
        """Return a copy so callers cannot mutate subsequent routing results."""

        return self._hyperplanes.clone()

    def encode(self, vectors: Tensor) -> Tensor:
        """Return packed approximate routing codes for the supplied vectors."""

        if not isinstance(vectors, Tensor):
            raise TypeError("vectors must be a torch.Tensor")
        if vectors.ndim < 1 or vectors.shape[-1] != self.input_dim:
            raise ValueError(f"vectors must end with dimension {self.input_dim}")
        if not vectors.is_floating_point():
            raise ValueError("vectors must have a floating-point dtype")
        cpu_vectors = vectors.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(cpu_vectors).all()):
            raise ValueError("vectors must contain only finite values")
        projections = torch.einsum("...d,tbd->...tb", cpu_vectors, self._hyperplanes)
        bits = projections.ge(0)
        dtype = torch.int32 if self.bits == 32 else torch.int64
        packed = torch.zeros(bits.shape[:-1], dtype=dtype, device="cpu")
        for bit_index in range(self.bits):
            packed.bitwise_or_(bits[..., bit_index].to(dtype) << bit_index)
        return packed.contiguous()


def build_page_spans(token_count: int, page_size: int) -> tuple[PageSpan, ...]:
    """Partition ``token_count`` positions into immutable, contiguous pages."""

    token_count = _positive_int(token_count, "token_count")
    page_size = _positive_int(page_size, "page_size")
    return tuple(
        PageSpan(page_id, start, min(start + page_size, token_count))
        for page_id, start in enumerate(range(0, token_count, page_size))
    )


def _validate_page_partition(
    pages: Iterable[PageSpan], token_count: int
) -> tuple[PageSpan, ...]:
    spans = tuple(pages)
    if not spans:
        raise ValueError("pages must not be empty")
    if any(not isinstance(page, PageSpan) for page in spans):
        raise TypeError("pages must contain PageSpan values")
    if len({page.page_id for page in spans}) != len(spans):
        raise ValueError("page IDs must be unique")
    expected_start = 0
    for page in spans:
        if page.start != expected_start:
            raise ValueError("pages must form a contiguous partition starting at token 0")
        expected_start = page.stop
    if expected_start != token_count:
        raise ValueError("pages must cover every token exactly once")
    return spans


def _validate_page_budget(
    page_budget: int, sink_pages: int, recent_pages: int
) -> tuple[int, int, int]:
    page_budget = _positive_int(page_budget, "page_budget")
    sink_pages = _positive_int(sink_pages, "sink_pages", allow_zero=True)
    recent_pages = _positive_int(recent_pages, "recent_pages", allow_zero=True)
    if page_budget < sink_pages + recent_pages:
        raise ValueError("page_budget must be at least sink_pages + recent_pages")
    return page_budget, sink_pages, recent_pages


def _mandatory_page_ids(
    pages: tuple[PageSpan, ...], sink_pages: int, recent_pages: int
) -> tuple[int, ...]:
    sink = pages[:sink_pages]
    recent = pages[max(0, len(pages) - recent_pages) :] if recent_pages else ()
    ids = {page.page_id for page in (*sink, *recent)}
    return tuple(page.page_id for page in pages if page.page_id in ids)


def _selection_from_ranking(
    pages: tuple[PageSpan, ...],
    *,
    ranking: list[tuple[float, int]],
    page_scores: dict[int, float],
    page_budget: int,
    sink_pages: int,
    recent_pages: int,
) -> PageSelection:
    mandatory = _mandatory_page_ids(pages, sink_pages, recent_pages)
    target_count = min(page_budget, len(pages))
    selected = set(mandatory)
    for _, page_id in ranking:
        if len(selected) >= target_count:
            break
        selected.add(page_id)
    selected_ids = tuple(page.page_id for page in pages if page.page_id in selected)
    selected_tokens = tuple(
        token
        for page in pages
        if page.page_id in selected
        for token in range(page.start, page.stop)
    )
    return PageSelection(
        page_ids=selected_ids,
        token_indices=selected_tokens,
        mandatory_page_ids=mandatory,
        ranked_page_ids=tuple(page_id for _, page_id in ranking),
        page_scores=tuple((page.page_id, page_scores[page.page_id]) for page in pages),
    )


def _forced_page_selection(
    pages: tuple[PageSpan, ...], sink_pages: int, recent_pages: int
) -> PageSelection | None:
    """Expand the mandatory sink/recent baseline without adding routed pages."""

    mandatory = _mandatory_page_ids(pages, sink_pages, recent_pages)
    if not mandatory:
        return None
    mandatory_set = set(mandatory)
    tokens = tuple(
        token
        for page in pages
        if page.page_id in mandatory_set
        for token in range(page.start, page.stop)
    )
    return PageSelection(
        page_ids=mandatory,
        token_indices=tokens,
        mandatory_page_ids=mandatory,
        ranked_page_ids=(),
        page_scores=(),
    )


def _routing_codes(codes: Tensor, name: str, *, token_codes: bool) -> Tensor:
    if not isinstance(codes, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_dimensions = 2 if token_codes else 1
    if codes.ndim != expected_dimensions:
        shape = "[tokens, tables]" if token_codes else "[tables]"
        raise ValueError(f"{name} must have shape {shape}")
    if codes.shape[-1] == 0:
        raise ValueError(f"{name} must contain at least one hash table")
    cpu_codes = codes.detach().to(device="cpu")
    if cpu_codes.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must contain packed int32 or int64 codes")
    return cpu_codes.contiguous()


def _hamming_distances(query_codes: Tensor, token_codes: Tensor) -> Tensor:
    bit_count = token_codes.element_size() * 8
    xor = torch.bitwise_xor(token_codes, query_codes.unsqueeze(0))
    shifts = torch.arange(bit_count, dtype=xor.dtype, device="cpu")
    unpacked = torch.bitwise_and(
        torch.bitwise_right_shift(xor.unsqueeze(-1), shifts),
        1,
    )
    return unpacked.sum(dim=-1)


def select_hashed_pages(
    query_codes: Tensor,
    token_codes: Tensor,
    pages: Iterable[PageSpan],
    *,
    page_budget: int,
    sink_pages: int = 1,
    recent_pages: int = 1,
) -> PageSelection:
    """Select pages by their closest token/table Hamming distance.

    Every token remains independently addressable even when routing codes collide.
    Page IDs break equal-distance ties, making selection deterministic.
    """

    page_budget, sink_pages, recent_pages = _validate_page_budget(
        page_budget, sink_pages, recent_pages
    )
    query = _routing_codes(query_codes, "query_codes", token_codes=False)
    tokens = _routing_codes(token_codes, "token_codes", token_codes=True)
    if query.dtype != tokens.dtype or query.shape != tokens.shape[1:]:
        raise ValueError("query and token routing-code dtype/table dimensions must match")
    spans = _validate_page_partition(pages, tokens.shape[0])

    distances = _hamming_distances(query, tokens)
    scores = {
        page.page_id: float(distances[page.start : page.stop].amin().item())
        for page in spans
    }
    ranking = sorted((score, page_id) for page_id, score in scores.items())
    return _selection_from_ranking(
        spans,
        ranking=ranking,
        page_scores=scores,
        page_budget=page_budget,
        sink_pages=sink_pages,
        recent_pages=recent_pages,
    )


def select_oracle_pages(
    attention_weights: Tensor,
    pages: Iterable[PageSpan],
    *,
    page_budget: int,
    sink_pages: int = 1,
    recent_pages: int = 1,
) -> PageSelection:
    """Select pages with the greatest exact attention mass."""

    page_budget, sink_pages, recent_pages = _validate_page_budget(
        page_budget, sink_pages, recent_pages
    )
    if not isinstance(attention_weights, Tensor):
        raise TypeError("attention_weights must be a torch.Tensor")
    if attention_weights.ndim != 1 or attention_weights.numel() == 0:
        raise ValueError("attention_weights must be a non-empty vector")
    weights = attention_weights.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("attention_weights must be finite and non-negative")
    total = float(weights.sum().item())
    if total <= 0:
        raise ValueError("attention_weights must have positive total mass")
    weights = weights / total
    spans = _validate_page_partition(pages, weights.numel())
    scores = {
        page.page_id: float(weights[page.start : page.stop].sum().item())
        for page in spans
    }
    ranking = sorted((-score, page_id) for page_id, score in scores.items())
    return _selection_from_ranking(
        spans,
        ranking=ranking,
        page_scores=scores,
        page_budget=page_budget,
        sink_pages=sink_pages,
        recent_pages=recent_pages,
    )


def select_minmax_pages(
    query: Tensor,
    keys: Tensor,
    pages: Iterable[PageSpan],
    page_budget: int,
    sink_pages: int = 1,
    recent_pages: int = 1,
) -> PageSelection:
    """Select pages using a training-free per-dimension min/max upper bound.

    For page bounds ``k_min`` and ``k_max``, the score is
    ``sum(max(q * k_min, q * k_max)) / sqrt(head_dim)``.  It upper-bounds the
    scaled query/key score of every token in the page.  Larger bounds rank first;
    page IDs provide a stable tie break.
    """

    page_budget, sink_pages, recent_pages = _validate_page_budget(
        page_budget, sink_pages, recent_pages
    )
    query_cpu, keys_cpu = _validate_query_keys(query, keys)
    spans = _validate_page_partition(pages, keys_cpu.shape[0])
    scale = 1.0 / math.sqrt(query_cpu.numel())
    scores: dict[int, float] = {}
    for page in spans:
        page_keys = keys_cpu[page.start : page.stop]
        key_min = page_keys.amin(dim=0)
        key_max = page_keys.amax(dim=0)
        upper_bound = torch.maximum(query_cpu * key_min, query_cpu * key_max).sum()
        scores[page.page_id] = float((upper_bound * scale).item())
    ranking = sorted((-score, page_id) for page_id, score in scores.items())
    return _selection_from_ranking(
        spans,
        ranking=ranking,
        page_scores=scores,
        page_budget=page_budget,
        sink_pages=sink_pages,
        recent_pages=recent_pages,
    )


def exact_attention_weights(query: Tensor, keys: Tensor) -> Tensor:
    """Compute the float32 scaled-dot-product attention distribution on CPU."""

    query_cpu, keys_cpu = _validate_query_keys(query, keys)
    scale = 1.0 / math.sqrt(query_cpu.numel())
    return torch.softmax(torch.mv(keys_cpu, query_cpu) * scale, dim=0)


def _validate_query_keys(query: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(query, Tensor) or not isinstance(keys, Tensor):
        raise TypeError("query and keys must be torch.Tensor values")
    if query.ndim != 1 or query.numel() == 0:
        raise ValueError("query must be a non-empty vector")
    if keys.ndim != 2 or keys.shape[0] == 0 or keys.shape[1] != query.shape[0]:
        raise ValueError("keys must have shape [tokens, query_dim]")
    if not query.is_floating_point() or not keys.is_floating_point():
        raise ValueError("query and keys must have floating-point dtypes")
    query_cpu = query.detach().to(device="cpu", dtype=torch.float32)
    keys_cpu = keys.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(query_cpu).all()) or not bool(torch.isfinite(keys_cpu).all()):
        raise ValueError("query and keys must contain only finite values")
    return query_cpu, keys_cpu


def _candidate_indices(indices: Iterable[int], token_count: int) -> tuple[int, ...]:
    try:
        candidates = tuple(indices)
    except TypeError as error:
        raise TypeError("candidate_token_indices must be an iterable of integers") from error
    if not candidates:
        raise ValueError("candidate_token_indices must not be empty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in candidates):
        raise ValueError("candidate token indices must be integers")
    if any(index < 0 or index >= token_count for index in candidates):
        raise ValueError("candidate token index is out of range")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate token indices must be unique")
    return tuple(sorted(candidates))


def evaluate_candidate_tokens(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    candidate_token_indices: Iterable[int],
    *,
    top_k: int = 64,
    candidate_page_count: int = 1,
    total_page_count: int = 1,
) -> AttentionMetrics:
    """Compare exact attention with exact attention over a candidate subset."""

    query_cpu, keys_cpu = _validate_query_keys(query, keys)
    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 2 or values.shape[0] != keys_cpu.shape[0] or values.shape[1] == 0:
        raise ValueError("values must have shape [tokens, value_dim]")
    if not values.is_floating_point():
        raise ValueError("values must have a floating-point dtype")
    values_cpu = values.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(values_cpu).all()):
        raise ValueError("values must contain only finite values")
    top_k = _positive_int(top_k, "top_k")
    candidate_page_count = _positive_int(candidate_page_count, "candidate_page_count")
    total_page_count = _positive_int(total_page_count, "total_page_count")
    if candidate_page_count > total_page_count:
        raise ValueError("candidate_page_count cannot exceed total_page_count")
    candidates = _candidate_indices(candidate_token_indices, keys_cpu.shape[0])

    scale = 1.0 / math.sqrt(query_cpu.numel())
    logits = torch.mv(keys_cpu, query_cpu) * scale
    weights = torch.softmax(logits, dim=0)
    candidate_index = torch.tensor(candidates, dtype=torch.long)
    candidate_logits = logits.index_select(0, candidate_index)
    candidate_weights = torch.softmax(candidate_logits, dim=0)

    exact_output = torch.matmul(weights, values_cpu)
    candidate_output = torch.matmul(
        candidate_weights, values_cpu.index_select(0, candidate_index)
    )
    output_delta = torch.linalg.vector_norm(candidate_output - exact_output)
    exact_norm = torch.linalg.vector_norm(exact_output)
    if float(exact_norm.item()) == 0.0:
        relative_output_l2 = 0.0 if float(output_delta.item()) == 0.0 else math.inf
    else:
        relative_output_l2 = float((output_delta / exact_norm).item())
    candidate_norm = torch.linalg.vector_norm(candidate_output)
    if float(exact_norm.item()) == 0.0 and float(candidate_norm.item()) == 0.0:
        cosine_similarity = 1.0
    elif float(exact_norm.item()) == 0.0 or float(candidate_norm.item()) == 0.0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = float(
            torch.dot(exact_output, candidate_output).div(exact_norm * candidate_norm).item()
        )
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

    evaluated_top_k = min(top_k, keys_cpu.shape[0])
    exact_top = torch.topk(weights, k=evaluated_top_k, sorted=False).indices
    candidate_mask = torch.zeros(keys_cpu.shape[0], dtype=torch.bool)
    candidate_mask[candidate_index] = True
    top_k_recall = float(candidate_mask[exact_top].float().mean().item())
    retained_mass = float(weights.index_select(0, candidate_index).sum().item())
    candidate_tokens = len(candidates)
    total_tokens = keys_cpu.shape[0]

    return AttentionMetrics(
        top_k=evaluated_top_k,
        top_k_recall=top_k_recall,
        retained_attention_mass=retained_mass,
        relative_output_l2=relative_output_l2,
        cosine_similarity=cosine_similarity,
        candidate_tokens=candidate_tokens,
        total_tokens=total_tokens,
        candidate_token_fraction=candidate_tokens / total_tokens,
        candidate_pages=candidate_page_count,
        total_pages=total_page_count,
        candidate_page_fraction=candidate_page_count / total_page_count,
    )


def evaluate_hashed_kv(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    config: HashedKVConfig | None = None,
) -> HashedKVShadowResult:
    """Run one deterministic hashed-KV shadow evaluation on CPU.

    This function measures routing only.  Every selector uses the original keys
    and values for its final attention calculation; no approximate code is treated
    as content identity and no cache entry is deduplicated.
    """

    if config is None:
        config = HashedKVConfig()
    if not isinstance(config, HashedKVConfig):
        raise TypeError("config must be a HashedKVConfig")
    query_cpu, keys_cpu = _validate_query_keys(query, keys)
    if not isinstance(values, Tensor) or values.ndim != 2 or values.shape[0] != keys_cpu.shape[0]:
        raise ValueError("values must have shape [tokens, value_dim]")
    pages = build_page_spans(keys_cpu.shape[0], config.page_size)
    page_budget = min(config.candidate_pages, len(pages))
    # When a sequence is shorter than the configured mandatory regions, their
    # overlap makes this runtime budget valid even if it is below their raw sum.
    runtime_sink = min(config.sink_pages, len(pages))
    runtime_recent = min(config.recent_pages, max(0, len(pages) - runtime_sink))
    page_budget = max(page_budget, runtime_sink + runtime_recent)
    forced_selection = _forced_page_selection(pages, runtime_sink, runtime_recent)

    hasher = RandomHyperplaneSimHash(
        keys_cpu.shape[1],
        bits=config.hash_bits,
        tables=config.hash_tables,
        seed=config.seed,
    )
    query_codes = hasher.encode(query_cpu)
    key_codes = hasher.encode(keys_cpu)
    hashed_selection = select_hashed_pages(
        query_codes,
        key_codes,
        pages,
        page_budget=page_budget,
        sink_pages=runtime_sink,
        recent_pages=runtime_recent,
    )
    minmax_selection = select_minmax_pages(
        query_cpu,
        keys_cpu,
        pages,
        page_budget,
        runtime_sink,
        runtime_recent,
    )
    exact_weights = exact_attention_weights(query_cpu, keys_cpu)
    oracle_selection = select_oracle_pages(
        exact_weights,
        pages,
        page_budget=page_budget,
        sink_pages=runtime_sink,
        recent_pages=runtime_recent,
    )
    forced_metrics = (
        evaluate_candidate_tokens(
            query_cpu,
            keys_cpu,
            values,
            forced_selection.token_indices,
            top_k=config.top_k,
            candidate_page_count=len(forced_selection.page_ids),
            total_page_count=len(pages),
        )
        if forced_selection is not None
        else None
    )
    hashed_metrics = evaluate_candidate_tokens(
        query_cpu,
        keys_cpu,
        values,
        hashed_selection.token_indices,
        top_k=config.top_k,
        candidate_page_count=len(hashed_selection.page_ids),
        total_page_count=len(pages),
    )
    minmax_metrics = evaluate_candidate_tokens(
        query_cpu,
        keys_cpu,
        values,
        minmax_selection.token_indices,
        top_k=config.top_k,
        candidate_page_count=len(minmax_selection.page_ids),
        total_page_count=len(pages),
    )
    oracle_metrics = evaluate_candidate_tokens(
        query_cpu,
        keys_cpu,
        values,
        oracle_selection.token_indices,
        top_k=config.top_k,
        candidate_page_count=len(oracle_selection.page_ids),
        total_page_count=len(pages),
    )
    return HashedKVShadowResult(
        config=config,
        head_dim=keys_cpu.shape[1],
        pages=pages,
        forced_selection=forced_selection,
        hashed_selection=hashed_selection,
        minmax_selection=minmax_selection,
        oracle_selection=oracle_selection,
        forced_metrics=forced_metrics,
        hashed_metrics=hashed_metrics,
        minmax_metrics=minmax_metrics,
        oracle_metrics=oracle_metrics,
        query_routing_codes=query_codes,
        key_routing_codes=key_codes,
        exact_attention_weights=exact_weights,
    )


def exact_content_digest(tensor: Tensor, *, namespace: str) -> str:
    """Return a namespaced SHA-256 digest of exact tensor metadata and bytes.

    ``namespace`` must canonically identify the cache schema, model/adapter route,
    positional convention, layer, dtype/layout contract, and quantizer when
    applicable.  This low-level digest is still only a lookup prefilter: verify a
    match with ``exact_content_equal`` before sharing stored KV data.
    """

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if tensor.layout is not torch.strided:
        raise ValueError("tensor must use strided layout")
    value = tensor.detach().to(device="cpu").contiguous().clone()
    digest = hashlib.sha256()
    namespace_bytes = namespace.encode("utf-8")
    digest.update(len(namespace_bytes).to_bytes(8, byteorder="big"))
    digest.update(namespace_bytes)
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dimension) for dimension in value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes(value.untyped_storage()))
    return digest.hexdigest()


def exact_content_equal(left: Tensor, right: Tensor) -> bool:
    """Check byte-exact tensor identity; approximate routing codes are irrelevant."""

    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        raise TypeError("left and right must be torch.Tensor values")
    if left.layout is not torch.strided or right.layout is not torch.strided:
        return False
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    left_value = left.detach().to(device="cpu").contiguous().clone()
    right_value = right.detach().to(device="cpu").contiguous().clone()
    return bytes(left_value.untyped_storage()) == bytes(right_value.untyped_storage())


__all__ = [
    "AttentionMetrics",
    "HashedKVConfig",
    "HashedKVShadowResult",
    "PageSelection",
    "PageSpan",
    "RandomHyperplaneSimHash",
    "build_page_spans",
    "evaluate_candidate_tokens",
    "evaluate_hashed_kv",
    "exact_attention_weights",
    "exact_content_digest",
    "exact_content_equal",
    "select_hashed_pages",
    "select_minmax_pages",
    "select_oracle_pages",
]
