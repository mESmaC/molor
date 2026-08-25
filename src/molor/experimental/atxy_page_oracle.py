"""Exact-payload page-oracle primitives for ATXY-addressed KV experiments.

The oracle deliberately observes the dense attention distribution for the
current query.  It is therefore an offline ceiling, never a deployable routing
policy.  At a fixed page count it chooses the non-mandatory pages with the
greatest exact attention mass, while retaining declared sink/recent pages.

ATXY addresses remain physical identities.  Nothing in this module treats an
address as a semantic hash or alters, compresses, or aliases a KV payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable

import torch
from torch import Tensor

from molor.experimental.hashed_kv import PageSpan, build_page_spans, exact_attention_weights


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fraction(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return result


def _support_spans(
    spans: Iterable[tuple[int, int]], *, token_count: int
) -> tuple[tuple[int, int], ...]:
    try:
        normalized = tuple(spans)
    except TypeError as error:
        raise TypeError("support_token_spans must be an iterable") from error
    for span in normalized:
        if (
            not isinstance(span, tuple)
            or len(span) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in span)
        ):
            raise ValueError("each support token span must be an integer (start, stop) tuple")
        start, stop = span
        if start < 0 or stop <= start or stop > token_count:
            raise ValueError("support token span is outside the causal token range")
    return normalized


def _mandatory_page_ids(
    pages: tuple[PageSpan, ...], *, sink_pages: int, recent_pages: int
) -> tuple[int, ...]:
    sink_pages = min(_positive_int(sink_pages, "sink_pages"), len(pages)) if sink_pages else 0
    recent_pages = (
        min(_positive_int(recent_pages, "recent_pages"), len(pages)) if recent_pages else 0
    )
    selected = {page.page_id for page in pages[:sink_pages]}
    if recent_pages:
        selected.update(page.page_id for page in pages[-recent_pages:])
    return tuple(page.page_id for page in pages if page.page_id in selected)


def _cosine_similarity(reference: Tensor, candidate: Tensor) -> float:
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    reference_value = float(reference_norm.item())
    candidate_value = float(candidate_norm.item())
    if reference_value == 0 and candidate_value == 0:
        return 1.0
    if reference_value == 0 or candidate_value == 0:
        return 0.0
    similarity = float(torch.dot(reference, candidate).div(reference_norm * candidate_norm).item())
    return max(-1.0, min(1.0, similarity))


@dataclass(frozen=True, slots=True)
class PageOracleBudgetMetrics:
    """Exact attention and output fidelity at one total candidate-page budget."""

    candidate_pages: int
    total_pages: int
    candidate_page_fraction: float
    candidate_tokens: int
    total_tokens: int
    candidate_token_fraction: float
    selected_page_ids: tuple[int, ...]
    mandatory_page_ids: tuple[int, ...]
    retained_attention_mass: float
    forced_attention_mass: float
    incremental_attention_mass_over_forced: float
    cold_attention_mass_recall: float | None
    top_k: int
    top_k_recall: float
    maximum_attention_token_page_recalled: bool
    top_attention_page_recall_1: float
    top_attention_page_recall_4: float
    support_page_recall: float | None
    output_relative_l2: float
    output_cosine_similarity: float
    output_delta_l2_squared: float
    exact_output_l2_squared: float


@dataclass(frozen=True, slots=True)
class ExactPageOracleCurve:
    """Sufficient statistics for every exact-mass page budget of one Q/K/V head."""

    pages: tuple[PageSpan, ...]
    page_masses: Tensor
    page_weighted_values: Tensor
    exact_output: Tensor
    mandatory_page_ids: tuple[int, ...]
    oracle_ranked_page_ids: tuple[int, ...]
    top_token_page_ids: tuple[int, ...]
    top_attention_page_ids: tuple[int, ...]
    top_k: int

    @property
    def total_tokens(self) -> int:
        return self.pages[-1].stop

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def minimum_candidate_pages(self) -> int:
        return len(self.mandatory_page_ids)

    def candidate_pages_for_fraction(self, fraction: float) -> int:
        fraction = _fraction(fraction, "candidate fraction")
        return min(
            self.total_pages,
            max(self.minimum_candidate_pages, math.ceil(self.total_pages * fraction)),
        )

    def minimum_candidate_pages_for_mass(self, target_mass: float) -> int:
        target_mass = _fraction(target_mass, "target mass")
        selected = set(self.mandatory_page_ids)
        mass = float(self.page_masses[list(self.mandatory_page_ids)].sum().item())
        if mass >= target_mass:
            return len(selected)
        for page_id in self.oracle_ranked_page_ids:
            selected.add(page_id)
            mass += float(self.page_masses[page_id].item())
            if mass >= target_mass:
                return len(selected)
        return self.total_pages

    def metrics_for_fraction(
        self,
        fraction: float,
        *,
        support_token_spans: Iterable[tuple[int, int]] = (),
    ) -> PageOracleBudgetMetrics:
        return self.metrics_for_pages(
            self.candidate_pages_for_fraction(fraction),
            support_token_spans=support_token_spans,
        )

    def metrics_for_pages(
        self,
        candidate_pages: int,
        *,
        support_token_spans: Iterable[tuple[int, int]] = (),
    ) -> PageOracleBudgetMetrics:
        candidate_pages = _positive_int(candidate_pages, "candidate_pages")
        if not self.minimum_candidate_pages <= candidate_pages <= self.total_pages:
            raise ValueError(
                "candidate_pages must include every mandatory page and not exceed total pages"
            )
        spans = _support_spans(support_token_spans, token_count=self.total_tokens)
        extra_count = candidate_pages - self.minimum_candidate_pages
        selected = set(self.mandatory_page_ids)
        selected.update(self.oracle_ranked_page_ids[:extra_count])
        selected_page_ids = tuple(page.page_id for page in self.pages if page.page_id in selected)
        index = torch.tensor(selected_page_ids, dtype=torch.long, device="cpu")

        if candidate_pages == self.total_pages:
            retained_mass = 1.0
            candidate_output = self.exact_output
        else:
            retained_mass = float(self.page_masses.index_select(0, index).sum().item())
            numerator = self.page_weighted_values.index_select(0, index).sum(dim=0)
            candidate_output = numerator / retained_mass

        forced_index = torch.tensor(self.mandatory_page_ids, dtype=torch.long, device="cpu")
        forced_mass = float(self.page_masses.index_select(0, forced_index).sum().item())
        incremental_mass = max(0.0, retained_mass - forced_mass)
        cold_mass = max(0.0, 1.0 - forced_mass)
        cold_recall = incremental_mass / cold_mass if cold_mass > 1e-12 else None

        selected_mask = torch.zeros(self.total_pages, dtype=torch.bool, device="cpu")
        selected_mask[index] = True
        top_k_hits = sum(bool(selected_mask[page_id]) for page_id in self.top_token_page_ids)
        top_k_recall = top_k_hits / self.top_k
        maximum_recalled = bool(selected_mask[self.top_token_page_ids[0]])

        def top_page_recall(count: int) -> float:
            page_ids = self.top_attention_page_ids[: min(count, self.total_pages)]
            return sum(bool(selected_mask[page_id]) for page_id in page_ids) / len(page_ids)

        support_page_ids = {
            page.page_id
            for start, stop in spans
            for page in self.pages
            if page.start < stop and page.stop > start
        }
        support_recall = (
            len(selected.intersection(support_page_ids)) / len(support_page_ids)
            if support_page_ids
            else None
        )

        delta = candidate_output - self.exact_output
        delta_squared = float(torch.dot(delta, delta).item())
        exact_squared = float(torch.dot(self.exact_output, self.exact_output).item())
        if exact_squared == 0:
            relative_l2 = 0.0 if delta_squared == 0 else math.inf
        else:
            relative_l2 = math.sqrt(delta_squared / exact_squared)
        candidate_tokens = sum(self.pages[page_id].length for page_id in selected_page_ids)

        return PageOracleBudgetMetrics(
            candidate_pages=candidate_pages,
            total_pages=self.total_pages,
            candidate_page_fraction=candidate_pages / self.total_pages,
            candidate_tokens=candidate_tokens,
            total_tokens=self.total_tokens,
            candidate_token_fraction=candidate_tokens / self.total_tokens,
            selected_page_ids=selected_page_ids,
            mandatory_page_ids=self.mandatory_page_ids,
            retained_attention_mass=retained_mass,
            forced_attention_mass=forced_mass,
            incremental_attention_mass_over_forced=incremental_mass,
            cold_attention_mass_recall=cold_recall,
            top_k=self.top_k,
            top_k_recall=top_k_recall,
            maximum_attention_token_page_recalled=maximum_recalled,
            top_attention_page_recall_1=top_page_recall(1),
            top_attention_page_recall_4=top_page_recall(4),
            support_page_recall=support_recall,
            output_relative_l2=relative_l2,
            output_cosine_similarity=_cosine_similarity(self.exact_output, candidate_output),
            output_delta_l2_squared=delta_squared,
            exact_output_l2_squared=exact_squared,
        )


def build_exact_page_oracle_curve(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    page_size: int,
    sink_pages: int = 1,
    recent_pages: int = 1,
    top_k: int = 32,
) -> ExactPageOracleCurve:
    """Build one CPU float32 exact-attention page-oracle curve.

    Dense QK/softmax and weighted-V statistics are calculated once.  Evaluating
    additional budgets then requires only page-level reductions.
    """

    weights = exact_attention_weights(query, keys)
    return build_exact_page_oracle_curve_from_weights(
        weights,
        values,
        page_size=page_size,
        sink_pages=sink_pages,
        recent_pages=recent_pages,
        top_k=top_k,
    )


def build_exact_page_oracle_curve_from_weights(
    attention_weights: Tensor,
    values: Tensor,
    *,
    page_size: int,
    sink_pages: int = 1,
    recent_pages: int = 1,
    top_k: int = 32,
) -> ExactPageOracleCurve:
    """Build a page curve from one validated dense attention distribution.

    Callers sweeping multiple page sizes should calculate dense QK/softmax once,
    then pass the same weights to this function for every geometry.
    """

    page_size = _positive_int(page_size, "page_size")
    top_k = _positive_int(top_k, "top_k")
    if not isinstance(attention_weights, Tensor):
        raise TypeError("attention_weights must be a torch.Tensor")
    if attention_weights.ndim != 1 or attention_weights.numel() == 0:
        raise ValueError("attention_weights must be a non-empty vector")
    if not attention_weights.is_floating_point():
        raise ValueError("attention_weights must have a floating-point dtype")
    weights = attention_weights.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("attention_weights must be finite and non-negative")
    total_mass = float(weights.sum().item())
    if not math.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("attention_weights must have positive total mass")
    if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("attention_weights must be normalized to unit mass")
    weights = (weights / total_mass).contiguous()
    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 2 or values.shape[0] != weights.numel() or values.shape[1] == 0:
        raise ValueError("values must have shape [tokens, value_dim]")
    if not values.is_floating_point():
        raise ValueError("values must have a floating-point dtype")
    values_cpu = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(values_cpu).all()):
        raise ValueError("values must contain only finite values")

    pages = build_page_spans(int(weights.numel()), page_size)
    mandatory = _mandatory_page_ids(
        pages,
        sink_pages=sink_pages,
        recent_pages=recent_pages,
    )
    page_masses = torch.empty(len(pages), dtype=torch.float32, device="cpu")
    page_weighted_values = torch.empty(
        (len(pages), values_cpu.shape[1]), dtype=torch.float32, device="cpu"
    )
    full_page_count = int(weights.numel()) // page_size
    full_token_count = full_page_count * page_size
    if full_page_count:
        full_weights = weights[:full_token_count].reshape(full_page_count, page_size)
        full_values = values_cpu[:full_token_count].reshape(
            full_page_count, page_size, values_cpu.shape[1]
        )
        page_masses[:full_page_count] = full_weights.sum(dim=1)
        page_weighted_values[:full_page_count] = torch.bmm(
            full_weights.unsqueeze(1), full_values
        ).squeeze(1)
    if full_token_count < weights.numel():
        final_page = pages[-1]
        final_weights = weights[full_token_count:]
        final_values = values_cpu[full_token_count:]
        page_masses[final_page.page_id] = final_weights.sum()
        page_weighted_values[final_page.page_id] = torch.matmul(final_weights, final_values)

    mandatory_set = set(mandatory)
    ranked = tuple(
        sorted(
            (page.page_id for page in pages if page.page_id not in mandatory_set),
            key=lambda page_id: (-float(page_masses[page_id].item()), page_id),
        )
    )
    evaluated_top_k = min(top_k, int(weights.numel()))
    top_token_indices = torch.topk(weights, k=evaluated_top_k, sorted=True).indices.tolist()
    top_token_page_ids = tuple(int(index) // page_size for index in top_token_indices)
    top_attention_page_ids = tuple(
        sorted(
            (page.page_id for page in pages),
            key=lambda page_id: (-float(page_masses[page_id].item()), page_id),
        )
    )
    exact_output = torch.matmul(weights, values_cpu)
    return ExactPageOracleCurve(
        pages=pages,
        page_masses=page_masses.contiguous(),
        page_weighted_values=page_weighted_values.contiguous(),
        exact_output=exact_output.contiguous(),
        mandatory_page_ids=mandatory,
        oracle_ranked_page_ids=ranked,
        top_token_page_ids=top_token_page_ids,
        top_attention_page_ids=top_attention_page_ids,
        top_k=evaluated_top_k,
    )
