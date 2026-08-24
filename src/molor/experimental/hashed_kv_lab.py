"""Capture real Qwen Q/K/V and grade hashed page routing in shadow mode."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import torch
from torch import Tensor

from molor.experimental.hashed_kv import HashedKVConfig, evaluate_hashed_kv


DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/hashed-kv-v1")
DEFAULT_LAYERS = (0, 13, 27)
DEFAULT_QUERY_HEADS = (0, 5, 10, 15)

_CONTEXT_PREFIX = """\
This is a synthetic heterogeneous context for a KV retrieval measurement.

"""

_EVIDENCE_LEAD = "Archive checkpoint: the archival recovery code is"
_RECOVERY_CODE = " ORCHID-7319"
_EVIDENCE_TAIL = ". Preserve that exact code for the final question.\n\n"

_FILLER_BLOCK = """\
Embedded systems note: a peripheral should remain in reset until its clock and pin routing are
configured. Software architecture note: isolate policy from transport and persist durable state
before acknowledging a transition. Finance note: cash flow and accounting profit measure different
things. Logic note: a conclusion is valid only when every allowed assignment satisfies it. Ocean
tides are driven primarily by differential lunar gravity, with a smaller solar contribution.

"""

_QUERY_SUFFIX = """\
End of archive. What exact archival recovery code was specified at the archive checkpoint?
"""


@dataclass(frozen=True)
class CapturedLayer:
    """CPU copy of one layer's real post-RoPE queries/keys and matching values."""

    layer_index: int
    query_positions: tuple[int, ...]
    queries: Tensor
    keys: Tensor
    values: Tensor


def _local_or_hub_model(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().to("cpu").contiguous().clone()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(dimension) for dimension in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes(contiguous.untyped_storage()))
    return digest.hexdigest()


def _compact_result(
    result: Any, *, known_evidence_span: tuple[int, int] | None = None
) -> dict[str, Any]:
    attention = result.exact_attention_weights.detach().to(dtype=torch.float32, device="cpu")
    top_count = min(result.config.top_k, int(attention.numel()))
    top_values, top_indices = torch.topk(attention, k=top_count)

    forced_tokens = (
        set(result.forced_selection.token_indices)
        if result.forced_selection is not None
        else set()
    )
    forced_mass = (
        float(attention[list(sorted(forced_tokens))].sum().item()) if forced_tokens else 0.0
    )
    cold_mass = max(0.0, 1.0 - forced_mass)

    def attribution(selection: Any) -> dict[str, float | None]:
        routed_tokens = sorted(set(selection.token_indices) - forced_tokens)
        incremental_mass = (
            float(attention[routed_tokens].sum().item()) if routed_tokens else 0.0
        )
        return {
            "incremental_attention_mass_over_forced": incremental_mass,
            "cold_attention_mass_recall": (
                incremental_mass / cold_mass if cold_mass > 1e-12 else None
            ),
        }

    known_evidence: dict[str, Any] | None = None
    if known_evidence_span is not None:
        evidence_start, evidence_stop = known_evidence_span
        evidence_pages = {
            page.page_id
            for page in result.pages
            if page.start < evidence_stop and page.stop > evidence_start
        }

        def page_recall(selection: Any | None) -> float | None:
            if not evidence_pages:
                return None
            if selection is None:
                return 0.0
            return len(evidence_pages.intersection(selection.page_ids)) / len(evidence_pages)

        known_evidence = {
            "token_span": [evidence_start, evidence_stop],
            "page_ids": sorted(evidence_pages),
            "page_recall": {
                "forced": page_recall(result.forced_selection),
                "hashed": page_recall(result.hashed_selection),
                "minmax": page_recall(result.minmax_selection),
                "oracle": page_recall(result.oracle_selection),
            },
        }

    return {
        "forced_selection": _jsonable(result.forced_selection),
        "hashed_selection": _jsonable(result.hashed_selection),
        "minmax_selection": _jsonable(result.minmax_selection),
        "oracle_selection": _jsonable(result.oracle_selection),
        "forced_metrics": _jsonable(result.forced_metrics),
        "hashed_metrics": _jsonable(result.hashed_metrics),
        "minmax_metrics": _jsonable(result.minmax_metrics),
        "oracle_metrics": _jsonable(result.oracle_metrics),
        "routing_metadata": {
            "bits_per_token_per_indexed_head": (
                result.config.hash_bits * result.config.hash_tables
            ),
            "query_code_shape": list(result.query_routing_codes.shape),
            "key_code_shape": list(result.key_routing_codes.shape),
            "query_code_sha256": _tensor_sha256(result.query_routing_codes),
            "key_code_sha256": _tensor_sha256(result.key_routing_codes),
            "minmax_float32_descriptor_if_materialized": {
                "bytes_per_page_per_indexed_head": 2 * result.head_dim * 4,
                "nominal_bytes_per_token_per_indexed_head": (
                    2 * result.head_dim * 4 / result.config.page_size
                ),
            },
        },
        "attribution": {
            "forced_attention_mass": forced_mass,
            "hashed": attribution(result.hashed_selection),
            "minmax": attribution(result.minmax_selection),
            "oracle": attribution(result.oracle_selection),
        },
        "known_evidence": known_evidence,
        "exact_attention_top": [
            {"token": int(index), "weight": float(weight)}
            for index, weight in zip(top_indices.tolist(), top_values.tolist(), strict=True)
        ],
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not records:
        return {"evaluations": 0}
    hashed = [record["result"]["hashed_metrics"] for record in records]
    minmax = [record["result"]["minmax_metrics"] for record in records]
    oracle = [record["result"]["oracle_metrics"] for record in records]
    forced = [
        record["result"]["forced_metrics"]
        for record in records
        if record["result"]["forced_metrics"] is not None
    ]

    def mean(items: list[dict[str, Any]], key: str) -> float:
        return sum(float(item[key]) for item in items) / len(items)

    def optional_mean(values: list[float | None]) -> float | None:
        present = [float(value) for value in values if value is not None]
        return sum(present) / len(present) if present else None

    summary: dict[str, float | int | None] = {
        "evaluations": len(records),
        "hashed_top_k_recall_mean": mean(hashed, "top_k_recall"),
        "hashed_retained_attention_mass_mean": mean(hashed, "retained_attention_mass"),
        "hashed_relative_output_l2_mean": mean(hashed, "relative_output_l2"),
        "minmax_top_k_recall_mean": mean(minmax, "top_k_recall"),
        "minmax_retained_attention_mass_mean": mean(minmax, "retained_attention_mass"),
        "minmax_relative_output_l2_mean": mean(minmax, "relative_output_l2"),
        "oracle_top_k_recall_mean": mean(oracle, "top_k_recall"),
        "oracle_retained_attention_mass_mean": mean(oracle, "retained_attention_mass"),
        "oracle_relative_output_l2_mean": mean(oracle, "relative_output_l2"),
        "candidate_token_fraction_mean": mean(hashed, "candidate_token_fraction"),
        "forced_top_k_recall_mean": mean(forced, "top_k_recall") if forced else None,
        "forced_retained_attention_mass_mean": (
            mean(forced, "retained_attention_mass") if forced else None
        ),
        "forced_relative_output_l2_mean": (
            mean(forced, "relative_output_l2") if forced else None
        ),
        "hashed_incremental_attention_mass_over_forced_mean": optional_mean(
            [
                record["result"]["attribution"]["hashed"][
                    "incremental_attention_mass_over_forced"
                ]
                for record in records
            ]
        ),
        "hashed_cold_attention_mass_recall_mean": optional_mean(
            [
                record["result"]["attribution"]["hashed"]["cold_attention_mass_recall"]
                for record in records
            ]
        ),
        "minmax_incremental_attention_mass_over_forced_mean": optional_mean(
            [
                record["result"]["attribution"]["minmax"][
                    "incremental_attention_mass_over_forced"
                ]
                for record in records
            ]
        ),
        "minmax_cold_attention_mass_recall_mean": optional_mean(
            [
                record["result"]["attribution"]["minmax"]["cold_attention_mass_recall"]
                for record in records
            ]
        ),
        "oracle_incremental_attention_mass_over_forced_mean": optional_mean(
            [
                record["result"]["attribution"]["oracle"][
                    "incremental_attention_mass_over_forced"
                ]
                for record in records
            ]
        ),
        "oracle_cold_attention_mass_recall_mean": optional_mean(
            [
                record["result"]["attribution"]["oracle"]["cold_attention_mass_recall"]
                for record in records
            ]
        ),
    }
    for selector in ("forced", "hashed", "minmax", "oracle"):
        summary[f"known_evidence_{selector}_page_recall_mean"] = optional_mean(
            [
                record["result"]["known_evidence"]["page_recall"][selector]
                for record in records
                if record["result"]["known_evidence"] is not None
            ]
        )
    return summary


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only comparable configurations; never average a mixed sweep."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        config = record["config"]
        key = (
            config["page_size"],
            config["hash_bits"],
            config["hash_tables"],
            config["candidate_pages"],
            config["sink_pages"],
            config["recent_pages"],
            config["top_k"],
            config["seed"],
            record["requested_candidate_fraction"],
        )
        grouped.setdefault(key, []).append(record)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        (
            page_size,
            hash_bits,
            hash_tables,
            candidate_pages,
            sink_pages,
            recent_pages,
            top_k,
            seed,
            requested_fraction,
        ) = key
        groups.append(
            {
                "configuration": {
                    "page_size": page_size,
                    "hash_bits": hash_bits,
                    "hash_tables": hash_tables,
                    "candidate_pages": candidate_pages,
                    "sink_pages": sink_pages,
                    "recent_pages": recent_pages,
                    "top_k": top_k,
                    "seed": seed,
                    "requested_candidate_fraction": requested_fraction,
                },
                **_summarize_group(grouped[key]),
            }
        )
    return {"evaluations": len(records), "configuration_groups": groups}


def _context_case(
    tokenizer: Any, *, context_file: Path | None, target_tokens: int
) -> tuple[Tensor, tuple[int, int] | None]:
    if context_file is not None:
        text = context_file.expanduser().read_text(encoding="utf-8")
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if not token_ids:
            raise ValueError("context file tokenized to an empty sequence")
        return torch.tensor(token_ids[:target_tokens], dtype=torch.long), None

    prefix = tokenizer(_CONTEXT_PREFIX, add_special_tokens=True)["input_ids"]
    evidence_lead = tokenizer(_EVIDENCE_LEAD, add_special_tokens=False)["input_ids"]
    recovery_code = tokenizer(_RECOVERY_CODE, add_special_tokens=False)["input_ids"]
    evidence_tail = tokenizer(_EVIDENCE_TAIL, add_special_tokens=False)["input_ids"]
    evidence = evidence_lead + recovery_code + evidence_tail
    filler = tokenizer(_FILLER_BLOCK, add_special_tokens=False)["input_ids"]
    suffix = tokenizer(_QUERY_SUFFIX, add_special_tokens=False)["input_ids"]
    available = target_tokens - len(prefix) - len(evidence) - len(suffix)
    if available < 1:
        raise ValueError(
            f"target_tokens={target_tokens} is too small for the built-in evidence case; "
            f"use at least {len(prefix) + len(evidence) + len(suffix) + 1}"
        )
    repeats = math.ceil(available / len(filler))
    filler_tokens = (filler * repeats)[:available]
    minimum_evidence_start = len(prefix)
    maximum_evidence_start = len(prefix) + available
    desired_code_start = (target_tokens // 2 // 64) * 64
    desired_evidence_start = desired_code_start - len(evidence_lead)
    evidence_start = min(
        maximum_evidence_start,
        max(minimum_evidence_start, desired_evidence_start),
    )
    left_filler_count = evidence_start - len(prefix)
    token_ids = (
        prefix
        + filler_tokens[:left_filler_count]
        + evidence
        + filler_tokens[left_filler_count:]
        + suffix
    )
    recovery_start = evidence_start + len(evidence_lead)
    recovery_span = (recovery_start, recovery_start + len(recovery_code))
    return torch.tensor(token_ids, dtype=torch.long), recovery_span


def _context_ids(tokenizer: Any, *, context_file: Path | None, target_tokens: int) -> Tensor:
    """Compatibility wrapper used by focused context-builder tests."""

    return _context_case(
        tokenizer, context_file=context_file, target_tokens=target_tokens
    )[0]


def _validate_context_length(token_count: int, max_position_embeddings: int) -> None:
    if token_count > max_position_embeddings:
        raise ValueError(
            f"context has {token_count} tokens but the model contract supports at most "
            f"{max_position_embeddings}"
        )


def _resolve_layers(requested: tuple[int, ...], layer_count: int) -> tuple[int, ...]:
    resolved: list[int] = []
    for index in requested:
        normalized = index if index >= 0 else layer_count + index
        if normalized < 0 or normalized >= layer_count:
            raise ValueError(f"layer index {index} is outside a {layer_count}-layer model")
        if normalized not in resolved:
            resolved.append(normalized)
    return tuple(resolved)


def _query_positions(token_count: int, query_count: int) -> tuple[int, ...]:
    if query_count < 1:
        raise ValueError("query_count must be positive")
    first = max(0, token_count - query_count)
    return tuple(range(first, token_count))


def _capture_qkv(
    model: Any,
    *,
    input_ids: Tensor,
    layer_indices: tuple[int, ...],
    query_positions: tuple[int, ...],
) -> tuple[CapturedLayer, ...]:
    try:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("the hashed KV lab currently requires a Qwen3 Transformers model") from error

    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("the hashed KV lab expects model.model.layers")

    captured: dict[int, CapturedLayer] = {}
    handles: list[Any] = []

    def hook_for(layer_index: int):
        def capture(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError("Qwen attention hook did not receive hidden/position states")

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            queries = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            keys = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            values = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
            positions = torch.tensor(query_positions, device=queries.device)
            selected_queries = queries.index_select(2, positions)
            captured[layer_index] = CapturedLayer(
                layer_index=layer_index,
                query_positions=query_positions,
                queries=selected_queries[0].detach().to("cpu"),
                keys=keys[0].detach().to("cpu"),
                values=values[0].detach().to("cpu"),
            )

        return capture

    for layer_index in layer_indices:
        attention = layers[layer_index].self_attn
        handles.append(attention.register_forward_pre_hook(hook_for(layer_index), with_kwargs=True))

    device = model.get_input_embeddings().weight.device
    batched_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(batched_ids)
    try:
        with torch.inference_mode():
            backbone(input_ids=batched_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = set(layer_indices) - set(captured)
    if missing:
        raise RuntimeError(f"failed to capture Q/K/V for layers {sorted(missing)}")
    return tuple(captured[index] for index in layer_indices)


def _selected_query_heads(requested: tuple[int, ...], query_head_count: int) -> tuple[int, ...]:
    selected: list[int] = []
    for index in requested:
        normalized = index if index >= 0 else query_head_count + index
        if normalized < 0 or normalized >= query_head_count:
            raise ValueError(
                f"query head {index} is outside a model with {query_head_count} query heads"
            )
        if normalized not in selected:
            selected.append(normalized)
    return tuple(selected)


def _evaluate_captures(
    captures: tuple[CapturedLayer, ...],
    *,
    page_sizes: tuple[int, ...],
    hash_bits: tuple[int, ...],
    hash_tables: tuple[int, ...],
    candidate_fractions: tuple[float, ...],
    sink_pages: int,
    recent_pages: int,
    top_k: int,
    query_heads: tuple[int, ...],
    seed: int,
    known_evidence_span: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for capture in captures:
        query_head_count = int(capture.queries.shape[0])
        kv_head_count = int(capture.keys.shape[0])
        if query_head_count % kv_head_count:
            raise RuntimeError("query head count is not divisible by KV head count")
        groups = query_head_count // kv_head_count
        selected_heads = _selected_query_heads(query_heads, query_head_count)

        for query_slot, logical_position in enumerate(capture.query_positions):
            token_count = logical_position + 1
            for query_head in selected_heads:
                kv_head = query_head // groups
                query = capture.queries[query_head, query_slot].float()
                keys = capture.keys[kv_head, :token_count].float()
                values = capture.values[kv_head, :token_count].float()
                for page_size in page_sizes:
                    total_pages = math.ceil(token_count / page_size)
                    for fraction in candidate_fractions:
                        candidate_pages = max(
                            sink_pages + recent_pages,
                            math.ceil(total_pages * fraction),
                        )
                        candidate_pages = min(candidate_pages, total_pages)
                        for bits in hash_bits:
                            for tables in hash_tables:
                                config = HashedKVConfig(
                                    page_size=page_size,
                                    hash_bits=bits,
                                    hash_tables=tables,
                                    candidate_pages=candidate_pages,
                                    sink_pages=min(sink_pages, candidate_pages),
                                    recent_pages=min(
                                        recent_pages,
                                        max(0, candidate_pages - min(sink_pages, candidate_pages)),
                                    ),
                                    top_k=min(top_k, token_count),
                                    seed=seed,
                                )
                                started = time.perf_counter()
                                result = evaluate_hashed_kv(query, keys, values, config=config)
                                elapsed = time.perf_counter() - started
                                records.append(
                                    {
                                        "layer": capture.layer_index,
                                        "query_position": logical_position,
                                        "query_head": query_head,
                                        "kv_head": kv_head,
                                        "token_count": token_count,
                                        "requested_candidate_fraction": fraction,
                                        "evaluation_seconds": elapsed,
                                        "config": asdict(config),
                                        "result": _compact_result(
                                            result,
                                            known_evidence_span=known_evidence_span,
                                        ),
                                    }
                                )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture real Qwen Q/K/V and grade hashed KV page routing in shadow mode."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--target-tokens", type=int, default=1024)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--query-heads", type=int, nargs="+", default=list(DEFAULT_QUERY_HEADS))
    parser.add_argument("--page-size", type=int, nargs="+", default=[64])
    parser.add_argument("--hash-bits", type=int, nargs="+", choices=(32, 64), default=[32])
    parser.add_argument("--hash-tables", type=int, nargs="+", default=[1])
    parser.add_argument("--candidate-fraction", type=float, nargs="+", default=[0.25])
    parser.add_argument("--sink-pages", type=int, default=1)
    parser.add_argument("--recent-pages", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8419)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.target_tokens < 1:
        parser.error("--target-tokens must be positive")
    if args.query_count < 1:
        parser.error("--query-count must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.sink_pages < 0 or args.recent_pages < 0:
        parser.error("--sink-pages and --recent-pages must be non-negative")
    if args.seed < 0 or args.seed > torch.iinfo(torch.int64).max:
        parser.error("--seed must be a non-negative signed 64-bit integer")
    if any(value <= 0 for value in args.page_size):
        parser.error("--page-size values must be positive")
    if any(value <= 0 for value in args.hash_tables):
        parser.error("--hash-tables values must be positive")
    if any(value <= 0 or value > 1 for value in args.candidate_fraction):
        parser.error("--candidate-fraction values must be in (0, 1]")

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit("Install molor[hf] for the hashed KV lab") from error

    model_id = _local_or_hub_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=not args.allow_download,
    )
    input_ids, known_evidence_span = _context_case(
        tokenizer,
        context_file=args.context_file,
        target_tokens=args.target_tokens,
    )
    source_config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=not args.allow_download,
    )
    max_position_embeddings = int(source_config.max_position_embeddings)
    try:
        _validate_context_length(int(input_ids.numel()), max_position_embeddings)
    except ValueError as error:
        parser.error(str(error))
    context_digest = hashlib.sha256(
        json.dumps(input_ids.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    print(f"Loading {model_id} in bfloat16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=not args.allow_download,
    )
    model.eval()
    layer_count = len(model.model.layers)
    layers = _resolve_layers(tuple(args.layers), layer_count)
    positions = _query_positions(int(input_ids.numel()), args.query_count)

    capture_started = time.perf_counter()
    captures = _capture_qkv(
        model,
        input_ids=input_ids,
        layer_indices=layers,
        query_positions=positions,
    )
    capture_seconds = time.perf_counter() - capture_started
    device = str(model.get_input_embeddings().weight.device)
    attention_implementation = str(model.config._attn_implementation)
    model_config = {
        "layers": int(model.config.num_hidden_layers),
        "query_heads": int(model.config.num_attention_heads),
        "kv_heads": int(model.config.num_key_value_heads),
        "head_dim": int(model.config.head_dim),
        "max_position_embeddings": int(model.config.max_position_embeddings),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = _evaluate_captures(
        captures,
        page_sizes=tuple(args.page_size),
        hash_bits=tuple(args.hash_bits),
        hash_tables=tuple(args.hash_tables),
        candidate_fractions=tuple(args.candidate_fraction),
        sink_pages=args.sink_pages,
        recent_pages=args.recent_pages,
        top_k=args.top_k,
        query_heads=tuple(args.query_heads),
        seed=args.seed,
        known_evidence_span=known_evidence_span,
    )
    summary = _summarize_records(records)
    report = {
        "schema_version": "molor.hashed-kv-shadow-report.v1",
        "status": "development-smoke-only; no live cache or sparse inference",
        "promotion_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_id,
        "model_config": model_config,
        "context_tokens": int(input_ids.numel()),
        "context_token_sha256": context_digest,
        "known_evidence_token_span": (
            list(known_evidence_span) if known_evidence_span is not None else None
        ),
        "layers": list(layers),
        "query_positions": list(positions),
        "capture_seconds": capture_seconds,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "attention_implementation": attention_implementation,
            "dtype": "bfloat16",
            "seed": args.seed,
        },
        "summary": summary,
        "records": records,
    }

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_directory / f"hashed-kv-shadow-{input_ids.numel()}t-{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Captured {len(captures)} layers in {capture_seconds:.2f}s; "
        f"wrote {len(records)} shadow evaluations to {output_path}",
        flush=True,
    )
    for group in summary["configuration_groups"]:
        config = group["configuration"]
        cold_recall = group["hashed_cold_attention_mass_recall_mean"]
        cold_recall_text = f"{cold_recall:.3f}" if cold_recall is not None else "n/a"
        print(
            "Configuration "
            f"page={config['page_size']}, bits={config['hash_bits']}, "
            f"tables={config['hash_tables']}, pages={config['candidate_pages']}: "
            f"hash recall={group['hashed_top_k_recall_mean']:.3f}, "
            f"mass={group['hashed_retained_attention_mass_mean']:.3f}, "
            f"cold recall={cold_recall_text}; "
            f"min/max recall={group['minmax_top_k_recall_mean']:.3f}, "
            f"mass={group['minmax_retained_attention_mass_mean']:.3f}",
            flush=True,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
