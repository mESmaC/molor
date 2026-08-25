"""Grade literal A/T/X/Y residual codebooks on held-out real Qwen K/V.

This is an offline payload experiment.  Four 36-entry additive codebooks create
one literal ATXY tuple per 128-dimensional K or V vector.  Codes never select
pages, alias payloads, or alter live inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Literal

import torch
from torch import Tensor

from molor.experimental.atxy_kv_lab import (
    CaseCapture,
    _attention_comparison,
    _context_tokens,
    _load_dataset_manifest,
    _reconstruction_metrics,
    _sample_tokens,
    _sha256_file,
    _stack_payload,
    _summarize_attention,
)
from molor.experimental.atxy_vq import (
    ATXY_VQ_CODEBOOK_COUNT,
    ATXY_VQ_CODEBOOK_SIZE,
)
from molor.experimental.atxy_vq_stack import AtxyResidualVQStack
from molor.experimental.hashed_kv_lab import (
    _capture_qkv,
    _local_or_hub_model,
    _query_positions,
    _resolve_layers,
    _selected_query_heads,
    _tensor_sha256,
    _validate_context_length,
)


DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/atxy-vq-v1")
_SCHEMA_VERSION = "molor.atxy-vq-shadow-report.v2"
Grouping = Literal["global", "layer", "layer-head"]


def _fit_grouped_codecs(
    calibration: tuple[Tensor, ...],
    *,
    grouping: Grouping,
    seed: int,
    max_iterations: int,
    codebook_dtype: torch.dtype,
    tuple_count: int,
) -> tuple[AtxyResidualVQStack, ...]:
    """Fit one global codec or one independent codec per transformer layer."""

    if not calibration:
        raise ValueError("calibration must not be empty")
    shape = calibration[0].shape
    if any(
        value.ndim != 4 or value.shape[0] != shape[0] or value.shape[2:] != shape[2:]
        for value in calibration
    ):
        raise ValueError("calibration payloads must share layer, head, and channel axes")
    if grouping == "global":
        samples = torch.cat(tuple(value.reshape(-1, value.shape[-1]) for value in calibration))
        return (
            AtxyResidualVQStack.fit(
                samples,
                tuple_count=tuple_count,
                seed=seed,
                max_iterations=max_iterations,
                codebook_dtype=codebook_dtype,
            ),
        )
    if grouping == "layer-head":
        codecs = []
        for layer in range(shape[0]):
            for head in range(shape[2]):
                samples = torch.cat(tuple(value[layer, :, head] for value in calibration))
                codecs.append(
                    AtxyResidualVQStack.fit(
                        samples,
                        tuple_count=tuple_count,
                        seed=seed + layer * shape[2] + head,
                        max_iterations=max_iterations,
                        codebook_dtype=codebook_dtype,
                    )
                )
        return tuple(codecs)
    if grouping != "layer":
        raise ValueError("grouping must be global, layer, or layer-head")
    codecs: list[AtxyResidualVQStack] = []
    for layer in range(shape[0]):
        samples = torch.cat(
            tuple(value[layer].reshape(-1, value.shape[-1]) for value in calibration)
        )
        codecs.append(
            AtxyResidualVQStack.fit(
                samples,
                tuple_count=tuple_count,
                seed=seed + layer,
                max_iterations=max_iterations,
                codebook_dtype=codebook_dtype,
            )
        )
    return tuple(codecs)


def _encode_grouped(
    value: Tensor,
    codecs: tuple[AtxyResidualVQStack, ...],
    *,
    grouping: Grouping,
) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
    """Encode a payload and retain the per-group sample counts needed to reshape it."""

    if value.ndim != 4:
        raise ValueError("value must have shape [layers, tokens, heads, channels]")
    layers, tokens, heads, channels = value.shape
    if grouping == "global":
        if len(codecs) != 1:
            raise ValueError("global grouping requires exactly one codec")
        flat = value.reshape(-1, channels)
        return (codecs[0].encode(flat),), (int(flat.shape[0]),)
    if grouping == "layer-head":
        if len(codecs) != layers * heads:
            raise ValueError("layer-head grouping requires one codec per layer and KV head")
        fields = tuple(
            codecs[layer * heads + head].encode(value[layer, :, head])
            for layer in range(layers)
            for head in range(heads)
        )
        return fields, tuple(tokens for _ in fields)
    if grouping != "layer" or len(codecs) != layers:
        raise ValueError("layer grouping requires one codec per layer")
    fields = tuple(
        codecs[layer].encode(value[layer].reshape(-1, channels)) for layer in range(layers)
    )
    return fields, tuple(tokens * heads for _ in range(layers))


def _decode_grouped(
    fields: tuple[Tensor, ...],
    codecs: tuple[AtxyResidualVQStack, ...],
    *,
    grouping: Grouping,
    shape: torch.Size,
    tuple_depth: int,
) -> Tensor:
    layers, tokens, heads, channels = shape
    if grouping == "global":
        if len(fields) != 1 or len(codecs) != 1:
            raise ValueError("global grouping requires one field tensor and codec")
        return codecs[0].decode(fields[0], tuple_depth=tuple_depth).reshape(shape).contiguous()
    if grouping == "layer-head":
        if len(fields) != layers * heads or len(codecs) != layers * heads:
            raise ValueError("layer-head grouping requires one field tensor and codec per stratum")
        reconstructed_layers: list[Tensor] = []
        for layer in range(layers):
            reconstructed_heads = tuple(
                codecs[layer * heads + head]
                .decode(fields[layer * heads + head], tuple_depth=tuple_depth)
                .reshape(tokens, channels)
                for head in range(heads)
            )
            reconstructed_layers.append(torch.stack(reconstructed_heads, dim=1))
        return torch.stack(reconstructed_layers, dim=0).contiguous()
    if grouping != "layer" or len(fields) != layers or len(codecs) != layers:
        raise ValueError("layer grouping requires one field tensor and codec per layer")
    reconstructed = tuple(
        codecs[layer]
        .decode(fields[layer], tuple_depth=tuple_depth)
        .reshape(tokens, heads, channels)
        for layer in range(layers)
    )
    return torch.stack(reconstructed, dim=0).contiguous()


def _progressive_storage(
    fields: tuple[Tensor, ...],
    codecs: tuple[AtxyResidualVQStack, ...],
    *,
    tuple_depth: int,
    native_bytes: int,
) -> dict[str, float | int]:
    if not codecs or not 1 <= tuple_depth <= codecs[0].tuple_count:
        raise ValueError("tuple_depth must be available in every codec")
    if any(codec.tuple_count != codecs[0].tuple_count for codec in codecs):
        raise ValueError("all grouped codecs must have the same tuple count")
    sample_count = sum(int(value.shape[0]) for value in fields)
    full_codebook_bytes = sum(codec.codebook_bytes for codec in codecs)
    codebook_bytes = full_codebook_bytes * tuple_depth // codecs[0].tuple_count
    field_bytes = sample_count * tuple_depth * ATXY_VQ_CODEBOOK_COUNT
    total_bytes = codebook_bytes + field_bytes
    return {
        "native_bytes": native_bytes,
        "codebook_bytes": codebook_bytes,
        "field_bytes": field_bytes,
        "total_bytes": total_bytes,
        "compression_ratio": native_bytes / total_bytes,
        "bits_per_vector": tuple_depth * ATXY_VQ_CODEBOOK_COUNT * 8,
        "information_bits_per_vector": (tuple_depth * ATXY_VQ_CODEBOOK_COUNT * math.log2(36)),
    }


def _codec_descriptor(codecs: tuple[AtxyResidualVQStack, ...]) -> dict[str, Any]:
    return {
        "codec_count": len(codecs),
        "tuples_per_codec": codecs[0].tuple_count,
        "channels": codecs[0].channels,
        "codebook_dtype": str(codecs[0].codebook_dtype),
        "codebook_bytes": sum(codec.codebook_bytes for codec in codecs),
        "codebook_sha256": [_tensor_sha256(codec.codebooks) for codec in codecs],
        "seeds": [codec.seed for codec in codecs],
    }


def _attention_records(
    capture: CaseCapture,
    *,
    arm: str,
    exact_keys: Tensor,
    exact_values: Tensor,
    candidate_keys: Tensor,
    candidate_values: Tensor,
    query_heads: tuple[int, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for layer_slot, layer_capture in enumerate(capture.layers):
        query_head_count = int(layer_capture.queries.shape[0])
        kv_head_count = int(layer_capture.keys.shape[0])
        if query_head_count % kv_head_count:
            raise RuntimeError("query head count is not divisible by KV head count")
        groups = query_head_count // kv_head_count
        selected_heads = _selected_query_heads(query_heads, query_head_count)
        for query_slot, logical_position in enumerate(layer_capture.query_positions):
            token_count = logical_position + 1
            for query_head in selected_heads:
                kv_head = query_head // groups
                query = layer_capture.queries[query_head, query_slot]
                exact_key_head = exact_keys[layer_slot, :token_count, kv_head]
                exact_value_head = exact_values[layer_slot, :token_count, kv_head]
                candidate_key_head = candidate_keys[layer_slot, :token_count, kv_head]
                candidate_value_head = candidate_values[layer_slot, :token_count, kv_head]
                variants = {
                    "K-only": (candidate_key_head, exact_value_head),
                    "V-only": (exact_key_head, candidate_value_head),
                    "K+V": (candidate_key_head, candidate_value_head),
                }
                for variant, (keys, values) in variants.items():
                    records.append(
                        {
                            "case_id": capture.case.case_id,
                            "arm": arm,
                            "variant": variant,
                            "layer": layer_capture.layer_index,
                            "query_position": logical_position,
                            "query_head": query_head,
                            "kv_head": kv_head,
                            "metrics": _attention_comparison(
                                query,
                                exact_key_head,
                                exact_value_head,
                                keys,
                                values,
                                top_k=top_k,
                            ),
                        }
                    )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit literal four-codebook ATXY VQ on calibration KV and grade held-out attention."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument(
        "--grouping",
        choices=("global", "layer", "layer-head"),
        nargs="+",
        default=["global"],
    )
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--query-heads", type=int, nargs="+", default=[0, 5, 10, 15])
    parser.add_argument("--fit-token-limit", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--tuple-count", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8419)
    parser.add_argument("--codebook-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if (
        args.query_count < 1
        or args.fit_token_limit < 1
        or args.max_iterations < 1
        or args.tuple_count < 1
        or args.top_k < 1
    ):
        parser.error("query count, fit limit, iterations, tuple count, and top-k must be positive")
    if args.seed < 0 or args.seed > torch.iinfo(torch.int64).max - 10_000:
        parser.error("--seed must be a non-negative signed 64-bit integer with grouping headroom")
    groupings = tuple(dict.fromkeys(args.grouping))
    codebook_dtype = torch.float32 if args.codebook_dtype == "float32" else torch.bfloat16

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit("Install molor[hf] for the ATXY VQ lab") from error

    try:
        cases = _load_dataset_manifest(args.dataset_manifest)
    except ValueError as error:
        parser.error(str(error))
    model_id = _local_or_hub_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=not args.allow_download)
    source_config = AutoConfig.from_pretrained(model_id, local_files_only=not args.allow_download)
    layer_count = int(source_config.num_hidden_layers)
    requested_layers = tuple(args.layers) if args.layers else tuple(range(layer_count))
    try:
        layers = _resolve_layers(requested_layers, layer_count)
    except ValueError as error:
        parser.error(str(error))

    case_tokens: dict[str, Tensor] = {}
    for case in cases:
        try:
            tokens = _context_tokens(tokenizer, case)
            _validate_context_length(
                int(tokens.numel()), int(source_config.max_position_embeddings)
            )
        except ValueError as error:
            parser.error(str(error))
        case_tokens[case.case_id] = tokens

    print(f"Loading {model_id} in bfloat16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=not args.allow_download,
    )
    model.eval()
    device = str(model.get_input_embeddings().weight.device)
    attention_implementation = str(model.config._attn_implementation)
    captures: dict[str, CaseCapture] = {}
    for case in cases:
        tokens = case_tokens[case.case_id]
        print(f"Capturing {case.split} case {case.case_id} ({tokens.numel():,} tokens)", flush=True)
        started = time.perf_counter()
        captured_layers = _capture_qkv(
            model,
            input_ids=tokens,
            layer_indices=layers,
            query_positions=_query_positions(int(tokens.numel()), args.query_count),
        )
        captures[case.case_id] = CaseCapture(
            case=case,
            token_count=int(tokens.numel()),
            token_sha256=hashlib.sha256(
                json.dumps(tokens.tolist(), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            capture_seconds=time.perf_counter() - started,
            layers=captured_layers,
        )
    del model, case_tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    calibration_captures = tuple(
        captures[case.case_id] for case in cases if case.split == "calibration"
    )
    heldout_captures = tuple(captures[case.case_id] for case in cases if case.split == "heldout")
    calibration_keys = tuple(
        _sample_tokens(_stack_payload(capture.layers, "keys"), args.fit_token_limit)
        for capture in calibration_captures
    )
    calibration_values = tuple(
        _sample_tokens(_stack_payload(capture.layers, "values"), args.fit_token_limit)
        for capture in calibration_captures
    )

    fitted: list[dict[str, Any]] = []
    for grouping_slot, grouping in enumerate(groupings):
        print(f"Fitting {grouping} literal ATXY codebooks", flush=True)
        started = time.perf_counter()
        key_codecs = _fit_grouped_codecs(
            calibration_keys,
            grouping=grouping,
            seed=args.seed + grouping_slot * 1_000,
            max_iterations=args.max_iterations,
            codebook_dtype=codebook_dtype,
            tuple_count=args.tuple_count,
        )
        value_codecs = _fit_grouped_codecs(
            calibration_values,
            grouping=grouping,
            seed=args.seed + grouping_slot * 1_000 + 500,
            max_iterations=args.max_iterations,
            codebook_dtype=codebook_dtype,
            tuple_count=args.tuple_count,
        )
        fitted.append(
            {
                "grouping": grouping,
                "key_codecs": key_codecs,
                "value_codecs": value_codecs,
                "fit_seconds": time.perf_counter() - started,
            }
        )
    del calibration_keys, calibration_values

    arm_records: list[dict[str, Any]] = []
    attention_records: list[dict[str, Any]] = []
    for capture in heldout_captures:
        exact_keys = _stack_payload(capture.layers, "keys")
        exact_values = _stack_payload(capture.layers, "values")
        native_bytes = exact_keys.numel() * exact_keys.element_size()
        native_bytes += exact_values.numel() * exact_values.element_size()
        for fitted_arm in fitted:
            grouping = fitted_arm["grouping"]
            key_codecs = fitted_arm["key_codecs"]
            value_codecs = fitted_arm["value_codecs"]
            encode_started = time.perf_counter()
            key_fields, _ = _encode_grouped(exact_keys, key_codecs, grouping=grouping)
            value_fields, _ = _encode_grouped(exact_values, value_codecs, grouping=grouping)
            encode_seconds = time.perf_counter() - encode_started
            for tuple_depth in range(1, args.tuple_count + 1):
                decode_started = time.perf_counter()
                reconstructed_keys = _decode_grouped(
                    key_fields,
                    key_codecs,
                    grouping=grouping,
                    shape=exact_keys.shape,
                    tuple_depth=tuple_depth,
                )
                reconstructed_values = _decode_grouped(
                    value_fields,
                    value_codecs,
                    grouping=grouping,
                    shape=exact_values.shape,
                    tuple_depth=tuple_depth,
                )
                decode_seconds = time.perf_counter() - decode_started
                key_storage = _progressive_storage(
                    key_fields,
                    key_codecs,
                    tuple_depth=tuple_depth,
                    native_bytes=exact_keys.numel() * exact_keys.element_size(),
                )
                value_storage = _progressive_storage(
                    value_fields,
                    value_codecs,
                    tuple_depth=tuple_depth,
                    native_bytes=exact_values.numel() * exact_values.element_size(),
                )
                total_bytes = int(key_storage["total_bytes"]) + int(value_storage["total_bytes"])
                arm = f"rvq-{grouping}-tuples{tuple_depth}"
                arm_records.append(
                    {
                        "case_id": capture.case.case_id,
                        "arm": arm,
                        "grouping": grouping,
                        "tuple_depth": tuple_depth,
                        "fit_seconds": fitted_arm["fit_seconds"],
                        "encode_seconds": encode_seconds,
                        "decode_seconds": decode_seconds,
                        "key_reconstruction": _reconstruction_metrics(
                            exact_keys, reconstructed_keys
                        ),
                        "value_reconstruction": _reconstruction_metrics(
                            exact_values, reconstructed_values
                        ),
                        "storage": {
                            "native_bytes": native_bytes,
                            "codebook_bytes": int(key_storage["codebook_bytes"])
                            + int(value_storage["codebook_bytes"]),
                            "field_bytes": int(key_storage["field_bytes"])
                            + int(value_storage["field_bytes"]),
                            "total_bytes": total_bytes,
                            "compression_ratio": native_bytes / total_bytes,
                            "stored_bits_per_k_or_v_vector": (
                                tuple_depth * ATXY_VQ_CODEBOOK_COUNT * 8
                            ),
                            "information_bits_per_k_or_v_vector": key_storage[
                                "information_bits_per_vector"
                            ],
                        },
                    }
                )
                attention_records.extend(
                    _attention_records(
                        capture,
                        arm=arm,
                        exact_keys=exact_keys,
                        exact_values=exact_values,
                        candidate_keys=reconstructed_keys,
                        candidate_values=reconstructed_values,
                        query_heads=tuple(args.query_heads),
                        top_k=args.top_k,
                    )
                )
                del reconstructed_keys, reconstructed_values

    config = {
        "codec_version": "molor.atxy-rvq-stack.v1",
        "equation": "K_or_V_hat = sum_over_tuples(C_A[a] + C_T[t] + C_X[x] + C_Y[y])",
        "codebooks_per_tuple": ATXY_VQ_CODEBOOK_COUNT,
        "entries_per_codebook": ATXY_VQ_CODEBOOK_SIZE,
        "tuple_count": args.tuple_count,
        "codebook_dtype": str(codebook_dtype),
        "groupings": list(groupings),
        "fit_token_limit_per_calibration_case": args.fit_token_limit,
        "max_iterations": args.max_iterations,
        "query_count": args.query_count,
        "query_heads": list(args.query_heads),
        "top_k": args.top_k,
        "seed": args.seed,
        "layers": list(layers),
    }
    report = {
        "schema_version": _SCHEMA_VERSION,
        "status": "development-shadow-only; no live cache, routing, or storage claim",
        "promotion_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_id,
        "model_config": {
            "layers": int(source_config.num_hidden_layers),
            "query_heads": int(source_config.num_attention_heads),
            "kv_heads": int(source_config.num_key_value_heads),
            "head_dim": int(source_config.head_dim),
            "max_position_embeddings": int(source_config.max_position_embeddings),
        },
        "dataset_manifest": str(args.dataset_manifest.expanduser().resolve()),
        "dataset_manifest_sha256": _sha256_file(args.dataset_manifest.expanduser().resolve()),
        "cases": [
            {
                "id": capture.case.case_id,
                "split": capture.case.split,
                "source": str(capture.case.path),
                "source_sha256": capture.case.source_sha256,
                "token_count": capture.token_count,
                "token_sha256": capture.token_sha256,
                "capture_seconds": capture.capture_seconds,
            }
            for capture in captures.values()
        ],
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "capture_device": device,
            "codec_device": "cpu",
            "attention_implementation": attention_implementation,
        },
        "codecs": [
            {
                "grouping": item["grouping"],
                "fit_seconds": item["fit_seconds"],
                "key": _codec_descriptor(item["key_codecs"]),
                "value": _codec_descriptor(item["value_codecs"]),
            }
            for item in fitted
        ],
        "arm_records": arm_records,
        "attention_summary": _summarize_attention(attention_records),
        "attention_records": attention_records,
    }
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"atxy-vq-shadow-{time.strftime('%Y%m%d-%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote shadow report to {output_path}", flush=True)
    for summary in report["attention_summary"]:
        if summary["variant"] == "K+V":
            print(
                f"{summary['arm']}: KL={summary['attention_kl_mean']:.5f}, "
                f"top1={summary['attention_top1_agreement']:.3f}, "
                f"output L2={summary['output_relative_l2_mean']:.3f}",
                flush=True,
            )


if __name__ == "__main__":  # pragma: no cover
    main()
