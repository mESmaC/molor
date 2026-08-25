"""Run a held-out, shadow-only ATXY tensor-codec experiment on real Qwen KV.

The continuous Tucker codec in this lab is a new MoLoR experiment inspired by
ATXY's four-axis organization.  It is not the exact CDAS address representation,
and this lab does not alter a live cache or combine compression with routing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from torch.nn import functional as F

from molor.experimental.atxy_kv import AtxyRanks, AtxyShape, AtxyTuckerCodec
from molor.experimental.hashed_kv_lab import (
    CapturedLayer,
    _capture_qkv,
    _local_or_hub_model,
    _query_positions,
    _resolve_layers,
    _selected_query_heads,
    _tensor_sha256,
    _validate_context_length,
)


DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/atxy-kv-v1")
_SCHEMA_VERSION = "molor.atxy-kv-shadow-report.v1"
_DATASET_SCHEMA = "molor.atxy-kv-dataset.v1"


@dataclass(frozen=True, slots=True)
class ContextCase:
    """One immutable calibration or held-out context source."""

    case_id: str
    split: Literal["calibration", "heldout"]
    path: Path
    source_sha256: str
    token_cap: int


@dataclass(frozen=True, slots=True)
class CaseCapture:
    """Captured dense truth and provenance for one context case."""

    case: ContextCase
    token_count: int
    token_sha256: str
    capture_seconds: float
    layers: tuple[CapturedLayer, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_dataset_manifest(path: Path) -> tuple[ContextCase, ...]:
    """Load, checksum, and leakage-check a context manifest."""

    manifest_path = path.expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read dataset manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != _DATASET_SCHEMA:
        raise ValueError(f"dataset manifest schema_version must be {_DATASET_SCHEMA}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("dataset manifest cases must be a non-empty list")

    cases: list[ContextCase] = []
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"dataset case {index} must be an object")
        case_id = raw.get("id")
        split = raw.get("split")
        source = raw.get("path")
        expected_digest = raw.get("sha256")
        token_cap = raw.get("token_cap")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"dataset case {index} id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"duplicate dataset case id: {case_id}")
        if split not in ("calibration", "heldout"):
            raise ValueError(f"dataset case {case_id} split must be calibration or heldout")
        if not isinstance(source, str) or not source:
            raise ValueError(f"dataset case {case_id} path must be a non-empty string")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_digest)
        ):
            raise ValueError(f"dataset case {case_id} sha256 must be 64 hexadecimal characters")
        token_cap = _positive_int(token_cap, f"dataset case {case_id} token_cap")
        source_path = (manifest_path.parent / source).resolve()
        if not source_path.is_file():
            raise ValueError(f"dataset case {case_id} source does not exist: {source_path}")
        actual_digest = _sha256_file(source_path)
        if actual_digest != expected_digest.lower():
            raise ValueError(
                f"dataset case {case_id} sha256 mismatch: expected "
                f"{expected_digest.lower()}, got {actual_digest}"
            )
        if actual_digest in seen_digests:
            raise ValueError(f"duplicate dataset content digest: {actual_digest}")
        seen_ids.add(case_id)
        seen_digests.add(actual_digest)
        cases.append(
            ContextCase(
                case_id=case_id,
                split=split,
                path=source_path,
                source_sha256=actual_digest,
                token_cap=token_cap,
            )
        )

    splits = {case.split for case in cases}
    if splits != {"calibration", "heldout"}:
        raise ValueError("dataset manifest must contain calibration and heldout cases")
    return tuple(cases)


def _context_tokens(tokenizer: Any, case: ContextCase) -> Tensor:
    """Tokenize only as much source text as is needed for the declared cap."""

    chunks: list[str] = []
    token_ids: list[int] = []
    with case.path.open("r", encoding="utf-8") as handle:
        while chunk := handle.read(64 * 1024):
            chunks.append(chunk)
            token_ids = tokenizer("".join(chunks), add_special_tokens=True)["input_ids"]
            if len(token_ids) >= case.token_cap:
                break
    if len(token_ids) < case.token_cap:
        raise ValueError(
            f"dataset case {case.case_id} has only {len(token_ids)} tokens, below its "
            f"declared token_cap={case.token_cap}"
        )
    return torch.tensor(token_ids[: case.token_cap], dtype=torch.long)


def _stack_payload(captures: tuple[CapturedLayer, ...], kind: Literal["keys", "values"]) -> Tensor:
    """Stack per-layer ``[X,T,Y]`` captures as one ``[A,T,X,Y]`` payload."""

    if not captures:
        raise ValueError("captures must not be empty")
    tensors = tuple(getattr(capture, kind) for capture in captures)
    expected = tensors[0].shape
    if any(tensor.shape != expected for tensor in tensors):
        raise ValueError(f"captured {kind} must have identical per-layer shapes")
    return torch.stack(tensors, dim=0).permute(0, 2, 1, 3).contiguous()


def _sample_tokens(value: Tensor, maximum: int) -> Tensor:
    maximum = _positive_int(maximum, "fit token limit")
    token_count = int(value.shape[1])
    if token_count <= maximum:
        return value
    indices = torch.div(
        torch.arange(maximum, dtype=torch.int64) * token_count,
        maximum,
        rounding_mode="floor",
    )
    return value.index_select(1, indices).contiguous()


def _reconstruction_metrics(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("reconstruction tensors must have identical shapes")
    exact = reference.detach().to(device="cpu", dtype=torch.float32)
    reconstructed = candidate.detach().to(device="cpu", dtype=torch.float32)
    delta = reconstructed - exact
    absolute = delta.abs().reshape(-1)
    exact_flat = exact.reshape(-1)
    candidate_flat = reconstructed.reshape(-1)
    denominator = max(float(torch.linalg.vector_norm(exact_flat).item()), 1e-12)
    return {
        "rmse": float(delta.square().mean().sqrt().item()),
        "max_abs": float(absolute.max().item()),
        "p99_abs": float(torch.quantile(absolute, 0.99).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / denominator,
        "cosine_similarity": _cosine_similarity(exact_flat, candidate_flat),
    }


def _cosine_similarity(first: Tensor, second: Tensor) -> float:
    """Calculate a stable bounded cosine while accumulating in float64."""

    first_flat = first.reshape(-1).to(torch.float32)
    second_flat = second.reshape(-1).to(torch.float32)
    dot = torch.sum(first_flat * second_flat, dtype=torch.float64)
    first_norm = torch.sum(first_flat.square(), dtype=torch.float64).sqrt()
    second_norm = torch.sum(second_flat.square(), dtype=torch.float64).sqrt()
    denominator = float((first_norm * second_norm).item())
    if denominator <= 1e-24:
        return 1.0 if bool(torch.equal(first_flat, second_flat)) else 0.0
    return max(-1.0, min(1.0, float(dot.item()) / denominator))


def _attention_comparison(
    query: Tensor,
    exact_keys: Tensor,
    exact_values: Tensor,
    candidate_keys: Tensor,
    candidate_values: Tensor,
    *,
    top_k: int,
) -> dict[str, float | bool | int]:
    """Compare one exact-Q dense attention calculation with reconstructed K/V."""

    if exact_keys.shape != candidate_keys.shape or exact_values.shape != candidate_values.shape:
        raise ValueError("exact and candidate K/V shapes must match")
    if exact_keys.shape[0] != exact_values.shape[0]:
        raise ValueError("K and V token counts must match")
    q = query.detach().to(device="cpu", dtype=torch.float32)
    keys = exact_keys.detach().to(device="cpu", dtype=torch.float32)
    values = exact_values.detach().to(device="cpu", dtype=torch.float32)
    approximated_keys = candidate_keys.detach().to(device="cpu", dtype=torch.float32)
    approximated_values = candidate_values.detach().to(device="cpu", dtype=torch.float32)
    scale = 1.0 / math.sqrt(q.numel())
    exact_logits = torch.mv(keys, q) * scale
    candidate_logits = torch.mv(approximated_keys, q) * scale
    exact_log_probabilities = F.log_softmax(exact_logits, dim=0)
    candidate_log_probabilities = F.log_softmax(candidate_logits, dim=0)
    exact_probabilities = exact_log_probabilities.exp()
    candidate_probabilities = candidate_log_probabilities.exp()
    exact_output = torch.mv(values.mT, exact_probabilities)
    candidate_output = torch.mv(approximated_values.mT, candidate_probabilities)
    logit_delta = candidate_logits - exact_logits
    output_delta = candidate_output - exact_output
    output_norm = max(float(torch.linalg.vector_norm(exact_output).item()), 1e-12)
    selected_top_k = min(_positive_int(top_k, "top_k"), int(exact_logits.numel()))
    exact_top = set(torch.topk(exact_probabilities, selected_top_k).indices.tolist())
    candidate_top = set(torch.topk(candidate_probabilities, selected_top_k).indices.tolist())
    return {
        "attention_kl": float(
            torch.sum(
                exact_probabilities * (exact_log_probabilities - candidate_log_probabilities)
            ).item()
        ),
        "attention_top1_agreement": bool(
            exact_probabilities.argmax().item() == candidate_probabilities.argmax().item()
        ),
        "attention_top_k": selected_top_k,
        "attention_top_k_recall": len(exact_top.intersection(candidate_top)) / selected_top_k,
        "logit_rmse": float(logit_delta.square().mean().sqrt().item()),
        "logit_max_abs": float(logit_delta.abs().max().item()),
        "output_relative_l2": float(torch.linalg.vector_norm(output_delta).item()) / output_norm,
        "output_cosine_similarity": _cosine_similarity(exact_output, candidate_output),
    }


def _parse_rank(value: str) -> AtxyRanks:
    try:
        parts = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("rank must be layers,heads,channels") from error
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("rank must contain exactly layers,heads,channels")
    try:
        return AtxyRanks(*parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _codec_descriptor(codec: AtxyTuckerCodec) -> dict[str, Any]:
    bases = {
        "layer": codec.layer_basis,
        "head": codec.head_basis,
        "channel": codec.channel_basis,
    }
    return {
        "shape": asdict(codec.shape),
        "ranks": asdict(codec.ranks),
        "core_dtype": str(codec.core_dtype),
        "reconstruction_dtype": str(codec.reconstruction_dtype),
        "identity": codec.is_identity,
        "basis_bytes": codec.basis_bytes,
        "basis_sha256": {
            name: (_tensor_sha256(basis) if basis is not None else None)
            for name, basis in bases.items()
        },
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability).item())


def _summarize_attention(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((record["arm"], record["variant"]), []).append(record["metrics"])
    summary: list[dict[str, Any]] = []
    for (arm, variant), metrics in sorted(groups.items()):

        def values(name: str) -> list[float]:
            return [float(metric[name]) for metric in metrics]

        summary.append(
            {
                "arm": arm,
                "variant": variant,
                "evaluations": len(metrics),
                "attention_kl_mean": sum(values("attention_kl")) / len(metrics),
                "attention_kl_p99": _quantile(values("attention_kl"), 0.99),
                "attention_top1_agreement": sum(
                    bool(metric["attention_top1_agreement"]) for metric in metrics
                )
                / len(metrics),
                "attention_top_k_recall_mean": sum(values("attention_top_k_recall")) / len(metrics),
                "output_relative_l2_mean": sum(values("output_relative_l2")) / len(metrics),
                "output_relative_l2_p99": _quantile(values("output_relative_l2"), 0.99),
                "output_cosine_similarity_mean": sum(values("output_cosine_similarity"))
                / len(metrics),
            }
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit ATXY-shaped K/V codecs on calibration contexts and grade held-out Qwen attention."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--query-heads", type=int, nargs="+", default=[0, 5, 10, 15])
    parser.add_argument("--rank", type=_parse_rank, action="append")
    parser.add_argument(
        "--key-rank",
        type=_parse_rank,
        action="append",
        help="key-side rank for a paired K/V arm; repeat alongside --value-rank",
    )
    parser.add_argument(
        "--value-rank",
        type=_parse_rank,
        action="append",
        help="value-side rank for a paired K/V arm; repeat alongside --key-rank",
    )
    parser.add_argument("--fit-token-limit", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8419)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.query_count < 1 or args.fit_token_limit < 1 or args.top_k < 1:
        parser.error("--query-count, --fit-token-limit, and --top-k must be positive")
    if args.seed < 0 or args.seed > torch.iinfo(torch.int64).max:
        parser.error("--seed must be a non-negative signed 64-bit integer")
    torch.manual_seed(args.seed)

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit("Install molor[hf] for the ATXY KV lab") from error

    try:
        cases = _load_dataset_manifest(args.dataset_manifest)
    except ValueError as error:
        parser.error(str(error))
    model_id = _local_or_hub_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=not args.allow_download,
    )
    source_config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=not args.allow_download,
    )
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
        positions = _query_positions(int(tokens.numel()), args.query_count)
        print(
            f"Capturing {case.split} case {case.case_id} ({tokens.numel():,} tokens)",
            flush=True,
        )
        started = time.perf_counter()
        captured_layers = _capture_qkv(
            model,
            input_ids=tokens,
            layer_indices=layers,
            query_positions=positions,
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
    del model
    del case_tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    calibration = tuple(captures[case.case_id] for case in cases if case.split == "calibration")
    heldout = tuple(captures[case.case_id] for case in cases if case.split == "heldout")
    calibration_keys = tuple(
        _sample_tokens(_stack_payload(capture.layers, "keys"), args.fit_token_limit)
        for capture in calibration
    )
    calibration_values = tuple(
        _sample_tokens(_stack_payload(capture.layers, "values"), args.fit_token_limit)
        for capture in calibration
    )
    first = calibration_keys[0]
    shape = AtxyShape(int(first.shape[0]), int(first.shape[2]), int(first.shape[3]))
    if bool(args.key_rank) != bool(args.value_rank):
        parser.error("--key-rank and --value-rank must be supplied together")
    if args.key_rank and len(args.key_rank) != len(args.value_rank):
        parser.error("--key-rank and --value-rank must have the same number of values")
    common_ranks = args.rank or (
        []
        if args.key_rank
        else [
            AtxyRanks(shape.layers, shape.heads, max(1, shape.channels // 2)),
            AtxyRanks(max(1, shape.layers // 2), shape.heads, max(1, shape.channels // 2)),
        ]
    )
    requested_pairs = [(ranks, ranks) for ranks in common_ranks]
    requested_pairs.extend(zip(args.key_rank or (), args.value_rank or (), strict=True))
    for key_ranks, value_ranks in requested_pairs:
        for kind, ranks in (("key", key_ranks), ("value", value_ranks)):
            if (
                ranks.layers > shape.layers
                or ranks.heads > shape.heads
                or ranks.channels > shape.channels
            ):
                parser.error(f"{kind} rank {ranks} exceeds captured shape {shape}")

    arm_codecs: list[tuple[str, AtxyTuckerCodec, AtxyTuckerCodec, float]] = []
    identity_key = AtxyTuckerCodec.identity(shape, dtype=first.dtype)
    identity_value = AtxyTuckerCodec.identity(shape, dtype=calibration_values[0].dtype)
    arm_codecs.append(("identity", identity_key, identity_value, 0.0))
    for key_ranks, value_ranks in requested_pairs:
        print(
            "Fitting ATXY Tucker "
            f"K={key_ranks.layers}x{key_ranks.heads}x{key_ranks.channels}, "
            f"V={value_ranks.layers}x{value_ranks.heads}x{value_ranks.channels}",
            flush=True,
        )
        started = time.perf_counter()
        key_codec = AtxyTuckerCodec.fit(calibration_keys, key_ranks, core_dtype=torch.bfloat16)
        value_codec = AtxyTuckerCodec.fit(
            calibration_values, value_ranks, core_dtype=torch.bfloat16
        )
        if key_ranks == value_ranks:
            arm_name = f"tucker-{key_ranks.layers}x{key_ranks.heads}x{key_ranks.channels}"
        else:
            arm_name = (
                f"tucker-k{key_ranks.layers}x{key_ranks.heads}x{key_ranks.channels}"
                f"-v{value_ranks.layers}x{value_ranks.heads}x{value_ranks.channels}"
            )
        arm_codecs.append(
            (
                arm_name,
                key_codec,
                value_codec,
                time.perf_counter() - started,
            )
        )
    del calibration_keys
    del calibration_values

    arm_records: list[dict[str, Any]] = []
    attention_records: list[dict[str, Any]] = []
    for capture in heldout:
        exact_keys = _stack_payload(capture.layers, "keys")
        exact_values = _stack_payload(capture.layers, "values")
        for arm, key_codec, value_codec, fit_seconds in arm_codecs:
            encode_started = time.perf_counter()
            key_core = key_codec.encode(exact_keys)
            value_core = value_codec.encode(exact_values)
            encode_seconds = time.perf_counter() - encode_started
            decode_started = time.perf_counter()
            reconstructed_keys = key_codec.decode(key_core)
            reconstructed_values = value_codec.decode(value_core)
            decode_seconds = time.perf_counter() - decode_started
            key_storage = key_codec.storage_bytes(key_core)
            value_storage = value_codec.storage_bytes(value_core)
            native_bytes = key_storage.native_bytes + value_storage.native_bytes
            encoded_bytes = key_storage.total_bytes + value_storage.total_bytes
            arm_records.append(
                {
                    "case_id": capture.case.case_id,
                    "arm": arm,
                    "fit_seconds": fit_seconds,
                    "encode_seconds": encode_seconds,
                    "decode_seconds": decode_seconds,
                    "key_reconstruction": _reconstruction_metrics(exact_keys, reconstructed_keys),
                    "value_reconstruction": _reconstruction_metrics(
                        exact_values, reconstructed_values
                    ),
                    "storage": {
                        "native_bytes": native_bytes,
                        "core_bytes": key_storage.core_bytes + value_storage.core_bytes,
                        "basis_bytes": key_storage.basis_bytes + value_storage.basis_bytes,
                        "total_bytes": encoded_bytes,
                        "compression_ratio": native_bytes / encoded_bytes,
                        "bytes_per_token": encoded_bytes / capture.token_count,
                    },
                }
            )

            for layer_slot, layer_capture in enumerate(capture.layers):
                query_head_count = int(layer_capture.queries.shape[0])
                kv_head_count = int(layer_capture.keys.shape[0])
                if query_head_count % kv_head_count:
                    raise RuntimeError("query head count is not divisible by KV head count")
                groups = query_head_count // kv_head_count
                selected_heads = _selected_query_heads(tuple(args.query_heads), query_head_count)
                for query_slot, logical_position in enumerate(layer_capture.query_positions):
                    token_count = logical_position + 1
                    for query_head in selected_heads:
                        kv_head = query_head // groups
                        exact_key_head = exact_keys[layer_slot, :token_count, kv_head]
                        exact_value_head = exact_values[layer_slot, :token_count, kv_head]
                        reconstructed_key_head = reconstructed_keys[
                            layer_slot, :token_count, kv_head
                        ]
                        reconstructed_value_head = reconstructed_values[
                            layer_slot, :token_count, kv_head
                        ]
                        variants = {
                            "K-only": (reconstructed_key_head, exact_value_head),
                            "V-only": (exact_key_head, reconstructed_value_head),
                            "K+V": (reconstructed_key_head, reconstructed_value_head),
                        }
                        for variant, (candidate_keys, candidate_values) in variants.items():
                            attention_records.append(
                                {
                                    "case_id": capture.case.case_id,
                                    "arm": arm,
                                    "variant": variant,
                                    "layer": layer_capture.layer_index,
                                    "query_position": logical_position,
                                    "query_head": query_head,
                                    "kv_head": kv_head,
                                    "metrics": _attention_comparison(
                                        layer_capture.queries[query_head, query_slot],
                                        exact_key_head,
                                        exact_value_head,
                                        candidate_keys,
                                        candidate_values,
                                        top_k=args.top_k,
                                    ),
                                }
                            )
            del key_core, value_core, reconstructed_keys, reconstructed_values

    config = {
        "codec_version": "molor.atxy-tucker-kv.v1",
        "equation": "Tucker projection across layer/KV-head/channel; token axis retained",
        "core_dtype": "torch.bfloat16",
        "fit_token_limit_per_calibration_case": args.fit_token_limit,
        "query_count": args.query_count,
        "query_heads": list(args.query_heads),
        "top_k": args.top_k,
        "seed": args.seed,
        "layers": list(layers),
        "rank_pairs": [
            {"key": asdict(key_ranks), "value": asdict(value_ranks)}
            for key_ranks, value_ranks in requested_pairs
        ],
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
        "config_sha256": _canonical_sha256(config),
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
                "arm": arm,
                "key": _codec_descriptor(key_codec),
                "value": _codec_descriptor(value_codec),
            }
            for arm, key_codec, value_codec, _ in arm_codecs
        ],
        "arm_records": arm_records,
        "attention_summary": _summarize_attention(attention_records),
        "attention_records": attention_records,
    }
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_directory / f"atxy-kv-shadow-{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote shadow report to {output_path}", flush=True)
    for summary in report["attention_summary"]:
        if summary["variant"] != "K+V":
            continue
        print(
            f"{summary['arm']}: KL={summary['attention_kl_mean']:.6f}, "
            f"top1={summary['attention_top1_agreement']:.3f}, "
            f"output L2={summary['output_relative_l2_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
