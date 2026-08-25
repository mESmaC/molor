"""Grade a per-layer/per-head ATXY KV latent ceiling on real Qwen Q/K/V.

This is a fixed, shadow-only development experiment.  It fits local linear
bases on a calibration context, evaluates nested rank prefixes on a file-
disjoint held-out context, and never installs a cache or mutates model state.
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
from typing import Any

import torch
from torch import Tensor

from molor.benchmark.lock import lock_backend_identity
from molor.experimental.atxy_kv_lab import (
    ContextCase,
    _canonical_sha256,
    _context_tokens,
    _cosine_similarity,
    _load_dataset_manifest,
    _reconstruction_metrics,
    _sha256_file,
    _stack_payload,
)
from molor.experimental.atxy_local_latent import (
    AtxyLocalLatentCodec,
    GQA_QUERY_HEADS_PER_KV_HEAD,
)
from molor.experimental.atxy_pages import (
    ATXY_PAGES_PER_LAYER,
    AtxyPageAddress,
    token_capacity_per_layer,
)
from molor.experimental.hashed_kv_lab import (
    CapturedLayer,
    _capture_qkv,
    _query_positions,
    _resolve_layers,
    _tensor_sha256,
    _validate_context_length,
)


DEFAULT_CONTRACT = Path("experiments/atxy-local-latent-v1/contract.json")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/atxy-local-latent-v1")
_CONTRACT_SCHEMA = "molor.atxy-local-latent-contract.v1"
_REPORT_SCHEMA = "molor.atxy-local-latent-shadow-report.v1"
_REQUIRED_GATE_KEYS = {
    "min_projected_compression_ratio",
    "max_attention_kl_mean",
    "max_attention_kl_p99",
    "min_attention_top1_agreement",
    "min_attention_top_k_recall_mean",
    "max_output_relative_l2_global",
    "min_output_cosine_similarity_mean",
}


@dataclass(frozen=True, slots=True)
class LocalLatentContract:
    """Validated immutable inputs for one development shadow run."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    model_path: Path
    model_config_sha256: str
    model_content_fingerprint: str
    expected_model_config: dict[str, int]
    dataset_manifest: Path
    dataset_manifest_sha256: str
    layers: tuple[int, ...]
    calibration_tokens: int
    heldout_tokens: int
    heldout_query_positions: int
    key_ranks: tuple[int, ...]
    value_ranks: tuple[int, ...]
    joint_rank_pairs: tuple[tuple[int, int], ...]
    projected_context_tokens: int
    page_size: int
    top_k: int
    seed: int
    gates: dict[str, float]


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be 64 hexadecimal characters")
    return value.lower()


def _source_tree_identity(root: Path) -> dict[str, Any]:
    """Hash every MoLoR Python source dependency before and after the run."""

    resolved = root.expanduser().resolve()
    files: list[dict[str, Any]] = []
    paths = sorted(
        resolved.rglob("*.py"),
        key=lambda item: item.relative_to(resolved).as_posix(),
    )
    for path in paths:
        stat = path.stat()
        files.append(
            {
                "name": path.relative_to(resolved).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"source tree contains no Python files: {resolved}")
    return {
        "root": str(resolved),
        "file_count": len(files),
        "content_fingerprint": _canonical_sha256({"files": files}),
        "files": files,
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create one LF JSON report and refuse every existing target."""

    encoded = json.dumps(payload, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _rank_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    ranks = tuple(_positive_int(rank, f"{name} entry") for rank in value)
    if len(set(ranks)) != len(ranks) or tuple(sorted(ranks)) != ranks:
        raise ValueError(f"{name} must contain unique ascending ranks")
    return ranks


def _index_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    indices: list[int] = []
    for entry in value:
        if not isinstance(entry, int) or isinstance(entry, bool) or entry < 0:
            raise ValueError(f"{name} entries must be non-negative integers")
        indices.append(entry)
    result = tuple(indices)
    if len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise ValueError(f"{name} must contain unique ascending indices")
    return result


def _load_contract(path: Path) -> LocalLatentContract:
    contract_path = path.expanduser().resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read experiment contract {contract_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != _CONTRACT_SCHEMA:
        raise ValueError(f"contract schema_version must be {_CONTRACT_SCHEMA}")
    if payload.get("status") != "development-shadow-only":
        raise ValueError("contract status must be development-shadow-only")
    if payload.get("latent_dtype") != "bfloat16" or payload.get("basis_dtype") != "float32":
        raise ValueError("contract must use bfloat16 latents and float32 bases")
    if payload.get("gate_scope") != "overall_and_each_selected_layer":
        raise ValueError("contract gate_scope must be overall_and_each_selected_layer")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("contract model must be an object")
    relative_model_path = model.get("path")
    if not isinstance(relative_model_path, str) or not relative_model_path:
        raise ValueError("contract model.path must be a non-empty string")
    model_path = (contract_path.parent / relative_model_path).resolve()
    if not model_path.is_dir():
        raise ValueError(f"contract model path does not exist: {model_path}")
    model_config_path = model_path / "config.json"
    if not model_config_path.is_file():
        raise ValueError(f"contract model config does not exist: {model_config_path}")
    expected_model_sha = _sha256(model.get("config_sha256"), "model.config_sha256")
    expected_model_fingerprint = _sha256(
        model.get("content_fingerprint"), "model.content_fingerprint"
    )
    actual_model_sha = _sha256_file(model_config_path)
    if actual_model_sha != expected_model_sha:
        raise ValueError(
            f"model config sha256 mismatch: expected {expected_model_sha}, got {actual_model_sha}"
        )
    expected_config_raw = model.get("expected_config")
    if not isinstance(expected_config_raw, dict):
        raise ValueError("contract model.expected_config must be an object")
    expected_config = {
        name: _positive_int(expected_config_raw.get(name), f"model.expected_config.{name}")
        for name in (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
    }
    if (
        expected_config["num_attention_heads"]
        != expected_config["num_key_value_heads"] * GQA_QUERY_HEADS_PER_KV_HEAD
    ):
        raise ValueError("contract model must use the supported 2:1 GQA mapping")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("contract dataset must be an object")
    relative_manifest = dataset.get("manifest")
    if not isinstance(relative_manifest, str) or not relative_manifest:
        raise ValueError("contract dataset.manifest must be a non-empty string")
    dataset_manifest = (contract_path.parent / relative_manifest).resolve()
    if not dataset_manifest.is_file():
        raise ValueError(f"contract dataset manifest does not exist: {dataset_manifest}")
    expected_manifest_sha = _sha256(dataset.get("manifest_sha256"), "dataset.manifest_sha256")
    actual_manifest_sha = _sha256_file(dataset_manifest)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            "dataset manifest sha256 mismatch: expected "
            f"{expected_manifest_sha}, got {actual_manifest_sha}"
        )

    layers = _index_list(payload.get("layers"), "layers")
    if any(layer >= expected_config["num_hidden_layers"] for layer in layers):
        raise ValueError("contract layer is outside the expected model")
    key_ranks = _rank_list(payload.get("key_ranks"), "key_ranks")
    value_ranks = _rank_list(payload.get("value_ranks"), "value_ranks")
    if max(key_ranks + value_ranks) > expected_config["head_dim"]:
        raise ValueError("contract rank exceeds the expected head dimension")
    raw_pairs = payload.get("joint_rank_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("joint_rank_pairs must be a non-empty list")
    pairs: list[tuple[int, int]] = []
    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, dict):
            raise ValueError(f"joint_rank_pairs entry {index} must be an object")
        pair = (
            _positive_int(raw_pair.get("key"), f"joint_rank_pairs[{index}].key"),
            _positive_int(raw_pair.get("value"), f"joint_rank_pairs[{index}].value"),
        )
        if pair[0] not in key_ranks or pair[1] not in value_ranks:
            raise ValueError("every joint rank must appear in its corresponding rank sweep")
        pairs.append(pair)
    if len(set(pairs)) != len(pairs):
        raise ValueError("joint_rank_pairs must be unique")

    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, dict) or set(raw_gates) != _REQUIRED_GATE_KEYS:
        raise ValueError(f"contract gates must contain exactly {sorted(_REQUIRED_GATE_KEYS)}")
    gates: dict[str, float] = {}
    for name, raw_value in raw_gates.items():
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise ValueError(f"gate {name} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"gate {name} must be finite and non-negative")
        gates[name] = value
    for probability_gate in (
        "min_attention_top1_agreement",
        "min_attention_top_k_recall_mean",
        "min_output_cosine_similarity_mean",
    ):
        if gates[probability_gate] > 1:
            raise ValueError(f"gate {probability_gate} must not exceed one")

    return LocalLatentContract(
        path=contract_path,
        sha256=_sha256_file(contract_path),
        payload=payload,
        model_path=model_path,
        model_config_sha256=actual_model_sha,
        model_content_fingerprint=expected_model_fingerprint,
        expected_model_config=expected_config,
        dataset_manifest=dataset_manifest,
        dataset_manifest_sha256=actual_manifest_sha,
        layers=layers,
        calibration_tokens=_positive_int(dataset.get("calibration_tokens"), "calibration_tokens"),
        heldout_tokens=_positive_int(dataset.get("heldout_tokens"), "heldout_tokens"),
        heldout_query_positions=_positive_int(
            payload.get("heldout_query_positions"), "heldout_query_positions"
        ),
        key_ranks=key_ranks,
        value_ranks=value_ranks,
        joint_rank_pairs=tuple(pairs),
        projected_context_tokens=_positive_int(
            payload.get("projected_context_tokens"), "projected_context_tokens"
        ),
        page_size=_positive_int(payload.get("page_size"), "page_size"),
        top_k=_positive_int(payload.get("top_k"), "top_k"),
        seed=_positive_int(payload.get("seed"), "seed"),
        gates=gates,
    )


def _stack_queries(captures: tuple[CapturedLayer, ...]) -> Tensor:
    """Stack per-layer ``[H,S,Y]`` queries as ``[A,S,H,Y]``."""

    if not captures:
        raise ValueError("captures must not be empty")
    expected_shape = captures[0].queries.shape
    expected_positions = captures[0].query_positions
    if any(capture.queries.shape != expected_shape for capture in captures):
        raise ValueError("captured queries must have identical per-layer shapes")
    if any(capture.query_positions != expected_positions for capture in captures):
        raise ValueError("captured query positions must match across layers")
    return (
        torch.stack(tuple(capture.queries for capture in captures), dim=0)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _attention_metrics(
    exact_weights: Tensor,
    candidate_weights: Tensor,
    exact_output: Tensor,
    candidate_output: Tensor,
    *,
    top_k: int,
) -> dict[str, float | bool | int]:
    if exact_weights.shape != candidate_weights.shape:
        raise ValueError("exact and candidate attention weights must have identical shapes")
    if exact_output.shape != candidate_output.shape:
        raise ValueError("exact and candidate outputs must have identical shapes")
    exact = exact_weights.detach().to(device="cpu", dtype=torch.float32)
    candidate = candidate_weights.detach().to(device="cpu", dtype=torch.float32)
    reference_output = exact_output.detach().to(device="cpu", dtype=torch.float32)
    approximated_output = candidate_output.detach().to(device="cpu", dtype=torch.float32)
    selected_top_k = min(_positive_int(top_k, "top_k"), int(exact.numel()))
    exact_top = set(torch.topk(exact, selected_top_k).indices.tolist())
    candidate_top = set(torch.topk(candidate, selected_top_k).indices.tolist())
    tiny = torch.finfo(torch.float32).tiny
    kl = torch.sum(exact * (exact.clamp_min(tiny).log() - candidate.clamp_min(tiny).log()))
    output_delta = approximated_output - reference_output
    output_error_energy = float(torch.sum(output_delta.square(), dtype=torch.float64).item())
    output_reference_energy = float(
        torch.sum(reference_output.square(), dtype=torch.float64).item()
    )
    return {
        "attention_kl": max(0.0, float(kl.item())),
        "attention_top1_agreement": bool(exact.argmax().item() == candidate.argmax().item()),
        "attention_top_k": selected_top_k,
        "attention_top_k_recall": len(exact_top.intersection(candidate_top)) / selected_top_k,
        "output_relative_l2": math.sqrt(output_error_energy)
        / max(math.sqrt(output_reference_energy), 1e-12),
        "output_cosine_similarity": _cosine_similarity(reference_output, approximated_output),
        "output_error_energy": output_error_energy,
        "output_reference_energy": output_reference_energy,
    }


def _exact_attention(query: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    grouped_query = query.to(torch.float32).reshape(
        query.shape[0],
        keys.shape[2],
        GQA_QUERY_HEADS_PER_KV_HEAD,
        query.shape[-1],
    )
    scores = torch.einsum("axgy,atxy->axgt", grouped_query, keys.to(torch.float32))
    grouped_weights = torch.softmax(scores / math.sqrt(keys.shape[-1]), dim=-1)
    grouped_output = torch.einsum("axgt,atxy->axgy", grouped_weights, values.to(torch.float32))
    weights = grouped_weights.reshape(query.shape[0], query.shape[1], keys.shape[1])
    output = grouped_output.reshape(query.shape[0], query.shape[1], values.shape[-1])
    return weights, output


def _latent_weights(query: Tensor, keys: Tensor, *, native_head_dim: int) -> Tensor:
    grouped_query = query.to(torch.float32).reshape(
        query.shape[0],
        keys.shape[2],
        GQA_QUERY_HEADS_PER_KV_HEAD,
        query.shape[-1],
    )
    scores = torch.einsum("axgr,atxr->axgt", grouped_query, keys.to(torch.float32))
    grouped_weights = torch.softmax(scores / math.sqrt(native_head_dim), dim=-1)
    return grouped_weights.reshape(query.shape[0], query.shape[1], keys.shape[1])


def _native_value_output(weights: Tensor, values: Tensor) -> Tensor:
    grouped_weights = weights.reshape(
        weights.shape[0],
        values.shape[2],
        GQA_QUERY_HEADS_PER_KV_HEAD,
        weights.shape[-1],
    )
    grouped_output = torch.einsum("axgt,atxy->axgy", grouped_weights, values.to(torch.float32))
    return grouped_output.reshape(weights.shape[0], weights.shape[1], values.shape[-1])


def _latent_value_output(
    weights: Tensor,
    values: Tensor,
    *,
    value_basis: Tensor,
    value_center: Tensor,
) -> Tensor:
    grouped_weights = weights.reshape(
        weights.shape[0],
        values.shape[2],
        GQA_QUERY_HEADS_PER_KV_HEAD,
        weights.shape[-1],
    )
    latent_output = torch.einsum("axgt,atxr->axgr", grouped_weights, values.to(torch.float32))
    output = torch.einsum("axgr,axyr->axgy", latent_output, value_basis)
    output.add_(value_center[:, :, None, :])
    return output.reshape(weights.shape[0], weights.shape[1], value_basis.shape[-2])


def _append_records(
    records: list[dict[str, Any]],
    *,
    arm: str,
    variant: str,
    layer_indices: tuple[int, ...],
    query_position: int,
    exact_weights: Tensor,
    candidate_weights: Tensor,
    exact_output: Tensor,
    candidate_output: Tensor,
    top_k: int,
) -> None:
    for layer_slot, layer_index in enumerate(layer_indices):
        for query_head in range(exact_weights.shape[1]):
            records.append(
                {
                    "arm": arm,
                    "variant": variant,
                    "layer": layer_index,
                    "query_position": query_position,
                    "query_head": query_head,
                    "kv_head": query_head // GQA_QUERY_HEADS_PER_KV_HEAD,
                    "metrics": _attention_metrics(
                        exact_weights[layer_slot, query_head],
                        candidate_weights[layer_slot, query_head],
                        exact_output[layer_slot, query_head],
                        candidate_output[layer_slot, query_head],
                        top_k=top_k,
                    ),
                }
            )


def _evaluate_latent_sweep(
    codec: AtxyLocalLatentCodec,
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    query_positions: tuple[int, ...],
    layer_indices: tuple[int, ...],
    key_ranks: tuple[int, ...],
    value_ranks: tuple[int, ...],
    joint_rank_pairs: tuple[tuple[int, int], ...],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if queries.shape[1] != len(query_positions):
        raise ValueError("query tensor and logical query-position count must match")
    maximum_rank = max(key_ranks + value_ranks)
    key_latents = codec.encode_keys(keys, rank=maximum_rank)
    query_latents = codec.encode_queries(queries, rank=maximum_rank)
    value_latents = codec.encode_values(values, rank=maximum_rank)
    value_basis = codec.value_basis
    value_center = codec.value_center

    records: list[dict[str, Any]] = []
    for query_slot, logical_position in enumerate(query_positions):
        token_count = logical_position + 1
        if token_count < 1 or token_count > keys.shape[1]:
            raise ValueError("logical query position is outside captured K/V")
        exact_weights, exact_output = _exact_attention(
            queries[:, query_slot],
            keys[:, :token_count],
            values[:, :token_count],
        )
        key_weight_cache: dict[int, Tensor] = {}
        for key_rank in key_ranks:
            candidate_weights = _latent_weights(
                query_latents[:, query_slot, :, :key_rank],
                key_latents[:, :token_count, :, :key_rank],
                native_head_dim=codec.shape.channels,
            )
            key_weight_cache[key_rank] = candidate_weights
            candidate_output = _native_value_output(
                candidate_weights,
                values[:, :token_count],
            )
            _append_records(
                records,
                arm=f"key-r{key_rank}",
                variant="K-only",
                layer_indices=layer_indices,
                query_position=logical_position,
                exact_weights=exact_weights,
                candidate_weights=candidate_weights,
                exact_output=exact_output,
                candidate_output=candidate_output,
                top_k=top_k,
            )

        for value_rank in value_ranks:
            candidate_output = _latent_value_output(
                exact_weights,
                value_latents[:, :token_count, :, :value_rank],
                value_basis=value_basis[..., :value_rank],
                value_center=value_center,
            )
            _append_records(
                records,
                arm=f"value-r{value_rank}",
                variant="V-only",
                layer_indices=layer_indices,
                query_position=logical_position,
                exact_weights=exact_weights,
                candidate_weights=exact_weights,
                exact_output=exact_output,
                candidate_output=candidate_output,
                top_k=top_k,
            )

        for key_rank, value_rank in joint_rank_pairs:
            candidate_weights = key_weight_cache[key_rank]
            candidate_output = _latent_value_output(
                candidate_weights,
                value_latents[:, :token_count, :, :value_rank],
                value_basis=value_basis[..., :value_rank],
                value_center=value_center,
            )
            _append_records(
                records,
                arm=f"joint-k{key_rank}-v{value_rank}",
                variant="K+V",
                layer_indices=layer_indices,
                query_position=logical_position,
                exact_weights=exact_weights,
                candidate_weights=candidate_weights,
                exact_output=exact_output,
                candidate_output=candidate_output,
                top_k=top_k,
            )

    latent_identity = {
        "keys_sha256": _tensor_sha256(key_latents),
        "queries_sha256": _tensor_sha256(query_latents),
        "values_sha256": _tensor_sha256(value_latents),
        "shapes": {
            "keys": list(key_latents.shape),
            "queries": list(query_latents.shape),
            "values": list(value_latents.shape),
        },
    }
    return records, latent_identity


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability).item())


def _summarize_records(records: list[dict[str, Any]], *, by_layer: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["arm"], record["variant"])
        if by_layer:
            key += (record["layer"],)
        grouped.setdefault(key, []).append(record["metrics"])
    summaries: list[dict[str, Any]] = []
    for key, metrics in sorted(grouped.items()):

        def values(name: str) -> list[float]:
            return [float(metric[name]) for metric in metrics]

        error_energy = sum(values("output_error_energy"))
        reference_energy = sum(values("output_reference_energy"))
        summary: dict[str, Any] = {
            "arm": key[0],
            "variant": key[1],
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
            "output_relative_l2_global": math.sqrt(error_energy)
            / max(math.sqrt(reference_energy), 1e-12),
            "output_cosine_similarity_mean": sum(values("output_cosine_similarity")) / len(metrics),
            "output_cosine_similarity_p01": _quantile(values("output_cosine_similarity"), 0.01),
        }
        if by_layer:
            summary["layer"] = key[2]
        summaries.append(summary)
    return summaries


def _gate_joint_arms(
    contract: LocalLatentContract,
    overall: list[dict[str, Any]],
    by_layer: list[dict[str, Any]],
    storage_by_arm: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    overall_by_arm = {summary["arm"]: summary for summary in overall}
    layers_by_arm: dict[str, list[dict[str, Any]]] = {}
    for summary in by_layer:
        layers_by_arm.setdefault(summary["arm"], []).append(summary)
    decisions: list[dict[str, Any]] = []
    for key_rank, value_rank in contract.joint_rank_pairs:
        arm = f"joint-k{key_rank}-v{value_rank}"
        if arm not in overall_by_arm or arm not in storage_by_arm:
            raise ValueError(f"missing overall metrics or storage projection for {arm}")
        observed_layers = {int(summary["layer"]) for summary in layers_by_arm.get(arm, [])}
        if len(layers_by_arm.get(arm, [])) != len(contract.layers) or observed_layers != set(
            contract.layers
        ):
            raise ValueError(
                f"layer summaries for {arm} must exactly cover fixed layers {contract.layers}"
            )
        expected_per_layer = (
            contract.heldout_query_positions * contract.expected_model_config["num_attention_heads"]
        )
        if int(overall_by_arm[arm]["evaluations"]) != expected_per_layer * len(contract.layers):
            raise ValueError(f"overall evaluation count is incomplete for {arm}")
        if any(int(summary["evaluations"]) != expected_per_layer for summary in layers_by_arm[arm]):
            raise ValueError(f"per-layer evaluation count is incomplete for {arm}")
        scoped = [overall_by_arm[arm], *layers_by_arm[arm]]
        observed = {
            "projected_compression_ratio": storage_by_arm[arm]["compression_ratio"],
            "attention_kl_mean_max_scope": max(
                float(summary["attention_kl_mean"]) for summary in scoped
            ),
            "attention_kl_p99_max_scope": max(
                float(summary["attention_kl_p99"]) for summary in scoped
            ),
            "attention_top1_agreement_min_scope": min(
                float(summary["attention_top1_agreement"]) for summary in scoped
            ),
            "attention_top_k_recall_mean_min_scope": min(
                float(summary["attention_top_k_recall_mean"]) for summary in scoped
            ),
            "output_relative_l2_global_max_scope": max(
                float(summary["output_relative_l2_global"]) for summary in scoped
            ),
            "output_cosine_similarity_mean_min_scope": min(
                float(summary["output_cosine_similarity_mean"]) for summary in scoped
            ),
        }
        checks = {
            "projected_compression_ratio": observed["projected_compression_ratio"]
            >= contract.gates["min_projected_compression_ratio"],
            "attention_kl_mean": observed["attention_kl_mean_max_scope"]
            <= contract.gates["max_attention_kl_mean"],
            "attention_kl_p99": observed["attention_kl_p99_max_scope"]
            <= contract.gates["max_attention_kl_p99"],
            "attention_top1_agreement": observed["attention_top1_agreement_min_scope"]
            >= contract.gates["min_attention_top1_agreement"],
            "attention_top_k_recall_mean": observed["attention_top_k_recall_mean_min_scope"]
            >= contract.gates["min_attention_top_k_recall_mean"],
            "output_relative_l2_global": observed["output_relative_l2_global_max_scope"]
            <= contract.gates["max_output_relative_l2_global"],
            "output_cosine_similarity_mean": observed["output_cosine_similarity_mean_min_scope"]
            >= contract.gates["min_output_cosine_similarity_mean"],
        }
        decisions.append(
            {
                "arm": arm,
                "key_rank": key_rank,
                "value_rank": value_rank,
                "observed_worst_scope": observed,
                "checks": checks,
                "all_fixed_development_gates_passed": all(checks.values()),
            }
        )
    passing = [decision for decision in decisions if decision["all_fixed_development_gates_passed"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda decision: (
                storage_by_arm[decision["arm"]]["compression_ratio"],
                -(decision["key_rank"] + decision["value_rank"]),
            ),
        )["arm"]
    return decisions, selected


def _project_storage(
    codec: AtxyLocalLatentCodec,
    contract: LocalLatentContract,
) -> dict[str, dict[str, Any]]:
    expected_shape = (
        len(contract.layers),
        contract.expected_model_config["num_key_value_heads"],
        contract.expected_model_config["head_dim"],
    )
    actual_shape = (codec.shape.layers, codec.shape.heads, codec.shape.channels)
    if actual_shape != expected_shape:
        raise ValueError(
            f"codec shape {actual_shape} does not match fixed contract shape {expected_shape}"
        )
    projections: dict[str, dict[str, Any]] = {}
    for key_rank, value_rank in contract.joint_rank_pairs:
        arm = f"joint-k{key_rank}-v{value_rank}"
        storage = codec.projected_storage_bytes(
            token_count=contract.projected_context_tokens,
            key_rank=key_rank,
            value_rank=value_rank,
        )
        latent_bytes_per_token = storage.latent_bytes // storage.token_count
        same_budget_capacity = max(
            0,
            (storage.native_bytes - storage.basis_bytes) // latent_bytes_per_token,
        )
        full_layer_count = contract.expected_model_config["num_hidden_layers"]
        full_model = {
            name: (getattr(storage, name) // codec.shape.layers) * full_layer_count
            for name in (
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
        }
        full_model["layer_count"] = full_layer_count
        full_model["compression_ratio"] = full_model["native_bytes"] / full_model["total_bytes"]
        full_model["savings_bytes"] = full_model["native_bytes"] - full_model["total_bytes"]
        projections[arm] = {
            **asdict(storage),
            "measured_layer_count": codec.shape.layers,
            "compression_ratio": storage.compression_ratio,
            "savings_bytes": storage.savings_bytes,
            "full_model_dimensional_projection": full_model,
            "arithmetic_same_native_budget_latent_token_capacity": same_budget_capacity,
            "arithmetic_same_budget_capacity_multiplier": same_budget_capacity
            / storage.token_count,
            "arithmetic_capacity_projection_exceeds_model_position_contract": same_budget_capacity
            > contract.expected_model_config["max_position_embeddings"],
        }
    return projections


def _reconstruction_sweep(
    codec: AtxyLocalLatentCodec,
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    *,
    key_ranks: tuple[int, ...],
    value_ranks: tuple[int, ...],
) -> dict[str, list[dict[str, Any]]]:
    maximum_rank = max(key_ranks + value_ranks)
    encoded_keys = codec.encode_keys(keys, rank=maximum_rank)
    encoded_queries = codec.encode_queries(queries, rank=maximum_rank)
    encoded_values = codec.encode_values(values, rank=maximum_rank)
    key_records: list[dict[str, Any]] = []
    value_records: list[dict[str, Any]] = []
    for rank in key_ranks:
        key_records.append(
            {
                "rank": rank,
                "keys": _reconstruction_metrics(
                    keys,
                    codec.decode_keys(encoded_keys[..., :rank]),
                ),
                "queries": _reconstruction_metrics(
                    queries,
                    codec.decode_queries(encoded_queries[..., :rank]),
                ),
            }
        )
    for rank in value_ranks:
        value_records.append(
            {
                "rank": rank,
                "values": _reconstruction_metrics(
                    values,
                    codec.decode_values(encoded_values[..., :rank]),
                ),
            }
        )
    return {"key_query": key_records, "value": value_records}


def _case_by_split(cases: tuple[ContextCase, ...], split: str) -> ContextCase:
    matches = [case for case in cases if case.split == split]
    if len(matches) != 1:
        raise ValueError(f"contract requires exactly one {split} case")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed ATXY local-latent shadow ceiling on real Qwen Q/K/V."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        contract = _load_contract(args.contract)
        cases = _load_dataset_manifest(contract.dataset_manifest)
        calibration_case = _case_by_split(cases, "calibration")
        heldout_case = _case_by_split(cases, "heldout")
    except ValueError as error:
        parser.error(str(error))
    if calibration_case.token_cap != contract.calibration_tokens:
        parser.error("calibration token cap does not match the fixed contract")
    if heldout_case.token_cap != contract.heldout_tokens:
        parser.error("held-out token cap does not match the fixed contract")
    torch.manual_seed(contract.seed)
    source_tree_root = Path(__file__).resolve().parents[1]
    source_tree_preflight = _source_tree_identity(source_tree_root)

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit("Install molor[hf] for the ATXY local-latent lab") from error

    model_id = str(contract.model_path)
    source_config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    for name, expected in contract.expected_model_config.items():
        actual = int(getattr(source_config, name))
        if actual != expected:
            parser.error(f"model config {name} mismatch: expected {expected}, got {actual}")
    try:
        layers = _resolve_layers(contract.layers, int(source_config.num_hidden_layers))
    except ValueError as error:
        parser.error(str(error))
    model_artifact_stack = lock_backend_identity(
        {
            "backend": "transformers",
            "model_artifact_path": model_id,
            "configured_expert_paths": [],
            "experts": [],
        },
        cache_path=Path("unused-no-digest-cache.json"),
    )
    if not model_artifact_stack["fully_content_addressed"]:
        parser.error("fixed local model could not be fully content-addressed")
    base_artifact = model_artifact_stack["artifacts"][0]
    actual_model_fingerprint = str(base_artifact["content_fingerprint"])
    if actual_model_fingerprint != contract.model_content_fingerprint:
        parser.error(
            "model content fingerprint mismatch: expected "
            f"{contract.model_content_fingerprint}, got {actual_model_fingerprint}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
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

    print(f"Loading fixed local model {model_id} in bfloat16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    device = str(model.get_input_embeddings().weight.device)
    attention_implementation = str(model.config._attn_implementation)
    captures: dict[str, tuple[CapturedLayer, ...]] = {}
    capture_metadata: list[dict[str, Any]] = []
    for case in (calibration_case, heldout_case):
        tokens = case_tokens[case.case_id]
        query_positions = (
            tuple(range(int(tokens.numel())))
            if case.split == "calibration"
            else _query_positions(int(tokens.numel()), contract.heldout_query_positions)
        )
        print(
            f"Capturing {case.split} {case.case_id}: {tokens.numel():,} tokens, "
            f"{len(query_positions):,} query positions",
            flush=True,
        )
        started = time.perf_counter()
        captures[case.case_id] = _capture_qkv(
            model,
            input_ids=tokens,
            layer_indices=layers,
            query_positions=query_positions,
        )
        capture_metadata.append(
            {
                "id": case.case_id,
                "split": case.split,
                "source": str(case.path),
                "source_sha256": case.source_sha256,
                "token_count": int(tokens.numel()),
                "token_sha256": hashlib.sha256(
                    json.dumps(tokens.tolist(), separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "query_positions": list(query_positions),
                "capture_seconds": time.perf_counter() - started,
            }
        )
    del model, case_tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    calibration_capture = captures[calibration_case.case_id]
    calibration_keys = _stack_payload(calibration_capture, "keys")
    calibration_queries = _stack_queries(calibration_capture)
    calibration_values = _stack_payload(calibration_capture, "values")
    maximum_rank = max(contract.key_ranks + contract.value_ranks)
    print(
        f"Fitting {len(layers) * int(source_config.num_key_value_heads)} local K/Q and V "
        f"bases at max rank {maximum_rank}",
        flush=True,
    )
    fit_started = time.perf_counter()
    codec = AtxyLocalLatentCodec.fit(
        calibration_keys,
        calibration_queries,
        calibration_values,
        max_rank=maximum_rank,
        latent_dtype=torch.bfloat16,
    )
    fit_seconds = time.perf_counter() - fit_started
    calibration_tensor_hashes = {
        "keys": _tensor_sha256(calibration_keys),
        "queries": _tensor_sha256(calibration_queries),
        "values": _tensor_sha256(calibration_values),
    }
    del captures[calibration_case.case_id]
    del calibration_keys, calibration_queries, calibration_values, calibration_capture
    gc.collect()

    heldout_capture = captures[heldout_case.case_id]
    heldout_keys = _stack_payload(heldout_capture, "keys")
    heldout_queries = _stack_queries(heldout_capture)
    heldout_values = _stack_payload(heldout_capture, "values")
    heldout_positions = heldout_capture[0].query_positions
    print(
        f"Evaluating {len(contract.key_ranks)} K ranks, {len(contract.value_ranks)} V ranks, "
        f"and {len(contract.joint_rank_pairs)} joint arms",
        flush=True,
    )
    evaluation_started = time.perf_counter()
    reconstruction = _reconstruction_sweep(
        codec,
        heldout_queries,
        heldout_keys,
        heldout_values,
        key_ranks=contract.key_ranks,
        value_ranks=contract.value_ranks,
    )
    attention_records, latent_identity = _evaluate_latent_sweep(
        codec,
        heldout_queries,
        heldout_keys,
        heldout_values,
        query_positions=heldout_positions,
        layer_indices=layers,
        key_ranks=contract.key_ranks,
        value_ranks=contract.value_ranks,
        joint_rank_pairs=contract.joint_rank_pairs,
        top_k=contract.top_k,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    heldout_tensor_hashes = {
        "keys": _tensor_sha256(heldout_keys),
        "queries": _tensor_sha256(heldout_queries),
        "values": _tensor_sha256(heldout_values),
    }
    overall_summary = _summarize_records(attention_records, by_layer=False)
    layer_summary = _summarize_records(attention_records, by_layer=True)
    storage_by_arm = _project_storage(codec, contract)
    gate_decisions, selected_arm = _gate_joint_arms(
        contract,
        overall_summary,
        layer_summary,
        storage_by_arm,
    )
    decision = (
        "local_latent_ceiling_passed_fixed_development_gates_at_selected_layers"
        if selected_arm is not None
        else "no_joint_arm_passed_fixed_development_gates"
    )

    page_count = math.ceil(contract.projected_context_tokens / contract.page_size)
    report = {
        "schema_version": _REPORT_SCHEMA,
        "status": "development-shadow-only; no live cache, context extension, or promotion claim",
        "promotion_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "selected_development_arm": selected_arm,
        "contract": {
            "path": str(contract.path),
            "sha256": contract.sha256,
            "payload": contract.payload,
        },
        "implementation": {
            "source_tree_preflight": source_tree_preflight,
            "postflight_requirement": (
                "Every src/molor Python byte must match preflight before report creation."
            ),
        },
        "model": {
            "path": model_id,
            "config_sha256": contract.model_config_sha256,
            "content_fingerprint": contract.model_content_fingerprint,
            "artifact_stack": model_artifact_stack,
            "config": contract.expected_model_config,
        },
        "dataset": {
            "manifest": str(contract.dataset_manifest),
            "manifest_sha256": contract.dataset_manifest_sha256,
            "cases": capture_metadata,
        },
        "capture_tensor_sha256": {
            "calibration": calibration_tensor_hashes,
            "heldout": heldout_tensor_hashes,
        },
        "codec": {
            "version": "molor.atxy-local-linear-latent.v1",
            "equation": "K basis = eig(0.5 mean(KK') + 0.5 mean(QQ')); V = centered local PCA",
            "shape": asdict(codec.shape),
            "max_rank": codec.max_rank,
            "latent_dtype": str(codec.latent_dtype),
            "reconstruction_dtype": str(codec.reconstruction_dtype),
            "basis_dtype": "torch.float32",
            "resident_max_rank_basis_bytes": codec.resident_basis_bytes,
            "basis_sha256": {
                "key": _tensor_sha256(codec.key_basis),
                "value": _tensor_sha256(codec.value_basis),
                "value_center": _tensor_sha256(codec.value_center),
            },
            "fit_seconds": fit_seconds,
        },
        "atxy_page_identity": {
            "page_size": contract.page_size,
            "projected_context_pages_per_layer": page_count,
            "pages_per_layer_namespace": ATXY_PAGES_PER_LAYER,
            "tokens_per_layer_namespace": token_capacity_per_layer(contract.page_size),
            "selected_layer_last_addresses": {
                str(layer): AtxyPageAddress.for_layer_page(layer, page_count - 1).text
                for layer in layers
            },
            "role": "exact physical page identity only; not a semantic hash or latent code",
        },
        "storage_projection": storage_by_arm,
        "reconstruction": reconstruction,
        "attention_summary": overall_summary,
        "attention_summary_by_layer": layer_summary,
        "attention_records": attention_records,
        "latent_identity": latent_identity,
        "fixed_gate_decisions": gate_decisions,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "capture_device": device,
            "codec_and_evaluation_device": "cpu",
            "attention_implementation": attention_implementation,
            "evaluation_seconds": evaluation_seconds,
        },
        "limitations": [
            "Only layers 0, 13, and 27 and one file-disjoint 4K held-out context are measured.",
            "The 40,960-token result is dimensional storage arithmetic, not a long-context quality run.",
            "No live cache, GPU latent kernel, codebook, hash router, TTL, or context-position extension is installed.",
            "Passing permits only a separately versioned latent-codebook experiment; it cannot promote a cache.",
        ],
    }
    source_tree_postflight = _source_tree_identity(source_tree_root)
    if source_tree_postflight != source_tree_preflight:
        raise SystemExit(
            "refusing to write report because the MoLoR source tree changed during run"
        )
    report["implementation"]["source_tree_postflight_verified"] = True
    report["report_payload_sha256"] = _canonical_sha256(report)
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        "atxy-local-latent-shadow-" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    )
    try:
        _write_json_exclusive(output_path, report)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing shadow report: {output_path}") from error
    print(f"Wrote fixed shadow report to {output_path}", flush=True)
    for gate in gate_decisions:
        storage = storage_by_arm[gate["arm"]]
        print(
            f"{gate['arm']}: {storage['compression_ratio']:.3f}x, "
            f"gates={'PASS' if gate['all_fixed_development_gates_passed'] else 'FAIL'}",
            flush=True,
        )
    print(f"Decision: {decision}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
