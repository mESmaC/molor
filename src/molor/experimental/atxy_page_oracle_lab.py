"""Development-only all-layer/head exact-payload ATXY page-oracle lab.

This lab captures unchanged dense Qwen Q/K/V, grades an attention-mass page
oracle on CPU, and writes a non-promotional shadow report.  The oracle uses
current-query dense attention and is not a runtime selector.  KV remains exact.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform
import time
from typing import Any, Iterable

import torch
from torch import Tensor

from molor.benchmark.lock import lock_backend_identity
from molor.experimental.atxy_page_oracle import build_exact_page_oracle_curve_from_weights
from molor.experimental.atxy_pages import AtxyPageAddress, token_capacity_per_layer
from molor.experimental.hashed_kv import build_page_spans, exact_attention_weights
from molor.experimental.hashed_kv_lab import (
    CapturedLayer,
    _capture_qkv,
    _tensor_sha256,
    _validate_context_length,
)


DEFAULT_MODEL = Path("models/Qwen3-1.7B")
DEFAULT_OUTPUT_DIRECTORY = Path("runs/atxy-page-oracle-v1")
_DATASET_SCHEMA = "molor.atxy-page-oracle-development-dataset.v1"
_REPORT_SCHEMA = "molor.atxy-page-oracle-development-report.v1"
_STATUS = "development-only"
_MASS_TARGETS = (0.98, 0.995, 0.999)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_stable_bytes(path: Path, name: str) -> bytes:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise ValueError(f"could not snapshot {name} {path}: {error}") from error
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise ValueError(f"{name} changed while it was being snapshotted: {path}")
    return payload


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _strict_json_bytes(value: bytes, name: str) -> Any:
    try:
        text = value.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not parse strict JSON {name}: {error}") from error


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _seal_report(payload: dict[str, Any]) -> dict[str, Any]:
    if "integrity" in payload:
        raise ValueError("unsealed report must not contain integrity")
    sealed = dict(payload)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON of this report with the integrity member omitted",
        "sha256": _canonical_sha256(sealed),
    }
    return sealed


def _token_sha256(token_ids: Tensor) -> str:
    values = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    digest = hashlib.sha256()
    digest.update(b"molor.atxy-page-oracle-token-ids.int64.v1\0")
    digest.update(bytes(values.untyped_storage()))
    return digest.hexdigest()


def _implementation_identity() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    paths = (
        Path(__file__).resolve(),
        directory / "atxy_page_oracle.py",
        directory / "atxy_pages.py",
        directory / "hashed_kv.py",
        directory / "hashed_kv_lab.py",
    )
    files = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    return {"files": files, "content_fingerprint": _canonical_sha256(files)}


def _local_model_stack_identity(model_path: Path) -> dict[str, Any]:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"model must be a local directory: {resolved}")
    identity = lock_backend_identity(
        {
            "backend": "transformers-local-atxy-page-oracle",
            "model_id": str(resolved),
            "model_artifact_path": str(resolved),
            "configured_expert_paths": [],
            "experts": [],
        },
        cache_path=Path("unused-atxy-page-oracle-lock-cache.json"),
    )
    if identity.get("fully_content_addressed") is not True:
        raise ValueError("local model/tokenizer/config/weight stack is not fully content-addressed")
    artifacts = identity.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("local model identity must contain exactly one base artifact")
    artifact = artifacts[0]
    files = artifact.get("files") if isinstance(artifact, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("local model identity contains no files")
    names = {
        item.get("name")
        for item in files
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if "config.json" not in names:
        raise ValueError("local model identity omitted config.json")
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    }
    if not names.intersection(tokenizer_names):
        raise ValueError("local model identity omitted tokenizer payload bytes")
    if not any(
        isinstance(name, str)
        and Path(name).suffix.casefold() in {".safetensors", ".bin", ".pt", ".pth"}
        for name in names
    ):
        raise ValueError("local model identity omitted model weight bytes")
    return identity


@dataclass(frozen=True, slots=True)
class QueryCheckpoint:
    token_count: int
    support_token_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class DevelopmentCase:
    case_id: str
    path: Path
    source_sha256: str
    source_size_bytes: int
    source_bytes: bytes
    source_text: str
    token_count: int
    queries: tuple[QueryCheckpoint, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentDatasetSnapshot:
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    manifest_bytes: bytes
    cases: tuple[DevelopmentCase, ...]


def _support_spans(value: object, *, token_count: int, case_id: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"case {case_id} query {token_count} needs support_token_spans")
    spans: list[tuple[int, int]] = []
    for raw_span in value:
        if (
            not isinstance(raw_span, list)
            or len(raw_span) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in raw_span)
        ):
            raise ValueError(f"case {case_id} support spans must be [start, stop] integer pairs")
        start, stop = raw_span
        if start < 0 or stop <= start or stop > token_count:
            raise ValueError(f"case {case_id} support span is outside query token range")
        spans.append((start, stop))
    if len(set(spans)) != len(spans):
        raise ValueError(f"case {case_id} has duplicate support spans")
    return tuple(spans)


def _load_dataset_manifest(path: Path) -> DevelopmentDatasetSnapshot:
    manifest_path = path.expanduser().resolve()
    manifest_bytes = _read_stable_bytes(manifest_path, "dataset manifest")
    payload = _strict_json_bytes(manifest_bytes, str(manifest_path))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "status", "cases"}:
        raise ValueError("dataset manifest must contain exactly schema_version, status, and cases")
    if payload["schema_version"] != _DATASET_SCHEMA or payload["status"] != _STATUS:
        raise ValueError("dataset manifest is not the development-only page-oracle schema")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("dataset manifest cases must be a non-empty list")

    cases: list[DevelopmentCase] = []
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict) or set(raw_case) != {
            "id",
            "path",
            "sha256",
            "token_count",
            "queries",
        }:
            raise ValueError(f"dataset case {index} has unexpected fields")
        case_id = raw_case["id"]
        source = raw_case["path"]
        expected_sha = raw_case["sha256"]
        token_count = _positive_int(raw_case["token_count"], f"dataset case {index} token_count")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError(f"dataset case {index} id must be non-empty and unique")
        if not isinstance(source, str) or not source:
            raise ValueError(f"dataset case {case_id} path must be a non-empty string")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_sha)
        ):
            raise ValueError(f"dataset case {case_id} sha256 must be 64 hexadecimal characters")
        source_path = (manifest_path.parent / source).resolve()
        if not source_path.is_file():
            raise ValueError(f"dataset case {case_id} source does not exist: {source_path}")
        source_bytes = _read_stable_bytes(source_path, f"dataset case {case_id} source")
        actual_sha = _sha256_bytes(source_bytes)
        if actual_sha != expected_sha.lower():
            raise ValueError(f"dataset case {case_id} source sha256 mismatch")
        if actual_sha in seen_digests:
            raise ValueError(f"dataset case {case_id} duplicates another source payload")
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"dataset case {case_id} source is not valid UTF-8") from error

        raw_queries = raw_case["queries"]
        if not isinstance(raw_queries, list) or not raw_queries:
            raise ValueError(f"dataset case {case_id} queries must be non-empty")
        queries: list[QueryCheckpoint] = []
        for query_index, raw_query in enumerate(raw_queries):
            if not isinstance(raw_query, dict) or set(raw_query) != {
                "token_count",
                "support_token_spans",
            }:
                raise ValueError(
                    f"dataset case {case_id} query {query_index} has unexpected fields"
                )
            query_tokens = _positive_int(
                raw_query["token_count"], f"dataset case {case_id} query token_count"
            )
            queries.append(
                QueryCheckpoint(
                    token_count=query_tokens,
                    support_token_spans=_support_spans(
                        raw_query["support_token_spans"],
                        token_count=query_tokens,
                        case_id=case_id,
                    ),
                )
            )
        query_counts = tuple(query.token_count for query in queries)
        if tuple(sorted(set(query_counts))) != query_counts:
            raise ValueError(f"dataset case {case_id} query token counts must increase uniquely")
        if query_counts[-1] != token_count:
            raise ValueError(f"dataset case {case_id} final query must equal token_count")

        seen_ids.add(case_id)
        seen_digests.add(actual_sha)
        cases.append(
            DevelopmentCase(
                case_id=case_id,
                path=source_path,
                source_sha256=actual_sha,
                source_size_bytes=len(source_bytes),
                source_bytes=source_bytes,
                source_text=source_text,
                token_count=token_count,
                queries=tuple(queries),
            )
        )
    return DevelopmentDatasetSnapshot(
        manifest_path=manifest_path,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        manifest_size_bytes=len(manifest_bytes),
        manifest_bytes=manifest_bytes,
        cases=tuple(cases),
    )


def _verify_dataset_snapshot(snapshot: DevelopmentDatasetSnapshot) -> None:
    manifest_bytes = _read_stable_bytes(snapshot.manifest_path, "dataset manifest postflight")
    if manifest_bytes != snapshot.manifest_bytes:
        raise RuntimeError("dataset manifest bytes changed after the preflight snapshot")
    for case in snapshot.cases:
        source_bytes = _read_stable_bytes(case.path, f"dataset case {case.case_id} postflight")
        if source_bytes != case.source_bytes:
            raise RuntimeError(f"dataset case {case.case_id} source bytes changed after preflight")


def _token_ids(tokenizer: Any, case: DevelopmentCase) -> Tensor:
    encoded = tokenizer(case.source_text, add_special_tokens=True)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError(f"case {case.case_id} tokenizer output has no input_ids")
    raw_ids = encoded["input_ids"]
    if isinstance(raw_ids, Tensor):
        raw_ids = raw_ids.detach().to(device="cpu").tolist()
    if isinstance(raw_ids, list) and len(raw_ids) == 1 and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]
    if not isinstance(raw_ids, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw_ids
    ):
        raise ValueError(f"case {case.case_id} tokenizer returned unsupported token IDs")
    if len(raw_ids) != case.token_count:
        raise ValueError(
            f"case {case.case_id} tokenized to {len(raw_ids)} tokens; "
            f"manifest requires exactly {case.token_count}"
        )
    return torch.tensor(raw_ids, dtype=torch.long)


def _capture_independent_prefixes(
    model: Any,
    *,
    input_ids: Tensor,
    layer_indices: tuple[int, ...],
    checkpoints: tuple[QueryCheckpoint, ...],
    capture_fn: Callable[..., tuple[CapturedLayer, ...]] = _capture_qkv,
) -> Iterator[tuple[QueryCheckpoint, list[CapturedLayer], float]]:
    """Capture each checkpoint in an independent causal-prefix forward."""

    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("input_ids must be a non-empty token vector")
    for checkpoint in checkpoints:
        if checkpoint.token_count > input_ids.numel():
            raise ValueError("checkpoint exceeds the snapshotted case tokens")
        prefix = input_ids[: checkpoint.token_count].contiguous()
        started = time.perf_counter()
        captures = list(
            capture_fn(
                model,
                input_ids=prefix,
                layer_indices=layer_indices,
                query_positions=(checkpoint.token_count - 1,),
            )
        )
        elapsed = time.perf_counter() - started
        if tuple(capture.layer_index for capture in captures) != layer_indices:
            raise RuntimeError("independent prefix capture returned an incomplete layer matrix")
        if any(
            capture.query_positions != (checkpoint.token_count - 1,)
            or capture.keys.shape[1] != checkpoint.token_count
            or capture.values.shape[1] != checkpoint.token_count
            for capture in captures
        ):
            raise RuntimeError("independent prefix capture returned non-causal Q/K/V geometry")
        yield checkpoint, captures, elapsed


def _nearest_rank(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    probability = _fraction(probability, "probability")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _distribution(values: Iterable[float]) -> dict[str, float]:
    items = tuple(float(value) for value in values)
    if not items or any(not math.isfinite(value) for value in items):
        raise ValueError("metric distribution must be non-empty and finite")
    return {
        "minimum": min(items),
        "p01": _nearest_rank(items, 0.01),
        "p05": _nearest_rank(items, 0.05),
        "p50": _nearest_rank(items, 0.50),
        "p95": _nearest_rank(items, 0.95),
        "p99": _nearest_rank(items, 0.99),
        "maximum": max(items),
        "mean": sum(items) / len(items),
    }


def _summarize_budget_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["page_size"], record["requested_candidate_fraction"], record["token_count"])
        grouped.setdefault(key, []).append(record)
    summaries: list[dict[str, Any]] = []
    for (page_size, fraction, token_count), group in sorted(grouped.items()):
        delta_squared = sum(record["output_delta_l2_squared"] for record in group)
        exact_squared = sum(record["exact_output_l2_squared"] for record in group)
        summaries.append(
            {
                "page_size": page_size,
                "requested_candidate_fraction": fraction,
                "token_count": token_count,
                "evaluations": len(group),
                "retained_attention_mass": _distribution(
                    record["retained_attention_mass"] for record in group
                ),
                "top_k_recall": _distribution(record["top_k_recall"] for record in group),
                "maximum_attention_token_page_recall": sum(
                    float(record["maximum_attention_token_page_recalled"]) for record in group
                )
                / len(group),
                "top_attention_page_recall_1": sum(
                    record["top_attention_page_recall_1"] for record in group
                )
                / len(group),
                "top_attention_page_recall_4": sum(
                    record["top_attention_page_recall_4"] for record in group
                )
                / len(group),
                "support_page_recall": _distribution(
                    record["support_page_recall"] for record in group
                ),
                "candidate_token_fraction": _distribution(
                    record["candidate_token_fraction"] for record in group
                ),
                "output_relative_l2": _distribution(
                    record["output_relative_l2"] for record in group
                ),
                "output_relative_l2_global": (
                    math.sqrt(delta_squared / exact_squared)
                    if exact_squared > 0
                    else (0.0 if delta_squared == 0 else math.inf)
                ),
                "output_cosine_similarity": _distribution(
                    record["output_cosine_similarity"] for record in group
                ),
            }
        )
    return summaries


def _summarize_minimum_mass_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["page_size"], record["target_mass"], record["token_count"])
        grouped.setdefault(key, []).append(record)
    return [
        {
            "page_size": key[0],
            "target_mass": key[1],
            "token_count": key[2],
            "evaluations": len(group),
            "required_candidate_page_fraction": _distribution(
                record["candidate_page_fraction"] for record in group
            ),
            "required_candidate_token_fraction": _distribution(
                record["candidate_token_fraction"] for record in group
            ),
        }
        for key, group in sorted(grouped.items())
    ]


def _summarize_physical_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["page_size"], record["requested_candidate_fraction"], record["token_count"])
        grouped.setdefault(key, []).append(record)
    return [
        {
            "page_size": key[0],
            "requested_candidate_fraction": key[1],
            "token_count": key[2],
            "layer_query_evaluations": len(group),
            "layer_union_physical_fetch_fraction": _distribution(
                record["layer_union_physical_fetch_fraction"] for record in group
            ),
            "support_page_recall": _distribution(record["support_page_recall"] for record in group),
        }
        for key, group in sorted(grouped.items())
    ]


def _support_page_ids(
    spans: tuple[tuple[int, int], ...], *, token_count: int, page_size: int
) -> set[int]:
    pages = build_page_spans(token_count, page_size)
    return {
        page.page_id
        for start, stop in spans
        for page in pages
        if page.start < stop and page.stop > start
    }


def _selected_token_count(page_ids: set[int], *, token_count: int, page_size: int) -> int:
    pages = build_page_spans(token_count, page_size)
    return sum(pages[page_id].length for page_id in page_ids)


def _evaluate_captured_layer(
    capture: CapturedLayer,
    *,
    case_id: str,
    checkpoints: tuple[QueryCheckpoint, ...],
    page_sizes: tuple[int, ...],
    candidate_fractions: tuple[float, ...],
    sink_pages: int,
    recent_pages: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(checkpoints) != 1:
        raise ValueError("layer scoring requires exactly one independently captured checkpoint")
    expected_positions = tuple(checkpoint.token_count - 1 for checkpoint in checkpoints)
    if capture.query_positions != expected_positions:
        raise ValueError("captured query positions differ from the case checkpoints")
    if capture.queries.ndim != 3 or capture.keys.ndim != 3 or capture.values.ndim != 3:
        raise ValueError("captured Q/K/V shapes are invalid")
    query_heads = int(capture.queries.shape[0])
    kv_heads = int(capture.keys.shape[0])
    if capture.queries.shape[1] != 1 or capture.keys.shape[1] != checkpoints[0].token_count:
        raise ValueError("captured Q/K/V must come from exactly one complete causal prefix")
    if capture.values.shape != capture.keys.shape or query_heads % kv_heads:
        raise ValueError("captured Q/K/V do not satisfy grouped-query attention")
    if capture.queries.shape[2] != capture.keys.shape[2]:
        raise ValueError("captured Q and K head dimensions differ")
    groups = query_heads // kv_heads
    head_dim = int(capture.keys.shape[2])
    element_bytes = capture.keys.element_size()
    budget_records: list[dict[str, Any]] = []
    minimum_mass_records: list[dict[str, Any]] = []
    physical_records: list[dict[str, Any]] = []

    for query_slot, checkpoint in enumerate(checkpoints):
        token_count = checkpoint.token_count
        selections: dict[tuple[int, float, int], set[int]] = {}
        for query_head in range(query_heads):
            kv_head = query_head // groups
            attention_weights = exact_attention_weights(
                capture.queries[query_head, query_slot],
                capture.keys[kv_head, :token_count],
            )
            for page_size in page_sizes:
                curve = build_exact_page_oracle_curve_from_weights(
                    attention_weights,
                    capture.values[kv_head, :token_count],
                    page_size=page_size,
                    sink_pages=sink_pages,
                    recent_pages=recent_pages,
                    top_k=top_k,
                )
                page_mass_sha256 = _tensor_sha256(curve.page_masses)
                for target_mass in _MASS_TARGETS:
                    required_pages = curve.minimum_candidate_pages_for_mass(target_mass)
                    required = curve.metrics_for_pages(
                        required_pages,
                        support_token_spans=checkpoint.support_token_spans,
                    )
                    minimum_mass_records.append(
                        {
                            "case_id": case_id,
                            "layer": capture.layer_index,
                            "query_position": token_count - 1,
                            "token_count": token_count,
                            "query_head": query_head,
                            "kv_head": kv_head,
                            "page_size": page_size,
                            "target_mass": target_mass,
                            "candidate_pages": required.candidate_pages,
                            "candidate_page_fraction": required.candidate_page_fraction,
                            "candidate_tokens": required.candidate_tokens,
                            "candidate_token_fraction": required.candidate_token_fraction,
                            "retained_attention_mass": required.retained_attention_mass,
                        }
                    )
                for fraction in candidate_fractions:
                    metrics = curve.metrics_for_fraction(
                        fraction,
                        support_token_spans=checkpoint.support_token_spans,
                    )
                    record = {
                        "case_id": case_id,
                        "layer": capture.layer_index,
                        "query_position": token_count - 1,
                        "token_count": token_count,
                        "query_head": query_head,
                        "kv_head": kv_head,
                        "page_size": page_size,
                        "requested_candidate_fraction": fraction,
                        "page_mass_sha256": page_mass_sha256,
                        **asdict(metrics),
                    }
                    budget_records.append(record)
                    selections[(page_size, fraction, query_head)] = set(metrics.selected_page_ids)

        for page_size in page_sizes:
            total_pages = math.ceil(token_count / page_size)
            dense_layer_payload_bytes = 2 * kv_heads * head_dim * token_count * element_bytes
            support_pages = _support_page_ids(
                checkpoint.support_token_spans,
                token_count=token_count,
                page_size=page_size,
            )
            for fraction in candidate_fractions:
                kv_unions: list[dict[str, Any]] = []
                layer_union: set[int] = set()
                for kv_head in range(kv_heads):
                    query_head_ids = range(kv_head * groups, (kv_head + 1) * groups)
                    union = set().union(
                        *(
                            selections[(page_size, fraction, query_head)]
                            for query_head in query_head_ids
                        )
                    )
                    layer_union.update(union)
                    selected_tokens = _selected_token_count(
                        union,
                        token_count=token_count,
                        page_size=page_size,
                    )
                    kv_unions.append(
                        {
                            "kv_head": kv_head,
                            "page_ids": sorted(union),
                            "page_fraction": len(union) / total_pages,
                            "token_fraction": selected_tokens / token_count,
                            "payload_bytes": 2 * head_dim * selected_tokens * element_bytes,
                        }
                    )
                layer_tokens = _selected_token_count(
                    layer_union,
                    token_count=token_count,
                    page_size=page_size,
                )
                addresses = tuple(
                    AtxyPageAddress.for_layer_page(capture.layer_index, page_id).text
                    for page_id in sorted(layer_union)
                )
                physical_records.append(
                    {
                        "case_id": case_id,
                        "layer": capture.layer_index,
                        "query_position": token_count - 1,
                        "token_count": token_count,
                        "page_size": page_size,
                        "requested_candidate_fraction": fraction,
                        "kv_head_unions": kv_unions,
                        "layer_union_page_ids": sorted(layer_union),
                        "layer_union_atxy_addresses": list(addresses),
                        "layer_union_atxy_address_sha256": _canonical_sha256(addresses),
                        "layer_union_pages": len(layer_union),
                        "total_pages": total_pages,
                        "layer_union_physical_fetch_fraction": layer_tokens / token_count,
                        "layer_union_payload_bytes": (
                            2 * kv_heads * head_dim * layer_tokens * element_bytes
                        ),
                        "dense_layer_payload_bytes": dense_layer_payload_bytes,
                        "support_page_recall": (
                            len(layer_union.intersection(support_pages)) / len(support_pages)
                        ),
                    }
                )
    return budget_records, minimum_mass_records, physical_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure an all-layer/head exact-attention page-oracle ceiling over exact Qwen KV."
        )
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--page-size", type=int, nargs="+", default=[64, 256])
    parser.add_argument(
        "--candidate-fraction",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.35, 0.5],
    )
    parser.add_argument("--sink-pages", type=int, default=1)
    parser.add_argument("--recent-pages", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8419)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if any(value <= 0 for value in args.page_size) or len(set(args.page_size)) != len(
        args.page_size
    ):
        parser.error("--page-size values must be positive and unique")
    try:
        fractions = tuple(
            _fraction(value, "candidate fraction") for value in args.candidate_fraction
        )
    except ValueError as error:
        parser.error(str(error))
    if len(set(fractions)) != len(fractions):
        parser.error("--candidate-fraction values must be unique")
    try:
        sink_pages = _non_negative_int(args.sink_pages, "sink_pages")
        recent_pages = _non_negative_int(args.recent_pages, "recent_pages")
        top_k = _positive_int(args.top_k, "top_k")
        seed = _non_negative_int(args.seed, "seed")
        dataset_snapshot = _load_dataset_manifest(args.dataset_manifest)
    except ValueError as error:
        parser.error(str(error))

    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit("Install molor[hf] for the ATXY page-oracle lab") from error

    implementation_preflight = _implementation_identity()
    torch.manual_seed(seed)
    model_path = Path(args.model).expanduser().resolve()
    try:
        model_stack_preflight = _local_model_stack_identity(model_path)
    except ValueError as error:
        parser.error(str(error))
    model_id = str(model_path)
    source_config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=True,
    )
    layer_count = int(source_config.num_hidden_layers)
    query_heads = int(source_config.num_attention_heads)
    kv_heads = int(source_config.num_key_value_heads)
    head_dim = int(source_config.head_dim)
    if layer_count > 36:
        parser.error("model has more layers than the A field can address")
    if query_heads % kv_heads:
        parser.error("model query heads are not divisible by KV heads")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )
    case_tokens: dict[str, Tensor] = {}
    for case in dataset_snapshot.cases:
        try:
            tokens = _token_ids(tokenizer, case)
            _validate_context_length(
                int(tokens.numel()), int(source_config.max_position_embeddings)
            )
        except ValueError as error:
            parser.error(str(error))
        case_tokens[case.case_id] = tokens

    print(f"Loading {model_id} in bfloat16 for unchanged dense capture", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    runtime_device = str(model.get_input_embeddings().weight.device)
    budget_records: list[dict[str, Any]] = []
    minimum_mass_records: list[dict[str, Any]] = []
    physical_records: list[dict[str, Any]] = []
    case_metadata: list[dict[str, Any]] = []
    layer_indices = tuple(range(layer_count))
    for case in dataset_snapshot.cases:
        tokens = case_tokens.pop(case.case_id)
        prefix_metadata: list[dict[str, Any]] = []
        for checkpoint, captures, capture_seconds in _capture_independent_prefixes(
            model,
            input_ids=tokens,
            layer_indices=layer_indices,
            checkpoints=case.queries,
        ):
            print(
                f"Captured independent prefix {case.case_id}@{checkpoint.token_count:,}: "
                f"{layer_count} layers, {query_heads} query heads",
                flush=True,
            )
            before_budget = len(budget_records)
            before_minimum = len(minimum_mass_records)
            before_physical = len(physical_records)
            scoring_started = time.perf_counter()
            while captures:
                capture = captures.pop(0)
                if (
                    capture.keys.dtype is not torch.bfloat16
                    or capture.values.dtype is not torch.bfloat16
                ):
                    raise RuntimeError("development oracle requires exact BF16 captured K/V")
                layer_budget, layer_minimum, layer_physical = _evaluate_captured_layer(
                    capture,
                    case_id=case.case_id,
                    checkpoints=(checkpoint,),
                    page_sizes=tuple(args.page_size),
                    candidate_fractions=fractions,
                    sink_pages=sink_pages,
                    recent_pages=recent_pages,
                    top_k=top_k,
                )
                budget_records.extend(layer_budget)
                minimum_mass_records.extend(layer_minimum)
                physical_records.extend(layer_physical)
                del capture
            scoring_seconds = time.perf_counter() - scoring_started
            observed_counts = (
                len(budget_records) - before_budget,
                len(minimum_mass_records) - before_minimum,
                len(physical_records) - before_physical,
            )
            expected_counts = (
                layer_count * query_heads * len(args.page_size) * len(fractions),
                layer_count * query_heads * len(args.page_size) * len(_MASS_TARGETS),
                layer_count * len(args.page_size) * len(fractions),
            )
            if observed_counts != expected_counts:
                raise RuntimeError(
                    "independent prefix produced an incomplete oracle matrix: "
                    f"expected {expected_counts}, got {observed_counts}"
                )
            prefix_metadata.append(
                {
                    "token_count": checkpoint.token_count,
                    "query_position": checkpoint.token_count - 1,
                    "capture_seconds": capture_seconds,
                    "cpu_scoring_seconds": scoring_seconds,
                    "matrix_counts": {
                        "budget_records": observed_counts[0],
                        "minimum_mass_records": observed_counts[1],
                        "physical_union_records": observed_counts[2],
                    },
                }
            )
            gc.collect()
        case_metadata.append(
            {
                "id": case.case_id,
                "source": str(case.path),
                "source_sha256": case.source_sha256,
                "source_size_bytes": case.source_size_bytes,
                "token_count": case.token_count,
                "token_ids_sha256": _token_sha256(tokens),
                "queries": [asdict(query) for query in case.queries],
                "independent_prefix_captures": prefix_metadata,
            }
        )
        del tokens
        gc.collect()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _verify_dataset_snapshot(dataset_snapshot)
    model_stack_postflight = _local_model_stack_identity(model_path)
    if model_stack_postflight != model_stack_preflight:
        raise RuntimeError("local model/tokenizer/config/weight bytes changed during the run")
    implementation_postflight = _implementation_identity()
    if implementation_postflight != implementation_preflight:
        raise RuntimeError("oracle implementation changed during the run; refusing evidence")

    report = {
        "schema_version": _REPORT_SCHEMA,
        "status": (
            "development-only exact-payload oracle; current-query dense-attention leak; "
            "no selector, compression, live cache, promotion, or context-extension claim"
        ),
        "promotion_eligible": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "oracle_definition": (
            "Mandatory sink/recent pages plus remaining pages ranked by exact current-query "
            "attention mass; exact BF16 K/V is retained for candidate attention."
        ),
        "model": {
            "id": model_id,
            "artifact_stack": model_stack_preflight,
            "postflight_verified": True,
            "config": {
                "layers": layer_count,
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "max_position_embeddings": int(source_config.max_position_embeddings),
            },
        },
        "dataset": {
            "manifest": str(dataset_snapshot.manifest_path),
            "manifest_sha256": dataset_snapshot.manifest_sha256,
            "manifest_size_bytes": dataset_snapshot.manifest_size_bytes,
            "postflight_verified": True,
            "cases": case_metadata,
        },
        "configuration": {
            "all_layers": True,
            "all_query_heads": True,
            "page_sizes": list(args.page_size),
            "candidate_fractions": list(fractions),
            "sink_pages": sink_pages,
            "recent_pages": recent_pages,
            "top_k": top_k,
            "minimum_mass_targets": list(_MASS_TARGETS),
            "seed": seed,
            "cache_dtype": "bfloat16",
            "scoring_dtype": "float32-cpu",
            "capture_protocol": "independent-causal-prefix-forward-per-checkpoint",
        },
        "atxy": {
            "meaning": "exact physical layer/page identity only",
            "page_capacity_per_layer": {
                str(page_size): token_capacity_per_layer(page_size) for page_size in args.page_size
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "capture_device": runtime_device,
            "capture_attention_implementation": "sdpa",
        },
        "implementation": {
            "preflight": implementation_preflight,
            "postflight_verified": True,
        },
        "summary": {
            "budget_groups": _summarize_budget_records(budget_records),
            "minimum_mass_groups": _summarize_minimum_mass_records(minimum_mass_records),
            "physical_union_groups": _summarize_physical_records(physical_records),
        },
        "records": budget_records,
        "minimum_mass_records": minimum_mass_records,
        "physical_union_records": physical_records,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.output_directory.expanduser().resolve()
        / f"atxy-page-oracle-development-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(output, _seal_report(report))
    print(
        f"Wrote {len(budget_records):,} exact-payload head records and "
        f"{len(physical_records):,} physical-union records to {output}",
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
