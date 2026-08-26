"""Build versioned, granular expert views from the pinned Qwen3.8 distillation set.

This materializer intentionally leaves ``qwen38_distillation_corpus.py`` and its v1 outputs
untouched.  It assigns every technically valid non-tool record to one of six source-defined
leaf experts, emits independent direct and explicit-trace streams, and separates sequences
into tokenizer-measured short and long tiers without truncation.

Admission is based only on technical properties.  License metadata is neither read nor used
to admit or reject a record.  Outputs contain only ``messages``; all other files contain
aggregate counts, hashes, and provenance rather than examples or row-level metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from molor.experimental import qwen38_distillation_corpus as foundation


SCHEMA_VERSION = "molor.qwen38-distillation-experts.v2"
MIXTURE_SCHEMA_VERSION = "molor.expert-mixture.v1"
DEFAULT_SEED = "molor-qwen38-distillation-experts-v2"
DEFAULT_SHORT_MAX_LENGTH = 1024
DEFAULT_LONG_MAX_LENGTH = 4096

LEAF_QUANTITATIVE_MATH = "quantitative_math"
LEAF_FORMAL_MATH = "formal_math"
LEAF_SCIENCE_REASONING = "science_reasoning"
LEAF_COMMONSENSE_LOGIC = "commonsense_logic"
LEAF_GENERAL_CODE_GENERATION = "general_code_generation"
LEAF_STRICT_INSTRUCTION_FOLLOWING = "strict_instruction_following"
LEAF_NAMES = (
    LEAF_QUANTITATIVE_MATH,
    LEAF_FORMAL_MATH,
    LEAF_SCIENCE_REASONING,
    LEAF_COMMONSENSE_LOGIC,
    LEAF_GENERAL_CODE_GENERATION,
    LEAF_STRICT_INSTRUCTION_FOLLOWING,
)
SPLIT_NAMES = foundation.SPLIT_NAMES

MODE_DIRECT = "direct"
MODE_TRACE = "trace"
TIER_SHORT = "short"
TIER_LONG = "long"

_SOURCES_BY_LEAF: dict[str, frozenset[str]] = {
    LEAF_QUANTITATIVE_MATH: frozenset({"MetaMathQA", "OrcaMath", "gsm8k"}),
    LEAF_FORMAL_MATH: frozenset({"NuminaMath-CoT"}),
    LEAF_SCIENCE_REASONING: frozenset({"SciQ", "QASC", "OpenBookQA"}),
    LEAF_COMMONSENSE_LOGIC: frozenset({"CommonsenseQA", "ARC-Easy", "ARC-Challenge"}),
    LEAF_GENERAL_CODE_GENERATION: frozenset({"Evol-Code", "CodeAlpaca", "MBPP", "HumanEval"}),
    LEAF_STRICT_INSTRUCTION_FOLLOWING: frozenset({"tulu-3", "Dolly", "IFEval"}),
}
_EXPECTED_DOMAIN_BY_LEAF = {
    LEAF_QUANTITATIVE_MATH: "math",
    LEAF_FORMAL_MATH: "math",
    LEAF_SCIENCE_REASONING: "reasoning",
    LEAF_COMMONSENSE_LOGIC: "reasoning",
    LEAF_GENERAL_CODE_GENERATION: "code",
    LEAF_STRICT_INSTRUCTION_FOLLOWING: "instruction",
}
_SOURCE_TO_LEAF = {source: leaf for leaf, sources in _SOURCES_BY_LEAF.items() for source in sources}


@dataclass(frozen=True, slots=True)
class Candidate:
    prompt_group: str
    family_group: str
    split: str
    leaf: str
    source: str
    domain: str
    quality_score: float
    content_hash: str
    order_rank: str
    direct_payload: str
    direct_token_count: int
    direct_tier: str
    trace_payload: str | None
    trace_token_count: int | None
    trace_tier: str | None
    trace_status: str


@dataclass(frozen=True, slots=True)
class OutputRecord:
    prompt_group: str
    family_group: str
    leaf: str
    source: str
    domain: str
    order_rank: str
    payload: str
    token_count: int


def leaf_for_source(source: str) -> str | None:
    """Return the deterministic leaf for one source label, including ``MATH/*``."""

    if source.startswith("MATH/") and len(source) > len("MATH/"):
        return LEAF_FORMAL_MATH
    return _SOURCE_TO_LEAF.get(source)


def _source_selectors() -> dict[str, list[str]]:
    selectors = {leaf: sorted(sources) for leaf, sources in _SOURCES_BY_LEAF.items()}
    selectors[LEAF_FORMAL_MATH].append("MATH/*")
    return selectors


def _stream_name(mode: str, tier: str) -> str:
    return f"{mode}-{tier}"


def _stream_names(*, include_long: bool) -> tuple[str, ...]:
    tiers = (TIER_SHORT, TIER_LONG) if include_long else (TIER_SHORT,)
    return tuple(_stream_name(mode, tier) for mode in (MODE_DIRECT, MODE_TRACE) for tier in tiers)


def _length_tier(
    token_count: int, *, short_max_length: int, long_max_length: int | None
) -> str | None:
    if token_count <= short_max_length:
        return TIER_SHORT
    if long_max_length is not None and token_count <= long_max_length:
        return TIER_LONG
    return None


def _create_candidate_table(connection: Any) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE candidates (
            prompt_group TEXT PRIMARY KEY,
            family_group TEXT NOT NULL,
            split TEXT NOT NULL,
            leaf TEXT NOT NULL,
            source TEXT NOT NULL,
            domain TEXT NOT NULL,
            quality_score REAL NOT NULL,
            content_hash TEXT NOT NULL,
            order_rank TEXT NOT NULL,
            direct_payload TEXT NOT NULL,
            direct_token_count INTEGER NOT NULL,
            direct_tier TEXT NOT NULL,
            trace_payload TEXT,
            trace_token_count INTEGER,
            trace_tier TEXT,
            trace_status TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _candidate_values(candidate: Candidate) -> tuple[object, ...]:
    return (
        candidate.prompt_group,
        candidate.family_group,
        candidate.split,
        candidate.leaf,
        candidate.source,
        candidate.domain,
        candidate.quality_score,
        candidate.content_hash,
        candidate.order_rank,
        candidate.direct_payload,
        candidate.direct_token_count,
        candidate.direct_tier,
        candidate.trace_payload,
        candidate.trace_token_count,
        candidate.trace_tier,
        candidate.trace_status,
    )


def _candidate_from_row(row: Sequence[Any]) -> Candidate:
    return Candidate(
        prompt_group=str(row[0]),
        family_group=str(row[1]),
        split=str(row[2]),
        leaf=str(row[3]),
        source=str(row[4]),
        domain=str(row[5]),
        quality_score=float(row[6]),
        content_hash=str(row[7]),
        order_rank=str(row[8]),
        direct_payload=str(row[9]),
        direct_token_count=int(row[10]),
        direct_tier=str(row[11]),
        trace_payload=None if row[12] is None else str(row[12]),
        trace_token_count=None if row[13] is None else int(row[13]),
        trace_tier=None if row[14] is None else str(row[14]),
        trace_status=str(row[15]),
    )


def _upsert_candidate(connection: Any, candidate: Candidate) -> Candidate | None:
    """Insert a candidate and return the deterministic duplicate loser, if one exists."""

    prior_row = connection.execute(
        "SELECT * FROM candidates WHERE prompt_group = ?", (candidate.prompt_group,)
    ).fetchone()
    if prior_row is None:
        connection.execute(
            "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _candidate_values(candidate),
        )
        return None
    prior = _candidate_from_row(prior_row)
    candidate_wins = candidate.quality_score > prior.quality_score or (
        candidate.quality_score == prior.quality_score
        and candidate.content_hash < prior.content_hash
    )
    if not candidate_wins:
        return candidate
    connection.execute(
        """
        UPDATE candidates SET
            family_group = ?, split = ?, leaf = ?, source = ?, domain = ?,
            quality_score = ?, content_hash = ?, order_rank = ?, direct_payload = ?,
            direct_token_count = ?, direct_tier = ?, trace_payload = ?,
            trace_token_count = ?, trace_tier = ?, trace_status = ?
        WHERE prompt_group = ?
        """,
        (
            candidate.family_group,
            candidate.split,
            candidate.leaf,
            candidate.source,
            candidate.domain,
            candidate.quality_score,
            candidate.content_hash,
            candidate.order_rank,
            candidate.direct_payload,
            candidate.direct_token_count,
            candidate.direct_tier,
            candidate.trace_payload,
            candidate.trace_token_count,
            candidate.trace_tier,
            candidate.trace_status,
            candidate.prompt_group,
        ),
    )
    return prior


def _token_stats(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
            "total": 0,
        }

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[max(index, 0)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
        "total": sum(ordered),
    }


def _nested_counts(counter: Mapping[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (outer, inner), count in sorted(counter.items()):
        nested[outer][inner] = count
    return dict(nested)


def _output_views(
    candidates: Sequence[Candidate], *, include_long: bool
) -> dict[str, dict[str, dict[str, list[OutputRecord]]]]:
    streams = _stream_names(include_long=include_long)
    views = {
        leaf: {stream: {split: [] for split in SPLIT_NAMES} for stream in streams}
        for leaf in LEAF_NAMES
    }
    for candidate in candidates:
        direct_stream = _stream_name(MODE_DIRECT, candidate.direct_tier)
        views[candidate.leaf][direct_stream][candidate.split].append(
            OutputRecord(
                candidate.prompt_group,
                candidate.family_group,
                candidate.leaf,
                candidate.source,
                candidate.domain,
                candidate.order_rank,
                candidate.direct_payload,
                candidate.direct_token_count,
            )
        )
        if candidate.trace_payload is not None:
            assert candidate.trace_token_count is not None and candidate.trace_tier is not None
            trace_stream = _stream_name(MODE_TRACE, candidate.trace_tier)
            views[candidate.leaf][trace_stream][candidate.split].append(
                OutputRecord(
                    candidate.prompt_group,
                    candidate.family_group,
                    candidate.leaf,
                    candidate.source,
                    candidate.domain,
                    candidate.order_rank,
                    candidate.trace_payload,
                    candidate.trace_token_count,
                )
            )
    for leaf_streams in views.values():
        for splits in leaf_streams.values():
            for rows in splits.values():
                rows.sort(key=lambda row: (row.order_rank, row.prompt_group))
    return views


def _overlap_receipt(
    views: Mapping[str, Mapping[str, Mapping[str, Sequence[OutputRecord]]]],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    for leaf, streams in views.items():
        receipt[leaf] = {}
        for stream, splits in streams.items():
            prompt_sets = {
                split: {row.prompt_group for row in rows} for split, rows in splits.items()
            }
            family_sets = {
                split: {row.family_group for row in rows} for split, rows in splits.items()
            }
            prompt_overlap = {
                "train_validation": len(prompt_sets["train"] & prompt_sets["validation"]),
                "train_test": len(prompt_sets["train"] & prompt_sets["test"]),
                "validation_test": len(prompt_sets["validation"] & prompt_sets["test"]),
            }
            family_overlap = {
                "train_validation": len(family_sets["train"] & family_sets["validation"]),
                "train_test": len(family_sets["train"] & family_sets["test"]),
                "validation_test": len(family_sets["validation"] & family_sets["test"]),
            }
            if any(prompt_overlap.values()) or any(family_overlap.values()):
                raise AssertionError(f"prompt or split-family leakage detected in {leaf}/{stream}")
            receipt[leaf][stream] = {
                "prompt_group": {**prompt_overlap, "zero_overlap": True},
                "split_family": {**family_overlap, "zero_overlap": True},
            }
    return receipt


def _stream_contracts(
    *, short_max_length: int, long_max_length: int | None
) -> dict[str, dict[str, object]]:
    contracts: dict[str, dict[str, object]] = {}
    for mode in (MODE_DIRECT, MODE_TRACE):
        mode_fields: dict[str, object]
        if mode == MODE_DIRECT:
            mode_fields = {
                "enable_thinking": False,
                "assistant_target": "<think>\\n\\n</think>\\n\\n{final}",
            }
        else:
            mode_fields = {
                "enable_thinking": True,
                "assistant_target": "<think>\\n{trace}\\n</think>\\n\\n{final}",
            }
        contracts[_stream_name(mode, TIER_SHORT)] = {
            **mode_fields,
            "minimum_rendered_tokens": 1,
            "maximum_rendered_tokens": short_max_length,
            "truncation": False,
        }
        if long_max_length is not None:
            contracts[_stream_name(mode, TIER_LONG)] = {
                **mode_fields,
                "minimum_rendered_tokens": short_max_length + 1,
                "maximum_rendered_tokens": long_max_length,
                "truncation": False,
            }
    return contracts


def _write_outputs(
    output_dir: Path,
    *,
    views: Mapping[str, Mapping[str, Mapping[str, Sequence[OutputRecord]]]],
    manifest: dict[str, Any],
    short_max_length: int,
    long_max_length: int | None,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        receipts: dict[str, dict[str, Any]] = {}
        for leaf in LEAF_NAMES:
            receipts[leaf] = {}
            for stream, splits in views[leaf].items():
                stream_dir = stage / leaf / stream
                stream_dir.mkdir(parents=True)
                receipts[leaf][stream] = {}
                for split in SPLIT_NAMES:
                    path = stream_dir / f"{split}.jsonl"
                    with path.open("w", encoding="utf-8", newline="\n") as output:
                        for row in splits[split]:
                            output.write(row.payload)
                            output.write("\n")
                    receipts[leaf][stream][split] = {
                        "path": path.relative_to(stage).as_posix(),
                        "sha256": foundation._sha256_file(path),
                        "bytes": path.stat().st_size,
                        "records": len(splits[split]),
                        "rendered_tokens": sum(row.token_count for row in splits[split]),
                    }

        contracts = _stream_contracts(
            short_max_length=short_max_length, long_max_length=long_max_length
        )
        mixture = {
            "schema_version": MIXTURE_SCHEMA_VERSION,
            "source_schema_version": SCHEMA_VERSION,
            "leaf_order": list(LEAF_NAMES),
            "leaf_source_selectors": _source_selectors(),
            "stream_contracts": contracts,
            "leaves": {
                leaf: {
                    "expected_domain": _EXPECTED_DOMAIN_BY_LEAF[leaf],
                    "streams": receipts[leaf],
                }
                for leaf in LEAF_NAMES
            },
            "contains_raw_examples": False,
            "record_metadata_policy": "aggregate-and-hash-only",
        }
        mixture_path = stage / "mixture-manifest.json"
        mixture_path.write_text(
            json.dumps(mixture, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest["outputs"] = receipts
        manifest["sidecars"] = {
            "mixture_manifest": {
                "path": mixture_path.relative_to(stage).as_posix(),
                "sha256": foundation._sha256_file(mixture_path),
                "bytes": mixture_path.stat().st_size,
            }
        }
        manifest_path = stage / "quality-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        sum_paths = sorted(
            path for path in stage.rglob("*") if path.is_file() and path.name != "sha256sums.txt"
        )
        sums_path = stage / "sha256sums.txt"
        sums_path.write_text(
            "".join(
                f"{foundation._sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
                for path in sum_paths
            ),
            encoding="ascii",
            newline="\n",
        )
        os.replace(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return target / "quality-manifest.json"


def materialize_qwen38_distillation_experts(
    *,
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    tokenizer: foundation.ChatTemplateTokenizer,
    tokenizer_path: str | Path,
    expected_input_hashes: Mapping[str | Path, str],
    benchmark_suite_paths: Sequence[str | Path],
    short_max_length: int = DEFAULT_SHORT_MAX_LENGTH,
    long_max_length: int | None = DEFAULT_LONG_MAX_LENGTH,
    seed: str = DEFAULT_SEED,
    required_teacher_model: str = foundation.DEFAULT_TEACHER_MODEL,
    source_revision: str = foundation.DEFAULT_SOURCE_REVISION,
) -> Path:
    """Materialize six leaf-expert corpora and return the aggregate manifest path."""

    if short_max_length < 8 or short_max_length > DEFAULT_SHORT_MAX_LENGTH:
        raise ValueError("short_max_length must be between 8 and 1024")
    if long_max_length is not None and not short_max_length < long_max_length <= 4096:
        raise ValueError("long_max_length must be greater than short_max_length and at most 4096")
    if not seed:
        raise ValueError("seed must be a non-empty string")
    if not required_teacher_model:
        raise ValueError("required_teacher_model must be non-empty")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("source_revision must be a full lowercase Git SHA-1")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional local tooling
        raise RuntimeError("pyarrow is required to materialize the Parquet corpus") from exc
    import sqlite3

    paths, input_receipts = foundation._verified_inputs(input_paths, expected_input_hashes)
    benchmark_index = foundation._load_benchmark_index(benchmark_suite_paths)
    tokenizer_receipt = foundation._tokenizer_identity(tokenizer_path)
    tokenizer_receipt["chat_template_modes"] = {
        MODE_DIRECT: {"enable_thinking": False},
        MODE_TRACE: {"enable_thinking": True},
    }

    input_by_source: Counter[str] = Counter()
    input_by_domain: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    rejected_by_source: Counter[tuple[str, str]] = Counter()
    rejected_by_domain: Counter[tuple[str, str]] = Counter()
    quarantined: Counter[str] = Counter()
    quarantined_by_source: Counter[tuple[str, str]] = Counter()
    quarantined_by_domain: Counter[tuple[str, str]] = Counter()
    admitted_by_leaf: Counter[str] = Counter()
    admitted_by_source: Counter[str] = Counter()
    admitted_by_domain: Counter[str] = Counter()
    duplicates_by_leaf: Counter[str] = Counter()
    duplicates_by_source: Counter[str] = Counter()
    duplicates_by_domain: Counter[str] = Counter()
    contamination_by_source: Counter[tuple[str, str]] = Counter()
    total_records = 0
    admitted_before_dedup = 0

    def reject(reason: str, source: str, domain: str) -> None:
        rejected[reason] += 1
        rejected_by_source[(source, reason)] += 1
        rejected_by_domain[(domain, reason)] += 1

    def quarantine(reason: str, source: str, domain: str) -> None:
        quarantined[reason] += 1
        quarantined_by_source[(source, reason)] += 1
        quarantined_by_domain[(domain, reason)] += 1

    with tempfile.TemporaryDirectory(prefix="molor-qwen38-distillation-experts-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "candidates.sqlite3")
        try:
            _create_candidate_table(connection)
            for path, receipt in zip(paths, input_receipts, strict=True):
                parquet = pq.ParquetFile(path)
                foundation._validate_parquet_schema(parquet.schema_arrow, pa)
                receipt["records"] = parquet.metadata.num_rows
                scanned = 0
                for batch in parquet.iter_batches(
                    batch_size=2048, columns=list(foundation._EXPECTED_COLUMNS)
                ):
                    for record in batch.to_pylist():
                        scanned += 1
                        total_records += 1
                        source_label, domain_label = foundation._record_labels(record)
                        input_by_source[source_label] += 1
                        input_by_domain[domain_label] += 1
                        parsed, reason = foundation._parse_record(
                            record, required_teacher_model=required_teacher_model
                        )
                        if parsed is None:
                            assert reason is not None
                            reject(reason, source_label, domain_label)
                            continue
                        if parsed.domain == "tool_use":
                            quarantine("tool_use", parsed.source, parsed.domain)
                            continue

                        leaf = leaf_for_source(parsed.source)
                        if leaf is None:
                            reject("unmapped_source", parsed.source, parsed.domain)
                            continue
                        if parsed.domain != _EXPECTED_DOMAIN_BY_LEAF[leaf]:
                            reject("source_domain_mismatch", parsed.source, parsed.domain)
                            continue

                        prompt_group = foundation.prompt_group_id(parsed.prompt)
                        family_group = foundation.split_family_id(parsed.prompt)
                        exact_overlap = prompt_group in benchmark_index.prompt_groups
                        family_overlap = family_group in benchmark_index.family_groups
                        if exact_overlap:
                            contamination_by_source[(parsed.source, "exact_prompt_overlap")] += 1
                        if family_overlap:
                            contamination_by_source[(parsed.source, "split_family_overlap")] += 1
                        if exact_overlap or family_overlap:
                            reject(
                                "benchmark_exact_prompt_overlap"
                                if exact_overlap
                                else "benchmark_split_family_overlap",
                                parsed.source,
                                parsed.domain,
                            )
                            continue

                        direct_target = f"<think>\n\n</think>\n\n{parsed.final}"
                        direct_messages = foundation._messages(parsed.prompt, direct_target)
                        _, direct_ids = foundation.render_final_token_ids(
                            tokenizer, direct_messages
                        )
                        direct_tier = _length_tier(
                            len(direct_ids),
                            short_max_length=short_max_length,
                            long_max_length=long_max_length,
                        )
                        if direct_tier is None:
                            reject("direct_rendered_above_max", parsed.source, parsed.domain)
                            continue

                        trace_payload: str | None = None
                        trace_token_count: int | None = None
                        trace_tier: str | None = None
                        trace_status = "no_explicit_trace"
                        if parsed.trace is not None:
                            trace_target = f"<think>\n{parsed.trace}\n</think>\n\n{parsed.final}"
                            trace_messages = foundation._messages(parsed.prompt, trace_target)
                            _, trace_ids = foundation.render_trace_token_ids(
                                tokenizer, trace_messages
                            )
                            trace_tier = _length_tier(
                                len(trace_ids),
                                short_max_length=short_max_length,
                                long_max_length=long_max_length,
                            )
                            if trace_tier is None:
                                trace_status = "rendered_above_max"
                            else:
                                trace_status = trace_tier
                                trace_payload = foundation._payload(trace_messages)
                                trace_token_count = len(trace_ids)

                        canonical_content = json.dumps(
                            {
                                "domain": parsed.domain,
                                "leaf": leaf,
                                "source": parsed.source,
                                "teacher_model": parsed.teacher_model,
                                "prompt": parsed.prompt,
                                "trace": parsed.trace,
                                "final": parsed.final,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        content_hash = foundation._sha256_bytes(canonical_content.encode("utf-8"))
                        order_rank = foundation._sha256_bytes(
                            f"{seed}\0order\0{prompt_group}\0{content_hash}".encode("utf-8")
                        )
                        candidate = Candidate(
                            prompt_group=prompt_group,
                            family_group=family_group,
                            split=foundation.split_for_family(family_group, seed=seed),
                            leaf=leaf,
                            source=parsed.source,
                            domain=parsed.domain,
                            quality_score=parsed.quality_score,
                            content_hash=content_hash,
                            order_rank=order_rank,
                            direct_payload=foundation._payload(direct_messages),
                            direct_token_count=len(direct_ids),
                            direct_tier=direct_tier,
                            trace_payload=trace_payload,
                            trace_token_count=trace_token_count,
                            trace_tier=trace_tier,
                            trace_status=trace_status,
                        )
                        loser = _upsert_candidate(connection, candidate)
                        admitted_before_dedup += 1
                        admitted_by_leaf[leaf] += 1
                        admitted_by_source[parsed.source] += 1
                        admitted_by_domain[parsed.domain] += 1
                        if loser is not None:
                            duplicates_by_leaf[loser.leaf] += 1
                            duplicates_by_source[loser.source] += 1
                            duplicates_by_domain[loser.domain] += 1
                if scanned != parquet.metadata.num_rows:
                    raise AssertionError(f"Parquet row-count mismatch while scanning {path}")
                connection.commit()
            rows = connection.execute(
                "SELECT * FROM candidates ORDER BY order_rank, prompt_group"
            ).fetchall()
            candidates = [_candidate_from_row(row) for row in rows]
        finally:
            connection.close()

    include_long = long_max_length is not None
    views = _output_views(candidates, include_long=include_long)
    overlap_receipt = _overlap_receipt(views)

    global_prompt_splits: dict[str, str] = {}
    global_family_splits: dict[str, str] = {}
    for candidate in candidates:
        if (
            global_prompt_splits.setdefault(candidate.prompt_group, candidate.split)
            != candidate.split
        ):
            raise AssertionError("a prompt group crossed splits")
        if (
            global_family_splits.setdefault(candidate.family_group, candidate.split)
            != candidate.split
        ):
            raise AssertionError("a split family crossed splits")

    canonical_by_leaf = Counter(candidate.leaf for candidate in candidates)
    canonical_by_source = Counter(candidate.source for candidate in candidates)
    canonical_by_domain = Counter(candidate.domain for candidate in candidates)
    canonical_by_split = Counter(candidate.split for candidate in candidates)
    trace_status_by_leaf: Counter[tuple[str, str]] = Counter(
        (candidate.leaf, candidate.trace_status) for candidate in candidates
    )
    trace_status_by_source: Counter[tuple[str, str]] = Counter(
        (candidate.source, candidate.trace_status) for candidate in candidates
    )
    trace_status_by_domain: Counter[tuple[str, str]] = Counter(
        (candidate.domain, candidate.trace_status) for candidate in candidates
    )

    counts_by_leaf: dict[str, Any] = {}
    direct_emissions = 0
    trace_emissions = 0
    for leaf in LEAF_NAMES:
        counts_by_leaf[leaf] = {"records": canonical_by_leaf[leaf], "streams": {}}
        for stream, splits in views[leaf].items():
            all_rows = [row for split_rows in splits.values() for row in split_rows]
            mode = stream.split("-", 1)[0]
            if mode == MODE_DIRECT:
                direct_emissions += len(all_rows)
            else:
                trace_emissions += len(all_rows)
            counts_by_leaf[leaf]["streams"][stream] = {
                "records": len(all_rows),
                "by_split": {split: len(splits[split]) for split in SPLIT_NAMES},
                "by_source": dict(sorted(Counter(row.source for row in all_rows).items())),
                "by_domain": dict(sorted(Counter(row.domain for row in all_rows).items())),
                "rendered_token_stats": _token_stats(row.token_count for row in all_rows),
            }

    rejected_total = sum(rejected.values())
    quarantined_total = sum(quarantined.values())
    duplicates_total = sum(duplicates_by_leaf.values())
    trace_non_emissions = sum(
        count
        for (leaf, status), count in trace_status_by_leaf.items()
        if status not in {TIER_SHORT, TIER_LONG}
    )
    input_accounted = rejected_total + quarantined_total + admitted_before_dedup
    dedup_accounted = len(candidates) + duplicates_total
    trace_accounted = trace_emissions + trace_non_emissions
    if input_accounted != total_records:
        raise AssertionError("input disposition accounting does not balance")
    if dedup_accounted != admitted_before_dedup:
        raise AssertionError("prompt deduplication accounting does not balance")
    if direct_emissions != len(candidates):
        raise AssertionError("every canonical candidate must have exactly one direct emission")
    if trace_accounted != len(candidates):
        raise AssertionError("trace disposition accounting does not balance")

    contamination_labels: dict[str, list[str]] = defaultdict(list)
    for source, label in sorted(contamination_by_source):
        contamination_labels[source].append(label)
    external_sources = {
        source: count
        for source, count in sorted(input_by_source.items())
        if source in foundation._KNOWN_EXTERNAL_EVALUATION_SOURCES or source.startswith("MATH/")
    }
    builder_path = Path(__file__).resolve()
    foundation_path = Path(foundation.__file__).resolve()
    effective_max = long_max_length or short_max_length
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "source_revision": source_revision,
            "inputs": input_receipts,
            "benchmark_suites": list(benchmark_index.receipts),
            "tokenizer": tokenizer_receipt,
            "builder": {"path": str(builder_path), "sha256": foundation._sha256_file(builder_path)},
            "foundation_builder": {
                "path": str(foundation_path),
                "sha256": foundation._sha256_file(foundation_path),
            },
        },
        "usage": {
            "scope": "private-only",
            "license_metadata_consulted_for_admission": False,
            "license_fields_read": False,
        },
        "contract": {
            "seed": seed,
            "short_max_rendered_tokens": short_max_length,
            "long_max_rendered_tokens": long_max_length,
            "effective_max_rendered_tokens": effective_max,
            "truncation": False,
            "required_teacher_model": required_teacher_model,
            "source_system_prompts": "discarded",
            "replacement_system_prompt_sha256": foundation._sha256_bytes(
                foundation.NEUTRAL_SYSTEM_PROMPT.encode("utf-8")
            ),
            "leaf_source_selectors": _source_selectors(),
            "expected_domain_by_leaf": _EXPECTED_DOMAIN_BY_LEAF,
            "stream_contracts": _stream_contracts(
                short_max_length=short_max_length, long_max_length=long_max_length
            ),
            "prompt_dedup": (
                "global Unicode NFKC + casefold + collapsed whitespace; highest quality_score, "
                "then lowest deterministic content SHA-256"
            ),
            "split_family": "SHA-256 of normalized prompt's first 256 characters",
            "split_policy": "seeded SHA-256 family assignment, 90/5/5, no record cap",
            "tool_use": "quarantined with aggregate counts; no raw shard is emitted",
            "record_metadata_policy": "aggregate-and-hash-only",
        },
        "quality": {
            "coverage": {
                "input_records": total_records,
                "input_by_source": dict(sorted(input_by_source.items())),
                "input_by_domain": dict(sorted(input_by_domain.items())),
                "rejected_records": rejected_total,
                "rejected_by_reason": dict(sorted(rejected.items())),
                "rejected_by_source_and_reason": _nested_counts(rejected_by_source),
                "rejected_by_domain_and_reason": _nested_counts(rejected_by_domain),
                "quarantined_records": quarantined_total,
                "quarantined_by_reason": dict(sorted(quarantined.items())),
                "quarantined_by_source_and_reason": _nested_counts(quarantined_by_source),
                "quarantined_by_domain_and_reason": _nested_counts(quarantined_by_domain),
                "admitted_before_prompt_dedup": admitted_before_dedup,
                "admitted_before_prompt_dedup_by_leaf": dict(sorted(admitted_by_leaf.items())),
                "admitted_before_prompt_dedup_by_source": dict(sorted(admitted_by_source.items())),
                "admitted_before_prompt_dedup_by_domain": dict(sorted(admitted_by_domain.items())),
                "prompt_duplicates_removed": duplicates_total,
                "prompt_duplicates_removed_by_leaf": dict(sorted(duplicates_by_leaf.items())),
                "prompt_duplicates_removed_by_source": dict(sorted(duplicates_by_source.items())),
                "prompt_duplicates_removed_by_domain": dict(sorted(duplicates_by_domain.items())),
                "canonical_records": len(candidates),
                "canonical_by_leaf": dict(sorted(canonical_by_leaf.items())),
                "canonical_by_source": dict(sorted(canonical_by_source.items())),
                "canonical_by_domain": dict(sorted(canonical_by_domain.items())),
                "canonical_by_split": {split: canonical_by_split[split] for split in SPLIT_NAMES},
                "input_disposition_equation": {
                    "input_records": total_records,
                    "rejected_plus_quarantined_plus_admitted": input_accounted,
                    "balanced": True,
                },
                "dedup_equation": {
                    "admitted_before_prompt_dedup": admitted_before_dedup,
                    "canonical_plus_duplicates": dedup_accounted,
                    "balanced": True,
                },
                "direct_emission_equation": {
                    "canonical_records": len(candidates),
                    "direct_emissions": direct_emissions,
                    "balanced": True,
                },
                "trace_disposition_equation": {
                    "canonical_records": len(candidates),
                    "trace_emissions_plus_non_emissions": trace_accounted,
                    "balanced": True,
                },
                "zero_silent_drops": True,
            },
            "counts_by_leaf": counts_by_leaf,
            "trace_disposition_by_leaf": _nested_counts(trace_status_by_leaf),
            "trace_disposition_by_source": _nested_counts(trace_status_by_source),
            "trace_disposition_by_domain": _nested_counts(trace_status_by_domain),
            "contamination": {
                "exact_prompt_matches": sum(
                    count
                    for (source, label), count in contamination_by_source.items()
                    if label == "exact_prompt_overlap"
                ),
                "split_family_matches": sum(
                    count
                    for (source, label), count in contamination_by_source.items()
                    if label == "split_family_overlap"
                ),
                "counts_by_source": _nested_counts(contamination_by_source),
                "source_labels": dict(contamination_labels),
                "known_external_evaluation_sources_present": external_sources,
                "known_external_evaluation_source_rows": sum(external_sources.values()),
                "external_evaluation_policy": (
                    "Rows are retained for private training, but results on their corresponding "
                    "public benchmark families must be treated as contaminated. Molor's frozen "
                    "suites are screened independently by exact prompt and 256-character family."
                ),
                "raw_examples_recorded": False,
            },
            "split_overlap_assertions": overlap_receipt,
            "global_prompt_split_overlap": 0,
            "global_split_family_overlap": 0,
            "all_outputs_within_declared_tiers": all(
                (
                    row.token_count <= short_max_length
                    if stream.endswith(f"-{TIER_SHORT}")
                    else short_max_length < row.token_count <= effective_max
                )
                for leaf_streams in views.values()
                for stream, splits in leaf_streams.items()
                for rows in splits.values()
                for row in rows
            ),
        },
    }
    return _write_outputs(
        Path(output_dir),
        views=views,
        manifest=manifest,
        short_max_length=short_max_length,
        long_max_length=long_max_length,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="explicit Parquet shard (repeat exactly three times; defaults to pinned shards)",
    )
    parser.add_argument(
        "--expected-input-sha256",
        action="append",
        default=[],
        metavar="PATH=SHA256",
        help="required expected digest for every custom input",
    )
    parser.add_argument(
        "--benchmark-suite",
        action="append",
        type=Path,
        help="explicit leakage suite (repeatable; defaults to the pinned Molor suites)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=foundation.DEFAULT_TOKENIZER)
    parser.add_argument("--short-max-length", type=int, default=DEFAULT_SHORT_MAX_LENGTH)
    parser.add_argument("--long-max-length", type=int, default=DEFAULT_LONG_MAX_LENGTH)
    parser.add_argument(
        "--omit-long",
        action="store_true",
        help="omit 1025-4096 streams and reject records above the short cap",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--teacher-model", default=foundation.DEFAULT_TEACHER_MODEL)
    parser.add_argument("--source-revision", default=foundation.DEFAULT_SOURCE_REVISION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = tuple(args.input) if args.input else foundation.DEFAULT_INPUTS
    suites = (
        tuple(args.benchmark_suite) if args.benchmark_suite else foundation.DEFAULT_BENCHMARK_SUITES
    )
    try:
        expected = (
            foundation._parse_expected_hashes(args.expected_input_sha256)
            if args.input
            else dict(foundation.DEFAULT_INPUT_HASHES)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not args.input and args.expected_input_sha256:
        parser.error("custom expected hashes require explicit --input arguments")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional local tooling
        parser.error(f"transformers is required: {exc}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    manifest = materialize_qwen38_distillation_experts(
        input_paths=paths,
        output_dir=args.output,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        expected_input_hashes=expected,
        benchmark_suite_paths=suites,
        short_max_length=args.short_max_length,
        long_max_length=None if args.omit_long else args.long_max_length,
        seed=args.seed,
        required_teacher_model=args.teacher_model,
        source_revision=args.source_revision,
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
