"""Materialize private, specialist SFT views from the pinned Qwen3.8 distillation set.

The builder is deliberately narrow: exactly three explicitly named Parquet shards must be
provided and every shard must have a caller-supplied SHA-256.  Admission is based only on
technical properties.  In particular, license metadata is neither read nor used to admit or
reject a record.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Protocol, Sequence
import unicodedata


SCHEMA_VERSION = "molor.qwen38-distillation-corpus.v1"
DEFAULT_SEED = "molor-qwen38-distillation-v1"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_TEACHER_MODEL = "qwen3.8-max-preview"
DEFAULT_SOURCE_REVISION = "ab9f8b289423c249fc0054507f045a12efb54b1b"
NEUTRAL_SYSTEM_PROMPT = (
    "You are a precise and capable assistant. Follow the user's request and provide a "
    "correct, clear answer."
)

VIEW_REASONING_MATH_FINAL = "reasoning-math-final"
VIEW_CODE_FINAL = "code-final"
VIEW_INSTRUCTION_FINAL = "instruction-final"
VIEW_REASONING_MATH_TRACE = "reasoning-math-trace"
VIEW_NAMES = (
    VIEW_REASONING_MATH_FINAL,
    VIEW_CODE_FINAL,
    VIEW_INSTRUCTION_FINAL,
    VIEW_REASONING_MATH_TRACE,
)
SPLIT_NAMES = ("train", "validation", "test")
_FINAL_VIEW_BY_DOMAIN = {
    "math": VIEW_REASONING_MATH_FINAL,
    "reasoning": VIEW_REASONING_MATH_FINAL,
    "code": VIEW_CODE_FINAL,
    "instruction": VIEW_INSTRUCTION_FINAL,
}
_KNOWN_DOMAINS = frozenset((*_FINAL_VIEW_BY_DOMAIN, "tool_use"))
_KNOWN_EXTERNAL_EVALUATION_SOURCES = frozenset(
    {
        "ARC-Challenge",
        "ARC-Easy",
        "CommonsenseQA",
        "HumanEval",
        "IFEval",
        "MBPP",
        "OpenBookQA",
        "QASC",
        "SciQ",
        "gsm8k",
    }
)

_DATASET_ROOT = Path.home() / "Documents" / "datasets" / "qwen3.8-max-distillation-50k"
DEFAULT_INPUTS = (
    _DATASET_ROOT / "data" / "train-00000-of-00001.parquet",
    _DATASET_ROOT / "data" / "validation-00000-of-00001.parquet",
    _DATASET_ROOT / "data" / "test-00000-of-00001.parquet",
)
DEFAULT_INPUT_HASHES = {
    str(DEFAULT_INPUTS[0].resolve()): (
        "406e4335f46fc960ea14459bde6d1ebab91001c4051abe057c9b1268dc5371b6"
    ),
    str(DEFAULT_INPUTS[1].resolve()): (
        "3cb00689251b1b2769ca5b3ac27d77e4895a7c15b659b4451495609e02477260"
    ),
    str(DEFAULT_INPUTS[2].resolve()): (
        "57eb75365465ad134c8f934bd031ded1485962e6fce156f6316b088300335d23"
    ),
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENIZER = _REPOSITORY_ROOT / "models" / "Qwen3-0.6B"
DEFAULT_BENCHMARK_SUITES = (
    _REPOSITORY_ROOT / "benchmarks" / "suites" / "molor-core-v1.json",
    _REPOSITORY_ROOT / "benchmarks" / "suites" / "molor-executable-v1.json",
)

_EXPECTED_COLUMNS = (
    "id",
    "domain",
    "source",
    "difficulty",
    "messages",
    "ground_truth",
    "quality_score",
    "teacher_model",
    "completion_tokens",
    "generation_time",
)
_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_RESERVED_CHAT_MARKERS = ("<|im_start|>", "<|im_end|>")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


class ChatTemplateTokenizer(Protocol):
    """Tokenizer surface used by the materializer and its unit tests."""

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> object: ...


@dataclass(frozen=True, slots=True)
class ParsedRecord:
    source: str
    domain: str
    prompt: str
    trace: str | None
    final: str
    quality_score: float
    teacher_model: str


@dataclass(frozen=True, slots=True)
class Candidate:
    prompt_group: str
    family_group: str
    split: str
    source: str
    domain: str
    quality_score: float
    content_hash: str
    order_rank: str
    final_payload: str
    final_token_count: int
    trace_payload: str | None
    trace_token_count: int | None


@dataclass(frozen=True, slots=True)
class OutputRecord:
    prompt_group: str
    family_group: str
    source: str
    domain: str
    order_rank: str
    payload: str
    token_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkIndex:
    prompt_groups: frozenset[str]
    family_groups: frozenset[str]
    receipts: tuple[dict[str, Any], ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = cleaned.strip()
    return cleaned or None


def normalize_prompt(prompt: str) -> str:
    """Normalize a prompt for exact global deduplication."""

    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(normalized.split())


def prompt_group_id(prompt: str) -> str:
    return _sha256_bytes(normalize_prompt(prompt).encode("utf-8"))


def split_family_id(prompt: str) -> str:
    """Hash the normalized first 256 characters used to keep related prompts together."""

    family_prefix = normalize_prompt(prompt)[:256]
    return _sha256_bytes(family_prefix.encode("utf-8"))


def split_for_family(family_group: str, *, seed: str = DEFAULT_SEED) -> str:
    """Assign one family to a deterministic 90/5/5 train/validation/test bucket."""

    digest = hashlib.sha256(f"{seed}\0split\0{family_group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 9_000:
        return "train"
    if bucket < 9_500:
        return "validation"
    return "test"


def _token_ids(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("chat template must return a non-empty flat token list")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError("chat template returned invalid token IDs")
    return list(value)


def _render_token_ids(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> tuple[list[int], list[int]]:
    if len(messages) != 3 or [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("distillation records must be system/user/assistant chats")
    target = messages[-1].get("content", "")
    if enable_thinking:
        if not target.startswith("<think>\n") or "\n</think>\n\n" not in target:
            raise ValueError("trace target is not a canonical explicit thinking target")
        trace, final = target[len("<think>\n") :].split("\n</think>\n\n", 1)
        if not trace.strip() or not final.strip():
            raise ValueError("trace target must have non-empty reasoning and final answer")
    else:
        prefix = "<think>\n\n</think>\n\n"
        if not target.startswith(prefix) or not target[len(prefix) :].strip():
            raise ValueError("final target is not a canonical explicit empty-thinking target")

    common = {
        "tokenize": True,
        "enable_thinking": enable_thinking,
        "truncation": False,
        "padding": False,
        "return_tensors": None,
        "return_dict": False,
    }
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **common)
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(messages, add_generation_prompt=False, **common)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        mode = "thinking-enabled" if enable_thinking else "thinking-disabled"
        raise ValueError(f"tokenizer chat template has an inconsistent {mode} training prefix")
    return prompt_ids, full_ids


def render_final_token_ids(
    tokenizer: ChatTemplateTokenizer, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    """Render and prefix-check an explicit empty-think final-only example."""

    return _render_token_ids(tokenizer, messages, enable_thinking=False)


def render_trace_token_ids(
    tokenizer: ChatTemplateTokenizer, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    """Render and prefix-check an explicit reasoning-trace example."""

    return _render_token_ids(tokenizer, messages, enable_thinking=True)


def _parse_assistant_content(content: str) -> tuple[str | None, str] | tuple[None, None]:
    open_count = content.count("<think>")
    close_count = content.count("</think>")
    if open_count == 0 and close_count == 0:
        final = _clean_text(content)
        return (None, final) if final is not None else (None, None)
    if open_count != 1 or close_count != 1:
        return None, None
    open_at = content.find("<think>")
    close_at = content.find("</think>")
    if open_at < 0 or close_at < open_at:
        return None, None
    # A small number of otherwise valid teacher answers put a short reasoning preamble before
    # the tag.  It is reasoning rather than final-answer text, so preserve it at the beginning of
    # the canonical trace instead of silently discarding the record or the preamble.
    preamble = _clean_text(content[:open_at])
    tagged_trace = _clean_text(content[open_at + len("<think>") : close_at])
    trace = "\n".join(part for part in (preamble, tagged_trace) if part is not None) or None
    final = _clean_text(content[close_at + len("</think>") :])
    if trace is None or final is None:
        return None, None
    return trace, final


def _record_labels(record: Mapping[str, Any]) -> tuple[str, str]:
    source = _clean_text(record.get("source")) or "<invalid-source>"
    domain = _clean_text(record.get("domain")) or "<invalid-domain>"
    return source, domain


def _parse_record(
    record: Mapping[str, Any], *, required_teacher_model: str
) -> tuple[ParsedRecord | None, str | None]:
    required_strings: dict[str, str] = {}
    for name in (
        "id",
        "domain",
        "source",
        "difficulty",
        "teacher_model",
        "generation_time",
    ):
        cleaned = _clean_text(record.get(name))
        if cleaned is None:
            return None, f"empty_or_invalid_{name}"
        required_strings[name] = cleaned
    # Ground truth is auxiliary provenance rather than the SFT target.  Some valid source
    # families intentionally leave it empty, so require its strict string type but do not use
    # its value (or emptiness) for admission.
    if not isinstance(record.get("ground_truth"), str):
        return None, "invalid_ground_truth_type"

    domain = required_strings["domain"]
    if domain not in _KNOWN_DOMAINS:
        return None, "unsupported_domain"
    if required_strings["teacher_model"] != required_teacher_model:
        return None, "teacher_model_mismatch"

    quality = record.get("quality_score")
    if isinstance(quality, bool) or not isinstance(quality, (int, float)):
        return None, "invalid_quality_score"
    quality_score = float(quality)
    if not math.isfinite(quality_score) or not 0.0 <= quality_score <= 10.0:
        return None, "invalid_quality_score"
    completion_tokens = record.get("completion_tokens")
    if type(completion_tokens) is not int or completion_tokens < 0:
        return None, "invalid_completion_tokens"

    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) != 3:
        return None, "invalid_messages_shape"
    expected_roles = ("system", "user", "assistant")
    contents: list[str] = []
    for raw, expected_role in zip(raw_messages, expected_roles, strict=True):
        if not isinstance(raw, dict) or set(raw) != {"role", "content"}:
            return None, "invalid_messages_shape"
        if raw.get("role") != expected_role:
            return None, "invalid_message_roles"
        content = _clean_text(raw.get("content"))
        if content is None:
            return None, "empty_message_content"
        contents.append(content)

    prompt = contents[1]
    if any(marker in prompt for marker in (*_RESERVED_CHAT_MARKERS, "<think>", "</think>")):
        return None, "reserved_chat_marker_in_prompt"
    if any(marker in contents[2] for marker in _RESERVED_CHAT_MARKERS):
        return None, "reserved_chat_marker_in_assistant"
    trace, final = _parse_assistant_content(contents[2])
    if final is None:
        return None, "malformed_or_empty_thinking_target"
    return (
        ParsedRecord(
            source=required_strings["source"],
            domain=domain,
            prompt=prompt,
            trace=trace,
            final=final,
            quality_score=quality_score,
            teacher_model=required_strings["teacher_model"],
        ),
        None,
    )


def _messages(prompt: str, assistant_target: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_target},
    ]


def _payload(messages: list[dict[str, str]]) -> str:
    return json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":"))


def _validate_parquet_schema(schema: object, pa: Any) -> None:
    if tuple(schema.names) != _EXPECTED_COLUMNS:
        raise ValueError(
            f"Parquet columns must be exactly {_EXPECTED_COLUMNS!r}, got {tuple(schema.names)!r}"
        )
    expected_types = {
        "id": pa.string(),
        "domain": pa.string(),
        "source": pa.string(),
        "difficulty": pa.string(),
        "messages": pa.list_(
            pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
        ),
        "ground_truth": pa.string(),
        "quality_score": pa.float64(),
        "teacher_model": pa.string(),
        "completion_tokens": pa.int64(),
        "generation_time": pa.string(),
    }
    for name, expected in expected_types.items():
        actual = schema.field(name).type
        if actual != expected:
            raise ValueError(f"Parquet column {name!r} must be {expected}, got {actual}")


def _verified_inputs(
    input_paths: Sequence[str | Path], expected_hashes: Mapping[str | Path, str]
) -> tuple[tuple[Path, ...], list[dict[str, Any]]]:
    paths = tuple(Path(path).resolve() for path in input_paths)
    if len(paths) != 3:
        raise ValueError("exactly three explicit Parquet input paths are required")
    if len(set(paths)) != 3:
        raise ValueError("the three Parquet input paths must be unique")
    resolved_expected: dict[Path, str] = {}
    for raw_path, raw_digest in expected_hashes.items():
        digest = str(raw_digest).lower()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid expected SHA-256 for {raw_path!r}")
        resolved_expected[Path(raw_path).resolve()] = digest
    if set(resolved_expected) != set(paths):
        raise ValueError("expected_input_hashes must contain exactly the three input paths")

    receipts: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.casefold() != ".parquet" or not path.is_file():
            raise FileNotFoundError(f"explicit Parquet input does not exist: {path}")
        actual = _sha256_file(path)
        expected = resolved_expected[path]
        if actual != expected:
            raise ValueError(
                f"input SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )
        receipts.append(
            {
                "path": str(path),
                "sha256": actual,
                "expected_sha256": expected,
                "hash_verified": True,
                "bytes": path.stat().st_size,
            }
        )
    return paths, receipts


def _load_benchmark_index(paths: Sequence[str | Path]) -> BenchmarkIndex:
    suite_paths = tuple(Path(path).resolve() for path in paths)
    if not suite_paths or len(set(suite_paths)) != len(suite_paths):
        raise ValueError("at least one unique explicit benchmark suite is required")
    prompt_groups: set[str] = set()
    family_groups: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for path in suite_paths:
        if not path.is_file():
            raise FileNotFoundError(f"benchmark suite does not exist: {path}")
        raw = path.read_bytes()
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid benchmark suite JSON {path}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
            raise ValueError(f"benchmark suite {path} must contain a cases list")
        case_count = 0
        for index, case in enumerate(document["cases"]):
            if not isinstance(case, dict):
                raise ValueError(f"benchmark suite {path} case {index} must be an object")
            prompt = _clean_text(case.get("prompt"))
            if prompt is None:
                raise ValueError(f"benchmark suite {path} case {index} has no prompt")
            prompt_groups.add(prompt_group_id(prompt))
            family_groups.add(split_family_id(prompt))
            case_count += 1
        receipts.append(
            {
                "path": str(path),
                "sha256": _sha256_bytes(raw),
                "bytes": len(raw),
                "cases": case_count,
            }
        )
    return BenchmarkIndex(frozenset(prompt_groups), frozenset(family_groups), tuple(receipts))


def _tokenizer_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local tokenizer directory does not exist: {root}")
    aggregate = hashlib.sha256()
    artifacts: list[dict[str, Any]] = []
    for name in _TOKENIZER_ARTIFACTS:
        artifact = root / name
        if not artifact.is_file():
            continue
        digest = _sha256_file(artifact)
        artifacts.append({"path": name, "sha256": digest, "bytes": artifact.stat().st_size})
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if not artifacts:
        raise ValueError(f"no tokenizer artifacts found in {root}")
    return {
        "path": str(root),
        "aggregate_sha256": aggregate.hexdigest(),
        "artifacts": artifacts,
        "chat_template_modes": {
            "final_views": {"enable_thinking": False},
            "trace_view": {"enable_thinking": True},
        },
    }


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
            source TEXT NOT NULL,
            domain TEXT NOT NULL,
            quality_score REAL NOT NULL,
            content_hash TEXT NOT NULL,
            order_rank TEXT NOT NULL,
            final_payload TEXT NOT NULL,
            final_token_count INTEGER NOT NULL,
            trace_payload TEXT,
            trace_token_count INTEGER
        ) WITHOUT ROWID;
        """
    )


def _insert_candidate(connection: Any, candidate: Candidate) -> None:
    connection.execute(
        """
        INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(prompt_group) DO UPDATE SET
            family_group=excluded.family_group,
            split=excluded.split,
            source=excluded.source,
            domain=excluded.domain,
            quality_score=excluded.quality_score,
            content_hash=excluded.content_hash,
            order_rank=excluded.order_rank,
            final_payload=excluded.final_payload,
            final_token_count=excluded.final_token_count,
            trace_payload=excluded.trace_payload,
            trace_token_count=excluded.trace_token_count
        WHERE excluded.quality_score > candidates.quality_score
           OR (excluded.quality_score = candidates.quality_score
               AND excluded.content_hash < candidates.content_hash)
        """,
        (
            candidate.prompt_group,
            candidate.family_group,
            candidate.split,
            candidate.source,
            candidate.domain,
            candidate.quality_score,
            candidate.content_hash,
            candidate.order_rank,
            candidate.final_payload,
            candidate.final_token_count,
            candidate.trace_payload,
            candidate.trace_token_count,
        ),
    )


def _candidate_from_row(row: Sequence[Any]) -> Candidate:
    return Candidate(
        prompt_group=str(row[0]),
        family_group=str(row[1]),
        split=str(row[2]),
        source=str(row[3]),
        domain=str(row[4]),
        quality_score=float(row[5]),
        content_hash=str(row[6]),
        order_rank=str(row[7]),
        final_payload=str(row[8]),
        final_token_count=int(row[9]),
        trace_payload=None if row[10] is None else str(row[10]),
        trace_token_count=None if row[11] is None else int(row[11]),
    )


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
    candidates: Sequence[Candidate],
) -> tuple[dict[str, dict[str, list[OutputRecord]]], int]:
    views = {view: {split: [] for split in SPLIT_NAMES} for view in VIEW_NAMES}
    quarantined = 0
    for candidate in candidates:
        if candidate.domain == "tool_use":
            quarantined += 1
            continue
        final_view = _FINAL_VIEW_BY_DOMAIN[candidate.domain]
        views[final_view][candidate.split].append(
            OutputRecord(
                candidate.prompt_group,
                candidate.family_group,
                candidate.source,
                candidate.domain,
                candidate.order_rank,
                candidate.final_payload,
                candidate.final_token_count,
            )
        )
        if (
            candidate.domain in {"math", "reasoning"}
            and candidate.trace_payload is not None
            and candidate.trace_token_count is not None
        ):
            views[VIEW_REASONING_MATH_TRACE][candidate.split].append(
                OutputRecord(
                    candidate.prompt_group,
                    candidate.family_group,
                    candidate.source,
                    candidate.domain,
                    candidate.order_rank,
                    candidate.trace_payload,
                    candidate.trace_token_count,
                )
            )
    for splits in views.values():
        for rows in splits.values():
            rows.sort(key=lambda row: (row.order_rank, row.prompt_group))
    return views, quarantined


def _overlap_receipt(views: Mapping[str, Mapping[str, Sequence[OutputRecord]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view, splits in views.items():
        prompt_sets = {split: {row.prompt_group for row in rows} for split, rows in splits.items()}
        family_sets = {split: {row.family_group for row in rows} for split, rows in splits.items()}
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
            raise AssertionError(f"prompt or split-family leakage detected in {view}")
        result[view] = {
            "prompt_group": {**prompt_overlap, "zero_overlap": True},
            "split_family": {**family_overlap, "zero_overlap": True},
        }
    return result


def _write_outputs(
    output_dir: Path,
    *,
    views: Mapping[str, Mapping[str, Sequence[OutputRecord]]],
    manifest: dict[str, Any],
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        receipts: dict[str, dict[str, Any]] = {}
        for view in VIEW_NAMES:
            view_dir = stage / view
            view_dir.mkdir()
            receipts[view] = {}
            for split in SPLIT_NAMES:
                path = view_dir / f"{split}.jsonl"
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in views[view][split]:
                        stream.write(row.payload)
                        stream.write("\n")
                receipts[view][split] = {
                    "path": path.relative_to(stage).as_posix(),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                    "records": len(views[view][split]),
                }
        manifest["outputs"] = receipts
        manifest_path = stage / "quality-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return target / "quality-manifest.json"


def materialize_qwen38_distillation_corpus(
    *,
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    tokenizer: ChatTemplateTokenizer,
    tokenizer_path: str | Path,
    expected_input_hashes: Mapping[str | Path, str],
    benchmark_suite_paths: Sequence[str | Path],
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: str = DEFAULT_SEED,
    required_teacher_model: str = DEFAULT_TEACHER_MODEL,
    source_revision: str = DEFAULT_SOURCE_REVISION,
) -> Path:
    """Build all deterministic specialist views and return the manifest path."""

    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if not seed:
        raise ValueError("seed must be a non-empty string")
    if not required_teacher_model:
        raise ValueError("required_teacher_model must be non-empty")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("source_revision must be a full lowercase Git SHA-1")

    # PyArrow is intentionally optional at import time and is loaded only when materializing.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional local tooling
        raise RuntimeError("pyarrow is required to materialize the Parquet corpus") from exc
    import sqlite3

    paths, input_receipts = _verified_inputs(input_paths, expected_input_hashes)
    benchmark_index = _load_benchmark_index(benchmark_suite_paths)
    tokenizer_receipt = _tokenizer_identity(tokenizer_path)

    rejections: Counter[str] = Counter()
    rejection_by_domain: Counter[tuple[str, str]] = Counter()
    rejection_by_source: Counter[tuple[str, str]] = Counter()
    input_by_domain: Counter[str] = Counter()
    input_by_source: Counter[str] = Counter()
    eligible_by_domain: Counter[str] = Counter()
    eligible_by_source: Counter[str] = Counter()
    contamination_by_source: Counter[tuple[str, str]] = Counter()
    trace_view_exclusions: Counter[str] = Counter()
    eligible_before_dedup = 0
    total_records = 0

    def reject(reason: str, source: str, domain: str) -> None:
        rejections[reason] += 1
        rejection_by_source[(source, reason)] += 1
        rejection_by_domain[(domain, reason)] += 1

    with tempfile.TemporaryDirectory(prefix="molor-qwen38-distillation-") as temporary:
        database = Path(temporary) / "candidates.sqlite3"
        connection = sqlite3.connect(database)
        try:
            _create_candidate_table(connection)
            for path, receipt in zip(paths, input_receipts, strict=True):
                parquet = pq.ParquetFile(path)
                _validate_parquet_schema(parquet.schema_arrow, pa)
                receipt["records"] = parquet.metadata.num_rows
                scanned = 0
                for batch in parquet.iter_batches(batch_size=2048, columns=list(_EXPECTED_COLUMNS)):
                    for record in batch.to_pylist():
                        scanned += 1
                        total_records += 1
                        source_label, domain_label = _record_labels(record)
                        input_by_source[source_label] += 1
                        input_by_domain[domain_label] += 1
                        parsed, reason = _parse_record(
                            record, required_teacher_model=required_teacher_model
                        )
                        if parsed is None:
                            assert reason is not None
                            reject(reason, source_label, domain_label)
                            continue

                        prompt_group = prompt_group_id(parsed.prompt)
                        family_group = split_family_id(parsed.prompt)
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

                        final_target = f"<think>\n\n</think>\n\n{parsed.final}"
                        final_messages = _messages(parsed.prompt, final_target)
                        _, final_ids = render_final_token_ids(tokenizer, final_messages)
                        if len(final_ids) > max_length:
                            reject("final_rendered_too_long", parsed.source, parsed.domain)
                            continue

                        trace_payload: str | None = None
                        trace_tokens: int | None = None
                        if parsed.trace is not None and parsed.domain in {"math", "reasoning"}:
                            trace_target = f"<think>\n{parsed.trace}\n</think>\n\n{parsed.final}"
                            trace_messages = _messages(parsed.prompt, trace_target)
                            _, trace_ids = render_trace_token_ids(tokenizer, trace_messages)
                            if len(trace_ids) <= max_length:
                                trace_payload = _payload(trace_messages)
                                trace_tokens = len(trace_ids)
                            else:
                                trace_view_exclusions["rendered_too_long"] += 1
                        elif parsed.trace is None and parsed.domain in {"math", "reasoning"}:
                            trace_view_exclusions["no_explicit_trace"] += 1

                        canonical_content = json.dumps(
                            {
                                "domain": parsed.domain,
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
                        content_hash = _sha256_bytes(canonical_content.encode("utf-8"))
                        order_rank = _sha256_bytes(
                            f"{seed}\0order\0{prompt_group}\0{content_hash}".encode("utf-8")
                        )
                        candidate = Candidate(
                            prompt_group=prompt_group,
                            family_group=family_group,
                            split=split_for_family(family_group, seed=seed),
                            source=parsed.source,
                            domain=parsed.domain,
                            quality_score=parsed.quality_score,
                            content_hash=content_hash,
                            order_rank=order_rank,
                            final_payload=_payload(final_messages),
                            final_token_count=len(final_ids),
                            trace_payload=trace_payload,
                            trace_token_count=trace_tokens,
                        )
                        _insert_candidate(connection, candidate)
                        eligible_before_dedup += 1
                        eligible_by_source[parsed.source] += 1
                        eligible_by_domain[parsed.domain] += 1
                if scanned != parquet.metadata.num_rows:
                    raise AssertionError(f"Parquet row-count mismatch while scanning {path}")
                connection.commit()

            rows = connection.execute(
                "SELECT prompt_group, family_group, split, source, domain, quality_score, "
                "content_hash, order_rank, final_payload, final_token_count, trace_payload, "
                "trace_token_count FROM candidates ORDER BY order_rank, prompt_group"
            ).fetchall()
            candidates = [_candidate_from_row(row) for row in rows]
        finally:
            connection.close()

    prompt_splits: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    for candidate in candidates:
        prior_prompt = prompt_splits.setdefault(candidate.prompt_group, candidate.split)
        prior_family = family_splits.setdefault(candidate.family_group, candidate.split)
        if prior_prompt != candidate.split or prior_family != candidate.split:
            raise AssertionError("prompt or 256-character split family crossed splits")

    views, quarantined = _output_views(candidates)
    overlap_receipt = _overlap_receipt(views)
    canonical_by_domain = Counter(candidate.domain for candidate in candidates)
    canonical_by_source = Counter(candidate.source for candidate in candidates)
    canonical_by_split = Counter(candidate.split for candidate in candidates)
    counts_by_view: dict[str, Any] = {}
    for view, splits in views.items():
        all_rows = [row for split_rows in splits.values() for row in split_rows]
        counts_by_view[view] = {
            "records": len(all_rows),
            "by_split": {split: len(splits[split]) for split in SPLIT_NAMES},
            "by_domain": dict(sorted(Counter(row.domain for row in all_rows).items())),
            "by_source": dict(sorted(Counter(row.source for row in all_rows).items())),
            "token_stats": _token_stats(row.token_count for row in all_rows),
        }

    contamination_labels: dict[str, list[str]] = defaultdict(list)
    for source, label in sorted(contamination_by_source):
        contamination_labels[source].append(label)
    external_evaluation_sources = {
        source: count
        for source, count in sorted(input_by_source.items())
        if source in _KNOWN_EXTERNAL_EVALUATION_SOURCES or source.startswith("MATH/")
    }

    builder_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "source_revision": source_revision,
            "inputs": input_receipts,
            "benchmark_suites": list(benchmark_index.receipts),
            "tokenizer": tokenizer_receipt,
            "builder": {"path": str(builder_path), "sha256": _sha256_file(builder_path)},
        },
        "usage": {
            "scope": "private-only",
            "license_metadata_consulted_for_admission": False,
            "license_fields_read": False,
        },
        "contract": {
            "seed": seed,
            "max_rendered_tokens": max_length,
            "truncation": False,
            "required_teacher_model": required_teacher_model,
            "source_system_prompts": "discarded",
            "replacement_system_prompt_sha256": _sha256_bytes(
                NEUTRAL_SYSTEM_PROMPT.encode("utf-8")
            ),
            "final_target": "<think>\\n\\n</think>\\n\\n{final}",
            "trace_target": "<think>\\n{trace}\\n</think>\\n\\n{final}",
            "final_chat_template_enable_thinking": False,
            "trace_chat_template_enable_thinking": True,
            "prompt_dedup": (
                "global Unicode NFKC + casefold + collapsed whitespace; highest quality_score, "
                "then lowest deterministic content SHA-256"
            ),
            "split_family": "SHA-256 of normalized prompt's first 256 characters",
            "split_policy": "seeded SHA-256 family assignment, 90/5/5, no record cap",
            "tool_use": "quarantined; no raw shard is emitted",
        },
        "quality": {
            "input_records": total_records,
            "rejections": dict(sorted(rejections.items())),
            "rejections_by_domain": _nested_counts(rejection_by_domain),
            "rejections_by_source": _nested_counts(rejection_by_source),
            "input_by_domain": dict(sorted(input_by_domain.items())),
            "input_by_source": dict(sorted(input_by_source.items())),
            "eligible_before_prompt_dedup": eligible_before_dedup,
            "eligible_before_prompt_dedup_by_domain": dict(sorted(eligible_by_domain.items())),
            "eligible_before_prompt_dedup_by_source": dict(sorted(eligible_by_source.items())),
            "unique_prompt_groups_after_dedup": len(candidates),
            "prompt_duplicates_removed": eligible_before_dedup - len(candidates),
            "canonical_by_domain": dict(sorted(canonical_by_domain.items())),
            "canonical_by_source": dict(sorted(canonical_by_source.items())),
            "canonical_by_split": {split: canonical_by_split[split] for split in SPLIT_NAMES},
            "tool_use_quarantined_prompt_groups": quarantined,
            "trace_view_exclusions": dict(sorted(trace_view_exclusions.items())),
            "counts_by_view": counts_by_view,
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
                "known_external_evaluation_sources_present": external_evaluation_sources,
                "known_external_evaluation_source_rows": sum(external_evaluation_sources.values()),
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
            "all_output_records_at_or_below_max_length": all(
                row.token_count <= max_length
                for splits in views.values()
                for rows in splits.values()
                for row in rows
            ),
        },
    }
    return _write_outputs(Path(output_dir), views=views, manifest=manifest)


def _parse_expected_hashes(values: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expected input hashes must use PATH=SHA256")
        raw_path, digest = value.rsplit("=", 1)
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid SHA-256 for {raw_path!r}")
        hashes[str(Path(raw_path).resolve())] = digest.lower()
    return hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="explicit Parquet shard (repeat exactly three times; defaults to the pinned shards)",
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
        help="explicit leakage suite (repeatable; defaults to the two pinned Molor suites)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = tuple(args.input) if args.input else DEFAULT_INPUTS
    suites = tuple(args.benchmark_suite) if args.benchmark_suite else DEFAULT_BENCHMARK_SUITES
    try:
        expected = (
            _parse_expected_hashes(args.expected_input_sha256)
            if args.input
            else dict(DEFAULT_INPUT_HASHES)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not args.input and args.expected_input_sha256:
        parser.error("custom expected hashes require explicit --input arguments")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional local tooling
        parser.error(f"transformers is required: {exc}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    manifest = materialize_qwen38_distillation_corpus(
        input_paths=paths,
        output_dir=args.output,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        expected_input_hashes=expected,
        benchmark_suite_paths=suites,
        max_length=args.max_length,
        seed=args.seed,
        required_teacher_model=args.teacher_model,
        source_revision=args.source_revision,
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
