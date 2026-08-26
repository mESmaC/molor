"""Materialize versioned, category-first non-security expert corpora.

The builder consumes only explicitly pinned non-security inputs.  Prepared Mythos and
Complete-FABLE rows are metadata-gated before their ``messages`` member is accessed.
Fable Premium is consumed once from its OpenAI-chat Parquet representation and converted
to bounded assistant-turn slices.  Fable-Max is recorded as an auxiliary archive only and
never emitted as messages SFT because it has no explicit assistant final answer.

Training JSONL rows contain exactly one top-level ``messages`` member.  Dataset identity,
quality, routing, length, contamination, deduplication, and source provenance remain in
aggregate/hash-only sidecars.
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
import sqlite3
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Protocol, Sequence
import unicodedata

from molor.experimental import qwen38_distillation_corpus as foundation


SCHEMA_VERSION = "molor.broad-distillation-experts.v1"
MIXTURE_SCHEMA_VERSION = "molor.expert-mixture.v1"
DEFAULT_SEED = "molor-broad-distillation-experts-v1"
DEFAULT_MAX_VARIANTS_PER_PROMPT_STREAM = 8
DEFAULT_PREMIUM_TARGETS_PER_SESSION = 4
DEFAULT_PREMIUM_CONTEXT_TOKENS = 8192

CORPUS_MYTHOS = "claude_mythos_distilled_25k"
CORPUS_COMPLETE_FABLE = "complete_fable_5_traces"
CORPUS_FABLE_PREMIUM = "fable_5_premium_openai_chat"
CORPUS_FABLE_MAX_AUXILIARY = "fable_5_max_reasoning_auxiliary"
CORPUS_NAMES = (
    CORPUS_MYTHOS,
    CORPUS_COMPLETE_FABLE,
    CORPUS_FABLE_PREMIUM,
    CORPUS_FABLE_MAX_AUXILIARY,
)
_EXPECTED_INPUT_COUNTS = {
    CORPUS_MYTHOS: 2,
    CORPUS_COMPLETE_FABLE: 2,
    CORPUS_FABLE_PREMIUM: 3,
    CORPUS_FABLE_MAX_AUXILIARY: 2,
}

LEAF_QUANTITATIVE_MATH = "quantitative_math"
LEAF_FORMAL_MATH = "formal_math"
LEAF_SCIENCE_REASONING = "science_reasoning"
LEAF_SCIENTIFIC_ENGINEERING = "scientific_engineering_reasoning"
LEAF_GENERAL_CODE = "general_code_generation"
LEAF_AGENTIC_CODE = "agentic_code_debugging"
LEAF_SOFTWARE_ARCHITECTURE = "software_architecture_design"
LEAF_SYSTEMS_DISTRIBUTED = "systems_distributed_engineering"
LEAF_FRONTEND_VISUAL = "frontend_visual_computing"
LEAF_ML_DATA = "ml_data_reasoning"
LEAF_TOOL_USE = "tool_use_execution"
LEAF_AGENTIC_PLANNING = "agentic_planning"
LEAF_GENERAL_REASONING = "general_expert_reasoning"
LEAF_STRICT_INSTRUCTION = "strict_instruction_following"
LEAF_NAMES = (
    LEAF_QUANTITATIVE_MATH,
    LEAF_FORMAL_MATH,
    LEAF_SCIENCE_REASONING,
    LEAF_SCIENTIFIC_ENGINEERING,
    LEAF_GENERAL_CODE,
    LEAF_AGENTIC_CODE,
    LEAF_SOFTWARE_ARCHITECTURE,
    LEAF_SYSTEMS_DISTRIBUTED,
    LEAF_FRONTEND_VISUAL,
    LEAF_ML_DATA,
    LEAF_TOOL_USE,
    LEAF_AGENTIC_PLANNING,
    LEAF_GENERAL_REASONING,
    LEAF_STRICT_INSTRUCTION,
)
LEAF_ALIAS_MAP = {
    "mathematical_formal_reasoning": [LEAF_QUANTITATIVE_MATH, LEAF_FORMAL_MATH],
    "scientific_engineering_reasoning": [
        LEAF_SCIENCE_REASONING,
        LEAF_SCIENTIFIC_ENGINEERING,
    ],
    "general_code_implementation": [LEAF_GENERAL_CODE],
    "strict_instruction_dialogue": [LEAF_STRICT_INSTRUCTION],
}

CONTRACT_DIRECT = "direct"
CONTRACT_TRACE = "trace"
CONTRACT_TOOL_DIRECT = "tool_direct"
CONTRACT_TOOL_TRACE = "tool_trace"
CONTRACT_NAMES = (
    CONTRACT_DIRECT,
    CONTRACT_TRACE,
    CONTRACT_TOOL_DIRECT,
    CONTRACT_TOOL_TRACE,
)

TIER_1K = "le1k"
TIER_4K = "1k_to_4k"
TIER_8K = "4k_to_8k"
TIER_32K = "8k_to_32k"
TIER_DEFERRED = "deferred_gt32k"
TRAINING_TIERS = (TIER_1K, TIER_4K, TIER_8K, TIER_32K)
TIER_LIMITS = {
    TIER_1K: (1, 1024),
    TIER_4K: (1025, 4096),
    TIER_8K: (4097, 8192),
    TIER_32K: (8193, 32768),
}
SPLIT_NAMES = ("train", "validation", "test")

_SECURITY_MARKERS = (
    "cyber",
    "security",
    "secops",
    "infosec",
    "redteam",
    "red-team",
    "malware",
    "exploit",
    "vulnerability",
    "pentest",
)
_SECURITY_METADATA_FIELDS = (
    "category",
    "domain",
    "topic",
    "source",
    "upstream",
    "subset",
    "task_type",
)
_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})
_RESERVED_CHAT_MARKERS = ("<|im_start|>", "<|im_end|>")

_MYTHOS_LEAF_BY_CATEGORY = {
    "advanced_coding": LEAF_GENERAL_CODE,
    "agentic_planning": LEAF_AGENTIC_PLANNING,
    "general_expert_qa": LEAF_GENERAL_REASONING,
    "mathematical_reasoning": LEAF_QUANTITATIVE_MATH,
    "scientific_analysis": LEAF_SCIENCE_REASONING,
}

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        LEAF_AGENTIC_PLANNING,
        ("agentic", "planning", "workflow orchestration", "task decomposition"),
    ),
    (
        LEAF_TOOL_USE,
        (
            "tool use",
            "tool-use",
            "function calling",
            "function-calling",
            "browser use",
            "computer use",
        ),
    ),
    (
        LEAF_AGENTIC_CODE,
        (
            "agentic coding",
            "code debugging",
            "debugging",
            "bug fix",
            "code repair",
            "code review",
            "refactor",
            "repository",
        ),
    ),
    (
        LEAF_FRONTEND_VISUAL,
        (
            "html",
            "css",
            "javascript",
            "typescript",
            "react",
            "frontend",
            "front-end",
            "webgl",
            "webgpu",
            "user interface",
            " ui ",
            " ux ",
            "visualization",
            "rendering",
            "shader",
            "3d animation",
            "computer graphics",
            "vfx",
        ),
    ),
    (
        LEAF_SOFTWARE_ARCHITECTURE,
        (
            "software architecture",
            "system architecture",
            "architectural design",
            "design pattern",
            "domain-driven",
            "microservice",
            "api design",
            "clean architecture",
            "software engineering",
        ),
    ),
    (
        LEAF_SYSTEMS_DISTRIBUTED,
        (
            "distributed",
            "consensus",
            "database",
            "storage",
            "cache",
            "operating system",
            "filesystem",
            "file system",
            "kernel",
            "concurrency",
            "parallel",
            "networking",
            "network protocol",
            "compiler",
            "runtime",
            "memory management",
            "lock-free",
            "crdt",
            "blockchain",
            "virtual machine",
        ),
    ),
    (
        LEAF_FORMAL_MATH,
        (
            "theorem",
            "proof",
            "formal logic",
            "sat/",
            "smt",
            "formal verification",
            "constraint solving",
            "number theory",
            "combinator",
        ),
    ),
    (
        LEAF_QUANTITATIVE_MATH,
        (
            "mathemat",
            "arithmetic",
            "algebra",
            "calculus",
            "numerical",
            "probability",
            "geometry",
            "optimization",
        ),
    ),
    (
        LEAF_SCIENTIFIC_ENGINEERING,
        (
            "materials",
            "nanofabrication",
            "mechanical",
            "electrical",
            "electronics",
            "aerospace",
            "thermodynamic",
            "fluid",
            "engineering",
        ),
    ),
    (
        LEAF_SCIENCE_REASONING,
        (
            "physics",
            "chemistry",
            "biology",
            "biomedical",
            "quantum",
            "genomics",
            "scientific analysis",
        ),
    ),
    (
        LEAF_ML_DATA,
        (
            "machine learning",
            "deep learning",
            "neural",
            "transformer",
            "artificial intelligence",
            "data science",
            "data analysis",
            "statistics",
            "computer vision",
            "natural language",
            "reinforcement learning",
        ),
    ),
    (
        LEAF_GENERAL_CODE,
        (
            "python",
            "rust",
            "c++",
            " c#",
            " java",
            " golang",
            " go ",
            "programming",
            "code generation",
            "implementation",
            "algorithm",
        ),
    ),
    (
        LEAF_STRICT_INSTRUCTION,
        (
            "instruction following",
            "instruction-following",
            "strict instruction",
            "format constraint",
            "structured output",
        ),
    ),
    (
        LEAF_GENERAL_REASONING,
        ("general reasoning", "expert qa", "expert question", "critical thinking"),
    ),
)

_PUBLIC_CONTAMINATION_LABELS = {
    CORPUS_MYTHOS: "unverified_possible_public_benchmark_derivation",
    CORPUS_COMPLETE_FABLE: "unverified_likely_mixed_public_source_derivation",
    CORPUS_FABLE_PREMIUM: "unverified_possible_public_benchmark_derivation",
    CORPUS_FABLE_MAX_AUXILIARY: "unverified_possible_public_benchmark_derivation",
}

_DATASET_ROOT = Path.home() / "Documents" / "datasets"
DEFAULT_INPUTS: dict[str, tuple[Path, ...]] = {
    CORPUS_MYTHOS: (
        _DATASET_ROOT / "prepared" / "claude-mythos-distilled-25k" / "train.jsonl",
        _DATASET_ROOT / "prepared" / "claude-mythos-distilled-25k" / "validation.jsonl",
    ),
    CORPUS_COMPLETE_FABLE: (
        _DATASET_ROOT / "prepared" / "complete-fable-5-traces" / "train.jsonl",
        _DATASET_ROOT / "prepared" / "complete-fable-5-traces" / "validation.jsonl",
    ),
    CORPUS_FABLE_PREMIUM: (
        _DATASET_ROOT / "fable-5-premium" / "openai_chat" / "train.parquet",
        _DATASET_ROOT / "fable-5-premium" / "openai_chat" / "validation.parquet",
        _DATASET_ROOT / "fable-5-premium" / "openai_chat" / "test.parquet",
    ),
    CORPUS_FABLE_MAX_AUXILIARY: (
        _DATASET_ROOT / "prepared" / "fable-5-max-reasoning-250" / "train.jsonl",
        _DATASET_ROOT / "prepared" / "fable-5-max-reasoning-250" / "validation.jsonl",
    ),
}
DEFAULT_INPUT_HASHES = {
    str(DEFAULT_INPUTS[CORPUS_MYTHOS][0].resolve()): (
        "addddc145d6f76f931657f3f9f24f45d41961cd0ab76f1fe31063da6bca7dfbf"
    ),
    str(DEFAULT_INPUTS[CORPUS_MYTHOS][1].resolve()): (
        "c6973ab564c5d1555ce6f0585a7940bb2c9bfde21945e56b6c122be4255f5448"
    ),
    str(DEFAULT_INPUTS[CORPUS_COMPLETE_FABLE][0].resolve()): (
        "70e7462373f5f6254a48c0f5a8c937195045dc1700b36cea25398e3686d83c8e"
    ),
    str(DEFAULT_INPUTS[CORPUS_COMPLETE_FABLE][1].resolve()): (
        "4b4ebdd4744ffb4ce4ec9da555738e7e3c5106f10ac0ec76ea7a156129f6770d"
    ),
    str(DEFAULT_INPUTS[CORPUS_FABLE_PREMIUM][0].resolve()): (
        "a8a004f39b05384284b04a6ae86706d9d8902a7e8b4695a0ad2afce19edaee8b"
    ),
    str(DEFAULT_INPUTS[CORPUS_FABLE_PREMIUM][1].resolve()): (
        "00b8c90c45eb261c4751b21ebc4baaf07ab5dc5f14a00bd464c35fb3f9b41f3d"
    ),
    str(DEFAULT_INPUTS[CORPUS_FABLE_PREMIUM][2].resolve()): (
        "54bcb781f6677c80681c5719f84c1762d80b0d5572409828915607f1f4e008d2"
    ),
    str(DEFAULT_INPUTS[CORPUS_FABLE_MAX_AUXILIARY][0].resolve()): (
        "dfd0b6ce765e13e12e91d2112a4be702589e7680e4c01b68e539e07a6802d4e1"
    ),
    str(DEFAULT_INPUTS[CORPUS_FABLE_MAX_AUXILIARY][1].resolve()): (
        "329f4482f32244e44db2c1902dc63f5b586c5b7ea354a242d26cc88cb2805365"
    ),
}


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict[str, Any]], **kwargs: Any) -> object: ...


@dataclass(frozen=True, slots=True)
class PreparedExample:
    corpus: str
    leaf: str
    contract: str
    prompt: str
    messages: list[dict[str, Any]]
    quality_score: float
    source_group: str
    metadata_category: str
    metadata_source: str


@dataclass(frozen=True, slots=True)
class Candidate:
    content_hash: str
    prompt_group: str
    family_group: str
    split: str
    corpus: str
    leaf: str
    contract: str
    tier: str
    token_count: int
    quality_score: float
    source_group: str
    order_rank: str
    payload: str


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


def _aggregate_label(value: object, *, fallback: str = "<unspecified>") -> str:
    cleaned = _clean_text(value)
    if cleaned is None:
        return fallback
    if len(cleaned) <= 128:
        return cleaned
    return f"sha256:{_sha256_bytes(cleaned.encode('utf-8'))}"


def normalize_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def prompt_group_id(prompt: str) -> str:
    return _sha256_bytes(normalize_prompt(prompt).encode("utf-8"))


def split_family_id(prompt: str) -> str:
    return _sha256_bytes(normalize_prompt(prompt)[:256].encode("utf-8"))


def split_for_family(family_group: str, *, seed: str = DEFAULT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}\0split\0{family_group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 9_000:
        return "train"
    if bucket < 9_500:
        return "validation"
    return "test"


def _metadata_is_security(value: object) -> bool:
    if not isinstance(value, str):
        return False
    folded = value.casefold().replace("_", "-")
    return any(marker in folded for marker in _SECURITY_MARKERS)


def metadata_security_gate(record: Mapping[str, object]) -> str | None:
    """Inspect only metadata fields and return the first security-bearing field."""

    for field in _SECURITY_METADATA_FIELDS:
        if _metadata_is_security(record.get(field)):
            return field
    return None


def _assert_non_security_path(path: Path) -> None:
    for part in path.parts:
        folded = part.casefold().replace("_", "-")
        if any(marker in folded for marker in _SECURITY_MARKERS):
            raise ValueError(f"input path is outside the non-security corpus boundary: {path}")


def _verified_inputs(
    input_paths: Mapping[str, Sequence[str | Path]],
    expected_hashes: Mapping[str | Path, str],
) -> tuple[dict[str, tuple[Path, ...]], list[dict[str, Any]]]:
    if set(input_paths) != set(CORPUS_NAMES):
        raise ValueError(f"input_paths must contain exactly {CORPUS_NAMES!r}")
    resolved: dict[str, tuple[Path, ...]] = {}
    all_paths: list[Path] = []
    for corpus in CORPUS_NAMES:
        paths = tuple(Path(path).resolve() for path in input_paths[corpus])
        if len(paths) != _EXPECTED_INPUT_COUNTS[corpus] or len(set(paths)) != len(paths):
            raise ValueError(
                f"{corpus} requires exactly {_EXPECTED_INPUT_COUNTS[corpus]} unique inputs"
            )
        for path in paths:
            _assert_non_security_path(path)
        resolved[corpus] = paths
        all_paths.extend(paths)
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("input files must be unique across corpora")

    normalized_hashes: dict[Path, str] = {}
    for raw_path, raw_digest in expected_hashes.items():
        digest = str(raw_digest).casefold()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid expected SHA-256 for {raw_path!r}")
        normalized_hashes[Path(raw_path).resolve()] = digest
    if set(normalized_hashes) != set(all_paths):
        raise ValueError("expected_hashes must contain exactly every explicit input path")

    receipts: list[dict[str, Any]] = []
    for corpus in CORPUS_NAMES:
        for path in resolved[corpus]:
            if not path.is_file():
                raise FileNotFoundError(f"explicit input does not exist: {path}")
            expected = normalized_hashes[path]
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"input SHA-256 mismatch for {path}: expected {expected}, got {actual}"
                )
            receipts.append(
                {
                    "corpus": corpus,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": actual,
                    "expected_sha256": expected,
                    "hash_verified": True,
                }
            )
    return resolved, receipts


def _leaf_for_category(corpus: str, category: object, upstream: object) -> str:
    if corpus == CORPUS_MYTHOS:
        return _MYTHOS_LEAF_BY_CATEGORY.get(str(category), LEAF_GENERAL_REASONING)
    folded = f" {str(category or '').casefold()} "
    for leaf, needles in _CATEGORY_RULES:
        if any(needle in folded for needle in needles):
            return leaf
    upstream_folded = str(upstream or "").casefold()
    if "tool-use" in upstream_folded:
        return LEAF_TOOL_USE
    if any(marker in upstream_folded for marker in ("claude-code", "coding", "agentic")):
        return LEAF_AGENTIC_CODE
    if any(marker in upstream_folded for marker in ("reasoning", "cot", "traces")):
        return LEAF_GENERAL_REASONING
    return LEAF_STRICT_INSTRUCTION


def _canonical_assistant_content(
    content: str, thinking: object, *, has_tool_calls: bool
) -> tuple[str, bool] | None:
    if any(marker in content for marker in _RESERVED_CHAT_MARKERS):
        return None
    open_count = content.count("<think>")
    close_count = content.count("</think>")
    if open_count != close_count or open_count > 1:
        return None
    explicit_trace: str | None = None
    final = content
    if open_count == 1:
        open_at = content.find("<think>")
        close_at = content.find("</think>")
        if close_at < open_at:
            return None
        preamble = _clean_text(content[:open_at])
        tagged = _clean_text(content[open_at + len("<think>") : close_at])
        explicit_trace = "\n".join(part for part in (preamble, tagged) if part) or None
        final = _clean_text(content[close_at + len("</think>") :]) or ""
        if explicit_trace is None:
            return None
    separate_trace = _clean_text(thinking)
    trace = "\n".join(part for part in (separate_trace, explicit_trace) if part) or None
    if has_tool_calls and not final:
        if trace is not None:
            return f"<think>\n{trace}\n</think>\n\n", True
        return "<think>\n\n</think>\n\n", False
    if not final:
        return None
    if trace is not None:
        return f"<think>\n{trace}\n</think>\n\n{final}", True
    return f"<think>\n\n</think>\n\n{final}", False


def _normalize_messages(
    raw_messages: object,
    *,
    top_level_reasoning: object = None,
    top_reasoning_target_index: int | None = None,
) -> tuple[list[dict[str, Any]], list[bool], bool, str] | tuple[None, None, None, str]:
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, None, None, "invalid_messages_shape"
    messages: list[dict[str, Any]] = []
    trace_flags: list[bool] = []
    has_tool = False
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            return None, None, None, "invalid_message"
        role = raw.get("role")
        content = raw.get("content", "")
        if role not in _ALLOWED_MESSAGE_ROLES or not isinstance(content, str):
            return None, None, None, "invalid_message"
        cleaned = _clean_text(content) or ""
        normalized: dict[str, Any] = {"role": role, "content": cleaned}
        has_tool_calls = bool(raw.get("tool_calls"))
        if role == "assistant":
            reasoning = (
                top_level_reasoning
                if top_reasoning_target_index == index
                else raw.get("thinking") or raw.get("reasoning_content") or raw.get("reasoning")
            )
            canonical = _canonical_assistant_content(
                cleaned, reasoning, has_tool_calls=has_tool_calls
            )
            if canonical is None:
                return None, None, None, "malformed_or_empty_assistant_target"
            normalized["content"], has_trace = canonical
            trace_flags.append(has_trace)
            if has_tool_calls:
                tool_calls = raw.get("tool_calls")
                if not isinstance(tool_calls, list):
                    return None, None, None, "invalid_tool_calls"
                normalized["tool_calls"] = tool_calls
                has_tool = True
        else:
            trace_flags.append(False)
        if role == "tool":
            tool_call_id = raw.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                return None, None, None, "invalid_tool_message"
            normalized["tool_call_id"] = tool_call_id
            has_tool = True
        if "[Tool:" in cleaned or "[Tool result]" in cleaned:
            has_tool = True
        messages.append(normalized)
    return messages, trace_flags, has_tool, "ok"


def _first_user_prompt(messages: Sequence[Mapping[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") == "user":
            prompt = _clean_text(message.get("content"))
            if prompt is not None:
                return prompt
    return None


def _contract(has_tool: bool, has_trace: bool) -> str:
    if has_tool:
        return CONTRACT_TOOL_TRACE if has_trace else CONTRACT_TOOL_DIRECT
    return CONTRACT_TRACE if has_trace else CONTRACT_DIRECT


def _messages_have_tool_activity(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") == "tool" or bool(message.get("tool_calls")):
            return True
        content = message.get("content")
        if isinstance(content, str) and ("[Tool:" in content or "[Tool result]" in content):
            return True
    return False


def _token_ids(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("chat template must return a non-empty flat token list")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError("chat template returned invalid token IDs")
    return list(value)


def render_token_ids(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
) -> tuple[list[int], list[int]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("training messages must end with an assistant target")
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
        raise ValueError("chat template has an inconsistent prompt/training prefix")
    return prompt_ids, full_ids


def _tier_for_length(token_count: int) -> str:
    for tier in TRAINING_TIERS:
        minimum, maximum = TIER_LIMITS[tier]
        if minimum <= token_count <= maximum:
            return tier
    return TIER_DEFERRED


def _payload(messages: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps(
        {"messages": list(messages)}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _select_even_targets(indices: Sequence[int], limit: int) -> tuple[list[int], int]:
    if len(indices) <= limit:
        return list(indices), 0
    if limit == 1:
        return [indices[-1]], len(indices) - 1
    positions = {round(position * (len(indices) - 1) / (limit - 1)) for position in range(limit)}
    selected = [indices[position] for position in sorted(positions)]
    if len(selected) < limit:
        for index in reversed(indices):
            if index not in selected:
                selected.append(index)
            if len(selected) == limit:
                break
        selected.sort()
    return selected, len(indices) - len(selected)


def _bounded_premium_slice(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, Any]],
    *,
    target_index: int,
    enable_thinking: bool,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], int, int] | None:
    prefix = messages[: target_index + 1]
    _, full_ids = render_token_ids(tokenizer, prefix, enable_thinking=enable_thinking)
    if len(full_ids) <= max_tokens:
        return prefix, len(full_ids), 0

    system_messages = [message for message in prefix if message.get("role") == "system"]
    user_indices = [
        index for index, message in enumerate(prefix[:-1]) if message.get("role") == "user"
    ]
    if not user_indices:
        return None
    latest_user = user_indices[-1]
    base = [*system_messages, prefix[latest_user]]
    tail_start = latest_user + 1
    character_budget = max_tokens * 2
    running = len(json.dumps(base, ensure_ascii=False))
    for index in range(target_index, latest_user, -1):
        running += len(json.dumps(prefix[index], ensure_ascii=False))
        tail_start = index
        if running >= character_budget:
            break

    while tail_start <= target_index:
        tail = [
            message
            for message in prefix[tail_start : target_index + 1]
            if message.get("role") not in {"system", "user"}
        ]
        candidate = [*base, *tail]
        if candidate[-1] is not prefix[target_index] and candidate[-1] != prefix[target_index]:
            candidate.append(prefix[target_index])
        try:
            _, ids = render_token_ids(tokenizer, candidate, enable_thinking=enable_thinking)
        except ValueError:
            tail_start += 1
            continue
        if len(ids) <= max_tokens:
            return candidate, len(ids), len(prefix) - len(candidate)
        step = max(1, (target_index - tail_start) // 3)
        tail_start += step

    minimal = [prefix[latest_user], prefix[target_index]]
    try:
        _, ids = render_token_ids(tokenizer, minimal, enable_thinking=enable_thinking)
    except ValueError:
        return None
    if len(ids) > max_tokens:
        return None
    return minimal, len(ids), len(prefix) - len(minimal)


def _benchmark_rejection(
    prompt: str, benchmark_index: foundation.BenchmarkIndex
) -> tuple[str | None, str, str]:
    prompt_group = prompt_group_id(prompt)
    family_group = split_family_id(prompt)
    if prompt_group in benchmark_index.prompt_groups:
        return "benchmark_exact_prompt_overlap", prompt_group, family_group
    if family_group in benchmark_index.family_groups:
        return "benchmark_split_family_overlap", prompt_group, family_group
    return None, prompt_group, family_group


def _write_jsonl_stream(
    connection: sqlite3.Connection,
    path: Path,
    *,
    leaf: str,
    contract: str,
    tier: str,
    split: str,
) -> dict[str, Any]:
    records = 0
    token_total = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        cursor = connection.execute(
            """
            SELECT c.payload, c.token_count
            FROM candidates c
            JOIN selected_keys s ON s.content_hash = c.content_hash
            WHERE c.leaf=? AND c.contract=? AND c.tier=? AND c.split=?
            ORDER BY c.order_rank, c.content_hash
            """,
            (leaf, contract, tier, split),
        )
        for payload, token_count in cursor:
            output.write(str(payload))
            output.write("\n")
            records += 1
            token_total += int(token_count)
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "records": records,
        "rendered_tokens": token_total,
    }


def _stream_contracts() -> dict[str, Any]:
    mode_contracts = {
        CONTRACT_DIRECT: {
            "enable_thinking": False,
            "target": "explicit empty-thinking assistant answer",
            "tool_messages": False,
        },
        CONTRACT_TRACE: {
            "enable_thinking": True,
            "target": "explicit non-empty thinking plus final answer",
            "tool_messages": False,
        },
        CONTRACT_TOOL_DIRECT: {
            "enable_thinking": False,
            "target": "assistant answer or tool call in a tool-bearing trajectory",
            "tool_messages": True,
        },
        CONTRACT_TOOL_TRACE: {
            "enable_thinking": True,
            "target": "thinking-bearing assistant answer or tool call in a tool trajectory",
            "tool_messages": True,
        },
    }
    tiers = {
        tier: {
            "minimum_rendered_tokens": minimum,
            "maximum_rendered_tokens": maximum,
            "truncation": False,
        }
        for tier, (minimum, maximum) in TIER_LIMITS.items()
    }
    tiers[TIER_DEFERRED] = {
        "minimum_rendered_tokens": 32769,
        "maximum_rendered_tokens": None,
        "emitted": False,
        "truncation": False,
    }
    return {"modes": mode_contracts, "length_tiers": tiers}


def _write_publication(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    manifest: dict[str, Any],
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        stream_rows = connection.execute(
            """
            SELECT DISTINCT c.leaf, c.contract, c.tier
            FROM candidates c JOIN selected_keys s ON s.content_hash=c.content_hash
            """
        ).fetchall()
        leaf_order = {leaf: index for index, leaf in enumerate(LEAF_NAMES)}
        contract_order = {name: index for index, name in enumerate(CONTRACT_NAMES)}
        tier_order = {name: index for index, name in enumerate(TRAINING_TIERS)}
        streams = sorted(
            ((str(row[0]), str(row[1]), str(row[2])) for row in stream_rows),
            key=lambda item: (
                leaf_order[item[0]],
                contract_order[item[1]],
                tier_order[item[2]],
            ),
        )
        receipts: dict[str, dict[str, Any]] = defaultdict(dict)
        for leaf, contract, tier in streams:
            stream_name = f"{contract}-{tier}"
            stream_dir = stage / leaf / stream_name
            stream_dir.mkdir(parents=True)
            split_receipts: dict[str, Any] = {}
            for split in SPLIT_NAMES:
                path = stream_dir / f"{split}.jsonl"
                receipt = _write_jsonl_stream(
                    connection,
                    path,
                    leaf=leaf,
                    contract=contract,
                    tier=tier,
                    split=split,
                )
                receipt["path"] = path.relative_to(stage).as_posix()
                split_receipts[split] = receipt
            receipts[leaf][stream_name] = split_receipts

        mixture = {
            "schema_version": MIXTURE_SCHEMA_VERSION,
            "source_schema_version": SCHEMA_VERSION,
            "leaf_order": list(LEAF_NAMES),
            "leaf_alias_map": LEAF_ALIAS_MAP,
            "stream_contracts": _stream_contracts(),
            "streams": dict(receipts),
            "trainer_row_contract": "top-level object contains exactly messages",
            "provenance_location": "aggregate sidecars only; never embedded in training rows",
            "sidecar_contains_raw_examples": False,
        }
        mixture_path = stage / "mixture-manifest.json"
        mixture_path.write_text(
            json.dumps(mixture, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest["outputs"] = dict(receipts)
        manifest["sidecars"] = {
            "mixture_manifest": {
                "path": mixture_path.relative_to(stage).as_posix(),
                "sha256": _sha256_file(mixture_path),
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
        (stage / "sha256sums.txt").write_text(
            "".join(
                f"{_sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
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


def materialize_broad_distillation_experts(
    *,
    input_paths: Mapping[str, Sequence[str | Path]],
    output_dir: str | Path,
    tokenizer: ChatTemplateTokenizer,
    tokenizer_path: str | Path,
    expected_input_hashes: Mapping[str | Path, str],
    benchmark_suite_paths: Sequence[str | Path],
    seed: str = DEFAULT_SEED,
    max_variants_per_prompt_stream: int = DEFAULT_MAX_VARIANTS_PER_PROMPT_STREAM,
    premium_targets_per_session: int = DEFAULT_PREMIUM_TARGETS_PER_SESSION,
    premium_context_tokens: int = DEFAULT_PREMIUM_CONTEXT_TOKENS,
) -> Path:
    """Build the granular broad-corpus rack and return its quality manifest path."""

    if not seed:
        raise ValueError("seed must be non-empty")
    for name, value in (
        ("max_variants_per_prompt_stream", max_variants_per_prompt_stream),
        ("premium_targets_per_session", premium_targets_per_session),
        ("premium_context_tokens", premium_context_tokens),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if premium_context_tokens > 32768:
        raise ValueError("premium_context_tokens must be at most 32768")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional local tooling
        raise RuntimeError("pyarrow is required to read Fable Premium") from exc

    paths, input_receipts = _verified_inputs(input_paths, expected_input_hashes)
    benchmark_index = foundation._load_benchmark_index(benchmark_suite_paths)
    tokenizer_receipt = foundation._tokenizer_identity(tokenizer_path)
    tokenizer_receipt["chat_template_modes"] = {
        contract: {"enable_thinking": contract in {CONTRACT_TRACE, CONTRACT_TOOL_TRACE}}
        for contract in CONTRACT_NAMES
    }

    source_input: Counter[str] = Counter()
    source_guarded: Counter[tuple[str, str]] = Counter()
    source_rejected: Counter[tuple[str, str]] = Counter()
    source_auxiliary: Counter[str] = Counter()
    source_eligible: Counter[str] = Counter()
    example_rejected: Counter[tuple[str, str]] = Counter()
    assigned: Counter[tuple[str, str]] = Counter()
    assigned_contract: Counter[tuple[str, str]] = Counter()
    deferred: Counter[tuple[str, str]] = Counter()
    deferred_contract: Counter[tuple[str, str]] = Counter()
    candidate_attempts: Counter[str] = Counter()
    duplicate_losers: Counter[tuple[str, str]] = Counter()
    contamination: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[tuple[str, str]] = Counter()
    upstream_counts: Counter[tuple[str, str]] = Counter()
    assigned_category_routes: Counter[tuple[str, str, str]] = Counter()
    assigned_source_routes: Counter[tuple[str, str, str]] = Counter()
    rendered_lengths: dict[str, list[int]] = defaultdict(list)
    deferred_lengths: dict[str, list[int]] = defaultdict(list)
    premium_stats: Counter[str] = Counter()
    premium_context_drops = 0

    def guard_source(corpus: str, record: Mapping[str, object]) -> bool:
        reason = metadata_security_gate(record)
        if reason is None:
            return False
        source_guarded[(corpus, f"security_metadata_{reason}")] += 1
        return True

    with tempfile.TemporaryDirectory(prefix="molor-broad-distillation-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "candidates.sqlite3")
        try:
            _create_candidate_table(connection)

            def admit(example: PreparedExample, *, precomputed_tokens: int | None = None) -> None:
                nonlocal premium_context_drops
                reject_reason, prompt_group, family_group = _benchmark_rejection(
                    example.prompt, benchmark_index
                )
                if reject_reason is not None:
                    example_rejected[(example.corpus, reject_reason)] += 1
                    contamination[(example.corpus, reject_reason)] += 1
                    return
                enable_thinking = example.contract in {
                    CONTRACT_TRACE,
                    CONTRACT_TOOL_TRACE,
                }
                try:
                    if precomputed_tokens is None:
                        _, ids = render_token_ids(
                            tokenizer,
                            example.messages,
                            enable_thinking=enable_thinking,
                        )
                        token_count = len(ids)
                    else:
                        token_count = precomputed_tokens
                except (TypeError, ValueError):
                    example_rejected[(example.corpus, "chat_template_error")] += 1
                    return
                assigned[(example.corpus, example.leaf)] += 1
                assigned_contract[(example.corpus, example.contract)] += 1
                route = f"{example.leaf}/{example.contract}"
                assigned_category_routes[(example.corpus, example.metadata_category, route)] += 1
                assigned_source_routes[(example.corpus, example.metadata_source, route)] += 1
                tier = _tier_for_length(token_count)
                rendered_lengths[example.corpus].append(token_count)
                if tier == TIER_DEFERRED or (
                    example.corpus == CORPUS_FABLE_PREMIUM and token_count > premium_context_tokens
                ):
                    deferred[(example.corpus, example.leaf)] += 1
                    deferred_contract[(example.corpus, example.contract)] += 1
                    deferred_lengths[example.corpus].append(token_count)
                    return

                payload = _payload(example.messages)
                content_hash = _sha256_bytes(payload.encode("utf-8"))
                order_rank = _sha256_bytes(
                    f"{seed}\0order\0{prompt_group}\0{content_hash}".encode("utf-8")
                )
                candidate = Candidate(
                    content_hash=content_hash,
                    prompt_group=prompt_group,
                    family_group=family_group,
                    split=split_for_family(family_group, seed=seed),
                    corpus=example.corpus,
                    leaf=example.leaf,
                    contract=example.contract,
                    tier=tier,
                    token_count=token_count,
                    quality_score=example.quality_score,
                    source_group=example.source_group,
                    order_rank=order_rank,
                    payload=payload,
                )
                candidate_attempts[example.corpus] += 1
                loser = _upsert_candidate(connection, candidate)
                if loser is not None:
                    duplicate_losers[(loser.corpus, loser.leaf)] += 1

            # Category-first prepared corpora.  The metadata gate appears before messages access.
            for corpus in (CORPUS_MYTHOS, CORPUS_COMPLETE_FABLE):
                for path in paths[corpus]:
                    with path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            source_input[corpus] += 1
                            if not line.strip():
                                source_rejected[(corpus, "blank_line")] += 1
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                source_rejected[(corpus, "invalid_json")] += 1
                                continue
                            if not isinstance(record, Mapping):
                                source_rejected[(corpus, "row_not_object")] += 1
                                continue
                            if guard_source(corpus, record):
                                continue
                            category = record.get("category")
                            upstream = record.get("upstream")
                            metadata_category = _aggregate_label(category)
                            metadata_source = _aggregate_label(upstream or record.get("source"))
                            category_counts[(corpus, metadata_category)] += 1
                            upstream_counts[(corpus, metadata_source)] += 1
                            raw_messages = record.get("messages")
                            messages, trace_flags, has_tool, reason = _normalize_messages(
                                raw_messages
                            )
                            if messages is None or trace_flags is None or has_tool is None:
                                source_rejected[(corpus, reason)] += 1
                                continue
                            if messages[-1].get("role") != "assistant":
                                source_rejected[
                                    (corpus, "conversation_not_assistant_terminated")
                                ] += 1
                                continue
                            prompt = _first_user_prompt(messages)
                            if prompt is None:
                                source_rejected[(corpus, "missing_user_prompt")] += 1
                                continue
                            reject_reason, _, _ = _benchmark_rejection(prompt, benchmark_index)
                            if reject_reason is not None:
                                source_rejected[(corpus, reject_reason)] += 1
                                contamination[(corpus, reject_reason)] += 1
                                continue
                            source_eligible[corpus] += 1
                            contract = _contract(has_tool, any(trace_flags))
                            leaf = (
                                LEAF_TOOL_USE
                                if has_tool
                                else _leaf_for_category(corpus, category, upstream)
                            )
                            admit(
                                PreparedExample(
                                    corpus=corpus,
                                    leaf=leaf,
                                    contract=contract,
                                    prompt=prompt,
                                    messages=messages,
                                    quality_score=0.0,
                                    source_group=str(upstream or record.get("source") or corpus),
                                    metadata_category=metadata_category,
                                    metadata_source=metadata_source,
                                )
                            )

            # Fable-Max is metadata-accounted but its messages/thinking payload is never accessed.
            corpus = CORPUS_FABLE_MAX_AUXILIARY
            for path in paths[corpus]:
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        source_input[corpus] += 1
                        if not line.strip():
                            source_rejected[(corpus, "blank_line")] += 1
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            source_rejected[(corpus, "invalid_json")] += 1
                            continue
                        if not isinstance(record, Mapping):
                            source_rejected[(corpus, "row_not_object")] += 1
                            continue
                        if guard_source(corpus, record):
                            continue
                        metadata_category = _aggregate_label(record.get("category"))
                        metadata_source = _aggregate_label(
                            record.get("upstream") or record.get("source")
                        )
                        category_counts[(corpus, metadata_category)] += 1
                        upstream_counts[(corpus, metadata_source)] += 1
                        source_auxiliary[corpus] += 1

            # Premium is read once from canonical OpenAI-chat Parquet.
            corpus = CORPUS_FABLE_PREMIUM
            required_premium_columns = {
                "messages",
                "model",
                "quality_scores",
                "source",
                "source_row_hash",
            }
            for path in paths[corpus]:
                parquet = pq.ParquetFile(path)
                columns = set(parquet.schema_arrow.names)
                if not required_premium_columns <= columns:
                    raise ValueError(f"premium Parquet schema is missing required columns: {path}")
                selected_columns = list(parquet.schema_arrow.names)
                for batch in parquet.iter_batches(batch_size=64, columns=selected_columns):
                    for record in batch.to_pylist():
                        source_input[corpus] += 1
                        if guard_source(corpus, record):
                            continue
                        metadata_category = _aggregate_label(record.get("category"))
                        metadata_source = _aggregate_label(record.get("source"))
                        category_counts[(corpus, metadata_category)] += 1
                        upstream_counts[(corpus, metadata_source)] += 1
                        raw_messages = record.get("messages")
                        if not isinstance(raw_messages, str):
                            source_rejected[(corpus, "messages_not_json_string")] += 1
                            continue
                        try:
                            parsed_messages = json.loads(raw_messages)
                        except json.JSONDecodeError:
                            source_rejected[(corpus, "invalid_messages_json")] += 1
                            continue
                        assistant_indices = (
                            [
                                index
                                for index, message in enumerate(parsed_messages)
                                if isinstance(message, Mapping)
                                and message.get("role") == "assistant"
                            ]
                            if isinstance(parsed_messages, list)
                            else []
                        )
                        if not assistant_indices:
                            source_rejected[(corpus, "no_assistant_targets")] += 1
                            continue
                        top_reasoning = _parse_top_reasoning(record.get("reasoning"))
                        top_reasoning_target = assistant_indices[-1] if top_reasoning else None
                        messages, trace_flags, session_has_tool, reason = _normalize_messages(
                            parsed_messages,
                            top_level_reasoning=top_reasoning,
                            top_reasoning_target_index=top_reasoning_target,
                        )
                        if messages is None or trace_flags is None or session_has_tool is None:
                            source_rejected[(corpus, reason)] += 1
                            continue
                        prompt = _first_user_prompt(messages)
                        if prompt is None:
                            source_rejected[(corpus, "missing_user_prompt")] += 1
                            continue
                        reject_reason, _, _ = _benchmark_rejection(prompt, benchmark_index)
                        if reject_reason is not None:
                            source_rejected[(corpus, reject_reason)] += 1
                            contamination[(corpus, reject_reason)] += 1
                            continue
                        source_eligible[corpus] += 1
                        premium_stats["sessions"] += 1
                        premium_stats["assistant_targets_total"] += len(assistant_indices)
                        selected_targets, capped = _select_even_targets(
                            assistant_indices, premium_targets_per_session
                        )
                        premium_stats["assistant_targets_selected"] += len(selected_targets)
                        premium_stats["assistant_targets_capped"] += capped
                        quality = _parse_quality(record.get("quality_scores"))
                        source_group = str(record.get("source_row_hash") or "<missing-session-id>")
                        for target_index in selected_targets:
                            target_trace = trace_flags[target_index]
                            bounded = _bounded_premium_slice(
                                tokenizer,
                                messages,
                                target_index=target_index,
                                enable_thinking=target_trace,
                                max_tokens=premium_context_tokens,
                            )
                            if bounded is None:
                                example_rejected[(corpus, "unable_to_form_bounded_slice")] += 1
                                continue
                            sliced, token_count, dropped_messages = bounded
                            premium_context_drops += dropped_messages
                            slice_has_tool = _messages_have_tool_activity(sliced)
                            contract = _contract(slice_has_tool, target_trace)
                            admit(
                                PreparedExample(
                                    corpus=corpus,
                                    leaf=LEAF_TOOL_USE if slice_has_tool else LEAF_AGENTIC_CODE,
                                    contract=contract,
                                    prompt=prompt,
                                    messages=sliced,
                                    quality_score=quality,
                                    source_group=source_group,
                                    metadata_category=metadata_category,
                                    metadata_source=metadata_source,
                                ),
                                precomputed_tokens=token_count,
                            )

            connection.commit()
            unique_candidates = int(
                connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            )
            connection.execute(
                """
                CREATE TABLE selected_keys AS
                SELECT content_hash FROM (
                  SELECT content_hash,
                    ROW_NUMBER() OVER (
                      PARTITION BY prompt_group, leaf, contract, tier
                      ORDER BY quality_score DESC, token_count ASC, order_rank, content_hash
                    ) AS variant_rank
                  FROM candidates
                ) WHERE variant_rank <= ?
                """,
                (max_variants_per_prompt_stream,),
            )
            connection.execute(
                "CREATE UNIQUE INDEX selected_keys_hash ON selected_keys(content_hash)"
            )
            emitted = int(connection.execute("SELECT COUNT(*) FROM selected_keys").fetchone()[0])
            capped_rows = connection.execute(
                """
                SELECT c.corpus, c.leaf, COUNT(*)
                FROM candidates c LEFT JOIN selected_keys s ON s.content_hash=c.content_hash
                WHERE s.content_hash IS NULL GROUP BY c.corpus, c.leaf
                """
            ).fetchall()
            variant_capped: Counter[tuple[str, str]] = Counter(
                {(str(corpus), str(leaf)): int(count) for corpus, leaf, count in capped_rows}
            )

            split_overlap = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT family_group FROM candidates
                      GROUP BY family_group HAVING COUNT(DISTINCT split) > 1
                    )
                    """
                ).fetchone()[0]
            )
            if split_overlap:
                raise AssertionError("prompt families crossed deterministic splits")

            input_total = sum(source_input.values())
            guarded_total = sum(source_guarded.values())
            source_rejected_total = sum(source_rejected.values())
            auxiliary_total = sum(source_auxiliary.values())
            eligible_total = sum(source_eligible.values())
            if (
                input_total
                != guarded_total + source_rejected_total + auxiliary_total + eligible_total
            ):
                raise AssertionError("source-row disposition accounting does not balance")
            example_rejected_total = sum(example_rejected.values())
            assigned_total = sum(assigned.values())
            prepared_opportunities = (
                source_eligible[CORPUS_MYTHOS] + source_eligible[CORPUS_COMPLETE_FABLE]
            )
            premium_opportunities = premium_stats["assistant_targets_selected"]
            if prepared_opportunities + premium_opportunities != (
                assigned_total + example_rejected_total
            ):
                raise AssertionError("example assignment accounting does not balance")
            deferred_total = sum(deferred.values())
            candidate_attempt_total = sum(candidate_attempts.values())
            if assigned_total != deferred_total + candidate_attempt_total:
                raise AssertionError("assigned/deferred/candidate accounting does not balance")
            duplicate_total = sum(duplicate_losers.values())
            if candidate_attempt_total != unique_candidates + duplicate_total:
                raise AssertionError("exact-content dedup accounting does not balance")
            variant_capped_total = sum(variant_capped.values())
            if unique_candidates != emitted + variant_capped_total:
                raise AssertionError("variant-cap accounting does not balance")
            if premium_stats["assistant_targets_total"] != (
                premium_stats["assistant_targets_selected"]
                + premium_stats["assistant_targets_capped"]
            ):
                raise AssertionError("premium target cap accounting does not balance")

            selected_counts = connection.execute(
                """
                SELECT c.leaf, c.contract, c.tier, c.corpus, c.split,
                       COUNT(*), SUM(c.token_count), MIN(c.token_count), MAX(c.token_count)
                FROM candidates c JOIN selected_keys s ON s.content_hash=c.content_hash
                GROUP BY c.leaf, c.contract, c.tier, c.corpus, c.split
                """
            ).fetchall()
            emitted_by_leaf: Counter[str] = Counter()
            emitted_by_corpus: Counter[str] = Counter()
            emitted_by_split: Counter[str] = Counter()
            stream_counts: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
            for (
                leaf,
                contract,
                tier,
                corpus,
                split,
                count,
                tokens,
                minimum,
                maximum,
            ) in selected_counts:
                count = int(count)
                emitted_by_leaf[str(leaf)] += count
                emitted_by_corpus[str(corpus)] += count
                emitted_by_split[str(split)] += count
                stream_counts[str(leaf)][f"{contract}-{tier}"][f"{corpus}/{split}"] = {
                    "records": count,
                    "rendered_tokens": int(tokens),
                    "minimum_rendered_tokens": int(minimum),
                    "maximum_rendered_tokens": int(maximum),
                }

            builder_path = Path(__file__).resolve()
            foundation_path = Path(foundation.__file__).resolve()
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "provenance": {
                    "inputs": input_receipts,
                    "benchmark_suites": list(benchmark_index.receipts),
                    "tokenizer": tokenizer_receipt,
                    "builder": {
                        "path": str(builder_path),
                        "sha256": _sha256_file(builder_path),
                    },
                    "foundation_builder": {
                        "path": str(foundation_path),
                        "sha256": _sha256_file(foundation_path),
                    },
                    "embedded_in_training_rows": False,
                },
                "contract": {
                    "seed": seed,
                    "leaf_order": list(LEAF_NAMES),
                    "leaf_alias_map": LEAF_ALIAS_MAP,
                    "stream_contracts": _stream_contracts(),
                    "prompt_group": "Unicode NFKC + casefold + collapsed whitespace",
                    "split_family": "first 256 normalized prompt characters, SHA-256",
                    "split_policy": "seeded SHA-256 family assignment, 90/5/5",
                    "exact_content_dedup": "SHA-256 of canonical messages-only payload",
                    "variant_cap": {
                        "maximum_per_prompt_leaf_contract_tier": (max_variants_per_prompt_stream),
                        "ranking": "quality descending, token count ascending, seeded order hash",
                    },
                    "premium": {
                        "canonical_representation": "openai_chat Parquet only",
                        "maximum_targets_per_session": premium_targets_per_session,
                        "maximum_context_tokens": premium_context_tokens,
                        "selection": "deterministic even coverage including first and last target",
                        "session_and_split_family_grouping": True,
                    },
                    "fable_max": "aggregate auxiliary archive only; no messages SFT emitted",
                    "truncation": False,
                    "training_rows": "top-level messages only",
                },
                "quality": {
                    "coverage": {
                        "input_rows": input_total,
                        "input_by_corpus": dict(sorted(source_input.items())),
                        "guarded_rows": guarded_total,
                        "guarded_by_corpus_and_reason": _nested_counts(source_guarded),
                        "source_rejected_rows": source_rejected_total,
                        "source_rejected_by_corpus_and_reason": _nested_counts(source_rejected),
                        "auxiliary_archive_rows": auxiliary_total,
                        "auxiliary_by_corpus": dict(sorted(source_auxiliary.items())),
                        "eligible_source_rows": eligible_total,
                        "eligible_by_corpus": dict(sorted(source_eligible.items())),
                        "example_rejections": example_rejected_total,
                        "example_rejected_by_corpus_and_reason": _nested_counts(example_rejected),
                        "assigned_examples": assigned_total,
                        "assigned_by_corpus_and_leaf": _nested_counts(assigned),
                        "assigned_by_corpus_and_contract": _nested_counts(assigned_contract),
                        "deferred_examples": deferred_total,
                        "deferred_by_corpus_and_leaf": _nested_counts(deferred),
                        "deferred_by_corpus_and_contract": _nested_counts(deferred_contract),
                        "candidate_attempts": candidate_attempt_total,
                        "candidate_attempts_by_corpus": dict(sorted(candidate_attempts.items())),
                        "exact_content_duplicates_removed": duplicate_total,
                        "duplicate_losers_by_corpus_and_leaf": _nested_counts(duplicate_losers),
                        "unique_candidates_after_dedup": unique_candidates,
                        "variant_capped": variant_capped_total,
                        "variant_capped_by_corpus_and_leaf": _nested_counts(variant_capped),
                        "emitted_examples": emitted,
                        "emitted_by_leaf": dict(sorted(emitted_by_leaf.items())),
                        "emitted_by_corpus": dict(sorted(emitted_by_corpus.items())),
                        "emitted_by_split": {
                            split: emitted_by_split[split] for split in SPLIT_NAMES
                        },
                        "source_row_equation": {
                            "input": input_total,
                            "guarded_plus_rejected_plus_auxiliary_plus_eligible": (
                                guarded_total
                                + source_rejected_total
                                + auxiliary_total
                                + eligible_total
                            ),
                            "balanced": True,
                        },
                        "example_assignment_equation": {
                            "opportunities": prepared_opportunities + premium_opportunities,
                            "assigned_plus_rejected": assigned_total + example_rejected_total,
                            "balanced": True,
                        },
                        "assignment_equation": {
                            "assigned": assigned_total,
                            "deferred_plus_candidates": deferred_total + candidate_attempt_total,
                            "balanced": True,
                        },
                        "dedup_equation": {
                            "candidate_attempts": candidate_attempt_total,
                            "unique_plus_duplicates": unique_candidates + duplicate_total,
                            "balanced": True,
                        },
                        "cap_equation": {
                            "unique_candidates": unique_candidates,
                            "emitted_plus_capped": emitted + variant_capped_total,
                            "balanced": True,
                        },
                        "zero_silent_drops": True,
                    },
                    "premium_target_accounting": {
                        **dict(sorted(premium_stats.items())),
                        "context_messages_dropped": premium_context_drops,
                        "balanced": True,
                    },
                    "rendered_token_stats_by_corpus": {
                        corpus: _token_stats(values)
                        for corpus, values in sorted(rendered_lengths.items())
                    },
                    "deferred_token_stats_by_corpus": {
                        corpus: _token_stats(values)
                        for corpus, values in sorted(deferred_lengths.items())
                    },
                    "safe_metadata": {
                        "category_distribution": _nested_counts(category_counts),
                        "source_or_upstream_distribution": _nested_counts(upstream_counts),
                        "assigned_routes_by_category": _nested_route_counts(
                            assigned_category_routes
                        ),
                        "assigned_routes_by_source_or_upstream": _nested_route_counts(
                            assigned_source_routes
                        ),
                        "long_metadata_values_are_hash_labeled": True,
                        "raw_examples_recorded": False,
                    },
                    "public_benchmark_contamination": {
                        "labels_by_corpus": _PUBLIC_CONTAMINATION_LABELS,
                        "local_suite_overlaps_rejected": _nested_counts(contamination),
                        "external_public_benchmark_content_scan": "not performed",
                        "evaluation_policy": (
                            "Treat public benchmark results as contaminated unless separately "
                            "screened; local frozen exact and 256-character family overlaps are "
                            "rejected."
                        ),
                    },
                    "stream_counts": {
                        leaf: dict(streams) for leaf, streams in stream_counts.items()
                    },
                    "global_split_family_overlap": split_overlap,
                    "all_emitted_within_32768_tokens": bool(
                        connection.execute(
                            """
                            SELECT COUNT(*)=0 FROM candidates c
                            JOIN selected_keys s ON s.content_hash=c.content_hash
                            WHERE c.token_count > 32768
                            """
                        ).fetchone()[0]
                    ),
                },
            }
            manifest_path = _write_publication(connection, Path(output_dir), manifest=manifest)
        finally:
            connection.close()
    return manifest_path


def _parse_input_assignments(values: Sequence[str]) -> dict[str, tuple[Path, ...]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError("custom inputs must use CORPUS=PATH")
        corpus, raw_path = value.split("=", 1)
        if corpus not in CORPUS_NAMES or not raw_path:
            raise ValueError(f"unknown or empty corpus assignment: {value!r}")
        grouped[corpus].append(Path(raw_path))
    if set(grouped) != set(CORPUS_NAMES):
        raise ValueError(f"custom inputs must assign every corpus in {CORPUS_NAMES!r}")
    return {corpus: tuple(grouped[corpus]) for corpus in CORPUS_NAMES}


def _parse_expected_hashes(values: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("expected input hashes must use PATH=SHA256")
        raw_path, digest = value.rsplit("=", 1)
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"invalid expected SHA-256 for {raw_path!r}")
        hashes[str(Path(raw_path).resolve())] = digest.casefold()
    return hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="CORPUS=PATH",
        help="explicit input; provide every pinned corpus/file or omit for defaults",
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
        help="explicit leakage suite; defaults to the two pinned Molor suites",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=foundation.DEFAULT_TOKENIZER)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--max-variants-per-prompt-stream",
        type=int,
        default=DEFAULT_MAX_VARIANTS_PER_PROMPT_STREAM,
    )
    parser.add_argument(
        "--premium-targets-per-session",
        type=int,
        default=DEFAULT_PREMIUM_TARGETS_PER_SESSION,
    )
    parser.add_argument(
        "--premium-context-tokens",
        type=int,
        default=DEFAULT_PREMIUM_CONTEXT_TOKENS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input:
        try:
            inputs = _parse_input_assignments(args.input)
            expected = _parse_expected_hashes(args.expected_input_sha256)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.expected_input_sha256:
            parser.error("custom expected hashes require explicit --input assignments")
        inputs = DEFAULT_INPUTS
        expected = DEFAULT_INPUT_HASHES
    suites = (
        tuple(args.benchmark_suite) if args.benchmark_suite else foundation.DEFAULT_BENCHMARK_SUITES
    )
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
    manifest = materialize_broad_distillation_experts(
        input_paths=inputs,
        output_dir=args.output,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer,
        expected_input_hashes=expected,
        benchmark_suite_paths=suites,
        seed=args.seed,
        max_variants_per_prompt_stream=args.max_variants_per_prompt_stream,
        premium_targets_per_session=args.premium_targets_per_session,
        premium_context_tokens=args.premium_context_tokens,
    )
    print(manifest)
    return 0


def _create_candidate_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE candidates (
            content_hash TEXT PRIMARY KEY,
            prompt_group TEXT NOT NULL,
            family_group TEXT NOT NULL,
            split TEXT NOT NULL,
            corpus TEXT NOT NULL,
            leaf TEXT NOT NULL,
            contract TEXT NOT NULL,
            tier TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            quality_score REAL NOT NULL,
            source_group TEXT NOT NULL,
            order_rank TEXT NOT NULL,
            payload TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _candidate_from_row(row: Sequence[Any]) -> Candidate:
    return Candidate(
        content_hash=str(row[0]),
        prompt_group=str(row[1]),
        family_group=str(row[2]),
        split=str(row[3]),
        corpus=str(row[4]),
        leaf=str(row[5]),
        contract=str(row[6]),
        tier=str(row[7]),
        token_count=int(row[8]),
        quality_score=float(row[9]),
        source_group=str(row[10]),
        order_rank=str(row[11]),
        payload=str(row[12]),
    )


def _candidate_values(candidate: Candidate) -> tuple[object, ...]:
    return (
        candidate.content_hash,
        candidate.prompt_group,
        candidate.family_group,
        candidate.split,
        candidate.corpus,
        candidate.leaf,
        candidate.contract,
        candidate.tier,
        candidate.token_count,
        candidate.quality_score,
        candidate.source_group,
        candidate.order_rank,
        candidate.payload,
    )


def _upsert_candidate(connection: sqlite3.Connection, candidate: Candidate) -> Candidate | None:
    prior_row = connection.execute(
        "SELECT * FROM candidates WHERE content_hash = ?", (candidate.content_hash,)
    ).fetchone()
    if prior_row is None:
        connection.execute(
            "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _candidate_values(candidate),
        )
        return None
    prior = _candidate_from_row(prior_row)
    candidate_wins = candidate.quality_score > prior.quality_score or (
        candidate.quality_score == prior.quality_score
        and (candidate.order_rank, candidate.corpus) < (prior.order_rank, prior.corpus)
    )
    if not candidate_wins:
        return candidate
    connection.execute(
        """
        UPDATE candidates SET
          prompt_group=?, family_group=?, split=?, corpus=?, leaf=?, contract=?, tier=?,
          token_count=?, quality_score=?, source_group=?, order_rank=?, payload=?
        WHERE content_hash=?
        """,
        (
            candidate.prompt_group,
            candidate.family_group,
            candidate.split,
            candidate.corpus,
            candidate.leaf,
            candidate.contract,
            candidate.tier,
            candidate.token_count,
            candidate.quality_score,
            candidate.source_group,
            candidate.order_rank,
            candidate.payload,
            candidate.content_hash,
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
            "p99": None,
            "max": None,
            "mean": None,
            "total": 0,
        }

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[max(0, index)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
        "total": sum(ordered),
    }


def _nested_counts(counter: Mapping[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (outer, inner), count in sorted(counter.items()):
        nested[outer][inner] = count
    return dict(nested)


def _nested_route_counts(
    counter: Mapping[tuple[str, str, str], int],
) -> dict[str, dict[str, dict[str, int]]]:
    nested: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (corpus, metadata_value, route), count in sorted(counter.items()):
        nested[corpus][metadata_value][route] = count
    return {
        corpus: {metadata: dict(routes) for metadata, routes in values.items()}
        for corpus, values in nested.items()
    }


def _parse_quality(value: object) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return 0.0
    overall = parsed.get("overall") if isinstance(parsed, Mapping) else None
    if isinstance(overall, bool) or not isinstance(overall, (int, float)):
        return 0.0
    score = float(overall)
    return score if math.isfinite(score) else 0.0


def _parse_top_reasoning(value: object) -> str | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned
    return _clean_text(parsed)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
