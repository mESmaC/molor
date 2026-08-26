"""Materialize the pinned software-architecture corpus into granular expert shards.

The builder has deliberately narrow authority.  It accepts exactly two pinned JSONL inputs,
refuses paths whose components name a cyber/security corpus, keeps the prepared training
partition in training, and splits only the prepared held-out partition.  Every input row reaches
one terminal state: emitted or represented by a content-free quarantine receipt.

Training shards contain exactly ``messages``.  Row-level lineage, prompt-family identities,
classification evidence, and rendered token counts live in separate provenance sidecars.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Protocol, Sequence
import unicodedata

from molor.experimental.length_tier_shards import (
    TIER_SPECS,
    sha256_file,
    tier_for_token_count,
    tokenizer_identity,
)


SCHEMA_VERSION = "molor.software-architecture-experts.v1"
DEFAULT_SEED = "molor-software-architecture-experts-v1"
EXPECTED_SOURCE = "software-architecture"

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

MODE_DIRECT = "direct"
MODE_TRACE = "trace"
MODE_TOOL = "tool"
MODES = (MODE_DIRECT, MODE_TRACE, MODE_TOOL)

LEAF_SOFTWARE_ARCHITECTURE_DESIGN = "software_architecture_design"
LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING = "systems_distributed_engineering"
LEAF_GENERAL_CODE_GENERATION = "general_code_generation"
LEAF_AGENTIC_CODE_DEBUGGING = "agentic_code_debugging"
LEAF_FRONTEND_VISUAL_COMPUTING = "frontend_visual_computing"
LEAF_ML_DATA_REASONING = "ml_data_reasoning"
LEAF_TERMINAL_CLI = "terminal_cli"
LEAVES = (
    LEAF_SOFTWARE_ARCHITECTURE_DESIGN,
    LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING,
    LEAF_GENERAL_CODE_GENERATION,
    LEAF_AGENTIC_CODE_DEBUGGING,
    LEAF_FRONTEND_VISUAL_COMPUTING,
    LEAF_ML_DATA_REASONING,
    LEAF_TERMINAL_CLI,
)

_TIER_NAMES = tuple(name for name, _, _ in TIER_SPECS)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_BLOCKED_PATH_COMPONENT = re.compile(r"(?:^|[-_. ])(?:cyber|security)(?:$|[-_. ])", re.I)
_BLOCKED_SOURCE_LABEL = re.compile(r"(?:^|[-_. ])(?:cyber|security)(?:$|[-_. ])", re.I)
_RESERVED_CHAT_MARKERS = ("<|im_start|>", "<|im_end|>")
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class ChatTemplateTokenizer(Protocol):
    """Tokenizer surface needed for exact full-chat length measurement."""

    def apply_chat_template(self, conversation: object, **kwargs: Any) -> object: ...


@dataclass(frozen=True, slots=True)
class InputPin:
    """Immutable identity for one source partition."""

    path: Path
    sha256: str
    record_count: int
    partition: str


@dataclass(frozen=True, slots=True)
class ValidatedRecord:
    """Validated input row whose message content may be emitted."""

    source_id: str
    messages: list[dict[str, str]]
    prompt: str
    mode: str


@dataclass(frozen=True, slots=True)
class PendingRecord:
    """Validated, classified row awaiting batch tokenization."""

    input_index: int
    input_partition: str
    line_number: int
    record_sha256: str
    source_id_sha256: str
    messages: list[dict[str, str]]
    prompt_group: str
    family_group: str
    split: str
    leaf: str
    classification_rules: tuple[str, ...]
    mode: str


@dataclass(frozen=True, slots=True)
class ClassificationFeature:
    """One auditable weighted lexical feature."""

    rule_id: str
    leaf: str
    weight: int
    pattern: re.Pattern[str]


def _feature(rule_id: str, leaf: str, weight: int, pattern: str) -> ClassificationFeature:
    return ClassificationFeature(rule_id, leaf, weight, re.compile(pattern, re.I))


# Specific vocabulary outweighs generic words.  Architecture is the explicit fallback, while
# its own features let genuinely architectural prompts beat weak single-word matches elsewhere.
_FEATURES = (
    _feature(
        "terminal.shell",
        LEAF_TERMINAL_CLI,
        8,
        r"\b(?:command[- ]line|cli|terminal|shell|bash|zsh|powershell|command prompt)\b",
    ),
    _feature(
        "terminal.commands",
        LEAF_TERMINAL_CLI,
        5,
        r"\b(?:shell script|command syntax|git command|docker command|linux command)\b",
    ),
    _feature(
        "frontend.frameworks",
        LEAF_FRONTEND_VISUAL_COMPUTING,
        8,
        r"\b(?:front[- ]?end|react|vue(?:\.js)?|angular|svelte|html|css|webgl|canvas)\b",
    ),
    _feature(
        "frontend.experience",
        LEAF_FRONTEND_VISUAL_COMPUTING,
        6,
        r"\b(?:user interface|ui design|user experience|responsive design|accessibility)\b",
    ),
    _feature(
        "frontend.visual",
        LEAF_FRONTEND_VISUAL_COMPUTING,
        5,
        r"\b(?:graphics?|rendering|visuali[sz]ation|computer vision|image processing)\b",
    ),
    _feature(
        "ml.modeling",
        LEAF_ML_DATA_REASONING,
        8,
        r"\b(?:machine learning|deep learning|artificial intelligence|neural networks?|"
        r"model training|feature engineering|natural language processing)\b",
    ),
    _feature(
        "ml.data-platform",
        LEAF_ML_DATA_REASONING,
        6,
        r"\b(?:data science|data analytics|data pipeline|data warehouse|data lake|etl|"
        r"big data|analytics pipeline|data architecture)\b",
    ),
    _feature(
        "debug.failures",
        LEAF_AGENTIC_CODE_DEBUGGING,
        8,
        r"\b(?:debug(?:ging)?|troubleshoot(?:ing)?|root cause|bug(?:s)?|exceptions?|"
        r"error handling|failure diagnosis)\b",
    ),
    _feature(
        "debug.testing",
        LEAF_AGENTIC_CODE_DEBUGGING,
        5,
        r"\b(?:unit test(?:ing)?|integration test(?:ing)?|end[- ]to[- ]end test(?:ing)?|"
        r"test automation|testing strateg(?:y|ies)|test plan|quality assurance)\b",
    ),
    _feature(
        "debug.delivery",
        LEAF_AGENTIC_CODE_DEBUGGING,
        5,
        r"\b(?:ci/cd|continuous integration|continuous delivery|code review|refactor(?:ing)?)\b",
    ),
    _feature(
        "debug.observability",
        LEAF_AGENTIC_CODE_DEBUGGING,
        3,
        r"\b(?:observability|monitoring|logging|profiling|diagnostics?)\b",
    ),
    _feature(
        "systems.distributed",
        LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING,
        9,
        r"\b(?:distributed systems?|distributed architecture|microservices?|serverless|"
        r"service mesh)\b",
    ),
    _feature(
        "systems.cloud-runtime",
        LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING,
        6,
        r"\b(?:cloud computing|cloud[- ]native|kubernetes|containers?|orchestration|"
        r"load balanc(?:er|ing)|auto[- ]?scal(?:e|ing))\b",
    ),
    _feature(
        "systems.reliability",
        LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING,
        6,
        r"\b(?:fault toleran(?:ce|t)|high availability|resilien(?:ce|t)|disaster recovery|"
        r"failover|replication|consensus|eventual consistency)\b",
    ),
    _feature(
        "systems.communication",
        LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING,
        5,
        r"\b(?:message queues?|message brokers?|event[- ]driven|publish[- ]subscribe|"
        r"asynchronous communication|distributed transaction)\b",
    ),
    _feature(
        "systems.performance",
        LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING,
        3,
        r"\b(?:scalability|horizontal scaling|vertical scaling|latency|throughput|"
        r"concurrency|parallelism|caching|performance bottleneck)\b",
    ),
    _feature(
        "systems.storage",
        LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING,
        3,
        r"\b(?:distributed database|database scaling|data consistency|sharding|partitioning)\b",
    ),
    _feature(
        "code.explicit-generation",
        LEAF_GENERAL_CODE_GENERATION,
        7,
        r"\b(?:write|generate|implement|develop)\b.{0,48}\b(?:code|function|class|program|api)\b",
    ),
    _feature(
        "code.languages",
        LEAF_GENERAL_CODE_GENERATION,
        6,
        r"\b(?:python|java(?:script)?|typescript|c\+\+|c#|rust|golang|ruby|php|swift|kotlin)\b",
    ),
    _feature(
        "code.algorithms",
        LEAF_GENERAL_CODE_GENERATION,
        5,
        r"\b(?:algorithm(?:s)?|data structures?|source code|programming)\b",
    ),
    _feature(
        "code.api-implementation",
        LEAF_GENERAL_CODE_GENERATION,
        3,
        r"\b(?:api implementation|sdk|library|framework implementation)\b",
    ),
    _feature(
        "architecture.patterns",
        LEAF_SOFTWARE_ARCHITECTURE_DESIGN,
        7,
        r"\b(?:architectural patterns?|design patterns?|model[- ]view[- ]controller|mvc|"
        r"hexagonal architecture|clean architecture|layered architecture|"
        r"domain[- ]driven design|event sourcing|cqrs)\b",
    ),
    _feature(
        "architecture.modularity",
        LEAF_SOFTWARE_ARCHITECTURE_DESIGN,
        6,
        r"\b(?:modularity|components?|modules?|coupling|cohesion|separation of concerns|"
        r"interfaces? and contracts?|solid principles?)\b",
    ),
    _feature(
        "architecture.process",
        LEAF_SOFTWARE_ARCHITECTURE_DESIGN,
        5,
        r"\b(?:architecture decision records?|design review|technical debt|"
        r"requirements analysis|system design|software design|architecture governance)\b",
    ),
    _feature(
        "architecture.general",
        LEAF_SOFTWARE_ARCHITECTURE_DESIGN,
        2,
        r"\b(?:software architecture|architectural|architecture)\b",
    ),
)

_TIE_PRECEDENCE = {
    LEAF_TERMINAL_CLI: 0,
    LEAF_FRONTEND_VISUAL_COMPUTING: 1,
    LEAF_ML_DATA_REASONING: 2,
    LEAF_AGENTIC_CODE_DEBUGGING: 3,
    LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING: 4,
    LEAF_GENERAL_CODE_GENERATION: 5,
    LEAF_SOFTWARE_ARCHITECTURE_DESIGN: 6,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_prompt(prompt: str) -> str:
    """Canonicalize a prompt for exact grouping without changing emitted text."""

    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(normalized.split())


def prompt_group_id(prompt: str) -> str:
    return sha256_bytes(normalize_prompt(prompt).encode("utf-8"))


def prompt_family_id(prompt: str) -> str:
    """Hash the first 256 normalized characters as the leakage/split family."""

    return sha256_bytes(normalize_prompt(prompt)[:256].encode("utf-8"))


def heldout_split_for_family(family_group: str, *, seed: str = DEFAULT_SEED) -> str:
    """Assign a held-out prompt family deterministically to validation or test."""

    if not seed:
        raise ValueError("seed must be non-empty")
    digest = hashlib.sha256(f"{seed}\0heldout\0{family_group}".encode()).digest()
    return SPLIT_VALIDATION if int.from_bytes(digest[:8], "big") % 2 == 0 else SPLIT_TEST


def classify_prompt(prompt: str) -> tuple[str, tuple[str, ...], dict[str, int]]:
    """Return a deterministic granular leaf, matched evidence, and all non-zero scores."""

    normalized = normalize_prompt(prompt)
    scores: Counter[str] = Counter()
    matches: defaultdict[str, list[str]] = defaultdict(list)
    for feature in _FEATURES:
        if feature.pattern.search(normalized) is not None:
            scores[feature.leaf] += feature.weight
            matches[feature.leaf].append(feature.rule_id)
    if not scores:
        return LEAF_SOFTWARE_ARCHITECTURE_DESIGN, ("architecture.fallback",), {}
    leaf = min(scores, key=lambda item: (-scores[item], _TIE_PRECEDENCE[item], item))
    return leaf, tuple(sorted(matches[leaf])), dict(sorted(scores.items()))


def _path_is_blocked(path: Path) -> bool:
    return any(_BLOCKED_PATH_COMPONENT.search(part) is not None for part in path.parts)


def _validate_input_pin(pin: InputPin) -> dict[str, Any]:
    path = pin.path.resolve()
    if _path_is_blocked(path):
        raise ValueError("refusing to inspect a cyber/security-named input path")
    if path.suffix.casefold() != ".jsonl" or not path.is_file():
        raise FileNotFoundError(f"pinned input JSONL does not exist: {path}")
    digest = pin.sha256.casefold()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError("input SHA-256 must be 64 hexadecimal characters")
    if type(pin.record_count) is not int or pin.record_count < 0:
        raise ValueError("input record_count must be a non-negative integer")
    if pin.partition not in {"prepared-train", "prepared-heldout"}:
        raise ValueError("input partition must be prepared-train or prepared-heldout")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"input SHA-256 mismatch for {path}: expected {digest}, got {actual}")
    return {
        "path": str(path),
        "partition": pin.partition,
        "bytes": path.stat().st_size,
        "expected_sha256": digest,
        "sha256": actual,
        "hash_verified": True,
        "expected_record_count": pin.record_count,
    }


def _validate_messages(value: object) -> tuple[ValidatedRecord | None, str | None]:
    if not isinstance(value, dict):
        return None, "not_json_object"

    # Metadata is checked before the messages member is accessed.  A guarded label never reaches
    # conversation validation, classification, tokenization, or an output shard.
    source = value.get("source")
    if not isinstance(source, str):
        return None, "invalid_source_metadata"
    if _BLOCKED_SOURCE_LABEL.search(source) is not None:
        return None, "guarded_source_metadata"
    if source != EXPECTED_SOURCE:
        return None, "unexpected_source_metadata"
    if set(value) != {"id", "source", "messages"}:
        return None, "unexpected_record_fields"
    source_id = value["id"]
    if not isinstance(source_id, str) or not source_id.strip():
        return None, "invalid_source_id"

    raw_messages = value["messages"]
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, "invalid_messages_container"
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            return None, "invalid_message_object"
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or role not in _VALID_ROLES:
            return None, "invalid_message_role"
        if not isinstance(content, str) or not content.strip():
            return None, "invalid_message_content"
        if any(marker in content for marker in _RESERVED_CHAT_MARKERS):
            return None, "reserved_chat_marker"
        messages.append({"role": role, "content": content})

    index = 0
    while index < len(messages) and messages[index]["role"] == "system":
        index += 1
    turns = messages[index:]
    if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return None, "invalid_conversation_boundaries"
    if any(message["role"] == "system" for message in turns):
        return None, "system_role_after_turn_start"

    roles = [message["role"] for message in turns]
    if "tool" in roles:
        mode = MODE_TOOL
    else:
        expected = "user"
        for role in roles:
            if role != expected:
                return None, "invalid_direct_role_order"
            expected = "assistant" if expected == "user" else "user"
        target = turns[-1]["content"].strip()
        has_open = "<think>" in target
        has_close = "</think>" in target
        if has_open or has_close:
            if not target.startswith("<think>") or target.count("<think>") != 1:
                return None, "malformed_trace_target"
            if target.count("</think>") != 1:
                return None, "malformed_trace_target"
            trace, final = target[len("<think>") :].split("</think>", 1)
            if not trace.strip() or not final.strip():
                return None, "empty_trace_or_final"
            mode = MODE_TRACE
        else:
            mode = MODE_DIRECT

    prompt = "\n\n".join(message["content"] for message in turns if message["role"] == "user")
    if not prompt.strip():
        return None, "empty_user_prompt"
    return ValidatedRecord(source_id, messages, prompt, mode), None


def _token_id_rows(value: object, *, expected: int) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError("batched chat template returned the wrong number of rows")
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("chat template returned an empty or non-list token row")
        if any(type(token_id) is not int or token_id < 0 for token_id in row):
            raise ValueError("chat template returned invalid token IDs")
        rows.append(list(row))
    return rows


def _batch_token_counts(
    tokenizer: ChatTemplateTokenizer,
    rows: Sequence[PendingRecord],
) -> tuple[list[int | None], list[str | None]]:
    counts: list[int | None] = [None] * len(rows)
    errors: list[str | None] = [None] * len(rows)
    by_mode: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_mode[row.mode].append(index)

    for mode, indices in sorted(by_mode.items()):
        conversations = [rows[index].messages for index in indices]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": False,
            "enable_thinking": mode == MODE_TRACE,
            "truncation": False,
            "padding": False,
            "return_tensors": None,
            "return_dict": False,
        }
        try:
            token_rows = _token_id_rows(
                tokenizer.apply_chat_template(conversations, **kwargs),
                expected=len(indices),
            )
        except (TypeError, ValueError, RuntimeError):
            # Isolate a bad conversation without turning a batch-level tokenizer exception into a
            # silent batch drop.
            for index in indices:
                try:
                    single = tokenizer.apply_chat_template(rows[index].messages, **kwargs)
                    if not isinstance(single, (list, tuple)) or not single:
                        raise ValueError("invalid single token row")
                    if any(type(token_id) is not int or token_id < 0 for token_id in single):
                        raise ValueError("invalid single token IDs")
                    counts[index] = len(single)
                except (TypeError, ValueError, RuntimeError):
                    errors[index] = "tokenizer_render_error"
        else:
            for index, token_ids in zip(indices, token_rows, strict=True):
                counts[index] = len(token_ids)
    return counts, errors


def _json_line(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _pairwise_overlap(values: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        "train_validation": len(values[SPLIT_TRAIN] & values[SPLIT_VALIDATION]),
        "train_test": len(values[SPLIT_TRAIN] & values[SPLIT_TEST]),
        "validation_test": len(values[SPLIT_VALIDATION] & values[SPLIT_TEST]),
    }


def _lineage_receipts(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).resolve()
        if _path_is_blocked(path):
            raise ValueError("refusing to inspect a cyber/security-named lineage path")
        if not path.is_file():
            raise FileNotFoundError(f"lineage artifact does not exist: {path}")
        receipts.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "superseded_lineage_alias",
                "ingested": False,
            }
        )
    return receipts


def _load_local_tokenizer(path: Path) -> ChatTemplateTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional CLI dependency
        raise RuntimeError("transformers is required to run this materializer") from exc
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def materialize_software_architecture_experts(
    *,
    train_pin: InputPin,
    heldout_pin: InputPin,
    output_dir: str | Path,
    tokenizer: ChatTemplateTokenizer,
    tokenizer_path: str | Path,
    expected_tokenizer_sha256: str,
    expected_chat_template_sha256: str,
    lineage_aliases: Sequence[str | Path] = (),
    seed: str = DEFAULT_SEED,
    batch_size: int = 512,
) -> Path:
    """Build the complete granular corpus and return its quality manifest."""

    if train_pin.partition != "prepared-train" or heldout_pin.partition != "prepared-heldout":
        raise ValueError("train_pin and heldout_pin partitions are fixed by contract")
    if not seed:
        raise ValueError("seed must be non-empty")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    input_receipts = [_validate_input_pin(train_pin), _validate_input_pin(heldout_pin)]
    tokenizer_receipt = tokenizer_identity(tokenizer_path)
    expected_tokenizer = expected_tokenizer_sha256.casefold()
    expected_template = expected_chat_template_sha256.casefold()
    if tokenizer_receipt["aggregate_sha256"] != expected_tokenizer:
        raise ValueError("tokenizer aggregate SHA-256 mismatch")
    if tokenizer_receipt["chat_template"]["sha256"] != expected_template:
        raise ValueError("chat-template SHA-256 mismatch")
    lineage_receipts = _lineage_receipts(lineage_aliases)

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    terminal_status: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    input_by_partition: Counter[str] = Counter()
    valid_by_leaf: Counter[str] = Counter()
    emitted_by_leaf: Counter[str] = Counter()
    emitted_by_split: Counter[str] = Counter()
    emitted_by_mode: Counter[str] = Counter()
    emitted_by_tier: Counter[str] = Counter()
    emitted_by_shard: Counter[tuple[str, str, str, str]] = Counter()
    classification_rule_counts: Counter[str] = Counter()
    rendered_tokens: Counter[tuple[str, str, str, str]] = Counter()
    min_tokens: dict[tuple[str, str, str, str], int] = {}
    max_tokens: dict[tuple[str, str, str, str], int] = {}
    prompt_sets = {split: set() for split in SPLITS}
    family_sets = {split: set() for split in SPLITS}
    train_repeated_prompts = 0
    train_repeated_families = 0
    heldout_repeated_prompts = 0
    heldout_repeated_families = 0
    heldout_seen_prompts: set[str] = set()
    heldout_seen_families: set[str] = set()
    streamed_counts: list[int] = []

    try:
        with ExitStack() as stack:
            data_writers: dict[tuple[str, str, str, str], Any] = {}
            provenance_writers: dict[tuple[str, str, str, str], Any] = {}
            for leaf in LEAVES:
                for mode in MODES:
                    for tier in _TIER_NAMES:
                        directory = staging / leaf / mode / tier
                        directory.mkdir(parents=True, exist_ok=True)
                        for split in SPLITS:
                            # Create the complete stable rack shape, but open writers lazily.  A
                            # fully populated rack has 630 data/sidecar files, which can exceed
                            # conservative Windows per-process stdio limits if all stay open.
                            (directory / f"{split}.jsonl").open(
                                "w", encoding="utf-8", newline="\n"
                            ).close()
                            (directory / f"{split}.provenance.jsonl").open(
                                "w", encoding="utf-8", newline="\n"
                            ).close()
            quarantine_writer = stack.enter_context(
                (staging / "quarantine.jsonl").open("w", encoding="utf-8", newline="\n")
            )

            pending: list[PendingRecord] = []

            def quarantine(
                *,
                input_index: int,
                partition: str,
                line_number: int,
                record_sha256: str,
                reason: str,
                leaf: str | None = None,
                prompt_group: str | None = None,
                family_group: str | None = None,
            ) -> None:
                terminal_status["quarantined"] += 1
                quarantine_reasons[reason] += 1
                receipt: dict[str, Any] = {
                    "input_index": input_index,
                    "input_partition": partition,
                    "line_number": line_number,
                    "reason": reason,
                    "record_sha256": record_sha256,
                }
                if leaf is not None:
                    receipt["leaf"] = leaf
                if prompt_group is not None:
                    receipt["prompt_group"] = prompt_group
                if family_group is not None:
                    receipt["family_group"] = family_group
                _json_line(quarantine_writer, receipt)

            def flush_pending() -> None:
                if not pending:
                    return
                counts, errors = _batch_token_counts(tokenizer, pending)
                for row, token_count, error in zip(pending, counts, errors, strict=True):
                    if error is not None or token_count is None:
                        quarantine(
                            input_index=row.input_index,
                            partition=row.input_partition,
                            line_number=row.line_number,
                            record_sha256=row.record_sha256,
                            reason=error or "tokenizer_render_error",
                            leaf=row.leaf,
                            prompt_group=row.prompt_group,
                            family_group=row.family_group,
                        )
                        continue
                    tier = tier_for_token_count(token_count)
                    key = (row.leaf, row.mode, tier, row.split)
                    if key not in data_writers:
                        directory = staging / row.leaf / row.mode / tier
                        data_writers[key] = stack.enter_context(
                            (directory / f"{row.split}.jsonl").open(
                                "a", encoding="utf-8", newline="\n"
                            )
                        )
                        provenance_writers[key] = stack.enter_context(
                            (directory / f"{row.split}.provenance.jsonl").open(
                                "a", encoding="utf-8", newline="\n"
                            )
                        )
                    _json_line(data_writers[key], {"messages": row.messages})
                    _json_line(
                        provenance_writers[key],
                        {
                            "classification_rules": list(row.classification_rules),
                            "family_group": row.family_group,
                            "input_index": row.input_index,
                            "input_line_number": row.line_number,
                            "input_partition": row.input_partition,
                            "leaf": row.leaf,
                            "mode": row.mode,
                            "prompt_group": row.prompt_group,
                            "record_sha256": row.record_sha256,
                            "rendered_tokens": token_count,
                            "source_id_sha256": row.source_id_sha256,
                            "split": row.split,
                            "tier": tier,
                        },
                    )
                    terminal_status["emitted"] += 1
                    emitted_by_leaf[row.leaf] += 1
                    emitted_by_split[row.split] += 1
                    emitted_by_mode[row.mode] += 1
                    emitted_by_tier[tier] += 1
                    emitted_by_shard[key] += 1
                    rendered_tokens[key] += token_count
                    min_tokens[key] = min(min_tokens.get(key, token_count), token_count)
                    max_tokens[key] = max(max_tokens.get(key, token_count), token_count)
                    prompt_sets[row.split].add(row.prompt_group)
                    family_sets[row.split].add(row.family_group)
                pending.clear()

            for input_index, pin in enumerate((train_pin, heldout_pin)):
                actual_count = 0
                streamed_digest = hashlib.sha256()
                with pin.path.resolve().open("rb") as source:
                    for line_number, raw_line in enumerate(source, start=1):
                        streamed_digest.update(raw_line)
                        actual_count += 1
                        input_by_partition[pin.partition] += 1
                        fingerprint = sha256_bytes(raw_line.rstrip(b"\r\n"))
                        try:
                            value = json.loads(raw_line.decode("utf-8"))
                        except UnicodeDecodeError:
                            quarantine(
                                input_index=input_index,
                                partition=pin.partition,
                                line_number=line_number,
                                record_sha256=fingerprint,
                                reason="invalid_utf8",
                            )
                            continue
                        except json.JSONDecodeError:
                            quarantine(
                                input_index=input_index,
                                partition=pin.partition,
                                line_number=line_number,
                                record_sha256=fingerprint,
                                reason="invalid_json",
                            )
                            continue

                        record, reason = _validate_messages(value)
                        if record is None:
                            quarantine(
                                input_index=input_index,
                                partition=pin.partition,
                                line_number=line_number,
                                record_sha256=fingerprint,
                                reason=reason or "internal_validation_error",
                            )
                            continue

                        prompt_group = prompt_group_id(record.prompt)
                        family_group = prompt_family_id(record.prompt)
                        leaf, rules, _ = classify_prompt(record.prompt)
                        valid_by_leaf[leaf] += 1
                        for rule in rules:
                            classification_rule_counts[rule] += 1

                        if pin.partition == "prepared-train":
                            split = SPLIT_TRAIN
                            train_repeated_prompts += prompt_group in prompt_sets[SPLIT_TRAIN]
                            train_repeated_families += family_group in family_sets[SPLIT_TRAIN]
                            # Populate the train identity index before tokenization so all held-out
                            # checks cover every technically valid prepared-training row.
                            prompt_sets[SPLIT_TRAIN].add(prompt_group)
                            family_sets[SPLIT_TRAIN].add(family_group)
                        else:
                            heldout_repeated_prompts += prompt_group in heldout_seen_prompts
                            heldout_repeated_families += family_group in heldout_seen_families
                            heldout_seen_prompts.add(prompt_group)
                            heldout_seen_families.add(family_group)
                            if prompt_group in prompt_sets[SPLIT_TRAIN]:
                                quarantine(
                                    input_index=input_index,
                                    partition=pin.partition,
                                    line_number=line_number,
                                    record_sha256=fingerprint,
                                    reason="heldout_train_exact_prompt_overlap",
                                    leaf=leaf,
                                    prompt_group=prompt_group,
                                    family_group=family_group,
                                )
                                continue
                            if family_group in family_sets[SPLIT_TRAIN]:
                                quarantine(
                                    input_index=input_index,
                                    partition=pin.partition,
                                    line_number=line_number,
                                    record_sha256=fingerprint,
                                    reason="heldout_train_prompt_family_overlap",
                                    leaf=leaf,
                                    prompt_group=prompt_group,
                                    family_group=family_group,
                                )
                                continue
                            split = heldout_split_for_family(family_group, seed=seed)

                        pending.append(
                            PendingRecord(
                                input_index=input_index,
                                input_partition=pin.partition,
                                line_number=line_number,
                                record_sha256=fingerprint,
                                source_id_sha256=sha256_bytes(record.source_id.encode("utf-8")),
                                messages=record.messages,
                                prompt_group=prompt_group,
                                family_group=family_group,
                                split=split,
                                leaf=leaf,
                                classification_rules=rules,
                                mode=record.mode,
                            )
                        )
                        if len(pending) >= batch_size:
                            flush_pending()
                flush_pending()
                streamed_counts.append(actual_count)
                if actual_count != pin.record_count:
                    raise ValueError(
                        f"record-count mismatch for {pin.partition}: expected "
                        f"{pin.record_count}, got {actual_count}"
                    )
                if streamed_digest.hexdigest() != pin.sha256.casefold():
                    raise ValueError(f"input changed while reading {pin.path.resolve()}")

        total_input = sum(streamed_counts)
        if total_input != terminal_status["emitted"] + terminal_status["quarantined"]:
            raise AssertionError("terminal accounting does not reconcile")
        prompt_overlap = _pairwise_overlap(prompt_sets)
        family_overlap = _pairwise_overlap(family_sets)
        if any(prompt_overlap.values()) or any(family_overlap.values()):
            raise AssertionError("prompt or prompt-family leakage crossed output splits")

        outputs: list[dict[str, Any]] = []
        for leaf in LEAVES:
            for mode in MODES:
                for tier in _TIER_NAMES:
                    for split in SPLITS:
                        key = (leaf, mode, tier, split)
                        relative = Path(leaf) / mode / tier / f"{split}.jsonl"
                        provenance_relative = relative.with_name(f"{split}.provenance.jsonl")
                        path = staging / relative
                        provenance = staging / provenance_relative
                        outputs.append(
                            {
                                "path": relative.as_posix(),
                                "provenance_path": provenance_relative.as_posix(),
                                "leaf": leaf,
                                "mode": mode,
                                "tier": tier,
                                "split": split,
                                "records": emitted_by_shard[key],
                                "rendered_tokens": rendered_tokens[key],
                                "min_rendered_tokens": min_tokens.get(key),
                                "max_rendered_tokens": max_tokens.get(key),
                                "bytes": path.stat().st_size,
                                "sha256": sha256_file(path),
                                "provenance_bytes": provenance.stat().st_size,
                                "provenance_sha256": sha256_file(provenance),
                            }
                        )
        if sum(receipt["records"] for receipt in outputs) != terminal_status["emitted"]:
            raise AssertionError("output shard receipt counts do not reconcile")

        quarantine_path = staging / "quarantine.jsonl"
        quarantine_count = sum(1 for _ in quarantine_path.open("rb"))
        if quarantine_count != terminal_status["quarantined"]:
            raise AssertionError("quarantine receipt count mismatch")

        for receipt, actual_count in zip(input_receipts, streamed_counts, strict=True):
            receipt["actual_record_count"] = actual_count
            receipt["record_count_verified"] = True
            receipt["stream_hash_verified"] = True

        builder_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "provenance": {
                "inputs": input_receipts,
                "tokenizer": {
                    **tokenizer_receipt,
                    "expected_aggregate_sha256": expected_tokenizer,
                    "expected_chat_template_sha256": expected_template,
                    "hashes_verified": True,
                },
                "builder": {"path": str(builder_path), "sha256": sha256_file(builder_path)},
                "lineage_aliases": lineage_receipts,
                "lineage_policy": (
                    "prior software-architecture samples are superseded aliases; hashes and "
                    "aggregate reports are retained for lineage, but rows are never ingested"
                ),
            },
            "contract": {
                "seed": seed,
                "input_scope": "only the two cryptographically pinned prepared software-architecture files",
                "blocked_path_policy": "refuse any input/lineage path component named cyber or security",
                "metadata_gate": "source label is checked before messages are accessed",
                "prepared_train_policy": "preserved in train; technically valid rows are not resplit",
                "prepared_heldout_policy": (
                    "train-overlapping prompt/family rows quarantined; remaining normalized "
                    "families split deterministically 50/50 into validation and test"
                ),
                "prompt_identity": "SHA-256(NFKC + casefold + collapsed whitespace)",
                "prompt_family": "SHA-256(first 256 characters of normalized prompt)",
                "classification": "weighted lexical rules v1; software_architecture_design fallback",
                "leaves": list(LEAVES),
                "modes": list(MODES),
                "token_tiers": [
                    {"name": name, "minimum": lower, "maximum": upper}
                    for name, lower, upper in TIER_SPECS
                ],
                "token_measurement": (
                    "complete Qwen chat template, truncation=False, padding=False; "
                    "enable_thinking only for trace mode"
                ),
                "training_schema": "exact messages-only object",
                "provenance_schema": "content-free row identity and routing metadata sidecar",
                "truncation": False,
                "license_metadata_consulted_for_admission": False,
                "usage": "private-only",
            },
            "accounting": {
                "input_records": total_input,
                "input_by_partition": dict(sorted(input_by_partition.items())),
                "terminal_status": dict(sorted(terminal_status.items())),
                "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
                "valid_classified_by_leaf": {leaf: valid_by_leaf[leaf] for leaf in LEAVES},
                "emitted_by_leaf": {leaf: emitted_by_leaf[leaf] for leaf in LEAVES},
                "emitted_by_split": {split: emitted_by_split[split] for split in SPLITS},
                "emitted_by_mode": {mode: emitted_by_mode[mode] for mode in MODES},
                "emitted_by_tier": {tier: emitted_by_tier[tier] for tier in _TIER_NAMES},
                "classification_rule_matches": dict(sorted(classification_rule_counts.items())),
                "repeated_prompt_rows": {
                    "prepared_train": train_repeated_prompts,
                    "prepared_heldout": heldout_repeated_prompts,
                },
                "repeated_prompt_family_rows": {
                    "prepared_train": train_repeated_families,
                    "prepared_heldout": heldout_repeated_families,
                },
                "fully_reconciled": True,
                "zero_silent_omissions": True,
            },
            "leakage": {
                "prompt_group_overlap": prompt_overlap,
                "prompt_family_overlap": family_overlap,
                "zero_prompt_overlap": not any(prompt_overlap.values()),
                "zero_prompt_family_overlap": not any(family_overlap.values()),
                "train_unique_prompt_groups": len(prompt_sets[SPLIT_TRAIN]),
                "train_unique_prompt_families": len(family_sets[SPLIT_TRAIN]),
                "heldout_unique_prompt_groups_seen": len(heldout_seen_prompts),
                "heldout_unique_prompt_families_seen": len(heldout_seen_families),
            },
            "quarantine": {
                "path": "quarantine.jsonl",
                "records": quarantine_count,
                "bytes": quarantine_path.stat().st_size,
                "sha256": sha256_file(quarantine_path),
                "content_free": True,
            },
            "outputs": outputs,
        }
        manifest_path = staging / "quality-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        sum_paths = [staging / receipt["path"] for receipt in outputs]
        sum_paths.extend(staging / receipt["provenance_path"] for receipt in outputs)
        sum_paths.extend((quarantine_path, manifest_path))
        sums_path = staging / "sha256sums.txt"
        with sums_path.open("w", encoding="ascii", newline="\n") as stream:
            for path in sorted(
                set(sum_paths), key=lambda item: item.relative_to(staging).as_posix()
            ):
                stream.write(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n")

        os.replace(staging, target)
        return target / "quality-manifest.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parse_sha256(value: str, *, name: str) -> str:
    digest = value.casefold()
    if _SHA256_RE.fullmatch(digest) is None:
        raise argparse.ArgumentTypeError(f"{name} must be a 64-character hexadecimal SHA-256")
    return digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--heldout-path", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--heldout-sha256", required=True)
    parser.add_argument("--train-records", type=int, required=True)
    parser.add_argument("--heldout-records", type=int, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lineage-alias", type=Path, action="append", default=[])
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = _load_local_tokenizer(args.tokenizer_path.resolve())
    manifest = materialize_software_architecture_experts(
        train_pin=InputPin(
            args.train_path.resolve(),
            _parse_sha256(args.train_sha256, name="train-sha256"),
            args.train_records,
            "prepared-train",
        ),
        heldout_pin=InputPin(
            args.heldout_path.resolve(),
            _parse_sha256(args.heldout_sha256, name="heldout-sha256"),
            args.heldout_records,
            "prepared-heldout",
        ),
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
        expected_tokenizer_sha256=_parse_sha256(args.tokenizer_sha256, name="tokenizer-sha256"),
        expected_chat_template_sha256=_parse_sha256(
            args.chat_template_sha256, name="chat-template-sha256"
        ),
        lineage_aliases=args.lineage_alias,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(json.dumps({"manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
