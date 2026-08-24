"""Build a deduplicated, balanced software-architecture SFT corpus.

The prepared source is treated as read-only.  Both prepared partitions are pooled,
grouped by a normalized/cleaned user prompt, and deterministically split again so a
prompt cannot leak between train and validation.  Trainer files contain exactly a
``messages`` field; selection details and source identity live in sidecars.

This builder deliberately repairs only mechanical defects (known UTF-8 mojibake,
control characters, and glued prose/list boundaries).  It does not rewrite answers
or ask another model to judge them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata
from typing import Any, Iterable, Iterator, Sequence


DATASET_NAME = "software-architecture-v1"
DEFAULT_SOURCE = Path.home() / "Documents" / "datasets" / "prepared" / "software-architecture"
SYSTEM = (
    "You are a pragmatic software architect. Give concrete, technically accurate advice. "
    "Explain tradeoffs, failure modes, operational constraints, and when an alternative "
    "design is preferable. Ask for missing requirements rather than inventing them."
)
LICENSE_NOTE = (
    "local prepared dataset; retain the upstream manifest and review rights before redistribution"
)

MIN_PROMPT_CHARS = 35
MAX_PROMPT_CHARS = 2200
MIN_ANSWER_CHARS = 300
MAX_ANSWER_CHARS = 7000
MIN_ANSWER_WORDS = 65
MAX_ANSWER_WORDS = 1000

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./_-]*")
SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SUSPICIOUS_ENCODING_RE = re.compile(r"(?:\ufffd|Ã|Â|â(?:€|„|†|‡|ˆ|‰|Š|‹|Œ|Ž|™|š|›|œ|ž|Ÿ))")

BLOCKED_ANSWER_RE = re.compile(
    r"(?i)(?:as an ai(?: language model)?|i (?:cannot|can't|won't) provide|"
    r"lorem ipsum|(?:insert|write|provide) (?:the |your )?(?:answer|response|text) here|"
    r"\[(?:insert|placeholder|todo)[^\]]*\]|<placeholder>|"
    r"this (?:response|answer) should|here is (?:a |the )?template)"
)
ROLE_LEAK_RE = re.compile(r"(?im)^\s*(?:system|user|assistant)\s*:\s+")
TRUNCATION_RE = re.compile(
    r"(?i)(?:\[(?:output|response|content)?\s*truncated\]|"
    r"\b(?:the )?(?:output|response|answer) (?:was|is) truncated\b)"
)

# This is intentionally finite and auditable.  The CP-1252 round trip below handles
# ordinary cases; these replacements cover mixed strings where that round trip cannot.
MOJIBAKE_REPLACEMENTS = {
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€�": "”",
    "â€\u009d": "”",
    "â€”": "—",
    "â€“": "–",
    "â€¦": "…",
    "â€¢": "•",
    "â†’": "→",
    "â†�": "←",
    "â‰¥": "≥",
    "â‰¤": "≤",
    "Â©": "©",
    "Â®": "®",
    "Â°": "°",
    "Â·": "·",
    "Â ": " ",
}

PROMPT_BOILERPLATE = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "could",
    "describe",
    "detail",
    "discuss",
    "elaborate",
    "explain",
    "for",
    "give",
    "how",
    "in",
    "include",
    "including",
    "is",
    "it",
    "me",
    "of",
    "on",
    "outline",
    "please",
    "provide",
    "role",
    "scenario",
    "software",
    "system",
    "systems",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
}

GENERIC_ANSWER_PHRASES = (
    "plays a crucial role",
    "is very important",
    "in today's rapidly evolving",
    "in the ever-evolving landscape",
    "it is essential to understand",
    "in conclusion",
    "in summary",
    "ultimately",
)

QUALITY_SIGNALS = {
    "tradeoffs": re.compile(r"(?i)\b(?:trade-?offs?|downside|advantage|disadvantage|cost)\b"),
    "failure_modes": re.compile(
        r"(?i)\b(?:failure|fault|outage|degrad|timeout|retry|cascade|bottleneck|risk)\w*\b"
    ),
    "operations": re.compile(
        r"(?i)\b(?:deploy|monitor|observab|metric|logging|trace|operate|on-call|rollback)\w*\b"
    ),
    "constraints": re.compile(
        r"(?i)\b(?:constraint|latency|throughput|budget|compliance|consistency|availability|"
        r"capacity|scale)\w*\b"
    ),
    "alternatives": re.compile(
        r"(?i)\b(?:alternative|instead|prefer|appropriate when|better fit|consider)\b"
    ),
    "concreteness": re.compile(
        r"(?i)\b(?:for example|for instance|e\.g\.|measure|threshold|step|first|second)\b"
    ),
}

# A row belongs to exactly one family.  Scores, rather than first-match wins, keep a
# prompt about (for example) testing microservices in the most strongly represented
# family.  Tuple order is the deterministic tie-break order.
TOPIC_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "security-and-privacy",
        (
            re.compile(r"(?i)\b(?:security|secure|threat\w*|vulnerab\w*|attack\w*|authentication|authorization)\b"),
            re.compile(r"(?i)\b(?:encrypt\w*|privacy|zero trust|least privilege|injection|xss|csrf)\b"),
        ),
    ),
    (
        "reliability-and-resilience",
        (
            re.compile(r"(?i)\b(?:reliab\w*|resilien\w*|fault toler\w*|high availability|disaster recovery)\b"),
            re.compile(r"(?i)\b(?:failover|redundan\w*|circuit breaker|bulkhead|recovery|chaos engineering)\b"),
        ),
    ),
    (
        "data-and-consistency",
        (
            re.compile(r"(?i)\b(?:database|data store|persistence|sql|nosql|schema|shard\w*|replica\w*)\b"),
            re.compile(r"(?i)\b(?:consistency|transaction\w*|acid|cap theorem|eventual consistency|cache invalidation)\b"),
        ),
    ),
    (
        "distributed-systems",
        (
            re.compile(r"(?i)\b(?:distributed system\w*|distributed computing|consensus|leader election)\b"),
            re.compile(r"(?i)\b(?:network partition|byzantine|lamport|vector clock|distributed transaction)\b"),
        ),
    ),
    (
        "services-and-apis",
        (
            re.compile(r"(?i)\b(?:microservice\w*|service-oriented|soa)\b"),
            re.compile(r"(?i)\b(?:api gateway|rest(?:ful)?|graphql|grpc|service mesh|service discovery)\b"),
        ),
    ),
    (
        "messaging-and-events",
        (
            re.compile(r"(?i)\b(?:event[- ]driven|event sourcing|cqrs|message (?:queue|broker)|pub(?:lish)?[/-]?sub)\b"),
            re.compile(r"(?i)\b(?:kafka|rabbitmq|asynchronous messag\w*|stream processing|saga pattern)\b"),
        ),
    ),
    (
        "performance-and-scalability",
        (
            re.compile(r"(?i)\b(?:scalab\w*|performance|latency|throughput|load balanc\w*|bottleneck\w*)\b"),
            re.compile(r"(?i)\b(?:cach(?:e|ing)|horizontal scal\w*|vertical scal\w*|capacity planning|profiling)\b"),
        ),
    ),
    (
        "cloud-and-serverless",
        (
            re.compile(r"(?i)\b(?:cloud|serverless|function as a service|faas|aws|azure|gcp)\b"),
            re.compile(r"(?i)\b(?:container\w*|kubernetes|docker|orchestrat\w*|infrastructure as code)\b"),
        ),
    ),
    (
        "testing-and-quality",
        (
            re.compile(r"(?i)\b(?:tests?|testing|testability|quality assurance|verification|validation)\b"),
            re.compile(r"(?i)\b(?:unit test|integration test|contract test|end-to-end|test pyramid)\b"),
        ),
    ),
    (
        "operations-and-observability",
        (
            re.compile(r"(?i)\b(?:devops|sre|observab\w*|monitor\w*|logging|tracing|telemetry)\b"),
            re.compile(r"(?i)\b(?:deployment\w*|continuous delivery|ci/cd|release\w*|rollback|incident\w*)\b"),
        ),
    ),
    (
        "patterns-and-modularity",
        (
            re.compile(r"(?i)\b(?:architectural pattern|design pattern|mvc|mvvm|layered architecture)\b"),
            re.compile(r"(?i)\b(?:hexagonal|clean architecture|domain[- ]driven|ddd|modular\w*|coupling|cohesion)\b"),
        ),
    ),
    (
        "modernization-and-integration",
        (
            re.compile(r"(?i)\b(?:legacy|moderniz\w*|migrat\w*|refactor\w*|strangler|brownfield)\b"),
            re.compile(r"(?i)\b(?:integration|interoperab\w*|third-party|enterprise service bus|middleware)\b"),
        ),
    ),
    (
        "client-and-presentation",
        (
            re.compile(r"(?i)\b(?:frontend|front-end|mobile|web application|user interface|presentation layer)\b"),
            re.compile(r"(?i)\b(?:single-page application|spa|progressive web|client-server|responsive)\b"),
        ),
    ),
    (
        "process-and-governance",
        (
            re.compile(r"(?i)\b(?:architecture decision|adr|governance|compliance|standardization)\b"),
            re.compile(r"(?i)\b(?:requirement\w*|stakeholder\w*|documentation|review\w*|roadmap|technical debt)\b"),
        ),
    ),
)
FALLBACK_FAMILY = "general-architecture"


@dataclass(frozen=True, slots=True)
class Candidate:
    prompt_hash: str
    prompt: str
    answer: str
    source_id: str
    source_partition: str
    source_line: int
    family: str
    quality_score: int
    answer_words: int
    repairs: tuple[str, ...]
    removed_prompt_paragraphs: int
    original_prompt_hash: str
    candidate_hash: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: str, purpose: str, value: str) -> str:
    return sha256_text(f"{seed}\0{purpose}\0{value}")


def _suspicious_count(value: str) -> int:
    return len(SUSPICIOUS_ENCODING_RE.findall(value))


def repair_text(value: str, *, answer: bool) -> tuple[str, tuple[str, ...]]:
    """Repair known byte-decoding and mechanical layout defects, never semantics."""

    repairs: list[str] = []
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff")

    before = value
    if _suspicious_count(value):
        try:
            round_tripped = value.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            round_tripped = value
        if _suspicious_count(round_tripped) < _suspicious_count(value):
            value = round_tripped
        for broken, fixed in MOJIBAKE_REPLACEMENTS.items():
            value = value.replace(broken, fixed)
        if value != before:
            repairs.append("utf8_mojibake")

    replaced_controls = CONTROL_RE.sub(" ", value)
    if replaced_controls != value:
        value = replaced_controls
        repairs.append("control_characters")
    if "\u00a0" in value:
        value = value.replace("\u00a0", " ")
        repairs.append("nonbreaking_spaces")

    # Preserve paragraphs and code-ish indentation but trim noisy line endings.
    value = "\n".join(line.rstrip() for line in value.splitlines())
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value).strip()

    if answer:
        formatted = re.sub(r"(?<=[.!?])(?=[A-Z](?:[a-z]|[ \t]))", " ", value)
        formatted = re.sub(r"(?<=[.!?:])(?=#{1,4}\s)", "\n\n", formatted)
        formatted = re.sub(r"(?<=[a-z)*]):(?=[A-Z][a-z])", ": ", formatted)
        formatted = re.sub(
            r"(?<=[a-z0-9).:])(?=(?:\d{1,2}\.\s+\*\*|[-*]\s+\*\*))",
            "\n",
            formatted,
        )
        formatted = re.sub(r"(?<=[a-z0-9).])(?=\*\*[^*\n]{2,80}\*\*\s*:)", "\n", formatted)
        if formatted != value:
            value = formatted
            repairs.append("glued_prose_or_list")

    return unicodedata.normalize("NFC", value), tuple(sorted(set(repairs)))


def content_terms(value: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD_RE.findall(value)
        if len(token) > 1 and token.casefold() not in PROMPT_BOILERPLATE
    }


def _redundant_prompt_pair(first: str, second: str) -> bool:
    first_key = canonical_prompt(first)
    second_key = canonical_prompt(second)
    if first_key == second_key or first_key in second_key or second_key in first_key:
        return True
    first_terms = content_terms(first)
    second_terms = content_terms(second)
    if len(first_terms) < 3 or len(second_terms) < 3:
        return False
    intersection = len(first_terms & second_terms)
    containment = intersection / min(len(first_terms), len(second_terms))
    jaccard = intersection / len(first_terms | second_terms)
    # This prepared corpus represents a single synthetic task twice: an imperative
    # instruction followed by its question-form paraphrase. Require shared domain
    # content before collapsing them; different subjects remain intact.
    return intersection >= 2 and (containment >= 0.30 or jaccard >= 0.24)


def clean_prompt(value: str) -> tuple[str, int]:
    """Remove only clearly redundant blank-line-separated question paraphrases."""

    paragraphs = [SPACE_RE.sub(" ", part).strip() for part in re.split(r"\n\s*\n", value)]
    paragraphs = [part for part in paragraphs if part]
    templated_instruction_pair = (
        len(paragraphs) == 2
        and paragraphs[1].endswith("?")
        and len(paragraphs[0]) <= 500
        and len(paragraphs[1]) <= 500
        and bool(
            re.match(
                r"(?i)^(?:analy[sz]e|compare|create|describe|design|discuss|examine|"
                r"explain|explore|outline|provide|use case|what|how|why|can|could)\b",
                paragraphs[0],
            )
        )
    )
    if len(paragraphs) != 2 or not (
        templated_instruction_pair or _redundant_prompt_pair(*paragraphs)
    ):
        return "\n\n".join(paragraphs), 0

    first_terms = content_terms(paragraphs[0])
    second_terms = content_terms(paragraphs[1])
    # Prefer the standalone phrasing carrying more non-boilerplate requirements. A
    # length tie-break rewards concrete constraints such as "24/7" or "under load".
    first_score = len(first_terms) * 100 + min(len(paragraphs[0]), 500)
    second_score = len(second_terms) * 100 + min(len(paragraphs[1]), 500)
    keep = paragraphs[1] if second_score > first_score else paragraphs[0]
    return keep, 1


def canonical_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w+#./-]+", " ", value)
    return SPACE_RE.sub(" ", value).strip()


def topic_family(prompt: str) -> str:
    scored: list[tuple[int, int, str]] = []
    for position, (family, patterns) in enumerate(TOPIC_PATTERNS):
        score = sum(len(pattern.findall(prompt)) for pattern in patterns)
        scored.append((score, -position, family))
    score, _, family = max(scored)
    return family if score else FALLBACK_FAMILY


def _sentence_keys(value: str) -> list[str]:
    pieces = re.split(r"(?:[.!?](?:\s+|$)|\n+)", value)
    return [canonical_prompt(piece) for piece in pieces if len(content_terms(piece)) >= 5]


def answer_quality(prompt: str, answer: str) -> tuple[int | None, str | None, int]:
    if not MIN_ANSWER_CHARS <= len(answer) <= MAX_ANSWER_CHARS:
        return None, "answer_chars", 0
    words = WORD_RE.findall(answer)
    word_count = len(words)
    if not MIN_ANSWER_WORDS <= word_count <= MAX_ANSWER_WORDS:
        return None, "answer_words", word_count
    if BLOCKED_ANSWER_RE.search(answer):
        return None, "blocked_or_template_answer", word_count
    if ROLE_LEAK_RE.search(answer):
        return None, "role_marker_leak", word_count
    if TRUNCATION_RE.search(answer):
        return None, "truncated_answer", word_count
    if _suspicious_count(answer):
        return None, "unrepaired_mojibake", word_count

    sentence_keys = _sentence_keys(answer)
    repeated_sentences = len(sentence_keys) - len(set(sentence_keys))
    if repeated_sentences >= 2 or (
        sentence_keys and repeated_sentences / len(sentence_keys) > 0.20
    ):
        return None, "repetitive_answer", word_count

    unique_ratio = len({word.casefold() for word in words}) / word_count
    if unique_ratio < 0.24:
        return None, "low_lexical_diversity", word_count

    prompt_terms = content_terms(prompt)
    answer_terms = content_terms(answer)
    prompt_coverage = len(prompt_terms & answer_terms) / max(1, len(prompt_terms))
    if prompt_coverage < 0.18:
        return None, "weak_prompt_relevance", word_count

    # A broad but auditable proxy: enough depth to be useful, prompt coverage,
    # architecture tradeoff signals, and readable structure.  It is used only to
    # choose among rows sharing the same normalized prompt.
    depth = min(word_count, 420) - max(0, word_count - 650) // 2
    signal_count = sum(bool(pattern.search(answer)) for pattern in QUALITY_SIGNALS.values())
    structure_count = min(
        10,
        len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)]|#{1,4}\s|\*\*[^*]+\*\*)", answer)),
    )
    generic_count = sum(answer.casefold().count(phrase) for phrase in GENERIC_ANSWER_PHRASES)
    score = (
        depth * 3
        + round(prompt_coverage * 240)
        + signal_count * 45
        + structure_count * 12
        + round(unique_ratio * 100)
        - generic_count * 25
        - repeated_sentences * 80
    )
    return score, None, word_count


def clean_row(
    row: dict[str, Any], *, partition: str, line_number: int
) -> tuple[Candidate | None, str | None]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, "message_schema"
    non_system = [message for message in messages if isinstance(message, dict) and message.get("role") != "system"]
    if (
        len(non_system) != 2
        or non_system[0].get("role") != "user"
        or non_system[1].get("role") != "assistant"
        or not isinstance(non_system[0].get("content"), str)
        or not isinstance(non_system[1].get("content"), str)
    ):
        return None, "message_schema"

    original_prompt = non_system[0]["content"]
    prompt, prompt_repairs = repair_text(original_prompt, answer=False)
    answer, answer_repairs = repair_text(non_system[1]["content"], answer=True)
    if _suspicious_count(prompt):
        return None, "unrepaired_prompt_mojibake"
    prompt, removed_paragraphs = clean_prompt(prompt)
    if not MIN_PROMPT_CHARS <= len(prompt) <= MAX_PROMPT_CHARS:
        return None, "prompt_chars"
    normalized = canonical_prompt(prompt)
    if len(content_terms(normalized)) < 3:
        return None, "weak_prompt"

    quality_score, reason, answer_words = answer_quality(prompt, answer)
    if quality_score is None:
        return None, reason

    prompt_hash = sha256_text(normalized)
    source_id = str(row.get("id") or f"{partition}:{line_number}")
    repairs = tuple(sorted(set(prompt_repairs + answer_repairs)))
    return (
        Candidate(
            prompt_hash=prompt_hash,
            prompt=prompt,
            answer=answer,
            source_id=source_id,
            source_partition=partition,
            source_line=line_number,
            family=topic_family(prompt),
            quality_score=quality_score,
            answer_words=answer_words,
            repairs=repairs,
            removed_prompt_paragraphs=removed_paragraphs,
            original_prompt_hash=sha256_text(original_prompt),
            candidate_hash=sha256_text(f"{prompt}\0{answer}\0{source_id}"),
        ),
        None,
    )


def source_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE candidates (
            prompt_hash TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            answer TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_partition TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            family TEXT NOT NULL,
            quality_score INTEGER NOT NULL,
            answer_words INTEGER NOT NULL,
            repairs TEXT NOT NULL,
            removed_prompt_paragraphs INTEGER NOT NULL,
            original_prompt_hash TEXT NOT NULL,
            candidate_hash TEXT NOT NULL,
            split_rank TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX candidates_family_rank ON candidates(family, split_rank);
        """
    )


UPSERT = """
INSERT INTO candidates (
    prompt_hash, prompt, answer, source_id, source_partition, source_line,
    family, quality_score, answer_words, repairs, removed_prompt_paragraphs,
    original_prompt_hash, candidate_hash, split_rank
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(prompt_hash) DO UPDATE SET
    prompt=excluded.prompt,
    answer=excluded.answer,
    source_id=excluded.source_id,
    source_partition=excluded.source_partition,
    source_line=excluded.source_line,
    family=excluded.family,
    quality_score=excluded.quality_score,
    answer_words=excluded.answer_words,
    repairs=excluded.repairs,
    removed_prompt_paragraphs=excluded.removed_prompt_paragraphs,
    original_prompt_hash=excluded.original_prompt_hash,
    candidate_hash=excluded.candidate_hash,
    split_rank=excluded.split_rank
WHERE excluded.quality_score > candidates.quality_score
   OR (excluded.quality_score = candidates.quality_score
       AND excluded.candidate_hash < candidates.candidate_hash)
"""


def candidate_values(candidate: Candidate, seed: str) -> tuple[Any, ...]:
    return (
        candidate.prompt_hash,
        candidate.prompt,
        candidate.answer,
        candidate.source_id,
        candidate.source_partition,
        candidate.source_line,
        candidate.family,
        candidate.quality_score,
        candidate.answer_words,
        json.dumps(candidate.repairs, separators=(",", ":")),
        candidate.removed_prompt_paragraphs,
        candidate.original_prompt_hash,
        candidate.candidate_hash,
        stable_rank(seed, "split", candidate.prompt_hash),
    )


def collect_candidates(
    connection: sqlite3.Connection,
    paths: Sequence[tuple[str, Path]],
    *,
    seed: str,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows_by_prepared_partition": Counter(),
        "accepted_candidate_rows": 0,
        "rejected": Counter(),
        "repairs": Counter(),
        "redundant_prompt_paragraphs_removed": 0,
    }
    pending: list[tuple[Any, ...]] = []
    for partition, path in paths:
        for line_number, row in source_rows(path):
            stats["rows_by_prepared_partition"][partition] += 1
            candidate, reason = clean_row(row, partition=partition, line_number=line_number)
            if candidate is None:
                stats["rejected"][str(reason)] += 1
                continue
            stats["accepted_candidate_rows"] += 1
            stats["repairs"].update(candidate.repairs)
            stats["redundant_prompt_paragraphs_removed"] += candidate.removed_prompt_paragraphs
            pending.append(candidate_values(candidate, seed))
            if len(pending) >= 2000:
                connection.executemany(UPSERT, pending)
                connection.commit()
                pending.clear()
    if pending:
        connection.executemany(UPSERT, pending)
        connection.commit()

    unique = connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
    stats["unique_normalized_prompts"] = unique
    stats["duplicate_eligible_rows"] = stats["accepted_candidate_rows"] - unique
    stats["rows_by_prepared_partition"] = dict(
        sorted(stats["rows_by_prepared_partition"].items())
    )
    stats["rejected"] = dict(sorted(stats["rejected"].items()))
    stats["repairs"] = dict(sorted(stats["repairs"].items()))
    return stats


def allocate_balanced(capacities: dict[str, int], total: int) -> dict[str, int]:
    if sum(capacities.values()) < total:
        raise RuntimeError(
            f"only {sum(capacities.values())} unique prompts survived; requested {total}"
        )
    quotas = {family: 0 for family in sorted(capacities)}
    remaining = total
    while remaining:
        eligible = [family for family in sorted(capacities) if quotas[family] < capacities[family]]
        if not eligible:
            raise RuntimeError("unable to allocate requested balanced sample")
        for family in eligible:
            if not remaining:
                break
            quotas[family] += 1
            remaining -= 1
    return quotas


def allocate_validation(total_quotas: dict[str, int], validation_size: int) -> dict[str, int]:
    selected_total = sum(total_quotas.values())
    raw = {
        family: count * validation_size / selected_total
        for family, count in total_quotas.items()
    }
    quotas = {family: math.floor(value) for family, value in raw.items()}
    remaining = validation_size - sum(quotas.values())
    order = sorted(raw, key=lambda family: (-(raw[family] - quotas[family]), family))
    for family in order[:remaining]:
        quotas[family] += 1
    return quotas


def row_to_candidate(row: Sequence[Any]) -> Candidate:
    return Candidate(
        prompt_hash=row[0],
        prompt=row[1],
        answer=row[2],
        source_id=row[3],
        source_partition=row[4],
        source_line=row[5],
        family=row[6],
        quality_score=row[7],
        answer_words=row[8],
        repairs=tuple(json.loads(row[9])),
        removed_prompt_paragraphs=row[10],
        original_prompt_hash=row[11],
        candidate_hash=row[12],
    )


SELECT_COLUMNS = """
prompt_hash, prompt, answer, source_id, source_partition, source_line,
family, quality_score, answer_words, repairs, removed_prompt_paragraphs,
original_prompt_hash, candidate_hash
"""


def select_balanced(
    connection: sqlite3.Connection,
    *,
    train_size: int,
    validation_size: int,
) -> tuple[list[Candidate], list[Candidate], dict[str, int], dict[str, int]]:
    capacities = {
        family: count
        for family, count in connection.execute(
            "SELECT family, count(*) FROM candidates GROUP BY family ORDER BY family"
        )
    }
    total_quotas = allocate_balanced(capacities, train_size + validation_size)
    validation_quotas = allocate_validation(total_quotas, validation_size)
    train: list[Candidate] = []
    validation: list[Candidate] = []
    for family in sorted(total_quotas):
        rows = connection.execute(
            f"SELECT {SELECT_COLUMNS} FROM candidates "
            "WHERE family = ? ORDER BY split_rank, prompt_hash LIMIT ?",
            (family, total_quotas[family]),
        ).fetchall()
        candidates = [row_to_candidate(row) for row in rows]
        validation_count = validation_quotas[family]
        validation.extend(candidates[:validation_count])
        train.extend(candidates[validation_count:])
    if len(train) != train_size or len(validation) != validation_size:
        raise RuntimeError("internal split allocation error")
    if {item.prompt_hash for item in train} & {item.prompt_hash for item in validation}:
        raise RuntimeError("normalized prompt leakage detected")
    return train, validation, capacities, total_quotas


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count, sha256_file(path)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def emit_split(
    candidates: list[Candidate],
    *,
    suffix: str,
    output_dir: Path,
    seed: str,
) -> dict[str, Any]:
    data_path = output_dir / f"{DATASET_NAME}{suffix}.jsonl"
    provenance_path = output_dir / f"{DATASET_NAME}{suffix}.provenance.jsonl"
    ordered = sorted(
        candidates,
        key=lambda candidate: stable_rank(seed, f"output{suffix}", candidate.prompt_hash),
    )
    data_rows = (
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": item.prompt},
                {"role": "assistant", "content": item.answer},
            ]
        }
        for item in ordered
    )
    provenance_rows = (
        {
            "prompt_sha256": item.prompt_hash,
            "original_prompt_sha256": item.original_prompt_hash,
            "source": "software-architecture",
            "source_id": item.source_id,
            "source_partition": item.source_partition,
            "source_line": item.source_line,
            "license": LICENSE_NOTE,
            "topic_family": item.family,
            "quality_score": item.quality_score,
            "answer_words": item.answer_words,
            "repairs": list(item.repairs),
            "redundant_prompt_paragraphs_removed": item.removed_prompt_paragraphs,
            "selection": "strongest accepted response per normalized cleaned prompt; balanced family sample",
        }
        for item in ordered
    )
    row_count, digest = write_jsonl_atomic(data_path, data_rows)
    provenance_count, provenance_digest = write_jsonl_atomic(provenance_path, provenance_rows)
    if row_count != provenance_count:
        raise RuntimeError(f"provenance mismatch for {data_path}")
    lengths = [item.answer_words for item in ordered]
    return {
        "path": str(data_path),
        "provenance_path": str(provenance_path),
        "rows": row_count,
        "sha256": digest,
        "provenance_sha256": provenance_digest,
        "answer_words": {
            "min": min(lengths, default=0),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "topic_families": dict(sorted(Counter(item.family for item in ordered).items())),
        "selected_source_partitions": dict(
            sorted(Counter(item.source_partition for item in ordered).items())
        ),
        "selected_repairs": dict(
            sorted(Counter(repair for item in ordered for repair in item.repairs).items())
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", default=DATASET_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_size < 1 or args.validation_size < 1:
        raise ValueError("train-size and validation-size must both be positive")
    paths = [
        ("prepared-train", args.source_dir / "train.jsonl"),
        ("prepared-validation", args.source_dir / "validation.jsonl"),
    ]
    for _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    snapshot = {
        str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for _, path in paths
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="molor-architecture-build-") as temporary_dir:
        database_path = Path(temporary_dir) / "candidates.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            initialize_database(connection)
            audit = collect_candidates(connection, paths, seed=args.seed)
            train, validation, capacities, selected_quotas = select_balanced(
                connection,
                train_size=args.train_size,
                validation_size=args.validation_size,
            )
        finally:
            connection.close()

    outputs = {
        "train": emit_split(
            train, suffix="", output_dir=args.output_dir, seed=args.seed
        ),
        "validation": emit_split(
            validation, suffix=".validation", output_dir=args.output_dir, seed=args.seed
        ),
    }
    train_prompts = {item.prompt_hash for item in train}
    validation_prompts = {item.prompt_hash for item in validation}
    report = {
        "dataset": DATASET_NAME,
        "parameters": {
            "seed": args.seed,
            "train_size": args.train_size,
            "validation_size": args.validation_size,
            "prompt_identity": "sha256(NFKC casefolded cleaned prompt with punctuation/whitespace normalized)",
            "response_selection": "highest deterministic quality score per normalized prompt; candidate hash tie-break",
            "split_method": "pool prepared partitions, group prompts, balance topic families, deterministic hash resplit",
            "trainer_row_schema": {"messages": "only field"},
            "quality_bounds": {
                "prompt_chars": [MIN_PROMPT_CHARS, MAX_PROMPT_CHARS],
                "answer_chars": [MIN_ANSWER_CHARS, MAX_ANSWER_CHARS],
                "answer_words": [MIN_ANSWER_WORDS, MAX_ANSWER_WORDS],
            },
        },
        "audit": audit,
        "eligible_topic_families": dict(sorted(capacities.items())),
        "selected_total_topic_quotas": dict(sorted(selected_quotas.items())),
        "leakage": {
            "train_unique_normalized_prompts": len(train_prompts),
            "validation_unique_normalized_prompts": len(validation_prompts),
            "train_validation_normalized_prompt_overlap": len(
                train_prompts & validation_prompts
            ),
            "grouped_across_prepared_partitions_before_resplit": True,
        },
        "outputs": outputs,
        "source_snapshot": snapshot,
        "source_files_unchanged": all(
            path.stat().st_size == snapshot[str(path)]["bytes"]
            and path.stat().st_mtime_ns == snapshot[str(path)]["mtime_ns"]
            for _, path in paths
        ),
    }
    report_path = args.output_dir / f"{DATASET_NAME}.report.json"
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "unique_normalized_prompts": audit["unique_normalized_prompts"],
                "train_rows": outputs["train"]["rows"],
                "validation_rows": outputs["validation"]["rows"],
                "prompt_overlap": report["leakage"][
                    "train_validation_normalized_prompt_overlap"
                ],
                "source_files_unchanged": report["source_files_unchanged"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
