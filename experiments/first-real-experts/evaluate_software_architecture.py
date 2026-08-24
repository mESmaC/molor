"""CPU-only paired evaluation for pre-generated software-architecture answers.

This script does not load a model.  It aligns base and expert outputs with the frozen
``software-architecture-v1`` validation set, reports mechanically checkable lexical
and structural diagnostics, and can create a blinded A/B review pack.

Accepted output JSONL rows are either a JSON string, an object containing one of
``output``, ``generated``, ``response``, or ``text``, or a trainer-style ``messages``
object whose final message is an assistant answer.  Optional ``index`` and
``prompt_sha256`` fields are validated when present.

Alternatively, ``--completion-report`` directly accepts the combined JSON written
by ``evaluate_completion.py`` and reads
``routes.base.examples[*].generated`` / ``routes.expert.examples[*].generated``.
Every report index, prompt message, reference, and available hash is checked against
the frozen validation rows before scoring.

None of the overlap metrics establish factual or architectural correctness.  The
reference corpus is synthetic prose, so a valid paraphrase may score poorly and a
confidently wrong imitation may score well.  Use the blinded review pack for the
actual quality decision.
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
import random
import re
import statistics
import unicodedata
from typing import Any, Iterable, Sequence


DATASET_NAME = "software-architecture-v1"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VALIDATION = SCRIPT_DIR / "data" / f"{DATASET_NAME}.validation.jsonl"
DEFAULT_PROVENANCE = (
    SCRIPT_DIR / "data" / f"{DATASET_NAME}.validation.provenance.jsonl"
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./_-]*")
SENTENCE_RE = re.compile(r"(?:[.!?](?:\s+|$)|\n+)")
THINK_RE = re.compile(r"(?is)^\s*<think>")
REFUSAL_RE = re.compile(
    r"(?i)\b(?:as an ai(?: language model)?|i (?:cannot|can't|won't)|"
    r"unable to (?:answer|provide|help)|i do not have enough information)\b"
)

STOPWORDS = {
    "a",
    "about",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "other",
    "should",
    "so",
    "such",
    "than",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "through",
    "to",
    "use",
    "used",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "within",
    "would",
    "you",
}

# Canonical labels with exact lexical aliases.  This makes the "concept" metric
# inspectable and reproducible; it is not an embedding or a semantic judge.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "abac": ("abac", "attribute based access control"),
    "acid": ("acid",),
    "adr": ("adr", "architecture decision record"),
    "api-gateway": ("api gateway",),
    "authentication": ("authentication",),
    "authorization": ("authorization",),
    "auto-scaling": ("auto scaling", "autoscaling"),
    "availability": ("availability", "high availability"),
    "backpressure": ("backpressure", "back pressure"),
    "blue-green-deployment": ("blue green deployment",),
    "bulkhead": ("bulkhead",),
    "cache-invalidation": ("cache invalidation",),
    "canary-deployment": ("canary deployment",),
    "cap-theorem": ("cap theorem",),
    "chaos-engineering": ("chaos engineering",),
    "circuit-breaker": ("circuit breaker",),
    "ci-cd": ("ci cd", "continuous integration continuous delivery"),
    "cloud": ("cloud", "cloud native"),
    "cohesion": ("cohesion",),
    "consistency": ("consistency", "consistent"),
    "containers": ("container", "containers", "containerization"),
    "cqrs": ("cqrs",),
    "cryptography": ("cryptography", "cryptographic"),
    "database-sharding": ("database sharding", "sharding"),
    "defense-in-depth": ("defense in depth", "defence in depth"),
    "disaster-recovery": ("disaster recovery",),
    "distributed-system": ("distributed system", "distributed systems"),
    "distributed-tracing": ("distributed tracing",),
    "domain-driven-design": ("domain driven design", "ddd"),
    "encryption": ("encryption", "encrypted"),
    "event-driven-architecture": ("event driven architecture",),
    "event-sourcing": ("event sourcing",),
    "eventual-consistency": ("eventual consistency",),
    "failover": ("failover", "fail over"),
    "fault-tolerance": ("fault tolerance", "fault tolerant"),
    "graceful-degradation": ("graceful degradation",),
    "graphql": ("graphql",),
    "grpc": ("grpc",),
    "hexagonal-architecture": ("hexagonal architecture",),
    "horizontal-scaling": ("horizontal scaling", "scale out"),
    "idempotency": ("idempotency", "idempotent"),
    "infrastructure-as-code": ("infrastructure as code", "iac"),
    "integration-testing": ("integration testing", "integration test"),
    "kubernetes": ("kubernetes",),
    "latency": ("latency",),
    "least-privilege": ("least privilege",),
    "load-balancing": ("load balancing", "load balancer"),
    "logging": ("logging", "logs"),
    "message-broker": ("message broker",),
    "message-queue": ("message queue", "message queues"),
    "microservices": ("microservice", "microservices"),
    "modularity": ("modularity", "modular"),
    "monolith": ("monolith", "monolithic"),
    "monitoring": ("monitoring", "monitor"),
    "mtls": ("mtls", "mutual tls"),
    "oauth": ("oauth", "oauth 2.0"),
    "observability": ("observability",),
    "openid-connect": ("openid connect", "oidc"),
    "orchestration": ("orchestration", "orchestrator"),
    "rate-limiting": ("rate limiting", "rate limit"),
    "rbac": ("rbac", "role based access control"),
    "redundancy": ("redundancy", "redundant"),
    "replication": ("replication", "replicated"),
    "resilience": ("resilience", "resilient"),
    "rest": ("rest", "restful"),
    "retries": ("retry", "retries"),
    "rollback": ("rollback", "roll back"),
    "saga": ("saga", "saga pattern"),
    "scalability": ("scalability", "scalable"),
    "schema-evolution": ("schema evolution",),
    "serverless": ("serverless",),
    "service-discovery": ("service discovery",),
    "service-mesh": ("service mesh",),
    "single-point-of-failure": ("single point of failure",),
    "slo": ("slo", "service level objective"),
    "sql-injection": ("sql injection",),
    "statelessness": ("statelessness", "stateless"),
    "strangler-pattern": ("strangler pattern",),
    "threat-modeling": ("threat modeling", "threat modelling"),
    "throughput": ("throughput",),
    "tls": ("tls", "transport layer security"),
    "tracing": ("tracing", "trace"),
    "transactions": ("transaction", "transactions"),
    "vertical-scaling": ("vertical scaling", "scale up"),
    "zero-trust": ("zero trust",),
}

QUALITY_SIGNALS = {
    "tradeoffs": re.compile(
        r"(?i)\b(?:trade-?offs?|advantage|disadvantage|downside|cost|benefit)\w*\b"
    ),
    "failure-modes": re.compile(
        r"(?i)\b(?:failure|fault|outage|timeout|retry|cascade|bottleneck|risk|degrad)\w*\b"
    ),
    "operations": re.compile(
        r"(?i)\b(?:deploy|monitor|observab|metric|logging|trace|operate|rollback|incident)\w*\b"
    ),
    "constraints": re.compile(
        r"(?i)\b(?:constraint|latency|throughput|budget|compliance|consistency|"
        r"availability|capacity|scale)\w*\b"
    ),
    "alternatives": re.compile(
        r"(?i)\b(?:alternative|instead|prefer|better fit|appropriate when|consider)\b"
    ),
    "concrete-example": re.compile(r"(?i)\b(?:for example|for instance|e\.g\.)\b"),
}

CONTINUOUS_METRICS = (
    "reference_term_recall",
    "reference_term_precision",
    "reference_term_f1",
    "reference_bigram_recall",
    "reference_concept_recall",
    "prompt_term_recall",
    "prompt_concept_recall",
    "reference_anchor_recall",
    "reference_quality_signal_recall",
)


@dataclass(frozen=True, slots=True)
class HeldOut:
    index: int
    prompt_hash: str
    family: str
    prompt: str
    reference: str
    prompt_messages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ScoredPair:
    heldout: HeldOut
    base_output: str
    expert_output: str
    base: dict[str, Any]
    expert: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_phrase(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^a-z0-9+#]+", " ", value)
    return " ".join(value.split())


def _stem(token: str) -> str:
    """A small transparent normalizer, not a linguistic or semantic stemmer."""

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(value: str) -> list[str]:
    return [
        _stem(token.casefold())
        for token in WORD_RE.findall(unicodedata.normalize("NFKC", value))
        if len(token) > 1 and token.casefold() not in STOPWORDS
    ]


def concepts(value: str) -> set[str]:
    padded = f" {normalized_phrase(value)} "
    found: set[str] = set()
    for concept, aliases in CONCEPT_ALIASES.items():
        if any(f" {normalized_phrase(alias)} " in padded for alias in aliases):
            found.add(concept)
    return found


def quality_signals(value: str) -> set[str]:
    return {
        label for label, pattern in QUALITY_SIGNALS.items() if pattern.search(value)
    }


def technical_anchors(value: str) -> set[str]:
    acronyms = {
        match.casefold()
        for match in re.findall(r"\b[A-Z][A-Z0-9./+-]{1,}\b", value)
    }
    numbers = {
        match.casefold()
        for match in re.findall(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)", value)
    }
    return acronyms | numbers


def recall(expected: set[Any], actual: set[Any]) -> float | None:
    return None if not expected else len(expected & actual) / len(expected)


def precision(expected: set[Any], actual: set[Any]) -> float | None:
    return None if not actual else len(expected & actual) / len(actual)


def f1(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return 0.0 if left + right == 0 else 2 * left * right / (left + right)


def repeated_sentence(value: str) -> bool:
    sentences = [
        normalized_phrase(sentence)
        for sentence in SENTENCE_RE.split(value)
        if len(content_tokens(sentence)) >= 5
    ]
    return len(sentences) != len(set(sentences))


def score_one(prompt: str, reference: str, generated: str) -> dict[str, Any]:
    reference_sequence = content_tokens(reference)
    generated_sequence = content_tokens(generated)
    prompt_terms = set(content_tokens(prompt))
    reference_terms = set(reference_sequence)
    generated_terms = set(generated_sequence)
    reference_bigrams = set(zip(reference_sequence, reference_sequence[1:]))
    generated_bigrams = set(zip(generated_sequence, generated_sequence[1:]))
    reference_concepts = concepts(reference)
    prompt_concepts = concepts(prompt)
    generated_concepts = concepts(generated)
    reference_anchors = technical_anchors(reference)
    generated_anchors = technical_anchors(generated)
    expected_signals = quality_signals(reference)
    actual_signals = quality_signals(generated)
    term_recall = recall(reference_terms, generated_terms)
    term_precision = precision(reference_terms, generated_terms)
    output_words = len(WORD_RE.findall(generated))
    reference_words = len(WORD_RE.findall(reference))
    return {
        "reference_term_recall": term_recall,
        "reference_term_precision": term_precision,
        "reference_term_f1": f1(term_recall, term_precision),
        "reference_bigram_recall": recall(reference_bigrams, generated_bigrams),
        "reference_concept_recall": recall(reference_concepts, generated_concepts),
        "prompt_term_recall": recall(prompt_terms, generated_terms),
        "prompt_concept_recall": recall(prompt_concepts, generated_concepts),
        "reference_anchor_recall": recall(reference_anchors, generated_anchors),
        "reference_quality_signal_recall": recall(expected_signals, actual_signals),
        "reference_concept_count": len(reference_concepts),
        "generated_concept_count": len(generated_concepts),
        "reference_anchor_count": len(reference_anchors),
        "reference_quality_signal_count": len(expected_signals),
        "output_words": output_words,
        "reference_words": reference_words,
        "length_ratio": output_words / max(1, reference_words),
        "empty": not generated.strip(),
        "refusal": bool(REFUSAL_RE.search(generated)),
        "thinking_trace": bool(THINK_RE.match(generated)),
        "repeated_sentence": repeated_sentence(generated),
    }


def jsonl_objects(path: Path) -> list[Any]:
    values: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return values


def chat_parts(value: object, *, path: Path, index: int) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"messages"}:
        raise ValueError(f"{path}:{index + 1}: expected a messages-only object")
    messages = value["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{path}:{index + 1}: invalid messages array")
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["content"], str)
        ):
            raise ValueError(f"{path}:{index + 1}: invalid message")
    if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
        raise ValueError(f"{path}:{index + 1}: expected final user/assistant pair")
    return messages[-2]["content"], messages[-1]["content"]


def load_heldout(validation: Path, provenance: Path) -> list[HeldOut]:
    rows = jsonl_objects(validation)
    sidecar = jsonl_objects(provenance)
    if len(rows) != len(sidecar):
        raise ValueError("validation and provenance row counts differ")
    heldout: list[HeldOut] = []
    for index, (row, metadata) in enumerate(zip(rows, sidecar, strict=True)):
        prompt, reference = chat_parts(row, path=validation, index=index)
        if not isinstance(metadata, dict):
            raise ValueError(f"{provenance}:{index + 1}: expected an object")
        prompt_hash = metadata.get("prompt_sha256")
        family = metadata.get("topic_family")
        if not isinstance(prompt_hash, str) or not isinstance(family, str):
            raise ValueError(
                f"{provenance}:{index + 1}: missing prompt_sha256/topic_family"
            )
        messages = row["messages"]
        prompt_messages = tuple(
            (message["role"], message["content"]) for message in messages[:-1]
        )
        heldout.append(
            HeldOut(
                index,
                prompt_hash,
                family,
                prompt,
                reference,
                prompt_messages,
            )
        )
    return heldout


def output_text(value: Any, *, path: Path, index: int, expected: HeldOut) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{index + 1}: expected a string or object")
    if "index" in value and value["index"] != index:
        raise ValueError(f"{path}:{index + 1}: index does not match row order")
    if "prompt_sha256" in value and value["prompt_sha256"] != expected.prompt_hash:
        raise ValueError(f"{path}:{index + 1}: prompt_sha256 mismatch")
    if "messages" in value:
        _, answer = chat_parts(
            {"messages": value["messages"]}, path=path, index=index
        )
        return answer
    keys = [key for key in ("output", "generated", "response", "text") if key in value]
    if len(keys) != 1 or not isinstance(value[keys[0]], str):
        raise ValueError(
            f"{path}:{index + 1}: needs exactly one string output/generated/response/text"
        )
    return value[keys[0]]


def load_outputs(
    path: Path, heldout: Sequence[HeldOut], *, allow_trailing_rows: bool = False
) -> list[str]:
    rows = jsonl_objects(path)
    if allow_trailing_rows and len(rows) >= len(heldout):
        rows = rows[: len(heldout)]
    if len(rows) != len(heldout):
        raise ValueError(
            f"{path} has {len(rows)} outputs; expected {len(heldout)} validation rows"
        )
    return [
        output_text(value, path=path, index=index, expected=heldout[index])
        for index, value in enumerate(rows)
    ]


def _completion_prompt_messages(
    value: object, *, path: Path, route: str, index: int
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{path}: routes.{route}.examples[{index}].prompt_messages must be a non-empty array"
        )
    messages: list[tuple[str, str]] = []
    for message_index, message in enumerate(value):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError(
                f"{path}: routes.{route}.examples[{index}].prompt_messages"
                f"[{message_index}] is invalid"
            )
        messages.append((message["role"], message["content"]))
    if messages[-1][0] != "user":
        raise ValueError(
            f"{path}: routes.{route}.examples[{index}] prompt must end with user"
        )
    return tuple(messages)


def _completion_route_outputs(
    route_value: object,
    *,
    route: str,
    path: Path,
    heldout: Sequence[HeldOut],
) -> list[str]:
    if not isinstance(route_value, dict) or not isinstance(
        route_value.get("examples"), list
    ):
        raise ValueError(f"{path}: routes.{route}.examples must be an array")
    examples = route_value["examples"]
    if not examples:
        raise ValueError(f"{path}: routes.{route}.examples is empty")
    if len(examples) > len(heldout):
        raise ValueError(
            f"{path}: routes.{route} has {len(examples)} rows; validation has {len(heldout)}"
        )

    outputs: list[str] = []
    for index, value in enumerate(examples):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: routes.{route}.examples[{index}] must be an object")
        expected = heldout[index]
        row_index = value.get("index")
        if type(row_index) is not int or row_index != index:
            raise ValueError(
                f"{path}: routes.{route}.examples[{index}].index must equal {index}"
            )
        prompt_messages = _completion_prompt_messages(
            value.get("prompt_messages"), path=path, route=route, index=index
        )
        if prompt_messages != expected.prompt_messages:
            raise ValueError(
                f"{path}: routes.{route}.examples[{index}] prompt does not match validation"
            )
        reference = value.get("reference")
        generated = value.get("generated")
        if not isinstance(reference, str) or reference != expected.reference:
            raise ValueError(
                f"{path}: routes.{route}.examples[{index}] reference does not match validation"
            )
        if not isinstance(generated, str):
            raise ValueError(
                f"{path}: routes.{route}.examples[{index}].generated must be a string"
            )
        if "reference_sha256" in value and value["reference_sha256"] != sha256_text(
            reference
        ):
            raise ValueError(
                f"{path}: routes.{route}.examples[{index}] reference_sha256 mismatch"
            )
        if "generated_sha256" in value and value["generated_sha256"] != sha256_text(
            generated
        ):
            raise ValueError(
                f"{path}: routes.{route}.examples[{index}] generated_sha256 mismatch"
            )
        outputs.append(generated)
    return outputs


def load_completion_report(
    path: Path,
    heldout: Sequence[HeldOut],
    *,
    validation_sha256: str,
) -> tuple[list[HeldOut], list[str], list[str]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid combined completion report JSON") from exc
    if not isinstance(report, dict) or not isinstance(report.get("routes"), dict):
        raise ValueError(f"{path}: expected an object containing routes")
    reported_validation_hash = report.get("validation_sha256")
    if (
        reported_validation_hash is not None
        and reported_validation_hash != validation_sha256
    ):
        raise ValueError(f"{path}: validation_sha256 does not match frozen validation")
    routes = report["routes"]
    if "base" not in routes or "expert" not in routes:
        raise ValueError(f"{path}: routes must contain base and expert")
    base_outputs = _completion_route_outputs(
        routes["base"], route="base", path=path, heldout=heldout
    )
    expert_outputs = _completion_route_outputs(
        routes["expert"], route="expert", path=path, heldout=heldout
    )
    if len(base_outputs) != len(expert_outputs):
        raise ValueError(f"{path}: base and expert route row counts differ")
    evaluated = report.get("evaluated_examples")
    if evaluated is not None and evaluated != len(base_outputs):
        raise ValueError(f"{path}: evaluated_examples does not match route row counts")
    selected = list(heldout[: len(base_outputs)])
    return selected, base_outputs, expert_outputs


def mean_metric(rows: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [float(row[name]) for row in rows if row[name] is not None]
    return {
        "mean": None if not values else statistics.fmean(values),
        "eligible_rows": len(values),
    }


def numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if not count:
        raise ValueError("cannot aggregate an empty evaluation slice")
    return {
        "rows": count,
        "metrics": {name: mean_metric(rows, name) for name in CONTINUOUS_METRICS},
        "length_ratio": numeric_summary([float(row["length_ratio"]) for row in rows]),
        "output_words": numeric_summary([float(row["output_words"]) for row in rows]),
        "rates": {
            name: sum(bool(row[name]) for row in rows) / count
            for name in ("empty", "refusal", "thinking_trace", "repeated_sentence")
        },
    }


def bootstrap_interval(
    differences: Sequence[float], *, samples: int, seed: str
) -> tuple[float, float] | None:
    if not differences or samples < 1:
        return None
    generator = random.Random(int(sha256_text(seed)[:16], 16))
    count = len(differences)
    means = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return (
        means[max(0, math.ceil(samples * 0.025) - 1)],
        means[max(0, math.ceil(samples * 0.975) - 1)],
    )


def paired_metric(
    pairs: Sequence[ScoredPair], name: str, *, bootstrap_samples: int, seed: str
) -> dict[str, Any]:
    values = [
        (float(pair.base[name]), float(pair.expert[name]))
        for pair in pairs
        if pair.base[name] is not None and pair.expert[name] is not None
    ]
    differences = [expert - base for base, expert in values]
    interval = bootstrap_interval(
        differences, samples=bootstrap_samples, seed=f"{seed}\0{name}"
    )
    epsilon = 1e-12
    return {
        "eligible_rows": len(values),
        "base_mean": None if not values else statistics.fmean(base for base, _ in values),
        "expert_mean": None if not values else statistics.fmean(expert for _, expert in values),
        "expert_minus_base_mean": (
            None if not differences else statistics.fmean(differences)
        ),
        "paired_bootstrap_95_percent_ci": None if interval is None else list(interval),
        "expert_wins": sum(difference > epsilon for difference in differences),
        "ties": sum(abs(difference) <= epsilon for difference in differences),
        "base_wins": sum(difference < -epsilon for difference in differences),
    }


def macro_family_metric(pairs: Sequence[ScoredPair], name: str) -> dict[str, Any]:
    grouped: dict[str, list[ScoredPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.heldout.family].append(pair)
    family_deltas: dict[str, float] = {}
    for family, items in sorted(grouped.items()):
        values = [
            float(item.expert[name]) - float(item.base[name])
            for item in items
            if item.base[name] is not None and item.expert[name] is not None
        ]
        if values:
            family_deltas[family] = statistics.fmean(values)
    return {
        "expert_minus_base_macro_family_mean": (
            None if not family_deltas else statistics.fmean(family_deltas.values())
        ),
        "family_deltas": family_deltas,
    }


def score_pairs(
    heldout: Sequence[HeldOut], base: Sequence[str], expert: Sequence[str]
) -> list[ScoredPair]:
    return [
        ScoredPair(
            item,
            base[index],
            expert[index],
            score_one(item.prompt, item.reference, base[index]),
            score_one(item.prompt, item.reference, expert[index]),
        )
        for index, item in enumerate(heldout)
    ]


def family_aggregates(
    pairs: Sequence[ScoredPair], route: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.heldout.family].append(getattr(pair, route))
    return {family: aggregate(rows) for family, rows in sorted(grouped.items())}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def balanced_review_sample(
    pairs: Sequence[ScoredPair], *, limit: int, seed: str
) -> list[ScoredPair]:
    grouped: dict[str, list[ScoredPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.heldout.family].append(pair)
    for family in grouped:
        grouped[family].sort(
            key=lambda item: sha256_text(
                f"{seed}\0review-rank\0{item.heldout.prompt_hash}"
            )
        )
    selected: list[ScoredPair] = []
    offsets = {family: 0 for family in grouped}
    while len(selected) < min(limit, len(pairs)):
        progressed = False
        for family in sorted(grouped):
            offset = offsets[family]
            if offset < len(grouped[family]):
                selected.append(grouped[family][offset])
                offsets[family] += 1
                progressed = True
                if len(selected) == min(limit, len(pairs)):
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda pair: pair.heldout.index)


def emit_blind_review(
    pairs: Sequence[ScoredPair],
    *,
    review_path: Path,
    key_path: Path,
    limit: int,
    seed: str,
) -> dict[str, Any]:
    selected = balanced_review_sample(pairs, limit=limit, seed=seed)
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for pair in selected:
        base_is_a = int(
            sha256_text(f"{seed}\0side\0{pair.heldout.prompt_hash}")[:2], 16
        ) % 2 == 0
        answer_a = pair.base_output if base_is_a else pair.expert_output
        answer_b = pair.expert_output if base_is_a else pair.base_output
        review_rows.append(
            {
                "index": pair.heldout.index,
                "prompt_sha256": pair.heldout.prompt_hash,
                "topic_family": pair.heldout.family,
                "prompt": pair.heldout.prompt,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "grade": {
                    "preferred": None,
                    "a_accuracy_1_to_5": None,
                    "b_accuracy_1_to_5": None,
                    "a_tradeoffs_and_failure_modes_1_to_5": None,
                    "b_tradeoffs_and_failure_modes_1_to_5": None,
                    "a_relevance_1_to_5": None,
                    "b_relevance_1_to_5": None,
                    "a_clarity_1_to_5": None,
                    "b_clarity_1_to_5": None,
                    "notes": "",
                },
            }
        )
        key_rows.append(
            {
                "index": pair.heldout.index,
                "prompt_sha256": pair.heldout.prompt_hash,
                "answer_a": "base" if base_is_a else "expert",
                "answer_b": "expert" if base_is_a else "base",
            }
        )
    review_count = write_jsonl_atomic(review_path, review_rows)
    key_count = write_jsonl_atomic(key_path, key_rows)
    if review_count != key_count:
        raise RuntimeError("blind review/key row count mismatch")
    return {
        "review_path": str(review_path.resolve()),
        "key_path": str(key_path.resolve()),
        "rows": review_count,
        "topic_families": dict(
            sorted(Counter(pair.heldout.family for pair in selected).items())
        ),
    }


def emit_details(path: Path, pairs: Sequence[ScoredPair]) -> int:
    return write_jsonl_atomic(
        path,
        (
            {
                "index": pair.heldout.index,
                "prompt_sha256": pair.heldout.prompt_hash,
                "topic_family": pair.heldout.family,
                "prompt": pair.heldout.prompt,
                "reference": pair.heldout.reference,
                "base_output": pair.base_output,
                "expert_output": pair.expert_output,
                "base_metrics": pair.base,
                "expert_metrics": pair.expert,
            }
            for pair in pairs
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--completion-report",
        type=Path,
        help="combined evaluate_completion.py JSON with base/expert route examples",
    )
    parser.add_argument("--base-outputs", type=Path)
    parser.add_argument("--expert-outputs", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all rows")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", default="software-architecture-v1-eval")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--blind-review", type=Path)
    parser.add_argument("--blind-key", type=Path)
    parser.add_argument("--review-limit", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.bootstrap_samples < 0 or args.review_limit < 1:
        raise ValueError("limit/bootstrap-samples cannot be negative; review-limit must be positive")
    if (args.blind_review is None) != (args.blind_key is None):
        raise ValueError("--blind-review and --blind-key must be supplied together")
    separate_outputs = args.base_outputs is not None or args.expert_outputs is not None
    if args.completion_report is not None and separate_outputs:
        raise ValueError(
            "--completion-report cannot be combined with --base-outputs/--expert-outputs"
        )
    if args.completion_report is None and (
        args.base_outputs is None or args.expert_outputs is None
    ):
        raise ValueError(
            "supply --completion-report or both --base-outputs and --expert-outputs"
        )

    validation = args.validation.resolve()
    provenance = args.provenance.resolve()
    validation_digest = sha256_file(validation)
    heldout = load_heldout(validation, provenance)
    input_metadata: dict[str, Any]
    if args.completion_report is not None:
        completion_path = args.completion_report.resolve()
        heldout, base_outputs, expert_outputs = load_completion_report(
            completion_path,
            heldout,
            validation_sha256=validation_digest,
        )
        input_metadata = {
            "input_mode": "evaluate_completion_combined_report",
            "completion_report": str(completion_path),
            "completion_report_sha256": sha256_file(completion_path),
        }
        if args.limit:
            if args.limit > len(heldout):
                raise ValueError(
                    f"--limit {args.limit} exceeds {len(heldout)} completion-report rows"
                )
            heldout = heldout[: args.limit]
            base_outputs = base_outputs[: args.limit]
            expert_outputs = expert_outputs[: args.limit]
    else:
        if args.base_outputs is None or args.expert_outputs is None:
            raise AssertionError("separate output paths were not validated")
        base_path = args.base_outputs.resolve()
        expert_path = args.expert_outputs.resolve()
        if args.limit:
            heldout = heldout[: args.limit]
        base_outputs = load_outputs(
            base_path, heldout, allow_trailing_rows=bool(args.limit)
        )
        expert_outputs = load_outputs(
            expert_path, heldout, allow_trailing_rows=bool(args.limit)
        )
        input_metadata = {
            "input_mode": "separate_jsonl",
            "base_outputs": str(base_path),
            "base_outputs_sha256": sha256_file(base_path),
            "expert_outputs": str(expert_path),
            "expert_outputs_sha256": sha256_file(expert_path),
        }
    pairs = score_pairs(heldout, base_outputs, expert_outputs)

    base_rows = [pair.base for pair in pairs]
    expert_rows = [pair.expert for pair in pairs]
    paired = {
        name: {
            **paired_metric(
                pairs,
                name,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            **macro_family_metric(pairs, name),
        }
        for name in CONTINUOUS_METRICS
    }
    report: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "validation": str(validation),
        "validation_sha256": validation_digest,
        "provenance": str(provenance),
        "provenance_sha256": sha256_file(provenance),
        **input_metadata,
        "evaluated_rows": len(pairs),
        "bootstrap_samples": args.bootstrap_samples,
        "metric_direction": "higher is more lexical agreement except diagnostic rates/length",
        "base": {
            "overall": aggregate(base_rows),
            "by_topic_family": family_aggregates(pairs, "base"),
        },
        "expert": {
            "overall": aggregate(expert_rows),
            "by_topic_family": family_aggregates(pairs, "expert"),
        },
        "paired_comparison": paired,
        "metric_notes": {
            "reference_term_recall_precision_f1": (
                "Set overlap after NFKC, case folding, stop-word removal, and a small "
                "auditable suffix normalizer. Extra valid detail lowers precision."
            ),
            "reference_bigram_recall": (
                "Exact adjacent normalized content-token pair recall; mainly an imitation/"
                "phrase-overlap diagnostic."
            ),
            "reference_concept_recall": (
                "Exact alias coverage from the script's finite architecture concept lexicon."
            ),
            "prompt_term_and_concept_recall": (
                "Checks lexical engagement with the question; parroting the prompt can inflate it."
            ),
            "reference_anchor_recall": (
                "Exact overlap of reference acronyms and numeric strings; evaluated only when present."
            ),
            "reference_quality_signal_recall": (
                "Coverage of reference discourse categories such as tradeoffs, failure modes, "
                "operations, constraints, alternatives, and examples."
            ),
            "macro_family_mean": (
                "Each represented topic family receives equal weight; small families have higher variance."
            ),
        },
        "limitations": [
            "No reported metric verifies factual correctness, design feasibility, or safety.",
            "The held-out answers are synthetic single references, not authoritative gold solutions.",
            "Correct paraphrases and valid alternative architectures may have low lexical overlap.",
            "A model can copy reference vocabulary while making incorrect relationships or claims.",
            "Prompt groups are held out, but architecture concepts and source-generation style overlap training.",
            "Use identical decoding settings for base and expert outputs and use blinded human review as the primary quality decision.",
        ],
    }
    if args.details_output is not None:
        report["details"] = {
            "path": str(args.details_output.resolve()),
            "rows": emit_details(args.details_output.resolve(), pairs),
        }
    if args.blind_review is not None and args.blind_key is not None:
        report["blind_review"] = emit_blind_review(
            pairs,
            review_path=args.blind_review.resolve(),
            key_path=args.blind_key.resolve(),
            limit=args.review_limit,
            seed=args.seed,
        )
    if args.output is not None:
        write_json_atomic(args.output.resolve(), report)
    summary = {
        "evaluated_rows": report["evaluated_rows"],
        "base_overall": report["base"]["overall"],
        "expert_overall": report["expert"]["overall"],
        "paired_comparison": report["paired_comparison"],
        "report_path": None if args.output is None else str(args.output.resolve()),
        "details": report.get("details"),
        "blind_review": report.get("blind_review"),
        "warning": "Overlap diagnostics are not semantic correctness; use blinded review.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
