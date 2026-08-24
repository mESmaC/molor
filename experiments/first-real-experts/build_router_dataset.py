"""Build the balanced, source-label bootstrap corpus for the prompt router.

Only existing *training* corpora are read.  Existing expert validation files remain untouched and
are not inputs to this script.  Each output row has the intentionally small router schema:

    {"prompt": "...latest user content...", "route_id": "..."}

System instructions and assistant answers are never copied into router features.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable


ROUTE_IDS = (
    "embedded-knowledge-v1",
    "ocb-can-verified-v1",
    "software-architecture-v1",
    "base/null",
)
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    route_id: str
    dataset: Path
    provenance: Path
    group_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    prompt: str
    normalized_prompt: str
    route_id: str
    source: str
    source_line: int
    group_key: str
    split_hint: str | None = None

    def output_row(self) -> dict[str, str]:
        return {"prompt": self.prompt, "route_id": self.route_id}


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_prompt(prompt: str) -> str:
    value = unicodedata.normalize("NFKC", prompt).casefold()
    value = " ".join(value.split())
    return value.strip()


def _structural_prompt(prompt: str) -> str:
    """Fallback family signature that groups obvious templated variants."""

    value = _normalize_prompt(prompt)
    value = re.sub(r"`[^`]{1,160}`", "<slot>", value)
    value = re.sub(r"\b0x[0-9a-f]+\b", "<number>", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", value)
    return value


def _latest_user_prompt(row: object, *, path: Path, line_number: int) -> str:
    if not isinstance(row, dict) or set(row) != {"messages"}:
        raise ValueError(f"{path}:{line_number}: expected one messages array")
    messages = row["messages"]
    if not isinstance(messages, list):
        raise ValueError(f"{path}:{line_number}: messages must be an array")
    for message in reversed(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{line_number}: invalid message")
        if message.get("role") == "user":
            prompt = message.get("content")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: user content must not be empty")
            return prompt.strip()
    raise ValueError(f"{path}:{line_number}: no user message")


def _group_value(provenance: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field_name in fields:
        value = provenance.get(field_name)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            rendered = str(value).strip()
            if rendered:
                return f"{field_name}:{rendered}"
    return None


def _read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _load_source(spec: SourceSpec) -> list[Candidate]:
    chat_rows = _read_jsonl(spec.dataset)
    provenance_rows = _read_jsonl(spec.provenance)
    if len(chat_rows) != len(provenance_rows):
        raise ValueError(
            f"dataset/provenance length mismatch for {spec.name}: "
            f"{len(chat_rows)} != {len(provenance_rows)}"
        )
    results: list[Candidate] = []
    for index, (chat_row, provenance_row) in enumerate(
        zip(chat_rows, provenance_rows, strict=True),
        start=1,
    ):
        if not isinstance(provenance_row, dict):
            raise ValueError(f"{spec.provenance}:{index}: provenance must be an object")
        prompt = _latest_user_prompt(chat_row, path=spec.dataset, line_number=index)
        normalized = _normalize_prompt(prompt)
        if not normalized:
            continue
        source_group = _group_value(provenance_row, spec.group_fields)
        if source_group is None:
            source_group = f"structural:{_structural_prompt(prompt)}"
        results.append(
            Candidate(
                prompt=prompt,
                normalized_prompt=normalized,
                route_id=spec.route_id,
                source=spec.name,
                source_line=index,
                # Prefixing the source prevents unrelated upstream ID namespaces colliding.
                group_key=f"{spec.name}/{source_group}",
            )
        )
    return results


def _deduplicate(candidates: Iterable[Candidate]) -> tuple[list[Candidate], dict[str, int]]:
    by_prompt: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_prompt.setdefault(candidate.normalized_prompt, []).append(candidate)
    kept: list[Candidate] = []
    exact_duplicates = 0
    cross_route_ambiguous = 0
    for rows in by_prompt.values():
        route_ids = {row.route_id for row in rows}
        if len(route_ids) > 1:
            cross_route_ambiguous += len(rows)
            continue
        # Curated intent seeds win an exact same-label collision with a source row so the
        # explicitly held-out family assignment is not accidentally discarded.
        rows.sort(key=lambda row: (row.split_hint is None, row.source, row.source_line))
        kept.append(rows[0])
        exact_duplicates += len(rows) - 1
    return kept, {
        "exact_duplicate_rows_removed": exact_duplicates,
        "cross_route_ambiguous_rows_removed": cross_route_ambiguous,
    }


def _split_for_group(group_key: str, seed: int) -> str:
    bucket = int(_stable_hash(seed, group_key)[:8], 16) % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"


_SEED_TEMPLATES = (
    "Explain {subject} with a concrete example.",
    "How would you debug {subject} on a real project?",
    "What mistakes commonly occur with {subject}?",
    "Give a practical implementation checklist for {subject}.",
    "Compare the main design choices for {subject}.",
    "Walk through a small problem involving {subject}.",
)


# Families, rather than paraphrases, are assigned to splits.  This curriculum teaches broad
# routing intent missing from corpus-template labels while preserving an honest family-held-out
# check.  Every route has 10 train, 2 validation, and 2 test families with six prompts each.
_MANUAL_SEED_FAMILIES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "embedded-knowledge-v1": {
        "train": {
            "c167-interrupts": (
                "C167 interrupt vector tables and priority levels",
                "configuring a C167 PEC channel for an interrupt source",
                "an unexpected C167 ISR that never fires",
                "C166/C167 trap handling and interrupt nesting",
                "saving registers correctly in a C167 interrupt handler",
                "C167 timer interrupt initialization",
            ),
            "embedded-c": (
                "volatile access to memory-mapped registers in embedded C",
                "integer promotion bugs in an 8-bit C peripheral driver",
                "safe bit masking of a hardware status register",
                "static storage and startup initialization on a microcontroller",
                "a reentrant C function called from an ISR and main loop",
                "bounded polling loops for embedded C firmware",
            ),
            "embedded-cpp": (
                "RAII for an interrupt guard in embedded C++",
                "avoiding heap allocation in microcontroller C++",
                "zero-cost abstractions for a C++ register wrapper",
                "placement new for statically allocated embedded objects",
                "exception-free error handling in embedded C++",
                "constexpr configuration tables in firmware",
            ),
            "assembly": (
                "an assembly startup routine that initializes stack pointers",
                "calling-convention register preservation in an ISR",
                "reading a mixed C and assembly disassembly listing",
                "an inline assembly memory barrier around device I/O",
                "a hand-written context switch in microcontroller assembly",
                "debugging a corrupted return address at instruction level",
            ),
            "linker-scripts": (
                "placing an interrupt vector table with a linker script",
                "defining FLASH and RAM MEMORY regions in a linker script",
                "copying the data section and clearing BSS at startup",
                "using KEEP for sections removed by garbage collection",
                "locating a DMA buffer at an aligned address",
                "diagnosing a linker region overflow in firmware",
            ),
            "misra-c": (
                "MISRA C essential type category violations",
                "documenting a justified MISRA deviation",
                "MISRA rules around casts and signed arithmetic",
                "checking switch statements for MISRA compliance",
                "MISRA-safe handling of function-like macros",
                "distinguishing mandatory and advisory MISRA findings",
            ),
            "microchip-sdk": (
                "configuring a PIC32 peripheral with the Microchip SDK",
                "an MPLAB Harmony clock configuration problem",
                "Microchip XC compiler pragmas and configuration bits",
                "using MCC-generated initialization without losing user edits",
                "a dsPIC DMA setup using Microchip headers",
                "debugging a Microchip peripheral library call",
            ),
            "memory-mapped-io": (
                "read-modify-write hazards on a peripheral register",
                "write-one-to-clear status bits",
                "register aliases for atomic set and clear operations",
                "peripheral register alignment and access width",
                "ordering writes to clock and reset registers",
                "a reserved-bit-safe hardware register update",
            ),
            "rtos-concurrency": (
                "priority inversion in a small RTOS",
                "passing data safely from an ISR to an RTOS task",
                "choosing a mutex, semaphore, or queue in firmware",
                "measuring worst-case RTOS task latency",
                "a deadlock between two embedded tasks",
                "using critical sections without breaking interrupt latency",
            ),
            "firmware-toolchain": (
                "reading a microcontroller map file to find RAM usage",
                "debugging optimized firmware with a JTAG probe",
                "compiler optimization changing hardware timing",
                "using objdump to inspect an embedded ELF file",
                "a boot failure before reaching main",
                "reproducing a cross-compiler ABI mismatch",
            ),
        },
        "validation": {
            "dma-coherency": (
                "cache maintenance around a DMA receive buffer",
                "a DMA descriptor visible to hardware but stale on the CPU",
                "aligning and invalidating cache lines for DMA",
                "non-cacheable memory regions for peripheral DMA",
                "memory barriers before handing a buffer to DMA",
                "debugging data corruption caused by DMA coherency",
            ),
            "bootloaders": (
                "an A/B firmware bootloader with rollback",
                "validating an application image before jumping to it",
                "relocating vector tables in a bootloader",
                "power-loss-safe firmware update metadata",
                "sharing flash erase routines between bootloader and application",
                "a bootloader that jumps to an invalid stack pointer",
            ),
        },
        "test": {
            "fixed-point": (
                "Q-format fixed-point multiplication with saturation",
                "choosing fixed-point scale for a sensor filter",
                "overflow analysis in fixed-point control code",
                "rounding negative fixed-point values",
                "converting ADC readings to fixed-point engineering units",
                "testing a fixed-point implementation against floating point",
            ),
            "peripheral-drivers": (
                "a UART driver that loses bytes under interrupt load",
                "SPI chip-select timing in a bare-metal driver",
                "recovering an I2C bus after a stuck slave",
                "double-buffered ADC sampling",
                "debouncing a GPIO input without blocking",
                "a watchdog reset loop during peripheral initialization",
            ),
        },
    },
    "ocb-can-verified-v1": {
        "train": {
            "dbc-motorola": (
                "decoding a Motorola big-endian CAN signal from a DBC",
                "Motorola bit numbering for a signal crossing byte boundaries",
                "a DBC @0 big-endian signal in an eight-byte payload",
                "mapping a Motorola start bit to payload bit positions",
                "extracting an unsigned big-endian DBC field",
                "encoding a multi-byte Motorola CAN signal",
            ),
            "dbc-intel": (
                "decoding an Intel little-endian CAN signal from a DBC",
                "a DBC @1 signal spanning three payload bytes",
                "mapping a little-endian start bit to byte offsets",
                "extracting an unsigned Intel-order CAN field",
                "encoding a little-endian DBC signal",
                "comparing Intel and Motorola DBC byte order",
            ),
            "start-bits": (
                "interpreting DBC start bit 7 for a 12-bit signal",
                "finding every payload bit used by a DBC signal",
                "checking whether two CAN signals overlap",
                "a signal whose start bit is at a byte boundary",
                "converting a DBC start bit into a mask and shift",
                "validating start bit and length against an eight-byte DLC",
            ),
            "scale-offset": (
                "applying DBC scale and offset to a raw value",
                "recovering a raw CAN value from a physical value",
                "checking physical min and max for a scaled DBC signal",
                "handling a fractional factor in CAN decoding",
                "a negative DBC offset with an unsigned raw field",
                "preserving units while decoding a CAN signal",
            ),
            "payload-decode": (
                "decoding several DBC signals from one hexadecimal CAN payload",
                "extracting a counter field from an observed CAN frame",
                "turning eight payload bytes into raw signal values",
                "checking DLC before decoding a CAN frame",
                "decoding a payload when unused bits are nonzero",
                "explaining each mask and shift in a CAN decode",
            ),
            "payload-encode": (
                "packing physical signal values into a CAN payload",
                "updating one DBC signal without corrupting neighboring bits",
                "clamping an out-of-range value before CAN encoding",
                "rounding a scaled value to a raw CAN integer",
                "encoding multiple fields into an initially zero payload",
                "verifying an encoded frame by decoding it again",
            ),
            "signed-signals": (
                "sign extending a signed 12-bit CAN signal",
                "decoding a negative two's-complement DBC value",
                "the raw range of a signed DBC field",
                "encoding a negative physical value into a CAN payload",
                "signed versus unsigned interpretation of identical payload bits",
                "a signed signal with scale and offset",
            ),
            "multiplexing": (
                "decoding a multiplexed DBC message",
                "selecting signals from a CAN multiplexer value",
                "validating mutually exclusive multiplexed fields",
                "a nested multiplexor in a DBC",
                "encoding the multiplexer before its dependent signals",
                "handling an unknown multiplexor value",
            ),
            "counters-checksums": (
                "checking an alive counter in repeated CAN frames",
                "detecting a rolling counter discontinuity",
                "calculating a payload checksum from a documented algorithm",
                "distinguishing a checksum signal from ordinary data",
                "excluding checksum bytes from their own calculation",
                "validating counter and checksum fields together",
            ),
            "can-fd": (
                "decoding a 32-byte CAN FD payload from a DBC",
                "DLC encoding versus payload length in CAN FD",
                "a DBC signal above byte seven in a CAN FD frame",
                "bit-rate switching in a captured CAN FD message",
                "migrating a classic CAN definition to CAN FD",
                "validating CAN FD signal bounds",
            ),
        },
        "validation": {
            "catalog-lookups": (
                "looking up a signal definition by CAN ID and message name",
                "reporting DBC start bit length order factor offset and unit",
                "finding every signal cataloged for a trusted CAN frame",
                "distinguishing hexadecimal and decimal CAN identifiers",
                "locating a named signal in a vehicle DBC catalog",
                "answering a CAN definition lookup without inventing semantics",
            ),
            "cross-byte-fields": (
                "a 13-bit CAN field crossing three bytes",
                "constructing a mask for a non-byte-aligned DBC signal",
                "a big-endian field that snakes across payload bytes",
                "testing boundary bits of a cross-byte CAN signal",
                "visualizing payload bits used by a cross-byte field",
                "comparing cross-byte extraction in both byte orders",
            ),
        },
        "test": {
            "dbc-validation": (
                "detecting overlapping signals in a DBC message",
                "rejecting a signal definition beyond the message DLC",
                "checking duplicate CAN IDs in a DBC catalog",
                "validating scale offset and physical ranges",
                "finding inconsistent byte-order annotations",
                "building test vectors for a DBC decoder",
            ),
            "bus-analysis": (
                "matching changing payload bits to a suspected CAN signal",
                "comparing two CAN captures at the signal level",
                "filtering a trace by arbitration ID and payload pattern",
                "identifying a periodic counter in raw CAN frames",
                "separating message frequency from signal semantics",
                "testing a hypothesized DBC definition against captures",
            ),
        },
    },
    "software-architecture-v1": {
        "train": {
            "distributed-systems": (
                "choosing consistency guarantees for a distributed system",
                "handling partial failure across networked services",
                "leader election and split-brain behavior",
                "idempotency in distributed request processing",
                "designing around network partitions",
                "ordering events across distributed components",
            ),
            "microservices": (
                "splitting a monolith into service boundaries",
                "avoiding a distributed monolith",
                "service ownership and deployment independence",
                "synchronous versus asynchronous microservice calls",
                "data ownership between microservices",
                "when a modular monolith is preferable to microservices",
            ),
            "databases": (
                "choosing a relational or document database",
                "database indexing for a read-heavy workload",
                "partitioning a rapidly growing data store",
                "transaction boundaries across aggregates",
                "read replicas and consistency lag",
                "schema migration without downtime",
            ),
            "caching": (
                "cache-aside versus write-through caching",
                "preventing a cache stampede",
                "choosing cache keys and invalidation rules",
                "multi-level application caching",
                "stale data and cache consistency",
                "when adding a cache makes a system worse",
            ),
            "api-design": (
                "versioning a public HTTP API",
                "idempotent retry behavior for an API",
                "pagination for a changing result set",
                "REST versus RPC for an internal service",
                "evolving an API without breaking clients",
                "rate limiting and backpressure at an API boundary",
            ),
            "event-driven": (
                "designing an event-driven order workflow",
                "at-least-once delivery and duplicate events",
                "event schemas and compatibility",
                "a transactional outbox pattern",
                "queues versus publish-subscribe topics",
                "recovering a failed event consumer",
            ),
            "consistency": (
                "sagas for a multi-service business transaction",
                "optimistic concurrency control",
                "eventual consistency in a user-facing workflow",
                "compensating actions after partial success",
                "exactly-once claims in messaging systems",
                "conflict resolution for concurrent updates",
            ),
            "resilience": (
                "timeouts retries and circuit breakers",
                "bulkheads for isolating service failure",
                "graceful degradation under dependency outage",
                "retry storms during a regional incident",
                "load shedding before system collapse",
                "multi-region disaster recovery objectives",
            ),
            "observability": (
                "logs metrics and traces for a service platform",
                "service-level indicators and objectives",
                "correlation IDs across asynchronous work",
                "high-cardinality telemetry tradeoffs",
                "alerting on symptoms instead of causes",
                "observability for a distributed transaction",
            ),
            "security-architecture": (
                "zero-trust boundaries between services",
                "centralized identity and service authorization",
                "secrets management for deployed workloads",
                "threat modeling an internet-facing API",
                "encryption and key management architecture",
                "tenant isolation in a shared platform",
            ),
        },
        "validation": {
            "data-platforms": (
                "batch versus streaming data pipelines",
                "a lakehouse versus a data warehouse",
                "data lineage across an analytics platform",
                "late-arriving events in stream processing",
                "orchestrating reliable ETL workflows",
                "serving analytical and transactional workloads",
            ),
            "deployment-platforms": (
                "Kubernetes versus managed serverless deployment",
                "blue-green versus canary releases",
                "multi-tenant cluster isolation",
                "progressive delivery with automated rollback",
                "platform engineering golden paths",
                "choosing deployment units for a large system",
            ),
        },
        "test": {
            "multitenancy": (
                "database-per-tenant versus shared-schema isolation",
                "noisy-neighbor controls in a SaaS platform",
                "tenant-aware authorization throughout a system",
                "customization without per-tenant forks",
                "migrating one tenant independently",
                "measuring and billing tenant resource usage",
            ),
            "cost-scaling": (
                "architecting for bursty demand without permanent overprovisioning",
                "storage cost versus query latency",
                "unit economics of a high-volume service",
                "reserved capacity versus autoscaling",
                "reducing cross-region data transfer cost",
                "a cost-aware retention and archival design",
            ),
        },
    },
    "base/null": {
        "train": {
            "terminal": (
                "finding large files with a shell command",
                "quoting a path containing brackets in PowerShell",
                "checking which process owns a TCP port",
                "using git to inspect an earlier revision",
                "redirecting standard error in Bash",
                "reading environment variables from the command line",
            ),
            "finance": (
                "compound interest on a savings balance",
                "the difference between a stock and a bond",
                "reading revenue and operating margin from a statement",
                "diversification in a long-term investment portfolio",
                "calculating a loan payment",
                "the risk and return tradeoff of an index fund",
            ),
            "business": (
                "validating demand before starting a small business",
                "basic cash-flow planning for a new company",
                "choosing a legal structure for a startup",
                "pricing a service business",
                "preparing to speak with a business lender",
                "tracking operating expenses and runway",
            ),
            "marketing": (
                "defining an ideal customer profile",
                "writing a simple go-to-market plan",
                "measuring conversion through a sales funnel",
                "testing a marketing channel on a small budget",
                "positioning a product against alternatives",
                "customer interviews for market research",
            ),
            "mathematics": (
                "solving a quadratic equation",
                "computing the area of a triangle",
                "interpreting the derivative of a function",
                "adding fractions with different denominators",
                "a conditional probability problem",
                "checking a system of two linear equations",
            ),
            "writing": (
                "rewriting an email to sound concise and polite",
                "organizing an argumentative essay",
                "fixing ambiguous pronouns in a paragraph",
                "summarizing a long passage in plain language",
                "drafting a professional meeting agenda",
                "improving the flow of a short report",
            ),
            "general-science": (
                "why seasons occur on Earth",
                "the difference between mass and weight",
                "how natural selection changes populations",
                "why metals conduct electricity",
                "the water cycle",
                "the role of enzymes in a chemical reaction",
            ),
            "planning": (
                "making a weekly study schedule",
                "breaking a large household project into steps",
                "planning a low-cost weekend trip",
                "prioritizing tasks with conflicting deadlines",
                "creating a grocery plan for several meals",
                "setting measurable goals for a new habit",
            ),
            "customer-service": (
                "responding to a delayed shipment complaint",
                "asking a customer for missing account details",
                "explaining a refund timeline clearly",
                "de-escalating a frustrated support conversation",
                "summarizing next steps after a service call",
                "writing a helpful response when a request is impossible",
            ),
            "language-learning": (
                "practicing common Spanish travel phrases",
                "the difference between affect and effect",
                "correcting verb tense in an English sentence",
                "learning polite greetings in French",
                "explaining a German noun case",
                "building vocabulary with example sentences",
            ),
        },
        "validation": {
            "cooking": (
                "substituting an ingredient in a soup",
                "scaling a recipe from four servings to six",
                "keeping roasted vegetables from becoming soggy",
                "planning a simple vegetarian dinner",
                "reading temperatures in a baking recipe",
                "using leftovers safely in a new meal",
            ),
            "travel": (
                "building a three-day city itinerary",
                "packing lightly for variable weather",
                "comparing a train trip with a rental car",
                "estimating a basic travel budget",
                "choosing activities for a family vacation",
                "writing a checklist for an international flight",
            ),
        },
        "test": {
            "creative-writing": (
                "brainstorming a mystery story premise",
                "writing dialogue between two old friends",
                "creating a name for a fictional town",
                "improving the opening of a short story",
                "describing a rainy street without clichés",
                "outlining a character's emotional arc",
            ),
            "general-knowledge": (
                "the causes of the industrial revolution",
                "how a parliamentary system works",
                "why time zones exist",
                "the difference between weather and climate",
                "how public libraries organize books",
                "the purpose of a scientific control group",
            ),
        },
    },
}


def _manual_seed_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    line_number = 0
    for route_id in ROUTE_IDS:
        by_split = _MANUAL_SEED_FAMILIES[route_id]
        if set(by_split) != set(SPLITS):
            raise AssertionError(f"manual seed split set is incomplete for {route_id}")
        for split in SPLITS:
            for family, subjects in by_split[split].items():
                if len(subjects) != len(_SEED_TEMPLATES):
                    raise AssertionError(f"seed family {route_id}/{family} needs six subjects")
                group_key = f"manual-seed/{route_id}/{family}"
                for template, subject in zip(_SEED_TEMPLATES, subjects, strict=True):
                    line_number += 1
                    prompt = template.format(subject=subject)
                    candidates.append(
                        Candidate(
                            prompt=prompt,
                            normalized_prompt=_normalize_prompt(prompt),
                            route_id=route_id,
                            source="manual-intent-seed-v1",
                            source_line=line_number,
                            group_key=group_key,
                            split_hint=split,
                        )
                    )
    return candidates


def _select_balanced(
    candidates: Iterable[Candidate],
    *,
    per_route: int,
    seed: int,
) -> tuple[dict[str, list[Candidate]], dict[str, Any]]:
    source_pools: dict[str, dict[str, list[Candidate]]] = {
        split: {route_id: [] for route_id in ROUTE_IDS} for split in SPLITS
    }
    seed_pools: dict[str, dict[str, list[Candidate]]] = {
        split: {route_id: [] for route_id in ROUTE_IDS} for split in SPLITS
    }
    group_splits: dict[str, str] = {}
    for candidate in candidates:
        split = candidate.split_hint or _split_for_group(candidate.group_key, seed)
        if split not in SPLITS:
            raise ValueError(f"invalid split hint: {split!r}")
        existing = group_splits.setdefault(candidate.group_key, split)
        if existing != split:
            raise AssertionError("one group was assigned to multiple splits")
        pools = seed_pools if candidate.split_hint is not None else source_pools
        pools[split][candidate.route_id].append(candidate)

    desired = {
        "train": int(per_route * SPLIT_RATIOS["train"]),
        "validation": int(per_route * SPLIT_RATIOS["validation"]),
    }
    desired["test"] = per_route - desired["train"] - desired["validation"]
    selected: dict[str, list[Candidate]] = {}
    report: dict[str, Any] = {
        "available": {},
        "selected": {},
        "manual_seed_counts": {},
        "desired_per_route": desired,
    }
    for split in SPLITS:
        report["available"][split] = {
            route_id: len(source_pools[split][route_id]) for route_id in ROUTE_IDS
        }
        seed_counts = {
            route_id: len(seed_pools[split][route_id]) for route_id in ROUTE_IDS
        }
        if len(set(seed_counts.values())) != 1:
            raise AssertionError(f"manual seed curriculum is not route-balanced in {split}")
        report["manual_seed_counts"][split] = seed_counts
        seed_count = next(iter(seed_counts.values()))
        if seed_count > desired[split]:
            raise ValueError(f"manual seed curriculum exceeds the {split} split budget")
        desired_source_count = desired[split] - seed_count
        balanced_count = min(
            desired_source_count,
            *(len(source_pools[split][route_id]) for route_id in ROUTE_IDS),
        )
        rows: list[Candidate] = []
        for route_id in ROUTE_IDS:
            ordered = sorted(
                source_pools[split][route_id],
                key=lambda row: _stable_hash(
                    seed,
                    split,
                    route_id,
                    row.normalized_prompt,
                ),
            )
            rows.extend(ordered[:balanced_count])
            rows.extend(seed_pools[split][route_id])
        rows.sort(
            key=lambda row: _stable_hash(seed, "output", split, row.route_id, row.normalized_prompt)
        )
        selected[split] = rows
        report["selected"][split] = {
            route_id: balanced_count + seed_count for route_id in ROUTE_IDS
        }
    return selected, report


def _write_jsonl(path: Path, rows: Iterable[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row.output_row(), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def _source_specs(data_root: Path) -> tuple[SourceSpec, ...]:
    return (
        SourceSpec(
            "embedded-knowledge",
            "embedded-knowledge-v1",
            data_root / "embedded-knowledge-v1.jsonl",
            data_root / "embedded-knowledge-v1.train.provenance.jsonl",
            ("source_group", "source_id", "source_content_sha256"),
        ),
        SourceSpec(
            "ocb-can",
            "ocb-can-verified-v1",
            data_root / "ocb-can-verified.train.jsonl",
            data_root / "ocb-can-verified.train.provenance.jsonl",
            ("message_group", "signal_group", "source_record_id"),
        ),
        SourceSpec(
            "software-architecture",
            "software-architecture-v1",
            data_root / "software-architecture-v1.jsonl",
            data_root / "software-architecture-v1.provenance.jsonl",
            ("source_id", "prompt_sha256"),
        ),
        # These unrelated, currently local corpora teach the router that a specialist is optional.
        SourceSpec(
            "base-financial-analysis",
            "base/null",
            data_root / "financial-analysis-verified.jsonl",
            data_root / "financial-analysis-verified.provenance.jsonl",
            ("group_fingerprint", "source_id"),
        ),
        SourceSpec(
            "base-terminal-cli",
            "base/null",
            data_root / "terminal-cli.jsonl",
            data_root / "terminal-cli.provenance.jsonl",
            ("task_sha256", "source_id"),
        ),
        SourceSpec(
            "base-customer-gtm",
            "base/null",
            data_root / "customer-gtm.jsonl",
            data_root / "customer-gtm.provenance.jsonl",
            ("source_id",),
        ),
    )


def build_router_dataset(
    *,
    data_root: Path,
    output_prefix: Path,
    per_route: int = 3_000,
    seed: int = 83,
) -> dict[str, Any]:
    if per_route < 4:
        raise ValueError("per_route must be at least 4")
    specs = _source_specs(data_root.resolve())
    missing = [path for spec in specs for path in (spec.dataset, spec.provenance) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"router source artifacts are missing: {missing!r}")
    candidates: list[Candidate] = []
    input_counts: dict[str, int] = {}
    source_artifacts: list[dict[str, Any]] = []
    for spec in specs:
        rows = _load_source(spec)
        candidates.extend(rows)
        input_counts[spec.name] = len(rows)
        source_artifacts.append(
            {
                "name": spec.name,
                "route_id": spec.route_id,
                "dataset": str(spec.dataset.resolve()),
                "dataset_sha256": _sha256_file(spec.dataset),
                "provenance": str(spec.provenance.resolve()),
                "provenance_sha256": _sha256_file(spec.provenance),
                "rows": len(rows),
                "group_fields": list(spec.group_fields),
            }
        )
    manual_candidates = _manual_seed_candidates()
    candidates.extend(manual_candidates)
    input_counts["manual-intent-seed-v1"] = len(manual_candidates)
    deduplicated, dedup_report = _deduplicate(candidates)
    selected, balance_report = _select_balanced(
        deduplicated,
        per_route=per_route,
        seed=seed,
    )
    outputs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = output_prefix.with_name(f"{output_prefix.name}.{split}.jsonl").resolve()
        _write_jsonl(path, selected[split])
        outputs[split] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "rows": len(selected[split]),
        }

    prompt_sets = {
        split: {row.normalized_prompt for row in rows} for split, rows in selected.items()
    }
    group_sets = {split: {row.group_key for row in rows} for split, rows in selected.items()}
    prompt_overlap = {
        f"{left}/{right}": len(prompt_sets[left].intersection(prompt_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    group_overlap = {
        f"{left}/{right}": len(group_sets[left].intersection(group_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(prompt_overlap.values()) or any(group_overlap.values()):
        raise AssertionError("router split construction leaked prompts or groups")
    report = {
        "schema_version": "molor.router-bootstrap-dataset.v1",
        "route_ids": list(ROUTE_IDS),
        "seed": seed,
        "requested_examples_per_route": per_route,
        "split_ratios": SPLIT_RATIOS,
        "prompt_extraction": "latest user message only",
        "system_messages_excluded": True,
        "assistant_answers_excluded": True,
        "existing_validation_files_used": False,
        "input_counts": input_counts,
        "manual_seed_curriculum": {
            "version": "manual-intent-seed-v1",
            "rows": len(manual_candidates),
            "families": {
                route_id: {
                    split: list(_MANUAL_SEED_FAMILIES[route_id][split])
                    for split in SPLITS
                }
                for route_id in ROUTE_IDS
            },
            "selected_counts": balance_report["manual_seed_counts"],
        },
        "source_artifacts": source_artifacts,
        "deduplication": dedup_report,
        "balance": balance_report,
        "normalized_prompt_overlap": prompt_overlap,
        "group_overlap": group_overlap,
        "outputs": outputs,
    }
    report_path = output_prefix.with_name(f"{output_prefix.name}.report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    experiment_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build balanced user-prompt-only source labels for the initial auto-router."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=experiment_root / "data",
        help="first-real-experts training-data directory",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=experiment_root / "data" / "router-bootstrap-v1",
        help="prefix used for .train/.validation/.test JSONL and .report.json",
    )
    parser.add_argument("--per-route", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    report = build_router_dataset(
        data_root=args.data_root,
        output_prefix=args.output_prefix,
        per_route=args.per_route,
        seed=args.seed,
    )
    print(json.dumps({"outputs": report["outputs"], "report": report["report_path"]}, indent=2))


if __name__ == "__main__":
    main()
