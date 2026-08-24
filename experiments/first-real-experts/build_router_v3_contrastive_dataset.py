"""Build a balanced five-route router corpus with focused contrastive intent families.

This builder does not read benchmark or intent-probe artifacts.  It combines the original
group-disjoint v1 source partitions, regenerated verified-code prompts, and new manually authored
semantic families.  Each contrastive family belongs to exactly one split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from build_router_v2_dataset import (
    CODE_ROUTE_ID,
    EXPERIMENT_ROOT,
    ROUTE_IDS,
    SPLITS,
    Candidate,
    _assert_unique_prompts,
    _load_v1_contract,
    _load_v1_rows,
    _normalize_prompt,
    _report_path,
    _sha256_file,
    _split_path,
    _stable_hash,
    _verified_code_rows,
    _write_jsonl,
)


DATA_ROOT = EXPERIMENT_ROOT / "data"
EMBEDDED_ROUTE_ID = "embedded-knowledge-v1"
ARCHITECTURE_ROUTE_ID = "software-architecture-v1"
BASE_ROUTE_ID = "base/null"
CAN_ROUTE_ID = "ocb-can-verified-v1"


@dataclass(frozen=True, slots=True)
class FamilySpec:
    route_id: str
    split: str
    family: str
    topics: tuple[str, ...]


CODE_TEMPLATES = (
    (
        "Diagnose {topic}. Return only a corrected, self-contained translation unit with all "
        "required declarations and headers."
    ),
    (
        "Repair a source file affected by {topic}. Produce the complete replacement source, not "
        "a patch, explanation, or partial snippet."
    ),
    (
        "Resolve this compiler or runtime correctness problem: {topic}. Output warning-clean "
        "compilable code that preserves the requested API."
    ),
)

EMBEDDED_TEMPLATES = (
    (
        "Explain {topic} at the microcontroller hardware and register-semantics level. State what "
        "manual evidence would confirm the diagnosis; do not write an implementation."
    ),
    (
        "Troubleshoot {topic} in deployed firmware. Describe likely causes, observations, and the "
        "relevant datasheet concepts rather than generating source code."
    ),
    (
        "Which embedded reference-manual facts govern {topic}, and how should an engineer reason "
        "about them before changing firmware?"
    ),
)

BASE_TEMPLATES = (
    "Answer a general-interest question about {topic} in clear everyday language.",
    "Give practical, nontechnical guidance about {topic} for a beginner.",
    "Explain the main ideas behind {topic} without introducing software or engineering advice.",
)

ARCHITECTURE_TEMPLATES = (
    (
        "Design a production software architecture for {topic}. Cover service boundaries, failure "
        "handling, data ownership, and tradeoffs; this is not a firmware or code-generation task."
    ),
    (
        "Compare viable distributed-system designs for {topic}, including rollout, observability, "
        "and operational consequences."
    ),
    (
        "Explain the application and data-platform architecture choices for {topic}. Focus on "
        "deployed services and storage systems rather than device-level implementation."
    ),
)

TEMPLATES = {
    CODE_ROUTE_ID: CODE_TEMPLATES,
    EMBEDDED_ROUTE_ID: EMBEDDED_TEMPLATES,
    BASE_ROUTE_ID: BASE_TEMPLATES,
    ARCHITECTURE_ROUTE_ID: ARCHITECTURE_TEMPLATES,
}


FAMILIES = (
    # Verified code: explicit diagnose/fix/complete-source intent.
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "declaration-and-header-repair",
        (
            "an implicit memcpy declaration caused by an omitted standard header in C11",
            "unknown bool and false tokens in a freestanding C translation unit",
            "an incomplete structure type used before its full declaration",
            "std::optional used without the corresponding C++20 standard header",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "linkage-and-symbol-repair",
        (
            "multiple external definitions emitted by a variable definition in a header",
            "a static declaration that conflicts with an earlier external declaration",
            "duplicate function bodies linked from two C translation units",
            "an unresolved C++ symbol caused by a namespace mismatch",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "warning-clean-conversion-repair",
        (
            "a shift count that can equal the promoted operand width",
            "an unsigned comparison against zero that is always false",
            "an enum value passed through an unsafe implicit narrowing conversion",
            "a printf format specifier that disagrees with the fixed-width argument type",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "cpp-type-and-overload-repair",
        (
            "ambiguous overload resolution between integral and iterator forms",
            "a member function called through a const object without a const qualifier",
            "template argument deduction failing for two different numeric types",
            "an iterator compared with a value instead of the matching end iterator",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "undefined-behavior-debug",
        (
            "an array index reaching one element beyond a fixed buffer",
            "a signed left shift whose mathematical result is not representable",
            "a pointer that escapes the lifetime of an automatic object",
            "a nullable output pointer dereferenced before validation",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "api-contract-preserving-repair",
        (
            "a failed parse operation that modifies the caller output before returning false",
            "a length check performed only after the first input byte is accessed",
            "a clamp routine that overflows in the narrow type before comparing limits",
            "a circular-buffer counter that eventually wraps independently of capacity",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "preprocessor-and-build-repair",
        (
            "an unterminated conditional-compilation branch around a public declaration",
            "different structure layouts selected in producer and consumer translation units",
            "a macro argument evaluated twice when it contains an increment expression",
            "a C header missing the linkage guard required by a C++ caller",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "train",
        "qualifier-and-memory-access-repair",
        (
            "a function that discards const from an input buffer",
            "a device polling loop whose register object is not volatile",
            "two restrict-qualified pointers that are used with overlapping storage",
            "an atomic object accessed through a non-atomic alias",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "validation",
        "ownership-and-lifetime-repair",
        (
            "a string_view returned from a temporary std::string",
            "a function returning the address of a local array",
            "two cleanup paths that can free the same allocation",
            "a span retained after the backing vector reallocates",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "validation",
        "compile-time-layout-repair",
        (
            "a failing static assertion for the size of a serialized header",
            "alignas applied with a weaker alignment than the object type requires",
            "offsetof used on a non-standard-layout C++ class",
            "a packed wire structure that produces unaligned member access",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "test",
        "standard-library-contract-repair",
        (
            "a vector iterator used after erase invalidates it",
            "an empty optional accessed without checking for a value",
            "an array subscript derived from size instead of size minus one",
            "a sort comparator that is not a strict weak ordering",
        ),
    ),
    FamilySpec(
        CODE_ROUTE_ID,
        "test",
        "concurrency-source-repair",
        (
            "a std::atomic initialized by copying another atomic object",
            "a mutex-containing type copied by a generated special member function",
            "an invalid failure memory order supplied to compare_exchange",
            "a worker lambda capturing a short-lived local variable by reference",
        ),
    ),
    # Embedded reference/explanation: hardware evidence and troubleshooting, no implementation.
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "reset-and-clock-tree",
        (
            "a peripheral remaining in reset after its bus clock is enabled",
            "a PLL that locks but feeds the wrong downstream clock divider",
            "a reset-cause register whose flags are cleared on read",
            "clock-domain startup ordering after a brownout reset",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "interrupt-controller-semantics",
        (
            "a pending interrupt flag that immediately retriggers after ISR return",
            "priority masking that blocks a supposedly higher-priority source",
            "an edge-triggered source lost while the interrupt line is disabled",
            "vector-table relocation during a bootloader-to-application handoff",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "dma-and-cache-coherency",
        (
            "a DMA engine reading an older cache line than the CPU recently modified",
            "a receive buffer whose ownership changes before cache invalidation",
            "a descriptor ring crossing a cache-line boundary",
            "memory ordering between descriptor publication and a DMA doorbell write",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "boot-and-flash-behavior",
        (
            "an interrupted flash erase leaving metadata inconsistent with the image",
            "a boot ROM choosing between two application slots after an incomplete update",
            "flash wait-state requirements changing with core frequency",
            "an option-byte configuration taking effect only after a reset sequence",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "analog-peripheral-diagnosis",
        (
            "an ADC sample capacitor failing to settle with a high-impedance source",
            "a comparator input behaving unpredictably near its common-mode limit",
            "reference-voltage noise appearing as code-dependent ADC error",
            "a DAC output buffer saturating before the documented rail",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "linker-and-runtime-startup",
        (
            "initialized data containing zeros because startup copied the wrong load address",
            "a retained crash record accidentally cleared with the BSS region",
            "an interrupt table removed by section garbage collection",
            "a stack and heap collision visible in the linker map",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "serial-bus-electrical-timing",
        (
            "an SPI peripheral sampling on the wrong clock edge",
            "I2C rise time exceeding the timing budget with several devices attached",
            "a UART receiver framing error caused by accumulated clock mismatch",
            "chip-select timing that violates a converter acquisition interval",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "train",
        "rtos-and-isr-timing",
        (
            "an ISR calling an RTOS API above the permitted interrupt priority",
            "a high-priority task starved by a medium-priority task holding no mutex",
            "tickless idle producing a late deadline after clock-source switching",
            "priority inheritance changing the observed blocking chain",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "validation",
        "timer-capture-reference",
        (
            "an input-capture timestamp wrapping between two signal edges",
            "a prescaler choice that trades measurement resolution for maximum interval",
            "a timer shadow register updating one cycle later than expected",
            "capture overflow and edge flags arriving in the same service interval",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "validation",
        "watchdog-and-fault-analysis",
        (
            "a watchdog reset occurring despite a visible periodic service call",
            "a fault-status register that must be decoded before clearing escalation state",
            "windowed-watchdog service performed too early rather than too late",
            "a lockup-reset path that bypasses normal crash logging",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "test",
        "hardware-debug-and-trace",
        (
            "a hardware breakpoint changing timing enough to hide the failure",
            "trace packets dropping when the debug output clock is undersized",
            "a watchpoint firing on debugger reads as well as CPU accesses",
            "instruction trace disagreeing with source after compiler optimization",
        ),
    ),
    FamilySpec(
        EMBEDDED_ROUTE_ID,
        "test",
        "low-power-wakeup-reference",
        (
            "a wake source asserted while its peripheral clock is gated",
            "retention RAM contents differing across two sleep depths",
            "an oscillator stabilization delay after deep-sleep exit",
            "a wake flag that must be cleared in a documented sequence",
        ),
    ),
    # Base: deliberately ordinary, nontechnical traffic.
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "gardening-and-seasons",
        (
            "choosing shade-tolerant flowers for a small yard",
            "knowing when common herbs are ready to harvest",
            "protecting young plants during an unexpected frost",
            "rotating garden crops between growing seasons",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "language-and-word-choice",
        (
            "choosing between fewer and less in ordinary writing",
            "making a paragraph transition sound more natural",
            "recognizing the tone difference between two greetings",
            "remembering new vocabulary through example sentences",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "literature-and-reading",
        (
            "identifying an unreliable narrator in a short story",
            "discussing how setting influences a novel's mood",
            "choosing a readable biography for a book club",
            "distinguishing a theme from a plot summary",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "visual-arts-appreciation",
        (
            "how warm and cool colors change the feeling of a painting",
            "what viewers can notice in a landscape composition",
            "planning a first visit to a local art exhibition",
            "the difference between a sketch and a finished study",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "personal-organization",
        (
            "sorting a crowded closet into manageable categories",
            "building a morning routine that is easy to maintain",
            "remembering recurring household errands",
            "breaking a long reading list into weekly goals",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "wildlife-and-nature",
        (
            "why some birds migrate while others remain year-round",
            "how tide pools support different kinds of animals",
            "ways to observe wildlife without disturbing it",
            "why deciduous trees shed their leaves",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "recreation-and-games",
        (
            "teaching a simple card game to first-time players",
            "organizing a casual neighborhood scavenger hunt",
            "choosing a cooperative tabletop game for a group",
            "keeping score fairly in a round-robin activity",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "train",
        "music-listening",
        (
            "hearing the difference between melody and harmony",
            "building a calm playlist for quiet reading",
            "recognizing a repeated musical theme",
            "describing rhythm without specialized vocabulary",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "validation",
        "clothing-and-fabric-care",
        (
            "preventing a wool sweater from losing its shape",
            "sorting laundry when care labels differ",
            "removing wrinkles without damaging delicate fabric",
            "storing seasonal clothing in a small space",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "validation",
        "community-etiquette",
        (
            "declining an invitation without sounding dismissive",
            "introducing two people at a social gathering",
            "asking a neighbor to reduce late-night noise politely",
            "thanking a volunteer group after an event",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "test",
        "storytelling-and-characters",
        (
            "giving a fictional character a believable motivation",
            "creating tension in a scene without physical danger",
            "choosing a memorable name for an imaginary village",
            "ending a short tale with a satisfying callback",
        ),
    ),
    FamilySpec(
        BASE_ROUTE_ID,
        "test",
        "geography-and-culture",
        (
            "how rivers influence where towns develop",
            "why regional festivals preserve local traditions",
            "comparing coastal and inland daily life",
            "how a landscape can shape traditional foods",
        ),
    ),
    # Software architecture: explicit service/data-platform counterexamples to embedded traffic.
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "distributed-consistency",
        (
            "inventory reservations across regions during a network partition",
            "read-your-writes behavior in a globally replicated application",
            "conflicting profile edits accepted by two active regions",
            "an authoritative ledger with eventually consistent projections",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "database-evolution",
        (
            "adding a required field while two application versions overlap",
            "backfilling a large table without exhausting replica capacity",
            "changing an identifier format while preserving old API clients",
            "retiring an index after a staged query migration",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "asynchronous-messaging",
        (
            "duplicate and out-of-order events delivered to a billing service",
            "poison messages repeatedly failing in a shared consumer group",
            "publishing an event atomically with a database transaction",
            "replaying an event log after projection logic changes",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "multitenancy-and-isolation",
        (
            "preventing one tenant's heavy query from degrading every customer",
            "choosing shared-schema versus database-per-tenant storage",
            "propagating tenant identity through asynchronous jobs",
            "moving a large tenant to dedicated infrastructure",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "release-and-migration-strategy",
        (
            "a canary release that requires rapid data-compatible rollback",
            "feature flags whose combinations create untested states",
            "a blue-green deployment with long-running background jobs",
            "progressively replacing a monolith endpoint with a service",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "resilience-and-capacity",
        (
            "retry amplification after a slow downstream dependency",
            "load shedding before worker queues exhaust memory",
            "bulkheads for several third-party provider integrations",
            "capacity planning for a sharply seasonal online service",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "security-boundaries",
        (
            "authorizing service-to-service calls in a zero-trust environment",
            "rotating application secrets without coordinated downtime",
            "isolating untrusted document processing from customer data",
            "auditing administrative actions across many services",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "train",
        "api-versioning-and-boundaries",
        (
            "evolving a public API used by independently deployed clients",
            "placing validation between an API gateway and domain services",
            "aggregating several backend calls without coupling their releases",
            "defining ownership of a shared customer concept across domains",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "validation",
        "observability-platform",
        (
            "correlating one request across queues and background workers",
            "controlling high-cardinality labels in a metrics platform",
            "defining service-level indicators for an asynchronous workflow",
            "sampling traces while retaining rare failure evidence",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "validation",
        "cost-aware-scaling",
        (
            "scaling a bursty service without permanent idle capacity",
            "tiering old objects while preserving occasional low-latency reads",
            "reducing cross-region transfer without losing recovery options",
            "assigning infrastructure cost to individual product workloads",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "test",
        "disaster-recovery",
        (
            "meeting recovery objectives after loss of an entire cloud region",
            "testing backups without exposing production credentials",
            "failing over a stateful service while avoiding split brain",
            "restoring dependent data stores to one consistent point in time",
        ),
    ),
    FamilySpec(
        ARCHITECTURE_ROUTE_ID,
        "test",
        "data-governance-platform",
        (
            "tracking lineage across batch and streaming transformations",
            "enforcing retention rules across analytical data products",
            "separating personally identifiable fields from broad analytics access",
            "evolving a shared event schema used by many teams",
        ),
    ),
)


def _contrastive_rows() -> dict[str, list[Candidate]]:
    results = {split: [] for split in SPLITS}
    family_keys: set[str] = set()
    for spec in FAMILIES:
        if spec.route_id not in TEMPLATES:
            raise ValueError(f"contrastive family has no templates: {spec.route_id}")
        if spec.split not in SPLITS:
            raise ValueError(f"contrastive family has invalid split: {spec.split}")
        key = f"contrastive-v3/{spec.route_id}/{spec.family}"
        if key in family_keys:
            raise ValueError(f"duplicate contrastive semantic family: {key}")
        family_keys.add(key)
        for topic in spec.topics:
            for template in TEMPLATES[spec.route_id]:
                prompt = template.format(topic=topic)
                results[spec.split].append(
                    Candidate(
                        prompt=prompt,
                        normalized_prompt=_normalize_prompt(prompt),
                        route_id=spec.route_id,
                        split=spec.split,
                        semantic_group=key,
                        source="manual-contrastive-intent-v3",
                        family=spec.family,
                    )
                )
    return results


def _select_balanced_v3(
    v1_rows: dict[str, list[Candidate]],
    code_rows: dict[str, list[Candidate]],
    contrastive_rows: dict[str, list[Candidate]],
    *,
    seed: int,
) -> tuple[dict[str, list[Candidate]], dict[str, dict[str, int]]]:
    selected: dict[str, list[Candidate]] = {}
    balance: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        v1_by_route = {
            route_id: [row for row in v1_rows[split] if row.route_id == route_id]
            for route_id in ROUTE_IDS
        }
        contrastive_by_route = {
            route_id: [row for row in contrastive_rows[split] if row.route_id == route_id]
            for route_id in ROUTE_IDS
        }
        mandatory = {
            route_id: list(contrastive_by_route[route_id]) for route_id in ROUTE_IDS
        }
        mandatory[CODE_ROUTE_ID] = [
            *code_rows[split],
            *mandatory[CODE_ROUTE_ID],
        ]
        target = len(mandatory[CODE_ROUTE_ID])
        if target < 1:
            raise ValueError(f"verified-code route has no {split} examples")

        rows: list[Candidate] = []
        for route_id in ROUTE_IDS:
            required = mandatory[route_id]
            needed = target - len(required)
            if needed < 0:
                raise ValueError(
                    f"mandatory {route_id} examples exceed verified-code capacity in {split}"
                )
            fill = sorted(
                v1_by_route[route_id],
                key=lambda row: _stable_hash(
                    seed,
                    split,
                    route_id,
                    row.normalized_prompt,
                ),
            )
            if len(fill) < needed:
                raise ValueError(
                    f"not enough v1 {route_id} examples for balanced {split}: "
                    f"need {needed}, have {len(fill)}"
                )
            rows.extend(required)
            rows.extend(fill[:needed])
        rows.sort(
            key=lambda row: _stable_hash(
                seed,
                "output",
                split,
                row.route_id,
                row.normalized_prompt,
            )
        )
        counts = dict(Counter(row.route_id for row in rows))
        if set(counts) != set(ROUTE_IDS) or set(counts.values()) != {target}:
            raise AssertionError(f"router v3 {split} split is not route-balanced")
        selected[split] = rows
        balance[split] = counts
    return selected, balance


def _overlap_counts(
    selected: dict[str, list[Candidate]],
    attribute: str,
) -> dict[str, int]:
    values = {
        split: {getattr(row, attribute) for row in selected[split]} for split in SPLITS
    }
    return {
        f"{left}/{right}": len(values[left].intersection(values[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }


def build_router_v3_dataset(
    *,
    v1_prefix: Path,
    output_prefix: Path,
    generator_seed: int = 9071,
    code_split_seed: int = 211,
    selection_seed: int = 307,
) -> dict[str, Any]:
    v1_report, v1_paths = _load_v1_contract(v1_prefix.resolve())
    v1_rows = _load_v1_rows(v1_paths)
    code_rows = _verified_code_rows(
        generator_seed=generator_seed,
        split_seed=code_split_seed,
    )
    contrastive_rows = _contrastive_rows()
    all_inputs = [
        row
        for split in SPLITS
        for row in (*v1_rows[split], *code_rows[split], *contrastive_rows[split])
    ]
    _assert_unique_prompts(all_inputs)
    selected, balance = _select_balanced_v3(
        v1_rows,
        code_rows,
        contrastive_rows,
        seed=selection_seed,
    )
    prompt_overlap = _overlap_counts(selected, "normalized_prompt")
    group_overlap = _overlap_counts(selected, "semantic_group")
    if any(prompt_overlap.values()) or any(group_overlap.values()):
        raise AssertionError("router v3 construction leaked prompts or semantic groups")

    outputs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = _split_path(output_prefix, split).resolve()
        rows, digest = _write_jsonl(path, selected[split])
        outputs[split] = {"path": str(path), "rows": rows, "sha256": digest}

    contrastive_counts = {
        split: dict(Counter(row.route_id for row in contrastive_rows[split]))
        for split in SPLITS
    }
    contrastive_families = {
        split: {
            route_id: sorted(
                {
                    row.family
                    for row in contrastive_rows[split]
                    if row.route_id == route_id and row.family is not None
                }
            )
            for route_id in ROUTE_IDS
        }
        for split in SPLITS
    }
    report = {
        "schema_version": "molor.router-bootstrap-dataset.v3",
        "route_ids": list(ROUTE_IDS),
        "evaluation_data_used": False,
        "probe_prompts_used": False,
        "seeds": {
            "verified_code_generator": generator_seed,
            "verified_code_split": code_split_seed,
            "balanced_selection": selection_seed,
        },
        "split_contract": {
            "v1": "original reported semantic-group-disjoint partitions preserved",
            "verified_code": "generated task groups remain in one stable 80/10/10 split",
            "contrastive": "each authored semantic family is assigned to exactly one split",
        },
        "v1_source": {
            "report": str(_report_path(v1_prefix).resolve()),
            "report_sha256": _sha256_file(_report_path(v1_prefix).resolve()),
            "schema_version": v1_report.get("schema_version"),
            "splits": {
                split: {
                    "path": str(v1_paths[split]),
                    "sha256": _sha256_file(v1_paths[split]),
                    "rows": len(v1_rows[split]),
                }
                for split in SPLITS
            },
        },
        "verified_code_source": {
            "generator": str((EXPERIMENT_ROOT / "build_verified_embedded_code.py").resolve()),
            "generator_sha256": _sha256_file(
                EXPERIMENT_ROOT / "build_verified_embedded_code.py"
            ),
            "rows": {split: len(code_rows[split]) for split in SPLITS},
            "semantic_groups": {
                split: len({row.semantic_group for row in code_rows[split]})
                for split in SPLITS
            },
        },
        "contrastive_source": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256_file(Path(__file__).resolve()),
            "shared_builder": str((EXPERIMENT_ROOT / "build_router_v2_dataset.py").resolve()),
            "shared_builder_sha256": _sha256_file(
                EXPERIMENT_ROOT / "build_router_v2_dataset.py"
            ),
            "counts": contrastive_counts,
            "families": contrastive_families,
            "semantic_groups": {
                split: len({row.semantic_group for row in contrastive_rows[split]})
                for split in SPLITS
            },
        },
        "balance": balance,
        "normalized_prompt_overlap": prompt_overlap,
        "semantic_group_overlap": group_overlap,
        "duplicate_normalized_prompts": 0,
        "outputs": outputs,
    }
    report_path = _report_path(output_prefix).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v1",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v3-contrastive",
    )
    parser.add_argument("--generator-seed", type=int, default=9071)
    parser.add_argument("--code-split-seed", type=int, default=211)
    parser.add_argument("--selection-seed", type=int, default=307)
    args = parser.parse_args()
    report = build_router_v3_dataset(
        v1_prefix=args.v1_prefix,
        output_prefix=args.output_prefix,
        generator_seed=args.generator_seed,
        code_split_seed=args.code_split_seed,
        selection_seed=args.selection_seed,
    )
    print(
        json.dumps(
            {
                "route_ids": report["route_ids"],
                "balance": report["balance"],
                "contrastive_counts": report["contrastive_source"]["counts"],
                "outputs": report["outputs"],
                "report": report["report_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
