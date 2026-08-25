"""Build the five-route v4 router corpus with split-owned routing hard negatives.

V4 preserves every source and split contract used by v3, then adds manually authored families for
three distribution gaps identified during development evaluation:

* CAN intents had no authored contrastive families in v3;
* short incident-style embedded questions were underrepresented; and
* ordinary prompts containing route vocabulary needed stronger base/null negatives.

The builder does not import, read, or copy exact rows from either intent probe or any benchmark
artifact. Aggregate probe failure analysis did inform the gap selection, however, so probe scores
after this build are development diagnostics rather than blind promotion evidence. Every authored
family belongs to exactly one split, and the final output remains balanced across the same five
routes as v3.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from build_router_v2_dataset import (
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
    _verified_code_rows,
    _write_jsonl,
)
from build_router_v3_contrastive_dataset import (
    BASE_ROUTE_ID,
    CAN_ROUTE_ID,
    EMBEDDED_ROUTE_ID,
    _contrastive_rows as _v3_contrastive_rows,
    _overlap_counts,
    _select_balanced_v3,
)


DATA_ROOT = EXPERIMENT_ROOT / "data"


@dataclass(frozen=True, slots=True)
class AuthoredFamily:
    """A semantic family whose prompts must remain in one split."""

    route_id: str
    split: str
    family: str
    prompts: tuple[str, ...]


CAN_TEMPLATES = (
    "{intent} Explain the CAN-specific mechanism and the evidence you would inspect.",
    "CAN problem: {intent} Diagnose the bus behavior before suggesting a fix.",
    "{intent} State what happens on the wire and why; keep the answer specific to CAN.",
)


CAN_FAMILIES = (
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "arbitration-and-frame-priority",
        (
            "Two standard data frames start together with identifiers 0x241 and 0x249.",
            "A data frame and a remote request use the same eleven-bit identifier.",
            "Standard and extended frames share the same eleven-bit base identifier.",
            "One transmitter loses arbitration yet reports no protocol error.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "fault-confinement-and-controller-state",
        (
            "A transmitter's error counter crosses the error-passive boundary.",
            "Repeated transmit errors push a controller beyond the bus-off threshold.",
            "A receiver accumulates errors while its transmit counter stays unchanged.",
            "A bus-off node is restarted before the required recovery observation.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "acknowledgement-and-retransmission",
        (
            "A valid frame sees no dominant acknowledgement bit on a one-node bench.",
            "The sender keeps retrying although another node can decode every payload.",
            "A silent-monitoring interface never acknowledges traffic it captures.",
            "The ACK delimiter is dominant after a receiver flags the frame valid.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "acceptance-filters-and-identifiers",
        (
            "A mask should admit standard identifiers 0x620 through 0x62F only.",
            "An extended identifier is accepted by a filter intended for standard frames.",
            "Two filter banks must separate diagnostic requests from broadcast telemetry.",
            "A controller's identifier mask unexpectedly treats one bit as don't-care.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "dbc-bit-layout-and-scaling",
        (
            "A signed Motorola field crosses a byte boundary and decodes with the wrong sign.",
            "An Intel-order field begins halfway through one byte and spans into the next.",
            "A physical DBC value must be converted back to raw units without bias.",
            "A negative raw field needs sign extension before factor and offset are applied.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "multiplexing-and-signal-validity",
        (
            "Overlapping DBC signals are valid under different multiplexor values.",
            "A decoder publishes branch signals even when their multiplexor is inactive.",
            "One frame uses an extended multiplexor with nested signal groups.",
            "A capture changes multiplexor value while stale branch values remain visible.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "can-fd-length-and-data-phase",
        (
            "A CAN FD frame carries forty-eight bytes although its DLC field is fourteen.",
            "Bit-rate switching is enabled but the data-phase sample point is unsuitable.",
            "A classic-CAN receiver encounters an FD frame on the same physical bus.",
            "The error-state indicator is recessive in a frame from one controller.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "train",
        "capture-diagnostics-and-bus-load",
        (
            "A trace shows error frames clustered after the network reaches high utilization.",
            "Two interfaces timestamp the same CAN frame with a growing offset.",
            "A periodic payload appears missing only when several high-priority IDs are active.",
            "An analyzer reports retransmissions as distinct application samples.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "validation",
        "physical-layer-and-termination",
        (
            "An unpowered network measures about sixty ohms between CAN high and CAN low.",
            "A long unterminated stub creates ringing near the receiver threshold.",
            "Common-mode voltage shifts when a distant node joins the network.",
            "Communication works at low bitrate but fails after the bitrate is increased.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "validation",
        "higher-layer-addressing-and-transport",
        (
            "A twenty-nine-bit vehicle-network identifier contains priority and address fields.",
            "A diagnostic payload exceeds one classical CAN frame and needs segmentation.",
            "A transport receiver observes a sequence-number gap in a multi-frame transfer.",
            "Two nodes claim the same network address during startup.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "test",
        "bit-timing-and-synchronization",
        (
            "A receiver samples too early after a long run of identical bits.",
            "Two nodes use the same nominal bitrate but incompatible sample points.",
            "Resynchronization repeatedly consumes the configured phase-segment margin.",
            "Oscillator tolerance becomes marginal only on the longest network branch.",
        ),
    ),
    AuthoredFamily(
        CAN_ROUTE_ID,
        "test",
        "gateway-integrity-and-freshness",
        (
            "A gateway forwards the same valid frame twice after reconnecting.",
            "An alive counter jumps while the payload checksum still verifies.",
            "A translated signal saturates because source and destination ranges differ.",
            "A delayed command frame arrives with a valid CRC but an obsolete counter.",
        ),
    ),
)


EMBEDDED_FAMILIES = (
    AuthoredFamily(
        EMBEDDED_ROUTE_ID,
        "train",
        "terse-cache-and-dma-incidents",
        (
            "DMA reads yesterday's bytes. Cache is enabled. What do I inspect first?",
            "RX descriptors look updated, but the CPU sees old payloads. Hardware-level triage?",
            "CPU writes vanish when the peripheral owns the buffer. Explain the likely boundary.",
            "A descriptor works when uncached and fails when cached. Give me the shortest diagnosis.",
        ),
    ),
    AuthoredFamily(
        EMBEDDED_ROUTE_ID,
        "train",
        "terse-reset-and-clock-incidents",
        (
            "Peripheral register reads are zero after wake. Clock or reset: how do I tell?",
            "The PLL says locked, yet the timer runs slow. Name the registers to verify.",
            "Boots under the debugger, hangs on cold power-up. Start the hardware triage.",
            "Brownout reboot loops before main. Which reset evidence matters?",
        ),
    ),
    AuthoredFamily(
        EMBEDDED_ROUTE_ID,
        "train",
        "terse-interrupt-and-rtos-incidents",
        (
            "ISR fires once, then never again. What flag semantics should I check?",
            "Higher-priority work is blocked by an RTOS call from an interrupt. Diagnose it.",
            "The pending bit returns immediately after ISR exit. Explain the hardware cause.",
            "Tickless idle misses one deadline after wake. Which clock transition is suspect?",
        ),
    ),
    AuthoredFamily(
        EMBEDDED_ROUTE_ID,
        "train",
        "terse-peripheral-electrical-incidents",
        (
            "ADC values drift when source impedance rises. Give the peripheral explanation.",
            "SPI works at 1 MHz and fails at 12 MHz. What timing evidence do I collect?",
            "UART framing errors grow with cable length. Start with the electrical checks.",
            "I2C edges are rounded and ACKs disappear. Diagnose without writing code.",
        ),
    ),
    AuthoredFamily(
        EMBEDDED_ROUTE_ID,
        "validation",
        "terse-memory-and-startup-incidents",
        (
            "Initialized globals are garbage before main. Where does startup usually go wrong?",
            "A crash record disappears on reset. Which startup section handling should I audit?",
            "The vector table moved, then every interrupt faults. What hardware state changed?",
            "Heap and stack collide only in release builds. What map evidence should I inspect?",
        ),
    ),
    AuthoredFamily(
        EMBEDDED_ROUTE_ID,
        "test",
        "terse-low-power-and-debug-incidents",
        (
            "Wake pin toggles, CPU stays asleep. Which retained flags and clocks matter?",
            "A breakpoint makes the bug disappear. Give me nonintrusive hardware evidence.",
            "Retention RAM survives light sleep but not deep sleep. What should the manual say?",
            "Trace output drops only at full clock. Which debug bandwidth limit is likely?",
        ),
    ),
)


BASE_FAMILIES = (
    AuthoredFamily(
        BASE_ROUTE_ID,
        "train",
        "ordinary-can-language",
        (
            "Can you suggest a quick lunch from rice, beans, and spinach?",
            "Can this wool sweater be washed by hand? Answer briefly.",
            "I found a dented can in the pantry. Should I use it?",
            "Can my neighbor's dog safely eat a small piece of apple?",
        ),
    ),
    AuthoredFamily(
        BASE_ROUTE_ID,
        "train",
        "buildings-not-software-architecture",
        (
            "Name the architectural style of a house with a central open courtyard.",
            "Compare Gothic and Romanesque arches in plain language.",
            "Why do traditional hot-climate buildings use thick walls?",
            "Give me three things to notice in the architecture of an old train station.",
        ),
    ),
    AuthoredFamily(
        BASE_ROUTE_ID,
        "train",
        "routes-not-model-routing",
        (
            "Route a two-hour walking loop through a park with minimal hills.",
            "Which bus route is simpler if I want to avoid two transfers?",
            "Plan a scenic driving route that returns before sunset.",
            "Explain how postal carriers organize a neighborhood route.",
        ),
    ),
    AuthoredFamily(
        BASE_ROUTE_ID,
        "train",
        "adapters-and-experts-in-everyday-context",
        (
            "Which plug adapter do I need for a trip from Canada to Portugal?",
            "Explain what an expert witness does to a middle-school student.",
            "My camera lens adapter is loose. What basic fit checks are safe?",
            "Suggest questions to ask an expert before restoring an antique chair.",
        ),
    ),
    AuthoredFamily(
        BASE_ROUTE_ID,
        "validation",
        "codes-and-embedded-words-outside-programming",
        (
            "What does a formal dress code usually mean for an afternoon wedding?",
            "The article has an embedded map. Summarize why maps help travel writing.",
            "Help me remember my building's entry code without writing it down publicly.",
            "A poem embeds one quotation inside another. Explain the effect.",
        ),
    ),
    AuthoredFamily(
        BASE_ROUTE_ID,
        "test",
        "routers-caches-and-terminals-outside-expert-scope",
        (
            "My home Wi-Fi router is in a cabinet. Where should I place it instead?",
            "Why did old railway terminals often become city landmarks?",
            "What is a geocache, and how does a beginner look for one responsibly?",
            "The woodworking router leaves burn marks. List simple setup checks.",
        ),
    ),
)


def _expand_can_families() -> tuple[AuthoredFamily, ...]:
    expanded: list[AuthoredFamily] = []
    for family in CAN_FAMILIES:
        prompts = tuple(
            template.format(intent=intent)
            for intent in family.prompts
            for template in CAN_TEMPLATES
        )
        expanded.append(
            AuthoredFamily(
                route_id=family.route_id,
                split=family.split,
                family=family.family,
                prompts=prompts,
            )
        )
    return tuple(expanded)


def _v4_authored_rows() -> dict[str, list[Candidate]]:
    """Materialize v4-only rows without consulting evaluation artifacts."""

    rows = {split: [] for split in SPLITS}
    families = (*_expand_can_families(), *EMBEDDED_FAMILIES, *BASE_FAMILIES)
    family_owner: dict[tuple[str, str], str] = {}
    for spec in families:
        if spec.route_id not in ROUTE_IDS:
            raise AssertionError(f"unknown v4 route: {spec.route_id}")
        if spec.split not in SPLITS:
            raise AssertionError(f"unknown v4 split: {spec.split}")
        owner_key = (spec.route_id, spec.family)
        previous = family_owner.setdefault(owner_key, spec.split)
        if previous != spec.split:
            raise AssertionError(f"v4 family crosses splits: {owner_key}")
        if not spec.prompts:
            raise AssertionError(f"v4 family has no prompts: {owner_key}")
        semantic_group = f"manual-v4/{spec.route_id}/{spec.family}"
        for prompt in spec.prompts:
            normalized = _normalize_prompt(prompt)
            rows[spec.split].append(
                Candidate(
                    prompt=prompt,
                    normalized_prompt=normalized,
                    route_id=spec.route_id,
                    split=spec.split,
                    semantic_group=semantic_group,
                    source="manual-hard-negative-intent-v4",
                    family=spec.family,
                )
            )
    _assert_unique_prompts(row for split in SPLITS for row in rows[split])
    return rows


def _family_inventory(rows: dict[str, list[Candidate]]) -> dict[str, dict[str, list[str]]]:
    return {
        split: {
            route_id: sorted(
                {
                    row.family
                    for row in rows[split]
                    if row.route_id == route_id and row.family is not None
                }
            )
            for route_id in ROUTE_IDS
        }
        for split in SPLITS
    }


def _route_counts(rows: dict[str, list[Candidate]]) -> dict[str, dict[str, int]]:
    return {
        split: dict(Counter(row.route_id for row in rows[split]))
        for split in SPLITS
    }


def build_router_v4_dataset(
    *,
    v1_prefix: Path,
    output_prefix: Path,
    generator_seed: int = 9071,
    code_split_seed: int = 211,
    selection_seed: int = 307,
) -> dict[str, Any]:
    """Build balanced v4 splits while retaining the complete v3 source contract."""

    v1_report, v1_paths = _load_v1_contract(v1_prefix.resolve())
    v1_rows = _load_v1_rows(v1_paths)
    code_rows = _verified_code_rows(
        generator_seed=generator_seed,
        split_seed=code_split_seed,
    )
    v3_rows = _v3_contrastive_rows()
    v4_rows = _v4_authored_rows()
    authored_rows = {
        split: [*v3_rows[split], *v4_rows[split]] for split in SPLITS
    }
    all_inputs = [
        row
        for split in SPLITS
        for row in (*v1_rows[split], *code_rows[split], *authored_rows[split])
    ]
    _assert_unique_prompts(all_inputs)
    selected, balance = _select_balanced_v3(
        v1_rows,
        code_rows,
        authored_rows,
        seed=selection_seed,
    )
    prompt_overlap = _overlap_counts(selected, "normalized_prompt")
    group_overlap = _overlap_counts(selected, "semantic_group")
    if any(prompt_overlap.values()) or any(group_overlap.values()):
        raise AssertionError("router v4 construction leaked prompts or semantic groups")

    outputs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = _split_path(output_prefix, split).resolve()
        count, digest = _write_jsonl(path, selected[split])
        outputs[split] = {"path": str(path), "rows": count, "sha256": digest}

    builder_path = Path(__file__).resolve()
    v3_builder = EXPERIMENT_ROOT / "build_router_v3_contrastive_dataset.py"
    v2_builder = EXPERIMENT_ROOT / "build_router_v2_dataset.py"
    report = {
        "schema_version": "molor.router-bootstrap-dataset.v4",
        "route_ids": list(ROUTE_IDS),
        "evaluation_artifacts_read_by_builder": False,
        "exact_probe_rows_used": False,
        "probe_gap_analysis_informed_curriculum": True,
        "probe_status_for_v4": "development_only_not_blind_promotion_evidence",
        "seeds": {
            "verified_code_generator": generator_seed,
            "verified_code_split": code_split_seed,
            "balanced_selection": selection_seed,
        },
        "split_contract": {
            "v1": "original reported semantic-group-disjoint partitions preserved",
            "verified_code": "generated task groups remain in one stable 80/10/10 split",
            "v3_contrastive": "the complete authored v3 family inventory is retained",
            "v4_authored": "each new hard-negative family belongs to exactly one split",
        },
        "source_identity": {
            "builder": str(builder_path),
            "builder_sha256": _sha256_file(builder_path),
            "v2_builder": str(v2_builder.resolve()),
            "v2_builder_sha256": _sha256_file(v2_builder),
            "v3_builder": str(v3_builder.resolve()),
            "v3_builder_sha256": _sha256_file(v3_builder),
            "v1_report": str(_report_path(v1_prefix).resolve()),
            "v1_report_sha256": _sha256_file(_report_path(v1_prefix).resolve()),
            "v1_schema_version": v1_report.get("schema_version"),
        },
        "v1_outputs": {
            split: {
                "path": str(v1_paths[split]),
                "rows": len(v1_rows[split]),
                "sha256": _sha256_file(v1_paths[split]),
            }
            for split in SPLITS
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
        "v3_contrastive_source": {
            "counts": _route_counts(v3_rows),
            "families": _family_inventory(v3_rows),
            "semantic_groups": {
                split: len({row.semantic_group for row in v3_rows[split]})
                for split in SPLITS
            },
        },
        "v4_authored_source": {
            "counts": _route_counts(v4_rows),
            "families": _family_inventory(v4_rows),
            "semantic_groups": {
                split: len({row.semantic_group for row in v4_rows[split]})
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
        default=DATA_ROOT / "router-bootstrap-v4-hard-negative",
    )
    parser.add_argument("--generator-seed", type=int, default=9071)
    parser.add_argument("--code-split-seed", type=int, default=211)
    parser.add_argument("--selection-seed", type=int, default=307)
    args = parser.parse_args()
    report = build_router_v4_dataset(
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
                "v4_authored_counts": report["v4_authored_source"]["counts"],
                "outputs": report["outputs"],
                "report": report["report_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
