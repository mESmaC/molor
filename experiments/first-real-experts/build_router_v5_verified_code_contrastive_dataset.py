"""Build a controlled v5 router corpus for embedded-code output intent.

The promoted v4 router improved embedded-hardware routing, but aggregate blind evaluation showed
that explicit firmware implementation requests remained its weakest single-label category. V5
therefore adds code-intent contrastives that contain embedded vocabulary while unambiguously asking
for complete source code.

This builder never reads an evaluation prompt or benchmark artifact. The aggregate weakness did
inform the curriculum, so the probe that exposed it is development evidence only for v5. Each new
semantic family belongs to exactly one split. To isolate the curriculum change, v5 starts from the
reported promoted-v4 outputs and replaces whole generated verified-code semantic groups in place:
every non-code row, every retained code row, every row position, and every route count otherwise
remain unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    _normalize_prompt,
    _report_path,
    _sha256_file,
    _split_path,
    _stable_hash,
    _verified_code_rows,
    _write_jsonl,
)
from build_router_v3_contrastive_dataset import _overlap_counts


DATA_ROOT = EXPERIMENT_ROOT / "data"


@dataclass(frozen=True, slots=True)
class CodeIntentFamily:
    """A firmware-code intent family owned by exactly one data split."""

    split: str
    family: str
    requirements: tuple[str, ...]


CODE_INTENT_TEMPLATES = (
    (
        "Write the complete, warning-clean C module for {requirement}. Include the public "
        "declarations, initialization, bounded error paths, and implementation; do not stop at "
        "a hardware explanation."
    ),
    (
        "The peripheral behavior is already understood. Produce self-contained compilable C "
        "source that implements {requirement}. Return the full module rather than pseudocode, a "
        "register tutorial, or a partial snippet."
    ),
    (
        "Turn this MCU requirement into production-oriented C code: {requirement}. Supply all "
        "needed types and functions, explicit timeout or fault handling, and a complete "
        "implementation."
    ),
)


CODE_INTENT_FAMILIES = (
    CodeIntentFamily(
        "train",
        "timer-output-and-waveform-modules",
        (
            "a complementary-PWM service that commits synchronized duty updates at period "
            "boundaries and forces safe outputs after a break event",
            "a timer-driven pulse sequencer that executes timestamped edges without busy-waiting "
            "and remains correct across counter wrap",
            "a one-shot output-compare scheduler with cancellation, capacity checks, and "
            "wrap-safe deadline ordering",
            "a multi-channel waveform queue that detects underruns and updates shadow compare "
            "registers atomically",
        ),
    ),
    CodeIntentFamily(
        "train",
        "external-flash-and-qspi-modules",
        (
            "a QSPI NOR command driver that discovers geometry, splits page programs, polls "
            "status with deadlines, and rejects out-of-range operations",
            "a power-fail-safe parameter journal over serial flash with commit markers, record "
            "validation, garbage collection, and bounded wear",
            "a controller that safely enters and leaves memory-mapped flash mode while handling "
            "cache maintenance and in-flight requests",
            "an SPI FRAM storage layer with checked addressing, scatter writes, readback "
            "verification, and transport-error propagation",
        ),
    ),
    CodeIntentFamily(
        "train",
        "usb-and-industrial-serial-modules",
        (
            "a USB bulk-device endpoint queue that handles short transfers, zero-length packets, "
            "cancellation, and disconnect cleanup",
            "an RS-485 half-duplex transport that controls driver enable, drains the transmitter, "
            "detects timeouts, and recovers from framing faults",
            "a LIN schedule client that validates protected identifiers and checksums, advances "
            "slots from a timer, and reports missing responses",
            "a bounded COBS serial framing layer with incremental receive state, malformed-frame "
            "recovery, and caller-owned buffers",
        ),
    ),
    CodeIntentFamily(
        "train",
        "streaming-audio-dma-modules",
        (
            "an I2S ping-pong streaming layer that exchanges buffers from interrupts and reports "
            "overrun or underrun without blocking",
            "a PDM microphone capture path that schedules decimation blocks and publishes only "
            "fully completed sample windows",
            "an audio sample-rate converter with fixed-size work buffers, saturating arithmetic, "
            "and explicit end-of-stream flushing",
            "a dual-channel DAC streamer that swaps prepared blocks at DMA boundaries and enters "
            "a defined idle level after starvation",
        ),
    ),
    CodeIntentFamily(
        "train",
        "power-sequencing-and-fault-control",
        (
            "a nonblocking rail sequencer that observes power-good inputs, enforces per-stage "
            "deadlines, rolls back safely, and records the first fault",
            "a battery-powered load controller with startup qualification, undervoltage lockout, "
            "retry backoff, and a latched terminal state",
            "a fan and thermal-protection controller with hysteresis, sensor-failure handling, "
            "rate limiting, and an emergency override",
            "a supervisor interface that captures reset reasons before clearing them and stores a "
            "checksum-protected boot history",
        ),
    ),
    CodeIntentFamily(
        "train",
        "calibration-and-fixed-point-signal-code",
        (
            "a piecewise-linear calibration table evaluator using checked fixed-point arithmetic, "
            "monotonic breakpoints, and saturating endpoint behavior",
            "a sensor conversion pipeline that applies offset and gain coefficients, rejects "
            "invalid calibration records, and never overflows intermediate values",
            "a streaming window-statistics component with bounded storage, numerically safe "
            "updates, reset semantics, and no dynamic allocation",
            "a fixed-point low-pass filter whose coefficient updates are atomic and whose output "
            "is clamped to the documented signal range",
        ),
    ),
    CodeIntentFamily(
        "train",
        "display-and-human-input-drivers",
        (
            "a scanned key-matrix driver with ghost-key rejection, elapsed-time debouncing, event "
            "queues, and no delay loops",
            "a monochrome display transport that tracks dirty regions, chunks transfers to a "
            "bounded staging buffer, and survives interrupted updates",
            "a capacitive-touch controller interface that drains interrupt-reported contacts, "
            "validates coordinates, and handles device resets",
            "an LED-matrix refresh engine with double-buffered frames, global brightness scaling, "
            "and deterministic blanking during row changes",
        ),
    ),
    CodeIntentFamily(
        "train",
        "portable-register-and-hal-modules",
        (
            "a clock-gate manager with reference counts, critical-section protection, invalid-ID "
            "checks, and a defined reset sequence",
            "a pin-multiplexing API that validates alternate functions and performs masked "
            "read-modify-write operations without disturbing adjacent pins",
            "a memory-mapped peripheral accessor with volatile semantics, field-range checks, and "
            "portable barriers around command publication",
            "a reset-domain controller that sequences assertion and release, waits with a finite "
            "deadline, and returns diagnostic status to its caller",
        ),
    ),
    CodeIntentFamily(
        "validation",
        "network-coprocessor-command-drivers",
        (
            "a framed command client for a radio coprocessor with transaction IDs, bounded "
            "retries, asynchronous notifications, and resynchronization after corrupt bytes",
            "a Wi-Fi module command queue that distinguishes solicited replies from unsolicited "
            "events and cancels pending operations on reset",
            "a packet-based modem interface with incremental parsing, per-command deadlines, and "
            "safe ownership of receive storage",
            "a host driver for a secure element with length-prefixed exchanges, wake timing, "
            "response authentication, and transport-fault recovery",
        ),
    ),
    CodeIntentFamily(
        "validation",
        "safety-monitor-and-diagnostics",
        (
            "a runtime stack-usage monitor that records high-water marks without dynamic "
            "allocation and reports corrupt guard regions",
            "a fault-injection controller that arms one test at a time, enforces authorization "
            "callbacks, and restores safe configuration after a deadline",
            "a redundant-sensor voter that validates freshness, detects disagreement with "
            "hysteresis, and exposes a deterministic degraded mode",
            "a diagnostic event recorder with fixed capacity, monotonic sequence numbers, "
            "checksum-protected entries, and power-loss-aware commits",
        ),
    ),
    CodeIntentFamily(
        "test",
        "single-wire-and-addressed-device-drivers",
        (
            "a single-wire temperature-device bus with reset timing, ROM selection, checked "
            "scratchpad reads, and nonblocking conversion polling",
            "an addressable-light transport that builds bounded frames, applies brightness limits, "
            "and schedules the required reset interval",
            "a digital-potentiometer driver with verified register writes, range checks, and "
            "recovery after a missing device response",
            "a daisy-chained shift-register interface with shadow state, atomic output commits, "
            "and explicit enable-line fault handling",
        ),
    ),
    CodeIntentFamily(
        "test",
        "event-timekeeping-and-alarm-modules",
        (
            "a calendar-alarm service that validates dates, handles leap years, and reschedules "
            "recurring alarms after missed wakeups",
            "a monotonic timestamp extender that combines a wrapping hardware counter with an "
            "overflow epoch under concurrent readers",
            "a bounded software-alarm queue driven by a low-power counter with cancellation and "
            "wrap-safe ordering",
            "a clock-discipline component that slews small corrections, rejects implausible "
            "samples, and never moves monotonic time backward",
        ),
    ),
)


def _v5_authored_rows() -> dict[str, list[Candidate]]:
    """Materialize v5-only rows without consulting evaluation artifacts."""

    rows = {split: [] for split in SPLITS}
    owners: dict[str, str] = {}
    for spec in CODE_INTENT_FAMILIES:
        if spec.split not in SPLITS:
            raise AssertionError(f"unknown v5 split: {spec.split}")
        previous = owners.setdefault(spec.family, spec.split)
        if previous != spec.split:
            raise AssertionError(f"v5 family crosses splits: {spec.family}")
        if not spec.requirements:
            raise AssertionError(f"v5 family has no requirements: {spec.family}")
        semantic_group = f"manual-v5/{CODE_ROUTE_ID}/{spec.family}"
        for requirement in spec.requirements:
            for template in CODE_INTENT_TEMPLATES:
                prompt = template.format(requirement=requirement)
                rows[spec.split].append(
                    Candidate(
                        prompt=prompt,
                        normalized_prompt=_normalize_prompt(prompt),
                        route_id=CODE_ROUTE_ID,
                        split=spec.split,
                        semantic_group=semantic_group,
                        source="manual-embedded-code-intent-v5",
                        family=spec.family,
                    )
                )
    _assert_unique_prompts(row for split in SPLITS for row in rows[split])
    return rows


def _load_v4_contract(prefix: Path) -> tuple[dict[str, Any], dict[str, list[Candidate]]]:
    report_path = _report_path(prefix).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"v4 router report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid v4 router report: {report_path}") from exc
    if report.get("schema_version") != "molor.router-bootstrap-dataset.v4":
        raise ValueError("v4 router report has an unexpected schema")
    if report.get("route_ids") != list(ROUTE_IDS):
        raise ValueError("v4 router report has an unexpected route axis")
    for field_name in ("normalized_prompt_overlap", "semantic_group_overlap"):
        overlap = report.get(field_name)
        if not isinstance(overlap, dict) or any(overlap.values()):
            raise ValueError(f"v4 router report does not prove zero {field_name}")

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("v4 router report has no output identities")
    results = {split: [] for split in SPLITS}
    for split in SPLITS:
        path = _split_path(prefix, split).resolve()
        expected = outputs.get(split)
        if not path.is_file():
            raise FileNotFoundError(f"v4 router split does not exist: {path}")
        if not isinstance(expected, dict) or expected.get("sha256") != _sha256_file(path):
            raise ValueError(f"v4 router split differs from its report: {path}")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or set(raw) != {"prompt", "route_id"}:
                        raise ValueError("row must contain exactly prompt and route_id")
                    prompt = raw["prompt"]
                    route_id = raw["route_id"]
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError("prompt must be non-empty text")
                    if route_id not in ROUTE_IDS:
                        raise ValueError(f"unknown route {route_id!r}")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                normalized = _normalize_prompt(prompt)
                results[split].append(
                    Candidate(
                        prompt=prompt.strip(),
                        normalized_prompt=normalized,
                        route_id=route_id,
                        split=split,
                        semantic_group=f"v4-preserved/{split}/{_stable_hash(normalized)}",
                        source="router-bootstrap-v4-promoted",
                    )
                )
        if expected.get("rows") != len(results[split]):
            raise ValueError(f"v4 router split row count differs from its report: {path}")
    _assert_unique_prompts(row for split in SPLITS for row in results[split])
    return report, results


def _generated_code_groups(
    v4_report: dict[str, Any],
) -> dict[str, dict[str, list[Candidate]]]:
    seeds = v4_report.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("v4 router report has no seed contract")
    generator_seed = seeds.get("verified_code_generator")
    split_seed = seeds.get("verified_code_split")
    if not isinstance(generator_seed, int) or not isinstance(split_seed, int):
        raise ValueError("v4 router report has invalid verified-code seeds")

    generated = _verified_code_rows(
        generator_seed=generator_seed,
        split_seed=split_seed,
    )
    grouped: dict[str, dict[str, list[Candidate]]] = {}
    for split in SPLITS:
        by_group: dict[str, list[Candidate]] = defaultdict(list)
        for row in generated[split]:
            by_group[row.semantic_group].append(row)
        grouped[split] = dict(by_group)
    return grouped


def _replace_generated_code_groups(
    v4_rows: dict[str, list[Candidate]],
    generated_groups: dict[str, dict[str, list[Candidate]]],
    authored_rows: dict[str, list[Candidate]],
    *,
    replacement_seed: int,
) -> tuple[dict[str, list[Candidate]], dict[str, dict[str, Any]]]:
    selected: dict[str, list[Candidate]] = {}
    replacement_report: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        normalized_v4 = {row.normalized_prompt for row in v4_rows[split]}
        eligible = {
            group: rows
            for group, rows in generated_groups[split].items()
            if all(row.normalized_prompt in normalized_v4 for row in rows)
        }
        ordered_groups = sorted(
            eligible,
            key=lambda group: _stable_hash(replacement_seed, split, group),
        )
        remove_groups: list[str] = []
        remove_count = 0
        target = len(authored_rows[split])
        for group in ordered_groups:
            group_size = len(eligible[group])
            if remove_count + group_size > target:
                continue
            remove_groups.append(group)
            remove_count += group_size
            if remove_count == target:
                break
        if remove_count != target:
            raise ValueError(
                f"cannot replace exactly {target} generated code rows in {split}; "
                f"selected {remove_count}"
            )

        removed_prompts = {
            row.normalized_prompt for group in remove_groups for row in eligible[group]
        }
        replacements = iter(
            sorted(
                authored_rows[split],
                key=lambda row: _stable_hash(
                    replacement_seed,
                    "replacement",
                    split,
                    row.normalized_prompt,
                ),
            )
        )
        rows: list[Candidate] = []
        changed_positions: list[int] = []
        for index, row in enumerate(v4_rows[split]):
            if row.normalized_prompt in removed_prompts:
                rows.append(next(replacements))
                changed_positions.append(index)
            else:
                rows.append(row)
        try:
            next(replacements)
        except StopIteration:
            pass
        else:
            raise AssertionError(f"unused v5 replacements remain in {split}")

        selected[split] = rows
        replacement_report[split] = {
            "removed_rows": len(removed_prompts),
            "removed_generated_semantic_groups": len(remove_groups),
            "removed_group_ids": remove_groups,
            "added_rows": len(authored_rows[split]),
            "unchanged_rows": len(rows) - len(changed_positions),
            "changed_positions": changed_positions,
            "all_replacements_preserve_route": all(
                v4_rows[split][index].route_id == rows[index].route_id == CODE_ROUTE_ID
                for index in changed_positions
            ),
        }
    return selected, replacement_report


def _route_counts(rows: dict[str, list[Candidate]]) -> dict[str, dict[str, int]]:
    return {split: dict(Counter(row.route_id for row in rows[split])) for split in SPLITS}


def build_router_v5_dataset(
    *,
    v4_prefix: Path,
    output_prefix: Path,
    replacement_seed: int = 5081,
) -> dict[str, Any]:
    """Build v5 as an in-place, code-route-only substitution over promoted v4."""

    v4_report, v4_rows = _load_v4_contract(v4_prefix.resolve())
    authored_rows = _v5_authored_rows()
    generated_groups = _generated_code_groups(v4_report)
    selected, replacements = _replace_generated_code_groups(
        v4_rows,
        generated_groups,
        authored_rows,
        replacement_seed=replacement_seed,
    )
    _assert_unique_prompts(row for split in SPLITS for row in selected[split])
    prompt_overlap = _overlap_counts(selected, "normalized_prompt")
    group_overlap = _overlap_counts(selected, "semantic_group")
    if any(prompt_overlap.values()) or any(group_overlap.values()):
        raise AssertionError("router v5 construction leaked prompts or semantic groups")

    balance = _route_counts(selected)
    expected_balance = v4_report.get("balance")
    if balance != expected_balance:
        raise AssertionError("router v5 changed the promoted v4 route balance")
    for split in SPLITS:
        for index, old_row in enumerate(v4_rows[split]):
            new_row = selected[split][index]
            if old_row.route_id != CODE_ROUTE_ID and old_row.output_row() != new_row.output_row():
                raise AssertionError("router v5 changed a non-code row")

    outputs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = _split_path(output_prefix, split).resolve()
        count, digest = _write_jsonl(path, selected[split])
        outputs[split] = {"path": str(path), "rows": count, "sha256": digest}

    builder_path = Path(__file__).resolve()
    v4_builder = EXPERIMENT_ROOT / "build_router_v4_hard_negative_dataset.py"
    v4_report_path = _report_path(v4_prefix).resolve()
    report = {
        "schema_version": "molor.router-bootstrap-dataset.v5",
        "route_ids": list(ROUTE_IDS),
        "evaluation_artifacts_read_by_builder": False,
        "exact_probe_rows_used": False,
        "blind_gap_analysis_informed_curriculum": True,
        "probe_status_for_v5": "development_only_after_gap_informed_curriculum",
        "seeds": {
            "verified_code_generator": v4_report["seeds"]["verified_code_generator"],
            "verified_code_split": v4_report["seeds"]["verified_code_split"],
            "v4_balanced_selection": v4_report["seeds"]["balanced_selection"],
            "v5_replacement": replacement_seed,
        },
        "split_contract": {
            "v4": "reported promoted-v4 outputs are loaded and hash-verified",
            "replacement": (
                "whole generated verified-code groups are replaced at their original positions"
            ),
            "unaffected": "all non-code rows and all non-replaced code rows remain unchanged",
            "v5_authored": "each new semantic family belongs to exactly one split",
        },
        "source_identity": {
            "builder": str(builder_path),
            "builder_sha256": _sha256_file(builder_path),
            "v4_builder": str(v4_builder.resolve()),
            "v4_builder_sha256": _sha256_file(v4_builder),
            "v4_report": str(v4_report_path),
            "v4_report_sha256": _sha256_file(v4_report_path),
        },
        "v4_outputs": {
            split: {
                "path": str(_split_path(v4_prefix, split).resolve()),
                "rows": len(v4_rows[split]),
                "sha256": _sha256_file(_split_path(v4_prefix, split).resolve()),
            }
            for split in SPLITS
        },
        "v5_authored_source": {
            "counts": {split: len(authored_rows[split]) for split in SPLITS},
            "families": {
                split: sorted(
                    {row.family for row in authored_rows[split] if row.family is not None}
                )
                for split in SPLITS
            },
            "semantic_groups": {
                split: len({row.semantic_group for row in authored_rows[split]}) for split in SPLITS
            },
        },
        "replacement": replacements,
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
        "--v4-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v4-hard-negative-s307",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v5-verified-code-contrastive",
    )
    parser.add_argument("--replacement-seed", type=int, default=5081)
    args = parser.parse_args()
    report = build_router_v5_dataset(
        v4_prefix=args.v4_prefix,
        output_prefix=args.output_prefix,
        replacement_seed=args.replacement_seed,
    )
    print(
        json.dumps(
            {
                "route_ids": report["route_ids"],
                "balance": report["balance"],
                "v5_authored_counts": report["v5_authored_source"]["counts"],
                "replacement": report["replacement"],
                "outputs": report["outputs"],
                "report": report["report_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
