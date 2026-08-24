"""Build a deterministic verified CAN expert shard from GLM's prepared OCB corpus.

Only trusted/gold ``table_row`` records are authoritative.  Prepared QA answers are
never copied: in particular, every physical-value-100 encode question is excluded.
The builder regenerates location facts and synthetic payload decodes from each table
row, round-trips every payload through the existing trusted CAN decoder, and writes
strict messages-only SFT files plus line-aligned provenance sidecars.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import build_trusted_can as can_math


POLICY_VERSION = "ocb-can-verified-v1"
OUTPUT_STEM = "ocb-can-verified"
SPLIT_SEED = "ocb-can-verified-message-split-v1"
LOCATION_VARIANTS = 16
DECODE_VARIANTS = 16
EXAMPLES_PER_SIGNAL = LOCATION_VARIANTS + DECODE_VARIANTS
SYSTEM_PROMPT = (
    "You are the verified OCB CAN expert. Return stored catalog facts exactly for lookup "
    "questions. For payload questions, follow DBC start-bit and byte-order conventions "
    "literally and calculate from the supplied definition. Do not invent undocumented "
    "signal semantics."
)

REQUIRED_ROW_FIELDS = {
    "signal_id",
    "signal_name",
    "start_bit",
    "bit_length",
    "scale",
    "offset",
    "unit",
    "byte_order",
    "is_signed",
    "training_eligible",
    "can_id",
    "message_name",
    "dlc",
    "make",
    "model",
    "year",
    "platform",
    "vehicle_tier",
    "data_trust",
}


@dataclass(frozen=True)
class SourceSignal:
    source_file: Path
    source_file_sha256: str
    source_partition: str
    source_line: int
    source_record_id: str
    source_record_sha256: str
    source_row_sha256: str
    row: dict[str, Any]


@dataclass(frozen=True)
class Example:
    message_group: str
    signal_group: str
    sort_key: tuple[str, int, int, int]
    messages: list[dict[str, str]]
    provenance: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    return str(value or "").strip()


def valid_text(value: object, *, required: bool = True) -> bool:
    text = clean_text(value)
    if required and not text:
        return False
    return "\ufffd" not in text and not any(
        ord(character) < 32 and character not in "\t\n\r" for character in text
    )


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["frame_length"] = result.get("dlc")
    return result


def vehicle_components(row: dict[str, Any]) -> dict[str, str | None]:
    return {
        "make": clean_text(row["make"]),
        "model": clean_text(row["model"]),
        "year": clean_text(row["year"]),
        "platform": clean_text(row.get("platform")) or None,
    }


def vehicle_text(row: dict[str, Any]) -> str:
    base = f"{clean_text(row['year'])} {clean_text(row['make'])} {clean_text(row['model'])}"
    platform = clean_text(row.get("platform"))
    return f"{base} ({platform})" if platform else base


def group_components(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "vehicle": vehicle_components(row),
        "message": {
            "can_id": int(row["can_id"]),
            "name": clean_text(row["message_name"]),
        },
    }


def message_group(row: dict[str, Any]) -> str:
    return f"ocb-message:{canonical_sha256(group_components(row))[:20]}"


def signal_group(row: dict[str, Any]) -> str:
    components = {**group_components(row), "signal_id": int(row["signal_id"])}
    return f"ocb-signal:{canonical_sha256(components)[:20]}"


def classify_qa(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return "malformed"
    first = messages[0]
    if not isinstance(first, dict) or not isinstance(first.get("content"), str):
        return "malformed"
    prompt = first["content"]
    if "physical value of 100" in prompt and "encode" in prompt.casefold():
        return "encode_physical_100"
    if prompt.startswith("Which CAN message carries"):
        return "location"
    if prompt.startswith("On a ") and "how do you decode" in prompt:
        return "definition_decode"
    return "other"


def scan_sources(paths: Sequence[Path]) -> tuple[list[SourceSignal], dict[str, Any]]:
    records: list[SourceSignal] = []
    file_reports: list[dict[str, Any]] = []
    global_kinds: Counter[str] = Counter()
    global_qa_types: Counter[str] = Counter()
    seen_record_ids: set[str] = set()

    for path in paths:
        digest = sha256_file(path)
        partition = path.stem
        kinds: Counter[str] = Counter()
        qa_types: Counter[str] = Counter()
        lines = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                lines += 1
                try:
                    wrapper = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(wrapper, dict):
                    raise ValueError(f"{path}:{line_number}: source record is not an object")
                kind = str(wrapper.get("kind") or "missing")
                kinds[kind] += 1
                global_kinds[kind] += 1
                record_id = wrapper.get("id")
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError(f"{path}:{line_number}: missing source record id")
                if record_id in seen_record_ids:
                    raise ValueError(f"duplicate source record id: {record_id}")
                seen_record_ids.add(record_id)

                if kind == "qa":
                    qa_type = classify_qa(wrapper.get("messages"))
                    qa_types[qa_type] += 1
                    global_qa_types[qa_type] += 1
                    continue
                if kind != "table_row":
                    continue
                row = wrapper.get("row")
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: table_row has no row object")
                records.append(
                    SourceSignal(
                        source_file=path.resolve(),
                        source_file_sha256=digest,
                        source_partition=partition,
                        source_line=line_number,
                        source_record_id=record_id,
                        source_record_sha256=canonical_sha256(wrapper),
                        source_row_sha256=canonical_sha256(row),
                        row=row,
                    )
                )
        file_reports.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "nonblank_lines": lines,
                "kinds": dict(sorted(kinds.items())),
                "qa_types_excluded": dict(sorted(qa_types.items())),
            }
        )

    return records, {
        "files": file_reports,
        "kinds": dict(sorted(global_kinds.items())),
        "qa_types_excluded": dict(sorted(global_qa_types.items())),
        "all_prepared_qa_rows_excluded": sum(global_qa_types.values()),
        "encode_physical_100_rows_excluded": global_qa_types["encode_physical_100"],
    }


def rejection_reason(source: SourceSignal) -> str | None:
    row = source.row
    missing = REQUIRED_ROW_FIELDS - set(row)
    if missing:
        return "missing_required_fields"
    if not (
        row["training_eligible"] == "gold"
        and row["data_trust"] == "trusted"
        and row["vehicle_tier"] == "gold"
    ):
        return "not_trusted_gold"
    integer_fields = ("signal_id", "start_bit", "bit_length", "can_id", "dlc", "is_signed")
    if any(type(row[field]) is not int for field in integer_fields):
        return "invalid_integer_field"
    if not 0 <= row["can_id"] <= 0x1FFFFFFF:
        return "invalid_can_identifier"
    valid_lengths = set(range(1, 9)) | {12, 16, 20, 24, 32, 48, 64}
    if row["dlc"] not in valid_lengths:
        return "invalid_payload_length"
    if row["bit_length"] < 1 or row["is_signed"] not in {0, 1}:
        return "invalid_field_encoding"
    if row["byte_order"] not in {"big_endian", "little_endian"}:
        return "invalid_field_encoding"
    if any(not valid_text(row[field]) for field in ("signal_name", "message_name", "make", "model", "year")):
        return "invalid_identity_text"
    if not valid_text(row.get("platform"), required=False) or not valid_text(
        row.get("unit"), required=False
    ):
        return "invalid_optional_text"
    for field in ("scale", "offset"):
        value = row[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return "nonfinite_conversion"
    if float(row["scale"]) == 0.0:
        return "zero_scale"
    normalized = normalized_row(row)
    if can_math.dbc_bit_positions(normalized) is None:
        return "field_traversal_outside_payload"
    return None


def filter_records(records: Sequence[SourceSignal]) -> tuple[list[SourceSignal], Counter[str]]:
    accepted_by_signal: dict[int, SourceSignal] = {}
    rejections: Counter[str] = Counter()
    for source in records:
        rejection = rejection_reason(source)
        if rejection is not None:
            rejections[rejection] += 1
            continue
        signal_id = int(source.row["signal_id"])
        previous = accepted_by_signal.get(signal_id)
        if previous is None:
            accepted_by_signal[signal_id] = source
        elif previous.source_row_sha256 == source.source_row_sha256:
            rejections["duplicate_identical_signal_row"] += 1
        else:
            raise ValueError(
                f"conflicting trusted/gold rows for signal_id {signal_id}: "
                f"{previous.source_file}:{previous.source_line} and "
                f"{source.source_file}:{source.source_line}"
            )
    return sorted(accepted_by_signal.values(), key=lambda item: int(item.row["signal_id"])), rejections


def format_decimal(value: Decimal) -> str:
    return can_math.format_decimal(value)


def decimal(value: object) -> Decimal:
    return Decimal(str(value))


def conversion_text(row: dict[str, Any]) -> str:
    scale = decimal(row["scale"])
    offset = decimal(row["offset"])
    sign = "+" if offset >= 0 else "-"
    return f"physical = raw x {format_decimal(scale)} {sign} {format_decimal(abs(offset))}"


def raw_limits(row: dict[str, Any]) -> tuple[int, int]:
    return can_math.raw_limits(normalized_row(row))


def physical_value(row: dict[str, Any], raw: int) -> Decimal:
    with localcontext() as context:
        context.prec = 60
        return decimal(raw) * decimal(row["scale"]) + decimal(row["offset"])


def format_can_id(value: int) -> str:
    return can_math.format_can_id(value)


def definition(row: dict[str, Any]) -> dict[str, Any]:
    can_id = int(row["can_id"])
    low, high = raw_limits(row)
    return {
        "trust": {
            "data_trust": row["data_trust"],
            "training_eligible": row["training_eligible"],
            "vehicle_tier": row["vehicle_tier"],
        },
        "vehicle": vehicle_components(row),
        "frame": {
            "message_name": clean_text(row["message_name"]),
            "can_id_hex": format_can_id(can_id),
            "can_id_decimal": can_id,
            "identifier_format": can_math.identifier_kind(can_id),
            "payload_length_bytes": int(row["dlc"]),
        },
        "signal": {
            "database_signal_id": int(row["signal_id"]),
            "name": clean_text(row["signal_name"]),
            "dbc_start_bit": int(row["start_bit"]),
            "bit_length": int(row["bit_length"]),
            "byte_order": row["byte_order"],
            "signedness": "signed" if row["is_signed"] else "unsigned",
            "raw_min": low,
            "raw_max": high,
            "scale": format_decimal(decimal(row["scale"])),
            "offset": format_decimal(decimal(row["offset"])),
            "unit": clean_text(row.get("unit")) or "unitless",
        },
    }


LOCATION_PROMPTS = (
    "According to the trusted OCB CAN catalog, locate {signal} on the {vehicle}.",
    "Which verified CAN frame and DBC field contain {signal} for the {vehicle}?",
    "Give the cataloged message identity and exact bit layout for {signal} on the {vehicle}.",
    "Look up {signal} for the {vehicle}; return its CAN ID, DLC, start bit, width, and encoding.",
    "I need the verified decoder coordinates for {signal} on the {vehicle}. Where is it stored?",
    "Identify the trusted OCB placement of {signal} for the {vehicle}, including conversion data.",
    "For the {vehicle}, state the canonical CAN location and decoder specification for {signal}.",
    "Verify the frame mapping, field geometry, and raw-to-physical conversion for {signal} on the {vehicle}.",
)

ANCHORED_LOCATION_PROMPTS = (
    "In trusted frame {message} ({can_id}) on the {vehicle}, locate the cataloged {signal} field.",
    "Give the exact DBC field coordinates for {signal} within {message} ({can_id}) on the {vehicle}.",
    "For {message} ({can_id}) on the {vehicle}, how is {signal} laid out and converted?",
    "Return the verified decoder geometry for {signal} in {message} ({can_id}), {vehicle}.",
    "Within the cataloged {message} frame ({can_id}) for the {vehicle}, where does {signal} live?",
    "State the start bit, width, order, signedness, and conversion for {signal} in {message} ({can_id}).",
    "Verify the trusted OCB placement of {signal} inside {message} ({can_id}) on the {vehicle}.",
    "Describe the canonical DBC extraction record for {signal} from {message} ({can_id}), {vehicle}.",
)


def location_messages(
    row: dict[str, Any], ordinal: int, *, unique_vehicle_signal: bool
) -> list[dict[str, str]]:
    values = {
        "signal": f"`{clean_text(row['signal_name'])}`",
        "vehicle": vehicle_text(row),
        "message": f"`{clean_text(row['message_name'])}`",
        "can_id": format_can_id(int(row["can_id"])),
    }
    templates = LOCATION_PROMPTS if unique_vehicle_signal else ANCHORED_LOCATION_PROMPTS
    prompt = templates[ordinal % len(templates)].format(**values)
    compact = ordinal >= len(templates)
    if compact:
        prompt += " Respond as one compact decoder record and include the raw domain and unit."
    else:
        prompt += " Include message name, CAN ID, DLC, start bit, width, byte order, signedness, scale, offset, and unit."

    low, high = raw_limits(row)
    can_id = int(row["can_id"])
    signedness = "signed" if row["is_signed"] else "unsigned"
    unit = clean_text(row.get("unit")) or "unitless"
    if compact:
        answer = (
            f"{vehicle_text(row)} | {clean_text(row['message_name'])} | "
            f"CAN {format_can_id(can_id)} ({can_id}) | DLC {row['dlc']} | "
            f"{clean_text(row['signal_name'])}: start {row['start_bit']}, width {row['bit_length']}, "
            f"{row['byte_order']}, {signedness}, raw {low}..{high}, "
            f"scale {format_decimal(decimal(row['scale']))}, "
            f"offset {format_decimal(decimal(row['offset']))}, unit {unit}."
        )
    else:
        answer = (
            f"On the {vehicle_text(row)}, `{clean_text(row['signal_name'])}` is in message "
            f"`{clean_text(row['message_name'])}`, CAN ID {format_can_id(can_id)} (decimal {can_id}), "
            f"DLC {row['dlc']}. It starts at DBC bit {row['start_bit']}, spans "
            f"{row['bit_length']} bits, uses {row['byte_order']} ordering, and is {signedness}. "
            f"Its raw domain is {low}..{high}; `{conversion_text(row)}` gives {unit}."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def deterministic_raw_values(row: dict[str, Any], count: int) -> list[int]:
    low, high = raw_limits(row)
    size = high - low + 1
    anchors = [
        low,
        high,
        0,
        1,
        -1,
        low + (size - 1) // 2,
        low + (size - 1) // 4,
        low + 3 * (size - 1) // 4,
    ]
    selected: list[int] = []
    for value in anchors:
        if low <= value <= high and value not in selected:
            selected.append(value)
    counter = 0
    target_unique = min(count, size)
    while len(selected) < target_unique:
        salt = f"{POLICY_VERSION}|raw|{row['signal_id']}|{counter}"
        value = low + can_math.stable_int(salt) % size
        if value not in selected:
            selected.append(value)
        counter += 1
    return [selected[index % len(selected)] for index in range(count)]


def deterministic_background(row: dict[str, Any], ordinal: int) -> bytes:
    seed = f"{POLICY_VERSION}|payload|{row['signal_id']}|{ordinal}".encode("utf-8")
    result = bytearray()
    counter = 0
    while len(result) < int(row["dlc"]):
        result.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(result[: int(row["dlc"])])


def payload_with_raw(row: dict[str, Any], raw: int, ordinal: int) -> tuple[bytes, bytes]:
    normalized = normalized_row(row)
    positions = can_math.dbc_bit_positions(normalized)
    if positions is None:
        raise AssertionError("accepted row lost its valid traversal")
    background = deterministic_background(row, ordinal)
    payload = bytearray(background)
    width = int(row["bit_length"])
    encoded = raw & ((1 << width) - 1)
    for field_index, position in enumerate(positions):
        shift = width - 1 - field_index if row["byte_order"] == "big_endian" else field_index
        mask = 1 << (position % 8)
        if (encoded >> shift) & 1:
            payload[position // 8] |= mask
        else:
            payload[position // 8] &= ~mask
    return bytes(payload), background


def decode_messages(row: dict[str, Any], ordinal: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    normalized = normalized_row(row)
    positions = can_math.dbc_bit_positions(normalized)
    if positions is None:
        raise AssertionError("accepted row lost its valid traversal")
    raw = deterministic_raw_values(row, DECODE_VARIANTS)[ordinal]
    low, high = raw_limits(row)
    if not low <= raw <= high:
        raise AssertionError(f"generated raw {raw} is outside {low}..{high}")
    payload, background = payload_with_raw(row, raw, ordinal)
    decoded = can_math.decode_payload(normalized, payload)
    if decoded != raw:
        raise AssertionError(
            f"signal {row['signal_id']} payload round trip failed: encoded {raw}, decoded {decoded}"
        )
    physical = physical_value(row, raw)
    payload_hex = " ".join(f"{byte:02X}" for byte in payload)
    supplied = json.dumps(definition(row), indent=2, ensure_ascii=False, sort_keys=True)
    prompt = (
        "Decode the named signal from this payload using the supplied trusted/gold OCB "
        "definition. Show the DBC bit traversal, extracted field integer, sign handling, "
        "and exact raw-to-physical calculation. Ignore unrelated payload bits and do not "
        "infer undocumented semantics.\n\n"
        f"SUPPLIED_SIGNAL_DEFINITION:\n{supplied}\n\n"
        f"SUPPLIED_PAYLOAD_HEX ({row['dlc']} bytes): {payload_hex}"
    )
    width = int(row["bit_length"])
    encoded = raw & ((1 << width) - 1)
    if row["is_signed"]:
        extraction = (
            f"The unsigned field pattern is {encoded}; {width}-bit two's-complement sign "
            f"extension gives raw {raw}."
        )
    else:
        extraction = f"The field is unsigned, so field pattern {encoded} gives raw {raw}."
    offset = decimal(row["offset"])
    answer = (
        f"Starting at DBC bit {row['start_bit']}, the {row['byte_order']} {width}-bit traversal "
        f"uses positions {list(positions)}. {extraction} Apply `{conversion_text(row)}`: "
        f"{raw} x {format_decimal(decimal(row['scale']))} "
        f"{'+' if offset >= 0 else '-'} {format_decimal(abs(offset))} = "
        f"{format_decimal(physical)} {clean_text(row.get('unit')) or 'unitless'}."
    )
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        {
            "generated_raw_value": raw,
            "raw_domain": [low, high],
            "generated_physical_value": format_decimal(physical),
            "generated_payload_hex": payload_hex,
            "background_payload_hex_before_field_write": " ".join(
                f"{byte:02X}" for byte in background
            ),
            "dbc_bit_positions": list(positions),
            "round_trip_decoded_raw": decoded,
            "round_trip_verified": True,
            "raw_representable": True,
        },
    )


def base_provenance(source: SourceSignal, task: str, ordinal: int) -> dict[str, Any]:
    row = source.row
    return {
        "example_id": (
            f"ocb:signal:{row['signal_id']}:{task}:{ordinal:02d}:{POLICY_VERSION}"
        ),
        "example_type": task,
        "variant_ordinal": ordinal,
        "policy_version": POLICY_VERSION,
        "message_group": message_group(row),
        "signal_group": signal_group(row),
        "split_group_components": group_components(row),
        "source": "GLM-prepared OCB CAN signal database",
        "source_file": str(source.source_file),
        "source_file_sha256": source.source_file_sha256,
        "source_partition_before_regrouping": source.source_partition,
        "source_line": source.source_line,
        "source_record_id": source.source_record_id,
        "source_record_sha256": source.source_record_sha256,
        "source_row_sha256": source.source_row_sha256,
        "source_kind": "table_row",
        "source_row": row,
        "source_qa_answer_used": False,
        "trust_filter": {
            "data_trust": row["data_trust"],
            "training_eligible": row["training_eligible"],
            "vehicle_tier": row["vehicle_tier"],
        },
        "license": "not specified in prepared record; local user-provided corpus",
    }


def make_examples(
    source: SourceSignal, *, unique_vehicle_signal: bool
) -> list[Example]:
    row = source.row
    result: list[Example] = []
    for ordinal in range(LOCATION_VARIANTS):
        messages = location_messages(row, ordinal, unique_vehicle_signal=unique_vehicle_signal)
        provenance = base_provenance(source, "location", ordinal)
        provenance["lookup_mode"] = (
            "vehicle_signal_recall" if unique_vehicle_signal else "message_anchored_location"
        )
        provenance["training_messages_sha256"] = canonical_sha256(messages)
        result.append(
            Example(
                message_group(row),
                signal_group(row),
                (source.source_partition, int(row["signal_id"]), 0, ordinal),
                messages,
                provenance,
            )
        )
    for ordinal in range(DECODE_VARIANTS):
        messages, generated = decode_messages(row, ordinal)
        provenance = base_provenance(source, "decode", ordinal)
        provenance.update(generated)
        provenance["training_messages_sha256"] = canonical_sha256(messages)
        result.append(
            Example(
                message_group(row),
                signal_group(row),
                (source.source_partition, int(row["signal_id"]), 1, ordinal),
                messages,
                provenance,
            )
        )
    return result


def unique_vehicle_signal_keys(records: Sequence[SourceSignal]) -> set[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for source in records:
        row = source.row
        key = (
            clean_text(row["make"]).casefold(),
            clean_text(row["model"]).casefold(),
            clean_text(row["year"]).casefold(),
            clean_text(row.get("platform")).casefold(),
            clean_text(row["signal_name"]).casefold(),
        )
        counts[key] += 1
    return {key for key, count in counts.items() if count == 1}


def vehicle_signal_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        clean_text(row["make"]).casefold(),
        clean_text(row["model"]).casefold(),
        clean_text(row["year"]).casefold(),
        clean_text(row.get("platform")).casefold(),
        clean_text(row["signal_name"]).casefold(),
    )


def choose_validation_groups(
    records: Sequence[SourceSignal], target_signals: int
) -> tuple[set[str], int]:
    grouped: dict[str, list[SourceSignal]] = defaultdict(list)
    for source in records:
        grouped[message_group(source.row)].append(source)
    if len(grouped) < 2:
        raise ValueError("at least two vehicle/message groups are required")

    ranked = sorted(
        grouped,
        key=lambda group: (can_math.stable_int(f"{SPLIT_SEED}|{group}"), group),
    )
    states: dict[int, tuple[str, ...]] = {0: ()}
    for group in ranked:
        size = len(grouped[group])
        additions = {
            count + size: selected + (group,)
            for count, selected in list(states.items())
            if count + size < len(records)
        }
        for count, selected in additions.items():
            states.setdefault(count, selected)
    candidates = [count for count in states if 0 < count < len(records)]
    chosen_count = min(
        candidates,
        key=lambda count: (
            abs(count - target_signals),
            count > target_signals,
            states[count],
        ),
    )
    return set(states[chosen_count]), chosen_count


def validate_examples(examples: Sequence[Example]) -> dict[str, Any]:
    ids: set[str] = set()
    message_hashes: set[str] = set()
    duplicate_messages = 0
    task_counts: Counter[str] = Counter()
    for example in examples:
        row = {"messages": example.messages}
        can_math.validate_training_row(row)
        example_id = str(example.provenance["example_id"])
        if example_id in ids:
            raise AssertionError(f"duplicate example id: {example_id}")
        ids.add(example_id)
        digest = canonical_sha256(example.messages)
        if digest in message_hashes:
            duplicate_messages += 1
        message_hashes.add(digest)
        if digest != example.provenance["training_messages_sha256"]:
            raise AssertionError(f"message hash mismatch for {example_id}")
        task_counts[str(example.provenance["example_type"])] += 1
        if example.provenance["source_kind"] != "table_row":
            raise AssertionError(f"non-table source used for {example_id}")
    if duplicate_messages:
        raise AssertionError(f"generated {duplicate_messages} duplicate message sets")
    return {
        "examples": len(examples),
        "unique_example_ids": len(ids),
        "unique_training_message_sets": len(message_hashes),
        "task_counts": dict(sorted(task_counts.items())),
        "duplicate_training_message_sets": duplicate_messages,
    }


def write_partition(output_dir: Path, name: str, examples: Iterable[Example]) -> dict[str, Any]:
    ordered = sorted(examples, key=lambda example: example.sort_key)
    data_path = output_dir / f"{OUTPUT_STEM}.{name}.jsonl"
    provenance_path = output_dir / f"{OUTPUT_STEM}.{name}.provenance.jsonl"
    data_digest = hashlib.sha256()
    provenance_digest = hashlib.sha256()
    task_counts: Counter[str] = Counter()
    data_chars: list[int] = []

    with data_path.open("w", encoding="utf-8", newline="\n") as data_stream, provenance_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as provenance_stream:
        for line_number, example in enumerate(ordered, start=1):
            training_row = {"messages": example.messages}
            can_math.validate_training_row(training_row)
            data_line = json.dumps(training_row, ensure_ascii=False, separators=(",", ":")) + "\n"
            provenance_line = json.dumps(
                {"line": line_number, **example.provenance},
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
            data_stream.write(data_line)
            provenance_stream.write(provenance_line)
            data_digest.update(data_line.encode("utf-8"))
            provenance_digest.update(provenance_line.encode("utf-8"))
            task_counts[str(example.provenance["example_type"])] += 1
            data_chars.append(len(data_line))

    return {
        "data_path": str(data_path.resolve()),
        "data_sha256": data_digest.hexdigest(),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": provenance_digest.hexdigest(),
        "rows": len(ordered),
        "message_groups": len({example.message_group for example in ordered}),
        "signal_groups": len({example.signal_group for example in ordered}),
        "task_counts": dict(sorted(task_counts.items())),
        "row_character_lengths": {
            "min": min(data_chars),
            "max": max(data_chars),
            "mean": sum(data_chars) / len(data_chars),
        },
    }


def audit_written_messages_only(output: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(output["data_path"])
    provenance_path = Path(output["provenance_path"])
    rows = 0
    with data_path.open("r", encoding="utf-8") as data_stream, provenance_path.open(
        "r", encoding="utf-8"
    ) as provenance_stream:
        for rows, (data_line, provenance_line) in enumerate(
            zip(data_stream, provenance_stream, strict=True), start=1
        ):
            data = json.loads(data_line)
            provenance = json.loads(provenance_line)
            if set(data) != {"messages"}:
                raise AssertionError(f"{data_path}:{rows}: training row is not messages-only")
            can_math.validate_training_row(data)
            if provenance.get("line") != rows:
                raise AssertionError(f"{provenance_path}:{rows}: sidecar line mismatch")
            if canonical_sha256(data["messages"]) != provenance.get("training_messages_sha256"):
                raise AssertionError(f"{provenance_path}:{rows}: sidecar message hash mismatch")
    if rows != output["rows"]:
        raise AssertionError(f"written row mismatch: {rows} != {output['rows']}")
    if sha256_file(data_path) != output["data_sha256"]:
        raise AssertionError(f"written data digest mismatch: {data_path}")
    if sha256_file(provenance_path) != output["provenance_sha256"]:
        raise AssertionError(f"written provenance digest mismatch: {provenance_path}")
    return {"rows_checked": rows, "messages_only": True, "line_alignment": True, "hashes_match": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.home() / "Documents" / "datasets" / "prepared" / "ocb-can-signal-db",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--target-validation-rows", type=int, default=1024)
    args = parser.parse_args()
    if args.target_validation_rows < EXAMPLES_PER_SIGNAL:
        parser.error(f"--target-validation-rows must be at least {EXAMPLES_PER_SIGNAL}")

    input_paths = [args.input_dir / "train.jsonl", args.input_dir / "validation.jsonl"]
    for path in input_paths:
        if not path.is_file():
            parser.error(f"prepared source file does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_records, source_report = scan_sources(input_paths)
    accepted, rejections = filter_records(source_records)
    if not accepted:
        raise ValueError("trusted/gold filters left no usable CAN signal rows")

    unique_keys = unique_vehicle_signal_keys(accepted)
    examples: list[Example] = []
    for source in accepted:
        examples.extend(
            make_examples(
                source,
                unique_vehicle_signal=vehicle_signal_key(source.row) in unique_keys,
            )
        )
    generation_audit = validate_examples(examples)

    target_validation_signals = max(
        1, round(args.target_validation_rows / EXAMPLES_PER_SIGNAL)
    )
    validation_groups, validation_signal_count = choose_validation_groups(
        accepted, target_validation_signals
    )
    train = [example for example in examples if example.message_group not in validation_groups]
    validation = [example for example in examples if example.message_group in validation_groups]

    train_message_groups = {example.message_group for example in train}
    validation_message_groups = {example.message_group for example in validation}
    train_signal_groups = {example.signal_group for example in train}
    validation_signal_groups = {example.signal_group for example in validation}
    leaked_message_groups = sorted(train_message_groups & validation_message_groups)
    leaked_signal_groups = sorted(train_signal_groups & validation_signal_groups)
    if leaked_message_groups or leaked_signal_groups:
        raise AssertionError("vehicle/message/signal grouped split leaked across partitions")

    outputs = {
        "train": write_partition(args.output_dir, "train", train),
        "validation": write_partition(args.output_dir, "validation", validation),
    }
    written_audits = {
        partition: audit_written_messages_only(output)
        for partition, output in outputs.items()
    }

    accepted_vehicle_counts = Counter(vehicle_text(source.row) for source in accepted)
    accepted_message_counts = Counter(message_group(source.row) for source in accepted)
    report = {
        "policy_version": POLICY_VERSION,
        "source": {
            **source_report,
            "authority": "trusted/gold table_row records only",
            "prepared_qa_answers_used": 0,
            "license": "not specified in prepared records; local user-provided corpus",
        },
        "filter": {
            "table_rows_seen": len(source_records),
            "requirements": [
                "data_trust=trusted",
                "training_eligible=gold",
                "vehicle_tier=gold",
                "complete vehicle/message/signal identity",
                "valid numeric 11-bit or 29-bit CAN identifier",
                "valid CAN/CAN-FD payload length",
                "in-payload DBC traversal under stored byte order",
                "finite nonzero scale and finite offset",
            ],
            "rejections": dict(sorted(rejections.items())),
            "accepted_signals": len(accepted),
            "accepted_message_groups": len(accepted_message_counts),
            "accepted_vehicles": dict(sorted(accepted_vehicle_counts.items())),
        },
        "generation": {
            "examples_per_signal": EXAMPLES_PER_SIGNAL,
            "location_variants_per_signal": LOCATION_VARIANTS,
            "decode_variants_per_signal": DECODE_VARIANTS,
            "balanced_location_decode": LOCATION_VARIANTS == DECODE_VARIANTS,
            "physical_to_raw_encode_examples": 0,
            "source_encode_physical_100_examples_used": 0,
            "payload_policy": (
                "deterministic noise in unrelated bits; target raw field overwritten exactly"
            ),
            "raw_policy": (
                "integer anchors plus deterministic hashed values strictly inside the exact "
                "signed/unsigned bit-width domain"
            ),
            "verification": (
                "every synthetic payload decoded through build_trusted_can.decode_payload and "
                "required to round-trip to its chosen raw integer"
            ),
            "unique_vehicle_signal_lookup_records": len(unique_keys),
            "message_anchored_lookup_records": len(accepted) - len(unique_keys),
            "audit": generation_audit,
        },
        "split": {
            "seed": SPLIT_SEED,
            "selection": "deterministic subset-sum nearest requested validation signal count",
            "group_key": (
                "vehicle(make/model/year/platform)/message(can_id/name), with every source "
                "signal and all generated variants kept atomic inside that group"
            ),
            "requested_validation_rows": args.target_validation_rows,
            "target_validation_signals": target_validation_signals,
            "actual_validation_signals": validation_signal_count,
            "train_message_groups": len(train_message_groups),
            "validation_message_groups": len(validation_message_groups),
            "train_signal_groups": len(train_signal_groups),
            "validation_signal_groups": len(validation_signal_groups),
            "message_group_leakage": leaked_message_groups,
            "signal_group_leakage": leaked_signal_groups,
            "validation_group_ids": sorted(validation_groups),
        },
        "outputs": outputs,
        "written_output_audits": written_audits,
        "limitations": [
            "Trusted/gold are internal corpus labels, not an independent external truth audit.",
            "The prepared table rows provide no signal descriptions, enumerations, or declared physical min/max fields.",
            "Synthetic decode payloads are generated from stored definitions, not copied from live captures.",
            "Repeated location paraphrases intentionally reinforce the small trusted/gold catalog slice.",
            "Ambiguous signal names are anchored to a supplied message instead of pretending the vehicle-level lookup is unique.",
        ],
    }
    report_path = args.output_dir / f"{OUTPUT_STEM}.report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "accepted_signals": len(accepted),
                "source_encode_100_excluded": source_report[
                    "encode_physical_100_rows_excluded"
                ],
                "train_rows": outputs["train"]["rows"],
                "validation_rows": outputs["validation"]["rows"],
                "train_task_counts": outputs["train"]["task_counts"],
                "validation_task_counts": outputs["validation"]["task_counts"],
                "group_leakage": len(leaked_message_groups) + len(leaked_signal_groups),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
