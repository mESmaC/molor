"""Build a small, fully grounded CAN SFT shard from the local OCB database.

The SQLite source is opened read-only and never modified.  Training JSONL contains
only the strict ``messages`` shape accepted by the current MoLoR SFT loader. Every
task supplies the complete signal definition it needs; source identity and filter
decisions are retained in sidecar/report files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Sequence


POLICY_VERSION = "grounded-can-v3"
OUTPUT_STEM = "grounded-can"
SPLIT_SEED = "trusted-can-message-group-split-v1"
SYSTEM_PROMPT = (
    "You solve CAN-bus tasks from the complete signal definition supplied in each prompt. "
    "Treat DBC start-bit and byte-order conventions literally, show calculations, and do "
    "not recall, invent, or infer database facts or undocumented signal semantics."
)

REQUIRED_COLUMNS = {
    "signals": {
        "id",
        "message_id",
        "name",
        "start_bit",
        "bit_length",
        "byte_order",
        "is_signed",
        "scale",
        "offset",
        "min_val",
        "max_val",
        "unit",
        "description",
        "data_trust",
        "training_eligible",
    },
    "messages": {
        "id",
        "vehicle_id",
        "bus_id",
        "can_id",
        "name",
        "length",
        "period_ms",
        "sender",
        "description",
        "data_trust",
        "protocol",
        "training_eligible",
    },
    "vehicles": {"id", "make", "model", "year", "platform", "source", "slug", "vehicle_type"},
    "vehicle_quality": {"vehicle_id", "tier"},
    "buses": {"id", "key", "name", "type", "speed_kbps"},
    "signal_tags": {"signal_id", "tag"},
    "signal_values": {"signal_id", "int_value", "label"},
}

PLACEHOLDER_NAME = re.compile(
    r"(?i)(?:^|_)(?:unknown|unnamed|unused|reserved|padding|zeros?|sig\d*|signal\d*|"
    r"new_signal\d*|boh\d*|set_me(?:_x)?\d*)(?:_|$)|^(?:counter|checksum|increasing|status\d*)$"
)
SPECULATIVE_TEXT = re.compile(
    r"(?i)\b(?:likely|perhaps|maybe|might|possibly|unknown|guess|suspect|inferred)\b"
)
WEAK_TAGS = {"placeholder", "unnamed", "unclassified"}
USEFUL_TAGS = {
    "characterized",
    "measurement",
    "status",
    "control_input",
    "object_detection",
    "ev_hybrid",
}


@dataclass(frozen=True)
class Example:
    group: str
    sort_key: tuple[int, int, int, str]
    messages: list[dict[str, str]]
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def inspect_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    available = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    missing_tables = set(REQUIRED_COLUMNS) - available
    if missing_tables:
        raise ValueError(f"database is missing tables: {sorted(missing_tables)}")

    schema: dict[str, list[str]] = {}
    for table, required in REQUIRED_COLUMNS.items():
        columns = [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]
        missing_columns = required - set(columns)
        if missing_columns:
            raise ValueError(f"{table} is missing columns: {sorted(missing_columns)}")
        schema[table] = columns
    return schema


def database_counts(connection: sqlite3.Connection) -> dict[str, int]:
    scalar_queries = {
        "signals_total": "SELECT COUNT(*) FROM signals",
        "signals_trusted_gold": (
            "SELECT COUNT(*) FROM signals "
            "WHERE data_trust = 'trusted' AND training_eligible = 'gold'"
        ),
        "messages_total": "SELECT COUNT(*) FROM messages",
        "messages_trusted_gold": (
            "SELECT COUNT(*) FROM messages "
            "WHERE data_trust = 'trusted' AND training_eligible = 'gold'"
        ),
        "vehicles_total": "SELECT COUNT(*) FROM vehicles",
        "vehicles_gold": "SELECT COUNT(*) FROM vehicle_quality WHERE tier = 'gold'",
    }
    return {name: int(connection.execute(query).fetchone()[0]) for name, query in scalar_queries.items()}


def load_candidates(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            s.id AS signal_id,
            s.name AS signal_name,
            s.start_bit,
            s.bit_length,
            s.byte_order,
            s.is_signed,
            s.scale,
            s.offset,
            s.min_val,
            s.max_val,
            s.unit,
            s.description AS signal_description,
            s.data_trust AS signal_trust,
            s.training_eligible AS signal_training_eligible,
            m.id AS message_id,
            m.can_id,
            m.name AS message_name,
            m.length AS frame_length,
            m.period_ms,
            m.sender,
            m.description AS message_description,
            m.data_trust AS message_trust,
            m.training_eligible AS message_training_eligible,
            m.protocol,
            v.id AS vehicle_id,
            v.make,
            v.model,
            v.year,
            v.platform,
            v.source AS vehicle_source,
            v.slug AS vehicle_slug,
            v.vehicle_type,
            vq.tier AS vehicle_tier,
            b.id AS bus_id,
            b.key AS bus_key,
            b.name AS bus_name,
            b.type AS bus_type,
            b.speed_kbps,
            (
                SELECT GROUP_CONCAT(st.tag, ',')
                FROM signal_tags AS st
                WHERE st.signal_id = s.id
            ) AS tags
        FROM signals AS s
        JOIN messages AS m ON m.id = s.message_id
        JOIN vehicles AS v ON v.id = m.vehicle_id
        JOIN vehicle_quality AS vq ON vq.vehicle_id = v.id
        LEFT JOIN buses AS b ON b.id = m.bus_id
        WHERE
            s.data_trust = 'trusted'
            AND s.training_eligible = 'gold'
            AND m.data_trust = 'trusted'
            AND m.training_eligible = 'gold'
            AND vq.tier = 'gold'
        ORDER BY v.id, m.id, s.id
        """
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        row["tags"] = sorted(set(filter(None, str(row.get("tags") or "").split(","))))
        candidates.append(row)
    return candidates


def load_signal_values(
    connection: sqlite3.Connection, signal_ids: Sequence[int]
) -> dict[int, list[tuple[int, str]]]:
    values: dict[int, list[tuple[int, str]]] = defaultdict(list)
    if not signal_ids:
        return values
    placeholders = ",".join("?" for _ in signal_ids)
    query = (
        "SELECT signal_id, int_value, label FROM signal_values "
        f"WHERE signal_id IN ({placeholders}) ORDER BY signal_id, int_value, label"
    )
    for row in connection.execute(query, tuple(signal_ids)):
        values[int(row[0])].append((int(row[1]), str(row[2])))
    return values


def clean_text(value: object) -> str:
    return str(value or "").strip()


def has_bad_text(value: object) -> bool:
    text = clean_text(value)
    return "\ufffd" in text or any(ord(character) < 32 and character not in "\t\n\r" for character in text)


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def dbc_bit_positions(row: dict[str, Any]) -> tuple[int, ...] | None:
    if not isinstance(row.get("frame_length"), int) or not isinstance(row.get("bit_length"), int):
        return None
    if not isinstance(row.get("start_bit"), int):
        return None
    frame_bits = int(row["frame_length"]) * 8
    bit_length = int(row["bit_length"])
    if frame_bits <= 0 or bit_length <= 0 or bit_length > frame_bits:
        return None
    byte_order = row.get("byte_order")
    if byte_order not in {"big_endian", "little_endian"}:
        return None

    position = int(row["start_bit"])
    positions: list[int] = []
    for _ in range(bit_length):
        if position < 0 or position >= frame_bits:
            return None
        positions.append(position)
        if byte_order == "little_endian":
            position += 1
        else:
            position = position + 15 if position % 8 == 0 else position - 1
    return tuple(positions)


def raw_limits(row: dict[str, Any]) -> tuple[int, int]:
    width = int(row["bit_length"])
    if int(row["is_signed"]):
        return -(1 << (width - 1)), (1 << (width - 1)) - 1
    return 0, (1 << width) - 1


def physical_limits(row: dict[str, Any]) -> tuple[float, float]:
    low_raw, high_raw = raw_limits(row)
    values = (
        low_raw * float(row["scale"]) + float(row["offset"]),
        high_raw * float(row["scale"]) + float(row["offset"]),
    )
    return min(values), max(values)


def allowed_raw_limits(row: dict[str, Any]) -> tuple[int, int] | None:
    """Intersect the integer raw domain with any declared physical bounds."""

    low, high = raw_limits(row)
    scale = Decimal(str(row["scale"]))
    offset = Decimal(str(row["offset"]))
    minimum = row.get("min_val")
    maximum = row.get("max_val")
    with localcontext() as context:
        context.prec = 50
        if scale > 0:
            if minimum is not None:
                boundary = (Decimal(str(minimum)) - offset) / scale
                low = max(low, int(boundary.to_integral_value(rounding=ROUND_CEILING)))
            if maximum is not None:
                boundary = (Decimal(str(maximum)) - offset) / scale
                high = min(high, int(boundary.to_integral_value(rounding=ROUND_FLOOR)))
        else:
            if maximum is not None:
                boundary = (Decimal(str(maximum)) - offset) / scale
                low = max(low, int(boundary.to_integral_value(rounding=ROUND_CEILING)))
            if minimum is not None:
                boundary = (Decimal(str(minimum)) - offset) / scale
                high = min(high, int(boundary.to_integral_value(rounding=ROUND_FLOOR)))
    return None if low > high else (low, high)


def physical_in_declared_range(row: dict[str, Any], physical: Decimal) -> bool:
    minimum = row.get("min_val")
    maximum = row.get("max_val")
    if minimum is not None and physical < Decimal(str(minimum)):
        return False
    if maximum is not None and physical > Decimal(str(maximum)):
        return False
    return True


def identifier_kind(can_id: int) -> str:
    return "standard 11-bit" if can_id <= 0x7FF else "extended 29-bit"


def first_rejection(row: dict[str, Any]) -> str | None:
    if clean_text(row.get("protocol")).upper() != "CAN":
        return "non_can_protocol"
    can_id = row.get("can_id")
    if not isinstance(can_id, int) or not 0 <= can_id <= 0x1FFFFFFF:
        return "invalid_11_or_29_bit_identifier"

    if row.get("vehicle_type") != "vehicle":
        return "non_vehicle_record"
    vehicle_fields = ("make", "model", "year", "vehicle_source", "vehicle_slug")
    if any(not clean_text(row.get(field)) for field in vehicle_fields):
        return "incomplete_vehicle_metadata"
    if any(has_bad_text(row.get(field)) for field in (*vehicle_fields, "platform")):
        return "invalid_vehicle_text"

    frame_length = row.get("frame_length")
    if not isinstance(frame_length, int):
        return "invalid_frame_length"
    is_can_fd = "FD" in clean_text(row.get("bus_type")).upper()
    valid_lengths = set(range(1, 9)) | ({12, 16, 20, 24, 32, 48, 64} if is_can_fd else set())
    if frame_length not in valid_lengths:
        return "invalid_frame_length"
    if dbc_bit_positions(row) is None:
        return "invalid_frame_bit_layout"

    if row.get("is_signed") not in {0, 1}:
        return "invalid_signedness"
    if not finite_number(row.get("scale")) or float(row["scale"]) == 0.0:
        return "zero_or_nonfinite_scale"
    if not finite_number(row.get("offset")):
        return "nonfinite_offset"

    name = clean_text(row.get("signal_name"))
    if not name or len(name) > 100 or has_bad_text(name):
        return "invalid_signal_name"
    if PLACEHOLDER_NAME.search(name):
        return "placeholder_or_generic_signal_name"
    tags = set(row.get("tags") or ())
    if tags & WEAK_TAGS:
        return "weak_signal_tag"
    if not tags & USEFUL_TAGS:
        return "no_useful_signal_metadata"
    if SPECULATIVE_TEXT.search(clean_text(row.get("signal_description"))):
        return "speculative_signal_description"
    if has_bad_text(row.get("signal_description")) or has_bad_text(row.get("unit")):
        return "invalid_signal_text"
    if int(row["bit_length"]) == 1 and clean_text(row.get("unit")):
        return "boolean_signal_with_physical_unit"

    theoretical_low, theoretical_high = physical_limits(row)
    minimum = row.get("min_val")
    maximum = row.get("max_val")
    tolerance = max(1.0, abs(theoretical_low), abs(theoretical_high)) * 1e-12
    if minimum is not None and not finite_number(minimum):
        return "invalid_declared_physical_range"
    if maximum is not None and not finite_number(maximum):
        return "invalid_declared_physical_range"
    if minimum is not None and not theoretical_low - tolerance <= float(minimum) <= theoretical_high + tolerance:
        return "declared_range_conflicts_with_conversion"
    if maximum is not None and not theoretical_low - tolerance <= float(maximum) <= theoretical_high + tolerance:
        return "declared_range_conflicts_with_conversion"
    if minimum is not None and maximum is not None and float(minimum) > float(maximum):
        return "invalid_declared_physical_range"
    if allowed_raw_limits(row) is None:
        return "declared_range_contains_no_integer_raw_value"
    return None


def find_overlaps(rows: Sequence[dict[str, Any]]) -> tuple[set[int], list[dict[str, Any]]]:
    by_message: dict[int, list[tuple[dict[str, Any], set[int]]]] = defaultdict(list)
    for row in rows:
        positions = dbc_bit_positions(row)
        if positions is not None:
            by_message[int(row["message_id"])].append((row, set(positions)))

    rejected: set[int] = set()
    overlaps: list[dict[str, Any]] = []
    for message_rows in by_message.values():
        for index, (left, left_bits) in enumerate(message_rows):
            for right, right_bits in message_rows[index + 1 :]:
                common = sorted(left_bits & right_bits)
                if not common:
                    continue
                rejected.update((int(left["signal_id"]), int(right["signal_id"])))
                overlaps.append(
                    {
                        "message_id": int(left["message_id"]),
                        "can_id": int(left["can_id"]),
                        "left_signal_id": int(left["signal_id"]),
                        "left_signal_name": str(left["signal_name"]),
                        "right_signal_id": int(right["signal_id"]),
                        "right_signal_name": str(right["signal_name"]),
                        "overlapping_dbc_bits": common,
                    }
                )
    return rejected, overlaps


def choose_raw(row: dict[str, Any], values: Sequence[tuple[int, str]]) -> int:
    allowed = allowed_raw_limits(row)
    if allowed is None:
        raise ValueError(f"signal {row['signal_id']} has no raw value inside its declared range")
    low, high = allowed
    usable_values = [value for value, label in values if low <= value <= high and clean_text(label)]
    selection_key = f"{POLICY_VERSION}|raw|{row['signal_id']}"
    if usable_values:
        nonzero_values = sorted(set(value for value in usable_values if value != 0))
        choices = nonzero_values or sorted(set(usable_values))
        return choices[stable_int(selection_key) % len(choices)]

    def choose_nonzero(range_low: int, range_high: int, salt: str) -> int | None:
        if range_low > range_high:
            return None
        count = range_high - range_low + 1
        if range_low <= 0 <= range_high:
            count -= 1
        if count <= 0:
            return None
        index = stable_int(f"{selection_key}|{salt}") % count
        selected = range_low + index
        if range_low <= 0 <= range_high and selected >= 0:
            selected += 1
        return selected

    # Prefer a nonzero interior value.  If the legal interval is very small, retain
    # nontriviality by allowing a nonzero endpoint before falling back to the sole value.
    selected = choose_nonzero(low + 1, high - 1, "interior")
    if selected is None:
        selected = choose_nonzero(low, high, "full")
    return low if selected is None else selected


def encode_payload(row: dict[str, Any], raw_value: int) -> bytes:
    positions = dbc_bit_positions(row)
    if positions is None:
        raise ValueError("cannot encode an invalid DBC layout")
    width = int(row["bit_length"])
    encoded = raw_value & ((1 << width) - 1)
    payload = bytearray(int(row["frame_length"]))
    for field_index, dbc_position in enumerate(positions):
        if row["byte_order"] == "big_endian":
            bit = (encoded >> (width - 1 - field_index)) & 1
        else:
            bit = (encoded >> field_index) & 1
        if bit:
            payload[dbc_position // 8] |= 1 << (dbc_position % 8)
    return bytes(payload)


def decode_payload(row: dict[str, Any], payload: bytes) -> int:
    positions = dbc_bit_positions(row)
    if positions is None:
        raise ValueError("cannot decode an invalid DBC layout")
    encoded = 0
    if row["byte_order"] == "big_endian":
        for dbc_position in positions:
            encoded = (encoded << 1) | ((payload[dbc_position // 8] >> (dbc_position % 8)) & 1)
    else:
        for field_index, dbc_position in enumerate(positions):
            encoded |= ((payload[dbc_position // 8] >> (dbc_position % 8)) & 1) << field_index
    width = int(row["bit_length"])
    if int(row["is_signed"]) and encoded & (1 << (width - 1)):
        return encoded - (1 << width)
    return encoded


def as_decimal(value: object) -> Decimal:
    return Decimal(str(value))


def format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def physical_value(row: dict[str, Any], raw_value: int) -> Decimal:
    with localcontext() as context:
        context.prec = 40
        return as_decimal(raw_value) * as_decimal(row["scale"]) + as_decimal(row["offset"])


def format_can_id(can_id: int) -> str:
    width = 3 if can_id <= 0x7FF else 8
    return f"0x{can_id:0{width}X}"


def format_vehicle(row: dict[str, Any]) -> str:
    return " ".join((clean_text(row["year"]), clean_text(row["make"]), clean_text(row["model"])))


def conversion_text(row: dict[str, Any]) -> str:
    scale = format_decimal(as_decimal(row["scale"]))
    offset = as_decimal(row["offset"])
    sign = "+" if offset >= 0 else "-"
    return f"physical = raw x {scale} {sign} {format_decimal(abs(offset))}"


def declared_range_text(row: dict[str, Any]) -> str:
    minimum = row.get("min_val")
    maximum = row.get("max_val")
    unit = clean_text(row.get("unit")) or "unitless"
    if minimum is None and maximum is None:
        return "No physical range is declared."
    if minimum is not None and maximum is not None:
        return (
            f"The declared physical range is {format_decimal(as_decimal(minimum))} to "
            f"{format_decimal(as_decimal(maximum))} {unit}."
        )
    relation = "at least" if minimum is not None else "at most"
    bound = minimum if minimum is not None else maximum
    return f"The declared physical range is {relation} {format_decimal(as_decimal(bound))} {unit}."


def enum_label(values: Sequence[tuple[int, str]], raw_value: int) -> str | None:
    labels = sorted({clean_text(label) for value, label in values if value == raw_value and clean_text(label)})
    return " / ".join(labels) if labels else None


def source_provenance(
    row: dict[str, Any], database: Path, database_sha256: str, example_type: str
) -> dict[str, Any]:
    group = f"vehicle:{row['vehicle_id']}/message:{row['message_id']}"
    return {
        "example_id": f"ocb:signal:{row['signal_id']}:{example_type}:{POLICY_VERSION}",
        "example_type": example_type,
        "group": group,
        "source": "local OCB SQLite database",
        "source_database": str(database.resolve()),
        "source_database_sha256": database_sha256,
        "source_table": "signals",
        "signal_id": int(row["signal_id"]),
        "message_id": int(row["message_id"]),
        "vehicle_id": int(row["vehicle_id"]),
        "vehicle_source": clean_text(row["vehicle_source"]),
        "signal_trust": row["signal_trust"],
        "message_trust": row["message_trust"],
        "signal_training_eligible": row["signal_training_eligible"],
        "message_training_eligible": row["message_training_eligible"],
        "vehicle_quality_tier": row["vehicle_tier"],
        "filter_policy": POLICY_VERSION,
        "license": "not specified in database; local user-provided source",
    }


def structured_definition(row: dict[str, Any]) -> dict[str, Any]:
    can_id = int(row["can_id"])
    return {
        "record_trust": {
            "signal": str(row["signal_trust"]),
            "message": str(row["message_trust"]),
            "vehicle_tier": str(row["vehicle_tier"]),
        },
        "vehicle": {
            "year": clean_text(row["year"]),
            "make": clean_text(row["make"]),
            "model": clean_text(row["model"]),
        },
        "frame": {
            "message_name": clean_text(row["message_name"]),
            "can_id_hex": format_can_id(can_id),
            "identifier_format": identifier_kind(can_id),
            "payload_length_bytes": int(row["frame_length"]),
        },
        "signal": {
            "database_signal_id": int(row["signal_id"]),
            "name": clean_text(row["signal_name"]),
            "dbc_start_bit": int(row["start_bit"]),
            "bit_length": int(row["bit_length"]),
            "byte_order": str(row["byte_order"]),
            "byte_order_convention": (
                "Motorola/big-endian"
                if row["byte_order"] == "big_endian"
                else "Intel/little-endian"
            ),
            "signedness": "signed" if int(row["is_signed"]) else "unsigned",
            "scale": format_decimal(as_decimal(row["scale"])),
            "offset": format_decimal(as_decimal(row["offset"])),
            "unit": clean_text(row.get("unit")) or "unitless",
            "declared_physical_min": (
                None
                if row.get("min_val") is None
                else format_decimal(as_decimal(row["min_val"]))
            ),
            "declared_physical_max": (
                None
                if row.get("max_val") is None
                else format_decimal(as_decimal(row["max_val"]))
            ),
        },
    }


def definition_block(definition: dict[str, Any]) -> str:
    return json.dumps(definition, indent=2, ensure_ascii=False, sort_keys=True)


def decimal_physical_limits(row: dict[str, Any]) -> tuple[Decimal, Decimal]:
    raw_low, raw_high = raw_limits(row)
    low_value = physical_value(row, raw_low)
    high_value = physical_value(row, raw_high)
    return min(low_value, high_value), max(low_value, high_value)


def attach_grounding(
    provenance: dict[str, Any], definition: dict[str, Any]
) -> dict[str, Any]:
    canonical = json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    provenance["grounding_definition"] = definition
    provenance["grounding_definition_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    provenance["requires_external_database_recall"] = False
    return provenance


def make_examples(
    row: dict[str, Any],
    signal_values: Sequence[tuple[int, str]],
    database: Path,
    database_sha256: str,
) -> list[Example]:
    can_id = int(row["can_id"])
    can_label = format_can_id(can_id)
    byte_order = "Motorola/big-endian" if row["byte_order"] == "big_endian" else "Intel/little-endian"
    signedness = "signed" if int(row["is_signed"]) else "unsigned"
    unit = clean_text(row.get("unit")) or "unitless"
    group = f"vehicle:{row['vehicle_id']}/message:{row['message_id']}"
    definition = structured_definition(row)
    supplied_record = definition_block(definition)
    positions = dbc_bit_positions(row)
    if positions is None:
        raise AssertionError(f"accepted signal {row['signal_id']} has no valid DBC bit traversal")
    raw_low, raw_high = raw_limits(row)
    physical_low, physical_high = decimal_physical_limits(row)

    validation_user = (
        "Mechanically audit the supplied trusted CAN signal definition. Confirm the identifier "
        "format and payload length, verify that the complete DBC bit traversal stays inside the "
        "payload, state the raw integer domain and representable physical interval, check any "
        "declared physical bounds, and give the decoder equation. Do not infer signal semantics.\n\n"
        "SUPPLIED_TRUSTED_SIGNAL_DEFINITION:\n"
        f"{supplied_record}"
    )
    declared_range = declared_range_text(row)
    validation_answer = (
        f"The definition is mechanically valid. {can_label} is a valid {identifier_kind(can_id)} "
        f"identifier and the payload length is {row['frame_length']} bytes ({row['frame_length'] * 8} "
        f"bits). Starting at DBC bit {row['start_bit']}, the {byte_order} traversal for width "
        f"{row['bit_length']} visits positions {list(positions)}; every position is inside "
        f"0..{row['frame_length'] * 8 - 1}. The {signedness} raw domain is {raw_low}..{raw_high}. "
        f"With `{conversion_text(row)}`, the full representable physical interval is "
        f"{format_decimal(physical_low)}..{format_decimal(physical_high)} {unit}. "
        f"{declared_range} Any declared bounds are compatible with the stored conversion and "
        "contain at least one integer raw value. This validates mechanics only, not undocumented "
        "semantic meaning."
    )
    validation_provenance = attach_grounding(
        source_provenance(row, database, database_sha256, "definition_validation"),
        definition,
    )

    raw_value = choose_raw(row, signal_values)
    payload = encode_payload(row, raw_value)
    decoded = decode_payload(row, payload)
    if decoded != raw_value:
        raise AssertionError(
            f"signal {row['signal_id']} failed generated payload round trip: {raw_value} != {decoded}"
        )
    physical = physical_value(row, raw_value)
    if not physical_in_declared_range(row, physical):
        raise AssertionError(
            f"signal {row['signal_id']} generated physical value {physical} outside "
            f"declared range {row.get('min_val')}..{row.get('max_val')}"
        )
    payload_hex = " ".join(f"{byte:02X}" for byte in payload)
    decode_user = (
        "Decode the named signal from the supplied payload using only the complete trusted "
        "definition below. Report the extracted raw integer, show sign handling and the "
        "raw-to-physical calculation, check the result against any declared range, and do not "
        "infer undocumented semantics.\n\n"
        "SUPPLIED_TRUSTED_SIGNAL_DEFINITION:\n"
        f"{supplied_record}\n\n"
        f"SUPPLIED_PAYLOAD_HEX ({row['frame_length']} bytes): {payload_hex}"
    )
    encoded_field = raw_value & ((1 << int(row["bit_length"])) - 1)
    if int(row["is_signed"]):
        extraction = (
            f"The field bit pattern is unsigned {encoded_field}; {row['bit_length']}-bit "
            f"two's-complement sign extension gives raw {raw_value}."
        )
    else:
        extraction = f"The field is unsigned, so the extracted raw integer is {raw_value}."
    range_check = (
        "No physical bounds are declared, so there is no declared-range comparison."
        if row.get("min_val") is None and row.get("max_val") is None
        else f"The result is inside the supplied declared range. {declared_range_text(row)}"
    )
    decode_answer = (
        f"Following the supplied {byte_order} traversal from DBC bit {row['start_bit']} across "
        f"{row['bit_length']} bits uses positions {list(positions)}. {extraction} Apply "
        f"`{conversion_text(row)}`: {format_decimal(as_decimal(raw_value))} x "
        f"{format_decimal(as_decimal(row['scale']))} "
        f"{'+' if as_decimal(row['offset']) >= 0 else '-'} "
        f"{format_decimal(abs(as_decimal(row['offset'])))} = {format_decimal(physical)} {unit}. "
        f"{range_check}"
    )
    label = enum_label(signal_values, raw_value)
    if label is not None:
        decode_answer += f" The database labels raw {raw_value} as `{label}`."
    decode_provenance = attach_grounding(
        source_provenance(row, database, database_sha256, "decode"), definition
    )
    decode_provenance["generated_raw_value"] = raw_value
    decode_provenance["generated_physical_value"] = format_decimal(physical)
    decode_provenance["declared_physical_min"] = (
        None if row.get("min_val") is None else format_decimal(as_decimal(row["min_val"]))
    )
    decode_provenance["declared_physical_max"] = (
        None if row.get("max_val") is None else format_decimal(as_decimal(row["max_val"]))
    )
    decode_provenance["generated_payload_hex"] = payload_hex
    decode_provenance["generated_payload_policy"] = "target signal bits set; all other bits zero"

    common_key = (int(row["vehicle_id"]), int(row["message_id"]), int(row["signal_id"]))
    return [
        Example(
            group,
            (*common_key, "definition_validation"),
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": validation_user},
                {"role": "assistant", "content": validation_answer},
            ],
            validation_provenance,
        ),
        Example(
            group,
            (*common_key, "decode"),
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": decode_user},
                {"role": "assistant", "content": decode_answer},
            ],
            decode_provenance,
        ),
    ]


def validate_training_row(row: dict[str, Any]) -> None:
    if set(row) != {"messages"} or not isinstance(row["messages"], list):
        raise ValueError("training row must contain only messages")
    messages = row["messages"]
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("training row must end with an assistant message")
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError("each message must contain exactly role and content")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("unsupported message role")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("message content must be non-empty text")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise ValueError("training row needs a user message before the final assistant message")


def audit_generated_ranges(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    total_decode = 0
    bounded_decode = 0
    violations: list[str] = []
    for record in records:
        if record.get("example_type") != "decode":
            continue
        total_decode += 1
        minimum = record.get("declared_physical_min")
        maximum = record.get("declared_physical_max")
        if minimum is None and maximum is None:
            continue
        bounded_decode += 1
        physical = Decimal(str(record["generated_physical_value"]))
        if minimum is not None and physical < Decimal(str(minimum)):
            violations.append(
                f"{record['example_id']}: {physical} is below declared minimum {minimum}"
            )
        if maximum is not None and physical > Decimal(str(maximum)):
            violations.append(
                f"{record['example_id']}: {physical} is above declared maximum {maximum}"
            )
    if violations:
        raise AssertionError("generated physical range audit failed: " + "; ".join(violations))
    return {
        "decode_examples": total_decode,
        "decode_examples_with_declared_bounds": bounded_decode,
        "violations": 0,
    }


def audit_written_range_sidecars(outputs: dict[str, dict[str, Any]]) -> dict[str, int]:
    records: list[dict[str, Any]] = []
    for output in outputs.values():
        provenance_path = Path(str(output["provenance_path"]))
        with provenance_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{provenance_path}:{line_number} is not an object")
                records.append(record)
    return audit_generated_ranges(records)


def audit_grounded_examples(examples: Sequence[Example]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    failures: list[str] = []
    required_frame_fields = {
        "message_name",
        "can_id_hex",
        "identifier_format",
        "payload_length_bytes",
    }
    required_signal_fields = {
        "database_signal_id",
        "name",
        "dbc_start_bit",
        "bit_length",
        "byte_order",
        "byte_order_convention",
        "signedness",
        "scale",
        "offset",
        "unit",
        "declared_physical_min",
        "declared_physical_max",
    }
    for example in examples:
        provenance = example.provenance
        example_id = str(provenance.get("example_id"))
        example_type = str(provenance.get("example_type"))
        type_counts[example_type] += 1
        if example_type not in {"decode", "definition_validation"}:
            failures.append(f"{example_id}: unsupported or recall-oriented task type")
            continue
        definition = provenance.get("grounding_definition")
        if not isinstance(definition, dict):
            failures.append(f"{example_id}: missing grounding definition")
            continue
        frame = definition.get("frame")
        signal = definition.get("signal")
        if not isinstance(frame, dict) or not required_frame_fields <= set(frame):
            failures.append(f"{example_id}: incomplete supplied frame definition")
        if not isinstance(signal, dict) or not required_signal_fields <= set(signal):
            failures.append(f"{example_id}: incomplete supplied signal definition")
        user_messages = [message["content"] for message in example.messages if message["role"] == "user"]
        if len(user_messages) != 1:
            failures.append(f"{example_id}: expected exactly one user prompt")
            continue
        supplied_block = "SUPPLIED_TRUSTED_SIGNAL_DEFINITION:\n" + definition_block(definition)
        if supplied_block not in user_messages[0]:
            failures.append(f"{example_id}: prompt does not contain its complete definition")
        if provenance.get("requires_external_database_recall") is not False:
            failures.append(f"{example_id}: external recall flag is not false")
        if example_type == "decode" and "SUPPLIED_PAYLOAD_HEX" not in user_messages[0]:
            failures.append(f"{example_id}: decode prompt is missing the payload")
    if failures:
        raise AssertionError("grounding audit failed: " + "; ".join(failures))
    return {
        "examples": len(examples),
        "fully_grounded_examples": len(examples),
        "external_database_recall_tasks": 0,
        "failures": 0,
        "example_types": dict(sorted(type_counts.items())),
    }


def numeric_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty token-length sequence")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = max(0, math.ceil(len(ordered) * fraction) - 1)
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def measure_token_lengths(examples: Sequence[Example], tokenizer: Any) -> dict[str, Any]:
    rows: list[tuple[str, int, int, int]] = []
    for example in examples:
        prompt_messages = example.messages[:-1]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
            enable_thinking=False,
        )
        full_ids = tokenizer.apply_chat_template(
            example.messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
            enable_thinking=False,
        )
        if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
            raise ValueError("tokenizer chat template did not return token lists")
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("tokenizer has inconsistent generation and training prefixes")
        rows.append(
            (
                str(example.provenance["example_type"]),
                len(full_ids),
                len(prompt_ids),
                len(full_ids) - len(prompt_ids),
            )
        )

    def summarize(selected: Sequence[tuple[str, int, int, int]]) -> dict[str, Any]:
        return {
            "rows": len(selected),
            "full_chat_tokens": numeric_summary([row[1] for row in selected]),
            "prompt_tokens": numeric_summary([row[2] for row in selected]),
            "assistant_target_tokens": numeric_summary([row[3] for row in selected]),
        }

    types = sorted({row[0] for row in rows})
    return {
        "all": summarize(rows),
        "by_example_type": {
            example_type: summarize([row for row in rows if row[0] == example_type])
            for example_type in types
        },
    }


def write_partition(
    output_dir: Path,
    name: str,
    examples: Iterable[Example],
) -> dict[str, Any]:
    ordered = sorted(examples, key=lambda example: example.sort_key)
    data_path = output_dir / f"{OUTPUT_STEM}.{name}.jsonl"
    provenance_path = output_dir / f"{OUTPUT_STEM}.{name}.provenance.jsonl"
    data_digest = hashlib.sha256()
    provenance_digest = hashlib.sha256()
    type_counts: Counter[str] = Counter()

    with data_path.open("w", encoding="utf-8", newline="\n") as data_stream, provenance_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as provenance_stream:
        for line_number, example in enumerate(ordered, start=1):
            row = {"messages": example.messages}
            validate_training_row(row)
            data_line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            provenance_line = json.dumps(
                {"line": line_number, **example.provenance},
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
            data_stream.write(data_line)
            provenance_stream.write(provenance_line)
            data_digest.update(data_line.encode("utf-8"))
            provenance_digest.update(provenance_line.encode("utf-8"))
            type_counts[str(example.provenance["example_type"])] += 1

    return {
        "data_path": str(data_path.resolve()),
        "data_sha256": data_digest.hexdigest(),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": provenance_digest.hexdigest(),
        "rows": len(ordered),
        "groups": len({example.group for example in ordered}),
        "example_types": dict(sorted(type_counts.items())),
    }


def select_validation_groups(groups: Sequence[str], fraction: float) -> set[str]:
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two vehicle/message groups are required for a leakage-safe split")
    count = max(1, min(len(unique) - 1, math.floor(len(unique) * fraction + 0.5)))
    ranked = sorted(unique, key=lambda group: (stable_int(f"{SPLIT_SEED}|{group}"), group))
    return set(ranked[:count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path.home() / "Documents" / "datasets" / "ocb_ocb.db",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models" / "Qwen3-1.7B",
    )
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error(f"database does not exist: {args.database}")
    if not 0.05 <= args.validation_fraction <= 0.45:
        parser.error("--validation-fraction must be between 0.05 and 0.45")
    if not args.tokenizer.is_dir():
        parser.error(f"local tokenizer directory does not exist: {args.tokenizer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_sha256 = sha256_file(args.database)
    with connect_read_only(args.database) as connection:
        schema = inspect_schema(connection)
        counts = database_counts(connection)
        candidates = load_candidates(connection)
        signal_values = load_signal_values(
            connection, [int(row["signal_id"]) for row in candidates]
        )

    rejection_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        rejection = first_rejection(row)
        if rejection is None:
            eligible.append(row)
        else:
            rejection_counts[rejection] += 1

    overlap_ids, overlap_details = find_overlaps(candidates)
    accepted: list[dict[str, Any]] = []
    for row in eligible:
        if int(row["signal_id"]) in overlap_ids:
            rejection_counts["overlapping_signal_layout"] += 1
        else:
            accepted.append(row)
    if not accepted:
        raise ValueError("strict filters left no trusted CAN signals")

    examples: list[Example] = []
    for row in accepted:
        examples.extend(
            make_examples(
                row,
                signal_values.get(int(row["signal_id"]), ()),
                args.database,
                database_sha256,
            )
        )
    in_memory_range_audit = audit_generated_ranges(
        example.provenance for example in examples
    )
    grounding_audit = audit_grounded_examples(examples)

    validation_groups = select_validation_groups(
        [example.group for example in examples], args.validation_fraction
    )
    train = [example for example in examples if example.group not in validation_groups]
    validation = [example for example in examples if example.group in validation_groups]
    train_groups = {example.group for example in train}
    actual_validation_groups = {example.group for example in validation}
    leaked_groups = sorted(train_groups & actual_validation_groups)
    if leaked_groups:
        raise AssertionError(f"split leaked groups: {leaked_groups}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    token_lengths = {
        "combined": measure_token_lengths(examples, tokenizer),
        "train": measure_token_lengths(train, tokenizer),
        "validation": measure_token_lengths(validation, tokenizer),
    }

    outputs = {
        "train": write_partition(args.output_dir, "train", train),
        "validation": write_partition(args.output_dir, "validation", validation),
    }
    written_range_audit = audit_written_range_sidecars(outputs)
    if written_range_audit != in_memory_range_audit:
        raise AssertionError(
            "written provenance range audit differs from the in-memory generation audit"
        )
    vehicle_counts = Counter(format_vehicle(row) for row in accepted)
    report = {
        "policy_version": POLICY_VERSION,
        "source": {
            "database": str(args.database.resolve()),
            "sha256": database_sha256,
            "opened_read_only": True,
            "license": "not specified in database; local user-provided source",
            "database_counts": counts,
            "inspected_schema": schema,
        },
        "filter": {
            "sql_candidates": len(candidates),
            "requirements": [
                "signal and message data_trust=trusted",
                "signal and message training_eligible=gold",
                "vehicle_quality tier=gold",
                "CAN protocol with numeric 11/29-bit identifier",
                "valid payload length and in-frame DBC bit traversal",
                "non-placeholder signal name and useful non-weak tags",
                "finite nonzero scale, finite offset, coherent declared physical range",
                "non-speculative signal description",
                "no overlapping signal layouts within a message",
            ],
            "rejections": dict(sorted(rejection_counts.items())),
            "accepted_signals": len(accepted),
            "accepted_messages": len({int(row["message_id"]) for row in accepted}),
            "accepted_vehicles": dict(sorted(vehicle_counts.items())),
            "overlap_details": overlap_details,
        },
        "generation": {
            "question_types": ["grounded definition validation", "grounded payload-to-physical decode"],
            "physical_to_raw_encoding_questions": 0,
            "examples_per_signal": 2,
            "grounding_audit": grounding_audit,
            "raw_selection_policy": (
                "deterministic nonzero/interior raw value constrained to any declared physical bounds"
            ),
            "synthetic_payload_policy": "target signal bits set; all other bits zero; round-trip verified",
            "declared_range_audit": written_range_audit,
        },
        "tokenization": {
            "tokenizer": str(args.tokenizer.resolve()),
            "chat_template_kwargs": {"enable_thinking": False},
            "lengths": token_lengths,
        },
        "split": {
            "seed": SPLIT_SEED,
            "requested_validation_fraction": args.validation_fraction,
            "group_key": "vehicle_id/message_id",
            "train_groups": len(train_groups),
            "validation_groups": len(actual_validation_groups),
            "group_leakage": leaked_groups,
        },
        "outputs": outputs,
        "limitations": [
            "The database has no explicit extended-frame flag; identifier format is inferred from numeric CAN ID range.",
            "Internal trusted/gold labels and source filenames are retained evidence, not independent external verification.",
            "The database does not specify an upstream dataset license; training rights must be verified by the user.",
            "Decode payloads are deterministically synthesized from stored definitions, not copied from live captures.",
            "Declared min/max fields are treated as physical bounds; rows inconsistent with their conversion are excluded.",
            "Every task is self-contained; held-out message groups test transformation rather than database recall.",
            "Coverage is intentionally narrow because all accepted rows must survive every strict quality gate.",
        ],
    }
    report_path = args.output_dir / f"{OUTPUT_STEM}.report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "accepted_signals": len(accepted),
                "train_rows": outputs["train"]["rows"],
                "validation_rows": outputs["validation"]["rows"],
                "train_groups": outputs["train"]["groups"],
                "validation_groups": outputs["validation"]["groups"],
                "max_full_chat_tokens": token_lengths["combined"]["all"]["full_chat_tokens"]["max"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
