"""Expand grounded-CAN v3 training decodes into a deterministic value curriculum.

This builder intentionally reads only the existing v3 train/provenance artifacts.  It
never opens the source database and never rewrites the v3 train or validation files.
Each definition-audit training row is retained once, while each training decode
definition is regenerated at up to eight distinct legal raw values.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


POLICY_VERSION = "grounded-can-curriculum-v4"
SOURCE_POLICY_VERSION = "grounded-can-v3"
OUTPUT_STEM = "grounded-can-curriculum"
TARGET_VALUES_PER_SIGNAL = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def validate_training_row(row: dict[str, Any]) -> None:
    if set(row) != {"messages"} or not isinstance(row["messages"], list):
        raise ValueError("training rows must contain exactly a messages list")
    messages = row["messages"]
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("training row must end with an assistant message")
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError("messages must contain exactly role and content")
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {message['role']}")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("message content must be non-empty text")


def aligned_records(data_path: Path, provenance_path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    data = load_jsonl(data_path)
    provenance = load_jsonl(provenance_path)
    if len(data) != len(provenance):
        raise ValueError(
            f"data/provenance row mismatch: {len(data)} data versus {len(provenance)} provenance"
        )
    aligned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for line_number, (row, source) in enumerate(zip(data, provenance), start=1):
        validate_training_row(row)
        if source.get("line") != line_number:
            raise ValueError(f"provenance line mismatch at row {line_number}")
        if source.get("filter_policy") != SOURCE_POLICY_VERSION:
            raise ValueError(
                f"row {line_number} is not from {SOURCE_POLICY_VERSION}: "
                f"{source.get('filter_policy')!r}"
            )
        definition = source.get("grounding_definition")
        if not isinstance(definition, dict):
            raise ValueError(f"row {line_number} lacks an embedded grounding definition")
        if canonical_sha256(definition) != source.get("grounding_definition_sha256"):
            raise ValueError(f"row {line_number} grounding definition hash mismatch")
        aligned.append((row, source))
    return aligned


def as_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def definition_parts(definition: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = definition.get("frame")
    signal = definition.get("signal")
    if not isinstance(frame, dict) or not isinstance(signal, dict):
        raise ValueError("grounding definition needs frame and signal objects")
    required_frame = {"payload_length_bytes"}
    required_signal = {
        "database_signal_id",
        "dbc_start_bit",
        "bit_length",
        "byte_order",
        "signedness",
        "scale",
        "offset",
        "unit",
        "declared_physical_min",
        "declared_physical_max",
    }
    if not required_frame <= set(frame) or not required_signal <= set(signal):
        raise ValueError("grounding definition is incomplete")
    return frame, signal


def raw_limits(signal: dict[str, Any]) -> tuple[int, int]:
    width = int(signal["bit_length"])
    if width <= 0:
        raise ValueError("bit length must be positive")
    if signal["signedness"] == "signed":
        return -(1 << (width - 1)), (1 << (width - 1)) - 1
    if signal["signedness"] != "unsigned":
        raise ValueError(f"unsupported signedness: {signal['signedness']!r}")
    return 0, (1 << width) - 1


def allowed_raw_limits(signal: dict[str, Any]) -> tuple[int, int]:
    low, high = raw_limits(signal)
    scale = as_decimal(signal["scale"])
    offset = as_decimal(signal["offset"])
    if not scale.is_finite() or scale == 0 or not offset.is_finite():
        raise ValueError("scale and offset must be finite and scale must be nonzero")
    minimum = signal.get("declared_physical_min")
    maximum = signal.get("declared_physical_max")
    with localcontext() as context:
        context.prec = 80
        if scale > 0:
            if minimum is not None:
                boundary = (as_decimal(minimum) - offset) / scale
                low = max(low, int(boundary.to_integral_value(rounding=ROUND_CEILING)))
            if maximum is not None:
                boundary = (as_decimal(maximum) - offset) / scale
                high = min(high, int(boundary.to_integral_value(rounding=ROUND_FLOOR)))
        else:
            if maximum is not None:
                boundary = (as_decimal(maximum) - offset) / scale
                low = max(low, int(boundary.to_integral_value(rounding=ROUND_CEILING)))
            if minimum is not None:
                boundary = (as_decimal(minimum) - offset) / scale
                high = min(high, int(boundary.to_integral_value(rounding=ROUND_FLOOR)))
    if low > high:
        raise ValueError("declared physical bounds contain no integer raw value")
    return low, high


def select_raw_values(signal: dict[str, Any], target: int = TARGET_VALUES_PER_SIGNAL) -> list[int]:
    """Select boundaries, sign probes, and spread-out interiors deterministically."""

    low, high = allowed_raw_limits(signal)
    population = high - low + 1
    wanted = min(target, population)
    selected: list[int] = []

    def add(value: int) -> None:
        if low <= value <= high and value not in selected and len(selected) < wanted:
            selected.append(value)

    # Domain/declared-range boundaries are always represented.  Values around zero
    # explicitly teach two's-complement sign behavior whenever those values are legal.
    for value in (low, high, -1, 0, 1, low + 1, high - 1):
        add(value)

    if low < 0:
        add((low + min(high, -1)) // 2)
    if high > 0:
        add((max(low, 1) + high) // 2)
    add((low + high) // 2)

    span = high - low
    for numerator in range(1, target):
        add(low + (span * numerator) // target)

    # Quantiles can collapse for narrow domains.  A stable modular walk fills any
    # remaining slots without enumerating a potentially 64-bit integer interval.
    if len(selected) < wanted:
        start = stable_int(f"{POLICY_VERSION}|{signal['database_signal_id']}|start") % population
        step = stable_int(f"{POLICY_VERSION}|{signal['database_signal_id']}|step") % population
        step = step or 1
        while math.gcd(step, population) != 1:
            step += 1
            if step >= population:
                step = 1
        for index in range(population):
            add(low + ((start + index * step) % population))
            if len(selected) == wanted:
                break

    if len(selected) != wanted:
        raise AssertionError(
            f"signal {signal['database_signal_id']} selected {len(selected)} of {wanted} values"
        )
    return selected


def dbc_bit_positions(frame: dict[str, Any], signal: dict[str, Any]) -> tuple[int, ...]:
    frame_bits = int(frame["payload_length_bytes"]) * 8
    width = int(signal["bit_length"])
    position = int(signal["dbc_start_bit"])
    byte_order = signal["byte_order"]
    if byte_order not in {"big_endian", "little_endian"}:
        raise ValueError(f"unsupported byte order: {byte_order!r}")
    positions: list[int] = []
    for _ in range(width):
        if not 0 <= position < frame_bits:
            raise ValueError("DBC traversal leaves the supplied payload")
        positions.append(position)
        if byte_order == "little_endian":
            position += 1
        else:
            position = position + 15 if position % 8 == 0 else position - 1
    return tuple(positions)


def encode_payload(frame: dict[str, Any], signal: dict[str, Any], raw_value: int) -> bytes:
    positions = dbc_bit_positions(frame, signal)
    width = int(signal["bit_length"])
    encoded = raw_value & ((1 << width) - 1)
    payload = bytearray(int(frame["payload_length_bytes"]))
    for field_index, dbc_position in enumerate(positions):
        shift = width - 1 - field_index if signal["byte_order"] == "big_endian" else field_index
        if (encoded >> shift) & 1:
            payload[dbc_position // 8] |= 1 << (dbc_position % 8)
    return bytes(payload)


def decode_payload(frame: dict[str, Any], signal: dict[str, Any], payload: bytes) -> int:
    positions = dbc_bit_positions(frame, signal)
    encoded = 0
    if signal["byte_order"] == "big_endian":
        for dbc_position in positions:
            encoded = (encoded << 1) | ((payload[dbc_position // 8] >> (dbc_position % 8)) & 1)
    else:
        for field_index, dbc_position in enumerate(positions):
            encoded |= ((payload[dbc_position // 8] >> (dbc_position % 8)) & 1) << field_index
    width = int(signal["bit_length"])
    if signal["signedness"] == "signed" and encoded & (1 << (width - 1)):
        encoded -= 1 << width
    return encoded


def physical_value(signal: dict[str, Any], raw_value: int) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return as_decimal(raw_value) * as_decimal(signal["scale"]) + as_decimal(signal["offset"])


def physical_is_legal(signal: dict[str, Any], value: Decimal) -> bool:
    minimum = signal.get("declared_physical_min")
    maximum = signal.get("declared_physical_max")
    return not (
        (minimum is not None and value < as_decimal(minimum))
        or (maximum is not None and value > as_decimal(maximum))
    )


def definition_block(definition: dict[str, Any]) -> str:
    return json.dumps(definition, indent=2, ensure_ascii=False, sort_keys=True)


def declared_range_text(signal: dict[str, Any]) -> str:
    minimum = signal.get("declared_physical_min")
    maximum = signal.get("declared_physical_max")
    unit = str(signal["unit"])
    if minimum is None and maximum is None:
        return "No physical range is declared."
    if minimum is not None and maximum is not None:
        return f"The declared physical range is {minimum} to {maximum} {unit}."
    relation = "at least" if minimum is not None else "at most"
    bound = minimum if minimum is not None else maximum
    return f"The declared physical range is {relation} {bound} {unit}."


def conversion_text(signal: dict[str, Any]) -> str:
    offset = as_decimal(signal["offset"])
    sign = "+" if offset >= 0 else "-"
    return f"physical = raw x {signal['scale']} {sign} {format_decimal(abs(offset))}"


def make_decode_messages(
    system_prompt: str, definition: dict[str, Any], raw_value: int
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    frame, signal = definition_parts(definition)
    positions = dbc_bit_positions(frame, signal)
    payload = encode_payload(frame, signal, raw_value)
    decoded = decode_payload(frame, signal, payload)
    if decoded != raw_value:
        raise AssertionError(
            f"signal {signal['database_signal_id']} round-trip failure: {raw_value} != {decoded}"
        )
    physical = physical_value(signal, raw_value)
    if not physical_is_legal(signal, physical):
        raise AssertionError(
            f"signal {signal['database_signal_id']} generated illegal physical value {physical}"
        )
    payload_hex = " ".join(f"{byte:02X}" for byte in payload)
    supplied_record = definition_block(definition)
    user = (
        "Decode the named signal from the supplied payload using only the complete trusted "
        "definition below. Report the extracted raw integer, show sign handling and the "
        "raw-to-physical calculation, check the result against any declared range, and do not "
        "infer undocumented semantics.\n\n"
        "SUPPLIED_TRUSTED_SIGNAL_DEFINITION:\n"
        f"{supplied_record}\n\n"
        f"SUPPLIED_PAYLOAD_HEX ({frame['payload_length_bytes']} bytes): {payload_hex}"
    )
    byte_order = (
        "Motorola/big-endian" if signal["byte_order"] == "big_endian" else "Intel/little-endian"
    )
    width = int(signal["bit_length"])
    encoded_field = raw_value & ((1 << width) - 1)
    if signal["signedness"] == "signed":
        extraction = (
            f"The field bit pattern is unsigned {encoded_field}; {width}-bit two's-complement "
            f"sign extension gives raw {raw_value}."
        )
    else:
        extraction = f"The field is unsigned, so the extracted raw integer is {raw_value}."
    if signal.get("declared_physical_min") is None and signal.get("declared_physical_max") is None:
        range_check = "No physical bounds are declared, so there is no declared-range comparison."
    else:
        range_check = f"The result is inside the supplied declared range. {declared_range_text(signal)}"
    offset = as_decimal(signal["offset"])
    answer = (
        f"Following the supplied {byte_order} traversal from DBC bit {signal['dbc_start_bit']} "
        f"across {width} bits uses positions {list(positions)}. {extraction} Apply "
        f"`{conversion_text(signal)}`: {raw_value} x {signal['scale']} "
        f"{'+' if offset >= 0 else '-'} {format_decimal(abs(offset))} = "
        f"{format_decimal(physical)} {signal['unit']}. {range_check}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]
    metadata = {
        "generated_raw_value": raw_value,
        "generated_physical_value": format_decimal(physical),
        "generated_payload_hex": payload_hex,
        "generated_payload_policy": "target signal bits set; all other bits zero",
        "round_trip_verified": True,
    }
    return messages, metadata


def make_provenance(
    source: dict[str, Any], example_type: str, ordinal: int | None = None
) -> dict[str, Any]:
    result = {key: value for key, value in source.items() if key != "line"}
    parent_id = str(source["example_id"])
    result["parent_example_id"] = parent_id
    result["filter_policy"] = POLICY_VERSION
    result["curriculum_source_policy"] = SOURCE_POLICY_VERSION
    result["example_type"] = example_type
    if ordinal is None:
        result["example_id"] = parent_id.replace(SOURCE_POLICY_VERSION, POLICY_VERSION)
        result["curriculum_role"] = "preserved_definition_audit"
    else:
        signal_id = int(source["signal_id"])
        result["example_id"] = f"ocb:signal:{signal_id}:decode:value-{ordinal:02d}:{POLICY_VERSION}"
        result["curriculum_role"] = "expanded_decode"
        result["curriculum_value_ordinal"] = ordinal
    for stale_key in (
        "generated_raw_value",
        "generated_physical_value",
        "generated_payload_hex",
        "generated_payload_policy",
    ):
        result.pop(stale_key, None)
    return result


def numeric_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def measure_token_lengths(rows: Sequence[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    measurements: list[tuple[str, int, int, int]] = []
    for row in rows:
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
            enable_thinking=False,
        )
        if full[: len(prompt)] != prompt:
            raise ValueError("tokenizer produced inconsistent prompt/training prefixes")
        example_type = "definition_validation" if "Mechanically audit" in messages[1]["content"] else "decode"
        measurements.append((example_type, len(full), len(prompt), len(full) - len(prompt)))

    def summarize(items: Sequence[tuple[str, int, int, int]]) -> dict[str, Any]:
        return {
            "rows": len(items),
            "full_chat_tokens": numeric_summary([item[1] for item in items]),
            "prompt_tokens": numeric_summary([item[2] for item in items]),
            "assistant_target_tokens": numeric_summary([item[3] for item in items]),
        }

    types = sorted({item[0] for item in measurements})
    return {
        "all": summarize(measurements),
        "by_example_type": {
            example_type: summarize(
                [item for item in measurements if item[0] == example_type]
            )
            for example_type in types
        },
    }


def write_jsonl_pair(
    data_path: Path,
    provenance_path: Path,
    records: Iterable[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    data_hash = hashlib.sha256()
    provenance_hash = hashlib.sha256()
    row_count = 0
    type_counts: Counter[str] = Counter()
    groups: set[str] = set()
    with data_path.open("w", encoding="utf-8", newline="\n") as data_stream, provenance_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as provenance_stream:
        for row_count, (row, provenance) in enumerate(records, start=1):
            validate_training_row(row)
            data_line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            provenance_line = json.dumps(
                {"line": row_count, **provenance},
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
            data_stream.write(data_line)
            provenance_stream.write(provenance_line)
            data_hash.update(data_line.encode("utf-8"))
            provenance_hash.update(provenance_line.encode("utf-8"))
            type_counts[str(provenance["example_type"])] += 1
            groups.add(str(provenance["group"]))
    return {
        "data_path": str(data_path.resolve()),
        "data_sha256": data_hash.hexdigest(),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": provenance_hash.hexdigest(),
        "rows": row_count,
        "groups": len(groups),
        "example_types": dict(sorted(type_counts.items())),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data", type=Path, default=here / "data" / "grounded-can.train.jsonl")
    parser.add_argument(
        "--input-provenance",
        type=Path,
        default=here / "data" / "grounded-can.train.provenance.jsonl",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=here / "data" / "grounded-can.validation.jsonl",
    )
    parser.add_argument(
        "--validation-provenance",
        type=Path,
        default=here / "data" / "grounded-can.validation.provenance.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "data")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=here.parents[1] / "models" / "Qwen3-1.7B",
    )
    parser.add_argument("--values-per-signal", type=int, default=TARGET_VALUES_PER_SIGNAL)
    args = parser.parse_args()
    input_paths = (
        args.input_data,
        args.input_provenance,
        args.validation_data,
        args.validation_provenance,
    )
    for path in input_paths:
        if not path.is_file():
            parser.error(f"required v3 artifact does not exist: {path}")
    if not args.tokenizer.is_dir():
        parser.error(f"local tokenizer does not exist: {args.tokenizer}")
    if not 2 <= args.values_per_signal <= 32:
        parser.error("--values-per-signal must be between 2 and 32")

    source_hashes_before = {str(path.resolve()): sha256_file(path) for path in input_paths}
    aligned = aligned_records(args.input_data, args.input_provenance)
    validation_provenance = load_jsonl(args.validation_provenance)
    validation_groups = {str(row["group"]) for row in validation_provenance}
    validation_signals = {int(row["signal_id"]) for row in validation_provenance}
    validation_definitions = {str(row["grounding_definition_sha256"]) for row in validation_provenance}

    by_signal: dict[int, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row, provenance in aligned:
        signal_id = int(provenance["signal_id"])
        example_type = str(provenance["example_type"])
        if example_type not in {"decode", "definition_validation"}:
            raise ValueError(f"unsupported source example type {example_type!r}")
        bucket = by_signal.setdefault(signal_id, {})
        if example_type in bucket:
            raise ValueError(f"signal {signal_id} has duplicate {example_type} source rows")
        bucket[example_type] = (row, provenance)

    output_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    coverage_rows: list[dict[str, Any]] = []
    round_trip_checks = 0
    range_checks = 0
    preserved_definition_rows = 0
    for signal_id in sorted(by_signal):
        source_rows = by_signal[signal_id]
        if set(source_rows) != {"decode", "definition_validation"}:
            raise ValueError(f"signal {signal_id} does not have exactly one source row of each type")
        definition_row, definition_source = source_rows["definition_validation"]
        decode_row, decode_source = source_rows["decode"]
        if definition_source["grounding_definition_sha256"] != decode_source["grounding_definition_sha256"]:
            raise ValueError(f"signal {signal_id} source rows disagree on their definition")

        # Retain the original definition-audit conversation exactly once.
        audit_provenance = make_provenance(definition_source, "definition_validation")
        output_records.append((definition_row, audit_provenance))
        preserved_definition_rows += 1

        definition = decode_source["grounding_definition"]
        frame, signal = definition_parts(definition)
        low, high = allowed_raw_limits(signal)
        raw_values = select_raw_values(signal, args.values_per_signal)
        payloads: set[str] = set()
        for ordinal, raw_value in enumerate(raw_values, start=1):
            messages, generated = make_decode_messages(
                decode_row["messages"][0]["content"], definition, raw_value
            )
            provenance = make_provenance(decode_source, "decode", ordinal)
            provenance.update(generated)
            provenance["declared_physical_min"] = signal.get("declared_physical_min")
            provenance["declared_physical_max"] = signal.get("declared_physical_max")
            provenance["allowed_raw_min"] = low
            provenance["allowed_raw_max"] = high
            provenance["curriculum_target_values"] = args.values_per_signal
            output_records.append(({"messages": messages}, provenance))
            if generated["generated_payload_hex"] in payloads:
                raise AssertionError(f"signal {signal_id} produced duplicate payloads")
            payloads.add(str(generated["generated_payload_hex"]))
            round_trip_checks += 1
            range_checks += 1

        coverage_rows.append(
            {
                "signal_id": signal_id,
                "allowed_raw_min": low,
                "allowed_raw_max": high,
                "legal_raw_population": high - low + 1,
                "selected_raw_values": raw_values,
                "selected_count": len(raw_values),
                "target_count": args.values_per_signal,
                "lower_boundary_covered": low in raw_values,
                "upper_boundary_covered": high in raw_values,
                "zero_covered_when_legal": not (low <= 0 <= high) or 0 in raw_values,
                "negative_covered_when_legal": not (low < 0) or any(value < 0 for value in raw_values),
                "positive_covered_when_legal": not (high > 0) or any(value > 0 for value in raw_values),
                "signed_negative_sign_bit_covered": (
                    signal["signedness"] != "signed"
                    or not (low < 0)
                    or any(value < 0 for value in raw_values)
                ),
            }
        )

    output_records.sort(
        key=lambda item: (
            int(item[1]["signal_id"]),
            0 if item[1]["example_type"] == "definition_validation" else 1,
            int(item[1].get("curriculum_value_ordinal", 0)),
        )
    )
    train_groups = {str(provenance["group"]) for _, provenance in output_records}
    train_signals = {int(provenance["signal_id"]) for _, provenance in output_records}
    train_definitions = {
        str(provenance["grounding_definition_sha256"]) for _, provenance in output_records
    }
    overlaps = {
        "group": sorted(train_groups & validation_groups),
        "signal_id": sorted(train_signals & validation_signals),
        "grounding_definition_sha256": sorted(train_definitions & validation_definitions),
    }
    if any(overlaps.values()):
        raise AssertionError(f"v4 train overlaps untouched v3 validation: {overlaps}")
    if not all(
        row[check]
        for row in coverage_rows
        for check in (
            "lower_boundary_covered",
            "upper_boundary_covered",
            "zero_covered_when_legal",
            "negative_covered_when_legal",
            "positive_covered_when_legal",
            "signed_negative_sign_bit_covered",
        )
    ):
        raise AssertionError("raw-value curriculum coverage audit failed")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    token_lengths = measure_token_lengths([row for row, _ in output_records], tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{OUTPUT_STEM}.train.jsonl"
    provenance_path = args.output_dir / f"{OUTPUT_STEM}.train.provenance.jsonl"
    outputs = write_jsonl_pair(data_path, provenance_path, output_records)

    source_hashes_after = {str(path.resolve()): sha256_file(path) for path in input_paths}
    if source_hashes_before != source_hashes_after:
        raise AssertionError("a v3 source artifact changed while building the curriculum")

    counts = [row["selected_count"] for row in coverage_rows]
    short_signals = [row for row in coverage_rows if row["selected_count"] < args.values_per_signal]
    report = {
        "policy_version": POLICY_VERSION,
        "source": {
            "policy_version": SOURCE_POLICY_VERSION,
            "artifacts_read_only": True,
            "artifacts": [
                {"path": path, "sha256_before": digest, "sha256_after": source_hashes_after[path]}
                for path, digest in source_hashes_before.items()
            ],
            "database_opened": False,
            "source_train_rows": len(aligned),
            "source_train_signals": len(by_signal),
        },
        "generation": {
            "target_decode_values_per_signal": args.values_per_signal,
            "definition_audit_rows_preserved_once": preserved_definition_rows,
            "decode_rows_generated": round_trip_checks,
            "raw_value_selection": (
                "legal interval boundaries, values around zero, signed negative/positive interiors, "
                "spread quantiles, then a deterministic coprime modular fill"
            ),
            "selected_values_per_signal": numeric_summary(counts),
            "signals_below_target_due_to_legal_domain_size": len(short_signals),
            "below_target_details": short_signals,
            "round_trip_checks": round_trip_checks,
            "round_trip_failures": 0,
            "declared_range_checks": range_checks,
            "declared_range_failures": 0,
            "unique_raw_values_per_signal": True,
            "unique_payloads_per_signal": True,
            "coverage_audit_failures": 0,
            "coverage_by_signal": coverage_rows,
        },
        "split_audit": {
            "validation_artifacts_reused": False,
            "validation_artifacts_untouched": True,
            "validation_data_sha256": source_hashes_after[str(args.validation_data.resolve())],
            "validation_provenance_sha256": source_hashes_after[
                str(args.validation_provenance.resolve())
            ],
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "group_overlap": overlaps["group"],
            "signal_id_overlap": overlaps["signal_id"],
            "definition_hash_overlap": overlaps["grounding_definition_sha256"],
        },
        "tokenization": {
            "tokenizer": str(args.tokenizer.resolve()),
            "chat_template_kwargs": {"enable_thinking": False},
            "lengths": token_lengths,
        },
        "outputs": outputs,
        "recommended_recipe": {
            "dataset": "data/grounded-can-curriculum.train.jsonl",
            "expert_name": "grounded-can-curriculum",
            "max_length": max(640, int(token_lengths["all"]["full_chat_tokens"]["max"])),
            "epochs": 2,
            "learning_rate": 0.00005,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "gradient_checkpointing": True,
            "seed": 50,
            "validation": "keep data/grounded-can.validation.jsonl unchanged",
        },
        "limitations": [
            "Payloads are synthetic and set all non-target bits to zero.",
            "The curriculum expands transformation coverage, not underlying signal-definition breadth.",
            "Signals with fewer legal raw integers than the target receive every legal value once.",
            "Upstream licensing remains whatever was recorded in the v3 provenance.",
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
                "signals": len(by_signal),
                "train_rows": outputs["rows"],
                "definition_audits": preserved_definition_rows,
                "decode_rows": round_trip_checks,
                "signals_below_target": len(short_signals),
                "max_full_chat_tokens": token_lengths["all"]["full_chat_tokens"]["max"],
                "data_sha256": outputs["data_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
