"""Direct-generation evaluation for the self-contained grounded CAN expert.

The primary metric is normalized whole-answer exact match.  The field metrics are
deliberately less brittle: they check the CAN calculations and mechanical claims
rather than requiring the reference answer's prose.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import torch

from molor.experimental.mixture import BASE_ROUTE, MixtureEngine
from molor.training.data import ChatExample, load_jsonl


NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
POSITIONS_PATTERN = re.compile(r"(?is)\bpositions?\s*(?:are|is|=|:)?\s*\[([^]]*)]")
RAW_PATTERN = re.compile(
    rf"(?i)\braw(?:\s+(?:integer|value))?\s*(?:is|=|:|of)?\s*({NUMBER})(?!\w)"
)
SCALE_PATTERN = re.compile(rf"(?i)\bscale\s*(?:is|=|:)?\s*({NUMBER})(?!\w)")
OFFSET_PATTERN = re.compile(rf"(?i)\boffset\s*(?:is|=|:)?\s*({NUMBER})(?!\w)")
UNSIGNED_FIELD_PATTERN = re.compile(
    rf"(?i)\b(?:field\s+bit\s+pattern\s+is\s+unsigned|"
    rf"unsigned(?:\s+field(?:\s+(?:bit\s+)?pattern)?|\s+integer|\s+value)?)"
    rf"\s*(?:is|=|:)?\s*({NUMBER})(?!\w)"
)
RAW_DOMAIN_PATTERN = re.compile(
    rf"(?i)\braw(?:\s+domain)?\s*(?:is|=|:)?\s*({NUMBER})\s*\.\.\s*({NUMBER})"
)
PHYSICAL_INTERVAL_PATTERN = re.compile(
    rf"(?i)\brepresentable\s+physical\s+interval\s+(?:is|=|:)\s*"
    rf"({NUMBER})\s*\.\.\s*({NUMBER})"
)
DECLARED_RANGE_PATTERN = re.compile(
    rf"(?i)\bdeclared\s+(?:physical\s+)?range\s+(?:is|=|:)\s*"
    rf"({NUMBER})\s*(?:\.\.|\bto\b)\s*({NUMBER})"
)
PHYSICAL_RESULT_PATTERN = re.compile(rf"(?i)=\s*({NUMBER})(?:\s+([^.;\n]+))?")
EQUATION_PATTERN = re.compile(
    rf"(?i)\bphysical\s*=\s*raw\s*(?:x|\*|×)\s*({NUMBER})\s*([+-])\s*({NUMBER})"
)
DEFINITION_MARKERS = (
    "SUPPLIED_TRUSTED_SIGNAL_DEFINITION:",
    "SUPPLIED_SIGNAL_DEFINITION:",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("×", "x").replace("−", "-")
    return " ".join(text.casefold().split())


def decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def decimal_equal(left: Decimal, right: Decimal) -> bool:
    tolerance = max(Decimal("1e-12"), abs(right) * Decimal("1e-12"))
    return abs(left - right) <= tolerance


def parse_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def parse_definition(prompt: str) -> dict[str, Any]:
    marker = next((candidate for candidate in DEFINITION_MARKERS if candidate in prompt), None)
    if marker is None:
        raise ValueError("prompt is missing a supported supplied signal-definition marker")
    tail = prompt.split(marker, maxsplit=1)[1].lstrip()
    definition, _ = json.JSONDecoder().raw_decode(tail)
    if not isinstance(definition, dict):
        raise ValueError("supplied signal definition is not an object")
    return definition


def task_type(prompt: str, provenance: dict[str, Any] | None = None) -> str:
    if provenance is not None:
        kind = provenance.get("example_type")
        if kind in {"decode", "definition_validation", "location"}:
            return str(kind)
    if prompt.startswith("Decode the named signal"):
        return "decode"
    if prompt.startswith("Mechanically audit"):
        return "definition_validation"
    raise ValueError("unsupported grounded CAN prompt type")


def definition_from_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    row = provenance.get("source_row")
    if not isinstance(row, dict):
        raise ValueError("location example provenance is missing source_row")
    required = {
        "signal_name",
        "start_bit",
        "bit_length",
        "scale",
        "offset",
        "byte_order",
        "is_signed",
        "can_id",
        "message_name",
        "dlc",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"location source_row is missing fields: {', '.join(missing)}")

    can_id = int(row["can_id"])
    width = int(row["bit_length"])
    signed = bool(row["is_signed"])
    raw_low = -(1 << (width - 1)) if signed else 0
    raw_high = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
    return {
        "vehicle": {
            key: row.get(key) for key in ("make", "model", "year", "platform")
        },
        "frame": {
            "message_name": str(row["message_name"]),
            "can_id_hex": f"0x{can_id:03X}" if can_id <= 0x7FF else f"0x{can_id:08X}",
            "can_id_decimal": can_id,
            "identifier_format": (
                "standard 11-bit" if can_id <= 0x7FF else "extended 29-bit"
            ),
            "payload_length_bytes": int(row["dlc"]),
        },
        "signal": {
            "name": str(row["signal_name"]),
            "dbc_start_bit": int(row["start_bit"]),
            "bit_length": width,
            "byte_order": str(row["byte_order"]),
            "signedness": "signed" if signed else "unsigned",
            "raw_min": raw_low,
            "raw_max": raw_high,
            "scale": str(row["scale"]),
            "offset": str(row["offset"]),
            "unit": str(row.get("unit") or "unitless"),
        },
    }


def dbc_positions(definition: dict[str, Any]) -> list[int]:
    signal = definition["signal"]
    position = int(signal["dbc_start_bit"])
    positions: list[int] = []
    for _ in range(int(signal["bit_length"])):
        positions.append(position)
        if signal["byte_order"] == "little_endian":
            position += 1
        elif signal["byte_order"] == "big_endian":
            position = position + 15 if position % 8 == 0 else position - 1
        else:
            raise ValueError(f"unsupported byte order: {signal['byte_order']}")
    return positions


def raw_limits(definition: dict[str, Any]) -> tuple[int, int]:
    signal = definition["signal"]
    width = int(signal["bit_length"])
    if signal["signedness"] == "signed":
        return -(1 << (width - 1)), (1 << (width - 1)) - 1
    return 0, (1 << width) - 1


def physical_limits(definition: dict[str, Any]) -> tuple[Decimal, Decimal]:
    signal = definition["signal"]
    low, high = raw_limits(definition)
    scale, offset = decimal(signal["scale"]), decimal(signal["offset"])
    values = (decimal(low) * scale + offset, decimal(high) * scale + offset)
    return min(values), max(values)


def decode_truth(prompt: str, definition: dict[str, Any]) -> tuple[int, int, Decimal]:
    payload_match = re.search(r"(?im)^SUPPLIED_PAYLOAD_HEX\s*\([^)]*\)\s*:\s*(.+)$", prompt)
    if payload_match is None:
        raise ValueError("decode prompt is missing its supplied payload")
    payload = bytes.fromhex(payload_match.group(1).strip())
    signal = definition["signal"]
    positions = dbc_positions(definition)
    encoded = 0
    if signal["byte_order"] == "big_endian":
        for position in positions:
            encoded = (encoded << 1) | ((payload[position // 8] >> (position % 8)) & 1)
    else:
        for field_index, position in enumerate(positions):
            encoded |= ((payload[position // 8] >> (position % 8)) & 1) << field_index
    width = int(signal["bit_length"])
    raw = encoded
    if signal["signedness"] == "signed" and encoded & (1 << (width - 1)):
        raw -= 1 << width
    physical = decimal(raw) * decimal(signal["scale"]) + decimal(signal["offset"])
    return encoded, raw, physical


def parsed_positions(text: str) -> list[int] | None:
    match = POSITIONS_PATTERN.search(text)
    if match is None:
        return None
    try:
        return [int(value) for value in re.findall(r"-?\d+", match.group(1))]
    except ValueError:
        return None


def contextual_decimal(pattern: re.Pattern[str], text: str, group: int = 1) -> Decimal | None:
    match = pattern.search(text)
    return None if match is None else parse_decimal(match.group(group))


def equation_matches(text: str, scale: Decimal, offset: Decimal) -> bool:
    for match in EQUATION_PATTERN.finditer(text):
        parsed_scale = parse_decimal(match.group(1))
        parsed_offset = parse_decimal(match.group(3))
        if parsed_scale is None or parsed_offset is None:
            continue
        if match.group(2) == "-":
            parsed_offset = -parsed_offset
        if decimal_equal(parsed_scale, scale) and decimal_equal(parsed_offset, offset):
            return True
    return False


def interval_matches(
    pattern: re.Pattern[str], text: str, expected: tuple[Decimal, Decimal]
) -> bool:
    match = pattern.search(text)
    if match is None:
        return False
    low, high = parse_decimal(match.group(1)), parse_decimal(match.group(2))
    return (
        low is not None
        and high is not None
        and decimal_equal(low, expected[0])
        and decimal_equal(high, expected[1])
    )


def physical_result_matches(text: str, expected: Decimal) -> bool:
    return any(
        parsed is not None and decimal_equal(parsed, expected)
        for parsed in (parse_decimal(match.group(1)) for match in PHYSICAL_RESULT_PATTERN.finditer(text))
    )


def unit_matches(text: str, unit: str) -> bool:
    normalized_unit = normalize_text(unit or "unitless")
    if normalized_unit == "unitless":
        return "unitless" in normalize_text(text)
    if normalized_unit == "%":
        return "%" in text or "percent" in normalize_text(text)
    return re.search(
        rf"(?<!\w){re.escape(normalized_unit)}(?!\w)", normalize_text(text)
    ) is not None


def declared_bounds(definition: dict[str, Any]) -> tuple[Decimal, Decimal] | None:
    signal = definition["signal"]
    low = signal.get("declared_physical_min")
    high = signal.get("declared_physical_max")
    if low is None and high is None:
        return None
    if low is None or high is None:
        return None
    return decimal(low), decimal(high)


def normalized_phrase_matches(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text).replace("_", " ").replace("-", " ")
    normalized_phrase = normalize_text(phrase).replace("_", " ").replace("-", " ")
    return normalized_phrase in normalized_text


def location_fields(definition: dict[str, Any], output: str) -> dict[str, bool]:
    frame, signal = definition["frame"], definition["signal"]
    scale, offset = decimal(signal["scale"]), decimal(signal["offset"])
    normalized = normalize_text(output)
    can_id_decimal = int(frame["can_id_decimal"])
    payload_bytes = int(frame["payload_length_bytes"])
    start_bit = int(signal["dbc_start_bit"])
    width = int(signal["bit_length"])
    raw_domain = tuple(decimal(value) for value in raw_limits(definition))
    signedness = normalize_text(str(signal["signedness"]))
    vehicle = definition.get("vehicle", {})
    vehicle_parts = [
        normalize_text(str(vehicle.get(key)))
        for key in ("year", "make", "model")
        if vehicle.get(key)
    ]
    equation_match = equation_matches(output, scale, offset)
    parsed_scale = contextual_decimal(SCALE_PATTERN, output)
    parsed_offset = contextual_decimal(OFFSET_PATTERN, output)
    return {
        "vehicle": all(part in normalized for part in vehicle_parts),
        "message_name": normalize_text(str(frame["message_name"])) in normalized,
        "signal_name": normalize_text(str(signal["name"])) in normalized,
        "can_id_hex": normalize_text(str(frame["can_id_hex"])) in normalized,
        "can_id_decimal": re.search(
            rf"(?i)(?:\bdecimal\s+|\(\s*){can_id_decimal}\b", output
        )
        is not None,
        "payload_bytes": re.search(
            rf"(?i)\b(?:DLC\s*(?:is|=|:)?\s*{payload_bytes}|{payload_bytes}\s+bytes?)\b",
            output,
        )
        is not None,
        "start_bit": re.search(
            rf"(?i)\b(?:DBC\s+bit|start(?:s|ing)?(?:\s+at)?(?:\s+DBC)?(?:\s+bit)?)"
            rf"\s*(?:is|=|:)?\s*{start_bit}\b",
            output,
        )
        is not None,
        "bit_width": re.search(
            rf"(?i)\b(?:spans?|width)\s*(?:is|=|:)?\s*{width}(?:\s+bits?)?\b",
            output,
        )
        is not None,
        "byte_order": normalized_phrase_matches(output, str(signal["byte_order"])),
        "signedness": re.search(
            rf"(?<!\w){re.escape(signedness)}(?!\w)", normalized
        )
        is not None,
        "raw_domain": interval_matches(RAW_DOMAIN_PATTERN, output, raw_domain),
        "scale": equation_match
        or (parsed_scale is not None and decimal_equal(parsed_scale, scale)),
        "offset": equation_match
        or (parsed_offset is not None and decimal_equal(parsed_offset, offset)),
        "physical_unit": unit_matches(output, str(signal.get("unit") or "unitless")),
    }


def score_fields(
    prompt: str,
    output: str,
    provenance: dict[str, Any] | None = None,
) -> tuple[str, dict[str, bool]]:
    kind = task_type(prompt, provenance)
    if kind == "location":
        if provenance is None:
            raise ValueError("location scoring requires aligned provenance")
        return kind, location_fields(definition_from_provenance(provenance), output)

    definition = parse_definition(prompt)
    frame, signal = definition["frame"], definition["signal"]
    positions = dbc_positions(definition)
    scale, offset = decimal(signal["scale"]), decimal(signal["offset"])
    bounds = declared_bounds(definition)
    has_declared_bounds = any(
        key in signal for key in ("declared_physical_min", "declared_physical_max")
    )
    normalized = normalize_text(output)

    common = {
        "dbc_traversal": parsed_positions(output) == positions,
        "decoder_equation": equation_matches(output, scale, offset),
    }
    if kind == "decode":
        encoded, raw, physical = decode_truth(prompt, definition)
        parsed_raw = contextual_decimal(RAW_PATTERN, output)
        fields = {
            **common,
            "raw_integer": parsed_raw is not None and decimal_equal(parsed_raw, decimal(raw)),
            "physical_value": physical_result_matches(output, physical),
            "physical_unit": unit_matches(output, str(signal["unit"])),
        }
        if signal["signedness"] == "signed":
            parsed_encoded = contextual_decimal(UNSIGNED_FIELD_PATTERN, output)
            fields["encoded_unsigned_integer"] = (
                parsed_encoded is not None and decimal_equal(parsed_encoded, decimal(encoded))
            )
            fields["sign_handling"] = (
                "two's-complement" in normalized or "twos-complement" in normalized
            ) and str(signal["bit_length"]) in normalized
        else:
            fields["unsigned_handling"] = "unsigned" in normalized
        if has_declared_bounds and bounds is None:
            fields["declared_range_status"] = (
                "no physical bounds" in normalized or "no declared" in normalized
            )
        elif bounds is not None:
            inside = bounds[0] <= physical <= bounds[1]
            inside_claim = "inside" in normalized or "within" in normalized
            outside_claim = "outside" in normalized or "out of" in normalized
            fields["declared_range_status"] = inside_claim if inside else outside_claim
            fields["declared_range_bounds"] = interval_matches(
                DECLARED_RANGE_PATTERN, output, bounds
            )
        return kind, fields

    raw_domain = tuple(decimal(value) for value in raw_limits(definition))
    physical_interval = physical_limits(definition)
    payload_bits = int(frame["payload_length_bytes"]) * 8
    fields = {
        **common,
        "mechanically_valid": (
            "mechanically valid" in normalized or "definition is valid" in normalized
        ) and "invalid" not in normalized,
        "can_identifier": normalize_text(str(frame["can_id_hex"])) in normalized,
        "identifier_format": normalize_text(str(frame["identifier_format"])) in normalized,
        "payload_bytes": re.search(
            rf"(?i)\b{int(frame['payload_length_bytes'])}\s+bytes?\b", output
        ) is not None,
        "payload_bits": re.search(rf"(?i)\b{payload_bits}\s+bits?\b", output) is not None,
        "traversal_inside_payload": (
            "every position is inside" in normalized
            or "all positions are inside" in normalized
            or "stays inside" in normalized
        ),
        "raw_domain": interval_matches(RAW_DOMAIN_PATTERN, output, raw_domain),
        "signedness": re.search(
            rf"(?<!\w){re.escape(normalize_text(str(signal['signedness'])))}(?!\w)",
            normalized,
        ) is not None,
        "physical_interval": interval_matches(
            PHYSICAL_INTERVAL_PATTERN, output, physical_interval
        ),
        "physical_unit": unit_matches(output, str(signal["unit"])),
    }
    if has_declared_bounds and bounds is None:
        fields["declared_bounds"] = "no physical bounds" in normalized or "no declared" in normalized
    elif bounds is not None:
        fields["declared_bounds"] = interval_matches(DECLARED_RANGE_PATTERN, output, bounds)
        fields["declared_bounds_compatible"] = (
            "bounds are compatible" in normalized or "range is compatible" in normalized
        )
    return kind, fields


def inferred_provenance_path(validation: Path) -> Path:
    return validation.with_name(f"{validation.stem}.provenance.jsonl")


def load_provenance(
    path: Path,
    examples: list[ChatExample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: provenance row must be an object")
            rows.append(row)
    if len(rows) != len(examples):
        raise ValueError(
            f"provenance/data row mismatch: {len(rows)} provenance rows for "
            f"{len(examples)} examples"
        )
    for index, (example, row) in enumerate(zip(examples, rows, strict=True), start=1):
        if row.get("line") not in {None, index}:
            raise ValueError(f"{path}:{index}: provenance line number is not aligned")
        expected_hash = row.get("training_messages_sha256")
        messages = [message.to_dict() for message in example.messages]
        if expected_hash is not None and canonical_sha256(messages) != expected_hash:
            raise ValueError(f"{path}:{index}: messages/provenance hash mismatch")
    return rows


def selection_rank(seed: int, index: int, provenance: dict[str, Any]) -> str:
    example_id = str(provenance.get("example_id", index))
    return hashlib.sha256(f"{seed}|{example_id}|{index}".encode()).hexdigest()


def select_examples(
    examples: list[ChatExample],
    provenance: list[dict[str, Any]] | None,
    *,
    limit: int,
    seed: int,
) -> tuple[list[ChatExample], list[dict[str, Any]] | None, list[int], dict[str, Any]]:
    selected_count = min(limit, len(examples))
    method = "leading_rows_legacy"
    indices = list(range(selected_count))

    if provenance is not None:
        cells: dict[tuple[str, str], list[int]] = defaultdict(list)
        complete_grouping = True
        for index, row in enumerate(provenance):
            signal_group = row.get("signal_group")
            kind = row.get("example_type")
            if not isinstance(signal_group, str) or kind not in {
                "decode",
                "definition_validation",
                "location",
            }:
                complete_grouping = False
                break
            cells[(signal_group, str(kind))].append(index)

        if complete_grouping and cells:
            ordered_cells = sorted(cells)
            quota, remainder = divmod(selected_count, len(ordered_cells))
            if quota <= min(len(cell) for cell in cells.values()):
                chosen: list[int] = []
                next_candidates: list[int] = []
                for cell in ordered_cells:
                    ranked = sorted(
                        cells[cell],
                        key=lambda index: selection_rank(seed, index, provenance[index]),
                    )
                    chosen.extend(ranked[:quota])
                    if quota < len(ranked):
                        next_candidates.append(ranked[quota])
                next_candidates.sort(
                    key=lambda index: selection_rank(seed, index, provenance[index])
                )
                chosen.extend(next_candidates[:remainder])
                indices = sorted(chosen)
                method = "provenance_signal_task_stratified_sha256_v1"
            else:
                indices = sorted(
                    range(len(examples)),
                    key=lambda index: selection_rank(seed, index, provenance[index]),
                )[:selected_count]
                indices.sort()
                method = "provenance_global_sha256_v1"
        else:
            indices = sorted(
                range(len(examples)),
                key=lambda index: selection_rank(seed, index, provenance[index]),
            )[:selected_count]
            indices.sort()
            method = "provenance_global_sha256_v1"

    selected_examples = [examples[index] for index in indices]
    selected_provenance = (
        [provenance[index] for index in indices] if provenance is not None else None
    )
    task_counts: Counter[str] = Counter()
    signal_groups: set[str] = set()
    message_groups: set[str] = set()
    example_ids: list[str] = []
    if selected_provenance is not None:
        for row in selected_provenance:
            task_counts[str(row.get("example_type", "unknown"))] += 1
            if isinstance(row.get("signal_group"), str):
                signal_groups.add(str(row["signal_group"]))
            if isinstance(row.get("message_group"), str):
                message_groups.add(str(row["message_group"]))
            example_ids.append(str(row.get("example_id", "")))
    selection = {
        "method": method,
        "seed": seed,
        "requested_limit": limit,
        "selected_examples": len(indices),
        "source_indices_zero_based": indices,
        "example_ids": example_ids,
        "task_counts": dict(sorted(task_counts.items())),
        "signal_groups": len(signal_groups),
        "message_groups": len(message_groups),
    }
    return selected_examples, selected_provenance, indices, selection


def adapter_targets(directory: Path) -> tuple[str, ...]:
    payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    return tuple(payload["target_modules"])


def generate(
    engine: MixtureEngine,
    examples: list[ChatExample],
    *,
    route: dict[str, float],
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer = engine.tokenizer
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = engine.model.get_input_embeddings().weight.device
    outputs: list[str] = []
    engine.model.eval()
    try:
        with engine.use_route(route, normalize=False), torch.inference_mode():
            for offset in range(0, len(examples), batch_size):
                batch = examples[offset : offset + batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        [message.to_dict() for message in example.messages[:-1]],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for example in batch
                ]
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_tokens,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                prompt_width = int(inputs["input_ids"].shape[1])
                generated = engine.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                outputs.extend(
                    tokenizer.batch_decode(
                        generated[:, prompt_width:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
    finally:
        tokenizer.padding_side = previous_padding_side
    return outputs


def score_route(
    examples: list[ChatExample],
    outputs: list[str],
    *,
    provenance: list[dict[str, Any]] | None = None,
    source_indices: list[int] | None = None,
) -> dict[str, Any]:
    if provenance is not None and len(provenance) != len(examples):
        raise ValueError("selected provenance is not aligned with examples")
    if source_indices is not None and len(source_indices) != len(examples):
        raise ValueError("source indices are not aligned with examples")
    details: list[dict[str, Any]] = []
    for index, (example, output) in enumerate(zip(examples, outputs, strict=True)):
        prompt = example.messages[-2].content
        expected = example.messages[-1].content
        row_provenance = provenance[index] if provenance is not None else None
        kind, fields = score_fields(prompt, output, row_provenance)
        details.append(
            {
                "index": index,
                "source_index_zero_based": (
                    source_indices[index] if source_indices is not None else index
                ),
                "example_id": (
                    row_provenance.get("example_id")
                    if row_provenance is not None
                    else None
                ),
                "signal_group": (
                    row_provenance.get("signal_group")
                    if row_provenance is not None
                    else None
                ),
                "message_group": (
                    row_provenance.get("message_group")
                    if row_provenance is not None
                    else None
                ),
                "task_type": kind,
                "normalized_exact_match": normalize_text(output) == normalize_text(expected),
                "all_structured_fields_match": all(fields.values()),
                "structured_fields": fields,
                "expected": expected,
                "generated": output.strip(),
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        field_totals: dict[str, list[int]] = {}
        for row in rows:
            for name, matched in row["structured_fields"].items():
                totals = field_totals.setdefault(name, [0, 0])
                totals[0] += int(matched)
                totals[1] += 1
        field_counts = {
            name: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]}
            for name, values in sorted(field_totals.items())
        }
        total_fields = sum(values[1] for values in field_totals.values())
        correct_fields = sum(values[0] for values in field_totals.values())
        count = len(rows)
        return {
            "examples": count,
            "normalized_exact_matches": sum(row["normalized_exact_match"] for row in rows),
            "normalized_exact_match_rate": (
                sum(row["normalized_exact_match"] for row in rows) / count if count else 0.0
            ),
            "all_structured_fields_matches": sum(
                row["all_structured_fields_match"] for row in rows
            ),
            "all_structured_fields_match_rate": (
                sum(row["all_structured_fields_match"] for row in rows) / count if count else 0.0
            ),
            "structured_field_accuracy": correct_fields / total_fields if total_fields else 0.0,
            "fields": field_counts,
        }

    kinds = sorted({str(row["task_type"]) for row in details})
    return {
        "overall": aggregate(details),
        "by_task_type": {
            kind: aggregate([row for row in details if row["task_type"] == kind])
            for kind in kinds
        },
        "examples": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        help=(
            "aligned validation provenance JSONL; inferred beside validation when present"
        ),
    )
    parser.add_argument("--limit", type=int, default=52)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/evaluation/grounded-can-generation.json"),
    )
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1:
        parser.error("limit and batch-size must be positive")
    if args.max_prompt_tokens < 64 or args.max_new_tokens < 1:
        parser.error("max-prompt-tokens must be at least 64 and max-new-tokens must be positive")

    model = args.model.resolve()
    adapter = args.adapter.resolve()
    validation = args.validation.resolve()
    output_path = args.output.resolve()
    all_examples = load_jsonl(validation)
    if args.provenance is not None:
        provenance_path: Path | None = args.provenance.resolve()
        if not provenance_path.is_file():
            parser.error(f"provenance file does not exist: {provenance_path}")
    else:
        candidate = inferred_provenance_path(validation)
        provenance_path = candidate if candidate.is_file() else None
    all_provenance = (
        load_provenance(provenance_path, all_examples)
        if provenance_path is not None
        else None
    )
    examples, provenance, source_indices, selection = select_examples(
        all_examples,
        all_provenance,
        limit=args.limit,
        seed=args.seed,
    )
    for index, example in enumerate(examples):
        prompt = example.messages[-2].content
        expected = example.messages[-1].content
        row_provenance = provenance[index] if provenance is not None else None
        _kind, expected_fields = score_fields(prompt, expected, row_provenance)
        if not all(expected_fields.values()):
            failed = sorted(name for name, matched in expected_fields.items() if not matched)
            source_index = source_indices[index]
            raise ValueError(
                f"reference answer at validation row {source_index + 1} does not satisfy "
                f"the evaluator fields: {', '.join(failed)}"
            )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    engine = MixtureEngine.from_pretrained(
        str(model),
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        target_modules=adapter_targets(adapter),
        local_files_only=True,
    )
    expert_name = engine.load_expert(adapter)
    routes = {"expert": {expert_name: 1.0}}
    if not args.skip_base:
        routes = {"base": {BASE_ROUTE: 1.0}, **routes}

    report: dict[str, Any] = {
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "model": str(model),
        "adapter": str(adapter),
        "adapter_artifacts": {
            "config_sha256": file_sha256(adapter / "adapter_config.json"),
            "model_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        },
        "validation": str(validation),
        "validation_sha256": file_sha256(validation),
        "provenance": str(provenance_path) if provenance_path is not None else None,
        "provenance_sha256": (
            file_sha256(provenance_path) if provenance_path is not None else None
        ),
        "selection": selection,
        "generation": {
            "greedy": True,
            "thinking": False,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
        },
        "routes": {},
    }
    for label, route in routes.items():
        generated = generate(
            engine,
            examples,
            route=route,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        report["routes"][label] = score_route(
            examples,
            generated,
            provenance=provenance,
            source_indices=source_indices,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    console_summary = {
        "report": str(output_path),
        "validation_sha256": report["validation_sha256"],
        "selection": {
            key: value
            for key, value in report["selection"].items()
            if key not in {"source_indices_zero_based", "example_ids"}
        },
        "routes": {
            label: route_report["overall"] for label, route_report in report["routes"].items()
        },
    }
    # Persist the complete UTF-8 report first.  Escape console-only output so a
    # Windows cp1252 terminal cannot fail on a unit or multiplication glyph.
    print(json.dumps(console_summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
