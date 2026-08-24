"""Build the reasoning-guided v3 embedded-code curriculum and fresh-base blend.

V3 keeps the independently parameterized v2 contracts and executable harnesses but
changes the teaching signal: masks and bit constants are formula-derived in the C,
all macros are standard and included, and one of three answer variants contains a
short explicit Qwen ``<think>...</think>`` calculation plan.  The other two variants
remain fenced-code-only and raw-code-only targets.

The builder also creates a deterministic v1-train + v3-train blend.  This is meant
for a new adapter on the frozen base; it deliberately excludes the v2 warm-start and
all validation rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import build_verified_embedded_code_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "data"
V1_TRAIN = DEFAULT_OUTPUT / "verified-embedded-code-v1.train.jsonl"
SEALED_SUITES = ROOT / "benchmarks" / "suites"
EVALUATION_DIRECTORY = ROOT / "runs" / "evaluation"
SEED = 240831
FENCE = "`" * 3


@dataclass(frozen=True, slots=True)
class Example:
    group: str
    family: str
    request: str
    code: str
    harness: str
    planning: str
    mode: str

    @property
    def prompt(self) -> str:
        if self.mode == "think_fenced":
            instruction = (
                "First give a concise calculation plan inside literal <think>...</think> tags. "
                "Then return exactly one c fenced source file containing the strict C11 "
                "implementation and required standard includes. Do not define main."
            )
        elif self.mode == "fenced":
            instruction = (
                "Return exactly one c fenced source file containing the strict C11 "
                "implementation and required standard includes. Do not include reasoning or "
                "define main."
            )
        elif self.mode == "raw":
            instruction = (
                "Output only the raw strict C11 translation unit with no Markdown, reasoning, "
                "or explanation. Do not define main."
            )
        else:
            raise ValueError(f"unknown output mode: {self.mode}")
        return f"{self.request}\n\n{instruction}"

    def row(self) -> dict[str, object]:
        source = self.code.rstrip()
        if self.mode == "think_fenced":
            answer = f"<think>\n{self.planning.rstrip()}\n</think>\n{FENCE}c\n{source}\n{FENCE}"
        elif self.mode == "fenced":
            answer = f"{FENCE}c\n{source}\n{FENCE}"
        else:
            answer = source
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": answer},
            ]
        }


def _signed_constant(value: int) -> str:
    return f"INT32_C({value})" if value >= 0 else f"(-INT32_C({-value}))"


def _three_modes(
    *, group: str, family: str, request: str, code: str, harness: str, planning: str
) -> list[Example]:
    return [
        Example(
            group=f"v3:{group}",
            family=family,
            request=request,
            code=code,
            harness=harness,
            planning=planning,
            mode=mode,
        )
        for mode in ("think_fenced", "fenced", "raw")
    ]


def _function_name(code: str, return_type: str) -> str:
    match = re.search(rf"\b{return_type}\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    if match is None:
        raise RuntimeError("could not recover generated function name")
    return match.group(1).replace("_v2_", "_v3_")


def _signed_example(base: v2.Example) -> list[Example]:
    _kind, start_text, width_text, scale_text, offset_text = base.group.split(":")
    start, width, scale, offset = map(
        int, (start_text, width_text, scale_text, offset_text)
    )
    name = _function_name(base.code, "bool")
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    modulus = 1 << width
    shifted_mask = mask << start
    code = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool {name}(const uint8_t frame[3], int16_t *result)
{{
    enum {{ FIELD_START = {start}, FIELD_WIDTH = {width} }};
    if ((frame == NULL) || (result == NULL)) {{
        return false;
    }}
    const uint32_t value_mask =
        (UINT32_C(1) << FIELD_WIDTH) - UINT32_C(1);
    const uint32_t sign_mask = UINT32_C(1) << (FIELD_WIDTH - 1);
    const uint32_t modulus = UINT32_C(1) << FIELD_WIDTH;
    const uint32_t packed = (uint32_t)frame[0] |
        ((uint32_t)frame[1] << 8U) |
        ((uint32_t)frame[2] << 16U);
    const uint32_t unsigned_raw = (packed >> FIELD_START) & value_mask;
    const int32_t signed_raw = ((unsigned_raw & sign_mask) != 0U)
        ? ((int32_t)unsigned_raw - (int32_t)modulus)
        : (int32_t)unsigned_raw;
    const int32_t scaled =
        (signed_raw * INT32_C({scale})) + {_signed_constant(offset)};
    *result = (int16_t)scaled;
    return true;
}}"""
    harness = base.harness.replace("_v2_", "_v3_")
    request = (
        f"Implement `bool {name}(const uint8_t frame[3], int16_t *result)`. Using LSB-zero "
        f"numbering across the three-byte little-endian container, extract the signed {width}-bit "
        f"two's-complement field beginning at bit {start}. Ignore all unrelated bits, sign-extend "
        f"without implementation-defined signed shifts, then store `signed_raw * {scale} + "
        f"({offset})`. The result range fits int16_t. Return false for either null pointer and "
        "leave `*result` unchanged on failure. Derive masks from the width in the implementation."
    )
    planning = (
        f"The field occupies container bits {start} through {start + width - 1}. Width {width} "
        f"gives value_mask=(1U<<{width})-1={mask} (0x{mask:X}), "
        f"sign_mask=1U<<{width - 1}={sign} (0x{sign:X}), and modulus=1U<<{width}={modulus}.\n"
        f"Pack the bytes at shifts 0, 8, and 16; shift right {start} and mask. If the sign bit is "
        f"set, subtract {modulus}, then calculate signed_raw*{scale}+({offset}) in int32_t.\n"
        "Check both pointers before any output write and assign the int16_t result only after all "
        f"calculations. The shifted field mask would be 0x{shifted_mask:X}."
    )
    return _three_modes(
        group=base.group,
        family="signed_cross_byte_formula_reasoning",
        request=request,
        code=code,
        harness=harness,
        planning=planning,
    )


def _register_example(base: v2.Example) -> list[Example]:
    values = tuple(int(value) for value in base.group.split(":")[1:])
    output_pin, input_pin, shift, width, route_value, set_bit, clear_bit = values
    function = _function_name(base.code, "void")
    suffix = function.rsplit("_", 1)[-1]
    names = {
        key: f"V3_{key.upper()}_{suffix}"
        for key in ("analog", "direction", "route", "control")
    }
    output_mask = 1 << output_pin
    input_mask = 1 << input_pin
    analog_mask = output_mask | input_mask
    route_value_mask = (1 << width) - 1
    route_mask = route_value_mask << shift
    encoded_route = route_value << shift
    set_mask = 1 << set_bit
    clear_mask = 1 << clear_bit
    code = f"""#include <stdint.h>

extern volatile uint32_t {names['analog']};
extern volatile uint32_t {names['direction']};
extern volatile uint32_t {names['route']};
extern volatile uint32_t {names['control']};

void {function}(void)
{{
    enum {{
        OUTPUT_PIN = {output_pin}, INPUT_PIN = {input_pin},
        ROUTE_SHIFT = {shift}, ROUTE_WIDTH = {width}, ROUTE_VALUE = {route_value},
        CONTROL_SET_BIT = {set_bit}, CONTROL_CLEAR_BIT = {clear_bit}
    }};
    const uint32_t output_mask = UINT32_C(1) << OUTPUT_PIN;
    const uint32_t input_mask = UINT32_C(1) << INPUT_PIN;
    const uint32_t analog_mask = output_mask | input_mask;
    const uint32_t route_mask =
        ((UINT32_C(1) << ROUTE_WIDTH) - UINT32_C(1)) << ROUTE_SHIFT;
    const uint32_t encoded_route =
        ((uint32_t)ROUTE_VALUE << ROUTE_SHIFT) & route_mask;
    const uint32_t control_set_mask = UINT32_C(1) << CONTROL_SET_BIT;
    const uint32_t control_clear_mask = UINT32_C(1) << CONTROL_CLEAR_BIT;

    {names['analog']} = {names['analog']} & ~analog_mask;
    {names['direction']} =
        ({names['direction']} | input_mask) & ~output_mask;
    {names['route']} =
        ({names['route']} & ~route_mask) | encoded_route;
    {names['control']} =
        ({names['control']} | control_set_mask) & ~control_clear_mask;
}}"""
    harness = (
        base.harness.replace("_v2_", "_v3_")
        .replace("V2_", "V3_")
    )
    request = (
        f"Implement `void {function}(void)` for four harness-provided volatile uint32_t objects "
        f"named `{names['analog']}`, `{names['direction']}`, `{names['route']}`, and "
        f"`{names['control']}`. Declare every object `extern`. Clear analog bits for pins "
        f"{output_pin} and {input_pin}; make pin {output_pin} an output by clearing its direction "
        f"bit and pin {input_pin} an input by setting its direction bit. Replace only route bits "
        f"{shift}..{shift + width - 1} with {route_value}. Set control bit {set_bit} and clear "
        f"control bit {clear_bit}. Preserve every unrelated bit. Derive all masks from bit "
        "positions and widths rather than embedding precomputed register constants."
    )
    planning = (
        f"Use standard UINT32_C from <stdint.h>; no custom bit macro is needed. output_mask="
        f"1U<<{output_pin}=0x{output_mask:08X}, input_mask=1U<<{input_pin}=0x{input_mask:08X}, "
        f"so analog_mask=0x{analog_mask:08X}.\n"
        f"route value mask=(1U<<{width})-1=0x{route_value_mask:X}; shifting by {shift} gives "
        f"route_mask=0x{route_mask:08X} and encoded value {route_value}<<{shift}="
        f"0x{encoded_route:08X}. Clear the field, then OR the encoded value.\n"
        f"control set mask is 1U<<{set_bit}=0x{set_mask:08X}; clear mask is "
        f"1U<<{clear_bit}=0x{clear_mask:08X}. Declare all four volatile registers extern and "
        "use masked read-modify-write expressions so unrelated bits survive."
    )
    return _three_modes(
        group=base.group,
        family="volatile_register_formula_reasoning",
        request=request,
        code=code,
        harness=harness,
        planning=planning,
    )


def _escape_example(base: v2.Example) -> list[Example]:
    prefix, reserved_a, reserved_b, xor_mask = (
        int(value) for value in base.group.split(":")[1:]
    )
    function = _function_name(base.code, "bool")
    code = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool {function}(const uint8_t *source, size_t length,
    uint8_t *destination, size_t capacity, size_t *written)
{{
    const uint8_t prefix = UINT8_C(0x{prefix:02x});
    const uint8_t reserved_a = UINT8_C(0x{reserved_a:02x});
    const uint8_t reserved_b = UINT8_C(0x{reserved_b:02x});
    const uint8_t transform = UINT8_C(0x{xor_mask:02x});
    if (written == NULL) {{
        return false;
    }}
    if (length == 0U) {{
        *written = 0U;
        return true;
    }}
    if ((source == NULL) || (destination == NULL)) {{
        return false;
    }}

    size_t required = 0U;
    for (size_t input_index = 0U; input_index < length; ++input_index) {{
        const bool escaped = (source[input_index] == reserved_a) ||
            (source[input_index] == reserved_b);
        const size_t increment = escaped ? 2U : 1U;
        if (required > (SIZE_MAX - increment)) {{
            return false;
        }}
        required += increment;
    }}
    if (required > capacity) {{
        return false;
    }}

    size_t output_index = 0U;
    for (size_t input_index = 0U; input_index < length; ++input_index) {{
        const uint8_t value = source[input_index];
        if ((value == reserved_a) || (value == reserved_b)) {{
            destination[output_index] = prefix;
            ++output_index;
            destination[output_index] = (uint8_t)(value ^ transform);
            ++output_index;
        }} else {{
            destination[output_index] = value;
            ++output_index;
        }}
    }}
    *written = required;
    return true;
}}"""
    harness = base.harness.replace("_v2_", "_v3_")
    request = (
        f"Implement `bool {function}(const uint8_t *source, size_t length, uint8_t "
        "*destination, size_t capacity, size_t *written)`. Copy ordinary bytes unchanged. "
        f"Escape byte 0x{reserved_a:02X} or 0x{reserved_b:02X} as prefix 0x{prefix:02X} "
        f"followed by `byte ^ 0x{xor_mask:02X}`. `written` is always required. For zero length, "
        "source and destination may be null; set `*written` to zero and succeed. For nonzero "
        "length both buffers are required and do not overlap. Preflight the complete encoded "
        "size without overflow. Invalid arguments or insufficient capacity must return false "
        "without changing destination or `*written`; encode only after successful preflight. "
        "Use only standard C11 operators and macros declared by the included standard headers."
    )
    planning = (
        f"An input byte needs two output bytes exactly when it equals 0x{reserved_a:02X} or "
        f"0x{reserved_b:02X}; otherwise it needs one. For each byte set increment to 2 or 1, "
        "check required > SIZE_MAX-increment, then add it. SIZE_MAX comes from <stdint.h>.\n"
        "Validate the written pointer first. Handle zero length next. For nonzero input validate "
        "both buffers. The entire sizing loop must be read-only; compare required with capacity "
        "before the encoding loop.\n"
        f"Only after preflight, emit 0x{prefix:02X} and value^0x{xor_mask:02X} for reserved "
        "bytes. Commit *written last, so every failure leaves destination and *written unchanged."
    )
    return _three_modes(
        group=base.group,
        family="transactional_escape_preflight_reasoning",
        request=request,
        code=code,
        harness=harness,
        planning=planning,
    )


def _build(seed: int) -> list[Example]:
    base_rows = v2._build(v2.SEED)
    unique: dict[str, v2.Example] = {}
    for row in base_rows:
        unique.setdefault(row.group, row)
    rows: list[Example] = []
    for base in sorted(unique.values(), key=lambda row: row.group):
        if base.group.startswith("signed-field:"):
            rows.extend(_signed_example(base))
        elif base.group.startswith("register:"):
            rows.extend(_register_example(base))
        elif base.group.startswith("escape:"):
            rows.extend(_escape_example(base))
        else:
            raise RuntimeError(f"unexpected v2 semantic family: {base.group}")
    random.Random(seed).shuffle(rows)
    if len(rows) != 720 or len({row.group for row in rows}) != 240:
        raise RuntimeError("unexpected v3 curriculum size")
    prompts = [row.prompt for row in rows]
    if len(prompts) != len(set(prompts)):
        raise RuntimeError("duplicate v3 prompt")
    return rows


def _split(rows: list[Example]) -> tuple[list[Example], list[Example]]:
    family_groups: dict[str, set[str]] = {}
    for row in rows:
        family_groups.setdefault(row.family, set()).add(row.group)
    validation_group_ids: set[str] = set()
    for family, groups in sorted(family_groups.items()):
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"{family}:{group}".encode("utf-8")
            ).digest(),
        )
        validation_group_ids.update(ordered[: max(1, len(ordered) // 8)])
    train = [row for row in rows if row.group not in validation_group_ids]
    validation = [row for row in rows if row.group in validation_group_ids]
    train_groups = {row.group for row in train}
    validation_groups = {row.group for row in validation}
    if train_groups & validation_groups:
        raise RuntimeError("v3 semantic groups overlap across splits")
    return train, validation


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _walk_strings(value: Any, key: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key and isinstance(child, str):
                found.append(child)
            found.extend(_walk_strings(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_strings(child, key))
    return found


def _denylist() -> tuple[set[str], set[str], list[dict[str, object]]]:
    prompt_hashes: set[str] = set()
    completion_hashes: set[str] = set()
    sources: list[dict[str, object]] = []
    paths = sorted(SEALED_SUITES.glob("*.json")) + sorted(
        EVALUATION_DIRECTORY.glob("verified-embedded-code-*-siblings.json")
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        prompts = _walk_strings(payload, "prompt")
        completions = _walk_strings(payload, "completion")
        prompt_hashes.update(
            hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()
            for text in prompts
        )
        completion_hashes.update(
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in completions
        )
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prompts": len(prompts),
                "completions": len(completions),
            }
        )
    return prompt_hashes, completion_hashes, sources


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"messages"} or not isinstance(row["messages"], list):
                raise ValueError(f"{path}:{line_number}: not messages-only JSONL")
            rows.append(row)
    return rows


def _audit_rows(
    rows: list[dict[str, object]], prompt_denylist: set[str], completion_denylist: set[str]
) -> None:
    for row in rows:
        messages = row["messages"]
        assert isinstance(messages, list)
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("message is not an object")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("message content is not text")
            if role == "user":
                digest = hashlib.sha256(_normalize(content).encode("utf-8")).hexdigest()
                if digest in prompt_denylist:
                    raise RuntimeError("training prompt exactly matches sealed/evaluator prompt")
            elif role == "assistant":
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest in completion_denylist:
                    raise RuntimeError("training answer exactly matches evaluator completion")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> tuple[int, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(temporary, path)
    return len(rows), digest.hexdigest()


def _stable_blend(
    v1_rows: list[dict[str, object]], v3_rows: list[dict[str, object]], seed: int
) -> list[dict[str, object]]:
    tagged = [("v1", index, row) for index, row in enumerate(v1_rows)] + [
        ("v3", index, row) for index, row in enumerate(v3_rows)
    ]
    tagged.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0]}:{item[1]}:".encode("utf-8")
            + json.dumps(item[2], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
    )
    return [row for _origin, _index, row in tagged]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wsl-distro", default="Ubuntu")
    args = parser.parse_args()

    rows = _build(args.seed)
    train, validation = _split(rows)
    verified_groups, compiler, flags = v2._validate(rows, args.wsl_distro)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_rows = [row.row() for row in train]
    validation_rows = [row.row() for row in validation]
    prompt_denylist, completion_denylist, denylist_sources = _denylist()
    _audit_rows(train_rows + validation_rows, prompt_denylist, completion_denylist)

    train_path = output / "verified-embedded-code-v3.train.jsonl"
    validation_path = output / "verified-embedded-code-v3.validation.jsonl"
    train_count, train_sha = _write_rows(train_path, train_rows)
    validation_count, validation_sha = _write_rows(validation_path, validation_rows)

    v1_rows = _read_jsonl(V1_TRAIN)
    combined_rows = _stable_blend(v1_rows, train_rows, args.seed)
    _audit_rows(combined_rows, prompt_denylist, completion_denylist)
    combined_path = output / "verified-embedded-code-v1-plus-v3.train.jsonl"
    combined_count, combined_sha = _write_rows(combined_path, combined_rows)

    modes = Counter(row.mode for row in rows)
    families = Counter(row.family for row in rows)
    manifest = {
        "schema_version": "molor.verified-code-curriculum.v3",
        "seed": args.seed,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "v2_generator_dependency": {
            "path": str(Path(v2.__file__).resolve()),
            "sha256": hashlib.sha256(Path(v2.__file__).read_bytes()).hexdigest(),
            "use": "parameter contracts and independent executable harnesses only",
        },
        "regression_audit": {
            "v2_score_1000": 75.063,
            "v1_score_1000": 152.191,
            "signed_decode": "generated a 9-bit value mask for a 13-bit field while using the 13-bit modulus",
            "register_rmw": "omitted extern declarations, invented UINT32_VALUE, and mixed pin/route/control masks",
            "escaping": "invented SIZEOF_UINT16, wrote during sizing, ignored capacity, and did not preserve failure atomicity",
            "v3_response": "formula-derived masks, standard-only macros, explicit calculation plans, and fresh-base breadth blend",
        },
        "decontamination": {
            "sources": denylist_sources,
            "normalized_prompt_hashes_denied": len(prompt_denylist),
            "completion_hashes_denied": len(completion_denylist),
            "exact_collisions": 0,
        },
        "verification": {
            "wsl_distro": args.wsl_distro,
            "compiler": compiler,
            "flags": flags,
            "translation_units_linked_and_executed": verified_groups,
        },
        "v3": {
            "families": dict(sorted(families.items())),
            "answer_modes": dict(sorted(modes.items())),
            "semantic_groups": {
                "total": len({row.group for row in rows}),
                "train": len({row.group for row in train}),
                "validation": len({row.group for row in validation}),
                "overlap": 0,
            },
            "splits": {
                "train": {"path": str(train_path), "rows": train_count, "sha256": train_sha},
                "validation": {
                    "path": str(validation_path),
                    "rows": validation_count,
                    "sha256": validation_sha,
                },
            },
        },
        "fresh_base_combined_train": {
            "path": str(combined_path),
            "rows": combined_count,
            "sha256": combined_sha,
            "composition": {
                "verified_embedded_code_v1_train": len(v1_rows),
                "verified_embedded_code_v3_train": len(train_rows),
            },
            "validation_rows_included": 0,
            "v2_rows_included": 0,
        },
    }
    manifest_path = output / "verified-embedded-code-v3.manifest.json"
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
