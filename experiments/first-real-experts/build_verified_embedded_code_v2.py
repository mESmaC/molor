"""Build a failure-driven, executable v2 curriculum for the embedded-code expert.

V1 taught isolated unsigned loads, masks, and fixed CRC loops.  This shard instead
teaches the three compositions missed by the independent sibling evaluation:

* signed cross-byte extraction followed by sign extension and affine scaling;
* complete volatile-register updates, including the required extern declarations;
* transactional variable-length escaping with an overflow-safe capacity preflight.

Every semantic group has two output-format variants, stays wholly in one split, and
is compiled, warning-checked, linked to an independently generated harness, and run
under WSL Ubuntu before either JSONL file is published.
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
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"
EVALUATION = ROOT / "runs" / "evaluation" / "verified-embedded-code-v1-siblings.json"
SEED = 240817
FENCE = "`" * 3


@dataclass(frozen=True, slots=True)
class Example:
    group: str
    family: str
    prompt: str
    code: str
    harness: str
    fenced: bool

    def row(self) -> dict[str, object]:
        answer = (
            f"{FENCE}c\n{self.code.rstrip()}\n{FENCE}"
            if self.fenced
            else self.code.rstrip()
        )
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": answer},
            ]
        }


def _instruction(fenced: bool) -> str:
    if fenced:
        return (
            "Return exactly one c fenced source file containing the strict C11 "
            "implementation and required standard includes. Do not define main."
        )
    return "Output only the raw strict C11 translation unit with no Markdown or explanation."


def _variants(
    *, group: str, family: str, request: str, code: str, harness: str
) -> list[Example]:
    return [
        Example(
            group=group,
            family=family,
            prompt=f"{request}\n\n{_instruction(fenced)}",
            code=code,
            harness=harness,
            fenced=fenced,
        )
        for fenced in (True, False)
    ]


def _signed_constant(value: int) -> str:
    return f"INT32_C({value})" if value >= 0 else f"(-INT32_C({-value}))"


def _signed_field_examples(rng: random.Random) -> list[Example]:
    candidates = [
        (start, width, scale, offset)
        for start in range(1, 8)
        for width in range(9, 14)
        for scale in (1, 2, 3, 4)
        for offset in (-251, -113, -29, 73, 191)
        if start + width <= 23
        and (start, width, scale, offset) != (2, 13, 3, 17)
    ]
    rng.shuffle(candidates)
    rows: list[Example] = []
    for index, (start, width, scale, offset) in enumerate(candidates[:80]):
        name = f"decode_sensor_field_v2_{index:02d}"
        value_mask = (1 << width) - 1
        sign_mask = 1 << (width - 1)
        modulus = 1 << width
        field_mask = value_mask << start
        scaled_low = (-(1 << (width - 1)) * scale) + offset
        scaled_high = (((1 << (width - 1)) - 1) * scale) + offset
        if not (-32768 <= scaled_low <= scaled_high <= 32767):
            raise AssertionError("generated affine result does not fit int16_t")
        code = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool {name}(const uint8_t frame[3], int16_t *result)
{{
    if ((frame == NULL) || (result == NULL)) {{
        return false;
    }}
    const uint32_t packed = (uint32_t)frame[0] |
        ((uint32_t)frame[1] << 8U) |
        ((uint32_t)frame[2] << 16U);
    const uint32_t unsigned_raw =
        (packed >> {start}U) & UINT32_C(0x{value_mask:08x});
    const int32_t signed_raw =
        ((unsigned_raw & UINT32_C(0x{sign_mask:08x})) != 0U)
        ? ((int32_t)unsigned_raw - INT32_C({modulus}))
        : (int32_t)unsigned_raw;
    const int32_t scaled =
        (signed_raw * INT32_C({scale})) + {_signed_constant(offset)};
    *result = (int16_t)scaled;
    return true;
}}"""
        harness = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

extern bool {name}(const uint8_t frame[3], int16_t *result);

int main(void)
{{
    uint8_t frame[3] = {{0U, 0U, 0U}};
    int16_t output = INT16_C(1234);
    if ({name}(NULL, &output) || (output != INT16_C(1234))) {{ return 1; }}
    if ({name}(frame, NULL)) {{ return 2; }}
    for (uint32_t raw = 0U; raw <= UINT32_C(0x{value_mask:08x}); ++raw) {{
        uint32_t packed = UINT32_C(0x00a5c35a);
        packed = (packed & ~UINT32_C(0x{field_mask:08x})) | (raw << {start}U);
        frame[0] = (uint8_t)(packed & UINT32_C(0xff));
        frame[1] = (uint8_t)((packed >> 8U) & UINT32_C(0xff));
        frame[2] = (uint8_t)((packed >> 16U) & UINT32_C(0xff));
        const int32_t signed_raw =
            ((raw & UINT32_C(0x{sign_mask:08x})) != 0U)
            ? ((int32_t)raw - INT32_C({modulus})) : (int32_t)raw;
        const int16_t expected = (int16_t)(
            (signed_raw * INT32_C({scale})) + {_signed_constant(offset)});
        output = INT16_C(-30000);
        if ((!{name}(frame, &output)) || (output != expected)) {{ return 3; }}
    }}
    return 0;
}}"""
        request = (
            f"Implement `bool {name}(const uint8_t frame[3], int16_t *result)`. "
            f"Using LSB-zero numbering across the three-byte little-endian container, extract "
            f"the signed {width}-bit two's-complement field beginning at bit {start}. Ignore "
            f"all unrelated bits, sign-extend it without implementation-defined shifts, then "
            f"store `signed_raw * {scale} + ({offset})`. The stated range always fits int16_t. "
            "Return false for either null pointer and leave `*result` unchanged on failure."
        )
        rows.extend(
            _variants(
                group=f"signed-field:{start}:{width}:{scale}:{offset}",
                family="signed_cross_byte_affine",
                request=request,
                code=code,
                harness=harness,
            )
        )
    return rows


def _register_examples(rng: random.Random) -> list[Example]:
    rows: list[Example] = []
    specs: set[tuple[int, int, int, int, int, int, int]] = set()
    while len(specs) < 80:
        output_pin, input_pin = rng.sample(range(0, 16), 2)
        shift = rng.choice((0, 4, 8, 12, 16, 20, 24, 27))
        width = rng.choice((2, 3, 4))
        if shift + width > 32:
            continue
        route_value = rng.randrange(1 << width)
        set_bit, clear_bit = rng.sample(range(0, 16), 2)
        spec = (output_pin, input_pin, shift, width, route_value, set_bit, clear_bit)
        if spec != (12, 3, 4, 3, 5, 2, 7):
            specs.add(spec)
    for index, spec in enumerate(sorted(specs)):
        output_pin, input_pin, shift, width, route_value, set_bit, clear_bit = spec
        analog_mask = (1 << output_pin) | (1 << input_pin)
        output_mask = 1 << output_pin
        input_mask = 1 << input_pin
        route_mask = ((1 << width) - 1) << shift
        encoded_route = route_value << shift
        set_mask = 1 << set_bit
        clear_mask = 1 << clear_bit
        names = {
            key: f"V2_{key.upper()}_{index:02d}"
            for key in ("analog", "direction", "route", "control")
        }
        function = f"configure_event_block_v2_{index:02d}"
        code = f"""#include <stdint.h>

extern volatile uint32_t {names['analog']};
extern volatile uint32_t {names['direction']};
extern volatile uint32_t {names['route']};
extern volatile uint32_t {names['control']};

void {function}(void)
{{
    {names['analog']} = {names['analog']} & ~UINT32_C(0x{analog_mask:08x});
    {names['direction']} =
        ({names['direction']} | UINT32_C(0x{input_mask:08x})) &
        ~UINT32_C(0x{output_mask:08x});
    {names['route']} =
        ({names['route']} & ~UINT32_C(0x{route_mask:08x})) |
        UINT32_C(0x{encoded_route:08x});
    {names['control']} =
        ({names['control']} | UINT32_C(0x{set_mask:08x})) &
        ~UINT32_C(0x{clear_mask:08x});
}}"""
        harness = f"""#include <stdint.h>

volatile uint32_t {names['analog']};
volatile uint32_t {names['direction']};
volatile uint32_t {names['route']};
volatile uint32_t {names['control']};
extern void {function}(void);

static uint32_t mix(uint32_t value)
{{
    value ^= value << 13U;
    value ^= value >> 17U;
    return value ^ (value << 5U);
}}

int main(void)
{{
    uint32_t state = UINT32_C(0x71d24a39);
    for (uint32_t trial = 0U; trial < 64U; ++trial) {{
        state = mix(state + trial + 1U);
        const uint32_t analog_before = state;
        state = mix(state);
        const uint32_t direction_before = state;
        state = mix(state);
        const uint32_t route_before = state;
        state = mix(state);
        const uint32_t control_before = state;
        {names['analog']} = analog_before;
        {names['direction']} = direction_before;
        {names['route']} = route_before;
        {names['control']} = control_before;
        {function}();
        if ({names['analog']} !=
            (analog_before & ~UINT32_C(0x{analog_mask:08x}))) {{ return 1; }}
        if ({names['direction']} !=
            ((direction_before | UINT32_C(0x{input_mask:08x})) &
             ~UINT32_C(0x{output_mask:08x}))) {{ return 2; }}
        if ({names['route']} !=
            ((route_before & ~UINT32_C(0x{route_mask:08x})) |
             UINT32_C(0x{encoded_route:08x}))) {{ return 3; }}
        if ({names['control']} !=
            ((control_before | UINT32_C(0x{set_mask:08x})) &
             ~UINT32_C(0x{clear_mask:08x}))) {{ return 4; }}
    }}
    return 0;
}}"""
        request = (
            f"Implement `void {function}(void)` for four harness-provided volatile uint32_t "
            f"objects named `{names['analog']}`, `{names['direction']}`, `{names['route']}`, "
            f"and `{names['control']}`. Declare all four objects `extern`. Clear analog bits "
            f"for pins {output_pin} and {input_pin}; make pin {output_pin} an output by clearing "
            f"its direction bit and pin {input_pin} an input by setting its direction bit. "
            f"Replace only route bits {shift}..{shift + width - 1} with {route_value}. Set "
            f"control bit {set_bit} and clear control bit {clear_bit}. Preserve every unrelated bit."
        )
        rows.extend(
            _variants(
                group="register:" + ":".join(str(value) for value in spec),
                family="volatile_masked_register_composition",
                request=request,
                code=code,
                harness=harness,
            )
        )
    return rows


def _escape_examples(rng: random.Random) -> list[Example]:
    rows: list[Example] = []
    byte_pool = tuple(range(0x40, 0x80))
    specs: set[tuple[int, int, int, int]] = set()
    while len(specs) < 80:
        prefix, reserved_a, reserved_b = rng.sample(byte_pool, 3)
        xor_mask = rng.choice((0x10, 0x20, 0x40, 0x55))
        spec = (prefix, min(reserved_a, reserved_b), max(reserved_a, reserved_b), xor_mask)
        if spec != (0x7D, 0x7D, 0x7E, 0x20):
            specs.add(spec)
    for index, (prefix, reserved_a, reserved_b, xor_mask) in enumerate(sorted(specs)):
        function = f"escape_packet_bytes_v2_{index:02d}"
        code = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool {function}(const uint8_t *source, size_t length,
    uint8_t *destination, size_t capacity, size_t *written)
{{
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
        const size_t increment =
            ((source[input_index] == UINT8_C(0x{reserved_a:02x})) ||
             (source[input_index] == UINT8_C(0x{reserved_b:02x}))) ? 2U : 1U;
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
        if ((value == UINT8_C(0x{reserved_a:02x})) ||
            (value == UINT8_C(0x{reserved_b:02x}))) {{
            destination[output_index] = UINT8_C(0x{prefix:02x});
            ++output_index;
            destination[output_index] = (uint8_t)(value ^ UINT8_C(0x{xor_mask:02x}));
            ++output_index;
        }} else {{
            destination[output_index] = value;
            ++output_index;
        }}
    }}
    *written = required;
    return true;
}}"""
        ordinary = next(value for value in (0x11, 0x22, 0x33) if value not in spec)
        expected = (
            prefix,
            reserved_a ^ xor_mask,
            ordinary,
            prefix,
            reserved_b ^ xor_mask,
        )
        harness = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

extern bool {function}(const uint8_t *source, size_t length,
    uint8_t *destination, size_t capacity, size_t *written);

int main(void)
{{
    const uint8_t source[3] = {{UINT8_C(0x{reserved_a:02x}),
        UINT8_C(0x{ordinary:02x}), UINT8_C(0x{reserved_b:02x})}};
    const uint8_t expected[5] = {{{', '.join(f'UINT8_C(0x{value:02x})' for value in expected)}}};
    uint8_t destination[8];
    size_t written = 99U;
    for (size_t index = 0U; index < 8U; ++index) {{ destination[index] = UINT8_C(0xa5); }}
    if ({function}(NULL, 1U, destination, 8U, &written) || (written != 99U)) {{ return 1; }}
    if ({function}(source, 3U, NULL, 8U, &written) || (written != 99U)) {{ return 2; }}
    if ({function}(source, 3U, destination, 4U, &written) || (written != 99U)) {{ return 3; }}
    for (size_t index = 0U; index < 8U; ++index) {{
        if (destination[index] != UINT8_C(0xa5)) {{ return 4; }}
    }}
    if ({function}(source, 3U, destination, 8U, NULL)) {{ return 5; }}
    if ((!{function}(NULL, 0U, NULL, 0U, &written)) || (written != 0U)) {{ return 6; }}
    written = 99U;
    if ((!{function}(source, 3U, destination, 5U, &written)) || (written != 5U)) {{ return 7; }}
    for (size_t index = 0U; index < 5U; ++index) {{
        if (destination[index] != expected[index]) {{ return 8; }}
    }}
    return 0;
}}"""
        request = (
            f"Implement `bool {function}(const uint8_t *source, size_t length, uint8_t "
            "*destination, size_t capacity, size_t *written)`. Copy ordinary bytes unchanged. "
            f"Escape byte 0x{reserved_a:02X} or 0x{reserved_b:02X} as prefix 0x{prefix:02X} "
            f"followed by `byte ^ 0x{xor_mask:02X}`. `written` is always required. For zero "
            "length, source and destination may be null; set `*written` to zero and succeed. "
            "For nonzero length both buffers are required and do not overlap. Preflight the "
            "entire encoded size without overflow. Invalid arguments or insufficient capacity "
            "must return false without changing destination or `*written`; write only after a "
            "successful preflight."
        )
        rows.extend(
            _variants(
                group=f"escape:{prefix}:{reserved_a}:{reserved_b}:{xor_mask}",
                family="transactional_variable_escape",
                request=request,
                code=code,
                harness=harness,
            )
        )
    return rows


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _heldout_hashes() -> tuple[set[str], set[str], str]:
    payload = json.loads(EVALUATION.read_text(encoding="utf-8"))
    identities = payload["disjointness_audit"]["task_identity"].values()
    prompt_hashes = {identity["normalized_prompt_sha256"] for identity in identities}
    completion_hashes = {
        case["generation"]["completion_sha256"]
        for route in ("base", "forced_adapter")
        for case in payload["routes"][route]["cases"]
    }
    return prompt_hashes, completion_hashes, hashlib.sha256(EVALUATION.read_bytes()).hexdigest()


def _build(seed: int) -> list[Example]:
    rng = random.Random(seed)
    rows = [
        *_signed_field_examples(rng),
        *_register_examples(rng),
        *_escape_examples(rng),
    ]
    prompts = [row.prompt for row in rows]
    if len(prompts) != len(set(prompts)):
        raise RuntimeError("duplicate generated prompt")
    heldout_prompts, heldout_completions, _ = _heldout_hashes()
    for row in rows:
        prompt_hash = hashlib.sha256(_normalize(row.prompt).encode("utf-8")).hexdigest()
        completion_hash = hashlib.sha256(row.code.encode("utf-8")).hexdigest()
        if prompt_hash in heldout_prompts or completion_hash in heldout_completions:
            raise RuntimeError("generated content collided with the heldout evaluation")
    return rows


def _split(rows: list[Example]) -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    validation: list[Example] = []
    for row in rows:
        bucket = int.from_bytes(hashlib.sha256(row.group.encode("utf-8")).digest()[:4], "big") % 8
        (validation if bucket == 0 else train).append(row)
    train_groups = {row.group for row in train}
    validation_groups = {row.group for row in validation}
    if train_groups & validation_groups:
        raise RuntimeError("semantic group overlap between train and validation")
    return train, validation


def _validate(rows: list[Example], distro: str) -> tuple[int, str, str]:
    groups: dict[str, Example] = {}
    for row in rows:
        groups.setdefault(row.group, row)
    with tempfile.TemporaryDirectory(prefix="molor-verified-v2-") as temporary:
        directory = Path(temporary)
        for index, row in enumerate(sorted(groups.values(), key=lambda item: item.group)):
            (directory / f"target-{index:04d}.c").write_text(
                row.code.rstrip() + "\n", encoding="utf-8", newline="\n"
            )
            (directory / f"harness-{index:04d}.c").write_text(
                row.harness.rstrip() + "\n", encoding="utf-8", newline="\n"
            )
        linux_directory = subprocess.run(
            ["wsl.exe", "-d", distro, "--exec", "/usr/bin/wslpath", "-a", "-u", str(directory)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        quoted = shlex.quote(linux_directory)
        command = f"""set -euo pipefail
root={quoted}
for target in "$root"/target-*.c; do
    stem="${{target##*/target-}}"
    stem="${{stem%.c}}"
    harness="$root/harness-$stem.c"
    executable="$root/program-$stem"
    gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -Wconversion -Wsign-conversion -Wshadow \
        "$target" "$harness" -o "$executable"
    "$executable"
done
gcc --version | head -n 1
"""
        result = subprocess.run(
            ["wsl.exe", "-d", distro, "--exec", "/bin/bash", "-lc", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    compiler = result.stdout.strip().splitlines()[-1]
    flags = "-std=c11 -O2 -Wall -Wextra -Werror -pedantic -Wconversion -Wsign-conversion -Wshadow"
    return len(groups), compiler, flags


def _write_jsonl(path: Path, rows: list[Example]) -> tuple[int, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row.row(), ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    os.replace(temporary, path)
    return len(rows), digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wsl-distro", default="Ubuntu")
    args = parser.parse_args()

    rows = _build(args.seed)
    train, validation = _split(rows)
    verified_groups, compiler, flags = _validate(rows, args.wsl_distro)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "verified-embedded-code-v2.train.jsonl"
    validation_path = output / "verified-embedded-code-v2.validation.jsonl"
    train_count, train_sha = _write_jsonl(train_path, train)
    validation_count, validation_sha = _write_jsonl(validation_path, validation)
    heldout_prompts, heldout_completions, evaluation_sha = _heldout_hashes()
    manifest = {
        "schema_version": "molor.verified-code-curriculum.v2",
        "seed": args.seed,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "failure_drivers": {
            "signed_decode": "v1 separated unsigned extraction from saturation and did not teach cross-byte sign extension plus affine scaling as one transaction",
            "register_rmw": "v1 register examples were simpler three-register compositions and did not reinforce complete extern declarations plus independent route/control clear-and-set masks",
            "escaping": "v1 had no variable-output encoder requiring a full capacity preflight and failure atomicity",
        },
        "decontamination": {
            "heldout_evaluation": str(EVALUATION),
            "heldout_evaluation_sha256": evaluation_sha,
            "heldout_normalized_prompt_hashes_denied": len(heldout_prompts),
            "heldout_completion_hashes_denied": len(heldout_completions),
            "exact_collisions": 0,
            "semantic_group_overlap": 0,
        },
        "verification": {
            "wsl_distro": args.wsl_distro,
            "compiler": compiler,
            "flags": flags,
            "translation_units_linked_and_executed": verified_groups,
        },
        "families": dict(sorted(Counter(row.family for row in rows).items())),
        "semantic_groups": {
            "total": len({row.group for row in rows}),
            "train": len({row.group for row in train}),
            "validation": len({row.group for row in validation}),
        },
        "splits": {
            "train": {"path": str(train_path), "rows": train_count, "sha256": train_sha},
            "validation": {
                "path": str(validation_path),
                "rows": validation_count,
                "sha256": validation_sha,
            },
        },
    }
    manifest_path = output / "verified-embedded-code-v2.manifest.json"
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
