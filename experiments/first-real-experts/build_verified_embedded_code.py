"""Build a compact, compiler-verified instruction corpus for an embedded-code expert.

The examples are deterministic siblings of real embedded programming work, not copies of the
sealed benchmark prompts or fixtures.  Every assistant target is syntax-checked with the WSL
toolchain before the messages-only JSONL files are published.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import tempfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"
FENCE = "`" * 3


@dataclass(frozen=True, slots=True)
class Example:
    group: str
    family: str
    language: str
    prompt: str
    code: str
    fenced: bool

    def row(self) -> dict[str, object]:
        answer = f"{FENCE}{self.language}\n{self.code.rstrip()}\n{FENCE}" if self.fenced else self.code.rstrip()
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": answer},
            ]
        }


def _format_instruction(language: str, *, fenced: bool) -> str:
    standard = "strict C11" if language == "c" else "C++20"
    if fenced:
        return (
            f"Return exactly one {language} fenced source file containing the {standard} "
            "implementation and required standard includes. Do not define main."
        )
    return f"Output only the raw {standard} translation unit with no Markdown or explanation."


def _add_variants(
    output: list[Example],
    *,
    group: str,
    family: str,
    language: str,
    request: str,
    code: str,
) -> None:
    output.append(
        Example(
            group=group,
            family=family,
            language=language,
            prompt=f"{request}\n\n{_format_instruction(language, fenced=True)}",
            code=code,
            fenced=True,
        )
    )
    output.append(
        Example(
            group=group,
            family=family,
            language=language,
            prompt=f"{request}\n\n{_format_instruction(language, fenced=False)}",
            code=code,
            fenced=False,
        )
    )


def _load_store_examples() -> list[Example]:
    rows: list[Example] = []
    stems = (
        "packet", "frame", "sample", "field", "payload", "register", "header", "record",
        "telemetry", "counter", "timestamp", "argument", "descriptor", "channel", "event", "word",
    )
    for width in (2, 3, 4):
        for endian in ("little", "big"):
            for index, stem in enumerate(stems):
                load_name = f"load_{stem}_{endian[:2]}{width}_{index}"
                if endian == "little":
                    terms = " |\n        ".join(
                        f"((uint32_t)source[{byte}] << {8 * byte}U)" for byte in range(width)
                    )
                else:
                    terms = " |\n        ".join(
                        f"((uint32_t)source[{byte}] << {8 * (width - byte - 1)}U)"
                        for byte in range(width)
                    )
                load_code = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool {load_name}(const uint8_t *source, size_t available, uint32_t *result)
{{
    if ((source == NULL) || (result == NULL) || (available < {width}U)) {{
        return false;
    }}
    const uint32_t decoded = {terms};
    *result = decoded;
    return true;
}}"""
                _add_variants(
                    rows,
                    group=f"load:{width}:{endian}:{index}",
                    family="bounded_endian_load",
                    language="c",
                    request=(
                        f"Implement `bool {load_name}(const uint8_t *source, size_t available, "
                        f"uint32_t *result)`. Decode the first {width} bytes as an unsigned "
                        f"{endian}-endian integer. Reject null pointers or short input without "
                        "modifying `*result`."
                    ),
                    code=load_code,
                )

                store_name = f"store_{stem}_{endian[:2]}{width}_{index}"
                assignments: list[str] = []
                for byte in range(width):
                    shift = 8 * byte if endian == "little" else 8 * (width - byte - 1)
                    assignments.append(
                        f"    destination[{byte}] = (uint8_t)((value >> {shift}U) & UINT32_C(0xff));"
                    )
                store_code = f"""#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool {store_name}(uint8_t *destination, size_t capacity, uint32_t value)
{{
    if ((destination == NULL) || (capacity < {width}U)) {{
        return false;
    }}
{chr(10).join(assignments)}
    return true;
}}"""
                _add_variants(
                    rows,
                    group=f"store:{width}:{endian}:{index}",
                    family="bounded_endian_store",
                    language="c",
                    request=(
                        f"Implement `bool {store_name}(uint8_t *destination, size_t capacity, "
                        f"uint32_t value)`. Store the low {width * 8} bits in {endian}-endian "
                        "byte order. Reject a null destination or short capacity without writing."
                    ),
                    code=store_code,
                )
    return rows


def _bitfield_examples() -> list[Example]:
    rows: list[Example] = []
    configurations = [
        (shift, width)
        for shift in (0, 1, 3, 5, 7, 9, 12, 16, 20, 24)
        for width in (1, 2, 3, 4, 5, 6, 8)
        if shift + width <= 32
    ]
    for index, (shift, width) in enumerate(configurations):
        low_mask = (1 << width) - 1
        shifted_mask = low_mask << shift
        extract_name = f"extract_status_field_{index}"
        extract_code = f"""#include <stdint.h>

uint32_t {extract_name}(uint32_t word)
{{
    return (word >> {shift}U) & UINT32_C(0x{low_mask:08x});
}}"""
        _add_variants(
            rows,
            group=f"extract:{shift}:{width}",
            family="bitfield_extract",
            language="c",
            request=(
                f"Implement `uint32_t {extract_name}(uint32_t word)` to return the unsigned "
                f"{width}-bit field beginning at bit {shift}. Avoid implementation-defined behavior."
            ),
            code=extract_code,
        )

        insert_name = f"replace_control_field_{index}"
        insert_code = f"""#include <stdint.h>

uint32_t {insert_name}(uint32_t original, uint32_t field_value)
{{
    const uint32_t mask = UINT32_C(0x{shifted_mask:08x});
    const uint32_t encoded = (field_value << {shift}U) & mask;
    return (original & ~mask) | encoded;
}}"""
        _add_variants(
            rows,
            group=f"insert:{shift}:{width}",
            family="bitfield_rmw",
            language="c",
            request=(
                f"Implement `uint32_t {insert_name}(uint32_t original, uint32_t field_value)`. "
                f"Replace only the {width}-bit field at bit {shift}, truncate excess field bits, "
                "and preserve every unrelated bit."
            ),
            code=insert_code,
        )
    return rows


def _register_examples(seed: int) -> list[Example]:
    rows: list[Example] = []
    rng = random.Random(seed)
    for index in range(64):
        pin_a, pin_b = rng.sample(range(0, 16), 2)
        field_shift = rng.choice((0, 4, 8, 12, 16, 20, 24))
        field_width = rng.choice((2, 3, 4))
        field_value = rng.randrange(1 << field_width)
        pin_mask = (1 << pin_a) | (1 << pin_b)
        field_mask = ((1 << field_width) - 1) << field_shift
        name = f"configure_capture_path_{index}"
        code = f"""#include <stdint.h>

extern volatile uint32_t IO_ANALOG_{index};
extern volatile uint32_t IO_DIRECTION_{index};
extern volatile uint32_t ROUTE_CONFIG_{index};

void {name}(void)
{{
    const uint32_t pins = UINT32_C(0x{pin_mask:08x});
    const uint32_t route_mask = UINT32_C(0x{field_mask:08x});
    IO_ANALOG_{index} &= ~pins;
    IO_DIRECTION_{index} =
        (IO_DIRECTION_{index} | UINT32_C(0x{1 << pin_b:08x})) &
        ~UINT32_C(0x{1 << pin_a:08x});
    ROUTE_CONFIG_{index} =
        (ROUTE_CONFIG_{index} & ~route_mask) |
        UINT32_C(0x{field_value << field_shift:08x});
}}"""
        _add_variants(
            rows,
            group=f"register:{index}",
            family="volatile_register_rmw",
            language="c",
            request=(
                f"Implement `void {name}(void)` for the supplied volatile uint32_t SFR stubs "
                f"`IO_ANALOG_{index}`, `IO_DIRECTION_{index}`, and `ROUTE_CONFIG_{index}`. Make "
                f"pins {pin_a} and {pin_b} digital, configure pin {pin_a} as output and pin {pin_b} "
                f"as input, then write value {field_value} into the {field_width}-bit route field "
                f"at bit {field_shift}. Preserve every unrelated register bit. Declare the SFRs extern."
            ),
            code=code,
        )
    return rows


def _saturation_examples() -> list[Example]:
    rows: list[Example] = []
    for bits, narrow, wide in ((8, "int8_t", "int32_t"), (16, "int16_t", "int32_t"), (32, "int32_t", "int64_t")):
        for operation, operator in (("add", "+"), ("sub", "-")):
            for index in range(8):
                name = f"saturating_{operation}_{bits}_{index}"
                code = f"""#include <stdint.h>

{narrow} {name}({narrow} left, {narrow} right)
{{
    const {wide} value = ({wide})left {operator} ({wide})right;
    if (value > ({wide})INT{bits}_MAX) {{
        return INT{bits}_MAX;
    }}
    if (value < ({wide})INT{bits}_MIN) {{
        return INT{bits}_MIN;
    }}
    return ({narrow})value;
}}"""
                _add_variants(
                    rows,
                    group=f"saturation:{bits}:{operation}:{index}",
                    family="defined_saturating_arithmetic",
                    language="c",
                    request=(
                        f"Implement `{narrow} {name}({narrow} left, {narrow} right)` with "
                        f"saturating {operation}. Clamp at the exact-width limits without signed "
                        "overflow or implementation-defined narrowing."
                    ),
                    code=code,
                )
    return rows


def _crc_examples() -> list[Example]:
    rows: list[Example] = []
    variants = (
        ("crc8_atm", 8, 0x07, 0x00, 0x00, False),
        ("crc8_sae_j1850", 8, 0x1D, 0xFF, 0xFF, False),
        ("crc16_xmodem", 16, 0x1021, 0x0000, 0x0000, False),
        ("crc16_modbus", 16, 0xA001, 0xFFFF, 0x0000, True),
        ("crc32_ieee_bitwise", 32, 0xEDB88320, 0xFFFFFFFF, 0xFFFFFFFF, True),
    )
    for base_name, bits, polynomial, initial, xorout, reflected in variants:
        ctype = f"uint{bits}_t"
        macro = f"UINT{bits}_C"
        hex_width = bits // 4
        top = 1 << (bits - 1)
        for index in range(10):
            name = f"{base_name}_{index}"
            if reflected:
                step = (
                    f"            crc = ((crc & {macro}(0x1)) != 0U)\n"
                    f"                ? ({ctype})((crc >> 1U) ^ {macro}(0x{polynomial:0{hex_width}x}))\n"
                    f"                : ({ctype})(crc >> 1U);"
                )
                mix = f"        crc ^= ({ctype})data[byte_index];"
                order = "least-significant bit first"
            else:
                shift = bits - 8
                step = (
                    f"            crc = ((crc & {macro}(0x{top:0{hex_width}x})) != 0U)\n"
                    f"                ? ({ctype})((crc << 1U) ^ {macro}(0x{polynomial:0{hex_width}x}))\n"
                    f"                : ({ctype})(crc << 1U);"
                )
                mix = f"        crc ^= ({ctype})(({ctype})data[byte_index] << {shift}U);"
                order = "most-significant bit first"
            code = f"""#include <stddef.h>
#include <stdint.h>

{ctype} {name}(const uint8_t *data, size_t length)
{{
    {ctype} crc = {macro}(0x{initial:0{hex_width}x});
    if ((data == NULL) && (length != 0U)) {{
        return ({ctype})(crc ^ {macro}(0x{xorout:0{hex_width}x}));
    }}
    for (size_t byte_index = 0U; byte_index < length; ++byte_index) {{
{mix}
        for (uint8_t bit = 0U; bit < 8U; ++bit) {{
{step}
        }}
    }}
    return ({ctype})(crc ^ {macro}(0x{xorout:0{hex_width}x}));
}}"""
            _add_variants(
                rows,
                group=f"crc:{base_name}:{index}",
                family="crc_parameter_fidelity",
                language="c",
                request=(
                    f"Implement `{ctype} {name}(const uint8_t *data, size_t length)` using width "
                    f"{bits}, polynomial 0x{polynomial:0{hex_width}X}, init "
                    f"0x{initial:0{hex_width}X}, xorout 0x{xorout:0{hex_width}X}, and process bytes "
                    f"{order}. No lookup table. A null pointer is valid only for zero length."
                ),
                code=code,
            )
    return rows


def _cpp_examples() -> list[Example]:
    rows: list[Example] = []
    for index in range(16):
        name = f"longest_nondecreasing_run_{index}"
        code = f"""#include <cstddef>
#include <cstdint>
#include <span>

std::size_t {name}(std::span<const std::int32_t> values)
{{
    if (values.empty()) {{
        return 0U;
    }}
    std::size_t best = 1U;
    std::size_t current = 1U;
    for (std::size_t index = 1U; index < values.size(); ++index) {{
        current = (values[index] >= values[index - 1U]) ? (current + 1U) : 1U;
        if (current > best) {{
            best = current;
        }}
    }}
    return best;
}}"""
        _add_variants(
            rows,
            group=f"cpp:run:{index}",
            family="cpp_linear_algorithms",
            language="cpp",
            request=(
                f"Implement `std::size_t {name}(std::span<const std::int32_t> values)` to return "
                "the longest contiguous nondecreasing run, or zero for empty input, in O(n) time."
            ),
            code=code,
        )

        search_name = f"first_not_less_index_{index}"
        search_code = f"""#include <cstddef>
#include <cstdint>
#include <span>

std::size_t {search_name}(std::span<const std::int32_t> sorted, std::int32_t target)
{{
    std::size_t low = 0U;
    std::size_t high = sorted.size();
    while (low < high) {{
        const std::size_t middle = low + ((high - low) / 2U);
        if (sorted[middle] < target) {{
            low = middle + 1U;
        }} else {{
            high = middle;
        }}
    }}
    return low;
}}"""
        _add_variants(
            rows,
            group=f"cpp:search:{index}",
            family="cpp_linear_algorithms",
            language="cpp",
            request=(
                f"Implement `std::size_t {search_name}(std::span<const std::int32_t> sorted, "
                "std::int32_t target)` as lower_bound semantics in O(log n), returning size when "
                "no element is at least target."
            ),
            code=search_code,
        )
    return rows


def build_examples(seed: int) -> list[Example]:
    rows = [
        *_load_store_examples(),
        *_bitfield_examples(),
        *_register_examples(seed),
        *_saturation_examples(),
        *_crc_examples(),
        *_cpp_examples(),
    ]
    prompts = [row.prompt for row in rows]
    if len(prompts) != len(set(prompts)):
        raise RuntimeError("generator produced duplicate prompts")
    return rows


def _split(examples: Iterable[Example]) -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    validation: list[Example] = []
    for example in examples:
        bucket = int.from_bytes(hashlib.sha256(example.group.encode("utf-8")).digest()[:4], "big") % 8
        (validation if bucket == 0 else train).append(example)
    if not train or not validation:
        raise RuntimeError("deterministic split produced an empty partition")
    train_groups = {row.group for row in train}
    validation_groups = {row.group for row in validation}
    if train_groups.intersection(validation_groups):
        raise RuntimeError("semantic groups leaked across train and validation")
    return train, validation


def _extract_code(example: Example) -> str:
    return example.code.rstrip() + "\n"


def _compile_all(examples: Iterable[Example]) -> dict[str, int]:
    unique: dict[tuple[str, str], str] = {}
    for example in examples:
        unique[(example.language, hashlib.sha256(example.code.encode("utf-8")).hexdigest())] = _extract_code(example)
    with tempfile.TemporaryDirectory(prefix="molor-verified-code-") as temporary:
        root = Path(temporary)
        counts: Counter[str] = Counter()
        for index, ((language, _digest), source) in enumerate(sorted(unique.items())):
            suffix = ".c" if language == "c" else ".cpp"
            (root / f"target-{index:04d}{suffix}").write_text(source, encoding="utf-8", newline="\n")
            counts[language] += 1
        converted = subprocess.run(
            [
                "wsl.exe",
                "-d",
                "Ubuntu",
                "--exec",
                "/usr/bin/wslpath",
                "-a",
                "-u",
                str(root.resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        quoted = shlex.quote(converted)
        command = (
            "set -euo pipefail; "
            f"find {quoted} -maxdepth 1 -type f -name '*.c' -print0 | "
            "xargs -0 -r -n1 gcc -std=c11 -Wall -Wextra -Werror -pedantic -fsyntax-only; "
            f"find {quoted} -maxdepth 1 -type f -name '*.cpp' -print0 | "
            "xargs -0 -r -n1 g++ -std=c++20 -Wall -Wextra -Werror -pedantic -fsyntax-only"
        )
        subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "--exec", "/bin/bash", "-lc", command],
            check=True,
        )
    return dict(sorted(counts.items()))


def _write_jsonl(path: Path, examples: Iterable[Example]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            line = json.dumps(example.row(), ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=9071)
    args = parser.parse_args()

    examples = build_examples(args.seed)
    train, validation = _split(examples)
    compile_counts = _compile_all(examples)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "verified-embedded-code-v1.train.jsonl"
    validation_path = output / "verified-embedded-code-v1.validation.jsonl"
    train_count, train_sha = _write_jsonl(train_path, train)
    validation_count, validation_sha = _write_jsonl(validation_path, validation)
    manifest = {
        "schema_version": "molor.verified-code-curriculum.v1",
        "seed": args.seed,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decontamination": {
            "policy": "new parameterized sibling tasks; no sealed benchmark prompts or fixtures consumed",
            "semantic_group_overlap": 0,
        },
        "compile_validation": {
            "toolchain": "WSL gcc -std=c11 and g++ -std=c++20, -Wall -Wextra -Werror -pedantic",
            "unique_translation_units": compile_counts,
        },
        "families": dict(sorted(Counter(row.family for row in examples).items())),
        "splits": {
            "train": {"path": str(train_path), "rows": train_count, "sha256": train_sha},
            "validation": {
                "path": str(validation_path),
                "rows": validation_count,
                "sha256": validation_sha,
            },
        },
    }
    manifest_path = output / "verified-embedded-code-v1.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
