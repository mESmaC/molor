"""Compare a frozen base and one forced LoRA adapter on fresh executable C tasks.

The cases in this file are manually authored siblings, not copies or parameter changes of the
sealed benchmark cases. They are never consumed by a training builder. Before generation the
tool checks their IDs, normalized prompts, and harness hashes against the sealed executable suite
and its registered fixtures. It also scans experiment JSONL files for exact user-prompt overlap.

Candidate code is compiled and executed by the repository's WSL2/Bubblewrap verifier, so model
output is treated as untrusted file content rather than interpolated into a shell command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch

from molor.benchmark import wsl_verifier
from molor.experimental.mixture import BASE_ROUTE, MixtureEngine


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEALED_SUITE = ROOT / "benchmarks" / "suites" / "molor-executable-v1.json"
DEFAULT_TRAINING_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = ROOT / "runs" / "evaluation" / "embedded-code-siblings-v1.json"
FENCE = chr(96) * 3
SYSTEM_PROMPT = (
    "You are completing a held-out executable embedded-code evaluation. Return exactly one "
    "fenced source file in the requested language, with no prose outside the fence. Do not "
    "define main. The implementation will be compiled separately and exercised by hidden tests."
)


@dataclass(frozen=True, slots=True)
class Task:
    case_id: str
    fixture_id: str
    prompt: str
    language: str
    filename: str
    harness: str
    expected_tests: int
    max_new_tokens: int


DECODE_HARNESS = r"""
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern bool decode_vibration_sample(const uint8_t frame[4], int16_t *out_mg);

static int16_t reference_value(uint16_t encoded)
{
    int32_t signed_raw = (int32_t)encoded;
    if ((encoded & UINT16_C(0x1000)) != 0U) {
        signed_raw -= INT32_C(8192);
    }
    return (int16_t)((signed_raw * INT32_C(3)) + INT32_C(17));
}

int main(void)
{
    unsigned passed = 0U;
    unsigned total = 0U;
    uint8_t frame[4] = {0U, 0U, UINT8_C(0xa5), UINT8_C(0x5a)};

    for (uint16_t encoded = 0U; encoded < UINT16_C(8192); ++encoded) {
        frame[0] = (uint8_t)((uint8_t)((encoded & UINT16_C(0x003f)) << 2U) |
                             (uint8_t)(encoded & UINT16_C(0x0003)));
        frame[1] = (uint8_t)((uint8_t)((encoded >> 6U) & UINT16_C(0x007f)) |
                             (uint8_t)((encoded & UINT16_C(1)) << 7U));
        int16_t output = INT16_C(2345);
        ++total;
        if (decode_vibration_sample(frame, &output) &&
            (output == reference_value(encoded))) {
            ++passed;
        }
    }

    int16_t sentinel = INT16_C(2718);
    ++total;
    if ((!decode_vibration_sample(NULL, &sentinel)) && (sentinel == INT16_C(2718))) {
        ++passed;
    }
    ++total;
    if (!decode_vibration_sample(frame, NULL)) {
        ++passed;
    }

    (void)printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


REGISTER_HARNESS = r"""
#include <stdint.h>
#include <stdio.h>

volatile uint32_t PORT_ANALOG;
volatile uint32_t PORT_DIRECTION;
volatile uint32_t EVENT_ROUTE;
volatile uint32_t EVENT_CONTROL;

extern void event_capture_init(void);

static int run_case(uint32_t analog, uint32_t direction,
                    uint32_t route, uint32_t control)
{
    PORT_ANALOG = analog;
    PORT_DIRECTION = direction;
    EVENT_ROUTE = route;
    EVENT_CONTROL = control;
    event_capture_init();

    const uint32_t pin_mask = (UINT32_C(1) << 3U) | (UINT32_C(1) << 12U);
    const uint32_t expected_analog = analog & ~pin_mask;
    const uint32_t expected_direction =
        (direction | (UINT32_C(1) << 3U)) & ~(UINT32_C(1) << 12U);
    const uint32_t expected_route =
        (route & ~(UINT32_C(7) << 4U)) | (UINT32_C(5) << 4U);
    const uint32_t expected_control =
        (control | (UINT32_C(1) << 2U)) & ~(UINT32_C(1) << 7U);

    return (PORT_ANALOG == expected_analog) &&
           (PORT_DIRECTION == expected_direction) &&
           (EVENT_ROUTE == expected_route) &&
           (EVENT_CONTROL == expected_control);
}

int main(void)
{
    static const uint32_t patterns[] = {
        UINT32_C(0), UINT32_C(0xffffffff),
        UINT32_C(0xaaaaaaaa), UINT32_C(0x13579bdf)
    };
    unsigned passed = 0U;
    unsigned total = 0U;

    for (unsigned left = 0U; left < 4U; ++left) {
        for (unsigned right = 0U; right < 4U; ++right) {
            ++total;
            if (run_case(patterns[left], patterns[right],
                         patterns[(left + right) & 3U],
                         patterns[(left + (3U * right)) & 3U])) {
                ++passed;
            }
        }
    }

    (void)printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


ESCAPE_HARNESS = r"""
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern bool encode_wire_frame(const uint8_t *source, size_t length,
                              uint8_t *destination, size_t capacity,
                              size_t *written);

static size_t reference_encode(const uint8_t *source, size_t length, uint8_t *destination)
{
    size_t output = 0U;
    for (size_t index = 0U; index < length; ++index) {
        if (source[index] == UINT8_C(0x7e)) {
            destination[output++] = UINT8_C(0x7d);
            destination[output++] = UINT8_C(0x5e);
        } else if (source[index] == UINT8_C(0x7d)) {
            destination[output++] = UINT8_C(0x7d);
            destination[output++] = UINT8_C(0x5d);
        } else {
            destination[output++] = source[index];
        }
    }
    return output;
}

static int all_equal(const uint8_t *data, size_t length, uint8_t value)
{
    for (size_t index = 0U; index < length; ++index) {
        if (data[index] != value) {
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    unsigned passed = 0U;
    unsigned total = 0U;
    uint8_t output[256];
    size_t written = 17U;

    ++total;
    if (encode_wire_frame(NULL, 0U, NULL, 0U, &written) && (written == 0U)) {
        ++passed;
    }
    ++total;
    if (!encode_wire_frame(NULL, 0U, NULL, 0U, NULL)) {
        ++passed;
    }

    (void)memset(output, 0xa5, sizeof(output));
    written = 77U;
    ++total;
    if ((!encode_wire_frame(NULL, 1U, output, sizeof(output), &written)) &&
        (written == 77U) && all_equal(output, sizeof(output), UINT8_C(0xa5))) {
        ++passed;
    }

    static const uint8_t one_byte[] = {UINT8_C(0x22)};
    written = 77U;
    ++total;
    if ((!encode_wire_frame(one_byte, 1U, NULL, 1U, &written)) && (written == 77U)) {
        ++passed;
    }

    static const uint8_t fixed[] = {
        UINT8_C(0x7e), UINT8_C(0x01), UINT8_C(0x7d)
    };
    (void)memset(output, 0xa5, sizeof(output));
    written = 77U;
    ++total;
    if ((!encode_wire_frame(fixed, sizeof(fixed), output, 4U, &written)) &&
        (written == 77U) && all_equal(output, sizeof(output), UINT8_C(0xa5))) {
        ++passed;
    }

    static const uint8_t fixed_expected[] = {
        UINT8_C(0x7d), UINT8_C(0x5e), UINT8_C(0x01),
        UINT8_C(0x7d), UINT8_C(0x5d)
    };
    (void)memset(output, 0xa5, sizeof(output));
    written = 77U;
    ++total;
    if (encode_wire_frame(fixed, sizeof(fixed), output, sizeof(output), &written) &&
        (written == sizeof(fixed_expected)) &&
        (memcmp(output, fixed_expected, sizeof(fixed_expected)) == 0) &&
        all_equal(&output[sizeof(fixed_expected)],
                  sizeof(output) - sizeof(fixed_expected), UINT8_C(0xa5))) {
        ++passed;
    }

    uint8_t source[96];
    uint8_t expected[192];
    uint32_t state = UINT32_C(0xc17a4e29);
    for (size_t length = 0U; length <= sizeof(source); ++length) {
        if (length > 0U) {
            state = (state * UINT32_C(1664525)) + UINT32_C(1013904223);
            source[length - 1U] = (length % 17U == 0U)
                                      ? UINT8_C(0x7e)
                                      : ((length % 11U == 0U)
                                             ? UINT8_C(0x7d)
                                             : (uint8_t)(state >> 24U));
        }
        const size_t required = reference_encode(source, length, expected);
        (void)memset(output, 0xa5, sizeof(output));
        written = 999U;
        ++total;
        if (encode_wire_frame(source, length, output, required, &written) &&
            (written == required) && (memcmp(output, expected, required) == 0) &&
            all_equal(&output[required], sizeof(output) - required, UINT8_C(0xa5))) {
            ++passed;
        }

        if (required > 0U) {
            (void)memset(output, 0xa5, sizeof(output));
            written = 999U;
            ++total;
            if ((!encode_wire_frame(source, length, output, required - 1U, &written)) &&
                (written == 999U) &&
                all_equal(output, sizeof(output), UINT8_C(0xa5))) {
                ++passed;
            }
        }
    }

    (void)printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


TASKS = (
    Task(
        case_id="HDEC01",
        fixture_id="heldout-sibling/decode-vibration-13bit-v1",
        language="c",
        filename="decode_vibration.c",
        expected_tests=8194,
        max_new_tokens=768,
        prompt=(
            "Implement this strict C11 function:\n\n"
            "bool decode_vibration_sample(const uint8_t frame[4], int16_t *out_mg);\n\n"
            "A signed 13-bit two's-complement field starts at bit 2 of frame[0]. Its low six "
            "bits are frame[0] bits 2..7 and its high seven bits are frame[1] bits 0..6, "
            "least-significant part first. Ignore every unrelated bit and compute "
            "out_mg = signed_raw * 3 + 17. Return false and do not modify *out_mg when either "
            "pointer is null; otherwise store the value and return true. Use only <stdbool.h>, "
            "<stddef.h>, <stdint.h>, or <limits.h>. Return exactly one c fenced source file "
            "with the implementation and required includes; do not define main."
        ),
        harness=DECODE_HARNESS,
    ),
    Task(
        case_id="HREG01",
        fixture_id="heldout-sibling/event-capture-registers-v1",
        language="c",
        filename="event_capture.c",
        expected_tests=16,
        max_new_tokens=640,
        prompt=(
            "Implement this C11 function for a fictional MCU register block:\n\n"
            "void event_capture_init(void);\n\n"
            "The hidden harness supplies these objects:\n"
            "extern volatile uint32_t PORT_ANALOG;\n"
            "extern volatile uint32_t PORT_DIRECTION;\n"
            "extern volatile uint32_t EVENT_ROUTE;\n"
            "extern volatile uint32_t EVENT_CONTROL;\n\n"
            "Make pins 3 and 12 digital by clearing their PORT_ANALOG bits. Configure pin 3 as "
            "input by setting its PORT_DIRECTION bit and pin 12 as output by clearing its bit. "
            "Set EVENT_ROUTE field bits 4..6 to binary 101. Set EVENT_CONTROL bit 2 and clear "
            "bit 7. Preserve every unrelated bit in all four registers. Return exactly one c "
            "fenced source file with any required standard includes; do not define main."
        ),
        harness=REGISTER_HARNESS,
    ),
    Task(
        case_id="HFRM01",
        fixture_id="heldout-sibling/transactional-wire-escape-v1",
        language="c",
        filename="wire_escape.c",
        expected_tests=199,
        max_new_tokens=1024,
        prompt=(
            "Implement this bounded C11 frame encoder:\n\n"
            "bool encode_wire_frame(const uint8_t *source, size_t length,\n"
            "                       uint8_t *destination, size_t capacity,\n"
            "                       size_t *written);\n\n"
            "Copy ordinary bytes unchanged. Encode 0x7E as 0x7D 0x5E and 0x7D as 0x7D 0x5D. "
            "The written pointer is always required. For a zero-length input, source and "
            "destination may be null; set *written to zero and return true. For nonzero input, "
            "source and destination are required. If an argument is invalid or capacity is too "
            "small, return false without changing destination or *written. On success set "
            "*written to the encoded length. Source and destination do not overlap. Use only "
            "<stdbool.h>, <stddef.h>, <stdint.h>, or <limits.h>. Return exactly one c fenced "
            "source file with the implementation and required includes; do not define main."
        ),
        harness=ESCAPE_HARNESS,
    ),
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def adapter_targets(directory: Path) -> tuple[str, ...]:
    payload = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    targets = payload.get("target_modules")
    if not isinstance(targets, list) or not targets or not all(isinstance(v, str) for v in targets):
        raise ValueError("adapter_config.json has no valid target_modules")
    return tuple(targets)


def artifact_hashes(directory: Path) -> dict[str, str]:
    names = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    return {name: sha256_file(directory / name) for name in names if (directory / name).is_file()}


def user_prompts_from_jsonl(path: Path) -> Iterable[str]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages") if isinstance(row, dict) else None
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "user"
                        and isinstance(message.get("content"), str)
                    ):
                        yield message["content"]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return


def disjointness_audit(
    *,
    sealed_suite: Path,
    training_data: Path,
    tasks: tuple[Task, ...],
) -> dict[str, Any]:
    suite = json.loads(sealed_suite.read_text(encoding="utf-8"))
    sealed_cases = suite.get("cases")
    if not isinstance(sealed_cases, list):
        raise ValueError(f"sealed suite has no cases: {sealed_suite}")

    task_ids = {task.case_id for task in tasks}
    fixture_ids = {task.fixture_id for task in tasks}
    prompt_hashes = {sha256_text(normalized_prompt(task.prompt)) for task in tasks}
    harness_hashes = {sha256_text(task.harness.strip()) for task in tasks}
    if len(task_ids) != len(tasks) or len(fixture_ids) != len(tasks):
        raise ValueError("held-out task IDs and fixture IDs must be unique")

    sealed_case_ids = {
        str(case.get("id")) for case in sealed_cases if isinstance(case, Mapping)
    }
    sealed_prompt_hashes = {
        sha256_text(normalized_prompt(str(case.get("prompt"))))
        for case in sealed_cases
        if isinstance(case, Mapping) and isinstance(case.get("prompt"), str)
    }
    registered_fixture_ids = set(wsl_verifier.FIXTURES)
    registered_harness_hashes = {
        sha256_text(fixture.harness.strip()) for fixture in wsl_verifier.FIXTURES.values()
    }

    overlaps = {
        "case_ids": sorted(task_ids & sealed_case_ids),
        "fixture_ids": sorted(fixture_ids & registered_fixture_ids),
        "normalized_prompt_sha256": sorted(prompt_hashes & sealed_prompt_hashes),
        "harness_sha256": sorted(harness_hashes & registered_harness_hashes),
    }
    if any(overlaps.values()):
        raise ValueError(f"held-out sibling tasks overlap sealed material: {overlaps}")

    training_prompt_hashes: set[str] = set()
    training_files = 0
    training_prompts = 0
    if training_data.is_dir():
        for path in sorted(training_data.glob("*.jsonl")):
            # Provenance files do not contain model-facing chat messages.
            if ".provenance." in path.name:
                continue
            training_files += 1
            for prompt in user_prompts_from_jsonl(path):
                training_prompts += 1
                training_prompt_hashes.add(sha256_text(normalized_prompt(prompt)))
    training_overlap = sorted(prompt_hashes & training_prompt_hashes)
    if training_overlap:
        raise ValueError(
            "held-out sibling prompt appears verbatim in experiment training data: "
            f"{training_overlap}"
        )

    return {
        "status": "pass",
        "authorship": "independently authored sibling contracts and harnesses",
        "sealed_suite": str(sealed_suite.resolve()),
        "sealed_suite_sha256": sha256_file(sealed_suite),
        "sealed_case_count": len(sealed_cases),
        "registered_fixture_count_before_siblings": len(registered_fixture_ids),
        "training_data_directory": str(training_data.resolve()),
        "training_jsonl_files_scanned": training_files,
        "training_user_prompts_scanned": training_prompts,
        "overlaps": {**overlaps, "training_normalized_prompt_sha256": training_overlap},
        "task_identity": {
            task.case_id: {
                "fixture_id": task.fixture_id,
                "prompt_sha256": sha256_text(task.prompt),
                "normalized_prompt_sha256": sha256_text(normalized_prompt(task.prompt)),
                "harness_sha256": sha256_text(task.harness.strip()),
            }
            for task in tasks
        },
    }


def register_sibling_fixtures(tasks: tuple[Task, ...]) -> None:
    for task in tasks:
        if task.fixture_id in wsl_verifier.FIXTURES:
            raise ValueError(f"fixture already registered: {task.fixture_id}")
        wsl_verifier.FIXTURES[task.fixture_id] = wsl_verifier.Fixture(
            fixture_id=task.fixture_id,
            language=task.language,
            harness=task.harness,
            expected_tests=task.expected_tests,
            misra_oriented=False,
        )


def generate_one(
    engine: MixtureEngine,
    task: Task,
    *,
    route: Mapping[str, float],
    max_new_tokens: int | None,
    thinking: bool,
) -> dict[str, Any]:
    tokenizer = engine.tokenizer
    if tokenizer is None:
        raise RuntimeError("mixture engine has no tokenizer")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    device = engine.model.get_input_embeddings().weight.device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    token_budget = max_new_tokens or task.max_new_tokens
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with engine.use_route(route, normalize=False), torch.inference_mode():
        output = engine.model.generate(
            **inputs,
            max_new_tokens=token_budget,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    completion_ids = output[0, prompt_tokens:]
    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    completion_tokens = int(completion_ids.numel())
    return {
        "completion": completion,
        "completion_sha256": sha256_text(completion),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "max_new_tokens": token_budget,
        "elapsed_seconds": elapsed,
        "tokens_per_second": completion_tokens / elapsed if elapsed > 0.0 else None,
    }


def verify_generation(task: Task, generation: Mapping[str, Any], route_label: str) -> dict[str, Any]:
    completion = generation.get("completion")
    if not isinstance(completion, str):
        raise TypeError("generation completion must be a string")
    request_id = sha256_text(f"{route_label}|{task.case_id}|{completion}")[:32]
    sample = {
        "completion": completion,
        "reference": {
            "fixture_id": task.fixture_id,
            "artifact": {
                "kind": "single_fence",
                "language": task.language,
                "filename": task.filename,
            },
        },
    }
    result = wsl_verifier._evaluate(sample, request_id)  # noqa: SLF001 - experiment seam
    return {
        "case_id": task.case_id,
        "fixture_id": task.fixture_id,
        "prompt": task.prompt,
        "prompt_sha256": sha256_text(task.prompt),
        "expected_tests": task.expected_tests,
        "generation": dict(generation),
        "verification": result,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    compile_passes = 0
    task_passes = 0
    format_passes = 0
    warning_clean = 0
    tests_passed = 0
    tests_expected = 0
    task_test_fractions: list[float] = []
    statuses: dict[str, int] = {}

    for row in rows:
        verification = row["verification"]
        details = verification.get("details")
        if not isinstance(details, Mapping):
            details = {}
        status = str(verification.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        expected = int(row["expected_tests"])
        passed = int(details.get("tests_passed", 0))
        compile_passes += int(bool(details.get("compile_passed", False)))
        format_passes += int(bool(details.get("artifact_extracted", False)))
        warning_clean += int(bool(details.get("warning_clean", False)))
        task_passes += int(bool(verification.get("passed", False)))
        tests_passed += min(expected, max(0, passed))
        tests_expected += expected
        task_test_fractions.append(min(1.0, max(0.0, passed / expected)))

    task_count = len(rows)
    compile_rate = compile_passes / task_count
    mean_test_fraction = sum(task_test_fractions) / task_count
    return {
        "tasks": task_count,
        "format_passes": format_passes,
        "compile_passes": compile_passes,
        "fully_passing_tasks": task_passes,
        "warning_clean_tasks": warning_clean,
        "format_pass_rate": format_passes / task_count,
        "compile_rate": compile_rate,
        "full_task_pass_rate": task_passes / task_count,
        "task_balanced_test_fraction": mean_test_fraction,
        "raw_tests_passed": tests_passed,
        "raw_tests_expected": tests_expected,
        "raw_test_fraction": tests_passed / tests_expected,
        "score_1000": round(1000.0 * ((0.2 * compile_rate) + (0.8 * mean_test_fraction)), 3),
        "status_counts": dict(sorted(statuses.items())),
    }


def numeric_delta(expert: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "format_passes",
        "compile_passes",
        "fully_passing_tasks",
        "task_balanced_test_fraction",
        "raw_tests_passed",
        "score_1000",
    )
    return {field: expert[field] - base[field] for field in fields}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen base versus one forced adapter on fresh executable C siblings."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sealed-suite", type=Path, default=DEFAULT_SEALED_SUITE)
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override each task's conservative generation budget.",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", action="store_true", help="Enable Qwen thinking mode.")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens < 64:
        raise SystemExit("--max-new-tokens must be at least 64")
    model = args.model.resolve()
    adapter = args.adapter.resolve()
    sealed_suite = args.sealed_suite.resolve()
    training_data = args.training_data.resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model}")
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter}")

    audit = disjointness_audit(
        sealed_suite=sealed_suite,
        training_data=training_data,
        tasks=TASKS,
    )
    register_sibling_fixtures(TASKS)

    print(f"Loading frozen base {model} and adapter {adapter}", flush=True)
    engine = MixtureEngine.from_pretrained(
        str(model),
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        target_modules=adapter_targets(adapter),
        local_files_only=True,
    )
    expert_name = engine.load_expert(adapter)
    engine.model.eval()
    routes = {
        "base": {BASE_ROUTE: 1.0},
        "forced_adapter": {expert_name: 1.0},
    }

    report: dict[str, Any] = {
        "schema_version": "molor.heldout-executable-comparison.v1",
        "suite_id": "embedded-code-siblings-v1",
        "suite_purpose": (
            "Small, independently authored regression lane for comparing base and forced adapter; "
            "not a replacement for the sealed benchmark."
        ),
        "model": str(model),
        "adapter": str(adapter),
        "adapter_name": expert_name,
        "adapter_sha256": artifact_hashes(adapter),
        "disjointness_audit": audit,
        "generation": {
            "greedy": True,
            "thinking": args.thinking,
            "dtype": args.dtype,
            "device": args.device,
            "max_new_tokens_override": args.max_new_tokens,
        },
        "verification": {
            "runner": "molor.benchmark.wsl_verifier via WSL2/Bubblewrap",
            "sandboxed": True,
            "candidate_execution": True,
            "scoring": (
                "Task-balanced score_1000 = 20% compile rate + 80% mean per-task hidden-test "
                "fraction. Raw test counts are diagnostic only."
            ),
        },
        "routes": {},
    }

    for route_label, route in routes.items():
        rows: list[dict[str, Any]] = []
        for index, task in enumerate(TASKS, start=1):
            print(f"[{route_label}] {index}/{len(TASKS)} generating {task.case_id}", flush=True)
            generation = generate_one(
                engine,
                task,
                route=route,
                max_new_tokens=args.max_new_tokens,
                thinking=args.thinking,
            )
            print(f"[{route_label}] {task.case_id} compiling and testing", flush=True)
            rows.append(verify_generation(task, generation, route_label))
        report["routes"][route_label] = {
            "route": dict(route),
            "summary": summarize(rows),
            "cases": rows,
        }

    base_summary = report["routes"]["base"]["summary"]
    expert_summary = report["routes"]["forced_adapter"]["summary"]
    report["forced_adapter_minus_base"] = numeric_delta(expert_summary, base_summary)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Saved executable comparison to {output}", flush=True)
    print(
        json.dumps(
            {
                "base": base_summary,
                "forced_adapter": expert_summary,
                "forced_adapter_minus_base": report["forced_adapter_minus_base"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
