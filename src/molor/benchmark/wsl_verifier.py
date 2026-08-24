"""External C/C++ benchmark verifier using WSL2 and Bubblewrap isolation.

The process speaks ``molor.verifier.v1`` on stdin/stdout. Model completions are treated only as
file data. Compiler and test argv are fixed by the allowlisted fixture below; no completion text is
ever interpolated into a shell command. WSL's Windows mounts are hidden inside Bubblewrap except
for one fresh temporary work directory, and the sandbox has no network or ambient capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence


PROTOCOL = "molor.verifier.v1"
VERIFIER = "wsl_code_v1"
MAX_REQUEST_BYTES = 1_000_000
MAX_SOURCE_BYTES = 80_000
MAX_DIAGNOSTIC_CHARS = 8_000
MAX_CAPTURE_BYTES = 64_000
WSL_DISTRO = "Ubuntu"


@dataclass(frozen=True, slots=True)
class Fixture:
    fixture_id: str
    language: str
    harness: str
    expected_tests: int
    misra_oriented: bool = False

    @property
    def suffix(self) -> str:
        return ".cpp" if self.language == "cpp" else ".c"

    @property
    def compiler(self) -> str:
        return "/usr/bin/g++" if self.language == "cpp" else "/usr/bin/gcc"

    @property
    def standard(self) -> str:
        return "-std=c++20" if self.language == "cpp" else "-std=c11"


CAN_HARNESS = r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

extern bool decode_torque(const uint8_t data[8], int16_t *out);

static int16_t reference_value(uint16_t bits)
{
    int32_t signed_raw = (int32_t)bits;
    if ((bits & UINT16_C(0x0800)) != 0U) {
        signed_raw -= INT32_C(4096);
    }
    return (int16_t)((signed_raw * INT32_C(5)) - INT32_C(200));
}

int main(void)
{
    unsigned passed = 0U;
    unsigned total = 0U;
    uint8_t frame[8] = {0U};
    for (uint16_t raw = 0U; raw < UINT16_C(4096); ++raw) {
        frame[1] = (uint8_t)(raw & UINT16_C(0x00ff));
        frame[2] = (uint8_t)((raw >> 8U) | UINT16_C(0x00a0));
        int16_t output = INT16_C(1234);
        ++total;
        if (decode_torque(frame, &output) && (output == reference_value(raw))) {
            ++passed;
        }
    }
    int16_t sentinel = INT16_C(2718);
    ++total;
    if ((!decode_torque(NULL, &sentinel)) && (sentinel == INT16_C(2718))) {
        ++passed;
    }
    ++total;
    if (!decode_torque(frame, NULL)) {
        ++passed;
    }
    (void)printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


RING_HARNESS = r"""
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint8_t storage[17];
    uint8_t head;
    uint8_t tail;
    uint8_t count;
} ring_t;

extern void ring_init(ring_t *ring);
extern bool ring_push(ring_t *ring, uint8_t value);
extern bool ring_pop(ring_t *ring, uint8_t *value);

static bool state_valid(const ring_t *ring)
{
    return (ring->head < UINT8_C(17)) && (ring->tail < UINT8_C(17)) &&
           (ring->count <= UINT8_C(17));
}

int main(void)
{
    unsigned passed = 0U;
    unsigned total = 0U;
    ring_t first;
    ring_t second;
    ring_init(&first);
    ring_init(&second);
    ++total;
    if (state_valid(&first) && (first.count == 0U) && state_valid(&second)) {
        ++passed;
    }
    for (unsigned round = 0U; round < 4000U; ++round) {
        for (uint8_t index = 0U; index < UINT8_C(17); ++index) {
            ++total;
            if (ring_push(&first, (uint8_t)(round + index)) && state_valid(&first)) {
                ++passed;
            }
        }
        ring_t full_snapshot = first;
        ++total;
        if ((!ring_push(&first, UINT8_C(0xee))) &&
            (first.head == full_snapshot.head) && (first.tail == full_snapshot.tail) &&
            (first.count == full_snapshot.count)) {
            ++passed;
        }
        for (uint8_t index = 0U; index < UINT8_C(17); ++index) {
            uint8_t output = UINT8_C(0xff);
            const uint8_t expected = (uint8_t)(round + index);
            ++total;
            if (ring_pop(&first, &output) && (output == expected) && state_valid(&first)) {
                ++passed;
            }
        }
        ring_t empty_snapshot = first;
        uint8_t output = UINT8_C(0xa5);
        ++total;
        if ((!ring_pop(&first, &output)) && (output == UINT8_C(0xa5)) &&
            (first.head == empty_snapshot.head) && (first.tail == empty_snapshot.tail) &&
            (first.count == empty_snapshot.count)) {
            ++passed;
        }
        ++total;
        if (ring_push(&second, (uint8_t)round) && state_valid(&second)) {
            uint8_t ignored = 0U;
            (void)ring_pop(&second, &ignored);
            ++passed;
        }
    }
    ring_init(NULL);
    ++total;
    if ((!ring_push(NULL, 1U)) && (!ring_pop(NULL, NULL))) {
        ++passed;
    }
    (void)printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


WINDOW_HARNESS = r"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <vector>

extern std::size_t longest_span_window(std::span<const std::int32_t> values,
                                       std::uint64_t limit);

static std::size_t reference(std::span<const std::int32_t> values, std::uint64_t limit)
{
    std::size_t best = 0U;
    for (std::size_t left = 0U; left < values.size(); ++left) {
        std::int32_t low = values[left];
        std::int32_t high = values[left];
        for (std::size_t right = left; right < values.size(); ++right) {
            low = std::min(low, values[right]);
            high = std::max(high, values[right]);
            const auto difference = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(high) - static_cast<std::int64_t>(low));
            if (difference <= limit) {
                best = std::max(best, (right - left) + 1U);
            }
        }
    }
    return best;
}

int main()
{
    unsigned passed = 0U;
    unsigned total = 0U;
    const std::vector<std::int32_t> edges = {
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::min(),
    };
    for (const std::uint64_t limit : {0ULL, 1ULL, 4294967295ULL}) {
        ++total;
        if (longest_span_window(edges, limit) == reference(edges, limit)) {
            ++passed;
        }
    }
    std::mt19937 generator(0x5a17c3d1U);
    for (unsigned trial = 0U; trial < 320U; ++trial) {
        const std::size_t size = static_cast<std::size_t>(generator() % 36U);
        std::vector<std::int32_t> values(size);
        for (auto &value : values) {
            value = static_cast<std::int32_t>(generator());
        }
        const std::uint64_t limit = static_cast<std::uint64_t>(generator());
        ++total;
        if (longest_span_window(values, limit) == reference(values, limit)) {
            ++passed;
        }
    }
    std::vector<std::int32_t> large(60000U);
    for (std::size_t index = 0U; index < large.size(); ++index) {
        large[index] = (index % 2U == 0U) ? std::numeric_limits<std::int32_t>::min()
                                         : std::numeric_limits<std::int32_t>::max();
    }
    ++total;
    if (longest_span_window(large, UINT64_MAX) == large.size()) {
        ++passed;
    }
    (void)std::printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


CRC_HARNESS = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern uint16_t crc16_ccitt_false(const uint8_t *data, size_t length);

static uint16_t reference_crc(const uint8_t *data, size_t length)
{
    uint16_t crc = UINT16_C(0xffff);
    for (size_t index = 0U; index < length; ++index) {
        crc ^= (uint16_t)((uint16_t)data[index] << 8U);
        for (unsigned bit = 0U; bit < 8U; ++bit) {
            crc = ((crc & UINT16_C(0x8000)) != 0U)
                      ? (uint16_t)((uint16_t)(crc << 1U) ^ UINT16_C(0x1021))
                      : (uint16_t)(crc << 1U);
        }
    }
    return crc;
}

int main(void)
{
    unsigned passed = 0U;
    unsigned total = 0U;
    static const uint8_t check[] = {'1','2','3','4','5','6','7','8','9'};
    ++total;
    if (crc16_ccitt_false(check, sizeof(check)) == UINT16_C(0x29b1)) {
        ++passed;
    }
    ++total;
    if (crc16_ccitt_false(NULL, 0U) == UINT16_C(0xffff)) {
        ++passed;
    }
    uint32_t state = UINT32_C(0x8d31f2a7);
    uint8_t bytes[257];
    for (size_t index = 0U; index < sizeof(bytes); ++index) {
        state = (state * UINT32_C(1664525)) + UINT32_C(1013904223);
        bytes[index] = (uint8_t)(state >> 24U);
    }
    for (size_t length = 0U; length <= sizeof(bytes); ++length) {
        ++total;
        if (crc16_ccitt_false(bytes, length) == reference_crc(bytes, length)) {
            ++passed;
        }
    }
    (void)printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


PPS_HARNESS = r"""
#include <stdint.h>
#include <stdio.h>

volatile uint32_t ANSELB;
volatile uint32_t TRISB;
volatile uint32_t RPB7R;
volatile uint32_t U1RXR;

extern void pps_uart1_init(void);

static int run_case(uint32_t analog, uint32_t direction, uint32_t tx, uint32_t rx)
{
    ANSELB = analog;
    TRISB = direction;
    RPB7R = tx;
    U1RXR = rx;
    pps_uart1_init();
    const uint32_t pin_mask = (UINT32_C(1) << 7U) | (UINT32_C(1) << 13U);
    const uint32_t expected_analog = analog & ~pin_mask;
    const uint32_t expected_direction =
        (direction & ~(UINT32_C(1) << 7U)) | (UINT32_C(1) << 13U);
    return (ANSELB == expected_analog) && (TRISB == expected_direction) &&
           (RPB7R == UINT32_C(1)) && (U1RXR == UINT32_C(3));
}

int main(void)
{
    unsigned passed = 0U;
    unsigned total = 0U;
    static const uint32_t patterns[] = {
        UINT32_C(0), UINT32_C(0xffffffff), UINT32_C(0xaaaaaaaa), UINT32_C(0x13579bdf)
    };
    for (unsigned a = 0U; a < 4U; ++a) {
        for (unsigned b = 0U; b < 4U; ++b) {
            ++total;
            if (run_case(patterns[a], patterns[b], patterns[b], patterns[a])) {
                ++passed;
            }
        }
    }
    (void)printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


MASTERMIND_HARNESS = r"""
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>
#include <vector>

struct mastermind_clue {
    std::array<std::uint8_t, 4> guess;
    std::uint8_t exact;
    std::uint8_t misplaced;
};

extern bool solve_mastermind(std::span<const mastermind_clue> clues,
                             std::array<std::uint8_t, 4> &solution);

static std::array<std::uint8_t, 4> decode_code(std::size_t encoded)
{
    std::array<std::uint8_t, 4> code{};
    for (std::size_t offset = 0U; offset < code.size(); ++offset) {
        code[(code.size() - 1U) - offset] = static_cast<std::uint8_t>(encoded % 6U);
        encoded /= 6U;
    }
    return code;
}

static std::pair<std::uint8_t, std::uint8_t> feedback(
    const std::array<std::uint8_t, 4> &secret,
    const std::array<std::uint8_t, 4> &guess)
{
    std::uint8_t exact = 0U;
    std::array<unsigned, 6> secret_counts{};
    std::array<unsigned, 6> guess_counts{};
    for (std::size_t index = 0U; index < secret.size(); ++index) {
        if (secret[index] == guess[index]) {
            ++exact;
        } else {
            ++secret_counts[secret[index]];
            ++guess_counts[guess[index]];
        }
    }
    unsigned misplaced = 0U;
    for (std::size_t digit = 0U; digit < secret_counts.size(); ++digit) {
        misplaced += std::min(secret_counts[digit], guess_counts[digit]);
    }
    return {exact, static_cast<std::uint8_t>(misplaced)};
}

static bool matches(const std::array<std::uint8_t, 4> &candidate,
                    const mastermind_clue &clue)
{
    return feedback(candidate, clue.guess) ==
           std::pair<std::uint8_t, std::uint8_t>{clue.exact, clue.misplaced};
}

int main()
{
    unsigned passed = 0U;
    unsigned total = 0U;
    for (std::size_t trial = 0U; trial < 48U; ++trial) {
        const auto secret = decode_code(((trial * 547U) + 233U) % 1296U);
        std::vector<std::array<std::uint8_t, 4>> candidates;
        candidates.reserve(1296U);
        for (std::size_t encoded = 0U; encoded < 1296U; ++encoded) {
            candidates.push_back(decode_code(encoded));
        }
        std::vector<mastermind_clue> clues;
        std::uint32_t state = static_cast<std::uint32_t>(0x9e3779b9U + trial);
        while (candidates.size() > 1U) {
            state = (state * UINT32_C(1664525)) + UINT32_C(1013904223);
            std::size_t choice = static_cast<std::size_t>(state) % candidates.size();
            if (candidates[choice] == secret) {
                choice = (choice + 1U) % candidates.size();
            }
            const auto guess = candidates[choice];
            const auto result = feedback(secret, guess);
            clues.push_back(mastermind_clue{guess, result.first, result.second});
            candidates.erase(
                std::remove_if(candidates.begin(), candidates.end(),
                    [&](const auto &candidate) { return !matches(candidate, clues.back()); }),
                candidates.end());
        }
        std::array<std::uint8_t, 4> answer{9U, 9U, 9U, 9U};
        ++total;
        if (solve_mastermind(std::span<const mastermind_clue>(clues), answer) &&
            (answer == secret)) {
            ++passed;
        }
    }

    const std::array<std::uint8_t, 4> sentinel{9U, 8U, 7U, 6U};
    std::array<std::uint8_t, 4> answer = sentinel;
    ++total;
    if ((!solve_mastermind(std::span<const mastermind_clue>{}, answer)) &&
        (answer == sentinel)) {
        ++passed;
    }

    const std::array<std::uint8_t, 4> zero_guess{0U, 0U, 0U, 0U};
    const std::array<mastermind_clue, 2> contradictory{{
        mastermind_clue{zero_guess, 4U, 0U},
        mastermind_clue{zero_guess, 0U, 0U}
    }};
    answer = sentinel;
    ++total;
    if ((!solve_mastermind(contradictory, answer)) && (answer == sentinel)) {
        ++passed;
    }

    const std::array<mastermind_clue, 1> invalid_digit{{
        mastermind_clue{{0U, 1U, 2U, 6U}, 0U, 0U}
    }};
    answer = sentinel;
    ++total;
    if ((!solve_mastermind(invalid_digit, answer)) && (answer == sentinel)) {
        ++passed;
    }

    const std::array<mastermind_clue, 1> invalid_counts{{
        mastermind_clue{zero_guess, 3U, 2U}
    }};
    answer = sentinel;
    ++total;
    if ((!solve_mastermind(invalid_counts, answer)) && (answer == sentinel)) {
        ++passed;
    }

    (void)std::printf("MOLOR_TESTS %u %u\n", passed, total);
    return (passed == total) ? 0 : 1;
}
"""


FIXTURES: dict[str, Fixture] = {
    "algorithms/longest-span-window-v1": Fixture(
        "algorithms/longest-span-window-v1", "cpp", WINDOW_HARNESS, 324
    ),
    "crypto/crc16-ccitt-false-v1": Fixture(
        "crypto/crc16-ccitt-false-v1", "c", CRC_HARNESS, 260
    ),
    "embedded/decode-torque-misra-v1": Fixture(
        "embedded/decode-torque-misra-v1", "c", CAN_HARNESS, 4098, misra_oriented=True
    ),
    "embedded/ring-buffer-misra-v1": Fixture(
        "embedded/ring-buffer-misra-v1", "c", RING_HARNESS, 148002, misra_oriented=True
    ),
    "pic32/uart1-pps-mx270-v1": Fixture(
        "pic32/uart1-pps-mx270-v1", "c", PPS_HARNESS, 16
    ),
    "logic/mastermind-unique-v1": Fixture(
        "logic/mastermind-unique-v1", "cpp", MASTERMIND_HARNESS, 52
    ),
}


_FENCE = re.compile(r"```([A-Za-z0-9_+.-]*)\s*\r?\n(.*?)```", re.DOTALL)
_BANNED_MISRA = (
    re.compile(r"\b(?:malloc|calloc|realloc|free)\s*\("),
    re.compile(r"\bgoto\b"),
    re.compile(r"\bunion\b"),
    re.compile(r"#\s*pragma\b"),
    re.compile(r"cppcheck-suppress", re.IGNORECASE),
    re.compile(r"\b(?:asm|__asm__)\b"),
    re.compile(r"#\s*include\s*[\"']"),
)


def _reply(
    request_id: str,
    status: str,
    score: float | None,
    *,
    passed: bool = False,
    details: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "status": status,
        "score": score,
        "passed": passed,
        "details": dict(details or {}),
    }
    if error is not None:
        payload["error"] = error[:4000]
    return payload


def _extract_source(completion: str, fixture: Fixture, artifact: Mapping[str, Any]) -> str:
    if len(completion.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("completion exceeds source-size limit")
    expected_language = str(artifact.get("language") or fixture.language).casefold()
    aliases = {"c"} if fixture.language == "c" else {"cpp", "c++", "cc", "cxx"}
    if expected_language not in aliases:
        raise ValueError("artifact language differs from fixture language")
    matches = [body for language, body in _FENCE.findall(completion) if language.casefold() in aliases]
    if len(matches) != 1:
        raise ValueError("completion must contain exactly one matching fenced source file")
    source = matches[0].strip() + "\n"
    if not source.strip() or "\x00" in source:
        raise ValueError("source is empty or contains NUL")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("extracted source exceeds source-size limit")
    return source


def _limited_process(command: Sequence[str], *, timeout: float) -> tuple[int, str, str]:
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    streams = (process.stdout, process.stderr)
    buffers: list[list[bytes]] = [[], []]
    sizes = [0, 0]
    overflow = threading.Event()

    def drain(index: int) -> None:
        stream = streams[index]
        if stream is None:
            return
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            remaining = MAX_CAPTURE_BYTES - sizes[index]
            if remaining > 0:
                buffers[index].append(chunk[:remaining])
                sizes[index] += min(len(chunk), remaining)
            if len(chunk) > remaining:
                overflow.set()
                return

    threads = [threading.Thread(target=drain, args=(index,), daemon=True) for index in range(2)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if overflow.is_set():
            process.kill()
            break
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=2.0)
            raise subprocess.TimeoutExpired(list(command), timeout)
        time.sleep(0.01)
    process.wait(timeout=2.0)
    for thread in threads:
        thread.join(timeout=1.0)
    decoded = [b"".join(parts).decode("utf-8", errors="replace") for parts in buffers]
    stdout, stderr = decoded
    if overflow.is_set():
        stderr += "\n<process terminated after exceeding output limit>"
    if len(stdout) > MAX_DIAGNOSTIC_CHARS:
        stdout = stdout[:MAX_DIAGNOSTIC_CHARS] + "...<truncated>"
    if len(stderr) > MAX_DIAGNOSTIC_CHARS:
        stderr = stderr[:MAX_DIAGNOSTIC_CHARS] + "...<truncated>"
    return process.returncode, stdout, stderr


def _wsl_path(path: Path) -> str:
    code, stdout, stderr = _limited_process(
        [
            "wsl",
            "-d",
            WSL_DISTRO,
            "--exec",
            "/usr/bin/wslpath",
            "-a",
            "-u",
            str(path.resolve()),
        ],
        timeout=10.0,
    )
    value = stdout.strip()
    if code != 0 or not value.startswith("/mnt/"):
        raise RuntimeError(f"could not translate verifier work directory: {stderr or stdout}")
    return value


def _sandbox(work: Path, argv: Sequence[str], *, timeout: float = 10.0) -> tuple[int, str, str]:
    wsl_work = _wsl_path(work)
    command = [
        "wsl",
        "-d",
        WSL_DISTRO,
        "--exec",
        "/usr/bin/timeout",
        "-s",
        "KILL",
        f"{max(1.0, timeout - 1.0):g}s",
        "/usr/bin/bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--disable-userns",
        "--cap-drop",
        "ALL",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--tmpfs",
        "/",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/sbin",
        "/sbin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--bind",
        wsl_work,
        "/work",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--chdir",
        "/work",
        "/usr/bin/prlimit",
        "--as=1073741824",
        "--cpu=4",
        "--nproc=32",
        "--nofile=64",
        "--fsize=16777216",
        "--",
        *argv,
    ]
    return _limited_process(command, timeout=timeout)


def _compiler_flags(fixture: Fixture) -> list[str]:
    return [
        fixture.standard,
        "-O1",
        "-g0",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Wconversion",
        "-Wshadow",
        "-Werror=return-type",
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        "-fno-omit-frame-pointer",
    ]


def _misra_scan(work: Path, source_filename: str, source: str) -> tuple[float, dict[str, Any]]:
    banned = [pattern.pattern for pattern in _BANNED_MISRA if pattern.search(source)]
    allowed_headers = {"stdbool.h", "stddef.h", "stdint.h", "limits.h"}
    angle_headers = re.findall(r"#\s*include\s*<([^>]+)>", source)
    invalid_headers = sorted(set(angle_headers).difference(allowed_headers))
    if banned or invalid_headers:
        return 0.0, {
            "banned_constructs": banned,
            "invalid_headers": invalid_headers,
            "cppcheck_available": True,
            "sandboxed": True,
        }

    cppcheck = "/usr/bin/cppcheck"
    addon = "/usr/lib/x86_64-linux-gnu/cppcheck/addons/misra.py"
    command = [
        cppcheck,
        "--std=c11",
        "--language=c",
        "--enable=warning,style,performance,portability",
        "--check-level=normal",
        "--addon-python=/usr/bin/python3",
        f"--addon={addon}",
        # The benchmark API deliberately has external linkage so the hidden harness can call it.
        # Rule 8.7 cannot be judged from the candidate translation unit in isolation.
        "--suppress=misra-c2012-8.7",
        "--error-exitcode=2",
        "--template={severity}:{id}:{message}",
        source_filename,
    ]
    try:
        code, stdout, stderr = _sandbox(work, command, timeout=15.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"sandboxed Cppcheck failed: {type(exc).__name__}: {exc}") from exc
    diagnostics = (stdout + "\n" + stderr).strip()
    if code not in {0, 2}:
        raise RuntimeError(f"sandboxed Cppcheck exited {code}: {diagnostics[:1000]}")
    return float(code == 0), {
        "cppcheck_available": True,
        "addon_available": True,
        "sandboxed": True,
        "declared_deviations": [
            "MISRA C:2012 Rule 8.7 (external linkage required by benchmark harness)"
        ],
        "clean": code == 0,
        "diagnostics": diagnostics[:MAX_DIAGNOSTIC_CHARS],
    }


def _evaluate(sample: Mapping[str, Any], request_id: str) -> dict[str, Any]:
    completion = sample.get("completion")
    reference = sample.get("reference")
    if not isinstance(completion, str) or not isinstance(reference, Mapping):
        return _reply(
            request_id,
            "invalid_reference",
            None,
            error="sample completion/reference has the wrong type",
        )
    fixture_id = reference.get("fixture_id")
    fixture = FIXTURES.get(fixture_id) if isinstance(fixture_id, str) else None
    artifact = reference.get("artifact")
    if fixture is None or not isinstance(artifact, Mapping):
        return _reply(
            request_id,
            "invalid_reference",
            None,
            error="unknown fixture or missing artifact contract",
        )
    try:
        source = _extract_source(completion, fixture, artifact)
    except ValueError as exc:
        return _reply(request_id, "format_fail", 0.0, error=str(exc))

    details: dict[str, Any] = {
        "fixture_id": fixture.fixture_id,
        "artifact_extracted": True,
        "compile_passed": False,
        "tests_passed": 0,
        "tests_total": 0,
        "warning_clean": False,
        "misra_oriented": fixture.misra_oriented,
    }
    with tempfile.TemporaryDirectory(prefix="molor-wsl-eval-") as temporary:
        work = Path(temporary).resolve()
        candidate = work / f"candidate{fixture.suffix}"
        candidate.write_text(source, encoding="utf-8", newline="\n")
        policy_score = 1.0
        if fixture.misra_oriented:
            policy_score, policy_details = _misra_scan(work, candidate.name, source)
            details["misra_addon"] = policy_details
        compile_argv = [
            fixture.compiler,
            *_compiler_flags(fixture),
            "-c",
            f"candidate{fixture.suffix}",
            "-o",
            "candidate.o",
        ]
        try:
            compile_code, compile_stdout, compile_stderr = _sandbox(work, compile_argv)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return _reply(
                request_id,
                "infra_error",
                None,
                details=details,
                error=f"sandbox compiler unavailable: {type(exc).__name__}: {exc}",
            )
        warnings = len(re.findall(r"\bwarning:", compile_stderr, flags=re.IGNORECASE))
        details["compiler_diagnostics"] = (compile_stdout + compile_stderr)[
            :MAX_DIAGNOSTIC_CHARS
        ]
        details["warning_count"] = warnings
        details["warning_clean"] = warnings == 0
        if compile_code != 0:
            score = 0.05 + (0.10 * policy_score)
            return _reply(request_id, "compile_fail", score, details=details)
        details["compile_passed"] = True

        candidate.unlink(missing_ok=True)
        result_token = secrets.token_hex(16)
        harness_source = fixture.harness.strip().replace(
            "MOLOR_TESTS ", f"MOLOR_TESTS {result_token} "
        )
        harness = work / f"harness{fixture.suffix}"
        harness.write_text(harness_source + "\n", encoding="utf-8", newline="\n")
        harness_argv = [
            fixture.compiler,
            *_compiler_flags(fixture),
            "-c",
            f"harness{fixture.suffix}",
            "-o",
            "harness.o",
        ]
        harness_code, harness_stdout, harness_stderr = _sandbox(work, harness_argv)
        if harness_code != 0:
            return _reply(
                request_id,
                "infra_error",
                None,
                details=details,
                error=(harness_stdout + harness_stderr)[:MAX_DIAGNOSTIC_CHARS],
            )
        harness.unlink(missing_ok=True)
        link_argv = [
            fixture.compiler,
            "candidate.o",
            "harness.o",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-o",
            "tests.bin",
        ]
        link_code, link_stdout, link_stderr = _sandbox(work, link_argv)
        if link_code != 0:
            details["link_diagnostics"] = (link_stdout + link_stderr)[:MAX_DIAGNOSTIC_CHARS]
            score = 0.25 + (0.10 * float(warnings == 0)) + (0.10 * policy_score)
            return _reply(request_id, "compile_fail", score, details=details)
        for artifact_path in (work / "candidate.o", work / "harness.o"):
            artifact_path.unlink(missing_ok=True)
        try:
            run_code, run_stdout, run_stderr = _sandbox(work, ["/work/tests.bin"], timeout=12.0)
        except subprocess.TimeoutExpired:
            run_code, run_stdout, run_stderr = 124, "", "test wall-time limit exceeded"
        summary_pattern = re.compile(
            rf"(?m)^MOLOR_TESTS {re.escape(result_token)} (\d+) (\d+)\r?$"
        )
        summaries = summary_pattern.findall(run_stdout)
        result_protocol_valid = len(summaries) == 1
        if result_protocol_valid:
            test_passed, test_total = (int(value) for value in summaries[0])
            result_protocol_valid = (
                test_total == fixture.expected_tests and 0 <= test_passed <= test_total
            )
        if not result_protocol_valid:
            test_passed, test_total = 0, fixture.expected_tests
        details["tests_passed"] = test_passed
        details["tests_total"] = test_total
        details["expected_tests"] = fixture.expected_tests
        details["result_protocol_valid"] = result_protocol_valid
        details["test_exit_code"] = run_code
        details["test_diagnostics"] = run_stderr[:MAX_DIAGNOSTIC_CHARS]
        test_fraction = test_passed / fixture.expected_tests
        if run_code != 0 and math.isclose(test_fraction, 1.0, abs_tol=1e-12):
            test_fraction = 0.75
        score = (
            0.05
            + 0.20
            + (0.55 * test_fraction)
            + (0.10 * float(warnings == 0))
            + (0.10 * policy_score)
        )
        score = min(1.0, max(0.0, score))
        passed = run_code == 0 and math.isclose(score, 1.0, abs_tol=1e-12)
        if passed:
            status = "pass"
        elif fixture.misra_oriented and policy_score < 1.0 and run_code == 0:
            status = "policy_violation"
        else:
            status = "test_fail"
        return _reply(request_id, status, score, passed=passed, details=details)


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        response = _reply("unknown", "infra_error", None, error="request exceeds size limit")
    else:
        try:
            request = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(request, Mapping) or request.get("protocol") != PROTOCOL:
                raise ValueError("invalid verifier protocol")
            request_id = request.get("request_id")
            sample = request.get("sample")
            if not isinstance(request_id, str) or not isinstance(sample, Mapping):
                raise ValueError("invalid request id or sample")
            response = _evaluate(sample, request_id)
        except Exception as exc:
            request_id = "unknown"
            if isinstance(locals().get("request"), Mapping):
                candidate_id = request.get("request_id")
                if isinstance(candidate_id, str):
                    request_id = candidate_id
            response = _reply(
                request_id,
                "infra_error",
                None,
                error=f"{type(exc).__name__}: {exc}",
            )
    json.dump(response, sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
