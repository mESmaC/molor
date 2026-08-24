"""Build a conservative terminal/tool-use SFT shard from the local Fable traces.

The prepared source is read-only.  Trainer JSONL rows contain exactly ``messages``;
license and source metadata are emitted to matching provenance sidecars.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


UPSTREAM = "greghavens/fable-5-coding-and-debugging-traces"
UPSTREAM_URL = (
    "https://huggingface.co/datasets/greghavens/"
    "fable-5-coding-and-debugging-traces"
)
LICENSE = "CC BY 4.0"
DEFAULT_SOURCE = (
    Path.home() / "Documents" / "datasets" / "prepared" / "complete-fable-5-traces"
)
MIN_CONTENT_CHARS = 256
HARD_MAX_CONTENT_CHARS = 32768
DEFAULT_MAX_CONTENT_CHARS = 16384

# These are synthetic/meta tasks rather than examples of useful terminal work.
EXCLUDED_CATEGORIES = {
    "seed-authoring",
    "full-distill-brownfield-orientation",
    "persistent-memory",
    "build-game",
    "project-game",
}

TOOL_MARKER_RE = re.compile(
    r"(?m)^[ \t]*\[Tool:[ \t]*([A-Za-z][A-Za-z0-9_. -]{0,40})\][ \t]*$"
)
ACTION_RE = re.compile(r"(?m)^\[Tool: (?:Bash|PowerShell)\]$")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WHITESPACE_RE = re.compile(r"\s+")

MOJIBAKE_MARKERS = ("\ufffd", "Ã", "Â", "â€", "ðŸ")
TRUNCATION_RE = re.compile(
    r"(?i)(?:\[?…?\s*earlier\s+truncated\s*…?\]?|"
    r"\[(?:output|content|text)?\s*truncated\]|"
    r"\b(?:output|content|response|conversation|context)\s+(?:was\s+|is\s+)?truncated\b)"
)
LOCAL_COMMAND_RE = re.compile(
    r"(?i)(?:local-command-caveat|<command-name>|<local-command-stdout>)"
)
PRIVATE_PATH_RE = re.compile(
    r"(?i)(?:/(?:home|Users)/[^/\s]+(?:/|\b)|"
    r"/root(?:/|\b)|"
    r"[A-Z]:\\Users\\[^\\\s]+(?:\\|\b)|"
    r"[A-Z]:/Users/[^/\s]+(?:/|\b)|"
    r"/tmp/(?:claude|codex)-[^/\s]+)"
)
CREDENTIAL_RE = re.compile(
    r"(?ix)(?:"
    r"-----BEGIN\s+[A-Z ]*PRIVATE\s+KEY-----|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bgh[opusr]_[A-Za-z0-9]{20,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b|"
    r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b|"
    r"\bAIza[0-9A-Za-z_-]{30,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}|"
    r"(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*"
    r"[\"'][^\"'\r\n]{8,}[\"']"
    r")"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class Candidate:
    task_hash: str
    messages: list[dict[str, str]]
    source_id: str
    source_partition: str
    category: str
    content_chars: int
    action_count: int
    completion_chars: int

    @property
    def quality(self) -> tuple[int, int, int, str]:
        """Prefer complete, exercised trajectories; source id is a stable tie-breaker."""

        return (
            self.action_count,
            self.content_chars,
            self.completion_chars,
            self.source_id,
        )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: str, purpose: str, task_hash: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{purpose}\0{task_hash}".encode()).digest()


def canonical_key(system: str, original_user: str) -> str:
    # Key normalization is deliberately stronger than content normalization so that
    # whitespace-only prompt variants cannot leak across the split.
    normalized = (
        WHITESPACE_RE.sub(" ", system).strip()
        + "\0"
        + WHITESPACE_RE.sub(" ", original_user).strip()
    )
    return sha256_text(normalized)


def normalize_content(content: str) -> str:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = ANSI_RE.sub("", content)
    content = TOOL_MARKER_RE.sub(lambda match: f"[Tool: {match.group(1).strip()}]", content)
    content = "\n".join(line.rstrip() for line in content.splitlines())
    content = re.sub(r"\n{4,}", "\n\n\n", content)
    return content.strip()


def clean_chat(value: object) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(value, list) or not 2 <= len(value) <= 8:
        return None, "message_count"

    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            return None, "message_schema"
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None, "message_schema"
        content = normalize_content(content)
        if not content:
            return None, "empty_message"
        messages.append({"role": role, "content": content})

    offset = 0
    if messages[0]["role"] == "system":
        offset = 1
        if any(message["role"] == "system" for message in messages[1:]):
            return None, "role_order"
    expected = "user"
    for message in messages[offset:]:
        if message["role"] != expected:
            return None, "role_order"
        expected = "assistant" if expected == "user" else "user"
    if not messages[offset:] or messages[-1]["role"] != "assistant":
        return None, "role_order"
    return messages, None


def first_user(messages: list[dict[str, str]]) -> str:
    return next(message["content"] for message in messages if message["role"] == "user")


def system_prompt(messages: list[dict[str, str]]) -> str:
    if messages[0]["role"] == "system":
        return messages[0]["content"]
    return ""


def completion_tail_chars(content: str) -> int:
    """Count prose after the final tool payload; zero means an unfinished tool call."""

    markers = list(TOOL_MARKER_RE.finditer(content))
    if not markers:
        return len(content)
    tail = content[markers[-1].end() :].lstrip()
    try:
        _, end = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError:
        # A malformed payload is not a trustworthy completed trajectory.
        return 0
    prose = tail[end:].strip()
    return len(prose) if len(prose.split()) >= 12 else 0


def unsafe_reason(text: str) -> str | None:
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return "mojibake"
    if CONTROL_RE.search(text):
        return "control_character"
    if TRUNCATION_RE.search(text):
        return "truncation"
    if LOCAL_COMMAND_RE.search(text):
        return "local_command_caveat"
    if PRIVATE_PATH_RE.search(text):
        return "private_user_path"
    if CREDENTIAL_RE.search(text):
        return "credential_material"
    return None


def source_rows(paths: Iterable[tuple[str, Path]]) -> Iterator[tuple[str, int, dict[str, Any]]]:
    for partition, path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                yield partition, line_number, row


def build_candidates(
    paths: list[tuple[str, Path]],
    *,
    max_content_chars: int,
) -> tuple[dict[str, Candidate], dict[str, Any]]:
    tasks: dict[str, Candidate] = {}
    stats: dict[str, Any] = {
        "all_rows": 0,
        "upstream_rows": 0,
        "eligible_rows": 0,
        "rejected": collections.Counter(),
        "eligible_categories": collections.Counter(),
    }

    for partition, line_number, row in source_rows(paths):
        stats["all_rows"] += 1
        if row.get("upstream") != UPSTREAM:
            stats["rejected"]["different_upstream"] += 1
            continue
        stats["upstream_rows"] += 1

        category = str(row.get("category") or "unknown")
        if category in EXCLUDED_CATEGORIES:
            stats["rejected"]["excluded_meta_category"] += 1
            continue

        messages, reason = clean_chat(row.get("messages"))
        if messages is None:
            stats["rejected"][str(reason)] += 1
            continue
        text = "\n".join(message["content"] for message in messages)
        content_chars = sum(len(message["content"]) for message in messages)
        if not MIN_CONTENT_CHARS <= content_chars <= max_content_chars:
            stats["rejected"]["content_length"] += 1
            continue
        reason = unsafe_reason(text)
        if reason:
            stats["rejected"][reason] += 1
            continue

        assistant_text = "\n".join(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        action_count = len(ACTION_RE.findall(assistant_text))
        if not action_count:
            stats["rejected"]["no_bash_or_powershell_action"] += 1
            continue
        completion_chars = completion_tail_chars(messages[-1]["content"])
        if completion_chars < 80:
            stats["rejected"]["unfinished_trajectory"] += 1
            continue

        task_hash = canonical_key(system_prompt(messages), first_user(messages))
        source_id = str(row.get("id") or f"{partition}:{line_number}")
        candidate = Candidate(
            task_hash=task_hash,
            messages=messages,
            source_id=source_id,
            source_partition=partition,
            category=category,
            content_chars=content_chars,
            action_count=action_count,
            completion_chars=completion_chars,
        )
        previous = tasks.get(task_hash)
        if previous is None or candidate.quality > previous.quality:
            tasks[task_hash] = candidate
        stats["eligible_rows"] += 1
        stats["eligible_categories"][category] += 1

    stats["unique_eligible_tasks"] = len(tasks)
    stats["duplicate_eligible_rows"] = stats["eligible_rows"] - len(tasks)
    return tasks, stats


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count, sha256_file(path)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[round((len(ordered) - 1) * fraction)]


def emit_split(
    name: str,
    candidates: list[Candidate],
    output_dir: Path,
) -> dict[str, Any]:
    data_path = output_dir / f"terminal-cli{name}.jsonl"
    provenance_path = output_dir / f"terminal-cli{name}.provenance.jsonl"
    ordered = sorted(candidates, key=lambda candidate: candidate.task_hash)

    data_rows = ({"messages": candidate.messages} for candidate in ordered)
    provenance_rows = (
        {
            "task_sha256": candidate.task_hash,
            "source": UPSTREAM,
            "source_url": UPSTREAM_URL,
            "source_id": candidate.source_id,
            "source_partition": candidate.source_partition,
            "category": candidate.category,
            "license": LICENSE,
            "selection": "best completed safe trajectory per system+original-user task",
            "content_chars": candidate.content_chars,
            "bash_or_powershell_actions": candidate.action_count,
        }
        for candidate in ordered
    )
    rows, digest = write_jsonl_atomic(data_path, data_rows)
    provenance_rows_count, provenance_digest = write_jsonl_atomic(
        provenance_path, provenance_rows
    )
    if rows != provenance_rows_count:
        raise RuntimeError(f"provenance mismatch for {data_path}")

    lengths = [candidate.content_chars for candidate in ordered]
    return {
        "path": str(data_path),
        "provenance_path": str(provenance_path),
        "rows": rows,
        "sha256": digest,
        "provenance_sha256": provenance_digest,
        "content_chars": {
            "min": min(lengths, default=0),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "categories": dict(
            sorted(collections.Counter(item.category for item in ordered).items())
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--validation-size", type=int, default=60)
    parser.add_argument(
        "--max-content-chars", type=int, default=DEFAULT_MAX_CONTENT_CHARS
    )
    parser.add_argument("--seed", default="terminal-cli-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.validation_size < args.limit:
        raise ValueError("validation-size must be positive and smaller than limit")
    if not MIN_CONTENT_CHARS <= args.max_content_chars <= HARD_MAX_CONTENT_CHARS:
        raise ValueError(
            f"max-content-chars must be {MIN_CONTENT_CHARS}..{HARD_MAX_CONTENT_CHARS}"
        )

    paths = [
        ("prepared-train", args.source_dir / "train.jsonl"),
        ("prepared-validation", args.source_dir / "validation.jsonl"),
    ]
    for _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_snapshot = {
        str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for _, path in paths
    }

    tasks, stats = build_candidates(paths, max_content_chars=args.max_content_chars)
    if len(tasks) < args.limit:
        raise RuntimeError(
            f"only {len(tasks)} unique tasks survived conservative filtering; "
            f"cannot satisfy requested limit {args.limit}"
        )

    selected = sorted(
        tasks.values(),
        key=lambda candidate: stable_rank(args.seed, "select", candidate.task_hash),
    )[: args.limit]
    split_order = sorted(
        selected,
        key=lambda candidate: stable_rank(args.seed, "split", candidate.task_hash),
    )
    validation = split_order[: args.validation_size]
    train = split_order[args.validation_size :]
    if {item.task_hash for item in train} & {item.task_hash for item in validation}:
        raise RuntimeError("task leakage detected between train and validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": emit_split("", train, args.output_dir),
        "validation": emit_split(".validation", validation, args.output_dir),
    }
    report = {
        "dataset": "terminal-cli",
        "upstream": UPSTREAM,
        "upstream_url": UPSTREAM_URL,
        "license": LICENSE,
        "parameters": {
            "seed": args.seed,
            "selected_unique_tasks": args.limit,
            "validation_size": args.validation_size,
            "minimum_content_chars": MIN_CONTENT_CHARS,
            "maximum_content_chars": args.max_content_chars,
            "absolute_allowed_maximum_content_chars": HARD_MAX_CONTENT_CHARS,
            "maximum_messages": 8,
            "required_action": "[Tool: Bash] or [Tool: PowerShell] in assistant content",
            "task_identity": "sha256(normalized system + NUL + normalized original user)",
            "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        },
        "audit": {
            **{key: value for key, value in stats.items() if key != "eligible_categories"},
            "eligible_categories": dict(sorted(stats["eligible_categories"].items())),
        },
        "leakage": {
            "train_validation_task_overlap": 0,
            "grouped_before_selection": True,
        },
        "outputs": outputs,
        "source_snapshot": source_snapshot,
        "source_files_unchanged": all(
            path.stat().st_size == source_snapshot[str(path)]["bytes"]
            and path.stat().st_mtime_ns == source_snapshot[str(path)]["mtime_ns"]
            for _, path in paths
        ),
    }
    report_path = args.output_dir / "terminal-cli.build-report.json"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "eligible_unique_tasks": len(tasks),
                "train_rows": outputs["train"]["rows"],
                "validation_rows": outputs["validation"]["rows"],
                "source_files_unchanged": report["source_files_unchanged"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
