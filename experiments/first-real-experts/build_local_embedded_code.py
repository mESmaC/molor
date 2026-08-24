"""Build a local-only embedded source-continuation LoRA corpus.

The input is the prepared AEVIT/OCB file corpus.  This script never changes that
input.  It retains the upstream train/validation file split, filters to embedded
source/build/documentation files, creates one or two deterministic continuation
windows per file, and writes trainer-compatible messages-only JSONL plus separate
provenance sidecars.

This material includes locally supplied/internal and third-party source with mixed
licensing.  The generated artifacts are for local/internal experimentation only;
do not redistribute them without a separate license review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from statistics import mean
from typing import Any, Iterable

from transformers import AutoTokenizer


SYSTEM = (
    "You are an embedded-systems source completion model. Continue the supplied "
    "file prefix with only the exact next source text. Do not explain, summarize, "
    "add Markdown fences, or invent omitted earlier content."
)

CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".s",
    ".asm",
    ".inc",
    ".ld",
    ".lds",
    ".icf",
    ".sct",
    ".cmake",
    ".mk",
    ".dts",
    ".dtsi",
    ".cfg",
    ".conf",
    ".def",
}
SPECIAL_FILENAMES = {"makefile", "gnumakefile", "cmakelists.txt"}
DOMAIN_TERMS = re.compile(
    r"\b(?:embedded|firmware|microcontroller|mcu|can(?:fd)?|lin|sent|c166|c167|"
    r"interrupt|register|peripheral|bootloader|linker|rtos|freertos|gpio|uart|"
    r"spi|i2c|adc|pwm|watchdog|memory map|toolchain|cross[- ]compil|mplab|xc32|"
    r"arm|cortex|nxp|microchip|vehicle|control module|board support|bsp)\b",
    re.IGNORECASE,
)
PATH_DENY_PARTS = {
    "assets",
    "images",
    "image",
    "fonts",
    "node_modules",
    "dist",
    ".git",
    "coverage",
}
SENSITIVE_PATH = re.compile(
    r"(?:^|/)(?:secret|secrets|credential|credentials|private[-_]?key|passwords?)(?:/|\.|$)",
    re.IGNORECASE,
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
LITERAL_SECRET = re.compile(
    r"(?im)^\s*(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)\s*"
    r"[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
)
# The corpus is English source/documentation.  These lead characters are reliable
# signs of UTF-8 bytes decoded as a Windows code page here (box drawing, emoji,
# smart quotes, micro signs, and similar corruption).  Reject rather than teach it.
MOJIBAKE = re.compile(r"(?:Ã|Â|â|ðŸ|ï¿½|�)")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(*parts: str) -> int:
    return int(sha256_text("\0".join(parts))[:16], 16)


def normalized_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff")


def extension_label(file_path: str) -> str:
    path = PurePosixPath(file_path.replace("\\", "/"))
    name = path.name.lower()
    if name in SPECIAL_FILENAMES:
        return name
    return path.suffix.lower() or "[no-extension]"


def eligibility(file_path: str, text: str) -> tuple[bool, str]:
    normalized_path = file_path.replace("\\", "/")
    path = PurePosixPath(normalized_path)
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    suffix = path.suffix.lower()
    if parts & PATH_DENY_PARTS:
        return False, "denied_path"
    if SENSITIVE_PATH.search(normalized_path):
        return False, "sensitive_path"
    if name in SPECIAL_FILENAMES:
        pass
    elif suffix == ".md":
        if not (DOMAIN_TERMS.search(normalized_path) or DOMAIN_TERMS.search(text[:16000])):
            return False, "unrelated_markdown"
    elif suffix not in CODE_EXTENSIONS:
        return False, "extension"
    if "\x00" in text or any(ord(char) < 9 for char in text):
        return False, "binary_or_controls"
    if MOJIBAKE.search(text):
        return False, "mojibake"
    if PRIVATE_KEY.search(text) or LITERAL_SECRET.search(text):
        return False, "possible_secret"
    if len(text.strip()) < 300:
        return False, "too_short_chars"
    return True, "eligible"


def read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            required = {"id", "source", "repo", "file_path", "text"}
            if not required.issubset(value) or not all(
                isinstance(value[key], str) for key in required
            ):
                raise ValueError(f"{path}:{line_number}: invalid source row")
            rows.append({key: value[key] for key in required})
    return rows


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def nearest_boundary(lines: list[str], target: int, low: int, high: int) -> int:
    """Choose a stable line boundary near target, favoring source structure."""

    def quality(index: int) -> int:
        previous = lines[index - 1].strip() if index else ""
        current = lines[index].lstrip() if index < len(lines) else ""
        score = 0
        if not previous:
            score += 5
        if current.startswith(("#", "/*", "//", "*", ".", "SECTIONS", "MEMORY")):
            score += 2
        if re.match(r"(?:static\s+|inline\s+|const\s+|unsigned\s+|void\s+|int\s+|[A-Za-z_]\w*\s+)?[A-Za-z_]\w*\s*\([^;]*$", current):
            score += 3
        if current.startswith(("if ", "if(", "for ", "for(", "while ", "while(", "switch")):
            score += 1
        return score

    candidates = range(max(low, target - 32), min(high, target + 32) + 1)
    return min(candidates, key=lambda index: (-quality(index), abs(index - target), index))


def make_window(
    tokenizer: Any,
    *,
    repo: str,
    file_path: str,
    text: str,
    anchor_fraction: float,
    max_length: int,
) -> dict[str, Any] | None:
    lines = text.splitlines(keepends=True)
    if len(lines) < 8:
        return None
    line_sizes = [max(1, token_count(tokenizer, line)) for line in lines]
    cumulative = [0]
    for size in line_sizes:
        cumulative.append(cumulative[-1] + size)
    if cumulative[-1] < 150:
        return None

    desired = int(cumulative[-1] * anchor_fraction)
    raw_split = min(range(1, len(lines)), key=lambda i: abs(cumulative[i] - desired))
    low = next((i for i in range(1, len(lines)) if cumulative[i] >= 80), len(lines))
    high = next(
        (i for i in range(len(lines) - 1, 0, -1) if cumulative[-1] - cumulative[i] >= 48),
        0,
    )
    if low > high:
        return None
    split = nearest_boundary(lines, raw_split, low, high)

    prefix_start = split
    while prefix_start > 0 and cumulative[split] - cumulative[prefix_start - 1] <= 590:
        prefix_start -= 1
    continuation_end = split
    while (
        continuation_end < len(lines)
        and cumulative[continuation_end + 1] - cumulative[split] <= 275
    ):
        continuation_end += 1
    if continuation_end == split:
        continuation_end = min(len(lines), split + 1)

    prefix_lines = lines[prefix_start:split]
    answer_lines = lines[split:continuation_end]
    if not prefix_lines or not answer_lines:
        return None

    def assemble() -> tuple[list[dict[str, str]], int, int, int]:
        prefix = "".join(prefix_lines)
        answer = "".join(answer_lines)
        user = (
            "Continue this embedded source file from the provided prefix. Return only "
            "the next source text.\n"
            f"Repository: {repo}\nPath: {file_path}\n\n<SOURCE_PREFIX>\n{prefix}"
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_dict=False,
        )
        return messages, len(rendered), token_count(tokenizer, prefix), token_count(tokenizer, answer)

    messages, full_tokens, prefix_tokens, answer_tokens = assemble()
    while full_tokens > max_length and len(prefix_lines) > 2:
        prefix_lines.pop(0)
        messages, full_tokens, prefix_tokens, answer_tokens = assemble()
    while full_tokens > max_length and len(answer_lines) > 2:
        answer_lines.pop()
        messages, full_tokens, prefix_tokens, answer_tokens = assemble()
    if full_tokens > max_length or prefix_tokens < 72 or answer_tokens < 40:
        return None
    if not messages[-1]["content"].strip():
        return None
    return {
        "messages": messages,
        "full_tokens": full_tokens,
        "prefix_tokens": prefix_tokens,
        "answer_tokens": answer_tokens,
        "source_line_start": prefix_start + 1,
        "continuation_line_start": split + 1,
        "source_line_end": continuation_end,
    }


def candidates_for_rows(
    tokenizer: Any,
    rows: Iterable[dict[str, str]],
    *,
    split: str,
    max_length: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen_file_content: set[str] = set()
    seen_examples: set[str] = set()
    for row in sorted(rows, key=lambda value: (value["repo"], value["file_path"], value["id"])):
        text = normalized_text(row["text"])
        allowed, reason = eligibility(row["file_path"], text)
        if not allowed:
            rejected[reason] += 1
            continue
        content_hash = sha256_text(text)
        if content_hash in seen_file_content:
            rejected["duplicate_file_content"] += 1
            continue
        seen_file_content.add(content_hash)
        # Only the choice between one and two windows needs a size estimate.  Avoid
        # tokenizing enormous generated/vendor files wholesale (some exceed the
        # base model's context warning threshold); every emitted window is still
        # rendered and counted exactly below.
        anchors = (0.42, 0.72) if len(text) >= 3600 else (0.58,)
        made = 0
        for anchor_index, anchor in enumerate(anchors):
            window = make_window(
                tokenizer,
                repo=row["repo"],
                file_path=row["file_path"],
                text=text,
                anchor_fraction=anchor,
                max_length=max_length,
            )
            if window is None:
                continue
            task_hash = sha256_text(
                json.dumps(window["messages"], ensure_ascii=False, separators=(",", ":"))
            )
            if task_hash in seen_examples:
                continue
            seen_examples.add(task_hash)
            made += 1
            candidates.append(
                {
                    **window,
                    "source_split": split,
                    "source_id": row["id"],
                    "source": row["source"],
                    "repo": row["repo"],
                    "file_path": row["file_path"],
                    "extension": extension_label(row["file_path"]),
                    "source_content_sha256": content_hash,
                    "example_sha256": task_hash,
                    "anchor_index": anchor_index,
                }
            )
        if not made:
            rejected["no_usable_window"] += 1
    return candidates, rejected


def diverse_select(
    candidates: list[dict[str, Any]], *, limit: int, per_repo_cap: int
) -> list[dict[str, Any]]:
    """Round-robin repo+extension buckets, with deterministic order within buckets."""

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate["repo"], candidate["extension"])].append(candidate)
    for key, values in buckets.items():
        values.sort(
            key=lambda value: (
                stable_int(value["example_sha256"], key[0], key[1]),
                value["file_path"],
                value["anchor_index"],
            )
        )
    ordered_keys = sorted(buckets, key=lambda key: (stable_int(*key), key))
    selected: list[dict[str, Any]] = []
    repo_counts: Counter[str] = Counter()
    while len(selected) < limit:
        progress = False
        for key in ordered_keys:
            values = buckets[key]
            if not values or repo_counts[key[0]] >= per_repo_cap:
                continue
            selected.append(values.pop(0))
            repo_counts[key[0]] += 1
            progress = True
            if len(selected) == limit:
                break
        if not progress:
            break
    return sorted(selected, key=lambda value: (value["repo"], value["file_path"], value["anchor_index"]))


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [value["full_tokens"] for value in examples]
    return {
        "examples": len(examples),
        "unique_source_files": len({(v["repo"], v["file_path"]) for v in examples}),
        "repositories": dict(sorted(Counter(v["repo"] for v in examples).items())),
        "extensions": dict(sorted(Counter(v["extension"] for v in examples).items())),
        "full_chat_tokens": {
            "min": min(tokens, default=0),
            "mean": round(mean(tokens), 2) if tokens else 0,
            "p50": percentile(tokens, 0.50),
            "p95": percentile(tokens, 0.95),
            "max": max(tokens, default=0),
        },
        "prefix_tokens": {
            "min": min((v["prefix_tokens"] for v in examples), default=0),
            "mean": round(mean(v["prefix_tokens"] for v in examples), 2) if examples else 0,
            "max": max((v["prefix_tokens"] for v in examples), default=0),
        },
        "assistant_tokens": {
            "min": min((v["answer_tokens"] for v in examples), default=0),
            "mean": round(mean(v["answer_tokens"] for v in examples), 2) if examples else 0,
            "max": max((v["answer_tokens"] for v in examples), default=0),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            # ASCII JSON escapes keep valid Unicode source (for example a documentation
            # tree's U+2502 BOX DRAWINGS LIGHT VERTICAL) unambiguous when inspected by
            # legacy Windows PowerShell, whose Get-Content default can misdecode UTF-8.
            # json.loads restores the exact source code points for training.
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    model = args.model.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = {split: source_root / f"{split}.jsonl" for split in ("train", "validation")}
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    source_rows = {split: read_rows(path) for split, path in source_paths.items()}
    source_keys = {
        split: {(row["repo"], row["file_path"]) for row in rows}
        for split, rows in source_rows.items()
    }
    overlap = source_keys["train"] & source_keys["validation"]
    if overlap:
        raise ValueError(f"upstream file split overlap: {len(overlap)} paths")

    candidate_sets: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, Counter[str]] = {}
    for split in ("validation", "train"):
        candidate_sets[split], rejected[split] = candidates_for_rows(
            tokenizer,
            source_rows[split],
            split=split,
            max_length=args.max_length,
        )

    validation_content = {value["source_content_sha256"] for value in candidate_sets["validation"]}
    validation_tasks = {value["example_sha256"] for value in candidate_sets["validation"]}
    clean_train: list[dict[str, Any]] = []
    for value in candidate_sets["train"]:
        if value["source_content_sha256"] in validation_content:
            rejected["train"]["cross_split_duplicate_file_content"] += 1
        elif value["example_sha256"] in validation_tasks:
            rejected["train"]["cross_split_duplicate_example"] += 1
        else:
            clean_train.append(value)

    selected = {
        "train": diverse_select(clean_train, limit=args.train_examples, per_repo_cap=args.per_repo_cap),
        "validation": diverse_select(
            candidate_sets["validation"],
            limit=args.validation_examples,
            per_repo_cap=args.validation_examples,
        ),
    }

    output_paths: dict[str, dict[str, Path]] = {}
    for split, values in selected.items():
        dataset_path = output_root / (
            "local-embedded-code.jsonl" if split == "train" else "local-embedded-code.validation.jsonl"
        )
        provenance_path = output_root / f"local-embedded-code.{split}.provenance.jsonl"
        write_jsonl(dataset_path, ({"messages": value["messages"]} for value in values))
        write_jsonl(
            provenance_path,
            (
                {
                    key: value[key]
                    for key in (
                        "example_sha256",
                        "source_split",
                        "source_id",
                        "source",
                        "repo",
                        "file_path",
                        "extension",
                        "source_content_sha256",
                        "source_line_start",
                        "continuation_line_start",
                        "source_line_end",
                        "anchor_index",
                        "full_tokens",
                        "prefix_tokens",
                        "answer_tokens",
                    )
                }
                for value in values
            ),
        )
        output_paths[split] = {"dataset": dataset_path, "provenance": provenance_path}

    selected_paths = {
        split: {(value["repo"], value["file_path"]) for value in values}
        for split, values in selected.items()
    }
    selected_tasks = {
        split: {value["example_sha256"] for value in values}
        for split, values in selected.items()
    }
    report: dict[str, Any] = {
        "dataset": "local-embedded-code",
        "purpose": "assistant-only embedded source continuation/code completion LoRA SFT",
        "license_and_distribution": (
            "Local/internal experimentation only. Source repositories have mixed, locally supplied, "
            "and potentially proprietary licensing; do not redistribute generated artifacts without "
            "a separate per-repository license review."
        ),
        "source": {
            split: {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(source_rows[split]),
            }
            for split, path in source_paths.items()
        },
        "model_tokenizer": str(model),
        "selection": {
            "seed_mechanism": "SHA-256 deterministic ordering; no runtime RNG",
            "max_full_chat_tokens": args.max_length,
            "requested_train_examples": args.train_examples,
            "requested_validation_examples": args.validation_examples,
            "per_repository_train_cap": args.per_repo_cap,
            "maximum_windows_per_file": 2,
            "upstream_train_validation_file_overlap": len(overlap),
            "selected_train_validation_file_overlap": len(
                selected_paths["train"] & selected_paths["validation"]
            ),
            "selected_train_validation_example_overlap": len(
                selected_tasks["train"] & selected_tasks["validation"]
            ),
        },
        "filter_rejections_by_source_row": {
            split: dict(sorted(counts.items())) for split, counts in rejected.items()
        },
        "candidate_windows": {split: len(values) for split, values in candidate_sets.items()},
        "outputs": {},
        "limitations": [
            "This is continuation SFT, not instruction-following, debugging, or verified compilation data.",
            "Source correctness was not established by building every upstream repository.",
            "Automated secret filtering is conservative but cannot replace a manual disclosure review.",
            "Repeated SDK/vendor idioms can remain even after exact whole-file and exact-example deduplication.",
            "The upstream validation split contains few files, so domain/repository coverage is limited.",
            "A 1024-token window teaches local syntax and idioms, not whole-project architecture.",
        ],
        "recommended_recipe": {
            "expert_name": "local-embedded-code",
            "max_length": args.max_length,
            "epochs": 1,
            "rank": 8,
            "alpha": 16,
            "learning_rate": 0.00005,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "evaluation": "assistant-only validation NLL plus held-out exact/prefix-sensitive completion checks",
        },
    }
    for split in ("train", "validation"):
        report["outputs"][split] = {
            **summarize(selected[split]),
            "dataset_path": str(output_paths[split]["dataset"]),
            "dataset_sha256": file_sha256(output_paths[split]["dataset"]),
            "provenance_path": str(output_paths[split]["provenance"]),
            "provenance_sha256": file_sha256(output_paths[split]["provenance"]),
        }
    report_path = output_root / "local-embedded-code.report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "Documents" / "datasets" / "prepared" / "code-corpus",
    )
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--model", type=Path, default=workspace / "models" / "Qwen3-1.7B")
    parser.add_argument("--train-examples", type=int, default=2200)
    parser.add_argument("--validation-examples", type=int, default=128)
    parser.add_argument("--per-repo-cap", type=int, default=350)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if args.train_examples < 1 or not 1 <= args.validation_examples <= 128:
        parser.error("example limits must be positive and validation must be <= 128")
    if args.max_length < 256:
        parser.error("max-length must be at least 256")
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, ensure_ascii=False))
