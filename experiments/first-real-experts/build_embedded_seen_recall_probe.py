"""Build a small, deterministic recall probe from embedded-knowledge-v1 training rows.

This is deliberately *not* a held-out benchmark.  Every probe answer was seen during
adapter SFT.  It is useful for checking whether an adapter loads, routes, and retains
representative corpus facts, but it cannot measure generalization.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from statistics import mean
from typing import Any, Iterable

from transformers import AutoTokenizer

from molor.training.data import AssistantOnlyCollator, load_jsonl


PROBE_NAME = "embedded-knowledge-v1.seen-recall-probe"
PROBE_LABEL = "training_seen_recall"
SEED = "embedded-knowledge-v1-seen-recall-probe-v1"
TASK_ORDER = (
    "api_contract",
    "macro_contract",
    "type_definition",
    "documentation",
    "linker_reference",
    "assembly_routine",
    "misra_note",
)
SPECIAL_TAGS = {
    "microchip": "microchip-mplab-xc",
    "c166": "c166-c167",
}
EMBEDDED_FOCUS_TAGS = {"vehicle-networks", "rtos-concurrency", "low-level-embedded"}
SELECTION_SLOTS = {
    "api_contract": ("microchip", "c166", "general", "general", "general"),
    "macro_contract": ("microchip", "c166", "general", "general", "general"),
    "type_definition": ("microchip", "c166", "general", "general", "general"),
    "documentation": ("microchip", "c166", "general", "general", "general"),
    "linker_reference": ("microchip", "general", "general", "general", "general"),
    "assembly_routine": ("microchip", "c166", "general", "general", "general"),
    "misra_note": ("microchip", "general", "general", "general", "general"),
}
WORD = re.compile(r"[a-z0-9_]+")
DOCUMENT_PATH_DENY = re.compile(r"(?:scrap|pastebin|tasks/completed)", re.IGNORECASE)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_example_sha256(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def stable_rank(*parts: str) -> int:
    return int(sha256_text("\0".join((SEED, *parts)))[:16], 16)


def normalized_words(value: str) -> frozenset[str]:
    return frozenset(WORD.findall(value.casefold()))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def stats(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = max(0, min(len(ordered) - 1, int(fraction * len(ordered) + 0.999999) - 1))
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": round(mean(ordered), 2),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def read_aligned_rows(dataset_path: Path, provenance_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as dataset_stream, provenance_path.open(
        "r", encoding="utf-8"
    ) as provenance_stream:
        pairs = zip_longest(dataset_stream, provenance_stream)
        for line_number, pair in enumerate(pairs, start=1):
            dataset_line, provenance_line = pair
            if dataset_line is None or provenance_line is None:
                raise ValueError("training dataset and provenance have different row counts")
            example = json.loads(dataset_line)
            provenance = json.loads(provenance_line)
            if set(example) != {"messages"} or not isinstance(example["messages"], list):
                raise ValueError(f"{dataset_path}:{line_number}: invalid trainer schema")
            messages = example["messages"]
            if canonical_example_sha256(messages) != provenance.get("example_sha256"):
                raise ValueError(f"{dataset_path}:{line_number}: example/provenance hash mismatch")
            if sha256_text(messages[-1]["content"]) != provenance.get("answer_sha256"):
                raise ValueError(f"{dataset_path}:{line_number}: answer/provenance hash mismatch")
            rows.append(
                {
                    "messages": messages,
                    "provenance": provenance,
                    "training_row_number": line_number,
                }
            )
    return rows


def eligible(row: dict[str, Any], task_type: str, profile: str) -> bool:
    provenance = row["provenance"]
    if provenance["task_type"] != task_type:
        return False
    if not (24 <= int(provenance["assistant_tokens"]) <= 160):
        return False
    if int(provenance["full_tokens"]) > 300:
        return False
    tags = set(provenance["domain_tags"])
    answer = row["messages"][-1]["content"].lstrip()
    if task_type == "misra_note" and answer.casefold().startswith("change log:"):
        return False
    if task_type == "documentation" and DOCUMENT_PATH_DENY.search(
        str(provenance["file_path"])
    ):
        return False
    if profile in SPECIAL_TAGS:
        return SPECIAL_TAGS[profile] in tags
    if profile == "general":
        if tags.intersection(SPECIAL_TAGS.values()):
            return False
        if task_type in {"api_contract", "macro_contract", "type_definition", "documentation"}:
            return bool(tags.intersection(EMBEDDED_FOCUS_TAGS))
        return True
    raise ValueError(f"unknown selection profile: {profile}")


def is_near_duplicate(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    provenance = candidate["provenance"]
    subject = str(provenance["subject"]).strip().casefold()
    question_words = normalized_words(candidate["messages"][-2]["content"])
    answer_words = normalized_words(candidate["messages"][-1]["content"])
    for existing in selected:
        existing_provenance = existing["provenance"]
        if str(existing_provenance["subject"]).strip().casefold() == subject:
            return True
        existing_question = normalized_words(existing["messages"][-2]["content"])
        existing_answer = normalized_words(existing["messages"][-1]["content"])
        if jaccard(question_words, existing_question) >= 0.86:
            return True
        if jaccard(answer_words, existing_answer) >= 0.82:
            return True
    return False


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_examples: set[str] = set()
    used_source_groups: set[str] = set()
    used_task_basenames: Counter[tuple[str, str]] = Counter()
    for task_type in TASK_ORDER:
        for slot_index, profile in enumerate(SELECTION_SLOTS[task_type], start=1):
            candidates = [row for row in rows if eligible(row, task_type, profile)]
            candidates.sort(
                key=lambda row: stable_rank(
                    task_type,
                    profile,
                    str(slot_index),
                    row["provenance"]["example_sha256"],
                )
            )
            chosen: dict[str, Any] | None = None
            for candidate in candidates:
                provenance = candidate["provenance"]
                if provenance["example_sha256"] in used_examples:
                    continue
                if provenance["source_group"] in used_source_groups:
                    continue
                basename = PurePosixPath(str(provenance["file_path"])).name.casefold()
                if used_task_basenames[(task_type, basename)] >= 2:
                    continue
                if is_near_duplicate(candidate, selected):
                    continue
                chosen = candidate
                break
            if chosen is None:
                raise RuntimeError(f"no eligible unique row for {task_type}:{profile}:{slot_index}")
            chosen = dict(chosen)
            chosen["selection_slot"] = f"{task_type}:{profile}:{slot_index}"
            selected.append(chosen)
            used_examples.add(chosen["provenance"]["example_sha256"])
            used_source_groups.add(chosen["provenance"]["source_group"])
            chosen_basename = PurePosixPath(str(chosen["provenance"]["file_path"])).name.casefold()
            used_task_basenames[(task_type, chosen_basename)] += 1
    return selected


def token_ids(tokenizer: Any, messages: list[dict[str, str]], *, generating: bool) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generating,
        enable_thinking=False,
        return_dict=False,
        truncation=False,
    )
    if not isinstance(value, list) or any(type(token_id) is not int for token_id in value):
        raise ValueError("tokenizer chat template did not return a flat token list")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    provenance_path = args.provenance.resolve()
    source_report_path = args.source_report.resolve()
    output_root = args.output_root.resolve()
    model_path = args.model.resolve()
    for path in (dataset_path, provenance_path, source_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    reported_train = source_report["outputs"]["train"]
    actual_dataset_hash = file_sha256(dataset_path)
    actual_provenance_hash = file_sha256(provenance_path)
    if actual_dataset_hash != reported_train["dataset_sha256"]:
        raise ValueError("training dataset hash no longer matches its frozen build report")
    if actual_provenance_hash != reported_train["provenance_sha256"]:
        raise ValueError("training provenance hash no longer matches its frozen build report")

    rows = read_aligned_rows(dataset_path, provenance_path)
    if len(rows) != int(reported_train["examples"]):
        raise ValueError("training row count no longer matches its frozen build report")
    selected = select_rows(rows)
    if len(selected) != 35:
        raise AssertionError(f"expected 35 probe rows, selected {len(selected)}")

    output_root.mkdir(parents=True, exist_ok=True)
    probe_path = output_root / f"{PROBE_NAME}.jsonl"
    probe_provenance_path = output_root / f"{PROBE_NAME}.provenance.jsonl"
    report_path = output_root / f"{PROBE_NAME}.report.json"
    write_jsonl(probe_path, ({"messages": row["messages"]} for row in selected))

    provenance_rows: list[dict[str, Any]] = []
    for probe_index, row in enumerate(selected, start=1):
        provenance_rows.append(
            {
                "probe_index": probe_index,
                "probe_label": PROBE_LABEL,
                "selection_slot": row["selection_slot"],
                "training_row_number": row["training_row_number"],
                **row["provenance"],
            }
        )
    write_jsonl(probe_provenance_path, provenance_rows)

    # Exercise the same strict schema loader and assistant-only collator used by SFT.
    loaded = load_jsonl(probe_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    collator = AssistantOnlyCollator(
        tokenizer,
        max_length=1536,
        chat_template_kwargs={"enable_thinking": False},
    )
    batch = collator(loaded)
    supervised_counts = (batch["labels"] != -100).sum(dim=1).tolist()

    full_tokens: list[int] = []
    prompt_tokens: list[int] = []
    assistant_tokens: list[int] = []
    for row, provenance, supervised_count in zip(selected, provenance_rows, supervised_counts):
        messages = row["messages"]
        prompt = token_ids(tokenizer, messages[:-1], generating=True)
        full = token_ids(tokenizer, messages, generating=False)
        if full[: len(prompt)] != prompt:
            raise ValueError("tokenizer produced inconsistent prompt and full-chat prefixes")
        measured_assistant = len(full) - len(prompt)
        measured = (len(full), len(prompt), measured_assistant)
        recorded = (
            int(provenance["full_tokens"]),
            int(provenance["prompt_tokens"]),
            int(provenance["assistant_tokens"]),
        )
        if measured != recorded:
            raise ValueError(f"token counts changed for {provenance['example_sha256']}")
        if int(supervised_count) != measured_assistant:
            raise ValueError("assistant-only collator target size differs from tokenizer measurement")
        full_tokens.append(len(full))
        prompt_tokens.append(len(prompt))
        assistant_tokens.append(measured_assistant)

    task_counts = Counter(row["task_type"] for row in provenance_rows)
    tag_counts = Counter(tag for row in provenance_rows for tag in row["domain_tags"])
    report: dict[str, Any] = {
        "probe": PROBE_NAME,
        "classification": PROBE_LABEL,
        "warning": (
            "Every row is copied from embedded-knowledge-v1 SFT training data. This is a "
            "training-seen recall/load-routing smoke probe, not a held-out or generalization benchmark."
        ),
        "appropriate_uses": [
            "confirm that the intended adapter is loaded and affects output",
            "track retention of representative training-corpus facts across adapter revisions",
            "inspect exact identifiers, declarations, and source-grounded snippets after training",
        ],
        "inappropriate_uses": [
            "claim generalization quality",
            "compare against held-out benchmark scores as if protocols were equivalent",
            "estimate factual accuracy outside the selected training rows",
        ],
        "source_training_artifacts": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": actual_dataset_hash,
            "provenance_path": str(provenance_path),
            "provenance_sha256": actual_provenance_hash,
            "build_report_path": str(source_report_path),
            "build_report_sha256": file_sha256(source_report_path),
            "rows": len(rows),
        },
        "selection": {
            "seed": SEED,
            "rows": len(selected),
            "task_quota": {task_type: 5 for task_type in TASK_ORDER},
            "slot_profiles": {key: list(value) for key, value in SELECTION_SLOTS.items()},
            "eligibility": {
                "assistant_tokens_min": 24,
                "assistant_tokens_max": 160,
                "full_chat_tokens_max": 300,
                "one_row_per_source_group": True,
                "max_rows_per_source_basename_per_task": 2,
                "general_api_macro_type_documentation_requires_one_of_tags": sorted(
                    EMBEDDED_FOCUS_TAGS
                ),
                "misra_changelog_only_rows_excluded": True,
            },
            "near_duplicate_guard": {
                "same_casefolded_subject_rejected": True,
                "question_word_set_jaccard_threshold": 0.86,
                "answer_word_set_jaccard_threshold": 0.82,
            },
        },
        "coverage": {
            "task_types": dict(sorted(task_counts.items())),
            "domain_tags": dict(sorted(tag_counts.items())),
            "unique_source_groups": len({row["source_group"] for row in provenance_rows}),
            "unique_subjects": len({str(row["subject"]).casefold() for row in provenance_rows}),
            "unique_answers": len({row["answer_sha256"] for row in provenance_rows}),
        },
        "integrity": {
            "source_hashes_match_frozen_build_report": True,
            "dataset_and_provenance_rows_aligned": True,
            "selected_examples_are_exact_training_row_copies": True,
            "selected_example_hashes_unique": len(
                {row["example_sha256"] for row in provenance_rows}
            )
            == len(provenance_rows),
            "selected_answer_hashes_unique": len(
                {row["answer_sha256"] for row in provenance_rows}
            )
            == len(provenance_rows),
        },
        "tokenizer_validation": {
            "model_path": str(model_path),
            "local_files_only": True,
            "enable_thinking": False,
            "trainer_schema_loader": "molor.training.data.load_jsonl",
            "assistant_only_collator": "molor.training.data.AssistantOnlyCollator",
            "loaded_rows": len(loaded),
            "batch_width_tokens": int(batch["input_ids"].shape[1]),
            "all_recorded_token_counts_reproduced": True,
            "full_chat": stats(full_tokens),
            "prompt": stats(prompt_tokens),
            "assistant_target": stats(assistant_tokens),
        },
        "outputs": {
            "probe_path": str(probe_path),
            "probe_sha256": file_sha256(probe_path),
            "provenance_path": str(probe_provenance_path),
            "provenance_sha256": file_sha256(probe_provenance_path),
            "report_path": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["outputs"]["report_sha256"] = file_sha256(report_path)
    return report


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=here / "data" / "embedded-knowledge-v1.jsonl",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=here / "data" / "embedded-knowledge-v1.train.provenance.jsonl",
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=here / "data" / "embedded-knowledge-v1.report.json",
    )
    parser.add_argument("--output-root", type=Path, default=here / "data")
    parser.add_argument("--model", type=Path, default=here.parent.parent / "models" / "Qwen3-1.7B")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, ensure_ascii=False))
