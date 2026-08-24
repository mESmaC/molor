"""Build verifier-gated finance shards while retaining aligned provenance."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator

from molor.rewards import EvaluationSample, FinanceCalculationVerifier, VerificationStatus


CALCULATION_MARKER = re.compile(r"^\s*Calculation\s*:", re.IGNORECASE | re.MULTILINE)
QUESTION_SEPARATOR = "\n\nQuestion:"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def messages_from(row: object, path: Path, line_number: int) -> list[dict[str, str]]:
    if not isinstance(row, dict) or set(row) != {"messages"}:
        raise ValueError(f"{path}:{line_number}: expected a messages-only training row")
    messages = row["messages"]
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        raise ValueError(f"{path}:{line_number}: invalid messages")
    if messages[-1].get("role") != "assistant" or not isinstance(
        messages[-1].get("content"), str
    ):
        raise ValueError(f"{path}:{line_number}: final message must be an assistant string")
    return messages


def paired_rows(data_path: Path, provenance_path: Path) -> Iterator[tuple[int, str, dict[str, Any]]]:
    with data_path.open("r", encoding="utf-8") as data_stream, provenance_path.open(
        "r", encoding="utf-8"
    ) as provenance_stream:
        pairs = zip_longest(data_stream, provenance_stream)
        for line_number, pair in enumerate(pairs, start=1):
            data_line, provenance_line = pair
            if data_line is None or provenance_line is None:
                raise ValueError(f"row-count mismatch: {data_path} and {provenance_path}")
            if not data_line.strip() or not provenance_line.strip():
                raise ValueError(f"blank aligned row at input line {line_number}")
            try:
                provenance = json.loads(provenance_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{provenance_path}:{line_number}: invalid JSON") from exc
            if not isinstance(provenance, dict) or provenance.get("line") != line_number:
                raise ValueError(f"{provenance_path}:{line_number}: misaligned line metadata")
            yield line_number, data_line.rstrip("\r\n"), provenance


def prompt_and_group(messages: list[dict[str, str]], source: str) -> tuple[str, str, str]:
    prompt = next(
        (
            str(message["content"])
            for message in reversed(messages[:-1])
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )
    if not prompt:
        raise ValueError("training row has no user prompt")
    evidence = prompt.split(QUESTION_SEPARATOR, maxsplit=1)[0]
    prompt_fingerprint = sha256_text(prompt)
    group_fingerprint = sha256_text(f"{source}|{evidence}")
    row_fingerprint = sha256_text(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return prompt_fingerprint, group_fingerprint, row_fingerprint


def filter_split(
    *,
    name: str,
    data_path: Path,
    provenance_path: Path,
    output_path: Path,
    output_provenance_path: Path,
    verifier: FinanceCalculationVerifier,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    counts: Counter[str] = Counter()
    source_input: Counter[str] = Counter()
    source_kept: Counter[str] = Counter()
    exclusion_reasons: Counter[str] = Counter()
    excluded_examples: list[dict[str, Any]] = []
    identities = {
        "source_ids": set(),
        "groups": set(),
        "prompts": set(),
        "rows": set(),
    }
    duplicate_groups = 0
    output_line = 0
    output_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    provenance_tmp = output_provenance_path.with_suffix(output_provenance_path.suffix + ".tmp")

    with output_tmp.open("w", encoding="utf-8", newline="\n") as output_stream, (
        provenance_tmp.open("w", encoding="utf-8", newline="\n")
    ) as provenance_stream:
        for input_line, raw_row, provenance in paired_rows(data_path, provenance_path):
            try:
                row = json.loads(raw_row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{data_path}:{input_line}: invalid JSON") from exc
            messages = messages_from(row, data_path, input_line)
            completion = messages[-1]["content"]
            source = str(provenance.get("source") or "unknown")
            source_id = str(provenance.get("source_id") or "")
            source_input[source] += 1
            counts["input_rows"] += 1

            status = "non_calculation"
            keep = CALCULATION_MARKER.search(completion) is None
            verification_error: str | None = None
            if keep:
                counts["kept_non_calculation"] += 1
            else:
                counts["calculation_rows"] += 1
                prompt = next(
                    (
                        str(message["content"])
                        for message in reversed(messages[:-1])
                        if message.get("role") == "user"
                    ),
                    "",
                )
                result = verifier.verify(
                    EvaluationSample(
                        sample_id=f"{name}:{input_line}",
                        prompt=prompt,
                        completion=completion,
                    )
                )
                status = result.status.value
                verification_error = result.error
                keep = result.status is VerificationStatus.PASS
                if keep:
                    counts["kept_verified_calculation"] += 1
                else:
                    counts["excluded_calculation"] += 1
                    counts[f"excluded_{status}"] += 1
                    reason = verification_error or status
                    exclusion_reasons[reason] += 1
                    if len(excluded_examples) < 20:
                        excluded_examples.append(
                            {
                                "input_line": input_line,
                                "source": source,
                                "source_id": source_id,
                                "status": status,
                                "reason": reason,
                                "completion_preview": completion.replace("\n", " | ")[:300],
                            }
                        )
            if not keep:
                continue

            prompt_id, group_id, row_id = prompt_and_group(messages, source)
            if group_id in identities["groups"]:
                duplicate_groups += 1
            identities["source_ids"].add(f"{source}|{source_id}")
            identities["groups"].add(group_id)
            identities["prompts"].add(prompt_id)
            identities["rows"].add(row_id)
            output_line += 1
            counts["kept_rows"] += 1
            source_kept[source] += 1
            output_stream.write(raw_row + "\n")
            provenance_stream.write(
                json.dumps(
                    {
                        **provenance,
                        "line": output_line,
                        "input_split": name,
                        "input_line": input_line,
                        "group_fingerprint": group_id,
                        "prompt_fingerprint": prompt_id,
                        "row_fingerprint": row_id,
                        "filter": "non_calculation" if status == "non_calculation" else "verified",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    output_tmp.replace(output_path)
    provenance_tmp.replace(output_provenance_path)
    report = {
        "split": name,
        "input": {
            "path": str(data_path),
            "provenance_path": str(provenance_path),
            "sha256": file_sha256(data_path),
        },
        "output": {
            "path": str(output_path),
            "provenance_path": str(output_provenance_path),
            "rows": output_line,
            "sha256": file_sha256(output_path),
        },
        "counts": dict(sorted(counts.items())),
        "sources": {
            source: {"input": source_input[source], "kept": source_kept[source]}
            for source in sorted(source_input)
        },
        "unique_group_count": len(identities["groups"]),
        "additional_rows_in_existing_groups": duplicate_groups,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "excluded_examples": excluded_examples,
    }
    return report, identities


def load_retained(data_path: Path, provenance_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_row, provenance in paired_rows(data_path, provenance_path):
        row = json.loads(raw_row)
        messages = messages_from(row, data_path, line_number)
        source = str(provenance.get("source") or "unknown")
        prompt_id, group_id, row_id = prompt_and_group(messages, source)
        expected = {
            "prompt_fingerprint": prompt_id,
            "group_fingerprint": group_id,
            "row_fingerprint": row_id,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{provenance_path}:{line_number}: identity metadata mismatch")
        records.append({"raw": raw_row, "provenance": provenance, **expected})
    return records


def write_resplit(
    records: list[dict[str, Any]],
    *,
    data_path: Path,
    provenance_path: Path,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    data_tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    provenance_tmp = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    sources: Counter[str] = Counter()
    identities = {"source_ids": set(), "groups": set(), "prompts": set(), "rows": set()}
    with data_tmp.open("w", encoding="utf-8", newline="\n") as data_stream, (
        provenance_tmp.open("w", encoding="utf-8", newline="\n")
    ) as provenance_stream:
        for output_line, record in enumerate(records, start=1):
            provenance = record["provenance"]
            source = str(provenance.get("source") or "unknown")
            source_id = str(provenance.get("source_id") or "")
            sources[source] += 1
            identities["source_ids"].add(f"{source}|{source_id}")
            identities["groups"].add(record["group_fingerprint"])
            identities["prompts"].add(record["prompt_fingerprint"])
            identities["rows"].add(record["row_fingerprint"])
            data_stream.write(record["raw"] + "\n")
            provenance_stream.write(
                json.dumps(
                    {
                        **provenance,
                        "line": output_line,
                        "resplit": "finance-evidence-group-hash-v1",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    data_tmp.replace(data_path)
    provenance_tmp.replace(provenance_path)
    return (
        {
            "path": str(data_path),
            "provenance_path": str(provenance_path),
            "rows": len(records),
            "groups": len(identities["groups"]),
            "sources": dict(sorted(sources.items())),
            "sha256": file_sha256(data_path),
            "provenance_sha256": file_sha256(provenance_path),
        },
        identities,
    )


def resplit_retained(
    *,
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    train_data = data_dir / "financial-analysis-verified.jsonl"
    train_provenance = data_dir / "financial-analysis-verified.provenance.jsonl"
    validation_data = data_dir / "financial-analysis-verified.validation.jsonl"
    validation_provenance = data_dir / "financial-analysis-verified.validation.provenance.jsonl"
    combined = [
        *load_retained(train_data, train_provenance),
        *load_retained(validation_data, validation_provenance),
    ]
    assigned: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for record in combined:
        target = (
            "validation"
            if stable_int(f"finance-validation-v1|{record['group_fingerprint']}") % 100 < 10
            else "train"
        )
        assigned[target].append(record)

    prospective: dict[str, dict[str, set[str]]] = {}
    for split, records in assigned.items():
        sets = {"source_ids": set(), "groups": set(), "prompts": set(), "rows": set()}
        for record in records:
            provenance = record["provenance"]
            sets["source_ids"].add(
                f"{provenance.get('source', 'unknown')}|{provenance.get('source_id', '')}"
            )
            sets["groups"].add(record["group_fingerprint"])
            sets["prompts"].add(record["prompt_fingerprint"])
            sets["rows"].add(record["row_fingerprint"])
        prospective[split] = sets
    overlaps = {
        kind: prospective["train"][kind] & prospective["validation"][kind]
        for kind in prospective["train"]
    }
    if any(overlaps.values()):
        summary = {kind: len(values) for kind, values in overlaps.items() if values}
        raise ValueError(f"group assignment left cross-split identity leakage: {summary}")

    outputs: dict[str, Any] = {}
    final_identities: dict[str, dict[str, set[str]]] = {}
    outputs["train"], final_identities["train"] = write_resplit(
        assigned["train"], data_path=train_data, provenance_path=train_provenance
    )
    outputs["validation"], final_identities["validation"] = write_resplit(
        assigned["validation"],
        data_path=validation_data,
        provenance_path=validation_provenance,
    )
    total_rows = len(combined)
    summary = {
        "assignment": "sha256('finance-validation-v1|' + group_fingerprint) mod 100 < 10",
        "target_validation_fraction": 0.1,
        "actual_validation_fraction": len(assigned["validation"]) / total_rows,
        "total_rows": total_rows,
        "total_groups": len({record["group_fingerprint"] for record in combined}),
        "outputs": outputs,
    }
    return summary, final_identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    verifier = FinanceCalculationVerifier(require_reference=False)

    splits: dict[str, tuple[str, str]] = {
        "train": ("financial-analysis.jsonl", "financial-analysis.provenance.jsonl"),
        "validation": (
            "financial-analysis.validation.jsonl",
            "financial-analysis.validation.provenance.jsonl",
        ),
    }
    reports: dict[str, Any] = {}
    for split, (data_name, provenance_name) in splits.items():
        suffix = "" if split == "train" else ".validation"
        reports[split], _ = filter_split(
            name=split,
            data_path=data_dir / data_name,
            provenance_path=data_dir / provenance_name,
            output_path=data_dir / f"financial-analysis-verified{suffix}.jsonl",
            output_provenance_path=(
                data_dir / f"financial-analysis-verified{suffix}.provenance.jsonl"
            ),
            verifier=verifier,
        )
        candidate = reports[split].pop("output")
        reports[split]["retained_before_group_resplit"] = {
            "rows": candidate["rows"],
            "sha256": candidate["sha256"],
        }

    resplit, identities = resplit_retained(data_dir=data_dir)

    leakage = {
        kind: {
            "count": len(identities["train"][kind] & identities["validation"][kind]),
            "examples": sorted(identities["train"][kind] & identities["validation"][kind])[:20],
        }
        for kind in ("source_ids", "groups", "prompts", "rows")
    }
    report = {
        "schema_version": "molor.finance-filter.v2",
        "policy": {
            "non_calculation": "keep",
            "calculation": "keep only FinanceCalculationVerifier PASS",
            "table_without_metadata": "exclude",
            "split_membership": "assign whole evidence groups to deterministic 90/10 split",
            "original_membership": "retain as input_split and input_line in provenance",
        },
        "verifier": verifier.manifest(),
        "input_filters": reports,
        "resplit": resplit,
        "cross_split_leakage": leakage,
    }
    report_path = data_dir / "financial-analysis-verified.report.json"
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_tmp.replace(report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
