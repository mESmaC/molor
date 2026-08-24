"""Build compact, provenance-tracked SFT shards from the prepared local corpora.

The source directory is treated as read-only.  Output chat JSONL files contain only the
``messages`` field required by the current MoLoR trainer; provenance is written beside them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


SYSTEMS = {
    "embedded-can": (
        "You are an embedded-systems and CAN-bus expert. Give precise bit positions, widths, "
        "endianness, signedness, scaling, units, and uncertainty. Never invent missing signal "
        "semantics."
    ),
    "financial-analysis": (
        "You are a financial-analysis expert. Use only the supplied evidence, show the concise "
        "calculation when one is needed, preserve units and scale, and clearly state the answer."
    ),
    "customer-gtm": (
        "You handle customer and revenue conversations for a real business. Be concise, helpful, "
        "honest about constraints, gather missing information, and move toward a concrete next step "
        "without pressure or fabricated promises."
    ),
    "software-architecture": (
        "You are a pragmatic software architect. Explain tradeoffs, failure modes, operational "
        "constraints, and when an alternative design is preferable."
    ),
}

LICENSES = {
    "ocb-can-signal-db": "local user-provided database export",
    "software-architecture": "local prepared dataset; upstream rights require retained manifest",
    "FinQA": "MIT repository; filing context remapped/retained with source provenance",
    "TAT-QA": "CC BY 4.0",
    "MultiDoGO": "CDLA-Permissive-2.0",
}


def stable_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


class MinHashSample:
    """Keep a deterministic uniform sample without loading the source corpus."""

    def __init__(self, limit: int, seed: str) -> None:
        self.limit = limit
        self.seed = seed
        self._heap: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
        self._keys: set[str] = set()
        self.seen = 0

    def add(
        self,
        key: str,
        messages: list[dict[str, str]],
        provenance: dict[str, Any],
    ) -> None:
        key = f"{provenance.get('source', 'unknown')}|{key}"
        if key in self._keys:
            return
        self.seen += 1
        score = stable_int(f"{self.seed}|{key}")
        item = (-score, key, {"messages": messages}, provenance)
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, item)
            self._keys.add(key)
        elif score < -self._heap[0][0]:
            removed = heapq.heapreplace(self._heap, item)
            self._keys.remove(removed[1])
            self._keys.add(key)

    def rows(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        ordered = sorted(self._heap, key=lambda item: (-item[0], item[1]))
        return [(item[2], item[3]) for item in ordered]


def valid_chat(messages: object) -> list[dict[str, str]] | None:
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        content = content.strip()
        if not content:
            return None
        cleaned.append({"role": role, "content": content})
    if cleaned[-1]["role"] != "assistant":
        return None
    if not any(message["role"] == "user" for message in cleaned[:-1]):
        return None
    return cleaned


def with_system(name: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEMS[name]}, *messages]


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(value, dict):
                yield value


def add_can_rows(
    path: Path,
    sample: MinHashSample,
    *,
    partition: str,
) -> dict[str, int]:
    stats = {"rows": 0, "qa": 0, "excluded_encode_100": 0, "invalid": 0}
    for row in jsonl(path):
        stats["rows"] += 1
        if row.get("kind") != "qa":
            continue
        messages = valid_chat(row.get("messages"))
        if messages is None:
            stats["invalid"] += 1
            continue
        prompt = messages[-2]["content"]
        # The source generator always chose physical=100 without checking the field range.
        # Those examples are frequently impossible for short signals, so exclude all of them.
        if "physical value of 100" in prompt:
            stats["excluded_encode_100"] += 1
            continue
        answer = messages[-1]["content"]
        if not 40 <= len(prompt) <= 900 or not 80 <= len(answer) <= 1800:
            stats["invalid"] += 1
            continue
        key = str(row.get("id") or hashlib.sha256(prompt.encode()).hexdigest())
        sample.add(
            key,
            with_system("embedded-can", messages),
            {
                "source": "ocb-can-signal-db",
                "source_id": key,
                "source_partition": partition,
                "license": LICENSES["ocb-can-signal-db"],
                "filter": "decode_or_lookup_only; fixed-100 encode template excluded",
            },
        )
        stats["qa"] += 1
    return stats


def add_architecture_rows(
    path: Path,
    sample: MinHashSample,
    *,
    partition: str,
) -> dict[str, int]:
    stats = {"rows": 0, "accepted": 0, "invalid": 0}
    blocked = ("as an ai language model", "i cannot provide", "i can't provide")
    for row in jsonl(path):
        stats["rows"] += 1
        messages = valid_chat(row.get("messages"))
        if messages is None:
            stats["invalid"] += 1
            continue
        prompt = messages[-2]["content"]
        answer = messages[-1]["content"]
        if (
            not 40 <= len(prompt) <= 1200
            or not 180 <= len(answer) <= 3500
            or any(marker in answer.lower() for marker in blocked)
        ):
            stats["invalid"] += 1
            continue
        key = str(row.get("id") or hashlib.sha256((prompt + answer).encode()).hexdigest())
        sample.add(
            key,
            with_system("software-architecture", messages[-2:]),
            {
                "source": "software-architecture",
                "source_id": key,
                "source_partition": partition,
                "license": LICENSES["software-architecture"],
            },
        )
        stats["accepted"] += 1
    return stats


def compact_evidence(values: Iterable[object], limit: int = 2300) -> str:
    lines: list[str] = []
    used = 0
    for value in values:
        text = " ".join(str(value).split())
        if not text or text in lines:
            continue
        remaining = limit - used
        if remaining <= 0:
            break
        text = text[:remaining]
        lines.append(text)
        used += len(text) + 3
    return "\n".join(f"- {line}" for line in lines)


def finqa_candidates(path: Path) -> Iterator[tuple[str, list[dict[str, str]], dict[str, Any]]]:
    documents = json.loads(path.read_text(encoding="utf-8"))
    for document in documents:
        qa = document.get("qa", {})
        question = qa.get("question")
        answer = qa.get("answer")
        if not isinstance(question, str) or not question.strip() or answer in (None, ""):
            continue
        gold = qa.get("gold_inds")
        evidence_values = list(gold.values()) if isinstance(gold, dict) else []
        if not evidence_values:
            evidence_values = [*document.get("pre_text", []), *document.get("post_text", [])]
        evidence = compact_evidence(evidence_values)
        if not evidence:
            continue
        prompt = f"Evidence:\n{evidence}\n\nQuestion: {question.strip()}"
        explanation = qa.get("explanation")
        program = qa.get("program")
        pieces: list[str] = []
        if isinstance(explanation, str) and explanation.strip():
            pieces.append(explanation.strip())
        if isinstance(program, str) and program.strip() and program.strip() not in pieces:
            pieces.append(f"Calculation: {program.strip()}")
        pieces.append(f"Answer: {str(answer).strip()}")
        response = "\n".join(pieces)
        if len(response) > 1800:
            continue
        source_id = str(document.get("id") or document.get("filename") or question)
        messages = with_system(
            "financial-analysis",
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        )
        yield source_id, messages, {
            "source": "FinQA",
            "source_id": source_id,
            "license": LICENSES["FinQA"],
            "upstream_split": "train",
        }


def table_text(table: object) -> str:
    if not isinstance(table, list):
        return ""
    return "\n".join(" | ".join(str(cell).strip() for cell in row) for row in table if isinstance(row, list))


def tatqa_candidates(path: Path) -> Iterator[tuple[str, list[dict[str, str]], dict[str, Any]]]:
    documents = json.loads(path.read_text(encoding="utf-8"))
    for document in documents:
        table = document.get("table", {}).get("table", [])
        paragraphs = document.get("paragraphs", [])
        paragraph_values = [
            paragraph.get("text", "") for paragraph in paragraphs if isinstance(paragraph, dict)
        ]
        base_evidence = compact_evidence([table_text(table), *paragraph_values])
        if not base_evidence:
            continue
        for qa in document.get("questions", []):
            if not isinstance(qa, dict):
                continue
            question = qa.get("question")
            answer = qa.get("answer")
            if not isinstance(question, str) or not question.strip() or answer in (None, ""):
                continue
            prompt = f"Evidence:\n{base_evidence}\n\nQuestion: {question.strip()}"
            derivation = qa.get("derivation")
            pieces: list[str] = []
            if isinstance(derivation, str) and derivation.strip():
                pieces.append(f"Calculation: {derivation.strip()}")
            if isinstance(answer, list):
                rendered_answer = "; ".join(str(item) for item in answer)
            else:
                rendered_answer = str(answer)
            scale = qa.get("scale")
            if isinstance(scale, str) and scale.strip():
                rendered_answer = f"{rendered_answer} {scale.strip()}"
            pieces.append(f"Answer: {rendered_answer.strip()}")
            response = "\n".join(pieces)
            source_id = str(qa.get("uid") or question)
            messages = with_system(
                "financial-analysis",
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            )
            yield source_id, messages, {
                "source": "TAT-QA",
                "source_id": source_id,
                "license": LICENSES["TAT-QA"],
                "upstream_split": "train",
            }


def dialogue_examples(path: Path) -> Iterator[tuple[str, list[dict[str, str]], dict[str, Any]]]:
    domain = path.stem
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        current_id: str | None = None
        turns: list[tuple[int, str, str]] = []

        def finish() -> tuple[str, list[dict[str, str]], dict[str, Any]] | None:
            if current_id is None:
                return None
            ordered = sorted(turns, key=lambda value: value[0])
            converted: list[dict[str, str]] = []
            seen_user = False
            for _, role, text in ordered:
                text = text.strip()
                if not text:
                    continue
                if role == "customer":
                    converted.append({"role": "user", "content": text})
                    seen_user = True
                elif role == "agent" and seen_user:
                    converted.append({"role": "assistant", "content": text})
            while converted and converted[-1]["role"] != "assistant":
                converted.pop()
            if len(converted) < 2 or converted[-1]["role"] != "assistant":
                return None
            # Retain only a compact recent dialogue and guarantee a leading user turn.
            converted = converted[-8:]
            while converted and converted[0]["role"] != "user":
                converted.pop(0)
            if len(converted) < 2:
                return None
            return current_id, with_system("customer-gtm", converted), {
                "source": "MultiDoGO",
                "source_id": current_id,
                "domain": domain,
                "license": LICENSES["MultiDoGO"],
            }

        for row in reader:
            conversation_id = str(row.get("conversationId", ""))
            if current_id is not None and conversation_id != current_id:
                completed = finish()
                if completed is not None:
                    yield completed
                turns = []
            current_id = conversation_id
            try:
                turn_number = int(row.get("turnNumber", len(turns)))
            except ValueError:
                turn_number = len(turns)
            turns.append(
                (turn_number, str(row.get("authorRole", "")).lower(), str(row.get("utterance", "")))
            )
        completed = finish()
        if completed is not None:
            yield completed


def write_sample(
    output_dir: Path,
    name: str,
    sample: MinHashSample,
) -> dict[str, Any]:
    output_path = output_dir / f"{name}.jsonl"
    provenance_path = output_dir / f"{name}.provenance.jsonl"
    digest = hashlib.sha256()
    with output_path.open("w", encoding="utf-8", newline="\n") as data_stream, provenance_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as provenance_stream:
        for line_number, (row, provenance) in enumerate(sample.rows(), start=1):
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            data_stream.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
            provenance_stream.write(
                json.dumps({"line": line_number, **provenance}, ensure_ascii=False) + "\n"
            )
    return {
        "path": str(output_path),
        "provenance_path": str(provenance_path),
        "eligible_before_sampling": sample.seen,
        "rows": len(sample.rows()),
        "sha256": digest.hexdigest(),
    }


def split_add(
    key: str,
    messages: list[dict[str, str]],
    provenance: dict[str, Any],
    train: MinHashSample,
    validation: MinHashSample,
) -> None:
    if stable_int(f"validation|{key}") % 100 < 8:
        validation.add(key, messages, provenance)
    else:
        train.add(key, messages, provenance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path.home() / "Documents" / "datasets" / "prepared",
    )
    parser.add_argument("--sources-root", type=Path, default=Path(__file__).parent / "sources")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--can-limit", type=int, default=4096)
    parser.add_argument("--finance-limit", type=int, default=2048)
    parser.add_argument("--gtm-limit", type=int, default=2048)
    parser.add_argument("--architecture-limit", type=int, default=2048)
    parser.add_argument("--validation-limit", type=int, default=512)
    args = parser.parse_args()
    limits = {
        "embedded-can": args.can_limit,
        "financial-analysis": args.finance_limit,
        "customer-gtm": args.gtm_limit,
        "software-architecture": args.architecture_limit,
    }
    if any(limit < 1 for limit in limits.values()) or args.validation_limit < 1:
        parser.error("sample limits must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samplers: dict[str, tuple[MinHashSample, MinHashSample]] = {
        name: (
            MinHashSample(limits[name], f"{name}-train-v1"),
            MinHashSample(args.validation_limit, f"{name}-validation-v1"),
        )
        for name in ("embedded-can", "financial-analysis", "customer-gtm", "software-architecture")
    }
    audit: dict[str, Any] = {"filters": {}, "outputs": {}}

    can_train, can_validation = samplers["embedded-can"]
    audit["filters"]["embedded-can-train"] = add_can_rows(
        args.prepared_root / "ocb-can-signal-db" / "train.jsonl", can_train, partition="train"
    )
    audit["filters"]["embedded-can-validation"] = add_can_rows(
        args.prepared_root / "ocb-can-signal-db" / "validation.jsonl",
        can_validation,
        partition="validation",
    )

    architecture_train, architecture_validation = samplers["software-architecture"]
    audit["filters"]["software-architecture-train"] = add_architecture_rows(
        args.prepared_root / "software-architecture" / "train.jsonl",
        architecture_train,
        partition="train",
    )
    audit["filters"]["software-architecture-validation"] = add_architecture_rows(
        args.prepared_root / "software-architecture" / "validation.jsonl",
        architecture_validation,
        partition="validation",
    )

    finance_train, finance_validation = samplers["financial-analysis"]
    for candidate in finqa_candidates(args.sources_root / "finqa" / "dataset" / "train.json"):
        split_add(*candidate, finance_train, finance_validation)
    for candidate in tatqa_candidates(
        args.sources_root / "tat-qa" / "dataset_raw" / "tatqa_dataset_train.json"
    ):
        split_add(*candidate, finance_train, finance_validation)

    gtm_train, gtm_validation = samplers["customer-gtm"]
    multidogo_dir = args.sources_root / "multidogo" / "data" / "unannotated"
    for path in sorted(multidogo_dir.glob("*.tsv")):
        for candidate in dialogue_examples(path):
            split_add(*candidate, gtm_train, gtm_validation)

    for name, (train, validation) in samplers.items():
        audit["outputs"][name] = {
            "train": write_sample(args.output_dir, name, train),
            "validation": write_sample(args.output_dir, f"{name}.validation", validation),
        }
    audit_path = args.output_dir / "build_report.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
