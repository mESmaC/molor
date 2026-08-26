"""Materialize pinned FinQA and TAT-QA snapshots as granular business SFT shards.

The emitted training files contain only ``messages``.  Dataset identity, upstream IDs,
content hashes, and split metadata live in parallel provenance sidecars.  Upstream test
partitions are emitted for evaluation but are never relabeled as training data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence
import unicodedata


SCHEMA_VERSION = "molor.business-finance-supervision.v1"
RECEIPT_SCHEMA = "molor.business-source-snapshot.v1"
LEAVES = ("business-finance", "investing-markets")
MODES = ("direct", "trace")
SPLITS = ("train", "validation", "test")
SPLIT_PRECEDENCE = ("test", "validation", "train")
SYSTEM_PROMPT = (
    "You are a careful business and financial analysis assistant. Use only the supplied "
    "context, preserve units and scale, show calculations when they are requested, and do "
    "not invent unavailable facts."
)
_BLOCKED_WORDS = ("cyber", "security")
_SPACE_RE = re.compile(r"\s+")
_INVESTING_TERMS = (
    "bond",
    "common stock",
    "dividend",
    "earnings per share",
    "equity security",
    "fair value",
    "investment",
    "market price",
    "portfolio",
    "preferred stock",
    "return on equity",
    "securities",
    "share price",
    "shareholder return",
    "stock price",
    "treasury stock",
    "valuation",
    "yield",
)
_INPUTS: Mapping[str, Mapping[str, str]] = {
    "finqa": {
        "train": "dataset/train.json",
        "validation": "dataset/dev.json",
        "test": "dataset/test.json",
    },
    "tat-qa": {
        "train": "dataset_raw/tatqa_dataset_train.json",
        "validation": "dataset_raw/tatqa_dataset_dev.json",
        "test": "dataset_raw/tatqa_dataset_test_gold.json",
    },
}


def _blocked(parts: Iterable[str]) -> bool:
    return any(word in part.casefold() for part in parts for word in _BLOCKED_WORDS)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _payload(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _normalize(value: object) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(value))).strip()


def _strict_object(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


@dataclass(frozen=True, slots=True)
class InputArtifact:
    source: str
    split: str
    relative_root: str
    relative_path: str
    path: Path
    sha256: str
    bytes: int
    git_commit: str


@dataclass(frozen=True, slots=True)
class Candidate:
    source: str
    split: str
    group_id: str
    upstream_id: str
    leaf: str
    direct_messages: tuple[Mapping[str, str], ...]
    trace_messages: tuple[Mapping[str, str], ...] | None
    artifact_sha256: str
    content_sha256: str


def _artifact_map(receipt: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    sources = receipt.get("sources")
    if not isinstance(sources, list):
        raise ValueError("receipt sources must be a list")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("receipt source entry must be an object")
        source_id = source.get("source_id")
        root = source.get("relative_root")
        artifacts = source.get("artifacts")
        if not isinstance(source_id, str) or not isinstance(root, str):
            raise ValueError("receipt source identity is malformed")
        if not isinstance(artifacts, list):
            raise ValueError(f"receipt artifacts are malformed for {source_id}")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                raise ValueError(f"receipt artifact is malformed for {source_id}")
            key = (source_id, artifact["path"])
            if key in result:
                raise ValueError(f"duplicate receipt artifact {key}")
            result[key] = {
                **artifact,
                "relative_root": root,
                "git_commit": source.get("git_commit"),
            }
    return result


def _resolve_inputs(snapshot_root: Path, receipt_path: Path) -> tuple[InputArtifact, ...]:
    root = snapshot_root.expanduser().resolve()
    receipt_file = receipt_path.expanduser().resolve()
    if _blocked(root.parts) or _blocked(receipt_file.parts):
        raise ValueError("business supervision inputs may not use cyber/security paths")
    receipt = _strict_object(receipt_file)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported business acquisition receipt")
    if receipt.get("license_or_rights_metadata_used_for_admission") is not False:
        raise ValueError("license metadata must not be an admission gate")
    artifacts = _artifact_map(receipt)
    resolved: list[InputArtifact] = []
    for source, splits in _INPUTS.items():
        for split in SPLIT_PRECEDENCE:
            relative_path = splits[split]
            pure = PurePosixPath(relative_path)
            if pure.is_absolute() or ".." in pure.parts or _blocked(pure.parts):
                raise ValueError(f"unsafe input path for {source}: {relative_path}")
            receipt_artifact = artifacts.get((source, relative_path))
            if receipt_artifact is None:
                raise ValueError(f"{source}/{relative_path} is not pinned in the receipt")
            relative_root = str(receipt_artifact["relative_root"])
            if _blocked(PurePosixPath(relative_root).parts):
                raise ValueError(f"unsafe source root for {source}")
            path = (root / relative_root / Path(*pure.parts)).resolve()
            source_root = (root / relative_root).resolve()
            if path.parent != source_root and source_root not in path.parents:
                raise ValueError(f"input escaped source root for {source}")
            expected_sha = receipt_artifact.get("sha256")
            expected_bytes = receipt_artifact.get("bytes")
            if not isinstance(expected_sha, str) or not isinstance(expected_bytes, int):
                raise ValueError(f"receipt hash metadata is malformed for {source}/{relative_path}")
            if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha:
                raise RuntimeError(f"content drift for {source}/{relative_path}")
            commit = receipt_artifact.get("git_commit")
            if not isinstance(commit, str) or len(commit) != 40:
                raise ValueError(f"{source} must have a pinned Git commit")
            resolved.append(
                InputArtifact(
                    source=source,
                    split=split,
                    relative_root=relative_root,
                    relative_path=relative_path,
                    path=path,
                    sha256=expected_sha,
                    bytes=expected_bytes,
                    git_commit=commit,
                )
            )
    return tuple(resolved)


def _leaf_for(text: str) -> str:
    normalized = _normalize(text).casefold()
    score = sum(normalized.count(term) for term in _INVESTING_TERMS)
    return "investing-markets" if score >= 1 else "business-finance"


def _messages(user: str, assistant: str) -> tuple[Mapping[str, str], ...]:
    return (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    )


def _answer(value: object, *, scale: object = "") -> str:
    if isinstance(value, list):
        rendered = "; ".join(_normalize(item) for item in value)
    else:
        rendered = _normalize(value)
    scale_text = _normalize(scale)
    if scale_text and scale_text.casefold() not in rendered.casefold():
        rendered = f"{rendered} {scale_text}"
    return rendered


def _table(rows: object) -> str:
    if not isinstance(rows, list):
        raise ValueError("table must be a list")
    rendered: list[str] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("table row must be a list")
        rendered.append(" | ".join(_normalize(cell).replace("|", "\\|") for cell in row))
    return "\n".join(rendered)


def _candidate(
    *,
    source: str,
    split: str,
    group_id: str,
    upstream_id: str,
    leaf_text: str,
    user: str,
    answer: str,
    trace: str | None,
    artifact_sha256: str,
) -> Candidate:
    if not all(_normalize(value) for value in (group_id, upstream_id, user, answer)):
        raise ValueError("candidate identity, prompt, and answer must be non-empty")
    direct_target = f"<think>\n\n</think>\n\n{answer}"
    direct = _messages(user, direct_target)
    trace_messages = None
    if trace and _normalize(trace):
        trace_messages = _messages(user, f"<think>\n{_normalize(trace)}\n</think>\n\n{answer}")
    content_hash = _sha256_bytes(_payload(direct).encode("utf-8"))
    return Candidate(
        source=source,
        split=split,
        group_id=group_id,
        upstream_id=upstream_id,
        leaf=_leaf_for(leaf_text),
        direct_messages=direct,
        trace_messages=trace_messages,
        artifact_sha256=artifact_sha256,
        content_sha256=content_hash,
    )


def _finqa_candidates(artifact: InputArtifact) -> Iterator[Candidate]:
    records = json.loads(artifact.path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"FinQA input is not a list: {artifact.path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("qa"), dict):
            raise ValueError(f"malformed FinQA record {index} in {artifact.path}")
        qa = record["qa"]
        model_input = qa.get("model_input")
        if not isinstance(model_input, list):
            raise ValueError(f"malformed FinQA model_input {index} in {artifact.path}")
        context_lines: list[str] = []
        for item in model_input:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"malformed FinQA context row {index} in {artifact.path}")
            context_lines.append(f"[{_normalize(item[0])}] {_normalize(item[1])}")
        question = _normalize(qa.get("question", ""))
        answer = _answer(qa.get("answer") or qa.get("exe_ans", ""))
        program = _normalize(qa.get("program", ""))
        explanation = _normalize(qa.get("explanation", ""))
        trace_parts = []
        if explanation:
            trace_parts.append(explanation)
        if program:
            trace_parts.append(f"Executable calculation: {program}")
        user = (
            "Financial report context:\n" + "\n".join(context_lines) + f"\n\nQuestion: {question}"
        )
        upstream_id = _normalize(record.get("id") or f"{artifact.split}-{index}")
        filename = _normalize(record.get("filename") or upstream_id)
        yield _candidate(
            source="finqa",
            split=artifact.split,
            group_id=f"finqa:{filename}",
            upstream_id=f"finqa:{upstream_id}",
            leaf_text=f"{question} {answer} {program}",
            user=user,
            answer=answer,
            trace="\n".join(trace_parts),
            artifact_sha256=artifact.sha256,
        )


def _tatqa_candidates(artifact: InputArtifact) -> Iterator[Candidate]:
    documents = json.loads(artifact.path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise ValueError(f"TAT-QA input is not a list: {artifact.path}")
    for document_index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"malformed TAT-QA document {document_index}")
        table = document.get("table")
        paragraphs = document.get("paragraphs")
        questions = document.get("questions")
        if (
            not isinstance(table, dict)
            or not isinstance(paragraphs, list)
            or not isinstance(questions, list)
        ):
            raise ValueError(f"malformed TAT-QA document {document_index}")
        table_uid = _normalize(table.get("uid") or f"{artifact.split}-{document_index}")
        table_text = _table(table.get("table"))
        paragraph_lines: list[str] = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                raise ValueError(f"malformed TAT-QA paragraph in {table_uid}")
            paragraph_lines.append(
                f"[{_normalize(paragraph.get('order', ''))}] "
                f"{_normalize(paragraph.get('text', ''))}"
            )
        context = "Financial report table:\n" + table_text
        if paragraph_lines:
            context += "\n\nFinancial report text:\n" + "\n".join(paragraph_lines)
        for question_index, question_record in enumerate(questions):
            if not isinstance(question_record, dict):
                raise ValueError(f"malformed TAT-QA question in {table_uid}")
            question = _normalize(question_record.get("question", ""))
            answer = _answer(
                question_record.get("answer", ""), scale=question_record.get("scale", "")
            )
            derivation = _normalize(question_record.get("derivation", ""))
            user = f"{context}\n\nQuestion: {question}"
            uid = _normalize(question_record.get("uid") or f"{table_uid}-{question_index}")
            yield _candidate(
                source="tat-qa",
                split=artifact.split,
                group_id=f"tat-qa:{table_uid}",
                upstream_id=f"tat-qa:{uid}",
                leaf_text=f"{question} {answer} {table_text[:2000]}",
                user=user,
                answer=answer,
                trace=derivation or None,
                artifact_sha256=artifact.sha256,
            )


def _candidate_stream(artifacts: Sequence[InputArtifact]) -> Iterator[Candidate]:
    split_priority = {"test": 0, "validation": 1, "train": 2}
    for artifact in sorted(artifacts, key=lambda item: (item.source, split_priority[item.split])):
        if artifact.source == "finqa":
            yield from _finqa_candidates(artifact)
        elif artifact.source == "tat-qa":
            yield from _tatqa_candidates(artifact)
        else:  # pragma: no cover - protected by the fixed input registry
            raise AssertionError(artifact.source)


def _provenance(candidate: Candidate, *, mode: str) -> dict[str, object]:
    messages = candidate.direct_messages if mode == "direct" else candidate.trace_messages
    if messages is None:
        raise AssertionError("trace provenance requested without trace messages")
    return {
        "artifact_sha256": candidate.artifact_sha256,
        "content_sha256": _sha256_bytes(_payload(messages).encode("utf-8")),
        "group_id": candidate.group_id,
        "leaf": candidate.leaf,
        "mode": mode,
        "source": candidate.source,
        "split": candidate.split,
        "upstream_id": candidate.upstream_id,
    }


def materialize_business_finance_supervision(
    *,
    snapshot_root: Path,
    receipt_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build deterministic direct/trace business-finance and investing shards."""

    artifacts = _resolve_inputs(snapshot_root, receipt_path)
    destination = output_dir.expanduser().resolve()
    if _blocked(destination.parts):
        raise ValueError("business supervision output may not use cyber/security paths")

    input_counts: Counter[tuple[str, str]] = Counter()
    duplicate_counts: Counter[tuple[str, str]] = Counter()
    trace_disposition: Counter[tuple[str, str, str]] = Counter()
    canonical: list[Candidate] = []
    seen_content: dict[str, Candidate] = {}
    seen_upstream: dict[str, Candidate] = {}
    group_splits: dict[str, str] = {}
    for candidate in _candidate_stream(artifacts):
        input_counts[(candidate.source, candidate.split)] += 1
        previous_split = group_splits.setdefault(candidate.group_id, candidate.split)
        if previous_split != candidate.split:
            raise ValueError(
                f"upstream document group crosses splits: {candidate.group_id} "
                f"({previous_split}, {candidate.split})"
            )
        if candidate.upstream_id in seen_upstream:
            duplicate_counts[(candidate.source, "duplicate_upstream_id")] += 1
            continue
        seen_upstream[candidate.upstream_id] = candidate
        if candidate.content_sha256 in seen_content:
            earlier = seen_content[candidate.content_sha256]
            reason = (
                "duplicate_content_cross_split"
                if earlier.split != candidate.split
                else "duplicate_content_within_split"
            )
            duplicate_counts[(candidate.source, reason)] += 1
            continue
        seen_content[candidate.content_sha256] = candidate
        canonical.append(candidate)
        trace_disposition[
            (
                candidate.source,
                candidate.split,
                "emitted" if candidate.trace_messages else "no_trace",
            )
        ] += 1

    rows: dict[tuple[str, str, str], list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for candidate in canonical:
        direct_row = {"messages": list(candidate.direct_messages)}
        rows[(candidate.leaf, "direct", candidate.split)].append(
            (_payload(direct_row), _provenance(candidate, mode="direct"))
        )
        if candidate.trace_messages is not None:
            trace_row = {"messages": list(candidate.trace_messages)}
            rows[(candidate.leaf, "trace", candidate.split)].append(
                (_payload(trace_row), _provenance(candidate, mode="trace"))
            )

    sums: list[str] = []
    output_receipts: list[dict[str, object]] = []
    for leaf in LEAVES:
        for mode in MODES:
            for split in SPLITS:
                key = (leaf, mode, split)
                ordered = sorted(rows.get(key, []), key=lambda item: item[1]["upstream_id"])
                shard = destination / leaf / mode / f"{split}.jsonl"
                sidecar = destination / leaf / mode / f"{split}.provenance.jsonl"
                shard_payload = "".join(payload + "\n" for payload, _ in ordered).encode("utf-8")
                sidecar_payload = "".join(
                    _payload(provenance) + "\n" for _, provenance in ordered
                ).encode("utf-8")
                _write_atomic(shard, shard_payload)
                _write_atomic(sidecar, sidecar_payload)
                shard_sha = _sha256_bytes(shard_payload)
                sidecar_sha = _sha256_bytes(sidecar_payload)
                relative_shard = shard.relative_to(destination).as_posix()
                relative_sidecar = sidecar.relative_to(destination).as_posix()
                sums.extend(
                    (
                        f"{shard_sha}  {relative_shard}",
                        f"{sidecar_sha}  {relative_sidecar}",
                    )
                )
                output_receipts.append(
                    {
                        "leaf": leaf,
                        "mode": mode,
                        "provenance_path": relative_sidecar,
                        "provenance_sha256": sidecar_sha,
                        "records": len(ordered),
                        "shard_path": relative_shard,
                        "shard_sha256": shard_sha,
                        "split": split,
                    }
                )

    input_total = sum(input_counts.values())
    duplicates_total = sum(duplicate_counts.values())
    canonical_total = len(canonical)
    if input_total != canonical_total + duplicates_total:
        raise AssertionError("input disposition accounting does not balance")
    direct_total = sum(item["records"] for item in output_receipts if item["mode"] == "direct")
    trace_total = sum(item["records"] for item in output_receipts if item["mode"] == "trace")
    no_trace_total = sum(
        count
        for (source, split, status), count in trace_disposition.items()
        if status == "no_trace"
    )
    if direct_total != canonical_total or trace_total + no_trace_total != canonical_total:
        raise AssertionError("emission accounting does not balance")

    def nested(counter: Mapping[tuple[str, str], int]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = defaultdict(dict)
        for (outer, inner), count in sorted(counter.items()):
            result[outer][inner] = count
        return dict(result)

    trace_nested: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (source, split, status), count in sorted(trace_disposition.items()):
        trace_nested[source][split][status] = count

    receipt_file = receipt_path.expanduser().resolve()
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "assistant_template": "qwen3_think_tags",
            "license_or_rights_metadata_used_for_admission": False,
            "messages_only_training_shards": True,
            "provenance_is_sidecar_only": True,
            "duplicate_precedence": "test_then_validation_then_train",
            "split_policy": "preserve_upstream_train_dev_test",
            "test_is_evaluation_only": True,
            "truncation": False,
        },
        "inputs": {
            "acquisition_receipt": {
                "path": receipt_file.name,
                "sha256": _sha256_file(receipt_file),
            },
            "builder": {
                "path": Path(__file__).name,
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "artifacts": [
                {
                    "bytes": artifact.bytes,
                    "git_commit": artifact.git_commit,
                    "path": f"{artifact.relative_root}/{artifact.relative_path}",
                    "sha256": artifact.sha256,
                    "source": artifact.source,
                    "split": artifact.split,
                }
                for artifact in artifacts
            ],
        },
        "quality": {
            "canonical_records": canonical_total,
            "direct_emissions": direct_total,
            "duplicate_records": duplicates_total,
            "duplicates_by_source_and_reason": nested(duplicate_counts),
            "input_by_source_and_split": nested(input_counts),
            "input_records": input_total,
            "split_group_overlap": 0,
            "trace_disposition": {source: dict(splits) for source, splits in trace_nested.items()},
            "trace_emissions": trace_total,
            "zero_silent_drops": True,
        },
        "outputs": output_receipts,
    }
    manifest_payload = _canonical_json(manifest)
    manifest_path = destination / "manifest.json"
    _write_atomic(manifest_path, manifest_payload)
    sums.append(f"{_sha256_bytes(manifest_payload)}  manifest.json")
    _write_atomic(
        destination / "sha256sums.txt",
        ("\n".join(sorted(sums)) + "\n").encode("utf-8"),
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = materialize_business_finance_supervision(
        snapshot_root=args.snapshot_root,
        receipt_path=args.receipt,
        output_dir=args.output,
    )
    print(json.dumps(manifest["quality"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
