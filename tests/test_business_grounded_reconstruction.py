from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.business_grounded_reconstruction import (
    OUTPUT_TIERS,
    ReconstructionPlan,
    SYSTEM_PROMPT,
    TASK_TYPE,
    materialize_grounded_reconstruction,
)
from molor.experimental.business_reference_corpus import LEAVES, SPLITS
from molor.experimental.length_tier_shards import tokenizer_identity
from molor.identity import canonical_json_bytes, fingerprint_json


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, str]], **kwargs: object
    ) -> list[int]:
        self.calls.append(dict(kwargs))
        rendered_size = sum(len(message["content"]) for message in conversation)
        return [1] * max(1, rendered_size)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _document(label: str, *, text: str | None = None) -> dict[str, str]:
    passage = text if text is not None else f"Verbatim source passage for {label}."
    return {
        "document_id": _sha(passage.encode()),
        "title": f"Recorded title {label}",
        "text": passage,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> tuple[str, int, int]:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha(payload), len(payload), len(rows)


def _build_reference_fixture(
    root: Path,
    *,
    duplicate_train_with_test: bool = False,
    long_label: tuple[str, str] | None = None,
) -> tuple[Path, dict[tuple[str, str], dict[str, str]]]:
    documents: dict[tuple[str, str], dict[str, str]] = {}
    for leaf in LEAVES:
        for split in SPLITS:
            label = f"{leaf}-{split}"
            text = "L" * 33_000 if long_label == (leaf, split) else None
            documents[(leaf, split)] = _document(label, text=text)
    if duplicate_train_with_test:
        documents[("business-core", "train")] = dict(documents[("business-core", "test")])

    source_rows: list[dict[str, object]] = []
    source_seen: set[str] = set()
    shards: list[dict[str, object]] = []
    for leaf in LEAVES:
        for split in SPLITS:
            document = documents[(leaf, split)]
            shard_path = root / "documents" / leaf / f"{split}.jsonl"
            digest, byte_count, records = _write_jsonl(shard_path, [document])
            shards.append(
                {
                    "bytes": byte_count,
                    "leaf_id": leaf,
                    "leaf_revision": "1",
                    "path": shard_path.relative_to(root).as_posix(),
                    "records": records,
                    "sha256": digest,
                    "split": split,
                }
            )
            document_id = document["document_id"]
            if duplicate_train_with_test and leaf == "business-core" and split == "train":
                continue
            if document_id in source_seen:
                continue
            source_seen.add(document_id)
            source_rows.append(
                {
                    "artifact_path": f"artifact/{leaf}.md",
                    "artifact_sha256": _sha(f"artifact-{leaf}".encode()),
                    "chunk_index": 0,
                    "content_sha256": document_id,
                    "family_sha256": _sha(f"family-{leaf}-{split}".encode()),
                    "internal_reference": "document",
                    "leaf_id": leaf,
                    "schema_version": "molor.business-reference-document-provenance.v1",
                    "source_id": f"source-{leaf}",
                    "split": split,
                    "status": "emitted",
                }
            )
    provenance_path = root / "provenance" / "documents.jsonl"
    provenance_sha, provenance_bytes, provenance_records = _write_jsonl(
        provenance_path, source_rows
    )
    manifest_content = {
        "schema_version": "molor.business-reference-corpus.v1",
        "shards": shards,
        "provenance_sidecars": {
            "documents": {
                "path": provenance_path.relative_to(root).as_posix(),
                "sha256": provenance_sha,
                "bytes": provenance_bytes,
                "records": provenance_records,
            }
        },
    }
    manifest = {**manifest_content, "content_sha256": fingerprint_json(manifest_content)}
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest_path, documents


def _plan(tmp_path: Path, manifest: Path) -> ReconstructionPlan:
    tokenizer_root = tmp_path / "qwen-tokenizer"
    tokenizer_root.mkdir(exist_ok=True)
    (tokenizer_root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "unit-qwen-chat-template"}),
        encoding="utf-8",
    )
    identity = tokenizer_identity(tokenizer_root)
    return ReconstructionPlan(
        reference_manifest=manifest,
        reference_manifest_sha256=_sha(manifest.read_bytes()),
        tokenizer_path=tokenizer_root,
        tokenizer_aggregate_sha256=identity["aggregate_sha256"],
        chat_template_sha256=identity["chat_template"]["sha256"],
    )


def _all_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_materialization_is_verbatim_labeled_tiered_and_deterministic(tmp_path) -> None:
    manifest_path, documents = _build_reference_fixture(tmp_path / "reference")
    plan = _plan(tmp_path, manifest_path)
    tokenizer = FakeQwenTokenizer()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = materialize_grounded_reconstruction(
        plan=plan,
        output_directory=first_root,
        tokenizer=tokenizer,
    )
    second = materialize_grounded_reconstruction(
        plan=plan,
        output_directory=second_root,
        tokenizer=FakeQwenTokenizer(),
    )

    assert first == second
    assert _all_files(first_root) == _all_files(second_root)
    assert first["task_contract"]["task_type"] == TASK_TYPE
    assert first["task_contract"]["contains_synthetic_question_answering"] is False
    assert first["task_contract"]["proves_answer_quality"] is False
    assert first["accounting"]["terminal_statuses"] == {
        "emitted": 18,
        "duplicate": 0,
        "quarantined": 0,
        "rejected": 0,
        "deferred": 0,
    }
    assert first["accounting"]["zero_silent_omissions"] is True
    assert first["accounting"]["fully_reconciled"] is True
    assert first["accounting"]["truncated_rows"] == 0
    assert first["accounting"]["deferred_rows"] == 0
    assert len(first["outputs"]) == len(LEAVES) * len(SPLITS) * len(OUTPUT_TIERS)

    observed: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for leaf in LEAVES:
        for split in SPLITS:
            for tier in OUTPUT_TIERS:
                messages_path = first_root / "messages" / leaf / split / f"{tier}.jsonl"
                provenance_path = first_root / "provenance" / leaf / split / f"{tier}.jsonl"
                message_lines = messages_path.read_text(encoding="utf-8").splitlines()
                provenance_lines = provenance_path.read_text(encoding="utf-8").splitlines()
                assert len(message_lines) == len(provenance_lines)
                for message_line, provenance_line in zip(
                    message_lines, provenance_lines, strict=True
                ):
                    provenance = json.loads(provenance_line)
                    observed[provenance["document_id"]] = (
                        json.loads(message_line),
                        provenance,
                    )
    assert len(observed) == 18
    for (leaf, split), source_document in documents.items():
        messages_row, provenance = observed[source_document["document_id"]]
        messages = messages_row["messages"]
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert "grounded reference reconstruction" in messages[1]["content"]
        assert source_document["title"] in messages[1]["content"]
        assert messages[2] == {"role": "assistant", "content": source_document["text"]}
        assert provenance["source_id"] == f"source-{leaf}"
        assert provenance["document_id"] == source_document["document_id"]
        assert provenance["assistant_target_sha256"] == source_document["document_id"]
        assert provenance["leaf_id"] == leaf
        assert provenance["split"] == split
        assert provenance["assistant_target_is_verbatim_source_passage"] is True
        assert provenance["is_synthetic_question_answering"] is False
    assert tokenizer.calls
    assert all(call["truncation"] is False for call in tokenizer.calls)
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)


def test_heldout_first_cross_split_duplicate_is_explicitly_quarantined(tmp_path) -> None:
    manifest_path, documents = _build_reference_fixture(
        tmp_path / "reference", duplicate_train_with_test=True
    )
    plan = _plan(tmp_path, manifest_path)
    output = tmp_path / "output"
    result = materialize_grounded_reconstruction(
        plan=plan,
        output_directory=output,
        tokenizer=FakeQwenTokenizer(),
    )

    assert result["accounting"]["terminal_statuses"] == {
        "emitted": 17,
        "duplicate": 0,
        "quarantined": 1,
        "rejected": 0,
        "deferred": 0,
    }
    assert result["accounting"]["reasons"]["opposite_split_document_id_match"] == 1
    ledger = [json.loads(line) for line in (output / "accounting.jsonl").read_text().splitlines()]
    collision = [row for row in ledger if row["reason"] == "opposite_split_document_id_match"]
    assert len(collision) == 1
    assert collision[0]["split"] == "train"
    assert collision[0]["duplicate_of"]["split"] == "test"
    assert collision[0]["document_id"] == documents[("business-core", "test")]["document_id"]
    assert result["accounting"]["zero_silent_omissions"] is True


def test_deferred_rows_fail_closed_without_publishing_output(tmp_path) -> None:
    manifest_path, _documents = _build_reference_fixture(
        tmp_path / "reference",
        long_label=("business-core", "train"),
    )
    plan = _plan(tmp_path, manifest_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="no deferred rows are permitted"):
        materialize_grounded_reconstruction(
            plan=plan,
            output_directory=output,
            tokenizer=FakeQwenTokenizer(),
        )
    assert not output.exists()


def test_manifest_and_tokenizer_pins_fail_before_output(tmp_path) -> None:
    manifest_path, _documents = _build_reference_fixture(tmp_path / "reference")
    valid = _plan(tmp_path, manifest_path)
    bad_manifest = ReconstructionPlan(
        reference_manifest=valid.reference_manifest,
        reference_manifest_sha256="0" * 64,
        tokenizer_path=valid.tokenizer_path,
        tokenizer_aggregate_sha256=valid.tokenizer_aggregate_sha256,
        chat_template_sha256=valid.chat_template_sha256,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_grounded_reconstruction(
            plan=bad_manifest,
            output_directory=tmp_path / "bad-manifest-output",
            tokenizer=FakeQwenTokenizer(),
        )
    bad_tokenizer = ReconstructionPlan(
        reference_manifest=valid.reference_manifest,
        reference_manifest_sha256=valid.reference_manifest_sha256,
        tokenizer_path=valid.tokenizer_path,
        tokenizer_aggregate_sha256="0" * 64,
        chat_template_sha256=valid.chat_template_sha256,
    )
    with pytest.raises(ValueError, match="tokenizer aggregate"):
        materialize_grounded_reconstruction(
            plan=bad_tokenizer,
            output_directory=tmp_path / "bad-tokenizer-output",
            tokenizer=FakeQwenTokenizer(),
        )
    assert not (tmp_path / "bad-manifest-output").exists()
    assert not (tmp_path / "bad-tokenizer-output").exists()
