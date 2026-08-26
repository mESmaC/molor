from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.business_finance_supervision import (
    materialize_business_finance_supervision,
)


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _finqa(identifier: str, filename: str, question: str) -> dict[str, object]:
    return {
        "filename": filename,
        "id": identifier,
        "qa": {
            "answer": "12%",
            "exe_ans": "12%",
            "explanation": "Subtract the earlier value and divide by it.",
            "model_input": [["table_1", "Revenue | 100 | 112"]],
            "program": "divide(subtract(112, 100), 100)",
            "question": question,
        },
    }


def _tatqa(document_id: str, question_id: str, question: str, derivation: str) -> dict[str, object]:
    return {
        "paragraphs": [{"order": 1, "text": "The company reported its annual results."}],
        "questions": [
            {
                "answer": "5",
                "derivation": derivation,
                "question": question,
                "scale": "million",
                "uid": question_id,
            }
        ],
        "table": {"table": [["Metric", "2025"], ["Cash", "5"]], "uid": document_id},
    }


def _fixture(
    tmp_path: Path,
    *,
    overlapping_group: bool = False,
    duplicate_content_across_splits: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "business-sources"
    finqa = root / "finqa" / "dataset"
    tatqa = root / "tat-qa" / "dataset_raw"
    _dump(finqa / "train.json", [_finqa("f-train", "report-a", "What was revenue growth?")])
    _dump(
        finqa / "dev.json",
        [
            _finqa(
                "f-dev",
                "report-a" if overlapping_group else "report-b",
                "What was the share price return?",
            )
        ],
    )
    test_question = (
        "What was revenue growth?" if duplicate_content_across_splits else "What was cash growth?"
    )
    _dump(finqa / "test.json", [_finqa("f-test", "report-c", test_question)])
    _dump(
        tatqa / "tatqa_dataset_train.json",
        [_tatqa("t-a", "t-train", "How much cash was reported?", "Read the cash row.")],
    )
    _dump(
        tatqa / "tatqa_dataset_dev.json",
        [_tatqa("t-b", "t-dev", "What was the stock price?", "")],
    )
    _dump(
        tatqa / "tatqa_dataset_test_gold.json",
        [_tatqa("t-c", "t-test", "What was the ending cash balance?", "Read the cash row.")],
    )

    sources = []
    paths = {
        "finqa": ("dataset/train.json", "dataset/dev.json", "dataset/test.json"),
        "tat-qa": (
            "dataset_raw/tatqa_dataset_train.json",
            "dataset_raw/tatqa_dataset_dev.json",
            "dataset_raw/tatqa_dataset_test_gold.json",
        ),
    }
    for source, relative_paths in paths.items():
        artifacts = []
        for relative in relative_paths:
            path = root / source / relative
            payload = path.read_bytes()
            artifacts.append(
                {
                    "bytes": len(payload),
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        sources.append(
            {
                "artifacts": artifacts,
                "git_commit": "a" * 40,
                "relative_root": source,
                "source_id": source,
            }
        )
    receipt = tmp_path / "receipt.json"
    _dump(
        receipt,
        {
            "license_or_rights_metadata_used_for_admission": False,
            "schema_version": "molor.business-source-snapshot.v1",
            "sources": sources,
        },
    )
    return root, receipt


def test_materializer_is_deterministic_messages_only_and_preserves_splits(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    outputs = (tmp_path / "output-a", tmp_path / "output-b")
    manifests = [
        materialize_business_finance_supervision(
            snapshot_root=root, receipt_path=receipt, output_dir=output
        )
        for output in outputs
    ]

    assert manifests[0] == manifests[1]
    assert manifests[0]["quality"]["input_records"] == 6
    assert manifests[0]["quality"]["canonical_records"] == 6
    assert manifests[0]["quality"]["direct_emissions"] == 6
    assert manifests[0]["quality"]["trace_emissions"] == 5
    assert manifests[0]["quality"]["zero_silent_drops"] is True
    assert manifests[0]["quality"]["split_group_overlap"] == 0
    assert (outputs[0] / "manifest.json").read_bytes() == (
        outputs[1] / "manifest.json"
    ).read_bytes()

    shard_paths = sorted(outputs[0].glob("*/*/*.jsonl"))
    training_shards = [path for path in shard_paths if ".provenance." not in path.name]
    assert len(training_shards) == 12
    for shard in training_shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"messages"}

    investing_validation = outputs[0] / "investing-markets" / "direct" / "validation.jsonl"
    assert len(investing_validation.read_text(encoding="utf-8").splitlines()) == 2
    business_test = outputs[0] / "business-finance" / "direct" / "test.jsonl"
    assert len(business_test.read_text(encoding="utf-8").splitlines()) == 2


def test_materializer_rejects_group_leakage_and_content_drift(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path, overlapping_group=True)
    with pytest.raises(ValueError, match="document group crosses splits"):
        materialize_business_finance_supervision(
            snapshot_root=root, receipt_path=receipt, output_dir=tmp_path / "out"
        )

    root, receipt = _fixture(tmp_path / "drift")
    with (root / "finqa" / "dataset" / "train.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(RuntimeError, match="content drift"):
        materialize_business_finance_supervision(
            snapshot_root=root, receipt_path=receipt, output_dir=tmp_path / "drift-out"
        )


def test_cross_split_duplicates_preserve_evaluation_copy(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path, duplicate_content_across_splits=True)
    output = tmp_path / "output"
    manifest = materialize_business_finance_supervision(
        snapshot_root=root, receipt_path=receipt, output_dir=output
    )

    assert manifest["contract"]["duplicate_precedence"] == "test_then_validation_then_train"
    assert manifest["quality"]["duplicate_records"] == 1
    assert manifest["quality"]["duplicates_by_source_and_reason"] == {
        "finqa": {"duplicate_content_cross_split": 1}
    }
    test_provenance = output / "business-finance" / "direct" / "test.provenance.jsonl"
    train_provenance = output / "business-finance" / "direct" / "train.provenance.jsonl"
    assert "finqa:f-test" in test_provenance.read_text(encoding="utf-8")
    assert "finqa:f-train" not in train_provenance.read_text(encoding="utf-8")


def test_materializer_rejects_blocked_paths(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    blocked = tmp_path / "security-output"
    with pytest.raises(ValueError, match="may not use cyber/security paths"):
        materialize_business_finance_supervision(
            snapshot_root=root, receipt_path=receipt, output_dir=blocked
        )
