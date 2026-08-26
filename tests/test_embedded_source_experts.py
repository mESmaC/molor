from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from molor.experimental.embedded_source_experts import (
    _OCB_FIELDS,
    _PRIOR_REPORT_RELATIONSHIPS,
    _qa_templates,
    assign_group_splits,
    assign_stratified_group_splits,
    classify_code_leaf,
    load_plan,
    materialize_embedded_source_experts,
    verify_materialized_embedded_source,
)
from molor.experimental.length_tier_shards import tokenizer_identity


class FakeTokenizer:
    def apply_chat_template(self, conversation: list[dict[str, str]], **_: object) -> list[int]:
        length = max(8, sum(len(message["content"].split()) for message in conversation) + 4)
        return list(range(length))


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_compact(value) + "\n" for value in values), encoding="utf-8")


def _descriptor(path: Path, *, records: int | None = None) -> dict[str, object]:
    payload = path.read_bytes()
    result: dict[str, object] = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    if records is not None:
        result["records"] = records
    return result


def _ocb_row(index: int) -> dict[str, object]:
    return {
        "signal_id": 1000 + index,
        "signal_name": f"SIGNAL_{index}",
        "start_bit": index,
        "bit_length": 8,
        "scale": 0.5,
        "offset": -1.0,
        "unit": "V",
        "byte_order": "little_endian",
        "is_signed": 0,
        "training_eligible": "gold",
        "can_id": 256 + index,
        "message_name": f"MESSAGE_{index}",
        "dlc": 8,
        "make": f"Make{index}",
        "model": f"Model{index}",
        "year": "2026",
        "platform": None,
        "vehicle_tier": "gold",
        "data_trust": "trusted",
    }


def _fixture(tmp_path: Path) -> tuple[Path, FakeTokenizer, int]:
    prepared = tmp_path / "prepared"
    code_rows = [
        {
            "id": f"code-{index}",
            "source": "code-corpus",
            "repo": f"repo-{index}",
            "file_path": path,
            "text": f"grounded file text {index} with enough content",
        }
        for index, path in enumerate(
            (
                "src/main.c",
                "scripts/build.sh",
                "ui/main.qml",
                "docs/README.md",
                "can/decoder.c",
                "src/device.cpp",
                "web/status.tsx",
                "notes/reference.txt",
                "tools/run.bat",
                "obd/parser.py",
                "src/second.c",
                "scripts/test.sh",
                "frontend/panel.tsx",
                "docs/guide.md",
                "can/second_decoder.c",
            )
        )
    ]
    _write_jsonl(prepared / "code-corpus" / "train.jsonl", code_rows[:12])
    _write_jsonl(prepared / "code-corpus" / "validation.jsonl", code_rows[12:])

    table_rows = [_ocb_row(index) for index in range(12)]
    ocb_records: list[dict[str, object]] = []
    for index, row in enumerate(table_rows):
        family_index = index % 3
        _, messages = _qa_templates(row)[family_index]
        ocb_records.extend(
            (
                {
                    "id": f"table-{index}",
                    "source": "ocb-can-signal-db",
                    "kind": "table_row",
                    "row": row,
                },
                {
                    "id": f"qa-{index}",
                    "source": "ocb-can-signal-db",
                    "kind": "qa",
                    "messages": list(messages),
                },
            )
        )
    _write_jsonl(prepared / "ocb-can-signal-db" / "train.jsonl", ocb_records[:20])
    _write_jsonl(prepared / "ocb-can-signal-db" / "validation.jsonl", ocb_records[20:])

    prepared_manifest = prepared / "manifest.json"
    prepared_manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "code-corpus": {"train_rows": 12, "validation_rows": 3},
                    "ocb-can-signal-db": {
                        "train_rows": 20,
                        "validation_rows": 4,
                        "table_rows": 12,
                        "qa_rows": 12,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    database = tmp_path / "ocb_ocb.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE training_corpus ("
        "signal_id INTEGER, signal_name TEXT, start_bit INTEGER, bit_length INTEGER, "
        "scale REAL, offset REAL, unit TEXT, byte_order TEXT, is_signed INTEGER, "
        "training_eligible TEXT, can_id INTEGER, message_name TEXT, dlc INTEGER, make TEXT, "
        "model TEXT, year TEXT, platform TEXT, vehicle_tier TEXT, data_trust TEXT)"
    )
    connection.executemany(
        f"INSERT INTO training_corpus VALUES ({','.join('?' for _ in _OCB_FIELDS)})",
        [tuple(row[field] for field in _OCB_FIELDS) for row in table_rows],
    )
    connection.commit()
    connection.close()
    view_digest = hashlib.sha256()
    for row in table_rows:
        view_digest.update(_compact(row).encode("utf-8"))
        view_digest.update(b"\n")

    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    tokenizer_root.joinpath("tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    identity = tokenizer_identity(tokenizer_root)

    report_root = tmp_path / "reports"
    report_root.mkdir()
    reports = []
    for name, relationship in _PRIOR_REPORT_RELATIONSHIPS.items():
        report = report_root / name
        report.write_text(
            json.dumps(
                {
                    "dataset": name,
                    "outputs": {"train": {"rows": 1, "sha256": "a" * 64}},
                }
            ),
            encoding="utf-8",
        )
        reports.append(
            {
                "name": name,
                **_descriptor(report),
                "relationship": relationship,
            }
        )

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "molor.embedded-source-plan.v1",
                "prepared_manifest": _descriptor(prepared_manifest),
                "code_corpus": {
                    split: _descriptor(prepared / "code-corpus" / f"{split}.jsonl", records=count)
                    for split, count in (("train", 12), ("validation", 3))
                },
                "ocb_corpus": {
                    split: _descriptor(
                        prepared / "ocb-can-signal-db" / f"{split}.jsonl", records=count
                    )
                    for split, count in (("train", 20), ("validation", 4))
                },
                "ocb_database": {
                    "path": str(database),
                    "view": "training_corpus",
                    "rows": 12,
                    "canonical_sha256": view_digest.hexdigest(),
                },
                "tokenizer": {
                    "path": str(tokenizer_root),
                    "aggregate_sha256": identity["aggregate_sha256"],
                    "chat_template_sha256": identity["chat_template"]["sha256"],
                },
                "prior_reports": reports,
                "split_policy": {
                    "evaluation_fraction": 0.05,
                    "max_eval_group_share": 0.5,
                    "seed": "molor.embedded-source.v1",
                },
            }
        ),
        encoding="utf-8",
    )
    return plan_path, FakeTokenizer(), len(code_rows) + len(ocb_records)


def test_metadata_classifier_covers_all_five_leaves() -> None:
    assert classify_code_leaf("repo", "scripts/build.sh") == "terminal-cli"
    assert classify_code_leaf("repo", "ui/Main.qml") == "frontend-visual-computing"
    assert classify_code_leaf("repo", "docs/README.md") == "embedded-knowledge"
    assert classify_code_leaf("repo", "src/main.c") == "embedded-code-generation"
    assert classify_code_leaf("obd-can-mod", "src/parser.py") == "embedded-can-diagnostics"


def test_group_split_is_deterministic_atomic_and_nonempty() -> None:
    counts = {f"group-{index}": 1 for index in range(30)}
    first, receipt = assign_group_splits(counts, namespace="synthetic")
    second, _ = assign_group_splits(counts, namespace="synthetic")
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert receipt["target_reached"] == {"validation": True, "test": True}

    features = {
        group: {"a", "b"} if index % 3 == 0 else {"a"} for index, group in enumerate(counts)
    }
    stratified, stratified_receipt = assign_stratified_group_splits(
        counts,
        features=features,
        required_features={"a", "b"},
        namespace="synthetic-stratified",
    )
    assert set(stratified.values()) == {"train", "validation", "test"}
    assert stratified_receipt["features_by_split"] == {
        "test": ["a", "b"],
        "train": ["a", "b"],
        "validation": ["a", "b"],
    }


def test_real_contract_materialization_and_independent_verification(tmp_path: Path) -> None:
    plan_path, tokenizer, physical_rows = _fixture(tmp_path)
    plan = load_plan(plan_path)
    outputs = (tmp_path / "output-a", tmp_path / "output-b")
    manifests = [
        materialize_embedded_source_experts(plan=plan, output_dir=output, tokenizer=tokenizer)
        for output in outputs
    ]
    assert manifests[0].read_bytes() == manifests[1].read_bytes()
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["accounting"]["physical_input_records"] == physical_rows
    assert manifest["accounting"]["emitted_physical_records"] == physical_rows
    assert manifest["accounting"]["non_emitted_physical_records"] == 0
    assert manifest["accounting"]["zero_silent_drops"] is True
    assert manifest["provenance"]["ocb_training_view"]["verified"] is True
    assert manifest["prior_lineage"]["reported_payloads_opened"] is False
    assert all(
        manifest["accounting"]["reference_by_leaf_and_split"][leaf][split] > 0
        for leaf in manifest["taxonomy"]["leaves"]
        for split in ("validation", "test")
    )
    assert all(
        value == 0
        for audit in manifest["leakage"].values()
        if isinstance(audit, dict)
        and audit
        and all(isinstance(item, int) for item in audit.values())
        for value in audit.values()
    )
    verification = verify_materialized_embedded_source(output_dir=outputs[0], tokenizer=tokenizer)
    assert verification["fully_verified"] is True
    assert verification["sft_rows_retokenized"] == 12
    assert verification["physical_input_records"] == physical_rows

    for shard in outputs[0].glob("sft/**/*.jsonl"):
        if ".provenance." in shard.name:
            continue
        for line in shard.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"messages"}
    for shard in outputs[0].glob("reference/**/*.jsonl"):
        if ".provenance." in shard.name:
            continue
        for line in shard.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"text"}


def test_denied_output_path_is_rejected_before_creation(tmp_path: Path) -> None:
    plan_path, tokenizer, _ = _fixture(tmp_path / "fixture")
    with pytest.raises(ValueError, match="denied path component"):
        materialize_embedded_source_experts(
            plan=load_plan(plan_path),
            output_dir=tmp_path / "security-output",
            tokenizer=tokenizer,
        )
