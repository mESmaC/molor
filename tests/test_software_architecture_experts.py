from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pytest

from molor.experimental import software_architecture_experts as subject
from molor.experimental.length_tier_shards import sha256_file, tokenizer_identity


class FakeTokenizer:
    def apply_chat_template(self, conversation: object, **kwargs: Any) -> object:
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is False
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False

        def render(messages: list[dict[str, str]]) -> list[int]:
            text = " ".join(message["content"] for message in messages)
            marker = re.search(r"\[TOKENS=(\d+)\]", text)
            count = int(marker.group(1)) if marker else max(1, len(text.split()) + 8)
            return list(range(count))

        assert isinstance(conversation, list)
        if conversation and isinstance(conversation[0], list):
            return [render(messages) for messages in conversation]
        return render(conversation)


def _row(identifier: str, prompt: str, answer: str = "A useful architectural answer.") -> dict:
    return {
        "id": identifier,
        "source": subject.EXPECTED_SOURCE,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _write_jsonl(path: Path, rows: list[object | bytes]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            if isinstance(row, bytes):
                stream.write(row.rstrip(b"\r\n") + b"\n")
            else:
                stream.write(
                    (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                )


def _tokenizer_dir(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "toy-template"}), encoding="utf-8"
    )
    return root, tokenizer_identity(root)


def _prompt_for_split(split: str) -> str:
    for index in range(10_000):
        prompt = f"Clean held-out architecture family {index}"
        if subject.heldout_split_for_family(subject.prompt_family_id(prompt)) == split:
            return prompt
    raise AssertionError("failed to locate a deterministic held-out family")


def test_granular_classification_and_fallback() -> None:
    cases = {
        "Show the Bash command-line workflow in a terminal": subject.LEAF_TERMINAL_CLI,
        "Build a responsive React user interface with CSS": (
            subject.LEAF_FRONTEND_VISUAL_COMPUTING
        ),
        "Design a machine learning feature engineering pipeline": subject.LEAF_ML_DATA_REASONING,
        "Debug the exception and perform root cause diagnosis": (
            subject.LEAF_AGENTIC_CODE_DEBUGGING
        ),
        "Explain consensus and replication in a distributed system": (
            subject.LEAF_SYSTEMS_DISTRIBUTED_ENGINEERING
        ),
        "Implement a Python function and algorithm": subject.LEAF_GENERAL_CODE_GENERATION,
        "Compare MVC and clean architecture design patterns": (
            subject.LEAF_SOFTWARE_ARCHITECTURE_DESIGN
        ),
        "Discuss a completely unspecified engineering consideration": (
            subject.LEAF_SOFTWARE_ARCHITECTURE_DESIGN
        ),
    }
    for prompt, expected in cases.items():
        leaf, rules, _ = subject.classify_prompt(prompt)
        assert leaf == expected
        assert rules


def test_prompt_normalization_family_and_tier_boundaries() -> None:
    assert subject.prompt_group_id(" A\r\nB ") == subject.prompt_group_id("a b")
    common = "x" * 256
    assert subject.prompt_family_id(common + " one") == subject.prompt_family_id(common + " two")
    assert subject.tier_for_token_count(1_024) == "le-1024"
    assert subject.tier_for_token_count(1_025) == "1025-4096"
    assert subject.tier_for_token_count(4_097) == "4097-8192"
    assert subject.tier_for_token_count(8_193) == "8193-32768"
    assert subject.tier_for_token_count(32_769) == "deferred-gt32768"


def test_blocked_path_is_refused_before_materialization(tmp_path: Path) -> None:
    # Construct the forbidden name only as a Path value.  It is intentionally never created,
    # opened, traversed, or hashed; refusal must happen before filesystem I/O.
    blocked = tmp_path / "cyber-input" / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(heldout, [_row("b", "Another architecture")])
    tokenizer_path, identity = _tokenizer_dir(tmp_path)

    with pytest.raises(ValueError, match="refusing to inspect"):
        subject.materialize_software_architecture_experts(
            train_pin=subject.InputPin(blocked, "0" * 64, 1, "prepared-train"),
            heldout_pin=subject.InputPin(heldout, sha256_file(heldout), 1, "prepared-heldout"),
            output_dir=tmp_path / "output",
            tokenizer=FakeTokenizer(),
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=identity["aggregate_sha256"],
            expected_chat_template_sha256=identity["chat_template"]["sha256"],
        )


def test_full_terminal_accounting_leakage_and_determinism(tmp_path: Path) -> None:
    shared_family = "z" * 256
    train_rows = [
        _row("arch", "Compare MVC and clean architecture patterns"),
        _row("systems", "Explain consensus in a distributed system"),
        _row("code", "Implement a Python function"),
        _row("debug", "Debug an exception with root cause analysis"),
        _row("frontend", "Create a responsive React UI with CSS"),
        _row("ml", "Build a machine learning data pipeline"),
        _row("cli", "Use a Bash command-line terminal workflow"),
        _row("shared", shared_family + " train"),
        _row(
            "trace",
            "Explain an architecture trace",
            "<think>explicit reasoning</think>final answer",
        ),
        {
            "id": "tool",
            "source": subject.EXPECTED_SOURCE,
            "messages": [
                {"role": "user", "content": "Use a tool for architecture"},
                {"role": "assistant", "content": "Calling tool"},
                {"role": "tool", "content": "tool result"},
                {"role": "assistant", "content": "Final tool answer"},
            ],
        },
    ]
    heldout_rows: list[object | bytes] = [
        _row("exact-overlap", "Compare MVC and clean architecture patterns"),
        _row("family-overlap", shared_family + " heldout"),
        _row("validation", _prompt_for_split(subject.SPLIT_VALIDATION), "[TOKENS=1024]"),
        _row("test", _prompt_for_split(subject.SPLIT_TEST), "[TOKENS=1025]"),
        {
            "id": "guarded",
            "source": "security-corpus",
            "messages": "metadata gate must reject before message validation",
        },
        b"{not-json",
    ]
    train_path = tmp_path / "train.jsonl"
    heldout_path = tmp_path / "heldout.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(heldout_path, heldout_rows)
    tokenizer_path, identity = _tokenizer_dir(tmp_path)
    lineage = tmp_path / "prior.report.json"
    lineage.write_text('{"rows":13000}\n', encoding="utf-8")

    kwargs = {
        "train_pin": subject.InputPin(
            train_path, sha256_file(train_path), len(train_rows), "prepared-train"
        ),
        "heldout_pin": subject.InputPin(
            heldout_path, sha256_file(heldout_path), len(heldout_rows), "prepared-heldout"
        ),
        "tokenizer": FakeTokenizer(),
        "tokenizer_path": tokenizer_path,
        "expected_tokenizer_sha256": identity["aggregate_sha256"],
        "expected_chat_template_sha256": identity["chat_template"]["sha256"],
        "lineage_aliases": [lineage],
        "batch_size": 3,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest_path = subject.materialize_software_architecture_experts(
        output_dir=first, **kwargs
    )
    second_manifest_path = subject.materialize_software_architecture_experts(
        output_dir=second, **kwargs
    )

    assert first_manifest_path.read_bytes() == second_manifest_path.read_bytes()
    assert (first / "sha256sums.txt").read_bytes() == (second / "sha256sums.txt").read_bytes()
    manifest = json.loads(first_manifest_path.read_bytes())
    assert manifest["accounting"]["input_records"] == 16
    assert manifest["accounting"]["terminal_status"] == {"emitted": 12, "quarantined": 4}
    assert manifest["accounting"]["emitted_by_split"] == {
        "test": 1,
        "train": 10,
        "validation": 1,
    }
    assert manifest["accounting"]["emitted_by_mode"] == {
        "direct": 10,
        "tool": 1,
        "trace": 1,
    }
    assert manifest["accounting"]["quarantine_reasons"] == {
        "guarded_source_metadata": 1,
        "heldout_train_exact_prompt_overlap": 1,
        "heldout_train_prompt_family_overlap": 1,
        "invalid_json": 1,
    }
    assert manifest["leakage"]["zero_prompt_overlap"] is True
    assert manifest["leakage"]["zero_prompt_family_overlap"] is True
    assert manifest["accounting"]["fully_reconciled"] is True
    assert manifest["accounting"]["zero_silent_omissions"] is True
    assert manifest["provenance"]["lineage_aliases"][0]["ingested"] is False

    populated = [receipt for receipt in manifest["outputs"] if receipt["records"]]
    assert sum(receipt["records"] for receipt in populated) == 12
    for receipt in populated:
        records = [json.loads(line) for line in (first / receipt["path"]).read_text().splitlines()]
        assert records
        assert all(set(record) == {"messages"} for record in records)
        sidecars = [
            json.loads(line)
            for line in (first / receipt["provenance_path"]).read_text().splitlines()
        ]
        assert len(sidecars) == len(records)
        assert all("messages" not in row for row in sidecars)

    quarantine = [
        json.loads(line) for line in (first / "quarantine.jsonl").read_text().splitlines()
    ]
    assert len(quarantine) == 4
    assert all("messages" not in row for row in quarantine)
    assert len((first / "sha256sums.txt").read_text().splitlines()) == 632


def test_hash_and_count_pins_fail_closed(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(train, [_row("a", "Architecture")])
    _write_jsonl(heldout, [_row("b", "Other architecture")])
    tokenizer_path, identity = _tokenizer_dir(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        subject.materialize_software_architecture_experts(
            train_pin=subject.InputPin(train, "0" * 64, 1, "prepared-train"),
            heldout_pin=subject.InputPin(heldout, sha256_file(heldout), 1, "prepared-heldout"),
            output_dir=tmp_path / "output",
            tokenizer=FakeTokenizer(),
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=identity["aggregate_sha256"],
            expected_chat_template_sha256=identity["chat_template"]["sha256"],
        )
