from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from molor.experimental.reasoning_corpus import (
    ALLOWED_SOURCES,
    DEFAULT_SPLIT_COUNTS,
    DEFAULT_TOKENIZER,
    NEUTRAL_SYSTEM_PROMPT,
    _split_source_quotas,
    materialize_reasoning_corpus,
    normalize_prompt,
    render_token_ids,
)


class PrefixConsistentTokenizer:
    """Tiny chat-template stand-in with Qwen's thinking-enabled prefix behavior."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_: Any,
    ) -> list[int]:
        assert enable_thinking is True
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in conversation
        )
        if add_generation_prompt:
            rendered += "<assistant><think>\n"
        return list(rendered.encode("utf-8"))


def test_default_quota_contract_is_source_disjoint_and_preregistered() -> None:
    assert _split_source_quotas(DEFAULT_SPLIT_COUNTS) == {
        "train": {ALLOWED_SOURCES[0]: 9_000, ALLOWED_SOURCES[1]: 3_000},
        "validation": {ALLOWED_SOURCES[2]: 512},
        "test": {ALLOWED_SOURCES[3]: 300, ALLOWED_SOURCES[4]: 212},
    }


def _record(
    source: str,
    prompt: str,
    *,
    thinking: str = "Work through the facts carefully.",
    answer: str = "This is the final answer.",
    system: str | None = None,
) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.extend(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "thinking": thinking,
                "content": answer,
            },
        ]
    )
    return {"id": prompt, "upstream": source, "messages": messages}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_inputs(tmp_path: Path) -> tuple[tuple[Path, Path], dict[str, str], Path]:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    source_counts = (5, 3, 3, 2, 2)
    for source_index, count in enumerate(source_counts):
        for row_index in range(count):
            rows.append(
                _record(
                    ALLOWED_SOURCES[source_index],
                    f"Solve source {source_index} item {row_index}.",
                    system="You are the source-specific assistant; ignore this prompt.",
                )
            )

    # These normalize to the same prompt group; only one deterministic canonical row may survive.
    rows.append(_record(ALLOWED_SOURCES[1], "  Normalize   This Prompt  "))
    rows.append(_record(ALLOWED_SOURCES[2], "normalize this prompt"))

    rows.extend(
        [
            _record(ALLOWED_SOURCES[0], "Tell me about Claude's approach."),
            _record(ALLOWED_SOURCES[0], "Solve this GSM8K benchmark item."),
            _record(ALLOWED_SOURCES[0], "Do not retain <think> tags."),
            _record(ALLOWED_SOURCES[0], "x" * 1_500),
            _record("unapproved/source", "This source is not approved."),
            {
                "upstream": ALLOWED_SOURCES[0],
                "messages": [
                    {"role": "user", "content": "Missing reasoning."},
                    {"role": "assistant", "content": "Answer only."},
                ],
            },
            {
                "upstream": ALLOWED_SOURCES[0],
                "messages": [
                    {"role": "user", "content": "First turn."},
                    {
                        "role": "assistant",
                        "thinking": "Think one.",
                        "content": "One.",
                    },
                    {"role": "user", "content": "Second turn."},
                    {
                        "role": "assistant",
                        "thinking": "Think two.",
                        "content": "Two.",
                    },
                ],
            },
        ]
    )
    first = tmp_path / "prepared-train.jsonl"
    second = tmp_path / "prepared-validation.jsonl"
    _write_jsonl(first, rows[::2])
    _write_jsonl(second, rows[1::2])
    hashes = {str(first.resolve()): _sha256(first), str(second.resolve()): _sha256(second)}
    return (first, second), hashes, tokenizer_root


def test_materializer_is_deterministic_strict_group_disjoint_and_auditable(
    tmp_path: Path,
) -> None:
    inputs, expected_hashes, tokenizer_root = _fixture_inputs(tmp_path)
    outputs = (tmp_path / "output-a", tmp_path / "output-b")

    manifests: list[dict[str, object]] = []
    for output in outputs:
        manifest_path = materialize_reasoning_corpus(
            input_paths=inputs,
            output_dir=output,
            tokenizer=PrefixConsistentTokenizer(),
            tokenizer_path=tokenizer_root,
            expected_input_hashes=expected_hashes,
            split_counts={"train": 4, "validation": 2, "test": 2},
            max_length=1024,
            seed="fixture-seed",
        )
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "quality-manifest.json"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    assert manifests[0] == manifests[1]

    prompt_groups: dict[str, set[str]] = {}
    expected_counts = {"train": 4, "validation": 2, "test": 2}
    for split, expected_count in expected_counts.items():
        path = outputs[0] / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == expected_count
        prompt_groups[split] = set()
        for row in rows:
            assert set(row) == {"messages"}
            assert [message["role"] for message in row["messages"]] == [
                "system",
                "user",
                "assistant",
            ]
            assert all(set(message) == {"role", "content"} for message in row["messages"])
            assert row["messages"][0]["content"] == NEUTRAL_SYSTEM_PROMPT
            assert "source-specific assistant" not in json.dumps(row)
            assert row["messages"][-1]["content"].startswith("<think>\n")
            assert "\n</think>\n\n" in row["messages"][-1]["content"]
            prompt_groups[split].add(normalize_prompt(row["messages"][1]["content"]))

    assert prompt_groups["train"].isdisjoint(prompt_groups["validation"])
    assert prompt_groups["train"].isdisjoint(prompt_groups["test"])
    assert prompt_groups["validation"].isdisjoint(prompt_groups["test"])

    manifest = manifests[0]
    assert manifest["quality"]["selected_records"] == 8
    assert manifest["quality"]["prompt_group_split_overlap"]["zero_overlap"] is True
    assert manifest["quality"]["source_split_overlap"]["zero_overlap"] is True
    assert manifest["quality"]["prompt_group_duplicates_removed"] == 1
    assert manifest["quality"]["rejections"] == {
        "benchmark_mention": 1,
        "not_simple_reasoning_turn": 2,
        "rendered_too_long": 1,
        "reserved_chat_marker": 1,
        "source_not_allowlisted": 1,
        "teacher_identity_mention": 1,
    }
    per_source = manifest["selection"]["per_source"]
    assert sum(item["shortfall"] for item in per_source.values()) == 0
    assert sum(item["same_source_fill_added"] for item in per_source.values()) == 0
    assert sum(item["cross_source_fill_added"] for item in per_source.values()) == 0
    assert sum(item["final_selected"] for item in per_source.values()) == 8
    assert manifest["selection"]["quotas_by_split"] == {
        "train": {ALLOWED_SOURCES[0]: 3, ALLOWED_SOURCES[1]: 1},
        "validation": {ALLOWED_SOURCES[2]: 2},
        "test": {ALLOWED_SOURCES[3]: 1, ALLOWED_SOURCES[4]: 1},
    }
    split_source_sets = {
        split: set(receipt["per_source"]) for split, receipt in manifest["splits"].items()
    }
    assert split_source_sets["train"] == set(ALLOWED_SOURCES[:2])
    assert split_source_sets["validation"] == {ALLOWED_SOURCES[2]}
    assert split_source_sets["test"] == set(ALLOWED_SOURCES[3:])
    assert manifest["token_stats"]["max"] <= 1024
    assert all(item["hash_verified"] for item in manifest["provenance"]["inputs"])
    for split, receipt in manifest["outputs"].items():
        assert receipt["sha256"] == _sha256(outputs[0] / receipt["path"])
        assert receipt["records"] == expected_counts[split]


def test_materializer_rejects_an_unverified_input_before_publication(tmp_path: Path) -> None:
    inputs, _, tokenizer_root = _fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_reasoning_corpus(
            input_paths=inputs,
            output_dir=tmp_path / "must-not-exist",
            tokenizer=PrefixConsistentTokenizer(),
            tokenizer_path=tokenizer_root,
            expected_input_hashes={
                str(inputs[0].resolve()): "0" * 64,
                str(inputs[1].resolve()): _sha256(inputs[1]),
            },
            split_counts={"train": 4, "validation": 2, "test": 2},
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_materializer_refuses_cross_source_fill_on_quota_shortfall(tmp_path: Path) -> None:
    inputs, expected_hashes, tokenizer_root = _fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="cross-source fill is forbidden"):
        materialize_reasoning_corpus(
            input_paths=inputs,
            output_dir=tmp_path / "must-not-cross-fill",
            tokenizer=PrefixConsistentTokenizer(),
            tokenizer_path=tokenizer_root,
            expected_input_hashes=expected_hashes,
            split_counts={"train": 20, "validation": 2, "test": 2},
        )
    assert not (tmp_path / "must-not-cross-fill").exists()


@pytest.mark.skipif(not DEFAULT_TOKENIZER.is_dir(), reason="local Qwen3-0.6B is unavailable")
def test_local_qwen_thinking_enabled_template_preserves_training_prefix() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user", "content": "What is two plus two?"},
        {
            "role": "assistant",
            "content": "<think>\nTwo plus two is four.\n</think>\n\n4",
        },
    ]
    prompt_ids, full_ids = render_token_ids(tokenizer, messages)
    assert full_ids[: len(prompt_ids)] == prompt_ids
    assert len(full_ids) <= 1024

    disabled_prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        truncation=False,
        padding=False,
        return_tensors=None,
        return_dict=False,
    )
    assert full_ids[: len(disabled_prompt)] != disabled_prompt
