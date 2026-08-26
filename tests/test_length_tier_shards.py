from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from molor.experimental.length_tier_shards import (
    InputShardDescriptor,
    LengthTierPlan,
    PLAN_SCHEMA_VERSION,
    TEMPLATE_MODES,
    TIER_SPECS,
    TokenizerDescriptor,
    load_plan,
    materialize_length_tiers,
    sha256_file,
    tier_for_token_count,
    tokenizer_identity,
)


class FakeTokenizer:
    """Return the synthetic LENGTH marker as the rendered token count."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, str]], **kwargs: object
    ) -> list[int]:
        self.calls.append(dict(kwargs))
        joined = "\n".join(message["content"] for message in conversation)
        match = re.search(r"LENGTH=(\d+)", joined)
        if match is None:
            raise RuntimeError("synthetic record has no length marker")
        return [7] * int(match.group(1))


def _messages(target: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "Synthetic system prompt."},
            {"role": "user", "content": "Synthetic user prompt."},
            {"role": "assistant", "content": target},
        ]
    }


def _tool_messages(length: int) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "Synthetic tool system prompt."},
            {"role": "user", "content": "Use the synthetic tool."},
            {"role": "assistant", "content": "Synthetic tool request."},
            {"role": "tool", "content": "Synthetic tool result."},
            {"role": "assistant", "content": f"Synthetic tool final. LENGTH={length}"},
        ]
    }


def _write_shard(path: Path, rows: list[object]) -> tuple[str, int]:
    with path.open("wb") as stream:
        for row in rows:
            if isinstance(row, bytes):
                stream.write(row)
            else:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )
            stream.write(b"\n")
    return sha256_file(path), len(rows)


def _write_tokenizer_root(root: Path) -> dict[str, object]:
    root.mkdir()
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "synthetic-chat-template-v1"}) + "\n",
        encoding="utf-8",
    )
    return tokenizer_identity(root)


def _make_plan(
    tokenizer_root: Path,
    tokenizer_receipt: dict[str, object],
    shards: list[tuple[Path, str, int, str]],
) -> LengthTierPlan:
    chat_template = tokenizer_receipt["chat_template"]
    assert isinstance(chat_template, dict)
    return LengthTierPlan(
        tokenizer=TokenizerDescriptor(
            path=tokenizer_root,
            aggregate_sha256=str(tokenizer_receipt["aggregate_sha256"]),
            chat_template_sha256=str(chat_template["sha256"]),
        ),
        inputs=tuple(
            InputShardDescriptor(
                path=path,
                sha256=digest,
                record_count=record_count,
                template_mode=mode,
            )
            for path, digest, record_count, mode in shards
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_materializer_is_deterministic_exact_and_never_truncates(tmp_path: Path) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_receipt = _write_tokenizer_root(tokenizer_root)

    direct_path = tmp_path / "direct.jsonl"
    direct_hash, direct_count = _write_shard(
        direct_path,
        [
            _messages("Direct boundary. LENGTH=1024"),
            _messages("Direct upper tier. LENGTH=1025"),
            _messages("<think>\n\n</think>\n\nCanonical direct. LENGTH=512"),
            _messages("<think>Trace in direct.</think>\n\nFinal. LENGTH=20"),
            {**_messages("Extra field. LENGTH=20"), "metadata": "not allowed"},
            b"{invalid-json",
        ],
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_hash, trace_count = _write_shard(
        trace_path,
        [
            _messages("<think>Trace.</think>\n\nFinal. LENGTH=4096"),
            _messages("<think>Trace.</think>\n\nFinal. LENGTH=4097"),
            _messages("<think>Unclosed trace. LENGTH=20"),
            _messages("<think>   </think>\n\nFinal. LENGTH=20"),
        ],
    )
    tool_path = tmp_path / "tool.jsonl"
    tool_hash, tool_count = _write_shard(
        tool_path,
        [
            _tool_messages(8192),
            _tool_messages(8193),
            _tool_messages(32768),
            _tool_messages(32769),
            _messages("Tool mode without tool role. LENGTH=20"),
        ],
    )
    plan = _make_plan(
        tokenizer_root,
        tokenizer_receipt,
        [
            (direct_path, direct_hash, direct_count, "direct"),
            (trace_path, trace_hash, trace_count, "trace"),
            (tool_path, tool_hash, tool_count, "tool"),
        ],
    )

    outputs = (tmp_path / "output-a", tmp_path / "output-b")
    tokenizers = (FakeTokenizer(), FakeTokenizer())
    for output, tokenizer in zip(outputs, tokenizers, strict=True):
        manifest_path = materialize_length_tiers(
            plan=plan,
            output_dir=output,
            tokenizer=tokenizer,
        )
        assert manifest_path == output / "manifest.json"

    relative_files = sorted(
        path.relative_to(outputs[0]) for path in outputs[0].rglob("*") if path.is_file()
    )
    assert relative_files
    for relative in relative_files:
        assert (outputs[0] / relative).read_bytes() == (outputs[1] / relative).read_bytes()

    expected_counts = {
        ("direct", "le-1024"): 2,
        ("direct", "1025-4096"): 1,
        ("trace", "1025-4096"): 1,
        ("trace", "4097-8192"): 1,
        ("tool", "4097-8192"): 1,
        ("tool", "8193-32768"): 2,
        ("tool", "deferred-gt32768"): 1,
    }
    for mode in TEMPLATE_MODES:
        for tier, _, _ in TIER_SPECS:
            rows = _read_jsonl(outputs[0] / mode / f"{tier}.jsonl")
            assert len(rows) == expected_counts.get((mode, tier), 0)
            for row in rows:
                assert set(row) == {"messages"}
                assert isinstance(row["messages"], list)
                assert all(set(message) == {"role", "content"} for message in row["messages"])

    manifest = json.loads((outputs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["truncation"] is False
    assert manifest["accounting"]["input_records"] == 15
    assert manifest["accounting"]["accepted_records"] == 9
    assert manifest["accounting"]["quarantined_records"] == 6
    assert manifest["accounting"]["fully_reconciled"] is True
    assert manifest["accounting"]["quarantine_reasons"] == {
        "empty_trace_or_final": 1,
        "invalid_json": 1,
        "malformed_trace_target": 1,
        "not_exact_messages_only_object": 1,
        "tool_mode_without_tool_role": 1,
        "trace_target_in_direct_mode": 1,
    }
    assert all(receipt["hash_verified"] for receipt in manifest["provenance"]["inputs"])
    assert all(receipt["record_count_verified"] for receipt in manifest["provenance"]["inputs"])
    assert all(receipt["stream_hash_verified"] for receipt in manifest["provenance"]["inputs"])
    assert "LENGTH=" not in json.dumps(manifest)
    quarantine_text = (outputs[0] / "quarantine.jsonl").read_text(encoding="utf-8")
    assert "LENGTH=" not in quarantine_text
    assert "Unclosed trace" not in quarantine_text

    calls = tokenizers[0].calls
    assert len(calls) == 9
    assert all(call["truncation"] is False for call in calls)
    assert all(call["padding"] is False for call in calls)
    assert all(call["add_generation_prompt"] is False for call in calls)
    assert sum(call["enable_thinking"] is True for call in calls) == 2


def test_pin_or_record_count_mismatch_never_publishes_output(tmp_path: Path) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_receipt = _write_tokenizer_root(tokenizer_root)
    shard = tmp_path / "input.jsonl"
    digest, count = _write_shard(shard, [_messages("Valid. LENGTH=10")])

    bad_input_plan = _make_plan(
        tokenizer_root,
        tokenizer_receipt,
        [(shard, "0" * 64, count, "direct")],
    )
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        materialize_length_tiers(
            plan=bad_input_plan,
            output_dir=tmp_path / "bad-input-output",
            tokenizer=FakeTokenizer(),
        )
    assert not (tmp_path / "bad-input-output").exists()

    wrong_count_plan = _make_plan(
        tokenizer_root,
        tokenizer_receipt,
        [(shard, digest, count + 1, "direct")],
    )
    with pytest.raises(ValueError, match="input record-count mismatch"):
        materialize_length_tiers(
            plan=wrong_count_plan,
            output_dir=tmp_path / "bad-count-output",
            tokenizer=FakeTokenizer(),
        )
    assert not (tmp_path / "bad-count-output").exists()

    bad_tokenizer_plan = LengthTierPlan(
        tokenizer=TokenizerDescriptor(
            path=tokenizer_root,
            aggregate_sha256="f" * 64,
            chat_template_sha256=str(tokenizer_receipt["chat_template"]["sha256"]),
        ),
        inputs=(
            InputShardDescriptor(
                path=shard,
                sha256=digest,
                record_count=count,
                template_mode="direct",
            ),
        ),
    )
    with pytest.raises(ValueError, match="tokenizer aggregate SHA-256 mismatch"):
        materialize_length_tiers(
            plan=bad_tokenizer_plan,
            output_dir=tmp_path / "bad-tokenizer-output",
            tokenizer=FakeTokenizer(),
        )
    assert not (tmp_path / "bad-tokenizer-output").exists()


def test_load_plan_resolves_relative_pins_and_rejects_unknown_fields(tmp_path: Path) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_receipt = _write_tokenizer_root(tokenizer_root)
    shard = tmp_path / "input.jsonl"
    digest, count = _write_shard(shard, [_messages("Valid. LENGTH=10")])
    chat_template = tokenizer_receipt["chat_template"]
    assert isinstance(chat_template, dict)
    document = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "tokenizer": {
            "path": "tokenizer",
            "aggregate_sha256": tokenizer_receipt["aggregate_sha256"],
            "chat_template_sha256": chat_template["sha256"],
        },
        "inputs": [
            {
                "path": "input.jsonl",
                "sha256": digest,
                "record_count": count,
                "template_mode": "direct",
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(document), encoding="utf-8")
    plan = load_plan(plan_path)
    assert plan.tokenizer.path == tokenizer_root.resolve()
    assert plan.inputs[0].path == shard.resolve()

    document["unexpected"] = True
    plan_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="plan must contain exactly"):
        load_plan(plan_path)


@pytest.mark.parametrize(
    ("token_count", "tier"),
    [
        (1, "le-1024"),
        (1024, "le-1024"),
        (1025, "1025-4096"),
        (4096, "1025-4096"),
        (4097, "4097-8192"),
        (8192, "4097-8192"),
        (8193, "8193-32768"),
        (32768, "8193-32768"),
        (32769, "deferred-gt32768"),
    ],
)
def test_tier_boundaries(token_count: int, tier: str) -> None:
    assert tier_for_token_count(token_count) == tier


@pytest.mark.parametrize("token_count", [0, -1, 1.5, True])
def test_tier_boundaries_reject_invalid_counts(token_count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tier_for_token_count(token_count)  # type: ignore[arg-type]
