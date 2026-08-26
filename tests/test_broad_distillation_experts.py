from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from molor.experimental.broad_distillation_experts import (  # noqa: E402
    CONTRACT_TOOL_DIRECT,
    CONTRACT_TOOL_TRACE,
    CORPUS_COMPLETE_FABLE,
    CORPUS_FABLE_MAX_AUXILIARY,
    CORPUS_FABLE_PREMIUM,
    CORPUS_MYTHOS,
    LEAF_NAMES,
    LEAF_TOOL_USE,
    _assert_non_security_path,
    materialize_broad_distillation_experts,
    metadata_security_gate,
)


class PrefixAwareTokenizer:
    """Deterministic byte tokenizer honoring Qwen's direct/trace prefix contract."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_: Any,
    ) -> list[int]:
        rendered = ""
        for message in conversation:
            role = message["role"]
            content = message.get("content", "")
            rendered += f"<{role}>\n{content}"
            if message.get("tool_calls"):
                rendered += json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
            if message.get("tool_call_id"):
                rendered += f"[{message['tool_call_id']}]"
            rendered += f"</{role}>\n"
        if add_generation_prompt:
            rendered += "<assistant>\n<think>\n"
            if not enable_thinking:
                rendered += "\n</think>\n\n"
        return list(rendered.encode("utf-8"))


_PREMIUM_SCHEMA = pa.schema(
    [
        pa.field("messages", pa.string()),
        pa.field("model", pa.string()),
        pa.field("quality_scores", pa.string()),
        pa.field("source", pa.string()),
        pa.field("source_row_hash", pa.string()),
        pa.field("reasoning", pa.string()),
        pa.field("category", pa.string()),
    ]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _messages(prompt: str, answer: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def _prepared(
    category: str,
    prompt: str,
    answer: str,
    *,
    upstream: str = "fixture-source",
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "upstream": upstream,
        "messages": messages if messages is not None else _messages(prompt, answer),
    }


def _write_jsonl(path: Path, rows: list[object], *, blank_line: bool = False) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    if blank_line:
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _premium_row(
    row_hash: str,
    messages: list[dict[str, Any]],
    *,
    reasoning: str | None = None,
    category: str = "agentic coding",
) -> dict[str, object]:
    return {
        "messages": json.dumps(messages, ensure_ascii=False),
        "model": "fixture-teacher",
        "quality_scores": json.dumps({"overall": 9.0}),
        "source": "fixture-premium",
        "source_row_hash": row_hash,
        "reasoning": reasoning,
        "category": category,
    }


def _write_fixture_inputs(
    root: Path,
) -> tuple[dict[str, tuple[Path, ...]], dict[str, str], str]:
    root.mkdir()
    sentinel = "forbidden-payload-sentinel"

    mythos_rows = [
        _prepared(
            "mathematical_reasoning",
            "Compute the Mythos fixture.",
            "<think>Use arithmetic.</think>\n\n42",
        ),
        {
            "category": "cybersecurity",
            "upstream": "blocked-fixture",
            "messages": _messages(sentinel, sentinel),
        },
    ]
    mythos_paths = (root / "mythos-train.jsonl", root / "mythos-validation.jsonl")
    _write_jsonl(mythos_paths[0], mythos_rows, blank_line=True)
    mythos_paths[1].write_text("{invalid json\n", encoding="utf-8")

    routed_rows = [
        _prepared("arithmetic problem", "Quantitative fixture", "Quantitative answer"),
        _prepared("theorem proving", "Formal fixture", "<think>Proof.</think>\n\nQED"),
        _prepared("physics reasoning", "Science fixture", "Science answer"),
        _prepared("mechanical engineering", "Engineering fixture", "Engineering answer"),
        _prepared("python code generation", "Code fixture", "print('ok')"),
        _prepared("repository code debugging", "Debug fixture", "Patch answer"),
        _prepared("software architecture", "Architecture fixture", "Architecture answer"),
        _prepared("distributed database", "Systems fixture", "Systems answer"),
        _prepared("react frontend", "Frontend fixture", "Frontend answer"),
        _prepared("machine learning data analysis", "ML fixture", "ML answer"),
        _prepared("agentic planning", "Planning fixture", "Planning answer"),
        _prepared("general reasoning", "General fixture", "General answer"),
        _prepared("instruction following", "Strict fixture", "Strict answer"),
    ]
    tool_messages = [
        {"role": "user", "content": "Use the fixture tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "Tool final"},
    ]
    routed_rows.append(
        _prepared(
            "function calling",
            "Use the fixture tool.",
            "unused",
            messages=tool_messages,
        )
    )
    exact_duplicate = _prepared("python code generation", "Exact duplicate fixture", "Same answer")
    routed_rows.extend([exact_duplicate, exact_duplicate.copy()])
    routed_rows.extend(
        [
            _prepared("instruction following", "Variant family fixture", "Variant one"),
            _prepared("instruction following", "  VARIANT FAMILY FIXTURE  ", "Variant two"),
            _prepared("instruction following", "Mid-length fixture", "m" * 4_500),
            _prepared("instruction following", "Long-length fixture", "l" * 9_000),
            _prepared("instruction following", "Deferred fixture", "d" * 33_000),
            _prepared("general reasoning", "Malformed trace", "<think>never closed"),
            {
                "category": "general reasoning",
                "upstream": "fixture-source",
                "messages": [{"role": "user", "content": "No assistant target"}],
            },
            _prepared(
                "general reasoning",
                "Sealed benchmark fixture",
                "Must be rejected",
            ),
        ]
    )
    complete_paths = (root / "complete-train.jsonl", root / "complete-validation.jsonl")
    _write_jsonl(complete_paths[0], routed_rows[::2])
    _write_jsonl(complete_paths[1], routed_rows[1::2])

    premium_tool_messages = [
        {"role": "user", "content": "Premium tool session"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "premium-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "premium result", "tool_call_id": "premium-call"},
        {"role": "assistant", "content": "Premium final"},
    ]
    premium_multi_messages = [
        {"role": "user", "content": "Premium multi session"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow up one"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Follow up two"},
        {"role": "assistant", "content": "Third answer"},
    ]
    premium_rows = [
        _premium_row("premium-tool", premium_tool_messages, reasoning="Tool reasoning"),
        _premium_row("premium-multi", premium_multi_messages),
        _premium_row(
            "premium-too-long",
            _messages("Premium oversized target", "p" * 5_000),
        ),
    ]
    premium_paths = (
        root / "premium-train.parquet",
        root / "premium-validation.parquet",
        root / "premium-test.parquet",
    )
    for path, rows in zip(premium_paths, ([premium_rows[0]], [premium_rows[1]], [premium_rows[2]])):
        pq.write_table(pa.Table.from_pylist(rows, schema=_PREMIUM_SCHEMA), path)

    max_paths = (root / "max-train.jsonl", root / "max-validation.jsonl")
    _write_jsonl(
        max_paths[0],
        [{"category": "general reasoning", "upstream": "max-fixture", "thinking": sentinel}],
    )
    _write_jsonl(
        max_paths[1],
        [{"category": "security research", "upstream": "blocked-fixture", "thinking": sentinel}],
    )

    paths = {
        CORPUS_MYTHOS: mythos_paths,
        CORPUS_COMPLETE_FABLE: complete_paths,
        CORPUS_FABLE_PREMIUM: premium_paths,
        CORPUS_FABLE_MAX_AUXILIARY: max_paths,
    }
    hashes = {
        str(path.resolve()): _sha256(path)
        for corpus_paths in paths.values()
        for path in corpus_paths
    }
    return paths, hashes, sentinel


def _write_tokenizer_root(root: Path) -> Path:
    path = root / "tokenizer"
    path.mkdir()
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return path


def _write_benchmark(root: Path) -> Path:
    path = root / "benchmark.json"
    path.write_text(
        json.dumps({"cases": [{"prompt": "Sealed benchmark fixture"}]}) + "\n",
        encoding="utf-8",
    )
    return path


def test_materializer_emits_deterministic_granular_rack_with_balanced_accounting(
    tmp_path: Path,
) -> None:
    inputs, hashes, sentinel = _write_fixture_inputs(tmp_path / "inputs")
    tokenizer_root = _write_tokenizer_root(tmp_path)
    benchmark = _write_benchmark(tmp_path)
    outputs = (tmp_path / "output-a", tmp_path / "output-b")

    for output in outputs:
        materialize_broad_distillation_experts(
            input_paths=inputs,
            output_dir=output,
            tokenizer=PrefixAwareTokenizer(),
            tokenizer_path=tokenizer_root,
            expected_input_hashes=hashes,
            benchmark_suite_paths=(benchmark,),
            seed="broad-fixture-seed",
            max_variants_per_prompt_stream=1,
            premium_targets_per_session=2,
            premium_context_tokens=4_096,
        )

    relative_files = sorted(
        path.relative_to(outputs[0]) for path in outputs[0].rglob("*") if path.is_file()
    )
    assert relative_files
    for relative in relative_files:
        assert (outputs[0] / relative).read_bytes() == (outputs[1] / relative).read_bytes()

    manifest = json.loads((outputs[0] / "quality-manifest.json").read_text(encoding="utf-8"))
    coverage = manifest["quality"]["coverage"]
    assert coverage["zero_silent_drops"] is True
    for equation in (
        "source_row_equation",
        "example_assignment_equation",
        "assignment_equation",
        "dedup_equation",
        "cap_equation",
    ):
        assert coverage[equation]["balanced"] is True
    assert coverage["guarded_rows"] == 2
    assert coverage["auxiliary_archive_rows"] == 1
    assert coverage["exact_content_duplicates_removed"] == 1
    assert coverage["variant_capped"] == 2
    assert coverage["deferred_examples"] == 1
    assert (
        coverage["example_rejected_by_corpus_and_reason"][CORPUS_FABLE_PREMIUM][
            "unable_to_form_bounded_slice"
        ]
        == 1
    )
    assert manifest["quality"]["premium_target_accounting"]["assistant_targets_capped"] == 1

    emitted_leaves = set(coverage["emitted_by_leaf"])
    assert emitted_leaves == set(LEAF_NAMES)
    assert coverage["emitted_by_leaf"][LEAF_TOOL_USE] >= 2
    assigned_contracts = coverage["assigned_by_corpus_and_contract"][CORPUS_FABLE_PREMIUM]
    assert assigned_contracts[CONTRACT_TOOL_DIRECT] == 1
    assert assigned_contracts[CONTRACT_TOOL_TRACE] == 1
    assert manifest["quality"]["global_split_family_overlap"] == 0
    assert manifest["quality"]["all_emitted_within_32768_tokens"] is True

    training_rows = 0
    output_text = ""
    for path in outputs[0].rglob("*.jsonl"):
        output_text += path.read_text(encoding="utf-8")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert set(row) == {"messages"}
            assert row["messages"][-1]["role"] == "assistant"
            training_rows += 1
    assert training_rows == coverage["emitted_examples"]
    assert sentinel not in output_text
    assert CORPUS_FABLE_MAX_AUXILIARY not in coverage["emitted_by_corpus"]

    sums = (outputs[0] / "sha256sums.txt").read_text(encoding="ascii").splitlines()
    assert sums == sorted(sums, key=lambda line: line.split("  ", 1)[1])


class MetadataOnlyMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == "category":
            return "cybersecurity"
        raise AssertionError(f"unexpected content access: {key}")

    def __iter__(self):
        return iter(("category", "messages"))

    def __len__(self) -> int:
        return 2

    def get(self, key: str, default: object = None) -> object:
        if key == "category":
            return "cybersecurity"
        raise AssertionError(f"unexpected content access: {key}")


def test_metadata_gate_stops_before_message_access() -> None:
    assert metadata_security_gate(MetadataOnlyMapping()) == "category"


def test_path_boundary_is_checked_before_any_file_read() -> None:
    with pytest.raises(ValueError, match="outside the non-security corpus boundary"):
        _assert_non_security_path(Path("C:/fixture/cyber-corpus/missing.jsonl"))
