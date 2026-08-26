from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from molor.experimental.qwen38_distillation_corpus import (  # noqa: E402
    DEFAULT_TEACHER_MODEL,
    NEUTRAL_SYSTEM_PROMPT,
    VIEW_CODE_FINAL,
    VIEW_INSTRUCTION_FINAL,
    VIEW_NAMES,
    VIEW_REASONING_MATH_FINAL,
    VIEW_REASONING_MATH_TRACE,
    materialize_qwen38_distillation_corpus,
    normalize_prompt,
    split_family_id,
)


class PrefixAwareTokenizer:
    """Small deterministic stand-in for Qwen3's final/trace prefix behavior."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_: Any,
    ) -> list[int]:
        rendered = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]
            if role in {"system", "user"}:
                rendered += f"<{role}>\n{content}</{role}>\n"
                continue
            assert role == "assistant"
            reasoning = ""
            final = content
            if "</think>" in content:
                reasoning, final = content.split("</think>", 1)
                reasoning = reasoning.split("<think>", 1)[-1].strip("\n")
                final = final.lstrip("\n")
            rendered += f"<assistant>\n<think>\n{reasoning}\n</think>\n\n{final}</assistant>\n"
        if add_generation_prompt:
            rendered += "<assistant>\n"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
        return list(rendered.encode("utf-8"))


_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("source", pa.string()),
        pa.field("difficulty", pa.string()),
        pa.field(
            "messages",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string()),
                        pa.field("content", pa.string()),
                    ]
                )
            ),
        ),
        pa.field("ground_truth", pa.string()),
        pa.field("quality_score", pa.float64()),
        pa.field("teacher_model", pa.string()),
        pa.field("completion_tokens", pa.int64()),
        pa.field("generation_time", pa.string()),
    ]
)


def _row(
    row_id: str,
    domain: str,
    prompt: str,
    assistant: str,
    *,
    source: str = "private/arbitrary-source-with-no-license-entry",
    quality: float = 9.0,
    ground_truth: str = "fixture answer",
) -> dict[str, object]:
    return {
        "id": row_id,
        "domain": domain,
        "source": source,
        "difficulty": "fixture",
        "messages": [
            {"role": "system", "content": "Discard this source-specific instruction."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "ground_truth": ground_truth,
        "quality_score": quality,
        "teacher_model": DEFAULT_TEACHER_MODEL,
        "completion_tokens": 8,
        "generation_time": "2026-08-25T00:00:00Z",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), path)


def _write_inputs(
    tmp_path: Path, rows: list[dict[str, object]]
) -> tuple[tuple[Path, ...], dict[str, str]]:
    inputs: list[Path] = []
    for index, split in enumerate(("train", "validation", "test")):
        path = tmp_path / f"{split}.parquet"
        _write_parquet(path, rows[index::3])
        inputs.append(path)
    hashes = {str(path.resolve()): _sha256(path) for path in inputs}
    return tuple(inputs), hashes


def _write_tokenizer_root(tmp_path: Path) -> Path:
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return root


def _write_benchmark_suite(tmp_path: Path, prompt: str) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"cases": [{"prompt": prompt}]}) + "\n", encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _all_output_rows(output: Path) -> list[dict[str, object]]:
    return [
        row
        for view in VIEW_NAMES
        for split in ("train", "validation", "test")
        for row in _read_rows(output / view / f"{split}.jsonl")
    ]


def _prompt(row: dict[str, object]) -> str:
    messages = row["messages"]
    assert isinstance(messages, list)
    return messages[1]["content"]


def _assistant(row: dict[str, object]) -> str:
    messages = row["messages"]
    assert isinstance(messages, list)
    return messages[2]["content"]


def _fixture_rows() -> tuple[list[dict[str, object]], str, str, str]:
    benchmark_prompt = "This prompt is an exact sealed benchmark case."
    duplicate_prompt = "  Normalize   This Private Prompt  "
    family_prefix = "Family grouped prompt " + ("z" * 280)
    rows = [
        _row(
            "math-trace",
            "math",
            "Derive the fixture result carefully.",
            "<think>Math trace.</think>\n\nMath final.",
        ),
        _row(
            "reasoning-direct",
            "reasoning",
            "Answer this private reasoning request directly.",
            "Reasoning final without a trace.",
        ),
        _row(
            "reasoning-preamble",
            "reasoning",
            "Preserve the teacher's reasoning preamble.",
            "Preamble reasoning.\n<think>Tagged reasoning.</think>\n\nPreamble final.",
            ground_truth="",
        ),
        _row(
            "code-trace",
            "code",
            "Write a tiny deterministic code fixture.",
            "<think>Code trace is stripped.</think>\n\nprint('ok')",
        ),
        _row(
            "instruction-direct",
            "instruction",
            "Follow this harmless formatting instruction.",
            "Instruction final.",
        ),
        _row(
            "arbitrary-source",
            "instruction",
            "Accept this source without consulting a license allowlist.",
            "Accepted final.",
            source="totally-private/unlisted-teacher-export",
        ),
        _row(
            "duplicate-low",
            "instruction",
            duplicate_prompt,
            "Lower-quality duplicate.",
            quality=5.0,
        ),
        _row(
            "duplicate-high",
            "instruction",
            "normalize this private prompt",
            "Higher-quality duplicate.",
            quality=9.9,
        ),
        _row(
            "family-a",
            "instruction",
            family_prefix + " alpha",
            "Family answer alpha.",
        ),
        _row(
            "family-b",
            "instruction",
            family_prefix + " beta",
            "Family answer beta.",
        ),
        _row(
            "benchmark-exact",
            "instruction",
            benchmark_prompt,
            "Must be excluded.",
        ),
        _row(
            "benchmark-near",
            "instruction",
            benchmark_prompt + " A distinct extension is not an exact prompt.",
            "Near but distinct.",
        ),
        _row(
            "malformed-trace",
            "reasoning",
            "Reject the malformed trace fixture.",
            "<think>Never closed.",
        ),
        _row(
            "overlength",
            "instruction",
            "Reject the rendered overlength fixture.",
            "x" * 2_000,
        ),
        _row(
            "tool-use",
            "tool_use",
            "Quarantine the tool-use fixture.",
            "Tool-use final.",
        ),
    ]
    return rows, benchmark_prompt, duplicate_prompt, family_prefix


def test_materializer_builds_deterministic_private_final_and_trace_views(
    tmp_path: Path,
) -> None:
    rows, benchmark_prompt, duplicate_prompt, family_prefix = _fixture_rows()
    inputs, expected_hashes = _write_inputs(tmp_path, rows)
    tokenizer_root = _write_tokenizer_root(tmp_path)
    benchmark = _write_benchmark_suite(tmp_path, benchmark_prompt)
    tokenizer = PrefixAwareTokenizer()
    outputs = (tmp_path / "output-a", tmp_path / "output-b")

    manifests: list[dict[str, object]] = []
    for output in outputs:
        manifest_path = materialize_qwen38_distillation_corpus(
            input_paths=inputs,
            output_dir=output,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_root,
            expected_input_hashes=expected_hashes,
            benchmark_suite_paths=(benchmark,),
            max_length=1024,
            seed="deterministic-fixture-seed",
        )
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    relative_files = sorted(
        path.relative_to(outputs[0]) for path in outputs[0].rglob("*") if path.is_file()
    )
    assert relative_files
    for relative in relative_files:
        assert (outputs[0] / relative).read_bytes() == (outputs[1] / relative).read_bytes()
    assert manifests[0] == manifests[1]

    final_rows = [
        row
        for view in (
            VIEW_REASONING_MATH_FINAL,
            VIEW_CODE_FINAL,
            VIEW_INSTRUCTION_FINAL,
        )
        for split in ("train", "validation", "test")
        for row in _read_rows(outputs[0] / view / f"{split}.jsonl")
    ]
    trace_rows = [
        row
        for split in ("train", "validation", "test")
        for row in _read_rows(outputs[0] / VIEW_REASONING_MATH_TRACE / f"{split}.jsonl")
    ]
    final_by_prompt = {_prompt(row): row for row in final_rows}
    trace_by_prompt = {_prompt(row): row for row in trace_rows}

    math_prompt = "Derive the fixture result carefully."
    assert _assistant(final_by_prompt[math_prompt]) == "<think>\n\n</think>\n\nMath final."
    assert _assistant(trace_by_prompt[math_prompt]) == (
        "<think>\nMath trace.\n</think>\n\nMath final."
    )
    assert "Answer this private reasoning request directly." not in trace_by_prompt
    preamble_prompt = "Preserve the teacher's reasoning preamble."
    assert _assistant(final_by_prompt[preamble_prompt]) == (
        "<think>\n\n</think>\n\nPreamble final."
    )
    assert _assistant(trace_by_prompt[preamble_prompt]) == (
        "<think>\nPreamble reasoning.\nTagged reasoning.\n</think>\n\nPreamble final."
    )
    code_prompt = "Write a tiny deterministic code fixture."
    assert _assistant(final_by_prompt[code_prompt]) == ("<think>\n\n</think>\n\nprint('ok')")
    assert code_prompt not in trace_by_prompt

    normalized_duplicates = [
        row
        for row in final_rows
        if normalize_prompt(_prompt(row)) == normalize_prompt(duplicate_prompt)
    ]
    assert len(normalized_duplicates) == 1
    assert _assistant(normalized_duplicates[0]).endswith("Higher-quality duplicate.")

    family_rows = [row for row in final_rows if _prompt(row).startswith(family_prefix)]
    assert len(family_rows) == 2
    family_split_membership = {
        split
        for split in ("train", "validation", "test")
        if any(
            _prompt(row).startswith(family_prefix)
            for row in _read_rows(outputs[0] / VIEW_INSTRUCTION_FINAL / f"{split}.jsonl")
        )
    }
    assert len(family_split_membership) == 1
    assert len({split_family_id(_prompt(row)) for row in family_rows}) == 1

    all_prompts = {_prompt(row) for row in _all_output_rows(outputs[0])}
    assert benchmark_prompt not in all_prompts
    assert benchmark_prompt + " A distinct extension is not an exact prompt." in all_prompts
    assert "Reject the malformed trace fixture." not in all_prompts
    assert "Reject the rendered overlength fixture." not in all_prompts
    assert "Quarantine the tool-use fixture." not in all_prompts
    assert "Accept this source without consulting a license allowlist." in all_prompts

    for view in VIEW_NAMES:
        for split in ("train", "validation", "test"):
            for row in _read_rows(outputs[0] / view / f"{split}.jsonl"):
                assert set(row) == {"messages"}
                messages = row["messages"]
                assert [message["role"] for message in messages] == [
                    "system",
                    "user",
                    "assistant",
                ]
                assert messages[0]["content"] == NEUTRAL_SYSTEM_PROMPT
                assert all(set(message) == {"role", "content"} for message in messages)
                kwargs = {
                    "tokenize": True,
                    "enable_thinking": view == VIEW_REASONING_MATH_TRACE,
                    "truncation": False,
                    "padding": False,
                    "return_tensors": None,
                    "return_dict": False,
                }
                prompt_ids = tokenizer.apply_chat_template(
                    messages[:-1], add_generation_prompt=True, **kwargs
                )
                full_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=False, **kwargs
                )
                assert full_ids[: len(prompt_ids)] == prompt_ids
                assert len(full_ids) <= 1024

    manifest = manifests[0]
    assert manifest["usage"] == {
        "scope": "private-only",
        "license_metadata_consulted_for_admission": False,
        "license_fields_read": False,
    }
    quality = manifest["quality"]
    assert quality["prompt_duplicates_removed"] == 1
    assert quality["rejections"]["benchmark_exact_prompt_overlap"] == 1
    assert quality["rejections"]["malformed_or_empty_thinking_target"] == 1
    assert quality["rejections"]["final_rendered_too_long"] == 1
    assert quality["tool_use_quarantined_prompt_groups"] == 1
    assert quality["global_prompt_split_overlap"] == 0
    assert quality["global_split_family_overlap"] == 0
    assert quality["all_output_records_at_or_below_max_length"] is True
    assert all(item["hash_verified"] for item in manifest["provenance"]["inputs"])


def test_materializer_rejects_input_hash_mismatch_before_publication(
    tmp_path: Path,
) -> None:
    rows = [
        _row(f"row-{index}", "instruction", f"Fixture prompt {index}.", "Answer.")
        for index in range(3)
    ]
    inputs, expected_hashes = _write_inputs(tmp_path, rows)
    expected_hashes[str(inputs[1].resolve())] = "0" * 64
    tokenizer_root = _write_tokenizer_root(tmp_path)
    benchmark = _write_benchmark_suite(tmp_path, "Unrelated benchmark prompt.")
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        materialize_qwen38_distillation_corpus(
            input_paths=inputs,
            output_dir=output,
            tokenizer=PrefixAwareTokenizer(),
            tokenizer_path=tokenizer_root,
            expected_input_hashes=expected_hashes,
            benchmark_suite_paths=(benchmark,),
        )
    assert not output.exists()
