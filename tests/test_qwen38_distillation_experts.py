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
)
from molor.experimental.qwen38_distillation_experts import (  # noqa: E402
    LEAF_COMMONSENSE_LOGIC,
    LEAF_FORMAL_MATH,
    LEAF_GENERAL_CODE_GENERATION,
    LEAF_NAMES,
    LEAF_QUANTITATIVE_MATH,
    LEAF_SCIENCE_REASONING,
    LEAF_STRICT_INSTRUCTION_FOLLOWING,
    materialize_qwen38_distillation_experts,
)


class PrefixAwareTokenizer:
    """Deterministic byte tokenizer with the Qwen final/trace prefix contract."""

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
    source: str,
    prompt: str,
    assistant: str,
    *,
    quality: float = 9.0,
) -> dict[str, object]:
    return {
        "id": row_id,
        "domain": domain,
        "source": source,
        "difficulty": "fixture",
        "messages": [
            {"role": "system", "content": "Discard the source-specific system prompt."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "ground_truth": "fixture",
        "quality_score": quality,
        "teacher_model": DEFAULT_TEACHER_MODEL,
        "completion_tokens": 8,
        "generation_time": "2026-08-25T00:00:00Z",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(
    tmp_path: Path, rows: list[dict[str, object]]
) -> tuple[tuple[Path, ...], dict[str, str]]:
    paths: list[Path] = []
    for index, split in enumerate(("train", "validation", "test")):
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[index::3], schema=_SCHEMA), path)
        paths.append(path)
    return tuple(paths), {str(path.resolve()): _sha256(path) for path in paths}


def _write_tokenizer_root(tmp_path: Path) -> Path:
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return root


def _write_benchmark(tmp_path: Path, prompt: str) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"cases": [{"prompt": prompt}]}) + "\n", encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fixture_rows() -> tuple[list[dict[str, object]], str]:
    benchmark_prompt = "This exact prompt belongs to the sealed fixture benchmark."
    duplicate_prompt = "Normalize this globally duplicated prompt."
    family_prefix = "Keep this long prompt family together: " + ("f" * 280)
    return (
        [
            _row(
                "quant",
                "math",
                "MetaMathQA",
                "Compute the private quantitative fixture.",
                "<think>Quantitative trace.</think>\n\nQuantitative answer.",
            ),
            _row(
                "formal",
                "math",
                "MATH/algebra",
                "Prove the private formal fixture.",
                "<think>Formal trace.</think>\n\nFormal answer.",
            ),
            _row(
                "science",
                "reasoning",
                "SciQ",
                "Explain the private science fixture.",
                "Science answer without an explicit trace.",
            ),
            _row(
                "commonsense",
                "reasoning",
                "ARC-Challenge",
                "Resolve the private logic fixture.",
                "<think>Logic trace.</think>\n\nLogic answer.",
            ),
            _row(
                "code",
                "code",
                "Evol-Code",
                "Implement the private code fixture.",
                "<think>Code trace.</think>\n\nprint('fixture')",
            ),
            _row(
                "instruction",
                "instruction",
                "IFEval",
                "Follow the private strict-format fixture.",
                "Instruction answer.",
            ),
            _row(
                "direct-short-trace-long",
                "math",
                "OrcaMath",
                "Use separate length tiers for direct and traced targets.",
                f"<think>{'r' * 500}</think>\n\nShort answer.",
            ),
            _row(
                "direct-long",
                "code",
                "CodeAlpaca",
                "Put this direct response in the long tier.",
                "d" * 500,
            ),
            _row(
                "duplicate-low",
                "math",
                "gsm8k",
                f"  {duplicate_prompt.upper()}  ",
                "Low quality duplicate.",
                quality=4.0,
            ),
            _row(
                "duplicate-high",
                "reasoning",
                "QASC",
                duplicate_prompt,
                "Higher quality duplicate.",
                quality=9.9,
            ),
            _row(
                "family-a",
                "instruction",
                "tulu-3",
                family_prefix + " alpha",
                "Family alpha.",
            ),
            _row(
                "family-b",
                "instruction",
                "Dolly",
                family_prefix + " beta",
                "Family beta.",
            ),
            _row(
                "benchmark",
                "reasoning",
                "OpenBookQA",
                benchmark_prompt,
                "Must be excluded.",
            ),
            _row(
                "malformed",
                "math",
                "NuminaMath-CoT",
                "Reject this malformed trace.",
                "<think>Never closed.",
            ),
            _row(
                "tool",
                "tool_use",
                "synthetic_tool_use",
                "Quarantine this tool-use row.",
                "Tool answer.",
            ),
            _row(
                "unmapped",
                "instruction",
                "private-unmapped-source",
                "Reject this unmapped source.",
                "Unmapped answer.",
            ),
            _row(
                "mismatch",
                "reasoning",
                "HumanEval",
                "Reject this source-domain mismatch.",
                "Mismatch answer.",
            ),
            _row(
                "above-max",
                "instruction",
                "Dolly",
                "Reject this rendered overlength response.",
                "x" * 2_000,
            ),
        ],
        benchmark_prompt,
    )


def test_materializer_builds_deterministic_granular_mixture_with_complete_accounting(
    tmp_path: Path,
) -> None:
    rows, benchmark_prompt = _fixture_rows()
    inputs, expected_hashes = _write_inputs(tmp_path, rows)
    tokenizer_root = _write_tokenizer_root(tmp_path)
    benchmark = _write_benchmark(tmp_path, benchmark_prompt)
    tokenizer = PrefixAwareTokenizer()
    outputs = (tmp_path / "output-a", tmp_path / "output-b")

    manifests: list[dict[str, object]] = []
    for output in outputs:
        manifest_path = materialize_qwen38_distillation_experts(
            input_paths=inputs,
            output_dir=output,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_root,
            expected_input_hashes=expected_hashes,
            benchmark_suite_paths=(benchmark,),
            short_max_length=400,
            long_max_length=1_024,
            seed="granular-fixture-seed",
        )
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    relative_files = sorted(
        path.relative_to(outputs[0]) for path in outputs[0].rglob("*") if path.is_file()
    )
    assert relative_files
    for relative in relative_files:
        assert (outputs[0] / relative).read_bytes() == (outputs[1] / relative).read_bytes()
    assert manifests[0] == manifests[1]

    expected_leaf_prompts = {
        LEAF_QUANTITATIVE_MATH: "Compute the private quantitative fixture.",
        LEAF_FORMAL_MATH: "Prove the private formal fixture.",
        LEAF_SCIENCE_REASONING: "Explain the private science fixture.",
        LEAF_COMMONSENSE_LOGIC: "Resolve the private logic fixture.",
        LEAF_GENERAL_CODE_GENERATION: "Implement the private code fixture.",
        LEAF_STRICT_INSTRUCTION_FOLLOWING: "Follow the private strict-format fixture.",
    }
    all_prompts_by_leaf: dict[str, set[str]] = {}
    for leaf in LEAF_NAMES:
        prompts: set[str] = set()
        for stream in ("direct-short", "direct-long", "trace-short", "trace-long"):
            for split in ("train", "validation", "test"):
                path = outputs[0] / leaf / stream / f"{split}.jsonl"
                assert path.is_file()
                for row in _read_rows(path):
                    assert set(row) == {"messages"}
                    messages = row["messages"]
                    assert isinstance(messages, list)
                    assert [message["role"] for message in messages] == [
                        "system",
                        "user",
                        "assistant",
                    ]
                    assert messages[0]["content"] == NEUTRAL_SYSTEM_PROMPT
                    assert all(set(message) == {"role", "content"} for message in messages)
                    prompts.add(messages[1]["content"])
                    enable_thinking = stream.startswith("trace-")
                    kwargs = {
                        "tokenize": True,
                        "enable_thinking": enable_thinking,
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
                    if stream.endswith("-short"):
                        assert len(full_ids) <= 400
                    else:
                        assert 400 < len(full_ids) <= 1_024
        all_prompts_by_leaf[leaf] = prompts
    for leaf, prompt in expected_leaf_prompts.items():
        assert prompt in all_prompts_by_leaf[leaf]

    long_trace_prompt = "Use separate length tiers for direct and traced targets."
    assert any(
        row["messages"][1]["content"] == long_trace_prompt
        for split in ("train", "validation", "test")
        for row in _read_rows(
            outputs[0] / LEAF_QUANTITATIVE_MATH / "direct-short" / f"{split}.jsonl"
        )
    )
    assert any(
        row["messages"][1]["content"] == long_trace_prompt
        for split in ("train", "validation", "test")
        for row in _read_rows(outputs[0] / LEAF_QUANTITATIVE_MATH / "trace-long" / f"{split}.jsonl")
    )

    all_prompts = set().union(*all_prompts_by_leaf.values())
    assert benchmark_prompt not in all_prompts
    assert "Reject this malformed trace." not in all_prompts
    assert "Quarantine this tool-use row." not in all_prompts
    assert "Reject this unmapped source." not in all_prompts
    assert "Reject this source-domain mismatch." not in all_prompts
    assert "Reject this rendered overlength response." not in all_prompts
    assert (
        "Normalize this globally duplicated prompt." in all_prompts_by_leaf[LEAF_SCIENCE_REASONING]
    )

    manifest = manifests[0]
    coverage = manifest["quality"]["coverage"]
    assert coverage["input_records"] == len(rows)
    assert coverage["rejected_by_reason"] == {
        "benchmark_exact_prompt_overlap": 1,
        "direct_rendered_above_max": 1,
        "malformed_or_empty_thinking_target": 1,
        "source_domain_mismatch": 1,
        "unmapped_source": 1,
    }
    assert coverage["quarantined_by_reason"] == {"tool_use": 1}
    assert coverage["prompt_duplicates_removed"] == 1
    assert coverage["zero_silent_drops"] is True
    for equation in (
        "input_disposition_equation",
        "dedup_equation",
        "direct_emission_equation",
        "trace_disposition_equation",
    ):
        assert coverage[equation]["balanced"] is True
    assert manifest["quality"]["global_prompt_split_overlap"] == 0
    assert manifest["quality"]["global_split_family_overlap"] == 0
    assert manifest["quality"]["all_outputs_within_declared_tiers"] is True

    mixture = json.loads((outputs[0] / "mixture-manifest.json").read_text(encoding="utf-8"))
    assert mixture["leaf_order"] == list(LEAF_NAMES)
    assert mixture["contains_raw_examples"] is False
    assert mixture["record_metadata_policy"] == "aggregate-and-hash-only"
    sums = (outputs[0] / "sha256sums.txt").read_text(encoding="ascii").splitlines()
    for line in sums:
        digest, relative = line.split("  ", 1)
        assert _sha256(outputs[0] / relative) == digest


def test_materializer_source_map_covers_the_pinned_non_tool_sources() -> None:
    from molor.experimental.qwen38_distillation_experts import leaf_for_source

    expected = {
        "MetaMathQA": LEAF_QUANTITATIVE_MATH,
        "OrcaMath": LEAF_QUANTITATIVE_MATH,
        "gsm8k": LEAF_QUANTITATIVE_MATH,
        "NuminaMath-CoT": LEAF_FORMAL_MATH,
        "MATH/geometry": LEAF_FORMAL_MATH,
        "SciQ": LEAF_SCIENCE_REASONING,
        "QASC": LEAF_SCIENCE_REASONING,
        "OpenBookQA": LEAF_SCIENCE_REASONING,
        "CommonsenseQA": LEAF_COMMONSENSE_LOGIC,
        "ARC-Easy": LEAF_COMMONSENSE_LOGIC,
        "ARC-Challenge": LEAF_COMMONSENSE_LOGIC,
        "Evol-Code": LEAF_GENERAL_CODE_GENERATION,
        "CodeAlpaca": LEAF_GENERAL_CODE_GENERATION,
        "MBPP": LEAF_GENERAL_CODE_GENERATION,
        "HumanEval": LEAF_GENERAL_CODE_GENERATION,
        "tulu-3": LEAF_STRICT_INSTRUCTION_FOLLOWING,
        "Dolly": LEAF_STRICT_INSTRUCTION_FOLLOWING,
        "IFEval": LEAF_STRICT_INSTRUCTION_FOLLOWING,
    }
    assert {source: leaf_for_source(source) for source in expected} == expected
    assert leaf_for_source("synthetic_tool_use") is None
    assert leaf_for_source("private-unknown") is None


def test_materializer_rejects_hash_mismatch_before_publication(tmp_path: Path) -> None:
    rows = [
        _row(f"row-{index}", "instruction", "Dolly", f"Prompt {index}.", "Answer.")
        for index in range(3)
    ]
    inputs, expected_hashes = _write_inputs(tmp_path, rows)
    expected_hashes[str(inputs[2].resolve())] = "0" * 64
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        materialize_qwen38_distillation_experts(
            input_paths=inputs,
            output_dir=output,
            tokenizer=PrefixAwareTokenizer(),
            tokenizer_path=_write_tokenizer_root(tmp_path),
            expected_input_hashes=expected_hashes,
            benchmark_suite_paths=(_write_benchmark(tmp_path, "Unrelated benchmark."),),
        )
    assert not output.exists()
