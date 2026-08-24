from dataclasses import asdict
import json

import pytest
import torch

from molor.experimental.hashed_kv import HashedKVConfig, evaluate_hashed_kv
from molor.experimental.hashed_kv_lab import (
    _compact_result,
    _context_case,
    _context_ids,
    _query_positions,
    _resolve_layers,
    _summarize_records,
    _validate_context_length,
)


class CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [2 + (ord(character) % 251) for character in text]}


def test_context_builder_is_exact_and_context_file_is_only_capped(tmp_path) -> None:
    tokenizer = CharacterTokenizer()
    built_in = _context_ids(tokenizer, context_file=None, target_tokens=1_024)
    _, evidence_span = _context_case(tokenizer, context_file=None, target_tokens=1_024)
    context_path = tmp_path / "context.txt"
    context_path.write_text("short context", encoding="utf-8")
    from_file = _context_ids(tokenizer, context_file=context_path, target_tokens=1_024)

    assert built_in.shape == (1_024,)
    assert evidence_span is not None
    assert evidence_span[0] % 64 == 0
    assert evidence_span[0] >= 64
    assert evidence_span[1] <= 1_024 - 64
    assert from_file.shape == (len("short context") + 1,)
    assert _query_positions(10, 3) == (7, 8, 9)
    assert _resolve_layers((0, -1, 0), 4) == (0, 3)
    _validate_context_length(1_024, 1_024)
    with pytest.raises(ValueError, match="supports at most"):
        _validate_context_length(1_025, 1_024)


def test_compact_report_excludes_dense_routing_tensors_and_summarizes() -> None:
    generator = torch.Generator().manual_seed(17)
    result = evaluate_hashed_kv(
        torch.randn(8, generator=generator),
        torch.randn(32, 8, generator=generator),
        torch.randn(32, 6, generator=generator),
        config=HashedKVConfig(
            page_size=4,
            candidate_pages=4,
            sink_pages=1,
            recent_pages=1,
            top_k=4,
            seed=9,
        ),
    )

    compact = _compact_result(result, known_evidence_span=(12, 16))
    serialized = json.dumps(compact)
    record = {
        "requested_candidate_fraction": 0.5,
        "config": asdict(result.config),
        "result": compact,
    }
    summary = _summarize_records([record])

    assert "key_routing_codes" not in serialized
    assert "exact_attention_weights" not in serialized
    assert "minmax_metrics" in compact
    assert "forced_metrics" in compact
    assert compact["attribution"]["forced_attention_mass"] > 0
    assert compact["known_evidence"]["page_ids"] == [3]
    assert compact["routing_metadata"]["bits_per_token_per_indexed_head"] == 32
    assert compact["routing_metadata"]["minmax_float32_descriptor_if_materialized"] == {
        "bytes_per_page_per_indexed_head": 64,
        "nominal_bytes_per_token_per_indexed_head": 16.0,
    }
    assert compact["routing_metadata"]["key_code_shape"] == [32, 1]
    assert len(compact["exact_attention_top"]) == 4
    assert summary["evaluations"] == 1
    assert len(summary["configuration_groups"]) == 1
    group = summary["configuration_groups"][0]
    assert group["hashed_top_k_recall_mean"] == result.hashed_metrics.top_k_recall
    assert group["minmax_top_k_recall_mean"] == result.minmax_metrics.top_k_recall
    assert group["known_evidence_forced_page_recall_mean"] == 0.0


def test_summary_keeps_incompatible_hash_configurations_separate() -> None:
    base_result = {
        "forced_metrics": {
            "top_k_recall": 0.25,
            "retained_attention_mass": 0.5,
            "relative_output_l2": 0.4,
        },
        "hashed_metrics": {
            "top_k_recall": 0.5,
            "retained_attention_mass": 0.75,
            "relative_output_l2": 0.2,
            "candidate_token_fraction": 0.25,
        },
        "minmax_metrics": {
            "top_k_recall": 0.75,
            "retained_attention_mass": 0.8,
            "relative_output_l2": 0.1,
        },
        "oracle_metrics": {
            "top_k_recall": 1.0,
            "retained_attention_mass": 0.9,
            "relative_output_l2": 0.05,
        },
        "attribution": {
            "hashed": {
                "incremental_attention_mass_over_forced": 0.25,
                "cold_attention_mass_recall": 0.5,
            },
            "minmax": {
                "incremental_attention_mass_over_forced": 0.3,
                "cold_attention_mass_recall": 0.6,
            },
            "oracle": {
                "incremental_attention_mass_over_forced": 0.4,
                "cold_attention_mass_recall": 0.8,
            },
        },
        "known_evidence": None,
    }
    config = {
        "page_size": 64,
        "hash_bits": 32,
        "hash_tables": 1,
        "candidate_pages": 4,
        "sink_pages": 1,
        "recent_pages": 1,
        "top_k": 32,
        "seed": 1,
    }
    records = [
        {
            "requested_candidate_fraction": 0.25,
            "config": config,
            "result": base_result,
        },
        {
            "requested_candidate_fraction": 0.25,
            "config": {**config, "hash_bits": 64},
            "result": base_result,
        },
    ]

    assert len(_summarize_records(records)["configuration_groups"]) == 2
