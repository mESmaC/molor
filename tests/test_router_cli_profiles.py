from __future__ import annotations

import json

import pytest
from torch import nn

from molor import RouterConfig
from molor.experimental.autorouter import PromptAutoRouter, ROUTER_CHECKPOINT_SCHEMA
from molor.model_profiles import (
    corpus_expert_root,
    router_feature_cache_root,
    router_output_root,
)
from molor.training.router import DEFAULT_ROUTE_IDS
from molor.training.router_cli import _WORKSPACE_ROOT, _parser, _resolve_adapters


def test_router_cli_defaults_to_isolated_06_paths() -> None:
    args = _parser().parse_args([])
    adapter_root = corpus_expert_root(args.model_dir, _WORKSPACE_ROOT)

    assert args.model_dir.name == "Qwen3-0.6B"
    assert adapter_root is not None
    assert adapter_root.name == "qwen3-0.6b-corpus-experts"
    assert router_output_root(args.model_dir, _WORKSPACE_ROOT).name == (
        "qwen3-0.6b-corpus-router-v1"
    )
    assert "qwen3-0.6b-development" in str(
        router_feature_cache_root(args.model_dir, _WORKSPACE_ROOT)
    )

    adapters, route_ids = _resolve_adapters([], adapter_root)
    assert tuple(adapters) == DEFAULT_ROUTE_IDS[:-1]
    assert all(path.parent == adapter_root for path in adapters.values())
    assert route_ids == DEFAULT_ROUTE_IDS


def test_router_checkpoint_rejects_a_different_base_before_loading_weights(tmp_path) -> None:
    checkpoint = tmp_path / "router"
    checkpoint.mkdir()
    (checkpoint / "router_config.json").write_text(
        json.dumps(
            {
                "schema_version": ROUTER_CHECKPOINT_SCHEMA,
                "router_config": RouterConfig(
                    hidden_size=4,
                    num_experts=1,
                    top_k=1,
                ).to_dict(),
                "route_ids": ["expert", "base/null"],
                "feature_contract": {},
                "model": {"model_id": "base-a"},
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "router.safetensors").write_bytes(b"not-read")

    with pytest.raises(ValueError, match="different frozen base model"):
        PromptAutoRouter.load_checkpoint(
            checkpoint,
            tokenizer=object(),
            input_embeddings=nn.Embedding(8, 4),
            available_experts=("expert",),
            expected_model_id="base-b",
        )
