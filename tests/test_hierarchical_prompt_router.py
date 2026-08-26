from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from molor.desktop.backend import DesktopSession
from molor.experimental.autorouter import (
    BASE_ROUTE,
    ROUTER_CHECKPOINT_SCHEMA,
    AutoRouterPolicy,
    PromptAutoRouter,
)
from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.experimental.hierarchical_prompt_router import HierarchicalPromptAutoRouter
from molor.training.hierarchical_router import (
    HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA,
    HierarchicalRouterAxes,
    HierarchicalRouterHeads,
    _artifact,
    _seal_checkpoint_bundle,
)
from molor.training.router import (
    RouterExample,
    RouterFeatureContract,
    extract_prompt_features,
)


WORKSPACE = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = WORKSPACE / "experiments" / "expert-rack-v5" / "expert-taxonomy-v1.json"


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []
        self.truncation_side = "right"

    def __call__(self, prompts: object, **kwargs: object) -> dict[str, torch.Tensor]:
        self.calls.append((prompts, kwargs))
        return {
            "input_ids": torch.tensor([[1, 2, 7]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        }


class _HeadTailTokenizer:
    pad_token_id = 0
    truncation_side = "right"

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, prompts: object, **kwargs: object) -> dict[str, object]:
        self.calls.append((prompts, kwargs))
        return {"input_ids": [[1, 2, 3, 4, 5, 6]]}


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    feature_contract: RouterFeatureContract | None = None,
    tokenizer: Any | None = None,
) -> tuple[Path, Path, dict[str, Path], nn.Embedding, Any]:
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    axes = HierarchicalRouterAxes.from_taxonomy(taxonomy)
    router = HierarchicalRouterHeads(hidden_size=4, axes=axes)
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
        root = router.root_network[-1]
        business = router.leaf_networks["business"][-1]
        assert isinstance(root, nn.Linear) and isinstance(business, nn.Linear)
        root.bias[axes.root_routes.index("business")] = 8.0
        business_leaves = dict(axes.family_leaves)["business"]
        business.bias[business_leaves.index("business-finance")] = 8.0

    adapters: dict[str, Path] = {}
    adapter_metadata: list[dict[str, object]] = []
    for leaf_id in taxonomy.leaf_ids:
        directory = tmp_path / "adapters" / leaf_id
        directory.mkdir(parents=True)
        config = directory / "adapter_config.json"
        weights = directory / "adapter_model.safetensors"
        config.write_text(json.dumps({"name": leaf_id}) + "\n", encoding="utf-8")
        weights.write_bytes(f"synthetic-{leaf_id}".encode())
        adapters[leaf_id] = directory
        adapter_metadata.append(
            {
                "route_id": leaf_id,
                "path": str(directory.resolve()),
                "config": _artifact(config),
                "weights": _artifact(weights),
            }
        )

    model = tmp_path / "model"
    model.mkdir()
    model_config = model / "config.json"
    model_weights = model / "model.safetensors"
    tokenizer_config = model / "tokenizer_config.json"
    model_config.write_text('{"hidden_size":4}\n', encoding="utf-8")
    model_weights.write_bytes(b"synthetic-frozen-embedding")
    tokenizer_config.write_text("{}\n", encoding="utf-8")
    sidecar = {
        "schema_version": HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA,
        "taxonomy": {
            "taxonomy_id": taxonomy.taxonomy_id,
            "revision": taxonomy.revision,
            "content_sha256": taxonomy.fingerprint,
            "artifact": _artifact(TAXONOMY_PATH),
        },
        "axes": axes.to_dict(),
        "model": {
            "model_id": str(model.resolve()),
            "hidden_size": 4,
            "config": _artifact(model_config),
            "embedding": {
                "tensor_name": "model.embed_tokens.weight",
                "shape": [8, 4],
                "dtype": "float32",
                "artifact": _artifact(model_weights),
            },
            "tokenizer_artifacts": [_artifact(tokenizer_config)],
        },
        "adapters": adapter_metadata,
        "feature_contract": (feature_contract or RouterFeatureContract(max_tokens=32)).to_dict(),
        "head_config": {
            "hidden_size": 4,
            "router_hidden_size": None,
            "root_classes": len(axes.root_routes),
            "conditional_classes": {
                family_id: len(leaves) for family_id, leaves in axes.family_leaves
            },
        },
    }
    checkpoint = tmp_path / "checkpoint"
    _seal_checkpoint_bundle(
        destination=checkpoint,
        router=router,
        sidecar=sidecar,
        taxonomy_path=TAXONOMY_PATH,
        adapters=adapters,
        model_id=model,
    )
    embedding = nn.Embedding(8, 4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    return checkpoint, model, adapters, embedding, tokenizer or _Tokenizer()


def test_checkpoint_runtime_extracts_exact_features_and_adapts_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, model, adapters, embedding, tokenizer = _checkpoint_fixture(tmp_path)
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    runtime = HierarchicalPromptAutoRouter.load_checkpoint(
        checkpoint,
        tokenizer=tokenizer,
        input_embeddings=embedding,
        available_experts=taxonomy.leaf_ids,
        adapters=adapters,
        expected_model_id=model,
    )
    observed: list[torch.Tensor] = []
    original = runtime.checkpoint.router.inference_logits

    def capture(features: torch.Tensor):
        observed.append(features.detach().cpu().clone())
        return original(features)

    monkeypatch.setattr(runtime.checkpoint.router, "inference_logits", capture)
    decision = runtime.decide(
        "build a forecast",
        AutoRouterPolicy(top_k=1, max_prompt_tokens=32),
    )

    expected = (embedding.weight[1].float() + embedding.weight[2].float()) / 2
    assert torch.equal(observed[0], expected)
    assert tokenizer.calls == [
        (
            ["build a forecast"],
            {
                "padding": True,
                "truncation": True,
                "max_length": 32,
                "add_special_tokens": True,
                "return_tensors": "pt",
            },
        )
    ]
    assert decision.selected_routes == ("business-finance",)
    assert decision.applied_route == (("business-finance", 1.0),)
    assert tuple(decision.probability_map) == (BASE_ROUTE, *taxonomy.leaf_ids)
    assert sum(decision.probability_map.values()) == pytest.approx(1.0)
    assert "base/null" not in decision.probability_map
    assert decision.checkpoint_loaded is True


def test_head_tail_runtime_features_match_training_extraction(tmp_path: Path) -> None:
    contract = RouterFeatureContract(
        max_tokens=4,
        truncation="head_tail",
        add_special_tokens=False,
    )
    tokenizer = _HeadTailTokenizer()
    checkpoint, model, adapters, embedding, tokenizer = _checkpoint_fixture(
        tmp_path,
        feature_contract=contract,
        tokenizer=tokenizer,
    )
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    runtime = HierarchicalPromptAutoRouter.load_checkpoint(
        checkpoint,
        tokenizer=tokenizer,
        input_embeddings=embedding,
        available_experts=taxonomy.leaf_ids,
        adapters=adapters,
        expected_model_id=model,
    )
    training_features, _labels = extract_prompt_features(
        [RouterExample("long prompt", BASE_ROUTE)],
        tokenizer=tokenizer,
        embedding=embedding.weight,
        route_ids=(BASE_ROUTE,),
        contract=contract,
        batch_size=1,
    )

    runtime_features = runtime._prompt_features("long prompt")

    assert torch.equal(runtime_features.cpu(), training_features)
    assert runtime_features.shape == (1, embedding.embedding_dim)
    assert tokenizer.calls == [
        (
            ["long prompt"],
            {
                "padding": False,
                "truncation": False,
                "add_special_tokens": False,
            },
        ),
        (
            ["long prompt"],
            {
                "padding": False,
                "truncation": False,
                "add_special_tokens": False,
            },
        ),
    ]


def test_runtime_requires_every_checkpoint_adapter_to_be_resident(tmp_path: Path) -> None:
    checkpoint, model, adapters, embedding, tokenizer = _checkpoint_fixture(tmp_path)
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)

    with pytest.raises(ValueError, match="resident adapters must exactly match every"):
        HierarchicalPromptAutoRouter.load_checkpoint(
            checkpoint,
            tokenizer=tokenizer,
            input_embeddings=embedding,
            available_experts=taxonomy.leaf_ids[:-1],
            adapters=adapters,
            expected_model_id=model,
        )


def test_runtime_preserves_manual_routing_and_hierarchy_top_two_limit(tmp_path: Path) -> None:
    checkpoint, model, adapters, embedding, tokenizer = _checkpoint_fixture(tmp_path)
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    runtime = HierarchicalPromptAutoRouter.load_checkpoint(
        checkpoint,
        tokenizer=tokenizer,
        input_embeddings=embedding,
        available_experts=taxonomy.leaf_ids,
        adapters=adapters,
        expected_model_id=model,
    )

    manual = runtime.decide(
        "not tokenized in manual mode",
        AutoRouterPolicy(
            mode="manual",
            manual_route={"business-finance": 3.0, BASE_ROUTE: 1.0},
            max_prompt_tokens=32,
        ),
    )
    assert manual.applied_route == ((BASE_ROUTE, 0.25), ("business-finance", 0.75))
    assert not tokenizer.calls
    with pytest.raises(ValueError, match="at most two active adapters"):
        runtime.decide(
            "route this",
            AutoRouterPolicy(top_k=3, max_prompt_tokens=32),
        )


def test_runtime_maps_hierarchical_null_fallback_to_desktop_base(tmp_path: Path) -> None:
    checkpoint, model, adapters, embedding, tokenizer = _checkpoint_fixture(tmp_path)
    taxonomy = load_expert_taxonomy(TAXONOMY_PATH)
    runtime = HierarchicalPromptAutoRouter.load_checkpoint(
        checkpoint,
        tokenizer=tokenizer,
        input_embeddings=embedding,
        available_experts=taxonomy.leaf_ids,
        adapters=adapters,
        expected_model_id=model,
    )

    decision = runtime.decide(
        "use the base",
        AutoRouterPolicy(null_bias=20.0, max_prompt_tokens=32),
    )

    assert decision.selected_routes == (BASE_ROUTE,)
    assert decision.applied_route == ((BASE_ROUTE, 1.0),)
    assert "base/null" not in decision.probability_map
    assert sum(decision.probability_map.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("schema", "expected_loader"),
    [
        (ROUTER_CHECKPOINT_SCHEMA, "flat"),
        (HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA, "hierarchical"),
    ],
)
def test_desktop_checkpoint_dispatch_preserves_flat_and_adds_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    expected_loader: str,
) -> None:
    checkpoint = tmp_path / expected_loader
    checkpoint.mkdir()
    (checkpoint / "router_config.json").write_text(
        json.dumps({"schema_version": schema}), encoding="utf-8"
    )
    sentinel = object()
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def flat(*args: Any, **kwargs: Any) -> object:
        calls.append(("flat", args, kwargs))
        return sentinel

    def hierarchical(*args: Any, **kwargs: Any) -> object:
        calls.append(("hierarchical", args, kwargs))
        return sentinel

    monkeypatch.setattr(PromptAutoRouter, "load_checkpoint", staticmethod(flat))
    monkeypatch.setattr(
        HierarchicalPromptAutoRouter,
        "load_checkpoint",
        staticmethod(hierarchical),
    )
    result = DesktopSession._load_prompt_router_checkpoint(
        checkpoint,
        tokenizer=object(),
        input_embeddings=object(),
        available_experts=("leaf",),
        adapters={"leaf": tmp_path / "adapter"},
        expected_model_id=str(tmp_path / "model"),
    )

    assert result is sentinel
    assert [name for name, _args, _kwargs in calls] == [expected_loader]
    if expected_loader == "flat":
        assert "adapters" not in calls[0][2]
    else:
        assert calls[0][2]["adapters"] == {"leaf": tmp_path / "adapter"}


def test_desktop_checkpoint_dispatch_rejects_unknown_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unknown"
    checkpoint.mkdir()
    (checkpoint / "router_config.json").write_text(
        json.dumps({"schema_version": "unknown.v1"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported router checkpoint schema"):
        DesktopSession._load_prompt_router_checkpoint(
            checkpoint,
            tokenizer=object(),
            input_embeddings=object(),
            available_experts=(),
            adapters={},
            expected_model_id=str(tmp_path / "model"),
        )


def test_desktop_checkpoint_dispatch_rejects_duplicate_schema_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "duplicate"
    checkpoint.mkdir()
    (checkpoint / "router_config.json").write_text(
        '{"schema_version":"first","schema_version":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key 'schema_version'"):
        DesktopSession._load_prompt_router_checkpoint(
            checkpoint,
            tokenizer=object(),
            input_embeddings=object(),
            available_experts=(),
            adapters={},
            expected_model_id=str(tmp_path / "model"),
        )
