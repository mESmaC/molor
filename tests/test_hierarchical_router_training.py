from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

import molor.training.hierarchical_router as hierarchical_router_module
from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.experimental.hierarchical_autorouter import (
    BASE_ROUTE,
    decide_hierarchical_route,
)
from molor.identity import fingerprint_json
from molor.training.hierarchical_router import (
    HierarchicalRouterAxes,
    HierarchicalRouterHeads,
    _artifact,
    _seal_checkpoint_bundle,
    _validate_dataset_isolation,
    evaluate_hierarchical_router,
    fit_hierarchical_router,
    load_hierarchical_router_checkpoint,
)
from molor.training.router import (
    RouterExample,
    RouterFeatureContract,
    RouterTrainingConfig,
    _tokenize_prompt_batch,
    load_router_jsonl,
)


WORKSPACE = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = WORKSPACE / "experiments" / "expert-rack-v5" / "expert-taxonomy-v1.json"
TRAINING_SCRIPT = WORKSPACE / "experiments" / "expert-rack-v5" / "train_hierarchical_router.py"


def _taxonomy():
    return load_expert_taxonomy(TAXONOMY_PATH)


def _training_cli():
    spec = importlib.util.spec_from_file_location(
        "hierarchical_router_training_cli_test", TRAINING_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TokenIdTokenizer:
    pad_token_id = 0

    def __init__(
        self,
        rows: dict[str, list[int]],
        *,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
    ) -> None:
        self.rows = rows
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, object]:
        rows = [list(self.rows[prompt]) for prompt in prompts]
        if kwargs.get("add_special_tokens"):
            rows = [[self.bos_token_id, *row, self.eos_token_id] for row in rows]
        if kwargs.get("truncation"):
            maximum = int(kwargs["max_length"])
            rows = [row[:maximum] for row in rows]
        if not kwargs.get("padding"):
            return {"input_ids": rows}
        width = max(len(row) for row in rows)
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}


def _features_and_routes(*, repeats: int = 3):
    taxonomy = _taxonomy()
    routes = (BASE_ROUTE, *taxonomy.leaf_ids)
    rows: list[torch.Tensor] = []
    labels: list[str] = []
    for route_index, route_id in enumerate(routes):
        for _ in range(repeats):
            row = torch.zeros(len(routes))
            row[route_index] = 4.0
            rows.append(row)
            labels.append(route_id)
    return torch.stack(rows), labels


def _checkpoint_fixture(
    tmp_path: Path,
) -> tuple[Path, HierarchicalRouterHeads, dict[str, Path]]:
    taxonomy = _taxonomy()
    axes = HierarchicalRouterAxes.from_taxonomy(taxonomy)
    router = HierarchicalRouterHeads(hidden_size=4, axes=axes, router_hidden_size=3)
    adapters: dict[str, Path] = {}
    adapter_metadata = []
    for leaf_id in taxonomy.leaf_ids:
        directory = tmp_path / "adapters" / leaf_id
        directory.mkdir(parents=True)
        config = directory / "adapter_config.json"
        weights = directory / "adapter_model.safetensors"
        config.write_text('{"name":"' + leaf_id + '"}\n', encoding="utf-8")
        weights.write_bytes((leaf_id + "-weights").encode())
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
    model_weights.write_bytes(b"frozen-embedding-fixture")
    tokenizer_config.write_text("{}\n", encoding="utf-8")
    sidecar = {
        "schema_version": "molor.hierarchical-router-checkpoint.v1",
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
                "shape": [11, 4],
                "dtype": "float32",
                "artifact": _artifact(model_weights),
            },
            "tokenizer_artifacts": [_artifact(tokenizer_config)],
        },
        "adapters": adapter_metadata,
        "feature_contract": RouterFeatureContract(max_tokens=32).to_dict(),
        "head_config": {
            "hidden_size": 4,
            "router_hidden_size": 3,
            "root_classes": len(axes.root_routes),
            "conditional_classes": {
                family_id: len(leaves) for family_id, leaves in axes.family_leaves
            },
        },
    }
    output = tmp_path / "checkpoint"
    _seal_checkpoint_bundle(
        destination=output,
        router=router,
        sidecar=sidecar,
        taxonomy_path=TAXONOMY_PATH,
        adapters=adapters,
        model_id=model,
    )
    return output, router, adapters


def test_axes_preserve_taxonomy_family_and_leaf_order() -> None:
    taxonomy = _taxonomy()
    axes = HierarchicalRouterAxes.from_taxonomy(taxonomy)

    assert axes.root_routes == (
        BASE_ROUTE,
        *(family.family_id for family in taxonomy.families),
    )
    assert axes.joint_routes == (BASE_ROUTE, *taxonomy.leaf_ids)
    assert tuple(axes.leaf_to_family) == taxonomy.leaf_ids


def test_cli_rejects_duplicate_adapter_overrides() -> None:
    cli = _training_cli()
    leaf_id = _taxonomy().leaf_ids[0]

    with pytest.raises(ValueError, match="supplied more than once"):
        cli.main(
            [
                "--train",
                "train.jsonl",
                "--validation",
                "validation.jsonl",
                "--adapter",
                f"{leaf_id}=first",
                "--adapter",
                f"{leaf_id}=second",
            ]
        )


def test_cli_preserves_right_default_and_accepts_explicit_head_tail() -> None:
    cli = _training_cli()
    required = ["--train", "train.jsonl", "--validation", "validation.jsonl"]

    assert cli._parser().parse_args(required).truncation == "right"
    assert cli._parser().parse_args([*required, "--truncation", "head_tail"]).truncation == (
        "head_tail"
    )


def test_head_tail_tokenization_preserves_rows_within_limit_and_right_pads() -> None:
    tokenizer = _TokenIdTokenizer({"first": [1, 2, 3], "second": [4]})

    input_ids, attention_mask = _tokenize_prompt_batch(
        ["first", "second"],
        tokenizer=tokenizer,
        contract=RouterFeatureContract(
            max_tokens=6,
            truncation="head_tail",
            add_special_tokens=False,
        ),
    )

    assert torch.equal(input_ids, torch.tensor([[1, 2, 3], [4, 0, 0]]))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1], [1, 0, 0]]).float())


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (4, [1, 2, 6, 7]),
        (5, [1, 2, 3, 6, 7]),
    ],
)
def test_head_tail_tokenization_uses_deterministic_even_and_odd_halves(
    maximum: int,
    expected: list[int],
) -> None:
    tokenizer = _TokenIdTokenizer({"long": [1, 2, 3, 4, 5, 6, 7]})

    input_ids, attention_mask = _tokenize_prompt_batch(
        ["long"],
        tokenizer=tokenizer,
        contract=RouterFeatureContract(
            max_tokens=maximum,
            truncation="head_tail",
            add_special_tokens=False,
        ),
    )

    assert input_ids.tolist() == [expected]
    assert attention_mask.tolist() == [[1.0] * maximum]


def test_head_tail_tokenization_slices_after_adding_special_tokens() -> None:
    tokenizer = _TokenIdTokenizer({"long": [1, 2, 3, 4, 5]})

    input_ids, attention_mask = _tokenize_prompt_batch(
        ["long"],
        tokenizer=tokenizer,
        contract=RouterFeatureContract(max_tokens=4, truncation="head_tail"),
    )

    assert input_ids.tolist() == [[101, 1, 5, 102]]
    assert attention_mask.tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_router_feature_contract_rejects_unknown_truncation() -> None:
    with pytest.raises(ValueError, match="right.*head_tail"):
        RouterFeatureContract(truncation="middle")


def test_joint_probabilities_are_normalized_exact_hierarchical_products() -> None:
    axes = HierarchicalRouterAxes.from_taxonomy(_taxonomy())
    router = HierarchicalRouterHeads(hidden_size=4, axes=axes)
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()

    probabilities = router.joint_probabilities(torch.ones(2, 4))

    assert probabilities.shape == (2, len(axes.joint_routes))
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert torch.allclose(probabilities[:, 0], torch.full((2,), 1 / len(axes.root_routes)))
    cursor = 1
    for _family_id, leaves in axes.family_leaves:
        expected = 1 / len(axes.root_routes) / len(leaves)
        assert torch.allclose(
            probabilities[:, cursor : cursor + len(leaves)],
            torch.full((2, len(leaves)), expected),
        )
        cursor += len(leaves)


def test_evaluation_fails_closed_when_a_leaf_has_no_support() -> None:
    features, routes = _features_and_routes(repeats=1)
    axes = HierarchicalRouterAxes.from_taxonomy(_taxonomy())
    router = HierarchicalRouterHeads(hidden_size=features.shape[1], axes=axes)

    with pytest.raises(ValueError, match="has no examples for routes"):
        evaluate_hierarchical_router(
            router,
            features[:-1],
            routes[:-1],
            batch_size=16,
            device=torch.device("cpu"),
        )


def test_runtime_device_equivalence_normalizes_unindexed_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert hierarchical_router_module._same_runtime_device(
        torch.device("cuda:0"), torch.device("cuda")
    )
    assert not hierarchical_router_module._same_runtime_device(
        torch.device("cuda:1"), torch.device("cuda")
    )
    assert hierarchical_router_module._same_runtime_device(torch.device("cpu"), torch.device("cpu"))


def test_partial_test_evaluation_reports_unsupported_routes_without_inventing_rows() -> None:
    features, routes = _features_and_routes(repeats=1)
    axes = HierarchicalRouterAxes.from_taxonomy(_taxonomy())
    router = HierarchicalRouterHeads(hidden_size=features.shape[1], axes=axes)
    missing_route = routes[-1]

    metrics = evaluate_hierarchical_router(
        router,
        features[:-1],
        routes[:-1],
        batch_size=16,
        device=torch.device("cpu"),
        require_all_routes=False,
    )

    assert metrics["unsupported_routes"] == [missing_route]
    assert metrics["per_route"][missing_route]["support"] == 0
    assert metrics["per_route"][missing_route]["recall"] is None
    assert metrics["per_route"][missing_route]["nll"] is None


def test_evaluation_reports_equal_weight_supported_route_macro_nll() -> None:
    axes = HierarchicalRouterAxes.from_taxonomy(_taxonomy())
    router = HierarchicalRouterHeads(hidden_size=4, axes=axes)
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.zero_()
    routes = [BASE_ROUTE, BASE_ROUTE, BASE_ROUTE, BASE_ROUTE, axes.joint_routes[1]]
    features = torch.ones(len(routes), 4)

    metrics = evaluate_hierarchical_router(
        router,
        features,
        routes,
        batch_size=2,
        device=torch.device("cpu"),
        require_all_routes=False,
    )
    probabilities = router.joint_probabilities(features)
    labels = torch.tensor([0, 0, 0, 0, 1])
    row_nll = -probabilities[torch.arange(len(labels)), labels].log()
    expected_macro = (row_nll[:4].mean() + row_nll[4:].mean()) / 2

    assert metrics["loss"] == pytest.approx(float(row_nll.mean().item()))
    assert metrics["supported_route_macro_nll"] == pytest.approx(float(expected_macro.item()))
    assert metrics["supported_route_macro_nll"] != pytest.approx(metrics["loss"])


def test_fit_selects_epoch_by_supported_route_macro_nll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, routes = _features_and_routes(repeats=1)
    validation_metrics = iter(
        (
            {"loss": 0.1, "supported_route_macro_nll": 0.9, "accuracy": 0.8, "macro_recall": 0.8},
            {"loss": 0.2, "supported_route_macro_nll": 0.7, "accuracy": 0.8, "macro_recall": 0.8},
            {"loss": 0.3, "supported_route_macro_nll": 0.6, "accuracy": 0.8, "macro_recall": 0.8},
            {"loss": 0.4, "supported_route_macro_nll": 0.5, "accuracy": 0.8, "macro_recall": 0.8},
        )
    )

    monkeypatch.setattr(
        hierarchical_router_module,
        "evaluate_hierarchical_router",
        lambda *_args, **_kwargs: next(validation_metrics),
    )
    _router, metrics = fit_hierarchical_router(
        features,
        routes,
        features,
        routes,
        taxonomy=_taxonomy(),
        training=RouterTrainingConfig(
            epochs=2,
            batch_size=len(routes),
            learning_rate=0.001,
            weight_decay=0.0,
            label_smoothing=0.0,
            load_balance_weight=0.0,
            z_loss_weight=0.0,
            patience=2,
            seed=83,
            device="cpu",
        ),
    )

    assert metrics["best_epoch"] == 2
    assert metrics["selection"] == {
        "metric": "validation.supported_route_macro_nll",
        "mode": "min",
        "best_value": 0.7,
        "minimum_improvement": 1e-8,
    }


def test_router_jsonl_partial_support_is_explicit_opt_in(tmp_path: Path) -> None:
    dataset = tmp_path / "partial-test.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "General request", "route_id": BASE_ROUTE}) + "\n",
        encoding="utf-8",
    )
    routes = ("specialist", BASE_ROUTE)

    with pytest.raises(ValueError, match="has no examples for routes"):
        load_router_jsonl(dataset, route_ids=routes)

    examples = load_router_jsonl(
        dataset,
        route_ids=routes,
        require_all_routes=False,
    )
    assert examples == [RouterExample("General request", BASE_ROUTE)]


def test_checkpoint_dataset_isolation_rejects_normalized_duplicates_and_leakage() -> None:
    with pytest.raises(ValueError, match="duplicate normalized"):
        _validate_dataset_isolation(
            {
                "train": [
                    RouterExample("Plan  a startup", "business-core"),
                    RouterExample(" plan A STARTUP ", "business-core"),
                ]
            }
        )
    with pytest.raises(ValueError, match="prompt leakage between train and validation"):
        _validate_dataset_isolation(
            {
                "train": [RouterExample("Route this prompt", "business-core")],
                "validation": [RouterExample(" route  this PROMPT ", "business-core")],
            }
        )


def test_evaluation_is_batched_and_never_builds_gradients() -> None:
    features, routes = _features_and_routes(repeats=1)
    axes = HierarchicalRouterAxes.from_taxonomy(_taxonomy())

    class ObservedRouter(HierarchicalRouterHeads):
        def __init__(self) -> None:
            super().__init__(hidden_size=features.shape[1], axes=axes)
            self.calls: list[tuple[int, bool]] = []

        def forward(self, batch: torch.Tensor):
            self.calls.append((batch.shape[0], torch.is_grad_enabled()))
            return super().forward(batch)

    router = ObservedRouter()
    metrics = evaluate_hierarchical_router(
        router,
        features,
        routes,
        batch_size=7,
        device=torch.device("cpu"),
    )

    assert metrics["examples"] == len(routes)
    assert router.calls
    assert all(rows <= 7 and gradients is False for rows, gradients in router.calls)


def test_class_balanced_hierarchical_fit_learns_every_leaf() -> None:
    features, routes = _features_and_routes(repeats=4)
    training = RouterTrainingConfig(
        epochs=80,
        batch_size=52,
        learning_rate=0.08,
        weight_decay=0.0,
        label_smoothing=0.0,
        load_balance_weight=0.0,
        z_loss_weight=0.0,
        patience=12,
        seed=83,
        device="cpu",
        router_hidden_size=None,
    )

    router, metrics = fit_hierarchical_router(
        features,
        routes,
        features.clone(),
        list(routes),
        taxonomy=_taxonomy(),
        training=training,
    )

    assert metrics["validation"]["accuracy"] == 1.0
    assert metrics["validation"]["macro_recall"] == 1.0
    assert metrics["validation"]["root_accuracy"] == 1.0
    assert metrics["validation"]["conditional_leaf_accuracy"] == 1.0
    assert metrics["validation"]["brier"] < 0.01
    assert metrics["class_balance"]["strategy"] == "inverse_frequency_per_head"
    assert router.axes.joint_routes == (BASE_ROUTE, *_taxonomy().leaf_ids)


def test_cpu_fit_is_bitwise_deterministic() -> None:
    features, routes = _features_and_routes(repeats=2)
    training = RouterTrainingConfig(
        epochs=4,
        batch_size=13,
        learning_rate=0.02,
        weight_decay=0.0,
        label_smoothing=0.0,
        load_balance_weight=0.0,
        z_loss_weight=0.0,
        patience=4,
        seed=101,
        device="cpu",
        router_hidden_size=5,
    )

    first, first_metrics = fit_hierarchical_router(
        features,
        routes,
        features,
        routes,
        taxonomy=_taxonomy(),
        training=training,
    )
    second, second_metrics = fit_hierarchical_router(
        features,
        routes,
        features,
        routes,
        taxonomy=_taxonomy(),
        training=training,
    )

    assert first_metrics == second_metrics
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def test_fit_rejects_nonfinite_features_before_optimization() -> None:
    features, routes = _features_and_routes(repeats=1)
    features[0, 0] = float("nan")

    with pytest.raises(ValueError, match="train features must contain only finite values"):
        fit_hierarchical_router(
            features,
            routes,
            torch.nan_to_num(features),
            routes,
            taxonomy=_taxonomy(),
            training=RouterTrainingConfig(epochs=1, patience=1, device="cpu"),
        )


def test_checkpoint_round_trip_matches_request_time_probability_contract(
    tmp_path: Path,
) -> None:
    directory, original, adapters = _checkpoint_fixture(tmp_path)
    model = tmp_path / "model"
    loaded = load_hierarchical_router_checkpoint(
        directory,
        taxonomy_path=TAXONOMY_PATH,
        adapters=adapters,
        expected_model_id=model,
    )
    features = torch.tensor([0.25, -0.5, 1.0, 0.75])

    expected = original.joint_probabilities(features.unsqueeze(0))[0]
    root_logits, conditional_logits = loaded.router.inference_logits(features)
    decision = decide_hierarchical_route(
        _taxonomy(),
        root_logits=root_logits,
        conditional_leaf_logits=conditional_logits,
    )
    observed = decision.joint_probability_map

    assert loaded.adapter_ids == _taxonomy().leaf_ids
    assert loaded.feature_contract == RouterFeatureContract(max_tokens=32)
    assert decision.root_probability_map[BASE_ROUTE] == pytest.approx(
        float(expected[0].item()), abs=1e-7
    )
    assert all(
        observed[route_id] == pytest.approx(float(expected[index].item()), abs=1e-7)
        for index, route_id in enumerate(loaded.axes.joint_routes[1:], start=1)
    )


def test_checkpoint_loader_rejects_sidecar_and_weight_tampering(tmp_path: Path) -> None:
    directory, _router, adapters = _checkpoint_fixture(tmp_path)
    config = directory / "router_config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["head_config"]["root_classes"] += 1
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic fingerprint drift"):
        load_hierarchical_router_checkpoint(
            directory,
            taxonomy_path=TAXONOMY_PATH,
            adapters=adapters,
        )

    payload["content_sha256"] = fingerprint_json(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="class counts differ"):
        load_hierarchical_router_checkpoint(
            directory,
            taxonomy_path=TAXONOMY_PATH,
            adapters=adapters,
        )

    payload["head_config"]["root_classes"] -= 1
    payload["content_sha256"] = fingerprint_json(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    config.write_text(json.dumps(payload), encoding="utf-8")
    weights = directory / "router.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="differs from the pinned artifact"):
        load_hierarchical_router_checkpoint(
            directory,
            taxonomy_path=TAXONOMY_PATH,
            adapters=adapters,
        )


def test_failed_checkpoint_readback_leaves_no_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, router, adapters = _checkpoint_fixture(tmp_path)
    sidecar = json.loads((directory / "router_config.json").read_text(encoding="utf-8"))
    sidecar.pop("checkpoint")
    sidecar.pop("content_sha256")
    destination = tmp_path / "failed-bundle"

    def fail_readback(*_args, **_kwargs):
        raise ValueError("synthetic readback failure")

    monkeypatch.setattr(
        hierarchical_router_module,
        "load_hierarchical_router_checkpoint",
        fail_readback,
    )
    with pytest.raises(ValueError, match="synthetic readback failure"):
        _seal_checkpoint_bundle(
            destination=destination,
            router=router,
            sidecar=sidecar,
            taxonomy_path=TAXONOMY_PATH,
            adapters=adapters,
            model_id=tmp_path / "model",
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-bundle.staging-*"))
