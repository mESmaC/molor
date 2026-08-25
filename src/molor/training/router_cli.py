"""CLI for the lightweight source-label prompt-router bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

from molor.model_profiles import (
    corpus_expert_root,
    router_feature_cache_root,
    router_output_root,
)
from molor.training.router import (
    DEFAULT_ROUTE_IDS,
    RouterFeatureContract,
    RouterTrainingConfig,
    train_router_checkpoint,
)


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_ROOT = _WORKSPACE_ROOT / "experiments" / "first-real-experts"
_DATA_ROOT = _EXPERIMENT_ROOT / "data"


def _adapter_override(value: str) -> tuple[str, Path]:
    route_id, separator, raw_path = value.partition("=")
    if not separator or not route_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("adapter must be ROUTE_ID=PATH")
    if route_id != route_id.strip() or any(character.isspace() for character in route_id):
        raise argparse.ArgumentTypeError("adapter route ID must not contain whitespace")
    if route_id == DEFAULT_ROUTE_IDS[-1]:
        raise argparse.ArgumentTypeError("base/null is structural and cannot have an adapter")
    return route_id, Path(raw_path.strip()).expanduser()


def _resolve_adapters(
    overrides: list[tuple[str, Path]],
    adapter_root: Path,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    adapters = {route_id: adapter_root / route_id for route_id in DEFAULT_ROUTE_IDS[:-1]}
    for route_id, path in overrides:
        adapters[route_id] = path
    return adapters, (*adapters, DEFAULT_ROUTE_IDS[-1])


def _progress(message: str) -> None:
    print(message, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a top-1 auto-router from cached frozen Qwen input embeddings; "
            "the causal LM is never loaded or run."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=_DATA_ROOT / "router-bootstrap-v1.train.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=_DATA_ROOT / "router-bootstrap-v1.validation.jsonl",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=_DATA_ROOT / "router-bootstrap-v1.test.jsonl",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_WORKSPACE_ROOT / "models" / "Qwen3-0.6B",
        help="local Hugging Face model directory containing Qwen input embeddings",
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=None,
        help="default adapter rack; omitted derives an isolated root from --model-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="router output; omitted derives an isolated path from --model-dir",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="pooled features; omitted derives a model-specific cache from --model-dir",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write pooled feature caches",
    )
    parser.add_argument(
        "--adapter",
        type=_adapter_override,
        action="append",
        default=[],
        metavar="ROUTE_ID=PATH",
        help=("override a default adapter path or add an expert route; repeat for several routes"),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    parser.add_argument("--z-loss-weight", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument(
        "--device",
        default="cpu",
        help="router-head training device; CPU is the non-disruptive default",
    )
    parser.add_argument(
        "--router-hidden-size",
        type=int,
        default=0,
        help="0 uses a linear head; use 256 for the small MLP ablation",
    )
    parser.add_argument(
        "--limit-per-route",
        type=int,
        default=None,
        help="debug/smoke limit applied independently to every split and route",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)

    adapter_root = args.adapter_root or corpus_expert_root(args.model_dir, _WORKSPACE_ROOT)
    output_dir = args.output_dir or router_output_root(args.model_dir, _WORKSPACE_ROOT)
    cache_dir = args.cache_dir or router_feature_cache_root(args.model_dir, _WORKSPACE_ROOT)
    if adapter_root is None:
        parser.error("unknown model lane: pass --adapter-root for a custom base")
    if output_dir is None:
        parser.error("unknown model lane: pass --output-dir for a custom base")
    if cache_dir is None and not args.no_cache:
        parser.error("unknown model lane: pass --cache-dir or --no-cache for a custom base")

    adapters, route_ids = _resolve_adapters(args.adapter, adapter_root)
    hidden_size = args.router_hidden_size
    if hidden_size < 0:
        parser.error("--router-hidden-size must be 0 or a positive integer")
    training = RouterTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        load_balance_weight=args.load_balance_weight,
        z_loss_weight=args.z_loss_weight,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        router_hidden_size=None if hidden_size == 0 else hidden_size,
    )
    feature_contract = RouterFeatureContract(max_tokens=args.max_tokens)
    result = train_router_checkpoint(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        model_dir=args.model_dir,
        adapters=adapters,
        output_dir=output_dir,
        route_ids=route_ids,
        feature_contract=feature_contract,
        training=training,
        feature_batch_size=args.feature_batch_size,
        cache_dir=None if args.no_cache else cache_dir,
        limit_per_route=args.limit_per_route,
        progress=_progress,
    )
    print(f"Saved router weights: {result.checkpoint_path}", flush=True)
    print(f"Saved router contract: {result.config_path}", flush=True)
    print(
        f"Validation accuracy: {result.metrics['validation']['accuracy']:.1%}",
        flush=True,
    )
    if "test" in result.metrics:
        print(f"Test accuracy: {result.metrics['test']['accuracy']:.1%}", flush=True)


if __name__ == "__main__":
    main()
