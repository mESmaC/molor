"""Train the taxonomy-bound root/family router for the granular 0.6B expert rack."""

from __future__ import annotations

import argparse
from pathlib import Path

from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.training.hierarchical_router import train_hierarchical_router_checkpoint
from molor.training.router import RouterFeatureContract, RouterTrainingConfig


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY = WORKSPACE / "experiments" / "expert-rack-v5" / "expert-taxonomy-v1.json"
DEFAULT_MODEL = WORKSPACE / "models" / "Qwen3-0.6B"
DEFAULT_ADAPTER_ROOT = WORKSPACE / "checkpoints" / "qwen3-0.6b-expert-rack-v5"
DEFAULT_OUTPUT = WORKSPACE / "checkpoints" / "qwen3-0.6b-hierarchical-router-v1"
DEFAULT_CACHE = WORKSPACE / "runs" / "qwen3-0.6b-hierarchical-router-v1" / "features"


def _adapter_override(value: str) -> tuple[str, Path]:
    leaf_id, separator, raw_path = value.partition("=")
    if not separator or not leaf_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("adapter must be LEAF_ID=PATH")
    if leaf_id != leaf_id.strip() or any(character.isspace() for character in leaf_id):
        raise argparse.ArgumentTypeError("adapter leaf ID must not contain whitespace")
    return leaf_id, Path(raw_path).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a class-balanced base/family/leaf router over frozen Qwen input embeddings."
        )
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--adapter",
        type=_adapter_override,
        action="append",
        default=[],
        metavar="LEAF_ID=PATH",
        help="override one canonical leaf adapter; repeat as needed",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    parser.add_argument("--z-loss-weight", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "router-head device; CPU is the deterministic default because the frozen feature "
            "matrix is already CPU-resident"
        ),
    )
    parser.add_argument("--router-hidden-size", type=int, default=256)
    parser.add_argument("--limit-per-route", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    taxonomy = load_expert_taxonomy(args.taxonomy)
    adapters = {leaf_id: args.adapter_root / leaf_id for leaf_id in taxonomy.leaf_ids}
    overridden: set[str] = set()
    for leaf_id, path in args.adapter:
        if leaf_id not in adapters:
            raise ValueError(f"adapter override is not a canonical taxonomy leaf: {leaf_id!r}")
        if leaf_id in overridden:
            raise ValueError(f"adapter override was supplied more than once: {leaf_id!r}")
        overridden.add(leaf_id)
        adapters[leaf_id] = path
    if args.router_hidden_size < 0:
        raise ValueError("--router-hidden-size must be zero or positive")
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
        router_hidden_size=(None if args.router_hidden_size == 0 else args.router_hidden_size),
    )
    result = train_hierarchical_router_checkpoint(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        taxonomy_path=args.taxonomy,
        model_dir=args.model_dir,
        adapters=adapters,
        output_dir=args.output_dir,
        feature_contract=RouterFeatureContract(max_tokens=args.max_tokens),
        training=training,
        feature_batch_size=args.feature_batch_size,
        cache_dir=None if args.no_cache else args.cache_dir,
        limit_per_route=args.limit_per_route,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Saved router weights: {result.checkpoint_path}", flush=True)
    print(f"Saved router contract: {result.config_path}", flush=True)
    print(
        f"Validation joint accuracy: {result.metrics['validation']['accuracy']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
