"""Train one frozen-seed binary Taskmaster admission gate on CPU.

The launcher intentionally has no test input.  It validates the exact v7
train/validation corpus and quarantined Taskmaster adapter, then trains a
linear two-route head.  Candidate selection and any blind evaluation belong to
the separately preregistered evaluator, never this launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_router_v7_taskmaster_gate_dataset import (
    DATA_ROOT,
    OUTPUT_DATASET_ID,
    ROUTE_IDS,
    TASKMASTER_ROUTE_ID,
)
from molor.training.router import (
    RouterFeatureContract,
    RouterTrainingConfig,
    RouterTrainingResult,
    train_router_checkpoint,
)


WORKSPACE = Path(__file__).resolve().parents[2]
MODEL_DIR = WORKSPACE / "models" / "Qwen3-1.7B"
ADAPTER_DIR = (
    WORKSPACE / "checkpoints" / "qwen3-1.7b-candidate-experts-quarantine" / TASKMASTER_ROUTE_ID
)
CANDIDATE_ROOT = WORKSPACE / "checkpoints" / "qwen3-1.7b-candidate-routers-quarantine"
FEATURE_CACHE_DIR = WORKSPACE / "runs" / "router-features" / OUTPUT_DATASET_ID
ALLOWED_SEEDS = (83, 8419, 27183)
DATASET_IDENTITIES = {
    "train": {
        "path": DATA_ROOT / f"{OUTPUT_DATASET_ID}.train.jsonl",
        "sha256": "96196ece3c4808c20ac28676727e536f40455dc78a860770bd207a4bf57d09d4",
        "size_bytes": 3050427,
    },
    "validation": {
        "path": DATA_ROOT / f"{OUTPUT_DATASET_ID}.validation.jsonl",
        "sha256": "93f9b74e08acf6556c5ac6a30265f5588fd936eab188045743d8050b8fe47421",
        "size_bytes": 371814,
    },
    "report": {
        "path": DATA_ROOT / f"{OUTPUT_DATASET_ID}.report.json",
        "sha256": "aefd55324fde12052d20275e52190656e90e728f5a536f0cdb62b5aea2594876",
        "size_bytes": 2974,
    },
}
ADAPTER_IDENTITIES = {
    "adapter_config.json": {
        "sha256": "da0feba91aa0013f94f01a1cff9265c658b8fcdf3f3a44b5128f5173d43f04f7",
        "size_bytes": 20989,
    },
    "adapter_model.safetensors": {
        "sha256": "e885123d61f71db0eb2779d41e3fb3cf7e9eb0d0db116f42ae657b5ac0ef7318",
        "size_bytes": 69772944,
    },
}
MODEL_IDENTITIES = {
    "config.json": {
        "sha256": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
        "size_bytes": 726,
    },
    "model.safetensors.index.json": {
        "sha256": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
        "size_bytes": 25605,
    },
    "model-00001-of-00002.safetensors": {
        "sha256": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "size_bytes": 3441185608,
    },
    "tokenizer.json": {
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "size_bytes": 11422654,
    },
    "tokenizer_config.json": {
        "sha256": "5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63",
        "size_bytes": 9675,
    },
    "special_tokens_map.json": {
        "sha256": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
        "size_bytes": 613,
    },
    "added_tokens.json": {
        "sha256": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
        "size_bytes": 707,
    },
    "vocab.json": {
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "size_bytes": 2776833,
    },
    "merges.txt": {
        "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "size_bytes": 1671853,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact(path: Path, identity: dict[str, object]) -> None:
    if path.is_symlink():
        raise ValueError(f"training artifact may not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = identity["size_bytes"]
    if path.stat().st_size != expected_size:
        raise ValueError(f"training artifact size differs from its frozen identity: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != identity["sha256"]:
        raise ValueError(f"training artifact hash differs from its frozen identity: {path}")


def candidate_output(seed: int) -> Path:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS!r}")
    return CANDIDATE_ROOT / f"router-v7-taskmaster-gate-seed{seed}"


def frozen_training_config(seed: int) -> RouterTrainingConfig:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS!r}")
    return RouterTrainingConfig(
        epochs=20,
        batch_size=128,
        learning_rate=2e-3,
        weight_decay=1e-4,
        label_smoothing=0.05,
        load_balance_weight=0.01,
        z_loss_weight=1e-4,
        patience=4,
        seed=seed,
        device="cpu",
        router_hidden_size=None,
    )


def preflight(*, seed: int, output_dir: Path) -> dict[str, Any]:
    expected_output = candidate_output(seed).resolve()
    destination = output_dir.expanduser().resolve()
    if destination != expected_output:
        raise ValueError(f"seed {seed} output must be the frozen quarantine path {expected_output}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {destination}")

    for identity in DATASET_IDENTITIES.values():
        _validate_artifact(Path(identity["path"]).resolve(), identity)
    for name, identity in ADAPTER_IDENTITIES.items():
        _validate_artifact((ADAPTER_DIR / name).resolve(), identity)
    for name, identity in MODEL_IDENTITIES.items():
        _validate_artifact((MODEL_DIR / name).resolve(), identity)

    return {
        "seed": seed,
        "output_dir": str(destination),
        "route_ids": list(ROUTE_IDS),
        "dataset_artifacts": {
            name: {
                "path": str(Path(identity["path"]).resolve()),
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }
            for name, identity in DATASET_IDENTITIES.items()
        },
        "adapter_dir": str(ADAPTER_DIR.resolve()),
        "adapter_artifacts": ADAPTER_IDENTITIES,
        "model_dir": str(MODEL_DIR.resolve()),
        "model_artifacts": MODEL_IDENTITIES,
        "test_path": None,
        "training_config": frozen_training_config(seed).to_dict(),
        "feature_contract": RouterFeatureContract(max_tokens=256).to_dict(),
    }


def train(*, seed: int, output_dir: Path | None = None) -> RouterTrainingResult:
    destination = candidate_output(seed) if output_dir is None else output_dir
    preflight(seed=seed, output_dir=destination)
    return train_router_checkpoint(
        train_path=DATASET_IDENTITIES["train"]["path"],
        validation_path=DATASET_IDENTITIES["validation"]["path"],
        test_path=None,
        model_dir=MODEL_DIR,
        adapters={TASKMASTER_ROUTE_ID: ADAPTER_DIR},
        output_dir=destination,
        route_ids=ROUTE_IDS,
        feature_contract=RouterFeatureContract(max_tokens=256),
        training=frozen_training_config(seed),
        feature_batch_size=64,
        cache_dir=FEATURE_CACHE_DIR,
        progress=lambda message: print(message, flush=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one preregistered CPU-only Taskmaster admission gate."
    )
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output_dir = candidate_output(args.seed) if args.output_dir is None else args.output_dir
    if args.preflight_only:
        receipt = preflight(seed=args.seed, output_dir=output_dir)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    result = train(seed=args.seed, output_dir=output_dir)
    print(f"Saved gate weights: {result.checkpoint_path}", flush=True)
    print(f"Saved gate contract: {result.config_path}", flush=True)
    print(
        f"Validation accuracy: {result.metrics['validation']['accuracy']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
