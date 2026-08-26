"""Plan, preflight, or materialize the canonical 0.6B rack-v5 fast wave."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molor.training.fast_expert_wave import build_fast_wave_plan, materialize_fast_wave


REPOSITORY = Path(__file__).resolve().parents[2]
RACK = Path(__file__).resolve().parent


def _assignments(
    values: list[str], *, pair: bool = False, integer: bool = False
) -> dict[object, float | int]:
    result: dict[object, float | int] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"override must be SELECTOR=VALUE: {raw!r}")
        selector, raw_value = raw.rsplit("=", 1)
        key: object
        if pair:
            if ":" not in selector:
                raise ValueError(f"leaf-stream selector must be LEAF:STREAM: {selector!r}")
            key = tuple(selector.split(":", 1))
            value: float | int = float(raw_value)
        else:
            key = selector
            if integer:
                if not raw_value.isdecimal() or int(raw_value) < 1:
                    raise ValueError(
                        f"leaf target must be an exact positive integer: {raw_value!r}"
                    )
                value = int(raw_value)
            else:
                value = float(raw_value)
        if key in result:
            raise ValueError(f"duplicate override: {selector!r}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=RACK / "chat-mixtures-v1")
    parser.add_argument("--taxonomy", type=Path, default=RACK / "expert-taxonomy-v1.json")
    parser.add_argument("--model", type=Path, default=REPOSITORY / "models" / "Qwen3-0.6B")
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=REPOSITORY / "checkpoints" / "qwen3-0.6b-expert-rack-v5-fast-v1",
    )
    parser.add_argument("--target-per-leaf", type=int, default=1024)
    parser.add_argument("--leaf-target", action="append", default=[], metavar="LEAF=ROWS")
    parser.add_argument("--stream-weight", action="append", default=[], metavar="STREAM=WEIGHT")
    parser.add_argument(
        "--leaf-stream-weight", action="append", default=[], metavar="LEAF:STREAM=WEIGHT"
    )
    parser.add_argument("--seed", default="molor-rack-v5-fast-wave/seed-1")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only and args.dry_run:
        parser.error("--plan-only and --dry-run are mutually exclusive")
    try:
        leaf_targets = {
            str(key): int(value)
            for key, value in _assignments(args.leaf_target, integer=True).items()
        }
        plan = build_fast_wave_plan(
            bundle_directory=args.bundle,
            taxonomy_path=args.taxonomy,
            model_id=args.model,
            adapter_output_directory=args.adapter_output,
            target_records_per_leaf=args.target_per_leaf,
            leaf_targets=leaf_targets,
            stream_weights={
                str(key): float(value)
                for key, value in _assignments(args.stream_weight).items()
            },
            leaf_stream_weights={
                key: float(value)
                for key, value in _assignments(args.leaf_stream_weight, pair=True).items()
            },
            seed=args.seed,
        )
        result = (
            plan.to_dict()
            if args.plan_only
            else materialize_fast_wave(plan, output_directory=args.output, dry_run=args.dry_run)
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
