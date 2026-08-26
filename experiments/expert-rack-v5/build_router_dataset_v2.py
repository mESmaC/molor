"""Build the rack-v5 v2 router dataset from safe expert shards and a curated curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import molor.experimental.router_dataset_v2 as router_v2_module
from molor.experimental.expert_mixture_registry import (
    ExpertMixtureSourcePaths,
    collect_expert_data,
)
from molor.experimental.router_dataset_v2 import build_router_dataset_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    parser.add_argument("--repository-root", type=Path, default=repository)
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=repository / "experiments" / "expert-rack-v5" / "router-curriculum-v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository.parent / "datasets" / "processed" / "molor-hierarchical-router-examples-v2"
        ),
    )
    parser.add_argument(
        "--skip-file-verification",
        action="store_true",
        help="trust pinned expert shard hashes/counts (only for manifest-shape development)",
    )
    args = parser.parse_args()

    paths = ExpertMixtureSourcePaths.workstation_defaults(args.repository_root)
    collected = collect_expert_data(paths, verify_files=not args.skip_file_verification)
    report = build_router_dataset_v2(
        collected,
        curriculum_path=args.curriculum,
        output_directory=args.output,
        builder_paths=(Path(router_v2_module.__file__).resolve(), Path(__file__).resolve()),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
