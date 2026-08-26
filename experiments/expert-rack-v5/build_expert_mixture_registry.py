"""Build the content-addressed rack-v5 chat recipe and router-example bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import molor.experimental.expert_mixture_registry as registry_module
from molor.experimental.expert_mixture_registry import (
    ExpertMixtureSourcePaths,
    build_expert_mixture_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repository = Path(__file__).resolve().parents[2]
    parser.add_argument("--repository-root", type=Path, default=default_repository)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_repository / "experiments" / "expert-rack-v5" / "chat-mixtures-v1",
    )
    parser.add_argument(
        "--router-output",
        type=Path,
        default=(
            default_repository.parent
            / "datasets"
            / "processed"
            / "molor-hierarchical-router-examples-v1"
        ),
        help="external payload directory for strict prompt/route_id router JSONL files",
    )
    parser.add_argument(
        "--skip-file-verification",
        action="store_true",
        help="trust pinned shard hashes/counts (intended only for fast manifest-shape tests)",
    )
    args = parser.parse_args()

    paths = ExpertMixtureSourcePaths.workstation_defaults(args.repository_root)
    report = build_expert_mixture_bundle(
        paths,
        output_directory=args.output,
        router_output_directory=args.router_output,
        verify_files=not args.skip_file_verification,
        builder_paths=(Path(registry_module.__file__).resolve(), Path(__file__).resolve()),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
