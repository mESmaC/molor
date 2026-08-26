"""Launch the pinned Qwen3-0.6B rack-v5 benchmark campaign."""

from __future__ import annotations

from pathlib import Path
import sys

from molor.benchmark.rack_launcher import main as launcher_main


DEFAULT_CONFIG = Path(__file__).with_name("benchmark-rack-v5-fast-r8-v1.json")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return launcher_main(("--config", str(DEFAULT_CONFIG), *arguments))


if __name__ == "__main__":
    raise SystemExit(main())
