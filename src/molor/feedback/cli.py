"""Export the local annotation notebook without opening the desktop app."""

from __future__ import annotations

import argparse
from pathlib import Path

from molor.feedback import DEFAULT_EXPORT_DIR, DEFAULT_FEEDBACK_DB, FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local molor rewards and preferences.")
    parser.add_argument("--db", type=Path, default=DEFAULT_FEEDBACK_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--min-reward", type=float, default=1.0)
    args = parser.parse_args()

    with FeedbackStore(args.db) as store:
        results = (
            store.export_reward_jsonl(args.output_dir / "rewards.jsonl"),
            store.export_positive_sft_jsonl(
                args.output_dir / "positive_sft.jsonl",
                min_reward=args.min_reward,
            ),
            store.export_preference_jsonl(args.output_dir / "preferences.jsonl"),
        )
    for result in results:
        print(f"{result.rows:>5} rows  {result.path}")


if __name__ == "__main__":  # pragma: no cover
    main()
