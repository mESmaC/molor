from __future__ import annotations

import json
from pathlib import Path

from molor.feedback import FeedbackStore, GenerationCreate


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_feedback_round_trip_and_training_exports(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite3"
    messages = ({"role": "user", "content": "Explain a cache."},)
    with FeedbackStore(database) as store:
        chosen = store.record_generation(
            GenerationCreate(
                conversation_id="chat-1",
                model_id="Qwen3-1.7B",
                messages=messages,
                response="A cache keeps likely reusable data nearby.",
                route={"precision": 1.0},
                settings={"temperature": 0.0, "max_new_tokens": 64},
                telemetry={"completion_tokens": 9, "tokens_per_second": 42.0},
                enable_thinking=False,
                reasoning_preset="direct",
                context_policy={"budget_tokens": 2048},
                cache_policy={"kv_mode": "dynamic"},
            )
        )
        rejected = store.record_generation(
            GenerationCreate(
                conversation_id="chat-1",
                model_id="Qwen3-1.7B",
                messages=messages,
                response="A cache is a cache.",
                route={"base": 1.0},
            )
        )
        store.grade_generation(
            chosen,
            reward=2,
            dimensions={"correctness": 2, "style": 1},
            notes="Useful and direct.",
        )
        store.grade_generation(
            rejected,
            reward=-2,
            corrected_answer="A cache stores recently used data in faster memory.",
        )
        store.record_preference(chosen, rejected, notes="Clear winner")

        raw = store.export_reward_jsonl(tmp_path / "rewards.jsonl")
        sft = store.export_positive_sft_jsonl(tmp_path / "sft.jsonl", min_reward=1)
        pairs = store.export_preference_jsonl(tmp_path / "pairs.jsonl")

        assert (raw.rows, sft.rows, pairs.rows) == (2, 2, 1)
        assert read_jsonl(sft.path) == [
            {
                "messages": [
                    *messages,
                    {
                        "role": "assistant",
                        "content": "A cache keeps likely reusable data nearby.",
                    },
                ]
            },
            {
                "messages": [
                    *messages,
                    {
                        "role": "assistant",
                        "content": "A cache stores recently used data in faster memory.",
                    },
                ]
            },
        ]
        assert read_jsonl(pairs.path)[0]["chosen"] == ("A cache keeps likely reusable data nearby.")

    with FeedbackStore(database) as reopened:
        assert [record.generation_id for record in reopened.list_generations(limit=2)] == [
            rejected,
            chosen,
        ]
