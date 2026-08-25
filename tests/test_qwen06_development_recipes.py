from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from molor.training.recipe import load_training_recipe


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEVELOPMENT_ROOT = _REPOSITORY_ROOT / "experiments" / "qwen3-0.6b-development"
_PROMOTION_ROOT = _REPOSITORY_ROOT / "experiments" / "first-real-experts"
_RECIPE_NAMES = (
    "embedded-knowledge-v1-recipe.json",
    "ocb-can-verified-v1-recipe.json",
    "software-architecture-v1-recipe.json",
    "verified-embedded-code-v1-recipe.json",
)


def _comparable_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparable = deepcopy(payload)
    comparable.pop("model_id")
    comparable.pop("output_dir")
    for expert in comparable["experts"]:
        expert["dataset"] = "<same prepared shard>"
    return comparable


def test_06_recipes_preserve_training_settings_but_isolate_model_and_output() -> None:
    for name in _RECIPE_NAMES:
        development_path = _DEVELOPMENT_ROOT / name
        promotion_path = _PROMOTION_ROOT / name
        recipe = load_training_recipe(development_path)

        assert Path(recipe.model_id).name == "Qwen3-0.6B"
        assert recipe.output_dir.name == "qwen3-0.6b-corpus-experts"
        assert all(expert.dataset.is_file() for expert in recipe.experts)
        assert _comparable_payload(development_path) == _comparable_payload(promotion_path)


def test_reasoning_recipe_uses_thinking_compatible_bounded_speed_profile() -> None:
    recipe = load_training_recipe(_DEVELOPMENT_ROOT / "reasoning-trace-v1-recipe.json")

    assert Path(recipe.model_id).name == "Qwen3-0.6B"
    assert recipe.experts[0].name == "reasoning-trace-v1"
    assert recipe.max_length == 1024
    assert recipe.micro_batch_size == 3
    assert recipe.gradient_accumulation_steps == 3
    assert recipe.gradient_checkpointing is True
    assert recipe.autocast_adapter_compute is True
    assert recipe.batching_strategy == "pretokenized_group_sorted"
    assert recipe.fused_adamw is False
    assert recipe.chat_template_kwargs == {"enable_thinking": True}
