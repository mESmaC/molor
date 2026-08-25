from pathlib import Path

import pytest

from molor.training import ExpertRecipe, TrainingRecipe


def _recipe(**overrides: object) -> TrainingRecipe:
    values: dict[str, object] = {
        "model_id": "model",
        "output_dir": Path("output"),
        "experts": (ExpertRecipe("expert", Path("train.jsonl")),),
        "target_modules": ("q_proj",),
    }
    values.update(overrides)
    return TrainingRecipe(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "strategy",
    ["online", "pretokenized", "pretokenized_group_sorted"],
)
def test_training_recipe_accepts_supported_batching_strategies(strategy: str) -> None:
    assert _recipe(batching_strategy=strategy).batching_strategy == strategy


def test_training_recipe_rejects_unknown_batching_strategy() -> None:
    with pytest.raises(ValueError, match="batching_strategy"):
        _recipe(batching_strategy="global_sorted")


def test_adapter_autocast_is_bfloat16_only() -> None:
    with pytest.raises(ValueError, match="requires dtype='bfloat16'"):
        _recipe(dtype="float16", autocast_adapter_compute=True)


@pytest.mark.parametrize("field", ["autocast_adapter_compute", "fused_adamw"])
def test_training_speed_switches_require_booleans(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _recipe(**{field: 1})
