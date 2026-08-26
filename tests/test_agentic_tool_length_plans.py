from __future__ import annotations

import hashlib
import json
from pathlib import Path

from molor.experimental.length_tier_shards import load_plan


PLAN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "expert-rack-v5"
    / "agentic-tool-length-plans"
)
RECEIPT_PATH = PLAN_ROOT.parent / "agentic-tool-length-tiers-receipt-v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_tool_plans_pin_expected_split_rows_and_direct_sft_mode() -> None:
    expected = {
        "taskmaster-tool-test.json": 1_430,
        "taskmaster-tool-train.json": 11_626,
        "taskmaster-tool-validation.json": 1_380,
        "terminal-cli-train.json": 540,
        "terminal-cli-validation.json": 60,
    }
    observed: dict[str, int] = {}
    for path in sorted(PLAN_ROOT.glob("*.json")):
        plan = load_plan(path)
        assert len(plan.inputs) == 1
        # These are tool-use tasks, but their chat schema has no literal tool-role turn.
        assert plan.inputs[0].template_mode == "direct"
        observed[path.name] = plan.inputs[0].record_count
    assert observed == expected


def test_training_plans_never_name_held_out_inputs() -> None:
    for path in PLAN_ROOT.glob("*-train.json"):
        plan = load_plan(path)
        for shard in plan.inputs:
            lowered = shard.path.name.casefold()
            assert ".validation." not in lowered
            assert ".test." not in lowered


def test_agentic_tool_receipt_pins_plans_and_reconciles() -> None:
    receipt = json.loads(RECEIPT_PATH.read_bytes())
    pinned = {item["path"]: item["sha256"] for item in receipt["inputs"]["plans"]}
    actual = {
        f"agentic-tool-length-plans/{path.name}": _sha256(path) for path in PLAN_ROOT.glob("*.json")
    }
    assert pinned == actual
    accounting = receipt["accounting"]
    assert accounting["input_records"] == accounting["accepted_records"] == 15_036
    assert accounting["quarantined_records"] == 0
    assert sum(accounting["by_split"].values()) == accounting["input_records"]
    assert sum(accounting["by_mode_and_tier"]["direct"].values()) == accounting["input_records"]
    assert accounting["fully_reconciled"] is True
    assert accounting["zero_silent_drops"] is True
    assert receipt["preflight_rejection"]["admitted_records"] == 0
    assert receipt["preflight_rejection"]["ingestible"] is False
    assert receipt["reproducibility_check"]["passed"] is True
