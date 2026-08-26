from __future__ import annotations

import hashlib
import json
from pathlib import Path

from molor.experimental.length_tier_shards import load_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = REPOSITORY_ROOT / "experiments" / "expert-rack-v5" / "business-finance-length-plans"
RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "expert-rack-v5"
    / "business-finance-length-tiers-receipt-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_business_finance_length_plans_are_leaf_and_split_isolated() -> None:
    expected = {
        "business-finance-test",
        "business-finance-train",
        "business-finance-validation",
        "investing-markets-test",
        "investing-markets-train",
        "investing-markets-validation",
    }
    plan_paths = sorted(PLAN_ROOT.glob("*.json"))
    assert {path.stem for path in plan_paths} == expected

    mode_totals = {"direct": 0, "trace": 0}
    split_totals = {"train": 0, "validation": 0, "test": 0}
    for plan_path in plan_paths:
        plan = load_plan(plan_path)
        split = next(name for name in split_totals if plan_path.stem.endswith(f"-{name}"))
        leaf = plan_path.stem.removesuffix(f"-{split}")
        assert {descriptor.template_mode for descriptor in plan.inputs} == {"direct", "trace"}
        assert len(plan.inputs) == 2
        for descriptor in plan.inputs:
            normalized = descriptor.path.as_posix()
            assert f"/{leaf}/{descriptor.template_mode}/{split}.jsonl" in normalized
            mode_totals[descriptor.template_mode] += descriptor.record_count
            split_totals[split] += descriptor.record_count

    assert mode_totals == {"direct": 24_760, "trace": 16_482}
    assert split_totals == {"train": 32_220, "validation": 4_252, "test": 4_770}


def test_business_finance_length_receipt_pins_every_plan_and_reconciles() -> None:
    receipt = json.loads(RECEIPT_PATH.read_bytes())
    pinned = {item["path"]: item["sha256"] for item in receipt["inputs"]["plans"]}
    actual = {
        f"business-finance-length-plans/{path.name}": _sha256(path)
        for path in PLAN_ROOT.glob("*.json")
    }
    assert pinned == actual

    accounting = receipt["accounting"]
    assert accounting["input_emissions"] == accounting["accepted_emissions"]
    assert accounting["quarantined_emissions"] == 0
    assert sum(accounting["by_split"].values()) == accounting["input_emissions"]
    assert accounting["fully_reconciled"] is True
    assert accounting["zero_silent_drops"] is True
    assert receipt["reproducibility_check"] == {
        "independent_rebuild_files": 102,
        "byte_identical_files": 102,
        "mismatches": 0,
        "passed": True,
    }
