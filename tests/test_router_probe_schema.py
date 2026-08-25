from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EVALUATOR = WORKSPACE / "experiments" / "first-real-experts" / "evaluate_router.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("molor_router_probe_schema_test", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_probe(path: Path, purpose: str) -> None:
    row = {
        "id": "probe-001",
        "prompt": "Write a complete bounded embedded C queue implementation.",
        "category": "verified-embedded-code-v1",
        "expected_route": "verified-embedded-code-v1",
        "acceptable_routes": ["verified-embedded-code-v1"],
        "purpose": purpose,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "purpose",
    ["small_intent_probe_never_training", "blind_intent_probe_never_training"],
)
def test_probe_loader_accepts_only_explicit_evaluation_markers(
    tmp_path: Path,
    purpose: str,
) -> None:
    evaluator = _load_evaluator()
    probe = tmp_path / "probe.jsonl"
    _write_probe(probe, purpose)

    items = evaluator.load_probe(probe)

    assert len(items) == 1
    assert items[0].purpose == purpose


def test_probe_loader_rejects_training_marker(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    probe = tmp_path / "probe.jsonl"
    _write_probe(probe, "router_training")

    with pytest.raises(ValueError, match="evaluation-only marker"):
        evaluator.load_probe(probe)


def test_probe_loader_rejects_mixed_evaluation_markers(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    probe = tmp_path / "probe.jsonl"
    first = {
        "id": "probe-001",
        "prompt": "Write a complete bounded embedded C queue implementation.",
        "category": "verified-embedded-code-v1",
        "expected_route": "verified-embedded-code-v1",
        "acceptable_routes": ["verified-embedded-code-v1"],
        "purpose": "small_intent_probe_never_training",
    }
    second = {
        **first,
        "id": "probe-002",
        "prompt": "Implement a complete embedded C ring buffer with explicit bounds.",
        "purpose": "blind_intent_probe_never_training",
    }
    probe.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one consistent evaluation-only purpose"):
        evaluator.load_probe(probe)
