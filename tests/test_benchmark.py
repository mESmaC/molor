from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.benchmark.models import load_suite
from molor.benchmark.runner import (
    RESULT_SCHEMA,
    RunOptions,
    _build_workload_contract,
    _cell_id,
    _fingerprint_workload,
    compare_runs,
    summarize_run,
)
from molor.benchmark.scoring import SCORER_VERSION, score_response


SUITE = Path("benchmarks/suites/molor-core-v1.json")


def _run_fixture(suite, run_id: str, stack: str, responses: dict[str, str]):
    case_lookup = {case.case_id: case for case in suite.cases}
    cases = [case_lookup[case_id] for case_id in responses]
    contract = _build_workload_contract(
        suite,
        cases,
        RunOptions(label=run_id, profile="fixture"),
    )
    workload_fingerprint = _fingerprint_workload(contract)
    evaluation_fingerprint = hashlib.sha256(
        f"{suite.fingerprint}:{SCORER_VERSION}".encode()
    ).hexdigest()
    manifest = {
        "run_id": run_id,
        "label": run_id,
        "profile": "fixture",
        "backend": {},
        "suite_fingerprint": suite.fingerprint,
        "scorer_version": SCORER_VERSION,
        "evaluation_fingerprint": evaluation_fingerprint,
        "workload_fingerprint": workload_fingerprint,
        "workload_contract": contract,
        "stack_fingerprint": stack,
    }
    records = []
    for cell in contract["cells"]:
        case_id = cell["case_id"]
        response = responses[case_id]
        scored = score_response(response, case_lookup[case_id].scorer)
        record = {
            "schema_version": RESULT_SCHEMA,
            "run_id": run_id,
            "suite_fingerprint": suite.fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "workload_fingerprint": workload_fingerprint,
            "stack_fingerprint": stack,
            "case_id": case_id,
            "repetition": cell["repetition"],
            "request": cell["request"],
            "response": response,
            **scored.to_dict(),
            "telemetry": {},
            "routing_decision": None,
            "error": None,
        }
        record["cell_id"] = _cell_id(
            suite_fingerprint=suite.fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
            workload_fingerprint=workload_fingerprint,
            stack_fingerprint=stack,
            case_id=case_id,
            repetition=cell["repetition"],
            request=cell["request"],
        )
        records.append(record)
    return summarize_run(suite, manifest, records), records


def test_bundled_suite_is_versioned_and_has_three_profiles() -> None:
    suite = load_suite(SUITE)
    assert len(suite.cases) == 64
    assert len(suite.select(profile="smoke")) == 10
    assert len(suite.select(profile="quick")) == 30
    assert len(suite.select(profile="full")) == 64


def test_deterministic_scorers_extract_requested_answer_forms() -> None:
    assert score_response("Answer: B", {"type": "choice", "answer": "B"}).score == 1
    assert score_response("Final answer: 3/8", {"type": "numeric", "answer": 0.375}).score == 1
    assert score_response(".375", {"type": "numeric", "answer": 0.375}).score == 1
    assert score_response("2000%", {"type": "numeric", "answer": 20}).score == 0
    assert (
        score_response(
            "37.5%", {"type": "numeric", "answer": 0.375, "percent_mode": "fraction"}
        ).score
        == 1
    )
    assert (
        score_response(
            '```json\n{"status":"ok","count":3}\n```',
            {"type": "json", "expected": {"status": "ok", "count": 3}},
        ).score
        == 1
    )
    assert (
        score_response(
            "Get-ChildItem -LiteralPath $targetPath",
            {"type": "command", "answers": ["Get-ChildItem -LiteralPath $targetPath"]},
        ).score
        == 1
    )
    assert (
        score_response(
            "get-childitem -literalpath $targetpath",
            {"type": "command", "answers": ["Get-ChildItem -LiteralPath $targetPath"]},
        ).score
        == 0
    )


def test_strict_json_and_terminal_only_contracts() -> None:
    expected = {"status": "ok", "count": 3}
    spec = {
        "type": "json",
        "expected": expected,
        "exact": True,
        "allow_fence": False,
        "minified": True,
    }
    assert score_response('{"status":"ok","count":3}', spec).score == 1
    assert score_response('{"status":"ok","count":3,"extra":1}', spec).score == 0
    assert score_response('{"status": "ok", "count": 3}', spec).score == 0
    assert score_response('```json\n{"status":"ok","count":3}\n```', spec).score == 0

    command_spec = {"type": "command", "answers": ["rg -c ERROR app.log"]}
    assert score_response("rg -c ERROR app.log", command_spec).score == 1
    assert score_response("Command: rg -c ERROR app.log", command_spec).score == 0
    assert score_response("`rg -c ERROR app.log`", command_spec).score == 0


def test_summary_uses_category_macro_weights_not_case_count() -> None:
    suite = load_suite(SUITE)
    summary, _ = _run_fixture(suite, "run-a", "stack-a", {"G01": "B", "E01": "0"})
    assert summary["quality_index"] == pytest.approx(1000 * 0.10 / (0.10 + 0.16))


def test_compare_reports_paired_delta(tmp_path: Path) -> None:
    suite = load_suite(SUITE)
    paths = []
    for name, response in (("base", "A"), ("candidate", "B")):
        directory = tmp_path / name
        directory.mkdir()
        summary, records = _run_fixture(suite, name, f"stack-{name}", {"G01": response})
        (directory / "summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        paths.append(directory)
    comparison = compare_runs(paths[0], paths[1])
    assert comparison["quality_index_delta"] == 1000
    assert comparison["paired"]["wins"] == 1
