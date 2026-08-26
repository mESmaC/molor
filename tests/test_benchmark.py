from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.benchmark import cli as benchmark_cli
from molor.benchmark import matrix as benchmark_matrix
from molor.benchmark.backends import GenerationResponse
from molor.benchmark.models import canonical_json, load_suite
from molor.benchmark.routing import load_route_evaluation_contract
from molor.benchmark.runner import (
    BenchmarkRunner,
    RESULT_SCHEMA,
    RunOptions,
    _build_workload_contract,
    _cell_id,
    _fingerprint_workload,
    compare_runs,
    summarize_run,
)
from molor.benchmark.scoring import SCORER_VERSION, score_response
from molor.experimental.expert_taxonomy import load_expert_taxonomy


SUITE = Path("benchmarks/suites/molor-core-v1.json")
TAXONOMY = Path("experiments/expert-rack-v5/expert-taxonomy-v1.json")
ROUTE_CONTRACT = Path(
    "experiments/expert-rack-v5/molor-core-v1-rack-v5-route-contract-v1.json"
)


def _run_fixture(
    suite,
    run_id: str,
    stack: str,
    responses: dict[str, str],
    *,
    selected_route: str | None = None,
    route_contract=None,
):
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
        "routing_evaluation": route_contract.provenance() if route_contract else None,
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
            "routing_decision": (
                {"selected_routes": [selected_route]} if selected_route is not None else None
            ),
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
    return summarize_run(suite, manifest, records, route_contract=route_contract), records


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


def test_v5_route_contract_repairs_routing_only_without_moving_quality_provenance() -> None:
    suite = load_suite(SUITE)
    taxonomy = load_expert_taxonomy(TAXONOMY)
    route_contract = load_route_evaluation_contract(
        ROUTE_CONTRACT,
        suite=suite,
        taxonomy=taxonomy,
    )

    plain, _ = _run_fixture(
        suite,
        "route-fixture",
        "stack-route",
        {"A01": "fixture"},
        selected_route="software-architecture-design",
    )
    routed, _ = _run_fixture(
        suite,
        "route-fixture",
        "stack-route",
        {"A01": "fixture"},
        selected_route="software-architecture-design",
        route_contract=route_contract,
    )

    assert route_contract.content_sha256 == (
        "757e933206bec154502227b950b0c5761d0257891afee7e3f9c590068976f196"
    )
    assert len(route_contract.case_routes) == 64
    assert route_contract.canonicalize_selected("software-architecture-v1") == (
        "software-architecture-design"
    )
    assert plain["routing"]["acceptable_top1"] == 0
    assert routed["routing"]["acceptable_top1"] == 1
    assert routed["routing"]["evaluation"]["route_contract_content_sha256"] == (
        route_contract.content_sha256
    )
    for field in (
        "suite_fingerprint",
        "evaluation_fingerprint",
        "workload_fingerprint",
        "result_fingerprint",
        "quality_index",
        "quality_percent",
    ):
        assert routed[field] == plain[field]


def test_route_contract_fails_closed_on_unresolved_non_base_route(tmp_path: Path) -> None:
    suite = load_suite(SUITE)
    taxonomy = load_expert_taxonomy(TAXONOMY)
    payload = json.loads(ROUTE_CONTRACT.read_text(encoding="utf-8"))
    payload["case_routes"]["G01"] = ["unresolved-specialist"]
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = hashlib.sha256(
        canonical_json(semantic).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "invalid-route-contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unresolved non-base route"):
        load_route_evaluation_contract(path, suite=suite, taxonomy=taxonomy)


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


def test_matrix_cli_thinking_override_is_explicit_and_tristate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool | None] = []

    def capture_matrix(*_args, **kwargs) -> None:
        observed.append(kwargs["enable_thinking_override"])

    monkeypatch.setattr(benchmark_cli, "run_matrix", capture_matrix)
    for flag in ((), ("--thinking",), ("--no-thinking",)):
        assert benchmark_cli.main(["matrix", "--label", "fixture", *flag]) == 0

    assert observed == [None, True, False]


def test_thinking_on_has_a_distinct_workload_fingerprint_without_moving_frozen_off() -> None:
    suite = load_suite(SUITE)
    cases = suite.select(profile="full")
    frozen_off = _build_workload_contract(
        suite,
        cases,
        RunOptions(label="ignored", profile="full"),
    )
    thinking_on = _build_workload_contract(
        suite,
        cases,
        RunOptions(label="ignored", profile="full", enable_thinking_override=True),
    )

    assert frozen_off["enable_thinking_override"] is None
    assert all(cell["request"]["enable_thinking"] is False for cell in frozen_off["cells"])
    assert _fingerprint_workload(frozen_off) == (
        "1184f848aa22c77f52c57b7017909d7605d4e74cc7eaaa004eb75ca87244a929"
    )
    assert all(cell["request"]["enable_thinking"] is True for cell in thinking_on["cells"])
    assert _fingerprint_workload(thinking_on) != _fingerprint_workload(frozen_off)


def test_matrix_propagates_thinking_override_to_every_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_suite(SUITE)
    observed_options: list[RunOptions] = []

    class FakeTeacherBackend:
        def __init__(self, _endpoint: str, **_kwargs) -> None:
            pass

        def identity(self) -> dict[str, object]:
            return {"backend": "fixture", "experts": []}

    class FakeRunner:
        def __init__(
            self,
            runner_suite,
            _backend,
            *,
            output_root: Path,
            options: RunOptions,
            run_directory: Path,
            route_contract=None,
        ) -> None:
            del output_root
            del route_contract
            self.suite = runner_suite
            self.options = options
            self.summary_path = run_directory / "summary.json"
            observed_options.append(options)

        def run(self, *, baseline_summary: Path | None = None) -> dict[str, object]:
            del baseline_summary
            cases = self.suite.select(
                profile=self.options.profile,
                categories=self.options.categories,
                limit=self.options.limit,
            )
            workload = _build_workload_contract(self.suite, cases, self.options)
            summary = {
                "quality_index": 0.0,
                "categories": {},
                "evaluation_fingerprint": "fixture-evaluator",
                "workload_fingerprint": _fingerprint_workload(workload),
                "performance": {},
                "routing": {},
                "result_cells": [],
            }
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return summary

    monkeypatch.setattr(benchmark_matrix, "TeacherApiBackend", FakeTeacherBackend)
    monkeypatch.setattr(benchmark_matrix, "BenchmarkRunner", FakeRunner)
    matrix_root = tmp_path / "matrix"
    summary_path = benchmark_matrix.run_matrix(
        suite,
        endpoint="http://127.0.0.1:8765",
        label="thinking-fixture",
        output_root=tmp_path,
        profile="smoke",
        categories=(),
        limit=1,
        repetitions=1,
        experts=(),
        include_top2=False,
        route_temperature=1.0,
        confidence_threshold=0.0,
        null_bias=0.0,
        timeout=1.0,
        matrix_directory=matrix_root,
        enable_thinking_override=True,
    )

    assert [options.enable_thinking_override for options in observed_options] == [True, True]
    manifest = json.loads((matrix_root / "matrix.json").read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert manifest["enable_thinking_override"] is True
    assert manifest["effective_enable_thinking"] is True
    assert summary["enable_thinking_override"] is True
    assert summary["effective_enable_thinking"] is True

    manifest["matrix_id"] = "preserved-matrix-id"
    manifest["created_at"] = "2000-01-01T00:00:00Z"
    manifest_path = matrix_root / "matrix.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    benchmark_matrix.run_matrix(
        suite,
        endpoint="http://127.0.0.1:8765",
        label="thinking-fixture",
        output_root=tmp_path,
        profile="smoke",
        categories=(),
        limit=1,
        repetitions=1,
        experts=(),
        include_top2=False,
        route_temperature=1.0,
        confidence_threshold=0.0,
        null_bias=0.0,
        timeout=1.0,
        matrix_directory=matrix_root,
        enable_thinking_override=True,
    )
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed["matrix_id"] == "preserved-matrix-id"
    assert resumed["created_at"] == "2000-01-01T00:00:00Z"
    assert len(observed_options) == 4

    before_drift = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="stable contract changed"):
        benchmark_matrix.run_matrix(
            suite,
            endpoint="http://127.0.0.1:8765",
            label="thinking-fixture",
            output_root=tmp_path,
            profile="smoke",
            categories=(),
            limit=1,
            repetitions=1,
            experts=(),
            include_top2=False,
            route_temperature=0.5,
            confidence_threshold=0.0,
            null_bias=0.0,
            timeout=1.0,
            matrix_directory=matrix_root,
            enable_thinking_override=True,
        )
    assert manifest_path.read_bytes() == before_drift
    assert len(observed_options) == 4


def test_runner_sends_effective_thinking_mode_to_generation(tmp_path: Path) -> None:
    suite = load_suite(SUITE)
    case = suite.select(profile="smoke", limit=1)[0]

    class CaptureBackend:
        request = None

        def identity(self) -> dict[str, object]:
            return {"backend": "fixture"}

        def generate(self, request):
            self.request = request
            return GenerationResponse(
                text="fixture response",
                elapsed_seconds=0.01,
                time_to_first_token_seconds=None,
                prompt_tokens=None,
                completion_tokens=None,
                tokens_per_second=None,
                finish_reason="stop",
                routing_decision=None,
                generation_id=None,
                provider_metadata={},
            )

    backend = CaptureBackend()
    options = RunOptions(
        label="thinking-fixture",
        profile="smoke",
        warmup=False,
        enable_thinking_override=True,
    )
    runner = BenchmarkRunner(suite, backend, output_root=tmp_path, options=options)
    runner.workload_fingerprint = _fingerprint_workload(
        _build_workload_contract(suite, [case], options)
    )
    result = runner._run_case(case, 0)

    assert backend.request is not None
    assert backend.request.enable_thinking is True
    assert result["request"]["enable_thinking"] is True
