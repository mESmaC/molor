from __future__ import annotations

import json
from pathlib import Path

import pytest

from molor.benchmark.cli import _parser as benchmark_parser
from molor.benchmark.rack_launcher import (
    build_launch_steps,
    load_rack_benchmark_config,
    main,
    render_powershell,
    validate_server_health,
)
from molor.teacher.cli import _parser as teacher_parser


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = (
    REPOSITORY
    / "experiments"
    / "expert-rack-v5"
    / "benchmark-rack-v5-fast-r8-v1.json"
)


def _config():
    return load_rack_benchmark_config(CONFIG)


def test_rack_v5_plan_pins_the_frozen_suite_and_complete_taxonomy() -> None:
    config = _config()

    assert config.suite.suite_id == "molor-core-v1"
    assert config.suite.version == "1.0.0"
    assert config.suite.fingerprint == (
        "0f2be48bcdec9a29d412f4c56f1c671fc276af73df572d78046c841f7afc4de5"
    )
    assert config.taxonomy_fingerprint == (
        "278748ac0cc029963d640460d36bb27b2cb1b8893b14e9e29d549b17c0d36dd1"
    )
    assert len(config.experts) == 25
    assert config.expert_rank == 8
    assert config.route_contract_fingerprint == (
        "757e933206bec154502227b950b0c5761d0257891afee7e3f9c590068976f196"
    )


def test_launch_plan_separates_all_expert_smoke_from_six_full_policies() -> None:
    config = _config()
    steps = build_launch_steps(config)

    assert len(steps) == 8
    assert steps[0].step_id == "serve"
    assert steps[0].argv.count("--expert") == 25
    assert "--eager-load" in steps[0].argv
    assert str(config.router_checkpoint) in steps[0].argv

    smoke = steps[1]
    assert smoke.step_id == "smoke-all-experts-thinking-off"
    assert smoke.argv[2] == "molor.benchmark.cli"
    assert smoke.argv[3] == "matrix"
    assert smoke.argv.count("--expert") == 25
    assert "--profile" in smoke.argv
    assert smoke.argv[smoke.argv.index("--profile") + 1] == "smoke"
    assert "--no-thinking" in smoke.argv
    assert smoke.argv[smoke.argv.index("--route-contract") + 1] == str(
        config.route_contract_path
    )
    assert smoke.argv[smoke.argv.index("--route-taxonomy") + 1] == str(
        config.taxonomy_path
    )

    full = steps[2:]
    assert tuple(step.step_id for step in full) == (
        "full-thinking-off-base",
        "full-thinking-off-auto-top1",
        "full-thinking-off-auto-top2",
        "full-thinking-on-base",
        "full-thinking-on-auto-top1",
        "full-thinking-on-auto-top2",
    )
    assert all("--expert" not in step.argv for step in full)
    assert all(step.argv[3] == "run" for step in full)
    assert all(step.argv[step.argv.index("--profile") + 1] == "full" for step in full)
    assert all("--route-contract" in step.argv for step in full)
    assert all("--route-taxonomy" in step.argv for step in full)
    assert "--baseline" not in full[0].argv
    assert "--baseline" in full[1].argv
    assert "--baseline" in full[2].argv
    assert "--baseline" not in full[3].argv
    assert "--baseline" in full[4].argv
    assert "--baseline" in full[5].argv
    assert full[2].argv[full[2].argv.index("--route-top-k") + 1] == "2"
    assert full[5].argv[full[5].argv.index("--route-top-k") + 1] == "2"

    teacher_args = teacher_parser().parse_args(list(steps[0].argv[3:]))
    assert len(teacher_args.expert) == 25
    for step in steps[1:]:
        parsed = benchmark_parser().parse_args(list(step.argv[3:]))
        assert parsed.command in {"matrix", "run"}


def test_server_health_must_match_every_pinned_artifact() -> None:
    config = _config()
    payload = {
        "status": "ok",
        "protocol_version": "molor.teacher-api.v1",
        "model_loaded": True,
        "model_id": str(config.model_directory),
        "experts": list(config.experts),
        "configured_expert_paths": [str(path) for path in config.expert_paths],
        "router_checkpoint": str(config.router_checkpoint),
        "router_loaded": True,
    }

    receipt = validate_server_health(config, payload)
    assert receipt["status"] == "ready"
    assert receipt["expert_count"] == 25

    payload["experts"] = [*config.experts[:-1], "wrong-leaf"]
    with pytest.raises(RuntimeError, match="expert axis"):
        validate_server_health(config, payload)


def test_config_parser_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["repository_root"] = str(REPOSITORY)
    rendered = json.dumps(payload, indent=2)
    rendered = rendered.replace(
        "{",
        '{\n  "schema_version": "molor.routed-rack-benchmark-launch.v1",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(rendered, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        load_rack_benchmark_config(path)


def test_powershell_rendering_quotes_every_argument() -> None:
    step = build_launch_steps(_config())[0]
    rendered = render_powershell(step)

    assert rendered.startswith("& '")
    assert "'--eager-load'" in rendered
    assert rendered.count("'--expert'") == 25


def test_plan_prints_preflight_and_live_health_gates(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(("--config", str(CONFIG), "plan")) == 0

    rendered = capsys.readouterr().out
    assert "'preflight'" in rendered
    assert "'verify-server'" in rendered
    assert "# smoke-all-experts-thinking-off" in rendered
    assert "# full-thinking-on-auto-top2" in rendered
