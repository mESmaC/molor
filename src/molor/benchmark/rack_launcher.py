"""Declarative launch planning for one content-pinned routed-expert benchmark rack.

This module deliberately keeps service startup and benchmark execution behind explicit
subcommands.  Loading a plan or printing commands performs no model load and uses no GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from molor.benchmark.models import load_suite
from molor.benchmark.routing import load_route_evaluation_contract
from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.model_profiles import complete_local_model
from molor.training.hierarchical_router import HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA


CONFIG_SCHEMA = "molor.routed-rack-benchmark-launch.v1"
_CAMPAIGN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_EXPERT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains unsupported constant {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = expected.difference(value)
    unknown = set(value).difference(expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields differ from the schema "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
        )


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a trimmed non-empty string")
    return value


def _repo_relative(root: Path, value: object, label: str) -> Path:
    raw = Path(_nonempty_string(value, label))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must be a repository-relative path without '..'")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository root") from exc
    return resolved


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class RouterPolicyPin:
    temperature: float
    confidence_threshold: float
    null_bias: float


@dataclass(frozen=True, slots=True)
class SuitePin:
    path: Path
    suite_id: str
    version: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RackBenchmarkConfig:
    source: Path
    repository_root: Path
    campaign_id: str
    model_directory: Path
    adapter_root: Path
    router_checkpoint: Path
    taxonomy_path: Path
    taxonomy_id: str
    taxonomy_revision: str
    taxonomy_fingerprint: str
    suite: SuitePin
    route_contract_path: Path
    route_contract_fingerprint: str
    output_root: Path
    feedback_database: Path
    endpoint: str
    port: int
    expert_rank: int
    experts: tuple[str, ...]
    router_policy: RouterPolicyPin
    timeout_seconds: float
    repetitions: int

    @property
    def expert_paths(self) -> tuple[Path, ...]:
        return tuple(self.adapter_root / expert for expert in self.experts)

    @property
    def smoke_matrix_directory(self) -> Path:
        return self.output_root / "matrices" / f"{self.campaign_id}-smoke-thinking-off"

    def full_directory(self, thinking: bool) -> Path:
        state = "thinking-on" if thinking else "thinking-off"
        return self.output_root / "full" / state


@dataclass(frozen=True, slots=True)
class LaunchStep:
    step_id: str
    argv: tuple[str, ...]
    requires_server: bool


def load_rack_benchmark_config(path: str | Path) -> RackBenchmarkConfig:
    source = Path(path).expanduser().resolve()
    payload = _strict_json_object(source, "rack benchmark config")
    expected = {
        "schema_version",
        "repository_root",
        "campaign_id",
        "model_directory",
        "adapter_root",
        "router_checkpoint",
        "taxonomy",
        "suite",
        "route_contract",
        "output_root",
        "feedback_database",
        "endpoint",
        "port",
        "expert_rank",
        "experts",
        "router_policy",
        "timeout_seconds",
        "repetitions",
    }
    _exact_keys(payload, expected, "rack benchmark config")
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError(f"unsupported rack benchmark schema: {payload['schema_version']!r}")

    raw_root = Path(_nonempty_string(payload["repository_root"], "repository_root"))
    repository_root = (
        raw_root.resolve() if raw_root.is_absolute() else (source.parent / raw_root).resolve()
    )
    if not repository_root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {repository_root}")
    campaign_id = _nonempty_string(payload["campaign_id"], "campaign_id")
    if _CAMPAIGN_PATTERN.fullmatch(campaign_id) is None:
        raise ValueError("campaign_id must be a bounded filesystem-safe slug")

    raw_taxonomy = payload["taxonomy"]
    if not isinstance(raw_taxonomy, Mapping):
        raise ValueError("taxonomy must be an object")
    _exact_keys(
        raw_taxonomy,
        {"path", "taxonomy_id", "revision", "fingerprint"},
        "taxonomy",
    )
    taxonomy_path = _repo_relative(repository_root, raw_taxonomy["path"], "taxonomy.path")
    taxonomy = load_expert_taxonomy(taxonomy_path)
    taxonomy_id = _nonempty_string(raw_taxonomy["taxonomy_id"], "taxonomy.taxonomy_id")
    taxonomy_revision = _nonempty_string(raw_taxonomy["revision"], "taxonomy.revision")
    taxonomy_fingerprint = _nonempty_string(
        raw_taxonomy["fingerprint"], "taxonomy.fingerprint"
    )
    if (
        taxonomy.taxonomy_id != taxonomy_id
        or taxonomy.revision != taxonomy_revision
        or taxonomy.fingerprint != taxonomy_fingerprint
    ):
        raise ValueError("taxonomy identity differs from its pinned launch contract")

    raw_suite = payload["suite"]
    if not isinstance(raw_suite, Mapping):
        raise ValueError("suite must be an object")
    _exact_keys(raw_suite, {"path", "suite_id", "version", "fingerprint"}, "suite")
    suite_path = _repo_relative(repository_root, raw_suite["path"], "suite.path")
    suite = load_suite(suite_path)
    suite_id = _nonempty_string(raw_suite["suite_id"], "suite.suite_id")
    suite_version = _nonempty_string(raw_suite["version"], "suite.version")
    suite_fingerprint = _nonempty_string(raw_suite["fingerprint"], "suite.fingerprint")
    if (
        suite.suite_id != suite_id
        or suite.version != suite_version
        or suite.fingerprint != suite_fingerprint
    ):
        raise ValueError("benchmark suite identity differs from its pinned launch contract")

    raw_route_contract = payload["route_contract"]
    if not isinstance(raw_route_contract, Mapping):
        raise ValueError("route_contract must be an object")
    _exact_keys(raw_route_contract, {"path", "fingerprint"}, "route_contract")
    route_contract_path = _repo_relative(
        repository_root,
        raw_route_contract["path"],
        "route_contract.path",
    )
    route_contract_fingerprint = _nonempty_string(
        raw_route_contract["fingerprint"],
        "route_contract.fingerprint",
    )
    route_contract = load_route_evaluation_contract(
        route_contract_path,
        suite=suite,
        taxonomy=taxonomy,
    )
    if route_contract.content_sha256 != route_contract_fingerprint:
        raise ValueError("route contract identity differs from its pinned launch contract")

    raw_experts = payload["experts"]
    if not isinstance(raw_experts, list) or not raw_experts:
        raise ValueError("experts must be a non-empty array")
    experts = tuple(raw_experts)
    if any(
        not isinstance(expert, str) or _EXPERT_PATTERN.fullmatch(expert) is None
        for expert in experts
    ):
        raise ValueError("experts must contain bounded kebab-case route IDs")
    if len(experts) != len(set(experts)):
        raise ValueError("experts must not contain duplicates")
    if experts != taxonomy.leaf_ids:
        raise ValueError("expert order must exactly match the pinned taxonomy leaf axis")

    raw_policy = payload["router_policy"]
    if not isinstance(raw_policy, Mapping):
        raise ValueError("router_policy must be an object")
    _exact_keys(
        raw_policy,
        {"temperature", "confidence_threshold", "null_bias"},
        "router_policy",
    )
    temperature = _finite(raw_policy["temperature"], "router_policy.temperature")
    confidence = _finite(
        raw_policy["confidence_threshold"], "router_policy.confidence_threshold"
    )
    null_bias = _finite(raw_policy["null_bias"], "router_policy.null_bias")
    if temperature <= 0:
        raise ValueError("router_policy.temperature must be positive")
    if not 0 <= confidence <= 1:
        raise ValueError("router_policy.confidence_threshold must be from zero to one")

    endpoint = _nonempty_string(payload["endpoint"], "endpoint").rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("endpoint must be a loopback HTTP URL")
    port = _positive_int(payload["port"], "port")
    if port > 65535 or parsed.port != port:
        raise ValueError("port must match the endpoint and be at most 65535")
    timeout = _finite(payload["timeout_seconds"], "timeout_seconds")
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive")

    return RackBenchmarkConfig(
        source=source,
        repository_root=repository_root,
        campaign_id=campaign_id,
        model_directory=_repo_relative(
            repository_root, payload["model_directory"], "model_directory"
        ),
        adapter_root=_repo_relative(repository_root, payload["adapter_root"], "adapter_root"),
        router_checkpoint=_repo_relative(
            repository_root, payload["router_checkpoint"], "router_checkpoint"
        ),
        taxonomy_path=taxonomy_path,
        taxonomy_id=taxonomy_id,
        taxonomy_revision=taxonomy_revision,
        taxonomy_fingerprint=taxonomy_fingerprint,
        suite=SuitePin(suite_path, suite_id, suite_version, suite_fingerprint),
        route_contract_path=route_contract_path,
        route_contract_fingerprint=route_contract_fingerprint,
        output_root=_repo_relative(repository_root, payload["output_root"], "output_root"),
        feedback_database=_repo_relative(
            repository_root, payload["feedback_database"], "feedback_database"
        ),
        endpoint=endpoint,
        port=port,
        expert_rank=_positive_int(payload["expert_rank"], "expert_rank"),
        experts=experts,  # type: ignore[arg-type]
        router_policy=RouterPolicyPin(temperature, confidence, null_bias),
        timeout_seconds=timeout,
        repetitions=_positive_int(payload["repetitions"], "repetitions"),
    )


def _same_path(left: object, right: Path) -> bool:
    if not isinstance(left, str) or not left.strip():
        return False
    try:
        return Path(left).expanduser().resolve() == right.resolve()
    except OSError:
        return False


def preflight_artifacts(config: RackBenchmarkConfig) -> dict[str, object]:
    """Verify the expected rack layout without loading weights or allocating a GPU."""

    if not complete_local_model(config.model_directory):
        raise FileNotFoundError(
            f"base model is absent or incomplete: {config.model_directory}"
        )
    target_modules: tuple[str, ...] | None = None
    checked_experts: list[str] = []
    for expert, directory in zip(config.experts, config.expert_paths, strict=True):
        config_path = directory / "adapter_config.json"
        weights_path = directory / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"expert {expert!r} needs adapter_config.json and adapter_model.safetensors"
            )
        payload = _strict_json_object(config_path, f"adapter {expert!r} config")
        if payload.get("name") != expert:
            raise ValueError(f"adapter {expert!r} exposes a different runtime name")
        if payload.get("rank") != config.expert_rank:
            raise ValueError(
                f"adapter {expert!r} has rank {payload.get('rank')!r}, "
                f"expected {config.expert_rank}"
            )
        if not _same_path(payload.get("base_model_id"), config.model_directory):
            raise ValueError(f"adapter {expert!r} targets a different base model")
        raw_targets = payload.get("target_modules")
        if (
            not isinstance(raw_targets, list)
            or not raw_targets
            or any(not isinstance(item, str) or not item for item in raw_targets)
        ):
            raise ValueError(f"adapter {expert!r} has no valid target_modules contract")
        current_targets = tuple(raw_targets)
        if target_modules is None:
            target_modules = current_targets
        elif current_targets != target_modules:
            raise ValueError("resident adapters target different projection sets")
        checked_experts.append(expert)

    router_config_path = config.router_checkpoint / "router_config.json"
    router_weights_path = config.router_checkpoint / "router.safetensors"
    if not router_config_path.is_file() or not router_weights_path.is_file():
        raise FileNotFoundError(
            "hierarchical router needs router_config.json and router.safetensors: "
            f"{config.router_checkpoint}"
        )
    router = _strict_json_object(router_config_path, "hierarchical router config")
    if router.get("schema_version") != HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA:
        raise ValueError("router checkpoint is not the hierarchical v1 schema")
    raw_taxonomy = router.get("taxonomy")
    if not isinstance(raw_taxonomy, Mapping) or (
        raw_taxonomy.get("taxonomy_id") != config.taxonomy_id
        or raw_taxonomy.get("revision") != config.taxonomy_revision
        or raw_taxonomy.get("content_sha256") != config.taxonomy_fingerprint
    ):
        raise ValueError("router checkpoint targets a different taxonomy")
    raw_adapters = router.get("adapters")
    adapter_ids = tuple(
        item.get("route_id") if isinstance(item, Mapping) else None
        for item in raw_adapters
    ) if isinstance(raw_adapters, list) else ()
    if adapter_ids != config.experts:
        raise ValueError("router checkpoint adapter axis differs from the launch rack")
    raw_model = router.get("model")
    if not isinstance(raw_model, Mapping) or not _same_path(
        raw_model.get("model_id"), config.model_directory
    ):
        raise ValueError("router checkpoint targets a different base model")
    return {
        "schema_version": "molor.routed-rack-benchmark-preflight.v1",
        "campaign_id": config.campaign_id,
        "base_model": str(config.model_directory),
        "expert_count": len(checked_experts),
        "experts": checked_experts,
        "expert_rank": config.expert_rank,
        "router_checkpoint": str(config.router_checkpoint),
        "taxonomy_fingerprint": config.taxonomy_fingerprint,
        "suite_fingerprint": config.suite.fingerprint,
        "route_contract_fingerprint": config.route_contract_fingerprint,
        "status": "ready",
    }


def _common_benchmark_args(config: RackBenchmarkConfig) -> list[str]:
    policy = config.router_policy
    return [
        sys.executable,
        "-m",
        "molor.benchmark.cli",
        "--suite-placeholder",  # replaced by the command builders
        "--route-contract",
        str(config.route_contract_path),
        "--route-taxonomy",
        str(config.taxonomy_path),
        "--endpoint",
        config.endpoint,
        "--output-root",
        str(config.output_root),
        "--repetitions",
        str(config.repetitions),
        "--route-temperature",
        str(policy.temperature),
        "--confidence-threshold",
        str(policy.confidence_threshold),
        "--null-bias",
        str(policy.null_bias),
        "--timeout",
        str(config.timeout_seconds),
    ]


def _benchmark_prefix(config: RackBenchmarkConfig, command: str) -> list[str]:
    common = _common_benchmark_args(config)
    marker = common.index("--suite-placeholder")
    common[marker : marker + 1] = [command, "--suite", str(config.suite.path)]
    return common


def build_launch_steps(config: RackBenchmarkConfig) -> tuple[LaunchStep, ...]:
    """Return the exact service, smoke, and two full-matrix command sequences."""

    serve = [
        sys.executable,
        "-m",
        "molor.teacher.cli",
        "serve",
        "--model",
        str(config.model_directory),
        "--db",
        str(config.feedback_database),
        "--port",
        str(config.port),
        "--router",
        str(config.router_checkpoint),
        "--eager-load",
    ]
    for path in config.expert_paths:
        serve.extend(("--expert", str(path)))

    smoke = _benchmark_prefix(config, "matrix")
    smoke.extend(
        (
            "--profile",
            "smoke",
            "--label",
            f"{config.campaign_id}-smoke-thinking-off",
            "--matrix-directory",
            str(config.smoke_matrix_directory),
            "--no-thinking",
        )
    )
    for expert in config.experts:
        smoke.extend(("--expert", expert))

    steps = [
        LaunchStep("serve", tuple(serve), False),
        LaunchStep("smoke-all-experts-thinking-off", tuple(smoke), True),
    ]
    for thinking in (False, True):
        state = "thinking-on" if thinking else "thinking-off"
        thinking_flag = "--thinking" if thinking else "--no-thinking"
        directory = config.full_directory(thinking)
        base_summary = directory / "base" / "summary.json"
        policies = (
            ("base", "base/null", 1),
            ("auto-top1", "auto", 1),
            ("auto-top2", "auto", 2),
        )
        for policy_id, route, top_k in policies:
            command = _benchmark_prefix(config, "run")
            command.extend(
                (
                    "--profile",
                    "full",
                    "--label",
                    f"{config.campaign_id}-full-{policy_id}-{state}",
                    "--run-directory",
                    str(directory / policy_id),
                    thinking_flag,
                    "--route",
                    route,
                    "--route-top-k",
                    str(top_k),
                )
            )
            if policy_id != "base":
                command.extend(("--baseline", str(base_summary)))
            steps.append(LaunchStep(f"full-{state}-{policy_id}", tuple(command), True))
    return tuple(steps)


def _health_payload(config: RackBenchmarkConfig) -> dict[str, Any]:
    request = Request(config.endpoint + "/health", method="GET")
    try:
        with urlopen(request, timeout=min(config.timeout_seconds, 15.0)) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"teacher API health returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"cannot reach the teacher API at {config.endpoint}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("teacher API health returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("teacher API health returned a non-object response")
    return payload


def validate_server_health(
    config: RackBenchmarkConfig, payload: Mapping[str, object]
) -> dict[str, object]:
    """Reject a healthy but wrong model/rack/router before any scored generation."""

    if payload.get("status") != "ok" or payload.get("protocol_version") != "molor.teacher-api.v1":
        raise RuntimeError("teacher API is not healthy under the expected protocol")
    if payload.get("model_loaded") is not True or payload.get("router_loaded") is not True:
        raise RuntimeError("teacher API must eagerly load both the model and hierarchical router")
    if not _same_path(payload.get("model_id"), config.model_directory):
        raise RuntimeError("teacher API loaded a different base model")
    if not _same_path(payload.get("router_checkpoint"), config.router_checkpoint):
        raise RuntimeError("teacher API loaded a different router checkpoint")
    raw_experts = payload.get("experts")
    if not isinstance(raw_experts, list) or tuple(raw_experts) != config.experts:
        raise RuntimeError("teacher API resident expert axis differs from the pinned rack")
    raw_paths = payload.get("configured_expert_paths")
    if not isinstance(raw_paths, list) or len(raw_paths) != len(config.expert_paths):
        raise RuntimeError("teacher API did not report every configured adapter path")
    if any(
        not _same_path(raw, expected)
        for raw, expected in zip(raw_paths, config.expert_paths, strict=True)
    ):
        raise RuntimeError("teacher API configured adapter paths differ from the pinned rack")
    return {
        "schema_version": "molor.routed-rack-server-health.v1",
        "campaign_id": config.campaign_id,
        "model_loaded": True,
        "router_loaded": True,
        "expert_count": len(config.experts),
        "status": "ready",
    }


def _powershell_argument(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_powershell(step: LaunchStep) -> str:
    return "& " + " ".join(_powershell_argument(argument) for argument in step.argv)


def _run_step(config: RackBenchmarkConfig, step: LaunchStep) -> None:
    completed = subprocess.run(
        step.argv,
        cwd=config.repository_root,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"launch step {step.step_id!r} exited {completed.returncode}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="print exact PowerShell commands without loading artifacts")
    commands.add_parser("preflight", help="verify the complete base/rack/router artifact layout")
    commands.add_parser("verify-server", help="verify the eagerly loaded loopback service")
    commands.add_parser("serve", help="preflight and launch the pinned teacher service")
    commands.add_parser("smoke", help="run base, all experts, auto top-1, and auto top-2")
    full = commands.add_parser("full", help="run only base, auto top-1, and auto top-2")
    full.add_argument("--thinking-mode", choices=("off", "on", "both"), default="both")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_rack_benchmark_config(args.config)
        steps = build_launch_steps(config)
        if args.command == "plan":
            launcher = (
                sys.executable,
                "-m",
                "molor.benchmark.rack_launcher",
                "--config",
                str(config.source),
            )
            print("# Terminal 1: preflight, then keep the service running")
            print(render_powershell(LaunchStep("preflight", (*launcher, "preflight"), False)))
            print(render_powershell(steps[0]))
            print("# Terminal 2: verify the eager load, then run in this order")
            print(
                render_powershell(
                    LaunchStep("verify-server", (*launcher, "verify-server"), False)
                )
            )
            for step in steps[1:]:
                print(f"# {step.step_id}")
                print(render_powershell(step))
            return 0
        if args.command == "preflight":
            print(json.dumps(preflight_artifacts(config), indent=2, sort_keys=True))
            return 0
        if args.command == "verify-server":
            print(
                json.dumps(
                    validate_server_health(config, _health_payload(config)),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "serve":
            preflight_artifacts(config)
            _run_step(config, steps[0])
            return 0

        validate_server_health(config, _health_payload(config))
        if args.command == "smoke":
            _run_step(config, steps[1])
            return 0
        if args.command == "full":
            selected = {
                "off": "full-thinking-off-",
                "on": "full-thinking-on-",
            }
            prefixes = (
                tuple(selected.values())
                if args.thinking_mode == "both"
                else (selected[args.thinking_mode],)
            )
            for step in steps[2:]:
                if step.step_id.startswith(prefixes):
                    _run_step(config, step)
            return 0
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"rack benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


__all__ = [
    "CONFIG_SCHEMA",
    "LaunchStep",
    "RackBenchmarkConfig",
    "build_launch_steps",
    "load_rack_benchmark_config",
    "main",
    "preflight_artifacts",
    "render_powershell",
    "validate_server_health",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the experiment wrapper
    raise SystemExit(main())
