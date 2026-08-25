"""Launch the local API or invoke it as a model-friendly command-line tool."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from molor.feedback import FeedbackStore
from molor.model_profiles import (
    RuntimeArtifactDefaults,
    default_model,
    feedback_database,
    runtime_artifact_defaults,
)
from molor.teacher.protocol import capabilities
from molor.teacher.server import serve
from molor.teacher.service import StudentConfig, TeacherStudentService


OPERATIONS = {
    "ask_student": ("POST", "/v1/student/generate"),
    "grade_student": ("POST", "/v1/student/grade"),
    "prefer_student_output": ("POST", "/v1/student/preference"),
    "next_student_review": ("GET_QUERY", "/v1/review/next"),
    "get_student_generation": ("GET_ID", "/v1/generations/"),
    "load_student": ("POST", "/v1/student/load"),
    "unload_student": ("POST", "/v1/student/unload"),
    "export_training_data": ("POST", "/v1/export"),
}
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _environment_paths(name: str, environment: Mapping[str, str]) -> tuple[str, ...]:
    raw = environment.get(name, "")
    return tuple(part.strip() for part in raw.split(os.pathsep) if part.strip())


def _parser(
    workspace_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> argparse.ArgumentParser:
    workspace_root = workspace_root or _WORKSPACE_ROOT
    environment = os.environ if environment is None else environment
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("serve", help="Run the loopback teacher/student API")
    server.add_argument(
        "--model",
        default=default_model(workspace_root, environment),
    )
    server.add_argument(
        "--expert-a",
        default=None,
        help="Primary resident expert; omitted selects a compatible lane default when available.",
    )
    server.add_argument(
        "--expert-b",
        default=None,
        help="Secondary resident expert; omitted selects a compatible lane default when available.",
    )
    server.add_argument(
        "--expert",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Additional resident expert adapter; repeat for an arbitrary expert rack. "
            "Defaults to MOLOR_EXPERTS/MOLOR_EXPERT_PATHS or the local architecture and "
            "verified-code experts when available."
        ),
    )
    server.add_argument(
        "--router",
        default=None,
        help="Prompt auto-router checkpoint; omitted selects a compatible lane default.",
    )
    server.add_argument(
        "--db",
        type=Path,
        default=None,
        help="feedback database; omitted selects the active model lane's notebook",
    )
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--eager-load", action="store_true")

    invoke = subparsers.add_parser("invoke", help="Call one tool and print JSON")
    invoke.add_argument("operation", choices=sorted(OPERATIONS))
    invoke.add_argument("--endpoint", default="http://127.0.0.1:8765")
    source = invoke.add_mutually_exclusive_group()
    source.add_argument("--json", dest="json_payload")
    source.add_argument("--input", type=Path)

    subparsers.add_parser("schema", help="Print the machine-readable tool contract")
    health = subparsers.add_parser("health", help="Check a running teacher API")
    health.add_argument("--endpoint", default="http://127.0.0.1:8765")
    return parser


def _resolve_serve_artifacts(
    args: argparse.Namespace,
    workspace_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> RuntimeArtifactDefaults:
    """Apply explicit/env overrides before architecture-compatible local defaults."""

    workspace_root = workspace_root or _WORKSPACE_ROOT
    environment = os.environ if environment is None else environment
    defaults = runtime_artifact_defaults(args.model, workspace_root)
    expert_a = (
        args.expert_a
        if args.expert_a is not None
        else environment.get("MOLOR_EXPERT_A", "").strip() or defaults.expert_a
    )
    expert_b = (
        args.expert_b
        if args.expert_b is not None
        else environment.get("MOLOR_EXPERT_B", "").strip() or defaults.expert_b
    )
    expert_paths = (
        tuple(args.expert)
        if args.expert is not None
        else _environment_paths("MOLOR_EXPERTS", environment)
        or _environment_paths("MOLOR_EXPERT_PATHS", environment)
        or defaults.expert_paths
    )
    router = (
        args.router
        if args.router is not None
        else environment.get("MOLOR_ROUTER", "").strip()
        or environment.get("MOLOR_ROUTER_CHECKPOINT", "").strip()
        or defaults.router
    )
    return RuntimeArtifactDefaults(expert_a, expert_b, expert_paths, router)


def _payload(args: argparse.Namespace) -> dict[str, object]:
    if args.json_payload is not None:
        raw = args.json_payload
    elif args.input is not None:
        raw = args.input.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        raw = "{}"
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("tool input must be a JSON object")
    return value


def _request(endpoint: str, method: str, path: str, payload: dict[str, object]) -> object:
    base = endpoint.rstrip("/")
    data = None
    if method == "GET_QUERY":
        path = f"{path}?{urlencode({'teacher_id': payload.get('teacher_id', '')})}"
        method = "GET"
    elif method == "GET_ID":
        path += quote(str(payload.get("generation_id") or ""), safe="")
        method = "GET"
    else:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=3600) as response:  # noqa: S310 - explicit local endpoint
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"teacher API returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"cannot reach teacher API at {base}: {exc.reason}") from exc


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "schema":
        print(json.dumps(capabilities(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "health":
        result = _request(args.endpoint, "GET", "/health", {})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "invoke":
        method, path = OPERATIONS[args.operation]
        result = _request(args.endpoint, method, path, _payload(args))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "serve":
        artifacts = _resolve_serve_artifacts(args)
        store = FeedbackStore(args.db or feedback_database(args.model, _WORKSPACE_ROOT))
        service = TeacherStudentService(
            StudentConfig(
                args.model,
                artifacts.expert_a,
                artifacts.expert_b,
                expert_paths=artifacts.expert_paths,
                router_checkpoint=artifacts.router,
            ),
            store,
        )
        try:
            if args.eager_load:
                print(json.dumps(service.load_student(), ensure_ascii=False), flush=True)
            serve(service, port=args.port)
        finally:
            service.unload_student()
            store.close()
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
