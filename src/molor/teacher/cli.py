"""Launch the local API or invoke it as a model-friendly command-line tool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from molor.feedback import DEFAULT_FEEDBACK_DB, FeedbackStore
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


def _first_existing(*candidates: str) -> str:
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir():
            return str(path.resolve())
    return ""


def _environment_paths(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    return tuple(part.strip() for part in raw.split(os.pathsep) if part.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("serve", help="Run the loopback teacher/student API")
    server.add_argument(
        "--model",
        default=os.environ.get("MOLOR_MODEL_ID")
        or _first_existing("models/Qwen3-1.7B")
        or "Qwen/Qwen3-1.7B",
    )
    server.add_argument(
        "--expert-a",
        default=os.environ.get("MOLOR_EXPERT_A")
        or _first_existing(
            "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
            "checkpoints/qwen3-1.7b-corpus-experts/software-architecture-v1",
            "checkpoints/qwen3-1.7b-real-experts/financial-analysis-verified",
            "checkpoints/qwen3-1.7b-real-experts/financial-analysis",
            "checkpoints/precision",
            "checkpoints/qwen3-1.7b-two-styles/precision",
        ),
    )
    server.add_argument(
        "--expert-b",
        default=os.environ.get("MOLOR_EXPERT_B")
        or _first_existing(
            "checkpoints/qwen3-1.7b-corpus-experts/ocb-can-verified-v1",
            "checkpoints/qwen3-1.7b-real-experts/local-embedded-code",
            "checkpoints/qwen3-1.7b-real-experts/terminal-cli",
            "checkpoints/creative",
            "checkpoints/qwen3-1.7b-two-styles/creative",
        ),
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
        default=(
            os.environ.get("MOLOR_ROUTER")
            or os.environ.get("MOLOR_ROUTER_CHECKPOINT")
            or _first_existing(
                "checkpoints/qwen3-1.7b-corpus-router-v3-mlp",
                "checkpoints/qwen3-1.7b-corpus-router-v1",
            )
        ),
        help="Prompt auto-router checkpoint directory.",
    )
    server.add_argument("--db", type=Path, default=DEFAULT_FEEDBACK_DB)
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
        configured_experts = (
            tuple(args.expert)
            if args.expert is not None
            else _environment_paths("MOLOR_EXPERTS")
            or _environment_paths("MOLOR_EXPERT_PATHS")
            or tuple(
                filter(
                    None,
                    (
                        _first_existing(
                            "checkpoints/qwen3-1.7b-corpus-experts/"
                            "software-architecture-v1"
                        ),
                        _first_existing(
                            "checkpoints/qwen3-1.7b-corpus-experts/"
                            "verified-embedded-code-v1"
                        ),
                    ),
                )
            )
        )
        store = FeedbackStore(args.db)
        service = TeacherStudentService(
            StudentConfig(
                args.model,
                args.expert_a,
                args.expert_b,
                expert_paths=configured_experts,
                router_checkpoint=args.router,
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
