"""Loopback-only JSON HTTP transport for the teacher/student service."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from molor.teacher.protocol import TOOL_SCHEMAS
from molor.teacher.service import TeacherStudentService


MAX_REQUEST_BYTES = 2 * 1024 * 1024


class _TeacherServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        service: TeacherStudentService,
    ) -> None:
        self.service = service
        super().__init__(address, TeacherRequestHandler)


class TeacherRequestHandler(BaseHTTPRequestHandler):
    server: _TeacherServer
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send(200, self.server.service.health())
                return
            if parsed.path == "/v1/capabilities":
                self._send(200, self.server.service.capabilities())
                return
            if parsed.path == "/openapi.json":
                self._send(200, self._openapi())
                return
            if parsed.path.startswith("/v1/generations/"):
                generation_id = unquote(parsed.path.removeprefix("/v1/generations/"))
                if not generation_id:
                    raise ValueError("generation_id is required")
                self._send(200, self.server.service.get_generation(generation_id))
                return
            if parsed.path == "/v1/review/next":
                teacher_id = parse_qs(parsed.query).get("teacher_id", [""])[0]
                self._send(200, self.server.service.next_review(teacher_id))
                return
            self._send(404, {"error": "not_found", "message": "Unknown endpoint"})
        except Exception as exc:  # request boundary
            self._error(exc)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            operations = {
                "/v1/student/generate": self.server.service.ask_student,
                "/v1/student/grade": self.server.service.grade_student,
                "/v1/student/preference": self.server.service.prefer_student_output,
                "/v1/student/load": lambda _payload: self.server.service.load_student(),
                "/v1/student/unload": lambda _payload: self.server.service.unload_student(),
                "/v1/export": self.server.service.export_training_data,
            }
            operation = operations.get(parsed.path)
            if operation is None:
                self._send(404, {"error": "not_found", "message": "Unknown endpoint"})
                return
            self._send(200, operation(payload))
        except Exception as exc:  # request boundary
            self._error(exc)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib API
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_json(self) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"request body must be at most {MAX_REQUEST_BYTES} bytes")
        body = self.rfile.read(length)
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def _error(self, exc: Exception) -> None:
        if isinstance(exc, KeyError):
            status, name = 404, "not_found"
        elif isinstance(exc, (TypeError, ValueError)):
            status, name = 400, "invalid_request"
        else:
            status, name = 500, "internal_error"
        self._send(status, {"error": name, "message": f"{type(exc).__name__}: {exc}"})

    def _send(self, status: int, payload: object) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"[teacher-api] {self.address_string()} {format % args}", flush=True)

    @staticmethod
    def _openapi() -> dict[str, object]:
        tools = {str(tool["name"]): tool for tool in TOOL_SCHEMAS}

        def post(tool_name: str, summary: str) -> dict[str, object]:
            return {
                "post": {
                    "summary": summary,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": tools[tool_name]["input_schema"],
                            }
                        },
                    },
                }
            }

        return {
            "openapi": "3.1.0",
            "info": {"title": "MoLoRA Teacher/Student API", "version": "1.0.0"},
            "servers": [{"url": "http://127.0.0.1:8765"}],
            "paths": {
                "/health": {"get": {"summary": "Server and lazy model status"}},
                "/v1/capabilities": {"get": {"summary": "Rubric and tool schemas"}},
                "/v1/student/generate": post(
                    "ask_student", "Generate and persist comparable student outputs"
                ),
                "/v1/student/grade": post(
                    "grade_student", "Record an independent teacher annotation"
                ),
                "/v1/student/preference": post(
                    "prefer_student_output", "Record a pairwise teacher preference"
                ),
                "/v1/student/load": {
                    "post": {"summary": "Eagerly load the configured student and expert rack"}
                },
                "/v1/student/unload": {
                    "post": {"summary": "Unload the student and release accelerator memory"}
                },
                "/v1/review/next": {
                    "get": {"summary": "Fetch the next unreviewed generation"}
                },
                "/v1/generations/{generation_id}": {
                    "get": {"summary": "Fetch output, annotations, and consensus"}
                },
                "/v1/export": {"post": {"summary": "Export RL/SFT/DPO JSONL"}},
            },
        }


def serve(service: TeacherStudentService, *, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be from 1 to 65535")
    server = _TeacherServer(("127.0.0.1", port), service)
    print(f"MoLoRA teacher API ready at http://127.0.0.1:{port}", flush=True)
    print("Tool contract: GET /v1/capabilities  OpenAPI: GET /openapi.json", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
