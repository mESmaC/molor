"""Strict authenticated loopback HTTP transport for local chat generation.

The transport is deliberately independent from model loading and execution.  A caller
injects a :class:`RuntimeChatService` that yields already-computed stream records.  This
module owns the untrusted HTTP boundary, validates every request and emitted record, and
turns service failures into a bounded terminal error without exposing exception text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hmac
import ipaddress
import json
import math
import re
import select
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore, Event, Lock, Thread
from typing import Any, Literal, Protocol, cast


CHAT_REQUEST_SCHEMA = "molor.chat-request.v1"
CHAT_STREAM_PROTOCOL = "molor.chat-stream.v1"
CHAT_STREAM_MEDIA_TYPE = "application/x-ndjson"

# The browser endpoint admits at most 64 KiB before the bridge adds a schema and a
# server-issued request ID.  The runtime therefore has a slightly larger independent
# wire cap while retaining the same, tighter semantic transcript limits.
MAX_REQUEST_BYTES = 128 * 1024
MAX_EVENT_BYTES = 64 * 1024
MAX_STREAM_BYTES = 32 * 1024 * 1024
MAX_STREAM_EVENTS = 200_000
MAX_MESSAGES = 64
MAX_MESSAGE_UTF16_UNITS = 12_000
MAX_TRANSCRIPT_UTF16_UNITS = 48_000

_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_TOKEN = re.compile(r"^[A-Za-z0-9._~-]{32,512}$")
_JS_WHITESPACE_ONLY = re.compile(
    "^[\\u0009-\\u000d\\u0020\\u00a0\\u1680\\u2000-\\u200a"
    "\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff]*$"
)
_PUBLIC_STREAM_ERROR = "The local runtime could not complete generation."
_ERROR_RESERVE_BYTES = 2 * 1024


class RuntimeWireError(ValueError):
    """A bounded, client-safe request or stream contract error."""

    def __init__(self, status: HTTPStatus, public_message: str) -> None:
        super().__init__(public_message)
        self.status = status
        self.public_message = public_message


def _is_finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class RuntimeChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class RuntimeChatRequest:
    """Validated bridge request; authenticated memory scope is intentionally absent."""

    request_id: str
    conversation_id: str
    messages: tuple[RuntimeChatMessage, ...]
    memory: Literal["auto", "off"]


class RuntimeChatService(Protocol):
    """Model-facing service injected behind the HTTP transport.

    Implementations must yield regularly while work is in flight (normally with
    ``heartbeat`` records) and must stop promptly when ``cancellation`` is set.
    """

    def stream(
        self,
        request: RuntimeChatRequest,
        *,
        trusted_scope_id: str,
        cancellation: Event,
    ) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class RuntimeHttpConfig:
    token: str
    trusted_scope_id: str
    host: str = "127.0.0.1"
    port: int = 8765
    max_request_bytes: int = MAX_REQUEST_BYTES
    read_timeout_seconds: float = 5.0
    max_concurrent_requests: int = 1

    def __post_init__(self) -> None:
        if self.host not in ("127.0.0.1", "::1"):
            raise ValueError("host must be the literal loopback address 127.0.0.1 or ::1")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 0 <= self.port <= 65_535
        ):
            raise ValueError("port must be an integer between 0 and 65535")
        if not isinstance(self.token, str) or _TOKEN.fullmatch(self.token) is None:
            raise ValueError("token must contain 32-512 URL-safe characters")
        _identifier(self.trusted_scope_id, "trusted_scope_id")
        if (
            not isinstance(self.max_request_bytes, int)
            or isinstance(self.max_request_bytes, bool)
            or not 1 <= self.max_request_bytes <= MAX_REQUEST_BYTES
        ):
            raise ValueError(f"max_request_bytes must be between 1 and {MAX_REQUEST_BYTES}")
        if (
            not _is_finite_real(self.read_timeout_seconds)
            or not 0.1 <= self.read_timeout_seconds <= 60
        ):
            raise ValueError("read_timeout_seconds must be in [0.1, 60]")
        if (
            not isinstance(self.max_concurrent_requests, int)
            or isinstance(self.max_concurrent_requests, bool)
            or not 1 <= self.max_concurrent_requests <= 64
        ):
            raise ValueError("max_concurrent_requests must be between 1 and 64")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a valid bounded identifier")
    return value


def _utf16_units(value: str) -> int:
    """Match JavaScript string-length accounting and reject lone surrogates."""

    try:
        return len(value.encode("utf-16-le", "strict")) // 2
    except UnicodeEncodeError as exc:
        raise ValueError("string contains an invalid Unicode surrogate") from exc


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has an unexpected field set")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON objects must not contain duplicate keys")
        result[key] = value
    return result


def parse_chat_request(payload: bytes) -> RuntimeChatRequest:
    """Parse and validate one exact UTF-8 bridge request body."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload:
        raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Invalid chat request.")
    if len(payload) > MAX_REQUEST_BYTES:
        raise RuntimeWireError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Chat request is too large.")
    try:
        text = payload.decode("utf-8", "strict")
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"unsupported JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Invalid chat request.") from None

    if not isinstance(raw, dict):
        raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Invalid chat request.")
    try:
        _exact_keys(
            raw,
            {"schema_version", "request_id", "conversation_id", "messages", "memory"},
            "chat request",
        )
        if raw["schema_version"] != CHAT_REQUEST_SCHEMA:
            raise ValueError("unsupported request schema")
        request_id = _identifier(raw["request_id"], "request_id")
        conversation_id = _identifier(raw["conversation_id"], "conversation_id")
        memory = raw["memory"]
        if memory not in ("auto", "off"):
            raise ValueError("unsupported memory mode")
        raw_messages = raw["messages"]
        if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= MAX_MESSAGES:
            raise ValueError("invalid message count")

        messages: list[RuntimeChatMessage] = []
        total_units = 0
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise ValueError("message must be an object")
            _exact_keys(raw_message, {"role", "content"}, "message")
            role = raw_message["role"]
            content = raw_message["content"]
            if role not in ("user", "assistant") or not isinstance(content, str):
                raise ValueError("invalid message")
            if not content or _JS_WHITESPACE_ONLY.fullmatch(content) is not None:
                raise ValueError("message content is blank")
            units = _utf16_units(content)
            if units > MAX_MESSAGE_UTF16_UNITS:
                raise ValueError("message is too long")
            total_units += units
            if total_units > MAX_TRANSCRIPT_UTF16_UNITS:
                raise ValueError("transcript is too long")
            messages.append(RuntimeChatMessage(cast(Any, role), content))
        if messages[-1].role != "user":
            raise ValueError("final message must be from the user")
    except (KeyError, TypeError, ValueError):
        raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Invalid chat request.") from None

    return RuntimeChatRequest(
        request_id=request_id,
        conversation_id=conversation_id,
        messages=tuple(messages),
        memory=cast(Any, memory),
    )


def _record(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} contains a non-string key")
    return cast(Mapping[str, Any], value)


def _bounded_string(value: object, label: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or _utf16_units(value) > maximum:
        raise ValueError(f"{label} must be a bounded non-empty string")
    return value


def _finite_number(
    value: object,
    label: str,
    maximum: int | float = _MAX_SAFE_INTEGER,
) -> int | float:
    if not _is_finite_real(value) or value < 0 or value > maximum:
        raise ValueError(f"{label} must be a finite non-negative number")
    return value


def _nonnegative_integer(value: object, label: str, maximum: int = _MAX_SAFE_INTEGER) -> int:
    normalized = _finite_number(value, label, maximum)
    if not isinstance(normalized, int):
        if not isinstance(normalized, float) or not normalized.is_integer():
            raise ValueError(f"{label} must be an integer")
    return int(normalized)


def validate_chat_stream_event(
    value: object,
    *,
    expected_request_id: str,
) -> dict[str, Any]:
    """Validate and normalize one event exactly as the UI bridge does."""

    expected_request_id = _identifier(expected_request_id, "expected_request_id")
    record = _record(value, "runtime event")
    event_type = record.get("type")
    if not isinstance(event_type, str):
        raise ValueError("runtime event must contain a type")

    def common_request_id() -> str:
        request_id = _identifier(record.get("request_id"), "request_id")
        if request_id != expected_request_id:
            raise ValueError("runtime event uses another request identity")
        return request_id

    if event_type == "generation.started":
        _exact_keys(
            record,
            {
                "type",
                "request_id",
                "runtime_source",
                "model_id",
                "backend_mode",
                "quantization",
                "device",
                "stack_fingerprint",
                "context_used",
                "context_limit",
                "input_tokens",
                "metrics_estimated",
            },
            event_type,
        )
        if record["runtime_source"] != "local":
            raise ValueError("the loopback runtime source must be local")
        if not isinstance(record["metrics_estimated"], bool):
            raise ValueError("metrics_estimated must be boolean")
        context_used = _nonnegative_integer(record["context_used"], "context_used")
        context_limit = _nonnegative_integer(record["context_limit"], "context_limit")
        if context_limit < 1 or context_used > context_limit:
            raise ValueError("context usage exceeds the context limit")
        return {
            "type": event_type,
            "request_id": common_request_id(),
            "runtime_source": "local",
            "model_id": _bounded_string(record["model_id"], "model_id"),
            "backend_mode": _bounded_string(record["backend_mode"], "backend_mode"),
            "quantization": _bounded_string(record["quantization"], "quantization"),
            "device": _bounded_string(record["device"], "device"),
            "stack_fingerprint": _bounded_string(record["stack_fingerprint"], "stack_fingerprint"),
            "context_used": context_used,
            "context_limit": context_limit,
            "input_tokens": _nonnegative_integer(record["input_tokens"], "input_tokens"),
            "metrics_estimated": record["metrics_estimated"],
        }

    if event_type == "route":
        _exact_keys(record, {"type", "request_id", "capability", "memory"}, event_type)
        raw_capability = record["capability"]
        if not isinstance(raw_capability, (list, tuple)) or not 1 <= len(raw_capability) <= 32:
            raise ValueError("capability must contain 1-32 routes")
        capability: list[dict[str, Any]] = []
        for index, raw_route in enumerate(raw_capability):
            route = _record(raw_route, f"capability[{index}]")
            _exact_keys(route, {"id", "weight"}, f"capability[{index}]")
            capability.append(
                {
                    "id": _identifier(route["id"], f"capability[{index}].id"),
                    "weight": _finite_number(route["weight"], f"capability[{index}].weight", 1),
                }
            )
        if len({route["id"] for route in capability}) != len(capability):
            raise ValueError("capability route IDs must be unique")
        memory = _record(record["memory"], "memory")
        _exact_keys(memory, {"mode", "version"}, "memory")
        mode = memory["mode"]
        if mode not in ("applied", "abstained", "unavailable", "off"):
            raise ValueError("memory mode is invalid")
        version = memory["version"]
        if version is not None:
            version = _bounded_string(version, "memory.version", 256)
        has_version = version is not None
        if (mode in ("applied", "abstained")) != has_version:
            raise ValueError("memory mode and version are inconsistent")
        return {
            "type": event_type,
            "request_id": common_request_id(),
            "capability": capability,
            "memory": {"mode": mode, "version": version},
        }

    if event_type == "retrieval":
        _exact_keys(
            record,
            {"type", "request_id", "count", "latency_ms", "consolidation_state"},
            event_type,
        )
        return {
            "type": event_type,
            "request_id": common_request_id(),
            "count": _nonnegative_integer(record["count"], "count", 1_024),
            "latency_ms": _finite_number(record["latency_ms"], "latency_ms"),
            "consolidation_state": _bounded_string(
                record["consolidation_state"], "consolidation_state", 256
            ),
        }

    if event_type == "token":
        expected_keys = {"type", "request_id", "index", "text"}
        if "ttft_ms" in record:
            expected_keys.add("ttft_ms")
        _exact_keys(record, expected_keys, event_type)
        result: dict[str, Any] = {
            "type": event_type,
            "request_id": common_request_id(),
            "index": _nonnegative_integer(record["index"], "token index"),
            "text": _bounded_string(record["text"], "token text", 16_384),
        }
        if "ttft_ms" in record:
            result["ttft_ms"] = _finite_number(record["ttft_ms"], "ttft_ms")
        return result

    if event_type in ("metrics", "generation.completed"):
        expected_keys = {
            "type",
            "request_id",
            "completion_tokens",
            "context_used",
            "elapsed_ms",
            "tokens_per_second",
        }
        if event_type == "generation.completed":
            expected_keys.add("finish_reason")
        _exact_keys(record, expected_keys, event_type)
        result = {
            "type": event_type,
            "request_id": common_request_id(),
            "completion_tokens": _nonnegative_integer(
                record["completion_tokens"], "completion_tokens"
            ),
            "context_used": _nonnegative_integer(record["context_used"], "context_used"),
            "elapsed_ms": _finite_number(record["elapsed_ms"], "elapsed_ms"),
            "tokens_per_second": _finite_number(record["tokens_per_second"], "tokens_per_second"),
        }
        if event_type == "generation.completed":
            result["finish_reason"] = _bounded_string(record["finish_reason"], "finish_reason", 128)
        return result

    if event_type == "error":
        _exact_keys(record, {"type", "request_id", "message"}, event_type)
        _bounded_string(record["message"], "error message", 1_024)
        return {
            "type": event_type,
            "request_id": common_request_id(),
            "message": _PUBLIC_STREAM_ERROR,
        }

    if event_type == "heartbeat":
        _exact_keys(record, {"type", "request_id", "phase"}, event_type)
        phase = record["phase"]
        if phase not in ("loading", "prefill", "decoding"):
            raise ValueError("heartbeat phase is invalid")
        return {
            "type": event_type,
            "request_id": common_request_id(),
            "phase": phase,
        }

    raise ValueError("unknown runtime event type")


class ChatStreamSequenceValidator:
    """Stateful ordering checks matching ``ui/lib/chat-stream.ts``."""

    def __init__(self, request_id: str) -> None:
        self.request_id = _identifier(request_id, "request_id")
        self.started = False
        self.route_received = False
        self.retrieval_received = False
        self.next_token_index = 0
        self.last_completion_tokens = 0
        self.last_context_used = 0
        self.terminal = False

    def accept(self, value: object) -> dict[str, Any]:
        if self.terminal:
            raise ValueError("runtime emitted data after a terminal event")
        event = validate_chat_stream_event(value, expected_request_id=self.request_id)
        event_type = event["type"]
        if event_type == "heartbeat":
            if not self.started:
                raise ValueError("heartbeat arrived before generation.started")
            return event
        if not self.started:
            if event_type == "error":
                self.terminal = True
                return event
            if event_type != "generation.started":
                raise ValueError("first event must be generation.started")
            self.started = True
            self.last_context_used = event["context_used"]
            return event

        if event_type == "generation.started":
            raise ValueError("generation.started may only appear once")
        if event_type == "route":
            if self.route_received or self.retrieval_received or self.next_token_index > 0:
                raise ValueError("capability route arrived out of order")
            self.route_received = True
            return event
        if event_type == "retrieval":
            if not self.route_received or self.retrieval_received or self.next_token_index > 0:
                raise ValueError("retrieval telemetry arrived out of order")
            self.retrieval_received = True
            return event
        if event_type == "token":
            self._require_ready()
            if event["index"] != self.next_token_index:
                raise ValueError("token indexes must be contiguous")
            self.next_token_index += 1
            return event
        if event_type in ("metrics", "generation.completed"):
            self._require_ready()
            completion_tokens = event["completion_tokens"]
            context_used = event["context_used"]
            if (
                completion_tokens < self.last_completion_tokens
                or context_used < self.last_context_used
            ):
                raise ValueError("runtime metrics moved backwards")
            self.last_completion_tokens = completion_tokens
            self.last_context_used = context_used
            if event_type == "generation.completed":
                self.terminal = True
            return event
        if event_type == "error":
            self.terminal = True
            return event
        raise AssertionError("validated event type was not handled")

    def _require_ready(self) -> None:
        if not self.route_received or not self.retrieval_received:
            raise ValueError("generation output arrived before route and retrieval telemetry")


def _encode_event(event: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("runtime event cannot be encoded as strict JSON") from exc
    if len(encoded) > MAX_EVENT_BYTES:
        raise ValueError("runtime event exceeded 64 KiB")
    return encoded + b"\n"


class RuntimeHTTPServer(ThreadingHTTPServer):
    """Threaded loopback-only server with one shared injectable chat service."""

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, config: RuntimeHttpConfig, service: RuntimeChatService) -> None:
        if not isinstance(config, RuntimeHttpConfig):
            raise TypeError("config must be a RuntimeHttpConfig")
        if not callable(getattr(service, "stream", None)):
            raise TypeError("service must provide a callable stream method")
        self.address_family = socket.AF_INET6 if config.host == "::1" else socket.AF_INET
        self.config = config
        self.chat_service = service
        self._slots = BoundedSemaphore(config.max_concurrent_requests)
        self._active_lock = Lock()
        self._active_cancellations: set[Event] = set()
        super().__init__((config.host, config.port), _RuntimeRequestHandler)

    @property
    def endpoint(self) -> str:
        host = f"[{self.config.host}]" if self.config.host == "::1" else self.config.host
        return f"http://{host}:{self.server_port}/v1/chat/stream"

    def _register(self, cancellation: Event) -> None:
        with self._active_lock:
            self._active_cancellations.add(cancellation)

    def _unregister(self, cancellation: Event) -> None:
        with self._active_lock:
            self._active_cancellations.discard(cancellation)

    def cancel_active_requests(self) -> None:
        with self._active_lock:
            active = tuple(self._active_cancellations)
        for cancellation in active:
            cancellation.set()


@dataclass(slots=True)
class RunningRuntimeServer:
    """Context-managed server thread intended for local launchers and tests."""

    server: RuntimeHTTPServer
    thread: Thread
    _closed: bool = False

    @property
    def endpoint(self) -> str:
        return self.server.endpoint

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.cancel_active_requests()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def __enter__(self) -> RunningRuntimeServer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def start_runtime_server(
    config: RuntimeHttpConfig,
    service: RuntimeChatService,
) -> RunningRuntimeServer:
    """Bind and start a daemon serving thread, returning a deterministic lifecycle handle."""

    server = RuntimeHTTPServer(config, service)
    thread = Thread(target=server.serve_forever, name="molor-runtime-http", daemon=True)
    try:
        thread.start()
    except BaseException:
        server.server_close()
        raise
    return RunningRuntimeServer(server, thread)


class _RuntimeRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "molor-runtime"
    sys_version = ""

    @property
    def runtime_server(self) -> RuntimeHTTPServer:
        return cast(RuntimeHTTPServer, self.server)

    def log_message(self, format: str, *args: object) -> None:
        # The default handler logs attacker-controlled paths and user-agent values.
        # The embedding application can add structured, redacted telemetry instead.
        return

    def send_error(
        self,
        code: int,
        message: str | None = None,
        explain: str | None = None,
    ) -> None:
        """Replace stdlib HTML diagnostics with a constant bounded response."""

        try:
            status = HTTPStatus(code)
        except ValueError:
            status = HTTPStatus.BAD_REQUEST
        self._send_json_error(status, "Invalid HTTP request.")

    def handle_expect_100(self) -> bool:
        self._send_json_error(HTTPStatus.EXPECTATION_FAILED, "Unsupported request expectation.")
        return False

    def do_POST(self) -> None:
        self.close_connection = True
        try:
            self._validate_peer_and_target()
            content_length = self._validate_headers()
            payload = self._read_body(content_length)
            request = parse_chat_request(payload)
        except RuntimeWireError as exc:
            self._send_json_error(exc.status, exc.public_message)
            return

        if not self.runtime_server._slots.acquire(blocking=False):
            self._send_json_error(HTTPStatus.SERVICE_UNAVAILABLE, "The local runtime is busy.")
            return

        cancellation = Event()
        self.runtime_server._register(cancellation)
        watcher_done = Event()
        watcher = Thread(
            target=self._watch_for_disconnect,
            args=(cancellation, watcher_done),
            name=f"molor-disconnect-{request.request_id}",
            daemon=True,
        )
        try:
            self._send_stream_headers()
            watcher.start()
            self._stream_service(request, cancellation)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            cancellation.set()
        except OSError:
            cancellation.set()
        finally:
            watcher_done.set()
            cancellation.set()
            if watcher.ident is not None:
                watcher.join(timeout=0.25)
            self.runtime_server._unregister(cancellation)
            self.runtime_server._slots.release()
            self.close_connection = True

    def do_GET(self) -> None:
        self._method_not_allowed()

    def do_HEAD(self) -> None:
        self._method_not_allowed()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self.close_connection = True
        self._send_json_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "Only POST is supported.",
            extra_headers=("Allow", "POST"),
        )

    def _validate_peer_and_target(self) -> None:
        try:
            peer = ipaddress.ip_address(self.client_address[0].split("%", 1)[0])
        except ValueError:
            raise RuntimeWireError(HTTPStatus.FORBIDDEN, "Loopback access required.") from None
        if not peer.is_loopback:
            raise RuntimeWireError(HTTPStatus.FORBIDDEN, "Loopback access required.")
        if self.path != "/v1/chat/stream":
            raise RuntimeWireError(HTTPStatus.NOT_FOUND, "Unknown runtime endpoint.")

        host_values = self.headers.get_all("Host", [])
        host = self.runtime_server.config.host
        expected = (
            f"[{host}]:{self.runtime_server.server_port}"
            if host == "::1"
            else f"{host}:{self.runtime_server.server_port}"
        )
        if len(host_values) != 1 or host_values[0].strip().lower() != expected.lower():
            raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Invalid Host header.")

    def _one_header(self, name: str, status: HTTPStatus, message: str) -> str:
        values = self.headers.get_all(name, [])
        if len(values) != 1:
            raise RuntimeWireError(status, message)
        return values[0].strip()

    def _validate_headers(self) -> int:
        if self.headers.get_all("Transfer-Encoding", []):
            raise RuntimeWireError(
                HTTPStatus.BAD_REQUEST,
                "Transfer-Encoding is not supported.",
            )
        authorization = self._one_header(
            "Authorization",
            HTTPStatus.UNAUTHORIZED,
            "Runtime authentication required.",
        )
        candidate = authorization[7:] if authorization.startswith("Bearer ") else ""
        try:
            candidate_bytes = candidate.encode("ascii", "strict")
        except UnicodeEncodeError:
            candidate_bytes = b""
        candidate_for_comparison = (
            candidate_bytes if len(candidate_bytes) <= 512 else candidate_bytes[:512]
        )
        matches = hmac.compare_digest(
            candidate_for_comparison,
            self.runtime_server.config.token.encode("ascii"),
        )
        if _TOKEN.fullmatch(candidate) is None or not matches:
            raise RuntimeWireError(
                HTTPStatus.UNAUTHORIZED,
                "Runtime authentication required.",
            )

        protocol = self._one_header(
            "X-Molor-Protocol",
            HTTPStatus.BAD_REQUEST,
            "Invalid runtime protocol.",
        )
        if protocol != CHAT_STREAM_PROTOCOL:
            raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Invalid runtime protocol.")
        accept = self._one_header(
            "Accept",
            HTTPStatus.NOT_ACCEPTABLE,
            "Accept must be application/x-ndjson.",
        )
        if accept.lower() != CHAT_STREAM_MEDIA_TYPE:
            raise RuntimeWireError(
                HTTPStatus.NOT_ACCEPTABLE,
                "Accept must be application/x-ndjson.",
            )

        content_type = self._one_header(
            "Content-Type",
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json.",
        )
        media_parts = [part.strip().lower() for part in content_type.split(";")]
        if (
            not media_parts
            or media_parts[0] != "application/json"
            or len(media_parts) > 2
            or (len(media_parts) == 2 and media_parts[1] != "charset=utf-8")
        ):
            raise RuntimeWireError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json.",
            )

        content_length = self._one_header(
            "Content-Length",
            HTTPStatus.LENGTH_REQUIRED,
            "Content-Length is required.",
        )
        if not content_length.isascii() or not content_length.isdecimal():
            raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length.")
        length = int(content_length)
        if length < 1:
            raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Request body is required.")
        if length > self.runtime_server.config.max_request_bytes:
            raise RuntimeWireError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Chat request is too large."
            )
        return length

    def _read_body(self, content_length: int) -> bytes:
        self.connection.settimeout(self.runtime_server.config.read_timeout_seconds)
        try:
            payload = self.rfile.read(content_length)
        except (OSError, socket.timeout):
            raise RuntimeWireError(HTTPStatus.REQUEST_TIMEOUT, "Chat request timed out.") from None
        if len(payload) != content_length:
            raise RuntimeWireError(HTTPStatus.BAD_REQUEST, "Incomplete chat request.")
        return payload

    def _send_stream_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", f"{CHAT_STREAM_MEDIA_TYPE}; charset=utf-8")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Molor-Protocol", CHAT_STREAM_PROTOCOL)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()

    def _stream_service(self, request: RuntimeChatRequest, cancellation: Event) -> None:
        sequence = ChatStreamSequenceValidator(request.request_id)
        total_bytes = 0
        event_count = 0
        iterator: Any = None

        def write_event(event: Mapping[str, Any], *, reserve_error: bool) -> None:
            nonlocal total_bytes, event_count
            line = _encode_event(event)
            byte_limit = MAX_STREAM_BYTES - (_ERROR_RESERVE_BYTES if reserve_error else 0)
            event_limit = MAX_STREAM_EVENTS - (1 if reserve_error else 0)
            if total_bytes + len(line) > byte_limit or event_count + 1 > event_limit:
                raise ValueError("runtime stream exceeded its resource limits")
            self.wfile.write(line)
            self.wfile.flush()
            total_bytes += len(line)
            event_count += 1

        def write_safe_error() -> None:
            if cancellation.is_set() or sequence.terminal:
                return
            error = sequence.accept(
                {
                    "type": "error",
                    "request_id": request.request_id,
                    "message": _PUBLIC_STREAM_ERROR,
                }
            )
            write_event(error, reserve_error=False)

        try:
            stream = self.runtime_server.chat_service.stream(
                request,
                trusted_scope_id=self.runtime_server.config.trusted_scope_id,
                cancellation=cancellation,
            )
            iterator = iter(stream)
            for raw_event in iterator:
                if cancellation.is_set():
                    return
                event = sequence.accept(raw_event)
                terminal = sequence.terminal
                write_event(event, reserve_error=not terminal)
                if terminal:
                    return
            write_safe_error()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
            cancellation.set()
            raise
        except OSError:
            cancellation.set()
            raise
        except Exception:
            write_safe_error()
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _watch_for_disconnect(self, cancellation: Event, done: Event) -> None:
        while not cancellation.is_set() and not done.wait(0.1):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    continue
                pending = self.connection.recv(1, socket.MSG_PEEK)
                if not pending:
                    cancellation.set()
                    return
                # The request body has already been consumed and keep-alive is disabled.
                # Any additional client bytes are unexpected; fail closed.
                cancellation.set()
                return
            except (OSError, ValueError):
                cancellation.set()
                return

    def _send_json_error(
        self,
        status: HTTPStatus,
        message: str,
        *,
        extra_headers: tuple[str, str] | None = None,
    ) -> None:
        self.close_connection = True
        payload = json.dumps(
            {"error": message},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        try:
            self.send_response(status)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            if status == HTTPStatus.UNAUTHORIZED:
                self.send_header("WWW-Authenticate", "Bearer")
            if extra_headers is not None:
                self.send_header(*extra_headers)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
                self.wfile.flush()
        except OSError:
            pass
