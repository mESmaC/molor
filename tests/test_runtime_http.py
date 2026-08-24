from __future__ import annotations

from collections.abc import Iterable, Mapping
from http import HTTPStatus
from http.client import HTTPConnection
import json
import math
from threading import Event
from typing import Any
from urllib.parse import urlsplit

import pytest

from molor.runtime.http import (
    CHAT_REQUEST_SCHEMA,
    CHAT_STREAM_MEDIA_TYPE,
    CHAT_STREAM_PROTOCOL,
    ChatStreamSequenceValidator,
    RuntimeChatRequest,
    RuntimeHttpConfig,
    RuntimeWireError,
    parse_chat_request,
    start_runtime_server,
    validate_chat_stream_event,
)


TOKEN = "local-runtime-token-0123456789abcdef"
SCOPE = "user/local"


def request_dict(
    request_id: str = "runtime-00000000-0000-4000-8000-000000000001",
) -> dict[str, Any]:
    return {
        "schema_version": CHAT_REQUEST_SCHEMA,
        "request_id": request_id,
        "conversation_id": "conversation/one",
        "messages": [{"role": "user", "content": "Hello locally"}],
        "memory": "auto",
    }


def request_bytes(**changes: Any) -> bytes:
    value = request_dict()
    value.update(changes)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def started(request_id: str) -> dict[str, Any]:
    return {
        "type": "generation.started",
        "request_id": request_id,
        "runtime_source": "local",
        "model_id": "molor-test-runtime",
        "backend_mode": "Prepared generation test",
        "quantization": "not-loaded",
        "device": "cpu",
        "stack_fingerprint": "a" * 64,
        "context_used": 10,
        "context_limit": 1_024,
        "input_tokens": 10,
        "metrics_estimated": False,
    }


def route(request_id: str, *, memory: str = "unavailable") -> dict[str, Any]:
    return {
        "type": "route",
        "request_id": request_id,
        "capability": [{"id": "base/null", "weight": 1.0}],
        "memory": {
            "mode": memory,
            "version": None if memory in ("off", "unavailable") else "b" * 64,
        },
    }


def retrieval(request_id: str) -> dict[str, Any]:
    return {
        "type": "retrieval",
        "request_id": request_id,
        "count": 0,
        "latency_ms": 0.25,
        "consolidation_state": "idle",
    }


def completed(request_id: str) -> dict[str, Any]:
    return {
        "type": "generation.completed",
        "request_id": request_id,
        "completion_tokens": 1,
        "context_used": 11,
        "elapsed_ms": 5,
        "tokens_per_second": 200,
        "finish_reason": "stop",
    }


def complete_events(request_id: str) -> tuple[dict[str, Any], ...]:
    return (
        started(request_id),
        route(request_id),
        retrieval(request_id),
        {
            "type": "token",
            "request_id": request_id,
            "index": 0,
            "text": "Hello.",
            "ttft_ms": 3,
        },
        {
            "type": "metrics",
            "request_id": request_id,
            "completion_tokens": 1,
            "context_used": 11,
            "elapsed_ms": 5,
            "tokens_per_second": 200,
        },
        completed(request_id),
    )


class RecordingService:
    def __init__(self) -> None:
        self.requests: list[tuple[RuntimeChatRequest, str]] = []

    def stream(
        self,
        request: RuntimeChatRequest,
        *,
        trusted_scope_id: str,
        cancellation: Event,
    ) -> Iterable[Mapping[str, Any]]:
        self.requests.append((request, trusted_scope_id))
        yield from complete_events(request.request_id)


def config(**changes: Any) -> RuntimeHttpConfig:
    values: dict[str, Any] = {
        "token": TOKEN,
        "trusted_scope_id": SCOPE,
        "port": 0,
    }
    values.update(changes)
    return RuntimeHttpConfig(**values)


def connection_for(endpoint: str) -> tuple[HTTPConnection, str]:
    parsed = urlsplit(endpoint)
    assert parsed.hostname is not None and parsed.port is not None
    return HTTPConnection(parsed.hostname, parsed.port, timeout=3), parsed.path


def post(
    endpoint: str,
    *,
    payload: bytes | None = None,
    headers: dict[str, str] | None = None,
    path: str | None = None,
) -> tuple[HTTPConnection, Any]:
    connection, endpoint_path = connection_for(endpoint)
    request_headers = {
        "Accept": CHAT_STREAM_MEDIA_TYPE,
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "X-Molor-Protocol": CHAT_STREAM_PROTOCOL,
    }
    if headers:
        request_headers.update(headers)
    connection.request(
        "POST",
        endpoint_path if path is None else path,
        body=request_bytes() if payload is None else payload,
        headers=request_headers,
    )
    return connection, connection.getresponse()


def read_events(response: Any) -> list[dict[str, Any]]:
    body = response.read().decode("utf-8")
    return [json.loads(line) for line in body.splitlines() if line]


def test_request_parser_accepts_only_the_exact_bounded_bridge_contract() -> None:
    parsed = parse_chat_request(request_bytes())

    assert parsed.request_id.startswith("runtime-")
    assert parsed.conversation_id == "conversation/one"
    assert parsed.memory == "auto"
    assert parsed.messages[0].content == "Hello locally"
    assert not hasattr(parsed, "scope_id")

    invalid_values = (
        {**request_dict(), "extra": True},
        {**request_dict(), "schema_version": "molor.chat-request.v2"},
        {**request_dict(), "memory": "always"},
        {**request_dict(), "messages": []},
        {
            **request_dict(),
            "messages": [{"role": "assistant", "content": "not final-user"}],
        },
        {
            **request_dict(),
            "messages": [{"role": "user", "content": "\ufeff\t\n"}],
        },
        {
            **request_dict(),
            "messages": [{"role": "user", "content": "😀" * 6_001}],
        },
    )
    for value in invalid_values:
        with pytest.raises(RuntimeWireError):
            parse_chat_request(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )


def test_request_parser_rejects_invalid_utf8_duplicate_keys_constants_and_surrogates() -> None:
    duplicate = request_bytes()[:-1] + b',"memory":"off"}'
    surrogate = request_bytes(messages=[{"role": "user", "content": "placeholder"}]).replace(
        b"placeholder", b"\\ud800"
    )

    for payload in (b"\xff", duplicate, b'{"value":NaN}', surrogate):
        with pytest.raises(RuntimeWireError) as exc_info:
            parse_chat_request(payload)
        assert exc_info.value.status == HTTPStatus.BAD_REQUEST


@pytest.mark.parametrize("host", ["localhost", "0.0.0.0", "192.168.1.10"])
def test_runtime_config_rejects_nonliteral_or_nonloopback_bindings(host: str) -> None:
    with pytest.raises(ValueError, match="literal loopback"):
        config(host=host)


def test_runtime_config_matches_the_bridge_token_and_scope_rules() -> None:
    with pytest.raises(ValueError, match="32-512"):
        config(token="short")
    with pytest.raises(ValueError, match="trusted_scope_id"):
        config(trusted_scope_id="scope with spaces")
    with pytest.raises(ValueError, match="port"):
        config(port=True)


def test_event_validator_matches_local_and_unavailable_memory_shapes() -> None:
    request_id = request_dict()["request_id"]
    assert (
        validate_chat_stream_event(started(request_id), expected_request_id=request_id)[
            "runtime_source"
        ]
        == "local"
    )
    assert validate_chat_stream_event(route(request_id), expected_request_id=request_id)[
        "memory"
    ] == {"mode": "unavailable", "version": None}

    invalid_source = {**started(request_id), "runtime_source": "simulator"}
    invalid_memory = route(request_id)
    invalid_memory["memory"] = {"mode": "unavailable", "version": "b" * 64}
    extra = {**retrieval(request_id), "private_path": "C:/secret"}
    nan_metrics = {
        "type": "metrics",
        "request_id": request_id,
        "completion_tokens": 0,
        "context_used": 10,
        "elapsed_ms": math.nan,
        "tokens_per_second": 0,
    }
    for event in (invalid_source, invalid_memory, extra, nan_metrics):
        with pytest.raises(ValueError):
            validate_chat_stream_event(event, expected_request_id=request_id)


def test_event_validator_sanitizes_service_error_text() -> None:
    request_id = request_dict()["request_id"]
    event = validate_chat_stream_event(
        {
            "type": "error",
            "request_id": request_id,
            "message": "C:/private/models/token=secret",
        },
        expected_request_id=request_id,
    )
    assert event == {
        "type": "error",
        "request_id": request_id,
        "message": "The local runtime could not complete generation.",
    }


def test_sequence_validator_requires_order_contiguous_tokens_and_monotonic_metrics() -> None:
    request_id = request_dict()["request_id"]
    sequence = ChatStreamSequenceValidator(request_id)
    for event in complete_events(request_id):
        sequence.accept(event)
    assert sequence.terminal

    wrong_index = ChatStreamSequenceValidator(request_id)
    wrong_index.accept(started(request_id))
    wrong_index.accept(route(request_id))
    wrong_index.accept(retrieval(request_id))
    with pytest.raises(ValueError, match="contiguous"):
        wrong_index.accept({"type": "token", "request_id": request_id, "index": 1, "text": "bad"})

    backwards = ChatStreamSequenceValidator(request_id)
    backwards.accept(started(request_id))
    backwards.accept(route(request_id))
    backwards.accept(retrieval(request_id))
    backwards.accept(
        {
            "type": "metrics",
            "request_id": request_id,
            "completion_tokens": 2,
            "context_used": 12,
            "elapsed_ms": 1,
            "tokens_per_second": 2,
        }
    )
    with pytest.raises(ValueError, match="backwards"):
        backwards.accept(completed(request_id))


def test_authenticated_server_streams_exact_close_delimited_ndjson() -> None:
    service = RecordingService()
    with start_runtime_server(config(), service) as running:
        connection, response = post(running.endpoint)
        try:
            assert response.status == HTTPStatus.OK
            assert response.getheader("Content-Type") == (
                f"{CHAT_STREAM_MEDIA_TYPE}; charset=utf-8"
            )
            assert response.getheader("X-Molor-Protocol") == CHAT_STREAM_PROTOCOL
            assert response.getheader("Content-Length") is None
            assert response.getheader("Connection") == "close"
            events = read_events(response)
        finally:
            connection.close()

    assert [event["type"] for event in events] == [
        "generation.started",
        "route",
        "retrieval",
        "token",
        "metrics",
        "generation.completed",
    ]
    assert events[1]["memory"] == {"mode": "unavailable", "version": None}
    assert service.requests[0][1] == SCOPE
    assert service.requests[0][0].conversation_id == "conversation/one"


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({"Authorization": "Bearer wrong-token-0123456789abcdefghijk"}, 401),
        ({"Authorization": f"Bearer {'é' * 40}"}, 401),
        ({"Accept": "application/json"}, 406),
        ({"Content-Type": "text/plain"}, 415),
        ({"X-Molor-Protocol": "molor.chat-stream.v2"}, 400),
        ({"Transfer-Encoding": "chunked"}, 400),
    ],
)
def test_server_rejects_invalid_security_and_protocol_headers(
    headers: dict[str, str], expected_status: int
) -> None:
    with start_runtime_server(config(), RecordingService()) as running:
        connection, response = post(running.endpoint, headers=headers)
        try:
            assert response.status == expected_status
            assert response.getheader("Connection") == "close"
            assert "error" in json.loads(response.read())
        finally:
            connection.close()


def test_server_rejects_wrong_path_malformed_body_and_oversized_body() -> None:
    with start_runtime_server(config(max_request_bytes=256), RecordingService()) as running:
        cases = (
            ({"path": "/v1/chat/stream?scope=user/other"}, HTTPStatus.NOT_FOUND),
            ({"payload": b"not-json"}, HTTPStatus.BAD_REQUEST),
            ({"payload": b"x" * 257}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
        )
        for options, status in cases:
            connection, response = post(running.endpoint, **options)
            try:
                assert response.status == status
                response.read()
            finally:
                connection.close()


def test_server_rejects_duplicate_authorization_headers() -> None:
    with start_runtime_server(config(), RecordingService()) as running:
        connection, path = connection_for(running.endpoint)
        payload = request_bytes()
        connection.putrequest("POST", path)
        connection.putheader("Accept", CHAT_STREAM_MEDIA_TYPE)
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.putheader("Authorization", f"Bearer {TOKEN}")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(payload)))
        connection.putheader("X-Molor-Protocol", CHAT_STREAM_PROTOCOL)
        connection.endheaders(payload)
        response = connection.getresponse()
        try:
            assert response.status == HTTPStatus.UNAUTHORIZED
            response.read()
        finally:
            connection.close()


def test_server_flushes_generation_started_before_service_continues() -> None:
    release = Event()

    class PausingService:
        def stream(
            self,
            request: RuntimeChatRequest,
            *,
            trusted_scope_id: str,
            cancellation: Event,
        ) -> Iterable[Mapping[str, Any]]:
            yield started(request.request_id)
            release.wait(2)
            yield route(request.request_id)
            yield retrieval(request.request_id)
            yield completed(request.request_id)

    with start_runtime_server(config(), PausingService()) as running:
        connection, response = post(running.endpoint)
        try:
            first_line = response.readline()
            assert json.loads(first_line)["type"] == "generation.started"
            release.set()
            remaining = [json.loads(line) for line in response.read().splitlines()]
            assert remaining[-1]["type"] == "generation.completed"
        finally:
            release.set()
            connection.close()


def test_client_disconnect_sets_the_service_cancellation_event() -> None:
    cancel_seen = Event()

    class CancellableService:
        def stream(
            self,
            request: RuntimeChatRequest,
            *,
            trusted_scope_id: str,
            cancellation: Event,
        ) -> Iterable[Mapping[str, Any]]:
            try:
                yield started(request.request_id)
                while not cancellation.wait(0.02):
                    yield {
                        "type": "heartbeat",
                        "request_id": request.request_id,
                        "phase": "prefill",
                    }
            finally:
                if cancellation.is_set():
                    cancel_seen.set()

    with start_runtime_server(config(), CancellableService()) as running:
        connection, response = post(running.endpoint)
        assert json.loads(response.readline())["type"] == "generation.started"
        response.close()
        connection.close()
        assert cancel_seen.wait(2)


def test_service_failures_become_a_sanitized_terminal_stream_error() -> None:
    class FailingService:
        def stream(
            self,
            request: RuntimeChatRequest,
            *,
            trusted_scope_id: str,
            cancellation: Event,
        ) -> Iterable[Mapping[str, Any]]:
            yield started(request.request_id)
            raise RuntimeError("C:/private/model/token=secret")

    with start_runtime_server(config(), FailingService()) as running:
        connection, response = post(running.endpoint)
        try:
            body = response.read().decode("utf-8")
        finally:
            connection.close()

    assert "private" not in body
    assert "secret" not in body
    events = [json.loads(line) for line in body.splitlines()]
    assert [event["type"] for event in events] == ["generation.started", "error"]
    assert events[-1]["message"] == "The local runtime could not complete generation."


def test_server_admission_is_bounded_and_released_after_completion() -> None:
    release = Event()

    class HoldingService:
        def stream(
            self,
            request: RuntimeChatRequest,
            *,
            trusted_scope_id: str,
            cancellation: Event,
        ) -> Iterable[Mapping[str, Any]]:
            yield started(request.request_id)
            release.wait(2)
            yield route(request.request_id)
            yield retrieval(request.request_id)
            yield completed(request.request_id)

    with start_runtime_server(config(max_concurrent_requests=1), HoldingService()) as running:
        first_connection, first_response = post(running.endpoint)
        assert json.loads(first_response.readline())["type"] == "generation.started"

        second_connection, second_response = post(
            running.endpoint,
            payload=request_bytes(request_id="runtime-00000000-0000-4000-8000-000000000002"),
        )
        try:
            assert second_response.status == HTTPStatus.SERVICE_UNAVAILABLE
            second_response.read()
        finally:
            second_connection.close()

        release.set()
        assert read_events(first_response)[-1]["type"] == "generation.completed"
        first_connection.close()

        third_connection, third_response = post(
            running.endpoint,
            payload=request_bytes(request_id="runtime-00000000-0000-4000-8000-000000000003"),
        )
        try:
            assert third_response.status == HTTPStatus.OK
            assert read_events(third_response)[-1]["type"] == "generation.completed"
        finally:
            third_connection.close()
