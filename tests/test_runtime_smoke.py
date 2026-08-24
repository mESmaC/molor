from __future__ import annotations

from http import HTTPStatus
from http.client import HTTPConnection
from io import StringIO
import json
from threading import Event
from urllib.parse import urlsplit

import pytest

from molor.runtime.deterministic import (
    ByteRuntimeTokenizer,
    DeterministicGenerationBackend,
)
from molor.runtime.http import (
    CHAT_REQUEST_SCHEMA,
    CHAT_STREAM_MEDIA_TYPE,
    CHAT_STREAM_PROTOCOL,
    RuntimeChatMessage,
    RuntimeChatRequest,
    RuntimeHttpConfig,
)
from molor.runtime.service import CoordinatedGenerationService
from molor.runtime.smoke import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SCOPE_ID,
    SMOKE_RESPONSE,
    SmokeConfigurationError,
    build_smoke_capability_bank,
    build_smoke_router_binding,
    build_smoke_service,
    load_smoke_http_config,
    main,
    start_smoke_server,
)


TOKEN = "local-runtime-token-0123456789abcdef"
REQUEST_ID = "runtime-00000000-0000-4000-8000-000000000010"


def runtime_request(memory: str = "auto") -> RuntimeChatRequest:
    return RuntimeChatRequest(
        request_id=REQUEST_ID,
        conversation_id="conversation/smoke",
        messages=(RuntimeChatMessage("user", "Exercise the local runtime"),),
        memory=memory,  # type: ignore[arg-type]
    )


def wire_request() -> bytes:
    return json.dumps(
        {
            "schema_version": CHAT_REQUEST_SCHEMA,
            "request_id": REQUEST_ID,
            "conversation_id": "conversation/smoke",
            "messages": [{"role": "user", "content": "Exercise the local runtime"}],
            "memory": "auto",
        },
        separators=(",", ":"),
    ).encode()


def test_smoke_manifests_are_stable_and_explicitly_synthetic() -> None:
    first = build_smoke_capability_bank()
    replay = build_smoke_capability_bank()
    binding = build_smoke_router_binding(first)

    assert first == replay
    assert first.fingerprint == replay.fingerprint
    assert first.base.model_id == "molor/deterministic-smoke-no-model"
    assert first.route_ids == ("smoke/unloaded-expert", "base/null")
    assert all("smoke/" in artifact.logical_name for artifact in first.base.weights)
    assert "unloaded" in first.adapters[0].adapter_id
    binding.assert_matches(first)
    assert "smoke/" in binding.checkpoint.logical_name


def test_smoke_service_composes_the_real_coordinator_and_deterministic_backend() -> None:
    service = build_smoke_service()

    assert isinstance(service, CoordinatedGenerationService)
    assert isinstance(service.renderer.tokenizer, ByteRuntimeTokenizer)
    assert isinstance(service.backend, DeterministicGenerationBackend)
    assert "DETERMINISTIC SMOKE" in service.backend.descriptor.backend_mode
    assert "NO MODEL WEIGHTS" in service.backend.descriptor.backend_mode
    assert service.backend.response_text == SMOKE_RESPONSE
    assert (
        service.renderer.tokenizer_fingerprint == service.backend.descriptor.tokenizer_fingerprint
    )

    with service.coordinator_factory() as coordinator:
        assert coordinator.capability_bank.route_ids[-1] == "base/null"
        assert coordinator.retrieval_renderer.tokenizer_fingerprint == (
            service.renderer.tokenizer_fingerprint
        )


@pytest.mark.parametrize("memory", ["auto", "off"])
def test_smoke_service_runs_the_full_preparation_and_execution_path(memory: str) -> None:
    service = build_smoke_service()
    events = list(
        service.stream(
            runtime_request(memory),
            trusted_scope_id=DEFAULT_SCOPE_ID,
            cancellation=Event(),
        )
    )

    assert events[0]["type"] == "generation.started"
    assert events[0]["runtime_source"] == "local"
    assert events[0]["backend_mode"] == "DETERMINISTIC SMOKE - NO MODEL WEIGHTS"
    assert events[1] == {
        "type": "route",
        "request_id": REQUEST_ID,
        "capability": [{"id": "base/null", "weight": 1.0}],
        "memory": {
            "mode": "unavailable" if memory == "auto" else "off",
            "version": None,
        },
    }
    assert events[2]["type"] == "retrieval"
    assert events[2]["consolidation_state"] == "deterministic-smoke/no-memory-worker"
    visible_text = "".join(event["text"] for event in events if event["type"] == "token")
    assert visible_text == SMOKE_RESPONSE
    assert events[-1]["type"] == "generation.completed"
    assert events[-1]["finish_reason"] == "stop"


def test_smoke_server_is_accepted_by_the_authenticated_ndjson_transport() -> None:
    config = RuntimeHttpConfig(
        token=TOKEN,
        trusted_scope_id=DEFAULT_SCOPE_ID,
        port=0,
    )
    with start_smoke_server(config) as running:
        endpoint = urlsplit(running.endpoint)
        assert endpoint.hostname is not None and endpoint.port is not None
        connection = HTTPConnection(endpoint.hostname, endpoint.port, timeout=5)
        connection.request(
            "POST",
            endpoint.path,
            body=wire_request(),
            headers={
                "Accept": CHAT_STREAM_MEDIA_TYPE,
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "X-Molor-Protocol": CHAT_STREAM_PROTOCOL,
            },
        )
        response = connection.getresponse()
        try:
            body = response.read().decode()
        finally:
            connection.close()

    assert response.status == HTTPStatus.OK
    assert response.getheader("X-Molor-Protocol") == CHAT_STREAM_PROTOCOL
    events = [json.loads(line) for line in body.splitlines()]
    assert events[0]["backend_mode"] == "DETERMINISTIC SMOKE - NO MODEL WEIGHTS"
    assert events[1]["memory"] == {"mode": "unavailable", "version": None}
    assert "".join(event["text"] for event in events if event["type"] == "token") == (
        SMOKE_RESPONSE
    )
    assert events[-1]["type"] == "generation.completed"


def test_smoke_config_requires_only_an_environment_bearer_secret() -> None:
    selected = load_smoke_http_config({"MOLOR_RUNTIME_TOKEN": TOKEN})

    assert selected.token == TOKEN
    assert selected.host == DEFAULT_HOST
    assert selected.port == DEFAULT_PORT
    assert selected.trusted_scope_id == DEFAULT_SCOPE_ID

    with pytest.raises(SmokeConfigurationError, match="required"):
        load_smoke_http_config({})
    with pytest.raises(SmokeConfigurationError, match="PORT"):
        load_smoke_http_config(
            {
                "MOLOR_RUNTIME_TOKEN": TOKEN,
                "MOLOR_RUNTIME_PORT": "８７６５",
            }
        )
    with pytest.raises(SmokeConfigurationError, match="literal loopback"):
        load_smoke_http_config(
            {
                "MOLOR_RUNTIME_TOKEN": TOKEN,
                "MOLOR_RUNTIME_HOST": "localhost",
            }
        )


def test_entrypoint_rejects_all_cli_configuration_without_echoing_it() -> None:
    output = StringIO()
    errors = StringIO()

    result = main(
        ["--token", TOKEN],
        environment={},
        stdout=output,
        stderr=errors,
    )

    assert result == 2
    assert TOKEN not in output.getvalue()
    assert TOKEN not in errors.getvalue()
    assert "accepts no arguments" in errors.getvalue()


def test_entrypoint_cleanly_closes_the_server_on_keyboard_interrupt() -> None:
    class InterruptingThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float) -> None:
            raise KeyboardInterrupt

    class FakeRunningServer:
        endpoint = "http://127.0.0.1:8765/v1/chat/stream"
        thread = InterruptingThread()

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    captured: dict[str, object] = {}
    fake = FakeRunningServer()

    def starter(config: RuntimeHttpConfig, service: CoordinatedGenerationService):
        captured["config"] = config
        captured["service"] = service
        return fake  # type: ignore[return-value]

    output = StringIO()
    errors = StringIO()
    result = main(
        [],
        environment={"MOLOR_RUNTIME_TOKEN": TOKEN},
        stdout=output,
        stderr=errors,
        server_starter=starter,
    )

    assert result == 0
    assert fake.closed
    assert isinstance(captured["service"], CoordinatedGenerationService)
    assert "NO MODEL WEIGHTS LOADED" in output.getvalue()
    assert "Stopping" in output.getvalue()
    assert TOKEN not in output.getvalue()
    assert TOKEN not in errors.getvalue()
