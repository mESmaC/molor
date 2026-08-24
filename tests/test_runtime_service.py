from contextlib import contextmanager
import threading

import pytest

from molor.memory import SQLiteMemoryStore
from molor.runtime.coordinator import (
    AbstainingMemoryGate,
    BoundedRetrievalRenderer,
    ChatMessage,
    GenerationRequest,
    NullCapabilityRouteSelector,
    RuntimeCoordinator,
    StaticMemoryVersionResolver,
)
from molor.runtime.deterministic import (
    ByteRuntimeTokenizer,
    DeterministicGenerationBackend,
)
from molor.runtime.execution import (
    BackendDescriptor,
    CancellationToken,
    DecodingConfig,
    PreparationTelemetry,
    TokenizerPromptRenderer,
)
from molor.runtime.http import (
    ChatStreamSequenceValidator,
    RuntimeChatMessage,
    RuntimeChatRequest,
)
from molor.runtime.manifests import ModelStackManifest
from molor.runtime.service import CoordinatedGenerationService

from test_runtime_coordinator import SCOPE, bank, binding


class RecordingCoordinatorFactory:
    def __init__(self, tokenizer: ByteRuntimeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.entered = 0
        self.exited = 0
        self.calls: list[tuple[str, GenerationRequest]] = []
        self.prepared = []

    def __call__(self):
        @contextmanager
        def opened():
            self.entered += 1
            try:
                with SQLiteMemoryStore() as store:
                    capability_bank = bank()
                    router_binding = binding(capability_bank)
                    runtime = RuntimeCoordinator(
                        capability_bank=capability_bank,
                        router_binding=router_binding,
                        route_selector=NullCapabilityRouteSelector(
                            capability_bank,
                            router_binding,
                        ),
                        memory_store=store,
                        version_resolver=StaticMemoryVersionResolver(),
                        memory_gate=AbstainingMemoryGate(),
                        retrieval_renderer=BoundedRetrievalRenderer(self.tokenizer),
                    )

                    class RecordingCoordinator:
                        def prepare(inner_self, *, trusted_scope_id, request):
                            self.calls.append((trusted_scope_id, request))
                            prepared = runtime.prepare(
                                trusted_scope_id=trusted_scope_id,
                                request=request,
                            )
                            self.prepared.append(prepared)
                            return prepared

                    yield RecordingCoordinator()
            finally:
                self.exited += 1

        return opened()


class RecordingBackend:
    def __init__(self, backend: DeterministicGenerationBackend) -> None:
        self.backend = backend
        self.descriptor = backend.descriptor
        self.started = 0
        self.closed = 0
        self.cancelled_at_close: list[bool] = []

    async def generate(self, plan, cancellation: CancellationToken):
        self.started += 1
        try:
            async for record in self.backend.generate(plan, cancellation):
                yield record
        finally:
            self.closed += 1
            self.cancelled_at_close.append(cancellation.is_cancelled)


class RecordingTelemetryProvider:
    def __init__(self) -> None:
        self.elapsed_ms: list[float] = []

    def __call__(self, prepared, preparation_elapsed_ms):
        self.elapsed_ms.append(preparation_elapsed_ms)
        return PreparationTelemetry(
            retrieval_latency_ms=preparation_elapsed_ms,
            consolidation_state="service-ready",
        )


class StepMonotonic:
    def __init__(self, *, start: float = 10.0, step: float = 0.025) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


def chat_request(
    request_id: str = "request/service",
    *,
    memory: str = "off",
    messages: tuple[RuntimeChatMessage, ...] | None = None,
) -> RuntimeChatRequest:
    return RuntimeChatRequest(
        request_id=request_id,
        conversation_id="conversation/service",
        messages=messages or (RuntimeChatMessage("user", "hello service"),),
        memory=memory,  # type: ignore[arg-type]
    )


def service_components(*, response_text: str = "service ready"):
    tokenizer = ByteRuntimeTokenizer(context_limit=4096)
    factory = RecordingCoordinatorFactory(tokenizer)
    capability_bank = bank()
    stack = ModelStackManifest(capability_bank)
    descriptor = BackendDescriptor(
        backend_id="deterministic/service-v1",
        model_id=stack.base.model_id,
        backend_mode="Deterministic service backend",
        quantization=stack.base.quantization,
        device="cpu/test",
        context_limit=tokenizer.context_limit,
        tokenizer_fingerprint=tokenizer.fingerprint,
        stack_fingerprint=stack.fingerprint,
    )
    backend = RecordingBackend(
        DeterministicGenerationBackend(
            descriptor,
            response_text=response_text,
            chunk_tokens=4,
        )
    )
    telemetry = RecordingTelemetryProvider()
    service = CoordinatedGenerationService(
        coordinator_factory=factory,
        renderer=TokenizerPromptRenderer(tokenizer),
        decoding=DecodingConfig(max_new_tokens=64),
        backend=backend,
        telemetry_provider=telemetry,
        monotonic=StepMonotonic(),
    )
    return service, factory, backend, telemetry


def assert_valid_sequence(events, request_id: str) -> None:
    validator = ChatStreamSequenceValidator(request_id)
    for event in events:
        validator.accept(event)
    assert validator.terminal


def test_service_streams_typed_events_through_the_complete_boundary() -> None:
    service, factory, backend, telemetry = service_components(response_text="ready 🌍")
    request = chat_request()

    events = list(
        service.stream(
            request,
            trusted_scope_id=SCOPE,
            cancellation=threading.Event(),
        )
    )

    assert_valid_sequence(events, request.request_id)
    assert [event["type"] for event in events[:6]] == [
        "generation.started",
        "route",
        "retrieval",
        "heartbeat",
        "heartbeat",
        "heartbeat",
    ]
    assert events[-1]["type"] == "generation.completed"
    assert "".join(event["text"] for event in events if event["type"] == "token") == ("ready 🌍")
    assert events[-1]["completion_tokens"] == len("ready 🌍".encode("utf-8"))
    assert events[2]["latency_ms"] == pytest.approx(25.0)
    assert telemetry.elapsed_ms == pytest.approx([25.0])
    assert factory.entered == factory.exited == 1
    assert backend.started == backend.closed == 1


def test_service_maps_the_exact_http_request_without_accepting_scope_from_content() -> None:
    service, factory, _, _ = service_components()
    request = chat_request(
        messages=(
            RuntimeChatMessage("user", "first"),
            RuntimeChatMessage("assistant", "reply"),
            RuntimeChatMessage("user", "final"),
        )
    )

    list(
        service.stream(
            request,
            trusted_scope_id=SCOPE,
            cancellation=threading.Event(),
        )
    )

    trusted_scope_id, mapped = factory.calls[0]
    assert trusted_scope_id == SCOPE
    assert mapped.request_id == request.request_id
    assert mapped.conversation_id == request.conversation_id
    assert mapped.memory_mode == request.memory
    assert mapped.retrieval_limit == 8
    assert mapped.messages == (
        ChatMessage("user", "first"),
        ChatMessage("assistant", "reply"),
        ChatMessage("user", "final"),
    )
    assert "scope" not in mapped.to_dict()


def test_service_distinguishes_unavailable_and_disabled_memory() -> None:
    service, factory, _, _ = service_components()

    automatic = list(
        service.stream(
            chat_request("request/auto", memory="auto"),
            trusted_scope_id=SCOPE,
            cancellation=threading.Event(),
        )
    )
    disabled = list(
        service.stream(
            chat_request("request/off", memory="off"),
            trusted_scope_id=SCOPE,
            cancellation=threading.Event(),
        )
    )

    automatic_route = next(event for event in automatic if event["type"] == "route")
    disabled_route = next(event for event in disabled if event["type"] == "route")
    assert automatic_route["memory"] == {"mode": "unavailable", "version": None}
    assert disabled_route["memory"] == {"mode": "off", "version": None}
    assert factory.prepared[0].retrieval_pin is not None
    assert factory.prepared[1].retrieval_pin is None


def test_external_cancellation_propagates_and_closes_generation() -> None:
    service, factory, backend, _ = service_components(response_text="several output chunks")
    cancellation = threading.Event()
    stream = service.stream(
        chat_request("request/cancel"),
        trusted_scope_id=SCOPE,
        cancellation=cancellation,
    )
    seen = []
    for event in stream:
        seen.append(event)
        if event["type"] == "token":
            cancellation.set()
            break

    remaining = list(stream)

    assert remaining == []
    assert cancellation.is_set()
    assert not any(event["type"] in ("error", "generation.completed") for event in seen)
    assert factory.entered == factory.exited == 1
    assert backend.started == backend.closed == 1
    assert backend.cancelled_at_close == [True]


def test_preparation_failure_is_sanitized_and_context_is_closed() -> None:
    class FailingFactory:
        def __init__(self) -> None:
            self.exited = False

        def __call__(self):
            @contextmanager
            def opened():
                try:

                    class FailingCoordinator:
                        def prepare(self, **kwargs):
                            raise RuntimeError("secret database path")

                    yield FailingCoordinator()
                finally:
                    self.exited = True

            return opened()

    baseline_service, _, backend, telemetry = service_components()
    factory = FailingFactory()
    service = CoordinatedGenerationService(
        coordinator_factory=factory,
        renderer=baseline_service.renderer,
        decoding=baseline_service.decoding,
        backend=backend,
        telemetry_provider=telemetry,
    )

    events = list(
        service.stream(
            chat_request("request/failure"),
            trusted_scope_id=SCOPE,
            cancellation=threading.Event(),
        )
    )

    assert events == [
        {
            "type": "error",
            "request_id": "request/failure",
            "message": "The local generation failed.",
        }
    ]
    assert "secret" not in str(events)
    assert factory.exited
    assert backend.started == backend.closed == 0


def test_nonfinite_preparation_duration_fails_before_telemetry_or_execution() -> None:
    baseline, factory, backend, telemetry = service_components()
    values = iter((-1e308, 1e308))
    service = CoordinatedGenerationService(
        coordinator_factory=factory,
        renderer=baseline.renderer,
        decoding=baseline.decoding,
        backend=backend,
        telemetry_provider=telemetry,
        monotonic=lambda: next(values),
    )

    events = list(
        service.stream(
            chat_request("request/clock-overflow"),
            trusted_scope_id=SCOPE,
            cancellation=threading.Event(),
        )
    )

    assert events == [
        {
            "type": "error",
            "request_id": "request/clock-overflow",
            "message": "The local generation failed.",
        }
    ]
    assert factory.entered == factory.exited == 1
    assert telemetry.elapsed_ms == []
    assert backend.started == backend.closed == 0


def test_consumer_close_cancels_backend_releases_lock_and_allows_reuse() -> None:
    service, factory, backend, _ = service_components(response_text="many chunks for cleanup")
    cancellation = threading.Event()
    stream = service.stream(
        chat_request("request/close"),
        trusted_scope_id=SCOPE,
        cancellation=cancellation,
    )
    while True:
        event = next(stream)
        if event["type"] == "token":
            break

    stream.close()

    assert cancellation.is_set()
    assert backend.started == backend.closed == 1
    assert backend.cancelled_at_close == [True]

    replay = list(
        service.stream(
            chat_request("request/reuse"),
            trusted_scope_id=SCOPE,
            cancellation=threading.Event(),
        )
    )
    assert replay[-1]["type"] == "generation.completed"
    assert factory.entered == factory.exited == 2
    assert backend.started == backend.closed == 2
