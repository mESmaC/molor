import asyncio
from dataclasses import replace

import pytest

from molor.memory import SQLiteMemoryStore
from molor.runtime.coordinator import (
    AbstainingMemoryGate,
    BoundedRetrievalRenderer,
    CapabilityRouteDecision,
    ChatMessage,
    GenerationRequest,
    RenderedRetrievalRecord,
    RuntimeCoordinator,
    StaticMemoryVersionResolver,
)
from molor.runtime.execution import (
    AdapterAxes,
    BackendDelta,
    BackendDescriptor,
    BackendFinished,
    BackendProgress,
    CancellationToken,
    DecodingConfig,
    ExecutionPlan,
    GenerationCancelled,
    GenerationCompleted,
    GenerationFailed,
    GenerationOrchestrator,
    GenerationStarted,
    Heartbeat,
    MetricsUpdated,
    PreparationTelemetry,
    PromptMessage,
    RouteSelected,
    TokenEmitted,
    TokenizerPromptRenderer,
)

from test_manifests import digest
from test_memory_store import job, observation, snapshot, version
from test_runtime_coordinator import SCOPE, ByteTokenCounter, bank, binding, coordinator


class ByteTokenizer:
    """Deterministic test tokenizer: each UTF-8 byte is one token."""

    fingerprint = digest("8")

    def __init__(self, context_limit: int = 4096) -> None:
        self.context_limit = context_limit

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def encode_chat(
        self,
        messages: tuple[PromptMessage, ...],
        *,
        add_generation_prompt: bool,
    ) -> tuple[int, ...]:
        rendered = "".join(
            f"<{message.role}>\n{message.content}\n</{message.role}>\n" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return tuple(rendered.encode("utf-8"))


class FakeClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def monotonic(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class ScriptedBackend:
    def __init__(
        self,
        descriptor: BackendDescriptor,
        clock: FakeClock,
        records: tuple[tuple[float, object], ...],
    ) -> None:
        self.descriptor = descriptor
        self.clock = clock
        self.records = records
        self.closed = False

    async def generate(self, plan: ExecutionPlan, cancellation: CancellationToken):
        try:
            for delay, record in self.records:
                cancellation.raise_if_cancelled()
                self.clock.advance(delay)
                await asyncio.sleep(0)
                if isinstance(record, Exception):
                    raise record
                yield record
        finally:
            self.closed = True


def prepared_request(
    *,
    messages: tuple[ChatMessage, ...] | None = None,
    memory_mode: str = "off",
):
    supplied_messages = messages or (ChatMessage("user", "hello"),)
    request = GenerationRequest(
        request_id="request/execution",
        conversation_id="conversation/execution",
        messages=supplied_messages,
        memory_mode=memory_mode,  # type: ignore[arg-type]
    )
    with SQLiteMemoryStore() as store:
        return coordinator(store).prepare(trusted_scope_id=SCOPE, request=request)


def descriptor_for(prepared, tokenizer: ByteTokenizer) -> BackendDescriptor:
    stack = prepared.serving_pin.model_stack
    return BackendDescriptor(
        backend_id="deterministic/byte-v1",
        model_id=stack.base.model_id,
        backend_mode="Deterministic byte backend",
        quantization=stack.base.quantization,
        device="cpu/test",
        context_limit=tokenizer.context_limit,
        tokenizer_fingerprint=tokenizer.fingerprint,
        stack_fingerprint=stack.fingerprint,
        metrics_estimated=False,
    )


def plan_for(
    prepared,
    *,
    tokenizer: ByteTokenizer | None = None,
    decoding: DecodingConfig | None = None,
) -> ExecutionPlan:
    selected_tokenizer = tokenizer or ByteTokenizer()
    selected_decoding = decoding or DecodingConfig(max_new_tokens=32)
    return ExecutionPlan.create(
        prepared=prepared,
        renderer=TokenizerPromptRenderer(selected_tokenizer),
        decoding=selected_decoding,
        backend=descriptor_for(prepared, selected_tokenizer),
    )


def retrieval_record(event_id: str, marker: str, text: str) -> RenderedRetrievalRecord:
    return RenderedRetrievalRecord(
        source_scope_id=SCOPE,
        source_event_id=event_id,
        source_event_fingerprint=digest(marker),
        score=1.0,
        text=text,
        token_count=len(text.encode("utf-8")),
        rendered_bytes=len(text.encode("utf-8")),
    )


def test_decoding_config_is_validated_and_fingerprinted() -> None:
    first = DecodingConfig(max_new_tokens=128, do_sample=True, temperature=0.7, seed=4)
    second = replace(first, seed=5)

    assert first.fingerprint != second.fingerprint
    with pytest.raises(ValueError, match="max_new_tokens"):
        DecodingConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="top_p"):
        DecodingConfig(top_p=0)
    with pytest.raises(ValueError, match="seed"):
        DecodingConfig(seed=-1)


def test_prompt_renderer_rejects_a_different_retrieval_tokenizer() -> None:
    prepared = prepared_request()

    class WrongTokenizer(ByteTokenizer):
        fingerprint = digest("9")

    with pytest.raises(ValueError, match="different tokenizers"):
        TokenizerPromptRenderer(WrongTokenizer()).render(
            prepared,
            DecodingConfig(max_new_tokens=16),
        )


def test_prompt_renderer_drops_whole_retrieval_records_before_history() -> None:
    messages = (
        ChatMessage("user", "old question"),
        ChatMessage("assistant", "old answer"),
        ChatMessage("user", "current question"),
    )
    prepared = prepared_request(messages=messages, memory_mode="auto")
    first = retrieval_record("event/first", "a", "first memory record")
    second = retrieval_record("event/second", "b", "second memory record")
    prepared = replace(prepared, retrieval_records=(first, second))
    decoding = DecodingConfig(max_new_tokens=16)

    generous = TokenizerPromptRenderer(ByteTokenizer(4096)).render(prepared, decoding)
    tight_limit = generous.input_tokens + decoding.max_new_tokens - 1
    tight = TokenizerPromptRenderer(ByteTokenizer(tight_limit)).render(prepared, decoding)

    assert tight.included_message_indices == (0, 1, 2)
    assert tight.dropped_message_indices == ()
    assert tight.included_retrieval_fingerprints == (first.fingerprint,)
    assert tight.dropped_retrieval_fingerprints == (second.fingerprint,)
    assert tight.input_tokens + decoding.max_new_tokens <= tight.context_limit
    assert "current question" not in str(tight.to_dict())


def test_prompt_renderer_drops_the_oldest_complete_history_turn() -> None:
    messages = (
        ChatMessage("user", "old question"),
        ChatMessage("assistant", "old answer"),
        ChatMessage("user", "newer question"),
        ChatMessage("assistant", "newer answer"),
        ChatMessage("user", "current question"),
    )
    prepared = prepared_request(messages=messages)
    decoding = DecodingConfig(max_new_tokens=16)
    generous = TokenizerPromptRenderer(ByteTokenizer(4096)).render(prepared, decoding)
    tight_limit = generous.input_tokens + decoding.max_new_tokens - 1

    tight = TokenizerPromptRenderer(ByteTokenizer(tight_limit)).render(prepared, decoding)

    assert tight.dropped_message_indices == (0, 1)
    assert tight.included_message_indices == (2, 3, 4)


def test_prompt_renderer_rejects_an_oversized_mandatory_prompt() -> None:
    prepared = prepared_request(messages=(ChatMessage("user", "x" * 100),))
    decoding = DecodingConfig(max_new_tokens=8)

    with pytest.raises(ValueError, match="mandatory"):
        TokenizerPromptRenderer(ByteTokenizer(32)).render(prepared, decoding)


def test_execution_plan_binds_prompt_axes_and_backend_identity() -> None:
    prepared = prepared_request()
    tokenizer = ByteTokenizer()
    plan = plan_for(prepared, tokenizer=tokenizer)

    assert plan.adapter_axes == AdapterAxes.from_prepared(prepared)
    assert plan.adapter_axes.capability.selected_adapter_fingerprints == ()
    assert plan.adapter_axes.memory_state == "off"
    assert plan.adapter_axes.memory is None
    assert "input_ids" not in plan.to_dict()["prompt"]

    wrong_backend = replace(
        descriptor_for(prepared, tokenizer),
        stack_fingerprint=digest("a"),
    )
    with pytest.raises(ValueError, match="different model stack"):
        ExecutionPlan.create(
            prepared=prepared,
            renderer=TokenizerPromptRenderer(tokenizer),
            decoding=DecodingConfig(max_new_tokens=16),
            backend=wrong_backend,
        )


def test_abstained_memory_axis_selects_only_null_without_materializing_adapter() -> None:
    source = observation("event/execution-source", 0, {"text": "remember this"})
    rollover = snapshot((source,))
    spec = job(rollover, cutoff=0)
    candidate = version(spec)
    with SQLiteMemoryStore() as store:
        store.append_event(source)
        store.record_rollover(rollover, spec, now_unix_ms=0)
        lease = store.lease_next("worker/execution", lease_seconds=10, now_unix_ms=100)
        assert lease is not None
        store.start_training(lease, now_unix_ms=200)
        store.start_evaluating(lease, now_unix_ms=300)
        store.register_candidate(lease, candidate, now_unix_ms=400)
        store.promote_candidate(spec.job_key, now_unix_ms=500)
        runtime = coordinator(
            store,
            resolver=StaticMemoryVersionResolver({candidate.fingerprint: candidate}),
        )
        prepared = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=GenerationRequest(
                request_id="request/memory-execution",
                conversation_id="conversation/execution",
                messages=(ChatMessage("user", "remember this"),),
            ),
        )

    axes = AdapterAxes.from_prepared(prepared)

    assert axes.memory_state == "abstained"
    assert axes.memory is not None
    assert axes.memory.selected_adapter_fingerprints == ()
    assert axes.memory.route_ids[axes.memory.selected_indices[0]] == "memory/null"


def test_forced_route_must_be_exact_one_hot() -> None:
    capability_bank = bank()
    router_binding = binding(capability_bank)

    with pytest.raises(ValueError, match="one-hot"):
        CapabilityRouteDecision(
            selector_fingerprint=digest("6"),
            selection_mode="forced",
            router_binding_fingerprint=router_binding.fingerprint,
            bank_fingerprint=capability_bank.fingerprint,
            route_ids=capability_bank.route_ids,
            weights=(0.0, 0.0, 0.5),
            selected_indices=(2,),
        )


def test_normalized_learned_route_must_sum_to_one() -> None:
    capability_bank = bank()
    router_binding = binding(capability_bank)

    class UnderweightSelector:
        fingerprint = digest("6")

        def select(self, request: GenerationRequest) -> CapabilityRouteDecision:
            return CapabilityRouteDecision(
                selector_fingerprint=self.fingerprint,
                selection_mode="learned",
                router_binding_fingerprint=router_binding.fingerprint,
                bank_fingerprint=capability_bank.fingerprint,
                route_ids=capability_bank.route_ids,
                weights=(0.2, 0.2, 0.0),
                selected_indices=(0, 1),
            )

    with SQLiteMemoryStore() as store:
        runtime = RuntimeCoordinator(
            capability_bank=capability_bank,
            router_binding=router_binding,
            route_selector=UnderweightSelector(),
            memory_store=store,
            version_resolver=StaticMemoryVersionResolver(),
            memory_gate=AbstainingMemoryGate(),
            retrieval_renderer=BoundedRetrievalRenderer(ByteTokenCounter()),
        )
        with pytest.raises(ValueError, match="sum to one"):
            runtime.prepare(
                trusted_scope_id=SCOPE,
                request=GenerationRequest(
                    request_id="request/underweight-route",
                    conversation_id="conversation/execution",
                    messages=(ChatMessage("user", "route me"),),
                    memory_mode="off",
                ),
            )


def test_stream_orchestrator_emits_exact_order_and_metrics() -> None:
    prepared = prepared_request()
    tokenizer = ByteTokenizer()
    plan = plan_for(prepared, tokenizer=tokenizer)
    clock = FakeClock()
    backend = ScriptedBackend(
        plan.backend,
        clock,
        (
            (0.0, BackendProgress("loading")),
            (0.1, BackendDelta("Hi ", tuple(b"Hi "))),
            (0.1, BackendDelta("🌍", tuple("🌍".encode("utf-8")))),
            (0.0, BackendFinished("stop")),
        ),
    )
    orchestrator = GenerationOrchestrator(backend, clock=clock)

    async def collect():
        return [
            event
            async for event in orchestrator.stream(
                plan,
                cancellation=CancellationToken(),
                preparation=PreparationTelemetry(2.5, "idle"),
            )
        ]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        "generation.started",
        "route",
        "retrieval",
        "heartbeat",
        "token",
        "metrics",
        "token",
        "metrics",
        "generation.completed",
    ]
    assert isinstance(events[0], GenerationStarted)
    assert isinstance(events[1], RouteSelected)
    assert isinstance(events[3], Heartbeat)
    tokens = [event for event in events if isinstance(event, TokenEmitted)]
    assert [event.index for event in tokens] == [0, 1]
    assert tokens[0].ttft_ms == pytest.approx(100.0)
    metrics = [event for event in events if isinstance(event, MetricsUpdated)]
    assert [event.completion_tokens for event in metrics] == [3, 7]
    assert metrics[-1].context_used == plan.prompt.input_tokens + 7
    completed = events[-1]
    assert isinstance(completed, GenerationCompleted)
    assert completed.completion_tokens == 7
    assert completed.tokens_per_second == pytest.approx(35.0)
    assert backend.closed


def test_cancellation_closes_the_backend_without_a_terminal_event() -> None:
    prepared = prepared_request()
    plan = plan_for(prepared)
    clock = FakeClock()
    backend = ScriptedBackend(
        plan.backend,
        clock,
        (
            (0.1, BackendDelta("first", tuple(b"first"))),
            (0.1, BackendDelta("second", tuple(b"second"))),
            (0.0, BackendFinished("stop")),
        ),
    )
    cancellation = CancellationToken()
    orchestrator = GenerationOrchestrator(backend, clock=clock)

    async def cancel_after_first_token():
        stream = orchestrator.stream(
            plan,
            cancellation=cancellation,
            preparation=PreparationTelemetry(0, "idle"),
        )
        seen = []
        while True:
            event = await anext(stream)
            seen.append(event)
            if isinstance(event, TokenEmitted):
                cancellation.cancel()
                break
        with pytest.raises(GenerationCancelled):
            await anext(stream)
        return seen

    events = asyncio.run(cancel_after_first_token())

    assert any(isinstance(event, TokenEmitted) for event in events)
    assert not any(isinstance(event, (GenerationCompleted, GenerationFailed)) for event in events)
    assert backend.closed


def test_backend_failures_are_sanitized_and_terminal() -> None:
    prepared = prepared_request()
    plan = plan_for(prepared)
    clock = FakeClock()
    backend = ScriptedBackend(
        plan.backend,
        clock,
        ((0.0, RuntimeError("secret backend path")),),
    )
    orchestrator = GenerationOrchestrator(backend, clock=clock)

    async def collect():
        return [
            event
            async for event in orchestrator.stream(
                plan,
                cancellation=CancellationToken(),
                preparation=PreparationTelemetry(0, "idle"),
            )
        ]

    events = asyncio.run(collect())

    assert isinstance(events[-1], GenerationFailed)
    assert "secret" not in events[-1].message
    assert sum(isinstance(event, GenerationFailed) for event in events) == 1
    assert backend.closed


def test_backend_cannot_overrun_the_output_reservation() -> None:
    prepared = prepared_request()
    plan = plan_for(prepared, decoding=DecodingConfig(max_new_tokens=4))
    clock = FakeClock()
    backend = ScriptedBackend(
        plan.backend,
        clock,
        (
            (0.1, BackendDelta("too many", tuple(b"12345"))),
            (0.0, BackendFinished("stop")),
        ),
    )

    async def collect():
        return [
            event
            async for event in GenerationOrchestrator(backend, clock=clock).stream(
                plan,
                cancellation=CancellationToken(),
                preparation=PreparationTelemetry(0, "idle"),
            )
        ]

    events = asyncio.run(collect())

    assert isinstance(events[-1], GenerationFailed)
    assert not any(isinstance(event, TokenEmitted) for event in events)
