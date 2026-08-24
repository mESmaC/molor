import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest

from molor.runtime.deterministic import (
    ByteRuntimeTokenizer,
    DeterministicGenerationBackend,
)
from molor.runtime.execution import (
    BackendDelta,
    BackendDescriptor,
    BackendFinished,
    BackendProgress,
    CancellationToken,
    DecodingConfig,
    ExecutionPlan,
    GenerationCancelled,
    GenerationCompleted,
    GenerationOrchestrator,
    PreparationTelemetry,
    PromptMessage,
    TokenEmitted,
    TokenizerPromptRenderer,
)

from test_runtime_execution import prepared_request


def plan_and_tokenizer(*, max_new_tokens: int = 64, context_limit: int = 4096):
    tokenizer = ByteRuntimeTokenizer(context_limit=context_limit)
    prepared = replace(
        prepared_request(),
        retrieval_tokenizer_fingerprint=tokenizer.fingerprint,
    )
    stack = prepared.serving_pin.model_stack
    descriptor = BackendDescriptor(
        backend_id="deterministic/byte-v1",
        model_id=stack.base.model_id,
        backend_mode="Deterministic byte backend",
        quantization=stack.base.quantization,
        device="cpu/test",
        context_limit=context_limit,
        tokenizer_fingerprint=tokenizer.fingerprint,
        stack_fingerprint=stack.fingerprint,
    )
    plan = ExecutionPlan.create(
        prepared=prepared,
        renderer=TokenizerPromptRenderer(tokenizer),
        decoding=DecodingConfig(max_new_tokens=max_new_tokens),
        backend=descriptor,
    )
    return plan, tokenizer


async def collect_backend(backend, plan, cancellation=None):
    selected_cancellation = cancellation or CancellationToken()
    return [record async for record in backend.generate(plan, selected_cancellation)]


def test_byte_tokenizer_has_stable_identity_and_exact_unicode_accounting() -> None:
    first = ByteRuntimeTokenizer(context_limit=1024)
    second = ByteRuntimeTokenizer(context_limit=1024)
    changed_context = ByteRuntimeTokenizer(context_limit=2048)
    text = "hello 🌍"

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed_context.fingerprint
    assert first.count_tokens(text) == len(text.encode("utf-8"))
    assert first.encode_text(text) == tuple(text.encode("utf-8"))
    assert first.decode_tokens(first.encode_text(text)) == text
    with pytest.raises(FrozenInstanceError):
        first.context_limit = 1  # type: ignore[misc]


def test_byte_chat_encoding_is_canonical_and_generation_marker_is_bound() -> None:
    tokenizer = ByteRuntimeTokenizer()
    messages = (
        PromptMessage("system", "system policy"),
        PromptMessage("user", "hello"),
    )

    first = tokenizer.encode_chat(messages, add_generation_prompt=True)
    replay = tokenizer.encode_chat(tuple(messages), add_generation_prompt=True)
    no_generation_marker = tokenizer.encode_chat(
        messages,
        add_generation_prompt=False,
    )

    assert first == replay
    assert first != no_generation_marker
    assert bytes(first).decode("utf-8").startswith("{")
    with pytest.raises(ValueError, match="must not be empty"):
        tokenizer.encode_chat((), add_generation_prompt=True)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        tokenizer.decode_tokens((256,))


def test_byte_tokenizer_and_backend_configuration_are_bounded() -> None:
    with pytest.raises(ValueError, match="context_limit"):
        ByteRuntimeTokenizer(context_limit=0)

    plan, _ = plan_and_tokenizer()
    with pytest.raises(ValueError, match="non-empty"):
        DeterministicGenerationBackend(plan.backend, response_text="")
    with pytest.raises(ValueError, match="chunk_tokens"):
        DeterministicGenerationBackend(plan.backend, chunk_tokens=0)
    with pytest.raises(ValueError, match="control"):
        DeterministicGenerationBackend(plan.backend, response_text="bad\x00text")


def test_deterministic_backend_emits_progress_exact_bytes_and_stop() -> None:
    plan, _ = plan_and_tokenizer(max_new_tokens=64)
    response = "Ready 🌍"
    backend = DeterministicGenerationBackend(
        plan.backend,
        response_text=response,
        chunk_tokens=2,
    )

    records = asyncio.run(collect_backend(backend, plan))
    progress = [record.phase for record in records if isinstance(record, BackendProgress)]
    deltas = [record for record in records if isinstance(record, BackendDelta)]
    finish = records[-1]

    assert progress == ["loading", "prefill", "decoding"]
    assert b"".join(bytes(delta.token_ids) for delta in deltas) == response.encode("utf-8")
    assert "".join(delta.text for delta in deltas) == response
    assert all(len(delta.token_ids) <= 2 for delta in deltas)
    assert isinstance(finish, BackendFinished)
    assert finish.finish_reason == "stop"
    assert backend.fingerprint == replace(backend).fingerprint
    with pytest.raises(FrozenInstanceError):
        backend.response_text = "changed"  # type: ignore[misc]


def test_deterministic_backend_truncates_to_exact_max_new_tokens() -> None:
    plan, _ = plan_and_tokenizer(max_new_tokens=3)
    backend = DeterministicGenerationBackend(
        plan.backend,
        response_text="A🌍Z",
        chunk_tokens=2,
    )

    records = asyncio.run(collect_backend(backend, plan))
    deltas = [record for record in records if isinstance(record, BackendDelta)]
    finish = records[-1]

    assert sum(delta.token_count for delta in deltas) == 3
    assert b"".join(bytes(delta.token_ids) for delta in deltas) == "A🌍Z".encode("utf-8")[:3]
    assert "".join(delta.text for delta in deltas) == "A"
    assert isinstance(finish, BackendFinished)
    assert finish.finish_reason == "length"


def test_deterministic_backend_honors_cancellation_between_records() -> None:
    plan, _ = plan_and_tokenizer()
    backend = DeterministicGenerationBackend(plan.backend)
    cancellation = CancellationToken()

    async def cancel_after_first_record():
        stream = backend.generate(plan, cancellation)
        first = await anext(stream)
        cancellation.cancel()
        with pytest.raises(GenerationCancelled):
            await anext(stream)
        return first

    first = asyncio.run(cancel_after_first_record())

    assert isinstance(first, BackendProgress)
    assert first.phase == "loading"


def test_deterministic_backend_rejects_a_different_descriptor() -> None:
    plan, _ = plan_and_tokenizer()
    changed_descriptor = replace(plan.backend, backend_id="deterministic/other")
    backend = DeterministicGenerationBackend(changed_descriptor)

    async def first_record():
        return await anext(backend.generate(plan, CancellationToken()))

    with pytest.raises(ValueError, match="different backend"):
        asyncio.run(first_record())


def test_deterministic_backend_runs_through_the_real_orchestrator() -> None:
    response = "smoke ready"
    plan, _ = plan_and_tokenizer(max_new_tokens=64)
    backend = DeterministicGenerationBackend(
        plan.backend,
        response_text=response,
        chunk_tokens=3,
    )
    orchestrator = GenerationOrchestrator(backend)

    async def collect_events():
        return [
            event
            async for event in orchestrator.stream(
                plan,
                cancellation=CancellationToken(),
                preparation=PreparationTelemetry(0, "deterministic-smoke"),
            )
        ]

    events = asyncio.run(collect_events())
    visible_text = "".join(event.text for event in events if isinstance(event, TokenEmitted))
    completed = events[-1]

    assert visible_text == response
    assert isinstance(completed, GenerationCompleted)
    assert completed.completion_tokens == len(response.encode("utf-8"))
    assert completed.finish_reason == "stop"
