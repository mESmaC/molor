"""Coordinator-to-executor integration behind the local HTTP transport.

The HTTP layer is deliberately synchronous and dependency-free while generation backends
use an async iterator.  This module owns that narrow bridge, creates one coordinator lease
per request, and serializes model execution so mutable adapter state cannot bleed between
concurrent requests.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
import threading
import time
from typing import AsyncIterator, Callable, Iterator, Protocol

from molor.runtime.coordinator import ChatMessage, GenerationRequest, PreparedGeneration
from molor.runtime.execution import (
    CancellationToken,
    DecodingConfig,
    ExecutionPlan,
    GenerationBackend,
    GenerationOrchestrator,
    PreparationTelemetry,
    RuntimeStreamEvent,
    TokenizerPromptRenderer,
)
from molor.runtime.http import RuntimeChatRequest


def _finite_nonnegative(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(value)


class CoordinatorFactory(Protocol):
    """Open a request-local coordinator and close its store after preparation."""

    def __call__(self) -> AbstractContextManager[object]: ...


class PreparationTelemetryProvider(Protocol):
    """Supply measured, non-fingerprinted preparation telemetry."""

    def __call__(
        self,
        prepared: PreparedGeneration,
        preparation_elapsed_ms: float,
    ) -> PreparationTelemetry: ...


@dataclass(frozen=True, slots=True)
class StaticPreparationTelemetry:
    """Explicit placeholder for runtimes without retrieval instrumentation yet."""

    retrieval_latency_ms: float = 0.0
    consolidation_state: str = "Not instrumented"

    def __post_init__(self) -> None:
        PreparationTelemetry(
            retrieval_latency_ms=self.retrieval_latency_ms,
            consolidation_state=self.consolidation_state,
        )

    def __call__(
        self,
        prepared: PreparedGeneration,
        preparation_elapsed_ms: float,
    ) -> PreparationTelemetry:
        if not isinstance(prepared, PreparedGeneration):
            raise TypeError("prepared must be a PreparedGeneration")
        _finite_nonnegative(preparation_elapsed_ms, "preparation_elapsed_ms")
        return PreparationTelemetry(
            retrieval_latency_ms=self.retrieval_latency_ms,
            consolidation_state=self.consolidation_state,
        )


class CoordinatedGenerationService:
    """Prepare, render, and execute bridge requests under one serialized model lease."""

    def __init__(
        self,
        *,
        coordinator_factory: CoordinatorFactory,
        renderer: TokenizerPromptRenderer,
        decoding: DecodingConfig,
        backend: GenerationBackend,
        telemetry_provider: PreparationTelemetryProvider,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(coordinator_factory):
            raise TypeError("coordinator_factory must be callable")
        if not isinstance(renderer, TokenizerPromptRenderer):
            raise TypeError("renderer must be a TokenizerPromptRenderer")
        if not isinstance(decoding, DecodingConfig):
            raise TypeError("decoding must be a DecodingConfig")
        if not callable(telemetry_provider):
            raise TypeError("telemetry_provider must be callable")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self.coordinator_factory = coordinator_factory
        self.renderer = renderer
        self.decoding = decoding
        self.backend = backend
        self.telemetry_provider = telemetry_provider
        self.monotonic = monotonic
        self.orchestrator = GenerationOrchestrator(backend)
        self._generation_lock = threading.Lock()

    def stream(
        self,
        request: RuntimeChatRequest,
        *,
        trusted_scope_id: str,
        cancellation: threading.Event,
    ) -> Iterator[dict[str, object]]:
        """Yield UI-contract records for one already-authenticated bridge request."""

        if not isinstance(request, RuntimeChatRequest):
            raise TypeError("request must be a RuntimeChatRequest")
        if not isinstance(cancellation, threading.Event):
            raise TypeError("cancellation must be a threading.Event")
        if cancellation.is_set():
            return

        generation_request = GenerationRequest(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            messages=tuple(
                ChatMessage(message.role, message.content) for message in request.messages
            ),
            memory_mode=request.memory,
        )
        token = CancellationToken(cancellation)

        try:
            started_at = self._read_monotonic()
            with self.coordinator_factory() as coordinator:
                prepare = getattr(coordinator, "prepare", None)
                if not callable(prepare):
                    raise TypeError("coordinator factory must yield a RuntimeCoordinator")
                prepared = prepare(
                    trusted_scope_id=trusted_scope_id,
                    request=generation_request,
                )
            finished_at = self._read_monotonic()
            if finished_at < started_at:
                raise RuntimeError("preparation clock moved backwards")
            preparation_elapsed_ms = _finite_nonnegative(
                (finished_at - started_at) * 1_000,
                "preparation_elapsed_ms",
            )
            preparation = self.telemetry_provider(
                prepared,
                preparation_elapsed_ms,
            )
            if not isinstance(preparation, PreparationTelemetry):
                raise TypeError("telemetry provider must return PreparationTelemetry")
            plan = ExecutionPlan.create(
                prepared=prepared,
                renderer=self.renderer,
                decoding=self.decoding,
                backend=self.backend.descriptor,
            )

            if not self._acquire_generation(token):
                return
            try:
                async_stream = self.orchestrator.stream(
                    plan,
                    cancellation=token,
                    preparation=preparation,
                )
                for event in _iterate_async_stream(async_stream):
                    token.raise_if_cancelled()
                    yield event.to_dict()
            finally:
                self._generation_lock.release()
        except GeneratorExit:
            token.cancel()
            raise
        except Exception:
            if not token.is_cancelled:
                yield {
                    "type": "error",
                    "request_id": request.request_id,
                    "message": "The local generation failed.",
                }

    def _acquire_generation(self, cancellation: CancellationToken) -> bool:
        while not cancellation.is_cancelled:
            if self._generation_lock.acquire(timeout=0.1):
                return True
        return False

    def _read_monotonic(self) -> float:
        value = self.monotonic()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("monotonic clock must return a real number")
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("monotonic clock must return a finite value")
        return value


def _iterate_async_stream(
    stream: AsyncIterator[RuntimeStreamEvent],
) -> Iterator[RuntimeStreamEvent]:
    """Drain one async generator from the synchronous request-handler thread."""

    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(anext(stream))
            except StopAsyncIteration:
                break
    finally:
        try:
            loop.run_until_complete(stream.aclose())
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
