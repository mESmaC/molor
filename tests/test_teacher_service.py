from __future__ import annotations

import threading
from typing import Any

import pytest

from molor.desktop.backend import (
    ChatTurn,
    DesktopGeneration,
    DesktopGenerationSettings,
    DesktopModelInfo,
    DesktopRoute,
    DesktopStats,
)
from molor.experimental import AutoRouteDecision, AutoRouterPolicy
from molor.feedback import FeedbackStore
from molor.teacher.protocol import TOOL_SCHEMAS
from molor.teacher.service import StudentConfig, TeacherStudentService


class _FakeSession:
    def __init__(self, decision: AutoRouteDecision) -> None:
        self.info = DesktopModelInfo(
            model_id="student-model",
            device="cpu",
            gpu="CPU",
            context_limit=4096,
            expert_a_name="embedded",
            expert_b_name="can",
            expert_names=("embedded", "can", "architecture"),
            router_checkpoint="router/checkpoint",
            router_loaded=True,
        )
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def stream_chat(
        self,
        messages: tuple[ChatTurn, ...],
        *,
        route: DesktopRoute | None,
        settings: DesktopGenerationSettings,
        cancellation: threading.Event,
        routing_policy: AutoRouterPolicy | None,
    ) -> Any:
        self.calls.append(
            {
                "messages": messages,
                "route": route,
                "settings": settings,
                "cancellation": cancellation,
                "routing_policy": routing_policy,
            }
        )
        yield DesktopGeneration(routing_decision=self.decision)
        yield DesktopGeneration(
            text="A routed answer.",
            stats=DesktopStats(
                model_id=self.info.model_id,
                device="cpu",
                dtype="float32",
                route=self.decision.applied_route,
                prompt_tokens=8,
                completion_tokens=4,
                context_tokens=12,
                context_limit=4096,
                context_token_budget=None,
                dropped_turns=0,
                dropped_tokens=0,
                cache_mode="dynamic",
                elapsed_seconds=0.1,
                time_to_first_token_seconds=0.01,
                tokens_per_second=40.0,
                finish_reason="stop",
            ),
            finished=True,
        )


def _service(
    decision: AutoRouteDecision,
) -> tuple[TeacherStudentService, FeedbackStore, _FakeSession]:
    store = FeedbackStore(":memory:")
    service = TeacherStudentService(
        StudentConfig(
            "student-model",
            "embedded/path",
            "can/path",
            expert_paths=("architecture/path",),
            router_checkpoint="router/checkpoint",
        ),
        store,
    )
    session = _FakeSession(decision)
    service._session = session  # type: ignore[assignment] - focused service boundary fake
    return service, store, session


def test_auto_routing_returns_and_persists_named_decision() -> None:
    decision = AutoRouteDecision(
        mode="auto",
        applied_route=(("architecture", 1.0),),
        probabilities=(
            ("base", 0.05),
            ("embedded", 0.05),
            ("can", 0.1),
            ("architecture", 0.8),
        ),
        selected_routes=("architecture",),
        confidence=0.8,
        reason="Architecture won.",
        checkpoint_loaded=True,
    )
    service, store, session = _service(decision)
    try:
        response = service.ask_student(
            {
                "prompt": "Design a service boundary.",
                "routing": {
                    "mode": "auto",
                    "top_k": 1,
                    "temperature": 0.75,
                    "confidence_threshold": 0.2,
                    "null_bias": -0.1,
                    "max_prompt_tokens": 192,
                    "enabled_experts": ["embedded", "can", "architecture"],
                },
            }
        )

        call = session.calls[0]
        assert call["route"] is None
        policy = call["routing_policy"]
        assert isinstance(policy, AutoRouterPolicy)
        assert policy.enabled_experts == ("embedded", "can", "architecture")
        candidate = response["candidates"][0]
        assert candidate["routing_decision"] == decision.to_dict()
        assert response["routing_policy"] == policy.to_dict()

        record = store.get_generation(candidate["generation_id"])
        assert record is not None
        assert record.route == {"architecture": 1.0}
        assert record.generation_settings["routing_policy"] == policy.to_dict()
        assert record.generation_settings["routing_policy_source"] == "routing"
        assert record.generation_settings["routing_decision"] == decision.to_dict()
    finally:
        store.close()


def test_legacy_a_b_route_remains_manual_but_persists_actual_name() -> None:
    decision = AutoRouteDecision(
        mode="manual",
        applied_route=(("embedded", 1.0),),
        probabilities=(
            ("base", 0.0),
            ("embedded", 1.0),
            ("can", 0.0),
            ("architecture", 0.0),
        ),
        selected_routes=("embedded",),
        confidence=1.0,
        reason="Applied the configured manual route.",
        checkpoint_loaded=False,
    )
    service, store, session = _service(decision)
    try:
        response = service.ask_student(
            {"prompt": "Explain volatile.", "route": {"base": 0.0, "expert_a": 1.0}}
        )

        call = session.calls[0]
        assert call["routing_policy"] is None
        assert call["route"] == DesktopRoute(0.0, 1.0, 0.0)
        candidate = response["candidates"][0]
        record = store.get_generation(candidate["generation_id"])
        assert record is not None
        assert record.route == {"embedded": 1.0}
        assert record.generation_settings["routing_policy_source"] == "legacy_route"
        assert record.generation_settings["routing_policy"]["manual_route"] == {"embedded": 1.0}
    finally:
        store.close()


def test_routing_contract_is_published_and_cannot_mix_with_legacy_route() -> None:
    ask_schema = next(tool for tool in TOOL_SCHEMAS if tool["name"] == "ask_student")
    assert "routing" in ask_schema["input_schema"]["properties"]

    with pytest.raises(ValueError, match="provide route or routing"):
        TeacherStudentService._routing_request(
            {
                "route": {"base": 1.0},
                "routing": {"mode": "auto"},
            }
        )


def test_nonpersistent_benchmark_generation_does_not_enter_feedback_store() -> None:
    decision = AutoRouteDecision(
        mode="manual",
        applied_route=(("base", 1.0),),
        probabilities=(("base", 1.0),),
        selected_routes=("base",),
        confidence=1.0,
        reason="Forced base.",
        checkpoint_loaded=True,
    )
    service, store, _session = _service(decision)
    try:
        response = service.ask_student(
            {
                "prompt": "Benchmark prompt",
                "persist": False,
                "routing": {"mode": "manual", "manual_route": {"base": 1.0}},
            }
        )
        assert response["candidates"][0]["generation_id"] is None
        assert store.list_generations(limit=10) == []
    finally:
        store.close()
