"""Core teacher/student operations, independent of the HTTP transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence
import uuid

from molor.desktop.backend import (
    ChatTurn,
    DesktopGenerationSettings,
    DesktopRoute,
    DesktopSession,
    LoadOptions,
)
from molor.experimental import AutoRouteDecision, AutoRouterPolicy
from molor.feedback import FeedbackStore, GenerationCreate, GenerationRecord
from molor.teacher.protocol import RUBRIC, capabilities


@dataclass(frozen=True, slots=True)
class StudentConfig:
    model_id: str
    expert_a_path: str = ""
    expert_b_path: str = ""
    expert_paths: tuple[str, ...] = ()
    router_checkpoint: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.expert_paths, (str, bytes)):
            raise TypeError("expert_paths must be a sequence of adapter paths")
        paths = tuple(self.expert_paths)
        if any(not isinstance(path, str) for path in paths):
            raise TypeError("expert_paths must contain only strings")
        object.__setattr__(self, "expert_paths", paths)


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _record_payload(record: GenerationRecord) -> dict[str, object]:
    return {
        "generation_id": record.generation_id,
        "conversation_id": record.conversation_id,
        "created_at": record.created_at,
        "graded_at": record.graded_at,
        "model_id": record.model_id,
        "messages": list(record.messages),
        "response": record.response,
        "route": record.route,
        "generation_settings": record.generation_settings,
        "reasoning_settings": record.reasoning_settings,
        "context_settings": record.context_settings,
        "cache_settings": record.cache_settings,
        "telemetry": record.telemetry,
        "human_reward": record.reward,
        "human_dimensions": record.dimensions,
        "human_notes": record.notes,
        "human_corrected_answer": record.corrected_answer,
    }


class TeacherStudentService:
    """Serialize local inference while allowing concurrent read/grade requests."""

    def __init__(self, config: StudentConfig, store: FeedbackStore) -> None:
        if not config.model_id.strip():
            raise ValueError("model_id is required")
        self.config = config
        self.store = store
        self._session: DesktopSession | None = None
        self._model_lock = threading.RLock()

    @property
    def model_loaded(self) -> bool:
        return self._session is not None

    def health(self) -> dict[str, object]:
        session = self._session
        info = session.info if session else None
        return {
            "status": "ok",
            "protocol_version": "molor.teacher-api.v1",
            "model_loaded": session is not None,
            "model_id": info.model_id if info else self.config.model_id,
            "experts": list(info.expert_names) if info else [],
            "configured_expert_paths": [
                path
                for path in (
                    self.config.expert_a_path,
                    self.config.expert_b_path,
                    *self.config.expert_paths,
                )
                if path.strip()
            ],
            "router_checkpoint": (
                info.router_checkpoint if info else self.config.router_checkpoint.strip() or None
            ),
            "router_loaded": bool(info and info.router_loaded),
            "database": str(self.store.path.resolve()),
        }

    def load_student(self) -> dict[str, object]:
        with self._model_lock:
            if self._session is None:
                self._session = DesktopSession.load(
                    LoadOptions(
                        model_id=self.config.model_id,
                        expert_a_path=self.config.expert_a_path,
                        expert_b_path=self.config.expert_b_path,
                        expert_paths=self.config.expert_paths,
                        router_checkpoint=self.config.router_checkpoint,
                    )
                )
            info = self._session.info
            return {
                "model_id": info.model_id,
                "device": info.device,
                "gpu": info.gpu,
                "context_limit": info.context_limit,
                "expert_a": info.expert_a_name,
                "expert_b": info.expert_b_name,
                "experts": list(info.expert_names),
                "router_checkpoint": info.router_checkpoint,
                "router_loaded": info.router_loaded,
            }

    def unload_student(self) -> dict[str, object]:
        with self._model_lock:
            session = self._session
            self._session = None
            if session is not None:
                del session
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return self.health()

    def ask_student(self, request: Mapping[str, object]) -> dict[str, object]:
        payload = _mapping(request, "request")
        messages = self._messages(payload)
        candidates = payload.get("candidates", 1)
        if isinstance(candidates, bool) or not isinstance(candidates, int) or not 1 <= candidates <= 8:
            raise ValueError("candidates must be an integer from 1 to 8")
        persist = payload.get("persist", True)
        if not isinstance(persist, bool):
            raise ValueError("persist must be boolean")
        conversation_id = str(payload.get("conversation_id") or uuid.uuid4().hex).strip()
        if not conversation_id:
            raise ValueError("conversation_id cannot be blank")
        teacher_session_id = str(payload.get("teacher_session_id") or "").strip() or None
        request_metadata = _mapping(payload.get("metadata"), "metadata")
        settings = self._settings(_mapping(payload.get("settings"), "settings"))
        route, routing_policy, routing_policy_source = self._routing_request(payload)
        raw_seed = payload.get("seed")
        if raw_seed is not None and (isinstance(raw_seed, bool) or not isinstance(raw_seed, int)):
            raise ValueError("seed must be an integer or null")

        results: list[dict[str, object]] = []
        response_policy: dict[str, object] | None = None
        with self._model_lock:
            self.load_student()
            assert self._session is not None
            for candidate_index in range(candidates):
                if raw_seed is not None:
                    self._seed(raw_seed + candidate_index)
                text_parts: list[str] = []
                final_stats = None
                routing_decision: AutoRouteDecision | None = None
                for update in self._session.stream_chat(
                    messages,
                    route=route,
                    settings=settings,
                    cancellation=threading.Event(),
                    routing_policy=routing_policy,
                ):
                    if update.routing_decision is not None:
                        if not isinstance(update.routing_decision, AutoRouteDecision):
                            raise TypeError("student returned an invalid routing decision")
                        if (
                            routing_decision is not None
                            and update.routing_decision != routing_decision
                        ):
                            raise RuntimeError("student changed routes during one generation")
                        routing_decision = update.routing_decision
                    if update.text:
                        if routing_decision is None:
                            raise RuntimeError(
                                "student emitted tokens before returning a routing decision"
                            )
                        text_parts.append(update.text)
                    if update.stats is not None:
                        final_stats = update.stats
                if routing_decision is None:
                    raise RuntimeError("student did not return a routing decision")
                response = "".join(text_parts).strip()
                if not response:
                    raise RuntimeError("student returned an empty generation")
                telemetry = asdict(final_stats) if final_stats is not None else {}
                decision_payload = routing_decision.to_dict()
                route_payload = dict(routing_decision.applied_route)
                effective_policy = (
                    routing_policy.to_dict()
                    if routing_policy is not None
                    else AutoRouterPolicy(
                        mode="manual",
                        manual_route=route_payload,
                    ).to_dict()
                )
                if response_policy is None:
                    response_policy = effective_policy
                generation_id: str | None = None
                if persist:
                    generation_id = self.store.record_generation(
                        GenerationCreate(
                            conversation_id=conversation_id,
                            model_id=self._session.info.model_id,
                            messages=tuple(
                                {"role": turn.role, "content": turn.content} for turn in messages
                            ),
                            response=response,
                            route=route_payload,
                            settings={
                                **settings.to_dict(),
                                "candidate_index": candidate_index,
                                "candidate_count": candidates,
                                "seed": None if raw_seed is None else raw_seed + candidate_index,
                                "teacher_session_id": teacher_session_id,
                                "request_metadata": request_metadata,
                                "routing_policy": effective_policy,
                                "routing_policy_source": routing_policy_source,
                                "routing_decision": decision_payload,
                            },
                            telemetry=telemetry,
                            system_instruction=settings.system_instruction,
                            enable_thinking=settings.enable_thinking,
                            reasoning_preset=settings.reasoning_preset,
                            context_policy={"token_budget": settings.context_token_budget},
                            cache_policy={"mode": settings.cache_mode},
                        )
                    )
                results.append(
                    {
                        "candidate_index": candidate_index,
                        "generation_id": generation_id,
                        "response": response,
                        "telemetry": telemetry,
                        "routing_decision": decision_payload,
                    }
                )
        return {
            "conversation_id": conversation_id,
            "teacher_session_id": teacher_session_id,
            "routing_policy": response_policy,
            "candidates": results,
            "comparison_ready": len(results) > 1,
        }

    def grade_student(self, request: Mapping[str, object]) -> dict[str, object]:
        payload = _mapping(request, "request")
        generation_id = str(payload.get("generation_id") or "").strip()
        if not generation_id:
            raise ValueError("generation_id is required")
        teacher = self._teacher(payload)
        scores = _mapping(payload.get("scores"), "scores")
        if "overall" not in scores:
            raise ValueError("scores.overall is required")
        dimensions = {name: value for name, value in scores.items() if name != "overall"}
        confidence = _finite_number(payload.get("confidence", 1.0), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be from 0 to 1")
        metadata = _mapping(payload.get("metadata"), "metadata")
        if teacher.get("provider"):
            metadata = {**metadata, "provider": teacher["provider"]}
        annotation_id = self.store.record_teacher_annotation(
            generation_id,
            teacher_id=str(teacher["id"]),
            teacher_model=str(teacher["model"]) if teacher.get("model") else None,
            overall_score=_finite_number(scores["overall"], "scores.overall"),
            dimensions={
                name: _finite_number(value, f"scores.{name}")
                for name, value in dimensions.items()
            },
            confidence=confidence,
            critique=str(payload.get("critique") or ""),
            corrected_answer=(
                None
                if payload.get("corrected_answer") is None
                else str(payload["corrected_answer"])
            ),
            rubric_version=str(payload.get("rubric_version") or RUBRIC["version"]),
            metadata=metadata,
        )
        return {
            "annotation_id": annotation_id,
            "generation_id": generation_id,
            "consensus": self.store.teacher_consensus(generation_id),
        }

    def prefer_student_output(self, request: Mapping[str, object]) -> dict[str, object]:
        payload = _mapping(request, "request")
        chosen = str(payload.get("chosen_generation_id") or "").strip()
        rejected = str(payload.get("rejected_generation_id") or "").strip()
        if not chosen or not rejected:
            raise ValueError("chosen_generation_id and rejected_generation_id are required")
        teacher = self._teacher(payload)
        confidence = _finite_number(payload.get("confidence", 1.0), "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be from 0 to 1")
        metadata = _mapping(payload.get("metadata"), "metadata")
        if teacher.get("provider"):
            metadata = {**metadata, "provider": teacher["provider"]}
        preference_id = self.store.record_preference(
            chosen,
            rejected,
            notes=str(payload.get("rationale") or ""),
            evaluator_id=str(teacher["id"]),
            evaluator_model=str(teacher["model"]) if teacher.get("model") else None,
            confidence=confidence,
            metadata=metadata,
        )
        return {
            "preference_id": preference_id,
            "chosen_generation_id": chosen,
            "rejected_generation_id": rejected,
        }

    def get_generation(self, generation_id: str) -> dict[str, object]:
        record = self.store.get_generation(generation_id)
        if record is None:
            raise KeyError(f"unknown generation {generation_id!r}")
        annotations = self.store.list_teacher_annotations(
            generation_id=generation_id,
            limit=10_000,
        )
        return {
            **_record_payload(record),
            "teacher_annotations": [asdict(annotation) for annotation in annotations],
            "teacher_consensus": self.store.teacher_consensus(generation_id),
        }

    def next_review(self, teacher_id: str) -> dict[str, object]:
        record = self.store.next_generation_for_teacher(teacher_id)
        return {"generation": None if record is None else self.get_generation(record.generation_id)}

    def export_training_data(self, request: Mapping[str, object]) -> dict[str, object]:
        payload = _mapping(request, "request")
        directory = Path(str(payload.get("directory") or "runs/feedback/exports")).resolve()
        teacher = self.store.export_teacher_reward_jsonl(directory / "teacher_rewards.jsonl")
        preference = self.store.export_preference_jsonl(directory / "preferences.jsonl")
        sft = self.store.export_training_sft_jsonl(directory / "training_sft.jsonl")
        return {
            "teacher_rewards": {"path": str(teacher.path), "rows": teacher.rows},
            "preferences": {"path": str(preference.path), "rows": preference.rows},
            "training_sft": {"path": str(sft.path), "rows": sft.rows},
        }

    @staticmethod
    def capabilities() -> dict[str, object]:
        return capabilities()

    @staticmethod
    def _messages(payload: Mapping[str, object]) -> tuple[ChatTurn, ...]:
        raw_messages = payload.get("messages")
        prompt = payload.get("prompt")
        if raw_messages is not None and prompt is not None:
            raise ValueError("provide prompt or messages, not both")
        if raw_messages is None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("a non-empty prompt or messages array is required")
            return (ChatTurn("user", prompt.strip()),)
        if isinstance(raw_messages, (str, bytes)) or not isinstance(raw_messages, Sequence):
            raise TypeError("messages must be an array")
        messages: list[ChatTurn] = []
        for index, raw_message in enumerate(raw_messages):
            message = _mapping(raw_message, f"messages[{index}]")
            if set(message) != {"role", "content"}:
                raise ValueError(f"messages[{index}] needs exactly role and content")
            role = message["role"]
            content = message["content"]
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"messages[{index}].role is unsupported")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"messages[{index}].content must be non-empty")
            messages.append(ChatTurn(str(role), content))
        if not messages or messages[-1].role != "user":
            raise ValueError("messages must end with a user turn")
        return tuple(messages)

    @staticmethod
    def _settings(payload: Mapping[str, object]) -> DesktopGenerationSettings:
        allowed = {
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "enable_thinking",
            "reasoning_preset",
            "system_instruction",
            "context_token_budget",
            "cache_mode",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown generation settings: {', '.join(sorted(unknown))}")
        values: dict[str, object] = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "enable_thinking": False,
            "reasoning_preset": "Teacher evaluation",
            "system_instruction": "Answer accurately, directly, and state important uncertainty.",
            "context_token_budget": None,
            "cache_mode": "dynamic",
            **payload,
        }
        max_new_tokens = values["max_new_tokens"]
        top_k = values["top_k"]
        context_budget = values["context_token_budget"]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if context_budget is not None and (
            isinstance(context_budget, bool) or not isinstance(context_budget, int)
        ):
            raise ValueError("context_token_budget must be an integer or null")
        if not isinstance(values["enable_thinking"], bool):
            raise ValueError("enable_thinking must be boolean")
        temperature = _finite_number(values["temperature"], "temperature")
        top_p = _finite_number(values["top_p"], "top_p")
        if not 0 <= temperature <= 10:
            raise ValueError("temperature must be from 0 to 10")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        return DesktopGenerationSettings(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=values["enable_thinking"],
            reasoning_preset=str(values["reasoning_preset"]),
            system_instruction=str(values["system_instruction"]),
            context_token_budget=context_budget,
            cache_mode=str(values["cache_mode"]),
        )

    @staticmethod
    def _route(payload: Mapping[str, object]) -> DesktopRoute:
        unknown = set(payload) - {"base", "expert_a", "expert_b"}
        if unknown:
            raise ValueError(f"unknown route fields: {', '.join(sorted(unknown))}")
        values = {
            "base": _finite_number(payload.get("base", 1.0), "route.base"),
            "expert_a": _finite_number(payload.get("expert_a", 0.0), "route.expert_a"),
            "expert_b": _finite_number(payload.get("expert_b", 0.0), "route.expert_b"),
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("route weights cannot be negative")
        return DesktopRoute(**values).normalized()

    @classmethod
    def _routing_request(
        cls,
        payload: Mapping[str, object],
    ) -> tuple[DesktopRoute | None, AutoRouterPolicy | None, str]:
        raw_route = payload.get("route")
        raw_routing = payload.get("routing")
        if raw_route is not None and raw_routing is not None:
            raise ValueError("provide route or routing, not both")
        if raw_route is not None:
            return cls._route(_mapping(raw_route, "route")), None, "legacy_route"
        if raw_routing is None:
            return None, AutoRouterPolicy(), "default_auto"
        return None, cls._routing(_mapping(raw_routing, "routing")), "routing"

    @staticmethod
    def _routing(payload: Mapping[str, object]) -> AutoRouterPolicy:
        allowed = {
            "mode",
            "top_k",
            "temperature",
            "confidence_threshold",
            "null_bias",
            "max_prompt_tokens",
            "enabled_experts",
            "manual_route",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown routing fields: {', '.join(sorted(unknown))}")
        raw_enabled = payload.get("enabled_experts", ())
        if isinstance(raw_enabled, (str, bytes)) or not isinstance(raw_enabled, Sequence):
            raise TypeError("routing.enabled_experts must be an array")
        manual_route = _mapping(payload.get("manual_route"), "routing.manual_route")
        return AutoRouterPolicy(
            mode=payload.get("mode", "auto"),
            top_k=payload.get("top_k", 1),
            temperature=payload.get("temperature", 1.0),
            confidence_threshold=payload.get("confidence_threshold", 0.0),
            null_bias=payload.get("null_bias", 0.0),
            max_prompt_tokens=payload.get("max_prompt_tokens", 256),
            enabled_experts=tuple(raw_enabled),
            manual_route=manual_route,
        )

    @staticmethod
    def _teacher(payload: Mapping[str, object]) -> dict[str, object]:
        teacher = _mapping(payload.get("teacher"), "teacher")
        teacher_id = teacher.get("id")
        if not isinstance(teacher_id, str) or not teacher_id.strip():
            raise ValueError("teacher.id is required")
        for key in ("model", "provider"):
            if teacher.get(key) is not None and not isinstance(teacher[key], str):
                raise TypeError(f"teacher.{key} must be a string")
        return {**teacher, "id": teacher_id.strip()}

    @staticmethod
    def _seed(seed: int) -> None:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
