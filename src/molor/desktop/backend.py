"""Small, Qt-free bridge between the desktop window and the experimental engine."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from molor.experimental.autorouter import (
    AutoRouteDecision,
    AutoRouterPolicy,
    PromptAutoRouter,
    base_fallback_decision,
    manual_route_decision,
)


@dataclass(frozen=True, slots=True)
class LoadOptions:
    """The handful of knobs worth exposing in the toy desktop app."""

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


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class DesktopGenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    enable_thinking: bool = False
    reasoning_preset: str = "Direct"
    system_instruction: str = "Answer directly and clearly."
    context_token_budget: int | None = None
    cache_mode: str = "dynamic"

    def to_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_thinking": self.enable_thinking,
            "reasoning_preset": self.reasoning_preset,
            "system_instruction": self.system_instruction,
            "context_token_budget": self.context_token_budget,
            "cache_mode": self.cache_mode,
        }


@dataclass(frozen=True, slots=True)
class DesktopRoute:
    """Convex route across the untouched base and two adapter deltas."""

    base: float
    expert_a: float
    expert_b: float

    def normalized(self) -> DesktopRoute:
        values = tuple(
            max(0.0, float(value)) for value in (self.base, self.expert_a, self.expert_b)
        )
        total = sum(values)
        if total == 0:
            return DesktopRoute(1.0, 0.0, 0.0)
        return DesktopRoute(*(value / total for value in values))


@dataclass(frozen=True, slots=True)
class DesktopModelInfo:
    model_id: str
    device: str
    gpu: str
    context_limit: int | None
    expert_a_name: str
    expert_b_name: str
    expert_names: tuple[str, ...] = ()
    router_checkpoint: str | None = None
    router_loaded: bool = False


@dataclass(frozen=True, slots=True)
class DesktopStats:
    model_id: str
    device: str
    dtype: str
    route: tuple[tuple[str, float], ...]
    prompt_tokens: int
    completion_tokens: int
    context_tokens: int
    context_limit: int | None
    context_token_budget: int | None
    dropped_turns: int
    dropped_tokens: int
    cache_mode: str
    elapsed_seconds: float
    time_to_first_token_seconds: float | None
    tokens_per_second: float | None
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class DesktopGeneration:
    """One streamed desktop update.

    ``text`` is always final-answer text. A leading Qwen thinking block is
    carried separately in ``reasoning_text`` so existing consumers continue to
    receive only the answer through the original field.
    """

    text: str = ""
    stats: DesktopStats | None = None
    finished: bool = False
    reasoning_text: str = ""
    reasoning_state: Literal["none", "streaming", "complete", "incomplete"] = "none"
    routing_decision: AutoRouteDecision | None = None


@dataclass(frozen=True, slots=True)
class _SeparatedGenerationText:
    answer: str = ""
    reasoning: str = ""
    reasoning_state: Literal["none", "streaming", "complete", "incomplete"] = "none"


class _QwenReasoningParser:
    """Separate the first leading Qwen thinking block across arbitrary chunks.

    A leading ``<think>`` is auto-detected even when thinking was not requested.
    When thinking *was* requested, output is held while it is ambiguous whether
    a model that omitted the opener will later emit ``</think>``. If neither
    marker appears, ``finish`` releases the buffered content as a direct answer.
    Once an opener is observed, reaching the end without a closer is instead an
    incomplete reasoning trace and is never promoted to an answer.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, *, enabled: bool) -> None:
        self._phase: Literal["probe", "reasoning", "answer"] = "probe"
        self._expect_reasoning = enabled
        self._pending = ""
        self._reasoning_state: Literal[
            "none", "streaming", "complete", "incomplete"
        ] = "none"
        self._finished = False

    @property
    def reasoning_state(
        self,
    ) -> Literal["none", "streaming", "complete", "incomplete"]:
        return self._reasoning_state

    def feed(self, text: str) -> _SeparatedGenerationText:
        if self._finished:
            raise RuntimeError("cannot feed a finished reasoning parser")
        if not text:
            return _SeparatedGenerationText(reasoning_state=self._reasoning_state)
        if self._phase == "answer":
            return _SeparatedGenerationText(
                answer=text,
                reasoning_state=self._reasoning_state,
            )

        self._pending += text
        if self._phase == "probe":
            return self._probe()
        return self._stream_reasoning()

    def finish(self) -> _SeparatedGenerationText:
        if self._finished:
            return _SeparatedGenerationText(reasoning_state=self._reasoning_state)
        self._finished = True
        if self._phase == "probe":
            # No marker ever disambiguated the output: the model answered
            # directly despite thinking mode being requested.
            answer, self._pending = self._pending, ""
            self._phase = "answer"
            return _SeparatedGenerationText(answer=answer, reasoning_state="none")
        if self._phase == "reasoning":
            reasoning, self._pending = self._pending, ""
            self._reasoning_state = "incomplete"
            return _SeparatedGenerationText(
                reasoning=reasoning,
                reasoning_state=self._reasoning_state,
            )
        return _SeparatedGenerationText(reasoning_state=self._reasoning_state)

    def _probe(self) -> _SeparatedGenerationText:
        leading = len(self._pending) - len(self._pending.lstrip())
        candidate = self._pending[leading:]
        if candidate.startswith(self._OPEN):
            self._pending = candidate[len(self._OPEN) :]
            self._phase = "reasoning"
            self._reasoning_state = "streaming"
            return self._stream_reasoning()

        if not self._expect_reasoning:
            # Outside requested thinking mode, hold only whitespace and a
            # partial leading marker. Ordinary answers still stream as soon as
            # the first non-matching character arrives.
            if not candidate or self._OPEN.startswith(candidate):
                return _SeparatedGenerationText(reasoning_state="none")
            answer, self._pending = self._pending, ""
            self._phase = "answer"
            return _SeparatedGenerationText(answer=answer, reasoning_state="none")

        # A partial leading opener remains ambiguous. All other no-opener
        # output is retained because Qwen may still emit a closing marker.
        closing = self._pending.find(self._CLOSE)
        if closing < 0:
            return _SeparatedGenerationText(reasoning_state=self._reasoning_state)

        reasoning = self._pending[:closing]
        answer = self._pending[closing + len(self._CLOSE) :]
        self._pending = ""
        self._phase = "answer"
        self._reasoning_state = "complete"
        return _SeparatedGenerationText(
            answer=answer,
            reasoning=reasoning,
            reasoning_state=self._reasoning_state,
        )

    def _stream_reasoning(self) -> _SeparatedGenerationText:
        closing = self._pending.find(self._CLOSE)
        if closing >= 0:
            reasoning = self._pending[:closing]
            answer = self._pending[closing + len(self._CLOSE) :]
            self._pending = ""
            self._phase = "answer"
            self._reasoning_state = "complete"
            return _SeparatedGenerationText(
                answer=answer,
                reasoning=reasoning,
                reasoning_state=self._reasoning_state,
            )

        # Emit everything except a suffix that could be the beginning of a
        # closing marker. This makes ``</think>`` safe at every possible split.
        retained = 0
        maximum = min(len(self._pending), len(self._CLOSE) - 1)
        for length in range(maximum, 0, -1):
            if self._pending.endswith(self._CLOSE[:length]):
                retained = length
                break
        boundary = len(self._pending) - retained
        reasoning = self._pending[:boundary]
        self._pending = self._pending[boundary:]
        return _SeparatedGenerationText(
            reasoning=reasoning,
            reasoning_state=self._reasoning_state,
        )


class DesktopSession:
    """Own one loaded base, an arbitrary resident expert rack, and optional router."""

    _EXPERT_A = "expert_a"
    _EXPERT_B = "expert_b"

    def __init__(
        self,
        engine: Any,
        info: DesktopModelInfo,
        *,
        auto_router: PromptAutoRouter | None = None,
        router_unavailable_reason: str = "No router checkpoint was configured.",
    ) -> None:
        self.engine = engine
        self.info = info
        self.auto_router = auto_router
        self.router_unavailable_reason = router_unavailable_reason

    @classmethod
    def load(cls, options: LoadOptions) -> DesktopSession:
        model_id = options.model_id.strip()
        if not model_id:
            raise ValueError("Enter a Hugging Face model ID or local model directory.")

        local_model = Path(model_id).expanduser()
        is_local_model = local_model.is_dir()
        model_source = str(local_model.resolve()) if is_local_model else model_id

        try:
            import torch
            from molor.experimental import MixtureEngine, QWEN3_TARGET_MODULES
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "The model extras are missing. Install this project with: "
                'pip install -e ".[hf,desktop]"'
            ) from exc

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        requested_paths = (
            options.expert_a_path,
            options.expert_b_path,
            *options.expert_paths,
        )
        target_modules = cls._adapter_target_modules(*requested_paths) or tuple(
            QWEN3_TARGET_MODULES
        )

        engine = MixtureEngine.from_pretrained(
            model_id=model_source,
            dtype=dtype,
            device_map="auto",
            target_modules=target_modules,
            local_files_only=is_local_model,
        )
        loaded_by_path: dict[str, str] = {}
        loaded_names: set[str] = set()
        slot_names: list[str] = []
        for raw_path in requested_paths:
            path_text = raw_path.strip()
            if not path_text:
                slot_names.append("")
                continue
            path = Path(path_text).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(f"Adapter path does not exist: {path}")
            path_key = str(path.resolve()).casefold()
            loaded_name = loaded_by_path.get(path_key)
            if loaded_name is None:
                loaded_name = cls._load_real_expert(engine, path, loaded_names)
                loaded_by_path[path_key] = loaded_name
                loaded_names.add(loaded_name)
            slot_names.append(loaded_name)

        expert_a = slot_names[0]
        expert_b = slot_names[1]
        router: PromptAutoRouter | None = None
        router_path_text = options.router_checkpoint.strip()
        router_display: str | None = None
        router_reason = "No router checkpoint was configured."
        if router_path_text:
            router_path = Path(router_path_text).expanduser()
            try:
                router_display = str(router_path.resolve())
            except OSError:
                router_display = str(router_path)
            if router_path.is_dir():
                router = PromptAutoRouter.load_checkpoint(
                    router_path,
                    tokenizer=engine.tokenizer,
                    input_embeddings=engine.model.get_input_embeddings(),
                    available_experts=engine.expert_names,
                    expected_model_id=engine.base_model_id,
                )
                router_reason = "Router checkpoint loaded."
            else:
                router_reason = f"Router checkpoint directory is unavailable: {router_path}"

        context_limit = cls._context_limit(engine)
        device = cls._device_label(engine)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return cls(
            engine,
            DesktopModelInfo(
                model_id=model_source,
                device=device,
                gpu=gpu,
                context_limit=context_limit,
                expert_a_name=expert_a,
                expert_b_name=expert_b,
                expert_names=tuple(engine.expert_names),
                router_checkpoint=router_display,
                router_loaded=router is not None,
            ),
            auto_router=router,
            router_unavailable_reason=router_reason,
        )

    @staticmethod
    def _adapter_target_modules(*raw_paths: str) -> tuple[str, ...] | None:
        discovered: list[tuple[str, ...]] = []
        for raw_path in raw_paths:
            if not raw_path.strip():
                continue
            directory = Path(raw_path.strip()).expanduser()
            config_path = directory / "adapter_config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"Adapter config does not exist: {config_path}")
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            raw_targets = payload.get("target_modules")
            if not isinstance(raw_targets, list) or not raw_targets:
                raise ValueError(f"Adapter has no target_modules list: {config_path}")
            targets = tuple(raw_targets)
            if any(not isinstance(name, str) or not name for name in targets):
                raise ValueError(f"Adapter target_modules are invalid: {config_path}")
            if len(targets) != len(set(targets)):
                raise ValueError(f"Adapter target_modules contain duplicates: {config_path}")
            discovered.append(targets)
        if not discovered:
            return None
        if any(targets != discovered[0] for targets in discovered[1:]):
            raise ValueError("Loaded experts target different projection sets")
        return discovered[0]

    @classmethod
    def _load_real_expert(
        cls,
        engine: Any,
        path: Path,
        loaded_names: set[str],
    ) -> str:
        """Load one real checkpoint under a stable, collision-free runtime name."""

        config_path = path / "adapter_config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw_name = payload.get("name")
        preferred = raw_name.strip() if isinstance(raw_name, str) else path.name
        preferred = re.sub(r"[^A-Za-z0-9_-]+", "_", preferred).strip("_")
        if not preferred or not preferred[0].isalnum():
            preferred = f"expert_{preferred}".strip("_")
        if preferred == "base":
            preferred = "base_expert"
        preferred = preferred[:64]
        name = preferred
        suffix = 2
        while name in loaded_names:
            tail = f"_{suffix}"
            name = preferred[: 64 - len(tail)] + tail
            suffix += 1
        loaded_name = engine.load_expert(path, name=name)
        return str(loaded_name or name)

    @staticmethod
    def _context_limit(engine: Any) -> int | None:
        candidates = (
            getattr(engine, "context_limit", None),
            getattr(getattr(engine, "tokenizer", None), "model_max_length", None),
            getattr(getattr(engine, "model", None), "config", None),
        )
        config = candidates[2]
        raw_values = [candidates[0], candidates[1]]
        if config is not None:
            raw_values.extend(
                getattr(config, name, None)
                for name in ("max_position_embeddings", "n_positions", "seq_length")
            )
        usable = [
            int(value)
            for value in raw_values
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**8
        ]
        return min(usable) if usable else None

    @staticmethod
    def _device_label(engine: Any) -> str:
        device_map = getattr(getattr(engine, "model", None), "hf_device_map", None)
        if isinstance(device_map, Mapping) and device_map:
            devices = sorted({str(device) for device in device_map.values()})
            return ", ".join(devices)
        try:
            return str(next(engine.model.parameters()).device)
        except (AttributeError, StopIteration, TypeError):
            return "unknown"

    def stream_chat(
        self,
        messages: Sequence[ChatTurn],
        *,
        route: DesktopRoute | Mapping[str, float] | None,
        settings: DesktopGenerationSettings,
        cancellation: threading.Event,
        routing_policy: AutoRouterPolicy | None = None,
    ) -> Iterator[DesktopGeneration]:
        from molor.experimental import GenerationSettings

        decision = self._routing_decision(messages, route, routing_policy)
        # Routing is a distinct pre-generation event so the desktop can render the
        # selected expert and the complete probability distribution before token 1.
        yield DesktopGeneration(routing_decision=decision)
        expert_route = dict(decision.applied_route)
        desktop_settings = settings
        temperature = float(desktop_settings.temperature)
        runtime_settings = GenerationSettings(
            temperature=temperature if temperature > 0 else 1.0,
            top_p=float(desktop_settings.top_p),
            top_k=int(desktop_settings.top_k),
            max_new_tokens=int(desktop_settings.max_new_tokens),
            do_sample=temperature > 0,
            context_token_budget=desktop_settings.context_token_budget,
            cache_mode=desktop_settings.cache_mode,
        )
        payload = [{"role": turn.role, "content": turn.content} for turn in messages]
        instruction = desktop_settings.system_instruction.strip()
        has_system = bool(payload and payload[0]["role"] == "system")
        if instruction and not has_system:
            payload.insert(0, {"role": "system", "content": instruction})
        updates = self.engine.stream_chat(
            payload,
            route=expert_route,
            normalize_route=True,
            settings=runtime_settings,
            cancellation=cancellation,
            enable_thinking=bool(desktop_settings.enable_thinking),
        )
        reasoning_parser = _QwenReasoningParser(
            enabled=bool(desktop_settings.enable_thinking)
        )
        for update in updates:
            separated = reasoning_parser.feed(str(update.text or ""))
            if update.finished:
                tail = reasoning_parser.finish()
                separated = _SeparatedGenerationText(
                    answer=separated.answer + tail.answer,
                    reasoning=separated.reasoning + tail.reasoning,
                    reasoning_state=tail.reasoning_state,
                )
            stats = None
            if update.stats is not None:
                stats = DesktopStats(
                    model_id=str(update.stats.model_id),
                    device=str(update.stats.device),
                    dtype=str(update.stats.dtype),
                    route=tuple(update.stats.route),
                    prompt_tokens=int(update.stats.prompt_tokens),
                    completion_tokens=int(update.stats.completion_tokens),
                    context_tokens=int(update.stats.context_tokens),
                    context_limit=self.info.context_limit,
                    context_token_budget=getattr(update.stats, "context_token_budget", None),
                    dropped_turns=int(getattr(update.stats, "dropped_turns", 0)),
                    dropped_tokens=int(getattr(update.stats, "dropped_tokens", 0)),
                    cache_mode=str(
                        getattr(update.stats, "cache_mode", desktop_settings.cache_mode)
                    ),
                    elapsed_seconds=float(update.stats.elapsed_seconds),
                    time_to_first_token_seconds=update.stats.time_to_first_token_seconds,
                    tokens_per_second=update.stats.tokens_per_second,
                    finish_reason=(
                        str(update.stats.finish_reason)
                        if update.stats.finish_reason is not None
                        else None
                    ),
                )
            yield DesktopGeneration(
                text=separated.answer,
                reasoning_text=separated.reasoning,
                reasoning_state=separated.reasoning_state,
                stats=stats,
                finished=bool(update.finished),
            )

    def _routing_decision(
        self,
        messages: Sequence[ChatTurn],
        route: DesktopRoute | Mapping[str, float] | None,
        policy: AutoRouterPolicy | None,
    ) -> AutoRouteDecision:
        manual_weights = self._manual_route_mapping(route)
        if policy is None:
            return manual_route_decision(self.info.expert_names, manual_weights)
        if not isinstance(policy, AutoRouterPolicy):
            raise TypeError("routing_policy must be an AutoRouterPolicy or None")
        if policy.mode == "manual":
            configured = dict(policy.manual_route)
            return manual_route_decision(
                self.info.expert_names,
                configured if configured else manual_weights,
            )
        if self.auto_router is None:
            return base_fallback_decision(
                self.info.expert_names,
                reason=self.router_unavailable_reason,
            )
        prompt = next(
            (
                turn.content
                for turn in reversed(tuple(messages))
                if turn.role == "user" and turn.content.strip()
            ),
            "",
        )
        if not prompt:
            return base_fallback_decision(
                self.info.expert_names,
                reason="No current-user prompt was available for automatic routing.",
            )
        return self.auto_router.decide(prompt, policy)

    def _manual_route_mapping(
        self,
        route: DesktopRoute | Mapping[str, float] | None,
    ) -> dict[str, float]:
        if route is None:
            return {"base": 1.0}
        if isinstance(route, Mapping):
            return dict(route)
        if not isinstance(route, DesktopRoute):
            raise TypeError("route must be a DesktopRoute, mapping, or None")
        mixture = route.normalized()
        weights: dict[str, float] = {"base": mixture.base}
        if self.info.expert_a_name:
            weights[self.info.expert_a_name] = (
                weights.get(self.info.expert_a_name, 0.0) + mixture.expert_a
            )
        if self.info.expert_b_name:
            weights[self.info.expert_b_name] = (
                weights.get(self.info.expert_b_name, 0.0) + mixture.expert_b
            )
        return weights
