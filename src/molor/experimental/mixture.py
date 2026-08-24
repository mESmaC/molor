"""A deliberately small Hugging Face runtime for manually mixed LoRA experts.

This module is the fast experimental path.  It instruments a normal causal LM in
place, keeps every adapter resident, and lets the unchanged model ``forward`` and
``generate`` methods read one shared route controller.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from queue import Empty
import re
import threading
import time
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

BASE_ROUTE = "base"
QWEN3_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_EXPERT_FORMAT = "molor.experimental-lora.v1"
_EXPERT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
CacheMode = Literal["dynamic", "offloaded", "disabled"]


def _validate_expert_name(name: str) -> str:
    if not isinstance(name, str) or not _EXPERT_NAME.fullmatch(name) or name == BASE_ROUTE:
        raise ValueError(
            "expert name must be 1-64 letters, digits, underscores, or hyphens "
            f"and cannot be {BASE_ROUTE!r}"
        )
    return name


def _validate_lora_hyperparameters(rank: int, alpha: float, dropout: float) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a positive finite number")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive finite number")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout must be finite and in [0, 1)")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1)")


class RouteController:
    """One route shared by every instrumented projection.

    The ``base`` route is a null adapter.  The frozen base projection is always
    evaluated; assigning mass to ``base`` simply reduces the remaining adapter mass.
    A context-local override lets generation requests use different routes without
    adding a routing argument to the Hugging Face model.
    """

    def __init__(self) -> None:
        self._expert_names: list[str] = []
        self._route: tuple[tuple[str, float], ...] = ((BASE_ROUTE, 1.0),)
        self._override: ContextVar[tuple[tuple[str, float], ...] | None] = ContextVar(
            "molor_expert_route", default=None
        )
        self._lock = threading.RLock()

    @property
    def expert_names(self) -> tuple[str, ...]:
        return tuple(self._expert_names)

    @property
    def route_names(self) -> tuple[str, ...]:
        return (BASE_ROUTE, *self._expert_names)

    @property
    def route(self) -> dict[str, float]:
        return dict(self.current())

    def register_expert(self, name: str) -> None:
        name = _validate_expert_name(name)
        with self._lock:
            if name in self._expert_names:
                raise ValueError(f"expert {name!r} already exists")
            self._expert_names.append(name)

    def unregister_expert(self, name: str) -> None:
        with self._lock:
            self._expert_names.remove(name)
            route = tuple((key, value) for key, value in self._route if key != name)
            self._route = route or ((BASE_ROUTE, 1.0),)

    def normalize(
        self, weights: Mapping[str, float] | None, *, normalize: bool
    ) -> tuple[tuple[str, float], ...]:
        if weights is None:
            return self.current()
        if not isinstance(weights, Mapping):
            raise TypeError("route weights must be a mapping")
        allowed = set(self.route_names)
        values: dict[str, float] = {}
        for name, raw_weight in weights.items():
            if name not in allowed:
                raise KeyError(f"unknown route {name!r}; available routes are {self.route_names!r}")
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise ValueError(f"route weight for {name!r} must be finite and non-negative")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"route weight for {name!r} must be finite and non-negative")
            if weight:
                values[name] = weight
        if not values:
            return ((BASE_ROUTE, 1.0),)
        if normalize:
            total = sum(values.values())
            values = {name: value / total for name, value in values.items()}
        return tuple((name, values[name]) for name in self.route_names if name in values)

    def set_route(
        self, weights: Mapping[str, float] | None, *, normalize: bool = True
    ) -> dict[str, float]:
        route = self.normalize(weights, normalize=normalize)
        with self._lock:
            self._route = route
        return dict(route)

    def current(self) -> tuple[tuple[str, float], ...]:
        override = self._override.get()
        return self._route if override is None else override

    @contextmanager
    def using(
        self, weights: Mapping[str, float] | None, *, normalize: bool = True
    ) -> Iterator[dict[str, float]]:
        route = self.normalize(weights, normalize=normalize)
        token = self._override.set(route)
        try:
            yield dict(route)
        finally:
            self._override.reset(token)


class _LoRADelta(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / rank
        self.dropout_probability = float(dropout)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs: Tensor) -> Tensor:
        branch_inputs = self.dropout(inputs).to(self.lora_A.dtype)
        branch = F.linear(F.linear(branch_inputs, self.lora_A), self.lora_B)
        return branch.to(inputs.dtype) * self.scaling


class MixtureLinear(nn.Module):
    """A frozen linear projection plus any number of resident LoRA deltas."""

    def __init__(self, base: nn.Linear, controller: RouteController) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be torch.nn.Linear")
        self.base = base
        self.base.requires_grad_(False)
        self.controller = controller
        self.experts = nn.ModuleDict()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def add_expert(
        self,
        name: str,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        parameter_dtype: torch.dtype | None = None,
    ) -> None:
        _validate_expert_name(name)
        _validate_lora_hyperparameters(rank, alpha, dropout)
        if name in self.experts:
            raise ValueError(f"expert {name!r} already exists in this projection")
        expert = _LoRADelta(
            self.in_features,
            self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=self.base.weight.device,
            dtype=parameter_dtype or self.base.weight.dtype,
        )
        expert.requires_grad_(False)
        self.experts[name] = expert
        # Modules attached after ``model.eval()`` otherwise keep PyTorch's
        # default training state and apply LoRA dropout during inference.
        expert.train(self.training)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        for name, weight in self.controller.current():
            if name != BASE_ROUTE and weight:
                output = output + self.experts[name](inputs) * weight
        return output


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    name: str
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    context_token_budget: int | None = None
    cache_mode: CacheMode = "dynamic"

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.context_token_budget is not None and (
            not isinstance(self.context_token_budget, int)
            or isinstance(self.context_token_budget, bool)
            or self.context_token_budget < 1
        ):
            raise ValueError("context_token_budget must be a positive integer or None")
        if self.cache_mode not in ("dynamic", "offloaded", "disabled"):
            raise ValueError("cache_mode must be 'dynamic', 'offloaded', or 'disabled'")

    def generation_kwargs(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("context_token_budget")
        cache_mode = values.pop("cache_mode")
        if not self.do_sample:
            values["temperature"] = None
            values["top_p"] = None
            values["top_k"] = None
        if cache_mode == "disabled":
            values["use_cache"] = False
        else:
            values["use_cache"] = True
            values["cache_implementation"] = cache_mode
        return values


@dataclass(frozen=True, slots=True)
class GenerationStats:
    model_id: str
    device: str
    dtype: str
    route: tuple[tuple[str, float], ...]
    prompt_tokens: int
    completion_tokens: int
    context_tokens: int
    context_limit: int | None
    elapsed_seconds: float
    time_to_first_token_seconds: float | None
    tokens_per_second: float
    finish_reason: str | None = None
    context_token_budget: int | None = None
    dropped_turns: int = 0
    dropped_tokens: int = 0
    cache_mode: CacheMode = "dynamic"


@dataclass(frozen=True, slots=True)
class GenerationUpdate:
    text: str
    stats: GenerationStats
    finished: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedChat:
    messages: tuple[dict[str, str], ...]
    inputs: Mapping[str, Tensor]
    prompt_tokens: int
    dropped_turns: int
    dropped_tokens: int


def _render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    template_kwargs: Mapping[str, object],
) -> tuple[Mapping[str, Tensor], int]:
    prompt = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if "input_ids" not in inputs or not isinstance(inputs["input_ids"], Tensor):
        raise RuntimeError("tokenizer did not return tensor input_ids")
    return inputs, int(inputs["input_ids"].shape[-1])


def _trim_chat_to_budget(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_prompt_tokens: int,
    template_kwargs: Mapping[str, object],
) -> _PreparedChat:
    """Drop oldest complete user turns until the exact rendered prompt fits."""

    if max_prompt_tokens < 1:
        raise ValueError("the context budget leaves no room for a prompt")
    prepared: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise ValueError("each chat message requires string role and content fields")
        prepared.append({"role": role, "content": content})
    user_indices = [index for index, message in enumerate(prepared) if message["role"] == "user"]
    if not user_indices:
        raise ValueError("chat messages must contain a user turn")

    initial_inputs, initial_tokens = _render_chat(tokenizer, prepared, template_kwargs)
    if initial_tokens <= max_prompt_tokens:
        return _PreparedChat(tuple(prepared), initial_inputs, initial_tokens, 0, 0)

    removable_turns: list[set[int]] = []
    for turn_number, start in enumerate(user_indices[:-1]):
        end = user_indices[turn_number + 1]
        removable_turns.append(
            {index for index in range(start, end) if prepared[index]["role"] != "system"}
        )

    dropped_indices: set[int] = set()
    current_inputs = initial_inputs
    current_tokens = initial_tokens
    dropped_turns = 0
    for turn in removable_turns:
        dropped_indices.update(turn)
        dropped_turns += 1
        current_messages = [
            message for index, message in enumerate(prepared) if index not in dropped_indices
        ]
        current_inputs, current_tokens = _render_chat(tokenizer, current_messages, template_kwargs)
        if current_tokens <= max_prompt_tokens:
            return _PreparedChat(
                tuple(current_messages),
                current_inputs,
                current_tokens,
                dropped_turns,
                max(0, initial_tokens - current_tokens),
            )

    raise ValueError(
        "context budget is too small to preserve all system messages and the newest user turn: "
        f"prompt needs {current_tokens} tokens but only {max_prompt_tokens} are available"
    )


class _CountingStreamer:
    """Count generated token IDs while delegating decoding to HF's streamer."""

    def __init__(self, streamer: Any) -> None:
        self.streamer = streamer
        self.completion_tokens = 0
        self._first_put = True
        self._ended = False
        self._lock = threading.Lock()

    def put(self, value: Tensor) -> None:
        if self._first_put:
            self._first_put = False
        else:
            self.completion_tokens += int(value.numel())
        self.streamer.put(value)

    def end(self) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
            self.streamer.end()

    def __iter__(self) -> Any:
        return iter(self.streamer)

    def __next__(self) -> str:
        return next(self.streamer)


class MixtureEngine:
    """A frozen causal LM, resident LoRA experts, and manual routed generation."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any | None,
        base_model_id: str,
        target_modules: Sequence[str],
        controller: RouteController,
        layers: Mapping[str, MixtureLinear],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.base_model_id = base_model_id
        self.target_modules = tuple(target_modules)
        self.controller = controller
        self.layers = dict(layers)
        self._expert_configs: dict[str, ExpertConfig] = {}
        self._generation_lock = threading.Lock()

    @classmethod
    def attach(
        cls,
        model: nn.Module,
        tokenizer: Any | None = None,
        *,
        base_model_id: str | None = None,
        target_modules: Sequence[str] = QWEN3_TARGET_MODULES,
    ) -> MixtureEngine:
        """Freeze and instrument selected ``nn.Linear`` projections in place."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        targets = tuple(dict.fromkeys(target_modules))
        if not targets or any(not isinstance(name, str) or not name for name in targets):
            raise ValueError("target_modules must contain at least one module suffix")
        if any(isinstance(module, MixtureLinear) for module in model.modules()):
            raise ValueError("model is already instrumented with mixture LoRA layers")

        was_training = model.training
        model.requires_grad_(False)
        controller = RouteController()
        matches = [
            (path, module)
            for path, module in model.named_modules()
            if isinstance(module, nn.Linear) and path.rsplit(".", maxsplit=1)[-1] in targets
        ]
        if not matches:
            raise ValueError(f"no nn.Linear modules matched target suffixes {targets!r}")

        layers: dict[str, MixtureLinear] = {}
        for path, module in matches:
            parent_path, _, child_name = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            replacement = MixtureLinear(module, controller)
            setattr(parent, child_name, replacement)
            layers[path] = replacement

        if base_model_id is None:
            config = getattr(model, "config", None)
            base_model_id = str(getattr(config, "_name_or_path", model.__class__.__name__))
        engine = cls(
            model=model,
            tokenizer=tokenizer,
            base_model_id=base_model_id,
            target_modules=targets,
            controller=controller,
            layers=layers,
        )
        # Replacements are newly constructed modules, so explicitly inherit
        # the model's pre-instrumentation train/eval state.
        model.train(was_training)
        return engine

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | Mapping[str, object] | None = "auto",
        target_modules: Sequence[str] = QWEN3_TARGET_MODULES,
        revision: str | None = None,
        local_files_only: bool = False,
        attn_implementation: str | None = None,
    ) -> MixtureEngine:
        """Load a real Hugging Face causal LM and instrument it for LoRA mixtures."""

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise RuntimeError("install molor[hf] to load Hugging Face models") from error

        common: dict[str, object] = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
        model_kwargs = {
            **common,
            "dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return cls.attach(
            model,
            tokenizer,
            base_model_id=model_id,
            target_modules=target_modules,
        )

    @property
    def expert_names(self) -> tuple[str, ...]:
        return self.controller.expert_names

    @property
    def route_names(self) -> tuple[str, ...]:
        return self.controller.route_names

    @property
    def route(self) -> dict[str, float]:
        return self.controller.route

    @property
    def context_limit(self) -> int | None:
        candidates: list[object] = []
        config = getattr(self.model, "config", None)
        if config is not None:
            candidates.extend(
                getattr(config, name, None)
                for name in ("max_position_embeddings", "n_positions", "seq_length")
            )
        candidates.append(getattr(self.tokenizer, "model_max_length", None))
        limits = [
            int(value)
            for value in candidates
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 100_000_000
        ]
        return min(limits) if limits else None

    def add_expert(
        self,
        name: str,
        *,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        parameter_dtype: torch.dtype | None = None,
    ) -> ExpertConfig:
        name = _validate_expert_name(name)
        _validate_lora_hyperparameters(rank, alpha, dropout)
        if name in self._expert_configs:
            raise ValueError(f"expert {name!r} already exists")
        self.controller.register_expert(name)
        try:
            for layer in self.layers.values():
                layer.add_expert(
                    name,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    parameter_dtype=parameter_dtype,
                )
        except Exception:
            for layer in self.layers.values():
                if name in layer.experts:
                    del layer.experts[name]
            self.controller.unregister_expert(name)
            raise
        config = ExpertConfig(name, rank, float(alpha), float(dropout), self.target_modules)
        self._expert_configs[name] = config
        return config

    def set_route(
        self, weights: Mapping[str, float] | None, *, normalize: bool = True
    ) -> dict[str, float]:
        return self.controller.set_route(weights, normalize=normalize)

    @contextmanager
    def use_route(
        self, weights: Mapping[str, float] | None, *, normalize: bool = True
    ) -> Iterator[dict[str, float]]:
        with self.controller.using(weights, normalize=normalize) as route:
            yield route

    def set_trainable_expert(self, name: str) -> tuple[nn.Parameter, ...]:
        if name not in self._expert_configs:
            raise KeyError(f"unknown expert {name!r}")
        for layer in self.layers.values():
            for expert_name, expert in layer.experts.items():
                expert.requires_grad_(expert_name == name)
        self.set_route({name: 1.0}, normalize=False)
        return self.expert_parameters(name)

    def expert_parameters(self, name: str) -> tuple[nn.Parameter, ...]:
        if name not in self._expert_configs:
            raise KeyError(f"unknown expert {name!r}")
        return tuple(
            parameter
            for layer in self.layers.values()
            for parameter in layer.experts[name].parameters()
        )

    def save_expert(self, name: str, directory: str | Path) -> Path:
        """Save one expert without serializing the frozen base or its peers."""

        if name not in self._expert_configs:
            raise KeyError(f"unknown expert {name!r}")
        try:
            from safetensors.torch import save_file
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise RuntimeError("install molor[hf] to save adapter safetensors") from error

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        config = self._expert_configs[name]
        payload = {
            "format": _EXPERT_FORMAT,
            "name": name,
            "base_model_id": self.base_model_id,
            "rank": config.rank,
            "alpha": config.alpha,
            "dropout": config.dropout,
            "target_modules": list(config.target_modules),
            "modules": {
                path: {
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                }
                for path, layer in self.layers.items()
            },
        }
        (output / "adapter_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tensors: dict[str, Tensor] = {}
        for path, layer in self.layers.items():
            expert = layer.experts[name]
            tensors[f"{path}.lora_A"] = expert.lora_A.detach().cpu().contiguous()
            tensors[f"{path}.lora_B"] = expert.lora_B.detach().cpu().contiguous()
        save_file(tensors, output / "adapter_model.safetensors")
        return output

    def load_expert(
        self,
        directory: str | Path,
        *,
        name: str | None = None,
        parameter_dtype: torch.dtype | None = None,
    ) -> str:
        """Load one saved expert into every already-instrumented projection."""

        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise RuntimeError("install molor[hf] to load adapter safetensors") from error

        source = Path(directory)
        payload = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
        if payload.get("format") != _EXPERT_FORMAT:
            raise ValueError("unsupported experimental adapter format")
        loaded_name = _validate_expert_name(name or payload["name"])
        if payload.get("base_model_id") != self.base_model_id:
            raise ValueError("adapter targets a different base model")
        if tuple(payload.get("target_modules", ())) != self.target_modules:
            raise ValueError("adapter target modules differ from this engine")
        modules = payload.get("modules")
        if not isinstance(modules, dict) or set(modules) != set(self.layers):
            raise ValueError("adapter projection set differs from this engine")

        self.add_expert(
            loaded_name,
            rank=int(payload["rank"]),
            alpha=float(payload["alpha"]),
            dropout=float(payload["dropout"]),
            parameter_dtype=parameter_dtype,
        )
        try:
            tensors = load_file(source / "adapter_model.safetensors", device="cpu")
            expected = {
                f"{path}.{tensor_name}"
                for path in self.layers
                for tensor_name in ("lora_A", "lora_B")
            }
            if set(tensors) != expected:
                raise ValueError("adapter tensor set differs from this engine")
            with torch.no_grad():
                for path, layer in self.layers.items():
                    expert = layer.experts[loaded_name]
                    expert.lora_A.copy_(tensors[f"{path}.lora_A"])
                    expert.lora_B.copy_(tensors[f"{path}.lora_B"])
        except Exception:
            self._remove_expert(loaded_name)
            raise
        return loaded_name

    def _remove_expert(self, name: str) -> None:
        for layer in self.layers.values():
            if name in layer.experts:
                del layer.experts[name]
        self._expert_configs.pop(name, None)
        if name in self.controller.expert_names:
            self.controller.unregister_expert(name)

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        route: Mapping[str, float] | None = None,
        normalize_route: bool = True,
        settings: GenerationSettings | None = None,
        cancellation: threading.Event | None = None,
        enable_thinking: bool | None = None,
    ) -> Iterator[GenerationUpdate]:
        """Stream decoded text and live stats from the real Hugging Face model."""

        if self.tokenizer is None:
            raise RuntimeError("stream_chat requires a tokenizer")
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise RuntimeError("install molor[hf] to stream Hugging Face generation") from error

        settings = settings or GenerationSettings()
        external_cancel = cancellation or threading.Event()
        local_cancel = threading.Event()
        route_snapshot = self.controller.normalize(route, normalize=normalize_route)
        route_mapping = dict(route_snapshot)

        template_kwargs: dict[str, object] = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        model_context_limit = self.context_limit
        if settings.context_token_budget is None:
            if model_context_limit is None:
                raise ValueError(
                    "context_token_budget is required when the model context limit is unknown"
                )
            context_token_budget = model_context_limit
        elif model_context_limit is None:
            context_token_budget = settings.context_token_budget
        else:
            context_token_budget = min(settings.context_token_budget, model_context_limit)
        max_prompt_tokens = context_token_budget - settings.max_new_tokens
        prepared_chat = _trim_chat_to_budget(
            self.tokenizer,
            messages,
            max_prompt_tokens=max_prompt_tokens,
            template_kwargs=template_kwargs,
        )
        inputs = prepared_chat.inputs
        input_device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        prompt_tokens = prepared_chat.prompt_tokens

        class CancelOnEvent(StoppingCriteria):
            def __call__(self, *args: object, **kwargs: object) -> bool:
                return external_cancel.is_set() or local_cancel.is_set()

        decoded_stream = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.1,
            clean_up_tokenization_spaces=False,
        )
        streamer = _CountingStreamer(decoded_stream)
        errors: list[BaseException] = []
        started = time.perf_counter()
        first_token_at: float | None = None

        def stats(*, finish_reason: str | None = None) -> GenerationStats:
            elapsed = max(time.perf_counter() - started, 1e-9)
            completion_tokens = streamer.completion_tokens
            first = None if first_token_at is None else first_token_at - started
            parameter = next(self.model.parameters())
            return GenerationStats(
                model_id=self.base_model_id,
                device=str(input_device),
                dtype=str(parameter.dtype).removeprefix("torch."),
                route=route_snapshot,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                context_tokens=prompt_tokens + completion_tokens,
                context_limit=model_context_limit,
                elapsed_seconds=elapsed,
                time_to_first_token_seconds=first,
                tokens_per_second=completion_tokens / elapsed,
                finish_reason=finish_reason,
                context_token_budget=context_token_budget,
                dropped_turns=prepared_chat.dropped_turns,
                dropped_tokens=prepared_chat.dropped_tokens,
                cache_mode=settings.cache_mode,
            )

        def generate() -> None:
            try:
                with (
                    self._generation_lock,
                    self.controller.using(route_mapping, normalize=False),
                    torch.inference_mode(),
                ):
                    self.model.generate(
                        **inputs,
                        **settings.generation_kwargs(),
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([CancelOnEvent()]),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            except BaseException as error:
                errors.append(error)
            finally:
                streamer.end()

        worker = threading.Thread(target=generate, name="molor-generation", daemon=True)
        worker.start()
        try:
            iterator = iter(streamer)
            while True:
                try:
                    text = next(iterator)
                except Empty:
                    if not worker.is_alive():
                        break
                    continue
                except StopIteration:
                    break
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield GenerationUpdate(text=text, stats=stats())
            worker.join(timeout=1.0)
            if errors:
                raise errors[0]
            if external_cancel.is_set() or local_cancel.is_set():
                reason = "cancelled"
            elif streamer.completion_tokens >= settings.max_new_tokens:
                reason = "length"
            else:
                reason = "stop"
            yield GenerationUpdate(text="", stats=stats(finish_reason=reason), finished=True)
        finally:
            local_cancel.set()
            worker.join(timeout=2.0)


__all__ = [
    "BASE_ROUTE",
    "QWEN3_TARGET_MODULES",
    "CacheMode",
    "ExpertConfig",
    "GenerationSettings",
    "GenerationStats",
    "GenerationUpdate",
    "MixtureEngine",
    "MixtureLinear",
    "RouteController",
]
