"""Configurable prompt-level routing for the fast experimental runtime.

The learned component is intentionally small: frozen input-token embeddings are
masked-mean pooled and passed through :class:`molor.router.TopKRouter`.  Runtime
policy (temperature, top-k, confidence fallback, and route masks) is applied after
the learned logits so experimenting with a knob does not mutate checkpoint identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from molor.config import RouterConfig
from molor.router import TopKRouter


BASE_ROUTE = "base"
NULL_ROUTE_ID = "base/null"
ROUTER_CHECKPOINT_SCHEMA = "molor.experimental-router-checkpoint.v1"


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _route_pairs(
    values: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    field_name: str,
    allow_zero: bool,
) -> tuple[tuple[str, float], ...]:
    items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    seen: set[str] = set()
    normalized: list[tuple[str, float]] = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{field_name} must contain (route, weight) pairs")
        name, raw_weight = item
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field_name} route names must be non-empty strings")
        if name in seen:
            raise ValueError(f"{field_name} contains duplicate route {name!r}")
        weight = _finite(raw_weight, f"{field_name}[{name!r}]")
        if weight < 0 or (not allow_zero and weight == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{field_name} weights must be {qualifier}")
        seen.add(name)
        normalized.append((name, weight))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class AutoRouterPolicy:
    """Small set of request-time controls around fixed learned router logits."""

    mode: Literal["auto", "manual"] = "auto"
    top_k: int = 1
    temperature: float = 1.0
    confidence_threshold: float = 0.0
    null_bias: float = 0.0
    max_prompt_tokens: int = 256
    enabled_experts: tuple[str, ...] = ()
    manual_route: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "manual"):
            raise ValueError("mode must be 'auto' or 'manual'")
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or not 1 <= self.top_k <= 64
        ):
            raise ValueError("top_k must be an integer from 1 to 64")
        temperature = _finite(self.temperature, "temperature")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        threshold = _finite(self.confidence_threshold, "confidence_threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        null_bias = _finite(self.null_bias, "null_bias")
        if (
            not isinstance(self.max_prompt_tokens, int)
            or isinstance(self.max_prompt_tokens, bool)
            or not 1 <= self.max_prompt_tokens <= 65_536
        ):
            raise ValueError("max_prompt_tokens must be an integer from 1 to 65536")
        if isinstance(self.enabled_experts, (str, bytes)):
            raise TypeError("enabled_experts must be a sequence of expert names")
        enabled = tuple(self.enabled_experts)
        if any(not isinstance(name, str) or not name or name == BASE_ROUTE for name in enabled):
            raise ValueError("enabled_experts must contain non-empty non-base route names")
        if len(enabled) != len(set(enabled)):
            raise ValueError("enabled_experts must not contain duplicates")
        if not isinstance(self.manual_route, Mapping):
            raise TypeError("manual_route must be a route-to-weight mapping")
        manual = _route_pairs(
            self.manual_route,
            field_name="manual_route",
            allow_zero=True,
        )
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "confidence_threshold", threshold)
        object.__setattr__(self, "null_bias", null_bias)
        object.__setattr__(self, "enabled_experts", enabled)
        object.__setattr__(self, "manual_route", MappingProxyType(dict(manual)))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold,
            "null_bias": self.null_bias,
            "max_prompt_tokens": self.max_prompt_tokens,
            "enabled_experts": list(self.enabled_experts),
            "manual_route": dict(self.manual_route),
        }


@dataclass(frozen=True, slots=True)
class AutoRouteDecision:
    """Observable result of either learned or manual prompt routing."""

    mode: Literal["auto", "manual"]
    applied_route: tuple[tuple[str, float], ...]
    probabilities: tuple[tuple[str, float], ...]
    selected_routes: tuple[str, ...]
    confidence: float
    reason: str
    checkpoint_loaded: bool

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "manual"):
            raise ValueError("mode must be 'auto' or 'manual'")
        applied = _route_pairs(
            self.applied_route,
            field_name="applied_route",
            allow_zero=False,
        )
        probabilities = _route_pairs(
            self.probabilities,
            field_name="probabilities",
            allow_zero=True,
        )
        if not applied or not probabilities:
            raise ValueError("a route decision must contain applied routes and probabilities")
        probability_names = {name for name, _weight in probabilities}
        if any(name not in probability_names for name, _weight in applied):
            raise ValueError("applied routes must be present in probabilities")
        if not math.isclose(sum(weight for _name, weight in applied), 1.0, abs_tol=1e-6):
            raise ValueError("applied route weights must sum to one")
        if not math.isclose(
            sum(weight for _name, weight in probabilities),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("route probabilities must sum to one")
        selected = tuple(self.selected_routes)
        if selected != tuple(name for name, _weight in applied):
            raise ValueError("selected_routes must match the applied route order")
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.checkpoint_loaded, bool):
            raise TypeError("checkpoint_loaded must be boolean")
        object.__setattr__(self, "applied_route", applied)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "selected_routes", selected)
        object.__setattr__(self, "confidence", confidence)

    @property
    def route(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.applied_route))

    @property
    def probability_map(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.probabilities))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "applied_route": dict(self.applied_route),
            "probabilities": dict(self.probabilities),
            "selected_routes": list(self.selected_routes),
            "confidence": self.confidence,
            "reason": self.reason,
            "checkpoint_loaded": self.checkpoint_loaded,
        }


def base_fallback_decision(
    expert_names: Sequence[str],
    *,
    reason: str,
    mode: Literal["auto", "manual"] = "auto",
) -> AutoRouteDecision:
    """Return an explicit base-only result over the full loaded route axis."""

    route_names = _runtime_route_names(expert_names)
    return AutoRouteDecision(
        mode=mode,
        applied_route=((BASE_ROUTE, 1.0),),
        probabilities=tuple(
            (route_name, 1.0 if route_name == BASE_ROUTE else 0.0) for route_name in route_names
        ),
        selected_routes=(BASE_ROUTE,),
        confidence=1.0,
        reason=reason,
        checkpoint_loaded=False,
    )


def manual_route_decision(
    expert_names: Sequence[str],
    weights: Mapping[str, float],
) -> AutoRouteDecision:
    """Validate and normalize an arbitrary named manual route."""

    route_names = _runtime_route_names(expert_names)
    allowed = set(route_names)
    supplied = _route_pairs(weights, field_name="manual_route", allow_zero=True)
    unknown = tuple(name for name, _weight in supplied if name not in allowed)
    if unknown:
        raise KeyError(f"unknown manual routes: {unknown!r}; loaded routes are {route_names!r}")
    positive = {name: weight for name, weight in supplied if weight > 0}
    if not positive:
        positive = {BASE_ROUTE: 1.0}
    total = sum(positive.values())
    normalized = {name: value / total for name, value in positive.items()}
    applied = tuple((name, normalized[name]) for name in route_names if name in normalized)
    probabilities = tuple((name, normalized.get(name, 0.0)) for name in route_names)
    return AutoRouteDecision(
        mode="manual",
        applied_route=applied,
        probabilities=probabilities,
        selected_routes=tuple(name for name, _weight in applied),
        confidence=max(normalized.values()),
        reason="Applied the configured manual route.",
        checkpoint_loaded=False,
    )


class PromptAutoRouter:
    """One trained prompt router bound to an embedding table and ordered routes."""

    def __init__(
        self,
        *,
        router: TopKRouter,
        tokenizer: Any,
        input_embeddings: nn.Module,
        route_ids: Sequence[str],
        available_experts: Sequence[str],
        checkpoint_directory: Path,
        feature_contract: Mapping[str, object],
    ) -> None:
        if not isinstance(router, TopKRouter):
            raise TypeError("router must be a TopKRouter")
        if router.config.mode != "sequence":
            raise ValueError("the experimental prompt router requires sequence mode")
        route_ids = tuple(route_ids)
        if len(route_ids) != router.config.num_routes or len(route_ids) != len(set(route_ids)):
            raise ValueError("route_ids must exactly match the router output axis")
        if route_ids.count(NULL_ROUTE_ID) != 1:
            raise ValueError(f"route_ids must contain exactly one {NULL_ROUTE_ID!r} route")
        experts = _expert_names(available_experts)
        if not isinstance(feature_contract, Mapping):
            raise TypeError("feature_contract must be a mapping")
        if feature_contract.get("source") != "input_embeddings":
            raise ValueError("router feature source must be 'input_embeddings'")
        if feature_contract.get("pooling") != "masked_mean":
            raise ValueError("router pooling must be 'masked_mean'")
        if feature_contract.get("prompt_scope") not in (None, "user_content_only"):
            raise ValueError("router prompt scope must be 'user_content_only'")
        max_tokens = feature_contract.get("max_tokens")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 65_536
        ):
            raise ValueError("router feature max_tokens must be an integer from 1 to 65536")
        if feature_contract.get("padding") != "longest":
            raise ValueError("router feature padding must be 'longest'")
        if feature_contract.get("truncation") != "right":
            raise ValueError("router feature truncation must be 'right'")
        add_special_tokens = feature_contract.get("add_special_tokens")
        if not isinstance(add_special_tokens, bool):
            raise ValueError("router feature add_special_tokens must be boolean")
        embedding_width = _embedding_width(input_embeddings)
        if embedding_width != router.config.hidden_size:
            raise ValueError(
                f"router expects embedding width {router.config.hidden_size}, "
                f"but the model exposes {embedding_width}"
            )
        self.router = router
        self.tokenizer = tokenizer
        self.input_embeddings = input_embeddings
        self.route_ids = route_ids
        self.available_experts = experts
        self.checkpoint_directory = checkpoint_directory.resolve()
        self.feature_contract = MappingProxyType(dict(feature_contract))
        self.max_prompt_tokens = max_tokens
        self.add_special_tokens = add_special_tokens

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_directory: str | Path,
        *,
        tokenizer: Any,
        input_embeddings: nn.Module,
        available_experts: Sequence[str],
        expected_model_id: str | None = None,
    ) -> PromptAutoRouter:
        """Load the small router network and its sidecar contract from a directory."""

        directory = Path(checkpoint_directory).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"router checkpoint directory does not exist: {directory}")
        config_path = directory / "router_config.json"
        weights_path = directory / "router.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"router config does not exist: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"router weights do not exist: {weights_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("router_config.json must contain an object")
        if payload.get("schema_version") != ROUTER_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported experimental router checkpoint schema")
        raw_config = payload.get("router_config")
        if not isinstance(raw_config, dict):
            raise ValueError("router checkpoint has no router_config object")
        config = RouterConfig(**raw_config)
        raw_route_ids = payload.get("route_ids")
        if not isinstance(raw_route_ids, list) or any(
            not isinstance(route_id, str) or not route_id for route_id in raw_route_ids
        ):
            raise ValueError("router checkpoint has invalid route_ids")
        route_ids = tuple(raw_route_ids)
        if len(route_ids) != config.num_routes:
            raise ValueError("router checkpoint route count differs from router_config")
        feature_contract = payload.get("feature_contract")
        if not isinstance(feature_contract, dict):
            raise ValueError("router checkpoint has no feature_contract object")
        model_payload = payload.get("model")
        stored_model_id = (
            model_payload.get("model_id")
            if isinstance(model_payload, dict)
            else payload.get("model_id")
        )
        if not isinstance(stored_model_id, str) or not stored_model_id.strip():
            raise ValueError("router checkpoint has no frozen-base model identity")
        if expected_model_id is not None:
            if _model_identity(stored_model_id) != _model_identity(expected_model_id):
                raise ValueError("router checkpoint targets a different frozen base model")

        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("install molor[hf] to load router safetensors") from exc

        tensors = load_file(str(weights_path), device="cpu")
        if not tensors or any(not key.startswith("network.") for key in tensors):
            raise ValueError("router.safetensors must contain only network.* tensors")
        network_state = {key.removeprefix("network."): value for key, value in tensors.items()}
        router = TopKRouter(config)
        router.network.load_state_dict(network_state, strict=True)
        embedding_weight = getattr(input_embeddings, "weight", None)
        if not isinstance(embedding_weight, Tensor):
            raise TypeError("input_embeddings must expose a tensor weight")
        target_dtype = (
            embedding_weight.dtype if embedding_weight.dtype.is_floating_point else torch.float32
        )
        router.to(device=embedding_weight.device, dtype=target_dtype)
        router.requires_grad_(False)
        router.eval()
        return cls(
            router=router,
            tokenizer=tokenizer,
            input_embeddings=input_embeddings,
            route_ids=route_ids,
            available_experts=available_experts,
            checkpoint_directory=directory,
            feature_contract=feature_contract,
        )

    def decide(self, prompt: str, policy: AutoRouterPolicy) -> AutoRouteDecision:
        """Route one current-user prompt and keep that route fixed for generation."""

        if not isinstance(policy, AutoRouterPolicy):
            raise TypeError("policy must be an AutoRouterPolicy")
        if policy.mode == "manual":
            return manual_route_decision(self.available_experts, policy.manual_route)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("auto routing requires a non-empty current-user prompt")
        if policy.max_prompt_tokens != self.max_prompt_tokens:
            raise ValueError(
                "router max_prompt_tokens is pinned by its training contract at "
                f"{self.max_prompt_tokens}, not {policy.max_prompt_tokens}"
            )

        loaded = set(self.available_experts)
        checkpoint_experts = {route_id for route_id in self.route_ids if route_id != NULL_ROUTE_ID}
        if policy.enabled_experts:
            unknown_loaded = set(policy.enabled_experts).difference(loaded)
            if unknown_loaded:
                raise KeyError(f"enabled experts are not loaded: {sorted(unknown_loaded)!r}")
            unknown_checkpoint = set(policy.enabled_experts).difference(checkpoint_experts)
            if unknown_checkpoint:
                raise KeyError(
                    "enabled experts are absent from this router checkpoint: "
                    f"{sorted(unknown_checkpoint)!r}"
                )
            enabled = set(policy.enabled_experts)
        else:
            enabled = loaded.intersection(checkpoint_experts)

        # The prompt representation is part of checkpoint identity.
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
            padding=False,
            add_special_tokens=self.add_special_tokens,
        )
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("tokenizer returned invalid input_ids for prompt routing")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if not isinstance(attention_mask, Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("tokenizer returned an invalid attention_mask")
        embedding_weight = getattr(self.input_embeddings, "weight", None)
        if not isinstance(embedding_weight, Tensor):
            raise TypeError("input_embeddings must expose a tensor weight")
        device = embedding_weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            hidden_states = self.input_embeddings(input_ids)
            output = self.router(hidden_states, attention_mask)
            # TopKRouter applies its checkpoint temperature. Recover the learned raw
            # logits so the request-time temperature has one clear interpretation.
            logits = output.logits.float()[0] * self.router.config.temperature
            null_index = self.route_ids.index(NULL_ROUTE_ID)
            logits[null_index] += policy.null_bias
            allowed = torch.tensor(
                [route_id == NULL_ROUTE_ID or route_id in enabled for route_id in self.route_ids],
                device=logits.device,
                dtype=torch.bool,
            )
            logits = logits.masked_fill(~allowed, float("-inf"))
            probabilities = torch.softmax(logits / policy.temperature, dim=-1)

        checkpoint_probabilities = {
            _runtime_route_id(route_id): float(probabilities[index].item())
            for index, route_id in enumerate(self.route_ids)
        }
        route_names = _runtime_route_names(self.available_experts)
        all_probabilities = tuple(
            (route_name, checkpoint_probabilities.get(route_name, 0.0))
            for route_name in route_names
        )
        confidence = max(value for _name, value in all_probabilities)
        if confidence < policy.confidence_threshold:
            return AutoRouteDecision(
                mode="auto",
                applied_route=((BASE_ROUTE, 1.0),),
                probabilities=all_probabilities,
                selected_routes=(BASE_ROUTE,),
                confidence=confidence,
                reason=(
                    f"Router confidence {confidence:.3f} was below the configured "
                    f"{policy.confidence_threshold:.3f} threshold; used base."
                ),
                checkpoint_loaded=True,
            )

        candidate_count = int(torch.count_nonzero(allowed).item())
        effective_top_k = min(policy.top_k, candidate_count)
        selected_indices = probabilities.topk(effective_top_k).indices.tolist()
        selected_weights = probabilities[selected_indices]
        if effective_top_k == 1:
            selected_weights = torch.ones_like(selected_weights)
        else:
            selected_weights = selected_weights / selected_weights.sum()
        selected_by_name = {
            _runtime_route_id(self.route_ids[index]): float(weight.item())
            for index, weight in zip(selected_indices, selected_weights, strict=True)
        }
        # Preserve posterior rank.  Resident rack order is an execution identity, not a
        # confidence order; reporting it here made the benchmark's first selected route
        # disagree with the actual top-1 posterior whenever a later resident won.
        selected_names = tuple(
            _runtime_route_id(self.route_ids[index]) for index in selected_indices
        )
        applied = tuple((name, selected_by_name[name]) for name in selected_names)
        return AutoRouteDecision(
            mode="auto",
            applied_route=applied,
            probabilities=all_probabilities,
            selected_routes=selected_names,
            confidence=confidence,
            reason="Learned prompt router selected " + ", ".join(selected_names) + ".",
            checkpoint_loaded=True,
        )


def _expert_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("expert names must be a sequence")
    names = tuple(values)
    if any(not isinstance(name, str) or not name or name == BASE_ROUTE for name in names):
        raise ValueError("expert names must be non-empty and cannot use the base route name")
    if len(names) != len(set(names)):
        raise ValueError("expert names must not contain duplicates")
    return names


def _runtime_route_names(expert_names: Sequence[str]) -> tuple[str, ...]:
    return (BASE_ROUTE, *_expert_names(expert_names))


def _runtime_route_id(checkpoint_route_id: str) -> str:
    return BASE_ROUTE if checkpoint_route_id == NULL_ROUTE_ID else checkpoint_route_id


def _embedding_width(input_embeddings: nn.Module) -> int:
    raw = getattr(input_embeddings, "embedding_dim", None)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    weight = getattr(input_embeddings, "weight", None)
    if isinstance(weight, Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    raise TypeError("input_embeddings must expose embedding_dim or a rank-two weight")


def _model_identity(raw_model_id: str) -> str:
    candidate = Path(raw_model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve()).casefold()
    return raw_model_id.strip()


__all__ = [
    "AutoRouteDecision",
    "AutoRouterPolicy",
    "NULL_ROUTE_ID",
    "PromptAutoRouter",
    "ROUTER_CHECKPOINT_SCHEMA",
    "base_fallback_decision",
    "manual_route_decision",
]
