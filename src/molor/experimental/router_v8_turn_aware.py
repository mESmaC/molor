"""Quarantined representation primitives for the preregistered router-v8 study.

Nothing in this module loads a language model, a router checkpoint, an adapter, or
an evaluation artifact.  It deliberately has no integration with the live
backend.  The module only freezes deterministic conversation rendering, the two
representation arms, and the exact negative-delegation seam that a future
artifact-complete implementation may use.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

ROUTER_V8_SCHEMA = "molor.router-v8-turn-aware-admission-preregistration.v1"
ROUTER_V8_STATUS = "frozen_quarantined_scaffold_before_training"
ROUTER_V8_THRESHOLD = 0.5
ROUTER_V8_SEEDS = (83, 8419, 27183)
ROUTER_V8_MAX_TURNS = 12
ROUTER_V8_MAX_TOKENS = 512
ROUTER_V8_CONTROL_MAX_TOKENS = 256
ROUTER_V8_EMBEDDING_WIDTH = 2048
ROUTER_V8_ROUTE_AXIS = ("taskmaster-coffee-api-v2-full-dialog-r16", "base/null")
ROUTER_V8_ROLES = ("system", "user", "assistant", "tool")
ROUTER_V8_ROLE_TO_ID = MappingProxyType({role: index for index, role in enumerate(ROUTER_V8_ROLES)})


class TokenizerLike(Protocol):
    """Small tokenizer surface needed by the deterministic renderer."""

    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


def canonical_json_bytes(value: object) -> bytes:
    """Return the unique UTF-8 JSON encoding used for all v8 identities."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical-JSON serializable") from exc
    return text.encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_plain_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _token_ids(
    tokenizer: TokenizerLike,
    content: str,
    *,
    field: str,
    add_special_tokens: bool,
) -> tuple[int, ...]:
    encoded = tokenizer.encode(content, add_special_tokens=add_special_tokens)
    if isinstance(encoded, (str, bytes)) or not isinstance(encoded, Sequence):
        raise TypeError(f"tokenizer output for {field} must be a sequence")
    result: list[int] = []
    for index, token_id in enumerate(encoded):
        result.append(_require_plain_int(token_id, f"{field} token {index}"))
    if not result:
        raise ValueError(f"tokenizer output for {field} must not be empty")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class StructuredTurn:
    """One validated turn supplied to the quarantined admission representation."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROUTER_V8_ROLE_TO_ID:
            raise ValueError(f"role must be one of {ROUTER_V8_ROLES!r}")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not self.content.strip():
            raise ValueError("content must not be blank")


@dataclass(frozen=True, slots=True)
class TurnRenderingPolicy:
    """Deterministic newest-first budget policy.

    Production v8 uses :data:`FROZEN_RENDERING_POLICY`.  Allowing smaller values
    here makes the allocation algorithm directly testable without changing the
    frozen production identity.
    """

    max_turns: int = ROUTER_V8_MAX_TURNS
    max_tokens: int = ROUTER_V8_MAX_TOKENS
    add_special_tokens: bool = True
    allocation: str = "current_user_full_then_prior_turns_newest_first"
    partial_prior_turn: str = "retain_token_suffix_and_stop"
    output_order: str = "chronological"
    require_final_user_turn: bool = True

    def __post_init__(self) -> None:
        _require_plain_int(self.max_turns, "max_turns", minimum=1)
        _require_plain_int(self.max_tokens, "max_tokens", minimum=1)
        if self.max_turns > ROUTER_V8_MAX_TURNS:
            raise ValueError(f"max_turns must not exceed {ROUTER_V8_MAX_TURNS}")
        if not isinstance(self.add_special_tokens, bool):
            raise TypeError("add_special_tokens must be boolean")
        expected = {
            "allocation": "current_user_full_then_prior_turns_newest_first",
            "partial_prior_turn": "retain_token_suffix_and_stop",
            "output_order": "chronological",
            "require_final_user_turn": True,
        }
        for field, required in expected.items():
            if getattr(self, field) != required:
                raise ValueError(f"{field} must be {required!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "add_special_tokens": self.add_special_tokens,
            "allocation": self.allocation,
            "partial_prior_turn": self.partial_prior_turn,
            "output_order": self.output_order,
            "require_final_user_turn": self.require_final_user_turn,
            "overlong_current_user": "reject",
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


FROZEN_RENDERING_POLICY = TurnRenderingPolicy()


@dataclass(frozen=True, slots=True)
class RenderedTurn:
    """One chronological, budgeted turn and its explicit routing features."""

    role: str
    role_id: int
    recency_id: int
    source_index: int
    input_ids: tuple[int, ...]
    original_token_count: int
    truncated_left: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "role_id": self.role_id,
            "recency_id": self.recency_id,
            "source_index": self.source_index,
            "input_ids": list(self.input_ids),
            "original_token_count": self.original_token_count,
            "retained_token_count": len(self.input_ids),
            "truncated_left": self.truncated_left,
        }


@dataclass(frozen=True, slots=True)
class RenderedConversation:
    """Immutable representation output consumed by a future Arm-B feature cache."""

    turns: tuple[RenderedTurn, ...]
    source_turn_count: int
    rendering_policy_fingerprint: str

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("rendered conversation must contain at least one turn")
        if self.turns[-1].role != "user" or self.turns[-1].recency_id != 0:
            raise ValueError("rendered conversation must end with the current user")

    @property
    def token_count(self) -> int:
        return sum(len(turn.input_ids) for turn in self.turns)

    def to_dict(self) -> dict[str, object]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "source_turn_count": self.source_turn_count,
            "retained_turn_count": len(self.turns),
            "token_count": self.token_count,
            "rendering_policy_fingerprint": self.rendering_policy_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


class DeterministicTurnRenderer:
    """Render structured turns under an explicit, content-independent budget."""

    def __init__(self, policy: TurnRenderingPolicy = FROZEN_RENDERING_POLICY) -> None:
        if not isinstance(policy, TurnRenderingPolicy):
            raise TypeError("policy must be a TurnRenderingPolicy")
        self.policy = policy

    def render(
        self,
        turns: Sequence[StructuredTurn],
        *,
        tokenizer: TokenizerLike,
    ) -> RenderedConversation:
        if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
            raise TypeError("turns must be a sequence of StructuredTurn values")
        if not turns:
            raise ValueError("turns must not be empty")
        for index, turn in enumerate(turns):
            if not isinstance(turn, StructuredTurn):
                raise TypeError(f"turn {index} must be a StructuredTurn")
        if turns[-1].role != "user":
            raise ValueError("the final turn must be the current user turn")

        first_source_index = max(0, len(turns) - self.policy.max_turns)
        candidate_indices = tuple(range(first_source_index, len(turns)))
        current_index = candidate_indices[-1]
        current_ids = _token_ids(
            tokenizer,
            turns[current_index].content,
            field="current user",
            add_special_tokens=self.policy.add_special_tokens,
        )
        if len(current_ids) > self.policy.max_tokens:
            raise ValueError(
                "current user token count exceeds the frozen budget; refusing silent truncation"
            )

        retained_reversed: list[RenderedTurn] = [
            RenderedTurn(
                role="user",
                role_id=ROUTER_V8_ROLE_TO_ID["user"],
                recency_id=0,
                source_index=current_index,
                input_ids=current_ids,
                original_token_count=len(current_ids),
                truncated_left=False,
            )
        ]
        remaining = self.policy.max_tokens - len(current_ids)
        for source_index in reversed(candidate_indices[:-1]):
            if remaining == 0:
                break
            turn = turns[source_index]
            input_ids = _token_ids(
                tokenizer,
                turn.content,
                field=f"turn {source_index}",
                add_special_tokens=self.policy.add_special_tokens,
            )
            original_count = len(input_ids)
            truncated = original_count > remaining
            if truncated:
                input_ids = input_ids[-remaining:]
            retained_reversed.append(
                RenderedTurn(
                    role=turn.role,
                    role_id=ROUTER_V8_ROLE_TO_ID[turn.role],
                    recency_id=current_index - source_index,
                    source_index=source_index,
                    input_ids=input_ids,
                    original_token_count=original_count,
                    truncated_left=truncated,
                )
            )
            remaining -= len(input_ids)
            if truncated:
                break

        return RenderedConversation(
            turns=tuple(reversed(retained_reversed)),
            source_turn_count=len(turns),
            rendering_policy_fingerprint=self.policy.fingerprint,
        )


def arm_a_metadata() -> dict[str, object]:
    """Frozen current-user masked-mean control, matching the v7 representation."""

    return {
        "arm_id": "A",
        "name": "current_user_masked_mean_control",
        "source": "frozen_input_embeddings",
        "embedding_tensor": "model.embed_tokens.weight",
        "embedding_width": ROUTER_V8_EMBEDDING_WIDTH,
        "prompt_scope": "user_content_only",
        "max_tokens": ROUTER_V8_CONTROL_MAX_TOKENS,
        "padding": "longest",
        "truncation": "right",
        "add_special_tokens": True,
        "pooling": "masked_mean",
        "head": {
            "layers": ["LayerNorm(2048)", "Linear(2048,2)"],
            "route_axis": list(ROUTER_V8_ROUTE_AXIS),
            "score": "softmax_taskmaster_probability",
        },
    }


def arm_b_metadata() -> dict[str, object]:
    """Frozen interface and architecture metadata for the experimental arm."""

    return {
        "arm_id": "B",
        "name": "turn_aware_admission_bigru",
        "source": "frozen_input_embeddings",
        "embedding_tensor": "model.embed_tokens.weight",
        "embedding_width": ROUTER_V8_EMBEDDING_WIDTH,
        "per_turn_pooling": "masked_mean",
        "rendering": FROZEN_RENDERING_POLICY.to_dict(),
        "rendering_fingerprint": FROZEN_RENDERING_POLICY.fingerprint,
        "role_axis": list(ROUTER_V8_ROLES),
        "architecture": {
            "content_projection": {
                "input_width": ROUTER_V8_EMBEDDING_WIDTH,
                "output_width": 224,
                "bias": True,
            },
            "role_embedding": {"classes": 4, "width": 16},
            "recency_embedding": {"classes": ROUTER_V8_MAX_TURNS, "width": 16},
            "turn_feature_width": 256,
            "sequence_encoder": {
                "type": "GRU",
                "layers": 1,
                "input_width": 256,
                "hidden_width_per_direction": 128,
                "bidirectional": True,
                "bias": True,
            },
            "head": ["LayerNorm(256)", "Dropout(0.1)", "Linear(256,1)"],
            "score": "sigmoid_taskmaster_probability",
        },
        "route_axis": list(ROUTER_V8_ROUTE_AXIS),
        "threshold": ROUTER_V8_THRESHOLD,
    }


def architecture_fingerprint() -> str:
    """Identity of both preregistered comparison arms."""

    return canonical_sha256({"arm_a": arm_a_metadata(), "arm_b": arm_b_metadata()})


def masked_mean_control(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Strict Arm-A pooling primitive over current-user input embeddings."""

    if not isinstance(token_embeddings, Tensor) or not isinstance(attention_mask, Tensor):
        raise TypeError("token_embeddings and attention_mask must be tensors")
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [batch,tokens,2048]")
    if token_embeddings.shape[-1] != ROUTER_V8_EMBEDDING_WIDTH:
        raise ValueError(f"token embedding width must be {ROUTER_V8_EMBEDDING_WIDTH}")
    if not token_embeddings.is_floating_point():
        raise TypeError("token_embeddings must have floating dtype")
    if attention_mask.dtype is not torch.bool:
        raise TypeError("attention_mask must have boolean dtype")
    if tuple(attention_mask.shape) != tuple(token_embeddings.shape[:2]):
        raise ValueError("attention_mask must match batch and token dimensions")
    if not torch.isfinite(token_embeddings).all():
        raise ValueError("token_embeddings must be finite")
    token_counts = attention_mask.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("every example must contain at least one retained token")
    mask = attention_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
    pooled = (token_embeddings * mask).sum(dim=1) / token_counts.to(
        dtype=token_embeddings.dtype
    ).unsqueeze(-1)
    if not torch.isfinite(pooled).all():
        raise RuntimeError("masked mean produced a non-finite feature")
    return pooled


class TurnAwareAdmissionEncoder(nn.Module):
    """Representation-only implementation of the frozen Arm-B architecture.

    Inputs are already-pooled frozen embedding features.  This class never owns
    or calls the base model and cannot route requests by itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self.content_projection = nn.Linear(ROUTER_V8_EMBEDDING_WIDTH, 224, bias=True)
        self.role_embedding = nn.Embedding(len(ROUTER_V8_ROLES), 16)
        self.recency_embedding = nn.Embedding(ROUTER_V8_MAX_TURNS, 16)
        self.sequence_encoder = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(256, 1, bias=True)

    @property
    def architecture_metadata(self) -> Mapping[str, object]:
        return MappingProxyType(arm_b_metadata())

    def forward(
        self,
        turn_embeddings: Tensor,
        role_ids: Tensor,
        recency_ids: Tensor,
        turn_mask: Tensor,
    ) -> Tensor:
        """Return one Taskmaster logit per conversation."""

        self._validate_inputs(turn_embeddings, role_ids, recency_ids, turn_mask)
        projected = self.content_projection(turn_embeddings)
        features = torch.cat(
            (
                projected,
                self.role_embedding(role_ids),
                self.recency_embedding(recency_ids),
            ),
            dim=-1,
        )
        lengths = turn_mask.sum(dim=1).to(device="cpu", dtype=torch.long)
        packed = pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.sequence_encoder(packed)
        summary = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        logits = self.output(self.dropout(self.layer_norm(summary))).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Arm-B encoder produced non-finite logits")
        return logits

    @staticmethod
    def _validate_inputs(
        turn_embeddings: Tensor,
        role_ids: Tensor,
        recency_ids: Tensor,
        turn_mask: Tensor,
    ) -> None:
        if any(
            not isinstance(value, Tensor)
            for value in (turn_embeddings, role_ids, recency_ids, turn_mask)
        ):
            raise TypeError("all Arm-B inputs must be tensors")
        if turn_embeddings.ndim != 3:
            raise ValueError("turn_embeddings must have shape [batch,turns,2048]")
        batch_size, turn_count, width = turn_embeddings.shape
        if batch_size < 1 or not 1 <= turn_count <= ROUTER_V8_MAX_TURNS:
            raise ValueError("Arm-B input must contain 1..12 turns per non-empty batch")
        if width != ROUTER_V8_EMBEDDING_WIDTH:
            raise ValueError(f"turn embedding width must be {ROUTER_V8_EMBEDDING_WIDTH}")
        if not turn_embeddings.is_floating_point():
            raise TypeError("turn_embeddings must have floating dtype")
        if not torch.isfinite(turn_embeddings).all():
            raise ValueError("turn_embeddings must be finite")
        expected_shape = (batch_size, turn_count)
        if tuple(role_ids.shape) != expected_shape or tuple(recency_ids.shape) != expected_shape:
            raise ValueError("role_ids and recency_ids must match the batch and turn dimensions")
        if tuple(turn_mask.shape) != expected_shape:
            raise ValueError("turn_mask must match the batch and turn dimensions")
        if role_ids.dtype is not torch.long or recency_ids.dtype is not torch.long:
            raise TypeError("role_ids and recency_ids must use torch.long")
        if turn_mask.dtype is not torch.bool:
            raise TypeError("turn_mask must use torch.bool")
        if role_ids.device != turn_embeddings.device:
            raise ValueError("role_ids must be on the same device as turn_embeddings")
        if (
            recency_ids.device != turn_embeddings.device
            or turn_mask.device != turn_embeddings.device
        ):
            raise ValueError("recency_ids and turn_mask must share the embedding device")
        if torch.any((role_ids < 0) | (role_ids >= len(ROUTER_V8_ROLES))):
            raise ValueError("role_ids contain an out-of-contract role")
        if torch.any((recency_ids < 0) | (recency_ids >= ROUTER_V8_MAX_TURNS)):
            raise ValueError("recency_ids must be in the frozen 0..11 range")
        for row in range(batch_size):
            mask_row = turn_mask[row]
            length = int(mask_row.sum().item())
            if length < 1:
                raise ValueError("every conversation must retain at least one turn")
            expected_mask = torch.arange(turn_count, device=turn_mask.device) < length
            if not torch.equal(mask_row, expected_mask):
                raise ValueError("turn_mask must be contiguous and right padded")
            expected_recency = torch.arange(
                length - 1,
                -1,
                -1,
                device=recency_ids.device,
                dtype=torch.long,
            )
            if not torch.equal(recency_ids[row, :length], expected_recency):
                raise ValueError("active recency_ids must decrease chronologically to zero")
            if int(role_ids[row, length - 1].item()) != ROUTER_V8_ROLE_TO_ID["user"]:
                raise ValueError("the newest active turn must be the current user")


DecisionT = TypeVar("DecisionT")


def select_taskmaster_or_exact_v4(
    taskmaster_probability: float,
    *,
    build_taskmaster_decision: Callable[[], DecisionT],
    delegate_exact_v4: Callable[[], DecisionT],
) -> DecisionT:
    """Apply the fixed gate and return v4's exact object on a negative decision.

    The callbacks intentionally take no transformed prompt or policy argument.
    Their caller closes over the original v4 inputs, preventing this quarantine
    seam from subtly rewriting v4 behavior.  A positive decision never consults
    v4; a negative decision returns the delegate object without copying it.  This
    representation-only seam cannot authenticate the callback's checkpoint; a
    future artifact-complete policy must hash-pin v4 before constructing it.
    """

    if isinstance(taskmaster_probability, bool) or not isinstance(
        taskmaster_probability, (int, float)
    ):
        raise TypeError("taskmaster_probability must be a real number")
    probability = float(taskmaster_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("taskmaster_probability must be finite and in [0,1]")
    if not callable(build_taskmaster_decision) or not callable(delegate_exact_v4):
        raise TypeError("both decision callbacks must be callable")
    if probability >= ROUTER_V8_THRESHOLD:
        return build_taskmaster_decision()
    return delegate_exact_v4()


def preregistered_preblind_gates() -> dict[str, object]:
    """Frozen gates applied before any future independent blind evaluation."""

    return {
        "all_seeds_must_pass": True,
        "original_validation": {
            "accuracy_min": 0.995,
            "taskmaster_precision_min": 0.99,
            "taskmaster_recall_min": 0.99,
            "false_taskmaster_max": 0,
            "negative_support": 650,
        },
        "turn_aware_development_challenge": {
            "rows": 240,
            "taskmaster_positive_rows": 60,
            "legacy_negative_rows": 180,
            "taskmaster_recall_min_count": 50,
            "false_taskmaster_max": 2,
            "overall_accuracy_min": 0.95,
        },
        "paired_consistency_min": 0.95,
        "arm_b_over_arm_a": {
            "taskmaster_recall_gain_min": 0.30,
            "additional_false_taskmaster_max": 1,
        },
        "exact_v4_negative_decision_identity": 1.0,
        "resource_guards": {
            "checkpoint_size_bytes_max": 8_388_608,
            "cpu_p95_latency_ms_max": 25.0,
            "cpu_p95_latency_ratio_vs_arm_a_max": 2.0,
        },
        "seed_selection": [
            "lowest_fresh_original_validation_cross_entropy",
            "lowest_numeric_seed",
        ],
    }


def preregistered_promotion_guards() -> dict[str, object]:
    """Frozen one-shot guards for a later, independently sealed 84-row probe."""

    return {
        "probe_rows": 84,
        "overall_exact_or_acceptable_min": 0.75,
        "exact_or_acceptable_gain_over_v4_min_count": 8,
        "taskmaster_exact_or_acceptable_min_count": 10,
        "taskmaster_rows": 12,
        "false_taskmaster_max": 1,
        "negative_rows": 72,
        "legacy_exact_drop_max_count": 1,
        "legacy_acceptable_drop_max_count": 1,
        "mixed_exact_drop_max_count": 1,
        "mixed_acceptable_drop_max_count": 1,
        "evaluation_attempts": 1,
        "post_probe_tuning_or_retry": False,
        "registration_side_effects": False,
    }


def artifact_identity(path: Path, *, repository_root: Path) -> dict[str, object]:
    """Compute a stable identity for one regular in-repository file."""

    root = repository_root.resolve()
    target = path.resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact must be inside repository_root") from exc
    if path.is_symlink() or not target.is_file():
        raise ValueError("artifact must be a regular non-symlink file")
    before = target.stat()
    data = target.read_bytes()
    after = target.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("artifact changed while its identity was computed")
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def validate_frozen_contract(payload: Mapping[str, Any], *, repository_root: Path) -> None:
    """Fail closed on semantic, integrity, or bound-source drift."""

    expected_top_level_keys = {
        "schema_version",
        "status",
        "experiment_id",
        "authorization",
        "route_contract",
        "seeds",
        "arms",
        "architecture_fingerprint",
        "preblind_gates",
        "promotion_guards",
        "forbidden_inputs",
        "artifact_bindings",
        "integrity",
    }
    if set(payload) != expected_top_level_keys:
        raise ValueError("contract must contain exactly the frozen top-level fields")
    if payload.get("schema_version") != ROUTER_V8_SCHEMA:
        raise ValueError("unsupported router-v8 preregistration schema")
    if payload.get("status") != ROUTER_V8_STATUS:
        raise ValueError("router-v8 preregistration is not in its frozen quarantine state")
    if payload.get("experiment_id") != "router-v8-turn-aware-admission":
        raise ValueError("unexpected router-v8 experiment identity")
    authorization = _require_mapping(payload.get("authorization"), "authorization")
    expected_authorization = {
        "training": False,
        "blind_evaluation": False,
        "registration": False,
        "live_backend_integration": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("quarantined scaffold must not authorize execution or registration")
    route_contract = _require_mapping(payload.get("route_contract"), "route_contract")
    expected_route_contract = {
        "route_axis": list(ROUTER_V8_ROUTE_AXIS),
        "taskmaster_threshold": ROUTER_V8_THRESHOLD,
        "comparison": "greater_than_or_equal",
        "negative": "return_exact_hash_pinned_v4_AutoRouteDecision_unchanged",
        "positive": "select_taskmaster-coffee-api-v2-full-dialog-r16_without_v4",
    }
    if dict(route_contract) != expected_route_contract:
        raise ValueError("route contract differs from the frozen v8 semantics")
    if payload.get("seeds") != list(ROUTER_V8_SEEDS):
        raise ValueError("candidate seed set or order differs from the preregistration")
    arms = _require_mapping(payload.get("arms"), "arms")
    if dict(arms) != {"A": arm_a_metadata(), "B": arm_b_metadata()}:
        raise ValueError("representation arms differ from the frozen architecture")
    if payload.get("architecture_fingerprint") != architecture_fingerprint():
        raise ValueError("architecture fingerprint mismatch")
    if payload.get("preblind_gates") != preregistered_preblind_gates():
        raise ValueError("preblind gates differ from the preregistration")
    if payload.get("promotion_guards") != preregistered_promotion_guards():
        raise ValueError("promotion guards differ from the preregistration")
    forbidden_inputs = payload.get("forbidden_inputs")
    expected_forbidden = [
        "taskmaster_test_split",
        "taskmaster_predictions",
        "sealed_blind_probe_contents",
        "per_example_evaluation_outputs",
        "one_shot_evaluation_report",
    ]
    if forbidden_inputs != expected_forbidden:
        raise ValueError("forbidden-input discipline differs from the frozen contract")

    integrity = _require_mapping(payload.get("integrity"), "integrity")
    if set(integrity) != {"canonical_payload_sha256"}:
        raise ValueError("integrity must contain only canonical_payload_sha256")
    unsigned = dict(payload)
    unsigned.pop("integrity", None)
    if integrity["canonical_payload_sha256"] != canonical_sha256(unsigned):
        raise ValueError("contract canonical payload hash mismatch")

    bindings = _require_mapping(payload.get("artifact_bindings"), "artifact_bindings")
    if set(bindings) != {"representation_module", "readme", "focused_tests"}:
        raise ValueError("artifact_bindings must contain exactly the frozen scaffold sources")
    root = repository_root.resolve()
    for name, raw_identity in bindings.items():
        identity = _require_mapping(raw_identity, f"artifact_bindings.{name}")
        if set(identity) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"artifact binding {name} has unexpected fields")
        relative = identity.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError(f"artifact binding {name} has an invalid repository path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"artifact binding {name} escapes the repository")
        expected = artifact_identity(root / relative_path, repository_root=root)
        if dict(identity) != expected:
            raise ValueError(f"artifact binding {name} does not match current bytes")


def load_frozen_contract(path: Path, *, repository_root: Path) -> Mapping[str, Any]:
    """Load a stable JSON snapshot and validate every frozen v8 invariant."""

    contract_path = path.resolve()
    if path.is_symlink() or not contract_path.is_file():
        raise ValueError("contract must be a regular non-symlink file")
    before = contract_path.stat()
    raw = contract_path.read_bytes()
    after = contract_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("contract changed while it was loaded")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("contract must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("contract must contain a JSON object")
    validate_frozen_contract(payload, repository_root=repository_root)
    return _deep_freeze_json(payload)
