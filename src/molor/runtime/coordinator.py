"""Immutable serving-identity, route, and retrieval planning.

This module deliberately stops before token generation.  It binds the identities and
control-plane decisions implemented here while leaving verified artifact loading,
tokenizer-specific prompt rendering, generation settings, seeds, and decoding to future
interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal, Mapping, Protocol

from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256
from molor.manifests import AdapterBankManifest
from molor.memory import (
    MemoryVersionManifest,
    RetrievalHit,
    SQLiteMemoryStore,
    SessionMemoryPin,
)
from molor.runtime.manifests import (
    ModelStackManifest,
    RouterBindingManifest,
    ServingPinManifest,
    memory_bank_manifest,
)


MemoryMode = Literal["auto", "off"]
MemoryRouteState = Literal["off", "unavailable", "abstained", "applied"]
SelectionMode = Literal["learned", "forced"]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_MAX_MESSAGES = 64
_MAX_MESSAGE_CHARACTERS = 12_000
_MAX_TRANSCRIPT_CHARACTERS = 48_000
_MAX_RETRIEVAL_POLICY_TOKENS = 65_536
_MAX_RETRIEVAL_RECORD_BYTES = 1024 * 1024


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a valid bounded identifier")
    return value


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One bounded user or assistant message supplied to a generation request."""

    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string")
        if len(self.content) > _MAX_MESSAGE_CHARACTERS:
            raise ValueError(f"content must not exceed {_MAX_MESSAGE_CHARACTERS} characters")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Validated client content plus a server-issued request identity.

    The authenticated memory scope is intentionally absent.  Callers must pass it
    separately to :meth:`RuntimeCoordinator.prepare` after authenticating the request.
    """

    schema_version = "molor.generation-request.v1"

    request_id: str
    conversation_id: str
    messages: tuple[ChatMessage, ...]
    memory_mode: MemoryMode = "auto"
    retrieval_limit: int = 8

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        _identifier(self.conversation_id, "conversation_id")
        messages = tuple(self.messages)
        if not 1 <= len(messages) <= _MAX_MESSAGES:
            raise ValueError(f"messages must contain between 1 and {_MAX_MESSAGES} items")
        if any(not isinstance(message, ChatMessage) for message in messages):
            raise TypeError("messages must contain only ChatMessage values")
        if messages[-1].role != "user":
            raise ValueError("the final message must be a user message")
        if sum(len(message.content) for message in messages) > _MAX_TRANSCRIPT_CHARACTERS:
            raise ValueError(
                f"messages must not exceed {_MAX_TRANSCRIPT_CHARACTERS} total characters"
            )
        if self.memory_mode not in ("auto", "off"):
            raise ValueError("memory_mode must be 'auto' or 'off'")
        if (
            not isinstance(self.retrieval_limit, int)
            or isinstance(self.retrieval_limit, bool)
            or not 1 <= self.retrieval_limit <= 64
        ):
            raise ValueError("retrieval_limit must be between 1 and 64")
        object.__setattr__(self, "messages", messages)

    @property
    def prompt(self) -> str:
        return self.messages[-1].content

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "messages": [message.to_dict() for message in self.messages],
            "memory_mode": self.memory_mode,
            "retrieval_limit": self.retrieval_limit,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


class TokenCounter(Protocol):
    """Tokenizer-specific counter bound to an immutable tokenizer identity."""

    @property
    def fingerprint(self) -> str: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class RenderedRetrievalRecord:
    """One source-labelled, tokenizer-counted memory record safe to hand off."""

    source_scope_id: str
    source_event_id: str
    source_event_fingerprint: str
    score: float
    text: str
    token_count: int
    rendered_bytes: int

    def __post_init__(self) -> None:
        _identifier(self.source_scope_id, "source_scope_id")
        _identifier(self.source_event_id, "source_event_id")
        object.__setattr__(
            self,
            "source_event_fingerprint",
            validate_sha256(
                self.source_event_fingerprint,
                "source_event_fingerprint",
            ),
        )
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("score must be a finite number in [0, 1]")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("text must be a non-empty string")
        encoded_size = len(self.text.encode("utf-8"))
        if (
            not isinstance(self.rendered_bytes, int)
            or isinstance(self.rendered_bytes, bool)
            or self.rendered_bytes != encoded_size
            or not 1 <= self.rendered_bytes <= _MAX_RETRIEVAL_RECORD_BYTES
        ):
            raise ValueError("rendered_bytes must be the bounded UTF-8 text size")
        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or not 1 <= self.token_count <= _MAX_RETRIEVAL_POLICY_TOKENS
        ):
            raise ValueError("token_count must be a bounded positive integer")
        object.__setattr__(self, "score", float(self.score))

    @property
    def text_fingerprint(self) -> str:
        return fingerprint_json({"text": self.text})

    def to_identity_dict(self) -> dict[str, Any]:
        """Return trace identity without copying retrieved content into logs."""

        return {
            "source_event_id": self.source_event_id,
            "source_event_fingerprint": self.source_event_fingerprint,
            "score": self.score,
            "text_fingerprint": self.text_fingerprint,
            "token_count": self.token_count,
            "rendered_bytes": self.rendered_bytes,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_identity_dict())


class RetrievalRenderer(Protocol):
    """Render retrieved records under an exact tokenizer-measured budget."""

    @property
    def fingerprint(self) -> str: ...

    @property
    def tokenizer_fingerprint(self) -> str: ...

    def render(
        self,
        hits: tuple[RetrievalHit, ...],
    ) -> tuple[RenderedRetrievalRecord, ...]: ...


@dataclass(frozen=True, slots=True)
class BoundedRetrievalRenderer:
    """Whole-record renderer with explicit trust labels and exact token counting.

    Records that exceed the per-record byte cap or the remaining token budget are
    omitted rather than character-truncated.  The injected counter must use the same
    immutable tokenizer that the eventual model executor will use.
    """

    schema_version = "molor.bounded-retrieval-renderer.v1"

    token_counter: TokenCounter
    max_tokens: int = 2048
    max_record_bytes: int = 32 * 1024
    _counter_fingerprint: str = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        counter_fingerprint = validate_sha256(
            self.token_counter.fingerprint,
            "token_counter.fingerprint",
        )
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or not 1 <= self.max_tokens <= _MAX_RETRIEVAL_POLICY_TOKENS
        ):
            raise ValueError("max_tokens must be a bounded positive integer")
        if (
            not isinstance(self.max_record_bytes, int)
            or isinstance(self.max_record_bytes, bool)
            or not 1 <= self.max_record_bytes <= _MAX_RETRIEVAL_RECORD_BYTES
        ):
            raise ValueError("max_record_bytes must be a bounded positive integer")
        object.__setattr__(self, "_counter_fingerprint", counter_fingerprint)
        object.__setattr__(
            self,
            "_fingerprint",
            fingerprint_json(
                {
                    "schema_version": self.schema_version,
                    "token_counter_fingerprint": counter_fingerprint,
                    "max_tokens": self.max_tokens,
                    "max_record_bytes": self.max_record_bytes,
                    "overflow_behavior": "omit-whole-record",
                    "trust_label": "untrusted-retrieved-memory",
                }
            ),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def tokenizer_fingerprint(self) -> str:
        """Return the immutable tokenizer identity used for record accounting."""

        return self._counter_fingerprint

    def render(
        self,
        hits: tuple[RetrievalHit, ...],
    ) -> tuple[RenderedRetrievalRecord, ...]:
        if (
            validate_sha256(
                self.token_counter.fingerprint,
                "token_counter.fingerprint",
            )
            != self._counter_fingerprint
        ):
            raise ValueError("token counter identity changed after renderer construction")
        hits = tuple(hits)
        if any(not isinstance(hit, RetrievalHit) for hit in hits):
            raise TypeError("hits must contain only RetrievalHit values")

        remaining_tokens = self.max_tokens
        rendered: list[RenderedRetrievalRecord] = []
        for hit in hits:
            source = hit.event
            text_bytes = canonical_json_bytes(
                {
                    "schema_version": "molor.retrieved-memory-record.v1",
                    "trust": "untrusted-retrieved-memory",
                    "source_event_id": source.event_id,
                    "source_event_fingerprint": source.fingerprint,
                    "source_kind": source.kind.value,
                    "payload": source.to_dict()["payload"],
                }
            )
            if len(text_bytes) > self.max_record_bytes:
                continue
            text = text_bytes.decode("utf-8")
            token_count = self.token_counter.count_tokens(text)
            if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 1:
                raise ValueError("token counter must return a positive integer")
            if token_count > remaining_tokens:
                continue
            rendered.append(
                RenderedRetrievalRecord(
                    source_scope_id=source.scope_id,
                    source_event_id=source.event_id,
                    source_event_fingerprint=source.fingerprint,
                    score=hit.score,
                    text=text,
                    token_count=token_count,
                    rendered_bytes=len(text_bytes),
                )
            )
            remaining_tokens -= token_count
        return tuple(rendered)


@dataclass(frozen=True, slots=True)
class CapabilityRouteDecision:
    """An immutable, bank-ordered capability route selected for one request."""

    selector_fingerprint: str
    selection_mode: SelectionMode
    router_binding_fingerprint: str
    bank_fingerprint: str
    route_ids: tuple[str, ...]
    weights: tuple[float, ...]
    selected_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selector_fingerprint",
            validate_sha256(self.selector_fingerprint, "selector_fingerprint"),
        )
        if self.selection_mode not in ("learned", "forced"):
            raise ValueError("selection_mode must be 'learned' or 'forced'")
        object.__setattr__(
            self,
            "router_binding_fingerprint",
            validate_sha256(
                self.router_binding_fingerprint,
                "router_binding_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "bank_fingerprint",
            validate_sha256(self.bank_fingerprint, "bank_fingerprint"),
        )
        route_ids = tuple(self.route_ids)
        weights = tuple(self.weights)
        selected = tuple(self.selected_indices)
        if not route_ids or len(route_ids) != len(weights):
            raise ValueError("route_ids and weights must describe the same non-empty axis")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_ids must not contain duplicates")
        if any(not isinstance(route_id, str) or not route_id for route_id in route_ids):
            raise ValueError("route_ids must contain non-empty strings")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 1
            for weight in weights
        ):
            raise ValueError("weights must be finite non-negative numbers")
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("selected_indices must be non-empty and unique")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(route_ids)
            for index in selected
        ):
            raise ValueError("selected_indices contains an out-of-range index")
        positive = tuple(index for index, weight in enumerate(weights) if weight > 0)
        if positive != tuple(sorted(selected)):
            raise ValueError("selected_indices must exactly identify positive route weights")
        if self.selection_mode == "forced" and (
            len(selected) != 1 or float(weights[selected[0]]) != 1.0
        ):
            raise ValueError("forced routes must be exact one-hot decisions")
        object.__setattr__(self, "route_ids", route_ids)
        object.__setattr__(self, "weights", tuple(float(weight) for weight in weights))
        object.__setattr__(self, "selected_indices", tuple(sorted(selected)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_fingerprint": self.selector_fingerprint,
            "selection_mode": self.selection_mode,
            "router_binding_fingerprint": self.router_binding_fingerprint,
            "bank_fingerprint": self.bank_fingerprint,
            "route_ids": list(self.route_ids),
            "weights": list(self.weights),
            "selected_indices": list(self.selected_indices),
        }


@dataclass(frozen=True, slots=True)
class MemoryRouteDecision:
    """A hard apply/null choice on the independent memory-adapter axis."""

    state: MemoryRouteState
    gate_policy_fingerprint: str
    attempted_version_fingerprint: str | None
    version_fingerprint: str | None
    memory_bank_fingerprint: str | None
    weights: tuple[float, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.state not in ("off", "unavailable", "abstained", "applied"):
            raise ValueError("unsupported memory route state")
        object.__setattr__(
            self,
            "gate_policy_fingerprint",
            validate_sha256(
                self.gate_policy_fingerprint,
                "gate_policy_fingerprint",
            ),
        )
        if not isinstance(self.reason, str) or not self.reason or len(self.reason) > 256:
            raise ValueError("reason must be a bounded non-empty string")

        if self.attempted_version_fingerprint is not None:
            object.__setattr__(
                self,
                "attempted_version_fingerprint",
                validate_sha256(
                    self.attempted_version_fingerprint,
                    "attempted_version_fingerprint",
                ),
            )

        has_version = self.version_fingerprint is not None
        has_bank = self.memory_bank_fingerprint is not None
        if has_version != has_bank:
            raise ValueError("memory version and bank fingerprints must be supplied together")
        if has_version:
            object.__setattr__(
                self,
                "version_fingerprint",
                validate_sha256(self.version_fingerprint, "version_fingerprint"),
            )
            object.__setattr__(
                self,
                "memory_bank_fingerprint",
                validate_sha256(
                    self.memory_bank_fingerprint,
                    "memory_bank_fingerprint",
                ),
            )

        raw_weights = tuple(self.weights)
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            for weight in raw_weights
        ):
            raise ValueError("memory weights must be finite real numbers")
        weights = tuple(float(weight) for weight in raw_weights)
        if self.state == "off":
            if self.attempted_version_fingerprint is not None:
                raise ValueError("an off route cannot attempt a memory version")
            if has_version or weights:
                raise ValueError("an off memory route cannot bind an adapter")
        elif self.state == "unavailable":
            if has_version or weights:
                raise ValueError("an unavailable memory route cannot bind an adapter")
        else:
            if not has_version:
                raise ValueError("applied and abstained memory routes require a bound version")
            if self.attempted_version_fingerprint != self.version_fingerprint:
                raise ValueError("effective and attempted memory versions must match")
            expected = (1.0, 0.0) if self.state == "applied" else (0.0, 1.0)
            if weights != expected:
                raise ValueError("the v0 memory gate must make an exact hard apply/null choice")
        object.__setattr__(self, "weights", weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "gate_policy_fingerprint": self.gate_policy_fingerprint,
            "attempted_version_fingerprint": self.attempted_version_fingerprint,
            "version_fingerprint": self.version_fingerprint,
            "memory_bank_fingerprint": self.memory_bank_fingerprint,
            "weights": list(self.weights),
            "reason": self.reason,
        }


class CapabilityRouteSelector(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def select(self, request: GenerationRequest) -> CapabilityRouteDecision: ...


class MemoryGate(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def select(
        self,
        request: GenerationRequest,
        *,
        version: MemoryVersionManifest | None,
        bank: AdapterBankManifest | None,
        attempted_version_fingerprint: str | None,
    ) -> MemoryRouteDecision: ...


class MemoryVersionResolver(Protocol):
    def resolve(self, fingerprint: str) -> MemoryVersionManifest: ...


@dataclass(frozen=True, slots=True)
class PreparedGeneration:
    """Immutable serving identities and decisions prepared before execution."""

    request: GenerationRequest
    router_binding: RouterBindingManifest
    serving_pin: ServingPinManifest
    capability_route: CapabilityRouteDecision
    memory_route: MemoryRouteDecision
    retrieval_pin: SessionMemoryPin | None
    retrieval_records: tuple[RenderedRetrievalRecord, ...]
    retrieval_policy_fingerprint: str
    retrieval_tokenizer_fingerprint: str
    memory_bank: AdapterBankManifest | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        if not isinstance(self.router_binding, RouterBindingManifest):
            raise TypeError("router_binding must be a RouterBindingManifest")
        if not isinstance(self.serving_pin, ServingPinManifest):
            raise TypeError("serving_pin must be a ServingPinManifest")
        if not isinstance(self.capability_route, CapabilityRouteDecision):
            raise TypeError("capability_route must be a CapabilityRouteDecision")
        if not isinstance(self.memory_route, MemoryRouteDecision):
            raise TypeError("memory_route must be a MemoryRouteDecision")
        if self.retrieval_pin is not None and not isinstance(
            self.retrieval_pin,
            SessionMemoryPin,
        ):
            raise TypeError("retrieval_pin must be a SessionMemoryPin or None")
        records = tuple(self.retrieval_records)
        if any(not isinstance(record, RenderedRetrievalRecord) for record in records):
            raise TypeError("retrieval_records must contain only RenderedRetrievalRecord values")
        object.__setattr__(
            self,
            "retrieval_policy_fingerprint",
            validate_sha256(
                self.retrieval_policy_fingerprint,
                "retrieval_policy_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "retrieval_tokenizer_fingerprint",
            validate_sha256(
                self.retrieval_tokenizer_fingerprint,
                "retrieval_tokenizer_fingerprint",
            ),
        )
        capability_bank = self.serving_pin.model_stack.bank
        self.router_binding.assert_matches(capability_bank)
        if self.capability_route.router_binding_fingerprint != self.router_binding.fingerprint:
            raise ValueError("capability decision uses a different router binding")
        if self.capability_route.bank_fingerprint != capability_bank.fingerprint:
            raise ValueError("capability decision uses a different adapter bank")
        if self.capability_route.route_ids != capability_bank.route_ids:
            raise ValueError("capability decision route order differs from the adapter bank")

        version = self.serving_pin.model_stack.memory_version
        serving_memory_pin = self.serving_pin.memory_pin
        retrieval_pin = self.retrieval_pin
        if self.request.memory_mode == "off":
            if (
                self.memory_route.state != "off"
                or retrieval_pin is not None
                or serving_memory_pin is not None
                or records
            ):
                raise ValueError("memory-off requests cannot carry a pin or retrieval records")
        else:
            if self.memory_route.state == "off" or retrieval_pin is None:
                raise ValueError("automatic memory requests require a pinned retrieval view")
            if retrieval_pin.session_id != self.request.request_id:
                raise ValueError("memory retrieval pin must be scoped to this request ID")
            if any(record.source_scope_id != retrieval_pin.scope_id for record in records):
                raise ValueError("retrieval records contain an event from another memory scope")
            if serving_memory_pin is not None and serving_memory_pin != retrieval_pin:
                raise ValueError("serving and retrieval pins describe different views")

        if version is None:
            if self.memory_bank is not None:
                raise ValueError("a memory bank cannot be bound without a memory version")
            if self.memory_route.version_fingerprint is not None:
                raise ValueError("memory route references a version absent from the model stack")
            if retrieval_pin is not None:
                attempted = retrieval_pin.version_fingerprint
                if self.memory_route.attempted_version_fingerprint != attempted:
                    raise ValueError("memory route attempted a version outside the retrieval pin")
                if attempted is not None and serving_memory_pin is not None:
                    raise ValueError(
                        "an unverified version cannot remain in the effective serving pin"
                    )
        else:
            if retrieval_pin is None or serving_memory_pin != retrieval_pin:
                raise ValueError("a memory-backed stack requires its exact retrieval pin")
            if self.memory_bank is None:
                raise ValueError("a memory-backed model stack requires a memory bank")
            if self.memory_bank != memory_bank_manifest(version):
                raise ValueError("memory bank does not contain the pinned memory adapter")
            if self.memory_bank.fingerprint != self.memory_route.memory_bank_fingerprint:
                raise ValueError("memory route uses a different memory bank")
            if version.fingerprint != self.memory_route.version_fingerprint:
                raise ValueError("memory route uses a different memory version")
            if retrieval_pin.version_fingerprint != version.fingerprint:
                raise ValueError("retrieval pin references a different memory version")
        object.__setattr__(self, "retrieval_records", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "molor.prepared-generation.v1",
            "request_fingerprint": self.request.fingerprint,
            "request_id": self.request.request_id,
            "conversation_id": self.request.conversation_id,
            "router_binding_fingerprint": self.router_binding.fingerprint,
            "serving_pin_fingerprint": self.serving_pin.fingerprint,
            "capability_route": self.capability_route.to_dict(),
            "memory_route": self.memory_route.to_dict(),
            "retrieval_pin_fingerprint": (
                None if self.retrieval_pin is None else self.retrieval_pin.fingerprint
            ),
            "retrieval_policy_fingerprint": self.retrieval_policy_fingerprint,
            "retrieval_tokenizer_fingerprint": self.retrieval_tokenizer_fingerprint,
            "retrieval_records": [record.to_identity_dict() for record in self.retrieval_records],
            "memory_bank_fingerprint": (
                None if self.memory_bank is None else self.memory_bank.fingerprint
            ),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


class NullCapabilityRouteSelector:
    """Explicit frozen-base control selector used before router weights are loaded."""

    def __init__(
        self,
        bank: AdapterBankManifest,
        router_binding: RouterBindingManifest,
    ) -> None:
        if not isinstance(bank, AdapterBankManifest):
            raise TypeError("bank must be an AdapterBankManifest")
        router_binding.assert_matches(bank)
        null_indices = tuple(route.route_index for route in bank.routes if route.is_null)
        if len(null_indices) != 1:
            raise ValueError("the null selector requires exactly one capability null route")
        self._bank = bank
        self._binding = router_binding
        self._null_index = null_indices[0]
        self._fingerprint = fingerprint_json(
            {
                "schema_version": "molor.null-capability-selector.v1",
                "router_binding_fingerprint": router_binding.fingerprint,
                "null_route_id": bank.route_by_index(self._null_index).route_id,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def select(self, request: GenerationRequest) -> CapabilityRouteDecision:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        weights = [0.0] * len(self._bank.routes)
        weights[self._null_index] = 1.0
        return CapabilityRouteDecision(
            selector_fingerprint=self.fingerprint,
            selection_mode="forced",
            router_binding_fingerprint=self._binding.fingerprint,
            bank_fingerprint=self._bank.fingerprint,
            route_ids=self._bank.route_ids,
            weights=tuple(weights),
            selected_indices=(self._null_index,),
        )


class AbstainingMemoryGate:
    """Fail-closed memory control until a verified learned gate is installed."""

    def __init__(self) -> None:
        self._fingerprint = fingerprint_json({"schema_version": "molor.abstaining-memory-gate.v1"})

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def select(
        self,
        request: GenerationRequest,
        *,
        version: MemoryVersionManifest | None,
        bank: AdapterBankManifest | None,
        attempted_version_fingerprint: str | None,
    ) -> MemoryRouteDecision:
        if request.memory_mode == "off":
            return MemoryRouteDecision(
                state="off",
                gate_policy_fingerprint=self.fingerprint,
                attempted_version_fingerprint=None,
                version_fingerprint=None,
                memory_bank_fingerprint=None,
                weights=(),
                reason="memory-disabled-by-request",
            )
        if version is None or bank is None:
            return MemoryRouteDecision(
                state="unavailable",
                gate_policy_fingerprint=self.fingerprint,
                attempted_version_fingerprint=attempted_version_fingerprint,
                version_fingerprint=None,
                memory_bank_fingerprint=None,
                weights=(),
                reason="no-verified-memory-version",
            )
        return MemoryRouteDecision(
            state="abstained",
            gate_policy_fingerprint=self.fingerprint,
            attempted_version_fingerprint=version.fingerprint,
            version_fingerprint=version.fingerprint,
            memory_bank_fingerprint=bank.fingerprint,
            weights=(0.0, 1.0),
            reason="control-gate-is-fail-closed",
        )


class StaticMemoryVersionResolver:
    """Small typed registry useful for tests and an initial artifact-loader boundary."""

    def __init__(
        self,
        versions: Mapping[str, MemoryVersionManifest] | None = None,
    ) -> None:
        resolved = {} if versions is None else dict(versions)
        if any(
            not isinstance(version, MemoryVersionManifest) or fingerprint != version.fingerprint
            for fingerprint, version in resolved.items()
        ):
            raise ValueError("versions must be keyed by their exact manifest fingerprint")
        self._versions = resolved

    def resolve(self, fingerprint: str) -> MemoryVersionManifest:
        normalized = validate_sha256(fingerprint, "fingerprint")
        try:
            return self._versions[normalized]
        except KeyError as exc:
            raise KeyError(f"unresolved memory version: {normalized}") from exc


class RuntimeCoordinator:
    """Prepare a request against one fixed capability bank and memory control plane."""

    def __init__(
        self,
        *,
        capability_bank: AdapterBankManifest,
        router_binding: RouterBindingManifest,
        route_selector: CapabilityRouteSelector,
        memory_store: SQLiteMemoryStore,
        version_resolver: MemoryVersionResolver,
        memory_gate: MemoryGate,
        retrieval_renderer: RetrievalRenderer,
    ) -> None:
        if not isinstance(capability_bank, AdapterBankManifest):
            raise TypeError("capability_bank must be an AdapterBankManifest")
        if not isinstance(router_binding, RouterBindingManifest):
            raise TypeError("router_binding must be a RouterBindingManifest")
        if not isinstance(memory_store, SQLiteMemoryStore):
            raise TypeError("memory_store must be a SQLiteMemoryStore")
        if type(memory_gate) is not AbstainingMemoryGate:
            raise TypeError(
                "memory v0 permits only AbstainingMemoryGate until artifact and "
                "invalidation guards are implemented"
            )
        if type(retrieval_renderer) is not BoundedRetrievalRenderer:
            raise TypeError(
                "runtime v0 permits only BoundedRetrievalRenderer for retrieved content"
            )
        router_binding.assert_matches(capability_bank)
        self.capability_bank = capability_bank
        self.router_binding = router_binding
        self.route_selector = route_selector
        self.memory_store = memory_store
        self.version_resolver = version_resolver
        self.memory_gate = memory_gate
        self.retrieval_renderer = retrieval_renderer
        self._retrieval_policy_fingerprint = validate_sha256(
            retrieval_renderer.fingerprint,
            "retrieval_renderer.fingerprint",
        )
        self._retrieval_tokenizer_fingerprint = validate_sha256(
            retrieval_renderer.tokenizer_fingerprint,
            "retrieval_renderer.tokenizer_fingerprint",
        )

    def prepare(
        self,
        *,
        trusted_scope_id: str,
        request: GenerationRequest,
    ) -> PreparedGeneration:
        """Pin, retrieve, route, and return an immutable execution plan.

        ``trusted_scope_id`` must come from server-side authentication or a fixed
        single-user configuration.  It is never copied from browser request data.
        """

        _identifier(trusted_scope_id, "trusted_scope_id")
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")

        capability_route = self.route_selector.select(request)
        self._validate_capability_route(capability_route)

        if request.memory_mode == "off":
            stack = ModelStackManifest(self.capability_bank)
            memory_route = self.memory_gate.select(
                request,
                version=None,
                bank=None,
                attempted_version_fingerprint=None,
            )
            if memory_route.state != "off":
                raise ValueError("memory gate must return the off route when memory is disabled")
            self._validate_memory_route(memory_route)
            return PreparedGeneration(
                request=request,
                router_binding=self.router_binding,
                serving_pin=ServingPinManifest(stack),
                capability_route=capability_route,
                memory_route=memory_route,
                retrieval_pin=None,
                retrieval_records=(),
                retrieval_policy_fingerprint=self._retrieval_policy_fingerprint,
                retrieval_tokenizer_fingerprint=self._retrieval_tokenizer_fingerprint,
                memory_bank=None,
            )

        # A serving pin is request-scoped.  conversation_id remains correlation and
        # durable-event ordering data; reusing it here would freeze retrieval forever.
        pin = self.memory_store.pin_session(trusted_scope_id, request.request_id)
        try:
            hits = self.memory_store.search_active_events(
                trusted_scope_id,
                request.prompt,
                limit=request.retrieval_limit,
                pin=pin,
            )
        except ValueError as exc:
            if "searchable token" not in str(exc):
                raise
            hits = ()
        retrieval_records = self.retrieval_renderer.render(hits)

        version = None
        memory_bank = None
        stack = ModelStackManifest(self.capability_bank)
        serving_pin = ServingPinManifest(
            stack,
            pin if pin.version_fingerprint is None else None,
        )
        attempted_version_fingerprint = pin.version_fingerprint
        if pin.version_fingerprint is not None:
            try:
                candidate = self.version_resolver.resolve(pin.version_fingerprint)
                if not isinstance(candidate, MemoryVersionManifest):
                    raise ValueError("resolver returned an invalid memory version")
                if candidate.fingerprint != pin.version_fingerprint:
                    raise ValueError(
                        "resolved memory version does not match the pinned fingerprint"
                    )
                if candidate.scope_id != trusted_scope_id:
                    raise ValueError("resolved memory version belongs to a different scope")
                if candidate.base_fingerprint != self.capability_bank.base.fingerprint:
                    raise ValueError("resolved memory version targets a different frozen base")
                candidate_bank = memory_bank_manifest(candidate)
                candidate_stack = ModelStackManifest(self.capability_bank, candidate)
                candidate_serving_pin = ServingPinManifest(candidate_stack, pin)
            except (LookupError, OSError, RuntimeError, ValueError):
                # Retrieval remains useful at the immutable cut.  The adapter plane
                # fails closed to the base stack and retains the attempted identity.
                pass
            else:
                version = candidate
                memory_bank = candidate_bank
                stack = candidate_stack
                serving_pin = candidate_serving_pin
        memory_route = self.memory_gate.select(
            request,
            version=version,
            bank=memory_bank,
            attempted_version_fingerprint=attempted_version_fingerprint,
        )
        if version is None and memory_route.state != "unavailable":
            raise ValueError("memory gate must fail closed when no version is available")
        self._validate_memory_route(memory_route)

        return PreparedGeneration(
            request=request,
            router_binding=self.router_binding,
            serving_pin=serving_pin,
            capability_route=capability_route,
            memory_route=memory_route,
            retrieval_pin=pin,
            retrieval_records=retrieval_records,
            retrieval_policy_fingerprint=self._retrieval_policy_fingerprint,
            retrieval_tokenizer_fingerprint=self._retrieval_tokenizer_fingerprint,
            memory_bank=memory_bank,
        )

    def _validate_capability_route(self, decision: CapabilityRouteDecision) -> None:
        if not isinstance(decision, CapabilityRouteDecision):
            raise TypeError("route selector must return a CapabilityRouteDecision")
        selector_fingerprint = validate_sha256(
            self.route_selector.fingerprint,
            "route_selector.fingerprint",
        )
        if decision.selector_fingerprint != selector_fingerprint:
            raise ValueError("route decision uses a different selector policy")
        if decision.router_binding_fingerprint != self.router_binding.fingerprint:
            raise ValueError("route selector uses a different router binding")
        if decision.bank_fingerprint != self.capability_bank.fingerprint:
            raise ValueError("route selector uses a different capability bank")
        if decision.route_ids != self.capability_bank.route_ids:
            raise ValueError("route selector changed the capability route order")
        if decision.selection_mode == "learned":
            if len(decision.selected_indices) != self.router_binding.router_config.top_k:
                raise ValueError("learned route cardinality differs from the router config")
            config = self.router_binding.router_config
            if config.normalize_top_k or config.uniform_top_k or config.straight_through:
                if not math.isclose(sum(decision.weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("normalized learned route weights must sum to one")
            if config.uniform_top_k:
                expected_weight = 1.0 / config.top_k
                if any(
                    not math.isclose(
                        decision.weights[index],
                        expected_weight,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                    for index in decision.selected_indices
                ):
                    raise ValueError("uniform learned routes must use equal selected weights")
        elif len(decision.selected_indices) != 1:
            raise ValueError("forced control routes must select exactly one route")

    def _validate_memory_route(self, decision: MemoryRouteDecision) -> None:
        if not isinstance(decision, MemoryRouteDecision):
            raise TypeError("memory gate must return a MemoryRouteDecision")
        gate_fingerprint = validate_sha256(
            self.memory_gate.fingerprint,
            "memory_gate.fingerprint",
        )
        if decision.gate_policy_fingerprint != gate_fingerprint:
            raise ValueError("memory decision uses a different gate policy")
