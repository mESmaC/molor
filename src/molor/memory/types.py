"""Immutable identities for cross-session memory and consolidation jobs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, ClassVar, Iterable

from molor.identity import fingerprint_json, freeze_json, thaw_json, validate_sha256
from molor.manifests import AdapterManifest, ArtifactDescriptor


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 1-256 characters using letters, digits, '.', '_', ':', '/', or '-'"
        )
    return value


def _nonnegative_integer(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _fingerprint_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    fingerprints = tuple(values)
    if not fingerprints:
        raise ValueError(f"{field_name} must not be empty")
    normalized = tuple(
        validate_sha256(value, f"{field_name}[{index}]") for index, value in enumerate(fingerprints)
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


class MemoryEventKind(StrEnum):
    """Whether an event adds, corrects, or removes durable memory content."""

    OBSERVATION = "observation"
    CORRECTION = "correction"
    TOMBSTONE = "tombstone"


@dataclass(frozen=True, slots=True)
class ContextRolloverPolicy:
    """Token high-water mark that leaves bounded generation headroom before rollover."""

    schema_version = "molor.context-rollover-policy.v1"

    max_context_tokens: int
    trigger_tokens: int
    minimum_generation_tokens: int

    def __post_init__(self) -> None:
        for field_name in (
            "max_context_tokens",
            "trigger_tokens",
            "minimum_generation_tokens",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.trigger_tokens >= self.max_context_tokens:
            raise ValueError("trigger_tokens must be below the hard context limit")
        if self.trigger_tokens + self.minimum_generation_tokens > self.max_context_tokens:
            raise ValueError("rollover trigger must leave the minimum generation headroom")

    def should_rollover(self, active_context_tokens: int) -> bool:
        _nonnegative_integer(active_context_tokens, "active_context_tokens")
        return active_context_tokens >= self.trigger_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_context_tokens": self.max_context_tokens,
            "trigger_tokens": self.trigger_tokens,
            "minimum_generation_tokens": self.minimum_generation_tokens,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """One immutable, session-ordered input to retrieval and consolidation."""

    schema_version: ClassVar[str] = "molor.memory-event.v1"

    event_id: str
    scope_id: str
    session_id: str
    sequence: int
    kind: MemoryEventKind
    payload: Any
    target_event_id: str | None = None
    target_event_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.scope_id, "scope_id")
        _identifier(self.session_id, "session_id")
        _nonnegative_integer(self.sequence, "sequence")
        if not isinstance(self.kind, MemoryEventKind):
            raise TypeError("kind must be a MemoryEventKind")

        has_target_id = self.target_event_id is not None
        has_target_fingerprint = self.target_event_fingerprint is not None
        if has_target_id != has_target_fingerprint:
            raise ValueError("target event ID and fingerprint must be supplied together")
        if self.kind is MemoryEventKind.OBSERVATION:
            if has_target_id:
                raise ValueError("observation events must not target another event")
            if self.payload is None:
                raise ValueError("observation events require a payload")
        else:
            if not has_target_id:
                raise ValueError("correction and tombstone events require a target")
            assert self.target_event_id is not None
            assert self.target_event_fingerprint is not None
            _identifier(self.target_event_id, "target_event_id")
            if self.target_event_id == self.event_id:
                raise ValueError("an event must not target itself")
            object.__setattr__(
                self,
                "target_event_fingerprint",
                validate_sha256(
                    self.target_event_fingerprint,
                    "target_event_fingerprint",
                ),
            )
            if self.kind is MemoryEventKind.CORRECTION and self.payload is None:
                raise ValueError("correction events require replacement payload")
            if self.kind is MemoryEventKind.TOMBSTONE and self.payload is not None:
                raise ValueError("tombstone events must not contain payload")

        object.__setattr__(self, "payload", freeze_json(self.payload))

    def assert_valid_target(self, target: MemoryEvent) -> None:
        """Validate a resolved correction/tombstone target without requiring a store."""

        if self.kind is MemoryEventKind.OBSERVATION:
            raise ValueError("observation events do not have targets")
        if not isinstance(target, MemoryEvent):
            raise TypeError("target must be a MemoryEvent")
        if target.scope_id != self.scope_id:
            raise ValueError("target event belongs to a different memory scope")
        if target.kind is MemoryEventKind.TOMBSTONE:
            raise ValueError("corrections and tombstones must target a content-bearing event")
        if (
            target.event_id != self.target_event_id
            or target.fingerprint != self.target_event_fingerprint
        ):
            raise ValueError("resolved target does not match the declared target identity")
        if target.session_id == self.session_id and target.sequence >= self.sequence:
            raise ValueError("an event must target an earlier event in the same session")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "payload": thaw_json(self.payload),
            "target_event_id": self.target_event_id,
            "target_event_fingerprint": self.target_event_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RolloverSnapshotManifest:
    """An ordered, contiguous event range durably summarized before context compaction."""

    schema_version: ClassVar[str] = "molor.rollover-snapshot.v1"

    snapshot_id: str
    scope_id: str
    session_id: str
    first_event_sequence: int
    last_event_sequence: int
    event_fingerprints: tuple[str, ...]
    summary: ArtifactDescriptor
    rollover_policy_fingerprint: str

    def __post_init__(self) -> None:
        _identifier(self.snapshot_id, "snapshot_id")
        _identifier(self.scope_id, "scope_id")
        _identifier(self.session_id, "session_id")
        first = _nonnegative_integer(self.first_event_sequence, "first_event_sequence")
        last = _nonnegative_integer(self.last_event_sequence, "last_event_sequence")
        if last < first:
            raise ValueError("last_event_sequence must not precede first_event_sequence")
        fingerprints = _fingerprint_tuple(self.event_fingerprints, "event_fingerprints")
        if len(fingerprints) != last - first + 1:
            raise ValueError("event fingerprints must cover one contiguous sequence range")
        if not isinstance(self.summary, ArtifactDescriptor):
            raise TypeError("summary must be an ArtifactDescriptor")
        object.__setattr__(self, "event_fingerprints", fingerprints)
        object.__setattr__(
            self,
            "rollover_policy_fingerprint",
            validate_sha256(self.rollover_policy_fingerprint, "rollover_policy_fingerprint"),
        )

    @classmethod
    def from_events(
        cls,
        snapshot_id: str,
        events: Iterable[MemoryEvent],
        *,
        summary: ArtifactDescriptor,
        rollover_policy_fingerprint: str,
    ) -> RolloverSnapshotManifest:
        event_tuple = tuple(events)
        if not event_tuple:
            raise ValueError("events must not be empty")
        if any(not isinstance(event, MemoryEvent) for event in event_tuple):
            raise TypeError("events must contain only MemoryEvent values")
        first = event_tuple[0]
        if any(event.scope_id != first.scope_id for event in event_tuple):
            raise ValueError("rollover events must belong to one memory scope")
        if any(event.session_id != first.session_id for event in event_tuple):
            raise ValueError("rollover events must belong to one session")
        expected_sequences = tuple(range(first.sequence, first.sequence + len(event_tuple)))
        actual_sequences = tuple(event.sequence for event in event_tuple)
        if actual_sequences != expected_sequences:
            raise ValueError("rollover events must be ordered and contiguous")
        return cls(
            snapshot_id=snapshot_id,
            scope_id=first.scope_id,
            session_id=first.session_id,
            first_event_sequence=first.sequence,
            last_event_sequence=event_tuple[-1].sequence,
            event_fingerprints=tuple(event.fingerprint for event in event_tuple),
            summary=summary,
            rollover_policy_fingerprint=rollover_policy_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "first_event_sequence": self.first_event_sequence,
            "last_event_sequence": self.last_event_sequence,
            "event_fingerprints": list(self.event_fingerprints),
            "summary": self.summary.to_dict(),
            "rollover_policy_fingerprint": self.rollover_policy_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ConsolidationJobSpec:
    """Content-addressed, retry-safe declaration of one consolidation attempt."""

    schema_version: ClassVar[str] = "molor.consolidation-job.v1"

    scope_id: str
    base_fingerprint: str
    parent_version_fingerprint: str | None
    snapshot_fingerprints: tuple[str, ...]
    item_cutoff_sequence: int
    dataset_builder_fingerprint: str
    trainer_fingerprint: str
    evaluation_suite_fingerprint: str
    target_lora_format_fingerprint: str
    policy_fingerprint: str

    def __post_init__(self) -> None:
        _identifier(self.scope_id, "scope_id")
        object.__setattr__(
            self,
            "base_fingerprint",
            validate_sha256(self.base_fingerprint, "base_fingerprint"),
        )
        if self.parent_version_fingerprint is not None:
            object.__setattr__(
                self,
                "parent_version_fingerprint",
                validate_sha256(
                    self.parent_version_fingerprint,
                    "parent_version_fingerprint",
                ),
            )
        object.__setattr__(
            self,
            "snapshot_fingerprints",
            _fingerprint_tuple(self.snapshot_fingerprints, "snapshot_fingerprints"),
        )
        _nonnegative_integer(self.item_cutoff_sequence, "item_cutoff_sequence")
        for field_name in (
            "dataset_builder_fingerprint",
            "trainer_fingerprint",
            "evaluation_suite_fingerprint",
            "target_lora_format_fingerprint",
            "policy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "base_fingerprint": self.base_fingerprint,
            "parent_version_fingerprint": self.parent_version_fingerprint,
            "snapshot_fingerprints": list(self.snapshot_fingerprints),
            "item_cutoff_sequence": self.item_cutoff_sequence,
            "dataset_builder_fingerprint": self.dataset_builder_fingerprint,
            "trainer_fingerprint": self.trainer_fingerprint,
            "evaluation_suite_fingerprint": self.evaluation_suite_fingerprint,
            "target_lora_format_fingerprint": self.target_lora_format_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
        }

    @property
    def job_key(self) -> str:
        return fingerprint_json(self.to_dict())


class JobState(StrEnum):
    """Persistent lifecycle states for at-least-once consolidation workers."""

    QUEUED = "queued"
    LEASED = "leased"
    TRAINING = "training"
    EVALUATING = "evaluating"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    SUPERSEDED = "superseded"
    RETRYABLE_FAILURE = "retryable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.PROMOTED,
            JobState.SUPERSEDED,
            JobState.TERMINAL_FAILURE,
            JobState.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class JobLease:
    """One expiring authority grant for a specific consolidation job attempt."""

    job_key: str
    lease_id: str
    worker_id: str
    attempt: int
    issued_at_unix_ms: int
    expires_at_unix_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_key", validate_sha256(self.job_key, "job_key"))
        _identifier(self.lease_id, "lease_id")
        _identifier(self.worker_id, "worker_id")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        issued = _nonnegative_integer(self.issued_at_unix_ms, "issued_at_unix_ms")
        expires = _nonnegative_integer(self.expires_at_unix_ms, "expires_at_unix_ms")
        if expires <= issued:
            raise ValueError("expires_at_unix_ms must be later than issued_at_unix_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "attempt": self.attempt,
            "issued_at_unix_ms": self.issued_at_unix_ms,
            "expires_at_unix_ms": self.expires_at_unix_ms,
        }


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Immutable queue-state snapshot; persistence may replace it but never mutate it."""

    spec: ConsolidationJobSpec
    state: JobState = JobState.QUEUED
    attempts: int = 0
    lease: JobLease | None = None
    candidate_version_fingerprint: str | None = None
    reason_code: str | None = None
    expected_head_generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ConsolidationJobSpec):
            raise TypeError("spec must be a ConsolidationJobSpec")
        if not isinstance(self.state, JobState):
            raise TypeError("state must be a JobState")
        _nonnegative_integer(self.attempts, "attempts")
        _nonnegative_integer(self.expected_head_generation, "expected_head_generation")
        if self.lease is not None and not isinstance(self.lease, JobLease):
            raise TypeError("lease must be a JobLease or None")
        if self.candidate_version_fingerprint is not None:
            object.__setattr__(
                self,
                "candidate_version_fingerprint",
                validate_sha256(
                    self.candidate_version_fingerprint,
                    "candidate_version_fingerprint",
                ),
            )
        if self.reason_code is not None:
            _identifier(self.reason_code, "reason_code")

        active_states = {JobState.LEASED, JobState.TRAINING, JobState.EVALUATING}
        candidate_states = {JobState.CANDIDATE, JobState.PROMOTED, JobState.SUPERSEDED}
        reason_states = {
            JobState.SUPERSEDED,
            JobState.RETRYABLE_FAILURE,
            JobState.TERMINAL_FAILURE,
            JobState.CANCELLED,
        }
        if self.state in active_states:
            if self.lease is None:
                raise ValueError("active job states require a lease")
            if self.lease.job_key != self.spec.job_key:
                raise ValueError("lease belongs to a different consolidation job")
            if self.lease.attempt != self.attempts:
                raise ValueError("lease attempt must equal the job attempt count")
        elif self.lease is not None:
            raise ValueError("only active job states may retain a lease")

        if self.state in candidate_states:
            if self.candidate_version_fingerprint is None:
                raise ValueError("candidate and promoted states require a memory version")
        elif self.candidate_version_fingerprint is not None:
            raise ValueError("this job state must not contain a candidate memory version")

        if self.state in reason_states:
            if self.reason_code is None:
                raise ValueError("failure, cancellation, and superseded states require a reason")
        elif self.reason_code is not None:
            raise ValueError("this job state must not contain a reason code")

        if self.state not in {JobState.QUEUED, JobState.CANCELLED} and self.attempts < 1:
            raise ValueError("this job state requires at least one attempt")

    @property
    def job_key(self) -> str:
        return self.spec.job_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "spec": self.spec.to_dict(),
            "state": self.state.value,
            "attempts": self.attempts,
            "lease": None if self.lease is None else self.lease.to_dict(),
            "candidate_version_fingerprint": self.candidate_version_fingerprint,
            "reason_code": self.reason_code,
            "expected_head_generation": self.expected_head_generation,
        }


@dataclass(frozen=True, slots=True)
class MemoryVersionManifest:
    """One complete, rebuildable memory adapter version for one authenticated scope."""

    schema_version: ClassVar[str] = "molor.memory-version.v1"

    version_id: str
    scope_id: str
    base_fingerprint: str
    parent_version_fingerprint: str | None
    consolidated_through_sequence: int
    consolidation_job_fingerprint: str
    corpus_manifest_fingerprint: str
    adapter: AdapterManifest
    trainer_fingerprint: str
    evaluation_report_fingerprint: str
    policy_fingerprint: str

    def __post_init__(self) -> None:
        _identifier(self.version_id, "version_id")
        _identifier(self.scope_id, "scope_id")
        object.__setattr__(
            self,
            "base_fingerprint",
            validate_sha256(self.base_fingerprint, "base_fingerprint"),
        )
        if self.parent_version_fingerprint is not None:
            object.__setattr__(
                self,
                "parent_version_fingerprint",
                validate_sha256(
                    self.parent_version_fingerprint,
                    "parent_version_fingerprint",
                ),
            )
        _nonnegative_integer(
            self.consolidated_through_sequence,
            "consolidated_through_sequence",
        )
        object.__setattr__(
            self,
            "consolidation_job_fingerprint",
            validate_sha256(
                self.consolidation_job_fingerprint,
                "consolidation_job_fingerprint",
            ),
        )
        if not isinstance(self.adapter, AdapterManifest):
            raise TypeError("adapter must be an AdapterManifest")
        if self.adapter.adapter_id == "memory/null":
            raise ValueError("memory/null is reserved for the explicit memory null route")
        if self.adapter.base.fingerprint != self.base_fingerprint:
            raise ValueError("memory adapter targets a different frozen base")
        for field_name in (
            "corpus_manifest_fingerprint",
            "trainer_fingerprint",
            "evaluation_report_fingerprint",
            "policy_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version_id": self.version_id,
            "scope_id": self.scope_id,
            "base_fingerprint": self.base_fingerprint,
            "parent_version_fingerprint": self.parent_version_fingerprint,
            "consolidated_through_sequence": self.consolidated_through_sequence,
            "consolidation_job_fingerprint": self.consolidation_job_fingerprint,
            "corpus_manifest_fingerprint": self.corpus_manifest_fingerprint,
            "adapter_fingerprint": self.adapter.fingerprint,
            "adapter": self.adapter.to_dict(),
            "trainer_fingerprint": self.trainer_fingerprint,
            "evaluation_report_fingerprint": self.evaluation_report_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class MemoryHead:
    """Current version pointer for one scope, suitable for compare-and-swap promotion."""

    schema_version: ClassVar[str] = "molor.memory-head.v1"

    scope_id: str
    generation: int
    version_fingerprint: str | None
    consolidated_through_sequence: int | None

    def __post_init__(self) -> None:
        _identifier(self.scope_id, "scope_id")
        _nonnegative_integer(self.generation, "generation")
        if self.version_fingerprint is None:
            if self.generation != 0 or self.consolidated_through_sequence is not None:
                raise ValueError("an empty memory head must use generation 0 and no cutoff")
        else:
            object.__setattr__(
                self,
                "version_fingerprint",
                validate_sha256(self.version_fingerprint, "version_fingerprint"),
            )
            if self.generation < 1:
                raise ValueError("a versioned memory head requires a positive generation")
            if self.consolidated_through_sequence is None:
                raise ValueError("a versioned memory head requires a consolidation cutoff")
            _nonnegative_integer(
                self.consolidated_through_sequence,
                "consolidated_through_sequence",
            )

    @classmethod
    def empty(cls, scope_id: str) -> MemoryHead:
        return cls(scope_id, 0, None, None)

    def promote(self, version: MemoryVersionManifest) -> MemoryHead:
        """Return the next head after validating parent, scope, and monotonic cutoff."""

        if not isinstance(version, MemoryVersionManifest):
            raise TypeError("version must be a MemoryVersionManifest")
        if version.scope_id != self.scope_id:
            raise ValueError("memory version belongs to a different scope")
        if version.parent_version_fingerprint != self.version_fingerprint:
            raise ValueError("memory version parent is not the current head")
        if (
            self.consolidated_through_sequence is not None
            and version.consolidated_through_sequence < self.consolidated_through_sequence
        ):
            raise ValueError("memory version consolidation cutoff must not move backwards")
        return MemoryHead(
            scope_id=self.scope_id,
            generation=self.generation + 1,
            version_fingerprint=version.fingerprint,
            consolidated_through_sequence=version.consolidated_through_sequence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "generation": self.generation,
            "version_fingerprint": self.version_fingerprint,
            "consolidated_through_sequence": self.consolidated_through_sequence,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class SessionMemoryPin:
    """One session's immutable memory head and independently versioned retrieval view."""

    schema_version: ClassVar[str] = "molor.session-memory-pin.v1"

    session_id: str
    scope_id: str
    head_generation: int
    version_fingerprint: str | None
    consolidated_through_sequence: int | None
    retrieval_cutoff_sequence: int | None

    def __post_init__(self) -> None:
        _identifier(self.session_id, "session_id")
        _identifier(self.scope_id, "scope_id")
        _nonnegative_integer(self.head_generation, "head_generation")
        if self.version_fingerprint is None:
            if self.head_generation != 0 or self.consolidated_through_sequence is not None:
                raise ValueError(
                    "an unversioned pin must use generation 0 and no consolidation cutoff"
                )
        else:
            object.__setattr__(
                self,
                "version_fingerprint",
                validate_sha256(self.version_fingerprint, "version_fingerprint"),
            )
            if self.head_generation < 1:
                raise ValueError("a versioned pin requires a positive head generation")
            if self.consolidated_through_sequence is None:
                raise ValueError("a versioned pin requires a consolidation cutoff")
            _nonnegative_integer(
                self.consolidated_through_sequence,
                "consolidated_through_sequence",
            )
        if self.retrieval_cutoff_sequence is not None:
            _nonnegative_integer(
                self.retrieval_cutoff_sequence,
                "retrieval_cutoff_sequence",
            )

    @classmethod
    def from_head(
        cls,
        session_id: str,
        head: MemoryHead,
        *,
        retrieval_cutoff_sequence: int | None,
    ) -> SessionMemoryPin:
        if not isinstance(head, MemoryHead):
            raise TypeError("head must be a MemoryHead")
        return cls(
            session_id=session_id,
            scope_id=head.scope_id,
            head_generation=head.generation,
            version_fingerprint=head.version_fingerprint,
            consolidated_through_sequence=head.consolidated_through_sequence,
            retrieval_cutoff_sequence=retrieval_cutoff_sequence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "scope_id": self.scope_id,
            "head_generation": self.head_generation,
            "version_fingerprint": self.version_fingerprint,
            "consolidated_through_sequence": self.consolidated_through_sequence,
            "retrieval_cutoff_sequence": self.retrieval_cutoff_sequence,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())
