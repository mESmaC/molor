from dataclasses import FrozenInstanceError, replace

import pytest

from molor.identity import fingerprint_json
from molor.memory import (
    ConsolidationJobSpec,
    ContextRolloverPolicy,
    JobLease,
    JobRecord,
    JobState,
    MemoryEvent,
    MemoryEventKind,
    MemoryHead,
    MemoryVersionManifest,
    RolloverSnapshotManifest,
    SessionMemoryPin,
)

from test_manifests import adapter, artifact, base, digest, lora


SCOPE = "user/alice"
SESSION = "session/one"


def test_context_rollover_triggers_before_the_hard_limit() -> None:
    policy = ContextRolloverPolicy(
        max_context_tokens=131_072,
        trigger_tokens=110_000,
        minimum_generation_tokens=16_384,
    )

    assert not policy.should_rollover(109_999)
    assert policy.should_rollover(110_000)
    assert policy.fingerprint == fingerprint_json(policy.to_dict())

    with pytest.raises(ValueError, match="generation headroom"):
        ContextRolloverPolicy(131_072, 120_000, 16_384)


def observation(
    event_id: str,
    sequence: int,
    *,
    scope_id: str = SCOPE,
    session_id: str = SESSION,
) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        scope_id=scope_id,
        session_id=session_id,
        sequence=sequence,
        kind=MemoryEventKind.OBSERVATION,
        payload={"fact": event_id},
    )


def job_spec(
    *,
    snapshots: tuple[str, ...] = (digest("1"), digest("2")),
    cutoff: int = 9,
    parent: str | None = None,
) -> ConsolidationJobSpec:
    return ConsolidationJobSpec(
        scope_id=SCOPE,
        base_fingerprint=base().fingerprint,
        parent_version_fingerprint=parent,
        snapshot_fingerprints=snapshots,
        item_cutoff_sequence=cutoff,
        dataset_builder_fingerprint=digest("3"),
        trainer_fingerprint=digest("4"),
        evaluation_suite_fingerprint=digest("5"),
        target_lora_format_fingerprint=lora().fingerprint,
        policy_fingerprint=digest("6"),
    )


def memory_version(
    *,
    version_id: str = "memory-v1",
    scope_id: str = SCOPE,
    parent: str | None = None,
    cutoff: int = 5,
    base_manifest=None,
    adapter_marker: str = "1",
) -> MemoryVersionManifest:
    frozen_base = base_manifest or base()
    memory_adapter = adapter(
        "memory/alice",
        adapter_marker,
        base_manifest=frozen_base,
        revision=version_id,
    )
    return MemoryVersionManifest(
        version_id=version_id,
        scope_id=scope_id,
        base_fingerprint=frozen_base.fingerprint,
        parent_version_fingerprint=parent,
        consolidated_through_sequence=cutoff,
        consolidation_job_fingerprint=digest("1"),
        corpus_manifest_fingerprint=digest("2"),
        adapter=memory_adapter,
        trainer_fingerprint=digest("3"),
        evaluation_report_fingerprint=digest("4"),
        policy_fingerprint=digest("6"),
    )


def test_memory_event_fingerprint_is_canonical_and_payload_is_deeply_immutable() -> None:
    raw_payload = {
        "profile": {"topics": ["systems", "models"], "timezone": "America/New_York"},
        "active": True,
    }
    event = MemoryEvent(
        "event-1",
        SCOPE,
        SESSION,
        0,
        MemoryEventKind.OBSERVATION,
        raw_payload,
    )
    equivalent = MemoryEvent(
        "event-1",
        SCOPE,
        SESSION,
        0,
        MemoryEventKind.OBSERVATION,
        {
            "active": True,
            "profile": {
                "timezone": "America/New_York",
                "topics": ["systems", "models"],
            },
        },
    )

    raw_payload["profile"]["topics"].append("mutated")

    assert event.fingerprint == equivalent.fingerprint
    assert event.fingerprint == fingerprint_json(event.to_dict())
    assert event.to_dict()["payload"]["profile"]["topics"] == ["systems", "models"]
    with pytest.raises(TypeError):
        event.payload["active"] = False
    with pytest.raises(FrozenInstanceError):
        event.sequence = 2  # type: ignore[misc]


def test_event_constructor_enforces_correction_and_tombstone_target_contracts() -> None:
    source = observation("event-1", 0)

    with pytest.raises(ValueError, match="must not target"):
        MemoryEvent(
            "event-2",
            SCOPE,
            SESSION,
            1,
            MemoryEventKind.OBSERVATION,
            {"fact": "new"},
            source.event_id,
            source.fingerprint,
        )
    with pytest.raises(ValueError, match="require a target"):
        MemoryEvent(
            "event-2",
            SCOPE,
            SESSION,
            1,
            MemoryEventKind.CORRECTION,
            {"fact": "new"},
        )
    with pytest.raises(ValueError, match="supplied together"):
        MemoryEvent(
            "event-2",
            SCOPE,
            SESSION,
            1,
            MemoryEventKind.TOMBSTONE,
            None,
            source.event_id,
        )
    with pytest.raises(ValueError, match="must not contain payload"):
        MemoryEvent(
            "event-2",
            SCOPE,
            SESSION,
            1,
            MemoryEventKind.TOMBSTONE,
            {"reason": "obsolete"},
            source.event_id,
            source.fingerprint,
        )
    with pytest.raises(ValueError, match="must not target itself"):
        MemoryEvent(
            "event-1",
            SCOPE,
            SESSION,
            1,
            MemoryEventKind.CORRECTION,
            {"fact": "new"},
            source.event_id,
            source.fingerprint,
        )


def test_resolved_event_targets_must_match_identity_scope_and_order() -> None:
    source = observation("event-1", 0)
    correction = MemoryEvent(
        "event-2",
        SCOPE,
        SESSION,
        1,
        MemoryEventKind.CORRECTION,
        {"fact": "replacement"},
        source.event_id,
        source.fingerprint,
    )
    correction.assert_valid_target(source)

    wrong_scope = observation("event-1", 0, scope_id="user/bob")
    with pytest.raises(ValueError, match="different memory scope"):
        correction.assert_valid_target(wrong_scope)
    wrong_identity = observation("event-other", 0)
    with pytest.raises(ValueError, match="declared target identity"):
        correction.assert_valid_target(wrong_identity)

    future = observation("future", 2)
    correction_of_future = MemoryEvent(
        "event-3",
        SCOPE,
        SESSION,
        1,
        MemoryEventKind.CORRECTION,
        {"fact": "replacement"},
        future.event_id,
        future.fingerprint,
    )
    with pytest.raises(ValueError, match="earlier event"):
        correction_of_future.assert_valid_target(future)

    tombstone = MemoryEvent(
        "event-4",
        SCOPE,
        SESSION,
        2,
        MemoryEventKind.TOMBSTONE,
        None,
        source.event_id,
        source.fingerprint,
    )
    tombstone.assert_valid_target(source)
    targets_tombstone = MemoryEvent(
        "event-5",
        SCOPE,
        SESSION,
        3,
        MemoryEventKind.CORRECTION,
        {"fact": "invalid"},
        tombstone.event_id,
        tombstone.fingerprint,
    )
    with pytest.raises(ValueError, match="content-bearing"):
        targets_tombstone.assert_valid_target(tombstone)


def test_rollover_snapshot_binds_ordered_contiguous_event_fingerprints() -> None:
    events = (observation("event-4", 4), observation("event-5", 5))
    summary = artifact("rollover.json", "a", media_type="application/json")
    snapshot = RolloverSnapshotManifest.from_events(
        "snapshot-1",
        events,
        summary=summary,
        rollover_policy_fingerprint=digest("b"),
    )

    assert snapshot.first_event_sequence == 4
    assert snapshot.last_event_sequence == 5
    assert snapshot.event_fingerprints == tuple(event.fingerprint for event in events)

    reversed_identity = RolloverSnapshotManifest(
        snapshot.snapshot_id,
        snapshot.scope_id,
        snapshot.session_id,
        snapshot.first_event_sequence,
        snapshot.last_event_sequence,
        tuple(reversed(snapshot.event_fingerprints)),
        snapshot.summary,
        snapshot.rollover_policy_fingerprint,
    )
    assert snapshot.fingerprint != reversed_identity.fingerprint

    with pytest.raises(ValueError, match="ordered and contiguous"):
        RolloverSnapshotManifest.from_events(
            "snapshot-2",
            reversed(events),
            summary=summary,
            rollover_policy_fingerprint=digest("b"),
        )
    with pytest.raises(ValueError, match="ordered and contiguous"):
        RolloverSnapshotManifest.from_events(
            "snapshot-2",
            (observation("event-0", 0), observation("event-2", 2)),
            summary=summary,
            rollover_policy_fingerprint=digest("b"),
        )
    with pytest.raises(ValueError, match="one memory scope"):
        RolloverSnapshotManifest.from_events(
            "snapshot-2",
            (observation("event-0", 0), observation("event-1", 1, scope_id="user/bob")),
            summary=summary,
            rollover_policy_fingerprint=digest("b"),
        )


def test_consolidation_job_key_is_content_addressed_and_order_sensitive() -> None:
    original = job_spec()
    equivalent = job_spec()
    reordered = job_spec(snapshots=tuple(reversed(original.snapshot_fingerprints)))
    later_cutoff = job_spec(cutoff=10)
    other_base = replace(original, base_fingerprint=digest("a"))
    other_policy = replace(original, policy_fingerprint=digest("b"))

    assert original.job_key == equivalent.job_key
    assert original.job_key == fingerprint_json(original.to_dict())
    assert original.job_key != reordered.job_key
    assert original.job_key != later_cutoff.job_key
    assert original.job_key != other_base.job_key
    assert original.job_key != other_policy.job_key
    with pytest.raises(ValueError, match="duplicates"):
        job_spec(snapshots=(digest("1"), digest("1")))


def test_job_lease_and_active_record_bind_job_key_and_attempt() -> None:
    spec = job_spec()
    lease = JobLease(spec.job_key, "lease-1", "worker-1", 1, 100, 200)
    active = JobRecord(spec, JobState.TRAINING, attempts=1, lease=lease)

    assert active.job_key == spec.job_key
    assert active.to_dict()["lease"] == lease.to_dict()
    with pytest.raises(ValueError, match="require a lease"):
        JobRecord(spec, JobState.LEASED, attempts=1)
    with pytest.raises(ValueError, match="attempt must equal"):
        JobRecord(spec, JobState.EVALUATING, attempts=2, lease=lease)

    another = job_spec(cutoff=10)
    with pytest.raises(ValueError, match="different consolidation job"):
        JobRecord(another, JobState.TRAINING, attempts=1, lease=lease)
    with pytest.raises(ValueError, match="positive integer"):
        JobLease(spec.job_key, "lease-1", "worker-1", 0, 100, 200)
    with pytest.raises(ValueError, match="later than"):
        JobLease(spec.job_key, "lease-1", "worker-1", 1, 200, 200)


def test_job_record_requires_state_specific_candidate_and_reason_fields() -> None:
    spec = job_spec()
    candidate_fingerprint = digest("f")
    candidate = JobRecord(
        spec,
        JobState.CANDIDATE,
        attempts=1,
        candidate_version_fingerprint=candidate_fingerprint,
    )
    promoted = JobRecord(
        spec,
        JobState.PROMOTED,
        attempts=1,
        candidate_version_fingerprint=candidate_fingerprint,
    )
    retryable = JobRecord(
        spec,
        JobState.RETRYABLE_FAILURE,
        attempts=1,
        reason_code="worker-lost",
    )
    cancelled = JobRecord(spec, JobState.CANCELLED, reason_code="user-request")

    assert candidate.state is JobState.CANDIDATE
    assert promoted.state.terminal
    assert not retryable.state.terminal
    assert cancelled.state.terminal
    with pytest.raises(ValueError, match="require a memory version"):
        JobRecord(spec, JobState.CANDIDATE, attempts=1)
    with pytest.raises(ValueError, match="require a reason"):
        JobRecord(spec, JobState.TERMINAL_FAILURE, attempts=1)
    with pytest.raises(ValueError, match="must not contain a candidate"):
        JobRecord(
            spec,
            JobState.QUEUED,
            candidate_version_fingerprint=candidate_fingerprint,
        )


def test_memory_version_binds_one_adapter_to_the_exact_frozen_base() -> None:
    frozen_base = base()
    version = memory_version(base_manifest=frozen_base)

    assert version.base_fingerprint == frozen_base.fingerprint
    assert version.adapter.base.fingerprint == version.base_fingerprint
    assert version.to_dict()["adapter"] == version.adapter.to_dict()
    assert version.fingerprint == fingerprint_json(version.to_dict())

    other_base = base(weights="e")
    with pytest.raises(ValueError, match="different frozen base"):
        MemoryVersionManifest(
            version_id="memory-invalid",
            scope_id=SCOPE,
            base_fingerprint=other_base.fingerprint,
            parent_version_fingerprint=None,
            consolidated_through_sequence=5,
            consolidation_job_fingerprint=digest("1"),
            corpus_manifest_fingerprint=digest("2"),
            adapter=version.adapter,
            trainer_fingerprint=digest("3"),
            evaluation_report_fingerprint=digest("4"),
            policy_fingerprint=digest("5"),
        )


def test_memory_version_fingerprint_commits_to_lineage_and_adapter() -> None:
    original = memory_version()
    with_parent = memory_version(parent=digest("a"))
    retrained = memory_version(adapter_marker="6")

    assert original.fingerprint != with_parent.fingerprint
    assert original.fingerprint != retrained.fingerprint


def test_memory_head_promotion_enforces_scope_parent_generation_and_cutoff() -> None:
    empty = MemoryHead.empty(SCOPE)
    first_version = memory_version()
    first_head = empty.promote(first_version)

    assert empty.generation == 0
    assert empty.version_fingerprint is None
    assert first_head.generation == 1
    assert first_head.version_fingerprint == first_version.fingerprint
    assert first_head.consolidated_through_sequence == 5
    assert first_head.to_dict()["schema_version"] == MemoryHead.schema_version

    second_version = memory_version(
        version_id="memory-v2",
        parent=first_head.version_fingerprint,
        cutoff=8,
        adapter_marker="2",
    )
    second_head = first_head.promote(second_version)
    assert second_head.generation == 2
    assert second_head.consolidated_through_sequence == 8

    with pytest.raises(ValueError, match="current head"):
        first_head.promote(memory_version(version_id="orphan", cutoff=8))
    with pytest.raises(ValueError, match="different scope"):
        empty.promote(memory_version(scope_id="user/bob"))
    with pytest.raises(ValueError, match="move backwards"):
        first_head.promote(
            memory_version(
                version_id="regression",
                parent=first_head.version_fingerprint,
                cutoff=4,
            )
        )
    with pytest.raises(ValueError, match="empty memory head"):
        MemoryHead(SCOPE, 1, None, None)


def test_session_pin_captures_one_head_and_valid_retrieval_cutoff() -> None:
    empty = MemoryHead.empty(SCOPE)
    empty_pin = SessionMemoryPin.from_head(
        SESSION,
        empty,
        retrieval_cutoff_sequence=3,
    )
    head = empty.promote(memory_version(cutoff=5))
    pin = SessionMemoryPin.from_head(
        SESSION,
        head,
        retrieval_cutoff_sequence=7,
    )

    assert empty_pin.head_generation == 0
    assert empty_pin.version_fingerprint is None
    assert pin.scope_id == head.scope_id
    assert pin.head_generation == head.generation
    assert pin.version_fingerprint == head.version_fingerprint
    assert pin.consolidated_through_sequence == 5
    assert pin.retrieval_cutoff_sequence == 7
    assert pin.to_dict()["schema_version"] == SessionMemoryPin.schema_version

    independent_retrieval_cut = SessionMemoryPin.from_head(
        SESSION,
        head,
        retrieval_cutoff_sequence=4,
    )
    assert independent_retrieval_cut.consolidated_through_sequence == 5
    assert independent_retrieval_cut.retrieval_cutoff_sequence == 4
    with pytest.raises(ValueError, match="unversioned pin"):
        SessionMemoryPin(SESSION, SCOPE, 1, None, None, 3)
