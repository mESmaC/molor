from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier

import pytest

from molor.memory import (
    ConsolidationJobSpec,
    JobState,
    LeaseLostError,
    MemoryEvent,
    MemoryEventKind,
    MemoryVersionManifest,
    RolloverSnapshotManifest,
    SQLiteMemoryStore,
    StaleMemoryHeadError,
)

from test_manifests import adapter, artifact, base, digest, lora


SCOPE = "user/alice"
SESSION = "session/one"


def observation(
    event_id: str,
    sequence: int,
    payload: object,
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
        payload=payload,
    )


def snapshot(
    events: tuple[MemoryEvent, ...],
    *,
    snapshot_id: str = "snapshot/one",
    marker: str = "a",
) -> RolloverSnapshotManifest:
    return RolloverSnapshotManifest.from_events(
        snapshot_id,
        events,
        summary=artifact("summary.json", marker, media_type="application/json"),
        rollover_policy_fingerprint=digest("b"),
    )


def job(
    rollover: RolloverSnapshotManifest,
    *,
    cutoff: int,
    trainer_marker: str = "d",
) -> ConsolidationJobSpec:
    return ConsolidationJobSpec(
        scope_id=rollover.scope_id,
        base_fingerprint=base().fingerprint,
        parent_version_fingerprint=None,
        snapshot_fingerprints=(rollover.fingerprint,),
        item_cutoff_sequence=cutoff,
        dataset_builder_fingerprint=digest("c"),
        trainer_fingerprint=digest(trainer_marker),
        evaluation_suite_fingerprint=digest("e"),
        target_lora_format_fingerprint=lora().fingerprint,
        policy_fingerprint=digest("9"),
    )


def version(
    spec: ConsolidationJobSpec,
    *,
    version_id: str = "memory/v1",
    adapter_marker: str = "6",
) -> MemoryVersionManifest:
    frozen_base = base()
    return MemoryVersionManifest(
        version_id=version_id,
        scope_id=spec.scope_id,
        base_fingerprint=frozen_base.fingerprint,
        parent_version_fingerprint=spec.parent_version_fingerprint,
        consolidated_through_sequence=spec.item_cutoff_sequence,
        consolidation_job_fingerprint=spec.job_key,
        corpus_manifest_fingerprint=digest("7"),
        adapter=adapter(
            "memory/alice",
            adapter_marker,
            base_manifest=frozen_base,
            revision=version_id,
        ),
        trainer_fingerprint=spec.trainer_fingerprint,
        evaluation_report_fingerprint=digest("8"),
        policy_fingerprint=digest("9"),
    )


def test_event_ledger_is_idempotent_ordered_scoped_and_persistent(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    first = observation("event/one", 0, {"fact": "Alice likes blue"})
    second = observation("event/two", 1, {"fact": "The project is molor"})
    other_scope = observation(
        "event/one",
        0,
        {"fact": "Bob likes green"},
        scope_id="user/bob",
    )

    with SQLiteMemoryStore(path) as store:
        assert store.append_event(first) == first
        assert store.append_event(first) == first
        store.append_event(second)
        store.append_event(other_scope)

        with pytest.raises(ValueError, match="different content"):
            store.append_event(observation("event/one", 2, {"fact": "changed"}))
        with pytest.raises(ValueError, match="append exactly"):
            store.append_event(observation("event/gap", 3, {"fact": "gap"}))

    with SQLiteMemoryStore(path) as reopened:
        assert reopened.list_events(SCOPE, SESSION) == (first, second)
        assert reopened.active_events(SCOPE) == (first, second)
        assert reopened.active_events("user/bob") == (other_scope,)


def test_store_allocates_observation_sequences_atomically_across_connections(tmp_path) -> None:
    path = tmp_path / "concurrent-memory.sqlite3"
    with SQLiteMemoryStore(path):
        pass
    barrier = Barrier(2)

    def append(marker: str) -> MemoryEvent:
        with SQLiteMemoryStore(path) as store:
            barrier.wait()
            return store.append_observation_auto(
                event_id=f"event/{marker}",
                scope_id=SCOPE,
                session_id=SESSION,
                payload={"text": marker},
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        events = tuple(executor.map(append, ("one", "two")))

    assert sorted(event.sequence for event in events) == [0, 1]
    with SQLiteMemoryStore(path) as reopened:
        persisted = reopened.list_events(SCOPE, SESSION)
        assert tuple(event.sequence for event in persisted) == (0, 1)
        repeated = reopened.append_observation_auto(
            event_id=events[0].event_id,
            scope_id=SCOPE,
            session_id=SESSION,
            payload=events[0].to_dict()["payload"],
        )
        assert repeated == events[0]
        with pytest.raises(ValueError, match="different content"):
            reopened.append_observation_auto(
                event_id=events[0].event_id,
                scope_id=SCOPE,
                session_id=SESSION,
                payload={"text": "changed"},
            )


def test_corrections_and_tombstones_take_effect_before_retrieval() -> None:
    source = observation("event/favorite", 0, {"favorite_color": "blue"})
    correction = MemoryEvent(
        event_id="event/favorite-correction",
        scope_id=SCOPE,
        session_id=SESSION,
        sequence=1,
        kind=MemoryEventKind.CORRECTION,
        payload={"favorite_color": "red"},
        target_event_id=source.event_id,
        target_event_fingerprint=source.fingerprint,
    )
    tombstone = MemoryEvent(
        event_id="event/favorite-deletion",
        scope_id=SCOPE,
        session_id=SESSION,
        sequence=2,
        kind=MemoryEventKind.TOMBSTONE,
        payload=None,
        target_event_id=correction.event_id,
        target_event_fingerprint=correction.fingerprint,
    )

    with SQLiteMemoryStore() as store:
        store.append_event(source)
        store.append_event(correction)

        assert store.active_events(SCOPE) == (correction,)
        hits = store.search_active_events(SCOPE, "favorite red")
        assert tuple(hit.event for hit in hits) == (correction,)
        assert store.search_active_events("user/bob", "favorite red") == ()

        with pytest.raises(ValueError, match="already been corrected"):
            store.append_event(
                MemoryEvent(
                    "event/branch",
                    SCOPE,
                    SESSION,
                    2,
                    MemoryEventKind.CORRECTION,
                    {"favorite_color": "green"},
                    source.event_id,
                    source.fingerprint,
                )
            )
        store.append_event(tombstone)
        assert store.active_events(SCOPE) == ()
        assert store.search_active_events(SCOPE, "favorite red") == ()


def test_session_pin_enforces_an_as_of_retrieval_view() -> None:
    source = observation("event/favorite", 0, {"favorite_color": "blue"})
    correction = MemoryEvent(
        "event/favorite-correction",
        SCOPE,
        SESSION,
        1,
        MemoryEventKind.CORRECTION,
        {"favorite_color": "red"},
        source.event_id,
        source.fingerprint,
    )
    tombstone = MemoryEvent(
        "event/favorite-deletion",
        SCOPE,
        SESSION,
        2,
        MemoryEventKind.TOMBSTONE,
        None,
        correction.event_id,
        correction.fingerprint,
    )

    with SQLiteMemoryStore() as store:
        store.append_event(source)
        cutoff = store.current_retrieval_cutoff(SCOPE)
        assert cutoff is not None
        pin = store.pin_session(
            SCOPE,
            "session/pinned",
            retrieval_cutoff_sequence=cutoff,
        )
        store.append_event(correction)

        assert store.active_events(SCOPE, pin=pin) == (source,)
        assert store.active_events(SCOPE) == (correction,)
        assert tuple(
            hit.event for hit in store.search_active_events(SCOPE, "favorite blue", pin=pin)
        ) == (source,)
        with pytest.raises(ValueError, match="different memory scope"):
            store.active_events("user/bob", pin=pin)

        store.append_event(tombstone)
        assert store.active_events(SCOPE, pin=pin) == (source,)
        assert store.active_events(SCOPE) == ()


def test_session_pin_cannot_name_or_drift_into_a_future_cutoff() -> None:
    with SQLiteMemoryStore() as store:
        with pytest.raises(ValueError, match="beyond the current event ledger frontier"):
            store.pin_session(
                SCOPE,
                "session/future",
                retrieval_cutoff_sequence=100,
            )

        pin = store.pin_session(SCOPE, "session/pinned")
        assert pin.retrieval_cutoff_sequence == 0
        store.append_event(observation("event/later", 0, {"text": "later"}))
        assert store.active_events(SCOPE, pin=pin) == ()

        forged = replace(pin, retrieval_cutoff_sequence=1)
        with pytest.raises(ValueError, match="differs from the store-issued"):
            store.active_events(SCOPE, pin=forged)


def test_rollover_and_job_are_one_idempotent_transaction(tmp_path) -> None:
    path = tmp_path / "queue.sqlite3"
    events = (
        observation("event/one", 0, {"text": "first"}),
        observation("event/two", 1, {"text": "second"}),
    )
    rollover = snapshot(events)
    spec = job(rollover, cutoff=1)

    with SQLiteMemoryStore(path) as store:
        for event in events:
            store.append_event(event)
        queued = store.record_rollover(rollover, spec, now_unix_ms=100)
        repeated = store.record_rollover(rollover, spec, now_unix_ms=200)

        assert queued.state is JobState.QUEUED
        assert repeated == queued
        assert store.get_snapshot(rollover.fingerprint) == rollover

    with SQLiteMemoryStore(path) as reopened:
        assert reopened.get_job(spec.job_key) == queued
        assert reopened.get_snapshot(rollover.fingerprint) == rollover


def test_failed_job_insert_rolls_back_snapshot_registration() -> None:
    event = observation("event/one", 0, {"text": "first"})
    rollover = snapshot((event,))
    invalid_job = replace(
        job(rollover, cutoff=0),
        snapshot_fingerprints=(rollover.fingerprint, digest("f")),
    )

    with SQLiteMemoryStore() as store:
        store.append_event(event)
        with pytest.raises(ValueError, match="unknown rollover snapshot"):
            store.record_rollover(rollover, invalid_job, now_unix_ms=0)
        with pytest.raises(KeyError, match="unknown rollover snapshot"):
            store.get_snapshot(rollover.fingerprint)


def test_queue_recovers_expired_leases_and_applies_retry_backoff() -> None:
    events = (observation("event/one", 0, {"text": "first"}),)
    rollover = snapshot(events)
    spec = job(rollover, cutoff=0)

    with SQLiteMemoryStore() as store:
        store.append_event(events[0])
        store.record_rollover(rollover, spec, now_unix_ms=0)
        first = store.lease_next("worker/one", lease_seconds=1, now_unix_ms=100)
        assert first is not None
        assert store.lease_next("worker/two", lease_seconds=1, now_unix_ms=500) is None

        second = store.lease_next("worker/two", lease_seconds=2, now_unix_ms=1100)
        assert second is not None
        assert second.attempt == 2
        with pytest.raises(LeaseLostError):
            store.start_training(first, now_unix_ms=1100)

        store.start_training(second, now_unix_ms=1200)
        failed = store.fail(
            second,
            retryable=True,
            reason_code="transient-oom",
            retry_after_seconds=3,
            now_unix_ms=1300,
        )
        assert failed.state is JobState.RETRYABLE_FAILURE
        assert store.lease_next("worker/three", lease_seconds=1, now_unix_ms=4299) is None
        third = store.lease_next("worker/three", lease_seconds=1, now_unix_ms=4300)
        assert third is not None
        assert third.attempt == 3


def test_heartbeat_is_idempotent_when_a_successful_response_is_lost() -> None:
    event = observation("event/one", 0, {"text": "first"})
    rollover = snapshot((event,))
    spec = job(rollover, cutoff=0)

    with SQLiteMemoryStore() as store:
        store.append_event(event)
        store.record_rollover(rollover, spec, now_unix_ms=0)
        original = store.lease_next("worker/one", lease_seconds=1, now_unix_ms=100)
        assert original is not None

        renewed = store.heartbeat(original, lease_seconds=2, now_unix_ms=200)
        replayed = store.heartbeat(original, lease_seconds=2, now_unix_ms=300)

        assert renewed.lease_id == replayed.lease_id == original.lease_id
        assert replayed.expires_at_unix_ms == 2300
        assert store.start_training(original, now_unix_ms=400).state is JobState.TRAINING


def test_two_connections_observe_one_live_lease(tmp_path) -> None:
    path = tmp_path / "shared-memory.sqlite3"
    event = observation("event/one", 0, {"text": "first"})
    rollover = snapshot((event,))
    spec = job(rollover, cutoff=0)

    with SQLiteMemoryStore(path) as first, SQLiteMemoryStore(path) as second:
        first.append_event(event)
        first.record_rollover(rollover, spec, now_unix_ms=0)
        first_lease = first.lease_next("worker/one", lease_seconds=1, now_unix_ms=100)
        assert first_lease is not None
        assert second.lease_next("worker/two", lease_seconds=1, now_unix_ms=500) is None

        recovered = second.lease_next("worker/two", lease_seconds=1, now_unix_ms=1100)
        assert recovered is not None
        assert recovered.attempt == 2


def test_persisted_content_identities_are_revalidated_on_read() -> None:
    event = observation("event/one", 0, {"text": "first"})
    rollover = snapshot((event,))
    spec = job(rollover, cutoff=0)

    with SQLiteMemoryStore() as store:
        store.append_event(event)
        store.record_rollover(rollover, spec, now_unix_ms=0)
        lease = store.lease_next("worker/one", lease_seconds=10, now_unix_ms=100)
        assert lease is not None
        store.start_training(lease, now_unix_ms=200)
        store.start_evaluating(lease, now_unix_ms=300)
        candidate = version(spec)
        store.register_candidate(lease, candidate, now_unix_ms=400)

        snapshot_data = rollover.to_dict()
        snapshot_data["snapshot_id"] = "snapshot/tampered"
        store._connection.execute(  # noqa: SLF001 - intentional corruption injection
            "UPDATE rollover_snapshots SET manifest_json = ? WHERE fingerprint = ?",
            (json.dumps(snapshot_data), rollover.fingerprint),
        )
        with pytest.raises(RuntimeError, match="content identity"):
            store.get_snapshot(rollover.fingerprint)

        job_data = spec.to_dict()
        job_data["trainer_fingerprint"] = digest("f")
        store._connection.execute(  # noqa: SLF001 - intentional corruption injection
            "UPDATE consolidation_jobs SET spec_json = ? WHERE job_key = ?",
            (json.dumps(job_data), spec.job_key),
        )
        with pytest.raises(RuntimeError, match="content identity"):
            store.get_job(spec.job_key)

        version_data = candidate.to_dict()
        version_data["policy_fingerprint"] = digest("a")
        store._connection.execute(  # noqa: SLF001 - intentional corruption injection
            "UPDATE memory_versions SET manifest_json = ? WHERE fingerprint = ?",
            (json.dumps(version_data), candidate.fingerprint),
        )
        with pytest.raises(RuntimeError, match="content identity"):
            store.get_version_dict(candidate.fingerprint)

        store._connection.execute(  # noqa: SLF001 - intentional corruption injection
            "UPDATE memory_events SET payload_json = ? WHERE fingerprint = ?",
            (json.dumps({"text": "tampered"}), event.fingerprint),
        )
        with pytest.raises(RuntimeError, match="content identity"):
            store.get_event(SCOPE, event.event_id)


def test_snapshot_columns_are_revalidated_at_registration_and_queue_boundaries() -> None:
    event = observation("event/one", 0, {"text": "first"})
    rollover = snapshot((event,))
    spec = job(rollover, cutoff=0)

    with SQLiteMemoryStore() as store:
        store.append_event(event)
        store.record_rollover(rollover, spec, now_unix_ms=0)

        store._connection.execute(  # noqa: SLF001 - intentional corruption injection
            "UPDATE rollover_snapshots SET session_id = ? WHERE fingerprint = ?",
            ("session/tampered", rollover.fingerprint),
        )
        with pytest.raises(RuntimeError, match="columns disagree"):
            store.get_snapshot(rollover.fingerprint)
        with pytest.raises(RuntimeError, match="columns disagree"):
            store.record_rollover(rollover, spec, now_unix_ms=1)

        store._connection.execute(  # noqa: SLF001 - intentional corruption injection
            """
            UPDATE rollover_snapshots
            SET session_id = ?, scope_id = ?
            WHERE fingerprint = ?
            """,
            (rollover.session_id, "user/bob", rollover.fingerprint),
        )
        distinct_job = replace(spec, trainer_fingerprint=digest("f"))
        with pytest.raises(RuntimeError, match="columns disagree"):
            store.enqueue(distinct_job, now_unix_ms=2)


def test_session_pin_keeps_retrieval_and_consolidation_coordinates_independent() -> None:
    event = observation("event/one", 0, {"text": "first"})
    rollover = snapshot((event,))
    spec = job(rollover, cutoff=100)

    with SQLiteMemoryStore() as store:
        store.append_event(event)
        store.record_rollover(rollover, spec, now_unix_ms=0)
        lease = store.lease_next("worker/one", lease_seconds=10, now_unix_ms=100)
        assert lease is not None
        store.start_training(lease, now_unix_ms=200)
        store.start_evaluating(lease, now_unix_ms=300)
        store.register_candidate(lease, version(spec), now_unix_ms=400)
        store.promote_candidate(spec.job_key, now_unix_ms=500)

        pin = store.pin_session(SCOPE, "session/pinned")

        assert pin.consolidated_through_sequence == 100
        assert pin.retrieval_cutoff_sequence == 1


def test_candidate_promotion_is_cas_and_sessions_keep_their_pin() -> None:
    first_event = observation("event/one", 0, {"text": "first"})
    second_event = observation(
        "event/two",
        0,
        {"text": "second"},
        session_id="session/two",
    )
    first_snapshot = snapshot((first_event,), snapshot_id="snapshot/one", marker="a")
    second_snapshot = snapshot((second_event,), snapshot_id="snapshot/two", marker="f")
    first_spec = job(first_snapshot, cutoff=1, trainer_marker="d")
    second_spec = job(second_snapshot, cutoff=1, trainer_marker="e")

    with SQLiteMemoryStore() as store:
        store.append_event(first_event)
        store.append_event(second_event)
        store.record_rollover(first_snapshot, first_spec, now_unix_ms=0)
        store.record_rollover(second_snapshot, second_spec, now_unix_ms=0)
        old_pin = store.pin_session(SCOPE, "session/old", retrieval_cutoff_sequence=1)

        first_lease = store.lease_next("worker/one", lease_seconds=10, now_unix_ms=100)
        assert first_lease is not None
        store.start_training(first_lease, now_unix_ms=200)
        store.start_evaluating(first_lease, now_unix_ms=300)
        first_version = version(first_spec)
        with pytest.raises(ValueError, match="exact consolidation job"):
            store.register_candidate(
                first_lease,
                replace(first_version, consolidation_job_fingerprint=digest("a")),
                now_unix_ms=325,
            )
        with pytest.raises(ValueError, match="policy differs"):
            store.register_candidate(
                first_lease,
                replace(first_version, policy_fingerprint=digest("a")),
                now_unix_ms=350,
            )
        store.register_candidate(first_lease, first_version, now_unix_ms=400)

        second_lease = store.lease_next("worker/two", lease_seconds=10, now_unix_ms=500)
        assert second_lease is not None
        store.start_training(second_lease, now_unix_ms=600)
        store.start_evaluating(second_lease, now_unix_ms=700)
        second_version = version(
            second_spec,
            version_id="memory/v2-racing",
            adapter_marker="5",
        )
        store.register_candidate(second_lease, second_version, now_unix_ms=800)

        store._connection.execute(  # noqa: SLF001 - intentional corruption injection
            """
            UPDATE consolidation_jobs
            SET candidate_version_fingerprint = ?
            WHERE job_key = ?
            """,
            (second_version.fingerprint, first_spec.job_key),
        )
        with pytest.raises(RuntimeError, match="candidate disagrees"):
            store.get_job(first_spec.job_key)
        with pytest.raises(RuntimeError, match="candidate disagrees"):
            store.promote_candidate(
                first_spec.job_key,
                expected_head_generation=0,
                now_unix_ms=850,
            )
        assert store.get_head(SCOPE).generation == 0
        store._connection.execute(  # noqa: SLF001 - intentional corruption repair
            """
            UPDATE consolidation_jobs
            SET candidate_version_fingerprint = ?
            WHERE job_key = ?
            """,
            (first_version.fingerprint, first_spec.job_key),
        )

        head, promoted = store.promote_candidate(
            first_spec.job_key,
            expected_head_generation=0,
            now_unix_ms=900,
        )
        assert head.version_fingerprint == first_version.fingerprint
        assert promoted.state is JobState.PROMOTED
        assert store.get_version_dict(first_version.fingerprint) == first_version.to_dict()

        with pytest.raises(StaleMemoryHeadError):
            store.promote_candidate(
                second_spec.job_key,
                expected_head_generation=0,
                now_unix_ms=1000,
            )
        assert store.get_job(second_spec.job_key).state is JobState.SUPERSEDED

        assert store.pin_session(SCOPE, "session/old", retrieval_cutoff_sequence=1) == old_pin
        new_pin = store.pin_session(SCOPE, "session/new", retrieval_cutoff_sequence=1)
        assert old_pin.version_fingerprint is None
        assert new_pin.version_fingerprint == first_version.fingerprint
        assert new_pin.head_generation == 1

        third_event = observation(
            "event/three",
            0,
            {"text": "third"},
            session_id="session/three",
        )
        third_snapshot = snapshot(
            (third_event,),
            snapshot_id="snapshot/three",
            marker="4",
        )
        third_spec = replace(
            job(third_snapshot, cutoff=2, trainer_marker="f"),
            parent_version_fingerprint=first_version.fingerprint,
        )
        store.append_event(third_event)
        third_job = store.record_rollover(third_snapshot, third_spec, now_unix_ms=1100)
        assert third_job.expected_head_generation == 1
        third_lease = store.lease_next("worker/three", lease_seconds=10, now_unix_ms=1200)
        assert third_lease is not None
        store.start_training(third_lease, now_unix_ms=1300)
        store.start_evaluating(third_lease, now_unix_ms=1400)
        third_version = version(
            third_spec,
            version_id="memory/v2",
            adapter_marker="4",
        )
        store.register_candidate(third_lease, third_version, now_unix_ms=1500)
        later_head, _ = store.promote_candidate(third_spec.job_key, now_unix_ms=1600)
        assert later_head.generation == 2
        assert later_head.version_fingerprint == third_version.fingerprint

        replayed_head, replayed_job = store.promote_candidate(
            first_spec.job_key,
            now_unix_ms=1700,
        )
        assert replayed_job.state is JobState.PROMOTED
        assert replayed_head.generation == 1
        assert replayed_head.version_fingerprint == first_version.fingerprint
        assert store.get_head(SCOPE) == later_head
        assert store.pin_session(SCOPE, "session/new", retrieval_cutoff_sequence=1) == new_pin

        fourth_event = observation(
            "event/four",
            0,
            {"text": "fourth"},
            session_id="session/four",
        )
        fourth_snapshot = snapshot(
            (fourth_event,),
            snapshot_id="snapshot/four",
            marker="3",
        )
        store.append_event(fourth_event)
        regression = replace(
            job(fourth_snapshot, cutoff=1),
            parent_version_fingerprint=third_version.fingerprint,
        )
        with pytest.raises(ValueError, match="cutoff must not move"):
            store.record_rollover(fourth_snapshot, regression, now_unix_ms=1800)

        changed_base = replace(
            job(fourth_snapshot, cutoff=3),
            parent_version_fingerprint=third_version.fingerprint,
            base_fingerprint=base(weights="e").fingerprint,
        )
        with pytest.raises(ValueError, match="changes the frozen base"):
            store.record_rollover(fourth_snapshot, changed_base, now_unix_ms=1900)
        with pytest.raises(KeyError, match="unknown rollover snapshot"):
            store.get_snapshot(fourth_snapshot.fingerprint)
