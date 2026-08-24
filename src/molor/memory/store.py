"""SQLite persistence for the authoritative memory ledger and consolidation queue."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator
from uuid import uuid4

from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256
from molor.manifests import ArtifactDescriptor
from molor.memory.types import (
    ConsolidationJobSpec,
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


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_ACTIVE_JOB_STATES = (JobState.LEASED, JobState.TRAINING, JobState.EVALUATING)
_CAPTURE_CURRENT = object()


class LeaseLostError(RuntimeError):
    """The supplied worker lease is stale, expired, or belongs to another attempt."""


class StaleMemoryHeadError(RuntimeError):
    """A candidate lost compare-and-swap promotion to a newer memory head."""


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One scope-filtered, logically active event returned by the v0 lexical retriever."""

    event: MemoryEvent
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        if not isinstance(self.score, float) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be a float in [0, 1]")


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a valid bounded identifier")
    return value


def _positive_integer(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _now_unix_ms() -> int:
    return time.time_ns() // 1_000_000


def _require_manifest_fields(
    data: Any,
    *,
    schema_version: str,
    fields: set[str],
    manifest_name: str,
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise RuntimeError(f"persisted {manifest_name} is not a JSON object")
    if data.get("schema_version") != schema_version:
        raise RuntimeError(f"persisted {manifest_name} has an unsupported schema version")
    if set(data) != fields:
        raise RuntimeError(f"persisted {manifest_name} has an unexpected field set")
    return data


class SQLiteMemoryStore:
    """Single-node durable ledger, snapshot registry, and leased consolidation queue.

    SQLite/WAL is the v0 coordination boundary.  Raw events remain authoritative; snapshots,
    retrieval hits, and memory LoRAs are derived representations that can be rebuilt.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        if isinstance(path, Path):
            database = str(path)
        elif isinstance(path, str) and path:
            database = path
        else:
            raise ValueError("path must be a non-empty string or Path")
        if database != ":memory:" and Path(database).is_dir():
            raise IsADirectoryError(database)

        self.path = database
        self._connection = sqlite3.connect(database, isolation_level=None, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if database != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
        self._initialize_schema()

    def __enter__(self) -> SQLiteMemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _initialize_schema(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, 1}:
            raise RuntimeError(f"unsupported memory database schema version: {version}")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK (sequence >= 0),
                event_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                target_event_id TEXT,
                target_event_fingerprint TEXT,
                UNIQUE (scope_id, session_id, sequence),
                UNIQUE (scope_id, event_id),
                UNIQUE (fingerprint),
                FOREIGN KEY (scope_id, target_event_id)
                    REFERENCES memory_events (scope_id, event_id)
            );

            CREATE TABLE IF NOT EXISTS rollover_snapshots (
                snapshot_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL UNIQUE,
                snapshot_id TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                first_event_sequence INTEGER NOT NULL,
                last_event_sequence INTEGER NOT NULL,
                manifest_json TEXT NOT NULL,
                UNIQUE (scope_id, snapshot_id),
                UNIQUE (scope_id, session_id, first_event_sequence, last_event_sequence)
            );

            CREATE TABLE IF NOT EXISTS memory_versions (
                fingerprint TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                base_fingerprint TEXT NOT NULL,
                parent_version_fingerprint TEXT,
                consolidated_through_sequence INTEGER NOT NULL,
                adapter_fingerprint TEXT NOT NULL,
                lora_format_fingerprint TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                UNIQUE (scope_id, version_id),
                FOREIGN KEY (parent_version_fingerprint)
                    REFERENCES memory_versions (fingerprint)
            );

            CREATE TABLE IF NOT EXISTS memory_heads (
                scope_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL CHECK (generation >= 0),
                version_fingerprint TEXT,
                consolidated_through_sequence INTEGER,
                FOREIGN KEY (version_fingerprint) REFERENCES memory_versions (fingerprint)
            );

            CREATE TABLE IF NOT EXISTS consolidation_jobs (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_key TEXT NOT NULL UNIQUE,
                scope_id TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                expected_head_generation INTEGER NOT NULL CHECK (expected_head_generation >= 0),
                lease_id TEXT,
                worker_id TEXT,
                lease_issued_at_unix_ms INTEGER,
                lease_expires_at_unix_ms INTEGER,
                candidate_version_fingerprint TEXT,
                reason_code TEXT,
                available_at_unix_ms INTEGER NOT NULL,
                created_at_unix_ms INTEGER NOT NULL,
                updated_at_unix_ms INTEGER NOT NULL,
                FOREIGN KEY (candidate_version_fingerprint)
                    REFERENCES memory_versions (fingerprint)
            );

            CREATE TABLE IF NOT EXISTS session_memory_pins (
                scope_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                head_generation INTEGER NOT NULL,
                version_fingerprint TEXT,
                consolidated_through_sequence INTEGER,
                retrieval_cutoff_sequence INTEGER,
                pin_json TEXT NOT NULL,
                PRIMARY KEY (scope_id, session_id),
                FOREIGN KEY (version_fingerprint) REFERENCES memory_versions (fingerprint)
            );

            CREATE INDEX IF NOT EXISTS memory_events_active_lookup
                ON memory_events (scope_id, target_event_id, ledger_id);
            CREATE INDEX IF NOT EXISTS consolidation_jobs_lease_order
                ON consolidation_jobs (state, available_at_unix_ms, queue_id);
            PRAGMA user_version = 1;
            """
        )

    # -- Authoritative event ledger -------------------------------------------------

    def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append one event, enforcing per-session ordering and idempotent event IDs."""

        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent")
        with self._transaction() as connection:
            return self._append_event(connection, event)

    def append_observation_auto(
        self,
        *,
        event_id: str,
        scope_id: str,
        session_id: str,
        payload: Any,
    ) -> MemoryEvent:
        """Append an observation with a sequence allocated inside the write transaction.

        Serving callers should prefer this method over reading a session tail and then
        constructing ``MemoryEvent`` themselves.  The latter creates a race when two
        requests append to the same conversation concurrently.  This method only
        makes sequence allocation atomic; the caller remains responsible for trusted
        provenance, admission, retention, conflict handling, and deciding whether an
        event is eligible for consolidation.
        """

        _identifier(event_id, "event_id")
        _identifier(scope_id, "scope_id")
        _identifier(session_id, "session_id")
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM memory_events WHERE scope_id = ? AND event_id = ?",
                (scope_id, event_id),
            ).fetchone()
            if existing is not None:
                sequence = existing["sequence"]
            else:
                previous = connection.execute(
                    """
                    SELECT MAX(sequence) AS maximum
                    FROM memory_events
                    WHERE scope_id = ? AND session_id = ?
                    """,
                    (scope_id, session_id),
                ).fetchone()
                sequence = 0 if previous["maximum"] is None else previous["maximum"] + 1
            event = MemoryEvent(
                event_id=event_id,
                scope_id=scope_id,
                session_id=session_id,
                sequence=sequence,
                kind=MemoryEventKind.OBSERVATION,
                payload=payload,
            )
            return self._append_event(connection, event)

    def _append_event(
        self,
        connection: sqlite3.Connection,
        event: MemoryEvent,
    ) -> MemoryEvent:
        existing = connection.execute(
            "SELECT * FROM memory_events WHERE scope_id = ? AND event_id = ?",
            (event.scope_id, event.event_id),
        ).fetchone()
        if existing is not None:
            stored = self._event_from_row(existing)
            if stored.fingerprint != event.fingerprint:
                raise ValueError("event ID already exists with different content")
            return stored

        previous = connection.execute(
            """
            SELECT MAX(sequence) AS maximum
            FROM memory_events
            WHERE scope_id = ? AND session_id = ?
            """,
            (event.scope_id, event.session_id),
        ).fetchone()
        expected_sequence = 0 if previous["maximum"] is None else previous["maximum"] + 1
        if event.sequence != expected_sequence:
            raise ValueError(
                "event sequence must append exactly after the current session tail "
                f"(expected {expected_sequence}, got {event.sequence})"
            )

        if event.kind is not MemoryEventKind.OBSERVATION:
            target = connection.execute(
                "SELECT * FROM memory_events WHERE scope_id = ? AND event_id = ?",
                (event.scope_id, event.target_event_id),
            ).fetchone()
            if target is None:
                raise ValueError("target event is absent from this memory scope")
            event.assert_valid_target(self._event_from_row(target))
            successor = connection.execute(
                """
                SELECT event_id FROM memory_events
                WHERE scope_id = ? AND target_event_id = ?
                LIMIT 1
                """,
                (event.scope_id, event.target_event_id),
            ).fetchone()
            if successor is not None:
                raise ValueError("target event has already been corrected or tombstoned")

        connection.execute(
            """
            INSERT INTO memory_events (
                scope_id, session_id, sequence, event_id, fingerprint, kind,
                payload_json, target_event_id, target_event_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.scope_id,
                event.session_id,
                event.sequence,
                event.event_id,
                event.fingerprint,
                event.kind.value,
                _json_text(event.to_dict()["payload"]),
                event.target_event_id,
                event.target_event_fingerprint,
            ),
        )
        return event

    def get_event(self, scope_id: str, event_id: str) -> MemoryEvent:
        _identifier(scope_id, "scope_id")
        _identifier(event_id, "event_id")
        row = self._connection.execute(
            "SELECT * FROM memory_events WHERE scope_id = ? AND event_id = ?",
            (scope_id, event_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown event in scope {scope_id!r}: {event_id!r}")
        return self._event_from_row(row)

    def list_events(self, scope_id: str, session_id: str) -> tuple[MemoryEvent, ...]:
        _identifier(scope_id, "scope_id")
        _identifier(session_id, "session_id")
        rows = self._connection.execute(
            """
            SELECT * FROM memory_events
            WHERE scope_id = ? AND session_id = ?
            ORDER BY sequence
            """,
            (scope_id, session_id),
        ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def current_retrieval_cutoff(self, scope_id: str) -> int:
        """Return the current database-wide event commit sequence for a scoped read."""

        _identifier(scope_id, "scope_id")
        row = self._connection.execute(
            "SELECT MAX(ledger_id) AS cutoff FROM memory_events",
        ).fetchone()
        return 0 if row["cutoff"] is None else row["cutoff"]

    def active_events(
        self,
        scope_id: str,
        *,
        pin: SessionMemoryPin | None = None,
    ) -> tuple[MemoryEvent, ...]:
        """Resolve content-bearing events at the latest or one pinned retrieval cut."""

        rows = self._active_event_rows(scope_id, pin=pin)
        return tuple(self._event_from_row(row) for row in rows)

    def search_active_events(
        self,
        scope_id: str,
        query: str,
        *,
        limit: int = 8,
        pin: SessionMemoryPin | None = None,
    ) -> tuple[RetrievalHit, ...]:
        """Run a deterministic lexical v0 retrieval after mandatory scope filtering."""

        _identifier(scope_id, "scope_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        _positive_integer(limit, "limit")
        query_tokens = set(_TOKEN_PATTERN.findall(query.casefold()))
        if not query_tokens:
            raise ValueError("query must contain at least one searchable token")

        scored: list[tuple[float, int, MemoryEvent]] = []
        rows = self._active_event_rows(scope_id, pin=pin)
        for row in rows:
            event = self._event_from_row(row)
            payload_tokens = set(_TOKEN_PATTERN.findall(row["payload_json"].casefold()))
            overlap = len(query_tokens.intersection(payload_tokens))
            if overlap:
                scored.append((overlap / len(query_tokens), row["ledger_id"], event))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(
            RetrievalHit(event=event, score=float(score)) for score, _, event in scored[:limit]
        )

    # -- Rollover snapshots ---------------------------------------------------------

    def record_rollover(
        self,
        snapshot: RolloverSnapshotManifest,
        job: ConsolidationJobSpec,
        *,
        now_unix_ms: int | None = None,
    ) -> JobRecord:
        """Atomically record a durable rollover snapshot and its consolidation job."""

        if not isinstance(snapshot, RolloverSnapshotManifest):
            raise TypeError("snapshot must be a RolloverSnapshotManifest")
        if not isinstance(job, ConsolidationJobSpec):
            raise TypeError("job must be a ConsolidationJobSpec")
        if snapshot.scope_id != job.scope_id:
            raise ValueError("snapshot and consolidation job belong to different scopes")
        if snapshot.fingerprint not in job.snapshot_fingerprints:
            raise ValueError("consolidation job does not reference the rollover snapshot")
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )

        with self._transaction() as connection:
            self._insert_snapshot(connection, snapshot)
            return self._insert_job(connection, job, now)

    def get_snapshot(self, fingerprint: str) -> RolloverSnapshotManifest:
        normalized = validate_sha256(fingerprint, "fingerprint")
        row = self._connection.execute(
            "SELECT * FROM rollover_snapshots WHERE fingerprint = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown rollover snapshot: {normalized}")
        return self._snapshot_from_row(row, expected_fingerprint=normalized)

    def enqueue(
        self,
        job: ConsolidationJobSpec,
        *,
        now_unix_ms: int | None = None,
    ) -> JobRecord:
        """Idempotently enqueue a job whose immutable snapshots already exist."""

        if not isinstance(job, ConsolidationJobSpec):
            raise TypeError("job must be a ConsolidationJobSpec")
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )
        with self._transaction() as connection:
            return self._insert_job(connection, job, now)

    # -- At-least-once consolidation queue -----------------------------------------

    def get_job(self, job_key: str) -> JobRecord:
        normalized = validate_sha256(job_key, "job_key")
        row = self._connection.execute(
            "SELECT * FROM consolidation_jobs WHERE job_key = ?", (normalized,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown consolidation job: {normalized}")
        return self._validated_job_from_row(self._connection, row)

    def lease_next(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        now_unix_ms: int | None = None,
    ) -> JobLease | None:
        """Lease one available job; expired active jobs are safely retried."""

        _identifier(worker_id, "worker_id")
        _positive_integer(lease_seconds, "lease_seconds")
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM consolidation_jobs
                WHERE (
                    state IN (?, ?) AND available_at_unix_ms <= ?
                ) OR (
                    state IN (?, ?, ?) AND lease_expires_at_unix_ms <= ?
                )
                ORDER BY queue_id
                LIMIT 1
                """,
                (
                    JobState.QUEUED.value,
                    JobState.RETRYABLE_FAILURE.value,
                    now,
                    JobState.LEASED.value,
                    JobState.TRAINING.value,
                    JobState.EVALUATING.value,
                    now,
                ),
            ).fetchone()
            if row is None:
                return None
            attempt = row["attempts"] + 1
            lease = JobLease(
                job_key=row["job_key"],
                lease_id=uuid4().hex,
                worker_id=worker_id,
                attempt=attempt,
                issued_at_unix_ms=now,
                expires_at_unix_ms=now + lease_seconds * 1000,
            )
            connection.execute(
                """
                UPDATE consolidation_jobs
                SET state = ?, attempts = ?, lease_id = ?, worker_id = ?,
                    lease_issued_at_unix_ms = ?, lease_expires_at_unix_ms = ?,
                    reason_code = NULL, updated_at_unix_ms = ?
                WHERE job_key = ?
                """,
                (
                    JobState.LEASED.value,
                    attempt,
                    lease.lease_id,
                    lease.worker_id,
                    lease.issued_at_unix_ms,
                    lease.expires_at_unix_ms,
                    now,
                    lease.job_key,
                ),
            )
            return lease

    def heartbeat(
        self,
        lease: JobLease,
        *,
        lease_seconds: int,
        now_unix_ms: int | None = None,
    ) -> JobLease:
        _positive_integer(lease_seconds, "lease_seconds")
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )
        with self._transaction() as connection:
            row = self._require_lease(connection, lease, now)
            renewed = JobLease(
                job_key=lease.job_key,
                lease_id=lease.lease_id,
                worker_id=lease.worker_id,
                attempt=lease.attempt,
                issued_at_unix_ms=lease.issued_at_unix_ms,
                expires_at_unix_ms=now + lease_seconds * 1000,
            )
            connection.execute(
                """
                UPDATE consolidation_jobs
                SET lease_expires_at_unix_ms = ?, updated_at_unix_ms = ?
                WHERE job_key = ?
                """,
                (renewed.expires_at_unix_ms, now, row["job_key"]),
            )
            return renewed

    def start_training(self, lease: JobLease, *, now_unix_ms: int | None = None) -> JobRecord:
        return self._active_transition(
            lease,
            expected=JobState.LEASED,
            target=JobState.TRAINING,
            now_unix_ms=now_unix_ms,
        )

    def start_evaluating(self, lease: JobLease, *, now_unix_ms: int | None = None) -> JobRecord:
        return self._active_transition(
            lease,
            expected=JobState.TRAINING,
            target=JobState.EVALUATING,
            now_unix_ms=now_unix_ms,
        )

    def fail(
        self,
        lease: JobLease,
        *,
        retryable: bool,
        reason_code: str,
        retry_after_seconds: int = 0,
        now_unix_ms: int | None = None,
    ) -> JobRecord:
        """Finish an active attempt as a retryable or terminal failure."""

        if not isinstance(retryable, bool):
            raise TypeError("retryable must be boolean")
        _identifier(reason_code, "reason_code")
        _nonnegative_integer(retry_after_seconds, "retry_after_seconds")
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )
        target = JobState.RETRYABLE_FAILURE if retryable else JobState.TERMINAL_FAILURE
        with self._transaction() as connection:
            self._require_lease(connection, lease, now)
            connection.execute(
                """
                UPDATE consolidation_jobs
                SET state = ?, lease_id = NULL, worker_id = NULL,
                    lease_issued_at_unix_ms = NULL, lease_expires_at_unix_ms = NULL,
                    reason_code = ?, available_at_unix_ms = ?, updated_at_unix_ms = ?
                WHERE job_key = ?
                """,
                (
                    target.value,
                    reason_code,
                    now + retry_after_seconds * 1000,
                    now,
                    lease.job_key,
                ),
            )
            return self._job_by_key(connection, lease.job_key)

    # -- Immutable versions, CAS promotion, and session pins -----------------------

    def register_candidate(
        self,
        lease: JobLease,
        version: MemoryVersionManifest,
        *,
        now_unix_ms: int | None = None,
    ) -> JobRecord:
        """Publish evaluated immutable artifacts as a candidate; this does not make them live."""

        if not isinstance(version, MemoryVersionManifest):
            raise TypeError("version must be a MemoryVersionManifest")
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT * FROM consolidation_jobs WHERE job_key = ?", (lease.job_key,)
            ).fetchone()
            if current is None:
                raise LeaseLostError("consolidation job no longer exists")
            if (
                current["state"] == JobState.CANDIDATE.value
                and current["candidate_version_fingerprint"] == version.fingerprint
            ):
                return self._validated_job_from_row(connection, current)

            row = self._require_lease(connection, lease, now)
            if JobState(row["state"]) is not JobState.EVALUATING:
                raise ValueError("only an evaluating job may publish a candidate")
            spec = self._spec_from_json(row["spec_json"])
            self._validate_candidate(connection, spec, version)
            manifest_json = _json_text(version.to_dict())
            existing_version = connection.execute(
                "SELECT manifest_json FROM memory_versions WHERE fingerprint = ?",
                (version.fingerprint,),
            ).fetchone()
            if existing_version is not None:
                if existing_version["manifest_json"] != manifest_json:
                    raise ValueError("memory version fingerprint collision")
            else:
                connection.execute(
                    """
                    INSERT INTO memory_versions (
                        fingerprint, version_id, scope_id, base_fingerprint,
                        parent_version_fingerprint, consolidated_through_sequence,
                        adapter_fingerprint, lora_format_fingerprint, manifest_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version.fingerprint,
                        version.version_id,
                        version.scope_id,
                        version.base_fingerprint,
                        version.parent_version_fingerprint,
                        version.consolidated_through_sequence,
                        version.adapter.fingerprint,
                        version.adapter.lora.fingerprint,
                        manifest_json,
                    ),
                )
            connection.execute(
                """
                UPDATE consolidation_jobs
                SET state = ?, lease_id = NULL, worker_id = NULL,
                    lease_issued_at_unix_ms = NULL, lease_expires_at_unix_ms = NULL,
                    candidate_version_fingerprint = ?, updated_at_unix_ms = ?
                WHERE job_key = ?
                """,
                (JobState.CANDIDATE.value, version.fingerprint, now, lease.job_key),
            )
            return self._job_by_key(connection, lease.job_key)

    def promote_candidate(
        self,
        job_key: str,
        *,
        expected_head_generation: int | None = None,
        now_unix_ms: int | None = None,
    ) -> tuple[MemoryHead, JobRecord]:
        """Atomically promote a candidate if the scope head still matches its pinned parent."""

        normalized = validate_sha256(job_key, "job_key")
        if expected_head_generation is not None:
            _nonnegative_integer(expected_head_generation, "expected_head_generation")
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )
        stale = False
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM consolidation_jobs WHERE job_key = ?", (normalized,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown consolidation job: {normalized}")
            persisted_job = self._validated_job_from_row(connection, row)
            stored_generation = row["expected_head_generation"]
            if expected_head_generation is None:
                expected_head_generation = stored_generation
            elif expected_head_generation != stored_generation:
                raise ValueError("expected head generation differs from the enqueued job")
            if JobState(row["state"]) is JobState.PROMOTED:
                version = connection.execute(
                    "SELECT * FROM memory_versions WHERE fingerprint = ?",
                    (row["candidate_version_fingerprint"],),
                ).fetchone()
                if version is None:
                    raise RuntimeError("promoted memory version is missing")
                self._version_dict_from_row(
                    version,
                    expected_fingerprint=row["candidate_version_fingerprint"],
                )
                promoted_head = MemoryHead(
                    scope_id=row["scope_id"],
                    generation=stored_generation + 1,
                    version_fingerprint=version["fingerprint"],
                    consolidated_through_sequence=version["consolidated_through_sequence"],
                )
                return promoted_head, persisted_job
            if JobState(row["state"]) is not JobState.CANDIDATE:
                raise ValueError("only a candidate job may be promoted")

            head = self._head_from_db(connection, row["scope_id"])
            spec = persisted_job.spec
            if (
                head.generation != expected_head_generation
                or head.version_fingerprint != spec.parent_version_fingerprint
            ):
                stale = True
                connection.execute(
                    """
                    UPDATE consolidation_jobs
                    SET state = ?, reason_code = ?, updated_at_unix_ms = ?
                    WHERE job_key = ?
                    """,
                    (JobState.SUPERSEDED.value, "stale_head", now, normalized),
                )
                result_head = head
                result_job = self._job_by_key(connection, normalized)
            else:
                version = connection.execute(
                    """
                    SELECT * FROM memory_versions
                    WHERE fingerprint = ?
                    """,
                    (row["candidate_version_fingerprint"],),
                ).fetchone()
                if version is None:
                    raise RuntimeError("candidate memory version is missing")
                self._version_dict_from_row(
                    version,
                    expected_fingerprint=row["candidate_version_fingerprint"],
                )
                result_head = MemoryHead(
                    scope_id=head.scope_id,
                    generation=head.generation + 1,
                    version_fingerprint=version["fingerprint"],
                    consolidated_through_sequence=version["consolidated_through_sequence"],
                )
                connection.execute(
                    """
                    INSERT INTO memory_heads (
                        scope_id, generation, version_fingerprint,
                        consolidated_through_sequence
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(scope_id) DO UPDATE SET
                        generation = excluded.generation,
                        version_fingerprint = excluded.version_fingerprint,
                        consolidated_through_sequence = excluded.consolidated_through_sequence
                    """,
                    (
                        result_head.scope_id,
                        result_head.generation,
                        result_head.version_fingerprint,
                        result_head.consolidated_through_sequence,
                    ),
                )
                connection.execute(
                    """
                    UPDATE consolidation_jobs
                    SET state = ?, updated_at_unix_ms = ?
                    WHERE job_key = ?
                    """,
                    (JobState.PROMOTED.value, now, normalized),
                )
                result_job = self._job_by_key(connection, normalized)
        if stale:
            raise StaleMemoryHeadError(
                "memory head changed after this consolidation job pinned its parent"
            )
        return result_head, result_job

    def get_head(self, scope_id: str) -> MemoryHead:
        _identifier(scope_id, "scope_id")
        return self._head_from_db(self._connection, scope_id)

    def pin_session(
        self,
        scope_id: str,
        session_id: str,
        *,
        retrieval_cutoff_sequence: int | None | object = _CAPTURE_CURRENT,
    ) -> SessionMemoryPin:
        """Atomically pin one memory head and an as-of retrieval commit sequence."""

        _identifier(scope_id, "scope_id")
        _identifier(session_id, "session_id")
        capture_current = retrieval_cutoff_sequence is _CAPTURE_CURRENT
        if not capture_current and retrieval_cutoff_sequence is not None:
            _nonnegative_integer(retrieval_cutoff_sequence, "retrieval_cutoff_sequence")
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM session_memory_pins
                WHERE scope_id = ? AND session_id = ?
                """,
                (scope_id, session_id),
            ).fetchone()
            if existing is not None:
                pin = self._pin_from_row(existing)
                if capture_current:
                    return pin
                if pin.retrieval_cutoff_sequence != retrieval_cutoff_sequence:
                    raise ValueError("session is already pinned to a different retrieval cutoff")
                return pin
            frontier_row = connection.execute(
                "SELECT MAX(ledger_id) AS cutoff FROM memory_events"
            ).fetchone()
            frontier = 0 if frontier_row["cutoff"] is None else frontier_row["cutoff"]
            if capture_current:
                retrieval_cutoff_sequence = frontier
            elif retrieval_cutoff_sequence is not None and retrieval_cutoff_sequence > frontier:
                raise ValueError("retrieval cutoff is beyond the current event ledger frontier")
            head = self._head_from_db(connection, scope_id)
            pin = SessionMemoryPin.from_head(
                session_id,
                head,
                retrieval_cutoff_sequence=retrieval_cutoff_sequence,
            )
            connection.execute(
                """
                INSERT INTO session_memory_pins (
                    scope_id, session_id, head_generation, version_fingerprint,
                    consolidated_through_sequence, retrieval_cutoff_sequence, pin_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pin.scope_id,
                    pin.session_id,
                    pin.head_generation,
                    pin.version_fingerprint,
                    pin.consolidated_through_sequence,
                    pin.retrieval_cutoff_sequence,
                    _json_text(pin.to_dict()),
                ),
            )
            return pin

    def get_version_dict(self, fingerprint: str) -> dict[str, Any]:
        """Return a persisted immutable manifest for handoff to an artifact registry/loader."""

        normalized = validate_sha256(fingerprint, "fingerprint")
        row = self._connection.execute(
            "SELECT * FROM memory_versions WHERE fingerprint = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown memory version: {normalized}")
        return self._version_dict_from_row(row, expected_fingerprint=normalized)

    # -- Internal serialization and validation -------------------------------------

    @staticmethod
    def _version_dict_from_row(
        row: sqlite3.Row,
        *,
        expected_fingerprint: str,
    ) -> dict[str, Any]:
        data = _require_manifest_fields(
            json.loads(row["manifest_json"]),
            schema_version=MemoryVersionManifest.schema_version,
            fields={
                "schema_version",
                "version_id",
                "scope_id",
                "base_fingerprint",
                "parent_version_fingerprint",
                "consolidated_through_sequence",
                "consolidation_job_fingerprint",
                "corpus_manifest_fingerprint",
                "adapter_fingerprint",
                "adapter",
                "trainer_fingerprint",
                "evaluation_report_fingerprint",
                "policy_fingerprint",
            },
            manifest_name="memory version",
        )
        if (
            row["fingerprint"] != expected_fingerprint
            or fingerprint_json(data) != expected_fingerprint
        ):
            raise RuntimeError("persisted memory version does not match its content identity")
        adapter_data = data["adapter"]
        if not isinstance(adapter_data, dict):
            raise RuntimeError("persisted memory adapter manifest is not a JSON object")
        if fingerprint_json(adapter_data) != data["adapter_fingerprint"]:
            raise RuntimeError("persisted memory adapter does not match its declared fingerprint")
        if adapter_data.get("base_fingerprint") != data["base_fingerprint"]:
            raise RuntimeError("persisted memory adapter targets a different base")
        lora_data = adapter_data.get("lora")
        if (
            not isinstance(lora_data, dict)
            or fingerprint_json(lora_data) != row["lora_format_fingerprint"]
        ):
            raise RuntimeError("persisted memory adapter LoRA format identity is inconsistent")
        flat_identity = (
            row["version_id"],
            row["scope_id"],
            row["base_fingerprint"],
            row["parent_version_fingerprint"],
            row["consolidated_through_sequence"],
            row["adapter_fingerprint"],
        )
        if flat_identity != (
            data["version_id"],
            data["scope_id"],
            data["base_fingerprint"],
            data["parent_version_fingerprint"],
            data["consolidated_through_sequence"],
            data["adapter_fingerprint"],
        ):
            raise RuntimeError("persisted memory version columns disagree with its manifest")
        return data

    def _active_event_rows(
        self,
        scope_id: str,
        *,
        pin: SessionMemoryPin | None,
    ) -> list[sqlite3.Row]:
        _identifier(scope_id, "scope_id")
        if pin is None:
            cutoff = 2**63 - 1
        else:
            if not isinstance(pin, SessionMemoryPin):
                raise TypeError("pin must be a SessionMemoryPin or None")
            if pin.scope_id != scope_id:
                raise ValueError("retrieval pin belongs to a different memory scope")
            persisted = self._connection.execute(
                """
                SELECT * FROM session_memory_pins
                WHERE scope_id = ? AND session_id = ?
                """,
                (pin.scope_id, pin.session_id),
            ).fetchone()
            if persisted is None:
                raise ValueError("retrieval pin was not issued by this memory store")
            issued_pin = self._pin_from_row(persisted)
            if issued_pin.fingerprint != pin.fingerprint:
                raise ValueError("retrieval pin differs from the store-issued session pin")
            if pin.retrieval_cutoff_sequence is None:
                return []
            cutoff = pin.retrieval_cutoff_sequence
        return self._connection.execute(
            """
            SELECT event.*
            FROM memory_events AS event
            WHERE event.scope_id = ?
              AND event.ledger_id <= ?
              AND event.kind != ?
              AND NOT EXISTS (
                  SELECT 1 FROM memory_events AS successor
                  WHERE successor.scope_id = event.scope_id
                    AND successor.target_event_id = event.event_id
                    AND successor.ledger_id <= ?
              )
            ORDER BY event.ledger_id
            """,
            (scope_id, cutoff, MemoryEventKind.TOMBSTONE.value, cutoff),
        ).fetchall()

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> MemoryEvent:
        event = MemoryEvent(
            event_id=row["event_id"],
            scope_id=row["scope_id"],
            session_id=row["session_id"],
            sequence=row["sequence"],
            kind=MemoryEventKind(row["kind"]),
            payload=json.loads(row["payload_json"]),
            target_event_id=row["target_event_id"],
            target_event_fingerprint=row["target_event_fingerprint"],
        )
        if event.fingerprint != row["fingerprint"]:
            raise RuntimeError("persisted memory event does not match its content identity")
        return event

    @staticmethod
    def _spec_from_json(value: str) -> ConsolidationJobSpec:
        data = _require_manifest_fields(
            json.loads(value),
            schema_version=ConsolidationJobSpec.schema_version,
            fields={
                "schema_version",
                "scope_id",
                "base_fingerprint",
                "parent_version_fingerprint",
                "snapshot_fingerprints",
                "item_cutoff_sequence",
                "dataset_builder_fingerprint",
                "trainer_fingerprint",
                "evaluation_suite_fingerprint",
                "target_lora_format_fingerprint",
                "policy_fingerprint",
            },
            manifest_name="consolidation job",
        )
        return ConsolidationJobSpec(
            scope_id=data["scope_id"],
            base_fingerprint=data["base_fingerprint"],
            parent_version_fingerprint=data["parent_version_fingerprint"],
            snapshot_fingerprints=tuple(data["snapshot_fingerprints"]),
            item_cutoff_sequence=data["item_cutoff_sequence"],
            dataset_builder_fingerprint=data["dataset_builder_fingerprint"],
            trainer_fingerprint=data["trainer_fingerprint"],
            evaluation_suite_fingerprint=data["evaluation_suite_fingerprint"],
            target_lora_format_fingerprint=data["target_lora_format_fingerprint"],
            policy_fingerprint=data["policy_fingerprint"],
        )

    @staticmethod
    def _snapshot_from_json(value: str) -> RolloverSnapshotManifest:
        data = _require_manifest_fields(
            json.loads(value),
            schema_version=RolloverSnapshotManifest.schema_version,
            fields={
                "schema_version",
                "snapshot_id",
                "scope_id",
                "session_id",
                "first_event_sequence",
                "last_event_sequence",
                "event_fingerprints",
                "summary",
                "rollover_policy_fingerprint",
            },
            manifest_name="rollover snapshot",
        )
        summary = data["summary"]
        if not isinstance(summary, dict) or set(summary) != {
            "logical_name",
            "media_type",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("persisted rollover summary has an unexpected field set")
        return RolloverSnapshotManifest(
            snapshot_id=data["snapshot_id"],
            scope_id=data["scope_id"],
            session_id=data["session_id"],
            first_event_sequence=data["first_event_sequence"],
            last_event_sequence=data["last_event_sequence"],
            event_fingerprints=tuple(data["event_fingerprints"]),
            summary=ArtifactDescriptor(
                logical_name=summary["logical_name"],
                media_type=summary["media_type"],
                sha256=summary["sha256"],
                size_bytes=summary["size_bytes"],
            ),
            rollover_policy_fingerprint=data["rollover_policy_fingerprint"],
        )

    @staticmethod
    def _snapshot_from_row(
        row: sqlite3.Row,
        *,
        expected_fingerprint: str,
    ) -> RolloverSnapshotManifest:
        snapshot = SQLiteMemoryStore._snapshot_from_json(row["manifest_json"])
        if (
            row["fingerprint"] != expected_fingerprint
            or snapshot.fingerprint != expected_fingerprint
        ):
            raise RuntimeError("persisted rollover snapshot does not match its content identity")
        if (
            row["snapshot_id"],
            row["scope_id"],
            row["session_id"],
            row["first_event_sequence"],
            row["last_event_sequence"],
        ) != (
            snapshot.snapshot_id,
            snapshot.scope_id,
            snapshot.session_id,
            snapshot.first_event_sequence,
            snapshot.last_event_sequence,
        ):
            raise RuntimeError("persisted rollover snapshot columns disagree with its manifest")
        return snapshot

    @staticmethod
    def _pin_from_json(value: str) -> SessionMemoryPin:
        data = _require_manifest_fields(
            json.loads(value),
            schema_version=SessionMemoryPin.schema_version,
            fields={
                "schema_version",
                "session_id",
                "scope_id",
                "head_generation",
                "version_fingerprint",
                "consolidated_through_sequence",
                "retrieval_cutoff_sequence",
            },
            manifest_name="session memory pin",
        )
        return SessionMemoryPin(
            session_id=data["session_id"],
            scope_id=data["scope_id"],
            head_generation=data["head_generation"],
            version_fingerprint=data["version_fingerprint"],
            consolidated_through_sequence=data["consolidated_through_sequence"],
            retrieval_cutoff_sequence=data["retrieval_cutoff_sequence"],
        )

    @staticmethod
    def _pin_from_row(row: sqlite3.Row) -> SessionMemoryPin:
        pin = SQLiteMemoryStore._pin_from_json(row["pin_json"])
        persisted_identity = (
            row["scope_id"],
            row["session_id"],
            row["head_generation"],
            row["version_fingerprint"],
            row["consolidated_through_sequence"],
            row["retrieval_cutoff_sequence"],
        )
        if persisted_identity != (
            pin.scope_id,
            pin.session_id,
            pin.head_generation,
            pin.version_fingerprint,
            pin.consolidated_through_sequence,
            pin.retrieval_cutoff_sequence,
        ):
            raise RuntimeError("persisted session pin columns disagree with its manifest")
        return pin

    def _insert_snapshot(
        self,
        connection: sqlite3.Connection,
        snapshot: RolloverSnapshotManifest,
    ) -> None:
        existing = connection.execute(
            """
            SELECT * FROM rollover_snapshots
            WHERE scope_id = ? AND snapshot_id = ?
            """,
            (snapshot.scope_id, snapshot.snapshot_id),
        ).fetchone()
        if existing is not None:
            persisted = self._snapshot_from_row(
                existing,
                expected_fingerprint=existing["fingerprint"],
            )
            if persisted.fingerprint != snapshot.fingerprint:
                raise ValueError("snapshot ID already exists with different content")
            return

        rows = connection.execute(
            """
            SELECT * FROM memory_events
            WHERE scope_id = ? AND session_id = ?
              AND sequence BETWEEN ? AND ?
            ORDER BY sequence
            """,
            (
                snapshot.scope_id,
                snapshot.session_id,
                snapshot.first_event_sequence,
                snapshot.last_event_sequence,
            ),
        ).fetchall()
        fingerprints = tuple(row["fingerprint"] for row in rows)
        if fingerprints != snapshot.event_fingerprints:
            raise ValueError("snapshot event range does not match the durable event ledger")
        overlap = connection.execute(
            """
            SELECT snapshot_id FROM rollover_snapshots
            WHERE scope_id = ? AND session_id = ?
              AND NOT (last_event_sequence < ? OR first_event_sequence > ?)
            LIMIT 1
            """,
            (
                snapshot.scope_id,
                snapshot.session_id,
                snapshot.first_event_sequence,
                snapshot.last_event_sequence,
            ),
        ).fetchone()
        if overlap is not None:
            raise ValueError("rollover snapshot overlaps an existing snapshot")
        connection.execute(
            """
            INSERT INTO rollover_snapshots (
                fingerprint, snapshot_id, scope_id, session_id,
                first_event_sequence, last_event_sequence, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.fingerprint,
                snapshot.snapshot_id,
                snapshot.scope_id,
                snapshot.session_id,
                snapshot.first_event_sequence,
                snapshot.last_event_sequence,
                _json_text(snapshot.to_dict()),
            ),
        )

    def _insert_job(
        self,
        connection: sqlite3.Connection,
        job: ConsolidationJobSpec,
        now: int,
    ) -> JobRecord:
        existing = connection.execute(
            "SELECT * FROM consolidation_jobs WHERE job_key = ?", (job.job_key,)
        ).fetchone()
        if existing is not None:
            if existing["spec_json"] != _json_text(job.to_dict()):
                raise ValueError("consolidation job key collision")
            return self._validated_job_from_row(connection, existing)

        placeholders = ",".join("?" for _ in job.snapshot_fingerprints)
        snapshots = connection.execute(
            f"""
            SELECT * FROM rollover_snapshots
            WHERE fingerprint IN ({placeholders})
            """,  # noqa: S608 - placeholders are generated only from a fixed count
            job.snapshot_fingerprints,
        ).fetchall()
        persisted_snapshots = tuple(
            self._snapshot_from_row(
                row,
                expected_fingerprint=row["fingerprint"],
            )
            for row in snapshots
        )
        found = {snapshot.fingerprint for snapshot in persisted_snapshots}
        if found != set(job.snapshot_fingerprints):
            raise ValueError("consolidation job references an unknown rollover snapshot")
        if any(snapshot.scope_id != job.scope_id for snapshot in persisted_snapshots):
            raise ValueError("consolidation job references a snapshot from another scope")

        head = self._head_from_db(connection, job.scope_id)
        if job.parent_version_fingerprint != head.version_fingerprint:
            raise ValueError("consolidation job parent is not the current memory head")
        if (
            head.consolidated_through_sequence is not None
            and job.item_cutoff_sequence < head.consolidated_through_sequence
        ):
            raise ValueError(
                "consolidation job cutoff must not move behind the current memory head"
            )
        if head.version_fingerprint is not None:
            parent = connection.execute(
                "SELECT base_fingerprint FROM memory_versions WHERE fingerprint = ?",
                (head.version_fingerprint,),
            ).fetchone()
            if parent is None or parent["base_fingerprint"] != job.base_fingerprint:
                raise ValueError("consolidation job changes the frozen base")
        connection.execute(
            """
            INSERT INTO consolidation_jobs (
                job_key, scope_id, spec_json, state, attempts,
                expected_head_generation, available_at_unix_ms,
                created_at_unix_ms, updated_at_unix_ms
            ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                job.job_key,
                job.scope_id,
                _json_text(job.to_dict()),
                JobState.QUEUED.value,
                head.generation,
                now,
                now,
                now,
            ),
        )
        return self._job_by_key(connection, job.job_key)

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        spec = SQLiteMemoryStore._spec_from_json(row["spec_json"])
        if spec.job_key != row["job_key"]:
            raise RuntimeError("persisted consolidation job does not match its content identity")
        if spec.scope_id != row["scope_id"]:
            raise RuntimeError("persisted consolidation job scope disagrees with its manifest")
        state = JobState(row["state"])
        lease = None
        if state in _ACTIVE_JOB_STATES:
            lease = JobLease(
                job_key=row["job_key"],
                lease_id=row["lease_id"],
                worker_id=row["worker_id"],
                attempt=row["attempts"],
                issued_at_unix_ms=row["lease_issued_at_unix_ms"],
                expires_at_unix_ms=row["lease_expires_at_unix_ms"],
            )
        return JobRecord(
            spec=spec,
            state=state,
            attempts=row["attempts"],
            lease=lease,
            candidate_version_fingerprint=row["candidate_version_fingerprint"],
            reason_code=row["reason_code"],
            expected_head_generation=row["expected_head_generation"],
        )

    def _job_by_key(self, connection: sqlite3.Connection, job_key: str) -> JobRecord:
        row = connection.execute(
            "SELECT * FROM consolidation_jobs WHERE job_key = ?", (job_key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown consolidation job: {job_key}")
        return self._validated_job_from_row(connection, row)

    def _validated_job_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> JobRecord:
        job = self._job_from_row(row)
        candidate_fingerprint = job.candidate_version_fingerprint
        if candidate_fingerprint is None:
            return job

        version_row = connection.execute(
            "SELECT * FROM memory_versions WHERE fingerprint = ?",
            (candidate_fingerprint,),
        ).fetchone()
        if version_row is None:
            raise RuntimeError("persisted consolidation job candidate version is missing")
        version = self._version_dict_from_row(
            version_row,
            expected_fingerprint=candidate_fingerprint,
        )
        candidate_binding = (
            version["scope_id"],
            version["base_fingerprint"],
            version["parent_version_fingerprint"],
            version["consolidated_through_sequence"],
            version["consolidation_job_fingerprint"],
            version["trainer_fingerprint"],
            version_row["lora_format_fingerprint"],
            version["policy_fingerprint"],
        )
        expected_binding = (
            job.spec.scope_id,
            job.spec.base_fingerprint,
            job.spec.parent_version_fingerprint,
            job.spec.item_cutoff_sequence,
            job.spec.job_key,
            job.spec.trainer_fingerprint,
            job.spec.target_lora_format_fingerprint,
            job.spec.policy_fingerprint,
        )
        if candidate_binding != expected_binding:
            raise RuntimeError(
                "persisted consolidation job candidate disagrees with its job manifest"
            )
        parent_fingerprint = version["parent_version_fingerprint"]
        if parent_fingerprint is not None:
            parent_row = connection.execute(
                "SELECT * FROM memory_versions WHERE fingerprint = ?",
                (parent_fingerprint,),
            ).fetchone()
            if parent_row is None:
                raise RuntimeError("persisted candidate parent memory version is missing")
            self._version_dict_from_row(
                parent_row,
                expected_fingerprint=parent_fingerprint,
            )
            if parent_row["base_fingerprint"] != version["base_fingerprint"]:
                raise RuntimeError("persisted candidate changes the frozen base across versions")
        return job

    def _require_lease(
        self,
        connection: sqlite3.Connection,
        lease: JobLease,
        now: int,
    ) -> sqlite3.Row:
        if not isinstance(lease, JobLease):
            raise TypeError("lease must be a JobLease")
        row = connection.execute(
            "SELECT * FROM consolidation_jobs WHERE job_key = ?", (lease.job_key,)
        ).fetchone()
        if row is None or JobState(row["state"]) not in _ACTIVE_JOB_STATES:
            raise LeaseLostError("job is not held by an active worker lease")
        identity = (
            row["lease_id"],
            row["worker_id"],
            row["attempts"],
            row["lease_issued_at_unix_ms"],
        )
        supplied = (
            lease.lease_id,
            lease.worker_id,
            lease.attempt,
            lease.issued_at_unix_ms,
        )
        if identity != supplied or row["lease_expires_at_unix_ms"] <= now:
            raise LeaseLostError("worker lease is stale or expired")
        return row

    def _active_transition(
        self,
        lease: JobLease,
        *,
        expected: JobState,
        target: JobState,
        now_unix_ms: int | None,
    ) -> JobRecord:
        now = (
            _now_unix_ms()
            if now_unix_ms is None
            else _nonnegative_integer(now_unix_ms, "now_unix_ms")
        )
        with self._transaction() as connection:
            row = self._require_lease(connection, lease, now)
            if JobState(row["state"]) is not expected:
                raise ValueError(f"job must be {expected.value} before entering {target.value}")
            connection.execute(
                """
                UPDATE consolidation_jobs
                SET state = ?, updated_at_unix_ms = ?
                WHERE job_key = ?
                """,
                (target.value, now, lease.job_key),
            )
            return self._job_by_key(connection, lease.job_key)

    def _validate_candidate(
        self,
        connection: sqlite3.Connection,
        spec: ConsolidationJobSpec,
        version: MemoryVersionManifest,
    ) -> None:
        if version.scope_id != spec.scope_id:
            raise ValueError("candidate memory version belongs to a different scope")
        if version.base_fingerprint != spec.base_fingerprint:
            raise ValueError("candidate frozen base differs from the consolidation job")
        if version.parent_version_fingerprint != spec.parent_version_fingerprint:
            raise ValueError("candidate parent differs from the consolidation job")
        if version.consolidated_through_sequence != spec.item_cutoff_sequence:
            raise ValueError("candidate consolidation cutoff differs from the job")
        if version.consolidation_job_fingerprint != spec.job_key:
            raise ValueError("candidate is not bound to this exact consolidation job")
        if version.trainer_fingerprint != spec.trainer_fingerprint:
            raise ValueError("candidate trainer differs from the consolidation job")
        if version.adapter.lora.fingerprint != spec.target_lora_format_fingerprint:
            raise ValueError("candidate LoRA format differs from the consolidation job")
        if version.policy_fingerprint != spec.policy_fingerprint:
            raise ValueError("candidate policy differs from the consolidation job")
        if version.parent_version_fingerprint is not None:
            parent = connection.execute(
                "SELECT base_fingerprint FROM memory_versions WHERE fingerprint = ?",
                (version.parent_version_fingerprint,),
            ).fetchone()
            if parent is None:
                raise ValueError("candidate parent memory version is unknown")
            if parent["base_fingerprint"] != version.base_fingerprint:
                raise ValueError("candidate changes the frozen base across memory versions")

    @staticmethod
    def _head_from_db(connection: sqlite3.Connection, scope_id: str) -> MemoryHead:
        row = connection.execute(
            "SELECT * FROM memory_heads WHERE scope_id = ?", (scope_id,)
        ).fetchone()
        if row is None:
            return MemoryHead.empty(scope_id)
        head = MemoryHead(
            scope_id=row["scope_id"],
            generation=row["generation"],
            version_fingerprint=row["version_fingerprint"],
            consolidated_through_sequence=row["consolidated_through_sequence"],
        )
        if head.version_fingerprint is not None:
            version = connection.execute(
                "SELECT * FROM memory_versions WHERE fingerprint = ?",
                (head.version_fingerprint,),
            ).fetchone()
            if version is None:
                raise RuntimeError("active memory head references a missing version")
            SQLiteMemoryStore._version_dict_from_row(
                version,
                expected_fingerprint=head.version_fingerprint,
            )
            if (
                version["scope_id"] != head.scope_id
                or version["consolidated_through_sequence"] != head.consolidated_through_sequence
            ):
                raise RuntimeError("active memory head disagrees with its version manifest")
        return head
