"""A small SQLite store for generations, scalar feedback, and pairwise choices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid

from molor.feedback.models import (
    ExportResult,
    GenerationCreate,
    GenerationRecord,
    PreferenceRecord,
    TeacherAnnotationRecord,
)

DEFAULT_FEEDBACK_DB = Path("runs/feedback/feedback.sqlite3")
DEFAULT_EXPORT_DIR = Path("runs/feedback/exports")
_SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: datetime | None) -> str:
    if value is None:
        return _now()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(value: Mapping[str, object], field_name: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized = dict(value)
    if any(not isinstance(key, str) or not key for key in normalized):
        raise ValueError(f"{field_name} keys must be non-empty strings")
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain finite JSON values") from exc
    return encoded, json.loads(encoded)


def _messages(value: Sequence[Mapping[str, str]]) -> tuple[str, tuple[dict[str, str], ...]]:
    if isinstance(value, (str, bytes)) or not value:
        raise ValueError("messages must be a non-empty sequence")
    normalized: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("each message needs exactly role and content")
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "user":
        raise ValueError("generation messages must end with the user prompt")
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return encoded, tuple(normalized)


def _score(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number from -2 to 2")
    score = float(value)
    if not math.isfinite(score) or not -2 <= score <= 2:
        raise ValueError(f"{field_name} must be a number from -2 to 2")
    return score


def _confidence(value: float, field_name: str = "confidence") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number from 0 to 1")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"{field_name} must be a number from 0 to 1")
    return confidence


class FeedbackStore:
    """Thread-safe local feedback storage with direct training-data exports."""

    def __init__(self, path: str | Path = DEFAULT_FEEDBACK_DB) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            if str(path) != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._migrate()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> FeedbackStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _SCHEMA_VERSION:
            raise RuntimeError(f"feedback database schema {version} is newer than this build")
        if version == 0:
            self._connection.executescript(
                """
                CREATE TABLE generations (
                    generation_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    graded_at TEXT,
                    prompt TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    response TEXT NOT NULL,
                    route_json TEXT NOT NULL,
                    generation_settings_json TEXT NOT NULL,
                    reasoning_settings_json TEXT NOT NULL,
                    context_settings_json TEXT NOT NULL,
                    cache_settings_json TEXT NOT NULL,
                    telemetry_json TEXT NOT NULL,
                    system_instruction TEXT,
                    reward REAL CHECK (reward IS NULL OR reward BETWEEN -2 AND 2),
                    dimensions_json TEXT NOT NULL DEFAULT '{}',
                    notes TEXT NOT NULL DEFAULT '',
                    corrected_answer TEXT
                );
                CREATE INDEX generations_created_at_idx ON generations(created_at DESC);
                CREATE INDEX generations_conversation_idx
                    ON generations(conversation_id, created_at DESC);
                CREATE TABLE preferences (
                    preference_id TEXT PRIMARY KEY,
                    chosen_generation_id TEXT NOT NULL REFERENCES generations(generation_id),
                    rejected_generation_id TEXT NOT NULL REFERENCES generations(generation_id),
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    CHECK (chosen_generation_id != rejected_generation_id)
                );
                CREATE INDEX preferences_created_at_idx ON preferences(created_at DESC);
                PRAGMA user_version = 1;
                """
            )
            self._connection.commit()
            version = 1
        if version == 1:
            self._connection.executescript(
                """
                ALTER TABLE preferences
                    ADD COLUMN evaluator_id TEXT NOT NULL DEFAULT 'human';
                ALTER TABLE preferences
                    ADD COLUMN evaluator_model TEXT;
                ALTER TABLE preferences
                    ADD COLUMN confidence REAL CHECK (
                        confidence IS NULL OR confidence BETWEEN 0 AND 1
                    );
                ALTER TABLE preferences
                    ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
                CREATE TABLE teacher_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    generation_id TEXT NOT NULL REFERENCES generations(generation_id),
                    teacher_id TEXT NOT NULL,
                    teacher_model TEXT,
                    created_at TEXT NOT NULL,
                    overall_score REAL NOT NULL CHECK (overall_score BETWEEN -2 AND 2),
                    dimensions_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    critique TEXT NOT NULL DEFAULT '',
                    corrected_answer TEXT,
                    rubric_version TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX teacher_annotations_generation_idx
                    ON teacher_annotations(generation_id, created_at DESC);
                CREATE INDEX teacher_annotations_teacher_idx
                    ON teacher_annotations(teacher_id, created_at DESC);
                PRAGMA user_version = 2;
                """
            )
            self._connection.commit()

    def record_generation(self, create: GenerationCreate) -> str:
        if not isinstance(create, GenerationCreate):
            raise TypeError("create must be a GenerationCreate")
        if not create.conversation_id.strip():
            raise ValueError("conversation_id is required")
        if not create.model_id.strip():
            raise ValueError("model_id is required")
        if not create.response.strip():
            raise ValueError("response is required")
        messages_json, messages = _messages(create.messages)
        route_json, route = _json_object(create.route, "route")
        for name, weight in route.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"route weight {name!r} must be finite and non-negative")
            if not math.isfinite(float(weight)) or float(weight) < 0:
                raise ValueError(f"route weight {name!r} must be finite and non-negative")
        settings_json, _ = _json_object(create.settings, "settings")
        telemetry_json, _ = _json_object(create.telemetry, "telemetry")
        context_json, _ = _json_object(create.context_policy, "context_policy")
        cache_json, _ = _json_object(create.cache_policy, "cache_policy")
        reasoning_json, _ = _json_object(
            {
                "enable_thinking": bool(create.enable_thinking),
                "preset": create.reasoning_preset,
            },
            "reasoning_settings",
        )
        if not isinstance(create.enable_thinking, bool):
            raise TypeError("enable_thinking must be boolean")
        if not isinstance(create.reasoning_preset, str) or not create.reasoning_preset:
            raise ValueError("reasoning_preset must be a non-empty string")
        if create.system_instruction is not None and not isinstance(create.system_instruction, str):
            raise TypeError("system_instruction must be a string or None")

        generation_id = uuid.uuid4().hex
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO generations (
                    generation_id, conversation_id, created_at, prompt, model_id,
                    messages_json, response, route_json, generation_settings_json,
                    reasoning_settings_json, context_settings_json, cache_settings_json,
                    telemetry_json, system_instruction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    create.conversation_id.strip(),
                    _timestamp(create.created_at),
                    messages[-1]["content"],
                    create.model_id.strip(),
                    messages_json,
                    create.response,
                    route_json,
                    settings_json,
                    reasoning_json,
                    context_json,
                    cache_json,
                    telemetry_json,
                    create.system_instruction,
                ),
            )
        return generation_id

    def grade_generation(
        self,
        generation_id: str,
        *,
        reward: float,
        dimensions: Mapping[str, float] | None = None,
        notes: str = "",
        corrected_answer: str | None = None,
        graded_at: datetime | None = None,
    ) -> GenerationRecord:
        overall = _score(reward, "reward")
        dimension_values = {
            name: _score(value, f"dimensions[{name!r}]")
            for name, value in (dimensions or {}).items()
        }
        dimensions_json, _ = _json_object(dimension_values, "dimensions")
        if not isinstance(notes, str):
            raise TypeError("notes must be a string")
        if corrected_answer is not None:
            if not isinstance(corrected_answer, str):
                raise TypeError("corrected_answer must be a string or None")
            corrected_answer = corrected_answer.strip() or None
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE generations
                SET reward = ?, dimensions_json = ?, notes = ?, corrected_answer = ?, graded_at = ?
                WHERE generation_id = ?
                """,
                (
                    overall,
                    dimensions_json,
                    notes,
                    corrected_answer,
                    _timestamp(graded_at),
                    generation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown generation {generation_id!r}")
        record = self.get_generation(generation_id)
        assert record is not None
        return record

    def record_preference(
        self,
        chosen_generation_id: str,
        rejected_generation_id: str,
        *,
        notes: str = "",
        evaluator_id: str = "human",
        evaluator_model: str | None = None,
        confidence: float | None = None,
        metadata: Mapping[str, object] | None = None,
        created_at: datetime | None = None,
    ) -> str:
        if chosen_generation_id == rejected_generation_id:
            raise ValueError("chosen and rejected generations must differ")
        chosen = self.get_generation(chosen_generation_id)
        rejected = self.get_generation(rejected_generation_id)
        if chosen is None or rejected is None:
            raise KeyError("both preference generations must exist")
        if chosen.messages != rejected.messages:
            raise ValueError("pairwise generations must have the same input messages")
        if not isinstance(notes, str):
            raise TypeError("notes must be a string")
        if not isinstance(evaluator_id, str) or not evaluator_id.strip():
            raise ValueError("evaluator_id must be a non-empty string")
        if evaluator_model is not None and not isinstance(evaluator_model, str):
            raise TypeError("evaluator_model must be a string or None")
        preference_confidence = (
            None if confidence is None else _confidence(confidence)
        )
        metadata_json, _ = _json_object(metadata or {}, "metadata")
        preference_id = uuid.uuid4().hex
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO preferences (
                    preference_id, chosen_generation_id, rejected_generation_id, notes, created_at,
                    evaluator_id, evaluator_model, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preference_id,
                    chosen_generation_id,
                    rejected_generation_id,
                    notes,
                    _timestamp(created_at),
                    evaluator_id.strip(),
                    evaluator_model.strip() if evaluator_model else None,
                    preference_confidence,
                    metadata_json,
                ),
            )
        return preference_id

    def record_teacher_annotation(
        self,
        generation_id: str,
        *,
        teacher_id: str,
        overall_score: float,
        dimensions: Mapping[str, float] | None = None,
        confidence: float = 1.0,
        critique: str = "",
        corrected_answer: str | None = None,
        teacher_model: str | None = None,
        rubric_version: str = "molor.teacher-rubric.v1",
        metadata: Mapping[str, object] | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Record one independent evaluator judgment without replacing other grades."""

        if self.get_generation(generation_id) is None:
            raise KeyError(f"unknown generation {generation_id!r}")
        if not isinstance(teacher_id, str) or not teacher_id.strip():
            raise ValueError("teacher_id must be a non-empty string")
        if teacher_model is not None and not isinstance(teacher_model, str):
            raise TypeError("teacher_model must be a string or None")
        if not isinstance(rubric_version, str) or not rubric_version.strip():
            raise ValueError("rubric_version must be a non-empty string")
        if not isinstance(critique, str):
            raise TypeError("critique must be a string")
        if corrected_answer is not None:
            if not isinstance(corrected_answer, str):
                raise TypeError("corrected_answer must be a string or None")
            corrected_answer = corrected_answer.strip() or None
        dimension_values = {
            name: _score(value, f"dimensions[{name!r}]")
            for name, value in (dimensions or {}).items()
        }
        dimensions_json, _ = _json_object(dimension_values, "dimensions")
        metadata_json, _ = _json_object(metadata or {}, "metadata")
        annotation_id = uuid.uuid4().hex
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO teacher_annotations (
                    annotation_id, generation_id, teacher_id, teacher_model, created_at,
                    overall_score, dimensions_json, confidence, critique, corrected_answer,
                    rubric_version, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    generation_id,
                    teacher_id.strip(),
                    teacher_model.strip() if teacher_model else None,
                    _timestamp(created_at),
                    _score(overall_score, "overall_score"),
                    dimensions_json,
                    _confidence(confidence),
                    critique,
                    corrected_answer,
                    rubric_version.strip(),
                    metadata_json,
                ),
            )
        return annotation_id

    def list_teacher_annotations(
        self,
        *,
        generation_id: str | None = None,
        teacher_id: str | None = None,
        limit: int = 100,
    ) -> list[TeacherAnnotationRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        clauses: list[str] = []
        values: list[object] = []
        if generation_id is not None:
            clauses.append("generation_id = ?")
            values.append(generation_id)
        if teacher_id is not None:
            clauses.append("teacher_id = ?")
            values.append(teacher_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM teacher_annotations{where} "  # noqa: S608
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._teacher_annotation(row) for row in rows]

    def next_generation_for_teacher(self, teacher_id: str) -> GenerationRecord | None:
        """Return the oldest generation this evaluator has not independently reviewed."""

        if not isinstance(teacher_id, str) or not teacher_id.strip():
            raise ValueError("teacher_id must be a non-empty string")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT generations.*
                FROM generations
                WHERE NOT EXISTS (
                    SELECT 1 FROM teacher_annotations
                    WHERE teacher_annotations.generation_id = generations.generation_id
                      AND teacher_annotations.teacher_id = ?
                )
                ORDER BY generations.created_at ASC, generations.rowid ASC
                LIMIT 1
                """,
                (teacher_id.strip(),),
            ).fetchone()
        return None if row is None else self._generation(row)

    def teacher_consensus(self, generation_id: str) -> dict[str, object]:
        annotations = self.list_teacher_annotations(generation_id=generation_id, limit=10_000)
        if not annotations:
            return {
                "generation_id": generation_id,
                "annotation_count": 0,
                "weighted_overall": None,
                "disagreement": None,
                "dimensions": {},
            }
        weights = [annotation.confidence for annotation in annotations]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            weights = [1.0] * len(annotations)
            weight_sum = float(len(annotations))
        weighted_overall = sum(
            annotation.overall_score * weight
            for annotation, weight in zip(annotations, weights, strict=True)
        ) / weight_sum
        variance = sum(
            weight * (annotation.overall_score - weighted_overall) ** 2
            for annotation, weight in zip(annotations, weights, strict=True)
        ) / weight_sum
        dimension_names = sorted(
            {name for annotation in annotations for name in annotation.dimensions}
        )
        dimensions: dict[str, float] = {}
        for name in dimension_names:
            values = [
                (annotation.dimensions[name], weight)
                for annotation, weight in zip(annotations, weights, strict=True)
                if name in annotation.dimensions
            ]
            dimensions[name] = sum(value * weight for value, weight in values) / sum(
                weight for _, weight in values
            )
        return {
            "generation_id": generation_id,
            "annotation_count": len(annotations),
            "weighted_overall": weighted_overall,
            "disagreement": math.sqrt(variance),
            "dimensions": dimensions,
        }

    def get_generation(self, generation_id: str) -> GenerationRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM generations WHERE generation_id = ?", (generation_id,)
            ).fetchone()
        return None if row is None else self._generation(row)

    def list_generations(
        self,
        *,
        limit: int = 100,
        conversation_id: str | None = None,
        graded_only: bool = False,
    ) -> list[GenerationRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        clauses: list[str] = []
        values: list[object] = []
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            values.append(conversation_id)
        if graded_only:
            clauses.append("reward IS NOT NULL")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM generations{where} "  # noqa: S608
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._generation(row) for row in rows]

    def list_preferences(self, *, limit: int = 100) -> list[PreferenceRecord]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM preferences ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._preference(row) for row in rows]

    def _all_generations(self, *, graded_only: bool = False) -> list[GenerationRecord]:
        where = " WHERE reward IS NOT NULL" if graded_only else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM generations{where} ORDER BY created_at ASC, rowid ASC"  # noqa: S608
            ).fetchall()
        return [self._generation(row) for row in rows]

    def _all_preferences(self) -> list[PreferenceRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM preferences ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
        return [self._preference(row) for row in rows]

    @staticmethod
    def _generation(row: sqlite3.Row) -> GenerationRecord:
        return GenerationRecord(
            generation_id=row["generation_id"],
            conversation_id=row["conversation_id"],
            created_at=row["created_at"],
            graded_at=row["graded_at"],
            prompt=row["prompt"],
            model_id=row["model_id"],
            messages=tuple(json.loads(row["messages_json"])),
            response=row["response"],
            route=json.loads(row["route_json"]),
            generation_settings=json.loads(row["generation_settings_json"]),
            reasoning_settings=json.loads(row["reasoning_settings_json"]),
            context_settings=json.loads(row["context_settings_json"]),
            cache_settings=json.loads(row["cache_settings_json"]),
            telemetry=json.loads(row["telemetry_json"]),
            system_instruction=row["system_instruction"],
            reward=row["reward"],
            dimensions=json.loads(row["dimensions_json"]),
            notes=row["notes"],
            corrected_answer=row["corrected_answer"],
        )

    @staticmethod
    def _preference(row: sqlite3.Row) -> PreferenceRecord:
        return PreferenceRecord(
            preference_id=row["preference_id"],
            chosen_generation_id=row["chosen_generation_id"],
            rejected_generation_id=row["rejected_generation_id"],
            notes=row["notes"],
            created_at=row["created_at"],
            evaluator_id=row["evaluator_id"],
            evaluator_model=row["evaluator_model"],
            confidence=row["confidence"],
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _teacher_annotation(row: sqlite3.Row) -> TeacherAnnotationRecord:
        return TeacherAnnotationRecord(
            annotation_id=row["annotation_id"],
            generation_id=row["generation_id"],
            teacher_id=row["teacher_id"],
            teacher_model=row["teacher_model"],
            created_at=row["created_at"],
            overall_score=row["overall_score"],
            dimensions=json.loads(row["dimensions_json"]),
            confidence=row["confidence"],
            critique=row["critique"],
            corrected_answer=row["corrected_answer"],
            rubric_version=row["rubric_version"],
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _write_jsonl(path: str | Path, rows: list[dict[str, object]]) -> ExportResult:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        return ExportResult(destination.resolve(), len(rows))

    def export_reward_jsonl(
        self, path: str | Path = DEFAULT_EXPORT_DIR / "rewards.jsonl"
    ) -> ExportResult:
        records = self._all_generations(graded_only=True)
        rows = [
            {
                "schema_version": "molor.manual-reward.v1",
                "generation_id": record.generation_id,
                "conversation_id": record.conversation_id,
                "created_at": record.created_at,
                "graded_at": record.graded_at,
                "prompt": record.prompt,
                "messages": list(record.messages),
                "response": record.response,
                "model_id": record.model_id,
                "route": record.route,
                "generation_settings": record.generation_settings,
                "reasoning_settings": record.reasoning_settings,
                "context_settings": record.context_settings,
                "cache_settings": record.cache_settings,
                "telemetry": record.telemetry,
                "reward": record.reward,
                "dimensions": record.dimensions,
                "notes": record.notes,
                "corrected_answer": record.corrected_answer,
            }
            for record in records
        ]
        return self._write_jsonl(path, rows)

    def export_positive_sft_jsonl(
        self,
        path: str | Path = DEFAULT_EXPORT_DIR / "positive_sft.jsonl",
        *,
        min_reward: float = 1.0,
        prefer_corrected: bool = True,
    ) -> ExportResult:
        threshold = _score(min_reward, "min_reward")
        records = self._all_generations(graded_only=True)
        rows: list[dict[str, object]] = []
        for record in records:
            corrected = record.corrected_answer
            if record.reward is None:
                continue
            if record.reward < threshold and not (prefer_corrected and corrected):
                continue
            target = corrected if prefer_corrected and corrected else record.response
            rows.append(
                {
                    "messages": [
                        *record.messages,
                        {"role": "assistant", "content": target},
                    ]
                }
            )
        return self._write_jsonl(path, rows)

    def export_training_sft_jsonl(
        self,
        path: str | Path = DEFAULT_EXPORT_DIR / "training_sft.jsonl",
        *,
        min_human_reward: float = 1.0,
        min_teacher_score: float = 1.0,
        min_teacher_confidence: float = 0.5,
    ) -> ExportResult:
        """Combine human positives and confidence-filtered teacher targets for SFT."""

        human_threshold = _score(min_human_reward, "min_human_reward")
        teacher_threshold = _score(min_teacher_score, "min_teacher_score")
        teacher_confidence = _confidence(min_teacher_confidence, "min_teacher_confidence")
        rows: list[dict[str, object]] = []
        seen: set[str] = set()

        def append(record: GenerationRecord, target: str) -> None:
            messages = [*record.messages, {"role": "assistant", "content": target}]
            key = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                rows.append({"messages": messages})

        for record in self._all_generations():
            if record.corrected_answer:
                append(record, record.corrected_answer)
            elif record.reward is not None and record.reward >= human_threshold:
                append(record, record.response)

            annotations = self.list_teacher_annotations(
                generation_id=record.generation_id,
                limit=10_000,
            )
            eligible = [
                annotation
                for annotation in annotations
                if annotation.confidence >= teacher_confidence
            ]
            corrections = [
                annotation for annotation in eligible if annotation.corrected_answer
            ]
            if corrections:
                best = max(
                    corrections,
                    key=lambda annotation: (
                        annotation.confidence,
                        annotation.overall_score,
                        annotation.created_at,
                    ),
                )
                assert best.corrected_answer is not None
                append(record, best.corrected_answer)
            elif eligible:
                consensus = self.teacher_consensus(record.generation_id)
                overall = consensus["weighted_overall"]
                if isinstance(overall, (int, float)) and overall >= teacher_threshold:
                    append(record, record.response)
        return self._write_jsonl(path, rows)

    def export_preference_jsonl(
        self, path: str | Path = DEFAULT_EXPORT_DIR / "preferences.jsonl"
    ) -> ExportResult:
        preferences = self._all_preferences()
        rows: list[dict[str, object]] = []
        for preference in preferences:
            chosen = self.get_generation(preference.chosen_generation_id)
            rejected = self.get_generation(preference.rejected_generation_id)
            assert chosen is not None and rejected is not None
            rows.append(
                {
                    "schema_version": "molor.pairwise-preference.v1",
                    "preference_id": preference.preference_id,
                    "messages": list(chosen.messages),
                    "chosen": chosen.response,
                    "rejected": rejected.response,
                    "chosen_generation_id": chosen.generation_id,
                    "rejected_generation_id": rejected.generation_id,
                    "notes": preference.notes,
                    "created_at": preference.created_at,
                    "evaluator_id": preference.evaluator_id,
                    "evaluator_model": preference.evaluator_model,
                    "confidence": preference.confidence,
                    "metadata": preference.metadata,
                }
            )
        return self._write_jsonl(path, rows)

    def export_teacher_reward_jsonl(
        self, path: str | Path = DEFAULT_EXPORT_DIR / "teacher_rewards.jsonl"
    ) -> ExportResult:
        rows: list[dict[str, object]] = []
        for generation in self._all_generations():
            annotations = self.list_teacher_annotations(
                generation_id=generation.generation_id,
                limit=10_000,
            )
            if not annotations:
                continue
            rows.append(
                {
                    "schema_version": "molor.teacher-reward.v1",
                    "generation_id": generation.generation_id,
                    "messages": list(generation.messages),
                    "response": generation.response,
                    "model_id": generation.model_id,
                    "route": generation.route,
                    "generation_settings": generation.generation_settings,
                    "reasoning_settings": generation.reasoning_settings,
                    "context_settings": generation.context_settings,
                    "cache_settings": generation.cache_settings,
                    "telemetry": generation.telemetry,
                    "annotations": [
                        {
                            "annotation_id": annotation.annotation_id,
                            "teacher_id": annotation.teacher_id,
                            "teacher_model": annotation.teacher_model,
                            "created_at": annotation.created_at,
                            "overall_score": annotation.overall_score,
                            "dimensions": annotation.dimensions,
                            "confidence": annotation.confidence,
                            "critique": annotation.critique,
                            "corrected_answer": annotation.corrected_answer,
                            "rubric_version": annotation.rubric_version,
                            "metadata": annotation.metadata,
                        }
                        for annotation in annotations
                    ],
                    "consensus": self.teacher_consensus(generation.generation_id),
                }
            )
        return self._write_jsonl(path, rows)
