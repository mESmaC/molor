"""Public values for the local generation-annotation notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class GenerationCreate:
    """Everything needed to reproduce and grade one completed generation."""

    conversation_id: str
    model_id: str
    messages: tuple[Mapping[str, str], ...]
    response: str
    route: Mapping[str, float] = field(default_factory=dict)
    settings: Mapping[str, object] = field(default_factory=dict)
    telemetry: Mapping[str, object] = field(default_factory=dict)
    system_instruction: str | None = None
    enable_thinking: bool = False
    reasoning_preset: str = "default"
    context_policy: Mapping[str, object] = field(default_factory=dict)
    cache_policy: Mapping[str, object] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation_id: str
    conversation_id: str
    created_at: str
    graded_at: str | None
    prompt: str
    model_id: str
    messages: tuple[dict[str, str], ...]
    response: str
    route: dict[str, float]
    generation_settings: dict[str, object]
    reasoning_settings: dict[str, object]
    context_settings: dict[str, object]
    cache_settings: dict[str, object]
    telemetry: dict[str, object]
    system_instruction: str | None
    reward: float | None
    dimensions: dict[str, float]
    notes: str
    corrected_answer: str | None


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    preference_id: str
    chosen_generation_id: str
    rejected_generation_id: str
    notes: str
    created_at: str
    evaluator_id: str = "human"
    evaluator_model: str | None = None
    confidence: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TeacherAnnotationRecord:
    """One independent teacher judgment for a student generation."""

    annotation_id: str
    generation_id: str
    teacher_id: str
    teacher_model: str | None
    created_at: str
    overall_score: float
    dimensions: dict[str, float]
    confidence: float
    critique: str
    corrected_answer: str | None
    rubric_version: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    rows: int
