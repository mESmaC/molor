"""Serializable values shared by verifier and reward components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Mapping

from molor.identity import (
    canonical_json_bytes,
    fingerprint_json,
    freeze_json,
    thaw_json,
    validate_json_value,
)


class VerificationStatus(StrEnum):
    """Typed verifier outcome; infrastructure/reference errors are not model rewards."""

    PASS = "pass"
    WRONG = "wrong"
    FORMAT_FAIL = "format_fail"
    COMPILE_FAIL = "compile_fail"
    TEST_FAIL = "test_fail"
    MODEL_TIMEOUT = "model_timeout"
    POLICY_VIOLATION = "policy_violation"
    VERIFIER_ERROR = "verifier_error"
    INFRA_ERROR = "infra_error"
    INVALID_REFERENCE = "invalid_reference"

    @property
    def scorable(self) -> bool:
        return self not in {
            VerificationStatus.VERIFIER_ERROR,
            VerificationStatus.INFRA_ERROR,
            VerificationStatus.INVALID_REFERENCE,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """One prompt/completion pair presented to deterministic verifiers."""

    sample_id: str
    prompt: str
    completion: str
    reference: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not isinstance(self.prompt, str) or not isinstance(self.completion, str):
            raise TypeError("prompt and completion must be strings")
        metadata = dict(self.metadata)
        validate_json_value(self.reference, "reference")
        validate_json_value(metadata, "metadata")
        object.__setattr__(self, "reference", freeze_json(self.reference))
        object.__setattr__(self, "metadata", freeze_json(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "completion": self.completion,
            "reference": thaw_json(self.reference),
            "metadata": thaw_json(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    @property
    def prompt_fingerprint(self) -> str:
        """Identify prompt/reference semantics independently of rollout completion and ID."""

        return fingerprint_json(
            {
                "prompt": self.prompt,
                "reference": thaw_json(self.reference),
                "metadata": thaw_json(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """A verifier's bounded score plus diagnostic information."""

    verifier: str
    status: VerificationStatus
    score: float | None
    passed: bool
    details: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.verifier:
            raise ValueError("verifier must not be empty")
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("status must be a VerificationStatus")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be boolean")
        if self.status.scorable:
            if (
                self.score is None
                or isinstance(self.score, bool)
                or not isinstance(self.score, (int, float))
                or not math.isfinite(self.score)
                or not 0.0 <= self.score <= 1.0
            ):
                raise ValueError("scorable verification results require a finite score in [0, 1]")
            object.__setattr__(self, "score", float(self.score))
        elif self.score is not None:
            raise ValueError("unscorable verification results must use score=None")
        if self.passed != (self.status is VerificationStatus.PASS):
            raise ValueError("passed must be true exactly when status is PASS")
        if self.duration_ms is not None:
            if (
                isinstance(self.duration_ms, bool)
                or not isinstance(self.duration_ms, (int, float))
                or not math.isfinite(self.duration_ms)
                or self.duration_ms < 0
            ):
                raise ValueError("duration_ms must be finite and non-negative")
            object.__setattr__(self, "duration_ms", float(self.duration_ms))
        details = dict(self.details)
        validate_json_value(details, "details")
        encoded_details = canonical_json_bytes(details)
        if len(encoded_details) > 64_000:
            raise ValueError("verification details must not exceed 64,000 encoded bytes")
        object.__setattr__(self, "details", freeze_json(details))
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if self.error is not None and len(self.error) > 4_000:
            object.__setattr__(self, "error", f"{self.error[:4_000]}...")

    @classmethod
    def unscorable(
        cls,
        verifier: str,
        status: VerificationStatus,
        error: str,
    ) -> VerificationResult:
        if status.scorable:
            raise ValueError("unscorable result requires an error status")
        return cls(verifier=verifier, status=status, score=None, passed=False, error=error)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "verifier": self.verifier,
            "status": self.status.value,
            "score": self.score,
            "passed": self.passed,
            "details": thaw_json(self.details),
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
        return payload
