"""Benchmark scoring bridge for deterministic and operator-managed verifiers.

The bridge deliberately does not create a sandbox.  An ``ExternalCommandVerifier`` registry is
operator configuration, and each configured command must itself launch an appropriate container or
VM.  Suite data selects a symbolic verifier name; model output is never interpolated into argv.

Semantic evidence binds the prompt/completion/reference, scorer declaration, evaluator manifest,
and verifier result.  External evidence can therefore be checked during resume/report generation
without executing untrusted output again.  This is content integrity and reproducibility evidence,
not a cryptographic signature against an attacker who can rewrite the whole run directory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from molor.benchmark.models import BenchmarkCase
from molor.benchmark.scoring import SCORER_VERSION, ScoreResult, ScoreSpecError, score_response
from molor.identity import fingerprint_json, freeze_json, thaw_json, validate_json_value
from molor.rewards import (
    EvaluationSample,
    ExternalCommandVerifier,
    VerificationResult,
    VerificationStatus,
)


EVALUATOR_SCHEMA = "molor.benchmark-evaluator.v1"
EVIDENCE_SCHEMA = "molor.benchmark-evidence.v1"
REGISTRY_SCHEMA = "molor.benchmark-verifiers.v1"
_EXTERNAL_SCORER_FIELDS = frozenset({"type", "verifier"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class EvaluationConfigError(ValueError):
    """The external-verifier registry or scorer declaration is invalid."""


class EvaluationError(RuntimeError):
    """Base class for evaluation failures distinct from model-generation failures."""


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    """Canonical semantic evaluator output with a content fingerprint."""

    payload: Mapping[str, Any]
    fingerprint: str

    def __post_init__(self) -> None:
        payload = dict(self.payload)
        validate_json_value(payload, "evaluation evidence")
        if payload.get("schema_version") != EVIDENCE_SCHEMA:
            raise ValueError(f"evaluation evidence must use {EVIDENCE_SCHEMA!r}")
        expected = fingerprint_json(payload)
        if self.fingerprint != expected:
            raise ValueError("evaluation evidence fingerprint does not match its payload")
        object.__setattr__(self, "payload", freeze_json(payload))

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> EvaluationEvidence:
        value = dict(payload)
        value["schema_version"] = EVIDENCE_SCHEMA
        return cls(payload=value, fingerprint=fingerprint_json(value))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvaluationEvidence:
        if not isinstance(raw, Mapping):
            raise TypeError("evaluation evidence must be an object")
        payload = dict(raw)
        fingerprint = payload.pop("evidence_fingerprint", None)
        if not isinstance(fingerprint, str):
            raise ValueError("evaluation evidence has no fingerprint")
        return cls(payload=payload, fingerprint=fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {**thaw_json(self.payload), "evidence_fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """One scorable benchmark result and its replay-free semantic evidence."""

    result: ScoreResult
    evidence: EvaluationEvidence

    def to_record(self) -> dict[str, Any]:
        return {**self.result.to_dict(), "evaluation_evidence": self.evidence.to_dict()}


class UnscorableEvaluation(EvaluationError):
    """A verifier/reference/infrastructure failure that must be masked, never scored as zero."""

    def __init__(
        self,
        verification_result: VerificationResult,
        evidence: EvaluationEvidence,
    ) -> None:
        if verification_result.status.scorable or verification_result.score is not None:
            raise ValueError("UnscorableEvaluation requires an unscorable verification result")
        self.verification_result = verification_result
        self.evidence = evidence
        detail = verification_result.error or "external evaluation was unscorable"
        super().__init__(
            f"{verification_result.verifier}:{verification_result.status.value}: {detail}"
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "score": None,
            "passed": False,
            "extracted": None,
            "details": {
                "scorer_type": "external",
                "verification": _verification_semantics(self.verification_result),
            },
            "evaluation_evidence": self.evidence.to_dict(),
            "evaluation_error": str(self),
        }


def _external_verifier_id(raw_spec: Mapping[str, Any]) -> str | None:
    if not isinstance(raw_spec, Mapping):
        raise ScoreSpecError("scorer must be an object")
    spec = dict(raw_spec)
    scorer_type = str(spec.get("type", "")).strip().lower()
    if scorer_type != "external":
        return None
    unknown = set(spec).difference(_EXTERNAL_SCORER_FIELDS)
    if unknown:
        raise ScoreSpecError(
            "external scorer has unknown fields: " + ", ".join(sorted(unknown))
        )
    verifier_id = spec.get("verifier")
    if not isinstance(verifier_id, str) or _IDENTIFIER.fullmatch(verifier_id) is None:
        raise ScoreSpecError("external.verifier must be a benchmark identifier")
    return verifier_id


def _case_sample(case: BenchmarkCase, completion: str, sample_id: str) -> EvaluationSample:
    # ``getattr`` keeps the bridge importable while the suite-v2 case fields are integrated.
    reference = getattr(case, "reference", None)
    raw_metadata = getattr(case, "metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("benchmark case metadata must be an object")
    return EvaluationSample(
        sample_id=sample_id,
        prompt=case.prompt,
        completion=completion,
        reference=reference,
        metadata=dict(raw_metadata),
    )


def _verification_semantics(result: VerificationResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload.pop("duration_ms", None)
    return payload


def _score_from_verification(
    verifier_id: str,
    result: VerificationResult,
) -> ScoreResult:
    if not result.status.scorable or result.score is None:
        raise ValueError("cannot make a benchmark score from an unscorable verification result")
    semantic_result = _verification_semantics(result)
    return ScoreResult(
        score=result.score,
        passed=result.passed,
        extracted=None,
        details={
            "scorer_type": "external",
            "verifier": verifier_id,
            "verification_status": result.status.value,
            "verification": semantic_result,
        },
    )


def _score_result_from_dict(raw: object) -> ScoreResult:
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation evidence result must be an object")
    payload = dict(raw)
    if set(payload) != {"score", "passed", "extracted", "details"}:
        raise ValueError("evaluation evidence result has unknown or missing fields")
    score = payload["score"]
    passed = payload["passed"]
    details = payload["details"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("evaluation evidence score must be numeric")
    if not isinstance(passed, bool) or not isinstance(details, Mapping):
        raise TypeError("evaluation evidence has invalid passed/details fields")
    return ScoreResult(
        score=float(score),
        passed=passed,
        extracted=payload["extracted"],
        details=dict(details),
    )


def _verification_from_dict(raw: object) -> VerificationResult:
    if not isinstance(raw, Mapping):
        raise ValueError("external evaluation evidence has no verification result")
    payload = dict(raw)
    allowed = {"verifier", "status", "score", "passed", "details", "error"}
    if not set(payload).issubset(allowed) or not {
        "verifier",
        "status",
        "score",
        "passed",
        "details",
    }.issubset(payload):
        raise ValueError("external verification evidence has unknown or missing fields")
    try:
        status = VerificationStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("external verification evidence has an invalid status") from exc
    return VerificationResult(
        verifier=payload["verifier"],
        status=status,
        score=payload["score"],
        passed=payload["passed"],
        details=payload["details"],
        error=payload.get("error"),
    )


def _load_registry_source(source: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return {"schema_version": REGISTRY_SCHEMA, "verifiers": {}}
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationConfigError(f"cannot read verifier registry {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationConfigError(f"invalid JSON in verifier registry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationConfigError("verifier registry must contain one JSON object")
    return value


def load_external_verifiers(
    source: str | Path | Mapping[str, Any] | None,
    *,
    required: Iterable[str] = (),
) -> dict[str, ExternalCommandVerifier]:
    """Build a fixed-argv verifier registry without launching any configured command."""

    raw = _load_registry_source(source)
    if set(raw) != {"schema_version", "verifiers"}:
        raise EvaluationConfigError("verifier registry has unknown or missing fields")
    if raw.get("schema_version") != REGISTRY_SCHEMA:
        raise EvaluationConfigError(f"verifier registry must use {REGISTRY_SCHEMA!r}")
    raw_verifiers = raw.get("verifiers")
    if not isinstance(raw_verifiers, Mapping):
        raise EvaluationConfigError("verifier registry verifiers must be an object")

    verifiers: dict[str, ExternalCommandVerifier] = {}
    allowed = {
        "command",
        "version",
        "runner_fingerprint",
        "timeout_seconds",
        "max_output_chars",
    }
    for verifier_id, raw_entry in raw_verifiers.items():
        if not isinstance(verifier_id, str) or _IDENTIFIER.fullmatch(verifier_id) is None:
            raise EvaluationConfigError("verifier registry keys must be benchmark identifiers")
        if not isinstance(raw_entry, Mapping):
            raise EvaluationConfigError(f"verifier {verifier_id!r} must be an object")
        entry = dict(raw_entry)
        unknown = set(entry).difference(allowed)
        if unknown:
            raise EvaluationConfigError(
                f"verifier {verifier_id!r} has unknown fields: " + ", ".join(sorted(unknown))
            )
        command = entry.get("command")
        if (
            isinstance(command, (str, bytes))
            or not isinstance(command, Sequence)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise EvaluationConfigError(
                f"verifier {verifier_id!r} command must be a non-empty argv array"
            )
        command = list(command)
        if command[0] == "{python}":
            command[0] = sys.executable
        version = entry.get("version")
        runner_fingerprint = entry.get("runner_fingerprint")
        if not isinstance(version, str) or not version.strip():
            raise EvaluationConfigError(f"verifier {verifier_id!r} version is required")
        if not isinstance(runner_fingerprint, str) or not runner_fingerprint.strip():
            raise EvaluationConfigError(
                f"verifier {verifier_id!r} runner_fingerprint is required"
            )
        timeout = entry.get("timeout_seconds", 30.0)
        max_output = entry.get("max_output_chars", 100_000)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise EvaluationConfigError(
                f"verifier {verifier_id!r} timeout_seconds must be finite and positive"
            )
        if isinstance(max_output, bool) or not isinstance(max_output, int) or max_output < 1:
            raise EvaluationConfigError(
                f"verifier {verifier_id!r} max_output_chars must be a positive integer"
            )
        verifiers[verifier_id] = ExternalCommandVerifier(
            list(command),
            name=verifier_id,
            version=version.strip(),
            runner_fingerprint=runner_fingerprint.strip(),
            timeout_seconds=float(timeout),
            max_output_chars=max_output,
        )

    requested = set(required)
    if any(not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None for name in requested):
        raise EvaluationConfigError("required verifier names must be benchmark identifiers")
    missing = requested.difference(verifiers)
    if missing:
        raise EvaluationConfigError("verifier registry is missing: " + ", ".join(sorted(missing)))
    return {name: verifiers[name] for name in sorted(requested or verifiers)}


class BenchmarkEvaluator:
    """Evaluate benchmark completions with legacy scorers or named external verifiers."""

    def __init__(
        self,
        external_verifiers: Mapping[str, ExternalCommandVerifier] | None = None,
    ) -> None:
        registry = dict(external_verifiers or {})
        for verifier_id, verifier in registry.items():
            if not isinstance(verifier_id, str) or _IDENTIFIER.fullmatch(verifier_id) is None:
                raise EvaluationConfigError("external verifier names must be benchmark identifiers")
            if not isinstance(verifier, ExternalCommandVerifier):
                raise TypeError("external verifier registry values must be ExternalCommandVerifier")
            if verifier.name != verifier_id:
                raise EvaluationConfigError(
                    f"registry key {verifier_id!r} differs from verifier name {verifier.name!r}"
                )
        self._external_verifiers = registry
        implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        verifier_manifests: dict[str, Any] = {}
        for verifier_id, verifier in sorted(registry.items()):
            verifier_manifests[verifier_id] = {
                **verifier.manifest(),
                # ExternalCommandVerifier intentionally omits argv.  Its security flags still
                # affect benchmark identity, so bind them here without persisting possible paths.
                "command_argv_sha256": fingerprint_json(list(verifier.command)),
            }
        manifest = {
            "schema_version": EVALUATOR_SCHEMA,
            "implementation_sha256": implementation_sha256,
            "legacy_scorer_version": SCORER_VERSION,
            "external_verifiers": verifier_manifests,
        }
        self._manifest = freeze_json(manifest)
        self._fingerprint = fingerprint_json(manifest)

    @classmethod
    def from_config(
        cls,
        source: str | Path | Mapping[str, Any] | None,
        *,
        required: Iterable[str] = (),
    ) -> BenchmarkEvaluator:
        return cls(load_external_verifiers(source, required=required))

    @staticmethod
    def required_external_verifiers(cases: Iterable[BenchmarkCase]) -> tuple[str, ...]:
        names: set[str] = set()
        for case in cases:
            verifier_id = _external_verifier_id(case.scorer)
            if verifier_id is not None:
                names.add(verifier_id)
        return tuple(sorted(names))

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def manifest_dict(self) -> dict[str, Any]:
        return thaw_json(self._manifest)

    def validate_case(self, case: BenchmarkCase) -> None:
        """Validate scorer wiring without running an external verifier."""

        verifier_id = _external_verifier_id(case.scorer)
        if verifier_id is None:
            score_response("", case.scorer)
            return
        if verifier_id not in self._external_verifiers:
            raise EvaluationConfigError(f"external verifier {verifier_id!r} is not configured")
        _case_sample(case, "", case.case_id)

    def evaluate(
        self,
        case: BenchmarkCase,
        completion: str,
        *,
        sample_id: str | None = None,
    ) -> EvaluationOutcome:
        if not isinstance(completion, str):
            raise TypeError("completion must be a string")
        resolved_sample_id = sample_id or case.case_id
        sample = _case_sample(case, completion, resolved_sample_id)
        verifier_id = _external_verifier_id(case.scorer)
        if verifier_id is None:
            result = score_response(completion, case.scorer)
            evidence = self._make_evidence(case, sample, "legacy", result=result)
            return EvaluationOutcome(result=result, evidence=evidence)

        verifier = self._external_verifiers.get(verifier_id)
        if verifier is None:
            raise EvaluationConfigError(f"external verifier {verifier_id!r} is not configured")
        try:
            verification = verifier.verify(sample)
        except OSError as exc:
            verification = VerificationResult.unscorable(
                verifier_id,
                VerificationStatus.INFRA_ERROR,
                f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            verification = VerificationResult.unscorable(
                verifier_id,
                VerificationStatus.VERIFIER_ERROR,
                f"{type(exc).__name__}: {exc}",
            )
        if not verification.status.scorable:
            evidence = self._make_evidence(
                case,
                sample,
                "external",
                verification=verification,
                verifier_id=verifier_id,
            )
            raise UnscorableEvaluation(verification, evidence)
        result = _score_from_verification(verifier_id, verification)
        evidence = self._make_evidence(
            case,
            sample,
            "external",
            result=result,
            verification=verification,
            verifier_id=verifier_id,
        )
        return EvaluationOutcome(result=result, evidence=evidence)

    def validate_evidence(
        self,
        case: BenchmarkCase,
        completion: str,
        raw_evidence: EvaluationEvidence | Mapping[str, Any],
        *,
        sample_id: str | None = None,
    ) -> EvaluationOutcome:
        """Validate stored evidence; external commands are never invoked by this method."""

        evidence = (
            raw_evidence
            if isinstance(raw_evidence, EvaluationEvidence)
            else EvaluationEvidence.from_dict(raw_evidence)
        )
        payload = thaw_json(evidence.payload)
        allowed = {
            "schema_version",
            "evaluator_fingerprint",
            "sample_fingerprint",
            "scorer_fingerprint",
            "mode",
            "result",
            "external_verifier",
            "verification_result",
        }
        if not set(payload).issubset(allowed):
            raise ValueError("evaluation evidence has unknown fields")
        resolved_sample_id = sample_id or case.case_id
        sample = _case_sample(case, completion, resolved_sample_id)
        if payload.get("evaluator_fingerprint") != self.fingerprint:
            raise ValueError("stored evidence belongs to a different evaluator")
        if payload.get("sample_fingerprint") != sample.fingerprint:
            raise ValueError("stored evidence belongs to a different benchmark sample")
        if payload.get("scorer_fingerprint") != fingerprint_json(dict(case.scorer)):
            raise ValueError("stored evidence belongs to a different scorer declaration")

        verifier_id = _external_verifier_id(case.scorer)
        if verifier_id is None:
            if payload.get("mode") != "legacy" or set(payload).intersection(
                {"external_verifier", "verification_result"}
            ):
                raise ValueError("stored evidence mode differs from the benchmark scorer")
            stored = _score_result_from_dict(payload.get("result"))
            recomputed = score_response(completion, case.scorer)
            if stored.to_dict() != recomputed.to_dict():
                raise ValueError("stored legacy score failed deterministic revalidation")
            return EvaluationOutcome(result=recomputed, evidence=evidence)

        if verifier_id not in self._external_verifiers:
            raise EvaluationConfigError(f"external verifier {verifier_id!r} is not configured")
        if payload.get("mode") != "external" or payload.get("external_verifier") != verifier_id:
            raise ValueError("stored evidence mode/verifier differs from the benchmark scorer")
        verification = _verification_from_dict(payload.get("verification_result"))
        if verification.verifier != verifier_id:
            raise ValueError("stored verification result has the wrong verifier name")
        if not verification.status.scorable:
            if payload.get("result") is not None:
                raise ValueError("unscorable external evidence must not contain a score result")
            raise UnscorableEvaluation(verification, evidence)
        expected = _score_from_verification(verifier_id, verification)
        stored = _score_result_from_dict(payload.get("result"))
        if stored.to_dict() != expected.to_dict():
            raise ValueError("stored external score contradicts its verification result")
        return EvaluationOutcome(result=expected, evidence=evidence)

    def _make_evidence(
        self,
        case: BenchmarkCase,
        sample: EvaluationSample,
        mode: str,
        *,
        result: ScoreResult | None = None,
        verification: VerificationResult | None = None,
        verifier_id: str | None = None,
    ) -> EvaluationEvidence:
        payload: dict[str, Any] = {
            "evaluator_fingerprint": self.fingerprint,
            "sample_fingerprint": sample.fingerprint,
            "scorer_fingerprint": fingerprint_json(dict(case.scorer)),
            "mode": mode,
            "result": result.to_dict() if result is not None else None,
        }
        if verification is not None:
            payload["external_verifier"] = verifier_id
            payload["verification_result"] = _verification_semantics(verification)
        return EvaluationEvidence.create(payload)


__all__ = [
    "EVALUATOR_SCHEMA",
    "EVIDENCE_SCHEMA",
    "REGISTRY_SCHEMA",
    "BenchmarkEvaluator",
    "EvaluationConfigError",
    "EvaluationError",
    "EvaluationEvidence",
    "EvaluationOutcome",
    "UnscorableEvaluation",
    "load_external_verifiers",
]
