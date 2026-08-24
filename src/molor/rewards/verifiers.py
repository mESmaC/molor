"""Deterministic, in-process verifiers that never execute model output."""

from __future__ import annotations

from fractions import Fraction
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from molor.rewards.base import Verifier
from molor.rewards.types import EvaluationSample, VerificationResult, VerificationStatus


def _preview(value: str, limit: int = 240) -> str:
    return value if len(value) <= limit else f"{value[:limit]}..."


class ExactMatchVerifier(Verifier):
    """Compare normalized completion text with a string reference."""

    def __init__(
        self,
        *,
        name: str = "exact_match",
        version: str = "1",
        case_sensitive: bool = True,
        collapse_whitespace: bool = False,
        strip: bool = False,
        unicode_form: str | None = None,
    ) -> None:
        if unicode_form not in {None, "NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError("unicode_form must be a standard Unicode normalization form or None")
        self.name = name
        self.version = version
        self.case_sensitive = case_sensitive
        self.collapse_whitespace = collapse_whitespace
        self.strip = strip
        self.unicode_form = unicode_form

    def _normalize(self, value: str) -> str:
        normalized = unicodedata.normalize(self.unicode_form, value) if self.unicode_form else value
        if self.strip:
            normalized = normalized.strip()
        if self.collapse_whitespace:
            normalized = " ".join(normalized.split())
        if not self.case_sensitive:
            normalized = normalized.casefold()
        return normalized

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        if not isinstance(sample.reference, str):
            return VerificationResult.unscorable(
                self.name,
                VerificationStatus.INVALID_REFERENCE,
                "exact-match reference must be a string",
            )
        predicted = self._normalize(sample.completion)
        expected = self._normalize(sample.reference)
        passed = predicted == expected
        return VerificationResult(
            verifier=self.name,
            status=VerificationStatus.PASS if passed else VerificationStatus.WRONG,
            score=float(passed),
            passed=passed,
            details={"predicted": _preview(predicted), "expected": _preview(expected)},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "case_sensitive": self.case_sensitive,
            "collapse_whitespace": self.collapse_whitespace,
            "strip": self.strip,
            "unicode_form": self.unicode_form,
        }


class RegexVerifier(Verifier):
    """Binary format verifier using a fixed operator-supplied regular expression."""

    def __init__(
        self,
        pattern: str,
        *,
        name: str = "regex",
        version: str = "1",
        full_match: bool = False,
        flags: int = 0,
    ) -> None:
        self.name = name
        self.version = version
        self.pattern = pattern
        self.full_match = full_match
        self.flags = flags
        self._compiled = re.compile(pattern, flags)

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        match = (
            self._compiled.fullmatch(sample.completion)
            if self.full_match
            else self._compiled.search(sample.completion)
        )
        passed = match is not None
        return VerificationResult(
            verifier=self.name,
            status=VerificationStatus.PASS if passed else VerificationStatus.FORMAT_FAIL,
            score=float(passed),
            passed=passed,
            details={"matched": _preview(match.group(0)) if match else None},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "pattern": self.pattern,
            "full_match": self.full_match,
            "flags": self.flags,
        }


class NumericAnswerVerifier(Verifier):
    """Extract and compare a final numeric answer with configurable tolerance."""

    _integer = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    _number_pattern = re.compile(
        rf"[+-]?(?:(?:{_integer})(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        rf"(?:\s*/\s*[+-]?{_integer})?%?"
    )
    _answer_patterns = (
        re.compile(r"\\boxed\{([^{}]+)\}", re.IGNORECASE),
        re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL),
        re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\r\n]+)", re.IGNORECASE),
    )

    def __init__(
        self,
        *,
        name: str = "numeric_answer",
        version: str = "1",
        absolute_tolerance: float = 1e-9,
        relative_tolerance: float = 1e-9,
        extraction: str = "whole",
    ) -> None:
        if (
            not math.isfinite(absolute_tolerance)
            or not math.isfinite(relative_tolerance)
            or absolute_tolerance < 0
            or relative_tolerance < 0
        ):
            raise ValueError("numeric tolerances must be finite and non-negative")
        self.name = name
        self.version = version
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        if extraction not in {"whole", "final"}:
            raise ValueError("extraction must be 'whole' or 'final'")
        self.extraction = extraction

    def _candidate(self, value: str) -> str:
        if self.extraction == "whole":
            candidate = value.strip()
            if self._number_pattern.fullmatch(candidate) is None:
                raise ValueError("completion is not exactly one numeric value")
            return candidate
        for pattern in self._answer_patterns:
            matches = pattern.findall(value)
            if len(matches) == 1:
                candidate_matches = self._number_pattern.findall(matches[0])
                if len(candidate_matches) == 1:
                    return candidate_matches[0]
                raise ValueError("final-answer marker contains an ambiguous numeric value")
            if len(matches) > 1:
                raise ValueError("multiple final-answer markers found")
        matches = self._number_pattern.findall(value)
        if len(matches) != 1:
            raise ValueError("expected exactly one numeric value or final-answer marker")
        return matches[0]

    @staticmethod
    def _literal_as_float(value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError("boolean is not a numeric answer")
        if isinstance(value, (int, float)):
            result = float(value)
        elif isinstance(value, str):
            candidate = value.strip().replace(",", "").replace(" ", "")
            if NumericAnswerVerifier._number_pattern.fullmatch(candidate) is None:
                raise ValueError("numeric reference is not exactly one numeric value")
            percent = candidate.endswith("%")
            if percent:
                candidate = candidate[:-1]
            result = float(Fraction(candidate)) if "/" in candidate else float(candidate)
            if percent:
                result /= 100.0
        else:
            raise TypeError("numeric reference must be a number or string")
        if not math.isfinite(result):
            raise ValueError("numeric answer must be finite")
        return result

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        try:
            expected = self._literal_as_float(sample.reference)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return VerificationResult.unscorable(
                self.name,
                VerificationStatus.INVALID_REFERENCE,
                str(exc),
            )
        try:
            predicted_text = self._candidate(sample.completion)
            predicted = self._literal_as_float(predicted_text)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return VerificationResult(
                verifier=self.name,
                status=VerificationStatus.FORMAT_FAIL,
                score=0.0,
                passed=False,
                error=str(exc),
            )
        passed = math.isclose(
            predicted,
            expected,
            abs_tol=self.absolute_tolerance,
            rel_tol=self.relative_tolerance,
        )
        return VerificationResult(
            verifier=self.name,
            status=VerificationStatus.PASS if passed else VerificationStatus.WRONG,
            score=float(passed),
            passed=passed,
            details={
                "extracted": predicted_text,
                "predicted": predicted,
                "expected": expected,
                "absolute_error": abs(predicted - expected),
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "extraction": self.extraction,
        }


def _json_leaf_score(expected: Any, actual: Any) -> tuple[int, int]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return 0, max(1, sum(_json_leaf_score(value, None)[1] for value in expected.values()))
        matched = 0
        total = 0
        for key, value in expected.items():
            child_matched, child_total = _json_leaf_score(value, actual.get(key, object()))
            matched += child_matched
            total += child_total
        return matched, max(1, total)
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            return 0, max(1, sum(_json_leaf_score(value, None)[1] for value in expected))
        matched = 0
        total = 0
        for index, value in enumerate(expected):
            child_actual = actual[index] if index < len(actual) else object()
            child_matched, child_total = _json_leaf_score(value, child_actual)
            matched += child_matched
            total += child_total
        return matched, max(1, total)
    same_type = type(expected) is type(actual)
    return (1, 1) if same_type and expected == actual else (0, 1)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_depth(value: Any) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(child) for child in value.values()), default=0)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return 1 + max((_json_depth(child) for child in value), default=0)
    return 0


class JsonSubsetVerifier(Verifier):
    """Parse strict JSON and score how much of the reference structure matches."""

    def __init__(
        self,
        *,
        name: str = "json_subset",
        version: str = "1",
        allow_markdown_fence: bool = False,
        allow_extra: bool = True,
        max_bytes: int = 100_000,
        max_depth: int = 64,
    ) -> None:
        self.name = name
        self.version = version
        self.allow_markdown_fence = allow_markdown_fence
        self.allow_extra = allow_extra
        if max_bytes < 1 or max_depth < 1:
            raise ValueError("JSON limits must be positive")
        self.max_bytes = max_bytes
        self.max_depth = max_depth

    def _text(self, completion: str) -> str:
        text = completion.strip()
        if self.allow_markdown_fence and text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        return text

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        expected = sample.reference
        if not isinstance(expected, (Mapping, tuple)):
            return VerificationResult.unscorable(
                self.name,
                VerificationStatus.INVALID_REFERENCE,
                "JSON-subset reference must be an object or array",
            )
        if _json_depth(expected) > self.max_depth:
            return VerificationResult.unscorable(
                self.name,
                VerificationStatus.INVALID_REFERENCE,
                "JSON-subset reference exceeds max_depth",
            )
        text = self._text(sample.completion)
        if len(text.encode("utf-8")) > self.max_bytes:
            return VerificationResult(
                verifier=self.name,
                status=VerificationStatus.FORMAT_FAIL,
                score=0.0,
                passed=False,
                error="JSON completion exceeds max_bytes",
            )
        try:
            actual = json.loads(
                text,
                object_pairs_hook=_no_duplicate_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return VerificationResult(
                verifier=self.name,
                status=VerificationStatus.FORMAT_FAIL,
                score=0.0,
                passed=False,
                error=str(exc),
            )
        if _json_depth(actual) > self.max_depth:
            return VerificationResult(
                verifier=self.name,
                status=VerificationStatus.FORMAT_FAIL,
                score=0.0,
                passed=False,
                error="JSON completion exceeds max_depth",
            )
        matched, total = _json_leaf_score(expected, actual)
        subset_passed = matched == total
        exact_structure = actual == expected
        passed = subset_passed and (self.allow_extra or exact_structure)
        return VerificationResult(
            verifier=self.name,
            status=VerificationStatus.PASS if passed else VerificationStatus.WRONG,
            score=matched / total,
            passed=passed,
            details={
                "matched_leaves": matched,
                "total_leaves": total,
                "exact_structure": exact_structure,
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "allow_markdown_fence": self.allow_markdown_fence,
            "allow_extra": self.allow_extra,
            "max_bytes": self.max_bytes,
            "max_depth": self.max_depth,
        }
