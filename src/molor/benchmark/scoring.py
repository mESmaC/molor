"""Deterministic, inspectable scorers for the bundled benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence


SCORER_VERSION = "molor.deterministic-scorers.v1.2"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    passed: bool
    extracted: object | None
    details: dict[str, object]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and from zero to one")

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "passed": self.passed,
            "extracted": self.extracted,
            "details": self.details,
        }


class ScoreSpecError(ValueError):
    """The scorer declaration is invalid."""


_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?\d+\s*/\s*\d+"
    r"|(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?%?"
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScoreSpecError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, label: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScoreSpecError(f"{label} must be an array")
    return list(value)


def _strings(value: object, label: str, *, nonempty: bool = True) -> list[str]:
    items = _sequence(value, label)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ScoreSpecError(f"{label} must contain non-empty strings")
    if nonempty and not items:
        raise ScoreSpecError(f"{label} must not be empty")
    return [str(item) for item in items]


def visible_answer(text: str) -> str:
    """Remove complete Qwen reasoning blocks before deterministic answer parsing."""

    return _THINK.sub("", text).strip()


def normalize_text(
    text: str,
    *,
    casefold: bool = True,
    strip_punctuation: bool = False,
) -> str:
    value = unicodedata.normalize("NFKC", visible_answer(text))
    value = " ".join(value.split())
    if strip_punctuation:
        value = "".join(
            character for character in value if not unicodedata.category(character).startswith("P")
        )
        value = " ".join(value.split())
    return value.casefold() if casefold else value


def _result(score: float, extracted: object | None, **details: object) -> ScoreResult:
    bounded = min(1.0, max(0.0, float(score)))
    return ScoreResult(
        score=bounded,
        passed=math.isclose(bounded, 1.0, abs_tol=1e-12),
        extracted=extracted,
        details=dict(details),
    )


def _score_exact(response: str, spec: Mapping[str, object]) -> ScoreResult:
    raw_answers = spec.get("answers")
    if raw_answers is None and "answer" in spec:
        raw_answers = [spec["answer"]]
    answers = _strings(raw_answers, "exact.answers")
    casefold = bool(spec.get("casefold", True))
    strip_punctuation = bool(spec.get("strip_punctuation", True))
    extracted = normalize_text(
        response,
        casefold=casefold,
        strip_punctuation=strip_punctuation,
    )
    expected = [
        normalize_text(answer, casefold=casefold, strip_punctuation=strip_punctuation)
        for answer in answers
    ]
    return _result(float(extracted in expected), extracted, expected=expected)


def _extract_choice(response: str, labels: Sequence[str]) -> str | None:
    answer = visible_answer(response).strip()
    label_set = {label.upper() for label in labels}
    patterns = (
        r"(?:final\s+answer|answer|choice)\s*(?:is|:|-)?\s*[\[(]?([A-Za-z0-9]+)[\])]?\s*[.!]?\s*$",
        r"^\s*[\[(]?([A-Za-z0-9]+)[\])]?\s*[.!]?\s*$",
        r"(?:^|\n)\s*[\[(]?([A-Za-z0-9]+)[\])]?\s*(?:[.)]|$)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
        for raw in reversed(matches):
            candidate = str(raw).upper()
            if candidate in label_set:
                return candidate
    return None


def _score_choice(response: str, spec: Mapping[str, object]) -> ScoreResult:
    expected = str(spec.get("answer") or "").strip().upper()
    if not expected:
        raise ScoreSpecError("choice.answer is required")
    raw_labels = spec.get("labels", ["A", "B", "C", "D"])
    labels = [label.upper() for label in _strings(raw_labels, "choice.labels")]
    if expected not in labels:
        raise ScoreSpecError("choice.answer must occur in choice.labels")
    extracted = _extract_choice(response, labels)
    return _result(float(extracted == expected), extracted, expected=expected)


def _parse_number(raw: str, *, percent_mode: str) -> float:
    value = raw.strip().replace(",", "")
    percent = value.endswith("%")
    if percent:
        if percent_mode == "reject":
            raise ValueError("percentage notation is not allowed")
        value = value[:-1]
    if "/" in value:
        parsed = float(Fraction(value.replace(" ", "")))
    else:
        parsed = float(value)
    if not percent:
        return parsed
    return parsed / 100.0 if percent_mode == "fraction" else parsed


def _extract_number(response: str, *, percent_mode: str) -> tuple[float | None, str | None]:
    answer = visible_answer(response)
    preferred = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\$?\s*(" + _NUMBER.pattern + r")",
        answer,
        flags=re.IGNORECASE,
    )
    candidates = preferred or _NUMBER.findall(answer)
    for raw in reversed(candidates):
        try:
            return _parse_number(raw, percent_mode=percent_mode), raw
        except (ValueError, ZeroDivisionError):
            continue
    return None, None


def _score_numeric(response: str, spec: Mapping[str, object]) -> ScoreResult:
    raw_expected = spec.get("answer")
    if isinstance(raw_expected, bool) or not isinstance(raw_expected, (int, float)):
        raise ScoreSpecError("numeric.answer must be a number")
    expected = float(raw_expected)
    atol = float(spec.get("atol", 1e-9))
    rtol = float(spec.get("rtol", 1e-9))
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ScoreSpecError("numeric tolerances must be non-negative finite numbers")
    percent_mode = str(spec.get("percent_mode", "reject")).strip().lower()
    if percent_mode not in {"reject", "fraction", "number"}:
        raise ScoreSpecError("numeric.percent_mode must be reject, fraction, or number")
    extracted, raw = _extract_number(response, percent_mode=percent_mode)
    passed = extracted is not None and math.isclose(extracted, expected, abs_tol=atol, rel_tol=rtol)
    return _result(
        float(passed),
        extracted,
        expected=expected,
        raw=raw,
        atol=atol,
        rtol=rtol,
        percent_mode=percent_mode,
    )


def _score_contains(response: str, spec: Mapping[str, object]) -> ScoreResult:
    casefold = bool(spec.get("casefold", True))
    text = normalize_text(response, casefold=casefold, strip_punctuation=False)
    required = _strings(spec.get("all", []), "contains.all", nonempty=False)
    any_terms = _strings(spec.get("any", []), "contains.any", nonempty=False)
    forbidden = _strings(spec.get("none", []), "contains.none", nonempty=False)
    if not required and not any_terms and not forbidden:
        raise ScoreSpecError("contains needs at least one of all, any, or none")

    def normalize(value: str) -> str:
        return normalize_text(value, casefold=casefold, strip_punctuation=False)

    required_hits = {term: normalize(term) in text for term in required}
    any_hits = {term: normalize(term) in text for term in any_terms}
    forbidden_hits = {term: normalize(term) in text for term in forbidden}
    checks: list[bool] = list(required_hits.values())
    if any_terms:
        checks.append(any(any_hits.values()))
    if forbidden:
        checks.append(not any(forbidden_hits.values()))
    score = sum(checks) / len(checks) if checks else 0.0
    return _result(
        score,
        text,
        required_hits=required_hits,
        any_hits=any_hits,
        forbidden_hits=forbidden_hits,
    )


def _extract_json(response: str) -> object | None:
    answer = visible_answer(response).strip()
    candidates = [answer]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced + candidates
    first_object, last_object = answer.find("{"), answer.rfind("}")
    if 0 <= first_object < last_object:
        candidates.append(answer[first_object : last_object + 1])
    first_array, last_array = answer.find("["), answer.rfind("]")
    if 0 <= first_array < last_array:
        candidates.append(answer[first_array : last_array + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_subset(actual: object, expected: object) -> tuple[int, int]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return 0, max(1, len(expected))
        hit = 0
        total = 0
        for key, expected_value in expected.items():
            child_hit, child_total = _json_subset(actual.get(key), expected_value)
            hit += child_hit
            total += child_total
        return hit, max(total, 1)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return 0, max(1, len(expected))
        pairs = [_json_subset(a, e) for a, e in zip(actual, expected, strict=True)]
        return sum(hit for hit, _ in pairs), max(1, sum(total for _, total in pairs))
    return (1, 1) if actual == expected else (0, 1)


def _score_json(response: str, spec: Mapping[str, object]) -> ScoreResult:
    expected = spec.get("expected")
    if expected is None:
        raise ScoreSpecError("json.expected is required")
    answer = visible_answer(response).strip()
    allow_fence = bool(spec.get("allow_fence", True))
    minified = bool(spec.get("minified", False))
    fenced = bool(re.search(r"```", answer))
    if allow_fence:
        extracted = _extract_json(response)
    else:
        try:
            extracted = json.loads(answer)
        except json.JSONDecodeError:
            extracted = None
    if extracted is None:
        return _result(
            0.0,
            None,
            expected=expected,
            valid_json=False,
            allow_fence=allow_fence,
            fenced=fenced,
            minified=minified,
        )
    hit, total = _json_subset(extracted, expected)
    exact = bool(spec.get("exact", True))
    score = float(extracted == expected) if exact else hit / total
    forbidden_keys = _strings(spec.get("forbidden_keys", []), "json.forbidden_keys", nonempty=False)
    forbidden_present = sorted(
        key for key in forbidden_keys if isinstance(extracted, Mapping) and key in extracted
    )
    if forbidden_present:
        score = 0.0
    canonical_minified = json.dumps(extracted, ensure_ascii=False, separators=(",", ":"))
    minified_match = answer == canonical_minified
    if fenced and not allow_fence:
        score = 0.0
    if minified and not minified_match:
        score = 0.0
    return _result(
        score,
        extracted,
        expected=expected,
        valid_json=True,
        exact=exact,
        allow_fence=allow_fence,
        fenced=fenced,
        minified=minified,
        minified_match=minified_match,
        forbidden_present=forbidden_present,
    )


def _score_regex(response: str, spec: Mapping[str, object]) -> ScoreResult:
    patterns = _strings(spec.get("all", []), "regex.all", nonempty=False)
    forbidden = _strings(spec.get("none", []), "regex.none", nonempty=False)
    if not patterns and not forbidden:
        raise ScoreSpecError("regex needs all or none patterns")
    flags = re.IGNORECASE | re.MULTILINE if bool(spec.get("ignore_case", True)) else re.MULTILINE
    answer = visible_answer(response)
    hits = {pattern: bool(re.search(pattern, answer, flags=flags)) for pattern in patterns}
    forbidden_hits = {
        pattern: bool(re.search(pattern, answer, flags=flags)) for pattern in forbidden
    }
    checks = list(hits.values()) + [not value for value in forbidden_hits.values()]
    return _result(
        sum(checks) / len(checks),
        answer,
        pattern_hits=hits,
        forbidden_hits=forbidden_hits,
    )


def _normalize_command(value: str) -> str:
    text = normalize_text(value, casefold=False, strip_punctuation=False)
    text = re.sub(r"\s*([|;&])\s*", r" \1 ", text)
    return " ".join(text.split())


def _score_command(response: str, spec: Mapping[str, object]) -> ScoreResult:
    commands = _strings(spec.get("answers"), "command.answers")
    answer = visible_answer(response).strip()
    only_command = bool(spec.get("only_command", True))
    invalid_wrapper = only_command and (
        "\n" in answer or "\r" in answer or "`" in answer or answer.lower().startswith("command:")
    )
    extracted = _normalize_command(answer)
    expected = [_normalize_command(command) for command in commands]
    return _result(
        float(not invalid_wrapper and extracted in expected),
        extracted,
        expected=expected,
        only_command=only_command,
        invalid_wrapper=invalid_wrapper,
    )


def _score_composite(response: str, spec: Mapping[str, object]) -> ScoreResult:
    raw_components = _sequence(spec.get("components"), "composite.components")
    if not raw_components:
        raise ScoreSpecError("composite.components must not be empty")
    results: list[dict[str, object]] = []
    weighted = 0.0
    total_weight = 0.0
    for index, raw_component in enumerate(raw_components):
        component = _mapping(raw_component, f"composite.components[{index}]")
        weight = component.pop("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise ScoreSpecError(f"composite.components[{index}].weight must be positive")
        result = score_response(response, component)
        weighted += result.score * float(weight)
        total_weight += float(weight)
        results.append({"weight": float(weight), **result.to_dict()})
    score = weighted / total_weight
    return _result(score, visible_answer(response), components=results)


def score_response(response: str, raw_spec: Mapping[str, object]) -> ScoreResult:
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    spec = _mapping(raw_spec, "scorer")
    scorer_type = str(spec.pop("type", "")).strip().lower()
    scorers = {
        "choice": _score_choice,
        "command": _score_command,
        "composite": _score_composite,
        "contains": _score_contains,
        "exact": _score_exact,
        "json": _score_json,
        "numeric": _score_numeric,
        "regex": _score_regex,
    }
    scorer = scorers.get(scorer_type)
    if scorer is None:
        raise ScoreSpecError(
            f"unsupported scorer type {scorer_type!r}; choose one of: " + ", ".join(sorted(scorers))
        )
    result = scorer(response, spec)
    return ScoreResult(
        score=result.score,
        passed=result.passed,
        extracted=result.extracted,
        details={"scorer_type": scorer_type, **result.details},
    )


def validate_score_spec(raw_spec: Mapping[str, object]) -> None:
    """Validate a scorer declaration without launching an external verifier."""

    spec = _mapping(raw_spec, "scorer")
    scorer_type = str(spec.get("type", "")).strip().lower()
    if scorer_type == "external":
        unknown = set(spec).difference({"type", "verifier"})
        if unknown:
            raise ScoreSpecError(
                "external scorer has unknown fields: " + ", ".join(sorted(unknown))
            )
        verifier = spec.get("verifier")
        if not isinstance(verifier, str) or _IDENTIFIER.fullmatch(verifier) is None:
            raise ScoreSpecError("external.verifier must be a benchmark identifier")
        return
    score_response("", spec)


__all__ = [
    "SCORER_VERSION",
    "ScoreResult",
    "ScoreSpecError",
    "normalize_text",
    "score_response",
    "validate_score_spec",
    "visible_answer",
]
