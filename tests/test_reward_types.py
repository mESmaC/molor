import json
import math

import pytest

from molor.rewards import EvaluationSample, VerificationResult, VerificationStatus


def test_sample_fingerprint_is_stable_across_metadata_order() -> None:
    left = EvaluationSample("case-1", "p", "c", 4, {"a": 1, "b": 2})
    right = EvaluationSample("case-1", "p", "c", 4, {"b": 2, "a": 1})

    assert left.fingerprint == right.fingerprint


def test_prompt_fingerprint_excludes_rollout_and_completion() -> None:
    left = EvaluationSample("rollout-a", "p", "first", {"answer": 4})
    right = EvaluationSample("rollout-b", "p", "second", {"answer": 4})
    changed_reference = EvaluationSample("rollout-c", "p", "second", {"answer": 5})

    assert left.fingerprint != right.fingerprint
    assert left.prompt_fingerprint == right.prompt_fingerprint
    assert left.prompt_fingerprint != changed_reference.prompt_fingerprint


def test_sample_rejects_non_json_nan() -> None:
    with pytest.raises(ValueError):
        EvaluationSample("case-1", "p", "c", math.nan)


def test_sample_rejects_non_string_json_keys() -> None:
    with pytest.raises(TypeError, match="non-string object key"):
        EvaluationSample("case-1", "p", "c", {1: "bad"})


def test_sample_and_result_are_deeply_immutable() -> None:
    reference = {"nested": [1]}
    metadata = {"nested": {"value": 2}}
    details = {"nested": [3]}
    sample = EvaluationSample("case-1", "p", "c", reference, metadata)
    result = VerificationResult(
        "check",
        VerificationStatus.PASS,
        1.0,
        True,
        details=details,
    )
    sample_fingerprint = sample.fingerprint

    reference["nested"].append(9)
    metadata["nested"]["value"] = 7
    details["nested"].append(8)

    assert sample.fingerprint == sample_fingerprint
    assert sample.to_dict()["reference"] == {"nested": [1]}
    assert result.to_dict()["details"] == {"nested": [3]}


def test_result_requires_score_only_for_scorable_statuses() -> None:
    with pytest.raises(ValueError, match="require a finite score"):
        VerificationResult("check", VerificationStatus.PASS, None, True)
    with pytest.raises(ValueError, match="score=None"):
        VerificationResult("check", VerificationStatus.INFRA_ERROR, 0.0, False)


def test_result_serializes_typed_status() -> None:
    result = VerificationResult(
        verifier="check",
        status=VerificationStatus.TEST_FAIL,
        score=0.25,
        passed=False,
        details={"passed_tests": 1, "total_tests": 4},
    )

    payload = result.to_dict()

    assert payload["status"] == "test_fail"
    assert json.loads(json.dumps(payload))["score"] == 0.25


def test_result_rejects_unbounded_or_non_json_details() -> None:
    with pytest.raises(TypeError, match="non-JSON"):
        VerificationResult(
            "check",
            VerificationStatus.PASS,
            1.0,
            True,
            details={"bad": object()},
        )


def test_result_rejects_boolean_score_and_duration() -> None:
    with pytest.raises(ValueError, match="finite score"):
        VerificationResult("check", VerificationStatus.PASS, True, True)
    with pytest.raises(ValueError, match="duration_ms"):
        VerificationResult(
            "check",
            VerificationStatus.PASS,
            1.0,
            True,
            duration_ms=True,
        )
