import sys
from pathlib import Path

from molor.rewards import (
    EvaluationSample,
    ExactMatchVerifier,
    ExternalCommandVerifier,
    JsonSubsetVerifier,
    NumericAnswerVerifier,
    RegexVerifier,
    VerificationStatus,
)


def sample(completion: str, reference: object) -> EvaluationSample:
    return EvaluationSample("case-1", "prompt", completion, reference)


def test_exact_match_is_strict_by_default() -> None:
    result = ExactMatchVerifier().verify(sample(" Answer ", "answer"))

    assert result.status is VerificationStatus.WRONG
    assert result.score == 0.0


def test_exact_match_normalization_is_explicit_and_manifested() -> None:
    verifier = ExactMatchVerifier(
        case_sensitive=False,
        collapse_whitespace=True,
        strip=True,
        unicode_form="NFKC",
    )

    result = verifier.verify(sample("  Ａnswer\n now ", "answer now"))

    assert result.status is VerificationStatus.PASS
    assert verifier.manifest()["unicode_form"] == "NFKC"


def test_numeric_whole_mode_rejects_explanatory_text() -> None:
    result = NumericAnswerVerifier().verify(sample("The answer is 42", 42))

    assert result.status is VerificationStatus.FORMAT_FAIL
    assert result.score == 0.0


def test_numeric_final_mode_handles_tag_and_fraction() -> None:
    verifier = NumericAnswerVerifier(extraction="final")

    result = verifier.verify(sample("Reasoning... <answer>1 / 2</answer>", 0.5))

    assert result.status is VerificationStatus.PASS


def test_numeric_final_mode_rejects_ambiguous_answers() -> None:
    verifier = NumericAnswerVerifier(extraction="final")

    result = verifier.verify(sample("Candidates are 1 and 2", 2))

    assert result.status is VerificationStatus.FORMAT_FAIL


def test_invalid_numeric_reference_is_unscorable() -> None:
    result = NumericAnswerVerifier().verify(sample("1", "not a number"))

    assert result.status is VerificationStatus.INVALID_REFERENCE
    assert result.score is None


def test_json_subset_returns_partial_quality() -> None:
    verifier = JsonSubsetVerifier()

    result = verifier.verify(
        sample(
            '{"tool":"search","args":{"q":"wrong"}}',
            {
                "tool": "search",
                "args": {"q": "correct"},
            },
        )
    )

    assert result.status is VerificationStatus.WRONG
    assert result.score == 0.5


def test_json_rejects_duplicates_nonstandard_constants_and_type_coercion() -> None:
    verifier = JsonSubsetVerifier()

    duplicate = verifier.verify(sample('{"x":1,"x":1}', {"x": 1}))
    nonstandard = verifier.verify(sample('{"x":NaN}', {"x": 1}))
    wrong_type = verifier.verify(sample('{"x":1}', {"x": True}))

    assert duplicate.status is VerificationStatus.FORMAT_FAIL
    assert nonstandard.status is VerificationStatus.FORMAT_FAIL
    assert wrong_type.status is VerificationStatus.WRONG


def test_json_extra_keys_can_be_forbidden() -> None:
    verifier = JsonSubsetVerifier(allow_extra=False)

    result = verifier.verify(sample('{"x":1,"extra":2}', {"x": 1}))

    assert result.status is VerificationStatus.WRONG
    assert result.score == 1.0


def test_regex_verifier_is_a_separate_format_signal() -> None:
    verifier = RegexVerifier(r"<answer>.*</answer>", full_match=True)

    assert verifier.verify(sample("<answer>ok</answer>", None)).passed
    assert not verifier.verify(sample("prefix <answer>ok</answer>", None)).passed


def test_external_verifier_uses_versioned_data_protocol() -> None:
    fixture = Path(__file__).parent / "fixtures" / "external_verifier.py"
    verifier = ExternalCommandVerifier(
        [sys.executable, str(fixture)],
        name="external_stub",
        version="1",
        runner_fingerprint="sha256:test-runner",
    )

    result = verifier.verify(sample("expected", "expected"))

    assert result.status is VerificationStatus.PASS
    assert result.details == {"stub": True}
    assert verifier.manifest()["runner_fingerprint"] == "sha256:test-runner"
