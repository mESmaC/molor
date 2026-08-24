from dataclasses import replace
import json

import pytest

from molor.rewards import (
    CostTerm,
    EvaluationSample,
    RewardContract,
    RewardEngine,
    RewardTerm,
    VerificationResult,
    VerificationStatus,
    Verifier,
)


class StaticVerifier(Verifier):
    def __init__(
        self,
        name: str,
        score: float | None,
        status: VerificationStatus,
        *,
        version: str = "1",
    ) -> None:
        self.name = name
        self.version = version
        self.score = score
        self.status = status

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        return VerificationResult(
            verifier=self.name,
            status=self.status,
            score=self.score,
            passed=self.status is VerificationStatus.PASS,
        )


class ExplodingVerifier(Verifier):
    name = "explodes"
    version = "1"

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        raise RuntimeError("broken verifier")


def base_sample() -> EvaluationSample:
    return EvaluationSample("case-1", "prompt", "completion", "reference")


def test_reward_contract_combines_quality_and_measured_cost() -> None:
    contract = RewardContract(
        name="code",
        version="1",
        terms=(RewardTerm("outcome", 0.9), RewardTerm("format", 0.1)),
        costs=(CostTerm("latency", 0.2),),
    )
    engine = RewardEngine(
        contract,
        [
            StaticVerifier("outcome", 0.8, VerificationStatus.WRONG),
            StaticVerifier("format", 1.0, VerificationStatus.PASS),
        ],
    )

    report = engine.evaluate(base_sample(), normalized_costs={"latency": 0.5})

    assert report.quality_reward == pytest.approx(0.82)
    assert report.cost_penalty == pytest.approx(0.1)
    assert report.reward == pytest.approx(0.72)
    assert not report.masked


def test_failed_hard_gate_uses_fixed_floor() -> None:
    contract = RewardContract(
        name="structured",
        version="1",
        terms=(RewardTerm("syntax", 1.0, gate=True),),
        failure_reward=-0.75,
    )
    engine = RewardEngine(
        contract,
        [StaticVerifier("syntax", 0.4, VerificationStatus.FORMAT_FAIL)],
    )

    report = engine.evaluate(base_sample())

    assert report.reward == -0.75
    assert report.gate_failures == ("syntax",)


def test_unscorable_result_masks_reward_instead_of_becoming_zero() -> None:
    contract = RewardContract(
        name="math",
        version="1",
        terms=(RewardTerm("answer", 1.0),),
    )
    engine = RewardEngine(
        contract,
        [StaticVerifier("answer", None, VerificationStatus.INVALID_REFERENCE)],
    )

    report = engine.evaluate(base_sample())

    assert report.masked
    assert report.reward is None
    assert report.quality_reward is None
    assert report.mask_reasons == ("answer:invalid_reference",)


def test_verifier_exception_is_masked_and_recorded() -> None:
    contract = RewardContract(
        name="broken",
        version="1",
        terms=(RewardTerm("explodes", 1.0),),
    )

    report = RewardEngine(contract, [ExplodingVerifier()]).evaluate(base_sample())

    result = report.verifier_results["explodes"]
    assert report.masked
    assert result.status is VerificationStatus.VERIFIER_ERROR
    assert "RuntimeError" in (result.error or "")
    assert result.duration_ms is not None


def test_required_cost_masks_reward_instead_of_silently_becoming_zero() -> None:
    contract = RewardContract(
        name="costed",
        version="1",
        terms=(RewardTerm("quality", 1.0),),
        costs=(CostTerm("cold_load", 0.1),),
    )
    engine = RewardEngine(
        contract,
        [StaticVerifier("quality", 1.0, VerificationStatus.PASS)],
    )

    report = engine.evaluate(base_sample())

    assert report.masked
    assert report.reward is None
    assert report.mask_reasons == ("cost:cold_load:missing",)


def test_fingerprint_changes_with_verifier_version_and_report_is_json() -> None:
    contract = RewardContract(
        name="versioned",
        version="1",
        terms=(RewardTerm("quality", 1.0),),
    )
    first = RewardEngine(
        contract,
        [StaticVerifier("quality", 1.0, VerificationStatus.PASS, version="1")],
    )
    second = RewardEngine(
        contract,
        [StaticVerifier("quality", 1.0, VerificationStatus.PASS, version="2")],
    )

    report = first.evaluate(base_sample())

    assert first.fingerprint != second.fingerprint
    assert len(first.verifiers["quality"].manifest()["implementation_sha256"]) == 64
    assert len(report.sample_fingerprint) == 64
    assert len(report.evaluation_fingerprint) == 64
    json.dumps(report.to_dict(), allow_nan=False)


def test_quality_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        RewardContract(
            name="bad",
            version="1",
            terms=(RewardTerm("quality", 0.5),),
        )


def test_reward_report_validates_public_construction_invariants() -> None:
    contract = RewardContract(
        name="strict",
        version="1",
        terms=(RewardTerm("quality", 1.0),),
    )
    report = RewardEngine(
        contract,
        [StaticVerifier("quality", 1.0, VerificationStatus.PASS)],
    ).evaluate(base_sample())

    with pytest.raises(ValueError, match="masked reports"):
        replace(report, masked=True, mask_reasons=("infra",))
    with pytest.raises(ValueError, match="evaluation_fingerprint"):
        replace(report, evaluation_fingerprint="0" * 64)


def test_reward_engine_rejects_boolean_normalized_costs() -> None:
    contract = RewardContract(
        name="costed",
        version="1",
        terms=(RewardTerm("quality", 1.0),),
        costs=(CostTerm("latency", 0.1),),
    )
    reward_engine = RewardEngine(
        contract,
        [StaticVerifier("quality", 1.0, VerificationStatus.PASS)],
    )

    with pytest.raises(TypeError, match="real number"):
        reward_engine.evaluate(base_sample(), normalized_costs={"latency": True})


def test_reward_report_cannot_leave_an_unscorable_result_unmasked() -> None:
    contract = RewardContract(
        name="strict",
        version="1",
        terms=(RewardTerm("quality", 1.0),),
    )
    report = RewardEngine(
        contract,
        [StaticVerifier("quality", 1.0, VerificationStatus.PASS)],
    ).evaluate(base_sample())
    bad_result = VerificationResult.unscorable(
        "quality",
        VerificationStatus.INFRA_ERROR,
        "worker lost",
    )

    with pytest.raises(ValueError, match="must be masked"):
        replace(report, verifier_results={"quality": bad_result})
