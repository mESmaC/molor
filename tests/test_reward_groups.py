from molor.rewards import (
    EvaluationSample,
    RewardContract,
    RewardEngine,
    RewardGroupStatus,
    RewardTerm,
    VerificationResult,
    VerificationStatus,
    Verifier,
    summarize_reward_group,
)


class ScoreVerifier(Verifier):
    name = "score"
    version = "1"

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        score = float(sample.reference)
        return VerificationResult(
            verifier=self.name,
            status=VerificationStatus.PASS if score == 1.0 else VerificationStatus.WRONG,
            score=score,
            passed=score == 1.0,
        )


def report(score: float):
    engine = RewardEngine(
        RewardContract("group", "1", (RewardTerm("score", 1.0),)),
        [ScoreVerifier()],
    )
    return engine.evaluate(EvaluationSample(f"case-{score}", "p", "c", score))


def masked_report():
    engine = RewardEngine(
        RewardContract("group", "1", (RewardTerm("score", 1.0),)),
        [ScoreVerifier()],
    )
    return engine.evaluate(EvaluationSample("case-masked", "p", "c", "not-a-number"))


def test_group_with_reward_variance_is_usable() -> None:
    summary = summarize_reward_group([report(0.0), report(1.0), report(0.5)])

    assert summary.status is RewardGroupStatus.SIGNAL
    assert summary.usable_for_relative_update
    assert summary.mean == 0.5


def test_zero_variance_group_is_preserved_but_not_used_for_relative_update() -> None:
    summary = summarize_reward_group([report(1.0), report(1.0)])

    assert summary.status is RewardGroupStatus.ZERO_VARIANCE
    assert not summary.usable_for_relative_update


def test_masked_rewards_are_not_converted_to_zero() -> None:
    scorable = report(0.5)
    masked = masked_report()

    summary = summarize_reward_group([scorable, masked])

    assert summary.status is RewardGroupStatus.INSUFFICIENT
    assert summary.scorable_count == 1
    assert summary.masked_count == 1


def test_all_masked_group_has_no_statistics() -> None:
    masked = masked_report()

    summary = summarize_reward_group([masked, masked])

    assert summary.status is RewardGroupStatus.ALL_MASKED
    assert summary.mean is None
