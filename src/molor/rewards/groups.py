"""Diagnostics for relative-policy reward groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import statistics
from typing import Iterable

from molor.rewards.contract import RewardReport


class RewardGroupStatus(StrEnum):
    """Whether a rollout group contains a usable relative-policy signal."""

    SIGNAL = "signal"
    ZERO_VARIANCE = "zero_variance"
    INSUFFICIENT = "insufficient"
    ALL_MASKED = "all_masked"


@dataclass(frozen=True, slots=True)
class RewardGroupSummary:
    status: RewardGroupStatus
    total_count: int
    scorable_count: int
    masked_count: int
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None

    @property
    def usable_for_relative_update(self) -> bool:
        return self.status is RewardGroupStatus.SIGNAL

    def to_dict(self) -> dict[str, int | float | bool | str | None]:
        return {
            "status": self.status.value,
            "total_count": self.total_count,
            "scorable_count": self.scorable_count,
            "masked_count": self.masked_count,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "usable_for_relative_update": self.usable_for_relative_update,
        }


def summarize_reward_group(
    reports: Iterable[RewardReport],
    *,
    zero_tolerance: float = 1e-12,
) -> RewardGroupSummary:
    """Classify a group without treating masked rollouts as zero reward."""

    if not math.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    report_list = list(reports)
    rewards = [report.reward for report in report_list if report.reward is not None]
    masked_count = len(report_list) - len(rewards)

    if not rewards:
        return RewardGroupSummary(
            status=RewardGroupStatus.ALL_MASKED,
            total_count=len(report_list),
            scorable_count=0,
            masked_count=masked_count,
            mean=None,
            standard_deviation=None,
            minimum=None,
            maximum=None,
        )

    mean = statistics.fmean(rewards)
    standard_deviation = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if len(rewards) < 2:
        status = RewardGroupStatus.INSUFFICIENT
    elif standard_deviation <= zero_tolerance:
        status = RewardGroupStatus.ZERO_VARIANCE
    else:
        status = RewardGroupStatus.SIGNAL

    return RewardGroupSummary(
        status=status,
        total_count=len(report_list),
        scorable_count=len(rewards),
        masked_count=masked_count,
        mean=mean,
        standard_deviation=standard_deviation,
        minimum=min(rewards),
        maximum=max(rewards),
    )
