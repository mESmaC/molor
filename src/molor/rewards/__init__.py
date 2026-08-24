"""Deterministic verifier and reward-contract API."""

from molor.rewards.base import Verifier
from molor.rewards.contract import (
    CostTerm,
    RewardContract,
    RewardEngine,
    RewardReport,
    RewardTerm,
)
from molor.rewards.external import ExternalCommandVerifier
from molor.rewards.finance import FinanceCalculationVerifier
from molor.rewards.groups import RewardGroupStatus, RewardGroupSummary, summarize_reward_group
from molor.rewards.table import (
    CostNormalizationRecord,
    ForcedRouteRecord,
    OracleSelection,
    PromptRef,
    RewardMatrix,
    RewardTable,
    RewardTableEntry,
)
from molor.rewards.types import EvaluationSample, VerificationResult, VerificationStatus
from molor.rewards.verifiers import (
    ExactMatchVerifier,
    JsonSubsetVerifier,
    NumericAnswerVerifier,
    RegexVerifier,
)

__all__ = [
    "CostTerm",
    "CostNormalizationRecord",
    "EvaluationSample",
    "ExactMatchVerifier",
    "ExternalCommandVerifier",
    "FinanceCalculationVerifier",
    "ForcedRouteRecord",
    "JsonSubsetVerifier",
    "NumericAnswerVerifier",
    "OracleSelection",
    "PromptRef",
    "RegexVerifier",
    "RewardContract",
    "RewardEngine",
    "RewardGroupStatus",
    "RewardGroupSummary",
    "RewardReport",
    "RewardMatrix",
    "RewardTable",
    "RewardTableEntry",
    "RewardTerm",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
    "summarize_reward_group",
]
