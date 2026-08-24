"""Versioned composition of verifier outcomes and measured serving costs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from molor.identity import (
    fingerprint_json,
    freeze_json,
    thaw_json,
    validate_json_value,
    validate_sha256,
)
from molor.rewards.base import Verifier
from molor.rewards.types import EvaluationSample, VerificationResult, VerificationStatus


@dataclass(frozen=True, slots=True)
class RewardTerm:
    """A quality signal; gated terms can force the contract failure reward."""

    verifier: str
    weight: float
    gate: bool = False

    def __post_init__(self) -> None:
        if not self.verifier:
            raise ValueError("reward-term verifier must not be empty")
        if not isinstance(self.gate, bool):
            raise TypeError("reward-term gate must be boolean")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("reward-term weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class CostTerm:
    """A normalized measured cost subtracted from quality reward."""

    key: str
    weight: float
    required: bool = True

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("cost key must not be empty")
        if not isinstance(self.required, bool):
            raise TypeError("cost-term required must be boolean")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(self.weight)
            or self.weight < 0
        ):
            raise ValueError("cost weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RewardContract:
    """An immutable declaration of how verifier scores become scalar reward."""

    name: str
    version: str
    terms: tuple[RewardTerm, ...]
    costs: tuple[CostTerm, ...] = ()
    failure_reward: float = -1.0
    minimum_reward: float = -1.0
    maximum_reward: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("reward-contract name and version must not be empty")
        if not self.terms:
            raise ValueError("reward contract requires at least one quality term")
        verifier_names = [term.verifier for term in self.terms]
        if len(verifier_names) != len(set(verifier_names)):
            raise ValueError("a verifier may appear only once in a reward contract")
        cost_keys = [term.key for term in self.costs]
        if len(cost_keys) != len(set(cost_keys)):
            raise ValueError("a cost may appear only once in a reward contract")
        quality_weight = sum(term.weight for term in self.terms)
        if not math.isclose(quality_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("quality-term weights must sum to 1")
        bounds = (self.failure_reward, self.minimum_reward, self.maximum_reward)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in bounds
        ) or not all(math.isfinite(value) for value in bounds):
            raise ValueError("reward bounds and failure reward must be finite")
        if not self.minimum_reward <= self.failure_reward <= self.maximum_reward:
            raise ValueError("failure_reward must be inside the configured reward bounds")
        if self.minimum_reward >= self.maximum_reward:
            raise ValueError("minimum_reward must be less than maximum_reward")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "terms": [
                {"verifier": term.verifier, "weight": term.weight, "gate": term.gate}
                for term in self.terms
            ],
            "costs": [
                {"key": term.key, "weight": term.weight, "required": term.required}
                for term in self.costs
            ],
            "failure_reward": self.failure_reward,
            "minimum_reward": self.minimum_reward,
            "maximum_reward": self.maximum_reward,
        }


@dataclass(frozen=True, slots=True)
class RewardReport:
    """Replayable result of applying one reward contract to one sample."""

    sample_id: str
    contract: str
    contract_version: str
    contract_fingerprint: str
    sample_fingerprint: str
    prompt_fingerprint: str
    evaluation_fingerprint: str
    engine_manifest: Mapping[str, Any]
    reward: float | None
    quality_reward: float | None
    cost_penalty: float
    masked: bool
    mask_reasons: tuple[str, ...]
    gate_failures: tuple[str, ...]
    verifier_results: Mapping[str, VerificationResult]
    normalized_costs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("sample_id", "contract", "contract_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must not be empty")
        for field_name in (
            "contract_fingerprint",
            "sample_fingerprint",
            "prompt_fingerprint",
            "evaluation_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name),
            )
        expected_evaluation = hashlib.sha256(
            f"{self.sample_fingerprint}:{self.contract_fingerprint}".encode("ascii")
        ).hexdigest()
        if self.evaluation_fingerprint != expected_evaluation:
            raise ValueError("evaluation_fingerprint does not match sample and contract")
        validate_json_value(self.engine_manifest, "engine_manifest")
        manifest = thaw_json(freeze_json(self.engine_manifest))
        if not isinstance(manifest, dict):
            raise TypeError("engine_manifest must be a mapping")
        if fingerprint_json(manifest) != self.contract_fingerprint:
            raise ValueError("engine_manifest does not match contract_fingerprint")
        if not isinstance(self.masked, bool):
            raise TypeError("masked must be boolean")
        if (
            isinstance(self.cost_penalty, bool)
            or not isinstance(self.cost_penalty, (int, float))
            or not math.isfinite(self.cost_penalty)
            or self.cost_penalty < 0
        ):
            raise ValueError("cost_penalty must be finite and non-negative")
        object.__setattr__(self, "cost_penalty", float(self.cost_penalty))

        for field_name in ("reward", "quality_reward"):
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{field_name} must be a real number or None")
                if not math.isfinite(value):
                    raise ValueError(f"{field_name} must be finite")
                object.__setattr__(self, field_name, float(value))
        if self.quality_reward is not None and not 0.0 <= self.quality_reward <= 1.0:
            raise ValueError("quality_reward must be in [0, 1]")

        if isinstance(self.mask_reasons, (str, bytes)):
            raise TypeError("mask_reasons must be an iterable of strings")
        if isinstance(self.gate_failures, (str, bytes)):
            raise TypeError("gate_failures must be an iterable of strings")
        mask_reasons = tuple(self.mask_reasons)
        gate_failures = tuple(self.gate_failures)
        if any(not isinstance(reason, str) or not reason for reason in mask_reasons):
            raise ValueError("mask_reasons must contain non-empty strings")
        if any(not isinstance(name, str) or not name for name in gate_failures):
            raise ValueError("gate_failures must contain non-empty strings")
        if len(mask_reasons) != len(set(mask_reasons)):
            raise ValueError("mask_reasons must not contain duplicates")
        if len(gate_failures) != len(set(gate_failures)):
            raise ValueError("gate_failures must not contain duplicates")

        results = dict(self.verifier_results)
        if not results:
            raise ValueError("verifier_results must not be empty")
        for name, result in results.items():
            if not isinstance(name, str) or not name:
                raise ValueError("verifier_results keys must be non-empty strings")
            if not isinstance(result, VerificationResult):
                raise TypeError("verifier_results values must be VerificationResult objects")
            if result.verifier != name:
                raise ValueError("verifier_results key must match result.verifier")
        unknown_gates = set(gate_failures).difference(results)
        if unknown_gates:
            raise ValueError(f"gate_failures reference unknown verifiers: {sorted(unknown_gates)}")
        unscorable_results = [
            name for name, result in results.items() if not result.status.scorable
        ]
        if unscorable_results and not self.masked:
            raise ValueError(
                f"reports with unscorable verifier results must be masked: {unscorable_results!r}"
            )
        invalid_gate_failures = [
            name
            for name in gate_failures
            if not results[name].status.scorable or results[name].passed
        ]
        if invalid_gate_failures:
            raise ValueError("gate_failures must reference scorable failed verifier results")

        costs: dict[str, float] = {}
        for key, value in self.normalized_costs.items():
            if not isinstance(key, str) or not key:
                raise ValueError("normalized_costs keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("normalized costs must be real numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError("normalized costs must be finite and in [0, 1]")
            costs[key] = numeric

        if self.masked:
            if self.reward is not None or self.quality_reward is not None:
                raise ValueError("masked reports must use reward=None and quality_reward=None")
            if not mask_reasons:
                raise ValueError("masked reports require at least one mask reason")
        else:
            if self.reward is None or self.quality_reward is None:
                raise ValueError("unmasked reports require numeric reward and quality_reward")
            if mask_reasons:
                raise ValueError("unmasked reports must not contain mask reasons")

        object.__setattr__(self, "mask_reasons", mask_reasons)
        object.__setattr__(self, "gate_failures", gate_failures)
        object.__setattr__(self, "verifier_results", MappingProxyType(results))
        object.__setattr__(self, "normalized_costs", MappingProxyType(costs))
        object.__setattr__(self, "engine_manifest", freeze_json(manifest))
        self._validate_engine_semantics(manifest)

    def _validate_engine_semantics(self, manifest: dict[str, Any]) -> None:
        """Recompute every reward-affecting field from the embedded immutable engine spec."""

        if set(manifest) != {"schema_version", "contract", "verifiers"}:
            raise ValueError("engine_manifest has unknown or missing top-level fields")
        if manifest["schema_version"] != "molor.reward-engine.v1":
            raise ValueError("unsupported reward-engine manifest schema")
        contract_payload = manifest["contract"]
        verifier_payload = manifest["verifiers"]
        if not isinstance(contract_payload, dict) or not isinstance(verifier_payload, dict):
            raise TypeError("engine_manifest contract and verifiers must be objects")
        try:
            contract = RewardContract(
                name=contract_payload["name"],
                version=contract_payload["version"],
                terms=tuple(
                    RewardTerm(
                        verifier=term["verifier"],
                        weight=term["weight"],
                        gate=term["gate"],
                    )
                    for term in contract_payload["terms"]
                ),
                costs=tuple(
                    CostTerm(
                        key=term["key"],
                        weight=term["weight"],
                        required=term["required"],
                    )
                    for term in contract_payload["costs"]
                ),
                failure_reward=contract_payload["failure_reward"],
                minimum_reward=contract_payload["minimum_reward"],
                maximum_reward=contract_payload["maximum_reward"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid reward contract in engine_manifest") from exc
        if set(contract_payload) != set(contract.to_dict()):
            raise ValueError("reward contract manifest has unknown or missing fields")
        if contract.to_dict() != contract_payload:
            raise ValueError("reward contract manifest is not canonical")
        if self.contract != contract.name or self.contract_version != contract.version:
            raise ValueError("report contract name/version differs from engine_manifest")

        expected_verifiers = tuple(term.verifier for term in contract.terms)
        if set(verifier_payload) != set(expected_verifiers):
            raise ValueError("engine_manifest verifier set differs from its reward contract")
        if set(self.verifier_results) != set(expected_verifiers):
            raise ValueError("report verifier results differ from its reward contract")

        weighted_quality = 0.0
        mask_reasons: list[str] = []
        gate_failures: list[str] = []
        for term in contract.terms:
            result = self.verifier_results[term.verifier]
            if result.score is None:
                mask_reasons.append(f"{term.verifier}:{result.status.value}")
            else:
                weighted_quality += term.weight * result.score
            if term.gate and result.status.scorable and not result.passed:
                gate_failures.append(term.verifier)

        expected_costs: dict[str, float] = {}
        cost_penalty = 0.0
        for term in contract.costs:
            if term.key not in self.normalized_costs:
                if term.required:
                    mask_reasons.append(f"cost:{term.key}:missing")
                    continue
                value = 0.0
            else:
                value = self.normalized_costs[term.key]
            expected_costs[term.key] = value
            cost_penalty += term.weight * value
        if dict(self.normalized_costs) != expected_costs:
            raise ValueError("report normalized costs differ from its reward contract")

        masked = bool(mask_reasons)
        quality_reward: float | None = None if masked else weighted_quality
        if masked:
            reward: float | None = None
        elif gate_failures:
            reward = contract.failure_reward
        else:
            reward = weighted_quality - cost_penalty
            reward = min(contract.maximum_reward, max(contract.minimum_reward, reward))

        if self.masked != masked or self.mask_reasons != tuple(mask_reasons):
            raise ValueError("report mask state differs from recomputed contract semantics")
        if self.gate_failures != tuple(gate_failures):
            raise ValueError("report gate failures differ from recomputed contract semantics")
        if not math.isclose(self.cost_penalty, cost_penalty, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("report cost penalty differs from recomputed contract semantics")
        if not self._same_optional_number(self.quality_reward, quality_reward):
            raise ValueError("report quality reward differs from recomputed contract semantics")
        if not self._same_optional_number(self.reward, reward):
            raise ValueError("report reward differs from recomputed contract semantics")

    @staticmethod
    def _same_optional_number(left: float | None, right: float | None) -> bool:
        if left is None or right is None:
            return left is right
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)

    @property
    def evaluation_spec_fingerprint(self) -> str:
        """Compatibility alias emphasizing that costs/results are excluded from this digest."""

        return self.evaluation_fingerprint

    @property
    def fingerprint(self) -> str:
        """Hash reward semantics while excluding nondeterministic diagnostic timing."""

        payload = self.to_dict()
        for result in payload["verifier_results"].values():
            result.pop("duration_ms", None)
        return fingerprint_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "contract": self.contract,
            "contract_version": self.contract_version,
            "contract_fingerprint": self.contract_fingerprint,
            "sample_fingerprint": self.sample_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "engine_manifest": thaw_json(self.engine_manifest),
            "reward": self.reward,
            "quality_reward": self.quality_reward,
            "cost_penalty": self.cost_penalty,
            "masked": self.masked,
            "mask_reasons": list(self.mask_reasons),
            "gate_failures": list(self.gate_failures),
            "verifier_results": {
                name: result.to_dict() for name, result in self.verifier_results.items()
            },
            "normalized_costs": dict(self.normalized_costs),
        }


class RewardEngine:
    """Evaluate samples with a fixed contract and fixed verifier implementations."""

    def __init__(self, contract: RewardContract, verifiers: Iterable[Verifier]) -> None:
        verifier_map: dict[str, Verifier] = {}
        for verifier in verifiers:
            if verifier.name in verifier_map:
                raise ValueError(f"duplicate verifier name: {verifier.name}")
            verifier_map[verifier.name] = verifier

        missing = {term.verifier for term in contract.terms}.difference(verifier_map)
        if missing:
            raise ValueError(f"contract references missing verifiers: {sorted(missing)}")
        unused = set(verifier_map).difference(term.verifier for term in contract.terms)
        if unused:
            raise ValueError(f"unused verifiers are not allowed: {sorted(unused)}")

        self.contract = contract
        self.verifiers: Mapping[str, Verifier] = MappingProxyType(verifier_map)
        used_verifiers = {
            term.verifier: self.verifiers[term.verifier].manifest() for term in self.contract.terms
        }
        manifest = {
            "schema_version": "molor.reward-engine.v1",
            "contract": self.contract.to_dict(),
            "verifiers": used_verifiers,
        }
        self.manifest: Mapping[str, Any] = freeze_json(manifest)
        self.fingerprint = fingerprint_json(manifest)

    def evaluate(
        self,
        sample: EvaluationSample,
        *,
        normalized_costs: Mapping[str, float] | None = None,
    ) -> RewardReport:
        supplied_costs = dict(normalized_costs or {})
        results: dict[str, VerificationResult] = {}
        gate_failures: list[str] = []
        mask_reasons: list[str] = []
        weighted_quality = 0.0

        for term in self.contract.terms:
            verifier = self.verifiers[term.verifier]
            started = time.perf_counter()
            try:
                result = verifier.verify(sample)
                if result.verifier != verifier.name:
                    raise ValueError(
                        f"verifier returned name {result.verifier!r}; expected {verifier.name!r}"
                    )
            except Exception as exc:  # A bad rollout must not crash a training batch.
                result = VerificationResult.unscorable(
                    verifier.name,
                    VerificationStatus.VERIFIER_ERROR,
                    f"{type(exc).__name__}: {exc}",
                )
            if result.duration_ms is None:
                result = replace(result, duration_ms=(time.perf_counter() - started) * 1_000.0)
            results[term.verifier] = result
            if result.score is None:
                mask_reasons.append(f"{term.verifier}:{result.status.value}")
            else:
                weighted_quality += term.weight * result.score
            if term.gate and result.status.scorable and not result.passed:
                gate_failures.append(term.verifier)

        used_costs: dict[str, float] = {}
        cost_penalty = 0.0
        for term in self.contract.costs:
            if term.key not in supplied_costs:
                if term.required:
                    mask_reasons.append(f"cost:{term.key}:missing")
                    continue
                value = 0.0
            else:
                supplied_value = supplied_costs[term.key]
                if isinstance(supplied_value, bool):
                    raise TypeError(f"normalized cost {term.key!r} must be a real number")
                value = float(supplied_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"normalized cost {term.key!r} must be finite and in [0, 1]")
            used_costs[term.key] = value
            cost_penalty += term.weight * value

        masked = bool(mask_reasons)
        quality_reward: float | None = None if masked else weighted_quality
        if masked:
            reward = None
        elif gate_failures:
            reward = self.contract.failure_reward
        else:
            reward = weighted_quality - cost_penalty
            reward = min(self.contract.maximum_reward, max(self.contract.minimum_reward, reward))

        sample_fingerprint = sample.fingerprint
        evaluation_fingerprint = hashlib.sha256(
            f"{sample_fingerprint}:{self.fingerprint}".encode("ascii")
        ).hexdigest()

        return RewardReport(
            sample_id=sample.sample_id,
            contract=self.contract.name,
            contract_version=self.contract.version,
            contract_fingerprint=self.fingerprint,
            sample_fingerprint=sample_fingerprint,
            prompt_fingerprint=sample.prompt_fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
            engine_manifest=self.manifest,
            reward=reward,
            quality_reward=quality_reward,
            cost_penalty=cost_penalty,
            masked=masked,
            mask_reasons=tuple(mask_reasons),
            gate_failures=tuple(gate_failures),
            verifier_results=MappingProxyType(results),
            normalized_costs=MappingProxyType(used_costs),
        )
