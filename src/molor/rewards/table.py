"""Dense prompt-by-route reward tables for supervised and bandit router training."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from molor.identity import (
    fingerprint_json,
    freeze_json,
    thaw_json,
    validate_json_value,
    validate_sha256,
)
from molor.manifests import AdapterBankManifest
from molor.rewards.contract import RewardReport
from molor.rewards.types import EvaluationSample


def _nonempty_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > 512 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} is too long or contains control characters")
    return value


@dataclass(frozen=True, slots=True)
class PromptRef:
    """Completion-independent identity for one declared row of a reward table."""

    prompt_id: str
    prompt_fingerprint: str

    def __post_init__(self) -> None:
        _nonempty_id(self.prompt_id, "prompt_id")
        object.__setattr__(
            self,
            "prompt_fingerprint",
            validate_sha256(self.prompt_fingerprint, "prompt_fingerprint"),
        )

    @classmethod
    def from_sample(cls, prompt_id: str, sample: EvaluationSample) -> PromptRef:
        if not isinstance(sample, EvaluationSample):
            raise TypeError("sample must be an EvaluationSample")
        return cls(prompt_id=prompt_id, prompt_fingerprint=sample.prompt_fingerprint)

    def to_dict(self) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_fingerprint": self.prompt_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ForcedRouteRecord:
    """Generation-time proof of the exact single route used for one completed sample."""

    schema_version = "molor.forced-route.v1"
    routing_mode = "forced_single"

    rollout_id: str
    sample_fingerprint: str
    prompt_fingerprint: str
    bank_fingerprint: str
    route_id: str
    route_index: int
    adapter_fingerprint: str | None
    generation_fingerprint: str
    generation_seed: int

    def __post_init__(self) -> None:
        for field_name in ("rollout_id", "route_id"):
            _nonempty_id(getattr(self, field_name), field_name)
        for field_name in (
            "sample_fingerprint",
            "prompt_fingerprint",
            "bank_fingerprint",
            "generation_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name),
            )
        if self.adapter_fingerprint is not None:
            object.__setattr__(
                self,
                "adapter_fingerprint",
                validate_sha256(self.adapter_fingerprint, "adapter_fingerprint"),
            )
        if (
            not isinstance(self.route_index, int)
            or isinstance(self.route_index, bool)
            or self.route_index < 0
        ):
            raise ValueError("route_index must be a non-negative integer")
        if (
            not isinstance(self.generation_seed, int)
            or isinstance(self.generation_seed, bool)
            or self.generation_seed < 0
        ):
            raise ValueError("generation_seed must be a non-negative integer")

    @classmethod
    def from_sample(
        cls,
        bank: AdapterBankManifest,
        route_id: str,
        sample: EvaluationSample,
        *,
        generation_fingerprint: str,
        generation_seed: int,
    ) -> ForcedRouteRecord:
        """Create this record where forced generation has access to the exact completed sample."""

        if not isinstance(bank, AdapterBankManifest):
            raise TypeError("bank must be an AdapterBankManifest")
        if not isinstance(sample, EvaluationSample):
            raise TypeError("sample must be an EvaluationSample")
        route = bank.route_by_id(route_id)
        return cls(
            rollout_id=sample.sample_id,
            sample_fingerprint=sample.fingerprint,
            prompt_fingerprint=sample.prompt_fingerprint,
            bank_fingerprint=bank.fingerprint,
            route_id=route.route_id,
            route_index=route.route_index,
            adapter_fingerprint=route.adapter_fingerprint,
            generation_fingerprint=generation_fingerprint,
            generation_seed=generation_seed,
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "routing_mode": self.routing_mode,
            "rollout_id": self.rollout_id,
            "sample_fingerprint": self.sample_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "bank_fingerprint": self.bank_fingerprint,
            "route_id": self.route_id,
            "route_index": self.route_index,
            "adapter_fingerprint": self.adapter_fingerprint,
            "generation_fingerprint": self.generation_fingerprint,
            "generation_seed": self.generation_seed,
        }


@dataclass(frozen=True, slots=True)
class CostNormalizationRecord:
    """Auditable binding between raw measurements and reward-affecting normalized costs."""

    schema_version = "molor.cost-normalization.v1"

    normalizer_fingerprint: str
    raw_measurements: Mapping[str, float] = field(default_factory=dict)
    normalized_costs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalizer_fingerprint",
            validate_sha256(self.normalizer_fingerprint, "normalizer_fingerprint"),
        )
        raw = self._validated_values(
            self.raw_measurements,
            field_name="raw measurements",
            bounded=False,
        )
        normalized = self._validated_values(
            self.normalized_costs,
            field_name="normalized costs",
            bounded=True,
        )
        object.__setattr__(self, "raw_measurements", MappingProxyType(raw))
        object.__setattr__(self, "normalized_costs", MappingProxyType(normalized))

    @staticmethod
    def _validated_values(
        values: Mapping[str, float],
        *,
        field_name: str,
        bounded: bool,
    ) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        result: dict[str, float] = {}
        for key, value in values.items():
            _nonempty_id(key, f"{field_name} key")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must contain real numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            if bounded and numeric > 1:
                raise ValueError("normalized costs must be in [0, 1]")
            result[key] = numeric
        return result

    @classmethod
    def from_report(
        cls,
        normalizer_fingerprint: str,
        report: RewardReport,
        *,
        raw_measurements: Mapping[str, float] | None = None,
    ) -> CostNormalizationRecord:
        if not isinstance(report, RewardReport):
            raise TypeError("report must be a RewardReport")
        return cls(
            normalizer_fingerprint=normalizer_fingerprint,
            raw_measurements=raw_measurements or {},
            normalized_costs=report.normalized_costs,
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalizer_fingerprint": self.normalizer_fingerprint,
            "raw_measurements": dict(self.raw_measurements),
            "normalized_costs": dict(self.normalized_costs),
        }


@dataclass(frozen=True, slots=True)
class RewardTableEntry:
    """One forced-route rollout bound to immutable prompt, route, and reward identities."""

    prompt_id: str
    prompt_fingerprint: str
    rollout_id: str
    route_id: str
    route_index: int
    adapter_fingerprint: str | None
    bank_fingerprint: str
    contract_fingerprint: str
    generation_fingerprint: str
    cost_normalizer_fingerprint: str
    generation_seed: int
    execution: ForcedRouteRecord
    cost_normalization: CostNormalizationRecord
    report: RewardReport
    raw_measurements: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("prompt_id", "rollout_id", "route_id"):
            _nonempty_id(getattr(self, field_name), field_name)
        if (
            not isinstance(self.route_index, int)
            or isinstance(self.route_index, bool)
            or self.route_index < 0
        ):
            raise ValueError("route_index must be a non-negative integer")
        if (
            not isinstance(self.generation_seed, int)
            or isinstance(self.generation_seed, bool)
            or self.generation_seed < 0
        ):
            raise ValueError("generation_seed must be a non-negative integer")
        for field_name in (
            "prompt_fingerprint",
            "bank_fingerprint",
            "contract_fingerprint",
            "generation_fingerprint",
            "cost_normalizer_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name),
            )
        if self.adapter_fingerprint is not None:
            object.__setattr__(
                self,
                "adapter_fingerprint",
                validate_sha256(self.adapter_fingerprint, "adapter_fingerprint"),
            )
        if not isinstance(self.execution, ForcedRouteRecord):
            raise TypeError("execution must be a ForcedRouteRecord")
        if not isinstance(self.cost_normalization, CostNormalizationRecord):
            raise TypeError("cost_normalization must be a CostNormalizationRecord")
        if not isinstance(self.report, RewardReport):
            raise TypeError("report must be a RewardReport")
        if self.rollout_id != self.report.sample_id:
            raise ValueError("rollout_id must match report.sample_id")
        if self.prompt_fingerprint != self.report.prompt_fingerprint:
            raise ValueError("entry prompt fingerprint must match its reward report")
        if self.contract_fingerprint != self.report.contract_fingerprint:
            raise ValueError("entry contract fingerprint must match its reward report")
        execution_fields = (
            "rollout_id",
            "prompt_fingerprint",
            "route_id",
            "route_index",
            "adapter_fingerprint",
            "bank_fingerprint",
            "generation_fingerprint",
            "generation_seed",
        )
        mismatched_execution = [
            field_name
            for field_name in execution_fields
            if getattr(self, field_name) != getattr(self.execution, field_name)
        ]
        if mismatched_execution:
            raise ValueError(
                "entry fields differ from forced-route execution: "
                + ", ".join(mismatched_execution)
            )
        if self.execution.sample_fingerprint != self.report.sample_fingerprint:
            raise ValueError("forced-route sample fingerprint must match its reward report")
        if self.cost_normalization.normalizer_fingerprint != self.cost_normalizer_fingerprint:
            raise ValueError("cost-normalization fingerprint differs from the entry")
        if dict(self.cost_normalization.normalized_costs) != dict(self.report.normalized_costs):
            raise ValueError("normalized costs differ from the reward report")

        measurements: dict[str, float] = {}
        for key, value in self.raw_measurements.items():
            _nonempty_id(key, "raw measurement key")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("raw measurements must be real numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("raw measurements must be finite and non-negative")
            measurements[key] = numeric
        object.__setattr__(self, "raw_measurements", MappingProxyType(measurements))
        if measurements != dict(self.cost_normalization.raw_measurements):
            raise ValueError("raw measurements differ from the cost-normalization record")

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(
            {
                "prompt_id": self.prompt_id,
                "prompt_fingerprint": self.prompt_fingerprint,
                "rollout_id": self.rollout_id,
                "route_id": self.route_id,
                "route_index": self.route_index,
                "adapter_fingerprint": self.adapter_fingerprint,
                "bank_fingerprint": self.bank_fingerprint,
                "contract_fingerprint": self.contract_fingerprint,
                "generation_fingerprint": self.generation_fingerprint,
                "cost_normalizer_fingerprint": self.cost_normalizer_fingerprint,
                "generation_seed": self.generation_seed,
                "raw_measurements": dict(self.raw_measurements),
                "execution_fingerprint": self.execution.fingerprint,
                "cost_normalization_fingerprint": self.cost_normalization.fingerprint,
                "report_fingerprint": self.report.fingerprint,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_fingerprint": self.prompt_fingerprint,
            "rollout_id": self.rollout_id,
            "route_id": self.route_id,
            "route_index": self.route_index,
            "adapter_fingerprint": self.adapter_fingerprint,
            "bank_fingerprint": self.bank_fingerprint,
            "contract_fingerprint": self.contract_fingerprint,
            "generation_fingerprint": self.generation_fingerprint,
            "cost_normalizer_fingerprint": self.cost_normalizer_fingerprint,
            "generation_seed": self.generation_seed,
            "raw_measurements": dict(self.raw_measurements),
            "execution": self.execution.to_dict(),
            "cost_normalization": self.cost_normalization.to_dict(),
            "report_fingerprint": self.report.fingerprint,
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RewardMatrix:
    """Immutable dense values plus an explicit validity mask."""

    prompt_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]
    valid: tuple[tuple[bool, ...], ...]

    def value_for(self, prompt_id: str, route_id: str) -> float | None:
        try:
            prompt_index = self.prompt_ids.index(prompt_id)
            route_index = self.route_ids.index(route_id)
        except ValueError as exc:
            raise KeyError(f"unknown prompt or route: {prompt_id!r}, {route_id!r}") from exc
        return self.values[prompt_index][route_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_ids": list(self.prompt_ids),
            "route_ids": list(self.route_ids),
            "values": [list(row) for row in self.values],
            "valid": [list(row) for row in self.valid],
        }


@dataclass(frozen=True, slots=True)
class OracleSelection:
    """Best route set, retaining ties and the gap to the next route."""

    prompt_id: str
    reward: float
    route_ids: tuple[str, ...]
    route_indices: tuple[int, ...]
    margin: float | None

    @property
    def is_tie(self) -> bool:
        return len(self.route_ids) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "reward": self.reward,
            "route_ids": list(self.route_ids),
            "route_indices": list(self.route_indices),
            "margin": self.margin,
            "is_tie": self.is_tie,
        }


class RewardTable:
    """A declared prompt × route cross-product populated by forced single-route rollouts."""

    schema_version = "molor.reward-table.v1"
    routing_mode = "forced_single"
    max_prompts = 1_000_000
    max_cells = 10_000_000

    def __init__(
        self,
        bank: AdapterBankManifest,
        prompts: Iterable[PromptRef],
        *,
        engine_manifest: Mapping[str, Any],
        generation_manifest: Mapping[str, Any],
        cost_normalizer_manifest: Mapping[str, Any],
    ) -> None:
        if not isinstance(bank, AdapterBankManifest):
            raise TypeError("bank must be an AdapterBankManifest")
        prompt_tuple = tuple(islice(iter(prompts), self.max_prompts + 1))
        if not prompt_tuple:
            raise ValueError("reward table requires at least one declared prompt")
        if len(prompt_tuple) > self.max_prompts:
            raise ValueError(f"reward table exceeds the {self.max_prompts} prompt limit")
        if any(not isinstance(prompt, PromptRef) for prompt in prompt_tuple):
            raise TypeError("prompts must contain only PromptRef values")
        prompt_ids = [prompt.prompt_id for prompt in prompt_tuple]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("declared prompt IDs must be unique")
        if len(prompt_tuple) * len(bank.routes) > self.max_cells:
            raise ValueError(f"reward table exceeds the {self.max_cells} cell limit")

        manifests: dict[str, Mapping[str, Any]] = {}
        for field_name, value in (
            ("engine_manifest", engine_manifest),
            ("generation_manifest", generation_manifest),
            ("cost_normalizer_manifest", cost_normalizer_manifest),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            validate_json_value(value, field_name)
            manifests[field_name] = freeze_json(value)
        thawed_engine = thaw_json(manifests["engine_manifest"])
        if thawed_engine.get("schema_version") != "molor.reward-engine.v1":
            raise ValueError("engine_manifest has an unsupported schema version")

        self.bank = bank
        self.prompts = prompt_tuple
        self.engine_manifest = manifests["engine_manifest"]
        self.generation_manifest = manifests["generation_manifest"]
        self.cost_normalizer_manifest = manifests["cost_normalizer_manifest"]
        self.contract_fingerprint = fingerprint_json(thaw_json(self.engine_manifest))
        self.generation_fingerprint = fingerprint_json(thaw_json(self.generation_manifest))
        self.cost_normalizer_fingerprint = fingerprint_json(
            thaw_json(self.cost_normalizer_manifest)
        )
        self._prompt_by_id = {prompt.prompt_id: prompt for prompt in prompt_tuple}
        self._prompt_order = {prompt.prompt_id: index for index, prompt in enumerate(prompt_tuple)}
        self._entries: dict[tuple[str, str], RewardTableEntry] = {}
        self._rollout_ids: set[str] = set()
        self._generation_seed_by_prompt: dict[str, int] = {}

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(prompt.prompt_id for prompt in self.prompts)

    @property
    def route_ids(self) -> tuple[str, ...]:
        return self.bank.route_ids

    @property
    def prompt_set_fingerprint(self) -> str:
        return fingerprint_json([prompt.to_dict() for prompt in self.prompts])

    @property
    def entries(self) -> tuple[RewardTableEntry, ...]:
        return tuple(
            sorted(
                self._entries.values(),
                key=lambda entry: (
                    self._prompt_order[entry.prompt_id],
                    entry.route_index,
                ),
            )
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(
            {
                "schema_version": self.schema_version,
                "routing_mode": self.routing_mode,
                "base_fingerprint": self.bank.base.fingerprint,
                "bank_fingerprint": self.bank.fingerprint,
                "routes": [route.to_dict() for route in self.bank.routes],
                "contract_fingerprint": self.contract_fingerprint,
                "generation_fingerprint": self.generation_fingerprint,
                "cost_normalizer_fingerprint": self.cost_normalizer_fingerprint,
                "prompt_set_fingerprint": self.prompt_set_fingerprint,
                "prompts": [prompt.to_dict() for prompt in self.prompts],
                "entry_fingerprints": [entry.fingerprint for entry in self.entries],
            }
        )

    def record(
        self,
        *,
        prompt_id: str,
        execution: ForcedRouteRecord,
        cost_normalization: CostNormalizationRecord,
        report: RewardReport,
    ) -> RewardTableEntry:
        """Add a scored generation without accepting a route label after the fact."""

        try:
            prompt = self._prompt_by_id[prompt_id]
        except KeyError as exc:
            raise KeyError(f"unknown declared prompt ID: {prompt_id!r}") from exc
        entry = RewardTableEntry(
            prompt_id=prompt.prompt_id,
            prompt_fingerprint=prompt.prompt_fingerprint,
            rollout_id=execution.rollout_id,
            route_id=execution.route_id,
            route_index=execution.route_index,
            adapter_fingerprint=execution.adapter_fingerprint,
            bank_fingerprint=execution.bank_fingerprint,
            contract_fingerprint=self.contract_fingerprint,
            generation_fingerprint=execution.generation_fingerprint,
            cost_normalizer_fingerprint=cost_normalization.normalizer_fingerprint,
            generation_seed=execution.generation_seed,
            execution=execution,
            cost_normalization=cost_normalization,
            report=report,
            raw_measurements=cost_normalization.raw_measurements,
        )
        self.add(entry)
        return entry

    def add(self, entry: RewardTableEntry) -> None:
        """Add a prebuilt cell after validating every identity boundary."""

        if not isinstance(entry, RewardTableEntry):
            raise TypeError("entry must be a RewardTableEntry")
        if entry.bank_fingerprint != self.bank.fingerprint:
            raise ValueError("entry was produced for a different adapter bank")
        if entry.contract_fingerprint != self.contract_fingerprint:
            raise ValueError("entry was produced by a different reward contract")
        if entry.generation_fingerprint != self.generation_fingerprint:
            raise ValueError("entry used a different generation configuration")
        if entry.cost_normalizer_fingerprint != self.cost_normalizer_fingerprint:
            raise ValueError("entry used a different cost normalizer")
        try:
            prompt = self._prompt_by_id[entry.prompt_id]
        except KeyError as exc:
            raise ValueError("entry references an undeclared prompt ID") from exc
        if entry.prompt_fingerprint != prompt.prompt_fingerprint:
            raise ValueError("entry prompt content differs from its declared prompt")
        route = self.bank.route_by_id(entry.route_id)
        if entry.route_index != route.route_index:
            raise ValueError("entry route index does not match its stable route ID")
        if entry.adapter_fingerprint != route.adapter_fingerprint:
            raise ValueError("entry adapter fingerprint does not match its stable route ID")
        key = (entry.prompt_id, entry.route_id)
        if key in self._entries:
            raise ValueError(f"duplicate reward-table entry: prompt={key[0]!r}, route={key[1]!r}")
        if entry.rollout_id in self._rollout_ids:
            raise ValueError(f"duplicate rollout ID: {entry.rollout_id!r}")
        if entry.prompt_id in self._generation_seed_by_prompt:
            expected_seed = self._generation_seed_by_prompt[entry.prompt_id]
            if entry.generation_seed != expected_seed:
                raise ValueError(
                    "all routes for one prompt must use the same generation seed "
                    f"(expected {expected_seed}, got {entry.generation_seed})"
                )
        else:
            self._generation_seed_by_prompt[entry.prompt_id] = entry.generation_seed
        self._entries[key] = entry
        self._rollout_ids.add(entry.rollout_id)

    def get(self, prompt_id: str, route_id: str) -> RewardTableEntry:
        try:
            return self._entries[(prompt_id, route_id)]
        except KeyError as exc:
            raise KeyError(
                f"no reward-table entry for prompt={prompt_id!r}, route={route_id!r}"
            ) from exc

    def missing(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (prompt.prompt_id, route.route_id)
            for prompt in self.prompts
            for route in self.bank.routes
            if (prompt.prompt_id, route.route_id) not in self._entries
        )

    @property
    def missing_count(self) -> int:
        return len(self.prompts) * len(self.bank.routes) - len(self._entries)

    def missing_preview(self, limit: int = 8) -> tuple[tuple[str, str], ...]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        missing = (
            (prompt.prompt_id, route.route_id)
            for prompt in self.prompts
            for route in self.bank.routes
            if (prompt.prompt_id, route.route_id) not in self._entries
        )
        return tuple(islice(missing, limit))

    def masked(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (entry.prompt_id, entry.route_id) for entry in self.entries if entry.report.masked
        )

    def assert_complete(self, *, require_scorable: bool = False) -> None:
        if self.missing_count:
            preview = ", ".join(f"{prompt}/{route}" for prompt, route in self.missing_preview())
            raise ValueError(f"reward table is missing {self.missing_count} entries: {preview}")
        if require_scorable:
            masked = self.masked()
            if masked:
                preview = ", ".join(f"{prompt}/{route}" for prompt, route in masked[:8])
                raise ValueError(f"reward table has {len(masked)} masked entries: {preview}")

    @property
    def structurally_complete(self) -> bool:
        return self.missing_count == 0

    @property
    def oracle_ready(self) -> bool:
        return self.structurally_complete and not self.masked()

    def _scorable_row(self, prompt_id: str) -> tuple[RewardTableEntry, ...]:
        if prompt_id not in self._prompt_by_id:
            raise KeyError(f"unknown declared prompt ID: {prompt_id!r}")
        missing = [
            route.route_id
            for route in self.bank.routes
            if (prompt_id, route.route_id) not in self._entries
        ]
        if missing:
            raise ValueError(
                f"prompt {prompt_id!r} has an incomplete route row; missing {missing!r}"
            )
        entries = tuple(self.get(prompt_id, route.route_id) for route in self.bank.routes)
        masked = [entry.route_id for entry in entries if entry.report.reward is None]
        if masked:
            raise ValueError(f"prompt {prompt_id!r} has masked route rewards: {masked!r}")
        return entries

    def reward_matrix(self) -> RewardMatrix:
        """Return dense rewards and validity after proving the declared table is complete."""

        self.assert_complete()
        values = tuple(
            tuple(
                self.get(prompt.prompt_id, route.route_id).report.reward
                for route in self.bank.routes
            )
            for prompt in self.prompts
        )
        return RewardMatrix(
            prompt_ids=self.prompt_ids,
            route_ids=self.route_ids,
            values=values,
            valid=tuple(tuple(value is not None for value in row) for row in values),
        )

    def oracle(self, prompt_id: str, *, tie_tolerance: float = 1e-12) -> OracleSelection:
        if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise ValueError("tie_tolerance must be finite and non-negative")
        entries = self._scorable_row(prompt_id)
        rewards = [float(entry.report.reward) for entry in entries]
        best_reward = max(rewards)
        winners = [
            entry
            for entry, reward in zip(entries, rewards, strict=True)
            if math.isclose(reward, best_reward, rel_tol=0.0, abs_tol=tie_tolerance)
        ]
        sorted_rewards = sorted(rewards, reverse=True)
        margin = None if len(sorted_rewards) < 2 else best_reward - sorted_rewards[1]
        return OracleSelection(
            prompt_id=prompt_id,
            reward=best_reward,
            route_ids=tuple(entry.route_id for entry in winners),
            route_indices=tuple(entry.route_index for entry in winners),
            margin=margin,
        )

    def regret(
        self,
        prompt_id: str,
        selected_route_id: str,
        *,
        tie_tolerance: float = 1e-12,
    ) -> float:
        entries = self._scorable_row(prompt_id)
        selected = next(
            (entry for entry in entries if entry.route_id == selected_route_id),
            None,
        )
        if selected is None:
            raise KeyError(f"unknown route ID for bank: {selected_route_id!r}")
        assert selected.report.reward is not None
        oracle = self.oracle(prompt_id, tie_tolerance=tie_tolerance)
        if selected_route_id in oracle.route_ids:
            return 0.0
        return oracle.reward - selected.report.reward

    def soft_targets(self, prompt_id: str, *, temperature: float = 1.0) -> Mapping[str, float]:
        """Convert one complete scorable reward row into a soft router target."""

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        entries = self._scorable_row(prompt_id)
        rewards = [float(entry.report.reward) for entry in entries]
        maximum = max(rewards)
        exponentials = [math.exp((reward - maximum) / temperature) for reward in rewards]
        denominator = sum(exponentials)
        return MappingProxyType(
            {
                entry.route_id: value / denominator
                for entry, value in zip(entries, exponentials, strict=True)
            }
        )

    def route_mean(self, route_id: str) -> float:
        self.bank.route_by_id(route_id)
        self.assert_complete(require_scorable=True)
        rewards = [self.get(prompt.prompt_id, route_id).report.reward for prompt in self.prompts]
        assert all(reward is not None for reward in rewards)
        return sum(float(reward) for reward in rewards) / len(rewards)

    def route_means(self) -> Mapping[str, float]:
        return MappingProxyType(
            {route.route_id: self.route_mean(route.route_id) for route in self.bank.routes}
        )

    def oracle_mean(self) -> float:
        self.assert_complete(require_scorable=True)
        return sum(self.oracle(prompt.prompt_id).reward for prompt in self.prompts) / len(
            self.prompts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "routing_mode": self.routing_mode,
            "base_fingerprint": self.bank.base.fingerprint,
            "bank_fingerprint": self.bank.fingerprint,
            "bank_manifest": self.bank.to_dict(),
            "routes": [route.to_dict() for route in self.bank.routes],
            "contract_fingerprint": self.contract_fingerprint,
            "engine_manifest": thaw_json(self.engine_manifest),
            "generation_fingerprint": self.generation_fingerprint,
            "generation_manifest": thaw_json(self.generation_manifest),
            "cost_normalizer_fingerprint": self.cost_normalizer_fingerprint,
            "cost_normalizer_manifest": thaw_json(self.cost_normalizer_manifest),
            "prompt_set_fingerprint": self.prompt_set_fingerprint,
            "prompts": [prompt.to_dict() for prompt in self.prompts],
            "structurally_complete": self.structurally_complete,
            "oracle_ready": self.oracle_ready,
            "missing_cell_count": self.missing_count,
            "masked_cell_count": len(self.masked()),
            "table_fingerprint": self.fingerprint,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_records(self) -> list[dict[str, Any]]:
        """Return a header/cells/footer stream that retains axes and completion state."""

        header = self.to_dict()
        entries = header.pop("entries")
        header["record_type"] = "table_header"
        cell_records = [{"record_type": "cell", "entry": entry} for entry in entries]
        footer = {
            "record_type": "table_footer",
            "entry_count": len(entries),
            "structurally_complete": self.structurally_complete,
            "oracle_ready": self.oracle_ready,
            "table_fingerprint": self.fingerprint,
        }
        return [header, *cell_records, footer]
