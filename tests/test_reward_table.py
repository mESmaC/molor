from dataclasses import replace
import json

import pytest

from molor.manifests import AdapterBankManifest
from molor.rewards import (
    CostNormalizationRecord,
    CostTerm,
    EvaluationSample,
    ExactMatchVerifier,
    ForcedRouteRecord,
    PromptRef,
    RewardContract,
    RewardEngine,
    RewardReport,
    RewardTable,
    RewardTerm,
)

from test_manifests import adapter


GENERATION_MANIFEST = {"temperature": 0, "max_tokens": 128}
COST_NORMALIZER_MANIFEST = {"name": "identity", "version": "1", "ranges": {}}


def bank(*, reverse: bool = False) -> AdapterBankManifest:
    adapters = (adapter("math", "1"), adapter("code", "2"))
    if reverse:
        adapters = tuple(reversed(adapters))
    return AdapterBankManifest("skills", "1", adapters)


def engine(*, with_cost: bool = False) -> RewardEngine:
    contract = RewardContract(
        "exact",
        "1",
        (RewardTerm("outcome", 1.0),),
        costs=(CostTerm("latency", 0.5),) if with_cost else (),
    )
    return RewardEngine(contract, [ExactMatchVerifier(name="outcome")])


def prompt_ref(
    prompt_id: str = "case-1",
    *,
    prompt: str = "prompt",
    reference: object = "correct",
) -> PromptRef:
    return PromptRef.from_sample(
        prompt_id,
        EvaluationSample("prompt-definition", prompt, "", reference),
    )


def evaluate_rollout(
    reward_engine: RewardEngine,
    rollout_id: str,
    *,
    passing: bool = True,
    prompt: str = "prompt",
    reference: object = "correct",
    normalized_costs: dict[str, float] | None = None,
) -> tuple[EvaluationSample, RewardReport]:
    completion = "correct" if passing else "wrong"
    sample = EvaluationSample(rollout_id, prompt, completion, reference)
    return sample, reward_engine.evaluate(
        sample,
        normalized_costs=normalized_costs,
    )


def report(reward_engine: RewardEngine, rollout_id: str, **kwargs):
    return evaluate_rollout(reward_engine, rollout_id, **kwargs)[1]


def record_route(
    table: RewardTable,
    reward_engine: RewardEngine,
    route_id: str,
    *,
    prompt_id: str = "case-1",
    rollout_id: str | None = None,
    generation_seed: int = 1,
    raw_measurements: dict[str, float] | None = None,
    **rollout_kwargs,
):
    sample, reward_report = evaluate_rollout(
        reward_engine,
        rollout_id or f"{prompt_id}/{route_id}",
        **rollout_kwargs,
    )
    execution = ForcedRouteRecord.from_sample(
        table.bank,
        route_id,
        sample,
        generation_fingerprint=table.generation_fingerprint,
        generation_seed=generation_seed,
    )
    normalization = CostNormalizationRecord.from_report(
        table.cost_normalizer_fingerprint,
        reward_report,
        raw_measurements=raw_measurements,
    )
    return table.record(
        prompt_id=prompt_id,
        execution=execution,
        cost_normalization=normalization,
        report=reward_report,
    )


def empty_table(
    reward_engine: RewardEngine,
    *,
    prompts: tuple[PromptRef, ...] = (prompt_ref(),),
    adapter_bank: AdapterBankManifest | None = None,
    cost_normalizer_manifest: dict = COST_NORMALIZER_MANIFEST,
) -> RewardTable:
    return RewardTable(
        adapter_bank or bank(),
        prompts,
        engine_manifest=reward_engine.manifest,
        generation_manifest=GENERATION_MANIFEST,
        cost_normalizer_manifest=cost_normalizer_manifest,
    )


def complete_table() -> RewardTable:
    reward_engine = engine()
    table = empty_table(reward_engine)
    record_route(
        table,
        reward_engine,
        "math",
        passing=True,
        raw_measurements={"latency_ms": 8.5},
    )
    record_route(
        table,
        reward_engine,
        "code",
        passing=False,
    )
    record_route(
        table,
        reward_engine,
        "base/null",
        rollout_id="case-1/base",
        passing=False,
    )
    return table


def test_declared_axes_detect_an_entire_missing_row_and_duplicates() -> None:
    reward_engine = engine()
    table = empty_table(
        reward_engine,
        prompts=(prompt_ref("case-1"), prompt_ref("case-2")),
    )
    sample, first = evaluate_rollout(reward_engine, "case-1/math")
    execution = ForcedRouteRecord.from_sample(
        table.bank,
        "math",
        sample,
        generation_fingerprint=table.generation_fingerprint,
        generation_seed=1,
    )
    normalization = CostNormalizationRecord.from_report(
        table.cost_normalizer_fingerprint,
        first,
    )
    table.record(
        prompt_id="case-1",
        execution=execution,
        cost_normalization=normalization,
        report=first,
    )

    assert table.missing() == (
        ("case-1", "code"),
        ("case-1", "base/null"),
        ("case-2", "math"),
        ("case-2", "code"),
        ("case-2", "base/null"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        table.record(
            prompt_id="case-1",
            execution=execution,
            cost_normalization=normalization,
            report=first,
        )
    with pytest.raises(ValueError, match="missing 5"):
        table.assert_complete()


def test_complete_table_produces_matrix_oracle_regret_and_soft_targets() -> None:
    table = complete_table()

    table.assert_complete(require_scorable=True)
    matrix = table.reward_matrix()
    oracle = table.oracle("case-1")
    targets = table.soft_targets("case-1", temperature=0.5)

    assert matrix.value_for("case-1", "math") == 1.0
    assert matrix.valid == ((True, True, True),)
    assert oracle.route_ids == ("math",)
    assert oracle.route_indices == (0,)
    assert oracle.margin == 1.0
    assert table.regret("case-1", "code") == 1.0
    assert table.route_mean("math") == 1.0
    assert table.oracle_mean() == 1.0
    assert sum(targets.values()) == pytest.approx(1.0)
    assert targets["math"] > targets["code"]
    json.dumps(table.to_dict(), allow_nan=False)
    records = table.to_records()
    assert records[0]["record_type"] == "table_header"
    assert records[-1] == {
        "record_type": "table_footer",
        "entry_count": 3,
        "structurally_complete": True,
        "oracle_ready": True,
        "table_fingerprint": table.fingerprint,
    }


def test_oracle_retains_ties() -> None:
    reward_engine = engine()
    table = empty_table(reward_engine)
    for route in table.bank.routes:
        record_route(
            table,
            reward_engine,
            route.route_id,
            passing=True,
        )

    oracle = table.oracle("case-1")

    assert oracle.is_tie
    assert oracle.route_ids == table.route_ids
    assert oracle.margin == 0.0
    assert all(table.regret("case-1", route_id) == 0.0 for route_id in oracle.route_ids)


def test_masked_entries_remain_present_but_oracle_operations_refuse_them() -> None:
    reward_engine = engine()
    table = empty_table(
        reward_engine,
        prompts=(prompt_ref(reference=42),),
    )
    for route in table.bank.routes:
        record_route(
            table,
            reward_engine,
            route.route_id,
            reference=42,
        )

    table.assert_complete()
    matrix = table.reward_matrix()
    assert len(table.masked()) == 3
    assert table.structurally_complete
    assert not table.oracle_ready
    assert matrix.values == ((None, None, None),)
    assert matrix.valid == ((False, False, False),)
    with pytest.raises(ValueError, match="masked entries"):
        table.assert_complete(require_scorable=True)
    with pytest.raises(ValueError, match="masked route rewards"):
        table.oracle("case-1")
    with pytest.raises(ValueError, match="masked route rewards"):
        table.soft_targets("case-1")


def test_prompt_content_mismatch_is_rejected_before_insert() -> None:
    reward_engine = engine()
    table = empty_table(reward_engine)
    changed_sample, changed_prompt_report = evaluate_rollout(
        reward_engine,
        "case-1/math",
        prompt="different prompt",
    )
    execution = ForcedRouteRecord.from_sample(
        table.bank,
        "math",
        changed_sample,
        generation_fingerprint=table.generation_fingerprint,
        generation_seed=1,
    )
    normalization = CostNormalizationRecord.from_report(
        table.cost_normalizer_fingerprint,
        changed_prompt_report,
    )

    with pytest.raises(ValueError, match="prompt fingerprint"):
        table.record(
            prompt_id="case-1",
            execution=execution,
            cost_normalization=normalization,
            report=changed_prompt_report,
        )


def test_forced_route_record_cannot_be_swapped_with_another_completion() -> None:
    reward_engine = engine()
    table = empty_table(reward_engine)
    math_sample, _ = evaluate_rollout(reward_engine, "case-1/math", passing=True)
    _, code_report = evaluate_rollout(reward_engine, "case-1/math", passing=False)
    math_execution = ForcedRouteRecord.from_sample(
        table.bank,
        "math",
        math_sample,
        generation_fingerprint=table.generation_fingerprint,
        generation_seed=1,
    )
    normalization = CostNormalizationRecord.from_report(
        table.cost_normalizer_fingerprint,
        code_report,
    )

    with pytest.raises(ValueError, match="sample fingerprint"):
        table.record(
            prompt_id="case-1",
            execution=math_execution,
            cost_normalization=normalization,
            report=code_report,
        )


def test_cost_normalization_must_match_reward_affecting_costs() -> None:
    reward_engine = engine(with_cost=True)
    table = empty_table(reward_engine)
    sample, reward_report = evaluate_rollout(
        reward_engine,
        "case-1/math",
        normalized_costs={"latency": 1.0},
    )
    execution = ForcedRouteRecord.from_sample(
        table.bank,
        "math",
        sample,
        generation_fingerprint=table.generation_fingerprint,
        generation_seed=1,
    )
    dishonest_normalization = CostNormalizationRecord(
        table.cost_normalizer_fingerprint,
        raw_measurements={"latency_ms": 1_000},
        normalized_costs={"latency": 0.0},
    )

    with pytest.raises(ValueError, match="normalized costs differ"):
        table.record(
            prompt_id="case-1",
            execution=execution,
            cost_normalization=dishonest_normalization,
            report=reward_report,
        )


def test_routes_for_one_prompt_must_use_a_matched_seed() -> None:
    reward_engine = engine()
    table = empty_table(reward_engine)
    record_route(table, reward_engine, "math", generation_seed=7)

    with pytest.raises(ValueError, match="same generation seed"):
        record_route(table, reward_engine, "code", generation_seed=8)


def test_entry_from_reordered_bank_or_different_normalizer_is_rejected() -> None:
    source = complete_table()
    reordered = RewardTable(
        bank(reverse=True),
        source.prompts,
        engine_manifest=source.engine_manifest,
        generation_manifest=source.generation_manifest,
        cost_normalizer_manifest=source.cost_normalizer_manifest,
    )
    different_normalizer = RewardTable(
        source.bank,
        source.prompts,
        engine_manifest=source.engine_manifest,
        generation_manifest=source.generation_manifest,
        cost_normalizer_manifest={"name": "different"},
    )

    with pytest.raises(ValueError, match="different adapter bank"):
        reordered.add(source.get("case-1", "math"))
    with pytest.raises(ValueError, match="different cost normalizer"):
        different_normalizer.add(source.get("case-1", "math"))


def test_report_and_table_fingerprints_cover_costs_but_exclude_duration() -> None:
    reward_engine = engine(with_cost=True)
    low_sample, low_cost = evaluate_rollout(
        reward_engine,
        "case-1/math",
        normalized_costs={"latency": 0.0},
    )
    high_sample, high_cost = evaluate_rollout(
        reward_engine,
        "case-1/math",
        normalized_costs={"latency": 1.0},
    )
    timed_result = replace(
        low_cost.verifier_results["outcome"],
        duration_ms=low_cost.verifier_results["outcome"].duration_ms + 10,
    )
    differently_timed = replace(
        low_cost,
        verifier_results={"outcome": timed_result},
    )

    assert low_cost.evaluation_fingerprint == high_cost.evaluation_fingerprint
    assert low_cost.fingerprint != high_cost.fingerprint
    assert low_cost.fingerprint == differently_timed.fingerprint

    low_table = empty_table(reward_engine)
    high_table = empty_table(reward_engine)
    low_table.record(
        prompt_id="case-1",
        execution=ForcedRouteRecord.from_sample(
            low_table.bank,
            "math",
            low_sample,
            generation_fingerprint=low_table.generation_fingerprint,
            generation_seed=1,
        ),
        cost_normalization=CostNormalizationRecord.from_report(
            low_table.cost_normalizer_fingerprint,
            low_cost,
        ),
        report=low_cost,
    )
    high_table.record(
        prompt_id="case-1",
        execution=ForcedRouteRecord.from_sample(
            high_table.bank,
            "math",
            high_sample,
            generation_fingerprint=high_table.generation_fingerprint,
            generation_seed=1,
        ),
        cost_normalization=CostNormalizationRecord.from_report(
            high_table.cost_normalizer_fingerprint,
            high_cost,
        ),
        report=high_cost,
    )
    assert low_table.fingerprint != high_table.fingerprint


def test_insertion_order_does_not_change_dense_axes_or_table_fingerprint() -> None:
    expected = complete_table()
    reward_engine = engine()
    reordered = empty_table(reward_engine)
    record_route(
        reordered,
        reward_engine,
        "base/null",
        rollout_id="case-1/base",
        passing=False,
    )
    record_route(reordered, reward_engine, "code", passing=False)
    record_route(
        reordered,
        reward_engine,
        "math",
        passing=True,
        raw_measurements={"latency_ms": 8.5},
    )

    assert reordered.reward_matrix() == expected.reward_matrix()
    assert reordered.fingerprint == expected.fingerprint


def test_prompt_limit_boundedly_consumes_a_lazy_iterable() -> None:
    class TinyRewardTable(RewardTable):
        max_prompts = 2

    consumed = 0

    def prompts():
        nonlocal consumed
        for index in range(100):
            consumed += 1
            yield prompt_ref(f"case-{index}")

    reward_engine = engine()
    with pytest.raises(ValueError, match="prompt limit"):
        TinyRewardTable(
            bank(),
            prompts(),
            engine_manifest=reward_engine.manifest,
            generation_manifest=GENERATION_MANIFEST,
            cost_normalizer_manifest=COST_NORMALIZER_MANIFEST,
        )

    assert consumed == 3
