from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from molor.experimental.router_v8_turn_aware import (
    ROUTER_V8_MAX_TURNS,
    ROUTER_V8_ROLE_TO_ID,
    ROUTER_V8_SCHEMA,
    ROUTER_V8_SEEDS,
    ROUTER_V8_STATUS,
    DeterministicTurnRenderer,
    StructuredTurn,
    TurnAwareAdmissionEncoder,
    TurnRenderingPolicy,
    architecture_fingerprint,
    arm_a_metadata,
    arm_b_metadata,
    canonical_sha256,
    load_frozen_contract,
    masked_mean_control,
    preregistered_preblind_gates,
    preregistered_promotion_guards,
    select_taskmaster_or_exact_v4,
    validate_frozen_contract,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "experiments/router-v8-turn-aware-admission/contract.json"


class _MapTokenizer:
    def __init__(self, values: dict[str, list[int]]) -> None:
        self.values = values
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return list(self.values[text])


def test_renderer_is_deterministic_and_budgets_newest_first() -> None:
    turns = [
        StructuredTurn("system", "old-system"),
        StructuredTurn("user", "old-user"),
        StructuredTurn("assistant", "recent-assistant"),
        StructuredTurn("user", "current-user"),
    ]
    tokenizer = _MapTokenizer(
        {
            "old-system": [10, 11, 12, 13],
            "old-user": [20, 21, 22, 23, 24],
            "recent-assistant": [30, 31, 32, 33],
            "current-user": [40, 41, 42],
        }
    )
    renderer = DeterministicTurnRenderer(TurnRenderingPolicy(max_turns=4, max_tokens=9))

    first = renderer.render(turns, tokenizer=tokenizer)
    second = renderer.render(turns, tokenizer=tokenizer)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.token_count == 9
    assert [turn.source_index for turn in first.turns] == [1, 2, 3]
    assert [turn.role for turn in first.turns] == ["user", "assistant", "user"]
    assert [turn.recency_id for turn in first.turns] == [2, 1, 0]
    assert first.turns[0].input_ids == (23, 24)
    assert first.turns[0].truncated_left is True
    assert first.turns[-1].input_ids == (40, 41, 42)
    assert first.turns[-1].truncated_left is False
    assert all(add_special_tokens is True for _, add_special_tokens in tokenizer.calls)


def test_renderer_limits_turns_before_tokenization() -> None:
    turns = [StructuredTurn("assistant", f"turn-{index}") for index in range(20)]
    turns.append(StructuredTurn("user", "current"))
    tokenizer = _MapTokenizer({turn.content: [index + 1] for index, turn in enumerate(turns)})

    rendered = DeterministicTurnRenderer().render(turns, tokenizer=tokenizer)

    assert len(rendered.turns) == ROUTER_V8_MAX_TURNS
    assert [turn.source_index for turn in rendered.turns] == list(range(9, 21))
    assert "turn-0" not in {content for content, _ in tokenizer.calls}
    assert rendered.turns[-1].role == "user"
    assert rendered.turns[-1].recency_id == 0


@pytest.mark.parametrize(
    ("turns", "message"),
    [
        ([], "must not be empty"),
        ([StructuredTurn("assistant", "not current")], "final turn"),
    ],
)
def test_renderer_rejects_missing_current_user(turns: list[StructuredTurn], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeterministicTurnRenderer().render(turns, tokenizer=_MapTokenizer({}))


def test_renderer_rejects_overlong_current_user_without_truncating() -> None:
    tokenizer = _MapTokenizer({"current": list(range(513))})

    with pytest.raises(ValueError, match="refusing silent truncation"):
        DeterministicTurnRenderer().render([StructuredTurn("user", "current")], tokenizer=tokenizer)


def test_structured_turn_and_tokenizer_fail_closed() -> None:
    with pytest.raises(ValueError, match="role"):
        StructuredTurn("unknown", "content")
    with pytest.raises(ValueError, match="blank"):
        StructuredTurn("user", "  ")
    tokenizer = _MapTokenizer({"current": []})
    with pytest.raises(ValueError, match="must not be empty"):
        DeterministicTurnRenderer().render([StructuredTurn("user", "current")], tokenizer=tokenizer)


def test_representation_identity_changes_with_semantics() -> None:
    arm_b = arm_b_metadata()
    assert architecture_fingerprint() == canonical_sha256(
        {"arm_a": arm_a_metadata(), "arm_b": arm_b}
    )
    changed = copy.deepcopy(arm_b)
    changed["architecture"]["content_projection"]["output_width"] = 223  # type: ignore[index]
    assert canonical_sha256({"arm_a": arm_a_metadata(), "arm_b": changed}) != (
        architecture_fingerprint()
    )


def test_masked_mean_control_matches_exact_boolean_mask() -> None:
    token_values = torch.tensor(
        [[[1.0], [3.0], [100.0]], [[2.0], [4.0], [6.0]]], dtype=torch.float32
    )
    embeddings = token_values.expand(-1, -1, 2048).clone()
    mask = torch.tensor([[True, True, False], [False, True, True]])

    pooled = masked_mean_control(embeddings, mask)

    assert pooled.shape == (2, 2048)
    assert torch.equal(pooled[:, 0], torch.tensor([2.0, 5.0]))


def test_masked_mean_control_rejects_empty_or_nonfinite_inputs() -> None:
    embeddings = torch.zeros(1, 2, 2048)
    with pytest.raises(ValueError, match="at least one"):
        masked_mean_control(embeddings, torch.zeros(1, 2, dtype=torch.bool))
    embeddings[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        masked_mean_control(embeddings, torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        masked_mean_control(torch.zeros(1, 2, 2048), torch.ones(1, 2))


def _valid_encoder_inputs() -> tuple[torch.Tensor, ...]:
    embeddings = torch.randn(2, 4, 2048)
    roles = torch.tensor(
        [
            [ROUTER_V8_ROLE_TO_ID["system"], 2, ROUTER_V8_ROLE_TO_ID["user"], 0],
            [2, ROUTER_V8_ROLE_TO_ID["user"], 0, 0],
        ],
        dtype=torch.long,
    )
    recency = torch.tensor([[2, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return embeddings, roles, recency, mask


def test_arm_b_encoder_matches_frozen_interface_and_checkpoint_budget() -> None:
    torch.manual_seed(83)
    encoder = TurnAwareAdmissionEncoder().eval()
    inputs = _valid_encoder_inputs()

    with torch.no_grad():
        first = encoder(*inputs)
        second = encoder(*inputs)

    assert first.shape == (2,)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert encoder.architecture_metadata["architecture"] == arm_b_metadata()["architecture"]
    state_size = sum(
        tensor.numel() * tensor.element_size() for tensor in encoder.state_dict().values()
    )
    assert (
        state_size
        <= preregistered_preblind_gates()["resource_guards"][  # type: ignore[index]
            "checkpoint_size_bytes_max"
        ]
    )


@pytest.mark.parametrize("failure", ["mask", "recency", "newest_role", "nonfinite"])
def test_arm_b_encoder_rejects_out_of_contract_inputs(failure: str) -> None:
    encoder = TurnAwareAdmissionEncoder().eval()
    embeddings, roles, recency, mask = _valid_encoder_inputs()
    if failure == "mask":
        mask[0] = torch.tensor([True, False, True, False])
    elif failure == "recency":
        recency[0, 0] = 7
    elif failure == "newest_role":
        roles[1, 1] = ROUTER_V8_ROLE_TO_ID["assistant"]
    else:
        embeddings[0, 0, 0] = torch.inf

    with pytest.raises((TypeError, ValueError)):
        encoder(embeddings, roles, recency, mask)


def test_negative_gate_returns_the_exact_v4_object() -> None:
    v4_decision = object()
    calls: list[str] = []

    result = select_taskmaster_or_exact_v4(
        0.499999,
        build_taskmaster_decision=lambda: calls.append("positive") or object(),
        delegate_exact_v4=lambda: calls.append("v4") or v4_decision,
    )

    assert result is v4_decision
    assert calls == ["v4"]


def test_positive_gate_is_inclusive_and_never_consults_v4() -> None:
    taskmaster_decision = object()
    calls: list[str] = []

    result = select_taskmaster_or_exact_v4(
        0.5,
        build_taskmaster_decision=lambda: calls.append("positive") or taskmaster_decision,
        delegate_exact_v4=lambda: calls.append("v4") or object(),
    )

    assert result is taskmaster_decision
    assert calls == ["positive"]


@pytest.mark.parametrize("probability", [True, -0.01, 1.01, float("nan"), float("inf")])
def test_gate_rejects_invalid_probability(probability: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        select_taskmaster_or_exact_v4(
            probability,  # type: ignore[arg-type]
            build_taskmaster_decision=object,
            delegate_exact_v4=object,
        )


def test_preregistration_freezes_seeds_gates_and_one_shot_guards() -> None:
    assert ROUTER_V8_SEEDS == (83, 8419, 27183)
    gates = preregistered_preblind_gates()
    assert gates["all_seeds_must_pass"] is True
    assert gates["original_validation"]["false_taskmaster_max"] == 0  # type: ignore[index]
    assert gates["turn_aware_development_challenge"]["rows"] == 240  # type: ignore[index]
    guards = preregistered_promotion_guards()
    assert guards["probe_rows"] == 84
    assert guards["evaluation_attempts"] == 1
    assert guards["post_probe_tuning_or_retry"] is False
    assert guards["registration_side_effects"] is False


def test_checked_in_contract_and_bound_sources_validate() -> None:
    contract = load_frozen_contract(CONTRACT_PATH, repository_root=REPOSITORY_ROOT)

    assert contract["schema_version"] == ROUTER_V8_SCHEMA
    assert contract["status"] == ROUTER_V8_STATUS
    with pytest.raises(TypeError):
        contract["authorization"]["training"] = True


def test_contract_validation_rejects_semantic_and_source_identity_drift() -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["route_contract"]["taskmaster_threshold"] = 0.6
    unsigned = copy.deepcopy(payload)
    unsigned.pop("integrity")
    payload["integrity"]["canonical_payload_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="route contract"):
        validate_frozen_contract(payload, repository_root=REPOSITORY_ROOT)

    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["artifact_bindings"]["readme"]["sha256"] = "0" * 64
    unsigned = copy.deepcopy(payload)
    unsigned.pop("integrity")
    payload["integrity"]["canonical_payload_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="does not match current bytes"):
        validate_frozen_contract(payload, repository_root=REPOSITORY_ROOT)


def test_contract_validation_rejects_rehashed_unknown_top_level_field() -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["probe_descriptor"] = {"path": "must-never-be-accepted.json"}
    unsigned = copy.deepcopy(payload)
    unsigned.pop("integrity")
    payload["integrity"]["canonical_payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="exactly the frozen top-level fields"):
        validate_frozen_contract(payload, repository_root=REPOSITORY_ROOT)


def test_contract_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "contract.json"
    duplicate.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_frozen_contract(duplicate, repository_root=REPOSITORY_ROOT)
