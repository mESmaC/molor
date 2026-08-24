from dataclasses import replace
import json

import pytest

from molor.manifests import AdapterBankManifest, BaseModelManifest
from molor.memory import MemoryHead, MemoryVersionManifest, SessionMemoryPin
from molor.runtime import ModelStackManifest, ServingPinManifest

from test_manifests import adapter, base, digest


def adapter_bank(base_manifest: BaseModelManifest | None = None) -> AdapterBankManifest:
    foundation = base_manifest or base()
    return AdapterBankManifest(
        "skills",
        "1",
        (
            adapter("math", "1", base_manifest=foundation),
            adapter("code", "2", base_manifest=foundation),
        ),
    )


def memory_version(
    base_manifest: BaseModelManifest,
    *,
    marker: str = "3",
    version_id: str = "memory/v1",
    scope_id: str = "user/alice",
    cutoff: int = 4,
) -> MemoryVersionManifest:
    return MemoryVersionManifest(
        version_id=version_id,
        scope_id=scope_id,
        base_fingerprint=base_manifest.fingerprint,
        parent_version_fingerprint=None,
        consolidated_through_sequence=cutoff,
        consolidation_job_fingerprint=digest("9"),
        corpus_manifest_fingerprint=digest("a"),
        adapter=adapter("memory/base", marker, base_manifest=base_manifest),
        trainer_fingerprint=digest("b"),
        evaluation_report_fingerprint=digest("c"),
        policy_fingerprint=digest("d"),
    )


def test_model_stack_preserves_bank_identity_and_serializes_complete_manifests() -> None:
    bank = adapter_bank()
    original_bank_fingerprint = bank.fingerprint
    version = memory_version(bank.base)

    stack = ModelStackManifest(bank, version)
    payload = stack.to_dict()

    assert bank.fingerprint == original_bank_fingerprint
    assert stack.bank_fingerprint == original_bank_fingerprint
    assert stack.base_fingerprint == bank.base.fingerprint
    assert stack.memory_version_fingerprint == version.fingerprint
    assert payload["bank_manifest"] == bank.to_dict()
    assert payload["memory_version_manifest"] == version.to_dict()
    json.dumps(payload, allow_nan=False)


def test_memory_version_changes_stack_identity_without_changing_bank_identity() -> None:
    bank = adapter_bank()
    first = ModelStackManifest(bank, memory_version(bank.base, marker="3"))
    second = ModelStackManifest(
        bank,
        memory_version(bank.base, marker="4", version_id="memory/v2"),
    )
    no_memory = ModelStackManifest(bank)

    assert first.fingerprint != second.fingerprint
    assert first.fingerprint != no_memory.fingerprint
    assert first.bank_fingerprint == second.bank_fingerprint == no_memory.bank_fingerprint
    assert no_memory.memory_version_fingerprint is None


def test_model_stack_rejects_memory_version_for_a_different_exact_base() -> None:
    bank = adapter_bank()
    changed_base = base(weights="7")
    version = memory_version(changed_base)

    with pytest.raises(ValueError, match="different frozen bases"):
        ModelStackManifest(bank, version)


def test_model_stack_rejects_adapter_reused_on_both_axes() -> None:
    foundation = base()
    version = memory_version(foundation)
    duplicated = AdapterBankManifest(
        "skills",
        "1",
        (version.adapter, adapter("code", "2", base_manifest=foundation)),
    )

    with pytest.raises(ValueError, match="same adapter"):
        ModelStackManifest(duplicated, version)

    colliding_id = AdapterBankManifest(
        "skills",
        "2",
        (
            adapter("memory/base", "8", base_manifest=foundation),
            adapter("code", "2", base_manifest=foundation),
        ),
    )
    with pytest.raises(ValueError, match="disjoint logical IDs"):
        ModelStackManifest(colliding_id, version)


def test_serving_pin_binds_stack_head_generation_version_and_retrieval_cutoff() -> None:
    bank = adapter_bank()
    version = memory_version(bank.base, cutoff=4)
    stack = ModelStackManifest(bank, version)
    head = MemoryHead(
        scope_id=version.scope_id,
        generation=7,
        version_fingerprint=version.fingerprint,
        consolidated_through_sequence=version.consolidated_through_sequence,
    )
    pin = SessionMemoryPin.from_head(
        "session/one",
        head,
        retrieval_cutoff_sequence=9,
    )

    serving = ServingPinManifest(stack, pin)
    later_head_generation = ServingPinManifest(stack, replace(pin, head_generation=8))
    later_retrieval_cutoff = ServingPinManifest(
        stack,
        replace(pin, retrieval_cutoff_sequence=10),
    )

    assert serving.model_stack_fingerprint == stack.fingerprint
    assert serving.memory_pin_fingerprint == pin.fingerprint
    assert serving.fingerprint != later_head_generation.fingerprint
    assert serving.fingerprint != later_retrieval_cutoff.fingerprint
    json.dumps(serving.to_dict(), allow_nan=False)


def test_serving_pin_rejects_missing_or_mismatched_memory_identity() -> None:
    bank = adapter_bank()
    version = memory_version(bank.base, cutoff=4)
    stack = ModelStackManifest(bank, version)

    with pytest.raises(ValueError, match="requires a session memory pin"):
        ServingPinManifest(stack)

    mismatched_version = SessionMemoryPin(
        session_id="session/one",
        scope_id=version.scope_id,
        head_generation=1,
        version_fingerprint=digest("e"),
        consolidated_through_sequence=4,
        retrieval_cutoff_sequence=4,
    )
    with pytest.raises(ValueError, match="different memory version"):
        ServingPinManifest(stack, mismatched_version)

    mismatched_scope = replace(mismatched_version, scope_id="user/bob")
    with pytest.raises(ValueError, match="different scopes"):
        ServingPinManifest(stack, mismatched_scope)

    mismatched_cutoff = SessionMemoryPin(
        session_id="session/one",
        scope_id=version.scope_id,
        head_generation=1,
        version_fingerprint=version.fingerprint,
        consolidated_through_sequence=5,
        retrieval_cutoff_sequence=5,
    )
    with pytest.raises(ValueError, match="different consolidation cutoffs"):
        ServingPinManifest(stack, mismatched_cutoff)


def test_stack_without_memory_accepts_no_pin_or_an_explicit_empty_head_pin() -> None:
    stack = ModelStackManifest(adapter_bank())
    empty_pin = SessionMemoryPin.from_head(
        "session/one",
        MemoryHead.empty("user/alice"),
        retrieval_cutoff_sequence=3,
    )

    unpinned = ServingPinManifest(stack)
    pinned_retrieval = ServingPinManifest(stack, empty_pin)

    assert unpinned.memory_pin_fingerprint is None
    assert pinned_retrieval.memory_pin_fingerprint == empty_pin.fingerprint

    versioned_pin = SessionMemoryPin(
        session_id="session/one",
        scope_id="user/alice",
        head_generation=1,
        version_fingerprint=digest("e"),
        consolidated_through_sequence=0,
        retrieval_cutoff_sequence=0,
    )
    with pytest.raises(ValueError, match="absent from the model stack"):
        ServingPinManifest(stack, versioned_pin)
