from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "experiments" / "atxy-staged-kv-v1" / "validate_paged_backing.py"
SPEC = importlib.util.spec_from_file_location("paged_backing_allocation_probe_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def tiny_config(*, gate_bytes: int = 1024) -> object:
    return probe.ProbeConfig(
        model_id="tiny-unpinned-test",
        num_layers=2,
        max_cache_len=8,
        page_token_capacity=4,
        num_key_value_heads=1,
        key_head_dim=2,
        value_head_dim=1,
        dtype=torch.float32,
        memory_gate_bytes=gate_bytes,
        pin_cpu_backing=False,
    )


def verified_test_provenance() -> dict[str, object]:
    return {
        "verified": True,
        "scope": "injected tiny unpinned unit-test geometry",
    }


def memory_snapshot(
    *,
    rss: int,
    private: int | None,
    supported: bool = True,
) -> dict[str, object]:
    return {
        "current_rss_bytes": rss,
        "peak_rss_bytes": rss,
        "private_committed_bytes": private,
        "source": "mock PROCESS_MEMORY_COUNTERS_EX",
        "gate_measurement_supported": supported,
        "portable_fallback": not supported,
        "limitations": [] if supported else ["mock private bytes unavailable"],
    }


class SnapshotSequence:
    def __init__(self, *snapshots: Mapping[str, object]) -> None:
        self.snapshots = list(snapshots)
        self.calls = 0

    def __call__(self) -> Mapping[str, object]:
        if self.calls >= len(self.snapshots):
            raise AssertionError("memory reader called too many times")
        result = self.snapshots[self.calls]
        self.calls += 1
        return result


class RecordingAllocator:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ) -> torch.Tensor:
        call_index = len(self.calls)
        self.calls.append(
            {
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "pin_memory": pin_memory,
            }
        )
        if call_index == self.fail_on_call:
            raise RuntimeError(f"injected allocation failure {call_index}")
        return torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)


def test_dry_run_projects_without_allocating_and_marks_gate_unevaluated() -> None:
    allocator = RecordingAllocator()
    reader = SnapshotSequence(memory_snapshot(rss=1000, private=None, supported=False))

    receipt = probe.run_probe(
        tiny_config(),
        allocate=False,
        memory_reader=reader,
        provenance_reader=verified_test_provenance,
        allocator=allocator,
    )

    assert receipt["status"] == "projected"
    assert receipt["mode"] == "dry_run_projection"
    assert allocator.calls == []
    assert reader.calls == 1
    assert receipt["measurements"]["allocation_attempted"] is False
    assert receipt["checks"]["measurement_supported_for_allocation"] is False
    assert receipt["checks"]["incremental_memory_gate_passed"] is None
    assert receipt["cache_telemetry"]["storage"]["live_backing_page_allocations"] == 0
    assert receipt["cache_telemetry"]["geometry"]["logical_backing_bytes"] == 192
    assert all(receipt["checks"]["projection"].values())


def test_tiny_unpinned_allocation_populates_every_page_with_bounded_inputs() -> None:
    allocator = RecordingAllocator()
    reader = SnapshotSequence(
        memory_snapshot(rss=1000, private=2000),
        memory_snapshot(rss=1400, private=2400),
    )

    receipt = probe.run_probe(
        tiny_config(),
        allocate=True,
        memory_reader=reader,
        provenance_reader=verified_test_provenance,
        allocator=allocator,
        permit_unpinned_test_allocation=True,
    )

    assert receipt["status"] == "completed"
    assert len(allocator.calls) == 8
    assert all(call["device"] == torch.device("cpu") for call in allocator.calls)
    assert all(call["pin_memory"] is False for call in allocator.calls)
    assert receipt["population"] == {
        "performed": True,
        "completed": True,
        "layers_completed": 2,
        "page_tensor_copies_completed": 8,
        "expected_page_tensor_copies": 8,
        "elements_copied": 48,
        "bytes_copied": 192,
        "reusable_input_tensors": 2,
        "reusable_input_bytes": 48,
        "bounded_input_contract": (
            "one reusable unpinned CPU K page and one reusable unpinned CPU V page"
        ),
        "validation": "torch.equal immediately after every page copy",
    }
    assert all(receipt["checks"]["allocation"].values())
    assert receipt["measurements"]["delta"] == {
        "incremental_rss_bytes": 400,
        "incremental_private_committed_bytes": 400,
        "maximum_incremental_process_bytes": 400,
        "gate_bytes": 1024,
        "rss_nonnegative": True,
        "private_nonnegative": True,
        "rss_within_gate": True,
        "private_within_gate": True,
        "passed": True,
        "gate_policy": ("both incremental current RSS and Windows PrivateUsage must be <= gate"),
    }
    assert receipt["cache_telemetry"]["storage"]["live_logical_backing_bytes"] == 192
    assert receipt["cache_telemetry"]["storage"]["staging_allocated"] is False


def test_unsupported_private_measurement_fails_before_any_allocation() -> None:
    allocator = RecordingAllocator()
    reader = SnapshotSequence(memory_snapshot(rss=1000, private=None, supported=False))

    receipt = probe.run_probe(
        tiny_config(),
        allocate=True,
        memory_reader=reader,
        provenance_reader=verified_test_provenance,
        allocator=allocator,
        permit_unpinned_test_allocation=True,
    )

    assert receipt["status"] == "failed"
    assert receipt["errors"] == [
        {
            "phase": "memory_preflight",
            "type": "ProbeValidationError",
            "message": (
                "allocation refused: exact RSS and private-byte measurement is unsupported"
            ),
        }
    ]
    assert allocator.calls == []
    assert receipt["measurements"]["allocation_attempted"] is False
    assert receipt["cache_telemetry"]["health"]["poisoned"] is False


def test_partial_layer_allocation_fails_closed_and_records_postfailure_memory() -> None:
    allocator = RecordingAllocator(fail_on_call=5)
    reader = SnapshotSequence(
        memory_snapshot(rss=1000, private=2000),
        memory_snapshot(rss=1300, private=2300),
    )

    receipt = probe.run_probe(
        tiny_config(),
        allocate=True,
        memory_reader=reader,
        provenance_reader=verified_test_provenance,
        allocator=allocator,
        permit_unpinned_test_allocation=True,
    )

    assert receipt["status"] == "failed"
    assert receipt["errors"][0]["phase"] == "allocate_and_populate"
    assert receipt["errors"][0]["type"] == "PagedSharedStagingCacheError"
    assert receipt["cache_telemetry"]["health"]["poisoned"] is True
    assert receipt["cache_telemetry"]["health"]["active_poison_reason"] == (
        "backing_allocation_failure"
    )
    assert receipt["cache_telemetry"]["storage"]["live_backing_page_allocations"] == 4
    assert receipt["population"]["layers_completed"] == 1
    assert receipt["population"]["page_tensor_copies_completed"] == 4
    assert receipt["measurements"]["after"]["current_rss_bytes"] == 1300
    assert receipt["measurements"]["delta"]["passed"] is True


def test_population_verification_failure_never_reports_completed(monkeypatch) -> None:
    reader = SnapshotSequence(
        memory_snapshot(rss=1000, private=2000),
        memory_snapshot(rss=1200, private=2200),
    )
    monkeypatch.setattr(probe.torch, "equal", lambda _left, _right: False)

    receipt = probe.run_probe(
        tiny_config(),
        allocate=True,
        memory_reader=reader,
        provenance_reader=verified_test_provenance,
        permit_unpinned_test_allocation=True,
    )

    assert receipt["status"] == "failed"
    assert receipt["errors"][0] == {
        "phase": "allocate_and_populate",
        "type": "ProbeValidationError",
        "message": "key page population verification failed",
    }
    assert receipt["population"]["performed"] is True
    assert receipt["population"]["completed"] is False
    assert receipt["population"]["page_tensor_copies_completed"] == 0
    assert receipt["cache_telemetry"]["storage"]["live_backing_page_allocations"] == 4
    assert receipt["measurements"]["after"]["current_rss_bytes"] == 1200


def test_memory_gate_failure_never_reports_completed() -> None:
    reader = SnapshotSequence(
        memory_snapshot(rss=1000, private=2000),
        memory_snapshot(rss=2500, private=3500),
    )

    receipt = probe.run_probe(
        tiny_config(gate_bytes=1024),
        allocate=True,
        memory_reader=reader,
        provenance_reader=verified_test_provenance,
        permit_unpinned_test_allocation=True,
    )

    assert receipt["status"] == "failed"
    assert receipt["errors"] == [
        {
            "phase": "memory_postflight",
            "type": "ProbeValidationError",
            "message": "incremental process memory exceeded the configured gate",
        }
    ]
    assert all(receipt["checks"]["allocation"].values())
    assert receipt["checks"]["incremental_memory_gate_passed"] is False
    assert receipt["measurements"]["delta"]["rss_within_gate"] is False
    assert receipt["measurements"]["delta"]["private_within_gate"] is False


def test_receipt_seal_and_atomic_exclusive_publication(tmp_path: Path) -> None:
    receipt = probe.run_probe(
        tiny_config(),
        allocate=False,
        memory_reader=SnapshotSequence(memory_snapshot(rss=1000, private=None, supported=False)),
        provenance_reader=verified_test_provenance,
    )
    sealed = probe.seal_receipt(receipt)
    output = tmp_path / "nested" / "receipt.json"

    probe.write_json_atomic_exclusive(output, sealed)

    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == sealed
    assert probe.verify_receipt_seal(observed) is True
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        probe.write_json_atomic_exclusive(output, {"replacement": True})
    assert output.read_bytes() == original
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_frozen_sources_and_six_gib_cli_ceiling_are_pinned() -> None:
    provenance = probe.verify_frozen_provenance()
    config = probe.ProbeConfig.frozen_qwen3_06b()

    assert provenance["verified"] is True
    assert provenance["files"]["model_config"]["sha256"] == (probe.FROZEN_MODEL_CONFIG_SHA256)
    assert provenance["files"]["paged_cache_source"]["sha256"] == (probe.FROZEN_CACHE_SOURCE_SHA256)
    assert config.memory_gate_bytes == 6 * 2**30
    assert probe._parse_gate_gib("5.5") == int(5.5 * 2**30)
    with pytest.raises(probe.argparse.ArgumentTypeError, match="<= 6 GiB"):
        probe._parse_gate_gib("6.01")
