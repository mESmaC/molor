from __future__ import annotations

from types import SimpleNamespace
import weakref

import pytest
import torch

import molor.experimental.exact_context_v2_lab as v2
from molor.experimental.exact_context_v2_lab import (
    CacheAwareRuntimeTelemetry,
    StaticWorkerRequest,
    _windows_counter_payload,
    build_lazy_offloaded_static_cache,
    development_command_contract,
    main,
    observe_after_cache_release,
    postflight_recheck_local_model_artifact,
    process_memory_v2,
    summarize_unique_storage_records,
    unique_live_cache_storage,
    validate_single_worker_request,
)


class FakeStaticCache:
    def __init__(
        self,
        *,
        config,
        max_cache_len: int,
        offloading: bool,
        offload_only_non_sliding: bool,
    ) -> None:
        self.config = config
        self.maximum = max_cache_len
        self.offloading = offloading
        self.only_non_sliding = offload_only_non_sliding
        self.is_initialized = False
        self.layers = []

    def get_max_length(self) -> int:
        return self.maximum


def test_single_worker_request_is_strictly_one_static_offloaded_cell() -> None:
    valid = StaticWorkerRequest(
        length=40_952,
        max_new_tokens=8,
        max_cache_len=40_960,
        trial_id=2,
    )
    validate_single_worker_request(valid, max_position_embeddings=40_960)

    replacements = [
        {"cache_implementation": "dynamic"},
        {"cache_placement": "dynamic"},
        {"repeats": 2},
        {"trial_id": -1},
        {"max_cache_len": 40_959},
        {"length": 40_953},
    ]
    for replacement in replacements:
        values = {**valid.__dict__, **replacement}
        with pytest.raises(ValueError):
            validate_single_worker_request(
                StaticWorkerRequest(**values),
                max_position_embeddings=40_960,
            )


def test_static_cache_constructor_is_lazy_and_forces_full_offload() -> None:
    cache = build_lazy_offloaded_static_cache(
        SimpleNamespace(name="config"),
        max_cache_len=32,
        static_cache_class=FakeStaticCache,
    )

    assert cache.maximum == 32
    assert cache.offloading is True
    assert cache.only_non_sliding is False
    assert cache.is_initialized is False


def test_unique_cache_storage_deduplicates_views_and_splits_devices() -> None:
    backing = torch.zeros(32, dtype=torch.float32)
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(keys=backing[:16], values=backing[16:]),
            SimpleNamespace(keys=torch.zeros(4, dtype=torch.float32), values=None),
        ]
    )

    measured = unique_live_cache_storage(cache)

    assert measured["unique_storage_count"] == 2
    assert measured["cpu_bytes"] == backing.untyped_storage().nbytes() + 16
    assert measured["cuda_bytes"] == 0
    assert measured["total_bytes"] == measured["cpu_bytes"]

    synthetic = summarize_unique_storage_records(
        [
            ("cpu", 1, 100),
            ("cpu", 1, 100),
            ("cuda:0", 2, 200),
            ("meta", 3, 50),
        ]
    )
    assert synthetic == {
        "unique_storage_count": 3,
        "cpu_bytes": 100,
        "cuda_bytes": 200,
        "other_device_bytes": 50,
        "total_bytes": 350,
        "by_device_bytes": {"cpu": 100, "cuda:0": 200, "meta": 50},
    }


def test_windows_private_usage_is_exposed_without_aliasing_working_set() -> None:
    payload = _windows_counter_payload(
        SimpleNamespace(
            WorkingSetSize=11,
            PeakWorkingSetSize=22,
            PrivateUsage=33,
            PagefileUsage=44,
            PeakPagefileUsage=55,
        )
    )

    assert payload["current_rss_bytes"] == 11
    assert payload["peak_rss_bytes"] == 22
    assert payload["private_committed_bytes"] == 33
    assert payload["pagefile_usage_bytes"] == 44
    assert "PROCESS_MEMORY_COUNTERS_EX" in payload["source"]


def test_live_process_memory_has_the_v2_private_commit_field() -> None:
    payload = process_memory_v2()

    assert payload["current_rss_bytes"] >= 0
    if payload["private_committed_bytes"] is not None:
        assert payload["private_committed_bytes"] >= 0


def test_cache_aware_cpu_telemetry_uses_a_weak_reference() -> None:
    class WeakCache:
        def __init__(self) -> None:
            self.layers = []

    cache = WeakCache()
    reference = weakref.ref(cache)
    telemetry = CacheAwareRuntimeTelemetry(
        torch.device("cpu"),
        process_memory_reader=lambda: {
            "current_rss_bytes": 1,
            "peak_rss_bytes": 2,
            "private_committed_bytes": 3,
        },
    )
    telemetry.bind_cache(cache)

    assert telemetry.observe("bound")["live_cache_storage"]["total_bytes"] == 0
    telemetry.clear_cache()
    del cache
    assert reference() is None
    assert telemetry.observe("cleared")["cuda"] is None


def test_post_release_observation_orders_release_collect_and_measurement() -> None:
    events: list[str] = []

    class WeakCache:
        pass

    class FakeTelemetry:
        def __init__(self, cache) -> None:
            self.reference = weakref.ref(cache)

        def cache_reference(self):
            return self.reference

        def synchronize(self) -> None:
            events.append("synchronize")

        def clear_cache(self) -> None:
            events.append("clear_cache")

        def observe(self, label: str):
            events.append(f"observe:{label}")
            cache = self.reference() if self.reference is not None else None
            return {
                "label": label,
                "live_cache_storage": unique_live_cache_storage(cache),
            }

    cache = WeakCache()
    holder = {"cache": cache}
    telemetry = FakeTelemetry(cache)
    del cache

    def release() -> None:
        events.append("release")
        holder["cache"] = None

    observation = observe_after_cache_release(
        telemetry,  # type: ignore[arg-type]
        release=release,
        collect=lambda: events.append("collect"),
        empty_cuda_cache=lambda: events.append("empty_cuda_cache"),
    )

    assert holder["cache"] is None
    assert observation["cache_release"] == {
        "cache_object_collected": True,
        "live_cache_storage_zero": True,
        "release_succeeded": True,
    }
    assert events == [
        "synchronize",
        "release",
        "collect",
        "empty_cuda_cache",
        "synchronize",
        "observe:after_cache_release",
        "clear_cache",
    ]


def test_retained_second_reference_cannot_claim_success_or_zero_storage() -> None:
    class WeakCache:
        def __init__(self) -> None:
            self.layers = [SimpleNamespace(keys=torch.zeros(8), values=torch.zeros(8))]

    cache = WeakCache()
    retained = cache
    holder = {"cache": cache}
    telemetry = CacheAwareRuntimeTelemetry(
        torch.device("cpu"),
        process_memory_reader=lambda: {
            "current_rss_bytes": 1,
            "peak_rss_bytes": 1,
            "private_committed_bytes": 1,
        },
    )
    telemetry.bind_cache(cache)
    del cache

    observation = observe_after_cache_release(
        telemetry,
        release=lambda: holder.update(cache=None),
        collect=lambda: None,
        empty_cuda_cache=lambda: None,
    )

    assert retained is not None
    assert observation["cache_release"]["cache_object_collected"] is False
    assert observation["cache_release"]["release_succeeded"] is False
    assert observation["live_cache_storage"]["total_bytes"] > 0


def test_decode_exception_traceback_is_conservatively_reported_as_retained() -> None:
    class WeakCache:
        def __init__(self) -> None:
            self.layers = [SimpleNamespace(keys=torch.zeros(8), values=None)]

    def failing_decode(cache) -> None:
        assert cache is not None
        raise RuntimeError("simulated decode failure")

    cache = WeakCache()
    holder = {"cache": cache}
    telemetry = CacheAwareRuntimeTelemetry(
        torch.device("cpu"),
        process_memory_reader=lambda: {
            "current_rss_bytes": 1,
            "peak_rss_bytes": 1,
            "private_committed_bytes": 1,
        },
    )
    telemetry.bind_cache(cache)
    del cache

    with pytest.raises(RuntimeError, match="simulated decode failure"):
        try:
            failing_decode(holder["cache"])
        finally:
            observation = observe_after_cache_release(
                telemetry,
                release=lambda: holder.update(cache=None),
                collect=lambda: None,
                empty_cuda_cache=lambda: None,
            )
            assert observation["cache_release"]["cache_object_collected"] is False
            assert observation["cache_release"]["release_succeeded"] is False
            assert observation["live_cache_storage"]["total_bytes"] > 0


def test_model_artifact_postflight_runs_during_failure_and_preserves_mismatch() -> None:
    report = {
        "identities": {
            "model_artifact": {"stack_fingerprint": "before"},
        }
    }
    calls: list[tuple[str, bool, None]] = []

    def mismatch_reader(model_id, *, is_local, revision):
        calls.append((model_id, is_local, revision))
        return {"stack_fingerprint": "after"}

    decode_error = RuntimeError("decode failed")
    postflight_recheck_local_model_artifact(
        report,
        model_id="model",
        is_local=True,
        active_error=decode_error,
        identity_reader=mismatch_reader,
        clock=lambda: 1.0,
    )

    assert calls == [("model", True, None)]
    recheck = report["identities"]["model_artifact_recheck"]
    assert recheck["status"] == "completed"
    assert recheck["matches_preload_identity"] is False

    with pytest.raises(RuntimeError, match="identity changed"):
        postflight_recheck_local_model_artifact(
            report,
            model_id="model",
            is_local=True,
            active_error=None,
            identity_reader=mismatch_reader,
            clock=lambda: 1.0,
        )


def test_existing_output_is_refused_before_worker_execution(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve me", encoding="utf-8")
    called = False

    def forbidden_worker(_args, _report) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(v2, "_run_worker", forbidden_worker)

    assert main(["--output", str(output)]) == 1
    assert called is False
    assert output.read_text(encoding="utf-8") == "preserve me"
    assert "before model load" in capsys.readouterr().err


def test_command_contract_emits_three_distinct_single_process_cells(capsys) -> None:
    contract = development_command_contract()

    assert contract["status"] == "development_unfrozen"
    assert contract["one_cell_per_process"] is True
    assert [item["trial_id"] for item in contract["commands"]] == [0, 1, 2]
    outputs = []
    for item in contract["commands"]:
        argv = item["argv"]
        assert argv[:3] == [
            r".\.venv\Scripts\python.exe",
            "-m",
            "molor.experimental.exact_context_v2_lab",
        ]
        assert argv.count("--length") == 1
        assert argv.count("--trial-id") == 1
        outputs.append(argv[argv.index("--output") + 1])
    assert len(set(outputs)) == 3

    assert main(["--emit-command-contract"]) == 0
    assert "development_unfrozen" in capsys.readouterr().out
