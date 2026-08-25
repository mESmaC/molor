from collections import UserDict
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from molor.experimental.exact_context_lab import (
    ContextRequest,
    _compare_run_token_identity,
    _manual_greedy_decode,
    _parser,
    _seal_report,
    _sdpa_backend_evidence,
    _sdpa_kernel_context,
    _synthetic_prompt_ids,
    _token_id_sha256,
    _validate_attention_policy,
    _validate_context_request,
    _validate_loaded_config_contract,
    _verify_report_integrity,
    _write_json_exclusive,
    main,
)
from torch.nn.attention import SDPBackend


class CharacterTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_attention_mask: bool,
    ) -> UserDict[str, list[int]]:
        assert add_special_tokens is False
        assert return_attention_mask is False
        return UserDict({"input_ids": [2 + ord(character) % 251 for character in text]})


class FakeCache:
    def __init__(self, *, offloading: bool) -> None:
        self.offloading = offloading
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, list[int] | int]] = []

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        cache_position,
        past_key_values,
        use_cache,
        return_dict,
        logits_to_keep,
    ):
        assert use_cache is True
        assert return_dict is True
        assert logits_to_keep == 1
        past_key_values.length += input_ids.shape[1]
        self.calls.append(
            {
                "input_tokens": input_ids.shape[1],
                "attention_tokens": attention_mask.shape[1],
                "cache_positions": cache_position.tolist(),
            }
        )
        logits = torch.zeros((1, 1, 13), dtype=torch.float32)
        next_token = 2 + past_key_values.length % 9
        logits[:, -1, next_token] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class FakeTelemetry:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_cuda_peaks(self) -> None:
        self.reset_count += 1

    def synchronize(self) -> None:
        return None

    def observe(self, label: str) -> dict[str, object]:
        return {
            "label": label,
            "process": {"current_rss_bytes": 100, "peak_rss_bytes": 120},
            "cuda": None,
        }


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def test_attention_cli_defaults_to_validated_production_like_policy() -> None:
    defaults = _parser().parse_args([])
    deterministic = _parser().parse_args(["--sdpa-backend", "math", "--deterministic-algorithms"])

    assert defaults.modes == ["offloaded", "dynamic"]
    assert defaults.attn_implementation == "sdpa"
    assert defaults.sdpa_backend == "cudnn"
    assert defaults.deterministic_algorithms is False
    assert deterministic.sdpa_backend == "math"
    assert deterministic.deterministic_algorithms is True


def test_attention_policy_rejects_incompatible_combinations_before_loading() -> None:
    _validate_attention_policy(
        attention_implementation="sdpa",
        sdpa_backend="cudnn",
        deterministic_algorithms=False,
        device_type="cuda",
    )
    _validate_attention_policy(
        attention_implementation="eager",
        sdpa_backend="auto",
        deterministic_algorithms=True,
        device_type="cpu",
    )

    with pytest.raises(ValueError, match="eager attention"):
        _validate_attention_policy(
            attention_implementation="eager",
            sdpa_backend="math",
            deterministic_algorithms=False,
            device_type="cpu",
        )
    with pytest.raises(ValueError, match="deterministic algorithms"):
        _validate_attention_policy(
            attention_implementation="sdpa",
            sdpa_backend="cudnn",
            deterministic_algorithms=True,
            device_type="cuda",
        )


def test_sdpa_context_resolves_backend_and_wraps_every_fake_forward() -> None:
    selected: list[object] = []
    active = False

    @contextmanager
    def fake_kernel(backend):
        selected.append(backend)
        yield

    with _sdpa_kernel_context("math", kernel_factory=fake_kernel):
        assert selected == [SDPBackend.MATH]
    with _sdpa_kernel_context("auto", kernel_factory=fake_kernel):
        pass
    assert selected == [SDPBackend.MATH]

    @contextmanager
    def fake_manual_context(name):
        nonlocal active
        assert name == "cudnn"
        active = True
        try:
            yield
        finally:
            active = False

    class ContextCheckingModel(FakeModel):
        def __call__(self, **kwargs):
            assert active is True
            return super().__call__(**kwargs)

    model = ContextCheckingModel()
    _manual_greedy_decode(
        model,
        prompt_ids=torch.tensor([2, 3, 4]),
        cache=FakeCache(offloading=True),
        max_new_tokens=2,
        device=torch.device("cpu"),
        telemetry=FakeTelemetry(),
        sdpa_backend="cudnn",
        sdpa_context_factory=fake_manual_context,
        clock=IncrementingClock(),
    )
    assert active is False
    assert len(model.calls) == 2

    evidence = _sdpa_backend_evidence("math", device_type="cpu")
    assert evidence["resolved_pytorch_enum"] == "MATH"
    assert evidence["availability"]["selected_build_or_library_available"] is True


def test_synthetic_prompt_has_exact_repeatable_token_identity() -> None:
    tokenizer = CharacterTokenizer()

    first = _synthetic_prompt_ids(tokenizer, 513)
    second = _synthetic_prompt_ids(tokenizer, 513)

    assert first.shape == (513,)
    assert torch.equal(first, second)
    assert _token_id_sha256(first) == _token_id_sha256(second)
    assert _token_id_sha256(first) != _token_id_sha256(first[:-1])
    with pytest.raises(ValueError, match="positive"):
        _synthetic_prompt_ids(tokenizer, 0)


def test_context_contract_rejects_any_claim_above_model_maximum() -> None:
    valid = ContextRequest(
        modes=("dynamic", "offloaded"),
        lengths=(8, 12),
        max_new_tokens=4,
        repeats=2,
    )
    _validate_context_request(valid, max_position_embeddings=16)

    with pytest.raises(ValueError, match="exceeds model max_position_embeddings"):
        _validate_context_request(
            ContextRequest(
                modes=("dynamic",),
                lengths=(13,),
                max_new_tokens=4,
                repeats=1,
            ),
            max_position_embeddings=16,
        )
    with pytest.raises(ValueError, match="unique"):
        _validate_context_request(
            ContextRequest(
                modes=("dynamic", "dynamic"),
                lengths=(8,),
                max_new_tokens=1,
                repeats=1,
            ),
            max_position_embeddings=16,
        )


def test_loaded_model_context_contract_must_match_preload_config() -> None:
    values = {
        "max_position_embeddings": 16,
        "vocab_size": 100,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "sliding_window": None,
    }
    source = SimpleNamespace(**values)
    loaded = SimpleNamespace(**values)
    request = ContextRequest(modes=("dynamic",), lengths=(8,), max_new_tokens=2, repeats=1)

    assert _validate_loaded_config_contract(source, loaded, request) == values
    loaded.max_position_embeddings = 12
    with pytest.raises(ValueError, match="max_position_embeddings"):
        _validate_loaded_config_contract(source, loaded, request)


def test_manual_greedy_prefill_decode_has_exact_positions_and_timings() -> None:
    model = FakeModel()
    cache = FakeCache(offloading=True)
    telemetry = FakeTelemetry()

    result = _manual_greedy_decode(
        model,
        prompt_ids=torch.tensor([2, 3, 4, 5]),
        cache=cache,
        max_new_tokens=3,
        device=torch.device("cpu"),
        telemetry=telemetry,
        clock=IncrementingClock(),
    )

    assert result["prompt_tokens"] == 4
    assert result["completion_tokens"] == 3
    assert result["completion_token_ids"] == [6, 7, 8]
    assert result["final_cache_tokens"] == 6
    assert result["timing"]["prefill_seconds"] == 0.25
    assert result["timing"]["ttft_seconds"] == 0.25
    assert [step["phase"] for step in result["steps"]] == [
        "prefill_to_first_token",
        "single_token_decode",
        "single_token_decode",
    ]
    assert model.calls == [
        {"input_tokens": 4, "attention_tokens": 4, "cache_positions": [0, 1, 2, 3]},
        {"input_tokens": 1, "attention_tokens": 5, "cache_positions": [4]},
        {"input_tokens": 1, "attention_tokens": 6, "cache_positions": [5]},
    ]
    assert telemetry.reset_count == 1
    assert result["memory_observations"][0]["label"] == "before_prefill"
    assert result["memory_observations"][-1]["label"] == "run_complete"


def test_manual_greedy_single_completion_is_prefill_only() -> None:
    model = FakeModel()
    result = _manual_greedy_decode(
        model,
        prompt_ids=torch.tensor([2, 3, 4, 5]),
        cache=FakeCache(offloading=False),
        max_new_tokens=1,
        device=torch.device("cpu"),
        telemetry=FakeTelemetry(),
        clock=IncrementingClock(),
    )

    assert result["completion_tokens"] == 1
    assert result["final_cache_tokens"] == 4
    assert result["timing"]["decode_model_step_seconds"] == 0
    assert result["timing"]["decode_tokens_per_second"] is None
    assert len(model.calls) == 1


def _run_record(mode: str, repeat: int, token_ids: list[int]) -> dict[str, object]:
    return {
        "prompt_tokens": 8,
        "repeat": repeat,
        "cache": {"mode": mode},
        "completion_token_ids": token_ids,
        "completion_token_ids_sha256": _token_id_sha256(token_ids),
    }


def test_token_identity_comparison_is_exact_across_modes_and_repeats() -> None:
    request = ContextRequest(
        modes=("dynamic", "offloaded"),
        lengths=(8,),
        max_new_tokens=2,
        repeats=2,
    )
    matching = [
        _run_record("dynamic", 0, [2, 3]),
        _run_record("offloaded", 0, [2, 3]),
        _run_record("dynamic", 1, [2, 3]),
        _run_record("offloaded", 1, [2, 3]),
    ]
    passed = _compare_run_token_identity(matching, expected_request=request)

    assert passed["all_completed_run_tokens_identical_by_prompt"] is True
    assert passed["matrix"]["complete_and_unique"] is True
    assert passed["failure_count"] == 0
    assert len(passed["dynamic_vs_offloaded_matched_pairs"]) == 2

    matching[-1] = _run_record("offloaded", 1, [2, 9])
    failed = _compare_run_token_identity(matching, expected_request=request)
    assert failed["all_completed_run_tokens_identical_by_prompt"] is False
    assert failed["failure_count"] > 0
    assert failed["dynamic_vs_offloaded_matched_pairs"][1]["first_mismatch_index"] == 1


def test_token_identity_comparison_rejects_missing_and_duplicate_matrix_cells() -> None:
    request = ContextRequest(
        modes=("dynamic", "offloaded"),
        lengths=(8,),
        max_new_tokens=2,
        repeats=1,
    )
    dynamic = _run_record("dynamic", 0, [2, 3])

    missing = _compare_run_token_identity([dynamic], expected_request=request)
    duplicate = _compare_run_token_identity(
        [dynamic, dynamic, _run_record("offloaded", 0, [2, 3])],
        expected_request=request,
    )

    assert missing["matrix"]["missing"] == [{"prompt_tokens": 8, "repeat": 0, "mode": "offloaded"}]
    assert missing["failure_count"] == 1
    assert duplicate["matrix"]["duplicates"] == [
        {"prompt_tokens": 8, "repeat": 0, "mode": "dynamic"}
    ]
    assert duplicate["failure_count"] == 1


def test_report_is_content_sealed_and_exclusively_created(tmp_path) -> None:
    path = tmp_path / "exact-context.json"
    sealed = _seal_report(
        {
            "schema_version": "molor.exact-context-report.v1",
            "status": "failed",
            "errors": [{"type": "Example", "message": "record me"}],
        }
    )

    assert _verify_report_integrity(sealed) is True
    _write_json_exclusive(path, sealed)
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        _write_json_exclusive(path, sealed)

    sealed["status"] = "completed"
    assert _verify_report_integrity(sealed) is False


def test_nonfinite_guard_still_writes_sealed_failure_evidence(tmp_path) -> None:
    output = tmp_path / "failed.json"

    exit_code = main(
        [
            "--minimum-free-cuda-gib",
            "nan",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["request"]["minimum_free_cuda_gib"] == "nan"
    assert "finite and non-negative" in report["errors"][0]["message"]
    assert _verify_report_integrity(report) is True
