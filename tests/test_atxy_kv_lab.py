import argparse
import hashlib
import json

import pytest
import torch

from molor.experimental.atxy_kv_lab import (
    _attention_comparison,
    _load_dataset_manifest,
    _parse_rank,
    _reconstruction_metrics,
    _stack_payload,
)
from molor.experimental.hashed_kv_lab import CapturedLayer


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(tmp_path, cases: list[dict[str, object]]):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "molor.atxy-kv-dataset.v1",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dataset_manifest_requires_checksum_bound_disjoint_splits(tmp_path) -> None:
    calibration = tmp_path / "calibration.txt"
    heldout = tmp_path / "heldout.txt"
    calibration.write_text("calibration context", encoding="utf-8")
    heldout.write_text("held-out context", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "calibration-a",
                "split": "calibration",
                "path": calibration.name,
                "sha256": _digest(calibration),
                "token_cap": 8,
            },
            {
                "id": "heldout-a",
                "split": "heldout",
                "path": heldout.name,
                "sha256": _digest(heldout),
                "token_cap": 8,
            },
        ],
    )

    cases = _load_dataset_manifest(manifest)

    assert [case.case_id for case in cases] == ["calibration-a", "heldout-a"]
    assert cases[0].path == calibration.resolve()
    assert cases[1].source_sha256 == _digest(heldout)

    duplicate = _write_manifest(
        tmp_path,
        [
            {
                "id": "calibration-a",
                "split": "calibration",
                "path": calibration.name,
                "sha256": _digest(calibration),
                "token_cap": 8,
            },
            {
                "id": "heldout-a",
                "split": "heldout",
                "path": calibration.name,
                "sha256": _digest(calibration),
                "token_cap": 8,
            },
        ],
    )
    with pytest.raises(ValueError, match="duplicate dataset content digest"):
        _load_dataset_manifest(duplicate)


def test_dataset_manifest_fails_closed_on_source_drift(tmp_path) -> None:
    calibration = tmp_path / "calibration.txt"
    heldout = tmp_path / "heldout.txt"
    calibration.write_text("original calibration", encoding="utf-8")
    heldout.write_text("heldout", encoding="utf-8")
    expected = _digest(calibration)
    calibration.write_text("modified calibration", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "calibration-a",
                "split": "calibration",
                "path": calibration.name,
                "sha256": expected,
                "token_cap": 4,
            },
            {
                "id": "heldout-a",
                "split": "heldout",
                "path": heldout.name,
                "sha256": _digest(heldout),
                "token_cap": 4,
            },
        ],
    )

    with pytest.raises(ValueError, match="sha256 mismatch"):
        _load_dataset_manifest(manifest)


def test_stack_payload_uses_atxy_layer_token_head_channel_order() -> None:
    captures = tuple(
        CapturedLayer(
            layer_index=layer,
            query_positions=(3,),
            queries=torch.zeros(4, 1, 3),
            keys=torch.full((2, 4, 3), float(layer + 1)),
            values=torch.full((2, 4, 3), float(10 + layer)),
        )
        for layer in range(2)
    )

    keys = _stack_payload(captures, "keys")
    values = _stack_payload(captures, "values")

    assert keys.shape == values.shape == (2, 4, 2, 3)
    assert torch.equal(keys[1], torch.full((4, 2, 3), 2.0))
    assert torch.equal(values[0], torch.full((4, 2, 3), 10.0))


def test_attention_comparison_isolates_key_and_value_error() -> None:
    generator = torch.Generator().manual_seed(27)
    query = torch.randn(8, generator=generator)
    keys = torch.randn(16, 8, generator=generator)
    values = torch.randn(16, 6, generator=generator)
    identity = _attention_comparison(
        query,
        keys,
        values,
        keys,
        values,
        top_k=4,
    )
    value_only = _attention_comparison(
        query,
        keys,
        values,
        keys,
        values * 0.5,
        top_k=4,
    )
    key_only = _attention_comparison(
        query,
        keys,
        values,
        keys.roll(1, dims=0),
        values,
        top_k=4,
    )

    assert identity["attention_kl"] == pytest.approx(0.0, abs=1e-8)
    assert identity["output_relative_l2"] == pytest.approx(0.0, abs=1e-8)
    assert value_only["attention_kl"] == pytest.approx(0.0, abs=1e-8)
    assert value_only["output_relative_l2"] > 0
    assert key_only["attention_kl"] > 0


def test_reconstruction_metrics_and_rank_parser() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    exact = _reconstruction_metrics(value, value.clone())

    assert exact["relative_l2"] == 0.0
    assert exact["cosine_similarity"] == pytest.approx(1.0)
    assert _parse_rank("14x8x64").layers == 14
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_rank("14x8")
