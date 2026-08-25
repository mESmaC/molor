from __future__ import annotations

import json
import os
from pathlib import Path

from molor.teacher.cli import _parser, _resolve_serve_artifacts


def _write_adapter(root: Path, relative: str) -> str:
    path = root / relative
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_id": str(root / "models" / "Qwen3-1.7B")}),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return str(path.resolve())


def _write_router(root: Path, relative: str) -> str:
    path = root / relative
    path.mkdir(parents=True)
    (path / "router_config.json").write_text(
        json.dumps({"model": {"model_id": str(root / "models" / "Qwen3-1.7B")}}),
        encoding="utf-8",
    )
    (path / "router.safetensors").write_bytes(b"weights")
    return str(path.resolve())


def test_06_teacher_defaults_to_base_only_when_only_17_artifacts_exist(tmp_path: Path) -> None:
    _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
    )
    _write_router(tmp_path, "checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled")
    args = _parser(tmp_path, {}).parse_args(["serve"])

    artifacts = _resolve_serve_artifacts(args, tmp_path, {})

    assert args.model == "Qwen/Qwen3-0.6B"
    assert artifacts.expert_a == ""
    assert artifacts.expert_b == ""
    assert artifacts.expert_paths == ()
    assert artifacts.router == ""


def test_explicit_17_model_recovers_its_compatible_defaults(tmp_path: Path) -> None:
    expert_a = _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
    )
    router = _write_router(tmp_path, "checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled")
    model = tmp_path / "models" / "Qwen3-1.7B"
    args = _parser(tmp_path, {}).parse_args(["serve", "--model", str(model)])

    artifacts = _resolve_serve_artifacts(args, tmp_path, {})

    assert artifacts.expert_a == expert_a
    assert artifacts.router == router


def test_explicit_17_hub_id_does_not_inherit_local_artifacts(tmp_path: Path) -> None:
    _write_adapter(
        tmp_path,
        "checkpoints/qwen3-1.7b-corpus-experts/embedded-knowledge-v1",
    )
    _write_router(tmp_path, "checkpoints/qwen3-1.7b-corpus-router-v4-mlp-controlled")
    args = _parser(tmp_path, {}).parse_args(["serve", "--model", "Qwen/Qwen3-1.7B"])

    artifacts = _resolve_serve_artifacts(args, tmp_path, {})

    assert artifacts.expert_a == ""
    assert artifacts.router == ""


def test_explicit_and_environment_artifacts_keep_precedence(tmp_path: Path) -> None:
    environment = {
        "MOLOR_EXPERT_A": "env-a",
        "MOLOR_EXPERT_B": "env-b",
        "MOLOR_EXPERTS": f"env-c{os.pathsep}env-d",
        "MOLOR_ROUTER": "env-router",
    }
    args = _parser(tmp_path, environment).parse_args(
        ["serve", "--expert-a", "cli-a", "--router", "cli-router"]
    )

    artifacts = _resolve_serve_artifacts(args, tmp_path, environment)

    assert artifacts.expert_a == "cli-a"
    assert artifacts.expert_b == "env-b"
    assert artifacts.expert_paths == ("env-c", "env-d")
    assert artifacts.router == "cli-router"
