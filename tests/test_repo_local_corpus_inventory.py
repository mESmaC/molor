from __future__ import annotations

import json
from pathlib import Path

import pytest

from molor.experimental.repo_local_corpus_inventory import (
    build_repo_local_corpus_inventory,
)
from molor.identity import fingerprint_json


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "expert-rack-v5"
    / "expert-taxonomy-v1.json"
)


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    sources = tmp_path / "sources"
    data.mkdir()
    sources.mkdir()
    for name in ("finqa", "multidogo", "taskmaster", "tat-qa"):
        container = sources / name
        container.mkdir()
        (container / "README.md").write_text("fixture", encoding="utf-8")
    return data, sources


def test_inventory_accounts_artifacts_and_admission_without_parsing_rows(tmp_path: Path) -> None:
    data, sources = _roots(tmp_path)
    (data / "terminal-cli.jsonl").write_text(
        '{"opaque":"one"}\n{"opaque":"two"}\n', encoding="utf-8"
    )
    (data / "terminal-cli.validation.jsonl").write_text('{"opaque":"held-out"}\n', encoding="utf-8")
    (data / "terminal-cli.build-report.json").write_text("{}", encoding="utf-8")
    (data / "financial-analysis-verified.jsonl").write_text(
        '{"opaque":"alias"}\n', encoding="utf-8"
    )

    inventory = build_repo_local_corpus_inventory(
        data_root=data,
        source_root=sources,
        taxonomy_path=TAXONOMY_PATH,
    )

    assert inventory["schema_version"] == "molor.repo-local-corpus-inventory.v1"
    assert inventory["summary"]["eligible_data_artifacts"] == 4
    assert inventory["summary"]["admitted_training_rows_before_cross-corpus-dedup"] == 2
    assert inventory["summary"]["zero_silent_eligible_omissions"] is True
    artifacts = {item["path"]: item for item in inventory["artifacts"]}
    assert artifacts["terminal-cli.jsonl"]["admitted_for_new_expert_training"] is True
    assert artifacts["terminal-cli.validation.jsonl"]["artifact_role"] == "evaluation"
    assert artifacts["financial-analysis-verified.jsonl"]["dataset_disposition"] == "superseded"
    semantic = {key: value for key, value in inventory.items() if key != "content_sha256"}
    assert inventory["content_sha256"] == fingerprint_json(semantic)


def test_inventory_fails_closed_on_unknown_eligible_artifact(tmp_path: Path) -> None:
    data, sources = _roots(tmp_path)
    (data / "unassigned-corpus.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no unique disposition"):
        build_repo_local_corpus_inventory(
            data_root=data,
            source_root=sources,
            taxonomy_path=TAXONOMY_PATH,
        )


def test_inventory_is_deterministic(tmp_path: Path) -> None:
    data, sources = _roots(tmp_path)
    (data / "taskmaster-coffee-api-v2.train.jsonl").write_text(
        '{"opaque":"tool"}\n', encoding="utf-8"
    )
    first = build_repo_local_corpus_inventory(
        data_root=data,
        source_root=sources,
        taxonomy_path=TAXONOMY_PATH,
    )
    second = build_repo_local_corpus_inventory(
        data_root=data,
        source_root=sources,
        taxonomy_path=TAXONOMY_PATH,
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
