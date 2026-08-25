from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.atxy_page_oracle_dataset import (
    CHECKPOINTS,
    CHECKSUM_NAME,
    DATASET_SCHEMA,
    DATASET_STATUS,
    FULL_MANIFEST_NAME,
    FULL_TOKEN_COUNT,
    PILOT_CHECKPOINTS,
    PILOT_MANIFEST_NAME,
    SMOKE_MANIFEST_NAME,
    DatasetAssets,
    RenderedCase,
    _load_local_tokenizer,
    render_dataset,
    write_assets_exclusive,
)
from molor.experimental.atxy_page_oracle_lab import _load_dataset_manifest


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL = REPOSITORY / "models" / "Qwen3-1.7B"
DATASET = REPOSITORY / "experiments" / "atxy-page-oracle-v1" / "dataset"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dummy_assets(payload: bytes) -> DatasetAssets:
    case = RenderedCase(case_id="dummy", source_text="", token_ids=(), checkpoints=())
    return DatasetAssets(
        files={"prompts/dummy.txt": payload},
        full_cases=(case,),
        pilot_case=case,
        smoke_case=case,
    )


def test_generated_asset_writer_is_idempotent_and_rejects_drift(tmp_path: Path) -> None:
    assets = _dummy_assets(b"synthetic\n")
    expected = tmp_path / "prompts" / "dummy.txt"

    assert write_assets_exclusive(tmp_path, assets) == (expected,)
    assert write_assets_exclusive(tmp_path, assets) == ()

    expected.write_bytes(b"changed\n")
    with pytest.raises(FileExistsError, match="different bytes"):
        write_assets_exclusive(tmp_path, assets)
    assert expected.read_bytes() == b"changed\n"


@pytest.mark.skipif(not (MODEL / "tokenizer.json").is_file(), reason="local Qwen tokenizer absent")
def test_checked_in_dataset_exactly_matches_deterministic_qwen_render() -> None:
    tokenizer = _load_local_tokenizer(MODEL)
    assets = render_dataset(tokenizer)

    assert set(assets.files) == {
        CHECKSUM_NAME,
        FULL_MANIFEST_NAME,
        PILOT_MANIFEST_NAME,
        SMOKE_MANIFEST_NAME,
        "prompts/mixed-development-01.txt",
        "prompts/mixed-development-02.txt",
        "prompts/mixed-development-03.txt",
        "prompts/mixed-development-04.txt",
        "smoke-prompts/mixed-development-01-smoke.txt",
    }
    for relative, expected in assets.files.items():
        assert (DATASET / relative).read_bytes() == expected

    assert len(assets.full_cases) == 4
    assert len({case.source_text for case in assets.full_cases}) == 4
    for case in assets.full_cases:
        assert len(case.token_ids) == FULL_TOKEN_COUNT
        assert tuple(checkpoint.token_count for checkpoint in case.checkpoints) == CHECKPOINTS
        spans = tuple(checkpoint.support_token_span for checkpoint in case.checkpoints)
        assert len(set(spans)) == len(CHECKPOINTS)
        assert all(span[1] <= checkpoint - 1 for span, checkpoint in zip(spans, CHECKPOINTS))
        assert "taskmaster" not in case.source_text.casefold()
        assert all(
            f"DOMAIN {domain}" in case.source_text
            for domain in ("SOFTWARE", "EMBEDDED", "FINANCE", "TERMINAL")
        )
        for checkpoint in case.checkpoints:
            assert all(
                checkpoint.support_token_ids != distractor
                for distractor in checkpoint.distractor_token_ids
            )

    assert assets.pilot_case.case_id == assets.full_cases[0].case_id
    assert assets.pilot_case.source_text == assets.full_cases[0].source_text
    assert assets.pilot_case.token_ids == assets.full_cases[0].token_ids
    assert (
        tuple(checkpoint.token_count for checkpoint in assets.pilot_case.checkpoints)
        == PILOT_CHECKPOINTS
    )

    assert len(assets.smoke_case.token_ids) == 4_096
    assert assets.smoke_case.token_ids == assets.full_cases[0].token_ids[:4_096]
    assert (
        assets.smoke_case.source_text
        == assets.full_cases[0].source_text[
            : assets.full_cases[0].checkpoints[0].query_end_character
        ]
    )


def test_checked_in_manifests_and_checksum_sidecar_are_strict_and_consistent() -> None:
    full_payload = json.loads((DATASET / FULL_MANIFEST_NAME).read_text(encoding="utf-8"))
    pilot_payload = json.loads((DATASET / PILOT_MANIFEST_NAME).read_text(encoding="utf-8"))
    smoke_payload = json.loads((DATASET / SMOKE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert set(full_payload) == {"schema_version", "status", "cases"}
    assert set(pilot_payload) == {"schema_version", "status", "cases"}
    assert set(smoke_payload) == {"schema_version", "status", "cases"}
    assert (
        full_payload["schema_version"]
        == pilot_payload["schema_version"]
        == smoke_payload["schema_version"]
        == DATASET_SCHEMA
    )
    assert (
        full_payload["status"]
        == pilot_payload["status"]
        == smoke_payload["status"]
        == DATASET_STATUS
    )
    assert len(full_payload["cases"]) == 4
    assert len(pilot_payload["cases"]) == 1
    assert len(smoke_payload["cases"]) == 1
    assert pilot_payload["cases"][0]["path"] == "prompts/mixed-development-01.txt"
    assert pilot_payload["cases"][0]["sha256"] == full_payload["cases"][0]["sha256"]

    full_snapshot = _load_dataset_manifest(DATASET / FULL_MANIFEST_NAME)
    pilot_snapshot = _load_dataset_manifest(DATASET / PILOT_MANIFEST_NAME)
    smoke_snapshot = _load_dataset_manifest(DATASET / SMOKE_MANIFEST_NAME)
    assert tuple(case.token_count for case in full_snapshot.cases) == (FULL_TOKEN_COUNT,) * 4
    assert pilot_snapshot.cases[0].token_count == FULL_TOKEN_COUNT
    assert smoke_snapshot.cases[0].token_count == 4_096
    assert tuple(query.token_count for query in full_snapshot.cases[0].queries) == CHECKPOINTS
    assert (
        tuple(query.token_count for query in pilot_snapshot.cases[0].queries) == PILOT_CHECKPOINTS
    )
    assert tuple(query.token_count for query in smoke_snapshot.cases[0].queries) == (4_096,)

    expected: dict[str, str] = {}
    for line in (DATASET / CHECKSUM_NAME).read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        assert len(digest) == 64
        assert relative not in expected
        expected[relative] = digest
    assert set(expected) == {
        FULL_MANIFEST_NAME,
        PILOT_MANIFEST_NAME,
        SMOKE_MANIFEST_NAME,
        *(f"prompts/mixed-development-{index:02d}.txt" for index in range(1, 5)),
        "smoke-prompts/mixed-development-01-smoke.txt",
    }
    for relative, digest in expected.items():
        assert _sha256((DATASET / relative).read_bytes()) == digest
