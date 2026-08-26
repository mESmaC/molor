from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.embedded_grounded_reconstruction import (
    LEAVES,
    MAXIMUM_RENDERED_TOKENS,
    OUTPUT_TIERS,
    ReconstructionPlan,
    SPLITS,
    SYSTEM_PROMPT,
    TASK_TYPE,
    assert_identical_materializations,
    materialize_embedded_grounded_reconstruction,
)
from molor.experimental.length_tier_shards import tokenizer_identity
from molor.identity import canonical_json_bytes


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, str]], **kwargs: object
    ) -> list[int]:
        self.calls.append(dict(kwargs))
        return [1] * max(1, sum(len(message["content"]) for message in conversation))


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path,
        "sha256": _sha(payload),
        "bytes": len(payload),
        "records": len(payload.splitlines()),
    }


def _source_fixture(
    root: Path,
    *,
    leak_group: bool = False,
    corrupt_content_pin: bool = False,
) -> tuple[Path, dict[tuple[str, str], dict[str, object]]]:
    outputs: list[dict[str, object]] = []
    expected: dict[tuple[str, str], dict[str, object]] = {}
    for leaf_index, leaf in enumerate(LEAVES):
        for split_index, split in enumerate(SPLITS):
            label = f"{leaf}-{split}"
            if (leaf, split) == ("embedded-code-generation", "train"):
                text = ("αβ source line\r\n" * 700) + "terminal character Ω"
            else:
                text = f"Verbatim {label}\nwith exact spacing."
            data_row = canonical_json_bytes({"text": text})
            data_payload = data_row + b"\n"
            data_path = root / "reference" / leaf / f"{split}.jsonl"
            data_pin = _write_bytes(data_path, data_payload)
            group_split = "validation" if leak_group and split == "train" else split
            group_id = f"private-group-{group_split}"
            provenance = {
                "content_sha256": (
                    "0" * 64 if corrupt_content_pin and not expected else _sha(text.encode("utf-8"))
                ),
                "disposition": "emitted",
                "file_path": f"tree/{leaf_index}/file-{split_index}.txt",
                "group_id": group_id,
                "leaf": leaf,
                "record_sha256": _sha(data_row),
                "repo": f"fixture-repo-{split}",
                "source": "fixture-source",
                "source_line": split_index + 1,
                "source_split": split,
                "split": split,
                "upstream_id": f"fixture-upstream-{leaf_index}-{split_index}",
            }
            provenance_payload = canonical_json_bytes(provenance) + b"\n"
            provenance_path = root / "reference" / leaf / f"{split}.provenance.jsonl"
            provenance_pin = _write_bytes(provenance_path, provenance_payload)
            outputs.append(
                {
                    "data": {
                        "bytes": data_pin["bytes"],
                        "path": data_path.relative_to(root).as_posix(),
                        "records": 1,
                        "sha256": data_pin["sha256"],
                    },
                    "leaf": leaf,
                    "modality": "reference_cpt",
                    "provenance": {
                        "bytes": provenance_pin["bytes"],
                        "path": provenance_path.relative_to(root).as_posix(),
                        "records": 1,
                        "sha256": provenance_pin["sha256"],
                    },
                    "split": split,
                }
            )
            expected[(leaf, split)] = {"text": text, "provenance": provenance}
    manifest = {"schema_version": "molor.embedded-source-experts.v1", "outputs": outputs}
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    )
    return manifest_path, expected


def _plan(tmp_path: Path, source_manifest: Path) -> ReconstructionPlan:
    tokenizer_root = tmp_path / "qwen-tokenizer"
    tokenizer_root.mkdir(parents=True, exist_ok=True)
    (tokenizer_root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "unit-qwen-chat-template"}),
        encoding="utf-8",
    )
    identity = tokenizer_identity(tokenizer_root)
    return ReconstructionPlan(
        source_manifest=source_manifest,
        source_manifest_sha256=_sha(source_manifest.read_bytes()),
        tokenizer_path=tokenizer_root,
        tokenizer_aggregate_sha256=identity["aggregate_sha256"],
        chat_template_sha256=identity["chat_template"]["sha256"],
    )


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_verbatim_chunking_tiers_provenance_and_deterministic_rebuild(tmp_path) -> None:
    source_manifest, expected = _source_fixture(tmp_path / "source")
    plan = _plan(tmp_path, source_manifest)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    tokenizer = FakeQwenTokenizer()

    first = materialize_embedded_grounded_reconstruction(
        plan=plan,
        output_directory=first_root,
        tokenizer=tokenizer,
    )
    second = materialize_embedded_grounded_reconstruction(
        plan=plan,
        output_directory=second_root,
        tokenizer=FakeQwenTokenizer(),
    )

    assert first == second
    identities = assert_identical_materializations(first_root, second_root)
    assert len(identities) == 28
    assert first["task_contract"]["task_type"] == TASK_TYPE
    assert first["task_contract"]["contains_synthetic_question_answering"] is False
    assert first["task_contract"]["proves_answer_quality"] is False
    assert first["accounting"]["selected_input_rows"] == 6
    assert first["accounting"]["emitted_message_records"] > 6
    assert first["accounting"]["zero_silent_omissions"] is True
    assert first["accounting"]["every_source_concatenation_exact"] is True
    assert first["accounting"]["truncated_rows"] == 0
    assert first["accounting"]["deferred_rows"] == 0
    assert len(first["outputs"]) == len(LEAVES) * len(SPLITS) * len(OUTPUT_TIERS)

    observed: dict[tuple[str, str], list[tuple[dict[str, object], dict[str, object]]]] = {}
    for leaf in LEAVES:
        for split in SPLITS:
            observed[(leaf, split)] = []
            for tier in OUTPUT_TIERS:
                message_rows = _records(first_root / "messages" / leaf / split / f"{tier}.jsonl")
                provenance_rows = _records(
                    first_root / "provenance" / leaf / split / f"{tier}.jsonl"
                )
                assert len(message_rows) == len(provenance_rows)
                for message_row, provenance_row in zip(
                    message_rows, provenance_rows, strict=True
                ):
                    assert set(message_row) == {"messages"}
                    messages = message_row["messages"]
                    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
                    assert messages[2]["content"]
                    rendered = sum(len(message["content"]) for message in messages)
                    assert rendered == provenance_row["rendered_tokens"]
                    assert rendered <= MAXIMUM_RENDERED_TOKENS
                    assert provenance_row["tier"] == tier
                    assert provenance_row[
                        "assistant_target_is_contiguous_verbatim_source_text"
                    ] is True
                    observed[(leaf, split)].append((message_row, provenance_row))

    for key, source in expected.items():
        rows = sorted(observed[key], key=lambda pair: pair[1]["chunk_index"])
        reconstructed = "".join(pair[0]["messages"][2]["content"] for pair in rows)
        assert reconstructed == source["text"]
        boundaries = [
            (pair[1]["character_start"], pair[1]["character_end"]) for pair in rows
        ]
        assert boundaries[0][0] == 0
        assert boundaries[-1][1] == len(source["text"])
        assert all(left[1] == right[0] for left, right in zip(boundaries, boundaries[1:]))
        for message_row, provenance_row in rows:
            user = message_row["messages"][1]["content"]
            upstream = source["provenance"]
            assert upstream["repo"] in user
            assert upstream["file_path"] in user
            assert key[0] in user
            assert f"Chunk index: {provenance_row['chunk_index']}" in user
            assert upstream["group_id"] not in user
            assert upstream["upstream_id"] not in user
            assert provenance_row["source_provenance"] == upstream

    ledger = _records(first_root / "accounting.jsonl")
    assert len(ledger) == 6
    assert all(row["concatenation_exact"] is True for row in ledger)
    assert all(row["characters_omitted"] == 0 for row in ledger)
    assert tokenizer.calls
    assert all(call["truncation"] is False for call in tokenizer.calls)
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)


def test_manifest_tokenizer_and_row_pins_fail_before_publish(tmp_path) -> None:
    source_manifest, _ = _source_fixture(tmp_path / "source")
    valid = _plan(tmp_path, source_manifest)
    bad_manifest = ReconstructionPlan(
        source_manifest=valid.source_manifest,
        source_manifest_sha256="0" * 64,
        tokenizer_path=valid.tokenizer_path,
        tokenizer_aggregate_sha256=valid.tokenizer_aggregate_sha256,
        chat_template_sha256=valid.chat_template_sha256,
    )
    with pytest.raises(ValueError, match="pinned input SHA-256 mismatch"):
        materialize_embedded_grounded_reconstruction(
            plan=bad_manifest,
            output_directory=tmp_path / "bad-manifest",
            tokenizer=FakeQwenTokenizer(),
        )
    bad_tokenizer = ReconstructionPlan(
        source_manifest=valid.source_manifest,
        source_manifest_sha256=valid.source_manifest_sha256,
        tokenizer_path=valid.tokenizer_path,
        tokenizer_aggregate_sha256="0" * 64,
        chat_template_sha256=valid.chat_template_sha256,
    )
    with pytest.raises(ValueError, match="tokenizer aggregate"):
        materialize_embedded_grounded_reconstruction(
            plan=bad_tokenizer,
            output_directory=tmp_path / "bad-tokenizer",
            tokenizer=FakeQwenTokenizer(),
        )
    corrupt_manifest, _ = _source_fixture(
        tmp_path / "corrupt-source", corrupt_content_pin=True
    )
    with pytest.raises(ValueError, match="content mismatch"):
        materialize_embedded_grounded_reconstruction(
            plan=_plan(tmp_path / "corrupt-plan", corrupt_manifest),
            output_directory=tmp_path / "bad-row",
            tokenizer=FakeQwenTokenizer(),
        )
    assert not (tmp_path / "bad-manifest").exists()
    assert not (tmp_path / "bad-tokenizer").exists()
    assert not (tmp_path / "bad-row").exists()


def test_opposite_split_group_leakage_fails_closed(tmp_path) -> None:
    source_manifest, _ = _source_fixture(tmp_path / "source", leak_group=True)
    with pytest.raises(ValueError, match="opposite-split source-group leakage"):
        materialize_embedded_grounded_reconstruction(
            plan=_plan(tmp_path, source_manifest),
            output_directory=tmp_path / "output",
            tokenizer=FakeQwenTokenizer(),
        )
    assert not (tmp_path / "output").exists()
