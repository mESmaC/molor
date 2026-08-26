from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.corpus_mixtures import (
    CorpusMixtureRecipe,
    CorpusMixtureShard,
    CorpusShardDescriptor,
    apply_weight_overrides,
    build_corpus_mixture,
    corpus_mixture_recipe_json_bytes,
    load_corpus_mixture_recipe,
    main,
    parse_weight_overrides,
)


def _chat(label: str, *, pad: bool = False) -> dict[str, object]:
    user = f"  {label}\r\n  " if pad else label
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"answer-{label}"},
        ]
    }


def _write_rows(path: Path, rows: list[object]) -> tuple[str, int]:
    encoded: list[bytes] = []
    for row in rows:
        if isinstance(row, bytes):
            encoded.append(row if row.endswith(b"\n") else row + b"\n")
        else:
            encoded.append(
                (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            )
    content = b"".join(encoded)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(rows)


def _descriptor(
    path: Path,
    *,
    shard_id: str,
    leaf_id: str,
    split: str,
    template_mode: str = "direct-v1",
) -> CorpusShardDescriptor:
    content = path.read_bytes()
    return CorpusShardDescriptor(
        shard_id=shard_id,
        shard_revision="1",
        leaf_id=leaf_id,
        leaf_revision="1",
        split=split,
        template_mode=template_mode,
        path=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
        record_count=len(content.splitlines()),
    )


def _training_recipe(
    shards: tuple[CorpusMixtureShard, ...],
    *,
    target_records: int,
    shortfall_policy: str = "error",
) -> CorpusMixtureRecipe:
    return CorpusMixtureRecipe(
        recipe_id="unit-mixture-v1",
        revision="1",
        usage="training",
        output_split="train",
        template_mode="direct-v1",
        seed="unit-mixture-seed-v1",
        target_records=target_records,
        shortfall_policy=shortfall_policy,
        shards=shards,
    )


def test_evaluation_build_is_deterministic_messages_only_and_fully_accounted(tmp_path) -> None:
    guard = tmp_path / "train-guard.jsonl"
    _write_rows(guard, [_chat("leak"), _chat("guard-only")])

    shard_a_path = tmp_path / "validation-a.jsonl"
    _write_rows(
        shard_a_path,
        [
            _chat("a1"),
            _chat("a1", pad=True),
            _chat("leak"),
            b"{not-json}\n",
            {"messages": [{"role": "assistant", "content": "missing user"}]},
            _chat("a2"),
            _chat("a3"),
        ],
    )
    shard_b_path = tmp_path / "validation-b.jsonl"
    _write_rows(
        shard_b_path,
        [_chat("b1"), _chat("a1"), _chat("b2"), _chat("b3")],
    )

    shard_a = CorpusMixtureShard(
        _descriptor(
            shard_a_path,
            shard_id="validation-a-v1",
            leaf_id="business-operations-v1",
            split="validation",
        ),
        weight=1.0,
        cap=2,
    )
    shard_b = CorpusMixtureShard(
        _descriptor(
            shard_b_path,
            shard_id="validation-b-v1",
            leaf_id="growth-gtm-v1",
            split="validation",
        ),
        weight=2.0,
    )
    recipe = CorpusMixtureRecipe(
        recipe_id="business-eval-v1",
        revision="1",
        usage="evaluation",
        output_split="validation",
        template_mode="direct-v1",
        seed="business-eval-seed-v1",
        target_records=3,
        shortfall_policy="error",
        shards=(shard_b, shard_a),
        leakage_guard_shards=(
            _descriptor(
                guard,
                shard_id="train-guard-v1",
                leaf_id="business-operations-v1",
                split="train",
            ),
        ),
    )

    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    first = build_corpus_mixture(
        recipe,
        recipe_directory=tmp_path,
        output_directory=first_output,
    )
    second = build_corpus_mixture(
        recipe,
        recipe_directory=tmp_path,
        output_directory=second_output,
    )

    assert first == second
    assert first["input_records"] == 11
    assert first["input_accounting"] == {
        "emitted": 3,
        "duplicate": 2,
        "quarantined": 2,
        "rejected": 1,
        "capped": 3,
    }
    assert first["zero_silent_drops"] is True
    assert first["leakage_guard"]["records_scanned"] == 2
    assert first["leakage_guard"]["unique_normalized_messages"] == 2
    assert first["reason_accounting"]["train_eval_exact_match"] == 1
    assert first["reason_accounting"]["normalized_messages_duplicate"] == 2

    output_rows = [
        json.loads(line) for line in (first_output / "messages.jsonl").read_text().splitlines()
    ]
    assert len(output_rows) == 3
    assert all(set(row) == {"messages"} for row in output_rows)
    assert len({json.dumps(row, sort_keys=True) for row in output_rows}) == 3
    assert all("leak" not in json.dumps(row) for row in output_rows)

    provenance = [
        json.loads(line) for line in (first_output / "provenance.jsonl").read_text().splitlines()
    ]
    assert len(provenance) == 11
    assert {row["status"] for row in provenance} == {
        "emitted",
        "duplicate",
        "quarantined",
        "rejected",
        "capped",
    }
    assert all("messages" not in row for row in provenance)
    for name in (
        "messages.jsonl",
        "provenance.jsonl",
        "resolved_recipe.json",
        "aggregate_report.json",
        "sha256sums.txt",
    ):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()

    dry_output = tmp_path / "dry-output"
    dry = build_corpus_mixture(
        recipe,
        recipe_directory=tmp_path,
        output_directory=dry_output,
        dry_run=True,
    )
    assert not dry_output.exists()
    assert dry["dry_run"] is True
    assert dry["content_identity"] == first["content_identity"]
    assert dry["artifacts"]["provenance.jsonl"] == first["artifacts"]["provenance.jsonl"]


def test_weight_overrides_adjust_leaf_mix_without_mutating_recipe_file(tmp_path, capsys) -> None:
    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    _write_rows(a_path, [_chat(f"a-{index}") for index in range(10)])
    _write_rows(b_path, [_chat(f"b-{index}") for index in range(10)])
    recipe = _training_recipe(
        (
            CorpusMixtureShard(
                _descriptor(a_path, shard_id="a-v1", leaf_id="leaf-a-v1", split="train"),
                weight=1.0,
            ),
            CorpusMixtureShard(
                _descriptor(b_path, shard_id="b-v1", leaf_id="leaf-b-v1", split="train"),
                weight=3.0,
            ),
        ),
        target_records=4,
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_bytes(corpus_mixture_recipe_json_bytes(recipe))
    canonical_bytes = recipe_path.read_bytes()

    baseline = build_corpus_mixture(
        recipe,
        recipe_directory=tmp_path,
        output_directory=tmp_path / "baseline",
    )
    assert baseline["emitted_by_leaf"] == {"leaf-a-v1": 1, "leaf-b-v1": 3}

    assert (
        main(
            [
                "--recipe",
                str(recipe_path),
                "--output",
                str(tmp_path / "override"),
                "--weight",
                "leaf:leaf-a-v1=9",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["emitted_by_leaf"] == {"leaf-a-v1": 3, "leaf-b-v1": 1}
    assert printed["weight_overrides"] == {"leaf:leaf-a-v1": 9.0}
    assert printed["canonical_recipe_sha256"] == recipe.fingerprint
    assert printed["resolved_recipe_sha256"] != recipe.fingerprint
    assert recipe_path.read_bytes() == canonical_bytes

    overrides = parse_weight_overrides(["leaf:leaf-a-v1=2", "shard:b-v1=7"])
    resolved = apply_weight_overrides(recipe, overrides)
    reversed_resolved = apply_weight_overrides(
        recipe,
        {"shard:b-v1": 7.0, "leaf:leaf-a-v1": 2.0},
    )
    assert recipe.shards[0].weight == 1.0
    assert {shard.descriptor.shard_id: shard.weight for shard in resolved.shards} == {
        "a-v1": 2.0,
        "b-v1": 7.0,
    }
    assert reversed_resolved.fingerprint == resolved.fingerprint
    with pytest.raises(ValueError, match="duplicate weight override"):
        parse_weight_overrides(["leaf:leaf-a-v1=2", "leaf:leaf-a-v1=3"])
    with pytest.raises(KeyError, match="unknown leaf"):
        apply_weight_overrides(recipe, {"leaf:missing-v1": 1.0})

    decimal_recipe = _training_recipe(
        (
            CorpusMixtureShard(
                _descriptor(a_path, shard_id="decimal-a-v1", leaf_id="decimal-v1", split="train"),
                weight=0.1,
            ),
            CorpusMixtureShard(
                _descriptor(b_path, shard_id="decimal-b-v1", leaf_id="decimal-v1", split="train"),
                weight=0.2,
            ),
        ),
        target_records=2,
    )
    first_decimal = apply_weight_overrides(decimal_recipe, {"leaf:decimal-v1": 0.3})
    second_decimal = apply_weight_overrides(decimal_recipe, {"leaf:decimal-v1": 0.3})
    assert [shard.weight for shard in first_decimal.shards] == [0.1, 0.2]
    assert first_decimal.fingerprint == second_decimal.fingerprint


def test_recipe_loader_is_strict_content_addressed_and_split_safe(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_rows(train_path, [_chat("train")])
    _write_rows(validation_path, [_chat("validation")])
    train = _descriptor(train_path, shard_id="train-v1", leaf_id="leaf-v1", split="train")
    validation = _descriptor(
        validation_path,
        shard_id="validation-v1",
        leaf_id="leaf-v1",
        split="validation",
    )
    recipe = _training_recipe((CorpusMixtureShard(train, 1.0),), target_records=1)
    path = tmp_path / "recipe.json"
    path.write_bytes(corpus_mixture_recipe_json_bytes(recipe))

    loaded = load_corpus_mixture_recipe(path)
    assert loaded == recipe
    assert loaded.fingerprint == recipe.fingerprint

    tampered = recipe.to_dict()
    tampered["target_records"] = 2
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        load_corpus_mixture_recipe(path)

    duplicate = corpus_mixture_recipe_json_bytes(recipe).decode("utf-8").rstrip()
    path.write_text(duplicate[:-1] + ', "recipe_id": "other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_corpus_mixture_recipe(path)

    with pytest.raises(ValueError, match="must emit the train split"):
        CorpusMixtureRecipe(
            recipe_id="bad-v1",
            revision="1",
            usage="training",
            output_split="validation",
            template_mode="direct-v1",
            seed="seed",
            target_records=1,
            shortfall_policy="error",
            shards=(CorpusMixtureShard(validation, 1.0),),
        )
    with pytest.raises(ValueError, match="require at least one pinned train guard"):
        CorpusMixtureRecipe(
            recipe_id="bad-eval-v1",
            revision="1",
            usage="evaluation",
            output_split="validation",
            template_mode="direct-v1",
            seed="seed",
            target_records=1,
            shortfall_policy="error",
            shards=(CorpusMixtureShard(validation, 1.0),),
        )
    with pytest.raises(ValueError, match="template_mode"):
        CorpusMixtureRecipe(
            recipe_id="bad-template-v1",
            revision="1",
            usage="training",
            output_split="train",
            template_mode="trace-v1",
            seed="seed",
            target_records=1,
            shortfall_policy="error",
            shards=(CorpusMixtureShard(train, 1.0),),
        )


def test_pins_shortfall_and_guard_validation_fail_closed(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    _write_rows(source, [_chat("one"), _chat("two")])
    descriptor = _descriptor(source, shard_id="source-v1", leaf_id="leaf-v1", split="train")
    strict = _training_recipe((CorpusMixtureShard(descriptor, 1.0),), target_records=3)
    with pytest.raises(ValueError, match="eligible records"):
        build_corpus_mixture(
            strict,
            recipe_directory=tmp_path,
            output_directory=tmp_path / "strict-output",
        )
    assert not (tmp_path / "strict-output").exists()

    permissive = replace_recipe_shortfall(strict)
    report = build_corpus_mixture(
        permissive,
        recipe_directory=tmp_path,
        output_directory=tmp_path / "permissive-output",
    )
    assert report["emitted_records"] == 2
    assert report["shortfall_records"] == 1
    assert report["zero_silent_drops"] is True

    bad_pin = CorpusShardDescriptor(
        **{**descriptor.to_dict(), "sha256": "0" * 64}  # type: ignore[arg-type]
    )
    bad_recipe = _training_recipe((CorpusMixtureShard(bad_pin, 1.0),), target_records=1)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_corpus_mixture(
            bad_recipe,
            recipe_directory=tmp_path,
            output_directory=tmp_path / "bad-pin-output",
        )

    eval_path = tmp_path / "eval.jsonl"
    guard_path = tmp_path / "bad-guard.jsonl"
    _write_rows(eval_path, [_chat("eval")])
    _write_rows(guard_path, [b"not-json\n"])
    evaluation = CorpusMixtureRecipe(
        recipe_id="guard-failure-v1",
        revision="1",
        usage="evaluation",
        output_split="test",
        template_mode="direct-v1",
        seed="seed",
        target_records=1,
        shortfall_policy="error",
        shards=(
            CorpusMixtureShard(
                _descriptor(eval_path, shard_id="eval-v1", leaf_id="leaf-v1", split="test"),
                1.0,
            ),
        ),
        leakage_guard_shards=(
            _descriptor(
                guard_path,
                shard_id="bad-guard-v1",
                leaf_id="leaf-v1",
                split="train",
            ),
        ),
    )
    with pytest.raises(ValueError, match="leakage guard"):
        build_corpus_mixture(
            evaluation,
            recipe_directory=tmp_path,
            output_directory=tmp_path / "bad-guard-output",
        )


def replace_recipe_shortfall(recipe: CorpusMixtureRecipe) -> CorpusMixtureRecipe:
    return CorpusMixtureRecipe(
        recipe_id=recipe.recipe_id,
        revision=recipe.revision,
        usage=recipe.usage,
        output_split=recipe.output_split,
        template_mode=recipe.template_mode,
        seed=recipe.seed,
        target_records=recipe.target_records,
        shortfall_policy="emit_available",
        shards=recipe.shards,
        leakage_guard_shards=recipe.leakage_guard_shards,
    )


def test_native_tool_messages_are_validated_and_preserved_losslessly(tmp_path: Path) -> None:
    source = tmp_path / "native-tool.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "Look up order 42"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-42",
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "arguments": '{"order_id":42}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "", "tool_call_id": "call-42"},
            {
                "role": "assistant",
                "content": "Order 42 has no recorded result.",
                "tool_calls": [],
            },
        ]
    }
    _write_rows(source, [row])
    descriptor = _descriptor(
        source,
        shard_id="native-tool-v1",
        leaf_id="tool-use-execution",
        split="train",
        template_mode="tool-direct-v1",
    )
    recipe = CorpusMixtureRecipe(
        recipe_id="native-tool-mixture-v1",
        revision="1",
        usage="training",
        output_split="train",
        template_mode="tool-direct-v1",
        seed="native-tool-seed-v1",
        target_records=1,
        shortfall_policy="error",
        shards=(CorpusMixtureShard(descriptor, 1.0),),
    )
    output = tmp_path / "native-tool-output"
    report = build_corpus_mixture(
        recipe,
        recipe_directory=tmp_path,
        output_directory=output,
    )
    assert report["input_accounting"]["emitted"] == 1
    assert report["input_accounting"]["quarantined"] == 0

    emitted = json.loads((output / "messages.jsonl").read_text(encoding="utf-8"))
    call = emitted["messages"][1]["tool_calls"][0]
    assert call == {
        "function": {"arguments": '{"order_id":42}', "name": "lookup_order"},
        "id": "call-42",
        "type": "function",
    }
    assert emitted["messages"][2] == {
        "content": "",
        "role": "tool",
        "tool_call_id": "call-42",
    }
    assert emitted["messages"][3]["tool_calls"] == []


def test_invalid_native_tool_linkage_is_categorized_without_silent_drop(tmp_path: Path) -> None:
    source = tmp_path / "tool-linkage.jsonl"
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": {}},
    }
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Use prior observations"},
                {"role": "user", "content": "Summarize"},
                {"role": "tool", "content": "prior result", "tool_call_id": "prior-1"},
                {"role": "assistant", "content": "summary"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
                {"role": "tool", "content": "orphan", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": "result", "tool_call_id": "wrong"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "assistant", "content": "answer"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        call,
                        {
                            "type": "function",
                            "function": {"name": "other", "arguments": {}},
                        },
                    ],
                },
                {"role": "tool", "content": "one", "tool_call_id": "call-1"},
                {"role": "tool", "content": "two"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        {
            "messages": [
                [["role", "user"], ["content", "coercible but not an object"]],
                [["role", "assistant"], ["content", "must be quarantined"]],
            ]
        },
    ]
    _write_rows(source, rows)
    descriptor = _descriptor(
        source,
        shard_id="tool-linkage-v1",
        leaf_id="tool-use-execution",
        split="train",
        template_mode="tool-direct-v1",
    )
    recipe = CorpusMixtureRecipe(
        recipe_id="tool-linkage-mixture-v1",
        revision="1",
        usage="training",
        output_split="train",
        template_mode="tool-direct-v1",
        seed="tool-linkage-seed-v1",
        target_records=1,
        shortfall_policy="error",
        shards=(CorpusMixtureShard(descriptor, 1.0),),
    )

    report = build_corpus_mixture(
        recipe,
        recipe_directory=tmp_path,
        output_directory=tmp_path / "tool-linkage-output",
    )

    assert report["input_accounting"] == {
        "emitted": 1,
        "duplicate": 0,
        "quarantined": 5,
        "rejected": 0,
        "capped": 0,
    }
    assert report["reason_accounting"] == {
        "invalid_chat_contract": 1,
        "mixed_tool_call_id_mode": 1,
        "orphan_tool_result": 1,
        "selected": 1,
        "tool_result_count_mismatch": 1,
        "unknown_or_duplicate_tool_call_id": 1,
    }
    assert report["zero_silent_drops"] is True
