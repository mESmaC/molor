from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molor.experimental.business_reference_mixtures import (
    ReferenceMixtureRecipe,
    ReferenceMixtureShard,
    ReferenceShardDescriptor,
    apply_overrides,
    build_reference_mixture,
    load_reference_mixture_recipe,
    main,
    parse_cap_overrides,
    parse_weight_overrides,
    recipes_from_reference_manifest,
    reference_mixture_recipe_json_bytes,
)


def _document(
    label: str,
    *,
    identifier_label: str | None = None,
    title: str | None = None,
    text: str | None = None,
) -> dict[str, str]:
    return {
        "document_id": hashlib.sha256((identifier_label or label).encode()).hexdigest(),
        "title": title if title is not None else f"Title {label}",
        "text": text if text is not None else f"Exact source text for {label}.",
    }


def _write_rows(path: Path, rows: list[object]) -> tuple[str, int]:
    encoded: list[bytes] = []
    for row in rows:
        if isinstance(row, bytes):
            encoded.append(row if row.endswith(b"\n") else row + b"\n")
        else:
            encoded.append(
                (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
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
) -> ReferenceShardDescriptor:
    content = path.read_bytes()
    return ReferenceShardDescriptor(
        shard_id=shard_id,
        shard_revision="1",
        leaf_id=leaf_id,
        leaf_revision="1",
        split=split,
        path=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
        record_count=len(content.splitlines()),
    )


def _guards(
    tmp_path: Path, leaves: tuple[str, ...], output_split: str
) -> tuple[ReferenceShardDescriptor, ...]:
    descriptors: list[ReferenceShardDescriptor] = []
    for leaf in leaves:
        for split in sorted({"train", "validation", "test"}.difference({output_split})):
            path = tmp_path / f"guard-{leaf}-{split}.jsonl"
            _write_rows(path, [_document(f"guard-{leaf}-{split}")])
            descriptors.append(
                _descriptor(
                    path,
                    shard_id=f"guard-{leaf}-{split}-v1",
                    leaf_id=leaf,
                    split=split,
                )
            )
    return tuple(descriptors)


def test_build_is_deterministic_document_only_guarded_and_fully_accounted(tmp_path) -> None:
    validation_guard = tmp_path / "validation-guard.jsonl"
    test_guard = tmp_path / "test-guard.jsonl"
    leak = _document("leak")
    exact_guard = _document("exact-guard")
    _write_rows(validation_guard, [leak, _document("validation-only")])
    _write_rows(test_guard, [exact_guard])

    a1 = _document("a1")
    shard_a_path = tmp_path / "train-a.jsonl"
    _write_rows(
        shard_a_path,
        [
            a1,
            dict(a1),
            {**a1, "text": "Conflicting text for one document ID."},
            leak,
            b"{not-json}\n",
            {"messages": []},
            {**_document("invalid-id"), "document_id": "not-a-sha"},
            _document("a2"),
            _document("a3"),
        ],
    )
    shard_b_path = tmp_path / "train-b.jsonl"
    _write_rows(
        shard_b_path,
        [
            _document("b1"),
            _document("same-as-a1", title=a1["title"], text=a1["text"]),
            _document(
                "same-as-guard",
                title=exact_guard["title"],
                text=exact_guard["text"],
            ),
            _document("b2"),
            _document("b3"),
        ],
    )
    leaf_a = "business-operations"
    leaf_b = "growth-gtm"
    validation_b = tmp_path / "validation-guard-b.jsonl"
    validation_b.write_bytes(validation_guard.read_bytes())
    test_b = tmp_path / "test-guard-b.jsonl"
    test_b.write_bytes(test_guard.read_bytes())
    recipe = ReferenceMixtureRecipe(
        recipe_id="business-reference-unit-v1",
        revision="1",
        usage="training",
        output_split="train",
        seed="business-reference-unit-seed-v1",
        target_records=3,
        shortfall_policy="error",
        shards=(
            ReferenceMixtureShard(
                _descriptor(
                    shard_a_path,
                    shard_id="train-a-v1",
                    leaf_id=leaf_a,
                    split="train",
                ),
                weight=1.0,
                cap=2,
            ),
            ReferenceMixtureShard(
                _descriptor(
                    shard_b_path,
                    shard_id="train-b-v1",
                    leaf_id=leaf_b,
                    split="train",
                ),
                weight=2.0,
            ),
        ),
        leakage_guard_shards=(
            _descriptor(
                validation_guard,
                shard_id="validation-a-v1",
                leaf_id=leaf_a,
                split="validation",
            ),
            _descriptor(
                test_guard,
                shard_id="test-a-v1",
                leaf_id=leaf_a,
                split="test",
            ),
            _descriptor(
                validation_b,
                shard_id="validation-b-v1",
                leaf_id=leaf_b,
                split="validation",
            ),
            _descriptor(
                test_b,
                shard_id="test-b-v1",
                leaf_id=leaf_b,
                split="test",
            ),
        ),
    )

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = build_reference_mixture(
        recipe, recipe_directory=tmp_path, output_directory=first_output
    )
    second = build_reference_mixture(
        recipe, recipe_directory=tmp_path, output_directory=second_output
    )

    assert first == second
    assert first["input_records"] == 14
    assert first["input_accounting"] == {
        "emitted": 3,
        "duplicate": 2,
        "quarantined": 4,
        "rejected": 2,
        "capped": 3,
    }
    assert first["zero_silent_drops"] is True
    assert first["leakage_guard"]["records_scanned"] == 6
    assert first["reason_accounting"]["opposite_split_document_id_match"] == 1
    assert first["reason_accounting"]["opposite_split_exact_match"] == 1
    assert first["reason_accounting"]["document_id_payload_conflict"] == 1

    output_rows = [
        json.loads(line) for line in (first_output / "documents.jsonl").read_text().splitlines()
    ]
    assert len(output_rows) == 3
    assert all(set(row) == {"document_id", "title", "text"} for row in output_rows)
    assert leak not in output_rows
    provenance = [
        json.loads(line) for line in (first_output / "provenance.jsonl").read_text().splitlines()
    ]
    assert len(provenance) == 14
    assert all("text" not in row and "title" not in row for row in provenance)
    for name in (
        "documents.jsonl",
        "provenance.jsonl",
        "resolved_recipe.json",
        "aggregate_report.json",
        "sha256sums.txt",
    ):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()

    dry_output = tmp_path / "dry"
    dry = build_reference_mixture(
        recipe,
        recipe_directory=tmp_path,
        output_directory=dry_output,
        dry_run=True,
    )
    assert not dry_output.exists()
    assert dry["content_identity"] == first["content_identity"]
    assert dry["artifacts"]["provenance.jsonl"] == first["artifacts"]["provenance.jsonl"]


def test_weight_and_cap_overrides_adjust_mix_without_mutating_recipe(tmp_path, capsys) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_rows(a, [_document(f"a-{index}") for index in range(10)])
    _write_rows(b, [_document(f"b-{index}") for index in range(10)])
    leaves = ("business-core", "growth-gtm")
    recipe = ReferenceMixtureRecipe(
        recipe_id="adjustable-reference-v1",
        revision="1",
        usage="training",
        output_split="train",
        seed="adjustable-reference-seed-v1",
        target_records=4,
        shortfall_policy="error",
        shards=(
            ReferenceMixtureShard(
                _descriptor(a, shard_id="a-v1", leaf_id=leaves[0], split="train"), 1.0
            ),
            ReferenceMixtureShard(
                _descriptor(b, shard_id="b-v1", leaf_id=leaves[1], split="train"), 3.0
            ),
        ),
        leakage_guard_shards=_guards(tmp_path, leaves, "train"),
    )
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_bytes(reference_mixture_recipe_json_bytes(recipe))
    canonical_bytes = recipe_path.read_bytes()

    baseline = build_reference_mixture(
        recipe, recipe_directory=tmp_path, output_directory=tmp_path / "baseline"
    )
    assert baseline["emitted_by_leaf"] == {"business-core": 1, "growth-gtm": 3}
    assert (
        main(
            [
                "--recipe",
                str(recipe_path),
                "--output",
                str(tmp_path / "override"),
                "--weight",
                "leaf:business-core=9",
                "--cap",
                "leaf:business-core=2",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["emitted_by_leaf"] == {"business-core": 2, "growth-gtm": 2}
    assert printed["weight_overrides"] == {"leaf:business-core": 9.0}
    assert printed["cap_overrides"] == {"leaf:business-core": 2}
    assert printed["canonical_recipe_sha256"] == recipe.fingerprint
    assert printed["resolved_recipe_sha256"] != recipe.fingerprint
    assert recipe_path.read_bytes() == canonical_bytes

    weights = parse_weight_overrides(["leaf:business-core=2", "shard:b-v1=7"])
    caps = parse_cap_overrides(["leaf:business-core=3", "shard:b-v1=none"])
    resolved = apply_overrides(recipe, weight_overrides=weights, cap_overrides=caps)
    assert {item.descriptor.shard_id: item.weight for item in resolved.shards} == {
        "a-v1": 2.0,
        "b-v1": 7.0,
    }
    assert {item.descriptor.shard_id: item.cap for item in resolved.shards} == {
        "a-v1": 3,
        "b-v1": None,
    }
    with pytest.raises(ValueError, match="duplicate cap override"):
        parse_cap_overrides(["leaf:business-core=2", "leaf:business-core=3"])


def test_recipe_is_content_addressed_split_safe_and_pin_checked(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    _write_rows(train, [_document("train")])
    guards = _guards(tmp_path, ("business-core",), "train")
    descriptor = _descriptor(train, shard_id="train-v1", leaf_id="business-core", split="train")
    recipe = ReferenceMixtureRecipe(
        recipe_id="content-addressed-v1",
        revision="1",
        usage="training",
        output_split="train",
        seed="content-addressed-seed-v1",
        target_records=1,
        shortfall_policy="error",
        shards=(ReferenceMixtureShard(descriptor, 1.0),),
        leakage_guard_shards=guards,
    )
    path = tmp_path / "recipe.json"
    path.write_bytes(reference_mixture_recipe_json_bytes(recipe))
    assert load_reference_mixture_recipe(path) == recipe

    tampered = recipe.to_dict()
    tampered["target_records"] = 2
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        load_reference_mixture_recipe(path)

    with pytest.raises(ValueError, match="both opposite splits"):
        ReferenceMixtureRecipe(
            recipe_id="missing-guard-v1",
            revision="1",
            usage="training",
            output_split="train",
            seed="seed",
            target_records=1,
            shortfall_policy="error",
            shards=(ReferenceMixtureShard(descriptor, 1.0),),
            leakage_guard_shards=guards[:1],
        )

    bad_pin = ReferenceShardDescriptor(
        **{**descriptor.to_dict(), "sha256": "0" * 64}  # type: ignore[arg-type]
    )
    bad_recipe = ReferenceMixtureRecipe(
        recipe_id="bad-pin-v1",
        revision="1",
        usage="training",
        output_split="train",
        seed="seed",
        target_records=1,
        shortfall_policy="error",
        shards=(ReferenceMixtureShard(bad_pin, 1.0),),
        leakage_guard_shards=guards,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_reference_mixture(
            bad_recipe,
            recipe_directory=tmp_path,
            output_directory=tmp_path / "bad-pin-output",
        )


def test_manifest_generates_full_three_split_recipes(tmp_path) -> None:
    shards: list[dict[str, object]] = []
    for leaf in ("business-core", "business-operations"):
        for split in ("train", "validation", "test"):
            path = tmp_path / f"{leaf}-{split}.jsonl"
            digest, records = _write_rows(path, [_document(f"{leaf}-{split}")])
            shards.append(
                {
                    "bytes": path.stat().st_size,
                    "leaf_id": leaf,
                    "leaf_revision": "1",
                    "path": path.name,
                    "records": records,
                    "sha256": digest,
                    "split": split,
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "molor.business-reference-corpus.v1",
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )
    recipes = recipes_from_reference_manifest(manifest)
    assert [recipe.output_split for recipe in recipes] == ["train", "validation", "test"]
    assert [recipe.target_records for recipe in recipes] == [2, 2, 2]
    assert all(len(recipe.shards) == 2 for recipe in recipes)
    assert all(len(recipe.leakage_guard_shards) == 4 for recipe in recipes)
