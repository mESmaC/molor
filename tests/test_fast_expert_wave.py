import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from molor.experimental.corpus_mixtures import (
    CorpusMixtureRecipe,
    CorpusMixtureShard,
    CorpusShardDescriptor,
    corpus_mixture_recipe_json_bytes,
)
from molor.training.fast_expert_wave import (
    FastWavePlan,
    FastWaveStream,
    build_fast_wave_plan,
    materialize_fast_wave,
)
from molor.training.recipe import load_training_recipe


def _row(prompt: str, answer: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stream(
    tmp_path: Path,
    *,
    name: str,
    mode: str,
    rows: list[dict[str, object]],
) -> FastWaveStream:
    source = tmp_path / f"{name}.jsonl"
    _write_rows(source, rows)
    descriptor = CorpusShardDescriptor(
        shard_id=f"source/{name}/train",
        shard_revision="revision-1",
        leaf_id="leaf-a",
        leaf_revision="taxonomy-revision-1",
        split="train",
        template_mode=mode,
        path=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        record_count=len(rows),
    )
    recipe = CorpusMixtureRecipe(
        recipe_id=f"{name}-recipe",
        revision="1",
        usage="training",
        output_split="train",
        template_mode=mode,
        seed=f"{name}-seed",
        target_records=len(rows),
        shortfall_policy="error",
        shards=(CorpusMixtureShard(descriptor, weight=1.0),),
    )
    recipe_path = tmp_path / f"{name}.recipe.json"
    recipe_path.write_bytes(corpus_mixture_recipe_json_bytes(recipe))
    return FastWaveStream(
        leaf_id="leaf-a",
        stream=name,
        recipe_path=recipe_path,
        canonical_recipe=recipe,
        resolved_recipe=recipe,
        weight=1.0,
        excluded_by_length=(),
    )


def test_materialization_is_deterministic_and_reconciles_every_candidate(tmp_path: Path) -> None:
    duplicate = _row("same", "same answer")
    direct = _stream(
        tmp_path,
        name="direct",
        mode="direct-v1",
        rows=[duplicate, _row("direct 1", "a"), _row("direct 2", "b")],
    )
    trace = _stream(
        tmp_path,
        name="trace",
        mode="trace-v1",
        rows=[duplicate, _row("trace 1", "c"), _row("trace 2", "d")],
    )
    plan = FastWavePlan(
        bundle_directory=tmp_path,
        taxonomy_path=tmp_path / "taxonomy.json",
        registry_path=tmp_path / "registry.json",
        model_id=str(tmp_path / "model"),
        adapter_output_directory=tmp_path / "adapters",
        target_records_per_leaf=3,
        leaf_targets={},
        streams=(direct, trace),
        leaf_ids=("leaf-a",),
        taxonomy_sha256="1" * 64,
        registry_sha256="2" * 64,
        registry_content_sha256="3" * 64,
        seed="wave-seed",
    )
    output = tmp_path / "wave"
    dry = materialize_fast_wave(plan, output_directory=output, dry_run=True)
    assert dry["dry_run"] is True
    assert not output.exists()

    actual = materialize_fast_wave(plan, output_directory=output)
    assert actual["content_sha256"] == dry["content_sha256"]
    assert actual["emitted_records"] == 3
    assert actual["zero_silent_drops"] is True
    leaf = actual["leaves"][0]
    assert leaf["declared_source_records"] == 6
    assert leaf["source_accounted_records"] == 6
    assert leaf["candidate_records"] == 6
    assert leaf["final_accounted_records"] == 6
    totals = {
        status: sum(row[status] for row in leaf["candidate_accounting"].values())
        for status in ("emitted", "duplicate", "capped")
    }
    assert totals == {"emitted": 3, "duplicate": 1, "capped": 2}
    provenance = [
        json.loads(line)
        for line in (output / "experts" / "leaf-a" / "provenance.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["template_mode"] for row in provenance} == {"direct-v1", "trace-v1"}
    recipe = load_training_recipe(output / "training-recipe.json")
    assert recipe.max_length == 1024
    assert recipe.rank == 8
    assert recipe.alpha == 16
    assert recipe.chat_template_kwargs == {"enable_thinking": True}
    assert [expert.name for expert in recipe.experts] == ["leaf-a"]
    with pytest.raises(FileExistsError):
        materialize_fast_wave(plan, output_directory=output, dry_run=True)


def test_real_bundle_plan_covers_25_leaves_and_classifies_long_shards() -> None:
    repository = Path(__file__).resolve().parents[1]
    rack = repository / "experiments" / "expert-rack-v5"
    plan = build_fast_wave_plan(
        bundle_directory=rack / "chat-mixtures-v1",
        taxonomy_path=rack / "expert-taxonomy-v1.json",
        model_id=repository / "models" / "Qwen3-0.6B",
        adapter_output_directory=repository / "checkpoints" / "fast-wave-test",
    )
    assert len(plan.leaf_ids) == 25
    assert {stream.leaf_id for stream in plan.streams} == set(plan.leaf_ids)
    assert len(plan.streams) == 45
    assert all(stream.resolved_recipe.shards for stream in plan.streams)
    assert all(stream.resolved_recipe.target_records <= 1024 for stream in plan.streams)
    assert sum(
        int(row["record_count"])
        for stream in plan.streams
        for row in stream.excluded_by_length
    ) > 0
    assert all(
        row["status"] == "excluded_length_tier_gt_1024"
        for stream in plan.streams
        for row in stream.excluded_by_length
    )
    with pytest.raises(KeyError, match="unknown stream"):
        build_fast_wave_plan(
            bundle_directory=rack / "chat-mixtures-v1",
            taxonomy_path=rack / "expert-taxonomy-v1.json",
            model_id=repository / "models" / "Qwen3-0.6B",
            adapter_output_directory=repository / "checkpoints" / "fast-wave-test",
            stream_weights={"not-a-stream": 1.0},
        )


def test_cli_rejects_fractional_leaf_target_without_truncating(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "experiments" / "expert-rack-v5" / "build_fast_expert_wave.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(tmp_path / "unused"),
            "--plan-only",
            "--leaf-target",
            "business-core=1.5",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "exact positive integer" in result.stderr
