"""Deterministic <=1024-token materialization for the rack-v5 fast expert wave."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from molor.experimental.corpus_mixtures import (
    CorpusMixtureRecipe,
    build_corpus_mixture,
    load_corpus_mixture_recipe,
)
from molor.experimental.expert_taxonomy import load_expert_taxonomy
from molor.identity import canonical_json_bytes, fingerprint_json, sha256_file
from molor.training.data import ChatExample


FAST_WAVE_SCHEMA = "molor.fast-expert-wave.v1"
FAST_WAVE_PLAN_SCHEMA = "molor.fast-expert-wave-plan.v1"
_SHORT_TIERS = frozenset({"le-1024", "le1k", "short"})


def _strict_json(path: Path) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"unsupported JSON constant {value!r}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain one object: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _weight(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _is_short_tier(value: str) -> bool:
    return value in _SHORT_TIERS or value.endswith("/le-1024")


@dataclass(frozen=True, slots=True)
class FastWaveStream:
    leaf_id: str
    stream: str
    recipe_path: Path
    canonical_recipe: CorpusMixtureRecipe
    resolved_recipe: CorpusMixtureRecipe
    weight: float
    excluded_by_length: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "leaf_id": self.leaf_id,
            "stream": self.stream,
            "weight": self.weight,
            "recipe_path": str(self.recipe_path),
            "recipe_file_sha256": sha256_file(self.recipe_path),
            "canonical_recipe_sha256": self.canonical_recipe.fingerprint,
            "resolved_short_recipe_sha256": self.resolved_recipe.fingerprint,
            "eligible_shards": len(self.resolved_recipe.shards),
            "eligible_declared_records": sum(
                shard.descriptor.record_count for shard in self.resolved_recipe.shards
            ),
            "excluded_by_length": list(self.excluded_by_length),
        }


@dataclass(frozen=True, slots=True)
class FastWavePlan:
    bundle_directory: Path
    taxonomy_path: Path
    registry_path: Path
    model_id: str
    adapter_output_directory: Path
    target_records_per_leaf: int
    leaf_targets: Mapping[str, int]
    streams: tuple[FastWaveStream, ...]
    leaf_ids: tuple[str, ...]
    taxonomy_sha256: str
    registry_sha256: str
    registry_content_sha256: str
    seed: str

    def target_for(self, leaf_id: str) -> int:
        return self.leaf_targets.get(leaf_id, self.target_records_per_leaf)

    def content_dict(self) -> dict[str, object]:
        return {
            "schema_version": FAST_WAVE_PLAN_SCHEMA,
            "bundle_directory": str(self.bundle_directory),
            "taxonomy_path": str(self.taxonomy_path),
            "taxonomy_sha256": self.taxonomy_sha256,
            "registry_path": str(self.registry_path),
            "registry_sha256": self.registry_sha256,
            "registry_content_sha256": self.registry_content_sha256,
            "model_id": self.model_id,
            "adapter_output_directory": str(self.adapter_output_directory),
            "seed": self.seed,
            "target_records_per_leaf": self.target_records_per_leaf,
            "leaf_targets": dict(sorted(self.leaf_targets.items())),
            "leaf_ids": list(self.leaf_ids),
            "streams": [stream.to_dict() for stream in self.streams],
        }

    def to_dict(self) -> dict[str, object]:
        content = self.content_dict()
        return {**content, "content_sha256": fingerprint_json(content)}


def build_fast_wave_plan(
    *,
    bundle_directory: str | Path,
    taxonomy_path: str | Path,
    model_id: str | Path,
    adapter_output_directory: str | Path,
    target_records_per_leaf: int = 1024,
    leaf_targets: Mapping[str, int] | None = None,
    stream_weights: Mapping[str, float] | None = None,
    leaf_stream_weights: Mapping[tuple[str, str], float] | None = None,
    seed: str = "molor-rack-v5-fast-wave/seed-1",
) -> FastWavePlan:
    """Resolve every rack leaf/stream and explicitly remove only >1024 training shards."""

    if not isinstance(target_records_per_leaf, int) or isinstance(
        target_records_per_leaf, bool
    ) or target_records_per_leaf < 1:
        raise ValueError("target_records_per_leaf must be a positive integer")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be non-empty")
    bundle = Path(bundle_directory).resolve()
    taxonomy_source = Path(taxonomy_path).resolve()
    registry_path = bundle / "shard-registry.json"
    recipes_root = bundle / "recipes" / "fast"
    taxonomy = load_expert_taxonomy(taxonomy_source)
    leaf_ids = tuple(sorted(taxonomy.leaf_ids))
    target_overrides = dict(leaf_targets or {})
    unknown_targets = set(target_overrides).difference(leaf_ids)
    if unknown_targets:
        raise KeyError(f"unknown leaf target(s): {sorted(unknown_targets)!r}")
    for leaf_id, value in target_overrides.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"leaf target for {leaf_id!r} must be a positive integer")

    registry = _strict_json(registry_path)
    claimed_registry = registry.get("content_sha256")
    if not isinstance(claimed_registry, str):
        raise ValueError("registry is missing content_sha256")
    registry_content = dict(registry)
    del registry_content["content_sha256"]
    if fingerprint_json(registry_content) != claimed_registry:
        raise ValueError("registry content_sha256 does not match its semantic payload")
    raw_shards = registry.get("chat_shards")
    if not isinstance(raw_shards, list):
        raise ValueError("registry chat_shards must be an array")
    registered: dict[str, Mapping[str, object]] = {}
    for row in raw_shards:
        if not isinstance(row, Mapping) or not isinstance(row.get("shard_id"), str):
            raise ValueError("registry contains an invalid chat shard")
        shard_id = str(row["shard_id"])
        if shard_id in registered:
            raise ValueError(f"registry repeats shard_id {shard_id!r}")
        registered[shard_id] = row

    global_weights = {key: _weight(value, f"stream_weights[{key!r}]") for key, value in (stream_weights or {}).items()}
    local_weights = {
        key: _weight(value, f"leaf_stream_weights[{key!r}]")
        for key, value in (leaf_stream_weights or {}).items()
    }
    streams: list[FastWaveStream] = []
    for leaf_id in leaf_ids:
        leaf_directory = recipes_root / leaf_id
        if not leaf_directory.is_dir():
            raise ValueError(f"missing fast recipe directory for leaf {leaf_id!r}")
        recipe_paths = sorted(leaf_directory.glob("*/train.json"))
        if not recipe_paths:
            raise ValueError(f"leaf {leaf_id!r} has no fast training recipe")
        for recipe_path in recipe_paths:
            stream_name = recipe_path.parent.name
            canonical = load_corpus_mixture_recipe(recipe_path)
            eligible = []
            excluded = []
            for shard in canonical.shards:
                descriptor = shard.descriptor
                registry_row = registered.get(descriptor.shard_id)
                if registry_row is None:
                    raise ValueError(f"recipe shard is absent from registry: {descriptor.shard_id}")
                if any(registry_row.get(key) != value for key, value in descriptor.to_dict().items()):
                    raise ValueError(f"recipe/registry descriptor mismatch: {descriptor.shard_id}")
                tier = registry_row.get("length_tier")
                if not isinstance(tier, str) or not tier:
                    raise ValueError(f"registry shard lacks a length tier: {descriptor.shard_id}")
                if _is_short_tier(tier):
                    eligible.append(shard)
                else:
                    excluded.append(
                        {
                            "shard_id": descriptor.shard_id,
                            "length_tier": tier,
                            "record_count": descriptor.record_count,
                            "status": "excluded_length_tier_gt_1024",
                        }
                    )
            if not eligible:
                raise ValueError(f"{leaf_id}/{stream_name} has no <=1024 shard")
            resolved = CorpusMixtureRecipe(
                recipe_id=f"{canonical.recipe_id}-le1024-wave",
                revision=canonical.revision,
                usage=canonical.usage,
                output_split=canonical.output_split,
                template_mode=canonical.template_mode,
                seed=canonical.seed,
                target_records=min(canonical.target_records, target_overrides.get(leaf_id, target_records_per_leaf)),
                shortfall_policy="emit_available",
                shards=tuple(eligible),
                leakage_guard_shards=canonical.leakage_guard_shards,
            )
            weight = local_weights.get(
                (leaf_id, stream_name), global_weights.get(stream_name, 1.0)
            )
            streams.append(
                FastWaveStream(
                    leaf_id=leaf_id,
                    stream=stream_name,
                    recipe_path=recipe_path.resolve(),
                    canonical_recipe=canonical,
                    resolved_recipe=resolved,
                    weight=weight,
                    excluded_by_length=tuple(excluded),
                )
            )
        if not any(stream.weight > 0 for stream in streams if stream.leaf_id == leaf_id):
            raise ValueError(f"all streams are disabled for leaf {leaf_id!r}")

    discovered = {stream.leaf_id for stream in streams}
    if discovered != set(leaf_ids):
        raise AssertionError("fast-wave stream discovery did not cover the taxonomy")
    available_pairs = {(stream.leaf_id, stream.stream) for stream in streams}
    unknown_local_weights = set(local_weights).difference(available_pairs)
    if unknown_local_weights:
        raise KeyError(f"unknown leaf/stream weight(s): {sorted(unknown_local_weights)!r}")
    available_streams = {stream.stream for stream in streams}
    unknown_global_weights = set(global_weights).difference(available_streams)
    if unknown_global_weights:
        raise KeyError(f"unknown stream weight(s): {sorted(unknown_global_weights)!r}")
    return FastWavePlan(
        bundle_directory=bundle,
        taxonomy_path=taxonomy_source,
        registry_path=registry_path,
        model_id=str(Path(model_id).resolve()),
        adapter_output_directory=Path(adapter_output_directory).resolve(),
        target_records_per_leaf=target_records_per_leaf,
        leaf_targets=target_overrides,
        streams=tuple(sorted(streams, key=lambda value: (value.leaf_id, value.stream))),
        leaf_ids=leaf_ids,
        taxonomy_sha256=taxonomy.fingerprint,
        registry_sha256=sha256_file(registry_path),
        registry_content_sha256=claimed_registry,
        seed=seed,
    )


@dataclass(slots=True)
class _Candidate:
    stream: str
    template_mode: str
    row_index: int
    content_sha256: str
    line: bytes
    priority: str
    status: str = "pending"


def _load_candidates(
    path: Path, *, stream: str, template_mode: str, leaf_id: str, seed: str
) -> list[_Candidate]:
    result: list[_Candidate] = []
    with path.open("r", encoding="utf-8") as source:
        for row_index, line in enumerate(source, start=1):
            value = json.loads(line)
            example = ChatExample.from_dict(value)
            normalized = {"messages": [message.to_dict() for message in example.messages]}
            encoded = canonical_json_bytes(normalized)
            content_sha = hashlib.sha256(encoded).hexdigest()
            priority = hashlib.sha256(
                f"{seed}\0{leaf_id}\0{stream}\0{content_sha}\0{row_index}".encode()
            ).hexdigest()
            result.append(
                _Candidate(
                    stream, template_mode, row_index, content_sha, encoded + b"\n", priority
                )
            )
    return result


def _select_candidates(
    candidates: list[_Candidate], *, weights: Mapping[str, float], target: int
) -> tuple[list[_Candidate], dict[str, dict[str, int]]]:
    groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.content_sha256].append(candidate)
    unique_by_stream: dict[str, list[_Candidate]] = defaultdict(list)
    for digest, duplicate_group in sorted(groups.items()):
        first_line = duplicate_group[0].line
        if any(item.line != first_line for item in duplicate_group[1:]):
            raise RuntimeError(f"SHA-256 collision while merging candidate {digest}")
        enabled = [item for item in duplicate_group if weights[item.stream] > 0]
        owner_pool = enabled or duplicate_group
        owner = min(owner_pool, key=lambda item: (item.priority, item.stream, item.row_index))
        unique_by_stream[owner.stream].append(owner)
        for item in duplicate_group:
            if item is not owner:
                item.status = "duplicate"
    for stream_candidates in unique_by_stream.values():
        stream_candidates.sort(key=lambda item: (item.priority, item.content_sha256))

    selected: list[_Candidate] = []
    positions = Counter()
    selected_by_stream = Counter()
    while len(selected) < target:
        available = [
            stream
            for stream, rows in unique_by_stream.items()
            if weights[stream] > 0 and positions[stream] < len(rows)
        ]
        if not available:
            break
        stream = min(
            available,
            key=lambda name: (
                Fraction(selected_by_stream[name] + 1, 1) / Fraction(str(weights[name])),
                name,
            ),
        )
        candidate = unique_by_stream[stream][positions[stream]]
        positions[stream] += 1
        selected_by_stream[stream] += 1
        candidate.status = "emitted"
        selected.append(candidate)
    for rows in unique_by_stream.values():
        for item in rows:
            if item.status == "pending":
                item.status = "capped"

    accounting: dict[str, dict[str, int]] = {}
    for stream in sorted(weights):
        stream_rows = [item for item in candidates if item.stream == stream]
        counts = Counter(item.status for item in stream_rows)
        accounting[stream] = {
            "candidates": len(stream_rows),
            "emitted": counts["emitted"],
            "duplicate": counts["duplicate"],
            "capped": counts["capped"],
        }
    return selected, accounting


def _training_recipe(plan: FastWavePlan, output: Path) -> dict[str, object]:
    return {
        "model_id": plan.model_id,
        "output_dir": str(plan.adapter_output_directory),
        "dtype": "bfloat16",
        "device": "cuda",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "max_length": 1024,
        "epochs": 1,
        "learning_rate": 0.00005,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "micro_batch_size": 3,
        "gradient_accumulation_steps": 3,
        "gradient_checkpointing": True,
        "autocast_adapter_compute": True,
        "batching_strategy": "pretokenized_group_sorted",
        "fused_adamw": False,
        "checkpoint_every_steps": 100,
        "max_grad_norm": 1.0,
        "seed": 73,
        "chat_template_kwargs": {"enable_thinking": True},
        "experts": [
            {
                "name": leaf_id,
                "dataset": str(Path("experts") / leaf_id / "train.jsonl"),
            }
            for leaf_id in plan.leaf_ids
        ],
    }


def _run_materialization(plan: FastWavePlan, stage: Path, intended_output: Path) -> dict[str, object]:
    streams_by_leaf: dict[str, list[FastWaveStream]] = defaultdict(list)
    for stream in plan.streams:
        streams_by_leaf[stream.leaf_id].append(stream)
    artifacts: list[dict[str, object]] = []
    leaf_reports: list[dict[str, object]] = []
    for leaf_id in plan.leaf_ids:
        leaf_directory = stage / "experts" / leaf_id
        leaf_directory.mkdir(parents=True)
        weights = {stream.stream: stream.weight for stream in streams_by_leaf[leaf_id]}
        candidates: list[_Candidate] = []
        source_reports: dict[str, object] = {}
        declared_records = 0
        excluded_records = 0
        eligible_records = 0
        for stream in streams_by_leaf[leaf_id]:
            declared_records += sum(
                shard.descriptor.record_count for shard in stream.canonical_recipe.shards
            )
            excluded_records += sum(
                int(row["record_count"]) for row in stream.excluded_by_length
            )
            eligible_records += sum(
                shard.descriptor.record_count for shard in stream.resolved_recipe.shards
            )
            stream_output = stage / ".streams" / leaf_id / stream.stream
            report = build_corpus_mixture(
                stream.resolved_recipe,
                recipe_directory=stream.recipe_path.parent,
                output_directory=stream_output,
                canonical_recipe_fingerprint=stream.canonical_recipe.fingerprint,
            )
            source_reports[stream.stream] = {
                "weight": stream.weight,
                "excluded_by_length": list(stream.excluded_by_length),
                "materializer": report,
            }
            candidates.extend(
                _load_candidates(
                    stream_output / "messages.jsonl",
                    stream=stream.stream,
                    template_mode=stream.resolved_recipe.template_mode,
                    leaf_id=leaf_id,
                    seed=plan.seed,
                )
            )
        if declared_records != eligible_records + excluded_records:
            raise AssertionError("source length-tier accounting does not reconcile")
        selected, candidate_accounting = _select_candidates(
            candidates, weights=weights, target=plan.target_for(leaf_id)
        )
        dataset = leaf_directory / "train.jsonl"
        with dataset.open("wb") as destination:
            for candidate in selected:
                destination.write(candidate.line)
        provenance = leaf_directory / "provenance.jsonl"
        with provenance.open("wb") as destination:
            for candidate in sorted(
                candidates, key=lambda item: (item.stream, item.row_index)
            ):
                destination.write(
                    canonical_json_bytes(
                        {
                            "content_sha256": candidate.content_sha256,
                            "source_row": candidate.row_index,
                            "stream": candidate.stream,
                            "template_mode": candidate.template_mode,
                            "status": candidate.status,
                        }
                    )
                    + b"\n"
                )
        final_accounted = sum(
            counts[status]
            for counts in candidate_accounting.values()
            for status in ("emitted", "duplicate", "capped")
        )
        source_accounted = excluded_records + sum(
            sum(int(value) for value in report["materializer"]["input_accounting"].values())
            for report in source_reports.values()  # type: ignore[union-attr]
        )
        leaf_report = {
            "leaf_id": leaf_id,
            "target_records": plan.target_for(leaf_id),
            "emitted_records": len(selected),
            "shortfall_records": plan.target_for(leaf_id) - len(selected),
            "declared_source_records": declared_records,
            "eligible_source_records": eligible_records,
            "excluded_length_tier_records": excluded_records,
            "source_accounted_records": source_accounted,
            "candidate_records": len(candidates),
            "candidate_accounting": candidate_accounting,
            "final_accounted_records": final_accounted,
            "source_reports": source_reports,
            "zero_silent_source_drops": source_accounted == declared_records,
            "zero_silent_candidate_drops": final_accounted == len(candidates),
            "dataset_sha256": sha256_file(dataset),
            "provenance_sha256": sha256_file(provenance),
        }
        if not leaf_report["zero_silent_source_drops"] or not leaf_report[
            "zero_silent_candidate_drops"
        ]:
            raise AssertionError("fast-wave accounting failed to reconcile")
        report_path = leaf_directory / "accounting.json"
        report_path.write_bytes(_json_bytes(leaf_report))
        for path, records in ((dataset, len(selected)), (provenance, len(candidates))):
            artifacts.append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "records": records,
                }
            )
        artifacts.append(
            {
                "path": report_path.relative_to(stage).as_posix(),
                "sha256": sha256_file(report_path),
                "bytes": report_path.stat().st_size,
            }
        )
        leaf_reports.append(leaf_report)
    shutil.rmtree(stage / ".streams")

    training_recipe = stage / "training-recipe.json"
    training_recipe.write_bytes(_json_bytes(_training_recipe(plan, intended_output)))
    artifacts.append(
        {
            "path": "training-recipe.json",
            "sha256": sha256_file(training_recipe),
            "bytes": training_recipe.stat().st_size,
        }
    )
    content = {
        "schema_version": FAST_WAVE_SCHEMA,
        "plan": plan.to_dict(),
        "output_directory": str(intended_output.resolve()),
        "leaf_count": len(plan.leaf_ids),
        "stream_count": len(plan.streams),
        "emitted_records": sum(int(row["emitted_records"]) for row in leaf_reports),
        "zero_silent_drops": all(
            row["zero_silent_source_drops"] and row["zero_silent_candidate_drops"]
            for row in leaf_reports
        ),
        "known_limitations": [
            (
                "Tool-message rows preserve tool_calls and tool_call_id in messages, but the "
                "current messages-only training contract has no top-level tools signature."
            )
        ],
        "leaves": leaf_reports,
        "artifacts": sorted(artifacts, key=lambda row: str(row["path"])),
    }
    manifest = {**content, "content_sha256": fingerprint_json(content)}
    manifest_path = stage / "wave-manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    sums = sorted(
        [*artifacts, {"path": "wave-manifest.json", "sha256": sha256_file(manifest_path)}],
        key=lambda row: str(row["path"]),
    )
    (stage / "sha256sums.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in sums), encoding="utf-8"
    )
    return manifest


def materialize_fast_wave(
    plan: FastWavePlan, *, output_directory: str | Path, dry_run: bool = False
) -> dict[str, object]:
    """Run an exact preflight or atomically publish all 25 leaf datasets and one recipe."""

    if not isinstance(plan, FastWavePlan):
        raise TypeError("plan must be a FastWavePlan")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="molor-fast-wave-preflight-") as temporary:
            manifest = _run_materialization(plan, Path(temporary), output)
        return {**manifest, "dry_run": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        manifest = _run_materialization(plan, stage, output)
        os.replace(stage, output)
        stage = None
        return {**manifest, "dry_run": False}
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


__all__ = [
    "FAST_WAVE_PLAN_SCHEMA",
    "FAST_WAVE_SCHEMA",
    "FastWavePlan",
    "FastWaveStream",
    "build_fast_wave_plan",
    "materialize_fast_wave",
]
