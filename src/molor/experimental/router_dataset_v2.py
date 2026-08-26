"""Deterministic, contamination-resistant router dataset construction for rack v5.

Router dataset v1 imported ``base/null`` labels from a legacy classifier corpus.  Those
labels were not true negatives: many prompts were about software, finance, mathematics,
and other rack specialties.  This module deliberately has no legacy-base input.  It
combines the already admitted expert chat shards with a small, independently authored
curriculum containing:

* high-confidence general-conversation negatives; and
* short, natural route anchors for every taxonomy leaf.

The curriculum is not an evaluation source.  Benchmark and route-contract paths are
rejected before they are read, curriculum prompts may not duplicate admitted chat
prompts, and normalized prompts have exactly one split owner.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sqlite3
from string import Formatter
import tempfile

from molor.experimental.expert_mixture_registry import (
    CollectedExpertData,
    ROUTER_MAX_PROMPT_CHARACTERS,
    ROUTER_SOURCE_PRIORITIES,
    _normalize_prompt,
    _resolve_router_owners,
    _scan_chat_for_router,
    _selection_rank,
)
from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256


ROUTER_CURRICULUM_V2_SCHEMA = "molor.router-curriculum.v2"
ROUTER_EXAMPLES_V2_MANIFEST_SCHEMA = "molor.hierarchical-router-examples.v2"
ROUTER_V2_RECORD_CAPS = {"train": 1_024, "validation": 256, "test": 256}
CURRICULUM_PRIORITY = -100
MAX_CURRICULUM_PROMPT_CHARACTERS = 512
MAX_GROUP_EXPANSION = 4_096
ANCHOR_SEEDS_PER_LEAF = 48
ANCHOR_SURFACE_VARIANTS = 4
ANCHOR_SEEDS_BY_SPLIT = {"train": 32, "validation": 8, "test": 8}
ANCHOR_ROWS_BY_SPLIT = {"train": 128, "validation": 32, "test": 32}
SHORT_PROMPT_MAX_CHARACTERS = 256
MINIMUM_SHORT_ROWS = {"train": 256, "validation": 64, "test": 64}
MINIMUM_LONG_SOURCE_ROWS = {"train": 64, "validation": 16, "test": 16}
SOURCE_STYLE_CAP_NUMERATOR = 3
SOURCE_STYLE_CAP_DENOMINATOR = 5
MINIMUM_BASE_ROWS = {"train": 512, "validation": 128, "test": 128}
MINIMUM_BASE_TOTAL = 1_200
MAXIMUM_BASE_TOTAL = 1_450
_SPLIT_ORDER = ("train", "validation", "test")
_FORBIDDEN_INPUT_MARKERS = (
    "benchmark",
    "molor-core",
    "route-contract",
    "route_contract",
)


@dataclass(frozen=True, slots=True)
class RouterCurriculumV2:
    """Validated, content-addressed v2 curriculum."""

    path: Path
    file_sha256: str
    content_sha256: str
    dataset_id: str
    revision: str
    split_weights: Mapping[str, int]
    base_groups_by_split: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]]
    anchor_seeds_by_leaf: Mapping[str, tuple[str, ...]]
    anchor_surface_variants: tuple[str, ...]
    base_deny_terms: tuple[str, ...]
    authoring_contract: Mapping[str, object]
    group_accounting: Mapping[str, Mapping[str, object]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON numeric constant {value!r}")


def _strict_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"curriculum must be a regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict UTF-8 JSON curriculum: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("router curriculum must contain one JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{field} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{field} must be a non-empty array")
    result = tuple(_string(item, f"{field}[]").strip() for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate strings")
    return result


def _ensure_nonbenchmark_input(path: Path) -> None:
    lowered = path.resolve().as_posix().casefold()
    marker = next((item for item in _FORBIDDEN_INPUT_MARKERS if item in lowered), None)
    if marker is not None:
        raise ValueError(
            f"router v2 refuses benchmark-linked curriculum inputs (matched {marker!r}): {path}"
        )


def _validate_prompt(prompt: str, field: str, *, short_anchor: bool) -> str:
    value = prompt.strip()
    if not value:
        raise ValueError(f"{field} must not be blank")
    maximum = 240 if short_anchor else MAX_CURRICULUM_PROMPT_CHARACTERS
    if len(value) > maximum:
        raise ValueError(f"{field} exceeds the {maximum}-character curriculum limit")
    if "\x00" in value:
        raise ValueError(f"{field} contains a NUL character")
    return value


def _expand_base_group(value: object, index: int) -> tuple[str, str, tuple[str, ...]]:
    group = _mapping(value, f"base_negative_groups[{index}]")
    _exact_keys(
        group,
        {"group_id", "split", "template", "slots"},
        f"base_negative_groups[{index}]",
    )
    group_id = _string(group["group_id"], f"base_negative_groups[{index}].group_id")
    split = _string(group["split"], f"base_negative_groups[{index}].split")
    if split not in _SPLIT_ORDER:
        raise ValueError(f"base-negative group {group_id!r} has unsupported split {split!r}")
    template = _string(group["template"], f"base_negative_groups[{index}].template")
    slots = _mapping(group["slots"], f"base_negative_groups[{index}].slots")
    if not slots:
        raise ValueError(f"base-negative group {group_id!r} needs at least one slot")

    parsed = tuple(
        field_name
        for _literal, field_name, format_spec, conversion in Formatter().parse(template)
        if field_name is not None and not format_spec and conversion is None
    )
    if set(parsed) != set(slots) or len(parsed) != len(set(parsed)):
        raise ValueError(
            f"base-negative group {group_id!r} placeholders must exactly match unique slot keys"
        )

    slot_names = tuple(sorted(slots))
    slot_values = tuple(
        _string_list(slots[name], f"base_negative_groups[{index}].slots.{name}")
        for name in slot_names
    )
    expansion = 1
    for values in slot_values:
        expansion *= len(values)
    if expansion > MAX_GROUP_EXPANSION:
        raise ValueError(
            f"base-negative group {group_id!r} expands to {expansion}, above "
            f"the {MAX_GROUP_EXPANSION} record limit"
        )
    prompts = tuple(
        _validate_prompt(
            template.format(**dict(zip(slot_names, combination, strict=True))),
            f"base-negative group {group_id!r}",
            short_anchor=False,
        )
        for combination in itertools.product(*slot_values)
    )
    return group_id, split, prompts


def _anchor_variants(value: object) -> tuple[str, ...]:
    variants = _string_list(value, "anchor_surface_variants")
    if len(variants) != ANCHOR_SURFACE_VARIANTS:
        raise ValueError(
            f"anchor_surface_variants must contain exactly {ANCHOR_SURFACE_VARIANTS} templates"
        )
    for index, template in enumerate(variants):
        parsed = tuple(Formatter().parse(template))
        fields = tuple(field for _literal, field, _spec, _conversion in parsed if field)
        if fields != ("prompt",) or any(spec or conversion for _, _, spec, conversion in parsed):
            raise ValueError(
                f"anchor_surface_variants[{index}] must contain one plain {{prompt}} field"
            )
        _validate_prompt(
            template.format(prompt="Handle this ordinary request."),
            f"anchor_surface_variants[{index}]",
            short_anchor=True,
        )
    return variants


def load_router_curriculum_v2(
    path: str | Path,
    *,
    expected_leaf_ids: Sequence[str],
) -> RouterCurriculumV2:
    """Load and strictly validate an independently authored v2 curriculum."""

    source = Path(path).resolve()
    _ensure_nonbenchmark_input(source)
    document = _strict_json(source)
    expected_keys = {
        "schema_version",
        "dataset_id",
        "revision",
        "authoring_contract",
        "split_weights",
        "base_negative_deny_terms",
        "base_negative_groups",
        "anchor_surface_variants",
        "leaf_anchors",
        "content_sha256",
    }
    _exact_keys(document, expected_keys, "router curriculum")
    if document["schema_version"] != ROUTER_CURRICULUM_V2_SCHEMA:
        raise ValueError("unsupported router curriculum schema")

    claimed = validate_sha256(
        _string(document["content_sha256"], "content_sha256"), "content_sha256"
    )
    unhashed = dict(document)
    del unhashed["content_sha256"]
    observed = fingerprint_json(unhashed)
    if claimed != observed:
        raise ValueError(
            f"router curriculum content hash mismatch: expected {claimed}, observed {observed}"
        )

    authoring = _mapping(document["authoring_contract"], "authoring_contract")
    _exact_keys(
        authoring,
        {
            "source_basis",
            "benchmark_material_used",
            "legacy_base_source_used",
            "intended_use",
        },
        "authoring_contract",
    )
    if authoring["benchmark_material_used"] is not False:
        raise ValueError("router v2 curriculum must declare benchmark_material_used=false")
    if authoring["legacy_base_source_used"] is not False:
        raise ValueError("router v2 curriculum must declare legacy_base_source_used=false")
    if authoring["intended_use"] != "router-training-only":
        raise ValueError("router v2 curriculum intended_use must be router-training-only")
    _string(authoring["source_basis"], "authoring_contract.source_basis")

    raw_weights = _mapping(document["split_weights"], "split_weights")
    _exact_keys(raw_weights, set(_SPLIT_ORDER), "split_weights")
    split_weights: dict[str, int] = {}
    for split in _SPLIT_ORDER:
        weight = raw_weights[split]
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
            raise ValueError(f"split_weights.{split} must be a positive integer")
        split_weights[split] = weight
    if split_weights != {"train": 4, "validation": 1, "test": 1}:
        raise ValueError("router v2 requires split_weights train=4, validation=1, test=1")

    deny_terms = tuple(
        _normalize_prompt(term)
        for term in _string_list(document["base_negative_deny_terms"], "deny terms")
    )
    if any(not term for term in deny_terms):
        raise ValueError("normalized base-negative deny terms must not be empty")

    raw_groups = document["base_negative_groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise TypeError("base_negative_groups must be a non-empty array")
    group_accounting: dict[str, dict[str, object]] = {}
    base_groups_by_split: dict[str, list[tuple[str, tuple[str, ...]]]] = {
        split: [] for split in _SPLIT_ORDER
    }
    base_prompts: list[str] = []
    for index, raw_group in enumerate(raw_groups):
        group_id, split, prompts = _expand_base_group(raw_group, index)
        if group_id in group_accounting:
            raise ValueError(f"duplicate base-negative group ID {group_id!r}")
        group_accounting[group_id] = {"split": split, "records": len(prompts)}
        base_groups_by_split[split].append((group_id, prompts))
        base_prompts.extend(prompts)

    base_balance = {
        split: sum(len(prompts) for _group_id, prompts in groups)
        for split, groups in base_groups_by_split.items()
    }
    base_total = sum(base_balance.values())
    if not MINIMUM_BASE_TOTAL <= base_total <= MAXIMUM_BASE_TOTAL:
        raise ValueError(
            f"base-negative curriculum must contain {MINIMUM_BASE_TOTAL}.."
            f"{MAXIMUM_BASE_TOTAL} rows, observed {base_total}"
        )
    for split in _SPLIT_ORDER:
        if base_balance[split] < MINIMUM_BASE_ROWS[split]:
            raise ValueError(
                f"base-negative {split} support {base_balance[split]} is below "
                f"the {MINIMUM_BASE_ROWS[split]}-row minimum"
            )
        if base_balance[split] > ROUTER_V2_RECORD_CAPS[split]:
            raise ValueError(
                f"base-negative {split} support {base_balance[split]} exceeds "
                f"the {ROUTER_V2_RECORD_CAPS[split]}-row route cap"
            )

    expected_leaves = tuple(sorted(expected_leaf_ids))
    if not expected_leaves or len(expected_leaves) != len(set(expected_leaves)):
        raise ValueError("expected_leaf_ids must be non-empty and unique")
    raw_anchors = _mapping(document["leaf_anchors"], "leaf_anchors")
    if set(raw_anchors) != set(expected_leaves):
        raise ValueError(
            "leaf_anchors must exactly cover the taxonomy: "
            f"missing={sorted(set(expected_leaves) - set(raw_anchors))}, "
            f"extra={sorted(set(raw_anchors) - set(expected_leaves))}"
        )
    surface_variants = _anchor_variants(document["anchor_surface_variants"])
    anchor_seeds_by_leaf: dict[str, tuple[str, ...]] = {}
    for leaf in expected_leaves:
        seeds = tuple(
            _validate_prompt(item, f"leaf_anchors.{leaf}[]", short_anchor=True)
            for item in _string_list(raw_anchors[leaf], f"leaf_anchors.{leaf}")
        )
        if len(seeds) != ANCHOR_SEEDS_PER_LEAF:
            raise ValueError(
                f"leaf {leaf!r} must contain exactly {ANCHOR_SEEDS_PER_LEAF} "
                f"semantic seeds, observed {len(seeds)}"
            )
        anchor_seeds_by_leaf[leaf] = seeds

    all_prompts = [*base_prompts]
    for leaf, seeds in anchor_seeds_by_leaf.items():
        for seed in seeds:
            all_prompts.extend(
                _validate_prompt(
                    template.format(prompt=seed),
                    f"leaf_anchors.{leaf} expanded prompt",
                    short_anchor=True,
                )
                for template in surface_variants
            )
    normalized = [_normalize_prompt(prompt) for prompt in all_prompts]
    if len(normalized) != len(set(normalized)):
        duplicates = sorted(value for value, count in Counter(normalized).items() if count > 1)
        raise ValueError(f"curriculum contains normalized duplicate prompts: {duplicates[:5]}")

    for prompt in base_prompts:
        normalized_prompt = _normalize_prompt(prompt)
        matched = next((term for term in deny_terms if term in normalized_prompt), None)
        if matched is not None:
            raise ValueError(
                f"base-negative prompt contains denied specialist cue {matched!r}: {prompt!r}"
            )

    return RouterCurriculumV2(
        path=source,
        file_sha256=_sha256_file(source),
        content_sha256=claimed,
        dataset_id=_string(document["dataset_id"], "dataset_id"),
        revision=_string(document["revision"], "revision"),
        split_weights=split_weights,
        base_groups_by_split={
            split: tuple(groups) for split, groups in base_groups_by_split.items()
        },
        anchor_seeds_by_leaf=anchor_seeds_by_leaf,
        anchor_surface_variants=surface_variants,
        base_deny_terms=deny_terms,
        authoring_contract=dict(authoring),
        group_accounting=dict(sorted(group_accounting.items())),
    )


def _partition_prompts(
    prompts: Sequence[str],
    *,
    split_weights: Mapping[str, int],
    namespace: str,
) -> dict[str, tuple[str, ...]]:
    """Assign each unique prompt to one split with deterministic largest-remainder quotas."""

    ranked = sorted(
        prompts,
        key=lambda prompt: (
            _selection_rank("molor-router-v2-curriculum", namespace, _normalize_prompt(prompt)),
            _normalize_prompt(prompt),
        ),
    )
    total_weight = sum(split_weights.values())
    raw = {split: len(ranked) * split_weights[split] / total_weight for split in _SPLIT_ORDER}
    quotas = {split: int(raw[split]) for split in _SPLIT_ORDER}
    remainder = len(ranked) - sum(quotas.values())
    remainder_order = sorted(
        _SPLIT_ORDER,
        key=lambda split: (-(raw[split] - quotas[split]), _SPLIT_ORDER.index(split)),
    )
    for split in remainder_order[:remainder]:
        quotas[split] += 1
    if any(quotas[split] < 1 for split in _SPLIT_ORDER):
        raise ValueError(f"curriculum namespace {namespace!r} cannot cover every split")

    partition: dict[str, tuple[str, ...]] = {}
    offset = 0
    for split in _SPLIT_ORDER:
        next_offset = offset + quotas[split]
        partition[split] = tuple(ranked[offset:next_offset])
        offset = next_offset
    if offset != len(ranked):
        raise AssertionError("curriculum partition accounting failed")
    return partition


def _insert_curriculum(
    connection: sqlite3.Connection,
    curriculum: RouterCurriculumV2,
) -> tuple[
    dict[str, dict[str, int]],
    set[str],
    set[str],
    dict[str, object],
]:
    balances: dict[str, Counter[str]] = {split: Counter() for split in _SPLIT_ORDER}
    anchor_seed_balance: dict[str, Counter[str]] = {split: Counter() for split in _SPLIT_ORDER}
    anchor_digests: set[str] = set()
    base_digests: set[str] = set()

    def insert(
        *,
        route_id: str,
        split: str,
        prompt: str,
        origin: str,
        rank_parts: Sequence[object],
        destination: set[str],
    ) -> None:
        normalized = _normalize_prompt(prompt)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        prior = connection.execute(
            "SELECT route_id, split, origin FROM occurrences WHERE digest = ? LIMIT 1",
            (digest,),
        ).fetchone()
        if prior is not None:
            raise ValueError(
                "curriculum prompt collides with admitted chat supervision; "
                f"route={route_id!r}, split={split!r}, existing={tuple(prior)!r}"
            )
        connection.execute(
            """
            INSERT INTO occurrences(
                digest, normalized, prompt, route_id, split, selection_rank, origin,
                route_priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                normalized,
                prompt,
                route_id,
                split,
                "!" + _selection_rank("molor-router-v2-curriculum", *rank_parts, digest),
                origin,
                CURRICULUM_PRIORITY,
            ),
        )
        balances[split][route_id] += 1
        destination.add(digest)

    base_group_balance = {split: {"groups": 0, "records": 0} for split in _SPLIT_ORDER}
    base_group_split_owners: dict[str, str] = {}
    for split in _SPLIT_ORDER:
        for group_id, prompts in curriculum.base_groups_by_split[split]:
            base_group_split_owners[group_id] = split
            base_group_balance[split]["groups"] += 1
            base_group_balance[split]["records"] += len(prompts)
            for sequence, prompt in enumerate(prompts):
                insert(
                    route_id="base/null",
                    split=split,
                    prompt=prompt,
                    origin=f"router-curriculum-v2/true-general-negative/{group_id}",
                    rank_parts=("base/null", split, group_id, sequence),
                    destination=base_digests,
                )

    for leaf, seeds in sorted(curriculum.anchor_seeds_by_leaf.items()):
        partition = _partition_prompts(
            seeds,
            split_weights=curriculum.split_weights,
            namespace=leaf,
        )
        for split in _SPLIT_ORDER:
            if len(partition[split]) != ANCHOR_SEEDS_BY_SPLIT[split]:
                raise AssertionError(
                    f"leaf {leaf!r} {split} seed partition has {len(partition[split])}, "
                    f"expected {ANCHOR_SEEDS_BY_SPLIT[split]}"
                )
            anchor_seed_balance[split][leaf] = len(partition[split])
            for seed_index, seed in enumerate(partition[split]):
                seed_family = hashlib.sha256(_normalize_prompt(seed).encode("utf-8")).hexdigest()
                for variant_index, template in enumerate(curriculum.anchor_surface_variants):
                    prompt = _validate_prompt(
                        template.format(prompt=seed),
                        f"leaf {leaf!r} expanded anchor",
                        short_anchor=True,
                    )
                    insert(
                        route_id=leaf,
                        split=split,
                        prompt=prompt,
                        origin="router-curriculum-v2/leaf-anchor",
                        rank_parts=(leaf, split, seed_family, seed_index, variant_index),
                        destination=anchor_digests,
                    )
            if balances[split][leaf] != ANCHOR_ROWS_BY_SPLIT[split]:
                raise AssertionError(
                    f"leaf {leaf!r} {split} anchor expansion has "
                    f"{balances[split][leaf]}, expected {ANCHOR_ROWS_BY_SPLIT[split]}"
                )
    return (
        {split: dict(sorted(counts.items())) for split, counts in balances.items()},
        anchor_digests,
        base_digests,
        {
            "anchor_semantic_seeds": {
                split: dict(sorted(counts.items())) for split, counts in anchor_seed_balance.items()
            },
            "anchor_surface_variants_per_seed": len(curriculum.anchor_surface_variants),
            "base_group_split_owners": dict(sorted(base_group_split_owners.items())),
            "base_group_balance": base_group_balance,
        },
    )


def _artifact(path: Path, *, records: int) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "records": records,
        "bytes": path.stat().st_size,
    }


@dataclass(frozen=True, slots=True)
class _OwnerCandidate:
    digest: str
    prompt: str
    route_id: str
    selection_rank: str
    origin: str

    @property
    def curriculum(self) -> bool:
        return self.origin.startswith("router-curriculum-v2/")

    @property
    def length_tier(self) -> str:
        return "short_le_256" if len(self.prompt) <= SHORT_PROMPT_MAX_CHARACTERS else "long_gt_256"

    @property
    def origin_style(self) -> str:
        if self.curriculum:
            return self.origin
        return self.origin.partition("/")[0]


def _select_route_candidates(
    candidates: Sequence[_OwnerCandidate],
    *,
    split: str,
    route_id: str,
) -> tuple[list[_OwnerCandidate], dict[str, object]]:
    """Select one route with curriculum, length, and source-style guarantees."""

    cap = ROUTER_V2_RECORD_CAPS[split]
    ranked = sorted(candidates, key=lambda row: (row.selection_rank, row.digest))
    curriculum = [row for row in ranked if row.curriculum]
    sources = [row for row in ranked if not row.curriculum]
    if len(curriculum) > cap:
        raise AssertionError(
            f"router v2 {split}/{route_id} curriculum has {len(curriculum)} rows, "
            f"above the {cap}-row cap"
        )

    style_cap = cap * SOURCE_STYLE_CAP_NUMERATOR // SOURCE_STYLE_CAP_DENOMINATOR
    selected = list(curriculum)
    selected_digests = {row.digest for row in selected}
    selected_source_styles: Counter[str] = Counter()

    def take(pool: Sequence[_OwnerCandidate], count: int) -> None:
        if count <= 0:
            return
        taken = 0
        for row in pool:
            if row.digest in selected_digests:
                continue
            style = row.origin_style
            if selected_source_styles[style] >= style_cap:
                continue
            if len(selected) >= cap:
                break
            selected.append(row)
            selected_digests.add(row.digest)
            selected_source_styles[style] += 1
            taken += 1
            if taken >= count:
                break

    available_length = Counter(row.length_tier for row in ranked)
    configured_short = MINIMUM_SHORT_ROWS[split]
    effective_short = min(configured_short, available_length["short_le_256"], cap)
    source_long = [row for row in sources if row.length_tier == "long_gt_256"]
    effective_long_source = min(
        MINIMUM_LONG_SOURCE_ROWS[split],
        len(source_long),
        cap - len(curriculum),
    )

    # Preserve a bounded long-source slice before filling with the usually denser short tier.
    take(source_long, effective_long_source)
    selected_short = sum(row.length_tier == "short_le_256" for row in selected)
    take(
        [row for row in sources if row.length_tier == "short_le_256"],
        effective_short - selected_short,
    )
    take(sources, cap - len(selected))

    selected_length = Counter(row.length_tier for row in selected)
    selected_long_source = sum(
        not row.curriculum and row.length_tier == "long_gt_256" for row in selected
    )
    if not {row.digest for row in curriculum}.issubset(selected_digests):
        raise AssertionError(f"router v2 {split}/{route_id} dropped curriculum rows")
    if selected_length["short_le_256"] < effective_short:
        raise AssertionError(
            f"router v2 {split}/{route_id} retained {selected_length['short_le_256']} "
            f"short rows, below the effective {effective_short}-row minimum"
        )
    if selected_long_source < effective_long_source:
        raise AssertionError(
            f"router v2 {split}/{route_id} retained {selected_long_source} long source "
            f"rows, below the effective {effective_long_source}-row minimum"
        )
    if any(count > style_cap for count in selected_source_styles.values()):
        raise AssertionError(f"router v2 {split}/{route_id} exceeded a source-style cap")

    source_available_styles = Counter(row.origin_style for row in sources)
    saturated_styles = sorted(
        style
        for style, available in source_available_styles.items()
        if available > selected_source_styles[style] and selected_source_styles[style] == style_cap
    )
    sparse_reasons: list[str] = []
    if available_length["short_le_256"] < configured_short:
        sparse_reasons.append("short-support-below-configured-minimum")
    if len(source_long) < MINIMUM_LONG_SOURCE_ROWS[split]:
        sparse_reasons.append("long-source-support-below-configured-minimum")
    if len(selected) < cap and len(selected) < len(ranked):
        sparse_reasons.append("source-style-cap-limited")
    elif len(selected) < cap:
        sparse_reasons.append("total-support-below-cap")

    return selected, {
        "cap": cap,
        "available_rows": len(ranked),
        "selected_rows": len(selected),
        "curriculum_available": len(curriculum),
        "curriculum_selected": sum(row.curriculum for row in selected),
        "source_available": len(sources),
        "source_selected": sum(not row.curriculum for row in selected),
        "length_tiers_available": dict(sorted(available_length.items())),
        "length_tiers_selected": dict(sorted(selected_length.items())),
        "configured_short_minimum": configured_short,
        "effective_short_minimum": effective_short,
        "short_minimum_met": selected_length["short_le_256"] >= effective_short,
        "configured_long_source_minimum": MINIMUM_LONG_SOURCE_ROWS[split],
        "effective_long_source_minimum": effective_long_source,
        "selected_long_source_rows": selected_long_source,
        "long_source_minimum_met": selected_long_source >= effective_long_source,
        "source_style_cap": style_cap,
        "source_styles_available": dict(sorted(source_available_styles.items())),
        "source_styles_selected": dict(sorted(selected_source_styles.items())),
        "source_styles_saturated": saturated_styles,
        "sparse_support": bool(sparse_reasons),
        "sparse_support_reasons": sparse_reasons,
    }


def build_router_dataset_v2(
    collected: CollectedExpertData,
    *,
    curriculum_path: str | Path,
    output_directory: str | Path,
    builder_paths: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Build a new v2 dataset without reading or mutating any v1 router artifact."""

    expert_routes = tuple(sorted(collected.family_by_leaf))
    curriculum = load_router_curriculum_v2(
        curriculum_path,
        expected_leaf_ids=expert_routes,
    )
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"router v2 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    database_path = stage / "router-examples-v2.sqlite3"
    connection = sqlite3.connect(database_path)
    published = False
    try:
        connection.executescript(
            """
            CREATE TABLE occurrences(
                digest TEXT NOT NULL,
                normalized TEXT NOT NULL,
                prompt TEXT NOT NULL,
                route_id TEXT NOT NULL,
                split TEXT NOT NULL,
                selection_rank TEXT NOT NULL,
                origin TEXT NOT NULL,
                route_priority INTEGER NOT NULL
            );
            CREATE INDEX occurrence_digest ON occurrences(digest);
            """
        )
        chat_rows = 0
        chat_prompts_length_bounded = 0
        for shard in collected.shards:
            rows, bounded = _scan_chat_for_router(connection, shard)
            chat_rows += rows
            chat_prompts_length_bounded += bounded
        connection.commit()

        (
            curriculum_balance,
            anchor_digests,
            base_digests,
            semantic_family_accounting,
        ) = _insert_curriculum(connection, curriculum)
        connection.commit()
        owner_resolution = _resolve_router_owners(connection)

        selected: dict[str, list[_OwnerCandidate]] = {split: [] for split in _SPLIT_ORDER}
        route_balance: dict[str, dict[str, int]] = {}
        length_style_accounting: dict[str, dict[str, dict[str, object]]] = {}
        capped_owners = 0
        for split in _SPLIT_ORDER:
            candidates_by_route: dict[str, list[_OwnerCandidate]] = {}
            raw_candidates = connection.execute(
                """
                SELECT o.digest, o.prompt, o.route_id, o.selection_rank, x.origin
                FROM owners AS o
                JOIN occurrences AS x
                  ON x.digest = o.digest
                 AND x.route_id = o.route_id
                 AND x.split = o.split
                 AND x.selection_rank = o.selection_rank
                WHERE o.split = ?
                ORDER BY o.route_id, o.selection_rank, o.digest
                """,
                (split,),
            )
            for digest, prompt, route_id, selection_rank, origin in raw_candidates:
                route = str(route_id)
                candidates_by_route.setdefault(route, []).append(
                    _OwnerCandidate(
                        digest=str(digest),
                        prompt=str(prompt),
                        route_id=route,
                        selection_rank=str(selection_rank),
                        origin=str(origin),
                    )
                )

            counts: Counter[str] = Counter()
            split_accounting: dict[str, dict[str, object]] = {}
            for route, candidates in sorted(candidates_by_route.items()):
                chosen, accounting = _select_route_candidates(
                    candidates,
                    split=split,
                    route_id=route,
                )
                selected[split].extend(chosen)
                counts[route] = len(chosen)
                capped_owners += len(candidates) - len(chosen)
                split_accounting[route] = accounting
            route_balance[split] = dict(sorted(counts.items()))
            length_style_accounting[split] = split_accounting

        required_routes = set(expert_routes) | {"base/null"}
        route_coverage: dict[str, dict[str, object]] = {}
        for split in _SPLIT_ORDER:
            present = set(route_balance[split])
            missing = sorted(required_routes - present)
            if missing:
                raise AssertionError(f"router v2 {split} split lacks required routes: {missing}")
            route_coverage[split] = {
                "routes_present": sorted(present),
                "routes_missing": missing,
                "full_route_support": True,
                "base_null_support": route_balance[split]["base/null"],
            }

        selected_digest_sets = {
            split: {row.digest for row in rows} for split, rows in selected.items()
        }
        overlap = {
            "train_validation": len(
                selected_digest_sets["train"] & selected_digest_sets["validation"]
            ),
            "train_test": len(selected_digest_sets["train"] & selected_digest_sets["test"]),
            "validation_test": len(
                selected_digest_sets["validation"] & selected_digest_sets["test"]
            ),
        }
        if any(overlap.values()):
            raise AssertionError(f"router v2 normalized prompt split overlap: {overlap}")

        selected_digests = set().union(*selected_digest_sets.values())
        missing_anchors = anchor_digests - selected_digests
        missing_base = base_digests - selected_digests
        if missing_anchors or missing_base:
            raise AssertionError(
                "curriculum rows were unexpectedly deduplicated or capped: "
                f"anchors={len(missing_anchors)}, base={len(missing_base)}"
            )

        outputs: dict[str, dict[str, object]] = {}
        selected_rows = 0
        for split in _SPLIT_ORDER:
            rows = sorted(
                selected[split],
                key=lambda item: (
                    _selection_rank("molor-router-output-v2", split, item.digest),
                    item.digest,
                ),
            )
            destination = stage / f"router-examples-v2.{split}.jsonl"
            with destination.open("wb") as stream:
                for row in rows:
                    stream.write(
                        canonical_json_bytes({"prompt": row.prompt, "route_id": row.route_id})
                    )
                    stream.write(b"\n")
            selected_rows += len(rows)
            outputs[split] = _artifact(destination, records=len(rows))

        curriculum_occurrences = len(anchor_digests) + len(base_digests)
        total_occurrences = chat_rows + curriculum_occurrences
        unique_owners = int(connection.execute("SELECT COUNT(*) FROM owners").fetchone()[0])
        resolved_losing = int(owner_resolution.get("resolved_conflict_losing_occurrences", 0))
        unresolved = int(owner_resolution.get("unresolved_conflict_occurrences", 0))
        duplicate_occurrences = int(owner_resolution.get("duplicate_or_cross_split_occurrences", 0))
        conflict_occurrences = resolved_losing + unresolved
        if unique_owners != int(owner_resolution.get("owners", 0)):
            raise AssertionError("router v2 resolved-owner accounting failed")
        if unique_owners != selected_rows + capped_owners:
            raise AssertionError("router v2 owner accounting failed")
        if total_occurrences != (
            conflict_occurrences + duplicate_occurrences + selected_rows + capped_owners
        ):
            raise AssertionError("router v2 occurrence accounting failed")

        builder_pins: list[dict[str, object]] = []
        for raw_path in sorted({Path(item).resolve() for item in builder_paths}, key=str):
            _ensure_nonbenchmark_input(raw_path)
            if raw_path.is_symlink() or not raw_path.is_file():
                raise ValueError(f"builder pin must be a regular non-symlink file: {raw_path}")
            builder_pins.append(
                {
                    "path": str(raw_path),
                    "sha256": _sha256_file(raw_path),
                    "bytes": raw_path.stat().st_size,
                }
            )

        manifest_content: dict[str, object] = {
            "schema_version": ROUTER_EXAMPLES_V2_MANIFEST_SCHEMA,
            "dataset_id": "molor-hierarchical-router-examples-v2",
            "revision": "2",
            "row_contract": {"exact_keys": ["prompt", "route_id"]},
            "route_contract": {
                "expert_routes": list(expert_routes),
                "negative_route": "base/null",
                "family_lookup": dict(sorted(collected.family_by_leaf.items())),
            },
            "evaluation_contract": {
                "all_splits": "fail-closed-full-route-support",
                "evidence_scope": (
                    "router-selection evaluation only; not expert-answer-quality evidence"
                ),
                "curriculum_validation_test_use": (
                    "optimization holdouts only; not independent generalization evidence"
                ),
            },
            "selection": {
                "seed": "molor-router-v2-curriculum",
                "per_route_caps": ROUTER_V2_RECORD_CAPS,
                "normalized_prompt": "NFKC+casefold+collapsed-whitespace",
                "maximum_source_prompt_characters": ROUTER_MAX_PROMPT_CHARACTERS,
                "maximum_curriculum_prompt_characters": MAX_CURRICULUM_PROMPT_CHARACTERS,
                "short_prompt_maximum_characters": SHORT_PROMPT_MAX_CHARACTERS,
                "minimum_short_rows": MINIMUM_SHORT_ROWS,
                "minimum_long_source_rows": MINIMUM_LONG_SOURCE_ROWS,
                "source_style_cap_fraction": {
                    "numerator": SOURCE_STYLE_CAP_NUMERATOR,
                    "denominator": SOURCE_STYLE_CAP_DENOMINATOR,
                    "basis": "per-route split cap; curriculum excluded",
                },
                "cross_split_policy": "test-then-validation-then-train",
                "curriculum_priority": CURRICULUM_PRIORITY,
                "curriculum_collision_policy": "fail-closed-against-admitted-chat",
                "route_conflicts": (
                    "lowest-source-priority-route-wins; equal-priority-ambiguity-excluded"
                ),
                "source_priorities": {
                    **dict(sorted(ROUTER_SOURCE_PRIORITIES.items())),
                    "unlisted-source-default": 50,
                    "curriculum": CURRICULUM_PRIORITY,
                },
            },
            "sources": {
                "expert_chat_shards": [
                    {
                        "shard_id": shard.descriptor.shard_id,
                        "shard_revision": shard.descriptor.shard_revision,
                        "sha256": shard.descriptor.sha256,
                        "records": shard.descriptor.record_count,
                    }
                    for shard in collected.shards
                ],
                "true_general_negatives": {
                    "source": "router-curriculum-v2/base-negative-groups",
                    "legacy_router_base_source_used": False,
                    "split_policy": "explicit-whole-template-family-ownership",
                    "groups": dict(curriculum.group_accounting),
                    "specialist_cue_deny_terms": list(curriculum.base_deny_terms),
                },
                "short_route_anchors": {
                    "source": "router-curriculum-v2/leaf-anchors",
                    "routes": {
                        leaf: {
                            "semantic_seeds": len(seeds),
                            "surface_variants_per_seed": len(curriculum.anchor_surface_variants),
                            "expanded_records": len(seeds)
                            * len(curriculum.anchor_surface_variants),
                        }
                        for leaf, seeds in sorted(curriculum.anchor_seeds_by_leaf.items())
                    },
                },
                "curriculum": {
                    "path": str(curriculum.path),
                    "sha256": curriculum.file_sha256,
                    "content_sha256": curriculum.content_sha256,
                    "dataset_id": curriculum.dataset_id,
                    "revision": curriculum.revision,
                    "authoring_contract": dict(curriculum.authoring_contract),
                },
                "taxonomy": {
                    "path": collected.taxonomy_path,
                    "sha256": collected.taxonomy_sha256,
                    "leaf_revision": collected.leaf_revision,
                },
            },
            "benchmark_isolation": {
                "benchmark_material_used": False,
                "benchmark_or_route_contract_input_paths": [],
                "input_path_policy": (
                    "reject benchmark, molor-core, and route-contract markers before read"
                ),
            },
            "curriculum_balance": curriculum_balance,
            "curriculum_semantic_family_accounting": semantic_family_accounting,
            "route_balance": route_balance,
            "route_coverage": route_coverage,
            "length_origin_style_accounting": length_style_accounting,
            "normalized_prompt_overlap": overlap,
            "accounting": {
                "chat_occurrences": chat_rows,
                "chat_prompts_length_bounded": chat_prompts_length_bounded,
                "curated_base_null_occurrences": len(base_digests),
                "short_anchor_occurrences": len(anchor_digests),
                "total_occurrences": total_occurrences,
                "route_conflict_occurrences": conflict_occurrences,
                "resolved_conflict_losing_occurrences": resolved_losing,
                "unresolved_conflict_occurrences": unresolved,
                "duplicate_or_cross_split_occurrences": duplicate_occurrences,
                "selected_rows": selected_rows,
                "capped_unique_prompts": capped_owners,
                "fully_reconciled": True,
                "zero_silent_drops": True,
            },
            "route_conflict_sets": owner_resolution["route_conflict_sets"],
            "outputs": outputs,
            "builders": builder_pins,
        }
        manifest = {**manifest_content, "content_sha256": fingerprint_json(manifest_content)}
        manifest_path = stage / "router-examples-v2.manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))

        connection.close()
        database_path.unlink()
        sums = [
            (path.name, _sha256_file(path))
            for path in sorted(stage.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "sha256sums.txt"
        ]
        (stage / "sha256sums.txt").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sums),
            encoding="utf-8",
            newline="",
        )
        os.replace(stage, output)
        published = True
        return {
            "output_directory": str(output),
            "manifest_path": str(output / manifest_path.name),
            "manifest_sha256": _sha256_file(output / manifest_path.name),
            "content_sha256": manifest["content_sha256"],
            "records": {split: outputs[split]["records"] for split in _SPLIT_ORDER},
            "route_balance": route_balance,
        }
    finally:
        connection.close()
        if not published and stage.exists():
            shutil.rmtree(stage)


__all__ = [
    "ROUTER_CURRICULUM_V2_SCHEMA",
    "ROUTER_EXAMPLES_V2_MANIFEST_SCHEMA",
    "ROUTER_V2_RECORD_CAPS",
    "RouterCurriculumV2",
    "build_router_dataset_v2",
    "load_router_curriculum_v2",
]
