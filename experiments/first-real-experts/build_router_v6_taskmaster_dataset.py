"""Build the sealed six-route router corpus with the quarantined Taskmaster v2 expert.

The promoted v4 router rows are retained exactly.  The sixth route is populated only from the
hash-pinned Taskmaster v2 training split and the hash-pinned Taskmaster v2 validation split.  The
Taskmaster test split, generation/evaluation files, predictions, and benchmark artifacts are never
discovered or read by this builder.

Taskmaster's original connected-component identities are carried by the matching provenance
sidecars.  Selection keeps every component whole: router training positives come from Taskmaster
train, while router validation and test positives are two disjoint subsets of Taskmaster
validation.  The resulting six routes retain v4's exact 936/130/130 balance.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Iterable

from build_router_v2_dataset import (
    Candidate,
    _assert_unique_prompts,
    _normalize_prompt,
    _report_path,
    _sha256_file,
    _split_path,
    _stable_hash,
    _write_jsonl,
)
from build_router_v4_hard_negative_dataset import (
    DATA_ROOT,
    ROUTE_IDS as V4_ROUTE_IDS,
    SPLITS,
)


TASKMASTER_DATASET_ID = "taskmaster-coffee-api-v2"
TASKMASTER_ROUTE_ID = "taskmaster-coffee-api-v2-full-dialog-r16"
ROUTE_IDS = (*V4_ROUTE_IDS[:-1], TASKMASTER_ROUTE_ID, V4_ROUTE_IDS[-1])
TARGETS = {"train": 936, "validation": 130, "test": 130}
TASKMASTER_SOURCE_SPLITS = ("train", "validation")
TASKMASTER_PROMPT_PREFIX = "Taskmaster-4 conversation context:\n"
TASKMASTER_PROVENANCE_SCHEMA = "molor.taskmaster-coffee-api-provenance.v2"
TASKMASTER_EXAMPLE_PREFIX = "tm4-coffee-api-v2:"
V4_SCHEMA = "molor.router-bootstrap-dataset.v4"
V6_SCHEMA = "molor.router-bootstrap-dataset.v6"

# These pins identify the already sealed local artifacts.  They also make it impossible to point
# the builder at a renamed held-out or generated artifact with a compatible-looking row schema.
PINNED_V4_REPORT_SHA256 = "b566cde269bbc4e9a2886685de5844a99dff8c6b343355800355dd72513bbddb"
PINNED_TASKMASTER_SHA256 = {
    "train_training": "500ac9caa235497f9fb86d1749911d50c0a494136050eb91cd1b807c8689d209",
    "train_provenance": "145dfcd83f734f57be0196ac36e58a1a1e1728464cd96cc53e6b1db6849efa0f",
    "validation_training": ("214aac61ddeca261d70a4381a15a973d17ac89d39d20007c2541f3724205176f"),
    "validation_provenance": ("c315ae5e4db52ee37cbb760c60745fdea5b371f2c451cfb1b90919883fae30e4"),
}


@dataclass(frozen=True, slots=True)
class TaskmasterRow:
    """One routing prompt paired with its sealed Taskmaster provenance."""

    candidate: Candidate
    example_id: str
    group_id: str


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _taskmaster_training_path(prefix: Path, split: str) -> Path:
    return prefix.with_name(f"{prefix.name}.{split}.jsonl")


def _taskmaster_provenance_path(prefix: Path, split: str) -> Path:
    return prefix.with_name(f"{prefix.name}.{split}.provenance.jsonl")


def _authorized_taskmaster_path(prefix: Path, *, split: str, provenance: bool) -> Path:
    """Resolve one explicit train/validation input without directory discovery."""

    if prefix.name != TASKMASTER_DATASET_ID:
        raise ValueError(
            f"Taskmaster prefix must end in {TASKMASTER_DATASET_ID!r}, got {prefix.name!r}"
        )
    if split not in TASKMASTER_SOURCE_SPLITS:
        raise ValueError(f"unauthorized Taskmaster source split: {split!r}")
    path = (
        _taskmaster_provenance_path(prefix, split)
        if provenance
        else _taskmaster_training_path(prefix, split)
    )
    expected_name = (
        f"{TASKMASTER_DATASET_ID}.{split}.provenance.jsonl"
        if provenance
        else f"{TASKMASTER_DATASET_ID}.{split}.jsonl"
    )
    if path.name != expected_name:
        raise AssertionError(f"Taskmaster input path escaped the allowlist: {path}")
    if path.is_symlink():
        raise ValueError(f"Taskmaster input may not be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"required Taskmaster input is missing: {path}")
    resolved = path.resolve()
    if resolved.name != expected_name:
        raise ValueError(f"resolved Taskmaster input escaped the allowlist: {resolved}")
    identity_key = f"{split}_{'provenance' if provenance else 'training'}"
    actual_hash = _sha256_file(resolved)
    if actual_hash != PINNED_TASKMASTER_SHA256[identity_key]:
        raise ValueError(
            f"Taskmaster {identity_key} differs from its sealed identity: {actual_hash}"
        )
    return resolved


def _validate_training_messages(
    payload: Any,
    provenance: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> str:
    if not isinstance(payload, dict) or set(payload) != {"messages"}:
        raise ValueError(f"{path}:{line_number}: row must contain exactly messages")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"{path}:{line_number}: expected exactly three training messages")
    expected_roles = ("system", "user", "assistant")
    for index, (message, role) in enumerate(zip(messages, expected_roles, strict=True)):
        if (
            not isinstance(message, dict)
            or set(message) != {"content", "role"}
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            or not message["content"]
        ):
            raise ValueError(f"{path}:{line_number}: malformed {role} message at position {index}")
    messages_hash = _sha256_text(_canonical_json(messages))
    if provenance.get("training_messages_sha256") != messages_hash:
        raise ValueError(f"{path}:{line_number}: training/provenance identity mismatch")
    prompt = messages[1]["content"]
    if not prompt.startswith(TASKMASTER_PROMPT_PREFIX):
        raise ValueError(f"{path}:{line_number}: unexpected Taskmaster routing prompt")
    context = prompt.removeprefix(TASKMASTER_PROMPT_PREFIX)
    if provenance.get("context_sha256") != _sha256_text(context):
        raise ValueError(f"{path}:{line_number}: context hash mismatch")
    if provenance.get("target_sha256") != _sha256_text(messages[2]["content"]):
        raise ValueError(f"{path}:{line_number}: target hash mismatch")
    return prompt


def _routing_component_id(source_group_ids: Iterable[str]) -> str:
    members = sorted(set(source_group_ids))
    if not members:
        raise ValueError("Taskmaster routing component has no source groups")
    if len(members) == 1:
        return members[0]
    digest = _sha256_text("\0".join(members))
    return f"tm4-coffee-v2-router-union:{digest}"


def _collapse_normalized_taskmaster_duplicates(
    rows: list[TaskmasterRow],
) -> tuple[list[TaskmasterRow], dict[str, int], dict[str, list[str]]]:
    """Deduplicate router-equivalent prompts without splitting source components."""

    by_prompt: dict[str, list[TaskmasterRow]] = defaultdict(list)
    source_group_ids = {row.group_id for row in rows}
    union_find = _UnionFind(source_group_ids)
    for row in rows:
        by_prompt[row.candidate.normalized_prompt].append(row)
    for prompt_rows in by_prompt.values():
        groups = sorted({row.group_id for row in prompt_rows})
        for group_id in groups[1:]:
            union_find.union(groups[0], group_id)

    source_groups_by_root: dict[str, list[str]] = defaultdict(list)
    for group_id in source_group_ids:
        source_groups_by_root[union_find.find(group_id)].append(group_id)
    routing_group_by_source: dict[str, str] = {}
    routing_component_unions: dict[str, list[str]] = {}
    for members in source_groups_by_root.values():
        routing_group = _routing_component_id(members)
        for source_group in members:
            routing_group_by_source[source_group] = routing_group
        if len(members) > 1:
            routing_component_unions[routing_group] = sorted(members)

    collapsed: list[TaskmasterRow] = []
    for normalized_prompt, prompt_rows in sorted(by_prompt.items()):
        representative = min(prompt_rows, key=lambda row: row.example_id)
        routing_groups = {routing_group_by_source[row.group_id] for row in prompt_rows}
        if len(routing_groups) != 1:
            raise AssertionError("normalized duplicate escaped Taskmaster component union")
        routing_group = next(iter(routing_groups))
        candidate = representative.candidate
        collapsed.append(
            TaskmasterRow(
                candidate=Candidate(
                    prompt=candidate.prompt,
                    normalized_prompt=normalized_prompt,
                    route_id=candidate.route_id,
                    split=candidate.split,
                    semantic_group=f"taskmaster-v2/{routing_group}",
                    source=candidate.source,
                ),
                example_id=representative.example_id,
                group_id=routing_group,
            )
        )
    _assert_unique_prompts(row.candidate for row in collapsed)
    return (
        collapsed,
        {
            "source_rows": len(rows),
            "routing_rows": len(collapsed),
            "normalized_prompt_duplicates_collapsed": len(rows) - len(collapsed),
            "source_components": len(source_group_ids),
            "routing_components": len(set(routing_group_by_source.values())),
            "routing_component_unions": len(routing_component_unions),
        },
        dict(sorted(routing_component_unions.items())),
    )


def _load_taskmaster_rows(
    prefix: Path,
    *,
    split: str,
) -> tuple[
    list[TaskmasterRow],
    dict[str, dict[str, Any]],
    dict[str, int],
    dict[str, list[str]],
]:
    training_path = _authorized_taskmaster_path(prefix, split=split, provenance=False)
    provenance_path = _authorized_taskmaster_path(prefix, split=split, provenance=True)
    rows: list[TaskmasterRow] = []
    seen_examples: set[str] = set()
    with (
        training_path.open("r", encoding="utf-8") as training_stream,
        provenance_path.open("r", encoding="utf-8") as provenance_stream,
    ):
        for line_number, pair in enumerate(
            zip_longest(training_stream, provenance_stream),
            start=1,
        ):
            training_line, provenance_line = pair
            if training_line is None or provenance_line is None:
                raise ValueError(f"Taskmaster {split} training/provenance row counts differ")
            if not training_line.strip() or not provenance_line.strip():
                raise ValueError(f"Taskmaster {split} contains a blank row at {line_number}")
            try:
                payload = json.loads(training_line)
                provenance = json.loads(provenance_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid Taskmaster {split} JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(provenance, dict):
                raise ValueError(f"Taskmaster provenance row {line_number} is not an object")
            if provenance.get("schema_version") != TASKMASTER_PROVENANCE_SCHEMA:
                raise ValueError(f"Taskmaster provenance row {line_number} has wrong schema")
            if provenance.get("split") != split:
                raise ValueError(f"Taskmaster provenance row {line_number} has wrong split")
            example_id = provenance.get("example_id")
            group_id = provenance.get("group_id")
            if (
                not isinstance(example_id, str)
                or not example_id.startswith(TASKMASTER_EXAMPLE_PREFIX)
                or example_id in seen_examples
            ):
                raise ValueError(f"Taskmaster provenance row {line_number} has invalid example_id")
            if not isinstance(group_id, str) or not group_id.startswith("tm4-coffee-v2-component:"):
                raise ValueError(f"Taskmaster provenance row {line_number} has invalid group_id")
            prompt = _validate_training_messages(
                payload,
                provenance,
                path=training_path,
                line_number=line_number,
            )
            seen_examples.add(example_id)
            rows.append(
                TaskmasterRow(
                    candidate=Candidate(
                        prompt=prompt,
                        normalized_prompt=_normalize_prompt(prompt),
                        route_id=TASKMASTER_ROUTE_ID,
                        split=split,
                        semantic_group=f"taskmaster-v2/{group_id}",
                        source=f"{TASKMASTER_DATASET_ID}.{split}",
                    ),
                    example_id=example_id,
                    group_id=group_id,
                )
            )
    source_row_count = len(rows)
    rows, routing_stats, routing_unions = _collapse_normalized_taskmaster_duplicates(rows)
    descriptors = {
        "training": {
            "path": str(training_path),
            "rows": source_row_count,
            "sha256": _sha256_file(training_path),
        },
        "provenance": {
            "path": str(provenance_path),
            "rows": source_row_count,
            "sha256": _sha256_file(provenance_path),
        },
    }
    return rows, descriptors, routing_stats, routing_unions


def _select_whole_components(
    rows: Iterable[TaskmasterRow],
    *,
    target: int,
    seed: int,
    namespace: str,
) -> tuple[list[TaskmasterRow], tuple[str, ...]]:
    by_group: dict[str, list[TaskmasterRow]] = defaultdict(list)
    for row in rows:
        by_group[row.group_id].append(row)
    ordered_groups = sorted(
        by_group,
        key=lambda group_id: _stable_hash(seed, namespace, group_id),
    )
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id in ordered_groups:
        size = len(by_group[group_id])
        additions: dict[int, tuple[str, ...]] = {}
        for current, selected_groups in sorted(reachable.items(), reverse=True):
            new_total = current + size
            if new_total <= target and new_total not in reachable and new_total not in additions:
                additions[new_total] = (*selected_groups, group_id)
        reachable.update(additions)
        if target in reachable:
            break
    if target not in reachable:
        available = sum(len(group_rows) for group_rows in by_group.values())
        raise ValueError(
            f"cannot select {target} whole Taskmaster components for {namespace}; "
            f"{available} rows available"
        )
    selected_group_ids = reachable[target]
    selected = [row for group_id in selected_group_ids for row in by_group[group_id]]
    selected.sort(
        key=lambda row: _stable_hash(
            seed,
            "taskmaster-output",
            namespace,
            row.group_id,
            row.example_id,
        )
    )
    if len(selected) != target:
        raise AssertionError(f"Taskmaster {namespace} selection has wrong size")
    return selected, tuple(sorted(selected_group_ids))


def _load_v4_rows(prefix: Path) -> tuple[dict[str, list[Candidate]], dict[str, Any]]:
    report_path = _report_path(prefix).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"v4 router report does not exist: {report_path}")
    report_hash = _sha256_file(report_path)
    if report_hash != PINNED_V4_REPORT_SHA256:
        raise ValueError(f"v4 router report differs from its sealed identity: {report_hash}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid v4 router report: {report_path}") from exc
    if report.get("schema_version") != V4_SCHEMA:
        raise ValueError("v4 router report has the wrong schema")
    if report.get("route_ids") != list(V4_ROUTE_IDS):
        raise ValueError("v4 router report has an unexpected route axis")
    if any(
        report.get(field)
        != {
            "train/test": 0,
            "train/validation": 0,
            "validation/test": 0,
        }
        for field in ("normalized_prompt_overlap", "semantic_group_overlap")
    ):
        raise ValueError("v4 router report does not prove split disjointness")

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("v4 router report has no output identities")
    results = {split: [] for split in SPLITS}
    for split in SPLITS:
        path = _split_path(prefix, split).resolve()
        descriptor = outputs.get(split)
        if not path.is_file():
            raise FileNotFoundError(f"v4 router split does not exist: {path}")
        if not isinstance(descriptor, dict) or descriptor.get("sha256") != _sha256_file(path):
            raise ValueError(f"v4 router split differs from its report: {path}")
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank v4 router row at {path}:{line_number}")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid v4 JSON at {path}:{line_number}") from exc
                if not isinstance(raw, dict) or set(raw) != {"prompt", "route_id"}:
                    raise ValueError(f"malformed v4 row at {path}:{line_number}")
                prompt = raw["prompt"]
                route_id = raw["route_id"]
                if (
                    not isinstance(prompt, str)
                    or not prompt.strip()
                    or route_id not in V4_ROUTE_IDS
                ):
                    raise ValueError(f"invalid v4 row at {path}:{line_number}")
                normalized = _normalize_prompt(prompt)
                results[split].append(
                    Candidate(
                        prompt=prompt,
                        normalized_prompt=normalized,
                        route_id=route_id,
                        split=split,
                        semantic_group=f"v4-preserved/{split}/{_stable_hash(normalized)}",
                        source="router-bootstrap-v4-hard-negative",
                    )
                )
        if len(results[split]) != descriptor.get("rows"):
            raise ValueError(f"v4 {split} row count differs from its report")
        expected_balance = {route_id: TARGETS[split] for route_id in V4_ROUTE_IDS}
        if dict(Counter(row.route_id for row in results[split])) != expected_balance:
            raise ValueError(f"v4 {split} route balance differs from the sealed contract")
    _assert_unique_prompts(row for split in SPLITS for row in results[split])
    source = {
        "report": {
            "path": str(report_path),
            "sha256": report_hash,
            "schema_version": report["schema_version"],
        },
        "outputs": {
            split: {
                "path": str(_split_path(prefix, split).resolve()),
                "rows": len(results[split]),
                "sha256": outputs[split]["sha256"],
            }
            for split in SPLITS
        },
    }
    return results, source


def _overlap_counts(
    selected: dict[str, list[Candidate]],
    attribute: str,
) -> dict[str, int]:
    values = {split: {getattr(row, attribute) for row in selected[split]} for split in SPLITS}
    return {
        f"{left}/{right}": len(values[left].intersection(values[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }


def build_router_v6_dataset(
    *,
    v4_prefix: Path = DATA_ROOT / "router-bootstrap-v4-hard-negative-s307",
    taskmaster_prefix: Path = DATA_ROOT / TASKMASTER_DATASET_ID,
    output_prefix: Path = DATA_ROOT / "router-bootstrap-v6-taskmaster",
    selection_seed: int = 613,
) -> dict[str, Any]:
    """Build hash-locked, group-disjoint six-route router splits."""

    v4_rows, v4_source = _load_v4_rows(v4_prefix.resolve())
    (
        taskmaster_train,
        train_descriptors,
        train_routing_stats,
        train_routing_unions,
    ) = _load_taskmaster_rows(taskmaster_prefix.resolve(), split="train")
    (
        taskmaster_validation,
        validation_descriptors,
        validation_routing_stats,
        validation_routing_unions,
    ) = _load_taskmaster_rows(
        taskmaster_prefix.resolve(),
        split="validation",
    )
    train_source_groups = {row.group_id for row in taskmaster_train}
    validation_source_groups = {row.group_id for row in taskmaster_validation}
    if train_source_groups.intersection(validation_source_groups):
        raise ValueError("Taskmaster train and validation components overlap")

    selected_train, train_groups = _select_whole_components(
        taskmaster_train,
        target=TARGETS["train"],
        seed=selection_seed,
        namespace="train",
    )
    selected_validation, validation_groups = _select_whole_components(
        taskmaster_validation,
        target=TARGETS["validation"],
        seed=selection_seed,
        namespace="validation",
    )
    heldout_validation_groups = set(validation_groups)
    remaining_validation = [
        row for row in taskmaster_validation if row.group_id not in heldout_validation_groups
    ]
    selected_test, test_groups = _select_whole_components(
        remaining_validation,
        target=TARGETS["test"],
        seed=selection_seed,
        namespace="test",
    )
    taskmaster_by_split = {
        "train": selected_train,
        "validation": selected_validation,
        "test": selected_test,
    }
    selected_group_sets = {
        "train": set(train_groups),
        "validation": set(validation_groups),
        "test": set(test_groups),
    }
    component_overlap = {
        f"{left}/{right}": len(selected_group_sets[left].intersection(selected_group_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(component_overlap.values()):
        raise AssertionError("Taskmaster components leaked across v6 splits")

    selected: dict[str, list[Candidate]] = {}
    balance: dict[str, dict[str, int]] = {}
    legacy_preservation: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        taskmaster_candidates = [row.candidate for row in taskmaster_by_split[split]]
        merged = [*v4_rows[split], *taskmaster_candidates]
        _assert_unique_prompts(merged)
        merged.sort(
            key=lambda row: _stable_hash(
                selection_seed,
                "v6-output",
                split,
                row.route_id,
                row.normalized_prompt,
            )
        )
        counts = dict(Counter(row.route_id for row in merged))
        expected = {route_id: TARGETS[split] for route_id in ROUTE_IDS}
        if counts != expected:
            raise AssertionError(f"router v6 {split} split is not route-balanced: {counts}")
        legacy_counter = Counter(
            (row.prompt, row.route_id) for row in merged if row.route_id != TASKMASTER_ROUTE_ID
        )
        source_counter = Counter((row.prompt, row.route_id) for row in v4_rows[split])
        if legacy_counter != source_counter:
            raise AssertionError(f"router v6 changed legacy v4 rows in {split}")
        selected[split] = merged
        balance[split] = counts
        legacy_preservation[split] = {
            "rows": len(v4_rows[split]),
            "exact_row_multiset_preserved": True,
        }

    prompt_overlap = _overlap_counts(selected, "normalized_prompt")
    semantic_group_overlap = _overlap_counts(selected, "semantic_group")
    if any(prompt_overlap.values()) or any(semantic_group_overlap.values()):
        raise AssertionError("router v6 construction leaked prompts or semantic groups")

    outputs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = _split_path(output_prefix, split).resolve()
        count, digest = _write_jsonl(path, selected[split])
        outputs[split] = {"path": str(path), "rows": count, "sha256": digest}

    builder_path = Path(__file__).resolve()
    report = {
        "schema_version": V6_SCHEMA,
        "route_ids": list(ROUTE_IDS),
        "evaluation_artifacts_read_by_builder": False,
        "taskmaster_test_split_read_by_builder": False,
        "taskmaster_prediction_artifacts_read_by_builder": False,
        "authorized_taskmaster_source_splits": list(TASKMASTER_SOURCE_SPLITS),
        "seeds": {"component_selection": selection_seed},
        "split_contract": {
            "legacy_v4": "every v4 prompt/route row is retained in its original split",
            "taskmaster_train": (
                "936 positives selected as whole components from Taskmaster v2 train"
            ),
            "taskmaster_validation": (
                "130 validation and 130 test positives selected as disjoint whole components "
                "from Taskmaster v2 validation"
            ),
        },
        "source_identity": {
            "builder": str(builder_path),
            "builder_sha256": _sha256_file(builder_path),
        },
        "legacy_v4_source": v4_source,
        "legacy_v4_preservation": legacy_preservation,
        "taskmaster_source": {
            "dataset_id": TASKMASTER_DATASET_ID,
            "route_id": TASKMASTER_ROUTE_ID,
            "source_split_mapping": {
                "train": "train",
                "validation": "validation",
                "test": "validation",
            },
            "inputs": {
                "train": train_descriptors,
                "validation": validation_descriptors,
            },
            "routing_normalization": {
                "train": train_routing_stats,
                "validation": validation_routing_stats,
            },
            "routing_component_unions": {
                "train": train_routing_unions,
                "validation": validation_routing_unions,
            },
            "available_rows": {
                "train": train_routing_stats["source_rows"],
                "validation": validation_routing_stats["source_rows"],
            },
            "available_components": {
                "train": train_routing_stats["source_components"],
                "validation": validation_routing_stats["source_components"],
            },
            "available_routing_rows": {
                "train": len(taskmaster_train),
                "validation": len(taskmaster_validation),
            },
            "available_routing_components": {
                "train": len(train_source_groups),
                "validation": len(validation_source_groups),
            },
            "selected_rows": {split: len(taskmaster_by_split[split]) for split in SPLITS},
            "selected_components": {split: len(selected_group_sets[split]) for split in SPLITS},
            "selected_component_ids": {
                split: sorted(selected_group_sets[split]) for split in SPLITS
            },
            "selected_component_overlap": component_overlap,
            "train_validation_source_component_overlap": 0,
        },
        "balance": balance,
        "normalized_prompt_overlap": prompt_overlap,
        "semantic_group_overlap": semantic_group_overlap,
        "duplicate_normalized_prompts": 0,
        "outputs": outputs,
    }
    report_path = _report_path(output_prefix).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v4-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v4-hard-negative-s307",
    )
    parser.add_argument(
        "--taskmaster-prefix",
        type=Path,
        default=DATA_ROOT / TASKMASTER_DATASET_ID,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v6-taskmaster",
    )
    parser.add_argument("--selection-seed", type=int, default=613)
    args = parser.parse_args()
    report = build_router_v6_dataset(
        v4_prefix=args.v4_prefix,
        taskmaster_prefix=args.taskmaster_prefix,
        output_prefix=args.output_prefix,
        selection_seed=args.selection_seed,
    )
    print(
        json.dumps(
            {
                "route_ids": report["route_ids"],
                "balance": report["balance"],
                "taskmaster_selected_components": report["taskmaster_source"][
                    "selected_components"
                ],
                "outputs": report["outputs"],
                "report": report["report_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
