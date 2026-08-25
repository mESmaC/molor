"""Deterministically shard blind review packs and merge external judgments.

The shard operation keeps every prompt group intact.  The merge operation uses the
original review pack as the candidate registry, so accidental cross-run, duplicate,
or fabricated candidate judgments fail closed before a merged file is written.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCHEMA_REVIEW = "molor.teacher-eval.review-pack.v1"
SCHEMA_JUDGMENTS = "molor.teacher-eval.judgments.v1"
SCHEMA_SHARD = "molor.teacher-eval.review-shard.v1"
SCHEMA_SHARD_INDEX = "molor.teacher-eval.review-shard-index.v1"
RUBRIC_DIMENSIONS = {
    "accuracy",
    "instruction_following",
    "reasoning",
    "relevance",
    "style",
    "calibration",
}


class ReviewPackError(RuntimeError):
    """A review pack or judgments file is unsafe to shard or merge."""


@dataclass(frozen=True)
class ReviewPack:
    payload: dict[str, Any]
    run_id: str
    manifest_sha256: str
    rubric_version: str
    teacher_template: dict[str, str]
    groups: tuple[dict[str, Any], ...]
    candidate_order: tuple[str, ...]
    candidate_to_prompt: Mapping[str, str]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReviewPackError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise ReviewPackError(f"cannot read {label} {path}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReviewPackError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewPackError(f"{label} must contain one JSON object")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, ValueError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ReviewPackError(f"cannot write {path}: {exc}") from exc


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewPackError(f"{label} must be a non-empty string")
    return value


def _teacher(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"id", "model", "provider"}:
        raise ReviewPackError(f"{label} must contain exactly id, model, and provider")
    return {
        field: _nonempty_string(value.get(field), f"{label}.{field}")
        for field in ("id", "model", "provider")
    }


def load_review_pack(path: Path) -> ReviewPack:
    """Load and validate the identities and candidate registry in a review pack."""

    resolved = path.expanduser().resolve()
    raw = _load_object(resolved, "review pack")
    if raw.get("schema_version") != SCHEMA_REVIEW:
        raise ReviewPackError(f"review_pack.schema_version must be {SCHEMA_REVIEW!r}")
    run_id = _nonempty_string(raw.get("run_id"), "review_pack.run_id")
    manifest_sha256 = _nonempty_string(raw.get("manifest_sha256"), "review_pack.manifest_sha256")
    template = raw.get("judgments_template")
    if not isinstance(template, Mapping):
        raise ReviewPackError("review_pack.judgments_template must be an object")
    if template.get("schema_version") != SCHEMA_JUDGMENTS:
        raise ReviewPackError(
            f"review_pack.judgments_template.schema_version must be {SCHEMA_JUDGMENTS!r}"
        )
    if template.get("run_id") != run_id:
        raise ReviewPackError("review pack and judgments template run IDs do not match")
    rubric_version = _nonempty_string(
        template.get("rubric_version"), "review_pack.judgments_template.rubric_version"
    )
    teacher_template = _teacher(template.get("teacher"), "review_pack.judgments_template.teacher")
    if not isinstance(template.get("grades"), list) or not isinstance(
        template.get("preferences"), list
    ):
        raise ReviewPackError("review pack judgments template needs grade and preference arrays")
    rubric = raw.get("rubric")
    if isinstance(rubric, Mapping) and rubric.get("version") not in {None, rubric_version}:
        raise ReviewPackError("review pack rubric and judgments template versions do not match")

    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list):
        raise ReviewPackError("review_pack.groups must be an array")
    groups: list[dict[str, Any]] = []
    prompt_keys: set[str] = set()
    candidate_order: list[str] = []
    candidate_to_prompt: dict[str, str] = {}
    for group_index, group in enumerate(groups_raw):
        label = f"review_pack.groups[{group_index}]"
        if not isinstance(group, Mapping):
            raise ReviewPackError(f"{label} must be an object")
        prompt_key = _nonempty_string(group.get("prompt_key"), f"{label}.prompt_key")
        if prompt_key in prompt_keys:
            raise ReviewPackError(f"duplicate prompt key {prompt_key!r} in review pack")
        prompt_keys.add(prompt_key)
        candidates = group.get("candidates")
        if not isinstance(candidates, list):
            raise ReviewPackError(f"{label}.candidates must be an array")
        for candidate_index, candidate in enumerate(candidates):
            candidate_label = f"{label}.candidates[{candidate_index}]"
            if not isinstance(candidate, Mapping):
                raise ReviewPackError(f"{candidate_label} must be an object")
            candidate_key = _nonempty_string(
                candidate.get("candidate_key"), f"{candidate_label}.candidate_key"
            )
            if candidate_key in candidate_to_prompt:
                raise ReviewPackError(f"duplicate candidate key {candidate_key!r} in review pack")
            candidate_order.append(candidate_key)
            candidate_to_prompt[candidate_key] = prompt_key
        groups.append(deepcopy(dict(group)))

    return ReviewPack(
        payload=deepcopy(raw),
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        rubric_version=rubric_version,
        teacher_template=teacher_template,
        groups=tuple(groups),
        candidate_order=tuple(candidate_order),
        candidate_to_prompt=candidate_to_prompt,
    )


def _review_identity(review: ReviewPack) -> dict[str, object]:
    return {
        "review_schema_version": SCHEMA_REVIEW,
        "judgments_schema_version": SCHEMA_JUDGMENTS,
        "run_id": review.run_id,
        "manifest_sha256": review.manifest_sha256,
        "rubric_version": review.rubric_version,
        "rubric_sha256": _fingerprint(review.payload.get("rubric")),
        "teacher_template": deepcopy(review.teacher_template),
    }


def _shard_assignments(review: ReviewPack, shard_count: int) -> list[list[int]]:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ReviewPackError("shard count must be a positive integer")
    if shard_count > len(review.groups):
        raise ReviewPackError("shard count cannot exceed the number of prompt groups")
    assignments: list[list[int]] = [[] for _ in range(shard_count)]
    candidate_loads = [0] * shard_count
    ranked_groups = sorted(
        range(len(review.groups)),
        key=lambda index: (
            -len(review.groups[index]["candidates"]),
            hashlib.sha256(
                (
                    f"{review.manifest_sha256}|review-shard|{review.groups[index]['prompt_key']}"
                ).encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )
    for group_index in ranked_groups:
        target = min(
            range(shard_count),
            key=lambda index: (candidate_loads[index], len(assignments[index]), index),
        )
        assignments[target].append(group_index)
        candidate_loads[target] += len(review.groups[group_index]["candidates"])
    for assignment in assignments:
        assignment.sort()
    return assignments


def shard_review_pack(
    review_pack_path: Path,
    output_dir: Path,
    shard_count: int,
    *,
    index_path: Path | None = None,
) -> dict[str, object]:
    """Write deterministic whole-group shards and an integrity index."""

    source = review_pack_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    review = load_review_pack(source)
    source_sha256 = _file_fingerprint(source)
    assignments = _shard_assignments(review, shard_count)
    width = max(3, len(str(shard_count)))
    resolved_index = (
        index_path.expanduser().resolve()
        if index_path is not None
        else destination / f"{source.stem}.shards.json"
    )
    planned_shard_paths = [
        destination / f"{source.stem}.shard-{index:0{width}d}-of-{shard_count:0{width}d}.json"
        for index in range(1, shard_count + 1)
    ]
    if resolved_index in planned_shard_paths or resolved_index == source:
        raise ReviewPackError("shard index path collides with an input or shard path")
    shard_records: list[dict[str, object]] = []

    for index, group_indices in enumerate(assignments, start=1):
        groups = [deepcopy(review.groups[group_index]) for group_index in group_indices]
        candidate_count = sum(len(group["candidates"]) for group in groups)
        name = f"{source.stem}.shard-{index:0{width}d}-of-{shard_count:0{width}d}.json"
        shard_path = destination / name
        payload = deepcopy(review.payload)
        payload["groups"] = groups
        payload["shard"] = {
            "schema_version": SCHEMA_SHARD,
            "source_review_pack_sha256": source_sha256,
            "index": index,
            "count": shard_count,
            "group_count": len(groups),
            "candidate_count": candidate_count,
        }
        _atomic_write_json(shard_path, payload)
        shard_records.append(
            {
                "file": name,
                "sha256": _file_fingerprint(shard_path),
                "group_count": len(groups),
                "candidate_count": candidate_count,
            }
        )

    index_payload = {
        "schema_version": SCHEMA_SHARD_INDEX,
        "source_review_pack": source.name,
        "source_review_pack_sha256": source_sha256,
        "identity": _review_identity(review),
        "shard_count": shard_count,
        "group_count": len(review.groups),
        "candidate_count": len(review.candidate_order),
        "shards": shard_records,
    }
    _atomic_write_json(resolved_index, index_payload)
    return {
        "operation": "shard",
        "source_review_pack": str(source),
        "source_review_pack_sha256": source_sha256,
        "index": str(resolved_index),
        "index_sha256": _file_fingerprint(resolved_index),
        "shard_count": shard_count,
        "group_count": len(review.groups),
        "candidate_count": len(review.candidate_order),
        "shards": shard_records,
    }


def _finite_number(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewPackError(f"{label} must be a finite number")
    if not math.isfinite(float(value)):
        raise ReviewPackError(f"{label} must be a finite number")


def _validate_grade(grade: object, label: str) -> str:
    allowed = {
        "candidate_key",
        "scores",
        "confidence",
        "critique",
        "corrected_answer",
        "metadata",
    }
    if not isinstance(grade, Mapping):
        raise ReviewPackError(f"{label} must be an object")
    unknown = set(grade).difference(allowed)
    if unknown:
        raise ReviewPackError(f"{label} has unknown fields: {sorted(unknown)}")
    candidate_key = _nonempty_string(grade.get("candidate_key"), f"{label}.candidate_key")
    scores = grade.get("scores")
    if not isinstance(scores, Mapping) or "overall" not in scores:
        raise ReviewPackError(f"{label}.scores must contain overall")
    unknown_scores = set(scores).difference(RUBRIC_DIMENSIONS | {"overall"})
    if unknown_scores:
        raise ReviewPackError(f"{label}.scores has unknown dimensions: {sorted(unknown_scores)}")
    for dimension, score in scores.items():
        _finite_number(score, f"{label}.scores.{dimension}")
        if not -2 <= float(score) <= 2:
            raise ReviewPackError(f"{label}.scores.{dimension} must be from -2 to 2")
    confidence = grade.get("confidence")
    _finite_number(confidence, f"{label}.confidence")
    if not 0 <= float(confidence) <= 1:
        raise ReviewPackError(f"{label}.confidence must be from zero to one")
    _nonempty_string(grade.get("critique"), f"{label}.critique")
    corrected = grade.get("corrected_answer")
    if corrected is not None:
        _nonempty_string(corrected, f"{label}.corrected_answer")
    metadata = grade.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ReviewPackError(f"{label}.metadata must be an object")
    return candidate_key


def _validate_preference(preference: object, label: str) -> tuple[str, str]:
    allowed = {
        "chosen_candidate_key",
        "rejected_candidate_key",
        "confidence",
        "rationale",
        "metadata",
    }
    if not isinstance(preference, Mapping):
        raise ReviewPackError(f"{label} must be an object")
    unknown = set(preference).difference(allowed)
    if unknown:
        raise ReviewPackError(f"{label} has unknown fields: {sorted(unknown)}")
    chosen = _nonempty_string(
        preference.get("chosen_candidate_key"), f"{label}.chosen_candidate_key"
    )
    rejected = _nonempty_string(
        preference.get("rejected_candidate_key"), f"{label}.rejected_candidate_key"
    )
    if chosen == rejected:
        raise ReviewPackError(f"{label} cannot compare a candidate with itself")
    confidence = preference.get("confidence")
    _finite_number(confidence, f"{label}.confidence")
    if not 0 <= float(confidence) <= 1:
        raise ReviewPackError(f"{label}.confidence must be from zero to one")
    _nonempty_string(preference.get("rationale"), f"{label}.rationale")
    metadata = preference.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ReviewPackError(f"{label}.metadata must be an object")
    return chosen, rejected


def merge_judgments(
    review_pack_path: Path,
    judgment_paths: Sequence[Path],
    output_path: Path,
    *,
    require_complete: bool = False,
) -> dict[str, object]:
    """Merge judgment shards after validating candidate and experiment identities."""

    source = review_pack_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    inputs = [path.expanduser().resolve() for path in judgment_paths]
    if not inputs:
        raise ReviewPackError("at least one judgments file is required")
    if len(inputs) != len(set(inputs)):
        raise ReviewPackError("judgments input paths must not contain duplicates")
    if output == source or output in inputs:
        raise ReviewPackError("merged output must not overwrite the review pack or an input")

    review = load_review_pack(source)
    expected = set(review.candidate_order)
    candidate_positions = {
        candidate_key: index for index, candidate_key in enumerate(review.candidate_order)
    }
    merged_grades: dict[str, dict[str, Any]] = {}
    merged_preferences: list[dict[str, Any]] = []
    preference_pairs: set[tuple[str, str]] = set()
    teacher_identity: dict[str, str] | None = None
    input_records: list[dict[str, object]] = []

    for path in inputs:
        raw = _load_object(path, "judgments")
        allowed = {
            "schema_version",
            "run_id",
            "rubric_version",
            "teacher",
            "grades",
            "preferences",
        }
        unknown = set(raw).difference(allowed)
        if unknown:
            raise ReviewPackError(f"judgments {path} has unknown fields: {sorted(unknown)}")
        if raw.get("schema_version") != SCHEMA_JUDGMENTS:
            raise ReviewPackError(f"judgments {path} has the wrong schema version")
        if raw.get("run_id") != review.run_id:
            raise ReviewPackError(f"judgments {path} run ID does not match the review pack")
        if raw.get("rubric_version") != review.rubric_version:
            raise ReviewPackError(f"judgments {path} rubric version does not match the review pack")
        teacher = _teacher(raw.get("teacher"), f"judgments {path}.teacher")
        if teacher_identity is None:
            teacher_identity = teacher
        elif teacher != teacher_identity:
            raise ReviewPackError("all judgments files must have the same teacher identity")
        grades = raw.get("grades")
        preferences = raw.get("preferences")
        if not isinstance(grades, list) or not isinstance(preferences, list):
            raise ReviewPackError(f"judgments {path} needs grade and preference arrays")

        for index, grade in enumerate(grades):
            label = f"judgments {path}.grades[{index}]"
            candidate_key = _validate_grade(grade, label)
            if candidate_key not in expected:
                raise ReviewPackError(f"{label} has unknown candidate key {candidate_key!r}")
            if candidate_key in merged_grades:
                raise ReviewPackError(f"duplicate grade for candidate key {candidate_key!r}")
            merged_grades[candidate_key] = deepcopy(dict(grade))

        for index, preference in enumerate(preferences):
            label = f"judgments {path}.preferences[{index}]"
            chosen, rejected = _validate_preference(preference, label)
            unknown_keys = [key for key in (chosen, rejected) if key not in expected]
            if unknown_keys:
                raise ReviewPackError(f"{label} has unknown candidate keys: {unknown_keys}")
            if review.candidate_to_prompt[chosen] != review.candidate_to_prompt[rejected]:
                raise ReviewPackError(f"{label} compares candidates from different prompt groups")
            pair = tuple(sorted((chosen, rejected)))
            if pair in preference_pairs:
                raise ReviewPackError(f"duplicate preference for candidate pair {pair}")
            preference_pairs.add(pair)
            merged_preferences.append(deepcopy(dict(preference)))

        input_records.append(
            {
                "file": str(path),
                "sha256": _file_fingerprint(path),
                "grade_count": len(grades),
                "preference_count": len(preferences),
            }
        )

    missing = [key for key in review.candidate_order if key not in merged_grades]
    if require_complete and missing:
        raise ReviewPackError(
            "merged judgments are missing "
            f"{len(missing)} of {len(review.candidate_order)} candidate grades "
            f"(missing_keys_sha256={_fingerprint(missing)})"
        )
    assert teacher_identity is not None
    grades_in_order = [merged_grades[key] for key in review.candidate_order if key in merged_grades]
    merged_preferences.sort(
        key=lambda item: (
            candidate_positions[str(item["chosen_candidate_key"])],
            candidate_positions[str(item["rejected_candidate_key"])],
            _canonical_json(item),
        )
    )
    payload = {
        "schema_version": SCHEMA_JUDGMENTS,
        "run_id": review.run_id,
        "rubric_version": review.rubric_version,
        "teacher": teacher_identity,
        "grades": grades_in_order,
        "preferences": merged_preferences,
    }
    _atomic_write_json(output, payload)
    input_records.sort(key=lambda item: (str(item["sha256"]), str(item["file"])))
    return {
        "operation": "merge",
        "review_pack": str(source),
        "review_pack_sha256": _file_fingerprint(source),
        "output": str(output),
        "output_sha256": _file_fingerprint(output),
        "input_count": len(inputs),
        "inputs": input_records,
        "expected_candidate_count": len(review.candidate_order),
        "grade_count": len(grades_in_order),
        "preference_count": len(merged_preferences),
        "missing_grade_count": len(missing),
        "missing_candidate_keys_sha256": _fingerprint(missing),
        "complete": not missing,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shard blind review packs and safely merge external judgments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard_parser = subparsers.add_parser("shard", help="split a review pack by prompt group")
    shard_parser.add_argument("--review-pack", type=Path, required=True)
    shard_parser.add_argument("--output-dir", type=Path, required=True)
    shard_parser.add_argument("--shards", type=int, required=True)
    shard_parser.add_argument("--index", type=Path)

    merge_parser = subparsers.add_parser("merge", help="merge blind-grader judgment files")
    merge_parser.add_argument("--review-pack", type=Path, required=True)
    merge_parser.add_argument("--judgments", type=Path, action="append", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "shard":
            summary = shard_review_pack(
                args.review_pack,
                args.output_dir,
                args.shards,
                index_path=args.index,
            )
        else:
            summary = merge_judgments(
                args.review_pack,
                args.judgments,
                args.output,
                require_complete=args.require_complete,
            )
        print(json.dumps(summary, sort_keys=True))
        return 0
    except ReviewPackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
