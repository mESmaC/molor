"""Build the fresh, quarantined Taskmaster-4 coffee API v2 curriculum.

Unlike v1, v2 consumes only the pinned full-dialog ``TM-4-2024/data``
annotations.  It never reads reward examples as training data.  The builder
hard-fails on any conversation overlap with every v1 reward-source provenance
sidecar, quarantines an entire full dialog when any embedded API annotation is
ambiguous or malformed, and assigns connected conversation components to one
deterministic split.  The only tool catalog placed in prompts is learned from
the resulting training split.

Generated files live under the gitignored ``data`` directory.  This builder,
the source manifest, recipe, promotion policy, v1 provenance descriptors, and
the generated build report form the content-addressed reproduction contract.
The script builds data only; it does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
DATASET_ID = "taskmaster-coffee-api-v2"
DATASET_SCHEMA = "molor.taskmaster-coffee-api-dataset.v2"
PROVENANCE_SCHEMA = "molor.taskmaster-coffee-api-provenance.v2"
EVALUATION_SCHEMA = "molor.taskmaster-coffee-api-evaluation.v2"
QUARANTINE_SCHEMA = "molor.taskmaster-coffee-api-quarantine.v2"
REPORT_SCHEMA = "molor.taskmaster-coffee-api-build-report.v2"
TOOL_CATALOG_SCHEMA = "molor.taskmaster-coffee-api-tool-catalog.v2"
SOURCE_MANIFEST = HERE / "taskmaster-coffee-api-v2-source.json"
RECIPE = HERE / "taskmaster-coffee-api-v2-recipe.json"
POLICY = HERE / "taskmaster-coffee-api-v2-evaluation-policy.json"
DEFAULT_SOURCE = HERE / "sources" / "taskmaster"
DEFAULT_OUTPUT = HERE / "data"
EXPECTED_RECIPE_OUTPUT_DIR = "../../checkpoints/qwen3-1.7b-candidate-experts-quarantine"
DEFAULT_V1_PROVENANCE = tuple(
    DEFAULT_OUTPUT / f"taskmaster-coffee-api-v1.{split}.provenance.jsonl"
    for split in ("train", "validation", "test")
)
SPLITS = ("train", "validation", "test")
_API_CONTEXT = re.compile(r"^api_(call|response)_([0-9]+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceDialogError(ValueError):
    """A full dialog cannot safely become canonical examples."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SourceLocator:
    source_file: str
    source_row_index: int
    conversation_id: str
    source_record_sha256: str
    utterance_index: int
    api_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_index": self.api_index,
            "conversation_id": self.conversation_id,
            "source_file": self.source_file,
            "source_record_sha256": self.source_record_sha256,
            "source_row_index": self.source_row_index,
            "utterance_index": self.utterance_index,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    context: list[dict[str, Any]]
    generation: dict[str, Any]
    context_canonical: str
    target_canonical: str
    locator: SourceLocator

    @property
    def context_sha256(self) -> str:
        return sha256_text(self.context_canonical)

    @property
    def target_sha256(self) -> str:
        return sha256_text(self.target_canonical)


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    group_id: str
    split: str
    context: list[dict[str, Any]]
    generation: dict[str, Any]
    context_sha256: str
    target_sha256: str
    source_records: tuple[SourceLocator, ...]


@dataclass(frozen=True, slots=True)
class DialogResult:
    candidates: tuple[Candidate, ...]
    raw_api_calls: int
    implicit_empty_show_menu_requests: int


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # The lexical root makes the data structure deterministic too, even
        # though only sorted component memberships are serialized.
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def canonical_json(value: Any) -> str:
    """Return the sole JSON representation used in prompts and identities."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SourceDialogError("non_canonical_json_value", str(exc)) from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def strict_json_loads(value: str, *, reason: str = "invalid_json") -> Any:
    if not isinstance(value, str):
        raise SourceDialogError(reason, "JSON input must be a string")
    try:
        return json.loads(
            value,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SourceDialogError(reason, str(exc)) from exc


def _read_json_document(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"could not read JSON document {path}: {exc}") from exc


def _input_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required build input is missing: {path}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _git_output(source: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not verify pinned Taskmaster checkout: {exc}") from exc
    return result.stdout.strip()


def load_source_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json_document(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != "molor.source-pin.v1":
        raise ValueError("unsupported Taskmaster source manifest")
    required = {
        "schema_version",
        "source_id",
        "repository_url",
        "resolved_commit",
        "release_path",
        "collection_method",
        "license",
        "attribution",
        "paper",
        "files",
    }
    if set(payload) != required:
        raise ValueError("Taskmaster source manifest fields do not match the v2 contract")
    release_path = payload.get("release_path")
    files = payload.get("files")
    if not isinstance(release_path, str) or not release_path:
        raise ValueError("source manifest release_path must be a non-empty string")
    if not isinstance(files, list) or not files:
        raise ValueError("source manifest files must be a non-empty array")
    data_prefix = f"{release_path}/data/"
    readme = f"{release_path}/README.md"
    data_files = 0
    seen_paths: set[str] = set()
    for descriptor in files:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("source manifest file descriptor must contain path and sha256")
        relative = descriptor["path"]
        expected_hash = descriptor["sha256"]
        if not isinstance(relative, str) or relative in seen_paths:
            raise ValueError("source manifest paths must be unique strings")
        seen_paths.add(relative)
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"invalid source SHA-256 for {relative!r}")
        if relative == readme:
            continue
        if not relative.startswith(data_prefix) or not re.fullmatch(
            r"data_[0-9]+\.json", relative.removeprefix(data_prefix)
        ):
            raise ValueError(
                "v2 source manifest may contain only the release README and full-dialog data files"
            )
        data_files += 1
    if data_files == 0:
        raise ValueError("v2 source manifest contains no full-dialog data files")
    if any("/reward/" in relative for relative in seen_paths):
        raise ValueError("v2 source manifest must not contain Taskmaster reward files")
    return payload


def verify_source(
    source: Path,
    manifest: dict[str, Any],
    *,
    verify_git: bool,
) -> list[dict[str, Any]]:
    if verify_git:
        expected_data_files = {
            f"{manifest['release_path']}/data/data_{index:02d}.json" for index in range(8)
        }
        actual_data_files = set(_data_files(manifest))
        if actual_data_files != expected_data_files:
            raise RuntimeError(
                "the pinned Taskmaster-4 build requires exactly data_00.json through data_07.json"
            )
        expected_commit = manifest["resolved_commit"]
        actual_commit = _git_output(source, "rev-parse", "HEAD")
        if actual_commit != expected_commit:
            raise RuntimeError(
                f"Taskmaster commit mismatch: expected {expected_commit}, found {actual_commit}"
            )
        dirty = _git_output(source, "status", "--porcelain", "--", manifest["release_path"])
        if dirty:
            raise RuntimeError("pinned Taskmaster-4 source has uncommitted changes")

    verified: list[dict[str, Any]] = []
    for descriptor in sorted(manifest["files"], key=lambda item: item["path"]):
        relative = descriptor["path"]
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned source file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != descriptor["sha256"]:
            raise RuntimeError(
                f"pinned source hash mismatch for {relative}: "
                f"expected {descriptor['sha256']}, found {actual_hash}"
            )
        verified.append(
            {"path": relative, "sha256": actual_hash, "size_bytes": path.stat().st_size}
        )
    return verified


def _validate_recipe_and_policy(
    recipe_path: Path,
    policy_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe = _read_json_document(recipe_path)
    policy = _read_json_document(policy_path)
    if not isinstance(recipe, dict):
        raise ValueError("unsupported v2 Taskmaster training recipe")
    experts = recipe.get("experts")
    if not isinstance(experts, list) or len(experts) != 1 or not isinstance(experts[0], dict):
        raise ValueError("v2 recipe must define exactly one candidate expert")
    expected_dataset = f"data/{DATASET_ID}.train.jsonl"
    if experts[0].get("dataset") != expected_dataset:
        raise ValueError("v2 recipe does not consume the deterministic v2 training file")
    if recipe.get("output_dir") != EXPECTED_RECIPE_OUTPUT_DIR:
        raise ValueError("v2 recipe output directory must be the approved quarantine root")

    if not isinstance(policy, dict) or policy.get("schema_version") != (
        "molor.expert-promotion-policy.v2"
    ):
        raise ValueError("unsupported v2 Taskmaster promotion policy")
    if policy.get("candidate") != experts[0].get("name"):
        raise ValueError("recipe and promotion policy candidate identities disagree")
    dataset_contract = policy.get("dataset_contract")
    if not isinstance(dataset_contract, dict) or dataset_contract.get("dataset_id") != DATASET_ID:
        raise ValueError("promotion policy does not bind the v2 dataset")
    if dataset_contract.get("v1_reward_source_conversation_overlap_required") != 0:
        raise ValueError("promotion policy must require zero v1 conversation overlap")
    if dataset_contract.get("tool_catalog_source") != "train split only":
        raise ValueError("promotion policy must require a train-only tool catalog")
    identity_inputs = dataset_contract.get("artifact_identity_inputs")
    if not isinstance(identity_inputs, dict):
        raise ValueError("promotion policy artifact input identities are missing")
    for identity_name in (
        "builder_sha256",
        "source_manifest_sha256",
        "recipe_sha256",
    ):
        if not _SHA256.fullmatch(str(identity_inputs.get(identity_name))):
            raise ValueError(f"promotion policy has an invalid {identity_name}")
    if "quarantine" not in str(policy.get("candidate_posture", "")).lower():
        raise ValueError("promotion policy must keep the candidate quarantined")
    selection = policy.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("promotion policy selection contract is missing")
    if selection.get("seed") != recipe.get("seed") or selection.get("epochs") != recipe.get(
        "epochs"
    ):
        raise ValueError("recipe and policy seed/epoch identities disagree")
    artifact_contract = policy.get("training_artifact_contract")
    if not isinstance(artifact_contract, dict):
        raise ValueError("training artifact contract is missing")
    base = artifact_contract.get("base_artifact")
    if not isinstance(base, dict) or not _SHA256.fullmatch(str(base.get("content_fingerprint"))):
        raise ValueError("base artifact content fingerprint is missing or invalid")
    required_identities = artifact_contract.get("required_file_identities")
    required = {
        "builder_sha256",
        "source_manifest_sha256",
        "recipe_sha256",
        "policy_sha256",
        "build_report_sha256",
        "train_dataset_sha256",
        "train_provenance_sha256",
        "tool_catalog_sha256",
        "trainer_source_tree_sha256",
    }
    if not isinstance(required_identities, list) or not required.issubset(required_identities):
        raise ValueError("training receipt identity requirements are incomplete")
    trainer_source = artifact_contract.get("trainer_source_contract")
    if not isinstance(trainer_source, dict) or not _SHA256.fullmatch(
        str(trainer_source.get("content_fingerprint"))
    ):
        raise ValueError("trainer source contract is missing or malformed")
    dependencies = artifact_contract.get("required_dependency_versions")
    if not isinstance(dependencies, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in dependencies.items()
    ):
        raise ValueError("dependency version contract is missing or malformed")
    return recipe, policy


def _load_v1_conversation_ids(
    paths: Sequence[Path],
) -> tuple[set[str], list[dict[str, Any]]]:
    expected_splits = {
        f"taskmaster-coffee-api-v1.{split}.provenance.jsonl": split for split in SPLITS
    }
    supplied_names = [Path(path).name for path in paths]
    if len(supplied_names) != len(expected_splits) or set(supplied_names) != set(expected_splits):
        raise ValueError(
            "v1 overlap audit requires exactly the train, validation, and test "
            "taskmaster-coffee-api-v1 provenance sidecars"
        )
    conversation_ids: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    for path in sorted((Path(path) for path in paths), key=lambda item: item.name):
        expected_split = expected_splits[path.name]
        if not path.is_file():
            raise FileNotFoundError(f"required v1 provenance sidecar is missing: {path}")
        rows = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank v1 provenance row at {path.name}:{line_number}")
                try:
                    payload = strict_json_loads(line)
                except SourceDialogError as exc:
                    raise ValueError(
                        f"invalid v1 provenance JSON at {path.name}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("source_records"), list
                ):
                    raise ValueError(
                        f"v1 provenance row lacks source_records at {path.name}:{line_number}"
                    )
                if payload.get("schema_version") != ("molor.taskmaster-coffee-api-provenance.v1"):
                    raise ValueError(f"wrong v1 provenance schema at {path.name}:{line_number}")
                if payload.get("split") != expected_split:
                    raise ValueError(
                        f"v1 provenance split does not match {path.name}:{line_number}"
                    )
                if payload.get("source_commit") != ("d92cb6af3005f1dc09c39e75e7daf4a04905e00b"):
                    raise ValueError(f"wrong v1 Taskmaster commit at {path.name}:{line_number}")
                if not payload["source_records"]:
                    raise ValueError(
                        f"v1 provenance source_records is empty at {path.name}:{line_number}"
                    )
                for record in payload["source_records"]:
                    if not isinstance(record, dict):
                        raise ValueError("v1 provenance source record is not an object")
                    conversation_id = record.get("conversation_id")
                    if not isinstance(conversation_id, str) or not conversation_id.strip():
                        raise ValueError("v1 provenance conversation_id is missing")
                    source_file = record.get("source_file")
                    if (
                        not isinstance(source_file, str)
                        or re.fullmatch(r"TM-4-2024/reward/reward_[0-9]+\.json", source_file)
                        is None
                    ):
                        raise ValueError("v1 provenance source file is not a reward JSON file")
                    source_record_sha256 = record.get("source_record_sha256")
                    if not isinstance(source_record_sha256, str) or not _SHA256.fullmatch(
                        source_record_sha256
                    ):
                        raise ValueError("v1 provenance source record hash is invalid")
                    source_row_index = record.get("source_row_index")
                    if (
                        not isinstance(source_row_index, int)
                        or isinstance(source_row_index, bool)
                        or source_row_index < 0
                    ):
                        raise ValueError("v1 provenance source row index is invalid")
                    conversation_ids.add(conversation_id)
                rows += 1
        descriptor = _input_descriptor(path)
        descriptor["rows"] = rows
        descriptors.append(descriptor)
    return conversation_ids, descriptors


def _raw_api_call_count(row: Any) -> int:
    if not isinstance(row, dict) or not isinstance(row.get("utterances"), list):
        return 0
    return sum(
        1
        for utterance in row["utterances"]
        if isinstance(utterance, dict) and isinstance(utterance.get("annotations"), list)
        for annotation in utterance["annotations"]
        if isinstance(annotation, dict) and annotation.get("name") == "api_call"
    )


def _raw_missing_show_menu_request_count(row: Any) -> int:
    """Count observable missing show_menu requests without repairing the row."""

    if not isinstance(row, dict) or not isinstance(row.get("utterances"), list):
        return 0
    count = 0
    for utterance in row["utterances"]:
        if not isinstance(utterance, dict) or not isinstance(utterance.get("annotations"), list):
            continue
        by_context: dict[str, dict[str, Any]] = defaultdict(dict)
        for annotation in utterance["annotations"]:
            if not isinstance(annotation, dict):
                continue
            context = annotation.get("context")
            name = annotation.get("name")
            if isinstance(context, str) and name in {"api_call", "request"}:
                by_context[context][str(name)] = annotation.get("value")
        count += sum(
            1
            for values in by_context.values()
            if values.get("api_call") == "show_menu" and "request" not in values
        )
    return count


def _raw_invalid_embedded_json_count(row: Any) -> int:
    """Count request/response fields rejected by the strict embedded parser."""

    if not isinstance(row, dict) or not isinstance(row.get("utterances"), list):
        return 0
    count = 0
    for utterance in row["utterances"]:
        if not isinstance(utterance, dict) or not isinstance(utterance.get("annotations"), list):
            continue
        for annotation in utterance["annotations"]:
            if not isinstance(annotation, dict) or annotation.get("name") not in {
                "request",
                "response",
            }:
                continue
            value = annotation.get("value")
            if not isinstance(value, str):
                count += 1
                continue
            try:
                parsed = strict_json_loads(value, reason="invalid_embedded_json")
            except SourceDialogError:
                count += 1
                continue
            if not isinstance(parsed, dict):
                count += 1
    return count


def _validate_api_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceDialogError("invalid_api_name", f"{field} must be a non-empty trimmed string")
    return value


def _embedded_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise SourceDialogError(
            "embedded_json_not_string", f"{field} annotation value must be a JSON string"
        )
    parsed = strict_json_loads(value, reason="invalid_embedded_json")
    if not isinstance(parsed, dict):
        raise SourceDialogError(
            "embedded_json_not_object", f"{field} annotation must decode to a JSON object"
        )
    canonical_json(parsed)
    return parsed


def _annotation_actions(utterance: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = utterance["annotations"]
    if not annotations:
        return []
    if utterance["speaker"] != "user":
        raise SourceDialogError(
            "api_annotation_on_non_user_turn", "API annotations may occur only on user turns"
        )
    grouped: dict[int, dict[str, Any]] = defaultdict(dict)
    for annotation in annotations:
        if not isinstance(annotation, dict) or set(annotation) != {"name", "value", "context"}:
            raise SourceDialogError(
                "invalid_annotation_schema", "annotation fields do not match Taskmaster-4"
            )
        name = annotation["name"]
        value = annotation["value"]
        context = annotation["context"]
        if name not in {"api_call", "request", "api_response", "response"}:
            raise SourceDialogError("unsupported_annotation", f"unsupported annotation {name!r}")
        if not isinstance(context, str):
            raise SourceDialogError(
                "invalid_annotation_context", "annotation context is not a string"
            )
        match = _API_CONTEXT.fullmatch(context)
        if match is None:
            raise SourceDialogError(
                "invalid_annotation_context", f"unsupported annotation context {context!r}"
            )
        kind, index_text = match.groups()
        expected_kind = "call" if name in {"api_call", "request"} else "response"
        if kind != expected_kind:
            raise SourceDialogError(
                "mismatched_annotation_context",
                f"{name!r} cannot use {context!r}",
            )
        api_index = int(index_text)
        if name in grouped[api_index]:
            raise SourceDialogError(
                "duplicate_api_annotation", f"API {api_index} repeats annotation {name!r}"
            )
        grouped[api_index][name] = value

    actions: list[dict[str, Any]] = []
    for api_index in sorted(grouped):
        values = grouped[api_index]
        required = {"api_call", "api_response", "response"}
        if not required.issubset(values) or set(values) - (required | {"request"}):
            raise SourceDialogError(
                "incomplete_api_annotation",
                f"API {api_index} lacks an unambiguous call/request/response pairing",
            )
        call_name = _validate_api_name(values["api_call"], field="api_call")
        response_name = _validate_api_name(values["api_response"], field="api_response")
        if call_name != response_name:
            raise SourceDialogError(
                "api_call_response_name_mismatch",
                f"API {api_index} calls {call_name!r} but responds as {response_name!r}",
            )
        implicit_empty_request = "request" not in values
        if implicit_empty_request:
            if call_name != "show_menu":
                raise SourceDialogError(
                    "missing_api_request",
                    f"API {api_index} {call_name!r} has no request annotation",
                )
            request: dict[str, Any] = {}
        else:
            request = _embedded_object(values["request"], field="request")
        response = _embedded_object(values["response"], field="response")
        actions.append(
            {
                "api_index": api_index,
                "api_name": call_name,
                "request": request,
                "response": response,
                "implicit_empty_request": implicit_empty_request,
            }
        )
    return actions


def _parse_dialog(row: Any, *, source_file: str, source_row_index: int) -> DialogResult:
    if not isinstance(row, dict):
        raise SourceDialogError("source_dialog_not_object", "source dialog must be an object")
    expected_keys = {"conversation_id", "instructions", "scenario", "utterances", "vertical"}
    if set(row) != expected_keys:
        raise SourceDialogError(
            "invalid_source_dialog_schema", "source dialog fields do not match Taskmaster-4"
        )
    conversation_id = row["conversation_id"]
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise SourceDialogError(
            "invalid_conversation_id", "conversation_id must be a non-empty string"
        )
    if row["vertical"] != "Coffee":
        raise SourceDialogError("unexpected_vertical", "v2 accepts only the Coffee vertical")
    if not isinstance(row["scenario"], str) or not row["scenario"].strip():
        raise SourceDialogError("invalid_scenario", "scenario must be non-empty text")
    if row["instructions"] is not None and not isinstance(row["instructions"], str):
        raise SourceDialogError("invalid_instructions", "instructions must be text or null")
    utterances = row["utterances"]
    if not isinstance(utterances, list) or not utterances:
        raise SourceDialogError("dialog_without_utterances", "utterances must be non-empty")

    source_record_sha256 = sha256_text(canonical_json(row))
    history: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    implicit_empty_requests = 0
    expected_api_index = 0
    for position, utterance in enumerate(utterances):
        if not isinstance(utterance, dict) or set(utterance) != {
            "annotations",
            "index",
            "segments",
            "speaker",
            "text",
        }:
            raise SourceDialogError(
                "invalid_utterance_schema", f"utterance {position} fields do not match Taskmaster-4"
            )
        if utterance["index"] != position:
            raise SourceDialogError(
                "nonsequential_utterance_index",
                f"utterance position {position} has index {utterance['index']!r}",
            )
        speaker = utterance["speaker"]
        role = {"user": "user", "assistant": "agent"}.get(speaker)
        if role is None:
            raise SourceDialogError(
                "unsupported_speaker", f"utterance {position} has speaker {speaker!r}"
            )
        text = utterance["text"]
        if not isinstance(text, str) or not text.strip():
            raise SourceDialogError(
                "empty_utterance_text", f"utterance {position} text must be non-empty"
            )
        if not isinstance(utterance["segments"], list):
            raise SourceDialogError("invalid_segments", "utterance segments must be an array")
        if not isinstance(utterance["annotations"], list):
            raise SourceDialogError("invalid_annotations", "utterance annotations must be an array")
        history.append({"role": role, "text": text})
        actions = _annotation_actions(utterance)
        for action in actions:
            api_index = action["api_index"]
            if api_index != expected_api_index:
                raise SourceDialogError(
                    "nonsequential_api_index",
                    f"expected API index {expected_api_index}, found {api_index}",
                )
            expected_api_index += 1
            generation = {
                "role": "agent",
                "api": {action["api_name"]: action["request"]},
            }
            context_canonical = canonical_json(history)
            target_canonical = canonical_json(generation)
            candidates.append(
                Candidate(
                    context=strict_json_loads(context_canonical),
                    generation=generation,
                    context_canonical=context_canonical,
                    target_canonical=target_canonical,
                    locator=SourceLocator(
                        source_file=source_file,
                        source_row_index=source_row_index,
                        conversation_id=conversation_id,
                        source_record_sha256=source_record_sha256,
                        utterance_index=position,
                        api_index=api_index,
                    ),
                )
            )
            history.append(generation)
            history.append(
                {
                    "role": "api",
                    "response": {action["api_name"]: action["response"]},
                }
            )
            if action["implicit_empty_request"]:
                implicit_empty_requests += 1
    raw_api_calls = _raw_api_call_count(row)
    if raw_api_calls != expected_api_index:
        raise SourceDialogError(
            "api_call_accounting_mismatch",
            f"found {raw_api_calls} raw api_call annotations but parsed {expected_api_index}",
        )
    return DialogResult(tuple(candidates), raw_api_calls, implicit_empty_requests)


def _data_files(manifest: dict[str, Any]) -> list[str]:
    prefix = f"{manifest['release_path']}/data/"
    return sorted(
        descriptor["path"]
        for descriptor in manifest["files"]
        if descriptor["path"].startswith(prefix)
    )


def _dialog_quarantine_row(
    *,
    row: Any,
    source_file: str,
    source_row_index: int,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": QUARANTINE_SCHEMA,
        "detail": detail,
        "discarded_api_calls": _raw_api_call_count(row),
        "disposition": "whole_dialog_quarantined",
        "reason": reason,
        "source_file": source_file,
        "source_row_index": source_row_index,
    }
    if isinstance(row, dict):
        conversation_id = row.get("conversation_id")
        if isinstance(conversation_id, str):
            result["conversation_id"] = conversation_id
        result["source_record_sha256"] = sha256_text(canonical_json(row))
    return result


def _load_candidates(
    source: Path,
    manifest: dict[str, Any],
    *,
    v1_conversation_ids: set[str],
) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, int], set[str]]:
    candidates: list[Candidate] = []
    quarantine: list[dict[str, Any]] = []
    conversation_ids: set[str] = set()
    counts = Counter()
    overlaps: set[str] = set()
    for relative in _data_files(manifest):
        payload = _read_json_document(source / relative)
        if not isinstance(payload, list):
            raise RuntimeError(f"{relative} is not a JSON array")
        for source_row_index, row in enumerate(payload):
            counts["source_dialogs"] += 1
            if isinstance(row, dict) and isinstance(row.get("utterances"), list):
                counts["source_utterances"] += len(row["utterances"])
            raw_api_calls = _raw_api_call_count(row)
            counts["source_api_calls"] += raw_api_calls
            counts["observed_missing_show_menu_requests"] += _raw_missing_show_menu_request_count(
                row
            )
            counts["observed_invalid_embedded_json_fields"] += _raw_invalid_embedded_json_count(row)
            conversation_id = row.get("conversation_id") if isinstance(row, dict) else None
            if isinstance(conversation_id, str):
                if conversation_id in conversation_ids:
                    raise RuntimeError(f"duplicate full-dialog conversation_id: {conversation_id}")
                conversation_ids.add(conversation_id)
                if conversation_id in v1_conversation_ids:
                    overlaps.add(conversation_id)
            try:
                parsed = _parse_dialog(
                    row,
                    source_file=relative,
                    source_row_index=source_row_index,
                )
            except SourceDialogError as exc:
                counts["quarantined_invalid_dialogs"] += 1
                counts["quarantined_api_calls"] += raw_api_calls
                quarantine.append(
                    _dialog_quarantine_row(
                        row=row,
                        source_file=relative,
                        source_row_index=source_row_index,
                        reason=exc.reason,
                        detail=str(exc),
                    )
                )
                continue
            if parsed.raw_api_calls == 0:
                counts["excluded_dialogs_without_api_calls"] += 1
                continue
            counts["valid_dialogs"] += 1
            counts["valid_dialog_api_calls"] += parsed.raw_api_calls
            counts["implicit_empty_show_menu_requests"] += parsed.implicit_empty_show_menu_requests
            candidates.extend(parsed.candidates)
    if overlaps:
        preview = ", ".join(sorted(overlaps)[:10])
        raise ValueError(
            f"v2 full-dialog conversations overlap v1 reward-source provenance: {preview}"
        )
    return candidates, quarantine, dict(counts), conversation_ids


def group_id_for_conversations(conversation_ids: Iterable[str]) -> str:
    members = sorted(set(conversation_ids))
    if not members or any(not isinstance(item, str) or not item for item in members):
        raise ValueError("conversation component needs non-empty string identities")
    digest = sha256_text(canonical_json(members))
    return f"tm4-coffee-v2-component:{digest}"


def split_for_group(identifier: str) -> str:
    digest = hashlib.sha256(f"{DATASET_ID}:split:{identifier}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.10:
        return "test"
    if bucket < 0.20:
        return "validation"
    return "train"


def _action_quarantine_row(
    candidate: Candidate,
    *,
    distinct_targets: int,
) -> dict[str, Any]:
    return {
        "schema_version": QUARANTINE_SCHEMA,
        "context_sha256": candidate.context_sha256,
        "detail": (
            f"canonical decision context has {distinct_targets} distinct canonical API targets"
        ),
        "disposition": "ambiguous_decision_action_quarantined",
        "reason": "ambiguous_decision_targets",
        "source_record": candidate.locator.to_dict(),
        "target_sha256": candidate.target_sha256,
    }


def _deduplicate_and_group(
    candidates: list[Candidate],
    quarantine: list[dict[str, Any]],
) -> tuple[list[Example], dict[str, int], dict[str, tuple[str, ...]]]:
    by_context: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_context[candidate.context_canonical].append(candidate)

    union_find = _UnionFind(candidate.locator.conversation_id for candidate in candidates)
    for rows in by_context.values():
        members = sorted({row.locator.conversation_id for row in rows})
        for member in members[1:]:
            union_find.union(members[0], member)
    component_members: dict[str, list[str]] = defaultdict(list)
    for conversation_id in sorted(union_find.parent):
        component_members[union_find.find(conversation_id)].append(conversation_id)
    group_by_conversation: dict[str, str] = {}
    serialized_components: dict[str, tuple[str, ...]] = {}
    for members in component_members.values():
        group_id = group_id_for_conversations(members)
        serialized_components[group_id] = tuple(sorted(members))
        for member in members:
            group_by_conversation[member] = group_id

    conflicting_contexts: set[str] = set()
    for context, rows in by_context.items():
        if len({row.target_canonical for row in rows}) > 1:
            conflicting_contexts.add(context)
            distinct_targets = len({row.target_canonical for row in rows})
            for candidate in rows:
                quarantine.append(
                    _action_quarantine_row(
                        candidate,
                        distinct_targets=distinct_targets,
                    )
                )

    by_content: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.context_canonical not in conflicting_contexts:
            by_content[(candidate.context_canonical, candidate.target_canonical)].append(candidate)

    examples: list[Example] = []
    for (context_canonical, target_canonical), rows in sorted(by_content.items()):
        representative = min(
            rows,
            key=lambda row: (
                row.locator.source_file,
                row.locator.source_row_index,
                row.locator.utterance_index,
                row.locator.api_index,
            ),
        )
        groups = {group_by_conversation[row.locator.conversation_id] for row in rows}
        if len(groups) != 1:
            raise RuntimeError("duplicate canonical decisions escaped component grouping")
        group_id = next(iter(groups))
        identifier_payload = {
            "context": representative.context,
            "generation": representative.generation,
        }
        example_id = f"tm4-coffee-api-v2:{sha256_text(canonical_json(identifier_payload))}"
        examples.append(
            Example(
                example_id=example_id,
                group_id=group_id,
                split=split_for_group(group_id),
                context=representative.context,
                generation=representative.generation,
                context_sha256=representative.context_sha256,
                target_sha256=representative.target_sha256,
                source_records=tuple(
                    sorted(
                        (row.locator for row in rows),
                        key=lambda locator: (
                            locator.source_file,
                            locator.source_row_index,
                            locator.utterance_index,
                            locator.api_index,
                        ),
                    )
                ),
            )
        )
    stats = {
        "canonical_decision_contexts": len(by_context),
        "ambiguous_decision_contexts": len(conflicting_contexts),
        "ambiguous_distinct_targets": sum(
            len({row.target_canonical for row in by_context[context]})
            for context in conflicting_contexts
        ),
        "quarantined_ambiguous_api_actions": sum(
            len(by_context[context]) for context in conflicting_contexts
        ),
        "deduplicated_api_actions": sum(len(rows) - 1 for rows in by_content.values()),
        "decision_components_before_ambiguity": len(serialized_components),
    }
    return examples, stats, dict(sorted(serialized_components.items()))


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _build_train_tool_catalog(examples: Sequence[Example]) -> dict[str, Any]:
    train = [example for example in examples if example.split == "train"]
    if not train:
        raise RuntimeError("deterministic split produced no training examples")
    key_sets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    argument_types: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for example in train:
        api_name, arguments = next(iter(example.generation["api"].items()))
        keys = tuple(sorted(arguments))
        key_sets[api_name].add(keys)
        for key, value in arguments.items():
            argument_types[api_name][key].add(_json_type(value))
    tools = []
    for api_name in sorted(key_sets):
        tools.append(
            {
                "argument_key_sets": [list(keys) for keys in sorted(key_sets[api_name])],
                "argument_types": {
                    key: sorted(types) for key, types in sorted(argument_types[api_name].items())
                },
                "name": api_name,
            }
        )
    train_api_names = {tool["name"] for tool in tools}
    heldout_api_names = {
        next(iter(example.generation["api"])) for example in examples if example.split != "train"
    }
    missing = sorted(heldout_api_names - train_api_names)
    if missing:
        raise RuntimeError(
            "held-out examples use API names absent from the train-only catalog: "
            + ", ".join(missing)
        )
    return {
        "schema_version": TOOL_CATALOG_SCHEMA,
        "dataset_id": DATASET_ID,
        "source_split": "train",
        "tools": tools,
    }


def messages_for(
    context_canonical: str,
    target_canonical: str,
    *,
    tool_catalog_canonical: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are the API planner for a coffee-ordering agent. The user supplies a "
        "Taskmaster-4 conversation context as canonical JSON. Return exactly one JSON "
        "object with role=agent and one api call. Use only this train-derived tool "
        f"catalog: {tool_catalog_canonical}. Output canonical minified JSON only: no "
        "Markdown, commentary, or thinking tags."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Taskmaster-4 conversation context:\n{context_canonical}",
        },
        {"role": "assistant", "content": target_canonical},
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row))
            stream.write("\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _output_descriptor(path: Path, rows: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        descriptor["rows"] = rows
    return descriptor


def _quarantine_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    source = row.get("source_record", row)
    return (
        str(source.get("source_file", "")),
        int(source.get("source_row_index", -1)),
        int(source.get("utterance_index", -1)),
        int(source.get("api_index", -1)),
        str(row.get("reason", "")),
    )


def build_dataset(
    source: Path,
    output: Path,
    source_manifest_path: Path = SOURCE_MANIFEST,
    recipe_path: Path = RECIPE,
    policy_path: Path = POLICY,
    v1_provenance_paths: Sequence[Path] = DEFAULT_V1_PROVENANCE,
    verify_pin: bool = True,
) -> dict[str, Any]:
    """Build deterministic v2 artifacts and return the build report."""

    source = Path(source)
    output = Path(output)
    source_manifest_path = Path(source_manifest_path)
    recipe_path = Path(recipe_path)
    policy_path = Path(policy_path)
    manifest = load_source_manifest(source_manifest_path)
    recipe, policy = _validate_recipe_and_policy(recipe_path, policy_path)
    builder_descriptor = _input_descriptor(Path(__file__).resolve())
    source_manifest_descriptor = _input_descriptor(source_manifest_path)
    recipe_descriptor = _input_descriptor(recipe_path)
    policy_descriptor = _input_descriptor(policy_path)
    actual_identity_inputs = {
        "builder_sha256": builder_descriptor["sha256"],
        "source_manifest_sha256": source_manifest_descriptor["sha256"],
        "recipe_sha256": recipe_descriptor["sha256"],
    }
    declared_identity_inputs = policy["dataset_contract"]["artifact_identity_inputs"]
    for identity_name, actual_hash in actual_identity_inputs.items():
        if declared_identity_inputs[identity_name] != actual_hash:
            raise ValueError(
                f"promotion policy {identity_name} is stale: expected {actual_hash}, "
                f"found {declared_identity_inputs[identity_name]}"
            )
    verified_files = verify_source(source, manifest, verify_git=verify_pin)
    v1_conversation_ids, v1_descriptors = _load_v1_conversation_ids(
        tuple(Path(path) for path in v1_provenance_paths)
    )
    candidates, quarantine, source_counts, source_conversation_ids = _load_candidates(
        source,
        manifest,
        v1_conversation_ids=v1_conversation_ids,
    )
    examples, dedup_stats, components = _deduplicate_and_group(candidates, quarantine)
    if not examples:
        raise RuntimeError("v2 source produced no unambiguous examples")
    quarantine.sort(key=_quarantine_sort_key)

    split_examples = {
        split: sorted(
            (example for example in examples if example.split == split),
            key=lambda example: example.example_id,
        )
        for split in SPLITS
    }
    group_owners: dict[str, str] = {}
    for split, rows in split_examples.items():
        for example in rows:
            previous = group_owners.setdefault(example.group_id, split)
            if previous != split:
                raise RuntimeError(f"component {example.group_id} leaked across splits")

    catalog = _build_train_tool_catalog(examples)
    catalog_path = output / f"{DATASET_ID}.tool-catalog.json"
    _write_json(catalog_path, catalog)
    catalog_canonical = canonical_json(catalog)
    catalog_sha256 = sha256_file(catalog_path)

    identity = {
        "builder_sha256": builder_descriptor["sha256"],
        "policy_sha256": policy_descriptor["sha256"],
        "recipe_sha256": recipe_descriptor["sha256"],
        "source_manifest_sha256": source_manifest_descriptor["sha256"],
        "tool_catalog_sha256": catalog_sha256,
    }

    outputs: dict[str, dict[str, Any]] = {
        "tool_catalog": _output_descriptor(catalog_path),
    }
    group_counts: dict[str, int] = {}
    api_counts: dict[str, dict[str, int]] = {}
    for split, rows in split_examples.items():
        training_path = output / f"{DATASET_ID}.{split}.jsonl"
        provenance_path = output / f"{DATASET_ID}.{split}.provenance.jsonl"

        materialized: list[tuple[Example, list[dict[str, str]], str]] = []
        for example in rows:
            messages = messages_for(
                canonical_json(example.context),
                canonical_json(example.generation),
                tool_catalog_canonical=catalog_canonical,
            )
            materialized.append((example, messages, sha256_text(canonical_json(messages))))
        _write_jsonl(
            training_path,
            ({"messages": messages} for _, messages, _ in materialized),
        )
        _write_jsonl(
            provenance_path,
            (
                {
                    "schema_version": PROVENANCE_SCHEMA,
                    **identity,
                    "context_sha256": example.context_sha256,
                    "example_id": example.example_id,
                    "group_id": example.group_id,
                    "license_spdx": manifest["license"]["spdx_id"],
                    "source_commit": manifest["resolved_commit"],
                    "source_records": [record.to_dict() for record in example.source_records],
                    "split": split,
                    "target_sha256": example.target_sha256,
                    "training_messages_sha256": messages_sha256,
                }
                for example, _, messages_sha256 in materialized
            ),
        )
        outputs[f"{split}_training"] = _output_descriptor(training_path, len(rows))
        outputs[f"{split}_provenance"] = _output_descriptor(provenance_path, len(rows))
        group_counts[split] = len({example.group_id for example in rows})
        api_counts[split] = dict(
            sorted(Counter(next(iter(example.generation["api"])) for example in rows).items())
        )

        if split != "train":
            evaluation_path = output / f"{DATASET_ID}.{split}.evaluation.jsonl"
            _write_jsonl(
                evaluation_path,
                (
                    {
                        "schema_version": EVALUATION_SCHEMA,
                        "example_id": example.example_id,
                        "expected": example.generation,
                        "expected_canonical": canonical_json(example.generation),
                        "expected_sha256": example.target_sha256,
                        "group_id": example.group_id,
                        "messages": messages[:-1],
                    }
                    for example, messages, _ in materialized
                ),
            )
            outputs[f"{split}_evaluation"] = _output_descriptor(evaluation_path, len(rows))

    quarantine_path = output / f"{DATASET_ID}.quarantine.jsonl"
    _write_jsonl(quarantine_path, quarantine)
    outputs["quarantine"] = _output_descriptor(quarantine_path, len(quarantine))

    output_example_count = sum(len(rows) for rows in split_examples.values())
    if output_example_count != len(examples):
        raise RuntimeError("split example accounting does not reconcile")
    if source_counts["source_dialogs"] != (
        source_counts.get("valid_dialogs", 0)
        + source_counts.get("excluded_dialogs_without_api_calls", 0)
        + source_counts.get("quarantined_invalid_dialogs", 0)
    ):
        raise RuntimeError("source dialog accounting does not reconcile")
    if source_counts["source_api_calls"] != (
        source_counts.get("valid_dialog_api_calls", 0)
        + source_counts.get("quarantined_api_calls", 0)
    ):
        raise RuntimeError("source API call accounting does not reconcile")

    source_contract = {
        "attribution": manifest["attribution"],
        "collection_method": manifest["collection_method"],
        "license": manifest["license"],
        "paper": manifest["paper"],
        "release_path": manifest["release_path"],
        "repository_url": manifest["repository_url"],
        "resolved_commit": manifest["resolved_commit"],
        "source_id": manifest["source_id"],
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "dataset_id": DATASET_ID,
        "build_inputs": {
            "builder": builder_descriptor,
            "policy": policy_descriptor,
            "recipe": recipe_descriptor,
            "source_manifest": source_manifest_descriptor,
            "v1_emitted_provenance_sidecars": v1_descriptors,
        },
        "source": source_contract,
        "verified_source_files": verified_files,
        "reproducibility_contract": {
            "base_artifact": policy["training_artifact_contract"]["base_artifact"],
            "dependency_versions": policy["training_artifact_contract"][
                "required_dependency_versions"
            ],
            "recipe_id": "taskmaster-coffee-api-v2-full-dialog-r16-single-run",
            "policy_id": policy["policy_id"],
            "canonical_json": "UTF-8, sorted object keys, minified, finite values",
            "determinism": "content-only ordering and SHA-256; no timestamps or output-root paths",
        },
        "selection": {
            "source_scope": "pinned TM-4-2024/data full-dialog annotations only",
            "reward_source_rows_consumed": 0,
            "v1_overlap_scope": policy["dataset_contract"]["v1_overlap_scope"],
            "malformed_dialog_policy": "quarantine the whole dialog; do not repair JSON",
            "missing_request_policy": "only show_menu may canonicalize a missing request to {}",
            "conflict_policy": (
                "quarantine every action whose exact canonical context has multiple targets"
            ),
            "deduplication": "exact canonical context plus target; retain every locator",
            "grouping": ("connected components of conversations sharing an exact decision context"),
            "split_policy": (
                "SHA-256(dataset-id + component-id), 80% train / 10% validation / 10% test"
            ),
            "tool_catalog": "API names, key sets, and types observed in train only",
            "trainer_row_schema": "messages is the only top-level field",
            "pinned_release_scope_note": (
                "The pinned eight data files are the complete manifest scope (3,710 rows); "
                "the upstream README says 6,500 dialogs, so this build makes no broader "
                "upstream-completeness claim."
            ),
        },
        "counts": {
            **source_counts,
            "v1_emitted_provenance_conversation_ids": len(v1_conversation_ids),
            "v1_emitted_provenance_conversation_overlap": 0,
            "candidate_api_actions_before_conflicts": len(candidates),
            **dedup_stats,
            "emitted_unique_examples": len(examples),
            "emitted_groups": len(group_owners),
            "empty_components_after_ambiguity": (len(components) - len(group_owners)),
            "component_conversations": sum(len(members) for members in components.values()),
            "examples_by_split": {split: len(split_examples[split]) for split in SPLITS},
            "groups_by_split": group_counts,
            "api_calls_by_split": api_counts,
        },
        "integrity": {
            "full_dialog_conversation_ids_unique": (
                len(source_conversation_ids) == source_counts["source_dialogs"]
            ),
            "v1_emitted_provenance_conversation_overlap_zero": True,
            "group_disjoint": True,
            "example_ids_unique": len({example.example_id for example in examples})
            == len(examples),
            "train_only_tool_catalog": True,
            "heldout_api_names_present_in_training": True,
            "source_api_call_accounting_reconciled": True,
            "source_dialog_accounting_reconciled": True,
            "consumer_verification_required": (
                "rehash every declared output and the report before training; the report "
                "is written last and a failed build does not publish a new receipt"
            ),
        },
        "outputs": dict(sorted(outputs.items())),
        "intended_checkpoint_posture": "candidate_quarantine_only_never_auto_promote",
    }
    report_path = output / f"{DATASET_ID}.build-report.json"
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the pinned, group-disjoint full-dialog Taskmaster-4 v2 curriculum."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument(
        "--v1-provenance",
        action="append",
        type=Path,
        default=None,
        help="v1 provenance sidecar; repeat for every split (defaults to all three)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_dataset(
        source=args.source.resolve(),
        output=args.output.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        recipe_path=args.recipe.resolve(),
        policy_path=args.policy.resolve(),
        v1_provenance_paths=(
            tuple(path.resolve() for path in args.v1_provenance)
            if args.v1_provenance is not None
            else DEFAULT_V1_PROVENANCE
        ),
        verify_pin=True,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
