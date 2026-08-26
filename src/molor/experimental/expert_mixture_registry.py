"""Content-addressed expert chat recipes and router examples for rack v5.

The registry deliberately separates three concepts which had previously been easy to
conflate:

* chat SFT shards admitted to ``corpus_mixtures``;
* reference/CPT documents which are not valid chat rows; and
* lineage aliases which must not be sampled a second time.

Only manifest-declared, messages-only JSONL shards are admitted.  Recipes reuse the
strict :mod:`molor.experimental.corpus_mixtures` v1 schema and protect every training
recipe with all held-out chat shards, and every evaluation recipe with all train chat
shards.  The router export is a separate strict ``{prompt, route_id}`` data plane.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from typing import Literal

from molor.experimental.corpus_mixtures import (
    CorpusMixtureRecipe,
    CorpusMixtureShard,
    CorpusShardDescriptor,
    corpus_mixture_recipe_json_bytes,
)
from molor.identity import canonical_json_bytes, fingerprint_json, validate_sha256
from molor.training.data import ChatExample


EXPERT_CHAT_REGISTRY_SCHEMA = "molor.expert-chat-shard-registry.v1"
EXPERT_CHAT_RECEIPT_SCHEMA = "molor.expert-chat-recipe-receipt.v1"
ROUTER_EXAMPLES_MANIFEST_SCHEMA = "molor.hierarchical-router-examples.v1"
SPLITS = ("train", "validation", "test")
SEMANTIC_STREAMS = ("direct", "trace", "tool")
RECIPE_STREAMS = ("direct", "trace", "tool", "tool-direct", "tool-trace")
SPLIT_PRECEDENCE = {"test": 0, "validation": 1, "train": 2}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

# A fast recipe is deliberately record-budgeted.  Exact token tiers remain separate
# shards, so weights can be adjusted without regenerating a canonical recipe.
FAST_RECORD_BUDGETS: dict[tuple[str, str], int] = {
    ("train", "direct"): 8_192,
    ("train", "trace"): 4_096,
    ("train", "tool"): 8_192,
    ("validation", "direct"): 512,
    ("validation", "trace"): 256,
    ("validation", "tool"): 512,
    ("test", "direct"): 512,
    ("test", "trace"): 256,
    ("test", "tool"): 512,
}
ROUTER_RECORD_CAPS = {"train": 4_096, "validation": 512, "test": 512}
ROUTER_MAX_PROMPT_CHARACTERS = 8_192
ROUTER_ELISION_MARKER = "\n\n[... router context elided ...]\n\n"
ROUTER_SOURCE_PRIORITIES = {
    "agentic-tool-length-tiers-v1": 0,
    "business-finance-length-tiers-v1": 0,
    "business-grounded-reconstruction-v1": 0,
    "embedded-grounded-reconstruction-v1": 0,
    "embedded-source-experts-v1": 0,
    "software-architecture-experts-v1": 1,
    "broad-distillation-experts-v1": 2,
    "qwen38-distillation-experts-v2": 3,
}

_SAFE_COMPONENT = re.compile(r"[^a-z0-9.-]+")


@dataclass(frozen=True, slots=True)
class ExpertMixtureSourcePaths:
    """Explicit allow-list of source roots; callers never discover sibling paths."""

    taxonomy: Path
    qwen38_v2: Path
    broad_v1: Path
    software_v1: Path
    business_finance_v1: Path
    business_grounded_v1: Path
    agentic_tool_v1: Path
    embedded_v1: Path
    embedded_grounded_v1: Path
    embedded_grounded_registration: Path
    business_reference_v1: Path
    router_base_report_v6: Path
    structured_tool_audit: Path

    @classmethod
    def workstation_defaults(cls, repository_root: str | Path) -> ExpertMixtureSourcePaths:
        repo = Path(repository_root).resolve()
        datasets = repo.parent / "datasets"
        processed = datasets / "processed"
        return cls(
            taxonomy=repo / "experiments" / "expert-rack-v5" / "expert-taxonomy-v1.json",
            qwen38_v2=repo / "datasets" / "qwen38-distillation-experts-v2",
            broad_v1=datasets / "broad-distillation-experts-v1",
            software_v1=processed / "molor-software-architecture-experts-v1",
            business_finance_v1=processed / "molor-business-finance-length-tiers-v1",
            business_grounded_v1=processed / "molor-business-grounded-reconstruction-v1",
            agentic_tool_v1=processed / "molor-agentic-tool-length-tiers-v1",
            embedded_v1=processed / "molor-embedded-source-experts-v1",
            embedded_grounded_v1=processed / "molor-embedded-grounded-reconstruction-v1",
            embedded_grounded_registration=(
                repo
                / "experiments"
                / "expert-rack-v5"
                / "embedded-grounded-reconstruction-v1"
                / "registration-fragment.json"
            ),
            business_reference_v1=processed / "molor-business-reference-v1",
            router_base_report_v6=(
                repo
                / "experiments"
                / "first-real-experts"
                / "data"
                / "router-bootstrap-v6-taskmaster.report.json"
            ),
            structured_tool_audit=(
                repo / "experiments" / "expert-rack-v5" / "structured-tool-audit-receipt-v1.json"
            ),
        )


@dataclass(frozen=True, slots=True)
class RegisteredChatShard:
    descriptor: CorpusShardDescriptor
    source_id: str
    source_revision: str
    semantic_stream: Literal["direct", "trace", "tool"] | str
    recipe_stream: str
    length_tier: str
    admission_role: str

    def __post_init__(self) -> None:
        if self.semantic_stream not in SEMANTIC_STREAMS:
            raise ValueError(f"unsupported semantic stream {self.semantic_stream!r}")
        if self.recipe_stream not in RECIPE_STREAMS:
            raise ValueError(f"unsupported recipe stream {self.recipe_stream!r}")
        if not self.source_id or not self.source_revision:
            raise ValueError("registered shards need source identity")
        if not self.length_tier or not self.admission_role:
            raise ValueError("registered shards need tier and admission role")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.descriptor.to_dict(),
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "semantic_stream": self.semantic_stream,
            "recipe_stream": self.recipe_stream,
            "length_tier": self.length_tier,
            "admission_role": self.admission_role,
        }


@dataclass(frozen=True, slots=True)
class RegisteredReferenceShard:
    source_id: str
    leaf_id: str
    split: str
    modality: str
    path: str
    sha256: str
    record_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "leaf_id": self.leaf_id,
            "split": self.split,
            "modality": self.modality,
            "path": self.path,
            "sha256": self.sha256,
            "record_count": self.record_count,
        }


@dataclass(frozen=True, slots=True)
class CollectedExpertData:
    taxonomy: dict[str, object]
    taxonomy_path: str
    taxonomy_sha256: str
    leaf_revision: str
    family_by_leaf: dict[str, str]
    shards: tuple[RegisteredChatShard, ...]
    references: tuple[RegisteredReferenceShard, ...]
    sources: tuple[dict[str, object], ...]
    dispositions: tuple[dict[str, object], ...]


def _strict_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON manifest must be a regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must contain one object: {path}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON numeric constant {value!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _self_hashed(payload: Mapping[str, object]) -> dict[str, object]:
    content = dict(payload)
    if "content_sha256" in content:
        raise ValueError("self-hashed payload already contains content_sha256")
    return {**content, "content_sha256": fingerprint_json(content)}


def _as_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _as_list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _as_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _as_count(value: object, field_name: str, *, allow_zero: bool = True) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def _digest(value: object, field_name: str) -> str:
    return validate_sha256(_as_string(value, field_name), field_name)


def _safe_component(value: str) -> str:
    normalized = _SAFE_COMPONENT.sub("-", value.casefold().replace("_", "-")).strip("-")
    if not normalized:
        raise ValueError(f"identifier component became empty: {value!r}")
    return normalized


def _resolve_declared(root: Path, relative_path: str) -> Path:
    candidate_value = Path(relative_path)
    if candidate_value.is_absolute():
        raise ValueError(f"manifest data path must be relative: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate_value).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"manifest data path escapes its source root: {relative_path!r}")
    return resolved


def _manifest_pin(root: Path, relative_path: str) -> dict[str, object]:
    path = _resolve_declared(root, relative_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing source manifest: {path}")
    return {
        "relative_path": Path(relative_path).as_posix(),
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _source_revision(source_id: str, pins: Sequence[Mapping[str, object]]) -> str:
    semantic_pins = [
        {"relative_path": pin["relative_path"], "sha256": pin["sha256"]}
        for pin in sorted(pins, key=lambda item: str(item["relative_path"]))
    ]
    return fingerprint_json({"source_id": source_id, "manifest_pins": semantic_pins})


class _Collector:
    def __init__(
        self,
        *,
        leaf_ids: set[str],
        aliases: Mapping[str, str],
        leaf_revision: str,
        verify_files: bool,
    ) -> None:
        self.leaf_ids = leaf_ids
        self.aliases = aliases
        self.leaf_revision = leaf_revision
        self.verify_files = verify_files
        self.shards: list[RegisteredChatShard] = []
        self.references: list[RegisteredReferenceShard] = []
        self.dispositions: list[dict[str, object]] = []
        self.sources: dict[str, dict[str, object]] = {}
        self._verified: dict[Path, tuple[str, int]] = {}
        self._physical_chat_paths: set[Path] = set()

    def canonical_leaf(self, value: str) -> str:
        candidate = self.aliases.get(value, value.replace("_", "-"))
        if candidate not in self.leaf_ids:
            raise ValueError(f"source leaf {value!r} does not resolve into the taxonomy")
        return candidate

    def begin_source(
        self,
        *,
        source_id: str,
        root: Path,
        manifest_relative_paths: Sequence[str],
        source_role: str,
    ) -> str:
        if source_id in self.sources:
            raise ValueError(f"duplicate source ID {source_id!r}")
        pins = [_manifest_pin(root, item) for item in manifest_relative_paths]
        revision = _source_revision(source_id, pins)
        self.sources[source_id] = {
            "source_id": source_id,
            "source_revision": revision,
            "root": str(root.resolve()),
            "source_role": source_role,
            "manifest_pins": pins,
            "declared_output_entries": 0,
            "chat_nonempty_shards": 0,
            "chat_empty_shards": 0,
            "blocked_chat_shards": 0,
            "blocked_chat_records": 0,
            "reference_shards": 0,
            "auxiliary_artifacts": {},
        }
        return revision

    def auxiliary(self, source_id: str, kind: str, count: int = 1) -> None:
        source = self.sources[source_id]
        values = source["auxiliary_artifacts"]
        assert isinstance(values, dict)
        values[kind] = int(values.get(kind, 0)) + count

    def _verify(self, path: Path, expected_sha256: str, expected_records: int) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"declared shard must be a regular non-symlink file: {path}")
        if not self.verify_files:
            return
        cached = self._verified.get(path)
        if cached is not None:
            if cached != (expected_sha256, expected_records):
                raise ValueError(f"physical shard registered with conflicting identity: {path}")
            return
        digest = hashlib.sha256()
        records = 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if line.strip():
                    records += 1
        observed = (digest.hexdigest(), records)
        expected = (expected_sha256, expected_records)
        if observed != expected:
            raise ValueError(
                f"declared shard identity mismatch for {path}: expected {expected}, observed {observed}"
            )
        self._verified[path] = observed

    def add_chat(
        self,
        *,
        source_id: str,
        root: Path,
        relative_path: str,
        leaf_id: str,
        split: str,
        rendered_mode: str,
        semantic_stream: str,
        recipe_stream: str | None = None,
        length_tier: str,
        sha256: str,
        records: int,
        admission_role: str,
        shard_suffix: str = "",
    ) -> None:
        source = self.sources[source_id]
        source["declared_output_entries"] = int(source["declared_output_entries"]) + 1
        leaf = self.canonical_leaf(leaf_id)
        if split not in SPLITS:
            raise ValueError(f"unsupported split {split!r}")
        if rendered_mode not in {"direct", "trace", "tool", "tool-direct", "tool-trace"}:
            raise ValueError(f"unsupported rendered mode {rendered_mode!r}")
        if semantic_stream not in SEMANTIC_STREAMS:
            raise ValueError(f"unsupported semantic stream {semantic_stream!r}")
        resolved_recipe_stream = recipe_stream or semantic_stream
        if resolved_recipe_stream not in RECIPE_STREAMS:
            raise ValueError(f"unsupported recipe stream {resolved_recipe_stream!r}")
        count = _as_count(records, "records")
        digest = _digest(sha256, "sha256")
        path = _resolve_declared(root, relative_path)
        if count == 0:
            if digest != EMPTY_SHA256:
                raise ValueError(f"empty declared shard has non-empty digest: {path}")
            source["chat_empty_shards"] = int(source["chat_empty_shards"]) + 1
            self.dispositions.append(
                {
                    "source_id": source_id,
                    "path": str(path),
                    "disposition": "not-admitted",
                    "reason": "manifest-declared-empty-chat-shard",
                }
            )
            self._verify(path, digest, count)
            return
        if path in self._physical_chat_paths:
            raise ValueError(f"physical chat shard would be registered twice: {path}")
        self._physical_chat_paths.add(path)
        self._verify(path, digest, count)
        source["chat_nonempty_shards"] = int(source["chat_nonempty_shards"]) + 1
        tier_component = _safe_component(length_tier)
        suffix = f"/{_safe_component(shard_suffix)}" if shard_suffix else ""
        shard_id = (
            f"{_safe_component(source_id)}/{leaf}/{resolved_recipe_stream}/"
            f"{tier_component}/{split}{suffix}"
        )
        descriptor = CorpusShardDescriptor(
            shard_id=shard_id,
            shard_revision=str(source["source_revision"]),
            leaf_id=leaf,
            leaf_revision=self.leaf_revision,
            split=split,
            template_mode=f"{rendered_mode}-v1",
            path=str(path),
            sha256=digest,
            record_count=count,
        )
        self.shards.append(
            RegisteredChatShard(
                descriptor=descriptor,
                source_id=source_id,
                source_revision=str(source["source_revision"]),
                semantic_stream=semantic_stream,
                recipe_stream=resolved_recipe_stream,
                length_tier=length_tier,
                admission_role=admission_role,
            )
        )

    def add_reference(
        self,
        *,
        source_id: str,
        root: Path,
        relative_path: str,
        leaf_id: str,
        split: str,
        modality: str,
        sha256: str,
        records: int,
    ) -> None:
        source = self.sources[source_id]
        source["declared_output_entries"] = int(source["declared_output_entries"]) + 1
        leaf = self.canonical_leaf(leaf_id)
        count = _as_count(records, "records")
        digest = _digest(sha256, "sha256")
        path = _resolve_declared(root, relative_path)
        if count == 0:
            self.dispositions.append(
                {
                    "source_id": source_id,
                    "path": str(path),
                    "disposition": "not-admitted",
                    "reason": "manifest-declared-empty-reference-shard",
                }
            )
            self._verify(path, digest, count)
            return
        self._verify(path, digest, count)
        source["reference_shards"] = int(source["reference_shards"]) + 1
        self.references.append(
            RegisteredReferenceShard(
                source_id=source_id,
                leaf_id=leaf,
                split=split,
                modality=modality,
                path=str(path),
                sha256=digest,
                record_count=count,
            )
        )

    def add_blocked_chat(
        self,
        *,
        source_id: str,
        root: Path,
        relative_path: str,
        sha256: str,
        records: int,
        reason: str,
    ) -> None:
        """Account for a manifest-declared chat mode the trainer cannot admit."""

        source = self.sources[source_id]
        source["declared_output_entries"] = int(source["declared_output_entries"]) + 1
        count = _as_count(records, "records")
        digest = _digest(sha256, "sha256")
        path = _resolve_declared(root, relative_path)
        self._verify(path, digest, count)
        source["blocked_chat_shards"] = int(source["blocked_chat_shards"]) + 1
        source["blocked_chat_records"] = int(source["blocked_chat_records"]) + count
        self.dispositions.append(
            {
                "source_id": source_id,
                "path": str(path),
                "records": count,
                "sha256": digest,
                "disposition": "blocked-contract-gap",
                "reason": reason,
            }
        )

    def add_nonchat_output(
        self, *, source_id: str, root: Path, relative_path: str, reason: str
    ) -> None:
        source = self.sources[source_id]
        source["declared_output_entries"] = int(source["declared_output_entries"]) + 1
        path = _resolve_declared(root, relative_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"declared auxiliary artifact is missing: {path}")
        self.dispositions.append(
            {
                "source_id": source_id,
                "path": str(path),
                "disposition": "not-chat-input",
                "reason": reason,
            }
        )


def _taxonomy_contract(
    path: Path,
) -> tuple[dict[str, object], str, set[str], dict[str, str], dict[str, str]]:
    taxonomy = _strict_json(path)
    claimed = _digest(taxonomy.get("content_sha256"), "taxonomy.content_sha256")
    semantic = {key: value for key, value in taxonomy.items() if key != "content_sha256"}
    if fingerprint_json(semantic) != claimed:
        raise ValueError("taxonomy content_sha256 does not match its semantic payload")
    families = _as_list(taxonomy.get("families"), "taxonomy.families")
    leaf_ids: set[str] = set()
    family_by_leaf: dict[str, str] = {}
    for family_index, raw_family in enumerate(families):
        family = _as_mapping(raw_family, f"taxonomy.families[{family_index}]")
        family_id = _as_string(family.get("family_id"), "family_id")
        for raw_leaf in _as_list(family.get("leaves"), f"taxonomy.{family_id}.leaves"):
            leaf = _as_mapping(raw_leaf, "taxonomy leaf")
            leaf_id = _as_string(leaf.get("leaf_id"), "leaf_id")
            if leaf_id in leaf_ids:
                raise ValueError(f"duplicate taxonomy leaf {leaf_id!r}")
            leaf_ids.add(leaf_id)
            family_by_leaf[leaf_id] = family_id
    raw_aliases = _as_mapping(taxonomy.get("aliases"), "taxonomy.aliases")
    aliases = {
        _as_string(key, "alias key"): _as_string(value, f"alias[{key}]")
        for key, value in raw_aliases.items()
    }
    if any(value not in leaf_ids for value in aliases.values()):
        raise ValueError("taxonomy alias points outside the leaf set")
    return taxonomy, _sha256_file(path), leaf_ids, family_by_leaf, aliases


def _nested_distillation_source(
    collector: _Collector,
    *,
    source_id: str,
    root: Path,
    container_key: str,
    primary_leaves: set[str] | None = None,
) -> None:
    manifest = _strict_json(root / "mixture-manifest.json")
    source = collector.sources[source_id]
    revision = str(source["source_revision"])
    if not revision:
        raise AssertionError("source revision missing")
    container = _as_mapping(manifest.get(container_key), f"{source_id}.{container_key}")
    for raw_leaf, raw_leaf_value in sorted(container.items()):
        leaf_value = _as_mapping(raw_leaf_value, f"{source_id}.{raw_leaf}")
        streams_value = leaf_value.get("streams") if container_key == "leaves" else raw_leaf_value
        streams = _as_mapping(streams_value, f"{source_id}.{raw_leaf}.streams")
        leaf = collector.canonical_leaf(str(raw_leaf))
        admission = (
            "primary"
            if primary_leaves is None or leaf in primary_leaves
            else "supplemental-distinct-source"
        )
        for raw_stream, raw_splits in sorted(streams.items()):
            stream = _as_string(raw_stream, "stream name")
            rendered_mode = stream.split("-", 1)[0]
            tiers = stream.split("-", 1)[1] if "-" in stream else "all"
            split_values = _as_mapping(raw_splits, f"{source_id}.{raw_leaf}.{stream}")
            for split, raw_output in sorted(split_values.items()):
                output = _as_mapping(raw_output, "distillation output")
                if rendered_mode in {"tool_direct", "tool_trace"}:
                    tool_recipe_stream = rendered_mode.replace("_", "-")
                    collector.add_chat(
                        source_id=source_id,
                        root=root,
                        relative_path=_as_string(output.get("path"), "output.path"),
                        leaf_id=leaf,
                        split=str(split),
                        rendered_mode=tool_recipe_stream,
                        semantic_stream="tool",
                        recipe_stream=tool_recipe_stream,
                        length_tier=tiers,
                        sha256=_as_string(output.get("sha256"), "output.sha256"),
                        records=_as_count(output.get("records"), "output.records"),
                        admission_role="primary-native-tool-trajectory",
                        shard_suffix=stream,
                    )
                    continue
                semantic_stream = rendered_mode
                collector.add_chat(
                    source_id=source_id,
                    root=root,
                    relative_path=_as_string(output.get("path"), "output.path"),
                    leaf_id=leaf,
                    split=str(split),
                    rendered_mode=rendered_mode,
                    semantic_stream=semantic_stream,
                    length_tier=tiers,
                    sha256=_as_string(output.get("sha256"), "output.sha256"),
                    records=_as_count(output.get("records"), "output.records"),
                    admission_role=admission,
                    shard_suffix=stream,
                )


def _software_source(collector: _Collector, root: Path) -> None:
    source_id = "software-architecture-experts-v1"
    manifest = _strict_json(root / "quality-manifest.json")
    for index, raw_output in enumerate(_as_list(manifest.get("outputs"), "software.outputs")):
        output = _as_mapping(raw_output, f"software.outputs[{index}]")
        mode = _as_string(output.get("mode"), "software.mode")
        collector.add_chat(
            source_id=source_id,
            root=root,
            relative_path=_as_string(output.get("path"), "software.path"),
            leaf_id=_as_string(output.get("leaf"), "software.leaf"),
            split=_as_string(output.get("split"), "software.split"),
            rendered_mode=mode,
            semantic_stream=mode,
            length_tier=_as_string(output.get("tier"), "software.tier"),
            sha256=_as_string(output.get("sha256"), "software.sha256"),
            records=_as_count(output.get("records"), "software.records"),
            admission_role="primary-distinct-source",
        )
        collector.auxiliary(source_id, "content-free-provenance-sidecar")
    collector.auxiliary(source_id, "quarantine-ledger")


def _length_tier_source(
    collector: _Collector,
    *,
    source_id: str,
    root: Path,
    branches: Mapping[str, tuple[str, str]],
) -> None:
    """Load split-local length-tier manifests.

    ``branches`` maps the physical branch directory to ``(leaf, semantic_stream)``.
    A semantic ``tool`` stream may intentionally be rendered as direct chat when its
    supervised target is an assistant-form tool call rather than a native tool role.
    """

    for branch, (leaf_id, forced_semantic_stream) in sorted(branches.items()):
        branch_root = root / branch
        for split in SPLITS:
            manifest_path = branch_root / split / "manifest.json"
            if not manifest_path.is_file():
                collector.dispositions.append(
                    {
                        "source_id": source_id,
                        "branch": branch,
                        "split": split,
                        "disposition": "missing-split",
                        "reason": "upstream-canonical-source-did-not-publish-this-split",
                    }
                )
                continue
            manifest = _strict_json(manifest_path)
            outputs = _as_list(manifest.get("outputs"), f"{source_id}.{branch}.{split}.outputs")
            for index, raw_output in enumerate(outputs):
                output = _as_mapping(raw_output, f"length output {index}")
                template_mode = output.get("template_mode")
                relative = _as_string(output.get("path"), "length output.path")
                if template_mode is None:
                    collector.add_nonchat_output(
                        source_id=source_id,
                        root=branch_root / split,
                        relative_path=relative,
                        reason="quarantine-ledger-not-training-payload",
                    )
                    continue
                rendered_mode = _as_string(template_mode, "length output.template_mode")
                records = _as_count(output.get("records"), "length output.records")
                semantic = (
                    "tool"
                    if forced_semantic_stream == "tool" and rendered_mode == "direct"
                    else rendered_mode
                )
                collector.add_chat(
                    source_id=source_id,
                    root=branch_root / split,
                    relative_path=relative,
                    leaf_id=leaf_id,
                    split=split,
                    rendered_mode=rendered_mode,
                    semantic_stream=semantic,
                    length_tier=_as_string(output.get("tier"), "length output.tier"),
                    sha256=_as_string(output.get("sha256"), "length output.sha256"),
                    records=records,
                    admission_role="primary",
                )


def _embedded_source(collector: _Collector, root: Path) -> None:
    source_id = "embedded-source-experts-v1"
    manifest = _strict_json(root / "manifest.json")
    for index, raw_output in enumerate(_as_list(manifest.get("outputs"), "embedded.outputs")):
        output = _as_mapping(raw_output, f"embedded.outputs[{index}]")
        data = _as_mapping(output.get("data"), "embedded output.data")
        modality = _as_string(output.get("modality"), "embedded output.modality")
        common = {
            "source_id": source_id,
            "root": root,
            "relative_path": _as_string(data.get("path"), "embedded data.path"),
            "leaf_id": _as_string(output.get("leaf"), "embedded output.leaf"),
            "split": _as_string(output.get("split"), "embedded output.split"),
            "sha256": _as_string(data.get("sha256"), "embedded data.sha256"),
            "records": _as_count(data.get("records"), "embedded data.records"),
        }
        if modality == "reference_cpt":
            collector.add_reference(**common, modality="reference-cpt")
        elif modality == "sft":
            relative_parts = Path(str(common["relative_path"])).parts
            template_family = relative_parts[3] if len(relative_parts) >= 4 else "unknown-family"
            mode = _as_string(output.get("mode"), "embedded output.mode")
            tier = _as_string(output.get("tier"), "embedded output.tier")
            collector.add_chat(
                **common,
                rendered_mode=mode,
                semantic_stream=mode,
                length_tier=f"{template_family}/{tier}",
                admission_role="primary-grounded-sft",
                shard_suffix=template_family,
            )
        else:
            raise ValueError(f"unknown embedded output modality {modality!r}")
        collector.auxiliary(source_id, "content-free-provenance-sidecar")
    collector.auxiliary(source_id, "non-emitted-disposition-ledger")


def _embedded_grounded_reconstruction(collector: _Collector, root: Path) -> None:
    source_id = "embedded-grounded-reconstruction-v1"
    manifest = _strict_json(root / "manifest.json")
    task = _as_mapping(manifest.get("task_contract"), "embedded-grounded.task_contract")
    if task.get("task_type") != "grounded_reference_reconstruction":
        raise ValueError("embedded grounded source changed task type")
    if task.get("concatenated_targets_exactly_reproduce_each_source") is not True:
        raise ValueError("embedded grounded source no longer guarantees exact reconstruction")
    if task.get("contains_synthetic_question_answering") is not False:
        raise ValueError("embedded grounded source changed synthetic-QA contract")
    for index, raw_output in enumerate(
        _as_list(manifest.get("outputs"), "embedded-grounded.outputs")
    ):
        output = _as_mapping(raw_output, f"embedded-grounded.outputs[{index}]")
        collector.add_chat(
            source_id=source_id,
            root=root,
            relative_path=_as_string(
                output.get("messages_path"), "embedded-grounded.messages_path"
            ),
            leaf_id=_as_string(output.get("leaf_id"), "embedded-grounded.leaf_id"),
            split=_as_string(output.get("split"), "embedded-grounded.split"),
            rendered_mode="direct",
            semantic_stream="direct",
            length_tier=_as_string(output.get("tier"), "embedded-grounded.tier"),
            sha256=_as_string(output.get("messages_sha256"), "embedded-grounded.messages_sha256"),
            records=_as_count(output.get("records"), "embedded-grounded.records"),
            admission_role="primary-grounded-reference-reconstruction",
        )
        collector.auxiliary(source_id, "content-free-provenance-sidecar")
    collector.auxiliary(source_id, "content-free-accounting-ledger")
    collector.auxiliary(source_id, "aggregate-report")


def _pin_embedded_grounded_registration(
    collector: _Collector,
    *,
    registration_path: Path,
    source_root: Path,
) -> None:
    """Bind the independently rebuilt embedded source's admission fragment."""

    source_id = "embedded-grounded-reconstruction-v1"
    fragment = _strict_json(registration_path)
    if fragment.get("schema_version") != "molor.data-plane-registration-fragment.v1":
        raise ValueError("embedded grounded registration schema changed")
    if fragment.get("logical_id") != source_id or fragment.get("state") != "supervised_ready":
        raise ValueError("embedded grounded registration is not supervised-ready")
    if set(_as_list(fragment.get("leaves"), "embedded registration.leaves")) != {
        "embedded-code-generation",
        "embedded-knowledge",
    }:
        raise ValueError("embedded grounded registration leaf set changed")
    if set(_as_list(fragment.get("splits"), "embedded registration.splits")) != set(SPLITS):
        raise ValueError("embedded grounded registration split set changed")

    manifest_path = (source_root / "manifest.json").resolve()
    declared_manifest_path = Path(
        _as_string(fragment.get("manifest_path"), "embedded registration.manifest_path")
    ).resolve()
    if declared_manifest_path != manifest_path:
        raise ValueError("embedded grounded registration points at another manifest")
    manifest = _strict_json(manifest_path)
    observed_manifest_sha = _sha256_file(manifest_path)
    if (
        _digest(fragment.get("manifest_sha256"), "embedded registration.manifest_sha256")
        != observed_manifest_sha
    ):
        raise ValueError("embedded grounded registration manifest file identity mismatch")
    if _digest(
        fragment.get("manifest_content_sha256"),
        "embedded registration.manifest_content_sha256",
    ) != _digest(manifest.get("content_sha256"), "embedded manifest.content_sha256"):
        raise ValueError("embedded grounded registration semantic identity mismatch")

    contract = _as_mapping(fragment.get("contract"), "embedded registration.contract")
    if (
        contract.get("grounded_reference_reconstruction") is not True
        or contract.get("contains_synthetic_question_answering") is not False
        or contract.get("truncation") is not False
        or contract.get("proves_answer_quality") is not False
        or contract.get("assistant_target") != "contiguous verbatim source-text chunk"
    ):
        raise ValueError("embedded grounded registration contract changed")

    source = collector.sources[source_id]
    registration_pin = {
        "path": str(registration_path.resolve()),
        "sha256": _sha256_file(registration_path),
        "manifest_sha256": observed_manifest_sha,
        "manifest_content_sha256": fragment["manifest_content_sha256"],
        "state": "supervised_ready",
    }
    source["registration_fragment"] = registration_pin
    pins = _as_list(source.get("manifest_pins"), "embedded source manifest_pins")
    source["source_revision"] = fingerprint_json(
        {
            "source_id": source_id,
            "manifest_pins": [
                {"relative_path": pin["relative_path"], "sha256": pin["sha256"]}
                for pin in sorted(
                    pins,
                    key=lambda item: (
                        str(item["relative_path"]) if isinstance(item, Mapping) else ""
                    ),
                )
                if isinstance(pin, Mapping)
            ],
            "registration_fragment_sha256": registration_pin["sha256"],
        }
    )


def _pin_structured_tool_audit(
    collector: _Collector,
    *,
    audit_path: Path,
) -> None:
    """Verify and bind the strict real-row tool audit to its descriptor set."""

    receipt = _strict_json(audit_path)
    if receipt.get("schema_version") != "molor.structured-tool-audit-receipt.v1":
        raise ValueError("structured-tool audit schema changed")
    if receipt.get("audit_id") != "molor-native-structured-tool-training-v1":
        raise ValueError("structured-tool audit identity changed")
    if receipt.get("training_performed") is not False:
        raise ValueError("structured-tool audit unexpectedly reports training")
    claimed_content = _digest(receipt.get("content_sha256"), "structured-tool audit.content_sha256")
    semantic_receipt = dict(receipt)
    semantic_receipt.pop("content_sha256")
    if fingerprint_json(semantic_receipt) != claimed_content:
        raise ValueError("structured-tool audit semantic identity mismatch")

    descriptor_contract = _as_mapping(
        receipt.get("audited_descriptor_set"), "structured-tool audit descriptor set"
    )
    source_id = "broad-distillation-experts-v1"
    expected_fields = [
        "shard_id",
        "path",
        "sha256",
        "record_count",
        "semantic_stream",
        "recipe_stream",
        "template_mode",
    ]
    if descriptor_contract.get("source_id") != source_id:
        raise ValueError("structured-tool audit source changed")
    if descriptor_contract.get("descriptor_fields") != expected_fields:
        raise ValueError("structured-tool audit descriptor projection changed")
    if descriptor_contract.get("canonical_order") != ["shard_id"]:
        raise ValueError("structured-tool audit descriptor order changed")
    if descriptor_contract.get("fingerprint_algorithm") != (
        "molor.identity.fingerprint_json(list[projected_descriptor])"
    ):
        raise ValueError("structured-tool audit fingerprint algorithm changed")
    audited_shards = sorted(
        (
            shard
            for shard in collector.shards
            if shard.source_id == source_id and shard.semantic_stream == "tool"
        ),
        key=lambda shard: shard.descriptor.shard_id,
    )
    projected_descriptors = [
        {
            "shard_id": shard.descriptor.shard_id,
            "path": shard.descriptor.path,
            "sha256": shard.descriptor.sha256,
            "record_count": shard.descriptor.record_count,
            "semantic_stream": shard.semantic_stream,
            "recipe_stream": shard.recipe_stream,
            "template_mode": shard.descriptor.template_mode,
        }
        for shard in audited_shards
    ]
    descriptor_fingerprint = fingerprint_json(projected_descriptors)
    if (
        _digest(
            descriptor_contract.get("descriptor_set_sha256"),
            "structured-tool audit.descriptor_set_sha256",
        )
        != descriptor_fingerprint
    ):
        raise ValueError("structured-tool audit does not match registered tool shards")
    tool_rows = sum(shard.descriptor.record_count for shard in audited_shards)
    if (
        descriptor_contract.get("shards") != len(audited_shards)
        or descriptor_contract.get("rows") != tool_rows
    ):
        raise ValueError("structured-tool audit descriptor accounting changed")

    source = collector.sources[source_id]
    if descriptor_contract.get("source_revision") != source["source_revision"]:
        raise ValueError("structured-tool audit source revision changed")
    declared_source_pins = _as_list(
        descriptor_contract.get("source_manifest_pins"),
        "structured-tool audit source manifest pins",
    )
    observed_source_pins = _as_list(
        source.get("manifest_pins"), "registered broad source manifest pins"
    )

    def source_pin_identity(raw_pin: object) -> tuple[str, str, int]:
        pin = _as_mapping(raw_pin, "structured-tool source pin")
        return (
            str(Path(_as_string(pin.get("path"), "structured-tool pin.path")).resolve()),
            _digest(pin.get("sha256"), "structured-tool pin.sha256"),
            _as_count(pin.get("bytes"), "structured-tool pin.bytes"),
        )

    if sorted(source_pin_identity(pin) for pin in declared_source_pins) != sorted(
        source_pin_identity(pin) for pin in observed_source_pins
    ):
        raise ValueError("structured-tool audit source manifests changed")

    aggregate = _as_mapping(receipt.get("aggregate_validation"), "structured-tool audit aggregate")
    if any(
        aggregate.get(key) != expected
        for key, expected in {
            "rows_declared": tool_rows,
            "rows_scanned": tool_rows,
            "rows_valid": tool_rows,
            "shards": len(audited_shards),
            "roundtrip_mismatches": 0,
            "zero_silent_drops": True,
        }.items()
    ):
        raise ValueError("structured-tool audit aggregate accounting changed")
    dispositions = _as_mapping(aggregate.get("dispositions"), "structured-tool audit dispositions")
    standard = _as_count(dispositions.get("accepted_standard"), "structured-tool accepted_standard")
    truncated = _as_count(
        dispositions.get("accepted_pre_assistant_truncated_observation_window"),
        "structured-tool accepted truncated observation windows",
    )
    if (
        standard + truncated != tool_rows
        or dispositions.get("quarantined") != 0
        or dispositions.get("rejected") != 0
    ):
        raise ValueError("structured-tool audit dispositions do not reconcile")
    failure_categories = _as_mapping(
        aggregate.get("failure_categories"), "structured-tool audit failure categories"
    )
    if not failure_categories or any(value != 0 for value in failure_categories.values()):
        raise ValueError("structured-tool audit contains strict-contract failures")

    qwen = _as_mapping(receipt.get("qwen3_prefix_validation"), "structured-tool Qwen validation")
    if any(
        qwen.get(key) != expected
        for key, expected in {
            "prefix_consistent_rows": tool_rows,
            "nonempty_final_targets": tool_rows,
            "prefix_mismatches": 0,
            "template_errors": 0,
        }.items()
    ):
        raise ValueError("structured-tool Qwen validation changed")
    if (
        _as_count(qwen.get("final_answer_targets"), "structured-tool final answers")
        + _as_count(qwen.get("final_tool_call_targets"), "structured-tool final calls")
        != tool_rows
    ):
        raise ValueError("structured-tool Qwen target accounting changed")

    streams = _as_mapping(receipt.get("streams"), "structured-tool audit streams")
    for receipt_stream, recipe_stream in (
        ("tool_direct", "tool-direct"),
        ("tool_trace", "tool-trace"),
    ):
        stream = _as_mapping(
            streams.get(receipt_stream), f"structured-tool audit stream {receipt_stream}"
        )
        matching = [shard for shard in audited_shards if shard.recipe_stream == recipe_stream]
        if (
            stream.get("rows") != sum(shard.descriptor.record_count for shard in matching)
            or stream.get("shards") != len(matching)
            or stream.get("template_mode") != f"{recipe_stream}-v1"
        ):
            raise ValueError(f"structured-tool audit stream {receipt_stream} changed")

    strict_contract = _as_mapping(receipt.get("strict_contract"), "structured-tool strict contract")
    required_true_contract_fields = {
        "assistant_only_masking": (
            "final_answer_or_tool_call_target_unmasked",
            "generation_prefix_and_prior_tool_trace_masked",
        ),
        "call_result_linkage": (
            "call_ids_unique_across_trace",
            "id_bearing_results_match_one_pending_call",
            "idless_calls_and_results_balance_positionally",
            "intermediate_calls_fully_resolved",
        ),
        "schema_preservation": (
            "empty_array_presence",
            "null_presence",
            "object_versus_encoded_string_arguments",
        ),
    }
    for section_name, field_names in required_true_contract_fields.items():
        section = _as_mapping(
            strict_contract.get(section_name), f"structured-tool contract {section_name}"
        )
        if any(section.get(field_name) is not True for field_name in field_names):
            raise ValueError(f"structured-tool contract {section_name} changed")
    truncated_contract = _as_mapping(
        strict_contract.get("pre_assistant_truncated_observation_window"),
        "structured-tool truncated observation contract",
    )
    if any(
        truncated_contract.get(field_name) is not True
        for field_name in (
            "admitted",
            "duplicate_non_null_result_ids_rejected",
            "must_close_with_assistant",
        )
    ):
        raise ValueError("structured-tool truncated observation contract changed")

    compatibility = _as_mapping(
        receipt.get("registry_compatibility"), "structured-tool registry compatibility"
    )
    if (
        compatibility.get("compatibility_key") != "audited_descriptor_set.descriptor_set_sha256"
        or compatibility.get("final_registry_file_or_content_hash_pinned") is not False
        or compatibility.get("final_registry_may_differ_outside_audited_descriptor_set") is not True
    ):
        raise ValueError("structured-tool registry compatibility contract changed")

    repository_root = audit_path.resolve().parents[2]
    implementation = _as_mapping(
        receipt.get("implementation"), "structured-tool implementation pins"
    )
    tokenizer_pins = _as_list(qwen.get("tokenizer_files"), "structured-tool tokenizer pins")
    code_pins = [
        *_as_list(implementation.get("files"), "structured-tool implementation files"),
        *_as_list(implementation.get("test_files"), "structured-tool test files"),
        *tokenizer_pins,
    ]
    for raw_pin in code_pins:
        pin = _as_mapping(raw_pin, "structured-tool code pin")
        relative = _as_string(pin.get("path"), "structured-tool code pin.path")
        resolved = _resolve_declared(repository_root, relative)
        if _sha256_file(resolved) != _digest(pin.get("sha256"), "structured-tool code pin.sha256"):
            raise ValueError(f"structured-tool audited dependency changed: {relative}")

    scope = _as_mapping(receipt.get("scope"), "structured-tool audit scope")
    if scope.get("source_id") != source_id or scope.get("out_of_scope_corpus_rows_accessed") != 0:
        raise ValueError("structured-tool audit scope changed")
    source["training_contract_audit"] = {
        "path": str(audit_path.resolve()),
        "sha256": _sha256_file(audit_path),
        "content_sha256": claimed_content,
        "descriptor_set_sha256": descriptor_fingerprint,
        "rows": tool_rows,
        "shards": len(audited_shards),
        "dispositions": dict(dispositions),
        "qwen_prefix_consistent_rows": qwen["prefix_consistent_rows"],
        "qwen_nonempty_final_targets": qwen["nonempty_final_targets"],
    }


def _business_references(collector: _Collector, root: Path) -> None:
    source_id = "business-reference-v1"
    manifest = _strict_json(root / "manifest.json")
    for index, raw_output in enumerate(
        _as_list(manifest.get("shards"), "business-reference.shards")
    ):
        output = _as_mapping(raw_output, f"business-reference.shards[{index}]")
        collector.add_reference(
            source_id=source_id,
            root=root,
            relative_path=_as_string(output.get("path"), "business-reference.path"),
            leaf_id=_as_string(output.get("leaf_id"), "business-reference.leaf_id"),
            split=_as_string(output.get("split"), "business-reference.split"),
            modality="reference-grounding",
            sha256=_as_string(output.get("sha256"), "business-reference.sha256"),
            records=_as_count(output.get("records"), "business-reference.records"),
        )
    collector.auxiliary(source_id, "content-free-provenance-sidecars", count=3)


def _business_grounded_reconstruction(collector: _Collector, root: Path) -> None:
    source_id = "business-grounded-reconstruction-v1"
    manifest = _strict_json(root / "manifest.json")
    task = _as_mapping(manifest.get("task_contract"), "business-grounded.task_contract")
    if task.get("task_type") != "grounded_reference_reconstruction":
        raise ValueError("business grounded source changed task type")
    if task.get("assistant_target_is_verbatim_source_passage") is not True:
        raise ValueError("business grounded source no longer pins verbatim targets")
    for index, raw_output in enumerate(
        _as_list(manifest.get("outputs"), "business-grounded.outputs")
    ):
        output = _as_mapping(raw_output, f"business-grounded.outputs[{index}]")
        collector.add_chat(
            source_id=source_id,
            root=root,
            relative_path=_as_string(
                output.get("messages_path"), "business-grounded.messages_path"
            ),
            leaf_id=_as_string(output.get("leaf_id"), "business-grounded.leaf_id"),
            split=_as_string(output.get("split"), "business-grounded.split"),
            rendered_mode="direct",
            semantic_stream="direct",
            length_tier=_as_string(output.get("tier"), "business-grounded.tier"),
            sha256=_as_string(output.get("messages_sha256"), "business-grounded.messages_sha256"),
            records=_as_count(output.get("records"), "business-grounded.records"),
            admission_role="primary-grounded-reference-reconstruction",
        )
        collector.auxiliary(source_id, "content-free-provenance-sidecar")
    collector.auxiliary(source_id, "content-free-accounting-ledger")


def collect_expert_data(
    paths: ExpertMixtureSourcePaths, *, verify_files: bool = True
) -> CollectedExpertData:
    taxonomy, taxonomy_sha, leaf_ids, family_by_leaf, aliases = _taxonomy_contract(paths.taxonomy)
    leaf_revision = (
        f"taxonomy-v1-{_as_string(taxonomy.get('content_sha256'), 'taxonomy hash')[:16]}"
    )
    collector = _Collector(
        leaf_ids=leaf_ids,
        aliases=aliases,
        leaf_revision=leaf_revision,
        verify_files=verify_files,
    )

    collector.begin_source(
        source_id="qwen38-distillation-experts-v2",
        root=paths.qwen38_v2,
        manifest_relative_paths=(
            "mixture-manifest.json",
            "quality-manifest.json",
            "sha256sums.txt",
        ),
        source_role="primary-for-commonsense-otherwise-supplemental",
    )
    collector.begin_source(
        source_id="broad-distillation-experts-v1",
        root=paths.broad_v1,
        manifest_relative_paths=(
            "mixture-manifest.json",
            "quality-manifest.json",
            "sha256sums.txt",
        ),
        source_role="primary-granular-distillation",
    )
    collector.begin_source(
        source_id="software-architecture-experts-v1",
        root=paths.software_v1,
        manifest_relative_paths=("quality-manifest.json", "sha256sums.txt"),
        source_role="primary-distinct-software-source",
    )
    finance_manifests = tuple(
        f"{leaf}/{split}/manifest.json"
        for leaf in ("business-finance", "investing-markets")
        for split in SPLITS
    )
    collector.begin_source(
        source_id="business-finance-length-tiers-v1",
        root=paths.business_finance_v1,
        manifest_relative_paths=finance_manifests,
        source_role="primary-finance-supervision",
    )
    collector.begin_source(
        source_id="business-grounded-reconstruction-v1",
        root=paths.business_grounded_v1,
        manifest_relative_paths=("manifest.json", "sha256sums.txt"),
        source_role="primary-grounded-reference-reconstruction",
    )
    agentic_manifests = tuple(
        [f"taskmaster-coffee-api-v2/{split}/manifest.json" for split in SPLITS]
        + [f"terminal-cli/{split}/manifest.json" for split in ("train", "validation")]
    )
    collector.begin_source(
        source_id="agentic-tool-length-tiers-v1",
        root=paths.agentic_tool_v1,
        manifest_relative_paths=agentic_manifests,
        source_role="primary-agentic-supervision",
    )
    collector.begin_source(
        source_id="embedded-source-experts-v1",
        root=paths.embedded_v1,
        manifest_relative_paths=("manifest.json", "sha256sums.txt"),
        source_role="primary-grounded-embedded-source",
    )
    collector.begin_source(
        source_id="embedded-grounded-reconstruction-v1",
        root=paths.embedded_grounded_v1,
        manifest_relative_paths=(
            "manifest.json",
            "aggregate_report.json",
            "sha256sums.txt",
        ),
        source_role="derived-grounded-reference-reconstruction",
    )
    _pin_embedded_grounded_registration(
        collector,
        registration_path=paths.embedded_grounded_registration,
        source_root=paths.embedded_grounded_v1,
    )
    collector.begin_source(
        source_id="business-reference-v1",
        root=paths.business_reference_v1,
        manifest_relative_paths=("manifest.json", "sha256sums.txt"),
        source_role="reference-only-not-chat-sft",
    )

    _nested_distillation_source(
        collector,
        source_id="qwen38-distillation-experts-v2",
        root=paths.qwen38_v2,
        container_key="leaves",
        primary_leaves={"commonsense-logic"},
    )
    _nested_distillation_source(
        collector,
        source_id="broad-distillation-experts-v1",
        root=paths.broad_v1,
        container_key="streams",
    )
    _software_source(collector, paths.software_v1)
    _length_tier_source(
        collector,
        source_id="business-finance-length-tiers-v1",
        root=paths.business_finance_v1,
        branches={
            "business-finance": ("business-finance", "direct"),
            "investing-markets": ("investing-markets", "direct"),
        },
    )
    _business_grounded_reconstruction(collector, paths.business_grounded_v1)
    _length_tier_source(
        collector,
        source_id="agentic-tool-length-tiers-v1",
        root=paths.agentic_tool_v1,
        branches={
            "taskmaster-coffee-api-v2": ("tool-use-execution", "tool"),
            "terminal-cli": ("terminal-cli", "direct"),
        },
    )
    _embedded_source(collector, paths.embedded_v1)
    _embedded_grounded_reconstruction(collector, paths.embedded_grounded_v1)
    _business_references(collector, paths.business_reference_v1)
    _pin_structured_tool_audit(
        collector,
        audit_path=paths.structured_tool_audit,
    )

    shard_ids = [item.descriptor.shard_id for item in collector.shards]
    if len(shard_ids) != len(set(shard_ids)):
        duplicates = sorted(item for item, count in Counter(shard_ids).items() if count > 1)
        raise ValueError(f"duplicate registered shard IDs: {duplicates}")

    return CollectedExpertData(
        taxonomy=taxonomy,
        taxonomy_path=str(paths.taxonomy.resolve()),
        taxonomy_sha256=taxonomy_sha,
        leaf_revision=leaf_revision,
        family_by_leaf=family_by_leaf,
        shards=tuple(sorted(collector.shards, key=lambda item: item.descriptor.shard_id)),
        references=tuple(
            sorted(
                collector.references,
                key=lambda item: (item.source_id, item.leaf_id, item.split, item.path),
            )
        ),
        sources=tuple(
            dict(collector.sources[source_id]) for source_id in sorted(collector.sources)
        ),
        dispositions=tuple(
            sorted(
                collector.dispositions,
                key=lambda item: (
                    str(item.get("source_id", "")),
                    str(item.get("path", "")),
                    str(item.get("branch", "")),
                    str(item.get("split", "")),
                ),
            )
        ),
    )


def _tier_weight(tier: str) -> float:
    value = tier.casefold().replace("_", "-")
    if "8k-to-32k" in value or "8193-32768" in value:
        return 0.15
    if "4k-to-8k" in value or "4097-8192" in value:
        return 0.35
    if "1k-to-4k" in value or "1025-4096" in value or "long" in value:
        return 0.7
    return 1.0


def _shard_weight(shard: RegisteredChatShard) -> float:
    role_weight = 0.35 if shard.admission_role == "supplemental-distinct-source" else 1.0
    return round(role_weight * _tier_weight(shard.length_tier), 6)


def build_canonical_recipes(
    collected: CollectedExpertData,
) -> dict[str, CorpusMixtureRecipe]:
    grouped: dict[tuple[str, str, str], list[RegisteredChatShard]] = defaultdict(list)
    for shard in collected.shards:
        grouped[
            (
                shard.descriptor.leaf_id,
                shard.recipe_stream,
                shard.descriptor.split,
            )
        ].append(shard)

    train_guards = tuple(
        shard.descriptor for shard in collected.shards if shard.descriptor.split == "train"
    )
    heldout_guards = tuple(
        shard.descriptor
        for shard in collected.shards
        if shard.descriptor.split in {"validation", "test"}
    )
    recipes: dict[str, CorpusMixtureRecipe] = {}
    for (leaf, stream, split), shards in sorted(grouped.items()):
        template_modes = {shard.descriptor.template_mode for shard in shards}
        if len(template_modes) != 1:
            raise ValueError(
                f"leaf/stream/split has multiple render contracts: {(leaf, stream, split)}"
            )
        usage = "training" if split == "train" else "evaluation"
        guards = heldout_guards if usage == "training" else train_guards
        available = sum(shard.descriptor.record_count for shard in shards)
        for variant in ("fast", "full"):
            if variant == "fast":
                budget_stream = "tool" if stream.startswith("tool") else stream
                budget = FAST_RECORD_BUDGETS[(split, budget_stream)]
                target = min(available, budget)
            else:
                budget = available
                target = available
            recipe_id = f"rack-v5-{leaf}-{stream}-{variant}-{split}"
            recipe = CorpusMixtureRecipe(
                recipe_id=recipe_id,
                revision="1",
                usage=usage,
                output_split=split,
                template_mode=next(iter(template_modes)),
                seed=f"molor-rack-v5/{leaf}/{stream}/{variant}/{split}/seed-1",
                target_records=target,
                shortfall_policy="emit_available",
                shards=tuple(
                    CorpusMixtureShard(
                        descriptor=shard.descriptor,
                        weight=_shard_weight(shard),
                        cap=(
                            min(shard.descriptor.record_count, budget)
                            if variant == "fast"
                            else None
                        ),
                    )
                    for shard in shards
                ),
                leakage_guard_shards=guards,
            )
            relative = (Path("recipes") / variant / leaf / stream / f"{split}.json").as_posix()
            recipes[relative] = recipe
    return dict(sorted(recipes.items()))


def _normalize_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _router_prompt(example: ChatExample) -> tuple[str, bool]:
    # Serving routes only the latest non-empty user turn.  Router supervision must use
    # that exact scope as well; serializing system/tool/history messages teaches the
    # classifier source-template shortcuts that do not exist at request time.
    prompt = next(
        (
            message.content.strip()
            for message in reversed(example.messages[:-1])
            if message.role == "user" and message.content.strip()
        ),
        "",
    )
    if not prompt:
        raise ValueError("router chat row has no non-empty user prompt before the answer")
    if len(prompt) <= ROUTER_MAX_PROMPT_CHARACTERS:
        return prompt, False
    retained = ROUTER_MAX_PROMPT_CHARACTERS - len(ROUTER_ELISION_MARKER)
    prefix = retained // 2
    suffix = retained - prefix
    return prompt[:prefix] + ROUTER_ELISION_MARKER + prompt[-suffix:], True


def _selection_rank(seed: str, *parts: object) -> str:
    payload = "\0".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scan_chat_for_router(
    connection: sqlite3.Connection,
    shard: RegisteredChatShard,
) -> tuple[int, int]:
    path = Path(shard.descriptor.path)
    digest = hashlib.sha256()
    records = 0
    truncated_prompts = 0
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            records += 1
            try:
                value = json.loads(raw_line.decode("utf-8"))
                example = ChatExample.from_dict(value)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid registered chat row {path}:{line_number}: {exc}"
                ) from exc
            prompt, truncated = _router_prompt(example)
            truncated_prompts += int(truncated)
            normalized = _normalize_prompt(prompt)
            prompt_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            connection.execute(
                """
                INSERT INTO occurrences(
                    digest, normalized, prompt, route_id, split, selection_rank, origin,
                    route_priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt_digest,
                    normalized,
                    prompt,
                    shard.descriptor.leaf_id,
                    shard.descriptor.split,
                    _selection_rank(
                        "molor-router-v1", shard.descriptor.versioned_id, line_number, prompt_digest
                    ),
                    shard.descriptor.versioned_id,
                    ROUTER_SOURCE_PRIORITIES.get(shard.source_id, 50),
                ),
            )
    if digest.hexdigest() != shard.descriptor.sha256 or records != shard.descriptor.record_count:
        raise ValueError(f"chat shard changed while router examples were extracted: {path}")
    return records, truncated_prompts


def _base_router_source(
    connection: sqlite3.Connection,
    report_path: Path,
) -> tuple[dict[str, object], int, int, int]:
    report = _strict_json(report_path)
    outputs = _as_mapping(report.get("outputs"), "router-v6.outputs")
    observed_rows = 0
    admitted_rows = 0
    truncated_prompts = 0
    pins: list[dict[str, object]] = []
    for split in SPLITS:
        output = _as_mapping(outputs.get(split), f"router-v6.outputs.{split}")
        reported_path = Path(_as_string(output.get("path"), "router-v6 output.path")).resolve()
        if reported_path.parent != report_path.resolve().parent:
            raise ValueError("router base output escapes the pinned report directory")
        expected_sha = _digest(output.get("sha256"), "router-v6 output.sha256")
        expected_rows = _as_count(output.get("rows"), "router-v6 output.rows")
        digest = hashlib.sha256()
        rows = 0
        base_rows = 0
        with reported_path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                rows += 1
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid router base row {reported_path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict) or set(value) != {"prompt", "route_id"}:
                    raise ValueError("router base rows must contain exactly prompt and route_id")
                prompt = _as_string(value["prompt"], "router base prompt")
                route = _as_string(value["route_id"], "router base route_id")
                if route != "base/null":
                    continue
                base_rows += 1
                if len(prompt) > ROUTER_MAX_PROMPT_CHARACTERS:
                    retained = ROUTER_MAX_PROMPT_CHARACTERS - len(ROUTER_ELISION_MARKER)
                    prefix = retained // 2
                    prompt = (
                        prompt[:prefix] + ROUTER_ELISION_MARKER + prompt[-(retained - prefix) :]
                    )
                    truncated_prompts += 1
                normalized = _normalize_prompt(prompt)
                prompt_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                connection.execute(
                    """
                    INSERT INTO occurrences(
                        digest, normalized, prompt, route_id, split, selection_rank, origin,
                        route_priority
                    ) VALUES (?, ?, ?, 'base/null', ?, ?, ?, 100)
                    """,
                    (
                        prompt_digest,
                        normalized,
                        prompt,
                        split,
                        _selection_rank("molor-router-base-v1", split, line_number, prompt_digest),
                        "router-bootstrap-v6/base-null-only",
                    ),
                )
        if digest.hexdigest() != expected_sha or rows != expected_rows:
            raise ValueError(f"router base source identity mismatch: {reported_path}")
        observed_rows += rows
        admitted_rows += base_rows
        pins.append(
            {
                "split": split,
                "path": str(reported_path),
                "sha256": expected_sha,
                "records": expected_rows,
                "base_null_records": base_rows,
            }
        )
    return (
        {
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _sha256_file(report_path),
            },
            "outputs": pins,
            "admission": "base-null-rows-only",
        },
        observed_rows,
        admitted_rows,
        truncated_prompts,
    )


def _resolve_router_owners(connection: sqlite3.Connection) -> dict[str, object]:
    """Choose one split owner per normalized prompt with explicit route precedence.

    More specific, curated sources have lower ``route_priority`` values.  A route
    represented at the best priority wins over broader supplemental labels.  If two
    different routes remain tied at that best priority, every occurrence of the
    prompt is excluded.  Split precedence is applied only after a route is resolved,
    preventing a training copy from owning a prompt that is also held out.
    """

    connection.execute(
        """
        CREATE TABLE owners(
            digest TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            route_id TEXT NOT NULL,
            split TEXT NOT NULL,
            selection_rank TEXT NOT NULL
        )
        """
    )
    totals: Counter[str] = Counter()
    route_sets: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    def resolve_group(rows: list[tuple[str, str, str, str, str, str, int]]) -> None:
        if not rows:
            return
        normalized_values = {row[1] for row in rows}
        routes = tuple(sorted({row[3] for row in rows}))
        minimum_priority = min(row[6] for row in rows)
        preferred_routes = {row[3] for row in rows if row[6] == minimum_priority}
        has_digest_collision = len(normalized_values) != 1
        has_route_conflict = len(routes) > 1
        conflict_key = routes

        if has_digest_collision or len(preferred_routes) != 1:
            totals["unresolved_conflict_occurrences"] += len(rows)
            totals["unresolved_conflict_digests"] += 1
            if has_digest_collision:
                totals["digest_collision_digests"] += 1
            detail = route_sets[conflict_key]
            detail["total_occurrences"] += len(rows)
            detail["digests"] += 1
            detail["unresolved_digests"] += 1
            return

        selected_route = next(iter(preferred_routes))
        eligible = [row for row in rows if row[3] == selected_route]
        losing = [row for row in rows if row[3] != selected_route]
        if losing:
            totals["resolved_conflict_losing_occurrences"] += len(losing)
            totals["resolved_conflict_digests"] += 1
            detail = route_sets[conflict_key]
            detail["total_occurrences"] += len(rows)
            detail["digests"] += 1
            detail["resolved_digests"] += 1
            detail["losing_occurrences"] += len(losing)
        elif has_route_conflict:
            raise AssertionError("router conflict accounting lost a route")

        owner = min(
            eligible,
            key=lambda row: (SPLIT_PRECEDENCE[row[4]], row[6], row[5]),
        )
        connection.execute(
            "INSERT INTO owners VALUES (?, ?, ?, ?, ?)",
            (owner[0], owner[2], owner[3], owner[4], owner[5]),
        )
        totals["duplicate_or_cross_split_occurrences"] += len(eligible) - 1
        totals["owners"] += 1

    current_digest: str | None = None
    group: list[tuple[str, str, str, str, str, str, int]] = []
    for raw_row in connection.execute(
        """
        SELECT digest, normalized, prompt, route_id, split, selection_rank, route_priority
        FROM occurrences
        ORDER BY digest, route_priority,
                 CASE split WHEN 'test' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END,
                 selection_rank
        """
    ):
        row = (
            str(raw_row[0]),
            str(raw_row[1]),
            str(raw_row[2]),
            str(raw_row[3]),
            str(raw_row[4]),
            str(raw_row[5]),
            int(raw_row[6]),
        )
        if current_digest is not None and row[0] != current_digest:
            resolve_group(group)
            group = []
        current_digest = row[0]
        group.append(row)
    resolve_group(group)
    connection.commit()

    return {
        **totals,
        "route_conflict_sets": [
            {"routes": list(routes), **dict(sorted(counts.items()))}
            for routes, counts in sorted(route_sets.items())
        ],
    }


def build_router_examples(
    collected: CollectedExpertData,
    *,
    base_report_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Extract balanced deterministic router examples from admitted chat prompts."""

    output_directory.mkdir(parents=True, exist_ok=False)
    database_path = output_directory.parent / f".{output_directory.name}.router-examples.sqlite3"
    if database_path.exists():
        raise FileExistsError(f"router staging database already exists: {database_path}")
    connection = sqlite3.connect(database_path)
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
        chat_prompts_truncated = 0
        for shard in collected.shards:
            shard_rows, shard_truncated = _scan_chat_for_router(connection, shard)
            chat_rows += shard_rows
            chat_prompts_truncated += shard_truncated
        base_identity, base_observed, base_admitted, base_prompts_truncated = _base_router_source(
            connection, base_report_path
        )
        connection.commit()

        owner_resolution = _resolve_router_owners(connection)

        selected_by_split: dict[str, list[tuple[str, str, str]]] = {split: [] for split in SPLITS}
        route_balance: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
        capped_owners = 0
        for split in SPLITS:
            per_route: Counter[str] = Counter()
            candidates = connection.execute(
                """
                SELECT digest, prompt, route_id FROM owners
                WHERE split = ? ORDER BY route_id, selection_rank, digest
                """,
                (split,),
            )
            for digest, prompt, route in candidates:
                route_id = str(route)
                if per_route[route_id] >= ROUTER_RECORD_CAPS[split]:
                    capped_owners += 1
                    continue
                per_route[route_id] += 1
                selected_by_split[split].append((str(digest), str(prompt), route_id))
            route_balance[split] = dict(sorted(per_route.items()))

        expert_routes = sorted(collected.family_by_leaf)
        route_coverage: dict[str, dict[str, object]] = {}
        for split in SPLITS:
            present = sorted(route for route in route_balance[split] if route != "base/null")
            missing = sorted(set(expert_routes).difference(present))
            if "base/null" not in route_balance[split]:
                raise AssertionError(f"router {split} split has no base/null negatives")
            if split in {"train", "validation"} and missing:
                raise AssertionError(
                    f"router {split} split lacks required expert routes: {missing}"
                )
            route_coverage[split] = {
                "expert_routes_present": present,
                "expert_routes_missing": missing,
                "full_expert_route_support": not missing,
                "base_null_support": route_balance[split]["base/null"],
            }

        output_artifacts: dict[str, dict[str, object]] = {}
        selected_rows = 0
        for split in SPLITS:
            rows = sorted(
                selected_by_split[split],
                key=lambda item: (
                    _selection_rank("molor-router-output-v1", split, item[0]),
                    item[0],
                ),
            )
            destination = output_directory / f"router-examples-v1.{split}.jsonl"
            digest = hashlib.sha256()
            with destination.open("wb") as stream:
                for _prompt_digest, prompt, route_id in rows:
                    encoded = canonical_json_bytes({"prompt": prompt, "route_id": route_id}) + b"\n"
                    digest.update(encoded)
                    stream.write(encoded)
            selected_rows += len(rows)
            output_artifacts[split] = {
                "path": destination.name,
                "sha256": digest.hexdigest(),
                "records": len(rows),
                "bytes": destination.stat().st_size,
            }

        total_admitted_occurrences = chat_rows + base_admitted
        unique_owners = int(connection.execute("SELECT COUNT(*) FROM owners").fetchone()[0])
        resolved_conflict_losing_occurrences = int(
            owner_resolution.get("resolved_conflict_losing_occurrences", 0)
        )
        unresolved_conflict_occurrences = int(
            owner_resolution.get("unresolved_conflict_occurrences", 0)
        )
        conflict_occurrences = (
            resolved_conflict_losing_occurrences + unresolved_conflict_occurrences
        )
        duplicate_occurrences = int(owner_resolution.get("duplicate_or_cross_split_occurrences", 0))
        legacy_nonadmitted = base_observed - base_admitted
        if unique_owners != int(owner_resolution.get("owners", 0)):
            raise AssertionError("router resolved-owner accounting failed")
        if unique_owners != selected_rows + capped_owners:
            raise AssertionError("router owner accounting failed")
        if total_admitted_occurrences != (
            conflict_occurrences + duplicate_occurrences + selected_rows + capped_owners
        ):
            raise AssertionError("router occurrence accounting failed")

        manifest = _self_hashed(
            {
                "schema_version": ROUTER_EXAMPLES_MANIFEST_SCHEMA,
                "dataset_id": "molor-hierarchical-router-examples-v1",
                "revision": "1",
                "row_contract": {"exact_keys": ["prompt", "route_id"]},
                "route_contract": {
                    "expert_routes": expert_routes,
                    "negative_route": "base/null",
                    "family_lookup": dict(sorted(collected.family_by_leaf.items())),
                },
                "evaluation_contract": {
                    "train_and_validation": "fail-closed-full-route-support",
                    "test": {
                        "coverage": "partial-route-support-permitted-and-explicit",
                        "unsupported_route_support": 0,
                        "unsupported_route_recall": None,
                        "macro_metrics": "supported-routes-only",
                    },
                    "evidence_scope": (
                        "router-selection evaluation only; not expert-answer-quality evidence"
                    ),
                },
                "split_precedence": list(sorted(SPLIT_PRECEDENCE, key=SPLIT_PRECEDENCE.get)),
                "selection": {
                    "seed": "molor-router-v1",
                    "per_route_caps": ROUTER_RECORD_CAPS,
                    "normalized_prompt": "NFKC+casefold+collapsed-whitespace",
                    "maximum_prompt_characters": ROUTER_MAX_PROMPT_CHARACTERS,
                    "overlength_prompt_policy": "retain-balanced-prefix-and-suffix-with-marker",
                    "cross_split_policy": "test-then-validation-then-train",
                    "route_conflicts": (
                        "lowest-source-priority-route-wins; equal-priority-ambiguity-excluded"
                    ),
                    "source_priorities": {
                        **dict(sorted(ROUTER_SOURCE_PRIORITIES.items())),
                        "unlisted-source-default": 50,
                        "base/null": 100,
                    },
                },
                "sources": {
                    "chat_shards": [
                        {
                            "shard_id": shard.descriptor.shard_id,
                            "shard_revision": shard.descriptor.shard_revision,
                            "sha256": shard.descriptor.sha256,
                            "records": shard.descriptor.record_count,
                        }
                        for shard in collected.shards
                    ],
                    "base_null": base_identity,
                },
                "accounting": {
                    "chat_occurrences": chat_rows,
                    "chat_prompts_length_bounded": chat_prompts_truncated,
                    "base_source_rows_observed": base_observed,
                    "base_null_occurrences_admitted": base_admitted,
                    "base_prompts_length_bounded": base_prompts_truncated,
                    "legacy_expert_rows_not_admitted": legacy_nonadmitted,
                    "route_conflict_occurrences": conflict_occurrences,
                    "resolved_conflict_losing_occurrences": (resolved_conflict_losing_occurrences),
                    "unresolved_conflict_occurrences": unresolved_conflict_occurrences,
                    "resolved_conflict_digests": int(
                        owner_resolution.get("resolved_conflict_digests", 0)
                    ),
                    "unresolved_conflict_digests": int(
                        owner_resolution.get("unresolved_conflict_digests", 0)
                    ),
                    "digest_collision_digests": int(
                        owner_resolution.get("digest_collision_digests", 0)
                    ),
                    "duplicate_or_cross_split_occurrences": duplicate_occurrences,
                    "selected_rows": selected_rows,
                    "capped_unique_prompts": capped_owners,
                    "fully_reconciled": True,
                    "zero_silent_drops": True,
                },
                "route_conflict_sets": owner_resolution["route_conflict_sets"],
                "route_coverage": route_coverage,
                "route_balance": route_balance,
                "normalized_prompt_overlap": {
                    "train_validation": 0,
                    "train_test": 0,
                    "validation_test": 0,
                },
                "outputs": output_artifacts,
            }
        )
        manifest_path = output_directory / "router-examples-v1.manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        return {
            "dataset_root": str(output_directory.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "content_sha256": manifest["content_sha256"],
            "records": {split: output_artifacts[split]["records"] for split in SPLITS},
            "route_balance": route_balance,
        }
    finally:
        connection.close()
        if database_path.exists():
            database_path.unlink()


def _lineage_contract() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aliases = [
        {
            "source_id": "qwen38-distillation-experts-v1",
            "canonical_source_id": "qwen38-distillation-experts-v2",
        },
        {
            "source_id": "software-architecture-v1",
            "canonical_source_id": "software-architecture-experts-v1",
        },
        {
            "source_id": "financial-analysis-verified-v1",
            "canonical_source_id": "business-finance-length-tiers-v1",
        },
        {
            "source_id": "financial-analysis-verified",
            "canonical_source_id": "business-finance-length-tiers-v1",
        },
        {
            "source_id": "taskmaster-coffee-api-v1",
            "canonical_source_id": "agentic-tool-length-tiers-v1",
        },
        {
            "source_id": "taskmaster-coffee-api-v2-repo-local",
            "canonical_source_id": "agentic-tool-length-tiers-v1",
        },
        {
            "source_id": "terminal-cli-v1-repo-local",
            "canonical_source_id": "agentic-tool-length-tiers-v1",
        },
        {
            "source_id": "terminal-cli-v1",
            "canonical_source_id": "agentic-tool-length-tiers-v1",
        },
        {
            "source_id": "embedded-knowledge-v1-repo-local",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "embedded-knowledge-v1",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "ocb-can-verified-v1-repo-local",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "ocb-can-verified-v1",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "trusted-can-v1-repo-local",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "trusted-can-v1",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "grounded-can-v1-repo-local",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "grounded-can-v1",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "grounded-can-curriculum-v1-repo-local",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "grounded-can-curriculum-v1",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "verified-embedded-code-v1-v2-v3-repo-local",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "verified-embedded-code-v1",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "verified-embedded-code-v2",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "verified-embedded-code-v3",
            "canonical_source_id": "embedded-source-experts-v1",
        },
        {
            "source_id": "verified-embedded-code-v1-plus-v3",
            "canonical_source_id": "embedded-source-experts-v1",
        },
    ]
    nonadmission = [
        {
            "source_id": item["source_id"],
            "disposition": "not-admitted",
            "reason": "lineage-alias-of-canonical-source",
        }
        for item in aliases
    ]
    nonadmission.extend(
        [
            {
                "source_id": "customer-gtm-v1",
                "disposition": "quarantined-pending-audit",
                "reason": "privacy-redaction-and-split-audit-not-complete",
            },
        ]
    )
    return aliases, nonadmission


def _leaf_readiness(
    collected: CollectedExpertData, recipes: Mapping[str, CorpusMixtureRecipe]
) -> list[dict[str, object]]:
    references_by_leaf: dict[str, list[RegisteredReferenceShard]] = defaultdict(list)
    for reference in collected.references:
        references_by_leaf[reference.leaf_id].append(reference)
    streams_by_leaf: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    chat_sources_by_leaf: dict[str, set[str]] = defaultdict(set)
    for shard in collected.shards:
        streams_by_leaf[shard.descriptor.leaf_id][shard.recipe_stream].add(shard.descriptor.split)
        chat_sources_by_leaf[shard.descriptor.leaf_id].add(shard.source_id)
    recipe_paths_by_leaf: dict[str, list[str]] = defaultdict(list)
    for path, recipe in recipes.items():
        leaf_ids = {shard.descriptor.leaf_id for shard in recipe.shards}
        if len(leaf_ids) != 1:
            raise AssertionError("canonical per-leaf recipe crossed leaf boundaries")
        recipe_paths_by_leaf[next(iter(leaf_ids))].append(path)

    leaves: list[dict[str, object]] = []
    for leaf in sorted(collected.family_by_leaf):
        stream_splits = {
            stream: sorted(splits, key=SPLITS.index)
            for stream, splits in sorted(streams_by_leaf.get(leaf, {}).items())
        }
        training_ready_streams = [
            stream
            for stream, splits in stream_splits.items()
            if {"train", "validation"}.issubset(splits)
        ]
        full_three_way_streams = [
            stream for stream, splits in stream_splits.items() if set(SPLITS).issubset(splits)
        ]
        references = references_by_leaf.get(leaf, [])
        admitted_sources = sorted(chat_sources_by_leaf.get(leaf, set()))
        reconstruction_only_sources = {
            "business-grounded-reconstruction-v1",
            "embedded-grounded-reconstruction-v1",
        }
        if (
            training_ready_streams
            and len(admitted_sources) == 1
            and admitted_sources[0] in reconstruction_only_sources
        ):
            status = "chat-ready-limited-grounded-reconstruction"
        elif training_ready_streams:
            status = "chat-ready"
        elif references:
            status = "reference-only-chat-gap"
        else:
            status = "no-admitted-data-gap"
        leaves.append(
            {
                "leaf_id": leaf,
                "family_id": collected.family_by_leaf[leaf],
                "status": status,
                "stream_splits": stream_splits,
                "training_ready_streams": sorted(training_ready_streams),
                "full_three_way_streams": sorted(full_three_way_streams),
                "reference_records": sum(item.record_count for item in references),
                "admitted_chat_source_ids": admitted_sources,
                "recipe_paths": sorted(recipe_paths_by_leaf.get(leaf, [])),
            }
        )
    return leaves


def write_expert_mixture_bundle(
    collected: CollectedExpertData,
    *,
    recipes: Mapping[str, CorpusMixtureRecipe],
    router_base_report: Path,
    output_directory: str | Path,
    router_output_directory: str | Path | None = None,
    builder_paths: Sequence[Path] = (),
) -> dict[str, object]:
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    external_router: Path | None = None
    external_router_created = False
    try:
        assert stage is not None
        recipe_records: list[dict[str, object]] = []
        for relative, recipe in sorted(recipes.items()):
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = corpus_mixture_recipe_json_bytes(recipe)
            destination.write_bytes(payload)
            leaf_ids = {item.descriptor.leaf_id for item in recipe.shards}
            streams = {
                shard.semantic_stream
                for shard in collected.shards
                if shard.descriptor.shard_id in {item.descriptor.shard_id for item in recipe.shards}
            }
            recipe_records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "content_sha256": recipe.fingerprint,
                    "recipe_id": recipe.recipe_id,
                    "leaf_id": next(iter(leaf_ids)),
                    "semantic_stream": next(iter(streams)),
                    "recipe_stream": relative.split("/")[3],
                    "variant": relative.split("/")[1],
                    "split": recipe.output_split,
                    "usage": recipe.usage,
                    "target_records": recipe.target_records,
                }
            )

        if router_output_directory is None:
            router_destination = stage / "router"
        else:
            external_router = Path(router_output_directory).resolve()
            if external_router.exists():
                raise FileExistsError(f"router output directory already exists: {external_router}")
            external_router.parent.mkdir(parents=True, exist_ok=True)
            router_destination = external_router
            external_router_created = True
        router_summary = build_router_examples(
            collected,
            base_report_path=router_base_report,
            output_directory=router_destination,
        )
        router_manifest = router_destination / "router-examples-v1.manifest.json"
        if external_router is None:
            router_summary["dataset_root"] = str((output / "router").resolve())
            router_summary["manifest_path"] = str(
                (output / "router" / "router-examples-v1.manifest.json").resolve()
            )
            router_summary["registry_manifest_copy_path"] = (
                "router/router-examples-v1.manifest.json"
            )
        else:
            registry_router_manifest = stage / "router-examples-v1.manifest.json"
            shutil.copyfile(router_manifest, registry_router_manifest)
            router_summary["registry_manifest_copy_path"] = "router-examples-v1.manifest.json"
        aliases, nonadmission = _lineage_contract()
        broad_source = next(
            (
                source
                for source in collected.sources
                if source["source_id"] == "broad-distillation-experts-v1"
            ),
            None,
        )
        structured_tool_audit = (
            _as_mapping(
                broad_source.get("training_contract_audit"),
                "broad source structured-tool audit",
            )
            if broad_source is not None
            else {}
        )
        leaves = _leaf_readiness(collected, recipes)
        ready_leaves = [
            item["leaf_id"] for item in leaves if str(item["status"]).startswith("chat-ready")
        ]
        gap_leaves = [
            item["leaf_id"] for item in leaves if not str(item["status"]).startswith("chat-ready")
        ]
        registry = _self_hashed(
            {
                "schema_version": EXPERT_CHAT_REGISTRY_SCHEMA,
                "registry_id": "molor-expert-rack-v5-chat-mixtures",
                "revision": "1",
                "taxonomy": {
                    "path": collected.taxonomy_path,
                    "file_sha256": collected.taxonomy_sha256,
                    "content_sha256": collected.taxonomy["content_sha256"],
                    "leaf_revision": collected.leaf_revision,
                },
                "contracts": {
                    "chat_rows": "top-level object contains exactly messages",
                    "recipe_schema": "molor.corpus-mixture-recipe.v1",
                    "global_leakage_guards": {
                        "training": "all validation and test chat shards across every leaf",
                        "evaluation": "all train chat shards across every leaf",
                    },
                    "semantic_streams": {
                        "direct": "direct assistant answer",
                        "trace": "assistant answer with admitted reasoning trace",
                        "tool": (
                            "tool-use supervision; native tool-direct/tool-trace and assistant-form "
                            "Taskmaster rows remain separate recipe streams"
                        ),
                    },
                    "adjustment": "repeatable leaf:ID and shard:ID weight overrides",
                    "full_data": "all manifest-admitted rows before exact dedup/leakage quarantine",
                    "fast_iteration": {
                        split: {
                            stream: FAST_RECORD_BUDGETS[(split, stream)]
                            for stream in SEMANTIC_STREAMS
                        }
                        for split in SPLITS
                    },
                },
                "source_semantics": {
                    "business-grounded-reconstruction-v1": {
                        "task_type": "grounded_reference_reconstruction",
                        "assistant_target": "verbatim canonical source passage",
                        "contains_synthetic_question_answering": False,
                        "proves_answer_quality": False,
                        "use": (
                            "grounding/reconstruction supervision; do not represent it as broad "
                            "business question-answering coverage"
                        ),
                    },
                    "embedded-grounded-reconstruction-v1": {
                        "task_type": "grounded_reference_reconstruction",
                        "input_modality": "reference-cpt",
                        "assistant_target": "contiguous verbatim source-text chunk",
                        "contains_synthetic_question_answering": False,
                        "proves_answer_quality": False,
                        "proves_source_correctness": False,
                        "lineage_parent": "embedded-source-experts-v1/reference",
                        "use": (
                            "domain-adaptation reconstruction supervision; do not represent it "
                            "as embedded question-answering or code-generation quality evidence"
                        ),
                    },
                    "agentic-tool-length-tiers-v1/taskmaster": {
                        "semantic_stream": "tool",
                        "render_template_mode": "direct-v1",
                        "target": "assistant-form tool call supervision",
                        "contains_native_tool_role_observations": False,
                    },
                    "broad-distillation-experts-v1/native-tool": {
                        "semantic_stream": "tool",
                        "recipe_streams": ["tool-direct", "tool-trace"],
                        "render_template_modes": ["tool-direct-v1", "tool-trace-v1"],
                        "preserves": [
                            "assistant.tool_calls",
                            "tool.role",
                            "tool.tool_call_id",
                            "empty tool results",
                        ],
                        "trainer_contract": "strict-lossless-qwen-compatible",
                        "pre_assistant_truncated_observation_window": (
                            "one contiguous initial tool-result block is explicitly admitted"
                        ),
                        "contract_audit": dict(structured_tool_audit),
                    },
                },
                "source_registry": list(collected.sources),
                "chat_shards": [item.to_dict() for item in collected.shards],
                "reference_and_cpt_shards": [item.to_dict() for item in collected.references],
                "manifest_output_dispositions": list(collected.dispositions),
                "lineage_aliases": aliases,
                "nonadmission": nonadmission,
                "leaf_readiness": leaves,
                "summary": {
                    "taxonomy_leaves": len(collected.family_by_leaf),
                    "chat_ready_leaves": len(ready_leaves),
                    "gap_or_reference_only_leaves": len(gap_leaves),
                    "ready_leaf_ids": ready_leaves,
                    "gap_leaf_ids": gap_leaves,
                    "chat_shards": len(collected.shards),
                    "chat_records_declared": sum(
                        item.descriptor.record_count for item in collected.shards
                    ),
                    "reference_shards": len(collected.references),
                    "reference_records_declared": sum(
                        item.record_count for item in collected.references
                    ),
                    "recipes": len(recipes),
                    "zero_silent_source_output_omissions": True,
                },
                "recipes": recipe_records,
                "router_examples": router_summary,
                "completed_source_hooks": [
                    {
                        "hook_id": "embedded-grounded-reconstruction-v1",
                        "admitted_leaves": [
                            "embedded-code-generation",
                            "embedded-knowledge",
                        ],
                        "splits": list(SPLITS),
                        "chat_contract": "messages-only direct-v1",
                        "admission_state": (
                            "independently-rebuilt-registration-fragment-and-shards-pinned"
                        ),
                        "lineage_policy": (
                            "derived chat reconstruction is trainable; parent reference/CPT rows "
                            "remain non-chat and are not sampled into chat recipes"
                        ),
                    }
                ],
                "known_evaluation_limitations": [
                    {
                        "source_id": "qwen38-distillation-experts-v2",
                        "limitation": (
                            "public benchmark-family supervision is training data; scores for "
                            "matching families must be reported as contaminated"
                        ),
                    },
                    {
                        "dataset_id": "molor-hierarchical-router-examples-v1",
                        "split": "test",
                        "route_id": "terminal-cli",
                        "support": 0,
                        "recall": None,
                        "macro_metric_policy": "supported-routes-only",
                        "limitation": (
                            "the canonical terminal source has no test shard; no train/validation "
                            "prompt was resplit or synthesized to conceal the missing support"
                        ),
                    },
                ],
            }
        )
        registry_path = stage / "shard-registry.json"
        registry_path.write_bytes(_json_bytes(registry))

        builder_pins = [
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(set(builder_paths), key=lambda item: str(item))
        ]
        receipt = _self_hashed(
            {
                "schema_version": EXPERT_CHAT_RECEIPT_SCHEMA,
                "bundle_id": "molor-expert-rack-v5-chat-mixtures-v1",
                "revision": "1",
                "registry": {
                    "path": "shard-registry.json",
                    "sha256": _sha256_file(registry_path),
                    "content_sha256": registry["content_sha256"],
                },
                "recipes": {
                    "count": len(recipe_records),
                    "aggregate_sha256": fingerprint_json(
                        [
                            {"path": item["path"], "sha256": item["sha256"]}
                            for item in recipe_records
                        ]
                    ),
                },
                "router_examples": router_summary,
                "builders": builder_pins,
                "accounting": {
                    "ready_leaves": len(ready_leaves),
                    "gap_leaves": len(gap_leaves),
                    "source_manifests": sum(
                        len(_as_list(item["manifest_pins"], "source manifest pins"))
                        for item in collected.sources
                    ),
                    "admission_receipts": sum(
                        int("registration_fragment" in item)
                        + int("training_contract_audit" in item)
                        for item in collected.sources
                    ),
                    "chat_shards": len(collected.shards),
                    "reference_shards": len(collected.references),
                    "manifest_output_dispositions": len(collected.dispositions),
                    "fully_reconciled": True,
                    "zero_silent_drops": True,
                },
            }
        )
        receipt_path = stage / "generation-receipt.json"
        receipt_path.write_bytes(_json_bytes(receipt))

        sums: list[tuple[str, str]] = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "sha256sums.txt":
                sums.append((path.relative_to(stage).as_posix(), _sha256_file(path)))
        (stage / "sha256sums.txt").write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in sums),
            encoding="utf-8",
            newline="",
        )
        os.replace(stage, output)
        stage = None
        external_router_created = False
        return {
            "output_directory": str(output),
            "registry_sha256": _sha256_file(output / "shard-registry.json"),
            "registry_content_sha256": registry["content_sha256"],
            "receipt_sha256": _sha256_file(output / "generation-receipt.json"),
            "recipes": len(recipe_records),
            "ready_leaves": ready_leaves,
            "gap_leaves": gap_leaves,
            "router_records": router_summary["records"],
        }
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if external_router_created and external_router is not None and external_router.exists():
            shutil.rmtree(external_router)


def build_expert_mixture_bundle(
    paths: ExpertMixtureSourcePaths,
    *,
    output_directory: str | Path,
    router_output_directory: str | Path | None = None,
    verify_files: bool = True,
    builder_paths: Sequence[Path] = (),
) -> dict[str, object]:
    collected = collect_expert_data(paths, verify_files=verify_files)
    recipes = build_canonical_recipes(collected)
    return write_expert_mixture_bundle(
        collected,
        recipes=recipes,
        router_base_report=paths.router_base_report_v6,
        output_directory=output_directory,
        router_output_directory=router_output_directory,
        builder_paths=builder_paths,
    )


__all__ = [
    "EXPERT_CHAT_RECEIPT_SCHEMA",
    "EXPERT_CHAT_REGISTRY_SCHEMA",
    "FAST_RECORD_BUDGETS",
    "ROUTER_EXAMPLES_MANIFEST_SCHEMA",
    "ROUTER_MAX_PROMPT_CHARACTERS",
    "ROUTER_RECORD_CAPS",
    "CollectedExpertData",
    "ExpertMixtureSourcePaths",
    "RegisteredChatShard",
    "RegisteredReferenceShard",
    "build_canonical_recipes",
    "build_expert_mixture_bundle",
    "build_router_examples",
    "collect_expert_data",
    "write_expert_mixture_bundle",
]
