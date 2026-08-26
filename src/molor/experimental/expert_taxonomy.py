"""Strict, content-addressed taxonomy for hierarchical expert routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from molor.identity import fingerprint_json, validate_sha256


EXPERT_TAXONOMY_SCHEMA = "molor.expert-taxonomy.v1"
_STATUSES = frozenset(
    {
        "candidate_existing",
        "materializing",
        "planned",
        "reference_ready",
        "supervised_ready",
    }
)
_STREAMS = frozenset({"direct", "tool", "trace"})


def _exact(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{field} fields differ from v1 "
            f"(missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)})"
        )


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{field} must be a non-empty string of at most 128 characters")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-._/" for character in value):
        raise ValueError(f"{field} contains unsupported characters")
    return value


def _strings(value: object, field: str, *, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{field} must be an array with at least {minimum} item(s)")
    result = tuple(_identifier(item, f"{field}[]") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains unsupported numeric constant {value!r}")


@dataclass(frozen=True, slots=True)
class ExpertLeaf:
    leaf_id: str
    description: str
    status: str
    allowed_streams: tuple[str, ...]
    source_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpertFamily:
    family_id: str
    description: str
    leaves: tuple[ExpertLeaf, ...]


@dataclass(frozen=True, slots=True)
class ExpertTaxonomy:
    taxonomy_id: str
    revision: str
    base_contract: Mapping[str, object]
    routing_contract: Mapping[str, object]
    families: tuple[ExpertFamily, ...]
    aliases: Mapping[str, str]
    superseded_composite_labels: Mapping[str, tuple[str, ...]]
    external_opaque_import: Mapping[str, object]
    fingerprint: str

    @property
    def leaves(self) -> tuple[ExpertLeaf, ...]:
        return tuple(leaf for family in self.families for leaf in family.leaves)

    @property
    def leaf_ids(self) -> tuple[str, ...]:
        return tuple(leaf.leaf_id for leaf in self.leaves)

    def resolve_leaf(self, value: str) -> str:
        if value in self.leaf_ids:
            return value
        try:
            return self.aliases[value]
        except KeyError as exc:
            raise KeyError(f"unknown expert leaf or alias: {value!r}") from exc


def _parse_leaf(value: object, field: str) -> ExpertLeaf:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    _exact(
        value,
        {"leaf_id", "description", "status", "allowed_streams", "source_labels"},
        field,
    )
    leaf_id = _identifier(value["leaf_id"], f"{field}.leaf_id")
    description = value["description"]
    status = value["status"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{field}.description must be non-empty")
    if status not in _STATUSES:
        raise ValueError(f"{field}.status must be one of {sorted(_STATUSES)}")
    streams = _strings(value["allowed_streams"], f"{field}.allowed_streams")
    if not set(streams).issubset(_STREAMS):
        raise ValueError(f"{field}.allowed_streams contains an unsupported stream")
    labels = _strings(value["source_labels"], f"{field}.source_labels")
    if list(streams) != sorted(streams) or list(labels) != sorted(labels):
        raise ValueError(f"{field} arrays must use canonical lexical ordering")
    return ExpertLeaf(leaf_id, description.strip(), str(status), streams, labels)


def _parse_family(value: object, field: str) -> ExpertFamily:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    _exact(value, {"family_id", "description", "leaves"}, field)
    family_id = _identifier(value["family_id"], f"{field}.family_id")
    description = value["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{field}.description must be non-empty")
    raw_leaves = value["leaves"]
    if not isinstance(raw_leaves, list) or not raw_leaves:
        raise ValueError(f"{field}.leaves must be a non-empty array")
    leaves = tuple(
        _parse_leaf(item, f"{field}.leaves[{index}]") for index, item in enumerate(raw_leaves)
    )
    leaf_ids = [leaf.leaf_id for leaf in leaves]
    if leaf_ids != sorted(leaf_ids) or len(leaf_ids) != len(set(leaf_ids)):
        raise ValueError(f"{field}.leaves must be unique and ordered by leaf_id")
    return ExpertFamily(family_id, description.strip(), leaves)


def load_expert_taxonomy(path: str | Path) -> ExpertTaxonomy:
    """Load and fully validate one immutable taxonomy artifact."""

    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("taxonomy must be a regular non-symlink file")
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != before.st_size
    ):
        raise RuntimeError("taxonomy changed while it was read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("taxonomy must be strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("taxonomy must contain one JSON object")
    _exact(
        value,
        {
            "schema_version",
            "taxonomy_id",
            "revision",
            "base_contract",
            "routing_contract",
            "families",
            "aliases",
            "superseded_composite_labels",
            "external_opaque_import",
            "content_sha256",
        },
        "taxonomy",
    )
    if value["schema_version"] != EXPERT_TAXONOMY_SCHEMA:
        raise ValueError("unsupported expert-taxonomy schema")

    base = value["base_contract"]
    if not isinstance(base, Mapping):
        raise TypeError("base_contract must be an object")
    _exact(
        base,
        {"model_id", "revision", "weights_sha256", "leaf_training_origin"},
        "base_contract",
    )
    _identifier(base["model_id"], "base_contract.model_id")
    _identifier(base["revision"], "base_contract.revision")
    validate_sha256(base["weights_sha256"], "base_contract.weights_sha256")  # type: ignore[arg-type]
    if base["leaf_training_origin"] != "independent_untouched_base":
        raise ValueError("every leaf must train independently from the untouched base")

    routing = value["routing_contract"]
    if not isinstance(routing, Mapping):
        raise TypeError("routing_contract must be an object")
    _exact(
        routing,
        {
            "root_routes",
            "decision_path",
            "joint_probability",
            "posterior_order",
            "blend_weight_semantics",
            "maximum_active_adapters",
            "fallback_route",
            "calibration_metrics",
        },
        "routing_contract",
    )
    if routing["decision_path"] != "root_family_then_conditional_leaf":
        raise ValueError("routing decision_path is unsupported")
    if routing["joint_probability"] != "p_family_times_p_leaf_given_family":
        raise ValueError("routing joint_probability is unsupported")
    if routing["posterior_order"] != "strict_descending_probability":
        raise ValueError("routing posterior_order must be probability ranked")
    if routing["blend_weight_semantics"] != "separate_normalized_adapter_contribution":
        raise ValueError("posterior probabilities may not be reused as blend weights")
    if routing["fallback_route"] != "base/null":
        raise ValueError("routing fallback must be base/null")
    if routing["maximum_active_adapters"] != 2:
        raise ValueError("v1 supports at most two active adapters")
    metrics = _strings(routing["calibration_metrics"], "routing_contract.calibration_metrics")
    if list(metrics) != sorted(metrics):
        raise ValueError("calibration_metrics must use canonical lexical ordering")

    raw_families = value["families"]
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("families must be a non-empty array")
    families = tuple(
        _parse_family(item, f"families[{index}]") for index, item in enumerate(raw_families)
    )
    family_ids = [family.family_id for family in families]
    if family_ids != sorted(family_ids) or len(family_ids) != len(set(family_ids)):
        raise ValueError("families must be unique and ordered by family_id")
    leaf_ids = [leaf.leaf_id for family in families for leaf in family.leaves]
    if len(leaf_ids) != len(set(leaf_ids)):
        raise ValueError("leaf IDs must be globally unique")
    root_routes = _strings(routing["root_routes"], "routing_contract.root_routes")
    if root_routes != ("base/null", *family_ids):
        raise ValueError("root_routes must be base/null followed by every canonical family")

    raw_aliases = value["aliases"]
    if not isinstance(raw_aliases, Mapping):
        raise TypeError("aliases must be an object")
    aliases: dict[str, str] = {}
    for alias, target in raw_aliases.items():
        alias_id = _identifier(alias, "aliases key")
        target_id = _identifier(target, f"aliases[{alias!r}]")
        if alias_id in leaf_ids or target_id not in leaf_ids:
            raise ValueError("aliases must be non-canonical names targeting canonical leaves")
        aliases[alias_id] = target_id
    if list(raw_aliases) != sorted(raw_aliases):
        raise ValueError("aliases must use canonical lexical key order")

    raw_composites = value["superseded_composite_labels"]
    if not isinstance(raw_composites, Mapping):
        raise TypeError("superseded_composite_labels must be an object")
    composites: dict[str, tuple[str, ...]] = {}
    for label, targets in raw_composites.items():
        label_id = _identifier(label, "superseded composite label")
        parsed_targets = _strings(targets, f"superseded_composite_labels[{label!r}]", minimum=2)
        if any(target not in leaf_ids for target in parsed_targets):
            raise ValueError("composite labels must split into canonical leaves")
        composites[label_id] = parsed_targets
    if list(raw_composites) != sorted(raw_composites):
        raise ValueError("superseded composite labels must use canonical lexical key order")

    opaque = value["external_opaque_import"]
    if not isinstance(opaque, Mapping):
        raise TypeError("external_opaque_import must be an object")
    _exact(
        opaque,
        {"family_id", "registration_status", "row_access", "required_artifacts"},
        "external_opaque_import",
    )
    _identifier(opaque["family_id"], "external_opaque_import.family_id")
    if opaque["registration_status"] != "pending_aggregate_validation":
        raise ValueError("opaque family must remain pending until aggregate validation")
    if opaque["row_access"] != "prohibited":
        raise ValueError("opaque corpus row access must remain prohibited")
    required = _strings(opaque["required_artifacts"], "external_opaque_import.required_artifacts")
    if list(required) != sorted(required):
        raise ValueError("opaque required_artifacts must use canonical lexical ordering")

    claimed = validate_sha256(value["content_sha256"], "content_sha256")  # type: ignore[arg-type]
    if value["content_sha256"] != claimed:
        raise ValueError("content_sha256 must use lowercase hexadecimal")
    semantic = {key: item for key, item in value.items() if key != "content_sha256"}
    fingerprint = fingerprint_json(semantic)
    if fingerprint != claimed:
        raise ValueError("taxonomy content_sha256 does not match its semantic payload")

    return ExpertTaxonomy(
        taxonomy_id=_identifier(value["taxonomy_id"], "taxonomy_id"),
        revision=_identifier(value["revision"], "revision"),
        base_contract=MappingProxyType(dict(base)),
        routing_contract=MappingProxyType(dict(routing)),
        families=families,
        aliases=MappingProxyType(aliases),
        superseded_composite_labels=MappingProxyType(composites),
        external_opaque_import=MappingProxyType(dict(opaque)),
        fingerprint=fingerprint,
    )


__all__ = [
    "EXPERT_TAXONOMY_SCHEMA",
    "ExpertFamily",
    "ExpertLeaf",
    "ExpertTaxonomy",
    "load_expert_taxonomy",
]
