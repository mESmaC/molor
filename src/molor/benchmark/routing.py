"""Content-pinned, taxonomy-aware routing evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from molor.benchmark.models import BenchmarkSuite, canonical_json
from molor.experimental.expert_taxonomy import ExpertTaxonomy


ROUTE_CONTRACT_SCHEMA = "molor.benchmark-route-contract.v1"
ROUTING_EVALUATION_SCHEMA = "molor.benchmark-routing-evaluation.v1"
ROUTE_PROBABILITY_AUDIT_SCHEMA = "molor.benchmark-route-probability-audit.v1"
BASE_ROUTE = "base/null"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"route contract contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"route contract contains unsupported constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("route contract must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("route contract must contain a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = expected.difference(value)
    unknown = set(value).difference(expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields differ from the schema "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
        )


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded benchmark identifier")
    return value


def _identity(
    value: object,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    _exact_keys(value, expected, label)
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class RouteEvaluationContract:
    """Canonical acceptable routes for every case in one immutable suite/taxonomy pair."""

    source: Path
    contract_id: str
    revision: str
    suite_id: str
    suite_revision: str
    suite_fingerprint: str
    taxonomy_id: str
    taxonomy_revision: str
    taxonomy_fingerprint: str
    case_routes: Mapping[str, tuple[str, ...]]
    content_sha256: str
    canonical_leaf_ids: frozenset[str]
    aliases: Mapping[str, str]

    def validate_suite(self, suite: BenchmarkSuite) -> None:
        if (
            self.suite_id != suite.suite_id
            or self.suite_revision != suite.version
            or self.suite_fingerprint != suite.fingerprint
            or set(self.case_routes) != {case.case_id for case in suite.cases}
        ):
            raise ValueError("routing evaluation contract targets a different benchmark suite")

    def canonicalize_selected(self, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("selected route must be a non-empty string")
        if value in {"base", BASE_ROUTE}:
            return BASE_ROUTE
        if value in self.canonical_leaf_ids:
            return value
        target = self.aliases.get(value)
        if target is None:
            raise ValueError(f"selected route {value!r} is absent from the pinned routing taxonomy")
        return target

    def acceptable_routes(self, case_id: str) -> tuple[str, ...]:
        try:
            return self.case_routes[case_id]
        except KeyError as exc:
            raise ValueError(f"routing contract has no case {case_id!r}") from exc

    def provenance(self) -> dict[str, object]:
        semantic = {
            "schema_version": ROUTING_EVALUATION_SCHEMA,
            "contract_id": self.contract_id,
            "revision": self.revision,
            "suite": {
                "suite_id": self.suite_id,
                "version": self.suite_revision,
                "content_sha256": self.suite_fingerprint,
            },
            "taxonomy": {
                "taxonomy_id": self.taxonomy_id,
                "revision": self.taxonomy_revision,
                "content_sha256": self.taxonomy_fingerprint,
            },
            "case_count": len(self.case_routes),
            "route_contract_content_sha256": self.content_sha256,
        }
        return {
            **semantic,
            "evaluation_fingerprint": hashlib.sha256(
                canonical_json(semantic).encode("utf-8")
            ).hexdigest(),
        }


def load_route_evaluation_contract(
    path: str | Path,
    *,
    suite: BenchmarkSuite,
    taxonomy: ExpertTaxonomy,
) -> RouteEvaluationContract:
    """Load a fully content-pinned per-case route contract and fail closed on drift."""

    unresolved_source = Path(path).expanduser()
    if unresolved_source.is_symlink() or not unresolved_source.is_file():
        raise FileNotFoundError(f"route contract must be a regular non-symlink file: {path}")
    source = unresolved_source.resolve()
    before = source.stat()
    payload = _strict_json_object(source)
    after = source.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise OSError(f"route contract changed while it was being read: {source}")
    _exact_keys(
        payload,
        {
            "schema_version",
            "contract_id",
            "revision",
            "suite",
            "taxonomy",
            "case_routes",
            "content_sha256",
        },
        "route contract",
    )
    if payload["schema_version"] != ROUTE_CONTRACT_SCHEMA:
        raise ValueError(f"unsupported route contract schema: {payload['schema_version']!r}")
    contract_id = _identifier(payload["contract_id"], "contract_id")
    revision = _identifier(payload["revision"], "revision")

    raw_suite = _identity(
        payload["suite"],
        {"suite_id", "version", "content_sha256"},
        "suite",
    )
    suite_id = _identifier(raw_suite["suite_id"], "suite.suite_id")
    suite_revision = _identifier(raw_suite["version"], "suite.version")
    suite_fingerprint = _sha256(raw_suite["content_sha256"], "suite.content_sha256")
    if (
        suite_id != suite.suite_id
        or suite_revision != suite.version
        or suite_fingerprint != suite.fingerprint
    ):
        raise ValueError("route contract targets a different benchmark suite")

    raw_taxonomy = _identity(
        payload["taxonomy"],
        {"taxonomy_id", "revision", "content_sha256"},
        "taxonomy",
    )
    taxonomy_id = _identifier(raw_taxonomy["taxonomy_id"], "taxonomy.taxonomy_id")
    taxonomy_revision = _identifier(raw_taxonomy["revision"], "taxonomy.revision")
    taxonomy_fingerprint = _sha256(raw_taxonomy["content_sha256"], "taxonomy.content_sha256")
    if (
        taxonomy_id != taxonomy.taxonomy_id
        or taxonomy_revision != taxonomy.revision
        or taxonomy_fingerprint != taxonomy.fingerprint
    ):
        raise ValueError("route contract targets a different expert taxonomy")

    raw_case_routes = payload["case_routes"]
    if not isinstance(raw_case_routes, Mapping):
        raise ValueError("case_routes must be an object")
    suite_case_ids = tuple(case.case_id for case in suite.cases)
    missing = set(suite_case_ids).difference(raw_case_routes)
    unknown = set(raw_case_routes).difference(suite_case_ids)
    if missing or unknown:
        raise ValueError(
            "route contract must cover the full frozen suite exactly "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
        )
    canonical_leaf_ids = frozenset(taxonomy.leaf_ids)
    case_routes: dict[str, tuple[str, ...]] = {}
    for case_id in suite_case_ids:
        raw_routes = raw_case_routes[case_id]
        if not isinstance(raw_routes, list) or not raw_routes:
            raise ValueError(f"case_routes[{case_id!r}] must be a non-empty array")
        if any(not isinstance(route, str) or not route for route in raw_routes):
            raise ValueError(f"case_routes[{case_id!r}] must contain non-empty strings")
        routes = tuple(raw_routes)
        if len(routes) != len(set(routes)):
            raise ValueError(f"case_routes[{case_id!r}] must not contain duplicates")
        for route in routes:
            if route == BASE_ROUTE or route in canonical_leaf_ids:
                continue
            alias = taxonomy.aliases.get(route)
            if alias is not None:
                raise ValueError(
                    f"case_routes[{case_id!r}] uses alias {route!r}; use canonical {alias!r}"
                )
            raise ValueError(
                f"case_routes[{case_id!r}] contains unresolved non-base route {route!r}"
            )
        case_routes[case_id] = routes

    claimed = _sha256(payload["content_sha256"], "content_sha256")
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed = hashlib.sha256(canonical_json(semantic).encode("utf-8")).hexdigest()
    if claimed != computed:
        raise ValueError("route contract semantic fingerprint drift")
    return RouteEvaluationContract(
        source=source,
        contract_id=contract_id,
        revision=revision,
        suite_id=suite_id,
        suite_revision=suite_revision,
        suite_fingerprint=suite_fingerprint,
        taxonomy_id=taxonomy_id,
        taxonomy_revision=taxonomy_revision,
        taxonomy_fingerprint=taxonomy_fingerprint,
        case_routes=MappingProxyType(case_routes),
        content_sha256=claimed,
        canonical_leaf_ids=canonical_leaf_ids,
        aliases=MappingProxyType(dict(taxonomy.aliases)),
    )


def evaluate_route_probabilities(
    suite: BenchmarkSuite,
    contract: RouteEvaluationContract,
    probability_rows: Mapping[str, Mapping[str, float]],
    *,
    route_order: Sequence[str],
) -> dict[str, object]:
    """Score an exact per-case probability table without running model generation.

    This is deliberately a top-1 routing audit, not an answer-quality metric.  It lets a
    checkpoint fail a frozen route contract before an expensive generation matrix starts.
    Prompts are never copied into the returned receipt.
    """

    contract.validate_suite(suite)
    if isinstance(route_order, (str, bytes)):
        raise TypeError("route_order must be a sequence of canonical route IDs")
    routes = tuple(route_order)
    expected_routes = {BASE_ROUTE, *contract.canonical_leaf_ids}
    if (
        not routes
        or routes[0] != BASE_ROUTE
        or len(routes) != len(set(routes))
        or set(routes) != expected_routes
    ):
        raise ValueError("route_order must be base/null followed by every canonical leaf exactly")
    if not isinstance(probability_rows, Mapping):
        raise TypeError("probability_rows must map case IDs to route probabilities")
    case_ids = tuple(case.case_id for case in suite.cases)
    missing = set(case_ids).difference(probability_rows)
    unknown = set(probability_rows).difference(case_ids)
    if missing or unknown:
        raise ValueError(
            "probability rows must cover the frozen suite exactly "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
        )

    route_counts = {route_id: 0 for route_id in routes}
    category_rows: dict[str, list[bool]] = {category: [] for category in suite.category_weights}
    predictions: list[dict[str, object]] = []
    acceptable_count = 0
    base_selected_count = 0
    specialist_count = 0
    specialist_acceptable_count = 0
    base_contract_count = 0
    base_contract_acceptable_count = 0
    for case in suite.cases:
        raw_row = probability_rows[case.case_id]
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_routes:
            raise ValueError(
                f"probabilities for {case.case_id!r} must cover every canonical route exactly"
            )
        values: dict[str, float] = {}
        for route_id in routes:
            raw_probability = raw_row[route_id]
            if isinstance(raw_probability, bool) or not isinstance(raw_probability, (int, float)):
                raise ValueError(f"probabilities[{case.case_id!r}][{route_id!r}] must be finite")
            probability = float(raw_probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"probabilities[{case.case_id!r}][{route_id!r}] must be in [0, 1]")
            values[route_id] = probability
        total = math.fsum(values.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"probabilities for {case.case_id!r} must sum to one")

        ranked = sorted(
            enumerate(routes),
            key=lambda item: (-values[item[1]], item[0]),
        )
        selected = ranked[0][1]
        confidence = values[selected]
        acceptable_routes = contract.acceptable_routes(case.case_id)
        acceptable = selected in acceptable_routes
        base_contract = acceptable_routes == (BASE_ROUTE,)
        acceptable_count += int(acceptable)
        base_selected_count += int(selected == BASE_ROUTE)
        route_counts[selected] += 1
        category_rows[case.category].append(acceptable)
        if base_contract:
            base_contract_count += 1
            base_contract_acceptable_count += int(acceptable)
        else:
            specialist_count += 1
            specialist_acceptable_count += int(acceptable)
        predictions.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "selected_route": selected,
                "confidence": confidence,
                "acceptable_routes": list(acceptable_routes),
                "acceptable": acceptable,
                "top_routes": [
                    {"route_id": route_id, "probability": values[route_id]}
                    for _index, route_id in ranked[:3]
                ],
            }
        )

    case_count = len(suite.cases)
    categories = {
        category: {
            "cases": len(outcomes),
            "acceptable_top1": sum(outcomes),
            "acceptable_top1_rate": sum(outcomes) / len(outcomes),
        }
        for category, outcomes in category_rows.items()
        if outcomes
    }
    return {
        "schema_version": ROUTE_PROBABILITY_AUDIT_SCHEMA,
        "case_count": case_count,
        "acceptable_top1": acceptable_count,
        "acceptable_top1_rate": acceptable_count / case_count,
        "specialist_cases": specialist_count,
        "specialist_acceptable_top1": specialist_acceptable_count,
        "specialist_acceptable_top1_rate": (
            specialist_acceptable_count / specialist_count if specialist_count else None
        ),
        "base_contract_cases": base_contract_count,
        "base_contract_acceptable_top1": base_contract_acceptable_count,
        "base_contract_acceptable_top1_rate": (
            base_contract_acceptable_count / base_contract_count if base_contract_count else None
        ),
        "base_selected": base_selected_count,
        "base_selection_rate": base_selected_count / case_count,
        "route_selection_counts": route_counts,
        "categories": categories,
        "predictions": predictions,
    }


__all__ = [
    "BASE_ROUTE",
    "ROUTE_PROBABILITY_AUDIT_SCHEMA",
    "ROUTE_CONTRACT_SCHEMA",
    "ROUTING_EVALUATION_SCHEMA",
    "RouteEvaluationContract",
    "evaluate_route_probabilities",
    "load_route_evaluation_contract",
]
