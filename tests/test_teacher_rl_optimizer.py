from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENT = WORKSPACE / "experiments" / "teacher-rl-v1"


def _load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def optimizer() -> ModuleType:
    sys.path.insert(0, str(EXPERIMENT))
    try:
        return _load_script(
            "molor_test_reward_router_optimizer",
            EXPERIMENT / "optimize_reward_router.py",
        )
    finally:
        sys.path.remove(str(EXPERIMENT))


def _write_parent_checkpoint(
    root: Path,
    *,
    route_ids: tuple[str, ...] = (
        "expert-a",
        "expert-b",
        "expert-c",
        "expert-d",
        "base/null",
    ),
    runtime_names: tuple[str, ...] | None = None,
) -> Path:
    root.mkdir()
    weights = root / "router.safetensors"
    weights.write_bytes(b"test-router-weights")
    model = root / "base-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test-base-weights")
    expert_names = runtime_names or route_ids[:-1]
    if len(expert_names) != len(route_ids) - 1:
        raise ValueError("runtime_names must align with expert routes")
    adapters = []
    for index, (route_id, runtime_name) in enumerate(
        zip(route_ids[:-1], expert_names, strict=True)
    ):
        directory = root / f"adapter-{index}"
        directory.mkdir()
        (directory / "adapter_config.json").write_text(
            json.dumps({"name": runtime_name}), encoding="utf-8"
        )
        (directory / "adapter_model.safetensors").write_bytes(
            f"test-adapter-weights-{index}".encode()
        )
        adapter = {"route_id": route_id, "path": str(directory)}
        if runtime_names is not None:
            adapter["runtime_name"] = runtime_name
        adapters.append(adapter)
    sidecar = {
        "schema_version": "molor.experimental-router-checkpoint.v1",
        "route_ids": list(route_ids),
        "route_to_index": {route_id: index for index, route_id in enumerate(route_ids)},
        "adapters": adapters,
        "model": {"model_id": str(model)},
        "router_config": {
            "hidden_size": 8,
            "num_experts": len(route_ids) - 1,
            "top_k": 1,
            "mode": "sequence",
            "router_hidden_size": 4,
            "temperature": 1.0,
            "include_null_expert": True,
            "normalize_top_k": True,
            "uniform_top_k": False,
            "straight_through": False,
        },
        "checkpoint": {
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "size_bytes": weights.stat().st_size,
        },
    }
    (root / "router_config.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return root


def _manifest(optimizer: ModuleType, routes: tuple[tuple[str, str], ...]) -> SimpleNamespace:
    enabled = [runtime_name for route_id, runtime_name in routes if route_id != "base/null"]
    return SimpleNamespace(
        forced_routes=tuple(
            optimizer.evaluation.RouteSpec(route_id, runtime_name, None)
            for route_id, runtime_name in routes
        ),
        manual_policy_defaults={"enabled_experts": enabled},
        auto_policy={"enabled_experts": enabled.copy()},
    )


def test_parent_contract_accepts_promoted_five_way_action_axis(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))

    assert parent.route_ids == (
        "expert-a",
        "expert-b",
        "expert-c",
        "expert-d",
        "base/null",
    )
    assert parent.router_config.num_routes == 5


def test_parent_contract_accepts_single_expert_plus_null(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(
        _write_parent_checkpoint(
            tmp_path / "router",
            route_ids=("expert-a", "base/null"),
        )
    )

    assert parent.route_ids == ("expert-a", "base/null")
    assert parent.router_config.num_routes == 2


@pytest.mark.parametrize("mutation", ["route_index", "adapter_order", "weights"])
def test_parent_contract_rejects_action_axis_or_artifact_identity_drift(
    optimizer: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _write_parent_checkpoint(tmp_path / "router")
    config_path = root / "router_config.json"
    sidecar = json.loads(config_path.read_text(encoding="utf-8"))
    if mutation == "route_index":
        sidecar["route_to_index"]["expert-a"] = 1
    elif mutation == "adapter_order":
        sidecar["adapters"][0], sidecar["adapters"][1] = (
            sidecar["adapters"][1],
            sidecar["adapters"][0],
        )
    else:
        (root / "router.safetensors").write_bytes(b"different-router-weights")
    config_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(optimizer.OptimizerError):
        optimizer._load_parent_contract(root)


def test_manifest_route_contract_accepts_matching_five_way_runtime_mapping(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    manifest = _manifest(
        optimizer,
        (
            ("expert-a", "expert-a"),
            ("expert-b", "expert-b"),
            ("expert-c", "expert-c"),
            ("expert-d", "expert-d"),
            ("base/null", "base"),
        ),
    )

    optimizer._validate_manifest_route_contract(manifest, parent)


def test_manifest_collection_order_does_not_redefine_parent_action_axis(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    manifest = _manifest(
        optimizer,
        (
            ("base/null", "base"),
            ("expert-d", "expert-d"),
            ("expert-b", "expert-b"),
            ("expert-a", "expert-a"),
            ("expert-c", "expert-c"),
        ),
    )

    optimizer._validate_manifest_route_contract(manifest, parent)


def test_manifest_route_contract_accepts_parent_runtime_aliases(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(
        _write_parent_checkpoint(
            tmp_path / "router",
            route_ids=("stable-a", "stable-b", "base/null"),
            runtime_names=("runtime-a", "runtime-b"),
        )
    )
    manifest = _manifest(
        optimizer,
        (
            ("stable-a", "runtime-a"),
            ("stable-b", "runtime-b"),
            ("base/null", "base"),
        ),
    )

    optimizer._validate_manifest_route_contract(manifest, parent)


def test_parent_contract_rejects_duplicate_runtime_aliases(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    root = _write_parent_checkpoint(
        tmp_path / "router",
        route_ids=("stable-a", "stable-b", "base/null"),
        runtime_names=("duplicate", "duplicate"),
    )

    with pytest.raises(optimizer.OptimizerError, match="runtime names must be unique"):
        optimizer._load_parent_contract(root)


def test_manifest_policy_must_enable_runtime_aliases_not_stable_route_ids(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(
        _write_parent_checkpoint(
            tmp_path / "router",
            route_ids=("stable-a", "stable-b", "base/null"),
            runtime_names=("runtime-a", "runtime-b"),
        )
    )
    manifest = _manifest(
        optimizer,
        (
            ("stable-a", "runtime-a"),
            ("stable-b", "runtime-b"),
            ("base/null", "base"),
        ),
    )
    manifest.manual_policy_defaults["enabled_experts"] = ["stable-a", "stable-b"]

    with pytest.raises(optimizer.OptimizerError, match="enabled_experts"):
        optimizer._validate_manifest_route_contract(manifest, parent)


@pytest.mark.parametrize("mutation", ["route_bank", "runtime_swap", "enabled_bank"])
def test_manifest_route_contract_rejects_identity_mismatch(
    optimizer: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    routes = [
        ("expert-a", "expert-a"),
        ("expert-b", "expert-b"),
        ("expert-c", "expert-c"),
        ("expert-d", "expert-d"),
        ("base/null", "base"),
    ]
    if mutation == "route_bank":
        routes[0] = ("unknown-expert", "unknown-expert")
    elif mutation == "runtime_swap":
        routes[0] = ("expert-a", "expert-b")
        routes[1] = ("expert-b", "expert-a")
    manifest = _manifest(optimizer, tuple(routes))
    if mutation == "enabled_bank":
        manifest.auto_policy["enabled_experts"].remove("expert-d")

    with pytest.raises(optimizer.OptimizerError):
        optimizer._validate_manifest_route_contract(manifest, parent)


def _provenance_manifest() -> SimpleNamespace:
    return SimpleNamespace(fingerprint="manifest-sha", run_id="provenance-run")


def _session_events(
    optimizer: ModuleType,
    parent,
    *,
    exact: bool,
) -> list[dict[str, object]]:
    manifest = _provenance_manifest()
    health = {
        "status": "ok",
        "protocol_version": "molor.teacher-api.v1",
        **optimizer._parent_health_identity(parent),
    }
    session: dict[str, object] = {
        "event": "collection_session",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.fingerprint,
        "capabilities_sha256": "capability-sha",
        "health": health,
    }
    rollout: dict[str, object] = {
        "event": "rollout",
        "status": "ok",
        "request_key": "request-key",
        "run_id": manifest.run_id,
        "manifest_sha256": manifest.fingerprint,
        "capabilities_sha256": "capability-sha",
    }
    if exact:
        contract = optimizer.evaluation._build_student_artifact_contract(health)
        session.update(
            {
                "artifact_provenance_status": optimizer.evaluation.EXACT_ARTIFACT_PROVENANCE,
                "student_artifact_contract": contract,
            }
        )
        rollout["artifact_contract_sha256"] = contract["contract_sha256"]
    return [session, rollout]


def test_exact_collection_provenance_binds_all_local_artifact_bytes(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    events = _session_events(optimizer, parent, exact=True)

    summary = optimizer._collection_provenance(events, _provenance_manifest(), parent)

    contract = events[0]["student_artifact_contract"]
    roles = [artifact["role"] for artifact in contract["artifacts"]]
    assert summary["status"] == optimizer.evaluation.EXACT_ARTIFACT_PROVENANCE
    assert roles.count("base") == 1
    assert roles.count("expert") == 4
    assert roles.count("router") == 1
    files = {
        (artifact["role"], file["name"])
        for artifact in contract["artifacts"]
        for file in artifact["files"]
    }
    assert ("base", "model.safetensors") in files
    assert ("expert", "adapter_config.json") in files
    assert ("expert", "adapter_model.safetensors") in files
    assert ("router", "router_config.json") in files
    assert ("router", "router.safetensors") in files


def test_exact_collection_provenance_rejects_current_byte_drift(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    events = _session_events(optimizer, parent, exact=True)
    model_path = Path(parent.sidecar["model"]["model_id"])
    (model_path / "model.safetensors").write_bytes(b"mutated-base-weights")

    with pytest.raises(optimizer.OptimizerError, match="artifact bytes differ"):
        optimizer._collection_provenance(events, _provenance_manifest(), parent)


def test_legacy_collection_provenance_requires_matching_health_paths_and_names(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    events = _session_events(optimizer, parent, exact=False)

    summary = optimizer._collection_provenance(events, _provenance_manifest(), parent)
    assert summary["status"] == optimizer.evaluation.LEGACY_ARTIFACT_PROVENANCE

    events[0]["health"]["experts"][0] = "wrong-runtime-name"
    with pytest.raises(optimizer.OptimizerError, match="paths/names do not match"):
        optimizer._collection_provenance(events, _provenance_manifest(), parent)


def test_collection_provenance_rejects_mixed_legacy_and_exact_successes(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    legacy = _session_events(optimizer, parent, exact=False)
    exact = _session_events(optimizer, parent, exact=True)
    exact[1]["request_key"] = "exact-request-key"

    with pytest.raises(optimizer.OptimizerError, match="mixes legacy and exact"):
        optimizer._collection_provenance(
            [*legacy, *exact],
            _provenance_manifest(),
            parent,
        )


def test_collection_provenance_rejects_multiple_exact_contract_generations(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    events = _session_events(optimizer, parent, exact=True)
    other_rollout = {
        **events[1],
        "request_key": "other-request-key",
        "artifact_contract_sha256": "other-contract-sha",
    }

    with pytest.raises(optimizer.OptimizerError, match="multiple exact artifact contract"):
        optimizer._collection_provenance(
            [*events, other_rollout],
            _provenance_manifest(),
            parent,
        )


def test_collection_provenance_rejects_multiple_legacy_session_identities(
    optimizer: ModuleType,
    tmp_path: Path,
) -> None:
    parent = optimizer._load_parent_contract(_write_parent_checkpoint(tmp_path / "router"))
    events = _session_events(optimizer, parent, exact=False)
    other_session = {**events[0], "capabilities_sha256": "other-capability-sha"}
    other_rollout = {
        **events[1],
        "request_key": "other-request-key",
        "capabilities_sha256": "other-capability-sha",
    }

    with pytest.raises(optimizer.OptimizerError, match="multiple collection session identities"):
        optimizer._collection_provenance(
            [*events, other_session, other_rollout],
            _provenance_manifest(),
            parent,
        )
