from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = WORKSPACE / "experiments" / "expert-rack-v5" / "audit_hierarchical_router.py"


def _audit_cli():
    spec = importlib.util.spec_from_file_location(
        "hierarchical_router_audit_cli_test", AUDIT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_base_contract_gate_defaults_to_eighty_percent_and_fails_closed() -> None:
    cli = _audit_cli()
    args = cli._parser().parse_args([])

    assert args.minimum_base_contract_rate == 0.80
    assert cli._base_contract_gate_failed(None, args.minimum_base_contract_rate)
    assert cli._base_contract_gate_failed(float("nan"), args.minimum_base_contract_rate)
    assert cli._base_contract_gate_failed(0.79, args.minimum_base_contract_rate)
    assert not cli._base_contract_gate_failed(0.80, args.minimum_base_contract_rate)
