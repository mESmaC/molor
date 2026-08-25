from __future__ import annotations

from pathlib import Path

import pytest

from molor.experimental import compare


def test_compare_reports_missing_06_experts_before_loading_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(compare, "_WORKSPACE_ROOT", tmp_path)
    model = tmp_path / "models" / "Qwen3-0.6B"

    with pytest.raises(SystemExit) as error:
        compare.main(["--model", str(model)])

    assert error.value.code == 2
    assert "--expert-a has no complete compatible adapter" in capsys.readouterr().err
