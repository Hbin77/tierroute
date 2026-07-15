# SPDX-License-Identifier: Apache-2.0
"""Tests for local-only RouterBench validation orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "validate_routerbench.py"
    spec = importlib.util.spec_from_file_location("validate_routerbench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_script_replays_without_downloading(
    tmp_path: Path, monkeypatch: object, capsys: object
) -> None:
    module = load_script()
    rows = (
        {
            "sample_id": "q1",
            "prompt": "prompt",
            "eval_name": "hellaswag",
            "oracle_model_to_route_to": "cheap",
            "cheap": 0.5,
            "cheap|model_response": "answer",
            "cheap|total_cost": 0.1,
            "premium": 0.9,
            "premium|model_response": "better",
            "premium|total_cost": 0.3,
        },
    )
    monkeypatch.setattr(module, "iter_routerbench_rows", lambda _: iter(rows))  # type: ignore[attr-defined]

    module.validate_and_replay(tmp_path / "unused.pkl", replay_limit=1)

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Converted examples: 1" in output
    assert "Dataset license: NOASSERTION" in output
