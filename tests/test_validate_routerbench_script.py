# SPDX-License-Identifier: Apache-2.0
"""Tests for local-only RouterBench validation orchestration."""

from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path
from types import ModuleType

from tierroute.eval import CandidateOutcome, EvaluationExample


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
    examples = (
        EvaluationExample(
            "q1",
            "prompt",
            "math",
            (
                CandidateOutcome("cheap", "answer", Decimal("0.1"), 0.5),
                CandidateOutcome("premium", "better", Decimal("0.3"), 0.9),
            ),
        ),
    )
    monkeypatch.setattr(module, "iter_routerbench_examples", lambda _: iter(examples))  # type: ignore[attr-defined]

    module.validate_and_replay(tmp_path / "unused.pkl", replay_limit=1)

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Converted examples: 1" in output
    assert "Dataset license: NOASSERTION" in output
