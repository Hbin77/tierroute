# SPDX-License-Identifier: Apache-2.0
"""Tests for local-only RouterBench validation orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


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
    semantic_sha256, _, _, _, _ = module._scan_semantic_rows(rows, expected_row_count=1)
    monkeypatch.setattr(module, "ROUTERBENCH_ROW_COUNT", 1)  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "ROUTERBENCH_COLUMN_COUNT", len(rows[0]))  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "ROUTERBENCH_IN_SCOPE_COUNT", 1)  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "ROUTERBENCH_MODEL_COUNT", 2)  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "ROUTERBENCH_DOMAIN_COUNTS", {"hellaswag": 1})  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "ROUTERBENCH_SEMANTIC_SHA256", semantic_sha256)  # type: ignore[attr-defined]

    module.validate_and_replay(tmp_path / "unused.pkl", replay_limit=1)

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "In-scope examples: 1" in output
    assert f"Semantic SHA-256: {semantic_sha256}" in output
    assert "Dataset license: NOASSERTION" in output


def test_semantic_digest_framing_preserves_utf8_and_negative_zero() -> None:
    module = load_script()
    rows = ({"text": "é", "score": -0.0}, {"text": "β", "score": 1.5})
    digest = module._SemanticDigestBuilder(  # type: ignore[attr-defined]
        expected_row_count=2,
        columns=("text", "score"),
    )

    for row in rows:
        digest.update(row)

    assert digest.hexdigest() == (
        "66b3e912abd30d17dde3a2f735acba1a54f4392cb95c35becd1ed403c4a0ea44"
    )


def test_local_pinned_artifact_matches_semantic_golden() -> None:
    module = load_script()
    artifact = Path(__file__).parents[1] / "data/routerbench/routerbench_0shot.pkl"
    if not artifact.is_file():
        pytest.skip("external NOASSERTION RouterBench artifact is not present")

    semantic_sha256, rows, in_scope_count, domain_counts, columns = module._scan_semantic_rows(
        module.iter_routerbench_rows(artifact),
        expected_row_count=module.ROUTERBENCH_ROW_COUNT,
        retain_limit=0,
    )
    response_models = {
        column.removesuffix("|model_response")
        for column in columns
        if column.endswith("|model_response")
    }

    assert semantic_sha256 == module.ROUTERBENCH_SEMANTIC_SHA256
    assert len(columns) == module.ROUTERBENCH_COLUMN_COUNT
    assert not rows
    assert in_scope_count == module.ROUTERBENCH_IN_SCOPE_COUNT
    assert len(response_models) == module.ROUTERBENCH_MODEL_COUNT
    assert dict(domain_counts) == module.ROUTERBENCH_DOMAIN_COUNTS
