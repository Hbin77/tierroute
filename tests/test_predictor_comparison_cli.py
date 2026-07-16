# SPDX-License-Identifier: Apache-2.0
"""Claim-safe CLI evidence for paired predictor estimation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tierroute.adapters import bundled_synthetic_path
from tierroute.cli import main


def _all_mapping_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                keys.append(str(key))
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(keys)


def test_compare_predictors_json_is_deterministic_shared_and_claim_safe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "compare-predictors",
        "--budget-scope",
        "per-query",
        "--gbm-estimators",
        "4",
        "--json",
    ]

    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert main(arguments) == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload["schema"] == "tierroute-predictor-comparison"
    assert payload["schema_version"] == 1
    assert payload["command"] == "compare-predictors"
    assert payload["claim_state"] == "SYNTHETIC-ONLY"
    assert payload["evidence_role"] == "descriptive-paired-estimation-only"
    assert payload["selection_protocol"] == "none-paired-estimation"
    assert payload["selected_family"] is None
    assert payload["performance_claim_allowed"] is False
    assert payload["network_used"] is False
    assert payload["comparison_direction"] == "gbm-minus-bilinear"
    assert tuple(payload["predictor_families"]) == ("bilinear", "gbm")
    assert payload["predictor_families"]["bilinear"]["predictor"]["kind"] == (
        "calibrated-bilinear-surface-v1"
    )
    assert payload["predictor_families"]["gbm"]["predictor"]["kind"] == (
        "calibrated-gbm-regression-stumps-surface-v1"
    )
    assert payload["predictor_families"]["gbm"]["predictor"]["n_estimators"] == 4
    families = payload["predictor_families"]
    paired_metrics = {
        name: family["paired_metrics_full_precision"] for name, family in families.items()
    }
    for tier in ("fast", "balanced", "premium"):
        assert payload["deltas"]["overall"]["tier_quality"][tier] == (
            paired_metrics["gbm"]["tier_quality"][tier]
            - paired_metrics["bilinear"]["tier_quality"][tier]
        )
    assert payload["deltas"]["overall"]["weighted_quality"] == (
        paired_metrics["gbm"]["weighted_quality"] - paired_metrics["bilinear"]["weighted_quality"]
    )
    assert payload["deltas"]["overall"]["oracle_gap_recovery"] == (
        paired_metrics["gbm"]["oracle_gap_recovery"]
        - paired_metrics["bilinear"]["oracle_gap_recovery"]
    )
    assert [row["name"] for row in payload["baselines"]] == [
        "always-cheapest",
        "always-premium",
        "random",
        "length-heuristic",
        "oracle",
        "domain-best-table",
    ]
    assert [row["held_out_domain"] for row in payload["deltas"]["held_out_domains"]] == [
        "code",
        "general",
        "math",
        "science",
    ]
    assert "winner" not in _all_mapping_keys(payload)


def test_compare_predictors_human_output_keeps_fixed_family_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "compare-predictors",
                "--budget-scope",
                "per-query",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "paired predictor estimation (wiring only)" in output
    assert output.index("\nbilinear") < output.index("\ngbm") < output.index("\nalways-cheapest")
    assert "Selection: not performed; this same outer evidence cannot select a winner." in output
    assert "Performance claim: prohibited" in output
    assert "Claim state: SYNTHETIC-ONLY" in output
    assert "Network: disabled" in output


def test_compare_predictors_human_output_escapes_terminal_controls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = json.loads(bundled_synthetic_path().read_text(encoding="utf-8"))
    payload["name"] = "visible\x1b[2J\nspoof\u202e"
    replay_path = tmp_path / "terminal-controls.json"
    replay_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "compare-predictors",
                "--budget-scope",
                "per-query",
                "--data",
                str(replay_path),
                "--gbm-estimators",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "visible\\u001b[2J\\u000aspoof\\u202e" in output
    assert "\x1b" not in output
    assert "\u202e" not in output


def test_compare_predictors_explicit_data_remains_unverified_user_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "compare-predictors",
                "--budget-scope",
                "per-query",
                "--data",
                str(bundled_synthetic_path()),
                "--gbm-estimators",
                "1",
                "--json",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["claim_state"] == "UNVERIFIED-USER-DATA"


def test_compare_predictors_requires_explicit_safe_search_arguments() -> None:
    with pytest.raises(SystemExit):
        main(["compare-predictors", "--json"])
    with pytest.raises(SystemExit):
        main(
            [
                "compare-predictors",
                "--budget-scope",
                "per-query",
                "--max-lambda-candidates",
                "1",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "compare-predictors",
                "--budget-scope",
                "per-query",
                "--allow-large-exhaustive-search",
            ]
        )
