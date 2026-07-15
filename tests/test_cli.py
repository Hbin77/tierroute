# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for clone-without-data quickstart commands."""

import json
from pathlib import Path

from tierroute.adapters import load_evaluation_dataset
from tierroute.cli import main
from tierroute.core import BudgetTier
from tierroute.demo import evaluate_six_baselines, route_prompt


def test_route_command_shows_decision_cost_and_predicted_quality(capsys: object) -> None:
    status = main(["route", "간단한 질문입니다.", "--tier", "fast"])
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert status == 0
    assert "selected model:" in output
    assert "estimated cost:" in output
    assert "predicted quality:" in output
    assert "network:           disabled" in output


def test_route_json_is_machine_readable_and_explicitly_synthetic(capsys: object) -> None:
    status = main(["route", "Prove x = x.", "--tier", "premium", "--json"])
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert status == 0
    assert payload["tier"] == "premium"
    assert payload["network_used"] is False
    assert payload["quality_kind"] == "synthetic demo prediction"


def test_six_baselines_run_end_to_end_on_bundled_data() -> None:
    results = evaluate_six_baselines(load_evaluation_dataset())

    assert [result.name for result in results] == [
        "always-cheapest",
        "always-premium",
        "random",
        "length-heuristic",
        "oracle",
        "domain-best-table",
    ]
    by_name = {result.name: result for result in results}
    assert by_name["always-cheapest"].gap_recovery == 0
    assert by_name["oracle"].gap_recovery == 1
    assert by_name["always-premium"].score.weighted_quality is None


def test_demo_router_changes_model_with_tier_and_difficulty() -> None:
    dataset = load_evaluation_dataset()

    fast = route_prompt(dataset, "hello", BudgetTier.FAST)
    premium = route_prompt(
        dataset,
        "Prove a difficult theorem with equations and check every step.",
        BudgetTier.PREMIUM,
    )

    assert fast.model_id == "swift"
    assert premium.model_id == "expert"


def test_evaluate_command_prints_all_required_baselines(capsys: object) -> None:
    assert main(["evaluate"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert "always-cheapest" in output
    assert "domain-best-table" in output
    assert "not benchmark claims" in output


def test_train_then_route_with_canonical_artifact(
    tmp_path: Path,
    capsys: object,
) -> None:
    artifact = tmp_path / "predictor.json"

    assert main(["train", "--output", str(artifact), "--json"]) == 0
    training = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert artifact.is_file()
    assert training["network_used"] is False
    assert training["training_examples"] == 8
    assert training["training_domains"] == ["code", "general", "math", "science"]
    assert training["model_ids"] == ["expert", "steady", "swift"]
    assert training["solver_id"] == "tierroute.centered-ridge-cholesky-python-v1"

    assert (
        main(
            [
                "route",
                "Prove a difficult theorem.",
                "--tier",
                "balanced",
                "--artifact",
                str(artifact),
                "--json",
            ]
        )
        == 0
    )
    route = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert route["network_used"] is False
    assert route["quality_kind"] == "calibrated bilinear artifact"
    assert route["model"] in training["model_ids"]
