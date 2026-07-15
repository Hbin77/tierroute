# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for clone-without-data quickstart commands."""

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from tierroute.adapters import bundled_synthetic_path, load_evaluation_dataset
from tierroute.cli import (
    DEFAULT_MAX_LAMBDA_CANDIDATES,
    _fraction_label,
    _fraction_payload,
    main,
)
from tierroute.core import BudgetTier, atomic_io
from tierroute.demo import evaluate_six_baselines, route_prompt
from tierroute.policies import lambda_tuning
from tierroute.policies.lambda_artifacts import LambdaPolicyArtifact
from tierroute.policies.lambda_tuning import tune_tier_lambdas
from tierroute.predictors import BilinearPredictorArtifact


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
    assert payload["lambda_cost"] == {"numerator": "2", "denominator": "25"}
    assert payload["accounting_scope"] == "per-query-illustrative"


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

    injected = route_prompt(
        dataset,
        "hello",
        BudgetTier.FAST,
        lambda_cost=Fraction(1, 3),
    )
    assert injected.lambda_cost == Fraction(1, 3)


def test_cli_fraction_rendering_supports_ten_thousand_digits() -> None:
    value = Fraction(1, 10**10000)

    payload = _fraction_payload(value)
    label = _fraction_label(value)

    assert payload["numerator"] == "1"
    assert len(payload["denominator"]) == 10001
    assert label == f"1/{payload['denominator']}"


def test_evaluate_command_prints_all_required_baselines(capsys: object) -> None:
    assert main(["evaluate"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert "always-cheapest" in output
    assert "domain-best-table" in output
    assert "not benchmark claims" in output
    assert "original-order outer-LODO" in output
    assert "outer training side" in output


def test_evaluate_json_declares_lodo_and_domain_fit_scope(capsys: object) -> None:
    assert main(["evaluate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["budget_scope"] == "per-query-illustrative"
    assert payload["validation_scope"] == "outer-lodo-original-order"
    assert payload["domain_table_fit"] == "outer-training-observable-tags-only"


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


def test_policy_output_requires_explicit_budget_scope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictor = tmp_path / "predictor.json"
    policy = tmp_path / "policy.json"

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "train",
                "--output",
                str(predictor),
                "--policy-output",
                str(policy),
            ]
        )

    assert caught.value.code == 2
    assert "--policy-output requires --budget-scope" in capsys.readouterr().err
    assert not predictor.exists()
    assert not policy.exists()


def test_train_rejects_predictor_and_policy_destination_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "artifact.json"
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "train",
                "--output",
                destination.name,
                "--policy-output",
                str(destination),
                "--budget-scope",
                "per-query",
            ]
        )

    assert caught.value.code == 2
    assert "must be different paths" in capsys.readouterr().err
    assert not destination.exists()


def test_train_bundle_avoids_the_old_fixed_temporary_name_collision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictor = tmp_path / ".policy.json.tmp"
    policy = tmp_path / "policy.json"

    assert (
        main(
            [
                "train",
                "--output",
                str(predictor),
                "--policy-output",
                str(policy),
                "--budget-scope",
                "per-query",
                "--max-lambda-candidates",
                "2",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    loaded_predictor = BilinearPredictorArtifact.load(predictor)
    loaded_policy = LambdaPolicyArtifact.load(policy)
    loaded_policy.validate_predictor(loaded_predictor)


def test_train_rejects_output_aliasing_the_source_before_clobbering_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "replay.json"
    original = bundled_synthetic_path().read_bytes()
    source.write_bytes(original)

    with pytest.raises(SystemExit) as caught:
        main(["train", "--data", str(source), "--output", str(source)])

    assert caught.value.code == 2
    assert "protected input path" in capsys.readouterr().err
    assert source.read_bytes() == original


def test_train_bundle_rolls_back_both_outputs_when_policy_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = tmp_path / "predictor.json"
    policy = tmp_path / "policy.json"
    predictor.write_text("old predictor", encoding="utf-8")
    policy.write_text("old policy", encoding="utf-8")
    real_replace = os.replace

    def fail_policy_stage(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == policy and ".stage." in Path(source).name:
            raise OSError("injected policy replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", fail_policy_stage)

    with pytest.raises(OSError, match="injected policy"):
        main(
            [
                "train",
                "--output",
                str(predictor),
                "--policy-output",
                str(policy),
                "--budget-scope",
                "per-query",
                "--max-lambda-candidates",
                "2",
            ]
        )

    assert predictor.read_text(encoding="utf-8") == "old predictor"
    assert policy.read_text(encoding="utf-8") == "old policy"
    assert not [
        path for path in tmp_path.iterdir() if ".stage." in path.name or ".backup." in path.name
    ]


def test_lambda_search_options_are_policy_only_and_mutually_exclusive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictor = tmp_path / "predictor.json"
    policy = tmp_path / "policy.json"

    with pytest.raises(SystemExit) as caught:
        main(["train", "--output", str(predictor), "--max-lambda-candidates", "2"])
    assert caught.value.code == 2
    assert "requires --policy-output" in capsys.readouterr().err

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "train",
                "--output",
                str(predictor),
                "--policy-output",
                str(policy),
                "--budget-scope",
                "per-query",
                "--max-lambda-candidates",
                "2",
                "--exhaustive-lambda-search",
            ]
        )
    assert caught.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "train",
                "--output",
                str(predictor),
                "--policy-output",
                str(policy),
                "--budget-scope",
                "per-query",
                "--allow-large-exhaustive-search",
            ]
        )
    assert caught.value.code == 2
    assert "requires --exhaustive-lambda-search" in capsys.readouterr().err


def test_cli_default_and_large_exhaustive_acknowledgement_are_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int | None, bool]] = []
    real_tune = tune_tier_lambdas

    def capture_tuning(*args: object, **kwargs: object) -> object:
        calls.append(
            (
                kwargs.get("max_candidates_per_tier"),  # type: ignore[arg-type]
                kwargs.get("allow_large_exhaustive"),  # type: ignore[arg-type]
            )
        )
        return real_tune(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("tierroute.cli.tune_tier_lambdas", capture_tuning)
    base = [
        "train",
        "--policy-output",
        str(tmp_path / "policy.json"),
        "--budget-scope",
        "per-query",
    ]

    assert main([*base, "--output", str(tmp_path / "predictor.json")]) == 0
    assert calls[-1] == (DEFAULT_MAX_LAMBDA_CANDIDATES, False)

    assert (
        main(
            [
                *base,
                "--output",
                str(tmp_path / "predictor-exhaustive.json"),
                "--policy-output",
                str(tmp_path / "policy-exhaustive.json"),
                "--exhaustive-lambda-search",
                "--allow-large-exhaustive-search",
            ]
        )
        == 0
    )
    assert calls[-1] == (None, True)


@pytest.mark.parametrize("search_args", [[], ["--exhaustive-lambda-search"]])
def test_cli_search_preflight_runs_before_predictor_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search_args: list[str],
) -> None:
    fitted = False

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        nonlocal fitted
        del args, kwargs
        fitted = True
        raise AssertionError("predictor fitting must not start")

    monkeypatch.setattr(lambda_tuning, "MAX_UNCONFIRMED_EXHAUSTIVE_CANDIDATES", 1)
    monkeypatch.setattr("tierroute.cli.fit_calibrated_bilinear", forbidden_fit)
    predictor = tmp_path / "predictor.json"
    policy = tmp_path / "policy.json"

    with pytest.raises(ValueError, match="refused before candidate materialization"):
        main(
            [
                "train",
                "--output",
                str(predictor),
                "--policy-output",
                str(policy),
                "--budget-scope",
                "per-query",
                *search_args,
            ]
        )

    assert fitted is False
    assert not predictor.exists()
    assert not policy.exists()


def test_train_then_route_with_canonical_exact_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictor_path = tmp_path / "predictor.json"
    policy_path = tmp_path / "policy.json"

    assert (
        main(
            [
                "train",
                "--output",
                str(predictor_path),
                "--policy-output",
                str(policy_path),
                "--budget-scope",
                "per-query",
                "--max-lambda-candidates",
                "2",
                "--json",
            ]
        )
        == 0
    )
    training = json.loads(capsys.readouterr().out)
    policy = LambdaPolicyArtifact.load(policy_path)

    assert policy_path.read_text(encoding="utf-8") == policy.to_json()
    assert training["network_used"] is False
    assert training["policy_artifact"] == str(policy_path)
    assert training["accounting_scope"] == "per-query"
    assert training["feasible"] is True
    assert training["weighted_training_score"] == pytest.approx(0.73125)
    assert set(training["lambda_by_tier"]) == {"fast", "balanced", "premium"}
    for detail in training["lambda_search"].values():
        assert detail["retained_candidates"] <= 2
        assert detail["observed_breakpoint_count"] > 0
        if detail["exhaustive"]:
            assert detail["derived_candidates"] == detail["retained_candidates"]
            assert detail["strategy"] == "exhaustive-breakpoints-v1"
        else:
            assert detail["derived_candidates"] is None
            assert detail["strategy"] == "bounded-bottom-hash-v2"
    balanced_search = training["lambda_search"]["balanced"]
    assert balanced_search["derived_candidates"] is None
    assert balanced_search["exhaustive"] is False
    assert balanced_search["observed_breakpoint_count"] > 0
    assert balanced_search["retained_candidates"] == 2
    assert balanced_search["strategy"] == "bounded-bottom-hash-v2"

    assert (
        main(
            [
                "route",
                "Prove a difficult theorem.",
                "--tier",
                "balanced",
                "--artifact",
                str(predictor_path),
                "--policy-artifact",
                str(policy_path),
                "--json",
            ]
        )
        == 0
    )
    route = json.loads(capsys.readouterr().out)
    selected_lambda = policy.lambda_by_tier[BudgetTier.BALANCED]

    assert route["network_used"] is False
    assert route["quality_kind"] == "calibrated bilinear + tuned exact-rational tier lambda"
    assert route["accounting_scope"] == "per-query"
    assert route["lambda_cost"] == {
        "numerator": str(selected_lambda.numerator),
        "denominator": str(selected_lambda.denominator),
    }
    assert route["lambda_search"] == training["lambda_search"]["balanced"]


def test_policy_route_fails_closed_on_provenance_catalogue_and_tier_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictor_path = tmp_path / "predictor.json"
    policy_path = tmp_path / "policy.json"
    assert (
        main(
            [
                "train",
                "--output",
                str(predictor_path),
                "--policy-output",
                str(policy_path),
                "--budget-scope",
                "per-query",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    route_args = [
        "route",
        "hello",
        "--artifact",
        str(predictor_path),
        "--policy-artifact",
        str(policy_path),
    ]

    altered_predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
    altered_predictor["training"]["seed"] = 99
    altered_predictor_path = tmp_path / "altered-predictor.json"
    altered_predictor_path.write_text(json.dumps(altered_predictor), encoding="utf-8")
    with pytest.raises(ValueError, match="predictor SHA-256"):
        main(
            [
                *route_args[:2],
                "--artifact",
                str(altered_predictor_path),
                *route_args[4:],
            ]
        )

    source = json.loads(bundled_synthetic_path().read_text(encoding="utf-8"))
    source["tier_specs"][0]["budget_limit"] = "0.36"
    tier_mismatch = tmp_path / "tier-mismatch.json"
    tier_mismatch.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="tier specifications"):
        main([*route_args, "--data", str(tier_mismatch)])

    source = json.loads(bundled_synthetic_path().read_text(encoding="utf-8"))
    for example in source["examples"]:
        for outcome in example["outcomes"]:
            if outcome["model_id"] == "swift":
                outcome["quoted_cost"] = "0.21"
    catalogue_mismatch = tmp_path / "catalogue-mismatch.json"
    catalogue_mismatch.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="tuning-data SHA-256"):
        main([*route_args, "--data", str(catalogue_mismatch)])

    source = json.loads(bundled_synthetic_path().read_text(encoding="utf-8"))
    source["examples"].reverse()
    replay_mismatch = tmp_path / "replay-mismatch.json"
    replay_mismatch.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="replay-order SHA-256"):
        main([*route_args, "--data", str(replay_mismatch)])


def test_cumulative_policy_route_requires_and_reports_current_remaining_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    predictor_path = tmp_path / "predictor.json"
    policy_path = tmp_path / "policy.json"
    assert (
        main(
            [
                "train",
                "--output",
                str(predictor_path),
                "--policy-output",
                str(policy_path),
                "--budget-scope",
                "per-query",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # The bundled limits intentionally model per-query accounting, so relabel a
    # validated artifact here only to isolate the cumulative CLI state contract.
    # Genuine cumulative tuner accounting is covered by test_lambda_tuning.py.
    policy_document = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_document["tuning"]["ledger_adapter"] = "cumulative"
    cumulative_policy = tmp_path / "cumulative-policy.json"
    cumulative_policy.write_text(json.dumps(policy_document), encoding="utf-8")
    arguments = [
        "route",
        "hello",
        "--tier",
        "balanced",
        "--artifact",
        str(predictor_path),
        "--policy-artifact",
        str(cumulative_policy),
        "--json",
    ]

    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2
    assert "cumulative policy requires --remaining-budget" in capsys.readouterr().err

    assert main([*arguments, "--remaining-budget", "0.50"]) == 0
    route = json.loads(capsys.readouterr().out)
    assert route["accounting_scope"] == "cumulative"
    assert route["remaining_budget"] == "0.50"


def test_policy_route_requires_predictor_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["route", "hello", "--policy-artifact", str(tmp_path / "policy.json")])

    assert caught.value.code == 2
    assert "--policy-artifact requires --artifact" in capsys.readouterr().err
