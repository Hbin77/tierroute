# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for clone-without-data quickstart commands."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import tierroute.cli as cli_module
import tierroute.predictors.native_ridge as native_ridge_module
from tierroute.adapters import bundled_synthetic_path, load_evaluation_dataset
from tierroute.cli import (
    DEFAULT_MAX_LAMBDA_CANDIDATES,
    _baseline_payload,
    _fraction_label,
    _fraction_payload,
    main,
)
from tierroute.core import BudgetTier, atomic_io
from tierroute.demo import evaluate_six_baselines, route_prompt
from tierroute.eval import EvaluationExample
from tierroute.policies import lambda_tuning
from tierroute.policies.lambda_artifacts import LambdaPolicyArtifact
from tierroute.policies.lambda_tuning import tune_tier_lambdas
from tierroute.predictors import NATIVE_C11_RIDGE_SOLVER_ID, BilinearPredictorArtifact
from tierroute.predictors._ridge import RidgeSolution, solve_centered_ridge

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_BUILD_SCRIPT = _REPOSITORY_ROOT / "scripts" / "build_native_ridge.py"
_NATIVE_SOURCE = _REPOSITORY_ROOT / "native" / "tierroute_ridge.c"


def _available_native_compiler() -> Path | None:
    if os.name == "nt":
        candidate = shutil.which("cl")
    else:
        candidate = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    return Path(candidate).resolve() if candidate else None


@pytest.fixture(scope="module")
def cli_compiled_native_ridge(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    compiler = _available_native_compiler()
    if compiler is None or not _NATIVE_SOURCE.is_file():
        pytest.skip("native C11 source or a platform compiler is unavailable")
    output = tmp_path_factory.mktemp("cli-native-ridge") / (
        "tierroute-ridge.exe" if os.name == "nt" else "tierroute-ridge"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(_NATIVE_BUILD_SCRIPT),
            "--output",
            str(output),
            "--compiler",
            str(compiler),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        pytest.fail(f"native build helper failed:\n{completed.stderr}")
    manifest = json.loads(completed.stdout)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["sha256"] == digest
    return output, digest


def test_route_command_shows_decision_cost_and_predicted_quality(capsys: object) -> None:
    status = main(["route", "간단한 질문입니다.", "--tier", "fast"])
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert status == 0
    assert "selected model:" in output
    assert "quoted cost:" in output
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
    assert payload["quoted_cost"] == payload["cost"]
    assert payload["realized_cost"] is None


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


def test_demo_json_is_deterministic_versioned_and_keeps_scopes_separate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo", "--json"]) == 0
    first = capsys.readouterr().out
    assert main(["demo", "--json"]) == 0
    second = capsys.readouterr().out
    payload = json.loads(first)

    assert first == second
    assert (
        hashlib.sha256(first.encode("utf-8")).hexdigest()
        == "2733b41f7e61d33acf8a34ed186ec848ddf86256402dacf74afa80e26d3dd4fa"
    )
    assert payload["schema"] == "tierroute-routing-stream-showcase"
    assert payload["schema_version"] == 1
    assert payload["network_used"] is False
    assert payload["claim_scope"] == "project-authored-synthetic-wiring-only"
    assert payload["accounting"]["budget_scope"] == "independent-per-query-illustrative"
    assert "reporting-only" in payload["accounting"]["cumulative_cost_scope"]
    assert "not a sequence-level oracle" in payload["accounting"]["quality_retention_scope"]

    steps = payload["stream"]["steps"]
    assert [step["example_id"] for step in steps] == [
        "synthetic-science-001",
        "synthetic-math-002",
        "synthetic-code-002",
    ]
    assert [step["tier"] for step in steps] == ["fast", "balanced", "premium"]
    assert [step["routing"]["model"] for step in steps] == ["swift", "steady", "expert"]
    assert all(
        step["routing"]["audited_benchmark_query_match"] is True
        and step["evaluation_scope"]["algorithm"] == "tierroute-evaluation-scope-v1"
        and step["evaluation_scope"]["max_calls_per_query"] == 1
        and len(step["evaluation_scope"]["sha256"]) == 64
        for step in steps
    )
    assert [step["cost"]["realized"] for step in steps] == ["0.2", "0.6", "1"]
    assert [step["cost"]["cumulative_realized_reporting_only"] for step in steps] == [
        "0.2",
        "0.8",
        "1.8",
    ]
    assert all(
        step["quality"]["observed"] <= step["quality"]["per_query_oracle"]["quality"]
        for step in steps
    )
    totals = payload["stream"]["totals"]
    assert totals["realized_cost_reporting_only"] == "1.8"
    assert totals["quality_retention"] == 1.0
    assert totals["quality_retention_exact"] == {"numerator": "1", "denominator": "1"}

    evidence = payload["benchmark_evidence"]
    assert evidence["validation_scope"] == "true-nested-lodo-original-order"
    assert [row["name"] for row in evidence["baselines"]] == [
        "always-cheapest",
        "always-premium",
        "random",
        "length-heuristic",
        "oracle",
        "domain-best-table",
    ]


def test_demo_human_output_shows_stream_and_interpretation_boundaries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo"]) == 0
    output = capsys.readouterr().out

    assert "tierroute offline routing stream showcase" in output
    assert "Step 1 [fast]" in output
    assert "Step 2 [balanced]" in output
    assert "Step 3 [premium]" in output
    assert "project-authored synthetic wiring evidence" in output
    assert "mixed-tier reporting-only" in output
    assert "not a sequence-level oracle or oracle-gap recovery" in output
    assert "Separate full-population learned + six-baseline evidence" in output
    assert "tierroute-nested-lodo" in output
    assert "domain-best-table" in output
    assert "network:            disabled" in output


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
    assert "Evaluation scope: tierroute-evaluation-scope-v1:" in output


def test_baseline_serializer_reads_algorithm_from_report_identity() -> None:
    row = evaluate_six_baselines(load_evaluation_dataset())[0]
    future_report = replace(
        row.report,
        evaluation_scope=replace(
            row.report.evaluation_scope,
            algorithm="tierroute-evaluation-scope-v2",
        ),
    )
    future_row = replace(row, report=future_report)

    assert _baseline_payload(future_row)["evaluation_scope"]["algorithm"] == (
        "tierroute-evaluation-scope-v2"
    )


def test_evaluate_json_declares_lodo_and_domain_fit_scope(capsys: object) -> None:
    assert main(["evaluate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert payload["budget_scope"] == "per-query-illustrative"
    assert payload["validation_scope"] == "outer-lodo-original-order"
    assert payload["domain_table_fit"] == "outer-training-observable-tags-only"
    assert payload["evaluation_scope"]["algorithm"] == "tierroute-evaluation-scope-v1"
    assert len(payload["evaluation_scope"]["sha256"]) == 64
    assert payload["evaluation_scope"]["max_calls_per_query"] == 1
    cheapest = payload["baselines"][0]
    assert cheapest["evaluation_scope"] == payload["evaluation_scope"]
    assert all(
        baseline["evaluation_scope"] == payload["evaluation_scope"]
        for baseline in payload["baselines"]
    )
    assert cheapest["total_realized_cost"] == cheapest["total_cost"] == "4.8"
    evidence = cheapest["cost_evidence"]
    assert evidence["scope"] == ("executed-replay-calls; overall is cross-tier diagnostic only")
    assert evidence["overall"] == {
        "executed_calls": 24,
        "exact_quote_calls": 24,
        "underquoted_calls": 0,
        "overquoted_calls": 0,
        "realized_over_budget_calls": 0,
        "total_quoted_cost": "4.8",
        "total_realized_cost": "4.8",
        "total_absolute_quote_error": "0",
        "total_underquoted_amount": "0",
        "total_overquoted_amount": "0",
        "net_quote_error": {"direction": "equal", "magnitude": "0"},
    }
    assert evidence["by_tier"]["fast"] == {
        "executed_calls": 8,
        "exact_quote_calls": 8,
        "underquoted_calls": 0,
        "overquoted_calls": 0,
        "realized_over_budget_calls": 0,
        "total_quoted_cost": "1.6",
        "total_realized_cost": "1.6",
        "total_absolute_quote_error": "0",
        "total_underquoted_amount": "0",
        "total_overquoted_amount": "0",
        "net_quote_error": {"direction": "equal", "magnitude": "0"},
        "query_count": 8,
        "failed_queries": 0,
        "budget_adapter": "per-query",
        "configured_limit": "0.35",
        "effective_total_limit": "2.8",
        "spent": "1.6",
        "over_budget_calls": 0,
    }
    rejected = payload["baselines"][1]["cost_evidence"]["by_tier"]["fast"]
    assert rejected["executed_calls"] == 0
    assert rejected["failed_queries"] == 8
    assert rejected["total_quoted_cost"] == "0"
    assert rejected["total_realized_cost"] == rejected["spent"] == "0"
    assert rejected["over_budget_calls"] == rejected["realized_over_budget_calls"] == 0


def test_evaluate_json_preserves_offsetting_quote_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = json.loads(bundled_synthetic_path().read_text(encoding="utf-8"))
    first_swift = source["examples"][0]["outcomes"][0]
    second_swift = source["examples"][1]["outcomes"][0]
    first_swift.update({"quoted_cost": "0.20", "cost": "0"})
    second_swift.update({"quoted_cost": "0.20", "cost": "0.40"})
    for example in source["examples"]:
        example["outcomes"][1].update({"quoted_cost": "0.30", "cost": "0.30"})
    replay = tmp_path / "offsetting-costs.json"
    replay.write_text(json.dumps(source), encoding="utf-8")

    assert main(["evaluate", "--data", str(replay), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    evidence = payload["baselines"][0]["cost_evidence"]

    assert evidence["overall"]["total_quoted_cost"] == "4.8"
    assert evidence["overall"]["total_realized_cost"] == "4.8"
    assert evidence["overall"]["total_absolute_quote_error"] == "1.2"
    assert evidence["overall"]["underquoted_calls"] == 3
    assert evidence["overall"]["overquoted_calls"] == 3
    assert evidence["overall"]["net_quote_error"] == {
        "direction": "equal",
        "magnitude": "0",
    }
    assert evidence["by_tier"]["fast"]["total_absolute_quote_error"] == "0.4"
    assert evidence["by_tier"]["fast"]["failed_queries"] == 1
    assert evidence["by_tier"]["fast"]["over_budget_calls"] == 1
    assert evidence["by_tier"]["fast"]["realized_over_budget_calls"] == 1


def test_evaluate_json_serializes_executed_zero_quote_and_zero_cost_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = json.loads(bundled_synthetic_path().read_text(encoding="utf-8"))
    for example in source["examples"]:
        example["outcomes"][0]["quoted_cost"] = "0"
    source["examples"][0]["outcomes"][0]["cost"] = "0"
    replay = tmp_path / "zero-costs.json"
    replay.write_text(json.dumps(source), encoding="utf-8")

    assert main(["evaluate", "--data", str(replay), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    evidence = payload["baselines"][0]["cost_evidence"]["overall"]

    assert evidence["executed_calls"] == 24
    assert evidence["exact_quote_calls"] == 3
    assert evidence["underquoted_calls"] == 21
    assert evidence["total_quoted_cost"] == "0"
    assert evidence["total_realized_cost"] == "4.2"


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
    assert "native_ridge_sha256" not in training

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


def test_explicit_python_reference_selection_preserves_default_json_and_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    default_json_path = tmp_path / "default-json.json"
    explicit_json_path = tmp_path / "explicit-json.json"

    assert main(["train", "--output", str(default_json_path), "--json"]) == 0
    default_json = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "train",
                "--output",
                str(explicit_json_path),
                "--ridge-solver",
                "python-reference",
                "--json",
            ]
        )
        == 0
    )
    explicit_json = json.loads(capsys.readouterr().out)

    assert default_json.pop("artifact") == str(default_json_path)
    assert explicit_json.pop("artifact") == str(explicit_json_path)
    assert default_json == explicit_json
    assert default_json_path.read_bytes() == explicit_json_path.read_bytes()

    default_text_path = tmp_path / "default-text.json"
    explicit_text_path = tmp_path / "explicit-text.json"
    assert main(["train", "--output", str(default_text_path)]) == 0
    default_text = capsys.readouterr().out.replace(str(default_text_path), "<artifact>")
    assert (
        main(
            [
                "train",
                "--output",
                str(explicit_text_path),
                "--ridge-solver",
                "python-reference",
            ]
        )
        == 0
    )
    explicit_text = capsys.readouterr().out.replace(str(explicit_text_path), "<artifact>")

    assert default_text == explicit_text
    assert default_text == (
        "tierroute predictor training\n"
        "  dataset:            tierroute synthetic smoke dataset\n"
        "  training examples:  8\n"
        "  training domains:   code, general, math, science\n"
        "  candidate models:   expert, steady, swift\n"
        "  feature dimension:  7\n"
        "  ridge solver:       tierroute.centered-ridge-cholesky-python-v1\n"
        "  artifact:           <artifact>\n"
        "  network:            disabled\n"
        "  note: synthetic data is a wiring test, not benchmark evidence\n"
    )


def test_train_native_solver_credentials_fail_closed_at_parser_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binary = tmp_path / "native-ridge"
    digest = "0" * 64
    cases = (
        (["--ridge-solver", "native-c11"], "requires both"),
        (
            ["--ridge-solver", "native-c11", "--native-ridge-binary", str(binary)],
            "requires both",
        ),
        (
            ["--ridge-solver", "native-c11", "--native-ridge-sha256", digest],
            "requires both",
        ),
        (["--native-ridge-binary", str(binary)], "require --ridge-solver native-c11"),
        (["--native-ridge-sha256", digest], "require --ridge-solver native-c11"),
        (
            [
                "--ridge-solver",
                "python-reference",
                "--native-ridge-binary",
                str(binary),
                "--native-ridge-sha256",
                digest,
            ],
            "require --ridge-solver native-c11",
        ),
        (
            [
                "--ridge-solver",
                "native-c11",
                "--native-ridge-binary",
                "relative-ridge",
                "--native-ridge-sha256",
                digest,
            ],
            "binary_path must be absolute",
        ),
        (
            [
                "--ridge-solver",
                "native-c11",
                "--native-ridge-binary",
                str(binary),
                "--native-ridge-sha256",
                "A" * 64,
            ],
            "64 lowercase hexadecimal",
        ),
    )

    for index, (options, expected_error) in enumerate(cases):
        output = tmp_path / f"invalid-{index}.json"
        with pytest.raises(SystemExit) as caught:
            main(["train", "--output", str(output), *options])
        assert caught.value.code == 2
        assert expected_error in capsys.readouterr().err
        assert not output.exists()


@pytest.mark.parametrize(
    ("candidate_kind", "expected_error"),
    (("missing", "cannot inspect"), ("wrong-digest", "SHA-256 does not match")),
)
def test_native_authentication_failure_is_controlled_and_writes_no_artifact(
    candidate_kind: str,
    expected_error: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binary = tmp_path / "wrong-native-ridge"
    if candidate_kind == "wrong-digest":
        binary.write_bytes(b"not-the-configured-digest")
        binary.chmod(0o700)
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
                "--budget-scope",
                "per-query",
                "--ridge-solver",
                "native-c11",
                "--native-ridge-binary",
                str(binary),
                "--native-ridge-sha256",
                "0" * 64,
            ]
        )

    assert caught.value.code == 2
    error_output = capsys.readouterr().err
    assert "native ridge training failed" in error_output
    assert expected_error in error_output
    assert "Traceback" not in error_output
    assert not predictor.exists()
    assert not policy.exists()


def test_native_adapter_io_failure_is_cli_controlled_and_writes_no_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binary = tmp_path / "native-ridge"
    binary.write_bytes(b"authenticated-but-I/O-fails-during-snapshot")
    binary.chmod(0o700)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    predictor = tmp_path / "predictor.json"
    policy = tmp_path / "policy.json"

    real_stream_binary = native_ridge_module._stream_binary
    stream_calls = 0

    def fail_snapshot_read(*args: object, **kwargs: object) -> int:
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 2:
            raise OSError("injected snapshot read failure")
        return real_stream_binary(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(native_ridge_module, "_stream_binary", fail_snapshot_read)

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
                "--ridge-solver",
                "native-c11",
                "--native-ridge-binary",
                str(binary),
                "--native-ridge-sha256",
                digest,
            ]
        )

    assert caught.value.code == 2
    error_output = capsys.readouterr().err
    assert "native ridge training failed" in error_output
    assert "snapshot native ridge binary" in error_output
    assert "Traceback" not in error_output
    assert not predictor.exists()
    assert not policy.exists()


@pytest.mark.parametrize("alias_destination", ("predictor", "policy"))
def test_train_rejects_output_aliasing_authenticated_native_binary(
    alias_destination: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binary = tmp_path / "native-ridge"
    original = b"project-owned-native-binary"
    binary.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()

    predictor = binary if alias_destination == "predictor" else tmp_path / "predictor.json"
    policy_options = (
        []
        if alias_destination == "predictor"
        else ["--policy-output", str(binary), "--budget-scope", "per-query"]
    )
    arguments = [
        "train",
        "--output",
        str(predictor),
        *policy_options,
        "--ridge-solver",
        "native-c11",
        "--native-ridge-binary",
        str(binary),
        "--native-ridge-sha256",
        digest,
    ]

    with pytest.raises(SystemExit) as caught:
        main(arguments)

    assert caught.value.code == 2
    assert "protected input path" in capsys.readouterr().err
    assert binary.read_bytes() == original
    if alias_destination == "policy":
        assert not predictor.exists()


def test_compiled_native_train_artifact_routes_without_binary_credentials(
    cli_compiled_native_ridge: tuple[Path, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    executable, digest = cli_compiled_native_ridge
    artifact_path = tmp_path / "native-predictor.json"

    assert (
        main(
            [
                "train",
                "--output",
                str(artifact_path),
                "--ridge-solver",
                "native-c11",
                "--native-ridge-binary",
                str(executable),
                "--native-ridge-sha256",
                digest,
                "--json",
            ]
        )
        == 0
    )
    training_output = capsys.readouterr().out
    training = json.loads(training_output)
    artifact = BilinearPredictorArtifact.load(artifact_path)

    assert training["solver_id"] == NATIVE_C11_RIDGE_SOLVER_ID
    assert training["native_ridge_sha256"] == digest
    assert training["network_used"] is None
    assert training["python_orchestration_network_used"] is False
    assert training["native_binary_audit"] == "caller-responsibility-unapproved"
    assert str(executable) not in training_output
    assert artifact.solver_id == NATIVE_C11_RIDGE_SOLVER_ID
    assert artifact.training_example_count == 8

    assert (
        main(
            [
                "route",
                "Prove a difficult theorem.",
                "--artifact",
                str(artifact_path),
                "--tier",
                "balanced",
                "--json",
            ]
        )
        == 0
    )
    route = json.loads(capsys.readouterr().out)
    assert route["quality_kind"] == "calibrated bilinear artifact"
    assert route["model"] in artifact.model_ids

    text_artifact_path = tmp_path / "native-predictor-text.json"
    assert (
        main(
            [
                "train",
                "--output",
                str(text_artifact_path),
                "--ridge-solver",
                "native-c11",
                "--native-ridge-binary",
                str(executable),
                "--native-ridge-sha256",
                digest,
            ]
        )
        == 0
    )
    native_text = capsys.readouterr().out
    assert "  Python network:     disabled\n" in native_text
    assert "  native audit:       caller-responsibility-unapproved\n" in native_text
    assert "  total network use:  not asserted\n" in native_text
    assert "  network:            disabled\n" not in native_text


def test_native_train_reuses_one_solver_for_artifact_and_policy_cross_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class RecordingNativeSolver:
        solver_id = NATIVE_C11_RIDGE_SOLVER_ID

        def __init__(self) -> None:
            self.solve_calls = 0

        def preflight(
            self,
            *,
            sample_count: int,
            feature_count: int,
            target_count: int,
        ) -> None:
            assert sample_count > 0
            assert feature_count > 0
            assert target_count > 0

        def solve(
            self,
            feature_rows: Sequence[Sequence[float]],
            target_columns: Sequence[Sequence[float]],
            *,
            ridge: float,
        ) -> RidgeSolution:
            self.solve_calls += 1
            return solve_centered_ridge(feature_rows, target_columns, ridge=ridge)

    binary = tmp_path / "native-ridge"
    binary.write_bytes(b"constructor-is-replaced-for-this-wiring-test")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    solver = RecordingNativeSolver()
    constructor_calls: list[tuple[object, object]] = []

    def construct_solver(binary_path: object, expected_sha256: object) -> RecordingNativeSolver:
        constructor_calls.append((binary_path, expected_sha256))
        return solver

    real_fit = cli_module.fit_calibrated_bilinear
    fit_solvers: list[object] = []

    def record_fit(
        training_examples: Sequence[EvaluationExample],
        *,
        config: object = None,
        embedding_provider: object = None,
        solver: object = None,
    ) -> BilinearPredictorArtifact:
        fit_solvers.append(solver)
        return real_fit(
            training_examples,
            config=config,  # type: ignore[arg-type]
            embedding_provider=embedding_provider,  # type: ignore[arg-type]
            solver=solver,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(cli_module, "NativeRidgeAdapter", construct_solver)
    monkeypatch.setattr(cli_module, "fit_calibrated_bilinear", record_fit)
    predictor = tmp_path / "predictor.json"
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
                "--ridge-solver",
                "native-c11",
                "--native-ridge-binary",
                str(binary),
                "--native-ridge-sha256",
                digest,
                "--json",
            ]
        )
        == 0
    )
    training = json.loads(capsys.readouterr().out)

    assert constructor_calls == [(binary, digest)]
    assert len(fit_solvers) == 5
    assert all(item is solver for item in fit_solvers)
    assert solver.solve_calls == 21
    assert training["native_ridge_sha256"] == digest
    assert training["network_used"] is None
    assert training["python_orchestration_network_used"] is False
    assert training["native_binary_audit"] == "caller-responsibility-unapproved"
    assert BilinearPredictorArtifact.load(predictor).solver_id == NATIVE_C11_RIDGE_SOLVER_ID
    LambdaPolicyArtifact.load(policy).validate_predictor(BilinearPredictorArtifact.load(predictor))


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
    assert training["evaluation_scope"]["algorithm"] == "tierroute-evaluation-scope-v1"
    assert len(training["evaluation_scope"]["sha256"]) == 64
    assert training["evaluation_scope"]["max_calls_per_query"] == 1
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
    assert route["remaining_budget"] == "0.5"


def test_policy_route_requires_predictor_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["route", "hello", "--policy-artifact", str(tmp_path / "policy.json")])

    assert caught.value.code == 2
    assert "--policy-artifact requires --artifact" in capsys.readouterr().err
