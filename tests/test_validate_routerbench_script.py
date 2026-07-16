# SPDX-License-Identifier: Apache-2.0
"""Tests for local-only RouterBench validation orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

DOMAINS = ("arc-challenge", "hellaswag", "mbpp", "winogrande")
MODEL_IDS = ("cheap", "mid", "premium")


def load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "validate_routerbench.py"
    spec = importlib.util.spec_from_file_location("validate_routerbench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolves postponed annotations through the defining module while
    # decorators run, so register it before executing the script body.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_row(
    sample_id: str,
    domain: str,
    *,
    sequence: int,
    costs: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Build one schema-valid synthetic row with stable column insertion order."""

    realized_costs = costs or (
        0.10 + sequence / 10_000,
        0.20 + sequence / 10_000,
        0.40 + sequence / 10_000,
    )
    row: dict[str, object] = {
        "sample_id": sample_id,
        "prompt": f"private prompt {sample_id}",
        "eval_name": domain,
        "oracle_model_to_route_to": "premium",
    }
    for model_number, (model_id, cost) in enumerate(zip(MODEL_IDS, realized_costs, strict=True)):
        row[model_id] = 0.2 + model_number * 0.3
        row[f"{model_id}|model_response"] = f"private response {sample_id} {model_id}"
        row[f"{model_id}|total_cost"] = cost
    return row


def make_balanced_rows(rows_per_domain: int) -> tuple[dict[str, object], ...]:
    """Interleave domains so restored source order is observable."""

    rows = []
    for index in range(rows_per_domain):
        for domain_number, domain in enumerate(DOMAINS):
            sequence = index * len(DOMAINS) + domain_number
            rows.append(
                make_row(
                    f"private-{domain}-{index}",
                    domain,
                    sequence=sequence,
                )
            )
    return tuple(rows)


def scan_balanced(
    module: ModuleType,
    rows: Sequence[Mapping[str, object]],
    *,
    calibration_per_domain: int = 2,
    evaluation_per_domain: int = 2,
) -> object:
    counts = Counter(str(row["eval_name"]) for row in rows)
    return module._scan_balanced_routerbench_split(
        rows,
        expected_row_count=len(rows),
        expected_domain_counts=counts,
        calibration_per_domain=calibration_per_domain,
        evaluation_per_domain=evaluation_per_domain,
        revision="synthetic-revision",
    )


def make_safe_document(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    rows = make_balanced_rows(3)
    split = scan_balanced(
        module,
        rows,
        calibration_per_domain=1,
        evaluation_per_domain=1,
    )
    monkeypatch.setattr(module, "ROUTERBENCH_CALIBRATION_PER_DOMAIN", 1)
    monkeypatch.setattr(module, "ROUTERBENCH_EVALUATION_PER_DOMAIN", 1)
    monkeypatch.setattr(module, "ROUTERBENCH_MODEL_COUNT", len(MODEL_IDS))
    tier_specs = module._diagnostic_tier_specs(
        {
            "cheap": module.as_cost("0.1"),
            "mid": module.as_cost("0.2"),
            "premium": module.as_cost("0.4"),
        }
    )
    candidate_evidence = SimpleNamespace(
        values=(module.as_cost("0"), module.as_cost("1")),
        exhaustive=False,
        total_derived_values=19,
        strategy="bounded-even-index-v1",
        observed_breakpoint_count=17,
    )
    selection = SimpleNamespace(
        tier=module.BudgetTier.FAST,
        candidates=candidate_evidence,
    )
    held_out_domain = DOMAINS[0]
    membership = SimpleNamespace(
        held_out_domain=held_out_domain,
        training_example_count=3,
        test_example_count=1,
        sha256="a" * 64,
        algorithm="tierroute-test-membership-v1",
    )
    evaluation_scope = SimpleNamespace(
        sha256="b" * 64,
        algorithm="tierroute-test-scope-v1",
    )
    benchmark = SimpleNamespace(
        fold_memberships=(membership,),
        learned=SimpleNamespace(
            folds=(
                SimpleNamespace(
                    held_out_domain=held_out_domain,
                    tuning=SimpleNamespace(selections=(selection,)),
                ),
            ),
            report=SimpleNamespace(evaluation_scope=evaluation_scope),
            prediction_sha256="c" * 64,
        ),
        predictor_kind="calibrated-bilinear-surface-v1",
        accounting_scope="per-query",
        data_sha256="d" * 64,
        replay_sha256="e" * 64,
        training_config=SimpleNamespace(
            ridge=1e-8,
            seed=0,
            solver_id="tierroute-test-solver-v1",
        ),
        lambda_search_config=SimpleNamespace(requested_mode="bounded"),
        baselines=SimpleNamespace(baseline_config_evidence_sha256="f" * 64),
    )
    return module._diagnostic_document(split, tier_specs, benchmark)


def nested_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key) for key in value),
            *(key for child in value.values() for key in nested_keys(child)),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {key for child in value for key in nested_keys(child)}
    return set()


def test_validation_script_replays_without_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch.setattr(module, "iter_routerbench_rows", lambda _: iter(rows))
    semantic_sha256, _, _, _, _ = module._scan_semantic_rows(rows, expected_row_count=1)
    monkeypatch.setattr(module, "ROUTERBENCH_ROW_COUNT", 1)
    monkeypatch.setattr(module, "ROUTERBENCH_COLUMN_COUNT", len(rows[0]))
    monkeypatch.setattr(module, "ROUTERBENCH_IN_SCOPE_COUNT", 1)
    monkeypatch.setattr(module, "ROUTERBENCH_MODEL_COUNT", 2)
    monkeypatch.setattr(module, "ROUTERBENCH_DOMAIN_COUNTS", {"hellaswag": 1})
    monkeypatch.setattr(module, "ROUTERBENCH_SEMANTIC_SHA256", semantic_sha256)

    module.validate_and_replay(tmp_path / "unused.pkl", replay_limit=1)

    output = capsys.readouterr().out
    assert "In-scope examples: 1" in output
    assert f"Semantic SHA-256: {semantic_sha256}" in output
    assert "Dataset license: NOASSERTION" in output
    assert "Always-cheapest mean quality" not in output
    assert "Replay cost" not in output


def test_semantic_digest_framing_preserves_utf8_and_negative_zero() -> None:
    module = load_script()
    rows = ({"text": "é", "score": -0.0}, {"text": "β", "score": 1.5})
    digest = module._SemanticDigestBuilder(
        expected_row_count=2,
        columns=("text", "score"),
    )

    for row in rows:
        digest.update(row)

    assert digest.hexdigest() == (
        "66b3e912abd30d17dde3a2f735acba1a54f4392cb95c35becd1ed403c4a0ea44"
    )


def test_balanced_split_keeps_bottom_k_and_restores_evaluation_source_order() -> None:
    module = load_script()
    rows = make_balanced_rows(6)

    split = scan_balanced(module, rows)

    calibration_ids = {selected.sample_id for selected in split.calibration}
    evaluation_ids = {selected.sample_id for selected in split.evaluation}
    assert len(split.calibration) == len(DOMAINS) * 2
    assert len(split.evaluation) == len(DOMAINS) * 2
    assert calibration_ids.isdisjoint(evaluation_ids)
    assert Counter(selected.domain for selected in split.calibration) == Counter(
        {domain: 2 for domain in DOMAINS}
    )
    assert Counter(selected.domain for selected in split.evaluation) == Counter(
        {domain: 2 for domain in DOMAINS}
    )
    assert [selected.row_number for selected in split.evaluation] == sorted(
        selected.row_number for selected in split.evaluation
    )

    for domain in DOMAINS:
        ranked = sorted(
            (
                (
                    module._selection_rank_sha256(
                        domain,
                        str(row["sample_id"]),
                        revision="synthetic-revision",
                    ),
                    row_number,
                    str(row["sample_id"]),
                )
                for row_number, row in enumerate(rows)
                if row["eval_name"] == domain
            ),
            key=lambda item: (item[0], item[1]),
        )
        expected_calibration = {item[2] for item in ranked[:2]}
        expected_evaluation = {item[2] for item in ranked[2:4]}
        assert {
            selected.sample_id for selected in split.calibration if selected.domain == domain
        } == expected_calibration
        assert {
            selected.sample_id for selected in split.evaluation if selected.domain == domain
        } == expected_evaluation


def test_balanced_membership_ignores_prompt_quality_cost_and_response_mutations() -> None:
    module = load_script()
    rows = make_balanced_rows(6)
    mutated_rows = []
    for row_number, source in enumerate(rows):
        row = dict(source)
        row["prompt"] = f"mutated secret prompt {row_number}"
        for model_number, model_id in enumerate(MODEL_IDS):
            row[model_id] = 0.99 - model_number / 10
            row[f"{model_id}|model_response"] = f"mutated secret response {row_number} {model_id}"
            row[f"{model_id}|total_cost"] = 2.0 + model_number + row_number / 1_000
        mutated_rows.append(row)

    original = scan_balanced(module, rows)
    mutated = scan_balanced(module, mutated_rows)

    def identity(split: object, role: str) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                selected.domain,
                selected.sample_id,
                selected.rank_sha256,
                selected.row_number,
            )
            for selected in getattr(split, role)
        )

    assert original.semantic_sha256 != mutated.semantic_sha256
    assert original.split_sha256 == mutated.split_sha256
    assert identity(original, "calibration") == identity(mutated, "calibration")
    assert identity(original, "evaluation") == identity(mutated, "evaluation")


def test_balanced_split_rejects_duplicate_mapped_sample_ids() -> None:
    module = load_script()
    rows = list(make_balanced_rows(3))
    rows[-1] = {**rows[-1], "sample_id": rows[0]["sample_id"]}

    with pytest.raises(ValueError, match="unique sample IDs"):
        scan_balanced(
            module,
            rows,
            calibration_per_domain=1,
            evaluation_per_domain=1,
        )


def test_balanced_split_requires_four_domains() -> None:
    module = load_script()
    rows = tuple(row for row in make_balanced_rows(3) if row["eval_name"] in DOMAINS[:3])

    with pytest.raises(ValueError, match="at least four domains"):
        scan_balanced(
            module,
            rows,
            calibration_per_domain=1,
            evaluation_per_domain=1,
        )


def test_balanced_split_rejects_declared_domain_too_small() -> None:
    module = load_script()
    rows = make_balanced_rows(1)

    with pytest.raises(ValueError, match="at least 2 rows"):
        scan_balanced(
            module,
            rows,
            calibration_per_domain=1,
            evaluation_per_domain=1,
        )


def test_balanced_split_rejects_unexpected_mapped_domain() -> None:
    module = load_script()
    rows = make_balanced_rows(3)
    unexpected = (*rows, make_row("private-gsm8k", "grade-school-math", sequence=len(rows)))
    counts = {domain: 3 for domain in DOMAINS}

    with pytest.raises(ValueError, match="unexpected mapped domain 'gsm8k'"):
        module._scan_balanced_routerbench_split(
            unexpected,
            expected_row_count=len(unexpected),
            expected_domain_counts=counts,
            calibration_per_domain=1,
            evaluation_per_domain=1,
            revision="synthetic-revision",
        )


def test_balanced_split_rejects_observed_domain_count_mismatch() -> None:
    module = load_script()
    rows = make_balanced_rows(3)
    counts = {domain: 3 for domain in DOMAINS}
    counts[DOMAINS[0]] = 4

    with pytest.raises(ValueError, match="domain counts mismatch"):
        module._scan_balanced_routerbench_split(
            rows,
            expected_row_count=len(rows),
            expected_domain_counts=counts,
            calibration_per_domain=1,
            evaluation_per_domain=1,
            revision="synthetic-revision",
        )


def test_calibration_quotes_are_per_model_maxima() -> None:
    module = load_script()
    raw_rows = (
        make_row("private-calibration-1", DOMAINS[0], sequence=0, costs=(0.10, 0.24, 0.41)),
        make_row("private-calibration-2", DOMAINS[1], sequence=1, costs=(0.16, 0.20, 0.47)),
        make_row("private-calibration-3", DOMAINS[2], sequence=2, costs=(0.12, 0.22, 0.43)),
    )
    selected = tuple(
        module._SelectedRouterBenchRow(
            row_number=row_number,
            domain=str(row["eval_name"]),
            sample_id=str(row["sample_id"]),
            rank_sha256=f"{row_number:064x}",
            row=row,
        )
        for row_number, row in enumerate(raw_rows)
    )

    quotes = module._maximum_calibration_quotes(selected, model_ids=MODEL_IDS)

    assert {model_id: module.canonical_cost_text(cost) for model_id, cost in quotes.items()} == {
        "cheap": "0.16",
        "mid": "0.24",
        "premium": "0.47",
    }


def test_calibration_quotes_reject_inconsistent_model_catalogue() -> None:
    module = load_script()
    first = make_row("private-calibration-1", DOMAINS[0], sequence=0)
    changed = make_row("private-calibration-2", DOMAINS[1], sequence=1)
    del changed["mid|model_response"]
    selected = tuple(
        module._SelectedRouterBenchRow(
            row_number=row_number,
            domain=str(row["eval_name"]),
            sample_id=str(row["sample_id"]),
            rank_sha256=f"{row_number:064x}",
            row=row,
        )
        for row_number, row in enumerate((first, changed))
    )

    with pytest.raises(ValueError, match="changed the model catalogue"):
        module._maximum_calibration_quotes(selected, model_ids=MODEL_IDS)


def test_evaluation_quote_overrun_aborts_before_predictor_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    split = scan_balanced(
        module,
        make_balanced_rows(3),
        calibration_per_domain=1,
        evaluation_per_domain=1,
    )
    # Mutating a retained evaluation payload does not alter membership evidence,
    # and creates a deliberate quote underrun relative to calibration-only maxima.
    split.evaluation[0].row["cheap|total_cost"] = 99.0
    benchmark_called = False

    def unexpected_benchmark(*args: object, **kwargs: object) -> object:
        nonlocal benchmark_called
        benchmark_called = True
        raise AssertionError("predictor fitting must not start after a quote overrun")

    monkeypatch.setattr(module, "iter_routerbench_rows", lambda _: iter(()))
    monkeypatch.setattr(module, "_scan_balanced_routerbench_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(module, "_validate_authenticated_metadata", lambda **kwargs: None)
    monkeypatch.setattr(module, "ROUTERBENCH_MODEL_COUNT", len(MODEL_IDS))
    monkeypatch.setattr(module, "ROUTERBENCH_DOMAIN_COUNTS", {domain: 3 for domain in DOMAINS})
    monkeypatch.setattr(module, "ROUTERBENCH_EVALUATION_PER_DOMAIN", 1)
    monkeypatch.setattr(module, "evaluate_per_query_bilinear_benchmark", unexpected_benchmark)

    with pytest.raises(ValueError, match="exceeds a calibration-only quote"):
        module.validate_nested_lodo(tmp_path / "unused.pkl")

    assert not benchmark_called


def test_validate_nested_lodo_synthetic_end_to_end_uses_real_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    rows = []
    for index in range(4):
        for domain_number, domain in enumerate(DOMAINS):
            sequence = index * len(DOMAINS) + domain_number
            row = make_row(
                f"private-e2e-{domain}-{index}",
                domain,
                sequence=sequence,
                costs=(0.10, 0.20, 0.40),
            )
            # Give the real trainer non-constant, bounded targets without encoding
            # split membership in any predictor-visible outcome value.
            row["cheap"] = 0.20 + sequence / 100
            row["mid"] = 0.45 + sequence / 100
            row["premium"] = 0.70 + sequence / 100
            rows.append(row)
    synthetic_rows = tuple(rows)
    original_scan = module._scan_balanced_routerbench_split

    def scan_small_fixed_scope(
        raw_rows: object,
        *,
        expected_row_count: int,
        expected_domain_counts: Mapping[str, int],
    ) -> object:
        return original_scan(
            raw_rows,
            expected_row_count=expected_row_count,
            expected_domain_counts=expected_domain_counts,
            calibration_per_domain=2,
            evaluation_per_domain=2,
            revision="synthetic-revision",
        )

    domain_counts = {domain: 4 for domain in DOMAINS}
    semantic_sha256, _, _, _, _ = module._scan_semantic_rows(
        synthetic_rows,
        expected_row_count=len(synthetic_rows),
    )
    monkeypatch.setattr(module, "iter_routerbench_rows", lambda _: iter(synthetic_rows))
    monkeypatch.setattr(module, "_scan_balanced_routerbench_split", scan_small_fixed_scope)
    monkeypatch.setattr(module, "ROUTERBENCH_REVISION", "synthetic-revision")
    monkeypatch.setattr(module, "ROUTERBENCH_SHA256", "1" * 64)
    monkeypatch.setattr(module, "ROUTERBENCH_ROW_COUNT", len(synthetic_rows))
    monkeypatch.setattr(module, "ROUTERBENCH_COLUMN_COUNT", len(synthetic_rows[0]))
    monkeypatch.setattr(module, "ROUTERBENCH_IN_SCOPE_COUNT", len(synthetic_rows))
    monkeypatch.setattr(module, "ROUTERBENCH_MODEL_COUNT", len(MODEL_IDS))
    monkeypatch.setattr(module, "ROUTERBENCH_DOMAIN_COUNTS", domain_counts)
    monkeypatch.setattr(module, "ROUTERBENCH_SEMANTIC_SHA256", semantic_sha256)
    monkeypatch.setattr(module, "ROUTERBENCH_CALIBRATION_PER_DOMAIN", 2)
    monkeypatch.setattr(module, "ROUTERBENCH_EVALUATION_PER_DOMAIN", 2)

    document = module.validate_nested_lodo(tmp_path / "does-not-exist.pkl")

    assert document["execution_status"] == "completed"
    assert document["result_status"] == "diagnostic"
    assert document["dataset_license"] == "NOASSERTION"
    assert document["redistribution_authorized"] is False
    assert document["official_skt_data"] is False
    assert document["competition_score"] is False
    assert document["network_used"] is False
    assert document["artifact_written_by_validator"] is False
    assert document["performance_metrics_published"] is False
    assert document["row_level_results_published"] is False
    assert document["evaluation"]["protocol"] == "true-nested-leave-one-domain-out"
    assert document["evaluation"]["baseline_names"] == list(module.BASELINE_NAMES)
    assert document["evaluation"]["baseline_count"] == 6
    assert document["evaluation"]["fold_count"] == len(DOMAINS)
    assert document["split"]["calibration_example_count"] == 8
    assert document["split"]["evaluation_example_count"] == 8
    assert document["quoted_costs"]["evaluation_preflight"] == "passed"
    assert document["quoted_costs"]["evaluation_quote_overrun_count"] == 0

    serialized = json.dumps(document, ensure_ascii=False, sort_keys=True)
    assert "private prompt" not in serialized
    assert "private response" not in serialized
    assert all(str(row["sample_id"]) not in serialized for row in synthetic_rows)
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--nested-lodo",), "--nested-lodo requires --acknowledge-noassertion"),
        (("--json",), "--json is only valid with --nested-lodo"),
        (
            ("--nested-lodo", "--acknowledge-noassertion", "--limit", "1"),
            "--limit cannot be combined with the fixed --nested-lodo scope",
        ),
    ),
)
def test_nested_cli_rejects_unsafe_or_ambiguous_combinations(
    arguments: tuple[str, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()
    monkeypatch.setattr(
        module,
        "validate_nested_lodo",
        lambda _: pytest.fail("invalid CLI arguments must fail before validation"),
    )

    with pytest.raises(SystemExit) as error:
        module.main(arguments)

    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_safe_json_contains_provenance_labels_but_no_private_rows_or_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()
    document = make_safe_document(module, monkeypatch)
    monkeypatch.setattr(module, "validate_nested_lodo", lambda _: document)

    module.main(("--nested-lodo", "--acknowledge-noassertion", "--json"))

    output = capsys.readouterr().out
    decoded = json.loads(output)
    assert decoded == document
    assert decoded["result_status"] == "diagnostic"
    assert decoded["claim_scope"] == ("external-routerbench-local-only-non-official-non-reportable")
    assert decoded["dataset_license"] == "NOASSERTION"
    assert decoded["redistribution_authorized"] is False
    assert decoded["official_skt_data"] is False
    assert decoded["competition_score"] is False
    assert decoded["feature_set"] == "surface-only"
    assert decoded["bge_m3_used"] is False
    assert decoded["budget_profile_official"] is False
    assert decoded["network_used"] is False
    assert decoded["artifact_written_by_validator"] is False
    assert decoded["performance_metrics_published"] is False
    assert decoded["row_level_results_published"] is False
    assert decoded["evaluation"]["baseline_names"] == list(module.BASELINE_NAMES)

    forbidden_keys = {
        "example_id",
        "gap_recovery",
        "mean_cost",
        "mean_quality",
        "oracle_gap_recovery",
        "output",
        "prompt",
        "quality",
        "response",
        "route",
        "sample_id",
        "selected_model_id",
        "total_cost",
        "weighted_quality",
    }
    assert nested_keys(decoded).isdisjoint(forbidden_keys)
    for private_value in (
        "private-arc-challenge-0",
        "private prompt",
        "private response",
    ):
        assert private_value not in output


def test_safe_human_output_warns_and_suppresses_private_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script()
    document = make_safe_document(module, monkeypatch)

    module._print_nested_diagnostic(document)

    output = capsys.readouterr().out
    assert output.startswith(module.ROUTERBENCH_DIAGNOSTIC_WARNING)
    assert "NOASSERTION" in output
    assert "not SKT data and not a competition score" in output
    assert "Performance, cost, gap, route, and row-level results: suppressed" in output
    assert "Validator-created artifact: none; network used: no" in output
    for private_value in (
        "private-arc-challenge-0",
        "private prompt",
        "private response",
    ):
        assert private_value not in output


def test_local_pinned_artifact_matches_semantic_golden_and_balanced_scope() -> None:
    module = load_script()
    artifact = Path(__file__).parents[1] / "data/routerbench/routerbench_0shot.pkl"
    if not artifact.is_file():
        pytest.skip("external NOASSERTION RouterBench artifact is not present")

    split = module._scan_balanced_routerbench_split(
        module.iter_routerbench_rows(artifact),
        expected_row_count=module.ROUTERBENCH_ROW_COUNT,
        expected_domain_counts=module.ROUTERBENCH_DOMAIN_COUNTS,
    )
    response_models = {
        column.removesuffix("|model_response")
        for column in split.columns
        if column.endswith("|model_response")
    }
    calibration_ids = {selected.sample_id for selected in split.calibration}
    evaluation_ids = {selected.sample_id for selected in split.evaluation}

    assert split.semantic_sha256 == module.ROUTERBENCH_SEMANTIC_SHA256
    assert len(split.columns) == module.ROUTERBENCH_COLUMN_COUNT
    assert split.in_scope_count == module.ROUTERBENCH_IN_SCOPE_COUNT
    assert len(response_models) == module.ROUTERBENCH_MODEL_COUNT
    assert dict(split.domain_counts) == module.ROUTERBENCH_DOMAIN_COUNTS
    assert len(split.calibration) == (
        len(module.ROUTERBENCH_DOMAIN_COUNTS) * module.ROUTERBENCH_CALIBRATION_PER_DOMAIN
    )
    assert len(split.evaluation) == (
        len(module.ROUTERBENCH_DOMAIN_COUNTS) * module.ROUTERBENCH_EVALUATION_PER_DOMAIN
    )
    assert calibration_ids.isdisjoint(evaluation_ids)
    assert Counter(selected.domain for selected in split.calibration) == Counter(
        {
            domain: module.ROUTERBENCH_CALIBRATION_PER_DOMAIN
            for domain in module.ROUTERBENCH_DOMAIN_COUNTS
        }
    )
    assert Counter(selected.domain for selected in split.evaluation) == Counter(
        {
            domain: module.ROUTERBENCH_EVALUATION_PER_DOMAIN
            for domain in module.ROUTERBENCH_DOMAIN_COUNTS
        }
    )
    assert [selected.row_number for selected in split.evaluation] == sorted(
        selected.row_number for selected in split.evaluation
    )
