# SPDX-License-Identifier: Apache-2.0
"""Learned-router nested-LODO benchmark contracts and CLI evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal

import pytest

import tierroute.policies.benchmark as benchmark_module
from tierroute.adapters import (
    PerQueryBudgetLedger,
    bundled_synthetic_path,
    load_evaluation_dataset,
)
from tierroute.cli import main
from tierroute.core import BudgetTier, sum_costs
from tierroute.eval import summarize_report
from tierroute.policies import (
    FOLD_MEMBERSHIP_HASH_ALGORITHM,
    BenchmarkBaselineConfig,
    BenchmarkLambdaSearchConfig,
    PerQueryNestedLodoBenchmark,
    evaluate_per_query_bilinear_benchmark,
    evaluate_per_query_lodo_baselines,
)


@pytest.fixture(scope="module")
def synthetic_benchmark() -> PerQueryNestedLodoBenchmark:
    dataset = load_evaluation_dataset()
    return evaluate_per_query_bilinear_benchmark(
        dataset.examples,
        dataset.tier_specs,
        max_candidates_per_tier=257,
    )


def test_benchmark_aligns_learned_router_and_six_baselines(
    synthetic_benchmark: PerQueryNestedLodoBenchmark,
) -> None:
    result = synthetic_benchmark
    assert result.accounting_scope == "per-query"
    assert result.predictor_kind == "calibrated-bilinear-surface-v1"
    assert result.example_count == 8
    assert result.domains == ("code", "general", "math", "science")
    assert result.model_ids == ("expert", "steady", "swift")
    assert result.baseline_config == BenchmarkBaselineConfig(
        cheap_model_id="swift",
        premium_model_id="expert",
        strong_model_id="expert",
        random_seed=2026,
        character_threshold=120,
    )
    assert result.lambda_search_config == BenchmarkLambdaSearchConfig(
        max_candidates_per_tier=257,
        allow_large_exhaustive=False,
    )
    assert result.data_sha256 == (
        "999d435a40f2db8c76aa205fa3e565b416ab53f6e402979a794a029277b71d60"
    )
    assert result.replay_sha256 == (
        "24be663ca438f388fa3086f638b39c64096a007cd1752820cb4c2ceb1daaa296"
    )
    assert result.learned.prediction_sha256 == (
        "5054653c0509da7685c2c1a3fec2e5492c7e5d935dd4cb1c08b7f040cd277932"
    )
    assert result.learned.report.evaluation_scope_sha256 == (
        "fde4ac2af181ca623238807f33124ab74b38027184e7f5051b61b056276c5aa2"
    )
    assert result.learned.report.max_calls_per_query == 1
    assert all(
        row.report.evaluation_scope == result.learned.report.evaluation_scope
        for row in result.baselines.baselines
    )

    assert result.learned.score.tier_quality == {
        BudgetTier.FAST: pytest.approx(0.60375),
        BudgetTier.BALANCED: pytest.approx(0.81375),
        BudgetTier.PREMIUM: pytest.approx(0.92625),
    }
    assert result.learned.score.weighted_quality == pytest.approx(0.73125)
    assert result.learned_gap_recovery == pytest.approx(1.0)
    assert result.learned_total_cost == Decimal("14.4")
    assert [tier.budget.spent for tier in result.learned.report.tiers] == [
        Decimal("1.6"),
        Decimal("4.8"),
        Decimal("8"),
    ]
    assert all(
        tier.budget.spent == sum_costs(query.cost for query in tier.queries)
        for tier in result.learned.report.tiers
    )
    quote_error = result.learned_quote_error.overall
    assert quote_error.call_count == 24
    assert quote_error.exact_quote_calls == 24
    assert quote_error.total_quoted_cost == Decimal("14.4")
    assert quote_error.total_realized_cost == Decimal("14.4")
    assert quote_error.total_absolute_quote_error == Decimal("0")
    assert all(tier.feasible for tier in result.learned.report.tiers)
    assert tuple(result.baseline_by_name) == (
        "always-cheapest",
        "always-premium",
        "random",
        "length-heuristic",
        "oracle",
        "domain-best-table",
    )


def test_fold_membership_evidence_is_compact_versioned_and_exact(
    synthetic_benchmark: PerQueryNestedLodoBenchmark,
) -> None:
    memberships = synthetic_benchmark.fold_memberships
    assert [
        (item.held_out_domain, item.training_example_count, item.test_example_count)
        for item in memberships
    ] == [
        ("code", 6, 2),
        ("general", 6, 2),
        ("math", 6, 2),
        ("science", 6, 2),
    ]
    assert all(item.algorithm == FOLD_MEMBERSHIP_HASH_ALGORITHM for item in memberships)
    assert [item.sha256 for item in memberships] == [
        "e1e6966a7c2ec729e8fb8fbe4851aeeddf4d453ad0136f43b4737a44e28550f3",
        "242e70f2e27b3d6f135cf8ab9cd22b3e5edf6aec83d152d316c19baae39cf131",
        "b69c68a4971e1bbd25511e49389ab600c35c432ec98b284d0d73ce9ed8d00014",
        "805c7ca31c6c1fcaa75ce30c399de5b2921ba5e7af775d404ebc3e37e9e19f39",
    ]

    with pytest.raises(ValueError, match="init=False"):
        replace(
            synthetic_benchmark,
            fold_memberships=(
                replace(memberships[0], sha256="0" * 64),
                *memberships[1:],
            ),
        )
    with pytest.raises(ValueError, match="init=False"):
        replace(synthetic_benchmark, learned_gap_recovery=1)
    with pytest.raises(ValueError, match="init=False"):
        replace(synthetic_benchmark, learned_total_cost=14)
    with pytest.raises(ValueError, match="always-cheapest calls"):
        replace(synthetic_benchmark.baselines, cheap_model_id="expert")
    with pytest.raises(ValueError, match="replay config evidence"):
        replace(synthetic_benchmark.baselines, random_seed=7)

    dataset = load_evaluation_dataset()
    alternate_baselines = evaluate_per_query_lodo_baselines(
        dataset.examples,
        dataset.tier_specs,
        PerQueryBudgetLedger,
        premium_model_id="expert",
        strong_model_id="expert",
        random_seed=7,
        character_threshold=121,
    )
    alternate_benchmark = replace(synthetic_benchmark, baselines=alternate_baselines)
    assert alternate_benchmark.baseline_config.random_seed == 7
    assert alternate_benchmark.baseline_config.character_threshold == 121
    with pytest.raises(ValueError, match="recorded search cap"):
        replace(
            synthetic_benchmark,
            lambda_search_config=BenchmarkLambdaSearchConfig(
                max_candidates_per_tier=2,
                allow_large_exhaustive=False,
            ),
        )


def test_benchmark_rejects_misaligned_scope_folds_and_query_order(
    synthetic_benchmark: PerQueryNestedLodoBenchmark,
) -> None:
    result = synthetic_benchmark
    learned = result.learned

    mismatched_scope = replace(
        learned.report.evaluation_scope,
        sha256="0" * 64,
    )
    scope_report = replace(learned.report, evaluation_scope=mismatched_scope)
    with pytest.raises(ValueError, match="one evaluation scope"):
        replace(result, learned=replace(learned, report=scope_report))

    reordered_folds = replace(learned, folds=tuple(reversed(learned.folds)))
    with pytest.raises(ValueError, match="outer folds must match"):
        replace(result, learned=reordered_folds)

    reordered_tiers = []
    for tier_result in learned.report.tiers:
        queries = tuple(reversed(tier_result.queries))
        budget = replace(
            tier_result.budget,
            query_order=tuple(query.example_id for query in queries),
        )
        reordered_tiers.append(replace(tier_result, queries=queries, budget=budget))
    reordered_report = replace(learned.report, tiers=tuple(reordered_tiers))
    reordered_learned = replace(
        learned,
        report=reordered_report,
        score=summarize_report(reordered_report),
    )
    with pytest.raises(ValueError, match="preserve the baseline example order"):
        replace(result, learned=reordered_learned)


def test_benchmark_rejects_wrong_ledger_spend_and_per_query_oracle_dominance(
    synthetic_benchmark: PerQueryNestedLodoBenchmark,
) -> None:
    result = synthetic_benchmark
    learned = result.learned

    first_tier = learned.report.tiers[0]
    cumulative_budget = replace(first_tier.budget, adapter_name="cumulative")
    cumulative_report = replace(
        learned.report,
        tiers=(replace(first_tier, budget=cumulative_budget), *learned.report.tiers[1:]),
    )
    with pytest.raises(ValueError, match="per-query accounting"):
        replace(result, learned=replace(learned, report=cumulative_report))

    wrong_spend_budget = replace(first_tier.budget, spent=Decimal("0"))
    wrong_spend_report = replace(
        learned.report,
        tiers=(replace(first_tier, budget=wrong_spend_budget), *learned.report.tiers[1:]),
    )
    with pytest.raises(ValueError, match="spend must equal"):
        replace(result, learned=replace(learned, report=wrong_spend_report))

    first_query, second_query, *remaining_queries = first_tier.queries
    assert first_query.quality is not None
    assert second_query.quality is not None
    shifted_queries = (
        replace(first_query, quality=first_query.quality + 0.001),
        replace(second_query, quality=second_query.quality - 0.001),
        *remaining_queries,
    )
    shifted_tier = replace(first_tier, queries=shifted_queries)
    shifted_report = replace(
        learned.report,
        tiers=(shifted_tier, *learned.report.tiers[1:]),
    )
    shifted_learned = replace(
        learned,
        report=shifted_report,
        score=summarize_report(shifted_report),
    )
    with pytest.raises(ValueError, match="cannot exceed the aligned oracle"):
        replace(result, learned=shifted_learned)


def test_fold_membership_hash_is_unambiguous_ordered_and_utf8_stable() -> None:
    fold_hash = benchmark_module._fold_membership_sha256
    hashes = {
        fold_hash("domain", ("ab", "c"), ("test",)),
        fold_hash("domain", ("a", "bc"), ("test",)),
        fold_hash("domain", ("c", "ab"), ("test",)),
        fold_hash("domain", ("ab",), ("c", "test")),
        fold_hash("other-domain", ("ab", "c"), ("test",)),
    }
    assert len(hashes) == 5

    non_ascii = fold_hash("수학", ("질문-🙂",), ("테스트-🚀",))
    assert non_ascii == fold_hash("수학", ("질문-🙂",), ("테스트-🚀",))
    assert len(non_ascii) == 64

    with pytest.raises(ValueError, match="valid Unicode"):
        fold_hash("\ud800", ("train",), ("test",))


def test_calibrated_bilinear_benchmark_requires_four_domains() -> None:
    dataset = load_evaluation_dataset()
    examples = tuple(example for example in dataset.examples if example.domain != "science")

    with pytest.raises(ValueError, match="requires at least four domains"):
        evaluate_per_query_bilinear_benchmark(examples, dataset.tier_specs)

    with pytest.raises(TypeError, match="config"):
        evaluate_per_query_bilinear_benchmark(
            dataset.examples,
            dataset.tier_specs,
            config=0,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="max_candidates_per_tier"):
        BenchmarkLambdaSearchConfig(  # type: ignore[arg-type]
            max_candidates_per_tier=True,
            allow_large_exhaustive=False,
        )
    with pytest.raises(ValueError, match="requires an exhaustive search"):
        BenchmarkLambdaSearchConfig(
            max_candidates_per_tier=2,
            allow_large_exhaustive=True,
        )


def test_benchmark_json_is_deterministic_versioned_and_explicitly_synthetic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = ["benchmark", "--budget-scope", "per-query", "--json"]
    assert main(arguments) == 0
    first_document = capsys.readouterr().out
    assert main(arguments) == 0
    second_document = capsys.readouterr().out
    assert first_document == second_document
    assert hashlib.sha256(first_document.encode("utf-8")).hexdigest() == (
        "8c0da78c9edbd3585e6afaeb33bd93671ff87f2bc94ee8fa3a158bf0875ac60b"
    )
    assert "synthetic-code-001" not in first_document

    payload = json.loads(first_document)
    assert payload["schema"] == "tierroute-benchmark"
    assert payload["schema_version"] == 1
    assert payload["network_used"] is False
    assert payload["budget_scope"] == "per-query"
    assert payload["validation_scope"] == "true-nested-lodo-original-order"
    assert payload["claim_scope"] == "project-authored-synthetic-wiring-only"
    assert payload["cost_scope"] == ("per-tier ledgers; cross-tier total is diagnostic only")
    assert payload["tier_specs"] == [
        {"tier": "fast", "budget_limit": "0.35", "weight": 0.5},
        {"tier": "balanced", "budget_limit": "0.7", "weight": 0.3},
        {"tier": "premium", "budget_limit": "1", "weight": 0.2},
    ]
    assert payload["lambda_search_config"] == {
        "requested_mode": "bounded-cap",
        "max_candidates_per_tier": 257,
        "allow_large_exhaustive": False,
    }
    baseline_config = payload["baseline_config"]
    assert baseline_config["schema"] == "tierroute-six-baseline-config-v1"
    assert baseline_config["evidence"]["algorithm"] == (
        "tierroute-six-baseline-config-evidence-sha256-v1"
    )
    assert baseline_config["evidence"]["sha256"] == (
        "d25ed40d7f39041122c00bc6b76c412f744fca3b4575135720a1d8d05df58d33"
    )
    assert baseline_config["always_cheapest"]["model_id"] == "swift"
    assert baseline_config["always_premium"]["model_id"] == "expert"
    assert baseline_config["random"]["seed"] == 2026
    assert baseline_config["length_heuristic"] == {
        "character_threshold": 120,
        "cheap_model_id": "swift",
        "strong_model_id": "expert",
        "difficulty_rule": "characters-ge-threshold-or-code-or-math-v1",
    }
    learned_router = payload["learned_router"]
    weights = {row["tier"]: row["weight"] for row in payload["tier_specs"]}
    recomputed_weighted_quality = sum(
        weights[tier] * quality for tier, quality in learned_router["tier_quality"].items()
    ) / sum(weights.values())
    assert recomputed_weighted_quality == pytest.approx(0.73125)
    assert learned_router["weighted_quality"] == pytest.approx(recomputed_weighted_quality)
    assert payload["learned_router"]["oracle_gap_recovery"] == pytest.approx(1.0)
    assert payload["learned_router"]["total_realized_cost"] == "14.4"
    assert [row["name"] for row in payload["baselines"]] == [
        "always-cheapest",
        "always-premium",
        "random",
        "length-heuristic",
        "oracle",
        "domain-best-table",
    ]
    assert all(
        row["evaluation_scope"] == payload["evaluation_scope"]
        for row in [payload["learned_router"], *payload["baselines"]]
    )
    folds = payload["learned_router"]["outer_folds"]
    assert [fold["held_out_domain"] for fold in folds] == [
        "code",
        "general",
        "math",
        "science",
    ]
    assert folds[2]["inner_tuning"]["weighted_quality"] == 0.736333
    assert all(set(fold["membership"]) == {"algorithm", "sha256"} for fold in folds)


def test_benchmark_human_output_labels_scope_cost_and_claims(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["benchmark", "--budget-scope", "per-query"]) == 0
    output = capsys.readouterr().out

    assert "tierroute nested-LODO benchmark" in output
    assert "tierroute-nested-lodo" in output
    assert "domain-best-table" in output
    assert "explicit per-query accounting" in output
    assert "cross-tier total is a diagnostic" in output
    assert "synthetic wiring evidence, not benchmark results" in output
    assert "Network: disabled" in output


def test_benchmark_human_output_labels_user_supplied_claim_responsibility(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "benchmark",
                "--budget-scope",
                "per-query",
                "--data",
                str(bundled_synthetic_path()),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "user-supplied replay" in output
    assert "claim and license validity are caller responsibilities" in output
    assert "bundled values" not in output


def test_benchmark_cli_requires_explicit_safe_search_arguments() -> None:
    with pytest.raises(SystemExit):
        main(["benchmark", "--json"])
    with pytest.raises(SystemExit):
        main(
            [
                "benchmark",
                "--budget-scope",
                "per-query",
                "--allow-large-exhaustive-search",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "benchmark",
                "--budget-scope",
                "per-query",
                "--max-lambda-candidates",
                "1",
            ]
        )
    with pytest.raises(SystemExit):
        main(["benchmark", "--budget-scope", "cumulative"])
    with pytest.raises(SystemExit):
        main(
            [
                "benchmark",
                "--budget-scope",
                "per-query",
                "--exhaustive-lambda-search",
                "--max-lambda-candidates",
                "2",
            ]
        )
