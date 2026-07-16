# SPDX-License-Identifier: Apache-2.0
"""Paired predictor comparison invariants and descriptive delta semantics."""

from __future__ import annotations

from dataclasses import replace

import pytest

import tierroute.policies.benchmark as benchmark_module
import tierroute.policies.predictor_comparison as comparison_module
from tierroute.adapters import load_evaluation_dataset
from tierroute.core import BudgetTier
from tierroute.eval import summarize_report
from tierroute.policies import (
    COMPARISON_DIRECTION,
    PAIRED_SELECTION_PROTOCOL,
    BenchmarkLambdaSearchConfig,
    PairedPredictorComparison,
    evaluate_per_query_gbm_benchmark,
    evaluate_per_query_paired_predictor_comparison,
)
from tierroute.predictors import GbmTrainingConfig


def _gbm_config() -> GbmTrainingConfig:
    return GbmTrainingConfig(
        n_estimators=2,
        learning_rate=0.2,
        min_samples_leaf=1,
    )


@pytest.fixture(scope="module")
def paired() -> PairedPredictorComparison:
    dataset = load_evaluation_dataset()
    return evaluate_per_query_paired_predictor_comparison(
        dataset.examples,
        dataset.tier_specs,
        gbm_config=_gbm_config(),
        max_candidates_per_tier=17,
    )


def _replace_first_fast_query(
    benchmark: benchmark_module.PerQueryNestedLodoBenchmark,
    **changes: object,
) -> benchmark_module.PerQueryNestedLodoBenchmark:
    report = benchmark.learned.report
    fast, *remaining_tiers = report.tiers
    first_query, *remaining_queries = fast.queries
    changed_query = replace(first_query, **changes)
    changed_fast = replace(fast, queries=(changed_query, *remaining_queries))
    changed_report = replace(report, tiers=(changed_fast, *remaining_tiers))
    changed_learned = replace(
        benchmark.learned,
        report=changed_report,
        score=summarize_report(changed_report),
    )
    return replace(benchmark, learned=changed_learned)


def test_paired_comparison_is_descriptive_and_shares_exact_baselines(
    paired: PairedPredictorComparison,
) -> None:
    assert paired.comparison_direction == COMPARISON_DIRECTION == "gbm-minus-bilinear"
    assert paired.selection_protocol == PAIRED_SELECTION_PROTOCOL == "none-paired-estimation"
    assert paired.selected_family is None
    assert paired.performance_claim_allowed is False
    assert not hasattr(paired, "winner")
    assert paired.bilinear.baselines is paired.gbm.baselines
    assert paired.bilinear.predictor_kind == "calibrated-bilinear-surface-v1"
    assert paired.gbm.predictor_kind == (
        "calibrated-gbm-regression-stumps-surface-v1"
    )
    assert [row.held_out_domain for row in paired.held_out_domain_deltas] == [
        "code",
        "general",
        "math",
        "science",
    ]
    with pytest.raises(TypeError):
        paired.tier_quality_delta[BudgetTier.FAST] = 0.0  # type: ignore[index]


def test_paired_delta_is_raw_gbm_minus_bilinear_without_rounding(
    paired: PairedPredictorComparison,
) -> None:
    first = paired.gbm.learned.report.tiers[0].queries[0]
    assert first.quality is not None
    changed_gbm = _replace_first_fast_query(paired.gbm, quality=first.quality - 0.08)
    changed = PairedPredictorComparison(paired.bilinear, changed_gbm)

    expected_tier_delta = (
        changed_gbm.learned.score.tier_quality[BudgetTier.FAST]
        - paired.bilinear.learned.score.tier_quality[BudgetTier.FAST]  # type: ignore[operator]
    )
    expected_weighted_delta = (
        changed_gbm.learned.score.weighted_quality
        - paired.bilinear.learned.score.weighted_quality  # type: ignore[operator]
    )
    expected_gap_delta = (
        changed_gbm.learned_gap_recovery - paired.bilinear.learned_gap_recovery  # type: ignore[operator]
    )
    assert changed.tier_quality_delta[BudgetTier.FAST] == expected_tier_delta
    assert changed.weighted_quality_delta == expected_weighted_delta
    assert changed.oracle_gap_recovery_delta == expected_gap_delta

    general = next(
        row for row in changed.held_out_domain_deltas if row.held_out_domain == "general"
    )
    assert general.tier_quality_delta[BudgetTier.FAST] == pytest.approx(-0.04)
    assert general.weighted_quality_delta == pytest.approx(-0.02)


def test_unavailable_operand_produces_none_without_weight_redistribution(
    paired: PairedPredictorComparison,
) -> None:
    changed_gbm = _replace_first_fast_query(
        paired.gbm,
        feasible=False,
        selected_model_id=None,
        quality=None,
        output=None,
        predicted_quality=None,
        selected_call_index=None,
        error="project-authored unavailable test row",
    )
    changed = PairedPredictorComparison(paired.bilinear, changed_gbm)

    assert changed.tier_quality_delta[BudgetTier.FAST] is None
    assert changed.tier_quality_delta[BudgetTier.BALANCED] == 0.0
    assert changed.weighted_quality_delta is None
    assert changed.oracle_gap_recovery_delta is None
    general = next(
        row for row in changed.held_out_domain_deltas if row.held_out_domain == "general"
    )
    assert general.tier_quality_delta[BudgetTier.FAST] is None
    assert general.weighted_quality_delta is None
    assert general.oracle_gap_recovery_delta is None
    code = next(row for row in changed.held_out_domain_deltas if row.held_out_domain == "code")
    assert code.weighted_quality_delta == 0.0


def test_pair_rejects_distinct_baseline_object_and_scope_drift(
    paired: PairedPredictorComparison,
) -> None:
    distinct_baselines = replace(paired.gbm.baselines)
    distinct_gbm = replace(paired.gbm, baselines=distinct_baselines)
    with pytest.raises(ValueError, match="exact same baseline object"):
        PairedPredictorComparison(paired.bilinear, distinct_gbm)

    with pytest.raises(ValueError, match="data and replay digests"):
        PairedPredictorComparison(
            paired.bilinear,
            replace(paired.gbm, data_sha256="0" * 64),
        )

    changed_search = replace(
        paired.gbm,
        lambda_search_config=BenchmarkLambdaSearchConfig(
            max_candidates_per_tier=18,
            allow_large_exhaustive=False,
        ),
    )
    with pytest.raises(ValueError, match="lambda search controls"):
        PairedPredictorComparison(paired.bilinear, changed_search)

    changed_membership = replace(paired.gbm)
    object.__setattr__(
        changed_membership,
        "fold_memberships",
        tuple(reversed(changed_membership.fold_memberships)),
    )
    with pytest.raises(ValueError, match="outer-fold membership digests"):
        PairedPredictorComparison(paired.bilinear, changed_membership)


def test_entry_preflights_before_fitting_and_computes_baselines_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    events: list[str] = []
    baseline_calls = 0
    original_pair_preflight = comparison_module.preflight_nested_lodo_gbm
    original_fit = benchmark_module.fit_calibrated_bilinear
    original_baselines = benchmark_module.evaluate_per_query_lodo_baselines

    def recording_pair_preflight(*args: object, **kwargs: object) -> object:
        events.append("aggregate-preflight")
        return original_pair_preflight(*args, **kwargs)  # type: ignore[arg-type]

    def recording_fit(*args: object, **kwargs: object) -> object:
        events.append("bilinear-fit")
        return original_fit(*args, **kwargs)  # type: ignore[arg-type]

    def recording_baselines(*args: object, **kwargs: object) -> object:
        nonlocal baseline_calls
        baseline_calls += 1
        return original_baselines(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        comparison_module,
        "preflight_nested_lodo_gbm",
        recording_pair_preflight,
    )
    monkeypatch.setattr(benchmark_module, "fit_calibrated_bilinear", recording_fit)
    monkeypatch.setattr(
        benchmark_module,
        "evaluate_per_query_lodo_baselines",
        recording_baselines,
    )
    result = evaluate_per_query_paired_predictor_comparison(
        dataset.examples,
        dataset.tier_specs,
        gbm_config=_gbm_config(),
        max_candidates_per_tier=17,
    )

    assert events[0] == "aggregate-preflight"
    assert "bilinear-fit" in events
    assert baseline_calls == 1
    assert result.bilinear.baselines is result.gbm.baselines


def test_public_gbm_benchmark_preflights_before_its_first_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()

    class PreflightSentinel(RuntimeError):
        pass

    def stop_at_preflight(*args: object, **kwargs: object) -> None:
        raise PreflightSentinel

    def unexpected_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("GBM fit must not start before aggregate preflight")

    monkeypatch.setattr(benchmark_module, "preflight_nested_lodo_gbm", stop_at_preflight)
    monkeypatch.setattr(benchmark_module, "fit_calibrated_gbm", unexpected_fit)
    with pytest.raises(PreflightSentinel):
        evaluate_per_query_gbm_benchmark(
            dataset.examples,
            dataset.tier_specs,
            config=_gbm_config(),
            max_candidates_per_tier=17,
        )


def test_showcase_rejects_non_bilinear_benchmark(
    paired: PairedPredictorComparison,
) -> None:
    from tierroute.showcase import build_routing_stream_showcase

    with pytest.raises(ValueError, match="supports only the calibrated bilinear"):
        build_routing_stream_showcase(load_evaluation_dataset(), paired.gbm)
