# SPDX-License-Identifier: Apache-2.0
"""Tests for weighted tier quality and oracle-gap recovery."""

from dataclasses import replace
from decimal import Decimal, Inexact, Rounded, localcontext

import pytest

from tierroute.core import BudgetTier
from tierroute.eval import (
    BudgetReport,
    EvaluationReport,
    ExactCostDifference,
    QueryResult,
    QuoteCostDirection,
    QuoteErrorReport,
    QuoteErrorSummary,
    ReplayCall,
    TierQuoteErrorSummary,
    TierResult,
    TierSpec,
    oracle_gap_recovery,
    summarize_quote_error,
    summarize_report,
    weighted_delta,
)

WEIGHTS = {
    BudgetTier.FAST: 0.5,
    BudgetTier.BALANCED: 0.3,
    BudgetTier.PREMIUM: 0.2,
}


def report(name: str, qualities: dict[BudgetTier, float | None]) -> EvaluationReport:
    tiers = []
    for tier, quality in qualities.items():
        calls = (
            (
                ReplayCall(
                    "model",
                    Decimal("1"),
                    Decimal("1"),
                    Decimal("1"),
                    Decimal(0),
                    True,
                ),
            )
            if quality is not None
            else ()
        )
        query = QueryResult(
            "q1",
            tier,
            quality is not None,
            "model" if quality is not None else None,
            Decimal("1") if quality is not None else Decimal(0),
            quality,
            "output" if quality is not None else None,
            error=None if quality is not None else "infeasible",
            calls=calls,
            selected_call_index=0 if quality is not None else None,
        )
        budget = BudgetReport("test", Decimal("1"), Decimal("1"), query.cost, 0, ("q1",))
        tiers.append(TierResult(TierSpec(tier, Decimal("1"), WEIGHTS[tier]), (query,), budget))
    return EvaluationReport(name, tuple(tiers))


def quote_error_report() -> EvaluationReport:
    calls = (
        ReplayCall("under", Decimal("0.1"), Decimal("0.3"), Decimal("1"), Decimal("0.7"), True),
        ReplayCall("over", Decimal("0.4"), Decimal("0.2"), Decimal("0.7"), Decimal("0.5"), True),
        ReplayCall("exact", Decimal(0), Decimal(0), Decimal("0.5"), Decimal("0.5"), True),
    )
    query = QueryResult(
        "q1",
        BudgetTier.FAST,
        True,
        "under",
        Decimal("0.5"),
        0.5,
        "answer",
        calls=calls,
        selected_call_index=0,
    )
    budget = BudgetReport(
        "per-query",
        Decimal("1"),
        Decimal("1"),
        Decimal("0.5"),
        0,
        ("q1",),
    )
    return EvaluationReport(
        "router",
        (TierResult(TierSpec(BudgetTier.FAST, Decimal("1"), 1.0), (query,), budget),),
    )


def test_weighted_tier_quality_uses_explicit_weights() -> None:
    result = report(
        "router",
        {
            BudgetTier.FAST: 0.6,
            BudgetTier.BALANCED: 0.8,
            BudgetTier.PREMIUM: 0.9,
        },
    )

    assert summarize_report(result).weighted_quality == pytest.approx(0.72)


def test_quote_error_preserves_offsetting_call_errors() -> None:
    summary = summarize_quote_error(quote_error_report()).overall

    assert summary.call_count == 3
    assert summary.total_quoted_cost == Decimal("0.5")
    assert summary.total_realized_cost == Decimal("0.5")
    assert summary.total_absolute_quote_error == Decimal("0.4")
    assert summary.total_underquoted_amount == Decimal("0.2")
    assert summary.total_overquoted_amount == Decimal("0.2")
    assert summary.underquoted_calls == 1
    assert summary.overquoted_calls == 1
    assert summary.exact_quote_calls == 1
    assert summary.realized_over_budget_calls == 0
    assert summary.net_quote_error == ExactCostDifference(
        QuoteCostDirection.EQUAL,
        Decimal(0),
    )


def test_quote_error_rejects_budget_and_call_evidence_mismatch() -> None:
    valid = quote_error_report()
    tier = valid.tiers[0]

    wrong_spend = EvaluationReport(
        "wrong-spend",
        (replace(tier, budget=replace(tier.budget, spent=Decimal("0.4"))),),
    )
    wrong_overruns = EvaluationReport(
        "wrong-overruns",
        (replace(tier, budget=replace(tier.budget, over_budget_calls=1)),),
    )

    with pytest.raises(ValueError, match="budget spend"):
        summarize_quote_error(wrong_spend)
    with pytest.raises(ValueError, match="over-budget call evidence"):
        summarize_quote_error(wrong_overruns)


@pytest.mark.parametrize(
    ("quoted", "realized", "direction"),
    [
        (
            Decimal("1.00000000000000000000000000004"),
            Decimal("0.99999999999999999999999999999"),
            QuoteCostDirection.REALIZED_BELOW_QUOTE,
        ),
        (
            Decimal("0.99999999999999999999999999999"),
            Decimal("1.00000000000000000000000000004"),
            QuoteCostDirection.REALIZED_ABOVE_QUOTE,
        ),
    ],
)
def test_quote_error_is_exact_under_decimal_rounding_traps(
    quoted: Decimal,
    realized: Decimal,
    direction: QuoteCostDirection,
) -> None:
    call = ReplayCall(
        "model",
        quoted,
        realized,
        Decimal("2"),
        Decimal(0),
        True,
    )
    query = QueryResult(
        "q1",
        BudgetTier.FAST,
        True,
        "model",
        realized,
        0.5,
        "answer",
        calls=(call,),
        selected_call_index=0,
    )
    tier = TierResult(
        TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),
        (query,),
        BudgetReport("per-query", Decimal("2"), Decimal("2"), realized, 0, ("q1",)),
    )

    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        summary = summarize_quote_error(EvaluationReport("router", (tier,))).overall

    assert summary.net_quote_error == ExactCostDifference(
        direction,
        Decimal("5e-29"),
    )


def test_quote_error_uses_adapter_outcomes_without_inventing_balance_semantics() -> None:
    call = ReplayCall(
        "model",
        Decimal("0.8"),
        Decimal("1.2"),
        Decimal("1"),
        Decimal("0.25"),
        False,
    )
    query = QueryResult(
        "q1",
        BudgetTier.FAST,
        False,
        None,
        Decimal("1.2"),
        None,
        None,
        error="adapter rejected the realized charge",
        calls=(call,),
    )
    tier = TierResult(
        TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),
        (query,),
        BudgetReport("custom", Decimal("1"), Decimal("1"), Decimal("1.2"), 1, ("q1",)),
    )

    summary = summarize_quote_error(EvaluationReport("router", (tier,))).overall

    assert summary.call_count == 1
    assert summary.underquoted_calls == 1
    assert summary.realized_over_budget_calls == 1
    assert summary.total_underquoted_amount == Decimal("0.4")


def test_quote_error_summary_rejects_impossible_public_values() -> None:
    summary = summarize_quote_error(quote_error_report()).overall

    with pytest.raises(ValueError, match="zero-call"):
        replace(
            summary,
            call_count=0,
            exact_quote_calls=0,
            underquoted_calls=0,
            overquoted_calls=0,
            total_absolute_quote_error=Decimal(0),
            total_underquoted_amount=Decimal(0),
            total_overquoted_amount=Decimal(0),
        )
    with pytest.raises(ValueError, match="underquote count and amount"):
        replace(summary, exact_quote_calls=2, underquoted_calls=0)
    with pytest.raises(ValueError, match="overquote count and amount"):
        replace(summary, exact_quote_calls=2, overquoted_calls=0)


def test_quote_error_report_rejects_duplicate_or_misaligned_tier_totals() -> None:
    tier_summary = summarize_quote_error(quote_error_report()).overall
    fast = TierQuoteErrorSummary(BudgetTier.FAST, tier_summary)
    balanced = TierQuoteErrorSummary(BudgetTier.BALANCED, tier_summary)
    doubled = QuoteErrorSummary(
        call_count=6,
        exact_quote_calls=2,
        underquoted_calls=2,
        overquoted_calls=2,
        realized_over_budget_calls=0,
        total_quoted_cost=Decimal("1"),
        total_realized_cost=Decimal("1"),
        total_absolute_quote_error=Decimal("0.8"),
        total_underquoted_amount=Decimal("0.4"),
        total_overquoted_amount=Decimal("0.4"),
        net_quote_error=ExactCostDifference(QuoteCostDirection.EQUAL, Decimal(0)),
    )

    assert QuoteErrorReport((fast, balanced), doubled).overall == doubled
    with pytest.raises(ValueError, match="unique tiers"):
        QuoteErrorReport((fast, fast), doubled)
    with pytest.raises(ValueError, match="exact tier aggregate"):
        QuoteErrorReport((fast, balanced), tier_summary)


def test_infeasible_tier_is_not_silently_renormalized() -> None:
    result = report("router", {BudgetTier.FAST: 0.6, BudgetTier.BALANCED: None})

    assert summarize_report(result).weighted_quality is None


def test_oracle_gap_recovery_handles_midpoint_negative_and_zero_gap() -> None:
    cheapest = report("cheap", {BudgetTier.FAST: 0.4})
    oracle = report("oracle", {BudgetTier.FAST: 0.8})

    assert oracle_gap_recovery(
        report("mid", {BudgetTier.FAST: 0.6}), cheapest, oracle
    ) == pytest.approx(0.5)
    assert oracle_gap_recovery(
        report("bad", {BudgetTier.FAST: 0.2}), cheapest, oracle
    ) == pytest.approx(-0.5)
    assert oracle_gap_recovery(cheapest, cheapest, cheapest) is None


def test_oracle_gap_recovery_detects_invalid_upper_bound() -> None:
    cheapest = report("cheap", {BudgetTier.FAST: 0.5})
    low_oracle = report("oracle", {BudgetTier.FAST: 0.4})
    high_router = report("router", {BudgetTier.FAST: 0.9})
    oracle = report("oracle", {BudgetTier.FAST: 0.8})

    with pytest.raises(ValueError, match="below cheapest"):
        oracle_gap_recovery(cheapest, cheapest, low_oracle)
    with pytest.raises(ValueError, match="exceeds oracle"):
        oracle_gap_recovery(high_router, cheapest, oracle)


def test_oracle_gap_recovery_rejects_duplicate_tiers() -> None:
    cheapest = report("cheap", {BudgetTier.FAST: 0.4})
    oracle = report("oracle", {BudgetTier.FAST: 0.8})
    duplicate = EvaluationReport("duplicate", cheapest.tiers * 2)

    with pytest.raises(ValueError, match="duplicate tier"):
        oracle_gap_recovery(duplicate, cheapest, oracle)


def test_oracle_gap_recovery_rejects_population_or_budget_scope_mismatch() -> None:
    cheapest = report("cheap", {BudgetTier.FAST: 0.4})
    oracle = report("oracle", {BudgetTier.FAST: 0.8})
    base_tier = cheapest.tiers[0]
    changed_query = replace(base_tier.queries[0], example_id="q2")
    other_population = EvaluationReport(
        "other-population",
        (
            replace(
                base_tier,
                queries=(changed_query,),
                budget=replace(base_tier.budget, query_order=("q2",)),
            ),
        ),
    )
    other_adapter = EvaluationReport(
        "other-adapter",
        (replace(base_tier, budget=replace(base_tier.budget, adapter_name="cumulative")),),
    )

    with pytest.raises(ValueError, match="evaluation scope mismatch"):
        oracle_gap_recovery(other_population, cheapest, oracle)
    with pytest.raises(ValueError, match="evaluation scope mismatch"):
        oracle_gap_recovery(other_adapter, cheapest, oracle)
    with pytest.raises(ValueError, match="evaluation scope mismatch"):
        weighted_delta(other_population, cheapest)
