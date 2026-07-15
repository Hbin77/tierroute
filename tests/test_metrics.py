# SPDX-License-Identifier: Apache-2.0
"""Tests for weighted tier quality and oracle-gap recovery."""

from decimal import Decimal

import pytest

from tierroute.core import BudgetTier
from tierroute.eval import (
    BudgetReport,
    EvaluationReport,
    QueryResult,
    TierResult,
    TierSpec,
    oracle_gap_recovery,
    summarize_report,
)

WEIGHTS = {
    BudgetTier.FAST: 0.5,
    BudgetTier.BALANCED: 0.3,
    BudgetTier.PREMIUM: 0.2,
}


def report(name: str, qualities: dict[BudgetTier, float | None]) -> EvaluationReport:
    tiers = []
    for tier, quality in qualities.items():
        query = QueryResult(
            "q1",
            tier,
            quality is not None,
            "model" if quality is not None else None,
            Decimal("1") if quality is not None else Decimal(0),
            quality,
            "output" if quality is not None else None,
            error=None if quality is not None else "infeasible",
        )
        budget = BudgetReport("test", Decimal("1"), Decimal("1"), query.cost, 0, ("q1",))
        tiers.append(TierResult(TierSpec(tier, Decimal("1"), WEIGHTS[tier]), (query,), budget))
    return EvaluationReport(name, tuple(tiers))


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
