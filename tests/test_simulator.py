# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for full-information offline replay."""

from decimal import Decimal

from tierroute.adapters import CumulativeBudgetLedger, PerQueryBudgetLedger
from tierroute.core import BudgetTier, CallModel
from tierroute.eval import (
    CandidateOutcome,
    EvaluationExample,
    OfflineSimulator,
    TierSpec,
    build_per_query_oracle_plan,
)
from tierroute.policies import AlwaysCheapestRouter, AlwaysPremiumRouter, OracleRouter

EXAMPLES = (
    EvaluationExample(
        "q1",
        "easy prompt",
        "general",
        (
            CandidateOutcome("cheap", "cheap one", Decimal("1"), 0.5),
            CandidateOutcome("premium", "premium one", Decimal("2"), 0.9),
        ),
    ),
    EvaluationExample(
        "q2",
        "hard prompt",
        "reasoning",
        (
            CandidateOutcome("cheap", "cheap two", Decimal("1"), 0.4),
            CandidateOutcome("premium", "premium two", Decimal("2"), 1.0),
        ),
    ),
)
TIER = TierSpec(BudgetTier.FAST, Decimal("2"), 1.0)


def test_simulator_replays_calls_then_selects_without_leaking_quality() -> None:
    simulator = OfflineSimulator(PerQueryBudgetLedger)

    result = simulator.run_tier(AlwaysCheapestRouter(), EXAMPLES, TIER)

    assert result.feasible is True
    assert result.mean_quality == 0.45
    assert result.budget.spent == Decimal("2")
    assert result.queries[0].selected_model_id == "cheap"
    assert "call cheap" in result.queries[0].decision_reason


def test_same_simulator_supports_cumulative_budget_via_adapter_only() -> None:
    simulator = OfflineSimulator(CumulativeBudgetLedger)

    result = simulator.run_tier(AlwaysPremiumRouter("premium"), EXAMPLES, TIER)

    assert result.feasible is False
    assert result.queries[0].quality == 0.9
    assert result.queries[1].quality is None
    assert result.budget.spent == Decimal("2")


def test_second_call_is_rejected_by_one_shot_limit_after_first_cost_is_charged() -> None:
    class CallsForever:
        def route(self, state: object) -> CallModel:
            return CallModel("cheap")

    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(CallsForever(), EXAMPLES[:1], TIER)

    assert result.feasible is False
    assert result.queries[0].cost == Decimal("1")
    assert "max_calls_per_query=1" in (result.queries[0].error or "")


def test_per_query_oracle_plan_is_budget_feasible_and_privileged() -> None:
    plan = build_per_query_oracle_plan(EXAMPLES, (TIER,))
    report = OfflineSimulator(PerQueryBudgetLedger).run_tier(OracleRouter(plan), EXAMPLES, TIER)

    assert report.mean_quality == 0.95
    assert {query.selected_model_id for query in report.queries} == {"premium"}
