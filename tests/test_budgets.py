# SPDX-License-Identifier: Apache-2.0
"""Tests for swappable budget-scope adapters."""

from decimal import Decimal, localcontext

import pytest

from tierroute.adapters import CumulativeBudgetLedger, PerQueryBudgetLedger


@pytest.mark.parametrize("ledger_type", [PerQueryBudgetLedger, CumulativeBudgetLedger])
def test_decimal_boundary_is_charged_exactly(ledger_type: object) -> None:
    ledger = ledger_type(Decimal("0.3"), 1)  # type: ignore[operator]
    ledger.begin_query("q1")

    assert ledger.charge_realized(Decimal("0.1")) is True
    assert ledger.charge_realized(Decimal("0.2")) is True
    assert ledger.remaining_budget == Decimal("0.0")


def test_per_query_budget_resets_for_each_query() -> None:
    ledger = PerQueryBudgetLedger(Decimal("1"), 2)

    ledger.begin_query("q1")
    assert ledger.charge_realized(Decimal("1")) is True
    ledger.finish_query()
    ledger.begin_query("q2")
    assert ledger.remaining_budget == Decimal("1")
    ledger.finish_query()

    assert ledger.report().effective_total_limit == Decimal("2")


def test_cumulative_budget_carries_remaining_amount() -> None:
    ledger = CumulativeBudgetLedger(Decimal("1"), 2)

    ledger.begin_query("q1")
    assert ledger.charge_realized(Decimal("0.75")) is True
    ledger.finish_query()
    ledger.begin_query("q2")
    assert ledger.remaining_budget == Decimal("0.25")
    assert ledger.charge_realized(Decimal("0.5")) is False
    assert ledger.remaining_budget == Decimal(0)
    ledger.finish_query()

    report = ledger.report()
    assert report.spent == Decimal("1.25")
    assert report.over_budget_calls == 1
    assert report.query_order == ("q1", "q2")


def test_cumulative_ledger_does_not_round_away_a_tiny_overspend() -> None:
    costs = (Decimal("0.33333333333333333333333333333"),) * 3 + (Decimal("5e-29"),)

    with localcontext() as context:
        context.prec = 2
        ledger = CumulativeBudgetLedger(Decimal("1"), len(costs))
        accepted = []
        for index, cost in enumerate(costs):
            ledger.begin_query(f"q{index}")
            accepted.append(ledger.charge_realized(cost))
            ledger.finish_query()
        report = ledger.report()

    assert accepted == [True, True, True, False]
    assert report.spent == Decimal("1.00000000000000000000000000004")
    assert report.over_budget_calls == 1


def test_per_query_total_limit_and_spend_are_context_independent() -> None:
    with localcontext() as context:
        context.prec = 2
        ledger = PerQueryBudgetLedger(Decimal("1.234567890123456789"), 3)
        for index in range(3):
            ledger.begin_query(f"q{index}")
            assert ledger.charge_realized(Decimal("1.234567890123456789"))
            ledger.finish_query()
        report = ledger.report()

    assert report.spent == Decimal("3.703703670370370367")
    assert report.effective_total_limit == Decimal("3.703703670370370367")
