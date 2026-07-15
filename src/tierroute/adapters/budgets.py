# SPDX-License-Identifier: Apache-2.0
"""Budget-scope adapters used until the official SKT semantics are confirmed."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from tierroute.core import Cost
from tierroute.eval.schemas import BudgetReport


def _validate_inputs(budget_limit: Cost, expected_queries: int) -> None:
    if not isinstance(budget_limit, Decimal) or not budget_limit.is_finite() or budget_limit < 0:
        raise ValueError("budget_limit must be a finite non-negative Decimal")
    if isinstance(expected_queries, bool) or not isinstance(expected_queries, int):
        raise TypeError("expected_queries must be an integer")
    if expected_queries < 1:
        raise ValueError("expected_queries must be positive")


def _validate_charge(cost: Cost) -> None:
    if not isinstance(cost, Decimal) or not cost.is_finite() or cost < 0:
        raise ValueError("cost must be a finite non-negative Decimal")


@dataclass(slots=True)
class PerQueryBudgetLedger:
    """Reset the configured budget at the start of every query."""

    budget_limit: Cost
    expected_queries: int
    _active_query: str | None = field(default=None, init=False, repr=False)
    _remaining: Cost = field(default=Decimal(0), init=False, repr=False)
    _spent: Cost = field(default=Decimal(0), init=False, repr=False)
    _over_budget_calls: int = field(default=0, init=False, repr=False)
    _query_order: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_inputs(self.budget_limit, self.expected_queries)

    def begin_query(self, example_id: str) -> None:
        if self._active_query is not None:
            raise RuntimeError("finish the active query before beginning another")
        self._active_query = example_id
        self._remaining = self.budget_limit
        self._query_order.append(example_id)

    @property
    def remaining_budget(self) -> Cost:
        if self._active_query is None:
            raise RuntimeError("begin_query must be called before reading budget")
        return self._remaining

    def charge_realized(self, cost: Cost) -> bool:
        """Record the post-call charge, including any unavoidable overspend."""

        _validate_charge(cost)
        if self._active_query is None:
            raise RuntimeError("begin_query must be called before charging")
        self._spent += cost
        if cost > self._remaining:
            self._over_budget_calls += 1
            self._remaining = Decimal(0)
            return False
        self._remaining -= cost
        return True

    def finish_query(self) -> None:
        if self._active_query is None:
            raise RuntimeError("no active query to finish")
        self._active_query = None

    def report(self) -> BudgetReport:
        return BudgetReport(
            adapter_name="per-query",
            configured_limit=self.budget_limit,
            effective_total_limit=self.budget_limit * self.expected_queries,
            spent=self._spent,
            over_budget_calls=self._over_budget_calls,
            query_order=tuple(self._query_order),
        )


@dataclass(slots=True)
class CumulativeBudgetLedger:
    """Share one configured budget across the ordered query stream."""

    budget_limit: Cost
    expected_queries: int
    _active_query: str | None = field(default=None, init=False, repr=False)
    _remaining: Cost = field(init=False, repr=False)
    _spent: Cost = field(default=Decimal(0), init=False, repr=False)
    _over_budget_calls: int = field(default=0, init=False, repr=False)
    _query_order: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_inputs(self.budget_limit, self.expected_queries)
        self._remaining = self.budget_limit

    def begin_query(self, example_id: str) -> None:
        if self._active_query is not None:
            raise RuntimeError("finish the active query before beginning another")
        self._active_query = example_id
        self._query_order.append(example_id)

    @property
    def remaining_budget(self) -> Cost:
        if self._active_query is None:
            raise RuntimeError("begin_query must be called before reading budget")
        return self._remaining

    def charge_realized(self, cost: Cost) -> bool:
        """Record the post-call charge and exhaust a cumulative budget on overspend."""

        _validate_charge(cost)
        if self._active_query is None:
            raise RuntimeError("begin_query must be called before charging")
        self._spent += cost
        if cost > self._remaining:
            self._over_budget_calls += 1
            self._remaining = Decimal(0)
            return False
        self._remaining -= cost
        return True

    def finish_query(self) -> None:
        if self._active_query is None:
            raise RuntimeError("no active query to finish")
        self._active_query = None

    def report(self) -> BudgetReport:
        return BudgetReport(
            adapter_name="cumulative",
            configured_limit=self.budget_limit,
            effective_total_limit=self.budget_limit,
            spent=self._spent,
            over_budget_calls=self._over_budget_calls,
            query_order=tuple(self._query_order),
        )
