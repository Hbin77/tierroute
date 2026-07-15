# SPDX-License-Identifier: Apache-2.0
"""Budget-ledger protocol consumed by the specification-independent simulator."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tierroute.core import Cost
from tierroute.eval.schemas import BudgetReport


@runtime_checkable
class BudgetLedger(Protocol):
    """Stateful accounting boundary implemented by budget-scope adapters."""

    def begin_query(self, example_id: str) -> None:
        """Start accounting for one query in deterministic dataset order."""
        ...

    @property
    def remaining_budget(self) -> Cost:
        """Return the amount exposed to the router for the active query."""
        ...

    def try_charge(self, cost: Cost) -> bool:
        """Atomically charge a call or reject it without changing spend."""
        ...

    def finish_query(self) -> None:
        """Close the active query."""
        ...

    def report(self) -> BudgetReport:
        """Return immutable accounting details."""
        ...


class BudgetLedgerFactory(Protocol):
    """Callable adapter constructor used by ``OfflineSimulator``."""

    def __call__(self, budget_limit: Cost, expected_queries: int) -> BudgetLedger: ...
