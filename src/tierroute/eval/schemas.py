# SPDX-License-Identifier: Apache-2.0
"""Private replay schemas for full-information offline evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from tierroute.core import BudgetTier, Cost, ModelSpec, sum_costs


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """A logged response and realized charge hidden until its model is called."""

    model_id: str
    output: str
    cost: Cost
    quality: float

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.output, str):
            raise TypeError("output must be a string")
        ModelSpec(self.model_id, self.cost)
        if isinstance(self.quality, bool) or not isinstance(self.quality, (int, float)):
            raise TypeError("quality must be a real number")
        if not math.isfinite(self.quality):
            raise ValueError("quality must be finite")


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """One prompt and all candidate outcomes in a replay dataset."""

    example_id: str
    prompt: str
    domain: str
    outcomes: tuple[CandidateOutcome, ...]
    candidate_models: tuple[ModelSpec, ...]
    router_metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for field_name in ("example_id", "prompt", "domain"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "candidate_models", tuple(self.candidate_models))
        if not self.outcomes:
            raise ValueError("outcomes must not be empty")
        if not self.candidate_models:
            raise ValueError("candidate_models must not be empty")
        outcome_ids = [outcome.model_id for outcome in self.outcomes]
        candidate_ids = [model.model_id for model in self.candidate_models]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("outcomes must have unique model_id values")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_models must have unique model_id values")
        if set(outcome_ids) != set(candidate_ids):
            raise ValueError("outcomes and candidate_models must contain the same model IDs")


@dataclass(frozen=True, slots=True)
class TierSpec:
    """One evaluation tier; a ledger adapter interprets ``budget_limit`` scope."""

    tier: BudgetTier
    budget_limit: Cost
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.tier, BudgetTier):
            raise TypeError("tier must be a BudgetTier")
        ModelSpec("budget-validation", self.budget_limit)
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("weight must be a real number")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class ReplayCall:
    """Structured quote and realized-charge evidence for one executed replay call.

    This record belongs to the evaluation result, never ``RouterState``.  It omits
    ground-truth quality so recording every executed call does not create a new label
    channel. ``within_budget`` is the ledger's post-charge result; it does not mean the
    provider call was avoided, because the realized charge is already recorded.
    """

    model_id: str
    quoted_cost: Cost
    realized_cost: Cost
    remaining_budget_before: Cost
    remaining_budget_after: Cost
    within_budget: bool

    def __post_init__(self) -> None:
        ModelSpec(self.model_id, self.quoted_cost)
        ModelSpec("realized-cost", self.realized_cost)
        ModelSpec("remaining-budget-before", self.remaining_budget_before)
        ModelSpec("remaining-budget-after", self.remaining_budget_after)
        if not isinstance(self.within_budget, bool):
            raise TypeError("within_budget must be a boolean")
        if self.quoted_cost > self.remaining_budget_before:
            raise ValueError("an executed replay call must have an affordable quoted cost")


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Outcome of simulating one prompt at one tier."""

    example_id: str
    tier: BudgetTier
    feasible: bool
    selected_model_id: str | None
    cost: Cost
    quality: float | None
    output: str | None
    predicted_quality: float | None = None
    decision_reason: str = ""
    error: str | None = None
    calls: tuple[ReplayCall, ...] = ()
    selected_call_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "calls", tuple(self.calls))
        if any(not isinstance(call, ReplayCall) for call in self.calls):
            raise TypeError("calls must contain ReplayCall values")
        ModelSpec("query-cost", self.cost)
        if self.cost != sum_costs(call.realized_cost for call in self.calls):
            raise ValueError("query cost must equal the exact sum of replayed call charges")
        if self.feasible:
            if (
                self.selected_model_id is None
                or self.quality is None
                or self.output is None
                or self.selected_call_index is None
            ):
                raise ValueError("a feasible query must select one replayed call")
            if any(not call.within_budget for call in self.calls):
                raise ValueError("a feasible query cannot contain an over-budget call")
        elif (
            self.selected_model_id is not None
            or self.quality is not None
            or self.output is not None
            or self.selected_call_index is not None
        ):
            raise ValueError("an infeasible query cannot select an output")
        if self.selected_call_index is not None:
            if (
                isinstance(self.selected_call_index, bool)
                or not isinstance(self.selected_call_index, int)
                or self.selected_call_index < 0
            ):
                raise TypeError("selected_call_index must be a non-negative integer or None")
            if self.selected_call_index >= len(self.calls):
                raise ValueError("selected_call_index is unavailable in replayed calls")
            if self.calls[self.selected_call_index].model_id != self.selected_model_id:
                raise ValueError("selected call and selected_model_id must agree")


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """Ledger summary whose scope is declared by its adapter."""

    adapter_name: str
    configured_limit: Cost
    effective_total_limit: Cost
    spent: Cost
    over_budget_calls: int
    query_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TierResult:
    """All query results and accounting for one tier."""

    tier_spec: TierSpec
    queries: tuple[QueryResult, ...]
    budget: BudgetReport

    @property
    def feasible(self) -> bool:
        return bool(self.queries) and all(query.feasible for query in self.queries)

    @property
    def mean_quality(self) -> float | None:
        if not self.feasible or any(query.quality is None for query in self.queries):
            return None
        return sum(query.quality for query in self.queries if query.quality is not None) / len(
            self.queries
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A router's results over all requested tiers."""

    router_name: str
    tiers: tuple[TierResult, ...]

    def by_tier(self) -> dict[BudgetTier, TierResult]:
        indexed: dict[BudgetTier, TierResult] = {}
        for result in self.tiers:
            tier = result.tier_spec.tier
            if tier in indexed:
                raise ValueError(f"duplicate tier in report: {tier.value}")
            indexed[tier] = result
        return indexed
