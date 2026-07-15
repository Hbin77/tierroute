# SPDX-License-Identifier: Apache-2.0
"""Private replay schemas for full-information offline evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tierroute.core import BudgetTier, Cost, ModelSpec


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """A logged response hidden from the router until its model is called."""

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

    def __post_init__(self) -> None:
        for field_name in ("example_id", "prompt", "domain"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        if not self.outcomes:
            raise ValueError("outcomes must not be empty")
        model_ids = [outcome.model_id for outcome in self.outcomes]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("outcomes must have unique model_id values")

    @property
    def candidate_models(self) -> tuple[ModelSpec, ...]:
        """Expose only IDs and costs to the router."""

        return tuple(ModelSpec(outcome.model_id, outcome.cost) for outcome in self.outcomes)


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


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """Ledger summary whose scope is declared by its adapter."""

    adapter_name: str
    configured_limit: Cost
    effective_total_limit: Cost
    spent: Cost
    rejected_calls: int
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
        return {result.tier_spec.tier: result for result in self.tiers}
