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
        try:
            quality = float(self.quality)
        except OverflowError as error:
            raise ValueError("quality must be finite") from error
        if not math.isfinite(quality):
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
        try:
            weight = float(self.weight)
        except OverflowError as error:
            raise ValueError("weight must be finite and positive") from error
        if not math.isfinite(weight) or weight <= 0:
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
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be a non-empty string")
        if not isinstance(self.tier, BudgetTier):
            raise TypeError("tier must be a BudgetTier")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a boolean")
        if self.selected_model_id is not None and (
            not isinstance(self.selected_model_id, str) or not self.selected_model_id.strip()
        ):
            raise ValueError("selected_model_id must be a non-empty string or None")
        if self.output is not None and not isinstance(self.output, str):
            raise TypeError("output must be a string or None")
        for field_name in ("quality", "predicted_quality"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number or None")
            try:
                normalized = float(value)
            except OverflowError as error:
                raise ValueError(f"{field_name} must be finite when provided") from error
            if not math.isfinite(normalized):
                raise ValueError(f"{field_name} must be finite when provided")
            object.__setattr__(self, field_name, normalized)
        if not isinstance(self.decision_reason, str):
            raise TypeError("decision_reason must be a string")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
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

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_name, str) or not self.adapter_name.strip():
            raise ValueError("adapter_name must be a non-empty string")
        for field_name in ("configured_limit", "effective_total_limit", "spent"):
            ModelSpec(f"budget-{field_name}", getattr(self, field_name))
        if isinstance(self.over_budget_calls, bool) or not isinstance(self.over_budget_calls, int):
            raise TypeError("over_budget_calls must be an integer")
        if self.over_budget_calls < 0:
            raise ValueError("over_budget_calls must be non-negative")
        query_order = tuple(self.query_order)
        if not query_order:
            raise ValueError("budget query_order must not be empty")
        if any(
            not isinstance(example_id, str) or not example_id.strip() for example_id in query_order
        ):
            raise ValueError("budget query_order must contain non-empty strings")
        if len(query_order) != len(set(query_order)):
            raise ValueError("budget query_order must contain unique example IDs")
        object.__setattr__(self, "query_order", query_order)


@dataclass(frozen=True, slots=True)
class TierResult:
    """All query results and accounting for one tier."""

    tier_spec: TierSpec
    queries: tuple[QueryResult, ...]
    budget: BudgetReport

    def __post_init__(self) -> None:
        if not isinstance(self.tier_spec, TierSpec):
            raise TypeError("tier_spec must be a TierSpec")
        queries = tuple(self.queries)
        if not queries:
            raise ValueError("tier queries must not be empty")
        if any(not isinstance(query, QueryResult) for query in queries):
            raise TypeError("queries must contain QueryResult values")
        if not isinstance(self.budget, BudgetReport):
            raise TypeError("budget must be a BudgetReport")
        tier = self.tier_spec.tier
        if any(query.tier is not tier for query in queries):
            raise ValueError("query tiers must match tier_spec")
        query_order = tuple(query.example_id for query in queries)
        if query_order != self.budget.query_order:
            raise ValueError("tier queries must match budget query_order")
        if self.budget.configured_limit != self.tier_spec.budget_limit:
            raise ValueError("budget configured_limit must match tier_spec")
        object.__setattr__(self, "queries", queries)

    @property
    def feasible(self) -> bool:
        return bool(self.queries) and all(query.feasible for query in self.queries)

    @property
    def mean_quality(self) -> float | None:
        if not self.feasible or any(query.quality is None for query in self.queries):
            return None
        try:
            mean = sum(query.quality for query in self.queries if query.quality is not None) / len(
                self.queries
            )
        except OverflowError as error:
            raise ValueError("tier mean quality must remain finite") from error
        if not math.isfinite(mean):
            raise ValueError("tier mean quality must remain finite")
        return mean


@dataclass(frozen=True, slots=True)
class EvaluationScopeIdentity:
    """Versioned replay/protocol identity shared by comparable reports."""

    algorithm: str
    sha256: str
    max_calls_per_query: int

    def __post_init__(self) -> None:
        if (
            type(self.algorithm) is not str
            or not self.algorithm
            or not self.algorithm.isascii()
            or any(not (character.isalnum() or character in "-._") for character in self.algorithm)
        ):
            raise ValueError("evaluation scope algorithm must be a non-empty ASCII identifier")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("evaluation scope sha256 must be lowercase SHA-256 hex")
        if type(self.max_calls_per_query) is not int:
            raise TypeError("max_calls_per_query must be an integer")
        if self.max_calls_per_query < 1:
            raise ValueError("max_calls_per_query must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A router's results over all requested tiers."""

    router_name: str
    tiers: tuple[TierResult, ...]
    evaluation_scope: EvaluationScopeIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.router_name, str) or not self.router_name.strip():
            raise ValueError("router_name must be a non-empty string")
        tiers = tuple(self.tiers)
        if not tiers:
            raise ValueError("evaluation report must contain at least one tier")
        if any(not isinstance(result, TierResult) for result in tiers):
            raise TypeError("tiers must contain TierResult values")
        if type(self.evaluation_scope) is not EvaluationScopeIdentity:
            raise TypeError("evaluation_scope must be an EvaluationScopeIdentity")
        if any(
            len(query.calls) > self.max_calls_per_query
            for result in tiers
            for query in result.queries
        ):
            raise ValueError("query call evidence exceeds max_calls_per_query")
        object.__setattr__(self, "tiers", tiers)

    @property
    def evaluation_scope_algorithm(self) -> str:
        return self.evaluation_scope.algorithm

    @property
    def evaluation_scope_sha256(self) -> str:
        return self.evaluation_scope.sha256

    @property
    def max_calls_per_query(self) -> int:
        return self.evaluation_scope.max_calls_per_query

    def by_tier(self) -> dict[BudgetTier, TierResult]:
        indexed: dict[BudgetTier, TierResult] = {}
        for result in self.tiers:
            tier = result.tier_spec.tier
            if tier in indexed:
                raise ValueError(f"duplicate tier in report: {tier.value}")
            indexed[tier] = result
        return indexed
