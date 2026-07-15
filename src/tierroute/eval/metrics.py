# SPDX-License-Identifier: Apache-2.0
"""Offline quality, oracle-gap, and exact quote-error metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from tierroute.core import (
    BudgetTier,
    Cost,
    ModelSpec,
    add_cost,
    subtract_cost,
    sum_costs,
)
from tierroute.eval.schemas import EvaluationReport, ReplayCall, TierResult


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Primary score components for one evaluation report."""

    tier_quality: Mapping[BudgetTier, float | None]
    weighted_quality: float | None


class QuoteCostDirection(str, Enum):
    """Direction of an exact realized-minus-quoted aggregate difference."""

    REALIZED_ABOVE_QUOTE = "realized-above-quote"
    EQUAL = "equal"
    REALIZED_BELOW_QUOTE = "realized-below-quote"


@dataclass(frozen=True, slots=True)
class ExactCostDifference:
    """A signed cost difference represented without a signed ``Cost`` value."""

    direction: QuoteCostDirection
    magnitude: Cost

    def __post_init__(self) -> None:
        if not isinstance(self.direction, QuoteCostDirection):
            raise TypeError("direction must be a QuoteCostDirection")
        ModelSpec("cost-difference", self.magnitude)
        if (self.direction is QuoteCostDirection.EQUAL) != (self.magnitude == 0):
            raise ValueError("equal cost differences must have exactly zero magnitude")


@dataclass(frozen=True, slots=True)
class QuoteErrorSummary:
    """Exact quote-versus-realized diagnostics over executed replay calls."""

    call_count: int
    exact_quote_calls: int
    underquoted_calls: int
    overquoted_calls: int
    realized_over_budget_calls: int
    total_quoted_cost: Cost
    total_realized_cost: Cost
    total_absolute_quote_error: Cost
    total_underquoted_amount: Cost
    total_overquoted_amount: Cost
    net_quote_error: ExactCostDifference

    def __post_init__(self) -> None:
        counts = (
            self.call_count,
            self.exact_quote_calls,
            self.underquoted_calls,
            self.overquoted_calls,
            self.realized_over_budget_calls,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("quote-error counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("quote-error counts must be non-negative")
        if self.exact_quote_calls + self.underquoted_calls + self.overquoted_calls != (
            self.call_count
        ):
            raise ValueError("quote direction counts must cover every executed call")
        if self.realized_over_budget_calls > self.call_count:
            raise ValueError("over-budget calls cannot exceed executed calls")
        for name, value in (
            ("total-quoted", self.total_quoted_cost),
            ("total-realized", self.total_realized_cost),
            ("absolute-error", self.total_absolute_quote_error),
            ("underquoted-amount", self.total_underquoted_amount),
            ("overquoted-amount", self.total_overquoted_amount),
        ):
            ModelSpec(name, value)
        if not isinstance(self.net_quote_error, ExactCostDifference):
            raise TypeError("net_quote_error must be an ExactCostDifference")
        if self.call_count == 0 and any(
            value != 0
            for value in (
                self.total_quoted_cost,
                self.total_realized_cost,
                self.total_absolute_quote_error,
                self.total_underquoted_amount,
                self.total_overquoted_amount,
            )
        ):
            raise ValueError("a zero-call summary must have zero cost totals")
        if (self.underquoted_calls == 0) != (self.total_underquoted_amount == 0):
            raise ValueError("underquote count and amount must either both be zero or positive")
        if (self.overquoted_calls == 0) != (self.total_overquoted_amount == 0):
            raise ValueError("overquote count and amount must either both be zero or positive")
        if self.total_absolute_quote_error != add_cost(
            self.total_underquoted_amount,
            self.total_overquoted_amount,
        ):
            raise ValueError("absolute quote error must include both error directions")
        if add_cost(self.total_realized_cost, self.total_overquoted_amount) != add_cost(
            self.total_quoted_cost,
            self.total_underquoted_amount,
        ):
            raise ValueError("quote-error amounts do not conserve exact cost totals")
        if self.net_quote_error != _exact_cost_difference(
            self.total_quoted_cost,
            self.total_realized_cost,
        ):
            raise ValueError("net quote error must match exact aggregate totals")


@dataclass(frozen=True, slots=True)
class TierQuoteErrorSummary:
    """Quote diagnostics for one configured tier."""

    tier: BudgetTier
    summary: QuoteErrorSummary

    def __post_init__(self) -> None:
        if not isinstance(self.tier, BudgetTier):
            raise TypeError("tier must be a BudgetTier")
        if not isinstance(self.summary, QuoteErrorSummary):
            raise TypeError("summary must be a QuoteErrorSummary")


@dataclass(frozen=True, slots=True)
class QuoteErrorReport:
    """Per-tier and cross-tier diagnostics, not a budget-compliance total."""

    tiers: tuple[TierQuoteErrorSummary, ...]
    overall: QuoteErrorSummary

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiers", tuple(self.tiers))
        if not self.tiers:
            raise ValueError("quote-error report must contain at least one tier")
        if any(not isinstance(result, TierQuoteErrorSummary) for result in self.tiers):
            raise TypeError("tiers must contain TierQuoteErrorSummary values")
        if not isinstance(self.overall, QuoteErrorSummary):
            raise TypeError("overall must be a QuoteErrorSummary")
        tiers = [result.tier for result in self.tiers]
        if len(tiers) != len(set(tiers)):
            raise ValueError("quote-error report must contain unique tiers")
        expected = _combine_quote_summaries(result.summary for result in self.tiers)
        if self.overall != expected:
            raise ValueError("overall quote-error summary must equal the exact tier aggregate")

    def by_tier(self) -> dict[BudgetTier, QuoteErrorSummary]:
        indexed: dict[BudgetTier, QuoteErrorSummary] = {}
        for result in self.tiers:
            if result.tier in indexed:
                raise ValueError(f"duplicate tier in quote-error report: {result.tier.value}")
            indexed[result.tier] = result.summary
        return indexed


def _exact_cost_difference(quoted: Cost, realized: Cost) -> ExactCostDifference:
    if realized > quoted:
        return ExactCostDifference(
            QuoteCostDirection.REALIZED_ABOVE_QUOTE,
            subtract_cost(realized, quoted),
        )
    if realized < quoted:
        return ExactCostDifference(
            QuoteCostDirection.REALIZED_BELOW_QUOTE,
            subtract_cost(quoted, realized),
        )
    return ExactCostDifference(
        QuoteCostDirection.EQUAL,
        subtract_cost(quoted, realized),
    )


def _combine_quote_summaries(summaries: Iterable[QuoteErrorSummary]) -> QuoteErrorSummary:
    ordered = tuple(summaries)
    total_quoted = sum_costs(summary.total_quoted_cost for summary in ordered)
    total_realized = sum_costs(summary.total_realized_cost for summary in ordered)
    total_underquoted = sum_costs(summary.total_underquoted_amount for summary in ordered)
    total_overquoted = sum_costs(summary.total_overquoted_amount for summary in ordered)
    return QuoteErrorSummary(
        call_count=sum(summary.call_count for summary in ordered),
        exact_quote_calls=sum(summary.exact_quote_calls for summary in ordered),
        underquoted_calls=sum(summary.underquoted_calls for summary in ordered),
        overquoted_calls=sum(summary.overquoted_calls for summary in ordered),
        realized_over_budget_calls=sum(summary.realized_over_budget_calls for summary in ordered),
        total_quoted_cost=total_quoted,
        total_realized_cost=total_realized,
        total_absolute_quote_error=add_cost(total_underquoted, total_overquoted),
        total_underquoted_amount=total_underquoted,
        total_overquoted_amount=total_overquoted,
        net_quote_error=_exact_cost_difference(total_quoted, total_realized),
    )


def _summarize_replay_calls(calls: Iterable[ReplayCall]) -> QuoteErrorSummary:
    ordered = tuple(calls)
    total_quoted = sum_costs(call.quoted_cost for call in ordered)
    total_realized = sum_costs(call.realized_cost for call in ordered)
    underquoted_amounts = tuple(
        subtract_cost(call.realized_cost, call.quoted_cost)
        for call in ordered
        if call.realized_cost > call.quoted_cost
    )
    overquoted_amounts = tuple(
        subtract_cost(call.quoted_cost, call.realized_cost)
        for call in ordered
        if call.realized_cost < call.quoted_cost
    )
    total_underquoted = sum_costs(underquoted_amounts)
    total_overquoted = sum_costs(overquoted_amounts)
    return QuoteErrorSummary(
        call_count=len(ordered),
        exact_quote_calls=sum(call.realized_cost == call.quoted_cost for call in ordered),
        underquoted_calls=len(underquoted_amounts),
        overquoted_calls=len(overquoted_amounts),
        realized_over_budget_calls=sum(not call.within_budget for call in ordered),
        total_quoted_cost=total_quoted,
        total_realized_cost=total_realized,
        total_absolute_quote_error=add_cost(total_underquoted, total_overquoted),
        total_underquoted_amount=total_underquoted,
        total_overquoted_amount=total_overquoted,
        net_quote_error=_exact_cost_difference(total_quoted, total_realized),
    )


def _tier_comparison_signature(result: TierResult) -> tuple[object, ...]:
    """Return fields that must match before reports can share one metric."""

    tier = result.tier_spec.tier
    query_order = tuple(query.example_id for query in result.queries)
    if len(query_order) != len(set(query_order)):
        raise ValueError(f"duplicate query ID in {tier.value} tier")
    if any(query.tier is not tier for query in result.queries):
        raise ValueError(f"query tier mismatch in {tier.value} tier")
    if result.budget.query_order != query_order:
        raise ValueError(f"budget/query order mismatch in {tier.value} tier")
    if result.budget.configured_limit != result.tier_spec.budget_limit:
        raise ValueError(f"budget limit mismatch inside {tier.value} tier")
    return (
        result.tier_spec.budget_limit,
        result.tier_spec.weight,
        result.budget.adapter_name,
        result.budget.configured_limit,
        result.budget.effective_total_limit,
        query_order,
    )


def summarize_quote_error(report: EvaluationReport) -> QuoteErrorReport:
    """Aggregate exact quote error while rejecting inconsistent replay evidence.

    The overall row is a cross-tier diagnostic only. It must not be interpreted as a
    shared budget because each tier owns an independent ledger and configured limit.
    """

    if not report.tiers:
        raise ValueError("report must contain at least one tier")
    seen_tiers: set[BudgetTier] = set()
    tier_summaries: list[TierQuoteErrorSummary] = []
    all_calls: list[ReplayCall] = []
    for result in report.tiers:
        _tier_comparison_signature(result)
        tier = result.tier_spec.tier
        if tier in seen_tiers:
            raise ValueError(f"duplicate tier in report: {tier.value}")
        seen_tiers.add(tier)
        calls = tuple(call for query in result.queries for call in query.calls)
        summary = _summarize_replay_calls(calls)
        query_spend = sum_costs(query.cost for query in result.queries)
        if query_spend != result.budget.spent:
            raise ValueError(f"budget spend and query costs disagree for {tier.value}")
        if summary.total_realized_cost != result.budget.spent:
            raise ValueError(f"call evidence and budget spend disagree for {tier.value}")
        if summary.realized_over_budget_calls != result.budget.over_budget_calls:
            raise ValueError(f"over-budget call evidence disagrees for {tier.value}")
        tier_summaries.append(TierQuoteErrorSummary(tier, summary))
        all_calls.extend(calls)
    return QuoteErrorReport(tuple(tier_summaries), _summarize_replay_calls(all_calls))


def _aligned_tiers(
    router: EvaluationReport,
    cheapest: EvaluationReport,
    oracle: EvaluationReport,
) -> tuple[
    dict[BudgetTier, TierResult],
    dict[BudgetTier, TierResult],
    dict[BudgetTier, TierResult],
]:
    """Reject comparisons across different populations or accounting semantics."""

    router_tiers = router.by_tier()
    cheapest_tiers = cheapest.by_tier()
    oracle_tiers = oracle.by_tier()
    if set(router_tiers) != set(cheapest_tiers) or set(router_tiers) != set(oracle_tiers):
        raise ValueError("router, cheapest, and oracle reports must contain the same tiers")
    for tier, router_result in router_tiers.items():
        signature = _tier_comparison_signature(router_result)
        if _tier_comparison_signature(cheapest_tiers[tier]) != signature:
            raise ValueError(f"router and cheapest evaluation scope mismatch for {tier.value}")
        if _tier_comparison_signature(oracle_tiers[tier]) != signature:
            raise ValueError(f"router and oracle evaluation scope mismatch for {tier.value}")
    return router_tiers, cheapest_tiers, oracle_tiers


def summarize_report(report: EvaluationReport) -> ScoreSummary:
    """Compute tier means and the explicit weighted mean.

    An infeasible or incomplete tier makes the primary score unavailable. Its weight is
    never silently redistributed to successful tiers.
    """

    if not report.tiers:
        raise ValueError("report must contain at least one tier")
    tier_quality: dict[BudgetTier, float | None] = {}
    numerator = 0.0
    denominator = 0.0
    complete = True
    for result in report.tiers:
        _tier_comparison_signature(result)
        tier = result.tier_spec.tier
        if tier in tier_quality:
            raise ValueError(f"duplicate tier in report: {tier.value}")
        quality = result.mean_quality
        tier_quality[tier] = quality
        complete = complete and quality is not None
        if quality is not None:
            numerator += result.tier_spec.weight * quality
        denominator += result.tier_spec.weight
    return ScoreSummary(tier_quality, numerator / denominator if complete else None)


def oracle_gap_recovery(
    router: EvaluationReport,
    cheapest: EvaluationReport,
    oracle: EvaluationReport,
    *,
    tolerance: float = 1e-12,
) -> float | None:
    """Return weighted recovery of the cheapest-to-oracle quality gap.

    Values below zero are meaningful and are not clamped. A zero oracle gap returns
    ``None`` because the recovery ratio is undefined.
    """

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    router_tiers, cheapest_tiers, oracle_tiers = _aligned_tiers(router, cheapest, oracle)

    numerator = 0.0
    denominator = 0.0
    for tier, router_result in router_tiers.items():
        cheap_result = cheapest_tiers[tier]
        oracle_result = oracle_tiers[tier]
        router_quality = router_result.mean_quality
        cheap_quality = cheap_result.mean_quality
        oracle_quality = oracle_result.mean_quality
        if router_quality is None or cheap_quality is None or oracle_quality is None:
            return None
        if oracle_quality + tolerance < cheap_quality:
            raise ValueError(f"oracle quality is below cheapest for {tier.value}")
        if router_quality > oracle_quality + tolerance:
            raise ValueError(f"router quality exceeds oracle for {tier.value}")
        weight = router_result.tier_spec.weight
        numerator += weight * (router_quality - cheap_quality)
        denominator += weight * (oracle_quality - cheap_quality)

    if abs(denominator) <= tolerance:
        return None
    return numerator / denominator


def weighted_delta(router: EvaluationReport, reference: EvaluationReport) -> float | None:
    """Return weighted-quality delta, or ``None`` if either report is infeasible."""

    router_tiers = router.by_tier()
    reference_tiers = reference.by_tier()
    if set(router_tiers) != set(reference_tiers):
        raise ValueError("router and reference reports must contain the same tiers")
    for tier, router_result in router_tiers.items():
        if _tier_comparison_signature(router_result) != _tier_comparison_signature(
            reference_tiers[tier]
        ):
            raise ValueError(f"router and reference evaluation scope mismatch for {tier.value}")
    router_score = summarize_report(router).weighted_quality
    reference_score = summarize_report(reference).weighted_quality
    if router_score is None or reference_score is None:
        return None
    return router_score - reference_score
