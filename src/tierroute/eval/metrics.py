# SPDX-License-Identifier: Apache-2.0
"""Tier-weighted quality and oracle-gap recovery metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from tierroute.core import BudgetTier
from tierroute.eval.schemas import EvaluationReport


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Primary score components for one evaluation report."""

    tier_quality: Mapping[BudgetTier, float | None]
    weighted_quality: float | None


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
    router_tiers = router.by_tier()
    cheapest_tiers = cheapest.by_tier()
    oracle_tiers = oracle.by_tier()
    if set(router_tiers) != set(cheapest_tiers) or set(router_tiers) != set(oracle_tiers):
        raise ValueError("router, cheapest, and oracle reports must contain the same tiers")

    numerator = 0.0
    denominator = 0.0
    for tier, router_result in router_tiers.items():
        cheap_result = cheapest_tiers[tier]
        oracle_result = oracle_tiers[tier]
        weights = {
            router_result.tier_spec.weight,
            cheap_result.tier_spec.weight,
            oracle_result.tier_spec.weight,
        }
        if len(weights) != 1:
            raise ValueError(f"tier weight mismatch for {tier.value}")
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

    router_score = summarize_report(router).weighted_quality
    reference_score = summarize_report(reference).weighted_quality
    if router_score is None or reference_score is None:
        return None
    return router_score - reference_score
