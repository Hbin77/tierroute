# SPDX-License-Identifier: Apache-2.0
"""Leakage-aware helpers for privileged and fitted baseline plans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from tierroute.core import BudgetTier
from tierroute.eval.schemas import EvaluationExample, TierSpec


def build_per_query_oracle_plan(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
) -> dict[tuple[BudgetTier, str], str]:
    """Build a per-query upper bound feasible under quotes and realized charges."""

    plan: dict[tuple[BudgetTier, str], str] = {}
    for tier_spec in tier_specs:
        for example in examples:
            quoted_costs = {model.model_id: model.cost for model in example.candidate_models}
            affordable = [
                outcome
                for outcome in example.outcomes
                if quoted_costs[outcome.model_id] <= tier_spec.budget_limit
                and outcome.cost <= tier_spec.budget_limit
            ]
            if not affordable:
                raise ValueError(
                    f"no affordable outcome for {tier_spec.tier.value}/{example.example_id}"
                )
            best = min(
                affordable,
                key=lambda outcome: (-outcome.quality, outcome.cost, outcome.model_id),
            )
            plan[(tier_spec.tier, example.example_id)] = best.model_id
    return plan


@dataclass(frozen=True, slots=True)
class DomainTablePlan:
    """Table fitted from training examples only, plus a safe fallback."""

    table: dict[tuple[BudgetTier, str], str]
    fallback_model_id: str


def fit_per_query_domain_table(
    training_examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
) -> DomainTablePlan:
    """Fit mean-quality model choices without reading a held-out domain."""

    if not training_examples:
        raise ValueError("training_examples must not be empty")
    cheapest = min(
        training_examples[0].candidate_models,
        key=lambda model: (model.cost, model.model_id),
    ).model_id
    table: dict[tuple[BudgetTier, str], str] = {}
    domains = sorted({example.domain for example in training_examples})
    for tier_spec in tier_specs:
        for domain in domains:
            totals: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            costs: dict[str, float] = defaultdict(float)
            for example in training_examples:
                if example.domain != domain:
                    continue
                quoted_costs = {model.model_id: model.cost for model in example.candidate_models}
                for outcome in example.outcomes:
                    if (
                        quoted_costs[outcome.model_id] <= tier_spec.budget_limit
                        and outcome.cost <= tier_spec.budget_limit
                    ):
                        totals[outcome.model_id] += outcome.quality
                        costs[outcome.model_id] += float(outcome.cost)
                        counts[outcome.model_id] += 1
            if not counts:
                continue
            best_model = min(
                counts,
                key=lambda model_id: (
                    -(totals[model_id] / counts[model_id]),
                    costs[model_id] / counts[model_id],
                    model_id,
                ),
            )
            table[(tier_spec.tier, domain)] = best_model
    return DomainTablePlan(table, cheapest)
