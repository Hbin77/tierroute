# SPDX-License-Identifier: Apache-2.0
"""Leakage-aware helpers for privileged and fitted baseline plans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from tierroute.core import BudgetTier
from tierroute.eval.schemas import EvaluationExample, TierSpec


def observable_domain_tag(metadata: Mapping[str, object]) -> str | None:
    """Return the pre-call domain tag, keeping split-only labels out of policies.

    ``EvaluationExample.domain`` exists to construct validation folds and may be
    private.  A domain-table policy may therefore use only the tag an adapter placed
    in ``RouterState.metadata`` before routing.  Missing tags deliberately select the
    baseline's fallback instead of borrowing the split label.
    """

    value = metadata.get("domain")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("observable domain metadata must be a non-empty string")
    return value


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
    """Fit mean-quality choices from observable training tags only.

    The validation-only ``EvaluationExample.domain`` field is intentionally never
    read here.  Rows without a pre-call observable domain tag contribute no table
    entry and will use the cheapest fallback at replay time.
    """

    if not training_examples:
        raise ValueError("training_examples must not be empty")
    cheapest = min(
        training_examples[0].candidate_models,
        key=lambda model: (model.cost, model.model_id),
    ).model_id
    tagged_examples = tuple(
        (example, tag)
        for example in training_examples
        if (tag := observable_domain_tag(example.router_metadata)) is not None
    )
    table: dict[tuple[BudgetTier, str], str] = {}
    domains = sorted({tag for _, tag in tagged_examples})
    for tier_spec in tier_specs:
        for domain in domains:
            totals: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            costs: dict[str, Fraction] = defaultdict(Fraction)
            for example, tag in tagged_examples:
                if tag != domain:
                    continue
                quoted_costs = {model.model_id: model.cost for model in example.candidate_models}
                for outcome in example.outcomes:
                    if (
                        quoted_costs[outcome.model_id] <= tier_spec.budget_limit
                        and outcome.cost <= tier_spec.budget_limit
                    ):
                        totals[outcome.model_id] += outcome.quality
                        costs[outcome.model_id] += Fraction(outcome.cost)
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
