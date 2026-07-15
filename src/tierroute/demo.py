# SPDX-License-Identifier: Apache-2.0
"""Self-contained routing decisions and six-baseline smoke evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tierroute.adapters import EvaluationDataset, PerQueryBudgetLedger
from tierroute.core import BudgetTier, CallModel, Cost, ModelSpec, RouterState, validate_action
from tierroute.eval import (
    EvaluationReport,
    OfflineSimulator,
    ScoreSummary,
    build_per_query_oracle_plan,
    fit_per_query_domain_table,
    oracle_gap_recovery,
    summarize_report,
)
from tierroute.features import SurfaceFeatures, extract_surface_features
from tierroute.policies import (
    AlwaysCheapestRouter,
    AlwaysPremiumRouter,
    DomainBestRouter,
    LambdaThresholdRouter,
    LengthHeuristicRouter,
    OracleRouter,
    RandomRouter,
)

_TIER_LAMBDAS = {
    BudgetTier.FAST: 0.80,
    BudgetTier.BALANCED: 0.35,
    BudgetTier.PREMIUM: 0.08,
}


@dataclass(frozen=True, slots=True)
class DemoQualityPredictor:
    """Explainable placeholder used only by the no-download CLI demonstration."""

    model_rank: dict[str, float]

    @classmethod
    def from_catalogue(cls, models: tuple[ModelSpec, ...]) -> DemoQualityPredictor:
        ordered = sorted(models, key=lambda model: (model.cost, model.model_id))
        denominator = max(len(ordered) - 1, 1)
        return cls({model.model_id: index / denominator for index, model in enumerate(ordered)})

    def predict(self, prompt: str, model_id: str) -> float:
        features = extract_surface_features(prompt)
        difficulty = min(
            1.0,
            features.character_count / 500
            + 0.22 * features.has_code
            + 0.22 * features.has_math
            + 0.02 * max(features.line_count - 1, 0),
        )
        try:
            rank = self.model_rank[model_id]
        except KeyError as error:
            raise KeyError(f"demo predictor has no model {model_id!r}") from error
        easy_quality = 0.82 + 0.12 * rank
        difficulty_penalty = 0.42 - 0.38 * rank
        return max(0.0, min(1.0, easy_quality - difficulty_penalty * difficulty))


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Serializable result of the offline CLI decision path."""

    tier: BudgetTier
    budget_limit: Cost
    model_id: str
    model_cost: Cost
    predicted_quality: float
    lambda_cost: float
    reason: str
    features: SurfaceFeatures


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One row in the bundled six-baseline scorecard."""

    name: str
    report: EvaluationReport
    score: ScoreSummary
    gap_recovery: float | None
    total_cost: Cost


def model_catalogue(dataset: EvaluationDataset) -> tuple[ModelSpec, ...]:
    """Return a catalogue after verifying constant model IDs and replay costs."""

    catalogue = dataset.examples[0].candidate_models
    signature = tuple((model.model_id, model.cost) for model in catalogue)
    for example in dataset.examples[1:]:
        current = tuple((model.model_id, model.cost) for model in example.candidate_models)
        if current != signature:
            raise ValueError("demo requires a stable model catalogue and cost per model")
    return catalogue


def route_prompt(dataset: EvaluationDataset, prompt: str, tier: BudgetTier) -> RouteDecision:
    """Make one offline lambda-policy decision using the bundled demo predictor."""

    try:
        tier_spec = next(spec for spec in dataset.tier_specs if spec.tier is tier)
    except StopIteration as error:
        raise ValueError(f"dataset does not configure tier {tier.value!r}") from error
    models = model_catalogue(dataset)
    predictor = DemoQualityPredictor.from_catalogue(models)
    lambda_cost = _TIER_LAMBDAS[tier]
    router = LambdaThresholdRouter(predictor, lambda_cost)
    state = RouterState(
        prompt=prompt,
        budget_tier=tier,
        remaining_budget=tier_spec.budget_limit,
        candidate_models=models,
        metadata={"example_id": "cli-prompt"},
    )
    action = router.route(state)
    validate_action(state, action)
    if not isinstance(action, CallModel):
        raise AssertionError("a fresh one-shot route must call a model")
    model = next(model for model in models if model.model_id == action.model_id)
    predicted_quality = action.predicted_quality
    if predicted_quality is None or not math.isfinite(predicted_quality):
        raise AssertionError("demo predictor must emit a finite quality estimate")
    return RouteDecision(
        tier=tier,
        budget_limit=tier_spec.budget_limit,
        model_id=model.model_id,
        model_cost=model.cost,
        predicted_quality=predicted_quality,
        lambda_cost=lambda_cost,
        reason=action.reason,
        features=extract_surface_features(prompt),
    )


def evaluate_six_baselines(dataset: EvaluationDataset) -> tuple[BaselineResult, ...]:
    """Run all required baselines on bundled or schema-compatible data."""

    models = model_catalogue(dataset)
    cheap = min(models, key=lambda model: (model.cost, model.model_id))
    premium = max(models, key=lambda model: (model.cost, model.model_id))
    oracle_plan = build_per_query_oracle_plan(dataset.examples, dataset.tier_specs)
    domain_plan = fit_per_query_domain_table(dataset.examples, dataset.tier_specs)
    routers = (
        ("always-cheapest", AlwaysCheapestRouter()),
        ("always-premium", AlwaysPremiumRouter(premium.model_id)),
        ("random", RandomRouter(seed=2026)),
        (
            "length-heuristic",
            LengthHeuristicRouter(cheap.model_id, premium.model_id, character_threshold=120),
        ),
        ("oracle", OracleRouter(oracle_plan)),
        ("domain-best-table", DomainBestRouter(domain_plan.table, domain_plan.fallback_model_id)),
    )
    simulator = OfflineSimulator(PerQueryBudgetLedger)
    reports = {
        name: simulator.run(router, dataset.examples, dataset.tier_specs, router_name=name)
        for name, router in routers
    }
    cheapest_report = reports["always-cheapest"]
    oracle_report = reports["oracle"]
    rows = []
    for name, _ in routers:
        report = reports[name]
        rows.append(
            BaselineResult(
                name=name,
                report=report,
                score=summarize_report(report),
                gap_recovery=oracle_gap_recovery(report, cheapest_report, oracle_report),
                total_cost=sum(
                    (query.cost for tier in report.tiers for query in tier.queries),
                    start=dataset.tier_specs[0].budget_limit * 0,
                ),
            )
        )
    return tuple(rows)
