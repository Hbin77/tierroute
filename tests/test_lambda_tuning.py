# SPDX-License-Identifier: Apache-2.0
"""Tests for exact, metric-direct tier lambda tuning."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product

import pytest

from tierroute.adapters import CumulativeBudgetLedger, PerQueryBudgetLedger
from tierroute.core import BudgetTier, CallModel, ModelSpec, RouterState
from tierroute.eval import BudgetReport, CandidateOutcome, DomainFold, EvaluationExample, TierSpec
from tierroute.policies import lambda_tuning
from tierroute.policies.lambda_threshold import route_from_predictions
from tierroute.policies.lambda_tuning import (
    CrossFittedPredictionTable,
    cross_fitted_prediction_table,
    derive_lambda_candidate_set,
    exact_lambda_candidates,
    fit_tiered_lambda_router_for_fold,
    nested_lodo_lambda_evaluation,
    tune_tier_lambdas,
)
from tierroute.predictors import StaticQualityPredictor


def _example(
    example_id: str,
    domain: str,
    models: tuple[ModelSpec, ...],
    qualities: dict[str, float],
    *,
    prompt: str | None = None,
    realized_costs: dict[str, Decimal] | None = None,
) -> EvaluationExample:
    realized_costs = realized_costs or {model.model_id: model.cost for model in models}
    return EvaluationExample(
        example_id=example_id,
        prompt=prompt or f"prompt {example_id}",
        domain=domain,
        outcomes=tuple(
            CandidateOutcome(
                model.model_id,
                f"output {model.model_id}",
                realized_costs[model.model_id],
                qualities[model.model_id],
            )
            for model in models
        ),
        candidate_models=models,
    )


def _table(
    examples: tuple[EvaluationExample, ...],
    predictions: dict[str, float],
) -> CrossFittedPredictionTable:
    return CrossFittedPredictionTable(
        {
            (example.example_id, model.model_id): predictions[model.model_id]
            for example in examples
            for model in example.candidate_models
        }
    )


def test_cumulative_tuning_rejects_exact_overspend_in_any_decimal_context() -> None:
    model = ModelSpec("only", Decimal("0"))
    realized_costs = (Decimal("0.33333333333333333333333333333"),) * 3 + (Decimal("5e-29"),)
    examples = tuple(
        _example(
            f"exact-{index}",
            "general",
            (model,),
            {"only": 0.5},
            realized_costs={"only": cost},
        )
        for index, cost in enumerate(realized_costs)
    )
    predictions = _table(examples, {"only": 0.5})

    with localcontext() as context:
        context.prec = 2
        with pytest.raises(ValueError, match="no fully feasible lambda"):
            tune_tier_lambdas(
                examples,
                (TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),),
                predictions,
                CumulativeBudgetLedger,
                lambda_grids={BudgetTier.FAST: (0,)},
            )


def test_grid_tunes_distinct_lambdas_and_direct_weighted_metric() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("middle", Decimal("2")),
        ModelSpec("premium", Decimal("4")),
    )
    examples = (
        _example(
            "q1",
            "general",
            models,
            {"cheap": 0.7, "middle": 0.0, "premium": 1.0},
        ),
    )
    specs = (
        TierSpec(BudgetTier.FAST, Decimal("2"), 0.6),
        TierSpec(BudgetTier.BALANCED, Decimal("2"), 0.3),
        TierSpec(BudgetTier.PREMIUM, Decimal("4"), 0.1),
    )
    predictions = _table(examples, {"cheap": 0.5, "middle": 0.7, "premium": 1.0})
    grids = {spec.tier: (0, Fraction(3, 10)) for spec in specs}

    result = tune_tier_lambdas(
        examples,
        specs,
        predictions,
        PerQueryBudgetLedger,
        lambda_grids=grids,
    )

    assert result.lambda_by_tier == {
        BudgetTier.FAST: Fraction(3, 10),
        BudgetTier.BALANCED: Fraction(3, 10),
        BudgetTier.PREMIUM: Fraction(0),
    }
    assert result.score.weighted_quality == pytest.approx(0.73)
    assert [selection.mean_quality for selection in result.selections] == [0.7, 0.7, 1.0]

    joint_scores = []
    for values in product((Fraction(0), Fraction(3, 10)), repeat=len(specs)):
        trial = tune_tier_lambdas(
            examples,
            specs,
            predictions,
            PerQueryBudgetLedger,
            lambda_grids={spec.tier: (value,) for spec, value in zip(specs, values, strict=True)},
        )
        joint_scores.append((trial.score.weighted_quality, values))
    best_score = max(score for score, _ in joint_scores if score is not None)
    best_schedules = [values for score, values in joint_scores if score == best_score]

    assert best_schedules == [(Fraction(3, 10), Fraction(3, 10), Fraction(0))]
    assert tuple(result.lambda_by_tier[spec.tier] for spec in specs) == best_schedules[0]


def test_tuning_ties_use_lower_spend_then_smaller_lambda() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    examples = (_example("q1", "general", models, {"cheap": 0.5, "premium": 0.5}),)
    spec = TierSpec(BudgetTier.FAST, Decimal("2"), 1.0)
    predictions = _table(examples, {"cheap": 0.0, "premium": 1.0})

    first = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        PerQueryBudgetLedger,
        lambda_grids={BudgetTier.FAST: (2, 0, 1, 2)},
    )
    second = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        PerQueryBudgetLedger,
        lambda_grids={BudgetTier.FAST: (1, 2, 0)},
    )

    assert first.lambda_by_tier[BudgetTier.FAST] == Fraction(1)
    assert first.lambda_by_tier == second.lambda_by_tier
    assert first.selections[0].realized_cost == Decimal("1")


def test_infeasible_realized_charge_never_wins() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("risky", Decimal("2")),
    )
    examples = (
        _example(
            "q1",
            "general",
            models,
            {"cheap": 0.5, "risky": 1.0},
            realized_costs={"cheap": Decimal("1"), "risky": Decimal("3")},
        ),
    )
    spec = TierSpec(BudgetTier.FAST, Decimal("2"), 1.0)
    predictions = _table(examples, {"cheap": 0.0, "risky": 1.0})

    result = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        PerQueryBudgetLedger,
        lambda_grids={BudgetTier.FAST: (0, 1)},
    )

    assert result.lambda_by_tier[BudgetTier.FAST] == Fraction(1)
    assert result.report.tiers[0].queries[0].selected_model_id == "cheap"
    with pytest.raises(ValueError, match="no fully feasible lambda"):
        tune_tier_lambdas(
            examples,
            (spec,),
            predictions,
            PerQueryBudgetLedger,
            lambda_grids={BudgetTier.FAST: (0,)},
        )


def test_exact_breakpoints_find_regime_missed_by_coarse_grid() -> None:
    models = (
        ModelSpec("cheap", Decimal("0")),
        ModelSpec("premium", Decimal("4")),
    )
    examples = (
        _example("q1", "a", models, {"cheap": 0.0, "premium": 1.0}),
        _example("q2", "b", models, {"cheap": 1.0, "premium": 0.0}),
    )
    predictions = CrossFittedPredictionTable(
        {
            ("q1", "cheap"): 0.5,
            ("q1", "premium"): 0.75,
            ("q2", "cheap"): 0.5,
            ("q2", "premium"): 0.625,
        }
    )
    spec = TierSpec(BudgetTier.FAST, Decimal("4"), 1.0)

    candidates = exact_lambda_candidates(examples, spec, predictions)
    exact = tune_tier_lambdas(examples, (spec,), predictions, PerQueryBudgetLedger)
    coarse = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        PerQueryBudgetLedger,
        lambda_grids={BudgetTier.FAST: (0, Fraction(1, 8))},
    )

    assert Fraction(1, 32) in candidates
    assert Fraction(1, 16) in candidates
    assert exact.lambda_by_tier[BudgetTier.FAST] == Fraction(1, 32)
    assert exact.score.weighted_quality == 1.0
    assert coarse.score.weighted_quality == 0.5


def test_bounded_candidates_are_labeled_deterministic_and_order_independent() -> None:
    models = (
        ModelSpec("cheap", Decimal("0")),
        ModelSpec("middle", Decimal("2")),
        ModelSpec("premium", Decimal("5")),
    )
    examples = (
        _example("q1", "a", models, {model.model_id: 0.5 for model in models}),
        _example("q2", "b", models, {model.model_id: 0.5 for model in models}),
    )
    predictions = CrossFittedPredictionTable(
        {
            ("q1", "cheap"): 0.0,
            ("q1", "middle"): 0.5,
            ("q1", "premium"): 1.0,
            ("q2", "cheap"): 0.0,
            ("q2", "middle"): 1.0,
            ("q2", "premium"): 1.25,
        }
    )
    spec = TierSpec(BudgetTier.FAST, Decimal("5"), 1.0)

    exhaustive = derive_lambda_candidate_set(examples, spec, predictions)
    capped = derive_lambda_candidate_set(examples, spec, predictions, max_candidates=3)
    nontruncated_cap = derive_lambda_candidate_set(
        examples,
        spec,
        predictions,
        max_candidates=len(exhaustive.values),
    )
    reversed_capped = derive_lambda_candidate_set(
        tuple(reversed(examples)),
        spec,
        predictions,
        max_candidates=3,
    )

    assert exhaustive.exhaustive
    assert exhaustive.strategy == "exhaustive-breakpoints-v1"
    assert exhaustive.total_derived_values == len(exhaustive.values)
    assert capped.exhaustive is False
    assert capped.total_derived_values is None
    assert capped.strategy == "bounded-bottom-hash-v1"
    assert capped.observed_breakpoint_count == 6
    assert len(capped.values) <= 3
    assert capped.values[0] == exhaustive.values[0]
    assert capped.values[-1] == exhaustive.values[-1]
    assert reversed_capped == capped
    assert nontruncated_cap == exhaustive
    assert nontruncated_cap.exhaustive is True
    assert nontruncated_cap.total_derived_values == len(nontruncated_cap.values)
    with pytest.raises(ValueError, match="mutually exclusive"):
        tune_tier_lambdas(
            examples,
            (spec,),
            predictions,
            PerQueryBudgetLedger,
            lambda_grids={BudgetTier.FAST: (0, 1)},
            max_candidates_per_tier=3,
        )


def test_derived_candidates_are_computed_once_and_reused_across_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    examples = (_example("q1", "general", models, {"cheap": 0.5, "premium": 1.0}),)
    predictions = _table(examples, {"cheap": 0.5, "premium": 1.0})
    specs = tuple(
        TierSpec(tier, Decimal("2"), weight)
        for tier, weight in (
            (BudgetTier.FAST, 0.6),
            (BudgetTier.BALANCED, 0.3),
            (BudgetTier.PREMIUM, 0.1),
        )
    )
    original = lambda_tuning.derive_lambda_candidate_set
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(lambda_tuning, "derive_lambda_candidate_set", counted)

    result = tune_tier_lambdas(
        examples,
        specs,
        predictions,
        PerQueryBudgetLedger,
        max_candidates_per_tier=3,
    )

    assert calls == 1
    assert len({selection.candidates.values for selection in result.selections}) == 1
    assert [selection.candidates.tier for selection in result.selections] == [
        spec.tier for spec in specs
    ]


def test_exhaustive_preflight_fails_before_candidate_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = (
        ModelSpec("cheap", Decimal("0")),
        ModelSpec("premium", Decimal("1")),
    )
    examples = (_example("q1", "general", models, {"cheap": 0.5, "premium": 1.0}),)
    predictions = _table(examples, {"cheap": 0.0, "premium": 1.0})
    spec = TierSpec(BudgetTier.FAST, Decimal("1"), 1.0)
    monkeypatch.setattr(lambda_tuning, "MAX_UNCONFIRMED_EXHAUSTIVE_CANDIDATES", 3)
    monkeypatch.setattr(
        lambda_tuning,
        "MAX_UNCONFIRMED_EXHAUSTIVE_UTILITY_EVALUATIONS",
        7,
    )

    def forbidden_candidate_map(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("candidate materialization must not start")

    monkeypatch.setattr(lambda_tuning, "_candidate_map", forbidden_candidate_map)

    with pytest.raises(ValueError, match="refused before candidate materialization") as caught:
        tune_tier_lambdas(
            examples,
            (spec,),
            predictions,
            PerQueryBudgetLedger,
        )

    message = str(caught.value)
    assert "candidate upper bound=4 (limit=3)" in message
    assert "utility-evaluation upper bound=8 (limit=7)" in message
    assert "max_candidates_per_tier" in message
    assert "allow_large_exhaustive=True" in message


def test_capped_and_acknowledged_exhaustive_searches_bypass_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = (
        ModelSpec("cheap", Decimal("0")),
        ModelSpec("premium", Decimal("1")),
    )
    examples = (_example("q1", "general", models, {"cheap": 0.5, "premium": 1.0}),)
    predictions = _table(examples, {"cheap": 0.0, "premium": 1.0})
    spec = TierSpec(BudgetTier.FAST, Decimal("1"), 1.0)
    monkeypatch.setattr(lambda_tuning, "MAX_UNCONFIRMED_EXHAUSTIVE_CANDIDATES", 1)
    monkeypatch.setattr(
        lambda_tuning,
        "MAX_UNCONFIRMED_EXHAUSTIVE_UTILITY_EVALUATIONS",
        1,
    )

    capped = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        PerQueryBudgetLedger,
        max_candidates_per_tier=2,
    )
    acknowledged = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        PerQueryBudgetLedger,
        allow_large_exhaustive=True,
    )

    assert len(capped.selections[0].candidates.values) == 2
    assert capped.selections[0].candidates.exhaustive is False
    assert acknowledged.selections[0].candidates.exhaustive is True
    with pytest.raises(TypeError, match="allow_large_exhaustive must be a boolean"):
        tune_tier_lambdas(
            examples,
            (spec,),
            predictions,
            PerQueryBudgetLedger,
            max_candidates_per_tier=2,
            allow_large_exhaustive=1,  # type: ignore[arg-type]
        )


def test_exhaustive_candidates_match_dense_decision_signature_oracle() -> None:
    models = (
        ModelSpec("cheap", Decimal("0")),
        ModelSpec("middle", Decimal("2")),
        ModelSpec("premium", Decimal("5")),
    )
    examples = (
        _example("q1", "a", models, {model.model_id: 0.5 for model in models}),
        _example("q2", "b", models, {model.model_id: 0.5 for model in models}),
    )
    predictions = CrossFittedPredictionTable(
        {
            ("q1", "cheap"): 0.0,
            ("q1", "middle"): 0.5,
            ("q1", "premium"): 1.0,
            ("q2", "cheap"): 0.25,
            ("q2", "middle"): 0.75,
            ("q2", "premium"): 1.0,
        }
    )
    spec = TierSpec(BudgetTier.FAST, Decimal("5"), 1.0)

    def signature(lambda_cost: Fraction) -> tuple[str, ...]:
        selected = []
        for example in examples:
            state = RouterState(
                prompt=example.prompt,
                budget_tier=BudgetTier.FAST,
                remaining_budget=Decimal("5"),
                candidate_models=models,
            )
            scores = predictions.for_example(
                example.example_id,
                tuple(model.model_id for model in models),
            )
            action = route_from_predictions(state, scores, lambda_cost)
            assert isinstance(action, CallModel)
            selected.append(action.model_id)
        return tuple(selected)

    exact = exact_lambda_candidates(examples, spec, predictions)
    dense = {Fraction(step, 120) for step in range(241)}
    dense.update((Fraction(1, 12), Fraction(3, 20), Fraction(1, 6), Fraction(1, 5)))
    dense.add(Fraction(5))

    assert {signature(value) for value in exact} == {signature(value) for value in dense}
    assert exact[0] == 0
    assert exact[-1] > max(value for value in exact[:-1])


def test_cumulative_tuning_preserves_recorded_query_order() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    examples = (
        _example("q_z", "z", models, {"cheap": 0.0, "premium": 1.0}),
        _example("q_a", "a", models, {"cheap": 0.5, "premium": 0.1}),
    )
    predictions = _table(examples, {"cheap": 0.4, "premium": 1.0})
    spec = TierSpec(BudgetTier.FAST, Decimal("3"), 1.0)

    result = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        CumulativeBudgetLedger,
        lambda_grids={BudgetTier.FAST: (0, 1)},
    )

    tier = result.report.tiers[0]
    assert result.lambda_by_tier[BudgetTier.FAST] == 0
    assert tier.budget.query_order == ("q_z", "q_a")
    assert [query.selected_model_id for query in tier.queries] == ["premium", "cheap"]


def test_infeasible_cumulative_candidate_does_not_abort_later_candidates() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    examples = tuple(
        _example(
            f"q{index}",
            domain,
            models,
            {"cheap": 0.6, "premium": 1.0},
        )
        for index, domain in enumerate(("a", "b"), start=1)
    )
    predictions = _table(examples, {"cheap": 0.0, "premium": 1.0})
    spec = TierSpec(BudgetTier.FAST, Decimal("2"), 1.0)

    result = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        CumulativeBudgetLedger,
        lambda_grids={BudgetTier.FAST: (0, 2)},
    )

    assert result.lambda_by_tier[BudgetTier.FAST] == 2
    assert result.report.tiers[0].feasible
    assert [query.selected_model_id for query in result.report.tiers[0].queries] == [
        "cheap",
        "cheap",
    ]


class _PooledAllowanceLedger:
    """Test adapter whose configured per-query allowance is pooled up front."""

    def __init__(self, budget_limit: Decimal, expected_queries: int) -> None:
        self.budget_limit = budget_limit
        self.expected_queries = expected_queries
        self.remaining = budget_limit * expected_queries
        self.spent = Decimal(0)
        self.active = False
        self.query_order: list[str] = []

    def begin_query(self, example_id: str) -> None:
        self.active = True
        self.query_order.append(example_id)

    @property
    def remaining_budget(self) -> Decimal:
        if not self.active:
            raise RuntimeError("query is not active")
        return self.remaining

    def charge_realized(self, cost: Decimal) -> bool:
        self.spent += cost
        if cost > self.remaining:
            self.remaining = Decimal(0)
            return False
        self.remaining -= cost
        return True

    def finish_query(self) -> None:
        self.active = False

    def report(self) -> BudgetReport:
        return BudgetReport(
            adapter_name="pooled-allowance",
            configured_limit=self.budget_limit,
            effective_total_limit=self.budget_limit * self.expected_queries,
            spent=self.spent,
            over_budget_calls=0,
            query_order=tuple(self.query_order),
        )


def test_exact_candidates_do_not_assume_ledger_maximum_remaining_budget() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("3")),
    )
    examples = (
        _example("q1", "a", models, {"cheap": 1.0, "premium": 0.0}),
        _example("q2", "b", models, {"cheap": 0.0, "premium": 1.0}),
    )
    predictions = CrossFittedPredictionTable(
        {
            ("q1", "cheap"): 0.0,
            ("q1", "premium"): 0.5,
            ("q2", "cheap"): 0.0,
            ("q2", "premium"): 1.5,
        }
    )
    spec = TierSpec(BudgetTier.FAST, Decimal("2"), 1.0)

    candidates = exact_lambda_candidates(examples, spec, predictions)
    result = tune_tier_lambdas(
        examples,
        (spec,),
        predictions,
        _PooledAllowanceLedger,
    )

    assert Fraction(1, 2) in candidates
    assert result.lambda_by_tier[BudgetTier.FAST] == Fraction(1, 4)
    assert result.score.weighted_quality == 1.0


def test_cross_fitting_keys_duplicate_prompts_by_private_example_id() -> None:
    models = (ModelSpec("cheap", Decimal("1")),)
    examples = tuple(
        _example(f"q-{domain}", domain, models, {"cheap": 0.5}, prompt="same prompt")
        for domain in ("a", "b", "c")
    )
    calls: list[tuple[str, ...]] = []

    def trainer(training: tuple[EvaluationExample, ...]) -> StaticQualityPredictor:
        calls.append(tuple(example.example_id for example in training))
        return StaticQualityPredictor({"cheap": 0.5})

    table = cross_fitted_prediction_table(examples, trainer)

    assert set(table.scores) == {(example.example_id, "cheap") for example in examples}
    assert len(calls) == 3


def test_prediction_materialization_fails_closed_on_invalid_predictors() -> None:
    models = (ModelSpec("cheap", Decimal("1")),)
    examples = (_example("q1", "a", models, {"cheap": 0.5}),)

    with pytest.raises(TypeError, match="must implement predict"):
        CrossFittedPredictionTable.from_predictor(examples, object())  # type: ignore[arg-type]

    class BooleanPredictor:
        def predict(self, prompt: str, model_id: str) -> bool:
            del prompt, model_id
            return True

    with pytest.raises(TypeError, match="real numbers"):
        CrossFittedPredictionTable.from_predictor(
            examples,
            BooleanPredictor(),  # type: ignore[arg-type]
        )

    class WrongLengthBatchPredictor:
        def predict_batch(self, prompts: object, model_ids: object) -> tuple[object, ...]:
            del prompts, model_ids
            return ()

    with pytest.raises(ValueError, match="wrong number of prompt rows"):
        CrossFittedPredictionTable.from_predictor(
            examples,
            WrongLengthBatchPredictor(),  # type: ignore[arg-type]
        )


def test_outer_fold_tuning_never_observes_held_out_rows() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    examples = tuple(
        _example(
            f"q-{domain}",
            domain,
            models,
            {"cheap": 0.6, "premium": 0.9},
        )
        for domain in ("a", "b", "c", "d")
    )
    training = tuple(example for example in examples if example.domain != "a")
    fold = DomainFold("a", training, (examples[0],))
    mutated_test = replace(
        examples[0],
        example_id="outer-sentinel",
        prompt="OUTER SENTINEL",
        outcomes=tuple(replace(outcome, quality=0.0) for outcome in examples[0].outcomes),
    )
    mutated_fold = DomainFold("a", training, (mutated_test,))
    observed: list[tuple[str, ...]] = []

    def trainer(rows: tuple[EvaluationExample, ...]) -> StaticQualityPredictor:
        observed.append(tuple(example.example_id for example in rows))
        return StaticQualityPredictor({"cheap": 0.6, "premium": 0.9})

    spec = TierSpec(BudgetTier.FAST, Decimal("2"), 1.0)
    first = fit_tiered_lambda_router_for_fold(
        fold,
        (spec,),
        trainer,
        PerQueryBudgetLedger,
    )
    first_trace = tuple(observed)
    observed.clear()
    second = fit_tiered_lambda_router_for_fold(
        mutated_fold,
        (spec,),
        trainer,
        PerQueryBudgetLedger,
    )

    assert all("q-a" not in call and "outer-sentinel" not in call for call in first_trace)
    assert all("q-a" not in call and "outer-sentinel" not in call for call in observed)
    assert first.tuning.lambda_by_tier == second.tuning.lambda_by_tier
    assert first.tuning.prediction_sha256 == second.tuning.prediction_sha256


def test_fold_and_nested_helpers_thread_exhaustive_preflight_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    examples = tuple(
        _example(
            f"q-{domain}",
            domain,
            models,
            {"cheap": 0.5, "premium": 0.8},
        )
        for domain in ("a", "b", "c", "d")
    )
    fold = DomainFold("a", examples[1:], (examples[0],))
    spec = TierSpec(BudgetTier.FAST, Decimal("8"), 1.0)

    def trainer(rows: tuple[EvaluationExample, ...]) -> StaticQualityPredictor:
        del rows
        return StaticQualityPredictor({"cheap": 0.5, "premium": 0.8})

    monkeypatch.setattr(lambda_tuning, "MAX_UNCONFIRMED_EXHAUSTIVE_CANDIDATES", 1)
    monkeypatch.setattr(
        lambda_tuning,
        "MAX_UNCONFIRMED_EXHAUSTIVE_UTILITY_EVALUATIONS",
        1,
    )

    with pytest.raises(ValueError, match="exhaustive lambda search refused"):
        fit_tiered_lambda_router_for_fold(
            fold,
            (spec,),
            trainer,
            PerQueryBudgetLedger,
        )
    acknowledged_fold = fit_tiered_lambda_router_for_fold(
        fold,
        (spec,),
        trainer,
        PerQueryBudgetLedger,
        allow_large_exhaustive=True,
    )
    capped_fold = fit_tiered_lambda_router_for_fold(
        fold,
        (spec,),
        trainer,
        PerQueryBudgetLedger,
        max_candidates_per_tier=2,
    )

    with pytest.raises(ValueError, match="exhaustive lambda search refused"):
        nested_lodo_lambda_evaluation(
            examples,
            (spec,),
            trainer,
            CumulativeBudgetLedger,
        )
    acknowledged_nested = nested_lodo_lambda_evaluation(
        examples,
        (spec,),
        trainer,
        CumulativeBudgetLedger,
        allow_large_exhaustive=True,
    )
    capped_nested = nested_lodo_lambda_evaluation(
        examples,
        (spec,),
        trainer,
        CumulativeBudgetLedger,
        max_candidates_per_tier=2,
    )

    assert acknowledged_fold.tuning.selections[0].candidates.exhaustive is True
    assert len(capped_fold.tuning.selections[0].candidates.values) <= 2
    assert all(
        item.tuning.selections[0].candidates.exhaustive for item in acknowledged_nested.folds
    )
    assert all(
        len(item.tuning.selections[0].candidates.values) <= 2 for item in capped_nested.folds
    )


def test_nested_lodo_replays_outer_predictions_once_in_original_order() -> None:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    examples = tuple(
        _example(
            example_id,
            domain,
            models,
            {"cheap": 0.5, "premium": 0.8},
            prompt="duplicate prompt",
        )
        for example_id, domain in (
            ("q-z", "z"),
            ("q-a", "a"),
            ("q-m", "m"),
            ("q-b", "b"),
        )
    )

    def trainer(rows: tuple[EvaluationExample, ...]) -> StaticQualityPredictor:
        assert len({row.domain for row in rows}) >= 2
        return StaticQualityPredictor({"cheap": 0.5, "premium": 0.8})

    spec = TierSpec(BudgetTier.FAST, Decimal("8"), 1.0)
    result = nested_lodo_lambda_evaluation(
        examples,
        (spec,),
        trainer,
        CumulativeBudgetLedger,
    )

    assert [fold.held_out_domain for fold in result.folds] == ["a", "b", "m", "z"]
    assert result.report.tiers[0].budget.query_order == tuple(
        example.example_id for example in examples
    )
    assert [query.example_id for query in result.report.tiers[0].queries] == [
        "q-z",
        "q-a",
        "q-m",
        "q-b",
    ]
    assert result.score.weighted_quality == pytest.approx(0.8)


def test_nested_lodo_requires_three_domains_for_inner_tuning() -> None:
    models = (ModelSpec("cheap", Decimal("1")),)
    examples = tuple(
        _example(f"q-{domain}", domain, models, {"cheap": 0.5}) for domain in ("a", "b")
    )

    with pytest.raises(ValueError, match="requires at least three domains"):
        nested_lodo_lambda_evaluation(
            examples,
            (TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),),
            lambda rows: StaticQualityPredictor({"cheap": float(len(rows))}),
            PerQueryBudgetLedger,
        )
