# SPDX-License-Identifier: Apache-2.0
"""Tests for the six baselines and default lambda policy."""

from decimal import Decimal
from fractions import Fraction

import pytest

from tierroute.core import (
    BudgetTier,
    CallModel,
    CallRecord,
    ModelSpec,
    RouterState,
    RoutingContractError,
    SelectOutput,
)
from tierroute.policies import (
    AlwaysCheapestRouter,
    AlwaysPremiumRouter,
    DomainBestRouter,
    LambdaThresholdRouter,
    LengthHeuristicRouter,
    OracleRouter,
    RandomRouter,
    TieredLambdaRouter,
    as_lambda,
    route_from_predictions,
)
from tierroute.predictors import StaticQualityPredictor

MODELS = (
    ModelSpec("cheap", Decimal("1")),
    ModelSpec("middle", Decimal("2")),
    ModelSpec("premium", Decimal("4")),
)


def state(
    prompt: str = "short prompt",
    *,
    budget: str = "4",
    tier: BudgetTier = BudgetTier.FAST,
    history: tuple[CallRecord, ...] = (),
    candidates: tuple[ModelSpec, ...] = MODELS,
) -> RouterState:
    return RouterState(
        prompt,
        tier,
        Decimal(budget),
        history,
        candidates,
        {"domain": "math"},
    )


def test_always_baselines_use_cost_tie_break_and_explicit_premium() -> None:
    tied = (ModelSpec("z", Decimal("1")), ModelSpec("a", Decimal("1")), *MODELS[1:])

    assert AlwaysCheapestRouter().route(state(candidates=tied)).model_id == "a"  # type: ignore[union-attr]
    premium = AlwaysPremiumRouter("middle").route(state())
    assert isinstance(premium, CallModel)
    assert premium.model_id == "middle"


def test_one_shot_baseline_selects_existing_output_after_call() -> None:
    action = AlwaysCheapestRouter().route(
        state(history=(CallRecord("cheap", Decimal("1"), "answer"),))
    )

    assert action == SelectOutput(0, reason="one-shot call completed")


def test_random_baseline_is_reproducible_and_candidate_order_independent() -> None:
    router = RandomRouter(seed=42)
    first = router.route(state())
    reversed_state = state(candidates=tuple(reversed(MODELS)))

    assert first == router.route(reversed_state)


def test_length_heuristic_escalates_at_exact_character_boundary() -> None:
    router = LengthHeuristicRouter("cheap", "premium", character_threshold=10)

    short = router.route(state("a" * 9))
    boundary = router.route(state("a" * 10))

    assert isinstance(short, CallModel) and short.model_id == "cheap"
    assert isinstance(boundary, CallModel) and boundary.model_id == "premium"


def test_oracle_and_domain_table_use_only_precomputed_model_ids() -> None:
    oracle = OracleRouter({(BudgetTier.FAST, "q1"): "premium"})
    domain = DomainBestRouter({(BudgetTier.FAST, "math"): "middle"}, "cheap")

    with pytest.raises(RoutingContractError, match="evaluation-only"):
        oracle.route(state())
    oracle_action = oracle.route_with_evaluation_context(state(), example_id="q1")
    assert isinstance(oracle_action, CallModel) and oracle_action.model_id == "premium"
    assert domain.route(state()).model_id == "middle"  # type: ignore[union-attr]
    unseen = RouterState(
        "prompt",
        BudgetTier.FAST,
        Decimal("4"),
        candidate_models=MODELS,
        metadata={"domain": "unseen"},
    )
    assert domain.route(unseen).model_id == "cheap"  # type: ignore[union-attr]


def test_domain_table_falls_back_when_fitted_choice_exceeds_budget() -> None:
    router = DomainBestRouter({(BudgetTier.FAST, "math"): "premium"}, "cheap")

    action = router.route(state(budget="1"))

    assert isinstance(action, CallModel)
    assert action.model_id == "cheap"


def test_lambda_policy_maximizes_quality_minus_cost_and_respects_budget() -> None:
    router = LambdaThresholdRouter(
        StaticQualityPredictor({"cheap": 0.5, "middle": 0.8, "premium": 0.95}),
        lambda_cost=0.1,
    )

    action = router.route(state(budget="2"))

    assert isinstance(action, CallModel)
    assert action.model_id == "middle"
    assert action.predicted_quality == pytest.approx(0.8)


def test_lambda_policy_uses_batch_predictor_once_per_prompt() -> None:
    class RecordingBatchPredictor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...]]] = []

        def predict(self, prompt: str, model_id: str) -> float:
            raise AssertionError(f"scalar path used for {prompt}/{model_id}")

        def predict_many(self, prompt: str, model_ids: object) -> dict[str, float]:
            ids = tuple(model_ids)  # type: ignore[arg-type]
            self.calls.append((prompt, ids))
            return {model_id: index / 10 for index, model_id in enumerate(ids)}

    predictor = RecordingBatchPredictor()
    router = LambdaThresholdRouter(predictor, lambda_cost=0)

    action = router.route(state("batch this prompt"))

    assert predictor.calls == [("batch this prompt", ("cheap", "middle", "premium"))]
    assert isinstance(action, CallModel)
    assert action.model_id == "premium"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, Fraction(0)),
        (0.1, Fraction.from_float(0.1)),
        (Decimal("0.1"), Fraction(1, 10)),
        (Fraction(2, 7), Fraction(2, 7)),
    ],
)
def test_lambda_policy_normalizes_supported_penalties_exactly(
    value: int | float | Decimal | Fraction,
    expected: Fraction,
) -> None:
    router = LambdaThresholdRouter(StaticQualityPredictor({}), value)

    assert router.lambda_cost == expected
    assert isinstance(router.lambda_cost, Fraction)
    assert as_lambda(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -1.0,
        Decimal("-0.1"),
        Fraction(-1, 3),
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        True,
    ],
)
def test_lambda_policy_rejects_invalid_penalties(
    value: int | float | Decimal | Fraction,
) -> None:
    with pytest.raises(ValueError, match="lambda_cost"):
        LambdaThresholdRouter(StaticQualityPredictor({}), value)


def test_lambda_policy_rejects_unsupported_penalty_types() -> None:
    with pytest.raises(TypeError, match="int, float, Decimal, or Fraction"):
        LambdaThresholdRouter(StaticQualityPredictor({}), "0.1")  # type: ignore[arg-type]


def test_exact_lambda_selection_handles_huge_decimal_cost_independent_of_order() -> None:
    huge = ModelSpec("huge", Decimal("1e10000"))
    small = ModelSpec("small", Decimal("1"))
    router = LambdaThresholdRouter(
        StaticQualityPredictor({"huge": 0.9, "small": 0.5}),
        lambda_cost=0,
    )

    forward = router.route(state(budget="1e10000", candidates=(small, huge)))
    reverse = router.route(state(budget="1e10000", candidates=(huge, small)))

    assert isinstance(forward, CallModel) and forward.model_id == "huge"
    assert reverse == forward


def test_exact_lambda_ties_use_decimal_cost_then_model_id() -> None:
    candidates = (
        ModelSpec("costly", Decimal("1")),
        ModelSpec("free-z", Decimal("0")),
        ModelSpec("free-a", Decimal("0")),
    )
    predictions = {"costly": 0.5, "free-z": 0.0, "free-a": 0.0}

    action = route_from_predictions(
        state(candidates=tuple(reversed(candidates))),
        predictions,
        Fraction(1, 2),
    )

    assert action.model_id == "free-a"
    assert action.predicted_quality == 0.0


def test_prediction_selection_requires_exact_affordable_model_coverage() -> None:
    one_model_state = state(budget="1")

    with pytest.raises(ValueError, match="every affordable model exactly"):
        route_from_predictions(one_model_state, {}, 0)
    with pytest.raises(ValueError, match="every affordable model exactly"):
        route_from_predictions(one_model_state, {"cheap": 0.5, "middle": 0.8}, 0)


@pytest.mark.parametrize("quality", [True, float("nan"), float("inf"), "not-a-score"])
def test_prediction_selection_rejects_invalid_quality(quality: object) -> None:
    with pytest.raises(ValueError, match="predicted quality for 'cheap' must be finite"):
        route_from_predictions(state(budget="1"), {"cheap": quality}, 0)  # type: ignore[dict-item]


def test_prediction_selection_rejects_a_second_one_shot_call() -> None:
    completed = state(history=(CallRecord("cheap", Decimal("1"), "answer"),))

    with pytest.raises(RoutingContractError, match="after a one-shot call"):
        route_from_predictions(completed, {model.model_id: 0.5 for model in MODELS}, 0)


def test_tiered_lambda_router_uses_immutable_exact_per_tier_values() -> None:
    configured = {
        BudgetTier.FAST: Fraction(1, 4),
        BudgetTier.PREMIUM: Decimal("0"),
    }
    router = TieredLambdaRouter(
        StaticQualityPredictor({"cheap": 0.5, "middle": 0.75, "premium": 1.0}),
        configured,
    )
    configured[BudgetTier.FAST] = Fraction(0)

    fast = router.route(state(tier=BudgetTier.FAST))
    premium = router.route(state(tier=BudgetTier.PREMIUM))

    assert router.lambda_by_tier[BudgetTier.FAST] == Fraction(1, 4)
    assert isinstance(fast, CallModel) and fast.model_id == "cheap"
    assert isinstance(premium, CallModel) and premium.model_id == "premium"
    with pytest.raises(TypeError):
        router.lambda_by_tier[BudgetTier.FAST] = Fraction(0)  # type: ignore[index]


def test_tiered_lambda_router_reports_unconfigured_tier() -> None:
    router = TieredLambdaRouter(
        StaticQualityPredictor({"cheap": 0.5}),
        {BudgetTier.FAST: 0},
    )

    with pytest.raises(RoutingContractError, match="budget tier 'balanced'"):
        router.route(
            state(
                budget="1",
                tier=BudgetTier.BALANCED,
                candidates=(MODELS[0],),
            )
        )


def test_tiered_lambda_router_requires_enum_keys() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TieredLambdaRouter(StaticQualityPredictor({}), {})
    with pytest.raises(TypeError, match="keys must be BudgetTier"):
        TieredLambdaRouter(
            StaticQualityPredictor({}),
            {"fast": 0},  # type: ignore[dict-item]
        )


def test_tiered_lambda_router_batches_once_and_selects_after_a_completed_call() -> None:
    class RecordingBatchPredictor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...]]] = []

        def predict_many(self, prompt: str, model_ids: object) -> dict[str, float]:
            ids = tuple(model_ids)  # type: ignore[arg-type]
            self.calls.append((prompt, ids))
            return {model_id: index / 10 for index, model_id in enumerate(ids)}

    predictor = RecordingBatchPredictor()
    router = TieredLambdaRouter(predictor, {BudgetTier.FAST: 0})  # type: ignore[arg-type]

    action = router.route(state("batch tiered prompt"))
    completed = router.route(state(history=(CallRecord("cheap", Decimal("1"), "answer"),)))

    assert predictor.calls == [("batch tiered prompt", ("cheap", "middle", "premium"))]
    assert isinstance(action, CallModel) and action.model_id == "premium"
    assert completed == SelectOutput(0, reason="one-shot call completed")
