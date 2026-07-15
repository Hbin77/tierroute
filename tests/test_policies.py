# SPDX-License-Identifier: Apache-2.0
"""Tests for the six baselines and default lambda policy."""

from decimal import Decimal

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
    history: tuple[CallRecord, ...] = (),
    candidates: tuple[ModelSpec, ...] = MODELS,
) -> RouterState:
    return RouterState(
        prompt,
        BudgetTier.FAST,
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


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_lambda_policy_rejects_invalid_penalties(value: float) -> None:
    with pytest.raises(ValueError, match="lambda_cost"):
        LambdaThresholdRouter(StaticQualityPredictor({}), value)
