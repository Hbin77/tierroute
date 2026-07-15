# SPDX-License-Identifier: Apache-2.0
"""Default one-shot Lagrangian quality-versus-cost policies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import TypeAlias

from tierroute.core import (
    BudgetTier,
    CallModel,
    ModelSpec,
    RouterAction,
    RouterState,
    RoutingContractError,
    SelectOutput,
)
from tierroute.predictors import BatchQualityPredictor, QualityPredictor

LambdaInput: TypeAlias = int | float | Decimal | Fraction


def as_lambda(value: LambdaInput) -> Fraction:
    """Normalize a finite, non-negative lambda without losing precision.

    Float inputs retain their exact binary value. Decimal and Fraction inputs retain
    their exact mathematical value. This distinction matters at policy breakpoints,
    where rounding a lambda can change the selected model.
    """

    if isinstance(value, bool):
        raise ValueError("lambda_cost must be finite and non-negative")
    if isinstance(value, Fraction):
        normalized = value
    elif isinstance(value, int):
        normalized = Fraction(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("lambda_cost must be finite and non-negative")
        normalized = Fraction.from_float(value)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("lambda_cost must be finite and non-negative")
        normalized = Fraction(value)
    else:
        raise TypeError("lambda_cost must be an int, float, Decimal, or Fraction")
    if normalized < 0:
        raise ValueError("lambda_cost must be finite and non-negative")
    return normalized


def _affordable_models(state: RouterState) -> tuple[ModelSpec, ...]:
    affordable = tuple(
        model for model in state.candidate_models if model.cost <= state.remaining_budget
    )
    if not affordable:
        raise RoutingContractError("no candidate model fits the remaining budget")
    return affordable


def _lambda_label(value: Fraction) -> str:
    """Render ordinary lambdas while keeping huge exact values safe to report."""

    if value.numerator.bit_length() <= 256 and value.denominator.bit_length() <= 256:
        return str(value)
    return (
        "exact-rational["
        f"numerator_bits={value.numerator.bit_length()},"
        f"denominator_bits={value.denominator.bit_length()}]"
    )


def route_from_predictions(
    state: RouterState,
    predictions: Mapping[str, float],
    lambda_cost: LambdaInput,
) -> CallModel:
    """Select one affordable call from supplied predictions using exact utility.

    This is the shared selection primitive for runtime routing and offline tuning.
    Callers must handle completed one-shot histories before invoking it. Utilities are
    exact fractions: the binary float prediction is converted with
    :meth:`Fraction.from_float`, while each Decimal cost is converted directly and is
    never rounded through a float.
    """

    if state.call_history:
        raise RoutingContractError("cannot select a new model after a one-shot call")
    normalized_lambda = as_lambda(lambda_cost)
    affordable = _affordable_models(state)
    expected_model_ids = {model.model_id for model in affordable}
    if set(predictions) != expected_model_ids:
        raise ValueError("predictions must return every affordable model exactly")

    scored: list[tuple[Fraction, Decimal, str, float]] = []
    for model in affordable:
        raw_quality = predictions[model.model_id]
        if isinstance(raw_quality, bool):
            raise ValueError(f"predicted quality for {model.model_id!r} must be finite")
        try:
            quality = float(raw_quality)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"predicted quality for {model.model_id!r} must be finite") from error
        if not math.isfinite(quality):
            raise ValueError(f"predicted quality for {model.model_id!r} must be finite")

        quality_fraction = Fraction.from_float(quality)
        utility = quality_fraction - normalized_lambda * Fraction(model.cost)
        scored.append((utility, model.cost, model.model_id, quality))

    _, _, model_id, quality = min(
        scored,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return CallModel(
        model_id,
        reason=(f"max exact predicted_quality - {_lambda_label(normalized_lambda)} * cost"),
        predicted_quality=quality,
    )


def _route_with_predictor(
    state: RouterState,
    predictor: QualityPredictor,
    lambda_cost: Fraction,
) -> RouterAction:
    if state.call_history:
        return SelectOutput(len(state.call_history) - 1, reason="one-shot call completed")

    affordable = _affordable_models(state)
    model_ids = tuple(model.model_id for model in affordable)
    if isinstance(predictor, BatchQualityPredictor):
        predictions = predictor.predict_many(state.prompt, model_ids)
    else:
        predictions = {
            model_id: predictor.predict(state.prompt, model_id) for model_id in model_ids
        }
    return route_from_predictions(state, predictions, lambda_cost)


@dataclass(frozen=True, slots=True)
class LambdaThresholdRouter:
    """Maximize predicted quality minus one fixed ``lambda_cost * cost``.

    This compatibility router shares one lambda across tiers. Fit that value only on
    training domains; deploy :class:`TieredLambdaRouter` when each service tier has
    been tuned directly against its own budgeted metric.
    """

    predictor: QualityPredictor
    lambda_cost: LambdaInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_cost", as_lambda(self.lambda_cost))

    def route(self, state: RouterState) -> RouterAction:
        """Route with one lambda shared by all service tiers."""

        return _route_with_predictor(state, self.predictor, self.lambda_cost)


@dataclass(frozen=True, slots=True)
class TieredLambdaRouter:
    """Use one immutable, exact lambda for each configured budget tier.

    Keeping the mapping immutable makes the fitted policy auditable: a routing run
    cannot silently change its quality-versus-cost trade-off after construction.
    """

    predictor: QualityPredictor
    lambda_by_tier: Mapping[BudgetTier, LambdaInput]

    def __post_init__(self) -> None:
        if not isinstance(self.lambda_by_tier, Mapping):
            raise TypeError("lambda_by_tier must be a mapping")
        if not self.lambda_by_tier:
            raise ValueError("lambda_by_tier must not be empty")
        normalized: dict[BudgetTier, Fraction] = {}
        for tier, value in self.lambda_by_tier.items():
            if not isinstance(tier, BudgetTier):
                raise TypeError("lambda_by_tier keys must be BudgetTier values")
            normalized[tier] = as_lambda(value)
        object.__setattr__(self, "lambda_by_tier", MappingProxyType(normalized))

    def route(self, state: RouterState) -> RouterAction:
        """Route with the lambda configured for ``state.budget_tier``."""

        if state.call_history:
            return SelectOutput(len(state.call_history) - 1, reason="one-shot call completed")
        try:
            lambda_cost = self.lambda_by_tier[state.budget_tier]
        except KeyError as error:
            raise RoutingContractError(
                f"no lambda configured for budget tier {state.budget_tier.value!r}"
            ) from error
        return _route_with_predictor(state, self.predictor, lambda_cost)
