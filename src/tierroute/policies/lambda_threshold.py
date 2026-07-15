# SPDX-License-Identifier: Apache-2.0
"""Default one-shot Lagrangian quality-versus-cost policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tierroute.core import (
    CallModel,
    Cost,
    RouterAction,
    RouterState,
    RoutingContractError,
    SelectOutput,
)
from tierroute.predictors import QualityPredictor


@dataclass(frozen=True, slots=True)
class LambdaThresholdRouter:
    """Maximize predicted quality minus ``lambda_cost * cost`` in one call.

    This is the one-shot constrained-routing core. The lambda value is intended to be
    tuned against the weighted tier metric on training domains, never on a LODO test fold.
    """

    predictor: QualityPredictor
    lambda_cost: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_cost) or self.lambda_cost < 0:
            raise ValueError("lambda_cost must be finite and non-negative")

    def route(self, state: RouterState) -> RouterAction:
        if state.call_history:
            return SelectOutput(len(state.call_history) - 1, reason="one-shot call completed")
        affordable = [
            model for model in state.candidate_models if model.cost <= state.remaining_budget
        ]
        if not affordable:
            raise RoutingContractError("no candidate model fits the remaining budget")

        scored: list[tuple[float, Cost, str, float]] = []
        for model in affordable:
            quality = float(self.predictor.predict(state.prompt, model.model_id))
            if not math.isfinite(quality):
                raise ValueError(f"predicted quality for {model.model_id!r} must be finite")
            utility = quality - self.lambda_cost * float(model.cost)
            scored.append((utility, model.cost, model.model_id, quality))
        _, _, model_id, quality = min(
            scored,
            key=lambda item: (-item[0], item[1], item[2]),
        )
        return CallModel(
            model_id,
            reason=f"max predicted_quality - {self.lambda_cost:g} * cost",
            predicted_quality=quality,
        )
