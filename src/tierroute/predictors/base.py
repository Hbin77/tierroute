# SPDX-License-Identifier: Apache-2.0
"""Quality predictor protocols and dependency-free reference implementations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class QualityPredictor(Protocol):
    """Predict the quality of one model on one prompt before calling it."""

    def predict(self, prompt: str, model_id: str) -> float:
        """Return a finite score where larger means better expected quality."""
        ...


@dataclass(frozen=True, slots=True)
class StaticQualityPredictor:
    """A transparent predictor useful for smoke tests and policy debugging."""

    scores: Mapping[str, float]

    def predict(self, prompt: str, model_id: str) -> float:
        del prompt
        try:
            score = float(self.scores[model_id])
        except KeyError as error:
            raise KeyError(f"no quality estimate for model {model_id!r}") from error
        if not math.isfinite(score):
            raise ValueError(f"quality estimate for {model_id!r} must be finite")
        return score


@dataclass(frozen=True, slots=True)
class BilinearQualityPredictor:
    """Score a prompt vector against model-specific learned weights.

    Training is deliberately outside this class. W2 can compare this fixed inference
    form with a GBM while keeping the policy API unchanged.
    """

    vectorizer: Callable[[str], Sequence[float]] = field(compare=False, repr=False)
    model_weights: Mapping[str, Sequence[float]]
    model_bias: Mapping[str, float] = field(default_factory=dict)

    def predict(self, prompt: str, model_id: str) -> float:
        try:
            weights = tuple(float(value) for value in self.model_weights[model_id])
        except KeyError as error:
            raise KeyError(f"no bilinear weights for model {model_id!r}") from error
        vector = tuple(float(value) for value in self.vectorizer(prompt))
        if len(vector) != len(weights):
            raise ValueError(
                f"feature width {len(vector)} does not match {len(weights)} weights "
                f"for {model_id!r}"
            )
        score = sum(value * weight for value, weight in zip(vector, weights, strict=True))
        score += float(self.model_bias.get(model_id, 0.0))
        if not math.isfinite(score):
            raise ValueError("bilinear prediction must be finite")
        return score
