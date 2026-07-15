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


@runtime_checkable
class BatchQualityPredictor(Protocol):
    """Predict all requested models while computing prompt features once."""

    def predict_many(self, prompt: str, model_ids: Sequence[str]) -> Mapping[str, float]:
        """Return one finite score for every requested model ID."""
        ...


@runtime_checkable
class BatchPromptQualityPredictor(Protocol):
    """Predict a prompt batch with one vectorization call."""

    def predict_batch(
        self, prompts: Sequence[str], model_ids: Sequence[str]
    ) -> tuple[Mapping[str, float], ...]:
        """Return one model-score mapping per prompt in input order."""
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
    batch_vectorizer: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def _score_vector(self, vector: tuple[float, ...], model_id: str) -> float:
        try:
            weights = tuple(float(value) for value in self.model_weights[model_id])
        except KeyError as error:
            raise KeyError(f"no bilinear weights for model {model_id!r}") from error
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

    def predict(self, prompt: str, model_id: str) -> float:
        vector = tuple(float(value) for value in self.vectorizer(prompt))
        return self._score_vector(vector, model_id)

    def predict_many(self, prompt: str, model_ids: Sequence[str]) -> Mapping[str, float]:
        """Score models after vectorizing the prompt exactly once."""

        vector = tuple(float(value) for value in self.vectorizer(prompt))
        return {model_id: self._score_vector(vector, model_id) for model_id in model_ids}

    def predict_batch(
        self, prompts: Sequence[str], model_ids: Sequence[str]
    ) -> tuple[Mapping[str, float], ...]:
        """Vectorize a prompt batch once and score every requested model."""

        prompts = tuple(prompts)
        model_ids = tuple(model_ids)
        if not prompts:
            raise ValueError("prompts must not be empty")
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError("model_ids must be non-empty and unique")
        if self.batch_vectorizer is None:
            vectors = tuple(self.vectorizer(prompt) for prompt in prompts)
        else:
            vectors = tuple(self.batch_vectorizer(prompts))
        if len(vectors) != len(prompts):
            raise ValueError("batch vectorizer returned the wrong number of rows")
        return tuple(
            {
                model_id: self._score_vector(tuple(float(value) for value in vector), model_id)
                for model_id in model_ids
            }
            for vector in vectors
        )
