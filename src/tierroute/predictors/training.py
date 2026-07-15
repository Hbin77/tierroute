# SPDX-License-Identifier: Apache-2.0
"""Deterministic, inner-LODO training for calibrated bilinear predictors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tierroute.eval import (
    DomainFold,
    EvaluationExample,
    evaluation_data_sha256,
    leave_one_domain_out,
)
from tierroute.features import EmbeddingProvider, PromptFeatureEncoder
from tierroute.predictors._ridge import CENTERED_RIDGE_SOLVER_ID
from tierroute.predictors.artifacts import BilinearPredictorArtifact
from tierroute.predictors.base import BilinearQualityPredictor
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.solvers import (
    RidgeSolver,
    fit_targets_with_solver,
    resolve_ridge_solver,
    validate_ridge_solver_id,
)


@dataclass(frozen=True, slots=True)
class BilinearTrainingConfig:
    """Reproducible trainer settings stored with every artifact."""

    ridge: float = 1.0
    seed: int = 0
    solver_id: str = CENTERED_RIDGE_SOLVER_ID

    def __post_init__(self) -> None:
        if isinstance(self.ridge, bool) or not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        validate_ridge_solver_id(self.solver_id)


@dataclass(frozen=True, slots=True)
class _FittedBase:
    encoder: PromptFeatureEncoder
    predictor: BilinearQualityPredictor
    model_weights: dict[str, tuple[float, ...]]
    model_bias: dict[str, float]


def _ordered_examples(
    examples: Sequence[EvaluationExample],
) -> tuple[EvaluationExample, ...]:
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    if not ordered:
        raise ValueError("training examples must not be empty")
    ids = [example.example_id for example in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("training examples must have unique example IDs")
    return ordered


def _model_ids(examples: tuple[EvaluationExample, ...]) -> tuple[str, ...]:
    expected = tuple(sorted(model.model_id for model in examples[0].candidate_models))
    for example in examples[1:]:
        current = tuple(sorted(model.model_id for model in example.candidate_models))
        if current != expected:
            raise ValueError("every training example must contain the same model catalogue")
    return expected


def training_data_sha256(examples: Sequence[EvaluationExample]) -> str:
    """Hash all fields that can influence predictor fitting."""

    return evaluation_data_sha256(examples)


def _fit_base(
    examples: Sequence[EvaluationExample],
    config: BilinearTrainingConfig,
    embedding_provider: EmbeddingProvider | None,
    solver: RidgeSolver,
) -> _FittedBase:
    ordered = _ordered_examples(examples)
    model_ids = _model_ids(ordered)
    prompts = tuple(example.prompt for example in ordered)
    encoder = PromptFeatureEncoder.fit(
        prompts,
        embedding_provider=embedding_provider,
    )
    solver.preflight(
        sample_count=len(ordered),
        feature_count=encoder.schema.dimension,
        target_count=len(model_ids),
    )
    feature_rows = encoder.transform_many(prompts)
    if len(feature_rows) != len(ordered) or any(
        len(row) != encoder.schema.dimension for row in feature_rows
    ):
        raise ValueError("encoded feature matrix has an unexpected shape")
    targets_by_model = {
        model_id: tuple(
            next(outcome.quality for outcome in example.outcomes if outcome.model_id == model_id)
            for example in ordered
        )
        for model_id in model_ids
    }
    fitted = fit_targets_with_solver(
        solver,
        feature_rows,
        targets_by_model,
        ridge=config.ridge,
    )
    model_weights = {model_id: fitted[model_id][0] for model_id in model_ids}
    model_bias = {model_id: fitted[model_id][1] for model_id in model_ids}
    if any(
        not math.isfinite(value)
        for model_id in model_ids
        for value in (*model_weights[model_id], model_bias[model_id])
    ):
        raise ValueError("bilinear fitting produced non-finite coefficients")

    predictor = BilinearQualityPredictor(
        vectorizer=encoder.transform_one,
        model_weights=model_weights,
        model_bias=model_bias,
        batch_vectorizer=encoder.transform_many,
    )
    return _FittedBase(encoder, predictor, model_weights, model_bias)


def fit_calibrated_bilinear(
    training_examples: Sequence[EvaluationExample],
    *,
    config: BilinearTrainingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> BilinearPredictorArtifact:
    """Fit base weights on all training data and calibration on inner LODO predictions.

    Callers performing outer LODO must pass only the outer fold's ``training`` side.
    :func:`fit_calibrated_bilinear_for_fold` provides that safer orchestration path.
    """

    config = config or BilinearTrainingConfig()
    solver = resolve_ridge_solver(config.solver_id)
    ordered = _ordered_examples(training_examples)
    model_ids = _model_ids(ordered)
    folds = leave_one_domain_out(ordered)
    oof_predictions: dict[str, list[float]] = {model_id: [] for model_id in model_ids}
    oof_targets: dict[str, list[float]] = {model_id: [] for model_id in model_ids}

    for fold in folds:
        fitted = _fit_base(fold.training, config, embedding_provider, solver)
        predictions_by_example = fitted.predictor.predict_batch(
            tuple(example.prompt for example in fold.test),
            model_ids,
        )
        for example, predictions in zip(
            fold.test,
            predictions_by_example,
            strict=True,
        ):
            outcomes = {outcome.model_id: outcome for outcome in example.outcomes}
            for model_id in model_ids:
                oof_predictions[model_id].append(float(predictions[model_id]))
                oof_targets[model_id].append(outcomes[model_id].quality)

    if any(len(values) != len(ordered) for values in oof_predictions.values()):
        raise AssertionError("inner LODO must predict every training example exactly once")
    calibrators = {
        model_id: IsotonicCalibrator.fit(
            oof_predictions[model_id],
            oof_targets[model_id],
        )
        for model_id in model_ids
    }
    fitted = _fit_base(ordered, config, embedding_provider, solver)
    return BilinearPredictorArtifact(
        feature_schema=fitted.encoder.schema,
        model_weights=fitted.model_weights,
        model_bias=fitted.model_bias,
        calibrators=calibrators,
        training_data_sha256=training_data_sha256(ordered),
        training_example_count=len(ordered),
        training_domains=tuple(sorted({example.domain for example in ordered})),
        ridge=config.ridge,
        seed=config.seed,
        solver_id=solver.solver_id,
    )


def fit_calibrated_bilinear_for_fold(
    fold: DomainFold,
    *,
    config: BilinearTrainingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> BilinearPredictorArtifact:
    """Fit exclusively on an outer LODO fold's training side."""

    return fit_calibrated_bilinear(
        fold.training,
        config=config,
        embedding_provider=embedding_provider,
    )
