# SPDX-License-Identifier: Apache-2.0
"""Deterministic, inner-LODO training for calibrated bilinear predictors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from tierroute.eval import DomainFold, EvaluationExample, leave_one_domain_out
from tierroute.features import EmbeddingProvider, PromptFeatureEncoder
from tierroute.predictors._ridge import fit_centered_ridge
from tierroute.predictors.artifacts import BilinearPredictorArtifact
from tierroute.predictors.base import BilinearQualityPredictor
from tierroute.predictors.calibration import IsotonicCalibrator


@dataclass(frozen=True, slots=True)
class BilinearTrainingConfig:
    """Reproducible trainer settings stored with every artifact."""

    ridge: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.ridge, bool) or not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


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

    ordered = _ordered_examples(examples)
    payload = []
    for example in ordered:
        outcomes = {outcome.model_id: outcome for outcome in example.outcomes}
        payload.append(
            {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "domain": example.domain,
                "models": [
                    {
                        "model_id": model.model_id,
                        "quoted_cost": format(model.cost, "f"),
                        "realized_cost": format(outcomes[model.model_id].cost, "f"),
                        "quality": outcomes[model.model_id].quality,
                    }
                    for model in sorted(
                        example.candidate_models,
                        key=lambda candidate: candidate.model_id,
                    )
                ],
            }
        )
    document = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(document).hexdigest()


def _fit_base(
    examples: Sequence[EvaluationExample],
    config: BilinearTrainingConfig,
    embedding_provider: EmbeddingProvider | None,
) -> _FittedBase:
    ordered = _ordered_examples(examples)
    model_ids = _model_ids(ordered)
    prompts = tuple(example.prompt for example in ordered)
    encoder = PromptFeatureEncoder.fit(
        prompts,
        embedding_provider=embedding_provider,
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
    fitted = fit_centered_ridge(feature_rows, targets_by_model, ridge=config.ridge)
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
    ordered = _ordered_examples(training_examples)
    model_ids = _model_ids(ordered)
    folds = leave_one_domain_out(ordered)
    oof_predictions: dict[str, list[float]] = {model_id: [] for model_id in model_ids}
    oof_targets: dict[str, list[float]] = {model_id: [] for model_id in model_ids}

    for fold in folds:
        fitted = _fit_base(fold.training, config, embedding_provider)
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
    fitted = _fit_base(ordered, config, embedding_provider)
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
