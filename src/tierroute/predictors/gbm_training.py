# SPDX-License-Identifier: Apache-2.0
"""Leakage-aware training for the dependency-free GBM reference predictor.

This module intentionally returns an in-memory predictor.  Portable artifact and
route-CLI support belong to a later schema version so the pinned bilinear artifact
bytes remain unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tierroute.eval import DomainFold, EvaluationExample, leave_one_domain_out
from tierroute.features import EmbeddingProvider, PromptFeatureEncoder
from tierroute.predictors.calibration import (
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
)
from tierroute.predictors.gbm import (
    GbmModel,
    GbmQualityPredictor,
    fit_gradient_boosted_stumps,
)

GBM_ALGORITHM_ID = "tierroute-gradient-boosted-regression-stumps-v1"
MAX_GBM_ESTIMATORS = 256
MAX_GBM_TRAINING_CELLS = 2_000_000
MAX_GBM_SPLIT_SCANS = 100_000_000
MAX_GBM_TOTAL_STUMPS = 65_536


@dataclass(frozen=True, slots=True)
class GbmTrainingConfig:
    """Fixed, deterministic reference-trainer settings.

    There is no seed because this implementation has no random sampling.  Keeping
    an unused seed would suggest a reproducibility control that does nothing.
    """

    n_estimators: int = 32
    learning_rate: float = 0.1
    min_samples_leaf: int = 2
    min_gain: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, int)
            or not 1 <= self.n_estimators <= MAX_GBM_ESTIMATORS
        ):
            raise ValueError(f"n_estimators must be an integer from 1 to {MAX_GBM_ESTIMATORS}")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(float(self.learning_rate))
            or not 0 < float(self.learning_rate) <= 1
        ):
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if (
            isinstance(self.min_samples_leaf, bool)
            or not isinstance(self.min_samples_leaf, int)
            or self.min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")
        if (
            isinstance(self.min_gain, bool)
            or not isinstance(self.min_gain, (int, float))
            or not math.isfinite(float(self.min_gain))
            or float(self.min_gain) < 0
        ):
            raise ValueError("min_gain must be finite and non-negative")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "min_gain", float(self.min_gain))


@dataclass(frozen=True, slots=True)
class _FittedBase:
    encoder: PromptFeatureEncoder
    predictor: GbmQualityPredictor
    models: dict[str, GbmModel]


def _ordered_examples(
    examples: Sequence[EvaluationExample],
) -> tuple[EvaluationExample, ...]:
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    if not ordered:
        raise ValueError("training examples must not be empty")
    ids = tuple(example.example_id for example in ordered)
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


def preflight_gbm_fit(
    *,
    sample_count: int,
    feature_count: int,
    target_count: int,
    config: GbmTrainingConfig,
) -> None:
    """Reject an unsafe reference fit before allocating the feature matrix.

    The split-scan estimate is deliberately conservative: every requested round is
    charged even though early stopping may produce fewer stumps.  Nested benchmark
    orchestration must add its own aggregate fold-amplification preflight later.
    """

    for name, value in (
        ("sample_count", sample_count),
        ("feature_count", feature_count),
        ("target_count", target_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(config, GbmTrainingConfig):
        raise TypeError("config must be a GbmTrainingConfig")
    dense_cells = sample_count * feature_count
    if dense_cells > MAX_GBM_TRAINING_CELLS:
        raise ValueError(
            "dependency-free GBM feature matrix exceeds the reviewed limit "
            f"({dense_cells:,} > {MAX_GBM_TRAINING_CELLS:,})"
        )
    possible_stumps = target_count * config.n_estimators
    if possible_stumps > MAX_GBM_TOTAL_STUMPS:
        raise ValueError(
            "dependency-free GBM ensemble exceeds the reviewed stump limit "
            f"({possible_stumps:,} > {MAX_GBM_TOTAL_STUMPS:,})"
        )
    split_scans = _estimated_split_scans(
        sample_count=sample_count,
        feature_count=feature_count,
        target_count=target_count,
        config=config,
    )
    if split_scans > MAX_GBM_SPLIT_SCANS:
        raise ValueError(
            "dependency-free GBM split scan exceeds the reviewed limit "
            f"({split_scans:,} > {MAX_GBM_SPLIT_SCANS:,})"
        )


def _estimated_split_scans(
    *,
    sample_count: int,
    feature_count: int,
    target_count: int,
    config: GbmTrainingConfig,
) -> int:
    return target_count * config.n_estimators * feature_count * max(sample_count - 1, 0)


def _preflight_calibrated_fit(
    ordered: tuple[EvaluationExample, ...],
    model_ids: tuple[str, ...],
    folds: tuple[DomainFold, ...],
    config: GbmTrainingConfig,
    embedding_provider: EmbeddingProvider | None,
) -> None:
    """Preflight every inner and final base fit before the first embedding call."""

    total_split_scans = 0
    for examples in (*(fold.training for fold in folds), ordered):
        subset = _ordered_examples(examples)
        encoder = PromptFeatureEncoder.fit(
            tuple(example.prompt for example in subset),
            embedding_provider=embedding_provider,
        )
        preflight_gbm_fit(
            sample_count=len(subset),
            feature_count=encoder.schema.dimension,
            target_count=len(model_ids),
            config=config,
        )
        total_split_scans += _estimated_split_scans(
            sample_count=len(subset),
            feature_count=encoder.schema.dimension,
            target_count=len(model_ids),
            config=config,
        )
    if total_split_scans > MAX_GBM_SPLIT_SCANS:
        raise ValueError(
            "calibrated dependency-free GBM split scan exceeds the reviewed limit "
            f"({total_split_scans:,} > {MAX_GBM_SPLIT_SCANS:,})"
        )


def _fit_base(
    examples: Sequence[EvaluationExample],
    config: GbmTrainingConfig,
    embedding_provider: EmbeddingProvider | None,
) -> _FittedBase:
    ordered = _ordered_examples(examples)
    model_ids = _model_ids(ordered)
    prompts = tuple(example.prompt for example in ordered)
    encoder = PromptFeatureEncoder.fit(prompts, embedding_provider=embedding_provider)
    preflight_gbm_fit(
        sample_count=len(ordered),
        feature_count=encoder.schema.dimension,
        target_count=len(model_ids),
        config=config,
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
    models = fit_gradient_boosted_stumps(
        feature_rows,
        targets_by_model,
        example_ids=tuple(example.example_id for example in ordered),
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        min_samples_leaf=config.min_samples_leaf,
        min_gain=config.min_gain,
    )
    predictor = GbmQualityPredictor(
        vectorizer=encoder.transform_one,
        models=models,
        batch_vectorizer=encoder.transform_many,
    )
    return _FittedBase(encoder=encoder, predictor=predictor, models=models)


def fit_calibrated_gbm(
    training_examples: Sequence[EvaluationExample],
    *,
    config: GbmTrainingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> PerModelCalibratedQualityPredictor:
    """Fit GBM heads and per-model isotonic calibration using inner LODO.

    Callers running an outer LODO protocol must pass only the outer fold's training
    side.  :func:`fit_calibrated_gbm_for_fold` makes that boundary explicit.
    This phase returns an in-memory predictor and deliberately provides no artifact
    serialization or route-CLI loading contract.
    """

    if config is None:
        config = GbmTrainingConfig()
    elif not isinstance(config, GbmTrainingConfig):
        raise TypeError("config must be a GbmTrainingConfig or None")
    ordered = _ordered_examples(training_examples)
    model_ids = _model_ids(ordered)
    folds = leave_one_domain_out(ordered)
    _preflight_calibrated_fit(
        ordered,
        model_ids,
        folds,
        config,
        embedding_provider,
    )
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
    return PerModelCalibratedQualityPredictor(
        base=fitted.predictor,
        calibrators=calibrators,
    )


def fit_calibrated_gbm_for_fold(
    fold: DomainFold,
    *,
    config: GbmTrainingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> PerModelCalibratedQualityPredictor:
    """Fit exclusively on an outer LODO fold's training side."""

    if not isinstance(fold, DomainFold):
        raise TypeError("fold must be a DomainFold")
    return fit_calibrated_gbm(
        fold.training,
        config=config,
        embedding_provider=embedding_provider,
    )
