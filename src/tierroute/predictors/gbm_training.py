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
from types import MappingProxyType

from tierroute.eval import DomainFold, EvaluationExample, leave_one_domain_out
from tierroute.features import EmbeddingProvider, PromptFeatureEncoder
from tierroute.predictors.calibration import (
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
)
from tierroute.predictors.gbm import (
    MAX_GBM_STUMPS_PER_MODEL,
    GbmModel,
    GbmQualityPredictor,
    _fit_gradient_boosted_stumps,
)
from tierroute.predictors.resource_limits import MAX_PREDICTOR_MODELS

GBM_ALGORITHM_ID = "tierroute-gradient-boosted-regression-stumps-v1"
MAX_GBM_ESTIMATORS = MAX_GBM_STUMPS_PER_MODEL
MAX_GBM_TRAINING_CELLS = 2_000_000
MAX_GBM_SPLIT_SCANS = 100_000_000
MAX_GBM_TOTAL_STUMPS = 65_536


def _config_float(value: object, name: str) -> float:
    """Normalize one exact built-in number once to avoid stateful coercion."""

    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a built-in real number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


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
        if type(self.n_estimators) is not int or not 1 <= self.n_estimators <= MAX_GBM_ESTIMATORS:
            raise ValueError(f"n_estimators must be an integer from 1 to {MAX_GBM_ESTIMATORS}")
        learning_rate = _config_float(self.learning_rate, "learning_rate")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if type(self.min_samples_leaf) is not int or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        min_gain = _config_float(self.min_gain, "min_gain")
        if min_gain < 0:
            raise ValueError("min_gain must be finite and non-negative")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "min_gain", min_gain)


@dataclass(frozen=True, slots=True)
class GbmNestedLodoWorkEstimate:
    """Immutable exact work estimate for one nested-LODO GBM benchmark.

    The three per-base-fit tuples follow execution order and make the aggregate
    auditable. Feature widths are fitted independently because a training subset
    can expose a different surface-tag vocabulary. An offline embedding provider,
    when supplied, contributes its declared width but is never asked to embed.
    """

    domain_count: int
    outer_fold_count: int
    calibrated_fit_count: int
    base_fit_count: int
    model_count: int
    estimator_count: int
    split_scans: int
    base_fit_sample_counts: tuple[int, ...]
    base_fit_feature_counts: tuple[int, ...]
    base_fit_split_scans: tuple[int, ...]

    def __post_init__(self) -> None:
        scalar_counts = (
            self.domain_count,
            self.outer_fold_count,
            self.calibrated_fit_count,
            self.base_fit_count,
            self.model_count,
            self.estimator_count,
        )
        if any(type(value) is not int or value < 1 for value in scalar_counts):
            raise ValueError("nested-LODO GBM work counts must be positive integers")
        if type(self.split_scans) is not int or self.split_scans < 0:
            raise ValueError("nested-LODO GBM split_scans must be a non-negative integer")
        sample_counts = tuple(self.base_fit_sample_counts)
        feature_counts = tuple(self.base_fit_feature_counts)
        split_scans = tuple(self.base_fit_split_scans)
        if not (
            len(sample_counts)
            == len(feature_counts)
            == len(split_scans)
            == self.base_fit_count
        ):
            raise ValueError("nested-LODO GBM base-fit detail lengths must match base_fit_count")
        if any(type(value) is not int or value < 1 for value in (*sample_counts, *feature_counts)):
            raise ValueError("nested-LODO GBM base-fit sizes must be positive integers")
        if any(type(value) is not int or value < 0 for value in split_scans):
            raise ValueError("nested-LODO GBM base-fit split scans must be non-negative integers")
        if sum(split_scans) != self.split_scans:
            raise ValueError("nested-LODO GBM base-fit split scans must sum to split_scans")
        object.__setattr__(self, "base_fit_sample_counts", sample_counts)
        object.__setattr__(self, "base_fit_feature_counts", feature_counts)
        object.__setattr__(self, "base_fit_split_scans", split_scans)


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
    if target_count > MAX_PREDICTOR_MODELS:
        raise ValueError(
            "GBM target catalogue exceeds the predictor limit "
            f"({target_count:,} > {MAX_PREDICTOR_MODELS:,})"
        )
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


def _normalized_gbm_config(config: GbmTrainingConfig | None) -> GbmTrainingConfig:
    if config is None:
        return GbmTrainingConfig()
    if type(config) is not GbmTrainingConfig:
        raise TypeError("config must be an exact GbmTrainingConfig or None")
    return config


def estimate_nested_lodo_gbm_work(
    examples: Sequence[EvaluationExample],
    *,
    config: GbmTrainingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> GbmNestedLodoWorkEstimate:
    """Enumerate the exact base fits in the benchmark's nested LODO call graph.

    For every outer fold, lambda tuning cross-fits one calibrated predictor per
    remaining domain and then fits one final calibrated predictor. Every calibrated
    predictor in turn cross-fits one base GBM per remaining training domain and fits
    one final base GBM. Schema fitting reads prompts only; no feature transformation
    or embedding call occurs in this estimator.

    Individual base-fit resource contracts are checked while enumerating. Use
    :func:`preflight_nested_lodo_gbm` to additionally enforce the aggregate split-scan
    limit before either predictor family starts fitting.
    """

    normalized_config = _normalized_gbm_config(config)
    ordered = _ordered_examples(examples)
    model_ids = _model_ids(ordered)
    domains = tuple(sorted({example.domain for example in ordered}))
    if len(domains) < 4:
        raise ValueError("nested-LODO GBM evaluation requires at least four domains")

    outer_folds = leave_one_domain_out(ordered)
    calibrated_fit_count = 0
    sample_counts: list[int] = []
    feature_counts: list[int] = []
    split_scans: list[int] = []

    for outer_fold in outer_folds:
        outer_training = _ordered_examples(outer_fold.training)
        lambda_folds = leave_one_domain_out(outer_training)
        calibrated_training_sets = (
            *(inner_fold.training for inner_fold in lambda_folds),
            outer_training,
        )
        calibrated_fit_count += len(calibrated_training_sets)

        for calibrated_examples in calibrated_training_sets:
            calibrated_ordered = _ordered_examples(calibrated_examples)
            calibration_folds = leave_one_domain_out(calibrated_ordered)
            base_training_sets = (
                *(calibration_fold.training for calibration_fold in calibration_folds),
                calibrated_ordered,
            )
            for base_examples in base_training_sets:
                base_ordered = _ordered_examples(base_examples)
                encoder = PromptFeatureEncoder.fit(
                    tuple(example.prompt for example in base_ordered),
                    embedding_provider=embedding_provider,
                )
                sample_count = len(base_ordered)
                feature_count = encoder.schema.dimension
                preflight_gbm_fit(
                    sample_count=sample_count,
                    feature_count=feature_count,
                    target_count=len(model_ids),
                    config=normalized_config,
                )
                estimated_scans = _estimated_split_scans(
                    sample_count=sample_count,
                    feature_count=feature_count,
                    target_count=len(model_ids),
                    config=normalized_config,
                )
                sample_counts.append(sample_count)
                feature_counts.append(feature_count)
                split_scans.append(estimated_scans)

    return GbmNestedLodoWorkEstimate(
        domain_count=len(domains),
        outer_fold_count=len(outer_folds),
        calibrated_fit_count=calibrated_fit_count,
        base_fit_count=len(sample_counts),
        model_count=len(model_ids),
        estimator_count=normalized_config.n_estimators,
        split_scans=sum(split_scans),
        base_fit_sample_counts=tuple(sample_counts),
        base_fit_feature_counts=tuple(feature_counts),
        base_fit_split_scans=tuple(split_scans),
    )


def preflight_nested_lodo_gbm(
    examples: Sequence[EvaluationExample],
    *,
    config: GbmTrainingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> GbmNestedLodoWorkEstimate:
    """Reject aggregate nested-LODO GBM work before any embedding or model fit."""

    estimate = estimate_nested_lodo_gbm_work(
        examples,
        config=config,
        embedding_provider=embedding_provider,
    )
    if estimate.split_scans > MAX_GBM_SPLIT_SCANS:
        raise ValueError(
            "nested-LODO dependency-free GBM split scan exceeds the reviewed limit "
            f"({estimate.split_scans:,} > {MAX_GBM_SPLIT_SCANS:,})"
        )
    return estimate


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
    models = _fit_gradient_boosted_stumps(
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
        calibrators=MappingProxyType(calibrators),
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
