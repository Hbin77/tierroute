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
from tierroute.features import (
    EmbeddingProvider,
    PromptFeatureEncoder,
    extract_surface_features,
)
from tierroute.predictors._targets import targets_by_model
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
MAX_GBM_NESTED_BASE_FITS = 4_096
MAX_GBM_NESTED_PROMPT_BYTE_VISITS = 64 * 1024 * 1024
# Every non-empty prompt receives at least the fallback ``general`` domain tag,
# so the fitted surface schema always contains 3 continuous + 2 binary + 1 tag.
_MIN_GBM_SURFACE_FEATURE_COUNT = 6


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
    """Immutable deterministic work estimate for one nested-LODO GBM benchmark.

    Base-fit counts, row visits, and fitted feature widths are exact for the call
    graph. ``split_scans`` conservatively charges every requested boosting round even
    when fitting may stop early; ``minimum_split_scans`` is its analytic lower bound.
    The per-fit tuples follow execution order and make the aggregate auditable. An
    offline embedding provider contributes its declared width but is never invoked.
    """

    domain_count: int
    outer_fold_count: int
    calibrated_fit_count: int
    base_fit_count: int
    model_count: int
    estimator_count: int
    training_row_visits: int
    prompt_byte_visits: int
    minimum_split_scans: int
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
            self.training_row_visits,
        )
        if any(type(value) is not int or value < 1 for value in scalar_counts):
            raise ValueError("nested-LODO GBM work counts must be positive integers")
        if type(self.split_scans) is not int or self.split_scans < 0:
            raise ValueError("nested-LODO GBM split_scans must be a non-negative integer")
        for name in ("prompt_byte_visits", "minimum_split_scans"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"nested-LODO GBM {name} must be a non-negative integer")
        if self.base_fit_count > self.training_row_visits:
            raise ValueError("base fits cannot exceed nested training-row visits")
        if self.minimum_split_scans > self.split_scans:
            raise ValueError("minimum split scans cannot exceed aggregate split scans")
        sample_counts = tuple(self.base_fit_sample_counts)
        feature_counts = tuple(self.base_fit_feature_counts)
        split_scans = tuple(self.base_fit_split_scans)
        if not (
            len(sample_counts) == len(feature_counts) == len(split_scans) == self.base_fit_count
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
    if type(config) is not GbmTrainingConfig:
        raise TypeError("config must be an exact GbmTrainingConfig")
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


def _embedding_dimension(provider: EmbeddingProvider | None) -> int:
    if provider is None:
        return 0
    dimension = provider.dimension
    if type(dimension) is not int or dimension < 1:
        raise ValueError("embedding provider dimension must be a positive integer")
    return dimension


def _nested_lodo_analytic_work(
    ordered: tuple[EvaluationExample, ...],
    *,
    domain_count: int,
    model_count: int,
    config: GbmTrainingConfig,
    embedding_dimension: int,
) -> tuple[int, int, int, int]:
    """Compute cheap exact memberships and a safe scan lower bound.

    Every row participates in the same number of base-training subsets even when
    domains are imbalanced. These checks run before fold construction, preventing
    the safety check itself from expanding into attacker-controlled ``O(D^3)`` work.
    """

    base_fit_count = domain_count * ((domain_count - 1) ** 2 + domain_count)
    if base_fit_count > MAX_GBM_NESTED_BASE_FITS:
        raise ValueError(
            "nested-LODO dependency-free GBM base-fit graph exceeds the reviewed limit "
            f"({base_fit_count:,} > {MAX_GBM_NESTED_BASE_FITS:,})"
        )
    membership_multiplier = (domain_count - 1) * (domain_count**2 - 3 * domain_count + 3)
    training_row_visits = len(ordered) * membership_multiplier
    split_positions = training_row_visits - base_fit_count
    minimum_feature_count = _MIN_GBM_SURFACE_FEATURE_COUNT + embedding_dimension
    minimum_split_scans = (
        model_count * config.n_estimators * minimum_feature_count * split_positions
    )
    if minimum_split_scans > MAX_GBM_SPLIT_SCANS:
        raise ValueError(
            "nested-LODO dependency-free GBM minimum split scan exceeds the reviewed limit "
            f"({minimum_split_scans:,} > {MAX_GBM_SPLIT_SCANS:,})"
        )
    prompt_bytes = 0
    for example in ordered:
        try:
            prompt_bytes += len(example.prompt.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError("GBM prompts must contain valid Unicode") from error
    prompt_byte_visits = prompt_bytes * membership_multiplier
    if prompt_byte_visits > MAX_GBM_NESTED_PROMPT_BYTE_VISITS:
        raise ValueError(
            "nested-LODO dependency-free GBM prompt scan exceeds the reviewed limit "
            f"({prompt_byte_visits:,} > {MAX_GBM_NESTED_PROMPT_BYTE_VISITS:,} bytes)"
        )
    return base_fit_count, training_row_visits, prompt_byte_visits, minimum_split_scans


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
    one final base GBM. Surface tags are extracted once per prompt, and no feature
    transformation or embedding call occurs in this estimator.

    Closed-form graph, minimum-scan, and repeated-prompt-byte guards run before folds
    or tags are materialized. Individual and running aggregate resource contracts are
    then checked while enumerating.
    """

    normalized_config = _normalized_gbm_config(config)
    ordered = _ordered_examples(examples)
    model_ids = _model_ids(ordered)
    domains = tuple(sorted({example.domain for example in ordered}))
    if len(domains) < 4:
        raise ValueError("nested-LODO GBM evaluation requires at least four domains")
    embedding_dimension = _embedding_dimension(embedding_provider)
    (
        expected_base_fit_count,
        training_row_visits,
        prompt_byte_visits,
        minimum_split_scans,
    ) = _nested_lodo_analytic_work(
        ordered,
        domain_count=len(domains),
        model_count=len(model_ids),
        config=normalized_config,
        embedding_dimension=embedding_dimension,
    )
    tags_by_example_id = {
        example.example_id: frozenset(extract_surface_features(example.prompt).domain_tags)
        for example in ordered
    }

    outer_folds = leave_one_domain_out(ordered)
    calibrated_fit_count = 0
    sample_counts: list[int] = []
    feature_counts: list[int] = []
    split_scans: list[int] = []
    running_split_scans = 0

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
                sample_count = len(base_ordered)
                domain_tags = set().union(
                    *(tags_by_example_id[example.example_id] for example in base_ordered)
                )
                feature_count = 5 + len(domain_tags) + embedding_dimension
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
                running_split_scans += estimated_scans
                if running_split_scans > MAX_GBM_SPLIT_SCANS:
                    raise ValueError(
                        "nested-LODO dependency-free GBM split scan exceeds the reviewed limit "
                        f"({running_split_scans:,} > {MAX_GBM_SPLIT_SCANS:,})"
                    )

    if len(sample_counts) != expected_base_fit_count:
        raise AssertionError("nested-LODO GBM call graph differs from its analytic count")
    if sum(sample_counts) != training_row_visits:
        raise AssertionError("nested-LODO GBM row visits differ from their analytic count")

    return GbmNestedLodoWorkEstimate(
        domain_count=len(domains),
        outer_fold_count=len(outer_folds),
        calibrated_fit_count=calibrated_fit_count,
        base_fit_count=len(sample_counts),
        model_count=len(model_ids),
        estimator_count=normalized_config.n_estimators,
        training_row_visits=training_row_visits,
        prompt_byte_visits=prompt_byte_visits,
        minimum_split_scans=minimum_split_scans,
        split_scans=running_split_scans,
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
    target_columns = targets_by_model(ordered, model_ids)
    models = _fit_gradient_boosted_stumps(
        feature_rows,
        target_columns,
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

    config = _normalized_gbm_config(config)
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
