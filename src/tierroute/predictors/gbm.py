# SPDX-License-Identifier: Apache-2.0
"""Deterministic, dependency-free gradient-boosted regression stumps."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from tierroute.predictors.resource_limits import MAX_PREDICTOR_MODELS

MAX_GBM_STUMPS_PER_MODEL = 256


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _model_ids(model_ids: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(model_ids)
    if not requested:
        raise ValueError("model_ids must not be empty")
    if any(not isinstance(model_id, str) or not model_id for model_id in requested):
        raise ValueError("model_ids must contain non-empty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("model_ids must be unique")
    return requested


@dataclass(frozen=True, slots=True)
class RegressionStump:
    """One binary regression tree using an observed value as its split boundary.

    Training stores the smallest feature value assigned to the right leaf.  The
    matching inference rule is therefore ``x < split_value`` for the left leaf and
    ``x >= split_value`` for the right leaf.  Avoiding an arithmetic midpoint also
    avoids overflow for finite features near the limits of binary64.
    """

    feature_index: int
    split_value: float
    left_value: float
    right_value: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.feature_index, bool)
            or not isinstance(self.feature_index, int)
            or self.feature_index < 0
        ):
            raise ValueError("feature_index must be a non-negative integer")
        for name in ("split_value", "left_value", "right_value"):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), label=name),
            )

    def predict_features(self, features: Sequence[float]) -> float:
        """Return the leaf value for one already-vectorized prompt."""

        if self.feature_index >= len(features):
            raise ValueError(
                f"feature width {len(features)} does not include index {self.feature_index}"
            )
        value = _finite_float(
            features[self.feature_index],
            label=f"feature[{self.feature_index}]",
        )
        return self.left_value if value < self.split_value else self.right_value


@dataclass(frozen=True, slots=True)
class GbmModel:
    """A single model head made of squared-error regression stumps."""

    feature_width: int
    base_value: float
    learning_rate: float
    stumps: tuple[RegressionStump, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_width",
            _positive_integer(self.feature_width, label="feature_width"),
        )
        object.__setattr__(
            self,
            "base_value",
            _finite_float(self.base_value, label="base_value"),
        )
        learning_rate = _finite_float(self.learning_rate, label="learning_rate")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        object.__setattr__(self, "learning_rate", learning_rate)
        stumps: list[RegressionStump] = []
        try:
            for stump in self.stumps:
                if len(stumps) >= MAX_GBM_STUMPS_PER_MODEL:
                    raise ValueError(
                        f"stumps exceed the per-model limit ({MAX_GBM_STUMPS_PER_MODEL:,})"
                    )
                if not isinstance(stump, RegressionStump):
                    raise TypeError("stumps must contain only RegressionStump values")
                stumps.append(stump)
        except RuntimeError as error:
            raise ValueError("stumps could not be read deterministically") from error
        if any(stump.feature_index >= self.feature_width for stump in stumps):
            raise ValueError("a stump feature index exceeds feature_width")
        object.__setattr__(self, "stumps", tuple(stumps))

    def predict_features(self, features: Sequence[float]) -> float:
        """Score one finite feature row."""

        row = tuple(
            _finite_float(value, label=f"feature[{index}]") for index, value in enumerate(features)
        )
        if len(row) != self.feature_width:
            raise ValueError(
                f"feature width {len(row)} does not match expected width {self.feature_width}"
            )
        try:
            score = math.fsum(
                (
                    self.base_value,
                    *(self.learning_rate * stump.predict_features(row) for stump in self.stumps),
                )
            )
        except OverflowError as error:
            raise ValueError("GBM prediction overflowed") from error
        if not math.isfinite(score):
            raise ValueError("GBM prediction must be finite")
        return score


@dataclass(frozen=True, slots=True)
class GbmQualityPredictor:
    """Expose model-specific GBM heads through the quality-predictor protocols."""

    vectorizer: Callable[[str], Sequence[float]] = field(compare=False, repr=False)
    models: Mapping[str, GbmModel]
    batch_vectorizer: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not callable(self.vectorizer):
            raise TypeError("vectorizer must be callable")
        if self.batch_vectorizer is not None and not callable(self.batch_vectorizer):
            raise TypeError("batch_vectorizer must be callable or None")
        if not isinstance(self.models, Mapping):
            raise ValueError("models must be a non-empty mapping")
        snapshot: dict[str, GbmModel] = {}
        try:
            for model_id, model in self.models.items():
                if len(snapshot) >= MAX_PREDICTOR_MODELS:
                    raise ValueError(
                        f"models exceed the predictor limit ({MAX_PREDICTOR_MODELS:,})"
                    )
                if not isinstance(model_id, str) or not model_id:
                    raise ValueError("model IDs must be non-empty strings")
                if model_id in snapshot:
                    raise ValueError("model IDs must be unique")
                if not isinstance(model, GbmModel):
                    raise TypeError("models must map IDs to GbmModel values")
                snapshot[model_id] = model
        except RuntimeError as error:
            raise ValueError("models could not be read deterministically") from error
        if not snapshot:
            raise ValueError("models must be a non-empty mapping")
        canonical = {model_id: snapshot[model_id] for model_id in sorted(snapshot)}
        widths = {model.feature_width for model in canonical.values()}
        if len(widths) != 1:
            raise ValueError("all GBM model heads must use the same feature width")
        object.__setattr__(self, "models", MappingProxyType(canonical))

    @property
    def feature_width(self) -> int:
        """Return the shared feature width of all model heads."""

        return next(iter(self.models.values())).feature_width

    def _row(self, values: Sequence[float]) -> tuple[float, ...]:
        row = tuple(
            _finite_float(value, label=f"feature[{index}]") for index, value in enumerate(values)
        )
        if len(row) != self.feature_width:
            raise ValueError(
                f"feature width {len(row)} does not match expected width {self.feature_width}"
            )
        return row

    def _score(self, row: tuple[float, ...], model_id: str) -> float:
        try:
            model = self.models[model_id]
        except KeyError as error:
            raise KeyError(f"no GBM head for model {model_id!r}") from error
        return model.predict_features(row)

    def predict(self, prompt: str, model_id: str) -> float:
        row = self._row(self.vectorizer(prompt))
        return self._score(row, model_id)

    def predict_many(self, prompt: str, model_ids: Sequence[str]) -> Mapping[str, float]:
        """Score requested models after vectorizing a prompt exactly once."""

        requested = _model_ids(model_ids)
        row = self._row(self.vectorizer(prompt))
        return {model_id: self._score(row, model_id) for model_id in requested}

    def predict_batch(
        self,
        prompts: Sequence[str],
        model_ids: Sequence[str],
    ) -> tuple[Mapping[str, float], ...]:
        """Vectorize a prompt batch once when a batch vectorizer is available."""

        prompts = tuple(prompts)
        if not prompts:
            raise ValueError("prompts must not be empty")
        requested = _model_ids(model_ids)
        if self.batch_vectorizer is None:
            raw_rows = tuple(self.vectorizer(prompt) for prompt in prompts)
        else:
            raw_rows = tuple(self.batch_vectorizer(prompts))
        if len(raw_rows) != len(prompts):
            raise ValueError("batch vectorizer returned the wrong number of rows")
        rows = tuple(self._row(values) for values in raw_rows)
        return tuple(
            {model_id: self._score(row, model_id) for model_id in requested} for row in rows
        )


@dataclass(frozen=True, slots=True)
class _SplitCandidate:
    feature_index: int
    split_value: float
    gain: float


def _squared_loss(residuals: Sequence[float]) -> float:
    try:
        loss = math.fsum(residual * residual for residual in residuals)
    except OverflowError as error:
        raise ValueError("GBM squared loss overflowed") from error
    if not math.isfinite(loss):
        raise ValueError("GBM squared loss must be finite")
    return loss


def _best_split(
    feature_rows: tuple[tuple[float, ...], ...],
    residuals: tuple[float, ...],
    sorted_indices: tuple[tuple[int, ...], ...],
    *,
    min_samples_leaf: int,
) -> _SplitCandidate | None:
    sample_count = len(feature_rows)
    if sample_count < 2 * min_samples_leaf:
        return None
    total = math.fsum(residuals)
    best: _SplitCandidate | None = None

    # Features and observed split values are scanned in ascending order.  Keeping
    # the first equal-gain candidate makes the tie rule explicit and deterministic.
    for feature_index, order in enumerate(sorted_indices):
        left_sum = 0.0
        compensation = 0.0
        for right_start in range(1, sample_count):
            residual = residuals[order[right_start - 1]]
            corrected = residual - compensation
            updated = left_sum + corrected
            compensation = (updated - left_sum) - corrected
            left_sum = updated

            left_feature = feature_rows[order[right_start - 1]][feature_index]
            split_value = feature_rows[order[right_start]][feature_index]
            if left_feature == split_value:
                continue
            right_count = sample_count - right_start
            if right_start < min_samples_leaf or right_count < min_samples_leaf:
                continue
            right_sum = math.fsum((total, -left_sum))
            try:
                gain = math.fsum(
                    (
                        left_sum * left_sum / right_start,
                        right_sum * right_sum / right_count,
                        -(total * total / sample_count),
                    )
                )
            except OverflowError as error:
                raise ValueError("GBM split gain overflowed") from error
            if not math.isfinite(gain):
                raise ValueError("GBM split gain must be finite")
            candidate = _SplitCandidate(feature_index, split_value, max(gain, 0.0))
            if best is None or candidate.gain > best.gain:
                best = candidate
    return best


def _fit_model(
    feature_rows: tuple[tuple[float, ...], ...],
    targets: tuple[float, ...],
    sorted_indices: tuple[tuple[int, ...], ...],
    *,
    n_estimators: int,
    learning_rate: float,
    min_samples_leaf: int,
    min_gain: float,
) -> GbmModel:
    try:
        base_value = math.fsum(targets) / len(targets)
    except OverflowError as error:
        raise ValueError("GBM base value overflowed") from error
    if not math.isfinite(base_value):
        raise ValueError("GBM base value must be finite")
    residuals = tuple(target - base_value for target in targets)
    previous_loss = _squared_loss(residuals)
    stumps: list[RegressionStump] = []

    for _ in range(n_estimators):
        candidate = _best_split(
            feature_rows,
            residuals,
            sorted_indices,
            min_samples_leaf=min_samples_leaf,
        )
        if candidate is None or candidate.gain <= min_gain:
            break
        left_indices = tuple(
            index
            for index, row in enumerate(feature_rows)
            if row[candidate.feature_index] < candidate.split_value
        )
        right_indices = tuple(
            index
            for index, row in enumerate(feature_rows)
            if row[candidate.feature_index] >= candidate.split_value
        )
        try:
            left_value = math.fsum(residuals[index] for index in left_indices) / len(left_indices)
            right_value = math.fsum(residuals[index] for index in right_indices) / len(
                right_indices
            )
        except OverflowError as error:
            raise ValueError("GBM leaf value overflowed") from error
        stump = RegressionStump(
            feature_index=candidate.feature_index,
            split_value=candidate.split_value,
            left_value=left_value,
            right_value=right_value,
        )
        next_residuals = tuple(
            residual - learning_rate * stump.predict_features(row)
            for residual, row in zip(residuals, feature_rows, strict=True)
        )
        next_loss = _squared_loss(next_residuals)
        if next_loss >= previous_loss:
            break
        stumps.append(stump)
        residuals = next_residuals
        previous_loss = next_loss

    return GbmModel(
        feature_width=len(feature_rows[0]),
        base_value=base_value,
        learning_rate=learning_rate,
        stumps=tuple(stumps),
    )


def _fit_gradient_boosted_stumps(
    feature_rows: Sequence[Sequence[float]],
    targets_by_model: Mapping[str, Sequence[float]],
    *,
    example_ids: Sequence[str],
    n_estimators: int,
    learning_rate: float,
    min_samples_leaf: int,
    min_gain: float,
) -> dict[str, GbmModel]:
    """Fit deterministic per-model squared-error GBM heads on preflighted input.

    Examples are canonically ordered by their unique IDs before any floating-point
    reduction.  Every boosting round fits a regression stump to the current
    residuals.  ``min_gain`` is the minimum unshrunk squared-error reduction; equal
    gains choose the lower feature index and then the lower observed split value. This
    is a private raw primitive; the public trainer owns resource preflight.
    """

    n_estimators = _positive_integer(n_estimators, label="n_estimators")
    if n_estimators > MAX_GBM_STUMPS_PER_MODEL:
        raise ValueError(f"n_estimators exceeds the reviewed limit ({MAX_GBM_STUMPS_PER_MODEL:,})")
    min_samples_leaf = _positive_integer(min_samples_leaf, label="min_samples_leaf")
    learning_rate = _finite_float(learning_rate, label="learning_rate")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    min_gain = _finite_float(min_gain, label="min_gain")
    if min_gain < 0.0:
        raise ValueError("min_gain must be non-negative")

    ids = tuple(example_ids)
    if not ids:
        raise ValueError("example_ids must not be empty")
    if any(not isinstance(example_id, str) or not example_id for example_id in ids):
        raise ValueError("example_ids must contain non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("example_ids must be unique")

    raw_rows = tuple(feature_rows)
    if len(raw_rows) != len(ids):
        raise ValueError("feature row count must match example_ids")
    rows = tuple(
        tuple(
            _finite_float(value, label=f"feature_rows[{row_index}][{feature_index}]")
            for feature_index, value in enumerate(row)
        )
        for row_index, row in enumerate(raw_rows)
    )
    feature_width = len(rows[0])
    if feature_width < 1:
        raise ValueError("feature rows must not be empty")
    if any(len(row) != feature_width for row in rows):
        raise ValueError("feature rows must have a consistent width")

    if not isinstance(targets_by_model, Mapping) or not targets_by_model:
        raise ValueError("targets_by_model must be a non-empty mapping")
    if any(not isinstance(model_id, str) or not model_id for model_id in targets_by_model):
        raise ValueError("target model IDs must be non-empty strings")
    raw_targets: dict[str, tuple[float, ...]] = {}
    for model_id in sorted(targets_by_model):
        values = tuple(
            _finite_float(value, label=f"targets_by_model[{model_id!r}][{index}]")
            for index, value in enumerate(targets_by_model[model_id])
        )
        if len(values) != len(rows):
            raise ValueError(f"target count for {model_id!r} must match feature rows")
        raw_targets[model_id] = values

    canonical_order = tuple(sorted(range(len(ids)), key=lambda index: ids[index]))
    ordered_ids = tuple(ids[index] for index in canonical_order)
    ordered_rows = tuple(rows[index] for index in canonical_order)
    ordered_targets = {
        model_id: tuple(values[index] for index in canonical_order)
        for model_id, values in raw_targets.items()
    }
    sorted_indices = tuple(
        tuple(
            sorted(
                range(len(ordered_rows)),
                key=lambda index: (ordered_rows[index][feature_index], ordered_ids[index]),
            )
        )
        for feature_index in range(feature_width)
    )
    return {
        model_id: _fit_model(
            ordered_rows,
            targets,
            sorted_indices,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            min_gain=min_gain,
        )
        for model_id, targets in ordered_targets.items()
    }
