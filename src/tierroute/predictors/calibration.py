# SPDX-License-Identifier: Apache-2.0
"""A small pool-adjacent-violators isotonic calibration layer."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby, pairwise

from tierroute.predictors.base import (
    BatchPromptQualityPredictor,
    BatchQualityPredictor,
    QualityPredictor,
)
from tierroute.predictors.resource_limits import MAX_PREDICTOR_CALIBRATOR_POINTS


def _normalized_calibration_sequence(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context} must be a list or tuple")
    result: list[float] = []
    try:
        iterator = iter(value)
        for item in iterator:
            if len(result) >= MAX_PREDICTOR_CALIBRATOR_POINTS:
                raise ValueError(
                    f"{context} exceeds the calibration limit ({MAX_PREDICTOR_CALIBRATOR_POINTS:,})"
                )
            if type(item) not in (int, float):
                raise ValueError("calibration parameters must be finite")
            try:
                number = float(item)
            except (OverflowError, ValueError) as error:
                raise ValueError("calibration parameters must be finite") from error
            if not math.isfinite(number):
                raise ValueError("calibration parameters must be finite")
            result.append(number)
    except RuntimeError as error:
        raise ValueError(f"{context} could not be read deterministically") from error
    return tuple(result)


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Monotone step calibration fitted without external dependencies."""

    upper_bounds: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        upper_bounds = _normalized_calibration_sequence(self.upper_bounds, "upper_bounds")
        values = _normalized_calibration_sequence(self.values, "values")
        if not upper_bounds or len(upper_bounds) != len(values):
            raise ValueError("upper_bounds and values must be non-empty and equally sized")
        if any(left >= right for left, right in pairwise(upper_bounds)):
            raise ValueError("upper_bounds must be strictly increasing")
        if any(left > right for left, right in pairwise(values)):
            raise ValueError("calibrated values must be non-decreasing")
        object.__setattr__(self, "upper_bounds", upper_bounds)
        object.__setattr__(self, "values", values)

    @classmethod
    def fit(cls, predictions: list[float], targets: list[float]) -> IsotonicCalibrator:
        """Fit a monotone step function with equal sample weights."""

        if len(predictions) != len(targets) or not predictions:
            raise ValueError("predictions and targets must be non-empty and equally sized")
        pairs = sorted((float(x), float(y)) for x, y in zip(predictions, targets, strict=True))
        if any(not math.isfinite(value) for pair in pairs for value in pair):
            raise ValueError("calibration samples must be finite")

        blocks: list[list[float]] = []
        for prediction, group in groupby(pairs, key=lambda pair: pair[0]):
            grouped_targets = [target for _, target in group]
            blocks.append([prediction, sum(grouped_targets), float(len(grouped_targets))])
            while len(blocks) >= 2:
                previous = blocks[-2]
                current = blocks[-1]
                if previous[1] / previous[2] <= current[1] / current[2]:
                    break
                blocks[-2:] = [[current[0], previous[1] + current[1], previous[2] + current[2]]]

        return cls(
            upper_bounds=tuple(block[0] for block in blocks),
            values=tuple(block[1] / block[2] for block in blocks),
        )

    def calibrate(self, prediction: float) -> float:
        """Calibrate one finite score, clipping to the end blocks."""

        prediction = float(prediction)
        if not math.isfinite(prediction):
            raise ValueError("prediction must be finite")
        index = min(bisect_left(self.upper_bounds, prediction), len(self.values) - 1)
        return self.values[index]


@dataclass(frozen=True, slots=True)
class CalibratedQualityPredictor:
    """Apply a fitted calibration layer to any base predictor."""

    base: QualityPredictor
    calibrator: IsotonicCalibrator

    def predict(self, prompt: str, model_id: str) -> float:
        return self.calibrator.calibrate(self.base.predict(prompt, model_id))


@dataclass(frozen=True, slots=True)
class PerModelCalibratedQualityPredictor:
    """Apply a separately cross-fitted isotonic layer to each candidate model."""

    base: QualityPredictor
    calibrators: Mapping[str, IsotonicCalibrator]

    def predict(self, prompt: str, model_id: str) -> float:
        try:
            calibrator = self.calibrators[model_id]
        except KeyError as error:
            raise KeyError(f"no calibrator for model {model_id!r}") from error
        return calibrator.calibrate(self.base.predict(prompt, model_id))

    def predict_many(self, prompt: str, model_ids: Sequence[str]) -> Mapping[str, float]:
        """Preserve the base predictor's one-vectorization batch path."""

        model_ids = tuple(model_ids)
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError("model_ids must be non-empty and unique")
        if isinstance(self.base, BatchQualityPredictor):
            raw = self.base.predict_many(prompt, model_ids)
        else:
            raw = {model_id: self.base.predict(prompt, model_id) for model_id in model_ids}
        if set(raw) != set(model_ids):
            raise ValueError("base predictor must return every requested model exactly")
        return {
            model_id: self.calibrators[model_id].calibrate(raw[model_id]) for model_id in model_ids
        }

    def predict_batch(
        self, prompts: Sequence[str], model_ids: Sequence[str]
    ) -> tuple[Mapping[str, float], ...]:
        """Calibrate a base predictor's prompt-batch output per model."""

        prompts = tuple(prompts)
        model_ids = tuple(model_ids)
        if not prompts:
            raise ValueError("prompts must not be empty")
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError("model_ids must be non-empty and unique")
        if isinstance(self.base, BatchPromptQualityPredictor):
            raw_rows = self.base.predict_batch(prompts, model_ids)
        else:
            raw_rows = tuple(
                {model_id: self.base.predict(prompt, model_id) for model_id in model_ids}
                for prompt in prompts
            )
        if len(raw_rows) != len(prompts):
            raise ValueError("base predictor returned the wrong number of prompt rows")
        if any(set(row) != set(model_ids) for row in raw_rows):
            raise ValueError("base predictor must return every requested model exactly")
        return tuple(
            {
                model_id: self.calibrators[model_id].calibrate(row[model_id])
                for model_id in model_ids
            }
            for row in raw_rows
        )
