# SPDX-License-Identifier: Apache-2.0
"""A small pool-adjacent-violators isotonic calibration layer."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from itertools import groupby

from tierroute.predictors.base import QualityPredictor


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Monotone step calibration fitted without external dependencies."""

    upper_bounds: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.upper_bounds or len(self.upper_bounds) != len(self.values):
            raise ValueError("upper_bounds and values must be non-empty and equally sized")
        if any(not math.isfinite(value) for value in (*self.upper_bounds, *self.values)):
            raise ValueError("calibration parameters must be finite")
        if any(
            left >= right
            for left, right in zip(self.upper_bounds, self.upper_bounds[1:], strict=False)
        ):
            raise ValueError("upper_bounds must be strictly increasing")
        if any(left > right for left, right in zip(self.values, self.values[1:], strict=False)):
            raise ValueError("calibrated values must be non-decreasing")

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
