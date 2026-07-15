# SPDX-License-Identifier: Apache-2.0
"""Dependency-free reference solver for centered ridge regression.

This module deliberately favors a small, auditable implementation over a
high-dimensional numerical backend.  It is suitable for deterministic tests
and modest feature matrices; callers with thousands of dense features should
use a separately reviewed accelerated backend.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

CENTERED_RIDGE_SOLVER_ID = "tierroute.centered-ridge-cholesky-python-v1"
_MAX_REFERENCE_MULTIPLY_ACCUMULATIONS = 100_000_000


@dataclass(frozen=True, slots=True)
class RidgeSolution:
    """Coefficients for target columns in the order supplied to the solver."""

    weights: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]


def fit_centered_ridge(
    feature_rows: Sequence[Sequence[float]],
    targets_by_model: Mapping[str, Sequence[float]],
    *,
    ridge: float,
) -> dict[str, tuple[tuple[float, ...], float]]:
    """Fit all model targets and return ``model_id: (weights, intercept)``.

    Model IDs are sorted before solving so a mapping's insertion order cannot
    affect coefficient arithmetic or artifact serialization.
    """

    if not targets_by_model:
        raise ValueError("targets_by_model must not be empty")
    if any(not isinstance(model_id, str) or not model_id.strip() for model_id in targets_by_model):
        raise TypeError("targets_by_model keys must be non-empty strings")
    model_ids = tuple(sorted(targets_by_model))
    solution = solve_centered_ridge(
        feature_rows,
        tuple(targets_by_model[model_id] for model_id in model_ids),
        ridge=ridge,
    )
    return {
        model_id: (weights, intercept)
        for model_id, weights, intercept in zip(
            model_ids,
            solution.weights,
            solution.intercepts,
            strict=True,
        )
    }


def solve_centered_ridge(
    feature_rows: Sequence[Sequence[float]],
    target_columns: Sequence[Sequence[float]],
    *,
    ridge: float,
) -> RidgeSolution:
    """Fit ridge coefficients with an unregularized intercept.

    ``feature_rows`` has shape ``(sample_count, feature_count)`` while
    ``target_columns`` has shape ``(target_count, sample_count)``.  The solver
    centers both matrices, solves ``(XᵀX + ridge I) W = XᵀY``, and recovers
    each intercept from the original means.  Centering is what leaves the
    intercept unregularized without adding a singular-looking intercept column
    to the normal equations.

    The Gram matrix and its SPD Cholesky factor are shared by all targets.  All
    reductions use :func:`math.fsum` so results are deterministic for a fixed
    input order and less sensitive to ordinary summation error.
    """

    ridge_value = _finite_real(ridge, location="ridge")
    if ridge_value <= 0.0:
        raise ValueError("ridge must be positive")

    features = _coerce_rectangular(feature_rows, name="feature_rows")
    sample_count = len(features)
    feature_count = len(features[0])
    if feature_count == 0:
        raise ValueError("feature_rows must contain at least one feature")
    targets = _coerce_rectangular(
        target_columns,
        name="target_columns",
        expected_width=sample_count,
    )
    work_estimate = _multiply_accumulation_estimate(
        sample_count,
        feature_count,
        len(targets),
    )
    if work_estimate > _MAX_REFERENCE_MULTIPLY_ACCUMULATIONS:
        raise ValueError(
            "reference ridge work estimate "
            f"{work_estimate:,} exceeds the audited limit "
            f"{_MAX_REFERENCE_MULTIPLY_ACCUMULATIONS:,}; "
            "use a separately reviewed accelerated backend with parity tests"
        )

    feature_means = tuple(
        _mean((row[index] for row in features), sample_count) for index in range(feature_count)
    )
    target_means = tuple(_mean(column, sample_count) for column in targets)
    centered_features = tuple(
        tuple(
            _finite_result(value - feature_means[index], operation="feature centering")
            for index, value in enumerate(row)
        )
        for row in features
    )
    centered_targets = tuple(
        tuple(_finite_result(value - target_mean, operation="target centering") for value in column)
        for column, target_mean in zip(targets, target_means, strict=True)
    )

    weights = _solve_primal(centered_features, centered_targets, ridge_value)

    intercepts = []
    for target_weights, target_mean in zip(
        weights,
        target_means,
        strict=True,
    ):
        intercept = _finite_result(
            target_mean
            - _checked_fsum(
                (
                    _finite_result(
                        mean * weight,
                        operation="intercept product",
                    )
                    for mean, weight in zip(
                        feature_means,
                        target_weights,
                        strict=True,
                    )
                ),
                operation="intercept accumulation",
            ),
            operation="intercept recovery",
        )
        intercepts.append(intercept)

    return RidgeSolution(weights, tuple(intercepts))


def _multiply_accumulation_estimate(
    sample_count: int,
    feature_count: int,
    target_count: int,
) -> int:
    """Bound the dominant products before allocating the Gram matrix.

    The count covers the symmetric Gram matrix, Cholesky factorization, all
    target right-hand sides, and residual verification. It is a conservative
    operation guard, not a wall-clock benchmark or a promise of performance.
    """

    gram = sample_count * feature_count * (feature_count + 1) // 2
    cholesky = feature_count * (feature_count - 1) * (feature_count + 1) // 6
    right_hand_sides = target_count * sample_count * feature_count
    residuals = target_count * feature_count * feature_count
    return gram + cholesky + right_hand_sides + residuals


def _coerce_rectangular(
    rows: Sequence[Sequence[float]],
    *,
    name: str,
    expected_width: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    try:
        materialized_rows = tuple(rows)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of numeric sequences") from error
    if not materialized_rows:
        raise ValueError(f"{name} must not be empty")

    converted = []
    width = expected_width
    for row_index, row in enumerate(materialized_rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"{name}[{row_index}] must be a numeric sequence")
        try:
            values = tuple(row)
        except TypeError as error:
            raise TypeError(f"{name}[{row_index}] must be a numeric sequence") from error
        if width is None:
            width = len(values)
        if len(values) != width:
            raise ValueError(f"{name} must be rectangular with width {width}")
        converted.append(
            tuple(
                _finite_real(value, location=f"{name}[{row_index}][{column_index}]")
                for column_index, value in enumerate(values)
            )
        )
    return tuple(converted)


def _finite_real(value: object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{location} must be a real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{location} must be representable as a finite float") from error
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be finite")
    return converted


def _mean(values: Iterable[float], count: int) -> float:
    # Dividing before fsum avoids overflowing a representable mean when many
    # same-sign values are close to the largest finite float.
    return _checked_fsum(
        (value / count for value in values),
        operation="mean accumulation",
    )


def _normal_matrix(
    centered_features: tuple[tuple[float, ...], ...],
    feature_count: int,
    ridge: float,
) -> tuple[tuple[float, ...], ...]:
    matrix = [[0.0] * feature_count for _ in range(feature_count)]
    for row_index in range(feature_count):
        for column_index in range(row_index + 1):
            value = _checked_fsum(
                (
                    _finite_result(
                        row[row_index] * row[column_index],
                        operation="Gram product",
                    )
                    for row in centered_features
                ),
                operation="Gram accumulation",
            )
            if row_index == column_index:
                value = _finite_result(value + ridge, operation="ridge regularization")
            matrix[row_index][column_index] = value
            matrix[column_index][row_index] = value
    return tuple(tuple(row) for row in matrix)


def _solve_primal(
    centered_features: tuple[tuple[float, ...], ...],
    centered_targets: tuple[tuple[float, ...], ...],
    ridge: float,
) -> tuple[tuple[float, ...], ...]:
    feature_count = len(centered_features[0])
    normal_matrix = _normal_matrix(centered_features, feature_count, ridge)
    factor = _cholesky(normal_matrix)
    solutions = []
    for centered_target in centered_targets:
        right_hand_side = tuple(
            _checked_fsum(
                (
                    _finite_result(
                        row[feature_index] * target,
                        operation="right-hand-side product",
                    )
                    for row, target in zip(
                        centered_features,
                        centered_target,
                        strict=True,
                    )
                ),
                operation="right-hand-side accumulation",
            )
            for feature_index in range(feature_count)
        )
        weights = _solve_cholesky(factor, right_hand_side)
        _verify_residual(normal_matrix, weights, right_hand_side)
        solutions.append(weights)
    return tuple(solutions)


def _cholesky(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Return the lower-triangular Cholesky factor of an SPD matrix."""

    dimension = len(matrix)
    factor = [[0.0] * dimension for _ in range(dimension)]
    for row_index in range(dimension):
        for column_index in range(row_index + 1):
            correction = _checked_fsum(
                (
                    _finite_result(
                        factor[row_index][inner] * factor[column_index][inner],
                        operation="Cholesky product",
                    )
                    for inner in range(column_index)
                ),
                operation="Cholesky accumulation",
            )
            remainder = _finite_result(
                matrix[row_index][column_index] - correction,
                operation="Cholesky subtraction",
            )
            if row_index == column_index:
                if remainder <= 0.0:
                    raise ArithmeticError(
                        f"ridge normal matrix lost positive definiteness at diagonal {row_index}"
                    )
                factor[row_index][column_index] = math.sqrt(remainder)
            else:
                factor[row_index][column_index] = _finite_result(
                    remainder / factor[column_index][column_index],
                    operation="Cholesky division",
                )
    return tuple(tuple(row) for row in factor)


def _solve_cholesky(
    factor: tuple[tuple[float, ...], ...],
    right_hand_side: tuple[float, ...],
) -> tuple[float, ...]:
    dimension = len(factor)
    forward = [0.0] * dimension
    for row_index in range(dimension):
        correction = _checked_fsum(
            (
                _finite_result(
                    factor[row_index][column_index] * forward[column_index],
                    operation="forward-substitution product",
                )
                for column_index in range(row_index)
            ),
            operation="forward-substitution accumulation",
        )
        forward[row_index] = _finite_result(
            (right_hand_side[row_index] - correction) / factor[row_index][row_index],
            operation="forward substitution",
        )

    solution = [0.0] * dimension
    for row_index in range(dimension - 1, -1, -1):
        correction = _checked_fsum(
            (
                _finite_result(
                    factor[column_index][row_index] * solution[column_index],
                    operation="back-substitution product",
                )
                for column_index in range(row_index + 1, dimension)
            ),
            operation="back-substitution accumulation",
        )
        solution[row_index] = _finite_result(
            (forward[row_index] - correction) / factor[row_index][row_index],
            operation="back substitution",
        )
    return tuple(solution)


def _verify_residual(
    matrix: tuple[tuple[float, ...], ...],
    solution: tuple[float, ...],
    right_hand_side: tuple[float, ...],
) -> None:
    """Reject a solve whose normal-equation backward residual is implausible."""

    dimension = len(matrix)
    for row_index, row in enumerate(matrix):
        products = tuple(
            _finite_result(value * coefficient, operation="residual product")
            for value, coefficient in zip(row, solution, strict=True)
        )
        reconstructed = _checked_fsum(products, operation="residual accumulation")
        residual = abs(
            _finite_result(
                reconstructed - right_hand_side[row_index],
                operation="residual subtraction",
            )
        )
        scale = _checked_fsum(
            (abs(value) for value in (*products, right_hand_side[row_index])),
            operation="residual scale",
        )
        # A dimension-scaled backward-error threshold catches corrupted solves
        # while allowing ordinary roundoff from Cholesky and triangular solves.
        tolerance = _finite_result(
            512.0 * max(1, dimension) * max(sys.float_info.epsilon * scale, math.ulp(scale)),
            operation="residual tolerance",
        )
        if residual > tolerance:
            raise ArithmeticError(
                "ridge solve failed residual verification at equation "
                f"{row_index}: residual={residual!r}, tolerance={tolerance!r}"
            )


def _checked_fsum(values: Iterable[float], *, operation: str) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ArithmeticError(f"{operation} was not numerically finite") from error
    return _finite_result(result, operation=operation)


def _finite_result(value: float, *, operation: str) -> float:
    if not math.isfinite(value):
        raise ArithmeticError(f"{operation} produced a non-finite value")
    return value
