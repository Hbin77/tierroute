# SPDX-License-Identifier: Apache-2.0
"""Static, dependency-free boundary for reviewed ridge training solvers.

Artifact parsing never imports an optional numerical package. An admitted ID
identifies a reviewed algorithm and artifact provenance; it does not approve or
locate an executable. Credentialed backends must independently authenticate
their exact binary, and distributed binaries remain subject to the project's
platform-specific license, link, offline, resource, and numerical-parity gates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from tierroute.predictors._ridge import (
    CENTERED_RIDGE_SOLVER_ID,
    RidgeSolution,
    preflight_reference_ridge,
    solve_centered_ridge,
)

NATIVE_C11_RIDGE_SOLVER_ID = "tierroute.centered-ridge-cholesky-c11-v1"

KNOWN_RIDGE_SOLVER_IDS = frozenset(
    {
        CENTERED_RIDGE_SOLVER_ID,
        NATIVE_C11_RIDGE_SOLVER_ID,
    }
)


class RidgeSolver(Protocol):
    """Training-only solver contract; inference consumes stored coefficients."""

    solver_id: str

    def preflight(
        self,
        *,
        sample_count: int,
        feature_count: int,
        target_count: int,
    ) -> None:
        """Reject unsupported work before a dense feature matrix is built."""

    def solve(
        self,
        feature_rows: Sequence[Sequence[float]],
        target_columns: Sequence[Sequence[float]],
        *,
        ridge: float,
    ) -> RidgeSolution:
        """Fit all target columns with one shared matrix factorization."""


@dataclass(frozen=True, slots=True)
class _ReferenceRidgeSolver:
    solver_id: str = CENTERED_RIDGE_SOLVER_ID

    def preflight(
        self,
        *,
        sample_count: int,
        feature_count: int,
        target_count: int,
    ) -> None:
        preflight_reference_ridge(
            sample_count=sample_count,
            feature_count=feature_count,
            target_count=target_count,
        )

    def solve(
        self,
        feature_rows: Sequence[Sequence[float]],
        target_columns: Sequence[Sequence[float]],
        *,
        ridge: float,
    ) -> RidgeSolution:
        return solve_centered_ridge(feature_rows, target_columns, ridge=ridge)


_REFERENCE_RIDGE_SOLVER = _ReferenceRidgeSolver()


def validate_ridge_solver_id(solver_id: object) -> str:
    """Return a reviewed canonical ID or fail without importing a backend."""

    if not isinstance(solver_id, str):
        raise TypeError("solver_id must be a string")
    if solver_id not in KNOWN_RIDGE_SOLVER_IDS:
        raise ValueError(f"unknown or unreviewed ridge solver_id: {solver_id!r}")
    return solver_id


def resolve_ridge_solver(solver_id: str) -> RidgeSolver:
    """Resolve a credential-free training implementation from its static ID.

    The native C11 solver deliberately cannot be reconstructed from an ID: its
    absolute executable path and authenticated SHA-256 digest are runtime
    credentials, not artifact metadata. Callers must construct and inject a
    :class:`~tierroute.predictors.native_ridge.NativeRidgeAdapter` explicitly.
    """

    validated = validate_ridge_solver_id(solver_id)
    if validated == CENTERED_RIDGE_SOLVER_ID:
        return _REFERENCE_RIDGE_SOLVER
    if validated == NATIVE_C11_RIDGE_SOLVER_ID:
        raise ValueError(
            "native C11 ridge solver requires an explicitly injected "
            "NativeRidgeAdapter with an absolute binary_path and expected_sha256"
        )
    raise AssertionError(f"known ridge solver has no resolver: {validated!r}")


def fit_targets_with_solver(
    solver: RidgeSolver,
    feature_rows: Sequence[Sequence[float]],
    targets_by_model: Mapping[str, Sequence[float]],
    *,
    ridge: float,
) -> dict[str, tuple[tuple[float, ...], float]]:
    """Fit sorted model targets and normalize backend scalars to built-in floats."""

    if not targets_by_model:
        raise ValueError("targets_by_model must not be empty")
    if any(not isinstance(model_id, str) or not model_id.strip() for model_id in targets_by_model):
        raise TypeError("targets_by_model keys must be non-empty strings")
    model_ids = tuple(sorted(targets_by_model))
    solution = solver.solve(
        feature_rows,
        tuple(targets_by_model[model_id] for model_id in model_ids),
        ridge=ridge,
    )
    if not isinstance(solution, RidgeSolution):
        raise TypeError("ridge solver must return RidgeSolution")
    if len(solution.weights) != len(model_ids) or len(solution.intercepts) != len(model_ids):
        raise ValueError("ridge solver returned the wrong target count")

    feature_count = len(feature_rows[0]) if len(feature_rows) > 0 else 0
    fitted: dict[str, tuple[tuple[float, ...], float]] = {}
    for model_id, weights, intercept in zip(
        model_ids,
        solution.weights,
        solution.intercepts,
        strict=True,
    ):
        if len(weights) != feature_count:
            raise ValueError("ridge solver returned the wrong coefficient width")
        if isinstance(intercept, bool) or any(isinstance(value, bool) for value in weights):
            raise ValueError("ridge solver returned boolean coefficients")
        try:
            normalized_weights = tuple(float(value) for value in weights)
            normalized_intercept = float(intercept)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("ridge solver returned non-numeric coefficients") from error
        if any(not math.isfinite(value) for value in (*normalized_weights, normalized_intercept)):
            raise ValueError("ridge solver returned non-finite coefficients")
        fitted[model_id] = (normalized_weights, normalized_intercept)
    return fitted
