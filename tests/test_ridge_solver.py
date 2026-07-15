# SPDX-License-Identifier: Apache-2.0
"""Tests for the project-owned dependency-free ridge reference solver."""

from __future__ import annotations

import math

import pytest

from tierroute.predictors import _ridge as ridge_solver
from tierroute.predictors._ridge import fit_centered_ridge, solve_centered_ridge
from tierroute.predictors.solvers import (
    KNOWN_RIDGE_SOLVER_IDS,
    fit_targets_with_solver,
    resolve_ridge_solver,
)


def test_model_mapping_wrapper_is_sorted_and_preserves_model_labels() -> None:
    fitted = fit_centered_ridge(
        ((0.0,), (1.0,), (2.0,)),
        {
            "z-model": (3.0, 2.0, 1.0),
            "a-model": (1.0, 2.0, 3.0),
        },
        ridge=1.0,
    )

    assert tuple(fitted) == ("a-model", "z-model")
    assert fitted["a-model"][0] == pytest.approx((2.0 / 3.0,))
    assert fitted["a-model"][1] == pytest.approx(4.0 / 3.0)
    assert fitted["z-model"][0] == pytest.approx((-2.0 / 3.0,))
    assert fitted["z-model"][1] == pytest.approx(8.0 / 3.0)


def test_model_mapping_wrapper_rejects_invalid_model_catalogue() -> None:
    with pytest.raises(ValueError, match="targets_by_model"):
        fit_centered_ridge(((1.0,),), {}, ridge=1.0)
    with pytest.raises(TypeError, match="keys"):
        fit_centered_ridge(((1.0,),), {"": (2.0,)}, ridge=1.0)
    with pytest.raises(TypeError, match="keys"):
        fit_centered_ridge(((1.0,),), {"  ": (2.0,)}, ridge=1.0)


def test_underdetermined_collinear_features_remain_finite() -> None:
    solution = solve_centered_ridge(
        ((1.0, 2.0, 3.0, 4.0), (2.0, 4.0, 6.0, 8.0)),
        ((1.0, 3.0),),
        ridge=0.5,
    )

    assert solution.weights[0] == pytest.approx(tuple(value / 15.5 for value in range(1, 5)))
    assert solution.intercepts[0] == pytest.approx(2.0 - 45.0 / 15.5)
    assert all(math.isfinite(value) for value in (*solution.weights[0], solution.intercepts[0]))


def test_reference_solver_fails_before_unreviewed_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal_matrix_called = False

    def unexpected_normal_matrix(*args: object, **kwargs: object) -> object:
        nonlocal normal_matrix_called
        normal_matrix_called = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(ridge_solver, "_MAX_REFERENCE_MULTIPLY_ACCUMULATIONS", 10)
    monkeypatch.setattr(ridge_solver, "_normal_matrix", unexpected_normal_matrix)

    with pytest.raises(ValueError, match="reviewed accelerated backend"):
        solve_centered_ridge(
            ((0.0, 1.0), (1.0, 0.0), (2.0, 3.0)),
            ((1.0, 2.0, 3.0),),
            ridge=1.0,
        )

    assert normal_matrix_called is False


def test_reference_work_estimate_covers_all_dominant_products() -> None:
    # Gram=18, Cholesky=1, two RHS=24, and two residuals=8.
    assert ridge_solver._multiply_accumulation_estimate(6, 2, 2) == 51
    assert (
        ridge_solver._multiply_accumulation_estimate(34_778, 16, 11)
        < ridge_solver._MAX_REFERENCE_MULTIPLY_ACCUMULATIONS
    )


@pytest.mark.parametrize(
    ("name", "value", "expected_exception"),
    [
        ("sample_count", 0, ValueError),
        ("sample_count", True, TypeError),
        ("feature_count", -1, ValueError),
        ("target_count", 1.5, TypeError),
    ],
)
def test_reference_preflight_validates_positive_integer_counts(
    name: str,
    value: object,
    expected_exception: type[Exception],
) -> None:
    counts: dict[str, object] = {
        "sample_count": 2,
        "feature_count": 3,
        "target_count": 1,
    }
    counts[name] = value

    with pytest.raises(expected_exception, match=name):
        ridge_solver.preflight_reference_ridge(**counts)  # type: ignore[arg-type]


def test_static_solver_resolver_preserves_reference_results() -> None:
    assert KNOWN_RIDGE_SOLVER_IDS == {ridge_solver.CENTERED_RIDGE_SOLVER_ID}
    solver = resolve_ridge_solver(ridge_solver.CENTERED_RIDGE_SOLVER_ID)
    features = ((0.0,), (1.0,), (2.0,))
    targets = {"z": (3.0, 2.0, 1.0), "a": (1.0, 2.0, 3.0)}

    solver.preflight(sample_count=3, feature_count=1, target_count=2)
    generic = fit_targets_with_solver(solver, features, targets, ridge=1.0)

    assert generic == fit_centered_ridge(features, targets, ridge=1.0)
    with pytest.raises(ValueError, match="unknown or unreviewed"):
        resolve_ridge_solver("unknown")
    assert (
        ridge_solver._multiply_accumulation_estimate(34_778, 1_030, 11)
        > ridge_solver._MAX_REFERENCE_MULTIPLY_ACCUMULATIONS
    )


def test_centered_ridge_matches_an_analytically_solvable_case() -> None:
    features = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    )
    targets = (
        (5.0, 7.0, 2.0, 4.0),
        (-2.0, -3.0, 2.0, 1.0),
    )

    solution = solve_centered_ridge(features, targets, ridge=1.0)

    assert solution.weights[0] == pytest.approx((1.0, -1.5))
    assert solution.weights[1] == pytest.approx((-0.5, 2.0))
    assert solution.intercepts == pytest.approx((4.75, -1.25))


def test_centered_ridge_matches_a_recorded_numpy_reference() -> None:
    features = (
        (1.0, 2.0, -1.0),
        (0.5, -3.0, 2.0),
        (4.0, 0.25, 1.5),
        (-2.0, 1.0, 0.0),
        (3.0, -1.0, 2.5),
    )
    targets = (
        (0.5, -1.0, 3.5, 1.25, 2.0),
        (2.0, 0.25, -2.0, 4.0, -1.5),
    )

    solution = solve_centered_ridge(features, targets, ridge=0.25)

    # Recorded from the equivalent centered numpy.linalg.solve formulation.
    expected_weights = (
        (0.14925603910700877, 1.149696111897428, 1.283841002890209),
        (-0.8701183141643969, 0.23385854633252867, -0.310485477597916),
    )
    expected_intercepts = (-0.05541943694470608, 2.0267180679615113)
    for actual, expected in zip(solution.weights, expected_weights, strict=True):
        assert actual == pytest.approx(expected, rel=2e-14, abs=2e-14)
    assert solution.intercepts == pytest.approx(
        expected_intercepts,
        rel=2e-14,
        abs=2e-14,
    )


def test_intercept_is_unregularized_and_target_shifts_do_not_change_weights() -> None:
    features = ((-2.0,), (-1.0,), (0.0,), (1.0,), (2.0,))
    base_target = (1.0, 2.0, 3.0, 4.0, 5.0)
    shifted_target = tuple(value + 1000.0 for value in base_target)

    solution = solve_centered_ridge(
        features,
        (base_target, shifted_target, (7.5,) * len(features)),
        ridge=2.0,
    )

    assert solution.weights[0] == solution.weights[1]
    assert solution.intercepts[1] - solution.intercepts[0] == pytest.approx(1000.0)
    assert solution.weights[2] == (0.0,)
    assert solution.intercepts[2] == 7.5


def test_multiple_targets_share_one_gram_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ridge_solver._cholesky
    calls = 0

    def recording_cholesky(
        matrix: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(ridge_solver, "_cholesky", recording_cholesky)

    solve_centered_ridge(
        ((0.0, 1.0), (1.0, 0.0), (2.0, 3.0)),
        ((1.0, 2.0, 3.0), (3.0, 1.0, 4.0), (-1.0, 2.0, 0.0)),
        ridge=0.5,
    )

    assert calls == 1


def test_solver_is_bitwise_deterministic_for_a_fixed_input_order() -> None:
    features = ((0.1, 2.0), (3.5, -4.0), (0.25, 8.0), (-1.0, 2.5))
    targets = ((2.0, 3.0, -1.0, 4.0), (0.0, -2.0, 5.0, 1.0))

    first = solve_centered_ridge(features, targets, ridge=0.125)
    second = solve_centered_ridge(features, targets, ridge=0.125)

    assert first == second


@pytest.mark.parametrize(
    ("features", "targets", "expected_exception", "message"),
    [
        ((), ((1.0,),), ValueError, "feature_rows must not be empty"),
        (((1.0,), (2.0, 3.0)), ((1.0, 2.0),), ValueError, "rectangular"),
        (((1.0,),), (), ValueError, "target_columns must not be empty"),
        (((1.0,), (2.0,)), ((1.0,),), ValueError, "width 2"),
        (((1.0,),), ((math.nan,),), ValueError, "finite"),
        (((True,),), ((1.0,),), TypeError, "real number"),
        ((("1",),), ((1.0,),), TypeError, "real number"),
    ],
)
def test_solver_rejects_invalid_matrix_inputs(
    features: tuple[tuple[object, ...], ...],
    targets: tuple[tuple[object, ...], ...],
    expected_exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(expected_exception, match=message):
        solve_centered_ridge(features, targets, ridge=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("ridge", [0.0, -1.0, math.inf, math.nan])
def test_solver_rejects_nonpositive_or_nonfinite_ridge(ridge: float) -> None:
    with pytest.raises(ValueError, match="ridge"):
        solve_centered_ridge(((1.0,),), ((2.0,),), ridge=ridge)


def test_solver_rejects_boolean_ridge() -> None:
    with pytest.raises(TypeError, match="ridge"):
        solve_centered_ridge(((1.0,),), ((2.0,),), ridge=True)  # type: ignore[arg-type]


def test_solver_detects_nonfinite_derived_arithmetic() -> None:
    with pytest.raises(ArithmeticError, match="Gram product"):
        solve_centered_ridge(
            ((1.0e308,), (-1.0e308,)),
            ((1.0, -1.0),),
            ridge=1.0,
        )


def test_residual_verification_rejects_a_corrupted_triangular_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def corrupted_solve(
        factor: tuple[tuple[float, ...], ...],
        right_hand_side: tuple[float, ...],
    ) -> tuple[float, ...]:
        return (1.0,) * len(right_hand_side)

    monkeypatch.setattr(ridge_solver, "_solve_cholesky", corrupted_solve)

    with pytest.raises(ArithmeticError, match="residual verification"):
        solve_centered_ridge(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0, 0.0),),
            ridge=1.0,
        )
