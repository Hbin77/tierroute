# SPDX-License-Identifier: Apache-2.0
"""Tests for the stable router contract."""

from decimal import Decimal, Inexact, Rounded, localcontext

import pytest

from tierroute.core import (
    BudgetTier,
    CallModel,
    CallRecord,
    ModelSpec,
    RouterState,
    RoutingContractError,
    SelectOutput,
    add_cost,
    as_cost,
    divide_cost,
    scale_cost,
    subtract_cost,
    sum_costs,
    validate_action,
)


def make_state(*, remaining_budget: Decimal = Decimal("2")) -> RouterState:
    return RouterState(
        prompt="Explain why the sky is blue.",
        budget_tier=BudgetTier.BALANCED,
        remaining_budget=remaining_budget,
        call_history=(CallRecord("small", Decimal("1"), "Rayleigh scattering"),),
        candidate_models=(
            ModelSpec("small", Decimal("1")),
            ModelSpec("large", Decimal("2")),
        ),
    )


def test_state_copies_sequences_to_immutable_tuples() -> None:
    history = [CallRecord("small", Decimal("1"), "answer")]
    candidates = [ModelSpec("small", Decimal("1"))]

    state = RouterState("prompt", BudgetTier.FAST, Decimal("1"), history, candidates)

    assert state.call_history == tuple(history)
    assert state.candidate_models == tuple(candidates)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ModelSpec("", Decimal("1")), "model_id"),
        (lambda: ModelSpec("small", Decimal("-1")), "non-negative"),
        (lambda: RouterState("", BudgetTier.FAST, Decimal("1")), "prompt"),
        (lambda: RouterState("prompt", BudgetTier.FAST, Decimal("Infinity")), "finite"),
    ],
)
def test_invalid_schemas_fail_early(factory: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]


def test_validate_action_accepts_affordable_call_and_existing_output() -> None:
    state = make_state()

    validate_action(state, CallModel("large"))
    validate_action(state, SelectOutput(0))


def test_validate_action_rejects_over_budget_call() -> None:
    with pytest.raises(RoutingContractError, match="only 1 remains"):
        validate_action(make_state(remaining_budget=Decimal("1")), CallModel("large"))


def test_validate_action_rejects_unknown_model_and_history_index() -> None:
    state = make_state()

    with pytest.raises(RoutingContractError, match="unknown candidate"):
        validate_action(state, CallModel("missing"))
    with pytest.raises(RoutingContractError, match="unavailable"):
        validate_action(state, SelectOutput(1))


def test_validate_action_rejects_call_when_candidate_catalogue_is_empty() -> None:
    state = RouterState("prompt", BudgetTier.FAST, Decimal("1"))

    with pytest.raises(RoutingContractError, match="unknown candidate"):
        validate_action(state, CallModel("unlisted"))


def test_as_cost_rejects_inexact_floats() -> None:
    assert as_cost("0.1") + as_cost("0.2") == as_cost("0.3")
    with pytest.raises(TypeError, match="not exact"):
        as_cost(0.1)


def test_cost_arithmetic_is_exact_under_a_low_precision_context() -> None:
    thirds = [Decimal("0.33333333333333333333333333333")] * 3
    tiny_overspend = Decimal("5e-29")

    with localcontext() as context:
        context.prec = 2
        subtotal = sum_costs(thirds)
        total = add_cost(subtotal, tiny_overspend)
        remaining = subtract_cost(Decimal("1"), subtotal)
        scaled = scale_cost(Decimal("1.234567890123456789"), 3)
        terminating_quotient = divide_cost(Decimal("1"), 8)
        repeating_quotient = divide_cost(Decimal("1"), 3)

    assert subtotal == Decimal("0.99999999999999999999999999999")
    assert total == Decimal("1.00000000000000000000000000004")
    assert remaining == Decimal("1e-29")
    assert scaled == Decimal("3.703703670370370367")
    assert terminating_quotient == Decimal("0.125")
    assert repeating_quotient == Decimal("0.333333333333333333333333333333333333333333333333333")


def test_repeating_cost_division_ignores_ambient_rounding_traps() -> None:
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        result = divide_cost(Decimal("1"), 3)

    assert result == Decimal("0.333333333333333333333333333333333333333333333333333")


def test_cost_arithmetic_validates_operands_and_non_negative_results() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        add_cost(1, Decimal("1"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        add_cost(Decimal("-1"), Decimal("1"))
    with pytest.raises(ValueError, match="negative result"):
        subtract_cost(Decimal("1"), Decimal("2"))
    with pytest.raises(TypeError, match="integer"):
        scale_cost(Decimal("1"), True)
    with pytest.raises(ValueError, match="non-negative"):
        scale_cost(Decimal("1"), -1)
    with pytest.raises(TypeError, match="integer"):
        divide_cost(Decimal("1"), True)
    with pytest.raises(ValueError, match="positive"):
        divide_cost(Decimal("1"), 0)
