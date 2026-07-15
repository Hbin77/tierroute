# SPDX-License-Identifier: Apache-2.0
"""Tests for the stable router contract."""

import pytest

from tierroute.core import (
    BudgetTier,
    CallModel,
    CallRecord,
    ModelSpec,
    RouterState,
    RoutingContractError,
    SelectOutput,
    validate_action,
)


def make_state(*, remaining_budget: float = 2.0) -> RouterState:
    return RouterState(
        prompt="Explain why the sky is blue.",
        budget_tier=BudgetTier.BALANCED,
        remaining_budget=remaining_budget,
        call_history=(CallRecord("small", 1.0, "Rayleigh scattering"),),
        candidate_models=(ModelSpec("small", 1.0), ModelSpec("large", 2.0)),
    )


def test_state_copies_sequences_to_immutable_tuples() -> None:
    history = [CallRecord("small", 1.0, "answer")]
    candidates = [ModelSpec("small", 1.0)]

    state = RouterState("prompt", BudgetTier.FAST, 1.0, history, candidates)

    assert state.call_history == tuple(history)
    assert state.candidate_models == tuple(candidates)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ModelSpec("", 1.0), "model_id"),
        (lambda: ModelSpec("small", -1.0), "non-negative"),
        (lambda: RouterState("", BudgetTier.FAST, 1.0), "prompt"),
        (lambda: RouterState("prompt", BudgetTier.FAST, float("inf")), "finite"),
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
        validate_action(make_state(remaining_budget=1.0), CallModel("large"))


def test_validate_action_rejects_unknown_model_and_history_index() -> None:
    state = make_state()

    with pytest.raises(RoutingContractError, match="unknown candidate"):
        validate_action(state, CallModel("missing"))
    with pytest.raises(RoutingContractError, match="unavailable"):
        validate_action(state, SelectOutput(1))

