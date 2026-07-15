# SPDX-License-Identifier: Apache-2.0
"""Router protocol and contract validation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tierroute.core.schemas import CallModel, RouterAction, RouterState, SelectOutput


class RoutingContractError(ValueError):
    """Raised when a router emits an action the current state cannot execute."""


@runtime_checkable
class Router(Protocol):
    """Specification-independent state-to-action routing interface."""

    def route(self, state: RouterState) -> RouterAction:
        """Return exactly one executable action for ``state``."""
        ...


def validate_action(state: RouterState, action: RouterAction) -> None:
    """Validate an action against candidates, history, and remaining budget."""

    if isinstance(action, CallModel):
        candidates = {model.model_id: model for model in state.candidate_models}
        if action.model_id not in candidates:
            raise RoutingContractError(f"unknown candidate model: {action.model_id}")
        cost = candidates[action.model_id].cost
        if cost > state.remaining_budget:
            raise RoutingContractError(
                f"model {action.model_id!r} costs {cost:g}, "
                f"but only {state.remaining_budget:g} remains"
            )
        return

    if isinstance(action, SelectOutput):
        if action.history_index >= len(state.call_history):
            raise RoutingContractError(
                f"history index {action.history_index} is unavailable; "
                f"history has {len(state.call_history)} entries"
            )
        return

    raise RoutingContractError(f"unsupported action type: {type(action).__name__}")
