# SPDX-License-Identifier: Apache-2.0
"""Stable schemas shared by routers, simulators, and external adapters.

Costs intentionally have no built-in currency or token unit. An adapter normalizes
the challenge-specific value into one non-negative scale before constructing these
objects. This keeps the routing core independent of the final SKT budget schema.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


class BudgetTier(str, Enum):
    """User-facing service tiers ordered by increasing budget."""

    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_negative_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A candidate model and its normalized one-call cost."""

    model_id: str
    cost: float
    display_name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        _require_non_negative_finite(self.cost, "cost")
        if self.display_name is not None:
            _require_non_empty(self.display_name, "display_name")


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One completed candidate call visible to a router.

    Ground-truth quality is deliberately absent. The offline harness keeps it in a
    private replay table so a policy cannot accidentally use evaluation labels.
    """

    model_id: str
    cost: float
    output: str
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        _require_non_negative_finite(self.cost, "cost")
        if not isinstance(self.output, str):
            raise TypeError("output must be a string")


@dataclass(frozen=True, slots=True)
class RouterState:
    """All information a policy may use for its next decision."""

    prompt: str
    budget_tier: BudgetTier
    remaining_budget: float
    call_history: tuple[CallRecord, ...] = ()
    candidate_models: tuple[ModelSpec, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty(self.prompt, "prompt")
        if not isinstance(self.budget_tier, BudgetTier):
            raise TypeError("budget_tier must be a BudgetTier")
        _require_non_negative_finite(self.remaining_budget, "remaining_budget")
        object.__setattr__(self, "call_history", tuple(self.call_history))
        object.__setattr__(self, "candidate_models", tuple(self.candidate_models))
        model_ids = [model.model_id for model in self.candidate_models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("candidate_models must have unique model_id values")


@dataclass(frozen=True, slots=True)
class CallModel:
    """Ask the environment to call one candidate model."""

    model_id: str
    reason: str = ""
    predicted_quality: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.model_id, "model_id")
        if self.predicted_quality is not None:
            if isinstance(self.predicted_quality, bool) or not isinstance(
                self.predicted_quality, (int, float)
            ):
                raise TypeError("predicted_quality must be a real number or None")
            if not math.isfinite(self.predicted_quality):
                raise ValueError("predicted_quality must be finite when provided")


@dataclass(frozen=True, slots=True)
class SelectOutput:
    """Finish by selecting an output already present in ``call_history``."""

    history_index: int
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.history_index, bool) or not isinstance(self.history_index, int):
            raise TypeError("history_index must be an integer")
        if self.history_index < 0:
            raise ValueError("history_index must be non-negative")


RouterAction: TypeAlias = CallModel | SelectOutput
