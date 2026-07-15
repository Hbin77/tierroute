# SPDX-License-Identifier: Apache-2.0
"""Six deterministic and reproducible routing baselines."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from tierroute.core import (
    BudgetTier,
    CallModel,
    ModelSpec,
    RouterAction,
    RouterState,
    RoutingContractError,
    SelectOutput,
)
from tierroute.eval.protocols import PrivilegedEvaluationRouter
from tierroute.features import extract_surface_features


def _finish_existing(state: RouterState, reason: str) -> SelectOutput | None:
    if not state.call_history:
        return None
    return SelectOutput(len(state.call_history) - 1, reason=reason)


def _require_candidates(state: RouterState) -> tuple[ModelSpec, ...]:
    if not state.candidate_models:
        raise RoutingContractError("router state has no candidate models")
    return state.candidate_models


def _by_id(state: RouterState, model_id: str) -> ModelSpec:
    for model in _require_candidates(state):
        if model.model_id == model_id:
            return model
    raise RoutingContractError(f"configured model {model_id!r} is not a candidate")


@dataclass(frozen=True, slots=True)
class AlwaysCheapestRouter:
    """Call the lowest-cost candidate, breaking ties by model ID."""

    def route(self, state: RouterState) -> RouterAction:
        if selected := _finish_existing(state, "one-shot call completed"):
            return selected
        model = min(_require_candidates(state), key=lambda item: (item.cost, item.model_id))
        return CallModel(model.model_id, reason="always-cheapest baseline")


@dataclass(frozen=True, slots=True)
class AlwaysPremiumRouter:
    """Always call the explicitly designated premium model."""

    premium_model_id: str

    def route(self, state: RouterState) -> RouterAction:
        if selected := _finish_existing(state, "one-shot call completed"):
            return selected
        model = _by_id(state, self.premium_model_id)
        return CallModel(model.model_id, reason="always-premium baseline")


@dataclass(frozen=True, slots=True)
class RandomRouter:
    """Uniformly choose an affordable model with order-independent seeded hashing."""

    seed: int = 0

    def route(self, state: RouterState) -> RouterAction:
        if selected := _finish_existing(state, "one-shot call completed"):
            return selected
        affordable = sorted(
            (model for model in _require_candidates(state) if model.cost <= state.remaining_budget),
            key=lambda model: model.model_id,
        )
        if not affordable:
            affordable = [
                min(_require_candidates(state), key=lambda item: (item.cost, item.model_id))
            ]
        material = f"{self.seed}\0{state.prompt}\0{state.budget_tier.value}".encode()
        index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(affordable)
        return CallModel(affordable[index].model_id, reason="seeded random baseline")


@dataclass(frozen=True, slots=True)
class LengthHeuristicRouter:
    """Escalate long, code, or math prompts to a configured strong model."""

    cheap_model_id: str
    strong_model_id: str
    character_threshold: int = 240

    def __post_init__(self) -> None:
        if self.character_threshold < 1:
            raise ValueError("character_threshold must be positive")

    def route(self, state: RouterState) -> RouterAction:
        if selected := _finish_existing(state, "one-shot call completed"):
            return selected
        cheap = _by_id(state, self.cheap_model_id)
        strong = _by_id(state, self.strong_model_id)
        features = extract_surface_features(state.prompt)
        difficult = (
            features.character_count >= self.character_threshold
            or features.has_code
            or features.has_math
        )
        chosen = strong if difficult and strong.cost <= state.remaining_budget else cheap
        return CallModel(
            chosen.model_id,
            reason=(
                "long/code/math prompt" if chosen is strong else "short surface-feature prompt"
            ),
        )


@dataclass(frozen=True, slots=True)
class OracleRouter(PrivilegedEvaluationRouter):
    """Privileged offline upper bound; invalid outside the evaluation harness."""

    plan: Mapping[tuple[BudgetTier, str], str]

    def route(self, state: RouterState) -> RouterAction:
        """Reject deployment-style use because the oracle needs hidden outcomes."""

        raise RoutingContractError(
            "oracle is evaluation-only; use OfflineSimulator's privileged context"
        )

    def route_with_evaluation_context(self, state: RouterState, *, example_id: str) -> RouterAction:
        """Route with an out-of-band key that ordinary policies never receive."""

        if selected := _finish_existing(state, "oracle call completed"):
            return selected
        if not example_id:
            raise RoutingContractError("oracle requires a private evaluation example ID")
        try:
            model_id = self.plan[(state.budget_tier, example_id)]
        except KeyError as error:
            raise RoutingContractError(
                f"oracle plan has no entry for {state.budget_tier.value}/{example_id}"
            ) from error
        _by_id(state, model_id)
        return CallModel(model_id, reason="budget-feasible oracle plan")


@dataclass(frozen=True, slots=True)
class DomainBestRouter:
    """Use a table fitted on training domains, with an explicit fallback model."""

    table: Mapping[tuple[BudgetTier, str], str]
    fallback_model_id: str

    def route(self, state: RouterState) -> RouterAction:
        if selected := _finish_existing(state, "domain-table call completed"):
            return selected
        domain = str(state.metadata.get("domain", ""))
        model_id = self.table.get((state.budget_tier, domain), self.fallback_model_id)
        configured = _by_id(state, model_id)
        if configured.cost > state.remaining_budget:
            fallback = _by_id(state, self.fallback_model_id)
            if fallback.cost <= state.remaining_budget:
                return CallModel(
                    fallback.model_id,
                    reason=f"domain table fallback ({domain or 'unseen'} domain)",
                )
        return CallModel(
            configured.model_id,
            reason=f"domain table ({domain or 'unseen'} domain)",
        )
