# SPDX-License-Identifier: Apache-2.0
"""Deterministic full-information replay without any live model calls."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from tierroute.core import (
    CallModel,
    CallRecord,
    Router,
    RouterAction,
    RouterState,
    RoutingContractError,
    SelectOutput,
    add_cost,
    validate_action,
)
from tierroute.eval.budgets import BudgetLedger, BudgetLedgerFactory
from tierroute.eval.protocols import PrivilegedEvaluationRouter
from tierroute.eval.schemas import (
    EvaluationExample,
    EvaluationReport,
    QueryResult,
    ReplayCall,
    TierResult,
    TierSpec,
)


@dataclass(frozen=True, slots=True)
class OfflineSimulator:
    """Replay logged outcomes through a router under a swappable budget ledger."""

    ledger_factory: BudgetLedgerFactory
    max_calls_per_query: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_calls_per_query, bool) or not isinstance(
            self.max_calls_per_query, int
        ):
            raise TypeError("max_calls_per_query must be an integer")
        if self.max_calls_per_query < 1:
            raise ValueError("max_calls_per_query must be positive")

    def run_tier(
        self,
        router: Router,
        examples: tuple[EvaluationExample, ...],
        tier_spec: TierSpec,
    ) -> TierResult:
        """Simulate every example in the supplied, recorded order."""

        examples = tuple(examples)
        if not examples:
            raise ValueError("examples must not be empty")
        example_ids = [example.example_id for example in examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("example_id values must be unique")

        ledger = self.ledger_factory(tier_spec.budget_limit, len(examples))
        queries: list[QueryResult] = []
        for example in examples:
            ledger.begin_query(example.example_id)
            try:
                queries.append(self._run_query(router, example, tier_spec, ledger))
            finally:
                ledger.finish_query()
        return TierResult(tier_spec, tuple(queries), ledger.report())

    def run(
        self,
        router: Router,
        examples: tuple[EvaluationExample, ...],
        tier_specs: tuple[TierSpec, ...],
        *,
        router_name: str | None = None,
    ) -> EvaluationReport:
        """Simulate a router independently at each configured tier."""

        tier_specs = tuple(tier_specs)
        if not tier_specs:
            raise ValueError("tier_specs must not be empty")
        tiers = [tier_spec.tier for tier_spec in tier_specs]
        if len(tiers) != len(set(tiers)):
            raise ValueError("tier_specs must contain unique tiers")
        results = tuple(self.run_tier(router, examples, spec) for spec in tier_specs)
        return EvaluationReport(router_name or type(router).__name__, results)

    def _run_query(
        self,
        router: Router,
        example: EvaluationExample,
        tier_spec: TierSpec,
        ledger: BudgetLedger,
    ) -> QueryResult:
        history: list[CallRecord] = []
        replayed_calls: list[ReplayCall] = []
        charged = Decimal(0)
        trace: list[str] = []
        outcome_by_model = {outcome.model_id: outcome for outcome in example.outcomes}
        quoted_cost_by_model = {model.model_id: model.cost for model in example.candidate_models}

        while True:
            state = RouterState(
                prompt=example.prompt,
                budget_tier=tier_spec.tier,
                remaining_budget=ledger.remaining_budget,
                call_history=tuple(history),
                candidate_models=example.candidate_models,
                metadata=example.router_metadata,
            )
            try:
                action = self._route_action(router, state, example.example_id)
                validate_action(state, action)
            except RoutingContractError as error:
                return self._failed_query(
                    example,
                    tier_spec,
                    charged,
                    trace,
                    replayed_calls,
                    str(error),
                )

            if isinstance(action, CallModel):
                if len(history) >= self.max_calls_per_query:
                    return self._failed_query(
                        example,
                        tier_spec,
                        charged,
                        trace,
                        replayed_calls,
                        f"max_calls_per_query={self.max_calls_per_query} exceeded",
                    )
                outcome = outcome_by_model[action.model_id]
                remaining_before_call = ledger.remaining_budget
                charged = add_cost(charged, outcome.cost)
                trace.append(f"call {outcome.model_id}: {action.reason}")
                within_budget = ledger.charge_realized(outcome.cost)
                replayed_calls.append(
                    ReplayCall(
                        model_id=outcome.model_id,
                        quoted_cost=quoted_cost_by_model[outcome.model_id],
                        realized_cost=outcome.cost,
                        remaining_budget_before=remaining_before_call,
                        remaining_budget_after=ledger.remaining_budget,
                        within_budget=within_budget,
                    )
                )
                if not within_budget:
                    return self._failed_query(
                        example,
                        tier_spec,
                        charged,
                        trace,
                        replayed_calls,
                        "realized cost "
                        f"{outcome.cost} exceeded remaining budget {remaining_before_call}",
                    )
                history.append(
                    CallRecord(
                        outcome.model_id,
                        outcome.cost,
                        outcome.output,
                        metadata={"predicted_quality": action.predicted_quality},
                    )
                )
                continue

            if isinstance(action, SelectOutput):
                record = history[action.history_index]
                outcome = outcome_by_model[record.model_id]
                trace.append(f"select history[{action.history_index}]: {action.reason}")
                prediction = record.metadata.get("predicted_quality")
                return QueryResult(
                    example_id=example.example_id,
                    tier=tier_spec.tier,
                    feasible=True,
                    selected_model_id=record.model_id,
                    cost=charged,
                    quality=outcome.quality,
                    output=record.output,
                    predicted_quality=float(prediction) if prediction is not None else None,
                    decision_reason=" -> ".join(trace),
                    calls=tuple(replayed_calls),
                    selected_call_index=action.history_index,
                )

            raise AssertionError("validate_action accepted an unknown action type")

    @staticmethod
    def _route_action(router: Router, state: RouterState, example_id: str) -> RouterAction:
        """Keep private evaluation identity outside the deployable router state."""

        if isinstance(router, PrivilegedEvaluationRouter):
            return router.route_with_evaluation_context(state, example_id=example_id)
        return router.route(state)

    @staticmethod
    def _failed_query(
        example: EvaluationExample,
        tier_spec: TierSpec,
        charged: Decimal,
        trace: list[str],
        replayed_calls: list[ReplayCall],
        error: str,
    ) -> QueryResult:
        return QueryResult(
            example_id=example.example_id,
            tier=tier_spec.tier,
            feasible=False,
            selected_model_id=None,
            cost=charged,
            quality=None,
            output=None,
            decision_reason=" -> ".join(trace),
            error=error,
            calls=tuple(replayed_calls),
        )
