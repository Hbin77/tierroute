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
from tierroute.eval.provenance import (
    EVALUATION_SCOPE_ALGORITHM,
    _evaluation_scope_sha256_from_snapshots,
    _snapshot_evaluation_scope,
)
from tierroute.eval.schemas import (
    EvaluationExample,
    EvaluationReport,
    EvaluationScopeIdentity,
    QueryResult,
    ReplayCall,
    TierResult,
    TierSpec,
)


@dataclass(frozen=True, slots=True)
class _PreparedEvaluation:
    """One immutable replay scope that internal repeated runs can safely share."""

    examples: tuple[EvaluationExample, ...]
    tier_specs: tuple[TierSpec, ...]
    identity: EvaluationScopeIdentity


@dataclass(frozen=True, slots=True)
class OfflineSimulator:
    """Replay logged outcomes through a router under a swappable budget ledger."""

    ledger_factory: BudgetLedgerFactory
    max_calls_per_query: int = 1

    def __post_init__(self) -> None:
        if type(self.max_calls_per_query) is not int:
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

        snapshot_examples, snapshot_specs = _snapshot_evaluation_scope(
            examples,
            (tier_spec,),
        )
        return self._run_tier(router, snapshot_examples, snapshot_specs[0])

    def _run_tier(
        self,
        router: Router,
        examples: tuple[EvaluationExample, ...],
        tier_spec: TierSpec,
    ) -> TierResult:
        """Replay one already-normalized immutable evaluation snapshot."""

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

        prepared = self._prepare_evaluation(examples, tier_specs)
        return self._run_prepared(router, prepared, router_name=router_name)

    def _prepare_evaluation(
        self,
        examples: tuple[EvaluationExample, ...],
        tier_specs: tuple[TierSpec, ...],
    ) -> _PreparedEvaluation:
        """Validate, freeze, and hash a replay before any router is invoked."""

        snapshot_examples, snapshot_specs = _snapshot_evaluation_scope(examples, tier_specs)
        identity = EvaluationScopeIdentity(
            EVALUATION_SCOPE_ALGORITHM,
            _evaluation_scope_sha256_from_snapshots(
                snapshot_examples,
                snapshot_specs,
                self.max_calls_per_query,
            ),
            self.max_calls_per_query,
        )
        return _PreparedEvaluation(snapshot_examples, snapshot_specs, identity)

    def _run_prepared(
        self,
        router: Router,
        prepared: _PreparedEvaluation,
        *,
        router_name: str | None = None,
    ) -> EvaluationReport:
        """Replay a scope created by this simulator's trusted preparation path."""

        if prepared.identity.max_calls_per_query != self.max_calls_per_query:
            raise ValueError("prepared evaluation call cap does not match simulator")
        results = tuple(
            self._run_tier(router, prepared.examples, spec) for spec in prepared.tier_specs
        )
        return EvaluationReport(
            router_name or type(router).__name__,
            results,
            prepared.identity,
        )

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
                        "budget ledger reported realized charge "
                        f"{outcome.cost} out of budget from remaining snapshot "
                        f"{remaining_before_call}",
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
