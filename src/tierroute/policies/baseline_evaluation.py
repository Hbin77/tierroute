# SPDX-License-Identifier: Apache-2.0
"""Leakage-free outer-LODO orchestration for the six required baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType

from tierroute.core import (
    BudgetTier,
    CallModel,
    Cost,
    ModelSpec,
    RouterAction,
    RouterState,
    RoutingContractError,
    SelectOutput,
    add_cost,
    scale_cost,
    subtract_cost,
    sum_costs,
)
from tierroute.eval import (
    BudgetReport,
    EvaluationExample,
    EvaluationReport,
    OfflineSimulator,
    ScoreSummary,
    TierSpec,
    build_per_query_oracle_plan,
    fit_per_query_domain_table,
    leave_one_domain_out,
    oracle_gap_recovery,
    summarize_report,
)
from tierroute.eval.budgets import BudgetLedger, BudgetLedgerFactory
from tierroute.eval.protocols import PrivilegedEvaluationRouter
from tierroute.policies.baselines import (
    AlwaysCheapestRouter,
    AlwaysPremiumRouter,
    DomainBestRouter,
    LengthHeuristicRouter,
    OracleRouter,
    RandomRouter,
)

BASELINE_NAMES = (
    "always-cheapest",
    "always-premium",
    "random",
    "length-heuristic",
    "oracle",
    "domain-best-table",
)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One aligned row in the six-baseline scorecard."""

    name: str
    report: EvaluationReport
    score: ScoreSummary
    gap_recovery: float | None
    total_cost: Cost


@dataclass(frozen=True, slots=True)
class DomainTableEntry:
    """One immutable, auditable entry fitted inside an outer fold."""

    tier: BudgetTier
    observable_domain_tag: str
    model_id: str


@dataclass(frozen=True, slots=True)
class OuterFoldBaselineEvidence:
    """Training/test identities and fitted table evidence for one outer fold."""

    held_out_domain: str
    training_example_ids: tuple[str, ...]
    test_example_ids: tuple[str, ...]
    fitted_domain_table_entries: tuple[DomainTableEntry, ...]
    fallback_model_id: str


@dataclass(frozen=True, slots=True)
class LodoSixBaselineEvaluation:
    """Six reports sharing one per-query population, order, and accounting scope."""

    folds: tuple[OuterFoldBaselineEvidence, ...]
    baselines: tuple[BaselineResult, ...]
    example_ids: tuple[str, ...]
    accounting_scope: str = field(default="per-query", init=False)

    def by_name(self) -> dict[str, BaselineResult]:
        """Index the six unique rows by their stable baseline name."""

        indexed = {result.name: result for result in self.baselines}
        if len(indexed) != len(self.baselines):
            raise ValueError("baseline evaluation contains duplicate names")
        return indexed


@dataclass(frozen=True, slots=True)
class _ScheduledEvaluationRouter(PrivilegedEvaluationRouter):
    """Replay fold-specific decisions without exposing example IDs to RouterState."""

    schedule: Mapping[tuple[BudgetTier, str], str]
    call_reason: str

    def route(self, state: RouterState) -> RouterAction:
        """Reject deployment use because the outer-fold schedule is row-specific."""

        raise RoutingContractError(
            "outer-fold schedule is evaluation-only; use OfflineSimulator's private context"
        )

    def route_with_evaluation_context(
        self,
        state: RouterState,
        *,
        example_id: str,
    ) -> RouterAction:
        """Read only the simulator's private row key, never hidden outcomes."""

        if state.call_history:
            return SelectOutput(len(state.call_history) - 1, reason="scheduled call completed")
        try:
            model_id = self.schedule[(state.budget_tier, example_id)]
        except KeyError as error:
            raise RoutingContractError(
                f"schedule has no entry for {state.budget_tier.value}/{example_id}"
            ) from error
        return CallModel(model_id, reason=self.call_reason)


@dataclass(slots=True)
class _PerQueryLedgerGuard:
    """Prove reset and accounting behavior on the ledger used by each replay."""

    delegate: BudgetLedger
    budget_limit: Cost
    expected_queries: int
    _active: bool = False
    _expected_remaining: Cost | None = None
    _spent: Cost = Decimal(0)
    _over_budget_calls: int = 0
    _query_order: list[str] = field(default_factory=list)

    def begin_query(self, example_id: str) -> None:
        self.delegate.begin_query(example_id)
        remaining = self.delegate.remaining_budget
        if remaining != self.budget_limit:
            raise ValueError("per-query ledger must reset the full configured limit at every query")
        self._active = True
        self._expected_remaining = self.budget_limit
        self._query_order.append(example_id)

    @property
    def remaining_budget(self) -> Cost:
        if not self._active or self._expected_remaining is None:
            raise RuntimeError("begin_query must be called before reading budget")
        actual = self.delegate.remaining_budget
        if actual != self._expected_remaining:
            raise ValueError("per-query ledger exposed an inconsistent remaining budget")
        return actual

    def charge_realized(self, cost: Cost) -> bool:
        before = self.remaining_budget
        accepted = self.delegate.charge_realized(cost)
        self._spent = add_cost(self._spent, cost)
        if cost > before:
            expected_accepted = False
            expected_remaining = Decimal(0)
            self._over_budget_calls += 1
        else:
            expected_accepted = True
            expected_remaining = subtract_cost(before, cost)
        if accepted is not expected_accepted:
            raise ValueError("per-query ledger returned an inconsistent charge result")
        self._expected_remaining = expected_remaining
        if self.delegate.remaining_budget != expected_remaining:
            raise ValueError("per-query ledger charged an inconsistent remaining budget")
        return accepted

    def finish_query(self) -> None:
        self.delegate.finish_query()
        self._active = False
        self._expected_remaining = None

    def report(self) -> BudgetReport:
        report = self.delegate.report()
        if (
            report.adapter_name != "per-query"
            or report.configured_limit != self.budget_limit
            or report.effective_total_limit != scale_cost(self.budget_limit, self.expected_queries)
            or report.spent != self._spent
            or report.over_budget_calls != self._over_budget_calls
            or report.query_order != tuple(self._query_order)
        ):
            raise ValueError("per-query ledger emitted an inconsistent accounting report")
        return report


@dataclass(frozen=True, slots=True)
class _GuardedPerQueryLedgerFactory:
    """Wrap every produced ledger, including factory calls after preflight."""

    delegate: BudgetLedgerFactory

    def __call__(self, budget_limit: Cost, expected_queries: int) -> BudgetLedger:
        return _PerQueryLedgerGuard(
            self.delegate(budget_limit, expected_queries),
            budget_limit,
            expected_queries,
        )


def _tier_specs(tier_specs: Sequence[TierSpec]) -> tuple[TierSpec, ...]:
    specs = tuple(tier_specs)
    if not specs:
        raise ValueError("tier_specs must not be empty")
    tiers = [spec.tier for spec in specs]
    if len(tiers) != len(set(tiers)):
        raise ValueError("tier_specs must contain unique tiers")
    return specs


def _stable_catalogue(examples: tuple[EvaluationExample, ...]) -> tuple[ModelSpec, ...]:
    """Require one model-ID/quote map while accepting irrelevant row order changes."""

    if not examples:
        raise ValueError("examples must not be empty")
    reference = {model.model_id: model.cost for model in examples[0].candidate_models}
    for example in examples[1:]:
        current = {model.model_id: model.cost for model in example.candidate_models}
        if current != reference:
            raise ValueError(
                "per-query LODO baselines require a stable model catalogue and quoted costs"
            )
    return tuple(sorted(examples[0].candidate_models, key=lambda model: model.model_id))


def _require_model_role(role: str, model_id: str, catalogue: tuple[ModelSpec, ...]) -> None:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"{role} must be a non-empty model ID")
    if model_id not in {model.model_id for model in catalogue}:
        raise ValueError(f"{role} {model_id!r} is absent from the stable model catalogue")


def _preflight_per_query_ledger(
    ledger_factory: BudgetLedgerFactory,
    specs: tuple[TierSpec, ...],
    query_count: int,
) -> None:
    """Fail before label-derived planning if a factory is not per-query accounting."""

    for spec in specs:
        ledger = ledger_factory(spec.budget_limit, query_count)
        for example_id in ("tierroute-preflight-1", "tierroute-preflight-2"):
            ledger.begin_query(example_id)
            if ledger.remaining_budget != spec.budget_limit:
                raise ValueError(
                    "evaluate_per_query_lodo_baselines requires a fresh per-query ledger"
                )
            if not ledger.charge_realized(spec.budget_limit):
                raise ValueError(
                    "evaluate_per_query_lodo_baselines requires a fresh per-query ledger"
                )
            if ledger.remaining_budget != Decimal(0):
                raise ValueError(
                    "evaluate_per_query_lodo_baselines requires a fresh per-query ledger"
                )
            ledger.finish_query()
        ledger.report()


def _domain_table_schedule(
    examples: tuple[EvaluationExample, ...],
    specs: tuple[TierSpec, ...],
) -> tuple[
    Mapping[tuple[BudgetTier, str], str],
    tuple[OuterFoldBaselineEvidence, ...],
]:
    """Fit each table on one outer training side and keep only its test decisions."""

    schedule: dict[tuple[BudgetTier, str], str] = {}
    evidence: list[OuterFoldBaselineEvidence] = []
    test_ids: list[str] = []
    for fold in leave_one_domain_out(examples):
        plan = fit_per_query_domain_table(fold.training, specs)
        router = DomainBestRouter(plan.table, plan.fallback_model_id)
        for example in fold.test:
            test_ids.append(example.example_id)
            for spec in specs:
                state = RouterState(
                    prompt=example.prompt,
                    budget_tier=spec.tier,
                    remaining_budget=spec.budget_limit,
                    candidate_models=example.candidate_models,
                    metadata=example.router_metadata,
                )
                action = router.route(state)
                if not isinstance(action, CallModel):
                    raise AssertionError("a fresh domain-table route must call one model")
                key = (spec.tier, example.example_id)
                if key in schedule:
                    raise AssertionError(
                        f"outer LODO decision overlaps: {spec.tier.value}/{key[1]}"
                    )
                schedule[key] = action.model_id
        entries = tuple(
            DomainTableEntry(tier, tag, model_id)
            for (tier, tag), model_id in sorted(
                plan.table.items(),
                key=lambda item: (item[0][0].value, item[0][1], item[1]),
            )
        )
        evidence.append(
            OuterFoldBaselineEvidence(
                held_out_domain=fold.held_out_domain,
                training_example_ids=tuple(example.example_id for example in fold.training),
                test_example_ids=tuple(example.example_id for example in fold.test),
                fitted_domain_table_entries=entries,
                fallback_model_id=plan.fallback_model_id,
            )
        )

    expected_ids = {example.example_id for example in examples}
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != expected_ids:
        raise AssertionError("outer LODO test folds must cover every example exactly once")
    expected_schedule = {(spec.tier, example.example_id) for spec in specs for example in examples}
    if set(schedule) != expected_schedule:
        raise AssertionError("outer LODO domain schedule must cover every tier/example pair")
    return MappingProxyType(schedule), tuple(evidence)


def _validate_report_scope(
    report: EvaluationReport,
    specs: tuple[TierSpec, ...],
    example_ids: tuple[str, ...],
) -> None:
    if len(report.tiers) != len(specs):
        raise AssertionError("baseline report changed the configured tier population")
    for result, spec in zip(report.tiers, specs, strict=True):
        query_ids = tuple(query.example_id for query in result.queries)
        if (
            result.tier_spec != spec
            or query_ids != example_ids
            or result.budget.query_order != example_ids
            or result.budget.adapter_name != "per-query"
            or result.budget.configured_limit != spec.budget_limit
            or result.budget.effective_total_limit
            != scale_cost(spec.budget_limit, len(example_ids))
            or result.budget.spent != sum_costs(query.cost for query in result.queries)
            or any(query.feasible and query.cost > spec.budget_limit for query in result.queries)
        ):
            raise AssertionError("six-baseline reports must share tier, order, and budget scope")


def _validate_oracle_upper_bound(
    reports: Mapping[str, EvaluationReport],
    *,
    tolerance: float = 1e-12,
) -> None:
    oracle_tiers = reports["oracle"].by_tier()
    for name, report in reports.items():
        for tier, result in report.by_tier().items():
            oracle_queries = {query.example_id: query for query in oracle_tiers[tier].queries}
            for query in result.queries:
                oracle_query = oracle_queries[query.example_id]
                if not oracle_query.feasible or oracle_query.quality is None:
                    raise AssertionError("per-query oracle plan must be feasible and complete")
                if (
                    query.feasible
                    and query.quality is not None
                    and query.quality > oracle_query.quality + tolerance
                ):
                    raise AssertionError(
                        f"{name} exceeds the per-query oracle for {tier.value}/{query.example_id}"
                    )


def evaluate_per_query_lodo_baselines(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
    ledger_factory: BudgetLedgerFactory,
    *,
    premium_model_id: str,
    strong_model_id: str,
    random_seed: int = 2026,
    character_threshold: int = 120,
) -> LodoSixBaselineEvaluation:
    """Evaluate all six baselines on one original-order outer-LODO replay.

    Folds create only the fitted domain-table decisions and their audit evidence.  The
    six routers are then replayed once over the same complete row order, so folds never
    create different populations or ordering.  This API accepts only a ledger that
    proves per-query semantics because its independent oracle is not a cumulative-stream
    oracle.

    Split-only ``EvaluationExample.domain`` values construct folds.  Domain-table
    fitting and lookup use only pre-call ``router_metadata["domain"]`` tags; an absent
    tag takes the cheapest fallback.  With identical split and observable domains,
    strict LODO therefore makes this baseline equal always-cheapest on held-out rows.
    """

    ordered = tuple(examples)
    specs = _tier_specs(tier_specs)
    if not ordered:
        raise ValueError("examples must not be empty")
    guarded_ledger_factory = _GuardedPerQueryLedgerFactory(ledger_factory)
    _preflight_per_query_ledger(guarded_ledger_factory, specs, len(ordered))
    catalogue = _stable_catalogue(ordered)
    _require_model_role("premium_model_id", premium_model_id, catalogue)
    _require_model_role("strong_model_id", strong_model_id, catalogue)
    domain_schedule, fold_evidence = _domain_table_schedule(ordered, specs)
    oracle_plan = build_per_query_oracle_plan(ordered, specs)

    cheap = min(catalogue, key=lambda model: (model.cost, model.model_id))
    routers = (
        ("always-cheapest", AlwaysCheapestRouter()),
        ("always-premium", AlwaysPremiumRouter(premium_model_id)),
        ("random", RandomRouter(seed=random_seed)),
        (
            "length-heuristic",
            LengthHeuristicRouter(
                cheap.model_id,
                strong_model_id,
                character_threshold=character_threshold,
            ),
        ),
        ("oracle", OracleRouter(oracle_plan)),
        (
            "domain-best-table",
            _ScheduledEvaluationRouter(domain_schedule, "outer-LODO domain-table decision"),
        ),
    )
    simulator = OfflineSimulator(guarded_ledger_factory)
    reports = {
        name: simulator.run(router, ordered, specs, router_name=name) for name, router in routers
    }
    example_ids = tuple(example.example_id for example in ordered)
    for report in reports.values():
        _validate_report_scope(report, specs, example_ids)
    _validate_oracle_upper_bound(reports)

    cheapest_report = reports["always-cheapest"]
    oracle_report = reports["oracle"]
    rows = tuple(
        BaselineResult(
            name=name,
            report=reports[name],
            score=summarize_report(reports[name]),
            gap_recovery=oracle_gap_recovery(
                reports[name],
                cheapest_report,
                oracle_report,
            ),
            total_cost=sum_costs(
                query.cost for tier in reports[name].tiers for query in tier.queries
            ),
        )
        for name in BASELINE_NAMES
    )
    return LodoSixBaselineEvaluation(fold_evidence, rows, example_ids)
