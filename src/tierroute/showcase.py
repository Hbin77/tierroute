# SPDX-License-Identifier: Apache-2.0
"""Audited, self-contained routing-stream evidence for the bundled showcase."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

from tierroute.adapters.budgets import PerQueryBudgetLedger
from tierroute.adapters.json_dataset import EvaluationDataset
from tierroute.core.costs import add_cost, sum_costs
from tierroute.core.schemas import BudgetTier, Cost, ModelSpec
from tierroute.eval.planning import build_per_query_oracle_plan
from tierroute.eval.provenance import (
    EVALUATION_SCOPE_ALGORITHM,
    evaluation_data_sha256,
    evaluation_replay_sha256,
    evaluation_scope_sha256,
)
from tierroute.eval.schemas import (
    EvaluationExample,
    EvaluationReport,
    EvaluationScopeIdentity,
    QueryResult,
    ReplayCall,
    TierSpec,
)
from tierroute.eval.simulator import OfflineSimulator
from tierroute.eval.validation import DomainFold, leave_one_domain_out
from tierroute.policies.baselines import OracleRouter
from tierroute.policies.benchmark import (
    BILINEAR_PREDICTOR_KIND,
    PerQueryNestedLodoBenchmark,
)
from tierroute.policies.lambda_tuning import (
    OuterFoldLambdaResult,
    TunedLambdaRouterForFold,
    fit_tiered_lambda_router_for_fold,
)
from tierroute.predictors.base import QualityPredictor
from tierroute.predictors.training import BilinearTrainingConfig, fit_calibrated_bilinear

STREAM_ID = "tierroute-bundled-three-tier-stream-v1"


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _quality(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _quality_fraction(value: float) -> Fraction:
    """Preserve the public decimal spelling used by showcase arithmetic."""

    return Fraction(str(value))


def _require_fraction(value: object, name: str) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError(f"{name} must be an exact Fraction")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class StreamAssignment:
    """One curated replay row and the per-query tier under which to show it."""

    example_id: str
    tier: BudgetTier

    def __post_init__(self) -> None:
        _require_text(self.example_id, "stream assignment example_id")
        if not isinstance(self.tier, BudgetTier):
            raise TypeError("stream assignment tier must be a BudgetTier")


BUNDLED_STREAM_ASSIGNMENTS = (
    StreamAssignment("synthetic-science-001", BudgetTier.FAST),
    StreamAssignment("synthetic-math-002", BudgetTier.BALANCED),
    StreamAssignment("synthetic-code-002", BudgetTier.PREMIUM),
)


@dataclass(frozen=True, slots=True)
class RoutingStreamStep:
    """One direct ordinary-router replay plus its evaluation-only oracle comparison."""

    index: int
    example_id: str
    prompt: str
    tier: BudgetTier
    budget_limit: Cost
    selected_model_id: str
    quoted_cost: Cost
    realized_cost: Cost
    predicted_quality: float
    observed_quality: float
    oracle_model_id: str
    oracle_realized_cost: Cost
    oracle_quality: float
    lambda_cost: Fraction
    decision_reason: str
    cumulative_realized_cost: Cost
    cumulative_observed_quality: Fraction
    cumulative_oracle_quality: Fraction
    evaluation_scope: EvaluationScopeIdentity
    cumulative_quality_retention: Fraction | None = field(init=False)

    def __post_init__(self) -> None:
        if type(self.index) is not int:
            raise TypeError("stream step index must be an integer")
        if self.index < 1:
            raise ValueError("stream step index must be positive")
        for name in ("example_id", "prompt", "selected_model_id", "oracle_model_id"):
            _require_text(getattr(self, name), f"stream step {name}")
        _require_text(self.decision_reason, "stream step decision_reason")
        if not isinstance(self.tier, BudgetTier):
            raise TypeError("stream step tier must be a BudgetTier")
        for name in (
            "budget_limit",
            "quoted_cost",
            "realized_cost",
            "oracle_realized_cost",
            "cumulative_realized_cost",
        ):
            ModelSpec(f"showcase-{name}", getattr(self, name))
        if self.quoted_cost > self.budget_limit or self.realized_cost > self.budget_limit:
            raise ValueError("selected showcase call must fit its per-query budget")
        if self.oracle_realized_cost > self.budget_limit:
            raise ValueError("showcase oracle call must fit its per-query budget")

        predicted = _quality(self.predicted_quality, "predicted_quality")
        observed = _quality(self.observed_quality, "observed_quality")
        oracle = _quality(self.oracle_quality, "oracle_quality")
        if observed < 0 or oracle < 0:
            raise ValueError("showcase observed and oracle quality must be non-negative")
        if _quality_fraction(observed) > _quality_fraction(oracle):
            raise ValueError("observed showcase quality cannot exceed its oracle")
        lambda_cost = _require_fraction(self.lambda_cost, "lambda_cost")
        cumulative_observed = _require_fraction(
            self.cumulative_observed_quality,
            "cumulative_observed_quality",
        )
        cumulative_oracle = _require_fraction(
            self.cumulative_oracle_quality,
            "cumulative_oracle_quality",
        )
        if cumulative_observed > cumulative_oracle:
            raise ValueError("cumulative observed quality cannot exceed cumulative oracle quality")
        if type(self.evaluation_scope) is not EvaluationScopeIdentity:
            raise TypeError("stream step evaluation_scope must be an EvaluationScopeIdentity")
        if self.evaluation_scope.max_calls_per_query != 1:
            raise ValueError("stream step evaluation scope must be one-shot")

        retention = None if cumulative_oracle == 0 else cumulative_observed / cumulative_oracle
        object.__setattr__(self, "predicted_quality", predicted)
        object.__setattr__(self, "observed_quality", observed)
        object.__setattr__(self, "oracle_quality", oracle)
        object.__setattr__(self, "lambda_cost", lambda_cost)
        object.__setattr__(self, "cumulative_observed_quality", cumulative_observed)
        object.__setattr__(self, "cumulative_oracle_quality", cumulative_oracle)
        object.__setattr__(self, "cumulative_quality_retention", retention)


def _expected_scope(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
) -> EvaluationScopeIdentity:
    return EvaluationScopeIdentity(
        EVALUATION_SCOPE_ALGORITHM,
        evaluation_scope_sha256(
            examples,
            tier_specs,
            max_calls_per_query=1,
        ),
        1,
    )


def _validate_dataset_binding(
    dataset: EvaluationDataset,
    benchmark: PerQueryNestedLodoBenchmark,
) -> tuple[str, str, EvaluationScopeIdentity]:
    if not isinstance(dataset, EvaluationDataset):
        raise TypeError("dataset must be an EvaluationDataset")
    if not isinstance(benchmark, PerQueryNestedLodoBenchmark):
        raise TypeError("benchmark must be a PerQueryNestedLodoBenchmark")
    if (
        benchmark.predictor_kind != BILINEAR_PREDICTOR_KIND
        or type(benchmark.training_config) is not BilinearTrainingConfig
    ):
        raise ValueError("showcase reconstruction supports only the calibrated bilinear family")

    examples = tuple(dataset.examples)
    specs = tuple(dataset.tier_specs)
    data_sha256 = evaluation_data_sha256(examples)
    replay_sha256 = evaluation_replay_sha256(examples)
    if data_sha256 != benchmark.data_sha256:
        raise ValueError("showcase dataset data hash does not match the benchmark")
    if replay_sha256 != benchmark.replay_sha256:
        raise ValueError("showcase dataset replay hash does not match the benchmark")
    benchmark_specs = tuple(result.tier_spec for result in benchmark.learned.report.tiers)
    if specs != benchmark_specs:
        raise ValueError("showcase dataset tier specs do not match the benchmark")
    example_ids = tuple(example.example_id for example in examples)
    if example_ids != benchmark.baselines.example_ids:
        raise ValueError("showcase dataset order does not match the benchmark population")

    benchmark_scope = _expected_scope(examples, specs)
    if benchmark.learned.report.evaluation_scope != benchmark_scope:
        raise ValueError("showcase dataset scope does not match the learned benchmark report")
    oracle_report = benchmark.baseline_by_name["oracle"].report
    if oracle_report.evaluation_scope != benchmark_scope:
        raise ValueError("showcase dataset scope does not match the benchmark oracle report")
    return data_sha256, replay_sha256, benchmark_scope


def _query_for(
    report: EvaluationReport,
    tier: BudgetTier,
    example_id: str,
) -> QueryResult:
    try:
        tier_result = report.by_tier()[tier]
    except KeyError as error:
        raise ValueError(f"report is missing configured tier {tier.value!r}") from error
    matches = tuple(query for query in tier_result.queries if query.example_id == example_id)
    if len(matches) != 1:
        raise ValueError(f"report must contain exactly one query for {tier.value}/{example_id}")
    return matches[0]


def _require_feasible_one_call(
    query: QueryResult,
    *,
    label: str,
    require_prediction: bool,
) -> ReplayCall:
    if (
        not query.feasible
        or query.selected_call_index != 0
        or len(query.calls) != 1
        or query.selected_model_id is None
        or query.quality is None
        or query.output is None
    ):
        raise ValueError(f"{label} must be a feasible one-call query")
    call = query.calls[0]
    if not call.within_budget or call.model_id != query.selected_model_id:
        raise ValueError(f"{label} call evidence must be selected and within budget")
    if query.cost != call.realized_cost:
        raise ValueError(f"{label} query cost must equal its one realized call")
    if require_prediction and query.predicted_quality is None:
        raise ValueError(f"{label} must retain the selected predicted quality")
    if not require_prediction and query.predicted_quality is not None:
        raise ValueError(f"{label} must not expose a routing prediction")
    return call


def _single_query_report(
    report: EvaluationReport,
    tier_spec: TierSpec,
    example_id: str,
    *,
    label: str,
    require_prediction: bool,
) -> tuple[QueryResult, ReplayCall]:
    if len(report.tiers) != 1:
        raise ValueError(f"{label} must contain one tier")
    tier_result = report.tiers[0]
    if tier_result.tier_spec != tier_spec or len(tier_result.queries) != 1:
        raise ValueError(f"{label} must contain one configured tier query")
    if (
        tier_result.budget.adapter_name != "per-query"
        or tier_result.budget.query_order != (example_id,)
        or tier_result.budget.effective_total_limit != tier_spec.budget_limit
        or tier_result.budget.spent != tier_result.queries[0].cost
    ):
        raise ValueError(f"{label} must retain exact one-row per-query accounting")
    query = tier_result.queries[0]
    if query.example_id != example_id:
        raise ValueError(f"{label} returned the wrong example")
    return query, _require_feasible_one_call(
        query,
        label=label,
        require_prediction=require_prediction,
    )


def _audited_fold_for_example(
    benchmark: PerQueryNestedLodoBenchmark,
    example_id: str,
) -> OuterFoldLambdaResult:
    matches = tuple(fold for fold in benchmark.learned.folds if example_id in fold.test_example_ids)
    if len(matches) != 1:
        raise ValueError("benchmark must bind each showcase example to exactly one outer fold")
    return matches[0]


@dataclass(frozen=True, slots=True)
class RoutingStreamShowcase:
    """Three-tier stream whose rows are direct replays bound to benchmark evidence."""

    dataset: EvaluationDataset = field(repr=False)
    benchmark: PerQueryNestedLodoBenchmark = field(repr=False)
    steps: tuple[RoutingStreamStep, ...]
    stream_id: str = field(default=STREAM_ID, init=False)
    data_sha256: str = field(init=False)
    replay_sha256: str = field(init=False)
    tier_specs: tuple[TierSpec, ...] = field(init=False)
    benchmark_evaluation_scope: EvaluationScopeIdentity = field(init=False)
    total_realized_cost: Cost = field(init=False)
    total_observed_quality: Fraction = field(init=False)
    total_oracle_quality: Fraction = field(init=False)
    quality_retention: Fraction | None = field(init=False)
    accounting_scope: str = field(default="per-query", init=False)
    cost_aggregation_scope: str = field(default="mixed-tier-reporting-only", init=False)

    def __post_init__(self) -> None:
        _require_text(self.stream_id, "stream_id")
        if not self.stream_id.isascii():
            raise ValueError("stream_id must be ASCII")
        data_sha256, replay_sha256, benchmark_scope = _validate_dataset_binding(
            self.dataset,
            self.benchmark,
        )
        steps = tuple(self.steps)
        if not steps or any(not isinstance(step, RoutingStreamStep) for step in steps):
            raise ValueError("showcase steps must contain RoutingStreamStep values")
        ordered_assignments = tuple(StreamAssignment(step.example_id, step.tier) for step in steps)
        if ordered_assignments != BUNDLED_STREAM_ASSIGNMENTS:
            raise ValueError("showcase steps must match the versioned bundled stream identity")

        examples_by_id = {example.example_id: example for example in self.dataset.examples}
        specs_by_tier = {spec.tier: spec for spec in self.dataset.tier_specs}
        if len(examples_by_id) != len(self.dataset.examples):
            raise ValueError("showcase dataset example IDs must be unique")
        if len(specs_by_tier) != len(self.dataset.tier_specs):
            raise ValueError("showcase dataset tiers must be unique")

        oracle_report = self.benchmark.baseline_by_name["oracle"].report
        seen_ids: set[str] = set()
        seen_tiers: set[BudgetTier] = set()
        running_cost = Decimal(0)
        running_observed = Fraction(0)
        running_oracle = Fraction(0)
        for expected_index, step in enumerate(steps, start=1):
            if step.index != expected_index:
                raise ValueError("showcase step indexes must be consecutive and one-based")
            if step.example_id in seen_ids:
                raise ValueError("showcase example IDs must be unique")
            seen_ids.add(step.example_id)
            seen_tiers.add(step.tier)
            try:
                example = examples_by_id[step.example_id]
                tier_spec = specs_by_tier[step.tier]
            except KeyError as error:
                raise ValueError("showcase step references an unknown example or tier") from error
            if step.prompt != example.prompt:
                raise ValueError("showcase prompt must match the bound dataset example")
            if step.budget_limit != tier_spec.budget_limit:
                raise ValueError("showcase budget must match the configured tier")
            if step.evaluation_scope != _expected_scope((example,), (tier_spec,)):
                raise ValueError("showcase step scope must match its one-row replay")

            learned_query = _query_for(
                self.benchmark.learned.report,
                step.tier,
                step.example_id,
            )
            learned_call = _require_feasible_one_call(
                learned_query,
                label="audited learned query",
                require_prediction=True,
            )
            oracle_query = _query_for(oracle_report, step.tier, step.example_id)
            oracle_call = _require_feasible_one_call(
                oracle_query,
                label="audited oracle query",
                require_prediction=False,
            )
            audited_fold = _audited_fold_for_example(self.benchmark, step.example_id)
            lambda_cost = audited_fold.tuning.lambda_by_tier[step.tier]
            if (
                step.selected_model_id != learned_query.selected_model_id
                or step.quoted_cost != learned_call.quoted_cost
                or step.realized_cost != learned_call.realized_cost
                or step.predicted_quality != learned_query.predicted_quality
                or step.observed_quality != learned_query.quality
                or step.decision_reason != learned_query.decision_reason
                or step.oracle_model_id != oracle_query.selected_model_id
                or step.oracle_realized_cost != oracle_call.realized_cost
                or step.oracle_quality != oracle_query.quality
                or step.lambda_cost != lambda_cost
            ):
                raise ValueError("showcase step conflicts with its audited benchmark evidence")

            running_cost = add_cost(running_cost, step.realized_cost)
            running_observed += _quality_fraction(step.observed_quality)
            running_oracle += _quality_fraction(step.oracle_quality)
            if (
                step.cumulative_realized_cost != running_cost
                or step.cumulative_observed_quality != running_observed
                or step.cumulative_oracle_quality != running_oracle
            ):
                raise ValueError("showcase cumulative evidence is not exactly conserved")
            expected_retention = None if running_oracle == 0 else running_observed / running_oracle
            if step.cumulative_quality_retention != expected_retention:
                raise ValueError("showcase cumulative retention is not exactly derived")

        if seen_tiers != set(specs_by_tier):
            raise ValueError("showcase assignments must cover every configured tier")
        total_cost = sum_costs(step.realized_cost for step in steps)
        if total_cost != running_cost:
            raise ValueError("showcase final realized cost is not conserved")
        quality_retention = None if running_oracle == 0 else running_observed / running_oracle

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "data_sha256", data_sha256)
        object.__setattr__(self, "replay_sha256", replay_sha256)
        object.__setattr__(self, "tier_specs", tuple(self.dataset.tier_specs))
        object.__setattr__(self, "benchmark_evaluation_scope", benchmark_scope)
        object.__setattr__(self, "total_realized_cost", total_cost)
        object.__setattr__(self, "total_observed_quality", running_observed)
        object.__setattr__(self, "total_oracle_quality", running_oracle)
        object.__setattr__(self, "quality_retention", quality_retention)


def _validate_assignments(
    assignments: Sequence[StreamAssignment],
    dataset: EvaluationDataset,
) -> tuple[StreamAssignment, ...]:
    selected = tuple(assignments)
    if not selected:
        raise ValueError("showcase assignments must not be empty")
    if any(not isinstance(assignment, StreamAssignment) for assignment in selected):
        raise TypeError("assignments must contain StreamAssignment values")
    ids = tuple(assignment.example_id for assignment in selected)
    if len(ids) != len(set(ids)):
        raise ValueError("showcase assignment example IDs must be unique")
    configured_tiers = {spec.tier for spec in dataset.tier_specs}
    tiers = tuple(assignment.tier for assignment in selected)
    assigned_tiers = set(tiers)
    if len(tiers) != len(set(tiers)):
        raise ValueError("showcase assignments must contain exactly one row per tier")
    if assigned_tiers != configured_tiers:
        raise ValueError("showcase assignments must cover every configured tier")
    available_ids = {example.example_id for example in dataset.examples}
    if any(example_id not in available_ids for example_id in ids):
        raise ValueError("showcase assignment references an unknown example ID")
    if selected != BUNDLED_STREAM_ASSIGNMENTS:
        raise ValueError("bundled showcase assignments must match the versioned stream identity")
    return selected


def _rebuild_fold_router(
    fold: DomainFold,
    benchmark: PerQueryNestedLodoBenchmark,
    tier_specs: tuple[TierSpec, ...],
) -> TunedLambdaRouterForFold:
    def train_predictor(training: tuple[EvaluationExample, ...]) -> QualityPredictor:
        return fit_calibrated_bilinear(
            training,
            config=benchmark.training_config,
        ).build_predictor()

    rebuilt = fit_tiered_lambda_router_for_fold(
        fold,
        tier_specs,
        train_predictor,
        PerQueryBudgetLedger,
        max_candidates_per_tier=benchmark.lambda_search_config.max_candidates_per_tier,
        allow_large_exhaustive=benchmark.lambda_search_config.allow_large_exhaustive,
    )
    try:
        audited = next(
            item for item in benchmark.learned.folds if item.held_out_domain == fold.held_out_domain
        )
    except StopIteration as error:
        raise ValueError("benchmark is missing a showcase outer fold") from error
    expected_training_ids = tuple(example.example_id for example in fold.training)
    expected_test_ids = tuple(example.example_id for example in fold.test)
    if (
        audited.training_example_ids != expected_training_ids
        or audited.test_example_ids != expected_test_ids
        or rebuilt.tuning != audited.tuning
    ):
        raise ValueError("reconstructed showcase fold does not match benchmark evidence")
    return rebuilt


def build_routing_stream_showcase(
    dataset: EvaluationDataset,
    benchmark: PerQueryNestedLodoBenchmark,
    *,
    assignments: Sequence[StreamAssignment] = BUNDLED_STREAM_ASSIGNMENTS,
) -> RoutingStreamShowcase:
    """Directly replay curated rows without exposing outcomes to the learned router.

    The learned call uses a reconstructed ordinary ``TieredLambdaRouter``. Only the
    separate oracle is privileged; the simulator consults logged outcomes after a model
    call to materialize its cost/output/quality evidence.
    """

    _validate_dataset_binding(dataset, benchmark)
    selected = _validate_assignments(assignments, dataset)
    examples = tuple(dataset.examples)
    tier_specs = tuple(dataset.tier_specs)
    examples_by_id = {example.example_id: example for example in examples}
    specs_by_tier = {spec.tier: spec for spec in tier_specs}
    folds_by_domain = {fold.held_out_domain: fold for fold in leave_one_domain_out(examples)}
    rebuilt_by_domain: dict[str, TunedLambdaRouterForFold] = {}
    oracle_audit = benchmark.baseline_by_name["oracle"].report
    simulator = OfflineSimulator(PerQueryBudgetLedger)

    running_cost = Decimal(0)
    running_observed = Fraction(0)
    running_oracle = Fraction(0)
    steps: list[RoutingStreamStep] = []
    for index, assignment in enumerate(selected, start=1):
        example = examples_by_id[assignment.example_id]
        tier_spec = specs_by_tier[assignment.tier]
        try:
            fold = folds_by_domain[example.domain]
        except KeyError as error:
            raise ValueError("showcase example is not covered by an outer LODO fold") from error
        rebuilt = rebuilt_by_domain.get(fold.held_out_domain)
        if rebuilt is None:
            rebuilt = _rebuild_fold_router(fold, benchmark, tier_specs)
            rebuilt_by_domain[fold.held_out_domain] = rebuilt

        # This router is ordinary: it receives prompt/tier/budget/candidates, empty
        # call history, and pre-call metadata, but never the private ID or outcomes.
        # The private example ID is reserved for the separate oracle below.
        learned_report = simulator.run(
            rebuilt.router,
            (example,),
            (tier_spec,),
            router_name="showcase-tier-lambda",
        )
        learned_query, learned_call = _single_query_report(
            learned_report,
            tier_spec,
            example.example_id,
            label="direct showcase learned report",
            require_prediction=True,
        )
        audited_learned = _query_for(
            benchmark.learned.report,
            assignment.tier,
            assignment.example_id,
        )
        if learned_query != audited_learned:
            raise ValueError("direct showcase learned replay differs from benchmark evidence")

        oracle_plan = build_per_query_oracle_plan((example,), (tier_spec,))
        oracle_report = simulator.run(
            OracleRouter(oracle_plan),
            (example,),
            (tier_spec,),
            router_name="showcase-per-query-oracle",
        )
        oracle_query, oracle_call = _single_query_report(
            oracle_report,
            tier_spec,
            example.example_id,
            label="direct showcase oracle report",
            require_prediction=False,
        )
        if learned_report.evaluation_scope != oracle_report.evaluation_scope:
            raise ValueError("direct learned and oracle showcase scopes must match")
        audited_oracle = _query_for(
            oracle_audit,
            assignment.tier,
            assignment.example_id,
        )
        if oracle_query != audited_oracle:
            raise ValueError("direct showcase oracle replay differs from benchmark evidence")
        if learned_query.quality is None or oracle_query.quality is None:
            raise AssertionError("feasible one-call showcase queries must retain quality")
        if learned_query.predicted_quality is None:
            raise AssertionError("learned showcase query must retain predicted quality")

        running_cost = add_cost(running_cost, learned_call.realized_cost)
        running_observed += _quality_fraction(learned_query.quality)
        running_oracle += _quality_fraction(oracle_query.quality)
        lambda_cost = rebuilt.tuning.lambda_by_tier[assignment.tier]
        steps.append(
            RoutingStreamStep(
                index=index,
                example_id=example.example_id,
                prompt=example.prompt,
                tier=assignment.tier,
                budget_limit=tier_spec.budget_limit,
                selected_model_id=learned_query.selected_model_id or "",
                quoted_cost=learned_call.quoted_cost,
                realized_cost=learned_call.realized_cost,
                predicted_quality=learned_query.predicted_quality,
                observed_quality=learned_query.quality,
                oracle_model_id=oracle_query.selected_model_id or "",
                oracle_realized_cost=oracle_call.realized_cost,
                oracle_quality=oracle_query.quality,
                lambda_cost=lambda_cost,
                decision_reason=learned_query.decision_reason,
                cumulative_realized_cost=running_cost,
                cumulative_observed_quality=running_observed,
                cumulative_oracle_quality=running_oracle,
                evaluation_scope=learned_report.evaluation_scope,
            )
        )

    return RoutingStreamShowcase(dataset, benchmark, tuple(steps))
