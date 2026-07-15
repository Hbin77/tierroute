# SPDX-License-Identifier: Apache-2.0
"""Exact tier-lambda tuning with cross-fitted, example-keyed predictions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from heapq import heappush, heapreplace
from itertools import pairwise
from types import MappingProxyType

from tierroute.core import (
    BudgetTier,
    RouterAction,
    RouterState,
    RoutingContractError,
    SelectOutput,
)
from tierroute.eval import (
    BudgetLedgerFactory,
    DomainFold,
    EvaluationExample,
    EvaluationReport,
    OfflineSimulator,
    ScoreSummary,
    TierResult,
    TierSpec,
    evaluation_data_sha256,
    evaluation_replay_sha256,
    leave_one_domain_out,
    summarize_report,
)
from tierroute.eval.protocols import PrivilegedEvaluationRouter
from tierroute.policies.lambda_threshold import (
    LambdaInput,
    TieredLambdaRouter,
    as_lambda,
    route_from_predictions,
)
from tierroute.predictors.base import (
    BatchPromptQualityPredictor,
    BatchQualityPredictor,
    QualityPredictor,
)

PredictionKey = tuple[str, str]
PredictorTrainer = Callable[[tuple[EvaluationExample, ...]], QualityPredictor]
MAX_UNCONFIRMED_EXHAUSTIVE_CANDIDATES = 100_000
MAX_UNCONFIRMED_EXHAUSTIVE_UTILITY_EVALUATIONS = 100_000_000


def _quality_fraction(value: int | float) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("predicted and realized quality values must be real numbers")
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("predicted and realized quality values must be finite")
    return Fraction.from_float(value)


def _prediction_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("predicted quality values must be real numbers")
    _quality_fraction(value)
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError("predicted quality must fit a finite float") from error
    if not math.isfinite(numeric):
        raise ValueError("predicted quality must fit a finite float")
    return numeric


def _ordered_examples(
    examples: Sequence[EvaluationExample],
) -> tuple[EvaluationExample, ...]:
    ordered = tuple(examples)
    if not ordered:
        raise ValueError("examples must not be empty")
    ids = tuple(example.example_id for example in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("examples must have unique example IDs")
    return ordered


def _tier_specs(tier_specs: Sequence[TierSpec]) -> tuple[TierSpec, ...]:
    result = tuple(tier_specs)
    if not result:
        raise ValueError("tier_specs must not be empty")
    tiers = tuple(spec.tier for spec in result)
    if len(tiers) != len(set(tiers)):
        raise ValueError("tier_specs must contain unique tiers")
    return result


def _model_ids(examples: tuple[EvaluationExample, ...]) -> tuple[str, ...]:
    expected = tuple(sorted(model.model_id for model in examples[0].candidate_models))
    for example in examples[1:]:
        current = tuple(sorted(model.model_id for model in example.candidate_models))
        if current != expected:
            raise ValueError("lambda tuning requires a stable model catalogue")
    return expected


def _validate_exhaustive_search_size(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    model_count: int,
    *,
    allow_large_exhaustive: bool,
) -> None:
    """Reject an unacknowledged exhaustive search before roots are materialized.

    One unequal-cost model pair can contribute at most one non-negative root. If there
    are ``r`` such pair occurrences, the exact set has at most ``r + 1`` boundaries
    (including zero), the adjacent midpoints, and one tail value: ``2 * (r + 1)``
    candidates. Multiplying that count by tiers, rows, and models bounds the dominant
    exact-utility work performed by simulator replay. Duplicate or negative roots make
    the real workload smaller, so callers can explicitly override a conservative bound.
    """

    if not isinstance(allow_large_exhaustive, bool):
        raise TypeError("allow_large_exhaustive must be a boolean")
    if allow_large_exhaustive:
        return

    unequal_cost_pair_occurrences = 0
    for example in examples:
        models = example.candidate_models
        unequal_cost_pair_occurrences += sum(
            left.cost != right.cost
            for index, left in enumerate(models)
            for right in models[index + 1 :]
        )
    candidate_upper_bound = 2 * (unequal_cost_pair_occurrences + 1)
    utility_evaluation_upper_bound = (
        candidate_upper_bound * len(tier_specs) * len(examples) * model_count
    )
    if (
        candidate_upper_bound <= MAX_UNCONFIRMED_EXHAUSTIVE_CANDIDATES
        and utility_evaluation_upper_bound <= MAX_UNCONFIRMED_EXHAUSTIVE_UTILITY_EVALUATIONS
    ):
        return
    raise ValueError(
        "exhaustive lambda search refused before candidate materialization: "
        f"candidate upper bound={candidate_upper_bound:,} "
        f"(limit={MAX_UNCONFIRMED_EXHAUSTIVE_CANDIDATES:,}); "
        f"utility-evaluation upper bound={utility_evaluation_upper_bound:,} "
        f"(limit={MAX_UNCONFIRMED_EXHAUSTIVE_UTILITY_EVALUATIONS:,}); "
        "set max_candidates_per_tier (for example 257), or set "
        "allow_large_exhaustive=True after reviewing resource requirements"
    )


@dataclass(frozen=True, slots=True)
class CrossFittedPredictionTable:
    """Immutable predictions keyed by private example ID and candidate model ID."""

    scores: Mapping[PredictionKey, float]

    def __post_init__(self) -> None:
        if not isinstance(self.scores, Mapping):
            raise TypeError("prediction scores must be a mapping")
        copied: dict[PredictionKey, float] = {}
        for key, value in self.scores.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(not isinstance(item, str) or not item.strip() for item in key)
            ):
                raise ValueError("prediction keys must be non-empty (example_id, model_id) pairs")
            copied[key] = _prediction_float(value)
        if not copied:
            raise ValueError("prediction table must not be empty")
        object.__setattr__(self, "scores", MappingProxyType(copied))

    def validate_examples(self, examples: Sequence[EvaluationExample]) -> None:
        """Require exactly one prediction for every supplied example/model pair."""

        ordered = _ordered_examples(examples)
        expected = {
            (example.example_id, model.model_id)
            for example in ordered
            for model in example.candidate_models
        }
        if set(self.scores) != expected:
            missing = sorted(expected - set(self.scores))
            extra = sorted(set(self.scores) - expected)
            raise ValueError(
                f"prediction table coverage mismatch: missing={missing}, extra={extra}"
            )

    def for_example(self, example_id: str, model_ids: Sequence[str]) -> Mapping[str, float]:
        """Return an exact model mapping for one private evaluation row."""

        model_ids = tuple(model_ids)
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError("model_ids must be non-empty and unique")
        try:
            return MappingProxyType(
                {model_id: self.scores[(example_id, model_id)] for model_id in model_ids}
            )
        except KeyError as error:
            raise RoutingContractError(
                f"prediction table has no score for example/model {error.args[0]!r}"
            ) from error

    def sha256(self, examples: Sequence[EvaluationExample]) -> str:
        """Hash predictions in the caller's evaluation order using exact float ratios."""

        ordered = _ordered_examples(examples)
        self.validate_examples(ordered)
        payload = []
        for example in ordered:
            models = []
            for model in sorted(example.candidate_models, key=lambda item: item.model_id):
                numerator, denominator = _quality_fraction(
                    self.scores[(example.example_id, model.model_id)]
                ).as_integer_ratio()
                models.append(
                    {
                        "model_id": model.model_id,
                        "numerator": str(numerator),
                        "denominator": str(denominator),
                    }
                )
            payload.append({"example_id": example.example_id, "models": models})
        try:
            document = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        except UnicodeEncodeError as error:
            raise ValueError("prediction identity contains invalid Unicode text") from error
        return hashlib.sha256(document).hexdigest()

    @classmethod
    def from_predictor(
        cls,
        examples: Sequence[EvaluationExample],
        predictor: QualityPredictor,
    ) -> CrossFittedPredictionTable:
        """Materialize scores without exposing example IDs to a deployable predictor."""

        ordered = _ordered_examples(examples)
        model_ids = _model_ids(ordered)
        prompts = tuple(example.prompt for example in ordered)
        if isinstance(predictor, BatchPromptQualityPredictor):
            rows = predictor.predict_batch(prompts, model_ids)
        elif isinstance(predictor, BatchQualityPredictor):
            rows = tuple(predictor.predict_many(prompt, model_ids) for prompt in prompts)
        elif isinstance(predictor, QualityPredictor):
            rows = tuple(
                {model_id: predictor.predict(prompt, model_id) for model_id in model_ids}
                for prompt in prompts
            )
        else:
            raise TypeError("predictor must implement predict, predict_many, or predict_batch")
        if len(rows) != len(ordered):
            raise ValueError("predictor returned the wrong number of prompt rows")
        scores: dict[PredictionKey, float] = {}
        for example, row in zip(ordered, rows, strict=True):
            if not isinstance(row, Mapping):
                raise ValueError("predictor rows must be model-score mappings")
            if set(row) != set(model_ids):
                raise ValueError("predictor must return every candidate model exactly")
            for model_id in model_ids:
                scores[(example.example_id, model_id)] = _prediction_float(row[model_id])
        return cls(scores)


def cross_fitted_prediction_table(
    examples: Sequence[EvaluationExample],
    train_predictor: PredictorTrainer,
) -> CrossFittedPredictionTable:
    """Predict every row once from a model fitted without that row's domain."""

    ordered = _ordered_examples(examples)
    scores: dict[PredictionKey, float] = {}
    for fold in leave_one_domain_out(ordered):
        predictor = train_predictor(fold.training)
        held_out = CrossFittedPredictionTable.from_predictor(fold.test, predictor)
        overlap = set(scores) & set(held_out.scores)
        if overlap:
            raise AssertionError(f"cross-fitted predictions overlap: {sorted(overlap)}")
        scores.update(held_out.scores)
    result = CrossFittedPredictionTable(scores)
    result.validate_examples(ordered)
    return result


def exact_lambda_candidates(
    examples: Sequence[EvaluationExample],
    tier_spec: TierSpec,
    predictions: CrossFittedPredictionTable,
) -> tuple[Fraction, ...]:
    """Return boundaries and one representative for every exact policy interval.

    Each model utility is an affine function of lambda, so its ordering can change
    only at a pairwise root. Between adjacent roots the first query's decision is
    constant; the realized charge and next remaining budget are therefore constant,
    and the same argument applies inductively to every later cumulative-ledger query.
    Boundaries preserve exact tie-break behavior, midpoints cover open intervals, and
    one value above the final root covers the unbounded tail.
    """

    if not isinstance(tier_spec, TierSpec):
        raise TypeError("tier_spec must be a TierSpec")
    candidates, _ = _materialize_exact_candidates(examples, predictions)
    return candidates


def _breakpoints(
    examples: tuple[EvaluationExample, ...],
    predictions: CrossFittedPredictionTable,
) -> Iterator[Fraction]:
    """Yield every non-negative pairwise root without retaining the full set."""

    for example in examples:
        scores = predictions.for_example(
            example.example_id,
            tuple(model.model_id for model in example.candidate_models),
        )
        # A ledger adapter defines the meaning of ``budget_limit`` and may expose a
        # larger initial balance (for example, a pooled per-query allowance). Derive
        # a safe superset from the full catalogue; the simulator remains the sole
        # authority on runtime affordability.
        points = tuple(
            (
                model.model_id,
                Fraction(model.cost),
                _quality_fraction(scores[model.model_id]),
            )
            for model in example.candidate_models
        )
        for index, (_, left_cost, left_quality) in enumerate(points):
            for _, right_cost, right_quality in points[index + 1 :]:
                if left_cost == right_cost:
                    continue
                root = (left_quality - right_quality) / (left_cost - right_cost)
                if root >= 0:
                    yield root


def _candidates_from_boundaries(boundaries: set[Fraction]) -> tuple[Fraction, ...]:
    boundaries.add(Fraction(0))
    ordered_boundaries = sorted(boundaries)
    candidates = set(ordered_boundaries)
    candidates.update((left + right) / 2 for left, right in pairwise(ordered_boundaries))
    candidates.add(ordered_boundaries[-1] + 1)
    return tuple(sorted(candidates))


def _materialize_exact_candidates(
    examples: Sequence[EvaluationExample],
    predictions: CrossFittedPredictionTable,
) -> tuple[tuple[Fraction, ...], int]:
    ordered = _ordered_examples(examples)
    predictions.validate_examples(ordered)
    boundaries = {Fraction(0)}
    observed = 0
    for root in _breakpoints(ordered, predictions):
        observed += 1
        boundaries.add(root)
    return _candidates_from_boundaries(boundaries), observed


def _breakpoint_priority(value: Fraction) -> tuple[int, int, int]:
    document = f"{value.numerator}/{value.denominator}".encode()
    digest = int.from_bytes(hashlib.sha256(document).digest(), "big")
    return digest, value.numerator, value.denominator


def _rank_spaced(
    values: tuple[Fraction, ...],
    maximum: int,
) -> tuple[Fraction, ...]:
    if len(values) <= maximum:
        return values
    last = len(values) - 1
    denominator = maximum - 1
    indices = tuple((index * last) // denominator for index in range(maximum))
    retained = tuple(values[index] for index in indices)
    if len(retained) != len(set(retained)):
        raise AssertionError("rank-spaced candidate selection must remain unique")
    return retained


def _bounded_candidates(
    examples: Sequence[EvaluationExample],
    predictions: CrossFittedPredictionTable,
    maximum: int,
) -> tuple[tuple[Fraction, ...], int, bool]:
    """Retain a deterministic bottom-hash sample of unique roots in bounded memory."""

    ordered = _ordered_examples(examples)
    predictions.validate_examples(ordered)
    # Entries negate the priority fields so heap[0] is the worst retained value.
    heap: list[tuple[int, int, int, Fraction]] = []
    retained: set[Fraction] = set()
    minimum: Fraction | None = None
    maximum_root: Fraction | None = None
    observed = 0
    overflowed = False
    for root in _breakpoints(ordered, predictions):
        observed += 1
        minimum = root if minimum is None or root < minimum else minimum
        maximum_root = root if maximum_root is None or root > maximum_root else maximum_root
        if root in retained:
            continue
        priority = _breakpoint_priority(root)
        entry = (-priority[0], -priority[1], -priority[2], root)
        if len(heap) < maximum:
            heappush(heap, entry)
            retained.add(root)
            continue
        overflowed = True
        worst = heap[0]
        worst_priority = (-worst[0], -worst[1], -worst[2])
        if priority < worst_priority:
            removed = heapreplace(heap, entry)[3]
            retained.remove(removed)
            retained.add(root)

    boundaries = {Fraction(0), *retained}
    if minimum is not None:
        boundaries.add(minimum)
    if maximum_root is not None:
        boundaries.add(maximum_root)
    derived = _candidates_from_boundaries(boundaries)
    truncated = overflowed or len(derived) > maximum
    return _rank_spaced(derived, maximum), observed, not truncated


@dataclass(frozen=True, slots=True)
class LambdaCandidateSet:
    """One tier's exact, bounded-memory, or caller-supplied search values."""

    tier: BudgetTier
    values: tuple[Fraction, ...]
    total_derived_values: int | None
    exhaustive: bool
    strategy: str = "exhaustive-breakpoints-v1"
    observed_breakpoint_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tier, BudgetTier):
            raise TypeError("candidate-set tier must be a BudgetTier")
        if not self.values or self.values != tuple(sorted(set(self.values))):
            raise ValueError("lambda candidate values must be non-empty, sorted, and unique")
        if any(not isinstance(value, Fraction) or value < 0 for value in self.values):
            raise ValueError("lambda candidate values must be non-negative Fractions")
        if self.total_derived_values is not None and (
            isinstance(self.total_derived_values, bool)
            or not isinstance(self.total_derived_values, int)
        ):
            raise TypeError("total_derived_values must be an integer or None")
        if self.total_derived_values is not None and self.total_derived_values < len(self.values):
            raise ValueError("total_derived_values cannot be smaller than retained values")
        if not isinstance(self.exhaustive, bool):
            raise TypeError("exhaustive must be a boolean")
        if self.exhaustive and self.total_derived_values != len(self.values):
            raise ValueError("an exhaustive candidate set must retain every derived value")
        if not isinstance(self.strategy, str) or self.strategy not in {
            "bounded-bottom-hash-v1",
            "exhaustive-breakpoints-v1",
            "explicit-grid-v1",
        }:
            raise ValueError("unknown lambda candidate derivation strategy")
        if isinstance(self.observed_breakpoint_count, bool) or not isinstance(
            self.observed_breakpoint_count, int
        ):
            raise TypeError("observed_breakpoint_count must be an integer")
        if self.observed_breakpoint_count < 0:
            raise ValueError("observed_breakpoint_count must be non-negative")
        if self.strategy == "exhaustive-breakpoints-v1" and not self.exhaustive:
            raise ValueError("exhaustive-breakpoints strategy must be exhaustive")
        if self.strategy == "bounded-bottom-hash-v1" and (
            self.exhaustive or self.total_derived_values is not None
        ):
            raise ValueError(
                "bounded-bottom-hash strategy must be non-exhaustive with unknown total"
            )
        if self.strategy == "explicit-grid-v1" and (
            self.exhaustive
            or self.total_derived_values != len(self.values)
            or self.observed_breakpoint_count != 0
        ):
            raise ValueError("explicit-grid strategy must report only its supplied values")


def derive_lambda_candidate_set(
    examples: Sequence[EvaluationExample],
    tier_spec: TierSpec,
    predictions: CrossFittedPredictionTable,
    *,
    max_candidates: int | None = None,
) -> LambdaCandidateSet:
    """Derive the complete exact set or a bounded-memory approximation."""

    if max_candidates is not None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
            raise TypeError("max_candidates must be an integer or None")
        if max_candidates < 2:
            raise ValueError("max_candidates must be at least two")
    if max_candidates is None:
        retained, observed = _materialize_exact_candidates(examples, predictions)
        exhaustive = True
        total = len(retained)
        strategy = "exhaustive-breakpoints-v1"
    else:
        retained, observed, exhaustive = _bounded_candidates(
            examples,
            predictions,
            max_candidates,
        )
        total = len(retained) if exhaustive else None
        strategy = "exhaustive-breakpoints-v1" if exhaustive else "bounded-bottom-hash-v1"
    return LambdaCandidateSet(
        tier=tier_spec.tier,
        values=retained,
        total_derived_values=total,
        exhaustive=exhaustive,
        strategy=strategy,
        observed_breakpoint_count=observed,
    )


@dataclass(frozen=True, slots=True)
class _PredictionTableLambdaRouter(PrivilegedEvaluationRouter):
    predictions: CrossFittedPredictionTable
    lambda_by_tier: Mapping[BudgetTier, Fraction]

    def route(self, state: RouterState) -> RouterAction:
        del state
        raise RoutingContractError(
            "cross-fitted prediction routing is evaluation-only; use OfflineSimulator"
        )

    def route_with_evaluation_context(self, state: RouterState, *, example_id: str) -> RouterAction:
        if state.call_history:
            return SelectOutput(len(state.call_history) - 1, reason="one-shot call completed")
        try:
            lambda_cost = self.lambda_by_tier[state.budget_tier]
        except KeyError as error:
            raise RoutingContractError(
                f"no tuned lambda for tier {state.budget_tier.value!r}"
            ) from error
        model_ids = tuple(
            model.model_id
            for model in state.candidate_models
            if model.cost <= state.remaining_budget
        )
        if not model_ids:
            return route_from_predictions(state, {}, lambda_cost)
        scores = self.predictions.for_example(example_id, model_ids)
        return route_from_predictions(state, scores, lambda_cost)


def _mean_quality_fraction(result: TierResult) -> Fraction | None:
    if not result.feasible or any(query.quality is None for query in result.queries):
        return None
    values = tuple(
        _quality_fraction(query.quality) for query in result.queries if query.quality is not None
    )
    return sum(values, start=Fraction(0)) / len(values)


@dataclass(frozen=True, slots=True)
class TierLambdaSelection:
    """Best fully feasible lambda and its replay evidence for one tier."""

    tier: BudgetTier
    lambda_cost: Fraction
    mean_quality: float
    realized_cost: Decimal
    candidates: LambdaCandidateSet
    report: TierResult

    def __post_init__(self) -> None:
        if not isinstance(self.tier, BudgetTier):
            raise TypeError("selected tier must be a BudgetTier")
        if not isinstance(self.lambda_cost, Fraction) or self.lambda_cost < 0:
            raise ValueError("selected lambda must be a non-negative Fraction")
        if (
            isinstance(self.mean_quality, bool)
            or not isinstance(self.mean_quality, (int, float))
            or not math.isfinite(self.mean_quality)
        ):
            raise ValueError("selected mean quality must be finite")
        if (
            not isinstance(self.realized_cost, Decimal)
            or not self.realized_cost.is_finite()
            or self.realized_cost < 0
        ):
            raise ValueError("selected realized cost must be a finite non-negative Decimal")
        if not isinstance(self.candidates, LambdaCandidateSet):
            raise TypeError("selected candidates must be a LambdaCandidateSet")
        if self.candidates.tier is not self.tier:
            raise ValueError("selected candidates must match the selected tier")
        if self.lambda_cost not in self.candidates.values:
            raise ValueError("selected lambda must occur in the candidate set")
        if not isinstance(self.report, TierResult) or self.report.tier_spec.tier is not self.tier:
            raise ValueError("selected report must match the selected tier")
        exact_quality = _mean_quality_fraction(self.report)
        if exact_quality is None or float(exact_quality) != float(self.mean_quality):
            raise ValueError("selected report must be feasible and match mean quality")
        if self.report.budget.spent != self.realized_cost:
            raise ValueError("selected report must match realized cost")
        object.__setattr__(self, "mean_quality", float(self.mean_quality))


@dataclass(frozen=True, slots=True)
class TierLambdaTuningResult:
    """Independent per-tier optima and their direct weighted-score replay."""

    selections: tuple[TierLambdaSelection, ...]
    report: EvaluationReport
    score: ScoreSummary
    data_sha256: str
    replay_sha256: str
    prediction_sha256: str
    example_count: int
    domains: tuple[str, ...]

    def __post_init__(self) -> None:
        selections = tuple(self.selections)
        tiers = tuple(selection.tier for selection in selections)
        if not tiers or len(tiers) != len(set(tiers)):
            raise ValueError("lambda selections must contain unique tiers")
        if tuple(result.tier_spec.tier for result in self.report.tiers) != tiers:
            raise ValueError("lambda selections and report tiers must align")
        if tuple(selection.report for selection in selections) != self.report.tiers:
            raise ValueError("lambda selections must retain the combined report tiers")
        if self.score != summarize_report(self.report):
            raise ValueError("lambda tuning score must summarize its report")
        for name in ("data_sha256", "replay_sha256", "prediction_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if isinstance(self.example_count, bool) or not isinstance(self.example_count, int):
            raise TypeError("example_count must be an integer")
        if self.example_count < 1:
            raise ValueError("example_count must be positive")
        domains = tuple(self.domains)
        if not domains or domains != tuple(sorted(set(domains))):
            raise ValueError("domains must be non-empty, sorted, and unique")
        if any(not isinstance(domain, str) or not domain.strip() for domain in domains):
            raise ValueError("domains must contain non-empty strings")
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "domains", domains)

    @property
    def lambda_by_tier(self) -> Mapping[BudgetTier, Fraction]:
        """Return an immutable deployment mapping."""

        return MappingProxyType(
            {selection.tier: selection.lambda_cost for selection in self.selections}
        )


def _candidate_map(
    tier_specs: tuple[TierSpec, ...],
    configured: Mapping[BudgetTier, Sequence[LambdaInput]] | None,
    examples: tuple[EvaluationExample, ...],
    predictions: CrossFittedPredictionTable,
    max_candidates_per_tier: int | None,
) -> Mapping[BudgetTier, LambdaCandidateSet]:
    tiers = {spec.tier for spec in tier_specs}
    if configured is not None and set(configured) != tiers:
        raise ValueError("configured lambda grids must match tier_specs exactly")
    result: dict[BudgetTier, LambdaCandidateSet] = {}
    shared: LambdaCandidateSet | None = None
    if configured is None:
        # Breakpoints depend only on the examples, model catalogue, and predictions.
        # Tier budgets affect simulator feasibility, not the affine utility roots, so
        # derive this potentially expensive stream once and reuse its exact values.
        shared = derive_lambda_candidate_set(
            examples,
            tier_specs[0],
            predictions,
            max_candidates=max_candidates_per_tier,
        )
    for spec in tier_specs:
        if shared is not None:
            candidate_set = LambdaCandidateSet(
                tier=spec.tier,
                values=shared.values,
                total_derived_values=shared.total_derived_values,
                exhaustive=shared.exhaustive,
                strategy=shared.strategy,
                observed_breakpoint_count=shared.observed_breakpoint_count,
            )
        else:
            assert configured is not None
            candidates = tuple(sorted({as_lambda(value) for value in configured[spec.tier]}))
            if not candidates:
                raise ValueError(f"lambda grid for {spec.tier.value} must not be empty")
            candidate_set = LambdaCandidateSet(
                tier=spec.tier,
                values=candidates,
                total_derived_values=len(candidates),
                exhaustive=False,
                strategy="explicit-grid-v1",
            )
        result[spec.tier] = candidate_set
    return MappingProxyType(result)


def tune_tier_lambdas(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
    predictions: CrossFittedPredictionTable,
    ledger_factory: BudgetLedgerFactory,
    *,
    lambda_grids: Mapping[BudgetTier, Sequence[LambdaInput]] | None = None,
    max_candidates_per_tier: int | None = None,
    allow_large_exhaustive: bool = False,
) -> TierLambdaTuningResult:
    """Tune each tier's realized quality through the real simulator.

    Tiers have independent ledgers and strictly positive weights, so independent tier
    maxima exactly maximize the existing weighted-tier objective without a Cartesian
    product. Infeasible candidates never participate and query order is preserved.
    The guarantee is exhaustive when ``max_candidates_per_tier`` is ``None``. A cap
    streams every root into a deterministic bounded-memory sample, then rank-spaces
    the small derived candidate set; metadata labels that approximation explicitly.
    Large exhaustive upper bounds fail before root materialization unless the caller
    explicitly sets ``allow_large_exhaustive=True`` after reviewing resource needs.
    """

    ordered = _ordered_examples(examples)
    specs = _tier_specs(tier_specs)
    if not isinstance(allow_large_exhaustive, bool):
        raise TypeError("allow_large_exhaustive must be a boolean")
    if lambda_grids is not None and max_candidates_per_tier is not None:
        raise ValueError("lambda_grids and max_candidates_per_tier are mutually exclusive")
    model_ids = _model_ids(ordered)
    predictions.validate_examples(ordered)
    if lambda_grids is None and max_candidates_per_tier is None:
        _validate_exhaustive_search_size(
            ordered,
            specs,
            len(model_ids),
            allow_large_exhaustive=allow_large_exhaustive,
        )
    candidates_by_tier = _candidate_map(
        specs,
        lambda_grids,
        ordered,
        predictions,
        max_candidates_per_tier,
    )
    simulator = OfflineSimulator(ledger_factory)
    selections = []
    for spec in specs:
        best: tuple[tuple[Fraction, Decimal, Fraction], TierResult] | None = None
        candidate_set = candidates_by_tier[spec.tier]
        for lambda_cost in candidate_set.values:
            router = _PredictionTableLambdaRouter(
                predictions,
                MappingProxyType({spec.tier: lambda_cost}),
            )
            report = simulator.run_tier(router, ordered, spec)
            quality = _mean_quality_fraction(report)
            if quality is None:
                continue
            ranking = (-quality, report.budget.spent, lambda_cost)
            if best is None or ranking < best[0]:
                best = (ranking, report)
        if best is None:
            raise ValueError(f"no fully feasible lambda for tier {spec.tier.value!r}")
        ranking, report = best
        quality = -ranking[0]
        selections.append(
            TierLambdaSelection(
                tier=spec.tier,
                lambda_cost=ranking[2],
                mean_quality=float(quality),
                realized_cost=report.budget.spent,
                candidates=candidate_set,
                report=report,
            )
        )

    combined = EvaluationReport(
        "tier-lambda-tuning",
        tuple(selection.report for selection in selections),
    )
    score = summarize_report(combined)
    if score.weighted_quality is None:
        raise AssertionError("selected fully feasible tiers must have a weighted score")
    return TierLambdaTuningResult(
        selections=tuple(selections),
        report=combined,
        score=score,
        data_sha256=evaluation_data_sha256(ordered),
        replay_sha256=evaluation_replay_sha256(ordered),
        prediction_sha256=predictions.sha256(ordered),
        example_count=len(ordered),
        domains=tuple(sorted({example.domain for example in ordered})),
    )


@dataclass(frozen=True, slots=True)
class TunedLambdaRouterForFold:
    """A final predictor/router fitted without an outer held-out domain."""

    held_out_domain: str
    router: TieredLambdaRouter
    tuning: TierLambdaTuningResult

    def __post_init__(self) -> None:
        if not isinstance(self.held_out_domain, str) or not self.held_out_domain.strip():
            raise ValueError("held_out_domain must be a non-empty string")
        if not isinstance(self.router, TieredLambdaRouter):
            raise TypeError("router must be a TieredLambdaRouter")
        if not isinstance(self.tuning, TierLambdaTuningResult):
            raise TypeError("tuning must be a TierLambdaTuningResult")
        if self.router.lambda_by_tier != self.tuning.lambda_by_tier:
            raise ValueError("router lambdas must match fold tuning evidence")
        if self.held_out_domain in self.tuning.domains:
            raise ValueError("held-out domain must not occur in fold tuning data")


def fit_tiered_lambda_router_for_fold(
    fold: DomainFold,
    tier_specs: Sequence[TierSpec],
    train_predictor: PredictorTrainer,
    ledger_factory: BudgetLedgerFactory,
    *,
    max_candidates_per_tier: int | None = None,
    allow_large_exhaustive: bool = False,
) -> TunedLambdaRouterForFold:
    """Tune on inner-LODO predictions, then refit only on outer training rows.

    Search caps and explicit large-exhaustive acknowledgement are forwarded unchanged
    to :func:`tune_tier_lambdas`.
    """

    predictions = cross_fitted_prediction_table(fold.training, train_predictor)
    tuning = tune_tier_lambdas(
        fold.training,
        tier_specs,
        predictions,
        ledger_factory,
        max_candidates_per_tier=max_candidates_per_tier,
        allow_large_exhaustive=allow_large_exhaustive,
    )
    final_predictor = train_predictor(fold.training)
    return TunedLambdaRouterForFold(
        held_out_domain=fold.held_out_domain,
        router=TieredLambdaRouter(final_predictor, tuning.lambda_by_tier),
        tuning=tuning,
    )


@dataclass(frozen=True, slots=True)
class OuterFoldLambdaResult:
    """Inner tuning evidence retained for one untouched outer domain."""

    held_out_domain: str
    training_example_ids: tuple[str, ...]
    test_example_ids: tuple[str, ...]
    tuning: TierLambdaTuningResult

    def __post_init__(self) -> None:
        if not isinstance(self.held_out_domain, str) or not self.held_out_domain.strip():
            raise ValueError("held_out_domain must be a non-empty string")
        training_ids = tuple(self.training_example_ids)
        test_ids = tuple(self.test_example_ids)
        if (
            not training_ids
            or not test_ids
            or len(training_ids) != len(set(training_ids))
            or len(test_ids) != len(set(test_ids))
            or set(training_ids) & set(test_ids)
        ):
            raise ValueError("outer fold IDs must be non-empty, unique, and disjoint")
        if not isinstance(self.tuning, TierLambdaTuningResult):
            raise TypeError("tuning must be a TierLambdaTuningResult")
        if self.held_out_domain in self.tuning.domains:
            raise ValueError("held-out domain must not occur in outer-fold tuning data")
        if any(
            tuple(query.example_id for query in tier.queries) != training_ids
            for tier in self.tuning.report.tiers
        ):
            raise ValueError("outer-fold training IDs must match tuning report order")
        object.__setattr__(self, "training_example_ids", training_ids)
        object.__setattr__(self, "test_example_ids", test_ids)


@dataclass(frozen=True, slots=True)
class NestedLodoLambdaResult:
    """Global outer-OOF replay with per-example schedules and original ordering."""

    folds: tuple[OuterFoldLambdaResult, ...]
    report: EvaluationReport
    score: ScoreSummary
    prediction_sha256: str

    def __post_init__(self) -> None:
        folds = tuple(self.folds)
        held_out_domains = tuple(fold.held_out_domain for fold in folds)
        if not folds or len(held_out_domains) != len(set(held_out_domains)):
            raise ValueError("nested LODO folds must contain unique held-out domains")
        if self.score != summarize_report(self.report):
            raise ValueError("nested LODO score must summarize its report")
        if (
            not isinstance(self.prediction_sha256, str)
            or len(self.prediction_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.prediction_sha256)
        ):
            raise ValueError("prediction_sha256 must be lowercase SHA-256 hex")
        held_out_ids = tuple(example_id for fold in folds for example_id in fold.test_example_ids)
        if len(held_out_ids) != len(set(held_out_ids)):
            raise ValueError("nested LODO test IDs must be covered exactly once")
        expected_ids = set(held_out_ids)
        if any(
            set(query.example_id for query in tier.queries) != expected_ids
            for tier in self.report.tiers
        ):
            raise ValueError("nested LODO folds and replay report must cover identical IDs")
        object.__setattr__(self, "folds", folds)


@dataclass(frozen=True, slots=True)
class _PerExampleLambdaRouter(PrivilegedEvaluationRouter):
    predictions: CrossFittedPredictionTable
    lambdas: Mapping[tuple[BudgetTier, str], Fraction]

    def route(self, state: RouterState) -> RouterAction:
        del state
        raise RoutingContractError("nested LODO routing is evaluation-only")

    def route_with_evaluation_context(self, state: RouterState, *, example_id: str) -> RouterAction:
        if state.call_history:
            return SelectOutput(len(state.call_history) - 1, reason="one-shot call completed")
        try:
            lambda_cost = self.lambdas[(state.budget_tier, example_id)]
        except KeyError as error:
            raise RoutingContractError(
                f"nested LODO schedule has no lambda for {state.budget_tier.value}/{example_id}"
            ) from error
        affordable_ids = tuple(
            model.model_id
            for model in state.candidate_models
            if model.cost <= state.remaining_budget
        )
        if not affordable_ids:
            return route_from_predictions(state, {}, lambda_cost)
        scores = self.predictions.for_example(example_id, affordable_ids)
        return route_from_predictions(state, scores, lambda_cost)


def nested_lodo_lambda_evaluation(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
    train_predictor: PredictorTrainer,
    ledger_factory: BudgetLedgerFactory,
    *,
    max_candidates_per_tier: int | None = None,
    allow_large_exhaustive: bool = False,
) -> NestedLodoLambdaResult:
    """Run true nested LODO and replay all outer predictions once in original order.

    Replaying once is required for a cumulative ledger: evaluating each outer fold with
    a fresh ledger would change the accounting problem and produce invalid evidence.
    The injected ledger factory receives each replay's query count and alone decides
    whether a configured limit is fixed-total or scales with population size; this
    orchestration never guesses or rescales unresolved challenge semantics.
    Search caps and large-exhaustive acknowledgement apply independently inside every
    outer fold before the final original-order replay.
    """

    ordered = _ordered_examples(examples)
    specs = _tier_specs(tier_specs)
    if len({example.domain for example in ordered}) < 3:
        raise ValueError("nested LODO lambda evaluation requires at least three domains")
    outer_predictions: dict[PredictionKey, float] = {}
    per_example_lambdas: dict[tuple[BudgetTier, str], Fraction] = {}
    fold_results = []
    for fold in leave_one_domain_out(ordered):
        inner_predictions = cross_fitted_prediction_table(fold.training, train_predictor)
        tuning = tune_tier_lambdas(
            fold.training,
            specs,
            inner_predictions,
            ledger_factory,
            max_candidates_per_tier=max_candidates_per_tier,
            allow_large_exhaustive=allow_large_exhaustive,
        )
        final_predictor = train_predictor(fold.training)
        held_out_predictions = CrossFittedPredictionTable.from_predictor(
            fold.test,
            final_predictor,
        )
        overlap = set(outer_predictions) & set(held_out_predictions.scores)
        if overlap:
            raise AssertionError(f"outer LODO predictions overlap: {sorted(overlap)}")
        outer_predictions.update(held_out_predictions.scores)
        for example in fold.test:
            for tier, lambda_cost in tuning.lambda_by_tier.items():
                per_example_lambdas[(tier, example.example_id)] = lambda_cost
        fold_results.append(
            OuterFoldLambdaResult(
                held_out_domain=fold.held_out_domain,
                training_example_ids=tuple(example.example_id for example in fold.training),
                test_example_ids=tuple(example.example_id for example in fold.test),
                tuning=tuning,
            )
        )

    predictions = CrossFittedPredictionTable(outer_predictions)
    predictions.validate_examples(ordered)
    expected_schedule = {(spec.tier, example.example_id) for spec in specs for example in ordered}
    if set(per_example_lambdas) != expected_schedule:
        raise AssertionError("nested LODO schedule does not cover every tier/example pair")
    router = _PerExampleLambdaRouter(
        predictions,
        MappingProxyType(per_example_lambdas),
    )
    report = OfflineSimulator(ledger_factory).run(
        router,
        ordered,
        specs,
        router_name="nested-lodo-tier-lambda",
    )
    return NestedLodoLambdaResult(
        folds=tuple(fold_results),
        report=report,
        score=summarize_report(report),
        prediction_sha256=predictions.sha256(ordered),
    )
