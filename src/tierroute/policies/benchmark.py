# SPDX-License-Identifier: Apache-2.0
"""Reportable per-query nested-LODO evaluation for the learned router."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from tierroute.adapters.budgets import PerQueryBudgetLedger
from tierroute.core.costs import scale_cost, sum_costs
from tierroute.core.schemas import Cost, ModelSpec
from tierroute.eval.metrics import QuoteErrorReport, oracle_gap_recovery, summarize_quote_error
from tierroute.eval.provenance import evaluation_data_sha256, evaluation_replay_sha256
from tierroute.eval.schemas import (
    EvaluationExample,
    TierSpec,
)
from tierroute.policies.baseline_evaluation import (
    BASELINE_NAMES,
    BaselineResult,
    LodoSixBaselineEvaluation,
    evaluate_per_query_lodo_baselines,
)
from tierroute.policies.lambda_tuning import (
    NestedLodoLambdaResult,
    nested_lodo_lambda_evaluation,
)
from tierroute.predictors.base import QualityPredictor
from tierroute.predictors.training import (
    BilinearTrainingConfig,
    fit_calibrated_bilinear,
)

FOLD_MEMBERSHIP_HASH_ALGORITHM = "tierroute-fold-membership-sha256-v1"


class _DigestWriter(Protocol):
    def update(self, value: bytes) -> None: ...


def _stable_catalogue(examples: tuple[EvaluationExample, ...]) -> tuple[ModelSpec, ...]:
    """Return one canonical model catalogue or reject quote drift between rows."""

    if not examples:
        raise ValueError("benchmark examples must not be empty")
    reference = {model.model_id: model.cost for model in examples[0].candidate_models}
    for example in examples[1:]:
        current = {model.model_id: model.cost for model in example.candidate_models}
        if current != reference:
            raise ValueError("benchmark requires a stable model catalogue and quoted costs")
    return tuple(sorted(examples[0].candidate_models, key=lambda model: model.model_id))


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _update_hash_text(digest: _DigestWriter, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("fold membership text must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("fold membership text must be valid Unicode") from error
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _fold_membership_sha256(
    held_out_domain: str,
    training_example_ids: tuple[str, ...],
    test_example_ids: tuple[str, ...],
) -> str:
    """Hash one ordered fold with a versioned, unambiguous byte contract."""

    digest = hashlib.sha256()
    digest.update(FOLD_MEMBERSHIP_HASH_ALGORITHM.encode("ascii"))
    _update_hash_text(digest, held_out_domain)
    digest.update(len(training_example_ids).to_bytes(8, "big"))
    for example_id in training_example_ids:
        _update_hash_text(digest, example_id)
    digest.update(len(test_example_ids).to_bytes(8, "big"))
    for example_id in test_example_ids:
        _update_hash_text(digest, example_id)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OuterFoldMembershipDigest:
    """Compact evidence for one exact ordered outer-fold membership."""

    held_out_domain: str
    training_example_count: int
    test_example_count: int
    sha256: str
    algorithm: str = field(default=FOLD_MEMBERSHIP_HASH_ALGORITHM, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.held_out_domain, str) or not self.held_out_domain.strip():
            raise ValueError("held_out_domain must be a non-empty string")
        for name in ("training_example_count", "test_example_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        _require_sha256(self.sha256, "fold membership sha256")


def _fold_membership_digests(
    learned: NestedLodoLambdaResult,
) -> tuple[OuterFoldMembershipDigest, ...]:
    return tuple(
        OuterFoldMembershipDigest(
            held_out_domain=fold.held_out_domain,
            training_example_count=len(fold.training_example_ids),
            test_example_count=len(fold.test_example_ids),
            sha256=_fold_membership_sha256(
                fold.held_out_domain,
                fold.training_example_ids,
                fold.test_example_ids,
            ),
        )
        for fold in learned.folds
    )


@dataclass(frozen=True, slots=True)
class PerQueryNestedLodoBenchmark:
    """Learned-policy result aligned with all six per-query baseline reports."""

    learned: NestedLodoLambdaResult
    baselines: LodoSixBaselineEvaluation
    data_sha256: str
    replay_sha256: str
    training_config: BilinearTrainingConfig
    learned_gap_recovery: float | None = field(init=False)
    learned_total_cost: Cost = field(init=False)
    learned_quote_error: QuoteErrorReport = field(init=False)
    example_count: int = field(init=False)
    domains: tuple[str, ...] = field(init=False)
    model_ids: tuple[str, ...] = field(init=False)
    fold_memberships: tuple[OuterFoldMembershipDigest, ...] = field(init=False)
    accounting_scope: str = field(default="per-query", init=False)
    predictor_kind: str = field(default="calibrated-bilinear-surface-v1", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.learned, NestedLodoLambdaResult):
            raise TypeError("learned must be a NestedLodoLambdaResult")
        if not isinstance(self.baselines, LodoSixBaselineEvaluation):
            raise TypeError("baselines must be a LodoSixBaselineEvaluation")
        if not isinstance(self.training_config, BilinearTrainingConfig):
            raise TypeError("training_config must be a BilinearTrainingConfig")
        if self.learned.report.router_name != "nested-lodo-tier-lambda":
            raise ValueError("learned report must use the nested LODO router identity")
        if tuple(row.name for row in self.baselines.baselines) != BASELINE_NAMES:
            raise ValueError("benchmark must contain all six canonical baselines")

        scopes = {
            self.learned.report.evaluation_scope,
            *(row.report.evaluation_scope for row in self.baselines.baselines),
        }
        if len(scopes) != 1:
            raise ValueError("learned and baseline reports must share one evaluation scope")

        learned_folds = tuple(
            (
                fold.held_out_domain,
                fold.training_example_ids,
                fold.test_example_ids,
            )
            for fold in self.learned.folds
        )
        baseline_folds = tuple(
            (
                fold.held_out_domain,
                fold.training_example_ids,
                fold.test_example_ids,
            )
            for fold in self.baselines.folds
        )
        if learned_folds != baseline_folds:
            raise ValueError("learned and baseline outer folds must match exactly")

        expected_ids = self.baselines.example_ids
        reference_report = self.baselines.baselines[0].report
        expected_specs = tuple(tier.tier_spec for tier in reference_report.tiers)
        expected_tiers = tuple(spec.tier for spec in expected_specs)
        if tuple(tier.tier_spec for tier in self.learned.report.tiers) != expected_specs:
            raise ValueError("learned and baseline tier specifications must match exactly")
        held_out_domains = tuple(fold.held_out_domain for fold in self.learned.folds)
        for fold in self.learned.folds:
            if fold.tuning.example_count != len(fold.training_example_ids):
                raise ValueError("fold tuning count must match its outer training membership")
            expected_training_domains = tuple(
                domain for domain in held_out_domains if domain != fold.held_out_domain
            )
            if fold.tuning.domains != expected_training_domains:
                raise ValueError("fold tuning domains must exclude only its outer holdout")
            if tuple(selection.tier for selection in fold.tuning.selections) != expected_tiers:
                raise ValueError("fold tuning tiers must match the learned replay tiers")
        for tier_result in self.learned.report.tiers:
            query_ids = tuple(query.example_id for query in tier_result.queries)
            if query_ids != expected_ids or tier_result.budget.query_order != expected_ids:
                raise ValueError("learned replay must preserve the baseline example order")
            if tier_result.budget.adapter_name != "per-query":
                raise ValueError("benchmark learned report must use per-query accounting")
            expected_limit = scale_cost(tier_result.tier_spec.budget_limit, len(expected_ids))
            if tier_result.budget.effective_total_limit != expected_limit:
                raise ValueError("per-query effective limit must scale by query count")
            if tier_result.budget.spent != sum_costs(query.cost for query in tier_result.queries):
                raise ValueError("learned tier spend must equal replayed query costs")

        by_name = {row.name: row for row in self.baselines.baselines}
        expected_gap = oracle_gap_recovery(
            self.learned.report,
            by_name["always-cheapest"].report,
            by_name["oracle"].report,
        )
        if expected_gap is not None:
            if type(expected_gap) is not float:
                raise TypeError("derived learned gap recovery must be a float or None")
            if not math.isfinite(expected_gap):
                raise ValueError("derived learned gap recovery must be finite")
        expected_cost = sum_costs(
            query.cost for tier in self.learned.report.tiers for query in tier.queries
        )
        ModelSpec("learned-benchmark-total-cost", expected_cost)
        expected_quote_error = summarize_quote_error(self.learned.report)

        oracle_by_tier = by_name["oracle"].report.by_tier()
        for tier, learned_tier in self.learned.report.by_tier().items():
            oracle_queries = {query.example_id: query for query in oracle_by_tier[tier].queries}
            for query in learned_tier.queries:
                oracle_query = oracle_queries[query.example_id]
                if not oracle_query.feasible or oracle_query.quality is None:
                    raise ValueError("per-query oracle must be feasible and complete")
                if (
                    query.feasible
                    and query.quality is not None
                    and query.quality > oracle_query.quality + 1e-12
                ):
                    raise ValueError("learned query quality cannot exceed the aligned oracle")

        _require_sha256(self.data_sha256, "data_sha256")
        _require_sha256(self.replay_sha256, "replay_sha256")
        held_out_ids = tuple(
            example_id for fold in self.learned.folds for example_id in fold.test_example_ids
        )
        if tuple(sorted(held_out_ids)) != tuple(sorted(expected_ids)):
            raise ValueError("outer-fold test coverage must match the replay population")
        example_count = len(expected_ids)

        domains = held_out_domains
        if not domains or domains != tuple(sorted(set(domains))):
            raise ValueError("nested LODO held-out domains must be sorted and unique")
        model_ids = self.baselines.candidate_model_ids
        fold_memberships = _fold_membership_digests(self.learned)
        object.__setattr__(self, "learned_gap_recovery", expected_gap)
        object.__setattr__(self, "learned_total_cost", expected_cost)
        object.__setattr__(self, "learned_quote_error", expected_quote_error)
        object.__setattr__(self, "example_count", example_count)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "model_ids", model_ids)
        object.__setattr__(self, "fold_memberships", fold_memberships)

    @property
    def baseline_by_name(self) -> Mapping[str, BaselineResult]:
        """Return an immutable-use mapping of canonical baseline name to result."""

        return MappingProxyType({row.name: row for row in self.baselines.baselines})


def evaluate_per_query_bilinear_benchmark(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
    *,
    config: BilinearTrainingConfig | None = None,
    max_candidates_per_tier: int | None = None,
    allow_large_exhaustive: bool = False,
) -> PerQueryNestedLodoBenchmark:
    """Evaluate calibrated bilinear routing and six baselines on one exact scope.

    Four domains are the minimum for this concrete trainer: each outer training side
    must retain three domains so its calibrated predictor can perform another inner
    LODO fit without collapsing to a one-domain training partition.
    """

    ordered = tuple(examples)
    specs = tuple(tier_specs)
    domains = tuple(sorted({example.domain for example in ordered}))
    if len(domains) < 4:
        raise ValueError("calibrated bilinear nested LODO benchmark requires at least four domains")
    catalogue = _stable_catalogue(ordered)
    if config is None:
        config = BilinearTrainingConfig()
    elif not isinstance(config, BilinearTrainingConfig):
        raise TypeError("config must be a BilinearTrainingConfig or None")

    def train_predictor(training: tuple[EvaluationExample, ...]) -> QualityPredictor:
        return fit_calibrated_bilinear(training, config=config).build_predictor()

    learned = nested_lodo_lambda_evaluation(
        ordered,
        specs,
        train_predictor,
        PerQueryBudgetLedger,
        max_candidates_per_tier=max_candidates_per_tier,
        allow_large_exhaustive=allow_large_exhaustive,
    )
    premium = max(catalogue, key=lambda model: (model.cost, model.model_id)).model_id
    baselines = evaluate_per_query_lodo_baselines(
        ordered,
        specs,
        PerQueryBudgetLedger,
        premium_model_id=premium,
        strong_model_id=premium,
        random_seed=2026,
        character_threshold=120,
    )
    return PerQueryNestedLodoBenchmark(
        learned=learned,
        baselines=baselines,
        data_sha256=evaluation_data_sha256(ordered),
        replay_sha256=evaluation_replay_sha256(ordered),
        training_config=config,
    )
