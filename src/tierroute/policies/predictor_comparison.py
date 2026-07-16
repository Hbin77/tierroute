# SPDX-License-Identifier: Apache-2.0
"""Paired descriptive comparison of the fixed bilinear and GBM predictors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from tierroute.core.schemas import BudgetTier
from tierroute.eval.schemas import EvaluationExample, EvaluationReport, TierSpec
from tierroute.policies.benchmark import (
    BILINEAR_PREDICTOR_KIND,
    GBM_PREDICTOR_KIND,
    PerQueryNestedLodoBenchmark,
    evaluate_per_query_bilinear_benchmark,
    evaluate_per_query_gbm_benchmark,
)
from tierroute.predictors.gbm_training import (
    GbmTrainingConfig,
    preflight_nested_lodo_gbm,
)
from tierroute.predictors.training import BilinearTrainingConfig

COMPARISON_DIRECTION = "gbm-minus-bilinear"
PAIRED_SELECTION_PROTOCOL = "none-paired-estimation"


def _optional_delta(gbm: float | None, bilinear: float | None) -> float | None:
    if gbm is None or bilinear is None:
        return None
    delta = gbm - bilinear
    if not math.isfinite(delta):
        raise ValueError("paired predictor delta must be finite")
    return delta


def _immutable_tier_deltas(
    values: Mapping[BudgetTier, float | None],
) -> Mapping[BudgetTier, float | None]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("tier_quality_delta must be a non-empty mapping")
    copied: dict[BudgetTier, float | None] = {}
    for tier, value in values.items():
        if not isinstance(tier, BudgetTier):
            raise TypeError("tier_quality_delta keys must be BudgetTier values")
        if value is not None:
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("tier quality deltas must be finite floats or None")
        copied[tier] = value
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class HeldOutDomainPredictorDelta:
    """Raw descriptive ``GBM - bilinear`` metrics for one untouched domain."""

    held_out_domain: str
    tier_quality_delta: Mapping[BudgetTier, float | None]
    weighted_quality_delta: float | None
    oracle_gap_recovery_delta: float | None
    comparison_direction: str = field(default=COMPARISON_DIRECTION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.held_out_domain, str) or not self.held_out_domain.strip():
            raise ValueError("held_out_domain must be a non-empty string")
        tier_deltas = _immutable_tier_deltas(self.tier_quality_delta)
        for name in ("weighted_quality_delta", "oracle_gap_recovery_delta"):
            value = getattr(self, name)
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite float or None")
        object.__setattr__(self, "tier_quality_delta", tier_deltas)


def _query_quality_by_tier(
    report: EvaluationReport,
    example_ids: tuple[str, ...],
) -> dict[BudgetTier, float | None]:
    expected = set(example_ids)
    result: dict[BudgetTier, float | None] = {}
    for tier_result in report.tiers:
        selected = tuple(
            query for query in tier_result.queries if query.example_id in expected
        )
        if (
            len(selected) != len(example_ids)
            or {query.example_id for query in selected} != expected
        ):
            raise ValueError("held-out fold IDs must match every compared report")
        if any(not query.feasible or query.quality is None for query in selected):
            result[tier_result.tier_spec.tier] = None
            continue
        mean = sum(query.quality for query in selected if query.quality is not None) / len(
            selected
        )
        if not math.isfinite(mean):
            raise ValueError("held-out domain mean quality must be finite")
        result[tier_result.tier_spec.tier] = mean
    return result


def _weighted_quality(
    qualities: Mapping[BudgetTier, float | None],
    report: EvaluationReport,
) -> float | None:
    if any(value is None for value in qualities.values()):
        return None
    numerator = 0.0
    denominator = 0.0
    for tier_result in report.tiers:
        quality = qualities[tier_result.tier_spec.tier]
        if quality is None:  # pragma: no cover - guarded without redistributing weights
            return None
        numerator += tier_result.tier_spec.weight * quality
        denominator += tier_result.tier_spec.weight
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise ValueError("held-out weighted quality must be finite")
    weighted = numerator / denominator
    if not math.isfinite(weighted):
        raise ValueError("held-out weighted quality must be finite")
    return weighted


def _oracle_gap_recovery(
    router: Mapping[BudgetTier, float | None],
    cheapest: Mapping[BudgetTier, float | None],
    oracle: Mapping[BudgetTier, float | None],
    report: EvaluationReport,
    *,
    tolerance: float = 1e-12,
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for tier_result in report.tiers:
        tier = tier_result.tier_spec.tier
        router_quality = router[tier]
        cheap_quality = cheapest[tier]
        oracle_quality = oracle[tier]
        if router_quality is None or cheap_quality is None or oracle_quality is None:
            return None
        if oracle_quality + tolerance < cheap_quality:
            raise ValueError(f"held-out oracle quality is below cheapest for {tier.value}")
        if router_quality > oracle_quality + tolerance:
            raise ValueError(f"held-out router quality exceeds oracle for {tier.value}")
        weight = tier_result.tier_spec.weight
        numerator += weight * (router_quality - cheap_quality)
        denominator += weight * (oracle_quality - cheap_quality)
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise ValueError("held-out oracle-gap aggregate must be finite")
    if abs(denominator) <= tolerance:
        return None
    recovery = numerator / denominator
    if not math.isfinite(recovery):
        raise ValueError("held-out oracle-gap recovery must be finite")
    return recovery


def _same_pair_scope(
    bilinear: PerQueryNestedLodoBenchmark,
    gbm: PerQueryNestedLodoBenchmark,
) -> None:
    if bilinear.predictor_kind != BILINEAR_PREDICTOR_KIND or type(
        bilinear.training_config
    ) is not BilinearTrainingConfig:
        raise ValueError("bilinear side must use the fixed calibrated bilinear family")
    if gbm.predictor_kind != GBM_PREDICTOR_KIND or type(
        gbm.training_config
    ) is not GbmTrainingConfig:
        raise ValueError("GBM side must use the fixed calibrated regression-stump family")
    if bilinear.baselines is not gbm.baselines:
        raise ValueError("paired predictors must share the exact same baseline object")
    if (
        bilinear.data_sha256 != gbm.data_sha256
        or bilinear.replay_sha256 != gbm.replay_sha256
    ):
        raise ValueError("paired predictors must share data and replay digests")
    if bilinear.accounting_scope != gbm.accounting_scope:
        raise ValueError("paired predictors must share accounting scope")
    if bilinear.example_count != gbm.example_count:
        raise ValueError("paired predictors must share example count")
    if bilinear.domains != gbm.domains:
        raise ValueError("paired predictors must share ordered domains")
    if bilinear.model_ids != gbm.model_ids:
        raise ValueError("paired predictors must share the model catalogue")
    if bilinear.lambda_search_config != gbm.lambda_search_config:
        raise ValueError("paired predictors must share lambda search controls")
    if bilinear.baseline_config != gbm.baseline_config:
        raise ValueError("paired predictors must share baseline configuration")
    if (
        bilinear.baselines.baseline_config_evidence_algorithm
        != gbm.baselines.baseline_config_evidence_algorithm
        or bilinear.baselines.baseline_config_evidence_sha256
        != gbm.baselines.baseline_config_evidence_sha256
    ):
        raise ValueError("paired predictors must share baseline configuration digests")
    if bilinear.fold_memberships != gbm.fold_memberships:
        raise ValueError("paired predictors must share exact outer-fold membership digests")

    bilinear_report = bilinear.learned.report
    gbm_report = gbm.learned.report
    if bilinear_report.evaluation_scope != gbm_report.evaluation_scope:
        raise ValueError("paired predictors must share evaluation scope")
    bilinear_specs = tuple(result.tier_spec for result in bilinear_report.tiers)
    gbm_specs = tuple(result.tier_spec for result in gbm_report.tiers)
    if bilinear_specs != gbm_specs:
        raise ValueError("paired predictors must share ordered tier specifications")
    bilinear_query_orders = tuple(
        tuple(query.example_id for query in result.queries)
        for result in bilinear_report.tiers
    )
    gbm_query_orders = tuple(
        tuple(query.example_id for query in result.queries) for result in gbm_report.tiers
    )
    if bilinear_query_orders != gbm_query_orders:
        raise ValueError("paired predictors must share replay query order")
    bilinear_folds = tuple(
        (fold.held_out_domain, fold.training_example_ids, fold.test_example_ids)
        for fold in bilinear.learned.folds
    )
    gbm_folds = tuple(
        (fold.held_out_domain, fold.training_example_ids, fold.test_example_ids)
        for fold in gbm.learned.folds
    )
    if bilinear_folds != gbm_folds:
        raise ValueError("paired predictors must share ordered outer folds")


def _held_out_delta(
    held_out_domain: str,
    test_example_ids: tuple[str, ...],
    bilinear: PerQueryNestedLodoBenchmark,
    gbm: PerQueryNestedLodoBenchmark,
) -> HeldOutDomainPredictorDelta:
    bilinear_quality = _query_quality_by_tier(bilinear.learned.report, test_example_ids)
    gbm_quality = _query_quality_by_tier(gbm.learned.report, test_example_ids)
    cheapest_report = bilinear.baseline_by_name["always-cheapest"].report
    oracle_report = bilinear.baseline_by_name["oracle"].report
    cheapest_quality = _query_quality_by_tier(cheapest_report, test_example_ids)
    oracle_quality = _query_quality_by_tier(oracle_report, test_example_ids)
    tier_deltas = {
        tier: _optional_delta(gbm_quality[tier], bilinear_quality[tier])
        for tier in bilinear_quality
    }
    bilinear_weighted = _weighted_quality(bilinear_quality, bilinear.learned.report)
    gbm_weighted = _weighted_quality(gbm_quality, gbm.learned.report)
    bilinear_gap = _oracle_gap_recovery(
        bilinear_quality,
        cheapest_quality,
        oracle_quality,
        bilinear.learned.report,
    )
    gbm_gap = _oracle_gap_recovery(
        gbm_quality,
        cheapest_quality,
        oracle_quality,
        gbm.learned.report,
    )
    return HeldOutDomainPredictorDelta(
        held_out_domain=held_out_domain,
        tier_quality_delta=tier_deltas,
        weighted_quality_delta=_optional_delta(gbm_weighted, bilinear_weighted),
        oracle_gap_recovery_delta=_optional_delta(gbm_gap, bilinear_gap),
    )


@dataclass(frozen=True, slots=True)
class PairedPredictorComparison:
    """Immutable paired estimate with no model-family selection semantics."""

    bilinear: PerQueryNestedLodoBenchmark
    gbm: PerQueryNestedLodoBenchmark
    tier_quality_delta: Mapping[BudgetTier, float | None] = field(init=False)
    weighted_quality_delta: float | None = field(init=False)
    oracle_gap_recovery_delta: float | None = field(init=False)
    held_out_domain_deltas: tuple[HeldOutDomainPredictorDelta, ...] = field(init=False)
    comparison_direction: str = field(default=COMPARISON_DIRECTION, init=False)
    selection_protocol: str = field(default=PAIRED_SELECTION_PROTOCOL, init=False)
    selected_family: None = field(default=None, init=False)
    performance_claim_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.bilinear) is not PerQueryNestedLodoBenchmark:
            raise TypeError("bilinear must be an exact PerQueryNestedLodoBenchmark")
        if type(self.gbm) is not PerQueryNestedLodoBenchmark:
            raise TypeError("gbm must be an exact PerQueryNestedLodoBenchmark")
        _same_pair_scope(self.bilinear, self.gbm)
        bilinear_quality = self.bilinear.learned.score.tier_quality
        gbm_quality = self.gbm.learned.score.tier_quality
        if tuple(bilinear_quality) != tuple(gbm_quality):
            raise ValueError("paired predictors must share score tier order")
        tier_deltas = _immutable_tier_deltas(
            {
                tier: _optional_delta(gbm_quality[tier], bilinear_quality[tier])
                for tier in bilinear_quality
            }
        )
        held_out = tuple(
            _held_out_delta(
                fold.held_out_domain,
                fold.test_example_ids,
                self.bilinear,
                self.gbm,
            )
            for fold in self.bilinear.learned.folds
        )
        object.__setattr__(self, "tier_quality_delta", tier_deltas)
        object.__setattr__(
            self,
            "weighted_quality_delta",
            _optional_delta(
                self.gbm.learned.score.weighted_quality,
                self.bilinear.learned.score.weighted_quality,
            ),
        )
        object.__setattr__(
            self,
            "oracle_gap_recovery_delta",
            _optional_delta(
                self.gbm.learned_gap_recovery,
                self.bilinear.learned_gap_recovery,
            ),
        )
        object.__setattr__(self, "held_out_domain_deltas", held_out)


def evaluate_per_query_paired_predictor_comparison(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
    *,
    bilinear_config: BilinearTrainingConfig | None = None,
    gbm_config: GbmTrainingConfig | None = None,
    max_candidates_per_tier: int | None = None,
    allow_large_exhaustive: bool = False,
) -> PairedPredictorComparison:
    """Estimate both fixed families on one scope without selecting a winner.

    The aggregate GBM call graph is preflighted before either family starts fitting.
    Baselines are evaluated once and the exact object is shared by both results.
    """

    ordered = tuple(examples)
    specs = tuple(tier_specs)
    if bilinear_config is None:
        bilinear_config = BilinearTrainingConfig()
    elif type(bilinear_config) is not BilinearTrainingConfig:
        raise TypeError("bilinear_config must be an exact BilinearTrainingConfig or None")
    if gbm_config is None:
        gbm_config = GbmTrainingConfig()
    elif type(gbm_config) is not GbmTrainingConfig:
        raise TypeError("gbm_config must be an exact GbmTrainingConfig or None")

    preflight_nested_lodo_gbm(ordered, config=gbm_config)
    bilinear = evaluate_per_query_bilinear_benchmark(
        ordered,
        specs,
        config=bilinear_config,
        max_candidates_per_tier=max_candidates_per_tier,
        allow_large_exhaustive=allow_large_exhaustive,
    )
    gbm = evaluate_per_query_gbm_benchmark(
        ordered,
        specs,
        config=gbm_config,
        max_candidates_per_tier=max_candidates_per_tier,
        allow_large_exhaustive=allow_large_exhaustive,
        _baselines=bilinear.baselines,
    )
    return PairedPredictorComparison(bilinear=bilinear, gbm=gbm)
