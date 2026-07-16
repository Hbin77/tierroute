# SPDX-License-Identifier: Apache-2.0
"""Bounded prepared raw-score to nested-LODO policy reference pipeline.

The prepared predictor modules deliberately stop at raw scores.  This module is the
small, evaluation-only bridge that applies the existing per-model isotonic layer,
exact lambda tuner, and offline simulator without copying their policy or budget
semantics.  It proves the complete dependency graph on bounded fixtures; it is not a
persistent/native prepared session or an all-domain deployable predictor artifact.

Only :func:`evaluate_prepared_reference_pipeline` is a supported derivation path.
Direct construction of the evidence records below describes self-declared canonical
content, not provenance or authenticity.  Expected digests must come from a trusted
channel independent of the objects being checked.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType

from tierroute.core.integer_text import integer_to_decimal
from tierroute.eval import (
    BudgetLedgerFactory,
    EvaluationExample,
    TierSpec,
    evaluation_data_sha256,
    evaluation_replay_sha256,
    leave_one_domain_out,
)
from tierroute.eval.provenance import _snapshot_evaluation_scope
from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.features.surface import SURFACE_DOMAIN_TAG_CATALOGUE
from tierroute.policies.lambda_tuning import (
    LambdaSearchPreflightEstimate,
    NestedLodoLambdaResult,
    estimate_lambda_search,
    nested_lodo_lambda_evaluation,
)
from tierroute.predictors.base import QualityPredictor
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.prepared_execution import (
    PREPARED_MOMENT_RIDGE_SOLVER_ID,
    PREPARED_RAW_SCORER_ID,
    PreparedCoefficientBlock,
    PreparedCoefficientBundle,
    PreparedRawScoreBlock,
    PreparedRawScoreBundle,
    PreparedReferenceExecutionEstimate,
    PreparedScoredFeatureShard,
    PreparedScoredFeatureShardBundle,
)
from tierroute.predictors.prepared_graph import (
    PreparedNestedLodoPlan,
    build_prepared_nested_lodo_plan,
)
from tierroute.predictors.prepared_store import (
    PreparedFeatureStore,
    prepared_fit_source_sha256,
)
from tierroute.predictors.resource_limits import MAX_PREDICTOR_CALIBRATOR_POINTS

PREPARED_REFERENCE_PIPELINE_ALGORITHM_ID = "tierroute.prepared-reference-pipeline-v1"
PREPARED_TARGET_SHARD_ALGORITHM_ID = "tierroute.prepared-target-shard-v1"
PREPARED_CALIBRATION_ALGORITHM_ID = "tierroute.prepared-isotonic-calibration-v1"
PREPARED_CALIBRATED_SCORE_ALGORITHM_ID = "tierroute.prepared-calibrated-score-block-v1"

MAX_PREPARED_PIPELINE_WORK_UNITS = 100_000_000
MAX_PREPARED_PIPELINE_NUMERIC_BYTES = 512 * 1024 * 1024
MAX_PREPARED_PIPELINE_REPORT_ROWS = 1_000_000
MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER = 257
MAX_PREPARED_PIPELINE_CANDIDATE_EVIDENCE_BYTES = 8 * 1024 * 1024
MAX_PREPARED_PIPELINE_PAIR_SCANS = 10_000_000
MAX_PREPARED_PIPELINE_UTILITY_EVALUATIONS = 100_000_000

_F64_BYTES = 8
_CALIBRATOR_POINT_BYTES = 2 * _F64_BYTES
_LAMBDA_PAIR_TRAVERSALS_PER_OUTER_FOLD = 5


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _sha256_hex(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


class _EvidenceWriter:
    """Tiny length-delimited evidence writer; hashes are identities, not signatures."""

    __slots__ = ("_digest",)

    def __init__(self, algorithm_id: str) -> None:
        self._digest = hashlib.sha256()
        self.token("algorithm", algorithm_id.encode("utf-8"))

    def token(self, label: str, payload: bytes) -> None:
        label_bytes = label.encode("ascii")
        self._digest.update(len(label_bytes).to_bytes(2, "big"))
        self._digest.update(label_bytes)
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)

    def text(self, label: str, value: str) -> None:
        self.token(label, value.encode("utf-8"))

    def integer(self, label: str, value: int) -> None:
        self.token(label, str(value).encode("ascii"))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _calibrator_payload(calibrator: IsotonicCalibrator) -> bytes:
    values = (*calibrator.upper_bounds, *calibrator.values)
    return struct.pack(f"<{len(values)}d", *values)


def _fraction_text_character_count(value: Fraction) -> int:
    """Count canonical Fraction text without CPython's global int digit limit."""

    numerator = value.numerator
    denominator = value.denominator
    count = len(integer_to_decimal(numerator))
    if denominator != 1:
        count += 1 + len(integer_to_decimal(denominator))
    return count


@dataclass(frozen=True, slots=True)
class PreparedReferencePipelineEstimate:
    """Closed-form admission evidence for prepared calibration and policy replay."""

    plan: PreparedNestedLodoPlan
    execution_estimate: PreparedReferenceExecutionEstimate | None
    lambda_search_estimates: tuple[LambdaSearchPreflightEstimate, ...] | None
    tier_count: int
    max_candidates_per_tier: int
    calibrated_subset_count: int
    calibration_row_memberships: int
    calibration_scalar_points: int
    calibrated_score_block_count: int
    calibrated_prediction_rows: int
    calibrated_prediction_cells: int
    raw_score_row_reads: int
    raw_score_cell_reads: int
    target_row_reads: int
    target_cell_reads: int
    pav_sort_work_units: int
    pav_linear_work_units: int
    calibration_apply_work_units: int
    lambda_pair_scan_upper_bound: int
    lambda_candidate_upper_bound: int
    lambda_utility_evaluation_upper_bound: int
    retained_report_rows: int
    candidate_evidence_upper_bound_bytes: int | None
    postprocess_work_units: int
    total_work_units: int
    postprocess_numeric_bytes: int
    modeled_numeric_storage_bytes: int

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("pipeline estimate plan must be exact")
        if self.execution_estimate is not None:
            if type(self.execution_estimate) is not PreparedReferenceExecutionEstimate:
                raise TypeError("execution_estimate must be exact or None")
            if self.execution_estimate.plan != self.plan:
                raise ValueError("execution_estimate plan does not match the pipeline plan")
        tier_count = _exact_nonnegative_int(self.tier_count, "tier_count")
        if tier_count == 0:
            raise ValueError("tier_count must be positive")
        cap = _exact_nonnegative_int(
            self.max_candidates_per_tier,
            "max_candidates_per_tier",
        )
        if not 2 <= cap <= MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER:
            raise ValueError(
                "max_candidates_per_tier must be between 2 and "
                f"{MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER}"
            )
        _validate_lambda_search_estimates(
            self.lambda_search_estimates,
            self.plan,
            tier_count=tier_count,
            max_candidates_per_tier=cap,
        )
        expected = _pipeline_estimate_values(
            self.plan,
            tier_count,
            cap,
            execution_estimate=self.execution_estimate,
            lambda_search_estimates=self.lambda_search_estimates,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match the prepared pipeline formula")
        if self.total_work_units > MAX_PREPARED_PIPELINE_WORK_UNITS:
            raise ValueError("prepared pipeline modeled work exceeds the reference limit")
        if self.modeled_numeric_storage_bytes > MAX_PREPARED_PIPELINE_NUMERIC_BYTES:
            raise ValueError(
                "prepared pipeline modeled numeric storage exceeds the reference limit"
            )
        if self.retained_report_rows > MAX_PREPARED_PIPELINE_REPORT_ROWS:
            raise ValueError("prepared pipeline retained report rows exceed the reference limit")
        if self.lambda_pair_scan_upper_bound > MAX_PREPARED_PIPELINE_PAIR_SCANS:
            raise ValueError("prepared pipeline pair scans exceed the reference limit")
        if self.lambda_utility_evaluation_upper_bound > MAX_PREPARED_PIPELINE_UTILITY_EVALUATIONS:
            raise ValueError("prepared pipeline utility evaluations exceed the reference limit")
        if (
            self.candidate_evidence_upper_bound_bytes is not None
            and self.candidate_evidence_upper_bound_bytes
            > MAX_PREPARED_PIPELINE_CANDIDATE_EVIDENCE_BYTES
        ):
            raise ValueError("prepared pipeline candidate evidence exceeds the reference limit")


def _validate_lambda_search_estimates(
    estimates: tuple[LambdaSearchPreflightEstimate, ...] | None,
    plan: PreparedNestedLodoPlan,
    *,
    tier_count: int,
    max_candidates_per_tier: int,
) -> None:
    if estimates is None:
        return
    if type(estimates) is not tuple:
        raise TypeError("lambda_search_estimates must be an exact tuple or None")
    if len(estimates) != len(plan.domains):
        raise ValueError("lambda_search_estimates must cover every outer fold exactly once")
    for domain_index, estimate in enumerate(estimates):
        if type(estimate) is not LambdaSearchPreflightEstimate:
            raise TypeError("lambda_search_estimates must contain exact estimates")
        if (
            estimate.example_count
            != plan.work.example_count - plan.domain_example_counts[domain_index]
            or estimate.tier_count != tier_count
            or estimate.model_count != plan.target_count
            or estimate.domain_count != len(plan.domains) - 1
            or estimate.max_candidates_per_tier != max_candidates_per_tier
        ):
            raise ValueError("lambda search estimate does not match its prepared outer fold")


def _pipeline_estimate_values(
    plan: PreparedNestedLodoPlan,
    tier_count: int,
    max_candidates_per_tier: int,
    *,
    execution_estimate: PreparedReferenceExecutionEstimate | None,
    lambda_search_estimates: tuple[LambdaSearchPreflightEstimate, ...] | None,
) -> dict[str, int | None]:
    domain_count = len(plan.domains)
    example_count = plan.work.example_count
    model_count = plan.target_count
    calibrated_subsets = tuple(
        subset
        for subset in plan.training_subsets
        if len(subset.domain_indices) in (domain_count - 2, domain_count - 1)
    )
    calibrated_subset_count = len(calibrated_subsets)
    maximum_calibration_rows = max(subset.row_count for subset in calibrated_subsets)
    if maximum_calibration_rows > MAX_PREDICTOR_CALIBRATOR_POINTS:
        raise ValueError("a prepared calibrator exceeds the predictor point limit")
    calibration_rows = sum(subset.row_count for subset in calibrated_subsets)
    calibration_points = calibration_rows * model_count
    calibrated_score_blocks = sum(
        domain_count - len(subset.domain_indices) for subset in calibrated_subsets
    )
    calibrated_rows = sum(
        sum(
            plan.domain_example_counts[index]
            for index in range(domain_count)
            if index not in subset.domain_indices
        )
        for subset in calibrated_subsets
    )
    calibrated_cells = calibrated_rows * model_count
    raw_rows = calibration_rows + calibrated_rows
    raw_cells = raw_rows * model_count

    pav_sort_work = sum(
        model_count * 2 * subset.row_count * max(1, (max(2, subset.row_count) - 1).bit_length())
        for subset in calibrated_subsets
    )
    pav_linear_work = 10 * calibration_points
    apply_work = 0
    for subset in calibrated_subsets:
        search_steps = max(1, (max(2, subset.row_count) - 1).bit_length())
        destination_rows = sum(
            plan.domain_example_counts[index]
            for index in range(domain_count)
            if index not in subset.domain_indices
        )
        apply_work += destination_rows * model_count * (search_steps + 3)

    if lambda_search_estimates is None:
        pair_count = model_count * (model_count - 1) // 2
        pair_scans = 0
        candidate_bound = 0
        utility_evaluations = 0
        for held_out_count in plan.domain_example_counts:
            outer_training_rows = example_count - held_out_count
            fold_pair_scans = outer_training_rows * pair_count
            fold_candidates = min(max_candidates_per_tier, 2 * (fold_pair_scans + 1))
            pair_scans += _LAMBDA_PAIR_TRAVERSALS_PER_OUTER_FOLD * fold_pair_scans
            candidate_bound += tier_count * fold_candidates
            utility_evaluations += tier_count * fold_candidates * outer_training_rows * model_count
        candidate_bytes = None
    else:
        pair_scans = _LAMBDA_PAIR_TRAVERSALS_PER_OUTER_FOLD * sum(
            estimate.pair_scan_occurrences for estimate in lambda_search_estimates
        )
        candidate_bound = tier_count * sum(
            estimate.candidate_upper_bound for estimate in lambda_search_estimates
        )
        utility_evaluations = sum(
            estimate.utility_evaluation_upper_bound for estimate in lambda_search_estimates
        )
        candidate_bytes = sum(
            estimate.estimated_policy_artifact_bytes for estimate in lambda_search_estimates
        )
    report_rows = tier_count * domain_count * example_count
    target_copy_cells = example_count * model_count
    outer_and_fold_prediction_cells = (
        example_count + max(example_count - count for count in plan.domain_example_counts)
    ) * model_count
    postprocess_numeric_bytes = _F64_BYTES * (
        2 * calibration_points
        + calibrated_cells
        + target_copy_cells
        + outer_and_fold_prediction_cells
        + 2 * maximum_calibration_rows * model_count
        + 4 * maximum_calibration_rows
    )
    target_rows = calibration_rows + example_count
    target_cells = target_rows * model_count
    postprocess_work = (
        raw_cells
        + target_cells
        + pav_sort_work
        + pav_linear_work
        + apply_work
        + pair_scans
        + utility_evaluations
        + report_rows
    )
    return {
        "calibrated_subset_count": calibrated_subset_count,
        "calibration_row_memberships": calibration_rows,
        "calibration_scalar_points": calibration_points,
        "calibrated_score_block_count": calibrated_score_blocks,
        "calibrated_prediction_rows": calibrated_rows,
        "calibrated_prediction_cells": calibrated_cells,
        "raw_score_row_reads": raw_rows,
        "raw_score_cell_reads": raw_cells,
        "target_row_reads": target_rows,
        "target_cell_reads": target_cells,
        "pav_sort_work_units": pav_sort_work,
        "pav_linear_work_units": pav_linear_work,
        "calibration_apply_work_units": apply_work,
        "lambda_pair_scan_upper_bound": pair_scans,
        "lambda_candidate_upper_bound": candidate_bound,
        "lambda_utility_evaluation_upper_bound": utility_evaluations,
        "retained_report_rows": report_rows,
        "candidate_evidence_upper_bound_bytes": candidate_bytes,
        "postprocess_work_units": postprocess_work,
        "total_work_units": max(
            plan.work.total_numeric_work_units,
            0 if execution_estimate is None else execution_estimate.total_work_units,
        )
        + postprocess_work,
        "postprocess_numeric_bytes": postprocess_numeric_bytes,
        "modeled_numeric_storage_bytes": max(
            plan.work.modeled_buffer_bytes,
            0 if execution_estimate is None else execution_estimate.modeled_numeric_storage_bytes,
        )
        + postprocess_numeric_bytes,
    }


def estimate_prepared_reference_pipeline(
    plan: PreparedNestedLodoPlan,
    *,
    tier_count: int,
    max_candidates_per_tier: int,
    execution_estimate: PreparedReferenceExecutionEstimate | None = None,
    lambda_search_estimates: tuple[LambdaSearchPreflightEstimate, ...] | None = None,
) -> PreparedReferencePipelineEstimate:
    """Preflight the complete prepared post-processing shape without reading rows."""

    if type(plan) is not PreparedNestedLodoPlan:
        raise TypeError("plan must be an exact PreparedNestedLodoPlan")
    validated_tier_count = _exact_nonnegative_int(tier_count, "tier_count")
    if validated_tier_count == 0:
        raise ValueError("tier_count must be positive")
    validated_cap = _exact_nonnegative_int(
        max_candidates_per_tier,
        "max_candidates_per_tier",
    )
    if not 2 <= validated_cap <= MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER:
        raise ValueError(
            "max_candidates_per_tier must be between 2 and "
            f"{MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER}"
        )
    if execution_estimate is not None:
        if type(execution_estimate) is not PreparedReferenceExecutionEstimate:
            raise TypeError("execution_estimate must be exact or None")
        if execution_estimate.plan != plan:
            raise ValueError("execution_estimate plan does not match the pipeline plan")
    _validate_lambda_search_estimates(
        lambda_search_estimates,
        plan,
        tier_count=validated_tier_count,
        max_candidates_per_tier=validated_cap,
    )
    values = _pipeline_estimate_values(
        plan,
        validated_tier_count,
        validated_cap,
        execution_estimate=execution_estimate,
        lambda_search_estimates=lambda_search_estimates,
    )
    return PreparedReferencePipelineEstimate(
        plan=plan,
        execution_estimate=execution_estimate,
        lambda_search_estimates=lambda_search_estimates,
        tier_count=validated_tier_count,
        max_candidates_per_tier=validated_cap,
        **values,
    )


@dataclass(frozen=True, slots=True)
class PreparedTargetShardEvidence:
    """Content identity for one canonical domain target shard."""

    domain_index: int
    domain: str
    example_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    sha256: str
    algorithm_id: str = field(default=PREPARED_TARGET_SHARD_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.domain_index, "target shard domain_index")
        if type(self.domain) is not str or not self.domain.strip():
            raise ValueError("target shard domain must be non-empty text")
        if type(self.example_ids) is not tuple or not self.example_ids:
            raise ValueError("target shard example_ids must be a non-empty exact tuple")
        if any(
            type(example_id) is not str or not example_id.strip() for example_id in self.example_ids
        ):
            raise ValueError("target shard example IDs must be non-empty exact strings")
        if self.example_ids != tuple(sorted(set(self.example_ids))):
            raise ValueError("target shard example_ids must be sorted and unique")
        if type(self.model_ids) is not tuple or not self.model_ids:
            raise ValueError("target shard model_ids must be a non-empty exact tuple")
        if any(type(model_id) is not str or not model_id.strip() for model_id in self.model_ids):
            raise ValueError("target shard model IDs must be non-empty exact strings")
        if self.model_ids != tuple(sorted(set(self.model_ids))):
            raise ValueError("target shard model_ids must be sorted and unique")
        _sha256_hex(self.sha256, "target shard sha256")


@dataclass(frozen=True, slots=True)
class PreparedCalibrationEvidence:
    """One per-model inner-LODO isotonic fit bound to raw and target shards."""

    training_subset_index: int
    training_domain_indices: tuple[int, ...]
    model_ids: tuple[str, ...]
    calibration_example_count: int
    raw_score_block_indices: tuple[int, ...]
    raw_score_block_sha256s: tuple[str, ...]
    target_shard_sha256s: tuple[str, ...]
    calibrators: tuple[IsotonicCalibrator, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_CALIBRATION_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.training_subset_index, "calibration subset index")
        if (
            type(self.training_domain_indices) is not tuple
            or not self.training_domain_indices
            or any(type(index) is not int or index < 0 for index in self.training_domain_indices)
            or self.training_domain_indices != tuple(sorted(set(self.training_domain_indices)))
        ):
            raise ValueError("calibration domains must be a sorted unique exact tuple")
        if type(self.model_ids) is not tuple or self.model_ids != tuple(
            sorted(set(self.model_ids))
        ):
            raise ValueError("calibration model IDs must be a sorted unique exact tuple")
        if any(type(model_id) is not str or not model_id.strip() for model_id in self.model_ids):
            raise ValueError("calibration model IDs must be non-empty exact strings")
        count = _exact_nonnegative_int(
            self.calibration_example_count,
            "calibration example count",
        )
        if count == 0 or count > MAX_PREDICTOR_CALIBRATOR_POINTS:
            raise ValueError("calibration example count is outside the predictor limit")
        child_count = len(self.training_domain_indices)
        if (
            type(self.raw_score_block_indices) is not tuple
            or len(self.raw_score_block_indices) != child_count
            or type(self.raw_score_block_sha256s) is not tuple
            or len(self.raw_score_block_sha256s) != child_count
            or type(self.target_shard_sha256s) is not tuple
            or len(self.target_shard_sha256s) != child_count
        ):
            raise ValueError("calibration source evidence has the wrong exact length")
        if any(type(index) is not int or index < 0 for index in self.raw_score_block_indices):
            raise ValueError("calibration block indices must be non-negative exact integers")
        for digest in (*self.raw_score_block_sha256s, *self.target_shard_sha256s):
            _sha256_hex(digest, "calibration source sha256")
        if type(self.calibrators) is not tuple or len(self.calibrators) != len(self.model_ids):
            raise ValueError("calibrators must match the exact model catalogue")
        if not all(type(item) is IsotonicCalibrator for item in self.calibrators):
            raise TypeError("calibrators must contain exact IsotonicCalibrator values")
        if any(len(calibrator.values) > count for calibrator in self.calibrators):
            raise ValueError("calibrator block count exceeds its calibration example count")
        writer = _EvidenceWriter(self.algorithm_id)
        writer.integer("training_subset_index", self.training_subset_index)
        for domain_index in self.training_domain_indices:
            writer.integer("training_domain_index", domain_index)
        writer.integer("calibration_example_count", count)
        for model_id in self.model_ids:
            writer.text("model_id", model_id)
        for index, raw_sha, target_sha in zip(
            self.raw_score_block_indices,
            self.raw_score_block_sha256s,
            self.target_shard_sha256s,
            strict=True,
        ):
            writer.integer("raw_score_block_index", index)
            writer.text("raw_score_block_sha256", raw_sha)
            writer.text("target_shard_sha256", target_sha)
        for calibrator in self.calibrators:
            writer.integer("calibrator_block_count", len(calibrator.values))
            writer.token("calibrator.f64le", _calibrator_payload(calibrator))
        object.__setattr__(self, "sha256", writer.hexdigest())


@dataclass(frozen=True, slots=True)
class PreparedCalibratedScoreEvidence:
    """Identity of one target-free calibrated destination score block."""

    training_subset_index: int
    scored_domain_index: int
    raw_score_block_index: int
    raw_score_block_sha256: str
    calibration_sha256: str
    scored_feature_shard_sha256: str
    example_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    scores_sha256: str
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_CALIBRATED_SCORE_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        for name in (
            "training_subset_index",
            "scored_domain_index",
            "raw_score_block_index",
        ):
            _exact_nonnegative_int(getattr(self, name), name)
        for digest in (
            self.raw_score_block_sha256,
            self.calibration_sha256,
            self.scored_feature_shard_sha256,
            self.scores_sha256,
        ):
            _sha256_hex(digest, "calibrated score sha256")
        if type(self.example_ids) is not tuple or self.example_ids != tuple(
            sorted(set(self.example_ids))
        ):
            raise ValueError("calibrated score example IDs must be sorted and unique")
        if any(
            type(example_id) is not str or not example_id.strip() for example_id in self.example_ids
        ):
            raise ValueError("calibrated score example IDs must be non-empty exact strings")
        if type(self.model_ids) is not tuple or self.model_ids != tuple(
            sorted(set(self.model_ids))
        ):
            raise ValueError("calibrated score model IDs must be sorted and unique")
        if any(type(model_id) is not str or not model_id.strip() for model_id in self.model_ids):
            raise ValueError("calibrated score model IDs must be non-empty exact strings")
        writer = _EvidenceWriter(self.algorithm_id)
        writer.integer("training_subset_index", self.training_subset_index)
        writer.integer("scored_domain_index", self.scored_domain_index)
        writer.integer("raw_score_block_index", self.raw_score_block_index)
        writer.text("raw_score_block_sha256", self.raw_score_block_sha256)
        writer.text("calibration_sha256", self.calibration_sha256)
        writer.text("scored_feature_shard_sha256", self.scored_feature_shard_sha256)
        for example_id in self.example_ids:
            writer.text("example_id", example_id)
        for model_id in self.model_ids:
            writer.text("model_id", model_id)
        writer.text("scores_sha256", self.scores_sha256)
        object.__setattr__(self, "sha256", writer.hexdigest())


@dataclass(frozen=True, slots=True)
class PreparedReferencePipelineResult:
    """Existing nested result plus the prepared lineage used to derive predictions."""

    estimate: PreparedReferencePipelineEstimate
    source_fit_sha256: str
    store_sha256: str
    raw_score_bundle_sha256: str
    evaluation_data_sha256: str
    evaluation_replay_sha256: str
    ridge: float
    solver_id: str
    scorer_id: str
    embedding_dimension: int
    embedding_identity: EmbeddingIdentity | None
    coefficient_block_sha256s: tuple[str, ...]
    scored_feature_shard_sha256s: tuple[str, ...]
    raw_score_block_sha256s: tuple[str, ...]
    target_shards: tuple[PreparedTargetShardEvidence, ...]
    calibrations: tuple[PreparedCalibrationEvidence, ...]
    calibrated_score_blocks: tuple[PreparedCalibratedScoreEvidence, ...]
    learned: NestedLodoLambdaResult
    all_searches_exhaustive: bool = field(init=False)
    algorithm_id: str = field(default=PREPARED_REFERENCE_PIPELINE_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.estimate) is not PreparedReferencePipelineEstimate:
            raise TypeError("estimate must be exact PreparedReferencePipelineEstimate")
        for digest in (
            self.source_fit_sha256,
            self.store_sha256,
            self.raw_score_bundle_sha256,
            self.evaluation_data_sha256,
            self.evaluation_replay_sha256,
        ):
            _sha256_hex(digest, "prepared pipeline sha256")
        if type(self.ridge) is not float:
            raise TypeError("ridge must be an exact float")
        if not math.isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        if self.solver_id != PREPARED_MOMENT_RIDGE_SOLVER_ID:
            raise ValueError("solver_id does not identify the prepared reference solver")
        if self.scorer_id != PREPARED_RAW_SCORER_ID:
            raise ValueError("scorer_id does not identify the prepared reference scorer")
        embedding_dimension = _exact_nonnegative_int(
            self.embedding_dimension,
            "embedding_dimension",
        )
        if self.estimate.plan.feature_count != (
            5 + len(SURFACE_DOMAIN_TAG_CATALOGUE) + embedding_dimension
        ):
            raise ValueError("embedding_dimension does not match the prepared plan")
        if (embedding_dimension == 0) != (self.embedding_identity is None):
            raise ValueError("embedding identity and dimension disagree")
        if (
            self.embedding_identity is not None
            and type(self.embedding_identity) is not EmbeddingIdentity
        ):
            raise TypeError("embedding_identity must be exact or None")
        domain_count = len(self.estimate.plan.domains)
        plan = self.estimate.plan
        for name, values, expected_count in (
            (
                "coefficient_block_sha256s",
                self.coefficient_block_sha256s,
                len(plan.training_subsets),
            ),
            (
                "scored_feature_shard_sha256s",
                self.scored_feature_shard_sha256s,
                domain_count,
            ),
            ("raw_score_block_sha256s", self.raw_score_block_sha256s, len(plan.score_blocks)),
        ):
            if type(values) is not tuple or len(values) != expected_count:
                raise ValueError(f"{name} has the wrong canonical length")
            for digest in values:
                _sha256_hex(digest, name)
        if (
            type(self.target_shards) is not tuple
            or len(self.target_shards) != domain_count
            or not all(type(item) is PreparedTargetShardEvidence for item in self.target_shards)
            or tuple(item.domain_index for item in self.target_shards) != tuple(range(domain_count))
        ):
            raise ValueError("target shards must cover canonical domains exactly once")
        model_ids = self.target_shards[0].model_ids
        if len(model_ids) != plan.target_count or any(
            shard.domain != plan.domains[shard.domain_index]
            or len(shard.example_ids) != plan.domain_example_counts[shard.domain_index]
            or shard.model_ids != model_ids
            for shard in self.target_shards
        ):
            raise ValueError("target shards do not match the prepared plan catalogue")
        all_example_ids = tuple(
            example_id for shard in self.target_shards for example_id in shard.example_ids
        )
        if len(all_example_ids) != plan.work.example_count or len(set(all_example_ids)) != len(
            all_example_ids
        ):
            raise ValueError("target shards must cover globally unique prepared example IDs")
        expected_subsets = tuple(
            index
            for index, subset in enumerate(plan.training_subsets)
            if len(subset.domain_indices) in (domain_count - 2, domain_count - 1)
        )
        if (
            type(self.calibrations) is not tuple
            or not all(type(item) is PreparedCalibrationEvidence for item in self.calibrations)
            or tuple(item.training_subset_index for item in self.calibrations) != expected_subsets
        ):
            raise ValueError("calibrations must cover every canonical prepared subset exactly once")
        subset_lookup = {
            subset.domain_indices: index for index, subset in enumerate(plan.training_subsets)
        }
        block_lookup = {
            (block.training_subset_index, block.scored_domain_index): index
            for index, block in enumerate(plan.score_blocks)
        }
        target_sha_by_domain = {shard.domain_index: shard.sha256 for shard in self.target_shards}
        for calibration in self.calibrations:
            subset = plan.training_subsets[calibration.training_subset_index]
            expected_raw_indices = tuple(
                block_lookup[
                    (
                        subset_lookup[
                            tuple(
                                domain_index
                                for domain_index in subset.domain_indices
                                if domain_index != calibration_domain
                            )
                        ],
                        calibration_domain,
                    )
                ]
                for calibration_domain in subset.domain_indices
            )
            if (
                calibration.training_domain_indices != subset.domain_indices
                or calibration.model_ids != model_ids
                or calibration.calibration_example_count != subset.row_count
                or calibration.raw_score_block_indices != expected_raw_indices
                or calibration.raw_score_block_sha256s
                != tuple(self.raw_score_block_sha256s[index] for index in expected_raw_indices)
                or calibration.target_shard_sha256s
                != tuple(target_sha_by_domain[index] for index in subset.domain_indices)
            ):
                raise ValueError("calibration evidence does not match its prepared subset")
        expected_blocks = tuple(
            (subset_index, domain_index)
            for subset_index in expected_subsets
            for domain_index in range(domain_count)
            if domain_index not in plan.training_subsets[subset_index].domain_indices
        )
        if (
            type(self.calibrated_score_blocks) is not tuple
            or not all(
                type(item) is PreparedCalibratedScoreEvidence
                for item in self.calibrated_score_blocks
            )
            or tuple(
                (item.training_subset_index, item.scored_domain_index)
                for item in self.calibrated_score_blocks
            )
            != expected_blocks
        ):
            raise ValueError("calibrated score blocks must cover the canonical graph exactly once")
        calibration_sha_by_subset = {
            calibration.training_subset_index: calibration.sha256
            for calibration in self.calibrations
        }
        target_ids_by_domain = {
            shard.domain_index: shard.example_ids for shard in self.target_shards
        }
        for block in self.calibrated_score_blocks:
            expected_raw_index = block_lookup[
                (block.training_subset_index, block.scored_domain_index)
            ]
            if (
                block.raw_score_block_index != expected_raw_index
                or block.raw_score_block_sha256 != self.raw_score_block_sha256s[expected_raw_index]
                or block.scored_feature_shard_sha256
                != self.scored_feature_shard_sha256s[block.scored_domain_index]
                or block.calibration_sha256
                != calibration_sha_by_subset[block.training_subset_index]
                or block.example_ids != target_ids_by_domain[block.scored_domain_index]
                or block.model_ids != model_ids
            ):
                raise ValueError("calibrated score evidence does not match its graph context")
        if type(self.learned) is not NestedLodoLambdaResult:
            raise TypeError("learned must be an exact NestedLodoLambdaResult")
        if (
            self.estimate.execution_estimate is None
            or self.estimate.lambda_search_estimates is None
        ):
            raise ValueError(
                "a complete prepared result requires execution and lambda-search estimates"
            )
        lambda_search_estimates = self.estimate.lambda_search_estimates
        tier_order = tuple(result.tier_spec.tier for result in self.learned.report.tiers)
        if len(tier_order) != self.estimate.tier_count:
            raise ValueError("learned tier count does not match the prepared estimate")
        if tuple(fold.held_out_domain for fold in self.learned.folds) != plan.domains:
            raise ValueError("learned outer folds must follow the prepared domain catalogue")
        all_id_set = set(all_example_ids)
        ids_by_domain = {shard.domain: set(shard.example_ids) for shard in self.target_shards}
        for fold, search_estimate in zip(
            self.learned.folds,
            lambda_search_estimates,
            strict=True,
        ):
            if (
                set(fold.test_example_ids) != ids_by_domain[fold.held_out_domain]
                or set(fold.training_example_ids)
                != all_id_set - ids_by_domain[fold.held_out_domain]
                or tuple(selection.tier for selection in fold.tuning.selections) != tier_order
                or any(
                    len(selection.candidates.values) > search_estimate.candidate_upper_bound
                    or selection.candidates.observed_breakpoint_count
                    > search_estimate.unequal_cost_pair_occurrences
                    or (
                        selection.candidates.total_derived_values is not None
                        and selection.candidates.total_derived_values
                        > search_estimate.derived_candidate_upper_bound
                    )
                    or selection.candidates.strategy
                    not in {"bounded-bottom-hash-v2", "exhaustive-breakpoints-v1"}
                    or any(
                        _fraction_text_character_count(candidate)
                        > search_estimate.maximum_candidate_fraction_characters
                        for candidate in selection.candidates.values
                    )
                    for selection in fold.tuning.selections
                )
            ):
                raise ValueError(
                    "learned fold membership, tiers, or candidate bounds do not match "
                    "prepared evidence"
                )
        exhaustive = all(
            selection.candidates.exhaustive
            for fold in self.learned.folds
            for selection in fold.tuning.selections
        )
        object.__setattr__(self, "all_searches_exhaustive", exhaustive)


def _fresh_embedding_identity(identity: EmbeddingIdentity | None) -> EmbeddingIdentity | None:
    return None if identity is None else replace(identity)


def _preflight_prepared_input_shape(
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
) -> PreparedNestedLodoPlan:
    """Bound every aggregate child tuple before any recursive reconstruction."""

    if type(store.plan) is not PreparedNestedLodoPlan:
        raise TypeError("prepared store plan must be exact")
    plan = build_prepared_nested_lodo_plan(
        store.plan.domains,
        store.plan.domain_example_counts,
        feature_count=store.plan.feature_count,
        target_count=store.plan.target_count,
    )
    if plan != store.plan:
        raise ValueError("prepared plan is not its canonical reconstruction")

    coefficients = raw_scores.coefficients
    if type(coefficients) is not PreparedCoefficientBundle:
        raise TypeError("raw-score coefficients must be an exact coefficient bundle")
    shards = raw_scores.feature_shards
    if type(shards) is not PreparedScoredFeatureShardBundle:
        raise TypeError("raw-score feature shards must be an exact shard bundle")
    if coefficients.plan != plan or shards.plan != plan:
        raise ValueError("prepared raw-score parents do not match the canonical plan")
    if type(coefficients.execution_estimate) is not PreparedReferenceExecutionEstimate:
        raise TypeError("prepared execution estimate must be exact")
    if coefficients.execution_estimate.plan != plan:
        raise ValueError("prepared execution estimate does not match the canonical plan")

    aggregate_children = (
        (
            "coefficient blocks",
            coefficients.blocks,
            len(plan.training_subsets),
            PreparedCoefficientBlock,
        ),
        (
            "feature shards",
            shards.shards,
            len(plan.domains),
            PreparedScoredFeatureShard,
        ),
        (
            "raw-score blocks",
            raw_scores.blocks,
            len(plan.score_blocks),
            PreparedRawScoreBlock,
        ),
    )
    for name, children, expected_count, expected_type in aggregate_children:
        if type(children) is not tuple:
            raise TypeError(f"{name} must be an exact tuple")
        if len(children) != expected_count:
            raise ValueError(f"{name} have the wrong canonical bounded length")
        if not all(type(child) is expected_type for child in children):
            raise TypeError(f"{name} must contain exact child values")

    feature_bytes = plan.work.example_count * plan.feature_count * _F64_BYTES
    target_bytes = plan.work.example_count * plan.target_count * _F64_BYTES
    if (
        type(store.feature_payload) is not bytes
        or type(store.target_payload) is not bytes
        or len(store.feature_payload) != feature_bytes
        or len(store.target_payload) != target_bytes
    ):
        raise ValueError("prepared store payloads have the wrong canonical bounded length")
    if any(
        type(block.weights_payload) is not bytes or type(block.intercepts_payload) is not bytes
        for block in coefficients.blocks
    ):
        raise TypeError("prepared coefficient payloads must be immutable bytes")
    if (
        sum(
            len(block.weights_payload) + len(block.intercepts_payload)
            for block in coefficients.blocks
        )
        != coefficients.execution_estimate.coefficient_bytes
    ):
        raise ValueError("prepared coefficient payloads exceed their bounded shape")
    if any(type(block.scores_payload) is not bytes for block in raw_scores.blocks):
        raise TypeError("prepared raw-score payloads must be immutable bytes")
    if sum(len(block.scores_payload) for block in raw_scores.blocks) != (
        coefficients.execution_estimate.score_bytes
    ):
        raise ValueError("prepared raw-score payloads exceed their bounded shape")
    return plan


def _validate_trusted_parent_identities(
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
    *,
    expected_source_fit_sha256: str,
    expected_store_sha256: str,
    expected_raw_score_sha256: str,
) -> None:
    """Reject cheap parent-identity mismatches before numeric leaf validation."""

    stored_source_sha = _sha256_hex(store.source_fit_sha256, "store source_fit_sha256")
    stored_store_sha = _sha256_hex(store.sha256, "store sha256")
    stored_raw_sha = _sha256_hex(raw_scores.sha256, "raw-score bundle sha256")
    coefficient_source_sha = _sha256_hex(
        raw_scores.coefficients.source_store_sha256,
        "coefficient source_store_sha256",
    )
    if not hmac.compare_digest(stored_source_sha, expected_source_fit_sha256):
        raise ValueError("prepared store does not match the trusted source-fit SHA-256")
    if not hmac.compare_digest(stored_store_sha, expected_store_sha256):
        raise ValueError("prepared store does not match the trusted store SHA-256")
    if not hmac.compare_digest(stored_raw_sha, expected_raw_score_sha256):
        raise ValueError("prepared raw scores do not match the trusted bundle SHA-256")
    if not hmac.compare_digest(coefficient_source_sha, stored_store_sha):
        raise ValueError("prepared raw scores do not descend from the exact feature store")


def _resnapshot_prepared_inputs(
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
) -> tuple[PreparedFeatureStore, PreparedRawScoreBundle]:
    """Re-run every leaf constructor so stale init=False digests cannot mask mutation."""

    plan = _preflight_prepared_input_shape(store, raw_scores)
    identity = _fresh_embedding_identity(store.embedding_identity)
    fresh_store = replace(store, plan=plan, embedding_identity=identity)

    coefficients = raw_scores.coefficients
    fresh_estimate = replace(coefficients.execution_estimate, plan=plan)
    fresh_coefficient_blocks: list[PreparedCoefficientBlock] = []
    for block in coefficients.blocks:
        schema_identity = _fresh_embedding_identity(block.feature_schema.embedding_identity)
        schema = replace(block.feature_schema, embedding_identity=schema_identity)
        fresh_coefficient_blocks.append(replace(block, plan=plan, feature_schema=schema))
    fresh_coefficients = replace(
        coefficients,
        plan=plan,
        embedding_identity=identity,
        execution_estimate=fresh_estimate,
        blocks=tuple(fresh_coefficient_blocks),
    )

    shards = raw_scores.feature_shards
    fresh_shard_rows: tuple[PreparedScoredFeatureShard, ...] = tuple(
        replace(shard, plan=plan, embedding_identity=identity) for shard in shards.shards
    )
    fresh_shards = replace(
        shards,
        plan=plan,
        embedding_identity=identity,
        shards=fresh_shard_rows,
    )
    fresh_raw_blocks: tuple[PreparedRawScoreBlock, ...] = tuple(
        replace(block, plan=plan) for block in raw_scores.blocks
    )
    fresh_raw = replace(
        raw_scores,
        coefficients=fresh_coefficients,
        feature_shards=fresh_shards,
        blocks=fresh_raw_blocks,
    )
    return fresh_store, fresh_raw


def _target_shard_sha256(
    plan: PreparedNestedLodoPlan,
    domain_index: int,
    example_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
    rows: Sequence[tuple[float, ...]],
) -> str:
    writer = _EvidenceWriter(PREPARED_TARGET_SHARD_ALGORITHM_ID)
    writer.integer("domain_index", domain_index)
    writer.text("domain", plan.domains[domain_index])
    for model_id in model_ids:
        writer.text("model_id", model_id)
    for example_id, row in zip(example_ids, rows, strict=True):
        writer.text("example_id", example_id)
        writer.token("targets.f64le", struct.pack(f"<{len(row)}d", *row))
    return writer.hexdigest()


def _calibrated_scores_sha256(
    example_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
    rows: tuple[tuple[float, ...], ...],
) -> str:
    writer = _EvidenceWriter(PREPARED_CALIBRATED_SCORE_ALGORITHM_ID)
    for model_id in model_ids:
        writer.text("model_id", model_id)
    for example_id, row in zip(example_ids, rows, strict=True):
        writer.text("example_id", example_id)
        writer.token("scores.f64le", struct.pack(f"<{len(row)}d", *row))
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class _PreparedBatchPredictor:
    model_ids: tuple[str, ...]
    rows_by_prompt_batch: Mapping[tuple[str, ...], tuple[tuple[float, ...], ...]]

    def predict_batch(
        self,
        prompts: Sequence[str],
        model_ids: Sequence[str],
    ) -> tuple[Mapping[str, float], ...]:
        prompt_key = tuple(prompts)
        requested_models = tuple(model_ids)
        if requested_models != self.model_ids:
            raise ValueError("prepared predictor requires the canonical sorted model catalogue")
        try:
            rows = self.rows_by_prompt_batch[prompt_key]
        except KeyError as error:
            raise ValueError("prepared predictor received an unknown prompt batch") from error
        return tuple(MappingProxyType(dict(zip(self.model_ids, row, strict=True))) for row in rows)

    def predict(self, prompt: str, model_id: str) -> float:
        matches = []
        model_index = self.model_ids.index(model_id)
        for prompts, rows in self.rows_by_prompt_batch.items():
            for candidate_prompt, row in zip(prompts, rows, strict=True):
                if candidate_prompt == prompt:
                    matches.append(row[model_index])
        if not matches or any(value != matches[0] for value in matches[1:]):
            raise ValueError("single-prompt prepared lookup is missing or ambiguous")
        return matches[0]


class _PreparedPipelineBuilder:
    __slots__ = (
        "block_by_context",
        "calibrated_blocks",
        "calibrations",
        "example_by_id",
        "examples",
        "examples_by_domain",
        "predictors",
        "raw_scores",
        "row_index_by_id",
        "store",
        "subset_by_domains",
        "target_shard_by_domain",
        "target_shards",
    )

    def __init__(
        self,
        examples: tuple[EvaluationExample, ...],
        store: PreparedFeatureStore,
        raw_scores: PreparedRawScoreBundle,
    ) -> None:
        self.examples = examples
        self.store = store
        self.raw_scores = raw_scores
        self.subset_by_domains = {
            subset.domain_indices: index for index, subset in enumerate(store.plan.training_subsets)
        }
        self.block_by_context = {
            (block.training_subset_index, block.scored_domain_index): index
            for index, block in enumerate(store.plan.score_blocks)
        }
        self.row_index_by_id = {
            example_id: index for index, example_id in enumerate(store.example_ids)
        }
        self.example_by_id = {example.example_id: example for example in examples}
        self.examples_by_domain = tuple(
            tuple(example for example in examples if example.domain == domain)
            for domain in store.plan.domains
        )
        target_shards: list[PreparedTargetShardEvidence] = []
        target_shard_by_domain: dict[int, PreparedTargetShardEvidence] = {}
        for domain_index, feature_shard in enumerate(raw_scores.feature_shards.shards):
            rows = tuple(
                store.target_row(self.row_index_by_id[example_id])
                for example_id in feature_shard.example_ids
            )
            evidence = PreparedTargetShardEvidence(
                domain_index=domain_index,
                domain=store.plan.domains[domain_index],
                example_ids=feature_shard.example_ids,
                model_ids=store.model_ids,
                sha256=_target_shard_sha256(
                    store.plan,
                    domain_index,
                    feature_shard.example_ids,
                    store.model_ids,
                    rows,
                ),
            )
            target_shards.append(evidence)
            target_shard_by_domain[domain_index] = evidence
        self.target_shards = tuple(target_shards)
        self.target_shard_by_domain = target_shard_by_domain
        self.calibrations: dict[int, PreparedCalibrationEvidence] = {}
        self.predictors: dict[int, _PreparedBatchPredictor] = {}
        self.calibrated_blocks: dict[tuple[int, int], PreparedCalibratedScoreEvidence] = {}

    def _calibration(self, subset_index: int) -> PreparedCalibrationEvidence:
        cached = self.calibrations.get(subset_index)
        if cached is not None:
            return cached
        plan = self.store.plan
        subset = plan.training_subsets[subset_index]
        domain_count = len(plan.domains)
        if len(subset.domain_indices) not in (domain_count - 2, domain_count - 1):
            raise ValueError("prepared calibrator subset has an unsupported graph depth")
        predictions = [[] for _ in self.store.model_ids]
        targets = [[] for _ in self.store.model_ids]
        raw_indices: list[int] = []
        raw_hashes: list[str] = []
        target_hashes: list[str] = []
        for calibration_domain in subset.domain_indices:
            base_domains = tuple(
                index for index in subset.domain_indices if index != calibration_domain
            )
            base_subset_index = self.subset_by_domains[base_domains]
            raw_index = self.block_by_context[(base_subset_index, calibration_domain)]
            raw_block = self.raw_scores.blocks[raw_index]
            example_ids = self.raw_scores.example_ids_for_block(raw_index)
            for row_position, example_id in enumerate(example_ids):
                score_row = raw_block.score_row(row_position)
                target_row = self.store.target_row(self.row_index_by_id[example_id])
                for model_index in range(len(self.store.model_ids)):
                    predictions[model_index].append(score_row[model_index])
                    targets[model_index].append(target_row[model_index])
            raw_indices.append(raw_index)
            raw_hashes.append(raw_block.sha256)
            target_hashes.append(self.target_shard_by_domain[calibration_domain].sha256)
        expected_count = subset.row_count
        if any(len(values) != expected_count for values in (*predictions, *targets)):
            raise AssertionError("prepared calibration did not cover its training subset once")
        calibrators = tuple(
            IsotonicCalibrator.fit(predictions[index], targets[index])
            for index in range(len(self.store.model_ids))
        )
        evidence = PreparedCalibrationEvidence(
            training_subset_index=subset_index,
            training_domain_indices=subset.domain_indices,
            model_ids=self.store.model_ids,
            calibration_example_count=expected_count,
            raw_score_block_indices=tuple(raw_indices),
            raw_score_block_sha256s=tuple(raw_hashes),
            target_shard_sha256s=tuple(target_hashes),
            calibrators=calibrators,
        )
        self.calibrations[subset_index] = evidence
        return evidence

    def predictor(self, training: tuple[EvaluationExample, ...]) -> QualityPredictor:
        domains = tuple(sorted({example.domain for example in training}))
        domain_to_index = {domain: index for index, domain in enumerate(self.store.plan.domains)}
        try:
            domain_indices = tuple(domain_to_index[domain] for domain in domains)
        except KeyError as error:
            raise ValueError("prepared trainer received a domain outside the plan") from error
        expected_training = tuple(
            example
            for example in self.examples
            if domain_to_index[example.domain] in domain_indices
        )
        if training != expected_training:
            raise ValueError("prepared trainer received rows outside the canonical nested fold")
        try:
            subset_index = self.subset_by_domains[domain_indices]
        except KeyError as error:
            raise ValueError("prepared trainer received an unsupported training subset") from error
        cached = self.predictors.get(subset_index)
        if cached is not None:
            return cached
        calibration = self._calibration(subset_index)
        rows_by_prompt_batch: dict[tuple[str, ...], tuple[tuple[float, ...], ...]] = {}
        subset = self.store.plan.training_subsets[subset_index]
        for destination in range(len(self.store.plan.domains)):
            if destination in subset.domain_indices:
                continue
            raw_index = self.block_by_context[(subset_index, destination)]
            raw_block = self.raw_scores.blocks[raw_index]
            canonical_ids = self.raw_scores.example_ids_for_block(raw_index)
            canonical_rows = tuple(
                tuple(
                    calibration.calibrators[model_index].calibrate(raw_score)
                    for model_index, raw_score in enumerate(raw_block.score_row(row_index))
                )
                for row_index in range(len(canonical_ids))
            )
            score_by_id = dict(zip(canonical_ids, canonical_rows, strict=True))
            destination_examples = self.examples_by_domain[destination]
            prompt_key = tuple(example.prompt for example in destination_examples)
            replay_rows = tuple(score_by_id[example.example_id] for example in destination_examples)
            existing = rows_by_prompt_batch.get(prompt_key)
            if existing is not None and existing != replay_rows:
                raise ValueError(
                    "prepared predictor cannot disambiguate identical prompt batches with "
                    "different precomputed scores"
                )
            rows_by_prompt_batch[prompt_key] = replay_rows
            feature_shard = self.raw_scores.feature_shards.shards[destination]
            self.calibrated_blocks[(subset_index, destination)] = PreparedCalibratedScoreEvidence(
                training_subset_index=subset_index,
                scored_domain_index=destination,
                raw_score_block_index=raw_index,
                raw_score_block_sha256=raw_block.sha256,
                calibration_sha256=calibration.sha256,
                scored_feature_shard_sha256=feature_shard.sha256,
                example_ids=canonical_ids,
                model_ids=self.store.model_ids,
                scores_sha256=_calibrated_scores_sha256(
                    canonical_ids,
                    self.store.model_ids,
                    canonical_rows,
                ),
            )
        predictor = _PreparedBatchPredictor(
            model_ids=self.store.model_ids,
            rows_by_prompt_batch=MappingProxyType(rows_by_prompt_batch),
        )
        self.predictors[subset_index] = predictor
        return predictor


def _validate_alignment(
    examples: tuple[EvaluationExample, ...],
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
    *,
    expected_source_fit_sha256: str,
    expected_store_sha256: str,
    expected_raw_score_sha256: str,
) -> None:
    source_digest = prepared_fit_source_sha256(examples, store.plan)
    if not hmac.compare_digest(source_digest, expected_source_fit_sha256):
        raise ValueError("prepared examples do not match the trusted source-fit SHA-256")
    if not hmac.compare_digest(store.source_fit_sha256, expected_source_fit_sha256):
        raise ValueError("prepared store does not match the trusted source-fit SHA-256")
    if not hmac.compare_digest(store.sha256, expected_store_sha256):
        raise ValueError("prepared store does not match the trusted store SHA-256")
    if not hmac.compare_digest(raw_scores.sha256, expected_raw_score_sha256):
        raise ValueError("prepared raw scores do not match the trusted bundle SHA-256")
    if (
        raw_scores.coefficients.source_store_sha256 != store.sha256
        or raw_scores.model_ids != store.model_ids
        or raw_scores.plan != store.plan
    ):
        raise ValueError("prepared raw scores do not descend from the exact feature store")
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    expected_ids = tuple(example.example_id for example in ordered)
    expected_prompts = tuple(
        hashlib.sha256(example.prompt.encode("utf-8")).hexdigest() for example in ordered
    )
    domain_to_index = {domain: index for index, domain in enumerate(store.plan.domains)}
    expected_domains = tuple(domain_to_index[example.domain] for example in ordered)
    expected_models = tuple(sorted(model.model_id for model in ordered[0].candidate_models))
    if (
        store.example_ids != expected_ids
        or store.prompt_sha256s != expected_prompts
        or store.domain_indices != expected_domains
        or store.model_ids != expected_models
    ):
        raise ValueError("prepared store row keys or model catalogue do not match replay data")


def evaluate_prepared_reference_pipeline(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
    ledger_factory: BudgetLedgerFactory,
    *,
    expected_source_fit_sha256: str,
    expected_store_sha256: str,
    expected_raw_score_sha256: str,
    max_candidates_per_tier: int = MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER,
) -> PreparedReferencePipelineResult:
    """Calibrate prepared scores, tune exact lambdas, and replay the existing report path.

    The candidate cap is intentionally bounded.  Every retained candidate set records
    whether it remained exhaustive; callers must inspect
    :attr:`PreparedReferencePipelineResult.all_searches_exhaustive` before making an
    exact-optimum claim.
    """

    if type(examples) is not tuple or not examples:
        raise TypeError("examples must be a non-empty exact tuple")
    if any(type(example) is not EvaluationExample for example in examples):
        raise TypeError("examples must contain exact EvaluationExample values")
    if type(tier_specs) is not tuple or not tier_specs:
        raise TypeError("tier_specs must be a non-empty exact tuple")
    if any(type(spec) is not TierSpec for spec in tier_specs):
        raise TypeError("tier_specs must contain exact TierSpec values")
    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    if type(raw_scores) is not PreparedRawScoreBundle:
        raise TypeError("raw_scores must be an exact PreparedRawScoreBundle")
    if not callable(ledger_factory):
        raise TypeError("ledger_factory must be callable")
    source_sha = _sha256_hex(expected_source_fit_sha256, "expected_source_fit_sha256")
    store_sha = _sha256_hex(expected_store_sha256, "expected_store_sha256")
    raw_sha = _sha256_hex(expected_raw_score_sha256, "expected_raw_score_sha256")
    cap = _exact_nonnegative_int(max_candidates_per_tier, "max_candidates_per_tier")
    if not 2 <= cap <= MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER:
        raise ValueError(
            "max_candidates_per_tier must be between 2 and "
            f"{MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER}"
        )

    plan = _preflight_prepared_input_shape(store, raw_scores)
    if len(examples) != plan.work.example_count:
        raise ValueError("examples do not match the prepared plan row count")

    # Shape admission runs before source traversal, raw-score reads, target reads,
    # isotonic sorting, root materialization, or replay. Cost-aware candidate evidence
    # is admitted after the evaluation scope is snapshotted below.
    estimate_prepared_reference_pipeline(
        plan,
        tier_count=len(tier_specs),
        max_candidates_per_tier=cap,
        execution_estimate=raw_scores.coefficients.execution_estimate,
    )
    _validate_trusted_parent_identities(
        store,
        raw_scores,
        expected_source_fit_sha256=source_sha,
        expected_store_sha256=store_sha,
        expected_raw_score_sha256=raw_sha,
    )

    snapshot_examples, snapshot_specs = _snapshot_evaluation_scope(examples, tier_specs)
    source_digest = prepared_fit_source_sha256(snapshot_examples, plan)
    if not hmac.compare_digest(source_digest, source_sha):
        raise ValueError("prepared examples do not match the trusted source-fit SHA-256")
    if len({example.domain for example in snapshot_examples}) < 4:
        raise ValueError("prepared calibrated nested LODO requires at least four domains")
    outer_folds = leave_one_domain_out(snapshot_examples)
    lambda_search_estimates = tuple(
        estimate_lambda_search(
            fold.training,
            snapshot_specs,
            max_candidates_per_tier=cap,
            allow_large_exhaustive=False,
        )
        for fold in outer_folds
    )
    estimate = estimate_prepared_reference_pipeline(
        plan,
        tier_count=len(snapshot_specs),
        max_candidates_per_tier=cap,
        execution_estimate=raw_scores.coefficients.execution_estimate,
        lambda_search_estimates=lambda_search_estimates,
    )

    # Reconstruct every prepared leaf before trusting stored init=False identities.
    snapshot_store, snapshot_raw = _resnapshot_prepared_inputs(store, raw_scores)
    estimate = estimate_prepared_reference_pipeline(
        snapshot_store.plan,
        tier_count=len(snapshot_specs),
        max_candidates_per_tier=cap,
        execution_estimate=snapshot_raw.execution_estimate,
        lambda_search_estimates=lambda_search_estimates,
    )
    _validate_alignment(
        snapshot_examples,
        snapshot_store,
        snapshot_raw,
        expected_source_fit_sha256=source_sha,
        expected_store_sha256=store_sha,
        expected_raw_score_sha256=raw_sha,
    )
    builder = _PreparedPipelineBuilder(snapshot_examples, snapshot_store, snapshot_raw)
    learned = nested_lodo_lambda_evaluation(
        snapshot_examples,
        snapshot_specs,
        builder.predictor,
        ledger_factory,
        max_candidates_per_tier=cap,
        allow_large_exhaustive=False,
    )

    domain_count = len(snapshot_store.plan.domains)
    expected_subset_indices = tuple(
        index
        for index, subset in enumerate(snapshot_store.plan.training_subsets)
        if len(subset.domain_indices) in (domain_count - 2, domain_count - 1)
    )
    if tuple(sorted(builder.calibrations)) != expected_subset_indices:
        raise AssertionError("nested replay did not consume every prepared calibrator subset")
    expected_block_keys = tuple(
        (subset_index, destination)
        for subset_index in expected_subset_indices
        for destination in range(domain_count)
        if destination not in snapshot_store.plan.training_subsets[subset_index].domain_indices
    )
    if tuple(sorted(builder.calibrated_blocks)) != expected_block_keys:
        raise AssertionError("nested replay did not consume every prepared destination block")

    return PreparedReferencePipelineResult(
        estimate=estimate,
        source_fit_sha256=source_sha,
        store_sha256=store_sha,
        raw_score_bundle_sha256=raw_sha,
        evaluation_data_sha256=evaluation_data_sha256(snapshot_examples),
        evaluation_replay_sha256=evaluation_replay_sha256(snapshot_examples),
        ridge=snapshot_raw.ridge,
        solver_id=PREPARED_MOMENT_RIDGE_SOLVER_ID,
        scorer_id=PREPARED_RAW_SCORER_ID,
        embedding_dimension=snapshot_raw.coefficients.embedding_dimension,
        embedding_identity=_fresh_embedding_identity(snapshot_raw.coefficients.embedding_identity),
        coefficient_block_sha256s=tuple(block.sha256 for block in snapshot_raw.coefficients.blocks),
        scored_feature_shard_sha256s=tuple(
            shard.sha256 for shard in snapshot_raw.feature_shards.shards
        ),
        raw_score_block_sha256s=tuple(block.sha256 for block in snapshot_raw.blocks),
        target_shards=builder.target_shards,
        calibrations=tuple(builder.calibrations[index] for index in expected_subset_indices),
        calibrated_score_blocks=tuple(
            builder.calibrated_blocks[key] for key in expected_block_keys
        ),
        learned=learned,
    )


__all__ = [
    "MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER",
    "MAX_PREPARED_PIPELINE_CANDIDATE_EVIDENCE_BYTES",
    "MAX_PREPARED_PIPELINE_NUMERIC_BYTES",
    "MAX_PREPARED_PIPELINE_PAIR_SCANS",
    "MAX_PREPARED_PIPELINE_REPORT_ROWS",
    "MAX_PREPARED_PIPELINE_UTILITY_EVALUATIONS",
    "MAX_PREPARED_PIPELINE_WORK_UNITS",
    "PREPARED_CALIBRATED_SCORE_ALGORITHM_ID",
    "PREPARED_CALIBRATION_ALGORITHM_ID",
    "PREPARED_REFERENCE_PIPELINE_ALGORITHM_ID",
    "PREPARED_TARGET_SHARD_ALGORITHM_ID",
    "PreparedCalibratedScoreEvidence",
    "PreparedCalibrationEvidence",
    "PreparedReferencePipelineEstimate",
    "PreparedReferencePipelineResult",
    "PreparedTargetShardEvidence",
    "estimate_prepared_reference_pipeline",
    "evaluate_prepared_reference_pipeline",
]
