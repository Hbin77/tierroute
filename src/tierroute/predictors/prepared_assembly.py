# SPDX-License-Identifier: Apache-2.0
"""Bounded in-memory assembly of a final prepared bilinear predictor artifact.

The public assembler deliberately accepts only the three immutable Python reference
parents.  Native receipts, mmap views, executable credentials, policy state, and
source examples are outside this module.  Four caller-trusted digests are checked
before and after a complete canonical resnapshot so cached ``init=False`` identities
cannot hide post-construction mutation.
"""

from __future__ import annotations

import hmac
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass, replace

from tierroute.features import FEATURE_SCHEMA_VERSION, EmbeddingIdentity, PromptFeatureSchema
from tierroute.features.embeddings import MAX_EMBEDDING_IDENTITY_TEXT_BYTES
from tierroute.features.encoding import MAX_FEATURE_METADATA_TEXT_BYTES
from tierroute.features.surface import (
    SURFACE_DOMAIN_TAG_CATALOGUE,
    SURFACE_FEATURE_ALGORITHM_ID,
)
from tierroute.predictors import prepared_execution as _execution
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.prepared_artifacts import (
    MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS,
    PREPARED_ALL_DOMAIN_ASSEMBLY_ALGORITHM_ID,
    PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID,
    PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID,
    PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID,
    PREPARED_PREDICTOR_CALIBRATION_INPUT_ALGORITHM_ID,
    PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID,
    PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID,
    PreparedAllDomainStatistics,
    PreparedArtifactLineage,
    PreparedBilinearPredictorArtifact,
    PreparedCalibrationSource,
    PreparedFinalCoefficient,
    PreparedModelCalibration,
    PreparedModelState,
    PreparedPredictorTargetShard,
    prepared_calibration_input_sha256,
)
from tierroute.predictors.prepared_execution import (
    PREPARED_COEFFICIENT_BLOCK_ALGORITHM_ID,
    PREPARED_COEFFICIENT_BUNDLE_ALGORITHM_ID,
    PREPARED_FEATURE_SHARD_ALGORITHM_ID,
    PREPARED_FEATURE_SHARD_BUNDLE_ALGORITHM_ID,
    PREPARED_MOMENT_RIDGE_SOLVER_ID,
    PREPARED_RAW_SCORE_BLOCK_ALGORITHM_ID,
    PREPARED_RAW_SCORE_BUNDLE_ALGORITHM_ID,
    PREPARED_RAW_SCORER_ID,
    PreparedCoefficientBlock,
    PreparedCoefficientBundle,
    PreparedRawScoreBlock,
    PreparedRawScoreBundle,
    PreparedReferenceExecutionEstimate,
    PreparedScoredFeatureShard,
    PreparedScoredFeatureShardBundle,
    build_prepared_scored_feature_shards,
)
from tierroute.predictors.prepared_graph import (
    MAX_PREPARED_DOMAIN_UTF8_BYTES,
    MAX_PREPARED_DOMAINS,
    MAX_PREPARED_EXAMPLES,
    MAX_PREPARED_FEATURES,
    MAX_PREPARED_TARGETS,
    PREPARED_GRAPH_ALGORITHM_ID,
    PreparedNestedLodoPlan,
    build_prepared_nested_lodo_plan,
)
from tierroute.predictors.prepared_store import (
    MAX_PREPARED_MODEL_ID_UTF8_BYTES,
    MAX_PREPARED_REFERENCE_NUMERIC_BYTES,
    MAX_PREPARED_REFERENCE_STATISTIC_SCALARS,
    MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES,
    PREPARED_FEATURE_STORE_ALGORITHM_ID,
    PREPARED_STATISTICS_ALGORITHM_ID,
    PREPARED_STATISTICS_BUNDLE_ALGORITHM_ID,
    PreparedDomainStatistics,
    PreparedDomainStatisticsBundle,
    PreparedFeatureStore,
    _bounded_text,
    _combine_domain_statistics,
    _packed_upper_index,
    _row_key_text_bytes,
)
from tierroute.predictors.resource_limits import (
    MAX_PREDICTOR_ARTIFACT_BYTES,
    MAX_PREDICTOR_CALIBRATOR_POINTS,
)

MAX_PREPARED_ASSEMBLY_WORK_UNITS = 500_000_000
MAX_PREPARED_ASSEMBLY_MODELED_BYTES = 512 * 1024 * 1024
MAX_PREPARED_ASSEMBLY_OBJECT_BYTES = 256 * 1024 * 1024

_F64_BYTES = 8
_CONTINUOUS_COUNT = 3
_BINARY_COUNT = 2
_TAG_OFFSET = _CONTINUOUS_COUNT + _BINARY_COUNT
_UNIVERSAL_SURFACE_DIMENSION = _TAG_OFFSET + len(SURFACE_DOMAIN_TAG_CATALOGUE)
_PYTHON_OBJECT_ESTIMATE_BYTES = 64
_PYTHON_NUMERIC_SLOT_ESTIMATE_BYTES = 32
_JSON_NUMBER_ESTIMATE_BYTES = 32
_JSON_STRUCTURE_ESTIMATE_BYTES = 24


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_binary64(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must fit finite binary64") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite positive binary64")
    return result


def _sha256_hex(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _snapshot_model_ids_for_preflight(
    value: object,
    *,
    expected_count: int,
    context: str,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{context} model_ids must be an exact tuple")
    if len(value) != expected_count:
        raise ValueError(f"{context} model catalogue has the wrong exact count")
    total_bytes = 0
    for model_id in value:
        _bounded_text(
            model_id,
            f"{context} model_id",
            max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
        )
        total_bytes += len(model_id.encode("utf-8"))
        if total_bytes > MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES:
            raise ValueError(f"{context} model catalogue exceeds the text-byte limit")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{context} model_ids must be sorted and unique")
    return value


def _validate_plan_layout(
    value: object,
    canonical_plan: PreparedNestedLodoPlan,
    context: str,
) -> None:
    """Compare only admitted layout primitives; phase three replaces nested nodes."""

    if type(value) is not PreparedNestedLodoPlan:
        raise TypeError(f"{context} plan must be exact")
    domains = value.domains
    counts = value.domain_example_counts
    if type(domains) is not tuple or len(domains) != len(canonical_plan.domains):
        raise ValueError(f"{context} plan domains have the wrong bounded shape")
    if type(counts) is not tuple or len(counts) != len(canonical_plan.domain_example_counts):
        raise ValueError(f"{context} plan counts have the wrong bounded shape")
    for domain in domains:
        _bounded_text(
            domain,
            f"{context} plan domain",
            max_bytes=MAX_PREPARED_DOMAIN_UTF8_BYTES,
        )
    for count in counts:
        if _exact_nonnegative_int(count, f"{context} plan domain count") == 0:
            raise ValueError(f"{context} plan domain counts must be positive")
    feature_count = _exact_nonnegative_int(value.feature_count, f"{context} feature_count")
    target_count = _exact_nonnegative_int(value.target_count, f"{context} target_count")
    if type(value.algorithm_id) is not str or value.algorithm_id != PREPARED_GRAPH_ALGORITHM_ID:
        raise ValueError(f"{context} plan has an unexpected algorithm identity")
    if (
        domains != canonical_plan.domains
        or counts != canonical_plan.domain_example_counts
        or feature_count != canonical_plan.feature_count
        or target_count != canonical_plan.target_count
    ):
        raise ValueError(f"{context} does not share the canonical prepared plan")


def _snapshot_domain_tags_for_preflight(
    value: object,
    *,
    active_tag_mask: int,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("coefficient schema domain_tags must be an exact tuple")
    if len(value) > len(SURFACE_DOMAIN_TAG_CATALOGUE):
        raise ValueError("coefficient schema has too many domain tags")
    for tag in value:
        _bounded_text(
            tag,
            "coefficient schema domain tag",
            max_bytes=MAX_FEATURE_METADATA_TEXT_BYTES,
        )
    if value != tuple(sorted(set(value))):
        raise ValueError("coefficient schema domain tags must be sorted and unique")
    expected = tuple(
        tag
        for tag_index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE)
        if active_tag_mask & (1 << tag_index)
    )
    if value != expected:
        raise ValueError("coefficient schema tags do not match its active_tag_mask")
    return value


def _validate_execution_estimate_shape(
    estimate: PreparedReferenceExecutionEstimate,
    plan: PreparedNestedLodoPlan,
) -> None:
    active_feature_counts = estimate.active_feature_counts
    if type(active_feature_counts) is not tuple:
        raise TypeError("prepared active_feature_counts must be an exact tuple")
    if len(active_feature_counts) != len(plan.training_subsets):
        raise ValueError("prepared active_feature_counts have the wrong bounded length")
    for width in active_feature_counts:
        _exact_nonnegative_int(width, "prepared active feature count")
        if not 1 <= width <= plan.feature_count:
            raise ValueError("prepared active feature count is outside the plan")
    for name in PreparedReferenceExecutionEstimate.__dataclass_fields__:
        if name not in {"plan", "active_feature_counts"}:
            _exact_nonnegative_int(getattr(estimate, name), name)


@dataclass(frozen=True, slots=True)
class PreparedAllDomainAssemblyEstimate:
    """Closed-form admission evidence for one bounded all-domain assembly."""

    plan: PreparedNestedLodoPlan
    active_feature_count: int
    input_numeric_bytes: int
    row_key_utf8_bytes: int
    aggregate_numeric_bytes: int
    solve_workspace_bytes: int
    target_shard_bytes: int
    per_model_pav_numeric_bytes: int
    retained_numeric_scalars: int
    retained_numeric_bytes: int
    statistics_resnapshot_object_bytes: int
    aggregate_object_bytes: int
    solve_object_bytes: int
    calibration_object_bytes: int
    object_amplification_bytes: int
    canonical_json_upper_bound_bytes: int
    parser_and_staging_bytes: int
    modeled_bytes: int
    resnapshot_work_units: int
    aggregate_work_units: int
    solve_work_units: int
    calibration_work_units: int
    total_work_units: int

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("assembly estimate plan must be exact")
        for name in self.__dataclass_fields__:
            if name != "plan":
                _exact_nonnegative_int(getattr(self, name), name)
        if not 1 <= self.active_feature_count <= self.plan.feature_count:
            raise ValueError("active_feature_count is outside the prepared plan")
        if self.object_amplification_bytes > MAX_PREPARED_ASSEMBLY_OBJECT_BYTES:
            raise ValueError("prepared assembly exceeds the Python-object amplification limit")
        if self.retained_numeric_scalars > MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS:
            raise ValueError("prepared artifact exceeds the numeric-scalar limit")
        if self.canonical_json_upper_bound_bytes > MAX_PREDICTOR_ARTIFACT_BYTES:
            raise ValueError("prepared artifact document estimate exceeds the artifact limit")
        if self.modeled_bytes > MAX_PREPARED_ASSEMBLY_MODELED_BYTES:
            raise ValueError("prepared assembly exceeds the modeled storage limit")
        if self.total_work_units > MAX_PREPARED_ASSEMBLY_WORK_UNITS:
            raise ValueError("prepared assembly exceeds the modeled work limit")


@dataclass(frozen=True, slots=True)
class _PreparedAssemblyShapeSnapshot:
    """Primitive-only facts admitted before the estimator performs arithmetic."""

    plan: PreparedNestedLodoPlan
    active_tag_mask: int
    embedding_dimension: int
    input_numeric_bytes: int
    store_row_key_utf8_bytes: int
    feature_shard_row_key_utf8_bytes: int
    serialized_metadata_bytes: int
    coefficient_cells: int
    score_cells: int

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("assembly shape snapshot plan must be exact")
        for name in self.__dataclass_fields__:
            if name != "plan":
                _exact_nonnegative_int(getattr(self, name), name)


def _active_feature_count(plan: PreparedNestedLodoPlan, mask: int) -> int:
    return _TAG_OFFSET + mask.bit_count() + (plan.feature_count - _UNIVERSAL_SURFACE_DIMENSION)


def _estimate_assembly(
    snapshot: _PreparedAssemblyShapeSnapshot,
) -> PreparedAllDomainAssemblyEstimate:
    """Estimate the complete code-owned traversal without reading numeric leaves."""

    plan = snapshot.plan
    n = plan.work.example_count
    d = plan.feature_count
    m = plan.target_count
    domain_count = len(plan.domains)
    active_width = _active_feature_count(plan, snapshot.active_tag_mask)

    statistics_scalars = domain_count * (d + m + d * (d + 1) // 2 + d * m)
    input_numeric_bytes = snapshot.input_numeric_bytes
    row_key_utf8_bytes = snapshot.store_row_key_utf8_bytes
    aggregate_scalars = d + m + d * (d + 1) // 2 + d * m
    aggregate_numeric_bytes = aggregate_scalars * _F64_BYTES
    solve_workspace_bytes = (
        2 * active_width * active_width + 2 * m * active_width + 2 * active_width + 3 * m
    ) * _F64_BYTES
    target_shard_bytes = max(
        n * m * _F64_BYTES,
        2 * max(plan.domain_example_counts) * m * _F64_BYTES,
    )
    # Assembly processes one model at a time: raw/target lists plus PAV blocks.
    per_model_pav_numeric_bytes = 4 * n * _F64_BYTES
    # The serialized root also retains three schema means, three scales, and ridge.
    retained_numeric_scalars = 7 + m * (active_width + 1) + 2 * n * m
    retained_numeric_bytes = retained_numeric_scalars * _F64_BYTES

    child_objects = (
        1
        + domain_count
        + 1
        + len(plan.training_subsets)
        + 1
        + domain_count
        + 1
        + len(plan.score_blocks)
    )
    row_objects = 4 * n + 4 * n * m
    statistics_resnapshot_object_bytes = (
        statistics_scalars * _PYTHON_NUMERIC_SLOT_ESTIMATE_BYTES
        + (5 * domain_count + 1) * _PYTHON_OBJECT_ESTIMATE_BYTES
    )
    aggregate_object_bytes = (
        3 * aggregate_scalars * _PYTHON_NUMERIC_SLOT_ESTIMATE_BYTES
        + 12 * _PYTHON_OBJECT_ESTIMATE_BYTES
    )
    solve_object_scalars = (
        4 * active_width * active_width + 4 * m * active_width + 12 * active_width + 8 * m
    )
    solve_object_bytes = (
        solve_object_scalars * _PYTHON_NUMERIC_SLOT_ESTIMATE_BYTES
        + (3 * active_width + 4 * m + 16) * _PYTHON_OBJECT_ESTIMATE_BYTES
    )
    calibration_object_scalars = 6 * n * m + m * (active_width + 1)
    calibration_object_bytes = (
        calibration_object_scalars * _PYTHON_NUMERIC_SLOT_ESTIMATE_BYTES
        + (8 * n + 12 * n * m + 12 * m + 8 * domain_count) * _PYTHON_OBJECT_ESTIMATE_BYTES
    )
    object_amplification_bytes = (
        (child_objects + row_objects + 8 * domain_count + 8 * m) * _PYTHON_OBJECT_ESTIMATE_BYTES
        + statistics_resnapshot_object_bytes
        + aggregate_object_bytes
        + solve_object_bytes
        + calibration_object_bytes
    )

    serialized_metadata_bytes = snapshot.serialized_metadata_bytes
    serialized_hash_bytes = (7 + 3 * domain_count + 2 * m) * 64
    calibration_source_fields = 8 * domain_count
    json_structure_count = 80 + 8 * m + calibration_source_fields
    canonical_json_upper_bound_bytes = (
        6 * serialized_metadata_bytes
        + serialized_hash_bytes
        + (retained_numeric_scalars + 4 * domain_count + 16) * _JSON_NUMBER_ESTIMATE_BYTES
        + json_structure_count * _JSON_STRUCTURE_ESTIMATE_BYTES
        + 64 * 1024
    )
    parser_and_staging_bytes = 3 * canonical_json_upper_bound_bytes
    modeled_bytes = (
        input_numeric_bytes
        + row_key_utf8_bytes
        + snapshot.feature_shard_row_key_utf8_bytes
        + aggregate_numeric_bytes
        + solve_workspace_bytes
        + target_shard_bytes
        + per_model_pav_numeric_bytes
        + retained_numeric_bytes
        + object_amplification_bytes
        + parser_and_staging_bytes
    )

    resnapshot_work_units = (
        2 * n * (d + m)
        + n * snapshot.embedding_dimension
        + 2 * domain_count * aggregate_scalars
        + 2 * snapshot.coefficient_cells
        + 2 * snapshot.score_cells
        + n * d
    )
    aggregate_work_units = domain_count * aggregate_scalars
    solve_work_units = active_width**3 + 2 * m * active_width * active_width + m * active_width
    sort_factor = max(1, (n - 1).bit_length())
    calibration_work_units = m * n * (sort_factor + 16)
    total_work_units = (
        resnapshot_work_units + aggregate_work_units + solve_work_units + calibration_work_units
    )
    return PreparedAllDomainAssemblyEstimate(
        plan=plan,
        active_feature_count=active_width,
        input_numeric_bytes=input_numeric_bytes,
        row_key_utf8_bytes=row_key_utf8_bytes,
        aggregate_numeric_bytes=aggregate_numeric_bytes,
        solve_workspace_bytes=solve_workspace_bytes,
        target_shard_bytes=target_shard_bytes,
        per_model_pav_numeric_bytes=per_model_pav_numeric_bytes,
        retained_numeric_scalars=retained_numeric_scalars,
        retained_numeric_bytes=retained_numeric_bytes,
        statistics_resnapshot_object_bytes=statistics_resnapshot_object_bytes,
        aggregate_object_bytes=aggregate_object_bytes,
        solve_object_bytes=solve_object_bytes,
        calibration_object_bytes=calibration_object_bytes,
        object_amplification_bytes=object_amplification_bytes,
        canonical_json_upper_bound_bytes=canonical_json_upper_bound_bytes,
        parser_and_staging_bytes=parser_and_staging_bytes,
        modeled_bytes=modeled_bytes,
        resnapshot_work_units=resnapshot_work_units,
        aggregate_work_units=aggregate_work_units,
        solve_work_units=solve_work_units,
        calibration_work_units=calibration_work_units,
        total_work_units=total_work_units,
    )


def _preflight_input_shape(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    raw_scores: PreparedRawScoreBundle,
) -> tuple[PreparedNestedLodoPlan, PreparedAllDomainAssemblyEstimate]:
    """Validate bounded parent shapes before traversing any numeric leaf."""

    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    if type(statistics) is not PreparedDomainStatisticsBundle:
        raise TypeError("statistics must be an exact PreparedDomainStatisticsBundle")
    if type(raw_scores) is not PreparedRawScoreBundle:
        raise TypeError("raw_scores must be an exact PreparedRawScoreBundle")
    if type(store.plan) is not PreparedNestedLodoPlan:
        raise TypeError("prepared store plan must be exact")
    plan = build_prepared_nested_lodo_plan(
        store.plan.domains,
        store.plan.domain_example_counts,
        feature_count=store.plan.feature_count,
        target_count=store.plan.target_count,
    )
    _validate_plan_layout(store.plan, plan, "prepared store")
    if not (
        4 <= len(plan.domains) <= MAX_PREPARED_DOMAINS
        and plan.work.example_count <= MAX_PREPARED_EXAMPLES
        and plan.feature_count <= MAX_PREPARED_FEATURES
        and plan.target_count <= MAX_PREPARED_TARGETS
    ):
        raise ValueError("prepared assembly shape exceeds the reviewed graph boundary")
    if type(store.algorithm_id) is not str or store.algorithm_id != (
        PREPARED_FEATURE_STORE_ALGORITHM_ID
    ):
        raise ValueError("prepared store has an unexpected algorithm identity")
    model_ids = _snapshot_model_ids_for_preflight(
        store.model_ids,
        expected_count=plan.target_count,
        context="prepared store",
    )
    row_count = plan.work.example_count
    example_ids = store.example_ids
    prompt_sha256s = store.prompt_sha256s
    domain_indices = store.domain_indices
    if type(example_ids) is not tuple or len(example_ids) != row_count:
        raise ValueError("store example_ids have the wrong canonical bounded length")
    if type(prompt_sha256s) is not tuple or len(prompt_sha256s) != row_count:
        raise ValueError("store prompt_sha256s have the wrong canonical bounded length")
    if type(domain_indices) is not tuple or len(domain_indices) != row_count:
        raise ValueError("store domain_indices have the wrong canonical bounded length")
    store_row_key_utf8_bytes = _row_key_text_bytes(
        example_ids,
        prompt_sha256s,
        require_canonical_order=True,
    )
    if any(
        type(domain_index) is not int or not 0 <= domain_index < len(plan.domains)
        for domain_index in domain_indices
    ):
        raise ValueError("store domain indices are outside the canonical plan")
    embedding_dimension = _exact_nonnegative_int(
        store.embedding_dimension,
        "store embedding_dimension",
    )
    if plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + embedding_dimension:
        raise ValueError("store embedding dimension does not match the prepared plan")
    embedding_identity = _fresh_embedding_identity(store.embedding_identity)
    if (embedding_dimension == 0) != (embedding_identity is None):
        raise ValueError("store embedding identity and dimension disagree")
    if embedding_dimension == 0:
        if store.embedding_snapshot_sha256 is not None:
            raise ValueError("surface-only store cannot declare an embedding snapshot")
    else:
        _sha256_hex(store.embedding_snapshot_sha256, "store embedding_snapshot_sha256")
    _sha256_hex(store.source_fit_sha256, "store source_fit_sha256")
    _sha256_hex(store.sha256, "store sha256")

    feature_payload = store.feature_payload
    target_payload = store.target_payload
    if type(feature_payload) is not bytes or type(target_payload) is not bytes:
        raise TypeError("prepared store payloads must be immutable bytes")
    feature_bytes = row_count * plan.feature_count * _F64_BYTES
    target_bytes = row_count * plan.target_count * _F64_BYTES
    if len(feature_payload) != feature_bytes or len(target_payload) != target_bytes:
        raise ValueError("prepared store payloads have the wrong bounded lengths")
    if feature_bytes + target_bytes > MAX_PREPARED_REFERENCE_NUMERIC_BYTES:
        raise ValueError("prepared store exceeds the reference numeric-byte limit")

    _validate_plan_layout(statistics.plan, plan, "prepared statistics")
    if type(statistics.algorithm_id) is not str or statistics.algorithm_id != (
        PREPARED_STATISTICS_BUNDLE_ALGORITHM_ID
    ):
        raise ValueError("prepared statistics have an unexpected algorithm identity")
    _snapshot_model_ids_for_preflight(
        statistics.model_ids,
        expected_count=plan.target_count,
        context="prepared statistics",
    )
    statistics_embedding_dimension = _exact_nonnegative_int(
        statistics.embedding_dimension,
        "statistics embedding_dimension",
    )
    statistics_embedding_identity = _fresh_embedding_identity(statistics.embedding_identity)
    if plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + statistics_embedding_dimension:
        raise ValueError("statistics embedding dimension does not match the prepared plan")
    if (statistics_embedding_dimension == 0) != (statistics_embedding_identity is None):
        raise ValueError("statistics embedding identity and dimension disagree")
    _sha256_hex(statistics.store_sha256, "statistics store_sha256")
    _sha256_hex(statistics.sha256, "statistics sha256")
    domain_statistics = statistics.domain_statistics
    if type(domain_statistics) is not tuple or len(domain_statistics) != len(plan.domains):
        raise ValueError("statistics children have the wrong canonical bounded length")
    if not all(type(child) is PreparedDomainStatistics for child in domain_statistics):
        raise TypeError("statistics children must be exact PreparedDomainStatistics values")
    expected_xx = plan.feature_count * (plan.feature_count + 1) // 2
    expected_xy = plan.feature_count * plan.target_count
    active_tag_mask = 0
    for domain_index, child in enumerate(domain_statistics):
        if (
            type(child.algorithm_id) is not str
            or child.algorithm_id != PREPARED_STATISTICS_ALGORITHM_ID
            or type(child.domain_index) is not int
            or child.domain_index != domain_index
            or type(child.row_count) is not int
            or child.row_count != plan.domain_example_counts[domain_index]
            or type(child.feature_count) is not int
            or child.feature_count != plan.feature_count
            or type(child.target_count) is not int
            or child.target_count != plan.target_count
            or type(child.feature_means) is not tuple
            or len(child.feature_means) != plan.feature_count
            or type(child.target_means) is not tuple
            or len(child.target_means) != plan.target_count
            or type(child.centered_xx_packed) is not tuple
            or len(child.centered_xx_packed) != expected_xx
            or type(child.centered_xy) is not tuple
            or len(child.centered_xy) != expected_xy
            or type(child.active_tag_mask) is not int
            or not 0 <= child.active_tag_mask < 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE)
        ):
            raise ValueError("statistics child has a malformed bounded shape")
        _sha256_hex(child.content_sha256, "statistics child content_sha256")
        _sha256_hex(child.sha256, "statistics child sha256")
        active_tag_mask |= child.active_tag_mask

    coefficients = raw_scores.coefficients
    shards = raw_scores.feature_shards
    if type(coefficients) is not PreparedCoefficientBundle:
        raise TypeError("raw-score coefficients must be an exact PreparedCoefficientBundle")
    if type(shards) is not PreparedScoredFeatureShardBundle:
        raise TypeError("raw-score feature shards must be an exact bundle")
    if type(coefficients.execution_estimate) is not PreparedReferenceExecutionEstimate:
        raise TypeError("prepared execution estimate must be exact")
    _validate_plan_layout(raw_scores.plan, plan, "prepared raw scores")
    _validate_plan_layout(coefficients.plan, plan, "prepared coefficients")
    _validate_plan_layout(
        coefficients.execution_estimate.plan,
        plan,
        "prepared execution estimate",
    )
    _validate_plan_layout(shards.plan, plan, "prepared feature shards")
    _validate_execution_estimate_shape(coefficients.execution_estimate, plan)
    _positive_binary64(coefficients.ridge, "coefficient bundle ridge")
    if (
        type(coefficients.algorithm_id) is not str
        or coefficients.algorithm_id != PREPARED_COEFFICIENT_BUNDLE_ALGORITHM_ID
        or type(shards.algorithm_id) is not str
        or shards.algorithm_id != PREPARED_FEATURE_SHARD_BUNDLE_ALGORITHM_ID
        or type(raw_scores.algorithm_id) is not str
        or raw_scores.algorithm_id != PREPARED_RAW_SCORE_BUNDLE_ALGORITHM_ID
    ):
        raise ValueError("prepared raw-score parent has an unexpected algorithm identity")
    _snapshot_model_ids_for_preflight(
        coefficients.model_ids,
        expected_count=plan.target_count,
        context="prepared coefficients",
    )
    coefficient_embedding_dimension = _exact_nonnegative_int(
        coefficients.embedding_dimension,
        "coefficient embedding_dimension",
    )
    coefficient_embedding_identity = _fresh_embedding_identity(coefficients.embedding_identity)
    if plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + coefficient_embedding_dimension:
        raise ValueError("coefficient embedding dimension does not match the prepared plan")
    if (coefficient_embedding_dimension == 0) != (coefficient_embedding_identity is None):
        raise ValueError("coefficient embedding identity and dimension disagree")
    _sha256_hex(coefficients.source_store_sha256, "coefficient source_store_sha256")
    _sha256_hex(
        coefficients.statistics_bundle_sha256,
        "coefficient statistics_bundle_sha256",
    )
    _sha256_hex(coefficients.sha256, "coefficient bundle sha256")
    domain_active_tag_masks = coefficients.domain_active_tag_masks
    if type(domain_active_tag_masks) is not tuple or len(domain_active_tag_masks) != len(
        plan.domains
    ):
        raise ValueError("coefficient domain tag masks do not match the plan")
    for mask in domain_active_tag_masks:
        value = _exact_nonnegative_int(mask, "coefficient domain active-tag mask")
        if value >= 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE):
            raise ValueError("coefficient domain active-tag mask has an unknown bit")
    shard_embedding_dimension = _exact_nonnegative_int(
        shards.embedding_dimension,
        "feature-shard embedding_dimension",
    )
    shard_embedding_identity = _fresh_embedding_identity(shards.embedding_identity)
    if plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + shard_embedding_dimension:
        raise ValueError("feature-shard embedding dimension does not match the prepared plan")
    if (shard_embedding_dimension == 0) != (shard_embedding_identity is None):
        raise ValueError("feature-shard embedding identity and dimension disagree")
    _sha256_hex(shards.sha256, "feature-shard bundle sha256")
    _sha256_hex(raw_scores.sha256, "raw-score bundle sha256")
    child_specs = (
        (
            coefficients.blocks,
            len(plan.training_subsets),
            PreparedCoefficientBlock,
            "coefficient blocks",
        ),
        (shards.shards, len(plan.domains), PreparedScoredFeatureShard, "feature shards"),
        (raw_scores.blocks, len(plan.score_blocks), PreparedRawScoreBlock, "raw-score blocks"),
    )
    for children, expected_count, child_type, name in child_specs:
        if type(children) is not tuple or len(children) != expected_count:
            raise ValueError(f"{name} have the wrong canonical bounded length")
        if not all(type(child) is child_type for child in children):
            raise TypeError(f"{name} must contain exact project values")
        for child in children:
            _validate_plan_layout(child.plan, plan, name)
    if any(
        type(block.algorithm_id) is not str
        or block.algorithm_id != PREPARED_COEFFICIENT_BLOCK_ALGORITHM_ID
        or type(block.solver_id) is not str
        or block.solver_id != PREPARED_MOMENT_RIDGE_SOLVER_ID
        for block in coefficients.blocks
    ):
        raise ValueError("prepared coefficient block has an unexpected frozen identity")
    coefficient_bytes = 0
    for block_index, block in enumerate(coefficients.blocks):
        _positive_binary64(block.ridge, "coefficient block ridge")
        schema = block.feature_schema
        if (
            type(schema) is not PromptFeatureSchema
            or type(schema.schema_version) is not int
            or schema.schema_version != FEATURE_SCHEMA_VERSION
            or type(schema.continuous_means) is not tuple
            or len(schema.continuous_means) != _CONTINUOUS_COUNT
            or type(schema.continuous_scales) is not tuple
            or len(schema.continuous_scales) != _CONTINUOUS_COUNT
            or type(schema.embedding_dimension) is not int
            or schema.embedding_dimension < 0
            or type(block.active_tag_mask) is not int
            or not 0 <= block.active_tag_mask < 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE)
        ):
            raise ValueError("prepared coefficient block has a malformed bounded shape")
        domain_tags = _snapshot_domain_tags_for_preflight(
            schema.domain_tags,
            active_tag_mask=block.active_tag_mask,
        )
        _fresh_embedding_identity(schema.embedding_identity)
        block_model_ids = _snapshot_model_ids_for_preflight(
            block.model_ids,
            expected_count=plan.target_count,
            context="prepared coefficient block",
        )
        if block_model_ids != model_ids:
            raise ValueError("prepared coefficient block model catalogue does not match the store")
        schema_dimension = _TAG_OFFSET + len(domain_tags) + schema.embedding_dimension
        if schema_dimension > plan.feature_count:
            raise ValueError("prepared coefficient schema exceeds the prepared plan")
        weights_payload = block.weights_payload
        intercepts_payload = block.intercepts_payload
        if type(weights_payload) is not bytes or type(intercepts_payload) is not bytes:
            raise TypeError("prepared coefficient payloads must be immutable bytes")
        expected_weight_bytes = plan.target_count * schema_dimension * _F64_BYTES
        if (
            len(weights_payload) != expected_weight_bytes
            or len(intercepts_payload) != plan.target_count * _F64_BYTES
            or type(block.subset_index) is not int
            or block.subset_index != block_index
        ):
            raise ValueError("prepared coefficient payload has the wrong exact length")
        _sha256_hex(block.subset_statistics_sha256, "subset_statistics_sha256")
        _sha256_hex(block.included_content_sha256, "included_content_sha256")
        _sha256_hex(block.sha256, "coefficient block sha256")
        coefficient_bytes += len(weights_payload) + len(intercepts_payload)
    if any(
        type(shard.algorithm_id) is not str
        or shard.algorithm_id != PREPARED_FEATURE_SHARD_ALGORITHM_ID
        for shard in shards.shards
    ):
        raise ValueError("prepared feature shard has an unexpected frozen identity")
    feature_shard_row_key_utf8_bytes = 0
    for domain_index, shard in enumerate(shards.shards):
        shard_example_ids = shard.example_ids
        shard_prompt_sha256s = shard.prompt_sha256s
        if (
            type(shard.domain_index) is not int
            or shard.domain_index != domain_index
            or type(shard.row_count) is not int
            or shard.row_count != plan.domain_example_counts[domain_index]
            or type(shard_example_ids) is not tuple
            or len(shard_example_ids) != shard.row_count
            or type(shard_prompt_sha256s) is not tuple
            or len(shard_prompt_sha256s) != shard.row_count
        ):
            raise ValueError("prepared feature shard has a malformed bounded shape")
        shard_dimension = _exact_nonnegative_int(
            shard.embedding_dimension,
            "feature-shard child embedding_dimension",
        )
        shard_identity = _fresh_embedding_identity(shard.embedding_identity)
        if plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + shard_dimension:
            raise ValueError("feature-shard child embedding dimension does not match the plan")
        if (shard_dimension == 0) != (shard_identity is None):
            raise ValueError("feature-shard child embedding identity and dimension disagree")
        feature_shard_row_key_utf8_bytes += _row_key_text_bytes(
            shard_example_ids,
            shard_prompt_sha256s,
            require_canonical_order=True,
        )
        if feature_shard_row_key_utf8_bytes > MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES:
            raise ValueError("feature-shard row keys exceed the reference text-byte limit")
        _sha256_hex(shard.feature_content_sha256, "feature-shard content_sha256")
        _sha256_hex(shard.sha256, "feature-shard sha256")
    if any(
        type(block.algorithm_id) is not str
        or block.algorithm_id != PREPARED_RAW_SCORE_BLOCK_ALGORITHM_ID
        or type(block.scorer_id) is not str
        or block.scorer_id != PREPARED_RAW_SCORER_ID
        for block in raw_scores.blocks
    ):
        raise ValueError("prepared raw-score block has an unexpected frozen identity")
    score_bytes = 0
    for block_index, block in enumerate(raw_scores.blocks):
        expected_score_bytes = (
            plan.score_blocks[block_index].row_count * plan.target_count * _F64_BYTES
        )
        block_model_ids = _snapshot_model_ids_for_preflight(
            block.model_ids,
            expected_count=plan.target_count,
            context="prepared raw-score block",
        )
        scores_payload = block.scores_payload
        if type(scores_payload) is not bytes:
            raise TypeError("prepared raw-score payloads must be immutable bytes")
        if (
            type(block.block_index) is not int
            or block.block_index != block_index
            or block_model_ids != model_ids
            or len(scores_payload) != expected_score_bytes
        ):
            raise ValueError("prepared raw-score block has a malformed bounded shape")
        _sha256_hex(block.coefficient_block_sha256, "coefficient_block_sha256")
        _sha256_hex(block.scored_feature_shard_sha256, "scored_feature_shard_sha256")
        _sha256_hex(block.sha256, "raw-score block sha256")
        score_bytes += len(scores_payload)

    per_domain_scalars = (
        plan.feature_count
        + plan.target_count
        + plan.feature_count * (plan.feature_count + 1) // 2
        + plan.feature_count * plan.target_count
    )
    if len(plan.domains) * per_domain_scalars > MAX_PREPARED_REFERENCE_STATISTIC_SCALARS:
        raise ValueError("prepared statistics exceed the reference scalar limit")
    statistics_scalars = len(plan.domains) * per_domain_scalars
    input_numeric_bytes = (
        feature_bytes
        + target_bytes
        + statistics_scalars * _F64_BYTES
        + coefficient_bytes
        + score_bytes
    )
    serialized_metadata_bytes = 2 * sum(
        len(domain.encode("utf-8")) for domain in plan.domains
    ) + sum(len(model_id.encode("utf-8")) for model_id in model_ids)
    serialized_metadata_bytes += sum(
        len(tag.encode("utf-8")) for tag in SURFACE_DOMAIN_TAG_CATALOGUE
    )
    if embedding_identity is not None:
        serialized_metadata_bytes += sum(
            len(getattr(embedding_identity, name).encode("utf-8"))
            for name in ("provider", "model_id", "revision", "pooling")
        )
        serialized_metadata_bytes += len(embedding_identity.asset_manifest_sha256)
    snapshot = _PreparedAssemblyShapeSnapshot(
        plan=plan,
        active_tag_mask=active_tag_mask,
        embedding_dimension=embedding_dimension,
        input_numeric_bytes=input_numeric_bytes,
        store_row_key_utf8_bytes=store_row_key_utf8_bytes,
        feature_shard_row_key_utf8_bytes=feature_shard_row_key_utf8_bytes,
        serialized_metadata_bytes=serialized_metadata_bytes,
        coefficient_cells=coefficient_bytes // _F64_BYTES,
        score_cells=score_bytes // _F64_BYTES,
    )
    estimate = _estimate_assembly(snapshot)
    return plan, estimate


def estimate_prepared_all_domain_assembly(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    raw_scores: PreparedRawScoreBundle,
) -> PreparedAllDomainAssemblyEstimate:
    """Return the exact bounded estimate without reading numeric leaf values."""

    return _preflight_input_shape(store, statistics, raw_scores)[1]


def _fresh_embedding_identity(identity: EmbeddingIdentity | None) -> EmbeddingIdentity | None:
    if identity is None:
        return None
    if type(identity) is not EmbeddingIdentity:
        raise TypeError("embedding identity must be exact")
    text_fields = {
        name: _bounded_text(
            getattr(identity, name),
            f"embedding {name}",
            max_bytes=MAX_EMBEDDING_IDENTITY_TEXT_BYTES,
        )
        for name in ("provider", "model_id", "revision", "pooling")
    }
    if type(identity.normalize) is not bool:
        raise TypeError("embedding normalize must be an exact boolean")
    asset_manifest_sha256 = _sha256_hex(
        identity.asset_manifest_sha256,
        "embedding asset_manifest_sha256",
    )
    return EmbeddingIdentity(
        provider=text_fields["provider"],
        model_id=text_fields["model_id"],
        revision=text_fields["revision"],
        pooling=text_fields["pooling"],
        normalize=identity.normalize,
        asset_manifest_sha256=asset_manifest_sha256,
    )


def _compare_cached_pins(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    raw_scores: PreparedRawScoreBundle,
    *,
    expected_source_fit_sha256: str,
    expected_store_sha256: str,
    expected_statistics_sha256: str,
    expected_raw_score_sha256: str,
) -> None:
    comparisons = (
        (
            store.source_fit_sha256,
            expected_source_fit_sha256,
            "prepared store does not match the trusted source-fit SHA-256",
        ),
        (
            store.sha256,
            expected_store_sha256,
            "prepared store does not match the trusted store SHA-256",
        ),
        (
            statistics.sha256,
            expected_statistics_sha256,
            "prepared statistics do not match the trusted bundle SHA-256",
        ),
        (
            raw_scores.sha256,
            expected_raw_score_sha256,
            "prepared raw scores do not match the trusted bundle SHA-256",
        ),
    )
    for actual, expected, message in comparisons:
        _sha256_hex(actual, "cached parent SHA-256")
        if not hmac.compare_digest(actual, expected):
            raise ValueError(message)


def _resnapshot_inputs(
    plan: PreparedNestedLodoPlan,
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    raw_scores: PreparedRawScoreBundle,
) -> tuple[
    PreparedFeatureStore,
    PreparedDomainStatisticsBundle,
    PreparedRawScoreBundle,
]:
    """Re-run every bounded constructor in the contractually fixed order."""

    store_identity = _fresh_embedding_identity(store.embedding_identity)
    fresh_store = replace(store, plan=plan, embedding_identity=store_identity)

    fresh_domain_statistics = tuple(replace(child) for child in statistics.domain_statistics)
    statistics_identity = _fresh_embedding_identity(statistics.embedding_identity)
    fresh_statistics = replace(
        statistics,
        plan=plan,
        embedding_identity=statistics_identity,
        domain_statistics=fresh_domain_statistics,
    )

    coefficients = raw_scores.coefficients
    coefficient_identity = _fresh_embedding_identity(coefficients.embedding_identity)
    fresh_estimate = replace(coefficients.execution_estimate, plan=plan)
    fresh_coefficient_blocks = []
    for block in coefficients.blocks:
        schema_identity = _fresh_embedding_identity(block.feature_schema.embedding_identity)
        schema = replace(block.feature_schema, embedding_identity=schema_identity)
        fresh_coefficient_blocks.append(replace(block, plan=plan, feature_schema=schema))
    fresh_coefficients = replace(
        coefficients,
        plan=plan,
        embedding_identity=coefficient_identity,
        execution_estimate=fresh_estimate,
        blocks=tuple(fresh_coefficient_blocks),
    )

    shards = raw_scores.feature_shards
    shard_bundle_identity = _fresh_embedding_identity(shards.embedding_identity)
    fresh_shard_rows = tuple(
        replace(
            shard,
            plan=plan,
            embedding_identity=_fresh_embedding_identity(shard.embedding_identity),
        )
        for shard in shards.shards
    )
    fresh_shards = replace(
        shards,
        plan=plan,
        embedding_identity=shard_bundle_identity,
        shards=fresh_shard_rows,
    )
    fresh_raw_blocks = tuple(replace(block, plan=plan) for block in raw_scores.blocks)
    fresh_raw_scores = replace(
        raw_scores,
        coefficients=fresh_coefficients,
        feature_shards=fresh_shards,
        blocks=fresh_raw_blocks,
    )
    return fresh_store, fresh_statistics, fresh_raw_scores


def _validate_cross_parent_associations(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    raw_scores: PreparedRawScoreBundle,
) -> PreparedScoredFeatureShardBundle:
    plan = store.plan
    coefficients = raw_scores.coefficients
    shards = raw_scores.feature_shards
    if (
        statistics.plan != plan
        or raw_scores.plan != plan
        or statistics.store_sha256 != store.sha256
        or coefficients.source_store_sha256 != store.sha256
        or coefficients.statistics_bundle_sha256 != statistics.sha256
        or statistics.model_ids != store.model_ids
        or raw_scores.model_ids != store.model_ids
        or statistics.embedding_identity != store.embedding_identity
        or coefficients.embedding_identity != store.embedding_identity
        or shards.embedding_identity != store.embedding_identity
        or statistics.embedding_dimension != store.embedding_dimension
        or coefficients.embedding_dimension != store.embedding_dimension
        or shards.embedding_dimension != store.embedding_dimension
        or coefficients.domain_active_tag_masks
        != tuple(child.active_tag_mask for child in statistics.domain_statistics)
    ):
        raise ValueError("prepared parents do not share one exact store/plan layout")
    if coefficients.ridge != raw_scores.ridge:
        raise ValueError("prepared coefficient and raw-score ridge values disagree")
    if any(block.solver_id != PREPARED_MOMENT_RIDGE_SOLVER_ID for block in coefficients.blocks):
        raise ValueError("prepared coefficient block has an unexpected solver identity")
    if any(block.scorer_id != PREPARED_RAW_SCORER_ID for block in raw_scores.blocks):
        raise ValueError("prepared raw-score block has an unexpected scorer identity")

    rebuilt = build_prepared_scored_feature_shards(store)
    if rebuilt != shards:
        raise ValueError("prepared scored-feature shards do not match the exact store")
    return rebuilt


def _combine_all_domain_statistics(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
) -> PreparedAllDomainStatistics:
    plan = store.plan
    dimension = plan.feature_count
    target_count = plan.target_count
    feature_means = [0.0] * dimension
    target_means = [0.0] * target_count
    centered_xx = [0.0] * (dimension * (dimension + 1) // 2)
    centered_xy = [0.0] * (dimension * target_count)
    row_count = 0
    active_tag_mask = 0
    for domain_index, domain in enumerate(statistics.domain_statistics):
        if domain.domain_index != domain_index:
            raise ValueError("domain statistics are not in canonical ascending order")
        row_count = _combine_domain_statistics(
            row_count,
            feature_means,
            target_means,
            centered_xx,
            centered_xy,
            domain,
        )
        active_tag_mask |= domain.active_tag_mask
    if row_count != plan.work.example_count:
        raise ValueError("all-domain statistics do not cover the exact prepared store")
    return PreparedAllDomainStatistics(
        plan=plan,
        store_sha256=store.sha256,
        statistics_bundle_sha256=statistics.sha256,
        model_ids=store.model_ids,
        embedding_identity=store.embedding_identity,
        embedding_dimension=store.embedding_dimension,
        domain_statistics_sha256s=tuple(domain.sha256 for domain in statistics.domain_statistics),
        row_count=row_count,
        active_tag_mask=active_tag_mask,
        feature_means=tuple(feature_means),
        target_means=tuple(target_means),
        centered_xx_packed=tuple(centered_xx),
        centered_xy=tuple(centered_xy),
    )


def _all_domain_schema(
    aggregate: PreparedAllDomainStatistics,
) -> tuple[PromptFeatureSchema, tuple[int, ...]]:
    scales = []
    for index in range(_CONTINUOUS_COUNT):
        diagonal = aggregate.centered_xx_packed[
            _packed_upper_index(aggregate.plan.feature_count, index, index)
        ]
        if not math.isfinite(diagonal) or diagonal < 0:
            raise ValueError("all-domain feature variance must be finite and non-negative")
        scale = math.sqrt(diagonal / aggregate.row_count)
        scales.append(scale if scale > 0 else 1.0)
    active_tags = tuple(
        tag
        for tag_index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE)
        if aggregate.active_tag_mask & (1 << tag_index)
    )
    schema = PromptFeatureSchema(
        continuous_means=tuple(aggregate.feature_means[:_CONTINUOUS_COUNT]),  # type: ignore[arg-type]
        continuous_scales=tuple(scales),  # type: ignore[arg-type]
        domain_tags=active_tags,
        embedding_dimension=aggregate.embedding_dimension,
        embedding_identity=aggregate.embedding_identity,
    )
    active_feature_indices = (
        *range(_TAG_OFFSET),
        *(
            _TAG_OFFSET + tag_index
            for tag_index in range(len(SURFACE_DOMAIN_TAG_CATALOGUE))
            if aggregate.active_tag_mask & (1 << tag_index)
        ),
        *range(_UNIVERSAL_SURFACE_DIMENSION, aggregate.plan.feature_count),
    )
    if len(active_feature_indices) != schema.dimension:
        raise AssertionError("all-domain schema width changed after admission")
    return schema, active_feature_indices


def _solve_all_domain_statistics(
    aggregate: PreparedAllDomainStatistics,
    *,
    ridge: float,
) -> PreparedFinalCoefficient:
    schema, active = _all_domain_schema(aggregate)
    width = len(active)
    target_count = len(aggregate.model_ids)
    scale_by_position = tuple(
        schema.continuous_scales[position] if position < _CONTINUOUS_COUNT else 1.0
        for position in range(width)
    )
    matrix = [[0.0] * width for _ in range(width)]
    for row_position, raw_row in enumerate(active):
        for column_position in range(row_position + 1):
            raw_column = active[column_position]
            moment = aggregate.centered_xx_packed[
                _packed_upper_index(aggregate.plan.feature_count, raw_row, raw_column)
            ]
            scaled = _execution._finite_scaled(
                moment,
                scale_by_position[row_position],
                "all-domain Gram scaling",
            )
            scaled = _execution._finite_scaled(
                scaled,
                scale_by_position[column_position],
                "all-domain Gram scaling",
            )
            if row_position == column_position:
                scaled = _execution._ridge_reference._finite_result(
                    scaled + ridge,
                    operation="all-domain ridge regularization",
                )
            matrix[row_position][column_position] = scaled
            matrix[column_position][row_position] = scaled
    normal_matrix = tuple(tuple(row) for row in matrix)
    factor = _execution._ridge_reference._cholesky(normal_matrix)

    weights: list[tuple[float, ...]] = []
    intercepts: list[float] = []
    encoded_means = tuple(
        0.0 if position < _CONTINUOUS_COUNT else aggregate.feature_means[raw_index]
        for position, raw_index in enumerate(active)
    )
    for target_index in range(target_count):
        right_hand_side = tuple(
            _execution._finite_scaled(
                aggregate.centered_xy[raw_index * target_count + target_index],
                scale_by_position[position],
                "all-domain right-hand-side scaling",
            )
            for position, raw_index in enumerate(active)
        )
        target_weights = _execution._ridge_reference._solve_cholesky(
            factor,
            right_hand_side,
        )
        _execution._verify_prepared_residual(
            normal_matrix,
            target_weights,
            right_hand_side,
        )
        products = tuple(
            _execution._ridge_reference._finite_result(
                mean * weight,
                operation="all-domain intercept product",
            )
            for mean, weight in zip(encoded_means, target_weights, strict=True)
        )
        intercept = _execution._ridge_reference._finite_result(
            aggregate.target_means[target_index]
            - _execution._ridge_reference._checked_fsum(
                products,
                operation="all-domain intercept accumulation",
            ),
            operation="all-domain intercept recovery",
        )
        weights.append(tuple(0.0 if value == 0.0 else value for value in target_weights))
        intercepts.append(0.0 if intercept == 0.0 else intercept)
    return PreparedFinalCoefficient(
        feature_schema=schema,
        active_feature_indices=active,
        model_ids=aggregate.model_ids,
        aggregate_statistics_sha256=aggregate.sha256,
        ridge=ridge,
        weights_payload=struct.pack(
            f"<{target_count * width}d",
            *(value for row in weights for value in row),
        ),
        intercepts_payload=struct.pack(f"<{target_count}d", *intercepts),
    )


def _build_target_shards(
    store: PreparedFeatureStore,
    feature_shards: PreparedScoredFeatureShardBundle,
) -> tuple[PreparedPredictorTargetShard, ...]:
    row_by_id = {example_id: row_index for row_index, example_id in enumerate(store.example_ids)}
    if len(row_by_id) != store.plan.work.example_count:
        raise ValueError("prepared store example IDs are not globally unique")
    target_stride = store.plan.target_count * _F64_BYTES
    seen: set[str] = set()
    target_shards = []
    for domain_index, feature_shard in enumerate(feature_shards.shards):
        payload = bytearray(feature_shard.row_count * target_stride)
        for position, example_id in enumerate(feature_shard.example_ids):
            if example_id in seen:
                raise ValueError("calibration target shards contain a duplicate example ID")
            seen.add(example_id)
            try:
                row_index = row_by_id[example_id]
            except KeyError as error:
                raise ValueError(
                    "calibration feature shard contains an unknown example ID"
                ) from error
            if store.domain_indices[row_index] != domain_index:
                raise ValueError("calibration target row belongs to the wrong domain")
            start = row_index * target_stride
            payload[position * target_stride : (position + 1) * target_stride] = (
                store.target_payload[start : start + target_stride]
            )
        target_shards.append(
            PreparedPredictorTargetShard(
                plan=store.plan,
                store_sha256=store.sha256,
                domain_index=domain_index,
                model_ids=store.model_ids,
                scored_feature_shard_sha256=feature_shard.sha256,
                example_ids=feature_shard.example_ids,
                prompt_sha256s=feature_shard.prompt_sha256s,
                targets_payload=bytes(payload),
            )
        )
    if seen != set(store.example_ids):
        raise ValueError("calibration target shards do not form the exact store partition")
    return tuple(target_shards)


def _select_semantic_context_indices(
    domain_count: int,
    subset_contexts: tuple[tuple[int, tuple[int, ...]], ...],
    block_contexts: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Select all-minus-one contexts by meaning, independent of tuple position."""

    if type(domain_count) is not int or not 1 <= domain_count <= MAX_PREPARED_DOMAINS:
        raise ValueError("semantic context domain_count is outside the reviewed range")
    if type(subset_contexts) is not tuple or type(block_contexts) is not tuple:
        raise TypeError("semantic contexts must be exact tuples")
    subset_by_domains: dict[tuple[int, ...], int] = {}
    for subset_index, domains in subset_contexts:
        if (
            type(subset_index) is not int
            or subset_index < 0
            or type(domains) is not tuple
            or domains != tuple(sorted(set(domains)))
        ):
            raise ValueError("semantic subset context is malformed")
        if domains in subset_by_domains:
            raise ValueError("semantic subset contexts contain a duplicate")
        subset_by_domains[domains] = subset_index
    block_by_context: dict[tuple[int, int], int] = {}
    for block_index, subset_index, scored_domain_index in block_contexts:
        if (
            type(block_index) is not int
            or block_index < 0
            or type(subset_index) is not int
            or subset_index < 0
            or type(scored_domain_index) is not int
            or not 0 <= scored_domain_index < domain_count
        ):
            raise ValueError("semantic score-block context is malformed")
        key = (subset_index, scored_domain_index)
        if key in block_by_context:
            raise ValueError("semantic score-block contexts contain a duplicate")
        block_by_context[key] = block_index
    all_domains = tuple(range(domain_count))
    selected = []
    for held_out_domain_index in all_domains:
        training_domains = tuple(
            domain_index for domain_index in all_domains if domain_index != held_out_domain_index
        )
        try:
            subset_index = subset_by_domains[training_domains]
            block_index = block_by_context[(subset_index, held_out_domain_index)]
        except KeyError as error:
            raise ValueError(
                "prepared contexts lack an exact all-but-held-out score block"
            ) from error
        selected.append((subset_index, block_index))
    if (
        len({subset_index for subset_index, _ in selected}) != domain_count
        or len({block_index for _, block_index in selected}) != domain_count
    ):
        raise ValueError("semantic contexts do not identify D distinct sources")
    return tuple(selected)


def _validate_semantic_calibration_contexts(
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
    feature_shards: PreparedScoredFeatureShardBundle,
) -> tuple[tuple[int, int], ...]:
    """Validate all D metadata-only OOF joins before any fit or target extraction."""

    plan = store.plan
    contexts = _select_semantic_context_indices(
        len(plan.domains),
        tuple(
            (subset_index, subset.domain_indices)
            for subset_index, subset in enumerate(plan.training_subsets)
        ),
        tuple(
            (
                block_index,
                graph_block.training_subset_index,
                graph_block.scored_domain_index,
            )
            for block_index, graph_block in enumerate(plan.score_blocks)
        ),
    )
    seen_example_ids: set[str] = set()
    for held_out_domain_index, (subset_index, block_index) in enumerate(contexts):
        graph_block = plan.score_blocks[block_index]
        coefficient = raw_scores.coefficients.blocks[subset_index]
        raw_block = raw_scores.blocks[block_index]
        feature_shard = feature_shards.shards[held_out_domain_index]
        expected_keys = tuple(
            (example_id, prompt_sha256)
            for example_id, prompt_sha256, domain_index in zip(
                store.example_ids,
                store.prompt_sha256s,
                store.domain_indices,
                strict=True,
            )
            if domain_index == held_out_domain_index
        )
        actual_keys = tuple(
            zip(
                feature_shard.example_ids,
                feature_shard.prompt_sha256s,
                strict=True,
            )
        )
        if (
            graph_block.training_subset_index != subset_index
            or graph_block.scored_domain_index != held_out_domain_index
            or graph_block.row_count != feature_shard.row_count
            or raw_block.row_count != feature_shard.row_count
            or raw_block.model_ids != store.model_ids
            or coefficient.model_ids != store.model_ids
            or raw_block.coefficient_block_sha256 != coefficient.sha256
            or raw_block.scored_feature_shard_sha256 != feature_shard.sha256
            or actual_keys != expected_keys
        ):
            raise ValueError("semantic calibration context does not join exactly")
        for example_id in feature_shard.example_ids:
            if example_id in seen_example_ids:
                raise ValueError("semantic calibration contexts contain a duplicate row")
            seen_example_ids.add(example_id)
    if seen_example_ids != set(store.example_ids):
        raise ValueError("semantic calibration contexts do not partition the store")
    return contexts


def _semantic_calibration_sources(
    raw_scores: PreparedRawScoreBundle,
    target_shards: tuple[PreparedPredictorTargetShard, ...],
    contexts: tuple[tuple[int, int], ...],
) -> tuple[PreparedCalibrationSource, ...]:
    plan = raw_scores.plan
    if type(contexts) is not tuple or len(contexts) != len(plan.domains):
        raise ValueError("semantic calibration contexts have the wrong exact count")
    sources = []
    for held_out_domain_index, (subset_index, block_index) in enumerate(contexts):
        raw_block = raw_scores.blocks[block_index]
        feature_shard = raw_scores.feature_shards.shards[held_out_domain_index]
        target_shard = target_shards[held_out_domain_index]
        if (
            raw_block.coefficient_block_sha256
            != raw_scores.coefficients.blocks[subset_index].sha256
            or raw_block.scored_feature_shard_sha256 != feature_shard.sha256
            or target_shard.scored_feature_shard_sha256 != feature_shard.sha256
            or raw_block.row_count != target_shard.row_count
            or target_shard.example_ids != feature_shard.example_ids
            or target_shard.prompt_sha256s != feature_shard.prompt_sha256s
        ):
            raise ValueError("semantic calibration source joins do not match exactly")
        sources.append(
            PreparedCalibrationSource(
                held_out_domain_index=held_out_domain_index,
                held_out_domain=plan.domains[held_out_domain_index],
                training_subset_index=subset_index,
                score_block_index=block_index,
                row_count=raw_block.row_count,
                raw_score_block_sha256=raw_block.sha256,
                scored_feature_shard_sha256=feature_shard.sha256,
                target_shard_sha256=target_shard.sha256,
            )
        )
    return tuple(sources)


def _joined_pairs(
    store: PreparedFeatureStore,
    row_by_id: dict[str, int],
    raw_scores: PreparedRawScoreBundle,
    target_shards: tuple[PreparedPredictorTargetShard, ...],
    sources: tuple[PreparedCalibrationSource, ...],
    model_index: int,
) -> Iterator[tuple[float, float]]:
    target_count = raw_scores.plan.target_count
    target_stride = target_count * _F64_BYTES
    for source in sources:
        raw_block = raw_scores.blocks[source.score_block_index]
        target_shard = target_shards[source.held_out_domain_index]
        for row_index, example_id in enumerate(target_shard.example_ids):
            raw_offset = (row_index * target_count + model_index) * _F64_BYTES
            try:
                store_row_index = row_by_id[example_id]
            except KeyError as error:
                raise ValueError("calibration target shard contains an unknown row") from error
            target_offset = store_row_index * target_stride + model_index * _F64_BYTES
            raw_value = struct.unpack_from("<d", raw_block.scores_payload, raw_offset)[0]
            target_value = struct.unpack_from(
                "<d",
                store.target_payload,
                target_offset,
            )[0]
            yield raw_value, target_value


def _fit_model_states(
    final_coefficient: PreparedFinalCoefficient,
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
    target_shards: tuple[PreparedPredictorTargetShard, ...],
    sources: tuple[PreparedCalibrationSource, ...],
) -> dict[str, PreparedModelState]:
    models: dict[str, PreparedModelState] = {}
    expected_rows = raw_scores.plan.work.example_count
    if expected_rows > MAX_PREDICTOR_CALIBRATOR_POINTS:
        raise ValueError("prepared calibration exceeds the predictor point limit")
    row_by_id = {example_id: row_index for row_index, example_id in enumerate(store.example_ids)}
    if len(row_by_id) != expected_rows:
        raise ValueError("prepared calibration store contains duplicate example IDs")
    for model_index, model_id in enumerate(raw_scores.model_ids):
        input_sha256 = prepared_calibration_input_sha256(
            model_id,
            sources,
            _joined_pairs(
                store,
                row_by_id,
                raw_scores,
                target_shards,
                sources,
                model_index,
            ),
        )
        predictions: list[float] = []
        targets: list[float] = []
        for prediction, target in _joined_pairs(
            store,
            row_by_id,
            raw_scores,
            target_shards,
            sources,
            model_index,
        ):
            predictions.append(prediction)
            targets.append(target)
        if len(predictions) != expected_rows or len(targets) != expected_rows:
            raise ValueError("prepared calibration did not cover every example exactly once")
        calibrator = IsotonicCalibrator.fit(predictions, targets)
        calibration = PreparedModelCalibration(
            model_id=model_id,
            sources=sources,
            input_sha256=input_sha256,
            calibrator=calibrator,
        )
        models[model_id] = PreparedModelState(
            weights=final_coefficient.weights_for_model_index(model_index),
            bias=final_coefficient.intercept_for_model_index(model_index),
            calibration=calibration,
        )
    return models


def assemble_prepared_bilinear_artifact(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    raw_scores: PreparedRawScoreBundle,
    *,
    expected_source_fit_sha256: str,
    expected_store_sha256: str,
    expected_statistics_sha256: str,
    expected_raw_score_sha256: str,
) -> PreparedBilinearPredictorArtifact:
    """Assemble one caller-pinned prepared artifact after complete resnapshotting."""

    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    if type(statistics) is not PreparedDomainStatisticsBundle:
        raise TypeError("statistics must be an exact PreparedDomainStatisticsBundle")
    if type(raw_scores) is not PreparedRawScoreBundle:
        raise TypeError("raw_scores must be an exact PreparedRawScoreBundle")
    pins = (
        _sha256_hex(expected_source_fit_sha256, "expected_source_fit_sha256"),
        _sha256_hex(expected_store_sha256, "expected_store_sha256"),
        _sha256_hex(expected_statistics_sha256, "expected_statistics_sha256"),
        _sha256_hex(expected_raw_score_sha256, "expected_raw_score_sha256"),
    )
    plan, estimate = _preflight_input_shape(store, statistics, raw_scores)
    _compare_cached_pins(
        store,
        statistics,
        raw_scores,
        expected_source_fit_sha256=pins[0],
        expected_store_sha256=pins[1],
        expected_statistics_sha256=pins[2],
        expected_raw_score_sha256=pins[3],
    )
    fresh_store, fresh_statistics, fresh_raw_scores = _resnapshot_inputs(
        plan,
        store,
        statistics,
        raw_scores,
    )
    _compare_cached_pins(
        fresh_store,
        fresh_statistics,
        fresh_raw_scores,
        expected_source_fit_sha256=pins[0],
        expected_store_sha256=pins[1],
        expected_statistics_sha256=pins[2],
        expected_raw_score_sha256=pins[3],
    )
    rebuilt_shards = _validate_cross_parent_associations(
        fresh_store,
        fresh_statistics,
        fresh_raw_scores,
    )
    semantic_contexts = _validate_semantic_calibration_contexts(
        fresh_store,
        fresh_raw_scores,
        rebuilt_shards,
    )

    aggregate = _combine_all_domain_statistics(fresh_store, fresh_statistics)
    final_coefficient = _solve_all_domain_statistics(
        aggregate,
        ridge=fresh_raw_scores.ridge,
    )
    if final_coefficient.feature_schema.dimension != estimate.active_feature_count:
        raise AssertionError("prepared all-domain width changed after admission")
    target_shards = _build_target_shards(fresh_store, rebuilt_shards)
    sources = _semantic_calibration_sources(
        fresh_raw_scores,
        target_shards,
        semantic_contexts,
    )
    models = _fit_model_states(
        final_coefficient,
        fresh_store,
        fresh_raw_scores,
        target_shards,
        sources,
    )
    lineage = PreparedArtifactLineage(
        assembly_algorithm_id=PREPARED_ALL_DOMAIN_ASSEMBLY_ALGORITHM_ID,
        graph_algorithm_id=PREPARED_GRAPH_ALGORITHM_ID,
        surface_feature_algorithm_id=SURFACE_FEATURE_ALGORITHM_ID,
        aggregate_statistics_algorithm_id=PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID,
        final_coefficient_algorithm_id=PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID,
        target_shard_algorithm_id=PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID,
        calibrator_input_algorithm_id=(PREPARED_PREDICTOR_CALIBRATION_INPUT_ALGORITHM_ID),
        calibrator_algorithm_id=PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID,
        source_fit_sha256=fresh_store.source_fit_sha256,
        store_sha256=fresh_store.sha256,
        statistics_bundle_sha256=fresh_statistics.sha256,
        raw_score_bundle_sha256=fresh_raw_scores.sha256,
        embedding_snapshot_sha256=fresh_store.embedding_snapshot_sha256,
        aggregate_statistics_sha256=aggregate.sha256,
        final_coefficient_sha256=final_coefficient.sha256,
        calibration_sources=sources,
    )
    return PreparedBilinearPredictorArtifact(
        algorithm_id=PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID,
        feature_schema=final_coefficient.feature_schema,
        models=models,
        training_domains=plan.domains,
        training_example_count=plan.work.example_count,
        ridge=fresh_raw_scores.ridge,
        lineage=lineage,
    )


__all__ = [
    "MAX_PREPARED_ASSEMBLY_MODELED_BYTES",
    "MAX_PREPARED_ASSEMBLY_OBJECT_BYTES",
    "MAX_PREPARED_ASSEMBLY_WORK_UNITS",
    "PreparedAllDomainAssemblyEstimate",
    "assemble_prepared_bilinear_artifact",
    "estimate_prepared_all_domain_assembly",
]
