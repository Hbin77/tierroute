# SPDX-License-Identifier: Apache-2.0
"""Bounded reference execution for the prepared nested-LODO graph.

This experimental module consumes the immutable raw store and per-domain moments
from :mod:`tierroute.predictors.prepared_store`.  It uses a distinct arithmetic
identity because Welford/Chan moments cannot reproduce the rowwise trainer's
``fmean``/``fsum`` operation order bit for bit.  The reference is intentionally
small: it proves complete-graph coefficient and raw-score wiring on bounded
fixtures, not scalable RouterBench execution or a native prepared session.
"""

from __future__ import annotations

import heapq
import math
import struct
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import pairwise

from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.features.encoding import PromptFeatureSchema
from tierroute.features.surface import (
    SURFACE_DOMAIN_TAG_CATALOGUE,
    SURFACE_FEATURE_ALGORITHM_ID,
)
from tierroute.predictors import _ridge as _ridge_reference
from tierroute.predictors.prepared_graph import (
    PreparedNestedLodoPlan,
)
from tierroute.predictors.prepared_store import (
    MAX_PREPARED_MODEL_ID_UTF8_BYTES,
    MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES,
    PreparedDomainStatisticsBundle,
    PreparedFeatureStore,
    PreparedSubsetStatistics,
    _bounded_text,
    _HashWriter,
    _packed_upper_index,
    _packed_upper_length,
    _row_key_text_bytes,
    _sha256_hex,
    _write_embedding_identity,
    _write_plan_identity,
    combine_prepared_subset_statistics,
)

PREPARED_MOMENT_RIDGE_SOLVER_ID = "tierroute.prepared-moment-ridge-cholesky-python-v1"
PREPARED_COEFFICIENT_BLOCK_ALGORITHM_ID = "tierroute.prepared-coefficient-block-v1"
PREPARED_COEFFICIENT_BUNDLE_ALGORITHM_ID = "tierroute.prepared-coefficient-bundle-v1"
PREPARED_FEATURE_SHARD_CONTENT_ALGORITHM_ID = "tierroute.prepared-scored-feature-content-v1"
PREPARED_FEATURE_SHARD_ALGORITHM_ID = "tierroute.prepared-scored-feature-shard-v1"
PREPARED_FEATURE_SHARD_BUNDLE_ALGORITHM_ID = "tierroute.prepared-scored-feature-shard-bundle-v1"
PREPARED_RAW_SCORER_ID = "tierroute.prepared-raw-dot-product-python-v1"
PREPARED_RAW_SCORE_BLOCK_ALGORITHM_ID = "tierroute.prepared-raw-score-block-v1"
PREPARED_RAW_SCORE_BUNDLE_ALGORITHM_ID = "tierroute.prepared-raw-score-bundle-v1"

# These are aggregate Python-reference ceilings, not native/session promises.  The
# complete planned RouterBench shape is rejected by these limits and by the earlier
# store/statistics limits before any full-dimensional execution begins.
MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS = 100_000_000
MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES = 512 * 1024 * 1024

_CONTINUOUS_COUNT = 3
_BINARY_COUNT = 2
_TAG_OFFSET = _CONTINUOUS_COUNT + _BINARY_COUNT
_UNIVERSAL_SURFACE_DIMENSION = _TAG_OFFSET + len(SURFACE_DOMAIN_TAG_CATALOGUE)
_F64_BYTES = 8
_PREPARED_RESIDUAL_ULP_FACTOR = 2_048.0


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exact_positive_int(value: object, name: str) -> int:
    result = _exact_nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _canonical_f64(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite binary64") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite positive binary64" if positive else "finite binary64"
        raise ValueError(f"{name} must be {qualifier}")
    return 0.0 if result == 0.0 else result


def _validate_f64_payload(payload: bytes, name: str) -> None:
    for (value,) in struct.iter_unpack("<d", payload):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite binary64 values")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise ValueError(f"{name} must use canonical positive zero")


def _canonical_f64_bytes(values: tuple[float, ...], name: str) -> bytes:
    normalized = tuple(_canonical_f64(value, name) for value in values)
    return struct.pack(f"<{len(normalized)}d", *normalized)


def _canonical_f64_matrix_bytes(
    rows: tuple[tuple[float, ...], ...],
    width: int,
    name: str,
) -> bytes:
    payload = bytearray(len(rows) * width * _F64_BYTES)
    for row_index, row in enumerate(rows):
        if len(row) != width:
            raise AssertionError("prepared coefficient matrix width changed during packing")
        for column_index, value in enumerate(row):
            struct.pack_into(
                "<d",
                payload,
                (row_index * width + column_index) * _F64_BYTES,
                _canonical_f64(value, name),
            )
    return bytes(payload)


def _write_feature_schema(writer: _HashWriter, schema: PromptFeatureSchema) -> None:
    writer.integer("feature_schema.version", schema.schema_version)
    writer.floats("feature_schema.continuous_means", schema.continuous_means)
    writer.floats("feature_schema.continuous_scales", schema.continuous_scales)
    for tag in schema.domain_tags:
        writer.text("feature_schema.domain_tag", tag)
    writer.integer("feature_schema.embedding_dimension", schema.embedding_dimension)
    if schema.embedding_identity is not None:
        _write_embedding_identity(writer, schema.embedding_identity)


def _active_feature_indices(
    plan: PreparedNestedLodoPlan,
    schema: PromptFeatureSchema,
) -> tuple[int, ...]:
    if type(plan) is not PreparedNestedLodoPlan:
        raise TypeError("plan must be an exact PreparedNestedLodoPlan")
    if type(schema) is not PromptFeatureSchema:
        raise TypeError("feature_schema must be an exact PromptFeatureSchema")
    if plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + schema.embedding_dimension:
        raise ValueError("feature schema embedding width does not match the prepared plan")
    catalogue_index = {tag: index for index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE)}
    try:
        active_tags = tuple(_TAG_OFFSET + catalogue_index[tag] for tag in schema.domain_tags)
    except KeyError as error:
        raise ValueError("feature schema contains a tag outside the fixed catalogue") from error
    embeddings = tuple(range(_UNIVERSAL_SURFACE_DIMENSION, plan.feature_count))
    active = (*range(_TAG_OFFSET), *active_tags, *embeddings)
    if len(active) != schema.dimension or any(left >= right for left, right in pairwise(active)):
        raise ValueError("feature schema does not map to a canonical prepared coordinate order")
    return active


def _validate_model_ids(model_ids: object, target_count: int, context: str) -> tuple[str, ...]:
    if type(model_ids) is not tuple:
        raise TypeError(f"{context} model_ids must be an exact tuple")
    if len(model_ids) != target_count:
        raise ValueError(f"{context} model catalogue has the wrong target count")
    for model_id in model_ids:
        _bounded_text(
            model_id,
            f"{context} model_id",
            max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
        )
    if model_ids != tuple(sorted(set(model_ids))):
        raise ValueError(f"{context} model_ids must be sorted and unique")
    return model_ids


def _active_widths_from_statistics(
    statistics: PreparedDomainStatisticsBundle,
) -> tuple[int, ...]:
    widths: list[int] = []
    for subset in statistics.plan.training_subsets:
        mask = 0
        for domain_index in subset.domain_indices:
            mask |= statistics.domain_statistics[domain_index].active_tag_mask
        widths.append(_TAG_OFFSET + mask.bit_count() + statistics.embedding_dimension)
    return tuple(widths)


def _estimate_execution_values(
    plan: PreparedNestedLodoPlan,
    active_feature_counts: tuple[int, ...],
) -> dict[str, int]:
    dimension = plan.feature_count
    target_count = plan.target_count
    per_subset_moment_cells = (
        dimension + target_count + _packed_upper_length(dimension) + dimension * target_count
    )
    combination_work = sum(
        len(subset.domain_indices) * per_subset_moment_cells for subset in plan.training_subsets
    )
    # Each sequentially combined subset is copied to immutable tuples, validated,
    # and content-hashed before it is solved and discarded.
    subset_statistics_scan_work = 3 * len(plan.training_subsets) * per_subset_moment_cells
    transform_work = sum(
        width * (width + 1) // 2 + width * target_count for width in active_feature_counts
    )
    # Active coordinates are derived from each subset statistic for solving, then
    # validated by each coefficient record, then retained once by the raw scorer.
    coordinate_preparation_work = 3 * sum(active_feature_counts)
    solve_work = sum(
        width**3 + 2 * target_count * width * width + target_count * width
        for width in active_feature_counts
    )
    score_selection_work = len(plan.score_blocks) * plan.work.example_count
    feature_hash_work = plan.work.example_count * dimension
    score_work = sum(
        block.row_count * target_count * active_feature_counts[block.training_subset_index]
        for block in plan.score_blocks
    )
    score_decode_encode_work = sum(
        block.row_count
        * (
            dimension
            + active_feature_counts[block.training_subset_index]
            + target_count * active_feature_counts[block.training_subset_index]
            + 3 * target_count
        )
        for block in plan.score_blocks
    )
    coefficient_cells = sum(target_count * (width + 1) for width in active_feature_counts)
    score_cells = sum(block.row_count * target_count for block in plan.score_blocks)
    # Coefficients are normalized, packed, validated, and hashed. Scores are packed
    # in the decode/encode term above, then validated and hashed here.
    numeric_payload_work = 4 * coefficient_cells + 2 * score_cells
    coefficient_bytes = coefficient_cells * _F64_BYTES
    score_bytes = score_cells * _F64_BYTES
    maximum_coefficient_block_bytes = target_count * (max(active_feature_counts) + 1) * _F64_BYTES
    maximum_score_block_bytes = (
        max(block.row_count for block in plan.score_blocks) * target_count * _F64_BYTES
    )
    feature_hash_row_bytes = dimension * _F64_BYTES
    subset_statistics_transient_bytes = per_subset_moment_cells * _F64_BYTES
    active_coordinate_cache_bytes = sum(active_feature_counts) * _F64_BYTES
    copy_transient_bytes = max(
        maximum_coefficient_block_bytes,
        maximum_score_block_bytes,
        feature_hash_row_bytes,
    )
    modeled_numeric_storage_bytes = (
        plan.work.feature_cache_bytes
        + plan.work.target_cache_bytes
        + plan.work.domain_statistics_bytes
        + coefficient_bytes
        + score_bytes
        + plan.work.solve_workspace_bytes
        + subset_statistics_transient_bytes
        + active_coordinate_cache_bytes
        + copy_transient_bytes
    )
    total_work = (
        combination_work
        + subset_statistics_scan_work
        + transform_work
        + coordinate_preparation_work
        + solve_work
        + score_selection_work
        + feature_hash_work
        + score_work
        + score_decode_encode_work
        + numeric_payload_work
    )
    return {
        "subset_combination_work_units": combination_work,
        "subset_statistics_scan_work_units": subset_statistics_scan_work,
        "moment_transform_work_units": transform_work,
        "coordinate_preparation_work_units": coordinate_preparation_work,
        "solve_work_units": solve_work,
        "score_selection_work_units": score_selection_work,
        "feature_hash_work_units": feature_hash_work,
        "score_work_units": score_work,
        "score_decode_encode_work_units": score_decode_encode_work,
        "numeric_payload_work_units": numeric_payload_work,
        "total_work_units": total_work,
        "coefficient_cells": coefficient_cells,
        "score_cells": score_cells,
        "coefficient_bytes": coefficient_bytes,
        "score_bytes": score_bytes,
        "workspace_bytes": plan.work.solve_workspace_bytes,
        "subset_statistics_transient_bytes": subset_statistics_transient_bytes,
        "active_coordinate_cache_bytes": active_coordinate_cache_bytes,
        "feature_hash_row_bytes": feature_hash_row_bytes,
        "copy_transient_bytes": copy_transient_bytes,
        "modeled_numeric_storage_bytes": modeled_numeric_storage_bytes,
    }


@dataclass(frozen=True, slots=True)
class PreparedReferenceExecutionEstimate:
    """Aggregate admission estimate for one complete solve-and-score reference run."""

    plan: PreparedNestedLodoPlan
    active_feature_counts: tuple[int, ...]
    subset_combination_work_units: int
    subset_statistics_scan_work_units: int
    moment_transform_work_units: int
    coordinate_preparation_work_units: int
    solve_work_units: int
    score_selection_work_units: int
    feature_hash_work_units: int
    score_work_units: int
    score_decode_encode_work_units: int
    numeric_payload_work_units: int
    total_work_units: int
    coefficient_cells: int
    score_cells: int
    coefficient_bytes: int
    score_bytes: int
    workspace_bytes: int
    subset_statistics_transient_bytes: int
    active_coordinate_cache_bytes: int
    feature_hash_row_bytes: int
    copy_transient_bytes: int
    modeled_numeric_storage_bytes: int

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("execution estimate plan must be exact")
        if type(self.active_feature_counts) is not tuple:
            raise TypeError("active_feature_counts must be an exact tuple")
        if len(self.active_feature_counts) != len(self.plan.training_subsets):
            raise ValueError("active_feature_counts do not match the prepared subsets")
        minimum_width = _TAG_OFFSET + (self.plan.feature_count - _UNIVERSAL_SURFACE_DIMENSION)
        for width in self.active_feature_counts:
            _exact_positive_int(width, "active feature count")
            if not minimum_width <= width <= self.plan.feature_count:
                raise ValueError("active feature count is outside the prepared raw layout")
        expected = _estimate_execution_values(self.plan, self.active_feature_counts)
        for name, value in expected.items():
            actual = _exact_nonnegative_int(getattr(self, name), name)
            if actual != value:
                raise ValueError(f"{name} does not match the reference execution formula")
        if self.coefficient_bytes > self.plan.work.coefficient_cache_bytes:
            raise ValueError("prepared coefficient bytes exceed the graph estimate")
        if self.score_cells != self.plan.work.scalar_score_count:
            raise ValueError("prepared score cells do not match the graph estimate")
        if self.score_bytes != self.plan.work.raw_score_cache_bytes:
            raise ValueError("prepared score bytes do not match the graph estimate")
        if self.score_work_units > self.plan.work.score_work_units:
            raise ValueError("prepared score work exceeds the graph estimate")
        if self.workspace_bytes > self.plan.work.solve_workspace_bytes:
            raise ValueError("prepared workspace exceeds the graph estimate")
        if self.total_work_units > MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS:
            raise ValueError("prepared reference execution exceeds the aggregate work limit")
        if self.modeled_numeric_storage_bytes > MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES:
            raise ValueError("prepared reference execution exceeds the numeric-storage limit")


def _execution_estimate(
    plan: PreparedNestedLodoPlan,
    active_feature_counts: tuple[int, ...],
) -> PreparedReferenceExecutionEstimate:
    return PreparedReferenceExecutionEstimate(
        plan=plan,
        active_feature_counts=active_feature_counts,
        **_estimate_execution_values(plan, active_feature_counts),
    )


@dataclass(frozen=True, slots=True)
class PreparedCoefficientBlock:
    """One target-major prepared-moment ridge solution for a canonical subset."""

    plan: PreparedNestedLodoPlan
    subset_index: int
    model_ids: tuple[str, ...]
    feature_schema: PromptFeatureSchema
    active_tag_mask: int
    subset_statistics_sha256: str
    included_content_sha256: str
    ridge: float
    weights_payload: bytes = field(repr=False)
    intercepts_payload: bytes = field(repr=False)
    sha256: str = field(init=False)
    solver_id: str = field(default=PREPARED_MOMENT_RIDGE_SOLVER_ID, init=False)
    algorithm_id: str = field(default=PREPARED_COEFFICIENT_BLOCK_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("coefficient plan must be exact")
        subset_index = _exact_nonnegative_int(self.subset_index, "coefficient subset_index")
        if subset_index >= len(self.plan.training_subsets):
            raise ValueError("coefficient subset_index is outside the prepared plan")
        _validate_model_ids(self.model_ids, self.plan.target_count, "coefficient")
        active = _active_feature_indices(self.plan, self.feature_schema)
        active_tag_mask = _exact_nonnegative_int(
            self.active_tag_mask,
            "coefficient active_tag_mask",
        )
        if active_tag_mask >= 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE):
            raise ValueError("coefficient active_tag_mask contains an unknown tag bit")
        expected_tags = tuple(
            tag
            for tag_index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE)
            if active_tag_mask & (1 << tag_index)
        )
        if self.feature_schema.domain_tags != expected_tags:
            raise ValueError("coefficient schema tags do not match its active_tag_mask")
        if any(
            value == 0.0 and math.copysign(1.0, value) < 0
            for value in self.feature_schema.continuous_means
        ):
            raise ValueError("coefficient feature schema means must use positive zero")
        _sha256_hex(self.subset_statistics_sha256, "subset_statistics_sha256")
        _sha256_hex(self.included_content_sha256, "included_content_sha256")
        ridge = _canonical_f64(self.ridge, "coefficient ridge", positive=True)
        if type(self.weights_payload) is not bytes or type(self.intercepts_payload) is not bytes:
            raise TypeError("coefficient payloads must be immutable bytes")
        expected_weight_bytes = self.plan.target_count * len(active) * _F64_BYTES
        expected_intercept_bytes = self.plan.target_count * _F64_BYTES
        if (
            expected_weight_bytes + expected_intercept_bytes
            > MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES
        ):
            raise ValueError("coefficient block exceeds the reference numeric-storage limit")
        if len(self.weights_payload) != expected_weight_bytes:
            raise ValueError("coefficient weights payload has the wrong exact length")
        if len(self.intercepts_payload) != expected_intercept_bytes:
            raise ValueError("coefficient intercept payload has the wrong exact length")
        _validate_f64_payload(self.weights_payload, "coefficient weights payload")
        _validate_f64_payload(self.intercepts_payload, "coefficient intercept payload")
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "sha256", _coefficient_block_sha256(self, active))

    @property
    def active_feature_indices(self) -> tuple[int, ...]:
        """Return the schema-derived universal raw coordinates."""

        return _active_feature_indices(self.plan, self.feature_schema)

    @property
    def feature_count(self) -> int:
        return self.feature_schema.dimension

    def weights_for_model_index(self, model_index: int) -> tuple[float, ...]:
        index = _exact_nonnegative_int(model_index, "model_index")
        if index >= self.plan.target_count:
            raise IndexError("model_index is outside the coefficient catalogue")
        return struct.unpack_from(
            f"<{self.feature_count}d",
            self.weights_payload,
            index * self.feature_count * _F64_BYTES,
        )

    def intercept_for_model_index(self, model_index: int) -> float:
        index = _exact_nonnegative_int(model_index, "model_index")
        if index >= self.plan.target_count:
            raise IndexError("model_index is outside the coefficient catalogue")
        return struct.unpack_from("<d", self.intercepts_payload, index * _F64_BYTES)[0]


def _coefficient_block_sha256(
    block: PreparedCoefficientBlock,
    active_feature_indices: tuple[int, ...],
) -> str:
    writer = _HashWriter(PREPARED_COEFFICIENT_BLOCK_ALGORITHM_ID)
    _write_plan_identity(writer, block.plan)
    writer.text("solver_id", block.solver_id)
    writer.integer("subset_index", block.subset_index)
    subset = block.plan.training_subsets[block.subset_index]
    for domain_index in subset.domain_indices:
        writer.integer("training_domain_index", domain_index)
    writer.integer("training_row_count", subset.row_count)
    writer.text("subset_statistics_sha256", block.subset_statistics_sha256)
    writer.text("included_content_sha256", block.included_content_sha256)
    for model_id in block.model_ids:
        writer.text("model_id", model_id)
    writer.integer("active_tag_mask", block.active_tag_mask)
    _write_feature_schema(writer, block.feature_schema)
    for feature_index in active_feature_indices:
        writer.integer("active_feature_index", feature_index)
    writer.token("ridge.f64le", struct.pack("<d", block.ridge))
    writer.token("weights.target-major.f64le", block.weights_payload)
    writer.token("intercepts.f64le", block.intercepts_payload)
    return writer.hexdigest()


def _finite_scaled(value: float, divisor: float, name: str) -> float:
    try:
        result = value / divisor
    except (OverflowError, ZeroDivisionError) as error:
        raise ArithmeticError(f"{name} was not finite") from error
    if not math.isfinite(result):
        raise ArithmeticError(f"{name} was not finite")
    return 0.0 if result == 0.0 else result


def _verify_prepared_residual(
    matrix: tuple[tuple[float, ...], ...],
    solution: tuple[float, ...],
    right_hand_side: tuple[float, ...],
) -> None:
    """Reject implausible prepared solves under the moment-path arithmetic contract.

    The prepared path has a distinct solver identity and empirical residual gate.
    Frozen ordinary-collinearity and high-dynamic stress corpora need more headroom
    than the row solver's 512-factor gate; 2,048 keeps a reviewed margin while the
    tolerance remains dimension- and magnitude-scaled.  This is a regression guard,
    not a universal forward-error bound. Every normal equation is still checked.
    """

    dimension = len(matrix)
    for row_index, row in enumerate(matrix):
        products = tuple(
            _ridge_reference._finite_result(
                value * coefficient,
                operation="prepared residual product",
            )
            for value, coefficient in zip(row, solution, strict=True)
        )
        reconstructed = _ridge_reference._checked_fsum(
            products,
            operation="prepared residual accumulation",
        )
        residual = abs(
            _ridge_reference._finite_result(
                reconstructed - right_hand_side[row_index],
                operation="prepared residual subtraction",
            )
        )
        scale = _ridge_reference._checked_fsum(
            (abs(value) for value in (*products, right_hand_side[row_index])),
            operation="prepared residual scale",
        )
        tolerance = _ridge_reference._finite_result(
            _PREPARED_RESIDUAL_ULP_FACTOR
            * max(1, dimension)
            * max(sys.float_info.epsilon * scale, math.ulp(scale)),
            operation="prepared residual tolerance",
        )
        if residual > tolerance:
            raise ArithmeticError(
                "prepared ridge solve failed residual verification at equation "
                f"{row_index}: residual={residual!r}, tolerance={tolerance!r}"
            )


def _solve_subset_statistics(
    statistics: PreparedSubsetStatistics,
    ridge: float,
) -> PreparedCoefficientBlock:
    active = statistics.active_feature_indices
    width = len(active)
    target_count = len(statistics.model_ids)
    scale_by_position = tuple(
        statistics.feature_schema.continuous_scales[position]
        if position < _CONTINUOUS_COUNT
        else 1.0
        for position in range(width)
    )

    matrix = [[0.0] * width for _ in range(width)]
    for row_position, raw_row in enumerate(active):
        for column_position in range(row_position + 1):
            raw_column = active[column_position]
            moment = statistics.centered_xx_packed[
                _packed_upper_index(statistics.plan.feature_count, raw_row, raw_column)
            ]
            scaled = _finite_scaled(moment, scale_by_position[row_position], "Gram scaling")
            scaled = _finite_scaled(
                scaled,
                scale_by_position[column_position],
                "Gram scaling",
            )
            if row_position == column_position:
                scaled = _ridge_reference._finite_result(
                    scaled + ridge,
                    operation="prepared ridge regularization",
                )
            matrix[row_position][column_position] = scaled
            matrix[column_position][row_position] = scaled
    normal_matrix = tuple(tuple(row) for row in matrix)
    factor = _ridge_reference._cholesky(normal_matrix)

    weights: list[tuple[float, ...]] = []
    intercepts: list[float] = []
    encoded_means = tuple(
        0.0 if position < _CONTINUOUS_COUNT else statistics.feature_means[raw_index]
        for position, raw_index in enumerate(active)
    )
    for target_index in range(target_count):
        right_hand_side = tuple(
            _finite_scaled(
                statistics.centered_xy[raw_index * target_count + target_index],
                scale_by_position[position],
                "right-hand-side scaling",
            )
            for position, raw_index in enumerate(active)
        )
        target_weights = _ridge_reference._solve_cholesky(factor, right_hand_side)
        _verify_prepared_residual(normal_matrix, target_weights, right_hand_side)
        intercept_products = tuple(
            _ridge_reference._finite_result(
                mean * weight,
                operation="prepared intercept product",
            )
            for mean, weight in zip(encoded_means, target_weights, strict=True)
        )
        intercept = _ridge_reference._finite_result(
            statistics.target_means[target_index]
            - _ridge_reference._checked_fsum(
                intercept_products,
                operation="prepared intercept accumulation",
            ),
            operation="prepared intercept recovery",
        )
        weights.append(tuple(0.0 if value == 0.0 else value for value in target_weights))
        intercepts.append(0.0 if intercept == 0.0 else intercept)

    return PreparedCoefficientBlock(
        plan=statistics.plan,
        subset_index=statistics.subset_index,
        model_ids=statistics.model_ids,
        feature_schema=statistics.feature_schema,
        active_tag_mask=statistics.active_tag_mask,
        subset_statistics_sha256=statistics.sha256,
        included_content_sha256=statistics.included_content_sha256,
        ridge=ridge,
        weights_payload=_canonical_f64_matrix_bytes(
            tuple(weights),
            width,
            "prepared coefficient",
        ),
        intercepts_payload=_canonical_f64_bytes(
            tuple(intercepts),
            "prepared intercept",
        ),
    )


@dataclass(frozen=True, slots=True)
class PreparedCoefficientBundle:
    """All canonical subset solutions retained without retaining subset moments."""

    plan: PreparedNestedLodoPlan
    model_ids: tuple[str, ...]
    source_store_sha256: str
    embedding_identity: EmbeddingIdentity | None
    embedding_dimension: int
    domain_active_tag_masks: tuple[int, ...]
    ridge: float
    statistics_bundle_sha256: str
    execution_estimate: PreparedReferenceExecutionEstimate
    blocks: tuple[PreparedCoefficientBlock, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_COEFFICIENT_BUNDLE_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("coefficient bundle plan must be exact")
        _validate_model_ids(self.model_ids, self.plan.target_count, "coefficient bundle")
        _sha256_hex(self.source_store_sha256, "coefficient source_store_sha256")
        embedding_dimension = _exact_nonnegative_int(
            self.embedding_dimension,
            "coefficient embedding_dimension",
        )
        if self.plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + embedding_dimension:
            raise ValueError("coefficient embedding width does not match the plan")
        if (embedding_dimension == 0) != (self.embedding_identity is None):
            raise ValueError("coefficient embedding identity and dimension disagree")
        if (
            self.embedding_identity is not None
            and type(self.embedding_identity) is not EmbeddingIdentity
        ):
            raise TypeError("coefficient embedding identity must be exact")
        if type(self.domain_active_tag_masks) is not tuple or len(
            self.domain_active_tag_masks
        ) != len(self.plan.domains):
            raise ValueError("coefficient domain tag masks do not match the plan")
        for mask in self.domain_active_tag_masks:
            value = _exact_nonnegative_int(mask, "coefficient domain active-tag mask")
            if value >= 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE):
                raise ValueError("coefficient domain active-tag mask has an unknown bit")
        ridge = _canonical_f64(self.ridge, "coefficient bundle ridge", positive=True)
        _sha256_hex(self.statistics_bundle_sha256, "statistics_bundle_sha256")
        if type(self.execution_estimate) is not PreparedReferenceExecutionEstimate:
            raise TypeError("execution_estimate must be exact")
        if self.execution_estimate.plan != self.plan:
            raise ValueError("execution estimate plan does not match the coefficient bundle")
        if type(self.blocks) is not tuple:
            raise TypeError("coefficient blocks must be an exact tuple")
        if len(self.blocks) != len(self.plan.training_subsets):
            raise ValueError("coefficient blocks have the wrong bounded length")
        if not all(type(block) is PreparedCoefficientBlock for block in self.blocks):
            raise TypeError("coefficient blocks must contain exact block values")
        active_widths = tuple(block.feature_count for block in self.blocks)
        if active_widths != self.execution_estimate.active_feature_counts:
            raise ValueError("coefficient widths do not match the execution estimate")
        payload_bytes = sum(
            len(block.weights_payload) + len(block.intercepts_payload) for block in self.blocks
        )
        if payload_bytes != self.execution_estimate.coefficient_bytes:
            raise ValueError("coefficient payload bytes do not match the execution estimate")
        for position, block in enumerate(self.blocks):
            subset = self.plan.training_subsets[position]
            expected_mask = 0
            for domain_index in subset.domain_indices:
                expected_mask |= self.domain_active_tag_masks[domain_index]
            if (
                block.plan != self.plan
                or block.subset_index != position
                or block.model_ids != self.model_ids
                or block.ridge != ridge
                or block.active_tag_mask != expected_mask
                or block.feature_schema.embedding_dimension != embedding_dimension
                or block.feature_schema.embedding_identity != self.embedding_identity
            ):
                raise ValueError("coefficient block does not match its canonical bundle position")
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "sha256", _coefficient_bundle_sha256(self))


def _coefficient_bundle_sha256(bundle: PreparedCoefficientBundle) -> str:
    writer = _HashWriter(PREPARED_COEFFICIENT_BUNDLE_ALGORITHM_ID)
    _write_plan_identity(writer, bundle.plan)
    writer.text("solver_id", PREPARED_MOMENT_RIDGE_SOLVER_ID)
    writer.text("source_store_sha256", bundle.source_store_sha256)
    writer.text("statistics_bundle_sha256", bundle.statistics_bundle_sha256)
    writer.integer("embedding_dimension", bundle.embedding_dimension)
    if bundle.embedding_identity is not None:
        _write_embedding_identity(writer, bundle.embedding_identity)
    for domain_index, active_tag_mask in enumerate(bundle.domain_active_tag_masks):
        writer.integer("domain_active_tag_mask.domain_index", domain_index)
        writer.integer("domain_active_tag_mask.value", active_tag_mask)
    writer.token("ridge.f64le", struct.pack("<d", bundle.ridge))
    for model_id in bundle.model_ids:
        writer.text("model_id", model_id)
    for block in bundle.blocks:
        writer.text("coefficient_block_sha256", block.sha256)
    return writer.hexdigest()


def build_prepared_coefficient_bundle(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    *,
    ridge: float,
) -> PreparedCoefficientBundle:
    """Preflight the whole reference graph, then combine/solve one subset at a time."""

    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    if type(statistics) is not PreparedDomainStatisticsBundle:
        raise TypeError("statistics must be an exact PreparedDomainStatisticsBundle")
    ridge_value = _canonical_f64(ridge, "prepared ridge", positive=True)
    if (
        statistics.plan != store.plan
        or statistics.store_sha256 != store.sha256
        or statistics.model_ids != store.model_ids
        or statistics.embedding_dimension != store.embedding_dimension
        or statistics.embedding_identity != store.embedding_identity
    ):
        raise ValueError("statistics do not match the exact prepared feature store")

    # Only bounded masks and graph nodes are inspected before this cumulative
    # admission check. No moment array is combined and no solve/output buffer is
    # allocated until the complete solve-and-score shape is accepted.
    active_widths = _active_widths_from_statistics(statistics)
    estimate = _execution_estimate(store.plan, active_widths)

    blocks: list[PreparedCoefficientBlock] = []
    for subset_index in range(len(store.plan.training_subsets)):
        subset_statistics = combine_prepared_subset_statistics(statistics, subset_index)
        block = _solve_subset_statistics(subset_statistics, ridge_value)
        if block.feature_count != active_widths[subset_index]:
            raise AssertionError("prepared subset width changed after preflight")
        blocks.append(block)
        # The full d²+dM moment object is deliberately not retained across subsets.
        del subset_statistics
    return PreparedCoefficientBundle(
        plan=store.plan,
        model_ids=store.model_ids,
        source_store_sha256=store.sha256,
        embedding_identity=store.embedding_identity,
        embedding_dimension=store.embedding_dimension,
        domain_active_tag_masks=tuple(
            domain.active_tag_mask for domain in statistics.domain_statistics
        ),
        ridge=ridge_value,
        statistics_bundle_sha256=statistics.sha256,
        execution_estimate=estimate,
        blocks=tuple(blocks),
    )


@dataclass(frozen=True, slots=True)
class PreparedScoredFeatureShard:
    """Target-free content identity for one canonical scored-domain feature shard."""

    plan: PreparedNestedLodoPlan
    domain_index: int
    row_count: int
    embedding_identity: EmbeddingIdentity | None
    embedding_dimension: int
    example_ids: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]
    feature_content_sha256: str
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_FEATURE_SHARD_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("feature-shard plan must be exact")
        domain_index = _exact_nonnegative_int(self.domain_index, "feature-shard domain_index")
        if domain_index >= len(self.plan.domains):
            raise ValueError("feature-shard domain_index is outside the prepared plan")
        row_count = _exact_positive_int(self.row_count, "feature-shard row_count")
        if row_count != self.plan.domain_example_counts[domain_index]:
            raise ValueError("feature-shard row_count does not match the prepared plan")
        embedding_dimension = _exact_nonnegative_int(
            self.embedding_dimension,
            "feature-shard embedding_dimension",
        )
        if self.plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + embedding_dimension:
            raise ValueError("feature-shard embedding width does not match the plan")
        if (embedding_dimension == 0) != (self.embedding_identity is None):
            raise ValueError("feature-shard embedding identity and dimension disagree")
        if (
            self.embedding_identity is not None
            and type(self.embedding_identity) is not EmbeddingIdentity
        ):
            raise TypeError("feature-shard embedding identity must be exact")
        if type(self.example_ids) is not tuple or type(self.prompt_sha256s) is not tuple:
            raise TypeError("feature-shard row keys must be exact tuples")
        if len(self.example_ids) != row_count or len(self.prompt_sha256s) != row_count:
            raise ValueError("feature-shard row keys have the wrong exact row count")
        _row_key_text_bytes(
            self.example_ids,
            self.prompt_sha256s,
            require_canonical_order=True,
        )
        _sha256_hex(self.feature_content_sha256, "feature_content_sha256")
        object.__setattr__(self, "sha256", _feature_shard_sha256(self))


def _feature_shard_sha256(shard: PreparedScoredFeatureShard) -> str:
    writer = _HashWriter(PREPARED_FEATURE_SHARD_ALGORITHM_ID)
    _write_plan_identity(writer, shard.plan)
    writer.text("surface.algorithm_id", SURFACE_FEATURE_ALGORITHM_ID)
    writer.text("content.algorithm_id", PREPARED_FEATURE_SHARD_CONTENT_ALGORITHM_ID)
    writer.integer("domain_index", shard.domain_index)
    writer.text("domain", shard.plan.domains[shard.domain_index])
    writer.integer("row_count", shard.row_count)
    writer.integer("embedding_dimension", shard.embedding_dimension)
    if shard.embedding_identity is not None:
        _write_embedding_identity(writer, shard.embedding_identity)
    for example_id, prompt_sha256 in zip(
        shard.example_ids,
        shard.prompt_sha256s,
        strict=True,
    ):
        writer.text("example_id", example_id)
        writer.text("prompt_sha256", prompt_sha256)
    writer.text("feature_content_sha256", shard.feature_content_sha256)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedScoredFeatureShardBundle:
    """All target-free scored feature-shard identities in canonical domain order."""

    plan: PreparedNestedLodoPlan
    embedding_identity: EmbeddingIdentity | None
    embedding_dimension: int
    shards: tuple[PreparedScoredFeatureShard, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_FEATURE_SHARD_BUNDLE_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("feature-shard bundle plan must be exact")
        embedding_dimension = _exact_nonnegative_int(
            self.embedding_dimension,
            "feature-shard embedding_dimension",
        )
        if self.plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + embedding_dimension:
            raise ValueError("feature-shard embedding width does not match the plan")
        if (embedding_dimension == 0) != (self.embedding_identity is None):
            raise ValueError("feature-shard embedding identity and dimension disagree")
        if (
            self.embedding_identity is not None
            and type(self.embedding_identity) is not EmbeddingIdentity
        ):
            raise TypeError("feature-shard embedding identity must be exact")
        if type(self.shards) is not tuple or len(self.shards) != len(self.plan.domains):
            raise ValueError("feature shards do not match the prepared domain count")
        if not all(type(shard) is PreparedScoredFeatureShard for shard in self.shards):
            raise TypeError("feature shards must contain exact shard values")
        for position, shard in enumerate(self.shards):
            if (
                shard.plan != self.plan
                or shard.domain_index != position
                or shard.embedding_dimension != embedding_dimension
                or shard.embedding_identity != self.embedding_identity
            ):
                raise ValueError("feature shard does not match its canonical domain position")
        total_row_key_bytes = sum(
            len(example_id.encode("utf-8")) + len(prompt_sha256)
            for shard in self.shards
            for example_id, prompt_sha256 in zip(
                shard.example_ids,
                shard.prompt_sha256s,
                strict=True,
            )
        )
        if total_row_key_bytes > MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES:
            raise ValueError("feature-shard row keys exceed the reference text-byte limit")
        previous_example_id: str | None = None
        row_key_count = 0
        for example_id in heapq.merge(*(shard.example_ids for shard in self.shards)):
            if example_id == previous_example_id:
                raise ValueError("feature-shard example IDs must be globally unique and complete")
            previous_example_id = example_id
            row_key_count += 1
        if row_key_count != self.plan.work.example_count:
            raise ValueError("feature-shard example IDs must be globally unique and complete")
        object.__setattr__(self, "sha256", _feature_shard_bundle_sha256(self))


def _feature_shard_bundle_sha256(bundle: PreparedScoredFeatureShardBundle) -> str:
    writer = _HashWriter(PREPARED_FEATURE_SHARD_BUNDLE_ALGORITHM_ID)
    _write_plan_identity(writer, bundle.plan)
    writer.text("surface.algorithm_id", SURFACE_FEATURE_ALGORITHM_ID)
    for tag in SURFACE_DOMAIN_TAG_CATALOGUE:
        writer.text("universal_domain_tag", tag)
    writer.integer("embedding_dimension", bundle.embedding_dimension)
    if bundle.embedding_identity is not None:
        _write_embedding_identity(writer, bundle.embedding_identity)
    for shard in bundle.shards:
        writer.text("feature_shard_sha256", shard.sha256)
    return writer.hexdigest()


def _build_scored_feature_shards(
    store: PreparedFeatureStore,
) -> PreparedScoredFeatureShardBundle:
    writers: list[_HashWriter] = []
    for domain_index, domain in enumerate(store.plan.domains):
        writer = _HashWriter(PREPARED_FEATURE_SHARD_CONTENT_ALGORITHM_ID)
        _write_plan_identity(writer, store.plan)
        writer.text("surface.algorithm_id", SURFACE_FEATURE_ALGORITHM_ID)
        for tag in SURFACE_DOMAIN_TAG_CATALOGUE:
            writer.text("universal_domain_tag", tag)
        writer.integer("domain_index", domain_index)
        writer.text("domain", domain)
        writer.integer("embedding_dimension", store.embedding_dimension)
        if store.embedding_identity is not None:
            _write_embedding_identity(writer, store.embedding_identity)
        writers.append(writer)

    example_ids: list[list[str]] = [[] for _ in store.plan.domains]
    prompt_sha256s: list[list[str]] = [[] for _ in store.plan.domains]
    feature_stride = store.plan.feature_count * _F64_BYTES
    for row_index, domain_index in enumerate(store.domain_indices):
        writer = writers[domain_index]
        writer.text("example_id", store.example_ids[row_index])
        writer.text("prompt_sha256", store.prompt_sha256s[row_index])
        writer.token(
            "raw_features.f64le",
            store.feature_payload[row_index * feature_stride : (row_index + 1) * feature_stride],
        )
        example_ids[domain_index].append(store.example_ids[row_index])
        prompt_sha256s[domain_index].append(store.prompt_sha256s[row_index])
    counts = tuple(len(values) for values in example_ids)
    if counts != store.plan.domain_example_counts:
        raise ValueError("feature-shard rows do not match the prepared plan")
    shards = tuple(
        PreparedScoredFeatureShard(
            plan=store.plan,
            domain_index=domain_index,
            row_count=counts[domain_index],
            embedding_identity=store.embedding_identity,
            embedding_dimension=store.embedding_dimension,
            example_ids=tuple(example_ids[domain_index]),
            prompt_sha256s=tuple(prompt_sha256s[domain_index]),
            feature_content_sha256=writers[domain_index].hexdigest(),
        )
        for domain_index in range(len(store.plan.domains))
    )
    return PreparedScoredFeatureShardBundle(
        plan=store.plan,
        embedding_identity=store.embedding_identity,
        embedding_dimension=store.embedding_dimension,
        shards=shards,
    )


def build_prepared_scored_feature_shards(
    store: PreparedFeatureStore,
) -> PreparedScoredFeatureShardBundle:
    """Build target-free per-domain feature identities without reading labels."""

    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    if (
        store.plan.work.example_count * store.plan.feature_count
        > MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS
    ):
        raise ValueError("prepared feature-shard hashing exceeds the aggregate work limit")
    return _build_scored_feature_shards(store)


@dataclass(frozen=True, slots=True)
class PreparedRawScoreBlock:
    """One canonical row-major/sorted-model raw score block."""

    plan: PreparedNestedLodoPlan
    block_index: int
    model_ids: tuple[str, ...]
    coefficient_block_sha256: str
    scored_feature_shard_sha256: str
    scores_payload: bytes = field(repr=False)
    sha256: str = field(init=False)
    scorer_id: str = field(default=PREPARED_RAW_SCORER_ID, init=False)
    algorithm_id: str = field(default=PREPARED_RAW_SCORE_BLOCK_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("raw-score plan must be exact")
        block_index = _exact_nonnegative_int(self.block_index, "raw-score block_index")
        if block_index >= len(self.plan.score_blocks):
            raise ValueError("raw-score block_index is outside the prepared plan")
        _validate_model_ids(self.model_ids, self.plan.target_count, "raw-score")
        _sha256_hex(self.coefficient_block_sha256, "coefficient_block_sha256")
        _sha256_hex(self.scored_feature_shard_sha256, "scored_feature_shard_sha256")
        if type(self.scores_payload) is not bytes:
            raise TypeError("raw-score payload must be immutable bytes")
        expected_bytes = (
            self.plan.score_blocks[block_index].row_count * self.plan.target_count * _F64_BYTES
        )
        if expected_bytes > MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES:
            raise ValueError("raw-score block exceeds the reference numeric-storage limit")
        if len(self.scores_payload) != expected_bytes:
            raise ValueError("raw-score payload has the wrong exact length")
        _validate_f64_payload(self.scores_payload, "raw-score payload")
        object.__setattr__(self, "sha256", _raw_score_block_sha256(self))

    @property
    def row_count(self) -> int:
        return self.plan.score_blocks[self.block_index].row_count

    def score_row(self, row_index: int) -> tuple[float, ...]:
        """Return one private-copy row in sorted model order."""

        index = _exact_nonnegative_int(row_index, "raw-score row_index")
        if index >= self.row_count:
            raise IndexError("raw-score row_index is outside the block")
        return struct.unpack_from(
            f"<{self.plan.target_count}d",
            self.scores_payload,
            index * self.plan.target_count * _F64_BYTES,
        )

    def iter_score_rows(self) -> Iterator[tuple[float, ...]]:
        """Iterate bounded private-copy rows without eager object expansion."""

        for row_index in range(self.row_count):
            yield self.score_row(row_index)


def _raw_score_block_sha256(block: PreparedRawScoreBlock) -> str:
    writer = _HashWriter(PREPARED_RAW_SCORE_BLOCK_ALGORITHM_ID)
    _write_plan_identity(writer, block.plan)
    writer.text("scorer_id", block.scorer_id)
    writer.integer("block_index", block.block_index)
    graph_block = block.plan.score_blocks[block.block_index]
    writer.integer("training_subset_index", graph_block.training_subset_index)
    writer.integer("scored_domain_index", graph_block.scored_domain_index)
    writer.integer("row_count", graph_block.row_count)
    for model_id in block.model_ids:
        writer.text("model_id", model_id)
    writer.text("coefficient_block_sha256", block.coefficient_block_sha256)
    writer.text("scored_feature_shard_sha256", block.scored_feature_shard_sha256)
    writer.token("scores.row-major-model-sorted.f64le", block.scores_payload)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedRawScoreBundle:
    """All graph raw scores, keyed by exact block context rather than prompt alone."""

    coefficients: PreparedCoefficientBundle
    feature_shards: PreparedScoredFeatureShardBundle
    blocks: tuple[PreparedRawScoreBlock, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_RAW_SCORE_BUNDLE_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.coefficients) is not PreparedCoefficientBundle:
            raise TypeError("raw-score coefficients must be an exact coefficient bundle")
        if type(self.feature_shards) is not PreparedScoredFeatureShardBundle:
            raise TypeError("feature_shards must be an exact shard bundle")
        if (
            self.feature_shards.plan != self.plan
            or self.feature_shards.embedding_dimension != self.coefficients.embedding_dimension
            or self.feature_shards.embedding_identity != self.coefficients.embedding_identity
        ):
            raise ValueError("feature_shards plan does not match the raw-score bundle")
        if type(self.blocks) is not tuple or len(self.blocks) != len(self.plan.score_blocks):
            raise ValueError("raw-score blocks have the wrong bounded length")
        if not all(type(block) is PreparedRawScoreBlock for block in self.blocks):
            raise TypeError("raw-score blocks must contain exact block values")
        payload_bytes = sum(len(block.scores_payload) for block in self.blocks)
        if payload_bytes != self.execution_estimate.score_bytes:
            raise ValueError("raw-score payload bytes do not match the execution estimate")
        for position, block in enumerate(self.blocks):
            graph_block = self.plan.score_blocks[position]
            if (
                block.plan != self.plan
                or block.block_index != position
                or block.model_ids != self.model_ids
                or block.coefficient_block_sha256
                != self.coefficients.blocks[graph_block.training_subset_index].sha256
                or block.scored_feature_shard_sha256
                != self.feature_shards.shards[graph_block.scored_domain_index].sha256
            ):
                raise ValueError("raw-score block does not match its canonical bundle position")
        object.__setattr__(self, "sha256", _raw_score_bundle_sha256(self))

    @property
    def plan(self) -> PreparedNestedLodoPlan:
        return self.coefficients.plan

    @property
    def model_ids(self) -> tuple[str, ...]:
        return self.coefficients.model_ids

    @property
    def ridge(self) -> float:
        return self.coefficients.ridge

    @property
    def execution_estimate(self) -> PreparedReferenceExecutionEstimate:
        return self.coefficients.execution_estimate

    @property
    def coefficient_block_sha256s(self) -> tuple[str, ...]:
        return tuple(block.sha256 for block in self.coefficients.blocks)

    def example_ids_for_block(self, block_index: int) -> tuple[str, ...]:
        """Return canonical row join keys for one exact graph block."""

        index = _exact_nonnegative_int(block_index, "raw-score block_index")
        if index >= len(self.plan.score_blocks):
            raise IndexError("raw-score block_index is outside the prepared plan")
        scored_domain = self.plan.score_blocks[index].scored_domain_index
        return self.feature_shards.shards[scored_domain].example_ids


def _raw_score_bundle_sha256(bundle: PreparedRawScoreBundle) -> str:
    writer = _HashWriter(PREPARED_RAW_SCORE_BUNDLE_ALGORITHM_ID)
    _write_plan_identity(writer, bundle.plan)
    writer.text("scorer_id", PREPARED_RAW_SCORER_ID)
    writer.text("coefficient_bundle_sha256", bundle.coefficients.sha256)
    writer.text("feature_shards_sha256", bundle.feature_shards.sha256)
    for block in bundle.blocks:
        writer.text("raw_score_block_sha256", block.sha256)
    return writer.hexdigest()


def _encode_store_row(
    store: PreparedFeatureStore,
    row_index: int,
    coefficient: PreparedCoefficientBlock,
    active_feature_indices: tuple[int, ...],
) -> tuple[float, ...]:
    raw = store.feature_row(row_index)
    encoded: list[float] = []
    for position, raw_index in enumerate(active_feature_indices):
        value = raw[raw_index]
        if position < _CONTINUOUS_COUNT:
            value = (value - coefficient.feature_schema.continuous_means[position]) / (
                coefficient.feature_schema.continuous_scales[position]
            )
        if not math.isfinite(value):
            raise ArithmeticError("prepared row encoding produced a non-finite value")
        encoded.append(0.0 if value == 0.0 else value)
    return tuple(encoded)


def _score_encoded_row(
    encoded: tuple[float, ...],
    coefficient: PreparedCoefficientBlock,
) -> tuple[float, ...]:
    scores: list[float] = []
    for model_index in range(coefficient.plan.target_count):
        weights = coefficient.weights_for_model_index(model_index)
        # Match BilinearQualityPredictor's ordinary Python sum and schema order.
        score = sum(value * weight for value, weight in zip(encoded, weights, strict=True))
        score += coefficient.intercept_for_model_index(model_index)
        if not math.isfinite(score):
            raise ArithmeticError("prepared raw scoring produced a non-finite value")
        scores.append(0.0 if score == 0.0 else score)
    return tuple(scores)


def build_prepared_raw_score_bundle(
    store: PreparedFeatureStore,
    coefficients: PreparedCoefficientBundle,
) -> PreparedRawScoreBundle:
    """Preflight aggregate scoring, then emit all canonical plan score blocks."""

    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    if type(coefficients) is not PreparedCoefficientBundle:
        raise TypeError("coefficients must be an exact PreparedCoefficientBundle")
    if (
        coefficients.plan != store.plan
        or coefficients.model_ids != store.model_ids
        or coefficients.source_store_sha256 != store.sha256
        or coefficients.embedding_dimension != store.embedding_dimension
        or coefficients.embedding_identity != store.embedding_identity
    ):
        raise ValueError("coefficient bundle does not match the prepared store layout")
    if store.embedding_dimension != (store.plan.feature_count - _UNIVERSAL_SURFACE_DIMENSION):
        raise ValueError("prepared store embedding layout is inconsistent")

    # Repeat the cumulative cap from exact retained coefficient widths before the
    # first feature hash, feature-row read, score allocation, or dot product.
    active_widths = tuple(block.feature_count for block in coefficients.blocks)
    estimate = _execution_estimate(store.plan, active_widths)
    if estimate != coefficients.execution_estimate:
        raise ValueError("coefficient execution estimate does not match its blocks")
    active_indices_by_subset = tuple(block.active_feature_indices for block in coefficients.blocks)

    feature_shards = _build_scored_feature_shards(store)
    score_blocks: list[PreparedRawScoreBlock] = []
    target_count = store.plan.target_count
    for block_index, graph_block in enumerate(store.plan.score_blocks):
        coefficient = coefficients.blocks[graph_block.training_subset_index]
        payload = bytearray(graph_block.row_count * target_count * _F64_BYTES)
        scored_row = 0
        for row_index, domain_index in enumerate(store.domain_indices):
            if domain_index != graph_block.scored_domain_index:
                continue
            encoded = _encode_store_row(
                store,
                row_index,
                coefficient,
                active_indices_by_subset[graph_block.training_subset_index],
            )
            scores = _score_encoded_row(encoded, coefficient)
            struct.pack_into(
                f"<{target_count}d",
                payload,
                scored_row * target_count * _F64_BYTES,
                *scores,
            )
            scored_row += 1
        if scored_row != graph_block.row_count:
            raise ValueError("scored rows do not match the prepared graph block")
        score_blocks.append(
            PreparedRawScoreBlock(
                plan=store.plan,
                block_index=block_index,
                model_ids=store.model_ids,
                coefficient_block_sha256=coefficient.sha256,
                scored_feature_shard_sha256=feature_shards.shards[
                    graph_block.scored_domain_index
                ].sha256,
                scores_payload=bytes(payload),
            )
        )
    return PreparedRawScoreBundle(
        coefficients=coefficients,
        feature_shards=feature_shards,
        blocks=tuple(score_blocks),
    )


__all__ = [
    "MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES",
    "MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS",
    "PREPARED_COEFFICIENT_BLOCK_ALGORITHM_ID",
    "PREPARED_COEFFICIENT_BUNDLE_ALGORITHM_ID",
    "PREPARED_FEATURE_SHARD_ALGORITHM_ID",
    "PREPARED_FEATURE_SHARD_BUNDLE_ALGORITHM_ID",
    "PREPARED_FEATURE_SHARD_CONTENT_ALGORITHM_ID",
    "PREPARED_MOMENT_RIDGE_SOLVER_ID",
    "PREPARED_RAW_SCORER_ID",
    "PREPARED_RAW_SCORE_BLOCK_ALGORITHM_ID",
    "PREPARED_RAW_SCORE_BUNDLE_ALGORITHM_ID",
    "PreparedCoefficientBlock",
    "PreparedCoefficientBundle",
    "PreparedRawScoreBlock",
    "PreparedRawScoreBundle",
    "PreparedReferenceExecutionEstimate",
    "PreparedScoredFeatureShard",
    "PreparedScoredFeatureShardBundle",
    "build_prepared_coefficient_bundle",
    "build_prepared_raw_score_bundle",
    "build_prepared_scored_feature_shards",
]
