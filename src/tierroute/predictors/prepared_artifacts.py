# SPDX-License-Identifier: Apache-2.0
"""Bounded, lineage-aware artifacts for prepared bilinear predictors.

The records in this module identify deterministic binary64 payloads.  Their SHA-256
values are content identities and caller-supplied substitution checks, not signatures
or provenance.  Persistence is JSON-only and performs no network or code execution.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
import struct
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType

import tierroute.predictors.artifacts as _shared_artifacts
from tierroute.features import (
    EmbeddingIdentity,
    EmbeddingProvider,
    PromptFeatureEncoder,
    PromptFeatureSchema,
)
from tierroute.features.surface import (
    SURFACE_DOMAIN_TAG_CATALOGUE,
    SURFACE_FEATURE_ALGORITHM_ID,
)
from tierroute.predictors.base import BilinearQualityPredictor
from tierroute.predictors.calibration import (
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
)
from tierroute.predictors.prepared_execution import (
    PREPARED_MOMENT_RIDGE_SOLVER_ID,
    PREPARED_RAW_SCORER_ID,
)
from tierroute.predictors.prepared_graph import (
    MAX_PREPARED_DOMAINS,
    MAX_PREPARED_EXAMPLES,
    MAX_PREPARED_FEATURES,
    MAX_PREPARED_SCORE_BLOCKS,
    MAX_PREPARED_TARGETS,
    MAX_PREPARED_TRAINING_SUBSETS,
    MIN_PREPARED_DOMAINS,
    PREPARED_GRAPH_ALGORITHM_ID,
    PreparedNestedLodoPlan,
    build_prepared_nested_lodo_plan,
)
from tierroute.predictors.prepared_store import (
    MAX_PREPARED_MODEL_ID_UTF8_BYTES,
    MAX_PREPARED_REFERENCE_NUMERIC_BYTES,
    MAX_PREPARED_REFERENCE_STATISTIC_SCALARS,
    MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES,
    _bounded_text,
    _HashWriter,
    _packed_upper_index,
    _packed_upper_length,
    _row_key_text_bytes,
    _sha256_hex,
    _write_embedding_identity,
    _write_plan_identity,
)
from tierroute.predictors.resource_limits import (
    MAX_PREDICTOR_ARTIFACT_BYTES,
    MAX_PREDICTOR_CALIBRATOR_POINTS,
    MAX_PREDICTOR_JSON_NUMBER_CHARACTERS,
    MAX_PREDICTOR_METADATA_TEXT_BYTES,
    MAX_PREDICTOR_METADATA_TOTAL_BYTES,
)

PREPARED_PREDICTOR_ARTIFACT_KIND = "tierroute-prepared-bilinear-predictor"
PREPARED_PREDICTOR_ARTIFACT_VERSION = 1
PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID = "tierroute.prepared-bilinear-artifact-v1"
PREPARED_ALL_DOMAIN_ASSEMBLY_ALGORITHM_ID = "tierroute.prepared-all-domain-assembly-v1"
PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID = "tierroute.prepared-all-domain-statistics-chan-v1"
PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID = "tierroute.prepared-all-domain-coefficient-v1"
PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID = "tierroute.prepared-predictor-target-shard-v1"
PREPARED_PREDICTOR_CALIBRATION_INPUT_ALGORITHM_ID = (
    "tierroute.prepared-predictor-calibration-input-v1"
)
PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID = "tierroute.prepared-predictor-isotonic-v1"

# The reviewed D7/N34,778/d1,036/M11 shape contains 776,530 serialized numeric
# parameters.  This tighter cap admits it while leaving room under the shared
# 32-MiB document boundary for keys, metadata, and parser amplification.
MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS = 800_000
MAX_PREPARED_ARTIFACT_MODELS = MAX_PREPARED_TARGETS
MAX_PREPARED_ARTIFACT_DOMAINS = MAX_PREPARED_DOMAINS
MAX_PREPARED_ARTIFACT_JSON_NUMBER_TOKENS = (
    MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS + 4 * MAX_PREPARED_DOMAINS + 8
)

_CONTINUOUS_COUNT = 3
_BINARY_COUNT = 2
_TAG_OFFSET = _CONTINUOUS_COUNT + _BINARY_COUNT
_UNIVERSAL_SURFACE_DIMENSION = _TAG_OFFSET + len(SURFACE_DOMAIN_TAG_CATALOGUE)
_F64_BYTES = 8


def _validate_artifact_document(document: str) -> None:
    """Apply this format's patchable byte cap before parsing or publication."""

    if type(document) is not str:
        raise ValueError("prepared predictor artifact must be exact text")
    try:
        encoded = document.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("prepared predictor artifact is not valid UTF-8 text") from error
    if len(document) > MAX_PREDICTOR_ARTIFACT_BYTES or len(encoded) > (
        MAX_PREDICTOR_ARTIFACT_BYTES
    ):
        raise ValueError(
            f"prepared predictor artifact exceeds {MAX_PREDICTOR_ARTIFACT_BYTES:,} UTF-8 bytes"
        )


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
        raise ValueError(f"{name} must fit finite binary64") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite positive binary64" if positive else "finite binary64"
        raise ValueError(f"{name} must be {qualifier}")
    return 0.0 if result == 0.0 else result


def _exact_f64_tuple(
    value: object,
    name: str,
    *,
    expected_length: int,
) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) != expected_length:
        raise ValueError(f"{name} has the wrong exact length")
    return tuple(_canonical_f64(item, name) for item in value)


def _json_f64_tuple(
    value: object,
    name: str,
    *,
    max_items: int,
) -> tuple[float, ...]:
    raw = _shared_artifacts._finite_tuple(value, name, max_items=max_items)
    return tuple(_canonical_f64(item, f"{name} item") for item in raw)


def _validate_f64_payload(payload: bytes, name: str) -> None:
    for (value,) in struct.iter_unpack("<d", payload):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite binary64 values")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise ValueError(f"{name} must use canonical positive zero")


def _canonical_f64_bytes(values: tuple[float, ...], name: str) -> bytes:
    normalized = tuple(_canonical_f64(value, name) for value in values)
    return struct.pack(f"<{len(normalized)}d", *normalized)


def _snapshot_schema(value: object) -> PromptFeatureSchema:
    if type(value) is not PromptFeatureSchema:
        raise TypeError("feature_schema must be an exact PromptFeatureSchema")
    schema = PromptFeatureSchema(
        continuous_means=tuple(
            _canonical_f64(item, "feature continuous mean") for item in value.continuous_means
        ),  # type: ignore[arg-type]
        continuous_scales=tuple(
            _canonical_f64(item, "feature continuous scale", positive=True)
            for item in value.continuous_scales
        ),  # type: ignore[arg-type]
        domain_tags=value.domain_tags,
        embedding_dimension=value.embedding_dimension,
        embedding_identity=value.embedding_identity,
        schema_version=value.schema_version,
    )
    catalogue = {tag: index for index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE)}
    try:
        positions = tuple(catalogue[tag] for tag in schema.domain_tags)
    except KeyError as error:
        raise ValueError("feature schema contains a tag outside the fixed catalogue") from error
    if positions != tuple(sorted(positions)):
        raise ValueError("feature schema tags must follow universal catalogue order")
    if schema.dimension > MAX_PREPARED_FEATURES:
        raise ValueError("feature schema exceeds the prepared feature limit")
    if _UNIVERSAL_SURFACE_DIMENSION + schema.embedding_dimension > MAX_PREPARED_FEATURES:
        raise ValueError("feature schema exceeds the prepared universal feature limit")
    return schema


def _snapshot_embedding_identity(
    identity: EmbeddingIdentity | None,
    dimension: int,
    context: str,
) -> EmbeddingIdentity | None:
    dimension = _exact_nonnegative_int(dimension, f"{context} embedding_dimension")
    if (dimension == 0) != (identity is None):
        raise ValueError(f"{context} embedding identity and dimension disagree")
    if identity is None:
        return None
    if type(identity) is not EmbeddingIdentity:
        raise TypeError(f"{context} embedding identity must be exact")
    return EmbeddingIdentity(
        provider=identity.provider,
        model_id=identity.model_id,
        revision=identity.revision,
        pooling=identity.pooling,
        normalize=identity.normalize,
        asset_manifest_sha256=identity.asset_manifest_sha256,
    )


def _active_feature_indices(schema: PromptFeatureSchema) -> tuple[int, ...]:
    catalogue = {tag: index for index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE)}
    tag_indices = tuple(_TAG_OFFSET + catalogue[tag] for tag in schema.domain_tags)
    embedding_indices = tuple(
        range(
            _UNIVERSAL_SURFACE_DIMENSION,
            _UNIVERSAL_SURFACE_DIMENSION + schema.embedding_dimension,
        )
    )
    active = (*range(_TAG_OFFSET), *tag_indices, *embedding_indices)
    if len(active) != schema.dimension:
        raise ValueError("feature schema does not map to prepared universal coordinates")
    return active


def _snapshot_model_ids(value: object, expected_count: int | None = None) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("model_ids must be an exact tuple")
    if expected_count is not None and len(value) != expected_count:
        raise ValueError("model catalogue has the wrong exact count")
    if not 0 < len(value) <= MAX_PREPARED_ARTIFACT_MODELS:
        raise ValueError("model catalogue is outside the prepared artifact limit")
    total_bytes = 0
    for model_id in value:
        _bounded_text(
            model_id,
            "prepared artifact model_id",
            max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
        )
        total_bytes += len(model_id.encode("utf-8"))
    if total_bytes > MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES:
        raise ValueError("model catalogue exceeds the prepared text-byte limit")
    if value != tuple(sorted(set(value))):
        raise ValueError("model IDs must be sorted and unique")
    return value


def _write_feature_schema(writer: _HashWriter, schema: PromptFeatureSchema) -> None:
    writer.integer("feature_schema.version", schema.schema_version)
    writer.floats("feature_schema.continuous_means", schema.continuous_means)
    writer.floats("feature_schema.continuous_scales", schema.continuous_scales)
    for tag in schema.domain_tags:
        writer.text("feature_schema.domain_tag", tag)
    writer.integer("feature_schema.embedding_dimension", schema.embedding_dimension)
    if schema.embedding_identity is not None:
        _write_embedding_identity(writer, schema.embedding_identity)


def _validate_universal_means(
    means: tuple[float, ...],
    active_tag_mask: int,
    context: str,
) -> None:
    if any(value < 0.0 for value in means[:_CONTINUOUS_COUNT]):
        raise ValueError(f"{context} continuous means must be non-negative")
    if any(
        not 0.0 <= value <= 1.0 for value in means[_CONTINUOUS_COUNT:_UNIVERSAL_SURFACE_DIMENSION]
    ):
        raise ValueError(f"{context} binary/tag means must be between zero and one")
    expected_mask = sum(
        1 << index
        for index in range(len(SURFACE_DOMAIN_TAG_CATALOGUE))
        if means[_TAG_OFFSET + index] > 0.0
    )
    if active_tag_mask != expected_mask:
        raise ValueError(f"{context} active_tag_mask does not match tag means")


@dataclass(frozen=True, slots=True)
class PreparedAllDomainStatistics:
    """Canonical ascending-domain Chan aggregate used by the final solve."""

    plan: PreparedNestedLodoPlan
    store_sha256: str
    statistics_bundle_sha256: str
    model_ids: tuple[str, ...]
    embedding_identity: EmbeddingIdentity | None
    embedding_dimension: int
    domain_statistics_sha256s: tuple[str, ...]
    row_count: int
    active_tag_mask: int
    feature_means: tuple[float, ...]
    target_means: tuple[float, ...]
    centered_xx_packed: tuple[float, ...]
    centered_xy: tuple[float, ...]
    feature_schema: PromptFeatureSchema = field(init=False)
    sha256: str = field(init=False)
    algorithm_id: str = field(
        default=PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID,
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self) is not PreparedAllDomainStatistics:
            raise TypeError("all-domain statistics must be an exact project type")
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("all-domain statistics plan must be exact")
        canonical_plan = build_prepared_nested_lodo_plan(
            self.plan.domains,
            self.plan.domain_example_counts,
            feature_count=self.plan.feature_count,
            target_count=self.plan.target_count,
        )
        if canonical_plan != self.plan:
            raise ValueError("all-domain statistics plan must be canonical")
        object.__setattr__(self, "plan", canonical_plan)
        _sha256_hex(self.store_sha256, "aggregate store_sha256")
        _sha256_hex(
            self.statistics_bundle_sha256,
            "aggregate statistics_bundle_sha256",
        )
        model_ids = _snapshot_model_ids(self.model_ids, self.plan.target_count)
        embedding_dimension = _exact_nonnegative_int(
            self.embedding_dimension,
            "aggregate embedding_dimension",
        )
        if self.plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + embedding_dimension:
            raise ValueError("aggregate embedding width does not match the plan")
        embedding_identity = _snapshot_embedding_identity(
            self.embedding_identity,
            embedding_dimension,
            "aggregate",
        )
        if type(self.domain_statistics_sha256s) is not tuple or len(
            self.domain_statistics_sha256s
        ) != len(self.plan.domains):
            raise ValueError("aggregate domain-statistics catalogue has the wrong length")
        domain_hashes = tuple(
            _sha256_hex(value, "domain_statistics_sha256")
            for value in self.domain_statistics_sha256s
        )
        row_count = _exact_positive_int(self.row_count, "aggregate row_count")
        if row_count != self.plan.work.example_count:
            raise ValueError("aggregate row_count does not match the prepared plan")
        mask = _exact_nonnegative_int(self.active_tag_mask, "aggregate active_tag_mask")
        if mask >= 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE):
            raise ValueError("aggregate active_tag_mask contains an unknown bit")
        dimension = self.plan.feature_count
        target_count = self.plan.target_count
        scalar_count = (
            dimension + target_count + _packed_upper_length(dimension) + dimension * target_count
        )
        if scalar_count > MAX_PREPARED_REFERENCE_STATISTIC_SCALARS:
            raise ValueError("all-domain aggregate exceeds the prepared statistic limit")
        feature_means = _exact_f64_tuple(
            self.feature_means,
            "aggregate feature_means",
            expected_length=dimension,
        )
        _validate_universal_means(feature_means, mask, "aggregate")
        target_means = _exact_f64_tuple(
            self.target_means,
            "aggregate target_means",
            expected_length=target_count,
        )
        centered_xx = _exact_f64_tuple(
            self.centered_xx_packed,
            "aggregate centered_xx_packed",
            expected_length=_packed_upper_length(dimension),
        )
        centered_xy = _exact_f64_tuple(
            self.centered_xy,
            "aggregate centered_xy",
            expected_length=dimension * target_count,
        )
        scales: list[float] = []
        for index in range(dimension):
            diagonal = centered_xx[_packed_upper_index(dimension, index, index)]
            if diagonal < 0.0:
                raise ValueError("aggregate centered feature diagonal must be non-negative")
            if index < _CONTINUOUS_COUNT:
                scale = math.sqrt(diagonal / row_count)
                scales.append(scale if scale > 0.0 else 1.0)
        active_tags = tuple(
            tag for index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE) if mask & (1 << index)
        )
        schema = PromptFeatureSchema(
            continuous_means=tuple(feature_means[:_CONTINUOUS_COUNT]),  # type: ignore[arg-type]
            continuous_scales=tuple(scales),  # type: ignore[arg-type]
            domain_tags=active_tags,
            embedding_dimension=embedding_dimension,
            embedding_identity=embedding_identity,
        )
        object.__setattr__(self, "model_ids", model_ids)
        object.__setattr__(self, "embedding_identity", embedding_identity)
        object.__setattr__(self, "embedding_dimension", embedding_dimension)
        object.__setattr__(self, "domain_statistics_sha256s", domain_hashes)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "active_tag_mask", mask)
        object.__setattr__(self, "feature_means", feature_means)
        object.__setattr__(self, "target_means", target_means)
        object.__setattr__(self, "centered_xx_packed", centered_xx)
        object.__setattr__(self, "centered_xy", centered_xy)
        object.__setattr__(self, "feature_schema", schema)
        object.__setattr__(self, "sha256", _all_domain_statistics_sha256(self))


def _all_domain_statistics_sha256(statistics: PreparedAllDomainStatistics) -> str:
    writer = _HashWriter(PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID)
    _write_plan_identity(writer, statistics.plan)
    writer.text("store_sha256", statistics.store_sha256)
    writer.text("statistics_bundle_sha256", statistics.statistics_bundle_sha256)
    writer.integer("model_count", len(statistics.model_ids))
    for model_id in statistics.model_ids:
        writer.text("model_id", model_id)
    writer.integer("embedding_dimension", statistics.embedding_dimension)
    if statistics.embedding_identity is not None:
        _write_embedding_identity(writer, statistics.embedding_identity)
    writer.integer("domain_count", len(statistics.domain_statistics_sha256s))
    for domain_index, child_sha256 in enumerate(statistics.domain_statistics_sha256s):
        writer.integer("domain_statistics.domain_index", domain_index)
        writer.text("domain_statistics.sha256", child_sha256)
    writer.integer("row_count", statistics.row_count)
    writer.integer("active_tag_mask", statistics.active_tag_mask)
    writer.floats("feature_means", statistics.feature_means)
    writer.floats("target_means", statistics.target_means)
    writer.floats("centered_xx_packed", statistics.centered_xx_packed)
    writer.floats("centered_xy", statistics.centered_xy)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedFinalCoefficient:
    """Final target-major coefficient payload and its recomputable identity."""

    feature_schema: PromptFeatureSchema
    active_feature_indices: tuple[int, ...]
    model_ids: tuple[str, ...]
    aggregate_statistics_sha256: str
    ridge: float
    weights_payload: bytes = field(repr=False)
    intercepts_payload: bytes = field(repr=False)
    sha256: str = field(init=False)
    solver_id: str = field(default=PREPARED_MOMENT_RIDGE_SOLVER_ID, init=False)
    raw_scorer_id: str = field(default=PREPARED_RAW_SCORER_ID, init=False)
    algorithm_id: str = field(
        default=PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID,
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self) is not PreparedFinalCoefficient:
            raise TypeError("final coefficient must be an exact project type")
        schema = _snapshot_schema(self.feature_schema)
        expected_active = _active_feature_indices(schema)
        if type(self.active_feature_indices) is not tuple:
            raise TypeError("active_feature_indices must be an exact tuple")
        if any(type(index) is not int for index in self.active_feature_indices):
            raise TypeError("active_feature_indices must contain exact integers")
        if self.active_feature_indices != expected_active:
            raise ValueError("active_feature_indices do not match the feature schema")
        model_ids = _snapshot_model_ids(self.model_ids)
        aggregate_sha256 = _sha256_hex(
            self.aggregate_statistics_sha256,
            "aggregate_statistics_sha256",
        )
        ridge = _canonical_f64(self.ridge, "final coefficient ridge", positive=True)
        if type(self.weights_payload) is not bytes or type(self.intercepts_payload) is not bytes:
            raise TypeError("final coefficient payloads must be immutable bytes")
        expected_weight_bytes = len(model_ids) * schema.dimension * _F64_BYTES
        expected_intercept_bytes = len(model_ids) * _F64_BYTES
        if expected_weight_bytes + expected_intercept_bytes > (
            MAX_PREPARED_REFERENCE_NUMERIC_BYTES
        ):
            raise ValueError("final coefficient exceeds the prepared numeric-byte limit")
        if len(self.weights_payload) != expected_weight_bytes:
            raise ValueError("final coefficient weights have the wrong exact byte length")
        if len(self.intercepts_payload) != expected_intercept_bytes:
            raise ValueError("final coefficient intercepts have the wrong exact byte length")
        _validate_f64_payload(self.weights_payload, "final coefficient weights")
        _validate_f64_payload(self.intercepts_payload, "final coefficient intercepts")
        object.__setattr__(self, "feature_schema", schema)
        object.__setattr__(self, "active_feature_indices", expected_active)
        object.__setattr__(self, "model_ids", model_ids)
        object.__setattr__(self, "aggregate_statistics_sha256", aggregate_sha256)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "sha256", _final_coefficient_sha256(self))

    def weights_for_model_index(self, model_index: int) -> tuple[float, ...]:
        """Return one immutable model row in serialized feature order."""

        index = _exact_nonnegative_int(model_index, "model_index")
        if index >= len(self.model_ids):
            raise IndexError("model_index is outside the final coefficient catalogue")
        return struct.unpack_from(
            f"<{self.feature_schema.dimension}d",
            self.weights_payload,
            index * self.feature_schema.dimension * _F64_BYTES,
        )

    def intercept_for_model_index(self, model_index: int) -> float:
        """Return one model intercept."""

        index = _exact_nonnegative_int(model_index, "model_index")
        if index >= len(self.model_ids):
            raise IndexError("model_index is outside the final coefficient catalogue")
        return struct.unpack_from("<d", self.intercepts_payload, index * _F64_BYTES)[0]


def _final_coefficient_sha256(coefficient: PreparedFinalCoefficient) -> str:
    writer = _HashWriter(PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID)
    writer.text(
        "aggregate_statistics_sha256",
        coefficient.aggregate_statistics_sha256,
    )
    writer.text("solver_id", coefficient.solver_id)
    writer.text("raw_scorer_id", coefficient.raw_scorer_id)
    writer.token("ridge.f64le", struct.pack("<d", coefficient.ridge))
    writer.text("surface.algorithm_id", SURFACE_FEATURE_ALGORITHM_ID)
    _write_feature_schema(writer, coefficient.feature_schema)
    for feature_index in coefficient.active_feature_indices:
        writer.integer("active_feature_index", feature_index)
    writer.integer("model_count", len(coefficient.model_ids))
    for model_id in coefficient.model_ids:
        writer.text("model_id", model_id)
    writer.token(
        "weights.target-major.f64le",
        coefficient.weights_payload,
    )
    writer.token("intercepts.f64le", coefficient.intercepts_payload)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedPredictorTargetShard:
    """Store-derived targets bound to one exact scored feature shard."""

    plan: PreparedNestedLodoPlan
    store_sha256: str
    domain_index: int
    model_ids: tuple[str, ...]
    scored_feature_shard_sha256: str
    example_ids: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]
    targets_payload: InitVar[bytes]
    sha256: str = field(init=False)
    algorithm_id: str = field(
        default=PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID,
        init=False,
    )

    def __post_init__(self, targets_payload: bytes) -> None:
        if type(self) is not PreparedPredictorTargetShard:
            raise TypeError("predictor target shard must be an exact project type")
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("predictor target-shard plan must be exact")
        canonical_plan = build_prepared_nested_lodo_plan(
            self.plan.domains,
            self.plan.domain_example_counts,
            feature_count=self.plan.feature_count,
            target_count=self.plan.target_count,
        )
        if canonical_plan != self.plan:
            raise ValueError("predictor target-shard plan must be canonical")
        object.__setattr__(self, "plan", canonical_plan)
        domain_index = _exact_nonnegative_int(
            self.domain_index,
            "target-shard domain_index",
        )
        if domain_index >= len(self.plan.domains):
            raise ValueError("target-shard domain_index is outside the plan")
        store_sha256 = _sha256_hex(self.store_sha256, "target-shard store_sha256")
        model_ids = _snapshot_model_ids(self.model_ids, self.plan.target_count)
        scored_sha256 = _sha256_hex(
            self.scored_feature_shard_sha256,
            "target-shard scored_feature_shard_sha256",
        )
        row_count = self.plan.domain_example_counts[domain_index]
        if type(self.example_ids) is not tuple or type(self.prompt_sha256s) is not tuple:
            raise TypeError("target-shard row keys must be exact tuples")
        if len(self.example_ids) != row_count or len(self.prompt_sha256s) != row_count:
            raise ValueError("target-shard row keys have the wrong exact row count")
        _row_key_text_bytes(
            self.example_ids,
            self.prompt_sha256s,
            require_canonical_order=True,
        )
        if type(targets_payload) is not bytes:
            raise TypeError("target-shard targets_payload must be immutable bytes")
        expected_bytes = row_count * len(model_ids) * _F64_BYTES
        if expected_bytes > MAX_PREPARED_REFERENCE_NUMERIC_BYTES:
            raise ValueError("target shard exceeds the prepared numeric-byte limit")
        if len(targets_payload) != expected_bytes:
            raise ValueError("target-shard payload has the wrong exact byte length")
        _validate_f64_payload(targets_payload, "target-shard payload")
        object.__setattr__(self, "domain_index", domain_index)
        object.__setattr__(self, "store_sha256", store_sha256)
        object.__setattr__(self, "model_ids", model_ids)
        object.__setattr__(self, "scored_feature_shard_sha256", scored_sha256)
        object.__setattr__(self, "sha256", _target_shard_sha256(self, targets_payload))

    @property
    def row_count(self) -> int:
        return self.plan.domain_example_counts[self.domain_index]


def _target_shard_sha256(
    shard: PreparedPredictorTargetShard,
    targets_payload: bytes,
) -> str:
    writer = _HashWriter(PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID)
    _write_plan_identity(writer, shard.plan)
    writer.text("store_sha256", shard.store_sha256)
    writer.integer("domain_index", shard.domain_index)
    writer.text("domain", shard.plan.domains[shard.domain_index])
    writer.integer("row_count", shard.row_count)
    writer.integer("model_count", len(shard.model_ids))
    for model_id in shard.model_ids:
        writer.text("model_id", model_id)
    writer.text(
        "scored_feature_shard_sha256",
        shard.scored_feature_shard_sha256,
    )
    for example_id, prompt_sha256 in zip(
        shard.example_ids,
        shard.prompt_sha256s,
        strict=True,
    ):
        writer.text("example_id", example_id)
        writer.text("prompt_sha256", prompt_sha256)
    writer.token(
        "targets.row-major-model-sorted.f64le",
        targets_payload,
    )
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedCalibrationSource:
    """One semantic all-but-held-out score/target join retained in lineage."""

    held_out_domain_index: int
    held_out_domain: str
    training_subset_index: int
    score_block_index: int
    row_count: int
    raw_score_block_sha256: str
    scored_feature_shard_sha256: str
    target_shard_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not PreparedCalibrationSource:
            raise TypeError("calibration source must be an exact project type")
        domain_index = _exact_nonnegative_int(
            self.held_out_domain_index,
            "calibration held_out_domain_index",
        )
        if domain_index >= MAX_PREPARED_ARTIFACT_DOMAINS:
            raise ValueError("calibration held_out_domain_index exceeds the prepared limit")
        _bounded_text(
            self.held_out_domain,
            "calibration held_out_domain",
            max_bytes=MAX_PREDICTOR_METADATA_TEXT_BYTES,
        )
        subset_index = _exact_nonnegative_int(
            self.training_subset_index,
            "calibration training_subset_index",
        )
        if subset_index >= MAX_PREPARED_TRAINING_SUBSETS:
            raise ValueError("calibration training_subset_index exceeds the graph limit")
        block_index = _exact_nonnegative_int(
            self.score_block_index,
            "calibration score_block_index",
        )
        if block_index >= MAX_PREPARED_SCORE_BLOCKS:
            raise ValueError("calibration score_block_index exceeds the graph limit")
        row_count = _exact_positive_int(self.row_count, "calibration row_count")
        if row_count > MAX_PREPARED_EXAMPLES:
            raise ValueError("calibration row_count exceeds the prepared limit")
        object.__setattr__(self, "held_out_domain_index", domain_index)
        object.__setattr__(self, "training_subset_index", subset_index)
        object.__setattr__(self, "score_block_index", block_index)
        object.__setattr__(self, "row_count", row_count)
        for name in (
            "raw_score_block_sha256",
            "scored_feature_shard_sha256",
            "target_shard_sha256",
        ):
            object.__setattr__(self, name, _sha256_hex(getattr(self, name), name))

    def to_dict(self) -> dict[str, object]:
        """Return the exact serialized lineage object."""

        return {
            "held_out_domain_index": self.held_out_domain_index,
            "held_out_domain": self.held_out_domain,
            "training_subset_index": self.training_subset_index,
            "score_block_index": self.score_block_index,
            "row_count": self.row_count,
            "raw_score_block_sha256": self.raw_score_block_sha256,
            "scored_feature_shard_sha256": self.scored_feature_shard_sha256,
            "target_shard_sha256": self.target_shard_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PreparedCalibrationSource:
        """Decode one strict calibration-source object."""

        if cls is not PreparedCalibrationSource:
            raise TypeError("calibration-source parsing requires the exact project type")
        item = _shared_artifacts._mapping(
            payload,
            "calibration_source",
            max_items=8,
        )
        _shared_artifacts._strict_fields(
            item,
            {
                "held_out_domain_index",
                "held_out_domain",
                "training_subset_index",
                "score_block_index",
                "row_count",
                "raw_score_block_sha256",
                "scored_feature_shard_sha256",
                "target_shard_sha256",
            },
            "calibration_source",
        )
        return cls(
            held_out_domain_index=item["held_out_domain_index"],  # type: ignore[arg-type]
            held_out_domain=item["held_out_domain"],  # type: ignore[arg-type]
            training_subset_index=item["training_subset_index"],  # type: ignore[arg-type]
            score_block_index=item["score_block_index"],  # type: ignore[arg-type]
            row_count=item["row_count"],  # type: ignore[arg-type]
            raw_score_block_sha256=item["raw_score_block_sha256"],  # type: ignore[arg-type]
            scored_feature_shard_sha256=item[  # type: ignore[arg-type]
                "scored_feature_shard_sha256"
            ],
            target_shard_sha256=item["target_shard_sha256"],  # type: ignore[arg-type]
        )


def _snapshot_calibration_sources(value: object) -> tuple[PreparedCalibrationSource, ...]:
    if type(value) is not tuple:
        raise TypeError("calibration sources must be an exact tuple")
    if not MIN_PREPARED_DOMAINS <= len(value) <= MAX_PREPARED_ARTIFACT_DOMAINS:
        raise ValueError("calibration sources have an unsupported domain count")
    sources: list[PreparedCalibrationSource] = []
    for source in value:
        if type(source) is not PreparedCalibrationSource:
            raise TypeError("calibration sources must contain exact source values")
        sources.append(
            PreparedCalibrationSource(
                held_out_domain_index=source.held_out_domain_index,
                held_out_domain=source.held_out_domain,
                training_subset_index=source.training_subset_index,
                score_block_index=source.score_block_index,
                row_count=source.row_count,
                raw_score_block_sha256=source.raw_score_block_sha256,
                scored_feature_shard_sha256=source.scored_feature_shard_sha256,
                target_shard_sha256=source.target_shard_sha256,
            )
        )
    snapshot = tuple(sources)
    if tuple(source.held_out_domain_index for source in snapshot) != tuple(range(len(snapshot))):
        raise ValueError("calibration sources must be in ascending held-out-domain order")
    if tuple(source.held_out_domain for source in snapshot) != tuple(
        sorted(source.held_out_domain for source in snapshot)
    ):
        raise ValueError("calibration source domains must be sorted")
    if len({source.held_out_domain for source in snapshot}) != len(snapshot):
        raise ValueError("calibration source domains must be unique")
    if sum(source.row_count for source in snapshot) > MAX_PREPARED_EXAMPLES:
        raise ValueError("calibration sources exceed the prepared example limit")
    return snapshot


def _write_calibration_source(
    writer: _HashWriter,
    source: PreparedCalibrationSource,
) -> None:
    writer.integer(
        "calibration_source.held_out_domain_index",
        source.held_out_domain_index,
    )
    writer.text("calibration_source.held_out_domain", source.held_out_domain)
    writer.integer(
        "calibration_source.training_subset_index",
        source.training_subset_index,
    )
    writer.integer(
        "calibration_source.score_block_index",
        source.score_block_index,
    )
    writer.integer("calibration_source.row_count", source.row_count)
    writer.text(
        "calibration_source.raw_score_block_sha256",
        source.raw_score_block_sha256,
    )
    writer.text(
        "calibration_source.scored_feature_shard_sha256",
        source.scored_feature_shard_sha256,
    )
    writer.text(
        "calibration_source.target_shard_sha256",
        source.target_shard_sha256,
    )


def prepared_calibration_input_sha256(
    model_id: str,
    sources: tuple[PreparedCalibrationSource, ...],
    joined_pairs: Iterable[tuple[float, float]],
) -> str:
    """Hash one model's D ordered OOF score/target streams without retaining N pairs."""

    _bounded_text(
        model_id,
        "calibration model_id",
        max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
    )
    source_snapshot = _snapshot_calibration_sources(sources)
    try:
        pair_iterator = iter(joined_pairs)
    except (TypeError, RuntimeError) as error:
        raise TypeError("joined_pairs must be an iterable of exact pairs") from error
    expected_pairs = sum(source.row_count for source in source_snapshot)

    def pair_parts() -> Iterable[bytes]:
        for pair_index in range(expected_pairs):
            try:
                pair = next(pair_iterator)
            except StopIteration as error:
                raise ValueError("joined_pairs ended before the declared row count") from error
            except (TypeError, RuntimeError) as error:
                raise ValueError("joined_pairs could not be read deterministically") from error
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("joined_pairs must contain exact two-value tuples")
            score = _canonical_f64(pair[0], f"joined_pairs[{pair_index}].raw_score")
            target = _canonical_f64(pair[1], f"joined_pairs[{pair_index}].target")
            yield struct.pack("<dd", score, target)
        try:
            next(pair_iterator)
        except StopIteration:
            return
        except (TypeError, RuntimeError) as error:
            raise ValueError("joined_pairs could not be read deterministically") from error
        raise ValueError("joined_pairs exceeds the declared row count")

    writer = _HashWriter(PREPARED_PREDICTOR_CALIBRATION_INPUT_ALGORITHM_ID)
    writer.text("model_id", model_id)
    writer.integer("calibration_source_count", len(source_snapshot))
    for source in source_snapshot:
        _write_calibration_source(writer, source)
    writer.integer("joined_pair_count", expected_pairs)
    writer.token_parts(
        "joined.raw-score-target.f64le",
        expected_pairs * 2 * _F64_BYTES,
        pair_parts(),
    )
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedModelCalibration:
    """One model's self-declared input root and recomputable isotonic child root."""

    model_id: str
    sources: tuple[PreparedCalibrationSource, ...]
    input_sha256: str
    calibrator: IsotonicCalibrator
    identity_sha256: str = field(init=False)
    algorithm_id: str = field(
        default=PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID,
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self) is not PreparedModelCalibration:
            raise TypeError("model calibration must be an exact project type")
        _bounded_text(
            self.model_id,
            "calibration model_id",
            max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
        )
        sources = _snapshot_calibration_sources(self.sources)
        input_sha256 = _sha256_hex(
            self.input_sha256,
            "calibration input_sha256",
        )
        if type(self.calibrator) is not IsotonicCalibrator:
            raise TypeError("calibrator must be an exact IsotonicCalibrator")
        bounds = tuple(
            _canonical_f64(value, "calibrator upper bound")
            for value in self.calibrator.upper_bounds
        )
        values = tuple(
            _canonical_f64(value, "calibrator value") for value in self.calibrator.values
        )
        calibrator = IsotonicCalibrator(bounds, values)
        if len(bounds) > sum(source.row_count for source in sources):
            raise ValueError("calibrator points exceed the calibration source rows")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "calibrator", calibrator)
        object.__setattr__(
            self,
            "identity_sha256",
            _model_calibration_sha256(input_sha256, calibrator),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the model-local serialized calibrator state."""

        return {
            "upper_bounds": list(self.calibrator.upper_bounds),
            "values": list(self.calibrator.values),
            "input_sha256": self.input_sha256,
            "identity_sha256": self.identity_sha256,
        }


def _model_calibration_sha256(
    input_sha256: str,
    calibrator: IsotonicCalibrator,
) -> str:
    writer = _HashWriter(PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID)
    writer.text("input_sha256", input_sha256)
    writer.floats("upper_bounds", calibrator.upper_bounds)
    writer.floats("values", calibrator.values)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedModelState:
    """One serialized model row with its prepared calibration identity."""

    weights: tuple[float, ...]
    bias: float
    calibration: PreparedModelCalibration

    def __post_init__(self) -> None:
        if type(self) is not PreparedModelState:
            raise TypeError("prepared model state must be an exact project type")
        if type(self.weights) is not tuple or not self.weights:
            raise TypeError("prepared model weights must be a non-empty exact tuple")
        if len(self.weights) > MAX_PREPARED_FEATURES:
            raise ValueError("prepared model weights exceed the prepared feature limit")
        weights = tuple(_canonical_f64(value, "prepared model weight") for value in self.weights)
        bias = _canonical_f64(self.bias, "prepared model bias")
        if type(self.calibration) is not PreparedModelCalibration:
            raise TypeError("prepared model calibration must be an exact project type")
        calibration = PreparedModelCalibration(
            model_id=self.calibration.model_id,
            sources=self.calibration.sources,
            input_sha256=self.calibration.input_sha256,
            calibrator=self.calibration.calibrator,
        )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "calibration", calibration)


@dataclass(frozen=True, slots=True)
class PreparedArtifactLineage:
    """Immutable roots retained by the inference-only artifact."""

    source_fit_sha256: str
    store_sha256: str
    statistics_bundle_sha256: str
    raw_score_bundle_sha256: str
    embedding_snapshot_sha256: str | None
    aggregate_statistics_sha256: str
    final_coefficient_sha256: str
    calibration_sources: tuple[PreparedCalibrationSource, ...]
    assembly_algorithm_id: str = PREPARED_ALL_DOMAIN_ASSEMBLY_ALGORITHM_ID
    graph_algorithm_id: str = PREPARED_GRAPH_ALGORITHM_ID
    surface_feature_algorithm_id: str = SURFACE_FEATURE_ALGORITHM_ID
    aggregate_statistics_algorithm_id: str = PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID
    final_coefficient_algorithm_id: str = PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID
    target_shard_algorithm_id: str = PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID
    calibrator_input_algorithm_id: str = PREPARED_PREDICTOR_CALIBRATION_INPUT_ALGORITHM_ID
    calibrator_algorithm_id: str = PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID

    def __post_init__(self) -> None:
        if type(self) is not PreparedArtifactLineage:
            raise TypeError("prepared artifact lineage must be an exact project type")
        frozen = {
            "assembly_algorithm_id": PREPARED_ALL_DOMAIN_ASSEMBLY_ALGORITHM_ID,
            "graph_algorithm_id": PREPARED_GRAPH_ALGORITHM_ID,
            "surface_feature_algorithm_id": SURFACE_FEATURE_ALGORITHM_ID,
            "aggregate_statistics_algorithm_id": (PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID),
            "final_coefficient_algorithm_id": PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID,
            "target_shard_algorithm_id": PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID,
            "calibrator_input_algorithm_id": (PREPARED_PREDICTOR_CALIBRATION_INPUT_ALGORITHM_ID),
            "calibrator_algorithm_id": PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID,
        }
        for name, expected in frozen.items():
            if type(getattr(self, name)) is not str or getattr(self, name) != expected:
                raise ValueError(f"{name} must equal {expected!r}")
        for name in (
            "source_fit_sha256",
            "store_sha256",
            "statistics_bundle_sha256",
            "raw_score_bundle_sha256",
            "aggregate_statistics_sha256",
            "final_coefficient_sha256",
        ):
            object.__setattr__(self, name, _sha256_hex(getattr(self, name), name))
        if self.embedding_snapshot_sha256 is not None:
            object.__setattr__(
                self,
                "embedding_snapshot_sha256",
                _sha256_hex(
                    self.embedding_snapshot_sha256,
                    "embedding_snapshot_sha256",
                ),
            )
        object.__setattr__(
            self,
            "calibration_sources",
            _snapshot_calibration_sources(self.calibration_sources),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact canonical lineage object."""

        return {
            "assembly_algorithm_id": self.assembly_algorithm_id,
            "graph_algorithm_id": self.graph_algorithm_id,
            "surface_feature_algorithm_id": self.surface_feature_algorithm_id,
            "aggregate_statistics_algorithm_id": self.aggregate_statistics_algorithm_id,
            "final_coefficient_algorithm_id": self.final_coefficient_algorithm_id,
            "target_shard_algorithm_id": self.target_shard_algorithm_id,
            "calibrator_input_algorithm_id": self.calibrator_input_algorithm_id,
            "calibrator_algorithm_id": self.calibrator_algorithm_id,
            "source_fit_sha256": self.source_fit_sha256,
            "store_sha256": self.store_sha256,
            "statistics_bundle_sha256": self.statistics_bundle_sha256,
            "raw_score_bundle_sha256": self.raw_score_bundle_sha256,
            "embedding_snapshot_sha256": self.embedding_snapshot_sha256,
            "aggregate_statistics_sha256": self.aggregate_statistics_sha256,
            "final_coefficient_sha256": self.final_coefficient_sha256,
            "calibration_sources": [source.to_dict() for source in self.calibration_sources],
        }


def _lineage_from_dict(payload: object) -> PreparedArtifactLineage:
    item = _shared_artifacts._mapping(payload, "lineage", max_items=16)
    expected = {
        "assembly_algorithm_id",
        "graph_algorithm_id",
        "surface_feature_algorithm_id",
        "aggregate_statistics_algorithm_id",
        "final_coefficient_algorithm_id",
        "target_shard_algorithm_id",
        "calibrator_input_algorithm_id",
        "calibrator_algorithm_id",
        "source_fit_sha256",
        "store_sha256",
        "statistics_bundle_sha256",
        "raw_score_bundle_sha256",
        "embedding_snapshot_sha256",
        "aggregate_statistics_sha256",
        "final_coefficient_sha256",
        "calibration_sources",
    }
    _shared_artifacts._strict_fields(item, expected, "lineage")
    raw_sources = item["calibration_sources"]
    if type(raw_sources) is not list:
        raise ValueError("lineage.calibration_sources must be an array")
    if not MIN_PREPARED_DOMAINS <= len(raw_sources) <= MAX_PREPARED_ARTIFACT_DOMAINS:
        raise ValueError("lineage.calibration_sources has an unsupported length")
    sources = tuple(
        PreparedCalibrationSource.from_dict(
            _shared_artifacts._mapping(
                raw_source,
                f"lineage.calibration_sources[{index}]",
                max_items=8,
            )
        )
        for index, raw_source in enumerate(raw_sources)
    )
    return PreparedArtifactLineage(
        source_fit_sha256=item["source_fit_sha256"],  # type: ignore[arg-type]
        store_sha256=item["store_sha256"],  # type: ignore[arg-type]
        statistics_bundle_sha256=item["statistics_bundle_sha256"],  # type: ignore[arg-type]
        raw_score_bundle_sha256=item["raw_score_bundle_sha256"],  # type: ignore[arg-type]
        embedding_snapshot_sha256=item["embedding_snapshot_sha256"],  # type: ignore[arg-type]
        aggregate_statistics_sha256=item[  # type: ignore[arg-type]
            "aggregate_statistics_sha256"
        ],
        final_coefficient_sha256=item["final_coefficient_sha256"],  # type: ignore[arg-type]
        calibration_sources=sources,
        assembly_algorithm_id=item["assembly_algorithm_id"],  # type: ignore[arg-type]
        graph_algorithm_id=item["graph_algorithm_id"],  # type: ignore[arg-type]
        surface_feature_algorithm_id=item[  # type: ignore[arg-type]
            "surface_feature_algorithm_id"
        ],
        aggregate_statistics_algorithm_id=item[  # type: ignore[arg-type]
            "aggregate_statistics_algorithm_id"
        ],
        final_coefficient_algorithm_id=item[  # type: ignore[arg-type]
            "final_coefficient_algorithm_id"
        ],
        target_shard_algorithm_id=item["target_shard_algorithm_id"],  # type: ignore[arg-type]
        calibrator_input_algorithm_id=item[  # type: ignore[arg-type]
            "calibrator_input_algorithm_id"
        ],
        calibrator_algorithm_id=item["calibrator_algorithm_id"],  # type: ignore[arg-type]
    )


def _semantic_plan(
    domains: tuple[str, ...],
    sources: tuple[PreparedCalibrationSource, ...],
    *,
    embedding_dimension: int,
    target_count: int,
) -> PreparedNestedLodoPlan:
    return build_prepared_nested_lodo_plan(
        domains,
        tuple(source.row_count for source in sources),
        feature_count=_UNIVERSAL_SURFACE_DIMENSION + embedding_dimension,
        target_count=target_count,
    )


def _validate_semantic_sources(
    plan: PreparedNestedLodoPlan,
    sources: tuple[PreparedCalibrationSource, ...],
) -> None:
    all_domains = tuple(range(len(plan.domains)))
    for domain_index, source in enumerate(sources):
        if (
            source.held_out_domain_index != domain_index
            or source.held_out_domain != plan.domains[domain_index]
            or source.row_count != plan.domain_example_counts[domain_index]
        ):
            raise ValueError("calibration source does not match its held-out domain")
        expected_training = tuple(index for index in all_domains if index != domain_index)
        matching_subsets = tuple(
            index
            for index, subset in enumerate(plan.training_subsets)
            if subset.domain_indices == expected_training
        )
        if len(matching_subsets) != 1:
            raise ValueError("prepared graph lacks one semantic all-but-held-out subset")
        expected_subset = matching_subsets[0]
        matching_blocks = tuple(
            index
            for index, block in enumerate(plan.score_blocks)
            if block.training_subset_index == expected_subset
            and block.scored_domain_index == domain_index
        )
        if len(matching_blocks) != 1:
            raise ValueError("prepared graph lacks one semantic OOF score block")
        if (
            source.training_subset_index != expected_subset
            or source.score_block_index != matching_blocks[0]
        ):
            raise ValueError("calibration source does not match its semantic graph context")


def _metadata_bytes_for_artifact(
    schema: PromptFeatureSchema,
    model_ids: tuple[str, ...],
    domains: tuple[str, ...],
    lineage: PreparedArtifactLineage,
) -> int:
    fixed_values = (
        PREPARED_PREDICTOR_ARTIFACT_KIND,
        PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID,
        PREPARED_MOMENT_RIDGE_SOLVER_ID,
        PREPARED_RAW_SCORER_ID,
        lineage.assembly_algorithm_id,
        lineage.graph_algorithm_id,
        lineage.surface_feature_algorithm_id,
        lineage.aggregate_statistics_algorithm_id,
        lineage.final_coefficient_algorithm_id,
        lineage.target_shard_algorithm_id,
        lineage.calibrator_input_algorithm_id,
        lineage.calibrator_algorithm_id,
    )
    total = sum(
        _shared_artifacts._metadata_bytes(value, "prepared artifact algorithm ID")
        for value in fixed_values
    )
    for model_id in model_ids:
        total += _shared_artifacts._metadata_bytes(model_id, "prepared artifact model ID")
    for domain in domains:
        # One occurrence in training and one in calibration_sources.
        total += 2 * _shared_artifacts._metadata_bytes(
            domain,
            "prepared artifact domain",
        )
    for tag in schema.domain_tags:
        total += _shared_artifacts._metadata_bytes(tag, "feature domain tag")
    identity = schema.embedding_identity
    if identity is not None:
        for name in ("provider", "model_id", "revision", "pooling"):
            total += _shared_artifacts._metadata_bytes(
                getattr(identity, name),
                f"embedding {name}",
            )
        total += len(identity.asset_manifest_sha256)
    return total


@dataclass(frozen=True, slots=True)
class PreparedBilinearPredictorArtifact:
    """Inference state plus immutable prepared-store lineage."""

    feature_schema: PromptFeatureSchema
    models: Mapping[str, PreparedModelState]
    training_domains: tuple[str, ...]
    training_example_count: int
    ridge: float
    lineage: PreparedArtifactLineage
    solver_id: str = PREPARED_MOMENT_RIDGE_SOLVER_ID
    raw_scorer_id: str = PREPARED_RAW_SCORER_ID
    algorithm_id: str = PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID
    artifact_kind: str = PREPARED_PREDICTOR_ARTIFACT_KIND
    artifact_version: int = PREPARED_PREDICTOR_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if type(self) is not PreparedBilinearPredictorArtifact:
            raise TypeError("prepared predictor artifact must be an exact project type")
        if (
            type(self.artifact_version) is not int
            or self.artifact_version != PREPARED_PREDICTOR_ARTIFACT_VERSION
        ):
            raise ValueError(f"artifact_version must equal {PREPARED_PREDICTOR_ARTIFACT_VERSION}")
        frozen = {
            "artifact_kind": PREPARED_PREDICTOR_ARTIFACT_KIND,
            "algorithm_id": PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID,
            "solver_id": PREPARED_MOMENT_RIDGE_SOLVER_ID,
            "raw_scorer_id": PREPARED_RAW_SCORER_ID,
        }
        for name, expected in frozen.items():
            if type(getattr(self, name)) is not str or getattr(self, name) != expected:
                raise ValueError(f"{name} must equal {expected!r}")
        schema = _snapshot_schema(self.feature_schema)
        models_input = _shared_artifacts._mapping(
            self.models,
            "models",
            max_items=MAX_PREPARED_ARTIFACT_MODELS,
        )
        model_ids = tuple(sorted(models_input))
        _snapshot_model_ids(model_ids)
        states: dict[str, PreparedModelState] = {}
        numeric_scalars = 6 + 1  # schema means/scales and ridge
        for model_id in model_ids:
            raw_state = models_input[model_id]
            if type(raw_state) is not PreparedModelState:
                raise TypeError("models must map IDs to exact PreparedModelState values")
            if type(raw_state.weights) is not tuple or len(raw_state.weights) != schema.dimension:
                raise ValueError("model weight width does not match the feature schema")
            state = PreparedModelState(
                weights=raw_state.weights,
                bias=raw_state.bias,
                calibration=raw_state.calibration,
            )
            if state.calibration.model_id != model_id:
                raise ValueError("model calibration ID does not match its model key")
            numeric_scalars += len(state.weights) + 1
            numeric_scalars += 2 * len(state.calibration.calibrator.upper_bounds)
            if numeric_scalars > MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS:
                raise ValueError(
                    "prepared artifact exceeds the numeric scalar limit "
                    f"({MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS:,})"
                )
            states[model_id] = state
        domains = _shared_artifacts._text_tuple(
            self.training_domains,
            "training_domains",
            max_items=MAX_PREPARED_ARTIFACT_DOMAINS,
        )
        if not MIN_PREPARED_DOMAINS <= len(domains) <= MAX_PREPARED_ARTIFACT_DOMAINS:
            raise ValueError("training_domains has an unsupported prepared domain count")
        if any(not domain.strip() for domain in domains) or domains != tuple(sorted(set(domains))):
            raise ValueError("training_domains must be sorted unique non-empty strings")
        example_count = _exact_positive_int(
            self.training_example_count,
            "training_example_count",
        )
        if example_count > MAX_PREPARED_EXAMPLES:
            raise ValueError("training_example_count exceeds the prepared example limit")
        ridge = _canonical_f64(self.ridge, "artifact ridge", positive=True)
        if type(self.lineage) is not PreparedArtifactLineage:
            raise TypeError("lineage must be an exact PreparedArtifactLineage")
        lineage = PreparedArtifactLineage(
            source_fit_sha256=self.lineage.source_fit_sha256,
            store_sha256=self.lineage.store_sha256,
            statistics_bundle_sha256=self.lineage.statistics_bundle_sha256,
            raw_score_bundle_sha256=self.lineage.raw_score_bundle_sha256,
            embedding_snapshot_sha256=self.lineage.embedding_snapshot_sha256,
            aggregate_statistics_sha256=self.lineage.aggregate_statistics_sha256,
            final_coefficient_sha256=self.lineage.final_coefficient_sha256,
            calibration_sources=self.lineage.calibration_sources,
            assembly_algorithm_id=self.lineage.assembly_algorithm_id,
            graph_algorithm_id=self.lineage.graph_algorithm_id,
            surface_feature_algorithm_id=self.lineage.surface_feature_algorithm_id,
            aggregate_statistics_algorithm_id=(self.lineage.aggregate_statistics_algorithm_id),
            final_coefficient_algorithm_id=self.lineage.final_coefficient_algorithm_id,
            target_shard_algorithm_id=self.lineage.target_shard_algorithm_id,
            calibrator_input_algorithm_id=self.lineage.calibrator_input_algorithm_id,
            calibrator_algorithm_id=self.lineage.calibrator_algorithm_id,
        )
        if example_count != sum(source.row_count for source in lineage.calibration_sources):
            raise ValueError("training_example_count does not match calibration sources")
        if (schema.embedding_dimension == 0) != (lineage.embedding_snapshot_sha256 is None):
            raise ValueError(
                "embedding snapshot lineage must be present exactly for embedded schemas"
            )
        plan = _semantic_plan(
            domains,
            lineage.calibration_sources,
            embedding_dimension=schema.embedding_dimension,
            target_count=len(model_ids),
        )
        _validate_semantic_sources(plan, lineage.calibration_sources)
        for state in states.values():
            if state.calibration.sources != lineage.calibration_sources:
                raise ValueError("model calibration sources do not match artifact lineage")
            if len(state.calibration.calibrator.upper_bounds) > example_count:
                raise ValueError("model calibrator exceeds training_example_count")
        if (
            _metadata_bytes_for_artifact(schema, model_ids, domains, lineage)
            > MAX_PREDICTOR_METADATA_TOTAL_BYTES
        ):
            raise ValueError(
                "prepared artifact metadata exceeds the aggregate limit "
                f"({MAX_PREDICTOR_METADATA_TOTAL_BYTES:,} UTF-8 bytes)"
            )
        weights_payload = b"".join(
            _canonical_f64_bytes(states[model_id].weights, "artifact model weights")
            for model_id in model_ids
        )
        intercepts_payload = _canonical_f64_bytes(
            tuple(states[model_id].bias for model_id in model_ids),
            "artifact model bias",
        )
        coefficient = PreparedFinalCoefficient(
            feature_schema=schema,
            active_feature_indices=_active_feature_indices(schema),
            model_ids=model_ids,
            aggregate_statistics_sha256=lineage.aggregate_statistics_sha256,
            ridge=ridge,
            weights_payload=weights_payload,
            intercepts_payload=intercepts_payload,
        )
        if not hmac.compare_digest(
            coefficient.sha256,
            lineage.final_coefficient_sha256,
        ):
            raise ValueError("final_coefficient_sha256 does not match serialized inference state")
        object.__setattr__(self, "feature_schema", schema)
        object.__setattr__(self, "models", MappingProxyType(states))
        object.__setattr__(self, "training_domains", domains)
        object.__setattr__(self, "training_example_count", example_count)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "lineage", lineage)
        # Direct construction is a serialization trust boundary.  An accepted object
        # must already fit the exact bounded document contract.
        self.to_json()

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Return the canonical model catalogue."""

        return tuple(self.models)

    def build_predictor(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> PerModelCalibratedQualityPredictor:
        """Rebuild offline inference; embedding work starts only on prediction."""

        encoder = PromptFeatureEncoder(self.feature_schema, embedding_provider)
        weights = {model_id: self.models[model_id].weights for model_id in self.model_ids}
        biases = {model_id: self.models[model_id].bias for model_id in self.model_ids}
        calibrators = {
            model_id: self.models[model_id].calibration.calibrator for model_id in self.model_ids
        }
        base = BilinearQualityPredictor(
            vectorizer=encoder.transform_one,
            model_weights=MappingProxyType(weights),
            model_bias=MappingProxyType(biases),
            batch_vectorizer=encoder.transform_many,
        )
        return PerModelCalibratedQualityPredictor(
            base,
            MappingProxyType(calibrators),
        )

    @classmethod
    def from_prepared_components(
        cls,
        coefficient: PreparedFinalCoefficient,
        calibrations: Mapping[str, PreparedModelCalibration],
        *,
        training_domains: tuple[str, ...],
        training_example_count: int,
        lineage: PreparedArtifactLineage,
    ) -> PreparedBilinearPredictorArtifact:
        """Materialize inference state from an admitted final solve and calibrators."""

        if cls is not PreparedBilinearPredictorArtifact:
            raise TypeError("component assembly requires the exact artifact type")
        if type(coefficient) is not PreparedFinalCoefficient:
            raise TypeError("coefficient must be an exact PreparedFinalCoefficient")
        calibration_input = _shared_artifacts._mapping(
            calibrations,
            "calibrations",
            max_items=MAX_PREPARED_ARTIFACT_MODELS,
        )
        if tuple(sorted(calibration_input)) != coefficient.model_ids:
            raise ValueError("calibrations do not match the final coefficient catalogue")
        if coefficient.sha256 != lineage.final_coefficient_sha256:
            raise ValueError("final coefficient does not match artifact lineage")
        models: dict[str, PreparedModelState] = {}
        for model_index, model_id in enumerate(coefficient.model_ids):
            calibration = calibration_input[model_id]
            if type(calibration) is not PreparedModelCalibration:
                raise TypeError("calibrations must contain exact model-calibration values")
            models[model_id] = PreparedModelState(
                weights=coefficient.weights_for_model_index(model_index),
                bias=coefficient.intercept_for_model_index(model_index),
                calibration=calibration,
            )
        return cls(
            feature_schema=coefficient.feature_schema,
            models=models,
            training_domains=training_domains,
            training_example_count=training_example_count,
            ridge=coefficient.ridge,
            lineage=lineage,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact canonical JSON-compatible object."""

        return {
            "algorithm_id": self.algorithm_id,
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "feature_schema": self.feature_schema.to_dict(),
            "models": {
                model_id: {
                    "weights": list(self.models[model_id].weights),
                    "bias": self.models[model_id].bias,
                    "calibrator": self.models[model_id].calibration.to_dict(),
                }
                for model_id in self.model_ids
            },
            "training": {
                "domains": list(self.training_domains),
                "example_count": self.training_example_count,
                "ridge": self.ridge,
                "solver_id": self.solver_id,
                "raw_scorer_id": self.raw_scorer_id,
            },
            "lineage": self.lineage.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize canonical UTF-8 JSON with finite numbers and one final newline."""

        document = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        _validate_artifact_document(document)
        return document

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> PreparedBilinearPredictorArtifact:
        """Validate and construct one version-1 prepared artifact."""

        if cls is not PreparedBilinearPredictorArtifact:
            raise TypeError("artifact parsing requires the exact project type")
        root = _shared_artifacts._mapping(payload, "artifact", max_items=7)
        _shared_artifacts._strict_fields(
            root,
            {
                "algorithm_id",
                "artifact_kind",
                "artifact_version",
                "feature_schema",
                "models",
                "training",
                "lineage",
            },
            "artifact",
        )
        if (
            type(root["artifact_version"]) is not int
            or root["artifact_version"] != PREPARED_PREDICTOR_ARTIFACT_VERSION
        ):
            raise ValueError(f"artifact_version must equal {PREPARED_PREDICTOR_ARTIFACT_VERSION}")
        if root["artifact_kind"] != PREPARED_PREDICTOR_ARTIFACT_KIND:
            raise ValueError(f"artifact_kind must equal {PREPARED_PREDICTOR_ARTIFACT_KIND!r}")
        if root["algorithm_id"] != PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID:
            raise ValueError(
                f"algorithm_id must equal {PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID!r}"
            )
        training = _shared_artifacts._mapping(
            root["training"],
            "training",
            max_items=5,
        )
        _shared_artifacts._strict_fields(
            training,
            {"domains", "example_count", "ridge", "solver_id", "raw_scorer_id"},
            "training",
        )
        if training["solver_id"] != PREPARED_MOMENT_RIDGE_SOLVER_ID:
            raise ValueError(f"solver_id must equal {PREPARED_MOMENT_RIDGE_SOLVER_ID!r}")
        if training["raw_scorer_id"] != PREPARED_RAW_SCORER_ID:
            raise ValueError(f"raw_scorer_id must equal {PREPARED_RAW_SCORER_ID!r}")
        domains = _shared_artifacts._text_tuple(
            training["domains"],
            "training.domains",
            max_items=MAX_PREPARED_ARTIFACT_DOMAINS,
        )
        example_count = _exact_positive_int(
            training["example_count"],
            "training.example_count",
        )
        ridge = _canonical_f64(training["ridge"], "training.ridge", positive=True)
        lineage = _lineage_from_dict(root["lineage"])
        feature_schema = PromptFeatureSchema.from_dict(
            _shared_artifacts._mapping(
                root["feature_schema"],
                "feature_schema",
                max_items=6,
            )
        )
        feature_schema = _snapshot_schema(feature_schema)
        models_payload = _shared_artifacts._mapping(
            root["models"],
            "models",
            max_items=MAX_PREPARED_ARTIFACT_MODELS,
        )
        model_ids = tuple(sorted(models_payload))
        _snapshot_model_ids(model_ids)
        models: dict[str, PreparedModelState] = {}
        numeric_scalars = 7
        for model_id in model_ids:
            item = _shared_artifacts._mapping(
                models_payload[model_id],
                f"models.{model_id}",
                max_items=3,
            )
            _shared_artifacts._strict_fields(
                item,
                {"weights", "bias", "calibrator"},
                f"models.{model_id}",
            )
            remaining = MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS - numeric_scalars
            weights = _json_f64_tuple(
                item["weights"],
                f"models.{model_id}.weights",
                max_items=min(feature_schema.dimension, remaining),
            )
            if len(weights) != feature_schema.dimension:
                raise ValueError(f"models.{model_id}.weights has the wrong feature width")
            numeric_scalars += len(weights) + 1
            bias = _canonical_f64(item["bias"], f"models.{model_id}.bias")
            calibrator_item = _shared_artifacts._mapping(
                item["calibrator"],
                f"models.{model_id}.calibrator",
                max_items=4,
            )
            _shared_artifacts._strict_fields(
                calibrator_item,
                {"upper_bounds", "values", "input_sha256", "identity_sha256"},
                f"models.{model_id}.calibrator",
            )
            remaining = MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS - numeric_scalars
            point_limit = min(
                example_count,
                MAX_PREDICTOR_CALIBRATOR_POINTS,
                remaining // 2,
            )
            bounds = _json_f64_tuple(
                calibrator_item["upper_bounds"],
                f"models.{model_id}.calibrator.upper_bounds",
                max_items=point_limit,
            )
            values = _json_f64_tuple(
                calibrator_item["values"],
                f"models.{model_id}.calibrator.values",
                max_items=point_limit,
            )
            numeric_scalars += len(bounds) + len(values)
            if numeric_scalars > MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS:
                raise ValueError(
                    "prepared artifact exceeds the numeric scalar limit "
                    f"({MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS:,})"
                )
            calibration = PreparedModelCalibration(
                model_id=model_id,
                sources=lineage.calibration_sources,
                input_sha256=calibrator_item["input_sha256"],  # type: ignore[arg-type]
                calibrator=IsotonicCalibrator(bounds, values),
            )
            supplied_identity = _sha256_hex(
                calibrator_item["identity_sha256"],
                f"models.{model_id}.calibrator.identity_sha256",
            )
            if not hmac.compare_digest(
                calibration.identity_sha256,
                supplied_identity,
            ):
                raise ValueError(
                    f"models.{model_id}.calibrator.identity_sha256 does not match state"
                )
            models[model_id] = PreparedModelState(
                weights=weights,
                bias=bias,
                calibration=calibration,
            )
        return cls(
            artifact_version=root["artifact_version"],  # type: ignore[arg-type]
            artifact_kind=root["artifact_kind"],  # type: ignore[arg-type]
            algorithm_id=root["algorithm_id"],  # type: ignore[arg-type]
            feature_schema=feature_schema,
            models=models,
            training_domains=domains,
            training_example_count=example_count,
            ridge=ridge,
            solver_id=training["solver_id"],  # type: ignore[arg-type]
            raw_scorer_id=training["raw_scorer_id"],  # type: ignore[arg-type]
            lineage=lineage,
        )

    @classmethod
    def from_json(cls, document: str) -> PreparedBilinearPredictorArtifact:
        """Parse bounded strict JSON without a trusted external pin."""

        if cls is not PreparedBilinearPredictorArtifact:
            raise TypeError("artifact parsing requires the exact project type")
        _validate_artifact_document(document)
        _shared_artifacts._preflight_json_structure(document)
        number_tokens = 0

        def count_number() -> None:
            nonlocal number_tokens
            number_tokens += 1
            if number_tokens > MAX_PREPARED_ARTIFACT_JSON_NUMBER_TOKENS:
                raise ValueError(
                    "prepared artifact exceeds the JSON number-token limit "
                    f"({MAX_PREPARED_ARTIFACT_JSON_NUMBER_TOKENS:,})"
                )

        def parse_int(token: str) -> int:
            count_number()
            if len(token) > MAX_PREDICTOR_JSON_NUMBER_CHARACTERS:
                raise ValueError("prepared artifact integer token is too long")
            return _shared_artifacts._bounded_json_integer(token)

        def parse_float(token: str) -> float:
            count_number()
            if len(token) > MAX_PREDICTOR_JSON_NUMBER_CHARACTERS:
                raise ValueError("prepared artifact float token is too long")
            return _shared_artifacts._bounded_json_float(token)

        def reject_constant(value: str) -> object:
            raise ValueError(f"non-standard JSON number {value!r} is forbidden")

        def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r} is forbidden")
                result[key] = value
            return result

        try:
            payload = json.loads(
                document,
                parse_int=parse_int,
                parse_float=parse_float,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError("prepared predictor artifact is not valid strict JSON") from error
        return cls.from_dict(_shared_artifacts._mapping(payload, "artifact", max_items=7))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_artifact_sha256: str,
    ) -> PreparedBilinearPredictorArtifact:
        """Load one pinned, stable, non-symlink regular-file document."""

        if cls is not PreparedBilinearPredictorArtifact:
            raise TypeError("artifact loading requires the exact project type")
        expected_sha256 = _sha256_hex(
            expected_artifact_sha256,
            "expected_artifact_sha256",
        )
        source = _artifact_path(path)
        payload = _read_pinned_artifact(source)
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise ValueError("prepared artifact SHA-256 does not match the trusted pin")
        try:
            document = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("prepared artifact is not valid UTF-8") from error
        artifact = cls.from_json(document)
        if artifact.to_json().encode("utf-8") != payload:
            raise ValueError("prepared artifact input bytes are not exact canonical JSON")
        return artifact

    def save(self, path: str | Path) -> Path:
        """Validate fully, then atomically publish exactly one same-directory stage."""

        if type(self) is not PreparedBilinearPredictorArtifact:
            raise TypeError("artifact save requires the exact project type")
        document = self.to_json()
        # Complete parsing and exact canonical validation before the stage exists.
        validated = PreparedBilinearPredictorArtifact.from_json(document)
        if validated.to_json() != document:
            raise ValueError("prepared artifact failed canonical round-trip validation")
        destination = _artifact_path(path)
        return _save_one_stage(destination, document.encode("utf-8"))


def _artifact_path(value: str | Path) -> Path:
    if type(value) is not str and not isinstance(value, Path):
        raise TypeError("artifact path must be text or a Path")
    try:
        return Path(value)
    except (TypeError, ValueError, OSError) as error:
        raise ValueError("artifact path is invalid") from error


def _is_reparse_point(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _nanosecond_time(details: os.stat_result, name: str) -> int:
    explicit = getattr(details, f"st_{name}_ns", None)
    if explicit is not None:
        return int(explicit)
    return int(getattr(details, f"st_{name}") * 1_000_000_000)


def _descriptor_fingerprint(details: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        _nanosecond_time(details, "mtime"),
        _nanosecond_time(details, "ctime"),
    )


def _validate_regular_node(details: os.stat_result, context: str) -> None:
    if stat.S_ISLNK(details.st_mode) or _is_reparse_point(details):
        raise ValueError(f"{context} must not be a symlink/reparse point")
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{context} must be a regular file")


def _read_pinned_artifact(source: Path) -> bytes:
    try:
        initial_path = source.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect prepared artifact: {source}") from error
    _validate_regular_node(initial_path, "prepared artifact")
    if not 0 < initial_path.st_size <= MAX_PREDICTOR_ARTIFACT_BYTES:
        raise ValueError("prepared artifact size is outside the document limit")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        _validate_regular_node(opened, "opened prepared artifact")
        if (initial_path.st_dev, initial_path.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ValueError("prepared artifact path changed while opening")
        if not 0 < opened.st_size <= MAX_PREDICTOR_ARTIFACT_BYTES:
            raise ValueError("prepared artifact size is outside the document limit")
        payload = bytearray()
        remaining = MAX_PREDICTOR_ARTIFACT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            remaining -= len(chunk)
        if len(payload) > MAX_PREDICTOR_ARTIFACT_BYTES:
            raise ValueError("prepared artifact exceeds the document limit")
        after = os.fstat(descriptor)
        if (
            _descriptor_fingerprint(opened) != _descriptor_fingerprint(after)
            or len(payload) != opened.st_size
        ):
            raise ValueError("prepared artifact changed while reading")
        try:
            final_path = source.lstat()
        except OSError as error:
            raise ValueError("prepared artifact path changed while reading") from error
        _validate_regular_node(final_path, "prepared artifact")
        if (final_path.st_dev, final_path.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ValueError("prepared artifact path changed while reading")
        return bytes(payload)
    except OSError as error:
        raise ValueError(f"cannot read prepared artifact: {source}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - invalid operating-system contract
            raise OSError("prepared artifact stage write made no progress")
        view = view[written:]


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lstat_optional(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"cannot inspect prepared artifact destination: {path}") from error


def _save_one_stage(destination: Path, payload: bytes) -> Path:
    if len(payload) > MAX_PREDICTOR_ARTIFACT_BYTES:
        raise ValueError("prepared artifact exceeds the document limit")
    original = _lstat_optional(destination)
    if original is not None:
        _validate_regular_node(original, "prepared artifact destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_details = destination.parent.lstat()
    except OSError as error:
        raise ValueError("cannot inspect prepared artifact destination directory") from error
    if stat.S_ISLNK(parent_details.st_mode) or _is_reparse_point(parent_details):
        raise ValueError("prepared artifact destination directory must not be a symlink")
    if not stat.S_ISDIR(parent_details.st_mode):
        raise ValueError("prepared artifact destination parent must be a directory")

    descriptor: int | None = None
    stage: Path | None = None
    published = False
    try:
        descriptor, raw_stage = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.stage.",
            suffix=".tmp",
        )
        stage = Path(raw_stage)
        staged = os.fstat(descriptor)
        _validate_regular_node(staged, "prepared artifact stage")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        final_stage = os.fstat(descriptor)
        if final_stage.st_size != len(payload):
            raise OSError("prepared artifact stage has the wrong byte length")
        os.close(descriptor)
        descriptor = None

        current = _lstat_optional(destination)
        if original is None:
            if current is not None:
                raise ValueError("prepared artifact destination appeared during staging")
        else:
            if current is None or _descriptor_fingerprint(current) != (
                _descriptor_fingerprint(original)
            ):
                raise ValueError("prepared artifact destination changed during staging")
            _validate_regular_node(current, "prepared artifact destination")
        stage_details = stage.lstat()
        _validate_regular_node(stage_details, "prepared artifact stage")
        if (stage_details.st_dev, stage_details.st_ino) != (
            final_stage.st_dev,
            final_stage.st_ino,
        ):
            raise ValueError("prepared artifact stage changed before publication")
        os.replace(stage, destination)
        published = True
        _fsync_parent(destination)
        return destination
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if stage is not None and not published:
            try:
                stage.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "MAX_PREPARED_ARTIFACT_DOMAINS",
    "MAX_PREPARED_ARTIFACT_MODELS",
    "MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS",
    "PREPARED_ALL_DOMAIN_ASSEMBLY_ALGORITHM_ID",
    "PREPARED_ALL_DOMAIN_STATISTICS_ALGORITHM_ID",
    "PREPARED_FINAL_COEFFICIENT_ALGORITHM_ID",
    "PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID",
    "PREPARED_PREDICTOR_ARTIFACT_KIND",
    "PREPARED_PREDICTOR_ARTIFACT_VERSION",
    "PREPARED_PREDICTOR_CALIBRATION_INPUT_ALGORITHM_ID",
    "PREPARED_PREDICTOR_CALIBRATOR_ALGORITHM_ID",
    "PREPARED_PREDICTOR_TARGET_SHARD_ALGORITHM_ID",
    "PreparedAllDomainStatistics",
    "PreparedArtifactLineage",
    "PreparedBilinearPredictorArtifact",
    "PreparedCalibrationSource",
    "PreparedFinalCoefficient",
    "PreparedModelCalibration",
    "PreparedModelState",
    "PreparedPredictorTargetShard",
    "prepared_calibration_input_sha256",
]
