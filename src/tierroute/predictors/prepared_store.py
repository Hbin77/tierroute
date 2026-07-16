# SPDX-License-Identifier: Apache-2.0
"""Bounded in-memory reference store for the prepared nested-LODO graph.

This experimental module proves canonical feature/target snapshots, per-domain
centered moments, and training-subset isolation.  It deliberately does not execute
an embedding provider, read or write files, solve ridge systems, score rows, or alter
the default predictor-training path.  The Python reference limits are intentionally
smaller than the graph planner's modeled compact-buffer limits.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise

from tierroute.core import ModelSpec
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.features.encoding import PromptFeatureSchema
from tierroute.features.surface import (
    SURFACE_DOMAIN_TAG_CATALOGUE,
    SURFACE_FEATURE_ALGORITHM_ID,
    extract_surface_features,
)
from tierroute.predictors.prepared_graph import (
    MAX_PREPARED_DOMAIN_UTF8_BYTES,
    MAX_PREPARED_DOMAINS,
    MAX_PREPARED_EXAMPLES,
    MAX_PREPARED_FEATURES,
    MAX_PREPARED_TARGETS,
    PREPARED_GRAPH_ALGORITHM_ID,
    PreparedNestedLodoPlan,
)

PREPARED_EMBEDDING_SNAPSHOT_ALGORITHM_ID = "tierroute.prepared-embedding-snapshot-v1"
PREPARED_FEATURE_STORE_ALGORITHM_ID = "tierroute.prepared-feature-store-v1"
PREPARED_STATISTICS_ALGORITHM_ID = "tierroute.prepared-domain-statistics-welford-v1"
PREPARED_STATISTICS_BUNDLE_ALGORITHM_ID = "tierroute.prepared-statistics-bundle-v1"
PREPARED_SUBSET_STATISTICS_ALGORITHM_ID = "tierroute.prepared-subset-statistics-chan-v1"

_DOMAIN_CONTENT_ALGORITHM_ID = "tierroute.prepared-domain-content-v1"
_FIT_SOURCE_ALGORITHM_ID = "tierroute.prepared-fit-source-v1"
_SUBSET_CONTENT_ALGORITHM_ID = "tierroute.prepared-subset-content-v1"

MAX_PREPARED_REFERENCE_PROMPT_UTF8_BYTES = 1024 * 1024
MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES = 64 * 1024 * 1024
MAX_PREPARED_REFERENCE_NUMERIC_BYTES = 512 * 1024 * 1024
MAX_PREPARED_REFERENCE_STATISTIC_SCALARS = 2_000_000
MAX_PREPARED_REFERENCE_STATISTIC_WORK_UNITS = 50_000_000
MAX_PREPARED_ROW_ID_UTF8_BYTES = 4 * 1024
MAX_PREPARED_MODEL_ID_UTF8_BYTES = 4 * 1024

_CONTINUOUS_COUNT = 3
_BINARY_COUNT = 2
_TAG_OFFSET = _CONTINUOUS_COUNT + _BINARY_COUNT
_UNIVERSAL_SURFACE_DIMENSION = _TAG_OFFSET + len(SURFACE_DOMAIN_TAG_CATALOGUE)
_F64_BYTES = 8
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _sha256_hex(value: object, name: str) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _bounded_text(value: object, name: str, *, max_bytes: int) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    if len(value) > max_bytes:
        raise ValueError(f"{name} exceeds the reviewed UTF-8 byte limit")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid UTF-8 text") from error
    if len(encoded) > max_bytes:
        raise ValueError(f"{name} exceeds the reviewed UTF-8 byte limit")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty non-whitespace text")
    return value


def _finite_f64(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite binary64") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite binary64")
    # The prepared format has one canonical representation for numerical zero.
    return 0.0 if result == 0.0 else result


def _finite_tuple(
    value: object,
    name: str,
    *,
    expected_length: int,
) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(value) != expected_length:
        raise ValueError(f"{name} has the wrong bounded length")
    return tuple(_finite_f64(item, name) for item in value)


def _validate_universal_feature_means(
    means: tuple[float, ...],
    active_tag_mask: int,
    context: str,
) -> None:
    if any(value < 0 for value in means[:_CONTINUOUS_COUNT]):
        raise ValueError(f"{context} continuous means must be non-negative")
    bounded_end = _UNIVERSAL_SURFACE_DIMENSION
    if any(not 0.0 <= value <= 1.0 for value in means[_CONTINUOUS_COUNT:bounded_end]):
        raise ValueError(f"{context} binary/tag means must be between zero and one")
    expected_mask = sum(
        1 << tag_index
        for tag_index in range(len(SURFACE_DOMAIN_TAG_CATALOGUE))
        if means[_TAG_OFFSET + tag_index] > 0.0
    )
    if active_tag_mask != expected_mask:
        raise ValueError(f"{context} active_tag_mask does not match tag means")


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _row_key_text_bytes(
    example_ids: tuple[str, ...],
    prompt_sha256s: tuple[str, ...],
    *,
    require_canonical_order: bool,
) -> int:
    total_bytes = 0
    previous_id: str | None = None
    for example_id, prompt_digest in zip(example_ids, prompt_sha256s, strict=True):
        _bounded_text(
            example_id,
            "row-key example_id",
            max_bytes=MAX_PREPARED_ROW_ID_UTF8_BYTES,
        )
        _sha256_hex(prompt_digest, "row-key prompt_sha256")
        encoded_id = example_id.encode("utf-8")
        total_bytes += len(encoded_id) + len(prompt_digest)
        if total_bytes > MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES:
            raise ValueError("prepared row keys exceed the reference text-byte limit")
        if require_canonical_order and previous_id is not None and previous_id >= example_id:
            raise ValueError("prepared row-key example IDs must be strictly increasing")
        previous_id = example_id
    return total_bytes


class _HashWriter:
    """Length-frame typed fields so concatenation cannot create ambiguity."""

    def __init__(self, namespace: str) -> None:
        self._digest = hashlib.sha256()
        self.text("namespace", namespace)

    def token(self, label: str, payload: bytes) -> None:
        label_bytes = label.encode("ascii")
        self._digest.update(struct.pack("<I", len(label_bytes)))
        self._digest.update(label_bytes)
        self._digest.update(struct.pack("<Q", len(payload)))
        self._digest.update(payload)

    def token_parts(self, label: str, byte_count: int, parts: Iterable[bytes]) -> None:
        label_bytes = label.encode("ascii")
        self._digest.update(struct.pack("<I", len(label_bytes)))
        self._digest.update(label_bytes)
        self._digest.update(struct.pack("<Q", byte_count))
        written = 0
        for part in parts:
            if type(part) is not bytes:
                raise TypeError("digest payload parts must be exact bytes")
            written += len(part)
            if written > byte_count:
                raise ValueError("digest payload parts exceed the declared length")
            self._digest.update(part)
        if written != byte_count:
            raise ValueError("digest payload parts do not match the declared length")

    def text(self, label: str, value: str) -> None:
        self.token(label, value.encode("utf-8"))

    def integer(self, label: str, value: int) -> None:
        self.token(label, struct.pack("<Q", value))

    def boolean(self, label: str, value: bool) -> None:
        self.token(label, b"\x01" if value else b"\x00")

    def floats(self, label: str, values: tuple[float, ...]) -> None:
        self.integer(f"{label}.count", len(values))
        for value in values:
            self.token(label, struct.pack("<d", value))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _write_embedding_identity(writer: _HashWriter, identity: EmbeddingIdentity) -> None:
    writer.text("embedding.provider", identity.provider)
    writer.text("embedding.model_id", identity.model_id)
    writer.text("embedding.revision", identity.revision)
    writer.text("embedding.pooling", identity.pooling)
    writer.boolean("embedding.normalize", identity.normalize)
    writer.text("embedding.asset_manifest_sha256", identity.asset_manifest_sha256)


def _write_plan_identity(writer: _HashWriter, plan: PreparedNestedLodoPlan) -> None:
    writer.text("graph.algorithm_id", plan.algorithm_id)
    writer.integer("graph.domain_count", len(plan.domains))
    for domain, count in zip(plan.domains, plan.domain_example_counts, strict=True):
        writer.text("graph.domain", domain)
        writer.integer("graph.domain_row_count", count)
    writer.integer("graph.feature_count", plan.feature_count)
    writer.integer("graph.target_count", plan.target_count)


@dataclass(frozen=True, slots=True)
class PreparedEmbeddingInput:
    """One caller-precomputed embedding joined to an exact prompt digest."""

    example_id: str
    prompt_sha256: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _bounded_text(
            self.example_id,
            "embedding example_id",
            max_bytes=MAX_PREPARED_ROW_ID_UTF8_BYTES,
        )
        _sha256_hex(self.prompt_sha256, "embedding prompt_sha256")
        if type(self.values) is not tuple:
            raise TypeError("embedding values must be an exact tuple")
        if not self.values or len(self.values) > MAX_PREPARED_FEATURES:
            raise ValueError("embedding values have an unsupported bounded width")
        normalized = tuple(_finite_f64(value, "embedding value") for value in self.values)
        object.__setattr__(self, "values", normalized)


@dataclass(frozen=True, slots=True)
class PreparedEmbeddingSnapshot:
    """Canonical immutable binary64 snapshot; its digest is identity, not provenance."""

    identity: EmbeddingIdentity
    dimension: int
    example_ids: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]
    payload: bytes = field(repr=False)
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_EMBEDDING_SNAPSHOT_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.identity) is not EmbeddingIdentity:
            raise TypeError("embedding identity must be an exact EmbeddingIdentity")
        dimension = _exact_positive_int(self.dimension, "embedding dimension")
        if dimension > MAX_PREPARED_FEATURES - _UNIVERSAL_SURFACE_DIMENSION:
            raise ValueError("embedding dimension exceeds the prepared feature limit")
        if type(self.example_ids) is not tuple or type(self.prompt_sha256s) is not tuple:
            raise TypeError("embedding row keys must be exact tuples")
        row_count = len(self.example_ids)
        if not 0 < row_count <= MAX_PREPARED_EXAMPLES:
            raise ValueError("embedding row count is outside the reviewed limit")
        if len(self.prompt_sha256s) != row_count:
            raise ValueError("embedding row-key fields must have equal lengths")
        expected_bytes = row_count * dimension * _F64_BYTES
        if expected_bytes > MAX_PREPARED_REFERENCE_NUMERIC_BYTES:
            raise ValueError("embedding payload exceeds the reference byte limit")
        if type(self.payload) is not bytes:
            raise TypeError("embedding payload must be immutable bytes")
        if len(self.payload) != expected_bytes:
            raise ValueError("embedding payload has the wrong exact byte length")
        _row_key_text_bytes(
            self.example_ids,
            self.prompt_sha256s,
            require_canonical_order=True,
        )
        for (value,) in struct.iter_unpack("<d", self.payload):
            if not math.isfinite(value):
                raise ValueError("embedding payload must contain only finite binary64 values")
            if value == 0.0 and math.copysign(1.0, value) < 0:
                raise ValueError("embedding payload must use canonical positive zero")
        object.__setattr__(self, "sha256", _embedding_snapshot_sha256(self))


def _embedding_snapshot_sha256(snapshot: PreparedEmbeddingSnapshot) -> str:
    return _embedding_snapshot_sha256_from_parts(
        snapshot.identity,
        snapshot.dimension,
        snapshot.example_ids,
        snapshot.prompt_sha256s,
        len(snapshot.payload),
        (snapshot.payload,),
    )


def _embedding_snapshot_sha256_from_parts(
    identity: EmbeddingIdentity,
    dimension: int,
    example_ids: tuple[str, ...],
    prompt_sha256s: tuple[str, ...],
    payload_byte_count: int,
    payload_parts: Iterable[bytes],
) -> str:
    writer = _HashWriter(PREPARED_EMBEDDING_SNAPSHOT_ALGORITHM_ID)
    _write_embedding_identity(writer, identity)
    writer.integer("dimension", dimension)
    writer.integer("row_count", len(example_ids))
    for example_id, prompt_digest in zip(
        example_ids,
        prompt_sha256s,
        strict=True,
    ):
        writer.text("example_id", example_id)
        writer.text("prompt_sha256", prompt_digest)
    writer.token_parts("payload.f64le", payload_byte_count, payload_parts)
    return writer.hexdigest()


def build_prepared_embedding_snapshot(
    rows: tuple[PreparedEmbeddingInput, ...],
    identity: EmbeddingIdentity,
    *,
    dimension: int,
) -> PreparedEmbeddingSnapshot:
    """Canonicalize already-computed embeddings without executing a provider."""

    if type(rows) is not tuple:
        raise TypeError("embedding rows must be an exact tuple")
    if not rows or len(rows) > MAX_PREPARED_EXAMPLES:
        raise ValueError("embedding row count is outside the reviewed limit")
    if type(identity) is not EmbeddingIdentity:
        raise TypeError("embedding identity must be an exact EmbeddingIdentity")
    dimension = _exact_positive_int(dimension, "embedding dimension")
    if dimension > MAX_PREPARED_FEATURES - _UNIVERSAL_SURFACE_DIMENSION:
        raise ValueError("embedding dimension exceeds the prepared feature limit")
    payload_bytes = len(rows) * dimension * _F64_BYTES
    if payload_bytes > MAX_PREPARED_REFERENCE_NUMERIC_BYTES:
        raise ValueError("embedding payload exceeds the reference byte limit")
    if any(type(row) is not PreparedEmbeddingInput for row in rows):
        raise TypeError("embedding rows must contain exact PreparedEmbeddingInput values")
    if any(len(row.values) != dimension for row in rows):
        raise ValueError("every embedding row must match the declared dimension")
    _row_key_text_bytes(
        tuple(row.example_id for row in rows),
        tuple(row.prompt_sha256 for row in rows),
        require_canonical_order=False,
    )
    ordered = tuple(sorted(rows, key=lambda row: row.example_id))
    if any(left.example_id >= right.example_id for left, right in pairwise(ordered)):
        raise ValueError("embedding rows must have unique example IDs")
    payload = bytearray(payload_bytes)
    for row_index, row in enumerate(ordered):
        struct.pack_into(
            f"<{dimension}d",
            payload,
            row_index * dimension * _F64_BYTES,
            *row.values,
        )
    return PreparedEmbeddingSnapshot(
        identity=identity,
        dimension=dimension,
        example_ids=tuple(row.example_id for row in ordered),
        prompt_sha256s=tuple(row.prompt_sha256 for row in ordered),
        payload=bytes(payload),
    )


@dataclass(frozen=True, slots=True)
class PreparedFeatureStore:
    """Canonical fit-relevant rows in compact immutable little-endian payloads."""

    plan: PreparedNestedLodoPlan
    model_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]
    domain_indices: tuple[int, ...]
    embedding_identity: EmbeddingIdentity | None
    embedding_dimension: int
    embedding_snapshot_sha256: str | None
    source_fit_sha256: str
    feature_payload: bytes = field(repr=False)
    target_payload: bytes = field(repr=False)
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_FEATURE_STORE_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("plan must be an exact PreparedNestedLodoPlan")
        if self.plan.algorithm_id != PREPARED_GRAPH_ALGORITHM_ID:
            raise ValueError("plan algorithm does not match the prepared store")
        row_count = self.plan.work.example_count
        dimension = _exact_nonnegative_int(self.embedding_dimension, "embedding dimension")
        if self.plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + dimension:
            raise ValueError("embedding dimension does not match the prepared feature count")
        if type(self.feature_payload) is not bytes or type(self.target_payload) is not bytes:
            raise TypeError("prepared numeric payloads must be immutable bytes")
        feature_bytes = row_count * self.plan.feature_count * _F64_BYTES
        target_bytes = row_count * self.plan.target_count * _F64_BYTES
        if feature_bytes + target_bytes > MAX_PREPARED_REFERENCE_NUMERIC_BYTES:
            raise ValueError("prepared store exceeds the reference numeric-byte limit")
        if len(self.feature_payload) != feature_bytes or len(self.target_payload) != target_bytes:
            raise ValueError("prepared numeric payload has the wrong exact byte length")
        if type(self.model_ids) is not tuple:
            raise TypeError("model_ids must be an exact tuple")
        if len(self.model_ids) != self.plan.target_count:
            raise ValueError("model catalogue does not match the prepared target count")
        for model_id in self.model_ids:
            _bounded_text(model_id, "model_id", max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES)
        if self.model_ids != tuple(sorted(self.model_ids)) or len(set(self.model_ids)) != len(
            self.model_ids
        ):
            raise ValueError("model IDs must be sorted and unique")
        for name, values in (
            ("example_ids", self.example_ids),
            ("prompt_sha256s", self.prompt_sha256s),
            ("domain_indices", self.domain_indices),
        ):
            if type(values) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            if len(values) != row_count:
                raise ValueError(f"{name} has the wrong exact row count")
        _row_key_text_bytes(
            self.example_ids,
            self.prompt_sha256s,
            require_canonical_order=True,
        )
        domain_counts = [0] * len(self.plan.domains)
        for domain_index in self.domain_indices:
            index = _exact_nonnegative_int(domain_index, "domain index")
            if index >= len(domain_counts):
                raise ValueError("domain index is outside the prepared catalogue")
            domain_counts[index] += 1
        if tuple(domain_counts) != self.plan.domain_example_counts:
            raise ValueError("row domains do not match the prepared plan counts")
        if dimension == 0:
            if self.embedding_identity is not None or self.embedding_snapshot_sha256 is not None:
                raise ValueError("surface-only stores cannot carry embedding provenance")
        else:
            if type(self.embedding_identity) is not EmbeddingIdentity:
                raise TypeError("embedded stores require an exact EmbeddingIdentity")
            _sha256_hex(self.embedding_snapshot_sha256, "embedding_snapshot_sha256")
        _sha256_hex(self.source_fit_sha256, "source_fit_sha256")
        self._validate_numeric_payloads()
        self._validate_embedding_snapshot_identity()
        object.__setattr__(self, "sha256", _feature_store_sha256(self))

    def _validate_numeric_payloads(self) -> None:
        dimension = self.plan.feature_count
        for flat_index, unpacked in enumerate(struct.iter_unpack("<d", self.feature_payload)):
            value = unpacked[0]
            if not math.isfinite(value):
                raise ValueError("feature payload must contain only finite binary64 values")
            if value == 0.0 and math.copysign(1.0, value) < 0:
                raise ValueError("feature payload must use canonical positive zero")
            column = flat_index % dimension
            if column < _CONTINUOUS_COUNT and value < 0:
                raise ValueError("raw continuous features must be non-negative")
            if _CONTINUOUS_COUNT <= column < _UNIVERSAL_SURFACE_DIMENSION and value not in (
                0.0,
                1.0,
            ):
                raise ValueError("binary and universal-tag features must be zero or one")
        for (value,) in struct.iter_unpack("<d", self.target_payload):
            if not math.isfinite(value):
                raise ValueError("target payload must contain only finite binary64 values")
            if value == 0.0 and math.copysign(1.0, value) < 0:
                raise ValueError("target payload must use canonical positive zero")

    def _validate_embedding_snapshot_identity(self) -> None:
        if self.embedding_dimension == 0:
            return
        identity = self.embedding_identity
        if identity is None:  # guarded above; keeps the type refinement explicit
            raise ValueError("embedded stores require an embedding identity")
        row_stride = self.plan.feature_count * _F64_BYTES
        embedding_offset = _UNIVERSAL_SURFACE_DIMENSION * _F64_BYTES
        embedding_bytes = self.embedding_dimension * _F64_BYTES
        digest = _embedding_snapshot_sha256_from_parts(
            identity,
            self.embedding_dimension,
            self.example_ids,
            self.prompt_sha256s,
            len(self.example_ids) * embedding_bytes,
            (
                self.feature_payload[
                    row_index * row_stride + embedding_offset : row_index * row_stride
                    + embedding_offset
                    + embedding_bytes
                ]
                for row_index in range(len(self.example_ids))
            ),
        )
        if digest != self.embedding_snapshot_sha256:
            raise ValueError("embedding feature columns do not match their snapshot SHA-256")

    def feature_row(self, row_index: int) -> tuple[float, ...]:
        """Return one private-copy row for reference tests and statistics."""

        index = _bounded_row_index(row_index, self.plan.work.example_count)
        return struct.unpack_from(
            f"<{self.plan.feature_count}d",
            self.feature_payload,
            index * self.plan.feature_count * _F64_BYTES,
        )

    def target_row(self, row_index: int) -> tuple[float, ...]:
        """Return one private-copy target row in sorted model order."""

        index = _bounded_row_index(row_index, self.plan.work.example_count)
        return struct.unpack_from(
            f"<{self.plan.target_count}d",
            self.target_payload,
            index * self.plan.target_count * _F64_BYTES,
        )


def _bounded_row_index(value: object, row_count: int) -> int:
    index = _exact_nonnegative_int(value, "row index")
    if index >= row_count:
        raise IndexError("row index is outside the prepared store")
    return index


def _feature_store_sha256(store: PreparedFeatureStore) -> str:
    writer = _HashWriter(PREPARED_FEATURE_STORE_ALGORITHM_ID)
    _write_plan_identity(writer, store.plan)
    writer.text("surface.algorithm_id", SURFACE_FEATURE_ALGORITHM_ID)
    writer.text("source_fit_sha256", store.source_fit_sha256)
    writer.integer("universal_tag_count", len(SURFACE_DOMAIN_TAG_CATALOGUE))
    for tag in SURFACE_DOMAIN_TAG_CATALOGUE:
        writer.text("universal_tag", tag)
    writer.integer("model_count", len(store.model_ids))
    for model_id in store.model_ids:
        writer.text("model_id", model_id)
    writer.integer("embedding_dimension", store.embedding_dimension)
    if store.embedding_identity is not None:
        _write_embedding_identity(writer, store.embedding_identity)
        writer.text("embedding.snapshot_sha256", store.embedding_snapshot_sha256 or "")
    writer.integer("row_count", len(store.example_ids))
    for example_id, prompt_digest, domain_index in zip(
        store.example_ids,
        store.prompt_sha256s,
        store.domain_indices,
        strict=True,
    ):
        writer.text("example_id", example_id)
        writer.text("prompt_sha256", prompt_digest)
        writer.integer("domain_index", domain_index)
    writer.token("features.f64le", store.feature_payload)
    writer.token("targets.f64le", store.target_payload)
    return writer.hexdigest()


def _validated_examples_for_store(
    examples: object,
    plan: PreparedNestedLodoPlan,
) -> tuple[tuple[EvaluationExample, ...], tuple[str, ...], tuple[str, ...]]:
    if type(examples) is not tuple:
        raise TypeError("prepared examples must be an exact tuple")
    if len(examples) != plan.work.example_count:
        raise ValueError("prepared examples do not match the plan row count")
    if any(type(example) is not EvaluationExample for example in examples):
        raise TypeError("prepared examples must contain exact EvaluationExample values")
    total_text_bytes = 0
    for example in examples:
        example_id = _bounded_text(
            example.example_id,
            "example_id",
            max_bytes=MAX_PREPARED_ROW_ID_UTF8_BYTES,
        )
        prompt = _bounded_text(
            example.prompt,
            "prompt",
            max_bytes=MAX_PREPARED_REFERENCE_PROMPT_UTF8_BYTES,
        )
        domain = _bounded_text(
            example.domain,
            "domain",
            max_bytes=MAX_PREPARED_DOMAIN_UTF8_BYTES,
        )
        total_text_bytes += len(example_id.encode("utf-8")) + len(prompt.encode("utf-8"))
        total_text_bytes += len(domain.encode("utf-8"))
        if total_text_bytes > MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES:
            raise ValueError("prepared source text exceeds the reference byte limit")
        if type(example.candidate_models) is not tuple:
            raise TypeError("candidate models must be an exact tuple")
        if type(example.outcomes) is not tuple:
            raise TypeError("outcomes must be an exact tuple")
        if (
            len(example.candidate_models) != plan.target_count
            or len(example.outcomes) != plan.target_count
        ):
            raise ValueError("row model/outcome counts do not match the prepared target count")
        if any(type(model) is not ModelSpec for model in example.candidate_models):
            raise TypeError("candidate models must contain exact ModelSpec values")
        if any(type(outcome) is not CandidateOutcome for outcome in example.outcomes):
            raise TypeError("outcomes must contain exact CandidateOutcome values")
        for model in example.candidate_models:
            model_id = _bounded_text(
                model.model_id,
                "model_id",
                max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
            )
            total_text_bytes += len(model_id.encode("utf-8"))
        for outcome in example.outcomes:
            model_id = _bounded_text(
                outcome.model_id,
                "outcome model_id",
                max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
            )
            total_text_bytes += len(model_id.encode("utf-8"))
            _finite_f64(outcome.quality, "target quality")
        if total_text_bytes > MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES:
            raise ValueError("prepared source text exceeds the reference byte limit")

    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    if any(left.example_id >= right.example_id for left, right in pairwise(ordered)):
        raise ValueError("prepared examples must have unique example IDs")

    domain_counts = [0] * len(plan.domains)
    domain_to_index = {domain: index for index, domain in enumerate(plan.domains)}
    model_ids: tuple[str, ...] | None = None
    prompt_digests: list[str] = []
    for example in ordered:
        prompt = example.prompt
        domain = example.domain
        try:
            domain_counts[domain_to_index[domain]] += 1
        except KeyError as error:
            raise ValueError("prepared example domain is absent from the plan") from error
        current_models = tuple(sorted(model.model_id for model in example.candidate_models))
        if len(current_models) != len(set(current_models)):
            raise ValueError("candidate model IDs must be unique")
        if model_ids is None:
            model_ids = current_models
        elif current_models != model_ids:
            raise ValueError("every row must contain the same model catalogue")
        outcomes = {outcome.model_id: outcome for outcome in example.outcomes}
        if set(outcomes) != set(current_models):
            raise ValueError("outcomes must match the stable model catalogue")
        for model_id in current_models:
            _finite_f64(outcomes[model_id].quality, "target quality")
        prompt_digests.append(_prompt_sha256(prompt))
    if tuple(domain_counts) != plan.domain_example_counts:
        raise ValueError("prepared example domain counts do not match the plan")
    if model_ids is None or len(model_ids) != plan.target_count:
        raise ValueError("model catalogue does not match the plan target count")
    return ordered, model_ids, tuple(prompt_digests)


def _source_fit_sha256_from_validated(
    ordered: tuple[EvaluationExample, ...],
    model_ids: tuple[str, ...],
    plan: PreparedNestedLodoPlan,
) -> str:
    writer = _HashWriter(_FIT_SOURCE_ALGORITHM_ID)
    _write_plan_identity(writer, plan)
    writer.integer("model_count", len(model_ids))
    for model_id in model_ids:
        writer.text("model_id", model_id)
    writer.integer("row_count", len(ordered))
    for example in ordered:
        writer.text("example_id", example.example_id)
        writer.text("prompt", example.prompt)
        writer.text("domain", example.domain)
        outcomes = {outcome.model_id: outcome for outcome in example.outcomes}
        for model_id in model_ids:
            writer.token(
                "quality.f64le",
                struct.pack("<d", _finite_f64(outcomes[model_id].quality, "target quality")),
            )
    return writer.hexdigest()


def prepared_fit_source_sha256(
    examples: tuple[EvaluationExample, ...],
    plan: PreparedNestedLodoPlan,
) -> str:
    """Hash only canonical predictor-fit source fields, excluding costs and outputs."""

    if type(plan) is not PreparedNestedLodoPlan:
        raise TypeError("plan must be an exact PreparedNestedLodoPlan")
    ordered, model_ids, _ = _validated_examples_for_store(examples, plan)
    return _source_fit_sha256_from_validated(ordered, model_ids, plan)


def build_prepared_feature_store(
    examples: tuple[EvaluationExample, ...],
    plan: PreparedNestedLodoPlan,
    *,
    embedding_snapshot: PreparedEmbeddingSnapshot | None = None,
    expected_embedding_sha256: str | None = None,
    expected_source_fit_sha256: str,
) -> PreparedFeatureStore:
    """Snapshot fit-relevant rows after all cheap alignment/resource checks pass."""

    if type(plan) is not PreparedNestedLodoPlan:
        raise TypeError("plan must be an exact PreparedNestedLodoPlan")
    expected_source_digest = _sha256_hex(
        expected_source_fit_sha256,
        "expected_source_fit_sha256",
    )
    if embedding_snapshot is None:
        if expected_embedding_sha256 is not None:
            raise ValueError("surface-only stores cannot accept an embedding digest")
        embedding_dimension = 0
        embedding_identity = None
        embedding_digest = None
        embedding_payload = b""
    else:
        if type(embedding_snapshot) is not PreparedEmbeddingSnapshot:
            raise TypeError("embedding_snapshot must be an exact PreparedEmbeddingSnapshot")
        expected_digest = _sha256_hex(
            expected_embedding_sha256,
            "expected_embedding_sha256",
        )
        if embedding_snapshot.sha256 != expected_digest:
            raise ValueError("embedding snapshot does not match the caller-expected SHA-256")
        embedding_dimension = embedding_snapshot.dimension
        embedding_identity = embedding_snapshot.identity
        embedding_digest = embedding_snapshot.sha256
        embedding_payload = embedding_snapshot.payload
    if plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + embedding_dimension:
        raise ValueError("plan feature count does not match the universal raw layout")
    numeric_bytes = plan.work.example_count * (
        plan.feature_count + plan.target_count
    ) * _F64_BYTES + len(embedding_payload)
    if numeric_bytes > MAX_PREPARED_REFERENCE_NUMERIC_BYTES:
        raise ValueError("prepared store exceeds the reference numeric-byte limit")

    ordered, model_ids, prompt_digests = _validated_examples_for_store(examples, plan)
    source_fit_digest = _source_fit_sha256_from_validated(ordered, model_ids, plan)
    if source_fit_digest != expected_source_digest:
        raise ValueError("prepared source does not match the caller-expected fit SHA-256")
    row_ids = tuple(example.example_id for example in ordered)
    if embedding_snapshot is not None:
        if embedding_snapshot.example_ids != row_ids:
            raise ValueError("embedding snapshot example IDs do not match prepared rows")
        if embedding_snapshot.prompt_sha256s != prompt_digests:
            raise ValueError("embedding snapshot prompt digests do not match prepared rows")

    features = bytearray(plan.work.example_count * plan.feature_count * _F64_BYTES)
    targets = bytearray(plan.work.example_count * plan.target_count * _F64_BYTES)
    domain_to_index = {domain: index for index, domain in enumerate(plan.domains)}
    domain_indices: list[int] = []
    for row_index, example in enumerate(ordered):
        surface = extract_surface_features(example.prompt)
        prompt_tags = set(surface.domain_tags)
        if len(prompt_tags) != len(surface.domain_tags) or not prompt_tags.issubset(
            SURFACE_DOMAIN_TAG_CATALOGUE
        ):
            raise ValueError("surface extractor emitted tags outside its fixed catalogue")
        embedding_offset = row_index * embedding_dimension * _F64_BYTES
        embedding_values = (
            ()
            if embedding_dimension == 0
            else struct.unpack_from(
                f"<{embedding_dimension}d",
                embedding_payload,
                embedding_offset,
            )
        )
        row = (
            math.log1p(surface.character_count),
            math.log1p(surface.word_count),
            math.log1p(surface.line_count),
            float(surface.has_code),
            float(surface.has_math),
            *(float(tag in prompt_tags) for tag in SURFACE_DOMAIN_TAG_CATALOGUE),
            *embedding_values,
        )
        struct.pack_into(
            f"<{plan.feature_count}d",
            features,
            row_index * plan.feature_count * _F64_BYTES,
            *row,
        )
        outcome_by_model = {outcome.model_id: outcome for outcome in example.outcomes}
        target_row = tuple(
            _finite_f64(outcome_by_model[model_id].quality, "target quality")
            for model_id in model_ids
        )
        struct.pack_into(
            f"<{plan.target_count}d",
            targets,
            row_index * plan.target_count * _F64_BYTES,
            *target_row,
        )
        domain_indices.append(domain_to_index[example.domain])
    return PreparedFeatureStore(
        plan=plan,
        model_ids=model_ids,
        example_ids=row_ids,
        prompt_sha256s=prompt_digests,
        domain_indices=tuple(domain_indices),
        embedding_identity=embedding_identity,
        embedding_dimension=embedding_dimension,
        embedding_snapshot_sha256=embedding_digest,
        source_fit_sha256=source_fit_digest,
        feature_payload=bytes(features),
        target_payload=bytes(targets),
    )


def _packed_upper_length(dimension: int) -> int:
    return dimension * (dimension + 1) // 2


def _packed_upper_index(dimension: int, row: int, column: int) -> int:
    if row > column:
        row, column = column, row
    return row * dimension - row * (row - 1) // 2 + (column - row)


@dataclass(frozen=True, slots=True)
class PreparedDomainStatistics:
    """Centered Welford moments for one canonical domain in universal coordinates."""

    domain_index: int
    row_count: int
    feature_count: int
    target_count: int
    active_tag_mask: int
    content_sha256: str
    feature_means: tuple[float, ...]
    target_means: tuple[float, ...]
    centered_xx_packed: tuple[float, ...]
    centered_xy: tuple[float, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_STATISTICS_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        domain_index = _exact_nonnegative_int(self.domain_index, "statistics domain index")
        if domain_index >= MAX_PREPARED_DOMAINS:
            raise ValueError("statistics domain index exceeds the reviewed domain limit")
        _exact_positive_int(self.row_count, "statistics row_count")
        if self.row_count > MAX_PREPARED_EXAMPLES:
            raise ValueError("statistics row_count exceeds the reviewed example limit")
        feature_count = _exact_positive_int(self.feature_count, "statistics feature_count")
        target_count = _exact_positive_int(self.target_count, "statistics target_count")
        if not _UNIVERSAL_SURFACE_DIMENSION <= feature_count <= MAX_PREPARED_FEATURES:
            raise ValueError("statistics feature_count is outside the universal prepared layout")
        if target_count > MAX_PREPARED_TARGETS:
            raise ValueError("statistics dimensions exceed the prepared limits")
        scalar_count = (
            feature_count
            + target_count
            + _packed_upper_length(feature_count)
            + feature_count * target_count
        )
        if scalar_count > MAX_PREPARED_REFERENCE_STATISTIC_SCALARS:
            raise ValueError("domain statistics exceed the reference scalar limit")
        mask = _exact_nonnegative_int(self.active_tag_mask, "active_tag_mask")
        if mask >= 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE):
            raise ValueError("active_tag_mask contains an unknown tag bit")
        _sha256_hex(self.content_sha256, "statistics content_sha256")
        feature_means = _finite_tuple(
            self.feature_means,
            "feature means",
            expected_length=feature_count,
        )
        _validate_universal_feature_means(feature_means, mask, "domain statistics")
        target_means = _finite_tuple(
            self.target_means,
            "target means",
            expected_length=target_count,
        )
        centered_xx = _finite_tuple(
            self.centered_xx_packed,
            "centered_xx_packed",
            expected_length=_packed_upper_length(feature_count),
        )
        centered_xy = _finite_tuple(
            self.centered_xy,
            "centered_xy",
            expected_length=feature_count * target_count,
        )
        for index in range(feature_count):
            if centered_xx[_packed_upper_index(feature_count, index, index)] < 0:
                raise ValueError("centered feature diagonal must be non-negative")
        object.__setattr__(self, "feature_means", feature_means)
        object.__setattr__(self, "target_means", target_means)
        object.__setattr__(self, "centered_xx_packed", centered_xx)
        object.__setattr__(self, "centered_xy", centered_xy)
        object.__setattr__(self, "sha256", _domain_statistics_sha256(self))


def _domain_statistics_sha256(statistics: PreparedDomainStatistics) -> str:
    writer = _HashWriter(PREPARED_STATISTICS_ALGORITHM_ID)
    writer.integer("domain_index", statistics.domain_index)
    writer.integer("row_count", statistics.row_count)
    writer.integer("feature_count", statistics.feature_count)
    writer.integer("target_count", statistics.target_count)
    writer.integer("active_tag_mask", statistics.active_tag_mask)
    writer.text("content_sha256", statistics.content_sha256)
    writer.floats("feature_means", statistics.feature_means)
    writer.floats("target_means", statistics.target_means)
    writer.floats("centered_xx_packed", statistics.centered_xx_packed)
    writer.floats("centered_xy", statistics.centered_xy)
    return writer.hexdigest()


@dataclass(slots=True)
class _MomentAccumulator:
    feature_count: int
    target_count: int
    count: int = 0
    active_tag_mask: int = 0
    feature_means: list[float] = field(init=False)
    target_means: list[float] = field(init=False)
    centered_xx: list[float] = field(init=False)
    centered_xy: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.feature_means = [0.0] * self.feature_count
        self.target_means = [0.0] * self.target_count
        self.centered_xx = [0.0] * _packed_upper_length(self.feature_count)
        self.centered_xy = [0.0] * (self.feature_count * self.target_count)

    def update(self, features: tuple[float, ...], targets: tuple[float, ...]) -> None:
        next_count = self.count + 1
        delta_x = [value - mean for value, mean in zip(features, self.feature_means, strict=True)]
        delta_y = [value - mean for value, mean in zip(targets, self.target_means, strict=True)]
        for index in range(self.feature_count):
            self.feature_means[index] += delta_x[index] / next_count
        for index in range(self.target_count):
            self.target_means[index] += delta_y[index] / next_count
        for row in range(self.feature_count):
            for column in range(row, self.feature_count):
                position = _packed_upper_index(self.feature_count, row, column)
                self.centered_xx[position] += delta_x[row] * (
                    features[column] - self.feature_means[column]
                )
            base = row * self.target_count
            for target in range(self.target_count):
                self.centered_xy[base + target] += delta_x[row] * (
                    targets[target] - self.target_means[target]
                )
        for tag_index in range(len(SURFACE_DOMAIN_TAG_CATALOGUE)):
            if features[_TAG_OFFSET + tag_index] == 1.0:
                self.active_tag_mask |= 1 << tag_index
        self.count = next_count


def _domain_content_writers(store: PreparedFeatureStore) -> tuple[_HashWriter, ...]:
    writers = []
    for domain_index, domain in enumerate(store.plan.domains):
        writer = _HashWriter(_DOMAIN_CONTENT_ALGORITHM_ID)
        writer.text("surface.algorithm_id", SURFACE_FEATURE_ALGORITHM_ID)
        writer.text("graph.algorithm_id", store.plan.algorithm_id)
        writer.integer("domain_index", domain_index)
        writer.text("domain", domain)
        writer.integer("feature_count", store.plan.feature_count)
        writer.integer("target_count", store.plan.target_count)
        writer.integer("model_count", len(store.model_ids))
        for model_id in store.model_ids:
            writer.text("model_id", model_id)
        writer.integer("embedding_dimension", store.embedding_dimension)
        if store.embedding_identity is not None:
            _write_embedding_identity(writer, store.embedding_identity)
        writers.append(writer)
    return tuple(writers)


@dataclass(frozen=True, slots=True)
class PreparedDomainStatisticsBundle:
    """All per-domain moments bound to one global store content digest."""

    plan: PreparedNestedLodoPlan
    store_sha256: str
    model_ids: tuple[str, ...]
    embedding_identity: EmbeddingIdentity | None
    embedding_dimension: int
    domain_statistics: tuple[PreparedDomainStatistics, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_STATISTICS_BUNDLE_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("statistics plan must be an exact PreparedNestedLodoPlan")
        _sha256_hex(self.store_sha256, "statistics store_sha256")
        if type(self.model_ids) is not tuple:
            raise TypeError("statistics model_ids must be an exact tuple")
        if len(self.model_ids) != self.plan.target_count:
            raise ValueError("statistics model catalogue does not match the plan")
        for model_id in self.model_ids:
            _bounded_text(
                model_id,
                "statistics model_id",
                max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
            )
        if self.model_ids != tuple(sorted(set(self.model_ids))):
            raise ValueError("statistics model_ids must be a sorted unique exact tuple")
        dimension = _exact_nonnegative_int(self.embedding_dimension, "embedding dimension")
        if self.plan.feature_count != _UNIVERSAL_SURFACE_DIMENSION + dimension:
            raise ValueError("statistics embedding dimension does not match the plan")
        per_domain_scalars = (
            self.plan.feature_count
            + self.plan.target_count
            + _packed_upper_length(self.plan.feature_count)
            + self.plan.feature_count * self.plan.target_count
        )
        if len(self.plan.domains) * per_domain_scalars > MAX_PREPARED_REFERENCE_STATISTIC_SCALARS:
            raise ValueError("statistics bundle exceeds the reference scalar limit")
        if (dimension == 0) != (self.embedding_identity is None):
            raise ValueError("statistics embedding identity and dimension disagree")
        if (
            self.embedding_identity is not None
            and type(self.embedding_identity) is not EmbeddingIdentity
        ):
            raise TypeError("statistics embedding identity must be exact")
        if type(self.domain_statistics) is not tuple or len(self.domain_statistics) != len(
            self.plan.domains
        ):
            raise ValueError("statistics domains do not match the prepared plan")
        for domain_index, statistics in enumerate(self.domain_statistics):
            if type(statistics) is not PreparedDomainStatistics:
                raise TypeError("domain_statistics must contain exact statistics values")
            if (
                statistics.domain_index != domain_index
                or statistics.row_count != self.plan.domain_example_counts[domain_index]
                or statistics.feature_count != self.plan.feature_count
                or statistics.target_count != self.plan.target_count
            ):
                raise ValueError("domain statistics do not match the prepared plan")
        object.__setattr__(self, "sha256", _statistics_bundle_sha256(self))


def _statistics_bundle_sha256(bundle: PreparedDomainStatisticsBundle) -> str:
    writer = _HashWriter(PREPARED_STATISTICS_BUNDLE_ALGORITHM_ID)
    _write_plan_identity(writer, bundle.plan)
    writer.text("store_sha256", bundle.store_sha256)
    writer.integer("model_count", len(bundle.model_ids))
    for model_id in bundle.model_ids:
        writer.text("model_id", model_id)
    writer.integer("embedding_dimension", bundle.embedding_dimension)
    if bundle.embedding_identity is not None:
        _write_embedding_identity(writer, bundle.embedding_identity)
    writer.integer("domain_count", len(bundle.domain_statistics))
    for statistics in bundle.domain_statistics:
        writer.text("domain_statistics_sha256", statistics.sha256)
    return writer.hexdigest()


def _preflight_reference_statistics(store: PreparedFeatureStore) -> None:
    dimension = store.plan.feature_count
    targets = store.plan.target_count
    per_domain = dimension + targets + _packed_upper_length(dimension) + dimension * targets
    scalar_count = len(store.plan.domains) * per_domain
    if scalar_count > MAX_PREPARED_REFERENCE_STATISTIC_SCALARS:
        raise ValueError("prepared reference statistics exceed the scalar limit")
    # Keep the same dominant-work unit as PreparedNestedLodoWorkEstimate: three
    # linear passes cover mean/delta/update work before the quadratic products.
    per_row_work = 3 * (dimension + targets) + _packed_upper_length(dimension) + dimension * targets
    work = store.plan.work.example_count * per_row_work
    if work > MAX_PREPARED_REFERENCE_STATISTIC_WORK_UNITS:
        raise ValueError("prepared reference statistics exceed the numeric-work limit")


def build_prepared_domain_statistics(
    store: PreparedFeatureStore,
) -> PreparedDomainStatisticsBundle:
    """Compute one reusable Welford moment block per domain after strict preflight."""

    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    _preflight_reference_statistics(store)
    accumulators = tuple(
        _MomentAccumulator(store.plan.feature_count, store.plan.target_count)
        for _ in store.plan.domains
    )
    content_writers = _domain_content_writers(store)
    feature_stride = store.plan.feature_count * _F64_BYTES
    target_stride = store.plan.target_count * _F64_BYTES
    for row_index, domain_index in enumerate(store.domain_indices):
        features = store.feature_row(row_index)
        targets = store.target_row(row_index)
        accumulators[domain_index].update(features, targets)
        writer = content_writers[domain_index]
        writer.text("example_id", store.example_ids[row_index])
        writer.text("prompt_sha256", store.prompt_sha256s[row_index])
        writer.token(
            "features.f64le",
            store.feature_payload[row_index * feature_stride : (row_index + 1) * feature_stride],
        )
        writer.token(
            "targets.f64le",
            store.target_payload[row_index * target_stride : (row_index + 1) * target_stride],
        )
    statistics = tuple(
        PreparedDomainStatistics(
            domain_index=domain_index,
            row_count=accumulator.count,
            feature_count=store.plan.feature_count,
            target_count=store.plan.target_count,
            active_tag_mask=accumulator.active_tag_mask,
            content_sha256=content_writers[domain_index].hexdigest(),
            feature_means=tuple(accumulator.feature_means),
            target_means=tuple(accumulator.target_means),
            centered_xx_packed=tuple(accumulator.centered_xx),
            centered_xy=tuple(accumulator.centered_xy),
        )
        for domain_index, accumulator in enumerate(accumulators)
    )
    return PreparedDomainStatisticsBundle(
        plan=store.plan,
        store_sha256=store.sha256,
        model_ids=store.model_ids,
        embedding_identity=store.embedding_identity,
        embedding_dimension=store.embedding_dimension,
        domain_statistics=statistics,
    )


@dataclass(frozen=True, slots=True)
class PreparedSubsetStatistics:
    """Canonical Chan combination for one graph training subset, before any solve."""

    plan: PreparedNestedLodoPlan
    subset_index: int
    domain_indices: tuple[int, ...]
    row_count: int
    model_ids: tuple[str, ...]
    feature_schema: PromptFeatureSchema
    active_tag_mask: int
    included_content_sha256: str
    feature_means: tuple[float, ...]
    target_means: tuple[float, ...]
    centered_xx_packed: tuple[float, ...]
    centered_xy: tuple[float, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=PREPARED_SUBSET_STATISTICS_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.plan) is not PreparedNestedLodoPlan:
            raise TypeError("subset plan must be an exact PreparedNestedLodoPlan")
        subset_index = _exact_nonnegative_int(self.subset_index, "subset_index")
        if subset_index >= len(self.plan.training_subsets):
            raise ValueError("subset_index is outside the prepared plan")
        expected_subset = self.plan.training_subsets[subset_index]
        if type(self.domain_indices) is not tuple or not self.domain_indices:
            raise ValueError("subset domain_indices must be a non-empty exact tuple")
        if len(self.domain_indices) > MAX_PREPARED_DOMAINS:
            raise ValueError("subset domain_indices exceed the reviewed domain limit")
        if any(
            type(domain_index) is not int or not 0 <= domain_index < MAX_PREPARED_DOMAINS
            for domain_index in self.domain_indices
        ):
            raise ValueError("subset domain_indices must contain bounded exact integers")
        if self.domain_indices != tuple(sorted(set(self.domain_indices))):
            raise ValueError("subset domain_indices must be sorted and unique")
        if self.domain_indices != expected_subset.domain_indices:
            raise ValueError("subset domain_indices do not match the prepared plan")
        _exact_positive_int(self.row_count, "subset row_count")
        if self.row_count > MAX_PREPARED_EXAMPLES:
            raise ValueError("subset row_count exceeds the reviewed example limit")
        if self.row_count != expected_subset.row_count:
            raise ValueError("subset row_count does not match the prepared plan")
        if type(self.model_ids) is not tuple:
            raise TypeError("subset model_ids must be an exact tuple")
        if len(self.model_ids) != self.plan.target_count:
            raise ValueError("subset model catalogue does not match the prepared plan")
        for model_id in self.model_ids:
            _bounded_text(
                model_id,
                "subset model_id",
                max_bytes=MAX_PREPARED_MODEL_ID_UTF8_BYTES,
            )
        if self.model_ids != tuple(sorted(set(self.model_ids))):
            raise ValueError("subset model_ids must be a sorted unique exact tuple")
        if type(self.feature_schema) is not PromptFeatureSchema:
            raise TypeError("feature_schema must be an exact PromptFeatureSchema")
        if any(
            value == 0.0 and math.copysign(1.0, value) < 0
            for value in self.feature_schema.continuous_means
        ):
            raise ValueError("feature schema means must use canonical positive zero")
        feature_count = _UNIVERSAL_SURFACE_DIMENSION + self.feature_schema.embedding_dimension
        if feature_count != self.plan.feature_count:
            raise ValueError("subset feature schema does not match the prepared plan")
        target_count = len(self.model_ids)
        scalar_count = (
            feature_count
            + target_count
            + _packed_upper_length(feature_count)
            + feature_count * target_count
        )
        if scalar_count > MAX_PREPARED_REFERENCE_STATISTIC_SCALARS:
            raise ValueError("subset statistics exceed the reference scalar limit")
        mask = _exact_nonnegative_int(self.active_tag_mask, "active_tag_mask")
        if mask >= 1 << len(SURFACE_DOMAIN_TAG_CATALOGUE):
            raise ValueError("active_tag_mask contains an unknown tag bit")
        expected_tags = tuple(
            tag for index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE) if mask & (1 << index)
        )
        if self.feature_schema.domain_tags != expected_tags:
            raise ValueError("feature schema tags do not match the included-domain mask")
        _sha256_hex(self.included_content_sha256, "included_content_sha256")
        feature_means = _finite_tuple(
            self.feature_means,
            "subset feature means",
            expected_length=feature_count,
        )
        _validate_universal_feature_means(feature_means, mask, "subset statistics")
        target_means = _finite_tuple(
            self.target_means,
            "subset target means",
            expected_length=target_count,
        )
        centered_xx = _finite_tuple(
            self.centered_xx_packed,
            "subset centered_xx_packed",
            expected_length=_packed_upper_length(feature_count),
        )
        centered_xy = _finite_tuple(
            self.centered_xy,
            "subset centered_xy",
            expected_length=feature_count * target_count,
        )
        for index in range(feature_count):
            diagonal = centered_xx[_packed_upper_index(feature_count, index, index)]
            if diagonal < 0:
                raise ValueError("subset centered feature diagonal must be non-negative")
        scales: list[float] = []
        for index in range(_CONTINUOUS_COUNT):
            diagonal = centered_xx[_packed_upper_index(feature_count, index, index)]
            scale = math.sqrt(diagonal / self.row_count)
            scales.append(scale if scale > 0 else 1.0)
        if self.feature_schema.continuous_means != feature_means[:_CONTINUOUS_COUNT] or (
            self.feature_schema.continuous_scales != tuple(scales)
        ):
            raise ValueError("feature schema scaling does not match subset moments")
        object.__setattr__(self, "feature_means", feature_means)
        object.__setattr__(self, "target_means", target_means)
        object.__setattr__(self, "centered_xx_packed", centered_xx)
        object.__setattr__(self, "centered_xy", centered_xy)
        object.__setattr__(self, "sha256", _subset_statistics_sha256(self))

    @property
    def active_feature_indices(self) -> tuple[int, ...]:
        """Map the fitted dynamic schema back to universal raw-store columns."""

        active_tags = tuple(
            _TAG_OFFSET + index
            for index in range(len(SURFACE_DOMAIN_TAG_CATALOGUE))
            if self.active_tag_mask & (1 << index)
        )
        embeddings = tuple(range(_UNIVERSAL_SURFACE_DIMENSION, len(self.feature_means)))
        return (*range(_TAG_OFFSET), *active_tags, *embeddings)


def _subset_statistics_sha256(statistics: PreparedSubsetStatistics) -> str:
    writer = _HashWriter(PREPARED_SUBSET_STATISTICS_ALGORITHM_ID)
    _write_plan_identity(writer, statistics.plan)
    writer.integer("subset_index", statistics.subset_index)
    writer.integer("row_count", statistics.row_count)
    for domain_index in statistics.domain_indices:
        writer.integer("domain_index", domain_index)
    for model_id in statistics.model_ids:
        writer.text("model_id", model_id)
    writer.integer("active_tag_mask", statistics.active_tag_mask)
    writer.text("included_content_sha256", statistics.included_content_sha256)
    writer.floats("continuous_means", statistics.feature_schema.continuous_means)
    writer.floats("continuous_scales", statistics.feature_schema.continuous_scales)
    writer.integer("feature_schema.version", statistics.feature_schema.schema_version)
    writer.integer(
        "feature_schema.embedding_dimension",
        statistics.feature_schema.embedding_dimension,
    )
    for tag in statistics.feature_schema.domain_tags:
        writer.text("feature_schema.domain_tag", tag)
    if statistics.feature_schema.embedding_identity is not None:
        _write_embedding_identity(writer, statistics.feature_schema.embedding_identity)
    writer.floats("feature_means", statistics.feature_means)
    writer.floats("target_means", statistics.target_means)
    writer.floats("centered_xx_packed", statistics.centered_xx_packed)
    writer.floats("centered_xy", statistics.centered_xy)
    return writer.hexdigest()


def _combine_domain_statistics(
    left_count: int,
    left_feature_means: list[float],
    left_target_means: list[float],
    left_xx: list[float],
    left_xy: list[float],
    right: PreparedDomainStatistics,
) -> int:
    if left_count == 0:
        left_feature_means[:] = right.feature_means
        left_target_means[:] = right.target_means
        left_xx[:] = right.centered_xx_packed
        left_xy[:] = right.centered_xy
        return right.row_count
    combined_count = left_count + right.row_count
    factor = left_count * right.row_count / combined_count
    delta_x = [
        right_mean - left_mean
        for left_mean, right_mean in zip(left_feature_means, right.feature_means, strict=True)
    ]
    delta_y = [
        right_mean - left_mean
        for left_mean, right_mean in zip(left_target_means, right.target_means, strict=True)
    ]
    for index in range(len(left_feature_means)):
        left_feature_means[index] += delta_x[index] * right.row_count / combined_count
    for index in range(len(left_target_means)):
        left_target_means[index] += delta_y[index] * right.row_count / combined_count
    dimension = len(left_feature_means)
    target_count = len(left_target_means)
    for row in range(dimension):
        for column in range(row, dimension):
            position = _packed_upper_index(dimension, row, column)
            left_xx[position] = _finite_moment_sum(
                left_xx[position],
                right.centered_xx_packed[position],
                factor * delta_x[row] * delta_x[column],
            )
        base = row * target_count
        for target in range(target_count):
            position = base + target
            left_xy[position] = _finite_moment_sum(
                left_xy[position],
                right.centered_xy[position],
                factor * delta_x[row] * delta_y[target],
            )
    return combined_count


def _finite_moment_sum(*values: float) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError("combined centered moment is not finite binary64") from error
    if not math.isfinite(result):
        raise ValueError("combined centered moment is not finite binary64")
    return 0.0 if result == 0.0 else result


def combine_prepared_subset_statistics(
    bundle: PreparedDomainStatisticsBundle,
    subset_index: int,
) -> PreparedSubsetStatistics:
    """Combine only included-domain moments and derive its training-only schema."""

    if type(bundle) is not PreparedDomainStatisticsBundle:
        raise TypeError("bundle must be an exact PreparedDomainStatisticsBundle")
    index = _exact_nonnegative_int(subset_index, "subset_index")
    if index >= len(bundle.plan.training_subsets):
        raise IndexError("subset_index is outside the prepared graph")
    subset = bundle.plan.training_subsets[index]
    dimension = bundle.plan.feature_count
    target_count = bundle.plan.target_count
    feature_means = [0.0] * dimension
    target_means = [0.0] * target_count
    centered_xx = [0.0] * _packed_upper_length(dimension)
    centered_xy = [0.0] * (dimension * target_count)
    row_count = 0
    active_tag_mask = 0
    content_writer = _HashWriter(_SUBSET_CONTENT_ALGORITHM_ID)
    for domain_index in subset.domain_indices:
        statistics = bundle.domain_statistics[domain_index]
        row_count = _combine_domain_statistics(
            row_count,
            feature_means,
            target_means,
            centered_xx,
            centered_xy,
            statistics,
        )
        active_tag_mask |= statistics.active_tag_mask
        content_writer.integer("domain_index", domain_index)
        content_writer.text("domain_content_sha256", statistics.content_sha256)
    if row_count != subset.row_count:
        raise ValueError("combined statistics row count does not match the graph subset")
    scales = []
    for feature_index in range(_CONTINUOUS_COUNT):
        diagonal = centered_xx[_packed_upper_index(dimension, feature_index, feature_index)]
        if not math.isfinite(diagonal) or diagonal < 0:
            raise ValueError("combined feature variance is not finite and non-negative")
        scale = math.sqrt(diagonal / row_count)
        scales.append(scale if scale > 0 else 1.0)
    active_tags = tuple(
        tag
        for tag_index, tag in enumerate(SURFACE_DOMAIN_TAG_CATALOGUE)
        if active_tag_mask & (1 << tag_index)
    )
    schema = PromptFeatureSchema(
        continuous_means=tuple(feature_means[:_CONTINUOUS_COUNT]),  # type: ignore[arg-type]
        continuous_scales=tuple(scales),  # type: ignore[arg-type]
        domain_tags=active_tags,
        embedding_dimension=bundle.embedding_dimension,
        embedding_identity=bundle.embedding_identity,
    )
    return PreparedSubsetStatistics(
        plan=bundle.plan,
        subset_index=index,
        domain_indices=subset.domain_indices,
        row_count=row_count,
        model_ids=bundle.model_ids,
        feature_schema=schema,
        active_tag_mask=active_tag_mask,
        included_content_sha256=content_writer.hexdigest(),
        feature_means=tuple(feature_means),
        target_means=tuple(target_means),
        centered_xx_packed=tuple(centered_xx),
        centered_xy=tuple(centered_xy),
    )


__all__ = [
    "MAX_PREPARED_REFERENCE_NUMERIC_BYTES",
    "MAX_PREPARED_REFERENCE_PROMPT_UTF8_BYTES",
    "MAX_PREPARED_REFERENCE_STATISTIC_SCALARS",
    "MAX_PREPARED_REFERENCE_STATISTIC_WORK_UNITS",
    "MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES",
    "PREPARED_EMBEDDING_SNAPSHOT_ALGORITHM_ID",
    "PREPARED_FEATURE_STORE_ALGORITHM_ID",
    "PREPARED_STATISTICS_ALGORITHM_ID",
    "PREPARED_STATISTICS_BUNDLE_ALGORITHM_ID",
    "PREPARED_SUBSET_STATISTICS_ALGORITHM_ID",
    "PreparedDomainStatistics",
    "PreparedDomainStatisticsBundle",
    "PreparedEmbeddingInput",
    "PreparedEmbeddingSnapshot",
    "PreparedFeatureStore",
    "PreparedSubsetStatistics",
    "build_prepared_domain_statistics",
    "build_prepared_embedding_snapshot",
    "build_prepared_feature_store",
    "combine_prepared_subset_statistics",
    "prepared_fit_source_sha256",
]
