# SPDX-License-Identifier: Apache-2.0
"""Authenticated file-backed storage for native prepared sessions.

The version-1 container is deliberately fixed-width and little-endian.  This
module owns persistence and authentication only; it never downloads assets or
invokes a native executable.
"""

from __future__ import annotations

import hashlib
import math
import mmap
import os
import stat
import struct
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import BinaryIO

from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.features.surface import SURFACE_DOMAIN_TAG_CATALOGUE
from tierroute.predictors.prepared_graph import (
    MAX_PREPARED_DOMAINS,
    MAX_PREPARED_EXAMPLES,
    MAX_PREPARED_FEATURES,
    MAX_PREPARED_TARGETS,
    PREPARED_GRAPH_ALGORITHM_ID,
    PreparedNestedLodoPlan,
)
from tierroute.predictors.prepared_store import PreparedFeatureStore

PREPARED_STORE_FILE_ID = "tierroute.prepared-store-file-f64le-v1"
PREPARED_STORE_GRAPH_IDENTITY_ID = "tierroute.prepared-store-file-graph-identity-v1"
PREPARED_STORE_EMBEDDING_IDENTITY_ID = "tierroute.prepared-store-file-embedding-identity-v1"
PREPARED_STORE_MODEL_CATALOGUE_ID = "tierroute.prepared-store-file-model-catalogue-v1"

STORE_MAGIC = b"TRPSTO01"
STORE_VERSION = 1
STORE_FLAGS = 0
STORE_HEADER_BYTES = 472
UNIVERSAL_SURFACE_WIDTH = 12

MAX_STORE_FILE_BYTES = 512 * 1024 * 1024
MAX_RESULT_FILE_BYTES = 128 * 1024 * 1024
MAX_MODELED_C_HEAP_BYTES = 512 * 1024 * 1024
MAX_PRIVATE_DISK_SCRATCH_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_NUMERIC_WORK_UNITS = 200_000_000_000

SESSION_REQUEST_HEADER_BYTES = 160
SESSION_RESULT_HEADER_BYTES = 448
COEFFICIENT_RECORD_HEADER_BYTES = 48
SCORE_RECORD_HEADER_BYTES = 32
MAX_ROW_ID_UTF8_BYTES = 4 * 1024

_HEADER = struct.Struct("<8sII15Q32s32s32s32s32s32s32s14Q")
_F64 = struct.Struct("<d")
_U16 = struct.Struct("<H")
_COPY_CHUNK_BYTES = 1024 * 1024
_TAG_OFFSET = 5
_TAG_MASK = (1 << len(SURFACE_DOMAIN_TAG_CATALOGUE)) - 1
_ZERO_SHA256 = b"\x00" * 32
_ASCII_WHITESPACE_BYTES = frozenset(b" \t\n\v\f\r")


class PreparedStoreFileError(ValueError):
    """A prepared-store persistence or authentication contract failed."""


@dataclass(frozen=True, slots=True)
class PreparedStoreFileReceipt:
    """Caller-pinned credentials for exactly one prepared-store file."""

    whole_file_sha256: str
    source_fit_sha256: str
    logical_store_sha256: str
    embedding_snapshot_sha256: str | None

    def __post_init__(self) -> None:
        _require_sha256(self.whole_file_sha256, "whole_file_sha256")
        _require_sha256(self.source_fit_sha256, "source_fit_sha256")
        _require_sha256(self.logical_store_sha256, "logical_store_sha256")
        if self.embedding_snapshot_sha256 is not None:
            _require_sha256(
                self.embedding_snapshot_sha256,
                "embedding_snapshot_sha256",
            )


@dataclass(frozen=True, slots=True)
class PreparedSessionEstimate:
    """Checked aggregate admission values mirrored by the C11 session."""

    active_feature_counts: tuple[int, ...]
    training_subset_count: int
    score_record_count: int
    score_row_memberships: int
    mapped_input_bytes: int
    file_backed_input_bytes: int
    modeled_c_heap_bytes: int
    private_disk_scratch_bytes: int
    coefficient_bytes: int
    score_bytes: int
    result_bytes: int
    authentication_validation_bytes_scanned: int
    output_numeric_cells_validated: int
    output_validation_work_units: int
    statistics_work_units: int
    solve_work_units: int
    score_work_units: int
    total_numeric_work_units: int

    def __post_init__(self) -> None:
        if type(self.active_feature_counts) is not tuple or not self.active_feature_counts:
            raise TypeError("active_feature_counts must be a nonempty exact tuple")
        for value in self.active_feature_counts:
            _require_positive_int(value, "active_feature_count")
        for name in self.__dataclass_fields__:
            if name != "active_feature_counts":
                _require_positive_int(getattr(self, name), name)
        if self.mapped_input_bytes > MAX_STORE_FILE_BYTES:
            raise PreparedStoreFileError("prepared store exceeds the 512 MiB file limit")
        if self.result_bytes > MAX_RESULT_FILE_BYTES:
            raise PreparedStoreFileError("prepared result exceeds the 128 MiB file limit")
        if self.modeled_c_heap_bytes > MAX_MODELED_C_HEAP_BYTES:
            raise PreparedStoreFileError("prepared session exceeds the 512 MiB C-heap limit")
        if self.private_disk_scratch_bytes > MAX_PRIVATE_DISK_SCRATCH_BYTES:
            raise PreparedStoreFileError("prepared session exceeds the 1 GiB scratch limit")
        if self.total_numeric_work_units > MAX_TOTAL_NUMERIC_WORK_UNITS:
            raise PreparedStoreFileError("prepared session exceeds the numeric-work limit")


@dataclass(frozen=True, slots=True)
class PreparedStoreFileMetadata:
    """Validated fixed-header metadata and its aggregate session estimate."""

    file_bytes: int
    domain_count: int
    row_count: int
    feature_count: int
    target_count: int
    row_key_offset: int
    row_key_bytes: int
    domain_index_offset: int
    domain_index_bytes: int
    feature_offset: int
    feature_bytes: int
    target_offset: int
    target_bytes: int
    graph_identity_sha256: str
    source_fit_sha256: str
    logical_store_sha256: str
    embedding_snapshot_sha256: str | None
    embedding_identity_sha256: str | None
    model_catalogue_sha256: str
    store_payload_sha256: str
    domain_row_counts: tuple[int, ...]
    domain_active_tag_masks: tuple[int, ...]
    estimate: PreparedSessionEstimate


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value <= 0:
        raise PreparedStoreFileError(f"{name} must be positive")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise PreparedStoreFileError(f"{name} must be lowercase SHA-256 hex")
    if any(character not in "0123456789abcdef" for character in value):
        raise PreparedStoreFileError(f"{name} must be lowercase SHA-256 hex")
    return value


def _row_id_has_content(payload: bytes) -> bool:
    return any(byte not in _ASCII_WHITESPACE_BYTES for byte in payload)


class _HashWriter:
    """Match the existing prepared-store length-framed identity encoding."""

    def __init__(self, namespace: str) -> None:
        self._digest = hashlib.sha256()
        self.text("namespace", namespace)

    def token(self, label: str, payload: bytes) -> None:
        label_bytes = label.encode("ascii")
        self._digest.update(struct.pack("<I", len(label_bytes)))
        self._digest.update(label_bytes)
        self._digest.update(struct.pack("<Q", len(payload)))
        self._digest.update(payload)

    def text(self, label: str, value: str) -> None:
        self.token(label, value.encode("utf-8"))

    def integer(self, label: str, value: int) -> None:
        self.token(label, struct.pack("<Q", value))

    def boolean(self, label: str, value: bool) -> None:
        self.token(label, b"\x01" if value else b"\x00")

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _graph_identity_sha256(plan: PreparedNestedLodoPlan) -> str:
    writer = _HashWriter(PREPARED_STORE_GRAPH_IDENTITY_ID)
    writer.text("graph.algorithm_id", plan.algorithm_id)
    writer.integer("graph.domain_count", len(plan.domains))
    for domain, count in zip(plan.domains, plan.domain_example_counts, strict=True):
        writer.text("graph.domain", domain)
        writer.integer("graph.domain_row_count", count)
    writer.integer("graph.feature_count", plan.feature_count)
    writer.integer("graph.target_count", plan.target_count)
    return writer.hexdigest()


def _embedding_identity_sha256(identity: EmbeddingIdentity | None) -> str | None:
    if identity is None:
        return None
    writer = _HashWriter(PREPARED_STORE_EMBEDDING_IDENTITY_ID)
    writer.text("embedding.provider", identity.provider)
    writer.text("embedding.model_id", identity.model_id)
    writer.text("embedding.revision", identity.revision)
    writer.text("embedding.pooling", identity.pooling)
    writer.boolean("embedding.normalize", identity.normalize)
    writer.text("embedding.asset_manifest_sha256", identity.asset_manifest_sha256)
    return writer.hexdigest()


def _model_catalogue_sha256(model_ids: tuple[str, ...]) -> str:
    writer = _HashWriter(PREPARED_STORE_MODEL_CATALOGUE_ID)
    writer.integer("model_count", len(model_ids))
    for model_id in model_ids:
        writer.text("model_id", model_id)
    return writer.hexdigest()


def estimate_prepared_session(
    *,
    domain_row_counts: tuple[int, ...],
    domain_active_tag_masks: tuple[int, ...],
    feature_count: int,
    target_count: int,
    store_file_bytes: int,
    row_key_bytes: int,
) -> PreparedSessionEstimate:
    """Return deterministic version-1 admission arithmetic without allocating data."""

    if type(domain_row_counts) is not tuple or type(domain_active_tag_masks) is not tuple:
        raise TypeError("domain counts and masks must be exact tuples")
    domain_count = len(domain_row_counts)
    if not 4 <= domain_count <= MAX_PREPARED_DOMAINS:
        raise PreparedStoreFileError("domain_count must be in [4, 7]")
    if len(domain_active_tag_masks) != domain_count:
        raise PreparedStoreFileError("domain masks must match the domain count")
    for count in domain_row_counts:
        _require_positive_int(count, "domain row count")
    for mask in domain_active_tag_masks:
        if type(mask) is not int or not 0 <= mask <= _TAG_MASK:
            raise PreparedStoreFileError("domain active-tag mask contains an unknown bit")
    _require_positive_int(feature_count, "feature_count")
    _require_positive_int(target_count, "target_count")
    _require_positive_int(store_file_bytes, "store_file_bytes")
    _require_positive_int(row_key_bytes, "row_key_bytes")
    if not UNIVERSAL_SURFACE_WIDTH <= feature_count <= MAX_PREPARED_FEATURES:
        raise PreparedStoreFileError("feature_count is outside the prepared-store range")
    if target_count > MAX_PREPARED_TARGETS:
        raise PreparedStoreFileError("target_count exceeds the prepared-store range")
    row_count = sum(domain_row_counts)
    if row_count > MAX_PREPARED_EXAMPLES:
        raise PreparedStoreFileError("row_count exceeds the prepared-store range")

    active_widths: list[int] = []
    coefficient_bytes = 0
    score_bytes = 0
    score_memberships = 0
    solve_work = 0
    score_work = 0
    score_records = 0
    output_cells = 0
    embedding_width = feature_count - UNIVERSAL_SURFACE_WIDTH
    all_indices = tuple(range(domain_count))
    for omitted_count in (3, 2, 1):
        for omitted in combinations(all_indices, omitted_count):
            omitted_set = frozenset(omitted)
            included = tuple(index for index in all_indices if index not in omitted_set)
            active_mask = 0
            for domain_index in included:
                active_mask |= domain_active_tag_masks[domain_index]
            width = _TAG_OFFSET + active_mask.bit_count() + embedding_width
            active_widths.append(width)
            coefficient_cells = 6 + target_count + target_count * width
            coefficient_bytes += COEFFICIENT_RECORD_HEADER_BYTES + 8 * coefficient_cells
            output_cells += coefficient_cells
            solve_work += width**3 + 2 * target_count * width**2 + target_count * width
            for domain_index in omitted:
                rows = domain_row_counts[domain_index]
                score_records += 1
                score_memberships += rows
                score_bytes += SCORE_RECORD_HEADER_BYTES + 8 * rows * target_count
                score_work += rows * target_count * width
                output_cells += rows * target_count

    result_bytes = SESSION_RESULT_HEADER_BYTES + coefficient_bytes + score_bytes
    d = feature_count
    m = target_count
    n = row_count
    statistics_work = 3 * n * (d + m) + n * d * (d + 1) // 2 + n * d * m
    maximum_width = max(active_widths)
    packed_dimension = d * (d + 1) // 2
    coefficient_weight_cells = sum(m * width for width in active_widths)
    # This is the exact reviewed malloc list in ``native/tierroute_prepared.c``:
    # retained targets/domain moments/output records, one subset workspace, and
    # bounded row-key/allocator overhead. The file-backed feature matrix is absent.
    modeled_heap = (
        8
        * (
            n * m
            + domain_count * (d + m + packed_dimension + d * m)
            + len(active_widths) * (6 + m)
            + coefficient_weight_cells
            + score_memberships * m
            + 3 * d
            + 2 * m
            + packed_dimension
            + d * m
            + 2 * maximum_width**2
            + m * maximum_width
        )
        + n
        + 73_728
    )
    private_scratch = SESSION_REQUEST_HEADER_BYTES + store_file_bytes + result_bytes
    total_work = statistics_work + solve_work + score_work
    domain_index_end = STORE_HEADER_BYTES + row_key_bytes + n
    domain_padding_bytes = (-domain_index_end) % 8
    feature_bytes = 8 * n * d
    target_bytes = 8 * n * m
    # The child authenticates every byte once, semantically parses row keys,
    # domains, padding, targets, and features, then rereads features for scoring.
    authentication_scan = (
        store_file_bytes
        + row_key_bytes
        + n
        + domain_padding_bytes
        + target_bytes
        + 2 * feature_bytes
    )
    output_validation_work = output_cells
    return PreparedSessionEstimate(
        active_feature_counts=tuple(active_widths),
        training_subset_count=len(active_widths),
        score_record_count=score_records,
        score_row_memberships=score_memberships,
        mapped_input_bytes=store_file_bytes,
        file_backed_input_bytes=store_file_bytes,
        modeled_c_heap_bytes=modeled_heap,
        private_disk_scratch_bytes=private_scratch,
        coefficient_bytes=coefficient_bytes,
        score_bytes=score_bytes,
        result_bytes=result_bytes,
        authentication_validation_bytes_scanned=authentication_scan,
        output_numeric_cells_validated=output_cells,
        output_validation_work_units=output_validation_work,
        statistics_work_units=statistics_work,
        solve_work_units=solve_work,
        score_work_units=score_work,
        total_numeric_work_units=total_work,
    )


def _digest_or_none(payload: bytes, name: str) -> str | None:
    if payload == _ZERO_SHA256:
        return None
    if len(payload) != 32:
        raise PreparedStoreFileError(f"{name} has the wrong digest width")
    return payload.hex()


def _parse_header(header: bytes, *, actual_file_bytes: int) -> PreparedStoreFileMetadata:
    if len(header) != STORE_HEADER_BYTES:
        raise PreparedStoreFileError("prepared store has a truncated 472-byte header")
    try:
        fields = _HEADER.unpack(header)
    except struct.error as error:  # pragma: no cover - guarded by exact length
        raise PreparedStoreFileError("prepared store header cannot be decoded") from error
    magic = fields[0]
    version = fields[1]
    flags = fields[2]
    (
        header_bytes,
        file_bytes,
        domain_count,
        row_count,
        feature_count,
        target_count,
        surface_width,
        row_key_offset,
        row_key_bytes,
        domain_index_offset,
        domain_index_bytes,
        feature_offset,
        feature_bytes,
        target_offset,
        target_bytes,
    ) = fields[3:18]
    (
        graph_digest,
        source_fit_digest,
        logical_store_digest,
        embedding_snapshot_digest,
        embedding_identity_digest,
        model_catalogue_digest,
        payload_digest,
    ) = fields[18:25]
    counts = tuple(fields[25:32])
    masks = tuple(fields[32:39])

    if magic != STORE_MAGIC:
        raise PreparedStoreFileError("prepared store magic must be TRPSTO01")
    if version != STORE_VERSION:
        raise PreparedStoreFileError("prepared store version must be 1")
    if flags != STORE_FLAGS:
        raise PreparedStoreFileError("prepared store flags must be zero")
    if header_bytes != STORE_HEADER_BYTES:
        raise PreparedStoreFileError("prepared store header size must be exactly 472")
    if file_bytes != actual_file_bytes:
        raise PreparedStoreFileError("prepared store file length does not match its header")
    if not STORE_HEADER_BYTES <= file_bytes <= MAX_STORE_FILE_BYTES:
        raise PreparedStoreFileError("prepared store file length is outside the reviewed limit")
    if not 4 <= domain_count <= MAX_PREPARED_DOMAINS:
        raise PreparedStoreFileError("prepared store domain count must be in [4, 7]")
    if not 1 <= row_count <= MAX_PREPARED_EXAMPLES:
        raise PreparedStoreFileError("prepared store row count is outside the reviewed limit")
    if not UNIVERSAL_SURFACE_WIDTH <= feature_count <= MAX_PREPARED_FEATURES:
        raise PreparedStoreFileError("prepared store feature count is outside the reviewed limit")
    if not 1 <= target_count <= MAX_PREPARED_TARGETS:
        raise PreparedStoreFileError("prepared store target count is outside the reviewed limit")
    if surface_width != UNIVERSAL_SURFACE_WIDTH:
        raise PreparedStoreFileError("prepared store universal surface width must be 12")
    if row_key_offset != STORE_HEADER_BYTES:
        raise PreparedStoreFileError("prepared store row-key offset must be exactly 472")
    minimum_row_key_bytes = row_count * (2 + 1 + 32)
    maximum_row_key_bytes = row_count * (2 + MAX_ROW_ID_UTF8_BYTES + 32)
    if not minimum_row_key_bytes <= row_key_bytes <= maximum_row_key_bytes:
        raise PreparedStoreFileError("prepared store row-key section has an impossible length")
    expected_domain_offset = row_key_offset + row_key_bytes
    if domain_index_offset != expected_domain_offset:
        raise PreparedStoreFileError("prepared store domain-index offset is not contiguous")
    if domain_index_bytes != row_count:
        raise PreparedStoreFileError("prepared store domain-index length must equal N")
    expected_feature_offset = (domain_index_offset + domain_index_bytes + 7) & ~7
    if feature_offset != expected_feature_offset or feature_offset % 8:
        raise PreparedStoreFileError("prepared store feature offset or padding is invalid")
    expected_feature_bytes = 8 * row_count * feature_count
    if feature_bytes != expected_feature_bytes:
        raise PreparedStoreFileError("prepared store feature section length is invalid")
    if target_offset != feature_offset + feature_bytes:
        raise PreparedStoreFileError("prepared store target offset is not contiguous")
    expected_target_bytes = 8 * row_count * target_count
    if target_bytes != expected_target_bytes:
        raise PreparedStoreFileError("prepared store target section length is invalid")
    if file_bytes != target_offset + target_bytes:
        raise PreparedStoreFileError("prepared store EOF or trailing-byte contract is invalid")

    required_digests = (
        (graph_digest, "graph identity"),
        (source_fit_digest, "source-fit"),
        (logical_store_digest, "logical store"),
        (model_catalogue_digest, "model catalogue"),
        (payload_digest, "payload"),
    )
    for digest, name in required_digests:
        if digest == _ZERO_SHA256:
            raise PreparedStoreFileError(f"prepared store {name} digest must not be zero")
    embedding_snapshot = _digest_or_none(embedding_snapshot_digest, "embedding snapshot")
    embedding_identity = _digest_or_none(embedding_identity_digest, "embedding identity")
    if (embedding_snapshot is None) != (embedding_identity is None):
        raise PreparedStoreFileError("prepared store embedding digests must both be set or absent")
    if feature_count == UNIVERSAL_SURFACE_WIDTH and embedding_snapshot is not None:
        raise PreparedStoreFileError("surface-only prepared stores cannot carry embedding digests")
    if feature_count > UNIVERSAL_SURFACE_WIDTH and embedding_snapshot is None:
        raise PreparedStoreFileError("embedded prepared stores require embedding digests")

    active_counts = counts[:domain_count]
    active_masks = masks[:domain_count]
    if any(value != 0 for value in counts[domain_count:]):
        raise PreparedStoreFileError("unused prepared-store domain counts must be zero")
    if any(value != 0 for value in masks[domain_count:]):
        raise PreparedStoreFileError("unused prepared-store tag masks must be zero")
    if any(value == 0 for value in active_counts) or sum(active_counts) != row_count:
        raise PreparedStoreFileError("prepared-store domain counts must be positive and sum to N")
    if any(mask > _TAG_MASK for mask in active_masks):
        raise PreparedStoreFileError("prepared-store tag mask contains an unknown bit")

    estimate = estimate_prepared_session(
        domain_row_counts=active_counts,
        domain_active_tag_masks=active_masks,
        feature_count=feature_count,
        target_count=target_count,
        store_file_bytes=file_bytes,
        row_key_bytes=row_key_bytes,
    )
    return PreparedStoreFileMetadata(
        file_bytes=file_bytes,
        domain_count=domain_count,
        row_count=row_count,
        feature_count=feature_count,
        target_count=target_count,
        row_key_offset=row_key_offset,
        row_key_bytes=row_key_bytes,
        domain_index_offset=domain_index_offset,
        domain_index_bytes=domain_index_bytes,
        feature_offset=feature_offset,
        feature_bytes=feature_bytes,
        target_offset=target_offset,
        target_bytes=target_bytes,
        graph_identity_sha256=graph_digest.hex(),
        source_fit_sha256=source_fit_digest.hex(),
        logical_store_sha256=logical_store_digest.hex(),
        embedding_snapshot_sha256=embedding_snapshot,
        embedding_identity_sha256=embedding_identity,
        model_catalogue_sha256=model_catalogue_digest.hex(),
        store_payload_sha256=payload_digest.hex(),
        domain_row_counts=active_counts,
        domain_active_tag_masks=active_masks,
        estimate=estimate,
    )


def _read_exact(descriptor: int, length: int, context: str) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        try:
            chunk = os.read(descriptor, min(_COPY_CHUNK_BYTES, length - len(chunks)))
        except OSError as error:
            raise PreparedStoreFileError(
                f"cannot read prepared store {context}: {error}"
            ) from error
        if not chunk:
            raise PreparedStoreFileError(f"prepared store is truncated in {context}")
        chunks.extend(chunk)
    return bytes(chunks)


def _write_exact(stream: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = stream.write(view)
        except (OSError, TypeError, ValueError) as error:
            raise PreparedStoreFileError(
                f"cannot copy authenticated prepared store: {error}"
            ) from error
        if written is None or written <= 0:
            raise PreparedStoreFileError("prepared-store destination write made no progress")
        view = view[written:]


def _receipt_matches_header(
    receipt: PreparedStoreFileReceipt,
    metadata: PreparedStoreFileMetadata,
) -> None:
    if receipt.source_fit_sha256 != metadata.source_fit_sha256:
        raise PreparedStoreFileError("prepared store source-fit digest is not caller-pinned")
    if receipt.logical_store_sha256 != metadata.logical_store_sha256:
        raise PreparedStoreFileError("prepared store logical digest is not caller-pinned")
    if receipt.embedding_snapshot_sha256 != metadata.embedding_snapshot_sha256:
        raise PreparedStoreFileError("prepared store embedding digest is not caller-pinned")


def _validate_f64_payload(
    payload: bytes,
    *,
    feature_width: int | None,
    domain_index: int | None,
    observed_masks: list[int],
) -> None:
    for column, (value,) in enumerate(struct.iter_unpack("<d", payload)):
        if not math.isfinite(value):
            raise PreparedStoreFileError("prepared store numeric cells must be finite binary64")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise PreparedStoreFileError("prepared store numeric zero must use positive zero")
        if feature_width is None:
            continue
        feature_column = column % feature_width
        if feature_column < 3 and value < 0.0:
            raise PreparedStoreFileError("prepared store continuous features must be non-negative")
        if 3 <= feature_column < UNIVERSAL_SURFACE_WIDTH and value not in (0.0, 1.0):
            raise PreparedStoreFileError("prepared store binary/tag features must be zero or one")
        if (
            domain_index is not None
            and _TAG_OFFSET <= feature_column < UNIVERSAL_SURFACE_WIDTH
            and value == 1.0
        ):
            observed_masks[domain_index] |= 1 << (feature_column - _TAG_OFFSET)


def _scan_store_descriptor(
    descriptor: int,
    *,
    actual_file_bytes: int,
    receipt: PreparedStoreFileReceipt | None,
    destination: BinaryIO | None,
) -> tuple[PreparedStoreFileMetadata, str]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise PreparedStoreFileError(f"cannot seek prepared store descriptor: {error}") from error
    header = _read_exact(descriptor, STORE_HEADER_BYTES, "header")
    metadata = _parse_header(header, actual_file_bytes=actual_file_bytes)
    if receipt is not None:
        _receipt_matches_header(receipt, metadata)

    whole_digest = hashlib.sha256(header)
    payload_digest = hashlib.sha256()
    if destination is not None:
        _write_exact(destination, header)

    def take(length: int, context: str) -> bytes:
        payload = _read_exact(descriptor, length, context)
        whole_digest.update(payload)
        payload_digest.update(payload)
        if destination is not None:
            _write_exact(destination, payload)
        return payload

    row_key_consumed = 0
    previous_id: str | None = None
    for _ in range(metadata.row_count):
        length_payload = take(_U16.size, "row-key ID length")
        (identifier_bytes,) = _U16.unpack(length_payload)
        if not 1 <= identifier_bytes <= MAX_ROW_ID_UTF8_BYTES:
            raise PreparedStoreFileError("prepared store row ID length is outside [1, 4096]")
        identifier_payload = take(identifier_bytes, "row-key ID")
        try:
            identifier = identifier_payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise PreparedStoreFileError("prepared store row ID must be strict UTF-8") from error
        if not _row_id_has_content(identifier_payload):
            raise PreparedStoreFileError(
                "prepared store row ID must contain a non-ASCII-whitespace byte"
            )
        if previous_id is not None and previous_id >= identifier:
            raise PreparedStoreFileError("prepared store row IDs must be strictly increasing")
        previous_id = identifier
        take(32, "row-key prompt digest")
        row_key_consumed += _U16.size + identifier_bytes + 32
        if row_key_consumed > metadata.row_key_bytes:
            raise PreparedStoreFileError("prepared store row-key records overrun their section")
    if row_key_consumed != metadata.row_key_bytes:
        raise PreparedStoreFileError("prepared store row-key records do not fill their section")

    domain_payload = take(metadata.domain_index_bytes, "domain-index section")
    observed_counts = [0] * metadata.domain_count
    for domain_index in domain_payload:
        if domain_index >= metadata.domain_count:
            raise PreparedStoreFileError("prepared store domain index is outside its catalogue")
        observed_counts[domain_index] += 1
    if tuple(observed_counts) != metadata.domain_row_counts:
        raise PreparedStoreFileError("prepared store domain indices contradict header counts")
    padding_bytes = metadata.feature_offset - (
        metadata.domain_index_offset + metadata.domain_index_bytes
    )
    if padding_bytes and any(take(padding_bytes, "domain-index padding")):
        raise PreparedStoreFileError("prepared store alignment padding must be all zero")

    feature_row_bytes = 8 * metadata.feature_count
    observed_masks = [0] * metadata.domain_count
    for row_index, domain_index in enumerate(domain_payload):
        row = take(feature_row_bytes, f"feature row {row_index}")
        _validate_f64_payload(
            row,
            feature_width=metadata.feature_count,
            domain_index=domain_index,
            observed_masks=observed_masks,
        )
    if tuple(observed_masks) != metadata.domain_active_tag_masks:
        raise PreparedStoreFileError("prepared store feature tags contradict header masks")

    target_row_bytes = 8 * metadata.target_count
    for row_index in range(metadata.row_count):
        row = take(target_row_bytes, f"target row {row_index}")
        _validate_f64_payload(
            row,
            feature_width=None,
            domain_index=None,
            observed_masks=observed_masks,
        )
    try:
        trailing = os.read(descriptor, 1)
    except OSError as error:
        raise PreparedStoreFileError(f"cannot check prepared-store EOF: {error}") from error
    if trailing:
        raise PreparedStoreFileError("prepared store contains trailing bytes")
    if payload_digest.hexdigest() != metadata.store_payload_sha256:
        raise PreparedStoreFileError("prepared store embedded payload SHA-256 does not match")
    whole_hexdigest = whole_digest.hexdigest()
    if receipt is not None and whole_hexdigest != receipt.whole_file_sha256:
        raise PreparedStoreFileError("prepared store whole-file SHA-256 is not caller-pinned")
    return metadata, whole_hexdigest


def _absolute_path(value: str | os.PathLike[str], name: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a path-like value") from error
    if isinstance(raw, bytes):
        raise TypeError(f"{name} must decode to a string path")
    if not raw or "\x00" in raw:
        raise PreparedStoreFileError(f"{name} must be a nonempty path without NUL bytes")
    if raw.startswith(("//", "\\\\")):
        raise PreparedStoreFileError(f"{name} must not be a UNC or device path")
    path = Path(raw)
    if not path.is_absolute():
        raise PreparedStoreFileError(f"{name} must be absolute")
    return path


def _lstat(path: Path, context: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise PreparedStoreFileError(f"cannot inspect {context}: {error}") from error


def _is_reparse_point(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _validate_source_node(details: os.stat_result) -> None:
    if stat.S_ISLNK(details.st_mode) or _is_reparse_point(details):
        raise PreparedStoreFileError("prepared store source must not be a symlink/reparse point")
    if not stat.S_ISREG(details.st_mode):
        raise PreparedStoreFileError("prepared store source must be a regular file")


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _nanosecond_time(details: os.stat_result, name: str) -> int:
    explicit = getattr(details, f"st_{name}_ns", None)
    if explicit is not None:
        return int(explicit)
    return int(getattr(details, f"st_{name}") * 1_000_000_000)


def _require_stable_source(
    before: os.stat_result,
    after: os.stat_result,
    *,
    compare_change_time: bool = True,
) -> None:
    if not _same_inode(before, after) or not stat.S_ISREG(after.st_mode):
        raise PreparedStoreFileError("prepared store source identity changed during authentication")
    changed = before.st_size != after.st_size or _nanosecond_time(
        before, "mtime"
    ) != _nanosecond_time(after, "mtime")
    if compare_change_time:
        changed = changed or _nanosecond_time(before, "ctime") != _nanosecond_time(after, "ctime")
    if changed:
        raise PreparedStoreFileError("prepared store source contents changed during authentication")


def _open_source(path: Path) -> tuple[int, os.stat_result, os.stat_result]:
    path_details = _lstat(path, "prepared store source")
    _validate_source_node(path_details)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PreparedStoreFileError(f"cannot open prepared store source: {error}") from error
    try:
        opened = os.fstat(descriptor)
        _validate_source_node(opened)
        if not _same_inode(path_details, opened):
            raise PreparedStoreFileError("prepared store path changed while it was opened")
        if opened.st_size < STORE_HEADER_BYTES or opened.st_size > MAX_STORE_FILE_BYTES:
            raise PreparedStoreFileError("prepared store source size is outside the reviewed limit")
        return descriptor, path_details, opened
    except BaseException:
        os.close(descriptor)
        raise


def _finish_source_authentication(
    path: Path,
    descriptor: int,
    opened: os.stat_result,
) -> None:
    try:
        final_descriptor = os.fstat(descriptor)
    except OSError as error:
        raise PreparedStoreFileError(f"cannot reinspect prepared store source: {error}") from error
    _require_stable_source(opened, final_descriptor)
    final_path = _lstat(path, "prepared store source after authentication")
    _validate_source_node(final_path)
    _require_stable_source(
        opened,
        final_path,
        # On Windows, descriptor and path stat expose the same file identity,
        # size, and mtime but can report different creation-time precision via
        # st_ctime_ns.  The same-interface fstat check above stays strict; the
        # authenticated private copy and receipt hash bind the consumed bytes.
        compare_change_time=os.name != "nt",
    )


def _destination_start(stream: BinaryIO) -> tuple[int, int, os.stat_result]:
    if getattr(stream, "closed", False):
        raise PreparedStoreFileError("prepared-store destination stream is closed")
    try:
        stream.flush()
        descriptor = stream.fileno()
        position = stream.tell()
        descriptor_position = os.lseek(descriptor, 0, os.SEEK_CUR)
        details = os.fstat(descriptor)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise PreparedStoreFileError(
            f"prepared-store destination must be a seekable regular binary file: {error}"
        ) from error
    if type(descriptor) is not int or type(position) is not int:
        raise PreparedStoreFileError("prepared-store destination has invalid descriptor state")
    if not stat.S_ISREG(details.st_mode):
        raise PreparedStoreFileError("prepared-store destination must be a regular file")
    if _is_reparse_point(details):
        raise PreparedStoreFileError("prepared-store destination must not be a reparse point")
    if os.name != "nt" and stat.S_IMODE(details.st_mode) != (stat.S_IRUSR | stat.S_IWUSR):
        raise PreparedStoreFileError("prepared-store destination must be owner-only mode 0600")
    if position != descriptor_position or position != details.st_size:
        raise PreparedStoreFileError("prepared-store destination must be positioned at exact EOF")
    return descriptor, position, details


def _rollback_destination(stream: BinaryIO, start: int, cause: BaseException) -> None:
    try:
        stream.flush()
        stream.seek(start)
        stream.truncate(start)
        stream.flush()
    except BaseException as cleanup_error:
        raise PreparedStoreFileError(
            f"prepared-store copy failed and destination rollback failed: {cleanup_error}"
        ) from cause


def copy_authenticated_prepared_store(
    source: str | os.PathLike[str],
    receipt: PreparedStoreFileReceipt,
    destination_fd: BinaryIO,
) -> PreparedStoreFileMetadata:
    """Authenticate and stream one store into an already-open file at exact EOF.

    The fixed header and aggregate estimate are rejected before payload bytes are
    copied. On any later failure, this function truncates the caller-owned stream
    back to its original length and leaves it open.
    """

    if type(receipt) is not PreparedStoreFileReceipt:
        raise TypeError("receipt must be an exact PreparedStoreFileReceipt")
    path = _absolute_path(source, "prepared store source")
    _, start, destination_details = _destination_start(destination_fd)
    descriptor, _, opened = _open_source(path)
    operation_error: BaseException | None = None
    try:
        if _same_inode(opened, destination_details):
            raise PreparedStoreFileError("prepared store source and destination must differ")
        try:
            metadata, _ = _scan_store_descriptor(
                descriptor,
                actual_file_bytes=opened.st_size,
                receipt=receipt,
                destination=destination_fd,
            )
            destination_fd.flush()
            _finish_source_authentication(path, descriptor, opened)
            if destination_fd.tell() != start + metadata.file_bytes:
                raise PreparedStoreFileError(
                    "prepared-store destination length changed while copying"
                )
            destination_after = os.fstat(destination_fd.fileno())
            if destination_after.st_size != start + metadata.file_bytes:
                raise PreparedStoreFileError("prepared-store destination is not at exact EOF")
            return metadata
        except BaseException as error:
            _rollback_destination(destination_fd, start, error)
            raise
    except BaseException as error:
        operation_error = error
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as error:
            if operation_error is None:
                raise PreparedStoreFileError(
                    f"cannot close authenticated prepared-store source: {error}"
                ) from error


class AuthenticatedPreparedStore:
    """Owner-only verified snapshot with an explicitly bounded mmap lifetime."""

    __slots__ = (
        "_closed",
        "_mapping",
        "_stream",
        "_temporary_directory",
        "metadata",
        "snapshot_path",
    )

    def __init__(
        self,
        *,
        temporary_directory: tempfile.TemporaryDirectory[str],
        snapshot_path: Path,
        stream: BinaryIO,
        mapping: mmap.mmap,
        metadata: PreparedStoreFileMetadata,
    ) -> None:
        self._temporary_directory = temporary_directory
        self.snapshot_path = snapshot_path
        self._stream = stream
        self._mapping = mapping
        self.metadata = metadata
        self._closed = False

    @property
    def mapping(self) -> mmap.mmap:
        """Return the read-only mapping while this context remains open."""

        if self._closed:
            raise ValueError("authenticated prepared-store mapping is closed")
        return self._mapping

    @property
    def closed(self) -> bool:
        """Whether the private mapping and workspace have been released."""

        return self._closed

    def close(self) -> None:
        """Release the mapping before deleting the private snapshot."""

        if self._closed:
            return
        try:
            self._mapping.close()
        except BaseException as caught:
            # In particular, an exported memoryview raises BufferError. Keep the
            # object open so the caller can release the view and retry cleanup.
            raise PreparedStoreFileError(
                f"cannot close authenticated prepared-store mapping: {caught}"
            ) from caught
        error: BaseException | None = None
        try:
            self._stream.close()
        except BaseException as caught:
            error = error or caught
        try:
            self._temporary_directory.cleanup()
        except BaseException as caught:
            error = error or caught
        if error is not None:
            raise PreparedStoreFileError(
                f"cannot close authenticated prepared-store snapshot: {error}"
            ) from error
        self._closed = True

    def __enter__(self) -> AuthenticatedPreparedStore:
        if self._closed:
            raise ValueError("authenticated prepared-store snapshot is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def authenticate_prepared_store_file(
    source: str | os.PathLike[str],
    receipt: PreparedStoreFileReceipt,
) -> AuthenticatedPreparedStore:
    """Copy one caller-pinned source into an owner-only, read-only mmap snapshot."""

    if type(receipt) is not PreparedStoreFileReceipt:
        raise TypeError("receipt must be an exact PreparedStoreFileReceipt")
    path = _absolute_path(source, "prepared store source")
    temporary = tempfile.TemporaryDirectory(prefix="tierroute-prepared-store-")
    directory = Path(temporary.name)
    stream: BinaryIO | None = None
    mapping: mmap.mmap | None = None
    try:
        if os.name != "nt":
            os.chmod(directory, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        snapshot = directory / "store.bin"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(snapshot, flags, stat.S_IRUSR | stat.S_IWUSR)
        stream = os.fdopen(descriptor, "w+b", closefd=True)
        metadata = copy_authenticated_prepared_store(path, receipt, stream)
        stream.flush()
        os.fsync(stream.fileno())
        if os.name != "nt":
            os.chmod(snapshot, stat.S_IRUSR)
        mapping = mmap.mmap(stream.fileno(), metadata.file_bytes, access=mmap.ACCESS_READ)
        return AuthenticatedPreparedStore(
            temporary_directory=temporary,
            snapshot_path=snapshot,
            stream=stream,
            mapping=mapping,
            metadata=metadata,
        )
    except BaseException:
        if mapping is not None:
            mapping.close()
        if stream is not None:
            stream.close()
        temporary.cleanup()
        raise


def _new_destination_path(value: str | os.PathLike[str]) -> tuple[Path, os.stat_result]:
    destination = _absolute_path(value, "prepared-store destination")
    try:
        existing = destination.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise PreparedStoreFileError(
            f"cannot inspect prepared-store destination: {error}"
        ) from error
    if existing is not None:
        raise PreparedStoreFileError("prepared-store destination must be a new absent path")
    parent = destination.parent
    parent_details = _lstat(parent, "prepared-store destination directory")
    if stat.S_ISLNK(parent_details.st_mode) or _is_reparse_point(parent_details):
        raise PreparedStoreFileError("prepared-store destination directory must not be a symlink")
    if not stat.S_ISDIR(parent_details.st_mode):
        raise PreparedStoreFileError("prepared-store destination parent must be a directory")
    return destination, parent_details


def _iterable(value: object, name: str) -> object:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an iterable of records")
    try:
        return iter(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error


def _next_record(iterator: object, name: str, index: int) -> object:
    try:
        return next(iterator)  # type: ignore[arg-type]
    except StopIteration as error:
        raise PreparedStoreFileError(f"{name} ended before row {index}") from error
    except RuntimeError as error:
        raise PreparedStoreFileError(f"{name} failed while reading row {index}") from error


def _require_exhausted(iterator: object, name: str, row_count: int) -> None:
    try:
        next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        return
    except RuntimeError as error:
        raise PreparedStoreFileError(f"{name} failed after row {row_count - 1}") from error
    raise PreparedStoreFileError(f"{name} contains more than {row_count} rows")


def _canonical_f64(value: object, context: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise TypeError(f"{context} must be an exact real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise PreparedStoreFileError(f"{context} must be finite binary64") from error
    if not math.isfinite(result):
        raise PreparedStoreFileError(f"{context} must be finite binary64")
    if result == 0.0 and math.copysign(1.0, result) < 0:
        raise PreparedStoreFileError(f"{context} must use canonical positive zero")
    return result


def _numeric_row(value: object, *, width: int, context: str) -> tuple[float, ...]:
    iterator = _iterable(value, context)
    cells: list[float] = []
    for column in range(width):
        cells.append(
            _canonical_f64(_next_record(iterator, context, column), f"{context}[{column}]")
        )
    try:
        next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        return tuple(cells)
    raise PreparedStoreFileError(f"{context} has more than {width} cells")


def _validated_model_ids(model_ids: object, target_count: int) -> tuple[str, ...]:
    if type(model_ids) is not tuple:
        raise TypeError("model_ids must be an exact tuple")
    if len(model_ids) != target_count:
        raise PreparedStoreFileError("model_ids must match the target count")
    total_bytes = 0
    for model_id in model_ids:
        if type(model_id) is not str or not model_id.strip():
            raise PreparedStoreFileError("model IDs must be nonempty exact strings")
        try:
            encoded = model_id.encode("utf-8")
        except UnicodeEncodeError as error:
            raise PreparedStoreFileError("model IDs must be valid UTF-8") from error
        if len(encoded) > MAX_ROW_ID_UTF8_BYTES:
            raise PreparedStoreFileError("model ID exceeds the 4096-byte limit")
        total_bytes += len(encoded)
    if model_ids != tuple(sorted(set(model_ids))):
        raise PreparedStoreFileError("model IDs must be sorted and unique")
    if total_bytes > 256 * 1024:
        raise PreparedStoreFileError("model catalogue exceeds the reviewed byte limit")
    return model_ids


def _validate_section_identities(
    *,
    plan: PreparedNestedLodoPlan,
    model_ids: object,
    embedding_identity: EmbeddingIdentity | None,
    embedding_snapshot_sha256: str | None,
    expected_source_fit_sha256: str,
    logical_store_sha256: str,
) -> tuple[tuple[str, ...], str, str, str | None, str | None, str, str]:
    if type(plan) is not PreparedNestedLodoPlan:
        raise TypeError("plan must be an exact PreparedNestedLodoPlan")
    if plan.algorithm_id != PREPARED_GRAPH_ALGORITHM_ID:
        raise PreparedStoreFileError("plan algorithm does not match the prepared graph")
    if not UNIVERSAL_SURFACE_WIDTH <= plan.feature_count <= MAX_PREPARED_FEATURES:
        raise PreparedStoreFileError("plan feature count is outside the prepared-store range")
    models = _validated_model_ids(model_ids, plan.target_count)
    source_digest = _require_sha256(expected_source_fit_sha256, "expected_source_fit_sha256")
    logical_digest = _require_sha256(logical_store_sha256, "logical_store_sha256")
    embedding_width = plan.feature_count - UNIVERSAL_SURFACE_WIDTH
    if embedding_width == 0:
        if embedding_identity is not None or embedding_snapshot_sha256 is not None:
            raise PreparedStoreFileError("surface-only stores cannot carry embedding provenance")
        embedding_digest = None
    else:
        if type(embedding_identity) is not EmbeddingIdentity:
            raise TypeError("embedded stores require an exact EmbeddingIdentity")
        embedding_digest = _require_sha256(
            embedding_snapshot_sha256,
            "embedding_snapshot_sha256",
        )
    graph_digest = _graph_identity_sha256(plan)
    identity_digest = _embedding_identity_sha256(embedding_identity)
    model_digest = _model_catalogue_sha256(models)
    return (
        models,
        source_digest,
        logical_digest,
        embedding_digest,
        identity_digest,
        graph_digest,
        model_digest,
    )


def _pack_store_header(
    *,
    file_bytes: int,
    plan: PreparedNestedLodoPlan,
    row_key_bytes: int,
    domain_index_offset: int,
    feature_offset: int,
    feature_bytes: int,
    target_offset: int,
    target_bytes: int,
    graph_digest: str,
    source_digest: str,
    logical_digest: str,
    embedding_digest: str | None,
    embedding_identity_digest: str | None,
    model_digest: str,
    payload_digest: str,
    active_masks: tuple[int, ...],
) -> bytes:
    domain_count = len(plan.domains)
    counts = (*plan.domain_example_counts, *(0 for _ in range(MAX_PREPARED_DOMAINS - domain_count)))
    masks = (*active_masks, *(0 for _ in range(MAX_PREPARED_DOMAINS - domain_count)))
    return _HEADER.pack(
        STORE_MAGIC,
        STORE_VERSION,
        STORE_FLAGS,
        STORE_HEADER_BYTES,
        file_bytes,
        domain_count,
        plan.work.example_count,
        plan.feature_count,
        plan.target_count,
        UNIVERSAL_SURFACE_WIDTH,
        STORE_HEADER_BYTES,
        row_key_bytes,
        domain_index_offset,
        plan.work.example_count,
        feature_offset,
        feature_bytes,
        target_offset,
        target_bytes,
        bytes.fromhex(graph_digest),
        bytes.fromhex(source_digest),
        bytes.fromhex(logical_digest),
        _ZERO_SHA256 if embedding_digest is None else bytes.fromhex(embedding_digest),
        _ZERO_SHA256
        if embedding_identity_digest is None
        else bytes.fromhex(embedding_identity_digest),
        bytes.fromhex(model_digest),
        bytes.fromhex(payload_digest),
        *counts,
        *masks,
    )


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new_stage(
    stage: Path,
    destination: Path,
    expected_parent: os.stat_result,
    expected_stage: os.stat_result,
) -> None:
    current_parent = _lstat(destination.parent, "prepared-store destination directory")
    if not _same_inode(expected_parent, current_parent):
        raise PreparedStoreFileError("prepared-store destination directory was replaced")
    current_stage = _lstat(stage, "prepared-store stage before publication")
    _validate_source_node(current_stage)
    if not _same_inode(expected_stage, current_stage):
        raise PreparedStoreFileError("prepared-store stage was replaced before publication")
    linked = False
    try:
        try:
            os.link(stage, destination, follow_symlinks=False)
        except TypeError:  # pragma: no cover - older Windows/Python API variant
            os.link(stage, destination)
        linked = True
        destination_details = _lstat(destination, "published prepared store")
        stage_details = _lstat(stage, "prepared-store stage")
        _validate_source_node(destination_details)
        if not _same_inode(expected_stage, destination_details) or not _same_inode(
            expected_stage, stage_details
        ):
            raise PreparedStoreFileError("published prepared store does not match its stage")
        if os.name != "nt" and stat.S_IMODE(destination_details.st_mode) != (
            stat.S_IRUSR | stat.S_IWUSR
        ):
            raise PreparedStoreFileError("published prepared store must be owner-only mode 0600")
        _fsync_parent(destination)
    except BaseException:
        if linked:
            try:
                destination_details = destination.lstat()
                if _same_inode(destination_details, expected_stage):
                    destination.unlink()
                    _fsync_parent(destination)
            except OSError:
                pass
        raise


def write_prepared_store_file_from_sections(
    *,
    destination: str | os.PathLike[str],
    plan: PreparedNestedLodoPlan,
    model_ids: tuple[str, ...],
    example_ids: Iterable[str],
    prompt_sha256s: Iterable[str],
    domain_indices: Iterable[int],
    feature_rows: Iterable[Sequence[float]],
    target_rows: Iterable[Sequence[float]],
    embedding_identity: EmbeddingIdentity | None,
    embedding_snapshot_sha256: str | None,
    expected_source_fit_sha256: str,
    logical_store_sha256: str,
) -> PreparedStoreFileReceipt:
    """Stream independently supplied sections into one authenticated new file.

    Every section iterable is consumed exactly once. The largest Python scratch
    objects are one feature row and the one-byte-per-row domain catalogue needed
    to validate per-domain tag masks after the domain section has been written.
    """

    destination_path, parent_details = _new_destination_path(destination)
    (
        _,
        source_digest,
        logical_digest,
        embedding_digest,
        embedding_identity_digest,
        graph_digest,
        model_digest,
    ) = _validate_section_identities(
        plan=plan,
        model_ids=model_ids,
        embedding_identity=embedding_identity,
        embedding_snapshot_sha256=embedding_snapshot_sha256,
        expected_source_fit_sha256=expected_source_fit_sha256,
        logical_store_sha256=logical_store_sha256,
    )
    row_count = plan.work.example_count
    feature_bytes = 8 * row_count * plan.feature_count
    target_bytes = 8 * row_count * plan.target_count
    minimum_row_key_bytes = row_count * (_U16.size + 1 + 32)
    minimum_domain_offset = STORE_HEADER_BYTES + minimum_row_key_bytes
    minimum_feature_offset = (minimum_domain_offset + row_count + 7) & ~7
    minimum_file_bytes = minimum_feature_offset + feature_bytes + target_bytes
    # Reject a plan whose smallest possible row-key encoding cannot fit before
    # creating a stage or consuming any caller iterable.
    estimate_prepared_session(
        domain_row_counts=plan.domain_example_counts,
        domain_active_tag_masks=tuple(_TAG_MASK for _ in plan.domains),
        feature_count=plan.feature_count,
        target_count=plan.target_count,
        store_file_bytes=minimum_file_bytes,
        row_key_bytes=minimum_row_key_bytes,
    )
    maximum_row_key_bytes_without_padding = MAX_STORE_FILE_BYTES - (
        STORE_HEADER_BYTES + row_count + feature_bytes + target_bytes
    )
    stage_descriptor, stage_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.stage.",
        suffix=".tmp",
    )
    stage = Path(stage_name)
    stream: BinaryIO | None = None
    try:
        stream = os.fdopen(stage_descriptor, "w+b", closefd=True)
        if os.name != "nt":
            os.fchmod(stream.fileno(), stat.S_IRUSR | stat.S_IWUSR)
        _write_exact(stream, b"\x00" * STORE_HEADER_BYTES)
        payload_digest = hashlib.sha256()

        def write_payload(payload: bytes) -> None:
            _write_exact(stream, payload)
            payload_digest.update(payload)

        id_iterator = _iterable(example_ids, "example_ids")
        prompt_iterator = _iterable(prompt_sha256s, "prompt_sha256s")
        previous_id: str | None = None
        row_key_bytes_written = 0
        for row_index in range(row_count):
            identifier = _next_record(id_iterator, "example_ids", row_index)
            prompt_digest = _next_record(prompt_iterator, "prompt_sha256s", row_index)
            if type(identifier) is not str or not identifier:
                raise PreparedStoreFileError("example IDs must be nonempty exact strings")
            try:
                identifier_payload = identifier.encode("utf-8")
            except UnicodeEncodeError as error:
                raise PreparedStoreFileError("example IDs must be valid UTF-8") from error
            if not _row_id_has_content(identifier_payload):
                raise PreparedStoreFileError("example IDs must contain a non-ASCII-whitespace byte")
            if len(identifier_payload) > MAX_ROW_ID_UTF8_BYTES:
                raise PreparedStoreFileError("example ID exceeds the 4096-byte limit")
            if previous_id is not None and previous_id >= identifier:
                raise PreparedStoreFileError("example IDs must be strictly increasing")
            previous_id = identifier
            prompt_hex = _require_sha256(prompt_digest, "prompt_sha256")
            row_key_record = (
                _U16.pack(len(identifier_payload)) + identifier_payload + bytes.fromhex(prompt_hex)
            )
            remaining_minimum = (row_count - row_index - 1) * (_U16.size + 1 + 32)
            if (
                row_key_bytes_written + len(row_key_record) + remaining_minimum
                > maximum_row_key_bytes_without_padding
            ):
                raise PreparedStoreFileError(
                    "prepared-store row-key section cannot fit the 512 MiB file limit"
                )
            write_payload(row_key_record)
            row_key_bytes_written += len(row_key_record)
        _require_exhausted(id_iterator, "example_ids", row_count)
        _require_exhausted(prompt_iterator, "prompt_sha256s", row_count)
        row_key_bytes = stream.tell() - STORE_HEADER_BYTES
        if row_key_bytes != row_key_bytes_written or row_key_bytes <= 0:
            raise AssertionError("prepared row-key section length changed during serialization")
        domain_index_offset = stream.tell()
        feature_offset = (domain_index_offset + row_count + 7) & ~7
        target_offset = feature_offset + feature_bytes
        file_bytes = target_offset + target_bytes
        # Worst-case masks reject unreviewed work before either large numeric
        # iterable is consumed. The exact-mask estimate is checked after features.
        estimate_prepared_session(
            domain_row_counts=plan.domain_example_counts,
            domain_active_tag_masks=tuple(_TAG_MASK for _ in plan.domains),
            feature_count=plan.feature_count,
            target_count=plan.target_count,
            store_file_bytes=file_bytes,
            row_key_bytes=row_key_bytes,
        )

        domain_iterator = _iterable(domain_indices, "domain_indices")
        domain_cache = bytearray(row_count)
        observed_counts = [0] * len(plan.domains)
        for row_index in range(row_count):
            domain_index = _next_record(domain_iterator, "domain_indices", row_index)
            if type(domain_index) is not int or not 0 <= domain_index < len(plan.domains):
                raise PreparedStoreFileError("domain index is outside the prepared catalogue")
            domain_cache[row_index] = domain_index
            observed_counts[domain_index] += 1
        _require_exhausted(domain_iterator, "domain_indices", row_count)
        if tuple(observed_counts) != plan.domain_example_counts:
            raise PreparedStoreFileError("domain indices contradict the prepared plan counts")
        write_payload(bytes(domain_cache))
        padding_bytes = feature_offset - stream.tell()
        if padding_bytes:
            write_payload(b"\x00" * padding_bytes)
        if stream.tell() != feature_offset:
            raise AssertionError("prepared feature offset changed during serialization")

        feature_iterator = _iterable(feature_rows, "feature_rows")
        observed_masks = [0] * len(plan.domains)
        feature_packer = struct.Struct(f"<{plan.feature_count}d")
        for row_index in range(row_count):
            row = _numeric_row(
                _next_record(feature_iterator, "feature_rows", row_index),
                width=plan.feature_count,
                context=f"feature_rows[{row_index}]",
            )
            for column, value in enumerate(row[:UNIVERSAL_SURFACE_WIDTH]):
                if column < 3 and value < 0.0:
                    raise PreparedStoreFileError("continuous features must be non-negative")
                if 3 <= column < UNIVERSAL_SURFACE_WIDTH and value not in (0.0, 1.0):
                    raise PreparedStoreFileError("binary/tag features must be zero or one")
                if _TAG_OFFSET <= column < UNIVERSAL_SURFACE_WIDTH and value == 1.0:
                    observed_masks[domain_cache[row_index]] |= 1 << (column - _TAG_OFFSET)
            write_payload(feature_packer.pack(*row))
        _require_exhausted(feature_iterator, "feature_rows", row_count)
        if stream.tell() != target_offset:
            raise AssertionError("prepared target offset changed during serialization")

        target_iterator = _iterable(target_rows, "target_rows")
        target_packer = struct.Struct(f"<{plan.target_count}d")
        for row_index in range(row_count):
            row = _numeric_row(
                _next_record(target_iterator, "target_rows", row_index),
                width=plan.target_count,
                context=f"target_rows[{row_index}]",
            )
            write_payload(target_packer.pack(*row))
        _require_exhausted(target_iterator, "target_rows", row_count)
        if stream.tell() != file_bytes:
            raise AssertionError("prepared-store exact file length changed during serialization")
        active_masks = tuple(observed_masks)
        estimate_prepared_session(
            domain_row_counts=plan.domain_example_counts,
            domain_active_tag_masks=active_masks,
            feature_count=plan.feature_count,
            target_count=plan.target_count,
            store_file_bytes=file_bytes,
            row_key_bytes=row_key_bytes,
        )
        header = _pack_store_header(
            file_bytes=file_bytes,
            plan=plan,
            row_key_bytes=row_key_bytes,
            domain_index_offset=domain_index_offset,
            feature_offset=feature_offset,
            feature_bytes=feature_bytes,
            target_offset=target_offset,
            target_bytes=target_bytes,
            graph_digest=graph_digest,
            source_digest=source_digest,
            logical_digest=logical_digest,
            embedding_digest=embedding_digest,
            embedding_identity_digest=embedding_identity_digest,
            model_digest=model_digest,
            payload_digest=payload_digest.hexdigest(),
            active_masks=active_masks,
        )
        stream.seek(0)
        _write_exact(stream, header)
        stream.flush()
        os.fsync(stream.fileno())
        metadata, whole_digest = _scan_store_descriptor(
            stream.fileno(),
            actual_file_bytes=file_bytes,
            receipt=None,
            destination=None,
        )
        if metadata.graph_identity_sha256 != graph_digest:
            raise PreparedStoreFileError("prepared-store graph identity changed during staging")
        receipt = PreparedStoreFileReceipt(
            whole_file_sha256=whole_digest,
            source_fit_sha256=source_digest,
            logical_store_sha256=logical_digest,
            embedding_snapshot_sha256=embedding_digest,
        )
        stage_details = os.fstat(stream.fileno())
        _validate_source_node(stage_details)
        stream.close()
        stream = None
        _publish_new_stage(stage, destination_path, parent_details, stage_details)
        stage.unlink()
        _fsync_parent(destination_path)
        return receipt
    except BaseException:
        if stream is not None:
            stream.close()
        try:
            stage.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_prepared_store_file(
    store: PreparedFeatureStore,
    destination: str | os.PathLike[str],
) -> PreparedStoreFileReceipt:
    """Persist one exact in-memory reference store without full-payload copies."""

    if type(store) is not PreparedFeatureStore:
        raise TypeError("store must be an exact PreparedFeatureStore")
    feature_stride = 8 * store.plan.feature_count
    target_stride = 8 * store.plan.target_count
    return write_prepared_store_file_from_sections(
        destination=destination,
        plan=store.plan,
        model_ids=store.model_ids,
        example_ids=iter(store.example_ids),
        prompt_sha256s=iter(store.prompt_sha256s),
        domain_indices=iter(store.domain_indices),
        feature_rows=(
            struct.unpack_from(
                f"<{store.plan.feature_count}d",
                store.feature_payload,
                row_index * feature_stride,
            )
            for row_index in range(store.plan.work.example_count)
        ),
        target_rows=(
            struct.unpack_from(
                f"<{store.plan.target_count}d",
                store.target_payload,
                row_index * target_stride,
            )
            for row_index in range(store.plan.work.example_count)
        ),
        embedding_identity=store.embedding_identity,
        embedding_snapshot_sha256=store.embedding_snapshot_sha256,
        expected_source_fit_sha256=store.source_fit_sha256,
        logical_store_sha256=store.sha256,
    )


__all__ = [
    "PREPARED_STORE_EMBEDDING_IDENTITY_ID",
    "PREPARED_STORE_FILE_ID",
    "PREPARED_STORE_GRAPH_IDENTITY_ID",
    "PREPARED_STORE_MODEL_CATALOGUE_ID",
    "AuthenticatedPreparedStore",
    "PreparedSessionEstimate",
    "PreparedStoreFileError",
    "PreparedStoreFileMetadata",
    "PreparedStoreFileReceipt",
    "authenticate_prepared_store_file",
    "copy_authenticated_prepared_store",
    "estimate_prepared_session",
    "write_prepared_store_file",
    "write_prepared_store_file_from_sections",
]
