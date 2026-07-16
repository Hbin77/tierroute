# SPDX-License-Identifier: Apache-2.0
"""Authenticated file-backed native prepared solve-and-score sessions.

This module is a training-time adapter for the project-owned ``TRPSES01`` C11
sidecar.  It never searches ``PATH`` or downloads an executable.  Both the
sidecar and the prepared store are authenticated with caller-pinned SHA-256
credentials, and the store is copied descriptor-stably straight behind the
fixed request header.  Successful output stays file-backed for its complete
lifetime; record construction never expands coefficient or score payloads into
Python byte strings or tuples.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import math
import mmap
import os
import secrets
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import BinaryIO, overload

from tierroute.predictors.prepared_files import (
    PreparedStoreFileMetadata,
    PreparedStoreFileReceipt,
    copy_authenticated_prepared_store,
)

PREPARED_SESSION_ENGINE_ID = "tierroute.prepared-session-c11-v1"
PREPARED_MOMENT_SOLVER_ID = "tierroute.prepared-moment-ridge-cholesky-c11-v1"
PREPARED_RAW_SCORER_ID = "tierroute.prepared-raw-dot-product-c11-v1"
PREPARED_SESSION_RESULT_ID = "tierroute.prepared-session-result-f64le-v1"

REQUEST_MAGIC = b"TRPSES01"
RESULT_MAGIC = b"TRPRES01"
PROTOCOL_VERSION = 1
REQUEST_FLAGS = 0
REQUEST_HEADER_BYTES = 160
RESULT_HEADER_BYTES = 448
COEFFICIENT_RECORD_HEADER_BYTES = 48
SCORE_RECORD_HEADER_BYTES = 32

STATUS_SUCCESS = 0
STATUS_PROTOCOL_ERROR = 1
STATUS_RESOURCE_ERROR = 2
STATUS_NUMERIC_ERROR = 3
STATUS_ALLOCATION_ERROR = 4
STATUS_SOLVE_ERROR = 5
STATUS_INTERNAL_ERROR = 6

MAX_STORE_FILE_BYTES = 512 * 1024 * 1024
MAX_RESULT_FILE_BYTES = 128 * 1024 * 1024
MAX_MODELED_C_HEAP_BYTES = 512 * 1024 * 1024
MAX_PRIVATE_DISK_SCRATCH_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_NUMERIC_WORK_UNITS = 200_000_000_000
MAX_BINARY_BYTES = 16 * 1024 * 1024
MAX_STDERR_BYTES = 16 * 1024
MAX_TIMEOUT_SECONDS = 3600.0

_KNOWN_STATUSES = frozenset(range(STATUS_SUCCESS, STATUS_INTERNAL_ERROR + 1))
_STATUS_NAMES = {
    STATUS_PROTOCOL_ERROR: "protocol/header",
    STATUS_RESOURCE_ERROR: "bounds/resource",
    STATUS_NUMERIC_ERROR: "numeric/input",
    STATUS_ALLOCATION_ERROR: "allocation",
    STATUS_SOLVE_ERROR: "solve/residual",
    STATUS_INTERNAL_ERROR: "I/O/internal",
}
_REQUEST_HEADER = struct.Struct("<8sII32s32s32sQQd24s")
_RESULT_HEADER = struct.Struct("<8sII32s32s32sQQQQQQQQQQdQQQQ32s32s32s32s32s32sQQQ")
_COEFFICIENT_HEADER = struct.Struct("<IIQQQQQ")
_SCORE_HEADER = struct.Struct("<IIIIQQ")
_F64 = struct.Struct("<d")
_COPY_CHUNK_BYTES = 1024 * 1024
_PIPE_CHUNK_BYTES = 64 * 1024
_STDERR_TRUNCATION_MARKER = b"\n[tierroute: stderr truncated]\n"

assert _REQUEST_HEADER.size == REQUEST_HEADER_BYTES
assert _RESULT_HEADER.size == RESULT_HEADER_BYTES
assert _COEFFICIENT_HEADER.size == COEFFICIENT_RECORD_HEADER_BYTES
assert _SCORE_HEADER.size == SCORE_RECORD_HEADER_BYTES


class NativePreparedError(RuntimeError):
    """Base class for prepared-session adapter failures."""


class NativePreparedIntegrityError(NativePreparedError):
    """A caller-pinned input failed its identity contract."""


class NativePreparedExecutionError(NativePreparedError):
    """The authenticated child failed at the process or filesystem boundary."""


class NativePreparedProtocolError(NativePreparedError):
    """The child emitted a malformed or unauthenticated result."""


class NativePreparedClosedError(NativePreparedError):
    """A result or payload view was used after its owner closed."""


class NativePreparedStatusError(NativePreparedError):
    """The child returned a known structured failure status."""

    def __init__(self, status: int, stderr: bytes) -> None:
        self.status = status
        self.stderr = stderr
        detail = _stderr_detail(stderr)
        super().__init__(
            f"native prepared session failed with status {status} ({_STATUS_NAMES[status]}){detail}"
        )


@dataclass(frozen=True, slots=True)
class _ExpectedCoefficient:
    subset_index: int
    domain_mask: int
    training_row_count: int
    active_tag_mask: int
    active_feature_count: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class _ExpectedScore:
    block_index: int
    training_subset_index: int
    scored_domain_index: int
    row_count: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class _ExpectedGraph:
    coefficients: tuple[_ExpectedCoefficient, ...]
    scores: tuple[_ExpectedScore, ...]
    coefficient_section_bytes: int
    score_section_bytes: int
    result_bytes: int
    score_row_memberships: int
    statistics_work_units: int
    solve_work_units: int
    score_work_units: int
    authentication_validation_bytes_scanned: int
    output_numeric_cells_validated: int


def _derive_expected_graph(metadata: PreparedStoreFileMetadata) -> _ExpectedGraph:
    domain_count = metadata.domain_count
    row_count = metadata.row_count
    feature_count = metadata.feature_count
    target_count = metadata.target_count
    counts = metadata.domain_row_counts
    tag_masks = metadata.domain_active_tag_masks
    if not 4 <= domain_count <= 7:
        raise NativePreparedIntegrityError("prepared store domain count must be in [4, 7]")
    if len(counts) != domain_count or len(tag_masks) != domain_count:
        raise NativePreparedIntegrityError("prepared store domain catalogues have the wrong length")
    if any(type(value) is not int or value <= 0 for value in counts):
        raise NativePreparedIntegrityError("prepared store domain row counts must be positive")
    if sum(counts) != row_count:
        raise NativePreparedIntegrityError("prepared store domain row counts do not sum to N")
    if feature_count < 12 or target_count < 1:
        raise NativePreparedIntegrityError("prepared store feature/target shape is invalid")
    if any(type(mask) is not int or not 0 <= mask < (1 << 7) for mask in tag_masks):
        raise NativePreparedIntegrityError("prepared store active-tag masks are invalid")

    full_mask = (1 << domain_count) - 1
    coefficients: list[_ExpectedCoefficient] = []
    for omitted_count in (3, 2, 1):
        for omitted in itertools.combinations(range(domain_count), omitted_count):
            omitted_mask = sum(1 << index for index in omitted)
            training_mask = full_mask ^ omitted_mask
            included = tuple(index for index in range(domain_count) if training_mask & (1 << index))
            active_tag_mask = 0
            for index in included:
                active_tag_mask |= tag_masks[index]
            active_feature_count = feature_count - 7 + active_tag_mask.bit_count()
            payload_bytes = 8 * (6 + target_count + target_count * active_feature_count)
            coefficients.append(
                _ExpectedCoefficient(
                    subset_index=len(coefficients),
                    domain_mask=training_mask,
                    training_row_count=sum(counts[index] for index in included),
                    active_tag_mask=active_tag_mask,
                    active_feature_count=active_feature_count,
                    payload_bytes=payload_bytes,
                )
            )

    scores: list[_ExpectedScore] = []
    for coefficient in coefficients:
        for domain_index, count in enumerate(counts):
            if coefficient.domain_mask & (1 << domain_index):
                continue
            scores.append(
                _ExpectedScore(
                    block_index=len(scores),
                    training_subset_index=coefficient.subset_index,
                    scored_domain_index=domain_index,
                    row_count=count,
                    payload_bytes=8 * count * target_count,
                )
            )

    coefficient_section_bytes = sum(
        COEFFICIENT_RECORD_HEADER_BYTES + record.payload_bytes for record in coefficients
    )
    score_section_bytes = sum(SCORE_RECORD_HEADER_BYTES + record.payload_bytes for record in scores)
    score_row_memberships = sum(record.row_count for record in scores)
    statistics_work_units = (
        3 * row_count * (feature_count + target_count)
        + row_count * feature_count * (feature_count + 1) // 2
        + row_count * feature_count * target_count
    )
    solve_work_units = sum(
        record.active_feature_count**3
        + 2 * target_count * record.active_feature_count**2
        + target_count * record.active_feature_count
        for record in coefficients
    )
    score_work_units = sum(
        record.row_count
        * target_count
        * coefficients[record.training_subset_index].active_feature_count
        for record in scores
    )
    authentication_validation_bytes_scanned = (
        metadata.file_bytes
        + metadata.row_key_bytes
        + metadata.domain_index_bytes
        + metadata.feature_offset
        - metadata.domain_index_offset
        - metadata.domain_index_bytes
        + metadata.target_bytes
        + 2 * metadata.feature_bytes
    )
    output_numeric_cells_validated = sum(
        record.payload_bytes // 8 for record in coefficients
    ) + sum(record.payload_bytes // 8 for record in scores)
    return _ExpectedGraph(
        coefficients=tuple(coefficients),
        scores=tuple(scores),
        coefficient_section_bytes=coefficient_section_bytes,
        score_section_bytes=score_section_bytes,
        result_bytes=RESULT_HEADER_BYTES + coefficient_section_bytes + score_section_bytes,
        score_row_memberships=score_row_memberships,
        statistics_work_units=statistics_work_units,
        solve_work_units=solve_work_units,
        score_work_units=score_work_units,
        authentication_validation_bytes_scanned=authentication_validation_bytes_scanned,
        output_numeric_cells_validated=output_numeric_cells_validated,
    )


def preflight_native_prepared_session(
    metadata: PreparedStoreFileMetadata,
    *,
    ridge: object,
) -> _ExpectedGraph:
    """Validate the complete aggregate shape before child launch."""

    if type(metadata) is not PreparedStoreFileMetadata:
        raise TypeError("metadata must be an exact PreparedStoreFileMetadata")
    _positive_f64(ridge, "ridge")
    expected = _derive_expected_graph(metadata)
    estimate = metadata.estimate
    exact_pairs = (
        ("coefficient_bytes", estimate.coefficient_bytes, expected.coefficient_section_bytes),
        ("score_bytes", estimate.score_bytes, expected.score_section_bytes),
        ("result_bytes", estimate.result_bytes, expected.result_bytes),
        ("training_subset_count", estimate.training_subset_count, len(expected.coefficients)),
        ("score_record_count", estimate.score_record_count, len(expected.scores)),
        (
            "score_row_memberships",
            estimate.score_row_memberships,
            expected.score_row_memberships,
        ),
        (
            "statistics_work_units",
            estimate.statistics_work_units,
            expected.statistics_work_units,
        ),
        ("solve_work_units", estimate.solve_work_units, expected.solve_work_units),
        ("score_work_units", estimate.score_work_units, expected.score_work_units),
        (
            "authentication_validation_bytes_scanned",
            estimate.authentication_validation_bytes_scanned,
            expected.authentication_validation_bytes_scanned,
        ),
        (
            "output_numeric_cells_validated",
            estimate.output_numeric_cells_validated,
            expected.output_numeric_cells_validated,
        ),
        (
            "output_validation_work_units",
            estimate.output_validation_work_units,
            expected.output_numeric_cells_validated,
        ),
        ("mapped_input_bytes", estimate.mapped_input_bytes, metadata.file_bytes),
        ("file_backed_input_bytes", estimate.file_backed_input_bytes, metadata.file_bytes),
        (
            "private_disk_scratch_bytes",
            estimate.private_disk_scratch_bytes,
            REQUEST_HEADER_BYTES + metadata.file_bytes + expected.result_bytes,
        ),
    )
    for name, actual, wanted in exact_pairs:
        if actual != wanted:
            raise NativePreparedIntegrityError(
                f"prepared store {name} estimate {actual} does not match {wanted}"
            )
    if tuple(estimate.active_feature_counts) != tuple(
        record.active_feature_count for record in expected.coefficients
    ):
        raise NativePreparedIntegrityError("prepared store active-feature estimates do not match")
    if metadata.file_bytes > MAX_STORE_FILE_BYTES:
        raise NativePreparedIntegrityError("prepared store exceeds the 512 MiB session limit")
    if expected.result_bytes > MAX_RESULT_FILE_BYTES:
        raise NativePreparedIntegrityError("prepared result exceeds the 128 MiB session limit")
    if estimate.modeled_c_heap_bytes > MAX_MODELED_C_HEAP_BYTES:
        raise NativePreparedIntegrityError("prepared C-heap estimate exceeds the 512 MiB limit")
    if estimate.private_disk_scratch_bytes > MAX_PRIVATE_DISK_SCRATCH_BYTES:
        raise NativePreparedIntegrityError("prepared private scratch exceeds the 1 GiB limit")
    total_work = (
        expected.statistics_work_units + expected.solve_work_units + expected.score_work_units
    )
    if estimate.total_numeric_work_units != total_work:
        raise NativePreparedIntegrityError("prepared total-work estimate does not match")
    if total_work > MAX_TOTAL_NUMERIC_WORK_UNITS:
        raise NativePreparedIntegrityError("prepared numeric work exceeds the 200B-unit limit")
    return expected


class _MappedLifetime:
    __slots__ = ("_closed", "_descriptor", "_lock", "_mapping", "_workspace")

    def __init__(self, descriptor: int, mapping: mmap.mmap, workspace: Path) -> None:
        self._descriptor = descriptor
        self._mapping = mapping
        self._workspace = workspace
        self._closed = False
        self._lock = threading.RLock()

    def require_open(self) -> mmap.mmap:
        with self._lock:
            return self._require_open_unlocked()

    def _require_open_unlocked(self) -> mmap.mmap:
        if self._closed or self._mapping is None:
            raise NativePreparedClosedError("native prepared result is closed")
        return self._mapping

    def read_f64(self, offset: int) -> float:
        """Read one cell while excluding a concurrent close."""

        with self._lock:
            mapping = self._require_open_unlocked()
            return _F64.unpack_from(mapping, offset)[0]

    def read_f64_slice(
        self,
        base_offset: int,
        start: int,
        stop: int,
        step: int,
    ) -> tuple[float, ...]:
        """Materialize one requested slice while excluding a concurrent close."""

        with self._lock:
            mapping = self._require_open_unlocked()
            return tuple(
                _F64.unpack_from(mapping, base_offset + position * 8)[0]
                for position in range(start, stop, step)
            )

    def header_sha256_and_size(self) -> tuple[bytes, str, int]:
        """Snapshot the fixed header and rehash the mapping under one lock."""

        with self._lock:
            mapping = self._require_open_unlocked()
            return bytes(mapping[:RESULT_HEADER_BYTES]), _mapping_sha256(mapping), len(mapping)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._mapping is not None:
                try:
                    self._mapping.close()
                except (BufferError, OSError) as error:
                    # Keep every resource live so an exported view can be released
                    # and cleanup retried without losing the descriptor/workspace.
                    raise NativePreparedExecutionError(
                        f"cannot close native prepared result mapping: {error}"
                    ) from error
                self._mapping = None
            if self._descriptor is not None:
                descriptor = self._descriptor
                try:
                    os.close(descriptor)
                except OSError as error:
                    raise NativePreparedExecutionError(
                        f"cannot close native prepared result descriptor: {error}"
                    ) from error
                self._descriptor = None
            if self._workspace is not None:
                workspace = self._workspace
                try:
                    shutil.rmtree(workspace)
                except OSError as error:
                    raise NativePreparedExecutionError(
                        f"cannot remove native prepared result workspace: {error}"
                    ) from error
                self._workspace = None
            self._closed = True

    def discard_before_transfer(self) -> None:
        """Drop only the map while the adapter still owns fd and workspace."""

        with self._lock:
            if self._closed:
                return
            try:
                assert self._mapping is not None
                self._mapping.close()
            except (BufferError, OSError) as error:
                raise NativePreparedProtocolError(
                    f"cannot discard invalid native prepared mapping: {error}"
                ) from error
            self._mapping = None
            # The adapter's local variables retain ownership of these two resources.
            self._descriptor = None
            self._workspace = None
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class NativePreparedFloat64View(Sequence[float]):
    """A non-owning finite-f64 view whose lifetime is checked on every access."""

    _lifetime: _MappedLifetime
    _offset: int
    _count: int
    row_count: int
    column_count: int

    def __len__(self) -> int:
        self._lifetime.require_open()
        return self._count

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return self._lifetime.read_f64_slice(self._offset, start, stop, step)
        if not isinstance(index, int):
            raise TypeError("native prepared payload indices must be integers or slices")
        normalized = index + self._count if index < 0 else index
        if not 0 <= normalized < self._count:
            raise IndexError("native prepared payload index is out of range")
        return self._lifetime.read_f64(self._offset + normalized * 8)

    def __iter__(self) -> Iterator[float]:
        for position in range(self._count):
            yield self._lifetime.read_f64(self._offset + position * 8)

    def at(self, row_index: int, column_index: int) -> float:
        """Read one matrix cell without materializing its row."""

        if type(row_index) is not int or type(column_index) is not int:
            raise TypeError("row and column indices must be exact integers")
        if not 0 <= row_index < self.row_count or not 0 <= column_index < self.column_count:
            raise IndexError("native prepared matrix index is out of range")
        position = row_index * self.column_count + column_index
        return self._lifetime.read_f64(self._offset + position * 8)


@dataclass(frozen=True, slots=True)
class NativePreparedCoefficientRecord:
    subset_index: int
    subset_domain_mask: int
    training_row_count: int
    active_tag_mask: int
    active_feature_count: int
    record_payload_bytes: int
    continuous_means: NativePreparedFloat64View
    continuous_scales: NativePreparedFloat64View
    intercepts: NativePreparedFloat64View
    weights: NativePreparedFloat64View


@dataclass(frozen=True, slots=True)
class NativePreparedScoreRecord:
    block_index: int
    training_subset_index: int
    scored_domain_index: int
    row_count: int
    record_payload_bytes: int
    scores: NativePreparedFloat64View


class NativePreparedSessionResult:
    """Context-managed, mmap-backed native session result."""

    __slots__ = (
        "_coefficients",
        "_lifetime",
        "_metadata",
        "_ridge",
        "_scores",
        "binary_sha256",
        "request_nonce",
        "result_sha256",
        "store_sha256",
    )

    def __init__(
        self,
        *,
        lifetime: _MappedLifetime,
        metadata: PreparedStoreFileMetadata,
        ridge: float,
        request_nonce: bytes,
        store_sha256: str,
        binary_sha256: str,
        result_sha256: str,
        coefficients: tuple[NativePreparedCoefficientRecord, ...],
        scores: tuple[NativePreparedScoreRecord, ...],
    ) -> None:
        self._lifetime = lifetime
        self._metadata = metadata
        self._ridge = ridge
        self.request_nonce = request_nonce
        self.store_sha256 = store_sha256
        self.binary_sha256 = binary_sha256
        self.result_sha256 = result_sha256
        self._coefficients = coefficients
        self._scores = scores

    @property
    def metadata(self) -> PreparedStoreFileMetadata:
        self._lifetime.require_open()
        return self._metadata

    @property
    def ridge(self) -> float:
        self._lifetime.require_open()
        return self._ridge

    @property
    def coefficients(self) -> tuple[NativePreparedCoefficientRecord, ...]:
        self._lifetime.require_open()
        return self._coefficients

    @property
    def scores(self) -> tuple[NativePreparedScoreRecord, ...]:
        self._lifetime.require_open()
        return self._scores

    @property
    def closed(self) -> bool:
        try:
            self._lifetime.require_open()
        except NativePreparedClosedError:
            return True
        return False

    def close(self) -> None:
        self._lifetime.close()

    def verify_integrity(self) -> None:
        """Rebind object metadata and every view to the authenticated mmap.

        The adapter validates the protocol before constructing this object.  A later
        policy stage can be separated from that call by arbitrary user code, so it
        must not trust mutable private attributes or stale ``init=False``-style
        evidence.  This method rehashes the private read-only result and verifies the
        canonical record graph and exact view offsets without materializing numeric
        payloads.  It deliberately does not claim provenance for the caller-pinned
        executable or store.
        """

        if type(self._metadata) is not PreparedStoreFileMetadata:
            raise NativePreparedIntegrityError("native result metadata must be exact")
        metadata = self._metadata
        expected = _derive_expected_graph(metadata)
        preflight_native_prepared_session(metadata, ridge=self._ridge)
        for name, value in (
            ("store_sha256", self.store_sha256),
            ("binary_sha256", self.binary_sha256),
            ("result_sha256", self.result_sha256),
        ):
            _sha256_hex(value, name)
        if type(self.request_nonce) is not bytes or len(self.request_nonce) != 32:
            raise NativePreparedIntegrityError("native result request nonce must be 32 bytes")
        if not any(self.request_nonce):
            raise NativePreparedIntegrityError("native result request nonce must be nonzero")
        ridge = _positive_f64(self._ridge, "native result ridge")
        if struct.pack("<d", ridge) != struct.pack("<d", self._ridge):
            raise NativePreparedIntegrityError("native result ridge is not canonical")
        header, mapped_sha256, mapped_bytes = self._lifetime.header_sha256_and_size()
        if not hmac.compare_digest(mapped_sha256, self.result_sha256):
            raise NativePreparedIntegrityError("native result mapping SHA-256 changed")
        if mapped_bytes != expected.result_bytes:
            raise NativePreparedIntegrityError("native result mapping size changed")
        (
            magic,
            version,
            status_code,
            response_nonce,
            response_store_sha,
            response_binary_sha,
            domain_count,
            row_count,
            feature_count,
            target_count,
            coefficient_count,
            score_count,
            score_row_memberships,
            coefficient_section_bytes,
            score_section_bytes,
            result_bytes,
            ridge_echo,
            statistics_work_units,
            solve_work_units,
            score_work_units,
            modeled_c_heap_bytes,
            store_payload_sha,
            logical_store_sha,
            source_fit_sha,
            embedding_snapshot_sha,
            model_catalogue_sha,
            graph_identity_sha,
            authentication_validation_bytes_scanned,
            output_numeric_cells_validated,
            file_backed_input_bytes,
        ) = _RESULT_HEADER.unpack(header)
        if magic != RESULT_MAGIC or version != PROTOCOL_VERSION or status_code != STATUS_SUCCESS:
            raise NativePreparedIntegrityError("native result header is not canonical success")
        exact_bytes = (
            ("request nonce", response_nonce, self.request_nonce),
            ("store SHA-256", response_store_sha, bytes.fromhex(self.store_sha256)),
            ("binary SHA-256", response_binary_sha, bytes.fromhex(self.binary_sha256)),
            (
                "store payload SHA-256",
                store_payload_sha,
                bytes.fromhex(metadata.store_payload_sha256),
            ),
            (
                "logical store SHA-256",
                logical_store_sha,
                bytes.fromhex(metadata.logical_store_sha256),
            ),
            ("source-fit SHA-256", source_fit_sha, bytes.fromhex(metadata.source_fit_sha256)),
            (
                "embedding snapshot SHA-256",
                embedding_snapshot_sha,
                _optional_digest_bytes(metadata.embedding_snapshot_sha256),
            ),
            (
                "model catalogue SHA-256",
                model_catalogue_sha,
                bytes.fromhex(metadata.model_catalogue_sha256),
            ),
            (
                "graph identity SHA-256",
                graph_identity_sha,
                bytes.fromhex(metadata.graph_identity_sha256),
            ),
        )
        for name, actual, wanted in exact_bytes:
            if not hmac.compare_digest(actual, wanted):
                raise NativePreparedIntegrityError(f"native result header {name} changed")
        expected_numbers = (
            ("domain count", domain_count, metadata.domain_count),
            ("row count", row_count, metadata.row_count),
            ("feature count", feature_count, metadata.feature_count),
            ("target count", target_count, metadata.target_count),
            ("coefficient count", coefficient_count, len(expected.coefficients)),
            ("score count", score_count, len(expected.scores)),
            ("score row memberships", score_row_memberships, expected.score_row_memberships),
            (
                "coefficient section bytes",
                coefficient_section_bytes,
                expected.coefficient_section_bytes,
            ),
            ("score section bytes", score_section_bytes, expected.score_section_bytes),
            ("result bytes", result_bytes, expected.result_bytes),
            ("statistics work", statistics_work_units, expected.statistics_work_units),
            ("solve work", solve_work_units, expected.solve_work_units),
            ("score work", score_work_units, expected.score_work_units),
            ("modeled C heap", modeled_c_heap_bytes, metadata.estimate.modeled_c_heap_bytes),
            (
                "authentication scan",
                authentication_validation_bytes_scanned,
                metadata.estimate.authentication_validation_bytes_scanned,
            ),
            (
                "validated numeric cells",
                output_numeric_cells_validated,
                metadata.estimate.output_numeric_cells_validated,
            ),
            (
                "file-backed input bytes",
                file_backed_input_bytes,
                metadata.estimate.file_backed_input_bytes,
            ),
        )
        for name, actual, wanted in expected_numbers:
            if actual != wanted:
                raise NativePreparedIntegrityError(f"native result header {name} changed")
        if struct.pack("<d", ridge_echo) != struct.pack("<d", self._ridge):
            raise NativePreparedIntegrityError("native result header ridge changed")
        if type(self._coefficients) is not tuple or len(self._coefficients) != len(
            expected.coefficients
        ):
            raise NativePreparedIntegrityError(
                "native result coefficient records have the wrong canonical length"
            )
        if type(self._scores) is not tuple or len(self._scores) != len(expected.scores):
            raise NativePreparedIntegrityError(
                "native result score records have the wrong canonical length"
            )

        offset = RESULT_HEADER_BYTES
        for actual, wanted in zip(self._coefficients, expected.coefficients, strict=True):
            if type(actual) is not NativePreparedCoefficientRecord:
                raise NativePreparedIntegrityError(
                    "native coefficient records must have exact types"
                )
            actual_header = (
                actual.subset_index,
                actual.subset_domain_mask,
                actual.training_row_count,
                actual.active_tag_mask,
                actual.active_feature_count,
                actual.record_payload_bytes,
            )
            wanted_header = (
                wanted.subset_index,
                wanted.domain_mask,
                wanted.training_row_count,
                wanted.active_tag_mask,
                wanted.active_feature_count,
                wanted.payload_bytes,
            )
            if actual_header != wanted_header:
                raise NativePreparedIntegrityError(
                    "native coefficient record metadata changed after parsing"
                )
            payload_offset = offset + COEFFICIENT_RECORD_HEADER_BYTES
            intercept_offset = payload_offset + 48
            weight_offset = intercept_offset + 8 * metadata.target_count
            expected_views = (
                (actual.continuous_means, payload_offset, 3, 1, 3),
                (actual.continuous_scales, payload_offset + 24, 3, 1, 3),
                (
                    actual.intercepts,
                    intercept_offset,
                    metadata.target_count,
                    1,
                    metadata.target_count,
                ),
                (
                    actual.weights,
                    weight_offset,
                    metadata.target_count * wanted.active_feature_count,
                    metadata.target_count,
                    wanted.active_feature_count,
                ),
            )
            for view, view_offset, count, rows, columns in expected_views:
                if (
                    type(view) is not NativePreparedFloat64View
                    or view._lifetime is not self._lifetime
                    or view._offset != view_offset
                    or view._count != count
                    or view.row_count != rows
                    or view.column_count != columns
                ):
                    raise NativePreparedIntegrityError(
                        "native coefficient payload view changed after parsing"
                    )
            offset = payload_offset + wanted.payload_bytes

        for actual, wanted in zip(self._scores, expected.scores, strict=True):
            if type(actual) is not NativePreparedScoreRecord:
                raise NativePreparedIntegrityError("native score records must have exact types")
            actual_header = (
                actual.block_index,
                actual.training_subset_index,
                actual.scored_domain_index,
                actual.row_count,
                actual.record_payload_bytes,
            )
            wanted_header = (
                wanted.block_index,
                wanted.training_subset_index,
                wanted.scored_domain_index,
                wanted.row_count,
                wanted.payload_bytes,
            )
            if actual_header != wanted_header:
                raise NativePreparedIntegrityError(
                    "native score record metadata changed after parsing"
                )
            payload_offset = offset + SCORE_RECORD_HEADER_BYTES
            view = actual.scores
            if (
                type(view) is not NativePreparedFloat64View
                or view._lifetime is not self._lifetime
                or view._offset != payload_offset
                or view._count != wanted.row_count * metadata.target_count
                or view.row_count != wanted.row_count
                or view.column_count != metadata.target_count
            ):
                raise NativePreparedIntegrityError(
                    "native score payload view changed after parsing"
                )
            offset = payload_offset + wanted.payload_bytes
        if offset != expected.result_bytes:
            raise NativePreparedIntegrityError("native result record graph changed size")

    def __enter__(self) -> NativePreparedSessionResult:
        self._lifetime.require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, traceback
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise


@dataclass(frozen=True, slots=True)
class NativePreparedSessionAdapter:
    """Invoke one explicit, authenticated prepared-session sidecar."""

    binary_path: str | os.PathLike[str]
    expected_sha256: str
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "binary_path", _absolute_path(self.binary_path, "binary_path"))
        object.__setattr__(
            self,
            "expected_sha256",
            _sha256_hex(self.expected_sha256, "expected_sha256"),
        )
        object.__setattr__(self, "timeout_seconds", _timeout(self.timeout_seconds))

    def run(
        self,
        store_path: str | os.PathLike[str],
        receipt: PreparedStoreFileReceipt,
        *,
        ridge: object,
    ) -> NativePreparedSessionResult:
        """Run the entire prepared graph in exactly one child invocation."""

        source = Path(_absolute_path(store_path, "store_path"))
        if type(receipt) is not PreparedStoreFileReceipt:
            raise TypeError("receipt must be an exact PreparedStoreFileReceipt")
        ridge_value = _positive_f64(ridge, "ridge")
        workspace = Path(tempfile.mkdtemp(prefix="tierroute-native-prepared-"))
        request_descriptor: int | None = None
        response_descriptor: int | None = None
        keep_workspace = False
        try:
            _secure_workspace(workspace)
            executable = workspace / ("prepared.exe" if os.name == "nt" else "prepared")
            _snapshot_verified_binary(Path(self.binary_path), executable, self.expected_sha256)
            request_path = workspace / "request.bin"
            response_path = workspace / "result.bin"
            request_descriptor = _create_private_file(request_path)
            metadata = _copy_store_into_request(
                source,
                receipt,
                request_descriptor,
            )
            expected = preflight_native_prepared_session(metadata, ridge=ridge_value)
            request_nonce = _nonzero_request_nonce()
            header = _REQUEST_HEADER.pack(
                REQUEST_MAGIC,
                PROTOCOL_VERSION,
                REQUEST_FLAGS,
                request_nonce,
                bytes.fromhex(receipt.whole_file_sha256),
                bytes.fromhex(self.expected_sha256),
                REQUEST_HEADER_BYTES + metadata.file_bytes,
                expected.result_bytes,
                ridge_value,
                b"\0" * 24,
            )
            _finish_request(request_descriptor, header, metadata.file_bytes)
            response_descriptor = _create_private_file(response_path)
            returncode, stderr = _execute_bounded(
                executable,
                request_descriptor,
                response_descriptor,
                workspace,
                response_limit=expected.result_bytes,
                timeout_seconds=self.timeout_seconds,
            )
            os.fsync(response_descriptor)
            result = _load_result(
                response_path=response_path,
                response_descriptor=response_descriptor,
                workspace=workspace,
                metadata=metadata,
                expected=expected,
                ridge=ridge_value,
                request_nonce=request_nonce,
                store_sha256=receipt.whole_file_sha256,
                binary_sha256=self.expected_sha256,
                returncode=returncode,
                stderr=stderr,
            )
            response_descriptor = None  # ownership moved to result lifetime
            keep_workspace = True
            _close_fd(request_descriptor, NativePreparedExecutionError, "close request")
            request_descriptor = None
            for disposable in (request_path, executable):
                try:
                    disposable.unlink()
                except OSError:
                    # On Windows the process may release executable mappings a
                    # moment later.  The result lifetime still removes the private
                    # directory, so this is not an authentication boundary.
                    pass
            return result
        except NativePreparedError:
            raise
        except OSError as error:
            raise NativePreparedExecutionError(
                f"cannot manage native prepared session workspace: {error}"
            ) from error
        finally:
            if request_descriptor is not None:
                _close_fd(
                    request_descriptor,
                    NativePreparedExecutionError,
                    "close native prepared request",
                    preserve=True,
                )
            if response_descriptor is not None:
                _close_fd(
                    response_descriptor,
                    NativePreparedExecutionError,
                    "close native prepared result",
                    preserve=True,
                )
            if not keep_workspace:
                shutil.rmtree(workspace, ignore_errors=True)

    def execute(
        self,
        store_path: str | os.PathLike[str],
        receipt: PreparedStoreFileReceipt,
        *,
        ridge: object,
    ) -> NativePreparedSessionResult:
        """Alias for :meth:`run` for orchestration-oriented callers."""

        return self.run(store_path, receipt, ridge=ridge)


def _copy_store_into_request(
    source: Path,
    receipt: PreparedStoreFileReceipt,
    descriptor: int,
) -> PreparedStoreFileMetadata:
    try:
        duplicate = os.dup(descriptor)
        with os.fdopen(duplicate, "r+b", closefd=True) as destination:
            # Materialize only the fixed placeholder so the authenticated store
            # copier starts at exact EOF.  Its own fixed-header aggregate
            # preflight runs before it appends any large payload bytes.
            destination.write(b"\0" * REQUEST_HEADER_BYTES)
            destination.flush()
            metadata = copy_authenticated_prepared_store(source, receipt, destination)
            destination.flush()
            os.fsync(destination.fileno())
            return metadata
    except NativePreparedError:
        raise
    except Exception as error:
        raise NativePreparedIntegrityError(
            f"cannot authenticate and copy prepared store: {error}"
        ) from error


def _finish_request(descriptor: int, header: bytes, store_bytes: int) -> None:
    expected_size = REQUEST_HEADER_BYTES + store_bytes
    try:
        if os.fstat(descriptor).st_size != expected_size:
            raise NativePreparedIntegrityError(
                "private prepared request does not contain the exact authenticated store"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(descriptor, header)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if metadata.st_size != expected_size:
            raise NativePreparedIntegrityError("private prepared request length changed")
        if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise NativePreparedIntegrityError("private prepared request is not owner-only")
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise NativePreparedExecutionError(
            f"cannot finalize native prepared request: {error}"
        ) from error


def _positive_f64(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite positive binary64") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite positive binary64")
    return result


def _nonzero_request_nonce() -> bytes:
    """Match the C protocol's nonzero nonce contract by rejection sampling."""

    while True:
        nonce = secrets.token_bytes(32)
        if any(nonce):
            return nonce


def _absolute_path(value: str | os.PathLike[str], name: str) -> str:
    try:
        path = os.fspath(value)
    except TypeError as error:
        raise TypeError(f"{name} must be path-like") from error
    if isinstance(path, bytes):
        raise TypeError(f"{name} must decode to a string path")
    if not path or "\0" in path:
        raise ValueError(f"{name} must be nonempty and contain no NUL")
    if path.startswith(("//", "\\\\")) or not os.path.isabs(path):
        raise ValueError(f"{name} must be an absolute local path")
    return os.path.abspath(path)


def _sha256_hex(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("timeout_seconds must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be in (0, {MAX_TIMEOUT_SECONDS:g}]")
    return result


def _secure_workspace(directory: Path) -> None:
    try:
        os.chmod(directory, 0o700)
        metadata = directory.stat()
    except OSError as error:
        raise NativePreparedIntegrityError(
            f"cannot secure native prepared workspace: {error}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise NativePreparedIntegrityError("native prepared workspace is not a directory")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise NativePreparedIntegrityError("native prepared workspace is not owner-only")


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _nanosecond_time(details: os.stat_result, name: str) -> int:
    explicit = getattr(details, f"st_{name}_ns", None)
    if explicit is not None:
        return int(explicit)
    return int(getattr(details, f"st_{name}") * 1_000_000_000)


def _stable_file(
    first: os.stat_result,
    second: os.stat_result,
    *,
    compare_change_time: bool = True,
) -> bool:
    stable = first.st_size == second.st_size and _nanosecond_time(
        first, "mtime"
    ) == _nanosecond_time(second, "mtime")
    if compare_change_time:
        stable = stable and _nanosecond_time(first, "ctime") == _nanosecond_time(second, "ctime")
    return stable


def _is_reparse_point(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _snapshot_verified_binary(source: Path, destination: Path, expected_sha256: str) -> None:
    try:
        path_metadata = source.lstat()
    except OSError as error:
        raise NativePreparedIntegrityError(
            f"cannot inspect native prepared binary: {error}"
        ) from error
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or _is_reparse_point(path_metadata)
        or not stat.S_ISREG(path_metadata.st_mode)
    ):
        raise NativePreparedIntegrityError(
            "native prepared binary must be a regular non-symlink file"
        )
    if os.name != "nt" and path_metadata.st_mode & 0o111 == 0:
        raise NativePreparedIntegrityError("native prepared binary is not executable")
    if not 0 < path_metadata.st_size <= MAX_BINARY_BYTES:
        raise NativePreparedIntegrityError("native prepared binary exceeds its reviewed size")
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(source, source_flags)
        opened = os.fstat(source_fd)
        if (
            not _same_file(path_metadata, opened)
            or _is_reparse_point(opened)
            or not stat.S_ISREG(opened.st_mode)
        ):
            raise NativePreparedIntegrityError("native prepared binary changed before open")
        destination_fd = _create_private_file(destination)
        digest = hashlib.sha256()
        copied = 0
        while chunk := os.read(source_fd, _COPY_CHUNK_BYTES):
            copied += len(chunk)
            if copied > MAX_BINARY_BYTES:
                raise NativePreparedIntegrityError("native prepared binary grew while read")
            digest.update(chunk)
            _write_all(destination_fd, chunk)
        os.fsync(destination_fd)
        final_opened = os.fstat(source_fd)
        try:
            final_path = source.lstat()
        except OSError:
            raise NativePreparedIntegrityError(
                "native prepared binary path changed while authenticated"
            ) from None
        if (
            copied != opened.st_size
            or stat.S_ISLNK(final_path.st_mode)
            or _is_reparse_point(final_path)
            or not stat.S_ISREG(final_path.st_mode)
            or not _stable_file(opened, final_opened)
            or not _same_file(opened, final_path)
            or not _stable_file(
                opened,
                final_path,
                # Windows path stat and descriptor stat can disagree on
                # st_ctime_ns creation-time precision.  Keep the descriptor
                # check strict and bind the private executable copy by SHA-256.
                compare_change_time=os.name != "nt",
            )
        ):
            raise NativePreparedIntegrityError("native prepared binary changed while authenticated")
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise NativePreparedIntegrityError("native prepared binary SHA-256 does not match")
        snapshot = os.fstat(destination_fd)
        if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size != copied:
            raise NativePreparedIntegrityError("native prepared binary snapshot is invalid")
    except NativePreparedError:
        raise
    except OSError as error:
        raise NativePreparedIntegrityError(
            f"cannot authenticate native prepared binary: {error}"
        ) from error
    finally:
        if destination_fd is not None:
            _close_fd(
                destination_fd,
                NativePreparedIntegrityError,
                "close prepared binary snapshot",
                preserve=True,
            )
        if source_fd is not None:
            _close_fd(
                source_fd,
                NativePreparedIntegrityError,
                "close prepared binary source",
                preserve=True,
            )
    if os.name != "nt":
        try:
            os.chmod(destination, 0o500)
        except OSError as error:
            raise NativePreparedIntegrityError(
                f"cannot make native prepared snapshot executable: {error}"
            ) from error


def _create_private_file(path: Path) -> int:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        metadata = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            _close_fd(
                descriptor,
                NativePreparedExecutionError,
                "close private file after inspection failure",
                preserve=True,
            )
        raise NativePreparedExecutionError(
            f"cannot create private file {path.name}: {error}"
        ) from error
    assert descriptor is not None
    if not stat.S_ISREG(metadata.st_mode):
        _close_fd(descriptor, NativePreparedExecutionError, "close nonregular private file")
        raise NativePreparedIntegrityError("native prepared private file is not regular")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o600:
        _close_fd(descriptor, NativePreparedExecutionError, "close insecure private file")
        raise NativePreparedIntegrityError("native prepared private file is not owner-only")
    return descriptor


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write in native prepared private file")
        offset += written


def _close_fd(
    descriptor: int,
    error_type: type[NativePreparedError],
    action: str,
    *,
    preserve: bool = False,
) -> None:
    active = preserve and sys.exc_info()[0] is not None
    try:
        os.close(descriptor)
    except OSError as error:
        if not active:
            raise error_type(f"cannot {action}: {error}") from error


@dataclass(slots=True)
class _PipeDrain:
    stream: BinaryIO
    destination: BinaryIO | None
    limit: int
    overflow: threading.Event
    retained: bytearray
    error: Exception | None = None

    def run(self) -> None:
        total = 0
        try:
            while chunk := self.stream.read(_PIPE_CHUNK_BYTES):
                accepted = chunk[: max(0, self.limit - total)]
                if self.destination is not None:
                    if accepted:
                        self.destination.write(accepted)
                else:
                    self.retained.extend(accepted)
                total += len(chunk)
                if total > self.limit:
                    self.overflow.set()
            if self.destination is not None:
                self.destination.flush()
        except Exception as error:
            self.error = error
        finally:
            try:
                self.stream.close()
            except Exception as error:
                if self.error is None:
                    self.error = error


def _execute_bounded(
    executable: Path,
    request_descriptor: int,
    response_descriptor: int,
    directory: Path,
    *,
    response_limit: int,
    timeout_seconds: float,
) -> tuple[int, bytes]:
    response_stream: BinaryIO | None = None
    try:
        os.lseek(request_descriptor, 0, os.SEEK_SET)
        response_stream = os.fdopen(os.dup(response_descriptor), "wb", closefd=True)
        process = subprocess.Popen(
            [str(executable)],
            stdin=request_descriptor,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=directory,
            env=_restricted_environment(directory),
            close_fds=True,
            start_new_session=os.name != "nt",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
            ),
        )
    except OSError as error:
        if response_stream is not None:
            try:
                response_stream.close()
            except OSError:
                pass
        raise NativePreparedExecutionError(
            f"cannot start native prepared child: {error}"
        ) from error
    assert response_stream is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    stderr_bytes = bytearray()
    stdout_drain = _PipeDrain(
        process.stdout,
        response_stream,
        response_limit,
        stdout_overflow,
        bytearray(),
    )
    stderr_drain = _PipeDrain(
        process.stderr,
        None,
        MAX_STDERR_BYTES,
        stderr_overflow,
        stderr_bytes,
    )
    threads = (
        threading.Thread(target=stdout_drain.run, name="tierroute-prepared-stdout", daemon=True),
        threading.Thread(target=stderr_drain.run, name="tierroute-prepared-stderr", daemon=True),
    )
    try:
        for thread in threads:
            thread.start()
    except BaseException as error:
        _terminate_process(process)
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        try:
            response_stream.close()
        except OSError:
            pass
        if isinstance(error, Exception):
            raise NativePreparedExecutionError(
                f"cannot start native prepared output drains: {error}"
            ) from error
        raise

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while process.poll() is None:
        if (
            stdout_overflow.is_set()
            or stdout_drain.error is not None
            or stderr_drain.error is not None
        ):
            _terminate_process(process)
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            timed_out = True
            _terminate_process(process)
            break
        try:
            process.wait(timeout=min(remaining, 0.05))
        except subprocess.TimeoutExpired:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _terminate_process(process)
        raise NativePreparedExecutionError("native prepared child could not be reaped") from None
    for thread in threads:
        thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in threads):
        _terminate_process(process)
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except OSError:
                pass
        for thread in threads:
            thread.join(timeout=1.0)
    if timed_out:
        raise NativePreparedExecutionError(
            f"native prepared child timed out after {timeout_seconds:g} seconds"
        )
    if stdout_overflow.is_set():
        raise NativePreparedProtocolError(
            f"native prepared result exceeded the {response_limit}-byte bound"
        )
    drain_error = stdout_drain.error or stderr_drain.error
    if drain_error is not None:
        raise NativePreparedExecutionError(
            f"cannot capture bounded native prepared output: {drain_error}"
        ) from drain_error
    stderr = bytes(stderr_bytes)
    if stderr_overflow.is_set():
        stderr += _STDERR_TRUNCATION_MARKER
    assert process.returncode is not None
    return process.returncode, stderr


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass


def _restricted_environment(directory: Path) -> dict[str, str]:
    environment = {
        "HOME": str(directory),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "",
        "TMPDIR": str(directory),
        "TZ": "UTC",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if os.name == "nt":
        for name in ("COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR"):
            value = os.environ.get(name)
            if value is not None:
                environment[name] = value
        environment["TEMP"] = str(directory)
        environment["TMP"] = str(directory)
    return environment


def _load_result(
    *,
    response_path: Path,
    response_descriptor: int,
    workspace: Path,
    metadata: PreparedStoreFileMetadata,
    expected: _ExpectedGraph,
    ridge: float,
    request_nonce: bytes,
    store_sha256: str,
    binary_sha256: str,
    returncode: int,
    stderr: bytes,
) -> NativePreparedSessionResult:
    try:
        path_metadata = response_path.lstat()
        opened_metadata = os.fstat(response_descriptor)
    except OSError as error:
        raise NativePreparedProtocolError(
            f"cannot inspect original native prepared result: {error}"
        ) from error
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISREG(path_metadata.st_mode)
        or not _same_file(path_metadata, opened_metadata)
        or not _stable_file(
            path_metadata,
            opened_metadata,
            compare_change_time=os.name != "nt",
        )
    ):
        raise NativePreparedProtocolError("native prepared result path no longer names stdout")
    if os.name != "nt" and stat.S_IMODE(opened_metadata.st_mode) != 0o600:
        raise NativePreparedProtocolError("native prepared result is not owner-only")
    if opened_metadata.st_size < RESULT_HEADER_BYTES:
        if returncode != 0:
            raise NativePreparedExecutionError(
                f"native prepared child crashed with status {returncode}{_stderr_detail(stderr)}"
            )
        raise NativePreparedProtocolError("native prepared result is truncated before its header")
    if opened_metadata.st_size > expected.result_bytes:
        raise NativePreparedProtocolError(
            "native prepared result exceeds its admitted exact-size upper bound"
        )
    try:
        mapping = mmap.mmap(response_descriptor, 0, access=mmap.ACCESS_READ)
    except (OSError, ValueError) as error:
        raise NativePreparedProtocolError(f"cannot map native prepared result: {error}") from error
    lifetime = _MappedLifetime(response_descriptor, mapping, workspace)
    try:
        unpacked = _RESULT_HEADER.unpack_from(mapping)
        (
            magic,
            version,
            status_code,
            response_nonce,
            response_store_sha,
            response_binary_sha,
            domain_count,
            row_count,
            feature_count,
            target_count,
            coefficient_count,
            score_count,
            score_row_memberships,
            coefficient_section_bytes,
            score_section_bytes,
            result_bytes,
            ridge_echo,
            statistics_work_units,
            solve_work_units,
            score_work_units,
            modeled_c_heap_bytes,
            store_payload_sha,
            logical_store_sha,
            source_fit_sha,
            embedding_snapshot_sha,
            model_catalogue_sha,
            graph_identity_sha,
            authentication_validation_bytes_scanned,
            output_numeric_cells_validated,
            file_backed_input_bytes,
        ) = unpacked
        if magic != RESULT_MAGIC or version != PROTOCOL_VERSION:
            raise NativePreparedProtocolError("native prepared result magic/version is invalid")
        if status_code not in _KNOWN_STATUSES:
            raise NativePreparedProtocolError("native prepared result has an unknown status")
        exact_bytes = (
            ("request nonce", response_nonce, request_nonce),
            ("store SHA-256", response_store_sha, bytes.fromhex(store_sha256)),
            ("binary SHA-256", response_binary_sha, bytes.fromhex(binary_sha256)),
            (
                "store payload SHA-256",
                store_payload_sha,
                bytes.fromhex(metadata.store_payload_sha256),
            ),
            (
                "logical store SHA-256",
                logical_store_sha,
                bytes.fromhex(metadata.logical_store_sha256),
            ),
            ("source-fit SHA-256", source_fit_sha, bytes.fromhex(metadata.source_fit_sha256)),
            (
                "embedding snapshot SHA-256",
                embedding_snapshot_sha,
                _optional_digest_bytes(metadata.embedding_snapshot_sha256),
            ),
            (
                "model catalogue SHA-256",
                model_catalogue_sha,
                bytes.fromhex(metadata.model_catalogue_sha256),
            ),
            (
                "graph identity SHA-256",
                graph_identity_sha,
                bytes.fromhex(metadata.graph_identity_sha256),
            ),
        )
        for name, actual, wanted in exact_bytes:
            if not hmac.compare_digest(actual, wanted):
                raise NativePreparedProtocolError(f"native prepared result {name} does not match")
        shape_numbers = (
            ("domain count", domain_count, metadata.domain_count),
            ("row count", row_count, metadata.row_count),
            ("feature count", feature_count, metadata.feature_count),
            ("target count", target_count, metadata.target_count),
        )
        for name, actual, wanted in shape_numbers:
            if actual != wanted:
                raise NativePreparedProtocolError(f"native prepared result {name} does not match")
        if struct.pack("<d", ridge_echo) != struct.pack("<d", ridge):
            raise NativePreparedProtocolError("native prepared result ridge does not match")

        if status_code != STATUS_SUCCESS:
            if returncode == 0:
                raise NativePreparedProtocolError(
                    "native prepared child returned failure status with successful exit"
                )
            if (
                opened_metadata.st_size != RESULT_HEADER_BYTES
                or result_bytes != RESULT_HEADER_BYTES
            ):
                raise NativePreparedProtocolError(
                    "native prepared failure result contains a payload"
                )
            raise NativePreparedStatusError(status_code, stderr)
        if returncode != 0:
            raise NativePreparedProtocolError(
                "native prepared child returned success with "
                f"nonzero process status {returncode}{_stderr_detail(stderr)}"
            )
        success_resource_numbers = (
            ("statistics work", statistics_work_units, expected.statistics_work_units),
            ("solve work", solve_work_units, expected.solve_work_units),
            ("score work", score_work_units, expected.score_work_units),
            (
                "modeled C heap",
                modeled_c_heap_bytes,
                metadata.estimate.modeled_c_heap_bytes,
            ),
            (
                "authentication/validation scan",
                authentication_validation_bytes_scanned,
                metadata.estimate.authentication_validation_bytes_scanned,
            ),
            (
                "output numeric-cell validation",
                output_numeric_cells_validated,
                metadata.estimate.output_numeric_cells_validated,
            ),
            (
                "file-backed input bytes",
                file_backed_input_bytes,
                metadata.estimate.file_backed_input_bytes,
            ),
        )
        for name, actual, wanted in success_resource_numbers:
            if actual != wanted:
                raise NativePreparedProtocolError(f"native prepared result {name} does not match")
        expected_header_numbers = (
            ("coefficient count", coefficient_count, len(expected.coefficients)),
            ("score count", score_count, len(expected.scores)),
            ("score memberships", score_row_memberships, expected.score_row_memberships),
            (
                "coefficient section bytes",
                coefficient_section_bytes,
                expected.coefficient_section_bytes,
            ),
            ("score section bytes", score_section_bytes, expected.score_section_bytes),
            ("result bytes", result_bytes, expected.result_bytes),
            ("file bytes", opened_metadata.st_size, expected.result_bytes),
        )
        for name, actual, wanted in expected_header_numbers:
            if actual != wanted:
                raise NativePreparedProtocolError(f"native prepared result {name} does not match")
        coefficients, offset = _parse_coefficients(
            lifetime,
            mapping,
            RESULT_HEADER_BYTES,
            metadata.target_count,
            expected.coefficients,
        )
        if offset != RESULT_HEADER_BYTES + expected.coefficient_section_bytes:
            raise NativePreparedProtocolError("native prepared coefficient section length changed")
        scores, offset = _parse_scores(
            lifetime,
            mapping,
            offset,
            metadata.target_count,
            expected.scores,
        )
        if offset != expected.result_bytes:
            raise NativePreparedProtocolError("native prepared score section has trailing bytes")
        # Hash the exact mapped bytes first, then repeat descriptor/path stability
        # checks so this root credential cannot describe a mid-hash mutation.
        result_sha256 = _mapping_sha256(mapping)
        final_opened = os.fstat(response_descriptor)
        final_path = response_path.lstat()
        if (
            not _stable_file(opened_metadata, final_opened)
            or not _same_file(opened_metadata, final_path)
            or not _stable_file(
                opened_metadata,
                final_path,
                compare_change_time=os.name != "nt",
            )
        ):
            raise NativePreparedProtocolError("native prepared result changed during validation")
        return NativePreparedSessionResult(
            lifetime=lifetime,
            metadata=metadata,
            ridge=ridge,
            request_nonce=request_nonce,
            store_sha256=store_sha256,
            binary_sha256=binary_sha256,
            result_sha256=result_sha256,
            coefficients=coefficients,
            scores=scores,
        )
    except BaseException:
        try:
            lifetime.discard_before_transfer()
        except NativePreparedProtocolError:
            pass
        raise


def _parse_coefficients(
    lifetime: _MappedLifetime,
    mapping: mmap.mmap,
    offset: int,
    target_count: int,
    expected_records: tuple[_ExpectedCoefficient, ...],
) -> tuple[tuple[NativePreparedCoefficientRecord, ...], int]:
    records: list[NativePreparedCoefficientRecord] = []
    for expected in expected_records:
        if offset + COEFFICIENT_RECORD_HEADER_BYTES > len(mapping):
            raise NativePreparedProtocolError("native prepared coefficient header is truncated")
        (
            subset_index,
            reserved,
            domain_mask,
            training_rows,
            active_tag_mask,
            active_feature_count,
            payload_bytes,
        ) = _COEFFICIENT_HEADER.unpack_from(mapping, offset)
        actual = (
            subset_index,
            domain_mask,
            training_rows,
            active_tag_mask,
            active_feature_count,
            payload_bytes,
        )
        wanted = (
            expected.subset_index,
            expected.domain_mask,
            expected.training_row_count,
            expected.active_tag_mask,
            expected.active_feature_count,
            expected.payload_bytes,
        )
        if reserved != 0 or actual != wanted:
            raise NativePreparedProtocolError(
                f"native prepared coefficient record {expected.subset_index} is not canonical"
            )
        payload_offset = offset + COEFFICIENT_RECORD_HEADER_BYTES
        payload_end = payload_offset + payload_bytes
        if payload_end > len(mapping):
            raise NativePreparedProtocolError("native prepared coefficient payload is truncated")
        _validate_f64_payload(mapping, payload_offset, payload_end, "coefficient")
        means = NativePreparedFloat64View(lifetime, payload_offset, 3, 1, 3)
        scales = NativePreparedFloat64View(lifetime, payload_offset + 24, 3, 1, 3)
        if any(value < 0.0 for value in means):
            raise NativePreparedProtocolError("native prepared continuous mean is negative")
        if any(value <= 0.0 for value in scales):
            raise NativePreparedProtocolError("native prepared continuous scale is not positive")
        intercept_offset = payload_offset + 48
        weight_offset = intercept_offset + 8 * target_count
        records.append(
            NativePreparedCoefficientRecord(
                subset_index=subset_index,
                subset_domain_mask=domain_mask,
                training_row_count=training_rows,
                active_tag_mask=active_tag_mask,
                active_feature_count=active_feature_count,
                record_payload_bytes=payload_bytes,
                continuous_means=means,
                continuous_scales=scales,
                intercepts=NativePreparedFloat64View(
                    lifetime, intercept_offset, target_count, 1, target_count
                ),
                weights=NativePreparedFloat64View(
                    lifetime,
                    weight_offset,
                    target_count * active_feature_count,
                    target_count,
                    active_feature_count,
                ),
            )
        )
        offset = payload_end
    return tuple(records), offset


def _parse_scores(
    lifetime: _MappedLifetime,
    mapping: mmap.mmap,
    offset: int,
    target_count: int,
    expected_records: tuple[_ExpectedScore, ...],
) -> tuple[tuple[NativePreparedScoreRecord, ...], int]:
    records: list[NativePreparedScoreRecord] = []
    for expected in expected_records:
        if offset + SCORE_RECORD_HEADER_BYTES > len(mapping):
            raise NativePreparedProtocolError("native prepared score header is truncated")
        (
            block_index,
            subset_index,
            domain_index,
            reserved,
            row_count,
            payload_bytes,
        ) = _SCORE_HEADER.unpack_from(mapping, offset)
        actual = (block_index, subset_index, domain_index, row_count, payload_bytes)
        wanted = (
            expected.block_index,
            expected.training_subset_index,
            expected.scored_domain_index,
            expected.row_count,
            expected.payload_bytes,
        )
        if reserved != 0 or actual != wanted:
            raise NativePreparedProtocolError(
                f"native prepared score record {expected.block_index} is not canonical"
            )
        payload_offset = offset + SCORE_RECORD_HEADER_BYTES
        payload_end = payload_offset + payload_bytes
        if payload_end > len(mapping):
            raise NativePreparedProtocolError("native prepared score payload is truncated")
        _validate_f64_payload(mapping, payload_offset, payload_end, "score")
        records.append(
            NativePreparedScoreRecord(
                block_index=block_index,
                training_subset_index=subset_index,
                scored_domain_index=domain_index,
                row_count=row_count,
                record_payload_bytes=payload_bytes,
                scores=NativePreparedFloat64View(
                    lifetime,
                    payload_offset,
                    row_count * target_count,
                    row_count,
                    target_count,
                ),
            )
        )
        offset = payload_end
    return tuple(records), offset


def _validate_f64_payload(mapping: mmap.mmap, start: int, end: int, label: str) -> None:
    view = memoryview(mapping)[start:end]
    try:
        for (value,) in struct.iter_unpack("<d", view):
            if not math.isfinite(value):
                raise NativePreparedProtocolError(
                    f"native prepared {label} payload contains a non-finite value"
                )
            if value == 0.0 and math.copysign(1.0, value) < 0.0:
                raise NativePreparedProtocolError(
                    f"native prepared {label} payload contains negative zero"
                )
    finally:
        view.release()


def _optional_digest_bytes(value: str | None) -> bytes:
    return b"\0" * 32 if value is None else bytes.fromhex(value)


def _mapping_sha256(mapping: mmap.mmap) -> str:
    digest = hashlib.sha256()
    view = memoryview(mapping)
    try:
        for offset in range(0, len(view), _COPY_CHUNK_BYTES):
            digest.update(view[offset : offset + _COPY_CHUNK_BYTES])
    finally:
        view.release()
    return digest.hexdigest()


def _stderr_detail(stderr: bytes) -> str:
    if not stderr:
        return ""
    detail = stderr.decode("utf-8", errors="replace").strip()
    return f": {detail}" if detail else ""


__all__ = [
    "COEFFICIENT_RECORD_HEADER_BYTES",
    "MAX_BINARY_BYTES",
    "MAX_MODELED_C_HEAP_BYTES",
    "MAX_PRIVATE_DISK_SCRATCH_BYTES",
    "MAX_RESULT_FILE_BYTES",
    "MAX_STDERR_BYTES",
    "MAX_STORE_FILE_BYTES",
    "MAX_TIMEOUT_SECONDS",
    "MAX_TOTAL_NUMERIC_WORK_UNITS",
    "PREPARED_MOMENT_SOLVER_ID",
    "PREPARED_RAW_SCORER_ID",
    "PREPARED_SESSION_ENGINE_ID",
    "PREPARED_SESSION_RESULT_ID",
    "REQUEST_HEADER_BYTES",
    "RESULT_HEADER_BYTES",
    "SCORE_RECORD_HEADER_BYTES",
    "NativePreparedClosedError",
    "NativePreparedCoefficientRecord",
    "NativePreparedError",
    "NativePreparedExecutionError",
    "NativePreparedFloat64View",
    "NativePreparedIntegrityError",
    "NativePreparedProtocolError",
    "NativePreparedScoreRecord",
    "NativePreparedSessionAdapter",
    "NativePreparedSessionResult",
    "NativePreparedStatusError",
    "preflight_native_prepared_session",
]
