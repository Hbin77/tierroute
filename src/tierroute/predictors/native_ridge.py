# SPDX-License-Identifier: Apache-2.0
"""Fail-closed adapter for the project-owned dense C11 ridge sidecar.

The sidecar is a training-time accelerator, not a runtime dependency. Callers
must name an absolute executable and its exact SHA-256 digest; this module does
not search ``PATH`` or download artifacts. A verified snapshot is executed from
a private temporary directory so a later path replacement cannot change the
bytes that run.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import secrets
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import BinaryIO, ClassVar, Protocol

from tierroute.predictors._ridge import RidgeSolution
from tierroute.predictors.solvers import NATIVE_C11_RIDGE_SOLVER_ID

REQUEST_MAGIC = b"TRRIDG01"
RESPONSE_MAGIC = b"TRRRES01"
PROTOCOL_VERSION = 1
REQUEST_FLAGS = 0

STATUS_SUCCESS = 0
STATUS_PROTOCOL_ERROR = 1
STATUS_RESOURCE_ERROR = 2
STATUS_NUMERIC_ERROR = 3
STATUS_ALLOCATION_ERROR = 4
STATUS_SOLVE_ERROR = 5
STATUS_INTERNAL_ERROR = 6

_KNOWN_STATUSES = frozenset(range(STATUS_SUCCESS, STATUS_INTERNAL_ERROR + 1))
_STATUS_NAMES = {
    STATUS_PROTOCOL_ERROR: "protocol/header",
    STATUS_RESOURCE_ERROR: "bounds/resource",
    STATUS_NUMERIC_ERROR: "numeric/input",
    STATUS_ALLOCATION_ERROR: "allocation",
    STATUS_SOLVE_ERROR: "solve/residual",
    STATUS_INTERNAL_ERROR: "I/O/internal",
}

# Keep these in sync with native/tierroute_ridge.c. The byte limit admits the
# intended RouterBench matrix while bounding accidental multi-gigabyte input.
MAX_SAMPLE_COUNT = 1_000_000
MAX_FEATURE_COUNT = 4_096
MAX_TARGET_COUNT = 256
MAX_BINARY_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 4 * 1024 * 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ALLOCATION_BYTES = 2 * 1024 * 1024 * 1024
MAX_WORK_UNITS = 32_000_000_000
MAX_STDERR_BYTES = 16 * 1024
MAX_TIMEOUT_SECONDS = 600.0

_REQUEST_HEADER = struct.Struct("<8sII32sQQQd")
_RESPONSE_HEADER = struct.Struct("<8sII32sQQ")
_COPY_CHUNK_BYTES = 1024 * 1024
_PIPE_CHUNK_BYTES = 64 * 1024


class NativeRidgeError(RuntimeError):
    """Base class for native-adapter failures."""


class NativeRidgeIntegrityError(NativeRidgeError):
    """The configured executable failed its identity contract."""


class NativeRidgeExecutionError(NativeRidgeError):
    """The authenticated executable failed its process boundary."""


class NativeRidgeProtocolError(NativeRidgeError):
    """The sidecar emitted a malformed or unauthenticated response."""


class NativeRidgeStatusError(NativeRidgeError):
    """The sidecar returned a known non-success status."""

    def __init__(self, status: int, stderr: bytes) -> None:
        self.status = status
        self.stderr = stderr
        detail = _stderr_detail(stderr)
        super().__init__(
            f"native ridge failed with status {status} ({_STATUS_NAMES[status]}){detail}"
        )


class _Digest(Protocol):
    def update(self, payload: bytes, /) -> object:
        """Add bytes to the digest state."""

    def hexdigest(self) -> str:
        """Return lowercase hexadecimal digest text."""


@dataclass(frozen=True, slots=True)
class NativeRidgeAdapter:
    """Invoke one explicitly authenticated dense-ridge sidecar."""

    solver_id: ClassVar[str] = NATIVE_C11_RIDGE_SOLVER_ID

    binary_path: str | os.PathLike[str]
    expected_sha256: str
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "binary_path", _validate_binary_path(self.binary_path))
        object.__setattr__(self, "expected_sha256", _validate_sha256(self.expected_sha256))
        object.__setattr__(self, "timeout_seconds", _validate_timeout(self.timeout_seconds))

    def preflight(
        self,
        *,
        sample_count: int,
        feature_count: int,
        target_count: int,
    ) -> None:
        """Reject work and authenticate the configured binary before materialization.

        The solve path repeats binary authentication while making its private
        executable snapshot. The second check closes the preflight-to-execution
        replacement window instead of treating this early check as a TOCTOU-safe
        execution credential.
        """

        preflight_native_ridge(
            sample_count=sample_count,
            feature_count=feature_count,
            target_count=target_count,
        )
        _verify_binary(Path(self.binary_path), self.expected_sha256)

    def solve(
        self,
        feature_rows: Sequence[Sequence[float]],
        target_columns: Sequence[Sequence[float]],
        *,
        ridge: float,
    ) -> RidgeSolution:
        """Solve centered ridge through the canonical little-endian protocol."""

        n, d, m, ridge_value = _validate_shape(feature_rows, target_columns, ridge)
        request_size = _request_size(n, d, m)
        response_size = _success_response_size(d, m)
        request_id = secrets.token_bytes(32)
        try:
            with tempfile.TemporaryDirectory(prefix="tierroute-native-ridge-") as name:
                directory = Path(name)
                _make_private_directory(directory)
                executable = directory / ("ridge.exe" if os.name == "nt" else "ridge")
                _snapshot_verified_binary(Path(self.binary_path), executable, self.expected_sha256)
                request_path = directory / "request.bin"
                response_path = directory / "response.bin"
                _write_request(
                    request_path,
                    request_id=request_id,
                    feature_rows=feature_rows,
                    target_columns=target_columns,
                    ridge=ridge_value,
                    expected_size=request_size,
                    feature_count=d,
                    target_count=m,
                )
                returncode, stderr, response_descriptor = _execute_bounded(
                    executable,
                    request_path,
                    response_path,
                    directory,
                    response_limit=response_size,
                    timeout_seconds=self.timeout_seconds,
                )
                try:
                    return _read_response(
                        response_path,
                        response_descriptor=response_descriptor,
                        request_id=request_id,
                        feature_count=d,
                        target_count=m,
                        stderr=stderr,
                        returncode=returncode,
                    )
                finally:
                    _close_descriptor(
                        response_descriptor,
                        error_type=NativeRidgeProtocolError,
                        action="cannot close the original native ridge stdout safely",
                    )
        except OSError as error:
            # All expected adapter-owned filesystem calls are wrapped closer to
            # their trust boundary. This final guard covers temporary-directory
            # creation/cleanup without converting numeric input exceptions.
            raise NativeRidgeExecutionError(
                f"cannot manage the native ridge private workspace: {error}"
            ) from error


def _validate_binary_path(path: str | os.PathLike[str]) -> str:
    try:
        value = os.fspath(path)
    except TypeError as error:
        raise TypeError("binary_path must be a path-like value") from error
    if isinstance(value, bytes):
        raise TypeError("binary_path must decode to a string path")
    if not value or "\x00" in value:
        raise ValueError("binary_path must be a non-empty path without NUL bytes")
    if value.startswith(("//", "\\\\")):
        raise ValueError("binary_path must not use a UNC or device-style path")
    if not os.path.isabs(value):
        raise ValueError("binary_path must be absolute")
    return os.path.abspath(value)


def _validate_sha256(digest: object) -> str:
    if not isinstance(digest, str):
        raise TypeError("expected_sha256 must be a string")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("expected_sha256 must be exactly 64 lowercase hexadecimal characters")
    return digest


def _validate_timeout(timeout_seconds: object) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
        raise TypeError("timeout_seconds must be a finite real number")
    value = float(timeout_seconds)
    if not math.isfinite(value) or value <= 0.0 or value > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be in (0, {MAX_TIMEOUT_SECONDS:g}]")
    return value


def _validate_shape(
    feature_rows: Sequence[Sequence[float]],
    target_columns: Sequence[Sequence[float]],
    ridge: object,
) -> tuple[int, int, int, float]:
    if isinstance(feature_rows, (str, bytes, bytearray)):
        raise TypeError("feature_rows must be a sequence of numeric sequences")
    if isinstance(target_columns, (str, bytes, bytearray)):
        raise TypeError("target_columns must be a sequence of numeric sequences")
    try:
        n = len(feature_rows)
        m = len(target_columns)
    except TypeError as error:
        raise TypeError("feature_rows and target_columns must be sized sequences") from error
    if n < 1 or n > MAX_SAMPLE_COUNT:
        raise ValueError(f"sample_count must be in [1, {MAX_SAMPLE_COUNT}]")
    if m < 1 or m > MAX_TARGET_COUNT:
        raise ValueError(f"target_count must be in [1, {MAX_TARGET_COUNT}]")
    try:
        first_row = feature_rows[0]
        d = len(first_row)
    except (IndexError, TypeError) as error:
        raise TypeError("feature_rows[0] must be a sized numeric sequence") from error
    if isinstance(first_row, (str, bytes, bytearray)):
        raise TypeError("feature_rows[0] must be a numeric sequence")
    if d < 1 or d > MAX_FEATURE_COUNT:
        raise ValueError(f"feature_count must be in [1, {MAX_FEATURE_COUNT}]")
    preflight_native_ridge(sample_count=n, feature_count=d, target_count=m)
    for index, row in enumerate(feature_rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"feature_rows[{index}] must be a numeric sequence")
        try:
            width = len(row)
        except TypeError as error:
            raise TypeError(f"feature_rows[{index}] must be a sized sequence") from error
        if width != d:
            raise ValueError(f"feature_rows must be rectangular with width {d}")
    for index, column in enumerate(target_columns):
        if isinstance(column, (str, bytes, bytearray)):
            raise TypeError(f"target_columns[{index}] must be a numeric sequence")
        try:
            width = len(column)
        except TypeError as error:
            raise TypeError(f"target_columns[{index}] must be a sized sequence") from error
        if width != n:
            raise ValueError(f"target_columns must have width {n}")
    ridge_value = _finite_real(ridge, location="ridge")
    if ridge_value <= 0.0:
        raise ValueError("ridge must be positive")
    return n, d, m, ridge_value


def preflight_native_ridge(*, sample_count: int, feature_count: int, target_count: int) -> None:
    """Reject unreviewed dense-sidecar work before request serialization."""

    counts = {
        "sample_count": (sample_count, MAX_SAMPLE_COUNT),
        "feature_count": (feature_count, MAX_FEATURE_COUNT),
        "target_count": (target_count, MAX_TARGET_COUNT),
    }
    for name, (value, maximum) in counts.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 1 or value > maximum:
            raise ValueError(f"{name} must be in [1, {maximum}]")
    _request_size(sample_count, feature_count, target_count)
    _success_response_size(feature_count, target_count)
    allocation_bytes = _native_allocation_bytes(sample_count, feature_count, target_count)
    if allocation_bytes > MAX_ALLOCATION_BYTES:
        raise ValueError(
            "native ridge allocation estimate "
            f"{allocation_bytes} exceeds limit {MAX_ALLOCATION_BYTES}"
        )
    work = _native_work_units(sample_count, feature_count, target_count)
    if work > MAX_WORK_UNITS:
        raise ValueError(f"native ridge work estimate {work} exceeds limit {MAX_WORK_UNITS}")


def _native_work_units(n: int, d: int, m: int) -> int:
    """Mirror the reviewed, checked C11 operation-count boundary."""

    return 3 * n * (d + m) + n * d * (d + 1) // 2 + n * d * m + d**3 + 2 * m * d**2 + m * d


def _native_allocation_bytes(n: int, d: int, m: int) -> int:
    """Mirror every heap-allocated binary64 array in the C11 implementation."""

    scalar_count = n * d + n * m + 2 * d * d + 2 * m * d + 2 * d + 3 * m
    return 8 * scalar_count


def _request_size(n: int, d: int, m: int) -> int:
    size = _REQUEST_HEADER.size + 8 * n * (d + m)
    if size > MAX_REQUEST_BYTES:
        raise ValueError(f"canonical request would use {size} bytes; limit is {MAX_REQUEST_BYTES}")
    return size


def _success_response_size(d: int, m: int) -> int:
    size = _RESPONSE_HEADER.size + 8 * m * (d + 1)
    if size > MAX_RESPONSE_BYTES:
        raise ValueError(
            f"canonical response would use {size} bytes; limit is {MAX_RESPONSE_BYTES}"
        )
    return size


def _finite_real(value: object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{location} must be a real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{location} must be representable as a finite float") from error
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be finite")
    return converted


def _make_private_directory(directory: Path) -> None:
    try:
        os.chmod(directory, stat.S_IRWXU)
        metadata = directory.stat()
    except OSError as error:
        raise NativeRidgeIntegrityError(
            f"cannot secure the native ridge temporary workspace: {error}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise NativeRidgeIntegrityError("temporary workspace is not a directory")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != stat.S_IRWXU:
        raise NativeRidgeIntegrityError("temporary workspace is not owner-only")


def _require_reviewed_binary_size(metadata: os.stat_result) -> None:
    if metadata.st_size < 1:
        raise NativeRidgeIntegrityError("native ridge binary must not be empty")
    if metadata.st_size > MAX_BINARY_BYTES:
        raise NativeRidgeIntegrityError(
            f"native ridge binary size {metadata.st_size} exceeds reviewed limit {MAX_BINARY_BYTES}"
        )


def _inspect_binary_path(source: Path) -> os.stat_result:
    try:
        metadata = source.lstat()
    except OSError as error:
        raise NativeRidgeIntegrityError(f"cannot inspect native ridge binary: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise NativeRidgeIntegrityError("native ridge binary must be a regular non-symlink file")
    if os.name != "nt" and metadata.st_mode & 0o111 == 0:
        raise NativeRidgeIntegrityError("native ridge binary is not executable")
    _require_reviewed_binary_size(metadata)
    return metadata


def _open_binary(source: Path, path_metadata: os.stat_result) -> tuple[int, os.stat_result]:
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, source_flags)
    except OSError as error:
        raise NativeRidgeIntegrityError(
            f"cannot open native ridge binary safely: {error}"
        ) from error
    try:
        opened_metadata = os.fstat(source_fd)
        _require_same_file(path_metadata, opened_metadata)
        _require_reviewed_binary_size(opened_metadata)
    except OSError as error:
        _close_descriptor(
            source_fd,
            error_type=NativeRidgeIntegrityError,
            action="cannot close native ridge binary after inspection failure",
        )
        raise NativeRidgeIntegrityError(
            f"cannot inspect the opened native ridge binary safely: {error}"
        ) from error
    except BaseException:
        _close_descriptor(
            source_fd,
            error_type=NativeRidgeIntegrityError,
            action="cannot close native ridge binary after inspection failure",
        )
        raise
    return source_fd, opened_metadata


def _stream_binary(
    source_fd: int,
    digest: _Digest,
    *,
    destination_fd: int | None,
) -> int:
    copied_bytes = 0
    while chunk := os.read(source_fd, _COPY_CHUNK_BYTES):
        copied_bytes += len(chunk)
        if copied_bytes > MAX_BINARY_BYTES:
            raise NativeRidgeIntegrityError(
                "native ridge binary exceeded the reviewed size limit while read"
            )
        digest.update(chunk)
        if destination_fd is not None:
            _write_all(destination_fd, chunk)
    return copied_bytes


def _finish_binary_authentication(
    source: Path,
    source_fd: int,
    path_metadata: os.stat_result,
    opened_metadata: os.stat_result,
    copied_bytes: int,
    digest: _Digest,
    expected_sha256: str,
) -> None:
    final_metadata = os.fstat(source_fd)
    _require_stable_file(opened_metadata, final_metadata)
    _require_reviewed_binary_size(final_metadata)
    if copied_bytes != opened_metadata.st_size:
        raise NativeRidgeIntegrityError(
            "native ridge binary byte count changed while it was authenticated"
        )
    try:
        final_path_metadata = source.lstat()
        _require_same_file(path_metadata, final_path_metadata)
        _require_same_file(opened_metadata, final_path_metadata)
        _require_stable_file(opened_metadata, final_path_metadata)
        _require_reviewed_binary_size(final_path_metadata)
    except (OSError, NativeRidgeIntegrityError):
        raise NativeRidgeIntegrityError(
            "native ridge binary changed while it was verified"
        ) from None
    if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
        raise NativeRidgeIntegrityError("native ridge binary SHA-256 does not match")


def _verify_binary(source: Path, expected_sha256: str) -> None:
    """Authenticate one bounded descriptor without creating an executable copy."""

    path_metadata = _inspect_binary_path(source)
    source_fd, opened_metadata = _open_binary(source, path_metadata)
    digest = hashlib.sha256()
    try:
        try:
            copied_bytes = _stream_binary(source_fd, digest, destination_fd=None)
            _finish_binary_authentication(
                source,
                source_fd,
                path_metadata,
                opened_metadata,
                copied_bytes,
                digest,
                expected_sha256,
            )
        finally:
            _close_descriptor(
                source_fd,
                error_type=NativeRidgeIntegrityError,
                action="cannot close authenticated native ridge binary safely",
            )
    except OSError as error:
        raise NativeRidgeIntegrityError(
            f"cannot authenticate native ridge binary safely: {error}"
        ) from error


def _snapshot_verified_binary(source: Path, destination: Path, expected_sha256: str) -> None:
    path_metadata = _inspect_binary_path(source)
    source_fd, opened_metadata = _open_binary(source, path_metadata)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        destination_fd = os.open(destination, destination_flags, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as error:
        _close_descriptor(
            source_fd,
            error_type=NativeRidgeIntegrityError,
            action="cannot close native ridge binary after snapshot creation failure",
        )
        raise NativeRidgeIntegrityError(
            f"cannot snapshot native ridge binary safely: {error}"
        ) from error
    digest = hashlib.sha256()
    try:
        copied_bytes = _stream_binary(source_fd, digest, destination_fd=destination_fd)
        os.fsync(destination_fd)
        destination_metadata = os.fstat(destination_fd)
        if not stat.S_ISREG(destination_metadata.st_mode) or destination_metadata.st_size != (
            copied_bytes
        ):
            raise NativeRidgeIntegrityError(
                "native ridge executable snapshot has an unexpected type or size"
            )
        _finish_binary_authentication(
            source,
            source_fd,
            path_metadata,
            opened_metadata,
            copied_bytes,
            digest,
            expected_sha256,
        )
    except OSError as error:
        try:
            destination.unlink()
        except OSError:
            pass
        raise NativeRidgeIntegrityError(
            f"cannot snapshot native ridge binary safely: {error}"
        ) from error
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    finally:
        try:
            _close_descriptor(
                destination_fd,
                error_type=NativeRidgeIntegrityError,
                action="cannot close native ridge executable snapshot safely",
            )
        finally:
            _close_descriptor(
                source_fd,
                error_type=NativeRidgeIntegrityError,
                action="cannot close authenticated native ridge binary safely",
            )
    try:
        os.chmod(destination, stat.S_IRUSR | stat.S_IXUSR)
    except OSError as error:
        try:
            destination.unlink()
        except OSError:
            pass
        raise NativeRidgeIntegrityError(
            f"cannot make the native ridge snapshot executable: {error}"
        ) from error


def _require_same_file(first: os.stat_result, second: os.stat_result) -> None:
    if (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino):
        raise NativeRidgeIntegrityError("native ridge binary inode changed during verification")
    if not stat.S_ISREG(second.st_mode):
        raise NativeRidgeIntegrityError("native ridge binary is no longer a regular file")


def _require_stable_file(first: os.stat_result, second: os.stat_result) -> None:
    first_mtime = getattr(first, "st_mtime_ns", int(first.st_mtime * 1_000_000_000))
    second_mtime = getattr(second, "st_mtime_ns", int(second.st_mtime * 1_000_000_000))
    if first.st_size != second.st_size or first_mtime != second_mtime:
        raise NativeRidgeIntegrityError("native ridge binary contents changed during verification")


def _write_all(file_descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while copying native ridge binary")
        offset += written


def _close_descriptor(
    descriptor: int,
    *,
    error_type: type[NativeRidgeError],
    action: str,
    preserve_error: BaseException | None = None,
) -> None:
    """Close an owned descriptor without replacing an exception in flight."""

    active_exception = preserve_error is not None or sys.exc_info()[0] is not None
    try:
        os.close(descriptor)
    except OSError as error:
        if not active_exception:
            raise error_type(f"{action}: {error}") from error


def _close_stream(
    stream: BinaryIO,
    *,
    action: str,
    preserve_error: BaseException | None = None,
) -> None:
    """Close an adapter-owned stream without hiding an exception in flight."""

    active_exception = preserve_error is not None or sys.exc_info()[0] is not None
    try:
        stream.close()
    except OSError as error:
        if not active_exception:
            raise NativeRidgeExecutionError(f"{action}: {error}") from error


def _write_request(
    path: Path,
    *,
    request_id: bytes,
    feature_rows: Sequence[Sequence[float]],
    target_columns: Sequence[Sequence[float]],
    ridge: float,
    expected_size: int,
    feature_count: int,
    target_count: int,
) -> None:
    n = len(feature_rows)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as error:
        raise NativeRidgeExecutionError(f"cannot create private request file: {error}") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(
                _REQUEST_HEADER.pack(
                    REQUEST_MAGIC,
                    PROTOCOL_VERSION,
                    REQUEST_FLAGS,
                    request_id,
                    n,
                    feature_count,
                    target_count,
                    ridge,
                )
            )
            feature_packer = struct.Struct(f"<{feature_count}d")
            for row_index, row in enumerate(feature_rows):
                values = tuple(
                    _finite_real(value, location=f"feature_rows[{row_index}][{column_index}]")
                    for column_index, value in enumerate(row)
                )
                stream.write(feature_packer.pack(*values))
            target_packer = struct.Struct(f"<{target_count}d")
            for sample_index in range(n):
                values = tuple(
                    _finite_real(
                        target_columns[target_index][sample_index],
                        location=f"target_columns[{target_index}][{sample_index}]",
                    )
                    for target_index in range(target_count)
                )
                stream.write(target_packer.pack(*values))
            stream.flush()
            os.fsync(stream.fileno())
            actual_size = os.fstat(stream.fileno()).st_size
    except (OSError, struct.error) as error:
        raise NativeRidgeExecutionError(
            f"cannot serialize native ridge request: {error}"
        ) from error
    if actual_size != expected_size:
        raise NativeRidgeExecutionError(
            f"native ridge request has size {actual_size}; expected {expected_size}"
        )
    try:
        request_metadata = path.stat()
    except OSError as error:
        raise NativeRidgeIntegrityError(
            f"cannot inspect the private native ridge request: {error}"
        ) from error
    if os.name != "nt" and stat.S_IMODE(request_metadata.st_mode) != (stat.S_IRUSR | stat.S_IWUSR):
        raise NativeRidgeIntegrityError("request file is not owner-only")


@dataclass(slots=True)
class _PipeDrain:
    stream: BinaryIO
    destination: BinaryIO | None
    limit: int
    overflow: threading.Event
    retained: bytearray
    failed: threading.Event
    error: Exception | None = None

    def run(self) -> None:
        total = 0
        try:
            while chunk := self.stream.read(_PIPE_CHUNK_BYTES):
                remaining = max(0, self.limit - total)
                accepted = chunk[:remaining]
                if self.destination is not None and accepted:
                    self.destination.write(accepted)
                elif accepted:
                    self.retained.extend(accepted)
                total += len(chunk)
                if total > self.limit:
                    self.overflow.set()
            if self.destination is not None:
                self.destination.flush()
        except Exception as error:
            self.error = error
            self.failed.set()
        finally:
            try:
                self.stream.close()
            except Exception as error:
                if self.error is None:
                    self.error = error
                    self.failed.set()


def _execute_bounded(
    executable: Path,
    request_path: Path,
    response_path: Path,
    directory: Path,
    *,
    response_limit: int,
    timeout_seconds: float,
) -> tuple[int, bytes, int]:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        response_descriptor = os.open(response_path, flags, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as error:
        raise NativeRidgeExecutionError(f"cannot create bounded response file: {error}") from error
    try:
        request_stream = request_path.open("rb")
    except OSError as error:
        _close_descriptor(
            response_descriptor,
            error_type=NativeRidgeExecutionError,
            action="cannot close bounded response file after request-open failure",
        )
        raise NativeRidgeExecutionError(f"cannot reopen private request file: {error}") from error
    try:
        response_stream = os.fdopen(os.dup(response_descriptor), "wb", closefd=True)
    except OSError as error:
        try:
            _close_stream(
                request_stream,
                action="cannot close private request after response-open failure",
            )
        finally:
            _close_descriptor(
                response_descriptor,
                error_type=NativeRidgeExecutionError,
                action="cannot close bounded response file after response-open failure",
            )
        raise NativeRidgeExecutionError(f"cannot retain bounded response file: {error}") from error
    try:
        process = subprocess.Popen(
            [str(executable)],
            stdin=request_stream,
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
        try:
            _close_stream(
                request_stream,
                action="cannot close private request after process-start failure",
            )
        finally:
            try:
                _close_stream(
                    response_stream,
                    action="cannot close bounded response after process-start failure",
                )
            finally:
                _close_descriptor(
                    response_descriptor,
                    error_type=NativeRidgeExecutionError,
                    action="cannot close bounded response file after process-start failure",
                )
        raise NativeRidgeExecutionError(f"cannot start native ridge binary: {error}") from error
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()
    pipe_failed = threading.Event()
    stderr_bytes = bytearray()
    stdout_drain = _PipeDrain(
        process.stdout,
        response_stream,
        response_limit,
        stdout_overflow,
        bytearray(),
        pipe_failed,
    )
    stderr_drain = _PipeDrain(
        process.stderr,
        None,
        MAX_STDERR_BYTES,
        stderr_overflow,
        stderr_bytes,
        pipe_failed,
    )
    stdout_thread = threading.Thread(
        target=stdout_drain.run,
        name="tierroute-native-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stderr_drain.run,
        name="tierroute-native-stderr",
        daemon=True,
    )
    started_threads: list[threading.Thread] = []
    try:
        stdout_thread.start()
        started_threads.append(stdout_thread)
        stderr_thread.start()
        started_threads.append(stderr_thread)
    except BaseException as error:
        _terminate_process(process)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        _close_stream(
            request_stream,
            action="cannot close private request after drain-start failure",
        )
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                try:
                    os.close(pipe.fileno())
                except OSError:
                    pass
        for thread in started_threads:
            thread.join(timeout=1.0)
        try:
            _close_stream(
                response_stream,
                action="cannot close bounded response after drain-start failure",
            )
        finally:
            _close_descriptor(
                response_descriptor,
                error_type=NativeRidgeExecutionError,
                action="cannot close bounded response file after drain-start failure",
            )
        if isinstance(error, Exception):
            raise NativeRidgeExecutionError(
                f"cannot start bounded output drains: {error}"
            ) from error
        raise
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    wait_failed = False
    drain_join_error: BaseException | None = None
    stream_cleanup_error: Exception | None = None
    loop_error: BaseException | None = None
    try:
        while process.poll() is None:
            if stdout_overflow.is_set() or pipe_failed.is_set():
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
                continue
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            wait_failed = True
            _terminate_process(process)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
    except BaseException as error:
        loop_error = error
        _terminate_process(process)
    finally:
        try:
            _close_stream(
                request_stream,
                action="cannot close the consumed native ridge request",
                preserve_error=loop_error,
            )
        except Exception as error:
            stream_cleanup_error = error
        try:
            _join_drain_threads(process, (stdout_thread, stderr_thread))
        except BaseException as error:
            drain_join_error = error
        finally:
            try:
                _close_stream(
                    response_stream,
                    action="cannot close the captured native ridge response",
                    preserve_error=loop_error or drain_join_error,
                )
            except Exception as error:
                if stream_cleanup_error is None:
                    stream_cleanup_error = error
    retained_stderr = bytes(stderr_bytes)
    if stderr_overflow.is_set():
        retained_stderr += b"\n[stderr truncated]"
    try:
        if loop_error is not None:
            raise loop_error
        if timed_out:
            raise NativeRidgeExecutionError(
                f"native ridge timed out after {timeout_seconds:g} seconds"
            )
        if wait_failed:
            raise NativeRidgeExecutionError("native ridge process could not be reaped")
        if stdout_overflow.is_set():
            raise NativeRidgeProtocolError(
                f"native ridge response exceeded the {response_limit}-byte bound"
            )
        drain_error = (
            drain_join_error or stream_cleanup_error or stdout_drain.error or stderr_drain.error
        )
        if drain_error is not None:
            if not isinstance(drain_error, Exception):
                raise drain_error
            raise NativeRidgeExecutionError(
                f"cannot capture bounded native ridge output: {drain_error}"
            ) from drain_error
        assert process.returncode is not None
        return process.returncode, retained_stderr, response_descriptor
    except BaseException:
        _close_descriptor(
            response_descriptor,
            error_type=NativeRidgeExecutionError,
            action="cannot close bounded response file after execution failure",
        )
        raise


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except OSError:
        pass


def _join_drain_threads(
    process: subprocess.Popen[bytes], threads: tuple[threading.Thread, threading.Thread]
) -> None:
    for thread in threads:
        thread.join(timeout=1.0)
    if not any(thread.is_alive() for thread in threads):
        return
    _terminate_process(process)
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            try:
                os.close(pipe.fileno())
            except OSError:
                pass
    for thread in threads:
        thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in threads):
        raise NativeRidgeExecutionError("native ridge output drains did not terminate")


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


def _read_response(
    path: Path,
    *,
    response_descriptor: int,
    request_id: bytes,
    feature_count: int,
    target_count: int,
    stderr: bytes,
    returncode: int,
) -> RidgeSolution:
    try:
        path_metadata = path.lstat()
    except OSError as error:
        raise NativeRidgeProtocolError(f"cannot inspect native ridge response: {error}") from error
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(path_metadata.st_mode):
        raise NativeRidgeProtocolError("native ridge response is not a regular file")
    try:
        opened_metadata = os.fstat(response_descriptor)
        os.lseek(response_descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise NativeRidgeProtocolError(
            f"cannot inspect the original native ridge stdout safely: {error}"
        ) from error
    if (path_metadata.st_dev, path_metadata.st_ino) != (
        opened_metadata.st_dev,
        opened_metadata.st_ino,
    ):
        raise NativeRidgeProtocolError("native ridge response path no longer names stdout")
    if not stat.S_ISREG(opened_metadata.st_mode):
        raise NativeRidgeProtocolError("native ridge stdout is not a regular file")
    if os.name != "nt" and stat.S_IMODE(opened_metadata.st_mode) != (stat.S_IRUSR | stat.S_IWUSR):
        raise NativeRidgeProtocolError("native ridge response is not owner-only")
    try:
        if opened_metadata.st_size < _RESPONSE_HEADER.size:
            raise NativeRidgeProtocolError("native ridge response is truncated before its header")
        header = _read_exact_response(response_descriptor, _RESPONSE_HEADER.size)
        magic, version, status_code, response_id, response_d, response_m = _RESPONSE_HEADER.unpack(
            header
        )
        if magic != RESPONSE_MAGIC:
            raise NativeRidgeProtocolError("native ridge response has the wrong magic")
        if version != PROTOCOL_VERSION:
            raise NativeRidgeProtocolError("native ridge response has an unsupported version")
        if status_code not in _KNOWN_STATUSES:
            raise NativeRidgeProtocolError("native ridge response has an unknown status")
        if not hmac.compare_digest(response_id, request_id):
            raise NativeRidgeProtocolError("native ridge response request ID does not match")
        if response_d != feature_count or response_m != target_count:
            raise NativeRidgeProtocolError("native ridge response dimensions do not match")
        deferred_error: NativeRidgeError | None = None
        if status_code != STATUS_SUCCESS:
            if opened_metadata.st_size != _RESPONSE_HEADER.size:
                raise NativeRidgeProtocolError("native ridge error response contains a payload")
            if returncode == 0:
                deferred_error = NativeRidgeProtocolError(
                    "native ridge returned an error status with a successful process exit"
                )
            else:
                deferred_error = NativeRidgeStatusError(status_code, stderr)
        else:
            if returncode != 0:
                deferred_error = NativeRidgeProtocolError(
                    "native ridge returned success with "
                    f"nonzero process status {returncode}{_stderr_detail(stderr)}"
                )
            expected_size = _success_response_size(feature_count, target_count)
            if opened_metadata.st_size != expected_size:
                raise NativeRidgeProtocolError(
                    "native ridge response has size "
                    f"{opened_metadata.st_size}; expected {expected_size}"
                )
            value_count = target_count * (feature_count + 1)
            payload = _read_exact_response(response_descriptor, 8 * value_count)
            if os.read(response_descriptor, 1):
                raise NativeRidgeProtocolError(
                    "native ridge response payload length changed while read"
                )
        _require_stable_response(opened_metadata, os.fstat(response_descriptor))
        try:
            final_path_metadata = path.lstat()
        except OSError as error:
            raise NativeRidgeProtocolError(
                f"native ridge response path changed while read: {error}"
            ) from error
        if (opened_metadata.st_dev, opened_metadata.st_ino) != (
            final_path_metadata.st_dev,
            final_path_metadata.st_ino,
        ):
            raise NativeRidgeProtocolError("native ridge response inode changed while read")
        _require_stable_response(opened_metadata, final_path_metadata)
        if deferred_error is not None:
            raise deferred_error
    except OSError as error:
        raise NativeRidgeProtocolError(
            f"cannot read native ridge response safely: {error}"
        ) from error
    values = struct.unpack(f"<{value_count}d", payload)
    if any(not math.isfinite(value) for value in values):
        raise NativeRidgeProtocolError("native ridge response contains non-finite coefficients")
    intercepts = tuple(values[:target_count])
    flat_weights = values[target_count:]
    weights = tuple(
        tuple(flat_weights[index * feature_count : (index + 1) * feature_count])
        for index in range(target_count)
    )
    return RidgeSolution(weights=weights, intercepts=intercepts)


def _require_stable_response(first: os.stat_result, second: os.stat_result) -> None:
    first_mtime = getattr(first, "st_mtime_ns", int(first.st_mtime * 1_000_000_000))
    second_mtime = getattr(second, "st_mtime_ns", int(second.st_mtime * 1_000_000_000))
    if first.st_size != second.st_size or first_mtime != second_mtime:
        raise NativeRidgeProtocolError("native ridge response changed while read")


def _read_exact_response(descriptor: int, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = os.read(descriptor, min(_PIPE_CHUNK_BYTES, length - len(chunks)))
        if not chunk:
            raise NativeRidgeProtocolError("native ridge response was truncated while read")
        chunks.extend(chunk)
    return bytes(chunks)


def _stderr_detail(stderr: bytes) -> str:
    if not stderr:
        return ""
    decoded = stderr.decode("utf-8", errors="replace").strip()
    return f": {decoded}" if decoded else ""
