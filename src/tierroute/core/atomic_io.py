# SPDX-License-Identifier: Apache-2.0
"""Validated, collision-safe replacement of one or more text artifacts.

Each destination is staged in its own directory with an exclusive random name. For
multi-file writes, existing contents are backed up before the first replacement and
ordinary failures roll every committed destination back. POSIX does not provide one
atomic operation for unrelated pathnames, so callers must still avoid concurrent
writers; policy-last ordering plus fail-closed hash validation protects readers from
accepting a transient mixed predictor/policy pair.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AtomicTextWrite:
    """One UTF-8 document, destination, and strict round-trip validator."""

    destination: str | Path
    document: str
    validator: Callable[[str], object]


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"cannot inspect artifact path: {path}") from error


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot resolve artifact path: {path}") from error


def _same_existing_file(left: Path, right: Path) -> bool:
    if _lstat(left) is None or _lstat(right) is None:
        return False
    try:
        return os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(f"cannot compare artifact paths: {left}, {right}") from error


def validate_write_paths(
    destinations: Sequence[str | Path],
    *,
    protected_paths: Sequence[str | Path] = (),
) -> tuple[Path, ...]:
    """Reject destination aliases and unsafe filesystem node types without writing."""

    paths = tuple(Path(path) for path in destinations)
    if not paths:
        raise ValueError("at least one write destination is required")
    resolved = tuple(_resolved(path) for path in paths)
    for index, (path, canonical) in enumerate(zip(paths, resolved, strict=True)):
        details = _lstat(path)
        if details is not None:
            if stat.S_ISLNK(details.st_mode):
                raise ValueError(f"write destination must not be a symbolic link: {path}")
            if not stat.S_ISREG(details.st_mode):
                raise ValueError(f"write destination must be a regular file or absent: {path}")
        for other_index in range(index):
            other = paths[other_index]
            other_canonical = resolved[other_index]
            if canonical == other_canonical or _same_existing_file(path, other):
                raise ValueError(f"write destinations must be different paths: {other}, {path}")
            if canonical in other_canonical.parents or other_canonical in canonical.parents:
                raise ValueError(
                    f"write destinations cannot be ancestors of one another: {other}, {path}"
                )

    protected = tuple(Path(path) for path in protected_paths)
    protected_resolved = tuple(_resolved(path) for path in protected)
    for path, canonical in zip(paths, resolved, strict=True):
        for protected_path, protected_canonical in zip(protected, protected_resolved, strict=True):
            if canonical == protected_canonical or _same_existing_file(path, protected_path):
                raise ValueError(
                    f"write destination aliases protected input path: {path}, {protected_path}"
                )
    return paths


def _write_stage(destination: Path, document: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.stage.",
        suffix=".tmp",
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _backup(destination: Path) -> Path | None:
    details = _lstat(destination)
    if details is None:
        return None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError(f"write destination changed to an unsafe node: {destination}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(destination, flags)
    backup_descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.backup.",
        suffix=".tmp",
    )
    backup = Path(raw_path)
    try:
        with (
            os.fdopen(source_descriptor, "rb") as source,
            os.fdopen(backup_descriptor, "wb") as target,
        ):
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
            target.flush()
            os.fchmod(target.fileno(), stat.S_IMODE(details.st_mode))
            os.fsync(target.fileno())
    except BaseException:
        try:
            os.close(source_descriptor)
        except OSError:
            pass
        try:
            os.close(backup_descriptor)
        except OSError:
            pass
        backup.unlink(missing_ok=True)
        raise
    return backup


def _fsync_directories(paths: Sequence[Path]) -> None:
    for directory in sorted({path.parent for path in paths}, key=str):
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _cleanup(paths: Sequence[Path | None]) -> None:
    for path in paths:
        if path is not None:
            path.unlink(missing_ok=True)


def replace_text_bundle(
    writes: Sequence[AtomicTextWrite],
    *,
    protected_paths: Sequence[str | Path] = (),
) -> tuple[Path, ...]:
    """Validate, stage, replace, verify, and rollback a text-artifact bundle.

    Replacements follow input order. Callers writing a predictor/policy pair should
    therefore put the policy last, so its hash binding is the final visible commit.
    """

    entries = tuple(writes)
    if not entries:
        raise ValueError("at least one text write is required")
    documents: list[bytes] = []
    for entry in entries:
        if not isinstance(entry, AtomicTextWrite):
            raise TypeError("writes must contain AtomicTextWrite values")
        if not isinstance(entry.document, str):
            raise TypeError("artifact document must be text")
        if not callable(entry.validator):
            raise TypeError("artifact validator must be callable")
        try:
            encoded = entry.document.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("artifact document must contain valid Unicode text") from error
        entry.validator(entry.document)
        documents.append(encoded)

    destinations = validate_write_paths(
        tuple(entry.destination for entry in entries),
        protected_paths=protected_paths,
    )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)

    stages: list[Path | None] = []
    backups: list[Path | None] = []
    committed: list[int] = []
    try:
        for destination, document in zip(destinations, documents, strict=True):
            stages.append(_write_stage(destination, document))
        for destination in destinations:
            backups.append(_backup(destination))
        for index, destination in enumerate(destinations):
            stage = stages[index]
            if stage is None:
                raise AssertionError("artifact stage disappeared before commit")
            os.replace(stage, destination)
            stages[index] = None
            committed.append(index)
        _fsync_directories(destinations)

        for entry, destination in zip(entries, destinations, strict=True):
            restored = destination.read_text(encoding="utf-8")
            if restored != entry.document:
                raise OSError(f"artifact verification mismatch after replacement: {destination}")
            entry.validator(restored)
    except Exception as error:
        rollback_errors: list[Exception] = []
        for index in reversed(committed):
            destination = destinations[index]
            backup = backups[index]
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
                    backups[index] = None
            except Exception as rollback_error:  # pragma: no cover - catastrophic filesystem fault
                rollback_errors.append(rollback_error)
        try:
            _fsync_directories(destinations)
        except Exception as rollback_error:  # pragma: no cover - catastrophic filesystem fault
            rollback_errors.append(rollback_error)
        if rollback_errors:
            details = "; ".join(str(item) for item in rollback_errors)
            raise OSError(
                f"artifact write failed and rollback was incomplete: {details}"
            ) from error
        raise
    finally:
        _cleanup(stages)
        _cleanup(backups)

    _fsync_directories(destinations)
    return destinations
