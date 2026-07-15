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
import unicodedata
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


def _portable_path_key(path: Path) -> tuple[str, ...]:
    """Return a conservative key across case/Unicode-normalizing filesystems."""

    return tuple(unicodedata.normalize("NFC", part).casefold() for part in _resolved(path).parts)


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
    portable_keys = tuple(_portable_path_key(path) for path in paths)
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
            key = portable_keys[index]
            other_key = portable_keys[other_index]
            if canonical == other_canonical or key == other_key or _same_existing_file(path, other):
                raise ValueError(f"write destinations must be different paths: {other}, {path}")
            if (
                canonical in other_canonical.parents
                or other_canonical in canonical.parents
                or key[: len(other_key)] == other_key
                or other_key[: len(key)] == key
            ):
                raise ValueError(
                    f"write destinations cannot be ancestors of one another: {other}, {path}"
                )

    protected = tuple(Path(path) for path in protected_paths)
    protected_resolved = tuple(_resolved(path) for path in protected)
    protected_keys = tuple(_portable_path_key(path) for path in protected)
    for path, canonical, key in zip(paths, resolved, portable_keys, strict=True):
        for protected_path, protected_canonical, protected_key in zip(
            protected, protected_resolved, protected_keys, strict=True
        ):
            if (
                canonical == protected_canonical
                or key == protected_key
                or _same_existing_file(path, protected_path)
            ):
                raise ValueError(
                    f"write destination aliases protected input path: {path}, {protected_path}"
                )
    return paths


_TEMP_ALLOCATION_ATTEMPTS = 100


def _reserve_temporary_path(
    destination: Path,
    kind: str,
    reserved: set[tuple[str, ...]],
    registry: list[Path | None],
) -> Path:
    """Allocate, close, reserve, and register a unique same-directory path."""

    for _ in range(_TEMP_ALLOCATION_ATTEMPTS):
        descriptor: int | None = None
        path: Path | None = None
        registered = False
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.{kind}.",
                suffix=".tmp",
            )
            path = Path(raw_path)
            key = _portable_path_key(path)
            if key not in reserved:
                reserved.add(key)
                registry.append(path)
                registered = True
            os.close(descriptor)
            descriptor = None
            if registered:
                return path
            path.unlink()
        except BaseException:
            close_error: BaseException | None = None
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as error:
                    close_error = error
            if path is not None and not registered:
                try:
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    close_error = close_error or error
            if close_error is not None:
                raise OSError(f"could not clean failed {kind} allocation") from close_error
            raise
    raise OSError(f"could not allocate a collision-free {kind} file for {destination}")


def _write_all(descriptor: int, document: bytes) -> None:
    view = memoryview(document)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive guard for an invalid OS contract
            raise OSError("artifact staging write made no progress")
        view = view[written:]


def _close_owned_descriptors(descriptors: Sequence[int | None]) -> list[BaseException]:
    errors: list[BaseException] = []
    for descriptor in descriptors:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:  # pragma: no cover - catastrophic OS failure
                errors.append(error)
    return errors


def _raise_operation_or_cleanup(
    operation_error: BaseException | None,
    cleanup_errors: Sequence[BaseException],
    context: str,
) -> None:
    if cleanup_errors:
        details = "; ".join(repr(error) for error in cleanup_errors)
        raise OSError(f"{context} cleanup failed: {details}") from operation_error
    if operation_error is not None:
        raise operation_error


def _write_stage(
    destination: Path,
    document: bytes,
    reserved: set[tuple[str, ...]],
    registry: list[Path | None],
) -> None:
    path = _reserve_temporary_path(destination, "stage", reserved, registry)
    descriptor: int | None = None
    operation_error: BaseException | None = None
    try:
        flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"stage path is not a regular file: {path}")
        _write_all(descriptor, document)
        os.fsync(descriptor)
    except BaseException as error:
        operation_error = error
    cleanup_errors = _close_owned_descriptors((descriptor,))
    _raise_operation_or_cleanup(operation_error, cleanup_errors, "stage descriptor")


def _backup(
    destination: Path,
    reserved: set[tuple[str, ...]],
    registry: list[Path | None],
    restore_identity_registry: list[frozenset[tuple[int, int]] | None],
) -> None:
    details = _lstat(destination)
    if details is None:
        registry.append(None)
        restore_identity_registry.append(None)
        return
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError(f"write destination changed to an unsafe node: {destination}")
    original_identity = (details.st_dev, details.st_ino)

    backup = _reserve_temporary_path(destination, "backup", reserved, registry)
    backup.unlink()
    try:
        os.link(destination, backup, follow_symlinks=False)
        linked = _lstat(backup)
        if linked is None:
            raise OSError(f"backup hard link disappeared: {backup}")
        restore_identity_registry.append(
            frozenset({original_identity, (linked.st_dev, linked.st_ino)})
        )
        return
    except (OSError, NotImplementedError, TypeError):
        backup.unlink(missing_ok=True)

    source_descriptor: int | None = None
    backup_descriptor: int | None = None
    operation_error: BaseException | None = None
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        source_descriptor = os.open(destination, source_flags)
        current = os.fstat(source_descriptor)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
            details.st_dev,
            details.st_ino,
        ):
            raise OSError(f"write destination changed while backing it up: {destination}")
        backup_descriptor = os.open(
            backup,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        while chunk := os.read(source_descriptor, 1024 * 1024):
            _write_all(backup_descriptor, chunk)
        os.chmod(backup, stat.S_IMODE(details.st_mode))
        os.fsync(backup_descriptor)
    except BaseException as error:
        operation_error = error
    cleanup_errors = _close_owned_descriptors((source_descriptor, backup_descriptor))
    _raise_operation_or_cleanup(operation_error, cleanup_errors, "backup descriptor")
    copied = _lstat(backup)
    if copied is None:
        raise OSError(f"backup copy disappeared: {backup}")
    restore_identity_registry.append(frozenset({original_identity, (copied.st_dev, copied.st_ino)}))


def _fsync_directories(paths: Sequence[Path]) -> None:
    # Windows has no portable directory-fsync API. File contents and atomic replaces
    # still work there; POSIX additionally receives directory-entry durability.
    if os.name == "nt":
        return
    for directory in sorted({path.parent for path in paths}, key=str):
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor: int | None = None
        operation_error: BaseException | None = None
        try:
            descriptor = os.open(directory, flags)
            os.fsync(descriptor)
        except BaseException as error:
            operation_error = error
        cleanup_errors = _close_owned_descriptors((descriptor,))
        _raise_operation_or_cleanup(operation_error, cleanup_errors, "directory descriptor")


def _cleanup(
    paths: Sequence[Path | None],
    *,
    preserve: frozenset[Path] = frozenset(),
) -> list[tuple[Path, BaseException]]:
    errors: list[tuple[Path, BaseException]] = []
    for path in paths:
        if path is not None and path not in preserve:
            try:
                path.unlink(missing_ok=True)
            except BaseException as error:
                errors.append((path, error))
    return errors


def _error_details(errors: Sequence[tuple[str, BaseException]]) -> str:
    return "; ".join(f"{context}: {error!r}" for context, error in errors)


def replace_text_bundle(
    writes: Sequence[AtomicTextWrite],
    *,
    protected_paths: Sequence[str | Path] = (),
) -> tuple[Path, ...]:
    """Validate, stage, replace, verify, and rollback a text-artifact bundle.

    Replacements follow input order. Callers writing a predictor/policy pair should
    therefore put the policy last, so its hash binding is the final visible commit.
    Concurrent writers and power-loss transactions across multiple pathnames are not
    supported; ordinary exceptions, including asynchronous Python exceptions, roll
    back every attempted replacement.
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

    reserved = {_portable_path_key(path) for path in destinations}
    reserved.update(_portable_path_key(Path(path)) for path in protected_paths)
    stages: list[Path | None] = []
    backups: list[Path | None] = []
    restore_identities: list[frozenset[tuple[int, int]] | None] = []
    attempted: list[int] = []
    try:
        for destination, document in zip(destinations, documents, strict=True):
            _write_stage(destination, document, reserved, stages)
        for destination in destinations:
            _backup(destination, reserved, backups, restore_identities)
        for index, destination in enumerate(destinations):
            stage = stages[index]
            if stage is None:
                raise AssertionError("artifact stage disappeared before commit")
            # Record intent first: an asynchronous exception may arrive after rename
            # succeeds but before ``os.replace`` returns to Python.
            attempted.append(index)
            os.replace(stage, destination)
            stages[index] = None
        _fsync_directories(destinations)

        for entry, document, destination in zip(entries, documents, destinations, strict=True):
            restored_bytes = destination.read_bytes()
            if restored_bytes != document:
                raise OSError(f"artifact verification mismatch after replacement: {destination}")
            restored = restored_bytes.decode("utf-8")
            entry.validator(restored)
    except BaseException as error:
        rollback_errors: list[tuple[str, BaseException]] = []
        preserve_backups: set[Path] = set()
        for index in reversed(attempted):
            destination = destinations[index]
            backup = backups[index]
            acceptable_identities = restore_identities[index]
            restore_error: BaseException | None = None
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    if acceptable_identities is None:
                        raise AssertionError("existing artifact backup has no restore identity")
                    current = _lstat(destination)
                    if (
                        current is None
                        or (
                            current.st_dev,
                            current.st_ino,
                        )
                        not in acceptable_identities
                    ):
                        os.replace(backup, destination)
            except BaseException as rollback_error:
                restore_error = rollback_error

            if restore_error is not None:
                # A syscall may raise after making its change visible (for example an
                # injected asynchronous exception). Reinspect the destination, but no
                # recovery inspection is itself allowed to abort the remaining
                # rollback loop.
                try:
                    restored = _lstat(destination)
                    restored_identity = (
                        None if restored is None else (restored.st_dev, restored.st_ino)
                    )
                    restored_ok = (
                        restored_identity is None
                        if acceptable_identities is None
                        else restored_identity in acceptable_identities
                    )
                except BaseException as inspection_error:
                    restored_ok = False
                    rollback_errors.append(
                        (f"inspect restored destination {destination}", inspection_error)
                    )
                if not restored_ok:
                    rollback_errors.append((f"restore {destination}", restore_error))
                    # Preserve unconditionally when restoration cannot be verified.
                    # Even an existence probe can fail or lie under the same fault.
                    if backup is not None:
                        preserve_backups.add(backup)
        try:
            _fsync_directories(destinations)
        except BaseException as sync_error:
            rollback_errors.append(("rollback directory sync", sync_error))

        cleanup_errors = _cleanup(stages)
        cleanup_errors.extend(_cleanup(backups, preserve=frozenset(preserve_backups)))
        for path, cleanup_error in cleanup_errors:
            rollback_errors.append((f"cleanup {path}", cleanup_error))
        try:
            _fsync_directories(destinations)
        except BaseException as sync_error:
            rollback_errors.append(("post-cleanup directory sync", sync_error))

        if rollback_errors:
            recovery = ", ".join(str(path) for path in sorted(preserve_backups, key=str))
            recovery_note = f"; preserved recovery backups: {recovery}" if recovery else ""
            raise OSError(
                "artifact write failed and rollback/cleanup was incomplete: "
                f"{_error_details(rollback_errors)}{recovery_note}"
            ) from error
        raise

    cleanup_errors = _cleanup(stages)
    cleanup_errors.extend(_cleanup(backups))
    committed_errors = [
        (f"cleanup {path}", cleanup_error) for path, cleanup_error in cleanup_errors
    ]
    try:
        _fsync_directories(destinations)
    except BaseException as sync_error:
        committed_errors.append(("post-cleanup directory sync", sync_error))
    if committed_errors:
        raise OSError(
            "artifacts were committed and validated, but temporary cleanup/durability "
            f"was incomplete: {_error_details(committed_errors)}"
        )
    return destinations
