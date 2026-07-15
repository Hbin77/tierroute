# SPDX-License-Identifier: Apache-2.0
"""Tests for validated collision-safe artifact replacement."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tierroute.core import atomic_io
from tierroute.core.atomic_io import AtomicTextWrite, replace_text_bundle, validate_write_paths


def _validate_jsonish(document: str) -> None:
    if not document.startswith("{") or not document.endswith("}\n"):
        raise ValueError("invalid test document")


def _debris(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if ".stage." in path.name or ".backup." in path.name
    )


def test_bundle_replaces_two_documents_and_leaves_no_debris(tmp_path: Path) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    first.write_text("old predictor", encoding="utf-8")
    second.write_text("old policy", encoding="utf-8")

    result = replace_text_bundle(
        (
            AtomicTextWrite(first, '{"kind":"predictor"}\n', _validate_jsonish),
            AtomicTextWrite(second, '{"kind":"policy"}\n', _validate_jsonish),
        )
    )

    assert result == (first, second)
    assert first.read_text(encoding="utf-8") == '{"kind":"predictor"}\n'
    assert second.read_text(encoding="utf-8") == '{"kind":"policy"}\n'
    assert _debris(tmp_path) == []


@pytest.mark.parametrize("preexisting", [False, True])
def test_second_replace_failure_rolls_back_entire_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    if preexisting:
        first.write_text("old predictor", encoding="utf-8")
        second.write_text("old policy", encoding="utf-8")
    real_replace = os.replace

    def fail_policy_stage(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == second and ".stage." in Path(source).name:
            raise OSError("injected policy replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", fail_policy_stage)

    with pytest.raises(OSError, match="injected policy"):
        replace_text_bundle(
            (
                AtomicTextWrite(first, '{"new":1}\n', _validate_jsonish),
                AtomicTextWrite(second, '{"new":2}\n', _validate_jsonish),
            )
        )

    if preexisting:
        assert first.read_text(encoding="utf-8") == "old predictor"
        assert second.read_text(encoding="utf-8") == "old policy"
    else:
        assert not first.exists()
        assert not second.exists()
    assert _debris(tmp_path) == []


def test_async_exception_after_policy_rename_restores_the_old_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    first.write_text("old predictor", encoding="utf-8")
    second.write_text("old policy", encoding="utf-8")
    real_replace = os.replace

    def interrupt_after_policy_rename(source: str | Path, destination: str | Path) -> None:
        real_replace(source, destination)
        if Path(destination) == second and ".stage." in Path(source).name:
            raise KeyboardInterrupt

    monkeypatch.setattr(atomic_io.os, "replace", interrupt_after_policy_rename)

    with pytest.raises(KeyboardInterrupt):
        replace_text_bundle(
            (
                AtomicTextWrite(first, '{"new":1}\n', _validate_jsonish),
                AtomicTextWrite(second, '{"new":2}\n', _validate_jsonish),
            )
        )

    assert first.read_text(encoding="utf-8") == "old predictor"
    assert second.read_text(encoding="utf-8") == "old policy"
    assert _debris(tmp_path) == []


def test_async_exception_after_copy_backup_restore_accepts_restored_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "predictor.json"
    original = b"old\r\ncontent\x1a"
    destination.write_bytes(original)
    validations = 0
    real_replace = os.replace

    def force_copy_backup(*args: object, **kwargs: object) -> None:
        raise OSError("injected hard-link unavailability")

    def fail_post_validation(document: str) -> None:
        nonlocal validations
        _validate_jsonish(document)
        validations += 1
        if validations == 2:
            raise ValueError("injected post-write validation failure")

    def interrupt_after_restore(source: str | Path, target: str | Path) -> None:
        real_replace(source, target)
        if ".backup." in Path(source).name:
            raise KeyboardInterrupt

    monkeypatch.setattr(atomic_io.os, "link", force_copy_backup)
    monkeypatch.setattr(atomic_io.os, "replace", interrupt_after_restore)

    with pytest.raises(ValueError, match="post-write validation"):
        replace_text_bundle((AtomicTextWrite(destination, '{"new":1}\n', fail_post_validation),))

    assert destination.read_bytes() == original
    assert _debris(tmp_path) == []


def test_backup_allocation_failure_does_not_open_the_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "predictor.json"
    destination.write_text("old predictor", encoding="utf-8")
    real_mkstemp = atomic_io.tempfile.mkstemp
    real_open = atomic_io.os.open
    allocations = 0
    source_descriptors: list[int] = []

    def fail_backup_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal allocations
        allocations += 1
        if allocations == 2:
            raise OSError("injected backup allocation failure")
        return real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]

    def track_source_open(path: str | Path, flags: int, mode: int = 0o777) -> int:
        descriptor = real_open(path, flags, mode)
        if Path(path) == destination:
            source_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(atomic_io.tempfile, "mkstemp", fail_backup_mkstemp)
    monkeypatch.setattr(atomic_io.os, "open", track_source_open)

    with pytest.raises(OSError, match="backup allocation"):
        replace_text_bundle((AtomicTextWrite(destination, '{"new":1}\n', _validate_jsonish),))

    assert source_descriptors == []
    assert destination.read_text(encoding="utf-8") == "old predictor"
    assert _debris(tmp_path) == []


def test_post_write_validation_failure_restores_existing_content(tmp_path: Path) -> None:
    destination = tmp_path / "predictor.json"
    destination.write_text("old predictor", encoding="utf-8")
    calls = 0

    def fail_second_validation(document: str) -> None:
        nonlocal calls
        _validate_jsonish(document)
        calls += 1
        if calls == 2:
            raise ValueError("injected post-write validation failure")

    with pytest.raises(ValueError, match="post-write validation"):
        replace_text_bundle((AtomicTextWrite(destination, '{"new":1}\n', fail_second_validation),))

    assert destination.read_text(encoding="utf-8") == "old predictor"
    assert _debris(tmp_path) == []


def test_temporary_names_cannot_alias_reserved_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    first.write_text("old predictor", encoding="utf-8")
    real_mkstemp = atomic_io.tempfile.mkstemp
    forced_kinds: set[str] = set()

    def force_destination_collision(*args: object, **kwargs: object) -> tuple[int, str]:
        prefix = str(kwargs.get("prefix", ""))
        kind = "stage" if ".stage." in prefix else "backup"
        if kind not in forced_kinds and not second.exists():
            forced_kinds.add(kind)
            descriptor = os.open(second, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            return descriptor, str(second)
        return real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(atomic_io.tempfile, "mkstemp", force_destination_collision)

    replace_text_bundle(
        (
            AtomicTextWrite(first, '{"new":1}\n', _validate_jsonish),
            AtomicTextWrite(second, '{"new":2}\n', _validate_jsonish),
        )
    )

    assert forced_kinds == {"stage", "backup"}
    assert first.read_text(encoding="utf-8") == '{"new":1}\n'
    assert second.read_text(encoding="utf-8") == '{"new":2}\n'
    assert _debris(tmp_path) == []


@pytest.mark.parametrize("helper_name", ["_write_stage", "_backup"])
def test_async_exception_after_helper_return_leaves_no_unregistered_debris(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    destination = tmp_path / "predictor.json"
    destination.write_text("old predictor", encoding="utf-8")
    real_helper = getattr(atomic_io, helper_name)

    def interrupt_after_registration(*args: object, **kwargs: object) -> None:
        real_helper(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic_io, helper_name, interrupt_after_registration)

    with pytest.raises(KeyboardInterrupt):
        replace_text_bundle((AtomicTextWrite(destination, '{"new":1}\n', _validate_jsonish),))

    assert destination.read_text(encoding="utf-8") == "old predictor"
    assert _debris(tmp_path) == []


def test_failed_restore_preserves_the_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    first.write_text("old predictor", encoding="utf-8")
    second.write_text("old policy", encoding="utf-8")
    validations = 0
    real_replace = os.replace

    def fail_policy_post_validation(document: str) -> None:
        nonlocal validations
        _validate_jsonish(document)
        validations += 1
        if validations == 2:
            raise ValueError("injected policy validation failure")

    def fail_policy_restore(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == second and ".backup." in Path(source).name:
            raise OSError("injected policy restore failure")
        real_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", fail_policy_restore)

    with pytest.raises(OSError, match="preserved recovery backups"):
        replace_text_bundle(
            (
                AtomicTextWrite(first, '{"new":1}\n', _validate_jsonish),
                AtomicTextWrite(second, '{"new":2}\n', fail_policy_post_validation),
            )
        )

    backups = [path for path in tmp_path.iterdir() if ".backup." in path.name]
    assert first.read_text(encoding="utf-8") == "old predictor"
    assert second.read_text(encoding="utf-8") == '{"new":2}\n'
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old policy"


def test_persistent_backup_cleanup_failure_does_not_stop_earlier_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    first.write_text("old predictor", encoding="utf-8")
    second.write_text("old policy", encoding="utf-8")
    real_replace = os.replace
    real_unlink = Path.unlink
    commit_failed = False

    def fail_policy_stage(source: str | Path, destination: str | Path) -> None:
        nonlocal commit_failed
        if Path(destination) == second and ".stage." in Path(source).name:
            commit_failed = True
            raise OSError("injected policy replacement failure")
        real_replace(source, destination)

    def fail_policy_backup_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if commit_failed and ".policy.json.backup." in path.name:
            raise PermissionError("injected persistent backup cleanup failure")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(atomic_io.os, "replace", fail_policy_stage)
    monkeypatch.setattr(Path, "unlink", fail_policy_backup_cleanup)

    with pytest.raises(OSError, match="rollback/cleanup was incomplete") as caught:
        replace_text_bundle(
            (
                AtomicTextWrite(first, '{"new":1}\n', _validate_jsonish),
                AtomicTextWrite(second, '{"new":2}\n', _validate_jsonish),
            )
        )

    assert isinstance(caught.value.__cause__, OSError)
    assert "injected policy replacement failure" in str(caught.value.__cause__)
    assert first.read_text(encoding="utf-8") == "old predictor"
    assert second.read_text(encoding="utf-8") == "old policy"
    debris = _debris(tmp_path)
    assert len(debris) == 1
    assert ".policy.json.backup." in debris[0].name
    assert debris[0].read_text(encoding="utf-8") == "old policy"


def test_rollback_inspection_failure_is_aggregated_and_other_files_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    first.write_text("old predictor", encoding="utf-8")
    second.write_text("old policy", encoding="utf-8")
    real_replace = os.replace
    real_lstat = atomic_io._lstat
    commit_failed = False

    def fail_policy_stage(source: str | Path, destination: str | Path) -> None:
        nonlocal commit_failed
        if Path(destination) == second and ".stage." in Path(source).name:
            commit_failed = True
            raise OSError("injected policy replacement failure")
        real_replace(source, destination)

    def fail_policy_rollback_inspection(path: Path):
        if commit_failed and path == second:
            raise PermissionError("injected rollback inspection failure")
        return real_lstat(path)

    monkeypatch.setattr(atomic_io.os, "replace", fail_policy_stage)
    monkeypatch.setattr(atomic_io, "_lstat", fail_policy_rollback_inspection)

    with pytest.raises(OSError, match="inspect restored destination") as caught:
        replace_text_bundle(
            (
                AtomicTextWrite(first, '{"new":1}\n', _validate_jsonish),
                AtomicTextWrite(second, '{"new":2}\n', _validate_jsonish),
            )
        )

    assert isinstance(caught.value.__cause__, OSError)
    assert first.read_text(encoding="utf-8") == "old predictor"
    assert second.read_text(encoding="utf-8") == "old policy"
    debris = _debris(tmp_path)
    assert len(debris) == 1
    assert ".policy.json.backup." in debris[0].name
    assert debris[0].read_text(encoding="utf-8") == "old policy"


def test_cleanup_failure_reports_committed_state_and_still_syncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"
    first.write_text("old predictor", encoding="utf-8")
    second.write_text("old policy", encoding="utf-8")
    real_unlink = Path.unlink
    real_sync = atomic_io._fsync_directories
    real_replace = os.replace
    commit_finished = False
    sync_calls = 0

    def track_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal commit_finished
        real_replace(source, destination)
        if Path(destination) == second and ".stage." in Path(source).name:
            commit_finished = True

    def fail_backup_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if commit_finished and ".backup." in path.name:
            raise PermissionError("injected backup cleanup failure")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    def track_sync(paths: object) -> None:
        nonlocal sync_calls
        sync_calls += 1
        real_sync(paths)  # type: ignore[arg-type]

    monkeypatch.setattr(atomic_io.os, "replace", track_replace)
    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)
    monkeypatch.setattr(atomic_io, "_fsync_directories", track_sync)

    with pytest.raises(OSError, match="artifacts were committed and validated"):
        replace_text_bundle(
            (
                AtomicTextWrite(first, '{"new":1}\n', _validate_jsonish),
                AtomicTextWrite(second, '{"new":2}\n', _validate_jsonish),
            )
        )

    assert first.read_text(encoding="utf-8") == '{"new":1}\n'
    assert second.read_text(encoding="utf-8") == '{"new":2}\n'
    assert sync_calls == 2


def test_exact_crlf_document_round_trips_without_newline_translation(tmp_path: Path) -> None:
    destination = tmp_path / "document.txt"
    document = "line one\r\nline two\r\n"

    replace_text_bundle((AtomicTextWrite(destination, document, lambda value: None),))

    assert destination.read_bytes() == document.encode("utf-8")


def test_stage_and_copy_backup_descriptors_request_binary_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "document.txt"
    destination.write_bytes(b"old\r\ncontent\x1a")
    real_open = atomic_io.os.open
    binary_flag = 1 << 29
    opened: list[tuple[Path, int]] = []

    def force_copy_backup(*args: object, **kwargs: object) -> None:
        raise OSError("injected hard-link unavailability")

    def record_open(path: str | Path, flags: int, mode: int = 0o777) -> int:
        opened.append((Path(path), flags))
        return real_open(path, flags & ~binary_flag, mode)

    monkeypatch.setattr(atomic_io.os, "O_BINARY", binary_flag, raising=False)
    monkeypatch.setattr(atomic_io.os, "link", force_copy_backup)
    monkeypatch.setattr(atomic_io.os, "open", record_open)

    replace_text_bundle((AtomicTextWrite(destination, "new\r\ncontent\x1a", lambda _: None),))

    # tempfile.mkstemp caches platform flags at import time; select tierroute's
    # explicit reopen/create calls rather than tempfile's initial allocations.
    stage_flags = [flags for path, flags in opened if ".stage." in path.name and flags & os.O_TRUNC]
    backup_flags = [
        flags
        for path, flags in opened
        if ".backup." in path.name
        and flags & os.O_CREAT
        and flags & os.O_EXCL
        and (flags & os.O_ACCMODE) == os.O_WRONLY
    ]
    source_flags = [flags for path, flags in opened if path == destination]
    assert stage_flags and all(flags & binary_flag for flags in stage_flags)
    assert backup_flags and all(flags & binary_flag for flags in backup_flags)
    assert source_flags and all(flags & binary_flag for flags in source_flags)


def test_validation_failure_writes_nothing(tmp_path: Path) -> None:
    first = tmp_path / "predictor.json"
    second = tmp_path / "policy.json"

    with pytest.raises(ValueError, match="invalid test"):
        replace_text_bundle(
            (
                AtomicTextWrite(first, '{"valid":true}\n', _validate_jsonish),
                AtomicTextWrite(second, "invalid", _validate_jsonish),
            )
        )

    assert not first.exists()
    assert not second.exists()
    assert list(tmp_path.iterdir()) == []


def test_path_validation_rejects_aliases_symlinks_and_ancestors(tmp_path: Path) -> None:
    protected = tmp_path / "replay.json"
    protected.write_text("source", encoding="utf-8")
    hardlink = tmp_path / "hardlink.json"
    os.link(protected, hardlink)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(protected)

    with pytest.raises(ValueError, match="protected input"):
        validate_write_paths((hardlink,), protected_paths=(protected,))
    with pytest.raises(ValueError, match="symbolic link"):
        validate_write_paths((symlink,))
    with pytest.raises(ValueError, match="ancestors"):
        validate_write_paths((tmp_path / "artifact", tmp_path / "artifact" / "policy.json"))
    with pytest.raises(ValueError, match="different paths"):
        validate_write_paths((tmp_path / "same.json", tmp_path / "." / "same.json"))
    with pytest.raises(ValueError, match="different paths"):
        validate_write_paths((tmp_path / "Artifact.json", tmp_path / "artifact.json"))

    assert protected.read_text(encoding="utf-8") == "source"


def test_existing_directory_is_not_a_write_destination(tmp_path: Path) -> None:
    destination = tmp_path / "directory"
    destination.mkdir()

    with pytest.raises(ValueError, match="regular file or absent"):
        validate_write_paths((destination,))
