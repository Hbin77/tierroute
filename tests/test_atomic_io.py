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

    assert protected.read_text(encoding="utf-8") == "source"


def test_existing_directory_is_not_a_write_destination(tmp_path: Path) -> None:
    destination = tmp_path / "directory"
    destination.mkdir()

    with pytest.raises(ValueError, match="regular file or absent"):
        validate_write_paths((destination,))
