# SPDX-License-Identifier: Apache-2.0
"""Regression tests for bundled and vendored dependency license inspection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import ModuleType
from typing import Any

import pytest


def _load_license_gate() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "check_licenses.py"
    spec = importlib.util.spec_from_file_location("tierroute_check_licenses", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load license gate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_licenses = _load_license_gate()


def test_reviewed_permissive_license_hashes_are_exactly_pinned() -> None:
    assert check_licenses._REVIEWED_PERMISSIVE_LICENSE_DOCUMENT_SHA256 == frozenset(
        {
            "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf",
            "808e10c8a6ab8deb149ff9b3fb19f447a808094606d712a9ca57fead3552599d",
        }
    )


@dataclass
class _FakeDistribution:
    root: Path
    name: str = "fixture-package"
    version: str = "1.0"
    listed_files: tuple[PurePath, ...] | None = None

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}

    @property
    def files(self) -> list[PurePath]:
        if self.listed_files is not None:
            return list(self.listed_files)
        return [path.relative_to(self.root) for path in self.root.rglob("*") if path.is_file()]

    def locate_file(self, path: Any) -> Path:
        return self.root / str(path)


@pytest.mark.parametrize(
    "path",
    [
        PurePath("COPYING3"),
        PurePath("COPYING.RUNTIME"),
        PurePath("LICENCE.txt"),
        PurePath("package", "licenses", "component.txt"),
        PurePath("package.dist-info", "LICENSES", "GPL-3.0.txt"),
        PurePath("NOTICE-THIRD-PARTY.txt"),
        PurePath("THIRD-PARTY-LICENSES.txt"),
        PurePath("LICENSES.txt"),
        PurePath("LICENCES.md"),
        PurePath("COPYRIGHTS.txt"),
    ],
)
def test_license_evidence_discovery_covers_common_layouts(path: PurePath) -> None:
    assert check_licenses._is_license_document(path)


def test_deep_scan_rejects_vendored_lgpl_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "package" / "_vendor" / "tool-2.0.dist-info" / "METADATA"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "Metadata-Version: 2.1\nName: tool\nVersion: 2.0\nLicense: LGPLv3\n\n",
        encoding="utf-8",
    )

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "bundled metadata" in violations[0]
    assert "LGPLv3" in violations[0]


def test_deep_scan_rejects_aggregated_gpl_license_text(tmp_path: Path) -> None:
    license_path = tmp_path / "package-1.0.dist-info" / "LICENSE.txt"
    license_path.parent.mkdir(parents=True)
    license_path.write_text(
        "Bundled component\nLicense: GPL-3.0-only\n",
        encoding="utf-8",
    )

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "bundled license document" in violations[0]


def test_deep_scan_rejects_sentence_form_gpl_declaration(tmp_path: Path) -> None:
    license_path = tmp_path / "package-1.0.dist-info" / "LICENSE"
    license_path.parent.mkdir(parents=True)
    license_path.write_text(
        "Component is distributed under GPL-3.0-only.\n",
        encoding="utf-8",
    )

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "GPL/LGPL/AGPL-family terms" in violations[0]


@pytest.mark.parametrize(
    "declaration",
    [
        "Component is licensed under GPL.",
        "Component is distributed under LGPL.",
        "Component is governed by GNU General Public License.",
    ],
)
def test_deep_scan_rejects_unversioned_sentence_form_gpl_declaration(
    tmp_path: Path,
    declaration: str,
) -> None:
    license_path = tmp_path / "package-1.0.dist-info" / "LICENSE"
    license_path.parent.mkdir(parents=True)
    license_path.write_text(f"{declaration}\n", encoding="utf-8")

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "GPL/LGPL/AGPL-family terms" in violations[0]


def test_reviewed_permissive_license_hash_exception_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    license_path = tmp_path / "package-1.0.dist-info" / "LICENSE"
    license_path.parent.mkdir(parents=True)
    reviewed_document = "Component was previously distributed under GPL.\n"
    reviewed_digest = hashlib.sha256(reviewed_document.encode("utf-8")).hexdigest()
    monkeypatch.setattr(
        check_licenses,
        "_REVIEWED_PERMISSIVE_LICENSE_DOCUMENT_SHA256",
        frozenset({reviewed_digest}),
    )
    license_path.write_text(reviewed_document, encoding="utf-8")

    assert (
        check_licenses._deep_license_violations(  # type: ignore[arg-type]
            [_FakeDistribution(tmp_path)]
        )
        == []
    )
    license_path.write_text(f"{reviewed_document}modified\n", encoding="utf-8")

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "GPL/LGPL/AGPL-family terms" in violations[0]


def test_installed_python310_typing_extensions_license_hash_is_reviewed() -> None:
    if sys.version_info >= (3, 11):
        pytest.skip("typing_extensions is a locked Python 3.10 compatibility dependency")
    distribution = importlib.metadata.distribution("typing_extensions")
    license_file = next(
        file
        for file in distribution.files or []
        if tuple(part.casefold() for part in file.parts[-2:]) == ("licenses", "license")
    )
    payload = Path(distribution.locate_file(license_file)).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    assert digest in check_licenses._REVIEWED_PERMISSIVE_LICENSE_DOCUMENT_SHA256
    assert not check_licenses._document_declares_banned_license(payload.decode("utf-8"))


def test_installed_pip_distlib_license_evidence_is_reviewed() -> None:
    distribution = importlib.metadata.distribution("pip")
    assert distribution.version == "26.1.2"
    license_files = [
        file
        for file in distribution.files or []
        if "/".join(file.parts).casefold().endswith("pip/_vendor/distlib/license.txt")
    ]
    assert len(license_files) == 2

    payloads = [Path(distribution.locate_file(file)).read_bytes() for file in license_files]
    assert {len(payload) for payload in payloads} == {14_531}
    assert {hashlib.sha256(payload).hexdigest() for payload in payloads} == {
        "808e10c8a6ab8deb149ff9b3fb19f447a808094606d712a9ca57fead3552599d"
    }
    assert all(
        not check_licenses._document_declares_banned_license(payload.decode("utf-8"))
        for payload in payloads
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "other bundled components that are licensed under GPL.",
        "third-party dependency that is licensed under GPL.",
        "external dependency which is governed by LGPL.",
        "This component is currently and previously licensed under GPL.",
        "other bundled components that are licensed under GPL-3.0-only.",
        "third-party modules which are licensed under AGPLv3.",
    ],
)
def test_deep_scan_rejects_declarations_disguised_as_external_references(
    tmp_path: Path,
    declaration: str,
) -> None:
    license_path = tmp_path / "THIRD-PARTY-LICENSES.txt"
    license_path.write_text(f"{declaration}\n", encoding="utf-8")

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "GPL/LGPL/AGPL-family terms" in violations[0]


def test_deep_scan_rejects_bare_versioned_gpl_identifier(tmp_path: Path) -> None:
    license_path = tmp_path / "package-1.0.dist-info" / "LICENSE"
    license_path.parent.mkdir(parents=True)
    license_path.write_text("LGPLv3\n", encoding="utf-8")

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "GPL/LGPL/AGPL-family terms" in violations[0]


def test_deep_scan_rejects_spdx_license_directory_filename(tmp_path: Path) -> None:
    license_path = tmp_path / "package-1.0.dist-info" / "LICENSES" / "GPL-3.0.txt"
    license_path.parent.mkdir(parents=True)
    license_path.write_text("license text fixture\n", encoding="utf-8")

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "bundled license path" in violations[0]


@pytest.mark.parametrize(
    "relative",
    [
        PurePath("package", "licenses", "component.txt"),
        PurePath("THIRD-PARTY-LICENSES.txt"),
        PurePath("NOTICE-THIRD-PARTY.txt"),
        PurePath("LICENSES.txt"),
        PurePath("LICENCES.md"),
        PurePath("COPYRIGHTS.txt"),
    ],
)
def test_deep_scan_rejects_gpl_in_common_third_party_evidence(
    tmp_path: Path,
    relative: PurePath,
) -> None:
    license_path = tmp_path / relative
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_text("License: GPL-3.0-only\n", encoding="utf-8")

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "GPL/LGPL/AGPL-family terms" in violations[0]


def test_deep_scan_does_not_confuse_compatibility_reference_with_gpl_license(
    tmp_path: Path,
) -> None:
    license_path = tmp_path / "package-1.0.dist-info" / "LICENSE.txt"
    license_path.parent.mkdir(parents=True)
    license_path.write_text(
        "PSF License\nThis permissive license is compatible with GPL-3.0.\n",
        encoding="utf-8",
    )

    assert (
        check_licenses._deep_license_violations(  # type: ignore[arg-type]
            [_FakeDistribution(tmp_path)]
        )
        == []
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "component-a: GPL",
        "component-a | GNU General Public License",
        "component-a: GPL-3.0; compatible with our policy",
    ],
)
def test_deep_scan_rejects_gpl_inventory_declaration_bypasses(
    tmp_path: Path,
    declaration: str,
) -> None:
    license_path = tmp_path / "THIRD-PARTY-LICENSES.txt"
    license_path.write_text(f"{declaration}\n", encoding="utf-8")

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "GPL/LGPL/AGPL-family terms" in violations[0]


def test_deep_scan_accepts_permissive_backend_and_vendored_metadata(tmp_path: Path) -> None:
    backend_license = tmp_path / "backend-1.0.dist-info" / "licenses" / "LICENSE"
    backend_license.parent.mkdir(parents=True)
    backend_license.write_text("BSD 3-Clause License\n", encoding="utf-8")
    vendored_metadata = tmp_path / "backend" / "vendor" / "parser-1.0.dist-info" / "METADATA"
    vendored_metadata.parent.mkdir(parents=True)
    vendored_metadata.write_text(
        "Metadata-Version: 2.1\nName: parser\nVersion: 1.0\nLicense: MIT\n\n",
        encoding="utf-8",
    )

    assert (
        check_licenses._deep_license_violations(  # type: ignore[arg-type]
            [_FakeDistribution(tmp_path)]
        )
        == []
    )


def test_deep_scan_rejects_unreviewed_vendored_metadata_license(tmp_path: Path) -> None:
    metadata_path = tmp_path / "package" / "_vendor" / "tool-2.0.dist-info" / "METADATA"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "Metadata-Version: 2.1\nName: tool\nVersion: 2.0\nLicense: Unreviewed-1.0\n\n",
        encoding="utf-8",
    )

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "unreviewed license" in violations[0]


def test_deep_scan_rejects_vendored_metadata_without_license(tmp_path: Path) -> None:
    metadata_path = tmp_path / "package" / "_vendor" / "tool-2.0.dist-info" / "METADATA"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "Metadata-Version: 2.1\nName: tool\nVersion: 2.0\n\n",
        encoding="utf-8",
    )

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [_FakeDistribution(tmp_path)]
    )

    assert len(violations) == 1
    assert "no reviewable license declaration" in violations[0]


def test_deep_scan_rejects_manifest_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "installed"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "LICENSE").write_text("MIT\n", encoding="utf-8")
    distribution = _FakeDistribution(
        root,
        listed_files=(PurePath("..", "outside", "LICENSE"),),
    )

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [distribution]
    )

    assert len(violations) == 1
    assert "escapes or has an unresolvable installed path" in violations[0]


def test_deep_scan_rejects_symlinked_license_evidence(tmp_path: Path) -> None:
    root = tmp_path / "installed"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("MIT\n", encoding="utf-8")
    try:
        (root / "LICENSE").symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    distribution = _FakeDistribution(root, listed_files=(PurePath("LICENSE"),))

    violations = check_licenses._deep_license_violations(  # type: ignore[arg-type]
        [distribution]
    )

    assert len(violations) == 1
    assert "is not a regular file" in violations[0]


def test_bounded_reader_detects_replacement_between_stat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "LICENSE"
    replacement = tmp_path / "replacement"
    evidence.write_text("MIT original\n", encoding="utf-8")
    replacement.write_text("MIT replacement\n", encoding="utf-8")
    original_open = os.open

    def replacing_open(path: object, flags: int) -> int:
        replacement.replace(evidence)
        return original_open(path, flags)

    monkeypatch.setattr(check_licenses.os, "open", replacing_open)

    with pytest.raises(check_licenses._AuditEvidenceError, match="changed while opening"):
        check_licenses._read_regular_evidence(evidence)


def test_bounded_reader_rejects_oversized_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "LICENSE"
    evidence.write_bytes(b"12345")
    monkeypatch.setattr(check_licenses, "_MAX_AUDIT_FILE_BYTES", 4)

    with pytest.raises(check_licenses._AuditEvidenceError, match="exceeds 4 bytes"):
        check_licenses._read_regular_evidence(evidence)
