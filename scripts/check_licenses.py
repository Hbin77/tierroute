# SPDX-License-Identifier: Apache-2.0
"""Fail when installed distributions use licenses outside the project allowlist."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from email.parser import Parser
from importlib import metadata
from pathlib import Path, PurePath
from typing import Any

ALLOWED_LICENSE_TERMS = frozenset(
    {
        "Apache Software License",
        "Apache-2.0",
        "BSD License",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "ISC License (ISCL)",
        "MIT",
        "MIT License",
        "PSF-2.0",
        "Python Software Foundation License",
    }
)

_BANNED_LICENSE_FAMILY = re.compile(
    r"(?:\b(?:AGPL|LGPL|GPL)(?:[- v]?\d+(?:\.\d+)*)?"
    r"(?:-only|-or-later|\+)?\b|"
    r"\bGNU\s+(?:Affero\s+|Lesser\s+)?General\s+Public\s+License\b)",
    flags=re.IGNORECASE,
)
_BANNED_LICENSE_DOCUMENT = re.compile(
    r"^\s*(?:"
    r"GNU\s+(?:Affero\s+|Lesser\s+)?General\s+Public\s+License\b|"
    r"(?:License|SPDX-License-Identifier)\s*:\s*"
    r"(?:AGPL|LGPL|GPL)(?:[- v]?\d+(?:\.\d+)*)?(?:-only|-or-later|\+)?\b"
    r")",
    flags=re.IGNORECASE | re.MULTILINE,
)
_LICENSE_DECLARATION_CONTEXT = re.compile(
    r"\b(?:licensed|distributed|released|governed)\b[^\n]{0,80}"
    r"\b(?:under|by)\b[^\n]{0,80}$",
    flags=re.IGNORECASE,
)
_BANNED_VERSIONED_IDENTIFIER = re.compile(
    r"\b(?:AGPL|LGPL|GPL)[- v]?\d+(?:\.\d+)*(?:-only|-or-later|\+)?\b",
    flags=re.IGNORECASE,
)
_LOCAL_COMPATIBILITY_PREFIX = re.compile(
    r"\bcompatib(?:le|ility)\s+(?:with|under)(?:\s+the)?\s*$",
    flags=re.IGNORECASE,
)
_LOCAL_COMPATIBILITY_SUFFIX = re.compile(
    r"^\s*(?:-\s*compatible\b|(?:is\s+)?compatible\b|compatibility\b)",
    flags=re.IGNORECASE,
)
_COMPONENT_LICENSE_DELIMITER = re.compile(r"[:|]\s*$")
_STANDALONE_LICENSE_PREFIX = re.compile(r"^\s*(?:[-*+]\s*)?$")
_STANDALONE_LICENSE_SUFFIX = re.compile(r"^\s*[.,;:()\[\]{}]*\s*$")
_BANNED_LICENSE_FILENAME = re.compile(
    r"^(?:AGPL|LGPL|GPL)(?:[-_ v]?\d+(?:\.\d+)*)?(?:-only|-or-later|\+)?"
    r"(?:\.[a-z0-9]+)?$",
    flags=re.IGNORECASE,
)
_EXPRESSION_OPERATOR = re.compile(r"\s+(?:AND|OR|WITH)\s+|\s*;\s*", re.IGNORECASE)
_MAX_AUDIT_FILE_BYTES = 4 * 1024 * 1024
_MAX_AUDIT_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_AUDIT_FILES = 10_000
# Exact PSF-2.0-family license evidence shipped by typing_extensions==4.16.0.
# Its compatibility footnotes mention GPL-covered *other* software and a
# historical Python 1.6.1 distribution. A hash exception is intentionally
# narrower than prose heuristics: any modified evidence is scanned normally.
_REVIEWED_PERMISSIVE_LICENSE_DOCUMENT_SHA256 = frozenset(
    {"3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf"}
)


def _is_allowlisted(license_expression: str) -> bool:
    if license_expression in ALLOWED_LICENSE_TERMS:
        return True
    terms = [term.strip().strip("()") for term in _EXPRESSION_OPERATOR.split(license_expression)]
    return bool(terms) and all(term in ALLOWED_LICENSE_TERMS for term in terms)


def _installed_distributions() -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "piplicenses",
        "--format=json",
        "--from=mixed",
        "--with-system",
        "--with-urls",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"pip-licenses failed with exit code {completed.returncode}: {detail}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("pip-licenses did not emit valid JSON") from error
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise RuntimeError("pip-licenses JSON must be a list of distribution records")
    return payload


def _is_license_document(path: PurePath) -> bool:
    name = path.name.casefold()
    parent_parts = path.parts[:-1]
    parent_names = {part.casefold() for part in parent_parts}
    dist_info_license_tree = any(
        part.casefold().endswith(".dist-info")
        and "licenses" in {child.casefold() for child in parent_parts[index + 1 :]}
        for index, part in enumerate(parent_parts)
    )
    text_in_license_tree = (
        bool({"license", "licenses"} & parent_names)
        and "__pycache__" not in parent_names
        and path.suffix.casefold() in {"", ".html", ".md", ".rst", ".text", ".txt"}
    )
    normalized_name = re.sub(r"[^a-z0-9]", "", name)
    return (
        dist_info_license_tree
        or text_in_license_tree
        or path.suffix.casefold() == ".license"
        or normalized_name.startswith(("noticethirdparty", "thirdpartylicense", "thirdpartynotice"))
        or _BANNED_LICENSE_FILENAME.fullmatch(name) is not None
        or name
        in {
            "copying",
            "copyright",
            "copyrights",
            "licence",
            "licences",
            "license",
            "licenses",
            "notice",
            "notices",
        }
        or re.fullmatch(r"copying(?:\d+|[._-].*)", name) is not None
        or name.startswith(
            (
                "copyright.",
                "licence.",
                "licence-",
                "licences.",
                "licences-",
                "license.",
                "license-",
                "licenses.",
                "licenses-",
                "notice.",
                "notice-",
                "notice_",
                "notices.",
                "notices-",
                "copyrights.",
                "copyrights-",
            )
        )
    )


def _is_nested_metadata(path: PurePath) -> bool:
    return (
        len(path.parts) > 2
        and path.name == "METADATA"
        and path.parts[-2].casefold().endswith(".dist-info")
    )


def _metadata_license_headers(document: str) -> tuple[str, ...]:
    message = Parser().parsestr(document, headersonly=True)
    classifiers = tuple(
        value.rsplit("::", maxsplit=1)[-1].strip()
        for value in (message.get_all("Classifier", failobj=[]) or [])
        if value.casefold().startswith("license ::")
    )
    values = [
        *(message.get_all("License-Expression", failobj=[]) or []),
        *(message.get_all("License", failobj=[]) or []),
        *classifiers,
    ]
    return tuple(value.strip() for value in values if value.strip())


def _display_expression(expression: str, *, limit: int = 160) -> str:
    compact = " ".join(expression.split())
    if len(compact) <= limit:
        return repr(compact)
    return repr(f"{compact[: limit - 3]}...")


def _document_declares_banned_license(document: str) -> bool:
    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
    if digest in _REVIEWED_PERMISSIVE_LICENSE_DOCUMENT_SHA256:
        return False
    if _BANNED_LICENSE_DOCUMENT.search(document):
        return True

    for line in document.splitlines():
        for match in _BANNED_LICENSE_FAMILY.finditer(line):
            prefix = line[: match.start()]
            suffix = line[match.end() :]

            # Permissive licenses often describe themselves as "GPL-compatible".
            # Exempt only the compatibility phrase touching this exact match; a
            # later phrase must not hide declarations such as
            # "component: GPL-3.0; compatible with our policy".
            if _LOCAL_COMPATIBILITY_PREFIX.search(prefix) or _LOCAL_COMPATIBILITY_SUFFIX.match(
                suffix
            ):
                continue

            declaration_context = _LICENSE_DECLARATION_CONTEXT.search(prefix)
            if declaration_context:
                return True

            if (
                _BANNED_VERSIONED_IDENTIFIER.fullmatch(match.group(0))
                or _COMPONENT_LICENSE_DELIMITER.search(prefix)
                or (
                    _STANDALONE_LICENSE_PREFIX.fullmatch(prefix)
                    and _STANDALONE_LICENSE_SUFFIX.fullmatch(suffix)
                )
            ):
                return True
    return False


class _AuditEvidenceError(RuntimeError):
    """Raised when installed license evidence cannot be read safely."""


def _read_regular_evidence(location: Path) -> tuple[str, int]:
    """Read one bounded regular file without following a final symlink."""

    try:
        before = os.lstat(location)
    except OSError as error:
        raise _AuditEvidenceError(f"cannot stat: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise _AuditEvidenceError("is not a regular file")
    if before.st_size > _MAX_AUDIT_FILE_BYTES:
        raise _AuditEvidenceError(f"exceeds {_MAX_AUDIT_FILE_BYTES:,} bytes")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(location, flags)
    except OSError as error:
        raise _AuditEvidenceError(f"cannot open safely: {error}") from error
    try:
        after = os.fstat(descriptor)
        if not stat.S_ISREG(after.st_mode):
            raise _AuditEvidenceError("changed to a non-regular file while opening")
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise _AuditEvidenceError("changed while opening")
        if after.st_size > _MAX_AUDIT_FILE_BYTES:
            raise _AuditEvidenceError(f"exceeds {_MAX_AUDIT_FILE_BYTES:,} bytes")

        chunks: list[bytes] = []
        remaining = _MAX_AUDIT_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_AUDIT_FILE_BYTES:
            raise _AuditEvidenceError(f"exceeds {_MAX_AUDIT_FILE_BYTES:,} bytes")
        finished = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            finished.st_dev,
            finished.st_ino,
            finished.st_size,
            finished.st_mtime_ns,
        ):
            raise _AuditEvidenceError("changed while reading")
    except OSError as error:
        raise _AuditEvidenceError(f"cannot read safely: {error}") from error
    finally:
        os.close(descriptor)

    try:
        return payload.decode("utf-8"), len(payload)
    except UnicodeError as error:
        raise _AuditEvidenceError("is not valid UTF-8 text") from error


def _deep_license_violations(
    distributions: list[metadata.Distribution] | None = None,
) -> list[str]:
    """Inspect bundled license files and nested distribution metadata.

    Top-level package metadata is insufficient for binary or vendored wheels:
    NumPy's metadata says BSD while platform wheels can include compiler
    runtimes, and setuptools' metadata says MIT while it vendors LGPL code.
    This bounded scan deliberately fails closed on unreadable audit evidence.
    """

    installed = list(metadata.distributions()) if distributions is None else distributions
    violations: list[str] = []
    audited_paths: set[Path] = set()
    total_bytes = 0
    file_count = 0

    for distribution in installed:
        name = distribution.metadata.get("Name") or "<unknown>"
        version = distribution.version or "<unknown>"
        try:
            distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
        except OSError as error:
            violations.append(f"{name}=={version}: cannot resolve installed root: {error}")
            continue
        files = distribution.files
        if not files:
            violations.append(f"{name}=={version}: installed file manifest is unavailable or empty")
            continue
        for relative in files:
            relative_path = PurePath(str(relative))
            is_metadata = _is_nested_metadata(relative_path)
            if not is_metadata and not _is_license_document(relative_path):
                continue
            unresolved = Path(distribution.locate_file(relative))
            try:
                location = unresolved.parent.resolve(strict=True) / unresolved.name
                location.relative_to(distribution_root)
            except (OSError, ValueError) as error:
                violations.append(
                    f"{name}=={version}: audit evidence {relative} escapes or has an "
                    f"unresolvable installed path: {error}"
                )
                continue
            if location in audited_paths:
                continue
            audited_paths.add(location)
            file_count += 1
            if file_count > _MAX_AUDIT_FILES:
                violations.append(f"deep license audit exceeds {_MAX_AUDIT_FILES:,} evidence files")
                return violations
            try:
                document, size = _read_regular_evidence(location)
            except _AuditEvidenceError as error:
                violations.append(
                    f"{name}=={version}: cannot safely read audit evidence {relative}: {error}"
                )
                continue
            total_bytes += size
            if total_bytes > _MAX_AUDIT_TOTAL_BYTES:
                violations.append(
                    f"deep license audit exceeds {_MAX_AUDIT_TOTAL_BYTES:,} total evidence bytes"
                )
                return violations

            if _BANNED_LICENSE_FILENAME.fullmatch(relative_path.name):
                violations.append(
                    f"{name}=={version}: bundled license path {relative} names a "
                    "GPL/LGPL/AGPL-family license"
                )
                continue

            if is_metadata:
                expressions = _metadata_license_headers(document)
                if not expressions:
                    violations.append(
                        f"{name}=={version}: bundled metadata {relative} has no "
                        "reviewable license declaration"
                    )
                for expression in expressions:
                    if _BANNED_LICENSE_FAMILY.search(expression):
                        violations.append(
                            f"{name}=={version}: bundled metadata {relative} declares "
                            "banned GPL/LGPL/AGPL-family license "
                            f"{_display_expression(expression)}"
                        )
                    elif not _is_allowlisted(expression):
                        violations.append(
                            f"{name}=={version}: bundled metadata {relative} declares "
                            f"unreviewed license {_display_expression(expression)}"
                        )
            elif _document_declares_banned_license(document):
                violations.append(
                    f"{name}=={version}: bundled license document {relative} "
                    "contains GPL/LGPL/AGPL-family terms"
                )
    return violations


def main() -> int:
    """Check every installed distribution and return a shell-friendly status."""

    try:
        distributions = _installed_distributions()
    except RuntimeError as error:
        print(f"License check error: {error}", file=sys.stderr)
        return 2

    violations: list[str] = []
    saw_tierroute = False
    unique_distributions: set[tuple[str, str, str]] = set()
    for row in distributions:
        name = str(row.get("Name") or "<unknown>").strip()
        version = str(row.get("Version") or "<unknown>").strip()
        license_expression = str(row.get("License") or "UNKNOWN").strip()
        unique_distributions.add((name.casefold(), version, license_expression))
        saw_tierroute = saw_tierroute or name.casefold() == "tierroute"

        if _BANNED_LICENSE_FAMILY.search(license_expression):
            violations.append(
                f"{name}=={version}: banned GPL/LGPL/AGPL-family license {license_expression!r}"
            )
        elif not _is_allowlisted(license_expression):
            violations.append(
                f"{name}=={version}: license {license_expression!r} is not allowlisted"
            )

    violations.extend(_deep_license_violations())

    if not saw_tierroute:
        violations.append("tierroute is not installed; run `python -m pip install -e .` first")

    if violations:
        print("License policy violations:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print(
        "License check passed: "
        f"{len(unique_distributions)} installed distributions use allowlisted licenses."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
