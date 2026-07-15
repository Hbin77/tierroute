# SPDX-License-Identifier: Apache-2.0
"""Fail when installed distributions use licenses outside the project allowlist."""

from __future__ import annotations

import json
import re
import subprocess
import sys
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
_EXPRESSION_OPERATOR = re.compile(r"\s+(?:AND|OR|WITH)\s+|\s*;\s*", re.IGNORECASE)


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
