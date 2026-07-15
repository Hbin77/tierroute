# SPDX-License-Identifier: Apache-2.0
"""Require Apache-2.0 SPDX identifiers in tracked or untracked project files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SPDX_IDENTIFIER = "SPDX-License-Identifier: Apache-2.0"
COMMENTABLE_SUFFIXES = frozenset({".lock", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"})
COMMENTABLE_NAMES = frozenset({".gitignore", "Makefile"})
HEADER_LINE_LIMIT = 10


def _project_files(repository: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip() or "no diagnostic output"
        raise RuntimeError(f"git ls-files failed: {detail}")
    return tuple(
        repository / Path(raw_path.decode("utf-8"))
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    )


def _is_commentable_source(path: Path) -> bool:
    return path.name in COMMENTABLE_NAMES or path.suffix.lower() in COMMENTABLE_SUFFIXES


def _has_spdx_header(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as stream:
            header = "".join(next(stream, "") for _ in range(HEADER_LINE_LIMIT))
    except (OSError, UnicodeError):
        return False
    return SPDX_IDENTIFIER in header


def main() -> int:
    """Check versionable source files and return a shell-friendly status."""

    repository = Path(__file__).resolve().parents[1]
    try:
        candidates = tuple(
            path for path in _project_files(repository) if _is_commentable_source(path)
        )
    except RuntimeError as error:
        print(f"SPDX check error: {error}", file=sys.stderr)
        return 2

    missing = [path.relative_to(repository) for path in candidates if not _has_spdx_header(path)]
    if missing:
        print("Missing Apache-2.0 SPDX header:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1

    print(f"SPDX check passed: {len(candidates)} versionable commentable files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
