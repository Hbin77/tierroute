# SPDX-License-Identifier: Apache-2.0
"""Build the fixed project-owned C11 prepared-session sidecar without downloads."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

# The ridge and prepared binaries deliberately share the already-reviewed secure
# compiler/snapshot/publisher implementation. Only the source path is different,
# and this entry point fixes and validates it below; no caller-supplied source and
# no PATH compiler discovery can enter the build.
import build_native_ridge as _secure_build

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _REPOSITORY_ROOT / "native" / "tierroute_prepared.c"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="absolute path for a new binary")
    parser.add_argument("--compiler", required=True, help="absolute compiler executable path")
    arguments = parser.parse_args(argv)
    try:
        output = _safe_output_path(arguments.output)
        compiler = _secure_build._safe_compiler_path(arguments.compiler)
        source = _safe_source_path()
        result = _secure_build._compile(source=source, output=output, compiler=compiler)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _safe_output_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError("--output must be a non-empty path without NUL bytes")
    if raw_path.startswith(("//", "\\\\")):
        raise ValueError("--output must not use a UNC or device-style path")
    output = Path(raw_path)
    if not output.is_absolute():
        raise ValueError("--output must be absolute")
    output = Path(os.path.abspath(output))
    if output == _SOURCE:
        raise ValueError("--output must not replace the fixed prepared C source")
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("--output must name a path that does not already exist")
    try:
        parent_metadata = output.parent.lstat()
    except OSError as error:
        raise ValueError(f"--output parent cannot be inspected: {error}") from error
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("--output parent must be an existing non-symlink directory")
    return output


def _safe_source_path() -> Path:
    try:
        metadata = _SOURCE.lstat()
    except OSError as error:
        raise ValueError(f"fixed prepared native source cannot be inspected: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("fixed prepared native source must be a regular non-symlink file")
    return _SOURCE


if __name__ == "__main__":
    sys.exit(main())
