# SPDX-License-Identifier: Apache-2.0
"""Download the pinned RouterBench zero-shot artifact with integrity checks.

RouterBench is opt-in data: it is not distributed with tierroute because the
dataset card does not declare a license (``NOASSERTION`` as of the pinned
revision).  This script uses only the Python standard library and never runs as
part of tierroute's inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import stat
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROUTERBENCH_FILENAME = "routerbench_0shot.pkl"
ROUTERBENCH_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
ROUTERBENCH_URL = (
    "https://huggingface.co/datasets/withmartian/routerbench/resolve/"
    f"{ROUTERBENCH_REVISION}/{ROUTERBENCH_FILENAME}?download=true"
)
ROUTERBENCH_SIZE = 99_567_659
ROUTERBENCH_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"
DEFAULT_DESTINATION = Path("data/routerbench") / ROUTERBENCH_FILENAME
DEFAULT_CHUNK_SIZE = 1024 * 1024
PRIVATE_FILE_MODE = 0o600


class DownloadIntegrityError(RuntimeError):
    """The downloaded bytes do not match the pinned RouterBench artifact."""


def _read_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if os.name == "posix":
        if not hasattr(os, "O_NOFOLLOW"):
            raise DownloadIntegrityError("this POSIX platform cannot enforce no-follow reads")
        flags |= os.O_NOFOLLOW
    return flags


def _file_identity(file_stat: os.stat_result) -> tuple[int, int]:
    return file_stat.st_dev, file_stat.st_ino


def _path_matches_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        file_stat = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(file_stat.st_mode) and _file_identity(file_stat) == identity


def _open_verified_regular_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    chunk_size: int,
) -> int | None:
    """Verify one no-follow regular-file descriptor and its current pathname."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    try:
        descriptor = os.open(path, _read_flags())
    except OSError:
        return None
    transferred = False
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_size != expected_size:
            return None
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, chunk_size):
            digest.update(chunk)
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256.lower()):
            return None
        if not _path_matches_identity(path, _file_identity(opened_stat)):
            return None
        transferred = True
        return descriptor
    finally:
        if not transferred:
            os.close(descriptor)


def _reuse_verified_private_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    chunk_size: int,
) -> bool:
    """Hash and chmod the same descriptor, rejecting symlinks and path swaps."""

    descriptor = _open_verified_regular_file(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        chunk_size=chunk_size,
    )
    if descriptor is None:
        return False
    try:
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        return _path_matches_identity(path, _file_identity(os.fstat(descriptor)))
    finally:
        os.close(descriptor)


def _create_private_part(destination: Path) -> tuple[int, Path, tuple[int, int]]:
    """Create an unpredictable same-directory staging file owned by this invocation."""

    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    path = Path(raw_path)
    identity: tuple[int, int] | None = None
    try:
        file_stat = os.fstat(descriptor)
        identity = _file_identity(file_stat)
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        if not stat.S_ISREG(file_stat.st_mode):
            raise DownloadIntegrityError("RouterBench staging descriptor is not a regular file")
        if not _path_matches_identity(path, identity):
            raise DownloadIntegrityError("RouterBench staging path changed during creation")
        return descriptor, path, identity
    except BaseException:
        os.close(descriptor)
        if identity is not None:
            _unlink_owned_path(path, identity)
        raise


def _unlink_owned_path(path: Path, identity: tuple[int, int]) -> None:
    """Remove only the exact staging inode created by this invocation."""

    if _path_matches_identity(path, identity):
        path.unlink(missing_ok=True)


def _reject_non_regular_destination(path: Path) -> None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DownloadIntegrityError(
            "cannot inspect RouterBench destination (path omitted)"
        ) from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise DownloadIntegrityError(
            "RouterBench destination must be absent or a regular file (path omitted)"
        )


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the lowercase SHA-256 digest of a file using bounded memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    expected_size: int = ROUTERBENCH_SIZE,
    expected_sha256: str = ROUTERBENCH_SHA256,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """Verify a no-follow regular file through one descriptor."""

    descriptor = _open_verified_regular_file(
        Path(path),
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        chunk_size=chunk_size,
    )
    if descriptor is None:
        return False
    os.close(descriptor)
    return True


def download_routerbench(
    destination: str | Path = DEFAULT_DESTINATION,
    *,
    timeout: float = 60.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """Download and atomically install the one pinned RouterBench artifact.

    A valid existing file is reused without a network request.  An invalid
    existing destination remains untouched unless a complete, verified
    ``.part`` file is ready to replace it.
    """

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    destination = Path(destination)
    _reject_non_regular_destination(destination)
    if _reuse_verified_private_file(
        destination,
        expected_size=ROUTERBENCH_SIZE,
        expected_sha256=ROUTERBENCH_SHA256,
        chunk_size=chunk_size,
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        ROUTERBENCH_URL,
        headers={"User-Agent": "tierroute-routerbench-downloader/0.1"},
    )
    digest = hashlib.sha256()
    downloaded_size = 0
    descriptor, part_path, part_identity = _create_private_part(destination)

    try:
        with (
            urlopen(request, timeout=timeout) as response,
            os.fdopen(descriptor, "wb", closefd=False) as output,
        ):
            while chunk := response.read(chunk_size):
                downloaded_size += len(chunk)
                if downloaded_size > ROUTERBENCH_SIZE:
                    raise DownloadIntegrityError(
                        f"RouterBench download exceeds the pinned size of {ROUTERBENCH_SIZE} bytes"
                    )
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        actual_sha256 = digest.hexdigest()
        if downloaded_size != ROUTERBENCH_SIZE:
            raise DownloadIntegrityError(
                "RouterBench size mismatch: "
                f"expected {ROUTERBENCH_SIZE}, got {downloaded_size} bytes"
            )
        if not hmac.compare_digest(actual_sha256, ROUTERBENCH_SHA256):
            raise DownloadIntegrityError(
                f"RouterBench SHA-256 mismatch: expected {ROUTERBENCH_SHA256}, got {actual_sha256}"
            )

        if not _path_matches_identity(part_path, part_identity):
            raise DownloadIntegrityError("RouterBench staging path identity changed")
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        os.replace(part_path, destination)
        if not _path_matches_identity(destination, part_identity):
            raise DownloadIntegrityError("RouterBench installed path identity changed")
        if not _reuse_verified_private_file(
            destination,
            expected_size=ROUTERBENCH_SIZE,
            expected_sha256=ROUTERBENCH_SHA256,
            chunk_size=chunk_size,
        ):
            raise DownloadIntegrityError(
                "RouterBench installed file failed post-replacement authentication"
            )
    finally:
        try:
            os.close(descriptor)
        finally:
            _unlink_owned_path(part_path, part_identity)

    return destination


def main() -> None:
    """Run the explicit, networked dataset download command."""

    parser = argparse.ArgumentParser(
        description="Download the pinned RouterBench 0-shot file after opt-in."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"destination path (default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args()

    try:
        download_routerbench(args.output)
    except DownloadIntegrityError as error:
        parser.exit(1, f"RouterBench download failed: {error}\n")
    except OSError:
        parser.exit(1, "RouterBench local/network operation failed (path omitted)\n")
    print("Verified RouterBench artifact (local path omitted)")
    print(f"SHA-256: {ROUTERBENCH_SHA256}")
    print("Dataset license: NOASSERTION; review the upstream terms before use or redistribution.")


if __name__ == "__main__":
    main()
