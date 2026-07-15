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


class DownloadIntegrityError(RuntimeError):
    """The downloaded bytes do not match the pinned RouterBench artifact."""


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
    """Return whether a regular file has the expected size and SHA-256 digest."""

    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size != expected_size:
            return False
        actual_sha256 = sha256_file(candidate, chunk_size=chunk_size)
    except OSError:
        return False
    return hmac.compare_digest(actual_sha256, expected_sha256.lower())


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
    if verify_file(
        destination,
        expected_size=ROUTERBENCH_SIZE,
        expected_sha256=ROUTERBENCH_SHA256,
        chunk_size=chunk_size,
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    part_path.unlink(missing_ok=True)
    request = Request(
        ROUTERBENCH_URL,
        headers={"User-Agent": "tierroute-routerbench-downloader/0.1"},
    )
    digest = hashlib.sha256()
    downloaded_size = 0

    try:
        with urlopen(request, timeout=timeout) as response, part_path.open("xb") as output:
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

        os.replace(part_path, destination)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

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

    destination = download_routerbench(args.output)
    print(f"Verified RouterBench artifact: {destination}")
    print(f"SHA-256: {ROUTERBENCH_SHA256}")
    print("Dataset license: NOASSERTION; review the upstream terms before use or redistribution.")


if __name__ == "__main__":
    main()
