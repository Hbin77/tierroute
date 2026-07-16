# SPDX-License-Identifier: Apache-2.0
"""Build the project-owned C11 ridge sidecar without downloads or PATH discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _REPOSITORY_ROOT / "native" / "tierroute_ridge.c"
_BUILD_TIMEOUT_SECONDS = 180
MAX_BINARY_BYTES = 16 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="absolute path for a new binary")
    parser.add_argument("--compiler", required=True, help="absolute compiler executable path")
    arguments = parser.parse_args(argv)
    try:
        output = _safe_output_path(arguments.output)
        compiler = _safe_compiler_path(arguments.compiler)
        source = _safe_source_path()
        result = _compile(source=source, output=output, compiler=compiler)
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
        raise ValueError("--output must not replace the fixed C source")
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


def _safe_compiler_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError("--compiler must be a non-empty path without NUL bytes")
    if raw_path.startswith(("//", "\\\\")):
        raise ValueError("--compiler must not use a UNC or device-style path")
    compiler = Path(raw_path)
    if not compiler.is_absolute():
        raise ValueError("--compiler must be absolute; PATH discovery is disabled")
    compiler = Path(os.path.abspath(compiler))
    try:
        metadata = compiler.lstat()
    except OSError as error:
        raise ValueError(f"--compiler cannot be inspected: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("--compiler must be a regular non-symlink file")
    if os.name != "nt" and metadata.st_mode & 0o111 == 0:
        raise ValueError("--compiler is not executable")
    return compiler


def _safe_source_path() -> Path:
    try:
        metadata = _SOURCE.lstat()
    except OSError as error:
        raise ValueError(f"fixed native source cannot be inspected: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("fixed native source must be a regular non-symlink file")
    return _SOURCE


def _compile(*, source: Path, output: Path, compiler: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="tierroute-native-build-", dir=output.parent) as name:
        directory = Path(name)
        os.chmod(directory, stat.S_IRWXU)
        source_snapshot, source_digest = _snapshot_source(source, directory)
        temporary_output = directory / (
            "tierroute_ridge.exe" if os.name == "nt" else "tierroute_ridge"
        )
        if os.name == "nt":
            command = [
                str(compiler),
                "/nologo",
                "/std:c11",
                "/O2",
                "/MT",
                "/W4",
                "/WX",
                str(source_snapshot),
                f"/Fo:{directory / 'tierroute_ridge.obj'}",
                f"/Fe:{temporary_output}",
            ]
        else:
            command = [
                str(compiler),
                "-std=c11",
                "-O3",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Wconversion",
                "-Wsign-conversion",
                "-Wshadow",
                "-Werror",
                str(source_snapshot),
                "-lm",
                "-o",
                str(temporary_output),
            ]
        environment = os.environ.copy()
        environment["LC_ALL"] = "C"
        try:
            completed = subprocess.run(
                command,
                shell=False,
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_BUILD_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"native ridge compilation exceeded the {_BUILD_TIMEOUT_SECONDS}-second time limit"
            ) from error
        if completed.returncode != 0:
            stderr = completed.stderr[-16_384:].decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"native ridge compilation failed with status {completed.returncode}: {stderr}"
            )
        try:
            metadata = temporary_output.lstat()
        except OSError as error:
            raise RuntimeError(f"compiler did not create the requested binary: {error}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("compiler output is not a regular non-symlink file")
        if metadata.st_size == 0:
            raise RuntimeError("compiler output is empty")
        if metadata.st_size > MAX_BINARY_BYTES:
            raise RuntimeError(
                f"compiler output size {metadata.st_size} exceeds reviewed limit {MAX_BINARY_BYTES}"
            )
        if os.name != "nt":
            os.chmod(temporary_output, stat.S_IRUSR | stat.S_IXUSR)
        # Never publish the compiler-owned inode directly. A compiler may leave a
        # background process or writable descriptor behind after returning. Copy
        # its completed bytes into a separately owned, authenticated snapshot so
        # later mutations of the compiler output cannot change the published file.
        verified_output, digest = _snapshot_compiler_output(temporary_output, directory)
        _publish_new_file(verified_output, output, expected_digest=digest)
    return {
        "compiler": str(compiler),
        "output": str(output),
        "sha256": digest,
        "source": str(source),
        "source_sha256": source_digest,
    }


def _snapshot_source(source: Path, directory: Path) -> tuple[Path, str]:
    """Copy one descriptor-stable source inode into the private build directory."""

    path_metadata = source.lstat()
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        raise RuntimeError(f"cannot open fixed native source safely: {error}") from error
    snapshot = directory / "tierroute_ridge.c"
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        destination_descriptor = os.open(snapshot, destination_flags, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as error:
        os.close(source_descriptor)
        raise RuntimeError(f"cannot create private native source snapshot: {error}") from error
    digest = hashlib.sha256()
    try:
        opened_metadata = os.fstat(source_descriptor)
        _require_same_source(path_metadata, opened_metadata)
        while chunk := os.read(source_descriptor, 1024 * 1024):
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
        os.fsync(destination_descriptor)
        _require_stable_source(opened_metadata, os.fstat(source_descriptor))
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)
    try:
        final_path_metadata = source.lstat()
        _require_same_source(opened_metadata, final_path_metadata)
        _require_stable_source(opened_metadata, final_path_metadata)
    except (OSError, RuntimeError):
        raise RuntimeError("fixed native source changed while it was snapshotted") from None
    return snapshot, digest.hexdigest()


def _require_same_source(first: os.stat_result, second: os.stat_result) -> None:
    if (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino):
        raise RuntimeError("fixed native source inode changed during snapshot")
    if not stat.S_ISREG(second.st_mode):
        raise RuntimeError("fixed native source is no longer a regular file")


def _require_stable_source(first: os.stat_result, second: os.stat_result) -> None:
    first_mtime = getattr(first, "st_mtime_ns", int(first.st_mtime * 1_000_000_000))
    second_mtime = getattr(second, "st_mtime_ns", int(second.st_mtime * 1_000_000_000))
    if first.st_size != second.st_size or first_mtime != second_mtime:
        raise RuntimeError("fixed native source contents changed during snapshot")


def _require_stable_output(first: os.stat_result, second: os.stat_result) -> None:
    first_mtime = getattr(first, "st_mtime_ns", int(first.st_mtime * 1_000_000_000))
    second_mtime = getattr(second, "st_mtime_ns", int(second.st_mtime * 1_000_000_000))
    if first.st_size != second.st_size or first_mtime != second_mtime:
        raise RuntimeError("compiler output changed while hashing")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while copying an authenticated file")
        offset += written


def _snapshot_compiler_output(source: Path, directory: Path) -> tuple[Path, str]:
    """Copy compiler bytes once into a descriptor-stable, private snapshot."""

    path_metadata = source.lstat()
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(path_metadata.st_mode):
        raise RuntimeError("compiler output changed to an unsafe filesystem node")
    if path_metadata.st_size < 1 or path_metadata.st_size > MAX_BINARY_BYTES:
        raise RuntimeError("compiler output is outside the reviewed binary-size limit")

    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        raise RuntimeError(f"cannot open compiler output safely: {error}") from error

    snapshot = directory / "verified-tierroute-ridge"
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        destination_descriptor = os.open(
            snapshot,
            destination_flags,
            stat.S_IRUSR | stat.S_IWUSR,
        )
    except OSError as error:
        os.close(source_descriptor)
        raise RuntimeError(f"cannot create verified compiler-output snapshot: {error}") from error

    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        opened_metadata = os.fstat(source_descriptor)
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ):
            raise RuntimeError("compiler output inode changed before snapshot")
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise RuntimeError("compiler output is not a regular file during snapshot")
        if opened_metadata.st_size < 1 or opened_metadata.st_size > MAX_BINARY_BYTES:
            raise RuntimeError("compiler output is outside the reviewed binary-size limit")
        while chunk := os.read(source_descriptor, 1024 * 1024):
            copied_bytes += len(chunk)
            if copied_bytes > MAX_BINARY_BYTES:
                raise RuntimeError(
                    "compiler output exceeded the reviewed size limit during snapshot"
                )
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
        os.fsync(destination_descriptor)
        final_metadata = os.fstat(source_descriptor)
        _require_stable_output(opened_metadata, final_metadata)
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)

    if copied_bytes != opened_metadata.st_size:
        raise RuntimeError("compiler output byte count changed during snapshot")
    final_path_metadata = source.lstat()
    if (opened_metadata.st_dev, opened_metadata.st_ino) != (
        final_path_metadata.st_dev,
        final_path_metadata.st_ino,
    ):
        raise RuntimeError("compiler output inode changed during snapshot")
    _require_stable_output(opened_metadata, final_path_metadata)
    if os.name != "nt":
        os.chmod(snapshot, stat.S_IRUSR | stat.S_IXUSR)

    expected_digest = digest.hexdigest()
    if _sha256(snapshot) != expected_digest:
        raise RuntimeError("verified compiler-output snapshot failed its integrity check")
    return snapshot, expected_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    path_metadata = path.lstat()
    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(path_metadata.st_mode):
        raise RuntimeError("compiler output changed to an unsafe filesystem node")
    if path_metadata.st_size < 1 or path_metadata.st_size > MAX_BINARY_BYTES:
        raise RuntimeError("compiler output is outside the reviewed binary-size limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    copied_bytes = 0
    try:
        opened_metadata = os.fstat(descriptor)
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        ):
            raise RuntimeError("compiler output inode changed while hashing")
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise RuntimeError("compiler output is not a regular file while hashing")
        if opened_metadata.st_size < 1 or opened_metadata.st_size > MAX_BINARY_BYTES:
            raise RuntimeError("compiler output is outside the reviewed binary-size limit")
        while chunk := os.read(descriptor, 1024 * 1024):
            copied_bytes += len(chunk)
            if copied_bytes > MAX_BINARY_BYTES:
                raise RuntimeError("compiler output exceeded the reviewed size limit while read")
            digest.update(chunk)
        final_metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require_stable_output(opened_metadata, final_metadata)
    if copied_bytes != opened_metadata.st_size:
        raise RuntimeError("compiler output byte count changed while hashing")
    final_path_metadata = path.lstat()
    if (opened_metadata.st_dev, opened_metadata.st_ino) != (
        final_path_metadata.st_dev,
        final_path_metadata.st_ino,
    ):
        raise RuntimeError("compiler output inode changed while hashing")
    _require_stable_output(opened_metadata, final_path_metadata)
    return digest.hexdigest()


def _publish_new_file(source: Path, destination: Path, *, expected_digest: str) -> None:
    """Publish authenticated bytes without overwriting and re-authenticate the result."""

    source_metadata = source.lstat()
    created_metadata: os.stat_result | None = None
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise RuntimeError(
            "--output appeared during compilation; refusing to overwrite it"
        ) from error
    except OSError:
        created_metadata = _copy_new_file(source, destination, expected_digest=expected_digest)
    else:
        try:
            linked_metadata = destination.lstat()
        except OSError:
            _remove_created_file(destination, source_metadata)
            raise
        if (source_metadata.st_dev, source_metadata.st_ino) != (
            linked_metadata.st_dev,
            linked_metadata.st_ino,
        ):
            raise RuntimeError("published output no longer names the verified snapshot")
        created_metadata = linked_metadata

    if created_metadata is None:
        raise RuntimeError("publisher did not create an output inode")
    try:
        if _sha256(destination) != expected_digest:
            raise RuntimeError("published output does not match its manifest SHA-256")
    except Exception:
        _remove_created_file(destination, created_metadata)
        raise


def _copy_new_file(
    source: Path,
    destination: Path,
    *,
    expected_digest: str,
) -> os.stat_result:
    """Fallback publisher that authenticates bytes read from the snapshot."""

    source_metadata = source.lstat()
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, source_flags)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(destination, flags, stat.S_IRUSR | stat.S_IWUSR)
    except FileExistsError as error:
        os.close(source_descriptor)
        raise RuntimeError(
            "--output appeared during compilation; refusing to overwrite it"
        ) from error
    except Exception:
        os.close(source_descriptor)
        raise

    try:
        created_metadata = os.fstat(descriptor)
    except Exception as error:
        # Without descriptor identity, unlinking by path could delete a racing
        # replacement. Close both descriptors and leave the O_EXCL-created,
        # mode-0600 placeholder (or replacement) untouched for caller inspection.
        close_error = _close_file_descriptors(descriptor, source_descriptor)
        if close_error is not None:
            raise RuntimeError(
                f"cannot close failed fallback output descriptors: {close_error}"
            ) from error
        raise RuntimeError("cannot authenticate newly created fallback output") from error

    try:
        created_path_metadata = destination.lstat()
        if (created_path_metadata.st_dev, created_path_metadata.st_ino) != (
            created_metadata.st_dev,
            created_metadata.st_ino,
        ):
            raise RuntimeError("new fallback output path changed immediately after creation")
    except Exception as error:
        close_error = _close_file_descriptors(descriptor, source_descriptor)
        _remove_created_file(destination, created_metadata)
        if close_error is not None:
            raise RuntimeError(
                f"cannot close failed fallback output descriptors: {close_error}"
            ) from error
        raise RuntimeError("cannot authenticate newly created fallback output") from error

    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        opened_source_metadata = os.fstat(source_descriptor)
        if (source_metadata.st_dev, source_metadata.st_ino) != (
            opened_source_metadata.st_dev,
            opened_source_metadata.st_ino,
        ):
            raise RuntimeError("verified snapshot inode changed before fallback copy")
        while chunk := os.read(source_descriptor, 1024 * 1024):
            copied_bytes += len(chunk)
            if copied_bytes > MAX_BINARY_BYTES:
                raise RuntimeError("verified snapshot exceeded the reviewed size limit")
            digest.update(chunk)
            _write_all(descriptor, chunk)
        os.fsync(descriptor)
        final_source_metadata = os.fstat(source_descriptor)
        _require_stable_output(opened_source_metadata, final_source_metadata)
        if copied_bytes != opened_source_metadata.st_size:
            raise RuntimeError("verified snapshot byte count changed during fallback copy")
        if digest.hexdigest() != expected_digest:
            raise RuntimeError("fallback copy source does not match its verified SHA-256")
        if os.name != "nt":
            os.fchmod(descriptor, stat.S_IRUSR | stat.S_IXUSR)
    except Exception as error:
        close_error = _close_file_descriptors(descriptor, source_descriptor)
        _remove_created_file(destination, created_metadata)
        if close_error is not None:
            raise RuntimeError(
                f"cannot close failed fallback output descriptors: {close_error}"
            ) from error
        raise
    close_error = _close_file_descriptors(descriptor, source_descriptor)
    if close_error is not None:
        _remove_created_file(destination, created_metadata)
        raise RuntimeError(
            f"cannot close fallback output descriptors: {close_error}"
        ) from close_error

    try:
        final_source_path_metadata = source.lstat()
        if (opened_source_metadata.st_dev, opened_source_metadata.st_ino) != (
            final_source_path_metadata.st_dev,
            final_source_path_metadata.st_ino,
        ):
            raise RuntimeError("verified snapshot inode changed during fallback copy")
        _require_stable_output(opened_source_metadata, final_source_path_metadata)
    except Exception:
        _remove_created_file(destination, created_metadata)
        raise
    return created_metadata


def _remove_created_file(destination: Path, created_metadata: os.stat_result) -> None:
    """Remove only the inode created by this invocation, never a replacement."""

    try:
        current_metadata = destination.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise RuntimeError(f"cannot inspect failed published output: {error}") from error
    if (current_metadata.st_dev, current_metadata.st_ino) != (
        created_metadata.st_dev,
        created_metadata.st_ino,
    ):
        return
    try:
        destination.unlink()
    except OSError as error:
        raise RuntimeError(f"cannot remove failed published output: {error}") from error


def _close_file_descriptors(*descriptors: int) -> OSError | None:
    """Attempt every close and return the first error, if any."""

    first_error: OSError | None = None
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
    return first_error


if __name__ == "__main__":
    sys.exit(main())
