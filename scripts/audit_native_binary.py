# SPDX-License-Identifier: Apache-2.0
"""Audit linkage and imported symbols for a project-owned native sidecar."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

_COMMAND_TIMEOUT_SECONDS = 30
_MAX_REPORT_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_BINARY_BYTES = 16 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_MACOS_DEPENDENCIES = frozenset({"/usr/lib/libSystem.B.dylib"})
_WINDOWS_DEPENDENCIES = frozenset({"kernel32.dll"})

# Keep this list about capabilities that violate the sidecar boundary. Normal
# process termination (for example ExitProcess in a static MSVC runtime) is not
# process creation and is intentionally allowed.
_FORBIDDEN_EXACT_SYMBOLS = frozenset(
    {
        "accept",
        "accept4",
        "bind",
        "connect",
        "execl",
        "execle",
        "execlp",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "freeaddrinfo",
        "getaddrinfo",
        "gethostbyaddr",
        "gethostbyname",
        "getnameinfo",
        "listen",
        "popen",
        "recv",
        "recvfrom",
        "recvmsg",
        "send",
        "sendmsg",
        "sendto",
        "shellexecutea",
        "shellexecuteex",
        "shellexecutew",
        "shutdown",
        "socket",
        "socketpair",
        "system",
        "vfork",
        "winexec",
        "wpopen",
        "wsystem",
        "wsastartup",
        "wsasocketa",
        "wsasocketw",
    }
)
_FORBIDDEN_SYMBOL_PREFIXES = (
    "createprocess",
    "dnsquery",
    "freeaddrinfo",
    "getaddrinfo",
    "gethostby",
    "getnameinfo",
    "inetntop",
    "inetpton",
    "internetopen",
    "internetconnect",
    "ntcreateprocess",
    "ntcreateuserprocess",
    "posix_spawn",
    "rtlcreateprocess",
    "rtlcreateuserprocess",
    "shellexecute",
    "spawn",
    "urldownloadto",
    "wexec",
    "winhttp",
    "wspawn",
    "wsaconnect",
    "zwcreateprocess",
    "zwcreateuserprocess",
)
_DYNAMIC_RESOLUTION_EXACT_SYMBOLS = frozenset(
    {
        "dlfunc",
        "dlinfo",
        "dlmopen",
        "dlopen",
        "dlsym",
        "getprocaddress",
        "ldrgetprocedureaddress",
        "ldrloaddll",
        "loadlibrary",
        "loadlibrarya",
        "loadlibraryexa",
        "loadlibraryexw",
        "loadlibraryw",
    }
)
_MACOS_DEPENDENCY_ROW = re.compile(
    r"^\s+(.+?) \(compatibility version [^(),]+, current version [^()]+\)$"
)
_MACOS_SYMBOL_ROW = re.compile(r"^[_A-Za-z?@$][._A-Za-z0-9?@$-]*$")
_WINDOWS_DLL_ROW = re.compile(r"^([A-Za-z0-9_.-]+\.dll)$", re.IGNORECASE)
_WINDOWS_IMPORT_ROW = re.compile(r"^[0-9A-Fa-f]+\s+([?@_A-Za-z][?@_A-Za-z0-9$]*)$")
_WINDOWS_ORDINAL_ROW = re.compile(
    r"^[0-9A-Fa-f]+\s+(?:ordinal\s+)?[0-9A-Fa-f]+$",
    re.IGNORECASE,
)
_WINDOWS_SUMMARY_ROW = re.compile(r"^[0-9A-Fa-f]+\s+\S+$")
_WINDOWS_IMPORT_METADATA = (
    re.compile(r"^[0-9A-Fa-f]+\s+Import Address Table$"),
    re.compile(r"^[0-9A-Fa-f]+\s+Import Name Table$"),
    re.compile(r"^[0-9A-Fa-f]+\s+time date stamp$"),
    re.compile(r"^[0-9A-Fa-f]+\s+Index of first forwarder reference$"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("macos", "windows"), required=True)
    parser.add_argument("--binary", required=True, help="absolute path to the local candidate")
    parser.add_argument("--source", required=True, help="absolute path to the reviewed C source")
    parser.add_argument(
        "--expected-binary-sha256",
        required=True,
        help="lowercase SHA-256 emitted for the binary by the build manifest",
    )
    parser.add_argument(
        "--expected-source-sha256",
        required=True,
        help="lowercase SHA-256 emitted for the source by the build manifest",
    )
    arguments = parser.parse_args(argv)
    try:
        binary = _safe_binary_path(arguments.binary)
        source = _safe_source_path(arguments.source)
        expected_binary_sha256 = _validate_sha256(
            arguments.expected_binary_sha256,
            option="--expected-binary-sha256",
        )
        expected_source_sha256 = _validate_sha256(
            arguments.expected_source_sha256,
            option="--expected-source-sha256",
        )
        _require_host(arguments.platform)
        reports, result = audit_candidate(
            arguments.platform,
            binary=binary,
            source=source,
            expected_binary_sha256=expected_binary_sha256,
            expected_source_sha256=expected_source_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    dependency_tool, import_tool = (
        ("otool -L", "nm -u -j")
        if arguments.platform == "macos"
        else ("dumpbin /DEPENDENTS", "dumpbin /IMPORTS")
    )
    print(f"=== {dependency_tool} ===")
    print(reports[0].rstrip())
    print(f"=== {import_tool} ===")
    print(reports[1].rstrip())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def audit_candidate(
    platform: str,
    *,
    binary: Path,
    source: Path,
    expected_binary_sha256: str,
    expected_source_sha256: str,
) -> tuple[tuple[str, str], dict[str, object]]:
    """Snapshot one build-manifest-bound candidate and audit only those snapshots."""

    expected_binary = _validate_sha256(
        expected_binary_sha256,
        option="expected_binary_sha256",
    )
    expected_source = _validate_sha256(
        expected_source_sha256,
        option="expected_source_sha256",
    )
    with tempfile.TemporaryDirectory(prefix="tierroute-native-audit-") as name:
        directory = Path(name)
        _make_private_directory(directory)
        binary_snapshot = directory / ("candidate.exe" if platform == "windows" else "candidate")
        source_snapshot = directory / "candidate.c"
        binary_digest = _snapshot_verified_file(
            binary,
            binary_snapshot,
            expected_sha256=expected_binary,
            maximum_bytes=_MAX_BINARY_BYTES,
            label="native binary",
        )
        source_digest = _snapshot_verified_file(
            source,
            source_snapshot,
            expected_sha256=expected_source,
            maximum_bytes=_MAX_SOURCE_BYTES,
            label="native source",
        )
        audited_source_digest = audit_source(source_snapshot)
        if not hmac.compare_digest(audited_source_digest, source_digest):
            raise RuntimeError("private native source snapshot hash changed during audit")
        reports = _collect_reports(platform, binary_snapshot)
        result = audit_reports(
            platform,
            dependency_report=reports[0],
            import_report=reports[1],
        )
    result["binary_sha256"] = binary_digest
    result["source_direct_capability_identifier_count"] = 0
    result["source_sha256"] = source_digest
    return reports, result


def audit_reports(
    platform: str,
    *,
    dependency_report: str,
    import_report: str,
) -> dict[str, object]:
    """Validate captured system-tool reports and return compact audit evidence."""

    if not isinstance(dependency_report, str) or not isinstance(import_report, str):
        raise TypeError("native audit reports must be text")
    import_dependencies: frozenset[str] = frozenset()
    normalized_import_dependencies: frozenset[str] | None = None
    if platform == "macos":
        dependencies = _parse_macos_dependencies(dependency_report)
        expected_dependencies = _MACOS_DEPENDENCIES
        symbols = _parse_macos_imports(import_report)
    elif platform == "windows":
        dependencies = _parse_windows_dependencies(dependency_report)
        expected_dependencies = _WINDOWS_DEPENDENCIES
        import_dependencies, symbols = _parse_windows_imports(import_report)
        normalized_import_dependencies = frozenset(
            value.casefold() for value in import_dependencies
        )
    else:
        raise ValueError(f"unsupported audit platform: {platform!r}")

    normalized_dependencies = frozenset(
        value if platform == "macos" else value.casefold() for value in dependencies
    )
    if normalized_dependencies != expected_dependencies:
        raise RuntimeError(
            "unexpected native dependencies: "
            f"expected {sorted(expected_dependencies)!r}, got {sorted(dependencies)!r}"
        )
    if (
        normalized_import_dependencies is not None
        and normalized_import_dependencies != normalized_dependencies
    ):
        raise RuntimeError(
            "Windows dependency/import reports disagree: "
            f"dependencies={sorted(dependencies)!r}, "
            f"import sections={sorted(import_dependencies)!r}"
        )
    if not symbols:
        raise RuntimeError("native import report did not contain any imported symbols")
    forbidden = _find_forbidden_symbols(symbols)
    dynamic_resolution = _find_dynamic_resolution_symbols(symbols)
    if platform == "macos":
        forbidden = forbidden | dynamic_resolution
    if forbidden:
        raise RuntimeError(
            f"forbidden process/network/dynamic-resolution imports: {sorted(forbidden)!r}"
        )
    return {
        "claim_scope": "source-portability-candidate-only",
        "dependency_count": len(dependencies),
        "dependencies": sorted(dependencies),
        "dynamic_resolution_import_count": len(dynamic_resolution),
        "dynamic_resolution_imports": sorted(dynamic_resolution),
        "forbidden_import_count": 0,
        "indirect_capability_absence_proven": False,
        "import_count": len(symbols),
        "platform": platform,
        "release_artifact_approved": False,
    }


def audit_source(path: Path) -> str:
    """Reject direct forbidden capability identifiers in the reviewed C source."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"native source snapshot cannot be inspected: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("native source snapshot must be a regular non-symlink file")
    if metadata.st_size > _MAX_SOURCE_BYTES:
        raise RuntimeError(f"native source exceeded {_MAX_SOURCE_BYTES} bytes")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_SOURCE_BYTES + 1)
    if len(payload) > _MAX_SOURCE_BYTES:
        raise RuntimeError(f"native source exceeded {_MAX_SOURCE_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("native source must be valid UTF-8") from error
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    forbidden = _find_forbidden_symbols(identifiers) | _find_dynamic_resolution_symbols(identifiers)
    if forbidden:
        raise RuntimeError(
            "reviewed C source contains direct process/network/dynamic-resolution identifiers: "
            f"{sorted(forbidden)!r}"
        )
    return hashlib.sha256(payload).hexdigest()


def _safe_binary_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError("--binary must be a non-empty path without NUL bytes")
    if raw_path.startswith(("//", "\\\\")):
        raise ValueError("--binary must not use a UNC or device-style path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError("--binary must be absolute")
    path = Path(os.path.abspath(path))
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"--binary cannot be inspected: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("--binary must be a regular non-symlink file")
    if metadata.st_size == 0:
        raise ValueError("--binary must not be empty")
    return path


def _safe_source_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError("--source must be a non-empty path without NUL bytes")
    if raw_path.startswith(("//", "\\\\")):
        raise ValueError("--source must not use a UNC or device-style path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError("--source must be absolute")
    path = Path(os.path.abspath(path))
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"--source cannot be inspected: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("--source must be a regular non-symlink file")
    if metadata.st_size == 0:
        raise ValueError("--source must not be empty")
    if metadata.st_size > _MAX_SOURCE_BYTES:
        raise ValueError(f"--source must not exceed {_MAX_SOURCE_BYTES} bytes")
    return path


def _validate_sha256(value: object, *, option: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{option} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{option} must be exactly 64 lowercase hexadecimal characters")
    return value


def _make_private_directory(path: Path) -> None:
    os.chmod(path, stat.S_IRWXU)
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("native audit temporary path is not a directory")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != stat.S_IRWXU:
        raise RuntimeError("native audit temporary directory is not owner-only")


def _snapshot_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    label: str,
) -> str:
    """Copy one stable regular-file inode and bind it to the build manifest hash."""

    expected = _validate_sha256(expected_sha256, option=f"expected {label} SHA-256")
    try:
        path_metadata = source.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} cannot be inspected: {error}") from error
    _require_regular_bounded_file(path_metadata, maximum_bytes=maximum_bytes, label=label)

    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        raise RuntimeError(f"cannot open {label} safely: {error}") from error
    try:
        destination_descriptor = os.open(
            destination,
            destination_flags,
            stat.S_IRUSR | stat.S_IWUSR,
        )
    except OSError as error:
        os.close(source_descriptor)
        raise RuntimeError(f"cannot create private {label} snapshot: {error}") from error

    digest = hashlib.sha256()
    total = 0
    opened_metadata: os.stat_result | None = None
    try:
        opened_metadata = os.fstat(source_descriptor)
        _require_same_file(path_metadata, opened_metadata, label=label)
        _require_regular_bounded_file(
            opened_metadata,
            maximum_bytes=maximum_bytes,
            label=label,
        )
        while chunk := os.read(source_descriptor, _COPY_CHUNK_BYTES):
            total += len(chunk)
            if total > maximum_bytes:
                raise RuntimeError(f"{label} exceeded {maximum_bytes} bytes while copied")
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
        os.fsync(destination_descriptor)
        if total == 0:
            raise RuntimeError(f"{label} must not be empty")
        final_descriptor_metadata = os.fstat(source_descriptor)
        _require_same_file(opened_metadata, final_descriptor_metadata, label=label)
        _require_stable_file(opened_metadata, final_descriptor_metadata, label=label)
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)

    assert opened_metadata is not None
    try:
        final_path_metadata = source.lstat()
        _require_same_file(opened_metadata, final_path_metadata, label=label)
        _require_stable_file(
            opened_metadata,
            final_path_metadata,
            label=label,
            # Python 3.12 deprecated st_ctime_ns on Windows, where it is the
            # creation time and path-based stat can disagree with descriptor
            # stat for an unchanged file. Keep the same-interface fstat check
            # above strict, and omit only this non-portable cross-interface field.
            compare_change_time=os.name != "nt",
        )
    except (OSError, RuntimeError):
        try:
            destination.unlink()
        except OSError:
            pass
        raise RuntimeError(f"{label} changed while it was snapshotted") from None

    actual_digest = digest.hexdigest()
    if not hmac.compare_digest(actual_digest, expected):
        try:
            destination.unlink()
        except OSError:
            pass
        raise RuntimeError(f"{label} SHA-256 does not match the build manifest")
    if os.name != "nt":
        mode = stat.S_IMODE(destination.stat().st_mode)
        if mode != (stat.S_IRUSR | stat.S_IWUSR):
            try:
                destination.unlink()
            except OSError:
                pass
            raise RuntimeError(f"private {label} snapshot is not owner-only")
    return actual_digest


def _require_regular_bounded_file(
    metadata: os.stat_result,
    *,
    maximum_bytes: int,
    label: str,
) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    if metadata.st_size == 0:
        raise RuntimeError(f"{label} must not be empty")
    if metadata.st_size > maximum_bytes:
        raise RuntimeError(f"{label} must not exceed {maximum_bytes} bytes")


def _require_same_file(
    first: os.stat_result,
    second: os.stat_result,
    *,
    label: str,
) -> None:
    if (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino):
        raise RuntimeError(f"{label} inode changed during snapshot")
    if not stat.S_ISREG(second.st_mode):
        raise RuntimeError(f"{label} is no longer a regular file")


def _require_stable_file(
    first: os.stat_result,
    second: os.stat_result,
    *,
    label: str,
    compare_change_time: bool = True,
) -> None:
    fields = ("st_size", "st_mtime_ns")
    if compare_change_time:
        fields += ("st_ctime_ns",)
    for field in fields:
        first_value = getattr(first, field, None)
        second_value = getattr(second, field, None)
        if first_value != second_value:
            raise RuntimeError(f"{label} metadata changed during snapshot")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while creating native audit snapshot")
        offset += written


def _require_host(platform: str) -> None:
    if platform == "macos" and sys.platform != "darwin":
        raise RuntimeError("the macOS audit must run on a macOS host")
    if platform == "windows" and os.name != "nt":
        raise RuntimeError("the Windows audit must run on a Windows host")


def _collect_reports(platform: str, binary: Path) -> tuple[str, str]:
    if platform == "macos":
        return (
            _run_tool("otool", "-L", str(binary)),
            _run_tool("nm", "-u", "-j", str(binary)),
        )
    return (
        _run_tool("dumpbin", "/NOLOGO", "/DEPENDENTS", str(binary)),
        _run_tool("dumpbin", "/NOLOGO", "/IMPORTS", str(binary)),
    )


def _run_tool(name: str, *arguments: str) -> str:
    discovered = shutil.which(name)
    if discovered is None:
        raise RuntimeError(f"required platform audit tool is unavailable: {name}")
    tool = os.path.abspath(discovered)
    try:
        completed = subprocess.run(
            [tool, *arguments],
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{name} timed out after {_COMMAND_TIMEOUT_SECONDS} seconds") from error
    if completed.returncode != 0:
        detail = completed.stderr[-16_384:].strip()
        raise RuntimeError(f"{name} failed with status {completed.returncode}: {detail}")
    if len(completed.stdout.encode("utf-8")) > _MAX_REPORT_BYTES:
        raise RuntimeError(f"{name} report exceeded {_MAX_REPORT_BYTES} bytes")
    return completed.stdout


def _parse_macos_dependencies(report: str) -> frozenset[str]:
    _reject_corrupt_tool_text(report, tool="otool -L")
    lines = report.splitlines()
    if not lines or not lines[0] or not lines[0].endswith(":"):
        raise RuntimeError("otool -L report is missing its binary header")
    dependencies: set[str] = set()
    for line in lines[1:]:
        if not line.strip():
            continue
        matched = _MACOS_DEPENDENCY_ROW.fullmatch(line)
        if matched is None:
            raise RuntimeError(f"unrecognized otool -L row: {line!r}")
        dependency = matched.group(1)
        if dependency in dependencies:
            raise RuntimeError(f"duplicate otool -L dependency: {dependency!r}")
        dependencies.add(dependency)
    if not dependencies:
        raise RuntimeError("otool -L report did not contain any dependencies")
    return frozenset(dependencies)


def _parse_windows_dependencies(report: str) -> frozenset[str]:
    body = _extract_dumpbin_body(
        report,
        marker="Image has the following dependencies:",
        tool="dumpbin /DEPENDENTS",
    )
    dependencies: set[str] = set()
    for line in body:
        stripped = line.strip()
        if not stripped:
            continue
        matched = _WINDOWS_DLL_ROW.fullmatch(stripped)
        if matched is None:
            raise RuntimeError(f"unrecognized dumpbin dependency row: {line!r}")
        dependency = matched.group(1)
        normalized = dependency.casefold()
        if any(item.casefold() == normalized for item in dependencies):
            raise RuntimeError(f"duplicate dumpbin dependency: {dependency!r}")
        dependencies.add(dependency)
    if not dependencies:
        raise RuntimeError("dumpbin /DEPENDENTS did not contain any dependencies")
    return frozenset(dependencies)


def _parse_macos_imports(report: str) -> frozenset[str]:
    _reject_corrupt_tool_text(report, tool="nm -u -j")
    symbols: set[str] = set()
    for line in report.splitlines():
        if not line:
            continue
        if line != line.strip() or _MACOS_SYMBOL_ROW.fullmatch(line) is None:
            raise RuntimeError(f"unrecognized nm -u -j symbol row: {line!r}")
        if line in symbols:
            raise RuntimeError(f"duplicate nm -u -j symbol row: {line!r}")
        symbols.add(line)
    if not symbols:
        raise RuntimeError("nm -u -j report did not contain any imported symbols")
    return frozenset(symbols)


def _parse_windows_imports(report: str) -> tuple[frozenset[str], frozenset[str]]:
    body = _extract_dumpbin_body(
        report,
        marker="Section contains the following imports:",
        tool="dumpbin /IMPORTS",
    )
    dependencies: set[str] = set()
    symbols: set[str] = set()
    index = 0
    while index < len(body):
        while index < len(body) and not body[index].strip():
            index += 1
        if index == len(body):
            break
        dependency_row = body[index].strip()
        dependency_match = _WINDOWS_DLL_ROW.fullmatch(dependency_row)
        if dependency_match is None:
            raise RuntimeError(f"expected a dumpbin import DLL row, got {body[index]!r}")
        dependency = dependency_match.group(1)
        normalized_dependency = dependency.casefold()
        if any(item.casefold() == normalized_dependency for item in dependencies):
            raise RuntimeError(f"duplicate dumpbin import section: {dependency!r}")
        dependencies.add(dependency)
        index += 1

        for metadata_pattern in _WINDOWS_IMPORT_METADATA:
            if index >= len(body):
                raise RuntimeError(f"truncated dumpbin import metadata for {dependency!r}")
            metadata_row = body[index].strip()
            if metadata_pattern.fullmatch(metadata_row) is None:
                raise RuntimeError(
                    f"unrecognized dumpbin import metadata for {dependency!r}: {body[index]!r}"
                )
            index += 1

        imported_for_dependency = 0
        while index < len(body):
            stripped = body[index].strip()
            if not stripped:
                index += 1
                continue
            if _WINDOWS_DLL_ROW.fullmatch(stripped) is not None:
                break
            if _WINDOWS_ORDINAL_ROW.fullmatch(stripped) is not None:
                raise RuntimeError(
                    f"ordinal-only dumpbin import is not auditable by name: {body[index]!r}"
                )
            import_match = _WINDOWS_IMPORT_ROW.fullmatch(stripped)
            if import_match is None:
                raise RuntimeError(f"unrecognized dumpbin import row: {body[index]!r}")
            symbol = import_match.group(1)
            if symbol in symbols:
                raise RuntimeError(f"duplicate dumpbin import symbol: {symbol!r}")
            symbols.add(symbol)
            imported_for_dependency += 1
            index += 1
        if imported_for_dependency == 0:
            raise RuntimeError(f"dumpbin import section for {dependency!r} has no named symbols")
    if not dependencies:
        raise RuntimeError("dumpbin /IMPORTS did not contain any DLL sections")
    if not symbols:
        raise RuntimeError("dumpbin /IMPORTS did not contain any imported symbols")
    return frozenset(dependencies), frozenset(symbols)


def _extract_dumpbin_body(report: str, *, marker: str, tool: str) -> tuple[str, ...]:
    _reject_corrupt_tool_text(report, tool=tool)
    lines = report.splitlines()
    marker_indices = [index for index, line in enumerate(lines) if line.strip() == marker]
    if len(marker_indices) != 1:
        raise RuntimeError(f"{tool} report must contain exactly one {marker!r} marker")
    marker_index = marker_indices[0]
    summary_indices = [
        index
        for index, line in enumerate(lines)
        if index > marker_index and line.strip() == "Summary"
    ]
    if len(summary_indices) != 1:
        raise RuntimeError(f"{tool} report must contain exactly one trailing Summary section")
    summary_index = summary_indices[0]

    preamble = [line.strip() for line in lines[:marker_index] if line.strip()]
    dump_rows = [line for line in preamble if line.startswith("Dump of file ")]
    file_type_rows = [line for line in preamble if line == "File Type: EXECUTABLE IMAGE"]
    if len(dump_rows) != 1 or len(file_type_rows) != 1:
        raise RuntimeError(f"{tool} report is missing its executable preamble")
    allowed_preamble = {dump_rows[0], "File Type: EXECUTABLE IMAGE"}
    unexpected_preamble = [line for line in preamble if line not in allowed_preamble]
    if unexpected_preamble:
        raise RuntimeError(f"unrecognized {tool} preamble rows: {unexpected_preamble!r}")
    if preamble.index(dump_rows[0]) >= preamble.index("File Type: EXECUTABLE IMAGE"):
        raise RuntimeError(f"{tool} executable preamble is out of order")

    for line in lines[summary_index + 1 :]:
        stripped = line.strip()
        if stripped and _WINDOWS_SUMMARY_ROW.fullmatch(stripped) is None:
            raise RuntimeError(f"unrecognized {tool} Summary row: {line!r}")
    return tuple(lines[marker_index + 1 : summary_index])


def _reject_corrupt_tool_text(report: str, *, tool: str) -> None:
    if not isinstance(report, str):
        raise TypeError(f"{tool} report must be text")
    if not report or "\x00" in report or "\ufffd" in report:
        raise RuntimeError(f"{tool} report is empty or contains undecodable bytes")


def _find_forbidden_symbols(symbols: Iterable[str]) -> frozenset[str]:
    forbidden = set()
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        if normalized in _FORBIDDEN_EXACT_SYMBOLS or normalized.startswith(
            _FORBIDDEN_SYMBOL_PREFIXES
        ):
            forbidden.add(symbol)
    return frozenset(forbidden)


def _find_dynamic_resolution_symbols(symbols: Iterable[str]) -> frozenset[str]:
    return frozenset(
        symbol
        for symbol in symbols
        if _normalize_symbol(symbol) in _DYNAMIC_RESOLUTION_EXACT_SYMBOLS
    )


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.casefold()
    if normalized.startswith("__imp_"):
        normalized = normalized[6:]
    normalized = normalized.lstrip("_")
    return normalized.split("@", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
