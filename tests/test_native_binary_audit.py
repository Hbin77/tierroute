# SPDX-License-Identifier: Apache-2.0
"""Tests for the platform-local native import audit."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tierroute.predictors.native_ridge import MAX_BINARY_BYTES


def _load_script() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "audit_native_binary.py"
    spec = importlib.util.spec_from_file_location("audit_native_binary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script()

_MACOS_DEPENDENCIES = """/private/tmp/tierroute-native-audit/candidate:
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
"""
_MACOS_IMPORTS = """___stack_chk_fail
___stderrp
_fread
_malloc
"""
_WINDOWS_DEPENDENCIES = r"""Dump of file C:\runner\temp\candidate.exe

File Type: EXECUTABLE IMAGE

  Image has the following dependencies:

    KERNEL32.dll

  Summary

        1000 .data
        1000 .rdata
        5000 .text
"""


def _windows_imports(*symbols: str) -> str:
    symbol_rows = "\n".join(
        f"                         {index + 0x2C8:X} {symbol}"
        for index, symbol in enumerate(symbols)
    )
    return rf"""Dump of file C:\runner\temp\candidate.exe

File Type: EXECUTABLE IMAGE

  Section contains the following imports:

    KERNEL32.dll
             14001A000 Import Address Table
             14001B000 Import Name Table
                     0 time date stamp
                     0 Index of first forwarder reference

{symbol_rows}

  Summary

        1000 .data
        1000 .rdata
        5000 .text
"""


def test_macos_reports_require_only_libsystem_and_reject_spawn_or_network() -> None:
    result = AUDIT.audit_reports(
        "macos",
        dependency_report=_MACOS_DEPENDENCIES,
        import_report=_MACOS_IMPORTS,
    )

    assert result["dependencies"] == ["/usr/lib/libSystem.B.dylib"]
    assert result["release_artifact_approved"] is False
    with pytest.raises(RuntimeError, match="unexpected native dependencies"):
        AUDIT.audit_reports(
            "macos",
            dependency_report=_MACOS_DEPENDENCIES + "\t/usr/lib/libobjc.A.dylib "
            "(compatibility version 1.0.0, current version 228.0.0)\n",
            import_report=_MACOS_IMPORTS,
        )
    with pytest.raises(
        RuntimeError,
        match=r"forbidden process/network/dynamic-resolution imports.*_posix_spawn",
    ):
        AUDIT.audit_reports(
            "macos",
            dependency_report=_MACOS_DEPENDENCIES,
            import_report=_MACOS_IMPORTS + "_posix_spawn\n",
        )
    with pytest.raises(
        RuntimeError,
        match=r"forbidden process/network/dynamic-resolution imports.*_connect",
    ):
        AUDIT.audit_reports(
            "macos",
            dependency_report=_MACOS_DEPENDENCIES,
            import_report=_MACOS_IMPORTS + "_connect\n",
        )
    with pytest.raises(
        RuntimeError,
        match=r"forbidden process/network/dynamic-resolution imports.*_dlopen",
    ):
        AUDIT.audit_reports(
            "macos",
            dependency_report=_MACOS_DEPENDENCIES,
            import_report=_MACOS_IMPORTS + "_dlopen\n",
        )


def test_windows_reports_require_static_crt_dependency_boundary() -> None:
    imports = _windows_imports(
        "GetLastError",
        "HeapAlloc",
        "ExitProcess",
        "GetProcAddress",
        "LoadLibraryExW",
    )

    result = AUDIT.audit_reports(
        "windows",
        dependency_report=_WINDOWS_DEPENDENCIES,
        import_report=imports,
    )

    assert result["dependencies"] == ["KERNEL32.dll"]
    assert result["import_count"] == 5
    assert result["dynamic_resolution_imports"] == ["GetProcAddress", "LoadLibraryExW"]
    assert result["indirect_capability_absence_proven"] is False
    with pytest.raises(RuntimeError, match="unexpected native dependencies"):
        AUDIT.audit_reports(
            "windows",
            dependency_report=_WINDOWS_DEPENDENCIES.replace(
                "    KERNEL32.dll", "    KERNEL32.dll\n    VCRUNTIME140.dll"
            ),
            import_report=imports,
        )


@pytest.mark.parametrize(
    "symbol",
    [
        "CreateProcessW",
        "ShellExecuteEx",
        "WinExec",
        "WSAStartup",
        "WSAConnect",
        "WinHttpOpen",
        "connect",
        "__imp__popen",
        "NtCreateProcessEx",
        "NtCreateUserProcess",
        "RtlCreateUserProcess",
        "ShellExecuteExW",
        "_wspawnl",
    ],
)
def test_windows_import_scan_rejects_process_and_network_capabilities(symbol: str) -> None:
    imports = _windows_imports("GetLastError", symbol)

    with pytest.raises(RuntimeError, match="forbidden process/network/dynamic-resolution"):
        AUDIT.audit_reports(
            "windows",
            dependency_report=_WINDOWS_DEPENDENCIES,
            import_report=imports,
        )


def test_import_audit_fails_closed_on_empty_or_unrecognized_reports() -> None:
    with pytest.raises(RuntimeError, match="empty or contains undecodable"):
        AUDIT.audit_reports("macos", dependency_report="", import_report=_MACOS_IMPORTS)
    with pytest.raises(RuntimeError, match="unrecognized nm"):
        AUDIT.audit_reports(
            "macos",
            dependency_report=_MACOS_DEPENDENCIES,
            import_report="no parseable import rows\n",
        )
    with pytest.raises(RuntimeError, match=r"exactly one.*marker"):
        AUDIT.audit_reports(
            "windows",
            dependency_report=_WINDOWS_DEPENDENCIES,
            import_report="no parseable import rows",
        )


def test_windows_import_parser_rejects_partial_or_ordinal_only_rows() -> None:
    valid = _windows_imports("GetLastError")
    partial = valid.replace(
        "                         2C8 GetLastError",
        "                         2C8 GetLastError unexpected-token",
    )
    ordinal = valid.replace(
        "                         2C8 GetLastError",
        "                         2C8 123",
    )

    with pytest.raises(RuntimeError, match="unrecognized dumpbin import row"):
        AUDIT.audit_reports(
            "windows",
            dependency_report=_WINDOWS_DEPENDENCIES,
            import_report=partial,
        )
    with pytest.raises(RuntimeError, match="ordinal-only"):
        AUDIT.audit_reports(
            "windows",
            dependency_report=_WINDOWS_DEPENDENCIES,
            import_report=ordinal,
        )


def test_windows_dependency_and_import_sections_must_agree() -> None:
    imports = _windows_imports("GetLastError").replace("KERNEL32.dll", "USER32.dll")

    with pytest.raises(RuntimeError, match="reports disagree"):
        AUDIT.audit_reports(
            "windows",
            dependency_report=_WINDOWS_DEPENDENCIES,
            import_report=imports,
        )


def test_snapshot_is_owner_only_and_bound_to_build_manifest(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    destination = tmp_path / "snapshot"
    payload = b"reviewed-native-candidate"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    assert (
        AUDIT._snapshot_verified_file(
            source,
            destination,
            expected_sha256=digest,
            maximum_bytes=1024,
            label="native binary",
        )
        == digest
    )
    assert destination.read_bytes() == payload
    if os.name != "nt":
        assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR


def test_cross_interface_stability_can_omit_only_nonportable_change_time() -> None:
    first = SimpleNamespace(st_size=23, st_mtime_ns=101, st_ctime_ns=202)
    changed_creation_time = SimpleNamespace(
        st_size=23,
        st_mtime_ns=101,
        st_ctime_ns=303,
    )

    AUDIT._require_stable_file(
        first,
        changed_creation_time,
        label="native binary",
        compare_change_time=False,
    )
    with pytest.raises(RuntimeError, match="metadata changed"):
        AUDIT._require_stable_file(
            first,
            changed_creation_time,
            label="native binary",
        )


@pytest.mark.parametrize("changed_field", ("st_size", "st_mtime_ns"))
def test_cross_interface_stability_keeps_content_metadata_guards(
    changed_field: str,
) -> None:
    first_values = {"st_size": 23, "st_mtime_ns": 101, "st_ctime_ns": 202}
    second_values = first_values | {changed_field: first_values[changed_field] + 1}

    with pytest.raises(RuntimeError, match="metadata changed"):
        AUDIT._require_stable_file(
            SimpleNamespace(**first_values),
            SimpleNamespace(**second_values),
            label="native binary",
            compare_change_time=False,
        )


def test_audit_and_adapter_share_binary_size_contract() -> None:
    assert AUDIT._MAX_BINARY_BYTES == MAX_BINARY_BYTES == 16 * 1024 * 1024


def test_audit_rejects_binary_above_shared_size_contract(tmp_path: Path) -> None:
    source = tmp_path / "oversized-candidate"
    destination = tmp_path / "snapshot"
    with source.open("wb") as stream:
        stream.seek(MAX_BINARY_BYTES)
        stream.write(b"x")

    with pytest.raises(RuntimeError, match="must not exceed"):
        AUDIT._snapshot_verified_file(
            source,
            destination,
            expected_sha256="0" * 64,
            maximum_bytes=AUDIT._MAX_BINARY_BYTES,
            label="native binary",
        )
    assert not destination.exists()


@pytest.mark.parametrize("path", ("//server/share/candidate", r"\\server\share\candidate"))
def test_audit_rejects_unc_and_device_style_inputs_on_every_host(path: str) -> None:
    with pytest.raises(ValueError, match="UNC or device-style"):
        AUDIT._safe_binary_path(path)
    with pytest.raises(ValueError, match="UNC or device-style"):
        AUDIT._safe_source_path(path)


def test_snapshot_rejects_build_manifest_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    destination = tmp_path / "snapshot"
    source.write_bytes(b"unexpected-candidate")

    with pytest.raises(RuntimeError, match="does not match the build manifest"):
        AUDIT._snapshot_verified_file(
            source,
            destination,
            expected_sha256="0" * 64,
            maximum_bytes=1024,
            label="native binary",
        )
    assert not destination.exists()


@pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "Windows denies renaming the open source handle; this original-path "
        "replacement race requires POSIX rename semantics"
    ),
)
def test_snapshot_rejects_original_path_replacement_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "candidate"
    destination = tmp_path / "snapshot"
    displaced = tmp_path / "displaced"
    payload = b"x" * (AUDIT._COPY_CHUNK_BYTES + 1)
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    real_write_all = AUDIT._write_all
    replaced = False

    def replace_path_then_write(descriptor: int, chunk: bytes) -> None:
        nonlocal replaced
        if not replaced:
            source.replace(displaced)
            source.write_bytes(payload)
            replaced = True
        real_write_all(descriptor, chunk)

    monkeypatch.setattr(AUDIT, "_write_all", replace_path_then_write)
    with pytest.raises(RuntimeError, match=r"changed (?:during snapshot|while it was snapshotted)"):
        AUDIT._snapshot_verified_file(
            source,
            destination,
            expected_sha256=digest,
            maximum_bytes=len(payload),
            label="native binary",
        )
    assert replaced
    assert not destination.exists()


def test_audit_tools_receive_private_snapshot_not_replaceable_originals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "candidate"
    source = tmp_path / "candidate.c"
    binary_payload = b"platform-binary"
    source_payload = b"int main(void) { return 0; }\n"
    binary.write_bytes(binary_payload)
    source.write_bytes(source_payload)
    binary_digest = hashlib.sha256(binary_payload).hexdigest()
    source_digest = hashlib.sha256(source_payload).hexdigest()

    def collect_reports(platform: str, snapshot: Path) -> tuple[str, str]:
        assert platform == "macos"
        assert snapshot != binary
        assert snapshot.parent != binary.parent
        assert snapshot.read_bytes() == binary_payload
        binary.write_bytes(b"replaced-after-snapshot")
        source.write_text("int changed(void) { return 1; }\n", encoding="utf-8")
        return _MACOS_DEPENDENCIES, _MACOS_IMPORTS

    monkeypatch.setattr(AUDIT, "_collect_reports", collect_reports)
    reports, result = AUDIT.audit_candidate(
        "macos",
        binary=binary,
        source=source,
        expected_binary_sha256=binary_digest,
        expected_source_sha256=source_digest,
    )

    assert reports == (_MACOS_DEPENDENCIES, _MACOS_IMPORTS)
    assert result["binary_sha256"] == binary_digest
    assert result["source_sha256"] == source_digest


def test_audit_cli_requires_both_build_manifest_hashes() -> None:
    with pytest.raises(SystemExit) as captured:
        AUDIT.main(
            [
                "--platform",
                "macos",
                "--binary",
                "/absolute/binary",
                "--source",
                "/absolute/source",
            ]
        )

    assert captured.value.code == 2


def test_platform_tool_timeout_becomes_controlled_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AUDIT.shutil, "which", lambda _name: sys.executable)

    def time_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("nm", AUDIT._COMMAND_TIMEOUT_SECONDS)

    monkeypatch.setattr(AUDIT.subprocess, "run", time_out)
    with pytest.raises(RuntimeError, match="timed out after 30 seconds"):
        AUDIT._run_tool("nm", "-u", "-j", "/absolute/candidate")


@pytest.mark.parametrize(
    "identifier",
    [
        "CreateProcessW",
        "NtCreateUserProcess",
        "RtlCreateUserProcess",
        "_wspawnl",
        "connect",
        "dlopen",
        "GetProcAddress",
        "LoadLibraryExW",
    ],
)
def test_reviewed_c_source_rejects_direct_capability_identifiers(
    tmp_path: Path, identifier: str
) -> None:
    source = tmp_path / "candidate.c"
    source.write_text(f"int main(void) {{ return {identifier} == 0; }}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="direct process/network/dynamic-resolution"):
        AUDIT.audit_source(source)


def test_reviewed_c_source_returns_its_hash_for_benign_code(tmp_path: Path) -> None:
    source = tmp_path / "candidate.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")

    assert AUDIT.audit_source(source) == AUDIT.hashlib.sha256(source.read_bytes()).hexdigest()
