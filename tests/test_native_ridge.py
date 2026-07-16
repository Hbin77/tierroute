# SPDX-License-Identifier: Apache-2.0
"""Protocol, process-boundary, and numerical tests for the C11 ridge adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import runpy
import shutil
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import pytest

import tierroute.predictors.native_ridge as native_ridge_module
from tierroute.adapters import load_evaluation_dataset
from tierroute.predictors._ridge import RidgeSolution, solve_centered_ridge
from tierroute.predictors.native_ridge import (
    MAX_BINARY_BYTES,
    MAX_STDERR_BYTES,
    NATIVE_C11_RIDGE_SOLVER_ID,
    NativeRidgeAdapter,
    NativeRidgeError,
    NativeRidgeExecutionError,
    NativeRidgeIntegrityError,
    NativeRidgeProtocolError,
    NativeRidgeStatusError,
)
from tierroute.predictors.training import BilinearTrainingConfig, fit_calibrated_bilinear

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_BUILD_SCRIPT = _REPOSITORY_ROOT / "scripts" / "build_native_ridge.py"
_NATIVE_SOURCE = _REPOSITORY_ROOT / "native" / "tierroute_ridge.c"
_FEATURES = ((1.0, 2.0), (3.0, 4.0))
_TARGETS = ((5.0, 7.0), (6.0, 8.0))
_RAW_REQUEST = struct.Struct("<8sII32sQQQd")
_RAW_RESPONSE = struct.Struct("<8sII32sQQ")
_RAW_REQUEST_ID = bytes(range(32))
_SCRIPT_FAKE_ONLY = pytest.mark.skipif(
    os.name == "nt",
    reason="shebang fake executables are POSIX-only; compiled C corpus still runs",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake(tmp_path: Path, action: str, *, name: str = "fake-ridge") -> Path:
    program = f"""#!{sys.executable}
import os
import struct
import sys
import time

REQUEST = struct.Struct("<8sII32sQQQd")
RESPONSE = struct.Struct("<8sII32sQQ")
request = sys.stdin.buffer.read()
if len(request) < REQUEST.size:
    raise SystemExit(91)
magic, version, flags, request_id, n, d, m, ridge = REQUEST.unpack_from(request)

def response(
    *,
    response_magic=b"TRRRES01",
    response_version=1,
    status=0,
    response_id=request_id,
    response_d=d,
    response_m=m,
    payload=b"",
):
    sys.stdout.buffer.write(
        RESPONSE.pack(
            response_magic,
            response_version,
            status,
            response_id,
            response_d,
            response_m,
        ) + payload
    )

{action}
"""
    path = tmp_path / name
    path.write_text(program, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return path


def _adapter(path: Path, *, timeout: float = 2.0) -> NativeRidgeAdapter:
    return NativeRidgeAdapter(path, _sha256(path), timeout_seconds=timeout)


def _valid_payload() -> str:
    return "struct.pack('<6d', 1.5, -2.5, 0.1, 0.2, -0.3, -0.4)"


class _FailingOSProxy:
    """Delegate to ``os`` except for one deterministic adapter-owned call."""

    def __init__(
        self,
        operation: str,
        *,
        fail_on_call: int,
        failure: BaseException,
    ) -> None:
        self._operation = operation
        self._fail_on_call = fail_on_call
        self._failure = failure
        self.calls = 0

    def __getattr__(self, name: str) -> object:
        value = getattr(os, name)
        if name != self._operation:
            return value

        def injected_failure(*args: object, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == self._fail_on_call:
                raise self._failure
            return value(*args, **kwargs)

        return injected_failure


@_SCRIPT_FAKE_ONLY
def test_fake_sidecar_receives_canonical_private_request_and_restricted_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TIERROUTE_MUST_NOT_LEAK", "secret")
    action = f"""
assert magic == b"TRRIDG01"
assert version == 1 and flags == 0
assert len(request_id) == 32
assert (n, d, m, ridge) == (2, 2, 2, 0.5)
assert len(request) == REQUEST.size + 8 * (n * d + n * m)
assert struct.unpack_from("<8d", request, REQUEST.size) == (
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
)
assert "TIERROUTE_MUST_NOT_LEAK" not in os.environ
assert os.path.realpath(os.getcwd()) == os.path.realpath(os.environ["HOME"])
assert os.path.realpath(os.getcwd()) == os.path.realpath(os.environ["TMPDIR"])
if os.name != "nt":
    assert os.fstat(sys.stdin.fileno()).st_mode & 0o777 == 0o600
    assert os.stat(os.getcwd()).st_mode & 0o777 == 0o700
response(payload={_valid_payload()})
"""
    executable = _write_fake(tmp_path, action)

    solution = _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)

    assert solution == RidgeSolution(
        weights=((0.1, 0.2), (-0.3, -0.4)),
        intercepts=(1.5, -2.5),
    )


def test_wrong_hash_never_executes_the_candidate(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    executable = _write_fake(tmp_path, f"open({str(marker)!r}, 'wb').close()")
    adapter = NativeRidgeAdapter(executable, "0" * 64)

    with pytest.raises(NativeRidgeIntegrityError, match="SHA-256"):
        adapter.solve(_FEATURES, _TARGETS, ridge=0.5)

    assert not marker.exists()


def test_solve_reauthenticates_after_successful_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "candidate-replaced-after-preflight"
    executable.write_bytes(b"initial-candidate")
    executable.chmod(0o700)
    adapter = _adapter(executable)
    adapter.preflight(sample_count=2, feature_count=2, target_count=2)
    executable.write_bytes(executable.read_bytes() + b"-replaced")
    executable.chmod(0o700)

    def unexpected_process(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(native_ridge_module.subprocess, "Popen", unexpected_process)

    with pytest.raises(NativeRidgeIntegrityError, match="SHA-256"):
        adapter.solve(_FEATURES, _TARGETS, ridge=0.5)


@pytest.mark.parametrize(
    ("operation", "fail_on_call", "error_type", "message"),
    (
        ("chmod", 1, NativeRidgeIntegrityError, "temporary workspace"),
        ("fstat", 1, NativeRidgeIntegrityError, "opened native ridge binary"),
        ("read", 1, NativeRidgeIntegrityError, "snapshot native ridge binary"),
        ("write", 1, NativeRidgeIntegrityError, "snapshot native ridge binary"),
        ("fsync", 1, NativeRidgeIntegrityError, "snapshot native ridge binary"),
        ("chmod", 2, NativeRidgeIntegrityError, "snapshot executable"),
        ("fsync", 2, NativeRidgeExecutionError, "serialize native ridge request"),
        ("fstat", 4, NativeRidgeExecutionError, "serialize native ridge request"),
    ),
)
def test_adapter_wraps_owned_workspace_auth_snapshot_and_request_os_errors(
    operation: str,
    fail_on_call: int,
    error_type: type[NativeRidgeError],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _write_fake(tmp_path, f"response(payload={_valid_payload()})")
    proxy = _FailingOSProxy(
        operation,
        fail_on_call=fail_on_call,
        failure=OSError("injected adapter I/O failure"),
    )
    monkeypatch.setattr(native_ridge_module, "os", proxy)

    with pytest.raises(error_type, match=message) as captured:
        _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)

    assert isinstance(captured.value.__cause__, OSError)
    assert proxy.calls == fail_on_call


def test_preflight_wraps_binary_authentication_os_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _write_fake(tmp_path, f"response(payload={_valid_payload()})")
    proxy = _FailingOSProxy(
        "read",
        fail_on_call=1,
        failure=OSError("injected authentication I/O failure"),
    )
    monkeypatch.setattr(native_ridge_module, "os", proxy)

    with pytest.raises(NativeRidgeIntegrityError, match="authenticate native ridge binary"):
        _adapter(executable).preflight(sample_count=2, feature_count=2, target_count=2)


@_SCRIPT_FAKE_ONLY
@pytest.mark.parametrize(
    ("operation", "fail_on_call", "message"),
    (
        ("fstat", 5, "original native ridge stdout"),
        ("read", 3, "read native ridge response"),
    ),
)
def test_adapter_wraps_owned_response_os_errors(
    operation: str,
    fail_on_call: int,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _write_fake(tmp_path, f"response(payload={_valid_payload()})")
    proxy = _FailingOSProxy(
        operation,
        fail_on_call=fail_on_call,
        failure=OSError("injected response I/O failure"),
    )
    monkeypatch.setattr(native_ridge_module, "os", proxy)

    with pytest.raises(NativeRidgeProtocolError, match=message) as captured:
        _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)

    assert isinstance(captured.value.__cause__, OSError)
    assert proxy.calls == fail_on_call


@pytest.mark.parametrize("failure", (KeyboardInterrupt(), SystemExit(73)))
def test_adapter_does_not_convert_process_control_base_exceptions(
    failure: BaseException,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _write_fake(tmp_path, f"response(payload={_valid_payload()})")
    proxy = _FailingOSProxy("fstat", fail_on_call=1, failure=failure)
    monkeypatch.setattr(native_ridge_module, "os", proxy)

    with pytest.raises(type(failure)) as captured:
        _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)

    if isinstance(failure, SystemExit):
        assert captured.value.code == 73


@_SCRIPT_FAKE_ONLY
def test_replaced_response_path_is_rejected_even_with_valid_stdout(tmp_path: Path) -> None:
    executable = _write_fake(
        tmp_path,
        "os.unlink('response.bin'); open('response.bin', 'wb').write(b'replaced'); "
        f"response(payload={_valid_payload()})",
    )
    with pytest.raises(NativeRidgeProtocolError, match="no longer names stdout"):
        _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)


@pytest.mark.parametrize(
    ("path_kind", "digest", "error_type", "message"),
    [
        ("relative", "0" * 64, ValueError, "absolute"),
        ("absolute", "A" * 64, ValueError, "lowercase"),
        ("absolute", "0" * 63, ValueError, "64 lowercase"),
        ("absolute", "0" * 64, None, ""),
    ],
)
def test_adapter_configuration_is_explicit_and_canonical(
    tmp_path: Path,
    path_kind: str,
    digest: str,
    error_type: type[Exception] | None,
    message: str,
) -> None:
    path = "relative" if path_kind == "relative" else str(tmp_path / "absolute-candidate")
    if error_type is None:
        assert NativeRidgeAdapter(path, digest).binary_path == path
    else:
        with pytest.raises(error_type, match=message):
            NativeRidgeAdapter(path, digest)


@pytest.mark.parametrize(
    "path",
    (
        "//server/share/tierroute-ridge",
        "//?/C:/tierroute-ridge.exe",
        r"\\server\share\tierroute-ridge.exe",
        r"\\?\C:\tierroute-ridge.exe",
    ),
)
def test_adapter_rejects_unc_and_device_style_paths_on_every_host(path: str) -> None:
    with pytest.raises(ValueError, match="UNC or device-style"):
        NativeRidgeAdapter(path, "0" * 64)


def test_oversized_sparse_binary_is_rejected_before_copy_or_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "oversized-ridge"
    with executable.open("wb") as stream:
        stream.seek(MAX_BINARY_BYTES)
        stream.write(b"x")
    executable.chmod(0o700)

    def unexpected_copy(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    def unexpected_process(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(native_ridge_module, "_write_all", unexpected_copy)
    monkeypatch.setattr(native_ridge_module.subprocess, "Popen", unexpected_process)

    with pytest.raises(NativeRidgeIntegrityError, match="exceeds reviewed limit"):
        NativeRidgeAdapter(executable, "0" * 64).solve(_FEATURES, _TARGETS, ridge=0.5)


def test_binary_size_is_rechecked_on_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "growing-ridge"
    executable.write_bytes(b"small")
    executable.chmod(0o700)
    digest = _sha256(executable)
    real_open = native_ridge_module.os.open
    grew = False

    def grow_before_open(path: object, flags: int, *args: object) -> int:
        nonlocal grew
        if not grew and os.fspath(path) == os.fspath(executable):
            grew = True
            with executable.open("r+b") as stream:
                stream.truncate(MAX_BINARY_BYTES + 1)
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(native_ridge_module.os, "open", grow_before_open)

    with pytest.raises(NativeRidgeIntegrityError, match="exceeds reviewed limit"):
        NativeRidgeAdapter(executable, digest).preflight(
            sample_count=2,
            feature_count=2,
            target_count=2,
        )
    assert grew is True


def test_binary_stream_count_cannot_exceed_reviewed_lstat_and_fstat_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "forged-growth-ridge"
    executable.write_bytes(b"small")
    executable.chmod(0o700)
    remaining = MAX_BINARY_BYTES + 1

    def forged_read(file_descriptor: int, count: int) -> bytes:
        del file_descriptor
        nonlocal remaining
        length = min(count, remaining)
        remaining -= length
        return b"x" * length

    monkeypatch.setattr(native_ridge_module.os, "read", forged_read)

    with pytest.raises(NativeRidgeIntegrityError, match="exceeded the reviewed size limit"):
        NativeRidgeAdapter(executable, "0" * 64).preflight(
            sample_count=2,
            feature_count=2,
            target_count=2,
        )


def test_resource_preflight_runs_before_binary_authentication(tmp_path: Path) -> None:
    missing = tmp_path / "missing-ridge"
    with pytest.raises(ValueError, match="work estimate"):
        NativeRidgeAdapter(missing, "0" * 64).preflight(
            sample_count=100_000,
            feature_count=1_000,
            target_count=1,
        )


@_SCRIPT_FAKE_ONLY
def test_symlink_and_non_executable_are_rejected(tmp_path: Path) -> None:
    executable = _write_fake(tmp_path, f"response(payload={_valid_payload()})")
    link = tmp_path / "link"
    link.symlink_to(executable)
    with pytest.raises(NativeRidgeIntegrityError, match="non-symlink"):
        _adapter(link).solve(_FEATURES, _TARGETS, ridge=0.5)

    if os.name != "nt":
        executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
        with pytest.raises(NativeRidgeIntegrityError, match="not executable"):
            NativeRidgeAdapter(executable, _sha256(executable)).solve(
                _FEATURES, _TARGETS, ridge=0.5
            )


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (f"response(response_magic=b'WRONGMAG', payload={_valid_payload()})", "wrong magic"),
        (f"response(response_version=2, payload={_valid_payload()})", "unsupported version"),
        (f"response(response_id=b'x' * 32, payload={_valid_payload()})", "ID does not match"),
        (f"response(response_d=d + 1, payload={_valid_payload()})", "dimensions do not match"),
        (f"response(response_m=m + 1, payload={_valid_payload()})", "dimensions do not match"),
        (f"response(status=99, payload={_valid_payload()})", "unknown status"),
        ("response(payload=struct.pack('<d', 1.0))", "size"),
        (f"response(payload={_valid_payload()} + b'x')", "exceeded"),
        (
            "response(payload=struct.pack('<6d', float('nan'), 0, 0, 0, 0, 0))",
            "non-finite",
        ),
        ("response(status=3, payload=b'x')", "error response contains a payload"),
    ],
)
@_SCRIPT_FAKE_ONLY
def test_malformed_or_unauthenticated_responses_fail_closed(
    tmp_path: Path, action: str, message: str
) -> None:
    executable = _write_fake(tmp_path, action)
    with pytest.raises(NativeRidgeProtocolError, match=message):
        _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)


@_SCRIPT_FAKE_ONLY
def test_known_error_status_and_capped_stderr_are_reported(tmp_path: Path) -> None:
    executable = _write_fake(
        tmp_path,
        "sys.stderr.buffer.write(b'x' * 100000); response(status=5); raise SystemExit(5)",
    )
    with pytest.raises(NativeRidgeStatusError) as captured:
        _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)

    assert captured.value.status == 5
    assert len(captured.value.stderr) <= MAX_STDERR_BYTES + len(b"\n[stderr truncated]")
    assert captured.value.stderr.endswith(b"[stderr truncated]")


@_SCRIPT_FAKE_ONLY
def test_error_status_requires_a_nonzero_process_exit(tmp_path: Path) -> None:
    executable = _write_fake(tmp_path, "response(status=5)")
    with pytest.raises(NativeRidgeProtocolError, match="error status with a successful"):
        _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5)


@_SCRIPT_FAKE_ONLY
def test_large_stderr_is_drained_without_deadlock_on_success(tmp_path: Path) -> None:
    executable = _write_fake(
        tmp_path,
        f"sys.stderr.buffer.write(b'x' * 100000); response(payload={_valid_payload()})",
    )
    assert _adapter(executable).solve(_FEATURES, _TARGETS, ridge=0.5).intercepts == (
        1.5,
        -2.5,
    )


@_SCRIPT_FAKE_ONLY
def test_timeout_and_nonzero_success_exit_fail_closed(tmp_path: Path) -> None:
    sleepy = _write_fake(tmp_path, "time.sleep(5)", name="sleepy")
    with pytest.raises(NativeRidgeExecutionError, match="timed out"):
        _adapter(sleepy, timeout=0.05).solve(_FEATURES, _TARGETS, ridge=0.5)

    failing = _write_fake(
        tmp_path,
        f"response(payload={_valid_payload()}); "
        "sys.stderr.write('controlled'); raise SystemExit(7)",
    )
    with pytest.raises(NativeRidgeProtocolError, match="nonzero process status 7: controlled"):
        _adapter(failing).solve(_FEATURES, _TARGETS, ridge=0.5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX process-group regression")
def test_timeout_kills_descendants_holding_output_pipes(tmp_path: Path) -> None:
    executable = _write_fake(
        tmp_path,
        "child = os.fork()\nif child == 0:\n    time.sleep(5)\n    os._exit(0)\ntime.sleep(5)",
    )
    started = time.monotonic()
    with pytest.raises(NativeRidgeExecutionError, match="timed out"):
        _adapter(executable, timeout=0.05).solve(_FEATURES, _TARGETS, ridge=0.5)
    assert time.monotonic() - started < 3.0


def test_reviewed_work_preflight_admits_routerbench_shape_and_rejects_excess() -> None:
    from tierroute.predictors.native_ridge import preflight_native_ridge

    preflight_native_ridge(sample_count=34_778, feature_count=1_036, target_count=11)
    with pytest.raises(ValueError, match="work estimate"):
        preflight_native_ridge(sample_count=100_000, feature_count=1_000, target_count=1)
    with pytest.raises(ValueError, match="allocation estimate"):
        preflight_native_ridge(sample_count=270_000, feature_count=1_000, target_count=1)


def test_adapter_preflights_work_before_rows_binary_or_process(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    executable = _write_fake(tmp_path, f"open({str(marker)!r}, 'wb').close()")

    class LazyFeatures:
        def __len__(self) -> int:
            return 100_000

        def __getitem__(self, index: int) -> tuple[float, ...]:
            if index == 0:
                return (0.0,) * 1_000
            raise AssertionError("work preflight must precede row traversal")

    class LazyTargets:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> tuple[float, ...]:
            raise AssertionError("work preflight must precede target traversal")

    with pytest.raises(ValueError, match="work estimate"):
        _adapter(executable).solve(  # type: ignore[arg-type]
            LazyFeatures(),
            LazyTargets(),
            ridge=1.0,
        )
    assert not marker.exists()


@pytest.mark.parametrize(
    ("features", "targets", "ridge", "error_type", "message"),
    [
        ((), ((1.0,),), 1.0, ValueError, "sample_count"),
        (((1.0,), (2.0, 3.0)), ((1.0, 2.0),), 1.0, ValueError, "rectangular"),
        (((1.0,),), (), 1.0, ValueError, "target_count"),
        (((1.0,),), ((1.0, 2.0),), 1.0, ValueError, "width 1"),
        (((float("nan"),),), ((1.0,),), 1.0, ValueError, "must be finite"),
        (((1.0,),), ((float("inf"),),), 1.0, ValueError, "must be finite"),
        (((1.0,),), ((1.0,),), 0.0, ValueError, "positive"),
        (((1.0,),), ((1.0,),), True, TypeError, "real number"),
        (((1.0,),), ((1.0,),), Fraction(10**10_000, 1), ValueError, "representable"),
    ],
)
def test_invalid_input_is_rejected_before_process_execution(
    tmp_path: Path,
    features: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    ridge: object,
    error_type: type[Exception],
    message: str,
) -> None:
    executable = _write_fake(tmp_path, "raise SystemExit(88)")
    with pytest.raises(error_type, match=message):
        _adapter(executable).solve(features, targets, ridge=ridge)  # type: ignore[arg-type]


def _available_compiler() -> Path | None:
    if os.name == "nt":
        candidate = shutil.which("cl")
    else:
        candidate = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    return Path(candidate).resolve() if candidate else None


def _requested_compiler_output(command: Sequence[str]) -> Path:
    if os.name == "nt":
        output_argument = next(item for item in command if item.startswith("/Fe:"))
        return Path(output_argument.removeprefix("/Fe:"))
    return Path(command[command.index("-o") + 1])


def test_build_helper_rejects_relative_or_existing_output(tmp_path: Path) -> None:
    relative = subprocess.run(
        [
            sys.executable,
            str(_BUILD_SCRIPT),
            "--output",
            "relative-binary",
            "--compiler",
            sys.executable,
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert relative.returncode == 2
    assert "--output must be absolute" in relative.stderr

    existing = tmp_path / "existing"
    existing.write_bytes(b"owner-data")
    refused = subprocess.run(
        [
            sys.executable,
            str(_BUILD_SCRIPT),
            "--output",
            str(existing),
            "--compiler",
            sys.executable,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert "does not already exist" in refused.stderr
    assert existing.read_bytes() == b"owner-data"


@pytest.mark.parametrize(
    "path",
    (
        "//server/share/tierroute-ridge",
        "//?/C:/tierroute-ridge.exe",
        r"\\server\share\tierroute-ridge.exe",
        r"\\?\C:\tierroute-ridge.exe",
    ),
)
def test_build_helper_rejects_unc_and_device_style_paths_on_every_host(path: str) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    for validator_name in ("_safe_output_path", "_safe_compiler_path"):
        with pytest.raises(ValueError, match="UNC or device-style"):
            build_module[validator_name](path)


def test_build_helper_refuses_oversized_compiler_output_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"

    def create_oversized_output(
        command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        del kwargs
        if os.name == "nt":
            output_argument = next(item for item in command if item.startswith("/Fe:"))
            generated = Path(output_argument.removeprefix("/Fe:"))
        else:
            generated = Path(command[command.index("-o") + 1])
        with generated.open("wb") as stream:
            stream.seek(MAX_BINARY_BYTES)
            stream.write(b"x")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", create_oversized_output)

    with pytest.raises(RuntimeError, match="exceeds reviewed limit"):
        compile_candidate(
            source=_NATIVE_SOURCE,
            output=output,
            compiler=Path(sys.executable),
        )
    assert not output.exists()


def test_build_helper_converts_compiler_timeout_to_controlled_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"

    def time_out(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        raise subprocess.TimeoutExpired(command, 180)

    monkeypatch.setattr(build_module["subprocess"], "run", time_out)

    with pytest.raises(RuntimeError, match="exceeded the 180-second time limit"):
        compile_candidate(
            source=_NATIVE_SOURCE,
            output=output,
            compiler=Path(sys.executable),
        )

    assert not output.exists()


def test_build_helper_never_links_the_compiler_owned_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"
    original_payload = b"authenticated compiler output"
    compiler_output: Path | None = None

    def create_output(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        nonlocal compiler_output
        compiler_output = _requested_compiler_output(command)
        compiler_output.write_bytes(original_payload)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    real_link = os.link

    def link_then_mutate_compiler_output(source: object, destination: object) -> None:
        real_link(source, destination)
        assert compiler_output is not None
        assert os.fspath(source) != os.fspath(compiler_output)
        compiler_output.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        compiler_output.write_bytes(b"background compiler mutation")

    monkeypatch.setattr(build_module["subprocess"], "run", create_output)
    monkeypatch.setattr(build_module["os"], "link", link_then_mutate_compiler_output)

    manifest = compile_candidate(
        source=_NATIVE_SOURCE,
        output=output,
        compiler=Path(sys.executable),
    )

    assert output.read_bytes() == original_payload
    assert manifest["sha256"] == _sha256(output)


def test_build_helper_fallback_reauthenticates_and_removes_mismatched_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"

    def create_output(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        _requested_compiler_output(command).write_bytes(b"authenticated compiler output")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    def mutate_snapshot_then_force_fallback(source: object, destination: object) -> None:
        del destination
        snapshot = Path(source)
        snapshot.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        snapshot.write_bytes(b"mutation before fallback reopen")
        raise OSError("forced hard-link failure")

    monkeypatch.setattr(build_module["subprocess"], "run", create_output)
    monkeypatch.setattr(build_module["os"], "link", mutate_snapshot_then_force_fallback)

    with pytest.raises(RuntimeError, match="does not match its verified SHA-256"):
        compile_candidate(
            source=_NATIVE_SOURCE,
            output=output,
            compiler=Path(sys.executable),
        )

    assert not output.exists()


def test_build_helper_closes_fds_but_preserves_replacement_when_destination_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"
    destination_descriptor: int | None = None
    failed_destination_fstat = False

    def create_output(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        _requested_compiler_output(command).write_bytes(b"authenticated compiler output")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    def force_fallback(source: object, destination: object) -> None:
        del source, destination
        raise OSError("forced hard-link failure")

    real_open = os.open
    real_fstat = os.fstat

    def track_destination_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal destination_descriptor
        descriptor = real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]
        if os.fspath(path) == os.fspath(output):
            destination_descriptor = descriptor
        return descriptor

    def fail_destination_fstat(descriptor: int) -> os.stat_result:
        nonlocal failed_destination_fstat
        if descriptor == destination_descriptor and not failed_destination_fstat:
            failed_destination_fstat = True
            output.unlink()
            output.write_bytes(b"replacement-owner-data")
            raise OSError("simulated destination fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(build_module["subprocess"], "run", create_output)
    monkeypatch.setattr(build_module["os"], "link", force_fallback)
    monkeypatch.setattr(build_module["os"], "open", track_destination_open)
    monkeypatch.setattr(build_module["os"], "fstat", fail_destination_fstat)

    with pytest.raises(RuntimeError, match="cannot authenticate newly created fallback output"):
        compile_candidate(
            source=_NATIVE_SOURCE,
            output=output,
            compiler=Path(sys.executable),
        )

    assert destination_descriptor is not None
    with pytest.raises(OSError):
        real_fstat(destination_descriptor)
    assert failed_destination_fstat is True
    assert output.read_bytes() == b"replacement-owner-data"


def test_build_helper_removes_authenticated_output_when_destination_lstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"
    failed_destination_lstat = False

    def create_output(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        _requested_compiler_output(command).write_bytes(b"authenticated compiler output")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    def force_fallback(source: object, destination: object) -> None:
        del source, destination
        raise OSError("forced hard-link failure")

    real_lstat = Path.lstat

    def fail_destination_lstat(path: Path) -> os.stat_result:
        nonlocal failed_destination_lstat
        if path == output and not failed_destination_lstat:
            failed_destination_lstat = True
            raise OSError("simulated destination lstat failure")
        return real_lstat(path)

    monkeypatch.setattr(build_module["subprocess"], "run", create_output)
    monkeypatch.setattr(build_module["os"], "link", force_fallback)
    monkeypatch.setattr(Path, "lstat", fail_destination_lstat)

    with pytest.raises(RuntimeError, match="cannot authenticate newly created fallback output"):
        compile_candidate(
            source=_NATIVE_SOURCE,
            output=output,
            compiler=Path(sys.executable),
        )

    assert failed_destination_lstat is True
    assert not output.exists()


def test_build_helper_attempts_all_closes_and_removes_output_on_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"
    destination_descriptor: int | None = None
    failed_destination_close = False

    def create_output(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        _requested_compiler_output(command).write_bytes(b"authenticated compiler output")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    def force_fallback(source: object, destination: object) -> None:
        del source, destination
        raise OSError("forced hard-link failure")

    real_open = os.open
    real_close = os.close

    def track_destination_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal destination_descriptor
        descriptor = real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]
        if os.fspath(path) == os.fspath(output):
            destination_descriptor = descriptor
        return descriptor

    def fail_destination_close(descriptor: int) -> None:
        nonlocal failed_destination_close
        if descriptor == destination_descriptor and not failed_destination_close:
            failed_destination_close = True
            real_close(descriptor)
            raise OSError("simulated destination close failure")
        real_close(descriptor)

    monkeypatch.setattr(build_module["subprocess"], "run", create_output)
    monkeypatch.setattr(build_module["os"], "link", force_fallback)
    monkeypatch.setattr(build_module["os"], "open", track_destination_open)
    monkeypatch.setattr(build_module["os"], "close", fail_destination_close)

    with pytest.raises(RuntimeError, match="cannot close fallback output descriptors"):
        compile_candidate(
            source=_NATIVE_SOURCE,
            output=output,
            compiler=Path(sys.executable),
        )

    assert failed_destination_close is True
    assert not output.exists()


def test_build_helper_reauthenticates_destination_and_removes_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    compile_candidate = build_module["_compile"]
    output = tmp_path / "published-ridge"

    def create_output(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
        del kwargs
        _requested_compiler_output(command).write_bytes(b"authenticated compiler output")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    real_sha256 = build_module["_sha256"]

    def tamper_before_destination_hash(path: Path) -> str:
        if path == output:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            path.write_bytes(b"mutation after publish")
        return real_sha256(path)

    monkeypatch.setattr(build_module["subprocess"], "run", create_output)
    monkeypatch.setitem(compile_candidate.__globals__, "_sha256", tamper_before_destination_hash)

    with pytest.raises(RuntimeError, match="does not match its manifest SHA-256"):
        compile_candidate(
            source=_NATIVE_SOURCE,
            output=output,
            compiler=Path(sys.executable),
        )

    assert not output.exists()


def test_native_binary_ceiling_matches_build_helper_and_protocol_document() -> None:
    build_module = runpy.run_path(str(_BUILD_SCRIPT), run_name="tierroute_native_build_test")
    protocol = (_REPOSITORY_ROOT / "docs" / "native-ridge-protocol.md").read_text(encoding="utf-8")

    assert build_module["MAX_BINARY_BYTES"] == MAX_BINARY_BYTES == 16 * 1024 * 1024
    assert "| sidecar executable bytes | 16 MiB |" in protocol


@pytest.mark.skipif(os.name == "nt", reason="symlink setup is platform-specific")
def test_build_helper_rejects_symlink_compiler_and_output_parent(tmp_path: Path) -> None:
    compiler_link = tmp_path / "compiler-link"
    compiler_link.symlink_to(sys.executable)
    refused_compiler = subprocess.run(
        [
            sys.executable,
            str(_BUILD_SCRIPT),
            "--output",
            str(tmp_path / "binary"),
            "--compiler",
            str(compiler_link),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_compiler.returncode == 2
    assert "regular non-symlink" in refused_compiler.stderr

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    refused_parent = subprocess.run(
        [
            sys.executable,
            str(_BUILD_SCRIPT),
            "--output",
            str(parent_link / "binary"),
            "--compiler",
            sys.executable,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused_parent.returncode == 2
    assert "non-symlink directory" in refused_parent.stderr


@pytest.fixture(scope="module")
def compiled_native_ridge(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    compiler = _available_compiler()
    if compiler is None or not _NATIVE_SOURCE.is_file():
        pytest.skip("native C11 source or a platform compiler is unavailable")
    output = tmp_path_factory.mktemp("native-ridge-build") / (
        "tierroute-ridge.exe" if os.name == "nt" else "tierroute-ridge"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(_BUILD_SCRIPT),
            "--output",
            str(output),
            "--compiler",
            str(compiler),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        pytest.fail(f"native build helper failed:\n{completed.stderr}")
    manifest = json.loads(completed.stdout)
    assert manifest["output"] == str(output)
    assert manifest["source"] == str(_NATIVE_SOURCE)
    assert manifest["source_sha256"] == _sha256(_NATIVE_SOURCE)
    assert manifest["compiler"] == str(compiler)
    assert manifest["sha256"] == _sha256(output)
    return output, manifest["sha256"]


def test_compiled_adapter_fits_a_small_predictor_artifact(
    compiled_native_ridge: tuple[Path, str],
) -> None:
    executable, digest = compiled_native_ridge
    adapter = NativeRidgeAdapter(executable, digest, timeout_seconds=30.0)
    examples = load_evaluation_dataset().examples

    artifact = fit_calibrated_bilinear(
        examples,
        config=BilinearTrainingConfig(solver_id=NATIVE_C11_RIDGE_SOLVER_ID),
        solver=adapter,
    )

    assert adapter.solver_id == NATIVE_C11_RIDGE_SOLVER_ID
    assert artifact.solver_id == NATIVE_C11_RIDGE_SOLVER_ID
    assert artifact.training_example_count == len(examples)
    assert all(
        math.isfinite(value)
        for model_id in artifact.model_ids
        for value in (*artifact.model_weights[model_id], artifact.model_bias[model_id])
    )


def _raw_request(
    *,
    magic: bytes = b"TRRIDG01",
    version: int = 1,
    flags: int = 0,
    sample_count: int = 2,
    feature_count: int = 2,
    target_count: int = 1,
    ridge: float = 1.0,
    values: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    trailing: bytes = b"",
) -> bytes:
    return (
        _RAW_REQUEST.pack(
            magic,
            version,
            flags,
            _RAW_REQUEST_ID,
            sample_count,
            feature_count,
            target_count,
            ridge,
        )
        + struct.pack(f"<{len(values)}d", *values)
        + trailing
    )


@pytest.mark.parametrize(
    ("request_bytes", "expected_status", "expected_echo"),
    [
        (b"\x00" * (_RAW_REQUEST.size - 1), 1, False),
        (_raw_request(magic=b"BADMAGIC", values=()), 1, True),
        (_raw_request(version=2, values=()), 1, True),
        (_raw_request(flags=1, values=()), 1, True),
        (_raw_request(sample_count=0, values=()), 2, True),
        (_raw_request(feature_count=0, values=()), 2, True),
        (_raw_request(target_count=0, values=()), 2, True),
        (_raw_request(sample_count=1_000_001, values=()), 2, True),
        (_raw_request(feature_count=4_097, values=()), 2, True),
        (_raw_request(target_count=257, values=()), 2, True),
        (
            _raw_request(sample_count=270_000, feature_count=1_000, values=()),
            2,
            True,
        ),
        (
            _raw_request(sample_count=100_000, feature_count=1_000, values=()),
            2,
            True,
        ),
        (
            _raw_request(
                sample_count=(1 << 64) - 1,
                feature_count=(1 << 64) - 1,
                target_count=(1 << 64) - 1,
                values=(),
            ),
            2,
            True,
        ),
        (_raw_request(ridge=float("nan"), values=()), 3, True),
        (_raw_request(ridge=float("inf"), values=()), 3, True),
        (_raw_request(ridge=0.0, values=()), 3, True),
        (_raw_request(values=(1.0, 2.0)), 1, True),
        (_raw_request(values=(float("nan"), 2.0, 3.0, 4.0, 5.0, 6.0)), 3, True),
        (_raw_request(values=(1.0, 2.0, 3.0, 4.0, 5.0, float("inf"))), 3, True),
        (_raw_request(trailing=b"x"), 1, True),
    ],
)
def test_compiled_native_rejects_malformed_raw_protocol_exactly(
    compiled_native_ridge: tuple[Path, str],
    request_bytes: bytes,
    expected_status: int,
    expected_echo: bool,
) -> None:
    path, digest = compiled_native_ridge
    assert _sha256(path) == digest
    completed = subprocess.run(
        [str(path)],
        input=request_bytes,
        check=False,
        capture_output=True,
        timeout=5,
    )

    assert completed.returncode != 0
    assert len(completed.stdout) == _RAW_RESPONSE.size
    magic, version, status_code, request_id, response_d, response_m = _RAW_RESPONSE.unpack(
        completed.stdout
    )
    assert magic == b"TRRRES01"
    assert version == 1
    assert status_code == expected_status
    if expected_echo:
        header = _RAW_REQUEST.unpack_from(request_bytes)
        assert request_id == _RAW_REQUEST_ID
        assert response_d == header[5]
        assert response_m == header[6]
    else:
        assert request_id == bytes(32)
        assert response_d == response_m == 0
    assert 0 < len(completed.stderr) <= MAX_STDERR_BYTES


@pytest.mark.parametrize(
    ("features", "targets", "ridge"),
    [
        (
            ((0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (3.0, 10.0)),
            ((1.0, 2.5, 4.0, 8.0), (-1.0, 0.0, 2.0, 7.0)),
            0.25,
        ),
        (
            ((1.0, 2.0, 2.0), (2.0, 4.0, 4.0), (3.0, 6.0, 6.0)),
            ((2.0, 4.0, 8.0), (1.0, -1.0, 3.0)),
            1e-3,
        ),
    ],
)
def test_compiled_native_matches_reference_for_multi_target_and_collinear_cases(
    compiled_native_ridge: tuple[Path, str],
    features: tuple[tuple[float, ...], ...],
    targets: tuple[tuple[float, ...], ...],
    ridge: float,
) -> None:
    path, digest = compiled_native_ridge
    actual = NativeRidgeAdapter(path, digest, timeout_seconds=30).solve(
        features, targets, ridge=ridge
    )
    expected = solve_centered_ridge(features, targets, ridge=ridge)
    assert actual.intercepts == pytest.approx(expected.intercepts, rel=1e-9, abs=1e-10)
    for actual_weights, expected_weights in zip(actual.weights, expected.weights, strict=True):
        assert actual_weights == pytest.approx(expected_weights, rel=1e-8, abs=1e-9)
    assert _predictions(actual, features) == pytest.approx(
        _predictions(expected, features), rel=1e-9, abs=1e-9
    )


def test_compiled_native_matches_reference_for_near_singular_ridge(
    compiled_native_ridge: tuple[Path, str],
) -> None:
    features = (
        (1.0, 1.0 + 1e-9, 2.0),
        (2.0, 2.0 - 1e-9, 4.0),
        (3.0, 3.0 + 2e-9, 6.0),
        (4.0, 4.0 - 2e-9, 8.0),
        (5.0, 5.0 + 1e-9, 10.0),
    )
    targets = ((1.0, 2.0, 4.0, 8.0, 16.0), (-2.0, -1.0, 0.0, 1.0, 2.0))
    ridge = 1e-7
    path, digest = compiled_native_ridge
    actual = NativeRidgeAdapter(path, digest, timeout_seconds=30).solve(
        features, targets, ridge=ridge
    )
    expected = solve_centered_ridge(features, targets, ridge=ridge)

    assert actual.intercepts == pytest.approx(expected.intercepts, rel=1e-7, abs=1e-8)
    for actual_weights, expected_weights in zip(actual.weights, expected.weights, strict=True):
        assert actual_weights == pytest.approx(expected_weights, rel=1e-6, abs=1e-8)
    assert _predictions(actual, features) == pytest.approx(
        _predictions(expected, features), rel=1e-7, abs=1e-8
    )


def test_compiled_native_handles_1024_dense_features_without_projection(
    compiled_native_ridge: tuple[Path, str],
) -> None:
    sample_count = 6
    feature_count = 1024
    features = tuple(
        tuple(
            math.sin((row + 1) * (column + 3) * 0.001) + 0.01 * (row - 2) * ((column % 11) - 5)
            for column in range(feature_count)
        )
        for row in range(sample_count)
    )
    targets = (
        tuple(0.5 * row * row - row + 2.0 for row in range(sample_count)),
        tuple(math.cos(row) for row in range(sample_count)),
    )
    ridge = 0.75
    expected = _dual_ridge(features, targets, ridge)
    path, digest = compiled_native_ridge

    actual = NativeRidgeAdapter(path, digest, timeout_seconds=120).solve(
        features, targets, ridge=ridge
    )

    assert len(actual.weights) == 2
    assert all(len(weights) == feature_count for weights in actual.weights)
    assert actual.intercepts == pytest.approx(expected.intercepts, rel=1e-7, abs=1e-8)
    for actual_weights, expected_weights in zip(actual.weights, expected.weights, strict=True):
        assert actual_weights == pytest.approx(expected_weights, rel=1e-6, abs=1e-8)
    assert _predictions(actual, features) == pytest.approx(
        _predictions(expected, features), rel=1e-7, abs=1e-8
    )


def _predictions(solution: RidgeSolution, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(
        intercept + math.fsum(value * weight for value, weight in zip(row, weights, strict=True))
        for row in features
        for weights, intercept in zip(solution.weights, solution.intercepts, strict=True)
    )


def _dual_ridge(
    features: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    ridge: float,
) -> RidgeSolution:
    """Small-n O(n^3 + n^2 d) oracle for the 1,024-dimensional fixture."""

    n = len(features)
    d = len(features[0])
    feature_means = tuple(math.fsum(row[j] for row in features) / n for j in range(d))
    centered = tuple(tuple(row[j] - feature_means[j] for j in range(d)) for row in features)
    kernel = [
        [
            math.fsum(centered[i][j] * centered[k][j] for j in range(d))
            + (ridge if i == k else 0.0)
            for k in range(n)
        ]
        for i in range(n)
    ]
    all_weights: list[tuple[float, ...]] = []
    intercepts: list[float] = []
    for target in targets:
        mean = math.fsum(target) / n
        alpha = _solve_dense(kernel, [value - mean for value in target])
        weights = tuple(math.fsum(centered[i][j] * alpha[i] for i in range(n)) for j in range(d))
        all_weights.append(weights)
        intercepts.append(mean - math.fsum(feature_means[j] * weights[j] for j in range(d)))
    return RidgeSolution(tuple(all_weights), tuple(intercepts))


def _solve_dense(
    matrix: Sequence[Sequence[float]], right_hand_side: Sequence[float]
) -> list[float]:
    augmented = [[*row, value] for row, value in zip(matrix, right_hand_side, strict=True)]
    size = len(augmented)
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        augmented[pivot_index], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_index],
        )
        pivot = augmented[pivot_index][pivot_index]
        assert pivot != 0.0
        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot
        for row in range(size):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for column in range(pivot_index, size + 1):
                augmented[row][column] -= factor * augmented[pivot_index][column]
    return [augmented[row][size] for row in range(size)]
