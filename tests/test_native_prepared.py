# SPDX-License-Identifier: Apache-2.0
"""Native prepared-session parity, trust-boundary, and lifetime tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

import tierroute.predictors.native_prepared as native_prepared_module
from tierroute.core import ModelSpec
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.predictors.native_prepared import (
    MAX_MODELED_C_HEAP_BYTES,
    MAX_RESULT_FILE_BYTES,
    NativePreparedClosedError,
    NativePreparedExecutionError,
    NativePreparedIntegrityError,
    NativePreparedProtocolError,
    NativePreparedSessionAdapter,
    NativePreparedSessionResult,
    NativePreparedStatusError,
    preflight_native_prepared_session,
)
from tierroute.predictors.prepared_execution import (
    PreparedCoefficientBundle,
    PreparedRawScoreBundle,
    build_prepared_coefficient_bundle,
    build_prepared_raw_score_bundle,
)
from tierroute.predictors.prepared_files import (
    PreparedStoreFileMetadata,
    PreparedStoreFileReceipt,
    authenticate_prepared_store_file,
    estimate_prepared_session,
    write_prepared_store_file,
)
from tierroute.predictors.prepared_graph import build_prepared_nested_lodo_plan
from tierroute.predictors.prepared_store import (
    PreparedFeatureStore,
    build_prepared_domain_statistics,
    build_prepared_feature_store,
    prepared_fit_source_sha256,
)

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_SOURCE = _ROOT / "native" / "tierroute_prepared.c"
_MODEL_IDS = ("cheap", "premium")
_FAKE_ONLY = pytest.mark.skipif(
    os.name == "nt",
    reason="shebang fake children are POSIX-only; compiled corpus remains portable",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_request_nonce_rejects_the_protocols_all_zero_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter((b"\0" * 32, b"\0" * 31 + b"\x01"))
    monkeypatch.setattr(
        native_prepared_module.secrets,
        "token_bytes",
        lambda count: next(responses) if count == 32 else b"\0" * count,
    )

    assert native_prepared_module._nonzero_request_nonce() == b"\0" * 31 + b"\x01"


def _examples() -> tuple[EvaluationExample, ...]:
    domains = ("alpha", "bravo", "charlie", "delta")
    prompts = (
        "Debug this Python API and code safely.",
        "Prove the equation x^2 + y^2 = 1.",
        "Review this legal contract and court statute.",
        "Assess a clinical medicine diagnosis.",
    )
    examples: list[EvaluationExample] = []
    for index in range(12):
        model_specs = (
            ModelSpec("premium", Decimal("5")),
            ModelSpec("cheap", Decimal("1")),
        )
        outcomes = tuple(
            CandidateOutcome(
                model_id=model_id,
                output=f"row-{index:02d}:{model_id}",
                cost=Decimal("1") if model_id == "cheap" else Decimal("5"),
                quality=(0.15 + 0.019 * index if model_id == "cheap" else 0.55 + 0.023 * index),
            )
            for model_id in _MODEL_IDS
        )
        examples.append(
            EvaluationExample(
                example_id=f"row-{index:02d}",
                prompt=f"{prompts[index % 4]} Variant {index}.",
                domain=domains[index % 4],
                candidate_models=model_specs,
                outcomes=outcomes,
            )
        )
    return tuple(examples)


def _examples_for_domain_count(
    domain_count: int,
    model_ids: tuple[str, ...],
) -> tuple[EvaluationExample, ...]:
    prompts = (
        "Debug this Python API and code safely.",
        "Prove the equation x^2 + y^2 = 1.",
        "Review this legal contract and court statute.",
        "Assess a clinical medicine diagnosis.",
    )
    costs = {"cheap": Decimal("1"), "mid": Decimal("3"), "premium": Decimal("5")}
    quality_bases = {"cheap": 0.2, "mid": 0.4, "premium": 0.6}
    examples: list[EvaluationExample] = []
    for domain_index in range(domain_count):
        for replica in range(2):
            index = 2 * domain_index + replica
            model_specs = tuple(
                ModelSpec(model_id, costs[model_id]) for model_id in reversed(model_ids)
            )
            outcomes = tuple(
                CandidateOutcome(
                    model_id=model_id,
                    output=f"domain-row-{index:02d}:{model_id}",
                    cost=costs[model_id],
                    quality=quality_bases[model_id]
                    + (0.013 + 0.002 * model_ids.index(model_id)) * index,
                )
                for model_id in model_ids
            )
            examples.append(
                EvaluationExample(
                    example_id=f"domain-row-{index:02d}",
                    prompt=f"{prompts[index % len(prompts)]} Variant {index}.",
                    domain=f"domain-{domain_index:02d}",
                    candidate_models=model_specs,
                    outcomes=outcomes,
                )
            )
    return tuple(examples)


def _assert_complete_reference_parity(
    actual_result: NativePreparedSessionResult,
    expected_coefficients: PreparedCoefficientBundle,
    expected_scores: PreparedRawScoreBundle,
) -> None:
    assert len(actual_result.coefficients) == len(expected_coefficients.blocks)
    assert len(actual_result.scores) == len(expected_scores.blocks)
    assert len(actual_result.result_sha256) == 64
    for actual, expected in zip(
        actual_result.coefficients,
        expected_coefficients.blocks,
        strict=True,
    ):
        assert actual.subset_index == expected.subset_index
        assert actual.active_tag_mask == expected.active_tag_mask
        assert actual.continuous_means[:] == pytest.approx(
            expected.feature_schema.continuous_means,
            rel=1e-10,
            abs=1e-11,
        )
        assert actual.continuous_scales[:] == pytest.approx(
            expected.feature_schema.continuous_scales,
            rel=1e-10,
            abs=1e-11,
        )
        assert actual.intercepts[:] == pytest.approx(
            tuple(
                expected.intercept_for_model_index(index)
                for index in range(expected.plan.target_count)
            ),
            rel=1e-8,
            abs=1e-9,
        )
        expected_weights = tuple(
            value
            for model_index in range(expected.plan.target_count)
            for value in expected.weights_for_model_index(model_index)
        )
        assert actual.weights[:] == pytest.approx(
            expected_weights,
            rel=1e-7,
            abs=1e-8,
        )
    for actual, expected in zip(actual_result.scores, expected_scores.blocks, strict=True):
        flattened = tuple(value for row in expected.iter_score_rows() for value in row)
        assert actual.scores[:] == pytest.approx(
            flattened,
            rel=1e-7,
            abs=1e-8,
        )


@pytest.fixture(scope="module")
def prepared_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    PreparedFeatureStore,
    PreparedCoefficientBundle,
    PreparedRawScoreBundle,
    Path,
    PreparedStoreFileReceipt,
]:
    examples = _examples()
    domains = tuple(reversed(sorted({example.domain for example in examples})))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=12,
        target_count=len(_MODEL_IDS),
    )
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )
    statistics = build_prepared_domain_statistics(store)
    coefficients = build_prepared_coefficient_bundle(store, statistics, ridge=1.0)
    scores = build_prepared_raw_score_bundle(store, coefficients)
    path = tmp_path_factory.mktemp("prepared-store") / "fixture.trpsto"
    receipt = write_prepared_store_file(store, path)
    return store, coefficients, scores, path, receipt


def _available_compiler() -> str | None:
    if os.name == "nt":
        return shutil.which("cl") or shutil.which("clang-cl")
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


@pytest.fixture(scope="module")
def compiled_native_prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    compiler = _available_compiler()
    if compiler is None or not _NATIVE_SOURCE.is_file():
        pytest.skip("native prepared C11 source or platform compiler is unavailable")
    output = tmp_path_factory.mktemp("native-prepared") / (
        "tierroute-prepared.exe" if os.name == "nt" else "tierroute-prepared"
    )
    if os.name == "nt":
        command = [
            compiler,
            "/nologo",
            "/std:c11",
            "/O2",
            "/MT",
            "/W4",
            "/WX",
            "/D_CRT_SECURE_NO_WARNINGS",
            str(_NATIVE_SOURCE),
            f"/Fo:{output.with_suffix('.obj')}",
            f"/Fe:{output}",
        ]
    else:
        command = [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(_NATIVE_SOURCE),
            "-lm",
            "-o",
            str(output),
        ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        pytest.fail(f"native prepared compile failed:\n{completed.stderr}")
    output.chmod(0o500)
    return output, _sha256(output)


def _assert_truncated_store_reached_authenticated_scan(
    executable: Path,
    domain_counts: tuple[int, ...],
    *,
    identifier_bytes: int,
) -> None:
    domain_count = len(domain_counts)
    row_count = sum(domain_counts)
    feature_count = 12
    target_count = 1
    row_key_bytes = row_count * (2 + identifier_bytes + 32)
    domain_offset = 472 + row_key_bytes
    feature_offset = (domain_offset + row_count + 7) & ~7
    feature_bytes = 8 * row_count * feature_count
    target_offset = feature_offset + feature_bytes
    target_bytes = 8 * row_count * target_count
    file_bytes = target_offset + target_bytes
    masks = (0,) * domain_count
    estimate = estimate_prepared_session(
        domain_row_counts=domain_counts,
        domain_active_tag_masks=masks,
        feature_count=feature_count,
        target_count=target_count,
        store_file_bytes=file_bytes,
        row_key_bytes=row_key_bytes,
    )
    digest = b"\x11" * 32
    store_header = struct.pack(
        "<8sII15Q32s32s32s32s32s32s32s14Q",
        b"TRPSTO01",
        1,
        0,
        472,
        file_bytes,
        domain_count,
        row_count,
        feature_count,
        target_count,
        12,
        472,
        row_key_bytes,
        domain_offset,
        row_count,
        feature_offset,
        feature_bytes,
        target_offset,
        target_bytes,
        digest,
        b"\x22" * 32,
        b"\x33" * 32,
        b"\0" * 32,
        b"\0" * 32,
        b"\x44" * 32,
        b"\x55" * 32,
        *domain_counts,
        *(0 for _ in range(7 - domain_count)),
        *masks,
        *(0 for _ in range(7 - domain_count)),
    )
    request = struct.pack(
        "<8sII32s32s32sQQd24s",
        b"TRPSES01",
        1,
        0,
        b"\x66" * 32,
        b"\x77" * 32,
        b"\x88" * 32,
        160 + file_bytes,
        estimate.result_bytes,
        1.0,
        b"\0" * 24,
    )
    with tempfile.TemporaryFile() as request_stream:
        request_stream.write(request + store_header)
        request_stream.seek(0)
        completed = subprocess.run(
            [str(executable)],
            stdin=request_stream,
            check=False,
            capture_output=True,
            timeout=10,
        )
    assert completed.returncode == 1
    assert len(completed.stdout) == 448
    assert struct.unpack_from("<8sII", completed.stdout) == (b"TRPRES01", 1, 1)
    assert struct.unpack_from("<Q", completed.stdout, 184)[0] == 448
    assert struct.unpack_from("<4Q", completed.stdout, 200) == (
        estimate.statistics_work_units,
        estimate.solve_work_units,
        estimate.score_work_units,
        estimate.modeled_c_heap_bytes,
    )
    assert struct.unpack_from("<3Q", completed.stdout, 424) == (
        estimate.authentication_validation_bytes_scanned,
        estimate.output_numeric_cells_validated,
        estimate.file_backed_input_bytes,
    )


def test_compiled_d4_session_matches_complete_python_reference(
    compiled_native_prepared: tuple[Path, str],
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    executable, digest = compiled_native_prepared
    _, reference_coefficients, reference_scores, store_path, receipt = prepared_fixture

    with NativePreparedSessionAdapter(executable, digest, timeout_seconds=120).run(
        store_path,
        receipt,
        ridge=1.0,
    ) as result:
        assert len(result.coefficients) == 14
        assert len(result.scores) == 28
        _assert_complete_reference_parity(result, reference_coefficients, reference_scores)


@pytest.mark.parametrize(
    ("domain_count", "model_ids"),
    (
        (5, _MODEL_IDS),
        (6, _MODEL_IDS),
        (7, ("cheap", "mid", "premium")),
    ),
)
def test_compiled_d5_to_d7_sessions_match_complete_python_reference(
    domain_count: int,
    model_ids: tuple[str, ...],
    compiled_native_prepared: tuple[Path, str],
    tmp_path: Path,
) -> None:
    examples = _examples_for_domain_count(domain_count, model_ids)
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=12,
        target_count=len(model_ids),
    )
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )
    statistics = build_prepared_domain_statistics(store)
    reference_coefficients = build_prepared_coefficient_bundle(store, statistics, ridge=1.0)
    reference_scores = build_prepared_raw_score_bundle(store, reference_coefficients)
    store_path = tmp_path / f"d{domain_count}.trpsto"
    receipt = write_prepared_store_file(store, store_path)
    executable, digest = compiled_native_prepared

    with NativePreparedSessionAdapter(executable, digest, timeout_seconds=120).run(
        store_path,
        receipt,
        ridge=1.0,
    ) as result:
        _assert_complete_reference_parity(result, reference_coefficients, reference_scores)


def test_result_views_fail_after_context_close(
    compiled_native_prepared: tuple[Path, str],
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    executable, digest = compiled_native_prepared
    _, _, _, store_path, receipt = prepared_fixture
    result = NativePreparedSessionAdapter(executable, digest).run(
        store_path,
        receipt,
        ridge=1.0,
    )
    view = result.scores[0].scores
    assert math.isfinite(view.at(0, 0))
    result.close()
    result.close()

    assert result.closed
    with pytest.raises(NativePreparedClosedError):
        _ = view[0]
    with pytest.raises(NativePreparedClosedError):
        _ = result.coefficients


def test_result_close_is_retryable_after_exported_mapping_view(
    compiled_native_prepared: tuple[Path, str],
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    executable, digest = compiled_native_prepared
    _, _, _, store_path, receipt = prepared_fixture
    result = NativePreparedSessionAdapter(executable, digest).run(
        store_path,
        receipt,
        ridge=1.0,
    )
    exported = memoryview(result._lifetime.require_open())
    with pytest.raises(NativePreparedExecutionError, match="result mapping"):
        result.close()
    assert not result.closed
    assert math.isfinite(result.scores[0].scores[0])
    exported.release()
    result.close()
    assert result.closed


def test_payload_read_and_concurrent_close_are_serialized(
    compiled_native_prepared: tuple[Path, str],
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, digest = compiled_native_prepared
    _, _, _, store_path, receipt = prepared_fixture
    result = NativePreparedSessionAdapter(executable, digest).run(
        store_path,
        receipt,
        ridge=1.0,
    )
    view = result.scores[0].scores
    original_f64 = native_prepared_module._F64
    read_entered = threading.Event()
    release_read = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    values: list[float] = []
    errors: list[BaseException] = []

    class BlockingF64:
        def unpack_from(self, mapping: object, offset: int) -> tuple[float]:
            read_entered.set()
            if not release_read.wait(timeout=5):
                raise AssertionError("concurrent read was not released")
            return original_f64.unpack_from(mapping, offset)

    monkeypatch.setattr(native_prepared_module, "_F64", BlockingF64())

    def read_one() -> None:
        try:
            values.append(view[0])
        except BaseException as error:
            errors.append(error)

    def close_result() -> None:
        close_started.set()
        try:
            result.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_finished.set()

    reader = threading.Thread(target=read_one)
    closer = threading.Thread(target=close_result)
    reader.start()
    assert read_entered.wait(timeout=5)
    closer.start()
    assert close_started.wait(timeout=5)
    try:
        assert not close_finished.wait(timeout=0.05)
    finally:
        release_read.set()
        reader.join(timeout=5)
        closer.join(timeout=5)

    assert not reader.is_alive()
    assert not closer.is_alive()
    assert not errors
    assert len(values) == 1 and math.isfinite(values[0])
    assert result.closed


@pytest.mark.parametrize(
    ("domain_counts", "identifier_bytes"),
    (
        ((5_000, 5_000, 5_000, 5_000), 4_096),
        ((104_000,) * 7, 1),
    ),
)
def test_compiled_preflight_has_no_unmirrored_row_key_or_membership_cap(
    domain_counts: tuple[int, ...],
    identifier_bytes: int,
    compiled_native_prepared: tuple[Path, str],
) -> None:
    row_count = sum(domain_counts)
    row_key_bytes = row_count * (2 + identifier_bytes + 32)
    if len(domain_counts) == 4:
        assert row_key_bytes > 64 * 1024 * 1024
    else:
        assert 22 * row_count > 16_000_000
    executable, _ = compiled_native_prepared
    _assert_truncated_store_reached_authenticated_scan(
        executable,
        domain_counts,
        identifier_bytes=identifier_bytes,
    )


def test_compiled_c_rejects_authenticated_ascii_whitespace_row_id(
    compiled_native_prepared: tuple[Path, str],
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    executable, binary_digest = compiled_native_prepared
    _, _, _, store_path, receipt = prepared_fixture
    with authenticate_prepared_store_file(store_path, receipt) as authenticated:
        metadata = authenticated.metadata
    expected = preflight_native_prepared_session(metadata, ridge=1.0)
    store = bytearray(store_path.read_bytes())
    (identifier_bytes,) = struct.unpack_from("<H", store, 472)
    store[474 : 474 + identifier_bytes] = b" " * identifier_bytes
    store[328:360] = hashlib.sha256(store[472:]).digest()
    store_digest = hashlib.sha256(store).digest()
    request = struct.pack(
        "<8sII32s32s32sQQd24s",
        b"TRPSES01",
        1,
        0,
        b"\x99" * 32,
        store_digest,
        bytes.fromhex(binary_digest),
        160 + len(store),
        expected.result_bytes,
        1.0,
        b"\0" * 24,
    )
    with tempfile.TemporaryFile() as request_stream:
        request_stream.write(request + store)
        request_stream.seek(0)
        completed = subprocess.run(
            [str(executable)],
            stdin=request_stream,
            check=False,
            capture_output=True,
            timeout=10,
        )

    assert completed.returncode == 1
    assert len(completed.stdout) == 448
    assert struct.unpack_from("<8sII", completed.stdout) == (b"TRPRES01", 1, 1)
    assert struct.unpack_from("<4Q", completed.stdout, 200) == (
        expected.statistics_work_units,
        expected.solve_work_units,
        expected.score_work_units,
        metadata.estimate.modeled_c_heap_bytes,
    )


def _write_fake_child(tmp_path: Path, mode: str) -> Path:
    program = f"""#!{sys.executable}
import itertools
import os
import struct
import sys
import time

MODE = {mode!r}
if MODE == "timeout":
    time.sleep(10)
    raise SystemExit(99)
if MODE == "crash":
    sys.stderr.write("deliberate crash")
    raise SystemExit(17)

request = sys.stdin.buffer.read()
header = struct.unpack_from("<8sII32s32s32sQQd24s", request)
nonce, store_sha, binary_sha = header[3:6]
ridge = header[8]
store = request[160:]
D = struct.unpack_from("<Q", store, 32)[0]
N = struct.unpack_from("<Q", store, 40)[0]
d = struct.unpack_from("<Q", store, 48)[0]
M = struct.unpack_from("<Q", store, 56)[0]
file_bytes = struct.unpack_from("<Q", store, 24)[0]
row_key_bytes = struct.unpack_from("<Q", store, 80)[0]
domain_offset = struct.unpack_from("<Q", store, 88)[0]
domain_bytes = struct.unpack_from("<Q", store, 96)[0]
feature_offset = struct.unpack_from("<Q", store, 104)[0]
feature_bytes = struct.unpack_from("<Q", store, 112)[0]
target_bytes = struct.unpack_from("<Q", store, 128)[0]
counts = struct.unpack_from("<7Q", store, 360)[:D]
masks = struct.unpack_from("<7Q", store, 416)[:D]
full_mask = (1 << D) - 1
coefficients = []
scores = []
statistics = 3*N*(d+M) + N*d*(d+1)//2 + N*d*M
solve = 0
score_work = 0
output_cells = 0
weight_cells = 0
maximum_width = 0
for omitted_count in (3, 2, 1):
    for omitted in itertools.combinations(range(D), omitted_count):
        mask = full_mask ^ sum(1 << index for index in omitted)
        rows = sum(counts[index] for index in range(D) if mask & (1 << index))
        tag_mask = 0
        for index in range(D):
            if mask & (1 << index):
                tag_mask |= masks[index]
        width = d - 7 + tag_mask.bit_count()
        maximum_width = max(maximum_width, width)
        weight_cells += M*width
        payload = struct.pack("<6d", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        payload += bytes(8 * (M + M*width))
        coefficient_header = struct.pack(
            "<IIQQQQQ", len(coefficients), 0, mask, rows,
            tag_mask, width, len(payload),
        )
        coefficients.append(coefficient_header + payload)
        output_cells += len(payload)//8
        solve += width**3 + 2*M*width**2 + M*width
        for domain in omitted:
            block_rows = counts[domain]
            score_payload = bytes(8 * block_rows * M)
            score_header = struct.pack(
                "<IIIIQQ", len(scores), len(coefficients)-1, domain,
                0, block_rows, len(score_payload),
            )
            scores.append(score_header + score_payload)
            output_cells += block_rows*M
            score_work += block_rows*M*width
coeff = b"".join(coefficients)
score = b"".join(scores)
memberships = sum(
    counts[domain]
    for subset in range(len(coefficients))
    for domain in range(D)
    if not (struct.unpack_from("<Q", coefficients[subset], 8)[0] & (1 << domain))
)
packed = d*(d+1)//2
heap_cells = (
    N*M + D*(d+M+packed+d*M) + len(coefficients)*(6+M)
    + weight_cells + memberships*M + 3*d + 2*M + packed + d*M
    + 2*maximum_width**2 + M*maximum_width
)
heap = 8*heap_cells + N + 73728
padding = feature_offset - domain_offset - domain_bytes
scan = file_bytes + row_key_bytes + N + padding + target_bytes + 2*feature_bytes
result_bytes = 448 + len(coeff) + len(score)
result_header = struct.pack(
    "<8sII32s32s32sQQQQQQQQQQdQQQQ32s32s32s32s32s32sQQQ",
    b"TRPRES01", 1, 0, nonce, store_sha, binary_sha,
    D, N, d, M, len(coefficients), len(scores), memberships,
    len(coeff), len(score), result_bytes, ridge, statistics, solve,
    score_work, heap, store[328:360], store[200:232], store[168:200],
    store[232:264], store[296:328], store[136:168], scan, output_cells,
    file_bytes,
)
output = bytearray(result_header + coeff + score)
returncode = 0
if MODE == "wrong_magic": output[0] ^= 1
elif MODE == "wrong_nonce": output[16] ^= 1
elif MODE == "wrong_store": output[48] ^= 1
elif MODE == "wrong_binary": output[80] ^= 1
elif MODE == "wrong_shape": struct.pack_into("<Q", output, 112, D + 1)
elif MODE == "wrong_ridge": struct.pack_into("<d", output, 192, ridge + 1.0)
elif MODE == "wrong_graph": output[392] ^= 1
elif MODE == "wrong_scan": struct.pack_into("<Q", output, 424, scan + 1)
elif MODE == "wrong_coefficient_order": struct.pack_into("<I", output, 448, 1)
elif MODE == "wrong_score_order": struct.pack_into("<I", output, 448 + len(coeff), 1)
elif MODE == "nan_payload": struct.pack_into("<d", output, 448 + 48, float("nan"))
elif MODE == "negative_zero": struct.pack_into("<d", output, 448 + len(coeff) + 32, -0.0)
elif MODE == "truncated": output = output[:-1]
elif MODE == "overlong": output += b"x"
elif MODE == "direct_sparse":
    with open("result.bin", "r+b") as direct_result:
        direct_result.truncate(header[7] + 4096)
elif MODE == "success_nonzero": returncode = 9
elif MODE == "error_zero":
    struct.pack_into("<I", output, 12, 5)
    struct.pack_into("<Q", output, 184, 448)
    for offset in (200, 208, 216, 224, 424, 432, 440):
        struct.pack_into("<Q", output, offset, 0)
    output = output[:448]
elif MODE == "structured_error":
    struct.pack_into("<I", output, 12, 5)
    struct.pack_into("<Q", output, 184, 448)
    for offset in (200, 208, 216, 224, 424, 432, 440):
        struct.pack_into("<Q", output, offset, 0)
    output = output[:448]
    returncode = 1
sys.stdout.buffer.write(output)
raise SystemExit(returncode)
"""
    path = tmp_path / f"fake-{mode}"
    path.write_text(program, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return path


def _fake_adapter(path: Path, *, timeout: float = 2.0) -> NativePreparedSessionAdapter:
    return NativePreparedSessionAdapter(path, _sha256(path), timeout_seconds=timeout)


@_FAKE_ONLY
def test_valid_fake_child_proves_request_lineage_and_mmap_reader(
    tmp_path: Path,
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    _, _, _, store_path, receipt = prepared_fixture
    child = _write_fake_child(tmp_path, "valid")

    with _fake_adapter(child).run(store_path, receipt, ridge=0.75) as result:
        assert result.binary_sha256 == _sha256(child)
        assert result.store_sha256 == receipt.whole_file_sha256
        assert result.ridge == 0.75
        assert len(result.coefficients) == 14
        assert len(result.scores) == 28
        assert result.coefficients[0].continuous_scales[:] == (1.0, 1.0, 1.0)
        assert result.scores[0].scores.at(0, 0) == 0.0


@_FAKE_ONLY
@pytest.mark.parametrize(
    "mode",
    (
        "wrong_magic",
        "wrong_nonce",
        "wrong_store",
        "wrong_binary",
        "wrong_shape",
        "wrong_ridge",
        "wrong_graph",
        "wrong_scan",
        "wrong_coefficient_order",
        "wrong_score_order",
        "nan_payload",
        "negative_zero",
        "truncated",
        "overlong",
        "direct_sparse",
        "success_nonzero",
        "error_zero",
    ),
)
def test_malformed_children_fail_closed(
    mode: str,
    tmp_path: Path,
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    _, _, _, store_path, receipt = prepared_fixture
    child = _write_fake_child(tmp_path, mode)

    with pytest.raises(NativePreparedProtocolError):
        _fake_adapter(child).run(store_path, receipt, ridge=1.0)


@_FAKE_ONLY
def test_structured_failure_status_is_distinct_from_malformed_output(
    tmp_path: Path,
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    _, _, _, store_path, receipt = prepared_fixture
    child = _write_fake_child(tmp_path, "structured_error")

    with pytest.raises(NativePreparedStatusError) as captured:
        _fake_adapter(child).run(store_path, receipt, ridge=1.0)

    assert captured.value.status == 5


@_FAKE_ONLY
@pytest.mark.parametrize("mode", ("timeout", "crash"))
def test_timeout_and_crash_are_execution_failures(
    mode: str,
    tmp_path: Path,
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    _, _, _, store_path, receipt = prepared_fixture
    child = _write_fake_child(tmp_path, mode)

    with pytest.raises(NativePreparedExecutionError):
        _fake_adapter(child, timeout=0.1).run(store_path, receipt, ridge=1.0)


@_FAKE_ONLY
def test_wrong_binary_and_store_credentials_never_launch_child(
    tmp_path: Path,
    prepared_fixture: tuple[
        PreparedFeatureStore,
        PreparedCoefficientBundle,
        PreparedRawScoreBundle,
        Path,
        PreparedStoreFileReceipt,
    ],
) -> None:
    _, _, _, store_path, receipt = prepared_fixture
    marker = tmp_path / "launched"
    child = tmp_path / "credential-child"
    child.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    child.chmod(0o700)
    with pytest.raises(NativePreparedIntegrityError, match="binary SHA-256"):
        NativePreparedSessionAdapter(child, "0" * 64).run(store_path, receipt, ridge=1.0)
    assert not marker.exists()

    valid_digest = _sha256(child)
    wrong_receipt = replace(receipt, whole_file_sha256="0" * 64)
    with pytest.raises(NativePreparedIntegrityError, match="prepared store"):
        NativePreparedSessionAdapter(child, valid_digest).run(
            store_path,
            wrong_receipt,
            ridge=1.0,
        )
    assert not marker.exists()


def test_official_d7_shape_passes_aggregate_preflight_without_materialization() -> None:
    counts = (4969, 4968, 4968, 4968, 4968, 4968, 4969)
    masks = (127,) * 7
    feature_count = 1036
    target_count = 11
    row_key_bytes = sum(2 + len(f"routerbench-{index:05d}") + 32 for index in range(sum(counts)))
    domain_offset = 472 + row_key_bytes
    feature_offset = (domain_offset + sum(counts) + 7) & ~7
    feature_bytes = 8 * sum(counts) * feature_count
    target_offset = feature_offset + feature_bytes
    target_bytes = 8 * sum(counts) * target_count
    file_bytes = target_offset + target_bytes
    estimate = estimate_prepared_session(
        domain_row_counts=counts,
        domain_active_tag_masks=masks,
        feature_count=feature_count,
        target_count=target_count,
        store_file_bytes=file_bytes,
        row_key_bytes=row_key_bytes,
    )
    digest = "1" * 64
    metadata = PreparedStoreFileMetadata(
        file_bytes=file_bytes,
        domain_count=7,
        row_count=sum(counts),
        feature_count=feature_count,
        target_count=target_count,
        row_key_offset=472,
        row_key_bytes=row_key_bytes,
        domain_index_offset=domain_offset,
        domain_index_bytes=sum(counts),
        feature_offset=feature_offset,
        feature_bytes=feature_bytes,
        target_offset=target_offset,
        target_bytes=target_bytes,
        graph_identity_sha256=digest,
        source_fit_sha256=digest,
        logical_store_sha256=digest,
        embedding_snapshot_sha256=digest,
        embedding_identity_sha256=digest,
        model_catalogue_sha256=digest,
        store_payload_sha256=digest,
        domain_row_counts=counts,
        domain_active_tag_masks=masks,
        estimate=estimate,
    )

    graph = preflight_native_prepared_session(metadata, ridge=1.0)

    assert len(graph.coefficients) == 63
    assert len(graph.scores) == 154
    assert graph.result_bytes <= MAX_RESULT_FILE_BYTES
    assert estimate.modeled_c_heap_bytes <= MAX_MODELED_C_HEAP_BYTES


def test_compiled_native_completes_1024_embedding_plus_surface_without_projection(
    compiled_native_prepared: tuple[Path, str],
    tmp_path: Path,
) -> None:
    # This bounded corpus uses the section writer so no 1,024-dimensional
    # embedding projection or Python reference solve can hide native cost. It is
    # still not the full official-shape benchmark.
    from tierroute.predictors.prepared_files import write_prepared_store_file_from_sections

    row_count = 8
    ids = tuple(f"wide-{index}" for index in range(row_count))
    domains = tuple(index % 4 for index in range(row_count))
    features = tuple(
        tuple(
            (
                float(index + 1),
                float(index % 3),
                float(index % 5),
                0.0,
                0.0,
                *([1.0] * 7),
                *(math.sin((index + 1) * (column + 1) * 0.001) for column in range(1024)),
            )
        )
        for index in range(row_count)
    )
    targets = tuple((0.25 * index,) for index in range(row_count))
    plan = build_prepared_nested_lodo_plan(
        ("alpha", "bravo", "charlie", "delta"),
        (2, 2, 2, 2),
        feature_count=1036,
        target_count=1,
    )
    embedding_identity = EmbeddingIdentity(
        provider="synthetic-test",
        model_id="wide-fixture",
        revision="fixed",
        pooling="mean",
        normalize=False,
        asset_manifest_sha256="7" * 64,
    )
    path = tmp_path / "wide.trpsto"
    receipt = write_prepared_store_file_from_sections(
        destination=path,
        plan=plan,
        model_ids=("model",),
        example_ids=ids,
        prompt_sha256s=tuple(hashlib.sha256(value.encode()).hexdigest() for value in ids),
        domain_indices=domains,
        feature_rows=features,
        target_rows=targets,
        embedding_identity=embedding_identity,
        embedding_snapshot_sha256="4" * 64,
        expected_source_fit_sha256="2" * 64,
        logical_store_sha256="3" * 64,
    )
    executable, digest = compiled_native_prepared

    assert _sha256(_NATIVE_SOURCE) != "0" * 64
    assert _sha256(executable) == digest
    assert _sha256(path) == receipt.whole_file_sha256
    started = time.perf_counter()
    with NativePreparedSessionAdapter(executable, digest, timeout_seconds=600).run(
        path,
        receipt,
        ridge=1.0,
    ) as result:
        assert len(result.coefficients) == 14
        assert len(result.scores) == 28
        assert all(record.active_feature_count == 1036 for record in result.coefficients)
        assert all(math.isfinite(value) for value in result.coefficients[0].intercepts)
        assert len(result.result_sha256) == 64
        assert result.result_sha256 != "0" * 64
        result_sha256 = result.result_sha256
    wall_seconds = time.perf_counter() - started
    compiler = _available_compiler()
    assert compiler is not None
    version_command = [compiler] if os.name == "nt" else [compiler, "--version"]
    version = subprocess.run(
        version_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version_line = (version.stdout + version.stderr).strip().splitlines()[0]
    pytest_parent_maximum_rss_bytes: int | None = None
    if os.name != "nt":
        import resource

        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pytest_parent_maximum_rss_bytes = int(
            raw_rss if sys.platform == "darwin" else raw_rss * 1024
        )
    print(
        "TIERROUTE_NATIVE_PREPARED_RECEIPT="
        + json.dumps(
            {
                "binary_sha256": digest,
                "compiler": compiler,
                "compiler_version": version_line,
                "fixture": "D4/N8/d1036/M1; 1024 embedding plus 12 surface",
                "platform": platform.platform(),
                "pytest_parent_maximum_rss_bytes": pytest_parent_maximum_rss_bytes,
                "result_sha256": result_sha256,
                "source_sha256": _sha256(_NATIVE_SOURCE),
                "store_sha256": receipt.whole_file_sha256,
                "wall_seconds": round(wall_seconds, 6),
            },
            sort_keys=True,
        )
    )
