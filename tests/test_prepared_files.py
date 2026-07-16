# SPDX-License-Identifier: Apache-2.0
"""Security, parity, and resource tests for file-backed prepared stores."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import struct
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import tierroute.predictors.prepared_files as prepared_files_module
from tierroute.core import ModelSpec
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.predictors.prepared_files import (
    MAX_MODELED_C_HEAP_BYTES,
    MAX_RESULT_FILE_BYTES,
    MAX_STORE_FILE_BYTES,
    MAX_TOTAL_NUMERIC_WORK_UNITS,
    PreparedStoreFileError,
    authenticate_prepared_store_file,
    copy_authenticated_prepared_store,
    estimate_prepared_session,
    write_prepared_store_file,
    write_prepared_store_file_from_sections,
)
from tierroute.predictors.prepared_graph import build_prepared_nested_lodo_plan
from tierroute.predictors.prepared_store import (
    PreparedEmbeddingInput,
    PreparedFeatureStore,
    build_prepared_embedding_snapshot,
    build_prepared_feature_store,
    prepared_fit_source_sha256,
)

_MODEL_IDS = ("cheap", "premium")
_ROWS = (
    ("r-001", "alpha", "Write Python code for an API."),
    ("r-002", "alpha", "Debug this Rust function."),
    ("r-003", "bravo", "Prove the equation x + 1 = 2."),
    ("r-004", "bravo", "수학 확률 문제를 증명하라."),
    ("r-005", "charlie", "Review this legal contract and statute."),
    ("r-006", "charlie", "Summarize the court precedent."),
    ("r-007", "delta", "Assess this clinical medical diagnosis."),
    ("r-008x", "delta", "Explain medicine options for a patient."),
)


def _examples() -> tuple[EvaluationExample, ...]:
    examples = []
    for ordinal, (example_id, domain, prompt) in enumerate(_ROWS, start=1):
        models = (
            ModelSpec("premium", Decimal("5")),
            ModelSpec("cheap", Decimal("1")),
        )
        outcomes = tuple(
            CandidateOutcome(
                model_id=model_id,
                output=f"{example_id}:{model_id}",
                cost=Decimal("1") if model_id == "cheap" else Decimal("5"),
                quality=(0.2 if model_id == "cheap" else 0.6) + ordinal / 100,
            )
            for model_id in _MODEL_IDS
        )
        examples.append(
            EvaluationExample(
                example_id=example_id,
                prompt=prompt,
                domain=domain,
                outcomes=outcomes,
                candidate_models=models,
            )
        )
    return tuple(reversed(examples))


def _store() -> PreparedFeatureStore:
    examples = _examples()
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=12,
        target_count=len(_MODEL_IDS),
    )
    return build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )


def _embedded_store() -> PreparedFeatureStore:
    examples = _examples()
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=13,
        target_count=len(_MODEL_IDS),
    )
    identity = EmbeddingIdentity(
        provider="fixture-provider",
        model_id="fixture-model",
        revision="fixed-revision",
        pooling="mean",
        normalize=True,
        asset_manifest_sha256="a" * 64,
    )
    snapshot = build_prepared_embedding_snapshot(
        tuple(
            PreparedEmbeddingInput(
                example_id=example.example_id,
                prompt_sha256=hashlib.sha256(example.prompt.encode("utf-8")).hexdigest(),
                values=(index / 10,),
            )
            for index, example in enumerate(examples, start=1)
        ),
        identity,
        dimension=1,
    )
    return build_prepared_feature_store(
        examples,
        plan,
        embedding_snapshot=snapshot,
        expected_embedding_sha256=snapshot.sha256,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )


def _write_store(path: Path) -> tuple[PreparedFeatureStore, object]:
    store = _store()
    receipt = write_prepared_store_file(store, path)
    return store, receipt


def _private_output(path: Path, prefix: bytes = b"") -> object:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    stream = os.fdopen(descriptor, "w+b")
    stream.write(prefix)
    stream.flush()
    return stream


def _mutate(path: Path, offset: int, payload: bytes) -> None:
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def test_store_round_trip_copy_and_streaming_writer_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    store, receipt = _write_store(first)
    with authenticate_prepared_store_file(first, receipt) as authenticated:
        metadata = authenticated.metadata
        assert authenticated.mapping[:8] == b"TRPSTO01"
        assert metadata.domain_count == 4
        assert metadata.row_count == 8
        assert metadata.feature_count == 12
        assert metadata.target_count == 2
        assert metadata.file_bytes == first.stat().st_size
        assert metadata.logical_store_sha256 == store.sha256
        assert metadata.source_fit_sha256 == store.source_fit_sha256
        assert metadata.estimate.training_subset_count == 14
        assert metadata.estimate.score_record_count == 28

    copied = tmp_path / "request.bin"
    stream = _private_output(copied, b"R" * 160)
    try:
        copied_metadata = copy_authenticated_prepared_store(first, receipt, stream)
        assert stream.tell() == 160 + copied_metadata.file_bytes
    finally:
        stream.close()
    assert copied.read_bytes()[160:] == first.read_bytes()

    second = tmp_path / "second.bin"
    feature_stride = 8 * store.plan.feature_count
    target_stride = 8 * store.plan.target_count
    second_receipt = write_prepared_store_file_from_sections(
        destination=second,
        plan=store.plan,
        model_ids=store.model_ids,
        example_ids=(value for value in store.example_ids),
        prompt_sha256s=(value for value in store.prompt_sha256s),
        domain_indices=(value for value in store.domain_indices),
        feature_rows=(
            struct.unpack_from(
                f"<{store.plan.feature_count}d",
                store.feature_payload,
                row * feature_stride,
            )
            for row in range(store.plan.work.example_count)
        ),
        target_rows=(
            struct.unpack_from(
                f"<{store.plan.target_count}d",
                store.target_payload,
                row * target_stride,
            )
            for row in range(store.plan.work.example_count)
        ),
        embedding_identity=None,
        embedding_snapshot_sha256=None,
        expected_source_fit_sha256=store.source_fit_sha256,
        logical_store_sha256=store.sha256,
    )
    assert second_receipt == receipt
    assert second.read_bytes() == first.read_bytes()


def test_identity_namespaces_have_golden_digests(tmp_path: Path) -> None:
    path = tmp_path / "store.bin"
    _, receipt = _write_store(path)
    with authenticate_prepared_store_file(path, receipt) as authenticated:
        metadata = authenticated.metadata
    assert (
        metadata.graph_identity_sha256
        == "b4427ffc2cef7e5495dcdcd10aa3a6a9cfde54023c67b90b2babdecafa7dca35"
    )
    assert (
        metadata.model_catalogue_sha256
        == "2af6dfdde36585393216afe97714661abc1af2ffb37069d598b79bed3e55b670"
    )

    embedded_path = tmp_path / "embedded.bin"
    embedded_store = _embedded_store()
    embedded_receipt = write_prepared_store_file(embedded_store, embedded_path)
    with authenticate_prepared_store_file(embedded_path, embedded_receipt) as authenticated:
        embedded_metadata = authenticated.metadata
    assert (
        embedded_metadata.embedding_identity_sha256
        == "287d095c7b757a1bfdbac8e1f3abeabb47e2280b5949d398fa09207d0dcdd9f0"
    )
    assert embedded_metadata.embedding_snapshot_sha256 == embedded_store.embedding_snapshot_sha256


@pytest.mark.parametrize(
    ("offset", "payload", "message"),
    [
        (0, b"BADMAGIC", "magic"),
        (12, struct.pack("<I", 1), "flags"),
        (104, struct.pack("<Q", 999_999), "feature offset"),
        (392, struct.pack("<Q", 1), "unused prepared-store domain counts"),
    ],
)
def test_malformed_fixed_header_is_rejected_before_copy(
    tmp_path: Path,
    offset: int,
    payload: bytes,
    message: str,
) -> None:
    path = tmp_path / "store.bin"
    _, receipt = _write_store(path)
    _mutate(path, offset, payload)
    output = tmp_path / "output.bin"
    stream = _private_output(output, b"prefix")
    try:
        with pytest.raises(PreparedStoreFileError, match=message):
            copy_authenticated_prepared_store(path, receipt, stream)
        assert stream.tell() == len(b"prefix")
        assert output.read_bytes() == b"prefix"
    finally:
        stream.close()


def test_bad_external_and_embedded_digests_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "store.bin"
    _, receipt = _write_store(path)
    wrong = replace(receipt, whole_file_sha256="0" * 64)
    with pytest.raises(PreparedStoreFileError, match="whole-file"):
        authenticate_prepared_store_file(path, wrong)
    wrong_source = replace(receipt, source_fit_sha256="1" * 64)
    with pytest.raises(PreparedStoreFileError, match="source-fit"):
        authenticate_prepared_store_file(path, wrong_source)
    wrong_logical = replace(receipt, logical_store_sha256="2" * 64)
    with pytest.raises(PreparedStoreFileError, match="logical"):
        authenticate_prepared_store_file(path, wrong_logical)

    _mutate(path, 328, b"\x01" * 32)
    changed = replace(
        receipt,
        whole_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(PreparedStoreFileError, match="payload SHA-256"):
        authenticate_prepared_store_file(path, changed)


def test_padding_and_trailing_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "store.bin"
    _, receipt = _write_store(path)
    with authenticate_prepared_store_file(path, receipt) as authenticated:
        metadata = authenticated.metadata
    padding_offset = metadata.domain_index_offset + metadata.domain_index_bytes
    assert metadata.feature_offset > padding_offset
    _mutate(path, padding_offset, b"\x01")
    with pytest.raises(PreparedStoreFileError, match="padding"):
        authenticate_prepared_store_file(path, receipt)

    trailing = tmp_path / "trailing.bin"
    _, trailing_receipt = _write_store(trailing)
    with trailing.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(PreparedStoreFileError, match="file length"):
        authenticate_prepared_store_file(trailing, trailing_receipt)


@pytest.mark.parametrize(
    ("kind", "payload", "message"),
    [
        ("feature", struct.pack("<d", float("nan")), "finite binary64"),
        ("feature", struct.pack("<d", -0.0), "positive zero"),
        ("domain", b"\xff", "domain index"),
        ("target", struct.pack("<d", float("inf")), "finite binary64"),
    ],
)
def test_malformed_sections_are_rejected(
    tmp_path: Path,
    kind: str,
    payload: bytes,
    message: str,
) -> None:
    path = tmp_path / "store.bin"
    _, receipt = _write_store(path)
    with authenticate_prepared_store_file(path, receipt) as authenticated:
        metadata = authenticated.metadata
    offsets = {
        "feature": metadata.feature_offset,
        "domain": metadata.domain_index_offset,
        "target": metadata.target_offset,
    }
    _mutate(path, offsets[kind], payload)
    with pytest.raises(PreparedStoreFileError, match=message):
        authenticate_prepared_store_file(path, receipt)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (0, -1.0, "continuous features"),
        (3, 0.5, "binary/tag features"),
        (5, 0.5, "binary/tag features"),
    ],
)
def test_surface_feature_value_contract_is_rechecked_from_file(
    tmp_path: Path,
    column: int,
    value: float,
    message: str,
) -> None:
    path = tmp_path / "store.bin"
    _, receipt = _write_store(path)
    with authenticate_prepared_store_file(path, receipt) as authenticated:
        feature_offset = authenticated.metadata.feature_offset
    _mutate(path, feature_offset + 8 * column, struct.pack("<d", value))
    with pytest.raises(PreparedStoreFileError, match=message):
        authenticate_prepared_store_file(path, receipt)


def test_row_keys_require_strict_utf8_and_python_order(tmp_path: Path) -> None:
    invalid_utf8 = tmp_path / "invalid-utf8.bin"
    _, receipt = _write_store(invalid_utf8)
    _mutate(invalid_utf8, 474, b"\xff")
    with pytest.raises(PreparedStoreFileError, match="strict UTF-8"):
        authenticate_prepared_store_file(invalid_utf8, receipt)

    whitespace = tmp_path / "whitespace.bin"
    _, whitespace_receipt = _write_store(whitespace)
    _mutate(whitespace, 474, b"     ")
    with pytest.raises(PreparedStoreFileError, match="non-ASCII-whitespace"):
        authenticate_prepared_store_file(whitespace, whitespace_receipt)

    duplicate = tmp_path / "duplicate.bin"
    _, duplicate_receipt = _write_store(duplicate)
    first_record_bytes = 2 + len("r-001") + 32
    second_identifier_offset = 472 + first_record_bytes + 2
    _mutate(duplicate, second_identifier_offset, b"r-001")
    with pytest.raises(PreparedStoreFileError, match="strictly increasing"):
        authenticate_prepared_store_file(duplicate, duplicate_receipt)


def test_header_tag_mask_and_domain_count_must_match_sections(tmp_path: Path) -> None:
    path = tmp_path / "tag.bin"
    _, receipt = _write_store(path)
    with authenticate_prepared_store_file(path, receipt) as authenticated:
        original_mask = authenticated.metadata.domain_active_tag_masks[0]
    changed_mask = original_mask ^ 1
    _mutate(path, 416, struct.pack("<Q", changed_mask))
    with pytest.raises(PreparedStoreFileError, match="feature tags"):
        authenticate_prepared_store_file(path, receipt)

    count_path = tmp_path / "count.bin"
    _, count_receipt = _write_store(count_path)
    _mutate(count_path, 360, struct.pack("<Q", 3))
    with pytest.raises(PreparedStoreFileError, match="sum to N"):
        authenticate_prepared_store_file(count_path, count_receipt)


def test_symlink_and_path_replacement_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "store.bin"
    _, receipt = _write_store(source)
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(source)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(PreparedStoreFileError, match="symlink"):
        authenticate_prepared_store_file(link, receipt)

    if os.name == "nt":
        pytest.skip(
            "Windows denies replacing an open source handle; the path-replacement "
            "race requires POSIX rename semantics"
        )

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(source.read_bytes())
    original_scan = prepared_files_module._scan_store_descriptor

    def replacing_scan(*args: object, **kwargs: object) -> object:
        result = original_scan(*args, **kwargs)  # type: ignore[arg-type]
        os.replace(replacement, source)
        return result

    monkeypatch.setattr(prepared_files_module, "_scan_store_descriptor", replacing_scan)
    with pytest.raises(PreparedStoreFileError, match="changed during authentication"):
        authenticate_prepared_store_file(source, receipt)


def test_cross_interface_stability_can_omit_only_nonportable_change_time() -> None:
    first = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=stat.S_IFREG,
        st_size=23,
        st_mtime_ns=101,
        st_ctime_ns=202,
    )
    changed_creation_time = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=stat.S_IFREG,
        st_size=23,
        st_mtime_ns=101,
        st_ctime_ns=303,
    )

    prepared_files_module._require_stable_source(
        first,
        changed_creation_time,
        compare_change_time=False,
    )
    with pytest.raises(PreparedStoreFileError, match="contents changed"):
        prepared_files_module._require_stable_source(first, changed_creation_time)


@pytest.mark.parametrize("changed_field", ("st_size", "st_mtime_ns"))
def test_cross_interface_stability_keeps_content_metadata_guards(changed_field: str) -> None:
    first_values = {
        "st_dev": 1,
        "st_ino": 2,
        "st_mode": stat.S_IFREG,
        "st_size": 23,
        "st_mtime_ns": 101,
        "st_ctime_ns": 202,
    }
    second_values = first_values | {changed_field: first_values[changed_field] + 1}

    with pytest.raises(PreparedStoreFileError, match="contents changed"):
        prepared_files_module._require_stable_source(
            SimpleNamespace(**first_values),
            SimpleNamespace(**second_values),
            compare_change_time=False,
        )


def test_private_snapshot_survives_source_mutation_and_close_lifetime(tmp_path: Path) -> None:
    source = tmp_path / "store.bin"
    _, receipt = _write_store(source)
    authenticated = authenticate_prepared_store_file(source, receipt)
    mapping = authenticated.mapping
    original_prefix = mapping[:16]
    _mutate(source, 0, b"BROKEN!!")
    assert mapping[:16] == original_prefix
    assert mapping[:8] == b"TRPSTO01"

    exported = memoryview(mapping)
    with pytest.raises(PreparedStoreFileError, match=r"cannot close.*mapping"):
        authenticated.close()
    assert not authenticated.closed
    exported.release()
    authenticated.close()
    assert authenticated.closed
    with pytest.raises(ValueError, match="closed"):
        _ = mapping[0]


def test_official_shape_admission_matches_closed_form() -> None:
    counts = (4_969, 4_969, 4_969, 4_969, 4_969, 4_969, 4_964)
    row_count = sum(counts)
    row_key_bytes = row_count * (2 + 20 + 32)
    domain_end = 472 + row_key_bytes + row_count
    feature_offset = (domain_end + 7) & ~7
    store_bytes = feature_offset + 8 * row_count * (1_036 + 11)
    estimate = estimate_prepared_session(
        domain_row_counts=counts,
        domain_active_tag_masks=(127,) * 7,
        feature_count=1_036,
        target_count=11,
        store_file_bytes=store_bytes,
        row_key_bytes=row_key_bytes,
    )
    assert row_count == 34_778
    assert estimate.training_subset_count == 63
    assert estimate.score_record_count == 154
    assert estimate.score_row_memberships == 22 * row_count
    assert len(estimate.active_feature_counts) == 63
    assert set(estimate.active_feature_counts) == {1_036}
    assert estimate.result_bytes == 448 + estimate.coefficient_bytes + estimate.score_bytes
    assert estimate.mapped_input_bytes == store_bytes
    assert estimate.file_backed_input_bytes == store_bytes
    expected_scan = (
        store_bytes
        + row_key_bytes
        + row_count
        + ((-domain_end) % 8)
        + 8 * row_count * 11
        + 2 * 8 * row_count * 1_036
    )
    assert estimate.authentication_validation_bytes_scanned == expected_scan
    assert estimate.output_numeric_cells_validated == (
        63 * (6 + 11 + 11 * 1_036) + 22 * row_count * 11
    )
    assert estimate.modeled_c_heap_bytes == 128_706_874
    assert estimate.authentication_validation_bytes_scanned == 874_667_176
    assert estimate.output_numeric_cells_validated == 9_135_295
    assert estimate.coefficient_bytes == 5_755_176
    assert estimate.score_bytes == 67_335_136
    assert estimate.result_bytes == 73_090_760
    assert estimate.statistics_work_units == 19_187_126_934
    assert estimate.solve_work_units == 71_540_189_532
    assert estimate.score_work_units == 8_719_261_936
    assert estimate.total_numeric_work_units == 99_446_578_402
    assert estimate.result_bytes < MAX_RESULT_FILE_BYTES
    assert estimate.mapped_input_bytes < MAX_STORE_FILE_BYTES
    assert estimate.modeled_c_heap_bytes < MAX_MODELED_C_HEAP_BYTES
    assert estimate.total_numeric_work_units < MAX_TOTAL_NUMERIC_WORK_UNITS
    assert math.comb(7, 3) + math.comb(7, 2) + 7 == 63


def test_writer_rejects_existing_or_nonabsolute_destination(tmp_path: Path) -> None:
    store = _store()
    existing = tmp_path / "existing.bin"
    existing.write_bytes(b"owned")
    with pytest.raises(PreparedStoreFileError, match="new absent"):
        write_prepared_store_file(store, existing)
    assert existing.read_bytes() == b"owned"
    with pytest.raises(PreparedStoreFileError, match="absolute"):
        write_prepared_store_file(store, Path("relative.bin"))


def test_stream_writer_stops_before_row_keys_can_exceed_file_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepared_files_module, "MAX_STORE_FILE_BYTES", 10_000)
    plan = build_prepared_nested_lodo_plan(
        ("alpha", "bravo", "charlie", "delta"),
        (10, 10, 10, 10),
        feature_count=12,
        target_count=1,
    )
    consumed = 0

    def example_ids() -> object:
        nonlocal consumed
        for index in range(40):
            consumed += 1
            yield f"{index:03d}-" + "x" * 396

    def forbidden_rows() -> object:
        raise AssertionError("numeric iterables must not be consumed after row-key refusal")
        yield 0

    destination = tmp_path / "bounded.trpstore"
    with pytest.raises(PreparedStoreFileError, match="row-key section cannot fit"):
        write_prepared_store_file_from_sections(
            destination=destination,
            plan=plan,
            model_ids=("model",),
            example_ids=example_ids(),
            prompt_sha256s=("a" * 64 for _ in range(40)),
            domain_indices=forbidden_rows(),
            feature_rows=forbidden_rows(),
            target_rows=forbidden_rows(),
            embedding_identity=None,
            embedding_snapshot_sha256=None,
            expected_source_fit_sha256="1" * 64,
            logical_store_sha256="2" * 64,
        )

    assert consumed < 40
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".bounded.trpstore.stage.*"))


def test_copy_requires_owner_only_destination(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX mode bits are unavailable on Windows")
    source = tmp_path / "store.bin"
    _, receipt = _write_store(source)
    output = tmp_path / "output.bin"
    output.touch(mode=0o644)
    with output.open("r+b") as stream:
        with pytest.raises(PreparedStoreFileError, match="0600"):
            copy_authenticated_prepared_store(source, receipt, stream)
