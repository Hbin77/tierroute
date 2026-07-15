# SPDX-License-Identifier: Apache-2.0
"""Tests for opt-in RouterBench download and loading boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from tierroute.adapters import routerbench

QUOTED_COSTS = {"model-a": Decimal("0.05")}


def load_download_module() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "download_routerbench.py"
    spec = importlib.util.spec_from_file_location("download_routerbench", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    """Small context-managed byte stream standing in for an HTTP response."""

    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self._stream.close()


def patch_artifact(module: ModuleType, monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(module, "ROUTERBENCH_SIZE", len(payload))
    monkeypatch.setattr(module, "ROUTERBENCH_SHA256", hashlib.sha256(payload).hexdigest())


def test_download_and_adapter_pin_the_same_upstream_artifact() -> None:
    downloader = load_download_module()

    assert downloader.ROUTERBENCH_REVISION == routerbench.ROUTERBENCH_REVISION
    assert downloader.ROUTERBENCH_URL == routerbench.ROUTERBENCH_URL
    assert downloader.ROUTERBENCH_SIZE == routerbench.ROUTERBENCH_SIZE == 99_567_659
    assert downloader.ROUTERBENCH_SHA256 == routerbench.ROUTERBENCH_SHA256


def test_checksum_accepts_fixture_and_rejects_changed_bytes(tmp_path: Path) -> None:
    downloader = load_download_module()
    fixture = tmp_path / "routerbench.pkl"
    fixture.write_bytes(b"local RouterBench fixture")
    expected = hashlib.sha256(fixture.read_bytes()).hexdigest()

    assert downloader.sha256_file(fixture, chunk_size=3) == expected
    assert downloader.verify_file(
        fixture,
        expected_size=fixture.stat().st_size,
        expected_sha256=expected,
        chunk_size=2,
    )

    fixture.write_bytes(b"changed RouterBench fixture")
    assert not downloader.verify_file(
        fixture,
        expected_size=fixture.stat().st_size,
        expected_sha256=expected,
    )


def test_download_reuses_existing_verified_file_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = load_download_module()
    payload = b"already verified"
    destination = tmp_path / "routerbench_0shot.pkl"
    destination.write_bytes(payload)
    patch_artifact(downloader, monkeypatch, payload)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail("a verified existing file must not trigger a network call")

    monkeypatch.setattr(downloader, "urlopen", fail_if_called)

    assert downloader.download_routerbench(destination, chunk_size=2) == destination
    assert destination.read_bytes() == payload


def test_download_streams_to_part_then_atomically_replaces_invalid_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = load_download_module()
    payload = b"verified replacement"
    destination = tmp_path / "routerbench_0shot.pkl"
    destination.write_bytes(b"old invalid bytes")
    patch_artifact(downloader, monkeypatch, payload)

    def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
        assert request.full_url == downloader.ROUTERBENCH_URL  # type: ignore[attr-defined]
        assert timeout == 5.0
        assert destination.read_bytes() == b"old invalid bytes"
        return FakeResponse(payload)

    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)

    downloader.download_routerbench(destination, timeout=5.0, chunk_size=3)

    assert destination.read_bytes() == payload
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_download_checksum_failure_keeps_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = load_download_module()
    destination = tmp_path / "routerbench_0shot.pkl"
    destination.write_bytes(b"old invalid bytes")
    patch_artifact(downloader, monkeypatch, b"expected payload")
    tampered = b"tampered payload"
    assert len(tampered) == downloader.ROUTERBENCH_SIZE
    monkeypatch.setattr(
        downloader,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(tampered),
    )

    with pytest.raises(downloader.DownloadIntegrityError, match="SHA-256 mismatch"):
        downloader.download_routerbench(destination, chunk_size=4)

    assert destination.read_bytes() == b"old invalid bytes"
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_adapter_rejects_arbitrary_pickle_before_importing_pandas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arbitrary_pickle = tmp_path / "arbitrary.pkl"
    arbitrary_pickle.write_bytes(b"not the pinned file")

    def fail_if_called() -> None:
        pytest.fail("pandas must not be imported for an unauthenticated pickle")

    monkeypatch.setattr(routerbench, "_import_pandas", fail_if_called)

    with pytest.raises(routerbench.RouterBenchIntegrityError, match="size mismatch"):
        routerbench.load_routerbench_dataframe(arbitrary_pickle)


def test_adapter_reports_missing_optional_pandas_after_integrity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "verified.pkl"
    payload = b"verified local fixture"
    fixture.write_bytes(payload)
    monkeypatch.setattr(routerbench, "ROUTERBENCH_SIZE", len(payload))
    monkeypatch.setattr(routerbench, "ROUTERBENCH_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(sys.modules, "pandas", None)

    with pytest.raises(routerbench.RouterBenchDependencyError, match="optional 'pandas'"):
        routerbench.load_routerbench_dataframe(fixture)


class FakeDataFrame:
    """Minimal wide-frame protocol used without installing pandas in CI."""

    def __init__(self, columns: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> None:
        self.columns = columns
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def itertuples(self, *, index: bool, name: None) -> Iterator[tuple[object, ...]]:
        assert index is False
        assert name is None
        return iter(self._rows)


def test_raw_row_iterator_validates_wide_schema_and_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "verified.pkl"
    payload = b"verified local fixture"
    fixture.write_bytes(payload)
    monkeypatch.setattr(routerbench, "ROUTERBENCH_SIZE", len(payload))
    monkeypatch.setattr(routerbench, "ROUTERBENCH_SHA256", hashlib.sha256(payload).hexdigest())
    columns = (
        "sample_id",
        "prompt",
        "eval_name",
        "oracle_model_to_route_to",
        "model-a",
        "model-a|model_response",
        "model-a|total_cost",
    )
    dataframe = FakeDataFrame(
        columns,
        (("gsm8k.1", "What is 1 + 1?", "gsm8k", "model-a", 1.0, "2", 0.001),),
    )
    fake_pandas = ModuleType("pandas")
    fake_pandas.read_pickle = lambda stream: dataframe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas", fake_pandas)

    rows = routerbench.iter_routerbench_rows(fixture)
    row = next(rows)

    assert row["sample_id"] == "gsm8k.1"
    assert routerbench.validate_routerbench_schema(dataframe) == ("model-a",)
    with pytest.raises(TypeError):
        row["sample_id"] = "changed"  # type: ignore[index]


def test_typed_conversion_filters_unmapped_domains_and_keeps_cost_exact() -> None:
    mapped = {
        "sample_id": "gsm8k.1",
        "prompt": "What is 1 + 1?",
        "eval_name": "grade-school-math",
        "oracle_model_to_route_to": "model-a",
        "model-a": 1.0,
        "model-a|model_response": "2",
        "model-a|total_cost": 0.1,
    }
    unmapped = {**mapped, "sample_id": "extra.1", "eval_name": "abstract2title"}

    example = routerbench.routerbench_row_to_example(
        mapped, row_number=0, quoted_costs=QUOTED_COSTS
    )

    assert example is not None
    assert example.domain == "gsm8k"
    assert example.router_metadata == {}
    assert str(example.outcomes[0].cost) == "0.1"
    assert (
        routerbench.routerbench_row_to_example(unmapped, row_number=1, quoted_costs=QUOTED_COSTS)
        is None
    )
    included = routerbench.routerbench_row_to_example(
        unmapped,
        row_number=1,
        quoted_costs=QUOTED_COSTS,
        include_unmapped=True,
    )
    assert included is not None and included.domain == "unmapped:abstract2title"


def test_no_correct_model_oracle_sentinel_is_accepted_but_not_exposed() -> None:
    row = {
        "sample_id": "q1",
        "prompt": "A question no candidate answered.",
        "eval_name": "hellaswag",
        "oracle_model_to_route_to": "no_model_correct",
        "model-a": 0.0,
        "model-a|model_response": "wrong",
        "model-a|total_cost": 0.1,
    }

    example = routerbench.routerbench_row_to_example(row, row_number=0, quoted_costs=QUOTED_COSTS)

    assert example is not None
    assert not hasattr(example, "oracle_model_to_route_to")


def test_quoted_costs_are_fitted_from_separate_rows_not_current_realized_cost() -> None:
    calibration = {
        "sample_id": "train",
        "prompt": "training prompt",
        "eval_name": "hellaswag",
        "oracle_model_to_route_to": "model-a",
        "model-a": 1.0,
        "model-a|model_response": "answer",
        "model-a|total_cost": 0.1,
    }
    evaluation = {**calibration, "sample_id": "test", "model-a|total_cost": 9.9}

    quoted = routerbench.estimate_routerbench_quoted_costs((calibration,))
    example = routerbench.routerbench_row_to_example(evaluation, row_number=0, quoted_costs=quoted)

    assert example is not None
    assert example.candidate_models[0].cost == Decimal("0.1")
    assert example.outcomes[0].cost == Decimal("9.9")


def test_schema_rejects_incomplete_model_triplet() -> None:
    dataframe = FakeDataFrame(
        (
            "sample_id",
            "prompt",
            "eval_name",
            "oracle_model_to_route_to",
            "model-a",
            "model-a|model_response",
        ),
        (("id", "prompt", "domain", "model-a", 1.0, "response"),),
    )

    with pytest.raises(routerbench.RouterBenchSchemaError, match="missing costs"):
        routerbench.validate_routerbench_schema(dataframe)
