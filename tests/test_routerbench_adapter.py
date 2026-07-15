# SPDX-License-Identifier: Apache-2.0
"""Tests for opt-in RouterBench download and loading boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import struct
from collections.abc import Iterator
from decimal import Decimal, localcontext
from pathlib import Path
from types import ModuleType

import pytest

from tierroute.adapters import routerbench

QUOTED_COSTS = {"model-a": Decimal("0.05")}
_DEFAULT_STATE = object()


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


def inert_global(target: tuple[str, str]) -> object:
    return routerbench._GlobalToken(*target)


def construct_node(
    target: tuple[str, str],
    args: tuple[object, ...],
    *,
    operation: str = "REDUCE",
    state: object = _DEFAULT_STATE,
) -> object:
    node = routerbench._ConstructNode(operation, inert_global(target), args)
    if state is not _DEFAULT_STATE:
        node.state = state
    return node


def dtype_node(kind: str) -> object:
    if kind == "object":
        return construct_node(
            routerbench._DTYPE,
            ("O8", False, True),
            state=(3, "|", None, None, None, -1, -1, 63),
        )
    return construct_node(
        routerbench._DTYPE,
        (kind, False, True),
        state=(3, "<", None, None, None, -1, -1, 0),
    )


def object_array(values: tuple[object, ...], shape: tuple[int, ...]) -> object:
    return construct_node(
        routerbench._RECONSTRUCT,
        (inert_global(routerbench._NDARRAY), (0,), b"b"),
        state=(1, shape, dtype_node("object"), False, list(values)),
    )


def block_node(array: object, start: int, stop: int) -> object:
    placement = construct_node(routerbench._SLICE, (start, stop, 1))
    return construct_node(routerbench._UNPICKLE_BLOCK, (array, placement, 2))


def tiny_routerbench_graph(
    *,
    cost_bytes: bytes | None = None,
    cost_dtype: str = "f8",
    row_stop: int = 1,
) -> object:
    columns = (
        "sample_id",
        "prompt",
        "eval_name",
        "model-a",
        "model-a|model_response",
        "model-a|total_cost",
        "oracle_model_to_route_to",
    )
    mixed = object_array(
        ("gsm8k.1", "What is 1 + 1?", "grade-school-math", 1.0, "2"),
        (5, 1),
    )
    cost = construct_node(
        routerbench._FROM_BUFFER,
        (
            bytearray(struct.pack("<d", 0.001) if cost_bytes is None else cost_bytes),
            dtype_node(cost_dtype),
            (1, 1),
            "C",
        ),
    )
    oracle = object_array(("model-a",), (1, 1))
    blocks = (
        block_node(mixed, 0, 5),
        block_node(cost, 5, 6),
        block_node(oracle, 6, 7),
    )
    column_axis = construct_node(
        routerbench._NEW_INDEX,
        (
            inert_global(routerbench._INDEX),
            {"data": object_array(columns, (7,)), "name": None},
        ),
    )
    row_axis = construct_node(
        routerbench._NEW_INDEX,
        (
            inert_global(routerbench._RANGE_INDEX),
            {"name": None, "start": 0, "stop": row_stop, "step": 1},
        ),
    )
    manager = construct_node(routerbench._BLOCK_MANAGER, (blocks, [column_axis, row_axis]))
    return construct_node(
        routerbench._DATAFRAME,
        (),
        operation="NEWOBJ",
        state={
            "_mgr": manager,
            "_typ": "dataframe",
            "_metadata": [],
            "attrs": {},
            "_flags": {"allows_duplicate_labels": True},
        },
    )


def configure_tiny_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    columns = (
        "sample_id",
        "prompt",
        "eval_name",
        "model-a",
        "model-a|model_response",
        "model-a|total_cost",
        "oracle_model_to_route_to",
    )
    monkeypatch.setattr(routerbench, "ROUTERBENCH_ROW_COUNT", 1)
    monkeypatch.setattr(routerbench, "ROUTERBENCH_COLUMNS", columns)
    monkeypatch.setattr(routerbench, "ROUTERBENCH_COLUMN_COUNT", len(columns))
    monkeypatch.setattr(
        routerbench,
        "_EXPECTED_BLOCK_LAYOUT",
        ((0, 5, "object-mixed"), (5, 6, "float64"), (6, 7, "object-string")),
    )


def encode_pickle_value(value: object) -> bytes:
    """Encode only the inert node/primitives used by the tiny protocol-5 fixture."""

    if isinstance(value, routerbench._GlobalToken):
        return (
            encode_pickle_value(value.module)
            + encode_pickle_value(value.name)
            + b"\x93"  # STACK_GLOBAL
        )
    if isinstance(value, routerbench._ConstructNode):
        opcode = {"REDUCE": b"R", "NEWOBJ": b"\x81"}[value.operation]
        encoded = encode_pickle_value(value.target) + encode_pickle_value(value.args) + opcode
        if value.state is not routerbench._UNBUILT:
            encoded += encode_pickle_value(value.state) + b"b"  # BUILD
        return encoded
    if value is None:
        return b"N"
    if value is True:
        return b"\x88"
    if value is False:
        return b"\x89"
    if type(value) is int:
        if 0 <= value <= 0xFF:
            return b"K" + bytes((value,))
        if 0 <= value <= 0xFFFF:
            return b"M" + struct.pack("<H", value)
        return b"J" + struct.pack("<i", value)
    if type(value) is float:
        return b"G" + struct.pack(">d", value)
    if type(value) is str:
        encoded = value.encode("utf-8")
        if len(encoded) <= 0xFF:
            return b"\x8c" + bytes((len(encoded),)) + encoded
        return b"X" + struct.pack("<I", len(encoded)) + encoded
    if type(value) is bytes:
        if len(value) > 0xFF:
            raise ValueError("tiny fixture bytes must fit SHORT_BINBYTES")
        return b"C" + bytes((len(value),)) + value
    if type(value) is bytearray:
        return b"\x96" + struct.pack("<Q", len(value)) + value
    if type(value) is tuple:
        if not value:
            return b")"
        encoded_items = b"".join(encode_pickle_value(item) for item in value)
        if len(value) <= 3:
            return encoded_items + {1: b"\x85", 2: b"\x86", 3: b"\x87"}[len(value)]
        return b"(" + encoded_items + b"t"
    if type(value) is list:
        if not value:
            return b"]"
        return b"](" + b"".join(encode_pickle_value(item) for item in value) + b"e"
    if type(value) is dict:
        if not value:
            return b"}"
        encoded_items = b"".join(
            encode_pickle_value(key) + encode_pickle_value(item) for key, item in value.items()
        )
        if len(value) == 1:
            return b"}" + encoded_items + b"s"
        return b"}(" + encoded_items + b"u"
    raise TypeError(f"unsupported tiny fixture value: {type(value).__name__}")


def framed_protocol5_body(body: bytes) -> bytes:
    return b"\x80\x05\x95" + struct.pack("<Q", len(body)) + body


def framed_protocol5_payload(root: object) -> bytes:
    return framed_protocol5_body(encode_pickle_value(root) + b".")


def test_download_and_adapter_pin_the_same_upstream_artifact() -> None:
    downloader = load_download_module()

    assert downloader.ROUTERBENCH_REVISION == routerbench.ROUTERBENCH_REVISION
    assert downloader.ROUTERBENCH_URL == routerbench.ROUTERBENCH_URL
    assert downloader.ROUTERBENCH_SIZE == routerbench.ROUTERBENCH_SIZE == 99_567_659
    assert downloader.ROUTERBENCH_SHA256 == routerbench.ROUTERBENCH_SHA256
    assert routerbench.ROUTERBENCH_ROW_COUNT == 36_497
    assert routerbench.ROUTERBENCH_COLUMN_COUNT == len(routerbench.ROUTERBENCH_COLUMNS) == 37
    assert (
        sum(column.endswith("|model_response") for column in routerbench.ROUTERBENCH_COLUMNS) == 11
    )


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


def test_adapter_rejects_arbitrary_pickle_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arbitrary_pickle = tmp_path / "arbitrary.pkl"
    arbitrary_pickle.write_bytes(b"not the pinned file")

    def fail_if_called(payload: bytes) -> None:
        pytest.fail(f"decoder must not receive unauthenticated bytes: {len(payload)}")

    monkeypatch.setattr(routerbench, "_decode_routerbench_payload", fail_if_called)

    with pytest.raises(routerbench.RouterBenchIntegrityError, match="size mismatch"):
        routerbench.load_routerbench_table(arbitrary_pickle)


def test_adapter_wraps_malformed_verified_wire_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "verified.pkl"
    payload = b"verified local fixture"
    fixture.write_bytes(payload)
    monkeypatch.setattr(routerbench, "ROUTERBENCH_SIZE", len(payload))
    monkeypatch.setattr(routerbench, "ROUTERBENCH_SHA256", hashlib.sha256(payload).hexdigest())

    with pytest.raises(routerbench.RouterBenchSchemaError, match="malformed RouterBench pickle"):
        routerbench.load_routerbench_table(fixture)


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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_tiny_layout(monkeypatch)
    payload = framed_protocol5_payload(tiny_routerbench_graph())
    fixture = tmp_path / "tiny-routerbench.pkl"
    fixture.write_bytes(payload)
    patch_artifact(routerbench, monkeypatch, payload)

    rows = routerbench.iter_routerbench_rows(fixture)
    row = next(rows)
    table = routerbench.load_routerbench_table(fixture)

    assert row["sample_id"] == "gsm8k.1"
    assert row["model-a|total_cost"] == pytest.approx(0.001)
    assert routerbench.validate_routerbench_schema(table) == ("model-a",)
    assert repr(table) == "RouterBenchTable(rows=1, columns=7)"
    with pytest.raises(TypeError):
        row["sample_id"] = "changed"  # type: ignore[index]
    with pytest.raises(AttributeError):
        table._columns = ()  # type: ignore[misc]


def stack_global_payload(module: str, name: str, *, reduce: bool) -> bytes:
    module_bytes = module.encode("utf-8")
    name_bytes = name.encode("utf-8")
    assert len(module_bytes) <= 255 and len(name_bytes) <= 255
    payload = (
        b"\x80\x05"
        + b"\x8c"
        + bytes((len(module_bytes),))
        + module_bytes
        + b"\x8c"
        + bytes((len(name_bytes),))
        + name_bytes
        + b"\x93"
    )
    if reduce:
        payload += b")R"
    return payload + b"."


def test_opcode_vm_keeps_allowed_global_and_reduce_inert() -> None:
    payload = stack_global_payload("builtins", "slice", reduce=True)

    decoded = routerbench._decode_pickle_graph(payload)

    assert isinstance(decoded, routerbench._ConstructNode)
    assert decoded.operation == "REDUCE"
    assert decoded.target == routerbench._GlobalToken("builtins", "slice")
    assert decoded.args == ()


def test_opcode_vm_build_updates_a_memoized_alias() -> None:
    target = routerbench._GlobalToken(*routerbench._DTYPE)
    args = ("O8", False, True)
    state = (3, "|", None, None, None, -1, -1, 63)
    body = (
        encode_pickle_value(target)
        + encode_pickle_value(args)
        + b"R\x94"  # REDUCE, MEMOIZE index 0
        + encode_pickle_value(state)
        + b"b"  # BUILD mutates the memoized node
        + b"h\x00"  # BINGET index 0
        + b"\x86."  # TUPLE2, STOP
    )

    decoded = routerbench._decode_pickle_graph(framed_protocol5_body(body))

    assert type(decoded) is tuple and len(decoded) == 2
    assert decoded[0] is decoded[1]
    assert isinstance(decoded[0], routerbench._ConstructNode)
    assert decoded[0].state == state


def test_opcode_vm_rejects_duplicate_dictionary_keys() -> None:
    key = encode_pickle_value("duplicate")
    body = b"}" + key + b"Ns" + key + b"Ns."

    with pytest.raises(routerbench.RouterBenchSchemaError, match="repeats a key"):
        routerbench._decode_pickle_graph(framed_protocol5_body(body))


def test_opcode_vm_rejects_forbidden_global_without_invoking_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail(f"payload callable was invoked with {args!r} {kwargs!r}")

    monkeypatch.setattr(routerbench.os, "system", fail_if_called)
    payload = stack_global_payload("os", "system", reduce=True)

    with pytest.raises(routerbench.RouterBenchSchemaError, match=r"forbidden global os\.system"):
        routerbench._decode_pickle_graph(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b"\x80\x04N.", "protocol 5"),
        (b"\x80\x05h\x00.", "invalid index"),
        (b"\x80\x05\x85.", "underflows the stack"),
        (b"\x80\x05\x97.", "forbidden opcode NEXT_BUFFER"),
        (b"\x80\x05N", "malformed RouterBench pickle"),
        (b"\x80\x05N.trailing", "STOP must be the final byte"),
        (b"\x80\x05\xff.", "malformed RouterBench pickle"),
    ),
    ids=(
        "wrong-protocol",
        "memo-miss",
        "stack-underflow",
        "forbidden-opcode",
        "missing-stop",
        "trailing-bytes",
        "unknown-opcode",
    ),
)
def test_opcode_vm_fails_closed_on_malformed_streams(payload: bytes, message: str) -> None:
    with pytest.raises(routerbench.RouterBenchSchemaError, match=message):
        routerbench._decode_pickle_graph(payload)


@pytest.mark.parametrize(
    ("graph", "message"),
    (
        (tiny_routerbench_graph(cost_bytes=b""), "expected 8 bytes"),
        (tiny_routerbench_graph(cost_dtype="f4"), "unexpected dtype"),
        (tiny_routerbench_graph(row_stop=2), "expected RangeIndex"),
    ),
    ids=("buffer-size", "dtype", "range-index"),
)
def test_structural_decoder_rejects_changed_layout(
    graph: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_tiny_layout(monkeypatch)

    with pytest.raises(routerbench.RouterBenchSchemaError, match=message):
        routerbench._decode_routerbench_graph(graph)


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model-a", float("nan"), "must be finite"),
        ("model-a", True, "non-numeric"),
        ("model-a|model_response", "", "invalid"),
        ("model-a|total_cost", float("inf"), "finite and non-negative"),
        ("model-a|total_cost", -0.1, "finite and non-negative"),
        ("oracle_model_to_route_to", "missing-model", "unknown oracle model"),
    ),
    ids=(
        "nan-quality",
        "bool-quality",
        "empty-response",
        "infinite-cost",
        "negative-cost",
        "oracle",
    ),
)
def test_row_validation_rejects_invalid_candidate_values(
    field: str,
    value: object,
    message: str,
) -> None:
    row = {
        "sample_id": "q1",
        "prompt": "prompt",
        "eval_name": "hellaswag",
        "oracle_model_to_route_to": "model-a",
        "model-a": 1.0,
        "model-a|model_response": "answer",
        "model-a|total_cost": 0.1,
    }
    row[field] = value

    with pytest.raises(routerbench.RouterBenchSchemaError, match=message):
        routerbench.validate_routerbench_row(row, ("model-a",), row_number=0)


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


def test_quoted_cost_estimation_is_independent_of_decimal_context() -> None:
    rows = tuple(
        {
            "sample_id": f"train-{index}",
            "prompt": "training prompt",
            "eval_name": "hellaswag",
            "oracle_model_to_route_to": "model-a",
            "model-a": 1.0,
            "model-a|model_response": "answer",
            "model-a|total_cost": cost,
        }
        for index, cost in enumerate((0.1, 0.2, 0.7))
    )

    with localcontext() as context:
        context.prec = 2
        low_precision = routerbench.estimate_routerbench_quoted_costs(rows)
    with localcontext() as context:
        context.prec = 60
        high_precision = routerbench.estimate_routerbench_quoted_costs(rows)

    assert low_precision == high_precision
    assert low_precision["model-a"] == Decimal(
        "0.333333333333333333333333333333333333333333333333333"
    )


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
