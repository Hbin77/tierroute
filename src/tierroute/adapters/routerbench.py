# SPDX-License-Identifier: Apache-2.0
"""Non-dispatching boundary adapter for the pinned RouterBench zero-shot file.

The upstream artifact is a pandas pickle, but this module never unpickles it.
After authenticating the exact bytes, a small :mod:`pickletools` virtual machine
decodes only inert primitives and project-owned graph nodes.  Payload globals
are names, never imported or invoked.  The graph must then match the pinned
DataFrame block layout before it becomes a project-owned read-only table.

RouterBench data is not bundled because its dataset license is ``NOASSERTION``.
Conversion into tierroute's private replay schema remains localized here, while
:func:`iter_routerbench_rows` is retained for schema diagnostics.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import pickletools
import stat
import struct
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tierroute.core import Cost, ModelSpec, add_cost, as_cost, divide_cost
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample

ROUTERBENCH_FILENAME = "routerbench_0shot.pkl"
ROUTERBENCH_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
ROUTERBENCH_URL = (
    "https://huggingface.co/datasets/withmartian/routerbench/resolve/"
    f"{ROUTERBENCH_REVISION}/{ROUTERBENCH_FILENAME}?download=true"
)
ROUTERBENCH_SIZE = 99_567_659
ROUTERBENCH_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"

_REQUIRED_COLUMNS = frozenset({"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"})
_RESPONSE_SUFFIX = "|model_response"
_COST_SUFFIX = "|total_cost"
_NO_CORRECT_MODEL = "no_model_correct"

# These constants describe the authenticated artifact, rather than a general
# pandas pickle format.  Keeping them separate also lets security tests exercise
# the structural validator with a deliberately tiny, production-shaped graph.
ROUTERBENCH_ROW_COUNT = 36_497
ROUTERBENCH_COLUMNS = (
    "sample_id",
    "prompt",
    "eval_name",
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "gpt-3.5-turbo-1106|model_response",
    "claude-instant-v1|model_response",
    "claude-v1|model_response",
    "claude-v2|model_response",
    "gpt-4-1106-preview|model_response",
    "meta/llama-2-70b-chat|model_response",
    "mistralai/mixtral-8x7b-chat|model_response",
    "zero-one-ai/Yi-34B-Chat|model_response",
    "WizardLM/WizardLM-13B-V1.2|model_response",
    "meta/code-llama-instruct-34b-chat|model_response",
    "mistralai/mistral-7b-chat|model_response",
    "gpt-3.5-turbo-1106|total_cost",
    "claude-instant-v1|total_cost",
    "claude-v1|total_cost",
    "claude-v2|total_cost",
    "gpt-4-1106-preview|total_cost",
    "meta/llama-2-70b-chat|total_cost",
    "mistralai/mixtral-8x7b-chat|total_cost",
    "zero-one-ai/Yi-34B-Chat|total_cost",
    "WizardLM/WizardLM-13B-V1.2|total_cost",
    "meta/code-llama-instruct-34b-chat|total_cost",
    "mistralai/mistral-7b-chat|total_cost",
    "oracle_model_to_route_to",
)
ROUTERBENCH_COLUMN_COUNT = len(ROUTERBENCH_COLUMNS)
_EXPECTED_BLOCK_LAYOUT = (
    (0, 24, "object-mixed"),
    (24, 25, "object-string"),
    (25, 26, "float64"),
    (26, 27, "float64"),
    (27, 28, "float64"),
    (28, 29, "float64"),
    (29, 30, "float64"),
    (30, 31, "float64"),
    (31, 32, "float64"),
    (32, 33, "float64"),
    (33, 34, "float64"),
    (34, 35, "float64"),
    (35, 36, "float64"),
    (36, 37, "object-string"),
)

_DATAFRAME = ("pandas.core.frame", "DataFrame")
_BLOCK_MANAGER = ("pandas.core.internals.managers", "BlockManager")
_UNPICKLE_BLOCK = ("pandas._libs.internals", "_unpickle_block")
_RECONSTRUCT = ("numpy.core.multiarray", "_reconstruct")
_NDARRAY = ("numpy", "ndarray")
_DTYPE = ("numpy", "dtype")
_SLICE = ("builtins", "slice")
_FROM_BUFFER = ("numpy.core.numeric", "_frombuffer")
_NEW_INDEX = ("pandas.core.indexes.base", "_new_Index")
_INDEX = ("pandas.core.indexes.base", "Index")
_RANGE_INDEX = ("pandas.core.indexes.range", "RangeIndex")
_ALLOWED_GLOBALS = frozenset(
    {
        _DATAFRAME,
        _BLOCK_MANAGER,
        _UNPICKLE_BLOCK,
        _RECONSTRUCT,
        _NDARRAY,
        _DTYPE,
        _SLICE,
        _FROM_BUFFER,
        _NEW_INDEX,
        _INDEX,
        _RANGE_INDEX,
    }
)
_ALLOWED_OPCODES = frozenset(
    {
        "APPENDS",
        "BINFLOAT",
        "BINGET",
        "BININT",
        "BININT1",
        "BININT2",
        "BINUNICODE",
        "BUILD",
        "BYTEARRAY8",
        "EMPTY_DICT",
        "EMPTY_LIST",
        "EMPTY_TUPLE",
        "FRAME",
        "LONG_BINGET",
        "MARK",
        "MEMOIZE",
        "NEWFALSE",
        "NEWOBJ",
        "NEWTRUE",
        "NONE",
        "PROTO",
        "REDUCE",
        "SETITEM",
        "SETITEMS",
        "SHORT_BINBYTES",
        "SHORT_BINUNICODE",
        "STACK_GLOBAL",
        "STOP",
        "TUPLE",
        "TUPLE1",
        "TUPLE2",
        "TUPLE3",
    }
)

_MARK = object()
_UNBUILT = object()


class RouterBenchError(RuntimeError):
    """Base error for the optional RouterBench boundary."""


class RouterBenchIntegrityError(RouterBenchError):
    """A file is not the exact pinned RouterBench artifact."""


class RouterBenchSchemaError(RouterBenchError):
    """The verified pickle does not contain the expected zero-shot wide schema."""


@dataclass(frozen=True, slots=True)
class _GlobalToken:
    """An inert module/name pair found in a ``STACK_GLOBAL`` opcode."""

    module: str
    name: str


@dataclass(slots=True)
class _ConstructNode:
    """An inert record of pickle object construction; nothing is dispatched."""

    operation: str
    target: object
    args: tuple[object, ...]
    state: object = _UNBUILT


@dataclass(frozen=True, slots=True)
class _ObjectColumn(Sequence[object]):
    values: tuple[object, ...]
    start: int
    length: int

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int | slice) -> object:
        if isinstance(index, slice):
            indexes = range(*index.indices(self.length))
            return tuple(self.values[self.start + offset] for offset in indexes)
        normalized = index + self.length if index < 0 else index
        if normalized < 0 or normalized >= self.length:
            raise IndexError(index)
        return self.values[self.start + normalized]


@dataclass(frozen=True, slots=True)
class _Float64Column(Sequence[object]):
    """A read-only little-endian float64 view over one authenticated buffer."""

    values: bytes
    length: int

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int | slice) -> object:
        if isinstance(index, slice):
            indexes = range(*index.indices(self.length))
            return tuple(self[offset] for offset in indexes)
        normalized = index + self.length if index < 0 else index
        if normalized < 0 or normalized >= self.length:
            raise IndexError(index)
        return struct.unpack_from("<d", self.values, normalized * 8)[0]


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class RouterBenchTable:
    """Minimal read-only table protocol used by the public row iterator."""

    _columns: tuple[str, ...]
    _data: tuple[Sequence[object], ...]

    def __init__(self, columns: tuple[str, ...], data: tuple[Sequence[object], ...]) -> None:
        if not data or len(columns) != len(data):
            raise ValueError("RouterBenchTable columns and data must be non-empty and aligned")
        row_count = len(data[0])
        if any(len(column) != row_count for column in data):
            raise ValueError("RouterBenchTable columns must have equal row counts")
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_data", data)

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    def __len__(self) -> int:
        return len(self._data[0]) if self._data else 0

    def __repr__(self) -> str:
        return f"RouterBenchTable(rows={len(self)}, columns={len(self._columns)})"

    def itertuples(self, *, index: bool = False, name: None = None) -> Iterator[tuple[object, ...]]:
        """Yield rows using the small subset of pandas' API used downstream."""

        if index is not False or name is not None:
            raise ValueError("RouterBenchTable supports only index=False and name=None")
        return iter(zip(*(iter(column) for column in self._data), strict=True))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 digest using bounded standard-library reads."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified_payload(path: str | Path) -> bytes:
    """Read the exact pinned bytes once, preventing path-swap pickle loading."""

    candidate = Path(path)
    try:
        with candidate.open("rb") as stream:
            file_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise RouterBenchIntegrityError("RouterBench path must be a regular file")
            if file_stat.st_size != ROUTERBENCH_SIZE:
                raise RouterBenchIntegrityError(
                    "RouterBench size mismatch: "
                    f"expected {ROUTERBENCH_SIZE}, got {file_stat.st_size} bytes"
                )
            payload = stream.read(ROUTERBENCH_SIZE + 1)
    except RouterBenchIntegrityError:
        raise
    except OSError:
        # Suppress the original filesystem exception because its rendered traceback
        # includes the caller's absolute local path.
        raise RouterBenchIntegrityError("cannot read RouterBench file (path omitted)") from None

    if len(payload) != ROUTERBENCH_SIZE:
        raise RouterBenchIntegrityError(
            f"RouterBench changed while being read: got {len(payload)} bytes"
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_sha256, ROUTERBENCH_SHA256):
        raise RouterBenchIntegrityError(
            f"RouterBench SHA-256 mismatch: expected {ROUTERBENCH_SHA256}, got {actual_sha256}"
        )
    return payload


def verify_routerbench_file(path: str | Path) -> Path:
    """Validate that ``path`` is the exact pinned artifact and return its path."""

    _read_verified_payload(path)
    return Path(path)


def _find_mark(stack: list[object], *, opcode: str, position: int) -> int:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is _MARK:
            return index
    raise RouterBenchSchemaError(f"{opcode} at byte {position} has no matching MARK")


def _global_token(value: object, expected: tuple[str, str], label: str) -> _GlobalToken:
    if not isinstance(value, _GlobalToken) or (value.module, value.name) != expected:
        dotted = ".".join(expected)
        raise RouterBenchSchemaError(f"RouterBench {label} must reference {dotted}")
    return value


def _construct_node(
    value: object,
    *,
    operation: str,
    target: tuple[str, str],
    label: str,
) -> _ConstructNode:
    if not isinstance(value, _ConstructNode) or value.operation != operation:
        raise RouterBenchSchemaError(
            f"RouterBench {label} must be represented by pickle {operation}"
        )
    _global_token(value.target, target, label)
    return value


def _exact_structure(value: object, expected: object) -> bool:
    """Compare decoded primitives without treating booleans as integers."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        assert isinstance(value, tuple)
        return len(value) == len(expected) and all(
            _exact_structure(actual, wanted) for actual, wanted in zip(value, expected, strict=True)
        )
    if isinstance(expected, list):
        assert isinstance(value, list)
        return len(value) == len(expected) and all(
            _exact_structure(actual, wanted) for actual, wanted in zip(value, expected, strict=True)
        )
    if isinstance(expected, dict):
        assert isinstance(value, dict)
        return value.keys() == expected.keys() and all(
            _exact_structure(value[key], wanted) for key, wanted in expected.items()
        )
    return value == expected


def _decode_pickle_graph(payload: bytes) -> object:
    """Interpret the pinned pickle opcodes without dispatching any callable.

    Production callers must pass only bytes returned by
    :func:`_read_verified_payload`. Direct calls exist solely for small security
    tests; this lower-level parser is not an authentication boundary.

    This is intentionally not a general-purpose pickle implementation.  It
    supports only the opcodes present in the authenticated artifact, represents
    construction opcodes as inert nodes, and rejects every global outside the
    audited allowlist.
    """

    stack: list[object] = []
    memo: list[object] = []
    unmatched_frame_ends: set[int] = set()
    last_frame_end: int | None = None
    saw_protocol = False
    saw_stop = False
    position = 0
    opcode_name = "pickle stream"

    try:
        for opcode_number, (opcode, argument, position) in enumerate(pickletools.genops(payload)):
            opcode_name = opcode.name
            unmatched_frame_ends.discard(position)
            if opcode_name not in _ALLOWED_OPCODES:
                raise RouterBenchSchemaError(
                    f"RouterBench pickle uses forbidden opcode {opcode_name} at byte {position}"
                )
            if opcode_number == 0 and opcode_name != "PROTO":
                raise RouterBenchSchemaError("RouterBench pickle must begin with PROTO")

            if opcode_name == "PROTO":
                if saw_protocol or opcode_number != 0 or argument != 5:
                    raise RouterBenchSchemaError("RouterBench pickle must use protocol 5 exactly")
                saw_protocol = True
            elif opcode_name == "FRAME":
                if type(argument) is not int or argument <= 0:
                    raise RouterBenchSchemaError(
                        f"RouterBench FRAME at byte {position} has an invalid length"
                    )
                frame_end = position + 9 + argument
                if frame_end > len(payload):
                    raise RouterBenchSchemaError(
                        f"RouterBench FRAME at byte {position} extends past EOF"
                    )
                if last_frame_end is not None and position < last_frame_end:
                    raise RouterBenchSchemaError(
                        f"RouterBench FRAME at byte {position} overlaps another frame"
                    )
                if frame_end != len(payload):
                    unmatched_frame_ends.add(frame_end)
                last_frame_end = frame_end
            elif opcode_name == "MARK":
                stack.append(_MARK)
            elif opcode_name == "MEMOIZE":
                if not stack or stack[-1] is _MARK:
                    raise RouterBenchSchemaError(
                        f"RouterBench MEMOIZE at byte {position} has no value"
                    )
                memo.append(stack[-1])
            elif opcode_name in {"BINGET", "LONG_BINGET"}:
                if type(argument) is not int or argument < 0 or argument >= len(memo):
                    raise RouterBenchSchemaError(
                        f"RouterBench {opcode_name} at byte {position} has an invalid index"
                    )
                stack.append(memo[argument])
            elif opcode_name in {
                "BINFLOAT",
                "BININT",
                "BININT1",
                "BININT2",
                "BINUNICODE",
                "BYTEARRAY8",
                "SHORT_BINBYTES",
                "SHORT_BINUNICODE",
            }:
                stack.append(argument)
            elif opcode_name == "NONE":
                stack.append(None)
            elif opcode_name == "NEWTRUE":
                stack.append(True)
            elif opcode_name == "NEWFALSE":
                stack.append(False)
            elif opcode_name == "EMPTY_LIST":
                stack.append([])
            elif opcode_name == "EMPTY_DICT":
                stack.append({})
            elif opcode_name == "EMPTY_TUPLE":
                stack.append(())
            elif opcode_name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                item_count = int(opcode_name[-1])
                if len(stack) < item_count or any(value is _MARK for value in stack[-item_count:]):
                    raise RouterBenchSchemaError(
                        f"RouterBench {opcode_name} at byte {position} underflows the stack"
                    )
                items = tuple(stack[-item_count:])
                del stack[-item_count:]
                stack.append(items)
            elif opcode_name == "TUPLE":
                mark = _find_mark(stack, opcode=opcode_name, position=position)
                items = tuple(stack[mark + 1 :])
                del stack[mark:]
                stack.append(items)
            elif opcode_name == "STACK_GLOBAL":
                if len(stack) < 2:
                    raise RouterBenchSchemaError(
                        f"RouterBench STACK_GLOBAL at byte {position} underflows the stack"
                    )
                name = stack.pop()
                module = stack.pop()
                if type(module) is not str or type(name) is not str:
                    raise RouterBenchSchemaError(
                        f"RouterBench STACK_GLOBAL at byte {position} needs string names"
                    )
                global_name = (module, name)
                if global_name not in _ALLOWED_GLOBALS:
                    raise RouterBenchSchemaError(
                        "RouterBench pickle references forbidden global "
                        f"{module}.{name} at byte {position}"
                    )
                stack.append(_GlobalToken(module, name))
            elif opcode_name in {"REDUCE", "NEWOBJ"}:
                if len(stack) < 2:
                    raise RouterBenchSchemaError(
                        f"RouterBench {opcode_name} at byte {position} underflows the stack"
                    )
                args = stack.pop()
                target = stack.pop()
                if type(args) is not tuple or not isinstance(target, _GlobalToken):
                    raise RouterBenchSchemaError(
                        f"RouterBench {opcode_name} at byte {position} has invalid operands"
                    )
                stack.append(_ConstructNode(opcode_name, target, args))
            elif opcode_name == "BUILD":
                if len(stack) < 2:
                    raise RouterBenchSchemaError(
                        f"RouterBench BUILD at byte {position} underflows the stack"
                    )
                state = stack.pop()
                instance = stack.pop()
                if not isinstance(instance, _ConstructNode) or instance.state is not _UNBUILT:
                    raise RouterBenchSchemaError(
                        f"RouterBench BUILD at byte {position} has an invalid target"
                    )
                # Mutation is deliberate: pickle BUILD mutates a memoized object,
                # and later BINGET references must observe the same inert state.
                instance.state = state
                stack.append(instance)
            elif opcode_name == "APPENDS":
                mark = _find_mark(stack, opcode=opcode_name, position=position)
                if mark == 0 or type(stack[mark - 1]) is not list:
                    raise RouterBenchSchemaError(
                        f"RouterBench APPENDS at byte {position} has an invalid target"
                    )
                target_list = stack[mark - 1]
                assert isinstance(target_list, list)
                target_list.extend(stack[mark + 1 :])
                del stack[mark:]
            elif opcode_name == "SETITEMS":
                mark = _find_mark(stack, opcode=opcode_name, position=position)
                values = stack[mark + 1 :]
                if mark == 0 or type(stack[mark - 1]) is not dict or len(values) % 2:
                    raise RouterBenchSchemaError(
                        f"RouterBench SETITEMS at byte {position} has invalid operands"
                    )
                target_dict = stack[mark - 1]
                assert isinstance(target_dict, dict)
                for key, value in zip(values[::2], values[1::2], strict=True):
                    if key in target_dict:
                        raise RouterBenchSchemaError(
                            f"RouterBench SETITEMS at byte {position} repeats a key"
                        )
                    target_dict[key] = value
                del stack[mark:]
            elif opcode_name == "SETITEM":
                if len(stack) < 3 or type(stack[-3]) is not dict:
                    raise RouterBenchSchemaError(
                        f"RouterBench SETITEM at byte {position} has invalid operands"
                    )
                value = stack.pop()
                key = stack.pop()
                target_dict = stack[-1]
                assert isinstance(target_dict, dict)
                if key in target_dict:
                    raise RouterBenchSchemaError(
                        f"RouterBench SETITEM at byte {position} repeats a key"
                    )
                target_dict[key] = value
            elif opcode_name == "STOP":
                if position != len(payload) - 1:
                    raise RouterBenchSchemaError(
                        "RouterBench pickle STOP must be the final byte with no trailing data"
                    )
                if len(stack) != 1 or stack[0] is _MARK:
                    raise RouterBenchSchemaError(
                        "RouterBench pickle must leave exactly one root object"
                    )
                saw_stop = True

        if not saw_protocol or not saw_stop:
            raise RouterBenchSchemaError("RouterBench pickle is missing PROTO or STOP")
        if unmatched_frame_ends:
            invalid_frame_end = min(unmatched_frame_ends)
            raise RouterBenchSchemaError(
                f"RouterBench FRAME ends inside an opcode at byte {invalid_frame_end}"
            )
        return stack[0]
    except RouterBenchSchemaError:
        raise
    except (
        IndexError,
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
        struct.error,
    ) as error:
        raise RouterBenchSchemaError(
            f"malformed RouterBench pickle near {opcode_name} at byte {position}"
        ) from error


def _decode_dtype(value: object, *, kind: str, label: str) -> None:
    node = _construct_node(value, operation="REDUCE", target=_DTYPE, label=label)
    expected_code = "O8" if kind == "object" else "f8"
    expected_state = (
        (3, "|", None, None, None, -1, -1, 63)
        if kind == "object"
        else (3, "<", None, None, None, -1, -1, 0)
    )
    if not _exact_structure(node.args, (expected_code, False, True)) or not _exact_structure(
        node.state, expected_state
    ):
        raise RouterBenchSchemaError(f"RouterBench {label} has an unexpected dtype")


def _decode_object_array(
    value: object,
    *,
    shape: tuple[int, ...],
    strings_only: bool,
    label: str,
) -> tuple[object, ...]:
    node = _construct_node(value, operation="REDUCE", target=_RECONSTRUCT, label=label)
    if len(node.args) != 3:
        raise RouterBenchSchemaError(f"RouterBench {label} has invalid ndarray arguments")
    _global_token(node.args[0], _NDARRAY, f"{label} ndarray")
    if not _exact_structure(node.args[1:], ((0,), b"b")):
        raise RouterBenchSchemaError(f"RouterBench {label} has invalid ndarray arguments")
    if type(node.state) is not tuple or len(node.state) != 5:
        raise RouterBenchSchemaError(f"RouterBench {label} has invalid ndarray state")
    version, actual_shape, dtype, fortran_order, values = node.state
    if type(version) is not int or version != 1 or not _exact_structure(actual_shape, shape):
        raise RouterBenchSchemaError(f"RouterBench {label} has an unexpected shape")
    _decode_dtype(dtype, kind="object", label=f"{label} dtype")
    if fortran_order is not False or type(values) is not list:
        raise RouterBenchSchemaError(f"RouterBench {label} has invalid object data")
    expected_values = math.prod(shape)
    if len(values) != expected_values:
        raise RouterBenchSchemaError(
            f"RouterBench {label} expected {expected_values} values, got {len(values)}"
        )
    if strings_only:
        valid_values = all(type(item) is str for item in values)
    else:
        valid_values = all(
            type(item) is str or (type(item) is float and math.isfinite(item)) for item in values
        )
    if not valid_values:
        constraint = "strings" if strings_only else "finite floats or strings"
        raise RouterBenchSchemaError(f"RouterBench {label} must contain only {constraint}")
    return tuple(values)


def _decode_float_array(value: object, *, label: str) -> bytes:
    node = _construct_node(value, operation="REDUCE", target=_FROM_BUFFER, label=label)
    if node.state is not _UNBUILT or len(node.args) != 4:
        raise RouterBenchSchemaError(f"RouterBench {label} has invalid buffer state")
    values, dtype, shape, order = node.args
    if type(values) is not bytearray:
        raise RouterBenchSchemaError(f"RouterBench {label} must use BYTEARRAY8 storage")
    _decode_dtype(dtype, kind="float64", label=f"{label} dtype")
    if (
        not _exact_structure(shape, (1, ROUTERBENCH_ROW_COUNT))
        or type(order) is not str
        or order != "C"
    ):
        raise RouterBenchSchemaError(f"RouterBench {label} has an unexpected shape or order")
    expected_size = ROUTERBENCH_ROW_COUNT * 8
    if len(values) != expected_size:
        raise RouterBenchSchemaError(
            f"RouterBench {label} expected {expected_size} bytes, got {len(values)}"
        )
    return bytes(values)


def _decode_block(
    value: object,
    *,
    block_number: int,
    start: int,
    stop: int,
    kind: str,
) -> tuple[Sequence[object], ...]:
    label = f"block {block_number}"
    node = _construct_node(value, operation="REDUCE", target=_UNPICKLE_BLOCK, label=label)
    if node.state is not _UNBUILT or len(node.args) != 3:
        raise RouterBenchSchemaError(f"RouterBench {label} has invalid state")
    array, placement, version = node.args
    placement_node = _construct_node(
        placement, operation="REDUCE", target=_SLICE, label=f"{label} placement"
    )
    if placement_node.state is not _UNBUILT or not _exact_structure(
        placement_node.args, (start, stop, 1)
    ):
        raise RouterBenchSchemaError(f"RouterBench {label} has an unexpected placement")
    if type(version) is not int or version != 2:
        raise RouterBenchSchemaError(f"RouterBench {label} has an unexpected version")

    width = stop - start
    if kind.startswith("object-"):
        values = _decode_object_array(
            array,
            shape=(width, ROUTERBENCH_ROW_COUNT),
            strings_only=kind == "object-string",
            label=f"{label} values",
        )
        return tuple(
            _ObjectColumn(values, column_offset * ROUTERBENCH_ROW_COUNT, ROUTERBENCH_ROW_COUNT)
            for column_offset in range(width)
        )
    if kind == "float64" and width == 1:
        values = _decode_float_array(array, label=f"{label} values")
        return (_Float64Column(values, ROUTERBENCH_ROW_COUNT),)
    raise RouterBenchSchemaError(f"RouterBench {label} has an unsupported layout kind")


def _decode_axes(value: object) -> tuple[str, ...]:
    if type(value) is not list or len(value) != 2:
        raise RouterBenchSchemaError("RouterBench BlockManager must contain exactly two axes")
    columns_axis = _construct_node(
        value[0], operation="REDUCE", target=_NEW_INDEX, label="column index"
    )
    if columns_axis.state is not _UNBUILT or len(columns_axis.args) != 2:
        raise RouterBenchSchemaError("RouterBench column index has invalid state")
    _global_token(columns_axis.args[0], _INDEX, "column index class")
    columns_kwargs = columns_axis.args[1]
    if type(columns_kwargs) is not dict or set(columns_kwargs) != {"data", "name"}:
        raise RouterBenchSchemaError("RouterBench column index has invalid arguments")
    if columns_kwargs["name"] is not None:
        raise RouterBenchSchemaError("RouterBench column index must be unnamed")
    columns = _decode_object_array(
        columns_kwargs["data"],
        shape=(ROUTERBENCH_COLUMN_COUNT,),
        strings_only=True,
        label="column index values",
    )

    row_axis = _construct_node(value[1], operation="REDUCE", target=_NEW_INDEX, label="row index")
    if row_axis.state is not _UNBUILT or len(row_axis.args) != 2:
        raise RouterBenchSchemaError("RouterBench row index has invalid state")
    _global_token(row_axis.args[0], _RANGE_INDEX, "row index class")
    row_kwargs = row_axis.args[1]
    expected_range = {
        "name": None,
        "start": 0,
        "stop": ROUTERBENCH_ROW_COUNT,
        "step": 1,
    }
    if type(row_kwargs) is not dict or not _exact_structure(row_kwargs, expected_range):
        raise RouterBenchSchemaError("RouterBench row index must be the expected RangeIndex")
    decoded = tuple(str(column) for column in columns)
    if decoded != ROUTERBENCH_COLUMNS:
        raise RouterBenchSchemaError("RouterBench column names or order changed")
    return decoded


def _decode_routerbench_graph(root: object) -> RouterBenchTable:
    """Validate an inert pickle graph and convert it to a read-only table."""

    dataframe = _construct_node(root, operation="NEWOBJ", target=_DATAFRAME, label="root")
    if not _exact_structure(dataframe.args, ()) or type(dataframe.state) is not dict:
        raise RouterBenchSchemaError("RouterBench root has invalid DataFrame state")
    state = dataframe.state
    expected_state_keys = {"_mgr", "_typ", "_metadata", "attrs", "_flags"}
    if set(state) != expected_state_keys:
        raise RouterBenchSchemaError("RouterBench DataFrame state keys changed")
    expected_metadata = {
        "_typ": "dataframe",
        "_metadata": [],
        "attrs": {},
        "_flags": {"allows_duplicate_labels": True},
    }
    if any(not _exact_structure(state[key], value) for key, value in expected_metadata.items()):
        raise RouterBenchSchemaError("RouterBench DataFrame metadata changed")

    manager = _construct_node(
        state["_mgr"], operation="REDUCE", target=_BLOCK_MANAGER, label="BlockManager"
    )
    if manager.state is not _UNBUILT or len(manager.args) != 2:
        raise RouterBenchSchemaError("RouterBench BlockManager has invalid state")
    blocks, axes = manager.args
    if type(blocks) is not tuple or len(blocks) != len(_EXPECTED_BLOCK_LAYOUT):
        raise RouterBenchSchemaError("RouterBench BlockManager block count changed")

    decoded_columns: list[Sequence[object]] = []
    for block_number, (block, layout) in enumerate(
        zip(blocks, _EXPECTED_BLOCK_LAYOUT, strict=True)
    ):
        start, stop, kind = layout
        decoded_columns.extend(
            _decode_block(
                block,
                block_number=block_number,
                start=start,
                stop=stop,
                kind=kind,
            )
        )
    if len(decoded_columns) != ROUTERBENCH_COLUMN_COUNT:
        raise RouterBenchSchemaError("RouterBench decoded column count changed")
    columns = _decode_axes(axes)
    table = RouterBenchTable(columns, tuple(decoded_columns))
    validate_routerbench_schema(table)
    return table


def _decode_routerbench_payload(payload: bytes) -> RouterBenchTable:
    """Decode already-authenticated bytes into a project-owned table.

    The public loader obtains ``payload`` exclusively from
    :func:`_read_verified_payload`; direct access is retained for synthetic
    protocol-composition tests.
    """

    try:
        return _decode_routerbench_graph(_decode_pickle_graph(payload))
    except RouterBenchSchemaError:
        raise
    except RecursionError as error:
        raise RouterBenchSchemaError("RouterBench pickle graph is too deeply nested") from error


def validate_routerbench_schema(dataframe: Any) -> tuple[str, ...]:
    """Validate RouterBench's wide dataframe columns and return model IDs."""

    try:
        columns = tuple(dataframe.columns)
    except (AttributeError, TypeError) as error:
        raise RouterBenchSchemaError("RouterBench pickle must contain a dataframe") from error

    if not columns or any(not isinstance(column, str) for column in columns):
        raise RouterBenchSchemaError("RouterBench columns must be non-empty strings")
    if len(columns) != len(set(columns)):
        raise RouterBenchSchemaError("RouterBench dataframe has duplicate columns")
    missing_required = sorted(_REQUIRED_COLUMNS.difference(columns))
    if missing_required:
        raise RouterBenchSchemaError(
            f"RouterBench dataframe is missing required columns: {', '.join(missing_required)}"
        )

    response_models = {
        column.removesuffix(_RESPONSE_SUFFIX)
        for column in columns
        if column.endswith(_RESPONSE_SUFFIX)
    }
    cost_models = {
        column.removesuffix(_COST_SUFFIX) for column in columns if column.endswith(_COST_SUFFIX)
    }
    if not response_models:
        raise RouterBenchSchemaError("RouterBench dataframe has no model response columns")
    if response_models != cost_models:
        missing_cost = sorted(response_models - cost_models)
        missing_response = sorted(cost_models - response_models)
        details = []
        if missing_cost:
            details.append(f"missing costs for {missing_cost}")
        if missing_response:
            details.append(f"missing responses for {missing_response}")
        raise RouterBenchSchemaError("incomplete RouterBench model columns: " + "; ".join(details))

    missing_performance = sorted(response_models.difference(columns))
    if missing_performance:
        raise RouterBenchSchemaError(
            "RouterBench dataframe is missing performance columns for: "
            + ", ".join(missing_performance)
        )
    try:
        if len(dataframe) == 0:
            raise RouterBenchSchemaError("RouterBench dataframe must contain at least one row")
    except TypeError as error:
        raise RouterBenchSchemaError("RouterBench dataframe must define its row count") from error

    return tuple(sorted(response_models))


def _require_non_empty_string(value: object, field_name: str, row_number: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RouterBenchSchemaError(
            f"RouterBench row {row_number} has an invalid {field_name!r} value"
        )


def _require_finite_number(
    value: object,
    field_name: str,
    row_number: int,
    *,
    non_negative: bool,
) -> None:
    if isinstance(value, bool):
        raise RouterBenchSchemaError(
            f"RouterBench row {row_number} has a non-numeric {field_name!r} value"
        )
    try:
        finite = math.isfinite(value)  # type: ignore[arg-type]
        negative = value < 0  # type: ignore[operator]
    except (TypeError, ValueError) as error:
        raise RouterBenchSchemaError(
            f"RouterBench row {row_number} has a non-numeric {field_name!r} value"
        ) from error
    if not finite or (non_negative and negative):
        constraint = "finite and non-negative" if non_negative else "finite"
        raise RouterBenchSchemaError(
            f"RouterBench row {row_number} {field_name!r} must be {constraint}"
        )


def validate_routerbench_row(
    row: Mapping[str, object], model_ids: Sequence[str], *, row_number: int
) -> None:
    """Validate identifiers plus model performance and cost values in one raw row."""

    for field_name in ("sample_id", "prompt", "eval_name", "oracle_model_to_route_to"):
        _require_non_empty_string(row[field_name], field_name, row_number)
    for model_id in model_ids:
        _require_finite_number(row[model_id], model_id, row_number, non_negative=False)
        _require_non_empty_string(
            row[f"{model_id}{_RESPONSE_SUFFIX}"],
            f"{model_id}{_RESPONSE_SUFFIX}",
            row_number,
        )
        _require_finite_number(
            row[f"{model_id}{_COST_SUFFIX}"],
            f"{model_id}{_COST_SUFFIX}",
            row_number,
            non_negative=True,
        )
    if row["oracle_model_to_route_to"] not in {*model_ids, _NO_CORRECT_MODEL}:
        raise RouterBenchSchemaError(f"RouterBench row {row_number} names an unknown oracle model")


def load_routerbench_table(path: str | Path) -> RouterBenchTable:
    """Authenticate and decode the pinned artifact without unpickling it."""

    return _decode_routerbench_payload(_read_verified_payload(path))


def iter_routerbench_rows(path: str | Path) -> Iterator[Mapping[str, object]]:
    """Return an iterator of shallowly read-only, validated RouterBench wide rows."""

    table = load_routerbench_table(path)
    model_ids = validate_routerbench_schema(table)
    columns = tuple(table.columns)
    try:
        rows = table.itertuples(index=False, name=None)
    except (AttributeError, TypeError) as error:
        raise RouterBenchSchemaError("RouterBench dataframe cannot iterate over rows") from error

    def generate() -> Iterator[Mapping[str, object]]:
        for row_number, values in enumerate(rows):
            if len(values) != len(columns):
                raise RouterBenchSchemaError(
                    f"RouterBench row {row_number} does not match the declared columns"
                )
            row = dict(zip(columns, values, strict=True))
            validate_routerbench_row(row, model_ids, row_number=row_number)
            yield MappingProxyType(row)

    return generate()


def normalize_routerbench_domain(eval_name: str) -> str | None:
    """Map eval names used by the paper to stable LODO domains.

    The pinned file contains 1,719 extra prompts whose provenance falls outside
    the paper's declared benchmark set. They are excluded by default instead of
    being silently mixed into validation.
    """

    if eval_name.startswith("mmlu-"):
        return "mmlu"
    if eval_name == "grade-school-math":
        return "gsm8k"
    if eval_name.startswith("mtbench"):
        return "mtbench"
    if eval_name in {"arc-challenge", "hellaswag", "mbpp", "winogrande"}:
        return eval_name
    return None


def routerbench_row_to_example(
    row: Mapping[str, object],
    *,
    row_number: int,
    quoted_costs: Mapping[str, Cost],
    include_unmapped: bool = False,
) -> EvaluationExample | None:
    """Convert a row using pre-call costs fitted outside this evaluation row."""

    model_ids = tuple(
        sorted(
            column.removesuffix(_RESPONSE_SUFFIX)
            for column in row
            if column.endswith(_RESPONSE_SUFFIX)
        )
    )
    validate_routerbench_row(row, model_ids, row_number=row_number)
    if set(quoted_costs) != set(model_ids):
        raise RouterBenchSchemaError("quoted_costs must cover every RouterBench model exactly")
    eval_name = str(row["eval_name"])
    domain = normalize_routerbench_domain(eval_name)
    if domain is None:
        if not include_unmapped:
            return None
        domain = f"unmapped:{eval_name}"

    outcomes = tuple(
        CandidateOutcome(
            model_id=model_id,
            output=str(row[f"{model_id}{_RESPONSE_SUFFIX}"]),
            cost=as_cost(str(row[f"{model_id}{_COST_SUFFIX}"])),
            quality=float(row[model_id]),
        )
        for model_id in model_ids
    )
    return EvaluationExample(
        example_id=str(row["sample_id"]),
        prompt=str(row["prompt"]),
        domain=domain,
        outcomes=outcomes,
        candidate_models=tuple(
            ModelSpec(model_id, quoted_costs[model_id]) for model_id in model_ids
        ),
    )


def iter_routerbench_examples(
    path: str | Path,
    *,
    quoted_costs: Mapping[str, Cost],
    include_unmapped: bool = False,
) -> Iterator[EvaluationExample]:
    """Yield typed replay examples from the authenticated RouterBench artifact."""

    for row_number, row in enumerate(iter_routerbench_rows(path)):
        example = routerbench_row_to_example(
            row,
            row_number=row_number,
            quoted_costs=quoted_costs,
            include_unmapped=include_unmapped,
        )
        if example is not None:
            yield example


def estimate_routerbench_quoted_costs(
    calibration_rows: Sequence[Mapping[str, object]],
) -> dict[str, Cost]:
    """Estimate model-level pre-call costs from a training-only calibration split.

    RouterBench stores realized response costs, which must never be exposed for the row
    being routed. Callers are responsible for supplying only training rows here.
    """

    if not calibration_rows:
        raise ValueError("calibration_rows must not be empty")
    totals: dict[str, Decimal] = defaultdict(Decimal)
    counts: dict[str, int] = defaultdict(int)
    expected_models: tuple[str, ...] | None = None
    for row_number, row in enumerate(calibration_rows):
        model_ids = tuple(
            sorted(
                column.removesuffix(_RESPONSE_SUFFIX)
                for column in row
                if column.endswith(_RESPONSE_SUFFIX)
            )
        )
        validate_routerbench_row(row, model_ids, row_number=row_number)
        if expected_models is None:
            expected_models = model_ids
        elif model_ids != expected_models:
            raise RouterBenchSchemaError("calibration rows have inconsistent model columns")
        for model_id in model_ids:
            totals[model_id] = add_cost(
                totals[model_id], as_cost(str(row[f"{model_id}{_COST_SUFFIX}"]))
            )
            counts[model_id] += 1
    if expected_models is None:
        raise AssertionError("non-empty calibration_rows produced no model catalogue")
    return {
        model_id: divide_cost(totals[model_id], counts[model_id]) for model_id in expected_models
    }
