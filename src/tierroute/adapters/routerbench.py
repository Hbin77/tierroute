# SPDX-License-Identifier: Apache-2.0
"""Safe, opt-in boundary adapter for the pinned RouterBench zero-shot file.

The upstream artifact is a pandas pickle, a format that can execute code while
loading.  This module therefore refuses every payload except the exact artifact
identified by its pinned byte length and SHA-256 digest.  RouterBench data is
not bundled because its dataset license is ``NOASSERTION``.

Conversion into tierroute's private replay schema remains localized here, while
:func:`iter_routerbench_rows` is retained for schema diagnostics.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import math
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tierroute.core import as_cost
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


class RouterBenchError(RuntimeError):
    """Base error for the optional RouterBench boundary."""


class RouterBenchIntegrityError(RouterBenchError):
    """A file is not the exact pinned RouterBench artifact."""


class RouterBenchDependencyError(RouterBenchError):
    """The optional pandas dependency is unavailable or unusable."""


class RouterBenchSchemaError(RouterBenchError):
    """The verified pickle does not contain the expected zero-shot wide schema."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 digest without importing pandas."""

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
    except OSError as error:
        raise RouterBenchIntegrityError(f"cannot read RouterBench file: {candidate}") from error

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


def _import_pandas() -> Any:
    """Import the optional pickle reader only after artifact verification."""

    try:
        import pandas
    except ImportError as error:
        raise RouterBenchDependencyError(
            "RouterBench loading requires the optional 'pandas' package and its "
            "dependencies; install a compatible pandas release before using this adapter"
        ) from error
    return pandas


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
    if row["oracle_model_to_route_to"] not in model_ids:
        raise RouterBenchSchemaError(f"RouterBench row {row_number} names an unknown oracle model")


def load_routerbench_dataframe(path: str | Path) -> Any:
    """Load and validate the pinned dataframe after authenticating its bytes."""

    payload = _read_verified_payload(path)
    pandas = _import_pandas()
    dataframe = pandas.read_pickle(io.BytesIO(payload))
    validate_routerbench_schema(dataframe)
    return dataframe


def iter_routerbench_rows(path: str | Path) -> Iterator[Mapping[str, object]]:
    """Return an iterator of shallowly read-only, validated RouterBench wide rows."""

    dataframe = load_routerbench_dataframe(path)
    model_ids = validate_routerbench_schema(dataframe)
    columns = tuple(dataframe.columns)
    try:
        rows = dataframe.itertuples(index=False, name=None)
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
    include_unmapped: bool = False,
) -> EvaluationExample | None:
    """Convert one validated wide row without exposing its oracle label."""

    model_ids = tuple(
        sorted(
            column.removesuffix(_RESPONSE_SUFFIX)
            for column in row
            if column.endswith(_RESPONSE_SUFFIX)
        )
    )
    validate_routerbench_row(row, model_ids, row_number=row_number)
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
    )


def iter_routerbench_examples(
    path: str | Path, *, include_unmapped: bool = False
) -> Iterator[EvaluationExample]:
    """Yield typed replay examples from the authenticated RouterBench artifact."""

    for row_number, row in enumerate(iter_routerbench_rows(path)):
        example = routerbench_row_to_example(
            row,
            row_number=row_number,
            include_unmapped=include_unmapped,
        )
        if example is not None:
            yield example
