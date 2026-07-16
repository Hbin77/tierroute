# SPDX-License-Identifier: Apache-2.0
"""Strict, bounded JSON adapter for bundled and user-provided replay datasets."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from tierroute.adapters.resource_limits import (
    MAX_REPLAY_COST_TEXT_BYTES,
    MAX_REPLAY_DATASET_BYTES,
    MAX_REPLAY_DOMAINS,
    MAX_REPLAY_EXAMPLES,
    MAX_REPLAY_JSON_NESTING_DEPTH,
    MAX_REPLAY_JSON_NUMBER_CHARACTERS,
    MAX_REPLAY_JSON_NUMBER_TOKENS,
    MAX_REPLAY_JSON_OBJECT_FIELDS,
    MAX_REPLAY_JSON_STRING_CHARACTERS,
    MAX_REPLAY_JSON_STRING_TOKENS,
    MAX_REPLAY_JSON_STRUCTURE_TOKENS,
    MAX_REPLAY_LODO_MEMBERSHIPS,
    MAX_REPLAY_METADATA_TEXT_BYTES,
    MAX_REPLAY_NESTED_LODO_MEMBERSHIPS,
    MAX_REPLAY_OUTCOMES_PER_EXAMPLE,
    MAX_REPLAY_OUTPUT_TEXT_BYTES,
    MAX_REPLAY_PROMPT_TEXT_BYTES,
    MAX_REPLAY_TIERS,
    MAX_REPLAY_TOTAL_OUTCOMES,
    MAX_REPLAY_TRAINING_OUTCOME_SCANS,
)
from tierroute.core import BudgetTier, ModelSpec, as_cost
from tierroute.eval import CandidateOutcome, EvaluationExample, TierSpec

_ROOT_FIELDS = {
    "schema_version",
    "name",
    "license",
    "provenance",
    "domain_labels_are_observable",
    "tier_specs",
    "examples",
}
_TIER_FIELDS = {"tier", "budget_limit", "weight"}
_EXAMPLE_FIELDS = {"example_id", "prompt", "domain", "outcomes"}
_OUTCOME_REQUIRED_FIELDS = {"model_id", "output", "cost", "quality"}
_OUTCOME_OPTIONAL_FIELDS = {"quoted_cost"}
_DECIMAL_TEXT_PATTERN = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?"
)


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Typed data and tier configuration loaded from one JSON document."""

    name: str
    license: str
    provenance: str
    domain_labels_are_observable: bool
    tier_specs: tuple[TierSpec, ...]
    examples: tuple[EvaluationExample, ...]


def bundled_synthetic_path() -> Path:
    """Return the installed path of the self-contained demonstration dataset."""

    resource = resources.files("tierroute.data").joinpath("synthetic.json")
    return Path(str(resource))


def _read_replay_document(source: Path) -> str:
    """Read one stable regular-file descriptor within the adapter byte limit."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(source, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("evaluation dataset must resolve to a regular file")
            if before.st_size < 0 or before.st_size > MAX_REPLAY_DATASET_BYTES:
                raise ValueError(f"evaluation dataset exceeds {MAX_REPLAY_DATASET_BYTES:,} bytes")

            payload = bytearray()
            remaining = MAX_REPLAY_DATASET_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                payload.extend(chunk)
                remaining -= len(chunk)
            if len(payload) > MAX_REPLAY_DATASET_BYTES:
                raise ValueError(f"evaluation dataset exceeds {MAX_REPLAY_DATASET_BYTES:,} bytes")

            after = os.fstat(descriptor)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity or len(payload) != before.st_size:
                raise ValueError("evaluation dataset changed while reading")
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError(f"cannot read evaluation dataset: {source}") from error

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"evaluation dataset is not valid UTF-8: {source}") from error


def _preflight_json_structure(document: str) -> None:
    """Bound parser-amplifying structure without materializing decoded JSON values."""

    containers: list[list[object]] = []
    number_tokens = 0
    string_tokens = 0
    structure_tokens = 0
    index = 0
    while index < len(document):
        character = document[index]
        if character == '"':
            string_tokens += 1
            if string_tokens > MAX_REPLAY_JSON_STRING_TOKENS:
                raise ValueError(
                    "evaluation dataset exceeds the JSON string-token limit "
                    f"({MAX_REPLAY_JSON_STRING_TOKENS:,})"
                )
            start = index
            index += 1
            while index < len(document):
                if index - start + 1 > MAX_REPLAY_JSON_STRING_CHARACTERS:
                    raise ValueError(
                        "evaluation dataset JSON string exceeds the lexical limit "
                        f"({MAX_REPLAY_JSON_STRING_CHARACTERS:,} characters)"
                    )
                if document[index] == "\\":
                    index += 2
                    continue
                if document[index] == '"':
                    break
                index += 1
        elif character in "[{":
            containers.append([character, 0])
            structure_tokens += 1
            if len(containers) > MAX_REPLAY_JSON_NESTING_DEPTH:
                raise ValueError(
                    "evaluation dataset exceeds the JSON nesting limit "
                    f"({MAX_REPLAY_JSON_NESTING_DEPTH:,})"
                )
        elif character in "]}":
            if containers:
                containers.pop()
        elif character == "-" or character in "0123456789":
            number_tokens += 1
            if number_tokens > MAX_REPLAY_JSON_NUMBER_TOKENS:
                raise ValueError(
                    "evaluation dataset exceeds the JSON number-token limit "
                    f"({MAX_REPLAY_JSON_NUMBER_TOKENS:,})"
                )
            start = index
            index += 1
            while index < len(document) and document[index] not in " \t\r\n,]}":
                index += 1
            if index - start > MAX_REPLAY_JSON_NUMBER_CHARACTERS:
                raise ValueError(
                    "evaluation dataset JSON number exceeds the lexical limit "
                    f"({MAX_REPLAY_JSON_NUMBER_CHARACTERS:,} characters)"
                )
            index -= 1
        elif character == ",":
            structure_tokens += 1
            if containers and containers[-1][0] == "{":
                containers[-1][1] = int(containers[-1][1]) + 1
                if int(containers[-1][1]) + 1 > MAX_REPLAY_JSON_OBJECT_FIELDS:
                    raise ValueError(
                        "evaluation dataset JSON object exceeds the field limit "
                        f"({MAX_REPLAY_JSON_OBJECT_FIELDS:,})"
                    )
        if structure_tokens > MAX_REPLAY_JSON_STRUCTURE_TOKENS:
            raise ValueError(
                "evaluation dataset exceeds the JSON structure-token limit "
                f"({MAX_REPLAY_JSON_STRUCTURE_TOKENS:,})"
            )
        index += 1


def _parse_strict_json(document: str) -> object:
    _preflight_json_structure(document)
    number_tokens = 0

    def count_number_token(token: str, *, floating: bool) -> int | float:
        nonlocal number_tokens
        number_tokens += 1
        if number_tokens > MAX_REPLAY_JSON_NUMBER_TOKENS:
            raise ValueError(
                "evaluation dataset exceeds the JSON number-token limit "
                f"({MAX_REPLAY_JSON_NUMBER_TOKENS:,})"
            )
        if len(token) > MAX_REPLAY_JSON_NUMBER_CHARACTERS:
            raise ValueError(
                "evaluation dataset JSON number exceeds the lexical limit "
                f"({MAX_REPLAY_JSON_NUMBER_CHARACTERS:,} characters)"
            )
        if not floating:
            return int(token)
        result = float(token)
        if not math.isfinite(result):
            raise ValueError("evaluation dataset JSON numbers must fit finite binary64")
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON number {value!r} is forbidden")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        if len(pairs) > MAX_REPLAY_JSON_OBJECT_FIELDS:
            raise ValueError(
                "evaluation dataset JSON object exceeds the field limit "
                f"({MAX_REPLAY_JSON_OBJECT_FIELDS:,})"
            )
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} is forbidden")
            result[key] = value
        return result

    try:
        return json.loads(
            document,
            parse_int=lambda token: count_number_token(token, floating=False),
            parse_float=lambda token: count_number_token(token, floating=True),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("evaluation dataset is not valid strict JSON") from error


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an object")
    return value


def _require_list(
    value: object,
    context: str,
    *,
    max_items: int,
) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{context} must be a list")
    if len(value) > max_items:
        raise ValueError(f"{context} exceeds the collection limit ({max_items:,})")
    return value


def _require_string(
    value: object,
    context: str,
    *,
    max_bytes: int | None = None,
) -> str:
    if max_bytes is None:
        max_bytes = MAX_REPLAY_METADATA_TEXT_BYTES
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must contain valid Unicode") from error
    if len(encoded) > max_bytes:
        raise ValueError(f"{context} exceeds the text limit ({max_bytes:,} UTF-8 bytes)")
    return value


def _require_cost_string(value: object, context: str) -> str:
    result = _require_string(value, context, max_bytes=MAX_REPLAY_COST_TEXT_BYTES)
    if _DECIMAL_TEXT_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{context} must use finite non-negative decimal syntax")
    return result


def _require_boolean(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _require_finite_number(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{context} must be a JSON number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{context} must fit finite binary64") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must fit finite binary64")
    return result


def _strict_fields(payload: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{context} fields mismatch: missing={missing}, extra={extra}")


def _strict_outcome_fields(payload: Mapping[str, object], context: str) -> None:
    fields = set(payload)
    missing = _OUTCOME_REQUIRED_FIELDS - fields
    extra = fields - _OUTCOME_REQUIRED_FIELDS - _OUTCOME_OPTIONAL_FIELDS
    if missing or extra:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def load_evaluation_dataset(path: str | Path | None = None) -> EvaluationDataset:
    """Load one bounded version-1 replay JSON file without network access."""

    source = Path(path) if path is not None else bundled_synthetic_path()
    document = _parse_strict_json(_read_replay_document(source))
    root = _require_mapping(document, "dataset")
    _strict_fields(root, _ROOT_FIELDS, "dataset")
    schema_version = root["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("dataset.schema_version must be the integer 1")
    domain_labels_are_observable = _require_boolean(
        root["domain_labels_are_observable"],
        "domain_labels_are_observable",
    )

    tier_items = _require_list(root["tier_specs"], "tier_specs", max_items=MAX_REPLAY_TIERS)
    example_items = _require_list(root["examples"], "examples", max_items=MAX_REPLAY_EXAMPLES)
    if not tier_items or not example_items:
        raise ValueError("dataset must contain at least one tier and one example")

    # Validate every raw collection and the LODO expansion before constructing any
    # dataclass graph. This boundary is deliberately independent of SKT's future schema.
    total_outcomes = 0
    max_outcomes_per_example = 0
    domains: set[str] = set()
    for index, raw_item in enumerate(example_items):
        item = _require_mapping(raw_item, f"examples[{index}]")
        _strict_fields(item, _EXAMPLE_FIELDS, f"examples[{index}]")
        outcomes = _require_list(
            item["outcomes"],
            f"examples[{index}].outcomes",
            max_items=MAX_REPLAY_OUTCOMES_PER_EXAMPLE,
        )
        if not outcomes:
            raise ValueError(f"examples[{index}].outcomes must not be empty")
        total_outcomes += len(outcomes)
        max_outcomes_per_example = max(max_outcomes_per_example, len(outcomes))
        if total_outcomes > MAX_REPLAY_TOTAL_OUTCOMES:
            raise ValueError(
                "dataset outcomes exceed the aggregate collection limit "
                f"({MAX_REPLAY_TOTAL_OUTCOMES:,})"
            )
        domain = _require_string(item["domain"], f"examples[{index}].domain")
        domains.add(domain)
        if len(domains) > MAX_REPLAY_DOMAINS:
            raise ValueError(f"dataset exceeds the domain limit ({MAX_REPLAY_DOMAINS:,})")
    lodo_memberships = len(example_items) * len(domains)
    if lodo_memberships > MAX_REPLAY_LODO_MEMBERSHIPS:
        raise ValueError(
            f"dataset exceeds the LODO membership limit ({MAX_REPLAY_LODO_MEMBERSHIPS:,})"
        )
    training_outcome_scans = lodo_memberships * max_outcomes_per_example**2
    if training_outcome_scans > MAX_REPLAY_TRAINING_OUTCOME_SCANS:
        raise ValueError(
            "dataset exceeds the training outcome-scan limit "
            f"({MAX_REPLAY_TRAINING_OUTCOME_SCANS:,})"
        )
    nested_lodo_memberships = len(example_items) * max(len(domains) - 1, 0) ** 2
    if nested_lodo_memberships > MAX_REPLAY_NESTED_LODO_MEMBERSHIPS:
        raise ValueError(
            "dataset exceeds the nested-LODO membership limit "
            f"({MAX_REPLAY_NESTED_LODO_MEMBERSHIPS:,})"
        )

    tier_specs = tuple(
        _parse_tier(_require_mapping(item, f"tier_specs[{index}]"), index)
        for index, item in enumerate(tier_items)
    )
    examples = tuple(
        _parse_example(
            _require_mapping(item, f"examples[{index}]"),
            index,
            domain_label_is_observable=domain_labels_are_observable,
        )
        for index, item in enumerate(example_items)
    )
    if len({spec.tier for spec in tier_specs}) != len(tier_specs):
        raise ValueError("dataset tier values must be unique")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("dataset example_id values must be unique")
    return EvaluationDataset(
        name=_require_string(root["name"], "name"),
        license=_require_string(root["license"], "license"),
        provenance=_require_string(root["provenance"], "provenance"),
        domain_labels_are_observable=domain_labels_are_observable,
        tier_specs=tier_specs,
        examples=examples,
    )


def _parse_tier(item: Mapping[str, object], index: int) -> TierSpec:
    context = f"tier_specs[{index}]"
    _strict_fields(item, _TIER_FIELDS, context)
    try:
        tier = BudgetTier(_require_string(item["tier"], f"{context}.tier"))
    except ValueError as error:
        raise ValueError(f"{context}.tier is not supported") from error
    return TierSpec(
        tier=tier,
        budget_limit=as_cost(_require_cost_string(item["budget_limit"], f"{context}.budget_limit")),
        weight=_require_finite_number(item["weight"], f"{context}.weight"),
    )


def _parse_example(
    item: Mapping[str, object], index: int, *, domain_label_is_observable: bool
) -> EvaluationExample:
    context = f"examples[{index}]"
    _strict_fields(item, _EXAMPLE_FIELDS, context)
    outcome_items = _require_list(
        item["outcomes"],
        f"{context}.outcomes",
        max_items=MAX_REPLAY_OUTCOMES_PER_EXAMPLE,
    )
    parsed = tuple(
        _parse_outcome(_require_mapping(value, f"{context}.outcomes[{outcome_index}]"), context)
        for outcome_index, value in enumerate(outcome_items)
    )
    domain = _require_string(item["domain"], f"{context}.domain")
    return EvaluationExample(
        example_id=_require_string(item["example_id"], f"{context}.example_id"),
        prompt=_require_string(
            item["prompt"],
            f"{context}.prompt",
            max_bytes=MAX_REPLAY_PROMPT_TEXT_BYTES,
        ),
        domain=domain,
        outcomes=tuple(outcome for outcome, _ in parsed),
        candidate_models=tuple(model for _, model in parsed),
        router_metadata={"domain": domain} if domain_label_is_observable else {},
    )


def _parse_outcome(
    item: Mapping[str, object],
    example_context: str,
) -> tuple[CandidateOutcome, ModelSpec]:
    context = f"{example_context}.outcome"
    _strict_outcome_fields(item, context)
    model_id = _require_string(item["model_id"], f"{context}.model_id")
    charged_text = _require_cost_string(item["cost"], f"{context}.cost")
    charged_cost = as_cost(charged_text)
    quoted_value = item["quoted_cost"] if "quoted_cost" in item else charged_text
    quoted_cost = as_cost(_require_cost_string(quoted_value, f"{context}.quoted_cost"))
    return (
        CandidateOutcome(
            model_id=model_id,
            output=_require_string(
                item["output"],
                f"{context}.output",
                max_bytes=MAX_REPLAY_OUTPUT_TEXT_BYTES,
            ),
            cost=charged_cost,
            quality=_require_finite_number(item["quality"], f"{context}.quality"),
        ),
        ModelSpec(model_id, quoted_cost),
    )
