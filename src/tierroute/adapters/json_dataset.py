# SPDX-License-Identifier: Apache-2.0
"""Strict JSON adapter for bundled and user-provided replay datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from tierroute.core import BudgetTier, as_cost
from tierroute.eval import CandidateOutcome, EvaluationExample, TierSpec


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Typed data and tier configuration loaded from one JSON document."""

    name: str
    license: str
    provenance: str
    tier_specs: tuple[TierSpec, ...]
    examples: tuple[EvaluationExample, ...]


def bundled_synthetic_path() -> Path:
    """Return the installed path of the self-contained demonstration dataset."""

    resource = resources.files("tierroute.data").joinpath("synthetic.json")
    return Path(str(resource))


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def load_evaluation_dataset(path: str | Path | None = None) -> EvaluationDataset:
    """Load a versioned replay JSON file without performing network access."""

    source = Path(path) if path is not None else bundled_synthetic_path()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation dataset: {source}") from error
    root = _require_mapping(document, "dataset")
    if root.get("schema_version") != 1:
        raise ValueError("dataset.schema_version must equal 1")

    tier_specs = tuple(
        _parse_tier(_require_mapping(item, f"tier_specs[{index}]"), index)
        for index, item in enumerate(_require_list(root.get("tier_specs"), "tier_specs"))
    )
    examples = tuple(
        _parse_example(_require_mapping(item, f"examples[{index}]"), index)
        for index, item in enumerate(_require_list(root.get("examples"), "examples"))
    )
    if not tier_specs or not examples:
        raise ValueError("dataset must contain at least one tier and one example")
    if len({spec.tier for spec in tier_specs}) != len(tier_specs):
        raise ValueError("dataset tier values must be unique")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("dataset example_id values must be unique")
    return EvaluationDataset(
        name=_require_string(root.get("name"), "name"),
        license=_require_string(root.get("license"), "license"),
        provenance=_require_string(root.get("provenance"), "provenance"),
        tier_specs=tier_specs,
        examples=examples,
    )


def _parse_tier(item: Mapping[str, object], index: int) -> TierSpec:
    try:
        tier = BudgetTier(_require_string(item.get("tier"), f"tier_specs[{index}].tier"))
    except ValueError as error:
        raise ValueError(f"tier_specs[{index}].tier is not supported") from error
    return TierSpec(
        tier=tier,
        budget_limit=as_cost(_require_string(item.get("budget_limit"), "budget_limit")),
        weight=float(item.get("weight")),  # type: ignore[arg-type]
    )


def _parse_example(item: Mapping[str, object], index: int) -> EvaluationExample:
    outcomes = tuple(
        _parse_outcome(_require_mapping(value, f"examples[{index}].outcomes[{outcome_index}]"))
        for outcome_index, value in enumerate(
            _require_list(item.get("outcomes"), f"examples[{index}].outcomes")
        )
    )
    return EvaluationExample(
        example_id=_require_string(item.get("example_id"), f"examples[{index}].example_id"),
        prompt=_require_string(item.get("prompt"), f"examples[{index}].prompt"),
        domain=_require_string(item.get("domain"), f"examples[{index}].domain"),
        outcomes=outcomes,
    )


def _parse_outcome(item: Mapping[str, object]) -> CandidateOutcome:
    return CandidateOutcome(
        model_id=_require_string(item.get("model_id"), "outcome.model_id"),
        output=_require_string(item.get("output"), "outcome.output"),
        cost=as_cost(_require_string(item.get("cost"), "outcome.cost")),
        quality=float(item.get("quality")),  # type: ignore[arg-type]
    )
