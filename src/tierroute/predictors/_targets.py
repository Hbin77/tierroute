# SPDX-License-Identifier: Apache-2.0
"""Linear-time target-column construction shared by predictor trainers."""

from __future__ import annotations

from tierroute.eval.schemas import EvaluationExample


def targets_by_model(
    examples: tuple[EvaluationExample, ...],
    model_ids: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    """Transpose validated outcome rows without repeated linear model-ID scans."""

    columns: dict[str, list[float]] = {model_id: [] for model_id in model_ids}
    for example in examples:
        seen: set[str] = set()
        for outcome in example.outcomes:
            if outcome.model_id in seen:
                raise ValueError("training outcomes must contain unique model IDs per row")
            seen.add(outcome.model_id)
            try:
                columns[outcome.model_id].append(outcome.quality)
            except KeyError as error:  # defensive if a schema object is forged
                raise ValueError(
                    "training outcomes must match the stable model catalogue"
                ) from error
        if seen != columns.keys():
            raise ValueError("training outcomes must cover the stable model catalogue")
    if any(len(values) != len(examples) for values in columns.values()):
        raise ValueError("training outcomes must cover every model exactly once per row")
    return {model_id: tuple(columns[model_id]) for model_id in model_ids}
