# SPDX-License-Identifier: Apache-2.0
"""Canonical hashes for replay data identity and order-sensitive evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from tierroute.eval.schemas import EvaluationExample


def _validated_examples(
    examples: Sequence[EvaluationExample],
) -> tuple[EvaluationExample, ...]:
    ordered = tuple(examples)
    if not ordered:
        raise ValueError("evaluation examples must not be empty")
    example_ids = tuple(example.example_id for example in ordered)
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("evaluation examples must have unique example IDs")
    return ordered


def _canonical_rows(examples: tuple[EvaluationExample, ...]) -> list[dict[str, object]]:
    rows = []
    for example in examples:
        outcomes = {outcome.model_id: outcome for outcome in example.outcomes}
        rows.append(
            {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "domain": example.domain,
                "models": [
                    {
                        "model_id": model.model_id,
                        "quoted_cost": format(model.cost, "f"),
                        "realized_cost": format(outcomes[model.model_id].cost, "f"),
                        "quality": outcomes[model.model_id].quality,
                    }
                    for model in sorted(
                        example.candidate_models,
                        key=lambda candidate: candidate.model_id,
                    )
                ],
            }
        )
    return rows


def _sha256(examples: tuple[EvaluationExample, ...]) -> str:
    try:
        document = json.dumps(
            _canonical_rows(examples),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except UnicodeEncodeError as error:
        raise ValueError("evaluation data contains invalid Unicode text") from error
    return hashlib.sha256(document).hexdigest()


def evaluation_data_sha256(examples: Sequence[EvaluationExample]) -> str:
    """Hash replay row content independent of caller order.

    Predictor fitting sorts rows by private example ID, so this identity can be
    compared directly with a fitted predictor artifact's training-data hash.
    """

    validated = _validated_examples(examples)
    return _sha256(tuple(sorted(validated, key=lambda example: example.example_id)))


def evaluation_replay_sha256(examples: Sequence[EvaluationExample]) -> str:
    """Hash replay row content in supplied order for cumulative budget evidence."""

    return _sha256(_validated_examples(examples))
