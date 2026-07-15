# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical replay-data identity and order-sensitive hashes."""

from dataclasses import replace
from decimal import Decimal

import pytest

from tierroute.adapters import load_evaluation_dataset
from tierroute.eval import evaluation_data_sha256, evaluation_replay_sha256
from tierroute.predictors import training_data_sha256


def test_data_identity_is_order_independent_but_replay_hash_is_not() -> None:
    examples = load_evaluation_dataset().examples
    reversed_examples = tuple(reversed(examples))

    assert evaluation_data_sha256(examples) == evaluation_data_sha256(reversed_examples)
    assert training_data_sha256(examples) == evaluation_data_sha256(examples)
    assert evaluation_replay_sha256(examples) != evaluation_replay_sha256(reversed_examples)


def test_data_identity_covers_replay_content() -> None:
    examples = load_evaluation_dataset().examples
    changed = (replace(examples[0], prompt=f"{examples[0].prompt} changed"), *examples[1:])

    assert evaluation_data_sha256(changed) != evaluation_data_sha256(examples)
    assert evaluation_replay_sha256(changed) != evaluation_replay_sha256(examples)


def test_data_hashes_reject_empty_or_duplicate_examples() -> None:
    example = load_evaluation_dataset().examples[0]

    with pytest.raises(ValueError, match="must not be empty"):
        evaluation_data_sha256(())
    with pytest.raises(ValueError, match="unique example IDs"):
        evaluation_replay_sha256((example, example))

    invalid_text = (replace(example, prompt="\ud800"),)
    with pytest.raises(ValueError, match="invalid Unicode text"):
        evaluation_data_sha256(invalid_text)


def test_extreme_zero_exponents_hash_as_canonical_zero() -> None:
    example = load_evaluation_dataset().examples[0]
    model_id = example.candidate_models[0].model_id

    def with_cost(cost: Decimal):
        return replace(
            example,
            candidate_models=tuple(
                replace(model, cost=cost) if model.model_id == model_id else model
                for model in example.candidate_models
            ),
            outcomes=tuple(
                replace(outcome, cost=cost) if outcome.model_id == model_id else outcome
                for outcome in example.outcomes
            ),
        )

    canonical = (with_cost(Decimal(0)),)
    extreme = (with_cost(Decimal("0e-100000000")),)

    assert evaluation_data_sha256(extreme) == evaluation_data_sha256(canonical)
    assert evaluation_replay_sha256(extreme) == evaluation_replay_sha256(canonical)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (Decimal("1"), Decimal("1.0")),
        (Decimal("0.1"), Decimal("0.10")),
        (Decimal("1E+2"), Decimal("100.00")),
    ],
)
def test_equivalent_nonzero_cost_encodings_share_provenance(
    left: Decimal,
    right: Decimal,
) -> None:
    example = load_evaluation_dataset().examples[0]
    model_id = example.candidate_models[0].model_id

    def with_cost(cost: Decimal):
        return replace(
            example,
            candidate_models=tuple(
                replace(model, cost=cost) if model.model_id == model_id else model
                for model in example.candidate_models
            ),
            outcomes=tuple(
                replace(outcome, cost=cost) if outcome.model_id == model_id else outcome
                for outcome in example.outcomes
            ),
        )

    assert evaluation_data_sha256((with_cost(left),)) == evaluation_data_sha256((with_cost(right),))
