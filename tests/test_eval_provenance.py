# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical replay-data identity and order-sensitive hashes."""

from dataclasses import replace

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
