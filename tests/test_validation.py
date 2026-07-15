# SPDX-License-Identifier: Apache-2.0
"""Tests for mandatory leave-one-domain-out splitting."""

from decimal import Decimal

from tierroute.core import ModelSpec
from tierroute.eval import CandidateOutcome, EvaluationExample, leave_one_domain_out


def example(example_id: str, domain: str) -> EvaluationExample:
    return EvaluationExample(
        example_id,
        "prompt",
        domain,
        (CandidateOutcome("model", "output", Decimal("1"), 0.5),),
        (ModelSpec("model", Decimal("1")),),
    )


def test_lodo_holds_out_each_complete_domain_without_overlap() -> None:
    examples = (example("q1", "math"), example("q2", "code"), example("q3", "math"))

    folds = leave_one_domain_out(examples)

    assert [fold.held_out_domain for fold in folds] == ["code", "math"]
    for fold in folds:
        assert {item.domain for item in fold.test} == {fold.held_out_domain}
        training_ids = {item.example_id for item in fold.training}
        test_ids = {item.example_id for item in fold.test}
        assert not (training_ids & test_ids)


def test_lodo_rejects_duplicate_ids_even_across_domains() -> None:
    examples = (example("duplicate", "math"), example("duplicate", "code"))

    try:
        leave_one_domain_out(examples)
    except ValueError as error:
        assert "unique example_id" in str(error)
    else:
        raise AssertionError("duplicate IDs must not enter LODO folds")
