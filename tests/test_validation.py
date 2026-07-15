# SPDX-License-Identifier: Apache-2.0
"""Tests for mandatory leave-one-domain-out splitting."""

from decimal import Decimal

from tierroute.eval import CandidateOutcome, EvaluationExample, leave_one_domain_out


def example(example_id: str, domain: str) -> EvaluationExample:
    return EvaluationExample(
        example_id,
        "prompt",
        domain,
        (CandidateOutcome("model", "output", Decimal("1"), 0.5),),
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
