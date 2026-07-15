# SPDX-License-Identifier: Apache-2.0
"""Tests for label-private evaluation schemas."""

from dataclasses import fields
from decimal import Decimal

from tierroute.core import BudgetTier, CallRecord, ModelSpec
from tierroute.eval import CandidateOutcome, EvaluationExample, TierSpec


def test_quality_label_is_not_exposed_by_core_call_record() -> None:
    assert "quality" not in {field.name for field in fields(CallRecord)}


def test_example_exposes_only_model_id_and_cost_to_router() -> None:
    example = EvaluationExample(
        "q1",
        "prompt",
        "reasoning",
        (CandidateOutcome("model", "secret output", Decimal("0.2"), 0.9),),
        (ModelSpec("model", Decimal("0.1")),),
    )

    assert example.candidate_models[0].model_id == "model"
    assert example.candidate_models[0].cost == Decimal("0.1")
    assert not hasattr(example.candidate_models[0], "quality")


def test_tier_spec_keeps_weight_explicit() -> None:
    tier = TierSpec(BudgetTier.FAST, Decimal("1"), 0.5)

    assert tier.weight == 0.5
