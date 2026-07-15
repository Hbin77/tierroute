# SPDX-License-Identifier: Apache-2.0
"""Tests for label-private evaluation schemas."""

from dataclasses import fields, replace
from decimal import Decimal

import pytest

from tierroute.core import BudgetTier, CallRecord, ModelSpec
from tierroute.eval import (
    CandidateOutcome,
    EvaluationExample,
    QueryResult,
    ReplayCall,
    TierSpec,
)


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


def test_query_result_binds_realized_cost_and_selection_to_call_evidence() -> None:
    call = ReplayCall(
        "model",
        Decimal("0.1"),
        Decimal("0.2"),
        Decimal("1"),
        Decimal("0.8"),
        True,
    )
    result = QueryResult(
        "q1",
        BudgetTier.FAST,
        True,
        "model",
        Decimal("0.2"),
        0.9,
        "answer",
        calls=(call,),
        selected_call_index=0,
    )

    assert result.calls == (call,)
    with pytest.raises(ValueError, match="exact sum"):
        replace(result, cost=Decimal("0.1"))
    with pytest.raises(ValueError, match="must agree"):
        replace(result, selected_model_id="other")
    with pytest.raises(TypeError, match="non-negative integer"):
        replace(result, selected_call_index=-1)


def test_infeasible_query_keeps_executed_overspend_call_without_selecting_it() -> None:
    overspend = ReplayCall(
        "model",
        Decimal("0.1"),
        Decimal("2"),
        Decimal("1"),
        Decimal(0),
        False,
    )

    result = QueryResult(
        "q1",
        BudgetTier.FAST,
        False,
        None,
        Decimal("2"),
        None,
        None,
        error="realized overspend",
        calls=(overspend,),
    )

    assert result.calls == (overspend,)
    with pytest.raises(ValueError, match="cannot select"):
        replace(result, selected_model_id="model")


def test_replay_call_rejects_an_executed_call_with_an_unaffordable_quote() -> None:
    valid = ReplayCall(
        "model",
        Decimal("0.5"),
        Decimal("0.5"),
        Decimal("1"),
        Decimal("0.5"),
        True,
    )

    with pytest.raises(ValueError, match="affordable quoted"):
        replace(valid, quoted_cost=Decimal("2"))
