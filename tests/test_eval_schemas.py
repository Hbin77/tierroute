# SPDX-License-Identifier: Apache-2.0
"""Tests for label-private evaluation schemas."""

from dataclasses import fields, replace
from decimal import Decimal

import pytest

from tierroute.core import BudgetTier, CallRecord, ModelSpec
from tierroute.eval import (
    BudgetReport,
    CandidateOutcome,
    EvaluationExample,
    EvaluationReport,
    EvaluationScopeIdentity,
    QueryResult,
    ReplayCall,
    TierResult,
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
    with pytest.raises(TypeError, match="real number"):
        replace(result, quality=True)
    with pytest.raises(ValueError, match="finite"):
        replace(result, quality=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        replace(result, predicted_quality=10**10000)


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


def test_evaluation_report_requires_scope_hash_and_enforces_call_cap() -> None:
    call = ReplayCall(
        "model",
        Decimal("0.5"),
        Decimal("0.5"),
        Decimal("1"),
        Decimal("0.5"),
        True,
    )
    query = QueryResult(
        "q1",
        BudgetTier.FAST,
        True,
        "model",
        Decimal("0.5"),
        0.8,
        "answer",
        calls=(call,),
        selected_call_index=0,
    )
    tier_spec = TierSpec(BudgetTier.FAST, Decimal("1"), 1.0)
    budget = BudgetReport("test", Decimal("1"), Decimal("1"), Decimal("0.5"), 0, ("q1",))
    tier = TierResult(tier_spec, (query,), budget)
    report = EvaluationReport(
        "router",
        (tier,),
        EvaluationScopeIdentity("tierroute-evaluation-scope-v1", "0" * 64, 1),
    )

    assert report.evaluation_scope_sha256 == "0" * 64
    assert report.evaluation_scope_algorithm == "tierroute-evaluation-scope-v1"
    for algorithm in ("", "scope with spaces", "스코프"):
        with pytest.raises(ValueError, match="ASCII identifier"):
            replace(report.evaluation_scope, algorithm=algorithm)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(report.evaluation_scope, sha256="A" * 64)
    with pytest.raises(TypeError, match="integer"):
        replace(report.evaluation_scope, max_calls_per_query=True)
    with pytest.raises(ValueError, match="positive"):
        replace(report.evaluation_scope, max_calls_per_query=0)

    class BehaviorBearingString(str):
        pass

    with pytest.raises(ValueError, match="ASCII identifier"):
        replace(report.evaluation_scope, algorithm=BehaviorBearingString("scope-v2"))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(report.evaluation_scope, sha256=BehaviorBearingString("0" * 64))
    with pytest.raises(TypeError, match="EvaluationScopeIdentity"):
        replace(report, evaluation_scope="0" * 64)  # type: ignore[arg-type]
    double_call = replace(
        query,
        cost=Decimal("1"),
        calls=(call, call),
        selected_call_index=1,
    )
    double_tier = TierResult(
        tier_spec,
        (double_call,),
        replace(budget, spent=Decimal("1")),
    )
    with pytest.raises(ValueError, match="exceeds max_calls_per_query"):
        replace(report, tiers=(double_tier,))
