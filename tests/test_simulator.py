# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for full-information offline replay."""

from decimal import Decimal, localcontext

import pytest

import tierroute.eval.provenance as provenance_module
from tierroute.adapters import CumulativeBudgetLedger, PerQueryBudgetLedger
from tierroute.core import BudgetTier, CallModel, ModelSpec, RouterState, SelectOutput
from tierroute.eval import (
    CandidateOutcome,
    EvaluationExample,
    OfflineSimulator,
    TierSpec,
    build_per_query_oracle_plan,
    weighted_delta,
)
from tierroute.policies import AlwaysCheapestRouter, AlwaysPremiumRouter, OracleRouter

MODELS = (ModelSpec("cheap", Decimal("1")), ModelSpec("premium", Decimal("2")))
EXAMPLES = (
    EvaluationExample(
        "q1",
        "easy prompt",
        "general",
        (
            CandidateOutcome("cheap", "cheap one", Decimal("1"), 0.5),
            CandidateOutcome("premium", "premium one", Decimal("2"), 0.9),
        ),
        MODELS,
    ),
    EvaluationExample(
        "q2",
        "hard prompt",
        "reasoning",
        (
            CandidateOutcome("cheap", "cheap two", Decimal("1"), 0.4),
            CandidateOutcome("premium", "premium two", Decimal("2"), 1.0),
        ),
        MODELS,
    ),
)
TIER = TierSpec(BudgetTier.FAST, Decimal("2"), 1.0)


def test_simulator_replays_calls_then_selects_without_leaking_quality() -> None:
    simulator = OfflineSimulator(PerQueryBudgetLedger)

    result = simulator.run_tier(AlwaysCheapestRouter(), EXAMPLES, TIER)

    assert result.feasible is True
    assert result.mean_quality == 0.45
    assert result.budget.spent == Decimal("2")
    assert result.queries[0].selected_model_id == "cheap"
    assert "call cheap" in result.queries[0].decision_reason


def test_same_simulator_supports_cumulative_budget_via_adapter_only() -> None:
    simulator = OfflineSimulator(CumulativeBudgetLedger)

    result = simulator.run_tier(AlwaysPremiumRouter("premium"), EXAMPLES, TIER)

    assert result.feasible is False
    assert result.queries[0].quality == 0.9
    assert result.queries[1].quality is None
    assert result.budget.spent == Decimal("2")


def test_second_call_is_rejected_by_one_shot_limit_after_first_cost_is_charged() -> None:
    class CallsForever:
        def route(self, state: object) -> CallModel:
            return CallModel("cheap")

    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(CallsForever(), EXAMPLES[:1], TIER)

    assert result.feasible is False
    assert result.queries[0].cost == Decimal("1")
    assert "max_calls_per_query=1" in (result.queries[0].error or "")


def test_per_query_oracle_plan_is_budget_feasible_and_privileged() -> None:
    plan = build_per_query_oracle_plan(EXAMPLES, (TIER,))
    report = OfflineSimulator(PerQueryBudgetLedger).run_tier(OracleRouter(plan), EXAMPLES, TIER)

    assert report.mean_quality == 0.95
    assert {query.selected_model_id for query in report.queries} == {"premium"}


def test_oracle_requires_quote_and_realized_charge_to_fit_budget() -> None:
    example = EvaluationExample(
        "q-oracle-cost",
        "prompt",
        "general",
        (
            CandidateOutcome("cheap", "ok", Decimal("1"), 0.4),
            CandidateOutcome("premium", "best", Decimal("1"), 1.0),
        ),
        (
            ModelSpec("cheap", Decimal("1")),
            ModelSpec("premium", Decimal("3")),
        ),
    )
    tier = TierSpec(BudgetTier.FAST, Decimal("1"), 1.0)

    plan = build_per_query_oracle_plan((example,), (tier,))

    assert plan[(BudgetTier.FAST, "q-oracle-cost")] == "cheap"


def test_policy_sees_quoted_cost_but_ledger_charges_hidden_realized_cost() -> None:
    example = EvaluationExample(
        "q-cost",
        "prompt",
        "general",
        (
            CandidateOutcome("quoted-cheap", "long output", Decimal("9"), 0.5),
            CandidateOutcome("quoted-high", "short", Decimal("1"), 0.6),
        ),
        (
            ModelSpec("quoted-cheap", Decimal("0.1")),
            ModelSpec("quoted-high", Decimal("0.2")),
        ),
    )

    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(
        AlwaysCheapestRouter(),
        (example,),
        TierSpec(BudgetTier.FAST, Decimal("9"), 1.0),
    )

    assert result.queries[0].selected_model_id == "quoted-cheap"
    assert result.queries[0].cost == Decimal("9")
    assert result.queries[0].selected_call_index == 0
    assert result.queries[0].calls[0].quoted_cost == Decimal("0.1")
    assert result.queries[0].calls[0].realized_cost == Decimal("9")
    assert result.queries[0].calls[0].remaining_budget_before == Decimal("9")
    assert result.queries[0].calls[0].remaining_budget_after == Decimal(0)
    assert result.queries[0].calls[0].within_budget


def test_unaffordable_quote_fails_before_a_cheaper_realized_call_is_attempted() -> None:
    example = EvaluationExample(
        "quote-too-high",
        "prompt",
        "general",
        (CandidateOutcome("model", "answer", Decimal("0.1"), 0.9),),
        (ModelSpec("model", Decimal("2")),),
    )

    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(
        AlwaysPremiumRouter("model"),
        (example,),
        TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),
    )

    assert not result.feasible
    assert result.queries[0].calls == ()
    assert result.queries[0].cost == Decimal(0)
    assert result.budget.spent == Decimal(0)
    assert result.budget.over_budget_calls == 0


def test_affordable_overquote_records_only_the_lower_realized_charge() -> None:
    example = EvaluationExample(
        "overquote",
        "prompt",
        "general",
        (CandidateOutcome("model", "answer", Decimal("0.2"), 0.9),),
        (ModelSpec("model", Decimal("0.8")),),
    )

    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(
        AlwaysPremiumRouter("model"),
        (example,),
        TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),
    )

    query = result.queries[0]
    assert query.feasible
    assert query.cost == result.budget.spent == Decimal("0.2")
    assert query.calls[0].quoted_cost == Decimal("0.8")
    assert query.calls[0].realized_cost == Decimal("0.2")
    assert query.calls[0].within_budget


def test_zero_cost_call_remains_visible_and_selectable() -> None:
    example = EvaluationExample(
        "zero",
        "prompt",
        "general",
        (CandidateOutcome("free", "answer", Decimal(0), 0.5),),
        (ModelSpec("free", Decimal(0)),),
    )

    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(
        AlwaysCheapestRouter(),
        (example,),
        TierSpec(BudgetTier.FAST, Decimal(0), 1.0),
    )

    query = result.queries[0]
    assert query.feasible
    assert query.selected_call_index == 0
    assert len(query.calls) == 1
    assert query.calls[0].quoted_cost == query.calls[0].realized_cost == Decimal(0)
    assert query.calls[0].within_budget


def test_multi_call_replay_records_each_charge_and_selected_call_index() -> None:
    class TwoCallsThenFirst:
        def route(self, state: object) -> CallModel | SelectOutput:
            history = state.call_history  # type: ignore[attr-defined]
            if not history:
                return CallModel("cheap", reason="first")
            if len(history) == 1:
                return CallModel("premium", reason="second")
            return SelectOutput(0, reason="keep first")

    result = OfflineSimulator(PerQueryBudgetLedger, max_calls_per_query=2).run_tier(
        TwoCallsThenFirst(),
        EXAMPLES[:1],
        TierSpec(BudgetTier.FAST, Decimal("3"), 1.0),
    )

    query = result.queries[0]
    assert query.feasible
    assert query.selected_model_id == "cheap"
    assert query.selected_call_index == 0
    assert [call.model_id for call in query.calls] == ["cheap", "premium"]
    assert [call.quoted_cost for call in query.calls] == [Decimal("1"), Decimal("2")]
    assert [call.realized_cost for call in query.calls] == [Decimal("1"), Decimal("2")]
    assert [call.remaining_budget_before for call in query.calls] == [
        Decimal("3"),
        Decimal("2"),
    ]
    assert [call.remaining_budget_after for call in query.calls] == [
        Decimal("2"),
        Decimal(0),
    ]
    assert query.cost == Decimal("3")


def test_realized_overspend_is_recorded_and_exhausts_cumulative_budget() -> None:
    models = (ModelSpec("cheap", Decimal("0.1")),)
    examples = tuple(
        EvaluationExample(
            f"q{index}",
            f"prompt {index}",
            "general",
            (CandidateOutcome("cheap", "answer", Decimal("9"), 0.5),),
            models,
        )
        for index in (1, 2)
    )

    result = OfflineSimulator(CumulativeBudgetLedger).run_tier(
        AlwaysCheapestRouter(),
        examples,
        TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),
    )

    assert result.feasible is False
    assert result.queries[0].cost == Decimal("9")
    assert len(result.queries[0].calls) == 1
    assert result.queries[0].calls[0].quoted_cost == Decimal("0.1")
    assert result.queries[0].calls[0].realized_cost == Decimal("9")
    assert not result.queries[0].calls[0].within_budget
    assert "reported realized charge 9 out of budget" in (result.queries[0].error or "")
    assert result.queries[1].cost == Decimal(0)
    assert result.queries[1].calls == ()
    assert result.budget.spent == Decimal("9")
    assert result.budget.over_budget_calls == 1


def test_simulator_costs_do_not_depend_on_decimal_context() -> None:
    model = ModelSpec("only", Decimal("0"))
    realized_costs = (Decimal("0.33333333333333333333333333333"),) * 3 + (Decimal("5e-29"),)
    examples = tuple(
        EvaluationExample(
            f"exact-{index}",
            f"prompt {index}",
            "general",
            (CandidateOutcome("only", "answer", cost, 0.5),),
            (model,),
        )
        for index, cost in enumerate(realized_costs)
    )

    with localcontext() as context:
        context.prec = 2
        result = OfflineSimulator(CumulativeBudgetLedger).run_tier(
            AlwaysCheapestRouter(),
            examples,
            TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),
        )

    assert [query.feasible for query in result.queries] == [True, True, True, False]
    assert result.queries[-1].cost == Decimal("5e-29")
    assert result.budget.spent == Decimal("1.00000000000000000000000000004")


def test_split_domain_is_not_exposed_without_explicit_router_metadata() -> None:
    class CapturingRouter:
        def __init__(self) -> None:
            self.metadata: list[dict[str, object]] = []

        def route(self, state: object) -> CallModel | SelectOutput:
            self.metadata.append(dict(state.metadata))  # type: ignore[attr-defined]
            if state.call_history:  # type: ignore[attr-defined]
                return SelectOutput(0)
            return CallModel("cheap")

    router = CapturingRouter()
    OfflineSimulator(PerQueryBudgetLedger).run_tier(router, EXAMPLES[:1], TIER)

    assert router.metadata
    assert all("domain" not in metadata for metadata in router.metadata)
    assert all("example_id" not in metadata for metadata in router.metadata)


def test_report_scope_uses_the_same_deeply_immutable_metadata_seen_by_router() -> None:
    router_metadata = {"domain": "math", "tags": ["reasoning"]}
    model_metadata = {"provider": {"name": "local"}}
    example = EvaluationExample(
        "immutable-metadata",
        "prompt",
        "split-only-domain",
        (CandidateOutcome("model", "answer", Decimal("1"), 0.8),),
        (ModelSpec("model", Decimal("1"), metadata=model_metadata),),
        router_metadata=router_metadata,
    )

    class MutationAttemptRouter:
        failures = 0

        def route(self, state: RouterState) -> CallModel | SelectOutput:
            if state.call_history:
                return SelectOutput(0)
            attempts = (
                lambda: state.metadata.__setitem__("domain", "changed"),
                lambda: state.metadata["tags"].append("changed"),  # type: ignore[union-attr]
                lambda: (
                    state.candidate_models[0]
                    .metadata["provider"]
                    .__setitem__(  # type: ignore[union-attr]
                        "name", "changed"
                    )
                ),
            )
            for attempt in attempts:
                try:
                    attempt()
                except (AttributeError, TypeError):
                    self.failures += 1
            return CallModel("model")

    router = MutationAttemptRouter()
    report = OfflineSimulator(PerQueryBudgetLedger).run(
        router,
        (example,),
        (TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),),
    )

    assert router.failures == 3
    assert router_metadata == {"domain": "math", "tags": ["reasoning"]}
    assert model_metadata == {"provider": {"name": "local"}}
    assert len(report.evaluation_scope_sha256) == 64
    assert report.max_calls_per_query == 1


def test_tier_budget_representation_is_normalized_before_routing_and_hashing() -> None:
    class RepresentationSensitiveRouter:
        def __init__(self) -> None:
            self.exponents: list[int] = []

        def route(self, state: RouterState) -> CallModel | SelectOutput:
            if state.call_history:
                return SelectOutput(0)
            exponent = state.remaining_budget.as_tuple().exponent
            assert isinstance(exponent, int)
            self.exponents.append(exponent)
            return CallModel("cheap" if exponent == 0 else "premium")

    simulator = OfflineSimulator(PerQueryBudgetLedger)
    plain_router = RepresentationSensitiveRouter()
    padded_router = RepresentationSensitiveRouter()
    plain = simulator.run(
        plain_router,
        EXAMPLES[:1],
        (TierSpec(BudgetTier.FAST, Decimal("2"), 1),),
    )
    padded = simulator.run(
        padded_router,
        EXAMPLES[:1],
        (TierSpec(BudgetTier.FAST, Decimal("2.00"), 1.0),),
    )

    assert plain.evaluation_scope == padded.evaluation_scope
    assert plain_router.exponents == padded_router.exponents == [0]
    assert plain.tiers[0].queries[0].selected_model_id == "cheap"
    assert padded.tiers[0].queries[0].selected_model_id == "cheap"
    assert plain.tiers[0].tier_spec.budget_limit.as_tuple().exponent == 0
    assert type(plain.tiers[0].tier_spec.weight) is float


def test_binary64_equivalent_integer_labels_have_one_oracle_replay_semantics() -> None:
    models = (ModelSpec("a", Decimal("1")), ModelSpec("b", Decimal("1")))

    def example(a_quality: int, b_quality: int) -> EvaluationExample:
        return EvaluationExample(
            "large-label",
            "prompt",
            "domain",
            (
                CandidateOutcome("a", "a-output", Decimal("1"), a_quality),
                CandidateOutcome("b", "b-output", Decimal("1"), b_quality),
            ),
            models,
        )

    spec = TierSpec(BudgetTier.FAST, Decimal("1"), 1.0)
    simulator = OfflineSimulator(PerQueryBudgetLedger)
    left = simulator._prepare_evaluation((example(2**53, 2**53 + 1),), (spec,))
    right = simulator._prepare_evaluation((example(2**53 + 1, 2**53),), (spec,))
    left_plan = build_per_query_oracle_plan(left.examples, left.tier_specs)
    right_plan = build_per_query_oracle_plan(right.examples, right.tier_specs)
    left_report = simulator._run_prepared(OracleRouter(left_plan), left, router_name="oracle")
    right_report = simulator._run_prepared(OracleRouter(right_plan), right, router_name="oracle")

    assert left.identity == right.identity
    assert all(type(outcome.quality) is float for outcome in left.examples[0].outcomes)
    assert left_plan == right_plan
    assert left_report.tiers[0].queries[0].selected_model_id == "a"
    assert right_report.tiers[0].queries[0].selected_model_id == "a"
    assert weighted_delta(left_report, right_report) == 0.0


def test_invalid_forged_metadata_fails_before_the_router_is_invoked() -> None:
    class CountingRouter:
        calls = 0

        def route(self, state: RouterState) -> CallModel:
            self.calls += 1
            return CallModel("cheap")

    forged = provenance_module._FrozenMetadata((("bad", object()),))
    example = EvaluationExample(
        "forged",
        "prompt",
        "domain",
        EXAMPLES[0].outcomes,
        EXAMPLES[0].candidate_models,
        forged,
    )
    router = CountingRouter()

    with pytest.raises(TypeError, match="unsupported type object"):
        OfflineSimulator(PerQueryBudgetLedger).run(router, (example,), (TIER,))
    assert router.calls == 0


def test_privileged_context_requires_nominal_oracle_marker() -> None:
    class MethodNameCollisionRouter:
        def route_with_evaluation_context(self, state: object, *, example_id: str) -> CallModel:
            raise AssertionError(f"ordinary router received private ID {example_id}")

        def route(self, state: object) -> CallModel | SelectOutput:
            if state.call_history:  # type: ignore[attr-defined]
                return SelectOutput(0)
            return CallModel("cheap")

    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(
        MethodNameCollisionRouter(), EXAMPLES[:1], TIER
    )

    assert result.feasible is True
