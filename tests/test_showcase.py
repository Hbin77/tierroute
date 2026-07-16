# SPDX-License-Identifier: Apache-2.0
"""Direct three-tier routing-stream evidence and fail-closed contracts."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

import tierroute.showcase as showcase_module
from tierroute.adapters.json_dataset import EvaluationDataset, load_evaluation_dataset
from tierroute.core.costs import sum_costs
from tierroute.core.schemas import BudgetTier
from tierroute.eval.schemas import EvaluationReport
from tierroute.eval.simulator import OfflineSimulator
from tierroute.policies.baselines import OracleRouter
from tierroute.policies.benchmark import (
    PerQueryNestedLodoBenchmark,
    evaluate_per_query_bilinear_benchmark,
)
from tierroute.showcase import (
    BUNDLED_STREAM_ASSIGNMENTS,
    RoutingStreamShowcase,
    StreamAssignment,
    build_routing_stream_showcase,
)


def _benchmark(dataset: EvaluationDataset) -> PerQueryNestedLodoBenchmark:
    return evaluate_per_query_bilinear_benchmark(
        dataset.examples,
        dataset.tier_specs,
        max_candidates_per_tier=257,
    )


def _replace_outcome(
    dataset: EvaluationDataset,
    *,
    example_id: str,
    model_id: str,
    cost: Decimal | None = None,
    quality: float | None = None,
) -> EvaluationDataset:
    examples = []
    for example in dataset.examples:
        if example.example_id != example_id:
            examples.append(example)
            continue
        outcomes = tuple(
            replace(
                outcome,
                cost=outcome.cost if cost is None else cost,
                quality=outcome.quality if quality is None else quality,
            )
            if outcome.model_id == model_id
            else outcome
            for outcome in example.outcomes
        )
        examples.append(replace(example, outcomes=outcomes))
    return replace(dataset, examples=tuple(examples))


@pytest.fixture(scope="module")
def showcase_evidence() -> tuple[
    EvaluationDataset,
    PerQueryNestedLodoBenchmark,
    RoutingStreamShowcase,
]:
    dataset = load_evaluation_dataset()
    benchmark = _benchmark(dataset)
    return dataset, benchmark, build_routing_stream_showcase(dataset, benchmark)


def test_bundled_showcase_stream_is_stable_and_covers_every_tier(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
) -> None:
    dataset, benchmark, showcase = showcase_evidence

    assert BUNDLED_STREAM_ASSIGNMENTS == (
        StreamAssignment("synthetic-science-001", BudgetTier.FAST),
        StreamAssignment("synthetic-math-002", BudgetTier.BALANCED),
        StreamAssignment("synthetic-code-002", BudgetTier.PREMIUM),
    )
    assert showcase.dataset is dataset
    assert showcase.benchmark is benchmark
    assert showcase.stream_id == "tierroute-bundled-three-tier-stream-v1"
    assert showcase.accounting_scope == "per-query"
    assert showcase.cost_aggregation_scope == "mixed-tier-reporting-only"
    assert showcase.data_sha256 == benchmark.data_sha256
    assert showcase.replay_sha256 == benchmark.replay_sha256
    assert showcase.tier_specs == dataset.tier_specs
    assert showcase.benchmark_evaluation_scope == benchmark.learned.report.evaluation_scope
    assert [
        (step.index, step.example_id, step.tier, step.selected_model_id) for step in showcase.steps
    ] == [
        (1, "synthetic-science-001", BudgetTier.FAST, "swift"),
        (2, "synthetic-math-002", BudgetTier.BALANCED, "steady"),
        (3, "synthetic-code-002", BudgetTier.PREMIUM, "expert"),
    ]
    assert {step.tier for step in showcase.steps} == {spec.tier for spec in dataset.tier_specs}
    assert [step.evaluation_scope.sha256 for step in showcase.steps] == [
        "e78396d6ddf7613059ea9a3fef2a345c65fc455fc2c76471b1a1ba7aff12e4f6",
        "2ccb5f61295594a86bccd5aa8b62d3606803968d370794339eaa615993b67037",
        "e3c36b67d8774a59bfcc05782995ee7be964b9445ca86b1d206b5b913a0a0e47",
    ]
    assert [step.lambda_cost for step in showcase.steps] == [
        Fraction(0),
        Fraction(0),
        Fraction(0),
    ]


def test_showcase_conserves_exact_realized_cost(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
) -> None:
    _, _, showcase = showcase_evidence

    assert [step.budget_limit for step in showcase.steps] == [
        Decimal("0.35"),
        Decimal("0.7"),
        Decimal("1"),
    ]
    assert [step.quoted_cost for step in showcase.steps] == [
        Decimal("0.2"),
        Decimal("0.6"),
        Decimal("1"),
    ]
    assert [step.realized_cost for step in showcase.steps] == [
        Decimal("0.2"),
        Decimal("0.6"),
        Decimal("1"),
    ]
    assert [step.cumulative_realized_cost for step in showcase.steps] == [
        Decimal("0.2"),
        Decimal("0.8"),
        Decimal("1.8"),
    ]
    assert showcase.total_realized_cost == Decimal("1.8")
    assert showcase.total_realized_cost == sum_costs(step.realized_cost for step in showcase.steps)


def test_showcase_replays_only_the_three_visible_rows_and_separate_oracles(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, benchmark, _ = showcase_evidence
    real_run = OfflineSimulator.run
    calls: list[tuple[str | None, int, int]] = []

    def capture_run(
        simulator: OfflineSimulator,
        router: object,
        examples: tuple[object, ...],
        tier_specs: tuple[object, ...],
        *,
        router_name: str | None = None,
    ) -> EvaluationReport:
        calls.append((router_name, len(examples), len(tier_specs)))
        return real_run(  # type: ignore[arg-type]
            simulator,
            router,
            examples,
            tier_specs,
            router_name=router_name,
        )

    monkeypatch.setattr(OfflineSimulator, "run", capture_run)
    build_routing_stream_showcase(dataset, benchmark)

    assert calls == [
        ("showcase-tier-lambda", 1, 1),
        ("showcase-per-query-oracle", 1, 1),
        ("showcase-tier-lambda", 1, 1),
        ("showcase-per-query-oracle", 1, 1),
        ("showcase-tier-lambda", 1, 1),
        ("showcase-per-query-oracle", 1, 1),
    ]


def test_showcase_running_cost_uses_realized_charge_not_quote() -> None:
    dataset = _replace_outcome(
        load_evaluation_dataset(),
        example_id="synthetic-science-001",
        model_id="swift",
        cost=Decimal("0.25"),
    )
    result = build_routing_stream_showcase(dataset, _benchmark(dataset))

    first = result.steps[0]
    assert first.quoted_cost == Decimal("0.2")
    assert first.realized_cost == Decimal("0.25")
    assert [step.cumulative_realized_cost for step in result.steps] == [
        Decimal("0.25"),
        Decimal("0.85"),
        Decimal("1.85"),
    ]
    assert result.total_realized_cost == Decimal("1.85")


def test_hidden_held_out_quality_does_not_change_the_ordinary_route(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
) -> None:
    _, _, original = showcase_evidence
    dataset = _replace_outcome(
        load_evaluation_dataset(),
        example_id="synthetic-science-001",
        model_id="swift",
        quality=0.1,
    )
    changed = build_routing_stream_showcase(dataset, _benchmark(dataset))

    assert changed.steps[0].selected_model_id == original.steps[0].selected_model_id
    assert changed.steps[0].predicted_quality == original.steps[0].predicted_quality
    assert changed.steps[0].observed_quality == 0.1


def test_showcase_oracle_dominance_and_exact_quality_retention(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
) -> None:
    _, _, showcase = showcase_evidence

    assert [step.oracle_model_id for step in showcase.steps] == ["swift", "steady", "expert"]
    assert [step.oracle_realized_cost for step in showcase.steps] == [
        Decimal("0.2"),
        Decimal("0.6"),
        Decimal("1"),
    ]
    assert [step.observed_quality for step in showcase.steps] == [0.78, 0.75, 0.96]
    assert [step.oracle_quality for step in showcase.steps] == [0.78, 0.75, 0.96]
    assert all(
        Fraction(str(step.observed_quality)) <= Fraction(str(step.oracle_quality))
        for step in showcase.steps
    )
    assert [step.cumulative_observed_quality for step in showcase.steps] == [
        Fraction(39, 50),
        Fraction(153, 100),
        Fraction(249, 100),
    ]
    assert [step.cumulative_oracle_quality for step in showcase.steps] == [
        Fraction(39, 50),
        Fraction(153, 100),
        Fraction(249, 100),
    ]
    assert all(step.cumulative_quality_retention == Fraction(1) for step in showcase.steps)
    assert showcase.total_observed_quality == Fraction(249, 100)
    assert showcase.total_oracle_quality == Fraction(249, 100)
    assert showcase.quality_retention == Fraction(1)
    assert showcase.quality_retention == (
        showcase.total_observed_quality / showcase.total_oracle_quality
    )

    zero_oracle = replace(
        showcase.steps[0],
        observed_quality=0.0,
        oracle_quality=0.0,
        cumulative_observed_quality=Fraction(0),
        cumulative_oracle_quality=Fraction(0),
    )
    assert zero_oracle.cumulative_quality_retention is None


def test_showcase_rejects_hash_replay_and_tier_spec_mismatches(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
) -> None:
    dataset, benchmark, _ = showcase_evidence
    first = dataset.examples[0]
    prompt_tampered = replace(
        dataset,
        examples=(replace(first, prompt=f"{first.prompt} tampered"), *dataset.examples[1:]),
    )
    with pytest.raises(ValueError, match="data hash"):
        build_routing_stream_showcase(prompt_tampered, benchmark)

    reordered = replace(dataset, examples=tuple(reversed(dataset.examples)))
    with pytest.raises(ValueError, match="replay hash"):
        build_routing_stream_showcase(reordered, benchmark)

    first_spec = dataset.tier_specs[0]
    tier_tampered = replace(
        dataset,
        tier_specs=(replace(first_spec, weight=0.49), *dataset.tier_specs[1:]),
    )
    with pytest.raises(ValueError, match="tier specs"):
        build_routing_stream_showcase(tier_tampered, benchmark)


def test_showcase_rejects_invalid_assignment_population(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
) -> None:
    dataset, benchmark, _ = showcase_evidence
    with pytest.raises(ValueError, match="unique"):
        build_routing_stream_showcase(
            dataset,
            benchmark,
            assignments=(
                BUNDLED_STREAM_ASSIGNMENTS[0],
                BUNDLED_STREAM_ASSIGNMENTS[0],
                *BUNDLED_STREAM_ASSIGNMENTS[1:],
            ),
        )
    with pytest.raises(ValueError, match="cover every configured tier"):
        build_routing_stream_showcase(
            dataset,
            benchmark,
            assignments=BUNDLED_STREAM_ASSIGNMENTS[:2],
        )
    with pytest.raises(ValueError, match="unknown example"):
        build_routing_stream_showcase(
            dataset,
            benchmark,
            assignments=(
                StreamAssignment("missing-example", BudgetTier.FAST),
                *BUNDLED_STREAM_ASSIGNMENTS[1:],
            ),
        )
    with pytest.raises(ValueError, match="exactly one row per tier"):
        build_routing_stream_showcase(
            dataset,
            benchmark,
            assignments=(
                *BUNDLED_STREAM_ASSIGNMENTS,
                StreamAssignment("synthetic-science-002", BudgetTier.FAST),
            ),
        )
    with pytest.raises(ValueError, match="versioned stream identity"):
        build_routing_stream_showcase(
            dataset,
            benchmark,
            assignments=(
                StreamAssignment("synthetic-general-001", BudgetTier.FAST),
                *BUNDLED_STREAM_ASSIGNMENTS[1:],
            ),
        )


def test_showcase_rejects_scope_and_direct_replay_tampering(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, benchmark, _ = showcase_evidence
    monkeypatch.setattr(
        showcase_module,
        "evaluation_scope_sha256",
        lambda *args, **kwargs: "0" * 64,
    )
    with pytest.raises(ValueError, match="dataset scope"):
        build_routing_stream_showcase(dataset, benchmark)
    monkeypatch.undo()

    real_run = OfflineSimulator.run

    def tamper_learned_report(
        simulator: OfflineSimulator,
        *args: object,
        **kwargs: object,
    ) -> EvaluationReport:
        report = real_run(simulator, *args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("router_name") != "showcase-tier-lambda":
            return report
        tier_result = report.tiers[0]
        query = tier_result.queries[0]
        tampered_query = replace(query, decision_reason=f"{query.decision_reason} tampered")
        return replace(report, tiers=(replace(tier_result, queries=(tampered_query,)),))

    monkeypatch.setattr(OfflineSimulator, "run", tamper_learned_report)
    with pytest.raises(ValueError, match="differs from benchmark evidence"):
        build_routing_stream_showcase(dataset, benchmark)
    monkeypatch.undo()

    def tamper_oracle_scope(
        simulator: OfflineSimulator,
        *args: object,
        **kwargs: object,
    ) -> EvaluationReport:
        report = real_run(simulator, *args, **kwargs)  # type: ignore[arg-type]
        router = args[0]
        if not isinstance(router, OracleRouter):
            return report
        scope = replace(report.evaluation_scope, sha256="0" * 64)
        return replace(report, evaluation_scope=scope)

    monkeypatch.setattr(OfflineSimulator, "run", tamper_oracle_scope)
    with pytest.raises(ValueError, match="scopes must match"):
        build_routing_stream_showcase(dataset, benchmark)


def test_showcase_rejects_cumulative_and_final_total_tampering(
    showcase_evidence: tuple[
        EvaluationDataset,
        PerQueryNestedLodoBenchmark,
        RoutingStreamShowcase,
    ],
) -> None:
    _, _, showcase = showcase_evidence
    reordered_steps = (
        replace(showcase.steps[1], index=1),
        replace(showcase.steps[0], index=2),
        showcase.steps[2],
    )
    with pytest.raises(ValueError, match="versioned bundled stream identity"):
        replace(showcase, steps=reordered_steps)
    tampered_first = replace(
        showcase.steps[0],
        cumulative_realized_cost=Decimal("0.3"),
    )
    with pytest.raises(ValueError, match="exactly conserved"):
        replace(showcase, steps=(tampered_first, *showcase.steps[1:]))
    with pytest.raises(ValueError, match="observed showcase quality"):
        replace(
            showcase.steps[0],
            observed_quality=showcase.steps[0].oracle_quality + 0.01,
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        replace(showcase.steps[0], observed_quality=-0.01)
    with pytest.raises(ValueError, match="must be non-negative"):
        replace(showcase.steps[0], oracle_quality=-0.01)
    with pytest.raises(ValueError, match="init=False"):
        replace(showcase, total_realized_cost=Decimal("9"))
