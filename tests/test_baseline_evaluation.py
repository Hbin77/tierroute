# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for leakage-free six-baseline outer-LODO replay."""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal

import pytest

import tierroute.eval.simulator as simulator_module
import tierroute.policies.baseline_evaluation as baseline_evaluation_module
from tierroute.adapters import CumulativeBudgetLedger, EvaluationDataset, PerQueryBudgetLedger
from tierroute.core import BudgetTier, ModelSpec
from tierroute.demo import evaluate_six_baselines
from tierroute.eval import (
    BudgetLedger,
    CandidateOutcome,
    EvaluationExample,
    TierSpec,
    summarize_report,
)
from tierroute.policies import (
    BASELINE_CONFIG_EVIDENCE_HASH_ALGORITHM,
    BASELINE_NAMES,
    LodoSixBaselineEvaluation,
    evaluate_per_query_lodo_baselines,
)

MODELS = (
    ModelSpec("cheap", Decimal("1")),
    ModelSpec("strong", Decimal("2")),
    ModelSpec("premium", Decimal("4")),
)
SPECS = (
    TierSpec(BudgetTier.FAST, Decimal("2"), 0.7),
    TierSpec(BudgetTier.PREMIUM, Decimal("4"), 0.3),
)


def _example(
    example_id: str,
    split_domain: str,
    *,
    observable_tag: str | None,
    qualities: tuple[float, float, float] = (0.4, 0.8, 0.9),
    reverse_catalogue: bool = False,
    prompt: str = "shared prompt",
    realized_costs: tuple[str, str, str] = ("1", "2", "4"),
) -> EvaluationExample:
    metadata = {} if observable_tag is None else {"domain": observable_tag}
    return EvaluationExample(
        example_id=example_id,
        prompt=prompt,
        domain=split_domain,
        outcomes=tuple(
            CandidateOutcome(model.model_id, model.model_id, Decimal(cost), quality)
            for model, cost, quality in zip(MODELS, realized_costs, qualities, strict=True)
        ),
        candidate_models=tuple(reversed(MODELS)) if reverse_catalogue else MODELS,
        router_metadata=metadata,
    )


def _interleaved_examples() -> tuple[EvaluationExample, ...]:
    """Use different split domains and shared observable tags to expose leakage."""

    return (
        _example("a1", "split-a", observable_tag="math"),
        _example("b1", "split-b", observable_tag="math", reverse_catalogue=True),
        _example("a2", "split-a", observable_tag="math", reverse_catalogue=True),
        _example("c1", "split-c", observable_tag=None),
        _example("b2", "split-b", observable_tag="math"),
        _example("c2", "split-c", observable_tag=None, reverse_catalogue=True),
    )


def _evaluate(examples: tuple[EvaluationExample, ...]) -> LodoSixBaselineEvaluation:
    return evaluate_per_query_lodo_baselines(
        examples,
        SPECS,
        PerQueryBudgetLedger,
        premium_model_id="premium",
        strong_model_id="strong",
        random_seed=2026,
        character_threshold=20,
    )


def _selected_models(
    evaluation: LodoSixBaselineEvaluation,
    name: str,
) -> dict[tuple[BudgetTier, str], str | None]:
    result = evaluation.by_name()[name]
    return {
        (tier.tier_spec.tier, query.example_id): query.selected_model_id
        for tier in result.report.tiers
        for query in tier.queries
    }


def test_six_baselines_share_one_original_order_outer_lodo_population() -> None:
    examples = _interleaved_examples()
    evaluation = _evaluate(examples)
    expected_ids = tuple(example.example_id for example in examples)

    assert evaluation.accounting_scope == "per-query"
    assert evaluation.example_ids == expected_ids
    assert (
        evaluation.cheap_model_id,
        evaluation.premium_model_id,
        evaluation.strong_model_id,
        evaluation.random_seed,
        evaluation.character_threshold,
    ) == ("cheap", "premium", "strong", 2026, 20)
    assert evaluation.baseline_config_evidence_algorithm == BASELINE_CONFIG_EVIDENCE_HASH_ALGORITHM
    assert len(evaluation.baseline_config_evidence_sha256) == 64
    assert tuple(result.name for result in evaluation.baselines) == BASELINE_NAMES
    assert {fold.held_out_domain for fold in evaluation.folds} == {
        "split-a",
        "split-b",
        "split-c",
    }
    assert sorted(
        example_id for fold in evaluation.folds for example_id in fold.test_example_ids
    ) == sorted(expected_ids)

    for fold in evaluation.folds:
        assert set(fold.training_example_ids).isdisjoint(fold.test_example_ids)
        assert set(fold.training_example_ids) | set(fold.test_example_ids) == set(expected_ids)
        assert {entry.observable_domain_tag for entry in fold.fitted_domain_table_entries} <= {
            "math"
        }
    for result in evaluation.baselines:
        for tier_result, spec in zip(result.report.tiers, SPECS, strict=True):
            assert tier_result.tier_spec == spec
            assert tuple(query.example_id for query in tier_result.queries) == expected_ids
            assert tier_result.budget.query_order == expected_ids
            assert tier_result.budget.adapter_name == "per-query"

    domain_choices = _selected_models(evaluation, "domain-best-table")
    for example_id in ("a1", "a2", "b1", "b2"):
        assert domain_choices[(BudgetTier.FAST, example_id)] == "strong"
        assert domain_choices[(BudgetTier.PREMIUM, example_id)] == "premium"
    for example_id in ("c1", "c2"):
        assert domain_choices[(BudgetTier.FAST, example_id)] == "cheap"
        assert domain_choices[(BudgetTier.PREMIUM, example_id)] == "cheap"


def test_baseline_row_rejects_quote_evidence_from_another_report() -> None:
    rows = _evaluate(_interleaved_examples()).by_name()

    with pytest.raises(ValueError, match="derived from its replay report"):
        replace(rows["always-cheapest"], quote_error=rows["always-premium"].quote_error)


def test_baseline_row_recomputes_score_name_cost_and_gap_representation() -> None:
    rows = _evaluate(_interleaved_examples()).by_name()
    cheapest = rows["always-cheapest"]

    with pytest.raises(ValueError, match="score must be derived"):
        replace(cheapest, score=rows["always-premium"].score)
    with pytest.raises(ValueError, match="name must match"):
        replace(cheapest, name="renamed")
    with pytest.raises(TypeError, match="Decimal"):
        replace(cheapest, total_cost=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="float or None"):
        replace(cheapest, gap_recovery=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        replace(cheapest, gap_recovery=float("nan"))


def test_suite_rejects_mixed_scope_or_fabricated_derived_gap() -> None:
    evaluation = _evaluate(_interleaved_examples())
    changed_examples = tuple(
        replace(
            example,
            outcomes=tuple(
                replace(outcome, quality=outcome.quality + 0.01) for outcome in example.outcomes
            ),
        )
        for example in _interleaved_examples()
    )
    changed = _evaluate(changed_examples)
    mixed_rows = list(evaluation.baselines)
    mixed_rows[2] = changed.baselines[2]

    with pytest.raises(ValueError, match="share one evaluation scope"):
        replace(evaluation, baselines=tuple(mixed_rows))

    wrong_rows = list(evaluation.baselines)
    random_row = wrong_rows[2]
    assert random_row.gap_recovery is not None
    wrong_rows[2] = replace(
        random_row,
        gap_recovery=math.nextafter(random_row.gap_recovery, math.inf),
    )
    with pytest.raises(ValueError, match="gap_recovery must be derived"):
        replace(evaluation, baselines=tuple(wrong_rows))

    missing_gap = list(evaluation.baselines)
    missing_gap[2] = replace(random_row, gap_recovery=None)
    with pytest.raises(ValueError, match="gap_recovery must be derived"):
        replace(evaluation, baselines=tuple(missing_gap))


def test_suite_requires_canonical_rows_population_and_fold_partition() -> None:
    evaluation = _evaluate(_interleaved_examples())

    with pytest.raises(ValueError, match="canonical names and order"):
        replace(evaluation, baselines=tuple(reversed(evaluation.baselines)))
    with pytest.raises(ValueError, match="canonical names and order"):
        replace(evaluation, baselines=evaluation.baselines[:-1])
    with pytest.raises(ValueError, match="example_ids must be unique"):
        replace(evaluation, example_ids=(*evaluation.example_ids[:-1], evaluation.example_ids[0]))
    bad_fold = replace(
        evaluation.folds[0],
        test_example_ids=(evaluation.example_ids[0],),
    )
    with pytest.raises(ValueError, match="partition"):
        replace(evaluation, folds=(bad_fold, *evaluation.folds[1:]))


def test_suite_binds_fold_scope_models_table_keys_and_tiers() -> None:
    evaluation = _evaluate(_interleaved_examples())
    changed = _evaluate(
        tuple(
            replace(example, domain=f"{example.domain}-changed")
            for example in _interleaved_examples()
        )
    )
    with pytest.raises(ValueError, match=r"fold evidence.*evaluation scope"):
        replace(evaluation, folds=(changed.folds[0], *evaluation.folds[1:]))

    fold = evaluation.folds[0]
    with pytest.raises(ValueError, match="candidate model IDs"):
        replace(
            evaluation,
            folds=(replace(fold, fallback_model_id="not-a-candidate"), *evaluation.folds[1:]),
        )

    entry = fold.fitted_domain_table_entries[0]
    duplicate_entries = (entry, entry, *fold.fitted_domain_table_entries[1:])
    with pytest.raises(ValueError, match="table keys must be unique"):
        replace(
            evaluation,
            folds=(
                replace(fold, fitted_domain_table_entries=duplicate_entries),
                *evaluation.folds[1:],
            ),
        )

    with pytest.raises(ValueError, match="tiers must exist"):
        replace(
            evaluation,
            folds=(
                replace(
                    fold,
                    fitted_domain_table_entries=(
                        replace(entry, tier=BudgetTier.BALANCED),
                        *fold.fitted_domain_table_entries[1:],
                    ),
                ),
                *evaluation.folds[1:],
            ),
        )


def test_suite_revalidates_the_per_query_oracle_upper_bound() -> None:
    evaluation = _evaluate(_interleaved_examples())
    rows = list(evaluation.baselines)
    oracle_index = tuple(result.name for result in rows).index("oracle")
    oracle = rows[oracle_index]
    tier = oracle.report.tiers[0]
    lowered_query = replace(tier.queries[0], quality=0.0)
    lowered_tier = replace(tier, queries=(lowered_query, *tier.queries[1:]))
    lowered_report = replace(oracle.report, tiers=(lowered_tier, *oracle.report.tiers[1:]))
    rows[oracle_index] = replace(
        oracle,
        report=lowered_report,
        score=summarize_report(lowered_report),
    )

    with pytest.raises(ValueError, match="exceeds the per-query oracle"):
        replace(evaluation, baselines=tuple(rows))


def test_six_baselines_prepare_and_hash_the_replay_scope_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_calls = 0
    hash_calls = 0
    original_snapshot = simulator_module._snapshot_evaluation_scope
    original_hash = simulator_module._evaluation_scope_sha256_from_snapshots

    def counted_snapshot(*args: object, **kwargs: object):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(*args, **kwargs)

    def counted_hash(*args: object, **kwargs: object):
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(*args, **kwargs)

    monkeypatch.setattr(simulator_module, "_snapshot_evaluation_scope", counted_snapshot)
    monkeypatch.setattr(
        simulator_module,
        "_evaluation_scope_sha256_from_snapshots",
        counted_hash,
    )

    _evaluate(_interleaved_examples())

    assert snapshot_calls == 1
    assert hash_calls == 1


def test_held_out_outcomes_cannot_change_that_folds_domain_decisions() -> None:
    original = _interleaved_examples()
    poisoned = tuple(
        replace(
            example,
            outcomes=tuple(
                replace(outcome, quality=quality)
                for outcome, quality in zip(
                    example.outcomes,
                    (1.0, 0.1, 0.0),
                    strict=True,
                )
            ),
        )
        if example.domain == "split-a"
        else example
        for example in original
    )

    first = _evaluate(original)
    second = _evaluate(poisoned)
    for name in ("always-cheapest", "always-premium", "random", "length-heuristic"):
        assert _selected_models(first, name) == _selected_models(second, name)

    first_domain = _selected_models(first, "domain-best-table")
    second_domain = _selected_models(second, "domain-best-table")
    for tier in (BudgetTier.FAST, BudgetTier.PREMIUM):
        for example_id in ("a1", "a2"):
            assert first_domain[(tier, example_id)] == second_domain[(tier, example_id)]


def test_per_query_suite_rejects_cumulative_accounting_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"domain fitting ran: {args!r}/{kwargs!r}")

    monkeypatch.setattr(
        baseline_evaluation_module,
        "fit_per_query_domain_table",
        forbidden_fit,
    )
    with pytest.raises(ValueError, match="per-query ledger"):
        evaluate_per_query_lodo_baselines(
            _interleaved_examples(),
            SPECS,
            CumulativeBudgetLedger,
            premium_model_id="premium",
            strong_model_id="strong",
        )


def test_seeded_random_decisions_do_not_depend_on_replay_order() -> None:
    examples = tuple(
        replace(example, prompt=f"prompt {example.example_id}")
        for example in _interleaved_examples()
    )

    forward = _evaluate(examples)
    backward = _evaluate(tuple(reversed(examples)))

    assert _selected_models(forward, "random") == _selected_models(backward, "random")


def test_baseline_parameters_are_validated_before_replay() -> None:
    with pytest.raises(TypeError, match="random_seed"):
        evaluate_per_query_lodo_baselines(
            _interleaved_examples(),
            SPECS,
            PerQueryBudgetLedger,
            premium_model_id="premium",
            strong_model_id="strong",
            random_seed=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="character_threshold"):
        evaluate_per_query_lodo_baselines(
            _interleaved_examples(),
            SPECS,
            PerQueryBudgetLedger,
            premium_model_id="premium",
            strong_model_id="strong",
            character_threshold=20.5,  # type: ignore[arg-type]
        )

    evaluation = _evaluate(_interleaved_examples())
    with pytest.raises(ValueError, match="replay config evidence"):
        replace(evaluation, random_seed=7)
    with pytest.raises(ValueError, match="replay config evidence"):
        replace(evaluation, character_threshold=21)


def test_guard_wraps_replay_ledgers_even_if_factory_changes_after_preflight() -> None:
    class SwitchingFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, budget_limit: Decimal, expected_queries: int) -> BudgetLedger:
            self.calls += 1
            ledger_type = (
                PerQueryBudgetLedger if self.calls <= len(SPECS) else CumulativeBudgetLedger
            )
            return ledger_type(budget_limit, expected_queries)

    factory = SwitchingFactory()
    with pytest.raises(ValueError, match="per-query ledger"):
        evaluate_per_query_lodo_baselines(
            _interleaved_examples(),
            SPECS,
            factory,
            premium_model_id="premium",
            strong_model_id="strong",
        )

    assert factory.calls > len(SPECS)


def test_catalogue_reordering_is_allowed_but_quote_drift_is_rejected() -> None:
    examples = _interleaved_examples()

    assert _evaluate(examples).example_ids == tuple(example.example_id for example in examples)

    drifted_models = (
        MODELS[0],
        replace(MODELS[1], cost=Decimal("2.1")),
        MODELS[2],
    )
    drifted = (replace(examples[0], candidate_models=drifted_models), *examples[1:])
    with pytest.raises(ValueError, match="stable model catalogue"):
        _evaluate(drifted)


def test_demo_wrapper_also_accepts_catalogue_reordering() -> None:
    examples = _interleaved_examples()
    dataset = EvaluationDataset(
        name="catalogue-order-test",
        license="Apache-2.0",
        provenance="project-authored test fixture",
        domain_labels_are_observable=True,
        tier_specs=SPECS,
        examples=examples,
    )

    assert tuple(result.name for result in evaluate_six_baselines(dataset)) == BASELINE_NAMES


def test_duplicate_ids_are_rejected_while_duplicate_prompts_remain_distinct() -> None:
    examples = _interleaved_examples()
    assert len({example.prompt for example in examples}) == 1
    assert _evaluate(examples).example_ids == ("a1", "b1", "a2", "c1", "b2", "c2")

    duplicated = (*examples[:-1], replace(examples[-1], example_id="a1"))
    with pytest.raises(ValueError, match="unique example_id"):
        _evaluate(duplicated)


def test_quote_affordability_and_realized_overspend_remain_distinct() -> None:
    examples = (
        _example(
            "overspend",
            "a",
            observable_tag=None,
            prompt="this is a deliberately long prompt",
            qualities=(0.4, 1.0, 0.9),
            realized_costs=("1", "3", "4"),
        ),
        _example("ordinary", "b", observable_tag=None, prompt="short"),
    )
    evaluation = evaluate_per_query_lodo_baselines(
        examples,
        (SPECS[0],),
        PerQueryBudgetLedger,
        premium_model_id="premium",
        strong_model_id="strong",
        character_threshold=10,
    )
    by_name = evaluation.by_name()
    length_query = by_name["length-heuristic"].report.tiers[0].queries[0]
    oracle_query = by_name["oracle"].report.tiers[0].queries[0]

    assert not length_query.feasible
    assert length_query.selected_model_id is None
    assert length_query.cost == Decimal("3")
    assert "reported realized charge 3 out of budget" in (length_query.error or "")
    assert oracle_query.feasible
    assert oracle_query.selected_model_id == "cheap"
    assert oracle_query.cost == Decimal("1")
