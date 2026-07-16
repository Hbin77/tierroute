# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity and fail-closed tests for the prepared policy bridge."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from decimal import Decimal

import pytest

import tierroute.policies.prepared_reference as prepared_module
from tierroute.adapters import (
    CumulativeBudgetLedger,
    PerQueryBudgetLedger,
    load_evaluation_dataset,
)
from tierroute.core import BudgetTier, ModelSpec
from tierroute.eval import TierSpec, leave_one_domain_out
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features.surface import SURFACE_DOMAIN_TAG_CATALOGUE
from tierroute.policies import (
    PreparedReferencePipelineResult,
    estimate_lambda_search,
    estimate_prepared_reference_pipeline,
    evaluate_per_query_bilinear_benchmark,
    evaluate_prepared_reference_pipeline,
    nested_lodo_lambda_evaluation,
)
from tierroute.predictors import fit_calibrated_bilinear
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.prepared_execution import (
    PreparedRawScoreBundle,
    build_prepared_coefficient_bundle,
    build_prepared_raw_score_bundle,
)
from tierroute.predictors.prepared_graph import build_prepared_nested_lodo_plan
from tierroute.predictors.prepared_store import (
    PreparedFeatureStore,
    build_prepared_domain_statistics,
    build_prepared_feature_store,
    prepared_fit_source_sha256,
)

_SURFACE_FEATURE_COUNT = 5 + len(SURFACE_DOMAIN_TAG_CATALOGUE)


@dataclass(frozen=True, slots=True)
class _PreparedFixture:
    examples: tuple[EvaluationExample, ...]
    tier_specs: tuple[TierSpec, ...]
    store: PreparedFeatureStore
    raw_scores: PreparedRawScoreBundle
    source_sha256: str


def _prepare(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
) -> _PreparedFixture:
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    model_count = len(examples[0].candidate_models)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=_SURFACE_FEATURE_COUNT,
        target_count=model_count,
    )
    source_sha256 = prepared_fit_source_sha256(examples, plan)
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=source_sha256,
    )
    coefficients = build_prepared_coefficient_bundle(
        store,
        build_prepared_domain_statistics(store),
        ridge=1.0,
    )
    return _PreparedFixture(
        examples=examples,
        tier_specs=tier_specs,
        store=store,
        raw_scores=build_prepared_raw_score_bundle(store, coefficients),
        source_sha256=source_sha256,
    )


def _evaluate(fixture: _PreparedFixture) -> PreparedReferencePipelineResult:
    return evaluate_prepared_reference_pipeline(
        fixture.examples,
        fixture.tier_specs,
        fixture.store,
        fixture.raw_scores,
        PerQueryBudgetLedger,
        expected_source_fit_sha256=fixture.source_sha256,
        expected_store_sha256=fixture.store.sha256,
        expected_raw_score_sha256=fixture.raw_scores.sha256,
        max_candidates_per_tier=257,
    )


@pytest.fixture(scope="module")
def bundled_fixture() -> _PreparedFixture:
    dataset = load_evaluation_dataset()
    return _prepare(tuple(dataset.examples), tuple(dataset.tier_specs))


@pytest.fixture(scope="module")
def bundled_result(bundled_fixture: _PreparedFixture) -> PreparedReferencePipelineResult:
    return _evaluate(bundled_fixture)


_SEVEN_DOMAIN_ROWS = (
    ("row-07", "golf", "Explain a general topic briefly."),
    ("row-02", "bravo", "Prove the equation x^2 + y^2 = 1."),
    ("row-05", "echo", "Review a legal court statute."),
    ("row-01", "alpha", "Debug Python code:\nprint('hello')"),
    ("row-06", "foxtrot", "Assess a clinical medicine diagnosis."),
    ("row-03", "charlie", "Analyze finance revenue and investment."),
    ("row-04", "delta", "Describe a science experiment."),
    ("row-12", "golf", "Summarize another general topic."),
    ("row-08", "alpha", "Write a Rust sorting algorithm."),
    ("row-11", "echo", "Compare a contract and legal precedent."),
    ("row-09", "charlie", "Explain accounting and stock revenue."),
    ("row-10", "charlie", "Assess an investment portfolio."),
)

_SEVEN_DOMAIN_QUALITIES = (
    (0.301, 0.188),
    (0.437, 0.232),
    (0.157, 0.441),
    (0.880, 0.780),
    (0.750, 0.289),
    (0.556, 0.335),
    (0.247, 0.190),
    (0.282, 0.888),
    (0.805, 0.786),
    (0.780, 0.264),
    (0.363, 0.633),
    (0.722, 0.826),
)


def _seven_domain_examples() -> tuple[EvaluationExample, ...]:
    rows = []
    for (example_id, domain, prompt), (cheap_quality, premium_quality) in zip(
        _SEVEN_DOMAIN_ROWS,
        _SEVEN_DOMAIN_QUALITIES,
        strict=True,
    ):
        rows.append(
            EvaluationExample(
                example_id=example_id,
                prompt=prompt,
                domain=domain,
                candidate_models=(
                    ModelSpec("premium", Decimal("2")),
                    ModelSpec("cheap", Decimal("1")),
                ),
                outcomes=(
                    CandidateOutcome(
                        "cheap",
                        f"{example_id}:cheap",
                        Decimal("1"),
                        cheap_quality,
                    ),
                    CandidateOutcome(
                        "premium",
                        f"{example_id}:premium",
                        Decimal("2"),
                        premium_quality,
                    ),
                ),
            )
        )
    return tuple(rows)


def _generated_domain_examples(
    domain_count: int,
    *,
    rows_per_domain: int = 1,
    duplicate_prompt: bool = False,
    constant_quality: bool = False,
    quoted_realized_mismatch: bool = False,
) -> tuple[EvaluationExample, ...]:
    rows = []
    for domain_index in range(domain_count):
        for row_index in range(rows_per_domain):
            index = domain_index * rows_per_domain + row_index
            cheap_quality = 0.5 if constant_quality else 0.2 + 0.04 * index
            premium_quality = 0.5 if constant_quality else 0.85 - 0.025 * index
            rows.append(
                EvaluationExample(
                    example_id=f"generated-{domain_index}-{row_index}",
                    prompt=(
                        "identical constant prompt"
                        if duplicate_prompt
                        else f"Explain generated domain {domain_index}, row {row_index}."
                    ),
                    domain=f"domain-{domain_index}",
                    candidate_models=(
                        ModelSpec("premium", Decimal("2")),
                        ModelSpec("cheap", Decimal("1")),
                    ),
                    outcomes=(
                        CandidateOutcome(
                            "cheap",
                            "cheap output",
                            Decimal("0.25") if quoted_realized_mismatch else Decimal("1"),
                            cheap_quality,
                        ),
                        CandidateOutcome(
                            "premium",
                            "premium output",
                            Decimal("2.5") if quoted_realized_mismatch else Decimal("2"),
                            premium_quality,
                        ),
                    ),
                )
            )
    return tuple(rows)


def test_bundled_prepared_pipeline_equals_authoritative_rowwise_result(
    bundled_fixture: _PreparedFixture,
    bundled_result: PreparedReferencePipelineResult,
) -> None:
    reference = evaluate_per_query_bilinear_benchmark(
        bundled_fixture.examples,
        bundled_fixture.tier_specs,
        max_candidates_per_tier=257,
    )
    result = bundled_result

    assert result.learned == reference.learned
    assert result.source_fit_sha256 == (
        "ffb9f23e22219fce3941075b3de40cf6174258ce676c637b05a530333d03bc33"
    )
    # Generated numeric identities are same-runtime evidence. The rowwise equality
    # above is the portable assertion; trusted fixture values must still be preserved.
    assert result.store_sha256 == bundled_fixture.store.sha256
    assert result.raw_score_bundle_sha256 == bundled_fixture.raw_scores.sha256
    assert result.evaluation_data_sha256 == reference.data_sha256
    assert result.evaluation_replay_sha256 == reference.replay_sha256
    assert result.all_searches_exhaustive is True
    assert result.ridge == 1.0
    assert result.solver_id == bundled_fixture.raw_scores.coefficients.blocks[0].solver_id
    assert result.scorer_id == bundled_fixture.raw_scores.blocks[0].scorer_id
    assert result.embedding_dimension == 0
    assert result.embedding_identity is None
    assert result.coefficient_block_sha256s == tuple(
        block.sha256 for block in bundled_fixture.raw_scores.coefficients.blocks
    )
    assert result.scored_feature_shard_sha256s == tuple(
        shard.sha256 for shard in bundled_fixture.raw_scores.feature_shards.shards
    )
    assert result.raw_score_block_sha256s == tuple(
        block.sha256 for block in bundled_fixture.raw_scores.blocks
    )


def test_bundled_evidence_covers_exact_calibration_and_destination_graph(
    bundled_result: PreparedReferencePipelineResult,
) -> None:
    result = bundled_result
    plan = result.estimate.plan
    domain_count = len(plan.domains)
    model_count = plan.target_count
    expected_subset_indices = tuple(
        index
        for index, subset in enumerate(plan.training_subsets)
        if len(subset.domain_indices) in (domain_count - 2, domain_count - 1)
    )

    assert len(result.target_shards) == domain_count
    assert len(result.calibrations) == 10
    assert len(result.calibrated_score_blocks) == 16
    assert tuple(item.training_subset_index for item in result.calibrations) == (
        expected_subset_indices
    )
    assert sum(item.calibration_example_count for item in result.calibrations) == (
        result.estimate.calibration_row_memberships
    )
    assert all(len(item.calibrators) == model_count for item in result.calibrations)
    assert all(
        item.training_domain_indices
        == plan.training_subsets[item.training_subset_index].domain_indices
        for item in result.calibrations
    )
    block_lookup = {
        (block.training_subset_index, block.scored_domain_index): index
        for index, block in enumerate(plan.score_blocks)
    }
    subset_lookup = {
        subset.domain_indices: index for index, subset in enumerate(plan.training_subsets)
    }
    for calibration in result.calibrations:
        expected_raw_indices = tuple(
            block_lookup[
                (
                    subset_lookup[
                        tuple(
                            item
                            for item in calibration.training_domain_indices
                            if item != calibration_domain
                        )
                    ],
                    calibration_domain,
                )
            ]
            for calibration_domain in calibration.training_domain_indices
        )
        assert calibration.raw_score_block_indices == expected_raw_indices


def test_modeled_row_reads_equal_actual_reference_accesses(
    bundled_fixture: _PreparedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"score": 0, "target": 0}
    original_score = prepared_module.PreparedRawScoreBlock.score_row
    original_target = prepared_module.PreparedFeatureStore.target_row

    def counted_score(
        block: prepared_module.PreparedRawScoreBlock,
        row_index: int,
    ) -> tuple[float, ...]:
        counts["score"] += 1
        return original_score(block, row_index)

    def counted_target(
        store: prepared_module.PreparedFeatureStore,
        row_index: int,
    ) -> tuple[float, ...]:
        counts["target"] += 1
        return original_target(store, row_index)

    monkeypatch.setattr(prepared_module.PreparedRawScoreBlock, "score_row", counted_score)
    monkeypatch.setattr(prepared_module.PreparedFeatureStore, "target_row", counted_target)
    result = _evaluate(bundled_fixture)

    assert counts == {
        "score": result.estimate.raw_score_row_reads,
        "target": result.estimate.target_row_reads,
    }
    lambda_estimates = result.estimate.lambda_search_estimates
    assert lambda_estimates is not None
    assert result.estimate.lambda_pair_scan_upper_bound == 5 * sum(
        estimate.pair_scan_occurrences for estimate in lambda_estimates
    )
    assert result.estimate.lambda_candidate_upper_bound == len(bundled_fixture.tier_specs) * sum(
        estimate.candidate_upper_bound for estimate in lambda_estimates
    )
    assert result.estimate.candidate_evidence_upper_bound_bytes == sum(
        estimate.estimated_policy_artifact_bytes for estimate in lambda_estimates
    )


def test_bounded_candidate_evidence_is_labeled_non_exhaustive(
    bundled_fixture: _PreparedFixture,
) -> None:
    result = evaluate_prepared_reference_pipeline(
        bundled_fixture.examples,
        bundled_fixture.tier_specs,
        bundled_fixture.store,
        bundled_fixture.raw_scores,
        PerQueryBudgetLedger,
        expected_source_fit_sha256=bundled_fixture.source_sha256,
        expected_store_sha256=bundled_fixture.store.sha256,
        expected_raw_score_sha256=bundled_fixture.raw_scores.sha256,
        max_candidates_per_tier=2,
    )
    reference = evaluate_per_query_bilinear_benchmark(
        bundled_fixture.examples,
        bundled_fixture.tier_specs,
        max_candidates_per_tier=2,
    )

    assert result.learned == reference.learned
    assert result.all_searches_exhaustive is False
    assert any(
        selection.candidates.strategy == "bounded-bottom-hash-v2"
        for fold in result.learned.folds
        for selection in fold.tuning.selections
    )
    assert all(
        selection.candidates.exhaustive or selection.candidates.strategy == "bounded-bottom-hash-v2"
        for fold in result.learned.folds
        for selection in fold.tuning.selections
    )


def test_budget_adapter_remains_outside_prepared_policy_core(
    bundled_fixture: _PreparedFixture,
) -> None:
    query_count = len(bundled_fixture.examples)
    cumulative_specs = tuple(
        replace(spec, budget_limit=spec.budget_limit * query_count)
        for spec in bundled_fixture.tier_specs
    )
    result = evaluate_prepared_reference_pipeline(
        bundled_fixture.examples,
        cumulative_specs,
        bundled_fixture.store,
        bundled_fixture.raw_scores,
        CumulativeBudgetLedger,
        expected_source_fit_sha256=bundled_fixture.source_sha256,
        expected_store_sha256=bundled_fixture.store.sha256,
        expected_raw_score_sha256=bundled_fixture.raw_scores.sha256,
        max_candidates_per_tier=257,
    )

    def train(training: tuple[EvaluationExample, ...]) -> object:
        return fit_calibrated_bilinear(training).build_predictor()

    reference = nested_lodo_lambda_evaluation(
        bundled_fixture.examples,
        cumulative_specs,
        train,
        CumulativeBudgetLedger,
        max_candidates_per_tier=257,
    )
    assert result.learned == reference
    assert all(tier.budget.adapter_name == "cumulative" for tier in result.learned.report.tiers)


def test_seven_domain_maximum_graph_matches_rowwise_end_to_end() -> None:
    examples = _seven_domain_examples()
    specs = (TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),)
    fixture = _prepare(examples, specs)
    result = _evaluate(fixture)
    reference = evaluate_per_query_bilinear_benchmark(
        examples,
        specs,
        max_candidates_per_tier=257,
    )

    assert len(fixture.store.plan.training_subsets) == 63
    assert len(fixture.raw_scores.blocks) == 154
    assert fixture.store.plan.work.score_row_memberships == 22 * len(examples)
    assert fixture.store.plan.work.scalar_score_count == 22 * len(examples) * 2
    assert len(result.calibrations) == 28
    assert len(result.calibrated_score_blocks) == 49
    assert result.estimate.calibrated_prediction_rows == 7 * len(examples)
    assert result.learned == reference.learned
    assert result.all_searches_exhaustive is True


@pytest.mark.parametrize("domain_count", (5, 6))
def test_intermediate_domain_graphs_match_rowwise_end_to_end(domain_count: int) -> None:
    examples = _generated_domain_examples(domain_count)
    specs = (TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),)
    fixture = _prepare(examples, specs)

    assert (
        _evaluate(fixture).learned
        == evaluate_per_query_bilinear_benchmark(
            examples,
            specs,
            max_candidates_per_tier=257,
        ).learned
    )


def test_constant_duplicate_prompt_ties_match_rowwise_end_to_end() -> None:
    examples = _generated_domain_examples(
        4,
        rows_per_domain=2,
        duplicate_prompt=True,
        constant_quality=True,
    )
    specs = (TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),)
    fixture = _prepare(examples, specs)
    result = _evaluate(fixture)
    reference = evaluate_per_query_bilinear_benchmark(
        examples,
        specs,
        max_candidates_per_tier=257,
    )

    assert result.learned == reference.learned
    assert all(
        len(calibrator.values) == 1
        for calibration in result.calibrations
        for calibrator in calibration.calibrators
    )


def test_one_ulp_quality_boundary_matches_rowwise_end_to_end() -> None:
    examples = list(
        _generated_domain_examples(
            4,
            rows_per_domain=2,
            duplicate_prompt=True,
            constant_quality=True,
        )
    )
    first = examples[0]
    examples[0] = replace(
        first,
        outcomes=(
            replace(first.outcomes[0], quality=math.nextafter(0.5, 1.0)),
            first.outcomes[1],
        ),
    )
    frozen_examples = tuple(examples)
    specs = (TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),)
    fixture = _prepare(frozen_examples, specs)

    assert (
        _evaluate(fixture).learned
        == evaluate_per_query_bilinear_benchmark(
            frozen_examples,
            specs,
            max_candidates_per_tier=257,
        ).learned
    )


def test_quoted_and_realized_cost_mismatch_matches_rowwise_replay() -> None:
    examples = _generated_domain_examples(4, quoted_realized_mismatch=True)
    specs = (TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),)
    fixture = _prepare(examples, specs)
    result = _evaluate(fixture)
    reference = evaluate_per_query_bilinear_benchmark(
        examples,
        specs,
        max_candidates_per_tier=257,
    )

    assert result.learned == reference.learned
    assert any(
        call.quoted_cost != call.realized_cost
        for tier in result.learned.report.tiers
        for query in tier.queries
        for call in query.calls
    )


@pytest.mark.parametrize("domain_count", (4, 5, 6, 7))
def test_estimate_closed_forms_for_every_reviewed_domain_count(domain_count: int) -> None:
    domains = tuple(f"domain-{index}" for index in range(domain_count))
    plan = build_prepared_nested_lodo_plan(
        domains,
        (1,) * domain_count,
        feature_count=_SURFACE_FEATURE_COUNT,
        target_count=2,
    )
    estimate = estimate_prepared_reference_pipeline(
        plan,
        tier_count=3,
        max_candidates_per_tier=257,
    )

    assert estimate.calibrated_subset_count == domain_count * (domain_count + 1) // 2
    assert estimate.calibration_row_memberships == (
        len(domains) * domain_count * (domain_count - 1) // 2
    )
    assert estimate.calibration_scalar_points == (estimate.calibration_row_memberships * 2)
    assert estimate.calibrated_score_block_count == domain_count**2
    assert estimate.calibrated_prediction_rows == len(domains) * domain_count
    assert estimate.raw_score_row_reads == (
        estimate.calibration_row_memberships + estimate.calibrated_prediction_rows
    )


def test_reversed_replay_order_keeps_fit_identity_and_matches_rowwise(
    bundled_fixture: _PreparedFixture,
) -> None:
    reversed_examples = tuple(reversed(bundled_fixture.examples))
    reversed_fixture = replace(bundled_fixture, examples=reversed_examples)
    result = _evaluate(reversed_fixture)
    reference = evaluate_per_query_bilinear_benchmark(
        reversed_examples,
        bundled_fixture.tier_specs,
        max_candidates_per_tier=257,
    )

    assert (
        prepared_fit_source_sha256(
            reversed_examples,
            bundled_fixture.store.plan,
        )
        == bundled_fixture.source_sha256
    )
    assert result.store_sha256 == bundled_fixture.store.sha256
    assert result.learned == reference.learned
    assert tuple(query.example_id for query in result.learned.report.tiers[0].queries) == tuple(
        example.example_id for example in reversed_examples
    )


def test_output_only_replay_mutation_reuses_fit_and_matches_rowwise(
    bundled_fixture: _PreparedFixture,
    bundled_result: PreparedReferencePipelineResult,
) -> None:
    first = bundled_fixture.examples[0]
    mutated_first = replace(
        first,
        outcomes=tuple(
            replace(outcome, output=f"changed:{outcome.output}") for outcome in first.outcomes
        ),
    )
    mutated_examples = (mutated_first, *bundled_fixture.examples[1:])
    mutated_fixture = replace(bundled_fixture, examples=mutated_examples)
    result = _evaluate(mutated_fixture)
    reference = evaluate_per_query_bilinear_benchmark(
        mutated_examples,
        bundled_fixture.tier_specs,
        max_candidates_per_tier=257,
    )

    assert (
        prepared_fit_source_sha256(
            mutated_examples,
            bundled_fixture.store.plan,
        )
        == bundled_fixture.source_sha256
    )
    assert result.learned == reference.learned
    assert result.learned.prediction_sha256 == bundled_result.learned.prediction_sha256
    assert result.learned.report.evaluation_scope_sha256 != (
        bundled_result.learned.report.evaluation_scope_sha256
    )
    assert tuple(
        selection.lambda_cost
        for fold in result.learned.folds
        for selection in fold.tuning.selections
    ) == tuple(
        selection.lambda_cost
        for fold in bundled_result.learned.folds
        for selection in fold.tuning.selections
    )


def test_outer_target_mutation_cannot_change_heldout_policy_or_local_lineage(
    bundled_fixture: _PreparedFixture,
    bundled_result: PreparedReferencePipelineResult,
) -> None:
    held_out_domain = bundled_fixture.store.plan.domains[0]
    example_index = next(
        index
        for index, example in enumerate(bundled_fixture.examples)
        if example.domain == held_out_domain
    )
    changed_example = bundled_fixture.examples[example_index]
    changed_example = replace(
        changed_example,
        outcomes=(
            replace(
                changed_example.outcomes[0],
                quality=changed_example.outcomes[0].quality + 0.03125,
            ),
            *changed_example.outcomes[1:],
        ),
    )
    changed_examples = tuple(
        changed_example if index == example_index else example
        for index, example in enumerate(bundled_fixture.examples)
    )
    changed_fixture = _prepare(changed_examples, bundled_fixture.tier_specs)
    changed_result = _evaluate(changed_fixture)

    original_fold = next(
        fold for fold in bundled_result.learned.folds if fold.held_out_domain == held_out_domain
    )
    changed_fold = next(
        fold for fold in changed_result.learned.folds if fold.held_out_domain == held_out_domain
    )
    assert changed_fold.tuning == original_fold.tuning

    held_out_index = bundled_fixture.store.plan.domains.index(held_out_domain)
    outer_subset = next(
        index
        for index, subset in enumerate(bundled_fixture.store.plan.training_subsets)
        if len(subset.domain_indices) == len(bundled_fixture.store.plan.domains) - 1
        and held_out_index not in subset.domain_indices
    )
    original_calibration = next(
        item for item in bundled_result.calibrations if item.training_subset_index == outer_subset
    )
    changed_calibration = next(
        item for item in changed_result.calibrations if item.training_subset_index == outer_subset
    )
    original_scores = next(
        item
        for item in bundled_result.calibrated_score_blocks
        if item.training_subset_index == outer_subset and item.scored_domain_index == held_out_index
    )
    changed_scores = next(
        item
        for item in changed_result.calibrated_score_blocks
        if item.training_subset_index == outer_subset and item.scored_domain_index == held_out_index
    )
    assert changed_calibration == original_calibration
    assert changed_scores == original_scores

    def heldout_selections(result: PreparedReferencePipelineResult) -> tuple[str | None, ...]:
        return tuple(
            query.selected_model_id
            for tier in result.learned.report.tiers
            for query in tier.queries
            if query.example_id
            in {
                example.example_id
                for example in changed_examples
                if example.domain == held_out_domain
            }
        )

    assert heldout_selections(changed_result) == heldout_selections(bundled_result)


def test_quality_mutation_fails_source_lineage_before_calibration(
    bundled_fixture: _PreparedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = bundled_fixture.examples[0]
    mutated_outcomes = (
        replace(first.outcomes[0], quality=first.outcomes[0].quality + 0.01),
        *first.outcomes[1:],
    )
    mutated_examples = (replace(first, outcomes=mutated_outcomes), *bundled_fixture.examples[1:])
    calls = {"score": 0, "target": 0, "fit": 0}

    def forbidden_score(*args: object, **kwargs: object) -> tuple[float, ...]:
        del args, kwargs
        calls["score"] += 1
        raise AssertionError("score_row must not run before source-lineage rejection")

    def forbidden_target(*args: object, **kwargs: object) -> tuple[float, ...]:
        del args, kwargs
        calls["target"] += 1
        raise AssertionError("target_row must not run before source-lineage rejection")

    def forbidden_fit(*args: object, **kwargs: object) -> IsotonicCalibrator:
        del args, kwargs
        calls["fit"] += 1
        raise AssertionError("isotonic fit must not run before source-lineage rejection")

    monkeypatch.setattr(prepared_module.PreparedRawScoreBlock, "score_row", forbidden_score)
    monkeypatch.setattr(prepared_module.PreparedFeatureStore, "target_row", forbidden_target)
    monkeypatch.setattr(prepared_module.IsotonicCalibrator, "fit", forbidden_fit)
    with pytest.raises(ValueError, match="trusted source-fit"):
        evaluate_prepared_reference_pipeline(
            mutated_examples,
            bundled_fixture.tier_specs,
            bundled_fixture.store,
            bundled_fixture.raw_scores,
            PerQueryBudgetLedger,
            expected_source_fit_sha256=bundled_fixture.source_sha256,
            expected_store_sha256=bundled_fixture.store.sha256,
            expected_raw_score_sha256=bundled_fixture.raw_scores.sha256,
        )
    assert calls == {"score": 0, "target": 0, "fit": 0}


def test_aggregate_admission_fails_before_source_or_numeric_reads(
    bundled_fixture: _PreparedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate = estimate_prepared_reference_pipeline(
        bundled_fixture.store.plan,
        tier_count=len(bundled_fixture.tier_specs),
        max_candidates_per_tier=257,
    )
    monkeypatch.setattr(
        prepared_module,
        "MAX_PREPARED_PIPELINE_WORK_UNITS",
        estimate.total_work_units - 1,
    )
    monkeypatch.setattr(
        prepared_module.PreparedRawScoreBlock,
        "score_row",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected score read")),
    )
    monkeypatch.setattr(
        prepared_module.PreparedFeatureStore,
        "target_row",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected target read")),
    )
    with pytest.raises(ValueError, match="modeled work"):
        _evaluate(bundled_fixture)


def test_cost_width_aggregate_evidence_limit_fails_before_numeric_resnapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cheap_cost = Decimal("1e-100000")
    premium_cost = Decimal("1e99999")
    examples = tuple(
        EvaluationExample(
            example_id=f"wide-cost-{index}",
            prompt=f"Wide exact-cost prompt {index}",
            domain=f"domain-{index}",
            candidate_models=(
                ModelSpec("cheap", cheap_cost),
                ModelSpec("premium", premium_cost),
            ),
            outcomes=(
                CandidateOutcome("cheap", "cheap", cheap_cost, 0.25 + index * 0.01),
                CandidateOutcome("premium", "premium", premium_cost, 0.75 - index * 0.01),
            ),
        )
        for index in range(7)
    )
    specs = (
        TierSpec(BudgetTier.FAST, premium_cost, 3.0),
        TierSpec(BudgetTier.BALANCED, premium_cost, 2.0),
        TierSpec(BudgetTier.PREMIUM, premium_cost, 1.0),
    )
    fixture = _prepare(examples, specs)
    fold_estimates = tuple(
        estimate_lambda_search(
            fold.training,
            specs,
            max_candidates_per_tier=2,
        )
        for fold in leave_one_domain_out(examples)
    )
    assert max(item.estimated_policy_artifact_bytes for item in fold_estimates) < (
        prepared_module.MAX_PREPARED_PIPELINE_CANDIDATE_EVIDENCE_BYTES
    )
    assert sum(item.estimated_policy_artifact_bytes for item in fold_estimates) > (
        prepared_module.MAX_PREPARED_PIPELINE_CANDIDATE_EVIDENCE_BYTES
    )
    monkeypatch.setattr(
        prepared_module,
        "_resnapshot_prepared_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("numeric resnapshot must not run for oversized candidate evidence")
        ),
    )

    with pytest.raises(ValueError, match="candidate evidence"):
        evaluate_prepared_reference_pipeline(
            examples,
            specs,
            fixture.store,
            fixture.raw_scores,
            PerQueryBudgetLedger,
            expected_source_fit_sha256=fixture.source_sha256,
            expected_store_sha256=fixture.store.sha256,
            expected_raw_score_sha256=fixture.raw_scores.sha256,
            max_candidates_per_tier=2,
        )


def test_legal_large_fraction_candidates_bypass_cpython_digit_cap() -> None:
    tiny_cost = Decimal("1e-5000")
    examples = tuple(
        EvaluationExample(
            example_id=f"legal-wide-{index}",
            prompt=f"Legal wide-cost prompt {index}",
            domain=f"domain-{index}",
            candidate_models=(
                ModelSpec("tiny", tiny_cost),
                ModelSpec("unit", Decimal("1")),
            ),
            outcomes=(
                CandidateOutcome("tiny", "tiny", tiny_cost, 0.2 + index * 0.01),
                CandidateOutcome("unit", "unit", Decimal("1"), 0.8 - index * 0.01),
            ),
        )
        for index in range(4)
    )
    specs = (TierSpec(BudgetTier.FAST, Decimal("1"), 1.0),)
    fixture = _prepare(examples, specs)
    result = evaluate_prepared_reference_pipeline(
        examples,
        specs,
        fixture.store,
        fixture.raw_scores,
        PerQueryBudgetLedger,
        expected_source_fit_sha256=fixture.source_sha256,
        expected_store_sha256=fixture.store.sha256,
        expected_raw_score_sha256=fixture.raw_scores.sha256,
        max_candidates_per_tier=2,
    )
    reference = evaluate_per_query_bilinear_benchmark(
        examples,
        specs,
        max_candidates_per_tier=2,
    )

    assert result.learned == reference.learned
    lambda_estimates = result.estimate.lambda_search_estimates
    assert lambda_estimates is not None
    assert max(item.maximum_candidate_fraction_characters for item in lambda_estimates) > 4_300
    assert result.estimate.candidate_evidence_upper_bound_bytes is not None
    assert result.estimate.candidate_evidence_upper_bound_bytes < (
        prepared_module.MAX_PREPARED_PIPELINE_CANDIDATE_EVIDENCE_BYTES
    )


def test_trusted_digests_and_exact_container_types_fail_closed(
    bundled_fixture: _PreparedFixture,
) -> None:
    arguments = dict(
        examples=bundled_fixture.examples,
        tier_specs=bundled_fixture.tier_specs,
        store=bundled_fixture.store,
        raw_scores=bundled_fixture.raw_scores,
        ledger_factory=PerQueryBudgetLedger,
        expected_source_fit_sha256=bundled_fixture.source_sha256,
        expected_store_sha256=bundled_fixture.store.sha256,
        expected_raw_score_sha256=bundled_fixture.raw_scores.sha256,
    )
    with pytest.raises(TypeError, match="exact tuple"):
        evaluate_prepared_reference_pipeline(
            **{**arguments, "examples": list(arguments["examples"])}
        )
    with pytest.raises(ValueError, match="expected_store_sha256"):
        evaluate_prepared_reference_pipeline(**{**arguments, "expected_store_sha256": "0"})
    with pytest.raises(ValueError, match="trusted store"):
        evaluate_prepared_reference_pipeline(**{**arguments, "expected_store_sha256": "0" * 64})
    with pytest.raises(ValueError, match="between 2 and 257"):
        evaluate_prepared_reference_pipeline(**arguments, max_candidates_per_tier=258)
    with pytest.raises(TypeError, match="exact integer"):
        evaluate_prepared_reference_pipeline(**arguments, max_candidates_per_tier=True)


def test_public_estimator_rejects_bad_types_before_formula_access(
    bundled_fixture: _PreparedFixture,
) -> None:
    plan = bundled_fixture.store.plan
    with pytest.raises(TypeError, match="tier_count must be an exact integer"):
        estimate_prepared_reference_pipeline(
            plan,
            tier_count=True,
            max_candidates_per_tier=257,
        )
    with pytest.raises(TypeError, match="max_candidates_per_tier must be an exact integer"):
        estimate_prepared_reference_pipeline(
            plan,
            tier_count=3,
            max_candidates_per_tier="2",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="execution_estimate must be exact"):
        estimate_prepared_reference_pipeline(
            plan,
            tier_count=3,
            max_candidates_per_tier=257,
            execution_estimate=object(),  # type: ignore[arg-type]
        )


def test_wrong_parent_digest_fails_before_recursive_resnapshot(
    bundled_fixture: _PreparedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepared_module,
        "_resnapshot_prepared_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recursive resnapshot must not run for a wrong parent digest")
        ),
    )
    with pytest.raises(ValueError, match="trusted store"):
        evaluate_prepared_reference_pipeline(
            bundled_fixture.examples,
            bundled_fixture.tier_specs,
            bundled_fixture.store,
            bundled_fixture.raw_scores,
            PerQueryBudgetLedger,
            expected_source_fit_sha256=bundled_fixture.source_sha256,
            expected_store_sha256="0" * 64,
            expected_raw_score_sha256=bundled_fixture.raw_scores.sha256,
        )


def test_source_mismatch_fails_before_recursive_numeric_validation(
    bundled_fixture: _PreparedFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = bundled_fixture.examples[0]
    changed = replace(
        first,
        outcomes=(
            replace(first.outcomes[0], quality=first.outcomes[0].quality + 0.015625),
            *first.outcomes[1:],
        ),
    )
    changed_examples = (changed, *bundled_fixture.examples[1:])
    monkeypatch.setattr(
        prepared_module,
        "_resnapshot_prepared_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("numeric resnapshot must not run for a source mismatch")
        ),
    )
    with pytest.raises(ValueError, match="trusted source-fit"):
        evaluate_prepared_reference_pipeline(
            changed_examples,
            bundled_fixture.tier_specs,
            bundled_fixture.store,
            bundled_fixture.raw_scores,
            PerQueryBudgetLedger,
            expected_source_fit_sha256=bundled_fixture.source_sha256,
            expected_store_sha256=bundled_fixture.store.sha256,
            expected_raw_score_sha256=bundled_fixture.raw_scores.sha256,
        )


def test_amplified_child_count_fails_before_leaf_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    fixture = _prepare(tuple(dataset.examples), tuple(dataset.tier_specs))
    coefficients = fixture.raw_scores.coefficients
    object.__setattr__(coefficients, "blocks", coefficients.blocks * 25)
    monkeypatch.setattr(
        prepared_module,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("leaf reconstruction must not run for amplified children")
        ),
    )

    with pytest.raises(ValueError, match="wrong canonical bounded length"):
        _evaluate(fixture)


def test_reconstruction_detects_post_construction_raw_payload_mutation() -> None:
    dataset = load_evaluation_dataset()
    fixture = _prepare(tuple(dataset.examples), tuple(dataset.tier_specs))
    block = fixture.raw_scores.blocks[0]
    original_value = struct.unpack_from("<d", block.scores_payload)[0]
    payload = bytearray(block.scores_payload)
    struct.pack_into("<d", payload, 0, original_value + 0.125)
    object.__setattr__(block, "scores_payload", bytes(payload))

    with pytest.raises(ValueError, match=r"canonical bundle position|trusted bundle"):
        _evaluate(fixture)


def test_result_derived_fields_cannot_be_replaced(
    bundled_fixture: _PreparedFixture,
    bundled_result: PreparedReferencePipelineResult,
) -> None:
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(bundled_result, all_searches_exhaustive=False)
    calibration = bundled_result.calibrations[0]
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(calibration, sha256="0" * 64)
    wrong_domain_shard = replace(
        bundled_result.target_shards[0],
        domain="wrong-domain",
    )
    with pytest.raises(ValueError, match="plan catalogue"):
        replace(
            bundled_result,
            target_shards=(wrong_domain_shard, *bundled_result.target_shards[1:]),
        )
    wrong_raw_indices = replace(
        calibration,
        raw_score_block_indices=tuple(index + 1 for index in calibration.raw_score_block_indices),
    )
    with pytest.raises(ValueError, match="prepared subset"):
        replace(
            bundled_result,
            calibrations=(wrong_raw_indices, *bundled_result.calibrations[1:]),
        )
    with pytest.raises(ValueError, match="solver_id"):
        replace(bundled_result, solver_id="unreviewed-solver")

    raw_index = calibration.raw_score_block_indices[0]
    altered_raw_catalog = list(bundled_result.raw_score_block_sha256s)
    altered_raw_catalog[raw_index] = "0" * 64
    with pytest.raises(ValueError, match="prepared subset"):
        replace(bundled_result, raw_score_block_sha256s=tuple(altered_raw_catalog))

    oversized_calibrator = IsotonicCalibrator(
        tuple(float(index) for index in range(calibration.calibration_example_count + 1)),
        tuple(float(index) for index in range(calibration.calibration_example_count + 1)),
    )
    with pytest.raises(ValueError, match="block count"):
        replace(
            calibration,
            calibrators=(oversized_calibrator, *calibration.calibrators[1:]),
        )

    preliminary = estimate_prepared_reference_pipeline(
        bundled_result.estimate.plan,
        tier_count=len(bundled_fixture.tier_specs),
        max_candidates_per_tier=257,
        execution_estimate=bundled_fixture.raw_scores.execution_estimate,
    )
    with pytest.raises(ValueError, match="lambda-search estimates"):
        replace(bundled_result, estimate=preliminary)

    one_spec = bundled_fixture.tier_specs[:1]
    one_lambda_estimates = tuple(
        estimate_lambda_search(
            fold.training,
            one_spec,
            max_candidates_per_tier=257,
        )
        for fold in leave_one_domain_out(bundled_fixture.examples)
    )
    one_tier_estimate = estimate_prepared_reference_pipeline(
        bundled_result.estimate.plan,
        tier_count=1,
        max_candidates_per_tier=257,
        execution_estimate=bundled_fixture.raw_scores.execution_estimate,
        lambda_search_estimates=one_lambda_estimates,
    )
    with pytest.raises(ValueError, match="tier count"):
        replace(bundled_result, estimate=one_tier_estimate)

    cap_two_lambda_estimates = tuple(
        estimate_lambda_search(
            fold.training,
            bundled_fixture.tier_specs,
            max_candidates_per_tier=2,
        )
        for fold in leave_one_domain_out(bundled_fixture.examples)
    )
    cap_two_estimate = estimate_prepared_reference_pipeline(
        bundled_result.estimate.plan,
        tier_count=len(bundled_fixture.tier_specs),
        max_candidates_per_tier=2,
        execution_estimate=bundled_fixture.raw_scores.execution_estimate,
        lambda_search_estimates=cap_two_lambda_estimates,
    )
    with pytest.raises(ValueError, match="candidate bounds"):
        replace(bundled_result, estimate=cap_two_estimate)

    first_fold = bundled_result.learned.folds[0]
    first_selection = first_fold.tuning.selections[0]
    search_estimates = bundled_result.estimate.lambda_search_estimates
    assert search_estimates is not None
    impossible_candidates = replace(
        first_selection.candidates,
        observed_breakpoint_count=search_estimates[0].unequal_cost_pair_occurrences + 1,
    )
    impossible_selection = replace(first_selection, candidates=impossible_candidates)
    impossible_tuning = replace(
        first_fold.tuning,
        selections=(impossible_selection, *first_fold.tuning.selections[1:]),
    )
    impossible_fold = replace(first_fold, tuning=impossible_tuning)
    impossible_learned = replace(
        bundled_result.learned,
        folds=(impossible_fold, *bundled_result.learned.folds[1:]),
    )
    with pytest.raises(ValueError, match="candidate bounds"):
        replace(bundled_result, learned=impossible_learned)


def test_identical_prompt_batch_with_incompatible_precomputed_rows_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domains = ("a", "b", "c", "d")
    examples = tuple(
        EvaluationExample(
            example_id=f"row-{index}",
            prompt="identical prompt",
            domain=domain,
            candidate_models=(
                ModelSpec("cheap", Decimal("1")),
                ModelSpec("premium", Decimal("2")),
            ),
            outcomes=(
                CandidateOutcome("cheap", "cheap", Decimal("1"), 0.1 + index * 0.03),
                CandidateOutcome("premium", "premium", Decimal("2"), 0.9 - index * 0.04),
            ),
        )
        for index, domain in enumerate((*domains, *domains), start=1)
    )
    fixture = _prepare(
        examples,
        (TierSpec(BudgetTier.FAST, Decimal("2"), 1.0),),
    )
    # Surface-only duplicate prompts normally produce identical raw rows.  Changing one
    # canonical destination block under a freshly trusted digest models an embedded
    # provider that produced example-specific values the prompt-only protocol cannot join.
    blocks = list(fixture.raw_scores.blocks)
    target_index = next(
        index
        for index, graph_block in enumerate(fixture.store.plan.score_blocks)
        if len(
            fixture.store.plan.training_subsets[graph_block.training_subset_index].domain_indices
        )
        == 2
    )
    target = blocks[target_index]
    payload = bytearray(target.scores_payload)
    original = struct.unpack_from("<d", payload, 0)[0]
    struct.pack_into("<d", payload, 0, original + 0.25)
    blocks[target_index] = replace(target, scores_payload=bytes(payload))
    altered = replace(fixture.raw_scores, blocks=tuple(blocks))
    altered_fixture = replace(fixture, raw_scores=altered)
    # Isolate the prompt-batch join guard from PAV clipping: the bridge must reject
    # two indistinguishable batches whenever their post-calibration rows differ.
    monkeypatch.setattr(
        prepared_module.IsotonicCalibrator,
        "calibrate",
        lambda self, prediction: prediction,
    )

    with pytest.raises(ValueError, match="cannot disambiguate identical prompt batches"):
        evaluate_prepared_reference_pipeline(
            altered_fixture.examples,
            altered_fixture.tier_specs,
            altered_fixture.store,
            altered,
            PerQueryBudgetLedger,
            expected_source_fit_sha256=altered_fixture.source_sha256,
            expected_store_sha256=altered_fixture.store.sha256,
            expected_raw_score_sha256=altered.sha256,
        )
