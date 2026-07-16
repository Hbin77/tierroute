# SPDX-License-Identifier: Apache-2.0
"""Parity, graph-coverage, and isolation tests for prepared reference execution."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal

import pytest

import tierroute.predictors.prepared_execution as execution_module
from tierroute.core import ModelSpec
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.features.encoding import PromptFeatureSchema
from tierroute.features.surface import SURFACE_DOMAIN_TAG_CATALOGUE
from tierroute.predictors._ridge import RidgeSolution, solve_centered_ridge
from tierroute.predictors.prepared_execution import (
    PreparedCoefficientBlock,
    PreparedCoefficientBundle,
    PreparedRawScoreBlock,
    PreparedRawScoreBundle,
    PreparedScoredFeatureShard,
    build_prepared_coefficient_bundle,
    build_prepared_raw_score_bundle,
    build_prepared_scored_feature_shards,
)
from tierroute.predictors.prepared_graph import build_prepared_nested_lodo_plan
from tierroute.predictors.prepared_store import (
    PreparedDomainStatisticsBundle,
    PreparedEmbeddingInput,
    PreparedFeatureStore,
    build_prepared_domain_statistics,
    build_prepared_embedding_snapshot,
    build_prepared_feature_store,
    prepared_fit_source_sha256,
)

_MODEL_IDS = ("cheap", "premium")
_EMBEDDING_DIMENSION = 4
_UNIVERSAL_SURFACE_DIMENSION = 5 + len(SURFACE_DOMAIN_TAG_CATALOGUE)

_FOUR_DOMAIN_ROWS = (
    ("row-08", "delta", "Assess a medical patient diagnosis in a clinical setting."),
    ("row-03", "charlie", "Review this legal contract and relevant court statute."),
    ("row-06", "bravo", "수학 확률 문제를 증명하라."),
    ("row-01", "alpha", "Debug this Python API:\nimport os"),
    ("row-07", "charlie", "Summarize the law and the controlling legal precedent."),
    ("row-02", "bravo", "Prove this geometry theorem with an equation."),
    ("row-05", "alpha", "Write a Rust algorithm for sorting."),
    ("row-04", "delta", "Explain medicine options for this clinical patient."),
)

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


@dataclass(frozen=True, slots=True)
class _Context:
    examples: tuple[EvaluationExample, ...]
    store: PreparedFeatureStore
    statistics: PreparedDomainStatisticsBundle


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider="fixture-provider",
        model_id="fixture-embedding",
        revision="fixed-revision",
        pooling="mean",
        normalize=True,
        asset_manifest_sha256="a" * 64,
    )


def _examples(
    rows: tuple[tuple[str, str, str], ...] = _FOUR_DOMAIN_ROWS,
) -> tuple[EvaluationExample, ...]:
    examples = []
    for example_id, domain, prompt in rows:
        ordinal = int(example_id.removeprefix("row-"))
        qualities = {
            "cheap": 0.20 + ordinal * 0.031,
            "premium": 0.61 + ordinal * 0.027,
        }
        examples.append(
            EvaluationExample(
                example_id=example_id,
                prompt=prompt,
                domain=domain,
                candidate_models=(
                    ModelSpec("premium", Decimal("5")),
                    ModelSpec("cheap", Decimal("1")),
                ),
                outcomes=tuple(
                    CandidateOutcome(
                        model_id=model_id,
                        output=f"{example_id}:{model_id}:output",
                        cost=Decimal("5") if model_id == "premium" else Decimal("1"),
                        quality=qualities[model_id],
                    )
                    for model_id in _MODEL_IDS
                ),
            )
        )
    return tuple(examples)


def _ordinary_eighty_examples() -> tuple[EvaluationExample, ...]:
    domains = ("alpha", "bravo", "charlie", "delta")
    prompt_stems = (
        "Debug this Python function and explain the API.",
        "Prove this algebra equation carefully.",
        "Review a legal contract and court ruling.",
        "Assess a medicine diagnosis and treatment.",
        "Analyze finance revenue and investment.",
        "Summarize this ordinary question.",
    )
    rows = tuple(
        (
            f"row-{index:03d}",
            domains[(index * 7) % len(domains)],
            prompt_stems[index % len(prompt_stems)] + "\n" * (index % 5) + " extra" * (index % 9),
        )
        for index in range(1, 81)
    )
    return _examples(rows)


def _stress_examples() -> tuple[EvaluationExample, ...]:
    rows = tuple(
        (f"row-{index:03d}", domain, "identical plain prompt")
        for index, domain in enumerate(
            ("alpha", "bravo", "charlie", "delta") * 2,
            start=101,
        )
    )
    return tuple(
        replace(
            example,
            outcomes=tuple(
                replace(
                    outcome,
                    quality=(
                        0.5
                        if outcome.model_id == "cheap"
                        else 0.2 + int(example.example_id.removeprefix("row-")) * 0.003
                    ),
                )
                for outcome in example.outcomes
            ),
        )
        for example in _examples(rows)
    )


def _stress_embeddings(
    examples: tuple[EvaluationExample, ...],
) -> dict[str, tuple[float, ...]]:
    return {
        example.example_id: (
            centered * 1_000.0,
            centered * 2_000.0,
            centered * 0.000_001,
            0.0,
        )
        for example in examples
        for centered in (float(int(example.example_id.removeprefix("row-")) - 104),)
    }


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _embedding_values(example_id: str) -> tuple[float, ...]:
    ordinal = int(example_id.removeprefix("row-"))
    return (
        ordinal + 0.125,
        -ordinal / 3.0,
        (-1.0 if ordinal % 2 else 1.0) * 0.25,
        0.0,
    )


def _context(
    examples: tuple[EvaluationExample, ...],
    *,
    embedding_dimension: int,
    embedding_values: dict[str, tuple[float, ...]] | None = None,
) -> _Context:
    input_domains = tuple(reversed(sorted({example.domain for example in examples})))
    input_counts = tuple(
        sum(example.domain == domain for example in examples) for domain in input_domains
    )
    plan = build_prepared_nested_lodo_plan(
        input_domains,
        input_counts,
        feature_count=_UNIVERSAL_SURFACE_DIMENSION + embedding_dimension,
        target_count=len(_MODEL_IDS),
    )
    source_digest = prepared_fit_source_sha256(examples, plan)
    if embedding_dimension == 0:
        store = build_prepared_feature_store(
            examples,
            plan,
            expected_source_fit_sha256=source_digest,
        )
    else:
        assert embedding_dimension == _EMBEDDING_DIMENSION
        snapshot = build_prepared_embedding_snapshot(
            tuple(
                PreparedEmbeddingInput(
                    example_id=example.example_id,
                    prompt_sha256=_prompt_sha256(example.prompt),
                    values=(embedding_values or {}).get(
                        example.example_id,
                        _embedding_values(example.example_id),
                    ),
                )
                for example in examples
            ),
            _identity(),
            dimension=embedding_dimension,
        )
        store = build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=snapshot,
            expected_embedding_sha256=snapshot.sha256,
            expected_source_fit_sha256=source_digest,
        )
    return _Context(
        examples=examples,
        store=store,
        statistics=build_prepared_domain_statistics(store),
    )


def _build_execution(context: _Context) -> tuple[PreparedCoefficientBundle, PreparedRawScoreBundle]:
    coefficients = build_prepared_coefficient_bundle(
        context.store,
        context.statistics,
        ridge=1.0,
    )
    return coefficients, build_prepared_raw_score_bundle(context.store, coefficients)


def _encoded_row(
    store: PreparedFeatureStore,
    row_index: int,
    schema: PromptFeatureSchema,
) -> tuple[float, ...]:
    active_tags = tuple(5 + SURFACE_DOMAIN_TAG_CATALOGUE.index(tag) for tag in schema.domain_tags)
    active = (
        *range(5),
        *active_tags,
        *range(_UNIVERSAL_SURFACE_DIMENSION, store.plan.feature_count),
    )
    raw = store.feature_row(row_index)
    return tuple(
        (raw[raw_index] - schema.continuous_means[position]) / schema.continuous_scales[position]
        if position < 3
        else raw[raw_index]
        for position, raw_index in enumerate(active)
    )


def _row_oracle(
    store: PreparedFeatureStore,
    coefficient: object,
    schema: PromptFeatureSchema,
) -> RidgeSolution:
    training_domains = set(store.plan.training_subsets[coefficient.subset_index].domain_indices)
    row_indices = tuple(
        row_index
        for row_index, domain_index in enumerate(store.domain_indices)
        if domain_index in training_domains
    )
    feature_rows = tuple(_encoded_row(store, row_index, schema) for row_index in row_indices)
    target_columns = tuple(
        tuple(store.target_row(row_index)[model_index] for row_index in row_indices)
        for model_index in range(store.plan.target_count)
    )
    return solve_centered_ridge(
        feature_rows,
        target_columns,
        ridge=coefficient.ridge,
    )


def _current_row_schema(
    context: _Context,
    coefficient: object,
) -> PromptFeatureSchema:
    training_domains = set(
        context.store.plan.training_subsets[coefficient.subset_index].domain_indices
    )
    prompts_by_id = {example.example_id: example.prompt for example in context.examples}
    prompts = tuple(
        prompts_by_id[example_id]
        for example_id, domain_index in zip(
            context.store.example_ids,
            context.store.domain_indices,
            strict=True,
        )
        if domain_index in training_domains
    )
    return PromptFeatureSchema.fit(
        prompts,
        embedding_dimension=context.store.embedding_dimension,
        embedding_identity=context.store.embedding_identity,
    )


@pytest.mark.parametrize("embedding_dimension", [0, _EMBEDDING_DIMENSION])
def test_prepared_coefficients_and_every_raw_block_match_independent_row_oracle(
    embedding_dimension: int,
) -> None:
    context = _context(_examples(), embedding_dimension=embedding_dimension)
    coefficients, scores = _build_execution(context)
    row_oracles: dict[int, tuple[PromptFeatureSchema, RidgeSolution]] = {}

    for coefficient in coefficients.blocks:
        oracle_schema = _current_row_schema(context, coefficient)
        assert coefficient.feature_schema.domain_tags == oracle_schema.domain_tags
        assert coefficient.feature_schema.continuous_means == pytest.approx(
            oracle_schema.continuous_means,
            rel=1e-15,
            abs=1e-15,
        )
        assert coefficient.feature_schema.continuous_scales == pytest.approx(
            oracle_schema.continuous_scales,
            rel=1e-15,
            abs=1e-15,
        )

        expected_active = (
            *range(5),
            *(5 + SURFACE_DOMAIN_TAG_CATALOGUE.index(tag) for tag in oracle_schema.domain_tags),
            *range(_UNIVERSAL_SURFACE_DIMENSION, context.store.plan.feature_count),
        )
        assert coefficient.active_feature_indices == expected_active
        oracle = _row_oracle(context.store, coefficient, oracle_schema)
        row_oracles[coefficient.subset_index] = (oracle_schema, oracle)
        for actual, expected in zip(
            tuple(
                coefficient.weights_for_model_index(index)
                for index in range(context.store.plan.target_count)
            ),
            oracle.weights,
            strict=True,
        ):
            assert actual == pytest.approx(expected, rel=1e-8, abs=1e-9)
        assert tuple(
            coefficient.intercept_for_model_index(index)
            for index in range(context.store.plan.target_count)
        ) == pytest.approx(oracle.intercepts, rel=1e-9, abs=1e-10)

    for block_index, (graph_block, score_block) in enumerate(
        zip(context.store.plan.score_blocks, scores.blocks, strict=True)
    ):
        assert score_block.block_index == block_index
        coefficient = coefficients.blocks[graph_block.training_subset_index]
        oracle_schema, oracle = row_oracles[graph_block.training_subset_index]
        scored_indices = tuple(
            index
            for index, domain_index in enumerate(context.store.domain_indices)
            if domain_index == graph_block.scored_domain_index
        )
        expected_rows = tuple(
            tuple(
                sum(
                    value * weight
                    for value, weight in zip(
                        _encoded_row(context.store, row_index, oracle_schema),
                        oracle.weights[model_index],
                        strict=True,
                    )
                )
                + oracle.intercepts[model_index]
                for model_index in range(context.store.plan.target_count)
            )
            for row_index in scored_indices
        )
        assert len(expected_rows) == graph_block.row_count
        for actual, expected in zip(
            score_block.iter_score_rows(),
            expected_rows,
            strict=True,
        ):
            assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_seven_domain_execution_has_exact_63_154_22n_and_22nm_structure() -> None:
    context = _context(
        _examples(_SEVEN_DOMAIN_ROWS),
        embedding_dimension=0,
    )
    coefficients, scores = _build_execution(context)
    plan = context.store.plan
    example_count = len(context.examples)
    target_count = len(_MODEL_IDS)

    assert len(plan.training_subsets) == 63
    assert len(coefficients.blocks) == 63
    assert len(plan.score_blocks) == 154
    assert len(scores.blocks) == 154
    assert plan.work.score_row_memberships == 22 * example_count
    assert sum(block.row_count for block in plan.score_blocks) == 22 * example_count
    assert plan.work.scalar_score_count == 22 * example_count * target_count
    assert coefficients.execution_estimate.score_cells == 22 * example_count * target_count
    assert sum(len(block.scores_payload) for block in scores.blocks) == (
        22 * example_count * target_count * 8
    )
    memberships = Counter(
        example_id
        for block_index in range(len(scores.blocks))
        for example_id in scores.example_ids_for_block(block_index)
    )
    assert memberships == Counter({example.example_id: 22 for example in context.examples})
    assert plan.domain_example_counts == (2, 1, 3, 1, 2, 1, 2)


def test_coefficient_builder_factorizes_once_per_unique_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_examples(), embedding_dimension=_EMBEDDING_DIMENSION)
    original = execution_module._ridge_reference._cholesky
    calls = 0

    def recording_cholesky(
        matrix: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(execution_module._ridge_reference, "_cholesky", recording_cholesky)
    coefficients = build_prepared_coefficient_bundle(
        context.store,
        context.statistics,
        ridge=1.0,
    )

    assert calls == len(context.store.plan.training_subsets)
    assert len(coefficients.blocks) == calls


def test_raw_scoring_does_not_read_target_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(_examples(), embedding_dimension=_EMBEDDING_DIMENSION)
    coefficients = build_prepared_coefficient_bundle(
        context.store,
        context.statistics,
        ridge=1.0,
    )

    def forbidden_target_row(self: PreparedFeatureStore, row_index: int) -> tuple[float, ...]:
        raise AssertionError(f"raw scoring read target row {row_index} from {self!r}")

    monkeypatch.setattr(PreparedFeatureStore, "target_row", forbidden_target_row)

    scores = build_prepared_raw_score_bundle(context.store, coefficients)

    assert len(scores.blocks) == len(context.store.plan.score_blocks)


def test_coefficient_cap_rejects_before_subset_combination_or_factorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_examples(), embedding_dimension=0)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"work ran before aggregate preflight: {args!r} {kwargs!r}")

    monkeypatch.setattr(execution_module, "MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS", 0)
    monkeypatch.setattr(execution_module, "combine_prepared_subset_statistics", forbidden)
    monkeypatch.setattr(execution_module._ridge_reference, "_cholesky", forbidden)

    with pytest.raises(ValueError, match="aggregate work limit"):
        build_prepared_coefficient_bundle(
            context.store,
            context.statistics,
            ridge=1.0,
        )


def test_raw_score_cap_rejects_before_feature_hash_or_row_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients = build_prepared_coefficient_bundle(
        context.store,
        context.statistics,
        ridge=1.0,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"score work ran before aggregate preflight: {args!r} {kwargs!r}")

    monkeypatch.setattr(execution_module, "MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS", 0)
    monkeypatch.setattr(execution_module, "_build_scored_feature_shards", forbidden)

    with pytest.raises(ValueError, match="aggregate work limit"):
        build_prepared_raw_score_bundle(context.store, coefficients)


def _mutate_domain(
    examples: tuple[EvaluationExample, ...],
    domain: str,
    *,
    prompt: str | None = None,
    quality_delta: float = 0.0,
) -> tuple[EvaluationExample, ...]:
    return tuple(
        replace(
            example,
            prompt=example.prompt if prompt is None else f"{prompt} {example.example_id}",
            outcomes=tuple(
                replace(outcome, quality=outcome.quality + quality_delta)
                for outcome in example.outcomes
            ),
        )
        if example.domain == domain
        else example
        for example in examples
    )


def _excluded_domain_block_indices(context: _Context, excluded_domain: str) -> tuple[int, int]:
    domain_index = context.store.plan.domains.index(excluded_domain)
    included = tuple(
        index for index in range(len(context.store.plan.domains)) if index != domain_index
    )
    subset_index = next(
        index
        for index, subset in enumerate(context.store.plan.training_subsets)
        if subset.domain_indices == included
    )
    block_index = next(
        index
        for index, block in enumerate(context.store.plan.score_blocks)
        if block.training_subset_index == subset_index and block.scored_domain_index == domain_index
    )
    return subset_index, block_index


def test_excluded_target_and_prompt_mutations_remain_local_to_expected_outputs() -> None:
    original_examples = _examples()
    quality_examples = _mutate_domain(
        original_examples,
        "delta",
        quality_delta=0.5,
    )
    prompt_examples = _mutate_domain(
        original_examples,
        "delta",
        prompt="Completely different Python code with equations:\nimport math",
    )
    contexts = tuple(
        _context(examples, embedding_dimension=_EMBEDDING_DIMENSION)
        for examples in (original_examples, quality_examples, prompt_examples)
    )
    executions = tuple(_build_execution(context) for context in contexts)
    subset_index, block_index = _excluded_domain_block_indices(contexts[0], "delta")
    delta_index = contexts[0].store.plan.domains.index("delta")

    original_coefficients, original_scores = executions[0]
    quality_coefficients, quality_scores = executions[1]
    prompt_coefficients, prompt_scores = executions[2]

    assert original_coefficients.blocks[subset_index].sha256 == (
        quality_coefficients.blocks[subset_index].sha256
    )
    assert original_coefficients.blocks[subset_index].sha256 == (
        prompt_coefficients.blocks[subset_index].sha256
    )
    assert original_scores.feature_shards.shards[delta_index].sha256 == (
        quality_scores.feature_shards.shards[delta_index].sha256
    )
    assert original_scores.blocks[block_index].sha256 == quality_scores.blocks[block_index].sha256
    assert original_scores.blocks[block_index].scores_payload == (
        quality_scores.blocks[block_index].scores_payload
    )

    assert original_scores.feature_shards.shards[delta_index].sha256 != (
        prompt_scores.feature_shards.shards[delta_index].sha256
    )
    assert original_scores.blocks[block_index].sha256 != prompt_scores.blocks[block_index].sha256


def test_domain_omitted_from_training_and_scoring_does_not_change_the_leaf_block() -> None:
    original_context = _context(_examples(), embedding_dimension=0)
    changed_context = _context(
        _mutate_domain(
            _examples(),
            "delta",
            prompt="Unrelated changed prompt with Python and equations:\nimport decimal",
            quality_delta=0.375,
        ),
        embedding_dimension=0,
    )
    original_coefficients, original_scores = _build_execution(original_context)
    changed_coefficients, changed_scores = _build_execution(changed_context)
    plan = original_context.store.plan
    alpha_index = plan.domains.index("alpha")
    bravo_index = plan.domains.index("bravo")
    subset_index = next(
        index
        for index, subset in enumerate(plan.training_subsets)
        if subset.domain_indices == (alpha_index,)
    )
    block_index = next(
        index
        for index, block in enumerate(plan.score_blocks)
        if block.training_subset_index == subset_index and block.scored_domain_index == bravo_index
    )

    assert original_coefficients.blocks[subset_index].sha256 == (
        changed_coefficients.blocks[subset_index].sha256
    )
    assert original_scores.feature_shards.shards[bravo_index].sha256 == (
        changed_scores.feature_shards.shards[bravo_index].sha256
    )
    assert original_scores.blocks[block_index].sha256 == (changed_scores.blocks[block_index].sha256)
    assert original_scores.blocks[block_index].scores_payload == (
        changed_scores.blocks[block_index].scores_payload
    )
    assert original_coefficients.sha256 != changed_coefficients.sha256
    assert original_scores.sha256 != changed_scores.sha256


def test_direct_execution_records_reject_malformed_payloads_and_bundle_reordering() -> None:
    context = _context(_examples(), embedding_dimension=_EMBEDDING_DIMENSION)
    coefficients, scores = _build_execution(context)
    coefficient = coefficients.blocks[0]
    score = scores.blocks[0]

    with pytest.raises(ValueError, match="positive zero"):
        replace(
            coefficient,
            weights_payload=struct.pack("<d", -0.0) + coefficient.weights_payload[8:],
        )
    with pytest.raises(ValueError, match="wrong exact length"):
        replace(coefficient, intercepts_payload=coefficient.intercepts_payload[:-1])
    with pytest.raises(ValueError, match="finite binary64"):
        replace(
            score,
            scores_payload=struct.pack("<d", float("nan")) + score.scores_payload[8:],
        )
    with pytest.raises(ValueError):
        replace(coefficients, blocks=tuple(reversed(coefficients.blocks)))
    with pytest.raises(ValueError):
        replace(scores, blocks=tuple(reversed(scores.blocks)))
    with pytest.raises(ValueError, match="formula"):
        replace(
            coefficients.execution_estimate,
            score_cells=coefficients.execution_estimate.score_cells + 1,
        )


def test_scored_feature_shards_are_target_free_but_prompt_sensitive() -> None:
    original_examples = _examples()
    target_context = _context(
        _mutate_domain(original_examples, "delta", quality_delta=0.25),
        embedding_dimension=0,
    )
    prompt_context = _context(
        _mutate_domain(original_examples, "delta", prompt="New prompt text and Python code"),
        embedding_dimension=0,
    )
    original_context = _context(original_examples, embedding_dimension=0)

    original = build_prepared_scored_feature_shards(original_context.store)
    target_changed = build_prepared_scored_feature_shards(target_context.store)
    prompt_changed = build_prepared_scored_feature_shards(prompt_context.store)
    delta_index = original_context.store.plan.domains.index("delta")

    assert original.sha256 == target_changed.sha256
    assert original.shards[delta_index].sha256 == target_changed.shards[delta_index].sha256
    assert original.shards[delta_index].sha256 != prompt_changed.shards[delta_index].sha256


def test_embedding_store_and_coefficient_lineage_must_match() -> None:
    context = _context(_examples(), embedding_dimension=_EMBEDDING_DIMENSION)
    coefficients, _ = _build_execution(context)
    wrong_identity = replace(_identity(), revision="wrong-revision")
    wrong_blocks = tuple(
        replace(
            block,
            feature_schema=replace(
                block.feature_schema,
                embedding_identity=wrong_identity,
            ),
        )
        for block in coefficients.blocks
    )
    wrong_coefficients = replace(
        coefficients,
        embedding_identity=wrong_identity,
        blocks=wrong_blocks,
    )

    with pytest.raises(ValueError, match="prepared store layout"):
        build_prepared_raw_score_bundle(context.store, wrong_coefficients)

    changed_context = _context(
        _mutate_domain(_examples(), "delta", quality_delta=0.01),
        embedding_dimension=_EMBEDDING_DIMENSION,
    )
    with pytest.raises(ValueError, match="prepared store layout"):
        build_prepared_raw_score_bundle(changed_context.store, coefficients)


def test_tag_masks_ridge_and_shard_identity_cannot_disagree_with_parents() -> None:
    context = _context(_examples(), embedding_dimension=_EMBEDDING_DIMENSION)
    coefficients, scores = _build_execution(context)
    block = next(block for block in coefficients.blocks if block.feature_schema.domain_tags)
    replacement_tag = next(
        tag for tag in SURFACE_DOMAIN_TAG_CATALOGUE if tag not in block.feature_schema.domain_tags
    )
    changed_tags = tuple(sorted((replacement_tag, *block.feature_schema.domain_tags[1:])))
    changed_mask = sum(1 << SURFACE_DOMAIN_TAG_CATALOGUE.index(tag) for tag in changed_tags)

    with pytest.raises(ValueError, match="active_tag_mask"):
        replace(
            block,
            feature_schema=replace(block.feature_schema, domain_tags=changed_tags),
        )
    changed_block = replace(
        block,
        feature_schema=replace(block.feature_schema, domain_tags=changed_tags),
        active_tag_mask=changed_mask,
    )
    changed_blocks = tuple(
        changed_block if candidate.subset_index == block.subset_index else candidate
        for candidate in coefficients.blocks
    )
    with pytest.raises(ValueError, match="canonical bundle position"):
        replace(coefficients, blocks=changed_blocks)

    ridge_blocks = tuple(replace(candidate, ridge=2.0) for candidate in coefficients.blocks)
    ridge_coefficients = replace(coefficients, ridge=2.0, blocks=ridge_blocks)
    with pytest.raises(ValueError, match="canonical bundle position"):
        replace(scores, coefficients=ridge_coefficients)

    wrong_identity = replace(_identity(), revision="wrong-shard-revision")
    with pytest.raises(ValueError, match="canonical domain position"):
        replace(
            scores.feature_shards,
            shards=(
                replace(
                    scores.feature_shards.shards[0],
                    embedding_identity=wrong_identity,
                ),
                *scores.feature_shards.shards[1:],
            ),
        )


class _FloatSubclass(float):
    pass


class _IntSubclass(int):
    pass


class _StringSubclass(str):
    pass


class _TupleSubclass(tuple):
    pass


class _BytesSubclass(bytes):
    pass


@pytest.mark.parametrize(
    "ridge",
    [
        True,
        math.nan,
        math.inf,
        -math.inf,
        0.0,
        -0.0,
        -1.0,
        _FloatSubclass(1.0),
        _IntSubclass(1),
    ],
)
def test_coefficient_builder_rejects_invalid_ridge_before_combination(
    ridge: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_examples(), embedding_dimension=0)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"subset work ran for invalid ridge: {args!r} {kwargs!r}")

    monkeypatch.setattr(execution_module, "combine_prepared_subset_statistics", forbidden)
    with pytest.raises((TypeError, ValueError), match="ridge"):
        build_prepared_coefficient_bundle(
            context.store,
            context.statistics,
            ridge=ridge,  # type: ignore[arg-type]
        )


def test_execution_caps_accept_exact_boundary_and_reject_one_unit_less(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients = build_prepared_coefficient_bundle(
        context.store,
        context.statistics,
        ridge=1.0,
    )
    estimate = coefficients.execution_estimate

    monkeypatch.setattr(
        execution_module,
        "MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS",
        estimate.total_work_units,
    )
    monkeypatch.setattr(
        execution_module,
        "MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES",
        estimate.modeled_numeric_storage_bytes,
    )
    accepted = build_prepared_coefficient_bundle(
        context.store,
        context.statistics,
        ridge=1.0,
    )
    assert accepted.execution_estimate == estimate

    monkeypatch.setattr(
        execution_module,
        "MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS",
        estimate.total_work_units - 1,
    )
    with pytest.raises(ValueError, match="aggregate work limit"):
        build_prepared_coefficient_bundle(
            context.store,
            context.statistics,
            ridge=1.0,
        )

    monkeypatch.setattr(
        execution_module,
        "MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS",
        estimate.total_work_units,
    )
    monkeypatch.setattr(
        execution_module,
        "MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES",
        estimate.modeled_numeric_storage_bytes - 1,
    )
    with pytest.raises(ValueError, match="numeric-storage limit"):
        build_prepared_coefficient_bundle(
            context.store,
            context.statistics,
            ridge=1.0,
        )


def test_subset_validation_and_hash_work_are_included_in_aggregate_preflight() -> None:
    plan = build_prepared_nested_lodo_plan(
        ("alpha", "bravo", "charlie", "delta", "echo"),
        (1, 1, 1, 1, 1),
        feature_count=139,
        target_count=32,
    )
    widths = (139,) * len(plan.training_subsets)
    values = execution_module._estimate_execution_values(plan, widths)
    per_subset_cells = 139 + 32 + 139 * 140 // 2 + 139 * 32

    assert values["subset_statistics_scan_work_units"] == (
        3 * len(plan.training_subsets) * per_subset_cells
    )
    assert values["subset_statistics_transient_bytes"] == per_subset_cells * 8
    assert values["total_work_units"] > (
        execution_module.MAX_PREPARED_REFERENCE_EXECUTION_WORK_UNITS
    )
    with pytest.raises(ValueError, match="aggregate work limit"):
        execution_module._execution_estimate(plan, widths)

    coordinate_plan = build_prepared_nested_lodo_plan(
        ("alpha", "bravo", "charlie", "delta"),
        (8_772, 8_772, 8_771, 8_771),
        feature_count=12,
        target_count=13,
    )
    coordinate_widths = (12,) * len(coordinate_plan.training_subsets)
    coordinate_values = execution_module._estimate_execution_values(
        coordinate_plan,
        coordinate_widths,
    )
    assert coordinate_values["coordinate_preparation_work_units"] == (
        3 * len(coordinate_plan.training_subsets) * 12
    )
    assert coordinate_values["total_work_units"] == 100_000_008
    with pytest.raises(ValueError, match="aggregate work limit"):
        execution_module._execution_estimate(coordinate_plan, coordinate_widths)


def test_score_decode_encode_work_crosses_the_aggregate_cap() -> None:
    plan = build_prepared_nested_lodo_plan(
        ("alpha", "bravo", "charlie", "delta"),
        (629, 629, 629, 628),
        feature_count=185,
        target_count=4,
    )
    widths = (178,) * len(plan.training_subsets)
    values = execution_module._estimate_execution_values(plan, widths)

    assert values["score_decode_encode_work_units"] == 19_136_635
    assert values["total_work_units"] - values["score_decode_encode_work_units"] == (97_276_353)
    assert values["total_work_units"] == 116_412_988
    with pytest.raises(ValueError, match="aggregate work limit"):
        execution_module._execution_estimate(plan, widths)


def test_numeric_payload_validation_and_hash_work_crosses_the_aggregate_cap() -> None:
    plan = build_prepared_nested_lodo_plan(
        ("alpha", "bravo", "charlie", "delta"),
        (62_949, 62_949, 62_949, 62_948),
        feature_count=12,
        target_count=1,
    )
    widths = (12,) * len(plan.training_subsets)
    values = execution_module._estimate_execution_values(plan, widths)

    assert values["numeric_payload_work_units"] == (
        4 * values["coefficient_cells"] + 2 * values["score_cells"]
    )
    assert values["numeric_payload_work_units"] == 3_525_858
    assert values["total_work_units"] - values["numeric_payload_work_units"] == 99_999_981
    assert values["total_work_units"] == 103_525_839
    with pytest.raises(ValueError, match="aggregate work limit"):
        execution_module._execution_estimate(plan, widths)


def test_score_rows_are_streamed_and_joined_by_preserved_example_ids() -> None:
    context = _context(_examples(), embedding_dimension=0)
    _, scores = _build_execution(context)
    first = scores.blocks[0]
    graph_block = context.store.plan.score_blocks[0]
    expected_ids = tuple(
        example_id
        for example_id, domain_index in zip(
            context.store.example_ids,
            context.store.domain_indices,
            strict=True,
        )
        if domain_index == graph_block.scored_domain_index
    )

    assert not hasattr(first, "score_rows")
    assert tuple(first.iter_score_rows()) == tuple(
        first.score_row(index) for index in range(first.row_count)
    )
    assert scores.example_ids_for_block(0) == expected_ids
    with pytest.raises(TypeError, match="exact integer"):
        first.score_row(True)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="outside"):
        first.score_row(first.row_count)


def test_shard_bundle_rejects_cross_domain_duplicate_ids_and_caps_before_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_examples(), embedding_dimension=0)
    shards = build_prepared_scored_feature_shards(context.store)
    duplicate = replace(
        shards.shards[0],
        example_ids=shards.shards[1].example_ids,
        prompt_sha256s=shards.shards[1].prompt_sha256s,
    )
    with pytest.raises(ValueError, match="globally unique"):
        replace(shards, shards=(duplicate, *shards.shards[1:]))

    def forbidden_merge(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"ID merge ran before text cap: {args!r} {kwargs!r}")

    monkeypatch.setattr(execution_module.heapq, "merge", forbidden_merge)
    with pytest.raises(ValueError, match="canonical domain position"):
        replace(shards, shards=tuple(reversed(shards.shards)))

    monkeypatch.setattr(execution_module, "MAX_PREPARED_REFERENCE_TEXT_UTF8_BYTES", 0)
    with pytest.raises(ValueError, match="text-byte limit"):
        replace(shards)


def test_child_numeric_cap_rejects_before_payload_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients, scores = _build_execution(context)

    def forbidden_scan(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"payload scanned before child cap: {args!r} {kwargs!r}")

    monkeypatch.setattr(execution_module, "MAX_PREPARED_REFERENCE_EXECUTION_NUMERIC_BYTES", 0)
    monkeypatch.setattr(execution_module, "_validate_f64_payload", forbidden_scan)
    with pytest.raises(ValueError, match="numeric-storage limit"):
        replace(coefficients.blocks[0])
    with pytest.raises(ValueError, match="numeric-storage limit"):
        replace(scores.blocks[0])


@pytest.mark.parametrize("payload_name", ["weights", "intercepts", "scores"])
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, -0.0])
def test_every_execution_payload_rejects_noncanonical_binary64(
    payload_name: str,
    bad_value: float,
) -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients, scores = _build_execution(context)
    if payload_name == "weights":
        record = coefficients.blocks[0]
        field_name = "weights_payload"
    elif payload_name == "intercepts":
        record = coefficients.blocks[0]
        field_name = "intercepts_payload"
    else:
        record = scores.blocks[0]
        field_name = "scores_payload"
    payload = getattr(record, field_name)
    corrupted = struct.pack("<d", bad_value) + payload[8:]

    with pytest.raises(ValueError, match=r"finite binary64|positive zero"):
        replace(record, **{field_name: corrupted})


@pytest.mark.parametrize("payload_name", ["weights", "intercepts", "scores"])
def test_every_execution_payload_requires_exact_bytes(payload_name: str) -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients, scores = _build_execution(context)
    if payload_name == "weights":
        record = coefficients.blocks[0]
        field_name = "weights_payload"
    elif payload_name == "intercepts":
        record = coefficients.blocks[0]
        field_name = "intercepts_payload"
    else:
        record = scores.blocks[0]
        field_name = "scores_payload"

    with pytest.raises(TypeError, match=r"immutable bytes|payloads"):
        replace(record, **{field_name: _BytesSubclass(getattr(record, field_name))})


@pytest.mark.parametrize(
    "ridge",
    [
        True,
        math.nan,
        math.inf,
        -math.inf,
        0.0,
        -0.0,
        -1.0,
        _FloatSubclass(1.0),
        _IntSubclass(1),
    ],
)
def test_direct_coefficient_records_reject_invalid_ridge(ridge: object) -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients, _ = _build_execution(context)

    with pytest.raises((TypeError, ValueError), match="ridge"):
        replace(coefficients.blocks[0], ridge=ridge)
    with pytest.raises((TypeError, ValueError), match="ridge"):
        replace(coefficients, ridge=ridge)


@pytest.mark.parametrize("invalid_index", [True, -1, 0.5, _IntSubclass(0)])
def test_public_execution_accessors_require_exact_nonnegative_indices(
    invalid_index: object,
) -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients, scores = _build_execution(context)
    coefficient = coefficients.blocks[0]
    score = scores.blocks[0]

    expected_error = (TypeError, ValueError)
    with pytest.raises(expected_error):
        coefficient.weights_for_model_index(invalid_index)  # type: ignore[arg-type]
    with pytest.raises(expected_error):
        coefficient.intercept_for_model_index(invalid_index)  # type: ignore[arg-type]
    with pytest.raises(expected_error):
        score.score_row(invalid_index)  # type: ignore[arg-type]
    with pytest.raises(expected_error):
        scores.example_ids_for_block(invalid_index)  # type: ignore[arg-type]

    with pytest.raises(IndexError, match="outside"):
        coefficient.weights_for_model_index(context.store.plan.target_count)
    with pytest.raises(IndexError, match="outside"):
        coefficient.intercept_for_model_index(context.store.plan.target_count)
    with pytest.raises(IndexError, match="outside"):
        score.score_row(score.row_count)
    with pytest.raises(IndexError, match="outside"):
        scores.example_ids_for_block(len(context.store.plan.score_blocks))


def test_execution_estimate_rejects_scalar_and_container_subclasses() -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients, _ = _build_execution(context)
    estimate = coefficients.execution_estimate

    with pytest.raises(TypeError, match="exact tuple"):
        replace(
            estimate,
            active_feature_counts=_TupleSubclass(estimate.active_feature_counts),
        )
    with pytest.raises(TypeError, match="exact integer"):
        replace(
            estimate,
            active_feature_counts=(
                _IntSubclass(estimate.active_feature_counts[0]),
                *estimate.active_feature_counts[1:],
            ),
        )
    for field_name in estimate.__dataclass_fields__:
        if field_name in {"plan", "active_feature_counts"}:
            continue
        value = getattr(estimate, field_name)
        with pytest.raises(TypeError, match="exact integer"):
            replace(estimate, **{field_name: _IntSubclass(value)})


def test_execution_records_reject_string_and_tuple_subclasses() -> None:
    context = _context(_examples(), embedding_dimension=0)
    coefficients, scores = _build_execution(context)
    coefficient = coefficients.blocks[0]
    shard = scores.feature_shards.shards[0]

    with pytest.raises((TypeError, ValueError), match="exact tuple"):
        replace(coefficient, model_ids=_TupleSubclass(coefficient.model_ids))
    with pytest.raises(ValueError, match="exact string"):
        replace(
            coefficient,
            model_ids=(_StringSubclass(coefficient.model_ids[0]), coefficient.model_ids[1]),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(
            coefficient,
            subset_statistics_sha256=_StringSubclass(coefficient.subset_statistics_sha256),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(
            coefficients,
            source_store_sha256=_StringSubclass(coefficients.source_store_sha256),
        )
    with pytest.raises((TypeError, ValueError), match=r"tuple|masks"):
        replace(
            coefficients,
            domain_active_tag_masks=_TupleSubclass(coefficients.domain_active_tag_masks),
        )
    with pytest.raises((TypeError, ValueError), match="exact tuple"):
        replace(coefficients, blocks=_TupleSubclass(coefficients.blocks))
    with pytest.raises(TypeError, match="exact tuples"):
        replace(shard, example_ids=_TupleSubclass(shard.example_ids))
    with pytest.raises(ValueError, match="exact string"):
        replace(
            shard,
            example_ids=(_StringSubclass(shard.example_ids[0]), *shard.example_ids[1:]),
        )
    with pytest.raises(ValueError, match="SHA-256"):
        replace(
            shard,
            prompt_sha256s=(
                _StringSubclass(shard.prompt_sha256s[0]),
                *shard.prompt_sha256s[1:],
            ),
        )
    with pytest.raises((TypeError, ValueError), match=r"tuple|domain count"):
        replace(
            scores.feature_shards,
            shards=_TupleSubclass(scores.feature_shards.shards),
        )
    with pytest.raises((TypeError, ValueError), match=r"tuple|bounded length"):
        replace(scores, blocks=_TupleSubclass(scores.blocks))


def test_prepared_residual_gate_accepts_ordinary_chan_fixture_and_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        _ordinary_eighty_examples(),
        embedding_dimension=_EMBEDDING_DIMENSION,
    )
    coefficients = build_prepared_coefficient_bundle(
        context.store,
        context.statistics,
        ridge=1.0,
    )

    assert len(coefficients.blocks) == 14
    for coefficient in coefficients.blocks:
        for model_index in range(context.store.plan.target_count):
            assert all(
                math.isfinite(value) for value in coefficient.weights_for_model_index(model_index)
            )
            assert math.isfinite(coefficient.intercept_for_model_index(model_index))

    with pytest.raises(ArithmeticError, match="residual verification"):
        execution_module._verify_prepared_residual(
            ((2.0, 0.0), (0.0, 2.0)),
            (100.0, 0.0),
            (1.0, 1.0),
        )

    reviewed_roundoff = 1.0 + 3_000 * math.ulp(1.0)
    monkeypatch.setattr(execution_module, "_PREPARED_RESIDUAL_ULP_FACTOR", 1_024.0)
    with pytest.raises(ArithmeticError, match="residual verification"):
        execution_module._verify_prepared_residual(
            ((1.0,),),
            (reviewed_roundoff,),
            (1.0,),
        )
    monkeypatch.setattr(execution_module, "_PREPARED_RESIDUAL_ULP_FACTOR", 2_048.0)
    execution_module._verify_prepared_residual(
        ((1.0,),),
        (reviewed_roundoff,),
        (1.0,),
    )


def test_zero_variance_constant_target_and_collinear_high_dynamic_embeddings() -> None:
    examples = _stress_examples()
    context = _context(
        examples,
        embedding_dimension=_EMBEDDING_DIMENSION,
        embedding_values=_stress_embeddings(examples),
    )
    coefficients, scores = _build_execution(context)

    for coefficient in coefficients.blocks:
        assert coefficient.feature_schema.continuous_scales == (1.0, 1.0, 1.0)
        oracle_schema = _current_row_schema(context, coefficient)
        oracle = _row_oracle(context.store, coefficient, oracle_schema)
        for model_index in range(context.store.plan.target_count):
            assert coefficient.weights_for_model_index(model_index) == pytest.approx(
                oracle.weights[model_index],
                rel=1e-6,
                abs=1e-15,
            )
            assert coefficient.intercept_for_model_index(model_index) == pytest.approx(
                oracle.intercepts[model_index],
                rel=1e-7,
                abs=1e-8,
            )
        assert coefficient.weights_for_model_index(0) == pytest.approx(
            (0.0,) * coefficient.feature_count,
            rel=0.0,
            abs=1e-15,
        )
        assert coefficient.intercept_for_model_index(0) == pytest.approx(0.5, abs=1e-8)
        assert max(abs(value) for value in coefficient.weights_for_model_index(1)) > 5e-10

    for graph_block, score_block in zip(
        context.store.plan.score_blocks,
        scores.blocks,
        strict=True,
    ):
        coefficient = coefficients.blocks[graph_block.training_subset_index]
        oracle_schema = _current_row_schema(context, coefficient)
        oracle = _row_oracle(context.store, coefficient, oracle_schema)
        scored_rows = tuple(
            row_index
            for row_index, domain_index in enumerate(context.store.domain_indices)
            if domain_index == graph_block.scored_domain_index
        )
        for row_position, row_index in enumerate(scored_rows):
            encoded = _encoded_row(context.store, row_index, oracle_schema)
            expected = tuple(
                sum(
                    value * weight
                    for value, weight in zip(
                        encoded,
                        oracle.weights[model_index],
                        strict=True,
                    )
                )
                + oracle.intercepts[model_index]
                for model_index in range(context.store.plan.target_count)
            )
            assert score_block.score_row(row_position) == pytest.approx(
                expected,
                rel=1e-7,
                abs=1e-8,
            )


def test_execution_record_framing_has_portable_little_endian_goldens() -> None:
    plan = build_prepared_nested_lodo_plan(
        ("alpha", "bravo", "charlie", "delta"),
        (1, 1, 1, 1),
        feature_count=12,
        target_count=2,
    )
    schema = PromptFeatureSchema(
        continuous_means=(0.0, 0.0, 0.0),
        continuous_scales=(1.0, 1.0, 1.0),
        domain_tags=(),
    )
    coefficient = PreparedCoefficientBlock(
        plan=plan,
        subset_index=0,
        model_ids=_MODEL_IDS,
        feature_schema=schema,
        active_tag_mask=0,
        subset_statistics_sha256="1" * 64,
        included_content_sha256="2" * 64,
        ridge=1.0,
        weights_payload=struct.pack("<10d", *map(float, range(10))),
        intercepts_payload=struct.pack("<2d", 0.25, 0.75),
    )
    shard = PreparedScoredFeatureShard(
        plan=plan,
        domain_index=0,
        row_count=1,
        embedding_identity=None,
        embedding_dimension=0,
        example_ids=("row-1",),
        prompt_sha256s=("4" * 64,),
        feature_content_sha256="3" * 64,
    )
    raw_score = PreparedRawScoreBlock(
        plan=plan,
        block_index=0,
        model_ids=_MODEL_IDS,
        coefficient_block_sha256=coefficient.sha256,
        scored_feature_shard_sha256=shard.sha256,
        scores_payload=struct.pack("<2d", 0.125, 0.875),
    )

    assert coefficient.sha256 == (
        "011d48c9c333027be371b82084527a342e04e994a67f4ab0384bf595ec7a8cae"
    )
    assert shard.sha256 == "0fee02ba296dc72241f38924815c8621d4fb2d8425c907446cea3d1d118da386"
    assert raw_score.sha256 == ("e5693b5fec84b8ab3f6bf928e9981dab0682edce2696574def8399e6d1b08494")
