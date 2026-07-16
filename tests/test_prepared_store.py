# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical prepared rows and leakage-isolated sufficient statistics."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import replace
from decimal import Decimal

import pytest

import tierroute.predictors.prepared_store as prepared_store_module
from tierroute.core import ModelSpec
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.features.encoding import PromptFeatureSchema
from tierroute.features.surface import SURFACE_DOMAIN_TAG_CATALOGUE
from tierroute.predictors.prepared_graph import (
    PreparedNestedLodoPlan,
    build_prepared_nested_lodo_plan,
)
from tierroute.predictors.prepared_store import (
    PreparedEmbeddingInput,
    PreparedEmbeddingSnapshot,
    PreparedFeatureStore,
    build_prepared_domain_statistics,
    build_prepared_embedding_snapshot,
    build_prepared_feature_store,
    combine_prepared_subset_statistics,
    prepared_fit_source_sha256,
)

_EMBEDDING_DIMENSION = 4
_UNIVERSAL_SURFACE_DIMENSION = 5 + len(SURFACE_DOMAIN_TAG_CATALOGUE)
_RAW_FEATURE_DIMENSION = _UNIVERSAL_SURFACE_DIMENSION + _EMBEDDING_DIMENSION
_TAG_OFFSET = 5
_MODEL_IDS = ("cheap", "premium")

# IDs intentionally interleave domains once they are put in canonical lexical order.
_EXAMPLE_DATA = (
    ("row-08", "delta", "Assess a medical patient diagnosis in a clinical setting."),
    ("row-03", "charlie", "Review this legal contract and relevant court statute."),
    ("row-06", "bravo", "수학 확률 문제를 증명하라."),
    ("row-01", "alpha", "Debug this Python API:\nimport os"),
    ("row-07", "charlie", "Summarize the law and the controlling legal precedent."),
    ("row-02", "bravo", "Prove this geometry theorem with an equation."),
    ("row-05", "alpha", "Write a Rust algorithm for sorting."),
    ("row-04", "delta", "Explain medicine options for this clinical patient."),
)


def _identity(*, revision: str = "fixed-revision") -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider="fixture-provider",
        model_id="fixture-embedding",
        revision=revision,
        pooling="mean",
        normalize=True,
        asset_manifest_sha256="a" * 64,
    )


def _examples() -> tuple[EvaluationExample, ...]:
    examples = []
    for example_id, domain, prompt in _EXAMPLE_DATA:
        ordinal = int(example_id.removeprefix("row-"))
        qualities = {
            "cheap": 0.20 + ordinal * 0.031,
            "premium": 0.61 + ordinal * 0.027,
        }
        models = (
            ModelSpec("premium", Decimal("5")),
            ModelSpec("cheap", Decimal("1")),
        )
        outcomes = tuple(
            CandidateOutcome(
                model_id=model_id,
                output=f"{example_id}:{model_id}:output",
                cost=Decimal("5") if model_id == "premium" else Decimal("1"),
                quality=qualities[model_id],
            )
            for model_id in ("cheap", "premium")
        )
        examples.append(
            EvaluationExample(
                example_id=example_id,
                prompt=prompt,
                domain=domain,
                outcomes=outcomes,
                candidate_models=models,
            )
        )
    return tuple(examples)


def _plan(
    examples: tuple[EvaluationExample, ...],
    *,
    embedding_dimension: int = _EMBEDDING_DIMENSION,
) -> PreparedNestedLodoPlan:
    # Reversed input proves that callers need not pre-canonicalize the domain catalogue.
    domains = tuple(reversed(sorted({example.domain for example in examples})))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    return build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=_UNIVERSAL_SURFACE_DIMENSION + embedding_dimension,
        target_count=len(_MODEL_IDS),
    )


def _embedding_values(example_id: str) -> tuple[float, ...]:
    ordinal = int(example_id.removeprefix("row-"))
    return (
        ordinal + 0.125,
        -ordinal / 3.0,
        (-1.0 if ordinal % 2 else 1.0) * 0.25,
        0.0,
    )


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _embedding_rows(
    examples: tuple[EvaluationExample, ...],
    *,
    value_overrides: dict[str, tuple[float, ...]] | None = None,
) -> tuple[PreparedEmbeddingInput, ...]:
    overrides = value_overrides or {}
    return tuple(
        PreparedEmbeddingInput(
            example_id=example.example_id,
            prompt_sha256=_prompt_sha256(example.prompt),
            values=overrides.get(example.example_id, _embedding_values(example.example_id)),
        )
        for example in examples
    )


def _snapshot(
    examples: tuple[EvaluationExample, ...],
    *,
    value_overrides: dict[str, tuple[float, ...]] | None = None,
    identity: EmbeddingIdentity | None = None,
) -> PreparedEmbeddingSnapshot:
    return build_prepared_embedding_snapshot(
        _embedding_rows(examples, value_overrides=value_overrides),
        identity or _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )


def _store(
    examples: tuple[EvaluationExample, ...],
    *,
    plan: PreparedNestedLodoPlan | None = None,
    value_overrides: dict[str, tuple[float, ...]] | None = None,
) -> PreparedFeatureStore:
    plan = plan or _plan(examples)
    snapshot = _snapshot(examples, value_overrides=value_overrides)
    return build_prepared_feature_store(
        examples,
        plan,
        embedding_snapshot=snapshot,
        expected_embedding_sha256=snapshot.sha256,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )


def _subset_index(plan: PreparedNestedLodoPlan, domain_names: tuple[str, ...]) -> int:
    wanted = tuple(plan.domains.index(domain) for domain in domain_names)
    return next(
        index
        for index, subset in enumerate(plan.training_subsets)
        if subset.domain_indices == wanted
    )


def _selected_rows(
    store: PreparedFeatureStore,
    domain_indices: tuple[int, ...],
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    selected = set(domain_indices)
    return tuple(
        (store.feature_row(index), store.target_row(index))
        for index, domain_index in enumerate(store.domain_indices)
        if domain_index in selected
    )


def _direct_centered_moments(
    rows: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    feature_count = len(rows[0][0])
    target_count = len(rows[0][1])
    row_count = len(rows)
    feature_means = tuple(
        math.fsum(features[column] for features, _ in rows) / row_count
        for column in range(feature_count)
    )
    target_means = tuple(
        math.fsum(targets[column] for _, targets in rows) / row_count
        for column in range(target_count)
    )
    centered_xx = tuple(
        math.fsum(
            (features[row] - feature_means[row]) * (features[column] - feature_means[column])
            for features, _ in rows
        )
        for row in range(feature_count)
        for column in range(row, feature_count)
    )
    centered_xy = tuple(
        math.fsum(
            (features[feature] - feature_means[feature]) * (targets[target] - target_means[target])
            for features, targets in rows
        )
        for feature in range(feature_count)
        for target in range(target_count)
    )
    return feature_means, target_means, centered_xx, centered_xy


def _packed_diagonal_index(dimension: int, index: int) -> int:
    return index * dimension - index * (index - 1) // 2


def _replace_example(
    examples: tuple[EvaluationExample, ...],
    example_id: str,
    *,
    prompt: str,
    quality_delta: float,
) -> tuple[EvaluationExample, ...]:
    changed = []
    for example in examples:
        if example.example_id != example_id:
            changed.append(example)
            continue
        changed.append(
            replace(
                example,
                prompt=prompt,
                outcomes=tuple(
                    replace(outcome, quality=outcome.quality + quality_delta)
                    for outcome in example.outcomes
                ),
            )
        )
    return tuple(changed)


def test_canonical_permutations_produce_identical_snapshots_stores_and_digests() -> None:
    examples = _examples()
    plan = _plan(examples)
    rows = _embedding_rows(examples)
    first_snapshot = build_prepared_embedding_snapshot(
        rows,
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )
    permuted_snapshot = build_prepared_embedding_snapshot(
        tuple(reversed(rows)),
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )

    assert first_snapshot == permuted_snapshot
    assert first_snapshot.example_ids == tuple(sorted(example.example_id for example in examples))
    assert len(first_snapshot.sha256) == 64

    permuted_examples = tuple(
        replace(
            example,
            candidate_models=tuple(reversed(example.candidate_models)),
            outcomes=tuple(reversed(example.outcomes)),
        )
        for example in reversed(examples)
    )
    first_store = build_prepared_feature_store(
        examples,
        plan,
        embedding_snapshot=first_snapshot,
        expected_embedding_sha256=first_snapshot.sha256,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )
    permuted_store = build_prepared_feature_store(
        permuted_examples,
        _plan(permuted_examples),
        embedding_snapshot=permuted_snapshot,
        expected_embedding_sha256=permuted_snapshot.sha256,
        expected_source_fit_sha256=prepared_fit_source_sha256(
            permuted_examples,
            _plan(permuted_examples),
        ),
    )

    assert first_store == permuted_store
    assert first_store.sha256 == permuted_store.sha256
    assert first_store.model_ids == _MODEL_IDS
    assert first_store.feature_payload == permuted_store.feature_payload
    assert first_store.target_payload == permuted_store.target_payload
    assert first_store.plan.feature_count == _RAW_FEATURE_DIMENSION == 12 + _EMBEDDING_DIMENSION

    first_bundle = build_prepared_domain_statistics(first_store)
    permuted_bundle = build_prepared_domain_statistics(permuted_store)
    assert first_bundle == permuted_bundle
    assert first_bundle.sha256 == permuted_bundle.sha256
    assert tuple(
        combine_prepared_subset_statistics(first_bundle, index).sha256
        for index in range(len(plan.training_subsets))
    ) == tuple(
        combine_prepared_subset_statistics(permuted_bundle, index).sha256
        for index in range(len(plan.training_subsets))
    )


def test_snapshot_digest_binds_identity_prompt_keys_and_binary64_payload() -> None:
    examples = _examples()
    snapshot = _snapshot(examples)
    changed_identity = build_prepared_embedding_snapshot(
        _embedding_rows(examples),
        _identity(revision="different-revision"),
        dimension=_EMBEDDING_DIMENSION,
    )
    changed_values = _snapshot(
        examples,
        value_overrides={"row-01": (99.0, 98.0, 97.0, 96.0)},
    )
    changed_prompt_rows = list(_embedding_rows(examples))
    changed_prompt_rows[0] = replace(changed_prompt_rows[0], prompt_sha256="0" * 64)
    changed_prompt = build_prepared_embedding_snapshot(
        tuple(changed_prompt_rows),
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )

    # This input contains no libm-derived values, so it is a portable framing/LE golden.
    assert snapshot.sha256 == "b6c0d470610669feb488119e988610bb2e70c6e068422e92565c0dd176718db1"
    assert len({snapshot.sha256, changed_identity.sha256, changed_values.sha256}) == 3
    assert changed_prompt.sha256 != snapshot.sha256
    assert snapshot.payload != changed_values.payload

    # The builder normalizes both binary64 zero encodings before hashing.
    signed_zero_rows = tuple(
        replace(row, values=(*row.values[:-1], -0.0)) for row in _embedding_rows(examples)
    )
    signed_zero_snapshot = build_prepared_embedding_snapshot(
        signed_zero_rows,
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )
    assert signed_zero_snapshot == snapshot


def test_fit_source_digest_has_a_portable_golden_and_excludes_replay_only_fields() -> None:
    examples = _examples()
    plan = _plan(examples)

    assert prepared_fit_source_sha256(examples, plan) == (
        "641a5ce697f44c972923cac9336353174cf55a654bc613bec2986f69e5d57e80"
    )


def test_embedding_inputs_reject_malformed_values_dimensions_and_duplicate_ids() -> None:
    prompt_digest = "1" * 64
    with pytest.raises(TypeError, match="exact tuple"):
        PreparedEmbeddingInput("row", prompt_digest, [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact real"):
        PreparedEmbeddingInput("row", prompt_digest, (1.0, True))
    with pytest.raises(ValueError, match="finite binary64"):
        PreparedEmbeddingInput("row", prompt_digest, (1.0, math.inf))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PreparedEmbeddingInput("row", "A" * 64, (1.0, 2.0))

    rows = _embedding_rows(_examples())
    with pytest.raises(ValueError, match="declared dimension"):
        build_prepared_embedding_snapshot(rows, _identity(), dimension=3)
    with pytest.raises(ValueError, match="unique example IDs"):
        build_prepared_embedding_snapshot(
            (*rows, rows[0]),
            _identity(),
            dimension=_EMBEDDING_DIMENSION,
        )
    with pytest.raises(TypeError, match="exact integer"):
        build_prepared_embedding_snapshot(
            rows,
            _identity(),
            dimension=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="combined centered moment"):
        prepared_store_module._finite_moment_sum(1e308, 1e308)


def test_feature_store_requires_exact_embedding_id_prompt_and_expected_digest_join() -> None:
    examples = _examples()
    plan = _plan(examples)
    rows = _embedding_rows(examples)
    snapshot = build_prepared_embedding_snapshot(
        rows,
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )

    with pytest.raises(ValueError, match="caller-expected SHA-256"):
        build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=snapshot,
            expected_embedding_sha256="0" * 64,
            expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
        )
    with pytest.raises(ValueError, match="caller-expected fit SHA-256"):
        build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=snapshot,
            expected_embedding_sha256=snapshot.sha256,
            expected_source_fit_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=snapshot,
            expected_embedding_sha256=None,
            expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
        )

    missing = build_prepared_embedding_snapshot(
        rows[:-1],
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )
    with pytest.raises(ValueError, match="example IDs"):
        build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=missing,
            expected_embedding_sha256=missing.sha256,
            expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
        )

    extra = PreparedEmbeddingInput(
        "row-99",
        _prompt_sha256("unrelated prompt"),
        (1.0, 2.0, 3.0, 4.0),
    )
    with_extra = build_prepared_embedding_snapshot(
        (*rows, extra),
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )
    with pytest.raises(ValueError, match="example IDs"):
        build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=with_extra,
            expected_embedding_sha256=with_extra.sha256,
            expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
        )

    wrong_prompt_rows = tuple(
        replace(row, prompt_sha256="f" * 64) if row.example_id == "row-01" else row for row in rows
    )
    wrong_prompt = build_prepared_embedding_snapshot(
        wrong_prompt_rows,
        _identity(),
        dimension=_EMBEDDING_DIMENSION,
    )
    with pytest.raises(ValueError, match="prompt digests"):
        build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=wrong_prompt,
            expected_embedding_sha256=wrong_prompt.sha256,
            expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
        )


def test_surface_only_store_has_no_provider_or_network_file_api_boundary() -> None:
    examples = _examples()
    plan = _plan(examples, embedding_dimension=0)
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )
    subset = combine_prepared_subset_statistics(
        build_prepared_domain_statistics(store),
        _subset_index(plan, ("alpha", "bravo")),
    )

    assert store.embedding_dimension == 0
    assert store.embedding_identity is None
    assert store.embedding_snapshot_sha256 is None
    assert plan.feature_count == _UNIVERSAL_SURFACE_DIMENSION == 12
    assert subset.feature_schema.embedding_dimension == 0
    assert "EmbeddingProvider" not in prepared_store_module.__dict__
    assert "socket" not in prepared_store_module.__dict__
    assert "subprocess" not in prepared_store_module.__dict__
    assert "Path" not in prepared_store_module.__dict__


def test_welford_and_chan_moments_match_direct_matrix_oracle() -> None:
    store = _store(_examples())
    bundle = build_prepared_domain_statistics(store)

    for domain_index, statistics in enumerate(bundle.domain_statistics):
        rows = _selected_rows(store, (domain_index,))
        feature_means, target_means, centered_xx, centered_xy = _direct_centered_moments(rows)
        assert statistics.row_count == len(rows)
        assert statistics.feature_means == pytest.approx(feature_means, rel=1e-13, abs=1e-13)
        assert statistics.target_means == pytest.approx(target_means, rel=1e-13, abs=1e-13)
        assert statistics.centered_xx_packed == pytest.approx(
            centered_xx,
            rel=1e-12,
            abs=1e-13,
        )
        assert statistics.centered_xy == pytest.approx(centered_xy, rel=1e-12, abs=1e-13)

    subset_index = _subset_index(store.plan, ("alpha", "bravo"))
    subset = combine_prepared_subset_statistics(bundle, subset_index)
    rows = _selected_rows(store, subset.domain_indices)
    feature_means, target_means, centered_xx, centered_xy = _direct_centered_moments(rows)

    assert subset.row_count == len(rows) == 4
    assert subset.feature_means == pytest.approx(feature_means, rel=1e-13, abs=1e-13)
    assert subset.target_means == pytest.approx(target_means, rel=1e-13, abs=1e-13)
    assert subset.centered_xx_packed == pytest.approx(centered_xx, rel=1e-12, abs=1e-13)
    assert subset.centered_xy == pytest.approx(centered_xy, rel=1e-12, abs=1e-13)


def test_uneven_domain_chan_combination_matches_direct_rows() -> None:
    examples = _examples()
    added = replace(
        examples[0],
        example_id="row-09",
        domain="alpha",
        prompt="Write a Python algorithm with two lines.\nreturn result",
    )
    uneven = (*examples, added)
    store = _store(uneven)
    bundle = build_prepared_domain_statistics(store)
    subset = combine_prepared_subset_statistics(
        bundle,
        _subset_index(store.plan, ("alpha", "charlie")),
    )
    rows = _selected_rows(store, subset.domain_indices)
    feature_means, target_means, centered_xx, centered_xy = _direct_centered_moments(rows)

    assert tuple(store.plan.domain_example_counts) == (3, 2, 2, 2)
    assert subset.row_count == 5
    assert subset.feature_means == pytest.approx(feature_means, rel=1e-13, abs=1e-13)
    assert subset.target_means == pytest.approx(target_means, rel=1e-13, abs=1e-13)
    assert subset.centered_xx_packed == pytest.approx(centered_xx, rel=1e-12, abs=1e-13)
    assert subset.centered_xy == pytest.approx(centered_xy, rel=1e-12, abs=1e-13)


def test_subset_tags_and_population_scaling_use_only_included_training_domains() -> None:
    store = _store(_examples())
    bundle = build_prepared_domain_statistics(store)
    subset_index = _subset_index(store.plan, ("alpha", "bravo"))
    subset = combine_prepared_subset_statistics(bundle, subset_index)
    selected = _selected_rows(store, subset.domain_indices)
    _, _, centered_xx, _ = _direct_centered_moments(selected)

    assert subset.feature_schema.domain_tags == ("code", "math")
    assert "law" not in subset.feature_schema.domain_tags
    assert "medicine" not in subset.feature_schema.domain_tags
    expected_mask = sum(1 << SURFACE_DOMAIN_TAG_CATALOGUE.index(tag) for tag in ("code", "math"))
    assert subset.active_tag_mask == expected_mask
    expected_active_indices = (
        *range(_TAG_OFFSET),
        _TAG_OFFSET + SURFACE_DOMAIN_TAG_CATALOGUE.index("code"),
        _TAG_OFFSET + SURFACE_DOMAIN_TAG_CATALOGUE.index("math"),
        *range(_UNIVERSAL_SURFACE_DIMENSION, _RAW_FEATURE_DIMENSION),
    )
    assert subset.active_feature_indices == expected_active_indices
    assert subset.feature_schema.dimension == len(expected_active_indices)

    expected_scales = tuple(
        math.sqrt(
            centered_xx[_packed_diagonal_index(_RAW_FEATURE_DIMENSION, index)] / len(selected)
        )
        or 1.0
        for index in range(3)
    )
    assert subset.feature_schema.continuous_means == pytest.approx(subset.feature_means[:3])
    assert subset.feature_schema.continuous_scales == pytest.approx(expected_scales)

    training_prompts = tuple(
        example.prompt
        for example in sorted(_examples(), key=lambda example: example.example_id)
        if store.plan.domains.index(example.domain) in subset.domain_indices
    )
    current_schema = PromptFeatureSchema.fit(
        training_prompts,
        embedding_dimension=_EMBEDDING_DIMENSION,
        embedding_identity=_identity(),
    )
    assert subset.feature_schema.domain_tags == current_schema.domain_tags
    assert subset.feature_schema.continuous_means == pytest.approx(
        current_schema.continuous_means,
        rel=1e-15,
        abs=1e-15,
    )
    assert subset.feature_schema.continuous_scales == pytest.approx(
        current_schema.continuous_scales,
        rel=1e-15,
        abs=1e-15,
    )
    assert (
        subset.feature_schema.continuous_means != current_schema.continuous_means
        or subset.feature_schema.continuous_scales != current_schema.continuous_scales
    )


def test_excluded_domain_mutation_cannot_change_included_subset_statistics() -> None:
    examples = _examples()
    plan = _plan(examples)
    original_store = _store(examples, plan=plan)
    original_bundle = build_prepared_domain_statistics(original_store)
    subset_index = _subset_index(plan, ("alpha", "bravo"))
    original_subset = combine_prepared_subset_statistics(original_bundle, subset_index)

    excluded_examples = _replace_example(
        examples,
        "row-07",
        prompt="Analyze finance, investment, stock, accounting, and revenue.\nSecond line.",
        quality_delta=0.123,
    )
    excluded_store = _store(
        excluded_examples,
        plan=plan,
        value_overrides={"row-07": (90.0, 91.0, 92.0, 93.0)},
    )
    excluded_bundle = build_prepared_domain_statistics(excluded_store)
    excluded_subset = combine_prepared_subset_statistics(excluded_bundle, subset_index)

    assert excluded_store.sha256 != original_store.sha256
    assert excluded_bundle.sha256 != original_bundle.sha256
    assert excluded_subset == original_subset
    assert excluded_subset.sha256 == original_subset.sha256
    assert excluded_subset.included_content_sha256 == original_subset.included_content_sha256
    for domain_index in original_subset.domain_indices:
        assert (
            excluded_bundle.domain_statistics[domain_index]
            == (original_bundle.domain_statistics[domain_index])
        )

    included_examples = _replace_example(
        examples,
        "row-01",
        prompt="Review a law, legal court, and contract question.",
        quality_delta=0.321,
    )
    included_store = _store(
        included_examples,
        plan=plan,
        value_overrides={"row-01": (70.0, 71.0, 72.0, 73.0)},
    )
    included_subset = combine_prepared_subset_statistics(
        build_prepared_domain_statistics(included_store),
        subset_index,
    )
    assert included_subset.sha256 != original_subset.sha256
    assert included_subset.included_content_sha256 != original_subset.included_content_sha256
    assert included_subset.feature_means != original_subset.feature_means


def test_outputs_and_costs_are_excluded_from_fit_relevant_store_digest() -> None:
    examples = _examples()
    plan = _plan(examples)
    snapshot = _snapshot(examples)
    original = build_prepared_feature_store(
        examples,
        plan,
        embedding_snapshot=snapshot,
        expected_embedding_sha256=snapshot.sha256,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )
    changed_nonfit_fields = tuple(
        replace(
            example,
            candidate_models=tuple(
                replace(model, cost=Decimal("999") + index)
                for index, model in enumerate(example.candidate_models)
            ),
            outcomes=tuple(
                replace(
                    outcome,
                    output=f"ignored replacement output {index}",
                    cost=Decimal("777") + index,
                )
                for index, outcome in enumerate(example.outcomes)
            ),
        )
        for example in examples
    )
    changed = build_prepared_feature_store(
        changed_nonfit_fields,
        plan,
        embedding_snapshot=snapshot,
        expected_embedding_sha256=snapshot.sha256,
        expected_source_fit_sha256=prepared_fit_source_sha256(changed_nonfit_fields, plan),
    )

    assert prepared_fit_source_sha256(changed_nonfit_fields, plan) == (
        prepared_fit_source_sha256(examples, plan)
    )
    assert changed == original
    assert changed.sha256 == original.sha256
    assert build_prepared_domain_statistics(changed) == build_prepared_domain_statistics(original)


def test_reference_caps_fail_before_dense_allocation_or_row_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = _examples()
    plan = _plan(examples)
    snapshot = _snapshot(examples)
    source_digest = prepared_fit_source_sha256(examples, plan)

    def fail_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("dense allocation happened before numeric-byte preflight")

    def fail_row_traversal(*args: object, **kwargs: object) -> None:
        raise AssertionError("row traversal happened before numeric-byte preflight")

    monkeypatch.setattr(prepared_store_module, "MAX_PREPARED_REFERENCE_NUMERIC_BYTES", 1)
    monkeypatch.setattr(prepared_store_module, "bytearray", fail_allocation, raising=False)
    monkeypatch.setattr(
        prepared_store_module,
        "_validated_examples_for_store",
        fail_row_traversal,
    )
    with pytest.raises(ValueError, match="numeric-byte limit"):
        build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=snapshot,
            expected_embedding_sha256=snapshot.sha256,
            expected_source_fit_sha256=source_digest,
        )

    monkeypatch.undo()
    store = _store(examples, plan=plan)

    class NeverConstructAccumulator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("moment allocation happened before statistics preflight")

    monkeypatch.setattr(prepared_store_module, "MAX_PREPARED_REFERENCE_STATISTIC_SCALARS", 1)
    monkeypatch.setattr(prepared_store_module, "_MomentAccumulator", NeverConstructAccumulator)
    with pytest.raises(ValueError, match="scalar limit"):
        build_prepared_domain_statistics(store)

    monkeypatch.undo()
    dimension = store.plan.feature_count
    target_count = store.plan.target_count
    packed = dimension * (dimension + 1) // 2
    exact_work = store.plan.work.example_count * (
        3 * (dimension + target_count) + packed + dimension * target_count
    )
    monkeypatch.setattr(
        prepared_store_module,
        "MAX_PREPARED_REFERENCE_STATISTIC_WORK_UNITS",
        exact_work - 1,
    )
    monkeypatch.setattr(prepared_store_module, "_MomentAccumulator", NeverConstructAccumulator)
    with pytest.raises(ValueError, match="numeric-work limit"):
        build_prepared_domain_statistics(store)


def test_public_records_reject_noncanonical_payloads_and_impossible_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = _examples()
    snapshot = _snapshot(examples)
    signed_zero_embedding = bytearray(snapshot.payload)
    struct.pack_into("<d", signed_zero_embedding, 0, -0.0)
    with pytest.raises(ValueError, match="canonical positive zero"):
        replace(snapshot, payload=bytes(signed_zero_embedding))
    with pytest.raises(ValueError, match="wrong exact byte length"):
        replace(snapshot, payload=snapshot.payload[:-8])
    nonfinite_embedding = bytearray(snapshot.payload)
    struct.pack_into("<d", nonfinite_embedding, 0, math.inf)
    with pytest.raises(ValueError, match="finite binary64"):
        replace(snapshot, payload=bytes(nonfinite_embedding))

    store = _store(examples)
    signed_zero_targets = bytearray(store.target_payload)
    struct.pack_into("<d", signed_zero_targets, 0, -0.0)
    with pytest.raises(ValueError, match="canonical positive zero"):
        replace(store, target_payload=bytes(signed_zero_targets))
    with pytest.raises(ValueError, match="wrong exact byte length"):
        replace(store, feature_payload=store.feature_payload[:-8])
    nonfinite_features = bytearray(store.feature_payload)
    struct.pack_into("<d", nonfinite_features, 0, math.nan)
    with pytest.raises(ValueError, match="finite binary64"):
        replace(store, feature_payload=bytes(nonfinite_features))
    changed_embedding_tail = bytearray(store.feature_payload)
    embedding_offset = _UNIVERSAL_SURFACE_DIMENSION * 8
    original_embedding_value = struct.unpack_from("<d", changed_embedding_tail, embedding_offset)[0]
    struct.pack_into("<d", changed_embedding_tail, embedding_offset, original_embedding_value + 0.5)
    with pytest.raises(ValueError, match="snapshot SHA-256"):
        replace(store, feature_payload=bytes(changed_embedding_tail))
    with pytest.raises(ValueError, match="snapshot SHA-256"):
        replace(store, embedding_snapshot_sha256="0" * 64)

    with pytest.raises(ValueError, match="universal prepared layout"):
        prepared_store_module.PreparedDomainStatistics(
            domain_index=0,
            row_count=1,
            feature_count=1,
            target_count=1,
            active_tag_mask=0,
            content_sha256="0" * 64,
            feature_means=(0.0,),
            target_means=(0.0,),
            centered_xx_packed=(0.0,),
            centered_xy=(0.0,),
        )

    bundle = build_prepared_domain_statistics(store)
    with pytest.raises(ValueError, match="row_count exceeds"):
        replace(
            bundle.domain_statistics[0],
            row_count=prepared_store_module.MAX_PREPARED_EXAMPLES + 1,
        )
    with pytest.raises(ValueError, match="active_tag_mask"):
        replace(bundle.domain_statistics[0], active_tag_mask=0)

    subset = combine_prepared_subset_statistics(
        bundle,
        _subset_index(store.plan, ("alpha", "bravo")),
    )
    malformed_xx = list(subset.centered_xx_packed)
    malformed_xx[_packed_diagonal_index(store.plan.feature_count, 3)] = -1.0
    with pytest.raises(ValueError, match="diagonal must be non-negative"):
        replace(subset, centered_xx_packed=tuple(malformed_xx))

    monkeypatch.setattr(prepared_store_module, "MAX_PREPARED_REFERENCE_STATISTIC_SCALARS", 1)
    with pytest.raises(ValueError, match="bundle exceeds"):
        replace(bundle)


def test_semantic_metadata_is_bound_into_bundle_and_subset_digests() -> None:
    store = _store(_examples())
    bundle = build_prepared_domain_statistics(store)
    subset = combine_prepared_subset_statistics(
        bundle,
        _subset_index(store.plan, ("alpha", "bravo")),
    )
    alternate_plan = build_prepared_nested_lodo_plan(
        ("able", "baker", "charlie-two", "delta-two"),
        store.plan.domain_example_counts,
        feature_count=store.plan.feature_count,
        target_count=store.plan.target_count,
    )

    changed_models = replace(bundle, model_ids=("model-a", "model-b"))
    changed_embedding = replace(
        bundle,
        embedding_identity=_identity(revision="bundle-revision-two"),
    )
    changed_plan = replace(bundle, plan=alternate_plan)
    assert (
        len(
            {
                bundle.sha256,
                changed_models.sha256,
                changed_embedding.sha256,
                changed_plan.sha256,
            }
        )
        == 4
    )

    changed_schema = replace(
        subset.feature_schema,
        embedding_identity=_identity(revision="subset-revision-two"),
    )
    changed_subset_embedding = replace(subset, feature_schema=changed_schema)
    changed_subset_plan = replace(subset, plan=alternate_plan)
    assert changed_subset_embedding.sha256 != subset.sha256
    assert changed_subset_plan.sha256 != subset.sha256

    with pytest.raises(ValueError, match="domain_indices do not match"):
        replace(subset, domain_indices=(0, 2))
