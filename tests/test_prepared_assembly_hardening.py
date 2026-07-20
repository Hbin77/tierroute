# SPDX-License-Identifier: Apache-2.0
"""Resource-accounting and fail-order hardening for prepared assembly."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal

import pytest

import tierroute.predictors.prepared_assembly as assembly_module
from tierroute.core import ModelSpec
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features import SURFACE_DOMAIN_TAG_CATALOGUE, EmbeddingIdentity
from tierroute.predictors.prepared_assembly import (
    assemble_prepared_bilinear_artifact,
    estimate_prepared_all_domain_assembly,
)
from tierroute.predictors.prepared_execution import (
    PreparedRawScoreBundle,
    PreparedScoredFeatureShardBundle,
    build_prepared_coefficient_bundle,
    build_prepared_raw_score_bundle,
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

_F64_BYTES = 8
_OBJECT_BYTES = 64
_MODEL_IDS = ("cheap", "premium")
_DOMAINS = ("alpha", "bravo", "charlie", "delta")
_UNIVERSAL_SURFACE_DIMENSION = 5 + len(SURFACE_DOMAIN_TAG_CATALOGUE)


@dataclass(frozen=True, slots=True)
class _Fixture:
    store: PreparedFeatureStore
    statistics: PreparedDomainStatisticsBundle
    raw_scores: PreparedRawScoreBundle


def _examples(counts: tuple[int, ...]) -> tuple[EvaluationExample, ...]:
    rows = []
    ordinal = 0
    for domain, count in zip(_DOMAINS, counts, strict=True):
        for domain_row in range(count):
            prompt = f"Debug this Python function for {domain} case {domain_row}."
            rows.append(
                EvaluationExample(
                    example_id=f"{domain}-{domain_row:03d}",
                    prompt=prompt,
                    domain=domain,
                    candidate_models=(
                        ModelSpec("premium", Decimal("2")),
                        ModelSpec("cheap", Decimal("1")),
                    ),
                    outcomes=(
                        CandidateOutcome(
                            model_id="cheap",
                            output="cheap output",
                            cost=Decimal("1"),
                            quality=0.20 + 0.025 * ordinal,
                        ),
                        CandidateOutcome(
                            model_id="premium",
                            output="premium output",
                            cost=Decimal("2"),
                            quality=0.90 - 0.017 * ordinal,
                        ),
                    ),
                )
            )
            ordinal += 1
    return tuple(reversed(rows))


def _fixture(
    counts: tuple[int, ...] = (2, 2, 2, 2),
    *,
    embedding_identity: EmbeddingIdentity | None = None,
) -> _Fixture:
    examples = _examples(counts)
    embedding_dimension = int(embedding_identity is not None)
    plan = build_prepared_nested_lodo_plan(
        _DOMAINS,
        counts,
        feature_count=_UNIVERSAL_SURFACE_DIMENSION + embedding_dimension,
        target_count=len(_MODEL_IDS),
    )
    source_sha256 = prepared_fit_source_sha256(examples, plan)
    if embedding_identity is None:
        store = build_prepared_feature_store(
            examples,
            plan,
            expected_source_fit_sha256=source_sha256,
        )
    else:
        snapshot = build_prepared_embedding_snapshot(
            tuple(
                PreparedEmbeddingInput(
                    example_id=example.example_id,
                    prompt_sha256=hashlib.sha256(example.prompt.encode("utf-8")).hexdigest(),
                    values=(float(index + 1),),
                )
                for index, example in enumerate(examples)
            ),
            embedding_identity,
            dimension=1,
        )
        store = build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=snapshot,
            expected_embedding_sha256=snapshot.sha256,
            expected_source_fit_sha256=source_sha256,
        )
    statistics = build_prepared_domain_statistics(store)
    coefficients = build_prepared_coefficient_bundle(store, statistics, ridge=1.0)
    return _Fixture(
        store=store,
        statistics=statistics,
        raw_scores=build_prepared_raw_score_bundle(store, coefficients),
    )


def _assemble(fixture: _Fixture):
    return assemble_prepared_bilinear_artifact(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
        expected_source_fit_sha256=fixture.store.source_fit_sha256,
        expected_store_sha256=fixture.store.sha256,
        expected_statistics_sha256=fixture.statistics.sha256,
        expected_raw_score_sha256=fixture.raw_scores.sha256,
    )


def test_retained_numeric_scalar_formula_and_cap_fail_before_resnapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )
    n = fixture.store.plan.work.example_count
    m = fixture.store.plan.target_count
    expected = 7 + m * (estimate.active_feature_count + 1) + 2 * n * m

    assert estimate.retained_numeric_scalars == expected
    monkeypatch.setattr(
        assembly_module,
        "MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS",
        expected - 1,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("numeric-cap rejection reached resnapshot")

    monkeypatch.setattr(assembly_module, "_resnapshot_inputs", forbidden)
    with pytest.raises(ValueError, match="numeric-scalar"):
        _assemble(fixture)


def test_resnapshot_work_units_cover_every_contractual_pass() -> None:
    fixture = _fixture()
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )
    plan = fixture.store.plan
    n = plan.work.example_count
    d = plan.feature_count
    m = plan.target_count
    domain_count = len(plan.domains)
    statistic_scalars = d + m + d * (d + 1) // 2 + d * m
    coefficient_cells = (
        sum(
            len(block.weights_payload) + len(block.intercepts_payload)
            for block in fixture.raw_scores.coefficients.blocks
        )
        // _F64_BYTES
    )
    score_cells = (
        sum(len(block.scores_payload) for block in fixture.raw_scores.blocks) // _F64_BYTES
    )
    contractual_minimum = (
        2 * n * (d + m)
        + n * fixture.store.embedding_dimension
        + 2 * domain_count * statistic_scalars
        + 2 * coefficient_cells
        + 2 * score_cells
        + n * d
    )

    assert estimate.resnapshot_work_units >= contractual_minimum


def test_skewed_domain_target_peak_uses_larger_of_full_and_double_shard() -> None:
    fixture = _fixture((1, 1, 1, 7))
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )
    plan = fixture.store.plan
    n = plan.work.example_count
    m = plan.target_count

    assert 2 * max(plan.domain_example_counts) > n
    assert estimate.target_shard_bytes == max(
        n * m * _F64_BYTES,
        2 * max(plan.domain_example_counts) * m * _F64_BYTES,
    )


def test_object_amplification_components_sum_and_cap_fail_before_resnapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )
    plan = fixture.store.plan
    n = plan.work.example_count
    m = plan.target_count
    domain_count = len(plan.domains)
    components = (
        estimate.statistics_resnapshot_object_bytes,
        estimate.aggregate_object_bytes,
        estimate.solve_object_bytes,
        estimate.calibration_object_bytes,
    )
    child_objects = (
        1
        + domain_count
        + 1
        + len(plan.training_subsets)
        + 1
        + domain_count
        + 1
        + len(plan.score_blocks)
    )
    row_objects = 4 * n + 4 * n * m
    structural_bytes = (child_objects + row_objects + 8 * domain_count + 8 * m) * _OBJECT_BYTES

    assert all(component > 0 for component in components)
    assert estimate.object_amplification_bytes == structural_bytes + sum(components)

    monkeypatch.setattr(
        assembly_module,
        "MAX_PREPARED_ASSEMBLY_OBJECT_BYTES",
        estimate.object_amplification_bytes,
    )
    assert (
        estimate_prepared_all_domain_assembly(
            fixture.store,
            fixture.statistics,
            fixture.raw_scores,
        ).object_amplification_bytes
        == estimate.object_amplification_bytes
    )
    monkeypatch.setattr(
        assembly_module,
        "MAX_PREPARED_ASSEMBLY_OBJECT_BYTES",
        estimate.object_amplification_bytes - 1,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("object-cap rejection reached resnapshot")

    monkeypatch.setattr(assembly_module, "_resnapshot_inputs", forbidden)
    with pytest.raises(ValueError, match="object amplification"):
        _assemble(fixture)


def test_escape_heavy_max_embedding_identity_fits_json_estimate() -> None:
    worst_case_json_character = "\x01"
    max_field = worst_case_json_character * 4096
    fixture = _fixture(
        embedding_identity=EmbeddingIdentity(
            provider=max_field,
            model_id=max_field,
            revision=max_field,
            pooling=max_field,
            normalize=False,
            asset_manifest_sha256="a" * 64,
        )
    )
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )

    artifact = _assemble(fixture)

    assert len(artifact.to_json().encode("utf-8")) <= estimate.canonical_json_upper_bound_bytes


def test_twice_serialized_escape_heavy_domains_fit_json_estimate() -> None:
    long_domains = tuple("\x01" * 4095 + suffix for suffix in "abcd")
    domain_by_original = dict(zip(_DOMAINS, long_domains, strict=True))
    examples = tuple(
        replace(example, domain=domain_by_original[example.domain])
        for example in _examples((1, 1, 1, 1))
    )
    plan = build_prepared_nested_lodo_plan(
        long_domains,
        (1, 1, 1, 1),
        feature_count=_UNIVERSAL_SURFACE_DIMENSION,
        target_count=len(_MODEL_IDS),
    )
    source_sha256 = prepared_fit_source_sha256(examples, plan)
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=source_sha256,
    )
    statistics = build_prepared_domain_statistics(store)
    coefficients = build_prepared_coefficient_bundle(store, statistics, ridge=1.0)
    fixture = _Fixture(
        store=store,
        statistics=statistics,
        raw_scores=build_prepared_raw_score_bundle(store, coefficients),
    )
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )

    artifact = _assemble(fixture)

    assert len(artifact.to_json().encode("utf-8")) <= estimate.canonical_json_upper_bound_bytes


def _track_second_pin_and_forbid_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    compare_calls = [0]
    original_compare = assembly_module._compare_cached_pins

    def tracked_compare(*args: object, **kwargs: object) -> None:
        compare_calls[0] += 1
        original_compare(*args, **kwargs)

    def forbidden_aggregate(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("invalid semantic input reached all-domain aggregation")

    monkeypatch.setattr(assembly_module, "_compare_cached_pins", tracked_compare)
    monkeypatch.setattr(
        assembly_module,
        "_combine_all_domain_statistics",
        forbidden_aggregate,
    )
    return compare_calls


def test_full_domain_aggregate_combines_children_in_ascending_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    visited: list[int] = []
    original_combine = assembly_module._combine_domain_statistics

    def recording_combine(*args: object, **kwargs: object) -> int:
        right = args[-1]
        visited.append(right.domain_index)  # type: ignore[attr-defined]
        return original_combine(*args, **kwargs)

    monkeypatch.setattr(
        assembly_module,
        "_combine_domain_statistics",
        recording_combine,
    )

    _assemble(fixture)

    assert visited == list(range(len(fixture.store.plan.domains)))


def test_statistics_child_permutation_is_rejected_during_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    object.__setattr__(
        fixture.statistics,
        "domain_statistics",
        tuple(reversed(fixture.statistics.domain_statistics)),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("permuted statistics reached resnapshot")

    monkeypatch.setattr(assembly_module, "_resnapshot_inputs", forbidden)
    with pytest.raises(ValueError, match="malformed bounded shape"):
        _assemble(fixture)


def test_statistics_child_numeric_tamper_fails_at_second_pin_before_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    compare_calls = _track_second_pin_and_forbid_aggregate(monkeypatch)
    child = fixture.statistics.domain_statistics[0]
    object.__setattr__(
        child,
        "target_means",
        (child.target_means[0] + 0.125, *child.target_means[1:]),
    )

    with pytest.raises(ValueError, match="trusted bundle"):
        _assemble(fixture)

    assert compare_calls == [2]


def _negative_zero_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(-0.0 if value == 0.0 else value for value in values)


def _assert_positive_zero(values: tuple[float, ...]) -> None:
    assert all(math.copysign(1.0, value) > 0.0 for value in values if value == 0.0)


def test_negative_zero_statistics_canonicalize_to_same_positive_zero_hashes() -> None:
    fixture = _fixture()
    normalized_children = tuple(
        replace(
            child,
            feature_means=_negative_zero_tuple(child.feature_means),
            target_means=_negative_zero_tuple(child.target_means),
            centered_xx_packed=_negative_zero_tuple(child.centered_xx_packed),
            centered_xy=_negative_zero_tuple(child.centered_xy),
        )
        for child in fixture.statistics.domain_statistics
    )
    normalized_statistics = replace(
        fixture.statistics,
        domain_statistics=normalized_children,
    )
    normalized_fixture = _Fixture(
        store=fixture.store,
        statistics=normalized_statistics,
        raw_scores=fixture.raw_scores,
    )

    assert any(
        value == 0.0
        for child in fixture.statistics.domain_statistics
        for value in (*child.centered_xx_packed, *child.centered_xy)
    )
    for original, normalized in zip(
        fixture.statistics.domain_statistics,
        normalized_children,
        strict=True,
    ):
        assert normalized.sha256 == original.sha256
        _assert_positive_zero(normalized.feature_means)
        _assert_positive_zero(normalized.target_means)
        _assert_positive_zero(normalized.centered_xx_packed)
        _assert_positive_zero(normalized.centered_xy)
    assert normalized_statistics.sha256 == fixture.statistics.sha256

    original_aggregate = assembly_module._combine_all_domain_statistics(
        fixture.store,
        fixture.statistics,
    )
    normalized_aggregate = assembly_module._combine_all_domain_statistics(
        fixture.store,
        normalized_statistics,
    )
    assert normalized_aggregate.sha256 == original_aggregate.sha256
    _assert_positive_zero(normalized_aggregate.feature_means)
    _assert_positive_zero(normalized_aggregate.target_means)
    _assert_positive_zero(normalized_aggregate.centered_xx_packed)
    _assert_positive_zero(normalized_aggregate.centered_xy)

    original_artifact = _assemble(fixture)
    normalized_artifact = _assemble(normalized_fixture)
    assert (
        normalized_artifact.lineage.aggregate_statistics_sha256
        == original_artifact.lineage.aggregate_statistics_sha256
    )
    assert (
        normalized_artifact.lineage.final_coefficient_sha256
        == original_artifact.lineage.final_coefficient_sha256
    )
    artifact_values = tuple(
        value
        for state in normalized_artifact.models.values()
        for value in (
            *state.weights,
            state.bias,
            *state.calibration.calibrator.upper_bounds,
            *state.calibration.calibrator.values,
        )
    )
    assert any(value == 0.0 for value in artifact_values)
    _assert_positive_zero(artifact_values)


def test_assembly_does_not_mutate_existing_prepared_graph() -> None:
    fixture = _fixture()
    plan = fixture.store.plan
    before = (
        plan.domains,
        plan.domain_example_counts,
        plan.training_subsets,
        plan.score_blocks,
        plan.work,
        plan.algorithm_id,
    )

    _assemble(fixture)

    assert (
        plan.domains,
        plan.domain_example_counts,
        plan.training_subsets,
        plan.score_blocks,
        plan.work,
        plan.algorithm_id,
    ) == before
    assert fixture.store.plan is plan
    assert fixture.statistics.plan is plan
    assert fixture.raw_scores.plan is plan


def test_coefficient_and_statistics_masks_disagree_after_second_pin_before_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    compare_calls = _track_second_pin_and_forbid_aggregate(monkeypatch)
    original_resnapshot = assembly_module._resnapshot_inputs

    def tampered_resnapshot(*args: object, **kwargs: object):
        fresh_store, fresh_statistics, fresh_raw_scores = original_resnapshot(
            *args,
            **kwargs,
        )
        masks = fresh_raw_scores.coefficients.domain_active_tag_masks
        assert masks[0] != 0
        object.__setattr__(
            fresh_raw_scores.coefficients,
            "domain_active_tag_masks",
            (0, *masks[1:]),
        )
        return fresh_store, fresh_statistics, fresh_raw_scores

    monkeypatch.setattr(assembly_module, "_resnapshot_inputs", tampered_resnapshot)
    with pytest.raises(ValueError, match="one exact store/plan layout"):
        _assemble(fixture)

    assert compare_calls == [2]


_SemanticTamper = Callable[
    [PreparedFeatureStore, PreparedRawScoreBundle, PreparedScoredFeatureShardBundle],
    None,
]


def _tamper_semantic_join(
    kind: str,
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
    rebuilt: PreparedScoredFeatureShardBundle,
) -> None:
    plan = store.plan
    contexts = assembly_module._select_semantic_context_indices(
        len(plan.domains),
        tuple((index, subset.domain_indices) for index, subset in enumerate(plan.training_subsets)),
        tuple(
            (
                index,
                block.training_subset_index,
                block.scored_domain_index,
            )
            for index, block in enumerate(plan.score_blocks)
        ),
    )
    _, block_index = contexts[0]
    raw_block = raw_scores.blocks[block_index]
    feature_shard = rebuilt.shards[0]

    if kind == "coefficient_hash":
        replacement = "f" * 64 if raw_block.coefficient_block_sha256 != "f" * 64 else "e" * 64
        object.__setattr__(raw_block, "coefficient_block_sha256", replacement)
    elif kind == "shard_hash":
        replacement = "f" * 64 if raw_block.scored_feature_shard_sha256 != "f" * 64 else "e" * 64
        object.__setattr__(raw_block, "scored_feature_shard_sha256", replacement)
    elif kind == "row":
        object.__setattr__(feature_shard, "row_count", feature_shard.row_count + 1)
    elif kind == "prompt":
        replacement = "0" * 64 if feature_shard.prompt_sha256s[0] != "0" * 64 else "1" * 64
        object.__setattr__(
            feature_shard,
            "prompt_sha256s",
            (replacement, *feature_shard.prompt_sha256s[1:]),
        )
    elif kind == "duplicate":
        object.__setattr__(
            feature_shard,
            "example_ids",
            (feature_shard.example_ids[0], feature_shard.example_ids[0]),
        )
    elif kind == "missing":
        object.__setattr__(
            feature_shard,
            "example_ids",
            ("not-a-store-row", *feature_shard.example_ids[1:]),
        )
    elif kind == "reordered":
        object.__setattr__(
            feature_shard,
            "example_ids",
            tuple(reversed(feature_shard.example_ids)),
        )
        object.__setattr__(
            feature_shard,
            "prompt_sha256s",
            tuple(reversed(feature_shard.prompt_sha256s)),
        )
    elif kind == "substituted":
        substitute = next(
            block
            for index, block in enumerate(raw_scores.blocks)
            if index != block_index and block.row_count == raw_block.row_count
        )
        blocks = list(raw_scores.blocks)
        blocks[block_index] = substitute
        object.__setattr__(raw_scores, "blocks", tuple(blocks))
    elif kind == "wrong_domain":
        other_shard = rebuilt.shards[1]
        object.__setattr__(
            feature_shard,
            "example_ids",
            (other_shard.example_ids[0], *feature_shard.example_ids[1:]),
        )
        object.__setattr__(
            feature_shard,
            "prompt_sha256s",
            (other_shard.prompt_sha256s[0], *feature_shard.prompt_sha256s[1:]),
        )
    elif kind == "cross_context":
        graph_block = plan.score_blocks[block_index]
        object.__setattr__(
            graph_block,
            "training_subset_index",
            (graph_block.training_subset_index + 1) % len(plan.training_subsets),
        )
    else:  # pragma: no cover - parametrization is closed below
        raise AssertionError(f"unknown semantic tamper: {kind}")


@pytest.mark.parametrize(
    "kind",
    [
        "coefficient_hash",
        "shard_hash",
        "row",
        "prompt",
        "duplicate",
        "missing",
        "reordered",
        "substituted",
        "wrong_domain",
        "cross_context",
    ],
)
def test_each_semantic_d_join_corruption_fails_after_second_pin_before_aggregate(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    compare_calls = _track_second_pin_and_forbid_aggregate(monkeypatch)
    original_cross_parent = assembly_module._validate_cross_parent_associations

    def tampered_cross_parent(
        store: PreparedFeatureStore,
        statistics: PreparedDomainStatisticsBundle,
        raw_scores: PreparedRawScoreBundle,
    ) -> PreparedScoredFeatureShardBundle:
        rebuilt = original_cross_parent(store, statistics, raw_scores)
        _tamper_semantic_join(kind, store, raw_scores, rebuilt)
        return rebuilt

    monkeypatch.setattr(
        assembly_module,
        "_validate_cross_parent_associations",
        tampered_cross_parent,
    )
    with pytest.raises(ValueError):
        _assemble(fixture)

    assert compare_calls == [2]
