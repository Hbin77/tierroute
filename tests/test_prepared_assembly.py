# SPDX-License-Identifier: Apache-2.0
"""Contract and numerical tests for bounded all-domain prepared assembly."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from decimal import Decimal

import pytest

import tierroute.predictors.prepared_assembly as assembly_module
from tierroute.core import ModelSpec
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample
from tierroute.features.surface import SURFACE_DOMAIN_TAG_CATALOGUE
from tierroute.predictors import fit_calibrated_bilinear
from tierroute.predictors.prepared_assembly import (
    assemble_prepared_bilinear_artifact,
    estimate_prepared_all_domain_assembly,
)
from tierroute.predictors.prepared_execution import (
    PreparedRawScoreBundle,
    build_prepared_coefficient_bundle,
    build_prepared_raw_score_bundle,
)
from tierroute.predictors.prepared_graph import build_prepared_nested_lodo_plan
from tierroute.predictors.prepared_store import (
    PreparedDomainStatisticsBundle,
    PreparedFeatureStore,
    build_prepared_domain_statistics,
    build_prepared_feature_store,
    prepared_fit_source_sha256,
)

_SURFACE_FEATURE_COUNT = 5 + len(SURFACE_DOMAIN_TAG_CATALOGUE)
_MODEL_IDS = ("cheap", "premium")
_DOMAIN_CASES = (
    ("code", "Debug this Python API implementation."),
    ("finance", "Analyze finance revenue carefully."),
    ("general", "Summarize an ordinary topic briefly."),
    ("law", "Review this legal statute carefully."),
    ("math", "Solve this algebra problem carefully."),
    ("medicine", "Assess this clinical diagnosis carefully."),
    ("science", "Explain this physics result carefully."),
)


@dataclass(frozen=True, slots=True)
class _Fixture:
    examples: tuple[EvaluationExample, ...]
    store: PreparedFeatureStore
    statistics: PreparedDomainStatisticsBundle
    raw_scores: PreparedRawScoreBundle


def _examples(domain_count: int) -> tuple[EvaluationExample, ...]:
    rows = []
    for domain_index, (domain, stem) in enumerate(_DOMAIN_CASES[:domain_count]):
        for row_index in range(2):
            ordinal = 2 * domain_index + row_index
            rows.append(
                EvaluationExample(
                    example_id=f"row-{domain_index}-{row_index}",
                    prompt=f"{stem} Case {row_index}.",
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
                            quality=0.18 + 0.055 * ordinal + 0.01 * (ordinal % 2),
                        ),
                        CandidateOutcome(
                            model_id="premium",
                            output="premium output",
                            cost=Decimal("2"),
                            quality=0.91 - 0.041 * ordinal + 0.015 * (ordinal % 3),
                        ),
                    ),
                )
            )
    return tuple(reversed(rows))


def _fixture(domain_count: int = 4) -> _Fixture:
    examples = _examples(domain_count)
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=_SURFACE_FEATURE_COUNT,
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
    return _Fixture(
        examples=examples,
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


@pytest.mark.parametrize("domain_count", [4, 5, 6, 7])
def test_all_domain_artifact_matches_authoritative_rowwise_training(
    domain_count: int,
) -> None:
    fixture = _fixture(domain_count)
    prepared = _assemble(fixture)
    reference = fit_calibrated_bilinear(fixture.examples)

    assert prepared.training_domains == reference.training_domains
    assert prepared.training_example_count == reference.training_example_count
    assert prepared.model_ids == reference.model_ids
    assert prepared.feature_schema.domain_tags == reference.feature_schema.domain_tags
    assert prepared.feature_schema.continuous_means == pytest.approx(
        reference.feature_schema.continuous_means,
        rel=1e-15,
        abs=1e-15,
    )
    assert prepared.feature_schema.continuous_scales == pytest.approx(
        reference.feature_schema.continuous_scales,
        rel=1e-15,
        abs=1e-15,
    )
    for model_id in prepared.model_ids:
        state = prepared.models[model_id]
        assert state.weights == pytest.approx(
            reference.model_weights[model_id],
            rel=1e-8,
            abs=1e-9,
        )
        assert state.bias == pytest.approx(
            reference.model_bias[model_id],
            rel=1e-9,
            abs=1e-10,
        )
        assert state.calibration.calibrator.upper_bounds == pytest.approx(
            reference.calibrators[model_id].upper_bounds,
            rel=1e-9,
            abs=1e-9,
        )
        assert state.calibration.calibrator.values == pytest.approx(
            reference.calibrators[model_id].values,
            rel=1e-15,
            abs=1e-15,
        )
    prompts = tuple(example.prompt for example in fixture.examples[:3])
    prepared_predictions = prepared.build_predictor().predict_batch(
        prompts,
        prepared.model_ids,
    )
    reference_predictions = reference.build_predictor().predict_batch(
        prompts,
        reference.model_ids,
    )
    for prepared_row, reference_row in zip(
        prepared_predictions,
        reference_predictions,
        strict=True,
    ):
        for model_id in prepared.model_ids:
            assert prepared_row[model_id] == pytest.approx(
                reference_row[model_id],
                rel=1e-15,
                abs=1e-15,
            )


def test_assembly_preserves_existing_graph_and_parent_identities() -> None:
    fixture = _fixture()
    plan = fixture.store.plan
    original = (
        plan.training_subsets,
        plan.score_blocks,
        fixture.store.sha256,
        fixture.statistics.sha256,
        fixture.raw_scores.sha256,
        fixture.raw_scores.coefficient_block_sha256s,
    )

    artifact = _assemble(fixture)

    assert (
        plan.training_subsets,
        plan.score_blocks,
        fixture.store.sha256,
        fixture.statistics.sha256,
        fixture.raw_scores.sha256,
        fixture.raw_scores.coefficient_block_sha256s,
    ) == original
    assert len(artifact.lineage.calibration_sources) == len(plan.domains)
    assert (
        sum(source.row_count for source in artifact.lineage.calibration_sources)
        == plan.work.example_count
    )


def test_semantic_selector_is_independent_of_context_tuple_position() -> None:
    fixture = _fixture(7)
    plan = fixture.store.plan
    subset_contexts = tuple(
        reversed(
            tuple(
                (index, subset.domain_indices) for index, subset in enumerate(plan.training_subsets)
            )
        )
    )
    block_contexts = tuple(
        reversed(
            tuple(
                (
                    index,
                    block.training_subset_index,
                    block.scored_domain_index,
                )
                for index, block in enumerate(plan.score_blocks)
            )
        )
    )

    selected = assembly_module._select_semantic_context_indices(
        len(plan.domains),
        subset_contexts,
        block_contexts,
    )

    for held_out, (subset_index, block_index) in enumerate(selected):
        assert plan.training_subsets[subset_index].domain_indices == tuple(
            index for index in range(len(plan.domains)) if index != held_out
        )
        assert (
            plan.score_blocks[block_index].training_subset_index,
            plan.score_blocks[block_index].scored_domain_index,
        ) == (subset_index, held_out)


def test_semantic_selector_rejects_missing_and_duplicate_contexts() -> None:
    fixture = _fixture()
    plan = fixture.store.plan
    subsets = tuple(
        (index, subset.domain_indices) for index, subset in enumerate(plan.training_subsets)
    )
    blocks = tuple(
        (index, block.training_subset_index, block.scored_domain_index)
        for index, block in enumerate(plan.score_blocks)
    )

    with pytest.raises(ValueError, match="lack an exact"):
        assembly_module._select_semantic_context_indices(
            len(plan.domains),
            subsets,
            blocks[:-1],
        )
    with pytest.raises(ValueError, match="duplicate"):
        assembly_module._select_semantic_context_indices(
            len(plan.domains),
            subsets,
            (*blocks, blocks[0]),
        )


@pytest.mark.parametrize(
    "pin_name",
    [
        "expected_source_fit_sha256",
        "expected_store_sha256",
        "expected_statistics_sha256",
        "expected_raw_score_sha256",
    ],
)
def test_each_trusted_pin_fails_before_resnapshot(
    pin_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    pins = {
        "expected_source_fit_sha256": fixture.store.source_fit_sha256,
        "expected_store_sha256": fixture.store.sha256,
        "expected_statistics_sha256": fixture.statistics.sha256,
        "expected_raw_score_sha256": fixture.raw_scores.sha256,
    }
    pins[pin_name] = "0" * 64 if pins[pin_name] != "0" * 64 else "1" * 64

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("trusted-pin rejection reached the resnapshot stage")

    monkeypatch.setattr(assembly_module, "_resnapshot_inputs", forbidden)
    with pytest.raises(ValueError, match="trusted"):
        assemble_prepared_bilinear_artifact(
            fixture.store,
            fixture.statistics,
            fixture.raw_scores,
            **pins,
        )


def test_closed_form_estimate_is_finite_and_precedes_numeric_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("estimate traversed a numeric row")

    monkeypatch.setattr(PreparedFeatureStore, "feature_row", forbidden)
    monkeypatch.setattr(PreparedFeatureStore, "target_row", forbidden)
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )

    assert estimate.total_work_units > 0
    assert estimate.modeled_bytes > 0
    assert estimate.active_feature_count == 5 + len(fixture.store.plan.domains)
    assert math.isfinite(float(estimate.total_work_units))


@pytest.mark.parametrize(
    ("cap_name", "estimate_name"),
    [
        ("MAX_PREPARED_ASSEMBLY_WORK_UNITS", "total_work_units"),
        ("MAX_PREPARED_ASSEMBLY_MODELED_BYTES", "modeled_bytes"),
        ("MAX_PREPARED_ASSEMBLY_OBJECT_BYTES", "object_amplification_bytes"),
        ("MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS", "retained_numeric_scalars"),
        ("MAX_PREDICTOR_ARTIFACT_BYTES", "canonical_json_upper_bound_bytes"),
    ],
)
def test_assembly_caps_accept_exact_boundary_and_reject_one_over_before_resnapshot(
    cap_name: str,
    estimate_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    estimate = estimate_prepared_all_domain_assembly(
        fixture.store,
        fixture.statistics,
        fixture.raw_scores,
    )
    exact = getattr(estimate, estimate_name)
    monkeypatch.setattr(assembly_module, cap_name, exact)
    assert (
        estimate_prepared_all_domain_assembly(
            fixture.store,
            fixture.statistics,
            fixture.raw_scores,
        )
        == estimate
    )
    monkeypatch.setattr(assembly_module, cap_name, exact - 1)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("resource rejection reached the resnapshot stage")

    monkeypatch.setattr(assembly_module, "_resnapshot_inputs", forbidden)
    with pytest.raises(ValueError):
        _assemble(fixture)


def _changed_f64_payload(payload: bytes) -> bytes:
    changed = bytearray(payload)
    value = struct.unpack_from("<d", changed, 0)[0]
    struct.pack_into("<d", changed, 0, value + 0.125)
    return bytes(changed)


@pytest.mark.parametrize(
    "mutation",
    ["store", "statistics", "coefficient", "feature_shard", "raw_score"],
)
def test_stale_cached_identity_mutations_fail_before_cross_parent_work(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    if mutation == "store":
        object.__setattr__(
            fixture.store,
            "target_payload",
            _changed_f64_payload(fixture.store.target_payload),
        )
    elif mutation == "statistics":
        child = fixture.statistics.domain_statistics[0]
        means = (child.target_means[0] + 0.125, *child.target_means[1:])
        object.__setattr__(child, "target_means", means)
    elif mutation == "coefficient":
        block = fixture.raw_scores.coefficients.blocks[0]
        object.__setattr__(
            block,
            "weights_payload",
            _changed_f64_payload(block.weights_payload),
        )
    elif mutation == "feature_shard":
        shard = fixture.raw_scores.feature_shards.shards[0]
        object.__setattr__(
            shard,
            "prompt_sha256s",
            ("f" * 64, *shard.prompt_sha256s[1:]),
        )
    else:
        block = fixture.raw_scores.blocks[0]
        object.__setattr__(
            block,
            "scores_payload",
            _changed_f64_payload(block.scores_payload),
        )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("stale mutation reached cross-parent validation")

    monkeypatch.setattr(
        assembly_module,
        "_validate_cross_parent_associations",
        forbidden,
    )
    with pytest.raises(ValueError):
        _assemble(fixture)


@pytest.mark.parametrize("identity", ["solver", "scorer"])
def test_mutated_init_false_execution_identity_fails_before_resnapshot(
    identity: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    if identity == "solver":
        object.__setattr__(
            fixture.raw_scores.coefficients.blocks[0],
            "solver_id",
            "unexpected-solver",
        )
    else:
        object.__setattr__(
            fixture.raw_scores.blocks[0],
            "scorer_id",
            "unexpected-scorer",
        )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("invalid frozen identity reached resnapshot")

    monkeypatch.setattr(assembly_module, "_resnapshot_inputs", forbidden)
    with pytest.raises(ValueError, match="frozen identity"):
        _assemble(fixture)
