# SPDX-License-Identifier: Apache-2.0
"""Numerical and one-shot route acceptance for prepared all-domain assembly."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cache, lru_cache
from types import MappingProxyType

import pytest

from tierroute.adapters import PerQueryBudgetLedger
from tierroute.core import (
    BudgetTier,
    CallModel,
    CallRecord,
    ModelSpec,
    RouterState,
    SelectOutput,
)
from tierroute.eval import (
    CandidateOutcome,
    EvaluationExample,
    OfflineSimulator,
    TierSpec,
)
from tierroute.features import (
    SURFACE_DOMAIN_TAG_CATALOGUE,
    EmbeddingIdentity,
    PromptFeatureEncoder,
    extract_surface_features,
)
from tierroute.policies.lambda_threshold import LambdaThresholdRouter
from tierroute.predictors import IsotonicCalibrator, fit_calibrated_bilinear
from tierroute.predictors._ridge import RidgeSolution, solve_centered_ridge
from tierroute.predictors.prepared_artifacts import PreparedBilinearPredictorArtifact
from tierroute.predictors.prepared_assembly import assemble_prepared_bilinear_artifact
from tierroute.predictors.prepared_execution import (
    PreparedRawScoreBundle,
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

_MODEL_IDS = ("cheap", "premium")
_UNIVERSAL_SURFACE_DIMENSION = 5 + len(SURFACE_DOMAIN_TAG_CATALOGUE)
_RIDGE = 1.0
_ORDINARY_SCHEMA_TOLERANCE = {"rel": 1e-15, "abs": 1e-15}
_ORDINARY_WEIGHT_TOLERANCE = {"rel": 1e-8, "abs": 1e-9}
_ORDINARY_BIAS_TOLERANCE = {"rel": 1e-9, "abs": 1e-10}
_ORDINARY_RAW_TOLERANCE = {"rel": 1e-9, "abs": 1e-9}
_STRICT_CALIBRATION_TOLERANCE = {"rel": 1e-15, "abs": 1e-15}
_STRESS_WEIGHT_TOLERANCE = {"rel": 1e-6, "abs": 1e-15}
_STRESS_BIAS_TOLERANCE = {"rel": 1e-7, "abs": 1e-8}
_STRESS_RAW_TOLERANCE = {"rel": 1e-7, "abs": 1e-8}

_TAG_PROMPT_STEMS = MappingProxyType(
    {
        "code": "Debug this Python API implementation.",
        "finance": "Analyze finance revenue carefully.",
        "general": "Summarize an ordinary topic briefly.",
        "law": "Review this legal statute carefully.",
        "math": "Solve this algebra equation carefully.",
        "medicine": "Assess this clinical diagnosis carefully.",
        "science": "Explain this physics experiment carefully.",
    }
)
_PROBE_PROMPTS = (
    "Debug this Python API in detail.",
    "Compare legal contract clauses.",
    "Give a short neutral explanation.",
)


@dataclass(frozen=True, slots=True)
class _FixtureEmbeddingProvider:
    dimension: int
    identity: EmbeddingIdentity
    vectors_by_prompt: MappingProxyType

    def embed(self, texts: object) -> tuple[tuple[float, ...], ...]:
        return tuple(self.vectors_by_prompt[text] for text in texts)  # type: ignore[index]


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    examples: tuple[EvaluationExample, ...]
    store: PreparedFeatureStore
    statistics: PreparedDomainStatisticsBundle
    raw_scores: PreparedRawScoreBundle
    artifact: PreparedBilinearPredictorArtifact
    embedding_provider: _FixtureEmbeddingProvider | None = None


@dataclass(frozen=True, slots=True)
class _RowwiseBase:
    encoder: PromptFeatureEncoder
    model_ids: tuple[str, ...]
    solution: RidgeSolution


@dataclass(frozen=True, slots=True)
class _CalibrationSample:
    example_id: str
    prediction: float
    target: float


@dataclass(frozen=True, slots=True)
class _PavBlock:
    upper_bound: float
    value: float
    example_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MutablePavBlock:
    upper_bound: float
    target_sum: Fraction
    count: int
    example_ids: tuple[str, ...]

    @property
    def exact_mean(self) -> Fraction:
        return self.target_sum / self.count


def _ordinary_examples(
    domain_count: int,
    *,
    constant_targets: bool = False,
) -> tuple[EvaluationExample, ...]:
    tags = SURFACE_DOMAIN_TAG_CATALOGUE[:domain_count]
    rows = []
    for domain_index, tag in enumerate(tags):
        for row_index in range(2):
            ordinal = 2 * domain_index + row_index
            prompt = f"{_TAG_PROMPT_STEMS[tag]} Case {row_index}." + (
                " Explain the result briefly." if row_index else ""
            )
            assert extract_surface_features(prompt).domain_tags == (tag,)
            qualities = (
                {"cheap": 0.25, "premium": 0.75}
                if constant_targets
                else {
                    "cheap": 0.17 + 0.047 * ordinal + 0.013 * ((domain_index + row_index) % 3),
                    "premium": 0.93
                    - 0.036 * ordinal
                    + 0.011 * ((2 * domain_index + row_index) % 3),
                }
            )
            rows.append(
                EvaluationExample(
                    example_id=f"row-{domain_index:02d}-{row_index}",
                    prompt=prompt,
                    domain=tag,
                    candidate_models=(
                        ModelSpec("premium", Decimal("2.375")),
                        ModelSpec("cheap", Decimal("1.125")),
                    ),
                    outcomes=tuple(
                        CandidateOutcome(
                            model_id=model_id,
                            output=f"{model_id} output for {ordinal}",
                            cost=(Decimal("1.125") if model_id == "cheap" else Decimal("2.375")),
                            quality=qualities[model_id],
                        )
                        for model_id in _MODEL_IDS
                    ),
                )
            )
    return tuple(reversed(rows))


def _embedding_identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider="tierroute-test-fixture",
        model_id="project-authored-high-dynamic",
        revision="fixture-v1",
        pooling="mean",
        normalize=False,
        asset_manifest_sha256="c" * 64,
    )


def _stress_examples_and_provider() -> tuple[
    tuple[EvaluationExample, ...],
    _FixtureEmbeddingProvider,
]:
    identity = _embedding_identity()
    examples = []
    vectors: dict[str, tuple[float, ...]] = {}
    domains = ("alpha", "bravo", "charlie", "delta")
    for offset in range(8):
        ordinal = 101 + offset
        centered = float(ordinal - 104)
        prompt = f"plain item {ordinal}"
        vector = (
            centered * 1_000.0,
            centered * 2_000.0,
            centered * 0.000_001,
            0.0,
        )
        vectors[prompt] = vector
        examples.append(
            EvaluationExample(
                example_id=f"stress-{ordinal}",
                prompt=prompt,
                domain=domains[offset % len(domains)],
                candidate_models=(
                    ModelSpec("premium", Decimal("2")),
                    ModelSpec("cheap", Decimal("1")),
                ),
                outcomes=(
                    CandidateOutcome(
                        "premium",
                        "premium stress output",
                        Decimal("2"),
                        0.2 + ordinal * 0.003,
                    ),
                    CandidateOutcome(
                        "cheap",
                        "cheap stress output",
                        Decimal("1"),
                        0.5,
                    ),
                ),
            )
        )
    vectors.update({prompt: (0.0, 0.0, 0.0, 0.0) for prompt in _PROBE_PROMPTS})
    provider = _FixtureEmbeddingProvider(
        dimension=4,
        identity=identity,
        vectors_by_prompt=MappingProxyType(vectors),
    )
    return tuple(reversed(examples)), provider


def _build_case(
    examples: tuple[EvaluationExample, ...],
    *,
    embedding_provider: _FixtureEmbeddingProvider | None = None,
) -> _PreparedCase:
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    embedding_dimension = 0 if embedding_provider is None else embedding_provider.dimension
    plan = build_prepared_nested_lodo_plan(
        tuple(reversed(domains)),
        tuple(reversed(counts)),
        feature_count=_UNIVERSAL_SURFACE_DIMENSION + embedding_dimension,
        target_count=len(_MODEL_IDS),
    )
    source_sha256 = prepared_fit_source_sha256(examples, plan)
    if embedding_provider is None:
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
                    values=embedding_provider.vectors_by_prompt[example.prompt],
                )
                for example in examples
            ),
            embedding_provider.identity,
            dimension=embedding_provider.dimension,
        )
        store = build_prepared_feature_store(
            examples,
            plan,
            embedding_snapshot=snapshot,
            expected_embedding_sha256=snapshot.sha256,
            expected_source_fit_sha256=source_sha256,
        )
    statistics = build_prepared_domain_statistics(store)
    coefficients = build_prepared_coefficient_bundle(store, statistics, ridge=_RIDGE)
    raw_scores = build_prepared_raw_score_bundle(store, coefficients)
    artifact = assemble_prepared_bilinear_artifact(
        store,
        statistics,
        raw_scores,
        expected_source_fit_sha256=store.source_fit_sha256,
        expected_store_sha256=store.sha256,
        expected_statistics_sha256=statistics.sha256,
        expected_raw_score_sha256=raw_scores.sha256,
    )
    return _PreparedCase(
        examples=examples,
        store=store,
        statistics=statistics,
        raw_scores=raw_scores,
        artifact=artifact,
        embedding_provider=embedding_provider,
    )


@cache
def _ordinary_case(domain_count: int) -> _PreparedCase:
    return _build_case(_ordinary_examples(domain_count))


@lru_cache(maxsize=1)
def _stress_case() -> _PreparedCase:
    examples, provider = _stress_examples_and_provider()
    return _build_case(examples, embedding_provider=provider)


@lru_cache(maxsize=1)
def _route_case() -> _PreparedCase:
    return _build_case(_ordinary_examples(4, constant_targets=True))


def _fit_rowwise_base(
    examples: tuple[EvaluationExample, ...],
    provider: _FixtureEmbeddingProvider | None,
) -> _RowwiseBase:
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    model_ids = tuple(sorted(model.model_id for model in ordered[0].candidate_models))
    prompts = tuple(example.prompt for example in ordered)
    encoder = PromptFeatureEncoder.fit(
        prompts,
        embedding_provider=provider,
    )
    feature_rows = encoder.transform_many(prompts)
    target_columns = tuple(
        tuple(
            next(outcome.quality for outcome in example.outcomes if outcome.model_id == model_id)
            for example in ordered
        )
        for model_id in model_ids
    )
    return _RowwiseBase(
        encoder=encoder,
        model_ids=model_ids,
        solution=solve_centered_ridge(
            feature_rows,
            target_columns,
            ridge=_RIDGE,
        ),
    )


def _rowwise_raw_predictions(
    fitted: _RowwiseBase,
    prompts: tuple[str, ...],
) -> tuple[dict[str, float], ...]:
    vectors = fitted.encoder.transform_many(prompts)
    return tuple(
        {
            model_id: sum(
                value * weight
                for value, weight in zip(
                    vector,
                    fitted.solution.weights[model_index],
                    strict=True,
                )
            )
            + fitted.solution.intercepts[model_index]
            for model_index, model_id in enumerate(fitted.model_ids)
        }
        for vector in vectors
    )


def _rowwise_oof(case: _PreparedCase) -> dict[str, tuple[_CalibrationSample, ...]]:
    samples: dict[str, list[_CalibrationSample]] = {model_id: [] for model_id in _MODEL_IDS}
    for held_out_domain in sorted({example.domain for example in case.examples}):
        training = tuple(example for example in case.examples if example.domain != held_out_domain)
        held_out = tuple(
            sorted(
                (example for example in case.examples if example.domain == held_out_domain),
                key=lambda example: example.example_id,
            )
        )
        fitted = _fit_rowwise_base(training, case.embedding_provider)
        predictions = _rowwise_raw_predictions(
            fitted,
            tuple(example.prompt for example in held_out),
        )
        for example, row in zip(held_out, predictions, strict=True):
            outcomes = {outcome.model_id: outcome.quality for outcome in example.outcomes}
            for model_id in _MODEL_IDS:
                samples[model_id].append(
                    _CalibrationSample(
                        example_id=example.example_id,
                        prediction=row[model_id],
                        target=outcomes[model_id],
                    )
                )
    return {model_id: tuple(rows) for model_id, rows in samples.items()}


def _prepared_semantic_oof(case: _PreparedCase) -> dict[str, tuple[_CalibrationSample, ...]]:
    examples_by_id = {example.example_id: example for example in case.examples}
    samples: dict[str, list[_CalibrationSample]] = {model_id: [] for model_id in _MODEL_IDS}
    for source in case.artifact.lineage.calibration_sources:
        score_block = case.raw_scores.blocks[source.score_block_index]
        assert score_block.sha256 == source.raw_score_block_sha256
        example_ids = case.raw_scores.example_ids_for_block(source.score_block_index)
        assert len(example_ids) == source.row_count
        for row_index, example_id in enumerate(example_ids):
            scores = score_block.score_row(row_index)
            outcomes = {
                outcome.model_id: outcome.quality for outcome in examples_by_id[example_id].outcomes
            }
            for model_index, model_id in enumerate(_MODEL_IDS):
                samples[model_id].append(
                    _CalibrationSample(
                        example_id=example_id,
                        prediction=scores[model_index],
                        target=outcomes[model_id],
                    )
                )
    return {model_id: tuple(rows) for model_id, rows in samples.items()}


def _independent_pav(samples: tuple[_CalibrationSample, ...]) -> tuple[_PavBlock, ...]:
    """Fit exact-rational equal-weight PAV without calling the project calibrator."""

    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (sample.prediction, sample.target, sample.example_id),
        )
    )
    blocks: list[_MutablePavBlock] = []
    cursor = 0
    while cursor < len(ordered):
        prediction = ordered[cursor].prediction
        group = []
        while cursor < len(ordered) and ordered[cursor].prediction == prediction:
            group.append(ordered[cursor])
            cursor += 1
        block = _MutablePavBlock(
            upper_bound=prediction,
            target_sum=sum(
                (Fraction.from_float(sample.target) for sample in group),
                start=Fraction(0),
            ),
            count=len(group),
            example_ids=tuple(sorted(sample.example_id for sample in group)),
        )
        blocks.append(block)
        while len(blocks) >= 2 and blocks[-2].exact_mean > blocks[-1].exact_mean:
            left, right = blocks[-2:]
            blocks[-2:] = [
                _MutablePavBlock(
                    upper_bound=right.upper_bound,
                    target_sum=left.target_sum + right.target_sum,
                    count=left.count + right.count,
                    example_ids=tuple(sorted((*left.example_ids, *right.example_ids))),
                )
            ]
    return tuple(
        _PavBlock(
            upper_bound=block.upper_bound,
            value=float(block.exact_mean),
            example_ids=block.example_ids,
        )
        for block in blocks
    )


def _independent_step(
    upper_bounds: tuple[float, ...],
    values: tuple[float, ...],
    value: float,
) -> float:
    for upper_bound, calibrated in zip(upper_bounds, values, strict=True):
        if value <= upper_bound:
            return calibrated
    return values[-1]


def _assert_every_bound_has_exact_step_semantics(
    calibrator: IsotonicCalibrator,
) -> None:
    for bound in calibrator.upper_bounds:
        for probe in (
            math.nextafter(bound, -math.inf),
            bound,
            math.nextafter(bound, math.inf),
        ):
            assert calibrator.calibrate(probe) == _independent_step(
                calibrator.upper_bounds,
                calibrator.values,
                probe,
            )


def _assert_complete_parity(
    case: _PreparedCase,
    *,
    weight_tolerance: dict[str, float],
    bias_tolerance: dict[str, float],
    raw_tolerance: dict[str, float],
) -> None:
    reference = fit_calibrated_bilinear(
        case.examples,
        embedding_provider=case.embedding_provider,
    )
    prepared = case.artifact
    assert prepared.training_domains == reference.training_domains
    assert prepared.training_example_count == reference.training_example_count
    assert prepared.model_ids == reference.model_ids
    assert prepared.feature_schema.domain_tags == reference.feature_schema.domain_tags
    assert prepared.feature_schema.feature_names == reference.feature_schema.feature_names
    assert prepared.feature_schema.continuous_means == pytest.approx(
        reference.feature_schema.continuous_means,
        **_ORDINARY_SCHEMA_TOLERANCE,
    )
    assert prepared.feature_schema.continuous_scales == pytest.approx(
        reference.feature_schema.continuous_scales,
        **_ORDINARY_SCHEMA_TOLERANCE,
    )
    for model_id in prepared.model_ids:
        state = prepared.models[model_id]
        assert state.weights == pytest.approx(
            reference.model_weights[model_id],
            **weight_tolerance,
        )
        assert state.bias == pytest.approx(
            reference.model_bias[model_id],
            **bias_tolerance,
        )

    expected_oof = _rowwise_oof(case)
    actual_oof = _prepared_semantic_oof(case)
    for model_id in prepared.model_ids:
        expected_rows = expected_oof[model_id]
        actual_rows = actual_oof[model_id]
        assert tuple(row.example_id for row in actual_rows) == tuple(
            row.example_id for row in expected_rows
        )
        assert tuple(row.target for row in actual_rows) == tuple(
            row.target for row in expected_rows
        )
        assert tuple(row.prediction for row in actual_rows) == pytest.approx(
            tuple(row.prediction for row in expected_rows),
            **raw_tolerance,
        )
        expected_pav = _independent_pav(expected_rows)
        actual_pav = _independent_pav(actual_rows)
        assert tuple(block.example_ids for block in actual_pav) == tuple(
            block.example_ids for block in expected_pav
        )
        calibrator = prepared.models[model_id].calibration.calibrator
        assert calibrator.upper_bounds == pytest.approx(
            tuple(block.upper_bound for block in expected_pav),
            **raw_tolerance,
        )
        assert calibrator.values == pytest.approx(
            tuple(block.value for block in expected_pav),
            **_STRICT_CALIBRATION_TOLERANCE,
        )
        assert calibrator.upper_bounds == pytest.approx(
            reference.calibrators[model_id].upper_bounds,
            **raw_tolerance,
        )
        assert calibrator.values == pytest.approx(
            reference.calibrators[model_id].values,
            **_STRICT_CALIBRATION_TOLERANCE,
        )
        _assert_every_bound_has_exact_step_semantics(calibrator)

    canonical_prompts = tuple(
        example.prompt for example in sorted(case.examples, key=lambda example: example.example_id)
    )
    probes = canonical_prompts + _PROBE_PROMPTS
    prepared_predictor = prepared.build_predictor(
        embedding_provider=case.embedding_provider,
    )
    reference_predictor = reference.build_predictor(
        embedding_provider=case.embedding_provider,
    )
    prepared_raw = prepared_predictor.base.predict_batch(probes, prepared.model_ids)
    reference_raw = reference_predictor.base.predict_batch(probes, reference.model_ids)
    prepared_calibrated = prepared_predictor.predict_batch(probes, prepared.model_ids)
    reference_calibrated = reference_predictor.predict_batch(probes, reference.model_ids)
    for prepared_row, reference_row in zip(prepared_raw, reference_raw, strict=True):
        for model_id in prepared.model_ids:
            assert prepared_row[model_id] == pytest.approx(
                reference_row[model_id],
                **raw_tolerance,
            )
    for prepared_row, reference_row in zip(
        prepared_calibrated,
        reference_calibrated,
        strict=True,
    ):
        for model_id in prepared.model_ids:
            assert prepared_row[model_id] == pytest.approx(
                reference_row[model_id],
                **_STRICT_CALIBRATION_TOLERANCE,
            )


@pytest.mark.parametrize("domain_count", (4, 5, 6, 7))
def test_surface_only_d4_to_d7_matches_authoritative_rowwise_path(
    domain_count: int,
) -> None:
    case = _ordinary_case(domain_count)

    assert case.artifact.feature_schema.domain_tags == SURFACE_DOMAIN_TAG_CATALOGUE[:domain_count]
    _assert_complete_parity(
        case,
        weight_tolerance=_ORDINARY_WEIGHT_TOLERANCE,
        bias_tolerance=_ORDINARY_BIAS_TOLERANCE,
        raw_tolerance=_ORDINARY_RAW_TOLERANCE,
    )


@pytest.mark.parametrize("domain_count", (4, 5, 6, 7))
def test_each_held_out_recognized_surface_tag_is_isolated(
    domain_count: int,
) -> None:
    case = _ordinary_case(domain_count)
    expected_tags = SURFACE_DOMAIN_TAG_CATALOGUE[:domain_count]

    assert case.artifact.feature_schema.domain_tags == expected_tags
    for held_out_index, held_out_tag in enumerate(expected_tags):
        source = case.artifact.lineage.calibration_sources[held_out_index]
        coefficient = case.raw_scores.coefficients.blocks[source.training_subset_index]
        assert source.held_out_domain == held_out_tag
        assert coefficient.feature_schema.domain_tags == tuple(
            tag for tag in expected_tags if tag != held_out_tag
        )
        assert held_out_tag not in coefficient.feature_schema.domain_tags


def test_high_dynamic_zero_variance_constant_target_and_collinearity() -> None:
    case = _stress_case()
    provider = case.embedding_provider
    assert provider is not None
    nonzero_magnitudes = tuple(
        abs(value)
        for vector in provider.vectors_by_prompt.values()
        for value in vector
        if value != 0.0
    )

    assert max(nonzero_magnitudes) / min(nonzero_magnitudes) >= 1e9
    assert all(vector[1] == 2.0 * vector[0] for vector in provider.vectors_by_prompt.values())
    assert all(vector[3] == 0.0 for vector in provider.vectors_by_prompt.values())
    assert case.artifact.feature_schema.continuous_scales == (1.0, 1.0, 1.0)
    assert case.artifact.feature_schema.domain_tags == ("general",)
    assert all(
        next(outcome.quality for outcome in example.outcomes if outcome.model_id == "cheap") == 0.5
        for example in case.examples
    )
    _assert_complete_parity(
        case,
        weight_tolerance=_STRESS_WEIGHT_TOLERANCE,
        bias_tolerance=_STRESS_BIAS_TOLERANCE,
        raw_tolerance=_STRESS_RAW_TOLERANCE,
    )
    assert case.artifact.models["cheap"].weights == pytest.approx(
        (0.0,) * case.artifact.feature_schema.dimension,
        rel=0.0,
        abs=1e-15,
    )
    assert case.artifact.models["cheap"].bias == pytest.approx(
        0.5,
        **_STRESS_BIAS_TOLERANCE,
    )
    assert case.artifact.models["cheap"].calibration.calibrator.values == pytest.approx(
        (0.5,),
        **_STRICT_CALIBRATION_TOLERANCE,
    )


def test_independent_pav_oracle_covers_exact_ties_and_adjacent_merges() -> None:
    samples = (
        _CalibrationSample("tie-high", 0.1, 0.8),
        _CalibrationSample("tie-low", 0.1, 0.4),
        _CalibrationSample("merge-left", 0.2, 0.2),
        _CalibrationSample("merge-right-high", 0.3, 0.9),
        _CalibrationSample("merge-right-low", 0.4, 0.7),
    )
    expected = _independent_pav(samples)
    calibrator = IsotonicCalibrator.fit(
        [sample.prediction for sample in samples],
        [sample.target for sample in samples],
    )

    assert tuple(block.example_ids for block in expected) == (
        ("merge-left", "tie-high", "tie-low"),
        ("merge-right-high", "merge-right-low"),
    )
    assert tuple(block.upper_bound for block in expected) == (0.2, 0.4)
    assert tuple(block.value for block in expected) == pytest.approx(
        (Fraction(7, 15), Fraction(4, 5)),
        **_STRICT_CALIBRATION_TOLERANCE,
    )
    assert calibrator.upper_bounds == tuple(block.upper_bound for block in expected)
    assert calibrator.values == pytest.approx(
        tuple(block.value for block in expected),
        **_STRICT_CALIBRATION_TOLERANCE,
    )
    _assert_every_bound_has_exact_step_semantics(calibrator)


def _independent_route_oracle(
    predictor: object,
    prompt: str,
    candidates: tuple[ModelSpec, ...],
    budget: Decimal,
    lambda_cost: Fraction,
) -> tuple[str, float, dict[str, Fraction]]:
    affordable = tuple(model for model in candidates if model.cost <= budget)
    model_ids = tuple(model.model_id for model in affordable)
    predictions = predictor.predict_many(prompt, model_ids)  # type: ignore[attr-defined]
    utilities = {
        model.model_id: Fraction.from_float(float(predictions[model.model_id]))
        - lambda_cost * Fraction(model.cost)
        for model in affordable
    }
    selected = min(
        affordable,
        key=lambda model: (-utilities[model.model_id], model.cost, model.model_id),
    )
    return selected.model_id, float(predictions[selected.model_id]), utilities


@pytest.mark.parametrize(
    ("case_name", "lambda_cost", "budget", "expected_model", "expect_exact_tie"),
    (
        ("unambiguous", Fraction(1, 10), Decimal("3.000"), "premium", False),
        ("budget-exclusion", Fraction(0), Decimal("2.374"), "cheap", False),
        ("exact-tie", Fraction(2, 5), Decimal("2.375"), "cheap", True),
    ),
)
def test_artifact_predictor_one_shot_route_and_per_query_accounting(
    case_name: str,
    lambda_cost: Fraction,
    budget: Decimal,
    expected_model: str,
    expect_exact_tie: bool,
) -> None:
    del case_name
    case = _route_case()
    predictor = case.artifact.build_predictor()
    prompt = "Give a short neutral explanation."
    candidates = (
        ModelSpec("premium", Decimal("2.375")),
        ModelSpec("cheap", Decimal("1.125")),
    )
    selected_model, predicted_quality, utilities = _independent_route_oracle(
        predictor,
        prompt,
        candidates,
        budget,
        lambda_cost,
    )
    assert selected_model == expected_model
    if expect_exact_tie:
        assert utilities["cheap"] == utilities["premium"]
    else:
        assert len(set(utilities.values())) == len(utilities)

    router = LambdaThresholdRouter(predictor, lambda_cost)
    initial_state = RouterState(
        prompt=prompt,
        budget_tier=BudgetTier.FAST,
        remaining_budget=budget,
        candidate_models=candidates,
    )
    action = router.route(initial_state)
    assert type(action) is CallModel
    assert action.model_id == selected_model
    assert action.predicted_quality == predicted_quality

    selected_cost = next(model.cost for model in candidates if model.model_id == selected_model)
    completed_action = router.route(
        RouterState(
            prompt=prompt,
            budget_tier=BudgetTier.FAST,
            remaining_budget=budget - selected_cost,
            call_history=(
                CallRecord(
                    selected_model,
                    selected_cost,
                    f"{selected_model} answer",
                ),
            ),
            candidate_models=candidates,
        )
    )
    assert type(completed_action) is SelectOutput
    assert completed_action.history_index == 0

    example = EvaluationExample(
        example_id=f"route-{expected_model}-{lambda_cost}",
        prompt=prompt,
        domain="route-fixture",
        candidate_models=candidates,
        outcomes=tuple(
            CandidateOutcome(
                model_id=model.model_id,
                output=f"{model.model_id} answer",
                cost=model.cost,
                quality=0.75 if model.model_id == "premium" else 0.25,
            )
            for model in candidates
        ),
    )
    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(
        router,
        (example,),
        TierSpec(BudgetTier.FAST, budget, 1.0),
    )
    query = result.queries[0]
    expected_remaining = budget - selected_cost

    assert result.feasible
    assert query.feasible
    assert query.selected_model_id == selected_model
    assert query.selected_call_index == 0
    assert query.cost == selected_cost
    assert query.predicted_quality == predicted_quality
    assert len(query.calls) == 1
    assert query.calls[0].model_id == selected_model
    assert query.calls[0].quoted_cost == selected_cost
    assert query.calls[0].realized_cost == selected_cost
    assert query.calls[0].remaining_budget_before == budget
    assert query.calls[0].remaining_budget_after == expected_remaining
    assert query.calls[0].within_budget
    assert result.budget.spent == selected_cost
    assert result.budget.over_budget_calls == 0
    assert result.budget.configured_limit == budget
    assert result.budget.effective_total_limit == budget
