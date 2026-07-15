# SPDX-License-Identifier: Apache-2.0
"""Tests for leakage-aware bilinear fitting and inner-LODO calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

import tierroute.predictors.training as training_module
from tierroute.adapters import load_evaluation_dataset
from tierroute.eval import DomainFold, EvaluationExample, leave_one_domain_out
from tierroute.features import EmbeddingIdentity
from tierroute.predictors import (
    BilinearTrainingConfig,
    fit_calibrated_bilinear,
    fit_calibrated_bilinear_for_fold,
    training_data_sha256,
)
from tierroute.predictors._ridge import (
    CENTERED_RIDGE_SOLVER_ID,
    RidgeSolution,
    fit_centered_ridge,
)
from tierroute.predictors.solvers import resolve_ridge_solver


class RecordingEmbeddingProvider:
    """Small deterministic embedding provider with an observable call trace."""

    dimension = 2
    identity = EmbeddingIdentity(
        provider="tierroute.tests.recording-v1",
        model_id="project-authored-test-embedding",
        revision="1",
        pooling="test-pool",
        normalize=False,
        asset_manifest_sha256="0" * 64,
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        batch = tuple(texts)
        self.calls.append(batch)
        return tuple(
            (float(len(text) % 17) / 17.0, float(sum(map(ord, text)) % 19) / 19.0) for text in batch
        )


def _science_fold(examples: tuple[EvaluationExample, ...]) -> DomainFold:
    return next(
        fold for fold in leave_one_domain_out(examples) if fold.held_out_domain == "science"
    )


def _mutate_science(examples: tuple[EvaluationExample, ...]) -> tuple[EvaluationExample, ...]:
    mutated = []
    for example in examples:
        if example.domain != "science":
            mutated.append(example)
            continue
        mutated.append(
            replace(
                example,
                prompt=f"HELD OUT SENTINEL {example.example_id}",
                outcomes=tuple(replace(outcome, quality=0.01) for outcome in example.outcomes),
            )
        )
    return tuple(mutated)


def test_outer_fold_training_never_observes_held_out_examples() -> None:
    examples = load_evaluation_dataset().examples
    original_fold = _science_fold(examples)
    mutated_fold = _science_fold(_mutate_science(examples))
    first_provider = RecordingEmbeddingProvider()
    second_provider = RecordingEmbeddingProvider()

    first = fit_calibrated_bilinear_for_fold(
        original_fold,
        embedding_provider=first_provider,
    )
    second = fit_calibrated_bilinear_for_fold(
        mutated_fold,
        embedding_provider=second_provider,
    )

    held_out_prompts = {example.prompt for example in original_fold.test}
    observed_prompts = {prompt for batch in first_provider.calls for prompt in batch}
    assert held_out_prompts.isdisjoint(observed_prompts)
    assert first.training_domains == ("code", "general", "math")
    assert first.training_example_count == 6
    assert first.to_json() == second.to_json()


def test_training_is_order_independent_and_model_labels_follow_ids() -> None:
    examples = load_evaluation_dataset().examples
    reordered = tuple(
        replace(
            example,
            outcomes=tuple(reversed(example.outcomes)),
            candidate_models=tuple(reversed(example.candidate_models)),
        )
        for example in reversed(examples)
    )

    first = fit_calibrated_bilinear(examples)
    second = fit_calibrated_bilinear(reordered)

    assert first.to_json() == second.to_json()
    assert first.training_data_sha256 == training_data_sha256(examples)
    assert len(first.training_data_sha256) == 64


def test_calibration_is_cross_fitted_separately_per_model() -> None:
    artifact = fit_calibrated_bilinear(load_evaluation_dataset().examples)

    assert set(artifact.calibrators) == set(artifact.model_weights)
    assert artifact.calibrators["swift"] is not artifact.calibrators["expert"]
    assert max(artifact.calibrators["swift"].values) < min(artifact.calibrators["expert"].values)
    assert all(
        math.isfinite(value) for weights in artifact.model_weights.values() for value in weights
    )


def test_artifact_predictor_batches_embeddings_across_models() -> None:
    training_provider = RecordingEmbeddingProvider()
    artifact = fit_calibrated_bilinear(
        load_evaluation_dataset().examples,
        embedding_provider=training_provider,
    )
    inference_provider = RecordingEmbeddingProvider()
    predictor = artifact.build_predictor(embedding_provider=inference_provider)
    prompts = ("Debug Python code", "Prove a math theorem", "General question")

    rows = predictor.predict_batch(prompts, artifact.model_ids)

    assert inference_provider.calls == [prompts]
    assert len(rows) == len(prompts)
    assert all(set(row) == set(artifact.model_ids) for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row.values())

    class WrongProvider(RecordingEmbeddingProvider):
        identity = replace(RecordingEmbeddingProvider.identity, revision="different")

    with pytest.raises(ValueError, match="identity"):
        artifact.build_predictor(embedding_provider=WrongProvider())


@pytest.mark.parametrize(
    "config",
    [
        BilinearTrainingConfig(ridge=1.0, seed=0),
        BilinearTrainingConfig(ridge=0.25, seed=2026),
    ],
)
def test_training_is_deterministic_for_fixed_config(config: BilinearTrainingConfig) -> None:
    examples = load_evaluation_dataset().examples

    assert fit_calibrated_bilinear(examples, config=config).to_json() == (
        fit_calibrated_bilinear(examples, config=config).to_json()
    )


def test_one_resolved_solver_reaches_every_inner_fold_and_final_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = resolve_ridge_solver(CENTERED_RIDGE_SOLVER_ID)
    preflights: list[tuple[int, int, int]] = []
    solves: list[tuple[int, int, int]] = []
    resolutions: list[str] = []

    class RecordingSolver:
        solver_id = CENTERED_RIDGE_SOLVER_ID

        def preflight(
            self,
            *,
            sample_count: int,
            feature_count: int,
            target_count: int,
        ) -> None:
            preflights.append((sample_count, feature_count, target_count))
            reference.preflight(
                sample_count=sample_count,
                feature_count=feature_count,
                target_count=target_count,
            )

        def solve(
            self,
            feature_rows: Sequence[Sequence[float]],
            target_columns: Sequence[Sequence[float]],
            *,
            ridge: float,
        ) -> RidgeSolution:
            solves.append((len(feature_rows), len(feature_rows[0]), len(target_columns)))
            return reference.solve(feature_rows, target_columns, ridge=ridge)

    recording = RecordingSolver()

    def recording_resolver(solver_id: str) -> RecordingSolver:
        resolutions.append(solver_id)
        return recording

    monkeypatch.setattr(training_module, "resolve_ridge_solver", recording_resolver)

    artifact = fit_calibrated_bilinear(load_evaluation_dataset().examples)

    assert resolutions == [CENTERED_RIDGE_SOLVER_ID]
    assert len(preflights) == len(solves) == 5
    assert preflights == solves
    assert preflights[-1][0] == artifact.training_example_count
    assert artifact.solver_id == CENTERED_RIDGE_SOLVER_ID


def test_solver_boundary_preserves_reference_artifact_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = load_evaluation_dataset().examples
    through_boundary = fit_calibrated_bilinear(examples)

    def legacy_fit(
        solver: object,
        feature_rows: Sequence[Sequence[float]],
        targets_by_model: Mapping[str, Sequence[float]],
        *,
        ridge: float,
    ) -> dict[str, tuple[tuple[float, ...], float]]:
        del solver
        return fit_centered_ridge(feature_rows, targets_by_model, ridge=ridge)

    monkeypatch.setattr(training_module, "fit_targets_with_solver", legacy_fit)
    through_legacy_path = fit_calibrated_bilinear(examples)

    # Exact equality is intentionally platform-local. The reference solver does
    # not claim cross-platform byte-identical floating-point coefficients.
    assert through_boundary.to_json() == through_legacy_path.to_json()


def test_reference_preflight_rejects_full_embedding_before_provider_call() -> None:
    class FullDimensionProvider(RecordingEmbeddingProvider):
        dimension = 1_024

        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            raise AssertionError(f"embedding allocation must not run: {tuple(texts)!r}")

    with pytest.raises(ValueError, match="reviewed accelerated backend"):
        fit_calibrated_bilinear(
            load_evaluation_dataset().examples,
            embedding_provider=FullDimensionProvider(),
        )


def test_training_requires_multiple_domains_and_safe_regularization() -> None:
    one_domain = tuple(
        example for example in load_evaluation_dataset().examples if example.domain == "science"
    )

    with pytest.raises(ValueError, match="at least two domains"):
        fit_calibrated_bilinear(one_domain)
    with pytest.raises(ValueError, match="ridge"):
        BilinearTrainingConfig(ridge=0)
    with pytest.raises(ValueError, match="ridge"):
        BilinearTrainingConfig(ridge=math.nan)
    with pytest.raises(ValueError, match="ridge"):
        BilinearTrainingConfig(ridge=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed"):
        BilinearTrainingConfig(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown or unreviewed"):
        BilinearTrainingConfig(solver_id="third-party.unreviewed-v1")
    with pytest.raises(TypeError, match="solver_id"):
        BilinearTrainingConfig(solver_id=1)  # type: ignore[arg-type]


def test_outer_fold_helper_rejects_malformed_boundaries() -> None:
    examples = load_evaluation_dataset().examples
    held_out = examples[0].domain
    wrong_test = next(example for example in examples if example.domain != held_out)

    with pytest.raises(ValueError, match="held-out domain"):
        DomainFold(
            held_out_domain=held_out,
            training=examples,
            test=(examples[0],),
        )
    with pytest.raises(ValueError, match="test example"):
        DomainFold(
            held_out_domain=held_out,
            training=tuple(example for example in examples if example.domain != held_out),
            test=(wrong_test,),
        )
