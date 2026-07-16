# SPDX-License-Identifier: Apache-2.0
"""Tests for leakage-aware GBM fitting and inner-LODO calibration."""

from __future__ import annotations

import math
import socket
import urllib.request
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

import tierroute.predictors.gbm_training as gbm_training_module
from tierroute.adapters import load_evaluation_dataset
from tierroute.eval import DomainFold, EvaluationExample, leave_one_domain_out
from tierroute.features import EmbeddingIdentity
from tierroute.predictors.calibration import (
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
)
from tierroute.predictors.gbm import GbmQualityPredictor
from tierroute.predictors.gbm_training import (
    GbmNestedLodoWorkEstimate,
    GbmTrainingConfig,
    estimate_nested_lodo_gbm_work,
    fit_calibrated_gbm,
    fit_calibrated_gbm_for_fold,
    preflight_gbm_fit,
    preflight_nested_lodo_gbm,
)


class RecordingEmbeddingProvider:
    """Deterministic local test provider with an observable batch trace."""

    dimension = 2
    identity = EmbeddingIdentity(
        provider="tierroute.tests.gbm-recording-v1",
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
            (
                float(len(text) % 17) / 17.0,
                float(sum(map(ord, text)) % 19) / 19.0,
            )
            for text in batch
        )


def _science_fold(examples: tuple[EvaluationExample, ...]) -> DomainFold:
    return next(
        fold for fold in leave_one_domain_out(examples) if fold.held_out_domain == "science"
    )


def _mutate_science(examples: tuple[EvaluationExample, ...]) -> tuple[EvaluationExample, ...]:
    return tuple(
        replace(
            example,
            prompt=f"HELD OUT SENTINEL {example.example_id}",
            outcomes=tuple(replace(outcome, quality=0.01) for outcome in example.outcomes),
        )
        if example.domain == "science"
        else example
        for example in examples
    )


def _small_config() -> GbmTrainingConfig:
    return GbmTrainingConfig(n_estimators=4, learning_rate=0.2, min_samples_leaf=1)


def test_gbm_outer_fold_training_never_observes_held_out_examples() -> None:
    examples = load_evaluation_dataset().examples
    original_fold = _science_fold(examples)
    mutated_fold = _science_fold(_mutate_science(examples))
    first_provider = RecordingEmbeddingProvider()
    second_provider = RecordingEmbeddingProvider()

    first = fit_calibrated_gbm_for_fold(
        original_fold,
        config=_small_config(),
        embedding_provider=first_provider,
    )
    second = fit_calibrated_gbm_for_fold(
        mutated_fold,
        config=_small_config(),
        embedding_provider=second_provider,
    )

    held_out_prompts = {example.prompt for example in original_fold.test}
    observed_prompts = {prompt for batch in first_provider.calls for prompt in batch}
    assert held_out_prompts.isdisjoint(observed_prompts)
    assert first == second


def test_gbm_training_is_order_independent_and_model_labels_follow_ids() -> None:
    examples = load_evaluation_dataset().examples
    reordered = tuple(
        replace(
            example,
            outcomes=tuple(reversed(example.outcomes)),
            candidate_models=tuple(reversed(example.candidate_models)),
        )
        for example in reversed(examples)
    )

    first = fit_calibrated_gbm(examples, config=_small_config())
    second = fit_calibrated_gbm(reordered, config=_small_config())

    assert first == second
    prompts = tuple(example.prompt for example in examples)
    model_ids = tuple(sorted(model.model_id for model in examples[0].candidate_models))
    assert first.predict_batch(prompts, model_ids) == second.predict_batch(prompts, model_ids)  # type: ignore[attr-defined]


def test_gbm_calibration_is_cross_fitted_separately_per_model() -> None:
    predictor = fit_calibrated_gbm(
        load_evaluation_dataset().examples,
        config=_small_config(),
    )

    assert isinstance(predictor, PerModelCalibratedQualityPredictor)
    assert isinstance(predictor.base, GbmQualityPredictor)
    assert set(predictor.calibrators) == set(predictor.base.models)
    assert predictor.calibrators["swift"] is not predictor.calibrators["expert"]
    assert max(predictor.calibrators["swift"].values) < min(predictor.calibrators["expert"].values)


def test_gbm_isotonic_fit_receives_one_oof_prediction_per_training_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = load_evaluation_dataset().examples
    original_fit = gbm_training_module.IsotonicCalibrator.fit
    observed_lengths: list[tuple[int, int]] = []

    def recording_fit(
        cls: type[IsotonicCalibrator],
        predictions: list[float],
        targets: list[float],
    ) -> IsotonicCalibrator:
        del cls
        observed_lengths.append((len(predictions), len(targets)))
        return original_fit(predictions, targets)

    monkeypatch.setattr(
        gbm_training_module.IsotonicCalibrator,
        "fit",
        classmethod(recording_fit),
    )

    fit_calibrated_gbm(examples, config=_small_config())

    model_count = len(examples[0].candidate_models)
    assert observed_lengths == [(len(examples), len(examples))] * model_count


def test_gbm_predictor_batches_embeddings_across_prompts_and_models() -> None:
    provider = RecordingEmbeddingProvider()
    predictor = fit_calibrated_gbm(
        load_evaluation_dataset().examples,
        config=_small_config(),
        embedding_provider=provider,
    )
    provider.calls.clear()
    prompts = ("Debug Python code", "Prove a math theorem", "General question")
    model_ids = ("expert", "swift")

    rows = predictor.predict_batch(prompts, model_ids)  # type: ignore[attr-defined]

    assert provider.calls == [prompts]
    assert len(rows) == len(prompts)
    assert all(set(row) == set(model_ids) for row in rows)
    assert all(math.isfinite(value) for row in rows for value in row.values())
    with pytest.raises(TypeError):
        predictor.calibrators["swift"] = predictor.calibrators["swift"]  # type: ignore[index]


def test_gbm_preflight_checks_exact_scan_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GbmTrainingConfig(
        n_estimators=2,
        learning_rate=0.1,
        min_samples_leaf=1,
    )
    monkeypatch.setattr(gbm_training_module, "MAX_GBM_SPLIT_SCANS", 4)
    preflight_gbm_fit(
        sample_count=3,
        feature_count=1,
        target_count=1,
        config=config,
    )
    with pytest.raises(ValueError, match="split scan"):
        preflight_gbm_fit(
            sample_count=4,
            feature_count=1,
            target_count=1,
            config=config,
        )


def test_gbm_preflight_rejects_before_embedding_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailIfEmbeddedProvider(RecordingEmbeddingProvider):
        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            raise AssertionError(f"embedding allocation must not run: {tuple(texts)!r}")

    monkeypatch.setattr(gbm_training_module, "MAX_GBM_TRAINING_CELLS", 1)

    with pytest.raises(ValueError, match="feature matrix"):
        fit_calibrated_gbm(
            load_evaluation_dataset().examples,
            config=_small_config(),
            embedding_provider=FailIfEmbeddedProvider(),
        )


def test_gbm_model_catalogue_limit_is_checked_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingEmbeddingProvider()
    monkeypatch.setattr(gbm_training_module, "MAX_PREDICTOR_MODELS", 2)

    with pytest.raises(ValueError, match="target catalogue"):
        fit_calibrated_gbm(
            load_evaluation_dataset().examples,
            config=_small_config(),
            embedding_provider=provider,
        )

    assert provider.calls == []


def test_stateful_numeric_config_is_rejected_before_embedding() -> None:
    class StatefulFloat(float):
        values = iter((0.1, 0.1, math.inf))

        def __float__(self) -> float:
            return next(type(self).values)

    provider = RecordingEmbeddingProvider()

    with pytest.raises(ValueError, match="built-in real number"):
        config = GbmTrainingConfig(learning_rate=StatefulFloat(0.1))
        fit_calibrated_gbm(
            load_evaluation_dataset().examples,
            config=config,
            embedding_provider=provider,
        )

    assert provider.calls == []


def test_gbm_calibrated_work_is_aggregated_before_any_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingEmbeddingProvider()
    monkeypatch.setattr(gbm_training_module, "MAX_GBM_SPLIT_SCANS", 200)

    with pytest.raises(ValueError, match=r"calibrated.*split scan"):
        fit_calibrated_gbm(
            load_evaluation_dataset().examples,
            config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
            embedding_provider=provider,
        )

    assert provider.calls == []


def test_nested_lodo_gbm_estimate_enumerates_the_complete_call_graph() -> None:
    examples = load_evaluation_dataset().examples
    config = GbmTrainingConfig(n_estimators=1, min_samples_leaf=1)

    estimate = estimate_nested_lodo_gbm_work(examples, config=config)

    domain_count = len({example.domain for example in examples})
    assert isinstance(estimate, GbmNestedLodoWorkEstimate)
    assert estimate.domain_count == domain_count == 4
    assert estimate.outer_fold_count == domain_count
    assert estimate.calibrated_fit_count == domain_count**2
    assert estimate.base_fit_count == domain_count * ((domain_count - 1) ** 2 + domain_count)
    assert estimate.model_count == len(examples[0].candidate_models)
    assert estimate.estimator_count == config.n_estimators
    assert estimate.split_scans == sum(estimate.base_fit_split_scans) == 2_241
    assert len(estimate.base_fit_sample_counts) == estimate.base_fit_count
    assert len(estimate.base_fit_feature_counts) == estimate.base_fit_count
    with pytest.raises(FrozenInstanceError):
        estimate.split_scans = 0  # type: ignore[misc]


def test_nested_lodo_gbm_estimate_fits_each_schema_without_embedding() -> None:
    examples = load_evaluation_dataset().examples
    provider = RecordingEmbeddingProvider()

    surface = estimate_nested_lodo_gbm_work(
        examples,
        config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
    )
    embedded = estimate_nested_lodo_gbm_work(
        examples,
        config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
        embedding_provider=provider,
    )

    assert len(set(surface.base_fit_feature_counts)) > 1
    assert embedded.base_fit_feature_counts == tuple(
        feature_count + provider.dimension for feature_count in surface.base_fit_feature_counts
    )
    assert embedded.split_scans > surface.split_scans
    assert provider.calls == []


def test_nested_lodo_gbm_preflight_rejects_aggregate_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = RecordingEmbeddingProvider()
    examples = load_evaluation_dataset().examples
    config = GbmTrainingConfig(n_estimators=1, min_samples_leaf=1)
    estimate = estimate_nested_lodo_gbm_work(
        examples,
        config=config,
        embedding_provider=provider,
    )
    monkeypatch.setattr(gbm_training_module, "MAX_GBM_SPLIT_SCANS", estimate.split_scans)

    assert (
        preflight_nested_lodo_gbm(
            examples,
            config=config,
            embedding_provider=provider,
        )
        == estimate
    )
    monkeypatch.setattr(gbm_training_module, "MAX_GBM_SPLIT_SCANS", estimate.split_scans - 1)

    with pytest.raises(ValueError, match=r"nested-LODO.*split scan"):
        preflight_nested_lodo_gbm(
            examples,
            config=config,
            embedding_provider=provider,
        )

    assert provider.calls == []


def test_nested_lodo_gbm_preflight_requires_four_domains_and_stable_catalogue() -> None:
    examples = load_evaluation_dataset().examples
    three_domains = tuple(example for example in examples if example.domain != "science")
    provider = RecordingEmbeddingProvider()

    with pytest.raises(ValueError, match="at least four domains"):
        preflight_nested_lodo_gbm(three_domains, embedding_provider=provider)

    changed = replace(
        examples[0],
        outcomes=examples[0].outcomes[:-1],
        candidate_models=examples[0].candidate_models[:-1],
    )
    with pytest.raises(ValueError, match="same model catalogue"):
        preflight_nested_lodo_gbm((changed, *examples[1:]), embedding_provider=provider)

    assert provider.calls == []


def test_nested_lodo_gbm_preflight_rejects_config_subclasses_before_embedding() -> None:
    @dataclass(frozen=True, slots=True)
    class GbmTrainingConfigSubclass(GbmTrainingConfig):
        pass

    provider = RecordingEmbeddingProvider()

    with pytest.raises(TypeError, match="exact GbmTrainingConfig"):
        preflight_nested_lodo_gbm(
            load_evaluation_dataset().examples,
            config=GbmTrainingConfigSubclass(),
            embedding_provider=provider,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"n_estimators": True}, "n_estimators"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": math.nan}, "learning_rate"),
        ({"learning_rate": 10**10_000}, "learning_rate"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"min_samples_leaf": True}, "min_samples_leaf"),
        ({"min_gain": -1}, "min_gain"),
        ({"min_gain": math.inf}, "min_gain"),
    ],
)
def test_gbm_training_config_rejects_unsafe_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GbmTrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_gbm_training_requires_multiple_domains_and_valid_fold_type() -> None:
    one_domain = tuple(
        example for example in load_evaluation_dataset().examples if example.domain == "science"
    )

    with pytest.raises(ValueError, match="at least two domains"):
        fit_calibrated_gbm(one_domain, config=_small_config())
    with pytest.raises(TypeError, match="DomainFold"):
        fit_calibrated_gbm_for_fold(object())  # type: ignore[arg-type]


def test_gbm_training_and_inference_remain_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("GBM training must not open a network connection")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(urllib.request, "urlopen", deny_network)

    predictor = fit_calibrated_gbm(
        load_evaluation_dataset().examples,
        config=_small_config(),
    )

    assert math.isfinite(predictor.predict("Explain a Python function", "swift"))
