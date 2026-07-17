# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical, bounded calibrated-GBM artifacts."""

from __future__ import annotations

import copy
import json
import math
import socket
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import tierroute.predictors.artifacts as shared_artifacts
import tierroute.predictors.gbm_artifacts as gbm_artifacts
from tierroute.adapters import load_evaluation_dataset
from tierroute.eval import DomainFold, EvaluationExample, leave_one_domain_out
from tierroute.features import EmbeddingIdentity
from tierroute.predictors import (
    GBM_ALGORITHM_ID,
    GBM_PREDICTOR_ARTIFACT_KIND,
    GBM_PREDICTOR_ARTIFACT_VERSION,
    GbmModel,
    GbmPredictorArtifact,
    GbmTrainingConfig,
    RegressionStump,
    fit_calibrated_gbm,
    fit_calibrated_gbm_artifact,
    fit_calibrated_gbm_artifact_for_fold,
)


def _config() -> GbmTrainingConfig:
    return GbmTrainingConfig(
        n_estimators=4,
        learning_rate=0.2,
        min_samples_leaf=1,
    )


@pytest.fixture(scope="module")
def artifact() -> GbmPredictorArtifact:
    return fit_calibrated_gbm_artifact(
        load_evaluation_dataset().examples,
        config=_config(),
    )


class LocalEmbeddingProvider:
    """Small deterministic provider used only by artifact identity tests."""

    dimension = 2
    identity = EmbeddingIdentity(
        provider="tierroute.tests.gbm-artifact-v1",
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


def test_gbm_artifact_is_canonical_and_round_trips(
    artifact: GbmPredictorArtifact,
    tmp_path: Path,
) -> None:
    document = artifact.to_json()
    path = artifact.save(tmp_path / "gbm.json")

    loaded = GbmPredictorArtifact.load(path)

    assert document.endswith("\n")
    assert loaded == artifact
    assert loaded.to_json() == document
    assert loaded.artifact_kind == GBM_PREDICTOR_ARTIFACT_KIND
    assert loaded.artifact_version == GBM_PREDICTOR_ARTIFACT_VERSION
    assert loaded.algorithm_id == GBM_ALGORITHM_ID
    assert (
        json.dumps(
            artifact.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        == document
    )
    prompt = "Prove a short math theorem."
    assert loaded.build_predictor().predict_many(prompt, loaded.model_ids) == (
        artifact.build_predictor().predict_many(prompt, artifact.model_ids)
    )


def test_gbm_artifact_fit_is_order_independent_and_matches_memory_predictor(
    artifact: GbmPredictorArtifact,
) -> None:
    examples = load_evaluation_dataset().examples
    reordered = tuple(
        replace(
            example,
            outcomes=tuple(reversed(example.outcomes)),
            candidate_models=tuple(reversed(example.candidate_models)),
        )
        for example in reversed(examples)
    )
    reordered_artifact = fit_calibrated_gbm_artifact(reordered, config=_config())
    in_memory = fit_calibrated_gbm(examples, config=_config())
    prompts = tuple(example.prompt for example in examples)

    assert reordered_artifact.to_json() == artifact.to_json()
    assert artifact.build_predictor().predict_batch(prompts, artifact.model_ids) == (
        in_memory.predict_batch(prompts, artifact.model_ids)
    )


def test_gbm_artifact_training_hash_covers_fit_inputs(
    artifact: GbmPredictorArtifact,
) -> None:
    examples = load_evaluation_dataset().examples
    changed = (replace(examples[0], prompt=f"{examples[0].prompt} changed"), *examples[1:])

    changed_artifact = fit_calibrated_gbm_artifact(changed, config=_config())

    assert changed_artifact.training_data_sha256 != artifact.training_data_sha256


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        "[]",
        '{"artifact_version":1,"artifact_version":1}',
        '{"artifact_version":NaN}',
        '{"artifact_version":Infinity}',
    ],
)
def test_gbm_artifact_rejects_non_strict_json(document: str) -> None:
    with pytest.raises(ValueError):
        GbmPredictorArtifact.from_json(document)


def test_gbm_artifact_rejects_pickle_fallback(tmp_path: Path) -> None:
    binary = tmp_path / "artifact.pkl"
    binary.write_bytes(b"\x80\x04N.")
    ascii_pickle = tmp_path / "ascii.pkl"
    ascii_pickle.write_bytes(b"N.")

    with pytest.raises(ValueError, match="cannot read GBM predictor artifact"):
        GbmPredictorArtifact.load(binary)
    with pytest.raises(ValueError, match="not valid strict JSON"):
        GbmPredictorArtifact.load(ascii_pickle)


def test_gbm_artifact_rejects_field_identity_and_numeric_mutations(
    artifact: GbmPredictorArtifact,
) -> None:
    valid = artifact.to_dict()
    model_id = artifact.model_ids[0]
    mutations: list[dict[str, object]] = []

    missing = copy.deepcopy(valid)
    missing.pop("training")
    mutations.append(missing)

    extra = copy.deepcopy(valid)
    extra["unexpected"] = True
    mutations.append(extra)

    bad_kind = copy.deepcopy(valid)
    bad_kind["artifact_kind"] = "tierroute-bilinear-predictor"
    mutations.append(bad_kind)

    bad_algorithm = copy.deepcopy(valid)
    bad_algorithm["algorithm_id"] = "unreviewed-gbm"
    mutations.append(bad_algorithm)

    boolean_version = copy.deepcopy(valid)
    boolean_version["artifact_version"] = True
    mutations.append(boolean_version)

    boolean_estimators = copy.deepcopy(valid)
    boolean_estimators["training"]["config"]["n_estimators"] = True  # type: ignore[index]
    mutations.append(boolean_estimators)

    string_rate = copy.deepcopy(valid)
    string_rate["training"]["config"]["learning_rate"] = "0.2"  # type: ignore[index]
    mutations.append(string_rate)

    non_finite_base = copy.deepcopy(valid)
    non_finite_base["models"][model_id]["base_value"] = math.inf  # type: ignore[index]
    mutations.append(non_finite_base)

    malformed_stump = copy.deepcopy(valid)
    malformed_stump["models"][model_id]["stumps"][0] = [0, 1.0, 2.0]  # type: ignore[index]
    mutations.append(malformed_stump)

    bad_feature_index = copy.deepcopy(valid)
    bad_feature_index["models"][model_id]["stumps"][0][0] = (  # type: ignore[index]
        artifact.feature_schema.dimension
    )
    mutations.append(bad_feature_index)

    too_few_estimators = copy.deepcopy(valid)
    too_few_estimators["training"]["config"]["n_estimators"] = 1  # type: ignore[index]
    mutations.append(too_few_estimators)

    missing_calibrator = copy.deepcopy(valid)
    missing_calibrator["calibrators"].pop(model_id)  # type: ignore[union-attr]
    mutations.append(missing_calibrator)

    bad_hash = copy.deepcopy(valid)
    bad_hash["training"]["data_sha256"] = "ABC"  # type: ignore[index]
    mutations.append(bad_hash)

    for payload in mutations:
        with pytest.raises((TypeError, ValueError)):
            GbmPredictorArtifact.from_dict(payload)


def test_gbm_artifact_rejects_direct_schema_rate_and_calibrator_mismatch(
    artifact: GbmPredictorArtifact,
) -> None:
    changed_schema = replace(
        artifact.feature_schema,
        domain_tags=(*artifact.feature_schema.domain_tags, "zz-extra"),
    )
    with pytest.raises(ValueError, match="feature width"):
        replace(artifact, feature_schema=changed_schema)

    model_id = artifact.model_ids[0]
    model = artifact.models[model_id]
    changed_models = dict(artifact.models)
    changed_models[model_id] = GbmModel(
        feature_width=model.feature_width,
        base_value=model.base_value,
        learning_rate=0.3,
        stumps=model.stumps,
    )
    with pytest.raises(ValueError, match="learning rate"):
        replace(artifact, models=changed_models)

    with pytest.raises(ValueError, match="training_example_count"):
        replace(artifact, training_example_count=1)


def test_gbm_artifact_rejects_nested_stump_subclasses(
    artifact: GbmPredictorArtifact,
) -> None:
    @dataclass(frozen=True, slots=True)
    class SpoofedStump(RegressionStump):
        pass

    model_id = artifact.model_ids[0]
    original = artifact.models[model_id]
    stump = original.stumps[0]
    spoofed = SpoofedStump(
        stump.feature_index,
        stump.split_value,
        stump.left_value,
        stump.right_value,
    )
    changed = dict(artifact.models)
    changed[model_id] = GbmModel(
        feature_width=original.feature_width,
        base_value=original.base_value,
        learning_rate=original.learning_rate,
        stumps=(spoofed, *original.stumps[1:]),
    )

    with pytest.raises(TypeError, match="exact RegressionStump"):
        replace(artifact, models=changed)


def test_gbm_artifact_snapshots_parameter_mappings(
    artifact: GbmPredictorArtifact,
) -> None:
    model_id = artifact.model_ids[0]
    document = artifact.to_json()

    with pytest.raises(TypeError):
        artifact.models[model_id] = artifact.models[model_id]  # type: ignore[index]
    with pytest.raises(AttributeError):
        artifact.calibrators.pop(model_id)  # type: ignore[attr-defined]

    assert artifact.to_json() == document
    assert all(
        type(stump) is RegressionStump
        for model in artifact.models.values()
        for stump in model.stumps
    )


def test_gbm_artifact_embedding_identity_is_enforced() -> None:
    provider = LocalEmbeddingProvider()
    artifact = fit_calibrated_gbm_artifact(
        load_evaluation_dataset().examples,
        config=_config(),
        embedding_provider=provider,
    )
    provider.calls.clear()

    predictor = artifact.build_predictor(embedding_provider=provider)
    assert math.isfinite(predictor.predict("offline embedding prompt", artifact.model_ids[0]))
    assert provider.calls == [("offline embedding prompt",)]

    with pytest.raises(ValueError, match="required"):
        artifact.build_predictor()

    class WrongProvider(LocalEmbeddingProvider):
        identity = replace(LocalEmbeddingProvider.identity, revision="wrong")

    with pytest.raises(ValueError, match="identity"):
        artifact.build_predictor(embedding_provider=WrongProvider())

    surface = fit_calibrated_gbm_artifact(
        load_evaluation_dataset().examples,
        config=_config(),
    )
    with pytest.raises(ValueError, match="not allowed"):
        surface.build_predictor(embedding_provider=provider)


def test_gbm_artifact_fit_load_and_predict_remain_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def deny_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("GBM artifact path must not use a network connection")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(urllib.request, "urlopen", deny_network)

    artifact = fit_calibrated_gbm_artifact(
        load_evaluation_dataset().examples,
        config=_config(),
    )
    loaded = GbmPredictorArtifact.load(artifact.save(tmp_path / "offline-gbm.json"))

    assert math.isfinite(loaded.build_predictor().predict("Explain Python", "swift"))


def test_gbm_artifact_document_limit_is_atomic(
    artifact: GbmPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document = artifact.to_json()
    byte_count = len(document.encode("utf-8"))
    source = tmp_path / "source.json"
    source.write_text(document, encoding="utf-8")

    monkeypatch.setattr(shared_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", byte_count)
    assert GbmPredictorArtifact.load(source).to_json() == document

    monkeypatch.setattr(shared_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", byte_count - 1)
    destination = tmp_path / "destination.json"
    destination.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="predictor artifact exceeds"):
        artifact.save(destination)
    assert destination.read_text(encoding="utf-8") == "preserve"


def test_gbm_artifact_number_token_limit_covers_unknown_fields(
    artifact: GbmPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = artifact.to_json()
    monkeypatch.setattr(shared_artifacts, "MAX_PREDICTOR_JSON_NUMBER_TOKENS", 1)

    with pytest.raises(ValueError, match="not valid strict JSON") as raised:
        GbmPredictorArtifact.from_json(document)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "number-token limit" in str(raised.value.__cause__)


def test_gbm_artifact_numeric_budget_rejects_during_payload_decode(
    artifact: GbmPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_scalars = sum(1 + 4 * len(model.stumps) for model in artifact.models.values())
    monkeypatch.setattr(
        gbm_artifacts,
        "MAX_GBM_ARTIFACT_NUMERIC_SCALARS",
        gbm_artifacts._FIXED_NUMERIC_SCALARS + model_scalars,
    )

    with pytest.raises(ValueError, match="numeric limit"):
        GbmPredictorArtifact.from_dict(artifact.to_dict())


def test_gbm_artifact_caps_run_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailIfEmbedded(LocalEmbeddingProvider):
        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            raise AssertionError(f"artifact cap must run first: {tuple(texts)!r}")

    monkeypatch.setattr(gbm_artifacts, "MAX_GBM_ARTIFACT_MODELS", 2)

    with pytest.raises(ValueError, match="model catalogue"):
        fit_calibrated_gbm_artifact(
            load_evaluation_dataset().examples,
            config=_config(),
            embedding_provider=FailIfEmbedded(),
        )


def test_gbm_artifact_outer_fold_ignores_held_out_mutations() -> None:
    examples = load_evaluation_dataset().examples
    original = _science_fold(examples)
    mutated_examples = tuple(
        replace(
            example,
            prompt=f"HELD OUT {example.example_id}",
            outcomes=tuple(replace(outcome, quality=0.01) for outcome in example.outcomes),
        )
        if example.domain == "science"
        else example
        for example in examples
    )
    mutated = _science_fold(mutated_examples)

    first = fit_calibrated_gbm_artifact_for_fold(original, config=_config())
    second = fit_calibrated_gbm_artifact_for_fold(mutated, config=_config())

    assert first.to_json() == second.to_json()
    with pytest.raises(TypeError, match="DomainFold"):
        fit_calibrated_gbm_artifact_for_fold(object())  # type: ignore[arg-type]
