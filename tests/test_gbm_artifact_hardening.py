# SPDX-License-Identifier: Apache-2.0
"""Adversarial boundary tests for the canonical calibrated-GBM artifact."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import ItemsView, Iterator, Mapping, Sequence
from pathlib import Path
from typing import get_type_hints

import pytest

import tierroute.predictors.artifacts as shared_artifacts
import tierroute.predictors.gbm_artifacts as gbm_artifacts
import tierroute.predictors.resource_limits as predictor_limits
from tierroute.adapters import load_evaluation_dataset
from tierroute.eval import EvaluationExample
from tierroute.features import EmbeddingIdentity, PromptFeatureSchema
from tierroute.predictors import (
    GbmModel,
    GbmPredictorArtifact,
    GbmTrainingConfig,
    IsotonicCalibrator,
    RegressionStump,
    fit_calibrated_gbm_artifact,
    fit_calibrated_gbm_artifact_for_fold,
)


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider="tierroute.tests.gbm-hardening-v1",
        model_id="project-authored-test-embedding",
        revision="1",
        pooling="test-pool",
        normalize=False,
        asset_manifest_sha256="1" * 64,
    )


def _tiny_artifact(
    *,
    schema: PromptFeatureSchema | None = None,
    stumps: tuple[RegressionStump, ...] | None = None,
) -> GbmPredictorArtifact:
    feature_schema = schema or PromptFeatureSchema(
        continuous_means=(0.0, 0.0, 0.0),
        continuous_scales=(1.0, 1.0, 1.0),
        domain_tags=("general",),
    )
    fitted_stumps = stumps or (RegressionStump(0, 1.0, -0.5, 0.5),)
    config = GbmTrainingConfig(
        n_estimators=len(fitted_stumps),
        learning_rate=0.5,
        min_samples_leaf=1,
        min_gain=0.0,
    )
    return GbmPredictorArtifact(
        feature_schema=feature_schema,
        models={
            "m": GbmModel(
                feature_width=feature_schema.dimension,
                base_value=0.25,
                learning_rate=config.learning_rate,
                stumps=fitted_stumps,
            )
        },
        calibrators={"m": IsotonicCalibrator((0.0,), (0.75,))},
        training_data_sha256="0" * 64,
        training_example_count=1,
        training_domains=("d",),
        training_config=config,
    )


def _zero_artifact(zero: float) -> GbmPredictorArtifact:
    schema = PromptFeatureSchema(
        continuous_means=(zero, zero, zero),
        continuous_scales=(1.0, 1.0, 1.0),
        domain_tags=("general",),
    )
    config = GbmTrainingConfig(
        n_estimators=1,
        learning_rate=0.5,
        min_samples_leaf=1,
        min_gain=zero,
    )
    return GbmPredictorArtifact(
        feature_schema=schema,
        models={
            "m": GbmModel(
                feature_width=schema.dimension,
                base_value=zero,
                learning_rate=config.learning_rate,
                stumps=(RegressionStump(0, zero, zero, zero),),
            )
        },
        calibrators={"m": IsotonicCalibrator((zero,), (zero,))},
        training_data_sha256="0" * 64,
        training_example_count=1,
        training_domains=("d",),
        training_config=config,
    )


def _json_preflight_counts(document: str) -> dict[str, int]:
    depth = 0
    maximum_depth = 0
    string_tokens = 0
    maximum_string_characters = 0
    structure_tokens = 0
    index = 0
    while index < len(document):
        character = document[index]
        if character == '"':
            string_tokens += 1
            start = index
            index += 1
            while index < len(document):
                if document[index] == "\\":
                    index += 2
                    continue
                if document[index] == '"':
                    break
                index += 1
            maximum_string_characters = max(
                maximum_string_characters,
                index - start + 1,
            )
        elif character in "[{":
            depth += 1
            maximum_depth = max(maximum_depth, depth)
            structure_tokens += 1
        elif character in "]}":
            depth -= 1
        elif character == ",":
            structure_tokens += 1
        index += 1
    return {
        "maximum_depth": maximum_depth,
        "string_tokens": string_tokens,
        "maximum_string_characters": maximum_string_characters,
        "structure_tokens": structure_tokens,
    }


def _count_json_numbers(value: object) -> int:
    if type(value) in (int, float):
        return 1
    if isinstance(value, Mapping):
        return sum(_count_json_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_json_numbers(item) for item in value)
    return 0


def test_public_gbm_artifact_fit_type_hints_resolve() -> None:
    for fit_function in (
        fit_calibrated_gbm_artifact,
        fit_calibrated_gbm_artifact_for_fold,
    ):
        assert get_type_hints(fit_function)["return"] is GbmPredictorArtifact


def test_tiny_gbm_artifact_has_reviewed_canonical_json_and_sha256() -> None:
    expected = (
        '{"algorithm_id":"tierroute-gradient-boosted-regression-stumps-v1",'
        '"artifact_kind":"tierroute-gbm-predictor","artifact_version":1,'
        '"calibrators":{"m":{"upper_bounds":[0.0],"values":[0.75]}},'
        '"feature_schema":{"continuous_means":[0.0,0.0,0.0],'
        '"continuous_scales":[1.0,1.0,1.0],"domain_tags":["general"],'
        '"embedding_dimension":0,"embedding_identity":null,"schema_version":1},'
        '"models":{"m":{"base_value":0.25,"stumps":[[0,1.0,-0.5,0.5]]}},'
        '"training":{"config":{"learning_rate":0.5,"min_gain":0.0,'
        '"min_samples_leaf":1,"n_estimators":1},'
        '"data_sha256":"0000000000000000000000000000000000000000000000000000000000000000",'
        '"domains":["d"],"example_count":1}}\n'
    )

    document = _tiny_artifact().to_json()

    assert document == expected
    assert hashlib.sha256(document.encode("utf-8")).hexdigest() == (
        "80be8b79e9a6aafd889845a0045c85340366f086a3cc6d4391423b8c2e3a7723"
    )
    assert GbmPredictorArtifact.from_json(document).to_json() == expected


def test_gbm_artifact_canonicalizes_signed_zero_deeply() -> None:
    positive = _zero_artifact(0.0).to_json()
    negative = _zero_artifact(-0.0).to_json()

    assert negative == positive
    assert "-0.0" not in negative


class _IntSubclass(int):
    pass


@pytest.mark.parametrize("spoof_location", ["feature_width", "feature_index"])
def test_gbm_artifact_rejects_integer_subclasses_in_direct_model_state(
    spoof_location: str,
) -> None:
    artifact = _tiny_artifact()
    stump_index: int = _IntSubclass(0) if spoof_location == "feature_index" else 0
    feature_width: int = (
        _IntSubclass(artifact.feature_schema.dimension)
        if spoof_location == "feature_width"
        else artifact.feature_schema.dimension
    )
    model = GbmModel(
        feature_width=feature_width,
        base_value=0.25,
        learning_rate=artifact.training_config.learning_rate,
        stumps=(RegressionStump(stump_index, 1.0, -0.5, 0.5),),
    )

    with pytest.raises(TypeError, match="must be an integer"):
        GbmPredictorArtifact(
            feature_schema=artifact.feature_schema,
            models={"m": model},
            calibrators=artifact.calibrators,
            training_data_sha256=artifact.training_data_sha256,
            training_example_count=artifact.training_example_count,
            training_domains=artifact.training_domains,
            training_config=artifact.training_config,
        )


class _GbmModelSubclass(GbmModel):
    pass


class _IsotonicCalibratorSubclass(IsotonicCalibrator):
    pass


class _PromptFeatureSchemaSubclass(PromptFeatureSchema):
    pass


class _GbmTrainingConfigSubclass(GbmTrainingConfig):
    pass


@pytest.mark.parametrize(
    ("spoof_location", "message"),
    [
        ("model", "exact GbmModel"),
        ("calibrator", "exact IsotonicCalibrator"),
        ("schema", "exact PromptFeatureSchema"),
        ("config", "exact GbmTrainingConfig"),
    ],
)
def test_gbm_artifact_rejects_project_type_subclasses_in_direct_state(
    spoof_location: str,
    message: str,
) -> None:
    artifact = _tiny_artifact()
    kwargs: dict[str, object] = {
        "feature_schema": artifact.feature_schema,
        "models": artifact.models,
        "calibrators": artifact.calibrators,
        "training_data_sha256": artifact.training_data_sha256,
        "training_example_count": artifact.training_example_count,
        "training_domains": artifact.training_domains,
        "training_config": artifact.training_config,
    }
    if spoof_location == "model":
        model = artifact.models["m"]
        kwargs["models"] = {
            "m": _GbmModelSubclass(
                feature_width=model.feature_width,
                base_value=model.base_value,
                learning_rate=model.learning_rate,
                stumps=model.stumps,
            )
        }
    elif spoof_location == "calibrator":
        calibrator = artifact.calibrators["m"]
        kwargs["calibrators"] = {
            "m": _IsotonicCalibratorSubclass(
                calibrator.upper_bounds,
                calibrator.values,
            )
        }
    elif spoof_location == "schema":
        schema = artifact.feature_schema
        kwargs["feature_schema"] = _PromptFeatureSchemaSubclass(
            continuous_means=schema.continuous_means,
            continuous_scales=schema.continuous_scales,
            domain_tags=schema.domain_tags,
        )
    else:
        config = artifact.training_config
        kwargs["training_config"] = _GbmTrainingConfigSubclass(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            min_samples_leaf=config.min_samples_leaf,
            min_gain=config.min_gain,
        )

    with pytest.raises(TypeError, match=message):
        GbmPredictorArtifact(**kwargs)  # type: ignore[arg-type]


class _SingleSnapshotMapping(Mapping[str, object]):
    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = dict(values)
        self.items_calls = 0

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def items(self) -> ItemsView[str, object]:
        self.items_calls += 1
        if self.items_calls > 1:
            raise AssertionError("stateful mapping was read more than once")
        return self._values.items()

    def clear_source(self) -> None:
        self._values.clear()


def test_gbm_artifact_snapshots_stateful_model_and_calibrator_mappings_once() -> None:
    source = _tiny_artifact()
    models = _SingleSnapshotMapping(source.models)
    calibrators = _SingleSnapshotMapping(source.calibrators)

    artifact = GbmPredictorArtifact(
        feature_schema=source.feature_schema,
        models=models,  # type: ignore[arg-type]
        calibrators=calibrators,  # type: ignore[arg-type]
        training_data_sha256=source.training_data_sha256,
        training_example_count=source.training_example_count,
        training_domains=source.training_domains,
        training_config=source.training_config,
    )
    document = artifact.to_json()
    models.clear_source()
    calibrators.clear_source()

    assert models.items_calls == 1
    assert calibrators.items_calls == 1
    assert document == source.to_json()
    assert artifact.to_json() == document


class _SingleSnapshotList(list[object]):
    def __init__(self, values: Sequence[object]) -> None:
        super().__init__(values)
        self.iter_calls = 0

    def __iter__(self) -> Iterator[object]:
        self.iter_calls += 1
        if self.iter_calls > 1:
            raise AssertionError("stateful list was read more than once")
        return super().__iter__()


def test_gbm_from_dict_snapshots_outer_and_inner_stump_list_subclasses_once() -> None:
    artifact = _tiny_artifact()
    payload = artifact.to_dict()
    inner = _SingleSnapshotList([0, 1.0, -0.5, 0.5])
    outer = _SingleSnapshotList([inner])
    payload["models"]["m"]["stumps"] = outer  # type: ignore[index]

    loaded = GbmPredictorArtifact.from_dict(payload)

    assert outer.iter_calls == 1
    assert inner.iter_calls == 1
    assert loaded.to_json() == artifact.to_json()


def test_invalid_gbm_header_is_rejected_before_nested_model_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _tiny_artifact().to_dict()
    payload["artifact_kind"] = "not-a-gbm-artifact"

    def unexpected_stump_decode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("invalid header reached nested stump decoding")

    monkeypatch.setattr(gbm_artifacts, "_stump_payload", unexpected_stump_decode)
    with pytest.raises(ValueError, match="artifact_kind"):
        GbmPredictorArtifact.from_dict(payload)


class _RecordingEmbeddingProvider:
    dimension = 2
    identity = _identity()

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        batch = tuple(texts)
        self.calls.append(batch)
        raise AssertionError(f"artifact preflight must precede embedding: {batch!r}")


def test_gbm_numeric_preflight_precedes_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RecordingEmbeddingProvider()
    monkeypatch.setattr(gbm_artifacts, "MAX_GBM_ARTIFACT_NUMERIC_SCALARS", 1)

    with pytest.raises(ValueError, match="worst-case numeric state"):
        fit_calibrated_gbm_artifact(
            load_evaluation_dataset().examples,
            config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
            embedding_provider=provider,
        )
    assert provider.calls == []


def test_gbm_metadata_preflight_precedes_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _RecordingEmbeddingProvider()
    monkeypatch.setattr(gbm_artifacts, "MAX_PREDICTOR_METADATA_TOTAL_BYTES", 1)

    with pytest.raises(ValueError, match="metadata exceeds"):
        fit_calibrated_gbm_artifact(
            load_evaluation_dataset().examples,
            config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
            embedding_provider=provider,
        )
    assert provider.calls == []


class _NeverUnequal:
    def __ne__(self, other: object) -> bool:
        del other
        return False


class _SpoofingEmbeddingProvider:
    def __init__(self, *, dimension: object, identity: object) -> None:
        self.dimension = dimension
        self.identity = identity
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        batch = tuple(texts)
        self.calls.append(batch)
        raise AssertionError(f"spoofed provider reached embedding: {batch!r}")


@pytest.mark.parametrize(
    ("dimension", "identity", "message"),
    [
        (_NeverUnequal(), _identity(), "exact integer"),
        (2, _NeverUnequal(), "exact EmbeddingIdentity"),
    ],
)
def test_gbm_artifact_fit_rejects_provider_equality_spoofs(
    dimension: object,
    identity: object,
    message: str,
) -> None:
    provider = _SpoofingEmbeddingProvider(dimension=dimension, identity=identity)
    expected = 2 if not isinstance(dimension, int) else _identity()
    spoof = dimension if not isinstance(dimension, int) else identity
    assert not (spoof != expected)

    with pytest.raises(TypeError, match=message):
        fit_calibrated_gbm_artifact(
            load_evaluation_dataset().examples,
            config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
            embedding_provider=provider,  # type: ignore[arg-type]
        )
    assert provider.calls == []


class _EvaluationExampleSubclass(EvaluationExample):
    pass


def test_gbm_artifact_fit_rejects_example_subclass_before_embedding() -> None:
    examples = load_evaluation_dataset().examples
    source = examples[0]
    spoofed = _EvaluationExampleSubclass(
        example_id=source.example_id,
        prompt=source.prompt,
        domain=source.domain,
        outcomes=source.outcomes,
        candidate_models=source.candidate_models,
        router_metadata=source.router_metadata,
    )
    provider = _RecordingEmbeddingProvider()

    with pytest.raises(TypeError, match="exact EvaluationExample"):
        fit_calibrated_gbm_artifact(
            (spoofed, *examples[1:]),
            config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
            embedding_provider=provider,
        )
    assert provider.calls == []


class _HugeLengthSingleExample(
    Sequence[EvaluationExample],
    Iterator[EvaluationExample],
):
    def __init__(self, example: EvaluationExample) -> None:
        self.example = example
        self.next_calls = 0

    def __len__(self) -> int:
        return sys.maxsize

    def __getitem__(self, index: int) -> EvaluationExample:
        if index == 0:
            return self.example
        raise IndexError(index)

    def __iter__(self) -> Iterator[EvaluationExample]:
        return self

    def __next__(self) -> EvaluationExample:
        self.next_calls += 1
        if self.next_calls == 1:
            return self.example
        raise StopIteration


def test_gbm_training_snapshot_ignores_adversarial_length_hint() -> None:
    source = load_evaluation_dataset().examples[0]
    examples = _HugeLengthSingleExample(source)
    provider = _RecordingEmbeddingProvider()

    with pytest.raises(ValueError, match="LODO requires at least two domains"):
        fit_calibrated_gbm_artifact(
            examples,
            config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
            embedding_provider=provider,
        )

    assert examples.next_calls == 2
    assert provider.calls == []


class _InfiniteExamples(Iterator[EvaluationExample]):
    def __init__(self, example: EvaluationExample) -> None:
        self.example = example
        self.next_calls = 0

    def __iter__(self) -> Iterator[EvaluationExample]:
        return self

    def __next__(self) -> EvaluationExample:
        self.next_calls += 1
        if self.next_calls > 3:
            raise AssertionError("training snapshot read past the reviewed limit sentinel")
        return self.example


def test_gbm_training_snapshot_stops_infinite_iterator_at_reviewed_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_evaluation_dataset().examples[0]
    examples = _InfiniteExamples(source)
    provider = _RecordingEmbeddingProvider()
    monkeypatch.setattr(gbm_artifacts, "MAX_GBM_ARTIFACT_TRAINING_EXAMPLES", 2)

    with pytest.raises(ValueError, match="training examples exceed the reviewed limit"):
        fit_calibrated_gbm_artifact(
            examples,
            config=GbmTrainingConfig(n_estimators=1, min_samples_leaf=1),
            embedding_provider=provider,
        )

    assert examples.next_calls == 3
    assert provider.calls == []


def test_mutated_exact_embedding_identity_is_rejected_at_direct_boundaries() -> None:
    identity = _identity()
    object.__setattr__(identity, "asset_manifest_sha256", "mutated")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PromptFeatureSchema(
            continuous_means=(0.0, 0.0, 0.0),
            continuous_scales=(1.0, 1.0, 1.0),
            domain_tags=("general",),
            embedding_dimension=1,
            embedding_identity=identity,
        )

    schema = PromptFeatureSchema(
        continuous_means=(0.0, 0.0, 0.0),
        continuous_scales=(1.0, 1.0, 1.0),
        domain_tags=("general",),
        embedding_dimension=1,
        embedding_identity=_identity(),
    )
    assert schema.embedding_identity is not None
    object.__setattr__(schema.embedding_identity, "asset_manifest_sha256", "mutated")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _tiny_artifact(schema=schema)


def test_reviewed_official_gbm_shape_fits_every_artifact_cap() -> None:
    domain_count = 7
    example_count = 34_778
    model_count = 11
    estimator_count = 256
    numeric_scalars = gbm_artifacts._FIXED_NUMERIC_SCALARS + model_count * (
        1 + 4 * estimator_count + 2 * example_count
    )
    possible_stumps = model_count * estimator_count

    assert numeric_scalars == 776_405
    assert numeric_scalars < gbm_artifacts.MAX_GBM_ARTIFACT_NUMERIC_SCALARS
    assert possible_stumps < gbm_artifacts.MAX_GBM_ARTIFACT_TOTAL_STUMPS
    assert model_count < gbm_artifacts.MAX_GBM_ARTIFACT_MODELS
    assert (
        gbm_artifacts.MAX_GBM_ARTIFACT_NUMERIC_SCALARS
        < shared_artifacts.MAX_PREDICTOR_JSON_NUMBER_TOKENS
    )

    structure_bytes = (
        numeric_scalars + possible_stumps + 12 * model_count + domain_count + domain_count + 128
    )
    conservative_document_bytes = (
        numeric_scalars * gbm_artifacts._MAX_CANONICAL_NUMBER_BYTES
        + gbm_artifacts.MAX_PREDICTOR_METADATA_TOTAL_BYTES * 6
        + structure_bytes
        + gbm_artifacts._MAX_CANONICAL_FIXED_KEY_BYTES
    )
    assert conservative_document_bytes == 31_981_447
    assert conservative_document_bytes < shared_artifacts.MAX_PREDICTOR_ARTIFACT_BYTES


def test_direct_gbm_artifact_canonicalization_enforces_document_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _tiny_artifact()
    document = artifact.to_json()
    exact_bytes = len(document.encode("utf-8"))
    monkeypatch.setattr(shared_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", exact_bytes)
    assert artifact.to_json() == document

    monkeypatch.setattr(shared_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", exact_bytes - 1)
    with pytest.raises(ValueError, match="predictor artifact exceeds"):
        _tiny_artifact()


class _BypassGbmArtifact(GbmPredictorArtifact):
    def __post_init__(self) -> None:
        """Deliberately bypass every project invariant for the save-boundary test."""

    def to_json(self) -> str:
        return '{"noncanonical":true}\n'


def test_inherited_save_rejects_artifact_subclass_without_touching_files(
    tmp_path: Path,
) -> None:
    artifact = _BypassGbmArtifact(
        feature_schema=object(),  # type: ignore[arg-type]
        models={},
        calibrators={},
        training_data_sha256="not-a-hash",
        training_example_count=0,
        training_domains=(),
        training_config=object(),  # type: ignore[arg-type]
    )
    new_path = tmp_path / "new.json"
    with pytest.raises(TypeError, match="exact project type"):
        artifact.save(new_path)
    assert not new_path.exists()

    existing_path = tmp_path / "existing.json"
    existing_path.write_text("preserve", encoding="utf-8")
    with pytest.raises(TypeError, match="exact project type"):
        artifact.save(existing_path)
    assert existing_path.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("limit_name", "count_name", "message"),
    [
        ("MAX_PREDICTOR_JSON_STRING_TOKENS", "string_tokens", "string-token limit"),
        (
            "MAX_PREDICTOR_JSON_STRING_CHARACTERS",
            "maximum_string_characters",
            "JSON string",
        ),
        ("MAX_PREDICTOR_JSON_NESTING_DEPTH", "maximum_depth", "nesting limit"),
        (
            "MAX_PREDICTOR_JSON_STRUCTURE_TOKENS",
            "structure_tokens",
            "structure-token limit",
        ),
    ],
)
def test_gbm_json_structural_preflight_rejects_before_json_loads(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    count_name: str,
    message: str,
) -> None:
    document = _tiny_artifact().to_json()
    exact_count = _json_preflight_counts(document)[count_name]
    monkeypatch.setattr(shared_artifacts, limit_name, exact_count)
    assert GbmPredictorArtifact.from_json(document).to_json() == document

    monkeypatch.setattr(shared_artifacts, limit_name, exact_count - 1)

    def unexpected_parse(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("preflight-rejected GBM artifact reached json.loads")

    monkeypatch.setattr(gbm_artifacts.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match=message):
        GbmPredictorArtifact.from_json(document)


def test_gbm_json_rejects_text_subclass_and_multibyte_oversize_before_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MisleadingDocument(str):
        def __len__(self) -> int:
            return 1

        def encode(self, *args: object, **kwargs: object) -> bytes:
            del args, kwargs
            return b"{}"

    def unexpected_parse(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("preflight-rejected GBM artifact reached json.loads")

    document = _tiny_artifact().to_json()
    monkeypatch.setattr(gbm_artifacts.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="must be text"):
        GbmPredictorArtifact.from_json(MisleadingDocument(document))

    multibyte = document.replace('"domains":["d"]', '"domains":["한글"]')
    assert len(multibyte.encode("utf-8")) > len(multibyte)
    monkeypatch.setattr(
        shared_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        len(multibyte),
    )
    with pytest.raises(ValueError, match="predictor artifact exceeds"):
        GbmPredictorArtifact.from_json(multibyte)


def test_gbm_json_number_token_exact_boundary_and_early_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _tiny_artifact()
    document = artifact.to_json()
    exact_count = _count_json_numbers(artifact.to_dict())
    monkeypatch.setattr(shared_artifacts, "MAX_PREDICTOR_JSON_NUMBER_TOKENS", exact_count)
    assert GbmPredictorArtifact.from_json(document).to_json() == document

    monkeypatch.setattr(
        shared_artifacts,
        "MAX_PREDICTOR_JSON_NUMBER_TOKENS",
        exact_count - 1,
    )

    def unexpected_stump_decode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("number-token rejection reached payload decoding")

    monkeypatch.setattr(gbm_artifacts, "_stump_payload", unexpected_stump_decode)
    with pytest.raises(ValueError, match="not valid strict JSON") as raised:
        GbmPredictorArtifact.from_json(document)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "number-token limit" in str(raised.value.__cause__)


def test_gbm_json_numeric_lexical_boundary() -> None:
    document = _tiny_artifact().to_json()
    exact_number = "1" * predictor_limits.MAX_PREDICTOR_JSON_NUMBER_CHARACTERS
    exact_document = document.replace('"example_count":1', f'"example_count":{exact_number}')

    loaded = GbmPredictorArtifact.from_json(exact_document)
    assert loaded.training_example_count == int(exact_number)

    oversized_number = exact_number + "1"
    oversized_document = document.replace(
        '"example_count":1',
        f'"example_count":{oversized_number}',
    )
    with pytest.raises(ValueError, match="not valid strict JSON") as raised:
        GbmPredictorArtifact.from_json(oversized_document)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "integer exceeds the JSON number limit" in str(raised.value.__cause__)


def test_gbm_total_stump_cap_applies_to_direct_and_mapping_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _tiny_artifact(
        stumps=(
            RegressionStump(0, 0.0, -0.5, 0.5),
            RegressionStump(1, 1.0, -0.25, 0.25),
        )
    )
    payload = artifact.to_dict()
    monkeypatch.setattr(gbm_artifacts, "MAX_GBM_ARTIFACT_TOTAL_STUMPS", 1)

    with pytest.raises(ValueError, match="aggregate stump limit"):
        GbmPredictorArtifact(
            feature_schema=artifact.feature_schema,
            models=artifact.models,
            calibrators=artifact.calibrators,
            training_data_sha256=artifact.training_data_sha256,
            training_example_count=artifact.training_example_count,
            training_domains=artifact.training_domains,
            training_config=artifact.training_config,
        )
    with pytest.raises(ValueError, match="aggregate stump limit"):
        GbmPredictorArtifact.from_dict(payload)
