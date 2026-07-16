# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical, fail-closed predictor JSON artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

import tierroute.predictors.artifacts as predictor_artifacts
import tierroute.predictors.resource_limits as predictor_limits
from tierroute.adapters import load_evaluation_dataset
from tierroute.policies import predictor_artifact_sha256
from tierroute.predictors import (
    NATIVE_C11_RIDGE_SOLVER_ID,
    BilinearPredictorArtifact,
    IsotonicCalibrator,
    fit_calibrated_bilinear,
)


@pytest.fixture(scope="module")
def artifact() -> BilinearPredictorArtifact:
    return fit_calibrated_bilinear(load_evaluation_dataset().examples)


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


def test_artifact_json_is_canonical_and_round_trips(
    artifact: BilinearPredictorArtifact,
    tmp_path: Path,
) -> None:
    document = artifact.to_json()
    path = artifact.save(tmp_path / "predictor.json")

    loaded = BilinearPredictorArtifact.load(path)

    assert document.endswith("\n")
    # Feature math and the reference solver explicitly promise platform-local, not
    # cross-platform, binary64 coefficients. Pin each verified platform independently.
    expected_sha256 = {
        "darwin": "8b1a5dd9d0bbf921144d0133e90d370f15a7ec899772d3e8c7b8295868a5a8b6",
        "linux": "af561ea74c360c0e7b201225a2b4b07f6cdbe64a50d81cc9333b662eb46c10b8",
    }.get(sys.platform)
    if expected_sha256 is None:
        pytest.skip(f"no reviewed synthetic predictor hash for platform {sys.platform!r}")
    assert hashlib.sha256(document.encode("utf-8")).hexdigest() == expected_sha256
    assert loaded.solver_id == "tierroute.centered-ridge-cholesky-python-v1"
    assert loaded.to_json() == document
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


def test_artifact_accepts_reviewed_native_solver_id_without_resolving_it(
    artifact: BilinearPredictorArtifact,
) -> None:
    native_artifact = replace(artifact, solver_id=NATIVE_C11_RIDGE_SOLVER_ID)

    loaded = BilinearPredictorArtifact.from_json(native_artifact.to_json())

    assert loaded.solver_id == NATIVE_C11_RIDGE_SOLVER_ID
    assert loaded.build_predictor().predict_many(
        "offline artifact inference",
        loaded.model_ids,
    ) == native_artifact.build_predictor().predict_many(
        "offline artifact inference",
        native_artifact.model_ids,
    )


def test_training_hash_changes_when_a_fit_input_changes() -> None:
    examples = load_evaluation_dataset().examples
    changed = (
        replace(examples[0], prompt=f"{examples[0].prompt} changed"),
        *examples[1:],
    )

    assert fit_calibrated_bilinear(examples).training_data_sha256 != (
        fit_calibrated_bilinear(changed).training_data_sha256
    )


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
def test_artifact_parser_rejects_non_strict_json(document: str) -> None:
    with pytest.raises(ValueError):
        BilinearPredictorArtifact.from_json(document)


def test_artifact_validation_rejects_schema_and_numeric_mutations(
    artifact: BilinearPredictorArtifact,
) -> None:
    valid = artifact.to_dict()
    model_id = artifact.model_ids[0]
    mutations = []

    missing = copy.deepcopy(valid)
    missing.pop("training")
    mutations.append(missing)

    extra = copy.deepcopy(valid)
    extra["unexpected"] = True
    mutations.append(extra)

    bad_hash = copy.deepcopy(valid)
    bad_hash["training"]["data_sha256"] = "ABC"  # type: ignore[index]
    mutations.append(bad_hash)

    bad_width = copy.deepcopy(valid)
    bad_width["model_weights"][model_id].append(1.0)  # type: ignore[index,union-attr]
    mutations.append(bad_width)

    missing_calibrator = copy.deepcopy(valid)
    missing_calibrator["calibrators"].pop(model_id)  # type: ignore[union-attr]
    mutations.append(missing_calibrator)

    boolean_version = copy.deepcopy(valid)
    boolean_version["artifact_version"] = True
    mutations.append(boolean_version)

    boolean_seed = copy.deepcopy(valid)
    boolean_seed["training"]["seed"] = True  # type: ignore[index]
    mutations.append(boolean_seed)

    boolean_ridge = copy.deepcopy(valid)
    boolean_ridge["training"]["ridge"] = True  # type: ignore[index]
    mutations.append(boolean_ridge)

    boolean_bias = copy.deepcopy(valid)
    boolean_bias["model_bias"][model_id] = False  # type: ignore[index]
    mutations.append(boolean_bias)

    float_version = copy.deepcopy(valid)
    float_version["artifact_version"] = 1.0
    mutations.append(float_version)

    string_weight = copy.deepcopy(valid)
    string_weight["model_weights"][model_id][0] = "0.5"  # type: ignore[index]
    mutations.append(string_weight)

    string_ridge = copy.deepcopy(valid)
    string_ridge["training"]["ridge"] = "1.0"  # type: ignore[index]
    mutations.append(string_ridge)

    unknown_solver = copy.deepcopy(valid)
    unknown_solver["training"]["solver_id"] = "unreviewed-solver"  # type: ignore[index]
    mutations.append(unknown_solver)

    empty_domains = copy.deepcopy(valid)
    empty_domains["training"]["domains"] = []  # type: ignore[index]
    mutations.append(empty_domains)

    non_finite_bias = copy.deepcopy(valid)
    non_finite_bias["model_bias"][model_id] = math.inf  # type: ignore[index]
    mutations.append(non_finite_bias)

    decreasing_bounds = copy.deepcopy(valid)
    decreasing_bounds["calibrators"][model_id] = {  # type: ignore[index]
        "upper_bounds": [0.2, 0.1],
        "values": [0.3, 0.4],
    }
    mutations.append(decreasing_bounds)

    for payload in mutations:
        with pytest.raises((TypeError, ValueError)):
            BilinearPredictorArtifact.from_dict(payload)


def test_artifact_defensively_copies_parameter_mappings(
    artifact: BilinearPredictorArtifact,
) -> None:
    model_id = artifact.model_ids[0]
    original = artifact.model_weights[model_id]

    with pytest.raises(TypeError):
        artifact.model_weights[model_id] = tuple(0.0 for _ in original)  # type: ignore[index]
    with pytest.raises(TypeError):
        artifact.model_bias[model_id] = 0.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        artifact.calibrators.pop(model_id)  # type: ignore[attr-defined]

    assert artifact.model_weights[model_id] == original


def test_artifact_defensively_freezes_nested_numeric_sequences(
    artifact: BilinearPredictorArtifact,
) -> None:
    means = list(artifact.feature_schema.continuous_means)
    scales = list(artifact.feature_schema.continuous_scales)
    schema = replace(
        artifact.feature_schema,
        continuous_means=means,  # type: ignore[arg-type]
        continuous_scales=scales,  # type: ignore[arg-type]
    )
    model_id = artifact.model_ids[0]
    upper_bounds = list(artifact.calibrators[model_id].upper_bounds)
    values = list(artifact.calibrators[model_id].values)
    calibrators = dict(artifact.calibrators)
    calibrators[model_id] = IsotonicCalibrator(
        upper_bounds,  # type: ignore[arg-type]
        values,  # type: ignore[arg-type]
    )
    frozen = replace(artifact, feature_schema=schema, calibrators=calibrators)
    document = frozen.to_json()

    means.append(0.0)
    scales.append(1.0)
    upper_bounds.append(upper_bounds[-1] + 1.0)
    values.append(values[-1])

    assert frozen.to_json() == document
    assert isinstance(frozen.feature_schema.continuous_means, tuple)
    assert isinstance(frozen.feature_schema.continuous_scales, tuple)
    assert isinstance(frozen.calibrators[model_id].upper_bounds, tuple)
    assert isinstance(frozen.calibrators[model_id].values, tuple)


def test_calibrator_snapshots_stateful_sequences_before_validation() -> None:
    class FlippingNumbers(list[float]):
        def __init__(self, first: list[float], later: list[float]) -> None:
            super().__init__()
            self.first = first
            self.later = later
            self.iterations = 0

        def __iter__(self) -> Iterator[float]:
            self.iterations += 1
            return iter(self.first if self.iterations == 1 else self.later)

    upper_bounds = FlippingNumbers([0.0], [math.nan])
    values = FlippingNumbers([0.5], [math.nan])
    calibrator = IsotonicCalibrator(
        upper_bounds,  # type: ignore[arg-type]
        values,  # type: ignore[arg-type]
    )

    assert upper_bounds.iterations == 1
    assert values.iterations == 1
    assert calibrator == IsotonicCalibrator((0.0,), (0.5,))


def test_artifact_snapshots_stateful_direct_inputs_once(
    artifact: BilinearPredictorArtifact,
) -> None:
    class FlippingMapping(Mapping[str, IsotonicCalibrator]):
        def __init__(
            self,
            first: Mapping[str, IsotonicCalibrator],
            later: Mapping[str, IsotonicCalibrator],
        ) -> None:
            self.first = first
            self.later = later
            self.iterations = 0

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            return iter(self.first if self.iterations == 1 else self.later)

        def __len__(self) -> int:
            return len(self.first)

        def __getitem__(self, key: str) -> IsotonicCalibrator:
            source = self.first if self.iterations <= 1 else self.later
            return source[key]

    later = {
        f"extra-{index}": next(iter(artifact.calibrators.values()))
        for index in range(predictor_limits.MAX_PREDICTOR_MODELS + 1)
    }
    source = FlippingMapping(dict(artifact.calibrators), later)
    frozen = replace(artifact, calibrators=source)

    assert source.iterations == 1
    assert dict(frozen.calibrators) == dict(artifact.calibrators)
    assert frozen.to_json() == artifact.to_json()


def test_artifact_normalizes_runtime_errors_from_direct_containers(
    artifact: BilinearPredictorArtifact,
) -> None:
    class ExplodingMapping(Mapping[str, IsotonicCalibrator]):
        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("boom")

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str) -> IsotonicCalibrator:
            raise KeyError(key)

    class ExplodingWeights(list[float]):
        def __iter__(self) -> Iterator[float]:
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="could not be read deterministically"):
        replace(artifact, calibrators=ExplodingMapping())

    weights = dict(artifact.model_weights)
    weights[artifact.model_ids[0]] = ExplodingWeights()
    with pytest.raises(ValueError, match="could not be read deterministically"):
        replace(artifact, model_weights=weights)


def test_artifact_rejects_sequence_subclass_width_lies(
    artifact: BilinearPredictorArtifact,
) -> None:
    class LyingWeights(list[float]):
        def __len__(self) -> int:
            return artifact.feature_schema.dimension

        def __iter__(self) -> Iterator[float]:
            return iter([0.0] * (artifact.feature_schema.dimension + 1))

    model_id = artifact.model_ids[0]
    weights = dict(artifact.model_weights)
    weights[model_id] = LyingWeights()

    with pytest.raises(ValueError, match="numeric limit"):
        replace(artifact, model_weights=weights)


def test_direct_integer_parameters_normalize_to_stable_binary64(
    artifact: BilinearPredictorArtifact,
) -> None:
    inexact_integer = 2**53 + 1
    schema = replace(
        artifact.feature_schema,
        continuous_means=[inexact_integer, 0, 0],  # type: ignore[arg-type]
        continuous_scales=[1, 1, 1],  # type: ignore[arg-type]
    )
    model_id = artifact.model_ids[0]
    calibrators = dict(artifact.calibrators)
    calibrators[model_id] = IsotonicCalibrator(
        [inexact_integer],  # type: ignore[arg-type]
        [inexact_integer],  # type: ignore[arg-type]
    )
    normalized = replace(artifact, feature_schema=schema, calibrators=calibrators)
    document = normalized.to_json()
    reloaded = BilinearPredictorArtifact.from_json(document)

    assert normalized.feature_schema.continuous_means[0] == float(inexact_integer)
    assert normalized.calibrators[model_id].upper_bounds[0] == float(inexact_integer)
    assert reloaded.to_json() == document
    assert predictor_artifact_sha256(reloaded) == predictor_artifact_sha256(normalized)


def test_artifact_weight_bias_and_ridge_integers_normalize_to_float_hash(
    artifact: BilinearPredictorArtifact,
) -> None:
    integer_weights = {
        model_id: [0] * artifact.feature_schema.dimension for model_id in artifact.model_ids
    }
    float_weights = {
        model_id: [0.0] * artifact.feature_schema.dimension for model_id in artifact.model_ids
    }
    integer_variant = replace(
        artifact,
        model_weights=integer_weights,
        model_bias={model_id: 0 for model_id in artifact.model_ids},
        ridge=1,
    )
    float_variant = replace(
        artifact,
        model_weights=float_weights,
        model_bias={model_id: 0.0 for model_id in artifact.model_ids},
        ridge=1.0,
    )

    assert all(
        type(value) is float
        for weights in integer_variant.model_weights.values()
        for value in weights
    )
    assert all(type(value) is float for value in integer_variant.model_bias.values())
    assert type(integer_variant.ridge) is float
    assert integer_variant.to_json() == float_variant.to_json()
    assert predictor_artifact_sha256(integer_variant) == predictor_artifact_sha256(float_variant)


def test_artifact_rejects_numeric_primitive_subclasses(
    artifact: BilinearPredictorArtifact,
) -> None:
    class LyingFloat(float):
        def __float__(self) -> float:
            return 1.0

    with pytest.raises(ValueError, match="ridge must be a number"):
        replace(artifact, ridge=LyingFloat(1.0))

    model_id = artifact.model_ids[0]
    weights = {key: list(value) for key, value in artifact.model_weights.items()}
    weights[model_id][0] = LyingFloat(weights[model_id][0])
    with pytest.raises(ValueError, match="must be a number"):
        replace(artifact, model_weights=weights)


def test_artifact_loader_never_falls_back_to_pickle(tmp_path: Path) -> None:
    binary_pickle = tmp_path / "binary.pkl"
    binary_pickle.write_bytes(b"\x80\x04N.")
    ascii_pickle = tmp_path / "ascii.pkl"
    ascii_pickle.write_bytes(b"N.")

    with pytest.raises(ValueError, match="cannot read predictor artifact"):
        BilinearPredictorArtifact.load(binary_pickle)
    with pytest.raises(ValueError, match="not valid strict JSON"):
        BilinearPredictorArtifact.load(ascii_pickle)


def test_artifact_document_limit_applies_to_to_json_from_json_load_and_save(
    artifact: BilinearPredictorArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = artifact.to_json()
    byte_count = len(document.encode("utf-8"))
    source = tmp_path / "source.json"
    source.write_text(document, encoding="utf-8")

    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", byte_count)
    assert artifact.to_json() == document
    assert BilinearPredictorArtifact.from_json(document).to_json() == document
    assert BilinearPredictorArtifact.load(source).to_json() == document

    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        byte_count - 1,
    )
    message = "predictor artifact exceeds"
    with pytest.raises(ValueError, match=message):
        artifact.to_json()
    with pytest.raises(ValueError, match=message):
        BilinearPredictorArtifact.from_json(document)
    with pytest.raises(ValueError, match=message):
        BilinearPredictorArtifact.load(source)

    destination = tmp_path / "existing.json"
    destination.write_text("preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        artifact.save(destination)
    assert destination.read_text(encoding="utf-8") == "preserve me"


def test_artifact_multibyte_document_exact_boundary(
    artifact: BilinearPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multibyte_artifact = replace(artifact, training_domains=("한글",))
    document = multibyte_artifact.to_json()
    byte_count = len(document.encode("utf-8"))
    assert byte_count > len(document)

    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", byte_count)
    assert BilinearPredictorArtifact.from_json(document).to_json() == document

    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", byte_count - 1)
    with pytest.raises(ValueError, match="predictor artifact exceeds"):
        BilinearPredictorArtifact.from_json(document)


def test_artifact_oversize_and_multibyte_rejection_precede_json_parsing(
    artifact: BilinearPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = artifact.to_json()
    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        len(document.encode("utf-8")) - 1,
    )

    def unexpected_parse(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("oversized artifact reached json.loads")

    monkeypatch.setattr(predictor_artifacts.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="predictor artifact exceeds"):
        BilinearPredictorArtifact.from_json(document)

    multibyte = '{"unexpected":"한"}'
    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        len(multibyte),
    )
    assert len(multibyte.encode("utf-8")) > len(multibyte)
    with pytest.raises(ValueError, match="predictor artifact exceeds"):
        BilinearPredictorArtifact.from_json(multibyte)


def test_artifact_rejects_text_subclasses_before_size_or_parse(
    artifact: BilinearPredictorArtifact,
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
        raise AssertionError("text subclass reached json.loads")

    monkeypatch.setattr(predictor_artifacts.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="must be text"):
        BilinearPredictorArtifact.from_json(MisleadingDocument(artifact.to_json()))


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
def test_artifact_structural_preflight_exact_boundary_and_early_rejection(
    artifact: BilinearPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    count_name: str,
    message: str,
) -> None:
    document = artifact.to_json()
    exact_count = _json_preflight_counts(document)[count_name]
    monkeypatch.setattr(predictor_artifacts, limit_name, exact_count)
    assert BilinearPredictorArtifact.from_json(document).to_json() == document

    monkeypatch.setattr(predictor_artifacts, limit_name, exact_count - 1)

    def unexpected_parse(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("preflight-rejected artifact reached json.loads")

    monkeypatch.setattr(predictor_artifacts.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match=message):
        BilinearPredictorArtifact.from_json(document)


def test_artifact_json_number_counter_bounds_unknown_fields(
    artifact: BilinearPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = artifact.to_json()
    unknown_numbers = document[:-2] + ',"ignored":[0,0,0]}\n'
    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_JSON_NUMBER_TOKENS",
        _count_json_numbers(artifact.to_dict()) + 2,
    )

    with pytest.raises(ValueError, match="not valid strict JSON") as raised:
        BilinearPredictorArtifact.from_json(unknown_numbers)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "number-token limit" in str(raised.value.__cause__)


def test_artifact_json_number_counter_exact_boundary(
    artifact: BilinearPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = artifact.to_json()
    exact_count = _count_json_numbers(artifact.to_dict())
    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_JSON_NUMBER_TOKENS",
        exact_count,
    )
    assert BilinearPredictorArtifact.from_json(document).to_json() == document

    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_JSON_NUMBER_TOKENS",
        exact_count - 1,
    )
    with pytest.raises(ValueError, match="not valid strict JSON") as raised:
        BilinearPredictorArtifact.from_json(document)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "number-token limit" in str(raised.value.__cause__)


@pytest.mark.parametrize(
    "document",
    [
        "[" * 2_000 + "0" + "]" * 2_000,
        None,
    ],
)
def test_artifact_parser_normalizes_recursion_and_non_text(
    document: object,
) -> None:
    with pytest.raises(ValueError):
        BilinearPredictorArtifact.from_json(document)  # type: ignore[arg-type]


def test_artifact_parser_rejects_oversized_numeric_tokens(
    artifact: BilinearPredictorArtifact,
) -> None:
    integer = "1" * (predictor_limits.MAX_PREDICTOR_JSON_NUMBER_CHARACTERS + 1)
    integer_document = artifact.to_json().replace('"seed":0', f'"seed":{integer}')
    with pytest.raises(ValueError, match="not valid strict JSON"):
        BilinearPredictorArtifact.from_json(integer_document)

    accepted_negative = "-" + "1" * (predictor_limits.MAX_PREDICTOR_JSON_NUMBER_CHARACTERS - 1)
    accepted_document = artifact.to_json().replace('"seed":0', f'"seed":{accepted_negative}')
    assert BilinearPredictorArtifact.from_json(accepted_document).seed == int(accepted_negative)

    rejected_negative = "-" + "1" * predictor_limits.MAX_PREDICTOR_JSON_NUMBER_CHARACTERS
    rejected_document = artifact.to_json().replace('"seed":0', f'"seed":{rejected_negative}')
    with pytest.raises(ValueError, match="not valid strict JSON"):
        BilinearPredictorArtifact.from_json(rejected_document)

    finite_fractional = "0." + "1" * (predictor_limits.MAX_PREDICTOR_JSON_NUMBER_CHARACTERS - 2)
    assert len(finite_fractional) == predictor_limits.MAX_PREDICTOR_JSON_NUMBER_CHARACTERS
    finite_document = artifact.to_json().replace('"ridge":1.0', f'"ridge":{finite_fractional}')
    assert math.isfinite(BilinearPredictorArtifact.from_json(finite_document).ridge)

    fractional = "0." + "1" * predictor_limits.MAX_PREDICTOR_JSON_NUMBER_CHARACTERS
    float_document = artifact.to_json().replace('"ridge":1.0', f'"ridge":{fractional}')
    with pytest.raises(ValueError, match="not valid strict JSON"):
        BilinearPredictorArtifact.from_json(float_document)

    overflowing_float = artifact.to_json().replace('"ridge":1.0', '"ridge":1e309')
    with pytest.raises(ValueError, match="not valid strict JSON") as raised:
        BilinearPredictorArtifact.from_json(overflowing_float)
    assert isinstance(raised.value.__cause__, ValueError)
    assert "finite binary64" in str(raised.value.__cause__)


def test_artifact_integer_contract_survives_minimum_cpython_digit_limit(
    artifact: BilinearPredictorArtifact,
) -> None:
    set_limit = getattr(sys, "set_int_max_str_digits", None)
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    if set_limit is None or get_limit is None:
        return
    original = get_limit()
    try:
        set_limit(640)
        bounded = replace(artifact, seed=10**639)
        document = bounded.to_json()
        assert BilinearPredictorArtifact.from_json(document).seed == bounded.seed
    finally:
        set_limit(original)


def test_artifact_rejects_escaped_surrogate_metadata(
    artifact: BilinearPredictorArtifact,
) -> None:
    payload = copy.deepcopy(artifact.to_dict())
    old_model_id = artifact.model_ids[0]
    for field in ("model_weights", "model_bias", "calibrators"):
        mapping = payload[field]
        mapping["\ud800"] = mapping.pop(old_model_id)  # type: ignore[union-attr]
    document = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    with pytest.raises(ValueError, match="valid Unicode"):
        BilinearPredictorArtifact.from_json(document)


def test_artifact_rejects_primitive_subclasses_at_trust_boundaries(
    artifact: BilinearPredictorArtifact,
) -> None:
    class LyingText(str):
        def strip(self, *args: object, **kwargs: object) -> str:
            del args, kwargs
            return "safe"

        def encode(self, *args: object, **kwargs: object) -> bytes:
            del args, kwargs
            return b"safe"

    model_id = artifact.model_ids[0]
    renamed_weights = dict(artifact.model_weights)
    renamed_bias = dict(artifact.model_bias)
    renamed_calibrators = dict(artifact.calibrators)
    for mapping in (renamed_weights, renamed_bias, renamed_calibrators):
        mapping[LyingText("m" * 5_000)] = mapping.pop(model_id)

    with pytest.raises(ValueError, match="string-keyed"):
        replace(
            artifact,
            model_weights=renamed_weights,
            model_bias=renamed_bias,
            calibrators=renamed_calibrators,
        )


def test_predictor_policy_hash_inherits_serialization_byte_limit(
    artifact: BilinearPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = artifact.to_json()
    byte_count = len(document.encode("utf-8"))
    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        byte_count,
    )
    assert predictor_artifact_sha256(artifact) == hashlib.sha256(document.encode()).hexdigest()

    monkeypatch.setattr(
        predictor_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        byte_count - 1,
    )
    with pytest.raises(ValueError, match="predictor artifact exceeds"):
        predictor_artifact_sha256(artifact)


def test_predictor_policy_hash_rejects_dynamic_serialization_subclasses(
    artifact: BilinearPredictorArtifact,
) -> None:
    class SpoofedArtifact(BilinearPredictorArtifact):
        def to_json(self) -> str:
            return "not the canonical artifact\n"

    spoofed = SpoofedArtifact(
        feature_schema=artifact.feature_schema,
        model_weights=artifact.model_weights,
        model_bias=artifact.model_bias,
        calibrators=artifact.calibrators,
        training_data_sha256=artifact.training_data_sha256,
        training_example_count=artifact.training_example_count,
        training_domains=artifact.training_domains,
        ridge=artifact.ridge,
        seed=artifact.seed,
        solver_id=artifact.solver_id,
        artifact_version=artifact.artifact_version,
    )

    with pytest.raises(TypeError, match="exact artifact type"):
        predictor_artifact_sha256(spoofed)


@pytest.mark.parametrize("field", ["weight", "bias", "ridge"])
def test_artifact_direct_numeric_overflow_is_a_value_error(
    artifact: BilinearPredictorArtifact,
    field: str,
) -> None:
    payload = copy.deepcopy(artifact.to_dict())
    huge = 10**10_000
    if field == "weight":
        model_id = artifact.model_ids[0]
        payload["model_weights"][model_id][0] = huge  # type: ignore[index]
    elif field == "bias":
        model_id = artifact.model_ids[0]
        payload["model_bias"][model_id] = huge  # type: ignore[index]
    else:
        payload["training"]["ridge"] = huge  # type: ignore[index]

    with pytest.raises(ValueError, match="finite"):
        BilinearPredictorArtifact.from_dict(payload)


def test_nested_predictor_numeric_overflow_is_a_value_error(
    artifact: BilinearPredictorArtifact,
) -> None:
    huge = 10**10_000
    with pytest.raises(ValueError, match="continuous means must be finite"):
        replace(
            artifact.feature_schema,
            continuous_means=(huge, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="calibration parameters must be finite"):
        IsotonicCalibrator((huge,), (0.5,))


def test_artifact_structural_limits_apply_to_direct_construction(
    artifact: BilinearPredictorArtifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_MODELS", 2)
    with pytest.raises(ValueError, match="model limit"):
        replace(artifact)

    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_MODELS", 4_096)
    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_NUMERIC_SCALARS", 10)
    with pytest.raises(ValueError, match="numeric scalar limit"):
        replace(artifact)

    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_NUMERIC_SCALARS", 1_000_000)
    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_TRAINING_DOMAINS", 3)
    with pytest.raises(ValueError, match="training_domains"):
        replace(artifact)

    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_TRAINING_DOMAINS", 4_096)
    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_METADATA_TEXT_BYTES", 1)
    with pytest.raises(ValueError, match="metadata limit"):
        replace(artifact)

    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_METADATA_TEXT_BYTES", 4 * 1024)
    monkeypatch.setattr(predictor_artifacts, "MAX_PREDICTOR_METADATA_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="aggregate limit"):
        replace(artifact)


def test_artifact_calibrator_cannot_exceed_recorded_training_rows(
    artifact: BilinearPredictorArtifact,
) -> None:
    with pytest.raises(ValueError, match="exceeds training_example_count"):
        replace(artifact, training_example_count=1)


def test_planned_routerbench_bge_shape_fits_predictor_resource_contract() -> None:
    models = 11
    features = 1_036
    examples = 34_778
    numeric_scalars = 7 + models * features + models + 2 * models * examples
    pessimistic_float_json_bytes = numeric_scalars * 26

    assert numeric_scalars == 776_530
    assert numeric_scalars < predictor_limits.MAX_PREDICTOR_NUMERIC_SCALARS
    assert pessimistic_float_json_bytes < predictor_limits.MAX_PREDICTOR_ARTIFACT_BYTES
