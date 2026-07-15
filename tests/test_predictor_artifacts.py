# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical, fail-closed predictor JSON artifacts."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tierroute.adapters import load_evaluation_dataset
from tierroute.predictors import BilinearPredictorArtifact, fit_calibrated_bilinear


@pytest.fixture(scope="module")
def artifact() -> BilinearPredictorArtifact:
    return fit_calibrated_bilinear(load_evaluation_dataset().examples)


def test_artifact_json_is_canonical_and_round_trips(
    artifact: BilinearPredictorArtifact,
    tmp_path: Path,
) -> None:
    document = artifact.to_json()
    path = artifact.save(tmp_path / "predictor.json")

    loaded = BilinearPredictorArtifact.load(path)

    assert document.endswith("\n")
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


def test_artifact_loader_never_falls_back_to_pickle(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    path.write_bytes(b"\x80\x04N.")

    with pytest.raises(ValueError, match="cannot read predictor artifact"):
        BilinearPredictorArtifact.load(path)
