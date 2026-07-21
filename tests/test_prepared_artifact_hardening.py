# SPDX-License-Identifier: Apache-2.0
"""Adversarial boundary tests for prepared predictor artifacts."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import socket
import struct
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import NoReturn

import pytest

import tierroute.predictors.artifacts as shared_artifacts
import tierroute.predictors.prepared_artifacts as prepared_artifacts
import tierroute.predictors.prepared_assembly as prepared_assembly
from tierroute.features import EmbeddingIdentity
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.prepared_artifacts import (
    PreparedAllDomainStatistics,
    PreparedArtifactLineage,
    PreparedBilinearPredictorArtifact,
    PreparedCalibrationSource,
    PreparedFinalCoefficient,
    PreparedModelCalibration,
    PreparedPredictorTargetShard,
    prepared_calibration_input_sha256,
)
from tierroute.predictors.prepared_graph import (
    PreparedNestedLodoPlan,
    build_prepared_nested_lodo_plan,
)
from tierroute.predictors.prepared_store import (
    _packed_upper_index,
    _packed_upper_length,
)


@dataclass(frozen=True, slots=True)
class _Fixture:
    plan: PreparedNestedLodoPlan
    aggregate: PreparedAllDomainStatistics
    coefficient: PreparedFinalCoefficient
    sources: tuple[PreparedCalibrationSource, ...]
    artifact: PreparedBilinearPredictorArtifact


def _semantic_sources(
    plan: PreparedNestedLodoPlan,
) -> tuple[PreparedCalibrationSource, ...]:
    sources = []
    for domain_index, domain in enumerate(plan.domains):
        included = tuple(index for index in range(len(plan.domains)) if index != domain_index)
        subset_index = next(
            index
            for index, subset in enumerate(plan.training_subsets)
            if subset.domain_indices == included
        )
        block_index = next(
            index
            for index, block in enumerate(plan.score_blocks)
            if block.training_subset_index == subset_index
            and block.scored_domain_index == domain_index
        )
        sources.append(
            PreparedCalibrationSource(
                held_out_domain_index=domain_index,
                held_out_domain=domain,
                training_subset_index=subset_index,
                score_block_index=block_index,
                row_count=plan.domain_example_counts[domain_index],
                raw_score_block_sha256=f"{domain_index + 1:x}" * 64,
                scored_feature_shard_sha256=f"{domain_index + 5:x}" * 64,
                target_shard_sha256=f"{domain_index + 9:x}" * 64,
            )
        )
    return tuple(sources)


def _fixture(*, embedded: bool = False) -> _Fixture:
    embedding_dimension = 2 if embedded else 0
    embedding_identity = (
        EmbeddingIdentity(
            provider="fixture-local",
            model_id="fixture/model",
            revision="frozen-revision",
            pooling="mean",
            normalize=True,
            asset_manifest_sha256="e" * 64,
        )
        if embedded
        else None
    )
    plan = build_prepared_nested_lodo_plan(
        ("a", "b", "c", "d"),
        (1, 1, 1, 1),
        feature_count=prepared_artifacts._UNIVERSAL_SURFACE_DIMENSION + embedding_dimension,
        target_count=2,
    )
    feature_means = (
        1.0,
        2.0,
        3.0,
        0.5,
        0.5,
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        *(0.25 for _ in range(embedding_dimension)),
    )
    centered_xx = [0.0] * _packed_upper_length(plan.feature_count)
    for index in range(plan.feature_count):
        centered_xx[_packed_upper_index(plan.feature_count, index, index)] = 1.0
    aggregate = PreparedAllDomainStatistics(
        plan=plan,
        store_sha256="1" * 64,
        statistics_bundle_sha256="2" * 64,
        model_ids=("m1", "m2"),
        embedding_identity=embedding_identity,
        embedding_dimension=embedding_dimension,
        domain_statistics_sha256s=tuple(character * 64 for character in "3456"),
        row_count=4,
        active_tag_mask=0b101,
        feature_means=feature_means,
        target_means=(0.4, 0.6),
        centered_xx_packed=tuple(centered_xx),
        centered_xy=(0.0,) * (plan.feature_count * 2),
    )
    width = aggregate.feature_schema.dimension
    weights = tuple(index / 10.0 for index in range(2 * width))
    coefficient = PreparedFinalCoefficient(
        feature_schema=aggregate.feature_schema,
        active_feature_indices=prepared_artifacts._active_feature_indices(aggregate.feature_schema),
        model_ids=("m1", "m2"),
        aggregate_statistics_sha256=aggregate.sha256,
        ridge=1.0,
        weights_payload=struct.pack(f"<{len(weights)}d", *weights),
        intercepts_payload=struct.pack("<2d", 0.1, 0.2),
    )
    sources = _semantic_sources(plan)
    calibrations = {}
    for model_index, model_id in enumerate(coefficient.model_ids):
        pairs = tuple(
            (float(row_index + model_index), float(row_index % 2))
            for row_index in range(plan.work.example_count)
        )
        calibrations[model_id] = PreparedModelCalibration(
            model_id=model_id,
            sources=sources,
            input_sha256=prepared_calibration_input_sha256(model_id, sources, pairs),
            calibrator=IsotonicCalibrator.fit(
                [prediction for prediction, _ in pairs],
                [target for _, target in pairs],
            ),
        )
    lineage = PreparedArtifactLineage(
        source_fit_sha256="a" * 64,
        store_sha256=aggregate.store_sha256,
        statistics_bundle_sha256=aggregate.statistics_bundle_sha256,
        raw_score_bundle_sha256="b" * 64,
        embedding_snapshot_sha256="c" * 64 if embedded else None,
        aggregate_statistics_sha256=aggregate.sha256,
        final_coefficient_sha256=coefficient.sha256,
        calibration_sources=sources,
    )
    artifact = PreparedBilinearPredictorArtifact.from_prepared_components(
        coefficient,
        calibrations,
        training_domains=plan.domains,
        training_example_count=plan.work.example_count,
        lineage=lineage,
    )
    return _Fixture(plan, aggregate, coefficient, sources, artifact)


def _nested(payload: dict[str, object], path: tuple[str | int, ...]) -> object:
    current: object = payload
    for component in path:
        if type(component) is int:
            assert type(current) is list
            current = current[component]
        else:
            assert type(current) is dict
            current = current[component]
    return current


class _PoisonIterable:
    def __iter__(self) -> NoReturn:
        raise AssertionError("bounded validation traversed a poisoned iterable")


class _PoisonLength:
    def __len__(self) -> NoReturn:
        raise AssertionError("bounded validation measured a poisoned payload")


class _PoisonDict(dict[str, object]):
    def items(self) -> NoReturn:
        raise AssertionError("bounded validation traversed poisoned mapping items")

    def __iter__(self) -> NoReturn:
        raise AssertionError("bounded validation traversed a poisoned mapping")


class _PoisonList(list[str]):
    def __iter__(self) -> NoReturn:
        raise AssertionError("bounded validation traversed a poisoned list")


def test_model_calibration_rejects_mutated_calibrator_before_traversal() -> None:
    fixture = _fixture()
    calibration = fixture.artifact.models["m1"].calibration
    object.__setattr__(calibration.calibrator, "upper_bounds", _PoisonIterable())

    with pytest.raises(TypeError, match="exact tuples"):
        PreparedModelCalibration(
            model_id=calibration.model_id,
            sources=calibration.sources,
            input_sha256=calibration.input_sha256,
            calibrator=calibration.calibrator,
        )


@pytest.mark.parametrize("entry_point", ["to_dict", "to_json"])
def test_artifact_reads_resnapshot_mutated_calibrator_before_serialization(
    entry_point: str,
) -> None:
    artifact = _fixture().artifact
    calibration = artifact.models["m1"].calibration
    object.__setattr__(calibration.calibrator, "values", _PoisonIterable())

    with pytest.raises(TypeError, match="exact tuples"):
        getattr(artifact, entry_point)()


def test_artifact_rejects_replaced_mapping_proxy_without_traversal() -> None:
    artifact = _fixture().artifact
    object.__setattr__(
        artifact,
        "models",
        MappingProxyType(_PoisonDict(artifact.models)),
    )

    with pytest.raises(ValueError, match="mapping changed"):
        artifact.to_json()


def test_artifact_rejects_training_domain_list_subclass_without_traversal() -> None:
    artifact = _fixture().artifact
    object.__setattr__(
        artifact,
        "training_domains",
        _PoisonList(artifact.training_domains),
    )

    with pytest.raises(TypeError, match="exact tuple"):
        artifact.to_json()


def test_artifact_to_json_rejects_mutated_schema_before_traversal() -> None:
    artifact = _fixture().artifact
    object.__setattr__(artifact.feature_schema, "continuous_means", _PoisonIterable())

    with pytest.raises(ValueError, match="three-value tuple"):
        artifact.to_json()


def test_artifact_caps_mutated_schema_tag_before_schema_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    object.__setattr__(artifact.feature_schema, "domain_tags", ("x" * 4097,))

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("oversized schema tag reached schema reconstruction")

    monkeypatch.setattr(type(artifact.feature_schema), "__post_init__", forbidden)
    with pytest.raises(ValueError, match="UTF-8 byte limit"):
        artifact.to_json()


def test_artifact_caps_mutated_embedding_text_before_identity_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture(embedded=True).artifact
    assert artifact.feature_schema.embedding_identity is not None
    object.__setattr__(
        artifact.feature_schema.embedding_identity,
        "provider",
        "x" * 4097,
    )

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("oversized identity reached identity reconstruction")

    monkeypatch.setattr(EmbeddingIdentity, "__post_init__", forbidden)
    with pytest.raises(ValueError, match="UTF-8 byte limit"):
        artifact.to_json()


def test_artifact_caps_mutated_training_domain_before_serialization() -> None:
    artifact = _fixture().artifact
    object.__setattr__(
        artifact,
        "training_domains",
        ("x" * 4097, *artifact.training_domains[1:]),
    )

    with pytest.raises(ValueError, match="UTF-8 byte limit"):
        artifact.to_json()


def test_from_prepared_components_checks_lineage_before_coefficient_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("invalid lineage reached coefficient extraction")

    monkeypatch.setattr(PreparedFinalCoefficient, "weights_for_model_index", forbidden)
    with pytest.raises(TypeError, match="lineage"):
        PreparedBilinearPredictorArtifact.from_prepared_components(
            fixture.coefficient,
            {model_id: state.calibration for model_id, state in fixture.artifact.models.items()},
            training_domains=fixture.plan.domains,
            training_example_count=fixture.plan.work.example_count,
            lineage=None,  # type: ignore[arg-type]
        )


def test_from_prepared_components_rejects_poisoned_payload_before_len() -> None:
    fixture = _fixture()
    object.__setattr__(fixture.coefficient, "weights_payload", _PoisonLength())

    with pytest.raises(TypeError, match="immutable bytes"):
        PreparedBilinearPredictorArtifact.from_prepared_components(
            fixture.coefficient,
            {model_id: state.calibration for model_id, state in fixture.artifact.models.items()},
            training_domains=fixture.plan.domains,
            training_example_count=fixture.plan.work.example_count,
            lineage=fixture.artifact.lineage,
        )


def test_artifact_global_scalar_cap_rejects_before_model_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("numeric overflow reached model-state copying")

    monkeypatch.setattr(prepared_artifacts, "MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS", 7)
    monkeypatch.setattr(PreparedModelCalibration, "__post_init__", forbidden)
    with pytest.raises(ValueError, match="numeric scalar limit"):
        PreparedBilinearPredictorArtifact(
            feature_schema=artifact.feature_schema,
            models=artifact.models,
            training_domains=artifact.training_domains,
            training_example_count=artifact.training_example_count,
            ridge=artifact.ridge,
            lineage=artifact.lineage,
        )


def test_component_factory_global_scalar_cap_rejects_before_model_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("factory numeric overflow reached model-state copying")

    monkeypatch.setattr(prepared_artifacts, "MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS", 7)
    monkeypatch.setattr(PreparedFinalCoefficient, "__post_init__", forbidden)
    monkeypatch.setattr(prepared_artifacts.PreparedModelState, "__post_init__", forbidden)
    with pytest.raises(ValueError, match="numeric scalar limit"):
        PreparedBilinearPredictorArtifact.from_prepared_components(
            fixture.coefficient,
            {model_id: state.calibration for model_id, state in fixture.artifact.models.items()},
            training_domains=fixture.plan.domains,
            training_example_count=fixture.plan.work.example_count,
            lineage=fixture.artifact.lineage,
        )


@pytest.mark.parametrize(
    ("path", "required"),
    (
        (("feature_schema",), "continuous_means"),
        (("models", "m1"), "weights"),
        (("models", "m1", "calibrator"), "input_sha256"),
        (("training",), "ridge"),
        (("lineage",), "store_sha256"),
        (("lineage", "calibration_sources", 0), "row_count"),
    ),
)
def test_nested_objects_reject_unknown_and_missing_fields(
    path: tuple[str | int, ...],
    required: str,
) -> None:
    artifact = _fixture().artifact
    for mutation in ("unknown", "missing"):
        payload = copy.deepcopy(artifact.to_dict())
        node = _nested(payload, path)
        assert type(node) is dict
        if mutation == "unknown":
            node["unexpected"] = None
        else:
            del node[required]
        with pytest.raises(ValueError, match="field"):
            PreparedBilinearPredictorArtifact.from_dict(payload)


def test_embedded_identity_rejects_unknown_and_missing_fields() -> None:
    artifact = _fixture(embedded=True).artifact
    for mutation in ("unknown", "missing"):
        payload = copy.deepcopy(artifact.to_dict())
        node = _nested(payload, ("feature_schema", "embedding_identity"))
        assert type(node) is dict
        if mutation == "unknown":
            node["unexpected"] = None
        else:
            del node["revision"]
        with pytest.raises(ValueError, match="field"):
            PreparedBilinearPredictorArtifact.from_dict(payload)


@pytest.mark.parametrize(
    "path",
    (
        ("models", "m1", "calibrator", "identity_sha256"),
        ("lineage", "calibration_sources", 0, "target_shard_sha256"),
        ("training", "solver_id"),
    ),
)
def test_nested_duplicate_json_members_are_rejected(
    path: tuple[str | int, ...],
) -> None:
    payload = _fixture().artifact.to_dict()
    value = _nested(payload, path)
    assert type(path[-1]) is str
    member = (
        json.dumps(path[-1], ensure_ascii=False)
        + ":"
        + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    document = _fixture().artifact.to_json()
    assert document.count(member) >= 1
    duplicated = document.replace(member, f"{member},{member}", 1)
    with pytest.raises(ValueError, match="strict JSON"):
        PreparedBilinearPredictorArtifact.from_json(duplicated)


def _artifact_numeric_scalars(artifact: PreparedBilinearPredictorArtifact) -> int:
    return 7 + sum(
        len(state.weights) + 1 + 2 * len(state.calibration.calibrator.upper_bounds)
        for state in artifact.models.values()
    )


def _json_number_tokens(document: str) -> int:
    count = 0

    def parse_int(token: str) -> int:
        nonlocal count
        count += 1
        return int(token)

    def parse_float(token: str) -> float:
        nonlocal count
        count += 1
        return float(token)

    json.loads(document, parse_int=parse_int, parse_float=parse_float)
    return count


def test_numeric_scalar_cap_accepts_exact_and_rejects_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    count = _artifact_numeric_scalars(artifact)
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS",
        count,
    )
    assert PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict()) == artifact
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREPARED_ARTIFACT_NUMERIC_SCALARS",
        count - 1,
    )
    with pytest.raises(ValueError, match="limit"):
        PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict())


def test_json_number_token_cap_accepts_exact_and_rejects_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    document = artifact.to_json()
    count = _json_number_tokens(document)
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREPARED_ARTIFACT_JSON_NUMBER_TOKENS",
        count,
    )
    assert PreparedBilinearPredictorArtifact.from_json(document) == artifact

    def no_construction(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("number-token overflow reached object construction")

    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREPARED_ARTIFACT_JSON_NUMBER_TOKENS",
        count - 1,
    )
    monkeypatch.setattr(
        PreparedBilinearPredictorArtifact,
        "from_dict",
        classmethod(no_construction),
    )
    with pytest.raises(ValueError, match="strict JSON"):
        PreparedBilinearPredictorArtifact.from_json(document)


def test_json_number_character_cap_accepts_exact_and_rejects_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_JSON_NUMBER_CHARACTERS",
        3,
    )
    with pytest.raises(ValueError, match="fields"):
        PreparedBilinearPredictorArtifact.from_json('{"n":123}')
    with pytest.raises(ValueError, match="strict JSON"):
        PreparedBilinearPredictorArtifact.from_json('{"n":1234}')


def test_parser_depth_cap_accepts_exact_and_rejects_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared_artifacts, "MAX_PREDICTOR_JSON_NESTING_DEPTH", 3)
    with pytest.raises(ValueError, match="artifact must be"):
        PreparedBilinearPredictorArtifact.from_json("[[[0]]]")
    with pytest.raises(ValueError, match="nesting"):
        PreparedBilinearPredictorArtifact.from_json("[[[[0]]]]")


def test_model_cap_accepts_exact_and_rejects_one_over_before_state_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    monkeypatch.setattr(prepared_artifacts, "MAX_PREPARED_ARTIFACT_MODELS", 2)
    assert PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict()) == artifact
    payload = artifact.to_dict()

    real_decode = prepared_artifacts._json_f64_tuple
    calls = 0

    def count_decode(*args: object, **kwargs: object) -> tuple[float, ...]:
        nonlocal calls
        calls += 1
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(prepared_artifacts, "_json_f64_tuple", count_decode)
    monkeypatch.setattr(prepared_artifacts, "MAX_PREPARED_ARTIFACT_MODELS", 1)
    with pytest.raises(ValueError, match="field-count limit"):
        PreparedBilinearPredictorArtifact.from_dict(payload)
    assert calls == 0


def test_calibrator_point_cap_accepts_exact_and_rejects_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    point_count = max(
        len(state.calibration.calibrator.upper_bounds) for state in artifact.models.values()
    )
    assert point_count > 0
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_CALIBRATOR_POINTS",
        point_count,
    )
    assert PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict()) == artifact
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_CALIBRATOR_POINTS",
        point_count - 1,
    )
    with pytest.raises(ValueError, match="limit"):
        PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict())


def test_metadata_caps_accept_exact_and_reject_one_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    artifact = fixture.artifact
    total = prepared_artifacts._metadata_bytes_for_artifact(
        artifact.feature_schema,
        artifact.model_ids,
        artifact.training_domains,
        artifact.lineage,
    )
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_METADATA_TOTAL_BYTES",
        total,
    )
    assert PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict()) == artifact
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_METADATA_TOTAL_BYTES",
        total - 1,
    )
    with pytest.raises(ValueError, match="aggregate limit"):
        PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict())

    source = fixture.sources[0]
    monkeypatch.setattr(prepared_artifacts, "MAX_PREDICTOR_METADATA_TEXT_BYTES", 1)
    assert replace(source, held_out_domain="a").held_out_domain == "a"
    with pytest.raises(ValueError, match="UTF-8 byte limit"):
        replace(source, held_out_domain="aa")


def test_document_cap_accepts_exact_and_rejects_before_parse_or_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    document = artifact.to_json()
    byte_count = len(document.encode("utf-8"))
    monkeypatch.setattr(prepared_artifacts, "MAX_PREDICTOR_ARTIFACT_BYTES", byte_count)
    assert PreparedBilinearPredictorArtifact.from_json(document) == artifact

    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        byte_count - 1,
    )

    def no_parse(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("oversized document reached JSON parsing")

    def no_stage(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("oversized document reached stage creation")

    monkeypatch.setattr(prepared_artifacts.json, "loads", no_parse)
    monkeypatch.setattr(prepared_artifacts.tempfile, "mkstemp", no_stage)
    with pytest.raises(ValueError, match="exceeds"):
        PreparedBilinearPredictorArtifact.from_json(document)
    with pytest.raises(ValueError, match="exceeds"):
        artifact.save(destination)
    assert destination.read_bytes() == b"old"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stat_copy(
    details: os.stat_result,
    **changes: int,
) -> SimpleNamespace:
    values = {
        "st_mode": details.st_mode,
        "st_dev": details.st_dev,
        "st_ino": details.st_ino,
        "st_size": details.st_size,
        "st_mtime_ns": details.st_mtime_ns,
        "st_ctime_ns": details.st_ctime_ns,
        "st_file_attributes": getattr(details, "st_file_attributes", 0),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_load_rejects_inode_mismatch_between_path_and_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fixture().artifact.save(tmp_path / "artifact.json")
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(source.read_bytes())
    alternate_stat = alternate.stat()
    monkeypatch.setattr(
        prepared_artifacts.os,
        "fstat",
        lambda descriptor: alternate_stat,
    )
    with pytest.raises(ValueError, match="changed while opening"):
        PreparedBilinearPredictorArtifact.load(
            source,
            expected_artifact_sha256=_digest(source),
        )


@pytest.mark.parametrize("changed_field", ("size", "mtime"))
def test_load_rejects_descriptor_size_or_mtime_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
) -> None:
    source = _fixture().artifact.save(tmp_path / "artifact.json")
    real_fstat = prepared_artifacts.os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal calls
        calls += 1
        details = real_fstat(descriptor)
        if calls != 2:
            return details
        if changed_field == "size":
            return _stat_copy(details, st_size=details.st_size + 1)
        return _stat_copy(details, st_mtime_ns=details.st_mtime_ns + 1)

    monkeypatch.setattr(prepared_artifacts.os, "fstat", changing_fstat)
    with pytest.raises(ValueError, match="changed while reading"):
        PreparedBilinearPredictorArtifact.load(
            source,
            expected_artifact_sha256=_digest(source),
        )


@pytest.mark.parametrize("change", ("grow", "truncate"))
def test_load_rejects_descriptor_growth_or_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    source = _fixture().artifact.save(tmp_path / "artifact.json")
    payload = source.read_bytes()
    chunks = iter((payload + b"x" if change == "grow" else payload[:-1], b""))
    monkeypatch.setattr(prepared_artifacts.os, "read", lambda descriptor, size: next(chunks))
    with pytest.raises(ValueError, match="changed while reading"):
        PreparedBilinearPredictorArtifact.load(
            source,
            expected_artifact_sha256=_digest(source),
        )


def test_load_rejects_final_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fixture().artifact.save(tmp_path / "artifact.json")
    expected_sha256 = _digest(source)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(source.read_bytes())
    real_lstat = Path.lstat
    source_lstats = 0

    def swapping_lstat(path: Path) -> os.stat_result:
        nonlocal source_lstats
        if path == source:
            source_lstats += 1
            if source_lstats == 2:
                return real_lstat(replacement)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", swapping_lstat)
    with pytest.raises(ValueError, match="path changed while reading"):
        PreparedBilinearPredictorArtifact.load(
            source,
            expected_artifact_sha256=expected_sha256,
        )


def test_load_rejects_symlink_fifo_device_and_directory(
    tmp_path: Path,
) -> None:
    artifact = _fixture().artifact
    source = artifact.save(tmp_path / "artifact.json")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        PreparedBilinearPredictorArtifact.load(
            directory,
            expected_artifact_sha256="0" * 64,
        )

    link = tmp_path / "link.json"
    try:
        link.symlink_to(source)
    except (NotImplementedError, OSError):
        pass
    else:
        with pytest.raises(ValueError, match="symlink"):
            PreparedBilinearPredictorArtifact.load(
                link,
                expected_artifact_sha256="0" * 64,
            )

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "artifact.fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular file"):
            PreparedBilinearPredictorArtifact.load(
                fifo,
                expected_artifact_sha256="0" * 64,
            )

    device = Path("/dev/null")
    if device.exists():
        with pytest.raises(ValueError, match="regular file"):
            PreparedBilinearPredictorArtifact.load(
                device,
                expected_artifact_sha256="0" * 64,
            )


def test_save_rejects_destination_swap_and_leaves_attacker_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"old")
    real_write_all = prepared_artifacts._write_all

    def swap_destination(descriptor: int, payload: bytes) -> None:
        real_write_all(descriptor, payload)
        destination.unlink()
        destination.write_bytes(b"attacker")

    monkeypatch.setattr(prepared_artifacts, "_write_all", swap_destination)
    with pytest.raises(ValueError, match="destination changed"):
        artifact.save(destination)
    assert destination.read_bytes() == b"attacker"
    assert not tuple(tmp_path.glob(".artifact.json.stage.*.tmp"))


@pytest.mark.parametrize("replacement_kind", ("regular", "symlink", "directory"))
def test_save_rejects_stage_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    if replacement_kind == "symlink" and not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    artifact = _fixture().artifact
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"old")
    real_write_all = prepared_artifacts._write_all

    def swap_stage(descriptor: int, payload: bytes) -> None:
        real_write_all(descriptor, payload)
        stages = tuple(tmp_path.glob(".artifact.json.stage.*.tmp"))
        assert len(stages) == 1
        stage = stages[0]
        stage.unlink()
        if replacement_kind == "regular":
            stage.write_bytes(payload)
        elif replacement_kind == "symlink":
            try:
                stage.symlink_to(destination)
            except (NotImplementedError, OSError):
                pytest.skip("symlinks are unavailable")
        else:
            stage.mkdir()

    monkeypatch.setattr(prepared_artifacts, "_write_all", swap_stage)
    expected = "stage changed" if replacement_kind == "regular" else "stage"
    with pytest.raises(ValueError, match=expected):
        artifact.save(destination)
    assert destination.read_bytes() == b"old"


def test_save_rejects_nonregular_destination_and_symlink_parent(
    tmp_path: Path,
) -> None:
    artifact = _fixture().artifact
    directory = tmp_path / "destination"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        artifact.save(directory)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "destination.fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular file"):
            artifact.save(fifo)

    target_parent = tmp_path / "real-parent"
    target_parent.mkdir()
    link_parent = tmp_path / "link-parent"
    try:
        link_parent.symlink_to(target_parent, target_is_directory=True)
    except (NotImplementedError, OSError):
        return
    with pytest.raises(ValueError, match="directory must not be a symlink"):
        artifact.save(link_parent / "artifact.json")


def test_save_validation_and_staging_failures_preserve_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"old")
    object.__setattr__(fixture.artifact.lineage, "final_coefficient_sha256", "0" * 64)

    def no_stage(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("invalid artifact reached stage creation")

    monkeypatch.setattr(prepared_artifacts.tempfile, "mkstemp", no_stage)
    with pytest.raises(ValueError, match="final_coefficient_sha256"):
        fixture.artifact.save(destination)
    assert destination.read_bytes() == b"old"

    monkeypatch.undo()

    def fail_write(descriptor: int, payload: bytes) -> NoReturn:
        del descriptor, payload
        raise OSError("injected stage failure")

    monkeypatch.setattr(prepared_artifacts, "_write_all", fail_write)
    with pytest.raises(OSError, match="injected stage failure"):
        _fixture().artifact.save(destination)
    assert destination.read_bytes() == b"old"
    assert not tuple(tmp_path.glob(".artifact.json.stage.*.tmp"))


def test_save_rejects_mutated_calibrator_before_stage_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    destination = tmp_path / "artifact.json"
    destination.write_bytes(b"old")
    object.__setattr__(
        artifact.models["m1"].calibration.calibrator,
        "upper_bounds",
        _PoisonIterable(),
    )

    def no_stage(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("mutated artifact reached stage creation")

    monkeypatch.setattr(prepared_artifacts.tempfile, "mkstemp", no_stage)
    with pytest.raises(TypeError, match="exact tuples"):
        artifact.save(destination)
    assert destination.read_bytes() == b"old"


def test_canonical_serializer_streams_without_json_dumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    expected = artifact.to_json()

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("prepared artifact used unbounded json.dumps")

    monkeypatch.setattr(prepared_artifacts.json, "dumps", forbidden)
    assert artifact.to_json() == expected


class _PoisonProvider:
    @property
    def dimension(self) -> NoReturn:
        raise AssertionError("surface artifact read provider dimension")

    @property
    def identity(self) -> NoReturn:
        raise AssertionError("surface artifact read provider identity")

    def embed(self, prompts: object) -> NoReturn:
        del prompts
        raise AssertionError("surface artifact called provider embed")


@pytest.mark.parametrize("mutation", ["weights", "calibrator"])
def test_build_predictor_resnapshots_nested_state_before_provider_access(
    mutation: str,
) -> None:
    artifact = _fixture(embedded=True).artifact
    state = artifact.models["m1"]
    if mutation == "weights":
        object.__setattr__(state, "weights", _PoisonIterable())
        expected = "weight width"
    else:
        object.__setattr__(state.calibration.calibrator, "values", _PoisonIterable())
        expected = "exact tuples"

    with pytest.raises((TypeError, ValueError), match=expected):
        artifact.build_predictor(embedding_provider=_PoisonProvider())


class _Provider:
    def __init__(self, dimension: int, identity: EmbeddingIdentity) -> None:
        self._dimension = dimension
        self._identity = identity
        self.embed_calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, prompts: object) -> NoReturn:
        del prompts
        self.embed_calls += 1
        raise RuntimeError("deferred embed")


def test_surface_predictor_rejects_provider_without_reading_its_properties() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _fixture().artifact.build_predictor(embedding_provider=_PoisonProvider())


def test_embedded_predictor_requires_exact_provider_and_defers_embed() -> None:
    artifact = _fixture(embedded=True).artifact
    identity = artifact.feature_schema.embedding_identity
    assert identity is not None
    with pytest.raises(ValueError, match="required"):
        artifact.build_predictor()
    with pytest.raises(ValueError, match="dimension"):
        artifact.build_predictor(
            embedding_provider=_Provider(artifact.feature_schema.embedding_dimension + 1, identity)
        )
    wrong_identity = replace(identity, revision="other-revision")
    with pytest.raises(ValueError, match="identity"):
        artifact.build_predictor(
            embedding_provider=_Provider(
                artifact.feature_schema.embedding_dimension,
                wrong_identity,
            )
        )

    provider = _Provider(artifact.feature_schema.embedding_dimension, identity)
    predictor = artifact.build_predictor(embedding_provider=provider)
    assert provider.embed_calls == 0
    with pytest.raises(RuntimeError, match="deferred embed"):
        predictor.predict("prompt", artifact.model_ids[0])
    assert provider.embed_calls == 1


def test_target_shard_hash_binds_major_fields_order_and_payload() -> None:
    plan = build_prepared_nested_lodo_plan(
        ("a", "b", "c", "d"),
        (2, 1, 1, 1),
        feature_count=prepared_artifacts._UNIVERSAL_SURFACE_DIMENSION,
        target_count=2,
    )

    def make(
        *,
        selected_plan: PreparedNestedLodoPlan = plan,
        store_sha256: str = "1" * 64,
        domain_index: int = 0,
        model_ids: tuple[str, ...] = ("m1", "m2"),
        scored_sha256: str = "2" * 64,
        example_ids: tuple[str, ...] = ("e1", "e2"),
        prompt_sha256s: tuple[str, ...] = ("3" * 64, "4" * 64),
        values: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    ) -> PreparedPredictorTargetShard:
        return PreparedPredictorTargetShard(
            plan=selected_plan,
            store_sha256=store_sha256,
            domain_index=domain_index,
            model_ids=model_ids,
            scored_feature_shard_sha256=scored_sha256,
            example_ids=example_ids,
            prompt_sha256s=prompt_sha256s,
            targets_payload=struct.pack(f"<{len(values)}d", *values),
        )

    baseline = make()
    changed_plan = build_prepared_nested_lodo_plan(
        ("a", "b", "c", "e"),
        (2, 1, 1, 1),
        feature_count=plan.feature_count,
        target_count=2,
    )
    variants = (
        make(selected_plan=changed_plan),
        make(store_sha256="9" * 64),
        make(
            domain_index=1,
            example_ids=("e3",),
            prompt_sha256s=("5" * 64,),
            values=(0.0, 0.1),
        ),
        make(model_ids=("m1", "m3")),
        make(scored_sha256="8" * 64),
        make(example_ids=("f1", "f2")),
        make(prompt_sha256s=("6" * 64, "7" * 64)),
        make(values=(0.2, 0.3, 0.0, 0.1)),
        make(values=(0.0, 0.1, 0.2, 0.4)),
    )
    assert all(variant.sha256 != baseline.sha256 for variant in variants)
    with pytest.raises(ValueError, match="strictly increasing"):
        make(example_ids=("e2", "e1"))
    with pytest.raises(ValueError, match="sorted"):
        make(model_ids=("m2", "m1"))

    assert make(values=(+0.0, 0.1, 0.2, 0.3)).sha256 == baseline.sha256
    with pytest.raises(ValueError, match="positive zero"):
        make(values=(-0.0, 0.1, 0.2, 0.3))


def test_calibration_hashes_bind_sources_pairs_and_calibrator_state() -> None:
    sources = _fixture().sources
    pairs = ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0))
    baseline = prepared_calibration_input_sha256("m1", sources, pairs)
    source_variants = (
        (*sources[:-1], replace(sources[-1], held_out_domain="e")),
        (
            *sources[:-1],
            replace(
                sources[-1],
                training_subset_index=sources[-1].training_subset_index + 1,
            ),
        ),
        (
            *sources[:-1],
            replace(sources[-1], score_block_index=sources[-1].score_block_index + 1),
        ),
        (
            *sources[:-1],
            replace(sources[-1], raw_score_block_sha256="f" * 64),
        ),
        (
            *sources[:-1],
            replace(sources[-1], scored_feature_shard_sha256="e" * 64),
        ),
        (
            *sources[:-1],
            replace(sources[-1], target_shard_sha256="d" * 64),
        ),
    )
    hashes = (
        prepared_calibration_input_sha256("m2", sources, pairs),
        *(
            prepared_calibration_input_sha256("m1", source_variant, pairs)
            for source_variant in source_variants
        ),
        prepared_calibration_input_sha256("m1", sources, tuple(reversed(pairs))),
        prepared_calibration_input_sha256(
            "m1",
            sources,
            (*pairs[:-1], (3.0, 0.5)),
        ),
    )
    assert all(value != baseline for value in hashes)
    with pytest.raises(ValueError, match="ascending"):
        prepared_calibration_input_sha256("m1", tuple(reversed(sources)), pairs)
    assert (
        prepared_calibration_input_sha256(
            "m1",
            sources,
            ((-0.0, -0.0), *pairs[1:]),
        )
        == baseline
    )

    calibrator = IsotonicCalibrator((0.0, 1.0), (0.0, 1.0))
    calibration = PreparedModelCalibration(
        model_id="m1",
        sources=sources,
        input_sha256=baseline,
        calibrator=calibrator,
    )
    changed_input = PreparedModelCalibration(
        model_id="m1",
        sources=sources,
        input_sha256="f" * 64,
        calibrator=calibrator,
    )
    changed_bounds = PreparedModelCalibration(
        model_id="m1",
        sources=sources,
        input_sha256=baseline,
        calibrator=IsotonicCalibrator((0.0, 2.0), (0.0, 1.0)),
    )
    changed_values = PreparedModelCalibration(
        model_id="m1",
        sources=sources,
        input_sha256=baseline,
        calibrator=IsotonicCalibrator((0.0, 1.0), (0.0, 0.75)),
    )
    assert (
        len(
            {
                calibration.identity_sha256,
                changed_input.identity_sha256,
                changed_bounds.identity_sha256,
                changed_values.identity_sha256,
            }
        )
        == 4
    )


def test_aggregate_hash_binds_major_fields_and_order() -> None:
    aggregate = _fixture().aggregate
    changed_plan = build_prepared_nested_lodo_plan(
        ("a", "b", "c", "e"),
        aggregate.plan.domain_example_counts,
        feature_count=aggregate.plan.feature_count,
        target_count=aggregate.plan.target_count,
    )
    feature_means = list(aggregate.feature_means)
    feature_means[0] += 0.25
    changed_mask_means = list(aggregate.feature_means)
    changed_mask_means[7] = 0.0
    xx = list(aggregate.centered_xx_packed)
    xx[_packed_upper_index(aggregate.plan.feature_count, 0, 0)] += 1.0
    xy = list(aggregate.centered_xy)
    xy[0] += 0.25
    variants = (
        replace(aggregate, plan=changed_plan),
        replace(aggregate, store_sha256="9" * 64),
        replace(aggregate, statistics_bundle_sha256="8" * 64),
        replace(aggregate, model_ids=("m1", "m3")),
        replace(
            aggregate,
            domain_statistics_sha256s=tuple(reversed(aggregate.domain_statistics_sha256s)),
        ),
        replace(
            aggregate,
            active_tag_mask=0b001,
            feature_means=tuple(changed_mask_means),
        ),
        replace(aggregate, feature_means=tuple(feature_means)),
        replace(aggregate, target_means=(0.5, 0.6)),
        replace(aggregate, centered_xx_packed=tuple(xx)),
        replace(aggregate, centered_xy=tuple(xy)),
    )
    assert all(variant.sha256 != aggregate.sha256 for variant in variants)
    with pytest.raises(ValueError, match="sorted"):
        replace(aggregate, model_ids=tuple(reversed(aggregate.model_ids)))
    zero_means = tuple(
        -0.0 if index == 8 else value for index, value in enumerate(aggregate.feature_means)
    )
    assert replace(aggregate, feature_means=zero_means).sha256 == aggregate.sha256

    embedded = _fixture(embedded=True).aggregate
    identity = embedded.embedding_identity
    assert identity is not None
    assert (
        replace(
            embedded,
            embedding_identity=replace(identity, revision="different-revision"),
        ).sha256
        != embedded.sha256
    )


def test_coefficient_hash_binds_major_fields_order_and_binary_payload() -> None:
    coefficient = _fixture().coefficient
    changed_schema = replace(
        coefficient.feature_schema,
        continuous_means=(
            coefficient.feature_schema.continuous_means[0] + 0.25,
            *coefficient.feature_schema.continuous_means[1:],
        ),
    )
    weights = list(
        struct.unpack(
            f"<{len(coefficient.weights_payload) // 8}d",
            coefficient.weights_payload,
        )
    )
    weights[0] = 0.125
    variants = (
        replace(coefficient, aggregate_statistics_sha256="f" * 64),
        replace(coefficient, ridge=2.0),
        replace(coefficient, feature_schema=changed_schema),
        replace(coefficient, model_ids=("m1", "m3")),
        replace(
            coefficient,
            weights_payload=struct.pack(f"<{len(weights)}d", *weights),
        ),
        replace(coefficient, intercepts_payload=struct.pack("<2d", 0.1, 0.25)),
    )
    assert all(variant.sha256 != coefficient.sha256 for variant in variants)
    with pytest.raises(ValueError, match="sorted"):
        replace(coefficient, model_ids=tuple(reversed(coefficient.model_ids)))
    with pytest.raises(ValueError, match="active_feature_indices"):
        replace(
            coefficient,
            active_feature_indices=tuple(reversed(coefficient.active_feature_indices)),
        )

    positive_zero = bytearray(coefficient.weights_payload)
    positive_zero[:8] = struct.pack("<d", +0.0)
    assert replace(coefficient, weights_payload=bytes(positive_zero)).sha256 == coefficient.sha256
    negative_zero = bytearray(coefficient.weights_payload)
    negative_zero[:8] = struct.pack("<d", -0.0)
    with pytest.raises(ValueError, match="positive zero"):
        replace(coefficient, weights_payload=bytes(negative_zero))


def test_non_provider_entry_points_have_no_network_parameters_or_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {
        "provider",
        "embedding_provider",
        "client",
        "session",
        "url",
        "download",
    }
    entry_points: tuple[Callable[..., object], ...] = (
        prepared_assembly.assemble_prepared_bilinear_artifact,
        PreparedBilinearPredictorArtifact.from_dict,
        PreparedBilinearPredictorArtifact.from_json,
        PreparedBilinearPredictorArtifact.load,
        PreparedBilinearPredictorArtifact.save,
    )
    for entry_point in entry_points:
        assert forbidden.isdisjoint(inspect.signature(entry_point).parameters)

    def no_socket(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("offline artifact path opened a socket")

    def no_download(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("offline artifact path attempted a download")

    monkeypatch.setattr(socket, "socket", no_socket)
    monkeypatch.setattr(urllib.request, "urlopen", no_download)
    artifact = _fixture().artifact
    assert PreparedBilinearPredictorArtifact.from_dict(artifact.to_dict()) == artifact
    assert PreparedBilinearPredictorArtifact.from_json(artifact.to_json()) == artifact
    destination = artifact.save(tmp_path / "artifact.json")
    assert (
        PreparedBilinearPredictorArtifact.load(
            destination,
            expected_artifact_sha256=_digest(destination),
        )
        == artifact
    )
    with pytest.raises(TypeError):
        prepared_assembly.assemble_prepared_bilinear_artifact(  # type: ignore[arg-type]
            None,
            None,
            None,
            expected_source_fit_sha256="0" * 64,
            expected_store_sha256="0" * 64,
            expected_statistics_sha256="0" * 64,
            expected_raw_score_sha256="0" * 64,
        )
