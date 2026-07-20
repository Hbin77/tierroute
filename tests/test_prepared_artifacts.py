# SPDX-License-Identifier: Apache-2.0
"""Focused tests for prepared predictor identities and pinned JSON persistence."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import pytest

import tierroute.predictors.prepared_artifacts as prepared_artifacts
from tierroute.features import EmbeddingIdentity, PromptFeatureSchema
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.prepared_artifacts import (
    PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID,
    PREPARED_PREDICTOR_ARTIFACT_KIND,
    PREPARED_PREDICTOR_ARTIFACT_VERSION,
    PreparedAllDomainStatistics,
    PreparedArtifactLineage,
    PreparedBilinearPredictorArtifact,
    PreparedCalibrationSource,
    PreparedFinalCoefficient,
    PreparedModelCalibration,
    PreparedModelState,
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
class _ArtifactFixture:
    plan: PreparedNestedLodoPlan
    aggregate: PreparedAllDomainStatistics
    coefficient: PreparedFinalCoefficient
    sources: tuple[PreparedCalibrationSource, ...]
    artifact: PreparedBilinearPredictorArtifact


def _semantic_sources(plan: PreparedNestedLodoPlan) -> tuple[PreparedCalibrationSource, ...]:
    sources = []
    hash_characters = "789a"
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
                raw_score_block_sha256=hash_characters[domain_index] * 64,
                scored_feature_shard_sha256="1234"[domain_index] * 64,
                target_shard_sha256="3456"[domain_index] * 64,
            )
        )
    return tuple(sources)


def _fixture(*, embedded: bool = False) -> _ArtifactFixture:
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
        feature_count=12 + embedding_dimension,
        target_count=2,
    )
    feature_means = [
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
    ]
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
        feature_means=tuple(feature_means),
        target_means=(0.4, 0.6),
        centered_xx_packed=tuple(centered_xx),
        centered_xy=(0.0,) * (plan.feature_count * 2),
    )
    coefficient_width = aggregate.feature_schema.dimension
    weights = tuple(index / 10.0 for index in range(2 * coefficient_width))
    active_indices = (0, 1, 2, 3, 4, 5, 7, *range(12, 12 + embedding_dimension))
    coefficient = PreparedFinalCoefficient(
        feature_schema=aggregate.feature_schema,
        active_feature_indices=active_indices,
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
            (float(row_index + model_index), float(row_index % 2)) for row_index in range(4)
        )
        calibrations[model_id] = PreparedModelCalibration(
            model_id=model_id,
            sources=sources,
            input_sha256=prepared_calibration_input_sha256(
                model_id,
                sources,
                pairs,
            ),
            calibrator=IsotonicCalibrator.fit(
                [prediction for prediction, _ in pairs],
                [target for _, target in pairs],
            ),
        )
    lineage = PreparedArtifactLineage(
        source_fit_sha256="a" * 64,
        store_sha256="1" * 64,
        statistics_bundle_sha256="2" * 64,
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
        training_example_count=4,
        lineage=lineage,
    )
    return _ArtifactFixture(plan, aggregate, coefficient, sources, artifact)


def test_prepared_artifact_has_frozen_canonical_json_and_round_trips() -> None:
    artifact = _fixture().artifact
    document = artifact.to_json()

    assert artifact.artifact_kind == PREPARED_PREDICTOR_ARTIFACT_KIND
    assert artifact.artifact_version == PREPARED_PREDICTOR_ARTIFACT_VERSION
    assert artifact.algorithm_id == PREPARED_PREDICTOR_ARTIFACT_ALGORITHM_ID
    assert document.endswith("\n") and not document.endswith("\n\n")
    assert hashlib.sha256(document.encode("utf-8")).hexdigest() == (
        "5b0af2cc5d48b877ca4e3e6063c0b12aa69dc65110b6356a475e72c8ab99b8aa"
    )
    assert PreparedBilinearPredictorArtifact.from_json(document) == artifact
    assert PreparedBilinearPredictorArtifact.from_json(document).to_json() == document
    payload = json.loads(document)
    assert set(payload) == {
        "algorithm_id",
        "artifact_kind",
        "artifact_version",
        "feature_schema",
        "models",
        "training",
        "lineage",
    }
    assert list(payload["models"]) == ["m1", "m2"]


def test_final_coefficient_and_calibrator_children_recompute_on_parse() -> None:
    artifact = _fixture().artifact
    payload = artifact.to_dict()
    payload["lineage"]["final_coefficient_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="final_coefficient_sha256"):
        PreparedBilinearPredictorArtifact.from_dict(payload)

    payload = artifact.to_dict()
    payload["models"]["m1"]["calibrator"]["identity_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="identity_sha256"):
        PreparedBilinearPredictorArtifact.from_dict(payload)


def test_target_shard_hashes_constructor_only_payload_without_retaining_it() -> None:
    plan = _fixture().plan
    first = PreparedPredictorTargetShard(
        plan=plan,
        store_sha256="1" * 64,
        domain_index=0,
        model_ids=("m1", "m2"),
        scored_feature_shard_sha256="2" * 64,
        example_ids=("e1",),
        prompt_sha256s=("3" * 64,),
        targets_payload=struct.pack("<2d", 0.1, 0.2),
    )
    second = PreparedPredictorTargetShard(
        plan=plan,
        store_sha256="1" * 64,
        domain_index=0,
        model_ids=("m1", "m2"),
        scored_feature_shard_sha256="2" * 64,
        example_ids=("e1",),
        prompt_sha256s=("3" * 64,),
        targets_payload=struct.pack("<2d", 0.1, 0.3),
    )

    assert first.sha256 == "9f7d4476eb0d750b73b23af369f8d38374515322db18f71c0099cbe849a393b6"
    assert first.sha256 != second.sha256
    assert not hasattr(first, "targets_payload")


def test_target_shard_rejects_stale_plan_and_snapshots_canonical_plan() -> None:
    fixture = _fixture()
    plan = fixture.plan
    object.__setattr__(
        plan,
        "training_subsets",
        tuple(reversed(plan.training_subsets)),
    )
    with pytest.raises(ValueError, match="plan must be canonical"):
        PreparedPredictorTargetShard(
            plan=plan,
            store_sha256="1" * 64,
            domain_index=0,
            model_ids=("m1", "m2"),
            scored_feature_shard_sha256="2" * 64,
            example_ids=("e1",),
            prompt_sha256s=("3" * 64,),
            targets_payload=struct.pack("<2d", 0.1, 0.2),
        )

    fresh = _fixture()
    assert fresh.aggregate.plan is not fresh.plan
    original_domains = fresh.aggregate.plan.domains
    object.__setattr__(fresh.plan, "domains", tuple(reversed(fresh.plan.domains)))
    assert fresh.aggregate.plan.domains == original_domains


def test_model_state_rejects_one_over_feature_cap_before_numeric_copy() -> None:
    calibration = _fixture().artifact.models["m1"].calibration
    with pytest.raises(ValueError, match="feature limit"):
        PreparedModelState(
            weights=(0.0,) * (prepared_artifacts.MAX_PREPARED_FEATURES + 1),
            bias=0.0,
            calibration=calibration,
        )


def test_final_coefficient_uses_universal_prepared_width_cap() -> None:
    identity = EmbeddingIdentity(
        provider="fixture-local",
        model_id="fixture/model",
        revision="frozen-revision",
        pooling="mean",
        normalize=True,
        asset_manifest_sha256="e" * 64,
    )
    exact_embedding_dimension = (
        prepared_artifacts.MAX_PREPARED_FEATURES - prepared_artifacts._UNIVERSAL_SURFACE_DIMENSION
    )
    exact_schema = PromptFeatureSchema(
        continuous_means=(0.0, 0.0, 0.0),
        continuous_scales=(1.0, 1.0, 1.0),
        domain_tags=(),
        embedding_dimension=exact_embedding_dimension,
        embedding_identity=identity,
    )
    exact_active = prepared_artifacts._active_feature_indices(exact_schema)
    coefficient = PreparedFinalCoefficient(
        feature_schema=exact_schema,
        active_feature_indices=exact_active,
        model_ids=("m1",),
        aggregate_statistics_sha256="1" * 64,
        ridge=1.0,
        weights_payload=struct.pack(
            f"<{exact_schema.dimension}d",
            *((0.0,) * exact_schema.dimension),
        ),
        intercepts_payload=struct.pack("<d", 0.0),
    )
    assert exact_active[-1] == prepared_artifacts.MAX_PREPARED_FEATURES - 1
    assert coefficient.feature_schema == exact_schema

    oversized_schema = PromptFeatureSchema(
        continuous_means=(0.0, 0.0, 0.0),
        continuous_scales=(1.0, 1.0, 1.0),
        domain_tags=(),
        embedding_dimension=exact_embedding_dimension + 1,
        embedding_identity=identity,
    )
    with pytest.raises(ValueError, match="universal feature limit"):
        PreparedFinalCoefficient(
            feature_schema=oversized_schema,
            active_feature_indices=prepared_artifacts._active_feature_indices(oversized_schema),
            model_ids=("m1",),
            aggregate_statistics_sha256="1" * 64,
            ridge=1.0,
            weights_payload=struct.pack(
                f"<{oversized_schema.dimension}d",
                *((0.0,) * oversized_schema.dimension),
            ),
            intercepts_payload=struct.pack("<d", 0.0),
        )


def test_final_coefficient_rejects_bool_active_coordinate() -> None:
    coefficient = _fixture().coefficient
    with pytest.raises(TypeError, match="exact integers"):
        PreparedFinalCoefficient(
            feature_schema=coefficient.feature_schema,
            active_feature_indices=(
                False,
                *coefficient.active_feature_indices[1:],
            ),
            model_ids=coefficient.model_ids,
            aggregate_statistics_sha256=coefficient.aggregate_statistics_sha256,
            ridge=coefficient.ridge,
            weights_payload=coefficient.weights_payload,
            intercepts_payload=coefficient.intercepts_payload,
        )


def test_calibration_input_hash_streams_exact_declared_rows() -> None:
    sources = _fixture().sources
    pairs = ((float(index), float(index % 2)) for index in range(4))
    expected = prepared_calibration_input_sha256("m1", sources, pairs)
    assert expected == prepared_calibration_input_sha256(
        "m1",
        sources,
        tuple((float(index), float(index % 2)) for index in range(4)),
    )
    with pytest.raises(ValueError, match="ended before"):
        prepared_calibration_input_sha256("m1", sources, ((0.0, 0.0),))
    with pytest.raises(ValueError, match="exceeds"):
        prepared_calibration_input_sha256(
            "m1",
            sources,
            tuple((float(index), 0.0) for index in range(5)),
        )


def test_pinned_descriptor_load_and_exact_canonical_bytes(tmp_path: Path) -> None:
    artifact = _fixture().artifact
    destination = artifact.save(tmp_path / "prepared.json")
    document = artifact.to_json()
    digest = hashlib.sha256(document.encode("utf-8")).hexdigest()

    loaded = PreparedBilinearPredictorArtifact.load(
        destination,
        expected_artifact_sha256=digest,
    )
    assert loaded == artifact
    with pytest.raises(ValueError, match="trusted pin"):
        PreparedBilinearPredictorArtifact.load(
            destination,
            expected_artifact_sha256="0" * 64,
        )

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(document + "\n", encoding="utf-8", newline="")
    noncanonical_digest = hashlib.sha256(noncanonical.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exact canonical JSON"):
        PreparedBilinearPredictorArtifact.load(
            noncanonical,
            expected_artifact_sha256=noncanonical_digest,
        )


def test_load_rejects_symlink_and_fifo_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    source = artifact.save(tmp_path / "source.json")
    symlink = tmp_path / "link.json"
    try:
        symlink.symlink_to(source)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    real_open = prepared_artifacts.os.open

    def guarded_open(path: object, flags: int, *args: object) -> int:
        if Path(path) == symlink:
            raise AssertionError("symlink reached os.open")
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(prepared_artifacts.os, "open", guarded_open)
    with pytest.raises(ValueError, match="symlink"):
        PreparedBilinearPredictorArtifact.load(
            symlink,
            expected_artifact_sha256="0" * 64,
        )

    if not hasattr(os, "mkfifo"):
        return
    fifo = tmp_path / "artifact.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        PreparedBilinearPredictorArtifact.load(
            fifo,
            expected_artifact_sha256="0" * 64,
        )


def test_save_uses_one_stage_without_a_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    destination = tmp_path / "prepared.json"
    destination.write_text("old", encoding="utf-8", newline="")
    calls = 0
    real_mkstemp = prepared_artifacts.tempfile.mkstemp

    def one_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal calls
        calls += 1
        return real_mkstemp(*args, **kwargs)

    def no_backup(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("single-document save must not create a backup")

    monkeypatch.setattr(prepared_artifacts.tempfile, "mkstemp", one_mkstemp)
    monkeypatch.setattr(prepared_artifacts.os, "link", no_backup)
    assert artifact.save(destination) == destination
    assert calls == 1
    assert destination.read_text(encoding="utf-8") == artifact.to_json()
    assert not tuple(tmp_path.glob(".prepared.json.*.tmp"))


def test_validation_and_staging_failure_leave_existing_destination_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    destination = tmp_path / "prepared.json"
    destination.write_bytes(b"old")
    object.__setattr__(
        fixture.artifact.lineage,
        "final_coefficient_sha256",
        "0" * 64,
    )

    def no_stage(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("invalid state reached stage creation")

    monkeypatch.setattr(prepared_artifacts.tempfile, "mkstemp", no_stage)
    with pytest.raises(ValueError, match="final_coefficient_sha256"):
        fixture.artifact.save(destination)
    assert destination.read_bytes() == b"old"

    valid = _fixture().artifact

    def fail_stage(descriptor: int, payload: bytes) -> NoReturn:
        del descriptor, payload
        raise OSError("injected staging failure")

    monkeypatch.undo()
    monkeypatch.setattr(prepared_artifacts, "_write_all", fail_stage)
    with pytest.raises(OSError, match="injected staging failure"):
        valid.save(destination)
    assert destination.read_bytes() == b"old"
    assert not tuple(tmp_path.glob(".prepared.json.*.tmp"))


def test_document_byte_limit_applies_before_parse_or_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _fixture().artifact
    document = artifact.to_json()
    exact_bytes = len(document.encode("utf-8"))
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        exact_bytes,
    )
    assert PreparedBilinearPredictorArtifact.from_json(document) == artifact

    destination = tmp_path / "prepared.json"
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        prepared_artifacts,
        "MAX_PREDICTOR_ARTIFACT_BYTES",
        exact_bytes - 1,
    )

    def no_stage(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("oversized document reached stage creation")

    monkeypatch.setattr(prepared_artifacts.tempfile, "mkstemp", no_stage)
    with pytest.raises(ValueError, match="exceeds"):
        artifact.to_json()
    with pytest.raises(ValueError, match="exceeds"):
        PreparedBilinearPredictorArtifact.from_json(document)
    with pytest.raises(ValueError, match="exceeds"):
        artifact.save(destination)
    assert destination.read_bytes() == b"old"


class _RecordingEmbeddingProvider:
    def __init__(self, identity: EmbeddingIdentity) -> None:
        self._identity = identity
        self.embed_calls = 0

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, prompts: object) -> NoReturn:
        del prompts
        self.embed_calls += 1
        raise RuntimeError("prediction reached the provider")


def test_build_predictor_only_snapshots_provider_until_prediction() -> None:
    artifact = _fixture(embedded=True).artifact
    identity = artifact.feature_schema.embedding_identity
    assert identity is not None
    provider = _RecordingEmbeddingProvider(identity)

    predictor = artifact.build_predictor(embedding_provider=provider)
    assert provider.embed_calls == 0
    with pytest.raises(RuntimeError, match="reached the provider"):
        predictor.predict("general prompt", "m1")
    assert provider.embed_calls == 1

    with pytest.raises(ValueError, match="not allowed"):
        _fixture().artifact.build_predictor(embedding_provider=provider)


@pytest.mark.parametrize(
    "document",
    (
        '{"artifact_version":1,"artifact_version":1}',
        '{"artifact_version":NaN}',
        "[]",
    ),
)
def test_prepared_json_rejects_non_strict_or_wrong_root(document: str) -> None:
    with pytest.raises(ValueError):
        PreparedBilinearPredictorArtifact.from_json(document)
