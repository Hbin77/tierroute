# SPDX-License-Identifier: Apache-2.0
"""Canonical, fail-closed JSON artifacts for calibrated GBM predictors."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import tierroute.predictors.artifacts as _shared_artifacts
from tierroute.core.atomic_io import AtomicTextWrite, replace_text_bundle
from tierroute.features import EmbeddingProvider, PromptFeatureEncoder, PromptFeatureSchema
from tierroute.predictors.calibration import (
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
)
from tierroute.predictors.gbm import GbmModel, GbmQualityPredictor, RegressionStump
from tierroute.predictors.gbm_training import (
    GBM_ALGORITHM_ID,
    MAX_GBM_TOTAL_STUMPS,
    GbmTrainingConfig,
)
from tierroute.predictors.resource_limits import (
    MAX_PREDICTOR_CALIBRATOR_POINTS,
    MAX_PREDICTOR_METADATA_TOTAL_BYTES,
    MAX_PREDICTOR_NUMERIC_SCALARS,
    MAX_PREDICTOR_TRAINING_DOMAINS,
)

GBM_PREDICTOR_ARTIFACT_KIND = "tierroute-gbm-predictor"
GBM_PREDICTOR_ARTIFACT_VERSION = 1
# The model cap keeps simultaneous model/domain/tag maxima below the shared JSON
# string-token budget. Stumps use fixed four-number arrays, so the trainer's complete
# aggregate stump contract remains serializable without repeating field-name strings.
MAX_GBM_ARTIFACT_MODELS = 256
MAX_GBM_ARTIFACT_TOTAL_STUMPS = MAX_GBM_TOTAL_STUMPS

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FIXED_NUMERIC_SCALARS = 14


def _normalized_training_hash(value: object) -> str:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("training_data_sha256 must be lowercase SHA-256 hex")
    return value


def _metadata_size(
    *,
    feature_schema: PromptFeatureSchema,
    model_ids: tuple[str, ...],
    training_domains: tuple[str, ...],
) -> int:
    total = _shared_artifacts._metadata_bytes(
        GBM_PREDICTOR_ARTIFACT_KIND,
        "artifact_kind",
    )
    total += _shared_artifacts._metadata_bytes(GBM_ALGORITHM_ID, "algorithm_id")
    for model_id in model_ids:
        # The canonical document repeats each ID in models and calibrators.
        total += 2 * _shared_artifacts._metadata_bytes(model_id, "artifact model ID")
    for domain in training_domains:
        total += _shared_artifacts._metadata_bytes(domain, "training domain")
    for tag in feature_schema.domain_tags:
        total += _shared_artifacts._metadata_bytes(tag, "feature domain tag")
    identity = feature_schema.embedding_identity
    if identity is not None:
        for name in ("provider", "model_id", "revision", "pooling"):
            total += _shared_artifacts._metadata_bytes(
                getattr(identity, name),
                f"embedding {name}",
            )
        total += len(identity.asset_manifest_sha256)
    return total


def _stump_payload(
    value: object,
    *,
    context: str,
    config: GbmTrainingConfig,
    running_total: int,
) -> tuple[tuple[RegressionStump, ...], int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be an array")
    stumps: list[RegressionStump] = []
    try:
        iterator = iter(value)
        for raw_stump in iterator:
            if len(stumps) >= config.n_estimators:
                raise ValueError(f"{context} exceeds training.config.n_estimators")
            running_total += 1
            if running_total > MAX_GBM_ARTIFACT_TOTAL_STUMPS:
                raise ValueError(
                    "GBM artifact exceeds the aggregate stump limit "
                    f"({MAX_GBM_ARTIFACT_TOTAL_STUMPS:,})"
                )
            if not isinstance(raw_stump, (list, tuple)):
                raise ValueError(f"{context}[{len(stumps)}] must be a four-number array")
            parts: list[object] = []
            try:
                for part in raw_stump:
                    if len(parts) >= 4:
                        raise ValueError(
                            f"{context}[{len(stumps)}] must contain exactly four numbers"
                        )
                    parts.append(part)
            except RuntimeError as error:
                raise ValueError(
                    f"{context}[{len(stumps)}] could not be read deterministically"
                ) from error
            if len(parts) != 4:
                raise ValueError(f"{context}[{len(stumps)}] must contain exactly four numbers")
            stumps.append(
                RegressionStump(
                    feature_index=_shared_artifacts._bounded_integer(
                        parts[0],
                        f"{context}[{len(stumps)}].feature_index",
                    ),
                    split_value=_shared_artifacts._finite_float(
                        parts[1],
                        f"{context}[{len(stumps)}].split_value",
                    ),
                    left_value=_shared_artifacts._finite_float(
                        parts[2],
                        f"{context}[{len(stumps)}].left_value",
                    ),
                    right_value=_shared_artifacts._finite_float(
                        parts[3],
                        f"{context}[{len(stumps)}].right_value",
                    ),
                )
            )
    except RuntimeError as error:
        raise ValueError(f"{context} could not be read deterministically") from error
    return tuple(stumps), running_total


@dataclass(frozen=True, slots=True)
class GbmPredictorArtifact:
    """All state needed for deterministic, calibrated offline GBM inference."""

    feature_schema: PromptFeatureSchema
    models: Mapping[str, GbmModel]
    calibrators: Mapping[str, IsotonicCalibrator]
    training_data_sha256: str
    training_example_count: int
    training_domains: tuple[str, ...]
    training_config: GbmTrainingConfig
    algorithm_id: str = GBM_ALGORITHM_ID
    artifact_kind: str = GBM_PREDICTOR_ARTIFACT_KIND
    artifact_version: int = GBM_PREDICTOR_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.artifact_version) is not int
            or self.artifact_version != GBM_PREDICTOR_ARTIFACT_VERSION
        ):
            raise ValueError(f"artifact_version must equal {GBM_PREDICTOR_ARTIFACT_VERSION}")
        if type(self.artifact_kind) is not str or self.artifact_kind != (
            GBM_PREDICTOR_ARTIFACT_KIND
        ):
            raise ValueError(f"artifact_kind must equal {GBM_PREDICTOR_ARTIFACT_KIND!r}")
        if type(self.algorithm_id) is not str or self.algorithm_id != GBM_ALGORITHM_ID:
            raise ValueError(f"algorithm_id must equal {GBM_ALGORITHM_ID!r}")
        if type(self.feature_schema) is not PromptFeatureSchema:
            raise TypeError("feature_schema must be an exact PromptFeatureSchema")
        if type(self.training_config) is not GbmTrainingConfig:
            raise TypeError("training_config must be an exact GbmTrainingConfig")

        training_hash = _normalized_training_hash(self.training_data_sha256)
        example_count = _shared_artifacts._bounded_integer(
            self.training_example_count,
            "training_example_count",
            positive=True,
        )
        domains = _shared_artifacts._text_tuple(
            self.training_domains,
            "training_domains",
            max_items=MAX_PREDICTOR_TRAINING_DOMAINS,
        )
        if any(not domain.strip() for domain in domains):
            raise ValueError("training_domains must be non-empty strings")
        if not domains or domains != tuple(sorted(set(domains))):
            raise ValueError("training_domains must be sorted and unique")

        models_input = _shared_artifacts._mapping(
            self.models,
            "models",
            max_items=MAX_GBM_ARTIFACT_MODELS,
        )
        calibrators_input = _shared_artifacts._mapping(
            self.calibrators,
            "calibrators",
            max_items=MAX_GBM_ARTIFACT_MODELS,
        )
        model_ids = tuple(sorted(models_input))
        if not model_ids or set(model_ids) != set(calibrators_input):
            raise ValueError("models and calibrators must cover identical non-empty model IDs")
        if (
            _metadata_size(
                feature_schema=self.feature_schema,
                model_ids=model_ids,
                training_domains=domains,
            )
            > MAX_PREDICTOR_METADATA_TOTAL_BYTES
        ):
            raise ValueError(
                "GBM artifact metadata exceeds the aggregate limit "
                f"({MAX_PREDICTOR_METADATA_TOTAL_BYTES:,} UTF-8 bytes)"
            )

        numeric_scalars = _FIXED_NUMERIC_SCALARS
        total_stumps = 0
        models_copy: dict[str, GbmModel] = {}
        for model_id in model_ids:
            model = models_input[model_id]
            if type(model) is not GbmModel:
                raise TypeError("models must map IDs to exact GbmModel values")
            if model.feature_width != self.feature_schema.dimension:
                raise ValueError(f"model {model_id!r} feature width does not match feature schema")
            if model.learning_rate != self.training_config.learning_rate:
                raise ValueError(f"model {model_id!r} learning rate does not match training config")
            if len(model.stumps) > self.training_config.n_estimators:
                raise ValueError(f"model {model_id!r} stump count exceeds training config")
            exact_stumps: list[RegressionStump] = []
            for stump in model.stumps:
                if type(stump) is not RegressionStump:
                    raise TypeError("artifact models must contain exact RegressionStump values")
                exact_stumps.append(
                    RegressionStump(
                        feature_index=stump.feature_index,
                        split_value=stump.split_value,
                        left_value=stump.left_value,
                        right_value=stump.right_value,
                    )
                )
            total_stumps += len(model.stumps)
            if total_stumps > MAX_GBM_ARTIFACT_TOTAL_STUMPS:
                raise ValueError(
                    "GBM artifact exceeds the aggregate stump limit "
                    f"({MAX_GBM_ARTIFACT_TOTAL_STUMPS:,})"
                )
            numeric_scalars += 1 + 4 * len(model.stumps)
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "GBM artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )
            models_copy[model_id] = GbmModel(
                feature_width=model.feature_width,
                base_value=model.base_value,
                learning_rate=model.learning_rate,
                stumps=tuple(exact_stumps),
            )

        calibrators_copy: dict[str, IsotonicCalibrator] = {}
        for model_id in model_ids:
            calibrator = calibrators_input[model_id]
            if type(calibrator) is not IsotonicCalibrator:
                raise TypeError("calibrators must map IDs to exact IsotonicCalibrator values")
            point_count = len(calibrator.upper_bounds)
            if point_count > example_count:
                raise ValueError(
                    f"calibrator for model {model_id!r} exceeds training_example_count"
                )
            numeric_scalars += 2 * point_count
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "GBM artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )
            calibrators_copy[model_id] = calibrator

        object.__setattr__(self, "models", MappingProxyType(models_copy))
        object.__setattr__(self, "calibrators", MappingProxyType(calibrators_copy))
        object.__setattr__(self, "training_data_sha256", training_hash)
        object.__setattr__(self, "training_example_count", example_count)
        object.__setattr__(self, "training_domains", domains)

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Return the canonical model catalogue."""

        return tuple(self.models)

    def build_predictor(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> PerModelCalibratedQualityPredictor:
        """Rebuild the calibrated predictor without network or code execution."""

        encoder = PromptFeatureEncoder(self.feature_schema, embedding_provider)
        base = GbmQualityPredictor(
            vectorizer=encoder.transform_one,
            models=self.models,
            batch_vectorizer=encoder.transform_many,
        )
        return PerModelCalibratedQualityPredictor(base, self.calibrators)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible object."""

        return {
            "algorithm_id": self.algorithm_id,
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "calibrators": {
                model_id: {
                    "upper_bounds": list(self.calibrators[model_id].upper_bounds),
                    "values": list(self.calibrators[model_id].values),
                }
                for model_id in self.model_ids
            },
            "feature_schema": self.feature_schema.to_dict(),
            "models": {
                model_id: {
                    "base_value": self.models[model_id].base_value,
                    "stumps": [
                        [
                            stump.feature_index,
                            stump.split_value,
                            stump.left_value,
                            stump.right_value,
                        ]
                        for stump in self.models[model_id].stumps
                    ],
                }
                for model_id in self.model_ids
            },
            "training": {
                "config": {
                    "learning_rate": self.training_config.learning_rate,
                    "min_gain": self.training_config.min_gain,
                    "min_samples_leaf": self.training_config.min_samples_leaf,
                    "n_estimators": self.training_config.n_estimators,
                },
                "data_sha256": self.training_data_sha256,
                "domains": list(self.training_domains),
                "example_count": self.training_example_count,
            },
        }

    def to_json(self) -> str:
        """Serialize deterministic strict JSON with a final newline."""

        document = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        _shared_artifacts._validate_artifact_document(document)
        return document

    def save(self, path: str | Path) -> Path:
        """Atomically write a locally validated artifact."""

        return replace_text_bundle(
            (
                AtomicTextWrite(
                    Path(path),
                    self.to_json(),
                    GbmPredictorArtifact.from_json,
                ),
            )
        )[0]

    @classmethod
    def from_json(cls, document: str) -> GbmPredictorArtifact:
        """Parse bounded strict JSON without pickle or code execution."""

        _shared_artifacts._validate_artifact_document(document)
        _shared_artifacts._preflight_json_structure(document)
        number_tokens = 0

        def count_number_token() -> None:
            nonlocal number_tokens
            number_tokens += 1
            if number_tokens > _shared_artifacts.MAX_PREDICTOR_JSON_NUMBER_TOKENS:
                raise ValueError(
                    "GBM predictor artifact exceeds the JSON number-token limit "
                    f"({_shared_artifacts.MAX_PREDICTOR_JSON_NUMBER_TOKENS:,})"
                )

        def bounded_integer(token: str) -> int:
            count_number_token()
            return _shared_artifacts._bounded_json_integer(token)

        def bounded_float(token: str) -> float:
            count_number_token()
            return _shared_artifacts._bounded_json_float(token)

        def reject_constant(value: str) -> object:
            raise ValueError(f"non-standard JSON number {value!r} is forbidden")

        def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r} is forbidden")
                result[key] = value
            return result

        try:
            payload = json.loads(
                document,
                parse_int=bounded_integer,
                parse_float=bounded_float,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError("GBM predictor artifact is not valid strict JSON") from error
        return cls.from_dict(_shared_artifacts._mapping(payload, "artifact", max_items=7))

    @classmethod
    def load(cls, path: str | Path) -> GbmPredictorArtifact:
        """Load one bounded local JSON document without network access."""

        try:
            with Path(path).open("rb") as stream:
                payload = stream.read(_shared_artifacts.MAX_PREDICTOR_ARTIFACT_BYTES + 1)
        except OSError as error:
            raise ValueError(f"cannot read GBM predictor artifact: {path}") from error
        if len(payload) > _shared_artifacts.MAX_PREDICTOR_ARTIFACT_BYTES:
            raise ValueError(
                "predictor artifact exceeds "
                f"{_shared_artifacts.MAX_PREDICTOR_ARTIFACT_BYTES:,} UTF-8 bytes"
            )
        try:
            document = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"cannot read GBM predictor artifact: {path}") from error
        return cls.from_json(document)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GbmPredictorArtifact:
        """Validate and construct one schema-v1 artifact."""

        payload = _shared_artifacts._mapping(payload, "artifact", max_items=7)
        _shared_artifacts._strict_fields(
            payload,
            {
                "algorithm_id",
                "artifact_kind",
                "artifact_version",
                "calibrators",
                "feature_schema",
                "models",
                "training",
            },
            "artifact",
        )
        training = _shared_artifacts._mapping(payload["training"], "training", max_items=4)
        _shared_artifacts._strict_fields(
            training,
            {"config", "data_sha256", "domains", "example_count"},
            "training",
        )
        config_payload = _shared_artifacts._mapping(
            training["config"],
            "training.config",
            max_items=4,
        )
        _shared_artifacts._strict_fields(
            config_payload,
            {"learning_rate", "min_gain", "min_samples_leaf", "n_estimators"},
            "training.config",
        )
        config = GbmTrainingConfig(
            n_estimators=_shared_artifacts._bounded_integer(
                config_payload["n_estimators"],
                "training.config.n_estimators",
                positive=True,
            ),
            learning_rate=_shared_artifacts._finite_float(
                config_payload["learning_rate"],
                "training.config.learning_rate",
            ),
            min_samples_leaf=_shared_artifacts._bounded_integer(
                config_payload["min_samples_leaf"],
                "training.config.min_samples_leaf",
                positive=True,
            ),
            min_gain=_shared_artifacts._finite_float(
                config_payload["min_gain"],
                "training.config.min_gain",
            ),
        )
        example_count = _shared_artifacts._bounded_integer(
            training["example_count"],
            "training.example_count",
            positive=True,
        )
        domains = _shared_artifacts._text_tuple(
            training["domains"],
            "training.domains",
            max_items=MAX_PREDICTOR_TRAINING_DOMAINS,
        )
        feature_schema = PromptFeatureSchema.from_dict(
            _shared_artifacts._mapping(
                payload["feature_schema"],
                "feature_schema",
                max_items=6,
            )
        )
        models_payload = _shared_artifacts._mapping(
            payload["models"],
            "models",
            max_items=MAX_GBM_ARTIFACT_MODELS,
        )
        calibrators_payload = _shared_artifacts._mapping(
            payload["calibrators"],
            "calibrators",
            max_items=MAX_GBM_ARTIFACT_MODELS,
        )

        total_stumps = 0
        models: dict[str, GbmModel] = {}
        for model_id, raw_model in models_payload.items():
            model = _shared_artifacts._mapping(
                raw_model,
                f"models.{model_id}",
                max_items=2,
            )
            _shared_artifacts._strict_fields(
                model,
                {"base_value", "stumps"},
                f"models.{model_id}",
            )
            stumps, total_stumps = _stump_payload(
                model["stumps"],
                context=f"models.{model_id}.stumps",
                config=config,
                running_total=total_stumps,
            )
            models[model_id] = GbmModel(
                feature_width=feature_schema.dimension,
                base_value=_shared_artifacts._finite_float(
                    model["base_value"],
                    f"models.{model_id}.base_value",
                ),
                learning_rate=config.learning_rate,
                stumps=stumps,
            )

        numeric_scalars = _FIXED_NUMERIC_SCALARS + sum(
            1 + 4 * len(model.stumps) for model in models.values()
        )
        if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
            raise ValueError(
                f"GBM artifact exceeds the numeric scalar limit ({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
            )
        calibrators: dict[str, IsotonicCalibrator] = {}
        max_points = min(example_count, MAX_PREDICTOR_CALIBRATOR_POINTS)
        for model_id, raw_calibrator in calibrators_payload.items():
            calibrator = _shared_artifacts._mapping(
                raw_calibrator,
                f"calibrators.{model_id}",
                max_items=2,
            )
            _shared_artifacts._strict_fields(
                calibrator,
                {"upper_bounds", "values"},
                f"calibrators.{model_id}",
            )
            remaining = MAX_PREDICTOR_NUMERIC_SCALARS - numeric_scalars
            upper_bounds = _shared_artifacts._finite_tuple(
                calibrator["upper_bounds"],
                f"calibrators.{model_id}.upper_bounds",
                max_items=min(max_points, remaining // 2),
            )
            remaining -= len(upper_bounds)
            values = _shared_artifacts._finite_tuple(
                calibrator["values"],
                f"calibrators.{model_id}.values",
                max_items=min(max_points, remaining),
            )
            numeric_scalars += len(upper_bounds) + len(values)
            calibrators[model_id] = IsotonicCalibrator(upper_bounds, values)

        return cls(
            artifact_version=payload["artifact_version"],  # type: ignore[arg-type]
            artifact_kind=payload["artifact_kind"],  # type: ignore[arg-type]
            algorithm_id=payload["algorithm_id"],  # type: ignore[arg-type]
            feature_schema=feature_schema,
            models=models,
            calibrators=calibrators,
            training_data_sha256=training["data_sha256"],  # type: ignore[arg-type]
            training_example_count=example_count,
            training_domains=domains,
            training_config=config,
        )
