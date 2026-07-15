# SPDX-License-Identifier: Apache-2.0
"""Portable, validated JSON artifacts for calibrated bilinear predictors."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tierroute.core.atomic_io import AtomicTextWrite, replace_text_bundle
from tierroute.features import EmbeddingProvider, PromptFeatureEncoder, PromptFeatureSchema
from tierroute.predictors._ridge import CENTERED_RIDGE_SOLVER_ID
from tierroute.predictors.base import BilinearQualityPredictor
from tierroute.predictors.calibration import (
    IsotonicCalibrator,
    PerModelCalibratedQualityPredictor,
)
from tierroute.predictors.solvers import validate_ridge_solver_id

PREDICTOR_ARTIFACT_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _strict_fields(payload: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{context} fields mismatch: missing={missing}, extra={extra}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed object")
    return value


def _finite_tuple(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{context} must contain JSON numbers")
    result = tuple(float(item) for item in value)
    if not result or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{context} must contain finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class BilinearPredictorArtifact:
    """All state needed for deterministic offline bilinear inference."""

    feature_schema: PromptFeatureSchema
    model_weights: Mapping[str, tuple[float, ...]]
    model_bias: Mapping[str, float]
    calibrators: Mapping[str, IsotonicCalibrator]
    training_data_sha256: str
    training_example_count: int
    training_domains: tuple[str, ...]
    ridge: float
    seed: int
    solver_id: str = CENTERED_RIDGE_SOLVER_ID
    artifact_version: int = PREDICTOR_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if type(self.artifact_version) is not int or self.artifact_version != (
            PREDICTOR_ARTIFACT_VERSION
        ):
            raise ValueError(f"artifact_version must equal {PREDICTOR_ARTIFACT_VERSION}")
        weights_copy: dict[str, tuple[float, ...]] = {}
        for model_id, weights in self.model_weights.items():
            if not isinstance(weights, (list, tuple)):
                raise ValueError(f"weights for model {model_id!r} must be a sequence")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in weights
            ):
                raise ValueError(f"weights for model {model_id!r} must be numbers")
            weights_copy[model_id] = tuple(float(value) for value in weights)
        bias_copy: dict[str, float] = {}
        for model_id, value in self.model_bias.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("model bias values must be numbers")
            bias_copy[model_id] = float(value)
        calibrators_copy = dict(self.calibrators)
        if any(
            not isinstance(calibrator, IsotonicCalibrator)
            for calibrator in calibrators_copy.values()
        ):
            raise ValueError("artifact calibrators must be isotonic calibrators")

        model_ids = set(weights_copy)
        if not model_ids or model_ids != set(self.model_bias) or model_ids != set(self.calibrators):
            raise ValueError("weights, bias, and calibrators must cover identical models")
        if any(not isinstance(model_id, str) or not model_id.strip() for model_id in model_ids):
            raise ValueError("artifact model IDs must be non-empty strings")
        for model_id, weights in weights_copy.items():
            if len(weights) != self.feature_schema.dimension:
                raise ValueError(f"weight width mismatch for model {model_id!r}")
            if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in weights):
                raise ValueError(f"weights for model {model_id!r} must be finite")
        if any(not math.isfinite(value) for value in bias_copy.values()):
            raise ValueError("model bias values must be finite")
        if not isinstance(self.training_data_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.training_data_sha256
        ):
            raise ValueError("training_data_sha256 must be lowercase SHA-256 hex")
        if isinstance(self.training_example_count, bool) or not isinstance(
            self.training_example_count, int
        ):
            raise TypeError("training_example_count must be an integer")
        if self.training_example_count < 1:
            raise ValueError("training_example_count must be positive")
        if not self.training_domains or self.training_domains != tuple(
            sorted(set(self.training_domains))
        ):
            raise ValueError("training_domains must be sorted and unique")
        if any(
            not isinstance(domain, str) or not domain.strip() for domain in self.training_domains
        ):
            raise ValueError("training_domains must be non-empty strings")
        if (
            isinstance(self.ridge, bool)
            or not isinstance(self.ridge, (int, float))
            or not math.isfinite(self.ridge)
            or self.ridge <= 0
        ):
            raise ValueError("ridge must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        validate_ridge_solver_id(self.solver_id)

        object.__setattr__(self, "model_weights", MappingProxyType(weights_copy))
        object.__setattr__(self, "model_bias", MappingProxyType(bias_copy))
        object.__setattr__(self, "calibrators", MappingProxyType(calibrators_copy))
        object.__setattr__(self, "ridge", float(self.ridge))

    @property
    def model_ids(self) -> tuple[str, ...]:
        """Return the deterministic candidate catalogue."""

        return tuple(sorted(self.model_weights))

    def build_predictor(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> PerModelCalibratedQualityPredictor:
        """Construct an offline predictor, requiring local embeddings when declared."""

        encoder = PromptFeatureEncoder(self.feature_schema, embedding_provider)
        base = BilinearQualityPredictor(
            vectorizer=encoder.transform_one,
            model_weights=self.model_weights,
            model_bias=self.model_bias,
            batch_vectorizer=encoder.transform_many,
        )
        return PerModelCalibratedQualityPredictor(base, self.calibrators)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON object."""

        return {
            "artifact_version": self.artifact_version,
            "feature_schema": self.feature_schema.to_dict(),
            "model_weights": {
                model_id: list(self.model_weights[model_id]) for model_id in self.model_ids
            },
            "model_bias": {
                model_id: float(self.model_bias[model_id]) for model_id in self.model_ids
            },
            "calibrators": {
                model_id: {
                    "upper_bounds": list(self.calibrators[model_id].upper_bounds),
                    "values": list(self.calibrators[model_id].values),
                }
                for model_id in self.model_ids
            },
            "training": {
                "data_sha256": self.training_data_sha256,
                "example_count": self.training_example_count,
                "domains": list(self.training_domains),
                "ridge": self.ridge,
                "seed": self.seed,
                "solver_id": self.solver_id,
            },
        }

    def to_json(self) -> str:
        """Serialize deterministic UTF-8 JSON; non-finite numbers are forbidden."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )

    def save(self, path: str | Path) -> Path:
        """Atomically write a JSON artifact and return its path."""

        destination = Path(path)
        return replace_text_bundle(
            (AtomicTextWrite(destination, self.to_json(), type(self).from_json),)
        )[0]

    @classmethod
    def from_json(cls, document: str) -> BilinearPredictorArtifact:
        """Parse strict JSON without pickle or code execution."""

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
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("predictor artifact is not valid strict JSON") from error
        return cls.from_dict(_mapping(payload, "artifact"))

    @classmethod
    def load(cls, path: str | Path) -> BilinearPredictorArtifact:
        """Read a local JSON artifact without network access."""

        try:
            document = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read predictor artifact: {path}") from error
        return cls.from_json(document)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BilinearPredictorArtifact:
        """Validate and construct a version-1 artifact."""

        _strict_fields(
            payload,
            {
                "artifact_version",
                "feature_schema",
                "model_weights",
                "model_bias",
                "calibrators",
                "training",
            },
            "artifact",
        )
        weights_payload = _mapping(payload["model_weights"], "model_weights")
        bias_payload = _mapping(payload["model_bias"], "model_bias")
        calibrator_payload = _mapping(payload["calibrators"], "calibrators")
        training = _mapping(payload["training"], "training")
        _strict_fields(
            training,
            {"data_sha256", "example_count", "domains", "ridge", "seed", "solver_id"},
            "training",
        )

        weights = {
            model_id: _finite_tuple(value, f"model_weights.{model_id}")
            for model_id, value in weights_payload.items()
        }
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in bias_payload.values()
        ):
            raise ValueError("model_bias values must be JSON numbers")
        bias = {model_id: float(value) for model_id, value in bias_payload.items()}
        calibrators = {}
        for model_id, value in calibrator_payload.items():
            item = _mapping(value, f"calibrators.{model_id}")
            _strict_fields(item, {"upper_bounds", "values"}, f"calibrators.{model_id}")
            calibrators[model_id] = IsotonicCalibrator(
                _finite_tuple(item["upper_bounds"], f"calibrators.{model_id}.upper_bounds"),
                _finite_tuple(item["values"], f"calibrators.{model_id}.values"),
            )
        domains = training["domains"]
        if not isinstance(domains, list) or any(not isinstance(item, str) for item in domains):
            raise ValueError("training.domains must be an array of strings")
        ridge = training["ridge"]
        if isinstance(ridge, bool) or not isinstance(ridge, (int, float)):
            raise ValueError("training.ridge must be a JSON number")

        return cls(
            artifact_version=payload["artifact_version"],  # type: ignore[arg-type]
            feature_schema=PromptFeatureSchema.from_dict(
                _mapping(payload["feature_schema"], "feature_schema")
            ),
            model_weights=weights,
            model_bias=bias,
            calibrators=calibrators,
            training_data_sha256=training["data_sha256"],  # type: ignore[arg-type]
            training_example_count=training["example_count"],  # type: ignore[arg-type]
            training_domains=tuple(domains),
            ridge=float(ridge),  # type: ignore[arg-type]
            seed=training["seed"],  # type: ignore[arg-type]
            solver_id=training["solver_id"],  # type: ignore[arg-type]
        )
