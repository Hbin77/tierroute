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
from tierroute.predictors.resource_limits import (
    MAX_PREDICTOR_ARTIFACT_BYTES,
    MAX_PREDICTOR_JSON_NESTING_DEPTH,
    MAX_PREDICTOR_JSON_NUMBER_CHARACTERS,
    MAX_PREDICTOR_JSON_NUMBER_TOKENS,
    MAX_PREDICTOR_JSON_STRING_CHARACTERS,
    MAX_PREDICTOR_JSON_STRING_TOKENS,
    MAX_PREDICTOR_JSON_STRUCTURE_TOKENS,
    MAX_PREDICTOR_METADATA_TEXT_BYTES,
    MAX_PREDICTOR_METADATA_TOTAL_BYTES,
    MAX_PREDICTOR_MODELS,
    MAX_PREDICTOR_NUMERIC_SCALARS,
    MAX_PREDICTOR_TRAINING_DOMAINS,
)
from tierroute.predictors.solvers import validate_ridge_solver_id

PREDICTOR_ARTIFACT_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_PREDICTOR_INTEGER_EXCLUSIVE = 10**MAX_PREDICTOR_JSON_NUMBER_CHARACTERS
_MIN_PREDICTOR_INTEGER_EXCLUSIVE = -(10 ** (MAX_PREDICTOR_JSON_NUMBER_CHARACTERS - 1))


def _validate_artifact_document(document: str) -> None:
    if type(document) is not str:
        raise ValueError("predictor artifact must be text")
    if len(document) > MAX_PREDICTOR_ARTIFACT_BYTES:
        raise ValueError(f"predictor artifact exceeds {MAX_PREDICTOR_ARTIFACT_BYTES:,} UTF-8 bytes")
    try:
        encoded = document.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("predictor artifact is not valid UTF-8 text") from error
    if len(encoded) > MAX_PREDICTOR_ARTIFACT_BYTES:
        raise ValueError(f"predictor artifact exceeds {MAX_PREDICTOR_ARTIFACT_BYTES:,} UTF-8 bytes")


def _preflight_json_structure(document: str) -> None:
    """Bound parser-amplifying JSON structure without allocating decoded values."""

    depth = 0
    string_tokens = 0
    structure_tokens = 0
    index = 0
    while index < len(document):
        character = document[index]
        if character == '"':
            string_tokens += 1
            if string_tokens > MAX_PREDICTOR_JSON_STRING_TOKENS:
                raise ValueError(
                    "predictor artifact exceeds the JSON string-token limit "
                    f"({MAX_PREDICTOR_JSON_STRING_TOKENS:,})"
                )
            start = index
            index += 1
            while index < len(document):
                if index - start + 1 > MAX_PREDICTOR_JSON_STRING_CHARACTERS:
                    raise ValueError(
                        "predictor artifact JSON string exceeds the lexical limit "
                        f"({MAX_PREDICTOR_JSON_STRING_CHARACTERS:,} characters)"
                    )
                if document[index] == "\\":
                    index += 2
                    continue
                if document[index] == '"':
                    break
                index += 1
        elif character in "[{":
            depth += 1
            structure_tokens += 1
            if depth > MAX_PREDICTOR_JSON_NESTING_DEPTH:
                raise ValueError(
                    "predictor artifact exceeds the JSON nesting limit "
                    f"({MAX_PREDICTOR_JSON_NESTING_DEPTH:,})"
                )
        elif character in "]}":
            depth -= 1
        elif character == ",":
            structure_tokens += 1
        if structure_tokens > MAX_PREDICTOR_JSON_STRUCTURE_TOKENS:
            raise ValueError(
                "predictor artifact exceeds the JSON structure-token limit "
                f"({MAX_PREDICTOR_JSON_STRUCTURE_TOKENS:,})"
            )
        index += 1


def _bounded_json_integer(token: str) -> int:
    if len(token) > MAX_PREDICTOR_JSON_NUMBER_CHARACTERS:
        raise ValueError(
            "predictor artifact integer exceeds the JSON number limit "
            f"({MAX_PREDICTOR_JSON_NUMBER_CHARACTERS:,} characters)"
        )
    return int(token)


def _bounded_json_float(token: str) -> float:
    if len(token) > MAX_PREDICTOR_JSON_NUMBER_CHARACTERS:
        raise ValueError(
            "predictor artifact float exceeds the JSON number limit "
            f"({MAX_PREDICTOR_JSON_NUMBER_CHARACTERS:,} characters)"
        )
    result = float(token)
    if not math.isfinite(result):
        raise ValueError("predictor artifact JSON numbers must fit finite binary64")
    return result


def _finite_float(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{context} must be a number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{context} must fit a finite float") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _bounded_integer(value: object, context: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    if value >= _MAX_PREDICTOR_INTEGER_EXCLUSIVE or value <= _MIN_PREDICTOR_INTEGER_EXCLUSIVE:
        raise ValueError(
            f"{context} exceeds the predictor artifact integer limit "
            f"({MAX_PREDICTOR_JSON_NUMBER_CHARACTERS:,} JSON characters)"
        )
    if positive and value < 1:
        raise ValueError(f"{context} must be positive")
    return value


def _metadata_bytes(value: str, context: str) -> int:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must contain valid Unicode") from error
    if len(encoded) > MAX_PREDICTOR_METADATA_TEXT_BYTES:
        raise ValueError(
            f"{context} exceeds the predictor metadata limit "
            f"({MAX_PREDICTOR_METADATA_TEXT_BYTES:,} UTF-8 bytes)"
        )
    return len(encoded)


def _strict_fields(payload: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{context} fields mismatch: missing={missing}, extra={extra}")


def _mapping(
    value: object,
    context: str,
    *,
    max_items: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a string-keyed object")
    result: dict[str, object] = {}
    try:
        iterator = iter(value.items())
        for item in iterator:
            if len(result) >= max_items:
                limit_name = (
                    "predictor model limit"
                    if context in {"model_weights", "model_bias", "calibrators"}
                    else "field-count limit"
                )
                raise ValueError(f"{context} exceeds the {limit_name} ({max_items:,})")
            key, item_value = item
            if type(key) is not str:
                raise ValueError(f"{context} must be a string-keyed object")
            if key in result:
                raise ValueError(f"{context} contains a duplicate key")
            result[key] = item_value
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{context} could not be read deterministically") from error
    return result


def _finite_tuple(
    value: object,
    context: str,
    *,
    max_items: int = MAX_PREDICTOR_NUMERIC_SCALARS,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be an array")
    result: list[float] = []
    try:
        iterator = iter(value)
        for item in iterator:
            if len(result) >= max_items:
                raise ValueError(f"{context} exceeds the predictor artifact numeric limit")
            result.append(_finite_float(item, f"{context} item"))
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{context} could not be read deterministically") from error
    if not result:
        raise ValueError(f"{context} must contain finite numbers")
    return tuple(result)


def _text_tuple(
    value: object,
    context: str,
    *,
    max_items: int,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be an array of strings")
    result: list[str] = []
    try:
        iterator = iter(value)
        for item in iterator:
            if len(result) >= max_items:
                raise ValueError(f"{context} exceeds the predictor artifact limit ({max_items:,})")
            if type(item) is not str:
                raise ValueError(f"{context} must be an array of strings")
            result.append(item)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{context} could not be read deterministically") from error
    return tuple(result)


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

        if type(self.feature_schema) is not PromptFeatureSchema:
            raise TypeError("feature_schema must be a PromptFeatureSchema")
        training_example_count = _bounded_integer(
            self.training_example_count,
            "training_example_count",
            positive=True,
        )
        seed = _bounded_integer(self.seed, "seed")
        ridge = _finite_float(self.ridge, "ridge")
        if ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if type(self.training_data_sha256) is not str or not _SHA256_PATTERN.fullmatch(
            self.training_data_sha256
        ):
            raise ValueError("training_data_sha256 must be lowercase SHA-256 hex")
        training_domains = _text_tuple(
            self.training_domains,
            "training_domains",
            max_items=MAX_PREDICTOR_TRAINING_DOMAINS,
        )
        if any(not domain.strip() for domain in training_domains):
            raise ValueError("training_domains must be non-empty strings")
        if not training_domains or training_domains != tuple(sorted(set(training_domains))):
            raise ValueError("training_domains must be sorted and unique")

        weights_input = _mapping(
            self.model_weights,
            "model_weights",
            max_items=MAX_PREDICTOR_MODELS,
        )
        bias_input = _mapping(
            self.model_bias,
            "model_bias",
            max_items=MAX_PREDICTOR_MODELS,
        )
        calibrators_copy = _mapping(
            self.calibrators,
            "calibrators",
            max_items=MAX_PREDICTOR_MODELS,
        )

        model_ids = set(weights_input)
        if not model_ids or model_ids != set(bias_input) or model_ids != set(calibrators_copy):
            raise ValueError("weights, bias, and calibrators must cover identical models")
        if any(
            type(calibrator) is not IsotonicCalibrator for calibrator in calibrators_copy.values()
        ):
            raise ValueError("artifact calibrators must be isotonic calibrators")

        metadata_bytes = _metadata_bytes(self.solver_id, "solver_id")
        for model_id in model_ids:
            metadata_bytes += 3 * _metadata_bytes(model_id, "artifact model ID")
        for domain in training_domains:
            metadata_bytes += _metadata_bytes(domain, "training domain")
        for tag in self.feature_schema.domain_tags:
            metadata_bytes += _metadata_bytes(tag, "feature domain tag")
        identity = self.feature_schema.embedding_identity
        if identity is not None:
            for name in ("provider", "model_id", "revision", "pooling"):
                metadata_bytes += _metadata_bytes(
                    getattr(identity, name),
                    f"embedding {name}",
                )
            metadata_bytes += len(identity.asset_manifest_sha256)
        if metadata_bytes > MAX_PREDICTOR_METADATA_TOTAL_BYTES:
            raise ValueError(
                "predictor artifact metadata exceeds the aggregate limit "
                f"({MAX_PREDICTOR_METADATA_TOTAL_BYTES:,} UTF-8 bytes)"
            )

        numeric_scalars = 7  # feature means/scales plus ridge
        weights_copy: dict[str, tuple[float, ...]] = {}
        for model_id, weights in weights_input.items():
            normalized_weights = _finite_tuple(
                weights,
                f"weights for model {model_id!r}",
                max_items=self.feature_schema.dimension,
            )
            if len(normalized_weights) != self.feature_schema.dimension:
                raise ValueError(f"weight width mismatch for model {model_id!r}")
            numeric_scalars += len(normalized_weights)
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "predictor artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )
            weights_copy[model_id] = normalized_weights

        bias_copy: dict[str, float] = {}
        for model_id, value in bias_input.items():
            numeric_scalars += 1
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "predictor artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )
            bias_copy[model_id] = _finite_float(value, "model bias value")

        for model_id, calibrator in calibrators_copy.items():
            point_count = len(calibrator.upper_bounds)
            if point_count > training_example_count:
                raise ValueError(
                    f"calibrator for model {model_id!r} exceeds training_example_count"
                )
            numeric_scalars += 2 * point_count
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "predictor artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )

        validate_ridge_solver_id(self.solver_id)

        object.__setattr__(self, "model_weights", MappingProxyType(weights_copy))
        object.__setattr__(self, "model_bias", MappingProxyType(bias_copy))
        object.__setattr__(self, "calibrators", MappingProxyType(calibrators_copy))
        object.__setattr__(self, "training_example_count", training_example_count)
        object.__setattr__(self, "training_domains", training_domains)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "ridge", ridge)

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
        _validate_artifact_document(document)
        return document

    def save(self, path: str | Path) -> Path:
        """Atomically write a JSON artifact and return its path."""

        destination = Path(path)
        return replace_text_bundle(
            (AtomicTextWrite(destination, self.to_json(), type(self).from_json),)
        )[0]

    @classmethod
    def from_json(cls, document: str) -> BilinearPredictorArtifact:
        """Parse strict JSON without pickle or code execution."""

        _validate_artifact_document(document)
        _preflight_json_structure(document)
        number_tokens = 0

        def count_number_token() -> None:
            nonlocal number_tokens
            number_tokens += 1
            if number_tokens > MAX_PREDICTOR_JSON_NUMBER_TOKENS:
                raise ValueError(
                    "predictor artifact exceeds the JSON number-token limit "
                    f"({MAX_PREDICTOR_JSON_NUMBER_TOKENS:,})"
                )

        def bounded_integer(token: str) -> int:
            count_number_token()
            return _bounded_json_integer(token)

        def bounded_float(token: str) -> float:
            count_number_token()
            return _bounded_json_float(token)

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
            raise ValueError("predictor artifact is not valid strict JSON") from error
        return cls.from_dict(_mapping(payload, "artifact", max_items=6))

    @classmethod
    def load(cls, path: str | Path) -> BilinearPredictorArtifact:
        """Read a local JSON artifact without network access."""

        try:
            with Path(path).open("rb") as stream:
                payload = stream.read(MAX_PREDICTOR_ARTIFACT_BYTES + 1)
        except OSError as error:
            raise ValueError(f"cannot read predictor artifact: {path}") from error
        if len(payload) > MAX_PREDICTOR_ARTIFACT_BYTES:
            raise ValueError(
                f"predictor artifact exceeds {MAX_PREDICTOR_ARTIFACT_BYTES:,} UTF-8 bytes"
            )
        try:
            document = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"cannot read predictor artifact: {path}") from error
        return cls.from_json(document)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BilinearPredictorArtifact:
        """Validate and construct a version-1 artifact."""

        payload = _mapping(payload, "artifact", max_items=6)
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
        weights_payload = _mapping(
            payload["model_weights"], "model_weights", max_items=MAX_PREDICTOR_MODELS
        )
        bias_payload = _mapping(payload["model_bias"], "model_bias", max_items=MAX_PREDICTOR_MODELS)
        calibrator_payload = _mapping(
            payload["calibrators"], "calibrators", max_items=MAX_PREDICTOR_MODELS
        )
        training = _mapping(payload["training"], "training", max_items=6)
        _strict_fields(
            training,
            {"data_sha256", "example_count", "domains", "ridge", "seed", "solver_id"},
            "training",
        )

        example_count = _bounded_integer(
            training["example_count"],
            "training.example_count",
            positive=True,
        )
        seed = _bounded_integer(training["seed"], "training.seed")
        domains = _text_tuple(
            training["domains"],
            "training.domains",
            max_items=MAX_PREDICTOR_TRAINING_DOMAINS,
        )

        feature_schema = PromptFeatureSchema.from_dict(
            _mapping(payload["feature_schema"], "feature_schema", max_items=6)
        )

        numeric_scalars = 7
        weights: dict[str, tuple[float, ...]] = {}
        for model_id, value in weights_payload.items():
            normalized = _finite_tuple(
                value,
                f"model_weights.{model_id}",
                max_items=feature_schema.dimension,
            )
            if len(normalized) != feature_schema.dimension:
                raise ValueError(f"model_weights.{model_id} has the wrong feature dimension")
            numeric_scalars += len(normalized)
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "predictor artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )
            weights[model_id] = normalized

        bias: dict[str, float] = {}
        for model_id, value in bias_payload.items():
            numeric_scalars += 1
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "predictor artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )
            bias[model_id] = _finite_float(value, f"model_bias.{model_id}")
        calibrators: dict[str, IsotonicCalibrator] = {}
        for model_id, value in calibrator_payload.items():
            item = _mapping(value, f"calibrators.{model_id}", max_items=2)
            _strict_fields(item, {"upper_bounds", "values"}, f"calibrators.{model_id}")
            upper_bounds = _finite_tuple(
                item["upper_bounds"],
                f"calibrators.{model_id}.upper_bounds",
                max_items=example_count,
            )
            values = _finite_tuple(
                item["values"],
                f"calibrators.{model_id}.values",
                max_items=example_count,
            )
            numeric_scalars += len(upper_bounds) + len(values)
            if numeric_scalars > MAX_PREDICTOR_NUMERIC_SCALARS:
                raise ValueError(
                    "predictor artifact exceeds the numeric scalar limit "
                    f"({MAX_PREDICTOR_NUMERIC_SCALARS:,})"
                )
            calibrators[model_id] = IsotonicCalibrator(upper_bounds, values)
        ridge = _finite_float(training["ridge"], "training.ridge")

        return cls(
            artifact_version=payload["artifact_version"],  # type: ignore[arg-type]
            feature_schema=feature_schema,
            model_weights=weights,
            model_bias=bias,
            calibrators=calibrators,
            training_data_sha256=training["data_sha256"],  # type: ignore[arg-type]
            training_example_count=example_count,
            training_domains=domains,
            ridge=ridge,
            seed=seed,
            solver_id=training["solver_id"],  # type: ignore[arg-type]
        )
