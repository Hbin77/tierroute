# SPDX-License-Identifier: Apache-2.0
"""Fitted, leakage-aware prompt feature encoding."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean

from tierroute.features.embeddings import EmbeddingIdentity, EmbeddingProvider
from tierroute.features.surface import extract_surface_features

FEATURE_SCHEMA_VERSION = 1
MAX_FEATURE_DIMENSION = 16_384
MAX_FEATURE_DOMAIN_TAGS = 4_096
MAX_FEATURE_METADATA_TEXT_BYTES = 4 * 1024
_CONTINUOUS_NAMES = (
    "log1p_character_count",
    "log1p_word_count",
    "log1p_line_count",
)
_BINARY_NAMES = ("has_code", "has_math")


def _bounded_sequence_snapshot(
    value: object,
    context: str,
    *,
    max_items: int,
) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context} must be a list or tuple")
    result: list[object] = []
    try:
        iterator = iter(value)
        for item in iterator:
            if len(result) >= max_items:
                raise ValueError(f"{context} exceeds the feature schema limit ({max_items:,})")
            result.append(item)
    except RuntimeError as error:
        raise ValueError(f"{context} could not be read deterministically") from error
    return tuple(result)


def _normalized_feature_numbers(
    value: object,
    context: str,
    *,
    positive: bool,
) -> tuple[float, float, float]:
    snapshot = _bounded_sequence_snapshot(value, context, max_items=3)
    if len(snapshot) != 3:
        raise ValueError("feature schema requires three continuous means and scales")
    normalized: list[float] = []
    for item in snapshot:
        if type(item) not in (int, float):
            raise ValueError(f"{context} must be finite")
        try:
            number = float(item)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{context} must be finite") from error
        if not math.isfinite(number) or (positive and number <= 0):
            qualifier = "finite and positive" if positive else "finite"
            raise ValueError(f"{context} must be {qualifier}")
        normalized.append(number)
    return tuple(normalized)  # type: ignore[return-value]


def _bounded_mapping_snapshot(
    value: object,
    context: str,
    *,
    max_items: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    result: dict[str, object] = {}
    try:
        iterator = iter(value.items())
        for item in iterator:
            if len(result) >= max_items:
                raise ValueError(f"{context} has too many fields")
            key, item_value = item
            if type(key) is not str:
                raise ValueError(f"{context} keys must be strings")
            if key in result:
                raise ValueError(f"{context} contains a duplicate key")
            result[key] = item_value
    except ValueError:
        raise
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{context} could not be read deterministically") from error
    return result


def _continuous_values(prompt: str) -> tuple[float, float, float]:
    features = extract_surface_features(prompt)
    return (
        math.log1p(features.character_count),
        math.log1p(features.word_count),
        math.log1p(features.line_count),
    )


def _embedding_provider_contract(
    provider: EmbeddingProvider,
) -> tuple[int, EmbeddingIdentity]:
    """Snapshot exact provider metadata before trusting equality or dimensions."""

    dimension = provider.dimension
    if type(dimension) is not int:
        raise TypeError("embedding provider dimension must be an exact integer")
    identity = provider.identity
    if type(identity) is not EmbeddingIdentity:
        raise TypeError("embedding provider identity must be an exact EmbeddingIdentity")
    # Reconstructing prevents a frozen dataclass instance that was mutated through
    # ``object.__setattr__`` from crossing the offline identity boundary unchecked.
    normalized_identity = EmbeddingIdentity(
        provider=identity.provider,
        model_id=identity.model_id,
        revision=identity.revision,
        pooling=identity.pooling,
        normalize=identity.normalize,
        asset_manifest_sha256=identity.asset_manifest_sha256,
    )
    return dimension, normalized_identity


@dataclass(frozen=True, slots=True)
class PromptFeatureSchema:
    """Serializable feature statistics fitted on training prompts only."""

    continuous_means: tuple[float, float, float]
    continuous_scales: tuple[float, float, float]
    domain_tags: tuple[str, ...]
    embedding_dimension: int = 0
    embedding_identity: EmbeddingIdentity | None = None
    schema_version: int = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"feature schema_version must equal {FEATURE_SCHEMA_VERSION}")
        means = _normalized_feature_numbers(
            self.continuous_means,
            "continuous means",
            positive=False,
        )
        scales = _normalized_feature_numbers(
            self.continuous_scales,
            "continuous scales",
            positive=True,
        )
        domain_tags = _bounded_sequence_snapshot(
            self.domain_tags,
            "domain_tags",
            max_items=MAX_FEATURE_DOMAIN_TAGS,
        )
        if any(type(tag) is not str or not tag.strip() for tag in domain_tags):
            raise ValueError("domain_tags must be non-empty strings")
        if domain_tags != tuple(sorted(set(domain_tags))):
            raise ValueError("domain_tags must be sorted and unique")
        for tag in domain_tags:
            try:
                encoded = tag.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("domain_tags must contain valid Unicode") from error
            if len(encoded) > MAX_FEATURE_METADATA_TEXT_BYTES:
                raise ValueError(
                    "domain tag exceeds the feature metadata limit "
                    f"({MAX_FEATURE_METADATA_TEXT_BYTES:,} UTF-8 bytes)"
                )
        if type(self.embedding_dimension) is not int:
            raise TypeError("embedding_dimension must be an integer")
        if self.embedding_dimension < 0:
            raise ValueError("embedding_dimension must be non-negative")
        if (
            self.embedding_identity is not None
            and type(self.embedding_identity) is not EmbeddingIdentity
        ):
            raise TypeError("embedding_identity must be an EmbeddingIdentity or None")
        embedding_identity = self.embedding_identity
        if embedding_identity is not None:
            # Snapshot exact fields so a frozen identity mutated through low-level
            # ``object.__setattr__`` cannot create a schema that will not round-trip.
            embedding_identity = EmbeddingIdentity(
                provider=embedding_identity.provider,
                model_id=embedding_identity.model_id,
                revision=embedding_identity.revision,
                pooling=embedding_identity.pooling,
                normalize=embedding_identity.normalize,
                asset_manifest_sha256=embedding_identity.asset_manifest_sha256,
            )
        if (self.embedding_dimension == 0) != (self.embedding_identity is None):
            raise ValueError(
                "embedding identity must be present exactly when embedding_dimension is positive"
            )
        dimension = len(_CONTINUOUS_NAMES) + len(_BINARY_NAMES) + len(domain_tags)
        dimension += self.embedding_dimension
        if dimension > MAX_FEATURE_DIMENSION:
            raise ValueError(
                f"feature dimension exceeds the schema limit ({MAX_FEATURE_DIMENSION:,})"
            )
        object.__setattr__(self, "continuous_means", means)
        object.__setattr__(self, "continuous_scales", scales)
        object.__setattr__(self, "domain_tags", domain_tags)
        object.__setattr__(self, "embedding_identity", embedding_identity)

    @classmethod
    def fit(
        cls,
        prompts: Sequence[str],
        *,
        embedding_dimension: int = 0,
        embedding_identity: EmbeddingIdentity | None = None,
    ) -> PromptFeatureSchema:
        """Fit scaling statistics and prompt-derived tag vocabulary."""

        prompts = tuple(prompts)
        if not prompts:
            raise ValueError("prompts must not be empty")
        continuous_rows = tuple(_continuous_values(prompt) for prompt in prompts)
        means = tuple(fmean(row[index] for row in continuous_rows) for index in range(3))
        scales = []
        for index, mean in enumerate(means):
            variance = fmean((row[index] - mean) ** 2 for row in continuous_rows)
            scale = math.sqrt(variance)
            scales.append(scale if scale > 0 else 1.0)
        domain_tags = tuple(
            sorted(
                {tag for prompt in prompts for tag in extract_surface_features(prompt).domain_tags}
            )
        )
        return cls(
            continuous_means=means,  # type: ignore[arg-type]
            continuous_scales=tuple(scales),  # type: ignore[arg-type]
            domain_tags=domain_tags,
            embedding_dimension=embedding_dimension,
            embedding_identity=embedding_identity,
        )

    @property
    def dimension(self) -> int:
        """Return the encoded vector width."""

        return (
            len(_CONTINUOUS_NAMES)
            + len(_BINARY_NAMES)
            + len(self.domain_tags)
            + self.embedding_dimension
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return stable, human-readable feature names."""

        return (
            *_CONTINUOUS_NAMES,
            *_BINARY_NAMES,
            *(f"domain:{tag}" for tag in self.domain_tags),
            *(f"embedding:{index}" for index in range(self.embedding_dimension)),
        )

    def encode_surface(self, prompt: str) -> tuple[float, ...]:
        """Encode only deterministic pre-call surface features."""

        features = extract_surface_features(prompt)
        continuous = _continuous_values(prompt)
        standardized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                continuous,
                self.continuous_means,
                self.continuous_scales,
                strict=True,
            )
        )
        prompt_tags = set(features.domain_tags)
        return (
            *standardized,
            float(features.has_code),
            float(features.has_math),
            *(float(tag in prompt_tags) for tag in self.domain_tags),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "continuous_means": list(self.continuous_means),
            "continuous_scales": list(self.continuous_scales),
            "domain_tags": list(self.domain_tags),
            "embedding_dimension": self.embedding_dimension,
            "embedding_identity": (
                None if self.embedding_identity is None else self.embedding_identity.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PromptFeatureSchema:
        """Load a strict JSON-compatible representation."""

        expected = {
            "schema_version",
            "continuous_means",
            "continuous_scales",
            "domain_tags",
            "embedding_dimension",
            "embedding_identity",
        }
        payload = _bounded_mapping_snapshot(
            payload,
            "feature schema",
            max_items=len(expected),
        )
        if set(payload) != expected:
            raise ValueError("feature schema fields do not match schema version 1")
        means_payload = payload["continuous_means"]
        scales_payload = payload["continuous_scales"]
        tags_payload = payload["domain_tags"]
        if type(means_payload) is not list or type(scales_payload) is not list:
            raise ValueError("feature means and scales must be arrays")
        if type(tags_payload) is not list:
            raise ValueError("feature domain_tags must be an array of strings")
        embedding_payload = payload["embedding_identity"]
        if embedding_payload is not None and not isinstance(embedding_payload, Mapping):
            raise ValueError("embedding_identity must be an object or null")
        return cls(
            continuous_means=means_payload,  # type: ignore[arg-type]
            continuous_scales=scales_payload,  # type: ignore[arg-type]
            domain_tags=tags_payload,  # type: ignore[arg-type]
            embedding_dimension=payload["embedding_dimension"],  # type: ignore[arg-type]
            embedding_identity=(
                None
                if embedding_payload is None
                else EmbeddingIdentity.from_dict(embedding_payload)
            ),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class PromptFeatureEncoder:
    """Apply a fitted schema and optional offline embedding provider."""

    schema: PromptFeatureSchema
    embedding_provider: EmbeddingProvider | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema.embedding_dimension == 0:
            if self.embedding_provider is not None:
                raise ValueError("embedding provider is not allowed for a surface-only schema")
            return
        if self.embedding_provider is None:
            raise ValueError("embedding provider is required by this feature schema")
        provider_dimension, provider_identity = _embedding_provider_contract(
            self.embedding_provider
        )
        if provider_dimension != self.schema.embedding_dimension:
            raise ValueError("embedding provider dimension does not match feature schema")
        if provider_identity != self.schema.embedding_identity:
            raise ValueError("embedding provider identity does not match feature schema")

    @classmethod
    def fit(
        cls,
        prompts: Sequence[str],
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> PromptFeatureEncoder:
        """Fit a schema without reading split labels or outcome quality."""

        if embedding_provider is None:
            embedding_dimension = 0
            embedding_identity = None
        else:
            embedding_dimension, embedding_identity = _embedding_provider_contract(
                embedding_provider
            )
        schema = PromptFeatureSchema.fit(
            prompts,
            embedding_dimension=embedding_dimension,
            embedding_identity=embedding_identity,
        )
        return cls(schema, embedding_provider)

    def transform_many(self, prompts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Encode prompts, batching embeddings exactly once when configured."""

        prompts = tuple(prompts)
        if not prompts:
            return ()
        surface_rows = tuple(self.schema.encode_surface(prompt) for prompt in prompts)
        if self.embedding_provider is None:
            embeddings: tuple[tuple[float, ...], ...] = tuple(() for _ in prompts)
        else:
            embeddings = tuple(
                tuple(float(value) for value in row)
                for row in self.embedding_provider.embed(prompts)
            )
            if len(embeddings) != len(prompts):
                raise ValueError("embedding provider returned the wrong number of rows")
        encoded = []
        for surface, embedding in zip(surface_rows, embeddings, strict=True):
            if len(embedding) != self.schema.embedding_dimension:
                raise ValueError("embedding row width does not match feature schema")
            row = (*surface, *embedding)
            if len(row) != self.schema.dimension or any(not math.isfinite(value) for value in row):
                raise ValueError("encoded feature row must have finite schema width")
            encoded.append(row)
        return tuple(encoded)

    def transform_one(self, prompt: str) -> tuple[float, ...]:
        """Encode one prompt."""

        return self.transform_many((prompt,))[0]
