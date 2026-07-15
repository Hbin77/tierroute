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
_CONTINUOUS_NAMES = (
    "log1p_character_count",
    "log1p_word_count",
    "log1p_line_count",
)
_BINARY_NAMES = ("has_code", "has_math")


def _continuous_values(prompt: str) -> tuple[float, float, float]:
    features = extract_surface_features(prompt)
    return (
        math.log1p(features.character_count),
        math.log1p(features.word_count),
        math.log1p(features.line_count),
    )


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
        if len(self.continuous_means) != 3 or len(self.continuous_scales) != 3:
            raise ValueError("feature schema requires three continuous means and scales")
        if any(
            isinstance(value, bool) or not math.isfinite(value) for value in self.continuous_means
        ):
            raise ValueError("continuous means must be finite")
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
            for value in self.continuous_scales
        ):
            raise ValueError("continuous scales must be finite and positive")
        if self.domain_tags != tuple(sorted(set(self.domain_tags))):
            raise ValueError("domain_tags must be sorted and unique")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.domain_tags):
            raise ValueError("domain_tags must be non-empty strings")
        if isinstance(self.embedding_dimension, bool) or not isinstance(
            self.embedding_dimension, int
        ):
            raise TypeError("embedding_dimension must be an integer")
        if self.embedding_dimension < 0:
            raise ValueError("embedding_dimension must be non-negative")
        if self.embedding_identity is not None and not isinstance(
            self.embedding_identity, EmbeddingIdentity
        ):
            raise TypeError("embedding_identity must be an EmbeddingIdentity or None")
        if (self.embedding_dimension == 0) != (self.embedding_identity is None):
            raise ValueError(
                "embedding identity must be present exactly when embedding_dimension is positive"
            )

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
        if set(payload) != expected:
            raise ValueError("feature schema fields do not match schema version 1")
        means_payload = payload["continuous_means"]
        scales_payload = payload["continuous_scales"]
        tags_payload = payload["domain_tags"]
        if not isinstance(means_payload, list) or not isinstance(scales_payload, list):
            raise ValueError("feature means and scales must be arrays")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (*means_payload, *scales_payload)
        ):
            raise ValueError("feature means and scales must be JSON numbers")
        if not isinstance(tags_payload, list) or any(
            not isinstance(tag, str) for tag in tags_payload
        ):
            raise ValueError("feature domain_tags must be an array of strings")
        embedding_payload = payload["embedding_identity"]
        if embedding_payload is not None and not isinstance(embedding_payload, Mapping):
            raise ValueError("embedding_identity must be an object or null")
        means = tuple(float(value) for value in means_payload)
        scales = tuple(float(value) for value in scales_payload)
        return cls(
            continuous_means=means,  # type: ignore[arg-type]
            continuous_scales=scales,  # type: ignore[arg-type]
            domain_tags=tuple(tags_payload),
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
        if self.embedding_provider.dimension != self.schema.embedding_dimension:
            raise ValueError("embedding provider dimension does not match feature schema")
        if self.embedding_provider.identity != self.schema.embedding_identity:
            raise ValueError("embedding provider identity does not match feature schema")

    @classmethod
    def fit(
        cls,
        prompts: Sequence[str],
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> PromptFeatureEncoder:
        """Fit a schema without reading split labels or outcome quality."""

        embedding_dimension = 0 if embedding_provider is None else embedding_provider.dimension
        embedding_identity = None if embedding_provider is None else embedding_provider.identity
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
