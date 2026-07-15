# SPDX-License-Identifier: Apache-2.0
"""Tests for fitted, prompt-only feature schemas and batched embeddings."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import replace

import pytest

import tierroute.features.embeddings as embedding_limits
import tierroute.features.encoding as feature_limits
from tierroute.features import (
    EmbeddingIdentity,
    PromptFeatureEncoder,
    PromptFeatureSchema,
    extract_surface_features,
)


class RecordingEmbeddingProvider:
    """Deterministic test provider that records every batch boundary."""

    dimension = 2
    identity = EmbeddingIdentity(
        provider="tierroute.tests.recording-v1",
        model_id="project-authored-test-embedding",
        revision="1",
        pooling="test-pool",
        normalize=False,
        asset_manifest_sha256="0" * 64,
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        batch = tuple(texts)
        self.calls.append(batch)
        return tuple((float(len(text)), float(text.count("a"))) for text in batch)


def test_feature_schema_uses_prompt_tags_not_split_labels() -> None:
    prompts = (
        "Debug this Python function.",
        "Prove this math theorem.",
        "Explain a clinical diagnosis.",
    )

    schema = PromptFeatureSchema.fit(prompts)
    encoded = schema.encode_surface("Interpret this legal contract.")

    assert schema.domain_tags == ("code", "math", "medicine")
    assert schema.feature_names[:5] == (
        "log1p_character_count",
        "log1p_word_count",
        "log1p_line_count",
        "has_code",
        "has_math",
    )
    assert "domain:law" not in schema.feature_names
    assert len(encoded) == schema.dimension
    assert encoded[-len(schema.domain_tags) :] == (0.0, 0.0, 0.0)


def test_feature_encoding_standardizes_log_counts_and_stays_finite() -> None:
    prompts = ("one", "two words\nsecond line")
    schema = PromptFeatureSchema.fit(prompts)
    prompt = "three words here"
    features = extract_surface_features(prompt)

    row = schema.encode_surface(prompt)

    raw = (
        math.log1p(features.character_count),
        math.log1p(features.word_count),
        math.log1p(features.line_count),
    )
    expected = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            raw,
            schema.continuous_means,
            schema.continuous_scales,
            strict=True,
        )
    )
    assert row[:3] == pytest.approx(expected)
    assert all(math.isfinite(value) for value in row)


def test_embedding_provider_is_called_once_for_a_prompt_batch() -> None:
    provider = RecordingEmbeddingProvider()
    prompts = ("alpha", "beta", "gamma")
    encoder = PromptFeatureEncoder.fit(prompts, embedding_provider=provider)

    rows = encoder.transform_many(prompts)

    assert provider.calls == [prompts]
    assert len(rows) == len(prompts)
    assert all(len(row) == encoder.schema.dimension for row in rows)
    assert rows[0][-2:] == (5.0, 2.0)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": True,
            "continuous_means": [0.0, 0.0, 0.0],
            "continuous_scales": [1.0, 1.0, 1.0],
            "domain_tags": ["general"],
            "embedding_dimension": 0,
            "embedding_identity": None,
        },
        {
            "schema_version": 1,
            "continuous_means": [False, 0.0, 0.0],
            "continuous_scales": [1.0, 1.0, 1.0],
            "domain_tags": ["general"],
            "embedding_dimension": 0,
            "embedding_identity": None,
        },
    ],
)
def test_feature_schema_rejects_boolean_numeric_fields(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        PromptFeatureSchema.from_dict(payload)


def test_feature_schema_and_embedding_metadata_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = PromptFeatureSchema.fit(("Debug code.", "Prove math."))

    monkeypatch.setattr(feature_limits, "MAX_FEATURE_DOMAIN_TAGS", 1)
    with pytest.raises(ValueError, match="domain_tags exceeds"):
        replace(schema)

    monkeypatch.setattr(feature_limits, "MAX_FEATURE_DOMAIN_TAGS", 4_096)
    monkeypatch.setattr(feature_limits, "MAX_FEATURE_DIMENSION", schema.dimension - 1)
    with pytest.raises(ValueError, match="feature dimension exceeds"):
        replace(schema)

    monkeypatch.setattr(feature_limits, "MAX_FEATURE_DIMENSION", 16_384)
    monkeypatch.setattr(feature_limits, "MAX_FEATURE_METADATA_TEXT_BYTES", 1)
    with pytest.raises(ValueError, match="feature metadata limit"):
        replace(schema, domain_tags=("ab",))
    with pytest.raises(ValueError, match="valid Unicode"):
        replace(schema, domain_tags=("\ud800",))

    identity = RecordingEmbeddingProvider.identity
    monkeypatch.setattr(embedding_limits, "MAX_EMBEDDING_IDENTITY_TEXT_BYTES", 1)
    with pytest.raises(ValueError, match="metadata limit"):
        replace(identity)
    with pytest.raises(ValueError, match="valid Unicode"):
        replace(identity, provider="\ud800")


def test_feature_schema_snapshots_stateful_sequences_before_validation() -> None:
    class FlippingNumbers(list[float]):
        def __init__(self, first: list[float], later: list[float]) -> None:
            super().__init__()
            self.first = first
            self.later = later
            self.iterations = 0

        def __iter__(self) -> Iterator[float]:
            self.iterations += 1
            return iter(self.first if self.iterations == 1 else self.later)

    means = FlippingNumbers([0.0, 0.0, 0.0], [math.nan, math.nan, math.nan])
    scales = FlippingNumbers([1.0, 1.0, 1.0], [math.nan, math.nan, math.nan])
    schema = PromptFeatureSchema(
        continuous_means=means,  # type: ignore[arg-type]
        continuous_scales=scales,  # type: ignore[arg-type]
        domain_tags=("general",),
    )

    assert means.iterations == 1
    assert scales.iterations == 1
    assert schema.continuous_means == (0.0, 0.0, 0.0)
    assert schema.continuous_scales == (1.0, 1.0, 1.0)


def test_feature_schema_rejects_primitive_subclass_dimension() -> None:
    class LyingDimension(int):
        def __radd__(self, other: object) -> int:
            del other
            return 0

    with pytest.raises(TypeError, match="embedding_dimension must be an integer"):
        PromptFeatureSchema(
            continuous_means=(0.0, 0.0, 0.0),
            continuous_scales=(1.0, 1.0, 1.0),
            domain_tags=("general",),
            embedding_dimension=LyingDimension(20_000),
            embedding_identity=RecordingEmbeddingProvider.identity,
        )


def test_feature_schema_dimension_uses_the_bounded_tag_snapshot() -> None:
    class LyingTags(list[str]):
        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="feature dimension exceeds"):
        PromptFeatureSchema(
            continuous_means=(0.0, 0.0, 0.0),
            continuous_scales=(1.0, 1.0, 1.0),
            domain_tags=LyingTags(["general"]),  # type: ignore[arg-type]
            embedding_dimension=feature_limits.MAX_FEATURE_DIMENSION - 5,
            embedding_identity=RecordingEmbeddingProvider.identity,
        )


def test_feature_schema_from_dict_defers_domain_copy_to_bounded_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = PromptFeatureSchema.fit(("Debug code.", "Prove math."))
    payload = schema.to_dict()
    tags = payload["domain_tags"]
    original_snapshot = feature_limits._bounded_sequence_snapshot

    def recording_snapshot(
        value: object,
        context: str,
        *,
        max_items: int,
    ) -> tuple[object, ...]:
        if context == "domain_tags":
            assert value is tags
        return original_snapshot(value, context, max_items=max_items)

    monkeypatch.setattr(feature_limits, "_bounded_sequence_snapshot", recording_snapshot)
    assert PromptFeatureSchema.from_dict(payload) == schema


def test_embedding_shape_and_non_finite_values_fail_closed() -> None:
    class BrokenProvider:
        dimension = 2
        identity = RecordingEmbeddingProvider.identity

        def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            return ((math.nan, 1.0),)

    encoder = PromptFeatureEncoder.fit(("prompt",), embedding_provider=BrokenProvider())

    with pytest.raises(ValueError, match="finite"):
        encoder.transform_many(("prompt",))


def test_embedding_identity_is_serialized_and_must_match() -> None:
    provider = RecordingEmbeddingProvider()
    schema = PromptFeatureEncoder.fit(("prompt",), embedding_provider=provider).schema

    assert PromptFeatureSchema.from_dict(schema.to_dict()) == schema

    class WrongProvider(RecordingEmbeddingProvider):
        identity = EmbeddingIdentity(
            provider="tierroute.tests.recording-v2",
            model_id="project-authored-test-embedding",
            revision="2",
            pooling="different-pool",
            normalize=True,
            asset_manifest_sha256="1" * 64,
        )

    with pytest.raises(ValueError, match="identity"):
        PromptFeatureEncoder(schema, WrongProvider())
