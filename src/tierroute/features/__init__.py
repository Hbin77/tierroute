# SPDX-License-Identifier: Apache-2.0
"""Prompt feature extraction."""

from tierroute.features.embeddings import (
    BGE_M3_LICENSE,
    BGE_M3_MODEL_ID,
    BGE_M3_REVISION,
    EmbeddingIdentity,
    EmbeddingProvider,
    LocalEmbeddingModel,
)
from tierroute.features.encoding import (
    FEATURE_SCHEMA_VERSION,
    PromptFeatureEncoder,
    PromptFeatureSchema,
)
from tierroute.features.surface import (
    SURFACE_DOMAIN_TAG_CATALOGUE,
    SURFACE_FEATURE_ALGORITHM_ID,
    SurfaceFeatures,
    extract_surface_features,
)

__all__ = [
    "BGE_M3_LICENSE",
    "BGE_M3_MODEL_ID",
    "BGE_M3_REVISION",
    "FEATURE_SCHEMA_VERSION",
    "SURFACE_DOMAIN_TAG_CATALOGUE",
    "SURFACE_FEATURE_ALGORITHM_ID",
    "EmbeddingIdentity",
    "EmbeddingProvider",
    "LocalEmbeddingModel",
    "PromptFeatureEncoder",
    "PromptFeatureSchema",
    "SurfaceFeatures",
    "extract_surface_features",
]
