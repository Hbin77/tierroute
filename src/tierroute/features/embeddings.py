# SPDX-License-Identifier: Apache-2.0
"""Offline-only embedding contracts.

The runtime API accepts a local path, never a Hub model identifier. A separate,
explicit preparation command will download model assets before evaluation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_LICENSE = "MIT"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Reproducibility identity for an offline embedding implementation and asset."""

    provider: str
    model_id: str
    revision: str
    pooling: str
    normalize: bool
    asset_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in ("provider", "model_id", "revision", "pooling"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"embedding {name} must be a non-empty string")
        if not isinstance(self.normalize, bool):
            raise TypeError("embedding normalize must be a boolean")
        if not isinstance(self.asset_manifest_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.asset_manifest_sha256
        ):
            raise ValueError("embedding asset_manifest_sha256 must be lowercase SHA-256 hex")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible identity."""

        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "asset_manifest_sha256": self.asset_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EmbeddingIdentity:
        """Load an exact version-1 identity object."""

        expected = {
            "provider",
            "model_id",
            "revision",
            "pooling",
            "normalize",
            "asset_manifest_sha256",
        }
        if set(payload) != expected:
            raise ValueError("embedding identity fields do not match schema version 1")
        if any(
            not isinstance(payload[name], str)
            for name in ("provider", "model_id", "revision", "pooling", "asset_manifest_sha256")
        ):
            raise ValueError("embedding identity text fields must be strings")
        if not isinstance(payload["normalize"], bool):
            raise ValueError("embedding identity normalize must be a boolean")
        return cls(
            provider=payload["provider"],  # type: ignore[arg-type]
            model_id=payload["model_id"],  # type: ignore[arg-type]
            revision=payload["revision"],  # type: ignore[arg-type]
            pooling=payload["pooling"],  # type: ignore[arg-type]
            normalize=payload["normalize"],  # type: ignore[arg-type]
            asset_manifest_sha256=payload["asset_manifest_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class LocalEmbeddingModel:
    """An embedding asset prepared outside the runtime path."""

    path: Path
    model_id: str = BGE_M3_MODEL_ID
    revision: str = BGE_M3_REVISION

    def validate(self) -> Path:
        """Return a resolved local directory or fail without network fallback."""

        resolved = self.path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"embedding model directory not found: {resolved}; "
                "prepare it explicitly before running offline"
            )
        return resolved


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal provider interface implemented by a local inference backend."""

    @property
    def dimension(self) -> int:
        """Embedding vector width."""
        ...

    @property
    def identity(self) -> EmbeddingIdentity:
        """Exact model, preprocessing, and local-asset identity."""
        ...

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed texts using only local assets."""
        ...
