# SPDX-License-Identifier: Apache-2.0
"""Offline-only embedding contracts.

The runtime API accepts a local path, never a Hub model identifier. A separate,
explicit preparation command will download model assets before evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_LICENSE = "MIT"


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

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed texts using only local assets."""
        ...
