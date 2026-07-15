# SPDX-License-Identifier: Apache-2.0
"""Deterministic prompt features that require no model or network access."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_PATTERN = re.compile(
    r"```|(?:^|\n)\s*(?:def|class|function|import|from|SELECT|public static)\b|"
    r"(?:console\.log|System\.out|#include|</?[a-z][^>]*>)",
    re.IGNORECASE,
)
_MATH_PATTERN = re.compile(
    r"\$[^$]+\$|\\(?:frac|sum|int|sqrt|begin)\b|[∑∫√≈≠≤≥]|\b(?:equation|proof|theorem)\b|"
    r"(?:방정식|증명|정리|미분|적분)",
    re.IGNORECASE,
)

_DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "code",
        re.compile(
            r"\b(?:python|javascript|typescript|java|rust|sql|api|debug|algorithm)\b|"
            r"(?:파이썬|자바스크립트|코드|디버그|알고리즘)",
            re.IGNORECASE,
        ),
    ),
    (
        "math",
        re.compile(
            r"\b(?:math|algebra|geometry|calculus|probability)\b|"
            r"(?:수학|대수|기하|미적분|확률)",
            re.IGNORECASE,
        ),
    ),
    (
        "science",
        re.compile(
            r"\b(?:physics|chemistry|biology|scientific)\b|(?:물리|화학|생물|과학)",
            re.IGNORECASE,
        ),
    ),
    (
        "medicine",
        re.compile(
            r"\b(?:medical|medicine|diagnosis|clinical|patient)\b|"
            r"(?:의학|진단|임상|환자)",
            re.IGNORECASE,
        ),
    ),
    (
        "law",
        re.compile(
            r"\b(?:law|legal|statute|contract|court)\b|(?:법률|법원|판례|계약서)",
            re.IGNORECASE,
        ),
    ),
    (
        "finance",
        re.compile(
            r"\b(?:finance|investment|stock|accounting|revenue)\b|"
            r"(?:금융|투자|주식|회계|매출)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class SurfaceFeatures:
    """Small, explainable features available before any model call."""

    character_count: int
    word_count: int
    line_count: int
    has_code: bool
    has_math: bool
    domain_tags: tuple[str, ...]

    def numeric_vector(self) -> tuple[float, ...]:
        """Return the stable numeric subset used by lightweight predictors."""

        return (
            float(self.character_count),
            float(self.word_count),
            float(self.line_count),
            float(self.has_code),
            float(self.has_math),
        )


def extract_surface_features(prompt: str) -> SurfaceFeatures:
    """Extract deterministic features; character count uses Unicode code points."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    has_code = bool(_CODE_PATTERN.search(prompt))
    has_math = bool(_MATH_PATTERN.search(prompt))
    tags = [name for name, pattern in _DOMAIN_PATTERNS if pattern.search(prompt)]
    if has_code and "code" not in tags:
        tags.append("code")
    if has_math and "math" not in tags:
        tags.append("math")

    return SurfaceFeatures(
        character_count=len(prompt),
        word_count=len(re.findall(r"\S+", prompt)),
        line_count=prompt.count("\n") + 1,
        has_code=has_code,
        has_math=has_math,
        domain_tags=tuple(tags or ("general",)),
    )
