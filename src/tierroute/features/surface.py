# SPDX-License-Identifier: Apache-2.0
"""Deterministic prompt features that require no model or network access."""

from __future__ import annotations

import re
from dataclasses import dataclass

SURFACE_FEATURE_ALGORITHM_ID = "tierroute.surface-features-v1"
SURFACE_DOMAIN_TAG_CATALOGUE = (
    "code",
    "finance",
    "general",
    "law",
    "math",
    "medicine",
    "science",
)

_CODE_PATTERN = re.compile(
    r"```|(?:^|[\r\n])[^\S\r\n]*(?:def|class|function|import|from|SELECT|public static)\b|"
    r"(?:console\.log|System\.out|#include)",
    re.IGNORECASE,
)
_HTML_TAG_START_PATTERN = re.compile(r"[a-z]", re.IGNORECASE)
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


def _contains_html_like_tag(prompt: str) -> bool:
    """Match the former ``</?[a-z][^>]*>`` alternative in linear time.

    Searching ``[^>]*`` again from every ``<a`` prefix is quadratic when a long
    prompt contains no closing ``>``. Once a syntactically valid start is found,
    one forward search is sufficient: without a later ``>``, no subsequent start
    can match either.
    """

    offset = 0
    while True:
        start = prompt.find("<", offset)
        if start < 0:
            return False
        cursor = start + 1
        if cursor < len(prompt) and prompt[cursor] == "/":
            cursor += 1
        if cursor < len(prompt) and _HTML_TAG_START_PATTERN.fullmatch(prompt[cursor]):
            return prompt.find(">", cursor + 1) >= 0
        offset = start + 1


def extract_surface_features(prompt: str) -> SurfaceFeatures:
    """Extract deterministic features; character count uses Unicode code points."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    has_code = bool(_CODE_PATTERN.search(prompt)) or _contains_html_like_tag(prompt)
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


__all__ = [
    "SURFACE_DOMAIN_TAG_CATALOGUE",
    "SURFACE_FEATURE_ALGORITHM_ID",
    "SurfaceFeatures",
    "extract_surface_features",
]
