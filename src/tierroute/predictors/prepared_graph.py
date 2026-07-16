# SPDX-License-Identifier: Apache-2.0
"""Immutable graph and resource contract for a future prepared LODO session.

This module enumerates reusable training-domain subsets and raw-score blocks only.
It does not encode features, fit predictors, execute native code, or change the
single-problem ridge protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

PREPARED_GRAPH_ALGORITHM_ID = "tierroute.prepared-nested-lodo-graph-v1"

MIN_PREPARED_DOMAINS = 4
MAX_PREPARED_DOMAINS = 64
MAX_PREPARED_EXAMPLES = 1_000_000
MAX_PREPARED_FEATURES = 4_096
MAX_PREPARED_TARGETS = 256
MAX_PREPARED_TRAINING_SUBSETS = 65_536
MAX_PREPARED_SCORE_BLOCKS = 131_072
MAX_PREPARED_SCORE_ROW_MEMBERSHIPS = 16_000_000
MAX_PREPARED_RESIDENT_BYTES = 2 * 1024 * 1024 * 1024
MAX_PREPARED_WORK_UNITS = 200_000_000_000
MAX_PREPARED_DOMAIN_UTF8_BYTES = 4 * 1024
MAX_PREPARED_CATALOGUE_UTF8_BYTES = 256 * 1024

_F64_BYTES = 8


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exact_positive_int(value: object, name: str) -> int:
    normalized = _exact_nonnegative_int(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True, slots=True)
class PreparedTrainingSubset:
    """One canonical training-domain subset and its exact row count."""

    domain_indices: tuple[int, ...]
    row_count: int

    def __post_init__(self) -> None:
        if type(self.domain_indices) is not tuple:
            raise TypeError("domain_indices must be an exact tuple")
        if not self.domain_indices:
            raise ValueError("domain_indices must not be empty")
        if any(type(index) is not int or index < 0 for index in self.domain_indices):
            raise ValueError("domain_indices must contain non-negative exact integers")
        if tuple(sorted(self.domain_indices)) != self.domain_indices or len(
            set(self.domain_indices)
        ) != len(self.domain_indices):
            raise ValueError("domain_indices must be strictly increasing")
        _exact_positive_int(self.row_count, "row_count")

    @property
    def domain_mask(self) -> int:
        """Return the derived bit mask; callers cannot provide a conflicting mask."""

        return sum(1 << index for index in self.domain_indices)


@dataclass(frozen=True, slots=True)
class PreparedScoreBlock:
    """Raw scores from one training subset over one excluded domain."""

    training_subset_index: int
    scored_domain_index: int
    row_count: int

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.training_subset_index, "training_subset_index")
        _exact_nonnegative_int(self.scored_domain_index, "scored_domain_index")
        _exact_positive_int(self.row_count, "row_count")


@dataclass(frozen=True, slots=True)
class PreparedNestedLodoWorkEstimate:
    """Closed-form counts, conservative bytes, and deterministic work units."""

    domain_count: int
    example_count: int
    feature_count: int
    target_count: int
    logical_calibrated_fit_count: int
    logical_base_fit_count: int
    logical_training_row_visits: int
    logical_raw_score_row_visits: int
    unique_training_subset_count: int
    unique_score_block_count: int
    score_row_membership_multiplier: int
    score_row_memberships: int
    scalar_score_count: int
    dot_product_positions: int
    feature_cache_bytes: int
    target_cache_bytes: int
    domain_statistics_bytes: int
    coefficient_cache_bytes: int
    raw_score_cache_bytes: int
    solve_workspace_bytes: int
    resident_bytes: int
    statistics_work_units: int
    solve_work_units: int
    score_work_units: int
    total_work_units: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _exact_positive_int(getattr(self, name), name)
        expected = _estimate_values(
            self.domain_count,
            self.example_count,
            self.feature_count,
            self.target_count,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match the prepared graph formula")


@dataclass(frozen=True, slots=True)
class PreparedNestedLodoPlan:
    """Canonical graph snapshot for nested evaluation, not a runnable session."""

    domains: tuple[str, ...]
    domain_example_counts: tuple[int, ...]
    feature_count: int
    target_count: int
    training_subsets: tuple[PreparedTrainingSubset, ...]
    score_blocks: tuple[PreparedScoreBlock, ...]
    work: PreparedNestedLodoWorkEstimate
    algorithm_id: str = field(default=PREPARED_GRAPH_ALGORITHM_ID, init=False)

    def __post_init__(self) -> None:
        canonical_domains, canonical_counts = _snapshot_inputs(
            self.domains,
            self.domain_example_counts,
            feature_count=self.feature_count,
            target_count=self.target_count,
        )
        if self.domains != canonical_domains or self.domain_example_counts != canonical_counts:
            raise ValueError("prepared plan domain catalogue must be canonical")
        if type(self.training_subsets) is not tuple or not all(
            type(node) is PreparedTrainingSubset for node in self.training_subsets
        ):
            raise TypeError("training_subsets must be an exact tuple of prepared subsets")
        if type(self.score_blocks) is not tuple or not all(
            type(node) is PreparedScoreBlock for node in self.score_blocks
        ):
            raise TypeError("score_blocks must be an exact tuple of prepared score blocks")
        if type(self.work) is not PreparedNestedLodoWorkEstimate:
            raise TypeError("work must be a PreparedNestedLodoWorkEstimate")
        expected_work = _preflight_counts(
            canonical_counts,
            feature_count=self.feature_count,
            target_count=self.target_count,
        )
        if self.work != expected_work:
            raise ValueError("work does not match the canonical prepared graph")
        expected_subsets = _enumerate_training_subsets(canonical_counts)
        if self.training_subsets != expected_subsets:
            raise ValueError("training_subsets do not match the canonical prepared graph")
        expected_blocks = _enumerate_score_blocks(expected_subsets, canonical_counts)
        if self.score_blocks != expected_blocks:
            raise ValueError("score_blocks do not match the canonical prepared graph")


def _snapshot_inputs(
    domains: tuple[str, ...],
    domain_example_counts: tuple[int, ...],
    *,
    feature_count: int,
    target_count: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if type(domains) is not tuple:
        raise TypeError("domains must be an exact tuple")
    if type(domain_example_counts) is not tuple:
        raise TypeError("domain_example_counts must be an exact tuple")
    if len(domains) != len(domain_example_counts):
        raise ValueError("domains and domain_example_counts must have equal lengths")
    if not MIN_PREPARED_DOMAINS <= len(domains) <= MAX_PREPARED_DOMAINS:
        raise ValueError("domain count is outside the reviewed prepared-graph range")

    encoded_total = 0
    validated_domains: list[str] = []
    for domain in domains:
        if type(domain) is not str:
            raise TypeError("every domain must be an exact string")
        if not domain.strip():
            raise ValueError("every domain must contain non-whitespace text")
        try:
            encoded = domain.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("every domain must be valid UTF-8 text") from error
        if len(encoded) > MAX_PREPARED_DOMAIN_UTF8_BYTES:
            raise ValueError("a domain exceeds the reviewed UTF-8 byte limit")
        encoded_total += len(encoded)
        if encoded_total > MAX_PREPARED_CATALOGUE_UTF8_BYTES:
            raise ValueError("domain catalogue exceeds the reviewed UTF-8 byte limit")
        validated_domains.append(domain)
    if len(set(validated_domains)) != len(validated_domains):
        raise ValueError("domains must be unique")

    validated_counts: list[int] = []
    total_examples = 0
    for count in domain_example_counts:
        count = _exact_positive_int(count, "domain example count")
        if count > MAX_PREPARED_EXAMPLES:
            raise ValueError("a domain example count exceeds the reviewed limit")
        total_examples += count
        if total_examples > MAX_PREPARED_EXAMPLES:
            raise ValueError("total example count exceeds the reviewed limit")
        validated_counts.append(count)

    feature_count = _exact_positive_int(feature_count, "feature_count")
    target_count = _exact_positive_int(target_count, "target_count")
    if feature_count > MAX_PREPARED_FEATURES:
        raise ValueError("feature_count exceeds the reviewed prepared-graph limit")
    if target_count > MAX_PREPARED_TARGETS:
        raise ValueError("target_count exceeds the reviewed prepared-graph limit")

    pairs = sorted(zip(validated_domains, validated_counts, strict=True), key=lambda pair: pair[0])
    return tuple(domain for domain, _ in pairs), tuple(count for _, count in pairs)


def _estimate_values(
    domain_count: int,
    example_count: int,
    feature_count: int,
    target_count: int,
) -> dict[str, int]:
    d_count = domain_count
    n = example_count
    d = feature_count
    m = target_count
    subset_count = math.comb(d_count, 3) + math.comb(d_count, 2) + d_count
    block_count = 3 * math.comb(d_count, 3) + 2 * math.comb(d_count, 2) + d_count
    membership_multiplier = math.comb(d_count, 2) + 1
    memberships = n * membership_multiplier
    feature_cache_bytes = _F64_BYTES * n * d
    target_cache_bytes = _F64_BYTES * n * m
    domain_statistics_bytes = _F64_BYTES * d_count * (1 + d + d * (d + 1) // 2 + m + d * m)
    coefficient_cache_bytes = _F64_BYTES * subset_count * m * (d + 1)
    raw_score_cache_bytes = _F64_BYTES * memberships * m
    solve_workspace_bytes = _F64_BYTES * (2 * d * d + 2 * m * d + 2 * d + 3 * m)
    resident_bytes = (
        feature_cache_bytes
        + target_cache_bytes
        + domain_statistics_bytes
        + coefficient_cache_bytes
        + raw_score_cache_bytes
        + solve_workspace_bytes
    )
    statistics_work_units = 3 * n * (d + m) + n * d * (d + 1) // 2 + n * d * m
    solve_work_units = subset_count * (d**3 + 2 * m * d * d + m * d)
    score_work_units = memberships * m * d
    return {
        "domain_count": d_count,
        "example_count": n,
        "feature_count": d,
        "target_count": m,
        "logical_calibrated_fit_count": d_count**2,
        "logical_base_fit_count": d_count * ((d_count - 1) ** 2 + d_count),
        "logical_training_row_visits": n * (d_count - 1) * ((d_count - 2) ** 2 + (d_count - 1)),
        "logical_raw_score_row_visits": n * (d_count * (d_count - 1) + 1),
        "unique_training_subset_count": subset_count,
        "unique_score_block_count": block_count,
        "score_row_membership_multiplier": membership_multiplier,
        "score_row_memberships": memberships,
        "scalar_score_count": memberships * m,
        "dot_product_positions": memberships * m * d,
        "feature_cache_bytes": feature_cache_bytes,
        "target_cache_bytes": target_cache_bytes,
        "domain_statistics_bytes": domain_statistics_bytes,
        "coefficient_cache_bytes": coefficient_cache_bytes,
        "raw_score_cache_bytes": raw_score_cache_bytes,
        "solve_workspace_bytes": solve_workspace_bytes,
        "resident_bytes": resident_bytes,
        "statistics_work_units": statistics_work_units,
        "solve_work_units": solve_work_units,
        "score_work_units": score_work_units,
        "total_work_units": statistics_work_units + solve_work_units + score_work_units,
    }


def _preflight_counts(
    domain_example_counts: tuple[int, ...],
    *,
    feature_count: int,
    target_count: int,
) -> PreparedNestedLodoWorkEstimate:
    values = _estimate_values(
        len(domain_example_counts),
        sum(domain_example_counts),
        feature_count,
        target_count,
    )
    if values["unique_training_subset_count"] > MAX_PREPARED_TRAINING_SUBSETS:
        raise ValueError("prepared training-subset count exceeds the reviewed limit")
    if values["unique_score_block_count"] > MAX_PREPARED_SCORE_BLOCKS:
        raise ValueError("prepared score-block count exceeds the reviewed limit")
    if values["score_row_memberships"] > MAX_PREPARED_SCORE_ROW_MEMBERSHIPS:
        raise ValueError("prepared score-row memberships exceed the reviewed limit")
    if values["resident_bytes"] > MAX_PREPARED_RESIDENT_BYTES:
        raise ValueError("prepared resident-byte estimate exceeds the reviewed limit")
    if values["total_work_units"] > MAX_PREPARED_WORK_UNITS:
        raise ValueError("prepared work estimate exceeds the reviewed limit")
    return PreparedNestedLodoWorkEstimate(**values)


def _enumerate_training_subsets(
    domain_example_counts: tuple[int, ...],
) -> tuple[PreparedTrainingSubset, ...]:
    domain_count = len(domain_example_counts)
    full_mask = (1 << domain_count) - 1
    nodes: list[PreparedTrainingSubset] = []
    for omitted_count in (3, 2, 1):
        for omitted in combinations(range(domain_count), omitted_count):
            omitted_mask = sum(1 << index for index in omitted)
            training_mask = full_mask ^ omitted_mask
            indices = tuple(index for index in range(domain_count) if training_mask & (1 << index))
            nodes.append(
                PreparedTrainingSubset(
                    domain_indices=indices,
                    row_count=sum(domain_example_counts[index] for index in indices),
                )
            )
    return tuple(nodes)


def _enumerate_score_blocks(
    training_subsets: tuple[PreparedTrainingSubset, ...],
    domain_example_counts: tuple[int, ...],
) -> tuple[PreparedScoreBlock, ...]:
    nodes: list[PreparedScoreBlock] = []
    for subset_index, subset in enumerate(training_subsets):
        for domain_index, row_count in enumerate(domain_example_counts):
            if not subset.domain_mask & (1 << domain_index):
                nodes.append(
                    PreparedScoreBlock(
                        training_subset_index=subset_index,
                        scored_domain_index=domain_index,
                        row_count=row_count,
                    )
                )
    return tuple(nodes)


def build_prepared_nested_lodo_plan(
    domains: tuple[str, ...],
    domain_example_counts: tuple[int, ...],
    *,
    feature_count: int,
    target_count: int,
) -> PreparedNestedLodoPlan:
    """Canonicalize, preflight, then enumerate the unique nested-LODO graph."""

    canonical_domains, canonical_counts = _snapshot_inputs(
        domains,
        domain_example_counts,
        feature_count=feature_count,
        target_count=target_count,
    )
    work = _preflight_counts(
        canonical_counts,
        feature_count=feature_count,
        target_count=target_count,
    )
    training_subsets = _enumerate_training_subsets(canonical_counts)
    score_blocks = _enumerate_score_blocks(training_subsets, canonical_counts)
    return PreparedNestedLodoPlan(
        domains=canonical_domains,
        domain_example_counts=canonical_counts,
        feature_count=feature_count,
        target_count=target_count,
        training_subsets=training_subsets,
        score_blocks=score_blocks,
        work=work,
    )


__all__ = [
    "PREPARED_GRAPH_ALGORITHM_ID",
    "PreparedNestedLodoPlan",
    "PreparedNestedLodoWorkEstimate",
    "PreparedScoreBlock",
    "PreparedTrainingSubset",
    "build_prepared_nested_lodo_plan",
]
