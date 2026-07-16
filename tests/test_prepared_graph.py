# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, replace
from itertools import combinations
from math import comb

import pytest

import tierroute.predictors.prepared_graph as prepared_graph_module
from tierroute.predictors.prepared_graph import (
    PREPARED_GRAPH_ALGORITHM_ID,
    PreparedNestedLodoPlan,
    PreparedNestedLodoWorkEstimate,
    PreparedScoreBlock,
    PreparedTrainingSubset,
    build_prepared_nested_lodo_plan,
)


def _mask(indices: tuple[int, ...]) -> int:
    return sum(1 << index for index in indices)


def _expected_masks(domain_count: int) -> tuple[int, ...]:
    full_mask = (1 << domain_count) - 1
    return tuple(
        full_mask ^ _mask(omitted)
        for omitted_count in (3, 2, 1)
        for omitted in combinations(range(domain_count), omitted_count)
    )


def _expected_score_keys(domain_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (training_mask, scored_domain)
        for training_mask in _expected_masks(domain_count)
        for scored_domain in range(domain_count)
        if not training_mask & (1 << scored_domain)
    )


def _logical_origins(domain_count: int) -> Counter[tuple[int, int]]:
    full = frozenset(range(domain_count))
    origins: Counter[tuple[int, int]] = Counter()
    for outer_held_out in range(domain_count):
        outer_training = full - {outer_held_out}
        for lambda_held_out in (*sorted(outer_training), None):
            calibrated = (
                outer_training if lambda_held_out is None else outer_training - {lambda_held_out}
            )
            for calibration_held_out in (*sorted(calibrated), None):
                training = (
                    calibrated
                    if calibration_held_out is None
                    else calibrated - {calibration_held_out}
                )
                scored = (
                    calibration_held_out
                    if calibration_held_out is not None
                    else lambda_held_out
                    if lambda_held_out is not None
                    else outer_held_out
                )
                origins[(_mask(tuple(sorted(training))), scored)] += 1
    return origins


def _plan(domain_count: int) -> PreparedNestedLodoPlan:
    return build_prepared_nested_lodo_plan(
        tuple(f"domain-{index}" for index in range(domain_count)),
        (1,) * domain_count,
        feature_count=1,
        target_count=1,
    )


@pytest.mark.parametrize("domain_count", range(4, 8))
def test_graph_matches_combinatorial_and_logical_call_oracles(domain_count: int) -> None:
    plan = _plan(domain_count)
    masks = tuple(subset.domain_mask for subset in plan.training_subsets)
    score_keys = tuple(
        (
            plan.training_subsets[block.training_subset_index].domain_mask,
            block.scored_domain_index,
        )
        for block in plan.score_blocks
    )

    assert masks == _expected_masks(domain_count)
    assert score_keys == _expected_score_keys(domain_count)
    assert len(set(masks)) == len(masks)
    assert len(set(score_keys)) == len(score_keys)

    expected_subsets = comb(domain_count, 3) + comb(domain_count, 2) + domain_count
    expected_blocks = 3 * comb(domain_count, 3) + 2 * comb(domain_count, 2) + domain_count
    expected_multiplier = comb(domain_count, 2) + 1
    assert plan.work.unique_training_subset_count == expected_subsets
    assert plan.work.unique_score_block_count == expected_blocks
    assert plan.work.score_row_membership_multiplier == expected_multiplier
    assert plan.work.score_row_memberships == domain_count * expected_multiplier

    logical = _logical_origins(domain_count)
    assert set(logical) == set(score_keys)
    assert sum(logical.values()) == domain_count * ((domain_count - 1) ** 2 + domain_count)
    for (training_mask, _), multiplicity in logical.items():
        omitted_count = domain_count - training_mask.bit_count()
        assert multiplicity == (1 if omitted_count == 1 else 2)


def test_four_domain_literal_order_and_counts() -> None:
    plan = _plan(4)

    assert tuple(subset.domain_mask for subset in plan.training_subsets) == (
        8,
        4,
        2,
        1,
        12,
        10,
        6,
        9,
        5,
        3,
        14,
        13,
        11,
        7,
    )
    assert plan.work.logical_calibrated_fit_count == 16
    assert plan.work.logical_base_fit_count == 52
    assert plan.work.unique_training_subset_count == 14
    assert plan.work.unique_score_block_count == 28
    assert plan.work.score_row_membership_multiplier == 7


def test_routerbench_shape_has_exact_reviewed_seven_domain_estimate() -> None:
    plan = build_prepared_nested_lodo_plan(
        tuple(f"domain-{index}" for index in range(7)),
        (4_968, 4_968, 4_968, 4_968, 4_968, 4_968, 4_970),
        feature_count=1_036,
        target_count=11,
    )
    work = plan.work

    assert plan.algorithm_id == PREPARED_GRAPH_ALGORITHM_ID
    assert len(plan.training_subsets) == work.unique_training_subset_count == 63
    assert len(plan.score_blocks) == work.unique_score_block_count == 154
    assert work.logical_calibrated_fit_count == 49
    assert work.logical_base_fit_count == 301
    assert work.logical_training_row_visits == 6_468_708
    assert work.logical_raw_score_row_visits == 1_495_454
    assert work.score_row_membership_multiplier == 22
    assert work.score_row_memberships == 765_116
    assert work.scalar_score_count == 8_416_276
    assert work.dot_product_positions == 8_719_261_936
    assert work.feature_cache_bytes == 288_240_064
    assert work.target_cache_bytes == 3_060_464
    assert work.domain_statistics_bytes == 30_778_160
    assert work.coefficient_cache_bytes == 5_749_128
    assert work.raw_score_cache_bytes == 67_330_208
    assert work.solve_workspace_bytes == 17_371_912
    assert work.modeled_buffer_bytes == 412_529_936
    assert work.statistics_work_units == 19_187_126_934
    assert work.solve_work_units == 71_540_189_532
    assert work.score_work_units == 8_719_261_936
    assert work.total_numeric_work_units == 99_446_578_402


def test_canonicalization_keeps_uneven_domain_counts_paired() -> None:
    domains = ("z", "a", "m", "é", "가", "x", "b")
    counts = (2, 3, 5, 7, 11, 13, 17)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=2,
        target_count=3,
    )
    reversed_plan = build_prepared_nested_lodo_plan(
        tuple(reversed(domains)),
        tuple(reversed(counts)),
        feature_count=2,
        target_count=3,
    )

    expected_pairs = tuple(sorted(zip(domains, counts, strict=True)))
    assert tuple(zip(plan.domains, plan.domain_example_counts, strict=True)) == expected_pairs
    assert reversed_plan == plan
    assert sum(block.row_count for block in plan.score_blocks) == 22 * sum(counts)
    for domain_index, count in enumerate(plan.domain_example_counts):
        assert (
            sum(
                block.row_count
                for block in plan.score_blocks
                if block.scored_domain_index == domain_index
            )
            == 22 * count
        )
    assert all(
        block.row_count == plan.domain_example_counts[block.scored_domain_index]
        for block in plan.score_blocks
    )
    assert all(
        subset.row_count
        == sum(plan.domain_example_counts[index] for index in subset.domain_indices)
        for subset in plan.training_subsets
    )


def test_asymmetric_shape_resource_formulas_have_an_independent_oracle() -> None:
    plan = build_prepared_nested_lodo_plan(
        ("delta", "alpha", "charlie", "bravo", "echo"),
        (2, 3, 5, 7, 11),
        feature_count=4,
        target_count=3,
    )
    d_count, n, d, m = 5, 28, 4, 3
    subsets = comb(d_count, 3) + comb(d_count, 2) + d_count
    memberships = n * (comb(d_count, 2) + 1)
    components = (
        8 * n * d,
        8 * n * m,
        8 * d_count * (1 + d + d * (d + 1) // 2 + m + d * m),
        8 * subsets * m * (d + 1),
        8 * memberships * m,
        8 * (2 * d * d + 2 * m * d + 2 * d + 3 * m),
    )
    statistics_work = 3 * n * (d + m) + n * d * (d + 1) // 2 + n * d * m
    solve_work = subsets * (d**3 + 2 * m * d * d + m * d)
    score_work = memberships * m * d

    assert (
        plan.work.feature_cache_bytes,
        plan.work.target_cache_bytes,
        plan.work.domain_statistics_bytes,
        plan.work.coefficient_cache_bytes,
        plan.work.raw_score_cache_bytes,
        plan.work.solve_workspace_bytes,
    ) == components
    assert plan.work.modeled_buffer_bytes == sum(components)
    assert plan.work.statistics_work_units == statistics_work
    assert plan.work.solve_work_units == solve_work
    assert plan.work.score_work_units == score_work
    assert plan.work.total_numeric_work_units == statistics_work + solve_work + score_work


@pytest.mark.parametrize(
    ("domains", "counts", "expected_error"),
    [
        (["a", "b", "c", "d"], (1, 1, 1, 1), TypeError),
        (("a", "b", "c", "d"), [1, 1, 1, 1], TypeError),
        (("a", "b", "c"), (1, 1, 1), ValueError),
        (("a", "b", "c", "d"), (1, 1, 1), ValueError),
        (("a", "b", "c", "a"), (1, 1, 1, 1), ValueError),
        (("a", "b", "c", " "), (1, 1, 1, 1), ValueError),
        (("a", "b", "c", "\ud800"), (1, 1, 1, 1), ValueError),
        (("a", "b", "c", object()), (1, 1, 1, 1), TypeError),
        (("a", "b", "c", []), (1, 1, 1, 1), TypeError),
        (("a", "b", "c", "d"), (1, 1, 1, True), TypeError),
        (("a", "b", "c", "d"), (1, 1, 1, 0), ValueError),
        (("a", "b", "c", "d"), (1, 1, 1, -1), ValueError),
    ],
)
def test_invalid_catalogues_fail_closed(
    domains: object,
    counts: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        build_prepared_nested_lodo_plan(  # type: ignore[arg-type]
            domains,
            counts,
            feature_count=1,
            target_count=1,
        )


@pytest.mark.parametrize(
    ("feature_count", "target_count", "expected_error"),
    [
        (True, 1, TypeError),
        (1, False, TypeError),
        (1.0, 1, TypeError),
        (1, 1.0, TypeError),
        (0, 1, ValueError),
        (1, 0, ValueError),
        (4_097, 1, ValueError),
        (1, 257, ValueError),
    ],
)
def test_dimensions_require_bounded_exact_integers(
    feature_count: object,
    target_count: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        build_prepared_nested_lodo_plan(
            ("a", "b", "c", "d"),
            (1, 1, 1, 1),
            feature_count=feature_count,  # type: ignore[arg-type]
            target_count=target_count,  # type: ignore[arg-type]
        )


def test_primitive_subclasses_and_oversized_values_are_rejected_without_coercion() -> None:
    class TupleSubclass(tuple):
        pass

    class StringSubclass(str):
        pass

    class IntSubclass(int):
        pass

    with pytest.raises(TypeError):
        build_prepared_nested_lodo_plan(
            TupleSubclass(("a", "b", "c", "d")),
            (1, 1, 1, 1),
            feature_count=1,
            target_count=1,
        )
    with pytest.raises(TypeError):
        build_prepared_nested_lodo_plan(
            (StringSubclass("a"), "b", "c", "d"),
            (1, 1, 1, 1),
            feature_count=1,
            target_count=1,
        )
    with pytest.raises(TypeError):
        build_prepared_nested_lodo_plan(
            ("a", "b", "c", "d"),
            (IntSubclass(1), 1, 1, 1),
            feature_count=1,
            target_count=1,
        )
    with pytest.raises(ValueError, match="domain example count exceeds"):
        build_prepared_nested_lodo_plan(
            ("a", "b", "c", "d"),
            (10**10_000, 1, 1, 1),
            feature_count=1,
            target_count=1,
        )
    with pytest.raises(ValueError, match="UTF-8 byte limit"):
        build_prepared_nested_lodo_plan(
            ("a" * 4_097, "b", "c", "d"),
            (1, 1, 1, 1),
            feature_count=1,
            target_count=1,
        )


@pytest.mark.parametrize(
    ("limit_name", "work_field", "message"),
    [
        (
            "MAX_PREPARED_TRAINING_SUBSETS",
            "unique_training_subset_count",
            "training-subset count",
        ),
        ("MAX_PREPARED_SCORE_BLOCKS", "unique_score_block_count", "score-block count"),
        (
            "MAX_PREPARED_SCORE_ROW_MEMBERSHIPS",
            "score_row_memberships",
            "score-row memberships",
        ),
        (
            "MAX_PREPARED_MODELED_BUFFER_BYTES",
            "modeled_buffer_bytes",
            "modeled-buffer estimate",
        ),
        (
            "MAX_PREPARED_NUMERIC_WORK_UNITS",
            "total_numeric_work_units",
            "numeric-work estimate",
        ),
    ],
)
def test_derived_resource_boundaries_and_pre_enumeration_refusal(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    work_field: str,
    message: str,
) -> None:
    def unexpected_enumeration(_: object) -> object:
        raise AssertionError("graph enumeration must not run before resource refusal")

    admitted = _plan(4)
    exact_value = getattr(admitted.work, work_field)
    monkeypatch.setattr(prepared_graph_module, limit_name, exact_value)
    assert _plan(4).work == admitted.work

    monkeypatch.setattr(prepared_graph_module, limit_name, exact_value - 1)
    monkeypatch.setattr(
        prepared_graph_module,
        "_enumerate_training_subsets",
        unexpected_enumeration,
    )
    with pytest.raises(ValueError, match=message):
        _plan(4)


def test_domain_limit_refusal_precedes_graph_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_enumeration(_: object) -> object:
        raise AssertionError("graph enumeration must not run before domain refusal")

    monkeypatch.setattr(
        prepared_graph_module,
        "_enumerate_training_subsets",
        unexpected_enumeration,
    )
    with pytest.raises(ValueError, match="domain count"):
        _plan(8)


def test_total_example_boundary_is_checked_without_expanding_rows() -> None:
    admitted = build_prepared_nested_lodo_plan(
        ("a", "b", "c", "d"),
        (999_997, 1, 1, 1),
        feature_count=1,
        target_count=1,
    )
    assert admitted.work.example_count == 1_000_000

    with pytest.raises(ValueError, match="total example count"):
        build_prepared_nested_lodo_plan(
            ("a", "b", "c", "d"),
            (999_998, 1, 1, 1),
            feature_count=1,
            target_count=1,
        )


def test_plan_and_nodes_are_immutable_and_tamper_evident() -> None:
    plan = _plan(4)
    assert isinstance(plan.work, PreparedNestedLodoWorkEstimate)
    assert isinstance(plan.training_subsets[0], PreparedTrainingSubset)
    assert isinstance(plan.score_blocks[0], PreparedScoreBlock)
    assert not hasattr(plan, "__dict__")

    with pytest.raises(FrozenInstanceError):
        plan.feature_count = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.training_subsets[0].row_count = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="training_subsets"):
        replace(plan, training_subsets=plan.training_subsets[:-1])
    with pytest.raises(ValueError, match="score_blocks"):
        replace(plan, score_blocks=plan.score_blocks[:-1])
    with pytest.raises(ValueError, match="formula"):
        replace(plan.work, logical_base_fit_count=plan.work.logical_base_fit_count + 1)

    first_block = plan.score_blocks[0]
    trained_domain = plan.training_subsets[first_block.training_subset_index].domain_indices[0]
    malformed = replace(first_block, scored_domain_index=trained_domain)
    with pytest.raises(ValueError, match="score_blocks"):
        replace(plan, score_blocks=(malformed, *plan.score_blocks[1:]))


def test_node_constructors_reject_noncanonical_fields() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        PreparedTrainingSubset((1, 0), 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        PreparedTrainingSubset((0, 0), 2)
    with pytest.raises(TypeError, match="exact integer"):
        PreparedScoreBlock(True, 0, 1)
    with pytest.raises(ValueError, match="positive"):
        PreparedScoreBlock(0, 1, 0)
    with pytest.raises(ValueError, match="bounded"):
        PreparedTrainingSubset((10**10_000,), 1)
    with pytest.raises(ValueError, match="example limit"):
        PreparedTrainingSubset((0,), 1_000_001)
    with pytest.raises(ValueError, match="graph limit"):
        PreparedScoreBlock(63, 0, 1)
    with pytest.raises(ValueError, match="domain limit"):
        PreparedScoreBlock(0, 7, 1)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("domain_count", 3, "domain_count"),
        ("domain_count", 65, "domain_count"),
        ("example_count", 1_000_001, "example_count"),
        ("feature_count", 4_097, "feature_count"),
        ("target_count", 257, "target_count"),
    ],
)
def test_work_estimate_direct_construction_cannot_bypass_primitive_limits(
    field_name: str,
    invalid_value: int,
    message: str,
) -> None:
    work = _plan(4).work
    with pytest.raises(ValueError, match=message):
        replace(work, **{field_name: invalid_value})


def test_work_estimate_direct_construction_cannot_bypass_derived_limits() -> None:
    values = prepared_graph_module._estimate_values(
        7,
        1_000_000,
        1,
        1,
    )
    with pytest.raises(ValueError, match="score-row memberships"):
        PreparedNestedLodoWorkEstimate(**values)
