# SPDX-License-Identifier: Apache-2.0
"""Validate and replay the locally downloaded, pinned RouterBench artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TextIO

from tierroute.adapters import PerQueryBudgetLedger
from tierroute.adapters.routerbench import (
    ROUTERBENCH_COLUMNS,
    ROUTERBENCH_REVISION,
    ROUTERBENCH_ROW_COUNT,
    ROUTERBENCH_SHA256,
    estimate_routerbench_quoted_costs,
    iter_routerbench_rows,
    normalize_routerbench_domain,
    routerbench_row_to_example,
)
from tierroute.core import BudgetTier, Cost, as_cost
from tierroute.eval import EvaluationExample, OfflineSimulator, TierSpec
from tierroute.policies import (
    BASELINE_NAMES,
    AlwaysCheapestRouter,
    PerQueryNestedLodoBenchmark,
    evaluate_per_query_bilinear_benchmark,
)

ROUTERBENCH_COLUMN_COUNT = len(ROUTERBENCH_COLUMNS)
ROUTERBENCH_IN_SCOPE_COUNT = 34_778
ROUTERBENCH_MODEL_COUNT = sum(column.endswith("|model_response") for column in ROUTERBENCH_COLUMNS)
ROUTERBENCH_DOMAIN_COUNTS = {
    "arc-challenge": 1_470,
    "gsm8k": 7_450,
    "hellaswag": 10_042,
    "mbpp": 427,
    "mmlu": 14_042,
    "mtbench": 80,
    "winogrande": 1_267,
}
ROUTERBENCH_SEMANTIC_SHA256 = "7b4749ad5c4bdb338c2317b306c382680b1a23dc83c73e29ab805b8f7e472e87"
_SEMANTIC_DIGEST_MAGIC = b"tierroute-routerbench-semantic-v1\0"
ROUTERBENCH_SPLIT_ALGORITHM = "tierroute-routerbench-domain-rank-v1"
ROUTERBENCH_SPLIT_DIGEST_ALGORITHM = "tierroute-routerbench-balanced-split-sha256-v1"
ROUTERBENCH_CALIBRATION_PER_DOMAIN = 64
ROUTERBENCH_EVALUATION_PER_DOMAIN = 8
ROUTERBENCH_MAX_LAMBDA_CANDIDATES = 32
ROUTERBENCH_QUOTE_RULE = "calibration-only-per-model-maximum-realized-cost-v1"
ROUTERBENCH_TIER_RULE = "sorted-quote-min-median-max-v1"
ROUTERBENCH_DIAGNOSTIC_SCHEMA = "tierroute-routerbench-local-diagnostic-v1"
ROUTERBENCH_SURROGATE_ID_ALGORITHM = "source-order-zero-padded-index-v1"
ROUTERBENCH_DIAGNOSTIC_WARNING = "LOCAL OPTIONAL VALIDATION — NON-OFFICIAL, NON-REPORTABLE"
_SPLIT_RANK_MAGIC = b"tierroute-routerbench-domain-rank-v1\0"
_SPLIT_DIGEST_MAGIC = b"tierroute-routerbench-balanced-split-sha256-v1\0"


def _update_sized_utf8(digest: object, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack(">Q", len(encoded)))  # type: ignore[attr-defined]
    digest.update(encoded)  # type: ignore[attr-defined]


class _SemanticDigestBuilder:
    """Incrementally frame decoded values for a layout-independent regression hash."""

    def __init__(self, *, expected_row_count: int, columns: tuple[str, ...]) -> None:
        if expected_row_count <= 0:
            raise ValueError("expected_row_count must be positive")
        if not columns or any(not isinstance(column, str) for column in columns):
            raise ValueError("RouterBench semantic columns must be non-empty strings")
        self.columns = columns
        self.expected_row_count = expected_row_count
        self.row_count = 0
        self.digest = hashlib.sha256()
        self.digest.update(_SEMANTIC_DIGEST_MAGIC)
        self.digest.update(struct.pack(">QQ", expected_row_count, len(columns)))
        for column in columns:
            _update_sized_utf8(self.digest, column)

    def update(self, row: Mapping[str, object]) -> None:
        """Add one row, preserving exact column order and binary64 values."""

        if tuple(row) != self.columns:
            raise ValueError(f"RouterBench row {self.row_count} changed column order")
        for column in self.columns:
            value = row[column]
            if isinstance(value, str):
                self.digest.update(b"S")
                _update_sized_utf8(self.digest, value)
            elif type(value) is float and math.isfinite(value):
                self.digest.update(b"F")
                self.digest.update(struct.pack(">d", value))
            else:
                raise ValueError(
                    f"RouterBench row {self.row_count} column {column!r} has unsupported "
                    f"semantic type {type(value).__name__}"
                )
        self.row_count += 1

    def hexdigest(self) -> str:
        """Finish only after observing the declared number of rows."""

        if self.row_count != self.expected_row_count:
            raise ValueError(
                "RouterBench row count mismatch: "
                f"expected {self.expected_row_count}, got {self.row_count}"
            )
        return self.digest.hexdigest()


def _scan_semantic_rows(
    raw_rows: Iterable[Mapping[str, object]],
    *,
    expected_row_count: int,
    retain_limit: int | None = None,
) -> tuple[str, tuple[Mapping[str, object], ...], int, Counter[str], tuple[str, ...]]:
    """Hash every value while retaining at most the requested LODO-scoped prefix."""

    if retain_limit is not None and retain_limit < 0:
        raise ValueError("retain_limit must be non-negative or None")

    iterator = iter(raw_rows)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("RouterBench decoder produced no rows") from error
    columns = tuple(first)
    digest = _SemanticDigestBuilder(expected_row_count=expected_row_count, columns=columns)

    in_scope: list[Mapping[str, object]] = []
    in_scope_count = 0
    domain_counts: Counter[str] = Counter()
    row_count = 0
    for row in chain((first,), iterator):
        digest.update(row)
        domain = normalize_routerbench_domain(str(row["eval_name"]))
        if domain is not None:
            if retain_limit is None or len(in_scope) < retain_limit:
                in_scope.append(row)
            in_scope_count += 1
            domain_counts[domain] += 1
        row_count += 1

    if row_count != digest.row_count:
        raise AssertionError("semantic digest row count diverged from domain scan")
    return digest.hexdigest(), tuple(in_scope), in_scope_count, domain_counts, columns


@dataclass(frozen=True, slots=True)
class _SelectedRouterBenchRow:
    """One bounded-memory split member whose local raw row is excluded from repr."""

    row_number: int
    domain: str
    sample_id: str
    rank_sha256: str
    row: Mapping[str, object] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _BalancedRouterBenchSplit:
    """Authenticated, balanced local subsets plus non-reversible membership evidence."""

    semantic_sha256: str
    columns: tuple[str, ...]
    in_scope_count: int
    domain_counts: tuple[tuple[str, int], ...]
    calibration: tuple[_SelectedRouterBenchRow, ...]
    evaluation: tuple[_SelectedRouterBenchRow, ...]
    split_sha256: str


def _selection_rank_sha256(
    domain: str,
    sample_id: str,
    *,
    revision: str = ROUTERBENCH_REVISION,
) -> str:
    """Rank rows without consulting prompts, outcomes, costs, or qualities."""

    digest = hashlib.sha256()
    digest.update(_SPLIT_RANK_MAGIC)
    for value in (revision, domain, sample_id):
        _update_sized_utf8(digest, value)
    return digest.hexdigest()


def _split_membership_sha256(
    calibration: Sequence[_SelectedRouterBenchRow],
    evaluation: Sequence[_SelectedRouterBenchRow],
    *,
    revision: str,
) -> str:
    """Bind only private membership identities, returning a public-safe digest."""

    digest = hashlib.sha256()
    digest.update(_SPLIT_DIGEST_MAGIC)
    _update_sized_utf8(digest, revision)
    for role, rows in (("calibration", calibration), ("evaluation", evaluation)):
        _update_sized_utf8(digest, role)
        digest.update(struct.pack(">Q", len(rows)))
        for selected in sorted(rows, key=lambda item: (item.domain, item.rank_sha256)):
            _update_sized_utf8(digest, selected.domain)
            _update_sized_utf8(digest, selected.sample_id)
    return digest.hexdigest()


def _retain_lowest_ranked(
    bucket: list[_SelectedRouterBenchRow],
    selected: _SelectedRouterBenchRow,
    *,
    maximum: int,
) -> None:
    """Keep a fixed-size bottom-k sample without retaining all decoded rows."""

    if len(bucket) < maximum:
        bucket.append(selected)
        return
    largest_index = max(
        range(len(bucket)),
        key=lambda index: (bucket[index].rank_sha256, bucket[index].row_number),
    )
    largest = bucket[largest_index]
    if (selected.rank_sha256, selected.row_number) < (
        largest.rank_sha256,
        largest.row_number,
    ):
        bucket[largest_index] = selected


def _scan_balanced_routerbench_split(
    raw_rows: Iterable[Mapping[str, object]],
    *,
    expected_row_count: int,
    expected_domain_counts: Mapping[str, int],
    calibration_per_domain: int = ROUTERBENCH_CALIBRATION_PER_DOMAIN,
    evaluation_per_domain: int = ROUTERBENCH_EVALUATION_PER_DOMAIN,
    revision: str = ROUTERBENCH_REVISION,
) -> _BalancedRouterBenchSplit:
    """Authenticate all rows while retaining an exact balanced bottom-k split."""

    if len(expected_domain_counts) < 4:
        raise ValueError("nested LODO selection requires at least four domains")
    if any(not isinstance(domain, str) or not domain.strip() for domain in expected_domain_counts):
        raise ValueError("expected domains must be non-empty strings")
    for name, value in (
        ("calibration_per_domain", calibration_per_domain),
        ("evaluation_per_domain", evaluation_per_domain),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    retained_per_domain = calibration_per_domain + evaluation_per_domain
    if any(
        type(count) is not int or count < retained_per_domain
        for count in expected_domain_counts.values()
    ):
        raise ValueError(f"every expected domain must contain at least {retained_per_domain} rows")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty string")

    iterator = iter(raw_rows)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("RouterBench decoder produced no rows") from error
    columns = tuple(first)
    semantic = _SemanticDigestBuilder(expected_row_count=expected_row_count, columns=columns)
    buckets = {domain: [] for domain in expected_domain_counts}
    domain_counts: Counter[str] = Counter()
    seen_sample_ids: set[str] = set()
    in_scope_count = 0

    for row_number, row in enumerate(chain((first,), iterator)):
        semantic.update(row)
        eval_name = row.get("eval_name")
        if not isinstance(eval_name, str) or not eval_name.strip():
            raise ValueError(f"RouterBench row {row_number} has an invalid eval_name")
        domain = normalize_routerbench_domain(eval_name)
        if domain is None:
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"RouterBench row {row_number} has an invalid sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError("RouterBench mapped rows must have unique sample IDs")
        seen_sample_ids.add(sample_id)
        if domain not in buckets:
            raise ValueError(f"RouterBench produced unexpected mapped domain {domain!r}")
        domain_counts[domain] += 1
        in_scope_count += 1
        rank_sha256 = _selection_rank_sha256(domain, sample_id, revision=revision)
        _retain_lowest_ranked(
            buckets[domain],
            _SelectedRouterBenchRow(
                row_number=row_number,
                domain=domain,
                sample_id=sample_id,
                rank_sha256=rank_sha256,
                row=row,
            ),
            maximum=retained_per_domain,
        )

    observed_counts = dict(domain_counts)
    expected_counts = dict(expected_domain_counts)
    if observed_counts != expected_counts:
        raise ValueError(
            f"RouterBench domain counts mismatch: expected {expected_counts}, got {observed_counts}"
        )

    calibration: list[_SelectedRouterBenchRow] = []
    evaluation: list[_SelectedRouterBenchRow] = []
    for domain in sorted(expected_domain_counts):
        ranked = sorted(
            buckets[domain],
            key=lambda item: (item.rank_sha256, item.row_number),
        )
        if len(ranked) != retained_per_domain:
            raise ValueError(
                f"RouterBench domain {domain!r} did not retain {retained_per_domain} rows"
            )
        calibration.extend(ranked[:calibration_per_domain])
        evaluation.extend(ranked[calibration_per_domain:])
    evaluation.sort(key=lambda item: item.row_number)
    calibration_rows = tuple(calibration)
    evaluation_rows = tuple(evaluation)
    return _BalancedRouterBenchSplit(
        semantic_sha256=semantic.hexdigest(),
        columns=columns,
        in_scope_count=in_scope_count,
        domain_counts=tuple(sorted(domain_counts.items())),
        calibration=calibration_rows,
        evaluation=evaluation_rows,
        split_sha256=_split_membership_sha256(
            calibration_rows,
            evaluation_rows,
            revision=revision,
        ),
    )


def _validate_authenticated_metadata(
    *,
    semantic_sha256: str,
    columns: Sequence[str],
    in_scope_count: int,
    domain_counts: Mapping[str, int],
) -> None:
    """Fail closed when the decoded artifact differs from the pinned contract."""

    if len(columns) != ROUTERBENCH_COLUMN_COUNT:
        raise ValueError(
            "RouterBench column count mismatch: "
            f"expected {ROUTERBENCH_COLUMN_COUNT}, got {len(columns)}"
        )
    if semantic_sha256 != ROUTERBENCH_SEMANTIC_SHA256:
        raise ValueError(
            "RouterBench semantic digest mismatch: "
            f"expected {ROUTERBENCH_SEMANTIC_SHA256}, got {semantic_sha256}"
        )
    if in_scope_count != ROUTERBENCH_IN_SCOPE_COUNT:
        raise ValueError(
            "RouterBench in-scope row count mismatch: "
            f"expected {ROUTERBENCH_IN_SCOPE_COUNT}, got {in_scope_count}"
        )
    if dict(domain_counts) != ROUTERBENCH_DOMAIN_COUNTS:
        raise ValueError(
            "RouterBench domain counts mismatch: "
            f"expected {ROUTERBENCH_DOMAIN_COUNTS}, got {dict(domain_counts)}"
        )


def _model_ids_from_columns(columns: Sequence[str]) -> tuple[str, ...]:
    model_ids = tuple(
        sorted(
            column.removesuffix("|model_response")
            for column in columns
            if column.endswith("|model_response")
        )
    )
    if len(model_ids) != ROUTERBENCH_MODEL_COUNT:
        raise ValueError(
            "RouterBench model count mismatch: "
            f"expected {ROUTERBENCH_MODEL_COUNT}, got {len(model_ids)}"
        )
    return model_ids


def _maximum_calibration_quotes(
    rows: Sequence[_SelectedRouterBenchRow],
    *,
    model_ids: Sequence[str],
) -> dict[str, Cost]:
    """Fit conservative pre-call quotes from calibration rows only."""

    if not rows:
        raise ValueError("RouterBench calibration rows must not be empty")
    expected_models = tuple(model_ids)
    if not expected_models:
        raise ValueError("RouterBench model catalogue must not be empty")
    maxima: dict[str, Cost] = {}
    for selected in rows:
        current_models = tuple(
            sorted(
                column.removesuffix("|model_response")
                for column in selected.row
                if column.endswith("|model_response")
            )
        )
        if current_models != expected_models:
            raise ValueError("RouterBench calibration rows changed the model catalogue")
        for model_id in expected_models:
            cost = as_cost(str(selected.row[f"{model_id}|total_cost"]))
            previous = maxima.get(model_id)
            if previous is None or cost > previous:
                maxima[model_id] = cost
    if tuple(sorted(maxima)) != expected_models:
        raise AssertionError("RouterBench maximum quote fitting lost a candidate model")
    return maxima


def _diagnostic_tier_specs(quoted_costs: Mapping[str, Cost]) -> tuple[TierSpec, ...]:
    """Create a mechanical, explicitly non-official three-tier profile."""

    if len(quoted_costs) < 3:
        raise ValueError("RouterBench diagnostic tiers require at least three models")
    ordered = tuple(sorted(quoted_costs.items(), key=lambda item: (item[1], item[0])))
    budgets = (ordered[0][1], ordered[len(ordered) // 2][1], ordered[-1][1])
    if not budgets[0] < budgets[1] < budgets[2]:
        raise ValueError("RouterBench diagnostic tier budgets must be strictly increasing")
    return (
        TierSpec(BudgetTier.FAST, budgets[0], 0.5),
        TierSpec(BudgetTier.BALANCED, budgets[1], 0.3),
        TierSpec(BudgetTier.PREMIUM, budgets[2], 0.2),
    )


def _convert_and_preflight_evaluation(
    rows: Sequence[_SelectedRouterBenchRow],
    *,
    quoted_costs: Mapping[str, Cost],
) -> tuple[EvaluationExample, ...]:
    """Convert every row and reject quote underruns before any predictor fit."""

    examples = []
    for selected in rows:
        example = routerbench_row_to_example(
            selected.row,
            row_number=selected.row_number,
            quoted_costs=quoted_costs,
        )
        if example is None:
            raise ValueError("balanced RouterBench evaluation retained an unmapped row")
        # Benchmark internals and their failure messages use only a deterministic
        # surrogate, never the external dataset's private sample identifier.
        examples.append(
            EvaluationExample(
                example_id=f"routerbench-evaluation-{len(examples):04d}",
                prompt=example.prompt,
                domain=example.domain,
                outcomes=example.outcomes,
                candidate_models=example.candidate_models,
                router_metadata=example.router_metadata,
            )
        )
    if not examples:
        raise ValueError("RouterBench balanced evaluation produced no examples")
    for example in examples:
        quote_by_model = {model.model_id: model.cost for model in example.candidate_models}
        if any(outcome.cost > quote_by_model[outcome.model_id] for outcome in example.outcomes):
            raise ValueError(
                "RouterBench evaluation cost exceeds a calibration-only quote; "
                "aborting before predictor fitting"
            )
    return tuple(examples)


def _candidate_search_evidence(
    benchmark: PerQueryNestedLodoBenchmark,
) -> tuple[list[dict[str, object]], bool]:
    folds: list[dict[str, object]] = []
    approximate = False
    membership_by_domain = {
        membership.held_out_domain: membership for membership in benchmark.fold_memberships
    }
    for fold in benchmark.learned.folds:
        searches = []
        for selection in fold.tuning.selections:
            candidates = selection.candidates
            approximate = approximate or not candidates.exhaustive
            searches.append(
                {
                    "tier": selection.tier.value,
                    "retained_candidate_count": len(candidates.values),
                    "total_derived_values": candidates.total_derived_values,
                    "exhaustive": candidates.exhaustive,
                    "strategy": candidates.strategy,
                    "observed_breakpoint_count": candidates.observed_breakpoint_count,
                }
            )
        membership = membership_by_domain[fold.held_out_domain]
        folds.append(
            {
                "held_out_domain": fold.held_out_domain,
                "training_example_count": membership.training_example_count,
                "test_example_count": membership.test_example_count,
                "membership_sha256": membership.sha256,
                "membership_algorithm": membership.algorithm,
                "lambda_searches": searches,
            }
        )
    return folds, approximate


def _diagnostic_document(
    split: _BalancedRouterBenchSplit,
    tier_specs: Sequence[TierSpec],
    benchmark: PerQueryNestedLodoBenchmark,
) -> dict[str, object]:
    """Expose only aggregate provenance and completion evidence, never metrics."""

    fold_evidence, approximate = _candidate_search_evidence(benchmark)
    evaluation_scope = benchmark.learned.report.evaluation_scope
    return {
        "schema": ROUTERBENCH_DIAGNOSTIC_SCHEMA,
        "warning": ROUTERBENCH_DIAGNOSTIC_WARNING,
        "result_status": "diagnostic",
        "execution_status": "completed",
        "claim_scope": "external-routerbench-local-only-non-official-non-reportable",
        "dataset_license": "NOASSERTION",
        "redistribution_authorized": False,
        "official_skt_data": False,
        "competition_score": False,
        "feature_set": "surface-only",
        "bge_m3_used": False,
        "budget_profile_official": False,
        "network_used": False,
        "artifact_written_by_validator": False,
        "error_detail_published": False,
        "performance_metrics_published": False,
        "row_level_results_published": False,
        "dataset": {
            "revision": ROUTERBENCH_REVISION,
            "byte_sha256": ROUTERBENCH_SHA256,
            "semantic_sha256": split.semantic_sha256,
            "row_grain": "sample_id",
            "in_scope_example_count": split.in_scope_count,
            "domain_counts": dict(split.domain_counts),
            "candidate_model_count": ROUTERBENCH_MODEL_COUNT,
        },
        "split": {
            "algorithm": ROUTERBENCH_SPLIT_ALGORITHM,
            "digest_algorithm": ROUTERBENCH_SPLIT_DIGEST_ALGORITHM,
            "sha256": split.split_sha256,
            "calibration_per_domain": ROUTERBENCH_CALIBRATION_PER_DOMAIN,
            "evaluation_per_domain": ROUTERBENCH_EVALUATION_PER_DOMAIN,
            "calibration_example_count": len(split.calibration),
            "evaluation_example_count": len(split.evaluation),
            "evaluation_source_order_restored": True,
            "benchmark_id_algorithm": ROUTERBENCH_SURROGATE_ID_ALGORITHM,
        },
        "quoted_costs": {
            "rule": ROUTERBENCH_QUOTE_RULE,
            "calibration_only": True,
            "evaluation_preflight": "passed",
            "quote_values_published": False,
        },
        "tier_profile": {
            "rule": ROUTERBENCH_TIER_RULE,
            "official": False,
            "budget_values_published": False,
            "tiers": [
                {
                    "tier": spec.tier.value,
                    "weight": float(spec.weight),
                }
                for spec in tier_specs
            ],
        },
        "evaluation": {
            "protocol": "quality-predictor-policy-nested-leave-one-domain-out",
            "quote_tier_calibration_scope": "global-disjoint-all-domain-calibration-pool",
            "end_to_end_domain_shift_claim": False,
            "predictor_kind": benchmark.predictor_kind,
            "accounting_scope": benchmark.accounting_scope,
            "data_sha256": benchmark.data_sha256,
            "replay_sha256": benchmark.replay_sha256,
            "prediction_sha256": benchmark.learned.prediction_sha256,
            "evaluation_scope_sha256": evaluation_scope.sha256,
            "evaluation_scope_algorithm": evaluation_scope.algorithm,
            "training_config": {
                "ridge": benchmark.training_config.ridge,
                "seed": benchmark.training_config.seed,
                "solver_id": benchmark.training_config.solver_id,
            },
            "lambda_search": {
                "max_candidates_per_tier": ROUTERBENCH_MAX_LAMBDA_CANDIDATES,
                "requested_mode": benchmark.lambda_search_config.requested_mode,
                "approximate": approximate,
                "folds": fold_evidence,
            },
            "baseline_names": list(BASELINE_NAMES),
            "baseline_count": len(BASELINE_NAMES),
            "baseline_config_evidence_sha256": (
                benchmark.baselines.baseline_config_evidence_sha256
            ),
            "oracle_role": "non-deployable-upper-bound",
            "fold_count": len(benchmark.fold_memberships),
        },
    }


def validate_nested_lodo(path: Path) -> dict[str, object]:
    """Run the fixed local diagnostic and return safe aggregate evidence only."""

    split = _scan_balanced_routerbench_split(
        iter_routerbench_rows(path),
        expected_row_count=ROUTERBENCH_ROW_COUNT,
        expected_domain_counts=ROUTERBENCH_DOMAIN_COUNTS,
    )
    _validate_authenticated_metadata(
        semantic_sha256=split.semantic_sha256,
        columns=split.columns,
        in_scope_count=split.in_scope_count,
        domain_counts=dict(split.domain_counts),
    )
    model_ids = _model_ids_from_columns(split.columns)
    quoted_costs = _maximum_calibration_quotes(split.calibration, model_ids=model_ids)
    tier_specs = _diagnostic_tier_specs(quoted_costs)
    examples = _convert_and_preflight_evaluation(
        split.evaluation,
        quoted_costs=quoted_costs,
    )
    expected_evaluation_count = len(ROUTERBENCH_DOMAIN_COUNTS) * ROUTERBENCH_EVALUATION_PER_DOMAIN
    if len(examples) != expected_evaluation_count:
        raise ValueError(
            "RouterBench evaluation count mismatch: "
            f"expected {expected_evaluation_count}, got {len(examples)}"
        )
    evaluation_domain_counts = Counter(example.domain for example in examples)
    expected_evaluation_domains = {
        domain: ROUTERBENCH_EVALUATION_PER_DOMAIN for domain in ROUTERBENCH_DOMAIN_COUNTS
    }
    if dict(evaluation_domain_counts) != expected_evaluation_domains:
        raise ValueError("RouterBench evaluation split is not domain-balanced")
    benchmark = evaluate_per_query_bilinear_benchmark(
        examples,
        tier_specs,
        max_candidates_per_tier=ROUTERBENCH_MAX_LAMBDA_CANDIDATES,
    )
    return _diagnostic_document(split, tier_specs, benchmark)


def validate_and_replay(path: Path, *, replay_limit: int) -> None:
    """Authenticate all rows, then replay a deterministic prefix or the full set."""

    if replay_limit < 0:
        raise ValueError("replay_limit must be non-negative")
    calibration_count = min(1_000, max(1, ROUTERBENCH_IN_SCOPE_COUNT // 5))
    retain_limit = None if replay_limit == 0 else calibration_count + replay_limit
    semantic_sha256, rows, in_scope_count, domain_counts, columns = _scan_semantic_rows(
        iter_routerbench_rows(path),
        expected_row_count=ROUTERBENCH_ROW_COUNT,
        retain_limit=retain_limit,
    )
    _validate_authenticated_metadata(
        semantic_sha256=semantic_sha256,
        columns=columns,
        in_scope_count=in_scope_count,
        domain_counts=domain_counts,
    )
    quoted_costs = estimate_routerbench_quoted_costs(rows[:calibration_count])
    evaluation_rows = rows[calibration_count:] or rows
    replay_rows = evaluation_rows if replay_limit == 0 else evaluation_rows[:replay_limit]
    replay = tuple(
        example
        for row_number, row in enumerate(replay_rows)
        if (
            example := routerbench_row_to_example(
                row,
                row_number=row_number,
                quoted_costs=quoted_costs,
            )
        )
        is not None
    )
    if not replay:
        raise ValueError("RouterBench conversion produced no replay examples")
    if len(replay[0].outcomes) != ROUTERBENCH_MODEL_COUNT:
        raise ValueError(
            "RouterBench model count mismatch: "
            f"expected {ROUTERBENCH_MODEL_COUNT}, got {len(replay[0].outcomes)}"
        )
    cheapest_model_id = min(quoted_costs, key=lambda model_id: (quoted_costs[model_id], model_id))
    maximum_charge = max(
        next(outcome.cost for outcome in example.outcomes if outcome.model_id == cheapest_model_id)
        for example in replay
    )
    budget = max(maximum_charge, quoted_costs[cheapest_model_id])
    result = OfflineSimulator(PerQueryBudgetLedger).run_tier(
        AlwaysCheapestRouter(),
        replay,
        TierSpec(BudgetTier.FAST, budget, 1.0),
    )
    if not result.feasible or result.mean_quality is None:
        raise RuntimeError("RouterBench smoke replay was infeasible")

    domains = sorted(domain_counts)
    print(f"Verified revision: {ROUTERBENCH_REVISION}")
    print(f"SHA-256: {ROUTERBENCH_SHA256}")
    print(f"Semantic SHA-256: {semantic_sha256}")
    print(f"In-scope examples: {in_scope_count}")
    print(f"Candidate models: {len(replay[0].outcomes)}")
    print(f"LODO domains: {', '.join(domains)}")
    print(f"Cost-calibration rows: {calibration_count}")
    print(f"Replayed examples: {len(replay)}")
    print("Smoke replay status: completed (performance metrics suppressed)")
    print(
        "Cost note: pre-call quotes are fitted on calibration rows; replay uses realized charges."
    )
    print("Dataset license: NOASSERTION; redistribution is not authorized by tierroute.")


def _diagnostic_failure_document() -> dict[str, object]:
    """Return a valid JSON failure envelope without reflecting private exceptions."""

    return {
        "schema": ROUTERBENCH_DIAGNOSTIC_SCHEMA,
        "warning": ROUTERBENCH_DIAGNOSTIC_WARNING,
        "result_status": "diagnostic",
        "execution_status": "failed",
        "claim_scope": "external-routerbench-local-only-non-official-non-reportable",
        "dataset_license": "NOASSERTION",
        "redistribution_authorized": False,
        "official_skt_data": False,
        "competition_score": False,
        "feature_set": "surface-only",
        "bge_m3_used": False,
        "budget_profile_official": False,
        "network_used": False,
        "artifact_written_by_validator": False,
        "error_detail_published": False,
        "performance_metrics_published": False,
        "row_level_results_published": False,
    }


def _print_exact_diagnostic_boundaries(*, file: TextIO | None = None) -> None:
    """Print the mandatory non-official claim labels with machine-readable values."""

    destination = sys.stdout if file is None else file
    for line in (
        "dataset_license=NOASSERTION",
        "redistribution_authorized=false",
        "official_skt_data=false",
        "competition_score=false",
        "feature_set=surface-only",
        "bge_m3_used=false",
        "budget_profile_official=false",
        "network_used=false",
    ):
        print(line, file=destination)


def _print_nested_diagnostic(document: Mapping[str, object]) -> None:
    """Render a short human-safe view of a validated diagnostic document."""

    dataset = document["dataset"]
    split = document["split"]
    evaluation = document["evaluation"]
    if not isinstance(dataset, Mapping) or not isinstance(split, Mapping):
        raise TypeError("diagnostic document has invalid dataset or split evidence")
    if not isinstance(evaluation, Mapping):
        raise TypeError("diagnostic document has invalid evaluation evidence")
    print(ROUTERBENCH_DIAGNOSTIC_WARNING)
    print("RouterBench license: NOASSERTION; not SKT data and not a competition score.")
    _print_exact_diagnostic_boundaries()
    print(f"Verified revision: {dataset['revision']}")
    print(f"SHA-256: {dataset['byte_sha256']}")
    print(f"Semantic SHA-256: {dataset['semantic_sha256']}")
    print(f"Split SHA-256: {split['sha256']}")
    print(f"Calibration examples: {split['calibration_example_count']}")
    print(f"Evaluation examples: {split['evaluation_example_count']}")
    print(f"Nested LODO folds: {evaluation['fold_count']}")
    print("Nested LODO scope: quality predictor and policy evaluation only")
    print("Quote/tier calibration: global disjoint all-domain pool; end-to-end claim: no")
    print(f"Completed baselines: {evaluation['baseline_count']}")
    print("Performance, cost, gap, route, and row-level results: suppressed")
    print("Validator-created artifact: none; network used: no")


def _emit_nested_failure(*, json_output: bool) -> None:
    """Emit no exception text, local path, prompt, row ID, or traceback."""

    if json_output:
        print(
            json.dumps(
                _diagnostic_failure_document(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    print(ROUTERBENCH_DIAGNOSTIC_WARNING, file=sys.stderr)
    _print_exact_diagnostic_boundaries(file=sys.stderr)
    print("execution_status=failed", file=sys.stderr)
    print("error_detail_published=false", file=sys.stderr)
    print(
        "Diagnostic failure details suppressed; local path and row data omitted.",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run local-only validation; this command never downloads data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/routerbench/routerbench_0shot.pkl"),
        help="path produced by download_routerbench.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke rows to replay after validating all rows (default: 200; 0 replays all)",
    )
    parser.add_argument(
        "--nested-lodo",
        action="store_true",
        help="run the fixed local-only learned-plus-six-baseline diagnostic",
    )
    parser.add_argument(
        "--acknowledge-noassertion",
        action="store_true",
        help="acknowledge that RouterBench redistribution/reporting rights are unconfirmed",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit safe aggregate diagnostic evidence as one JSON document",
    )
    args = parser.parse_args(argv)
    if args.nested_lodo:
        if not args.acknowledge_noassertion:
            parser.error("--nested-lodo requires --acknowledge-noassertion")
        if args.limit is not None:
            parser.error("--limit cannot be combined with the fixed --nested-lodo scope")
        try:
            document = validate_nested_lodo(args.input)
        except Exception:
            _emit_nested_failure(json_output=args.json)
            raise SystemExit(1) from None
        if args.json:
            print(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        else:
            _print_nested_diagnostic(document)
        return
    if args.acknowledge_noassertion:
        parser.error("--acknowledge-noassertion is only valid with --nested-lodo")
    if args.json:
        parser.error("--json is only valid with --nested-lodo")
    try:
        validate_and_replay(args.input, replay_limit=200 if args.limit is None else args.limit)
    except Exception:
        print(
            "RouterBench smoke validation failed; details and local path suppressed.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
