# SPDX-License-Identifier: Apache-2.0
"""Validate and replay the locally downloaded, pinned RouterBench artifact."""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import chain
from pathlib import Path

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
from tierroute.core import BudgetTier
from tierroute.eval import OfflineSimulator, TierSpec
from tierroute.policies import AlwaysCheapestRouter

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
    print(f"Always-cheapest mean quality: {result.mean_quality:.6f}")
    print(f"Replay cost: {result.budget.spent}")
    print(
        "Cost note: pre-call quotes are fitted on calibration rows; replay uses realized charges."
    )
    print("Dataset license: NOASSERTION; redistribution is not authorized by tierroute.")


def main() -> None:
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
        default=200,
        help="number of rows to replay after validating all rows; 0 replays all",
    )
    args = parser.parse_args()
    validate_and_replay(args.input, replay_limit=args.limit)


if __name__ == "__main__":
    main()
