# SPDX-License-Identifier: Apache-2.0
"""Validate and replay the locally downloaded, pinned RouterBench artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from tierroute.adapters import PerQueryBudgetLedger
from tierroute.adapters.routerbench import (
    ROUTERBENCH_REVISION,
    ROUTERBENCH_SHA256,
    estimate_routerbench_quoted_costs,
    iter_routerbench_rows,
    normalize_routerbench_domain,
    routerbench_row_to_example,
)
from tierroute.core import BudgetTier
from tierroute.eval import OfflineSimulator, TierSpec
from tierroute.policies import AlwaysCheapestRouter


def validate_and_replay(path: Path, *, replay_limit: int) -> None:
    """Authenticate all rows, then replay a deterministic prefix or the full set."""

    if replay_limit < 0:
        raise ValueError("replay_limit must be non-negative")
    rows = tuple(
        row
        for row in iter_routerbench_rows(path)
        if normalize_routerbench_domain(str(row["eval_name"])) is not None
    )
    if not rows:
        raise ValueError("RouterBench conversion produced no in-scope examples")
    calibration_count = min(1_000, max(1, len(rows) // 5))
    quoted_costs = estimate_routerbench_quoted_costs(rows[:calibration_count])
    examples = tuple(
        example
        for row_number, row in enumerate(rows)
        if (
            example := routerbench_row_to_example(
                row,
                row_number=row_number,
                quoted_costs=quoted_costs,
            )
        )
        is not None
    )
    evaluation_pool = examples[calibration_count:] or examples
    replay = evaluation_pool if replay_limit == 0 else evaluation_pool[:replay_limit]
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

    domains = sorted({example.domain for example in examples})
    print(f"Verified revision: {ROUTERBENCH_REVISION}")
    print(f"SHA-256: {ROUTERBENCH_SHA256}")
    print(f"Converted examples: {len(examples)}")
    print(f"Candidate models: {len(examples[0].outcomes)}")
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
