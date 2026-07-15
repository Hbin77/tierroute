# SPDX-License-Identifier: Apache-2.0
"""Validate and replay the locally downloaded, pinned RouterBench artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from tierroute.adapters import PerQueryBudgetLedger
from tierroute.adapters.routerbench import (
    ROUTERBENCH_REVISION,
    ROUTERBENCH_SHA256,
    iter_routerbench_examples,
)
from tierroute.core import BudgetTier
from tierroute.eval import OfflineSimulator, TierSpec
from tierroute.policies import AlwaysCheapestRouter


def validate_and_replay(path: Path, *, replay_limit: int) -> None:
    """Authenticate all rows, then replay a deterministic prefix or the full set."""

    if replay_limit < 0:
        raise ValueError("replay_limit must be non-negative")
    examples = tuple(iter_routerbench_examples(path))
    if not examples:
        raise ValueError("RouterBench conversion produced no in-scope examples")
    replay = examples if replay_limit == 0 else examples[:replay_limit]
    budget = max(min(outcome.cost for outcome in example.outcomes) for example in replay)
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
    print(f"Replayed examples: {len(replay)}")
    print(f"Always-cheapest mean quality: {result.mean_quality:.6f}")
    print(f"Replay cost: {result.budget.spent}")
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
