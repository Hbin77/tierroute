# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for offline routing and the bundled smoke demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tierroute.adapters import load_evaluation_dataset
from tierroute.core import BudgetTier
from tierroute.demo import (
    BaselineResult,
    RouteDecision,
    evaluate_six_baselines,
    model_catalogue,
    route_prompt,
)
from tierroute.predictors import (
    BilinearPredictorArtifact,
    BilinearTrainingConfig,
    fit_calibrated_bilinear,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tierroute",
        description="Offline-first, budget-aware LLM routing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    route_parser = subparsers.add_parser("route", help="route one prompt without model calls")
    route_parser.add_argument("prompt", help="prompt text to classify and route")
    route_parser.add_argument(
        "--tier",
        choices=[tier.value for tier in BudgetTier],
        default=BudgetTier.BALANCED.value,
        help="budget tier (default: balanced)",
    )
    route_parser.add_argument(
        "--data",
        type=Path,
        help="versioned JSON model catalogue (default: bundled synthetic data)",
    )
    route_parser.add_argument(
        "--artifact",
        type=Path,
        help="local calibrated bilinear JSON artifact (default: synthetic demo predictor)",
    )
    route_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    evaluate_parser = subparsers.add_parser("evaluate", help="run all six baselines on replay data")
    evaluate_parser.add_argument(
        "--data",
        type=Path,
        help="versioned JSON replay data (default: bundled synthetic data)",
    )
    evaluate_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    train_parser = subparsers.add_parser(
        "train",
        help="fit an inner-LODO calibrated bilinear artifact offline",
    )
    train_parser.add_argument(
        "--data",
        type=Path,
        help="versioned JSON replay data (default: bundled synthetic data)",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination JSON artifact",
    )
    train_parser.add_argument("--ridge", type=float, default=1.0, help="positive ridge penalty")
    train_parser.add_argument("--seed", type=int, default=0, help="recorded reproducibility seed")
    train_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    subparsers.add_parser("demo", help="run the self-contained offline quickstart")
    return parser


def _route_payload(decision: RouteDecision) -> dict[str, Any]:
    return {
        "tier": decision.tier.value,
        "budget_limit": str(decision.budget_limit),
        "model": decision.model_id,
        "cost": str(decision.model_cost),
        "predicted_quality": round(decision.predicted_quality, 4),
        "lambda_cost": decision.lambda_cost,
        "reason": decision.reason,
        "features": {
            "characters": decision.features.character_count,
            "words": decision.features.word_count,
            "lines": decision.features.line_count,
            "has_code": decision.features.has_code,
            "has_math": decision.features.has_math,
            "domain_tags": list(decision.features.domain_tags),
        },
        "network_used": False,
        "quality_kind": decision.quality_kind,
    }


def _print_route(decision: RouteDecision) -> None:
    print("tierroute routing decision")
    print(f"  tier:              {decision.tier.value}")
    print(f"  budget limit:      {decision.budget_limit}")
    print(f"  selected model:    {decision.model_id}")
    print(f"  estimated cost:    {decision.model_cost}")
    print(f"  predicted quality: {decision.predicted_quality:.3f} ({decision.quality_kind})")
    print(f"  policy:            one-shot lambda={decision.lambda_cost:g}")
    print(f"  reason:            {decision.reason}")
    print(
        "  features:          "
        f"chars={decision.features.character_count}, "
        f"code={decision.features.has_code}, math={decision.features.has_math}, "
        f"domains={','.join(decision.features.domain_tags)}"
    )
    print("  network:           disabled")


def _baseline_payload(result: BaselineResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "tier_quality": {
            tier.value: None if quality is None else round(quality, 6)
            for tier, quality in result.score.tier_quality.items()
        },
        "weighted_quality": (
            None
            if result.score.weighted_quality is None
            else round(result.score.weighted_quality, 6)
        ),
        "oracle_gap_recovery": (
            None if result.gap_recovery is None else round(result.gap_recovery, 6)
        ),
        "total_cost": str(result.total_cost),
        "feasible": all(tier.feasible for tier in result.report.tiers),
    }


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _training_payload(
    artifact: BilinearPredictorArtifact,
    output: Path,
    dataset_name: str,
) -> dict[str, Any]:
    return {
        "artifact": str(output),
        "artifact_version": artifact.artifact_version,
        "dataset": dataset_name,
        "feature_dimension": artifact.feature_schema.dimension,
        "model_ids": list(artifact.model_ids),
        "training_data_sha256": artifact.training_data_sha256,
        "training_examples": artifact.training_example_count,
        "training_domains": list(artifact.training_domains),
        "ridge": artifact.ridge,
        "seed": artifact.seed,
        "solver_id": artifact.solver_id,
        "network_used": False,
    }


def _print_scorecard(dataset_name: str, results: tuple[BaselineResult, ...]) -> None:
    print(f"Dataset: {dataset_name}")
    print("Budget scope: illustrative per-query limits (official SKT scope unresolved)")
    header = (
        f"{'baseline':<20} {'fast':>7} {'balanced':>9} {'premium':>8} {'weighted':>9} {'gap':>7}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        quality = result.score.tier_quality
        print(
            f"{result.name:<20} "
            f"{_format_score(quality.get(BudgetTier.FAST)):>7} "
            f"{_format_score(quality.get(BudgetTier.BALANCED)):>9} "
            f"{_format_score(quality.get(BudgetTier.PREMIUM)):>8} "
            f"{_format_score(result.score.weighted_quality):>9} "
            f"{_format_score(result.gap_recovery):>7}"
        )
    print("Note: bundled numbers are synthetic smoke checks, not benchmark claims.")
    print("Domain-table smoke fitting uses this sample; real evaluation must use LODO folds.")


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI and return a process status code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset = load_evaluation_dataset(getattr(args, "data", None))

    if args.command == "route":
        predictor = None
        quality_kind = None
        if args.artifact is not None:
            artifact = BilinearPredictorArtifact.load(args.artifact)
            catalogue_ids = tuple(sorted(model.model_id for model in model_catalogue(dataset)))
            if artifact.model_ids != catalogue_ids:
                raise ValueError("artifact model catalogue does not match routing data")
            predictor = artifact.build_predictor()
            quality_kind = "calibrated bilinear artifact"
        decision = route_prompt(
            dataset,
            args.prompt,
            BudgetTier(args.tier),
            predictor=predictor,
            quality_kind=quality_kind,
        )
        if args.json:
            print(json.dumps(_route_payload(decision), ensure_ascii=False, sort_keys=True))
        else:
            _print_route(decision)
        return 0

    if args.command == "train":
        artifact = fit_calibrated_bilinear(
            dataset.examples,
            config=BilinearTrainingConfig(ridge=args.ridge, seed=args.seed),
        )
        output = artifact.save(args.output)
        payload = _training_payload(artifact, output, dataset.name)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print("tierroute predictor training")
            print(f"  dataset:            {dataset.name}")
            print(f"  training examples:  {artifact.training_example_count}")
            print(f"  training domains:   {', '.join(artifact.training_domains)}")
            print(f"  candidate models:   {', '.join(artifact.model_ids)}")
            print(f"  feature dimension:  {artifact.feature_schema.dimension}")
            print(f"  ridge solver:       {artifact.solver_id}")
            print(f"  artifact:           {output}")
            print("  network:            disabled")
            print("  note: synthetic data is a wiring test, not benchmark evidence")
        return 0

    if args.command == "evaluate":
        results = evaluate_six_baselines(dataset)
        if args.json:
            payload = {
                "dataset": dataset.name,
                "provenance": dataset.provenance,
                "budget_scope": "per-query-illustrative",
                "baselines": [_baseline_payload(result) for result in results],
            }
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            _print_scorecard(dataset.name, results)
        return 0

    if args.command == "demo":
        print("tierroute offline quickstart\n")
        prompts = (
            (BudgetTier.FAST, "What gas do plants take in?"),
            (BudgetTier.BALANCED, "Prove that sqrt(2) is irrational."),
            (
                BudgetTier.PREMIUM,
                "Debug an async payment retry race and propose an idempotent implementation.",
            ),
        )
        for tier, prompt in prompts:
            decision = route_prompt(dataset, prompt, tier)
            print(
                f"[{tier.value:<8}] {decision.model_id:<7} "
                f"cost={decision.model_cost} predicted_quality={decision.predicted_quality:.3f}"
            )
        print()
        _print_scorecard(dataset.name, evaluate_six_baselines(dataset))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
