# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for offline routing and the bundled smoke demo."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from tierroute.adapters import (
    CumulativeBudgetLedger,
    PerQueryBudgetLedger,
    bundled_synthetic_path,
    load_evaluation_dataset,
)
from tierroute.core import BudgetTier, as_cost, canonical_cost_text
from tierroute.core.atomic_io import AtomicTextWrite, replace_text_bundle, validate_write_paths
from tierroute.core.integer_text import integer_to_decimal
from tierroute.demo import (
    BaselineResult,
    RouteDecision,
    evaluate_six_baselines,
    model_catalogue,
    route_prompt,
)
from tierroute.eval import QuoteErrorSummary
from tierroute.policies.lambda_artifacts import LambdaPolicyArtifact
from tierroute.policies.lambda_tuning import (
    TierLambdaTuningResult,
    cross_fitted_prediction_table,
    preflight_lambda_search,
    tune_tier_lambdas,
)
from tierroute.predictors import (
    BilinearPredictorArtifact,
    BilinearTrainingConfig,
    fit_calibrated_bilinear,
)

DEFAULT_MAX_LAMBDA_CANDIDATES = 257


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
    route_parser.add_argument(
        "--policy-artifact",
        type=Path,
        help="local exact tier-lambda JSON artifact (requires --artifact)",
    )
    route_parser.add_argument(
        "--remaining-budget",
        help="current exact budget; required only for a cumulative policy artifact",
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
    train_parser.add_argument(
        "--policy-output",
        type=Path,
        help="also tune and save an exact tier-lambda JSON artifact",
    )
    train_parser.add_argument(
        "--budget-scope",
        choices=("per-query", "cumulative"),
        help="required accounting semantics when --policy-output is used",
    )
    lambda_search = train_parser.add_mutually_exclusive_group()
    lambda_search.add_argument(
        "--max-lambda-candidates",
        type=int,
        help=(f"deterministic per-tier candidate cap (default: {DEFAULT_MAX_LAMBDA_CANDIDATES})"),
    )
    lambda_search.add_argument(
        "--exhaustive-lambda-search",
        action="store_true",
        help="retain every exact breakpoint candidate instead of applying the cap",
    )
    train_parser.add_argument(
        "--allow-large-exhaustive-search",
        action="store_true",
        help="acknowledge and bypass the exhaustive-search resource preflight",
    )
    train_parser.add_argument("--ridge", type=float, default=1.0, help="positive ridge penalty")
    train_parser.add_argument("--seed", type=int, default=0, help="recorded reproducibility seed")
    train_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    subparsers.add_parser("demo", help="run the self-contained offline quickstart")
    return parser


def _fraction_payload(value: Fraction) -> dict[str, str]:
    return {
        "numerator": integer_to_decimal(value.numerator),
        "denominator": integer_to_decimal(value.denominator),
    }


def _save_training_artifacts(
    artifact: BilinearPredictorArtifact,
    output: Path,
    *,
    policy: LambdaPolicyArtifact | None,
    policy_output: Path | None,
    data_source: Path,
) -> tuple[Path, Path | None]:
    """Commit a predictor and optional bound policy as one rollback-safe bundle."""

    predictor_document = artifact.to_json()
    writes = [AtomicTextWrite(output, predictor_document, BilinearPredictorArtifact.from_json)]
    if policy is None:
        if policy_output is not None:
            raise AssertionError("policy output was supplied without a policy")
    else:
        if policy_output is None:
            raise AssertionError("policy output is required for a policy")
        policy.validate_predictor(artifact)

        def validate_bound_policy(document: str) -> None:
            restored_predictor = BilinearPredictorArtifact.from_json(predictor_document)
            restored_policy = LambdaPolicyArtifact.from_json(document)
            restored_policy.validate_predictor(restored_predictor)

        writes.append(AtomicTextWrite(policy_output, policy.to_json(), validate_bound_policy))

    saved = replace_text_bundle(tuple(writes), protected_paths=(data_source,))
    return saved[0], None if len(saved) == 1 else saved[1]


def _fraction_label(value: Fraction) -> str:
    numerator = integer_to_decimal(value.numerator)
    if value.denominator == 1:
        return numerator
    return f"{numerator}/{integer_to_decimal(value.denominator)}"


def _candidate_total_label(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _route_payload(decision: RouteDecision) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tier": decision.tier.value,
        "budget_limit": canonical_cost_text(decision.budget_limit),
        "remaining_budget": canonical_cost_text(decision.remaining_budget),
        "model": decision.model_id,
        "cost": canonical_cost_text(decision.model_cost),
        "quoted_cost": canonical_cost_text(decision.model_cost),
        "realized_cost": None,
        "predicted_quality": round(decision.predicted_quality, 4),
        "lambda_cost": _fraction_payload(decision.lambda_cost),
        "accounting_scope": decision.accounting_scope,
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
    if decision.lambda_candidates_exhaustive is not None:
        payload["lambda_search"] = {
            "exhaustive": decision.lambda_candidates_exhaustive,
            "retained_candidates": decision.lambda_candidates_retained,
            "derived_candidates": decision.lambda_candidates_derived,
            "strategy": decision.lambda_candidate_strategy,
            "observed_breakpoint_count": decision.lambda_observed_breakpoint_count,
        }
    return payload


def _print_route(decision: RouteDecision) -> None:
    print("tierroute routing decision")
    print(f"  tier:              {decision.tier.value}")
    print(f"  budget limit:      {canonical_cost_text(decision.budget_limit)}")
    print(f"  remaining budget:  {canonical_cost_text(decision.remaining_budget)}")
    print(f"  selected model:    {decision.model_id}")
    print(f"  quoted cost:       {canonical_cost_text(decision.model_cost)}")
    print(f"  predicted quality: {decision.predicted_quality:.3f} ({decision.quality_kind})")
    print(f"  policy:            one-shot lambda={_fraction_label(decision.lambda_cost)}")
    print(f"  accounting scope:  {decision.accounting_scope}")
    if decision.lambda_candidates_exhaustive is not None:
        print(
            "  lambda search:     "
            f"retained={decision.lambda_candidates_retained}/"
            f"{_candidate_total_label(decision.lambda_candidates_derived)}, "
            f"exhaustive={str(decision.lambda_candidates_exhaustive).lower()}, "
            f"strategy={decision.lambda_candidate_strategy}, "
            f"observed_breakpoints={decision.lambda_observed_breakpoint_count}"
        )
    print(f"  reason:            {decision.reason}")
    print(
        "  features:          "
        f"chars={decision.features.character_count}, "
        f"code={decision.features.has_code}, math={decision.features.has_math}, "
        f"domains={','.join(decision.features.domain_tags)}"
    )
    print("  network:           disabled")


def _quote_error_summary_payload(summary: QuoteErrorSummary) -> dict[str, Any]:
    return {
        "executed_calls": summary.call_count,
        "exact_quote_calls": summary.exact_quote_calls,
        "underquoted_calls": summary.underquoted_calls,
        "overquoted_calls": summary.overquoted_calls,
        "realized_over_budget_calls": summary.realized_over_budget_calls,
        "total_quoted_cost": canonical_cost_text(summary.total_quoted_cost),
        "total_realized_cost": canonical_cost_text(summary.total_realized_cost),
        "total_absolute_quote_error": canonical_cost_text(summary.total_absolute_quote_error),
        "total_underquoted_amount": canonical_cost_text(summary.total_underquoted_amount),
        "total_overquoted_amount": canonical_cost_text(summary.total_overquoted_amount),
        "net_quote_error": {
            "direction": summary.net_quote_error.direction.value,
            "magnitude": canonical_cost_text(summary.net_quote_error.magnitude),
        },
    }


def _baseline_payload(result: BaselineResult) -> dict[str, Any]:
    quote_error_by_tier = result.quote_error.by_tier()
    tier_cost_evidence = {}
    for tier_result in result.report.tiers:
        tier = tier_result.tier_spec.tier
        tier_cost_evidence[tier.value] = {
            **_quote_error_summary_payload(quote_error_by_tier[tier]),
            "query_count": len(tier_result.queries),
            "failed_queries": sum(not query.feasible for query in tier_result.queries),
            "budget_adapter": tier_result.budget.adapter_name,
            "configured_limit": canonical_cost_text(tier_result.budget.configured_limit),
            "effective_total_limit": canonical_cost_text(tier_result.budget.effective_total_limit),
            "spent": canonical_cost_text(tier_result.budget.spent),
            "over_budget_calls": tier_result.budget.over_budget_calls,
        }
    return {
        "name": result.name,
        "evaluation_scope": {
            "algorithm": result.report.evaluation_scope_algorithm,
            "sha256": result.report.evaluation_scope_sha256,
            "max_calls_per_query": result.report.max_calls_per_query,
        },
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
        "total_cost": canonical_cost_text(result.total_cost),
        "total_realized_cost": canonical_cost_text(result.total_cost),
        "cost_evidence": {
            "scope": "executed-replay-calls; overall is cross-tier diagnostic only",
            "overall": _quote_error_summary_payload(result.quote_error.overall),
            "by_tier": tier_cost_evidence,
        },
        "feasible": all(tier.feasible for tier in result.report.tiers),
    }


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _training_payload(
    artifact: BilinearPredictorArtifact,
    output: Path,
    dataset_name: str,
    *,
    policy_output: Path | None = None,
    tuning: TierLambdaTuningResult | None = None,
    accounting_scope: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    if policy_output is None:
        return payload
    if tuning is None or accounting_scope is None:
        raise AssertionError("policy output requires tuning evidence and accounting scope")
    payload.update(
        {
            "policy_artifact": str(policy_output),
            "accounting_scope": accounting_scope,
            "evaluation_scope": {
                "algorithm": tuning.report.evaluation_scope_algorithm,
                "sha256": tuning.report.evaluation_scope_sha256,
                "max_calls_per_query": tuning.report.max_calls_per_query,
            },
            "lambda_by_tier": {
                selection.tier.value: _fraction_payload(selection.lambda_cost)
                for selection in tuning.selections
            },
            "lambda_search": {
                selection.tier.value: {
                    "retained_candidates": len(selection.candidates.values),
                    "derived_candidates": selection.candidates.total_derived_values,
                    "exhaustive": selection.candidates.exhaustive,
                    "strategy": selection.candidates.strategy,
                    "observed_breakpoint_count": (selection.candidates.observed_breakpoint_count),
                }
                for selection in tuning.selections
            },
            "tier_training_quality": {
                tier.value: quality for tier, quality in tuning.score.tier_quality.items()
            },
            "feasible": all(selection.report.feasible for selection in tuning.selections),
            "weighted_training_score": tuning.score.weighted_quality,
        }
    )
    return payload


def _print_scorecard(dataset_name: str, results: tuple[BaselineResult, ...]) -> None:
    if not results:
        raise ValueError("scorecard requires at least one baseline result")
    print(f"Dataset: {dataset_name}")
    print("Budget scope: illustrative per-query limits (official SKT scope unresolved)")
    print(
        f"Evaluation scope: {results[0].report.evaluation_scope_algorithm}:"
        f"{results[0].report.evaluation_scope_sha256} "
        f"(max calls/query={results[0].report.max_calls_per_query})"
    )
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
    print("Validation: all six reports use one original-order outer-LODO population.")
    print("Domain table: fit on each outer training side; unseen tags use cheapest fallback.")


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI and return a process status code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "route" and args.policy_artifact is not None and args.artifact is None:
        parser.error("route --policy-artifact requires --artifact")
    if args.command == "route" and args.remaining_budget is not None:
        if args.policy_artifact is None:
            parser.error("route --remaining-budget requires --policy-artifact")
    if args.command == "train":
        if args.policy_output is not None and args.budget_scope is None:
            parser.error("train --policy-output requires --budget-scope")
        if args.policy_output is None and args.budget_scope is not None:
            parser.error("train --budget-scope requires --policy-output")
        if args.policy_output is None and args.exhaustive_lambda_search:
            parser.error("train --exhaustive-lambda-search requires --policy-output")
        if args.policy_output is None and args.max_lambda_candidates is not None:
            parser.error("train --max-lambda-candidates requires --policy-output")
        if args.allow_large_exhaustive_search and not args.exhaustive_lambda_search:
            parser.error(
                "train --allow-large-exhaustive-search requires --exhaustive-lambda-search"
            )
        if args.max_lambda_candidates is not None and args.max_lambda_candidates < 2:
            parser.error("train --max-lambda-candidates must be at least 2")
        data_source = args.data if args.data is not None else bundled_synthetic_path()
        destinations = (
            (args.output,) if args.policy_output is None else (args.output, args.policy_output)
        )
        try:
            validate_write_paths(destinations, protected_paths=(data_source,))
        except ValueError as error:
            parser.error(f"train artifact paths are unsafe: {error}")
    dataset = load_evaluation_dataset(getattr(args, "data", None))

    if args.command == "route":
        predictor = None
        quality_kind = None
        lambda_cost = None
        accounting_scope = "per-query-illustrative"
        candidate_exhaustive = None
        candidate_retained = None
        candidate_derived = None
        candidate_strategy = None
        observed_breakpoint_count = None
        remaining_budget = None
        if args.artifact is not None:
            artifact = BilinearPredictorArtifact.load(args.artifact)
            catalogue_ids = tuple(sorted(model.model_id for model in model_catalogue(dataset)))
            if artifact.model_ids != catalogue_ids:
                raise ValueError("artifact model catalogue does not match routing data")
            predictor = artifact.build_predictor()
            quality_kind = "calibrated bilinear artifact"
            if args.policy_artifact is not None:
                policy = LambdaPolicyArtifact.load(args.policy_artifact)
                policy.validate_predictor(artifact)
                if policy.tier_specs != dataset.tier_specs:
                    raise ValueError("policy tier specifications do not match routing data")
                policy.validate_tuning_data(dataset.examples)
                if policy.ledger_adapter_name not in {"per-query", "cumulative"}:
                    raise ValueError("policy artifact uses an unsupported accounting scope")
                tier = BudgetTier(args.tier)
                if policy.ledger_adapter_name == "cumulative":
                    if args.remaining_budget is None:
                        parser.error("routing a cumulative policy requires --remaining-budget")
                    try:
                        remaining_budget = as_cost(args.remaining_budget)
                    except (TypeError, ValueError) as error:
                        parser.error(f"invalid --remaining-budget: {error}")
                    tier_limit = next(
                        spec.budget_limit for spec in policy.tier_specs if spec.tier is tier
                    )
                    if remaining_budget > tier_limit:
                        parser.error("--remaining-budget cannot exceed the configured tier budget")
                elif args.remaining_budget is not None:
                    parser.error("--remaining-budget is not valid for a per-query policy artifact")
                lambda_cost = policy.lambda_by_tier[tier]
                candidate_set = next(item for item in policy.candidate_sets if item.tier is tier)
                accounting_scope = policy.ledger_adapter_name
                candidate_exhaustive = candidate_set.exhaustive
                candidate_retained = len(candidate_set.values)
                candidate_derived = candidate_set.total_derived_values
                candidate_strategy = candidate_set.strategy
                observed_breakpoint_count = candidate_set.observed_breakpoint_count
                quality_kind = "calibrated bilinear + tuned exact-rational tier lambda"
        decision = route_prompt(
            dataset,
            args.prompt,
            BudgetTier(args.tier),
            predictor=predictor,
            quality_kind=quality_kind,
            lambda_cost=lambda_cost,
            accounting_scope=accounting_scope,
            remaining_budget=remaining_budget,
            lambda_candidates_exhaustive=candidate_exhaustive,
            lambda_candidates_retained=candidate_retained,
            lambda_candidates_derived=candidate_derived,
            lambda_candidate_strategy=candidate_strategy,
            lambda_observed_breakpoint_count=observed_breakpoint_count,
        )
        if args.json:
            print(json.dumps(_route_payload(decision), ensure_ascii=False, sort_keys=True))
        else:
            _print_route(decision)
        return 0

    if args.command == "train":
        if args.policy_output is not None:
            candidate_cap = (
                None
                if args.exhaustive_lambda_search
                else (args.max_lambda_candidates or DEFAULT_MAX_LAMBDA_CANDIDATES)
            )
            preflight_lambda_search(
                dataset.examples,
                dataset.tier_specs,
                max_candidates_per_tier=candidate_cap,
                allow_large_exhaustive=args.allow_large_exhaustive_search,
            )
        config = BilinearTrainingConfig(ridge=args.ridge, seed=args.seed)
        tuning = None
        policy = None
        artifact = fit_calibrated_bilinear(
            dataset.examples,
            config=config,
        )
        if args.policy_output is not None:
            predictions = cross_fitted_prediction_table(
                dataset.examples,
                lambda training: fit_calibrated_bilinear(
                    training,
                    config=config,
                ).build_predictor(),
            )
            ledger_factory = (
                PerQueryBudgetLedger if args.budget_scope == "per-query" else CumulativeBudgetLedger
            )
            tuning = tune_tier_lambdas(
                dataset.examples,
                dataset.tier_specs,
                predictions,
                ledger_factory,
                max_candidates_per_tier=candidate_cap,
                allow_large_exhaustive=args.allow_large_exhaustive_search,
            )
            policy = LambdaPolicyArtifact.from_tuning(
                artifact,
                tuning,
                dataset.tier_specs,
                args.budget_scope,
            )
        data_source = args.data if args.data is not None else bundled_synthetic_path()
        output, policy_output = _save_training_artifacts(
            artifact,
            args.output,
            policy=policy,
            policy_output=args.policy_output,
            data_source=data_source,
        )
        payload = _training_payload(
            artifact,
            output,
            dataset.name,
            policy_output=policy_output,
            tuning=tuning,
            accounting_scope=args.budget_scope,
        )
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
            if policy_output is not None and tuning is not None:
                print(f"  policy artifact:    {policy_output}")
                print(f"  accounting scope:  {args.budget_scope}")
                for selection in tuning.selections:
                    candidates = selection.candidates
                    print(
                        f"  {selection.tier.value} lambda:"
                        f"     {_fraction_label(selection.lambda_cost)} "
                        f"(candidates={len(candidates.values)}/"
                        f"{_candidate_total_label(candidates.total_derived_values)}, "
                        f"exhaustive={str(candidates.exhaustive).lower()}, "
                        f"strategy={candidates.strategy}, "
                        f"observed_breakpoints={candidates.observed_breakpoint_count})"
                    )
                print(
                    "  tuning feasible:   "
                    f"{str(all(item.report.feasible for item in tuning.selections)).lower()}"
                )
                print(f"  weighted score:    {_format_score(tuning.score.weighted_quality)}")
            print("  network:            disabled")
            print("  note: synthetic data is a wiring test, not benchmark evidence")
        return 0

    if args.command == "evaluate":
        results = evaluate_six_baselines(dataset)
        if args.json:
            evaluation_scope = {
                "algorithm": results[0].report.evaluation_scope_algorithm,
                "sha256": results[0].report.evaluation_scope_sha256,
                "max_calls_per_query": results[0].report.max_calls_per_query,
            }
            payload = {
                "dataset": dataset.name,
                "provenance": dataset.provenance,
                "budget_scope": "per-query-illustrative",
                "validation_scope": "outer-lodo-original-order",
                "domain_table_fit": "outer-training-observable-tags-only",
                "evaluation_scope": evaluation_scope,
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
                "quoted_cost="
                f"{canonical_cost_text(decision.model_cost)} "
                f"predicted_quality={decision.predicted_quality:.3f}"
            )
        print()
        _print_scorecard(dataset.name, evaluate_six_baselines(dataset))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
