# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for offline routing and the bundled smoke demo."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from tierroute.adapters import (
    CumulativeBudgetLedger,
    PerQueryBudgetLedger,
    bundled_synthetic_path,
    load_evaluation_dataset,
)
from tierroute.core import BudgetTier, Cost, as_cost, canonical_cost_text
from tierroute.core.atomic_io import AtomicTextWrite, replace_text_bundle, validate_write_paths
from tierroute.core.integer_text import integer_to_decimal
from tierroute.demo import (
    BaselineResult,
    RouteDecision,
    evaluate_six_baselines,
    model_catalogue,
    route_prompt,
)
from tierroute.eval import EvaluationReport, QuoteErrorReport, QuoteErrorSummary, ScoreSummary
from tierroute.policies.benchmark import (
    PerQueryNestedLodoBenchmark,
    evaluate_per_query_bilinear_benchmark,
)
from tierroute.policies.lambda_artifacts import LambdaPolicyArtifact
from tierroute.policies.lambda_tuning import (
    TierLambdaSelection,
    TierLambdaTuningResult,
    cross_fitted_prediction_table,
    preflight_lambda_search,
    tune_tier_lambdas,
)
from tierroute.policies.predictor_comparison import (
    PairedPredictorComparison,
    evaluate_per_query_paired_predictor_comparison,
)
from tierroute.predictors import (
    GBM_ALGORITHM_ID,
    NATIVE_C11_RIDGE_SOLVER_ID,
    BilinearPredictorArtifact,
    BilinearTrainingConfig,
    GbmTrainingConfig,
    RidgeSolver,
    fit_calibrated_bilinear,
)
from tierroute.predictors.native_ridge import NativeRidgeAdapter, NativeRidgeError
from tierroute.predictors.solvers import CENTERED_RIDGE_SOLVER_ID
from tierroute.showcase import RoutingStreamShowcase, build_routing_stream_showcase

DEFAULT_MAX_LAMBDA_CANDIDATES = 257
_TERMINAL_UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def _safe_terminal_text(value: str) -> str:
    """Escape terminal controls while preserving ordinary Unicode display text."""

    if not isinstance(value, str):
        raise TypeError("terminal text must be a string")
    escaped: list[str] = []
    for character in value:
        if unicodedata.category(character) not in _TERMINAL_UNSAFE_CATEGORIES:
            escaped.append(character)
            continue
        codepoint = ord(character)
        if codepoint <= 0xFFFF:
            escaped.append(f"\\u{codepoint:04x}")
        else:  # pragma: no cover - current unsafe categories are in the BMP
            escaped.append(f"\\U{codepoint:08x}")
    return "".join(escaped)


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

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="compare learned nested-LODO routing with all six baselines",
    )
    benchmark_parser.add_argument(
        "--data",
        type=Path,
        help="versioned JSON replay data (default: bundled synthetic data)",
    )
    benchmark_parser.add_argument(
        "--budget-scope",
        choices=("per-query",),
        required=True,
        help="explicit accounting semantics; cumulative is gated on an official sequence oracle",
    )
    benchmark_lambda_search = benchmark_parser.add_mutually_exclusive_group()
    benchmark_lambda_search.add_argument(
        "--max-lambda-candidates",
        type=int,
        help=(f"deterministic per-tier candidate cap (default: {DEFAULT_MAX_LAMBDA_CANDIDATES})"),
    )
    benchmark_lambda_search.add_argument(
        "--exhaustive-lambda-search",
        action="store_true",
        help="retain every exact breakpoint candidate instead of applying the cap",
    )
    benchmark_parser.add_argument(
        "--allow-large-exhaustive-search",
        action="store_true",
        help="acknowledge and bypass the exhaustive-search resource preflight",
    )
    benchmark_parser.add_argument(
        "--ridge",
        type=float,
        default=1.0,
        help="positive ridge penalty",
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="recorded reproducibility seed",
    )
    benchmark_parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )

    comparison_parser = subparsers.add_parser(
        "compare-predictors",
        help="estimate bilinear and GBM routing on identical nested-LODO evidence",
    )
    comparison_parser.add_argument(
        "--data",
        type=Path,
        help="versioned JSON replay data (default: bundled synthetic data)",
    )
    comparison_parser.add_argument(
        "--budget-scope",
        choices=("per-query",),
        required=True,
        help="explicit accounting semantics; cumulative comparison remains gated",
    )
    comparison_lambda_search = comparison_parser.add_mutually_exclusive_group()
    comparison_lambda_search.add_argument(
        "--max-lambda-candidates",
        type=int,
        help=(f"deterministic per-tier candidate cap (default: {DEFAULT_MAX_LAMBDA_CANDIDATES})"),
    )
    comparison_lambda_search.add_argument(
        "--exhaustive-lambda-search",
        action="store_true",
        help="retain every exact breakpoint candidate instead of applying the cap",
    )
    comparison_parser.add_argument(
        "--allow-large-exhaustive-search",
        action="store_true",
        help="acknowledge and bypass the exhaustive-search resource preflight",
    )
    comparison_parser.add_argument("--ridge", type=float, default=1.0)
    comparison_parser.add_argument("--seed", type=int, default=0)
    comparison_parser.add_argument("--gbm-estimators", type=int, default=32)
    comparison_parser.add_argument("--gbm-learning-rate", type=float, default=0.1)
    comparison_parser.add_argument("--gbm-min-samples-leaf", type=int, default=2)
    comparison_parser.add_argument("--gbm-min-gain", type=float, default=0.0)
    comparison_parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON",
    )

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
    train_parser.add_argument(
        "--ridge-solver",
        choices=("python-reference", "native-c11"),
        default="python-reference",
        help="reviewed ridge implementation (default: python-reference)",
    )
    train_parser.add_argument(
        "--native-ridge-binary",
        type=Path,
        help="absolute path to the authenticated C11 sidecar; native-c11 only",
    )
    train_parser.add_argument(
        "--native-ridge-sha256",
        help="exact lowercase SHA-256 of the C11 sidecar; native-c11 only",
    )
    train_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    demo_parser = subparsers.add_parser(
        "demo",
        help="replay the self-contained learned-router stream showcase",
    )
    demo_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
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
    native_ridge_binary: Path | None = None,
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

    protected_paths = (
        (data_source,) if native_ridge_binary is None else (data_source, native_ridge_binary)
    )
    saved = replace_text_bundle(tuple(writes), protected_paths=protected_paths)
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
    print(f"  selected model:    {_safe_terminal_text(decision.model_id)}")
    print(f"  quoted cost:       {canonical_cost_text(decision.model_cost)}")
    print(
        "  predicted quality: "
        f"{decision.predicted_quality:.3f} ({_safe_terminal_text(decision.quality_kind)})"
    )
    print(f"  policy:            one-shot lambda={_fraction_label(decision.lambda_cost)}")
    print(f"  accounting scope:  {_safe_terminal_text(decision.accounting_scope)}")
    if decision.lambda_candidates_exhaustive is not None:
        print(
            "  lambda search:     "
            f"retained={decision.lambda_candidates_retained}/"
            f"{_candidate_total_label(decision.lambda_candidates_derived)}, "
            f"exhaustive={str(decision.lambda_candidates_exhaustive).lower()}, "
            f"strategy={_safe_terminal_text(decision.lambda_candidate_strategy)}, "
            f"observed_breakpoints={decision.lambda_observed_breakpoint_count}"
        )
    print(f"  reason:            {_safe_terminal_text(decision.reason)}")
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


def _evaluation_result_payload(
    *,
    name: str,
    report: EvaluationReport,
    score: ScoreSummary,
    gap_recovery: float | None,
    total_cost: Cost,
    quote_error: QuoteErrorReport,
) -> dict[str, Any]:
    """Serialize one learned or baseline report with identical cost evidence."""

    quote_error_by_tier = quote_error.by_tier()
    tier_cost_evidence = {}
    for tier_result in report.tiers:
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
        "name": name,
        "evaluation_scope": {
            "algorithm": report.evaluation_scope_algorithm,
            "sha256": report.evaluation_scope_sha256,
            "max_calls_per_query": report.max_calls_per_query,
        },
        "tier_quality": {
            tier.value: None if quality is None else round(quality, 6)
            for tier, quality in score.tier_quality.items()
        },
        "weighted_quality": (
            None if score.weighted_quality is None else round(score.weighted_quality, 6)
        ),
        "oracle_gap_recovery": (None if gap_recovery is None else round(gap_recovery, 6)),
        "total_cost": canonical_cost_text(total_cost),
        "total_realized_cost": canonical_cost_text(total_cost),
        "cost_evidence": {
            "scope": "executed-replay-calls; overall is cross-tier diagnostic only",
            "overall": _quote_error_summary_payload(quote_error.overall),
            "by_tier": tier_cost_evidence,
        },
        "feasible": all(tier.feasible for tier in report.tiers),
    }


def _baseline_payload(result: BaselineResult) -> dict[str, Any]:
    return _evaluation_result_payload(
        name=result.name,
        report=result.report,
        score=result.score,
        gap_recovery=result.gap_recovery,
        total_cost=result.total_cost,
        quote_error=result.quote_error,
    )


def _lambda_search_payload(selection: TierLambdaSelection) -> dict[str, Any]:
    candidates = selection.candidates
    return {
        "selected_lambda": _fraction_payload(selection.lambda_cost),
        "retained_candidates": len(candidates.values),
        "derived_candidates": candidates.total_derived_values,
        "exhaustive": candidates.exhaustive,
        "strategy": candidates.strategy,
        "observed_breakpoint_count": candidates.observed_breakpoint_count,
    }


def _benchmark_predictor_payload(result: PerQueryNestedLodoBenchmark) -> dict[str, Any]:
    """Serialize one exact supported in-memory family without weakening artifact v1."""

    config = result.training_config
    if type(config) is BilinearTrainingConfig:
        return {
            "kind": result.predictor_kind,
            "feature_set": "surface-only",
            "ridge": config.ridge,
            "seed": config.seed,
            "solver_id": config.solver_id,
        }
    if type(config) is GbmTrainingConfig:
        return {
            "kind": result.predictor_kind,
            "feature_set": "surface-only",
            "algorithm_id": GBM_ALGORITHM_ID,
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "min_samples_leaf": config.min_samples_leaf,
            "min_gain": config.min_gain,
        }
    raise TypeError("benchmark uses an unsupported training configuration")


def _benchmark_payload(
    result: PerQueryNestedLodoBenchmark,
    *,
    dataset_name: str,
    dataset_license: str,
    provenance: str,
    bundled_synthetic: bool,
) -> dict[str, Any]:
    scope = result.learned.report.evaluation_scope
    baseline_config = result.baseline_config
    lambda_search_config = result.lambda_search_config
    learned = _evaluation_result_payload(
        name=result.learned.report.router_name,
        report=result.learned.report,
        score=result.learned.score,
        gap_recovery=result.learned_gap_recovery,
        total_cost=result.learned_total_cost,
        quote_error=result.learned_quote_error,
    )
    learned["prediction_sha256"] = result.learned.prediction_sha256
    folds = []
    for fold, membership in zip(
        result.learned.folds,
        result.fold_memberships,
        strict=True,
    ):
        tuning = fold.tuning
        folds.append(
            {
                "held_out_domain": fold.held_out_domain,
                "training_examples": membership.training_example_count,
                "test_examples": membership.test_example_count,
                "membership": {
                    "algorithm": membership.algorithm,
                    "sha256": membership.sha256,
                },
                "inner_tuning": {
                    "data_sha256": tuning.data_sha256,
                    "replay_sha256": tuning.replay_sha256,
                    "prediction_sha256": tuning.prediction_sha256,
                    "evaluation_scope": {
                        "algorithm": tuning.report.evaluation_scope_algorithm,
                        "sha256": tuning.report.evaluation_scope_sha256,
                        "max_calls_per_query": tuning.report.max_calls_per_query,
                    },
                    "lambda_search": {
                        selection.tier.value: _lambda_search_payload(selection)
                        for selection in tuning.selections
                    },
                    "weighted_quality": (
                        None
                        if tuning.score.weighted_quality is None
                        else round(tuning.score.weighted_quality, 6)
                    ),
                },
            }
        )
    learned["outer_folds"] = folds
    return {
        "schema": "tierroute-benchmark",
        "schema_version": 1,
        "command": "benchmark",
        "dataset": dataset_name,
        "dataset_license": dataset_license,
        "provenance": provenance,
        "claim_scope": (
            "project-authored-synthetic-wiring-only"
            if bundled_synthetic
            else "user-supplied-replay; claim and license validity are caller responsibilities"
        ),
        "network_used": False,
        "budget_scope": result.accounting_scope,
        "validation_scope": "true-nested-lodo-original-order",
        "cost_scope": "per-tier ledgers; cross-tier total is diagnostic only",
        "data_sha256": result.data_sha256,
        "replay_sha256": result.replay_sha256,
        "example_count": result.example_count,
        "domains": list(result.domains),
        "model_ids": list(result.model_ids),
        "tier_specs": [
            {
                "tier": tier_result.tier_spec.tier.value,
                "budget_limit": canonical_cost_text(tier_result.tier_spec.budget_limit),
                "weight": float(tier_result.tier_spec.weight),
            }
            for tier_result in result.learned.report.tiers
        ],
        "evaluation_scope": {
            "algorithm": scope.algorithm,
            "sha256": scope.sha256,
            "max_calls_per_query": scope.max_calls_per_query,
        },
        "predictor": _benchmark_predictor_payload(result),
        "lambda_search_config": {
            "requested_mode": lambda_search_config.requested_mode,
            "max_candidates_per_tier": lambda_search_config.max_candidates_per_tier,
            "allow_large_exhaustive": lambda_search_config.allow_large_exhaustive,
        },
        "baseline_config": {
            "schema": baseline_config.schema,
            "evidence": {
                "algorithm": result.baselines.baseline_config_evidence_algorithm,
                "sha256": result.baselines.baseline_config_evidence_sha256,
            },
            "always_cheapest": {
                "model_id": baseline_config.cheap_model_id,
                "selection_rule": baseline_config.cheapest_model_selection_rule,
            },
            "always_premium": {
                "model_id": baseline_config.premium_model_id,
                "role_selection_rule": baseline_config.premium_model_selection_rule,
            },
            "random": {
                "seed": baseline_config.random_seed,
                "selection_algorithm": baseline_config.random_selection_algorithm,
            },
            "length_heuristic": {
                "character_threshold": baseline_config.character_threshold,
                "cheap_model_id": baseline_config.cheap_model_id,
                "strong_model_id": baseline_config.strong_model_id,
                "difficulty_rule": baseline_config.length_difficulty_rule,
            },
            "oracle": {
                "selection_rule": baseline_config.oracle_selection_rule,
            },
            "domain_best_table": {
                "fit_rule": baseline_config.domain_table_fit_rule,
                "unseen_tag_fallback_model_id": baseline_config.cheap_model_id,
            },
        },
        "learned_router": learned,
        "baselines": [_baseline_payload(row) for row in result.baselines.baselines],
    }


def _comparison_delta_payload(
    *,
    tier_quality: Mapping[BudgetTier, float | None],
    weighted_quality: float | None,
    oracle_gap_recovery: float | None,
) -> dict[str, Any]:
    """Preserve binary64 subtraction results without presentation rounding."""

    return {
        "tier_quality": {
            tier.value: tier_quality[tier]
            for tier in (BudgetTier.FAST, BudgetTier.BALANCED, BudgetTier.PREMIUM)
            if tier in tier_quality
        },
        "weighted_quality": weighted_quality,
        "oracle_gap_recovery": oracle_gap_recovery,
    }


def _predictor_comparison_payload(
    result: PairedPredictorComparison,
    *,
    dataset_name: str,
    dataset_license: str,
    provenance: str,
    bundled_synthetic: bool,
) -> dict[str, Any]:
    """Serialize paired estimation once, without creating a family-selection claim."""

    bilinear = _benchmark_payload(
        result.bilinear,
        dataset_name=dataset_name,
        dataset_license=dataset_license,
        provenance=provenance,
        bundled_synthetic=bundled_synthetic,
    )
    gbm = _benchmark_payload(
        result.gbm,
        dataset_name=dataset_name,
        dataset_license=dataset_license,
        provenance=provenance,
        bundled_synthetic=bundled_synthetic,
    )
    if bilinear["baselines"] != gbm["baselines"]:
        raise ValueError("paired predictor payload requires one shared baseline evaluation")
    return {
        "schema": "tierroute-predictor-comparison",
        "schema_version": 1,
        "command": "compare-predictors",
        "dataset": dataset_name,
        "dataset_license": dataset_license,
        "provenance": provenance,
        "claim_state": "SYNTHETIC-ONLY" if bundled_synthetic else "UNVERIFIED-USER-DATA",
        "network_used": False,
        "budget_scope": result.bilinear.accounting_scope,
        "validation_scope": "true-nested-lodo-original-order",
        "evidence_role": "descriptive-paired-estimation-only",
        "comparison_direction": result.comparison_direction,
        "selection_protocol": result.selection_protocol,
        "selected_family": result.selected_family,
        "performance_claim_allowed": result.performance_claim_allowed,
        "data_sha256": result.bilinear.data_sha256,
        "replay_sha256": result.bilinear.replay_sha256,
        "example_count": result.bilinear.example_count,
        "domains": list(result.bilinear.domains),
        "model_ids": list(result.bilinear.model_ids),
        "tier_specs": bilinear["tier_specs"],
        "evaluation_scope": bilinear["evaluation_scope"],
        "lambda_search_config": bilinear["lambda_search_config"],
        "baseline_config": bilinear["baseline_config"],
        "predictor_families": {
            "bilinear": {
                "predictor": bilinear["predictor"],
                "learned_router": bilinear["learned_router"],
                "paired_metrics_full_precision": _comparison_delta_payload(
                    tier_quality=result.bilinear.learned.score.tier_quality,
                    weighted_quality=result.bilinear.learned.score.weighted_quality,
                    oracle_gap_recovery=result.bilinear.learned_gap_recovery,
                ),
            },
            "gbm": {
                "predictor": gbm["predictor"],
                "learned_router": gbm["learned_router"],
                "paired_metrics_full_precision": _comparison_delta_payload(
                    tier_quality=result.gbm.learned.score.tier_quality,
                    weighted_quality=result.gbm.learned.score.weighted_quality,
                    oracle_gap_recovery=result.gbm.learned_gap_recovery,
                ),
            },
        },
        "deltas": {
            "direction": result.comparison_direction,
            "overall": _comparison_delta_payload(
                tier_quality=result.tier_quality_delta,
                weighted_quality=result.weighted_quality_delta,
                oracle_gap_recovery=result.oracle_gap_recovery_delta,
            ),
            "held_out_domains": [
                {
                    "held_out_domain": row.held_out_domain,
                    **_comparison_delta_payload(
                        tier_quality=row.tier_quality_delta,
                        weighted_quality=row.weighted_quality_delta,
                        oracle_gap_recovery=row.oracle_gap_recovery_delta,
                    ),
                }
                for row in result.held_out_domain_deltas
            ],
        },
        "baselines": bilinear["baselines"],
    }


def _showcase_payload(result: RoutingStreamShowcase) -> dict[str, Any]:
    """Serialize the stream separately from its full-population benchmark evidence."""

    benchmark = _benchmark_payload(
        result.benchmark,
        dataset_name=result.dataset.name,
        dataset_license=result.dataset.license,
        provenance=result.dataset.provenance,
        bundled_synthetic=True,
    )
    steps = []
    for step in result.steps:
        retention = step.cumulative_quality_retention
        steps.append(
            {
                "index": step.index,
                "example_id": step.example_id,
                "prompt": step.prompt,
                "tier": step.tier.value,
                "budget_limit": canonical_cost_text(step.budget_limit),
                "evaluation_scope": {
                    "algorithm": step.evaluation_scope.algorithm,
                    "sha256": step.evaluation_scope.sha256,
                    "max_calls_per_query": step.evaluation_scope.max_calls_per_query,
                },
                "routing": {
                    "model": step.selected_model_id,
                    "lambda_cost": _fraction_payload(step.lambda_cost),
                    "predicted_quality": round(step.predicted_quality, 6),
                    "reason": step.decision_reason,
                    "audited_benchmark_query_match": True,
                },
                "cost": {
                    "quoted": canonical_cost_text(step.quoted_cost),
                    "realized": canonical_cost_text(step.realized_cost),
                    "cumulative_realized_reporting_only": canonical_cost_text(
                        step.cumulative_realized_cost
                    ),
                },
                "quality": {
                    "kind": "project-authored-synthetic-observed-replay",
                    "observed": round(step.observed_quality, 6),
                    "per_query_oracle": {
                        "model": step.oracle_model_id,
                        "realized_cost": canonical_cost_text(step.oracle_realized_cost),
                        "quality": round(step.oracle_quality, 6),
                    },
                    "cumulative_observed": round(
                        float(step.cumulative_observed_quality),
                        6,
                    ),
                    "cumulative_observed_exact": _fraction_payload(
                        step.cumulative_observed_quality
                    ),
                    "cumulative_per_query_oracle": round(
                        float(step.cumulative_oracle_quality),
                        6,
                    ),
                    "cumulative_per_query_oracle_exact": _fraction_payload(
                        step.cumulative_oracle_quality
                    ),
                    "cumulative_retention": (
                        None if retention is None else round(float(retention), 6)
                    ),
                    "cumulative_retention_exact": (
                        None if retention is None else _fraction_payload(retention)
                    ),
                },
            }
        )
    retention = result.quality_retention
    return {
        "schema": "tierroute-routing-stream-showcase",
        "schema_version": 1,
        "command": "demo",
        "stream_id": result.stream_id,
        "dataset": result.dataset.name,
        "dataset_license": result.dataset.license,
        "provenance": result.dataset.provenance,
        "claim_scope": "project-authored-synthetic-wiring-only",
        "network_used": False,
        "data_sha256": result.data_sha256,
        "replay_sha256": result.replay_sha256,
        "policy": {
            "kind": result.benchmark.predictor_kind,
            "routing": "one-shot exact-rational tier lambda",
            "validation": "direct one-row replays match audited true nested-LODO queries",
        },
        "accounting": {
            "budget_scope": "independent-per-query-illustrative",
            "cumulative_cost_scope": ("mixed-tier-reporting-only; not an official shared budget"),
            "oracle_scope": "independent-budget-feasible-per-query-evaluation-only",
            "quality_retention_formula": (
                "sum(observed synthetic quality) / sum(independent per-query oracle quality)"
            ),
            "quality_retention_scope": (
                "unweighted display ratio; not a sequence-level oracle or oracle-gap recovery"
            ),
        },
        "stream": {
            "step_count": len(result.steps),
            "steps": steps,
            "totals": {
                "realized_cost_reporting_only": canonical_cost_text(result.total_realized_cost),
                "observed_quality": round(float(result.total_observed_quality), 6),
                "observed_quality_exact": _fraction_payload(result.total_observed_quality),
                "per_query_oracle_quality": round(float(result.total_oracle_quality), 6),
                "per_query_oracle_quality_exact": _fraction_payload(result.total_oracle_quality),
                "quality_retention": (None if retention is None else round(float(retention), 6)),
                "quality_retention_exact": (
                    None if retention is None else _fraction_payload(retention)
                ),
            },
        },
        "benchmark_evidence": benchmark,
    }


def _print_showcase(result: RoutingStreamShowcase) -> None:
    """Render the exact three-tier stream before the separate benchmark table."""

    print("tierroute offline routing stream showcase\n")
    print(f"Dataset: {_safe_terminal_text(result.dataset.name)}")
    print("Claim: project-authored synthetic wiring evidence, not benchmark results")
    print("Budget scope: independent illustrative per-query limits")
    print("Running cost: mixed-tier reporting-only; not an official shared budget")
    print(
        "Retention: unweighted sum(observed synthetic quality) / "
        "sum(independent per-query oracle quality)"
    )
    print("Oracle note: not a sequence-level oracle or oracle-gap recovery\n")
    for step in result.steps:
        retention = step.cumulative_quality_retention
        retention_label = "N/A" if retention is None else f"{100 * float(retention):.1f}%"
        print(f"Step {step.index} [{step.tier.value}] {_safe_terminal_text(step.example_id)}")
        print(f"  prompt:            {_safe_terminal_text(step.prompt)}")
        print(
            f"  route:             {_safe_terminal_text(step.selected_model_id)} "
            f"(one-shot lambda={_fraction_label(step.lambda_cost)})"
        )
        print(
            "  cost:              "
            f"budget={canonical_cost_text(step.budget_limit)}, "
            f"quoted={canonical_cost_text(step.quoted_cost)}, "
            f"realized={canonical_cost_text(step.realized_cost)}"
        )
        print(
            "  synthetic quality: "
            f"observed={step.observed_quality:.3f}, "
            "per-query oracle="
            f"{step.oracle_quality:.3f} ({_safe_terminal_text(step.oracle_model_id)})"
        )
        print(
            "  running display:   "
            f"cost={canonical_cost_text(step.cumulative_realized_cost)}, "
            f"quality retention={retention_label}"
        )
    final_retention = result.quality_retention
    final_label = "N/A" if final_retention is None else f"{100 * float(final_retention):.1f}%"
    print("\nStream summary:")
    print(
        f"  realized cost:      {canonical_cost_text(result.total_realized_cost)} (reporting-only)"
    )
    print(f"  quality retention:  {final_label} (unweighted display ratio)")
    print("  network:            disabled")


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
    native_ridge_sha256: str | None = None,
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
    }
    if native_ridge_sha256 is None:
        payload["network_used"] = False
    else:
        payload.update(
            {
                "native_binary_audit": "caller-responsibility-unapproved",
                "native_ridge_sha256": native_ridge_sha256,
                "network_used": None,
                "python_orchestration_network_used": False,
            }
        )
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
    print(f"Dataset: {_safe_terminal_text(dataset_name)}")
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


def _print_benchmark(
    result: PerQueryNestedLodoBenchmark,
    dataset_name: str,
    *,
    bundled_synthetic: bool,
) -> None:
    """Render judge-facing learned-versus-baseline evidence without hiding scope."""

    print("tierroute nested-LODO benchmark\n")
    print(f"Dataset: {_safe_terminal_text(dataset_name)}")
    print("Budget scope: explicit per-query accounting")
    print("Validation: true nested LODO; outer predictions replay once in original order")
    print(
        "Evaluation scope: "
        f"{result.learned.report.evaluation_scope_algorithm}:"
        f"{result.learned.report.evaluation_scope_sha256} "
        f"(max calls/query={result.learned.report.max_calls_per_query})"
    )
    header = (
        f"{'method':<26} {'fast':>7} {'balanced':>9} {'premium':>8} "
        f"{'weighted':>9} {'gap':>7} {'cost':>8}"
    )
    print(header)
    print("-" * len(header))

    def print_row(
        name: str,
        score: ScoreSummary,
        gap: float | None,
        total_cost: Cost,
    ) -> None:
        quality = score.tier_quality
        print(
            f"{name:<26} "
            f"{_format_score(quality.get(BudgetTier.FAST)):>7} "
            f"{_format_score(quality.get(BudgetTier.BALANCED)):>9} "
            f"{_format_score(quality.get(BudgetTier.PREMIUM)):>8} "
            f"{_format_score(score.weighted_quality):>9} "
            f"{_format_score(gap):>7} "
            f"{canonical_cost_text(total_cost):>8}"
        )

    print_row(
        "tierroute-nested-lodo",
        result.learned.score,
        result.learned_gap_recovery,
        result.learned_total_cost,
    )
    for row in result.baselines.baselines:
        print_row(row.name, row.score, row.gap_recovery, row.total_cost)

    print("\nLearned tier ledger evidence:")
    print(f"{'tier':<10} {'quality':>9} {'spent':>10} {'effective limit':>16} {'feasible':>10}")
    for tier_result in result.learned.report.tiers:
        quality = tier_result.mean_quality
        print(
            f"{tier_result.tier_spec.tier.value:<10} "
            f"{_format_score(quality):>9} "
            f"{canonical_cost_text(tier_result.budget.spent):>10} "
            f"{canonical_cost_text(tier_result.budget.effective_total_limit):>16} "
            f"{str(tier_result.feasible).lower():>10}"
        )
    print("\nOuter folds: " + ", ".join(_safe_terminal_text(item) for item in result.domains))
    print(f"Outer prediction SHA-256: {result.learned.prediction_sha256}")
    print("Cost note: the cross-tier total is a diagnostic over independent tier ledgers.")
    if bundled_synthetic:
        print("Claim note: bundled values are synthetic wiring evidence, not benchmark results.")
    else:
        print(
            "Claim note: user-supplied replay; claim and license validity are caller "
            "responsibilities."
        )
    print("Network: disabled")


def _print_predictor_comparison(
    result: PairedPredictorComparison,
    dataset_name: str,
    *,
    bundled_synthetic: bool,
) -> None:
    """Render fixed-order paired estimates without ranking the predictor families."""

    print("tierroute paired predictor estimation (wiring only)\n")
    print(f"Dataset: {_safe_terminal_text(dataset_name)}")
    print(f"Claim state: {'SYNTHETIC-ONLY' if bundled_synthetic else 'UNVERIFIED-USER-DATA'}")
    print("Budget scope: explicit per-query accounting")
    print("Validation: both families use identical true nested-LODO outer evidence")
    header = (
        f"{'method':<26} {'fast':>7} {'balanced':>9} {'premium':>8} "
        f"{'weighted':>9} {'gap':>7} {'cost':>8}"
    )
    print(header)
    print("-" * len(header))

    def print_row(
        name: str,
        score: ScoreSummary,
        gap: float | None,
        total_cost: Cost,
    ) -> None:
        quality = score.tier_quality
        print(
            f"{name:<26} "
            f"{_format_score(quality.get(BudgetTier.FAST)):>7} "
            f"{_format_score(quality.get(BudgetTier.BALANCED)):>9} "
            f"{_format_score(quality.get(BudgetTier.PREMIUM)):>8} "
            f"{_format_score(score.weighted_quality):>9} "
            f"{_format_score(gap):>7} "
            f"{canonical_cost_text(total_cost):>8}"
        )

    print_row(
        "bilinear",
        result.bilinear.learned.score,
        result.bilinear.learned_gap_recovery,
        result.bilinear.learned_total_cost,
    )
    print_row(
        "gbm",
        result.gbm.learned.score,
        result.gbm.learned_gap_recovery,
        result.gbm.learned_total_cost,
    )
    for row in result.bilinear.baselines.baselines:
        print_row(row.name, row.score, row.gap_recovery, row.total_cost)

    print("\nDelta direction: GBM - bilinear (descriptive paired estimate)")
    print(
        "Weighted-quality delta: "
        f"{_format_score(result.weighted_quality_delta)}; "
        "oracle-gap delta: "
        f"{_format_score(result.oracle_gap_recovery_delta)}"
    )
    print("Selection: not performed; this same outer evidence cannot select a winner.")
    print("Performance claim: prohibited for this comparison output.")
    print("Network: disabled")


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI and return a process status code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    ridge_solver: RidgeSolver | None = None
    ridge_solver_id = CENTERED_RIDGE_SOLVER_ID
    native_ridge_binary: Path | None = None
    native_ridge_sha256: str | None = None
    if args.command == "route" and args.policy_artifact is not None and args.artifact is None:
        parser.error("route --policy-artifact requires --artifact")
    if args.command == "route" and args.remaining_budget is not None:
        if args.policy_artifact is None:
            parser.error("route --remaining-budget requires --policy-artifact")
    if args.command == "train":
        native_ridge_binary = args.native_ridge_binary
        native_ridge_sha256 = args.native_ridge_sha256
        if args.ridge_solver == "python-reference":
            if native_ridge_binary is not None or native_ridge_sha256 is not None:
                parser.error(
                    "train --native-ridge-binary and --native-ridge-sha256 require "
                    "--ridge-solver native-c11"
                )
        else:
            if native_ridge_binary is None or native_ridge_sha256 is None:
                parser.error(
                    "train --ridge-solver native-c11 requires both "
                    "--native-ridge-binary and --native-ridge-sha256"
                )
            try:
                ridge_solver = NativeRidgeAdapter(
                    native_ridge_binary,
                    native_ridge_sha256,
                )
            except (TypeError, ValueError) as error:
                parser.error(f"invalid native ridge configuration: {error}")
            ridge_solver_id = NATIVE_C11_RIDGE_SOLVER_ID
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
        protected_paths = (
            (data_source,) if native_ridge_binary is None else (data_source, native_ridge_binary)
        )
        try:
            validate_write_paths(destinations, protected_paths=protected_paths)
        except ValueError as error:
            parser.error(f"train artifact paths are unsafe: {error}")
    if args.command == "benchmark":
        if args.allow_large_exhaustive_search and not args.exhaustive_lambda_search:
            parser.error(
                "benchmark --allow-large-exhaustive-search requires --exhaustive-lambda-search"
            )
        if args.max_lambda_candidates is not None and args.max_lambda_candidates < 2:
            parser.error("benchmark --max-lambda-candidates must be at least 2")
    if args.command == "compare-predictors":
        if args.allow_large_exhaustive_search and not args.exhaustive_lambda_search:
            parser.error(
                "compare-predictors --allow-large-exhaustive-search requires "
                "--exhaustive-lambda-search"
            )
        if args.max_lambda_candidates is not None and args.max_lambda_candidates < 2:
            parser.error("compare-predictors --max-lambda-candidates must be at least 2")
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
        config = BilinearTrainingConfig(
            ridge=args.ridge,
            seed=args.seed,
            solver_id=ridge_solver_id,
        )
        tuning = None
        policy = None
        try:
            artifact = fit_calibrated_bilinear(
                dataset.examples,
                config=config,
                solver=ridge_solver,
            )
            if args.policy_output is not None:
                predictions = cross_fitted_prediction_table(
                    dataset.examples,
                    lambda training: fit_calibrated_bilinear(
                        training,
                        config=config,
                        solver=ridge_solver,
                    ).build_predictor(),
                )
                ledger_factory = (
                    PerQueryBudgetLedger
                    if args.budget_scope == "per-query"
                    else CumulativeBudgetLedger
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
        except NativeRidgeError as error:
            parser.error(f"native ridge training failed: {_safe_terminal_text(str(error))}")
        data_source = args.data if args.data is not None else bundled_synthetic_path()
        output, policy_output = _save_training_artifacts(
            artifact,
            args.output,
            policy=policy,
            policy_output=args.policy_output,
            data_source=data_source,
            native_ridge_binary=native_ridge_binary,
        )
        payload = _training_payload(
            artifact,
            output,
            dataset.name,
            policy_output=policy_output,
            tuning=tuning,
            accounting_scope=args.budget_scope,
            native_ridge_sha256=native_ridge_sha256,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print("tierroute predictor training")
            print(f"  dataset:            {_safe_terminal_text(dataset.name)}")
            print(f"  training examples:  {artifact.training_example_count}")
            print(
                "  training domains:   "
                + ", ".join(_safe_terminal_text(item) for item in artifact.training_domains)
            )
            print(
                "  candidate models:   "
                + ", ".join(_safe_terminal_text(item) for item in artifact.model_ids)
            )
            print(f"  feature dimension:  {artifact.feature_schema.dimension}")
            print(f"  ridge solver:       {artifact.solver_id}")
            if native_ridge_sha256 is not None:
                print(f"  native SHA-256:     {native_ridge_sha256}")
            print(f"  artifact:           {_safe_terminal_text(str(output))}")
            if policy_output is not None and tuning is not None:
                print(f"  policy artifact:    {_safe_terminal_text(str(policy_output))}")
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
            if native_ridge_sha256 is None:
                print("  network:            disabled")
            else:
                print("  Python network:     disabled")
                print("  native audit:       caller-responsibility-unapproved")
                print("  total network use:  not asserted")
            print("  note: synthetic data is a wiring test, not benchmark evidence")
        return 0

    if args.command == "benchmark":
        candidate_cap = (
            None
            if args.exhaustive_lambda_search
            else (args.max_lambda_candidates or DEFAULT_MAX_LAMBDA_CANDIDATES)
        )
        config = BilinearTrainingConfig(ridge=args.ridge, seed=args.seed)
        benchmark = evaluate_per_query_bilinear_benchmark(
            dataset.examples,
            dataset.tier_specs,
            config=config,
            max_candidates_per_tier=candidate_cap,
            allow_large_exhaustive=args.allow_large_exhaustive_search,
        )
        if args.json:
            print(
                json.dumps(
                    _benchmark_payload(
                        benchmark,
                        dataset_name=dataset.name,
                        dataset_license=dataset.license,
                        provenance=dataset.provenance,
                        bundled_synthetic=args.data is None,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            _print_benchmark(
                benchmark,
                dataset.name,
                bundled_synthetic=args.data is None,
            )
        return 0

    if args.command == "compare-predictors":
        candidate_cap = (
            None
            if args.exhaustive_lambda_search
            else (args.max_lambda_candidates or DEFAULT_MAX_LAMBDA_CANDIDATES)
        )
        comparison = evaluate_per_query_paired_predictor_comparison(
            dataset.examples,
            dataset.tier_specs,
            bilinear_config=BilinearTrainingConfig(ridge=args.ridge, seed=args.seed),
            gbm_config=GbmTrainingConfig(
                n_estimators=args.gbm_estimators,
                learning_rate=args.gbm_learning_rate,
                min_samples_leaf=args.gbm_min_samples_leaf,
                min_gain=args.gbm_min_gain,
            ),
            max_candidates_per_tier=candidate_cap,
            allow_large_exhaustive=args.allow_large_exhaustive_search,
        )
        if args.json:
            print(
                json.dumps(
                    _predictor_comparison_payload(
                        comparison,
                        dataset_name=dataset.name,
                        dataset_license=dataset.license,
                        provenance=dataset.provenance,
                        bundled_synthetic=args.data is None,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            _print_predictor_comparison(
                comparison,
                dataset.name,
                bundled_synthetic=args.data is None,
            )
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
        benchmark = evaluate_per_query_bilinear_benchmark(
            dataset.examples,
            dataset.tier_specs,
            config=BilinearTrainingConfig(),
            max_candidates_per_tier=DEFAULT_MAX_LAMBDA_CANDIDATES,
        )
        showcase = build_routing_stream_showcase(dataset, benchmark)
        if args.json:
            print(
                json.dumps(
                    _showcase_payload(showcase),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            _print_showcase(showcase)
            print("\nSeparate full-population learned + six-baseline evidence:\n")
            _print_benchmark(benchmark, dataset.name, bundled_synthetic=True)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
