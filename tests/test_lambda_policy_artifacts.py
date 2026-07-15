# SPDX-License-Identifier: Apache-2.0
"""Tests for strict, provenance-bound exact-lambda policy artifacts."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

import tierroute.policies.lambda_artifacts as lambda_artifacts
from tierroute.adapters import PerQueryBudgetLedger
from tierroute.core import BudgetTier, CallModel, ModelSpec, RouterState
from tierroute.eval import (
    CandidateOutcome,
    EvaluationExample,
    TierSpec,
    evaluation_data_sha256,
)
from tierroute.features import PromptFeatureSchema
from tierroute.policies.lambda_artifacts import (
    LambdaPolicyArtifact,
    predictor_artifact_sha256,
)
from tierroute.policies.lambda_tuning import (
    CrossFittedPredictionTable,
    TierLambdaTuningResult,
    derive_lambda_candidate_set,
    tune_tier_lambdas,
)
from tierroute.predictors import BilinearPredictorArtifact, IsotonicCalibrator


def _policy_examples() -> tuple[EvaluationExample, ...]:
    models = (
        ModelSpec("cheap", Decimal("1")),
        ModelSpec("premium", Decimal("2")),
    )
    return tuple(
        EvaluationExample(
            example_id=f"q-{index}",
            prompt=prompt,
            domain=domain,
            outcomes=(
                CandidateOutcome("cheap", "cheap output", Decimal("1"), 0.5),
                CandidateOutcome("premium", "premium output", Decimal("2"), 0.9),
            ),
            candidate_models=models,
        )
        for index, (prompt, domain) in enumerate(
            (("Explain this idea.", "general"), ("Prove x + 0 = x.", "math")),
            start=1,
        )
    )


@pytest.fixture(scope="module")
def predictor_artifact() -> BilinearPredictorArtifact:
    examples = _policy_examples()
    schema = PromptFeatureSchema.fit(tuple(example.prompt for example in examples))
    zero_weights = tuple(0.0 for _ in range(schema.dimension))
    return BilinearPredictorArtifact(
        feature_schema=schema,
        model_weights={"cheap": zero_weights, "premium": zero_weights},
        model_bias={"cheap": 0.0, "premium": 0.0},
        calibrators={
            "cheap": IsotonicCalibrator((0.0,), (0.4,)),
            "premium": IsotonicCalibrator((0.0,), (0.9,)),
        },
        training_data_sha256=evaluation_data_sha256(examples),
        training_example_count=2,
        training_domains=("general", "math"),
        ridge=0.1,
        seed=7,
    )


@pytest.fixture(scope="module")
def tuning_result() -> TierLambdaTuningResult:
    examples = _policy_examples()
    models = examples[0].candidate_models
    predictions = CrossFittedPredictionTable(
        {
            (example.example_id, model.model_id): (0.4 if model.model_id == "cheap" else 0.9)
            for example in examples
            for model in models
        }
    )
    specs = (
        TierSpec(BudgetTier.FAST, Decimal("2"), 0.75),
        TierSpec(BudgetTier.PREMIUM, Decimal("2"), 0.25),
    )
    return tune_tier_lambdas(
        examples,
        specs,
        predictions,
        PerQueryBudgetLedger,
        lambda_grids={spec.tier: (0, 1) for spec in specs},
    )


@pytest.fixture(scope="module")
def policy_artifact(
    predictor_artifact: BilinearPredictorArtifact,
    tuning_result: TierLambdaTuningResult,
) -> LambdaPolicyArtifact:
    specs = tuple(selection.report.tier_spec for selection in tuning_result.selections)
    return LambdaPolicyArtifact.from_tuning(
        predictor_artifact,
        tuning_result,
        specs,
        "per-query",
    )


def test_policy_json_is_canonical_and_round_trips(
    policy_artifact: LambdaPolicyArtifact,
    tmp_path: Path,
) -> None:
    document = policy_artifact.to_json()
    path = policy_artifact.save(tmp_path / "nested" / "policy.json")

    loaded = LambdaPolicyArtifact.load(path)

    assert path == tmp_path / "nested" / "policy.json"
    assert document.endswith("\n")
    assert document == (
        json.dumps(
            policy_artifact.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    assert loaded.to_json() == document
    assert loaded.lambda_by_tier == policy_artifact.lambda_by_tier
    assert json.loads(document)["lambdas"]["fast"] == {
        "denominator": "1",
        "numerator": "0",
    }
    candidate_set = json.loads(document)["tuning"]["candidate_sets"][0]
    assert candidate_set["strategy"] == "explicit-grid-v1"
    assert candidate_set["observed_breakpoint_count"] == 0
    assert candidate_set["total_derived_values"] == 2
    assert candidate_set["exhaustive"] is False


def test_unknown_bounded_candidate_total_round_trips_as_json_null(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    payload = copy.deepcopy(policy_artifact.to_dict())
    candidate_set = payload["tuning"]["candidate_sets"][0]  # type: ignore[index]
    candidate_set["total_derived_values"] = None
    candidate_set["exhaustive"] = False
    candidate_set["strategy"] = "bounded-bottom-hash-v1"
    candidate_set["observed_breakpoint_count"] = 17

    loaded = LambdaPolicyArtifact.from_dict(payload)
    restored = json.loads(loaded.to_json())["tuning"]["candidate_sets"][0]

    assert loaded.candidate_sets[0].total_derived_values is None
    assert restored["total_derived_values"] is None
    assert restored["strategy"] == "bounded-bottom-hash-v1"
    assert restored["observed_breakpoint_count"] == 17


def test_ten_thousand_digit_lambda_round_trips_without_interpreter_tuning(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    huge_lambda = Fraction(1, 10**10000)
    candidate_sets = tuple(
        replace(
            item,
            values=(huge_lambda,),
            total_derived_values=1,
            exhaustive=False,
            strategy="explicit-grid-v1",
            observed_breakpoint_count=0,
        )
        for item in policy_artifact.candidate_sets
    )
    huge_policy = replace(
        policy_artifact,
        lambda_by_tier={tier: huge_lambda for tier in policy_artifact.lambda_by_tier},
        candidate_sets=candidate_sets,
    )

    document = huge_policy.to_json()
    restored = LambdaPolicyArtifact.from_json(document)

    denominator = json.loads(document)["lambdas"]["fast"]["denominator"]
    assert len(denominator) == 10001
    assert restored.lambda_by_tier == huge_policy.lambda_by_tier


def test_policy_artifact_accepts_lambda_from_minimum_legal_cost(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    models = (
        ModelSpec("free", Decimal(0)),
        ModelSpec("positive", Decimal("1e-100000")),
    )
    example = EvaluationExample(
        example_id="wide-lambda",
        prompt="Choose an exact cost boundary.",
        domain="math",
        outcomes=(
            CandidateOutcome("free", "free", Decimal(0), 0.0),
            CandidateOutcome("positive", "positive", Decimal("1e-100000"), 1.0),
        ),
        candidate_models=models,
    )
    predictions = CrossFittedPredictionTable(
        {("wide-lambda", "free"): 0.0, ("wide-lambda", "positive"): 1.0}
    )
    candidates = derive_lambda_candidate_set(
        (example,),
        TierSpec(BudgetTier.FAST, Decimal("1e-100000"), 1.0),
        predictions,
        max_candidates=2,
    )
    selected = candidates.values[-1]
    assert selected > 10**100000

    aligned_candidates = tuple(
        replace(candidates, tier=item.tier) for item in policy_artifact.candidate_sets
    )
    wide_policy = replace(
        policy_artifact,
        lambda_by_tier={tier: selected for tier in policy_artifact.lambda_by_tier},
        candidate_sets=aligned_candidates,
    )

    assert LambdaPolicyArtifact.from_json(wide_policy.to_json()).lambda_by_tier == (
        wide_policy.lambda_by_tier
    )


def test_policy_artifact_rejects_oversized_exact_integers_before_parsing(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    payload = copy.deepcopy(policy_artifact.to_dict())
    payload["lambdas"]["fast"] = {  # type: ignore[index]
        "numerator": "1",
        "denominator": "1" + "0" * lambda_artifacts.MAX_POLICY_INTEGER_DECIMAL_DIGITS,
    }

    with pytest.raises(ValueError, match="denominator exceeds the policy artifact integer limit"):
        LambdaPolicyArtifact.from_dict(payload)

    oversized = Fraction(1, 10**lambda_artifacts.MAX_POLICY_INTEGER_DECIMAL_DIGITS)
    with pytest.raises(ValueError, match="policy artifact integer limit"):
        replace(
            policy_artifact,
            lambda_by_tier={tier: oversized for tier in policy_artifact.lambda_by_tier},
        )


def test_policy_artifact_bounds_ledger_adapter_metadata(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    oversized_name = "l" * (lambda_artifacts.MAX_POLICY_LEDGER_ADAPTER_NAME_BYTES + 1)

    with pytest.raises(ValueError, match="ledger_adapter_name exceeds"):
        replace(policy_artifact, ledger_adapter_name=oversized_name)


def test_policy_artifact_document_and_candidate_limits_apply_before_load(
    policy_artifact: LambdaPolicyArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = policy_artifact.to_json()
    path = tmp_path / "policy.json"
    path.write_text(document, encoding="utf-8")
    monkeypatch.setattr(lambda_artifacts, "MAX_POLICY_ARTIFACT_BYTES", len(document) - 1)

    with pytest.raises(ValueError, match="artifact exceeds"):
        LambdaPolicyArtifact.from_json(document)
    with pytest.raises(ValueError, match="artifact exceeds"):
        LambdaPolicyArtifact.load(path)
    with pytest.raises(ValueError, match="artifact exceeds"):
        policy_artifact.to_json()

    payload = copy.deepcopy(policy_artifact.to_dict())
    monkeypatch.setattr(lambda_artifacts, "MAX_POLICY_ARTIFACT_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(lambda_artifacts, "MAX_POLICY_CANDIDATES_PER_TIER", 1)
    with pytest.raises(ValueError, match="per-tier candidate limit"):
        LambdaPolicyArtifact.from_dict(payload)
    with pytest.raises(ValueError, match="per-tier candidate limit"):
        replace(policy_artifact)


def test_from_tuning_binds_provenance_and_builds_a_router(
    predictor_artifact: BilinearPredictorArtifact,
    tuning_result: TierLambdaTuningResult,
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    policy_artifact.validate_predictor(predictor_artifact)
    assert policy_artifact.predictor_sha256 == predictor_artifact_sha256(predictor_artifact)
    assert policy_artifact.tuning_data_sha256 == predictor_artifact.training_data_sha256
    assert policy_artifact.tuning_replay_sha256 == tuning_result.replay_sha256
    assert policy_artifact.prediction_sha256 == tuning_result.prediction_sha256
    assert policy_artifact.ledger_adapter_name == "per-query"

    router = policy_artifact.build_router(predictor_artifact)
    action = router.route(
        RouterState(
            prompt="Route this prompt.",
            budget_tier=BudgetTier.FAST,
            remaining_budget=Decimal("2"),
            candidate_models=(
                ModelSpec("cheap", Decimal("1")),
                ModelSpec("premium", Decimal("2")),
            ),
        )
    )

    assert isinstance(action, CallModel)
    assert action.model_id == "premium"
    assert router.lambda_by_tier == policy_artifact.lambda_by_tier

    specs = tuple(selection.report.tier_spec for selection in tuning_result.selections)
    with pytest.raises(ValueError, match="ledger adapter"):
        LambdaPolicyArtifact.from_tuning(
            predictor_artifact,
            tuning_result,
            specs,
            "cumulative",
        )

    mismatched_tuning = replace(tuning_result, data_sha256="b" * 64)
    with pytest.raises(ValueError, match="tuning data SHA-256"):
        LambdaPolicyArtifact.from_tuning(
            predictor_artifact,
            mismatched_tuning,
            specs,
            "per-query",
        )


def test_predictor_hash_and_training_metadata_mismatches_fail_closed(
    predictor_artifact: BilinearPredictorArtifact,
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    changed_bias = replace(
        predictor_artifact,
        model_bias={"cheap": 0.1, "premium": 0.0},
    )
    with pytest.raises(ValueError, match="predictor SHA-256"):
        policy_artifact.validate_predictor(changed_bias)
    with pytest.raises(ValueError, match="predictor SHA-256"):
        policy_artifact.build_router(changed_bias)

    changed_data = replace(predictor_artifact, training_data_sha256="b" * 64)
    data_bound_policy = replace(
        policy_artifact,
        predictor_sha256=predictor_artifact_sha256(changed_data),
    )
    with pytest.raises(ValueError, match="training-data hashes"):
        data_bound_policy.validate_predictor(changed_data)

    count_mismatch = replace(policy_artifact, training_example_count=3)
    with pytest.raises(ValueError, match="example counts"):
        count_mismatch.validate_predictor(predictor_artifact)

    domain_mismatch = replace(
        policy_artifact,
        training_domains=("general", "other"),
    )
    with pytest.raises(ValueError, match="training domains"):
        domain_mismatch.validate_predictor(predictor_artifact)


def test_tuning_data_validation_binds_content_and_replay_order(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    examples = _policy_examples()

    policy_artifact.validate_tuning_data(examples)
    with pytest.raises(ValueError, match="replay-order SHA-256"):
        policy_artifact.validate_tuning_data(tuple(reversed(examples)))

    changed = (replace(examples[0], prompt="changed prompt"), examples[1])
    with pytest.raises(ValueError, match="tuning-data SHA-256"):
        policy_artifact.validate_tuning_data(changed)


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        "[]",
        '{"artifact_version":1,"artifact_version":1}',
        '{"artifact_version":NaN}',
        '{"artifact_version":Infinity}',
        '{"artifact_version":-Infinity}',
        "[" * 2_000 + "0" + "]" * 2_000,
    ],
)
def test_parser_rejects_non_strict_json(document: str) -> None:
    with pytest.raises(ValueError):
        LambdaPolicyArtifact.from_json(document)


def test_artifact_rejects_unencodable_metadata_and_non_float_weights(
    policy_artifact: LambdaPolicyArtifact,
    predictor_artifact: BilinearPredictorArtifact,
) -> None:
    surrogate_ledger = copy.deepcopy(policy_artifact.to_dict())
    surrogate_ledger["tuning"]["ledger_adapter"] = "\ud800"  # type: ignore[index]
    with pytest.raises(ValueError, match="valid Unicode scalar values"):
        LambdaPolicyArtifact.from_dict(surrogate_ledger)

    surrogate_domain = copy.deepcopy(policy_artifact.to_dict())
    surrogate_domain["tuning"]["domains"] = ["\ud800"]  # type: ignore[index]
    with pytest.raises(ValueError, match="valid Unicode scalar values"):
        LambdaPolicyArtifact.from_dict(surrogate_domain)

    with pytest.raises(ValueError, match="valid Unicode"):
        replace(predictor_artifact, training_domains=("\ud800",))

    inexact_weight = copy.deepcopy(policy_artifact.to_dict())
    inexact_weight["tuning"]["tier_specs"][0]["weight"] = {  # type: ignore[index]
        "numerator": "1",
        "denominator": "10",
    }
    with pytest.raises(ValueError, match="exact finite-float encoding"):
        LambdaPolicyArtifact.from_dict(inexact_weight)

    overflowing_weight = copy.deepcopy(policy_artifact.to_dict())
    overflowing_weight["tuning"]["tier_specs"][0]["weight"] = {  # type: ignore[index]
        "numerator": "1" + "0" * 400,
        "denominator": "1",
    }
    with pytest.raises(ValueError, match="fit a finite float"):
        LambdaPolicyArtifact.from_dict(overflowing_weight)


def test_schema_rejects_missing_and_unknown_fields(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    valid = policy_artifact.to_dict()
    mutations: list[dict[str, object]] = []

    missing_top = copy.deepcopy(valid)
    missing_top.pop("tuning")
    mutations.append(missing_top)

    extra_top = copy.deepcopy(valid)
    extra_top["unexpected"] = True
    mutations.append(extra_top)

    missing_tuning = copy.deepcopy(valid)
    missing_tuning["tuning"].pop("domains")  # type: ignore[union-attr]
    mutations.append(missing_tuning)

    extra_tuning = copy.deepcopy(valid)
    extra_tuning["tuning"]["unexpected"] = True  # type: ignore[index]
    mutations.append(extra_tuning)

    missing_spec = copy.deepcopy(valid)
    missing_spec["tuning"]["tier_specs"][0].pop("weight")  # type: ignore[index,union-attr]
    mutations.append(missing_spec)

    extra_spec = copy.deepcopy(valid)
    extra_spec["tuning"]["tier_specs"][0]["unexpected"] = True  # type: ignore[index]
    mutations.append(extra_spec)

    missing_candidate = copy.deepcopy(valid)
    missing_candidate["tuning"]["candidate_sets"][0].pop("exhaustive")  # type: ignore[index,union-attr]
    mutations.append(missing_candidate)

    missing_candidate_strategy = copy.deepcopy(valid)
    missing_candidate_strategy["tuning"]["candidate_sets"][0].pop(  # type: ignore[index,union-attr]
        "strategy"
    )
    mutations.append(missing_candidate_strategy)

    missing_observed_count = copy.deepcopy(valid)
    missing_observed_count["tuning"]["candidate_sets"][0].pop(  # type: ignore[index,union-attr]
        "observed_breakpoint_count"
    )
    mutations.append(missing_observed_count)

    extra_candidate = copy.deepcopy(valid)
    extra_candidate["tuning"]["candidate_sets"][0]["unexpected"] = True  # type: ignore[index]
    mutations.append(extra_candidate)

    missing_fraction = copy.deepcopy(valid)
    missing_fraction["lambdas"]["fast"].pop("denominator")  # type: ignore[index,union-attr]
    mutations.append(missing_fraction)

    extra_fraction = copy.deepcopy(valid)
    extra_fraction["lambdas"]["fast"]["unexpected"] = "1"  # type: ignore[index]
    mutations.append(extra_fraction)

    for payload in mutations:
        with pytest.raises((TypeError, ValueError)):
            LambdaPolicyArtifact.from_dict(payload)


@pytest.mark.parametrize(
    "fraction",
    [
        {"numerator": 0, "denominator": "1"},
        {"numerator": "00", "denominator": "1"},
        {"numerator": "-0", "denominator": "1"},
        {"numerator": "+1", "denominator": "1"},
        {"numerator": "1.0", "denominator": "1"},
        {"numerator": "1", "denominator": 1},
        {"numerator": "1", "denominator": "0"},
        {"numerator": "1", "denominator": "-1"},
        {"numerator": "1", "denominator": "01"},
        {"numerator": "2", "denominator": "4"},
        {"numerator": "-1", "denominator": "1"},
    ],
)
def test_lambda_fraction_must_be_nonnegative_reduced_and_canonical(
    policy_artifact: LambdaPolicyArtifact,
    fraction: dict[str, object],
) -> None:
    payload = copy.deepcopy(policy_artifact.to_dict())
    payload["lambdas"]["fast"] = fraction  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        LambdaPolicyArtifact.from_dict(payload)


def test_tier_and_candidate_metadata_is_validated(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    valid = policy_artifact.to_dict()
    mutations: list[dict[str, object]] = []

    unknown_lambda_tier = copy.deepcopy(valid)
    unknown_lambda_tier["lambdas"]["turbo"] = unknown_lambda_tier["lambdas"].pop(  # type: ignore[index,union-attr]
        "fast"
    )
    mutations.append(unknown_lambda_tier)

    unknown_spec_tier = copy.deepcopy(valid)
    unknown_spec_tier["tuning"]["tier_specs"][0]["tier"] = "turbo"  # type: ignore[index]
    mutations.append(unknown_spec_tier)

    duplicate_spec_tier = copy.deepcopy(valid)
    duplicate_spec_tier["tuning"]["tier_specs"][1]["tier"] = "fast"  # type: ignore[index]
    mutations.append(duplicate_spec_tier)

    invalid_budget = copy.deepcopy(valid)
    invalid_budget["tuning"]["tier_specs"][0]["budget_limit"] = "NaN"  # type: ignore[index]
    mutations.append(invalid_budget)

    zero_weight = copy.deepcopy(valid)
    zero_weight["tuning"]["tier_specs"][0]["weight"] = {  # type: ignore[index]
        "numerator": "0",
        "denominator": "1",
    }
    mutations.append(zero_weight)

    unknown_candidate_tier = copy.deepcopy(valid)
    unknown_candidate_tier["tuning"]["candidate_sets"][0]["tier"] = "turbo"  # type: ignore[index]
    mutations.append(unknown_candidate_tier)

    misaligned_candidate_tier = copy.deepcopy(valid)
    misaligned_candidate_tier["tuning"]["candidate_sets"][0]["tier"] = "premium"  # type: ignore[index]
    mutations.append(misaligned_candidate_tier)

    empty_values = copy.deepcopy(valid)
    empty_values["tuning"]["candidate_sets"][0]["values"] = []  # type: ignore[index]
    mutations.append(empty_values)

    unsorted_values = copy.deepcopy(valid)
    unsorted_values["tuning"]["candidate_sets"][0]["values"] = [  # type: ignore[index]
        {"numerator": "1", "denominator": "1"},
        {"numerator": "0", "denominator": "1"},
    ]
    mutations.append(unsorted_values)

    duplicate_values = copy.deepcopy(valid)
    duplicate_values["tuning"]["candidate_sets"][0]["values"] = [  # type: ignore[index]
        {"numerator": "0", "denominator": "1"},
        {"numerator": "0", "denominator": "1"},
    ]
    mutations.append(duplicate_values)

    negative_value = copy.deepcopy(valid)
    negative_value["tuning"]["candidate_sets"][0]["values"][0] = {  # type: ignore[index]
        "numerator": "-1",
        "denominator": "1",
    }
    mutations.append(negative_value)

    boolean_total = copy.deepcopy(valid)
    boolean_total["tuning"]["candidate_sets"][0]["total_derived_values"] = True  # type: ignore[index]
    mutations.append(boolean_total)

    too_small_total = copy.deepcopy(valid)
    too_small_total["tuning"]["candidate_sets"][0]["total_derived_values"] = 1  # type: ignore[index]
    mutations.append(too_small_total)

    non_boolean_exhaustive = copy.deepcopy(valid)
    non_boolean_exhaustive["tuning"]["candidate_sets"][0]["exhaustive"] = 1  # type: ignore[index]
    mutations.append(non_boolean_exhaustive)

    inconsistent_exhaustive = copy.deepcopy(valid)
    inconsistent_exhaustive["tuning"]["candidate_sets"][0]["exhaustive"] = True  # type: ignore[index]
    inconsistent_exhaustive["tuning"]["candidate_sets"][0][  # type: ignore[index]
        "total_derived_values"
    ] = 3
    mutations.append(inconsistent_exhaustive)

    unknown_total_exhaustive = copy.deepcopy(valid)
    unknown_total_exhaustive["tuning"]["candidate_sets"][0][  # type: ignore[index]
        "total_derived_values"
    ] = None
    unknown_total_exhaustive["tuning"]["candidate_sets"][0][  # type: ignore[index]
        "exhaustive"
    ] = True
    mutations.append(unknown_total_exhaustive)

    unknown_strategy = copy.deepcopy(valid)
    unknown_strategy["tuning"]["candidate_sets"][0][  # type: ignore[index]
        "strategy"
    ] = "random-breakpoints-v1"
    mutations.append(unknown_strategy)

    boolean_observed_count = copy.deepcopy(valid)
    boolean_observed_count["tuning"]["candidate_sets"][0][  # type: ignore[index]
        "observed_breakpoint_count"
    ] = True
    mutations.append(boolean_observed_count)

    negative_observed_count = copy.deepcopy(valid)
    negative_observed_count["tuning"]["candidate_sets"][0][  # type: ignore[index]
        "observed_breakpoint_count"
    ] = -1
    mutations.append(negative_observed_count)

    for payload in mutations:
        with pytest.raises((TypeError, ValueError)):
            LambdaPolicyArtifact.from_dict(payload)


def test_candidate_fraction_uses_the_same_strict_canonical_encoding(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    payload = copy.deepcopy(policy_artifact.to_dict())
    payload["tuning"]["candidate_sets"][0]["values"][0] = {  # type: ignore[index]
        "numerator": "2",
        "denominator": "4",
    }

    with pytest.raises(ValueError, match="reduced canonical fraction"):
        LambdaPolicyArtifact.from_dict(payload)


def test_loader_rejects_pickle_and_non_utf8_binary(tmp_path: Path) -> None:
    ascii_pickle = tmp_path / "ascii.pkl"
    ascii_pickle.write_bytes(b"N.")
    binary_pickle = tmp_path / "binary.pkl"
    binary_pickle.write_bytes(b"\x80\x04N.")

    with pytest.raises(ValueError, match="not valid strict JSON"):
        LambdaPolicyArtifact.load(ascii_pickle)
    with pytest.raises(ValueError, match="cannot read lambda policy artifact"):
        LambdaPolicyArtifact.load(binary_pickle)


def test_policy_state_is_defensively_copied_and_immutable(
    policy_artifact: LambdaPolicyArtifact,
) -> None:
    mutable_lambdas = dict(policy_artifact.lambda_by_tier)
    mutable_specs = list(policy_artifact.tier_specs)
    mutable_candidates = list(policy_artifact.candidate_sets)
    copied = replace(
        policy_artifact,
        lambda_by_tier=mutable_lambdas,
        tier_specs=mutable_specs,  # type: ignore[arg-type]
        candidate_sets=mutable_candidates,  # type: ignore[arg-type]
    )

    mutable_lambdas[BudgetTier.FAST] = Fraction(999)
    mutable_specs.clear()
    mutable_candidates.clear()

    assert copied.lambda_by_tier[BudgetTier.FAST] == Fraction(0)
    assert copied.tier_specs == policy_artifact.tier_specs
    assert copied.candidate_sets == policy_artifact.candidate_sets
    with pytest.raises(TypeError):
        copied.lambda_by_tier[BudgetTier.FAST] = Fraction(1)  # type: ignore[index]
    with pytest.raises(AttributeError):
        copied.training_domains = ("changed",)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        copied.candidate_sets[0].values = (Fraction(5),)  # type: ignore[misc]
