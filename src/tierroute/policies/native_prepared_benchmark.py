# SPDX-License-Identifier: Apache-2.0
"""Authenticated native prepared scores to per-query benchmark evidence.

This module joins the file-backed ``TRPSTO01``/``TRPRES01`` training session to
the existing isotonic, nested-LODO lambda, replay, and six-baseline machinery.
It deliberately uses two phases:

1. consume authenticated mmap-backed targets and native raw scores into a bounded,
   owned calibrated snapshot without invoking a caller callback; then reauthenticate
   both mappings, close the owned store, and stop consulting the caller-owned result;
2. run lambda tuning, learned replay, and the six baselines using only the owned
   snapshot.

No raw-score bundle is materialized.  The only temporarily retained numerical score
payload is the calibrated ``D²`` destination graph needed by the policy stage.  The native
semantic identity excludes the request nonce and executable bytes, while a separate
execution receipt records those exact run credentials.  Neither identity is a
signature or a provenance approval.

The returned result is score-payload-free: it retains fitted isotonic breakpoints as
auditable policy parameters, but no mmap, native view, raw-score matrix, target matrix,
or calibrated-score matrix.  A caller must pin the result-file SHA-256 outside the
result object.  Concurrent same-process code that deliberately mutates and restores
result-object credentials between checks, or reaches private file descriptors to flip
and restore bytes during one locked read, is outside this in-process evidence model;
persistent mutation and credential replacement are checked on both sides of consumption.

The public high-level entry point consumes a caller-owned, already authenticated native
result and owns only its private prepared-store snapshot.  This remains an experimental,
training/evaluation-only path: it does not alter native execution, the default trainer,
the routing CLI, the base wheel, or per-query accounting semantics.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType

from tierroute.adapters import PerQueryBudgetLedger
from tierroute.core import ModelSpec, scale_cost, sum_costs
from tierroute.core.integer_text import integer_to_decimal
from tierroute.eval import (
    EvaluationExample,
    TierSpec,
    evaluation_data_sha256,
    evaluation_replay_sha256,
    leave_one_domain_out,
)
from tierroute.eval.metrics import QuoteErrorReport, oracle_gap_recovery, summarize_quote_error
from tierroute.eval.planning import observable_domain_tag
from tierroute.eval.provenance import _snapshot_evaluation_scope
from tierroute.features.embeddings import EmbeddingIdentity
from tierroute.policies.baseline_evaluation import (
    BASELINE_NAMES,
    LodoSixBaselineEvaluation,
    evaluate_per_query_lodo_baselines,
)
from tierroute.policies.lambda_tuning import (
    NestedLodoLambdaResult,
    estimate_lambda_search,
    nested_lodo_lambda_evaluation,
)
from tierroute.policies.prepared_reference import (
    MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER,
    PreparedReferencePipelineEstimate,
    estimate_prepared_reference_pipeline,
)
from tierroute.predictors.base import QualityPredictor
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.native_prepared import (
    PREPARED_MOMENT_SOLVER_ID,
    PREPARED_RAW_SCORER_ID,
    PREPARED_SESSION_ENGINE_ID,
    PREPARED_SESSION_RESULT_ID,
    NativePreparedCoefficientRecord,
    NativePreparedScoreRecord,
    NativePreparedSessionResult,
)
from tierroute.predictors.prepared_files import (
    COEFFICIENT_RECORD_HEADER_BYTES,
    UNIVERSAL_SURFACE_WIDTH,
    AuthenticatedPreparedStore,
    PreparedStoreFileMetadata,
    PreparedStoreFileReceipt,
    authenticate_prepared_store_file,
    validate_prepared_store_context,
)
from tierroute.predictors.prepared_graph import (
    PreparedNestedLodoPlan,
    build_prepared_nested_lodo_plan,
)
from tierroute.predictors.prepared_store import prepared_fit_source_sha256

NATIVE_PREPARED_BENCHMARK_ALGORITHM_ID = "tierroute.native-prepared-benchmark-v1"
NATIVE_PREPARED_POLICY_SNAPSHOT_ID = "tierroute.native-prepared-policy-snapshot-v1"
NATIVE_PREPARED_EXECUTION_RECEIPT_ID = "tierroute.native-prepared-execution-receipt-v1"
NATIVE_PREPARED_SEMANTIC_SCORE_ID = "tierroute.native-prepared-semantic-score-v1"
NATIVE_PREPARED_TARGET_SHARD_ID = "tierroute.native-prepared-target-shard-v1"
NATIVE_PREPARED_CALIBRATION_ID = "tierroute.native-prepared-calibration-v1"
NATIVE_PREPARED_CALIBRATED_BLOCK_ID = "tierroute.native-prepared-calibrated-block-v1"
NATIVE_PREPARED_CALIBRATED_EVIDENCE_ID = "tierroute.native-prepared-calibrated-score-evidence-v1"
NATIVE_PREPARED_LEARNED_EVIDENCE_ID = "tierroute.native-prepared-learned-evidence-v1"

MAX_NATIVE_PREPARED_COMBINED_REPORT_ROWS = 2_000_000
MAX_NATIVE_PREPARED_BASELINE_REPORT_ROWS = 1_000_000
MAX_NATIVE_PREPARED_BASELINE_TAG_COMPARISONS = 50_000_000
MAX_NATIVE_PREPARED_BASELINE_WORK_UNITS = 100_000_000
MAX_NATIVE_PREPARED_BRIDGE_WORK_UNITS = 200_000_000
MAX_NATIVE_PREPARED_OWNED_NUMERIC_BYTES = 512 * 1024 * 1024

_BASELINE_COUNT = len(BASELINE_NAMES)
# Conservative modeled validation/evidence bookkeeping for each retained query in
# each baseline report.  It is a reviewed accounting unit, not measured CPU work.
_BASELINE_CONSTRUCTOR_EVIDENCE_WORK_PER_QUERY = 32
_F64 = struct.Struct("<d")
_U16 = struct.Struct("<H")
_F64_BYTES = 8


@dataclass(frozen=True, slots=True)
class NativePreparedBenchmarkConfig:
    """Reviewed, per-query-only controls for the native policy benchmark."""

    max_candidates_per_tier: int = MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER
    random_seed: int = 2026
    character_threshold: int = 120
    accounting_scope: str = field(default="per-query", init=False)
    allow_large_exhaustive: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        cap = _exact_nonnegative_int(
            self.max_candidates_per_tier,
            "max_candidates_per_tier",
        )
        if not 2 <= cap <= MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER:
            raise ValueError(
                "max_candidates_per_tier must be between 2 and "
                f"{MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER}"
            )
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an exact integer")
        if not -(2**63) <= self.random_seed <= 2**63 - 1:
            raise ValueError("random_seed must fit a signed 64-bit integer")
        if type(self.character_threshold) is not int:
            raise TypeError("character_threshold must be an exact integer")
        if not 1 <= self.character_threshold <= 2**63 - 1:
            raise ValueError("character_threshold must be a positive signed 64-bit integer")


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _sha256_hex(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _canonical_f64(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite binary64") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier} binary64")
    return 0.0 if result == 0.0 else result


class _EvidenceWriter:
    """Length-frame evidence fields so concatenation is unambiguous."""

    __slots__ = ("_digest",)

    def __init__(self, namespace: str) -> None:
        self._digest = hashlib.sha256()
        self.text("namespace", namespace)

    def token(self, label: str, payload: bytes) -> None:
        if type(payload) is not bytes:
            raise TypeError("evidence payloads must be exact bytes")
        label_bytes = label.encode("ascii")
        self._digest.update(struct.pack("<I", len(label_bytes)))
        self._digest.update(label_bytes)
        self._digest.update(struct.pack("<Q", len(payload)))
        self._digest.update(payload)

    def text(self, label: str, value: str) -> None:
        self.token(label, value.encode("utf-8"))

    def integer(self, label: str, value: int) -> None:
        self.token(label, integer_to_decimal(value).encode("ascii"))

    def boolean(self, label: str, value: bool) -> None:
        self.token(label, b"\x01" if value else b"\x00")

    def f64(self, label: str, value: float) -> None:
        self.token(label, struct.pack("<d", value))

    def fraction(self, label: str, value: Fraction) -> None:
        numerator = integer_to_decimal(value.numerator)
        denominator = integer_to_decimal(value.denominator)
        self.text(label, f"{numerator}/{denominator}")

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _canonical_plan(plan: PreparedNestedLodoPlan) -> PreparedNestedLodoPlan:
    if type(plan) is not PreparedNestedLodoPlan:
        raise TypeError("plan must be an exact PreparedNestedLodoPlan")
    canonical = build_prepared_nested_lodo_plan(
        plan.domains,
        plan.domain_example_counts,
        feature_count=plan.feature_count,
        target_count=plan.target_count,
    )
    if canonical != plan:
        raise ValueError("prepared plan is not its canonical reconstruction")
    return canonical


def _stable_model_ids(examples: tuple[EvaluationExample, ...]) -> tuple[str, ...]:
    catalogue = tuple(sorted(examples[0].candidate_models, key=lambda model: model.model_id))
    model_ids = tuple(model.model_id for model in catalogue)
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("native benchmark model IDs must be unique")
    for example in examples[1:]:
        candidate_catalogue = tuple(
            sorted(example.candidate_models, key=lambda model: model.model_id)
        )
        if candidate_catalogue != catalogue:
            raise ValueError("native benchmark requires stable model IDs and quoted costs")
    return model_ids


@dataclass(frozen=True, slots=True)
class NativePreparedBaselineFoldEstimate:
    """Observable metadata width admitted for one outer LODO training side."""

    held_out_domain: str
    training_rows: int
    tagged_rows: int
    unique_observable_tags: int

    def __post_init__(self) -> None:
        if type(self.held_out_domain) is not str or not self.held_out_domain.strip():
            raise ValueError("held_out_domain must be non-empty exact text")
        training_rows = _exact_nonnegative_int(self.training_rows, "training_rows")
        tagged_rows = _exact_nonnegative_int(self.tagged_rows, "tagged_rows")
        tags = _exact_nonnegative_int(
            self.unique_observable_tags,
            "unique_observable_tags",
        )
        if training_rows == 0:
            raise ValueError("baseline outer-fold training rows must be positive")
        if tagged_rows > training_rows or tags > tagged_rows:
            raise ValueError("baseline observable-tag counts are inconsistent")


def _baseline_fold_estimates(
    examples: tuple[EvaluationExample, ...],
) -> tuple[NativePreparedBaselineFoldEstimate, ...]:
    estimates: list[NativePreparedBaselineFoldEstimate] = []
    for fold in leave_one_domain_out(examples):
        tags = tuple(
            tag
            for example in fold.training
            if (tag := observable_domain_tag(example.router_metadata)) is not None
        )
        estimates.append(
            NativePreparedBaselineFoldEstimate(
                held_out_domain=fold.held_out_domain,
                training_rows=len(fold.training),
                tagged_rows=len(tags),
                unique_observable_tags=len(set(tags)),
            )
        )
    return tuple(estimates)


@dataclass(frozen=True, slots=True)
class NativePreparedBenchmarkEstimate:
    """Aggregate admission for native consumption, policy replay, and six baselines.

    This is deterministic modeled work/storage, not measured RSS, latency, or a
    universal Python-object peak.  File reauthentication bytes are reported as I/O
    evidence and deliberately are not counted as numeric work units.
    """

    plan: PreparedNestedLodoPlan
    store_metadata: PreparedStoreFileMetadata
    policy_estimate: PreparedReferencePipelineEstimate
    baseline_folds: tuple[NativePreparedBaselineFoldEstimate, ...]
    baseline_count: int
    baseline_query_replays: int
    baseline_metadata_prep_work_units: int
    baseline_tag_comparison_work_units: int
    baseline_domain_table_work_units: int
    baseline_oracle_work_units: int
    baseline_replay_work_units: int
    baseline_constructor_evidence_work_units: int
    baseline_work_units: int
    baseline_retained_rows: int
    semantic_score_cells_hashed: int
    semantic_coefficient_cells_hashed: int
    target_cells_validated: int
    materialized_feature_bytes: int
    materialized_raw_score_bytes: int
    mapped_file_bytes: int
    row_index_bytes: int
    mapping_reauthentication_bytes: int
    owned_calibrated_score_bytes: int
    combined_report_rows: int
    bridge_work_units: int
    owned_numeric_bytes: int

    def __post_init__(self) -> None:
        plan = _canonical_plan(self.plan)
        if type(self.store_metadata) is not PreparedStoreFileMetadata:
            raise TypeError("store_metadata must be exact PreparedStoreFileMetadata")
        if type(self.policy_estimate) is not PreparedReferencePipelineEstimate:
            raise TypeError("policy_estimate must be exact PreparedReferencePipelineEstimate")
        if self.policy_estimate.plan != plan:
            raise ValueError("policy estimate does not match the native benchmark plan")
        if (
            type(self.baseline_folds) is not tuple
            or len(self.baseline_folds) != len(plan.domains)
            or any(
                type(item) is not NativePreparedBaselineFoldEstimate for item in self.baseline_folds
            )
            or tuple(item.held_out_domain for item in self.baseline_folds) != plan.domains
        ):
            raise ValueError("baseline preflight folds must match canonical LODO domains")
        for domain_index, fold in enumerate(self.baseline_folds):
            expected_rows = plan.work.example_count - plan.domain_example_counts[domain_index]
            if fold.training_rows != expected_rows:
                raise ValueError("baseline preflight training rows do not match the plan")
        metadata = self.store_metadata
        if (
            metadata.domain_count != len(plan.domains)
            or metadata.row_count != plan.work.example_count
            or metadata.feature_count != plan.feature_count
            or metadata.target_count != plan.target_count
            or metadata.domain_row_counts != plan.domain_example_counts
        ):
            raise ValueError("native store metadata does not match the benchmark plan")
        values = _native_benchmark_estimate_values(
            plan,
            metadata,
            self.policy_estimate,
            self.baseline_folds,
        )
        for name, expected in values.items():
            actual = getattr(self, name)
            _exact_nonnegative_int(actual, name)
            if actual != expected:
                raise ValueError(f"{name} does not match the native benchmark formula")
        if self.baseline_count != _BASELINE_COUNT:
            raise ValueError("native benchmark must retain exactly six baselines")
        if self.baseline_retained_rows > MAX_NATIVE_PREPARED_BASELINE_REPORT_ROWS:
            raise ValueError("native benchmark baseline rows exceed the reviewed limit")
        if self.baseline_tag_comparison_work_units > MAX_NATIVE_PREPARED_BASELINE_TAG_COMPARISONS:
            raise ValueError("native benchmark observable-tag comparisons exceed the limit")
        if self.baseline_work_units > MAX_NATIVE_PREPARED_BASELINE_WORK_UNITS:
            raise ValueError("native benchmark baseline work exceeds the reviewed limit")
        if self.combined_report_rows > MAX_NATIVE_PREPARED_COMBINED_REPORT_ROWS:
            raise ValueError("native benchmark retained report rows exceed the reviewed limit")
        if self.bridge_work_units > MAX_NATIVE_PREPARED_BRIDGE_WORK_UNITS:
            raise ValueError("native benchmark bridge work exceeds the reviewed limit")
        if self.owned_numeric_bytes > MAX_NATIVE_PREPARED_OWNED_NUMERIC_BYTES:
            raise ValueError("native benchmark owned numeric bytes exceed the reviewed limit")


def _native_benchmark_estimate_values(
    plan: PreparedNestedLodoPlan,
    metadata: PreparedStoreFileMetadata,
    policy: PreparedReferencePipelineEstimate,
    baseline_folds: tuple[NativePreparedBaselineFoldEstimate, ...],
) -> dict[str, int]:
    n = plan.work.example_count
    m = plan.target_count
    tier_count = policy.tier_count
    baseline = _baseline_estimate_values(n, m, tier_count, baseline_folds)
    baseline_query_replays = baseline["baseline_query_replays"]
    baseline_work = baseline["baseline_work_units"]
    coefficient_payload_bytes = metadata.estimate.coefficient_bytes - (
        metadata.estimate.training_subset_count * COEFFICIENT_RECORD_HEADER_BYTES
    )
    if coefficient_payload_bytes < 0 or coefficient_payload_bytes % _F64_BYTES:
        raise ValueError("native coefficient payload estimate is not binary64-aligned")
    semantic_coefficient_cells = coefficient_payload_bytes // _F64_BYTES
    semantic_score_cells = plan.work.scalar_score_count
    target_cells = n * m
    mapped_file_bytes = metadata.file_bytes + metadata.estimate.result_bytes
    row_index_bytes = _F64_BYTES * n
    mapping_reauthentication_bytes = 2 * metadata.estimate.result_bytes + metadata.file_bytes
    owned_score_bytes = policy.calibrated_prediction_cells * _F64_BYTES
    combined_rows = policy.retained_report_rows + baseline_query_replays
    bridge_work = (
        policy.postprocess_work_units
        + baseline_work
        + semantic_score_cells
        + semantic_coefficient_cells
        + target_cells
    )
    # ``postprocess_numeric_bytes`` includes one calibrated destination table.  The
    # two-phase bridge deliberately retains the immutable byte snapshot while the
    # existing evaluator caches its unpacked predictor rows, so one additional
    # binary64-modeled copy is live at the same time.  This still does not claim to
    # model Python tuple/object allocator overhead.
    owned_numeric_bytes = policy.postprocess_numeric_bytes + owned_score_bytes + row_index_bytes
    return {
        **baseline,
        "semantic_score_cells_hashed": semantic_score_cells,
        "semantic_coefficient_cells_hashed": semantic_coefficient_cells,
        "target_cells_validated": target_cells,
        "materialized_feature_bytes": 0,
        "materialized_raw_score_bytes": 0,
        "mapped_file_bytes": mapped_file_bytes,
        "row_index_bytes": row_index_bytes,
        "mapping_reauthentication_bytes": mapping_reauthentication_bytes,
        "owned_calibrated_score_bytes": owned_score_bytes,
        "combined_report_rows": combined_rows,
        "bridge_work_units": bridge_work,
        "owned_numeric_bytes": owned_numeric_bytes,
    }


def _baseline_estimate_values(
    example_count: int,
    model_count: int,
    tier_count: int,
    baseline_folds: tuple[NativePreparedBaselineFoldEstimate, ...],
) -> dict[str, int]:
    baseline_query_replays = _BASELINE_COUNT * tier_count * example_count
    training_rows = sum(fold.training_rows for fold in baseline_folds)
    metadata_prep_work = 2 * training_rows
    tag_comparison_work = tier_count * sum(
        fold.unique_observable_tags * fold.tagged_rows for fold in baseline_folds
    )
    domain_table_work = tier_count * sum(
        2 * model_count * fold.tagged_rows + model_count * fold.unique_observable_tags
        for fold in baseline_folds
    )
    oracle_work = 3 * tier_count * example_count * model_count
    replay_work = _BASELINE_COUNT * tier_count * example_count * (2 * model_count + 8)
    constructor_evidence_work = (
        _BASELINE_COUNT * _BASELINE_CONSTRUCTOR_EVIDENCE_WORK_PER_QUERY * tier_count * example_count
    )
    baseline_work = (
        metadata_prep_work
        + tag_comparison_work
        + domain_table_work
        + oracle_work
        + replay_work
        + constructor_evidence_work
    )
    return {
        "baseline_count": _BASELINE_COUNT,
        "baseline_query_replays": baseline_query_replays,
        "baseline_metadata_prep_work_units": metadata_prep_work,
        "baseline_tag_comparison_work_units": tag_comparison_work,
        "baseline_domain_table_work_units": domain_table_work,
        "baseline_oracle_work_units": oracle_work,
        "baseline_replay_work_units": replay_work,
        "baseline_constructor_evidence_work_units": constructor_evidence_work,
        "baseline_work_units": baseline_work,
        "baseline_retained_rows": baseline_query_replays,
    }


def _preflight_snapshot(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    plan: PreparedNestedLodoPlan,
    metadata: PreparedStoreFileMetadata,
    receipt: PreparedStoreFileReceipt,
    embedding_identity: EmbeddingIdentity | None,
    *,
    max_candidates_per_tier: int,
) -> tuple[
    tuple[EvaluationExample, ...],
    tuple[TierSpec, ...],
    tuple[str, ...],
    NativePreparedBenchmarkEstimate,
]:
    if type(examples) is not tuple or not examples:
        raise TypeError("examples must be a non-empty exact tuple")
    if any(type(example) is not EvaluationExample for example in examples):
        raise TypeError("examples must contain exact EvaluationExample values")
    if type(tier_specs) is not tuple or not tier_specs:
        raise TypeError("tier_specs must be a non-empty exact tuple")
    if any(type(spec) is not TierSpec for spec in tier_specs):
        raise TypeError("tier_specs must contain exact TierSpec values")
    plan = _canonical_plan(plan)
    if type(metadata) is not PreparedStoreFileMetadata:
        raise TypeError("metadata must be exact PreparedStoreFileMetadata")
    if type(receipt) is not PreparedStoreFileReceipt:
        raise TypeError("receipt must be an exact PreparedStoreFileReceipt")
    cap = _exact_nonnegative_int(max_candidates_per_tier, "max_candidates_per_tier")
    if not 2 <= cap <= MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER:
        raise ValueError(
            "max_candidates_per_tier must be between 2 and "
            f"{MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER}"
        )
    if len(examples) != plan.work.example_count:
        raise ValueError("evaluation row count does not match the prepared plan")
    if tuple(sorted({example.domain for example in examples})) != plan.domains:
        raise ValueError("evaluation domains do not match the prepared plan")
    model_ids = _stable_model_ids(examples)
    if len(model_ids) != plan.target_count:
        raise ValueError("evaluation model catalogue does not match the prepared plan")
    # Bind the cheap, fixed identities before deep-copying arbitrary metadata values.
    validate_prepared_store_context(
        metadata,
        receipt,
        plan,
        model_ids,
        embedding_identity,
    )
    snapshot_examples, snapshot_specs = _snapshot_evaluation_scope(examples, tier_specs)
    if len(snapshot_examples) != plan.work.example_count:
        raise ValueError("evaluation snapshot row count does not match the prepared plan")
    if tuple(sorted({example.domain for example in snapshot_examples})) != plan.domains:
        raise ValueError("evaluation snapshot domains do not match the prepared plan")
    if _stable_model_ids(snapshot_examples) != model_ids:
        raise ValueError("evaluation model catalogue changed while taking its snapshot")
    source_digest = prepared_fit_source_sha256(snapshot_examples, plan)
    if not hmac.compare_digest(source_digest, receipt.source_fit_sha256):
        raise ValueError("evaluation snapshot does not match the trusted source-fit SHA-256")
    folds = leave_one_domain_out(snapshot_examples)
    baseline_folds = _baseline_fold_estimates(snapshot_examples)
    baseline_values = _baseline_estimate_values(
        len(snapshot_examples),
        len(model_ids),
        len(snapshot_specs),
        baseline_folds,
    )
    if (
        baseline_values["baseline_tag_comparison_work_units"]
        > MAX_NATIVE_PREPARED_BASELINE_TAG_COMPARISONS
    ):
        raise ValueError("native benchmark observable-tag comparisons exceed the limit")
    if baseline_values["baseline_work_units"] > MAX_NATIVE_PREPARED_BASELINE_WORK_UNITS:
        raise ValueError("native benchmark baseline work exceeds the reviewed limit")
    if baseline_values["baseline_retained_rows"] > MAX_NATIVE_PREPARED_BASELINE_REPORT_ROWS:
        raise ValueError("native benchmark baseline rows exceed the reviewed limit")
    lambda_estimates = tuple(
        estimate_lambda_search(
            fold.training,
            snapshot_specs,
            max_candidates_per_tier=cap,
            allow_large_exhaustive=False,
        )
        for fold in folds
    )
    policy_estimate = estimate_prepared_reference_pipeline(
        plan,
        tier_count=len(snapshot_specs),
        max_candidates_per_tier=cap,
        execution_estimate=None,
        lambda_search_estimates=lambda_estimates,
    )
    values = _native_benchmark_estimate_values(
        plan,
        metadata,
        policy_estimate,
        baseline_folds,
    )
    estimate = NativePreparedBenchmarkEstimate(
        plan=plan,
        store_metadata=metadata,
        policy_estimate=policy_estimate,
        baseline_folds=baseline_folds,
        **values,
    )
    return snapshot_examples, snapshot_specs, model_ids, estimate


def preflight_native_prepared_benchmark(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    plan: PreparedNestedLodoPlan,
    metadata: PreparedStoreFileMetadata,
    receipt: PreparedStoreFileReceipt,
    embedding_identity: EmbeddingIdentity | None,
    *,
    max_candidates_per_tier: int = MAX_PREPARED_PIPELINE_CANDIDATES_PER_TIER,
) -> NativePreparedBenchmarkEstimate:
    """Admit the full bridge before native execution or policy score reads."""

    return _preflight_snapshot(
        examples,
        tier_specs,
        plan,
        metadata,
        receipt,
        embedding_identity,
        max_candidates_per_tier=max_candidates_per_tier,
    )[3]


@dataclass(frozen=True, slots=True)
class NativePreparedExecutionReceipt:
    """Nonce-bearing exact run receipt, separate from semantic score identity."""

    request_nonce_hex: str
    store_file_sha256: str
    binary_sha256: str
    result_sha256: str
    result_sha256_caller_pinned: bool
    ridge: float
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_EXECUTION_RECEIPT_ID, init=False)
    engine_id: str = field(default=PREPARED_SESSION_ENGINE_ID, init=False)
    result_format_id: str = field(default=PREPARED_SESSION_RESULT_ID, init=False)

    def __post_init__(self) -> None:
        if (
            type(self.request_nonce_hex) is not str
            or len(self.request_nonce_hex) != 64
            or any(character not in "0123456789abcdef" for character in self.request_nonce_hex)
            or not any(character != "0" for character in self.request_nonce_hex)
        ):
            raise ValueError("request_nonce_hex must be a nonzero 32-byte lowercase hex value")
        for name in ("store_file_sha256", "binary_sha256", "result_sha256"):
            _sha256_hex(getattr(self, name), name)
        if type(self.result_sha256_caller_pinned) is not bool:
            raise TypeError("result_sha256_caller_pinned must be a boolean")
        ridge = _canonical_f64(self.ridge, "execution receipt ridge", positive=True)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "sha256", _execution_receipt_sha256(self))


def _execution_receipt_sha256(receipt: NativePreparedExecutionReceipt) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_EXECUTION_RECEIPT_ID)
    writer.text("engine_id", receipt.engine_id)
    writer.text("result_format_id", receipt.result_format_id)
    writer.text("request_nonce_hex", receipt.request_nonce_hex)
    writer.text("store_file_sha256", receipt.store_file_sha256)
    writer.text("binary_sha256", receipt.binary_sha256)
    writer.text("result_sha256", receipt.result_sha256)
    writer.boolean("result_sha256_caller_pinned", receipt.result_sha256_caller_pinned)
    writer.f64("ridge", receipt.ridge)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class NativePreparedTargetShardEvidence:
    """One exact domain target identity consumed from the authenticated store."""

    domain_index: int
    domain: str
    example_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    target_sha256: str
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_TARGET_SHARD_ID, init=False)

    def __post_init__(self) -> None:
        index = _exact_nonnegative_int(self.domain_index, "target shard domain_index")
        if type(self.domain) is not str or not self.domain.strip():
            raise ValueError("target shard domain must be non-empty")
        if type(self.example_ids) is not tuple or not self.example_ids:
            raise ValueError("target shard example_ids must be a non-empty exact tuple")
        if any(type(value) is not str or not value for value in self.example_ids):
            raise ValueError("target shard example IDs must be non-empty exact strings")
        if len(self.example_ids) != len(set(self.example_ids)):
            raise ValueError("target shard example IDs must be unique")
        if (
            type(self.model_ids) is not tuple
            or not self.model_ids
            or self.model_ids != tuple(sorted(set(self.model_ids)))
        ):
            raise ValueError("target shard model IDs must be sorted and unique")
        _sha256_hex(self.target_sha256, "target_sha256")
        object.__setattr__(self, "domain_index", index)
        object.__setattr__(self, "sha256", _target_shard_evidence_sha256(self))


def _target_shard_evidence_sha256(evidence: NativePreparedTargetShardEvidence) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_TARGET_SHARD_ID)
    writer.integer("domain_index", evidence.domain_index)
    writer.text("domain", evidence.domain)
    for example_id in evidence.example_ids:
        writer.text("example_id", example_id)
    for model_id in evidence.model_ids:
        writer.text("model_id", model_id)
    writer.text("target_sha256", evidence.target_sha256)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class NativePreparedCalibrationEvidence:
    """One per-model inner-LODO isotonic calibration from native raw scores."""

    training_subset_index: int
    training_domain_indices: tuple[int, ...]
    model_ids: tuple[str, ...]
    calibration_example_count: int
    raw_score_block_indices: tuple[int, ...]
    raw_score_block_sha256s: tuple[str, ...]
    target_shard_sha256s: tuple[str, ...]
    calibrators: tuple[IsotonicCalibrator, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_CALIBRATION_ID, init=False)

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.training_subset_index, "calibration subset index")
        count = _exact_nonnegative_int(
            self.calibration_example_count,
            "calibration example count",
        )
        if count == 0:
            raise ValueError("calibration example count must be positive")
        if (
            type(self.training_domain_indices) is not tuple
            or not self.training_domain_indices
            or self.training_domain_indices != tuple(sorted(set(self.training_domain_indices)))
        ):
            raise ValueError("calibration domains must be a sorted unique exact tuple")
        if (
            type(self.model_ids) is not tuple
            or not self.model_ids
            or self.model_ids != tuple(sorted(set(self.model_ids)))
        ):
            raise ValueError("calibration model IDs must be sorted and unique")
        if (
            type(self.raw_score_block_indices) is not tuple
            or len(self.raw_score_block_indices) != len(self.training_domain_indices)
            or type(self.raw_score_block_sha256s) is not tuple
            or len(self.raw_score_block_sha256s) != len(self.raw_score_block_indices)
            or type(self.target_shard_sha256s) is not tuple
            or len(self.target_shard_sha256s) != len(self.training_domain_indices)
        ):
            raise ValueError("calibration lineage tuples have inconsistent lengths")
        for index in self.raw_score_block_indices:
            _exact_nonnegative_int(index, "calibration raw-score block index")
        for digest in (*self.raw_score_block_sha256s, *self.target_shard_sha256s):
            _sha256_hex(digest, "calibration parent SHA-256")
        if (
            type(self.calibrators) is not tuple
            or len(self.calibrators) != len(self.model_ids)
            or any(type(item) is not IsotonicCalibrator for item in self.calibrators)
        ):
            raise ValueError("calibration must contain one exact calibrator per model")
        if any(len(item.values) > count for item in self.calibrators):
            raise ValueError("calibrator point counts cannot exceed the admitted calibration rows")
        object.__setattr__(self, "sha256", _calibration_evidence_sha256(self))


def _calibration_evidence_sha256(evidence: NativePreparedCalibrationEvidence) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_CALIBRATION_ID)
    writer.integer("training_subset_index", evidence.training_subset_index)
    writer.integer("calibration_example_count", evidence.calibration_example_count)
    for index in evidence.training_domain_indices:
        writer.integer("training_domain_index", index)
    for model_id in evidence.model_ids:
        writer.text("model_id", model_id)
    for index, raw_sha in zip(
        evidence.raw_score_block_indices,
        evidence.raw_score_block_sha256s,
        strict=True,
    ):
        writer.integer("raw_score_block_index", index)
        writer.text("raw_score_block_sha256", raw_sha)
    for digest in evidence.target_shard_sha256s:
        writer.text("target_shard_sha256", digest)
    for calibrator in evidence.calibrators:
        writer.integer("calibrator_point_count", len(calibrator.values))
        for value in (*calibrator.upper_bounds, *calibrator.values):
            writer.f64("calibrator.f64", value)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class _NativePreparedCalibratedScoreBlock:
    """Owned row-major calibrated scores for one policy destination."""

    training_subset_index: int
    scored_domain_index: int
    raw_score_block_index: int
    raw_score_block_sha256: str
    calibration_sha256: str
    example_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    scores_payload: bytes = field(repr=False)
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_CALIBRATED_BLOCK_ID, init=False)

    def __post_init__(self) -> None:
        for name in (
            "training_subset_index",
            "scored_domain_index",
            "raw_score_block_index",
        ):
            _exact_nonnegative_int(getattr(self, name), name)
        _sha256_hex(self.raw_score_block_sha256, "raw_score_block_sha256")
        _sha256_hex(self.calibration_sha256, "calibration_sha256")
        if type(self.example_ids) is not tuple or not self.example_ids:
            raise ValueError("calibrated block example IDs must be a non-empty exact tuple")
        if len(self.example_ids) != len(set(self.example_ids)):
            raise ValueError("calibrated block example IDs must be unique")
        if (
            type(self.model_ids) is not tuple
            or not self.model_ids
            or self.model_ids != tuple(sorted(set(self.model_ids)))
        ):
            raise ValueError("calibrated block model IDs must be sorted and unique")
        if type(self.scores_payload) is not bytes:
            raise TypeError("calibrated score payload must be immutable bytes")
        expected_bytes = len(self.example_ids) * len(self.model_ids) * _F64_BYTES
        if len(self.scores_payload) != expected_bytes:
            raise ValueError("calibrated score payload has the wrong exact length")
        for (value,) in struct.iter_unpack("<d", self.scores_payload):
            if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0.0):
                raise ValueError("calibrated scores must be finite canonical binary64")
        object.__setattr__(self, "sha256", _calibrated_block_sha256(self))

    def score_row(self, row_index: int) -> tuple[float, ...]:
        index = _exact_nonnegative_int(row_index, "calibrated score row index")
        if index >= len(self.example_ids):
            raise IndexError("calibrated score row index is outside the block")
        return struct.unpack_from(
            f"<{len(self.model_ids)}d",
            self.scores_payload,
            index * len(self.model_ids) * _F64_BYTES,
        )


def _calibrated_block_sha256(block: _NativePreparedCalibratedScoreBlock) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_CALIBRATED_BLOCK_ID)
    writer.integer("training_subset_index", block.training_subset_index)
    writer.integer("scored_domain_index", block.scored_domain_index)
    writer.integer("raw_score_block_index", block.raw_score_block_index)
    writer.text("raw_score_block_sha256", block.raw_score_block_sha256)
    writer.text("calibration_sha256", block.calibration_sha256)
    for model_id in block.model_ids:
        writer.text("model_id", model_id)
    for example_id in block.example_ids:
        writer.text("example_id", example_id)
    writer.token("scores.row-major.f64le", block.scores_payload)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class NativePreparedSemanticScoreIdentity:
    """Nonce- and executable-independent identity of native numerical outputs."""

    graph_identity_sha256: str
    source_fit_sha256: str
    logical_store_sha256: str
    store_payload_sha256: str
    model_catalogue_sha256: str
    embedding_identity_sha256: str | None
    embedding_snapshot_sha256: str | None
    ridge: float
    coefficient_record_sha256s: tuple[str, ...]
    raw_score_block_sha256s: tuple[str, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_SEMANTIC_SCORE_ID, init=False)
    solver_id: str = field(default=PREPARED_MOMENT_SOLVER_ID, init=False)
    scorer_id: str = field(default=PREPARED_RAW_SCORER_ID, init=False)

    def __post_init__(self) -> None:
        for name in (
            "graph_identity_sha256",
            "source_fit_sha256",
            "logical_store_sha256",
            "store_payload_sha256",
            "model_catalogue_sha256",
        ):
            _sha256_hex(getattr(self, name), name)
        for name in ("embedding_identity_sha256", "embedding_snapshot_sha256"):
            value = getattr(self, name)
            if value is not None:
                _sha256_hex(value, name)
        if (self.embedding_identity_sha256 is None) != (self.embedding_snapshot_sha256 is None):
            raise ValueError("embedding identity and snapshot digests must appear together")
        ridge = _canonical_f64(self.ridge, "semantic ridge", positive=True)
        if type(self.coefficient_record_sha256s) is not tuple or not (
            self.coefficient_record_sha256s
        ):
            raise ValueError("semantic coefficient identities must be a non-empty exact tuple")
        if type(self.raw_score_block_sha256s) is not tuple or not self.raw_score_block_sha256s:
            raise ValueError("semantic raw-score identities must be a non-empty exact tuple")
        for digest in (*self.coefficient_record_sha256s, *self.raw_score_block_sha256s):
            _sha256_hex(digest, "semantic record SHA-256")
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "sha256", _semantic_score_identity_sha256(self))


def _semantic_score_identity_sha256(identity: NativePreparedSemanticScoreIdentity) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_SEMANTIC_SCORE_ID)
    writer.text("solver_id", identity.solver_id)
    writer.text("scorer_id", identity.scorer_id)
    for name in (
        "graph_identity_sha256",
        "source_fit_sha256",
        "logical_store_sha256",
        "store_payload_sha256",
        "model_catalogue_sha256",
    ):
        writer.text(name, getattr(identity, name))
    if identity.embedding_identity_sha256 is not None:
        writer.text("embedding_identity_sha256", identity.embedding_identity_sha256)
        writer.text("embedding_snapshot_sha256", identity.embedding_snapshot_sha256 or "")
    writer.f64("ridge", identity.ridge)
    for digest in identity.coefficient_record_sha256s:
        writer.text("coefficient_record_sha256", digest)
    for digest in identity.raw_score_block_sha256s:
        writer.text("raw_score_block_sha256", digest)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class _NativePreparedPolicySnapshot:
    """Owned calibrated graph returned only after both mmaps are reauthenticated."""

    estimate: NativePreparedBenchmarkEstimate
    execution_receipt: NativePreparedExecutionReceipt
    semantic_scores: NativePreparedSemanticScoreIdentity
    model_ids: tuple[str, ...]
    embedding_identity: EmbeddingIdentity | None
    evaluation_data_sha256: str
    evaluation_replay_sha256: str
    example_ids: tuple[str, ...]
    target_shards: tuple[NativePreparedTargetShardEvidence, ...]
    calibrations: tuple[NativePreparedCalibrationEvidence, ...]
    calibrated_score_blocks: tuple[_NativePreparedCalibratedScoreBlock, ...]
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_POLICY_SNAPSHOT_ID, init=False)

    def __post_init__(self) -> None:
        if type(self.estimate) is not NativePreparedBenchmarkEstimate:
            raise TypeError("estimate must be exact NativePreparedBenchmarkEstimate")
        if type(self.execution_receipt) is not NativePreparedExecutionReceipt:
            raise TypeError("execution_receipt must be exact NativePreparedExecutionReceipt")
        if type(self.semantic_scores) is not NativePreparedSemanticScoreIdentity:
            raise TypeError("semantic_scores must be exact NativePreparedSemanticScoreIdentity")
        plan = self.estimate.plan
        metadata = self.estimate.store_metadata
        if (
            self.semantic_scores.graph_identity_sha256 != metadata.graph_identity_sha256
            or self.semantic_scores.source_fit_sha256 != metadata.source_fit_sha256
            or self.semantic_scores.logical_store_sha256 != metadata.logical_store_sha256
            or self.semantic_scores.store_payload_sha256 != metadata.store_payload_sha256
            or self.semantic_scores.model_catalogue_sha256 != metadata.model_catalogue_sha256
            or self.semantic_scores.embedding_identity_sha256 != metadata.embedding_identity_sha256
            or self.semantic_scores.embedding_snapshot_sha256 != metadata.embedding_snapshot_sha256
            or self.semantic_scores.ridge != self.execution_receipt.ridge
        ):
            raise ValueError("native execution and semantic identities do not share one store")
        if (
            type(self.model_ids) is not tuple
            or self.model_ids != tuple(sorted(set(self.model_ids)))
            or len(self.model_ids) != plan.target_count
        ):
            raise ValueError("policy snapshot model IDs do not match the prepared plan")
        embedding_width = plan.feature_count - UNIVERSAL_SURFACE_WIDTH
        if embedding_width == 0:
            if self.embedding_identity is not None:
                raise ValueError("surface-only snapshot cannot retain an embedding identity")
        elif type(self.embedding_identity) is not EmbeddingIdentity:
            raise TypeError("embedded snapshot requires an exact EmbeddingIdentity")
        _sha256_hex(self.evaluation_data_sha256, "evaluation_data_sha256")
        _sha256_hex(self.evaluation_replay_sha256, "evaluation_replay_sha256")
        if (
            type(self.example_ids) is not tuple
            or len(self.example_ids) != plan.work.example_count
            or len(self.example_ids) != len(set(self.example_ids))
        ):
            raise ValueError("policy snapshot example IDs must cover the plan exactly once")
        if (
            type(self.target_shards) is not tuple
            or len(self.target_shards) != len(plan.domains)
            or any(
                type(item) is not NativePreparedTargetShardEvidence for item in self.target_shards
            )
            or tuple(item.domain_index for item in self.target_shards)
            != tuple(range(len(plan.domains)))
        ):
            raise ValueError("policy target shards must cover canonical domains exactly once")
        target_ids = tuple(
            example_id for shard in self.target_shards for example_id in shard.example_ids
        )
        if set(target_ids) != set(self.example_ids) or len(target_ids) != len(self.example_ids):
            raise ValueError("policy target shards must cover every example exactly once")
        expected_subsets = tuple(
            index
            for index, subset in enumerate(plan.training_subsets)
            if len(subset.domain_indices) in (len(plan.domains) - 2, len(plan.domains) - 1)
        )
        if (
            type(self.calibrations) is not tuple
            or any(
                type(item) is not NativePreparedCalibrationEvidence for item in self.calibrations
            )
            or tuple(item.training_subset_index for item in self.calibrations) != expected_subsets
        ):
            raise ValueError("policy calibrations do not cover the canonical subset graph")
        expected_blocks = tuple(
            (subset_index, domain_index)
            for subset_index in expected_subsets
            for domain_index in range(len(plan.domains))
            if domain_index not in plan.training_subsets[subset_index].domain_indices
        )
        if (
            type(self.calibrated_score_blocks) is not tuple
            or any(
                type(item) is not _NativePreparedCalibratedScoreBlock
                for item in self.calibrated_score_blocks
            )
            or tuple(
                (item.training_subset_index, item.scored_domain_index)
                for item in self.calibrated_score_blocks
            )
            != expected_blocks
        ):
            raise ValueError("calibrated blocks do not cover the canonical policy graph")
        if sum(len(block.scores_payload) for block in self.calibrated_score_blocks) != (
            self.estimate.owned_calibrated_score_bytes
        ):
            raise ValueError("owned calibrated score bytes do not match the preflight")
        object.__setattr__(self, "sha256", _policy_snapshot_sha256(self))


def _policy_snapshot_sha256(snapshot: _NativePreparedPolicySnapshot) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_POLICY_SNAPSHOT_ID)
    writer.text("execution_receipt_sha256", snapshot.execution_receipt.sha256)
    writer.text("semantic_score_sha256", snapshot.semantic_scores.sha256)
    writer.text("evaluation_data_sha256", snapshot.evaluation_data_sha256)
    writer.text("evaluation_replay_sha256", snapshot.evaluation_replay_sha256)
    for model_id in snapshot.model_ids:
        writer.text("model_id", model_id)
    for example_id in snapshot.example_ids:
        writer.text("example_id", example_id)
    for item in snapshot.target_shards:
        writer.text("target_shard_sha256", item.sha256)
    for item in snapshot.calibrations:
        writer.text("calibration_sha256", item.sha256)
    for item in snapshot.calibrated_score_blocks:
        writer.text("calibrated_score_block_sha256", item.sha256)
    return writer.hexdigest()


class _AuthenticatedStorePolicyView:
    """Validated row keys and bounded target access over one locked store snapshot."""

    __slots__ = (
        "_store",
        "_target_layout",
        "example_ids",
        "example_ids_by_domain",
        "example_row_indices_by_domain",
        "model_ids",
        "plan",
        "target_shards",
    )

    def __init__(
        self,
        store: AuthenticatedPreparedStore,
        examples: tuple[EvaluationExample, ...],
        plan: PreparedNestedLodoPlan,
        model_ids: tuple[str, ...],
    ) -> None:
        if type(store) is not AuthenticatedPreparedStore or store.closed:
            raise TypeError("store must be an open exact AuthenticatedPreparedStore")
        self._store = store
        self.plan = plan
        self.model_ids = model_ids
        domain_indices = store.domain_indices()

        ordered = tuple(sorted(examples, key=lambda example: example.example_id))
        expected_ids = tuple(example.example_id for example in ordered)
        domain_to_index = {domain: index for index, domain in enumerate(plan.domains)}
        expected_domains = bytes(domain_to_index[example.domain] for example in ordered)
        if domain_indices != expected_domains:
            raise ValueError(
                "authenticated prepared-store row keys or domains do not match evaluation data"
            )
        for example, (row_id, prompt_sha256) in zip(
            ordered,
            store.iter_row_keys(),
            strict=True,
        ):
            expected_prompt_sha256 = hashlib.sha256(example.prompt.encode("utf-8")).hexdigest()
            if row_id != example.example_id or prompt_sha256 != expected_prompt_sha256:
                raise ValueError(
                    "authenticated prepared-store row keys do not match evaluation data"
                )
        self.example_ids = expected_ids
        self.example_ids_by_domain = tuple(
            tuple(
                example_id
                for example_id, observed_domain in zip(
                    self.example_ids,
                    domain_indices,
                    strict=True,
                )
                if observed_domain == domain_index
            )
            for domain_index in range(len(plan.domains))
        )
        self.example_row_indices_by_domain = tuple(
            tuple(
                row_index
                for row_index, observed_domain in enumerate(domain_indices)
                if observed_domain == domain_index
            )
            for domain_index in range(len(plan.domains))
        )
        self._target_layout = struct.Struct(f"<{len(model_ids)}d")

        target_writers = [
            _EvidenceWriter(f"{NATIVE_PREPARED_TARGET_SHARD_ID}.payload") for _ in plan.domains
        ]
        for domain_index, writer in enumerate(target_writers):
            writer.integer("domain_index", domain_index)
            writer.text("domain", plan.domains[domain_index])
            for model_id in model_ids:
                writer.text("model_id", model_id)
        for row_index, (example_id, domain_index) in enumerate(
            zip(self.example_ids, domain_indices, strict=True)
        ):
            row = self.target_row(row_index)
            example = ordered[row_index]
            outcome_by_model = {outcome.model_id: outcome for outcome in example.outcomes}
            expected_row = tuple(
                _canonical_f64(outcome_by_model[model_id].quality, "evaluation target quality")
                for model_id in model_ids
            )
            if struct.pack(f"<{len(model_ids)}d", *row) != struct.pack(
                f"<{len(model_ids)}d", *expected_row
            ):
                raise ValueError(
                    "authenticated prepared-store targets do not match evaluation data"
                )
            writer = target_writers[domain_index]
            writer.text("example_id", example_id)
            writer.token("targets.f64le", struct.pack(f"<{len(row)}d", *row))
        self.target_shards = tuple(
            NativePreparedTargetShardEvidence(
                domain_index=domain_index,
                domain=plan.domains[domain_index],
                example_ids=self.example_ids_by_domain[domain_index],
                model_ids=model_ids,
                target_sha256=target_writers[domain_index].hexdigest(),
            )
            for domain_index in range(len(plan.domains))
        )

    def target_row(self, row_index: int) -> tuple[float, ...]:
        index = _exact_nonnegative_int(row_index, "prepared target row index")
        if index >= len(self.example_ids):
            raise IndexError("prepared target row index is outside the store")
        return self._store.target_row(index)


def _coefficient_semantic_sha256(
    record: NativePreparedCoefficientRecord,
    model_ids: tuple[str, ...],
) -> str:
    writer = _EvidenceWriter(f"{NATIVE_PREPARED_SEMANTIC_SCORE_ID}.coefficient")
    writer.text("solver_id", PREPARED_MOMENT_SOLVER_ID)
    for name in (
        "subset_index",
        "subset_domain_mask",
        "training_row_count",
        "active_tag_mask",
        "active_feature_count",
        "record_payload_bytes",
    ):
        writer.integer(name, getattr(record, name))
    for model_id in model_ids:
        writer.text("model_id", model_id)
    for label, view in (
        ("continuous_mean", record.continuous_means),
        ("continuous_scale", record.continuous_scales),
        ("intercept", record.intercepts),
        ("weight", record.weights),
    ):
        for row_index in range(view.row_count):
            for column_index in range(view.column_count):
                writer.f64(label, view.at(row_index, column_index))
    return writer.hexdigest()


def _raw_score_semantic_sha256(
    record: NativePreparedScoreRecord,
    example_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
) -> str:
    writer = _EvidenceWriter(f"{NATIVE_PREPARED_SEMANTIC_SCORE_ID}.raw-score-block")
    writer.text("scorer_id", PREPARED_RAW_SCORER_ID)
    for name in (
        "block_index",
        "training_subset_index",
        "scored_domain_index",
        "row_count",
        "record_payload_bytes",
    ):
        writer.integer(name, getattr(record, name))
    for model_id in model_ids:
        writer.text("model_id", model_id)
    for example_id in example_ids:
        writer.text("example_id", example_id)
    for row_index in range(record.scores.row_count):
        for column_index in range(record.scores.column_count):
            writer.f64("score", record.scores.at(row_index, column_index))
    return writer.hexdigest()


def _build_native_policy_snapshot(
    examples: tuple[EvaluationExample, ...],
    plan: PreparedNestedLodoPlan,
    model_ids: tuple[str, ...],
    estimate: NativePreparedBenchmarkEstimate,
    store: AuthenticatedPreparedStore,
    receipt: PreparedStoreFileReceipt,
    native_result: NativePreparedSessionResult,
    embedding_identity: EmbeddingIdentity | None,
    *,
    expected_binary_sha256: str,
    expected_result_sha256: str | None,
) -> _NativePreparedPolicySnapshot:
    """Consume all mapped data without invoking a caller-controlled callback."""

    if type(store) is not AuthenticatedPreparedStore or store.closed:
        raise TypeError("store must be an open exact AuthenticatedPreparedStore")
    if type(native_result) is not NativePreparedSessionResult or native_result.closed:
        raise TypeError("native_result must be an open exact NativePreparedSessionResult")
    binary_sha = _sha256_hex(expected_binary_sha256, "expected_binary_sha256")
    result_sha = (
        None
        if expected_result_sha256 is None
        else _sha256_hex(expected_result_sha256, "expected_result_sha256")
    )
    if (
        native_result.metadata != store.metadata
        or native_result.metadata != estimate.store_metadata
    ):
        raise ValueError("native result and authenticated store metadata do not match")
    if not hmac.compare_digest(native_result.store_sha256, receipt.whole_file_sha256):
        raise ValueError("native result does not descend from the caller-pinned store file")
    if not hmac.compare_digest(native_result.binary_sha256, binary_sha):
        raise ValueError("native result does not descend from the caller-pinned binary")
    if result_sha is not None and not hmac.compare_digest(native_result.result_sha256, result_sha):
        raise ValueError("native result does not match the caller-pinned result SHA-256")
    execution_receipt = NativePreparedExecutionReceipt(
        request_nonce_hex=native_result.request_nonce.hex(),
        store_file_sha256=native_result.store_sha256,
        binary_sha256=native_result.binary_sha256,
        result_sha256=native_result.result_sha256,
        result_sha256_caller_pinned=result_sha is not None,
        ridge=native_result.ridge,
    )
    policy_store = _AuthenticatedStorePolicyView(store, examples, plan, model_ids)
    coefficients = native_result.coefficients
    raw_records = native_result.scores
    coefficient_hashes = tuple(
        _coefficient_semantic_sha256(record, model_ids) for record in coefficients
    )
    raw_hashes = tuple(
        _raw_score_semantic_sha256(
            record,
            policy_store.example_ids_by_domain[record.scored_domain_index],
            model_ids,
        )
        for record in raw_records
    )
    metadata = store.metadata
    semantic_scores = NativePreparedSemanticScoreIdentity(
        graph_identity_sha256=metadata.graph_identity_sha256,
        source_fit_sha256=metadata.source_fit_sha256,
        logical_store_sha256=metadata.logical_store_sha256,
        store_payload_sha256=metadata.store_payload_sha256,
        model_catalogue_sha256=metadata.model_catalogue_sha256,
        embedding_identity_sha256=metadata.embedding_identity_sha256,
        embedding_snapshot_sha256=metadata.embedding_snapshot_sha256,
        ridge=native_result.ridge,
        coefficient_record_sha256s=coefficient_hashes,
        raw_score_block_sha256s=raw_hashes,
    )

    subset_by_domains = {
        subset.domain_indices: index for index, subset in enumerate(plan.training_subsets)
    }
    block_by_context = {
        (block.training_subset_index, block.scored_domain_index): index
        for index, block in enumerate(plan.score_blocks)
    }
    domain_count = len(plan.domains)
    calibration_subset_indices = tuple(
        index
        for index, subset in enumerate(plan.training_subsets)
        if len(subset.domain_indices) in (domain_count - 2, domain_count - 1)
    )
    calibrations: list[NativePreparedCalibrationEvidence] = []
    calibration_by_subset: dict[int, NativePreparedCalibrationEvidence] = {}
    for subset_index in calibration_subset_indices:
        subset = plan.training_subsets[subset_index]
        predictions = [[] for _ in model_ids]
        targets = [[] for _ in model_ids]
        raw_indices: list[int] = []
        raw_parent_hashes: list[str] = []
        target_parent_hashes: list[str] = []
        for calibration_domain in subset.domain_indices:
            base_domains = tuple(
                domain_index
                for domain_index in subset.domain_indices
                if domain_index != calibration_domain
            )
            base_subset_index = subset_by_domains[base_domains]
            raw_index = block_by_context[(base_subset_index, calibration_domain)]
            raw_record = raw_records[raw_index]
            target_row_indices = policy_store.example_row_indices_by_domain[calibration_domain]
            for row_index, target_row_index in enumerate(target_row_indices):
                target_row = policy_store.target_row(target_row_index)
                for model_index in range(len(model_ids)):
                    predictions[model_index].append(raw_record.scores.at(row_index, model_index))
                    targets[model_index].append(target_row[model_index])
            raw_indices.append(raw_index)
            raw_parent_hashes.append(raw_hashes[raw_index])
            target_parent_hashes.append(policy_store.target_shards[calibration_domain].sha256)
        if any(len(values) != subset.row_count for values in (*predictions, *targets)):
            raise AssertionError("native calibration did not cover its subset exactly once")
        evidence = NativePreparedCalibrationEvidence(
            training_subset_index=subset_index,
            training_domain_indices=subset.domain_indices,
            model_ids=model_ids,
            calibration_example_count=subset.row_count,
            raw_score_block_indices=tuple(raw_indices),
            raw_score_block_sha256s=tuple(raw_parent_hashes),
            target_shard_sha256s=tuple(target_parent_hashes),
            calibrators=tuple(
                IsotonicCalibrator.fit(predictions[index], targets[index])
                for index in range(len(model_ids))
            ),
        )
        calibrations.append(evidence)
        calibration_by_subset[subset_index] = evidence

    calibrated_blocks: list[_NativePreparedCalibratedScoreBlock] = []
    for subset_index in calibration_subset_indices:
        subset = plan.training_subsets[subset_index]
        calibration = calibration_by_subset[subset_index]
        for destination in range(domain_count):
            if destination in subset.domain_indices:
                continue
            raw_index = block_by_context[(subset_index, destination)]
            raw_record = raw_records[raw_index]
            example_ids = policy_store.example_ids_by_domain[destination]
            payload = bytearray(len(example_ids) * len(model_ids) * _F64_BYTES)
            for row_index in range(len(example_ids)):
                for model_index, calibrator in enumerate(calibration.calibrators):
                    calibrated = _canonical_f64(
                        calibrator.calibrate(raw_record.scores.at(row_index, model_index)),
                        "native calibrated score",
                    )
                    struct.pack_into(
                        "<d",
                        payload,
                        (row_index * len(model_ids) + model_index) * _F64_BYTES,
                        calibrated,
                    )
            calibrated_blocks.append(
                _NativePreparedCalibratedScoreBlock(
                    training_subset_index=subset_index,
                    scored_domain_index=destination,
                    raw_score_block_index=raw_index,
                    raw_score_block_sha256=raw_hashes[raw_index],
                    calibration_sha256=calibration.sha256,
                    example_ids=example_ids,
                    model_ids=model_ids,
                    scores_payload=bytes(payload),
                )
            )

    # Rehash after the final score/target read.  Any changed private snapshot or
    # result object fails before owned evidence can cross the mmap boundary.
    native_result.verify_integrity()
    if (
        not hmac.compare_digest(native_result.store_sha256, receipt.whole_file_sha256)
        or not hmac.compare_digest(native_result.binary_sha256, binary_sha)
        or (
            result_sha is not None
            and not hmac.compare_digest(native_result.result_sha256, result_sha)
        )
    ):
        raise ValueError("native result credentials changed during policy consumption")
    final_store_sha256, final_store_bytes = store.sha256_and_size()
    if final_store_bytes != metadata.file_bytes or not hmac.compare_digest(
        final_store_sha256, receipt.whole_file_sha256
    ):
        raise ValueError("authenticated prepared-store snapshot changed during consumption")
    return _NativePreparedPolicySnapshot(
        estimate=estimate,
        execution_receipt=execution_receipt,
        semantic_scores=semantic_scores,
        model_ids=model_ids,
        embedding_identity=embedding_identity,
        evaluation_data_sha256=evaluation_data_sha256(examples),
        evaluation_replay_sha256=evaluation_replay_sha256(examples),
        example_ids=tuple(example.example_id for example in examples),
        target_shards=policy_store.target_shards,
        calibrations=tuple(calibrations),
        calibrated_score_blocks=tuple(calibrated_blocks),
    )


@dataclass(frozen=True, slots=True)
class NativePreparedCalibratedScoreEvidence:
    """Payload-free durable evidence for one consumed calibrated destination."""

    training_subset_index: int
    scored_domain_index: int
    raw_score_block_index: int
    raw_score_block_sha256: str
    calibration_sha256: str
    example_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    scores_sha256: str
    calibrated_block_sha256: str
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_CALIBRATED_EVIDENCE_ID, init=False)

    def __post_init__(self) -> None:
        for name in (
            "training_subset_index",
            "scored_domain_index",
            "raw_score_block_index",
        ):
            _exact_nonnegative_int(getattr(self, name), name)
        for name in (
            "raw_score_block_sha256",
            "calibration_sha256",
            "scores_sha256",
            "calibrated_block_sha256",
        ):
            _sha256_hex(getattr(self, name), name)
        if type(self.example_ids) is not tuple or not self.example_ids:
            raise ValueError("calibrated evidence example IDs must be a non-empty exact tuple")
        if len(self.example_ids) != len(set(self.example_ids)):
            raise ValueError("calibrated evidence example IDs must be unique")
        if (
            type(self.model_ids) is not tuple
            or not self.model_ids
            or self.model_ids != tuple(sorted(set(self.model_ids)))
        ):
            raise ValueError("calibrated evidence model IDs must be sorted and unique")
        object.__setattr__(self, "sha256", _calibrated_score_evidence_sha256(self))


def _scores_payload_sha256(block: _NativePreparedCalibratedScoreBlock) -> str:
    writer = _EvidenceWriter(f"{NATIVE_PREPARED_CALIBRATED_BLOCK_ID}.scores")
    for model_id in block.model_ids:
        writer.text("model_id", model_id)
    for example_id in block.example_ids:
        writer.text("example_id", example_id)
    writer.token("scores.row-major.f64le", block.scores_payload)
    return writer.hexdigest()


def _calibrated_score_evidence_sha256(
    evidence: NativePreparedCalibratedScoreEvidence,
) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_CALIBRATED_EVIDENCE_ID)
    writer.integer("training_subset_index", evidence.training_subset_index)
    writer.integer("scored_domain_index", evidence.scored_domain_index)
    writer.integer("raw_score_block_index", evidence.raw_score_block_index)
    writer.text("raw_score_block_sha256", evidence.raw_score_block_sha256)
    writer.text("calibration_sha256", evidence.calibration_sha256)
    for model_id in evidence.model_ids:
        writer.text("model_id", model_id)
    for example_id in evidence.example_ids:
        writer.text("example_id", example_id)
    writer.text("scores_sha256", evidence.scores_sha256)
    writer.text("calibrated_block_sha256", evidence.calibrated_block_sha256)
    return writer.hexdigest()


def _payload_free_calibrated_evidence(
    blocks: tuple[_NativePreparedCalibratedScoreBlock, ...],
) -> tuple[NativePreparedCalibratedScoreEvidence, ...]:
    return tuple(
        NativePreparedCalibratedScoreEvidence(
            training_subset_index=block.training_subset_index,
            scored_domain_index=block.scored_domain_index,
            raw_score_block_index=block.raw_score_block_index,
            raw_score_block_sha256=block.raw_score_block_sha256,
            calibration_sha256=block.calibration_sha256,
            example_ids=block.example_ids,
            model_ids=block.model_ids,
            scores_sha256=_scores_payload_sha256(block),
            calibrated_block_sha256=block.sha256,
        )
        for block in blocks
    )


@dataclass(frozen=True, slots=True)
class _OwnedPreparedBatchPredictor:
    model_ids: tuple[str, ...]
    rows_by_prompt_batch: Mapping[tuple[str, ...], tuple[tuple[float, ...], ...]]

    def predict_batch(
        self,
        prompts: Sequence[str],
        model_ids: Sequence[str],
    ) -> tuple[Mapping[str, float], ...]:
        prompt_key = tuple(prompts)
        requested_models = tuple(model_ids)
        if requested_models != self.model_ids:
            raise ValueError("native prepared predictor requires the canonical model catalogue")
        try:
            rows = self.rows_by_prompt_batch[prompt_key]
        except KeyError as error:
            raise ValueError(
                "native prepared predictor received an unknown prompt batch"
            ) from error
        return tuple(MappingProxyType(dict(zip(self.model_ids, row, strict=True))) for row in rows)

    def predict(self, prompt: str, model_id: str) -> float:
        try:
            model_index = self.model_ids.index(model_id)
        except ValueError as error:
            raise ValueError("native prepared predictor received an unknown model") from error
        matches = []
        for prompts, rows in self.rows_by_prompt_batch.items():
            for candidate_prompt, row in zip(prompts, rows, strict=True):
                if candidate_prompt == prompt:
                    matches.append(row[model_index])
        if not matches or any(value != matches[0] for value in matches[1:]):
            raise ValueError("single-prompt native prepared lookup is missing or ambiguous")
        return matches[0]


class _OwnedPolicyBuilder:
    """Serve only owned calibrated rows to the existing nested evaluator."""

    __slots__ = (
        "_block_by_context",
        "_examples",
        "_examples_by_domain",
        "_plan",
        "_predictors",
        "_snapshot",
        "_subset_by_domains",
    )

    def __init__(
        self,
        examples: tuple[EvaluationExample, ...],
        snapshot: _NativePreparedPolicySnapshot,
    ) -> None:
        self._examples = examples
        self._snapshot = snapshot
        self._plan = snapshot.estimate.plan
        self._subset_by_domains = {
            subset.domain_indices: index for index, subset in enumerate(self._plan.training_subsets)
        }
        self._block_by_context = {
            (block.training_subset_index, block.scored_domain_index): block
            for block in snapshot.calibrated_score_blocks
        }
        self._examples_by_domain = tuple(
            tuple(example for example in examples if example.domain == domain)
            for domain in self._plan.domains
        )
        self._predictors: dict[int, _OwnedPreparedBatchPredictor] = {}

    def predictor(self, training: tuple[EvaluationExample, ...]) -> QualityPredictor:
        domains = tuple(sorted({example.domain for example in training}))
        domain_to_index = {domain: index for index, domain in enumerate(self._plan.domains)}
        try:
            domain_indices = tuple(domain_to_index[domain] for domain in domains)
        except KeyError as error:
            raise ValueError(
                "native prepared trainer received a domain outside the plan"
            ) from error
        expected_training = tuple(
            example
            for example in self._examples
            if domain_to_index[example.domain] in domain_indices
        )
        if training != expected_training:
            raise ValueError("native prepared trainer received noncanonical fold rows")
        try:
            subset_index = self._subset_by_domains[domain_indices]
        except KeyError as error:
            raise ValueError("native prepared trainer received an unsupported subset") from error
        cached = self._predictors.get(subset_index)
        if cached is not None:
            return cached
        subset = self._plan.training_subsets[subset_index]
        rows_by_prompt_batch: dict[tuple[str, ...], tuple[tuple[float, ...], ...]] = {}
        for destination in range(len(self._plan.domains)):
            if destination in subset.domain_indices:
                continue
            try:
                block = self._block_by_context[(subset_index, destination)]
            except KeyError as error:
                raise ValueError("owned calibrated graph is missing a destination") from error
            score_by_id = {
                example_id: block.score_row(row_index)
                for row_index, example_id in enumerate(block.example_ids)
            }
            destination_examples = self._examples_by_domain[destination]
            prompt_key = tuple(example.prompt for example in destination_examples)
            replay_rows = tuple(score_by_id[example.example_id] for example in destination_examples)
            existing = rows_by_prompt_batch.get(prompt_key)
            if existing is not None and existing != replay_rows:
                raise ValueError(
                    "native prepared predictor cannot disambiguate identical prompt batches"
                )
            rows_by_prompt_batch[prompt_key] = replay_rows
        predictor = _OwnedPreparedBatchPredictor(
            model_ids=self._snapshot.model_ids,
            rows_by_prompt_batch=MappingProxyType(rows_by_prompt_batch),
        )
        self._predictors[subset_index] = predictor
        return predictor


def _learned_evidence_sha256(learned: NestedLodoLambdaResult) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_LEARNED_EVIDENCE_ID)
    writer.text("prediction_sha256", learned.prediction_sha256)
    writer.text("evaluation_scope_algorithm", learned.report.evaluation_scope.algorithm)
    writer.text("evaluation_scope_sha256", learned.report.evaluation_scope.sha256)
    writer.integer("max_calls_per_query", learned.report.evaluation_scope.max_calls_per_query)
    for fold in learned.folds:
        writer.text("held_out_domain", fold.held_out_domain)
        for example_id in fold.training_example_ids:
            writer.text("training_example_id", example_id)
        for example_id in fold.test_example_ids:
            writer.text("test_example_id", example_id)
        writer.text("tuning_data_sha256", fold.tuning.data_sha256)
        writer.text("tuning_replay_sha256", fold.tuning.replay_sha256)
        writer.text("tuning_prediction_sha256", fold.tuning.prediction_sha256)
        for selection in fold.tuning.selections:
            writer.text("tier", selection.tier.value)
            writer.fraction("lambda", selection.lambda_cost)
            writer.f64("mean_quality", selection.mean_quality)
            writer.text("realized_cost", str(selection.realized_cost))
            writer.text("candidate_strategy", selection.candidates.strategy)
            writer.boolean("candidate_exhaustive", selection.candidates.exhaustive)
            writer.integer(
                "candidate_observed_breakpoints",
                selection.candidates.observed_breakpoint_count,
            )
            for candidate in selection.candidates.values:
                writer.fraction("candidate", candidate)
    for tier in learned.report.tiers:
        writer.text("report_tier", tier.tier_spec.tier.value)
        for query in tier.queries:
            writer.text("example_id", query.example_id)
            writer.boolean("feasible", query.feasible)
            writer.text("selected_model_id", query.selected_model_id or "")
            writer.text("cost", str(query.cost))
            writer.boolean("quality_present", query.quality is not None)
            if query.quality is not None:
                writer.f64("quality", query.quality)
            writer.boolean(
                "predicted_quality_present",
                query.predicted_quality is not None,
            )
            if query.predicted_quality is not None:
                writer.f64("predicted_quality", query.predicted_quality)
    return writer.hexdigest()


@dataclass(frozen=True, slots=True)
class NativePreparedBenchmarkResult:
    """Durable native-policy result aligned with six per-query baselines."""

    estimate: NativePreparedBenchmarkEstimate
    config: NativePreparedBenchmarkConfig
    execution_receipt: NativePreparedExecutionReceipt
    semantic_scores: NativePreparedSemanticScoreIdentity
    evaluation_data_sha256: str
    evaluation_replay_sha256: str
    target_shards: tuple[NativePreparedTargetShardEvidence, ...]
    calibrations: tuple[NativePreparedCalibrationEvidence, ...]
    calibrated_score_blocks: tuple[NativePreparedCalibratedScoreEvidence, ...]
    learned: NestedLodoLambdaResult
    baselines: LodoSixBaselineEvaluation
    learned_evidence_sha256: str
    policy_snapshot_sha256: str = field(init=False)
    all_searches_exhaustive: bool = field(init=False)
    learned_gap_recovery: float | None = field(init=False)
    learned_total_cost: Decimal = field(init=False)
    learned_quote_error: QuoteErrorReport = field(init=False)
    sha256: str = field(init=False)
    algorithm_id: str = field(default=NATIVE_PREPARED_BENCHMARK_ALGORITHM_ID, init=False)
    accounting_scope: str = field(default="per-query", init=False)

    def __post_init__(self) -> None:
        if type(self.estimate) is not NativePreparedBenchmarkEstimate:
            raise TypeError("estimate must be exact NativePreparedBenchmarkEstimate")
        if type(self.config) is not NativePreparedBenchmarkConfig:
            raise TypeError("config must be exact NativePreparedBenchmarkConfig")
        if (
            self.config.max_candidates_per_tier
            != self.estimate.policy_estimate.max_candidates_per_tier
        ):
            raise ValueError("config candidate cap does not match the admitted estimate")
        if type(self.execution_receipt) is not NativePreparedExecutionReceipt:
            raise TypeError("execution_receipt must be exact NativePreparedExecutionReceipt")
        if not self.execution_receipt.result_sha256_caller_pinned:
            raise ValueError("durable native benchmark requires a caller-pinned result SHA-256")
        if type(self.semantic_scores) is not NativePreparedSemanticScoreIdentity:
            raise TypeError("semantic_scores must be exact NativePreparedSemanticScoreIdentity")
        for name in (
            "evaluation_data_sha256",
            "evaluation_replay_sha256",
            "learned_evidence_sha256",
        ):
            _sha256_hex(getattr(self, name), name)
        if type(self.learned) is not NestedLodoLambdaResult:
            raise TypeError("learned must be exact NestedLodoLambdaResult")
        if type(self.baselines) is not LodoSixBaselineEvaluation:
            raise TypeError("baselines must be exact LodoSixBaselineEvaluation")
        if self.learned_evidence_sha256 != _learned_evidence_sha256(self.learned):
            raise ValueError("learned evidence SHA-256 does not match the learned result")
        plan = self.estimate.plan
        metadata = self.estimate.store_metadata
        semantic_metadata = (
            self.semantic_scores.graph_identity_sha256,
            self.semantic_scores.source_fit_sha256,
            self.semantic_scores.logical_store_sha256,
            self.semantic_scores.store_payload_sha256,
            self.semantic_scores.model_catalogue_sha256,
            self.semantic_scores.embedding_identity_sha256,
            self.semantic_scores.embedding_snapshot_sha256,
        )
        expected_semantic_metadata = (
            metadata.graph_identity_sha256,
            metadata.source_fit_sha256,
            metadata.logical_store_sha256,
            metadata.store_payload_sha256,
            metadata.model_catalogue_sha256,
            metadata.embedding_identity_sha256,
            metadata.embedding_snapshot_sha256,
        )
        if semantic_metadata != expected_semantic_metadata:
            raise ValueError("semantic score identity does not match admitted store metadata")
        if self.execution_receipt.ridge != self.semantic_scores.ridge:
            raise ValueError("execution and semantic score identities disagree on ridge")
        if len(self.semantic_scores.coefficient_record_sha256s) != len(
            plan.training_subsets
        ) or len(self.semantic_scores.raw_score_block_sha256s) != len(plan.score_blocks):
            raise ValueError("semantic score record coverage does not match the plan")
        domain_count = len(plan.domains)
        expected_subsets = tuple(
            index
            for index, subset in enumerate(plan.training_subsets)
            if len(subset.domain_indices) in (domain_count - 2, domain_count - 1)
        )
        if (
            type(self.target_shards) is not tuple
            or len(self.target_shards) != domain_count
            or any(
                type(item) is not NativePreparedTargetShardEvidence for item in self.target_shards
            )
            or tuple((item.domain_index, item.domain) for item in self.target_shards)
            != tuple(enumerate(plan.domains))
        ):
            raise ValueError("benchmark target shards must cover every domain")
        target_model_ids = self.target_shards[0].model_ids
        if len(target_model_ids) != plan.target_count or any(
            item.model_ids != target_model_ids for item in self.target_shards
        ):
            raise ValueError("benchmark target shards must share one model catalogue")
        if (
            type(self.calibrations) is not tuple
            or tuple(item.training_subset_index for item in self.calibrations) != expected_subsets
            or any(
                type(item) is not NativePreparedCalibrationEvidence for item in self.calibrations
            )
        ):
            raise ValueError("benchmark calibrations must cover the canonical policy subsets")
        subset_by_domains = {
            subset.domain_indices: index for index, subset in enumerate(plan.training_subsets)
        }
        block_by_context = {
            (block.training_subset_index, block.scored_domain_index): index
            for index, block in enumerate(plan.score_blocks)
        }
        for calibration in self.calibrations:
            subset = plan.training_subsets[calibration.training_subset_index]
            expected_raw_indices = tuple(
                block_by_context[
                    (
                        subset_by_domains[
                            tuple(
                                domain_index
                                for domain_index in subset.domain_indices
                                if domain_index != calibration_domain
                            )
                        ],
                        calibration_domain,
                    )
                ]
                for calibration_domain in subset.domain_indices
            )
            if (
                calibration.training_domain_indices != subset.domain_indices
                or calibration.calibration_example_count != subset.row_count
                or calibration.model_ids != target_model_ids
                or calibration.raw_score_block_indices != expected_raw_indices
                or calibration.raw_score_block_sha256s
                != tuple(
                    self.semantic_scores.raw_score_block_sha256s[index]
                    for index in expected_raw_indices
                )
                or calibration.target_shard_sha256s
                != tuple(self.target_shards[index].sha256 for index in subset.domain_indices)
            ):
                raise ValueError("benchmark calibration lineage does not match the plan")
        expected_blocks = tuple(
            (subset_index, domain_index)
            for subset_index in expected_subsets
            for domain_index in range(domain_count)
            if domain_index not in plan.training_subsets[subset_index].domain_indices
        )
        if (
            type(self.calibrated_score_blocks) is not tuple
            or any(
                type(item) is not NativePreparedCalibratedScoreEvidence
                for item in self.calibrated_score_blocks
            )
            or tuple(
                (item.training_subset_index, item.scored_domain_index)
                for item in self.calibrated_score_blocks
            )
            != expected_blocks
        ):
            raise ValueError("benchmark calibrated evidence must use exact payload-free records")
        calibration_by_subset = {item.training_subset_index: item for item in self.calibrations}
        for block in self.calibrated_score_blocks:
            expected_raw_index = block_by_context[
                (block.training_subset_index, block.scored_domain_index)
            ]
            if (
                block.raw_score_block_index != expected_raw_index
                or block.raw_score_block_sha256
                != self.semantic_scores.raw_score_block_sha256s[expected_raw_index]
                or block.calibration_sha256
                != calibration_by_subset[block.training_subset_index].sha256
                or block.example_ids != self.target_shards[block.scored_domain_index].example_ids
                or block.model_ids != target_model_ids
            ):
                raise ValueError("benchmark calibrated score lineage does not match the plan")
        if self.learned.report.router_name != "nested-lodo-tier-lambda":
            raise ValueError("native learned report must use the nested LODO router identity")
        if tuple(row.name for row in self.baselines.baselines) != BASELINE_NAMES:
            raise ValueError("native benchmark must contain all six canonical baselines")
        if (
            self.baselines.random_seed != self.config.random_seed
            or self.baselines.character_threshold != self.config.character_threshold
        ):
            raise ValueError("native benchmark baseline controls do not match config")
        scopes = {
            self.learned.report.evaluation_scope,
            *(row.report.evaluation_scope for row in self.baselines.baselines),
        }
        if len(scopes) != 1:
            raise ValueError("native learned and baseline reports must share one scope")
        learned_folds = tuple(
            (fold.held_out_domain, fold.training_example_ids, fold.test_example_ids)
            for fold in self.learned.folds
        )
        baseline_folds = tuple(
            (fold.held_out_domain, fold.training_example_ids, fold.test_example_ids)
            for fold in self.baselines.folds
        )
        if learned_folds != baseline_folds:
            raise ValueError("native learned and baseline outer folds must match exactly")
        if tuple(row[0] for row in learned_folds) != plan.domains:
            raise ValueError("native benchmark fold order does not match the prepared plan")
        for domain_index, fold in enumerate(self.baselines.folds):
            if set(fold.test_example_ids) != set(self.target_shards[domain_index].example_ids):
                raise ValueError("native benchmark fold rows do not match target shards")
        expected_ids = self.baselines.example_ids
        shard_ids = tuple(
            example_id for shard in self.target_shards for example_id in shard.example_ids
        )
        if len(shard_ids) != len(set(shard_ids)) or set(shard_ids) != set(expected_ids):
            raise ValueError("native target evidence and benchmark replay IDs differ")
        for domain_index, shard in enumerate(self.target_shards):
            if len(shard.example_ids) != plan.domain_example_counts[domain_index]:
                raise ValueError("native target shard row count does not match the plan")
        reference_report = self.baselines.baselines[0].report
        expected_specs = tuple(tier.tier_spec for tier in reference_report.tiers)
        if (
            self.baselines.candidate_model_ids != target_model_ids
            or len(expected_specs) != self.estimate.policy_estimate.tier_count
            or tuple(tier.tier_spec for tier in self.learned.report.tiers) != expected_specs
        ):
            raise ValueError("native learned and baseline tier specifications must match")
        cap = self.estimate.policy_estimate.max_candidates_per_tier
        expected_tiers = tuple(spec.tier for spec in expected_specs)
        for fold in self.learned.folds:
            if tuple(selection.tier for selection in fold.tuning.selections) != expected_tiers:
                raise ValueError("native fold tuning tiers do not match replay tiers")
            if any(len(selection.candidates.values) > cap for selection in fold.tuning.selections):
                raise ValueError("native fold lambda candidates exceed the admitted cap")
        for tier in self.learned.report.tiers:
            query_ids = tuple(query.example_id for query in tier.queries)
            if query_ids != expected_ids or tier.budget.query_order != expected_ids:
                raise ValueError("native learned replay does not preserve baseline row order")
            if tier.budget.adapter_name != "per-query":
                raise ValueError("native six-baseline benchmark requires per-query accounting")
            if tier.budget.effective_total_limit != scale_cost(
                tier.tier_spec.budget_limit,
                len(expected_ids),
            ):
                raise ValueError("native learned per-query budget limit is inconsistent")
            if tier.budget.spent != sum_costs(query.cost for query in tier.queries):
                raise ValueError("native learned spend does not equal replayed calls")
        baseline_by_name = self.baselines.by_name()
        expected_gap = oracle_gap_recovery(
            self.learned.report,
            baseline_by_name["always-cheapest"].report,
            baseline_by_name["oracle"].report,
        )
        if expected_gap is not None and not math.isfinite(expected_gap):
            raise ValueError("native learned oracle-gap recovery must be finite")
        expected_cost = sum_costs(
            query.cost for tier in self.learned.report.tiers for query in tier.queries
        )
        ModelSpec("native-learned-total-cost", expected_cost)
        expected_quote_error = summarize_quote_error(self.learned.report)
        oracle_by_tier = baseline_by_name["oracle"].report.by_tier()
        for tier, learned_tier in self.learned.report.by_tier().items():
            oracle_queries = {query.example_id: query for query in oracle_by_tier[tier].queries}
            for query in learned_tier.queries:
                oracle_query = oracle_queries[query.example_id]
                if not oracle_query.feasible or oracle_query.quality is None:
                    raise ValueError("aligned per-query oracle must be feasible and complete")
                if (
                    query.feasible
                    and query.quality is not None
                    and query.quality > oracle_query.quality + 1e-12
                ):
                    raise ValueError("native learned quality cannot exceed the aligned oracle")
        exhaustive = all(
            selection.candidates.exhaustive
            for fold in self.learned.folds
            for selection in fold.tuning.selections
        )
        object.__setattr__(self, "all_searches_exhaustive", exhaustive)
        object.__setattr__(self, "learned_gap_recovery", expected_gap)
        object.__setattr__(self, "learned_total_cost", expected_cost)
        object.__setattr__(self, "learned_quote_error", expected_quote_error)
        object.__setattr__(
            self,
            "policy_snapshot_sha256",
            _returned_policy_snapshot_sha256(self),
        )
        object.__setattr__(self, "sha256", _native_benchmark_result_sha256(self))


def _returned_policy_snapshot_sha256(result: NativePreparedBenchmarkResult) -> str:
    """Reconstruct the private owned-snapshot identity without retaining payloads."""

    writer = _EvidenceWriter(NATIVE_PREPARED_POLICY_SNAPSHOT_ID)
    writer.text("execution_receipt_sha256", result.execution_receipt.sha256)
    writer.text("semantic_score_sha256", result.semantic_scores.sha256)
    writer.text("evaluation_data_sha256", result.evaluation_data_sha256)
    writer.text("evaluation_replay_sha256", result.evaluation_replay_sha256)
    for model_id in result.target_shards[0].model_ids:
        writer.text("model_id", model_id)
    for example_id in result.baselines.example_ids:
        writer.text("example_id", example_id)
    for item in result.target_shards:
        writer.text("target_shard_sha256", item.sha256)
    for item in result.calibrations:
        writer.text("calibration_sha256", item.sha256)
    for item in result.calibrated_score_blocks:
        writer.text("calibrated_score_block_sha256", item.calibrated_block_sha256)
    return writer.hexdigest()


def _native_benchmark_result_sha256(result: NativePreparedBenchmarkResult) -> str:
    writer = _EvidenceWriter(NATIVE_PREPARED_BENCHMARK_ALGORITHM_ID)
    writer.integer("max_candidates_per_tier", result.config.max_candidates_per_tier)
    writer.integer("random_seed", result.config.random_seed)
    writer.integer("character_threshold", result.config.character_threshold)
    writer.text("accounting_scope", result.config.accounting_scope)
    writer.boolean("allow_large_exhaustive", result.config.allow_large_exhaustive)
    writer.text("policy_snapshot_sha256", result.policy_snapshot_sha256)
    writer.text("execution_receipt_sha256", result.execution_receipt.sha256)
    writer.text("semantic_score_sha256", result.semantic_scores.sha256)
    writer.text("evaluation_data_sha256", result.evaluation_data_sha256)
    writer.text("evaluation_replay_sha256", result.evaluation_replay_sha256)
    writer.text("learned_evidence_sha256", result.learned_evidence_sha256)
    writer.text(
        "baseline_config_evidence_sha256",
        result.baselines.baseline_config_evidence_sha256,
    )
    for item in result.target_shards:
        writer.text("target_shard_sha256", item.sha256)
    for item in result.calibrations:
        writer.text("calibration_sha256", item.sha256)
    for item in result.calibrated_score_blocks:
        writer.text("calibrated_score_evidence_sha256", item.sha256)
    return writer.hexdigest()


_DEFAULT_NATIVE_PREPARED_BENCHMARK_CONFIG = NativePreparedBenchmarkConfig()


def evaluate_native_prepared_per_query_benchmark(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    store_path: str | os.PathLike[str],
    store_receipt: PreparedStoreFileReceipt,
    native_result: NativePreparedSessionResult,
    *,
    expected_binary_sha256: str,
    expected_result_sha256: str,
    embedding_identity: EmbeddingIdentity | None,
    config: NativePreparedBenchmarkConfig = _DEFAULT_NATIVE_PREPARED_BENCHMARK_CONFIG,
) -> NativePreparedBenchmarkResult:
    """Consume one native session into a bounded, per-query benchmark.

    The caller retains ownership of ``native_result``.  This function authenticates
    and closes its own prepared-store snapshot before policy or baseline replay.  It
    never builds ``PreparedRawScoreBundle`` and exposes no ledger callback, cumulative
    accounting, cascade behavior, or network path.
    """

    if type(config) is not NativePreparedBenchmarkConfig:
        raise TypeError("config must be exact NativePreparedBenchmarkConfig")
    if type(store_receipt) is not PreparedStoreFileReceipt:
        raise TypeError("store_receipt must be exact PreparedStoreFileReceipt")
    if type(native_result) is not NativePreparedSessionResult or native_result.closed:
        raise TypeError("native_result must be an open exact NativePreparedSessionResult")
    if not isinstance(store_path, (str, os.PathLike)) or isinstance(store_path, bytes):
        raise TypeError("store_path must be text or an os.PathLike value")
    if embedding_identity is not None and type(embedding_identity) is not EmbeddingIdentity:
        raise TypeError("embedding_identity must be exact EmbeddingIdentity or None")
    binary_sha256 = _sha256_hex(expected_binary_sha256, "expected_binary_sha256")
    result_sha256 = _sha256_hex(expected_result_sha256, "expected_result_sha256")

    # Check the caller-pinned credentials and fixed shapes before taking a deep
    # evaluation snapshot or authenticating another file.
    if not hmac.compare_digest(native_result.binary_sha256, binary_sha256):
        raise ValueError("native result does not match the caller-pinned binary")
    if not hmac.compare_digest(native_result.result_sha256, result_sha256):
        raise ValueError("native result does not match the caller-pinned result file")
    if not hmac.compare_digest(
        native_result.store_sha256,
        store_receipt.whole_file_sha256,
    ):
        raise ValueError("native result does not match the caller-pinned store")
    metadata = native_result.metadata
    # Authenticate the current result mapping before any deep evaluation snapshot,
    # fold construction, metadata-tag traversal, or lambda preflight.
    native_result.verify_integrity()
    if type(examples) is not tuple or not examples:
        raise TypeError("examples must be a non-empty exact tuple")
    if any(type(example) is not EvaluationExample for example in examples):
        raise TypeError("examples must contain exact EvaluationExample values")
    counts_by_domain: dict[str, int] = {}
    for example in examples:
        counts_by_domain[example.domain] = counts_by_domain.get(example.domain, 0) + 1
        if len(counts_by_domain) > 7:
            raise ValueError("native prepared benchmark supports at most seven domains")
    domains = tuple(sorted(counts_by_domain))
    domain_counts = tuple(counts_by_domain[domain] for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        domain_counts,
        feature_count=metadata.feature_count,
        target_count=metadata.target_count,
    )
    snapshot_examples, snapshot_specs, model_ids, estimate = _preflight_snapshot(
        examples,
        tier_specs,
        plan,
        metadata,
        store_receipt,
        embedding_identity,
        max_candidates_per_tier=config.max_candidates_per_tier,
    )

    store = authenticate_prepared_store_file(store_path, store_receipt)
    try:
        validate_prepared_store_context(
            store.metadata,
            store_receipt,
            plan,
            model_ids,
            embedding_identity,
        )
        snapshot = _build_native_policy_snapshot(
            snapshot_examples,
            plan,
            model_ids,
            estimate,
            store,
            store_receipt,
            native_result,
            embedding_identity,
            expected_binary_sha256=binary_sha256,
            expected_result_sha256=result_sha256,
        )
    except BaseException:
        # A cleanup failure must never replace the authentication or consumption
        # failure that explains why no benchmark was returned.
        for _ in range(2):
            try:
                store.close()
                break
            except BaseException:
                pass
        raise
    else:
        try:
            store.close()
        except BaseException:
            try:
                store.close()
            except BaseException:
                pass
            raise

    # From this point onward no mmap-backed store or native view is consulted.
    builder = _OwnedPolicyBuilder(snapshot_examples, snapshot)
    learned = nested_lodo_lambda_evaluation(
        snapshot_examples,
        snapshot_specs,
        builder.predictor,
        PerQueryBudgetLedger,
        max_candidates_per_tier=config.max_candidates_per_tier,
        allow_large_exhaustive=config.allow_large_exhaustive,
    )
    catalogue = tuple(
        sorted(snapshot_examples[0].candidate_models, key=lambda model: model.model_id)
    )
    premium_model_id = max(
        catalogue,
        key=lambda model: (model.cost, model.model_id),
    ).model_id
    baselines = evaluate_per_query_lodo_baselines(
        snapshot_examples,
        snapshot_specs,
        PerQueryBudgetLedger,
        premium_model_id=premium_model_id,
        strong_model_id=premium_model_id,
        random_seed=config.random_seed,
        character_threshold=config.character_threshold,
    )
    result = NativePreparedBenchmarkResult(
        estimate=estimate,
        config=config,
        execution_receipt=snapshot.execution_receipt,
        semantic_scores=snapshot.semantic_scores,
        evaluation_data_sha256=snapshot.evaluation_data_sha256,
        evaluation_replay_sha256=snapshot.evaluation_replay_sha256,
        target_shards=snapshot.target_shards,
        calibrations=snapshot.calibrations,
        calibrated_score_blocks=_payload_free_calibrated_evidence(snapshot.calibrated_score_blocks),
        learned=learned,
        baselines=baselines,
        learned_evidence_sha256=_learned_evidence_sha256(learned),
    )
    if result.policy_snapshot_sha256 != snapshot.sha256:
        raise AssertionError("payload-free policy identity changed during handoff")
    return result
