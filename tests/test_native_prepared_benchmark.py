# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity for the native prepared policy benchmark."""

from __future__ import annotations

import hashlib
import math
import mmap
import os
import shutil
import struct
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from pathlib import Path

import pytest

import tierroute.policies.native_prepared_benchmark as native_benchmark_module
import tierroute.predictors.prepared_execution as prepared_execution_module
from tierroute.adapters import load_evaluation_dataset
from tierroute.core import ModelSpec
from tierroute.eval import CandidateOutcome, EvaluationExample, TierSpec
from tierroute.policies import (
    NativePreparedBenchmarkConfig,
    NativePreparedBenchmarkResult,
    NativePreparedCalibrationEvidence,
    PerQueryNestedLodoBenchmark,
    evaluate_native_prepared_per_query_benchmark,
    evaluate_per_query_bilinear_benchmark,
    preflight_native_prepared_benchmark,
)
from tierroute.predictors.calibration import IsotonicCalibrator
from tierroute.predictors.native_prepared import (
    NativePreparedFloat64View,
    NativePreparedIntegrityError,
    NativePreparedSessionAdapter,
    NativePreparedSessionResult,
)
from tierroute.predictors.prepared_files import (
    AuthenticatedPreparedStore,
    PreparedStoreFileReceipt,
    authenticate_prepared_store_file,
    write_prepared_store_file,
    write_prepared_store_file_from_sections,
)
from tierroute.predictors.prepared_graph import build_prepared_nested_lodo_plan
from tierroute.predictors.prepared_store import (
    build_prepared_feature_store,
    prepared_fit_source_sha256,
)

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_SOURCE = _ROOT / "native" / "tierroute_prepared.c"
_MODEL_COSTS = {
    "cheap": Decimal("0.20"),
    "mid": Decimal("0.60"),
    "premium": Decimal("1.00"),
}
_MODEL_QUALITY_BASES = {"cheap": 0.2, "mid": 0.4, "premium": 0.6}
_PROMPTS = (
    "Debug this Python API and code safely.",
    "Prove the equation x^2 + y^2 = 1.",
    "Review this legal contract and court statute.",
    "Assess a clinical medicine diagnosis.",
)
_DEFAULT_CONFIG = NativePreparedBenchmarkConfig()


class _ExplodingMetadata(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        del key
        raise AssertionError("deep metadata traversal happened before result verification")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("deep metadata traversal happened before result verification")

    def __len__(self) -> int:
        raise AssertionError("deep metadata traversal happened before result verification")


def _object_graph(root: object) -> tuple[object, ...]:
    observed: list[object] = []
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        observed.append(value)
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif isinstance(value, Mapping):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
    return tuple(observed)


@pytest.fixture(scope="module")
def compiled_native_prepared_benchmark(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str]:
    compiler = (
        shutil.which("cl") or shutil.which("clang-cl")
        if os.name == "nt"
        else shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    )
    if compiler is None or not _NATIVE_SOURCE.is_file():
        pytest.skip("native prepared C11 source or platform compiler is unavailable")
    executable = tmp_path_factory.mktemp("native-policy-benchmark") / (
        "tierroute-prepared.exe" if os.name == "nt" else "tierroute-prepared"
    )
    if os.name == "nt":
        command = [
            compiler,
            "/nologo",
            "/std:c11",
            "/O2",
            "/MT",
            "/W4",
            "/WX",
            "/D_CRT_SECURE_NO_WARNINGS",
            str(_NATIVE_SOURCE),
            f"/Fo:{executable.with_suffix('.obj')}",
            f"/Fe:{executable}",
        ]
    else:
        command = [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(_NATIVE_SOURCE),
            "-lm",
            "-o",
            str(executable),
        ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        pytest.fail(f"native prepared compile failed:\n{completed.stderr}")
    executable.chmod(0o500)
    return executable, hashlib.sha256(executable.read_bytes()).hexdigest()


def _generated_examples(
    domain_row_counts: tuple[int, ...],
    model_ids: tuple[str, ...],
) -> tuple[EvaluationExample, ...]:
    examples: list[EvaluationExample] = []
    row_index = 0
    for domain_index, row_count in enumerate(domain_row_counts):
        for _ in range(row_count):
            model_specs = tuple(
                ModelSpec(model_id, _MODEL_COSTS[model_id]) for model_id in reversed(model_ids)
            )
            outcomes = tuple(
                CandidateOutcome(
                    model_id=model_id,
                    output=f"domain-row-{row_index:02d}:{model_id}",
                    cost=_MODEL_COSTS[model_id],
                    quality=_MODEL_QUALITY_BASES[model_id]
                    + (0.013 + 0.002 * model_ids.index(model_id)) * row_index,
                )
                for model_id in model_ids
            )
            examples.append(
                EvaluationExample(
                    example_id=f"domain-row-{row_index:02d}",
                    prompt=f"{_PROMPTS[row_index % len(_PROMPTS)]} Variant {row_index}.",
                    domain=f"domain-{domain_index:02d}",
                    candidate_models=model_specs,
                    outcomes=outcomes,
                )
            )
            row_index += 1
    return tuple(examples)


def _run_native_benchmark(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    store_path: Path,
    compiled_native: tuple[Path, str],
    *,
    config: NativePreparedBenchmarkConfig = _DEFAULT_CONFIG,
) -> tuple[NativePreparedBenchmarkResult, PerQueryNestedLodoBenchmark]:
    with _open_native_session(examples, store_path, compiled_native) as (
        receipt,
        native_result,
        binary_sha256,
    ):
        result = evaluate_native_prepared_per_query_benchmark(
            examples,
            tier_specs,
            store_path,
            receipt,
            native_result,
            expected_binary_sha256=binary_sha256,
            expected_result_sha256=native_result.result_sha256,
            embedding_identity=None,
            config=config,
        )
        assert not native_result.closed
    reference = evaluate_per_query_bilinear_benchmark(
        examples,
        tier_specs,
        max_candidates_per_tier=config.max_candidates_per_tier,
    )
    return result, reference


@contextmanager
def _open_native_session(
    examples: tuple[EvaluationExample, ...],
    store_path: Path,
    compiled_native: tuple[Path, str],
) -> Iterator[tuple[PreparedStoreFileReceipt, NativePreparedSessionResult, str]]:
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=12,
        target_count=len(examples[0].candidate_models),
    )
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )
    receipt = write_prepared_store_file(store, store_path)
    executable, binary_sha256 = compiled_native

    with NativePreparedSessionAdapter(
        executable,
        binary_sha256,
        timeout_seconds=120,
    ).run(store_path, receipt, ridge=1.0) as native_result:
        yield receipt, native_result, binary_sha256


def _assert_reference_parity(
    result: NativePreparedBenchmarkResult,
    reference: PerQueryNestedLodoBenchmark,
    *,
    domain_count: int,
) -> None:
    assert result.learned == reference.learned
    assert result.baselines == reference.baselines
    assert result.evaluation_data_sha256 == reference.data_sha256
    assert result.evaluation_replay_sha256 == reference.replay_sha256

    actual_folds = tuple(
        (fold.held_out_domain, fold.training_example_ids, fold.test_example_ids)
        for fold in result.learned.folds
    )
    reference_folds = tuple(
        (fold.held_out_domain, fold.training_example_ids, fold.test_example_ids)
        for fold in reference.learned.folds
    )
    assert actual_folds == reference_folds
    actual_decisions = tuple(
        (
            tier.tier_spec.tier,
            tuple(
                (query.example_id, query.selected_model_id, query.feasible, query.calls)
                for query in tier.queries
            ),
        )
        for tier in result.learned.report.tiers
    )
    reference_decisions = tuple(
        (
            tier.tier_spec.tier,
            tuple(
                (query.example_id, query.selected_model_id, query.feasible, query.calls)
                for query in tier.queries
            ),
        )
        for tier in reference.learned.report.tiers
    )
    assert actual_decisions == reference_decisions
    assert tuple(tier.budget for tier in result.learned.report.tiers) == tuple(
        tier.budget for tier in reference.learned.report.tiers
    )

    assert len(result.target_shards) == domain_count
    assert len(result.calibrations) == domain_count * (domain_count + 1) // 2
    assert len(result.calibrated_score_blocks) == domain_count**2
    assert result.estimate.materialized_feature_bytes == 0
    assert result.estimate.materialized_raw_score_bytes == 0
    assert all(not hasattr(block, "scores_payload") for block in result.calibrated_score_blocks)
    estimate = result.estimate
    tier_count = estimate.policy_estimate.tier_count
    example_count = estimate.plan.work.example_count
    model_count = estimate.plan.target_count
    assert estimate.baseline_metadata_prep_work_units == 2 * sum(
        fold.training_rows for fold in estimate.baseline_folds
    )
    assert estimate.baseline_tag_comparison_work_units == tier_count * sum(
        fold.unique_observable_tags * fold.tagged_rows for fold in estimate.baseline_folds
    )
    assert estimate.baseline_domain_table_work_units == tier_count * sum(
        2 * model_count * fold.tagged_rows + model_count * fold.unique_observable_tags
        for fold in estimate.baseline_folds
    )
    assert estimate.baseline_oracle_work_units == (3 * tier_count * example_count * model_count)
    assert estimate.baseline_replay_work_units == (
        6 * tier_count * example_count * (2 * model_count + 8)
    )
    assert estimate.baseline_constructor_evidence_work_units == (
        6 * 32 * tier_count * example_count
    )
    assert estimate.baseline_work_units == sum(
        (
            estimate.baseline_metadata_prep_work_units,
            estimate.baseline_tag_comparison_work_units,
            estimate.baseline_domain_table_work_units,
            estimate.baseline_oracle_work_units,
            estimate.baseline_replay_work_units,
            estimate.baseline_constructor_evidence_work_units,
        )
    )
    assert estimate.bridge_work_units == sum(
        (
            estimate.policy_estimate.postprocess_work_units,
            estimate.baseline_work_units,
            estimate.semantic_score_cells_hashed,
            estimate.semantic_coefficient_cells_hashed,
            estimate.target_cells_validated,
        )
    )
    assert estimate.owned_numeric_bytes == sum(
        (
            estimate.policy_estimate.postprocess_numeric_bytes,
            estimate.owned_calibrated_score_bytes,
            estimate.row_index_bytes,
        )
    )


def test_native_d4_policy_benchmark_matches_authoritative_rowwise_reference(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset()
    result, reference = _run_native_benchmark(
        dataset.examples,
        dataset.tier_specs,
        tmp_path / "d4-policy-benchmark.trpstore",
        compiled_native_prepared_benchmark,
    )

    _assert_reference_parity(result, reference, domain_count=4)


@pytest.mark.parametrize(
    ("domain_row_counts", "model_ids"),
    (
        ((2, 2, 2, 2, 2), ("cheap", "premium")),
        ((2, 2, 2, 2, 2, 2), ("cheap", "premium")),
        ((1, 2, 1, 3, 2, 1, 2), ("cheap", "mid", "premium")),
    ),
    ids=("d5-two-models", "d6-two-models", "d7-uneven-three-models"),
)
def test_native_d5_to_d7_policy_benchmarks_match_authoritative_rowwise_reference(
    domain_row_counts: tuple[int, ...],
    model_ids: tuple[str, ...],
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
) -> None:
    examples = _generated_examples(domain_row_counts, model_ids)
    tier_specs = load_evaluation_dataset().tier_specs
    result, reference = _run_native_benchmark(
        examples,
        tier_specs,
        tmp_path / f"d{len(domain_row_counts)}-policy-benchmark.trpstore",
        compiled_native_prepared_benchmark,
    )

    _assert_reference_parity(result, reference, domain_count=len(domain_row_counts))


def test_native_cap_two_handles_ties_duplicate_prompts_one_ulp_and_realized_costs(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
) -> None:
    generated = _generated_examples((2, 2, 2, 2), ("cheap", "premium"))
    examples = tuple(
        replace(
            example,
            prompt="Duplicate prompt with x^2 and Python code.",
            outcomes=tuple(
                replace(
                    outcome,
                    quality=(
                        math.nextafter(0.5, 1.0)
                        if index == 0 and outcome.model_id == "premium"
                        else 0.5
                    ),
                    cost=(Decimal("1.10") if outcome.model_id == "premium" else outcome.cost),
                )
                for outcome in example.outcomes
            ),
        )
        for index, example in enumerate(generated)
    )
    config = NativePreparedBenchmarkConfig(max_candidates_per_tier=2)
    tier_specs = load_evaluation_dataset().tier_specs
    result, reference = _run_native_benchmark(
        examples,
        tier_specs,
        tmp_path / "cap-two-adversarial.trpstore",
        compiled_native_prepared_benchmark,
        config=config,
    )

    _assert_reference_parity(result, reference, domain_count=4)
    assert result.config.max_candidates_per_tier == 2
    assert all(
        len(selection.candidates.values) <= 2
        for fold in result.learned.folds
        for selection in fold.tuning.selections
    )


def test_result_integrity_runs_before_deep_evaluation_snapshot(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    guarded_examples = tuple(
        replace(example, router_metadata=_ExplodingMetadata()) for example in dataset.examples
    )
    store_path = tmp_path / "verify-before-snapshot.trpstore"
    with _open_native_session(
        dataset.examples,
        store_path,
        compiled_native_prepared_benchmark,
    ) as (receipt, native_result, binary_sha256):
        monkeypatch.setattr(
            NativePreparedSessionResult,
            "verify_integrity",
            lambda self: (_ for _ in ()).throw(RuntimeError("integrity sentinel")),
        )
        with pytest.raises(RuntimeError, match="integrity sentinel"):
            evaluate_native_prepared_per_query_benchmark(
                guarded_examples,
                dataset.tier_specs,
                store_path,
                receipt,
                native_result,
                expected_binary_sha256=binary_sha256,
                expected_result_sha256=native_result.result_sha256,
                embedding_identity=None,
            )


def test_wrong_external_credentials_fail_before_store_authentication(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    store_path = tmp_path / "credential-preflight.trpstore"
    with _open_native_session(
        dataset.examples,
        store_path,
        compiled_native_prepared_benchmark,
    ) as (receipt, native_result, binary_sha256):
        monkeypatch.setattr(
            native_benchmark_module,
            "authenticate_prepared_store_file",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("store authentication must not start")
            ),
        )
        invalid_cases = (
            {
                "expected_binary_sha256": "0" * 64,
                "expected_result_sha256": native_result.result_sha256,
                "store_receipt": receipt,
            },
            {
                "expected_binary_sha256": binary_sha256,
                "expected_result_sha256": "0" * 64,
                "store_receipt": receipt,
            },
            {
                "expected_binary_sha256": binary_sha256,
                "expected_result_sha256": native_result.result_sha256,
                "store_receipt": replace(receipt, whole_file_sha256="0" * 64),
            },
        )
        for case in invalid_cases:
            with pytest.raises(ValueError, match="caller-pinned"):
                evaluate_native_prepared_per_query_benchmark(
                    dataset.examples,
                    dataset.tier_specs,
                    store_path,
                    case["store_receipt"],
                    native_result,
                    expected_binary_sha256=case["expected_binary_sha256"],
                    expected_result_sha256=case["expected_result_sha256"],
                    embedding_identity=None,
                )


def test_final_result_pin_is_rechecked_after_last_mapping_verification(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    store_path = tmp_path / "final-result-pin.trpstore"
    original_verify = NativePreparedSessionResult.verify_integrity
    calls = 0

    def verify_then_replace_pin(self: NativePreparedSessionResult) -> None:
        nonlocal calls
        calls += 1
        original_verify(self)
        if calls == 2:
            self.result_sha256 = "f" * 64

    with _open_native_session(
        dataset.examples,
        store_path,
        compiled_native_prepared_benchmark,
    ) as (receipt, native_result, binary_sha256):
        pinned_result_sha256 = native_result.result_sha256
        monkeypatch.setattr(
            NativePreparedSessionResult,
            "verify_integrity",
            verify_then_replace_pin,
        )
        with pytest.raises(ValueError, match="credentials changed"):
            evaluate_native_prepared_per_query_benchmark(
                dataset.examples,
                dataset.tier_specs,
                store_path,
                receipt,
                native_result,
                expected_binary_sha256=binary_sha256,
                expected_result_sha256=pinned_result_sha256,
                embedding_identity=None,
            )
        assert calls == 2


def test_bridge_uses_at_only_closes_store_before_replay_and_returns_no_score_payload(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    store_path = tmp_path / "two-phase-at-only.trpstore"
    authenticated_stores: list[AuthenticatedPreparedStore] = []
    original_authenticate = native_benchmark_module.authenticate_prepared_store_file
    original_nested = native_benchmark_module.nested_lodo_lambda_evaluation
    original_baselines = native_benchmark_module.evaluate_per_query_lodo_baselines
    calls = {"learned": 0, "baselines": 0}

    def capture_store(*args: object, **kwargs: object) -> AuthenticatedPreparedStore:
        store = original_authenticate(*args, **kwargs)
        authenticated_stores.append(store)
        return store

    def learned_after_close(*args: object, **kwargs: object) -> object:
        assert len(authenticated_stores) == 1 and authenticated_stores[0].closed
        calls["learned"] += 1
        return original_nested(*args, **kwargs)

    def baselines_after_close(*args: object, **kwargs: object) -> object:
        assert len(authenticated_stores) == 1 and authenticated_stores[0].closed
        calls["baselines"] += 1
        return original_baselines(*args, **kwargs)

    def forbidden_view_access(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("native bridge must consume float views only through at()")

    monkeypatch.setattr(native_benchmark_module, "authenticate_prepared_store_file", capture_store)
    monkeypatch.setattr(
        native_benchmark_module,
        "nested_lodo_lambda_evaluation",
        learned_after_close,
    )
    monkeypatch.setattr(
        native_benchmark_module,
        "evaluate_per_query_lodo_baselines",
        baselines_after_close,
    )
    monkeypatch.setattr(
        prepared_execution_module,
        "PreparedRawScoreBundle",
        forbidden_view_access,
    )

    with _open_native_session(
        dataset.examples,
        store_path,
        compiled_native_prepared_benchmark,
    ) as (receipt, native_result, binary_sha256):
        with pytest.raises(TypeError, match="exact integers"):
            native_result.scores[0].scores.at(True, 0)
        monkeypatch.setattr(NativePreparedFloat64View, "__getitem__", forbidden_view_access)
        monkeypatch.setattr(NativePreparedFloat64View, "__iter__", forbidden_view_access)
        monkeypatch.setattr(NativePreparedFloat64View, "__len__", forbidden_view_access)
        result = evaluate_native_prepared_per_query_benchmark(
            dataset.examples,
            dataset.tier_specs,
            store_path,
            receipt,
            native_result,
            expected_binary_sha256=binary_sha256,
            expected_result_sha256=native_result.result_sha256,
            embedding_identity=None,
        )
        assert not native_result.closed
    assert calls == {"learned": 1, "baselines": 1}
    assert result.learned.report.router_name == "nested-lodo-tier-lambda"
    graph = _object_graph(result)
    forbidden_types = (mmap.mmap, memoryview, NativePreparedFloat64View)
    assert not any(isinstance(value, forbidden_types) for value in graph)
    assert not any(hasattr(value, "scores_payload") for value in graph)


def test_policy_failure_remains_primary_when_owned_store_cleanup_also_fails(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    store_path = tmp_path / "primary-cleanup-error.trpstore"
    captured: list[AuthenticatedPreparedStore] = []
    original_authenticate = native_benchmark_module.authenticate_prepared_store_file
    original_close = AuthenticatedPreparedStore.close

    def capture_store(*args: object, **kwargs: object) -> AuthenticatedPreparedStore:
        store = original_authenticate(*args, **kwargs)
        captured.append(store)
        return store

    monkeypatch.setattr(native_benchmark_module, "authenticate_prepared_store_file", capture_store)
    monkeypatch.setattr(
        native_benchmark_module,
        "_build_native_policy_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("policy primary")),
    )
    monkeypatch.setattr(
        AuthenticatedPreparedStore,
        "close",
        lambda self: (_ for _ in ()).throw(OSError("cleanup secondary")),
    )
    try:
        with _open_native_session(
            dataset.examples,
            store_path,
            compiled_native_prepared_benchmark,
        ) as (receipt, native_result, binary_sha256):
            with pytest.raises(RuntimeError, match="policy primary"):
                evaluate_native_prepared_per_query_benchmark(
                    dataset.examples,
                    dataset.tier_specs,
                    store_path,
                    receipt,
                    native_result,
                    expected_binary_sha256=binary_sha256,
                    expected_result_sha256=native_result.result_sha256,
                    embedding_identity=None,
                )
    finally:
        monkeypatch.setattr(AuthenticatedPreparedStore, "close", original_close)
        for store in captured:
            store.close()


def test_durable_result_rejects_an_unpinned_execution_receipt(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset()
    result, _ = _run_native_benchmark(
        dataset.examples,
        dataset.tier_specs,
        tmp_path / "durable-result-pin.trpstore",
        compiled_native_prepared_benchmark,
    )

    with pytest.raises(ValueError, match="caller-pinned result"):
        replace(
            result,
            execution_receipt=replace(
                result.execution_receipt,
                result_sha256_caller_pinned=False,
            ),
        )


def test_calibration_evidence_rejects_more_points_than_admitted_rows() -> None:
    with pytest.raises(ValueError, match="point counts"):
        NativePreparedCalibrationEvidence(
            training_subset_index=0,
            training_domain_indices=(0,),
            model_ids=("model",),
            calibration_example_count=1,
            raw_score_block_indices=(0,),
            raw_score_block_sha256s=("0" * 64,),
            target_shard_sha256s=("1" * 64,),
            calibrators=(
                IsotonicCalibrator(
                    upper_bounds=(0.0, 1.0),
                    values=(0.0, 1.0),
                ),
            ),
        )


def test_authenticated_store_targets_are_compared_bit_exactly_to_evaluation_rows(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset()
    examples = dataset.examples
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=12,
        target_count=len(examples[0].candidate_models),
    )
    source_sha256 = prepared_fit_source_sha256(examples, plan)
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=source_sha256,
    )
    feature_stride = 8 * plan.feature_count
    target_stride = 8 * plan.target_count

    def target_rows() -> Iterator[tuple[float, ...]]:
        for row_index in range(plan.work.example_count):
            row = list(
                struct.unpack_from(
                    f"<{plan.target_count}d",
                    store.target_payload,
                    row_index * target_stride,
                )
            )
            if row_index == 0:
                row[0] = math.nextafter(row[0], math.inf)
            yield tuple(row)

    store_path = tmp_path / "forged-self-declared-target.trpstore"
    receipt = write_prepared_store_file_from_sections(
        destination=store_path,
        plan=plan,
        model_ids=store.model_ids,
        example_ids=iter(store.example_ids),
        prompt_sha256s=iter(store.prompt_sha256s),
        domain_indices=iter(store.domain_indices),
        feature_rows=(
            struct.unpack_from(
                f"<{plan.feature_count}d",
                store.feature_payload,
                row_index * feature_stride,
            )
            for row_index in range(plan.work.example_count)
        ),
        target_rows=target_rows(),
        embedding_identity=None,
        embedding_snapshot_sha256=None,
        expected_source_fit_sha256=source_sha256,
        logical_store_sha256=store.sha256,
    )
    executable, binary_sha256 = compiled_native_prepared_benchmark
    with NativePreparedSessionAdapter(
        executable,
        binary_sha256,
        timeout_seconds=120,
    ).run(store_path, receipt, ridge=1.0) as native_result:
        with pytest.raises(ValueError, match="targets do not match evaluation data"):
            evaluate_native_prepared_per_query_benchmark(
                examples,
                dataset.tier_specs,
                store_path,
                receipt,
                native_result,
                expected_binary_sha256=binary_sha256,
                expected_result_sha256=native_result.result_sha256,
                embedding_identity=None,
            )


def test_observable_tag_bound_fails_before_lambda_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_evaluation_dataset()
    examples = tuple(
        replace(example, router_metadata={"domain": f"observable-{index}"})
        for index, example in enumerate(dataset.examples)
    )
    domains = tuple(sorted({example.domain for example in examples}))
    counts = tuple(sum(example.domain == domain for example in examples) for domain in domains)
    plan = build_prepared_nested_lodo_plan(
        domains,
        counts,
        feature_count=12,
        target_count=len(examples[0].candidate_models),
    )
    store = build_prepared_feature_store(
        examples,
        plan,
        expected_source_fit_sha256=prepared_fit_source_sha256(examples, plan),
    )
    store_path = tmp_path / "observable-tag-bound.trpstore"
    receipt = write_prepared_store_file(store, store_path)
    with authenticate_prepared_store_file(store_path, receipt) as authenticated:
        metadata = authenticated.metadata

    monkeypatch.setattr(
        native_benchmark_module,
        "MAX_NATIVE_PREPARED_BASELINE_TAG_COMPARISONS",
        0,
    )
    monkeypatch.setattr(
        native_benchmark_module,
        "estimate_lambda_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("lambda preflight must not run after tag-bound rejection")
        ),
    )
    with pytest.raises(ValueError, match="observable-tag comparisons"):
        preflight_native_prepared_benchmark(
            examples,
            dataset.tier_specs,
            plan,
            metadata,
            receipt,
            None,
        )


@pytest.mark.skipif(not hasattr(os, "pwrite"), reason="os.pwrite is unavailable")
def test_persistent_native_result_mutation_fails_the_external_pin(
    compiled_native_prepared_benchmark: tuple[Path, str],
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset()
    store_path = tmp_path / "persistent-result-mutation.trpstore"
    with _open_native_session(
        dataset.examples,
        store_path,
        compiled_native_prepared_benchmark,
    ) as (receipt, native_result, binary_sha256):
        pinned_result_sha256 = native_result.result_sha256
        lifetime = native_result._lifetime
        mapping = lifetime.require_open()
        offset = len(mapping) - 1
        changed = bytes((mapping[offset] ^ 1,))
        assert os.pwrite(lifetime._descriptor, changed, offset) == 1
        with pytest.raises(NativePreparedIntegrityError, match="mapping SHA-256 changed"):
            evaluate_native_prepared_per_query_benchmark(
                dataset.examples,
                dataset.tier_specs,
                store_path,
                receipt,
                native_result,
                expected_binary_sha256=binary_sha256,
                expected_result_sha256=pinned_result_sha256,
                embedding_identity=None,
            )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_candidates_per_tier": 1},
        {"max_candidates_per_tier": 258},
        {"max_candidates_per_tier": True},
        {"random_seed": 2**63},
        {"random_seed": -(2**63) - 1},
        {"random_seed": True},
        {"character_threshold": 0},
        {"character_threshold": 2**63},
        {"character_threshold": True},
    ),
)
def test_native_benchmark_config_rejects_unreviewed_controls(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        NativePreparedBenchmarkConfig(**kwargs)
