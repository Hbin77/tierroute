# SPDX-License-Identifier: Apache-2.0
"""Exercise offline training, nested benchmarking, and artifact routing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _require_empty_offline_home() -> Path:
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable}=1 is required for the training smoke test")

    raw_home = os.environ.get("HF_HOME")
    if not raw_home:
        raise RuntimeError("HF_HOME must point to a dedicated empty directory")
    hf_home = Path(raw_home)
    if not hf_home.is_dir() or any(hf_home.iterdir()):
        raise RuntimeError(f"HF_HOME must be an existing empty directory: {hf_home}")
    return hf_home


def _run_cli(executable: str, *arguments: str) -> str:
    completed = subprocess.run(
        [executable, *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"tierroute {' '.join(arguments)} failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    return completed.stdout


def main() -> int:
    """Train and consume one artifact without external data or network access."""

    hf_home = _require_empty_offline_home()
    executable = shutil.which("tierroute")
    if executable is None:
        raise RuntimeError("installed `tierroute` console script was not found on PATH")

    with tempfile.TemporaryDirectory(prefix="tierroute-training-smoke-") as temporary:
        artifact = Path(temporary) / "predictor.json"
        policy = Path(temporary) / "policy.json"
        training = json.loads(
            _run_cli(
                executable,
                "train",
                "--output",
                str(artifact),
                "--policy-output",
                str(policy),
                "--budget-scope",
                "per-query",
                "--json",
            )
        )
        if (
            training.get("network_used") is not False
            or not artifact.is_file()
            or not policy.is_file()
        ):
            raise RuntimeError(
                "training smoke did not produce offline predictor and policy artifacts"
            )
        if training.get("model_ids") != ["expert", "steady", "swift"]:
            raise RuntimeError("training smoke returned an unexpected model catalogue")
        if training.get("training_examples") != 8:
            raise RuntimeError("training smoke returned an unexpected example count")
        if training.get("solver_id") != "tierroute.centered-ridge-cholesky-python-v1":
            raise RuntimeError("training smoke returned an unexpected ridge solver ID")
        if training.get("accounting_scope") != "per-query":
            raise RuntimeError("training smoke returned the wrong policy accounting scope")
        if training.get("feasible") is not True or training.get("weighted_training_score") is None:
            raise RuntimeError("training smoke did not report a feasible weighted policy score")
        if set(training.get("lambda_search", {})) != {"fast", "balanced", "premium"}:
            raise RuntimeError("training smoke did not report every tier's lambda search")

        benchmark = json.loads(
            _run_cli(
                executable,
                "benchmark",
                "--budget-scope",
                "per-query",
                "--json",
            )
        )
        if (
            benchmark.get("schema") != "tierroute-benchmark"
            or benchmark.get("schema_version") != 1
            or benchmark.get("network_used") is not False
            or benchmark.get("budget_scope") != "per-query"
            or benchmark.get("validation_scope") != "true-nested-lodo-original-order"
        ):
            raise RuntimeError("nested benchmark did not preserve its offline report contract")
        learned = benchmark.get("learned_router", {})
        if (
            learned.get("feasible") is not True
            or learned.get("weighted_quality") is None
            or learned.get("oracle_gap_recovery") is None
            or len(learned.get("prediction_sha256", "")) != 64
        ):
            raise RuntimeError("nested benchmark did not return complete learned-router evidence")
        baseline_names = [row.get("name") for row in benchmark.get("baselines", [])]
        if baseline_names != [
            "always-cheapest",
            "always-premium",
            "random",
            "length-heuristic",
            "oracle",
            "domain-best-table",
        ]:
            raise RuntimeError("nested benchmark did not return all six baselines")
        if any(
            row.get("evaluation_scope") != benchmark.get("evaluation_scope")
            for row in [learned, *benchmark["baselines"]]
        ):
            raise RuntimeError("learned and baseline benchmark scopes do not match")
        folds = learned.get("outer_folds", [])
        if len(folds) != 4 or any(
            fold.get("membership", {}).get("algorithm") != "tierroute-fold-membership-sha256-v1"
            or len(fold.get("membership", {}).get("sha256", "")) != 64
            for fold in folds
        ):
            raise RuntimeError("nested benchmark fold membership evidence is incomplete")

        predictor_route = json.loads(
            _run_cli(
                executable,
                "route",
                "Prove that sqrt(2) is irrational.",
                "--tier",
                "balanced",
                "--artifact",
                str(artifact),
                "--json",
            )
        )
        if predictor_route.get("network_used") is not False:
            raise RuntimeError("artifact route did not confirm offline operation")
        if predictor_route.get("quality_kind") != "calibrated bilinear artifact":
            raise RuntimeError("artifact route did not use the trained predictor")
        if predictor_route.get("model") not in training["model_ids"]:
            raise RuntimeError("artifact route selected a model outside the trained catalogue")

        policy_route = json.loads(
            _run_cli(
                executable,
                "route",
                "Prove that sqrt(2) is irrational.",
                "--tier",
                "balanced",
                "--artifact",
                str(artifact),
                "--policy-artifact",
                str(policy),
                "--json",
            )
        )
        if policy_route.get("network_used") is not False:
            raise RuntimeError("policy artifact route did not confirm offline operation")
        if policy_route.get("quality_kind") != (
            "calibrated bilinear + tuned exact-rational tier lambda"
        ):
            raise RuntimeError("policy artifact route did not use the tuned exact lambda")
        if policy_route.get("accounting_scope") != "per-query":
            raise RuntimeError("policy artifact route returned the wrong accounting scope")
        if policy_route.get("lambda_cost") != training["lambda_by_tier"]["balanced"]:
            raise RuntimeError("policy artifact route did not preserve the exact tuned lambda")
        if policy_route.get("lambda_search") != training["lambda_search"]["balanced"]:
            raise RuntimeError("policy artifact route did not preserve candidate-search evidence")

    if any(hf_home.iterdir()):
        raise RuntimeError("training or artifact routing wrote to HF_HOME")

    print(
        "Training smoke passed: predictor fit, nested benchmark, exact policy tuning, "
        "and both routes ran offline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
