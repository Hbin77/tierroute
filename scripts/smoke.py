# SPDX-License-Identifier: Apache-2.0
"""Exercise every installed CLI path without network access or external data."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from tierroute.adapters import load_evaluation_dataset
from tierroute.demo import model_catalogue
from tierroute.features import PromptFeatureSchema
from tierroute.predictors import BilinearPredictorArtifact, IsotonicCalibrator

REQUIRED_BASELINES = {
    "always-cheapest",
    "always-premium",
    "random",
    "length-heuristic",
    "oracle",
    "domain-best-table",
}


def _require_empty_offline_home() -> Path:
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable}=1 is required for the smoke test")

    raw_home = os.environ.get("HF_HOME")
    if not raw_home:
        raise RuntimeError("HF_HOME must point to a dedicated empty directory")
    hf_home = Path(raw_home)
    if not hf_home.is_dir():
        raise RuntimeError(f"HF_HOME is not a directory: {hf_home}")
    if any(hf_home.iterdir()):
        raise RuntimeError(f"HF_HOME is not empty: {hf_home}")
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
    """Run core route, artifact route, evaluate, and demo through the console script."""

    hf_home = _require_empty_offline_home()
    executable = shutil.which("tierroute")
    if executable is None:
        raise RuntimeError("installed `tierroute` console script was not found on PATH")

    route = json.loads(
        _run_cli(
            executable,
            "route",
            "Prove that sqrt(2) is irrational.",
            "--tier",
            "fast",
            "--json",
        )
    )
    if route.get("tier") != "fast" or route.get("network_used") is not False:
        raise RuntimeError("route smoke output did not confirm fast-tier offline routing")
    for key in ("model", "cost", "predicted_quality"):
        if key not in route:
            raise RuntimeError(f"route smoke output is missing {key!r}")

    dataset = load_evaluation_dataset()
    schema = PromptFeatureSchema.fit(tuple(example.prompt for example in dataset.examples))
    model_ids = tuple(sorted(model.model_id for model in model_catalogue(dataset)))
    smoke_scores = {
        model_id: (index + 1) / len(model_ids) for index, model_id in enumerate(model_ids)
    }
    smoke_artifact = BilinearPredictorArtifact(
        feature_schema=schema,
        model_weights={model_id: (0.0,) * schema.dimension for model_id in model_ids},
        model_bias=smoke_scores,
        calibrators={
            model_id: IsotonicCalibrator((0.0,), (score,))
            for model_id, score in smoke_scores.items()
        },
        training_data_sha256="0" * 64,
        training_example_count=len(dataset.examples),
        training_domains=tuple(sorted({example.domain for example in dataset.examples})),
        ridge=1.0,
        seed=0,
    )
    with tempfile.TemporaryDirectory(prefix="tierroute-core-artifact-") as temporary:
        artifact_path = smoke_artifact.save(Path(temporary) / "predictor.json")
        artifact_route = json.loads(
            _run_cli(
                executable,
                "route",
                "Route with a local artifact.",
                "--tier",
                "balanced",
                "--artifact",
                str(artifact_path),
                "--json",
            )
        )
    if artifact_route.get("quality_kind") != "calibrated bilinear artifact":
        raise RuntimeError("core artifact inference smoke did not load the JSON predictor")
    if artifact_route.get("network_used") is not False:
        raise RuntimeError("core artifact inference smoke did not confirm offline routing")

    evaluation = json.loads(_run_cli(executable, "evaluate", "--json"))
    baseline_names = {row.get("name") for row in evaluation.get("baselines", [])}
    if baseline_names != REQUIRED_BASELINES:
        raise RuntimeError(f"evaluate smoke returned unexpected baselines: {baseline_names}")

    demo = _run_cli(executable, "demo")
    if "tierroute offline quickstart" not in demo or "domain-best-table" not in demo:
        raise RuntimeError("demo smoke output is incomplete")

    if any(hf_home.iterdir()):
        raise RuntimeError("the CLI wrote to HF_HOME during an offline smoke test")

    print("CLI smoke passed: core and artifact route, evaluate, and demo ran offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
