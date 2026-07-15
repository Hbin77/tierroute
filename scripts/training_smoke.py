# SPDX-License-Identifier: Apache-2.0
"""Exercise offline training, canonical artifact loading, and artifact routing."""

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
        training = json.loads(_run_cli(executable, "train", "--output", str(artifact), "--json"))
        if training.get("network_used") is not False or not artifact.is_file():
            raise RuntimeError("training smoke did not produce an offline artifact")
        if training.get("model_ids") != ["expert", "steady", "swift"]:
            raise RuntimeError("training smoke returned an unexpected model catalogue")
        if training.get("training_examples") != 8:
            raise RuntimeError("training smoke returned an unexpected example count")

        route = json.loads(
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
        if route.get("network_used") is not False:
            raise RuntimeError("artifact route did not confirm offline operation")
        if route.get("quality_kind") != "calibrated bilinear artifact":
            raise RuntimeError("artifact route did not use the trained predictor")
        if route.get("model") not in training["model_ids"]:
            raise RuntimeError("artifact route selected a model outside the trained catalogue")

    if any(hf_home.iterdir()):
        raise RuntimeError("training or artifact routing wrote to HF_HOME")

    print("Training smoke passed: fit, JSON artifact load, and route ran fully offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
