# SPDX-License-Identifier: Apache-2.0
"""Exercise every installed CLI path without network access or external data."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

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
    """Run route, evaluate, and demo through the installed console script."""

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

    evaluation = json.loads(_run_cli(executable, "evaluate", "--json"))
    baseline_names = {row.get("name") for row in evaluation.get("baselines", [])}
    if baseline_names != REQUIRED_BASELINES:
        raise RuntimeError(f"evaluate smoke returned unexpected baselines: {baseline_names}")

    demo = _run_cli(executable, "demo")
    if "tierroute offline quickstart" not in demo or "domain-best-table" not in demo:
        raise RuntimeError("demo smoke output is incomplete")

    if any(hf_home.iterdir()):
        raise RuntimeError("the CLI wrote to HF_HOME during an offline smoke test")

    print("CLI smoke passed: route, evaluate, and demo ran offline on bundled data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
