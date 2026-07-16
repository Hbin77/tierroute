# SPDX-License-Identifier: Apache-2.0
"""Package-level smoke tests."""

import importlib.metadata
from pathlib import Path

import tierroute

_ROOT = Path(__file__).resolve().parents[1]


def test_package_has_version() -> None:
    assert tierroute.__version__ == "0.1.0"


def test_runtime_and_training_add_no_distribution_requirement() -> None:
    requirements = importlib.metadata.requires("tierroute") or ()
    normalized = tuple(requirement.lower() for requirement in requirements)

    assert all("routerbench" not in requirement for requirement in normalized)
    assert all(not requirement.startswith("pandas") for requirement in normalized)
    assert all(not requirement.startswith("numpy") for requirement in normalized)
    assert all('extra == "training"' not in requirement for requirement in normalized)


def test_release_gate_blocks_prepared_protocol_payloads_by_suffix_and_magic() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ignore = (_ROOT / ".gitignore").read_text(encoding="utf-8")

    for suffix in ("*.trprequest", "*.trpresult", "*.trpsto", "*.trpstore"):
        assert suffix in ignore
        assert suffix.removeprefix("*") in workflow
    for magic in ("TRPRES01", "TRPSES01", "TRPSTO01"):
        assert f'b"{magic}"' in workflow
    assert "sdist_magic_payloads" in workflow
