# SPDX-License-Identifier: Apache-2.0
"""Package-level smoke tests."""

import importlib.metadata

import tierroute


def test_package_has_version() -> None:
    assert tierroute.__version__ == "0.1.0"


def test_routerbench_reader_adds_no_distribution_requirement() -> None:
    requirements = importlib.metadata.requires("tierroute") or ()
    normalized = tuple(requirement.lower() for requirement in requirements)

    assert all("routerbench" not in requirement for requirement in normalized)
    assert all(not requirement.startswith("pandas") for requirement in normalized)
