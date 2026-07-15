# SPDX-License-Identifier: Apache-2.0
"""Tests for the bundled, project-authored quickstart dataset."""

from decimal import Decimal

from tierroute.adapters import bundled_synthetic_path, load_evaluation_dataset
from tierroute.core import BudgetTier


def test_bundled_dataset_is_complete_and_explicitly_synthetic() -> None:
    dataset = load_evaluation_dataset()

    assert bundled_synthetic_path().is_file()
    assert dataset.license == "Apache-2.0"
    assert "not benchmark evidence" in dataset.provenance
    assert dataset.domain_labels_are_observable is True
    assert len(dataset.examples) == 8
    assert {example.domain for example in dataset.examples} == {
        "general",
        "code",
        "math",
        "science",
    }
    assert [spec.tier for spec in dataset.tier_specs] == list(BudgetTier)
    assert dataset.examples[0].router_metadata["domain"] == dataset.examples[0].domain
    assert "example_id" not in dataset.examples[0].router_metadata


def test_bundled_costs_are_exact_and_model_catalogue_is_stable() -> None:
    dataset = load_evaluation_dataset()

    for example in dataset.examples:
        assert [outcome.model_id for outcome in example.outcomes] == [
            "swift",
            "steady",
            "expert",
        ]
        assert [outcome.cost for outcome in example.outcomes] == [
            Decimal("0.20"),
            Decimal("0.60"),
            Decimal("1.00"),
        ]
