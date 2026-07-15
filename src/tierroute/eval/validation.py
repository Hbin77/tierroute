# SPDX-License-Identifier: Apache-2.0
"""Domain-shift validation utilities; random splitting is intentionally absent."""

from __future__ import annotations

from dataclasses import dataclass

from tierroute.eval.schemas import EvaluationExample


@dataclass(frozen=True, slots=True)
class DomainFold:
    """One leave-one-domain-out train/test partition."""

    held_out_domain: str
    training: tuple[EvaluationExample, ...]
    test: tuple[EvaluationExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.held_out_domain, str) or not self.held_out_domain.strip():
            raise ValueError("held_out_domain must be a non-empty string")
        if not self.training or not self.test:
            raise ValueError("a domain fold requires non-empty training and test partitions")
        if any(example.domain == self.held_out_domain for example in self.training):
            raise ValueError("the held-out domain must not appear in fold training data")
        if any(example.domain != self.held_out_domain for example in self.test):
            raise ValueError("every fold test example must belong to held_out_domain")
        training_ids = {example.example_id for example in self.training}
        test_ids = {example.example_id for example in self.test}
        if len(training_ids) != len(self.training) or len(test_ids) != len(self.test):
            raise ValueError("a domain fold requires unique example IDs")
        if training_ids & test_ids:
            raise ValueError("domain fold training and test example IDs must be disjoint")


def leave_one_domain_out(examples: tuple[EvaluationExample, ...]) -> tuple[DomainFold, ...]:
    """Create deterministic LODO folds sorted by held-out domain."""

    examples = tuple(examples)
    example_ids = [example.example_id for example in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("LODO requires unique example_id values")
    domains = sorted({example.domain for example in examples})
    if len(domains) < 2:
        raise ValueError("LODO requires at least two domains")
    return tuple(
        DomainFold(
            held_out_domain=domain,
            training=tuple(example for example in examples if example.domain != domain),
            test=tuple(example for example in examples if example.domain == domain),
        )
        for domain in domains
    )
