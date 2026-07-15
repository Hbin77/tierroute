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
