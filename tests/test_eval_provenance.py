# SPDX-License-Identifier: Apache-2.0
"""Tests for canonical replay-data identity and order-sensitive hashes."""

from dataclasses import replace
from decimal import Decimal
from types import MappingProxyType

import pytest

import tierroute.eval.provenance as provenance_module
from tierroute.adapters import load_evaluation_dataset
from tierroute.core import BudgetTier, ModelSpec
from tierroute.eval import (
    CandidateOutcome,
    EvaluationExample,
    TierSpec,
    evaluation_data_sha256,
    evaluation_replay_sha256,
    evaluation_scope_sha256,
)
from tierroute.predictors import training_data_sha256


def _scope(examples: tuple[object, ...], *, max_calls_per_query: int = 1) -> str:
    dataset = load_evaluation_dataset()
    return evaluation_scope_sha256(
        examples,  # type: ignore[arg-type]
        dataset.tier_specs,
        max_calls_per_query=max_calls_per_query,
    )


def test_data_identity_is_order_independent_but_replay_hash_is_not() -> None:
    examples = load_evaluation_dataset().examples
    reversed_examples = tuple(reversed(examples))

    assert evaluation_data_sha256(examples) == evaluation_data_sha256(reversed_examples)
    assert training_data_sha256(examples) == evaluation_data_sha256(examples)
    assert evaluation_replay_sha256(examples) != evaluation_replay_sha256(reversed_examples)


def test_bundled_provenance_algorithms_have_pinned_versioned_bytes() -> None:
    dataset = load_evaluation_dataset()

    assert evaluation_data_sha256(dataset.examples) == (
        "999d435a40f2db8c76aa205fa3e565b416ab53f6e402979a794a029277b71d60"
    )
    assert evaluation_replay_sha256(dataset.examples) == (
        "24be663ca438f388fa3086f638b39c64096a007cd1752820cb4c2ceb1daaa296"
    )
    assert (
        evaluation_scope_sha256(
            dataset.examples,
            dataset.tier_specs,
            max_calls_per_query=1,
        )
        == "fde4ac2af181ca623238807f33124ab74b38027184e7f5051b61b056276c5aa2"
    )


def test_data_identity_covers_replay_content() -> None:
    examples = load_evaluation_dataset().examples
    changed = (replace(examples[0], prompt=f"{examples[0].prompt} changed"), *examples[1:])

    assert evaluation_data_sha256(changed) != evaluation_data_sha256(examples)
    assert evaluation_replay_sha256(changed) != evaluation_replay_sha256(examples)


def test_evaluation_scope_covers_every_replay_and_policy_input() -> None:
    examples = load_evaluation_dataset().examples
    first = examples[0]
    first_outcome = first.outcomes[0]
    first_model = first.candidate_models[0]
    changes = (
        replace(first, domain=f"{first.domain}-changed"),
        replace(first, router_metadata={"domain": "changed"}),
        replace(
            first,
            outcomes=(replace(first_outcome, output="changed output"), *first.outcomes[1:]),
        ),
        replace(
            first,
            outcomes=(
                replace(first_outcome, quality=first_outcome.quality + 0.01),
                *first.outcomes[1:],
            ),
        ),
        replace(
            first,
            outcomes=(
                replace(first_outcome, cost=first_outcome.cost + Decimal("0.01")),
                *first.outcomes[1:],
            ),
        ),
        replace(first, outcomes=tuple(reversed(first.outcomes))),
        replace(
            first,
            candidate_models=(
                replace(first_model, cost=first_model.cost + Decimal("0.01")),
                *first.candidate_models[1:],
            ),
        ),
        replace(
            first,
            candidate_models=(
                replace(first_model, display_name="changed display name"),
                *first.candidate_models[1:],
            ),
        ),
        replace(
            first,
            candidate_models=(
                replace(first_model, metadata={"provider": "changed"}),
                *first.candidate_models[1:],
            ),
        ),
        replace(first, candidate_models=tuple(reversed(first.candidate_models))),
    )

    original_scope = _scope(examples)
    for changed_first in changes:
        assert _scope((changed_first, *examples[1:])) != original_scope


def test_evaluation_scope_covers_tier_order_weight_budget_and_call_cap() -> None:
    dataset = load_evaluation_dataset()
    specs = dataset.tier_specs
    original = evaluation_scope_sha256(
        dataset.examples,
        specs,
        max_calls_per_query=1,
    )

    assert (
        evaluation_scope_sha256(
            dataset.examples,
            tuple(reversed(specs)),
            max_calls_per_query=1,
        )
        != original
    )
    assert (
        evaluation_scope_sha256(
            dataset.examples,
            (replace(specs[0], weight=specs[0].weight + 0.01), *specs[1:]),
            max_calls_per_query=1,
        )
        != original
    )
    assert (
        evaluation_scope_sha256(
            dataset.examples,
            (replace(specs[0], budget_limit=specs[0].budget_limit + Decimal("0.01")), *specs[1:]),
            max_calls_per_query=1,
        )
        != original
    )
    assert evaluation_scope_sha256(dataset.examples, specs, max_calls_per_query=2) != original


def test_scope_normalizes_equivalent_tier_and_quality_representations() -> None:
    dataset = load_evaluation_dataset()
    example = dataset.examples[0]
    integer_quality = replace(
        example,
        outcomes=(replace(example.outcomes[0], quality=1), *example.outcomes[1:]),
    )
    float_quality = replace(
        example,
        outcomes=(replace(example.outcomes[0], quality=1.0), *example.outcomes[1:]),
    )
    integer_spec = TierSpec(BudgetTier.FAST, Decimal("1.00"), 1)
    float_spec = TierSpec(BudgetTier.FAST, Decimal("1"), 1.0)

    assert evaluation_scope_sha256(
        (integer_quality,),
        (integer_spec,),
        max_calls_per_query=1,
    ) == evaluation_scope_sha256(
        (float_quality,),
        (float_spec,),
        max_calls_per_query=1,
    )


def test_integer_quality_keeps_legacy_data_and_replay_hash_bytes() -> None:
    examples = (
        EvaluationExample(
            "integer-quality",
            "prompt",
            "domain",
            (CandidateOutcome("m", "o", Decimal("1.00"), 1),),
            (ModelSpec("m", Decimal("1.0")),),
        ),
    )

    assert type(examples[0].outcomes[0].quality) is int
    assert evaluation_data_sha256(examples) == (
        "146aa99988078048a5c7a4d0fdd5a9d15eba39fd56f1d8a68980cdd6436fbc5b"
    )
    assert evaluation_replay_sha256(examples) == evaluation_data_sha256(examples)


def test_scope_metadata_is_value_canonical_and_fails_closed() -> None:
    dataset = load_evaluation_dataset()
    first = dataset.examples[0]
    left = replace(
        first,
        router_metadata={"nested": {"b": [1, Decimal("1.00")], "a": -0.0}},
    )
    right = replace(
        first,
        router_metadata={"nested": {"a": -0.0, "b": (1, Decimal("1"))}},
    )

    assert _scope((left, *dataset.examples[1:])) == _scope((right, *dataset.examples[1:]))

    unsupported = replace(first, router_metadata={"bad": object()})
    with pytest.raises(TypeError, match="unsupported type"):
        _scope((unsupported,))
    non_string_key = replace(first, router_metadata={1: "bad"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="keys must be plain strings"):
        _scope((non_string_key,))
    non_finite = replace(first, router_metadata={"bad": float("nan")})
    with pytest.raises(ValueError, match="finite float"):
        _scope((non_finite,))
    cyclic: list[object] = []
    cyclic.append(cyclic)
    cycle = replace(first, router_metadata={"cycle": cyclic})
    with pytest.raises(ValueError, match="cyclic metadata"):
        _scope((cycle,))


def test_scope_rejects_custom_mappings_without_dispatching_their_hooks() -> None:
    first = load_evaluation_dataset().examples[0]

    class HookedDict(dict[str, object]):
        hook_calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.hook_calls += 1
            raise AssertionError("custom iterator must not run")

        def __getitem__(self, key: str) -> object:
            self.hook_calls += 1
            raise AssertionError("custom lookup must not run")

    direct = HookedDict({"domain": "math"})
    with pytest.raises(TypeError, match="unsupported type HookedDict"):
        _scope((replace(first, router_metadata=direct),))
    assert direct.hook_calls == 0

    wrapped = HookedDict({"domain": "math"})
    proxy = MappingProxyType(wrapped)
    with pytest.raises(TypeError, match="unsupported type mappingproxy"):
        _scope((replace(first, router_metadata=proxy),))
    assert wrapped.hook_calls == 0


def test_scope_metadata_resource_limits_fail_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = load_evaluation_dataset().examples[0]

    monkeypatch.setattr(provenance_module, "_MAX_METADATA_NODES", 2)
    too_many = replace(first, router_metadata={"items": [1, 2]})
    with pytest.raises(ValueError, match="canonical values"):
        _scope((too_many,))

    monkeypatch.setattr(provenance_module, "_MAX_METADATA_NODES", 100)
    monkeypatch.setattr(provenance_module, "_MAX_METADATA_ENCODED_BYTES", 4)
    too_much_text = replace(first, router_metadata={"key": "value"})
    with pytest.raises(ValueError, match="encoded-payload limit"):
        _scope((too_much_text,))

    monkeypatch.setattr(provenance_module, "_MAX_METADATA_ENCODED_BYTES", 1024)
    monkeypatch.setattr(provenance_module, "_MAX_METADATA_INTEGER_BITS", 4)
    too_wide = replace(first, router_metadata={"integer": 16})
    with pytest.raises(ValueError, match="bit metadata integer limit"):
        _scope((too_wide,))

    monkeypatch.setattr(provenance_module, "_MAX_METADATA_INTEGER_BITS", 1_000_000)
    monkeypatch.setattr(provenance_module, "_MAX_EVALUATION_METADATA_NODES", 1)
    with pytest.raises(ValueError, match="node snapshot limit"):
        _scope((first,))


@pytest.mark.parametrize(
    "repeated_value",
    [2**20, Decimal("1234567890")],
)
def test_scope_charges_repeated_numeric_payload_per_occurrence(
    monkeypatch: pytest.MonkeyPatch,
    repeated_value: object,
) -> None:
    first = load_evaluation_dataset().examples[0]
    monkeypatch.setattr(provenance_module, "_MAX_METADATA_ENCODED_BYTES", 15)
    repeated = replace(first, router_metadata={"v": [repeated_value, repeated_value]})

    with pytest.raises(ValueError, match="encoded-payload limit"):
        _scope((repeated,))


def test_scope_charges_shared_prompt_and_output_per_logical_occurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_output = "x" * 40
    example = EvaluationExample(
        "alias",
        "p",
        "d",
        (
            CandidateOutcome("a", shared_output, Decimal("1"), 0.5),
            CandidateOutcome("b", shared_output, Decimal("1"), 0.5),
        ),
        (ModelSpec("a", Decimal("1")), ModelSpec("b", Decimal("1"))),
    )
    spec = TierSpec(BudgetTier.FAST, Decimal("1"), 1.0)
    monkeypatch.setattr(provenance_module, "_MAX_EVALUATION_SCOPE_ENCODED_BYTES", 100)

    with pytest.raises(ValueError, match=r"evaluation scope.*encoded-payload limit"):
        evaluation_scope_sha256((example,), (spec,), max_calls_per_query=1)


def test_scope_rejects_numeric_subclass_hooks_without_dispatching_them() -> None:
    dataset = load_evaluation_dataset()
    first = dataset.examples[0]

    class HookedDecimal(Decimal):
        hook_calls = 0

        def __format__(self, spec: str) -> str:
            type(self).hook_calls += 1
            raise AssertionError("custom Decimal formatting must not run")

        def is_zero(self) -> bool:
            type(self).hook_calls += 1
            raise AssertionError("custom Decimal zero test must not run")

    hooked_cost = HookedDecimal("1")
    changed_model = replace(first.candidate_models[0], cost=hooked_cost)
    HookedDecimal.hook_calls = 0
    with pytest.raises(TypeError, match="plain Decimal"):
        _scope((replace(first, candidate_models=(changed_model, *first.candidate_models[1:])),))
    assert HookedDecimal.hook_calls == 0

    class HookedFloat(float):
        hook_calls = 0
        hooks_enabled = False

        def __float__(self) -> float:
            type(self).hook_calls += 1
            if type(self).hooks_enabled:
                raise AssertionError("custom numeric conversion must not run")
            return super().__float__()

    hooked_quality = HookedFloat(0.5)
    changed_outcome = replace(first.outcomes[0], quality=hooked_quality)
    HookedFloat.hook_calls = 0
    HookedFloat.hooks_enabled = True
    with pytest.raises(TypeError, match="plain real number"):
        _scope((replace(first, outcomes=(changed_outcome, *first.outcomes[1:])),))
    assert HookedFloat.hook_calls == 0

    HookedFloat.hooks_enabled = False
    hooked_weight = HookedFloat(1.0)
    hooked_spec = TierSpec(BudgetTier.FAST, Decimal("1"), hooked_weight)
    HookedFloat.hook_calls = 0
    HookedFloat.hooks_enabled = True
    with pytest.raises(TypeError, match="plain real number"):
        evaluation_scope_sha256((first,), (hooked_spec,), max_calls_per_query=1)
    assert HookedFloat.hook_calls == 0


def test_data_hashes_reject_empty_or_duplicate_examples() -> None:
    example = load_evaluation_dataset().examples[0]

    with pytest.raises(ValueError, match="must not be empty"):
        evaluation_data_sha256(())
    with pytest.raises(ValueError, match="unique example IDs"):
        evaluation_replay_sha256((example, example))

    invalid_text = (replace(example, prompt="\ud800"),)
    with pytest.raises(ValueError, match="invalid Unicode text"):
        evaluation_data_sha256(invalid_text)


def test_extreme_zero_exponents_hash_as_canonical_zero() -> None:
    example = load_evaluation_dataset().examples[0]
    model_id = example.candidate_models[0].model_id

    def with_cost(cost: Decimal):
        return replace(
            example,
            candidate_models=tuple(
                replace(model, cost=cost) if model.model_id == model_id else model
                for model in example.candidate_models
            ),
            outcomes=tuple(
                replace(outcome, cost=cost) if outcome.model_id == model_id else outcome
                for outcome in example.outcomes
            ),
        )

    canonical = (with_cost(Decimal(0)),)
    extreme = (with_cost(Decimal("0e-100000000")),)

    assert evaluation_data_sha256(extreme) == evaluation_data_sha256(canonical)
    assert evaluation_replay_sha256(extreme) == evaluation_replay_sha256(canonical)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (Decimal("1"), Decimal("1.0")),
        (Decimal("0.1"), Decimal("0.10")),
        (Decimal("1E+2"), Decimal("100.00")),
    ],
)
def test_equivalent_nonzero_cost_encodings_share_provenance(
    left: Decimal,
    right: Decimal,
) -> None:
    example = load_evaluation_dataset().examples[0]
    model_id = example.candidate_models[0].model_id

    def with_cost(cost: Decimal):
        return replace(
            example,
            candidate_models=tuple(
                replace(model, cost=cost) if model.model_id == model_id else model
                for model in example.candidate_models
            ),
            outcomes=tuple(
                replace(outcome, cost=cost) if outcome.model_id == model_id else outcome
                for outcome in example.outcomes
            ),
        )

    assert evaluation_data_sha256((with_cost(left),)) == evaluation_data_sha256((with_cost(right),))
