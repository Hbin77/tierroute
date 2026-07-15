# SPDX-License-Identifier: Apache-2.0
"""Canonical hashes for replay data identity and order-sensitive evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from tierroute.core import ModelSpec, as_cost, canonical_cost_text
from tierroute.eval.schemas import CandidateOutcome, EvaluationExample, TierSpec

EVALUATION_SCOPE_ALGORITHM = "tierroute-evaluation-scope-v1"
_MAX_METADATA_DEPTH = 32
_MAX_METADATA_NODES = 100_000
_MAX_METADATA_INTEGER_BITS = 1_000_000
_MAX_METADATA_DECIMAL_POSITIONS = 100_000
_MAX_METADATA_ENCODED_BYTES = 8 * 1024 * 1024
_MAX_EVALUATION_METADATA_NODES = 10_000_000
_MAX_EVALUATION_METADATA_ENCODED_BYTES = 256 * 1024 * 1024
_MAX_EVALUATION_SCOPE_ENCODED_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _FrozenMetadata(Mapping[str, object]):
    """Canonical immutable mapping passed to routers during offline replay."""

    _items: tuple[tuple[str, object], ...]
    _keys: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_keys", tuple(key for key, _ in self._items))

    def __getitem__(self, key: str) -> object:
        if type(key) is not str:
            raise KeyError(key)
        index = bisect_left(self._keys, key)
        if index < len(self._items) and self._items[index][0] == key:
            return self._items[index][1]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(slots=True)
class _ScopeSnapshotBudget:
    encoded_bytes: int = 0

    def add_payload(self, byte_count: int) -> None:
        self.encoded_bytes += byte_count
        if self.encoded_bytes > _MAX_EVALUATION_SCOPE_ENCODED_BYTES:
            raise ValueError(
                "evaluation scope exceeds the "
                f"{_MAX_EVALUATION_SCOPE_ENCODED_BYTES:,}-byte encoded-payload limit"
            )


@dataclass(slots=True)
class _MetadataMaterializationBudget:
    scope: _ScopeSnapshotBudget
    nodes: int = 0
    encoded_bytes: int = 0

    def add_node(self) -> None:
        self.nodes += 1
        if self.nodes > _MAX_EVALUATION_METADATA_NODES:
            raise ValueError(
                "evaluation metadata exceeds the "
                f"{_MAX_EVALUATION_METADATA_NODES:,}-node snapshot limit"
            )

    def add_payload(self, byte_count: int) -> None:
        self.scope.add_payload(byte_count)
        self.encoded_bytes += byte_count
        if self.encoded_bytes > _MAX_EVALUATION_METADATA_ENCODED_BYTES:
            raise ValueError(
                "evaluation metadata exceeds the "
                f"{_MAX_EVALUATION_METADATA_ENCODED_BYTES:,}-byte snapshot encoded-payload limit"
            )


@dataclass(slots=True)
class _MetadataBudget:
    materialization: _MetadataMaterializationBudget
    nodes: int = 0
    encoded_bytes: int = 0
    active_containers: set[int] = field(default_factory=set)

    def add_text(self, value: str, path: str) -> None:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{path} contains invalid Unicode text") from error
        self.add_payload(len(encoded), path)

    def add_payload(self, byte_count: int, path: str) -> None:
        self.encoded_bytes += byte_count
        self.materialization.add_payload(byte_count)
        if self.encoded_bytes > _MAX_METADATA_ENCODED_BYTES:
            raise ValueError(
                f"{path} exceeds the {_MAX_METADATA_ENCODED_BYTES:,}-byte "
                "metadata encoded-payload limit"
            )


def _validated_examples(
    examples: Sequence[EvaluationExample],
) -> tuple[EvaluationExample, ...]:
    ordered = tuple(examples)
    if not ordered:
        raise ValueError("evaluation examples must not be empty")
    if any(type(example) is not EvaluationExample for example in ordered):
        raise TypeError("evaluation examples must contain EvaluationExample values")
    example_ids = tuple(
        _plain_text(example.example_id, f"examples[{index}].example_id")
        for index, example in enumerate(ordered)
    )
    if len(example_ids) != len(set(example_ids)):
        raise ValueError(
            "evaluation examples must have unique example_id values (unique example IDs)"
        )
    return ordered


def _canonical_metadata_decimal(value: Decimal, path: str) -> Decimal:
    if not value.is_finite():
        raise ValueError(f"{path} must be a finite Decimal")
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f"{path} must be a finite Decimal")
    trailing_zeros = 0
    for digit in reversed(digits):
        if digit != 0:
            break
        trailing_zeros += 1
    if trailing_zeros == len(digits):
        return Decimal(0)
    significant_digits = digits[: len(digits) - trailing_zeros]
    canonical_exponent = exponent + trailing_zeros
    highest_exclusive = canonical_exponent + len(significant_digits)
    if (
        len(digits) > _MAX_METADATA_DECIMAL_POSITIONS
        or canonical_exponent < -_MAX_METADATA_DECIMAL_POSITIONS
        or highest_exclusive > _MAX_METADATA_DECIMAL_POSITIONS
    ):
        raise ValueError(
            f"{path} exceeds the supported metadata Decimal range "
            f"({_MAX_METADATA_DECIMAL_POSITIONS:,} positions)"
        )
    return Decimal((sign, significant_digits, canonical_exponent))


def _freeze_metadata_value(
    value: object,
    *,
    path: str,
    depth: int,
    budget: _MetadataBudget,
) -> object:
    budget.materialization.add_node()
    budget.nodes += 1
    if budget.nodes > _MAX_METADATA_NODES:
        raise ValueError(f"evaluation metadata exceeds {_MAX_METADATA_NODES:,} canonical values")
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError(f"{path} exceeds maximum metadata depth {_MAX_METADATA_DEPTH}")
    if value is None:
        return value
    if type(value) is bool:
        budget.add_payload(1, path)
        return value
    if type(value) is str:
        budget.add_text(value, path)
        return value
    if type(value) is int:
        if value.bit_length() > _MAX_METADATA_INTEGER_BITS:
            raise ValueError(
                f"{path} exceeds the {_MAX_METADATA_INTEGER_BITS:,}-bit metadata integer limit"
            )
        hex_digits = max(1, (value.bit_length() + 3) // 4)
        budget.add_payload(2 + (1 if value < 0 else 0) + hex_digits, path)
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be a finite float")
        budget.add_payload(len(value.hex()), path)
        return value
    if type(value) is Decimal:
        canonical = _canonical_metadata_decimal(value, path)
        _, digits, exponent = canonical.as_tuple()
        budget.add_payload(1 + len(digits) + len(str(exponent)), path)
        return canonical
    if type(value) is _FrozenMetadata:
        container_id = id(value)
        if container_id in budget.active_containers:
            raise ValueError(f"{path} contains a cyclic metadata container")
        if type(value._items) is not tuple:
            raise TypeError(f"{path} has an invalid internal immutable mapping")
        budget.active_containers.add(container_id)
        items: list[tuple[str, object]] = []
        seen_keys: set[str] = set()
        try:
            for index, pair in enumerate(value._items):
                if type(pair) is not tuple or len(pair) != 2:
                    raise TypeError(f"{path}[{index}] is not an immutable key/value pair")
                key, item = pair
                if type(key) is not str:
                    raise TypeError(f"{path} keys must be plain strings")
                if key in seen_keys:
                    raise ValueError(f"{path} keys must be unique")
                seen_keys.add(key)
                budget.add_text(key, f"{path} key")
                items.append(
                    (
                        key,
                        _freeze_metadata_value(
                            item,
                            path=f"{path}[{key!r}]",
                            depth=depth + 1,
                            budget=budget,
                        ),
                    )
                )
            return _FrozenMetadata(tuple(sorted(items, key=lambda pair: pair[0])))
        finally:
            budget.active_containers.remove(container_id)
    if type(value) in (list, tuple):
        container_id = id(value)
        if container_id in budget.active_containers:
            raise ValueError(f"{path} contains a cyclic metadata container")
        budget.active_containers.add(container_id)
        try:
            return tuple(
                _freeze_metadata_value(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    budget=budget,
                )
                for index, item in enumerate(value)
            )
        finally:
            budget.active_containers.remove(container_id)
    if type(value) is dict:
        container_id = id(value)
        if container_id in budget.active_containers:
            raise ValueError(f"{path} contains a cyclic metadata container")
        budget.active_containers.add(container_id)
        items: list[tuple[str, object]] = []
        try:
            for key in value:
                if type(key) is not str:
                    raise TypeError(f"{path} keys must be plain strings")
                budget.add_text(key, f"{path} key")
            for key in sorted(value):
                items.append(
                    (
                        key,
                        _freeze_metadata_value(
                            value[key],
                            path=f"{path}[{key!r}]",
                            depth=depth + 1,
                            budget=budget,
                        ),
                    )
                )
            return _FrozenMetadata(tuple(items))
        finally:
            budget.active_containers.remove(container_id)
    raise TypeError(
        f"{path} has unsupported type {type(value).__name__}; metadata must use "
        "plain JSON-like values or finite Decimal values"
    )


def _freeze_metadata(
    metadata: Mapping[str, object],
    path: str,
    materialization: _MetadataMaterializationBudget,
) -> _FrozenMetadata:
    budget = _MetadataBudget(materialization)
    frozen = _freeze_metadata_value(metadata, path=path, depth=0, budget=budget)
    if not isinstance(frozen, _FrozenMetadata):
        raise TypeError(f"{path} must be a plain mapping")
    return frozen


def _plain_text(
    value: object,
    path: str,
    scope_budget: _ScopeSnapshotBudget | None = None,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be a plain string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{path} contains invalid Unicode text") from error
    if scope_budget is not None:
        scope_budget.add_payload(len(encoded))
    return value


def _canonical_scope_cost(
    value: object,
    path: str,
    scope_budget: _ScopeSnapshotBudget,
) -> Decimal:
    """Copy an exact built-in Decimal without invoking subclass hooks."""

    if type(value) is not Decimal:
        raise TypeError(f"{path} must be a plain Decimal")
    canonical = canonical_cost_text(value)
    scope_budget.add_payload(len(canonical))
    return as_cost(canonical)


def _plain_binary64(
    value: object,
    path: str,
    scope_budget: _ScopeSnapshotBudget,
) -> float:
    """Normalize a built-in int/float while rejecting behavior-bearing subclasses."""

    if type(value) not in (int, float):
        raise TypeError(f"{path} must be a plain real number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{path} must be finite")
    scope_budget.add_payload(len(normalized.hex()))
    return normalized


def _snapshot_evaluation_examples(
    examples: Sequence[EvaluationExample],
    *,
    _scope_budget: _ScopeSnapshotBudget | None = None,
) -> tuple[EvaluationExample, ...]:
    """Copy policy-visible metadata into a canonical, deeply immutable form."""

    ordered = _validated_examples(examples)
    scope_budget = _scope_budget or _ScopeSnapshotBudget()
    materialization = _MetadataMaterializationBudget(scope_budget)
    snapshots: list[EvaluationExample] = []
    for example_index, example in enumerate(ordered):
        example_id = _plain_text(
            example.example_id,
            f"examples[{example_index}].example_id",
            scope_budget,
        )
        prompt = _plain_text(
            example.prompt,
            f"examples[{example_index}].prompt",
            scope_budget,
        )
        domain = _plain_text(
            example.domain,
            f"examples[{example_index}].domain",
            scope_budget,
        )
        router_metadata = _freeze_metadata(
            example.router_metadata,
            f"examples[{example_index}].router_metadata",
            materialization,
        )
        models: list[ModelSpec] = []
        for model_index, model in enumerate(example.candidate_models):
            path = f"examples[{example_index}].candidate_models[{model_index}]"
            if type(model) is not ModelSpec:
                raise TypeError(f"{path} must be a ModelSpec")
            models.append(
                ModelSpec(
                    _plain_text(model.model_id, f"{path}.model_id", scope_budget),
                    _canonical_scope_cost(model.cost, f"{path}.cost", scope_budget),
                    (
                        None
                        if model.display_name is None
                        else _plain_text(
                            model.display_name,
                            f"{path}.display_name",
                            scope_budget,
                        )
                    ),
                    _freeze_metadata(model.metadata, f"{path}.metadata", materialization),
                )
            )
        outcomes: list[CandidateOutcome] = []
        for outcome_index, outcome in enumerate(example.outcomes):
            path = f"examples[{example_index}].outcomes[{outcome_index}]"
            if type(outcome) is not CandidateOutcome:
                raise TypeError(f"{path} must be a CandidateOutcome")
            quality = _plain_binary64(outcome.quality, f"{path}.quality", scope_budget)
            outcomes.append(
                CandidateOutcome(
                    _plain_text(outcome.model_id, f"{path}.model_id", scope_budget),
                    _plain_text(outcome.output, f"{path}.output", scope_budget),
                    _canonical_scope_cost(outcome.cost, f"{path}.cost", scope_budget),
                    quality,
                )
            )
        snapshots.append(
            EvaluationExample(
                example_id,
                prompt,
                domain,
                tuple(outcomes),
                tuple(models),
                router_metadata,
            )
        )
    return tuple(snapshots)


def _snapshot_tier_specs(
    tier_specs: Sequence[TierSpec],
    *,
    _scope_budget: _ScopeSnapshotBudget | None = None,
) -> tuple[TierSpec, ...]:
    """Normalize the tier values used by both routing and scope hashing."""

    specs = tuple(tier_specs)
    if not specs:
        raise ValueError("tier_specs must not be empty")
    snapshots: list[TierSpec] = []
    scope_budget = _scope_budget or _ScopeSnapshotBudget()
    for index, spec in enumerate(specs):
        path = f"tier_specs[{index}]"
        if type(spec) is not TierSpec:
            raise TypeError("tier_specs must contain TierSpec values")
        snapshots.append(
            TierSpec(
                spec.tier,
                _canonical_scope_cost(
                    spec.budget_limit,
                    f"{path}.budget_limit",
                    scope_budget,
                ),
                _plain_binary64(spec.weight, f"{path}.weight", scope_budget),
            )
        )
    tiers = tuple(spec.tier for spec in snapshots)
    if len(tiers) != len(set(tiers)):
        raise ValueError("tier_specs must contain unique tiers")
    return tuple(snapshots)


def _snapshot_evaluation_scope(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
) -> tuple[tuple[EvaluationExample, ...], tuple[TierSpec, ...]]:
    """Freeze examples and tiers under one aggregate encoded-payload budget."""

    scope_budget = _ScopeSnapshotBudget()
    snapshots = _snapshot_evaluation_examples(examples, _scope_budget=scope_budget)
    specs = _snapshot_tier_specs(tier_specs, _scope_budget=scope_budget)
    return snapshots, specs


class _ScopeHashWriter:
    """Write a typed, length-delimited stream directly into SHA-256."""

    __slots__ = ("_hash",)

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def token(self, kind: bytes, payload: bytes = b"") -> None:
        if len(kind) > 0xFFFF or len(payload) > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("evaluation scope token exceeds encoding limits")
        self._hash.update(len(kind).to_bytes(2, "big"))
        self._hash.update(kind)
        self._hash.update(len(payload).to_bytes(8, "big"))
        self._hash.update(payload)

    def count(self, kind: bytes, value: int) -> None:
        if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("evaluation scope count exceeds encoding limits")
        self.token(kind, value.to_bytes(8, "big"))

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _strict_utf8(value: str, path: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{path} contains invalid Unicode text") from error


def _write_metadata(writer: _ScopeHashWriter, value: object) -> None:
    if value is None:
        writer.token(b"metadata-none")
        return
    if type(value) is bool:
        writer.token(b"metadata-bool", b"1" if value else b"0")
        return
    if type(value) is str:
        writer.token(b"metadata-string", _strict_utf8(value, "metadata string"))
        return
    if type(value) is int:
        writer.token(b"metadata-integer", hex(value).encode("ascii"))
        return
    if type(value) is float:
        writer.token(b"metadata-float", value.hex().encode("ascii"))
        return
    if type(value) is Decimal:
        sign, digits, exponent = value.as_tuple()
        writer.token(b"metadata-decimal")
        writer.token(b"decimal-sign", bytes((sign,)))
        writer.token(b"decimal-digits", bytes(digits))
        writer.token(b"decimal-exponent", str(exponent).encode("ascii"))
        return
    if type(value) is tuple:
        writer.count(b"metadata-sequence", len(value))
        for item in value:
            _write_metadata(writer, item)
        return
    if type(value) is _FrozenMetadata:
        writer.count(b"metadata-mapping", len(value._items))
        for key, item in value._items:
            writer.token(b"metadata-key", _strict_utf8(key, "metadata key"))
            _write_metadata(writer, item)
        return
    raise AssertionError(f"metadata snapshot contains unsupported type {type(value).__name__}")


def _write_evaluation_scope(
    writer: _ScopeHashWriter,
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    max_calls_per_query: int,
) -> None:
    writer.token(b"algorithm", EVALUATION_SCOPE_ALGORITHM.encode("ascii"))
    writer.count(b"max-calls-per-query", max_calls_per_query)
    writer.count(b"tier-specs", len(tier_specs))
    for spec in tier_specs:
        writer.token(b"tier-spec")
        writer.token(b"tier", spec.tier.value.encode("ascii"))
        writer.token(b"budget-limit", canonical_cost_text(spec.budget_limit).encode("ascii"))
        writer.token(b"weight", spec.weight.hex().encode("ascii"))
    writer.count(b"examples", len(examples))
    for example in examples:
        writer.token(b"example")
        writer.token(b"example-id", _strict_utf8(example.example_id, "example_id"))
        writer.token(b"prompt", _strict_utf8(example.prompt, "prompt"))
        writer.token(b"split-domain", _strict_utf8(example.domain, "split domain"))
        writer.token(b"router-metadata")
        _write_metadata(writer, example.router_metadata)
        writer.count(b"candidate-models", len(example.candidate_models))
        for model in example.candidate_models:
            writer.token(b"candidate-model")
            writer.token(b"model-id", _strict_utf8(model.model_id, "model_id"))
            writer.token(b"quoted-cost", canonical_cost_text(model.cost).encode("ascii"))
            if model.display_name is None:
                writer.token(b"display-name-none")
            else:
                writer.token(
                    b"display-name",
                    _strict_utf8(model.display_name, "display_name"),
                )
            writer.token(b"model-metadata")
            _write_metadata(writer, model.metadata)
        writer.count(b"outcomes", len(example.outcomes))
        for outcome in example.outcomes:
            writer.token(b"outcome")
            writer.token(b"model-id", _strict_utf8(outcome.model_id, "outcome model_id"))
            writer.token(b"output", _strict_utf8(outcome.output, "outcome output"))
            writer.token(
                b"realized-cost",
                canonical_cost_text(outcome.cost).encode("ascii"),
            )
            writer.token(b"quality", float(outcome.quality).hex().encode("ascii"))


def _evaluation_scope_sha256_from_snapshots(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    max_calls_per_query: int,
) -> str:
    """Hash trusted internal snapshots without copying the full replay again."""

    writer = _ScopeHashWriter()
    _write_evaluation_scope(writer, examples, tier_specs, max_calls_per_query)
    return writer.hexdigest()


def _canonical_rows(examples: tuple[EvaluationExample, ...]) -> list[dict[str, object]]:
    rows = []
    for example in examples:
        outcomes = {outcome.model_id: outcome for outcome in example.outcomes}
        rows.append(
            {
                "example_id": example.example_id,
                "prompt": example.prompt,
                "domain": example.domain,
                "models": [
                    {
                        "model_id": model.model_id,
                        "quoted_cost": canonical_cost_text(model.cost),
                        "realized_cost": canonical_cost_text(outcomes[model.model_id].cost),
                        "quality": outcomes[model.model_id].quality,
                    }
                    for model in sorted(
                        example.candidate_models,
                        key=lambda candidate: candidate.model_id,
                    )
                ],
            }
        )
    return rows


def _sha256(examples: tuple[EvaluationExample, ...]) -> str:
    try:
        document = json.dumps(
            _canonical_rows(examples),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except UnicodeEncodeError as error:
        raise ValueError("evaluation data contains invalid Unicode text") from error
    return hashlib.sha256(document).hexdigest()


def evaluation_data_sha256(examples: Sequence[EvaluationExample]) -> str:
    """Hash replay row content independent of caller order.

    Predictor fitting sorts rows by private example ID, so this identity can be
    compared directly with a fitted predictor artifact's training-data hash.
    """

    validated = _validated_examples(examples)
    return _sha256(tuple(sorted(validated, key=lambda example: example.example_id)))


def evaluation_replay_sha256(examples: Sequence[EvaluationExample]) -> str:
    """Hash replay row content in supplied order for cumulative budget evidence."""

    return _sha256(_validated_examples(examples))


def evaluation_scope_sha256(
    examples: Sequence[EvaluationExample],
    tier_specs: Sequence[TierSpec],
    *,
    max_calls_per_query: int,
) -> str:
    """Hash the complete normalized replay scope that can affect an evaluation.

    This versioned identity is intentionally separate from the stable training and
    policy-artifact hashes above. It includes row and candidate order, every label and
    output, split-only fields, and the policy-visible metadata snapshot. Unsupported
    metadata fails closed instead of relying on ``repr`` or pickle. Ordered tier specs
    and the call cap are included because both change routing or metric semantics.
    """

    if type(max_calls_per_query) is not int:
        raise TypeError("max_calls_per_query must be an integer")
    if max_calls_per_query < 1:
        raise ValueError("max_calls_per_query must be positive")
    snapshots, specs = _snapshot_evaluation_scope(examples, tier_specs)
    return _evaluation_scope_sha256_from_snapshots(
        snapshots,
        specs,
        max_calls_per_query,
    )
