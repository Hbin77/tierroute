# SPDX-License-Identifier: Apache-2.0
"""Strict canonical artifacts for exact tier-lambda policies."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

from tierroute.core import BudgetTier, as_cost
from tierroute.core.atomic_io import AtomicTextWrite, replace_text_bundle
from tierroute.core.integer_text import decimal_to_integer, integer_to_decimal
from tierroute.eval import (
    EvaluationExample,
    TierSpec,
    evaluation_data_sha256,
    evaluation_replay_sha256,
)
from tierroute.features import EmbeddingProvider
from tierroute.policies.lambda_threshold import TieredLambdaRouter
from tierroute.policies.lambda_tuning import LambdaCandidateSet, TierLambdaTuningResult
from tierroute.policies.resource_limits import (
    MAX_POLICY_ARTIFACT_BYTES,
    MAX_POLICY_CANDIDATES_PER_TIER,
    MAX_POLICY_INTEGER_DECIMAL_DIGITS,
    MAX_POLICY_LEDGER_ADAPTER_NAME_BYTES,
)
from tierroute.predictors.artifacts import BilinearPredictorArtifact

LAMBDA_POLICY_ARTIFACT_VERSION = 1
LAMBDA_NUMERIC_CONVENTION = "exact-fraction-v1"
_MAX_POLICY_INTEGER_EXCLUSIVE = 10**MAX_POLICY_INTEGER_DECIMAL_DIGITS
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INTEGER_PATTERN = re.compile(r"0|-?[1-9][0-9]*")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*")


def predictor_artifact_sha256(artifact: BilinearPredictorArtifact) -> str:
    """Hash the predictor's canonical JSON bytes."""

    try:
        document = artifact.to_json().encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("predictor artifact contains invalid Unicode text") from error
    return hashlib.sha256(document).hexdigest()


def _strict_fields(payload: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{context} fields mismatch: missing={missing}, extra={extra}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed object")
    return value


def _utf8_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must contain valid Unicode scalar values") from error
    return value


def _fraction_dict(value: Fraction) -> dict[str, str]:
    _validate_fraction_size(value, "fraction")
    return {
        "numerator": integer_to_decimal(value.numerator),
        "denominator": integer_to_decimal(value.denominator),
    }


def _validate_fraction_size(value: Fraction, context: str) -> None:
    if (
        abs(value.numerator) >= _MAX_POLICY_INTEGER_EXCLUSIVE
        or value.denominator >= _MAX_POLICY_INTEGER_EXCLUSIVE
    ):
        raise ValueError(
            f"{context} exceeds the policy artifact integer limit "
            f"({MAX_POLICY_INTEGER_DECIMAL_DIGITS:,} decimal digits)"
        )


def _fraction(value: object, context: str, *, nonnegative: bool = False) -> Fraction:
    payload = _mapping(value, context)
    _strict_fields(payload, {"numerator", "denominator"}, context)
    numerator = payload["numerator"]
    denominator = payload["denominator"]
    if isinstance(numerator, str):
        numerator_digits = len(numerator) - int(numerator.startswith("-"))
        if numerator_digits > MAX_POLICY_INTEGER_DECIMAL_DIGITS:
            raise ValueError(f"{context}.numerator exceeds the policy artifact integer limit")
    if isinstance(denominator, str) and len(denominator) > MAX_POLICY_INTEGER_DECIMAL_DIGITS:
        raise ValueError(f"{context}.denominator exceeds the policy artifact integer limit")
    if not isinstance(numerator, str) or not _INTEGER_PATTERN.fullmatch(numerator):
        raise ValueError(f"{context}.numerator must be a canonical integer string")
    if not isinstance(denominator, str) or not _POSITIVE_INTEGER_PATTERN.fullmatch(denominator):
        raise ValueError(f"{context}.denominator must be a canonical positive integer string")
    result = Fraction(decimal_to_integer(numerator), decimal_to_integer(denominator))
    if (
        integer_to_decimal(result.numerator) != numerator
        or integer_to_decimal(result.denominator) != denominator
    ):
        raise ValueError(f"{context} must be a reduced canonical fraction")
    if nonnegative and result < 0:
        raise ValueError(f"{context} must be non-negative")
    return result


def _validate_artifact_document(document: str) -> None:
    if not isinstance(document, str):
        raise ValueError("lambda policy artifact must be text")
    if len(document) > MAX_POLICY_ARTIFACT_BYTES:
        raise ValueError(
            f"lambda policy artifact exceeds {MAX_POLICY_ARTIFACT_BYTES:,} UTF-8 bytes"
        )
    try:
        encoded = document.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("lambda policy artifact is not valid UTF-8 text") from error
    if len(encoded) > MAX_POLICY_ARTIFACT_BYTES:
        raise ValueError(
            f"lambda policy artifact exceeds {MAX_POLICY_ARTIFACT_BYTES:,} UTF-8 bytes"
        )


@dataclass(frozen=True, slots=True)
class LambdaPolicyArtifact:
    """All fitted policy state, accounting identity, and provenance hashes."""

    lambda_by_tier: Mapping[BudgetTier, Fraction]
    predictor_sha256: str
    tuning_data_sha256: str
    tuning_replay_sha256: str
    prediction_sha256: str
    training_example_count: int
    training_domains: tuple[str, ...]
    tier_specs: tuple[TierSpec, ...]
    ledger_adapter_name: str
    candidate_sets: tuple[LambdaCandidateSet, ...]
    artifact_version: int = LAMBDA_POLICY_ARTIFACT_VERSION
    numeric_convention: str = LAMBDA_NUMERIC_CONVENTION

    def __post_init__(self) -> None:
        if type(self.artifact_version) is not int or self.artifact_version != (
            LAMBDA_POLICY_ARTIFACT_VERSION
        ):
            raise ValueError(f"artifact_version must equal {LAMBDA_POLICY_ARTIFACT_VERSION}")
        if self.numeric_convention != LAMBDA_NUMERIC_CONVENTION:
            raise ValueError(f"numeric_convention must equal {LAMBDA_NUMERIC_CONVENTION!r}")
        if not isinstance(self.lambda_by_tier, Mapping):
            raise TypeError("lambda_by_tier must be a mapping")
        lambdas = dict(self.lambda_by_tier)
        if not lambdas or any(not isinstance(tier, BudgetTier) for tier in lambdas):
            raise ValueError("lambda_by_tier must use non-empty BudgetTier keys")
        if any(not isinstance(value, Fraction) or value < 0 for value in lambdas.values()):
            raise ValueError("lambda_by_tier values must be non-negative Fractions")
        for tier, value in lambdas.items():
            _validate_fraction_size(value, f"lambda_by_tier[{tier.value}]")
        for name in (
            "predictor_sha256",
            "tuning_data_sha256",
            "tuning_replay_sha256",
            "prediction_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if isinstance(self.training_example_count, bool) or not isinstance(
            self.training_example_count, int
        ):
            raise TypeError("training_example_count must be an integer")
        if self.training_example_count < 1:
            raise ValueError("training_example_count must be positive")
        if not self.training_domains or self.training_domains != tuple(
            sorted(set(self.training_domains))
        ):
            raise ValueError("training_domains must be non-empty, sorted, and unique")
        for domain in self.training_domains:
            _utf8_text(domain, "training domain")
        specs = tuple(self.tier_specs)
        if any(not isinstance(spec, TierSpec) for spec in specs):
            raise TypeError("tier_specs must contain TierSpec values")
        tiers = tuple(spec.tier for spec in specs)
        if not specs or len(tiers) != len(set(tiers)) or set(tiers) != set(lambdas):
            raise ValueError("tier_specs and lambda_by_tier must cover identical unique tiers")
        ledger_adapter_name = _utf8_text(self.ledger_adapter_name, "ledger_adapter_name")
        if len(ledger_adapter_name.encode("utf-8")) > MAX_POLICY_LEDGER_ADAPTER_NAME_BYTES:
            raise ValueError(
                "ledger_adapter_name exceeds the policy artifact metadata limit "
                f"({MAX_POLICY_LEDGER_ADAPTER_NAME_BYTES:,} UTF-8 bytes)"
            )
        candidate_sets = tuple(self.candidate_sets)
        if any(not isinstance(item, LambdaCandidateSet) for item in candidate_sets):
            raise TypeError("candidate_sets must contain LambdaCandidateSet values")
        if tuple(item.tier for item in candidate_sets) != tiers:
            raise ValueError("candidate_sets must align with tier_specs order")
        for candidate_set in candidate_sets:
            if len(candidate_set.values) > MAX_POLICY_CANDIDATES_PER_TIER:
                raise ValueError(
                    "candidate set exceeds the policy artifact per-tier candidate limit "
                    f"({MAX_POLICY_CANDIDATES_PER_TIER:,})"
                )
            for value in candidate_set.values:
                _validate_fraction_size(value, f"candidate_sets[{candidate_set.tier.value}]")
            if lambdas[candidate_set.tier] not in candidate_set.values:
                raise ValueError("each selected lambda must occur in its candidate set")
        object.__setattr__(self, "lambda_by_tier", MappingProxyType(lambdas))
        object.__setattr__(self, "tier_specs", specs)
        object.__setattr__(self, "candidate_sets", candidate_sets)

    @classmethod
    def from_tuning(
        cls,
        predictor: BilinearPredictorArtifact,
        tuning: TierLambdaTuningResult,
        tier_specs: tuple[TierSpec, ...],
        ledger_adapter_name: str,
    ) -> LambdaPolicyArtifact:
        """Bind a tuning result to the exact final predictor and accounting scope."""

        specs = tuple(tier_specs)
        if tuple(selection.tier for selection in tuning.selections) != tuple(
            spec.tier for spec in specs
        ):
            raise ValueError("tuning selections and tier_specs must align")
        if any(
            selection.report.tier_spec != spec
            for selection, spec in zip(tuning.selections, specs, strict=True)
        ):
            raise ValueError("tuning report tier specifications do not match")
        if any(
            selection.report.budget.adapter_name != ledger_adapter_name
            for selection in tuning.selections
        ):
            raise ValueError("tuning report ledger adapter does not match policy metadata")
        if tuning.data_sha256 != predictor.training_data_sha256:
            raise ValueError("tuning data SHA-256 does not match predictor training data")
        if tuning.example_count != predictor.training_example_count:
            raise ValueError("tuning example count does not match predictor training data")
        if tuning.domains != predictor.training_domains:
            raise ValueError("tuning domains do not match predictor training data")
        return cls(
            lambda_by_tier=tuning.lambda_by_tier,
            predictor_sha256=predictor_artifact_sha256(predictor),
            tuning_data_sha256=tuning.data_sha256,
            tuning_replay_sha256=tuning.replay_sha256,
            prediction_sha256=tuning.prediction_sha256,
            training_example_count=predictor.training_example_count,
            training_domains=predictor.training_domains,
            tier_specs=specs,
            ledger_adapter_name=ledger_adapter_name,
            candidate_sets=tuple(selection.candidates for selection in tuning.selections),
        )

    def validate_predictor(self, predictor: BilinearPredictorArtifact) -> None:
        """Fail closed if policy and predictor provenance do not match."""

        if predictor_artifact_sha256(predictor) != self.predictor_sha256:
            raise ValueError("policy artifact predictor SHA-256 does not match")
        if predictor.training_data_sha256 != self.tuning_data_sha256:
            raise ValueError("policy and predictor training-data hashes do not match")
        if predictor.training_example_count != self.training_example_count:
            raise ValueError("policy and predictor training example counts do not match")
        if predictor.training_domains != self.training_domains:
            raise ValueError("policy and predictor training domains do not match")

    def validate_tuning_data(self, examples: Sequence[EvaluationExample]) -> None:
        """Fail closed unless replay content and order match policy tuning evidence."""

        ordered = tuple(examples)
        if evaluation_data_sha256(ordered) != self.tuning_data_sha256:
            raise ValueError("policy tuning-data SHA-256 does not match routing data")
        if evaluation_replay_sha256(ordered) != self.tuning_replay_sha256:
            raise ValueError("policy tuning replay-order SHA-256 does not match routing data")
        if len(ordered) != self.training_example_count:
            raise ValueError("policy tuning example count does not match routing data")
        domains = tuple(sorted({example.domain for example in ordered}))
        if domains != self.training_domains:
            raise ValueError("policy tuning domains do not match routing data")

    def build_router(
        self,
        predictor: BilinearPredictorArtifact,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> TieredLambdaRouter:
        """Build a deployable offline router after provenance verification."""

        self.validate_predictor(predictor)
        return TieredLambdaRouter(
            predictor.build_predictor(embedding_provider=embedding_provider),
            self.lambda_by_tier,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible canonical object."""

        return {
            "artifact_version": self.artifact_version,
            "numeric_convention": self.numeric_convention,
            "lambdas": {
                tier.value: _fraction_dict(self.lambda_by_tier[tier])
                for tier in sorted(self.lambda_by_tier, key=lambda item: item.value)
            },
            "predictor_sha256": self.predictor_sha256,
            "tuning": {
                "data_sha256": self.tuning_data_sha256,
                "replay_sha256": self.tuning_replay_sha256,
                "prediction_sha256": self.prediction_sha256,
                "example_count": self.training_example_count,
                "domains": list(self.training_domains),
                "ledger_adapter": self.ledger_adapter_name,
                "tier_specs": [
                    {
                        "tier": spec.tier.value,
                        "budget_limit": str(spec.budget_limit),
                        "weight": _fraction_dict(Fraction.from_float(float(spec.weight))),
                    }
                    for spec in self.tier_specs
                ],
                "candidate_sets": [
                    {
                        "tier": item.tier.value,
                        "values": [_fraction_dict(value) for value in item.values],
                        "total_derived_values": item.total_derived_values,
                        "exhaustive": item.exhaustive,
                        "strategy": item.strategy,
                        "observed_breakpoint_count": item.observed_breakpoint_count,
                    }
                    for item in self.candidate_sets
                ],
            },
        }

    def to_json(self) -> str:
        """Serialize deterministic strict JSON."""

        document = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        _validate_artifact_document(document)
        return document

    def save(self, path: str | Path) -> Path:
        """Atomically save the canonical JSON artifact."""

        destination = Path(path)
        return replace_text_bundle(
            (AtomicTextWrite(destination, self.to_json(), type(self).from_json),)
        )[0]

    @classmethod
    def from_json(cls, document: str) -> LambdaPolicyArtifact:
        """Parse strict JSON without pickle or code execution."""

        _validate_artifact_document(document)

        def reject_constant(value: str) -> object:
            raise ValueError(f"non-standard JSON number {value!r} is forbidden")

        def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r} is forbidden")
                result[key] = value
            return result

        try:
            payload = json.loads(
                document,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("lambda policy artifact is not valid strict JSON") from error
        return cls.from_dict(_mapping(payload, "artifact"))

    @classmethod
    def load(cls, path: str | Path) -> LambdaPolicyArtifact:
        """Load one local policy artifact without network access."""

        try:
            with Path(path).open("rb") as stream:
                payload = stream.read(MAX_POLICY_ARTIFACT_BYTES + 1)
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read lambda policy artifact: {path}") from error
        if len(payload) > MAX_POLICY_ARTIFACT_BYTES:
            raise ValueError(
                f"lambda policy artifact exceeds {MAX_POLICY_ARTIFACT_BYTES:,} UTF-8 bytes"
            )
        try:
            document = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"cannot read lambda policy artifact: {path}") from error
        return cls.from_json(document)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LambdaPolicyArtifact:
        """Validate and construct a version-1 policy artifact."""

        _strict_fields(
            payload,
            {
                "artifact_version",
                "numeric_convention",
                "lambdas",
                "predictor_sha256",
                "tuning",
            },
            "artifact",
        )
        lambdas_payload = _mapping(payload["lambdas"], "lambdas")
        if len(lambdas_payload) > len(BudgetTier):
            raise ValueError("lambdas contains too many budget tiers")
        lambdas: dict[BudgetTier, Fraction] = {}
        for raw_tier, value in lambdas_payload.items():
            try:
                tier = BudgetTier(raw_tier)
            except ValueError as error:
                raise ValueError(f"unknown budget tier {raw_tier!r}") from error
            lambdas[tier] = _fraction(value, f"lambdas.{raw_tier}", nonnegative=True)

        tuning = _mapping(payload["tuning"], "tuning")
        _strict_fields(
            tuning,
            {
                "data_sha256",
                "replay_sha256",
                "prediction_sha256",
                "example_count",
                "domains",
                "ledger_adapter",
                "tier_specs",
                "candidate_sets",
            },
            "tuning",
        )
        domains = tuning["domains"]
        if not isinstance(domains, list) or any(not isinstance(item, str) for item in domains):
            raise ValueError("tuning.domains must be an array of strings")
        specs_payload = tuning["tier_specs"]
        if not isinstance(specs_payload, list):
            raise ValueError("tuning.tier_specs must be an array")
        if len(specs_payload) > len(BudgetTier):
            raise ValueError("tuning.tier_specs contains too many entries")
        specs = []
        for index, value in enumerate(specs_payload):
            item = _mapping(value, f"tuning.tier_specs[{index}]")
            _strict_fields(item, {"tier", "budget_limit", "weight"}, "tier_spec")
            if not isinstance(item["tier"], str) or not isinstance(item["budget_limit"], str):
                raise ValueError("tier spec tier and budget_limit must be strings")
            weight = _fraction(item["weight"], "tier_spec.weight")
            if weight <= 0:
                raise ValueError("tier spec weight must be positive")
            try:
                float_weight = float(weight)
            except OverflowError as error:
                raise ValueError("tier spec weight must fit a finite float") from error
            if float_weight <= 0 or Fraction.from_float(float_weight) != weight:
                raise ValueError("tier spec weight must use the exact finite-float encoding")
            specs.append(
                TierSpec(
                    BudgetTier(item["tier"]),
                    as_cost(item["budget_limit"]),
                    float_weight,
                )
            )

        candidates_payload = tuning["candidate_sets"]
        if not isinstance(candidates_payload, list):
            raise ValueError("tuning.candidate_sets must be an array")
        if len(candidates_payload) > len(BudgetTier):
            raise ValueError("tuning.candidate_sets contains too many entries")
        candidate_sets = []
        for index, value in enumerate(candidates_payload):
            item = _mapping(value, f"tuning.candidate_sets[{index}]")
            _strict_fields(
                item,
                {
                    "tier",
                    "values",
                    "total_derived_values",
                    "exhaustive",
                    "strategy",
                    "observed_breakpoint_count",
                },
                "candidate_set",
            )
            values = item["values"]
            if not isinstance(item["tier"], str) or not isinstance(values, list):
                raise ValueError("candidate set tier must be a string and values an array")
            if len(values) > MAX_POLICY_CANDIDATES_PER_TIER:
                raise ValueError(
                    "candidate set exceeds the policy artifact per-tier candidate limit "
                    f"({MAX_POLICY_CANDIDATES_PER_TIER:,})"
                )
            candidate_sets.append(
                LambdaCandidateSet(
                    tier=BudgetTier(item["tier"]),
                    values=tuple(
                        _fraction(candidate, "candidate_set.value", nonnegative=True)
                        for candidate in values
                    ),
                    total_derived_values=item["total_derived_values"],  # type: ignore[arg-type]
                    exhaustive=item["exhaustive"],  # type: ignore[arg-type]
                    strategy=item["strategy"],  # type: ignore[arg-type]
                    observed_breakpoint_count=item[  # type: ignore[arg-type]
                        "observed_breakpoint_count"
                    ],
                )
            )

        return cls(
            artifact_version=payload["artifact_version"],  # type: ignore[arg-type]
            numeric_convention=payload["numeric_convention"],  # type: ignore[arg-type]
            lambda_by_tier=lambdas,
            predictor_sha256=payload["predictor_sha256"],  # type: ignore[arg-type]
            tuning_data_sha256=tuning["data_sha256"],  # type: ignore[arg-type]
            tuning_replay_sha256=tuning["replay_sha256"],  # type: ignore[arg-type]
            prediction_sha256=tuning["prediction_sha256"],  # type: ignore[arg-type]
            training_example_count=tuning["example_count"],  # type: ignore[arg-type]
            training_domains=tuple(domains),
            tier_specs=tuple(specs),
            ledger_adapter_name=tuning["ledger_adapter"],  # type: ignore[arg-type]
            candidate_sets=tuple(candidate_sets),
        )
