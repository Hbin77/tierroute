<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluation scope identity

## Purpose

Cross-report metrics are meaningful only when every policy was evaluated on the same
ordered replay and protocol. Query IDs, tier names, and ledger totals are not enough:
two replays can reuse those shallow fields while changing a quality label, output,
cost, candidate order, or router-visible metadata. `EvaluationReport` therefore
requires an `EvaluationScopeIdentity(algorithm, sha256, max_calls_per_query)` value.
The digest must be lowercase SHA-256 text, and the positive call cap is part of the
same immutable comparison identity.

The current algorithm identifier is `tierroute-evaluation-scope-v1`. It is an
accidental-mix and reproducibility identity, not an authenticated signature. A caller
constructing schemas directly can copy or lie about an otherwise well-formed digest;
persisted reports still need a trusted provenance and transport boundary.

## Scope-v1 input

The digest covers the following values in their supplied order:

1. The algorithm identifier and positive `max_calls_per_query`.
2. Every ordered `TierSpec`: tier value, canonical exact budget, and finite binary64
   weight encoded with `float.hex()`.
3. Every ordered `EvaluationExample`:
   - private example ID, prompt, and split domain;
   - canonical router metadata;
   - candidate models in original order, including model ID, canonical quoted cost,
     optional display name, and canonical model metadata;
   - outcomes in original order, including model ID, complete output text, canonical
     realized cost, and finite binary64 quality.

Router names, actions, selections, predictions, and result rows are excluded. They are
the policy-dependent values that the metric is intended to compare. Outcome order is
included conservatively even though the current simulator indexes outcomes by model
ID. Candidate order must be included because it is visible in `RouterState`.

Tier budget, weight, and order are included in the digest, while each report's tier and
budget records are also checked structurally. This defense in depth detects corrupted
reports even when a digest was copied incorrectly.

## Canonical encoding and immutable replay snapshot

The simulator first copies tier specs, replay costs, and router/model metadata into a
canonical immutable replay snapshot. Exact built-in dictionaries with plain string
keys become sorted immutable mappings. Lists and tuples both become tuples, removing
source mutability, alias topology, and container-type ambiguity before the router
observes them. Mapping proxies, dictionary
subclasses, and other custom mappings are rejected because reading them can dispatch
user-defined code. Supported leaf values are `None`, `bool`, plain `int`, finite plain
`float`, plain `str`, and finite plain `Decimal`. Decimal trailing zeros and all zero
encodings are normalized exactly.

Every metadata value is encoded with an explicit type tag. Integers use hexadecimal
text, floats use `float.hex()`, and Decimals use sign, coefficient digits, and exponent.
Scope-v1 streams typed tokens directly into SHA-256: each token has a two-byte unsigned
tag-length, the ASCII tag, an eight-byte unsigned payload-length, and the payload;
sequence counts use an eight-byte unsigned payload. Text is strict UTF-8. Costs use the
same context-independent `canonical_cost_text()` representation as the rest of the
project. Candidate, realized, and tier-budget costs are canonicalized in the same
immutable snapshot used for replay, so a representation-sensitive policy cannot see a
different cost representation under the same digest. Tier weights and outcome quality
become plain binary64 values for scope semantics. This length-delimited stream is
unambiguous without materializing a second full replay document. `repr`, pickle,
custom encoders, numeric subclasses, and user-defined serialization hooks are never
used at the scope boundary.

The metadata snapshot fails before routing when it encounters:

- a custom or unsupported container, non-string mapping key, invalid Unicode, or
  nonfinite number;
- a cycle or nesting deeper than 32;
- more than 100,000 values in one metadata root;
- more than 8 MiB of key/value and canonical numeric payload in one metadata root;
- an integer wider than 1,000,000 bits; or
- a Decimal outside the project's 100,000-position exact resource range.

Across one replay snapshot, metadata is additionally capped at 10,000,000 logical
values and 256 MiB of encoded payload. The complete scope—including IDs, prompts,
outputs, display names, costs, qualities, weights, and metadata—is capped at 1 GiB of
logical encoded payload. Reusing one source object does not make repeated logical
occurrences free. Prompts and outputs are hashed in full within this aggregate limit;
the JSON adapter does not impose a separate prompt/output cap.

## Comparison and derived-evidence contract

`weighted_delta()` and `oracle_gap_recovery()` compare the complete identity—algorithm,
digest, and call cap—before numeric work, then require identical ordered tiers,
budgets, weights, adapter names, effective limits, and query order. The digest does not
attempt to hash executable ledger code. Adapter semantics remain localized and are
evidenced through the ledger's structured report.

`BaselineResult` recomputes its score, quote-error summary, and exact realized total
from its own report. `LodoSixBaselineEvaluation` requires the six canonical rows in
order, one scope, one original query order, valid outer-fold partitions, per-query
accounting, scope-bound fold evidence, candidate-model membership, unique fitted-table
keys, and an exact recomputation of every oracle-gap value from the suite's own
cheapest and oracle reports. Rounded CLI values are presentation output and must not
be used to reconstruct authoritative derived fields.

## Versioning

The older `evaluation_data_sha256()` and `evaluation_replay_sha256()` algorithms remain
byte-compatible because predictor and policy artifacts already depend on them. They do
not substitute for scope-v1. Any change to scope-v1 fields or canonical bytes requires
a new algorithm identifier, updated CLI evidence, a changelog entry, and golden tests
that preserve the old algorithm's known vectors.
