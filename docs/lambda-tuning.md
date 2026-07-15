<!-- SPDX-License-Identifier: Apache-2.0 -->

# Exact tier-lambda tuning design

This note explains the P0 one-shot policy, why its exhaustive search is finite, and
where evaluation-only information is kept. It describes implemented behavior; it does
not claim that the bundled synthetic result is a benchmark score.

## Runtime decision and numeric convention

For an affordable model `m`, tierroute computes

```text
utility(m, lambda) = predicted_quality(m) - lambda * quoted_cost(m)
```

Predictions are finite Python floats and are converted with `Fraction.from_float`.
Quoted costs are exact `Decimal` values and are converted directly to `Fraction`.
Lambda is stored as a reduced rational numerator/denominator. No cost passes through a
binary float, which avoids overflow and `0 * inf -> NaN` for valid values such as
`Decimal("1e10000")`.

Budget aggregation does not use ambient-context `Decimal` operators. Project-owned
helpers align decimal coefficient/exponent tuples with Python integers, so addition,
subtraction, and integer scaling are exact even if a caller changes global decimal
precision. A learned mean quote that is mathematically repeating uses an explicit
half-even precision contract instead of inheriting caller state.

The shared runtime/evaluator selection order is:

1. greater exact utility;
2. lower exact quoted cost;
3. lexicographically smaller model ID.

After one call, the one-shot router selects that recorded output. Cascade or a second
call is outside this policy.

## Why the exhaustive candidate set is complete

For a fixed prompt and model, utility is an affine function of non-negative lambda.
Two model orderings can change only at

```text
lambda = (quality_i - quality_j) / (cost_i - cost_j)
```

when the costs differ and the result is non-negative. Tierroute derives these roots
from every model pair in the catalogue. It does not assume that `TierSpec.budget_limit`
is the maximum balance a ledger can expose; the ledger and simulator remain the sole
authorities on affordability.

The exhaustive set contains:

- lambda zero;
- every non-negative pairwise root;
- the midpoint of every adjacent pair of roots;
- one value greater than the final root.

Boundaries exercise the exact cost/model-ID tie-break. Midpoints cover every open
interval, and the last value covers the unbounded tail.

This remains complete for a cumulative ledger. Inside one root interval the first
query selects the same model, so it incurs the same realized charge and produces the
same next ledger state. Applying the same argument query by query proves that the
entire decision, charge, and remaining-budget trajectory is constant in that interval.

`max_candidates_per_tier=None` materializes and evaluates this complete exact set. The
default capped path instead keeps memory bounded while it streams every non-negative
pairwise breakpoint occurrence:

1. retain a deterministic bottom-hash sample of unique roots plus the minimum and
   maximum root;
2. derive boundaries, adjacent midpoints, and the tail from only those retained roots;
3. rank-space that derived set to at most the configured cap.

When every unique root and derived candidate fits the cap, this path has still retained
the complete set and records `exhaustive: true` with its exact count. If either stage is
truncated, the result is approximate even though every retained number is exact. That
case records `exhaustive: false`, its bounded-search strategy, the observed breakpoint
occurrence count, and an unknown (`null`) complete candidate count; it is not presented
as a global continuous optimum. The uncapped path is the way to request exhaustive
coverage, subject to the resource preflight below.

## Exhaustive-search resource preflight

With `n` examples, `m` models, and `T` tiers, at most `n * choose(m, 2)` unequal-cost
pair occurrences can contribute roots. Boundaries, midpoints, and the tail therefore
produce at most twice that count plus two candidates. Replaying every candidate has
worst-case work `O(T * n^2 * m^3)`.

Before materializing roots, an uncapped search computes conservative upper bounds and
refuses the run when either exceeds:

- 100,000 retained candidates; or
- 100,000,000 model-utility evaluations.

Duplicate or negative roots make real work smaller, so the Python API offers the
explicit `allow_large_exhaustive=True` acknowledgement for a reviewed false positive.
The CLI deliberately requires two flags:

```bash
tierroute train ... \
  --exhaustive-lambda-search \
  --allow-large-exhaustive-search
```

The second flag is invalid without the first. Capped and explicit-grid searches bypass
the guard. At the pinned RouterBench shape (34,778 rows, 11 models, three tiers), the
conservative bounds are 3,825,582 candidates and 4,390,520,996,268 utility evaluations;
use `max_candidates_per_tier=257` for the practical nested-LODO path and report its
artifact-labeled exhaustive/truncated status.

## Direct metric tuning

Each retained lambda is run through `OfflineSimulator` with the caller-selected ledger.
The tuner uses realized replay quality and charge, not the predictor's training loss.
An incomplete or over-budget replay is ineligible. Feasible ties are resolved by:

1. greater exact mean realized quality;
2. lower realized spend;
3. smaller exact lambda.

The primary score is a positive-weighted mean of tier qualities. Each tier owns an
independent ledger and its lambda affects only that tier, so

```text
argmax_(lambda_fast, ...) sum_t weight_t * quality_t(lambda_t)
```

decomposes into one independent maximum per tier. This is the same answer as a
Cartesian-product search over the retained finite grids without its exponential cost.
It is a full exact finite joint optimum only when those grids are marked exhaustive;
truncated capped grids remain approximate as described above.

The ledger factory receives both the configured limit and replay query count. An
adapter therefore decides whether the limit is fixed-total, per-query, or pooled.
Neither the tuner nor nested LODO silently rescales it. Until SK Telecom confirms the
official interpretation, policy creation requires an explicit `per-query` or
`cumulative` choice and records that adapter identity.

## Leakage boundary and nested LODO

Ordinary deployable routers receive prompt, tier, remaining budget, call history, and
candidate models. They never receive replay example ID, split domain, uncalled output,
realized cost, or ground-truth quality.

Private example IDs are used only by a nominal evaluation-only router to join
cross-fitted predictions to logged outcomes:

```text
outer training rows
  -> inner LODO predictor fits
  -> one OOF prediction per (example_id, model_id)
  -> lambda tuning through the simulator
  -> predictor refit on all outer training rows
  -> prediction on the untouched outer domain
```

After every outer fold, tierroute replays all outer-OOF predictions once in the
original full-dataset order. Concatenating reports from separately reset folds would
be invalid for cumulative accounting.

Generic nested lambda evaluation requires at least three domains. The current
calibrated bilinear trainer itself performs inner LODO calibration, so a fully nested
run generally needs at least four domains: one outer holdout, one lambda-tuning
holdout, and at least two domains for the predictor's calibration fit.

## Artifact provenance

Predictor and policy state use strict canonical JSON; pickle and unknown fields are
rejected. A policy artifact records:

- the canonical predictor artifact SHA-256;
- an order-independent hash of training/metric-relevant replay content;
- an order-sensitive hash of that replay content in evaluation order;
- the ordered OOF prediction hash as audit metadata;
- example count and domains;
- exact tier specs and ledger adapter identity;
- selected exact lambdas, retained candidate counts, search strategy, and observed
  breakpoint occurrence counts. The complete derived-candidate count is `null` for a
  truncated bounded search because it was intentionally never materialized.

Artifact-backed CLI routing fails closed on a predictor, dataset, replay order, model
catalogue, or tier-spec mismatch. A cumulative policy additionally requires the caller
to provide the current exact `--remaining-budget`. Routing cannot recompute the OOF
prediction hash from a final fitted predictor; verifying that field requires
reproducing the cross-fitted table and comparing its digest. It is not an authenticated
claim by itself.

Training serializes and round-trip-validates both documents before writing, rejects
input/output aliases, stages each file under a random exclusive name in its destination
directory, and replaces the policy last. Existing files are backed up and both paths
are rolled back on an ordinary write or verification failure. Two unrelated POSIX
pathnames cannot be switched in one filesystem operation, so concurrent writers remain
unsupported; a transient mixed pair fails the policy-to-predictor hash check.
