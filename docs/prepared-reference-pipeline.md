<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepared reference policy pipeline

The implementation is `tierroute.policies.prepared_reference`. It connects the
bounded [prepared raw-score reference](prepared-reference-execution.md) to tierroute's
existing per-model isotonic calibration, exact/bounded lambda search, and
`OfflineSimulator`. The returned learned value is the existing
`NestedLodoLambdaResult`; this module does not define a parallel budget, decision, or
report schema.

This is proof-oriented, in-memory reference code for synthetic and frozen fixtures.
It is not a scalable or persistent prepared session, a native execution protocol, an
all-domain deployable predictor, a CLI path, or performance evidence.

A later bounded consumer in `tierroute.policies.native_prepared_benchmark` applies the
same calibration/lambda/report semantics to caller-pinned native prepared results, then
runs the learned policy and all six baselines with fixed per-query accounting. It is a
separate API and does not enlarge this reference module's scope or create an all-domain
artifact, shipped command/trainer, official-data result, or performance claim.

## Public entry point

```python
evaluate_prepared_reference_pipeline(
    examples: tuple[EvaluationExample, ...],
    tier_specs: tuple[TierSpec, ...],
    store: PreparedFeatureStore,
    raw_scores: PreparedRawScoreBundle,
    ledger_factory: BudgetLedgerFactory,
    *,
    expected_source_fit_sha256: str,
    expected_store_sha256: str,
    expected_raw_score_sha256: str,
    max_candidates_per_tier: int = 257,
) -> PreparedReferencePipelineResult
```

The three expected digests are mandatory trusted expectations. Supplying values read
from the same untrusted objects satisfies the function signature but provides no
independent substitution detection. SHA-256 values here are deterministic content
identities, not signatures, origin authentication, or proof of correct derivation.

`estimate_prepared_reference_pipeline(...)` exposes the reviewed preflight without
reading raw-score or target rows. A prepared execution estimate may be supplied so the
aggregate estimate retains the larger of the logical-plan and actual reference-
execution work/storage values. Plan-only calls cannot know the decimal width of model
costs, so their `candidate_evidence_upper_bound_bytes` is `None`. The public evaluator
adds one exact `LambdaSearchPreflightEstimate` per outer fold before it admits any
candidate-evidence byte claim.

## Exact graph mapping

Let `D` be the number of canonical domains, `N` the total example count, and `M` the
sorted model count. For each calibrated training-domain set `S` of size `D-2` or
`D-1`:

1. For every calibration domain `c in S`, read raw block
   `(training=S-{c}, scored=c)`.
2. Join the block's canonical example IDs to targets from domain `c` only.
3. Fit one `IsotonicCalibrator` per sorted model over the complete inner-LODO rows.
4. For every destination `h not in S`, read raw block `(training=S, scored=h)` and
   apply only the calibrators fitted on `S`.

This yields:

| Quantity | Formula | `D=7` |
| --- | ---: | ---: |
| unique calibrated subsets | `C(D,2) + D` | 28 |
| calibration row memberships | `C(D,2)N` | `21N` |
| calibration scalar points | `C(D,2)NM` | `21NM` |
| calibrated destination blocks | `D^2` | 49 |
| calibrated prediction rows | `DN` | `7N` |
| calibrated prediction cells | `DNM` | `7NM` |
| raw rows read by this layer | `[C(D,2)+D]N` | `28N` |

The raw bundle still contains its full 63-subset, 154-block, `22N`/`22NM` structure at
seven domains. This layer reads only the blocks needed for calibration and nested
policy replay; it does not eagerly expand the full raw bundle into Python mappings.

## Ordering contract

Two different orders are intentional:

- calibration inputs follow canonical plan-domain order and canonical example-ID
  order inside each scored-domain shard;
- lambda tuning and the final global replay preserve the caller's original example
  order.

The model axis is always sorted model-ID order. Tier order remains the supplied
`TierSpec` order. The final outer predictions are joined by private example ID and
replayed once globally; per-fold reports are never concatenated with fresh ledgers.

Prepared predictors implement the existing prompt-batch protocol because the ordinary
evaluation interface intentionally does not expose private example IDs. Repeated
prompts inside one known batch remain positionally joined. If two indistinguishable
prompt batches would require different precomputed rows, the reference fails closed
instead of guessing. A future prepared-session API should use an explicitly reviewed
private evaluation join rather than weakening the deployable predictor protocol.

## Validation and admission order

Before the first `score_row()`, `target_row()`, isotonic fit, lambda-root
materialization, ledger construction, or router replay, the builder performs:

1. exact argument, trusted-digest format, and candidate-cap validation;
2. canonical plan reconstruction plus exact aggregate child counts, child types, and
   numeric payload lengths before recursive child traversal;
3. aggregate plan-only calibration/policy admission and cheap stored parent-identity
   comparison with the trusted source/store/raw expectations;
4. one bounded immutable replay-scope snapshot, followed by an independently
   recomputed source-fit identity comparison;
5. a cost-width-aware lambda-search estimate for every outer fold, followed by
   aggregate work, pair-scan, utility, and lambda candidate/policy-artifact
   admission;
6. recursive reconstruction of the plan, store, coefficient blocks, feature shards,
   and raw-score blocks so stale `init=False` hashes cannot conceal post-construction
   mutation;
7. repeated trusted source/store/raw comparison and exact row/domain/prompt/model
   joins over the reconstructed values.

The post-processing estimate includes modeled PAV sorting/linear work, calibrated
application, all outer-fold pair scans and candidate utility replays, retained report
rows, the lambda candidate/policy-artifact estimate, target copying, prediction tables, and calibrator numeric
state. Pair work counts all five current implementation traversals per outer fold:
prepared admission, nested admission, tuning admission, derivation admission, and
root materialization. Current ceilings are:

| Guard | Limit |
| --- | ---: |
| aggregate modeled work | 100,000,000 units |
| modeled numeric storage | 512 MiB |
| aggregate lambda pair scans | 10,000,000 |
| aggregate lambda utility evaluations | 100,000,000 |
| retained report rows | 1,000,000 |
| cost-width-aware aggregate lambda candidate/policy-artifact estimate | 8 MiB |
| candidate cap per tier | 2 through 257 |
| samples in one calibrator | existing 500,000-point limit |

These are reviewed logical/numeric admission units. They exclude Python object headers,
allocator fragmentation, caller-owned objects, interpreter state, and other resident
memory. They are not peak-RSS or wall-clock promises. The ordinary lambda preflight
continues to enforce its exact-rational peak and policy-artifact bounds independently.
The work and memory of an arbitrary caller-supplied `ledger_factory`, including its
side effects, are also outside these code-owned bounds. Existing audited per-query and
cumulative adapters are exercised for semantic parity; passing another callback does
not make that callback audited or resource-bounded by this estimate.

No `allow_large_exhaustive` bypass is exposed. A cap can truncate the exact candidate
set; every retained `LambdaCandidateSet` records that fact, and
`PreparedReferencePipelineResult.all_searches_exhaustive` is derived from all outer
fold/tier records. A false value forbids an exact-optimum claim even when the selected
route happens to match an exhaustive run.

## Evidence records

The result retains:

- a per-domain target-shard identity over canonical IDs, sorted models, and exact
  binary64 targets;
- one calibration record for every `D-2` and `D-1` training subset, including exact
  raw-block indices/hashes, target-shard hashes, calibrators, and a versioned identity;
- one target-free calibrated-score record for every `D^2` destination context,
  including its calibration identity, raw block, feature shard, row/model keys, and
  calibrated score identity;
- source-fit, store, raw-bundle, evaluation-data, and replay-order identities;
- the ridge value, versioned prepared solver/scorer IDs, embedding configuration, and
  ordered coefficient-block, scored-feature-shard, and raw-score-block digest
  catalogues;
- the real `NestedLodoLambdaResult`, including exact candidate fractions, selected
  lambdas, decisions, calls, costs, budgets, fold tuning evidence, report, score, and
  global outer-prediction identity.

Result constructors validate canonical graph coverage and cross-record joins, but
direct construction still proves only self-declared consistency. The public builder is
the supported derivation path.

## Parity boundary

Prepared Welford/Chan moment reduction and the rowwise trainer use different floating-
point operation orders. Raw coefficients, raw scores, and some isotonic upper bounds
can therefore differ by small amounts, and their numeric digests are not promised
across Python versions or platforms. The implementation does not introduce an epsilon
into PAV grouping, exact lambda roots, or routing ties.

On the bundled four-domain synthetic replay, the prepared bridge and authoritative
rowwise path produce the same complete `NestedLodoLambdaResult`, including the
same-runtime outer prediction identity. An uneven seven-domain fixture also matches all seven fold
results, candidate evidence, selected lambdas, decisions, accounting, and final report
while exercising the full 63/154/`22N` graph. Constant, bounded-candidate, replay-order,
one-ULP, quoted-versus-realized-cost, lineage-tamper, post-construction mutation,
cumulative-ledger, intermediate five/six-domain, and held-out-target noninterference
tests cover the principal boundaries. Generated numeric digests are not fixed as
cross-runtime goldens.

These stable fixtures are regression evidence, not a theorem that tolerance-close raw
scores must always lead to identical PAV partitions or exact decisions. Any future
official-data parity gate must compare PAV partitions, exact candidates, lambdas, and
reports directly and fail closed on divergence.

## Deliberate non-claims

This reference establishes bounded end-to-end wiring only. It does not establish:

- RouterBench or SK Telecom data execution;
- `bge-m3` provider availability or model-weight redistribution rights;
- scalable, persistent, native, or cross-platform prepared-session artifacts;
- an all-domain final predictor or `LambdaPolicyArtifact`;
- CLI/runtime integration;
- latency, memory, quality, oracle-gap, or cost improvement;
- an official per-query or cumulative budget interpretation.

The caller still chooses the budget-ledger adapter. Tests prove that the prepared
policy bridge preserves both existing per-query and cumulative adapter behavior; they
do not resolve the organizer's budget semantics. The separate native high-level
consumer deliberately fixes `PerQueryBudgetLedger`; it is not cumulative or cascade
evidence.
