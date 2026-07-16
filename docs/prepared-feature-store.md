<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepared feature-store reference contract

Status: experimental, bounded, in-memory Python reference implementation. It defines
canonical fit-relevant rows, reusable per-domain centered moments, and
training-subset isolation for the [prepared nested-LODO graph](prepared-session-graph.md).
It is not a production session, a persistent cache, or a replacement for the current
predictor-training path.

The implementation is `tierroute.predictors.prepared_store`. It performs no network
operation, embedding inference, file I/O, native execution, ridge solve, scoring,
calibration, or lambda selection.

A separate bounded
[prepared reference execution](prepared-reference-execution.md) now consumes these
records to solve all unique graph subsets and emit all raw-score blocks on small
fixtures. That downstream evidence does not enlarge this store module's scope or make
either module a scalable or persistent prepared session.

## Public workflow

The intended construction order is:

1. Build a `PreparedNestedLodoPlan` with the exact domain counts, raw feature width,
   and target count.
2. Compute `prepared_fit_source_sha256(examples, plan)` from the independently
   trusted, normalized fit source and retain that expected value.
3. If embeddings are used, construct one `PreparedEmbeddingInput` per example and
   call `build_prepared_embedding_snapshot(rows, identity, dimension=E)`.
4. Call `build_prepared_feature_store(...)`, passing the source expectation and, when
   embedded, the snapshot plus its independently expected SHA-256 value.
5. Call `build_prepared_domain_statistics(store)` once.
6. Call `combine_prepared_subset_statistics(bundle, subset_index)` for any training
   subset enumerated by the graph.
7. Optionally pass the exact store/statistics pair to
   `build_prepared_coefficient_bundle(...)` from the prepared execution module.
8. Optionally pass the exact store/coefficient pair to
   `build_prepared_raw_score_bundle(...)`; its separate contract and tighter
   cumulative caps apply.

The builders and result types are:

```text
PreparedEmbeddingInput(example_id, prompt_sha256, values)

build_prepared_embedding_snapshot(rows, identity, *, dimension)
    -> PreparedEmbeddingSnapshot

prepared_fit_source_sha256(examples, plan) -> str

build_prepared_feature_store(
    examples,
    plan,
    *,
    embedding_snapshot=None,
    expected_embedding_sha256=None,
    expected_source_fit_sha256,
) -> PreparedFeatureStore

build_prepared_domain_statistics(store)
    -> PreparedDomainStatisticsBundle

combine_prepared_subset_statistics(bundle, subset_index)
    -> PreparedSubsetStatistics
```

Inputs use exact tuples and exact reviewed dataclass types. The builders reject
duplicate IDs, invalid or oversized text, non-finite binary64 values, inconsistent
domain counts, inconsistent model catalogues, incorrect dimensions, and malformed
lowercase SHA-256 values. Examples, embedding rows, model IDs, and domains are
canonicalized by Python string order where their owning builder defines an order.

`PreparedFeatureStore.feature_row(index)` and `target_row(index)` expose private tuple
copies for reference validation. They are not a streaming or zero-copy interface.

The fit-source digest binds canonical example IDs, full prompts, validation domains,
sorted model IDs, and canonical binary64 quality labels under the plan identity. It
excludes costs, outputs, display names, and metadata because those fields cannot affect
predictor fitting. The store builder recomputes it and requires equality with
`expected_source_fit_sha256` before feature extraction or dense allocation. Calling
the helper on an untrusted object and immediately trusting its result is only a
content check, not independent source authentication.

## Precomputed embedding boundary

This module never calls an `EmbeddingProvider`. Embeddings must be computed outside
this path and supplied as `PreparedEmbeddingInput` values containing:

- the exact `example_id`;
- `SHA-256(prompt.encode("utf-8"))` for the exact prompt used to compute the row; and
- one finite, fixed-width binary64 tuple.

`build_prepared_embedding_snapshot` sorts rows by `example_id`, requires unique IDs
and one declared width, canonicalizes numerical zero through the input type, and packs
the values as immutable little-endian binary64 bytes. Its `EmbeddingIdentity` records
provider, model ID, revision, pooling, normalization, and asset-manifest digest.

An embedded feature store requires `expected_embedding_sha256`. The store builder
checks all of the following before constructing its dense output payload:

- the supplied snapshot digest equals the caller-provided expected digest;
- snapshot IDs exactly equal the canonical example IDs, with no missing or extra row;
- snapshot prompt digests exactly equal the digests recomputed from the examples; and
- the plan feature width equals `12 + snapshot.dimension`.

The expected digest must come from a separately trusted workflow. Passing a digest
read from the same untrusted object only restates that object's claim. A surface-only
store uses `E = 0`, accepts neither a snapshot nor an expected embedding digest, and
requires plan width 12.

## Fixed universal raw layout

Let `E` be the embedding width. Every stored raw feature row has exactly `12 + E`
binary64 values in this order:

| Columns | Values |
|---|---|
| `0..2` | `log1p(character_count)`, `log1p(word_count)`, `log1p(line_count)` |
| `3..4` | `has_code`, `has_math` as zero or one |
| `5..11` | fixed tag indicators in the order below |
| `12..12+E-1` | precomputed embedding values |

The fixed tag catalogue is:

```text
code, finance, general, law, math, medicine, science
```

Code publishes this contract as `SURFACE_FEATURE_ALGORITHM_ID` and
`SURFACE_DOMAIN_TAG_CATALOGUE`. Any semantic extractor or catalogue change must bump
the algorithm identity rather than silently reinterpreting existing store digests.

The global store deliberately keeps every fixed tag column. It does not fit a dynamic
tag vocabulary across all domains. Dynamic tag selection happens only when a graph
training subset is combined.

Target rows contain realized quality values in canonical sorted `model_id` order.
Candidate-model costs, realized outcome costs, output text, display names, and metadata
are not predictor-fit inputs and are excluded from the feature/target payload and its
digest. Model IDs and quality labels are included.

The raw continuous values are not standardized in the store. A
`PreparedSubsetStatistics.feature_schema` supplies the population means and scales
derived from that subset alone. `active_feature_indices` maps its dynamic schema back
to the universal raw columns: the first five surface columns, included-domain active
tag columns, and all embedding columns.

## Digests identify content; they do not authenticate it

The exported algorithm identities are:

```text
tierroute.prepared-embedding-snapshot-v1
tierroute.prepared-feature-store-v1
tierroute.prepared-domain-statistics-welford-v1
tierroute.prepared-statistics-bundle-v1
tierroute.prepared-subset-statistics-chan-v1
```

Digest fields use namespace-separated SHA-256 over typed, length-framed fields.
Integers and numerical payloads have explicit little-endian encodings, so ambiguous
string concatenation is not part of the format.

The embedding snapshot digest binds its algorithm namespace, complete
`EmbeddingIdentity`, dimension, canonical `(example_id, prompt_sha256)` keys, and
binary64 payload. A direct feature-store record recomputes that snapshot digest from
its embedding columns and rejects a mismatched claim. The feature-store digest binds
the caller-checked fit-source digest, graph and surface algorithm IDs,
domain catalogue and counts, fixed tag catalogue, canonical model IDs, embedding
identity and snapshot digest when present, canonical row keys and domain indices, and
the feature and target payloads.

Per-domain content digests bind that domain's identity, shared feature/model/embedding
configuration, canonical row IDs and prompt digests, and the domain's feature and
target bytes. A domain-statistics digest additionally binds its counts, active-tag
mask, means, and centered moments. The bundle digest binds the global store digest and
all domain-statistics digests.

A bundle digest additionally binds the complete plan, model catalogue, embedding
configuration, store digest, and all domain-statistics digests. A subset's
`included_content_sha256` is derived only from the indices and content
digests of its included domains. Its final digest also binds the subset index, row
count, model IDs, active-tag mask, fitted continuous means/scales, and combined
moments, its plan, the complete feature-schema version, and embedding identity. This
separate included-content root is what permits the leakage invariant:
an excluded-domain row change can change the global store and bundle digests without
changing the subset statistics or subset digest.

These hashes are deterministic content identities. They support corruption or
substitution checks only when compared with a separately trusted expected value.
They do **not** prove:

- who produced or approved an embedding or dataset;
- that the declared embedding model, revision, pooling, or asset actually produced
  the supplied values;
- that an asset-manifest hash was obtained from a trusted publisher;
- data ownership, consent, license, or official SK Telecom provenance; or
- authenticity against an actor that can replace both content and expected digest.

Signatures, trusted distribution, source-license evidence, and dataset provenance
remain external responsibilities.

The documented builder sequence is the supported derivation path. Direct dataclass
constructors enforce bounded canonical structure and bind every supplied semantic
field, but they cannot prove that a self-declared source digest, domain-content digest,
or moment tuple was derived from trusted rows. Consumers must not treat a directly
assembled record as builder/provenance evidence merely because its self-computed digest
is internally consistent. Direct leaves are self-declared per-record canonical values,
not aggregate loaders, derivation proofs, provenance proofs, or evidence that a
builder's cumulative admission ran. Substitution detection requires comparison with a
separately trusted expected digest; hashes are not authentication.

Little-endian framing is canonical for the binary64 values actually stored. The
fit-source and precomputed-embedding fixtures have portable framing goldens because
they do not run feature arithmetic. No source-to-store or source-to-subset digest is
promised identical across platforms: `math.log1p`, `math.sqrt`, Welford, Chan,
`math.fsum`, and Python's Unicode/regular-expression behavior may change generated
feature bits or tags across runtime versions. Cross-platform cache/digest claims
require separate evidence; mathematical comparisons use declared tolerances.

## Welford domain moments and Chan subset combination

For each domain, let feature row `x` have width `d = 12 + E`, target row `y` have
width `M`, and row count be `n`. The reference stores:

```text
mu_x = mean(x)
mu_y = mean(y)
C_xx = sum((x - mu_x) (x - mu_x)^T)       # packed upper triangle
C_xy = sum((x - mu_x) (y - mu_y)^T)       # feature-major dense matrix
```

Rows are visited in canonical global example-ID order. For a new row, Welford's
centered update is:

```text
n'     = n + 1
delta_x = x - mu_x
delta_y = y - mu_y
mu_x'   = mu_x + delta_x / n'
mu_y'   = mu_y + delta_y / n'
C_xx'   = C_xx + delta_x (x - mu_x')^T
C_xy'   = C_xy + delta_x (y - mu_y')^T
```

The active-tag mask is the bitwise union of tags observed in that domain. The full
moments stay in universal raw coordinates even when a later subset activates fewer
tag columns.

To combine existing blocks `A` and `B`, define:

```text
n       = n_A + n_B
delta_x = mu_x_B - mu_x_A
delta_y = mu_y_B - mu_y_A
f       = n_A * n_B / n

mu_x = mu_x_A + delta_x * n_B / n
mu_y = mu_y_A + delta_y * n_B / n
C_xx = C_xx_A + C_xx_B + f * delta_x delta_x^T
C_xy = C_xy_A + C_xy_B + f * delta_x delta_y^T
```

`combine_prepared_subset_statistics` applies this Chan combination in ascending
canonical domain-index order and reads only the domains named by the graph subset.
For the first three continuous features it computes population scaling:

```text
scale_i = sqrt(C_xx[i, i] / n)
```

A zero scale is replaced by `1.0`. The active tag vocabulary is the fixed catalogue
filtered by the union of included-domain masks.

These formulas are mathematically equivalent to direct centered matrix accumulation,
and tests compare them approximately. They are not a bitwise-parity promise against
the existing row trainer. Welford, Chan, `statistics.fmean`, per-row standardization,
and dot-product accumulation use different binary64 operation orders. Centered
moments do not retain the old path's intermediate rounding history, so equal real
arithmetic can still produce different least-significant bits. The downstream
[prepared execution contract](prepared-reference-execution.md) records reviewed
fixture-specific coefficient, intercept, and raw-score tolerances. Those gates are
regression evidence only, not bitwise equality, a cross-platform numerical-digest
promise, or a universal error bound.

## Leakage invariants

For a fixed plan, model catalogue, feature algorithms, and embedding identity:

- subset means, variances, active tags, and moments use included domains only;
- excluded prompts cannot introduce an active tag or affect continuous scaling;
- excluded quality or embedding-row changes cannot affect included moments;
- excluded-domain content changes do not affect `included_content_sha256` or the
  subset digest;
- included prompt, quality, or embedding changes do affect the relevant domain
  content identity and subset result; and
- input order, candidate-model order, outcome order, and embedding-row order do not
  alter canonical payloads or digests.

Changing a shared input such as the plan, model catalogue, surface algorithm, or
embedding identity is not an excluded-domain-only mutation and is expected to change
the relevant identities.

The embedding statement is conditional on the supplied included embedding rows
remaining fixed. Provider execution is outside this module; a batch-coupled or
stateful preparation process that changes included embeddings after an excluded
prompt mutation will correctly change the included-domain digest rather than being
hidden as noninterference.

## Reference admission limits

These Python reference limits are deliberately tighter than the graph planner's
modeled compact-buffer limits:

| Limit | Value |
|---|---:|
| one prompt | 1 MiB UTF-8 |
| aggregate validated source text | 64 MiB UTF-8 |
| aggregate canonical row-key text per snapshot/store | 64 MiB UTF-8 |
| one row ID | 4 KiB UTF-8 |
| one model ID | 4 KiB UTF-8 |
| one snapshot/direct-store or combined embedded-build numeric admission | 512 MiB |
| domain-statistic scalars | 2,000,000 |
| domain-statistic numeric work units | 50,000,000 |

The graph still supplies the outer limits, including four through seven domains,
1,000,000 examples, 4,096 raw features, and 256 targets. Therefore `E` cannot exceed
`4,096 - 12`.

For `D` domains, `N` rows, raw width `d`, and `M` targets, the reference checks:

```text
embedding snapshot bytes = 8*N*E
direct feature-store bytes = 8*N*(d + M)
embedded store construction admission = 8*N*(E + d + M)

statistics scalars = D * (d + M + d*(d+1)/2 + d*M)
statistics work    = N * (3*(d + M) + d*(d+1)/2 + d*M)
```

Snapshot construction and a direct store each apply their own admission. Building an
embedded store additionally counts its caller-owned snapshot payload together with the
new feature and target payloads, so the two admissions are not independent in that
workflow. The store rejects this combined size before traversing source rows or
allocating dense feature and target outputs. The
statistics builder rejects scalar or work excess before allocating accumulators or
traversing rows.

Fit-source text and canonical row-key text have separate 64 MiB admissions. Snapshot
construction rejects aggregate row-key text before canonical sorting or digest work;
direct snapshot/store records reject it before computing their content digests.

These are admission guards, not peak-RSS guarantees. They omit Python object,
allocator, temporary-list, hashing, and caller-owned input memory. In particular, a
shape admitted by `prepared_graph` can still be rejected by this smaller Python
reference. The graph document's RouterBench-sized modeled shape is not a claim that
the reference statistics builder admits or executes that shape. The planned embedded
RouterBench construction exceeds this store workflow's combined 512 MiB admission,
and its statistics construction exceeds the 50,000,000-unit cap. The later prepared
execution layer also has its own 100,000,000-unit aggregate cap, so that planned shape
is rejected before full-dimensional reference execution.

## Exact exclusions and next parity gate

This slice does not implement:

- embedding-provider execution, model download, or any runtime network access;
- snapshot persistence, loading, mmap, cache eviction, locking, or path-race defense;
- a native prepared-session protocol or a new protocol magic/version;
- calibration, lambda tuning, routing, or reports inside the store layer itself;
- a final all-domain deployable predictor; or
- GBM preparation or the routing baselines.

The separate bounded
[prepared reference execution](prepared-reference-execution.md) now implements the
previously open coefficient and raw-score steps for small synthetic/frozen fixtures.
The bounded [prepared policy pipeline](prepared-reference-pipeline.md) consumes those
scores through calibration, lambda tuning, and final replay on frozen fixtures. Neither
successor adds persistence, native execution, scalable RouterBench execution, an
all-domain artifact, or a reportable performance result.

The default bilinear training and evaluation path remains unchanged. Before a
prepared execution path can replace any repeated fit, the complete parity gate must
use a frozen corpus and independently reviewed tolerances to compare, in order:

1. selected feature coordinates, training-only schemas, and sufficient statistics;
2. per-subset ridge coefficients and intercepts — now exercised by the bounded
   reference, not yet by a scalable replacement;
3. every prepared raw-score block — now exercised by the bounded reference, not yet
   by a persistent or native session;
4. isotonic calibration, exact lambda candidate/tie selection, tier decisions, and
   final per-fold/aggregate reports — now equal on stable bounded four- and seven-
   domain fixtures, not yet an official-shape scalable replacement.

The corpus must include ordinary data, high-dynamic-range numerical cases, zero
variance columns, permutations, and near-tie decisions. Any field claimed to be
bitwise stable must be identified separately from fields accepted under a documented
numeric tolerance. If byte-for-byte compatibility with the current row arithmetic is
required, the design must either replay the required row operations or move both
paths to one newly versioned arithmetic contract; the current centered moments alone
cannot reconstruct prior intermediate rounding.

Until that gate passes, these modules are evidence for canonicalization, isolation,
and bounded coefficient/raw-score parity. They are not evidence of speedup, cost
reduction, quality retention, calibration/lambda/final-report parity, full
RouterBench-scale execution, or production readiness. Issue #9 remains open.
