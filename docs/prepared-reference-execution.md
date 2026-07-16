<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepared reference execution contract

Status: experimental, bounded, in-memory, standard-library-only Python reference.
The implementation is `tierroute.predictors.prepared_execution`. It consumes the
canonical [prepared feature store](prepared-feature-store.md), its per-domain
sufficient statistics, and the immutable
[nested-LODO graph](prepared-session-graph.md). It then solves one coefficient block
per unique training subset and emits every graph-prescribed raw-score block.

This is executable structural and numerical-parity evidence for small synthetic and
frozen fixtures. It is not a scalable prepared session, a persistent cache, a native
protocol, or a completed predictor-training/reporting path. It does not change the
default trainer, evaluator, CLI, or routing policy. Issue #9 remains open.

## Supported workflow

The only supported derivation path is the builder sequence:

```text
PreparedFeatureStore
  -> build_prepared_domain_statistics(store)
  -> build_prepared_coefficient_bundle(store, statistics, ridge=...)
  -> build_prepared_raw_score_bundle(store, coefficients)
```

`build_prepared_scored_feature_shards(store)` is also public when a caller needs the
target-free per-domain scored-feature identities without constructing scores. The
complete result types are:

```text
PreparedReferenceExecutionEstimate
PreparedCoefficientBlock
PreparedCoefficientBundle
PreparedScoredFeatureShard
PreparedScoredFeatureShardBundle
PreparedRawScoreBlock
PreparedRawScoreBundle
```

The coefficient builder validates exact store/statistics lineage, canonical model and
embedding configuration, a finite positive binary64 ridge value, per-domain active-tag
masks, and the complete aggregate execution estimate before combining one moment
array or factorizing a matrix. It combines, solves, and discards each subset's moments
sequentially. One Cholesky factorization is shared by all model targets in that subset.

The raw-score builder requires the exact source-store identity retained by the
coefficient bundle. It recomputes the aggregate estimate from the retained coefficient
widths before the first feature-shard hash, feature-row read, score allocation, or dot
product. It then emits blocks in graph order, rows in canonical scored-domain example
order, and columns in sorted model-ID order. `score_row()` and
`iter_score_rows()` expose private tuple rows; there is intentionally no eager
all-rows object expansion. `example_ids_for_block()` supplies the corresponding row
join keys.

Exact built-in container and scalar types are part of this hostile-input boundary.
The records reject subclassed tuples, strings, bytes, integers, and floats where an
exact type is required, as well as malformed lengths, non-finite values, negative
zero, inconsistent parent identities, reordered children, duplicate global example
IDs, and invalid indices.

## Moment-to-coefficient arithmetic

For one combined training subset, let `A` be the active universal raw coordinates,
`s_i` the population scale for active coordinate `i`, `C_xx` the unnormalized
centered feature cross-product, and `C_xy` the unnormalized centered feature/target
cross-product. Only the first three surface coordinates use `s_i`; binary, tag, and
embedding coordinates use scale one. The reference constructs:

```text
G[i,j] = C_xx[A[i], A[j]] / s_i / s_j
H[i,m] = C_xy[A[i], m] / s_i

(G + ridge I) W[:,m] = H[:,m]
```

There is no division by the training row count: both centered moment arrays are sums,
and ridge is added in that same summed normal-equation scale. The active coordinates
are the five non-tag surface columns, the included training domains' active fixed-tag
columns, and every embedding column, all in canonical schema order.

The first three encoded means are zero after standardization. Noncontinuous encoded
means remain in raw coordinates, so each intercept is recovered as:

```text
intercept[m] = target_mean[m]
               - sum(noncontinuous_mean[i] * W[i,m])
```

Scoring standardizes only the first three coordinates and follows the ordinary Python
`sum` order used by `BilinearQualityPredictor`:

```text
raw_score[m] = sum(encoded[i] * W[i,m] for i in schema_order) + intercept[m]
```

The prepared solver and scorer deliberately have identities distinct from the row
solver and native dense-ridge protocol:

```text
tierroute.prepared-moment-ridge-cholesky-python-v1
tierroute.prepared-raw-dot-product-python-v1
```

Welford/Chan moments cannot reproduce every intermediate rounding operation of the
rowwise `fmean`/`fsum` trainer. Tests therefore compare independently refit row schemas,
coefficients, intercepts, and every raw-score row with reviewed tolerances. Ordinary
fixtures currently use coefficient gates `rel=1e-8, abs=1e-9`, intercept gates
`rel=1e-9, abs=1e-10`, and score gates `rel=1e-9, abs=1e-9`; the separate
zero-variance, constant-target, exactly-collinear, high-dynamic-range fixture uses the
explicit looser gates recorded in its test. These are regression tolerances, not
universal error bounds.

Every normal equation also passes an empirical residual guard whose factor is 2,048
times the dimension- and magnitude-scaled binary64 allowance. The factor is frozen by
ordinary and adversarial regression fixtures. It is a solver-corruption/failure guard,
not a theorem, a conditioning certificate, a forward-error bound, or the
cross-implementation parity criterion.

Numerical parity is tolerance-based only. The implementation promises neither bitwise
coefficient/score equality nor cross-platform equality of generated numerical
digests. Little-endian framing goldens prove canonical hashing of manually supplied
binary64 bits; they do not prove that different Python/runtime/CPU platforms generate
the same arithmetic bits.

## Complete graph structure

The reference consumes every training subset and score block enumerated by a valid
four-through-seven-domain nested-LODO plan. For `D` domains, each example occurs in
exactly `C(D,2) + 1` score blocks. At `D = 7` this is exactly 22 memberships per
example, so the reference materializes exactly:

```text
63 coefficient blocks
154 raw-score blocks
22N scored example-row memberships
22NM scalar scores for M model targets
```

The seven-domain regression uses unequal counts `(2, 1, 3, 1, 2, 1, 2)`, verifies
all 63 solves and 154 blocks, and proves that every canonical example ID occurs exactly
22 times. This is structural evidence only. It is not a wall-time, memory-efficiency,
quality, or cost-savings result.

## Content identity, lineage, and locality

Coefficient, scored-feature-shard, and raw-score records use namespace-separated,
typed, length-framed SHA-256 identities. Their canonical numerical payloads are
little-endian binary64. In outline:

- a coefficient block binds the plan/subset, included-domain statistic identities,
  schema and active coordinates, model IDs, ridge, weights, and intercepts;
- a coefficient bundle binds the source store, complete statistic bundle, embedding
  configuration, per-domain tag masks, ridge, and ordered block identities;
- a target-free feature shard binds its scored domain, row IDs, prompt hashes,
  universal feature configuration, embedding configuration, and raw feature bytes;
- a raw-score block binds its exact graph context, coefficient block, feature shard,
  sorted model catalogue, and row-major score bytes; and
- aggregate bundle identities bind their ordered child identities.

These hashes identify claimed content; they do not authenticate it. They do not prove
who produced a source, that an embedding identity generated the supplied values, that
statistics or coefficients were honestly derived, that data are licensed, or that a
record came from SK Telecom. Substitution detection requires comparison with an
expected digest obtained through a separately trusted channel. Recomputing and
trusting a digest from the same untrusted object is only an internal content check.
Signatures, trusted distribution, license evidence, and provenance remain external.

The public builders above are the only supported derivation path. Direct leaf
dataclass constructors validate bounded, canonical, per-record fields and compute an
identity over those self-declared values. They are not aggregate deserializers or
loaders, do not rerun cumulative execution admission, and are not evidence of
provenance or correct derivation. A self-consistent directly assembled leaf must never
be accepted as builder evidence merely because its computed digest is valid. Bundle
constructors add canonical parent/child consistency checks, but cannot retroactively
prove how a child was produced.

Locality is asserted only for the relevant individual records, not for global bundle
SHA-256 values. If one domain's targets or prompt-derived inputs change, aggregate
store/statistics/coefficient/raw-score bundle identities may change because other
subsets include that domain. For the coefficient block whose training subset excludes
that domain:

- changing only the excluded targets leaves the coefficient block unchanged, and its
  raw-score block on that domain also remains unchanged because scoring never reads
  targets; and
- changing the excluded prompt-derived features leaves the coefficient block
  unchanged but changes that scored-domain feature shard and therefore the raw-score
  block, as it should.

No global aggregate-SHA locality claim follows from these block-level invariants.

## Aggregate reference admission

The execution layer applies two cumulative ceilings before coefficient construction or
scoring begins:

| Admission | Ceiling |
|---|---:|
| deterministic modeled work units | 100,000,000 |
| modeled binary64 numeric storage | 512 MiB |

The work estimate accounts for subset combination; repeated immutable
copy/validation/hash scans; moment transformation; active-coordinate derivation;
factorization and multi-target solves; scored-domain row selection; feature hashing;
row decoding/encoding; dot-product positions; repeated coefficient unpacking; and
coefficient/score normalization, packing, validation, and hashing.

The modeled numeric-storage estimate includes the graph's feature, target, domain
statistics, coefficient, raw-score, and one-solve-workspace buffers plus one subset
statistics transient, the retained active-coordinate cache, and the largest modeled
copy transient. These are reviewed deterministic numeric admission units. They exclude
Python objects, container headers, allocator/process overhead, caller-owned source and
snapshot memory, and other implementation temporaries. They are not peak RSS,
wall-clock estimates, throughput claims, or complete machine-resource guarantees.

The planned RouterBench shape (`D=7`, `N=34,778`, `d=1,036`, `M=11`) is useful only
as planning input. The graph-only compact-buffer estimate admits its abstract shape,
but the earlier embedded store's combined 512 MiB admission, the 50,000,000-unit
statistics cap, and this 100,000,000-unit execution cap reject the full reference
workflow before full-dimensional execution. Thus no RouterBench execution or
performance claim follows from this module.

## Deliberate exclusions and open gate

This slice performs no network access, model inference, provider execution, download,
file I/O, persistence, mmap, locking, subprocess, or native-code execution. It does
not implement or prove:

- scalable/full-dimensional prepared execution or a persistent prepared session;
- isotonic calibration, lambda tuning/selection, near-tie policy decisions, or final
  all-domain artifact assembly;
- final evaluation-report parity or any benchmark quality, cost, oracle-gap, or
  routing result;
- runtime/CLI integration, a native prepared protocol, or cross-platform artifacts;
- bge-m3 execution, RouterBench execution, official SK Telecom data, or data
  redistribution rights; or
- GBM comparison, production readiness, wall-time speedup, or memory savings.

The next replacement gate starts at the raw-score outputs established here and must
prove calibration, exact lambda candidate/tie selection, routing decisions, and final
fold/aggregate reports on a frozen corpus. A scalable implementation additionally
needs authenticated offline embedding preparation, persistence/session semantics,
new protocol identities, fail-closed loading and resource controls, and three-platform
artifact/link audits. Until those gates pass, Issue #9 remains open and the current
rowwise training path remains authoritative.
