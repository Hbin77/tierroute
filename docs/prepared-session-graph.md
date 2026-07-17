<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepared nested-LODO graph contract

Status: experimental graph enumeration and resource preflight. A separate bounded
[in-memory feature-store and sufficient-statistics reference](prepared-feature-store.md)
and [prepared reference execution](prepared-reference-execution.md) now exercise this
graph through every unique coefficient solve and raw-score block on bounded fixtures.
A later authenticated [file-backed native prepared session](native-prepared-session-protocol.md)
executes the same graph in one project-owned C11 child, and its bounded public consumer
produces fixed per-query learned-plus-six-baseline evidence on project-authored D4–D7
fixtures. No persistent reusable cache, default trainer/CLI replacement, all-domain
artifact, or official full-shape execution is implemented.

## Scope

The current bilinear benchmark has three nested levels: an outer held-out domain,
lambda-tuning cross-fitting inside the outer training side, and calibration
cross-fitting inside each predictor fit. Repeating the ordinary trainer follows that
logical call graph exactly, but many base ridge fits have the same training-domain
membership.

`tierroute.predictors.prepared_graph` describes the unique base-training subsets and
the held-out-domain raw-score blocks needed by that graph. It is intentionally separate
from `RidgeSolver`, predictor artifacts, and the native dense-ridge protocol. The
planner accepts exact immutable tuples, keeps each domain/count pair together, sorts
the pairs by Python string order, and assigns mask bits from that canonical catalogue.
It never reads examples, prompts, embeddings, targets, or model outputs.

`tierroute.predictors.prepared_store` is an experimental consumer of this contract. It
adds canonical in-memory fit rows, training-only dynamic tags and population scaling,
and per-domain Welford/Chan moments. It does not change the planner or default training
path; its exact trust boundary and smaller Python reference caps are documented
separately.

`tierroute.predictors.prepared_execution` is the next experimental consumer. It
combines and discards one subset moment object at a time, shares one Cholesky
factorization across that subset's model targets, and emits target-free feature-shard
identities plus row-major raw scores for every graph block. Its distinct arithmetic,
lineage, cumulative-admission, and nonclaim boundaries are documented separately.

This scope supports four through seven domains and covers nested evaluation only.
Training one final deployable predictor on all domains would add one all-domain base
subset and solve. The existing size-`D-1` raw-score blocks already provide the OOF
inputs for its per-model calibrators; final artifact assembly is still outside this
plan. The extra all-domain solve is not silently included in the counts below. The
paired GBM experiment and the six routing baselines are also outside this ridge graph.

## Exact graph

Let `D` be the domain count and `N` the total example count. Training subsets are
enumerated by omitted-domain count `3`, then `2`, then `1`; combinations and scored
domains use ascending canonical indices.

| Quantity | Closed form | `D = 7` |
|---|---:|---:|
| logical calibrated-predictor fits | `D^2` | 49 |
| logical base-ridge fits | `D * ((D - 1)^2 + D)` | 301 |
| unique base-training subsets | `C(D,3) + C(D,2) + D` | 63 |
| unique subset/scored-domain blocks | `3*C(D,3) + 2*C(D,2) + D` | 154 |
| prepared score-row memberships per example | `C(D,2) + 1` | 22 |

For seven domains, the 63 subsets split into 35 size-four, 21 size-five, and
seven size-six subsets. Their complements produce 105, 42, and seven score blocks.
Every example therefore occurs in exactly 22 prepared score blocks even when domain
sizes are unequal, so the total is exactly `22N`.

`22N` means scored example-row memberships, not floating-point operations. With `M`
model targets it represents `22NM` scalar raw scores, and scoring those rows with `d`
features represents `22NMd` dot-product positions.

The raw blocks are inputs to later Python orchestration, not the entire calibrated
pipeline. At seven domains there are 28 unique calibrated training sets
(`C(D,2) + D`), whose OOF calibration inputs occupy `C(D,2)N = 21N` memberships in the
raw cache. Lambda tuning and outer prediction consume 49 post-calibrated blocks with
`DN = 7N` row memberships. Those calibrator and policy dependencies remain owned by
the existing Python path and are not materialized as nodes in this graph-only slice.

For comparison, the unchanged repeated path visits

```text
N * (D - 1) * ((D - 2)^2 + (D - 1))
```

base-training rows and

```text
N * (D * (D - 1) + 1)
```

raw-score rows. At `D = 7`, those are `186N` and `43N`. The graph contract does not
claim a measured reduction: the bounded reference now materializes the deduplicated
target structure on small fixtures, but there is no scalable execution, wall-time,
peak-memory, quality, or cost result.

## Resource estimate

The planner computes all derived counts and rejects the request before combinations
are materialized. Its modeled numeric buffers keep the current binary64 feature
semantics. A binary32 cache is not interchangeable with the current encoder or native
protocol and would need coefficient, calibration, lambda-selection, and final-report
parity evidence before adoption.

For `U` unique subsets, `R = N * (C(D,2) + 1)` prepared score rows, feature width `d`,
and target count `M`, the estimate contains:

```text
feature cache bytes       = 8*N*d
target cache bytes        = 8*N*M
packed domain-stat bytes  = 8*D*(1 + d + d*(d+1)/2 + M + d*M)
coefficient cache bytes   = 8*U*M*(d+1)
raw-score cache bytes     = 8*R*M
one solve workspace bytes = 8*(2*d^2 + 2*M*d + 2*d + 3*M)

statistics work = 3*N*(d+M) + N*d*(d+1)/2 + N*d*M
solve work      = U*(d^3 + 2*M*d^2 + M*d)
score work      = R*M*d
```

These are modeled binary64 numeric-buffer bytes and dominant numeric-work units, not
wall-clock, peak-RSS, or complete-work upper bounds. They omit Python graph objects,
allocator/process/serialization overhead, tag-schema work, subset-stat combination,
calibration, and lambda tuning. For the planned RouterBench shape (`D=7`, `N=34,778`,
`d=1,036`, `M=11`), the graph has 765,116 prepared score rows,
99,446,578,402 modeled numeric-work units, and 412,529,936 modeled numeric-buffer
bytes. This is an admitted preflight shape, not a completed benchmark or a performance
claim. It is admitted by this graph-only compact-buffer model, but the subsequent
prepared store/statistics and execution-reference ceilings reject the planned
full-dimensional workflow before it runs: the embedded-store construction has a
combined 512 MiB admission, statistics have a 50,000,000-unit work cap, and execution
has a 100,000,000-unit aggregate work cap. Graph admission is therefore not execution
admission.

The reviewed graph-only ceilings are seven domains, 1,000,000 examples, 4,096
features, 256 targets, 63 subsets, 154 score blocks, 16,000,000 score-row memberships,
2 GiB modeled numeric buffers, and 200,000,000,000 modeled numeric-work units. Domain
labels are bounded both individually and in aggregate. These maxima are not necessarily
simultaneously admissible because every derived limit is checked. Every ceiling is
independent of the single-solve native protocol and can only be raised after a new
resource review.

## Trust boundary and remaining work

The graph rejects non-exact container/scalar types, duplicate or invalid Unicode
domains, non-positive counts or dimensions, malformed node relationships, and any
resource estimate beyond the reviewed limits. It does not authenticate a row store or
prove training-data isolation by itself. The
[prepared feature-store reference](prepared-feature-store.md) now tests canonical
fit-content identity and subset isolation, but its digests are not authenticity or
provenance evidence.

A future official-shape deployable prepared path still needs all of the following:

- audited offline embedding preparation and licensed input data;
- persistent reusable cache/all-domain artifact assembly bound to the authenticated
  source, store, binary, graph, and policy identities;
- complete `D7/N34778/d1036/M11` execution plus broader near-tie and end-to-end parity;
- shipped command/trainer integration without changing the fixed per-query evidence
  into an unverified cumulative or cascade claim; and
- pinned-toolchain distributable artifacts with macOS, Linux-musl, and Windows-MSVC
  link/import audits. The current CI compiles ephemeral source candidates only; and
- the entrant's independent maintainer walkthrough and sign-off.

The graph contains no network operation, dependency, model asset, dataset, executable,
or build tool. Its bounded reference and native successors establish synthetic/frozen
coefficient-to-fixed-per-query-report wiring only; none of these paths makes bge-m3,
RouterBench execution, official SK Telecom data, cost savings, quality retention, or
full-dimensional training an implemented claim. Issue #9 remains open.
