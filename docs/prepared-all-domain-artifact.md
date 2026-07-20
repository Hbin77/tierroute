<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepared all-domain predictor artifact

Status: experimental, bounded, in-memory Python reference assembly with canonical
JSON persistence. The implementation is split between
`tierroute.predictors.prepared_assembly` and
`tierroute.predictors.prepared_artifacts`.

This slice closes one narrow gap in the prepared Python reference. It takes the exact
in-memory feature store, per-domain statistics bundle, and complete prepared raw-score
bundle produced by the existing public builders. It then performs one all-domain
ridge solve, fits one out-of-fold isotonic calibrator per model, and returns a
strictly versioned inference artifact. It is an offline library API: assembly,
serialization, save, and load perform no provider call, network access, download,
subprocess, native execution, or model inference.

The artifact is intentionally separate from the
[prepared policy pipeline](prepared-reference-pipeline.md) and the
[native prepared-session protocol](native-prepared-session-protocol.md). It does not
change the existing bilinear artifact v1 or GBM artifact v1 formats.

## Public API

The supported assembly entry point is:

```python
from tierroute.predictors import (
    PreparedBilinearPredictorArtifact,
    assemble_prepared_bilinear_artifact,
    estimate_prepared_all_domain_assembly,
)

artifact = assemble_prepared_bilinear_artifact(
    store,
    statistics,
    raw_scores,
    expected_source_fit_sha256=trusted_source_fit_sha256,
    expected_store_sha256=trusted_store_sha256,
    expected_statistics_sha256=trusted_statistics_sha256,
    expected_raw_score_sha256=trusted_raw_score_sha256,
)
```

Its complete signature is:

```text
assemble_prepared_bilinear_artifact(
    store: PreparedFeatureStore,
    statistics: PreparedDomainStatisticsBundle,
    raw_scores: PreparedRawScoreBundle,
    *,
    expected_source_fit_sha256,
    expected_store_sha256,
    expected_statistics_sha256,
    expected_raw_score_sha256,
) -> PreparedBilinearPredictorArtifact
```

The three parents must come from the same canonical prepared graph and store. The four
expected digests are mandatory caller-retained pins. A caller should obtain them
through a separately trusted workflow; copying a digest from the object being checked
does not create trust.

`estimate_prepared_all_domain_assembly(store, statistics, raw_scores)` exposes the
same shallow admission calculation without performing the all-domain numerical work.
It returns a `PreparedAllDomainAssemblyEstimate`.

The artifact supports `to_dict()`, `to_json()`, `from_dict()`, and `from_json()`.
File persistence is explicit:

```python
import hashlib

path = artifact.save("prepared-predictor.json")
expected = hashlib.sha256(path.read_bytes()).hexdigest()  # retain through a trusted channel
loaded = PreparedBilinearPredictorArtifact.load(
    path,
    expected_artifact_sha256=expected,
)
```

The example only illustrates the API. A digest computed from the same untrusted file
immediately before loading is a consistency check, not authentication.

## Six fail-closed assembly phases

The public assembler follows a fixed order. Expensive arithmetic and calibration are
not allowed to begin until every earlier phase succeeds.

1. **Exact types, shallow structure, and resource admission.** The assembler checks
   the exact three parent types and lowercase expected-digest syntax, canonically
   reconstructs the graph, validates frozen algorithm and solver/scorer identities,
   child counts, tuple shapes, byte payload lengths, tag masks, and configured graph
   limits, and computes the complete assembly estimate. This phase uses lengths and
   metadata; it does not walk numeric leaf values.
2. **First trusted-pin comparison.** The cached source-fit, store, statistics-bundle,
   and raw-score-bundle SHA-256 values must match the four expected pins.
3. **Complete canonical resnapshot.** Every admitted parent and bounded descendant is
   reconstructed through its own validating constructor. This revalidates finite
   binary64 payloads and recomputes constructor-owned `init=False` identities instead
   of trusting cached fields after possible mutation.
4. **Second trusted-pin comparison.** The same four expected digests must match the
   newly reconstructed parents. A stale cached digest cannot carry changed content
   into the numerical phase.
5. **Cross-parent and store-derived joins.** Store, statistics, coefficients, feature
   shards, raw-score blocks, model catalogues, embedding identities, masks, ridge,
   solver, and scorer must describe one exact layout. Scored feature shards are rebuilt
   from the store. Calibration sources are selected semantically as the `D`
   all-but-one-domain score contexts, not by assuming they occupy the last `D` tuple
   positions. Missing, duplicate, reordered, or mismatched row and prompt keys fail.
6. **Materialization.** Per-domain moments are combined in canonical ascending-domain
   order with Chan arithmetic; one final Cholesky factor solves every all-domain model
   target. Store-derived target-shard identities bind the out-of-fold joins without
   retaining an `N × M` target matrix in the artifact. Each model receives one
   equal-weight PAV isotonic fit over all `D` semantic held-out streams. The final
   coefficient, calibrators, lineage, and canonical artifact are then constructed.

Failure is reported by exception. There is no partial artifact result.

## Canonical identities and lineage

The root JSON format is frozen as:

```text
artifact_kind    tierroute-prepared-bilinear-predictor
artifact_version 1
algorithm_id     tierroute.prepared-bilinear-artifact-v1
```

The new derivation namespaces are:

```text
tierroute.prepared-all-domain-assembly-v1
tierroute.prepared-all-domain-statistics-chan-v1
tierroute.prepared-all-domain-coefficient-v1
tierroute.prepared-predictor-target-shard-v1
tierroute.prepared-predictor-calibration-input-v1
tierroute.prepared-predictor-isotonic-v1
```

The lineage also freezes the existing prepared graph, surface-feature,
moment-ridge-solver, and raw-scorer identities:

```text
tierroute.prepared-nested-lodo-graph-v1
tierroute.surface-features-v1
tierroute.prepared-moment-ridge-cholesky-python-v1
tierroute.prepared-raw-dot-product-python-v1
```

It retains:

- caller-pinned source-fit, store, statistics-bundle, and raw-score-bundle roots;
- the optional precomputed-embedding snapshot root;
- recomputed all-domain-statistics and final-coefficient roots; and
- one ordered calibration source per held-out domain, including its semantic subset
  and score-block indices, row count, raw-score-block root, scored-feature-shard root,
  and store-derived target-shard root.

Each model calibrator stores a calibration-input root over its ordered
`(raw score, target)` binary64 pairs plus source catalogue, and a separate identity
over that input root and the fitted isotonic step function. Numerical hashes use
namespace-separated, typed, length-framed fields and canonical little-endian binary64
payloads. The serialized artifact recomputes its final-coefficient identity from its
schema, weights, biases, ridge, and model order. Parsing recomputes calibrator
identities and rejects unknown or missing fields, duplicate JSON keys, non-finite
numbers, and inconsistent lineage. `from_json()` may canonicalize otherwise valid
self-declared JSON; only pinned `load()` additionally requires the input bytes to be
the exact canonical reserialization.

These SHA-256 values identify exact declared content. In particular, the source
digests are **not** authenticity, authorship, license, or provenance attestations.
They do not prove that a parent was honestly derived, that an embedding provider
created the supplied values, or that data came from SK Telecom. Substitution detection
requires an expected digest obtained through a separately trusted channel.

## Resource model and caps

The estimate covers the implementation-owned resnapshot, all-domain aggregation,
solve, per-model calibration, retained state, canonical JSON, parsing/staging, and
their reviewed object-amplification allowance. Important reported fields include
input and retained numeric bytes, row-key bytes, aggregate and solve workspace,
largest target-shard transient, one-model-at-a-time PAV storage, canonical-JSON upper
bound, object-amplification components, and phase-specific work units.

Assembly is rejected before numeric traversal when any applicable boundary is
exceeded:

| Boundary | Ceiling |
|---|---:|
| domains | 4 through 7 |
| prepared examples | 1,000,000 |
| universal prepared features | 4,096 |
| model targets | 256 |
| applicable prepared numeric payload | 512 MiB |
| all per-domain statistic children combined | 2,000,000 scalars |
| retained artifact numeric scalars | 800,000 |
| canonical JSON document/estimate | 32 MiB |
| modeled Python-object amplification | 256 MiB |
| complete modeled assembly storage | 512 MiB |
| complete modeled assembly work | 500,000,000 units |
| isotonic input rows | 500,000 |

Strict JSON decoding additionally limits a number token to 640 characters, the
format-specific number-token count to 800,036, nesting depth to 32, string-token count
to 32,768, one encoded string token to 24,578 characters, structure tokens to
1,100,000, one metadata value to 4 KiB, and aggregate metadata to 1 MiB.

The estimate is a conservative admission contract for the current Python
implementation, not a measurement. It is not peak RSS, wall time, throughput, an
allocator guarantee, or evidence that the official full shape was executed.
Caller-owned input graphs, interpreter/process overhead outside the modeled objects,
filesystem cache, and unrelated process memory are outside the estimate.

## Provider and persistence boundaries

Assembly and persistence are provider-free. They consume only the supplied prepared
records and local bytes. They never import or consult the native prepared result
consumer, never call an embedding provider, and never access a network.

`build_predictor(embedding_provider=...)` is the explicit inference boundary:

- a surface-only artifact needs no provider and rejects an unnecessary one;
- an embedded artifact requires a provider whose declared identity and dimension
  match the stored schema;
- constructing the predictor immediately reads and validates that declared provider
  metadata but does not call `embed()`; and
- the first embedding invocation occurs only when prediction encodes a prompt.

The caller remains responsible for supplying an audited local offline provider.
`build_predictor` does not download a model, authorize a provider, or prove that a
provider implementation is network-free.

`save()` validates the complete canonical document before creating exactly one random,
exclusive same-directory stage. It writes and syncs that stage, verifies that the
destination and stage did not change, replaces the destination once, and syncs the
parent directory where supported. It deliberately creates no backup. Validation or
staging failure leaves an existing destination unchanged under the tested ordinary
OS/Python failure model.

`load()` requires `expected_artifact_sha256`. It rejects symlinks/reparse points and
non-regular nodes, opens without following the final link where supported, bounds the
read, verifies descriptor and path stability, compares the exact bytes with the
trusted pin, parses strict JSON, and requires byte-for-byte canonical reserialization.
This is pinned single-document persistence, not a transactional bundle, multi-writer
protocol, signature scheme, or power-loss guarantee across unrelated paths.

## Synthetic acceptance boundary

Project-authored tests exercise the following behavior:

- surface-only `D = 4, 5, 6, 7` fixtures compare all-domain schemas, weights,
  intercepts, held-out raw scores, isotonic partitions, and final predictions with the
  authoritative rowwise fitting path under explicit numerical tolerances;
- every recognized held-out surface tag is absent from its own all-but-one-domain
  coefficient schema;
- a high-dynamic-range embedding fixture covers a magnitude ratio of at least
  `10^9`, exact collinearity, a zero-variance embedding column, zero-variance surface
  columns, and a constant model target;
- an independent exact-rational PAV oracle covers tied raw scores, adjacent violating
  blocks, exact block membership, and step-function probes at each upper bound and at
  the neighboring `nextafter` values; and
- a synthetic one-shot route oracle covers an unambiguous selection, budget exclusion,
  and an exact-utility tie, then checks the call/final-select transition and exact
  per-query ledger accounting through `OfflineSimulator`.

The prepared moment path and the rowwise path use different floating-point reduction
orders. Their coefficient and raw-score comparisons are tolerance-based, not bitwise
or universally cross-platform equal. PAV partitions, calibrated values, and routing
decisions are checked directly on the frozen fixtures rather than inferred merely
from raw-score closeness.

These fixtures use final labels from all domains: coefficients are fit on the complete
training set and calibration joins cover every row once through held-out predictions.
They are appropriate for constructing an inference artifact. They are not an
unbiased quality estimate. Same-fixture parity only shows that two implementations
agree on those project-authored cases; it does not establish predictive quality,
generalization, or competition performance.

Verification commands for this slice are listed below; recorded test counts, commit
hashes, and CI runs should be added only after the final reviewed branch and merged
state exist:

```bash
ruff check \
  src/tierroute/predictors/prepared_artifacts.py \
  src/tierroute/predictors/prepared_assembly.py \
  tests/test_prepared_artifacts.py \
  tests/test_prepared_artifact_hardening.py \
  tests/test_prepared_assembly.py \
  tests/test_prepared_assembly_hardening.py \
  tests/test_prepared_assembly_numerics.py
ruff format --check \
  src/tierroute/predictors/prepared_artifacts.py \
  src/tierroute/predictors/prepared_assembly.py \
  tests/test_prepared_artifacts.py \
  tests/test_prepared_artifact_hardening.py \
  tests/test_prepared_assembly.py \
  tests/test_prepared_assembly_hardening.py \
  tests/test_prepared_assembly_numerics.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q \
  tests/test_prepared_artifacts.py \
  tests/test_prepared_artifact_hardening.py \
  tests/test_prepared_assembly.py \
  tests/test_prepared_assembly_hardening.py \
  tests/test_prepared_assembly_numerics.py
```

## Exact non-claims and open gate

This slice does **not**:

- consume `NativePreparedSessionResult`, native result files, mmap views, executable
  credentials, or the native prepared benchmark bridge;
- add `tierroute train`, `tierroute route`, showcase, policy-artifact, or prepared
  policy-pipeline integration;
- implement or run bge-m3, include official SK Telecom or RouterBench data, or grant
  redistribution rights for either;
- execute the complete official `D7/N34778/d1036/M11` workflow;
- establish a benchmark, quality gain, OOD/generalization result, oracle-gap result,
  cost reduction, speedup, memory saving, throughput result, production readiness, or
  distributable native/release artifact; or
- approve source provenance, a provider, a compiler, native bytes, or network behavior.

The implementation itself remains offline and has no runtime network path. A future
provider must be separately audited and supplied locally; automatic weight download
remains forbidden. Native-result consumption, CLI/trainer/policy integration,
official-shape end-to-end execution and direct parity, bge-m3 asset/provider evidence,
licensed challenge data, and audited multi-platform release artifacts are still open
work. [Issue #9](https://github.com/Hbin77/tierroute/issues/9) therefore remains open.
