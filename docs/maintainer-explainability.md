<!-- SPDX-License-Identifier: Apache-2.0 -->

# Maintainer explainability review

[한국어 실행 워크시트](maintainer-explainability.ko.md)

## How to use this document

This is the human review packet for contest-critical tierroute code. It is deliberately
answer-first: each section states the invariant, points to the implementation and tests,
and asks questions the entrant must answer in their own words. Reading this document or
receiving a green CI result is not sign-off. The owner must trace the code, run the
focused tests, perform a mutation drill, and complete the table at the end.

The re-audited **implementation snapshot** is
`a1d7bd7dd835a1ab88e85e805df167985ca699be`. A later packet/document commit is a
separate value and does not replace the reviewed
implementation commit. Record both. A green result on this snapshot does not attest
to later source changes.

### Optional 60–80 minute partial route

After the locked environment is ready, an entrant may spend an estimated 60–80 minutes
on Korean mutation cards **1 → 2 → 3 → 7**. Their mutation
sources and primary test nodes are byte-unchanged from the older packet snapshot
`c6491508533655baa76c7b50bfdadacbc1612e60` through the reviewed implementation
snapshot. That machine diff and the automated re-audit are not human sign-off.

For each card, the entrant must run the baseline, write the expected failure before
the mutation, observe pytest exit status 1 and the named invariant, reverse the patch,
rerun the baseline, and verify an empty tracked/untracked tree and HF cache. Only those
four rows may then be signed, and only if the entrant answers the five explanation
questions in their own words. Cards 4, 5, 6, and 8 and every AI-assistance ledger row
remain blank or **PENDING** until separately executed and reviewed. The optional
RouterBench local-artifact download/E2E is excluded. If time expires or any restoration
or cleanup check fails, leave that card and every remaining row **PENDING**; a diff or
green CI result is not partial human credit. At packet publication all eight rows are
blank; only a later entrant execution may change them.

### Current-snapshot machine re-audit — not human sign-off

On 2026-07-17 the mutation wiring was re-executed at
`a1d7bd7dd835a1ab88e85e805df167985ca699be` in detached throwaway worktrees with the
locked Python 3.12.11 environment and an empty offline `HF_HOME`:

| Card | Machine re-audit result |
|---|---|
| 1 | 1 baseline pass → pytest exit 1 with the exact-cost boundary error → 1 restored pass |
| 2 | 1 baseline pass → pytest exit 1 with the cost/call-sum error → 1 restored pass |
| 3 | 1 baseline pass → pytest exit 1 with the cap-identity digest collision → 1 restored pass |
| 4 | 7 baseline passes → pytest exit 1 with `3 failed, 4 passed` → 7 restored passes |
| 5 | 558 baseline passes across 111+92+16+98+241 → pytest exit 1 with the held-out prompt assertion → 558 restored passes |
| 6 | 104 baseline passes → pytest exit 1 with `DID NOT RAISE <class 'ValueError'>` → 104 restored passes |
| 7 | baseline node pass → pytest exit 1 with `DID NOT RAISE`, zero calls to patched `os.system` → restored five-node suite pass |
| 8 | 1 baseline pass → pytest exit 1 with the exact new-policy assertion → 1 restored pass; expanded suite 25 passes, package suite 3 passes, license 12 and SPDX 146 gates pass |

Each card ended with an empty tracked/untracked status and HF cache. Card 8's three
dedicated basetemp directories contained no staging/backup debris; the broader
failure-injection suite intentionally retains recovery backups, so no empty-global-temp
claim is made. This is automated/AI-assisted evidence only. Every human-owned row below
remains blank and **PENDING**.

## 1. Router contract and exact cost arithmetic

**Invariant.** A router sees only pre-call state and completed call history, then either
calls one listed model or selects one recorded output. Costs are exact, unit-free,
non-negative `Decimal` values; caller decimal context must not change affordability,
totals, canonical bytes, or resource failure behavior.

- Core: [`core/schemas.py`](../src/tierroute/core/schemas.py),
  [`core/router.py`](../src/tierroute/core/router.py),
  [`core/costs.py`](../src/tierroute/core/costs.py), and
  [`core/integer_text.py`](../src/tierroute/core/integer_text.py)
- Strongest evidence: [`test_core.py`](../tests/test_core.py) and
  [`test_integer_text.py`](../tests/test_integer_text.py)
- Design context: [README architecture](../README.md#router-contract-and-architecture)

Owner questions:

1. Why is float rejected for a budget boundary while predictor quality may use binary64?
2. Where are candidate membership, call-history selection, and affordable quoted cost
   checked, and which failure occurs before a replayed provider charge?
3. How do the project-owned add/subtract/scale/divide helpers avoid caller-context
   rounding, and why is the 100,000-position resource bound part of the API contract?
4. Which cost values are quotes and which are realized charges?

## 2. Budget adapters, replay, and executed-call evidence

**Invariant.** The simulator never performs a live model call. Replay JSON first passes
one descriptor-stable, strict, finite adapter boundary. The simulator consumes a logged
outcome only after a valid `CallModel`, charges the injected ledger with the realized
cost, and records that executed call even when the ledger rejects an overspend. Budget
scope remains adapter-owned; the shared core does not guess per-query versus cumulative
semantics.

- Core: [`eval/simulator.py`](../src/tierroute/eval/simulator.py),
  [`eval/schemas.py`](../src/tierroute/eval/schemas.py),
  [`eval/budgets.py`](../src/tierroute/eval/budgets.py), and
  [`adapters/budgets.py`](../src/tierroute/adapters/budgets.py), with orchestration in
  [`eval/protocols.py`](../src/tierroute/eval/protocols.py) and
  [`eval/planning.py`](../src/tierroute/eval/planning.py). Replay ingestion is in
  [`adapters/json_dataset.py`](../src/tierroute/adapters/json_dataset.py), with limits in
  [`adapters/resource_limits.py`](../src/tierroute/adapters/resource_limits.py)
- Strongest evidence: [`test_simulator.py`](../tests/test_simulator.py),
  [`test_budgets.py`](../tests/test_budgets.py), and quote/accounting cases in
  [`test_metrics.py`](../tests/test_metrics.py), plus the hostile-input matrix in
  [`test_json_dataset.py`](../tests/test_json_dataset.py)
- Design context: [evaluation-scope trust boundary](evaluation-scope.md) and
  [replay JSON boundary](replay-json.md)

Owner questions:

1. Why does a realized overspend remain in `QueryResult.calls` and `BudgetReport.spent`?
2. What is visible in `RouterState` before and after the first call, and where are
   uncalled outputs and quality labels kept private?
3. How do `begin_query`, `finish_query`, `remaining_budget`, and `charge_realized`
   differ between the two bundled ledger adapters?
4. Why is the shipped default one-shot even though the interface can represent another
   call or a prior-output selection?
5. Trace a `--data` file through descriptor checks, lexical preflight, strict schema,
   collection/text limits, and outer/nested LODO work bounds. Why is there no unlimited
   override, and how would an incompatible official schema be added?

## 3. Complete evaluation-scope identity

**Invariant.** Cross-report evidence is comparable only when algorithm, complete
ordered replay digest, and call cap match. The digest is computed from the same
canonical immutable snapshot used by replay. It is collision-resistant accidental-mix
evidence, not an authenticated signature or a hash of executable ledger code.

- Core: [`eval/provenance.py`](../src/tierroute/eval/provenance.py),
  [`eval/schemas.py`](../src/tierroute/eval/schemas.py), and
  [`eval/simulator.py`](../src/tierroute/eval/simulator.py), with comparison checks in
  [`eval/metrics.py`](../src/tierroute/eval/metrics.py)
- Strongest evidence: [`test_eval_provenance.py`](../tests/test_eval_provenance.py),
  scope cases in [`test_simulator.py`](../tests/test_simulator.py), and
  cross-report cases in [`test_metrics.py`](../tests/test_metrics.py)
- Design context: [evaluation-scope byte contract](evaluation-scope.md)

Owner questions:

1. Which replay, tier, metadata, ordering, and protocol fields enter scope-v1, and which
   policy-dependent result fields are intentionally excluded?
2. Why must candidate/realized/tier costs and binary64 labels be normalized before both
   hashing and replay? Explain the `Decimal("1")` versus `Decimal("1.00")` and
   `2**53` versus `2**53 + 1` regressions.
3. How do typed length-delimited tokens avoid concatenation/type ambiguity, and what
   change requires `tierroute-evaluation-scope-v2`?
4. Why are custom mappings, numeric subclasses, cycles, and repeated logical payloads
   rejected or bounded before routing?

## 4. Metrics, leakage-free benchmarking, and the stream showcase

**Invariant.** Tier-weighted quality applies each tier's explicit weight and never
redistributes the weight of an infeasible tier. Reportable learned-policy validation
uses true nested leave-one-domain-out splits: predictor fitting, calibration, and lambda
tuning remain inside each outer training side. The domain-table baseline is also fitted
only on that side and can read only an explicit pre-call metadata tag, not the split
label. The learned router and all six policies replay the same original row order and
per-query ledger scope. Scores, exact cost evidence, and oracle-gap recovery are derived
again from bound reports rather than trusted as stored numbers. The three-step showcase
directly replays the same outer-fold learned policies and must agree with the nested
result; its running values remain presentation diagnostics outside the full benchmark.

- Core: [`eval/metrics.py`](../src/tierroute/eval/metrics.py),
  [`eval/planning.py`](../src/tierroute/eval/planning.py),
  [`eval/validation.py`](../src/tierroute/eval/validation.py), and
  [`policies/baselines.py`](../src/tierroute/policies/baselines.py), with orchestration in
  [`policies/baseline_evaluation.py`](../src/tierroute/policies/baseline_evaluation.py)
  and [`policies/benchmark.py`](../src/tierroute/policies/benchmark.py). The presentation
  boundary is [`showcase.py`](../src/tierroute/showcase.py)
- Strongest evidence: [`test_metrics.py`](../tests/test_metrics.py),
  [`test_validation.py`](../tests/test_validation.py),
  [`test_policies.py`](../tests/test_policies.py), and
  [`test_baseline_evaluation.py`](../tests/test_baseline_evaluation.py), plus the
  learned-versus-baseline contract in [`test_benchmark.py`](../tests/test_benchmark.py),
  showcase invariants in [`test_showcase.py`](../tests/test_showcase.py), and human/JSON
  CLI coverage in [`test_cli.py`](../tests/test_cli.py)
- Smoke-lane evidence: inference-only commands in
  [`scripts/smoke.py`](../scripts/smoke.py), training-backed benchmark/showcase in
  [`scripts/training_smoke.py`](../scripts/training_smoke.py), and the public target
  contract in [`test_reproduction_contract.py`](../tests/test_reproduction_contract.py)
- Design context: [literature and novelty review](literature-and-novelty.md) and
  [README evaluation section](../README.md#evaluation)

The report-shaped CLI is:

```bash
tierroute benchmark --budget-scope per-query [--data path/to/replay.json] [--json]
```

It requires the learned report and every baseline to share one
`EvaluationScopeIdentity` and publishes per-fold train/test counts plus a versioned
digest binding the held-out domain and exact ordered memberships, not raw example IDs.
The membership digest is a compact reproducibility identity, not authenticated proof.
The same JSON records tier weights and limits, resolved baseline roles and algorithm
parameters, their versioned config-to-replay-decision evidence digest, and requested
lambda-search resource controls. The bundled synthetic replay is wiring-only. A caller
supplying `--data` owns the license and the validity of any empirical or competition
claim. Cumulative and cascade reports remain gated on official sequence semantics and a
sequence-level oracle.

The separate presentation commands are:

```bash
tierroute demo
tierroute demo --json
```

They select exactly three bundled synthetic rows, one each at Fast, Balanced, and
Premium. Each row is directly replayed as one example and one tier through
`OfflineSimulator` using the learned/tuned policy fitted on its outer training side;
the result must equal that row/tier in the nested learned report. Each step exposes its
illustrative budget, quote, realized charge, observed quality, independent per-query
oracle quality, running realized cost, and unweighted running retention. The retention
formula is `sum(observed) / sum(independent per-query oracle)`. Its denominator is not a
sequence-level oracle, and the ratio is not oracle-gap recovery. The mixed-tier running
cost is also reporting-only, not shared or cumulative budget accounting. All inputs and
values are project-authored synthetic wiring evidence. The full learned and six-baseline
populations remain in `tierroute benchmark`, outside this three-row stream.
The JSON contract is `tierroute-routing-stream-showcase` version 1: `stream.steps`
contains the curated rows, `accounting` states the interpretation boundaries,
`stream.totals` conserves the displayed values, and `benchmark_evidence` carries the
separate full-population learned-plus-six-baseline report. The bundled steps end at a
reporting-only realized-cost sum of `1.8` and an unweighted retention of `1/1`; those
fixture values are wiring assertions, not performance claims.

Korean mutation card 4 directly breaks only explicit tier-weight aggregation. Its
seven-node run must exit with pytest status 1 and `3 failed, 4 passed`. The independent
LODO and showcase nodes are sentinels; their passing does not make that one mutation
proof of leakage control or presentation semantics.

Owner questions:

1. Write the tier-weighted quality and oracle-gap recovery formulas, including when
   either result is `None` and why weights are never redistributed.
2. Why is a per-query oracle an upper bound under independent query budgets but not
   necessarily under a cumulative stream budget?
3. Distinguish split-only `EvaluationExample.domain` from observable
   `router_metadata["domain"]`; what leakage would occur if they were conflated?
4. Which constructor checks prevent mixed scope, fabricated score/gap/cost evidence,
   bad fold partitions, duplicate table keys, or unknown fold models?
5. Name all six baseline policies and explain the seeded random identity, length
   threshold/ties, unseen-domain fallback, and the oracle's privileged-label boundary.
6. Why does true nested LODO refit the predictor, calibrator, and lambda inside every
   outer training side, and what do the compact fold-membership digests bind without
   publishing raw example IDs? What do they not authenticate?
7. Trace one showcase row from its outer-fold learned router through the direct
   one-example/one-tier replay and nested-result equality check. Derive its running
   retention and explain why neither that ratio nor mixed-tier cost has official
   sequence-budget meaning.

## 5. Fitted features, bilinear/GBM quality prediction, and calibration

**Invariant.** Feature schema fitting, scaling, tag vocabulary, ridge coefficients, GBM
stumps, and isotonic calibration use training-side data only. Inner-LODO out-of-fold
predictions fit calibration; the predictor is then refit on all outer training rows.
The bilinear artifact v1 contract remains unchanged: it is strict schema-bound JSON,
and unknown schema/model/solver identities fail closed. GBM has a separate
`tierroute-gbm-predictor` artifact kind at version 1. Its library contract provides
bounded canonical `to_dict`/`from_dict` and `to_json`/`from_json`, atomic local `save`,
bounded local `load`, and offline `build_predictor` with exact embedding identity
checks. This does not add a GBM train/route CLI, lambda-policy binding, family-selection
decision, or deployment claim; the no-selection paired-estimation command remains its
only shipped CLI path. No `bge-m3` inference provider or asset bundle is shipped, and
the library artifact alone adds no external/official-data, performance, quality, or
cost-savings evidence.
The optional C11 ridge process is a training-only, explicitly authenticated solver; a
known solver ID never substitutes for its absolute path and exact binary hash.

The prepared execution reference remains bounded in-memory Python proof code, not the
production trainer. The native policy consumer described below does not route through
this reference object graph; it reconstructs the same bounded policy inputs directly
from authenticated native views. For active raw coordinates `a_i` and continuous scales
`s_i` (one for noncontinuous coordinates), the in-memory reference constructs
`G_ij = Cxx[a_i,a_j] / (s_i s_j) + ridge * 1[i=j]` and
`h_i,m = Cxy[a_i,m] / s_i`, performs exactly one Cholesky factorization per unique
training subset for all model targets, and recovers
`b_m = mean(y_m) - sum_i mean(z_i) w_i,m`; standardized continuous means are zero and
the remaining encoded means retain their raw moment means. Scoring standardizes only
the first three continuous coordinates and uses ordinary Python `sum` in schema order.
Canonical plan, subset, catalogue, active-coordinate, row, and sorted-model order bind
target-major coefficient and row-major raw-score records.

The separate prepared policy bridge uses `(training subset, scored domain)` raw blocks
to fit the same per-model inner-LODO isotonic calibrators as the row path, then delegates
candidate search, lambda selection, adapter accounting, and global original-order
replay to the existing policy/simulator code. It retains `C(D,2)+D` calibration records,
`D^2` calibrated destination identities, trusted source/store/raw expectations, and the
real `NestedLodoLambdaResult`. Cost-aware outer-fold estimates count all five current
pair traversals and bound the aggregate lambda candidate/policy-artifact estimate from
the actual Decimal widths. The result also exposes ridge, solver/scorer, embedding, and ordered
child-digest configuration. It does not manufacture a second report schema or an
all-domain deployable artifact. Arbitrary injected ledger callback work and side
effects are outside the code-owned resource estimate.

The experimental native prepared path is a separate training-side vertical slice.
`prepared_files.py` writes or authenticates the fixed little-endian `TRPSTO01` store.
A caller-pinned receipt binds its whole-file bytes, source-fit identity, logical
prepared-store identity, and optional precomputed-embedding snapshot. Authentication
uses regular-file checks, descriptor/path metadata stability, exact section validation,
streaming hashes, and owner-only private copies; a digest without a separately trusted
expected value remains only an identity. `native_prepared.py` also authenticates one
explicit absolute executable, copies the store directly behind one `TRPSES01` request
header, and launches one child for the complete admitted graph. The C11 sidecar combines
domain moments, solves every subset with one factorization shared across targets, and
emits every score block in one exact-size `TRPRES01` result.

Python and C mirror the same public file/result/heap/scratch/work ceilings, so an input
cannot pass public preflight and encounter an undisclosed smaller C-only shape cap.
The request nonce is sampled until nonzero rather than giving one random all-zero draw
protocol meaning. Known nonzero status is a structured child failure; unknown status,
bad identity,
length, record order, digest, resource echo, non-finite payload, or exit/status mismatch
is a protocol failure. Successful records expose bounded binary64 views over one
context-managed mmap. One reentrant lock serializes read/view creation with close.
Closing while an exported view exists must remain retryable;
after a successful close every record/view access fails and private workspace cleanup
owns the descriptor on both POSIX and Windows paths. The adapter itself neither invokes
policy replay nor publishes an artifact.

The public `evaluate_native_prepared_per_query_benchmark` consumer supplies the bounded
policy integration. It requires an open exact `NativePreparedSessionResult` owned by the
caller, the trusted prepared-store receipt, and mandatory caller-retained expected
binary and result-file SHA-256 values. Before deep example/fold/lambda traversal it
compares the external binary/result/store pins and verifies the current result mapping.
Phase one authenticates an owned private store snapshot, compares its targets bit-
exactly with the evaluation rows, consumes native views only through `at()`, and builds
an owned calibrated snapshot without calling caller code. It verifies both mappings
again after the last read, rechecks the external result pin, closes only its owned store,
and stops consulting the still-open caller-owned result. Phase two uses only owned data,
the fixed `PerQueryBudgetLedger`, the existing nested lambda evaluator, and the six-
baseline evaluator. The returned evidence graph keeps calibration parameters and hashes
but no mmap, native view, raw/target/calibrated matrix, or private score payload.
The admitted config restricts candidate cap to exact integers 2 through 257, seed to
signed 64-bit, and character threshold to positive signed 64-bit. A direct calibrator
record cannot retain more points than its declared calibration rows. The owned numeric
estimate includes `policy.postprocess_numeric_bytes`, one additional simultaneous
`owned_calibrated_score_bytes` copy, and `row_index_bytes`; it does not treat Python
tuple/object allocator overhead as binary64 payload.
Baseline constructor/evidence work is the six reports times a conservative 32 modeled
bookkeeping units per retained tier/query row (`192 * T * N`), not measured CPU work.

- Core: [`features/encoding.py`](../src/tierroute/features/encoding.py),
  [`features/surface.py`](../src/tierroute/features/surface.py),
  [`features/embeddings.py`](../src/tierroute/features/embeddings.py),
  [`predictors/base.py`](../src/tierroute/predictors/base.py),
  [`predictors/training.py`](../src/tierroute/predictors/training.py),
  [`predictors/gbm.py`](../src/tierroute/predictors/gbm.py),
  [`predictors/gbm_training.py`](../src/tierroute/predictors/gbm_training.py),
  [`predictors/gbm_artifacts.py`](../src/tierroute/predictors/gbm_artifacts.py),
  [`predictors/_ridge.py`](../src/tierroute/predictors/_ridge.py),
  [`predictors/solvers.py`](../src/tierroute/predictors/solvers.py),
  [`predictors/prepared_graph.py`](../src/tierroute/predictors/prepared_graph.py),
  [`predictors/prepared_store.py`](../src/tierroute/predictors/prepared_store.py),
  [`predictors/prepared_execution.py`](../src/tierroute/predictors/prepared_execution.py),
  [`predictors/prepared_files.py`](../src/tierroute/predictors/prepared_files.py),
  [`predictors/native_prepared.py`](../src/tierroute/predictors/native_prepared.py),
  [`predictors/native_ridge.py`](../src/tierroute/predictors/native_ridge.py),
  [`native/tierroute_ridge.c`](../native/tierroute_ridge.c),
  [`native/tierroute_prepared.c`](../native/tierroute_prepared.c),
  [`predictors/calibration.py`](../src/tierroute/predictors/calibration.py), and
  [`predictors/artifacts.py`](../src/tierroute/predictors/artifacts.py), with limits in
  [`predictors/resource_limits.py`](../src/tierroute/predictors/resource_limits.py),
  plus [`policies/predictor_comparison.py`](../src/tierroute/policies/predictor_comparison.py),
  [`policies/prepared_reference.py`](../src/tierroute/policies/prepared_reference.py), and
  [`policies/native_prepared_benchmark.py`](../src/tierroute/policies/native_prepared_benchmark.py)
- Strongest evidence: [`test_feature_encoding.py`](../tests/test_feature_encoding.py),
  [`test_features_predictors.py`](../tests/test_features_predictors.py),
  [`test_bilinear_training.py`](../tests/test_bilinear_training.py),
  [`test_gbm_core.py`](../tests/test_gbm_core.py),
  [`test_gbm_training.py`](../tests/test_gbm_training.py),
  [`test_gbm_artifacts.py`](../tests/test_gbm_artifacts.py),
  [`test_gbm_artifact_hardening.py`](../tests/test_gbm_artifact_hardening.py),
  [`test_predictor_comparison.py`](../tests/test_predictor_comparison.py),
  [`test_predictor_comparison_cli.py`](../tests/test_predictor_comparison_cli.py),
  [`test_ridge_solver.py`](../tests/test_ridge_solver.py), and
  [`test_predictor_artifacts.py`](../tests/test_predictor_artifacts.py), plus
  [`test_native_ridge.py`](../tests/test_native_ridge.py) and
  [`test_prepared_graph.py`](../tests/test_prepared_graph.py), plus
  [`test_prepared_store.py`](../tests/test_prepared_store.py) and
  [`test_prepared_execution.py`](../tests/test_prepared_execution.py), plus
  [`test_prepared_reference_pipeline.py`](../tests/test_prepared_reference_pipeline.py),
  [`test_prepared_files.py`](../tests/test_prepared_files.py), and
  [`test_native_prepared.py`](../tests/test_native_prepared.py), plus
  [`test_native_prepared_benchmark.py`](../tests/test_native_prepared_benchmark.py)
- Design context: [lambda/training design](lambda-tuning.md) and
  [prepared graph contract](prepared-session-graph.md), the
  [prepared feature-store reference](prepared-feature-store.md), the
  [prepared execution reference](prepared-reference-execution.md), the
  [prepared policy-pipeline reference](prepared-reference-pipeline.md), plus the
  [native ridge protocol](native-ridge-protocol.md) and
  [native prepared-session protocol](native-prepared-session-protocol.md)

The prepared **execution slice** began at `f4b07bc`, its primary parity suite at
`608468b`, and admission/locality security regressions were hardened through
`2ac1b50`. A focused local run on Darwin arm64 with Python 3.12.11 reports 62 passed.
That focused count is local software evidence, not performance or human sign-off.
[PR #47](https://github.com/Hbin77/tierroute/pull/47) implementation/spec-head
[CI run `29524753168`](https://github.com/Hbin77/tierroute/actions/runs/29524753168) at
`8ec9cc1`
passed the dependency-free wheel, macOS/Windows native-source jobs, Python 3.10
(921 passed, one expected skip), and Python 3.12 (920 passed, two expected skips).
Those PR #47 counts do not cover the later prepared policy bridge. That bridge merged
in [PR #48](https://github.com/Hbin77/tierroute/pull/48) at
`566678c9c0181d9bcb76378ab423858150bff7b4`; its implementation is `63e288e` with
tests at `3249a3c`. Local Darwin arm64 verification recorded Python 3.10 with 954
passed and Python 3.12 with 953 passed plus one expected compatibility skip. The
[PR-head CI run `29530846709`](https://github.com/Hbin77/tierroute/actions/runs/29530846709)
at `cfa0c72` and
[merged-main run `29531008829`](https://github.com/Hbin77/tierroute/actions/runs/29531008829)
both passed Python 3.10/3.12, dependency-free wheel, and macOS/Windows native-source
jobs. Automated CI still does not replace the owner walkthrough below.

The later focused file-backed/native-session run reports 64 passed locally on Darwin,
including 38 native-session cases. They include actual compiled D4-D7
coefficient/raw-score parity with the complete Python reference on small surface-only
fixtures and one `D4/N8/d1036/M1` completion using 12
surface plus 1,024 synthetic embedding coordinates without projection. The official
`D7/N34778/d1036/M11` tuple has exact aggregate preflight only: no full store/session or
official/RouterBench data was run. Those tests are local software evidence, not a
speed, memory-efficiency, throughput, quality, or cost result.
[PR #50](https://github.com/Hbin77/tierroute/pull/50) merged at
`ffa8b8059985298df9d1cf0feec20374589afc1c`; its
[PR-head CI](https://github.com/Hbin77/tierroute/actions/runs/29537455566) and
[merged-main CI](https://github.com/Hbin77/tierroute/actions/runs/29537633261) passed
macOS/Windows ephemeral source compile, protocol/parity, and link/import audits. This is
not release-artifact approval, and no human sign-off is implied.

The subsequent current-tree native policy benchmark tests compile the same C11 source
for surface-only D4-D7 fixtures and require strict equality with the authoritative
rowwise `NestedLodoLambdaResult` and six-baseline result. D7 uses uneven row counts
`(1, 2, 1, 3, 2, 1, 2)` and three models. Adversarial cases lock external credential
fail-order, mapping verification before deep traversal and after the final read,
persistent mutation rejection, bit-exact targets, `at()`-only access, owned-store close
before replay, primary-error preservation, bounded configuration, and a payload-free
returned graph. The focused native run recorded 89 passes. In the locked full suite,
Python 3.10.19 with pip 26.1.2 recorded 1,044 passes with no skip; Python 3.12.10 with pip
26.1.2 recorded 1,043 passes and one expected skip for the locked Python 3.10
`typing_extensions` compatibility dependency.
[Implementation/spec branch-push CI `29542245699`](https://github.com/Hbin77/tierroute/actions/runs/29542245699)
at `9ed400d580e288bb9648a300a8de12a5c2200fff`,
[final PR-head CI `29543435978`](https://github.com/Hbin77/tierroute/actions/runs/29543435978)
at `304decd0a591fcfc5e5a1e04f35bf20b22c17cea`, and
[merged-main CI `29543610611`](https://github.com/Hbin77/tierroute/actions/runs/29543610611)
at `c7b717ce1226fcfd70d696d0124aa8df294033c8` each passed all five jobs: Python
3.10, Python 3.12, dependency-free wheel, Native source portability macOS, and Native
source portability Windows. [PR #52](https://github.com/Hbin77/tierroute/pull/52)
merged implementation/spec commits `f159e04`, `85393e2`, `a8e0896`, and `9ed400d`,
plus evidence commits `77e5c47` and `304decd`. PR #50's earlier platform jobs remain
separate session-layer evidence. The human walkthrough remains **PENDING**, no
distributable release artifact is approved, and this evidence does not complete issue #9.

[PR #56](https://github.com/Hbin77/tierroute/pull/56) separately implemented the GBM
artifact beginning at `5d1d727` and hardened it through `4de98de` and `5be3642`. At the
final evidence head `ef8606f34d8a7706a19ae2303d742a06c955d3cb`,
[push CI `29548164885`](https://github.com/Hbin77/tierroute/actions/runs/29548164885) and
[PR CI `29548166228`](https://github.com/Hbin77/tierroute/actions/runs/29548166228)
each passed all five jobs. The
[merged-main run `29548281471`](https://github.com/Hbin77/tierroute/actions/runs/29548281471)
at `a1d7bd7dd835a1ab88e85e805df167985ca699be` also passed all five jobs. This is
automated evidence for the library-only GBM artifact and current tree, not human
walkthrough, GBM CLI/policy binding, official-data, quality, or cost-savings evidence.

Korean card 5's one-line mutation directly tests only bilinear outer-fold isolation.
The remaining sub-boundaries require their own baseline evidence and entrant
explanation at the same reviewed implementation commit:

| Card 5 sub-review | Entrant explanation, exact command/result, and status |
|---|---|
| 5a feature, bilinear, and calibration |  |
| 5b GBM core, training, and artifact |  |
| 5c paired comparison |  |
| 5d prepared graph, store, execution, and reference |  |
| 5e native ridge and prepared file/session |  |
| 5f native policy bridge |  |

The top-level Card 5 sign-off row remains blank or **PENDING** until every 5a–5f row
contains the entrant's own explanation and exact command/result. A passing baseline or
the single bilinear mutation is insufficient. The re-audited five baseline groups
recorded 111, 92, 16, 98, and 241 tests respectively (558 total); this machine evidence
does not replace human sign-off.

Owner questions:

1. Which feature statistics are fitted rather than computed independently, and why
   must their fit occur inside the outer training fold?
2. Describe the centered-ridge normal equations and Cholesky solve; why does the
   reference solver have a conservative work guard and a fixed solver ID?
3. Why is isotonic calibration fitted from out-of-fold rather than in-sample predictions?
4. Which identities make a silent schema, catalogue, solver, embedding, or training-data
   change fail closed? Distinguish the unchanged bilinear artifact v1 from the separate
   `tierroute-gbm-predictor` v1 contract. Trace the predictor byte, numeric-token,
   structure, metadata, stump, and calibration limits through construction,
   `to_dict`/`from_dict`, `to_json`/`from_json`, atomic saving, bounded loading, and
   offline reconstruction. Why does GBM library reconstruction not imply policy-hash,
   train/route CLI, or deployment integration?
5. What is implemented today at the embedding boundary? Explain that no inference
   provider or asset path is shipped, then name the revision, license, asset-manifest,
   and offline checks required before `bge-m3` can be claimed.
6. For the GBM core, derive one residual update and split-gain calculation. Explain the
   feature/split tie break, observed right-boundary rule, no-positive-gain early stop,
   and why all inner/final work is preflighted before embedding.
7. Enumerate the complete paired nested-LODO GBM call graph. Why are the six baselines
   computed once, why are deltas `GBM - bilinear`, and why can the same outer evidence
   estimate a difference but not select a winning family without bias?
8. Trace a native request from Python count/allocation/work preflight through binary
   authentication, little-endian parsing, centered Gram/RHS construction, one shared
   Cholesky factor, residual verification, and the exit/status cross-check.
9. Why are the C11 solver ID and executable SHA-256 separate identities? Explain why
   loading a predictor needs only the former while training must authenticate both.
10. Derive why a single 1,024-feature solve does not make 301-fit nested LODO feasible.
    Why does the prepared contract enumerate 63 unique training subsets, 154 score
    blocks, and `22N` scored-row memberships for seven-domain nested evaluation?
11. Why is the prepared feature cache binary64? Trace the caller-checked source and
    precomputed-embedding digests, fixed 12+E raw layout, per-domain Welford moments,
    included-domain Chan combination, dynamic-tag isolation, moment equations above,
    shared-target factorization, intercept recovery, and schema-ordered dot product.
    Why is agreement with the independently refitted row oracle tolerance-based rather
    than bitwise or a cross-platform digest promise?
12. Derive the seven-domain `22N` row-membership and `22NM` scalar-score counts. Which
    canonical orders and example-ID join keys make 63 coefficient blocks and 154 raw
    score blocks deterministic, and why is global bundle-digest locality not promised
    even though an excluded-domain mutation leaves each unaffected coefficient/raw
    block unchanged?
13. Which three builders are the supported derivation path? Explain why direct leaf
    constructors validate only self-declared per-record canonical content, not origin,
    aggregate association, or derivation; why a digest is not authentication; and why
    substitution detection requires comparison with a separately trusted expected
    digest.
14. Enumerate the aggregate work and modeled numeric-storage categories before any
    subset combination/factorization or feature hashing/row scoring. Why are the
    100,000,000-unit and 512 MiB ceilings admission controls rather than peak-RSS,
    allocator, Python-object, caller-owned-memory, wall-clock, or speedup guarantees?
    Why does the residual factor 2,048 remain an empirical frozen-fixture regression
    guard rather than a universal numerical-error bound? Which provider, persistence,
    native execution, all-domain artifact, platform, and official-data gates keep
    issue #9 open?
15. For calibrated training set `S`, derive why calibration reads
    `(S-{c}, c)` for every `c in S` and prediction reads `(S, h)` for `h not in S`.
    Why are there `C(D,2)+D` calibration records, each with one calibrator per model,
    `D^2` calibrated destination blocks, and
    `[C(D,2)+D]N` raw row reads? Explain canonical fit order versus original replay
    order, trusted digest expectations, why five current pair traversals are counted,
    how Decimal width bounds the aggregate lambda candidate/policy-artifact estimate,
    the candidate-cap exhaustive
    flag, the injected-ledger exclusion, and why stable frozen-result equality is not
    universal near-tie parity.
16. Trace `TRPSTO01` from exclusive owner-only staging through row/domain/feature/target
    serialization, header/payload/whole-file hashes, publication, lstat/open/fstat
    stability checks, and private-copy authentication. Which receipt values need an
    independently trusted origin, and why is a matching content digest not provenance?
17. Trace one `TRPSES01` request and `TRPRES01` response. Why must public and C resource
    ceilings match exactly, why is there one child for all subsets/score blocks, and how
    do the nonce, store/binary/graph identities, exact record order, exit/status pair,
    finite scan, and scale-aware residual gate fail closed?
18. Why do result payloads remain mmap-backed? Explain ownership of the descriptor and
    private workspace, why the same reentrant lock must serialize read and close, why
    close with an exported view must be retryable, and which
    accesses fail after a successful close. Which Windows cleanup paths are exercised
    by the source-portability CI, and why is that still not release-artifact approval?
19. Distinguish the earlier D4-D7 small surface-only solve/score parity fixtures, the
    `D4/N8/d1036/M1` synthetic completion, and the aggregate-only
    `D7/N34778/d1036/M11` preflight. Why does none establish bge-m3, RouterBench,
    official-data, an all-domain artifact, or performance?
20. Trace the public native policy consumer from the three external pins through initial
    result verification, `at()`-only mapped reads, final reauthentication, owned-store
    close, and fixed per-query learned-plus-six-baseline replay. Why does the caller-owned
    result remain open, why can no caller callback run during phase one, and which object
    types and numerical payloads must be absent from the returned graph? Explain what the
    strict D4-D7 rowwise equality and adversarial tests prove—and what they do not.

## 6. Exact lambda routing, tuning, and policy artifacts

**Invariant.** Runtime and tuning use one routing function for
`predicted_quality - lambda * quoted_cost`, with exact rational lambda and deterministic
tie breaks. An exhaustive claim includes every boundary/interval/tail candidate. A cap
is an explicitly labeled deterministic approximation, not a smaller exhaustive search.
Nested LODO tunes only inside each outer training side. The current policy hash and
bundle bind the exact canonical JSON of a `BilinearPredictorArtifact` only.
`GbmPredictorArtifact` v1 is a separate library-only contract and is not connected to
the lambda policy or route CLI. SHA-256 supplies content identity, not origin or
provenance authentication.

- Core: [`policies/lambda_threshold.py`](../src/tierroute/policies/lambda_threshold.py),
  [`policies/lambda_tuning.py`](../src/tierroute/policies/lambda_tuning.py),
  [`policies/lambda_artifacts.py`](../src/tierroute/policies/lambda_artifacts.py), and
  [`policies/resource_limits.py`](../src/tierroute/policies/resource_limits.py), with
  shared exact-integer parsing in
  [`core/integer_text.py`](../src/tierroute/core/integer_text.py)
- Strongest evidence: [`test_policies.py`](../tests/test_policies.py),
  [`test_lambda_tuning.py`](../tests/test_lambda_tuning.py), and
  [`test_lambda_policy_artifacts.py`](../tests/test_lambda_policy_artifacts.py)
- Design context: [exact lambda tuning](lambda-tuning.md)

Korean card 6 mutates only predictor-content binding and its fail-closed mismatch
guard. Exact route selection and lambda tuning remain independent subclaims covered by
`test_policies.py` and `test_lambda_tuning.py`; this one mutation does not sign them off.

Owner questions:

1. Derive where two model utility lines exchange order and why `Fraction` is retained
   even when quality predictions are binary64.
2. State the deterministic tie-break order in runtime routing and lambda selection.
3. How does bounded bottom-hash sampling avoid materializing all roots, and which fields
   distinguish approximate from exhaustive evidence?
4. What does a policy artifact bind, and why must cumulative artifact routing receive
   explicit remaining budget?
5. Why do pairwise utility-line intersections form a complete one-shot candidate set,
   and what additional proof would a cumulative sequence oracle require?

## 7. RouterBench hostile-data and provenance boundary

**Invariant.** RouterBench is optional, never committed, and has unresolved dataset
licensing. A user-triggered preparation step downloads one pinned artifact. Validation
checks size and SHA before a non-dispatching decoder accepts only the exact reviewed
pickle/data shape; runtime never calls `pickle.load`, pandas, or user-defined globals.
The downloader rejects symlink/non-regular destinations, owns an unpredictable
same-directory `0600` staging inode, checks that inode before atomic replacement, and
re-authenticates the installed path. Cleanup never intentionally removes a different
inode owned by another invocation.
The default prefix replay remains a smoke check. The separate nested-LODO diagnostic
requires `--nested-lodo --acknowledge-noassertion`. Human output begins with `LOCAL
OPTIONAL VALIDATION — NON-OFFICIAL, NON-REPORTABLE`, while JSON carries the same exact
string in a required warning field. The path performs no network or file writes and
authorizes no performance claim or predictor-family selection.

**Diagnostic design.** A digest framed from pinned revision, normalized domain, and
`sample_id` ranks rows independently of prompt, output, quality, and cost. Each of the
seven pinned domains contributes exactly 64 calibration rows and the next 8 evaluation
rows; evaluation returns to source order. Calibration-only per-model maximum charges
become fixed pre-call quotes, and every evaluation charge must fit its quote before any
fit. Sorted quote minimum/median/maximum mechanically define three diagnostic budgets
with weights `0.5`/`0.3`/`0.2`; they are not official tier definitions. A surface-only
bilinear predictor (no `bge-m3`) and all six baselines share the 56-row evaluation scope.
Nested LODO governs quality-predictor fitting, lambda tuning, learned replay, and the
domain-table baseline, with a disclosed approximate lambda cap of 32 candidates per
tier. The fixed quote/tier calibration pool is separate from evaluation but spans all
seven domains, so the complete diagnostic is not an end-to-end domain-shift claim.

Only aggregate provenance, structure, configuration, and completion evidence may reach
human or JSON output. Prompts, outputs, `sample_id` values, row decisions, performance,
realized-cost, and oracle-gap results remain suppressed, and no converted data,
features, predictions, learned artifact, redirected output, or result belongs in Git.
Benchmark internals receive deterministic source-order surrogate IDs, and the CLI emits
a fixed path/row-free failure envelope rather than reflecting exception text or a
traceback.
The source row grain is `sample_id`; duplicated prompt text, domain imbalance, and
heterogeneous upstream evaluators are reasons this diagnostic cannot be described as a
RouterBench paper reproduction, SKT data, an official score, or reportable evidence.

- Core: [`adapters/routerbench.py`](../src/tierroute/adapters/routerbench.py),
  [`download_routerbench.py`](../scripts/download_routerbench.py), and
  [`validate_routerbench.py`](../scripts/validate_routerbench.py)
- Strongest evidence: [`test_routerbench_adapter.py`](../tests/test_routerbench_adapter.py)
  and [`test_validate_routerbench_script.py`](../tests/test_validate_routerbench_script.py)
- Design context: [SBOM data entry](../SBOM.md#data-assets) and
  [literature/data boundary](literature-and-novelty.md)

Owner questions:

1. Why is a correct checksum necessary but insufficient for safe pickle handling?
2. Which opcode/global/shape limits prevent code execution or resource abuse?
3. Distinguish artifact SHA, decoded semantic SHA, and evaluation data/replay/scope hashes.
4. Why does an MIT code repository not establish a license for the separate dataset?
5. How are checksum-before-decode and same-file/path-replacement hazards handled, and
   how do unpredictable staging names, inode checks, and post-install authentication
   prevent concurrent or substituted staging content from being accepted? Why must
   quoted candidate costs remain separate from realized RouterBench charges?
6. Why is membership ranked only from revision, normalized domain, and `sample_id`, and
   why must mutations to prompt, quality, cost, or output leave membership unchanged?
7. Why use exactly 64 calibration plus 8 evaluation rows per domain, restore evaluation
   source order, and reject duplicate IDs or an incomplete seven-domain population?
8. Why is a calibration maximum a safer quote than a mean, and why must all evaluation
   charges be preflighted before predictor fitting?
9. Why are min/median/max quote budgets, weights, surface-only features, and the
   32-candidate lambda cap diagnostic configuration rather than contest claims, and why
   does global all-domain quote calibration prevent an end-to-end domain-shift claim?
10. Which fields may the human/JSON output expose, which data-derived materials must
    remain local, and why do domain/evaluator limitations rule out paper reproduction?

## 8. Atomic files, offline operation, build, and license compliance

**Invariant.** The bilinear predictor/policy bundle is staged under random exclusive
regular files and validated before `_save_training_artifacts` publishes the predictor
first and its bound policy last. Rename intent is recorded before the operating-system
rename so rollback can cover an asynchronous exception after rename succeeds but before
control returns to Python. This is not a guarantee for arbitrary asynchronous timing,
power loss, or concurrent writers. Failed cleanup or restoration may deliberately
preserve authenticated recovery backups for inspection. GBM artifact `save()` is a
separate single-file atomic contract, not a policy bundle. Runtime/training/evaluation
never download. The base wheel has no third-party runtime dependency. CI rejects
unallowlisted dependency metadata, unsafe or unreadable bundled evidence, and detected
GPL-family license documents; manual review still covers native and non-Python assets.

- Core: [`core/atomic_io.py`](../src/tierroute/core/atomic_io.py),
  [`predictors/artifacts.py`](../src/tierroute/predictors/artifacts.py),
  [`predictors/gbm_artifacts.py`](../src/tierroute/predictors/gbm_artifacts.py),
  [`policies/lambda_artifacts.py`](../src/tierroute/policies/lambda_artifacts.py),
  [`cli.py`](../src/tierroute/cli.py), [`check_licenses.py`](../scripts/check_licenses.py),
  [`check_spdx.py`](../scripts/check_spdx.py),
  [`build_native_ridge.py`](../scripts/build_native_ridge.py),
  [`build_native_prepared.py`](../scripts/build_native_prepared.py),
  [`audit_native_binary.py`](../scripts/audit_native_binary.py), and
  [`ci.yml`](../.github/workflows/ci.yml)
- Strongest evidence: [`test_atomic_io.py`](../tests/test_atomic_io.py),
  [`test_gbm_artifacts.py`](../tests/test_gbm_artifacts.py),
  [`test_gbm_artifact_hardening.py`](../tests/test_gbm_artifact_hardening.py),
  [`test_offline_runtime.py`](../tests/test_offline_runtime.py),
  [`test_license_gate.py`](../tests/test_license_gate.py),
  [`test_cli.py`](../tests/test_cli.py), [`test_package.py`](../tests/test_package.py), and
  [`test_reproduction_contract.py`](../tests/test_reproduction_contract.py)
- Design context: [dependency-license audit](dependency-license-audit.md),
  [SBOM](../SBOM.md), and [CONTRIBUTING](../CONTRIBUTING.md)

Owner questions:

1. Why are predictable temporary names, symlinks, input/output aliases, and policy-first
   bundle publication unsafe?
2. Which failures roll back, and why are concurrent-writer and cross-path power-loss
   atomicity not claimed?
3. Prove the base wheel is dependency-free and explain why build/dev tools still appear
   in the SBOM and license gate.
4. Which commands and environment variables demonstrate an empty, offline model cache?

Korean card 8 mutates only rollback-intent ordering. Offline execution,
dependency-free packaging, SPDX/license gates, GBM single-file saving, and native
source-portability are independent subclaims and require their own command results.

The current-main card-8 re-audit recorded 25 passes in the expanded recovery suite,
three passes in the complete package suite, 12 allowed dependency licenses, and SPDX
coverage for 146 files. Merged-main CI
[`29548281471`](https://github.com/Hbin77/tierroute/actions/runs/29548281471) passed all
five jobs at `a1d7bd7dd835a1ab88e85e805df167985ca699be`; both Python jobs executed the
locked `make reproduce-inference` and `make reproduce-training` lanes. The platform
jobs built and audited ephemeral source candidates and uploaded no artifact. This does
not approve Linux-musl, a distributable binary, the chosen toolchain, or network
isolation during environment installation. Some failure-injection tests intentionally
retain recovery backups, so the review inspects only its three dedicated mutation
basetemp directories rather than claiming that the entire temporary root is debris-free.

## Gated and intentionally incomplete work

| Item | Current boundary | What unblocks it |
|---|---|---|
| Cascade/sequential escalation | Interface can represent it; shipped policy remains one-shot | Written simulator semantics for sequential calls, budget accumulation, history schema, and final output selection |
| Cumulative sequence oracle | No cumulative oracle-gap claim | Official cumulative budget semantics followed by a sequence-level optimization and tests |
| Local `bge-m3` features | Revision/license contract only; no weights or provider shipped | Reviewed preparation/distribution plan, offline local provider, SBOM/model-card update, and locked tests |
| Dense C11 ridge solve | Project-owned source, protocol, authenticated adapter, local parity, and macOS/Windows ephemeral source-portability evidence; no binary in the wheel | Explicit local opt-in only; macOS/Linux-musl/Windows-MSVC distributable release artifacts still need link/import approval |
| Full-dimensional nested ridge | Exact 63-subset/154-block/`22N` graph, bounded in-memory coefficient-to-report references, and an authenticated file-backed single-invocation native solve/score slice exist. A public bounded Python consumer now produces fixed per-query learned and six-baseline evidence from caller-pinned native results; surface-only D4-D7 fixtures, including uneven three-model D7, strictly match the rowwise result. One local D4/N8/d1036/M1 synthetic run keeps all 1,024 embedding coordinates. PR #50's earlier session-only macOS/Windows source-portability CI passes; final PR #52 head CI run `29543435978` at `304decd0a591fcfc5e5a1e04f35bf20b22c17cea` and merged-main CI run `29543610611` at `c7b717ce1226fcfd70d696d0124aa8df294033c8` cover the current policy benchmark test and all five jobs. These are merged-main source-portability evidence, not distributable release-artifact approval, and the human walkthrough remains pending. The official D7/N34778/d1036/M11 shape is preflight-only; no provider, all-domain artifact, shipped command/trainer integration, official data, Linux-musl/distributable cross-platform release artifact, or performance claim exists | Audited offline local provider, all-domain artifact and command integration, full official-shape execution/parity, broader near-tie checks, three-platform release-artifact audits, licensed-data evidence, human walkthrough, and issue #9 completion |
| GBM artifact deployment integration | Distinct canonical `tierroute-gbm-predictor` artifact v1 is implemented at library level with bounded JSON, atomic save, bounded load, and offline reconstruction; bilinear v1 is unchanged. There is no GBM train/route selection or lambda-policy binding | Reviewed `train`/`route` and policy binding plus unbiased family-selection and deployment evidence |
| Reportable predictor-family selection | Same-fold descriptive paired runner; `selected_family=null`; no reportable selection claim | Licensed data plus preregistered untouched or selection-aware evidence |
| Official SK Telecom data | No committed data or official result | Data release plus written license/schema confirmation |

An owner must be able to distinguish every row above from implemented behavior. A
planned interface or document is not evidence that a model, policy, or benchmark result
exists.

## Focused review commands

Run these in the exact locked development environment, with an empty `HF_HOME`:

```bash
hf_home="$(mktemp -d)"
trap 'rm -rf "$hf_home"' EXIT HUP INT TERM
HF_HOME="$hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q \
  tests/test_core.py tests/test_integer_text.py \
  tests/test_budgets.py tests/test_simulator.py tests/test_json_dataset.py \
  tests/test_eval_provenance.py tests/test_eval_schemas.py tests/test_metrics.py \
  tests/test_validation.py tests/test_policies.py tests/test_baseline_evaluation.py \
  tests/test_benchmark.py tests/test_predictor_comparison.py \
  tests/test_predictor_comparison_cli.py tests/test_showcase.py \
  tests/test_feature_encoding.py \
  tests/test_features_predictors.py tests/test_bilinear_training.py \
  tests/test_gbm_core.py tests/test_gbm_training.py \
  tests/test_gbm_artifacts.py tests/test_gbm_artifact_hardening.py \
  tests/test_prepared_graph.py tests/test_prepared_store.py \
  tests/test_prepared_execution.py tests/test_prepared_reference_pipeline.py \
  tests/test_prepared_files.py tests/test_native_prepared.py \
  tests/test_native_prepared_benchmark.py \
  tests/test_native_ridge.py tests/test_ridge_solver.py tests/test_predictor_artifacts.py \
  tests/test_lambda_tuning.py tests/test_lambda_policy_artifacts.py \
  tests/test_routerbench_adapter.py tests/test_validate_routerbench_script.py \
  tests/test_atomic_io.py tests/test_offline_runtime.py \
  tests/test_license_gate.py tests/test_cli.py tests/test_package.py \
  tests/test_reproduction_contract.py
HF_HOME="$hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  make verify PYTHON=python
test -z "$(find "$hf_home" -mindepth 1 -print -quit)"
```

To review the build boundary itself, choose an already installed compiler explicitly;
the output path must be absolute and nonexistent. The helper neither searches `PATH`
nor downloads, but it cannot attest that the selected toolchain stayed offline:

```bash
python scripts/build_native_prepared.py \
  --compiler /absolute/path/to/clang \
  --output /absolute/new/path/tierroute-prepared
```

`make verify` reaches the benchmark, paired estimation, and showcase through
`training-smoke`, and
`reproduce-training` executes that fitting path. `reproduce-inference` intentionally
does not invoke `tierroute benchmark`, `tierroute compare-predictors`, or
`tierroute demo` and does not fit the bilinear predictor/lambda policy; its evaluation
fits only the outer-training domain-table baseline. GBM training and library-artifact
paths are exercised by their pytest modules, while the paired CLI smoke remains
training-only; neither path performs deployment routing or GBM policy selection.

For each boundary, make one temporary local mutation that violates its stated
invariant, run the named focused test, observe the expected failure, and restore the
mutation before committing. Never weaken a test merely to make the drill pass.

## Human owner sign-off

Only the human entrant fills this table. Use a real name or stable Git identity, an ISO
date, the exact reviewed implementation commit, the packet commit, and a short note
naming the mutation/failure drill.

| Boundary | Owner | Date | Reviewed implementation / packet commit | Status and notes |
|---|---|---|---|---|
| Router contract and exact costs |  |  |  |  |
| Budget adapters, replay, and call evidence |  |  |  |  |
| Complete evaluation-scope identity |  |  |  |  |
| Metrics, learned-versus-six-baseline nested LODO, and showcase |  |  |  |  |
| Features, ridge/GBM predictors, prepared references, file/native session, native policy benchmark, and calibration |  |  |  |  |
| Exact lambda tuning and policy artifacts |  |  |  |  |
| RouterBench hostile-data and local diagnostic boundary |  |  |  |  |
| Atomic I/O, offline, build, and licenses |  |  |  |  |

Do not batch-mark these rows complete based only on the prose above. Each status is a
claim about the entrant's present understanding and must remain independently auditable.
Blank human-owned cells mean unsigned; automation must not populate them.
