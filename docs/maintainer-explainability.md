<!-- SPDX-License-Identifier: Apache-2.0 -->

# Maintainer explainability review

[한국어 실행 워크시트](maintainer-explainability.ko.md)

## How to use this document

This is the human review packet for contest-critical tierroute code. It is deliberately
answer-first: each section states the invariant, points to the implementation and tests,
and asks questions the entrant must answer in their own words. Reading this document or
receiving a green CI result is not sign-off. The owner must trace the code, run the
focused tests, perform a mutation drill, and complete the table at the end.

The mutation walkthrough companion is pinned to implementation snapshot commit
`c6491508533655baa76c7b50bfdadacbc1612e60`. Review the current commit instead if the
code has moved, and record that exact commit in the sign-off table. A green result on
the pinned snapshot does not attest to later source changes.

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

**Invariant.** Reportable learned-policy validation uses true nested
leave-one-domain-out splits: predictor fitting, calibration, and lambda tuning remain
inside each outer training side. The domain-table baseline is also fitted only on that
side and can read only an explicit pre-call metadata tag, not the split label. The
learned router and all six policies replay the same original row order and per-query
ledger scope. Scores, exact cost evidence, and oracle-gap recovery are derived again
from bound reports rather than trusted as stored numbers. The three-step showcase
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
Bilinear artifact v1 is strict schema-bound JSON, and unknown schema/model/solver
identities fail closed. GBM state remains in-memory and has no artifact or deployment
CLI contract; the no-selection paired-estimation command is its only shipped CLI path.
The optional C11 ridge process is a training-only, explicitly authenticated solver; a
known solver ID never substitutes for its absolute path and exact binary hash.

- Core: [`features/encoding.py`](../src/tierroute/features/encoding.py),
  [`features/surface.py`](../src/tierroute/features/surface.py),
  [`features/embeddings.py`](../src/tierroute/features/embeddings.py),
  [`predictors/base.py`](../src/tierroute/predictors/base.py),
  [`predictors/training.py`](../src/tierroute/predictors/training.py),
  [`predictors/gbm.py`](../src/tierroute/predictors/gbm.py),
  [`predictors/gbm_training.py`](../src/tierroute/predictors/gbm_training.py),
  [`predictors/_ridge.py`](../src/tierroute/predictors/_ridge.py),
  [`predictors/solvers.py`](../src/tierroute/predictors/solvers.py),
  [`predictors/native_ridge.py`](../src/tierroute/predictors/native_ridge.py),
  [`native/tierroute_ridge.c`](../native/tierroute_ridge.c),
  [`predictors/calibration.py`](../src/tierroute/predictors/calibration.py), and
  [`predictors/artifacts.py`](../src/tierroute/predictors/artifacts.py), with limits in
  [`predictors/resource_limits.py`](../src/tierroute/predictors/resource_limits.py),
  plus [`policies/predictor_comparison.py`](../src/tierroute/policies/predictor_comparison.py)
- Strongest evidence: [`test_feature_encoding.py`](../tests/test_feature_encoding.py),
  [`test_features_predictors.py`](../tests/test_features_predictors.py),
  [`test_bilinear_training.py`](../tests/test_bilinear_training.py),
  [`test_gbm_core.py`](../tests/test_gbm_core.py),
  [`test_gbm_training.py`](../tests/test_gbm_training.py),
  [`test_predictor_comparison.py`](../tests/test_predictor_comparison.py),
  [`test_predictor_comparison_cli.py`](../tests/test_predictor_comparison_cli.py),
  [`test_ridge_solver.py`](../tests/test_ridge_solver.py), and
  [`test_predictor_artifacts.py`](../tests/test_predictor_artifacts.py), plus
  [`test_native_ridge.py`](../tests/test_native_ridge.py)
- Design context: [lambda/training design](lambda-tuning.md) and
  [native ridge protocol](native-ridge-protocol.md)

Owner questions:

1. Which feature statistics are fitted rather than computed independently, and why
   must their fit occur inside the outer training fold?
2. Describe the centered-ridge normal equations and Cholesky solve; why does the
   reference solver have a conservative work guard and a fixed solver ID?
3. Why is isotonic calibration fitted from out-of-fold rather than in-sample predictions?
4. Which identities make a silent schema, catalogue, solver, or training-data change
   fail closed? Trace the predictor byte, numeric-token, structure, metadata, and
   calibration limits through construction, parsing, loading, saving, and policy hash.
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
    What do the 63 unique training subsets and `22N` unique score rows change?

## 6. Exact lambda routing, tuning, and policy artifacts

**Invariant.** Runtime and tuning use one routing function for
`predicted_quality - lambda * quoted_cost`, with exact rational lambda and deterministic
tie breaks. An exhaustive claim includes every boundary/interval/tail candidate. A cap
is an explicitly labeled deterministic approximation, not a smaller exhaustive search.
Nested LODO tunes only inside each outer training side.

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

**Invariant.** Predictor/policy outputs are staged under random exclusive regular files,
validated, and published with rollback-safe ordering. Runtime/training/evaluation never
download. The base wheel has no third-party runtime dependency. CI rejects unallowlisted
dependency metadata, unsafe or unreadable bundled evidence, and detected GPL-family
license documents; manual review still covers native and non-Python assets.

- Core: [`core/atomic_io.py`](../src/tierroute/core/atomic_io.py),
  [`predictors/artifacts.py`](../src/tierroute/predictors/artifacts.py),
  [`policies/lambda_artifacts.py`](../src/tierroute/policies/lambda_artifacts.py),
  [`cli.py`](../src/tierroute/cli.py), [`check_licenses.py`](../scripts/check_licenses.py),
  [`check_spdx.py`](../scripts/check_spdx.py), and
  [`ci.yml`](../.github/workflows/ci.yml)
- Strongest evidence: [`test_atomic_io.py`](../tests/test_atomic_io.py),
  [`test_offline_runtime.py`](../tests/test_offline_runtime.py),
  [`test_license_gate.py`](../tests/test_license_gate.py),
  [`test_cli.py`](../tests/test_cli.py), and [`test_package.py`](../tests/test_package.py)
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

## Gated and intentionally incomplete work

| Item | Current boundary | What unblocks it |
|---|---|---|
| Cascade/sequential escalation | Interface can represent it; shipped policy remains one-shot | Written simulator semantics for sequential calls, budget accumulation, history schema, and final output selection |
| Cumulative sequence oracle | No cumulative oracle-gap claim | Official cumulative budget semantics followed by a sequence-level optimization and tests |
| Local `bge-m3` features | Revision/license contract only; no weights or provider shipped | Reviewed preparation/distribution plan, offline local provider, SBOM/model-card update, and locked tests |
| Dense C11 ridge solve | Project-owned source, protocol, authenticated adapter, local parity and macOS link evidence; no binary in the wheel | Explicit local opt-in only; Linux-musl and Windows-MSVC release artifacts still need link/import approval |
| Full-dimensional nested ridge | Dense sidecar still repeats feature work and 301 fits | Prepared raw-feature session, 63-subset reuse, batched `22N` scores, leakage/noninterference parity, three-platform audits, and issue #9 completion |
| GBM artifact and deployment CLI | In-memory state; paired estimation only | Separate artifact schema plus reviewed `train`/`route` integration |
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
  tests/test_ridge_solver.py tests/test_predictor_artifacts.py \
  tests/test_lambda_tuning.py tests/test_lambda_policy_artifacts.py \
  tests/test_routerbench_adapter.py tests/test_validate_routerbench_script.py \
  tests/test_atomic_io.py tests/test_offline_runtime.py \
  tests/test_license_gate.py tests/test_cli.py tests/test_package.py \
  tests/test_reproduction_contract.py
HF_HOME="$hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  make verify PYTHON=python
test -z "$(find "$hf_home" -mindepth 1 -print -quit)"
```

`make verify` reaches the benchmark, paired estimation, and showcase through
`training-smoke`, and
`reproduce-training` executes that fitting path. `reproduce-inference` intentionally
does not invoke `tierroute benchmark`, `tierroute compare-predictors`, or
`tierroute demo` and does not fit the bilinear predictor/lambda policy; its evaluation
fits only the outer-training domain-table baseline. The in-memory GBM path is exercised
by its pytest modules and the training-only paired CLI smoke, never by deployment
routing.

For each boundary, make one temporary local mutation that violates its stated
invariant, run the named focused test, observe the expected failure, and restore the
mutation before committing. Never weaken a test merely to make the drill pass.

## Human owner sign-off

Only the human entrant fills this table. Use a real name or stable Git identity, an ISO
date, the exact reviewed commit, and a short note naming the mutation/failure drill.

| Boundary | Owner | Date | Reviewed commit | Status and notes |
|---|---|---|---|---|
| Router contract and exact costs |  |  |  |  |
| Budget adapters, replay, and call evidence |  |  |  |  |
| Complete evaluation-scope identity |  |  |  |  |
| Metrics, learned-versus-six-baseline nested LODO, and showcase |  |  |  |  |
| Features, ridge/GBM predictors, and calibration |  |  |  |  |
| Exact lambda tuning and policy artifacts |  |  |  |  |
| RouterBench hostile-data and local diagnostic boundary |  |  |  |  |
| Atomic I/O, offline, build, and licenses |  |  |  |  |

Do not batch-mark these rows complete based only on the prose above. Each status is a
claim about the entrant's present understanding and must remain independently auditable.
Blank human-owned cells mean unsigned; automation must not populate them.
