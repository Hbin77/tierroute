<!-- SPDX-License-Identifier: Apache-2.0 -->

# tierroute

[한국어](README.ko.md)

`tierroute` is an offline-first, budget-aware LLM router. It maps each prompt and
budget tier to an affordable candidate model with a one-shot Lagrangian policy:

```text
choose m = argmax_m [predicted_quality(prompt, m) - lambda(tier) * cost(m)]
```

The project is being developed for the student division of the 2026 Open Source
Developer Competition, SK Telecom challenge **“Efficient LLM Routing Challenge.”**
It is currently pre-alpha: the routing contracts, replay simulator, six baselines,
quality and exact quote-error metrics, leakage-aware calibrated bilinear training,
an in-memory deterministic GBM reference trainer, paired descriptive family estimation,
exact tier-lambda tuning, strict v1 bilinear-predictor/policy artifacts, and an
external-data-free demo are implemented.
The CLI selects a model but does **not** call an LLM or return a model completion.

## Quickstart

Python 3.10 or newer is required. From a fresh checkout:

```bash
cd tierroute
python -m venv .venv
```

Activate it with `. .venv/bin/activate` in a POSIX-compatible shell or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install -e .
```

Run one routing decision, all six replay baselines, the learned-versus-baseline
benchmark, paired predictor estimation, and the training-backed three-step
showcase:

```bash
tierroute route "Prove that sqrt(2) is irrational." --tier fast
tierroute evaluate
tierroute benchmark --budget-scope per-query
tierroute compare-predictors --budget-scope per-query
tierroute demo
```

The equivalent module entry point is `python -m tierroute`. Machine-readable output
is available for `route`, `evaluate`, `benchmark`, `compare-predictors`, `demo`, and
`train` with `--json`;
a compatible versioned replay JSON can be supplied to evaluation and benchmarking:

```bash
tierroute route "Debug this Python function" --tier balanced --json
tierroute evaluate --data src/tierroute/data/synthetic.json --json
tierroute benchmark --budget-scope per-query \
  --data src/tierroute/data/synthetic.json --json
tierroute compare-predictors --budget-scope per-query --json
HF_HUB_OFFLINE=1 tierroute demo --json
```

`route --json` is a pre-execution decision: `cost` remains a semantic alias for
`quoted_cost`, while `realized_cost` is `null`. `evaluate --json` replays logged
outcomes and adds per-tier and cross-tier `cost_evidence` for the calls that replay
actually executed. Neither command makes a live provider call.

The bundled prompts, costs, outputs, predicted qualities, scorecard, and benchmark
rows are project-authored **synthetic smoke-test values**. They verify wiring and are
not a benchmark result, an empirical model comparison, or a competition score.

### Three-step learned-router showcase

The human and machine-readable showcase commands are:

```bash
tierroute demo
tierroute demo --json
```

The demo presents a deterministic three-prompt stream with one bundled synthetic row
at each of Fast, Balanced, and Premium. For every step, it takes the learned/tuned
policy fitted strictly on that row's outer-LODO training side and runs a direct
one-example, one-tier `OfflineSimulator` replay. That direct result must match the same
row and tier in the nested-LODO learned result audited by `tierroute benchmark`; a
mismatch fails the showcase instead of being hidden.

Each step reports its illustrative per-query budget, quoted and realized cost,
observed synthetic quality, independent per-query oracle quality, running realized
cost, and the unweighted running quality-retention ratio:

| Step / tier | Bundled row | Budget | Model | Quoted → realized | Observed / oracle | Running cost | Retention |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 / Fast | `synthetic-science-001` | 0.35 | `swift` | 0.2 → 0.2 | 0.78 / 0.78 | 0.2 | 100% |
| 2 / Balanced | `synthetic-math-002` | 0.7 | `steady` | 0.6 → 0.6 | 0.75 / 0.75 | 0.8 | 100% |
| 3 / Premium | `synthetic-code-002` | 1 | `expert` | 1 → 1 | 0.96 / 0.96 | 1.8 | 100% |

```text
running quality retention = sum(observed synthetic quality so far)
                            / sum(independent per-query oracle quality so far)
```

If the cumulative oracle sum is zero, retention is undefined and the CLI emits
`N/A`/JSON `null` instead of dividing by zero.

> **Interpretation boundary:** the running realized cost adds calls from different
> tiers that each used an independent per-query ledger. It is reporting-only, not an
> official shared or cumulative budget. The retention denominator is the sum of
> independent per-query oracle values, not a sequence-level oracle; the ratio is not
> oracle-gap recovery and does not use the official tier weights. Every prompt, cost,
> quality, and oracle value is project-authored synthetic wiring evidence, not an
> empirical or competition result.

The three selected stream rows are a presentation view only. Human output follows them
with a clearly separate full-population learned-plus-six-baseline table; JSON uses the
versioned `tierroute-routing-stream-showcase` schema and keeps the three rows under
`stream.steps`, the reporting rules under `accounting`, and the complete benchmark
under `benchmark_evidence`. That evidence is independently available through
`tierroute benchmark --budget-scope per-query`; the demo does not replace or summarize
the full population with its three selected rows. Each JSON step exposes `budget_limit`,
`cost.quoted`, `cost.realized`, `cost.cumulative_realized_reporting_only`,
`quality.observed`, `quality.per_query_oracle`, and `quality.cumulative_retention`.

### Offline predictor and policy training

Training and inference use no third-party numerical package. A project-owned,
deterministic centered-ridge Cholesky solver fits every model target against one shared
factorization and leaves the intercept unregularized. The resulting surface-feature
artifact is strict canonical JSON and records the solver ID used to produce it:

The deployment commands below remain bilinear-only. GBM state is still in-memory only:
it has no versioned artifact or `train`/`route`/showcase integration. The separate
`compare-predictors` command evaluates both fixed families but never selects one and
does not authorize a performance claim.

```python
from tierroute.adapters import load_evaluation_dataset
from tierroute.predictors import GbmTrainingConfig, fit_calibrated_gbm

examples = load_evaluation_dataset().examples
predictor = fit_calibrated_gbm(examples, config=GbmTrainingConfig())
model_ids = tuple(model.model_id for model in examples[0].candidate_models)
scores = predictor.predict_many("Explain why binary search is logarithmic.", model_ids)
```

This bundled-data call is a deterministic wiring demonstration, not a measured
predictive-quality result.

```bash
tierroute train --output artifacts/synthetic-bilinear.json --json
tierroute route "Prove that sqrt(2) is irrational." \
  --tier balanced \
  --artifact artifacts/synthetic-bilinear.json \
  --json
```

To fit the complete one-shot policy, make the unresolved budget semantics explicit and
write a separate policy artifact:

```bash
tierroute train \
  --output artifacts/synthetic-bilinear.json \
  --policy-output artifacts/synthetic-policy.json \
  --budget-scope per-query \
  --json
tierroute route "Prove that sqrt(2) is irrational." \
  --tier balanced \
  --artifact artifacts/synthetic-bilinear.json \
  --policy-artifact artifacts/synthetic-policy.json \
  --json
```

Policy fitting creates inner-LODO out-of-fold predictions keyed by private example ID,
streams every non-negative exact rational pairwise quality/cost breakpoint occurrence,
and replays every retained candidate through the selected budget ledger. By default,
the CLI keeps a bounded deterministic `bounded-bottom-hash-v2` sample of roots plus the
minimum and maximum, derives boundaries, midpoints, and a tail from those retained roots, then
rank-spaces the result to at most 257 candidates. If every unique root and derived
candidate fits that cap, the result remains complete and reports `exhaustive: true`
with its exact count. Only actual truncation is approximate; it reports
`exhaustive: false` and an unknown complete count (`null`), together with the search
strategy and observed breakpoint-occurrence count. Use `--exhaustive-lambda-search`
to request materialization and evaluation of the full exact finite set. Before any
predictor fitting or root materialization, a conservative preflight refuses more than
10,000,000 pair scans, 100,000 possible candidates, 100,000,000 utility evaluations,
a 256 MiB estimated peak for exact-rational candidate state, or 8 MiB of estimated
serialized policy evidence. Only after reviewing all five estimates may a caller add
`--allow-large-exhaustive-search` to an exhaustive CLI run. The default
257-candidate run fits the bundled synthetic data, but every cap is checked against the
actual dataset's retained-work and integer-width estimates and must be reduced if it
fails.
At the pinned RouterBench shape (34,778 rows, 11 models, three tiers), the conservative
bounds are 3,825,582 candidates and about 4.39 trillion utility evaluations, so the
257-candidate default would require 294,952,218 evaluations and is refused. A cap of 64
requires a conservative 73,451,136 evaluations and is the documented starting point
for that full shape; its 1,912,790 pair scans are also below the separate scan limit.
Every run must still pass the dataset-dependent artifact-size estimate. Its artifact
labels whether the retained search was complete or truncated. The selected lambda
itself always remains an exact numerator/denominator
pair. Version 2 hashes signed, self-delimiting binary integer identities, avoiding
Python's decimal integer rendering limit; version-1 strategy metadata remains loadable
because artifacts embed the retained values and strategy version explicitly.

A policy trained with `--budget-scope cumulative` can be routed only when the caller
also supplies the current exact state with `--remaining-budget`. This command does not
invent an initial balance or silently reuse a per-query assumption.

Use `--data path/to/replay.json` on training and benchmark commands for another
version-1 replay dataset. The `train` command fits deployment artifacts on all supplied
rows; both isotonic calibration and lambda selection use out-of-fold predictions. It
does not produce a reportable benchmark result.

Use the dedicated benchmark runner for leakage-free learned-versus-baseline evidence:

```bash
tierroute benchmark --budget-scope per-query
tierroute benchmark --budget-scope per-query \
  --data path/to/replay.json --json
```

The command performs true nested LODO: each outer training side gets its own inner-LODO
out-of-fold predictions and lambda tuning, then a predictor refit on that complete
outer-training side; only the untouched outer domain is scored. It then replays the
learned router and all six canonical baselines once over the identical original-order
population under the same `PerQueryBudgetLedger` and `EvaluationScopeIdentity`. JSON
output includes counts and a versioned SHA-256 digest binding the held-out domain and
exact ordered train/test membership instead of disclosing raw example IDs. This is a
compact reproducibility identity, not authenticated proof. It also records tier budget
limits and weights, resolved baseline roles/seeds/thresholds/rule identities, and the
requested lambda-search cap or exhaustive override so the weighted result is
independently reproducible from the replay. A separate versioned SHA-256 evidence
digest binds the baseline parameters to the exact ordered call decisions they produced.
The command runs offline and accepts only `--budget-scope per-query`. Cumulative
benchmarking and cascade claims remain gated until the organizer confirms sequence-level
budget and call-history semantics and tierroute implements a sequence-level oracle.

Running this command with the bundled synthetic data proves benchmark wiring only. It
is not empirical evidence. With `--data`, the caller is responsible for the replay
data's license and for the validity of any benchmark or competition claim derived from
the output.

Use the separate paired-estimation runner to inspect the two fixed surface-only
predictor families on the same outer evidence:

```bash
tierroute compare-predictors --budget-scope per-query
tierroute compare-predictors --budget-scope per-query \
  --data path/to/replay.json --json
```

Before either family fits or embeds, this command enumerates the complete outer LODO,
lambda-tuning LODO, and calibration LODO GBM call graph and applies the reviewed
aggregate split-scan limit. It computes the six baselines once, requires both family
results to share the exact replay, scope, tier, fold, catalogue, search, and baseline
evidence. With `--json`, it reports raw binary64 `GBM - bilinear` tier, weighted,
oracle-gap, and held-out-domain deltas; the human view rounds its displayed global
weighted/oracle-gap deltas and omits the domain table. Missing operands produce JSON
`null`; weights are never redistributed. The schema fixes `selection_protocol` to
`none-paired-estimation`,
`selected_family` to `null`, and `performance_claim_allowed` to `false`, and has no
`winner` field. Bundled output is `SYNTHETIC-ONLY`; `--data` output is
`UNVERIFIED-USER-DATA`. Neither state supports a superiority, deployment, quality-gain,
or cost-savings claim. Selecting a family requires separately held-out evidence or an
additional family-selection-aware validation protocol.

The built-in solver is an auditable reference backend for the surface schema and
modest matrices, with complexity `O(n*d^2 + d^3)`. tierroute also contains an
experimental project-owned C11 implementation of one bounded dense solve. Its
versioned protocol, authenticated subprocess adapter, shared multi-target Cholesky,
resource preflight, malformed-input corpus, and unprojected 1,024-feature parity tests
are documented in [the native protocol](docs/native-ridge-protocol.md). Source is in
the sdist only, and the wheel contains no executable or native dependency. The default
trainer and CLI behavior remain on the Python reference solver; only an explicit
`train --ridge-solver native-c11` opt-in selects a caller-chosen local executable and
authenticates that exact byte sequence.

From a source checkout, the helper can invoke an explicitly chosen system compiler to
build a new local candidate. The helper performs no download or PATH discovery, but it
does not sandbox the compiler or prove that the compiler/toolchain itself stayed
offline. Both arguments must be absolute, the output must not already exist, and the
command emits the source and executable SHA-256 values:

```bash
python scripts/build_native_ridge.py \
  --compiler /absolute/path/to/clang \
  --output /absolute/new/path/tierroute-ridge
```

Use the exact `sha256` emitted by that command. The helper and adapter reject executable
candidates larger than 16 MiB. They also reject paths beginning with `//` or `\\`, which
covers UNC and device-style spellings on every host; an already mapped drive or mounted
network filesystem cannot be detected portably and remains the caller's responsibility.
The CLI neither searches for nor builds the executable, records the digest but not the
executable path in its JSON result, and does not need either credential when routing
from the resulting predictor artifact:

```bash
tierroute train \
  --output /absolute/new/path/predictor.json \
  --ridge-solver native-c11 \
  --native-ridge-binary /absolute/new/path/tierroute-ridge \
  --native-ridge-sha256 SHA256_FROM_BUILD_OUTPUT \
  --json
```

The digest authenticates the bytes the caller selected; it is not an approval, source-
provenance attestation, import audit, or proof that those bytes cannot use a network.
For native training JSON, `network_used` is therefore `null`,
`python_orchestration_network_used` is `false`, and `native_binary_audit` is
`caller-responsibility-unapproved`. Resource preflight authenticates the bounded binary
before embedding materialization, and `solve` authenticates it again while creating the
private snapshot to close the replacement window. The configured timeout begins only
after the child process starts; pre-authentication filesystem I/O and request
serialization are bounded by byte ceilings but are not covered by that child timeout.

This dense sidecar alone does not make the reportable full RouterBench run feasible:
the current nested path still repeats feature work and 301 fits. An experimental
[prepared graph contract](docs/prepared-session-graph.md) now proves that the
seven-domain nested-evaluation graph contains 63 unique base-training subsets,
154 subset/domain score blocks, and `22N` scored-row memberships, and it preflights a
binary64 modeled-buffer and dominant-numeric-work estimate before enumeration. It does
not execute any fit or score, and the estimate is not a peak-memory or complete-work
bound. A separate bounded [prepared feature-store reference](docs/prepared-feature-store.md)
now snapshots canonical little-endian binary64 fit rows from caller-checked source and
precomputed-embedding digests, builds reusable per-domain Welford moments, and combines
only included domains with Chan arithmetic to recover training-only tags and population
scales. Excluded-domain mutation and direct-constructor adversarial tests protect this
isolation boundary. The reference performs no provider inference or file I/O and does
not solve, score, calibrate, or replace the default trainer; its arithmetic is not yet
bitwise parity with the current row path.

Full training with the planned 1,024-dimensional bge-m3 embedding (up to 1,036 total
features) remains gated on an audited offline local provider, a scalable persistent
prepared session with coefficient and batched-score execution, end-to-end parity, plus
audited Linux-musl and Windows-MSVC artifacts. tierroute will not silently reduce or
discard embedding dimensions. The existing row-training path keeps its conservative
operation guard, static reviewed solver ID, pre-embedding preflight, and unknown-ID rejection;
inference remains dependency-free because it uses only stored coefficients.

## What is implemented

- Context-independent exact `Decimal` cost accounting and typed
  `RouterState`/`RouterAction` contracts; caller precision cannot round away an
  overspend. Nonzero costs are supported within decimal positions `-100000` through
  `99999`, with at most 100,000 coefficient digits; inputs or results outside this
  explicit resource contract fail before silent underflow or unbounded expansion.
- Swappable per-query and cumulative budget ledgers; the demo uses illustrative
  per-query limits until the official budget scope is confirmed.
- One-shot lambda routing with exact rational utility, immutable per-tier schedules,
  complete exhaustive breakpoint search or explicitly labeled truncated
  bounded-memory approximate search, and six reproducible baselines.
- Full-information offline replay: ground-truth quality and uncalled outputs never
  reach `RouterState`. Every simulated call that consumes a logged outcome records its
  quote, realized charge, ledger balance snapshots, and ledger result. A realized
  overspend is still an executed replay call and remains in exact spend evidence.
- Every full evaluation report carries a required `EvaluationScopeIdentity` using
  `tierroute-evaluation-scope-v1` over the ordered tier specs, call cap, complete
  ordered replay, outputs, labels, candidate order, and policy-visible metadata.
  Replay-visible costs and metadata are copied into one canonical immutable snapshot
  before routing; unsupported or cyclic objects fail closed instead of being
  serialized with `repr` or pickle.
- A fitted surface-feature schema (log-scaled counts, code/math signals, and
  prompt-derived domain tags), project-owned deterministic centered-ridge fitting,
  inner-LODO out-of-fold predictions, and separate isotonic calibration per model.
- Dependency-free squared-error gradient boosting with deterministic regression
  stumps, one ensemble per model, stable ordering and tie-breaking, conservative
  pre-embedding resource guards, inner-LODO out-of-fold prediction, and per-model
  isotonic calibration. A complete nested-work preflight and paired descriptive runner
  cover modest surface-only replays; synthetic tests establish algorithm wiring only.
- Canonical, strictly validated JSON bilinear predictor artifacts v1; pickle is never
  accepted for predictor loading. Reads, parsing, serialization, saving, and policy hashing share a
  32 MiB UTF-8 limit. Version-1 structure is capped at 4,096 models, training domains,
  and feature tags; 16,384 total feature dimensions; 1,000,000 numeric scalars; 640
  characters per JSON number; 4 KiB per metadata value; and 1 MiB aggregate metadata.
  Before decoding, a lexical pass that does not materialize decoded JSON values caps
  nesting at 32, JSON string tokens at 32,768, each encoded string token at 24,578
  characters, and opening containers/commas at 1,100,000; numeric callbacks allow
  1,000,005 tokens including five fixed fields.
  Direct containers are snapshotted once and numeric parameters normalize to finite
  binary64. Each calibrator has at most one point per recorded training example. These
  limits cover the planned 11-model, 1,036-feature, 34,778-row RouterBench/bge-m3 shape
  with explicit headroom. Batch prediction vectorizes or embeds each prompt batch once.
- Canonical policy artifacts bind the exact predictor hash, training/metric-relevant
  replay content and order, tier specs, ledger identity, and retained candidate-search
  evidence. They record the OOF prediction hash as audit metadata; verifying it
  requires reproducing the cross-fitted prediction table because routing has no OOF
  table to recompute it from. Loading is bounded to 8 MiB, 404,096 decimal digits per
  exact integer (covering candidates derivable from the core cost contract), and
  100,000 retained candidates per tier before expensive parsing. Ledger-adapter names
  are limited to 4 KiB; the pre-fit artifact estimate includes actual encoded domains
  and tier-budget text rather than treating metadata as a fixed-size constant.
- Predictor and policy files use random exclusive staging, post-write validation, and
  rollback-safe policy-last bundle replacement; input aliases and unsafe output nodes
  fail before fitting. Ordinary OS/Python failures roll back every attempted path, but
  concurrent writers and power-loss atomicity across unrelated pathnames are not
  supported.
- True nested LODO orchestration keeps every outer domain out of predictor fitting,
  calibration, and lambda tuning.
- A report-shaped per-query benchmark CLI compares that nested-LODO learned router
  against all six baselines on one identical evaluation scope and publishes compact,
  versioned outer-fold membership digests.
- A separate paired-estimation CLI runs calibrated bilinear and GBM predictors over
  identical nested-LODO evidence, shares one six-baseline evaluation, emits
  full-precision descriptive deltas in machine-readable JSON, and hard-codes
  no-selection/no-performance-claim metadata.
- A three-step Fast/Balanced/Premium showcase directly replays the corresponding
  outer-fold learned policies through `OfflineSimulator`, checks agreement with the
  nested result, and labels its mixed-tier running cost and unweighted retention as
  synthetic reporting-only values.
- A per-query outer-LODO six-baseline suite fits every domain table on its outer
  training side, records fold evidence, and replays all methods once on the same
  original-order rows. A live guard verifies that the actual ledger used by every
  replay resets, charges, and reports per-query accounting as declared.
- Tier-weighted quality, oracle-gap recovery, and deterministic leave-one-domain-out
  (LODO) folds, plus exact per-tier and cross-tier quote-versus-realized diagnostics.
  No random-split helper is provided.
- Strict JSON loading plus an opt-in, pinned RouterBench boundary adapter.

Without `--artifact`, the no-download CLI uses a transparent synthetic demo predictor.
The deterministic GBM state remains in-memory; the paired-estimation runner is the only
shipped CLI path that fits it. A local `bge-m3` embedding backend, GBM
artifacts and deployment CLI integration, and a licensed reportable family-selection
experiment remain planned. No predictor-family superiority claim is made.

## Router contract and architecture

The stable decision boundary is:

```text
state(prompt, budget_tier, remaining_budget, call_history, candidate_models)
  -> CallModel(model_id) | SelectOutput(history_index)
```

Ground-truth quality and uncalled outputs exist only in the replay harness. Costs have
no built-in currency or token unit: an adapter normalizes the challenge-specific unit
before creating core objects. Policies see only pre-call quoted costs; realized charges
remain private with logged outcomes until a call is replayed. Dataset IDs and
split-only domain labels are also absent from ordinary router state. The non-deployable
oracle and outer-fold replay schedule receive a private example key through a nominal
evaluation-only boundary. The schedule contains only decisions fitted on outer training
rows from pre-call observable metadata; it never injects the split label into policy
state.

`ReplayCall` evidence belongs only to evaluation results and does not create a new
router label channel. `QueryResult.cost` is the exact sum of every executed call's
realized charge, including a call whose ledger result is false. Calls rejected before
an outcome is replayed are absent. `selected_call_index` identifies the returned logged
call today (normally index 0 in one-shot replay); selecting a nonzero or earlier call is
future cascade readiness, not a claim that a history-adaptive cascade is implemented.
Balance snapshots and `within_budget` preserve the chosen adapter's semantics rather
than deriving an official budget rule in core.

```text
JSON / RouterBench boundary ──> typed replay examples ──> OfflineSimulator
                                      │                       │
prompt ─> fitted feature encoder ─> calibrated predictor ─> policy <─ budget ledger
                                                     │
                                            CallModel / SelectOutput

core/        stable state, action, model, and validation contracts
features/    offline surface features, fitted schema, local embedding contract
predictors/  bilinear training/artifacts, in-memory deterministic GBM, calibration
policies/    exact one-shot lambda policy, tuning/artifacts, baselines, paired estimation
eval/        replay, accounting protocol, metrics, planning, and LODO
adapters/    budget-scope and external-dataset uncertainty boundaries
```

The simulator defaults to one call per query. Cascade escalation remains disabled
unless SK Telecom confirms sequential multi-call evaluation semantics; any future
schema or accounting changes stay in `adapters/` rather than leaking into the core.

## Evaluation

For tier `t`, let `Q_t` be mean quality across all feasible queries and `w_t` its
configured weight. The primary local summary is:

```text
weighted tier quality = sum_t(w_t * Q_t) / sum_t(w_t)
```

An incomplete or budget-infeasible tier makes the weighted score unavailable; its
weight is never redistributed. The bundled fixture uses Fast/Balanced/Premium weights
`0.5/0.3/0.2` to exercise low-budget emphasis, but these are illustrative rather than
official SK Telecom weights.

Cost evidence is computed over executed logged replay calls. For one call,
`underquoted` means `realized_cost > quoted_cost`, while `overquoted` means the reverse.
The total absolute quote error sums each call's non-negative error magnitude, so equal
and opposite errors cannot cancel. The separate net error reports the exact direction
and magnitude of `sum(realized) - sum(quoted)` without float arithmetic or a
division-by-zero-prone percentage. Every tier row also binds call-level realized cost
to `BudgetReport.spent` and ledger over-budget counts. The overall row is only a
cross-tier diagnostic: tiers have independent ledgers, so it is not a shared budget or
a budget-compliance verdict. The legacy top-level `total_cost` and its explicit
`total_realized_cost` alias both equal the overall realized total and have the same
cross-tier-only meaning.

Under the current per-query accounting, oracle-gap recovery measures how much of the
weighted quality interval from always-cheapest to the independently budget-feasible
per-query oracle was recovered:

```text
sum_t w_t * (Q_router,t - Q_cheapest,t)
-------------------------------------------------
sum_t w_t * (Q_oracle,t - Q_cheapest,t)
```

It is undefined when the oracle and cheapest scores are equal, and negative values are
preserved. The bundled oracle planner is a per-query upper bound only. It is not a
cumulative-stream oracle: a cumulative report needs a sequence-level plan that is not
implemented yet. The six baselines are:

| Baseline | Decision rule |
| --- | --- |
| `always-cheapest` | Lowest cost, then model ID for ties |
| `always-premium` | Explicitly designated premium model; may be infeasible in a lower tier |
| `random` | Seeded, order-independent choice among affordable models |
| `length-heuristic` | Strong model for long/code/math prompts when affordable |
| `oracle` | Privileged per-query, budget-feasible quality upper bound |
| `domain-best-table` | Per-tier mean-quality table fitted from observable training tags, with cheapest fallback |

Lambda tuning maximizes this same realized metric, not a proxy loss. For each prompt,
model utilities are affine functions of lambda, so decisions can change only at exact
pairwise quality/cost intersections. The exhaustive search evaluates every boundary,
one representative inside every open interval, and one tail value after the final
boundary. Each candidate is replayed through `OfflineSimulator`; infeasible candidates
cannot win. Since tiers have independent ledgers and positive weights, optimizing each
tier independently is exactly equivalent to a Cartesian joint search over the retained
candidate sets. This is a full exact finite joint optimum whenever the retained set is
marked exhaustive; a truncated bounded search remains approximate. See
[docs/lambda-tuning.md](docs/lambda-tuning.md) for the proof, tie-breaks, and leakage
boundary.

`tierroute evaluate` calls `evaluate_per_query_lodo_baselines`. Each outer fold fits its
domain table on training rows only; only that fold's test decisions are retained. All
six methods are then replayed once over the identical original row order and checked
against the same per-query accounting contract. The bundled data and tier weights are
still synthetic smoke inputs, so their numbers are not benchmark evidence. Cross-report
metrics require the same evaluation-scope identity before checking tier and ledger fields.
The six-baseline constructor then recomputes every score, realized total, quote summary,
and oracle-gap value from its own reports; mixed or stale rows fail closed. JSON and
text CLI output expose the scope algorithm, digest, and `max_calls_per_query`.

`tierroute benchmark --budget-scope per-query` adds the calibrated bilinear one-shot
router through true nested LODO, then compares it with those same six baseline reports.
The learned report and every baseline must carry the identical evaluation-scope
identity, tier specifications, query order, and per-query accounting evidence. Each
outer fold records training/test counts and a
`tierroute-fold-membership-sha256-v1` digest over its held-out domain and exact ordered
memberships; the CLI does not expose the underlying example IDs. This digest is compact
reproducibility evidence, not an authenticated signature. Cumulative and cascade
evaluation remain gated as described above.

`tierroute compare-predictors --budget-scope per-query` keeps that existing bilinear
benchmark contract intact and adds an independently tuned calibrated GBM result over
the identical outer folds. The six baselines are evaluated once and shared. Deltas are
descriptive `GBM - bilinear` estimates; using the same outer evidence to choose a family
would introduce family-selection bias, so the result deliberately contains no winner
or deployment recommendation.

`tierroute demo [--json]` is intentionally narrower than that benchmark. It selects
three bundled rows, one per tier, and directly replays each row with the same
outer-training-only learned policy. Its running realized cost is a mixed-tier display
sum over independent per-query ledgers. Its unweighted quality retention is
`sum(observed) / sum(independent per-query oracle)`. Neither value is shared-budget
accounting, a sequence-level oracle comparison, or oracle-gap recovery. The human
command appends the separate full benchmark table; JSON retains it under
`benchmark_evidence`, outside the three-row `stream.steps` array.

The scope digest is an accidental-mix and reproducibility identity, not an authenticated
signature. It excludes router actions so different policies remain comparable. Ledger
implementation semantics cannot be hashed safely; the metric layer separately compares
the adapter name, configured/effective limits, query order, and recorded accounting.
The exact field and canonical-byte contract is documented in
[docs/evaluation-scope.md](docs/evaluation-scope.md).

A dataset domain reaches `RouterState` only when its adapter explicitly places a valid
pre-call tag in `router_metadata["domain"]`; split-only labels remain private. If the
observable tag is identical to the LODO split domain, the held-out tag is unseen and the
domain-table baseline deliberately reduces to always-cheapest through its fallback.
Cross-domain generalization is possible only when a separately observable tag is shared
across split domains. Cumulative comparison remains gated on a sequence-level oracle.

## Data and model assets

Runtime routing and evaluation make no network calls. Downloads must be explicit,
separate preparation steps; automatic Hugging Face fallback is prohibited. Downloaded
datasets and model weights are ignored by Git and must not be committed without a
verified redistribution license. Locally fitted files under `artifacts/` are also
ignored by default so data-derived parameters receive an explicit provenance and
license review before any intentional release commit.

### Bundled synthetic data

`src/tierroute/data/synthetic.json` and its license sidecar are authored for this
project and licensed Apache-2.0. The replay JSON schema is versioned (`schema_version:
1`) and records tier specifications plus every candidate model's output, exact string
cost, and quality for each prompt.

All `--data` commands share one strict, bounded loader: stable regular-file reads,
strict UTF-8/JSON, exact fields and primitive types, 256 MiB input, 100,000 examples,
1,000,000 total outcomes, 1 MiB per prompt/output, and bounded outer/nested LODO work.
There is no unlimited override. See the complete
[version-1 schema, limits, and migration rule](docs/replay-json.md).

### RouterBench (optional and opt-in)

RouterBench is not bundled. Its dataset card declares no license at the pinned revision,
so tierroute records it as **`NOASSERTION`** and does not grant redistribution rights.
Review the [dataset card at the pinned revision](https://huggingface.co/datasets/withmartian/routerbench/blob/784021482c3f320c6619ed4b3bb3b41a21424fcb/README.md)
and obtain any permission you require before opting in.

- Artifact: `routerbench_0shot.pkl`
- Revision: `784021482c3f320c6619ed4b3bb3b41a21424fcb`
- Size: `99,567,659` bytes
- SHA-256: `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`

No optional reader packages are required. Run the explicit download from an installed
core checkout:

```bash
python scripts/download_routerbench.py \
  --output data/routerbench/routerbench_0shot.pkl
```

On POSIX systems the downloader creates the partial file and normalizes the verified
destination to owner-only mode `0600`; inability to enforce that mode fails closed.
Each invocation owns an unpredictable same-directory staging name, verifies its open
descriptor identity before replacement, and re-authenticates the installed destination
before success. Symlink/non-regular destinations are rejected, so concurrent runs do
not share or delete one predictable staging path.

The upstream file uses the pickle wire format, but tierroute does **not** call
`pickle.load`, `pickle.Unpickler`, or `pandas.read_pickle`. The adapter first requires
the exact pinned size and SHA-256, then uses a project-owned, non-dispatching standard-
library opcode decoder. Referenced globals remain inert data: no callable named by the
payload is imported or invoked. Unexpected opcodes, globals, block layouts, dtypes, shapes, memo
references, trailing bytes, or table schema are rejected. This decoder intentionally
supports only the exact artifact above and adds no pandas or NumPy dependency.

As a decoder regression oracle, local validation also requires exactly 36,497 rows by
37 columns and the canonical semantic SHA-256
`7b4749ad5c4bdb338c2317b306c382680b1a23dc83c73e29ab805b8f7e472e87`. The
semantic digest frames column order, UTF-8 strings, and IEEE-754 binary64 values; it
does not replace the artifact SHA-256 used for authentication. The declared benchmark
mapping retains 34,778 examples across 11 models and 7 LODO domains.

After downloading, validate and replay a deterministic prefix with Hugging Face offline
mode set. The validator itself contains no network client and reads only the local path:

```bash
HF_HUB_OFFLINE=1 python scripts/validate_routerbench.py --limit 200
```

The full pinned artifact validation checks 34,778 in-scope rows across 11 models and
identifies 7 LODO domains, then converts only the requested replay prefix to keep memory
bounded. Those counts are artifact/schema validation facts, **not** a model-quality
benchmark claim. The current
`evaluate --data` option remains JSON-only and does not accept this pickle directly.
Because RouterBench stores post-response realized costs, validation fits model-level
pre-call quotes on a separate calibration prefix and never exposes the routed row's
realized charge to a policy. That artifact-order prefix contains only `arc-challenge`
rows, so the default command is a structural smoke check: its replay performance and
cost values are suppressed and its quotes are not representative cross-domain evidence.
The balanced diagnostic below replaces that prefix for learned-policy wiring checks.

The authenticated wire table is materialized in memory. Allow at least 512 MiB of
headroom; the default prefix validation measured about 290 MB maximum RSS on the
reference Python 3.12 environment. `--limit` bounds typed replay retention, while
`--limit 0` intentionally replays all post-calibration rows.

The prefix replay above remains the default smoke path. A separate, explicitly
acknowledged local diagnostic exercises the learned policy and all six canonical
baselines without changing that default:

```bash
HF_HUB_OFFLINE=1 python scripts/validate_routerbench.py \
  --nested-lodo --acknowledge-noassertion

# Emit the same provenance/structure evidence as one JSON document.
HF_HUB_OFFLINE=1 python scripts/validate_routerbench.py \
  --nested-lodo --acknowledge-noassertion --json
```

Human output starts with **`LOCAL OPTIONAL VALIDATION — NON-OFFICIAL,
NON-REPORTABLE`**; JSON records that exact banner in its required warning field. This
is a network-free diagnostic over external RouterBench data whose license remains
`NOASSERTION`; it is not SK Telecom data, an official challenge score, or reportable
contest evidence. The acknowledgement flag is mandatory.

Selection is deterministic and content-independent. Within each of the pinned seven
normalized domains, rows are ranked by a framed digest of the pinned revision, domain,
and `sample_id`; the first 64 form a calibration pool and the next 8 form the evaluation
pool. Evaluation is restored to source order, producing 448 calibration rows and one
shared 56-row evaluation scope. The row grain is `sample_id`. A per-model pre-call
quote is the maximum realized charge observed only in that model's calibration pool,
and every evaluation charge is checked against its fixed quote before fitting begins.

The three diagnostic tier budgets are mechanically selected from the minimum, median,
and maximum of the sorted model quotes and use weights `0.5`, `0.3`, and `0.2`. These
are diagnostic parameters, not official budget tiers, and their cost values are not
emitted. The surface-feature-only bilinear policy (no `bge-m3`) applies nested LODO to
quality-predictor fitting, lambda tuning, learned replay, and the domain-table baseline,
with an explicitly approximate lambda search capped at 32 candidates per tier. Quote
and tier calibration instead uses the disjoint global calibration pool spanning all
seven domains. Therefore this is not an end-to-end domain-shift claim. The learned
policy and all six baselines are replayed on the same 56 rows.

Human and `--json` output expose only aggregate provenance, structure, configuration,
and completion evidence. Prompt/output text, sample IDs, row decisions, and
performance, realized-cost, or oracle-gap results are suppressed. The validator does
not write a converted dataset, predictions, learned artifact, or result file. Before
benchmark orchestration it replaces external sample IDs with deterministic local
surrogates; the CLI also suppresses exception details and tracebacks on failure. Do not
commit redirected output or any RouterBench-derived artifact. Domain imbalance and
heterogeneous upstream evaluators remain material limitations, so this bounded local
diagnostic is not a reproduction of the RouterBench paper.

### bge-m3 (planned, local-only)

The embedding contract pins `BAAI/bge-m3` at revision
`5617a9f61b028005a4858fdac845db406aefb181` (MIT). Weights are not bundled and no
runtime downloader exists. The planned provider will accept only a prepared local path
and must fail closed under `HF_HUB_OFFLINE=1` rather than resolving a Hub model ID.
Full training at up to 1,036 total features remains gated on the prepared-session and
three-platform release checks above. The experimental one-solve C11 candidate is not
evidence that the complete nested experiment has run; embedding dimensions will not be
silently projected away.

SK Telecom challenge data is likewise excluded until its license and redistribution
terms are confirmed in writing.

## Development checks

```bash
make install-dev PYTHON=python
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 pytest
tierroute route "offline smoke" --tier fast
tierroute evaluate
tierroute benchmark --budget-scope per-query
tierroute compare-predictors --budget-scope per-query
tierroute demo
tierroute train --output artifacts/synthetic-bilinear.json --json
tierroute route "artifact smoke" --artifact artifacts/synthetic-bilinear.json --json
tierroute train --output artifacts/synthetic-bilinear.json \
  --policy-output artifacts/synthetic-policy.json --budget-scope per-query --json
tierroute route "policy smoke" --artifact artifacts/synthetic-bilinear.json \
  --policy-artifact artifacts/synthetic-policy.json --json
```

Run `install-dev` only inside the project virtual environment; it removes the
setuptools copy that some Python 3.10 `ensurepip` installations leave behind, then
installs the audited lock with flit_core.

Two locked, no-external-data reproduction lanes are available:

```bash
make reproduce-inference PYTHON=python  # fast: installed routing and evaluation
make reproduce-training PYTHON=python   # complete: fitting, benchmark, comparison, demo
```

Both create an empty temporary Hugging Face cache and force offline mode. The fast lane
skips `tierroute train`, `tierroute benchmark`, `tierroute compare-predictors`,
`tierroute demo`, and all bilinear/lambda-policy fitting. It exercises installed
synthetic prediction, artifact loading, routing, and the six-baseline evaluation;
evaluation fits only the required outer-training domain table, not the learned
predictor. The complete lane additionally runs lint, SPDX, tests, license and install
checks, then its training smoke fits and consumes synthetic predictor/policy artifacts,
executes the nested-LODO benchmark and paired predictor estimation, and runs the
training-backed three-step demo. Thus benchmark, comparison, and showcase fitting run
only in `training-smoke`/`reproduce-training`, not the inference lane. `make reproduce`
remains an alias for the complete lane. These targets install the pinned reviewed
development packages but do not remove every unrelated distribution. Start from a
fresh dedicated virtual environment so unrelated packages cannot contaminate the
reproduction claim.

CI runs linting, tests, a
dependency-free wheel install, both CLI smoke paths, offline-mode checks, and a
dependency-license gate. GPL-family dependencies are not accepted. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution and compliance checklist and
[SBOM.md](SBOM.md) for the dependency inventory. Actual wheel-content approvals and
rejections are recorded in
[docs/dependency-license-audit.md](docs/dependency-license-audit.md).

The primary-source review, prior-art comparison, and exact boundary between implemented,
planned, and gated claims are recorded in
[docs/literature-and-novelty.md](docs/literature-and-novelty.md). Read it before reusing
performance, OOD, or novelty language in a report or presentation.

The claim-gated five-page submission structure, architecture-diagram source, metric
evidence record, and final render checklist are maintained in
[docs/submission-report-outline.md](docs/submission-report-outline.md). Its placeholders
are not contest results and must never be filled with synthetic demo values.

Material development-assistant use, evidence limits, and human review status are
recorded in [docs/ai-assistance-audit.md](docs/ai-assistance-audit.md). The critical
invariant walkthrough packet and owner sign-off table are in
[docs/maintainer-explainability.md](docs/maintainer-explainability.md). CI and AI-agent
reviews are automated evidence; they are not human owner sign-off.

## Open questions

These decisions remain adapter- or configuration-local until official answers arrive:

1. Is each tier budget scoped per query or cumulatively across an ordered stream, and
   what exact call-history fields are visible to the router?
2. Does the official simulator permit sequential calls and selection from prior outputs?
   Cascade routing stays out of scope until confirmed.
3. What license and redistribution terms govern SK Telecom data, and what are the
   official Fast/Balanced/Premium weights, cost units, hidden-data schema, and scoring
   details? No SK Telecom data will be committed before written license confirmation.
4. Are randomized expected-cost mixtures legal? The answer determines whether the
   RouterBench Zero policy is a valid additional comparison, not one of the required
   six baselines.
5. Should a tierroute-trained ridge/bilinear-plus-isotonic predictor artifact be
   declared as an Appendix 2 type-3 self-developed model? Obtain the organizer's written
   interpretation before the final declaration; it is not external-model fine-tuning.

## License

Project-authored code and documentation are licensed under [Apache-2.0](LICENSE).
Source and documentation files carry SPDX identifiers. Third-party datasets and model
assets retain their own terms and are not relicensed by tierroute.
