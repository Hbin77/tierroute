<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

Notable project changes are recorded here. Releases follow semantic versioning while
the public API is pre-1.0.

## [Unreleased]

### Added

- Fitted prompt-feature schemas and deterministic per-model ridge training.
- Inner-LODO out-of-fold prediction with a separate isotonic calibrator per model.
- Canonical, fail-closed JSON predictor artifacts and `tierroute train`/
  `tierroute route --artifact` CLI paths.
- Exact rational one-shot utility shared by runtime routing and offline evaluation,
  including deterministic behavior for non-float-representable `Decimal` costs.
- Cross-fitted tier-lambda tuning that directly replays the weighted budget metric,
  with full exact breakpoint search or an explicitly approximate, bounded-memory
  bottom-hash search that records its strategy and observed breakpoint occurrences.
- True nested-LODO policy evaluation with one original-order outer-OOF replay so fold
  orchestration cannot reset sequence state; reportable cumulative claims remain gated.
- A `tierroute benchmark --budget-scope per-query [--data ...] [--json]` runner that
  compares the true nested-LODO learned router with all six canonical baselines under
  one identical per-query evaluation scope, including compact versioned outer-fold
  membership digests, tier weights, resolved baseline parameters/rule identities, and
  requested lambda-search resource controls. A versioned digest binds the baseline
  parameters to their ordered replay decisions. Bundled synthetic rows are labeled as
  wiring-only; licenses and claims for user-supplied replay data remain the caller's
  responsibility. The fitting command runs in `training-smoke`/`reproduce-training`,
  not the inference lane.
- Canonical policy artifacts bound to predictor, data, replay-order, tier-spec, ledger,
  and candidate-search provenance, with the OOF prediction digest recorded as
  reproducibility audit metadata.
- `train --policy-output --budget-scope` and `route --policy-artifact` offline CLI
  reproduction paths; cumulative routes require explicit remaining budget state.
- A dependency-free wheel CI job plus a fully offline predictor/policy
  fit/save/load/route smoke test.
- Separate locked `reproduce-inference` and `reproduce-training` paths for fast
  installed inference review and the complete bundled-data pipeline; `reproduce`
  remains the complete-path alias, and dual-Python CI executes both public targets.
- A static ridge-solver boundary that resolves once per calibrated fit, preflights
  before dense embedding allocation, and carries the same reviewed implementation
  through every inner-LODO fit and final refit without changing version-1 artifact
  bytes.
- A primary-source literature review and novelty matrix covering RouteLLM,
  RouterBench, FrugalGPT, unified routing/cascading, and the three
  organizer-recommended inference-time generation materials, plus a targeted
  2025–2026 routing landscape check, with implemented, planned, and officially gated
  claims separated.
- A leakage-free per-query outer-LODO suite for all six required baselines. It records
  fold train/test evidence, replays one shared original-order population, verifies the
  actual ledger's reset/charge/report behavior, and keeps cumulative oracle claims gated.
- Structured evidence for every executed logged replay call, including realized
  overspends, plus exact per-tier and cross-tier quote-versus-realized diagnostics.
  JSON routing labels its pre-call quote and leaves realized cost unknown; JSON
  evaluation reconciles realized call totals and over-budget counts with each ledger.
- A required `tierroute-evaluation-scope-v1` report identity over ordered tier specs,
  call cap, complete replay content, candidate/outcome order, outputs, labels, and
  canonical policy-visible metadata. Evaluation CLI output exposes the algorithm,
  digest, and call cap.
- A truthful [development-assistance ledger](docs/ai-assistance-audit.md), Korean
  submission-disclosure draft, and contest-critical
  [maintainer explainability packet](docs/maintainer-explainability.md), with all human
  walkthroughs explicitly pending until a named entrant performs and records them.
- A documented version-1 replay JSON resource contract sized from the planned
  RouterBench mapping, including outer and nested LODO amplification limits.

### Changed

- Canonicalize every CLI JSON cost string, including legacy `cost` and `total_cost`
  keys, so equivalent decimal encodings have one representation. Numeric meaning and
  keys remain stable, but textual trailing zeros such as `"0.20"` become `"0.2"`.
- Make `QueryResult` require executed-call evidence whose realized charges exactly sum
  to `cost`, plus a selected call for every feasible result. Make `BaselineResult`
  require quote evidence derived from its own replay report. These are intentional
  pre-1.0 constructor-contract changes.
- Require `EvaluationReport` callers to supply an
  `EvaluationScopeIdentity(algorithm, sha256, max_calls_per_query)`. Fold evidence now
  requires that same identity, and `LodoSixBaselineEvaluation` requires the stable
  candidate-model IDs. Cross-report metrics reject a mismatched complete identity
  before numeric work; score summaries are immutable, baseline rows recompute their
  scores, and the six-report suite validates canonical row order, fold partitions,
  scope-bound fold/table evidence, candidate membership, per-query oracle dominance,
  and exact oracle-gap derivation. These are intentional pre-1.0 constructor-contract
  changes.
- Canonicalize candidate, realized, and tier-budget costs in the immutable replay
  snapshot used by both routing and scope hashing; normalize tier weights to plain
  binary64 there as well. Repeated baseline and lambda-policy runs reuse one prepared
  snapshot and digest without weakening the public fail-closed input boundary.
- Make the bundled six-baseline CLI use outer-training-only domain tables and a shared
  original-order outer-LODO replay. Domain-table fitting now reads only observable
  pre-call metadata tags, never validation-only split domains; unseen tags use the
  cheapest fallback. The synthetic inputs remain illustrative, and cumulative
  oracle-gap reporting still needs a sequence-level oracle.
- Replace setuptools with the dependency-free `flit_core` build backend after a
  wheel-content audit found vendored LGPL code that top-level metadata did not report.
  The license gate now scans bundled license documents and nested vendored metadata in
  addition to `pip-licenses` output.
- Replace the optional NumPy training path with a project-owned deterministic
  centered-ridge Cholesky reference solver, and remove the `training` extra and lock.
- Record the exact ridge solver ID in strict predictor artifacts and CLI training
  output. Full bge-m3-scale fitting remains gated on a reviewed accelerated backend
  with numerical parity tests; a conservative work guard rejects accidental oversized
  reference-solver jobs before Gram construction.
- Replace the pandas/NumPy RouterBench reader with a dependency-free,
  non-dispatching standard-library decoder for the exact pinned artifact.
- Remove the `routerbench` optional extra, `requirements-routerbench.lock`,
  `RouterBenchDependencyError`, and `load_routerbench_dataframe`; callers now use
  `load_routerbench_table`, which returns the immutable project-owned table.
- Pin a canonical decoded-table digest and exact row, column, model, and domain counts
  as a regression oracle independent of the artifact checksum.
- Make cost addition, subtraction, integer scaling, replay totals, and quote estimation
  independent of the caller's mutable `Decimal` context. Canonicalize equivalent tuple
  representations, enforce an explicit 100,000-position/digit resource range, and
  reject out-of-range exact or repeating-quotient results before underflow.
- Refuse unacknowledged bounded or exhaustive lambda searches above conservative
  retained-candidate, utility-work, or 256 MiB exact-rational state bounds before
  predictor fitting; the CLI requires a separate explicit exhaustive-search override.
- Version bounded root sampling as `bounded-bottom-hash-v2`, using signed,
  self-delimiting integer identities for arbitrarily large exact roots while retaining
  version-1 artifact loading.

### Planned

- Add a sequence-level oracle before reporting oracle-gap recovery under cumulative
  accounting, after the organizer confirms the official budget semantics.
- Add a GPL-family-free accelerated ridge backend with numerical parity and resource
  evidence before the full-dimensional bge-m3 experiment.
- Train and compare a calibrated GBM against the bilinear predictor on licensed data.
- Add a local-only inference backend for the pinned MIT-licensed bge-m3 revision.
- Connect official SK Telecom data and scoring only after its schema, weights, and
  redistribution terms are confirmed.

### Security

- Replay JSON now uses one descriptor-stable, bounded regular-file reader and a lexical
  preflight before strict JSON parsing. Duplicate/unknown/missing fields, nonstandard or
  overflowing numbers, implicit primitive coercions, invalid Unicode, oversized text,
  excessive collections, outer/nested LODO work, and reference-training target scans
  fail closed before command-specific route/evaluate/train work. This intentionally
  rejects formerly ignored or coerced pre-1.0 inputs while preserving the bundled
  synthetic bytes and optional `quoted_cost` fallback.
- Authenticate RouterBench bytes before structural decoding; referenced pickle globals
  remain inert and no callable named by the payload is imported or invoked.
- Predictor artifacts accept strict JSON only; duplicate keys, unknown fields,
  non-finite numbers, invalid dimensions, and pickle bytes fail closed. Bounded binary
  reads and a pre-decode lexical pass now enforce a 32 MiB document limit plus bounded
  nesting, strings, structure, and numeric tokens. Single-snapshot direct input,
  finite-binary64 normalization, and planned-shape model, feature, scalar, metadata, and
  calibration caps apply across construction, parsing, serialization, save validation,
  and policy hashing. This is an intentional pre-1.0 rejection of oversized version-1
  inputs; valid canonical bytes and the pinned platform-local predictor SHA-256 values
  remain unchanged.
- Policy artifacts additionally reject malformed/noncanonical rational values,
  predictor/data/order mismatches, invalid Unicode metadata, and unsafe binary input.
  Bounded reads now reject artifacts above 8 MiB, 404,096 digits per exact integer, or
  100,000 retained candidates per tier before expensive big-integer parsing. The
  integer limit covers candidates derivable from the public core cost contract;
  ledger-adapter metadata is capped at 4 KiB.
- Evaluation metadata is copied into a sorted, deeply immutable value tree before
  routing. Cycles, custom containers, non-string keys, nonfinite numbers, excessive
  depth/node/encoded-payload/integer/Decimal ranges, invalid Unicode, and aggregate
  replay payloads over the documented limit fail before replay; hashing never invokes
  `repr`, pickle, or user-defined serialization hooks.
- Dependency license enforcement now scans bounded regular-file evidence, nested
  vendored metadata, common third-party notice layouts, and GPL-family filenames. Its
  only document exceptions are exact reviewed PSF-family license hashes used by the
  development lock; modified evidence fails closed.
- Lambda-search preflight now counts all breakpoint pair scans in linear catalogue
  time and rejects more than 10,000,000 scans. It also estimates repeated candidate
  evidence against the hard 8 MiB policy-artifact limit before predictor fitting.
- Exact scaling now combines 2/5 factors across the cost coefficient and integer
  multiplier before range validation; a separate `1e200000` raw-factor bound follows
  directly from the supported nonzero input and output magnitude range.
- Stage artifacts under exclusive random names, reject source/destination aliases, and
  publish predictor/policy pairs policy-last with validation, backup, and rollback.
  Rollback now continues through asynchronous inspection/cleanup failures, preserves
  unverifiable recovery backups, and distinguishes incomplete cleanup after a verified
  commit; concurrent writers and bundle-wide power-loss atomicity remain unsupported.

## [0.1.0] - 2026-07-15

### Added

- Apache-2.0 Python package with a `src/` layout and `tierroute` CLI.
- Typed router state/action contracts with exact decimal costs.
- Specification-independent replay simulator and swappable per-query/cumulative budget
  ledgers.
- Weighted tier-quality and oracle-gap-recovery metrics plus LODO fold generation.
- One-shot lambda policy and six required baselines: cheapest, premium, random, length
  heuristic, oracle, and domain-best table.
- Offline surface features, bilinear predictor form, and isotonic calibration.
- Project-authored synthetic data for `route`, `evaluate`, and `demo` quickstarts.
- Opt-in RouterBench downloader and boundary adapter pinned by revision, size, and
  SHA-256; RouterBench data remains unbundled with license `NOASSERTION`.
- English and Korean documentation plus community contribution templates.

### Security

- Runtime network access is prohibited; model and data downloads are separate explicit
  preparation paths.
- RouterBench bytes are authenticated before pickle deserialization. Pickle remains
  executable input and is documented as requiring an isolated, trusted workflow.
