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
- True nested-LODO policy evaluation with one original-order outer-OOF replay for
  cumulative accounting.
- Canonical policy artifacts bound to predictor, data, replay-order, tier-spec, ledger,
  and candidate-search provenance, with the OOF prediction digest recorded as
  reproducibility audit metadata.
- `train --policy-output --budget-scope` and `route --policy-artifact` offline CLI
  reproduction paths; cumulative routes require explicit remaining budget state.
- A dependency-free wheel CI job plus a fully offline predictor/policy
  fit/save/load/route smoke test.
- A static ridge-solver boundary that resolves once per calibrated fit, preflights
  before dense embedding allocation, and carries the same reviewed implementation
  through every inner-LODO fit and final refit without changing version-1 artifact
  bytes.
- A primary-source literature review and novelty matrix covering RouteLLM,
  RouterBench, FrugalGPT, unified routing/cascading, and the three
  organizer-recommended inference-time generation materials, plus a targeted
  2025–2026 routing landscape check, with implemented, planned, and officially gated
  claims separated.

### Changed

- Clarify that the bundled oracle and six-baseline CLI are illustrative per-query
  infrastructure: cumulative oracle-gap reporting needs a sequence-level oracle, and
  reportable domain-table comparisons must fit only on each outer training fold.
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

- Train and compare a calibrated GBM against the bilinear predictor on licensed data.
- Add a local-only inference backend for the pinned MIT-licensed bge-m3 revision.
- Connect official SK Telecom data and scoring only after its schema, weights, and
  redistribution terms are confirmed.

### Security

- Authenticate RouterBench bytes before structural decoding; referenced pickle globals
  remain inert and no callable named by the payload is imported or invoked.
- Predictor artifacts accept strict JSON only; duplicate keys, unknown fields,
  non-finite numbers, invalid dimensions, and pickle bytes fail closed.
- Policy artifacts additionally reject malformed/noncanonical rational values,
  predictor/data/order mismatches, invalid Unicode metadata, and unsafe binary input.
  Bounded reads now reject artifacts above 8 MiB, 404,096 digits per exact integer, or
  100,000 retained candidates per tier before expensive big-integer parsing. The
  integer limit covers candidates derivable from the public core cost contract;
  ledger-adapter metadata is capped at 4 KiB.
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
