<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

Notable project changes are recorded here. Releases follow semantic versioning while
the public API is pre-1.0.

## [Unreleased]

### Changed

- Replace the pandas/NumPy RouterBench reader with a dependency-free,
  non-dispatching standard-library decoder for the exact pinned artifact.
- Remove the `routerbench` optional extra, `requirements-routerbench.lock`,
  `RouterBenchDependencyError`, and `load_routerbench_dataframe`; callers now use
  `load_routerbench_table`, which returns the immutable project-owned table.
- Pin a canonical decoded-table digest and exact row, column, model, and domain counts
  as a regression oracle independent of the artifact checksum.

### Security

- Authenticate RouterBench bytes before structural decoding; referenced pickle globals
  remain inert and no callable named by the payload is imported or invoked.

### Planned

- Train and compare calibrated GBM and bilinear quality predictors on licensed data.
- Add a local-only inference backend for the pinned MIT-licensed bge-m3 revision.
- Connect official SK Telecom data and scoring only after its schema, weights, and
  redistribution terms are confirmed.

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
