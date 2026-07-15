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
- A dependency-free wheel CI job plus a fully offline fit/save/load/route smoke test.

### Changed

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
