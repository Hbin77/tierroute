<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to tierroute

Thank you for helping make budget-aware LLM routing reproducible and safe. Small,
reviewable changes are preferred. Please discuss substantial policy, schema, or
dependency changes in an issue before implementation.

## Set up a development checkout

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 pytest
tierroute route "offline smoke" --tier fast
tierroute evaluate
tierroute demo
tierroute train --output artifacts/synthetic-bilinear.json --json
tierroute route "artifact smoke" --artifact artifacts/synthetic-bilinear.json --json
tierroute train --output artifacts/synthetic-bilinear.json \
  --policy-output artifacts/synthetic-policy.json --budget-scope per-query --json
tierroute route "policy smoke" --artifact artifacts/synthetic-bilinear.json \
  --policy-artifact artifacts/synthetic-policy.json --json
```

All tests and demos must pass without network access. A preparation script may access
the network only when the user runs it explicitly; routing, feature extraction,
prediction, evaluation, and tests must never download data or model weights.

## Choose the right boundary

- Keep stable state/action contracts and unit-free exact costs in `core/`.
- Put uncertain challenge schemas, budget scope, and external formats in `adapters/`.
- Put deterministic pre-call signals in `features/` and quality estimates in
  `predictors/`.
- Put model-selection decisions in `policies/`; do not let a policy read replay labels
  or uncalled outputs.
- Put accounting, replay, metrics, and domain-shift validation in `eval/`.
- Keep one-shot routing as the default. Do not add cascade behavior until sequential
  calls and accounting are confirmed by the challenge organizer.

Use LODO for reportable validation. A domain-derived table, calibrator, threshold, or
predictor must be fitted only on the training side of a fold. Random splits are not an
acceptable substitute for domain-shift evaluation.

For calibrated predictors, fit the isotonic layer from inner-LODO out-of-fold
predictions inside the outer training fold. Tune tier lambdas only from cross-fitted
predictions on that same outer training side, then refit the deployable predictor on
the full outer training side. Never fit feature scaling, a tag vocabulary,
calibration, or a policy threshold on the outer held-out domain. Predictor and policy
artifacts use strict JSON only; do not introduce pickle, `eval`, or an automatic
compatibility fallback.

Runtime and tuning must share `route_from_predictions`, including its exact utility and
tie-break order. An exhaustive lambda claim requires the full boundary/interval/tail
candidate set. If a cap is used, stream roots into the deterministic bounded bottom-hash
sample plus extrema, then derive and rank-space the retained-root candidates. If the cap
actually truncates roots or derived values, label the result `exhaustive: false`, record
the strategy and observed breakpoint-occurrence count, and leave the intentionally
unmaterialized complete candidate count unknown. If everything fits, retain the exact
count and `exhaustive: true`.
Uncapped searches must keep the candidate/utility preflight. Do not raise its limits or
use `allow_large_exhaustive=True` without recording the measured resource rationale;
the CLI's bounded default is 257 candidates.
Budget normalization belongs to the injected ledger adapter; do not infer official
per-query or cumulative semantics in a policy.

## Code and test expectations

1. Add `SPDX-License-Identifier: Apache-2.0` in the appropriate comment syntax to every
   project-authored source, test, script, configuration, and documentation file where
   the format permits comments.
2. Preserve exact costs with `Decimal`; never introduce float comparisons at budget
   boundaries. Aggregate, subtract, scale, or average costs through the project-owned
   context-independent helpers. Preserve fitted lambdas as `Fraction` values and
   serialize their exact numerator/denominator representation.
3. Save JSON artifacts through the project-owned atomic bundle writer. Never use a
   predictable temporary pathname or sequentially publish a bound predictor/policy
   pair without rollback and input-alias checks.
4. Add focused tests for behavior, failure paths, determinism, and offline operation.
5. Keep public interfaces typed and explain non-obvious routing or metric choices in a
   short docstring or design comment.
6. Run `make verify`, including both core and training/artifact CLI smoke paths, before
   opening a pull request.

## Licensing, data, and dependencies

- Do not commit downloaded datasets, model weights, credentials, generated private
  outputs, or SK Telecom data without written redistribution permission.
- Attribute every third-party code, data, model, font, and other asset with its source,
  revision/version, license, and purpose.
- GPL-family dependencies are not accepted. Check direct and transitive licenses before
  proposing a package.
- Update `SBOM.md` in the same pull request whenever a dependency or external asset is
  added, removed, or pinned differently.
- Treat pickle-format bytes as hostile input. Never relax RouterBench's pinned size/SHA,
  opcode/global, or structural checks; never replace its non-dispatching decoder with
  `pickle.load`, `pickle.Unpickler`, `pandas.read_pickle`, or another callable-dispatching
  deserializer.
- Keep synthetic fixtures clearly labeled; their values must never be reported as
  benchmark evidence.

If licensing is unclear, leave the artifact out and ask for review. A source URL alone
does not establish redistribution rights.

## Commits and pull requests

Use focused [Conventional Commits](https://www.conventionalcommits.org/) messages, for
example:

```text
feat(policies): add calibrated fallback policy
test(eval): cover cumulative budget exhaustion
docs(readme): clarify RouterBench opt-in
```

Before requesting review:

- Link the issue or explain the user-visible problem.
- Describe the chosen boundary and any assumptions.
- Include tests and documentation needed to reproduce the change.
- Complete the pull request checklist truthfully.
- Avoid unrelated formatting or generated-file churn.

By submitting a contribution, you agree that it is licensed under Apache-2.0 as
described in [LICENSE](LICENSE). Please follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
in all project spaces.
