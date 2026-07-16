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
make install-dev PYTHON=python
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

`install-dev` removes any setuptools copy left by Python 3.10's `ensurepip` before
installing the exact lock. Setuptools is not part of tierroute's reviewed build graph;
use a dedicated virtual environment for this target.

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
compatibility fallback. Preserve predictor limits for bytes, JSON-number width, model/
domain/tag counts, feature dimension, numeric scalars, metadata, and calibration points
on construction, lexical preflight, parsing, loading, serialization, and saving. Keep
direct inputs single-snapshot, exact-primitive, and finite-binary64 normalized; never
validate one view of a caller-owned container and store another. Preserve policy
artifact byte, integer-digit, and per-tier candidate limits on both construction and
loading. Raise any limit only with measured parser/resource evidence, planned-shape
headroom, and matching adversarial tests; do not change valid version-1 predictor bytes
because policy artifacts bind their SHA-256.

Replay JSON has a separate adapter-owned resource contract documented in
`docs/replay-json.md`. Preserve its descriptor-stable regular-file read, pre-parser
lexical limits, duplicate/nonstandard-number rejection, exact version-1 fields and
types, collection/text bounds, outer/nested LODO amplification checks, and the reference
trainer's compound outcome-scan bound. Do not add an environment or CLI bypass. If
official data exceeds the contract, record authenticated shape measurements and use a
reviewed constant change or a separate schema adapter with exact-limit and
limit-plus-one tests.

Runtime and tuning must share `route_from_predictions`, including its exact utility and
tie-break order. An exhaustive lambda claim requires the full boundary/interval/tail
candidate set. If a cap is used, stream roots into the deterministic bounded bottom-hash
sample plus extrema, then derive and rank-space the retained-root candidates. If the cap
actually truncates roots or derived values, label the result `exhaustive: false`, record
the strategy and observed breakpoint-occurrence count, and leave the intentionally
unmaterialized complete candidate count unknown. If everything fits, retain the exact
count and `exhaustive: true`.
Changing the bottom-hash byte identity requires a new strategy version and a golden
vector; existing artifact strategy values must remain loadable. Bounded and uncapped
searches must keep the candidate, utility, and exact-rational byte-width preflight
before fitting or root creation. Pair-scan work and estimated serialized policy evidence
are guarded separately and must stay in that same pre-fit path. Do not raise its limits or use
`allow_large_exhaustive=True` without
recording the measured resource rationale. The CLI's bounded default is 257 candidates,
but the cap is dataset-dependent; the pinned full RouterBench shape needs a cap of 64
to stay below the default utility-work limit.
Budget normalization belongs to the injected ledger adapter; do not infer official
per-query or cumulative semantics in a policy.

## Code and test expectations

1. Add `SPDX-License-Identifier: Apache-2.0` in the appropriate comment syntax to every
   project-authored source, test, script, configuration, and documentation file where
   the format permits comments.
2. Preserve exact costs with `Decimal`; never introduce float comparisons at budget
   boundaries. Aggregate, subtract, scale, or average costs through the project-owned
   context-independent helpers. The supported decimal-position/coefficient bound is a
   deliberate resource contract; changes require boundary, equivalent-representation,
   underflow, and expansion tests plus documentation. Preserve fitted lambdas as
   `Fraction` values and serialize their exact numerator/denominator representation.
3. Save JSON artifacts through the project-owned atomic bundle writer. Never use a
   predictable temporary pathname or sequentially publish a bound predictor/policy
   pair without rollback and input-alias checks. Do not claim bundle-wide atomicity for
   concurrent writers or power loss across unrelated pathnames.
4. Preserve executed-call evidence in replay evaluation. A call that consumed a logged
   outcome remains recorded even if the ledger returns false; a call rejected before
   replay is not executed. Require `QueryResult.cost` to equal the exact sum of its
   realized call charges, and reconcile tier call evidence with `BudgetReport.spent`
   and its over-budget count. Treat balance snapshots and the ledger result as adapter
   evidence; do not infer unconfirmed budget semantics in shared schemas.
5. Preserve the versioned evaluation-scope contract. Do not reuse artifact hashes as a
   substitute, omit a router-visible replay field, accept mutable/custom metadata via
   `repr`, or compare reports before checking the complete scope identity (algorithm,
   digest, and call cap). A scope algorithm byte change requires a new version and
   golden vectors for the old hashes.
6. Add focused tests for behavior, failure paths, determinism, and offline operation.
7. Keep public interfaces typed and explain non-obvious routing or metric choices in a
   short docstring or design comment.
8. Run `make verify`, including both core and training/artifact CLI smoke paths, before
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

## AI-assisted contributions

The contributor remains responsible for every submitted line, claim, license, and
behavior, whether or not an AI coding assistant proposed it. When an assistant makes a
material contribution, complete the pull request's disclosure with:

- the tool or service name;
- the exact model/version/snapshot when it is actually available, or the truthful value
  `not exposed/not retained`;
- the assisted activities and affected paths or algorithms;
- the human validation actually performed in that pull request; and
- the matching entry in [the assistance audit](docs/ai-assistance-audit.md), when the
  change is material to a contest claim or critical invariant.

Do not invent an AI-authored-line percentage, reconstruct a model identity from memory,
or commit private prompts and credentials merely to make a disclosure appear precise.
An AI-agent/subagent review and a successful CI run are automated evidence, not an
independent human review. Name a human reviewer only when a durable record from that
person exists.

Changes to a boundary in
[the maintainer explainability packet](docs/maintainer-explainability.md) must update
its source/test map when needed and return the affected owner sign-off row to
**Pending**. Only the human entrant may complete that row after tracing the code and
performing the required failure/mutation drill. A contribution with no material AI
assistance should state `None` in the pull request instead of omitting the section.

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
- Disclose material AI assistance and distinguish automated evidence from a named human
  walkthrough; link the audit row or record the walkthrough as pending.
- Complete the pull request checklist truthfully.
- Avoid unrelated formatting or generated-file churn.

By submitting a contribution, you agree that it is licensed under Apache-2.0 as
described in [LICENSE](LICENSE). Please follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
in all project spaces.
