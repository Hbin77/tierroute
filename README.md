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
metrics, leakage-aware calibrated bilinear training, and an external-data-free demo
are implemented. The CLI selects a model but does **not** call an LLM or return a
model completion.

## Quickstart

Python 3.10 or newer is required. From a fresh checkout:

```bash
cd tierroute
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run one routing decision, all six replay baselines, and the combined demo:

```bash
tierroute route "Prove that sqrt(2) is irrational." --tier fast
tierroute evaluate
tierroute demo
```

The equivalent module entry point is `python -m tierroute`. Machine-readable output
is available for `route` and `evaluate` with `--json`; a compatible versioned replay
JSON can be supplied to evaluation:

```bash
tierroute route "Debug this Python function" --tier balanced --json
tierroute evaluate --data src/tierroute/data/synthetic.json --json
HF_HUB_OFFLINE=1 tierroute demo
```

The bundled prompts, costs, outputs, predicted qualities, and scorecard are
project-authored **synthetic smoke-test values**. They verify wiring and are not a
benchmark result, an empirical model comparison, or a competition score.

### Offline predictor training

Training is an explicit preparation step with a pinned optional NumPy dependency.
The resulting surface-feature artifact is strict canonical JSON and inference returns
to the dependency-free, network-free runtime path:

```bash
python -m pip install -e '.[training]'
tierroute train --output artifacts/synthetic-bilinear.json --json
tierroute route "Prove that sqrt(2) is irrational." \
  --tier balanced \
  --artifact artifacts/synthetic-bilinear.json \
  --json
```

Use `--data path/to/replay.json` on both commands for another version-1 replay dataset.
The bundled-data command proves fit/save/load/route wiring only. It does not produce a
reportable benchmark result. The CLI fits a deployable artifact on all supplied rows,
using inner LODO predictions for per-model isotonic calibration. Reportable outer-LODO
experiments must instead call `fit_calibrated_bilinear_for_fold` on each training fold
and score only that fold's held-out domain. Artifact routing still uses the CLI's
illustrative, hard-coded tier lambdas; learned nested-LODO lambda tuning is a separate
next milestone, so this path is not yet a trained end-to-end budget policy.

## What is implemented

- Exact `Decimal` cost accounting and typed `RouterState`/`RouterAction` contracts.
- Swappable per-query and cumulative budget ledgers; the demo uses illustrative
  per-query limits until the official budget scope is confirmed.
- One-shot lambda routing and six reproducible baselines.
- Full-information offline replay: labels stay hidden until a selected logged outcome
  is replayed, so the policy cannot read ground truth through `RouterState`.
- A fitted surface-feature schema (log-scaled counts, code/math signals, and
  prompt-derived domain tags), deterministic per-model ridge fitting, inner-LODO
  out-of-fold predictions, and separate isotonic calibration per model.
- Canonical, strictly validated JSON predictor artifacts; pickle is never accepted for
  predictor loading. Batch prediction vectorizes or embeds each prompt batch once.
- Tier-weighted quality, oracle-gap recovery, and deterministic leave-one-domain-out
  (LODO) folds. No random-split helper is provided.
- Strict JSON loading plus an opt-in, pinned RouterBench boundary adapter.

Without `--artifact`, the no-download CLI uses a transparent synthetic demo predictor.
A local `bge-m3` embedding backend and GBM-versus-bilinear experiment remain planned;
they are not represented as finished features.

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
oracle alone receives a private example key through a nominal evaluation-only boundary.

```text
JSON / RouterBench boundary ──> typed replay examples ──> OfflineSimulator
                                      │                       │
prompt ─> fitted feature encoder ─> calibrated predictor ─> policy <─ budget ledger
                                                     │
                                            CallModel / SelectOutput

core/        stable state, action, model, and validation contracts
features/    offline surface features, fitted schema, local embedding contract
predictors/  bilinear trainer, per-model calibration, strict JSON artifacts
policies/    one-shot lambda policy and required baselines
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

Oracle-gap recovery measures how much of the weighted quality interval from
always-cheapest to the budget-feasible oracle was recovered:

```text
sum_t w_t * (Q_router,t - Q_cheapest,t)
-------------------------------------------------
sum_t w_t * (Q_oracle,t - Q_cheapest,t)
```

It is undefined when the oracle and cheapest scores are equal, and negative values are
preserved. The six baselines are:

| Baseline | Decision rule |
| --- | --- |
| `always-cheapest` | Lowest cost, then model ID for ties |
| `always-premium` | Explicitly designated premium model; may be infeasible in a lower tier |
| `random` | Seeded, order-independent choice among affordable models |
| `length-heuristic` | Strong model for long/code/math prompts when affordable |
| `oracle` | Privileged per-query, budget-feasible quality upper bound |
| `domain-best-table` | Per-tier mean-quality table fitted on training domains, with cheapest fallback |

`tierroute evaluate` fits the domain table on the tiny bundled sample only as an
end-to-end smoke check, and says so in its output. Reportable experiments must fit on
the training side of every LODO fold and evaluate only on the held-out domain. A dataset
domain reaches `RouterState` only when its adapter explicitly marks that label as
observable before routing; split-only labels remain private.

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
realized charge to a policy.

The authenticated wire table is materialized in memory. Allow at least 512 MiB of
headroom; the default prefix validation measured about 290 MB maximum RSS on the
reference Python 3.12 environment. `--limit` bounds typed replay retention, while
`--limit 0` intentionally replays all post-calibration rows.

### bge-m3 (planned, local-only)

The embedding contract pins `BAAI/bge-m3` at revision
`5617a9f61b028005a4858fdac845db406aefb181` (MIT). Weights are not bundled and no
runtime downloader exists. The planned provider will accept only a prepared local path
and must fail closed under `HF_HUB_OFFLINE=1` rather than resolving a Hub model ID.

SK Telecom challenge data is likewise excluded until its license and redistribution
terms are confirmed in writing.

## Development checks

```bash
python -m pip install -e '.[dev,training]'
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 pytest
tierroute route "offline smoke" --tier fast
tierroute evaluate
tierroute demo
tierroute train --output artifacts/synthetic-bilinear.json --json
tierroute route "artifact smoke" --artifact artifacts/synthetic-bilinear.json --json
```

`make reproduce` installs the exact development lock and runs the complete bundled-data
pipeline, including training and artifact-backed routing. CI runs linting, tests, a
dependency-free core install, both CLI smoke paths, offline-mode checks, and a
dependency-license gate. GPL-family dependencies are not accepted. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution and compliance checklist and
[SBOM.md](SBOM.md) for the dependency inventory.

## Open questions

These decisions remain adapter- or configuration-local until official answers arrive:

1. Is each tier budget scoped per query or cumulatively across an ordered stream, and
   what exact call-history fields are visible to the router?
2. Does the official simulator permit sequential calls and selection from prior outputs?
   Cascade routing stays out of scope until confirmed.
3. What license and redistribution terms govern SK Telecom data, and what are the
   official Fast/Balanced/Premium weights? No SK Telecom data will be committed before
   written license confirmation.
4. Does the submission's GPL-family prohibition also exclude compatible native runtime
   components disclosed inside a preparation-only NumPy wheel? The GCC Runtime Library
   Exception and dynamically linked LGPL component are recorded in the SBOM; NumPy and
   the RouterBench pickle reader will be replaced if the rule is literal at that level.

## License

Project-authored code and documentation are licensed under [Apache-2.0](LICENSE).
Source and documentation files carry SPDX identifiers. Third-party datasets and model
assets retain their own terms and are not relicensed by tierroute.
