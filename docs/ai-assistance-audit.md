<!-- SPDX-License-Identifier: Apache-2.0 -->

# AI assistance and review audit

## Status and purpose

This is a truthful development-process record, not a claim that automated checks prove
human understanding. The 2026 Open Source Developer Contest rules, Article 9(5), permit
commercial AI services for coding and debugging assistance but warn that insufficient
understanding of AI-written code can reduce the evaluation score. The project plan also
requires the assistance scope to be disclosed in the submission's AI-model appendix.

The retrospective ledger below covers the repository through merge commit `614a018`
(PR #16). Its initial owner walkthrough status is deliberately **pending**. Only the
human entrant may change a row to complete, in a later commit that records the entrant's
explicit attestation after performing the review protocol in
[maintainer-explainability.md](maintainer-explainability.md).
Later rows record material assisted work outside that retrospective snapshot.

## Assistance boundary

| Field | Recorded fact |
|---|---|
| Service | OpenAI Codex coding agent; the current development environment identifies a GPT-5 basis |
| Exact model revision | The exact historical model ID/checkpoint was not exposed or retained; no reproducible revision is claimed |
| Period covered | Repository work from 2026-07-15 through 2026-07-17, including issue #9's prepared graph/store/execution/policy references, authenticated file-backed native prepared-session slice, native per-query policy benchmark bridge, Issue #55's canonical library-level GBM artifact v1, and Issue #58's current-main maintainer walkthrough re-audit |
| Material assistance | Requirement decomposition, implementation and refactoring proposals, test design, adversarial review, documentation drafting, local verification commands, and issue/branch/PR workflow |
| Human-supplied decisions | Contest scope, architecture, package/license choices, one-shot default, LODO requirement, baseline set, offline/network prohibition, dependency policy, and approval gates |
| Runtime role | None. Codex is not imported, packaged, called, or required by tierroute at build, training, evaluation, or inference time |
| Data/model role | Codex is not a tierroute feature, predictor, runtime component, or benchmark candidate. No assistant model weights or SK Telecom private data are represented in this ledger |
| External actions | Repository-scoped GitHub issues, branches, PRs, and merges were performed as part of the requested development workflow. Registration, contest forms, and email sending remain human-only approval gates |
| Software inventory | Development assistance is disclosed here, not in `SBOM.md`, because no assistant library, model, weight, or service client is shipped. A future runtime/build dependency must still enter the SBOM normally |

The service-side checkpoint, training corpus, and a meaningful percentage of
"AI-authored lines" are unknown. This project does not invent those values. Prompt
transcripts are not treated as the canonical contribution record because their
availability and product retention are not controlled by this repository. Reviewable
Git commits, issue/PR rationale, tests, and this ledger are the durable evidence.

AI output was treated as a proposed change, not as proof of correctness, originality,
license compatibility, or security. The repository contributor remains responsible for
the Apache-2.0 contribution, third-party attribution, and every merged behavior. CI and
AI-agent review provide useful automated evidence, but neither may be relabeled as
human owner sign-off.

As of the `614a018` snapshot, the ten pull requests merged by that snapshot had zero
submitted GitHub reviews, zero issue comments, and zero review comments. Their
successful checks are automated evidence only. References in older development records
to an "independent review" or "independent audit" are classified in this ledger as
AI-agent or automated evidence unless a named human review record is linked explicitly.

Ledger evidence uses four labels:

- **REPO**: directly verifiable from committed source, tests, Git history, or GitHub
  pull-request/check metadata.
- **ASSISTANCE_RECORD**: material scope recorded by the active AI-assisted development
  workflow; it is not a human authorship, understanding, or approval claim.
- **NOT_RECORDED**: the repository or service did not retain enough information to make
  a narrower claim.
- **PENDING**: a future human action is required and must not be inferred from CI,
  commits, authorship, or AI review.

## Retrospective change ledger

"Pending" means that the code may be tested and merged while the entrant's explicit
explain-without-assistance walkthrough has not yet been recorded. It does not mean the
change is known to be defective.

Merge, source, test, and check facts below are **REPO** evidence. Assisted-scope entries
are **ASSISTANCE_RECORD**; the exact historical model revision is **NOT_RECORDED**;
every owner walkthrough is **PENDING**.

| Change set | Merge/evidence | Material assisted scope | Durable technical evidence | Human owner walkthrough |
|---|---|---|---|---|
| Bootstrap commits `f501a21` through `db74e58` | Direct commits; **REPO** | Package/core/eval/policy/adapter scaffold, synthetic data, CLI, CI, and initial community files | Conventional commits; current full-suite and offline CI cover the resulting tree | **PENDING** |
| [PR #6](https://github.com/Hbin77/tierroute/pull/6) | `3a4d769`; **REPO** | Community-health and contribution documentation | Reviewable documentation diff; merged CI | **PENDING** |
| [PR #8](https://github.com/Hbin77/tierroute/pull/8) | `802958b`; **REPO** | Non-dispatching pinned RouterBench decoder and semantic regression evidence | Pinned checksum/shape/semantic tests and malicious-pickle rejection tests; merged CI | **PENDING** |
| [PR #2](https://github.com/Hbin77/tierroute/pull/2) | `f257823`; **REPO** | Fitted features, calibrated bilinear predictor, deterministic ridge replacement, artifacts, CLI training | Predictor, feature, ridge, artifact, CLI, and offline-training tests; merged CI | **PENDING** |
| [PR #4](https://github.com/Hbin77/tierroute/pull/4) | `28f45d0`; **REPO** | Exact lambda utility/tuning, nested LODO, policy artifacts, atomic bundle safety, resource guards | Lambda, artifact, atomic-I/O, exact-cost, CLI, and adversarial resource tests; merged CI | **PENDING** |
| [PR #10](https://github.com/Hbin77/tierroute/pull/10) | `5d40e74`; **REPO** | Build-backend replacement and deep dependency-license audit | Dependency-free wheel job, nested-license inspection, and locked license tests; merged CI | **PENDING** |
| [PR #11](https://github.com/Hbin77/tierroute/pull/11) | `24e7ad9`; **REPO** | Project-owned ridge-solver boundary and platform-local parity contract | Solver resolution, workload preflight, parity, artifact-loading, and offline tests; merged CI | **PENDING** |
| [PR #12](https://github.com/Hbin77/tierroute/pull/12) | `da6a16c`; **REPO** | Primary-source literature/novelty synthesis and claim-boundary review | Source-linked literature matrix with implemented/planned/gated distinctions; merged CI | **PENDING** |
| [PR #13](https://github.com/Hbin77/tierroute/pull/13) | `b85a77e`; **REPO** | Leakage-free per-query outer-LODO six-baseline orchestration | Fold leakage, observable-tag, order, ledger-guard, oracle, and baseline tests; merged CI | **PENDING** |
| [PR #14](https://github.com/Hbin77/tierroute/pull/14) | `cf56748`; **REPO** | Executed-call quote/realized-cost evidence and exact error reporting | Overspend, quote direction, accounting conservation, schema, and CLI tests, plus updated documentation; merged CI | **PENDING** |
| [PR #16](https://github.com/Hbin77/tierroute/pull/16) | `614a018`; **REPO** | Complete replay-scope identity, immutable canonical snapshot, comparison/fold binding, and resource hardening | Python 3.10: 409 tests; Python 3.12: 408 tests plus one expected compatibility skip; dual CI and dependency-free wheel green | **PENDING** |
| [PR #18](https://github.com/Hbin77/tierroute/pull/18) | PR and branch commits; **REPO** | This assistance ledger, explainability packet, submission draft, and future disclosure governance | Source/test/link audit and dual-Python locked verification are recorded in the PR; no named human review is claimed | **PENDING** |
| [PR #22](https://github.com/Hbin77/tierroute/pull/22) | PR and branch commits; **REPO** | Predictor artifact resource-contract design, implementation, adversarial tests, documentation, debugging, and repository workflow | Pinned platform-local v1 predictor hashes; exact parser/snapshot boundary tests; Python 3.10: 447 tests; Python 3.12: 446 tests plus one expected compatibility skip; dual locked offline verification | **PENDING** |
| [PR #23](https://github.com/Hbin77/tierroute/pull/23) | PR and branch commits; **REPO** | Replay JSON trust-boundary audit, finite limit design, implementation, adversarial tests, documentation, debugging, and repository workflow | Descriptor/UTF-8/parser/schema/collection/compound-training boundary tests; unchanged bundled-data SHA-256; Python 3.10: 530 tests; Python 3.12: 529 tests plus one expected compatibility skip; dual locked offline and dependency-free-wheel verification | **PENDING** |
| [PR #25](https://github.com/Hbin77/tierroute/pull/25) | PR and branch commits; **REPO** | Explicit locked inference/full-training reproduction lanes, Makefile contract tests, evidence-gate synchronization, documentation, and automated review | Both reproduction targets passed in fresh Python 3.10/3.12 environments; Python 3.10: 533 tests; Python 3.12: 532 tests plus one expected compatibility skip; Ruff, SPDX, licenses, install checks, offline inference, and offline training passed | **PENDING** |
| [PR #28](https://github.com/Hbin77/tierroute/pull/28) | `e87409a`; **REPO** | Reportable true nested-LODO learned-router benchmark, six-baseline comparison binding, deterministic evidence schema, tests, documentation, and repository workflow | Fold membership and prediction digests, canonical baseline configuration evidence, bounded lambda-search controls, deterministic benchmark JSON, adversarial comparison checks, and dual-Python offline/dependency-free-wheel CI | **PENDING** |
| [PR #29](https://github.com/Hbin77/tierroute/pull/29) | `0eb78cf`; **REPO** | Audited three-step Fast/Balanced/Premium showcase, direct nested-fold replay matching, scope/conservation guards, tests, documentation, and repository workflow | Canonical stream identity, fail-closed benchmark-query matching, separate oracle replay, exact cost/quality accounting, pinned JSON SHA-256, dual-Python offline/dependency-free-wheel CI, and current-tree fresh-clone audit in issue #19 | **PENDING** |
| [PR #31](https://github.com/Hbin77/tierroute/pull/31) | PR and branch commits; **REPO** | Stale assistance-ledger discovery and correction, current-tree reproduction evidence review, bilingual quickstart portability wording, documentation review, and repository workflow | Issue #30 acceptance record; merge/evidence links for PRs #28 and #29; issue #19 current-main fresh-clone audit; diff, Ruff, SPDX, and dual-Python CI checks | **PENDING** |
| [PR #33](https://github.com/Hbin77/tierroute/pull/33) | PR and branch commits; **REPO** | Plan/repository gap audit, issue drafting, five-page submission-source and architecture drafting, claim-boundary review, verification, and repository workflow | Issue #32 acceptance record; explicit implemented/measured/planned/organizer-gated states; per-number evidence template; synthetic-result prohibition; 555 tests passed with one expected skip in the pinned local environment | **PENDING** |
| [PR #35](https://github.com/Hbin77/tierroute/pull/35) | PR and branch commits; **REPO** | RouterBench local-diagnostic design, deterministic balanced-split and prefit-quote guards, private-download hardening, failure-envelope and claim-boundary review, implementation, adversarial tests, documentation, verification, and repository workflow | Issue #34 acceptance record; synthetic real-benchmark learned-plus-six-baseline E2E; membership-mutation, duplicate/domain/catalogue, prefit-overrun, staging-substitution, symlink, post-install re-authentication, source-archive exclusion, CLI-gate, surrogate-ID, and path-free failure-output checks; clean locked Python 3.10: 584 tests; clean locked Python 3.12: 583 tests plus one expected compatibility skip; license and offline reproduction gates passed | **PENDING** |
| [PR #37](https://github.com/Hbin77/tierroute/pull/37) | PR and branch commits; **REPO** | Current-main submission-outline evidence audit, claim-boundary correction, implementation-record design, documentation, verification, and repository workflow | Issue #36 acceptance record; stable `I-*` source/test/commit/CI bindings; aggregate oracle-gap template correction; PR #35 non-reportable RouterBench boundary; issue #19 fresh-clone evidence; relative-link, Ruff, SPDX, benchmark, and reproduction-contract checks | **PENDING** |
| [PR #39](https://github.com/Hbin77/tierroute/pull/39) | PR and branch commits; **REPO** | Korean eight-boundary explainability and mutation-workflow design, exact source/test mapping, fail-stop shell safety, claim-boundary review, documentation, verification, and repository workflow | Issue #38 acceptance record; eight forward/reverse patch checks without executing a deliberate mutation; 22 cited pytest nodes; blank Korean/English human sign-off tables; separate RouterBench synthetic versus optional local-artifact evidence; locked Python 3.12: 583 tests plus one expected compatibility skip; Ruff, SPDX, license, offline, and training smoke gates passed | **PENDING** |
| [PR #41](https://github.com/Hbin77/tierroute/pull/41) | implementation through `c649150` plus documentation commits; **REPO** | Deterministic regression-stump objective, residual/split/tie/resource design, inner-LODO calibrated GBM trainer, adversarial tests, claim boundaries, documentation, review, and repository workflow | Issue #40 acceptance record; hand-derived boosting, strict loss decrease, stable ordering, held-out isolation, exact OOF coverage, immutable bounded predictor state, aggregate pre-embedding work/catalogue rejection, offline inference, and full CI evidence recorded in PR #41 | **PENDING** |
| [PR #43](https://github.com/Hbin77/tierroute/pull/43) | branch commits and review evidence; **REPO** | Complete nested-GBM work preflight, paired bilinear/GBM evaluation design, full-precision JSON descriptive deltas, no-selection claim schema, CLI/smoke/tests, documentation, independent audit, and repository workflow | Issue #42 acceptance record; exact call-graph enumeration, pre-embedding rejection, identical scope/fold/tier/catalogue/search evidence, one shared baseline object, unavailable-value propagation, legacy benchmark golden preservation, socket-denial execution, and PR #43 CI/review evidence | **PENDING** |
| Issue #9 C11 dense candidate | branch source and review evidence; **REPO** | Dependency/license alternative audit, binary-protocol and C11 numerical design, authenticated process adapter, resource and path-race hardening, adversarial tests, documentation, verification, and repository workflow | Project-owned source and fixed protocol; strict C11/static/sanitizer checks; malformed corpus; reference/1,024-dimensional parity; platform-local binary/link hashes; full nested prepared-session and three-platform release gates remain open | **PENDING** |
| Issue #9 prepared nested-LODO graph slice | branch source, tests, and review evidence; **REPO** | Combinatorial call-graph derivation, immutable graph/resource-contract design, adversarial tests, binary64 parity-boundary review, documentation, and repository workflow | Independent logical-call oracle; exact 63-subset/154-block/`22N` seven-domain regression; pre-enumeration count/modeled-buffer/numeric-work refusal; no prepared execution, performance, bge-m3, or official-data claim | **PENDING** |
| Issue #9 prepared feature-store reference slice | branch source, tests, and review evidence; **REPO** | Fixed raw-layout and digest design, caller-checked source/precomputed-embedding boundary, bounded immutable store, Welford/Chan sufficient-statistics implementation, leakage and direct-constructor adversarial tests, documentation, claim review, and repository workflow | Canonical source/embedding golden identities; training-only tag/scaling and direct-matrix moment oracles; uneven-domain and excluded-domain noninterference; fail-before-traversal caps; no provider, file, solve, scoring, performance, bge-m3, or official-data claim; coefficient-to-report parity remains open | **PENDING** |
| [PR #47](https://github.com/Hbin77/tierroute/pull/47) — Issue #9 bounded prepared execution reference slice | implementation `f4b07bc`, tests `608468b`, admission/locality hardening `2ac1b50`, and branch documentation; **REPO** and **ASSISTANCE_RECORD** | AI-assisted implementation, adversarial and security-regression test generation, numerical/trust-boundary review, evidence and maintainer-document drafting, verification planning, and repository workflow for the bounded moment-ridge/coefficient/raw-score reference | Independent row-refit tolerance oracle; one Cholesky factor per subset shared across targets; exact 63-coefficient/154-score-block/`22N`/`22NM` regression; target-free scoring, lineage/locality, canonical-payload, exact-type, cap-boundary, residual, and little-endian golden tests; focused local Darwin arm64/Python 3.12.11: 62 passed; [implementation/spec-head CI run `29524753168`](https://github.com/Hbin77/tierroute/actions/runs/29524753168) at `8ec9cc1` passed Python 3.10/3.12, dependency-free wheel, and macOS/Windows native-source jobs. No provider, persistence, native execution, calibration, lambda, final-report, performance, bge-m3, RouterBench, or official-data claim; issue #9 remains open | **PENDING** |
| [PR #48](https://github.com/Hbin77/tierroute/pull/48) — Issue #9 bounded prepared calibration/lambda/report bridge slice | implementation `63e288e`, tests `3249a3c`, merge `566678c9c0181d9bcb76378ab423858150bff7b4`, documentation and review evidence; **REPO** and **ASSISTANCE_RECORD** | AI-assisted exact graph mapping, resource-estimate and evidence-schema implementation, adversarial test generation, same-runtime parity debugging, algorithm/security/test review, documentation drafting, verification planning, and repository workflow for the bounded raw-score-to-report bridge | Existing isotonic/lambda/simulator reuse; exact `C(D,2)+D` calibration and `D^2` destination coverage; D4–D7 full learned-result parity; cap/tie/1-ULP/quote-realized/cumulative/replay-order/lineage checks; cost-width, five-pass pair-work, digit-cap, child-amplification, and fail-order guards; local Python 3.10 954 passed and Python 3.12 953 passed/1 expected skip; [PR-head CI `29530846709`](https://github.com/Hbin77/tierroute/actions/runs/29530846709) and [merged-main CI `29531008829`](https://github.com/Hbin77/tierroute/actions/runs/29531008829) passed Python 3.10/3.12, dependency-free wheel, and macOS/Windows native-source jobs. No human review is implied. No provider, persistence, native/scalable session, all-domain artifact, prepared six-baseline wrapper, performance, bge-m3, RouterBench, or official-data claim; issue #9 remains open | **PENDING** |
| [PR #50](https://github.com/Hbin77/tierroute/pull/50) — Issue #9 authenticated file-backed native prepared-session slice | implementation head `7ee6188f4bd0a958210e302a837738e763e1fe65`, merge `ffa8b8059985298df9d1cf0feec20374589afc1c`, protocol, tests, and review evidence; **REPO** and **ASSISTANCE_RECORD** | AI-assisted fixed-file/protocol design, streaming authenticated persistence, single-invocation C11 moment-solve/raw-score implementation, adapter and mmap-lifetime design, Windows metadata-portability debugging, release-payload gate hardening, adversarial/security/numerical test generation, claim review, documentation drafting, verification planning, and repository workflow | Project-owned dependency-free `TRPSTO01`/`TRPSES01`/`TRPRES01` source; caller-pinned whole/source/logical/embedding and binary identities; descriptor/path-race/private-copy checks; public/C admission-limit parity without hidden C-only caps; nonzero nonce rejection sampling; retryable lock-serialized exported-view close/read behavior; structured-status, timeout/crash, malformed-result, lineage, and mmap-lifetime failures; focused local Darwin store/native run 64 passed, including 38 native-session cases; actual compiled D4–D7 complete-reference parity on small surface fixtures; and unprojected `D4/N8/d1036/M1` 12+1,024 synthetic-feature completion. [PR-head CI `29537455566`](https://github.com/Hbin77/tierroute/actions/runs/29537455566) and [merged-main CI `29537633261`](https://github.com/Hbin77/tierroute/actions/runs/29537633261) passed Python 3.10/3.12, dependency-free wheel/sdist integrity, and macOS/Windows ephemeral source compile, protocol/parity, and link/import audits. The official `D7/N34778/d1036/M11` shape is preflight-only. No human review, release-artifact approval, bge-m3 provider, official/RouterBench data, policy/all-domain/six-baseline integration, or performance/quality/cost claim is implied; issue #9 remains open | **PENDING** |
| [PR #52](https://github.com/Hbin77/tierroute/pull/52) — Issue #9 native prepared per-query policy benchmark bridge slice | implementation/spec commits `f159e04`, `85393e2`, `a8e0896`, and `9ed400d`; evidence commits `77e5c47` and `304decd`; merge `c7b717ce1226fcfd70d696d0124aa8df294033c8`; [implementation/spec branch-push CI `29542245699`](https://github.com/Hbin77/tierroute/actions/runs/29542245699) at `9ed400d580e288bb9648a300a8de12a5c2200fff`, [final PR-head CI `29543435978`](https://github.com/Hbin77/tierroute/actions/runs/29543435978) at `304decd0a591fcfc5e5a1e04f35bf20b22c17cea`, and [merged-main CI `29543610611`](https://github.com/Hbin77/tierroute/actions/runs/29543610611); merged source, tests, and documentation; **REPO** and **ASSISTANCE_RECORD** | AI-assisted two-phase ownership and credential-boundary design, bounded fixed-per-query learned-plus-six-baseline implementation, evidence-schema and return-graph review, adversarial test generation, parity debugging, documentation drafting, local verification planning, and claim-boundary audit | Public `evaluate_native_prepared_per_query_benchmark` consumes a caller-owned open result plus mandatory external binary/result/store pins; verifies before deep traversal and after the last mapped read; uses `at()`-only native views; closes its owned store before fixed `PerQueryBudgetLedger` learned and six-baseline replay; and returns no mapped view or score matrix. Compiled surface-only D4-D7 results strictly equal the rowwise learned and six-baseline results, including an uneven three-model D7 fixture. Adversarial tests cover credential fail-order, persistent mutation, final pin replacement, bit-exact targets, phase close, primary-error preservation, bounds, and recursive return-graph absence. Locked local evidence: focused native run 89 passed; Python 3.10.19 with pip 26.1.2 full suite 1,044 passed with no skip; Python 3.12.10 with pip 26.1.2 full suite 1,043 passed and one expected skip for the locked Python 3.10 `typing_extensions` compatibility dependency. All three CI runs passed Python 3.10, Python 3.12, dependency-free wheel, Native source portability macOS, and Native source portability Windows. This is merged-main source-portability evidence, not distributable release-artifact approval; no human review is claimed. No external-data, provider, all-domain artifact, command, trainer, quality, cost-reduction, or performance result is claimed; issue #9 remains open | **PENDING** |
| [PR #54](https://github.com/Hbin77/tierroute/pull/54) — Issue #53 post-PR #52 current-state evidence synchronization | evidence merge binding `cf6b9e0`; report/contract claim synchronization `5e83402`; assistance record `779f3ba`; [branch-push CI `29544955116`](https://github.com/Hbin77/tierroute/actions/runs/29544955116) and [pre-final-evidence PR-head CI `29545002706`](https://github.com/Hbin77/tierroute/actions/runs/29545002706) at `779f3ba739f7966f02a6fad88e53b4cd673121c6`; branch documentation; **ASSISTANCE_RECORD** | AI-assisted stale-claim discovery, immutable submission-evidence drafting, architecture-topology correction, historical-boundary preservation, cross-document claim audit, independent overclaim and link review, verification planning, and repository workflow | PR #52 final-head/merge/CI identities are distinguished from earlier slice evidence; `I-PREPARED-NATIVE-POLICY-85393E2` binds exact source/test/CI and fixed per-query limits; native evidence no longer flows into the unrelated paired-family sink; the three-module 89-case source, configured bounded-search semantics, store/result/example/tier inputs, remaining human gate, and official/full-shape/release/performance non-claims are explicit. Two independent audits report no remaining blocker; relative-link and `git diff --check` audits pass; locked Python 3.12 full verification passed with 1,043 tests and one expected skip. Both recorded CI runs passed Python 3.10, Python 3.12, the dependency-free wheel, and macOS/Windows native-source jobs. No source, dependency, or SBOM-inventory change is made, and no human review is implied | **PENDING** |
| [PR #56](https://github.com/Hbin77/tierroute/pull/56) — [Issue #55](https://github.com/Hbin77/tierroute/issues/55) canonical calibrated GBM artifact v1 | implementation `5d1d727`; hardening `4de98de`; adversarial tests and implementation head `5be3642`; [push CI `29547428173`](https://github.com/Hbin77/tierroute/actions/runs/29547428173) and [PR CI `29547447826`](https://github.com/Hbin77/tierroute/actions/runs/29547447826) passed at that implementation head; documentation head `43c3353` with [push CI `29548001060`](https://github.com/Hbin77/tierroute/actions/runs/29548001060) and [PR CI `29548002589`](https://github.com/Hbin77/tierroute/actions/runs/29548002589) passed; final evidence head `ef8606f34d8a7706a19ae2303d742a06c955d3cb` with [push CI `29548164885`](https://github.com/Hbin77/tierroute/actions/runs/29548164885) and [PR CI `29548166228`](https://github.com/Hbin77/tierroute/actions/runs/29548166228) passed; merged as `a1d7bd7dd835a1ab88e85e805df167985ca699be` with [merged-main CI `29548281471`](https://github.com/Hbin77/tierroute/actions/runs/29548281471) passed; merged source, tests, and documentation; **REPO** and **ASSISTANCE_RECORD** | AI-assisted artifact-schema and canonical-serialization design, bounded fit/load/save and offline reconstruction implementation, exact-type and provider/data trust-boundary hardening, parser/resource-limit review, adversarial test generation, documentation drafting, claim review, verification planning, and repository workflow | Canonical strict-JSON GBM artifact v1 is implemented as a library-only boundary with exact tiny-document/SHA, round-trip, signed-zero, snapshot, parser, resource-cap, pre-embedding, offline, and atomic-save evidence in [`test_gbm_artifacts.py`](../tests/test_gbm_artifacts.py) and [`test_gbm_artifact_hardening.py`](../tests/test_gbm_artifact_hardening.py). The bilinear v1 artifact bytes are unchanged. There is no GBM CLI/policy integration, external or official-data result, bge-m3 backend or asset, deployment/performance/quality/savings evidence, dependency change, or SBOM-inventory change. Each final-head push/PR run and the merged-main run passed all five jobs; no human review or walkthrough is claimed | **PENDING** |
| [Issue #58](https://github.com/Hbin77/tierroute/issues/58) — current-main maintainer walkthrough re-audit | implementation snapshot `a1d7bd7dd835a1ab88e85e805df167985ca699be`; packet-pin commit `f34c729`; PR #56 evidence-sync commit `132f873`; bilingual mutation re-audit commit `43c008d`; branch documentation and local re-audit evidence; **REPO** and **ASSISTANCE_RECORD** | AI-assisted stale-snapshot discovery, eight-card mutation-boundary re-audit, shell safety and stale-bytecode review, subclaim decomposition, bilingual documentation drafting, CI-evidence synchronization, claim review, verification planning, and repository workflow | Cards 1/2/3/7 retain byte-identical mutation sources but were re-executed at the reviewed snapshot; cards 4/5/6/8 were re-audited against current paths and claim boundaries. Card 5 is split into six independently human-owned sub-reviews; Card 6 is explicitly bilinear-artifact-only; Card 8 scopes asynchronous rollback, basetemp cleanup, offline reproduction, package, SPDX, license, and ephemeral native-source evidence. Detached local mutation evidence is recorded in the packet; clean locked Python 3.12 verification passed 1,094 tests with one expected compatibility skip. No human row is populated, and automated re-audit does not establish entrant understanding, official-data, performance, quality, savings, deployment, release-artifact, or issue #9 completion | **PENDING** |

The four issue #9 prepared execution/policy/native-session/native-policy rows record generated code, tests, and documentation
as AI-assisted work. They make no claim that those lines were written without
assistance, that automated or AI-agent review proves understanding, or that the entrant
has completed the walkthrough. The human owner must independently derive the moment
equations and `22N` structure; trace builder versus direct-constructor trust; explain
the calibration graph, five lambda pair traversals, cost-width admission, empirical
2,048 Python-reference residual factor, authenticated file/child lifetime, C11 residual
gate, public-versus-native cap agreement, external pin origins, the two-phase mapped-to-
owned boundary, fixed per-query replay, and payload-free return graph; run predicted
failure mutations; and record reviewed commits before changing **PENDING**. The PR #50
row's lack-of-policy-integration statement is historical to that merged slice; the
subsequent pending row records the merged PR #52 bridge without rewriting the
earlier merge evidence.

Future material AI-assisted changes must add or update one ledger row in the same PR.
Pure typo or formatting changes may state the assistance in the PR without adding a
row, provided they do not change a critical invariant or submission claim.

## What completion requires

For each critical boundary, the owner must be able to do all of the following without
asking the assistant for the answer:

1. State the invariant and why the chosen package boundary owns it.
2. Trace one successful input and one failure path through the implementation.
3. Explain the threat, leakage, numerical, or accounting bug prevented by the strongest
   adversarial test.
4. Name the trust boundary and one limitation that remains intentionally unresolved.
5. Run the focused tests, predict how a deliberate mutation should fail, and confirm the
   observed failure.
6. Record owner, date, reviewed commit, and notes in the sign-off table; never backdate
   or infer sign-off from CI.

The required source/test map and blank sign-off table are in
[maintainer-explainability.md](maintainer-explainability.md).

## Submission appendix draft (Korean)

The text below is a **draft**, not a completed declaration. Update the tool period and
review status at submission time, and do not change future tense to a completion claim
until the owner sign-off table is actually complete.

> 개발 과정에서 코드 작성·디버깅 보조를 위해 OpenAI Codex 코딩 에이전트를
> 사용하였다. 현재 개발 환경은 GPT-5 기반이라고 표시하지만, 당시 사용된 정확한 모델
> ID/checkpoint/revision은 서비스에서 노출되지 않았고 별도로 보존하지 않아 특정하지
> 않는다. 사용 범위는 요구사항 분해, 구현·리팩터링 제안, 테스트와
> 공격적 실패 사례 설계, 코드 리뷰, 문서 초안 및 로컬 검증 자동화이다. 대회 구조,
> one-shot 정책, LODO 검증, 오프라인 실행, 라이선스와 승인 gate는 참가자가 정한
> 제약이다. Codex는 출품작의 런타임·학습·평가·추론 경로에 탑재되지 않으며 출품
> 모델이나 벤치마크 후보가 아니다. 제안된 변경은 이슈·Conventional Commit·PR과
> 테스트로 추적하며, 자동 검사나 AI 리뷰를 참가자의 이해도 검토로 대신하지 않는다.
> 최종 제출 전 참가자가 핵심 모듈별 동작 원리·실패 경로·제약을 직접 설명하고 검증한
> 기록을 저장소의 maintainer sign-off 표에 남긴다.

If a local `bge-m3` provider or another model is later implemented, disclose that model
separately under the contest's deployed-model category. This development-assistance
entry does not replace the model license, revision, weight, or training disclosure.
Whether a trained tierroute ridge/bilinear-plus-isotonic artifact belongs in Appendix 2
type 3 remains an organizer question. It is not external-model fine-tuning; obtain a
written interpretation before completing the final declaration.

## Maintenance rules

- Keep statuses factual. `Pending`, `in progress`, and `complete` are not interchangeable.
- Record the exact reviewed Git commit; a later code change invalidates only the affected
  boundary and must move that row back to pending.
- Link critical PRs to an issue, state AI/tool assistance or `None`, and preserve the
  relevant commands/results in the PR body.
- Do not add private prompts, credentials, contest-only data, or unverifiable percentages
  merely to make this ledger look more detailed.
- Update the submission draft when the actual Appendix 2 form is available; the form and
  organizer instructions take precedence over this repository draft.
