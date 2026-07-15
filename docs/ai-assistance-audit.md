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
The last row records the current governance PR outside that retrospective snapshot.

## Assistance boundary

| Field | Recorded fact |
|---|---|
| Service | OpenAI Codex coding agent; the current development environment identifies a GPT-5 basis |
| Exact model revision | The exact historical model ID/checkpoint was not exposed or retained; no reproducible revision is claimed |
| Period covered | Repository work on 2026-07-15 and 2026-07-16 through PR #16 |
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

As of the `614a018` snapshot, all ten merged pull requests listed below have zero
submitted GitHub reviews, zero issue comments, and zero review comments. Their
successful checks are automated evidence only. References in older development records
to an "independent review" or "independent audit" are classified in this ledger as
AI-agent or automated evidence unless a named human review record is linked explicitly.

Ledger evidence uses four labels:

- **REPO**: directly verifiable from committed source, tests, Git history, or GitHub
  pull-request/check metadata.
- **MAINTAINER_DECLARATION**: a statement made and owned by the human entrant; it is not
  inferred from authorship metadata.
- **NOT_RECORDED**: the repository or service did not retain enough information to make
  a narrower claim.
- **PENDING**: a future human action is required and must not be inferred from CI,
  commits, authorship, or AI review.

## Retrospective change ledger

"Pending" means that the code may be tested and merged while the entrant's explicit
explain-without-assistance walkthrough has not yet been recorded. It does not mean the
change is known to be defective.

Merge, source, test, and check facts below are **REPO** evidence. Assisted-scope entries
are **MAINTAINER_DECLARATION**; the exact historical model revision is
**NOT_RECORDED**; every owner walkthrough is **PENDING**.

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
