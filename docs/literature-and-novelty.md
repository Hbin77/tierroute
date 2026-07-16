<!-- SPDX-License-Identifier: Apache-2.0 -->

# Literature evidence and novelty boundary

This note records the W1 close reading behind tierroute's design. It separates prior
research, organizer requirements, current implementation, and future work so that the
README, result report, and judging answers make only reproducible claims. The source
snapshot date is **2026-07-16**.

## Claim discipline

The labels in this document have precise meanings:

- **Implemented** means the behavior exists on `main` and is exercised by tests or the
  bundled smoke pipeline.
- **Planned** means the architecture has an interface or design slot, but a report must
  not present it as working functionality.
- **Gated** means work must wait for an official answer, a license decision, or a
  separately reviewed dependency/backend.

Results reported by a cited paper apply to that paper's models, datasets, cost model,
and split. They are motivation, not tierroute benchmark results. Fast, Balanced, and
Premium are organizer-defined tiers; their existence and the organizer's low-budget
emphasis are requirements rather than tierroute novelty.

## Source ledger

| Source | Reviewed snapshot | What it establishes for this project |
|---|---|---|
| [SK Telecom task page](https://www.kossa.kr/materials/2026/ossp/tasks-skt.html) | Accessed 2026-07-16 | Three budget tiers, low-budget emphasis, router input/action shape, local-only execution, public/private replay outline, and the three organizer-recommended materials below. |
| [RouteLLM](https://arxiv.org/abs/2406.18665) | arXiv `2406.18665v4` | Preference-trained strong/weak routing, a win-probability threshold controlling strong-model call share, matrix-factorization routing, call-share/performance metrics, and conditional OOD evidence. |
| [RouterBench](https://arxiv.org/abs/2403.12031) | arXiv `2403.12031v2` | Pre-generated multi-model outcomes, cost-quality evaluation, oracle construction, and predictive routing without live model inference. |
| [FrugalGPT](https://openreview.net/forum?id=cSimKw5p6R) | TMLR final paper, OpenReview `cSimKw5p6R` | Budget-constrained prompt adaptation, approximation, and sequential LLM cascading with response-quality scoring. |
| [A Unified Approach to Routing and Cascading for LLMs](https://proceedings.mlr.press/v267/dekoninck25a.html) | ICML 2025, PMLR 267; arXiv `2410.10347v3` | Linear cost-quality routing, optimal cascading under the paper's formulation, cascade routing, and the distinct roles of ex-ante and post-hoc quality estimates. |
| [Beyond Decoding tutorial](https://cmu-l3.github.io/neurips2024-inference-tutorial/) | NeurIPS 2024 tutorial | The organizer-recommended taxonomy of generation, meta-generation, and efficient generation. |
| [From Decoding to Meta-Generation](https://arxiv.org/abs/2406.16838) | arXiv `2406.16838v2`; TMLR 2024 | A formal black-box view of chained, parallel, step-level, and refinement meta-generators, plus token-cost and latency analysis. |
| [Inference Scaling Laws](https://arxiv.org/abs/2408.00724) | arXiv `2408.00724v3`; ICLR 2025 | Empirical evidence that the compute-optimal model and inference strategy depend on the budget, with important task- and strategy-specific limits. |
| [BEST-Route](https://proceedings.mlr.press/v267/ding25d.html) | ICML 2025, PMLR 267 | Joint selection of a model and best-of-n sample count under a quality target. |
| [Learning to Route LLMs with Confidence Tokens](https://proceedings.mlr.press/v267/chuang25b.html) | ICML 2025, PMLR 267 | A calibrated response-conditioned binary cascade from a small to a large model. |
| [Universal Model Routing](https://openreview.net/forum?id=ka82fvJ5f1) | ICLR 2026; arXiv `2502.08773v2` | Bilinear prompt-model scoring, cost-sensitive one-shot routing, and adaptation to previously unseen models. |
| [PROTEUS](https://arxiv.org/abs/2601.19402) | arXiv `2601.19402v3`; EuroMLSys 2026 submission | A query-adaptive PPO router conditioned on a runtime accuracy target with Lagrangian control. |
| [LLMRouterBench](https://aclanthology.org/2026.findings-acl.1881/) | Findings of ACL 2026 | A broader logged multi-model collection/evaluation framework with adapters, baselines, oracle metrics, and Pareto analysis. |

The [2026 contest rules](https://api.osscontest.kr/static/uploads/b3b4491a-3bbe-454e-a1d8-6ed475b01b14.pdf)
are a separate compliance source. They do not establish algorithmic novelty. The
project records third-party software, data, and model terms in [SBOM.md](../SBOM.md).

## Close reading and design consequences

### RouteLLM

RouteLLM studies a binary decision between a stronger, more expensive model and a
weaker, cheaper model. Its routers estimate the strong model's preference win
probability and compare it with a threshold that controls how frequently the strong
model is used. Its matrix-factorization router learns bilinear model and prompt
representations. The paper reports performance-gap recovery, average performance-gap
recovery, and a call-performance threshold metric over call-share/performance curves.
tierroute's oracle-gap recovery is analogous normalization, but its endpoints are the
always-cheapest policy and, currently only under per-query accounting, an independently
budget-feasible multi-model oracle rather than RouteLLM's fixed weak and strong models.
The names and values are not interchangeable.

The OOD result needs a narrow statement. Arena-only routers were close to or worse than
random on the paper's [MMLU](https://arxiv.org/html/2406.18665#S5.T2) and
[GSM8K](https://arxiv.org/html/2406.18665#S5.T3) transfer experiments, while small
in-domain augmentation improved them. Its other transfer result adds unseen
Claude/Llama model pairs on the same MT-Bench benchmark; that is cross-model transfer,
not prompt-domain OOD robustness. The evidence therefore supports **domain-shift
testing**, not a universal claim that RouteLLM always collapses OOD.

tierroute uses RouteLLM as a related binary one-shot and matrix-factorization comparison.
The exact `predicted_quality - lambda * cost` lineage instead comes from RouterBench and
the unified-routing formulation. tierroute differs today by handling a multi-model
catalogue, using one immutable lambda per budget tier, calibrating per-model scores, and
requiring nested leave-one-domain-out evaluation for reportable learned-policy results.
The lambda form and bilinear idea themselves are prior art and must not be claimed as new.

RouteLLM code is Apache-2.0, but its
[`arena_battles_embeddings`](https://huggingface.co/datasets/routellm/arena_battles_embeddings/blob/6cc7277cab42bb81c094c58cbe45b2e3646a9201/README.md)
and [`mmlu_battles`](https://huggingface.co/datasets/routellm/mmlu_battles/blob/c1ce42bbe822f90c5177fb1f475a162d7e95a121/README.md)
cards do not declare dataset licenses at the pinned revisions.
[`gpt4_judge_battles`](https://huggingface.co/datasets/routellm/gpt4_judge_battles/blob/2a1afe8d0659904c0f6f59de6179e086fdb027c7/README.md)
declares Apache-2.0 but derives from
[Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar/blob/3c6b4c47fa1cc38869f9f32dce1699f7abad8b06/README.md),
whose card adds artifact-specific competition, research/non-commercial,
upstream-model, OpenAI-data, privacy, and safety qualifications. tierroute copies none
of these artifacts and treats reuse as `NOASSERTION` pending a separate review.

### RouterBench

RouterBench stores prompts, outputs, quality measurements, and costs from multiple
models. This lets a router be trained and evaluated against logged outcomes without
issuing live LLM calls. Its evaluation traces the cost-quality trade-off, includes an
oracle that chooses the best-quality response with a cheapest-cost tie-break, and uses
normalized area under the cost-quality curve, called AIQ, to compare systems. Its
prompt-independent [**Zero router**](https://arxiv.org/html/2403.12031#S3.SS2) mixes
standalone models on their non-decreasing convex hull. This is a stronger control than
cheapest, premium, or uniform random when randomized expected-cost policies are legal.
The [learned predictive router](https://arxiv.org/html/2403.12031#S5.SS1) selects by
`lambda * predicted_performance - cost`; its lambda therefore moves in the opposite
direction from tierroute's cost-penalty lambda, although the rankings are equivalent
after a positive reciprocal reparameterization.

On the paper's [within-task 70/30 experiments](https://arxiv.org/html/2403.12031#S5.SS3),
learned KNN/MLP routers do not consistently beat Zero across tasks. Its
[cascade sensitivity study](https://arxiv.org/html/2403.12031#S5.SS2) also uses true
final quality as the judge and adds synthetic judge error; it is an upper-bound
analysis rather than a deployable response-quality estimator. These results strengthen
two controls for tierroute: compare learned routing against prompt-independent cost
frontiers, and do not infer practical cascade gains from an oracle-like judge.

tierroute uses the benchmark only as an optional boundary validation source. Its
simulator keeps all uncalled outcomes and ground-truth quality private until the chosen
action is replayed. The current pinned 0-shot artifact is never committed. The
[pinned Hugging Face dataset card](https://huggingface.co/datasets/withmartian/routerbench/blob/784021482c3f320c6619ed4b3bb3b41a21424fcb/README.md)
does not declare a dataset license, so tierroute records `NOASSERTION`; the MIT license
of RouterBench's GitHub code repository does not license that separate dataset.

Offline replay, an oracle, and cost-quality curves are consequently benchmark-aligned
infrastructure, not research novelty. tierroute's additional engineering contribution
is the strict typed boundary, authenticated optional download, exact cost arithmetic
and replay accounting, and provenance-bound artifacts. Zero is a valuable seventh
diagnostic if the challenge permits randomized expected-cost mixtures; it remains gated
because the current six-baseline contract and hard-budget semantics do not establish
that legality.
RouterBench also combines heterogeneous evaluators, including exact match and model
judging, behind normalized quality values. Any report must therefore retain evaluator
and per-domain provenance instead of treating every normalized label as identical.

### FrugalGPT

The [final FrugalGPT paper](https://openreview.net/forum?id=cSimKw5p6R) frames LLM use as
a budget-constrained optimization problem and explores
prompt adaptation, LLM approximation, and LLM cascades. Its cascade calls services in
sequence, scores the query/response pair for reliability, and either returns the
current response or escalates according to learned thresholds. It returns the response
at the stopping stage rather than choosing the best response ex post from every model
it called. Because uncertain queries can reach every service, escalation accumulates
all invoked call costs.

The paper reports large savings and occasional quality improvements on its evaluated
datasets and APIs. Those numbers are not transferable tierroute claims. Its main
experiments use random 50/50 splits and it separately studies synthetic label shift.
The paper also states practical limits: the cascade needs labeled examples and assumes
that training and test examples have the same or a similar distribution.
Its formal and implemented online budget accounts for invoked API calls; the released
accumulator does not add scorer/router inference cost, while training and data
collection are discussed as upfront costs. tierroute must distinguish model-call spend
from total system cost in any future operational claim.

tierroute does not currently implement this cascade. Its state/action contract can
represent another call or selection of a prior output, but the shipped policy makes
one call and then selects that output. Sequential escalation remains gated on the
official simulator's call and accounting semantics.

### Unified routing and cascading

[Dekoninck et al. §2](https://arxiv.org/html/2410.10347#S2) formulate expected-quality
maximization under an expected-cost budget. Their one-shot router selects by an
estimated quality-minus-lambda-cost trade-off; a randomized mixture between
deterministic tie policies can be required to hit an exact expected budget. Their
[cascade](https://arxiv.org/html/2410.10347#S3) treats sequences of calls as
“supermodels,” and [cascade routing](https://arxiv.org/html/2410.10347#S4) generalizes
both one-shot routing and fixed-order cascading by choosing what to run next after
observing earlier outputs.

That theorem does not transfer unchanged to tierroute: its objective uses estimated
expectations over a representative distribution, whereas tierroute adds a hard
affordability filter, deterministic lower-cost ties, and an explicit remaining balance.
Complete finite breakpoint replay is exact only inside that deterministic configured
policy class.

An expected-budget policy may spend more than its nominal budget on one query and
offset that cost elsewhere. It does not solve finite-horizon allocation over an ordered
stream with a remaining balance. tierroute's hard per-query/cumulative feasible set is
therefore different, not merely a stricter implementation of the theorem.

The most useful design result is conditional: reliable **ex-ante** quality estimates
are critical for routing, while cascading needs a materially more informative
**post-hoc** response-quality estimate. If observing an answer adds little information,
the paper says direct routing is more effective than paying for the cascade. Its
experiments find cascade routing stronger than the compared individual strategies in
the evaluated settings.

The general cascade uses stage-specific trade-off parameters and compares stopping
with every future supermodel. A simple confidence threshold is equivalent only under
the paper's restrictive conditions on costs, future-model estimates, and supermodel
quality. FrugalGPT-style thresholding is a practical restricted cascade, not the
unified optimum.

The defensible conclusion is therefore not “cascade always loses” or “pure cascade is
always dominated because the first call is wasted.” Cascade routing is a more expressive
strategy under the paper's assumptions. tierroute keeps one-shot routing as the default
because the challenge's sequential-call and budget semantics are unconfirmed and no
validated post-hoc estimator exists yet. This preserves the current technical decision
while correcting an over-broad rationale.

A future adapter must also settle output semantics. FrugalGPT explicitly returns its
first threshold-accepted response. The unified formulation values a supermodel through
an expected maximum, and its reference implementation selects the called response with
highest estimated quality, while parts of the paper's cascade prose describe returning
the latest response. tierroute cannot treat earlier-output selection as an automatic
consequence of the theorem.

Both released workflows use all-model responses offline for estimator fitting and
policy calibration. At deployment, uncalled responses and realized correctness are
unavailable; a best-realized-model choice is an oracle or training label, not a
deployable action.

### Organizer-recommended meta-generation materials

The NeurIPS tutorial and its TMLR survey treat a meta-generator as a program that calls
black-box generation subroutines and combines partial or complete sequences. The survey
groups common patterns into chained, parallel, step-level search, and refinement, and
separately analyzes token cost and generation speed. This is the broad conceptual home
for routing and cascading, but tierroute's official action space is narrower: it may
call a listed candidate or select a previously called candidate output. It does not
control token decoding or generate arbitrary extra samples.

Wu et al. compare model sizes and inference strategies under FLOP budgets. In their
mathematical-reasoning experiments, smaller models with additional sampling or their
REBASE tree search can occupy the Pareto frontier and outperform larger-model
configurations at tested budgets. Returns eventually saturate, and the best allocation
changes with budget and verifier quality.

This supports the premise that “largest model” is not automatically compute-optimal.
It does not prove that repeated sampling, voting, or tree search is legal or beneficial
in the SK Telecom simulator. Those methods need generation freedom, verifier signals,
and accounting semantics that have not been confirmed.

### Broader 2025–2026 novelty check

The four core papers and three organizer materials are not enough to support a “first”
claim in 2026. A targeted primary-source check found further direct overlap:

- [**BEST-Route**](https://proceedings.mlr.press/v267/ding25d.html) chooses a model and
  best-of-n sample count from prompt difficulty,
  then finds a low-cost configuration that meets a quality threshold. Its main split
  is random 8K/1K/1K with a separate MT-Bench OOD experiment. It does not carry an
  exact remaining balance or call history, but it precludes novelty claims around
  prompt difficulty, model choice, and inference-compute allocation.
- [**Confidence Tokens**](https://proceedings.mlr.press/v267/chuang25b.html) always
  calls a small model, conditions a calibrated confidence
  decision on that response, and escalates to a large model when confidence is low.
  This precludes novelty claims around calibrated response-based escalation. tierroute
  currently avoids the first-call cost by making an ex-ante one-shot choice and does
  not implement this history-adaptive cascade.
- [**Universal Model Routing**](https://openreview.net/forum?id=ka82fvJ5f1) already
  combines bilinear prompt-model representations
  with a predicted-loss-plus-lambda-cost rule and can add unseen models using a small
  labeled validation set. Its main generalization target is unseen models rather than
  prompt domains. tierroute's catalogue is fixed by its artifact, so unseen-model
  adaptation is not a current claim; nested prompt-domain LODO is the relevant
  difference.
- [**PROTEUS**](https://arxiv.org/abs/2601.19402) conditions a learned policy on a
  runtime accuracy target, uses named
  service-level examples, and applies Lagrangian control. Its constraint is an
  aggregate accuracy floor rather than tierroute's explicit tier, exact remaining
  balance, and replay ledger. It precludes claims that tier- or SLA-aware Lagrangian
  routing is new.
- [**LLMRouterBench**](https://aclanthology.org/2026.findings-acl.1881/) standardizes
  logged collection, evaluation, router adapters, ten
  baselines, cost-quality frontiers, and oracle comparisons. Its published main results
  use within-dataset 70/30 splits over five seeds rather than nested domain holdouts.
  It precludes novelty claims around offline multi-model benchmarking, adapters,
  logged replay, Pareto analysis, or oracle gaps.

The last benchmark is distinct from the 2024 RouterBench dataset. Its
[official code repository](https://github.com/ynulihao/LLMRouterBench/tree/c77cb0506949d8f959e97967d2fefca0e8ff1b05)
has an MIT badge but no `LICENSE` file at the reviewed commit, and its
[official dataset revision](https://huggingface.co/datasets/NPULH/LLMRouterBench/tree/0e5af1b84bf73437a01a1849c0f1d2468baa93fc)
has no declared dataset license. Both remain `NOASSERTION`; tierroute does not copy or
redistribute them.

This broader check makes the safe position explicit: tierroute is an offline-first,
auditable implementation of established cost-sensitive routing primitives. Its current
differences are exact cost arithmetic and ledger replay, adapter-localized specification
uncertainty, end-to-end nested LODO for the learned policy, and reproducible calibrated
artifacts—not a newly invented routing objective.

## Comparison matrix

“Not reported” means the item was not a stated contribution of the reviewed paper; it
does not assert that no external implementation could add it.

| Dimension | RouteLLM | RouterBench | FrugalGPT | Unified routing/cascading | tierroute at this snapshot |
|---|---|---|---|---|---|
| Primary role | Learned binary strong/weak router | Multi-model benchmark and reference routers | Cost-saving framework and learned API cascade | Theory and algorithms unifying routing and cascading | Challenge-facing, offline-first multi-model router library |
| Decision timing | One model call | Logged single-choice routing is central | Sequential calls with response-based stopping | One-shot, cascade, or dynamically ordered cascade routing | **Implemented:** one call, then select it; adaptive reuse of history is **planned** |
| Cost control | Win-probability threshold controls strong-model call share; no hard dollar budget or named tier | Willingness-to-pay and cost-quality frontier | Explicit expected-budget service selection | Lambda for an expected-cost budget | **Implemented:** deterministically tuned per-tier lambda under the configured replay plus per-query/cumulative ledgers |
| Quality signal | Preference win probability; several router models including MF | Logged scores and learned performance estimates | Query/response reliability score | Ex-ante and post-hoc estimators with uncertainty | **Implemented:** surface-feature bilinear scores and an in-memory deterministic stump-GBM core, each with per-model isotonic calibration, plus a paired descriptive runner. **Planned:** local bge-m3 and a licensed family-selection-aware experiment |
| Distribution-shift evidence | Conditional MMLU/GSM8K OOD weakness and in-domain recovery | Main predictive setup uses within-task 70/30 splits and also reports held-out-task plots | Main setup uses random 50/50 splits, calls out same/similar distributions, and studies synthetic label shift | Multiple benchmarks and estimator-noise studies; no tierroute-style nested LODO reported | **Implemented for both fixed surface-only families:** nested LODO covers predictor fit, calibration, lambda selection, and outer scoring on identical folds. **Not implemented:** a reportable licensed-data result or unbiased family selection |
| Primary reporting | Call-share/performance curves, PGR/APGR/CPT | AIQ over a shared cost-quality frontier and oracle | Quality under budget and cost savings | Area under cost-quality curve | **Implemented:** configured tier-weighted quality, per-query oracle-gap formula, same-outer-fold six-baseline report, and exact quote-error evidence. **Planned:** cumulative sequence oracle. |
| Offline replay | Can evaluate learned routes, but live provider integration is in scope | Core benefit of pre-generated outcomes | Built around commercial API calls in the paper | Mix of logged and real benchmark experiments | **Implemented:** no-network runtime, labels isolated from router state, executed-call cost evidence, synthetic clone-first demo |
| Budget-tier awareness | User-selected win-probability threshold and call share, not named challenge tiers | Continuous cost-quality evaluation | Budget is explicit, not the challenge's three-tier contract | Expected budget parameter | **Implemented:** Fast/Balanced/Premium are first-class state and policy keys; official weights remain **gated** |
| Call-history contract | Not central to binary one-shot routing | Not a central predictive-router input | Earlier responses drive cascade decisions | Earlier outputs update sequential decisions | **Implemented:** typed history/action boundary; **not implemented:** history-adaptive policy |
| Audit and packaging focus | Research library | Dataset/benchmark tooling | Research prototype | Research code | **Implemented:** exact decimal/rational decisions, strict JSON, hashes, SPDX, license gate, offline CI |

## Defensible tierroute claims today

The following are current engineering differentiators, not claims that tierroute
invented cost-aware model routing:

1. A specification-independent typed boundary maps prompt, tier, remaining budget,
   candidate metadata, and call history to either a model call or prior-output
   selection. Unconfirmed challenge details stay in adapters.
2. The same exact rational `predicted_quality - lambda * cost` decision primitive is
   used for fitting evidence and runtime routing, while `Decimal` ledgers prevent
   context-dependent budget rounding.
3. Each tier's lambda is selected by replaying its configured budget ledger, and
   positive independent tier weights summarize the selected per-tier reports. An
   exhaustive candidate set is optimal only inside the deterministic configured class;
   the bounded default can be approximate. The bundled weights are illustrative until
   SK Telecom publishes the official values. Feasibility is established on replayed
   outcomes; exact arithmetic detects an overspend but cannot guarantee an unseen
   realized charge from a quoted-affordable call. Every executed logged call preserves
   its quote, realized charge, adapter balance snapshots, and ledger result; exact
   tier-level diagnostics retain both absolute and net quote error.
4. The learned-policy evaluator is nested LODO: no outer domain reaches predictor
   fitting, isotonic calibration, or lambda selection. Cumulative results are replayed
   once in original order rather than resetting the ledger at fold boundaries.
5. The clone-first demo runs all six required baselines on one original-order outer-LODO
   population under illustrative per-query accounting. Each domain table sees only its
   outer training rows and pre-call observable domain tags; fold evidence records the
   exact train/test IDs. The ledger used by every replay is behaviorally checked for
   per-query reset and accounting. The oracle remains a per-query upper bound, so a
   cumulative sequence-level oracle is still planned.
6. Runtime inference, the bundled demo, and CI operate without network access. External
   data and model assets use explicit preparation boundaries, fixed identities, and
   license review rather than implicit downloads.
7. Predictor and policy artifacts bind training/replay content and relevant
   configuration to canonical, strictly validated provenance.
8. A bounded standard-library prepared reference reuses each canonical training
   subset once, constructs centered ridge equations from Welford/Chan moments, shares
   one Cholesky factor across model targets, and emits every graph-ordered raw-score
   block. On the seven-domain synthetic regression it proves the exact 63-subset,
   154-block, `22N` row-membership, and `22NM` scalar-score structure. This is an
   auditable engineering reference, not a claim that moment-based ridge, factor reuse,
   or batched scoring is novel.
9. A bounded policy bridge derives the exact prepared calibration/destination graph and
   reuses the existing lambda tuner and simulator. Four- and seven-domain frozen
   fixtures match the complete rowwise nested result, with intermediate-domain and
   near-tie regressions as additional boundaries. Cost-width-aware admission counts all
   current pair traversals and bounds the aggregate lambda candidate/policy-artifact
   estimate. This is engineering parity evidence, not a claim that isotonic calibration,
   Lagrangian routing, or nested evaluation is novel.

## Claims tierroute must not make

- Lambda routing, bilinear model-prompt scoring, isotonic calibration, offline replay,
  and oracle comparison are not new inventions of tierroute.
- Prompt-difficulty routing, best-of-n allocation, SLA/tier-conditioned Lagrangian
  control, confidence-based escalation, benchmark adapters, and unseen-model bilinear
  routing all have direct prior art in the broader sources above.
- No RouterBench, SK Telecom, or real-model score has been produced by the bundled
  synthetic demo.
- The RouteLLM evidence does not show universal OOD collapse.
- Beating individual models is not enough to claim useful learned routing when a
  prompt-independent convex-hull mixture such as RouterBench Zero can match it under
  legal expected-cost semantics.
- Cascading is neither implemented nor proven inferior. A history field in the API is
  not evidence of a history-adaptive policy.
- The local tier weights are not official, and current optimization must be described
  as challenge-aligned rather than the final official scoring function.
- A local bge-m3 inference provider, full-dimensional bge-m3 training run, reportable
  licensed-data family-selection experiment, OOD fallback, and online remaining-budget
  adaptation are not complete.
- The project-owned C11 sidecar implements one bounded dense ridge solve; it does not
  remove repeated feature work, nested-fold factorizations, or scoring from the complete
  validation graph and is not evidence that a full-dimensional run has completed.
- A separate experimental `TRPSTO01`/`TRPSES01` path now authenticates a file-backed
  prepared store and executes the complete canonical solve-and-score graph in one C11
  child. Small surface-only D4–D7 fixtures match the Python reference within tolerance,
  and one D4/N8/d1036/M1 synthetic fixture exercises 12 surface plus 1,024 embedding
  columns without projection. This is implementation evidence, not a full-shape,
  throughput, peak-memory, or speedup result.
- The prepared references establish bounded canonical rows, caller-checked content
  identities, domain moments, subset isolation, moment-ridge coefficients, and complete
  raw-score plus calibration/lambda/final-report wiring on synthetic/frozen fixtures.
  Stable four- and seven-domain corpora match the rowwise nested result, while raw
  numerical agreement remains tolerance-based rather than bitwise or a cross-platform
  digest promise. They do not authenticate dataset/model provenance, run an embedding
  provider or all-domain artifact, connect the native result to the prepared policy
  bridge/six-baseline report, execute the planned full RouterBench shape, or prove
  throughput, peak memory, quality, cost savings, or a speedup. The official-size D7
  tuple has exact aggregate admission evidence only, not a completed execution. Their
  resource estimates also do not bound the work or side effects of an arbitrary
  caller-supplied ledger callback. Issue #9 remains open.
- Prepared content digests are deterministic content identities, not authenticity or
  provenance proofs. Supported builders enforce derivation/topology associations;
  direct leaf constructors validate only self-declared canonical record content.
  Substitution detection requires a separately trusted expected digest. Global bundle
  digest locality is not promised when an excluded domain changes; only the individual
  unaffected coefficient/raw blocks have locality evidence.
- The GBM core has no versioned artifact or deployment CLI integration. Its paired
  estimation CLI deliberately cannot select a winner from the same outer evidence and
  supplies no evidence of predictive gain or superiority over the bilinear family.
- bge-m3 is a planned controlled feature ablation, not the core novelty or an assured
  performance gain. More candidate models likewise do not guarantee a better frontier;
  model-subset curation must be measured.
- Paper-reported savings or quality improvements are not tierroute results.
- “Exact” means exact utility and, when requested and preflight permits it, complete
  enumeration of tierroute's deterministic lambda candidates under the configured
  replay. It is not a claim of global optimality over randomized expected-budget
  policies; the unified formulation can require a randomized mixing coefficient.
- The per-query six-baseline comparison and oracle are not valid cumulative-stream
  evidence. No cumulative oracle-gap claim is permitted without a sequence-level plan.
- The cross-tier quote-error total is a diagnostic over independent tier replays, not a
  shared budget or proof of official budget compliance.

## Decision record and falsification plan

| Decision | Evidence-based reason | What would change it |
|---|---|---|
| One-shot lambda routing is P0 | It matches established linear cost-quality routing and needs only ex-ante estimates. The same action can be replayed through either current ledger; no cumulative-budget optimality is claimed. | Official sequential-call semantics plus a validated post-hoc quality estimator can unlock a separately evaluated cascade. |
| LODO replaces random split | Domain transfer is the anticipated public-to-private risk, and RouteLLM shows that some transfer settings can approach random routing without in-domain support. | Nothing short of an official split that precludes domain shift; random split may remain a diagnostic but not the reportable result. |
| Supervised full-information learning, not offline RL | The current harness and RouterBench expose every candidate outcome offline, and the task page describes candidate outputs and quality in public data. Direct prediction is appropriate only if the released SK Telecom schema preserves that full-information supervision. | If the released data reveals only bandit feedback, the formulation must change. |
| Latency is secondary | The task page says latency is a tie-break, while quality under weighted budgets is primary. | A later official scoring specification that assigns latency a non-tie score. |

The implemented per-query baseline orchestrator fits each domain table only on its outer
training side, retains decisions only for that fold's test rows, and replays every
method once on the same original-order population and behaviorally verified per-query
ledger. Per-query experiments may report the existing oracle-gap metric. A cumulative
experiment first needs a sequence-level oracle; the independent per-query plan is
neither guaranteed feasible nor an upper bound for a stream. Every reportable JSON
result must show per-tier feasibility, mean quality, executed-call quoted and realized
totals, absolute quote error, and ledger over-budget counts. The cross-tier cost row
must remain labeled as a diagnostic over independent ledgers. Cross-report metrics
must first match the required versioned evaluation-scope identity, whose digest covers the full
ordered replay, tier configuration, call cap, outputs, labels, candidate order, and
canonical immutable router/model metadata. Baseline score and oracle-gap fields are
derived evidence and must be recomputed by the six-report suite rather than trusted
from persisted rows. Planned ablations are
surface-only versus local bge-m3 features, bilinear versus GBM, uncalibrated versus
isotonic, shared versus tier-specific lambda, full versus curated model catalogues, and
LODO versus a clearly
labeled non-reportable random diagnostic. A new claim survives only if it holds on
untouched outer domains and its artifact records the exact data, policy, and environment
identity. No superiority over the broader systems is claimed without an aligned model
pool, data split, cost model, and metric.

## Implementation traceability

| Claim | Repository evidence |
|---|---|
| State/action boundary and validation | [`core/schemas.py`](../src/tierroute/core/schemas.py), [`core/router.py`](../src/tierroute/core/router.py) |
| Per-query and cumulative budget uncertainty | [`adapters/budgets.py`](../src/tierroute/adapters/budgets.py) |
| Exact one-shot tier policy | [`policies/lambda_threshold.py`](../src/tierroute/policies/lambda_threshold.py) |
| Cross-fitted tuning and nested LODO | [`policies/lambda_tuning.py`](../src/tierroute/policies/lambda_tuning.py), [`eval/validation.py`](../src/tierroute/eval/validation.py) |
| Per-query outer-LODO six-baseline comparison | [`policies/baseline_evaluation.py`](../src/tierroute/policies/baseline_evaluation.py), [`eval/planning.py`](../src/tierroute/eval/planning.py) |
| Tier metric and oracle-gap recovery | [`eval/metrics.py`](../src/tierroute/eval/metrics.py) |
| Executed-call quote/realized evidence | [`eval/schemas.py`](../src/tierroute/eval/schemas.py), [`eval/simulator.py`](../src/tierroute/eval/simulator.py) |
| Exact quote-error aggregation and CLI report | [`eval/metrics.py`](../src/tierroute/eval/metrics.py), [`cli.py`](../src/tierroute/cli.py) |
| Complete evaluation-scope identity and immutable metadata snapshot | [`evaluation-scope.md`](evaluation-scope.md), [`eval/provenance.py`](../src/tierroute/eval/provenance.py), [`eval/simulator.py`](../src/tierroute/eval/simulator.py) |
| Bilinear fit and isotonic calibration | [`predictors/training.py`](../src/tierroute/predictors/training.py), [`predictors/calibration.py`](../src/tierroute/predictors/calibration.py) |
| In-memory deterministic GBM and inner-LODO calibration | [`predictors/gbm.py`](../src/tierroute/predictors/gbm.py), [`predictors/gbm_training.py`](../src/tierroute/predictors/gbm_training.py) |
| Paired bilinear/GBM nested-LODO estimation with no family selection | [`policies/predictor_comparison.py`](../src/tierroute/policies/predictor_comparison.py), [`policies/benchmark.py`](../src/tierroute/policies/benchmark.py), [`test_predictor_comparison.py`](../tests/test_predictor_comparison.py) |
| Local embedding identity, provider still absent | [`features/embeddings.py`](../src/tierroute/features/embeddings.py) |
| Prepared graph and bounded feature/statistics isolation reference | [`predictors/prepared_graph.py`](../src/tierroute/predictors/prepared_graph.py), [`predictors/prepared_store.py`](../src/tierroute/predictors/prepared_store.py), [`prepared-feature-store.md`](prepared-feature-store.md), [`test_prepared_store.py`](../tests/test_prepared_store.py) |
| Bounded prepared moment-ridge and complete raw-score reference | [`predictors/prepared_execution.py`](../src/tierroute/predictors/prepared_execution.py), [`prepared-reference-execution.md`](prepared-reference-execution.md), [`test_prepared_execution.py`](../tests/test_prepared_execution.py) |
| Bounded prepared calibration/lambda/final-report bridge | [`policies/prepared_reference.py`](../src/tierroute/policies/prepared_reference.py), [`prepared-reference-pipeline.md`](prepared-reference-pipeline.md), [`test_prepared_reference_pipeline.py`](../tests/test_prepared_reference_pipeline.py) |
| Optional RouterBench boundary | [`adapters/routerbench.py`](../src/tierroute/adapters/routerbench.py), [`download_routerbench.py`](../scripts/download_routerbench.py) |
| Reproducibility and license inventory | [`SBOM.md`](../SBOM.md), [`dependency-license-audit.md`](dependency-license-audit.md) |

## Reportability backlog

These are implementation tasks, not questions for the organizer:

1. Add a sequence-level oracle before computing oracle-gap recovery under cumulative
   accounting. The independent per-query plan is valid only for per-query budgets.
2. Replace the bounded Python prepared reference with an audited scalable session only
   after an offline local provider exists, then promote frozen coefficient-to-
   calibration/lambda/final-report parity to the official shape, broader near ties,
   and all-domain artifact assembly. The current 100,000,000
   work-unit and 512 MiB modeled numeric-storage admission ceilings intentionally reject
   the planned RouterBench shape; they count reviewed numeric operations/storage, not
   Python-object or allocator overhead, caller-owned memory, peak RSS, wall-clock, or
   speedup. The residual factor 2,048 is an empirical regression guard, not a universal
   error bound.

Quote-versus-realized error reporting is implemented over executed logged replay calls.
It preserves offsetting call errors through an absolute-error total and separately
reports exact net direction and magnitude. This is retrospective evidence: runtime
affordability still uses a quote and cannot promise ex-ante feasibility when an unseen
realized charge differs.

## Open evidence gates

The following remain official-answer or compliance gates:

1. Whether a budget resets per query or applies to an ordered stream.
2. Whether sequential calls are permitted and which call-history fields are visible.
3. If sequential calls are legal, whether the final action returns the stopping output
   or may select any earlier called output.
4. The exact official tier weights, cost units, hidden-data schema, and scoring details.
5. The SK Telecom dataset license and redistribution permission.
6. Whether randomized expected-cost mixtures are legal, which determines whether the
   RouterBench Zero router is a valid additional baseline.
7. A scalable persistent/native prepared execution session that promotes the bounded
   coefficient/raw-score/calibration/lambda/report reference to official-shape execution
   and an all-domain artifact, plus audited GPL-family-free Linux-musl and Windows-MSVC artifacts, before
   the parity-tested C11 dense solver can support a reportable full-dimensional bge-m3
   experiment.

Until those gates close, new semantics remain in `adapters/`, cascade remains disabled,
SK Telecom data remains outside the repository, and synthetic results remain labeled
as smoke tests.
