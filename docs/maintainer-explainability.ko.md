<!-- SPDX-License-Identifier: Apache-2.0 -->

# 유지관리자 핵심 경계 설명·변이 워크시트

[한국어 README로 돌아가기](../README.ko.md) ·
[영문 검토 패킷](maintainer-explainability.md)

이 문서는 대회 핵심 코드를 참가자가 직접 추적하고 설명하기 위한
실행 워크시트다. 문서를 읽었거나 CI가 성공했다는 사실만으로는 서명할 수
없다. 참가자가 정확한 코드 경로를 읽고, 실패를 먼저 예측하고, 전용 임시
worktree에서 제시된 변이를 한 번씩 수행한 뒤, 복구와 테스트 성공을 직접
확인해야 한다.

이 실습의 기준 소스는 구현 스냅샷 커밋
`c6491508533655baa76c7b50bfdadacbc1612e60`다. 소스가 바뀌면 현재 검토한 정확한
커밋으로 절차와 변이를 다시 감사하고, 서명 표에는 그 커밋을 기록한다.
기준 스냅샷의 성공은 이후 소스 변경을 인증하지 않는다.

## 1. 서명 판정 기준

각 경계는 참가자가 준비된 답을 낭독하지 않고 다음 다섯 가지를 자신의 말로
설명할 수 있을 때만 서명 가능하다.

1. **성공 경로:** 입력에서 소유 심볼·검사를 거쳐 출력·증거가 나오는 순서.
2. **실패 경로:** 실행 전 예상한 가드와 예외·assertion, 위험한 부작용 전에
   실패해야 하는 이유.
3. **신뢰 경계:** 일반 라우터 상태, replay 비공개·특권 증거, adapter 소유 의미,
   보고용 값을 누가 볼 수 있고 어디에 쓸 수 있는지.
4. **가장 강한 테스트:** 정확한 pytest node와 그 node가 잡는 회귀·공격. “CI green”은
   답이 아니다.
5. **의도적 한계:** 현재 주장하지 않는 것과 새 권한·공식 스펙·증거 중 무엇이
   있어야 해제되는지.

또한 변이 전 지정 node 성공, 실행 전 실패 예측, 예상한 변이 실패, 소스
복구, 같은 node 재성공, 빈 `git status`, 빈 `HF_HOME`을 모두 확인해야 한다.
AI 문구와 자동화 성공은 참가자 서명이 아니다.

## 2. 최소 용어집

- **불변식(invariant):** 정상·적대적 경로 모두에서 유지되어야 하는 규칙.
- **fail closed:** 증거가 불확실하거나 불일치하면 추측하지 않고 사용 불가·오류로
  처리하는 규칙.
- **신뢰 경계:** 어떤 주체가 어떤 데이터를 보고 의사결정에 쓸 수 있는지를 나누는 선.
- **locked environment:** `requirements-dev.lock`의 정확한 의존성 버전. 설치 과정까지
  air-gapped라는 주장은 아니다.
- **throwaway worktree:** 유지관리자의 현재 편집과 분리한 detached 임시 checkout.
- **변이 실습:** 테스트를 약화하지 않고, 제품 코드의 불변식을 한 번 임시로
  깨뜨리는 실험.
- **outer LODO:** 하나의 domain 전체를 fitting·tuning에서 제외한 외부 검증.
- **inner LODO:** outer-training 안에서만 OOF 예측·보정·tuning을 하는 내부 검증.
- **evaluation scope:** 순서가 있는 replay와 call cap을 결합한 버전화 식별자.
  불일치 감지용이지 서명·인증이 아니다.
- **privileged oracle:** 비공개 label을 볼 수 있는 오프라인 상한 라우터. 런타임 정책이
  아니다.
- **quoted/realized cost:** 호출 전 예산 가능성을 판단하는 견적 / replay에서 실제
  차감하는 비용.
- **oracle-gap recovery:** cheapest에서 oracle까지의 간격 중 회수한 비율. 현재
  tier별 값이 아닌 하나의 전체 집계값이며, 쿼리별 예산 범위에서만 보고한다.
- **showcase:** 합성 3개 항목의 직접 replay와 별도 benchmark 증거. 공식 점수·공유
  누적 예산이 아니다.

## 3. 전용 worktree와 잠긴 개발 환경

다음 블록은 원래 작업 트리가 깨끗한지 먼저 확인하고, 변이 전용 worktree와
venv, 빈 Hugging Face cache, 테스트 임시 디렉터리를 만든다. 현재 작업이 있으면
첫 번째 `test` 명령에서 중단한다.

```bash
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"
review_commit="c6491508533655baa76c7b50bfdadacbc1612e60"
git -C "$repo" cat-file -e "$review_commit^{commit}"

review_root="$(mktemp -d "${TMPDIR:-/tmp}/tierroute-explain.XXXXXX")"
worktree="$review_root/worktree"
venv="$review_root/venv"
hf_home="$review_root/hf-home"
mkdir -p "$review_root/tmp" "$hf_home"

git -C "$repo" worktree add --detach "$worktree" "$review_commit"
python3.12 -m venv "$venv"
make -C "$worktree" install-dev PYTHON="$venv/bin/python"

export PATH="$venv/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export HF_HOME="$hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$review_root/tmp"
cd "$worktree"
test "$(git rev-parse HEAD)" = "$review_commit"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
python --version
python -m pip check

expect_pytest_failure() {
  local label="$1"
  local expected="$2"
  shift 2
  local log="$review_root/$label.log"
  local pytest_status
  if "$@" >"$log" 2>&1; then
    echo "ERROR: $label mutation unexpectedly passed; inspect $log" >&2
    return 1
  else
    pytest_status=$?
  fi
  if [ "$pytest_status" -ne 1 ]; then
    echo "ERROR: $label exited $pytest_status, not pytest assertion-failure status 1" >&2
    return 1
  fi
  grep -Fq -- "$expected" "$log"
  printf '%s expected failure observed (exit %s; log %s)\n' \
    "$label" "$pytest_status" "$log"
}
```

`install-dev`는 엄격히 잠긴 wheel을 로컬 cache에서 찾지 못하면 네트워크로 받을 수
있다. 이는 준비 단계이며 런타임 오프라인 주장과 분리해 기록한다. 설치 후
모든 아래 명령은 위 오프라인 환경변수를 유지한다. 전체 런타임의 socket
차단 증거는 경계 8의 전용 테스트와 CI로 확인한다.

각 실습에서는 먼저 같은 테스트 명령이 PASS하는지 확인한다. 제시된 patch만
적용하고, 실행 전에 실패 메시지를 자신의 말로 기록한다. 변이 실행 후에는
각 카드가 저장한 patch를 `git apply -R`로 정확히 되돌린다. `git reset --hard`,
`git clean -fdx`, 유지관리자 작업 트리에서의 `git restore`는 쓰지 않는다.
`set -euo pipefail`은 사전 테스트·patch·복구·정리 검사 중 하나라도 실패하면 즉시
중단시킨다. 예상된 pytest 실패만 `expect_pytest_failure`가 별도 log에 캡처하고,
비정상 종료와 예상 메시지를 둘 다 확인한 후 진행한다.
설정 블록과 카드는 한 전용 shell에서 순서대로 실행해 안전 옵션·함수·환경을
계속 유지한다.

## 4. 여덟 개 경계 변이 실습

### 카드 1 — Router 계약과 정확 비용

**불변식.** `validate_action`은 목록에 있는 모델과 유효한 기존 출력만 허용하며,
견적 비용이 잔여 예산과 **같은** 호출은 가능해야 한다. 예산 경계는 float가 아닌
제품 소유 `Decimal` 연산으로 판단한다.

- 소유 심볼: [`validate_action`](../src/tierroute/core/router.py),
  [`as_cost`](../src/tierroute/core/schemas.py)
- 임시 변이: 잔여 예산과 같은 비용도 거부하도록 `>`를 `>=`로 변경한다.

```bash
file="src/tierroute/core/router.py"
test -z "$(git diff --name-only)"
python -m pytest -q tests/test_core.py::test_validate_action_accepts_affordable_call_and_existing_output
git apply <<'PATCH'
diff --git a/src/tierroute/core/router.py b/src/tierroute/core/router.py
--- a/src/tierroute/core/router.py
+++ b/src/tierroute/core/router.py
@@ -29,7 +29,7 @@ def validate_action(state: RouterState, action: RouterAction) -> None:
         if action.model_id not in candidates:
             raise RoutingContractError(f"unknown candidate model: {action.model_id}")
         cost = candidates[action.model_id].cost
-        if cost > state.remaining_budget:
+        if cost >= state.remaining_budget:
             raise RoutingContractError(
                 f"model {action.model_id!r} costs {cost:g}, "
                 f"but only {state.remaining_budget:g} remains"
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-01.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-01 "model 'large' costs 2, but only 2 remains" \
  python -m pytest -q \
  tests/test_core.py::test_validate_action_accepts_affordable_call_and_existing_output
```

**예상 실패.** 비용 2와 예산 2인 정상 호출이
`RoutingContractError: model 'large' costs 2, but only 2 remains`로 거부되고 pytest가
비정상 종료해야 한다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q tests/test_core.py::test_validate_action_accepts_affordable_call_and_existing_output
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
```

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

### 카드 2 — 예산 adapter, replay, 실행된 call 증거

**불변식.** 시뮬레이터는 live call을 하지 않고 로그를 replay한다. 실현 비용
초과로 ledger가 실패해도 실제 시도·차감된 call은 `QueryResult.calls`와
`BudgetReport.spent`에 남아야 한다. 쿼리별·누적 범위는 adapter가 소유한다.

- 소유 심볼: [`OfflineSimulator._failed_query`](../src/tierroute/eval/simulator.py),
  [`QueryResult`](../src/tierroute/eval/schemas.py)
- 임시 변이: 실패 결과에서 이미 replay한 call 목록을 빈 tuple로 버린다.

```bash
file="src/tierroute/eval/simulator.py"
test -z "$(git diff --name-only)"
python -m pytest -q tests/test_simulator.py::test_realized_overspend_is_recorded_and_exhausts_cumulative_budget
git apply <<'PATCH'
diff --git a/src/tierroute/eval/simulator.py b/src/tierroute/eval/simulator.py
--- a/src/tierroute/eval/simulator.py
+++ b/src/tierroute/eval/simulator.py
@@ -278,4 +278,4 @@ class OfflineSimulator:
             decision_reason=" -> ".join(trace),
             error=error,
-            calls=tuple(replayed_calls),
+            calls=(),
         )
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-02.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-02 "query cost must equal the exact sum of replayed call charges" \
  python -m pytest -q \
  tests/test_simulator.py::test_realized_overspend_is_recorded_and_exhausts_cumulative_budget
```

**예상 실패.** `QueryResult.__post_init__`이 차감 비용과 call 합이 다른 결과를
`ValueError: query cost must equal the exact sum of replayed call charges`로 fail closed하고
pytest가 비정상 종료해야 한다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q tests/test_simulator.py::test_realized_overspend_is_recorded_and_exhausts_cumulative_budget
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
```

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

### 카드 3 — 완전한 evaluation-scope 식별자

**불변식.** 서로 비교할 보고서는 알고리즘, 순서가 있는 전체 replay, tier 정의,
call cap이 같아야 한다. 버전화 digest는 이 범위를 결합하지만, 외부 신뢰 기준이
없으므로 인증 서명은 아니다.

- 소유 심볼: [`_write_evaluation_scope`](../src/tierroute/eval/provenance.py),
  [`EvaluationScopeIdentity`](../src/tierroute/eval/schemas.py)
- 임시 변이: 실제 `max_calls_per_query`를 해시하지 않고 항상 1을 기록한다.

```bash
file="src/tierroute/eval/provenance.py"
test -z "$(git diff --name-only)"
python -m pytest -q tests/test_eval_provenance.py::test_evaluation_scope_covers_tier_order_weight_budget_and_call_cap
git apply <<'PATCH'
diff --git a/src/tierroute/eval/provenance.py b/src/tierroute/eval/provenance.py
--- a/src/tierroute/eval/provenance.py
+++ b/src/tierroute/eval/provenance.py
@@ -543,4 +543,4 @@ def _write_evaluation_scope(
 ) -> None:
     writer.token(b"algorithm", EVALUATION_SCOPE_ALGORITHM.encode("ascii"))
-    writer.count(b"max-calls-per-query", max_calls_per_query)
+    writer.count(b"max-calls-per-query", 1)
     writer.count(b"tier-specs", len(tier_specs))
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-03.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-03 \
  "fde4ac2af181ca623238807f33124ab74b38027184e7f5051b61b056276c5aa2" \
  python -m pytest -q \
  tests/test_eval_provenance.py::test_evaluation_scope_covers_tier_order_weight_budget_and_call_cap
```

**예상 실패.** cap 1과 cap 2가 같은 digest를 만들어 “같지 않아야 한다”는
최종 assertion이 실패하고 pytest가 비정상 종료해야 한다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q tests/test_eval_provenance.py::test_evaluation_scope_covers_tier_order_weight_budget_and_call_cap
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
```

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

### 카드 4 — 지표, nested LODO 6종 비교, showcase

**불변식.** 가중 품질은 각 tier의 명시적 weight를 적용하고 불가능 tier의
weight를 재분배하지 않아야 한다. 예측기·보정·lambda·domain table은 outer training
안에서만 학습하며, 학습 라우터와 6종 기준선은 같은 순서·tier·쿼리별 ledger·
scope에서 replay된다. 3-step showcase는 해당 outer-fold 라우터의 직접 replay와
nested 결과가 같아야 하지만, 공식 점수·공유 누적 예산은 아니다.

- 소유 심볼: [`summarize_report`](../src/tierroute/eval/metrics.py),
  [`evaluate_per_query_lodo_baselines`](../src/tierroute/policies/baseline_evaluation.py),
  [`evaluate_per_query_bilinear_benchmark`](../src/tierroute/policies/benchmark.py),
  [`build_routing_stream_showcase`](../src/tierroute/showcase.py)
- 임시 변이: tier weight 곱을 제거하고 품질을 분자에 그대로 더한다.

```bash
file="src/tierroute/eval/metrics.py"
test -z "$(git diff --name-only)"
python -m pytest -q \
  tests/test_metrics.py::test_weighted_tier_quality_uses_explicit_weights \
  tests/test_baseline_evaluation.py::test_six_baselines_share_one_original_order_outer_lodo_population \
  tests/test_benchmark.py::test_benchmark_aligns_learned_router_and_six_baselines \
  tests/test_showcase.py::test_bundled_showcase_stream_is_stable_and_covers_every_tier \
  tests/test_cli.py::test_demo_json_is_deterministic_versioned_and_keeps_scopes_separate
git apply <<'PATCH'
diff --git a/src/tierroute/eval/metrics.py b/src/tierroute/eval/metrics.py
--- a/src/tierroute/eval/metrics.py
+++ b/src/tierroute/eval/metrics.py
@@ -393,6 +393,6 @@ def summarize_report(report: EvaluationReport) -> ScoreSummary:
         tier_quality[tier] = quality
         complete = complete and quality is not None
         if quality is not None:
-            contribution = result.tier_spec.weight * quality
+            contribution = quality
             if not math.isfinite(contribution):
                 raise ValueError("weighted quality contribution must be finite")
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-04.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-04 "3 failed, 2 passed" python -m pytest -q \
  tests/test_metrics.py::test_weighted_tier_quality_uses_explicit_weights \
  tests/test_baseline_evaluation.py::test_six_baselines_share_one_original_order_outer_lodo_population \
  tests/test_benchmark.py::test_benchmark_aligns_learned_router_and_six_baselines \
  tests/test_showcase.py::test_bundled_showcase_stream_is_stable_and_covers_every_tier \
  tests/test_cli.py::test_demo_json_is_deterministic_versioned_and_keeps_scopes_separate
```

**예상 실패.** metric node의 가중 품질이 `0.72` 대신 `2.3`이 되고,
benchmark node는 `0.73125` 대신 `2.34375`를 얻어 FAIL해야 한다. demo JSON
node도 내부 benchmark 증거가 바뀌어 FAIL해야 한다. 구조만 검사하는
baseline과 직접 showcase node는 PASS해야 한다. 서로 다른 하위 경계를 감시하는
테스트임을 설명해야 한다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q \
  tests/test_metrics.py::test_weighted_tier_quality_uses_explicit_weights \
  tests/test_baseline_evaluation.py::test_six_baselines_share_one_original_order_outer_lodo_population \
  tests/test_benchmark.py::test_benchmark_aligns_learned_router_and_six_baselines \
  tests/test_showcase.py::test_bundled_showcase_stream_is_stable_and_covers_every_tier \
  tests/test_cli.py::test_demo_json_is_deterministic_versioned_and_keeps_scopes_separate
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
```

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

### 카드 5 — fitted feature, ridge/GBM 품질 예측, calibration

**불변식.** feature schema·scaling·tag vocabulary, ridge·GBM stump, isotonic
calibration은 outer training에서만 fit한다. 보정은 inner-LODO OOF 예측을 쓰고,
예측기는 outer training 전체로 다시 fit한다. bilinear artifact v1만 엄격한 직렬화
계약을 가지며 GBM은 인메모리 전용이다. 현재 embedding 경계는 provider 주입을
지원하지만 `bge-m3` 가중치·추론 provider를 배포한다고 주장하지 않는다.

- 소유 심볼: [`fit_calibrated_bilinear_for_fold`](../src/tierroute/predictors/training.py),
  [`fit_calibrated_gbm_for_fold`](../src/tierroute/predictors/gbm_training.py),
  [`RegressionStump`](../src/tierroute/predictors/gbm.py),
  [`PromptFeatureSchema`](../src/tierroute/features/encoding.py),
  [`IsotonicCalibrator`](../src/tierroute/predictors/calibration.py)
- 임시 변이: outer 학습에 held-out test 행을 합쳐 embedding provider에 노출한다.

변이 전에는 residual 갱신·split gain·feature/split 동률 규칙·양의 gain이 없을 때의
조기 종료를 직접 유도하고, 모든 inner/final 작업량 검사가 첫 embedding 호출보다
앞서는 이유를 설명한다. 아래 변이는 ridge와 GBM이 공유해야 하는 outer-fold 격리
불변식을 점검한다.

```bash
file="src/tierroute/predictors/training.py"
test -z "$(git diff --name-only)"
python -m pytest -q tests/test_bilinear_training.py::test_outer_fold_training_never_observes_held_out_examples
python -m pytest -q tests/test_gbm_core.py tests/test_gbm_training.py
git apply <<'PATCH'
diff --git a/src/tierroute/predictors/training.py b/src/tierroute/predictors/training.py
--- a/src/tierroute/predictors/training.py
+++ b/src/tierroute/predictors/training.py
@@ -200,6 +200,6 @@ def fit_calibrated_bilinear_for_fold(
     """Fit exclusively on an outer LODO fold's training side."""
 
     return fit_calibrated_bilinear(
-        fold.training,
+        fold.training + fold.test,
         config=config,
         embedding_provider=embedding_provider,
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-05.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-05 "held_out_prompts.isdisjoint(observed_prompts)" \
  python -m pytest -q \
  tests/test_bilinear_training.py::test_outer_fold_training_never_observes_held_out_examples
```

**예상 실패.** 주입된 provider가 held-out prompt를 관찰하여
`held_out_prompts.isdisjoint(observed_prompts)` assertion이 실패하고 pytest가 비정상
종료해야 한다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q tests/test_bilinear_training.py::test_outer_fold_training_never_observes_held_out_examples
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
```

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

### 카드 6 — 정확 lambda routing·tuning·policy artifact

**불변식.** 런타임과 tuning은 동일한
`predicted_quality - lambda * quoted_cost` 선택 함수와 정확한 유리수 lambda,
결정적 tie break를 쓴다. policy artifact는 predictor 바이트·학습 데이터·순서·tier·
ledger·탐색 증거에 결합되어야 한다. 제한된 후보 탐색은 exhaustive로 표현하지
않는다.

- 소유 심볼: [`route_from_predictions`](../src/tierroute/policies/lambda_threshold.py),
  [`tune_tier_lambdas`](../src/tierroute/policies/lambda_tuning.py),
  [`predictor_artifact_sha256`](../src/tierroute/policies/lambda_artifacts.py)
- 임시 변이: 실제 predictor canonical JSON 대신 항상 같은 64자 0 digest를 반환한다.

```bash
file="src/tierroute/policies/lambda_artifacts.py"
test -z "$(git diff --name-only)"
python -m pytest -q tests/test_lambda_policy_artifacts.py::test_predictor_hash_and_training_metadata_mismatches_fail_closed
git apply <<'PATCH'
diff --git a/src/tierroute/policies/lambda_artifacts.py b/src/tierroute/policies/lambda_artifacts.py
--- a/src/tierroute/policies/lambda_artifacts.py
+++ b/src/tierroute/policies/lambda_artifacts.py
@@ -50,7 +50,7 @@ def predictor_artifact_sha256(artifact: BilinearPredictorArtifact) -> str:
         document = artifact.to_json().encode("utf-8")
     except UnicodeEncodeError as error:
         raise ValueError("predictor artifact contains invalid Unicode text") from error
-    return hashlib.sha256(document).hexdigest()
+    return "0" * 64
 
 
 def _strict_fields(payload: Mapping[str, object], expected: set[str], context: str) -> None:
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-06.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-06 "DID NOT RAISE <class 'ValueError'>" \
  python -m pytest -q \
  tests/test_lambda_policy_artifacts.py::test_predictor_hash_and_training_metadata_mismatches_fail_closed
```

**예상 실패.** bias가 바뀐 predictor를 artifact가 구분하지 못해 해당 가드의
`Failed: DID NOT RAISE <class 'ValueError'>`로 pytest가 비정상 종료해야 한다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q tests/test_lambda_policy_artifacts.py::test_predictor_hash_and_training_metadata_mismatches_fail_closed
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
```

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

### 카드 7 — RouterBench 적대적 데이터·provenance 경계

**불변식.** RouterBench는 선택적·미커밋·`NOASSERTION`이다. 사용자가 준비를
명시적으로 실행한 때만 pinned 바이트를 받고, 크기와 SHA-256을 확인한 다음
dispatch하지 않는 전용 VM이 허용된 그래프 모양만 inert node로 해석한다. 런타임은
`pickle.load`, pandas, payload가 지정한 callable을 호출하지 않는다. 로컬 diagnostic은
공식 SKT 데이터·성능 점수·보고 가능 결과가 아니다.

- 소유 심볼: [`_decode_pickle_graph`](../src/tierroute/adapters/routerbench.py),
  [`_scan_balanced_routerbench_split`](../scripts/validate_routerbench.py)
- 임시 변이: 허용 목록에 없는 `os.system`을 하나의 예외로 통과시킨다. 전용
  VM은 여전히 inert node만 만들며 이 실습은 `pickle.loads`를 쓰지 않는다.

```bash
file="src/tierroute/adapters/routerbench.py"
test -z "$(git diff --name-only)"
python -m pytest -q tests/test_routerbench_adapter.py::test_opcode_vm_rejects_forbidden_global_without_invoking_it
git apply <<'PATCH'
diff --git a/src/tierroute/adapters/routerbench.py b/src/tierroute/adapters/routerbench.py
--- a/src/tierroute/adapters/routerbench.py
+++ b/src/tierroute/adapters/routerbench.py
@@ -499,7 +499,7 @@ def _decode_pickle_graph(payload: bytes) -> object:
                         f"RouterBench STACK_GLOBAL at byte {position} needs string names"
                     )
                 global_name = (module, name)
-                if global_name not in _ALLOWED_GLOBALS:
+                if global_name not in _ALLOWED_GLOBALS and global_name != ("os", "system"):
                     raise RouterBenchSchemaError(
                         "RouterBench pickle references forbidden global "
                         f"{module}.{name} at byte {position}"
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-07.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-07 "DID NOT RAISE" \
  python -m pytest -q \
  tests/test_routerbench_adapter.py::test_opcode_vm_rejects_forbidden_global_without_invoking_it
```

**예상 실패.** 금지 global이 거부되지 않아
`Failed: DID NOT RAISE RouterBenchSchemaError`로 pytest가 비정상 종료해야 한다.
테스트가 monkeypatch한 `os.system`은 **호출되지 않아야** 한다. 이는 allowlist
회귀를 검사하는 실습이지 코드 실행 실험이 아니다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q \
  tests/test_routerbench_adapter.py::test_opcode_vm_rejects_forbidden_global_without_invoking_it \
  tests/test_validate_routerbench_script.py::test_balanced_membership_ignores_prompt_quality_cost_and_response_mutations \
  tests/test_validate_routerbench_script.py::test_evaluation_quote_overrun_aborts_before_predictor_fit \
  tests/test_validate_routerbench_script.py::test_validate_nested_lodo_synthetic_end_to_end_uses_real_benchmark \
  tests/test_validate_routerbench_script.py::test_safe_json_contains_provenance_labels_but_no_private_rows_or_metrics
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
```

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

### 카드 8 — atomic I/O, 오프라인, 빌드, 라이선스

**불변식.** predictor·policy 문서는 독점적 임시 일반 파일에 staging하고 검증한
뒤 policy를 마지막에 게시한다. rename 시도는 비동기 예외가 어느 시점에 들어와도
복구할 수 있게 **rename 전**에 기록해야 한다. 런타임·학습·평가는 다운로드하지
않고, release wheel은 제3자 런타임 의존성이 없으며, GPL 계열 증거는 fail closed한다.

- 소유 심볼: [`replace_text_bundle`](../src/tierroute/core/atomic_io.py),
  [`check_licenses.py`](../scripts/check_licenses.py), [CI](../.github/workflows/ci.yml)
- 임시 변이: `attempted.append(index)`를 `os.replace`의 뒤로 옮겨 rename 성공과 시도
  기록 사이의 빈틈을 만든다.

```bash
file="src/tierroute/core/atomic_io.py"
test -z "$(git diff --name-only)"
python -m pytest -q tests/test_atomic_io.py::test_async_exception_after_policy_rename_restores_the_old_pair
git apply <<'PATCH'
diff --git a/src/tierroute/core/atomic_io.py b/src/tierroute/core/atomic_io.py
--- a/src/tierroute/core/atomic_io.py
+++ b/src/tierroute/core/atomic_io.py
@@ -372,6 +372,6 @@ def replace_text_bundle(entries: Sequence[TextBundleEntry]) -> None:
             # Record intent first: an asynchronous exception may arrive after rename
             # succeeds but before ``os.replace`` returns to Python.
-            attempted.append(index)
             os.replace(stage, destination)
+            attempted.append(index)
             stages[index] = None
         _fsync_directories(destinations)
PATCH
git diff --check
test "$(git diff --name-only)" = "$file"
mutation_patch="$review_root/card-08.patch"
git diff --binary -- "$file" > "$mutation_patch"
expect_pytest_failure card-08 "old policy" \
  python -m pytest -q \
  tests/test_atomic_io.py::test_async_exception_after_policy_rename_restores_the_old_pair
```

**예상 실패.** 주입된 `KeyboardInterrupt`가 policy rename 후, 시도 기록 전에 들어와
predictor는 되돌리지만 policy는 `{"new":2}\n`으로 남는다. `old policy`를 기대하는
assertion이 실패해야 한다. 테스트 파일은 pytest 전용 임시 디렉터리에만 있다.

```bash
git apply --check -R "$mutation_patch"
git apply -R "$mutation_patch"
python -m pytest -q \
  tests/test_atomic_io.py \
  tests/test_offline_runtime.py::test_runtime_commands_never_open_a_socket \
  tests/test_license_gate.py::test_deep_scan_rejects_vendored_lgpl_metadata \
  tests/test_license_gate.py::test_deep_scan_rejects_aggregated_gpl_license_text \
  tests/test_package.py::test_runtime_and_training_add_no_distribution_requirement \
  tests/test_reproduction_contract.py::test_training_reproduction_preserves_complete_locked_pipeline
python scripts/check_licenses.py
git diff --exit-code -- "$file"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
test -z "$(find "$TMPDIR" -type f \
  \( -name '.*.stage.*.tmp' -o -name '.*.backup.*.tmp' \) -print -quit)"
```

release wheel의 의존성 0개 주장은 위 package node 하나가 아니라 CI의
dependency-free wheel job에서 새 venv에 wheel을 설치한 결과와 함께 판단한다. 검토한
정확한 commit의 CI run URL과 그 job 상태를 참가자 기록에 함께 남긴다.

```text
실행 전 예측:
관찰한 실패:
복구 후 결과:
내 말로 설명한 핵심:
남은 질문:
```

## 5. RouterBench 증거는 두 레코드로 분리한다

### A. 코드·프로젝트 작성 합성 contract

외부 artifact가 없는 상태에서 다음을 실행한다.

```bash
python -m pytest -q -rs \
  tests/test_routerbench_adapter.py \
  tests/test_validate_routerbench_script.py
```

코드·합성 테스트는 PASS하고, pinned 외부 artifact 전용 node는 파일이 없어 SKIP하는
것이 예상된다. 이 증거는 decoder·downloader 가드, 결정적 balanced split, quote
preflight, 합성 real-benchmark 배선, 출력 억제를 검사한다. 실제 99,567,659-byte
artifact, semantic golden, 실제 corpus E2E, 성능을 증명하지 않는다.

```text
evidence_kind:
external_artifact_present:
pinned_artifact_e2e:
exact command/result:
Python/platform:
reviewed commit:
performance evidence:
entrant notes:
```

### B. 선택적 pinned local-artifact E2E

데이터 라이선스는 `NOASSERTION`이므로 선택적 로컬 검증으로만 수행한다. 명시적
준비 단계에서만 네트워크를 쓰고 그 사실을 기록한다.

```bash
make download-routerbench PYTHON=python

python -m pytest -q \
  tests/test_validate_routerbench_script.py::test_local_pinned_artifact_matches_semantic_golden_and_balanced_scope
python scripts/validate_routerbench.py \
  --nested-lodo --acknowledge-noassertion --json
```

`HF_HUB_OFFLINE=1`과 `TRANSFORMERS_OFFLINE=1`은 Hugging Face 자동 다운로드를 금지할
뿐 임의 socket을 운영체제 수준에서 차단하지는 않는다. 별도의 실제 네트워크 차단을
적용·기록하지 않았다면 `독립적 네트워크 차단 미적용; 오프라인 환경변수와
코드 검토만 사용`이라고 정직하게 기록한다. diagnostic JSON의 선언 필드만으로
실제 네트워크 미사용을 추론하지 않는다.

기대하는 고정 식별자는 다음과 같다.

- revision: `784021482c3f320c6619ed4b3bb3b41a21424fcb`
- byte size: `99,567,659`
- byte SHA-256: `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`
- semantic SHA-256: `7b4749ad5c4bdb338c2317b306c382680b1a23dc83c73e29ab805b8f7e472e87`
- 구조: calibration 448행, evaluation 56행, 7 domains, 11 candidate models

실제 출력의 split SHA, commit, Python/platform, 명령, 시각, 준비 네트워크 사용,
검증 네트워크 차단·관찰 방식, PASS/FAIL을 빈 기록지에 직접 적는다. 실제
데이터는 ignored·untracked 로컬 파일로만 두고 커밋하지 않는다. JSON은
저장소 안에 redirect·보존하지 않고, 정말 필요하면 승인된 비공개 외부 위치에만
최소 메타데이터를 보존한다. 행 데이터·
성능·비용·gap 값을 발행하지 않고, 반드시 비-SKT·비공식·비보고용으로 표시한다.

```text
evidence_kind:
reviewed commit:
revision:
byte size/SHA-256:
semantic SHA-256:
emitted split SHA:
rows/domains/models:
Python/platform:
preparation command/network used:
validation command/network denial or observation:
timestamp:
pass/fail:
dataset license:
official SKT data:
competition score:
reportable:
performance metrics published:
entrant notes:
```

## 6. 참가자 기록 템플릿

다음 양식을 각 경계에 복사해 **참가자만** 채운다. 자동화·AI·리뷰어가
소유자, 날짜, 결과, 서명을 추정하거나 대신 채우지 않는다.

```text
경계 번호/제목:
소유자 또는 안정적 Git 식별자:
ISO 날짜:
검토한 정확한 commit:
Python/platform:
변이 전 정확한 명령/결과:
변이 전에 작성한 예측:
정확한 one-line diff:
예상 실패 node/메시지:
관찰한 실패/종료 상태:
성공 경로 설명:
실패 경로 설명:
신뢰 경계 설명:
가장 강한 테스트 설명:
의도적 한계:
복구 명령:
복구 후 정확한 명령/결과:
최종 git status 출력:
최종 HF_HOME 확인:
검토 commit의 dependency-free-wheel CI URL/결과:
참가자 서명/상태:
```

## 7. 사람 소유자 서명

아래 표의 사람 소유 칸은 의도적으로 빈칸이며, 빈칸은 미서명을 뜻한다.
테스트·CI·AI 리뷰만으로 `Complete`로
바꾸지 않는다. 코드 변경으로 해당 불변식이 바뀌면 기존 서명은 무효하며,
참가자가 재검토하고 새 상태를 기록할 때까지 그 행을 비워 둔다.

| 경계 | 소유자 | 날짜 | 검토한 commit | 상태·메모 |
|---|---|---|---|---|
| Router 계약과 정확 비용 |  |  |  |  |
| 예산 adapter, replay, call 증거 |  |  |  |  |
| 완전한 evaluation-scope 식별자 |  |  |  |  |
| 지표, nested LODO 6종 비교, showcase |  |  |  |  |
| feature, ridge/GBM 예측기, calibration |  |  |  |  |
| 정확 lambda tuning과 policy artifact |  |  |  |  |
| RouterBench 적대적 데이터·로컬 diagnostic |  |  |  |  |
| atomic I/O, 오프라인, 빌드, 라이선스 |  |  |  |  |

## 8. 정리

모든 변이를 복구한 뒤, 필요한 최소·비민감 서명 메타데이터만 승인된 비공개
검토 위치로 복사한다. 변이 log·patch·RouterBench 전체 JSON을 저장소에 복사하지
않는다. 먼저 아래 명령으로 tracked·untracked 소스와 **ignored 파일**을 둘 다
검토한다. `git status --porcelain`만으로는 ignored RouterBench artifact·cache·증거를 볼
수 없다.

```bash
cd "$repo"
test -z "$(git -C "$worktree" status --porcelain=v1 --untracked-files=all)"
test -z "$(find "$HF_HOME" -mindepth 1 -print -quit)"
git -C "$worktree" status --short --ignored
printf 'worktree retained for ignored-file inspection: %s\n' "$worktree"
printf 'review temporary files retained for inspection: %s\n' "$review_root"
```

위 절차는 worktree와 `review_root`를 자동 삭제하지 않는다. 현재 Git은
`worktree remove`에 `--force`를 붙이지 않아도 ignored 파일을 함께 삭제할 수 있다.
참가자가 ignored 목록을 직접 읽고, RouterBench artifact를 포함한 필요 증거를 승인된
비공개 위치로 옮겼으며, 나머지 ignored 내용의 삭제를 의도했음을 확인한 후에만
별도로 `git -C "$repo" worktree remove "$worktree"`를 실행한다. 그 다음 남은 전용
venv·log·patch가 더 필요하지 않은지 확인하고 `review_root`만 별도로 정리한다.
