<!-- SPDX-License-Identifier: Apache-2.0 -->

# tierroute

[English](README.md)

`tierroute`는 프롬프트 난이도와 budget tier를 바탕으로 비용 한도 안의 후보
LLM을 선택하는 오프라인 우선 라우터입니다. 기본 정책은 다음 one-shot
Lagrangian 정책입니다.

```text
선택 모델 = argmax_m [예측 품질(prompt, m) - lambda(tier) * 비용(m)]
```

이 프로젝트는 2026 오픈소스 개발자대회 학생부문, SK텔레콤 지정과제
**“Efficient LLM Routing Challenge”** 출품을 위해 개발 중입니다. 현재 pre-alpha로,
라우팅 계약·replay 시뮬레이터·베이스라인 6종·지표·누수 방지 calibrated
bilinear 학습·exact tier-lambda 튜닝·엄격한 예측기/정책 artifact·정확한
견적-실현 비용 오차 지표·외부 데이터가 필요 없는 데모가 구현되어 있습니다.
CLI는 모델을 **선택**할 뿐 실제 LLM을 호출하거나 답변을 생성하지
않습니다.

## 빠른 시작

Python 3.10 이상이 필요합니다. 새로 받은 저장소에서 실행합니다.

```bash
cd tierroute
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

단일 라우팅, 베이스라인 6종 평가, 통합 데모를 차례로 실행합니다.

```bash
tierroute route "루트 2가 무리수임을 증명해 줘." --tier fast
tierroute evaluate
tierroute demo
```

`python -m tierroute`도 같은 진입점입니다. `route`, `evaluate`, `train`에는
`--json`을 쓸 수 있고, 버전이 명시된 호환 replay JSON을 평가 입력으로 지정할 수
있습니다.

```bash
tierroute route "이 Python 함수를 디버그해 줘" --tier balanced --json
tierroute evaluate --data src/tierroute/data/synthetic.json --json
HF_HUB_OFFLINE=1 tierroute demo
```

`route --json`은 실행 전 결정입니다. `cost`는 `quoted_cost`의 의미상 alias로
남고 `realized_cost`는 `null`입니다. `evaluate --json`은 로그된 outcome을
replay하고 실제로 실행된 replay 호출의 tier별·tier 횡단 `cost_evidence`를
출력합니다. 두 명령 모두 실제 provider를 호출하지 않습니다.

동봉된 프롬프트·비용·출력·예측 품질·scorecard 수치는 프로젝트가 만든 **합성
smoke-test 값**입니다. 벤치마크 결과, 실제 모델 비교, 대회 점수가 아닙니다.

### 오프라인 예측기와 정책 학습

학습과 추론에는 제3자 수치 계산 패키지가 필요하지 않습니다. 프로젝트가 작성한
결정론적 centered-ridge Cholesky solver가 모든 모델 target을 하나의 공유
factorization으로 맞추고 intercept에는 ridge penalty를 적용하지 않습니다. 표면
특징 artifact는 엄격한 canonical JSON이며 생성에 쓴 solver ID도 기록합니다.

```bash
tierroute train --output artifacts/synthetic-bilinear.json --json
tierroute route "루트 2가 무리수임을 증명해 줘." \
  --tier balanced \
  --artifact artifacts/synthetic-bilinear.json \
  --json
```

완전한 one-shot 정책을 맞출 때는 미확정 예산 의미를 명시하고 별도의 정책
artifact를 저장합니다.

```bash
tierroute train \
  --output artifacts/synthetic-bilinear.json \
  --policy-output artifacts/synthetic-policy.json \
  --budget-scope per-query \
  --json
tierroute route "루트 2가 무리수임을 증명해 줘." \
  --tier balanced \
  --artifact artifacts/synthetic-bilinear.json \
  --policy-artifact artifacts/synthetic-policy.json \
  --json
```

정책 학습은 비공개 example ID로 구분한 inner-LODO out-of-fold 예측을 만들고,
모든 모델 쌍의 음이 아닌 exact rational 품질/비용 breakpoint 발생을 스트리밍한
다음, 남긴 모든 후보를 선택한 budget ledger로 replay합니다. 기본 CLI는 root의
결정론적인 `bounded-bottom-hash-v2` 표본과 최솟값·최댓값을 남기고, 그 root들로
경계·중간값·tail을 만든 뒤 최종 결과를 최대 257개로 rank-spacing합니다. 이
cap 안에 모든 고유 root와 파생 후보가 들어오면 결과는 여전히 완전하며 정확한
개수와 `exhaustive: true`를 기록합니다. 실제 truncation이 일어난 경우에만
근사이며 `exhaustive: false`, 알 수 없는 전체 후보 개수 `null`, 탐색 strategy,
관측한 breakpoint 발생 횟수를 기록합니다. 전체 exact 유한 집합의 materialize·평가를
요청하려면 `--exhaustive-lambda-search`를 사용합니다. predictor fitting이나 root 생성을
시작하기 전에 보수적으로 계산한 모델 쌍 scan 상한 10,000,000회, 후보 상한 100,000개,
utility 평가 상한 100,000,000회, exact-rational 후보 상태의 추정 peak 256 MiB,
직렬화할 정책 근거의 추정치 8 MiB 중 하나라도 넘으면 즉시 거부합니다. 다섯 산정량을
모두 검토한 뒤에만 exhaustive CLI 실행에 `--allow-large-exhaustive-search`를
함께 지정할 수 있습니다. 기본 257개 capped 실행은 동봉한 합성 데이터에서는 이 범위
안에 있지만, 모든 cap은 실제 데이터셋의 retained work·정수 폭 preflight를 통과해야 하며
실패하면 줄여야 합니다. 고정 RouterBench 규모(34,778행, 모델 11개, tier 3개)의 보수적
상한은 후보 3,825,582개와 약 4.39조 회의 utility 평가입니다. 기본 cap 257은
294,952,218회를 요구해 거부되므로, 보수적 평가량이 73,451,136회인 cap 64를 이 전체
규모의 시작점으로 문서화합니다. 모델 쌍 scan 1,912,790회도 별도 scan 한도 안에
있지만, 데이터별 artifact 크기 추정도 통과해야 합니다. 이 경우에도 artifact는 retained
탐색의 complete/truncated 여부를 기록합니다. 선택된 lambda 자체는 언제나 정확한
분자/분모로 유지됩니다. v2는
부호와 길이를 자체 포함하는 이진 정수 identity를 hash하여 Python의 10진 정수 출력 제한을
피합니다. artifact에는 값과 strategy version이 함께 들어 있으므로 기존 v1 strategy
metadata도 계속 로드할 수 있습니다.

`--budget-scope cumulative`로 학습한 정책을 라우팅할 때는 현재 상태인 정확한
`--remaining-budget`도 반드시 전달해야 합니다. CLI는 초기 잔액을 추측하거나
쿼리당 예산 가정을 몰래 재사용하지 않습니다.

다른 version-1 replay JSON에는 두 명령 모두 `--data path/to/replay.json`을 추가합니다.
동봉 데이터 명령은 학습·저장·로드·라우팅 배선만 검증하며 보고 가능한 벤치마크
결과를 만들지 않습니다. CLI는 전체 입력으로 배포용 artifact를 맞추고, isotonic
calibration과 lambda 선택은 모두 out-of-fold 예측을 사용합니다. 보고용 실험은
`nested_lodo_lambda_evaluation`을 사용해야 합니다. 각 outer 학습 영역에서만 inner-LODO
lambda와 전체 학습 예측기를 다시 맞추고, 전혀 보지 않은 outer domain만 점수화합니다.
누적 예산이 fold마다 초기화되지 않도록 모든 outer 예측을 원래 전체 순서로 한 번만
replay합니다.

내장 solver는 표면 특징 schema와 적당한 크기의 행렬을 위한 감사 가능한 참조
backend이며 복잡도는 `O(n*d^2 + d^3)`입니다. 계획된 1,024차원 bge-m3 임베딩과
표면 특징을 합친 약 1,030차원으로 RouterBench 전체 보고 실험을 하려면 별도로
검토한 가속 backend와 수치 parity 테스트가 필요합니다. 참조 solver에 맞추기 위해
임베딩 차원을 조용히 줄이거나 버리지 않습니다. 보수적인 연산량 guard가 검토되지
않은 큰 작업이 cubic 참조 경로에 들어가기 전에 즉시 실패시킵니다. 학습은 정적으로
검토한 solver ID 하나를 한 번만 resolve하고 같은 solver를 모든 inner-LODO fit과
최종 refit에 전달합니다. 밀집 임베딩을 materialize하기 전에 preflight하며,
추론은 저장된 coefficient만 쓰므로 선택적 수치 의존성 없이 artifact를 읽습니다.
로드 시 solver ID만 검증하고 학습 solver를 resolve하거나 실행하지 않으며, 알 수 없는
ID는 계속 fail-closed 처리합니다.

## 현재 구현 범위

- 호출자의 정밀도가 미세한 초과지출을 반올림하지 못하는 context 독립적 `Decimal`
  정확 비용 계산과 `RouterState`/`RouterAction` 타입 계약. 0이 아닌 비용은 10진 위치
  `-100000`부터 `99999`까지, coefficient 최대 100,000자리 범위에서 지원하며 입력이나
  연산 결과가 이 명시적 자원 계약을 벗어나면 조용한 underflow·무제한 확장 전에 실패
- 쿼리당·누적 예산 ledger 교체 구조(공식 범위 확정 전 데모는 예시용 쿼리당 한도)
- exact rational utility, 불변 tier별 schedule, 완전 exhaustive breakpoint 탐색 또는
  명시적으로 표시한 truncated bounded-memory 근사 탐색을 쓰는 one-shot lambda 라우팅과
  재현 가능한 베이스라인 6종
- full-information offline replay: 정답 품질과 미호출 출력은 `RouterState`에
  들어가지 않음. 로그된 outcome을 소모한 모든 모의 호출에 견적·실현 청구액·
  ledger 잔액 snapshot·ledger 판정을 기록. 실현 초과지출 호출도 이미 실행된
  replay 호출이므로 정확한 지출 근거에 남음
- 모든 전체 평가 보고서는 정렬된 tier 명세, 호출 상한, 전체 replay 순서, 출력,
  label, 후보 순서, 정책이 보는 metadata를 결합한 필수
  `EvaluationScopeIdentity`(`tierroute-evaluation-scope-v1`)를 가짐. 라우팅 전에
  replay가 보는 비용과 metadata를 하나의 canonical 불변 snapshot으로 복사하며,
  지원하지 않는 객체나 cycle은 `repr`·pickle로 추측해 직렬화하지 않고
  fail-closed 처리
- log-scaled 길이·코드/수식·프롬프트 유래 domain tag의 fitted schema, 프로젝트가
  작성한 결정론적 centered-ridge, inner-LODO out-of-fold 예측, 모델별 독립
  isotonic calibration
- 엄격히 검증하는 canonical JSON 예측기 artifact. 예측기 로더는 pickle을 받지 않고,
  batch 예측은 프롬프트 batch를 한 번만 vectorize/embed
- canonical 정책 artifact는 정확한 predictor hash, 학습/지표에 관련된 replay
  내용과 순서, tier 명세, ledger 식별자, 남긴 후보 탐색 근거를 함께 결합합니다. OOF 예측
  hash는 감사 메타데이터로 기록하며, 라우팅 시 OOF 표가 없으므로 검증하려면 cross-fitted
  예측 표를 재현해 대조해야 합니다. 비용이 큰 parsing 전에 파일 8 MiB, 코어 비용 계약에서
  파생 가능한 후보를 포괄하는 exact 정수당 10진 404,096자리, tier당 retained 후보
  100,000개 상한을 적용합니다. ledger adapter 이름은 4 KiB로 제한하며, fitting 전 artifact
  추정에는 고정 metadata 여유분만 쓰지 않고 실제 인코딩한 domain과 tier budget 텍스트를
  포함합니다.
- 예측기와 정책 파일은 배타적인 무작위 staging, 저장 후 검증, 정책을 마지막에 교체하는
  rollback-safe bundle 저장을 사용하며 입력 alias와 안전하지 않은 출력 node는 fitting 전에
  거부합니다. 일반 OS/Python 예외에서는 시도한 모든 경로를 복구하지만, 동시 writer와 서로
  다른 pathname 전체에 대한 전원 장애 atomicity는 지원하지 않습니다.
- 모든 outer domain을 predictor fitting·calibration·lambda tuning에서 제외하는 true
  nested LODO orchestration
- domain table을 각 outer 학습 부분에서만 맞추고 fold 근거를 기록한 뒤, 6개 방법을
  같은 원본 순서 행에서 한 번씩 replay하는 쿼리별 outer-LODO 베이스라인 suite. 실제
  replay에 쓰는 ledger가 선언대로 쿼리마다 reset·charge·report하는지도 guard가 검증
- tier 가중 품질, oracle gap 회수율, 결정론적 leave-one-domain-out(LODO), tier별·
  tier 횡단 exact 견적-실현 비용 진단. random split 도우미는 의도적으로 제공하지 않음
- 엄격한 JSON 로더와 opt-in 방식의 고정 RouterBench 경계 어댑터

`--artifact`가 없으면 다운로드 없는 CLI는 설명 가능한 합성 데모 예측기를
사용합니다. 로컬 `bge-m3` 임베딩 백엔드와 GBM/bilinear 비교 실험은 아직
계획 항목이며 완료된 기능으로 주장하지 않습니다.

## Router 계약과 아키텍처

안정적인 의사결정 경계는 다음과 같습니다.

```text
state(prompt, budget_tier, remaining_budget, call_history, candidate_models)
  -> CallModel(model_id) | SelectOutput(history_index)
```

정답 품질과 미호출 출력은 replay 하네스 안에만 존재합니다. 코어의 비용에는 통화나
토큰 단위가 내장되어 있지 않으며, 어댑터가 대회 단위를 하나의 음이 아닌 값으로
정규화합니다. 정책에는 호출 전 견적 비용만 보이고 실제 청구액은 호출을 replay할
때까지 private outcome에 남습니다. 데이터셋 ID와 split 전용 domain label도 일반
라우터 상태에서 제외합니다. 배포 불가한 oracle과 outer-fold replay
schedule만 명시적인 평가 전용 경계를 통해 비공개 example key를 받습니다.
schedule은 outer 학습 행과 호출 전에 관찰 가능한 metadata로 낸 결정만 담으며,
split label을 정책 상태에 주입하지 않습니다.

`ReplayCall` 근거는 평가 결과에만 속하며 라우터에 새 label channel을 만들지
않습니다. `QueryResult.cost`는 ledger 판정이 false인 호출까지 포함한 모든
실행 replay 호출의 실현 청구액 exact 합입니다. outcome을 replay하기 전에
거부된 호출은 제외합니다. `selected_call_index`는 오늘의 one-shot replay에서도
반환한 로그 호출(보통 index 0)을 식별합니다. 0이 아닌 이전 호출을 선택하는
능력은 향후 cascade 스키마 준비일 뿐, history-adaptive cascade 구현 주장이
아닙니다. 잔액 snapshot과 `within_budget`은 코어가 공식 예산 규칙을 추론하지
않고 선택한 adapter의 의미를 보존합니다.

```text
JSON / RouterBench 경계 ──> typed replay examples ──> OfflineSimulator
                                  │                       │
prompt ─> fitted 특징 encoder ─> calibrated 예측기 ─> 정책 <──── 예산 ledger
                                      │
                              CallModel / SelectOutput

core/        상태·행동·모델·검증 계약
features/    오프라인 표면 특징, fitted schema, 로컬 임베딩 계약
predictors/  bilinear 학습기, 모델별 calibration, 엄격한 JSON artifact
policies/    exact one-shot lambda 정책, 튜닝/artifact, 필수 베이스라인
eval/        replay, 예산 protocol, 지표, planning, LODO
adapters/    예산 범위와 외부 데이터 불확실성 경계
```

시뮬레이터 기본값은 쿼리당 1회 호출입니다. SK텔레콤이 순차 다중 호출 평가를
허용한다고 확인하기 전까지 cascade 승급은 비활성입니다. 향후 스키마·예산 변경은
코어가 아니라 `adapters/`에 국지화합니다.

## 평가

tier `t`의 평균 품질을 `Q_t`, 가중치를 `w_t`라 하면 기본 로컬 지표는 다음과
같습니다.

```text
tier 가중 품질 = sum_t(w_t * Q_t) / sum_t(w_t)
```

한 tier라도 불완전하거나 예산을 위반하면 가중 점수는 계산하지 않으며, 실패한 tier의
가중치를 다른 tier에 재분배하지 않습니다. 합성 fixture의 Fast/Balanced/Premium
가중치 `0.5/0.3/0.2`는 저예산 고가중 동작을 확인하기 위한 예시일 뿐 SK텔레콤
공식 가중치가 아닙니다.

비용 근거는 실행된 로그 replay 호출에서 계산합니다. 호출 하나에서 `underquoted`는
`realized_cost > quoted_cost`, `overquoted`는 그 반대입니다. 전체 absolute
quote error는 호출별 음이 아닌 오차 크기를 더하므로 반대 방향 오차가 상쇄되지
않습니다. 별도 net error는 float나 0으로 나누는 percentage 없이
`sum(realized) - sum(quoted)`의 exact 방향과 크기를 제공합니다. Tier 행은 호출
실현 비용을 `BudgetReport.spent`, ledger 초과 횟수와 대조합니다. overall 행은
tier ledger가 서로 독립이므로 tier 횡단 진단일 뿐 공유 예산이나 예산 준수
판정이 아닙니다. 기존 최상위 `total_cost`와 명시적 alias인
`total_realized_cost`도 overall 실현 합계와 같으며 동일하게 tier 횡단 진단일
뿐입니다.

현재 쿼리별 회계에서 oracle gap 회수율은 always-cheapest에서 각 쿼리마다 독립적으로
예산 내인 oracle까지의 가중 품질 간격 중 라우터가 회수한 비율입니다.

```text
sum_t w_t * (Q_router,t - Q_cheapest,t)
-------------------------------------------------
sum_t w_t * (Q_oracle,t - Q_cheapest,t)
```

oracle과 cheapest가 같으면 정의되지 않으며 음수도 그대로 보존합니다. 동봉 oracle
planner는 쿼리별 예산에서만 상한입니다. 누적 스트림의 oracle이 아니며, 누적 결과에는
아직 구현하지 않은 sequence-level 계획이 필요합니다.

| 베이스라인 | 선택 규칙 |
| --- | --- |
| `always-cheapest` | 최저 비용, 동률이면 모델 ID 순 |
| `always-premium` | 지정 premium 모델. 낮은 tier에서는 예산 위반 가능 |
| `random` | 예산 내 모델 중 seed 기반·순서 독립 선택 |
| `length-heuristic` | 긴 프롬프트 또는 코드/수식이면 예산 내 strong 모델 |
| `oracle` | 정답을 사용하는 쿼리별 예산 내 품질 상한 |
| `domain-best-table` | 학습 행의 관찰 가능한 tag로 맞춘 tier별 평균 품질표, 미등록 tag는 cheapest |

Lambda 튜닝은 proxy loss가 아니라 이 실제 지표를 직접 최대화합니다. 각 프롬프트에서
모델 utility는 lambda의 affine 함수이므로 선택은 exact 품질/비용 교차점에서만 바뀔 수
있습니다. 완전 탐색은 모든 경계, 인접 경계 사이 열린 구간의 대표값 하나, 마지막 경계
뒤 tail 값 하나를 검사합니다. 각 후보는 `OfflineSimulator`로 replay하므로 infeasible
후보는 선택될 수 없습니다. Tier ledger가 서로 독립이고 모든 가중치가 양수이므로 tier별
최적화는 남긴 후보 집합의 Cartesian joint search와 정확히 같습니다. 이 결과가 전체
exact 유한 joint 최적임을 보장하는 것은 후보 집합이 `exhaustive: true`일 때뿐이며,
truncated bounded 탐색은 근사로 남습니다. 증명·동률 규칙·누수 경계는
[docs/lambda-tuning.md](docs/lambda-tuning.md)에 설명합니다.

`tierroute evaluate`는 `evaluate_per_query_lodo_baselines`를 호출합니다. 각 outer
fold에서 학습 행만으로 domain table을 맞춘 다음 해당 fold의 test 결정만
남깁니다. 그런 다음 6개 방법을 같은 원본 행 순서로 한 번씩 replay하고 동일한
쿼리별 회계 계약을 검증합니다. 동봉 데이터와 tier 가중치는 여전히 합성 smoke
입력이므로 그 숫자는 벤치마크 근거가 아닙니다. 보고서 간 지표는 tier·ledger 필드보다
먼저 동일한 evaluation-scope identity를 요구합니다. 6-baseline 생성자는 각 보고서에서
score·실현 비용 합계·견적 오차 요약·oracle gap을 다시 계산하므로 다른 replay의 행이나
오래된 파생값을 섞으면 실패합니다. JSON과 text CLI는 scope algorithm·digest·
`max_calls_per_query`를 표시합니다.

Scope digest는 실수로 다른 평가를 섞는 일을 막고 재현성을 확인하는 식별자이며 인증
서명은 아닙니다. 서로 다른 정책을 비교할 수 있도록 router action은 포함하지 않습니다.
Ledger 구현 의미론은 안전하게 hash할 수 없으므로 metric layer가 adapter 이름,
configured/effective limit, query 순서, 기록된 회계를 별도로 대조합니다.
정확한 필드와 canonical byte 계약은
[docs/evaluation-scope.md](docs/evaluation-scope.md)에 기록했습니다.

데이터셋 domain은 어댑터가 호출 전에 유효한 tag를 `router_metadata["domain"]`에
명시한 경우에만 `RouterState`에 전달되며, split 전용 label은 비공개로 유지합니다.
관찰 tag가 LODO split domain과 같으면 held-out tag는 학습에서 보지 못했으므로
domain-table은 의도대로 cheapest fallback, 즉 always-cheapest와 같아집니다. split domain을
넘어 공유되는 별도의 관찰 tag가 있을 때만 domain table이 일반화할 수 있습니다.
누적 비교는 sequence-level oracle가 있어야 하므로 계속 보류합니다.

## 데이터와 모델 자산

런타임 라우팅과 평가는 네트워크를 호출하지 않습니다. 다운로드는 명시적인 별도 준비
단계여야 하며 Hugging Face 자동 fallback은 금지합니다. 다운로드 데이터와 모델
가중치는 Git에서 제외하며, 재배포 라이선스가 확인되지 않으면 커밋하지 않습니다.
`artifacts/`의 로컬 학습 파일도 기본적으로 무시하여, 데이터 유래 parameter를 의도적으로
배포하기 전에 provenance와 라이선스를 따로 검토하게 합니다.

### 동봉 합성 데이터

`src/tierroute/data/synthetic.json`과 license sidecar는 프로젝트가 직접 만들었고
Apache-2.0으로 배포합니다. replay JSON은 `schema_version: 1`이며 tier 명세와 각
프롬프트에 대한 모든 후보 모델의 출력·문자열 비용·품질을 기록합니다.

### RouterBench(선택·opt-in)

RouterBench는 저장소에 포함하지 않습니다. 고정 revision의 dataset card에 라이선스가
선언되어 있지 않아 tierroute는 **`NOASSERTION`**으로 기록하며 재배포 권리를 부여하지
않습니다. 사용 전 [고정 revision의 dataset card](https://huggingface.co/datasets/withmartian/routerbench/blob/784021482c3f320c6619ed4b3bb3b41a21424fcb/README.md)를
검토하고 필요한 허가를 직접 확보해야 합니다.

- 파일: `routerbench_0shot.pkl`
- Revision: `784021482c3f320c6619ed4b3bb3b41a21424fcb`
- 크기: `99,567,659` bytes
- SHA-256: `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`

별도 reader 패키지는 필요하지 않습니다. core를 설치한 checkout에서 명시적
다운로드를 실행합니다.

```bash
python scripts/download_routerbench.py \
  --output data/routerbench/routerbench_0shot.pkl
```

upstream 파일은 pickle wire format을 사용하지만 tierroute는 `pickle.load`,
`pickle.Unpickler`, `pandas.read_pickle`을 호출하지 않습니다. 어댑터는 먼저 정확히
고정된 크기와 SHA-256을 요구한 뒤, 프로젝트가 작성한 non-dispatching 표준
라이브러리 opcode decoder로 구조만 해석합니다. 참조된 global은 비활성 데이터로
남아 payload가 지칭하는 callable을 import하거나 실행하지 않습니다. 예상하지 않은 opcode,
global, block layout, dtype, shape, memo 참조, trailing byte, table schema는 거부합니다.
이 decoder는 위의 정확한 artifact만 의도적으로 지원하며 pandas와 NumPy 의존성을
추가하지 않습니다.

decoder 회귀 oracle로 로컬 검증은 정확히 36,497행×37열과 canonical semantic
SHA-256 `7b4749ad5c4bdb338c2317b306c382680b1a23dc83c73e29ab805b8f7e472e87`도
요구합니다. semantic digest는 열 순서, UTF-8 문자열, IEEE-754 binary64 값을 framing하며,
인증에 사용하는 artifact SHA-256을 대체하지 않습니다. 선언된 benchmark mapping에는
11개 모델과 7개 LODO domain에 속한 34,778개 예제가 남습니다.

다운로드 후 Hugging Face offline mode를 설정하고 전체 파일과 결정론적 prefix를
검증합니다. validator 자체에는 network client가 없으며 지정한 로컬 경로만 읽습니다.

```bash
HF_HUB_OFFLINE=1 python scripts/validate_routerbench.py --limit 200
```

고정 artifact 전체 검증은 in-scope row 34,778개, 모델 11개, LODO domain 7개를
확인한 뒤 메모리를 제한하기 위해 지정한 replay prefix만 typed example로 변환합니다.
이 개수는 artifact/schema 검증 사실이며 **모델 품질 benchmark claim이 아닙니다.** 현재
`evaluate --data`는 이
pickle이 아니라 replay JSON만 입력받습니다. RouterBench 비용은 응답 후 실현 비용이므로
검증 스크립트는 별도 calibration prefix에서 모델별 호출 전 견적을 맞추고, 라우팅할
행의 실제 청구액은 정책에 노출하지 않습니다.

인증된 wire table은 메모리에 materialize됩니다. 최소 512 MiB 여유를 권장하며 기준
Python 3.12 환경에서 기본 prefix 검증의 최대 RSS는 약 290 MB였습니다. `--limit`는
typed replay 보유량을 제한하고, `--limit 0`은 의도적으로 calibration 이후 전체 행을
replay합니다.

### bge-m3(계획, local-only)

임베딩 계약은 `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`(MIT)을 고정합니다. 가중치는 동봉하지
않고 런타임 downloader도 없습니다. 계획된 provider는 미리 준비한 로컬 경로만 받고,
`HF_HUB_OFFLINE=1`에서 Hub ID를 조회하지 않고 즉시 실패하도록 구현합니다.
약 1,030개 전체 특징 학습도 프로젝트 참조 구현과 parity test를 통과한 승인된 가속
solver가 마련될 때까지 gate로 남깁니다. 임베딩 차원을 조용히 투영해 버리지 않습니다.

SK텔레콤 대회 데이터도 라이선스와 재배포 조건을 서면 확인하기 전까지 포함하지 않습니다.

## 개발 검증

```bash
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

`install-dev`는 프로젝트 가상환경 안에서만 실행하세요. 일부 Python 3.10
`ensurepip`이 남긴 setuptools를 제거한 뒤, 감사한 flit_core 기반 lock을 설치합니다.

`make reproduce`는 정확한 개발 lock을 설치하고 동봉 데이터 전체 파이프라인을
실행하며 학습과 artifact 라우팅도 포함합니다. CI는 lint, 테스트, 의존성 없는 wheel
설치, 두 CLI smoke 경로, offline mode, 의존성 라이선스 gate를 검사합니다. GPL 계열
의존성은 허용하지 않습니다. 기여·컴플라이언스 절차는
[CONTRIBUTING.md](CONTRIBUTING.md), 의존성 목록은 [SBOM.md](SBOM.md), 실제 wheel
내용을 기준으로 한 승인·거부 기록은
[docs/dependency-license-audit.md](docs/dependency-license-audit.md)를 참고하세요.

핵심 연구의 원문 검토, 선행연구 비교, 구현 완료·계획·공식 확인 대기 주장의 정확한
경계는 [docs/literature-and-novelty.md](docs/literature-and-novelty.md)에 기록합니다.
보고서나 발표에서 성능·OOD·novelty 문구를 재사용하기 전에 확인하세요.

개발 보조 도구의 실질적 사용 범위, 증거의 한계, 사람 검토 상태는
[docs/ai-assistance-audit.md](docs/ai-assistance-audit.md)에 기록합니다. 핵심 불변식별
설명·실패 경로 검토 자료와 참가자 서명 표는
[docs/maintainer-explainability.md](docs/maintainer-explainability.md)에 있습니다. CI와
AI 에이전트 리뷰는 자동화 증거이며 참가자의 사람 검토 서명을 대신하지 않습니다.

## Open questions

공식 답변 전까지 다음 결정은 어댑터 또는 설정에만 둡니다.

1. tier 예산은 쿼리마다 초기화되는가, 정렬된 스트림 전체에 누적되는가? 라우터에
   공개되는 호출 이력 필드는 정확히 무엇인가?
2. 공식 시뮬레이터가 순차 다중 호출과 기존 출력 선택을 허용하는가? 확인 전에는
   cascade를 범위에 넣지 않습니다.
3. SK텔레콤 데이터의 라이선스·재배포 조건과 공식 Fast/Balanced/Premium 가중치는
   무엇인가? 라이선스를 서면 확인하기 전 SK텔레콤 데이터는 커밋하지 않습니다.
4. tierroute가 학습한 ridge/bilinear+isotonic 예측기 artifact를 붙임2 유형3 자체 개발
   모델로 신고해야 하는가? 외부 모델 파인튜닝은 아니며, 최종 명세 전에 사무국의
   서면 해석을 받습니다.

## 라이선스

프로젝트가 작성한 코드와 문서는 [Apache-2.0](LICENSE)입니다. 소스와 문서에는 SPDX
식별자를 둡니다. 제3자 데이터·모델 자산은 각자의 조건을 유지하며 tierroute 라이선스로
재허가되지 않습니다.
