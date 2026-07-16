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
bilinear 학습·인메모리 결정론적 GBM 참조 학습기·두 계열의 paired 기술 추정·exact
tier-lambda 튜닝·엄격한 bilinear 예측기 v1/정책 artifact·정확한 견적-실현 비용
오차 지표·bounded prepared moment solve/raw-score 참조 경로·외부 데이터가 필요
없는 데모가 구현되어 있습니다.
CLI는 모델을 **선택**할 뿐 실제 LLM을 호출하거나 답변을 생성하지
않습니다.

## 빠른 시작

Python 3.10 이상이 필요합니다. 새로 받은 저장소에서 실행합니다.

```bash
cd tierroute
python -m venv .venv
```

POSIX 호환 셸에서는 `. .venv/bin/activate`, Windows PowerShell에서는
`.\.venv\Scripts\Activate.ps1`로 가상환경을 활성화한 뒤 설치합니다.

```bash
python -m pip install -e .
```

단일 라우팅, 베이스라인 6종 평가, 학습 라우터와 베이스라인 비교 벤치마크, paired
예측기 추정, 학습 기반 3단계 showcase를 차례로 실행합니다.

```bash
tierroute route "루트 2가 무리수임을 증명해 줘." --tier fast
tierroute evaluate
tierroute benchmark --budget-scope per-query
tierroute compare-predictors --budget-scope per-query
tierroute demo
```

`python -m tierroute`도 같은 진입점입니다. `route`, `evaluate`, `benchmark`,
`compare-predictors`, `demo`, `train`에는 `--json`을 쓸 수 있고, 버전이 명시된
호환 replay JSON을 평가와 벤치마크 입력으로 지정할 수 있습니다.

```bash
tierroute route "이 Python 함수를 디버그해 줘" --tier balanced --json
tierroute evaluate --data src/tierroute/data/synthetic.json --json
tierroute benchmark --budget-scope per-query \
  --data src/tierroute/data/synthetic.json --json
tierroute compare-predictors --budget-scope per-query --json
HF_HUB_OFFLINE=1 tierroute demo --json
```

`route --json`은 실행 전 결정입니다. `cost`는 `quoted_cost`의 의미상 alias로
남고 `realized_cost`는 `null`입니다. `evaluate --json`은 로그된 outcome을
replay하고 실제로 실행된 replay 호출의 tier별·tier 횡단 `cost_evidence`를
출력합니다. 두 명령 모두 실제 provider를 호출하지 않습니다.

동봉된 프롬프트·비용·출력·예측 품질·scorecard·benchmark 행은 프로젝트가 만든
**합성 smoke-test 값**입니다. 배선 검증용이며 벤치마크 결과, 실제 모델 비교,
대회 점수가 아닙니다.

### 3단계 학습 라우터 showcase

사람용 출력과 machine-readable 출력은 다음 명령으로 실행합니다.

```bash
tierroute demo
tierroute demo --json
```

데모는 Fast, Balanced, Premium마다 동봉된 합성 행 하나를 선택해 결정론적인 3개
프롬프트 stream으로 보여 줍니다. 각 단계는 해당 행의 outer-LODO 학습 부분에서만
맞춘 learned/tuned policy를 가져와 example 하나·tier 하나인 직접
`OfflineSimulator` replay를 수행합니다. 직접 결과는 `tierroute benchmark`가 감사한
nested-LODO 학습 결과의 같은 행·tier와 반드시 일치해야 하며, 다르면 숨기지 않고
showcase가 실패합니다.

각 단계는 예시용 쿼리별 예산, 견적·실현 비용, 관찰된 합성 품질, 독립적인 쿼리별
oracle 품질, 누적 실현 비용, 가중치를 적용하지 않은 누적 품질 보존율을 표시합니다.

| 단계 / tier | 동봉 행 | 예산 | 모델 | 견적 → 실현 | 관찰 / oracle | 누적 비용 | 보존율 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 / Fast | `synthetic-science-001` | 0.35 | `swift` | 0.2 → 0.2 | 0.78 / 0.78 | 0.2 | 100% |
| 2 / Balanced | `synthetic-math-002` | 0.7 | `steady` | 0.6 → 0.6 | 0.75 / 0.75 | 0.8 | 100% |
| 3 / Premium | `synthetic-code-002` | 1 | `expert` | 1 → 1 | 0.96 / 0.96 | 1.8 | 100% |

```text
누적 품질 보존율 = 지금까지 관찰된 합성 품질의 합
                   / 지금까지 독립적인 쿼리별 oracle 품질의 합
```

누적 oracle 합이 0이면 보존율은 정의되지 않으므로 0으로 나누지 않고 사람용 출력은
`N/A`, JSON은 `null`을 냅니다.

> **해석 경계:** 누적 실현 비용은 서로 독립적인 쿼리별 ledger를 사용한 서로 다른
> tier의 호출을 더한 표시용 값입니다. 공식 공유 예산이나 누적 예산 회계가 아닙니다.
> 보존율 분모는 독립적인 쿼리별 oracle 값의 합이지 sequence-level oracle이 아니며,
> 이 비율은 oracle-gap 회수율도 아니고 공식 tier 가중치도 사용하지 않습니다. 모든
> 프롬프트·비용·품질·oracle 값은 프로젝트가 작성한 합성 배선 근거이며 실증 결과나
> 대회 결과가 아닙니다.

선택한 stream 행 3개는 presentation view일 뿐입니다. 사람용 출력은 그 뒤에 명확히
분리된 전체 모집단의 학습 라우터와 베이스라인 6종 표를 이어서 보여 줍니다. JSON은
버전이 지정된 `tierroute-routing-stream-showcase` schema를 사용해 3개 행을
`stream.steps`, 표시 규칙을 `accounting`, 전체 benchmark를 `benchmark_evidence`에
서로 분리합니다. 같은 전체 근거는
`tierroute benchmark --budget-scope per-query`로 독립 실행할 수 있으며, 데모는
선택 행 3개로 전체 모집단을 대신하거나 요약하지 않습니다. 각 JSON step은
`budget_limit`, `cost.quoted`, `cost.realized`,
`cost.cumulative_realized_reporting_only`, `quality.observed`,
`quality.per_query_oracle`, `quality.cumulative_retention`을 제공합니다.

### 오프라인 예측기와 정책 학습

학습과 추론에는 제3자 수치 계산 패키지가 필요하지 않습니다. 프로젝트가 작성한
결정론적 centered-ridge Cholesky solver가 모든 모델 target을 하나의 공유
factorization으로 맞추고 intercept에는 ridge penalty를 적용하지 않습니다. 표면
특징 artifact는 엄격한 canonical JSON이며 생성에 쓴 solver ID도 기록합니다.

아래 배포 명령은 계속 bilinear 전용입니다. GBM 상태는 인메모리 전용이며 versioned
artifact, `train`/`route`/showcase 연동을 제공하지 않습니다. 별도
`compare-predictors` 명령은 두 고정 계열을 평가하지만 어느 계열도 선택하지 않으며
성능 주장을 허용하지 않습니다.

```python
from tierroute.adapters import load_evaluation_dataset
from tierroute.predictors import GbmTrainingConfig, fit_calibrated_gbm

examples = load_evaluation_dataset().examples
predictor = fit_calibrated_gbm(examples, config=GbmTrainingConfig())
model_ids = tuple(model.model_id for model in examples[0].candidate_models)
scores = predictor.predict_many("이진 탐색이 로그 시간인 이유를 설명해 줘.", model_ids)
```

이 동봉 데이터 호출은 결정론적 배선 데모일 뿐, 측정한 예측 품질 결과가 아닙니다.

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

다른 version-1 replay JSON에는 학습과 벤치마크 명령에
`--data path/to/replay.json`을 추가합니다. `train`은 전체 입력으로 배포용 artifact를
맞추며 isotonic calibration과 lambda 선택은 모두 out-of-fold 예측을 사용하지만,
보고 가능한 벤치마크 결과를 만들지는 않습니다.

누수 없는 학습 라우터와 베이스라인 비교 근거는 전용 benchmark runner로 만듭니다.

```bash
tierroute benchmark --budget-scope per-query
tierroute benchmark --budget-scope per-query \
  --data path/to/replay.json --json
```

이 명령은 true nested LODO를 수행합니다. 각 outer 학습 영역에서만 inner-LODO
out-of-fold 예측과 lambda tuning을 수행한 뒤 그 outer 학습 영역 전체로 predictor를
refit하며, 전혀 보지 않은 outer domain만 점수화합니다. 이어서 학습 라우터와 정식
베이스라인 6종을 같은 원본 순서 모집단, 같은 `PerQueryBudgetLedger`와
`EvaluationScopeIdentity`에서 각각 한 번씩 replay합니다. JSON은 raw example ID
대신 held-out domain과 정확한 순서 train/test membership을 묶는 개수와 버전이
지정된 SHA-256 digest를 기록합니다. 이는 compact 재현성 identity이며 인증된 증명은
아닙니다. 또한 tier 예산 한도·가중치, 확정된 baseline 역할·seed·threshold·규칙
identity, 요청한 lambda-search cap 또는 exhaustive override를 기록하므로 replay에서
가중 결과를 독립적으로 재현할 수 있습니다. 별도의 versioned SHA-256 evidence digest는
baseline 파라미터를 그것이 만든 정확한 순서의 call 결정과 결합합니다. 명령은
offline으로 실행되고 `--budget-scope per-query`만 허용합니다. 누적 벤치마크와
cascade 주장은 주최 측이 sequence-level 예산·호출 이력 의미를 확정하고 tierroute가
sequence-level oracle을 구현할 때까지 gate 상태입니다.

동봉 합성 데이터로 실행한 결과는 벤치마크 배선만 검증하며 실증 근거가 아닙니다.
`--data`를 쓰는 경우 replay 데이터 라이선스와 그 출력에서 도출한 벤치마크·대회
주장의 타당성은 호출자 책임입니다.

두 고정 표면 특징 예측기 계열을 같은 outer 근거에서 확인하려면 별도 paired 추정
runner를 사용합니다.

```bash
tierroute compare-predictors --budget-scope per-query
tierroute compare-predictors --budget-scope per-query \
  --data path/to/replay.json --json
```

이 명령은 어느 계열도 fitting·embedding하기 전에 outer LODO, lambda-tuning LODO,
calibration LODO까지 이어지는 GBM 전체 호출 그래프를 열거하고 검토된 aggregate
split-scan 상한을 적용합니다. 베이스라인 6종은 한 번만 계산하며, 두 계열 결과가
동일한 replay·scope·tier·fold·모델 catalogue·검색 설정·baseline 근거를 공유해야
합니다. `--json`에서는 raw binary64 `GBM - bilinear`
tier·가중 품질·oracle-gap·held-out-domain 차이를 기록합니다. 사람용 출력은 표시하는
전체 가중 품질/oracle-gap 차이를 반올림하며 domain 표는 생략합니다. 피연산자 하나라도
없으면 가중치를 재분배하지 않고 JSON `null`을 냅니다. schema는
`selection_protocol=none-paired-estimation`,
`selected_family=null`, `performance_claim_allowed=false`를 고정하며 `winner` 필드가
없습니다. 동봉 출력은 `SYNTHETIC-ONLY`, `--data` 출력은
`UNVERIFIED-USER-DATA`입니다. 어느 상태도 우위·배포 추천·품질 향상·비용 절감 주장을
허용하지 않습니다. 계열 선택에는 별도 untouched 근거나 계열 선택까지 포함한 추가
검증 protocol이 필요합니다.

내장 solver는 표면 특징 schema와 적당한 크기의 행렬을 위한 감사 가능한 참조
backend이며 복잡도는 `O(n*d^2 + d^3)`입니다. tierroute에는 한 번의 bounded
dense solve를 수행하는 프로젝트 작성 C11 실험 구현도 있습니다. versioned protocol,
인증된 subprocess adapter, 다중 target 공유 Cholesky, 자원 preflight, malformed
입력 corpus, 줄이지 않은 1,024차원 parity 테스트는
[네이티브 protocol 문서](docs/native-ridge-protocol.md)에 기록합니다. 소스는 sdist에만
들어가고 wheel에는 실행 파일이나 네이티브 의존성이 없습니다. 기본 trainer와 CLI는
Python 참조 solver를 그대로 사용하며, 호출자가 고른 로컬 실행 파일의 정확한 byte를
인증하는 `train --ridge-solver native-c11` opt-in만 이 경로를 선택합니다.

소스 checkout에서 helper는 사용자가 명시적으로 고른 system compiler를 실행해 새 로컬
candidate를 build할 수 있습니다. helper 자체는 download나 PATH 탐색을 하지 않지만,
compiler를 sandbox하지 않으며 compiler/toolchain 자체가 네트워크를 쓰지 않았음을
증명하지도 않습니다. 두 인자는 모두 절대경로여야 하고 output은 미리 존재하면 안
되며, 명령은 소스와 실행 파일 SHA-256을 출력합니다.

```bash
python scripts/build_native_ridge.py \
  --compiler /absolute/path/to/clang \
  --output /absolute/new/path/tierroute-ridge
```

위 명령이 출력한 정확한 `sha256`을 사용합니다. helper와 adapter는 16 MiB보다 큰
실행 파일 candidate를 거부합니다. 모든 host에서 `//` 또는 `\\`로 시작하는 UNC·device
형식 경로도 거부하지만, 이미 연결된 drive나 mount된 network filesystem은 이식성 있게
판별할 수 없으므로 호출자가 확인해야 합니다. CLI는 실행 파일을 검색하거나 build하지
않고 JSON 결과에는 digest만 기록하며 경로는 기록하지 않습니다. 생성된 predictor
artifact로 route할 때는 실행 파일과 digest가 모두 필요 없습니다.

```bash
tierroute train \
  --output /absolute/new/path/predictor.json \
  --ridge-solver native-c11 \
  --native-ridge-binary /absolute/new/path/tierroute-ridge \
  --native-ridge-sha256 BUILD_출력의_SHA256 \
  --json
```

digest는 호출자가 선택한 byte의 동일성만 인증합니다. 승인, 소스 provenance 증명,
import audit, 네트워크 불사용 증명이 아닙니다. 따라서 native 학습 JSON은
`network_used`를 `null`, `python_orchestration_network_used`를 `false`,
`native_binary_audit`를 `caller-responsibility-unapproved`로 기록합니다. 자원
preflight는 embedding materialization 전에 크기가 제한된 binary를 인증하고,
`solve`는 교체 구간을 막기 위해 private snapshot을 만들면서 다시 인증합니다. 설정한
timeout은 child process가 시작된 뒤에만 적용됩니다. 인증 전 filesystem I/O와 request
serialization은 byte 상한으로 제한하지만 그 child timeout에는 포함되지 않습니다.

이 dense sidecar 하나만으로는 RouterBench 전체 보고 실험이 가능해지지 않습니다.
현재 배포 nested 경로는 특징 계산과 301개 fit을 여전히 반복합니다. 실험적
[prepared graph 계약](docs/prepared-session-graph.md)은 7-domain nested 평가에 고유
base-training subset 63개, subset/domain score block 154개, 정확히 `22N`
scored-row membership이 필요함을 고정합니다. Bounded
[prepared feature-store reference](docs/prepared-feature-store.md)는 호출자가 확인한
source·사전 계산 embedding digest에서 canonical little-endian binary64 fit row를
snapshot하고, 재사용 가능한 domain별 Welford moment를 만든 뒤 포함 domain만 Chan
방식으로 결합해 training-only tag와 population scale을 복원합니다.

Bounded·standard-library-only
[prepared execution reference](docs/prepared-reference-execution.md)는 이 store와 moment를 사용합니다.
canonical subset을 하나씩 결합·solve·폐기하고, subset별 하나의 Cholesky factor를
모든 model target에 공유하며, target와 무관한 domain별 feature shard를 만들고,
허용된 모든 feature coordinate를 사용한 row-major raw-score block을 생성합니다.
7개 domain에서 전체 구조는 domain 행 수가 균등하지 않아도 정확히 coefficient
block 63개, score block 154개, `22N` scored-row membership, `22NM` scalar raw
score입니다. Frozen 합성 테스트는 독립적으로 row를 fit한 reference와 수치 tolerance
내에서 같음을 검증합니다. Moment 축약은 연산 순서가 다르므로 bitwise parity가
아니며, 숫자 payload digest가 Python/platform 산술 구현 간에 일치한다고
보장하지 않습니다.

Bounded
[prepared policy-pipeline reference](docs/prepared-reference-pipeline.md)는 이 raw
block을 기존 model별 isotonic calibrator, exact/bounded lambda 탐색,
`OfflineSimulator`에 연결합니다. 새 budget/report schema를 만들지 않고 canonical
target·calibration·calibrated-score lineage와 기존 `NestedLodoLambdaResult`를
반환합니다. 동봉 4-domain replay와 불균등 7-domain fixture에서 후보 근거, exact 선택
lambda, 결정, 회계, 최종 보고서까지 기존 rowwise 경로의 전체 nested 결과와 같습니다.
7-domain fixture는 coefficient block 63개, raw-score block 154개, `22N` raw
membership, calibrated subset 28개, calibrated destination 49개를 모두 거칩니다.

지원하는 derivation 경로는 public builder 함수뿐입니다. Leaf dataclass 생성자를
직접 호출하면 스스로 선언한 canonical record만 검증합니다. 이는 aggregate loader,
주장한 입력에서 파생됐다는 증명, provenance 증명이 아닙니다. Versioned SHA-256은
content identity이지 인증이 아니며, 바꾸치기를 탐지하려면 신뢰하는 expected digest와
비교해야 합니다. Reference preflight는 검토한 수치 admission unit과 모델링한 숫자
payload/storage를 계산하고 현재 구현의 lambda pair 순회를 모두 세며, 실제 Decimal
비용 자릿수로 aggregate lambda candidate/policy-artifact byte 상한을 계산합니다. 다만
Python object
graph, allocator overhead, 호출자 소유 입력, 기타 process memory와 호출자가 주입한
임의 ledger factory 내부의 작업·부작용은 포함하지 않습니다. Peak RSS 견적이나
wall-time 보장이 아닙니다. 고정한 RouterBench/bge-m3 전체 크기는 bounded feature-store,
statistics, reference-execution cap에 의해 계속 거부됩니다.

이 reference들은 provider 추론과 file I/O를 하지 않으며 성능·품질·비용 감소를
주장하지 않습니다. Policy bridge는 bounded in-memory 배선 증거이지 CLI/runtime 연동,
native protocol, persistent/scalable session, all-domain 배포 artifact, 기본 trainer
대체가 아닙니다. Tolerance-close raw score가 모든 PAV partition과 exact 결정을
보존한다고 보장하지 않으며, official-data parity는 epsilon을 추가하지 않고 직접
비교해 불일치 시 실패해야 합니다. 숫자 digest도 cross-platform 보장이 아닌 local
근거입니다. 따라서 [Issue #9](https://github.com/Hbin77/tierroute/issues/9)는 아직
열려 있습니다.

계획된 1,024차원 bge-m3 임베딩과 표면 특징을 합친 최대 1,036차원 전체 학습은 감사된
offline local provider, scalable authenticated persistent prepared session과 CLI
재현, official-shape end-to-end parity, 그리고 감사된 Linux-musl·Windows-MSVC
artifact가 있을 때까지 gate로 남습니다. 임베딩 차원을
조용히 줄이거나 버리지 않습니다. 기존 row-training 경로의 보수적 연산량 guard,
정적 reviewed solver ID, pre-embedding preflight, unknown-ID 거부는 그대로며, 추론은
저장된 coefficient만 사용하므로 의존성이 없습니다.

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
- 모델별 regression stump ensemble, 고정된 순서·동률 규칙, 임베딩 호출 전 보수적
  자원 상한, inner-LODO out-of-fold 예측, 모델별 isotonic calibration을 갖춘 제3자
  의존성 없는 squared-error gradient boosting. 전체 nested-work preflight와 modest
  surface-only replay용 paired 기술 runner를 제공하며 합성 테스트는 알고리즘 배선만 입증
- 엄격히 검증하는 canonical JSON bilinear 예측기 artifact v1. 예측기 로더는 pickle을 받지 않고,
  읽기·parsing·직렬화·저장·정책 hash에 동일한 UTF-8 32 MiB 상한을 적용합니다. v1 구조는
  모델·학습 domain·feature tag 각각 4,096개, 전체 feature 16,384차원, 수치 scalar
  1,000,000개, JSON 숫자당 640자, metadata 값당 4 KiB·전체 1 MiB로 제한합니다. decoding
  전 decoded JSON 값을 materialize하지 않는 lexical pass는 nesting 32, JSON string token
  32,768개, 인코딩된 string token당 24,578자, 여는 container/comma 1,100,000개를 제한하고
  숫자 callback은 고정 필드 5개를 포함해 1,000,005개까지만 받습니다. 직접 전달한
  container는 한 번만 snapshot하고 수치는 finite binary64로 정규화합니다. calibrator
  point는 기록한 학습 example 수를 넘지
  않습니다. 이 범위는 계획된 11모델·1,036특징·34,778행 RouterBench/bge-m3 규모를 명시적
  여유와 함께 포괄합니다. batch 예측은 프롬프트 batch를 한 번만 vectorize/embed합니다.
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
- Bounded standard-library prepared reference는 고유 nested-LODO subset moment를 순차적으로
  결합하고, subset당 하나의 factorization으로 모든 model target을 solve하며,
  target-free feature shard를 바인딩해 모든 canonical raw-score block을 생성합니다.
  7개 domain의 테스트된 구조는 정확히 subset 63개, block 154개, `22N` row
  membership, `22NM` score cell입니다. 이는 합성 구조·수치 tolerance 근거일 뿐,
  scalable 실험이나 성능 결과가 아닙니다.
- Bounded prepared policy bridge는 `C(D,2)+D` inner-LODO calibration context와 `D^2`
  calibrated destination block을 만든 뒤 기존 lambda tuner와 simulator를 재사용합니다.
  4·7-domain frozen fixture는 rowwise 전체 nested 결과와 일치하며, trusted digest,
  aggregate preflight, 원본 순서 replay, 두 budget adapter, held-out-target
  noninterference를 테스트합니다. 배포 가능한 prepared artifact나 보편적 exact-parity
  주장은 아닙니다.
- true nested-LODO 학습 라우터와 베이스라인 6종을 동일한 평가 scope에서 비교하고,
  버전이 지정된 compact outer-fold membership digest를 공개하는 쿼리별 보고 형태의
  benchmark CLI
- calibrated bilinear와 GBM을 동일한 nested-LODO 근거에서 실행하고 하나의 6-baseline
  평가를 공유하며 기계 판독 JSON에 full-precision 기술 차이와
  no-selection/no-performance-claim metadata를 내는 별도 paired-estimation CLI
- 대응 outer-fold 학습 정책을 `OfflineSimulator`로 직접 replay하고 nested 결과와의
  일치를 검사하며, tier 혼합 누적 비용과 무가중 보존율을 합성 표시용 값으로 명시하는
  Fast/Balanced/Premium 3단계 showcase
- domain table을 각 outer 학습 부분에서만 맞추고 fold 근거를 기록한 뒤, 6개 방법을
  같은 원본 순서 행에서 한 번씩 replay하는 쿼리별 outer-LODO 베이스라인 suite. 실제
  replay에 쓰는 ledger가 선언대로 쿼리마다 reset·charge·report하는지도 guard가 검증
- tier 가중 품질, oracle gap 회수율, 결정론적 leave-one-domain-out(LODO), tier별·
  tier 횡단 exact 견적-실현 비용 진단. random split 도우미는 의도적으로 제공하지 않음
- 엄격한 JSON 로더와 opt-in 방식의 고정 RouterBench 경계 어댑터

`--artifact`가 없으면 다운로드 없는 CLI는 설명 가능한 합성 데모 예측기를
사용합니다. 결정론적 GBM 상태는 인메모리이며, 제공하는 CLI 중에는 비배포
paired-estimation runner만 이를 학습합니다. 로컬 `bge-m3` 임베딩 백엔드, GBM
artifact와 배포 CLI 연동, 라이선스가 확인된 보고 가능 계열 선택 실험은 아직 계획
항목이며 어떤 예측기 계열의 우위도 주장하지 않습니다.

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
predictors/  bilinear 학습/artifact, 인메모리 결정론적 GBM, 모델별 calibration
policies/    exact one-shot lambda 정책, 튜닝/artifact, 필수 베이스라인, paired 추정
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

`tierroute benchmark --budget-scope per-query`는 calibrated bilinear one-shot
라우터를 true nested LODO로 추가한 뒤 같은 베이스라인 보고서 6개와 비교합니다.
학습 보고서와 모든 베이스라인은 동일한 evaluation-scope identity, tier 명세, query
순서, 쿼리별 회계 근거를 가져야 합니다. 각 outer fold는 train/test 개수와 정확한
순서 membership 및 held-out domain의 `tierroute-fold-membership-sha256-v1`
digest를 기록하며 CLI는 example ID 자체를 노출하지 않습니다. 이 digest는 compact
재현성 근거이지 인증 서명은 아닙니다. 누적·cascade 평가는 위 설명대로 계속 gate
상태입니다.

`tierroute compare-predictors --budget-scope per-query`는 기존 bilinear benchmark
계약을 바꾸지 않고, 동일한 outer fold에 독립적으로 튜닝한 calibrated GBM 결과를
추가합니다. 베이스라인 6종은 한 번 평가해 공유합니다. 차이는 기술적인
`GBM - bilinear` 추정치일 뿐이며, 같은 outer 근거로 계열을 선택하면 family-selection
bias가 생기므로 결과에는 winner나 배포 추천이 없습니다.

`tierroute demo [--json]`은 의도적으로 이 benchmark보다 좁습니다. tier마다 동봉 행
하나씩 총 3개를 선택하고 같은 outer-training-only 학습 정책으로 각 행을 직접
replay합니다. 누적 실현 비용은 서로 독립적인 쿼리별 ledger를 합친 tier 혼합 표시
합계입니다. 가중치를 적용하지 않은 품질 보존율은
`sum(관찰 품질) / sum(독립적인 쿼리별 oracle 품질)`입니다. 둘 다 공유 예산 회계,
sequence-level oracle 비교, oracle-gap 회수율이 아닙니다. 사람용 명령은 분리된 전체
benchmark 표를 뒤에 출력하고, JSON은 3행 `stream.steps` 밖의
`benchmark_evidence`에 그 근거를 유지합니다.

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

모든 `--data` 명령은 하나의 엄격하고 유한한 loader를 공유합니다. 안정된 regular-file
읽기, strict UTF-8/JSON, 정확한 field·primitive type, 입력 256 MiB, example 100,000개,
전체 outcome 1,000,000개, prompt/output 각각 1 MiB, outer/nested LODO 작업량 한도를
강제하며 무제한 우회 옵션은 없습니다. 전체 계약과 공식 데이터 migration 규칙은
[version-1 replay JSON 문서](docs/replay-json.md)에 있습니다.

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

POSIX 환경에서는 downloader가 partial file을 만들 때부터 검증된 최종 파일까지
소유자 전용 `0600` 권한을 강제하며, 이 권한을 보장할 수 없으면 실패 처리합니다.
각 실행은 동일 directory 안의 예측 불가능한 staging 이름을 독점하고, 교체 전에 열린
descriptor의 identity를 확인하며, 성공을 반환하기 전에 설치된 파일을 다시 인증합니다.
symlink와 non-regular destination은 거부하므로 동시 실행이 하나의 예측 가능한 staging
경로를 공유하거나 서로 삭제하지 않습니다.

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
행의 실제 청구액은 정책에 노출하지 않습니다. 다만 artifact 순서의 이 prefix는 전부
`arc-challenge` 행이므로 기본 명령은 구조 smoke check일 뿐입니다. replay 성능·비용
값은 출력하지 않고, 이 견적을 cross-domain 근거로 사용하지 않습니다. 아래의 균형
diagnostic은 학습 정책 배선을 점검할 때 이 prefix를 대체합니다.

인증된 wire table은 메모리에 materialize됩니다. 최소 512 MiB 여유를 권장하며 기준
Python 3.12 환경에서 기본 prefix 검증의 최대 RSS는 약 290 MB였습니다. `--limit`는
typed replay 보유량을 제한하고, `--limit 0`은 의도적으로 calibration 이후 전체 행을
replay합니다.

위 prefix replay는 기본 smoke 경로로 그대로 유지됩니다. 학습 정책과 canonical
베이스라인 6종을 점검하는 로컬 diagnostic은 다음처럼 명시적으로 동의해야만
실행됩니다.

```bash
HF_HUB_OFFLINE=1 python scripts/validate_routerbench.py \
  --nested-lodo --acknowledge-noassertion

# 같은 provenance/구조 근거를 하나의 JSON 문서로 출력합니다.
HF_HUB_OFFLINE=1 python scripts/validate_routerbench.py \
  --nested-lodo --acknowledge-noassertion --json
```

일반 출력은 **`LOCAL OPTIONAL VALIDATION — NON-OFFICIAL, NON-REPORTABLE`**
배너로 시작하고 JSON은 필수 warning field에 동일한 문구를 기록합니다. 이는
데이터셋 라이선스가 `NOASSERTION`인 외부 RouterBench 데이터를 대상으로 하는
network-free diagnostic이며, SK텔레콤 데이터나 공식 대회 점수, 보고 가능한 대회
근거가 아닙니다. 동의 플래그는 필수입니다.

선택은 결정론적이며 행 내용과 독립적입니다. 고정된 7개 normalized domain마다
고정 revision, domain, `sample_id`를 framing한 digest로 행의 순위를 매깁니다.
처음 64개는 calibration pool, 다음 8개는 evaluation pool이며 evaluation은 다시
원본 순서로 복원됩니다. 따라서 calibration 448행과 공유 evaluation scope 56행을
사용하며 행 grain은 `sample_id`입니다. 모델별 호출 전 견적은 해당 모델의
calibration pool에서만 관측한 실현 비용의 최댓값으로 고정하고, 학습을 시작하기 전에
모든 evaluation 비용이 이 견적 이하인지 사전 검사합니다.

진단용 tier budget 3개는 정렬한 모델 견적의 최솟값·중앙값·최댓값으로 기계적으로
선택하고 가중치 `0.5`, `0.3`, `0.2`를 사용합니다. 이 값은 공식 budget tier가
아니며 해당 비용 값 자체도 출력하지 않습니다. 표면 특징만 쓰는 bilinear 정책
(`bge-m3` 미사용)의 품질 predictor 학습, lambda tuning, learned replay,
domain-table baseline에는 nested LODO를 적용하고, tier당 후보 32개로 제한한 명시적
approximate lambda search를 사용합니다. 반면 quote와 tier calibration은 7개 domain을
모두 포함하는 별도의 전역 calibration pool을 사용하므로 end-to-end domain-shift
주장이 아닙니다. 학습 정책과 베이스라인 6종은 모두 같은 56행에서 replay됩니다.

일반 출력과 `--json` 출력은 집계 provenance, 구조, 설정, 완료 근거만 노출합니다.
prompt/output 문자열, sample ID, 행별 결정, 성능·실현 비용·oracle gap 결과는 숨깁니다.
validator는 변환 데이터셋, 예측, 학습 artifact, 결과 파일을 쓰지 않습니다. benchmark
orchestration 전 외부 sample ID를 결정론적 로컬 surrogate로 바꾸고, CLI 실패 시 예외
상세와 traceback도 출력하지 않습니다. 리다이렉트한 출력을 포함한 RouterBench 유래
artifact를 커밋하지 마십시오. domain 불균형과 upstream evaluator의 이질성은 중요한
제약으로 남으므로, 이 제한된 로컬 diagnostic은 RouterBench 논문 재현이 아닙니다.

### bge-m3(계획, local-only)

임베딩 계약은 `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`(MIT)을 고정합니다. 가중치는 동봉하지
않고 런타임 downloader도 없습니다. 계획된 provider는 미리 준비한 로컬 경로만 받고,
`HF_HUB_OFFLINE=1`에서 Hub ID를 조회하지 않고 즉시 실패하도록 구현합니다.
최대 1,036개 전체 특징 학습은 위 prepared-session과 3플랫폼 release 검사가 끝날
때까지 gate로 남습니다. 실험적 one-solve C11 candidate는 전체 nested 실험을
실행했다는 근거가 아니며, 임베딩 차원을 조용히 투영해 버리지 않습니다.

SK텔레콤 대회 데이터도 라이선스와 재배포 조건을 서면 확인하기 전까지 포함하지 않습니다.

## 개발 검증

```bash
make install-dev PYTHON=python
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 pytest
tierroute route "offline smoke" --tier fast
tierroute evaluate
tierroute benchmark --budget-scope per-query
tierroute compare-predictors --budget-scope per-query
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

외부 데이터가 필요 없는 잠금 재현 경로를 두 가지로 제공합니다.

```bash
make reproduce-inference PYTHON=python  # 빠른 경로: 설치본 라우팅·평가
make reproduce-training PYTHON=python   # 전체: 검사·학습·benchmark·comparison·demo
```

두 경로 모두 비어 있는 임시 Hugging Face cache를 만들고 offline mode를 강제합니다.
빠른 경로는 `tierroute train`, `tierroute benchmark`, `tierroute
compare-predictors`, `tierroute demo`, 모든 bilinear/lambda-policy fitting을 건너뛰고
설치된 합성 predictor, artifact 로드, 라우팅, 베이스라인 6종 평가를 실행합니다.
평가는 필수 outer-training domain table만 맞추며 학습 predictor는 맞추지 않습니다.
전체 경로는 lint, SPDX, test, license, 설치 검사에 이어 training smoke에서 합성
predictor/policy artifact를 학습·소비하고 nested-LODO benchmark, paired predictor
추정, 학습 기반 3단계 demo를 실행합니다. 따라서 benchmark·comparison·showcase
fitting은 `training-smoke`와 `reproduce-training`에서만 실행되며 inference
경로에는 들어가지 않습니다.
`make reproduce`는 전체 경로의 호환 alias입니다. 이 target들은 검토된 exact 개발
lock을 설치하지만 모든 무관한 기존 package까지 제거하지는 않습니다. 무관한 package가
재현 주장을 오염하지 않도록 반드시 새 전용 가상환경에서 시작합니다.

CI는 lint, 테스트, 의존성 없는 wheel
설치, 두 CLI smoke 경로, offline mode, 의존성 라이선스 gate를 검사합니다. GPL 계열
의존성은 허용하지 않습니다. 기여·컴플라이언스 절차는
[CONTRIBUTING.md](CONTRIBUTING.md), 의존성 목록은 [SBOM.md](SBOM.md), 실제 wheel
내용을 기준으로 한 승인·거부 기록은
[docs/dependency-license-audit.md](docs/dependency-license-audit.md)를 참고하세요.

핵심 연구의 원문 검토, 선행연구 비교, 구현 완료·계획·공식 확인 대기 주장의 정확한
경계는 [docs/literature-and-novelty.md](docs/literature-and-novelty.md)에 기록합니다.
보고서나 발표에서 성능·OOD·novelty 문구를 재사용하기 전에 확인하세요.

주장 상태를 강제하는 5페이지 제출 구조, 아키텍처 도식 원본, 수치별 증거 원장,
최종 렌더 점검표는
[docs/submission-report-outline.md](docs/submission-report-outline.md)에 유지합니다.
그 문서의 자리표시는 대회 결과가 아니며 합성 데모 값으로 채우면 안 됩니다.

개발 보조 도구의 실질적 사용 범위, 증거의 한계, 사람 검토 상태는
[docs/ai-assistance-audit.md](docs/ai-assistance-audit.md)에 기록합니다. 핵심 불변식별
설명·실패 경로 검토와 8개 임시 변이 실습, 참가자 서명 표는
[한국어 유지관리자 실행 워크시트](docs/maintainer-explainability.ko.md)에 있습니다.
[영문 원본](docs/maintainer-explainability.md)도 같은 경계와 서명 표를 유지합니다. CI와 AI
에이전트 리뷰는 자동화 증거이며 참가자의 사람 검토 서명을 대신하지 않습니다.

## Open questions

공식 답변 전까지 다음 결정은 어댑터 또는 설정에만 둡니다.

1. tier 예산은 쿼리마다 초기화되는가, 정렬된 스트림 전체에 누적되는가? 라우터에
   공개되는 호출 이력 필드는 정확히 무엇인가?
2. 공식 시뮬레이터가 순차 다중 호출과 기존 출력 선택을 허용하는가? 확인 전에는
   cascade를 범위에 넣지 않습니다.
3. SK텔레콤 데이터의 라이선스·재배포 조건과 공식 Fast/Balanced/Premium 가중치,
   비용 단위, 비공개 데이터 schema, 채점 세부사항은 무엇인가? 라이선스를 서면
   확인하기 전 SK텔레콤 데이터는 커밋하지 않습니다.
4. 기대 비용을 기준으로 무작위 혼합하는 정책이 허용되는가? 답변에 따라 필수 6종이
   아닌 RouterBench Zero 정책을 추가 비교 대상으로 쓸 수 있는지가 결정됩니다.
5. tierroute가 학습한 ridge/bilinear+isotonic 예측기 artifact를 붙임2 유형3 자체 개발
   모델로 신고해야 하는가? 외부 모델 파인튜닝은 아니며, 최종 명세 전에 사무국의
   서면 해석을 받습니다.

## 라이선스

프로젝트가 작성한 코드와 문서는 [Apache-2.0](LICENSE)입니다. 소스와 문서에는 SPDX
식별자를 둡니다. 제3자 데이터·모델 자산은 각자의 조건을 유지하며 tierroute 라이선스로
재허가되지 않습니다.
