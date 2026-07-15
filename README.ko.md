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
bilinear 학습·exact tier-lambda 튜닝·엄격한 예측기/정책 artifact·외부 데이터가
필요 없는 데모가 구현되어 있습니다. CLI는 모델을 **선택**할 뿐 실제 LLM을
호출하거나 답변을 생성하지 않습니다.

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
결정론적인 bounded bottom-hash 표본과 최솟값·최댓값을 남기고, 그 root들로
경계·중간값·tail을 만든 뒤 최종 결과를 최대 257개로 rank-spacing합니다. 이
bounded-memory 탐색은 근사이며
`exhaustive: false`로 표시됩니다. 완전한 후보 개수는 알 수 없으므로 `null`로
기록하고, 탐색 strategy와 관측한 breakpoint 발생 횟수를 함께 남깁니다. 전체 exact
유한 집합을 모두 materialize·평가하려면 `--exhaustive-lambda-search`를 사용합니다.
선택된 lambda 자체는 언제나 정확한 분자/분모로 유지됩니다.

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
않은 큰 작업이 cubic 참조 경로에 들어가기 전에 즉시 실패시킵니다.

## 현재 구현 범위

- `Decimal` 기반 정확한 비용 계산과 `RouterState`/`RouterAction` 타입 계약
- 쿼리당·누적 예산 ledger 교체 구조(공식 범위 확정 전 데모는 예시용 쿼리당 한도)
- exact rational utility, 불변 tier별 schedule, 완전 exhaustive breakpoint 탐색 또는
  명시적으로 표시한 bounded-memory 근사 탐색을 쓰는 one-shot lambda 라우팅과
  재현 가능한 베이스라인 6종
- full-information offline replay: 선택된 로그 출력을 재생하기 전까지 정답 품질과
  미호출 출력은 `RouterState`에 노출하지 않음
- log-scaled 길이·코드/수식·프롬프트 유래 domain tag의 fitted schema, 프로젝트가
  작성한 결정론적 centered-ridge, inner-LODO out-of-fold 예측, 모델별 독립
  isotonic calibration
- 엄격히 검증하는 canonical JSON 예측기 artifact. 예측기 로더는 pickle을 받지 않고,
  batch 예측은 프롬프트 batch를 한 번만 vectorize/embed
- canonical 정책 artifact는 정확한 predictor hash, 학습/지표에 관련된 replay
  내용과 순서, OOF 예측 hash, tier 명세, ledger 식별자, 남긴 후보 탐색 근거를 함께 결합
- 모든 outer domain을 predictor fitting·calibration·lambda tuning에서 제외하는 true
  nested LODO orchestration
- tier 가중 품질, oracle gap 회수율, 결정론적 leave-one-domain-out(LODO). random
  split 도우미는 의도적으로 제공하지 않음
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
라우터 상태에서 제외합니다. 배포 불가한 oracle만 명시적인 평가 전용 경계를 통해
비공개 example key를 받습니다.

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

oracle gap 회수율은 always-cheapest에서 예산 내 oracle까지의 가중 품질 간격 중
라우터가 회수한 비율입니다.

```text
sum_t w_t * (Q_router,t - Q_cheapest,t)
-------------------------------------------------
sum_t w_t * (Q_oracle,t - Q_cheapest,t)
```

oracle과 cheapest가 같으면 정의되지 않으며 음수도 그대로 보존합니다.

| 베이스라인 | 선택 규칙 |
| --- | --- |
| `always-cheapest` | 최저 비용, 동률이면 모델 ID 순 |
| `always-premium` | 지정 premium 모델. 낮은 tier에서는 예산 위반 가능 |
| `random` | 예산 내 모델 중 seed 기반·순서 독립 선택 |
| `length-heuristic` | 긴 프롬프트 또는 코드/수식이면 예산 내 strong 모델 |
| `oracle` | 정답을 사용하는 쿼리별 예산 내 품질 상한 |
| `domain-best-table` | 학습 도메인의 tier별 평균 품질표, 미등록 도메인은 cheapest |

Lambda 튜닝은 proxy loss가 아니라 이 실제 지표를 직접 최대화합니다. 각 프롬프트에서
모델 utility는 lambda의 affine 함수이므로 선택은 exact 품질/비용 교차점에서만 바뀔 수
있습니다. 완전 탐색은 모든 경계, 인접 경계 사이 열린 구간의 대표값 하나, 마지막 경계
뒤 tail 값 하나를 검사합니다. 각 후보는 `OfflineSimulator`로 replay하므로 infeasible
후보는 선택될 수 없습니다. Tier ledger가 서로 독립이고 모든 가중치가 양수이므로 tier별
최적화는 남긴 후보 집합의 Cartesian joint search와 정확히 같습니다. 이 결과가 전체
exact 유한 joint 최적임을 보장하는 경로는 uncapped 탐색뿐이며, 기본 bounded 탐색은
근사로 남습니다. 증명·동률 규칙·누수 경계는
[docs/lambda-tuning.md](docs/lambda-tuning.md)에 설명합니다.

`tierroute evaluate`는 작은 합성 데이터에서 end-to-end 배선만 확인하기 위해 같은
샘플로 domain table을 맞추며 출력에도 이를 경고합니다. 보고 가능한 실험은 각 LODO
fold의 학습 부분에서만 적합하고 완전히 제외한 도메인에서 평가해야 합니다. 데이터셋
domain은 어댑터가 호출 전에 관찰 가능한 label로 명시한 경우에만 `RouterState`에
전달되며, split 전용 label은 비공개로 유지합니다.

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

`make reproduce`는 정확한 개발 lock을 설치하고 동봉 데이터 전체 파이프라인을
실행하며 학습과 artifact 라우팅도 포함합니다. CI는 lint, 테스트, 의존성 없는 wheel
설치, 두 CLI smoke 경로, offline mode, 의존성 라이선스 gate를 검사합니다. GPL 계열
의존성은 허용하지 않습니다. 기여·컴플라이언스 절차는
[CONTRIBUTING.md](CONTRIBUTING.md), 의존성 목록은 [SBOM.md](SBOM.md)를 참고하세요.

## Open questions

공식 답변 전까지 다음 결정은 어댑터 또는 설정에만 둡니다.

1. tier 예산은 쿼리마다 초기화되는가, 정렬된 스트림 전체에 누적되는가? 라우터에
   공개되는 호출 이력 필드는 정확히 무엇인가?
2. 공식 시뮬레이터가 순차 다중 호출과 기존 출력 선택을 허용하는가? 확인 전에는
   cascade를 범위에 넣지 않습니다.
3. SK텔레콤 데이터의 라이선스·재배포 조건과 공식 Fast/Balanced/Premium 가중치는
   무엇인가? 라이선스를 서면 확인하기 전 SK텔레콤 데이터는 커밋하지 않습니다.

## 라이선스

프로젝트가 작성한 코드와 문서는 [Apache-2.0](LICENSE)입니다. 소스와 문서에는 SPDX
식별자를 둡니다. 제3자 데이터·모델 자산은 각자의 조건을 유지하며 tierroute 라이선스로
재허가되지 않습니다.
