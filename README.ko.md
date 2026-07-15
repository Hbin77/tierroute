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
**“Efficient LLM Routing Challenge”** 출품을 위해 개발 중입니다. 현재는 pre-alpha
W1 스캐폴드로, 라우팅 계약·replay 시뮬레이터·베이스라인 6종·지표·외부 데이터가
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

`python -m tierroute`도 같은 진입점입니다. `route`와 `evaluate`에는 `--json`을
쓸 수 있고, 버전이 명시된 호환 replay JSON을 평가 입력으로 지정할 수 있습니다.

```bash
tierroute route "이 Python 함수를 디버그해 줘" --tier balanced --json
tierroute evaluate --data src/tierroute/data/synthetic.json --json
HF_HUB_OFFLINE=1 tierroute demo
```

동봉된 프롬프트·비용·출력·예측 품질·scorecard 수치는 프로젝트가 만든 **합성
smoke-test 값**입니다. 벤치마크 결과, 실제 모델 비교, 대회 점수가 아닙니다.

## 현재 구현 범위

- `Decimal` 기반 정확한 비용 계산과 `RouterState`/`RouterAction` 타입 계약
- 쿼리당·누적 예산 ledger 교체 구조(공식 범위 확정 전 데모는 예시용 쿼리당 한도)
- one-shot lambda 라우팅과 재현 가능한 베이스라인 6종
- full-information offline replay: 선택된 로그 출력을 재생하기 전까지 정답 품질과
  미호출 출력은 `RouterState`에 노출하지 않음
- 길이·행·코드/수식·도메인 표면 특징, bilinear 예측기 추론 형태, 의존성 없는
  isotonic calibration
- tier 가중 품질, oracle gap 회수율, 결정론적 leave-one-domain-out(LODO). random
  split 도우미는 의도적으로 제공하지 않음
- 엄격한 JSON 로더와 opt-in 방식의 고정 RouterBench 경계 어댑터

다운로드 없는 CLI는 설명 가능한 합성 데모 예측기를 사용합니다. 로컬 `bge-m3`
임베딩 백엔드, 학습 파이프라인, GBM/bilinear 비교 실험은 계획 항목이며 완료된
기능으로 주장하지 않습니다.

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
prompt ─> 표면 특징 ─> 품질 예측기 ─> 정책 <──────────── 예산 ledger
                                      │
                              CallModel / SelectOutput

core/        상태·행동·모델·검증 계약
features/    오프라인 표면 특징과 로컬 임베딩 계약
predictors/  예측기 protocol, bilinear 형태, isotonic calibration
policies/    one-shot lambda 정책과 필수 베이스라인
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

`tierroute evaluate`는 작은 합성 데이터에서 end-to-end 배선만 확인하기 위해 같은
샘플로 domain table을 맞추며 출력에도 이를 경고합니다. 보고 가능한 실험은 각 LODO
fold의 학습 부분에서만 적합하고 완전히 제외한 도메인에서 평가해야 합니다. 데이터셋
domain은 어댑터가 호출 전에 관찰 가능한 label로 명시한 경우에만 `RouterState`에
전달되며, split 전용 label은 비공개로 유지합니다.

## 데이터와 모델 자산

런타임 라우팅과 평가는 네트워크를 호출하지 않습니다. 다운로드는 명시적인 별도 준비
단계여야 하며 Hugging Face 자동 fallback은 금지합니다. 다운로드 데이터와 모델
가중치는 Git에서 제외하며, 재배포 라이선스가 확인되지 않으면 커밋하지 않습니다.

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
```

`make reproduce`는 정확한 개발 lock을 설치하고 동봉 데이터 전체 파이프라인을
실행합니다. CI는 lint, 테스트, CLI smoke, offline mode, 의존성 라이선스 gate를 검사합니다. GPL
계열 의존성은 허용하지 않습니다. 기여·컴플라이언스 절차는
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
