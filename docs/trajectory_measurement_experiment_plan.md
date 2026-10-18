# A 궤적 측정 개선: 실험 설계 및 자동화 계획

작성: 2026-10-17 (Asia/Seoul)
상태: 2026-10-17 파일럿 구현 및 실행 완료. 결과는
`docs/trajectory_measurement_experiment_results.md`를 기준으로 한다.

> 실행 메모: E1은 원본 좌표 표적 측정과 1280px GMC를 결합한 진단이다.
> 원본 ROI만 잘라 처리하는 최종 최적화 구현은 아니며, 이번 결과가 회귀여서 제품
> 경로에는 적용하지 않았다. E5는 학습 모델이 아니라 과거 프레임 후보의 인과적
> 연속 확인 단계까지만 검증했다.

## 1. 브랜치와 보존 범위

- 새 브랜치: `codex/trajectory-measurement-experiments`.
- 기준: fetch한 `origin/dev`, `a240387b269f80460785bc305aea5eef6ad6d00c`.
- 작업 폴더: `/Users/yuchan/Documents/형주플젝/skydetect-ai-trajectory-experiments`.
- 기존 폴더: `/Users/yuchan/Documents/형주플젝/skydetect-ai-apart-fix-cvat-comparison`.
- 기존 HEAD: `de1f6d8`, 브랜치 `codex/apart-fix-cvat-comparison`.
- 기존 폴더의 미커밋 변경 11개는 그대로 유지했다. 새 폴더로 복사/커밋하지 않았다.
- 새 작업은 별도 Git worktree다. 기존 checkout, 실행 중인 서버, 데이터 경로를 바꾸지 않는다.
- 자동 실행 시 다른 checkout의 코드를 import하지 않도록 cwd와 Python 환경을 명시한다.
- 설계 이후 dev가 바뀌어도 실험 도중 자동 pull/merge하지 않는다. 새 기준선은 별도 실험이다.

## 2. 목적, 가설, 제외 범위

목적은 작은 비행체의 위치 오차와 운동 신호 왜곡을 줄이되, 누락이나 오재획득으로
좋아 보이는 점수를 만들지 않는 것이다. 화면에서 박스를 유지하는 것과 관측의 정확도를 구분한다.

가설은 여섯 가지이며, 아래 순서대로 검증한다.

1. 축소 과정에서 객체의 유효 픽셀 수가 줄어 위치 측정이 불안정해질 수 있다.
2. 외형 매칭의 정수 최고점/성긴 후보 격자가 시간별 좌표 변화에 영향을 줄 수 있다.
3. 응답의 모호성과 실제 시간 간격을 반영하면 고정 잡음 모델보다 나을 수 있다.
4. 재획득 후보를 여러 증거로 확인하면 다른 객체/배경으로의 이동을 줄일 수 있다.
5. 정렬한 짧은 시간 구간의 증거는 저대비 단일 프레임보다 유용할 수 있다.
6. 앞의 개선 이후에도 남는 잡음은 제한적인 오프라인 RTS로 줄일 수 있다.

가설은 확인된 원인이 아니다. 현재 4개 영상은 반복 분석한 개발 자료다.
Zero-Loss, 서브픽셀 정확도, 물리적 가속도 복원, 급기동 보존을 보장하지 않는다.
수동 초기 ROI가 있는 SOT가 대상이며 전체 화면 자동 초기 탐지, 새 SOT 백본 학습,
GAN/IMM/새 C 학습, API/UI 재설계와 public 스키마 변경은 이번 범위 밖이다.

## 3. 코드에서 확인한 출발점

| 위치 | 확인한 사실 | 설계 |
| --- | --- | --- |
| `tracking_video_io.py` | 폭 제한으로 프레임을 축소한다. | 원본과 처리 영상의 객체 크기를 먼저 측정한다. |
| `manual_roi_tracker.py` | 프레임들을 list로 로드한다. | 원본 전체 영상을 메모리에 추가 적재하지 않는다. |
| `online_appearance.py` | 후보 격자 최소 간격 3px, template 응답은 minMaxLoc 최고점이다. | 국소 정밀 탐색과 peak refinement를 분리한다. |
| `manual_roi_tracker.py` | KLT는 이미 소수 좌표를 사용한다. contrast 보정에는 정수 최고점이 있다. | A 전체가 정수 추적이라고 설명하지 않는다. |
| `foreground_motion.py` | Kalman은 고정 F/Q/R이다. | timestamp 기반 별도 출력 정제부터 시험한다. |
| `manual_roi_tracker.py` | 관측을 기록한 뒤 Kalman correct를 호출하고 반환 위치를 버린다. | 내부 예측용 KF와 B 전달 좌표 정제를 구분한다. |
| `tracking_adapter.py` | visible만 export, 처리 해상도 포함, 정규화 값 clamp. | 관측 계약 유지, 범위 초과를 별도 감사한다. |
| `feature_core.py` | 실제 research.features를 호출하며 계약 ID와 좌표 정책을 반환한다. | B 공식을 재구현하거나 실험 도중 바꾸지 않는다. |
| `research/tracking_evaluation.py` | dev에는 기본 single/dataset 채점만 있다. | 미커밋 확장 기능을 검토 후 이식한다. |

파일명은 `ai_server/services/` 기준(명시한 research 제외).

## 4. 데이터와 초기 조건 동결

기존 자료 진입점:

- `/Users/yuchan/Documents/형주플젝/visual_checks/local_eval_20261006/inputs/`
- `/Users/yuchan/Documents/형주플젝/visual_checks/local_eval_20261006/gt_handoff/manifest.json`
- 기존 보고서: 같은 폴더의 `temporal_b_results/`, `motion_signals/`.

| sample_id | 클래스 | XML track | 독립 영상 그룹 |
| --- | --- | --- | --- |
| bird_2_drone | drone | 0 | bird_2 |
| drone_1_drone | drone | 0 | drone_1 |
| drone_1_bird_1 | bird | 1 | drone_1 |
| drone_1_bird_2 | bird | 2 | drone_1 |
| drone_5 | drone | 0~17의 검증된 동일 객체 조각 | drone_5 |
| drone_6 | drone | 0 | drone_6 |

`bird_2` 영상은 새 폴더/파일명이 잘못된 드론이다. drone_1의 bird_2와 다르다.
drone_5 조각은 시간 중복/객체 동일성을 다시 검사하고 합친다.
4개 영상/6개 객체이지 독립 표본 6개가 아니다. 두 새는 같은 영상 출처다.

### 매니페스트 필수 정보

아래는 구현할 runner의 입력 계약이며 기존 sample.json을 대체한다고 가정하지 않는다.

- experiment_id, schema_version, base/code/evaluator SHA, dirty diff hash, 환경 lock/hash.
- video/XML/기존 산출물 경로와 SHA256, source_group_id, label 및 확인 근거.
- XML track ID, 원본/잘린 영상 대응, 명시적 frame offset, timestamp 정책.
- 초기 ROI 원본 좌표 xywh, 초기 프레임, ROI 출처와 당시 tuning 전체 값.
- 고정 평가 시작/종료 프레임, GT 관측 가능성, 처리 시간 제한과 종료 이유.
- 원본/처리 크기, 좌표 변환, CMC 기준 프레임/행렬/해시, B 계약/설정, C 모델 해시.
- variant별 설정, seed, template 갱신 정책, 출력 루트, 시간/메모리 제한.

과거 수동 ROI를 metadata/원본 요청에서 복원한다. 없으면 `blocked_missing_roi`로 중단한다.
GT bbox로 조용히 대체하지 않는다. GT 초기 bbox 실험은 별도 `oracle_init` 진단으로만 수행한다.
과거 conf/튜닝이 불명확하면 기본값을 과거 값으로 간주하지 않는다.
기존 전체 GT/CMC window JSON은 참조 자료이고, 새 variant의 CMC로 바뀐 평가에 그대로 재사용하지 않는다.

평가 창은 GT와 계획된 입력 구간으로 미리 고정한다. A가 LOST/EXITED/일찍 종료해도
그 때문에 분모를 줄이지 않는다. GT에서 화면 밖/관측 불가인 프레임은 별도 집계한다.
평균 FPS 기반 기존 timestamp를 보존해 우선 비교한다. VFR/PTS 불일치가 발견되면
기준선과 후보 모두 같은 정책으로 재생성하는 별도 실험으로 분리한다.

## 5. 단계 P0: 평가 도구와 기준선 복구 (개선 전에 필수)

### P0a. 평가 기능 이식

기존 폴더에서 파일별 hash와 tracked diff/untracked 목록을 기록하고 필요한 코드를 이식한다.
새 dev의 B 코드를 이전 파일 전체로 덮어쓰지 않는다.

- `research/tracking_evaluation.py`: 상대/시간 오차, 실제 B 비교 연결, GT/A 비교 영상.
- `research/tracking_feature_comparison.py`: 실제 B 호출, GT/full/matched 입력, CMC 검증.
- `research/tracking_signal_comparison.py`: 실제 B 중간 신호와 경계 분석.
- `research/features.py`: 진단용 return_signals 변경만 최신 코드에 맞춰 이식.
- 관련 테스트 4종 및 `docs/tracking_evaluation.md`; GT export는 필요한 경우만 별도 이식.

기본 feature 결과가 진단 옵션 on/off에서 동일한지 테스트한다. GT만 얹은 영상이 아니라
A 좌표/GT/누락/시간/범례가 실제로 함께 표시되는지 대표 프레임을 확인한다.
평가 포팅 커밋과 추적 개선 커밋을 분리한다. 이식 중 원본 변경을 발견하면 hash 불일치로 멈춘다.

### P0b. 기준선 두 종류

- `H0`: 기존 저장 A와 보고서. 과거 결과 재현용이며 새 dev의 개선 전 수치로 대체하지 않는다.
- `D0`: 고정 dev SHA와 복원한 동일 ROI/tuning으로 새로 실행한 기준선.
- `D0`를 같은 환경에서 2회 실행한다. 숫자 좌표/상태 차이를 검사하고 ID/생성시각 등 비결정 메타데이터는 제외한다.
- 불일치가 있으면 seed, OpenCV thread/환경을 기록하고 해결 또는 반복 변동폭을 정하기 전 승자를 정하지 않는다.
- CMC 행렬, 원시 관측, state/source 변경, 템플릿 갱신, clamp 이벤트를 진단 산출물로 남긴다.
- 진단 계측만 추가한 실행이 D0와 같은 결과인지 회귀 테스트한다.

H0와 D0가 다른 것은 최신 dev 변경의 영향일 수 있다. E1 이후의 효과는 D0 대비로 평가한다.
P0 완료 조건: 6개 모두 실행 또는 구체적인 blocked 이유, 동일 단위 보고서, 실제 비교 영상,
B 계약 검증, 재현성 결과. 실행/파싱 실패를 점수 0 또는 좋은 표본만의 평균으로 숨기지 않는다.

## 6. 단계별 실험 매트릭스

아래 variant ID는 설계 이름이다. 아직 구현/실행 가능한 옵션이 아니다.
모든 조합을 곱하지 않고 단독 효과를 본 뒤 통과 후보만 결합한다.

| ID | 변경 | 대조군 | 핵심 판정 |
| --- | --- | --- | --- |
| E1 | 원본 해상도 ROI 측정 | D0 | 작은 객체 정보 손실과 비용 |
| E2a | appearance 국소 1px 재탐색 | D0 | 성긴 후보 격자 영향 |
| E2b | E2a + 제한된 subpixel peak fit | E2a | 소수 위치 추정의 추가 효과 |
| E2c | contrast center 보정의 subpixel화 | E2b | 다른 위치 보정 경로의 영향 |
| E12 | 통과한 E1 + E2 | D0 및 각 단독 | 상호작용/회귀 |
| E3a | 동일 관측의 timestamp KF 출력 | raw 관측 | 시간 모델/출력 필터 효과 |
| E3b | E3a + 모호성 기반 R | E3a | 적응 관측 잡음 효과 |
| E4 | 재획득 확인 | 고정된 이전 승인 후보 | 오재획득/누락 절충 |
| E5 | 짧은 인과적 시간 증거 누적 | 동일 이전 후보 | 저대비에서의 추가 효과 |
| E6 | segment별 RTS | 같은 관측/Q/R의 KF | 미래 관측을 이용한 후처리 효과 |

단독 실험에 실패한 기능을 누적 조합에 몰래 포함하지 않는다. E3~E6는 1차 실행 범위 밖이다.

### E1: 해상도 정보 보존

1. GT 폭/높이를 원본/처리 좌표에서 프레임별 측정해 <5, 5~10, 10~20, >=20px로 요약한다.
2. 객체 ROI는 원본 픽셀에서 crop하고 CMC는 기존 처리 해상도에서 유지한다.
3. bbox, LK window, 검색 반경, 잡음 단위 등을 좌표 배율에 맞게 변환한다.
4. 같은 화면 영역을 비교하고 후보 ROI 확대만으로 탐색 범위가 넓어지는 교란을 막는다.
5. 원본이 이미 처리 폭 이하이면 no-op 대조로 기록한다. 개선이 나올 것이라 가정하지 않는다.

축소 S: source->processed, crop C: source->crop, CMC H: processed->reference라면
crop 위치의 기준 좌표는 `H * S * inverse(C) * p_crop`이다. 단순 crop이면 C는 이동 변환이다.
S는 실제 디코딩 크기의 x/y 배율로 계산한다. export 해상도는 고정된 전체 처리 평면이고
ROI 크기가 아니다. source, processed, crop, reference를 별도 타입/변환 기록으로 관리한다.

1차 원인 분리에서는 D0의 저장 CMC를 모든 후보에 replay한다. 이는 실시간 deploy 점수가
아닌 통제 실험이다. 통과 후보는 자체 CMC를 재추정하는 end-to-end 재실험을 반드시 한다.
GT는 어떤 경우에도 CMC 추정/후보 선택에 사용하지 않는다.

원본 프레임 list를 추가하지 않고 순차 디코딩, 현재 프레임+제한된 buffer를 사용한다.
전체 원본 추적은 자원 점검 후 짧은 진단용만 허용하며 E1의 핵심 구현으로 삼지 않는다.

### E2: 국소 위치 추정

- coarse 후보를 정한 뒤 bounded neighborhood에서 1px로 재탐색한다.
- 선택 응답 주위 3x3 peak fit은 음의 곡률, 유한 값, 내부 최고점 조건을 검사한다.
- 보정량은 축별 최대 0.5 grid pixel로 제한하고 범위 밖/평평/경계 peak는 정수 결과로 fallback한다.
- 후보 최고점, 배제 반경 밖 두 번째 peak, PSR/APCE, fit 곡률, 보정량, fallback 사유를 저장한다.
- 소수 좌표로 descriptor/patch를 실제 sampling하는 경로를 검증한다. 마지막 출력 숫자만 보간하지 않는다.
- KLT, template update, score threshold를 동시에 바꾸지 않는다.

합성 테스트는 분석적으로 생성한 이동 표적과 실제 patch 이동을 구분한다.
0/0.25/0.5/0.75px 이동, 3/5/10/20px 크기, 저대비, blur, distractor, 경계/평평한 응답을 포함한다.
합성 patch warp 자체의 보간 오차를 실제 서브픽셀 정확도로 주장하지 않는다.
GT 수작업 오차가 있으므로 실제 영상에서 작은 차이를 과도하게 해석하지 않는다.

### E3: 관측 불확실성과 출력 정제

우선 저장된 동일 관측을 재생하는 독립 정제기를 만든다. 검색용 KF는 바꾸지 않는다.
상태는 [x,y,vx,vy], F(dt)는 실제 timestamp, Q(dt)는 명시한 일정 속도/가속 잡음 모델이다.
R은 양의 하한/상한을 가진 위치 공분산이며 단위는 처리 평면 px^2다.
CMC 불확실성도 진단하지만 비슷한 시각 증거를 독립 관측으로 중복 보정하지 않는다.

E3a는 고정 R, E3b는 응답 모호성/FB error/source agreement를 이용한 R이다.
confidence를 곧바로 확률 또는 1/R로 해석하지 않는다. Q/R 설정 수는 사전 제한한다.
초기 속도는 미래 GT를 쓰지 않고 추정하며 초기 공분산/워밍업을 기록한다.
초기 경계 점을 삭제해서 점수를 높이지 않는다. innovation 하나만으로 급기동을 제거하지 않는다.

1차 E3는 관측 accept/reject를 바꾸지 않는 출력 비교다. 안정화 좌표는 실제 관측 프레임에만
저장한다. 후보 거부/검색 경로 피드백은 별도 end-to-end variant로 취급한다.
raw/filtered TrackSequence를 별도 파일로 유지하고 public 필드 의미를 몰래 변경하지 않는다.

### E4: 재획득

- 내부 SUSPECT/REACQUIRING에서 템플릿 오염을 막는다. public lifecycle은 유지한다.
- covariance와 lost duration으로 검색 범위를 제한 확장한다.
- appearance, 방향/크기, 다중 후보 차이, 연속 프레임 확인으로 후보를 검증한다.
- 확정 이전 후보는 observation history에 넣지 않는다. 과거 후보를 소급 삽입하지 않는 정책부터 시작한다.
- motion source라고 모두 재획득 이벤트로 분류하지 않는다. 실제 LOST->관측 확정 사건을 센다.
- 정답 재획득 시간, 잘못된 객체/배경 고정, 미복구, 확인 지연을 각각 보고한다.
- 실제 가림 자료가 부족하면 인위적 가림 테스트와 실제 테스트를 명확히 분리한다.

### E5: 시간 증거

과거 프레임만 사용하는 정렬된 응답 누적을 우선한다. 중심 시각을 바꾸지 않는다.
예: 0.1/0.2초의 두 horizon을 FPS별 프레임 수로 변환하고 최대 9프레임 buffer로 제한한다.
정렬 실패/오류 전파를 기록하고 누적을 중지한다. 정렬 없는 이미지 평균은 사용하지 않는다.
새 ConvLSTM 학습은 포함하지 않는다. 빠른 방향 전환에서 지연도 측정한다.

### E6: RTS

미래 관측을 쓰는 오프라인 실험으로 표시한다. KF와 같은 관측/Q/R을 사용한다.
긴 gap, CMC 불연속, 불확실 재획득에서 segment를 나누며 gap 임계값을 사전 고정한다.
관측 없는 프레임을 새로운 측정으로 export하지 않는다. segment 경계 포함 전체 지표를 보고한다.
속력 변화/회전 이벤트의 진폭 손실, 시각 이동과 위치 오차를 함께 확인한다.

## 7. 평가 기준과 비교 좌표

### 공간

- 원본 좌표 GT/A 거리 및 기준 폭 1280에 환산한 Median/P95. 종횡비를 보존한다.
- 매 프레임 거리/GT bbox 폭, 거리/GT bbox 대각선을 계산한 뒤 Median/P95.
- 객체 크기는 동일 평면으로 변환한다. 0 크기, outside, unknown을 제외 사유와 함께 집계한다.
- 5px 이하 비율은 보조 지표이며 객체 크기와 무관한 성공의 유일한 기준으로 삼지 않는다.
- 비교 영상은 원본 화면 좌표를 사용한다. CMC 좌표는 필요 시 inverse transform한다.

### 시간과 B

- e(t)=A(t)-GT(t)의 연속 프레임 차이와 시간 정규화한 벡터 오차.
- GT/A 경로 길이 비율은 동일한 유효 연속 pair만 비교하고 정지 GT는 undefined.
- pair coverage와 누락을 함께 보고 gap을 가로질러 임의 차분하지 않는다.
- B는 실제 런타임/계약/기준 폭 1920으로 계산한다. 공간 보고서의 1280과 단위를 섞지 않는다.
- GT 전체 고정 창 / A와 동일 관측 시점의 GT / A를 모두 계산한다.
- 전체 GT 대비 결과는 누락도 포함하고, matched 비교는 표본 선택 효과가 있으므로 단독 채택 근거가 아니다.
- 9개 B 특징의 GT/A 실제 값, 절대 차이, 분모가 충분한 경우 상대 차이를 보고한다.
- B 가속도는 현재 정의의 속력 변화량이다. 벡터 가속도 크기라고 바꾸어 설명하지 않는다.
- 시작/끝 필터 경계, 내부, 실제 재획득/누락 주변을 별도 진단하되 전체 점수를 대체하지 않는다.
- 가속도 P95가 비슷한 것만으로 시간 신호가 같다고 판단하지 않는다.

### C와 관측 신뢰성

- RF 모델/feature contract/config 일치 확인 후 고정 모델로 GT/D0/variant를 비교한다.
- classifier 직접 출력과 실제 quality/rule gate 결과를 분리한다. 거부/uncertain을 삭제하지 않는다.
- MiniRocket은 해당 모델, 전처리, 독립 source split 증빙이 확보될 때 별도 동일-input 비교만 추가한다.
- 새 모델 다운로드/학습/threshold tuning을 A 실험과 섞지 않는다.
- 6개 객체의 클래스별 결과는 사례 보고이지 일반화 정확도/유의성 검증이 아니다.
- C 개선이 없더라도 원인을 분리하며 C가 맞히도록 A 좌표를 조작하지 않는다.

### CMC, 범위 초과, 커버리지

GT에는 후보에 대응하는 전체 affine 변환을 적용한다. 점별 이동량만 더하지 않는다.
CMC replay 수치 재현은 실제 카메라 운동의 정확성 검증이 아니다.
기존 clamp는 D0에서 보존하고 count/원좌표를 저장한다. 이 동작 변경은 별도 variant로 한다.
모든 후보에서 범위 초과를 숨기지 않고 false position/거부를 집계한다.
고정 GT 관측 가능 창 대비 유효 관측률, 최장 누락, B 전달률, C 판정 가능률을 함께 보고한다.
오재획득은 새로운 GT identity/중심 근거와 육안 검수로 확정한다. 확인 전에는 suspected로 남긴다.

## 8. 채택 기준과 중단 기준

자동 runner는 `pass`, `reject`, `needs_review`, `blocked`를 구분한다.
아래 수치는 1차 파일럿의 개발 triage 기준 제안이며 통계적/논문 기반 보장값이 아니다.
실행 전 설정에 잠그고, 결과를 본 뒤 바꾸면 새 실험 ID로 재실행한다.

1. 계약/좌표/타임라인 테스트 실패, 결과 overwrite, GT가 runtime 추적에 유입되면 즉시 blocked.
2. sample당 고정 창 관측률이 D0보다 1%p 넘게 감소하거나 최장 gap이 0.1초 넘게 늘면 자동 채택 금지.
3. 새 오재획득이 한 건이라도 확인되면 reject 또는 원인 수정 후 새 variant. 미검수 영상은 needs_review.
4. sample별 위치 P95가 `max(D0의 5%, 기준 폭 1280에서 0.25px)` 넘게 증가하면 needs_review.
5. 해당 단계의 사전 지정 주 지표가 영상 그룹 중앙값 기준 10% 이상 개선하면 후보로 올린다.
   E1/E2 주 지표는 위치 P95, E3/E6는 시간 정규화 오차 변화 P95, E4는 오재획득/복구시간,
   E5는 사전 저대비 구간의 관측률/위치 오차다. 분모 0/이벤트 없음은 N/A이지 pass가 아니다.
6. 작은 값의 비율 효과를 막기 위해 절대 차이를 병기하고, 위치/시간 지표 악화가 교환되면 자동 승자를 정하지 않는다.
7. 급기동 이벤트의 실제성/GT 불확실성, B 특징 회귀, C 오류 증가는 사람이 검토한다.
8. 관측이 없거나 B가 거부한 sample을 빼고 평균내지 않는다. 비교 가능 수와 실패 이유를 표시한다.

영상 내 객체를 먼저 요약하고 4개 영상 그룹을 동등 가중한다. 객체별 표도 유지한다.
프레임 수를 독립 표본 수로 간주하지 않는다. 4개 영상으로 유의한 일반화 개선을 주장하지 않는다.
최종 채택은 별도 출처의 미튜닝 영상으로 재검증한 뒤 판단한다.

## 9. 자동 실행 설계 (미구현)

새 `research/trajectory_experiments/`에 아래 책임을 분리한다.

| 모듈 | 책임 |
| --- | --- |
| manifest.py | 입력 계약, hash, ROI/시간/객체 매칭 검증 |
| runner.py / __main__.py | dry-run, 단계 선택, 제한/중단/resume |
| variants.py | 명시적 변경 목록, off이면 기존 경로 호출 |
| diagnostics.py | 관측/응답/CMC/리소스 기록 |
| compare.py | 기존 채점기와 실제 B/C 호출, 채택 기준 평가 |
| report.py | 비교 표/그래프/동일 시각 영상/실패 목록 |

추적 모듈에는 opt-in hook만 추가한다. 기본 API/UI/TrackSequence 동작은 유지한다.
평가 도구는 GT를 읽지만 추적 실행에는 video, ROI, tuning만 전달한다.
실험 config와 private absolute paths는 로컬 manifest로 분리하고 영상/모델/큰 산출물은 Git에 넣지 않는다.

계획된 CLI (아직 존재하지 않으므로 실행 명령으로 안내하지 말 것):

```text
python -m research.trajectory_experiments validate --manifest <local.json>
python -m research.trajectory_experiments run --manifest <local.json> --stage baseline --dry-run
python -m research.trajectory_experiments run --manifest <local.json> --stage baseline
python -m research.trajectory_experiments run --manifest <local.json> --stage measurement
python -m research.trajectory_experiments report --run-dir <run>
```

상태 전이: planned -> preflight -> running -> evaluated -> needs_review/completed/failed.
resume은 code/config/input/env/model hash가 모두 같은 완료 artifact에만 허용한다.
부분 출력은 임시 폴더에 쓰고 성공 시 complete marker를 만든다. 실패 로그는 남긴다.
파일명만 같은 오래된 캐시는 재사용하지 않는다. 하위 작업 실패를 숨긴 exit 0을 금지한다.

기본 자원안: worker 1, sample당 10분, 한 stage 120분, RSS 상한 4GiB.
실제 기준선이 이를 초과하면 제한을 우회하지 않고 이유/측정치를 보고한 뒤 config를 명시적으로 변경한다.
전체 4K 적재, 무제한 sweep, 백그라운드 자동 재실행, 원본 덮어쓰기를 금지한다.
종료 시 자식 프로세스/영상 writer를 닫는다. 네트워크/가중치 다운로드는 첫 배치에서 사용하지 않는다.

### 산출물

```text
<output>/<experiment_id>/
  manifest.resolved.json
  provenance.json
  preflight.json
  status.json
  baseline/<sample>/...
  variants/<variant>/<sample>/
    metadata.json
    trajectory.csv
    observations.jsonl
    camera_transforms.json
    track_sequence_raw.json
    track_sequence_filtered.json  # 정제를 수행한 경우만
    metrics.json
  evaluation/
    comparison.csv
    b_features.csv
    c_predictions.csv
    motion_signals/
    comparison_videos/
    report.html
    decision.json
```

비교 영상은 GT 파랑, D0 주황, 후보 초록으로 범례/프레임/원본 시각을 명시한다.
예측은 점선, 실제 관측은 실선 등으로 구분하고 없으면 표시하지 않는다.
3개 궤적이 겹치면 같은 프레임의 좌우 비교를 병행한다. 재생 길이/FPS/원본 매칭을 검증한다.
대표 시작/중간/누락/종료 프레임을 렌더링해 육안 확인한다.

## 10. 테스트와 구현 단위

1. 평가 포팅: 기존 tests + 새 dev contract 일치 + B 진단 on/off 수치 일치.
2. runner: missing ROI/path/hash mismatch/중복 frame/실패 artifact/resume/resource timeout.
3. 좌표: crop/resize/CMC round-trip, 비등방 배율, 화면 경계, source/processed 단위 혼동 방지.
4. E1: 원본=처리 no-op, ROI 외 이동, streaming buffer 상한, baseline replay CMC.
5. E2: 합성 fractional 이동, plateau/경계/복수 peak/NaN, fallback, off parity.
6. E3: dt=0/음수 거부, 불규칙 dt, PSD covariance, 정속/급회전/호버링, 초기 경계, gap 보존.
7. E4/E5: distractor/가림/모호한 후보, 동일 ID 확인, 미래 프레임 미사용.
8. E6: segment 경계와 관측 인덱스 보존, 동일 관측 KF 대비, 미래 사용 표시.
9. end-to-end: 실제 스키마 parse, processed 크기, frame/timestamp/conf 보존, GT/A overlay 확인.

각 단위는 테스트와 작은 smoke run 후 커밋 가능한 크기로 분리한다.
현재 브랜치에는 계획 문서만 있으며 위 테스트를 실행했다고 주장하지 않는다.

## 11. 1차 자동화 작업의 종료 지점

첫 위임은 P0a/P0b와 dry-run 가능한 runner까지만 구현/검증한다.
기준선과 필수 입력이 확보되면 다음 위임에서 E1/E2 단독 및 승인 조합을 실행한다.
E3~E6, 새 분류기 학습, API 기본값 변경, merge/push는 자동으로 진행하지 않는다.
최종 사용자 보고는 실행 수/실패 수, 객체별 전후 표, 비용, 비교 영상, 보류 사유와
다음에 검증할 가설을 포함한다. 개선이 없으면 그대로 보고한다.

## 12. 참고 아이디어의 범위

- Remote Sensing 2022, 14, 777: 상관 응답의 subpixel 위치와 관측 신뢰도 아이디어.
- OITrack, Remote Sensing 2025, 17, 2042: 불확실성/탐색 확대. 궤적 보완점을 실관측으로 수출하지 않는다.
- Sensors 2026, 26, 4639: 시간 문맥 아이디어. <10px 비행체 성능 근거로 사용하지 않는다.
- Sensors 2026, 26, 2528: 레이더 분류 연구의 재현율/오탐 절충. 영상에 물리량을 직접 이식하지 않는다.

논문의 수치/threshold를 우리 데이터의 설정값으로 그대로 복사하지 않는다.
이 문서는 기존 trajectory_quality_v2_plan의 KF/RTS 우선순위를 측정 검증 우선으로 수정한 새 계획이다.
