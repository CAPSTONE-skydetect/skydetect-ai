# C파트 제안: MiniRocket + 선형분류기로 2D 궤적 시계열 분류

작성일: 2026-09-27 | 대상: B/C 담당자 | 상태: **검토 제안, 학습 및 도입 미실행**

업데이트: 사용자가 MiniRocket + 선형분류기와 **2초 창**을 선택했다. 확정한 B 입력 구현은
[시계열 입력 계약 v1](TRAJECTORY_SEQUENCE_V1.md)을 따른다. 아래 4초/2초 비교 등은 당시 제안의 검토 배경이며,
현재 데이터는 30Hz, 4채널, 60샘플로 생성한다. C 학습은 아직 수행하지 않았다.

2026-09-28 전달 사항: 공통 전처리 계약은 `trajectory-sequence-1.0.1`, 기본 설정 ID는 `eb9be8154ee0f404`다.
정수 ms 기록의 FPS 추정을 수정했으며 `(N,4,60)` 형태와 채널 순서는 유지했다. 구 1.0.0 배열과 혼합하지 않는다.
현재 로컬 재처리본은 실제 train 156/validation 59개 개발용 창이며 test는 재처리하지 않았다.
합성 후보는 아직 채택 기준을 통과하지 않아 진단용 NPZ를 최종 학습 데이터로 전달하지 않는다.
실험 결과와 이후 결정은 [원인 진단 및 결정](SEQUENCE_GAP_FINDINGS.md)에 있다.

검토 기준: `test/B-simultest1`, HEAD `1ad2400` 및 현재 실제 A 궤적 전처리 진단 결과.
이 문서에 제안된 데이터 계약은 기존 Feature v4/API를 변경한 것이 아니다.

## 1. 전달 요약

> 현재의 9개 요약 특징 + RF를 대체할 우선 후보로, **bbox 없는 중심 궤적 시계열 -> MiniRocket -> RidgeClassifier**를 제안합니다. A는 변경하지 않고 B가 시간과 형태를 보존한 입력을 만들며, C가 변환기와 분류기를 학습합니다. CNN/LSTM/Transformer를 동시에 비교하기보다 이 후보 하나를 먼저 검증하고, 같은 실제 영상 그룹 평가에서 기존 방식보다 유효한지 판단하고자 합니다.

핵심은 모델 이름을 바꾸는 것이 아니라, 전체 궤적을 평균·분산 등으로 먼저 축약하지 않고 **짧은 구간의 움직임 패턴을 분류기에 전달하는 것**입니다. 다만 MiniRocket도 최종적으로 시계열을 특징 벡터로 변환합니다. 전체 시간 순서를 그대로 기억하는 모델은 아닙니다.

**추천안과 성능 보장은 다릅니다.** 현재 자료는 이 후보를 시험할 이유를 제공하지만, MiniRocket의 우월성이나 시뮬레이터의 현실성을 입증하지 않습니다. 목표 정확도를 미리 약속하지 않습니다.

| 결정 항목 | 제안 |
|---|---|
| 첫 지원 범위 | 확보한 새 vs 멀티콥터. 실제 드론 기종 확인 후 확정; fixed-wing은 초기 지원 범위 밖으로 명시 |
| A | 기존 추적과 출력 유지 |
| B | 중심 좌표·시간 기반 고정 시간 창, 전처리, 그룹 분할, 제한적 증강, 시뮬레이터 관측 보정 |
| C 후보 | 다변량 MiniRocket + 표준화 + RidgeClassifier |
| 비교 기준 | 기존 9개 특징 + RF. 새로운 입력에 기존 RF를 그대로 재사용하는 것이 아님 |
| 우선 평가 | 실제 원본 영상/촬영 그룹을 분리한 평가 |
| 합성 데이터 | 추가했을 때 실제 평가가 개선되는 경우에만 채택 |
| 이번 요청 | 모델 후보와 입력 계약에 대한 C의 검토·합의. 즉시 기존 모델 교체 요청은 아님 |

## 2. 왜 지금 시계열 모델을 제안하는가

### 확인된 사실

현재 [전처리 진단 보고서](output/a_preprocessing_audit/report.md)는 실제 A 출력 54개를 분석했습니다. 이 출력 폴더는 로컬 생성 자료이므로 제안서만 전달할 경우 보고서도 별도로 첨부해야 합니다.

| 항목 | 결과 | 해석 한계 |
|---|---|---|
| 파일 수 | bird 33개, drone 21개 | 독립 비행 54회라는 뜻은 아님 |
| 임시 원본 그룹 | 49개 | 원본 촬영 세션 확인 전의 잠정 묶음 |
| 평활화 후 경로 길이 유지율 중앙값 | 새 98.6%, 드론 98.7% | 전체 경로가 크게 사라진 것은 아님 |
| 3~10Hz 위치 진동 에너지 유지율 중앙값 | 새 35.2%, 드론 18.5% | 세부 진동은 양쪽 모두 감소; 날갯짓 손실의 확정 증거는 아님 |
| 기존 RF 기본 설정 재실행 | 정답 18/54, 보류 11/54 | 이미 살펴본 개발 자료의 파일 단위 재실행. 독립 test 성능이 아님 |
| 실제 시간 중앙값 | 새 4.767초, 드론 10초 | 길이 자체가 클래스 단서가 될 위험 |
| 촬영 종횡비 | 새는 4:3/16:9 혼합, 드론은 모두 16:9 | 촬영 장비·출처 편향 가능 |

사용자가 관찰한 물결, 감속 후 정지·반전, 활공·선회는 시간적 패턴의 후보입니다. 전체 평균만으로는 서로 다른 패턴이 같은 값이 될 수 있습니다. 따라서 시계열 표현을 검증할 가치가 있습니다.

그러나 다음은 아직 가설입니다.

- 새의 작은 물결이 실제 날갯짓 때문에 생겼는지, ROI 중심 변화나 CMC 잔차 때문인지.
- 시계열 모델이 물결을 활용해 처음 보는 촬영 조건에서도 분류를 개선하는지.
- 오분류의 주원인이 요약 특징인지, 합성-실제 차이인지, A의 관측 한계인지.

현재 자료에서도 드론에 반복적인 작은 움직임이 관찰됩니다. **주기성 = 새**라는 규칙을 두면 안 됩니다. A overlay의 raw 좌표와 B가 받는 CMC 보정 좌표가 다를 수 있으므로, 모델 입력은 실제 B 전달 좌표로 통일합니다.

## 3. MiniRocket은 무엇인가

```text
A의 시간 순서가 있는 중심 좌표
                  ↓ B
고정 시간 창 + 좌표 정규화 + 변위 채널
                  ↓ C
MiniRocket: 여러 시간 간격의 고정 합성곱 패턴 적용
                  ↓
패턴별 임계값 초과 비율(PPV)로 특징 벡터 생성
                  ↓
StandardScaler: 변환된 특징의 스케일 조정
                  ↓
RidgeClassifier: 특징을 조합해 bird / drone 점수 계산
                  ↓
창별 점수 집계 + 품질/보류 정책
```

### 3.1 원리

MiniRocket은 Dempster, Schmidt, Webb의 KDD 2021 시계열 분류 방법입니다. CNN처럼 합성곱 필터 전체를 역전파로 학습하는 대신 정해진 필터 구조를 사용합니다. 원 논문은 다양한 시계열 벤치마크에서 평가했으며, **이 프로젝트의 새·드론 A 궤적에서 검증한 논문은 아닙니다.** [원 논문](https://arxiv.org/abs/2012.08791)

직관적으로는 '짧은 상승 후 하강', '완만한 변화', '국소 진동'에 반응하는 여러 패턴 검사를 수행하는 방식입니다. 이 예시는 이해를 위한 비유이며 특정 필터가 특정 비행 행동을 의미한다고 보장하지 않습니다.

한 변환 특징은 다음처럼 이해할 수 있습니다.

```text
r_k(t) = k번째 합성곱 필터의 시간별 반응
z_k    = mean_t[ r_k(t) > b_k ]
```

`z_k`는 임계값을 넘는 시간 위치의 비율인 PPV입니다. `b_k` 같은 변환 파라미터는 학습 자료로 정하므로, **라벨을 안 쓴다는 이유로 test까지 넣고 fit하면 안 됩니다.** 서로 다른 dilation은 필터가 참조하는 시간 간격을 바꿉니다. [저자 구현](https://github.com/angus924/minirocket)

권장 구현인 aeon의 MiniRocket은 길이 9의 84개 기본 필터 조합을 사용하고, 기본 설정의 변환 특징은 약 1만 개입니다. 다변량 입력을 지원하지만 일반 구현은 **가변 길이와 누락값을 지원하지 않습니다.** 입력은 `(샘플, 채널, 시간)` 형태입니다. [aeon 변환기 API](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MiniRocket.html)

### 3.2 무엇을 기대하며, 무엇을 기대하면 안 되는가

| 기대할 수 있는 검증 대상 | 보장하지 않는 것 |
|---|---|
| 요약 통계에서 사라진 국소 파형·변화 패턴 활용 | 주기적 움직임의 원인이 날갯짓인지 판별 |
| 여러 시간 간격에서 패턴 감지 | 원래 물체의 3D 속도·거리 복원 |
| CPU 중심의 비교적 단순한 학습 구성 | GPU가 없어도 항상 정해진 시간 내 학습 완료 |
| 수작업 특징을 추가하지 않고 시계열 표현 시험 | 적은 독립 실데이터 문제 해결 |
| 좌표와 변위를 함께 활용 | 카메라 각도·투영·추적 오차에 대한 자동 불변성 |

특히 PPV는 시간축 전체에서 비율을 집계합니다. 필터가 포착하는 범위 안의 순서에는 반응할 수 있지만, **긴 시간에 걸친 행동 순서나 사건의 절대 시점을 모두 보존하지는 않습니다.** '어디까지나 전체 궤적을 완벽히 이해하는 모델'이라는 설명은 부정확합니다.

## 4. 선형분류기는 무엇이며 왜 Ridge를 붙이는가

MiniRocket이 만든 특징 벡터를 `z`라 하면 분류 점수는 다음 형태입니다.

```text
score = w₁z₁ + w₂z₂ + ... + wₘzₘ + b
```

RidgeClassifier는 이 가중치가 지나치게 커지지 않도록 L2 규제를 사용합니다. 개념적으로 다음 목적을 최소화합니다.

```text
분류용 목표값에 대한 제곱오차 + alpha × 가중치 제곱합
```

선형이라는 말은 **MiniRocket 변환 이후 공간에서 선형**이라는 뜻입니다. 원본 궤적을 직선 하나로 구분한다는 뜻이 아닙니다. `alpha`가 규제 강도이며, 수만 개 특징에 비해 독립 사례가 적은 상황에서는 규제가 중요합니다. [RidgeClassifier 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)

aeon의 공식 예제도 MiniRocket과 Ridge 계열의 조합을 소개합니다. 여기서는 그룹별 평가를 명시적으로 제어하기 위해 기본 `RidgeClassifierCV`를 그대로 쓰기보다 **MiniRocket + StandardScaler + RidgeClassifier**를 직접 묶는 구성을 제안합니다. [공식 예제](https://www.aeon-toolkit.org/en/stable/examples/transformations/minirocket.html)

| 선택 | 제안 |
|---|---|
| RidgeClassifier | 첫 후보. `alpha=1.0`으로 최소 실행; 조정한다면 개발 그룹 안에서 `0.1, 1, 10`만 비교 |
| LogisticRegression | 확률 출력이 반드시 필요할 때 논의할 대안. 이번에 병렬 모델 탐색 대상으로 추가하지 않음 |
| CNN/LSTM/Transformer | 부적합하다는 뜻은 아님. 현재 일정에서는 구조·학습 설정 탐색을 넓히지 않음 |
| 기존 RF | 동일 그룹·평가 단위로 비교할 기준 모델 |

### 확률과 점수는 반드시 구분

RidgeClassifier의 `decision_function`은 확률이 아닙니다. 점수 0.73을 '드론 확률 73%'라고 표시하면 안 됩니다. 또한 확인한 aeon `MiniRocketClassifier` 구현은 내부 분류기에 `predict_proba`가 없으면 예측 클래스에 1, 나머지에 0을 넣어 반환합니다. 이것은 확신도 추정이 아닙니다. 사용할 설치 버전에서도 확인해야 합니다. [aeon 분류기 소스](https://github.com/aeon-toolkit/aeon/blob/main/aeon/classification/convolution_based/_minirocket.py)

초기 계약은 `label`, `decision_score`, `score_type`, `abstain_reason`을 권장합니다. 기존 API가 확률을 강제한다면 별도 합의가 필요합니다. 확률 보정을 추가할 경우에도 학습과 분리된 실제 개발 그룹이 필요하며, 소표본 보정의 신뢰성은 별도로 검증해야 합니다. [확률 보정 공식 문서](https://scikit-learn.org/stable/modules/calibration.html)

## 5. B에서 C로 전달할 입력 계약 초안

**아래는 제안 설정이지 현재 구현된 계약이 아닙니다.** C 합의 후 B에서 데이터셋 생성기와 서비스 전처리가 같은 구현을 사용하도록 만듭니다.

### 5.1 좌표와 채널

정규화된 A 좌표를 `cx, cy`, 그 좌표에 대응하는 처리 영상 크기를 `W, H`라 하면:

```text
p_i = (cx_i, cy_i × H/W)
c   = 각 좌표 성분의 창 내부 중앙값
s   = max(창 내부 ||p_i - c||의 90백분위수, s_floor)
q_i = (p_i - c) / s
d_i = q_i - q_(i-1)        # 첫 샘플은 (0, 0)
```

`q_x, q_y, d_x, d_y`의 4개 채널을 제안합니다. 화면 좌표의 y축은 아래쪽이 양수인 상태로 일관되게 유지합니다. `d`는 일정한 1/30초당 정규화 변위이지 물리 속도가 아닙니다. `s_floor`는 정지에 가까운 궤적의 잡음 확대를 막는 공통 하한이며, 학습용 실제 자료에서 정한 후 manifest에 고정해야 합니다. 매우 낮은 이동량은 품질 메타데이터로 기록하고 별도 성능을 봅니다.

- 가로·세로를 따로 표준화하지 않습니다. 그러면 원·타원이나 물결의 상대 모양이 바뀝니다.
- 단일 스케일 정규화는 일정한 확대·축소 차이를 줄이지만 원근 투영, 방향, 시간에 따라 바뀌는 줌은 제거하지 못합니다.
- 절대 화면 이동량 정보를 일부 포기하는 선택입니다. 멀리 있는 물체의 사라진 미세 움직임을 복원하지 못합니다.
- 첫 후보에 bbox, conf, 파일명, species, subtype, source ID, 해상도 자체를 입력 채널로 넣지 않습니다. W/H는 기하 복원에만 사용합니다.
- `s_floor`와 정규화 방식은 새·드론 공통입니다. 클래스별 전처리는 금지합니다.

### 5.2 시간·누락·길이

| 항목 | 계약 초안 / 합의할 점 |
|---|---|
| 시간 기준 | A의 `timestamp_ms` 우선. 시뮬레이터에도 초 단위 시간 제공 |
| 재표본화 | 실제 시간을 유지해 30Hz. 원래 저FPS 신호를 올려도 새로운 관측 정보가 생기지는 않음 |
| 창 길이 | 우선 검토안 4초, 120샘플. `[t0, t0+4)` 구간의 `t0+k/30`, `k=0..119` |
| 창 채택 | 4초 구간에 원본 시간 지지가 있어야 함. 경계 밖 외삽 금지 |
| 창 이동 | 최초 1초 간격 제안. 같은 원본에서 나온 모든 창은 같은 split |
| 긴 gap | 창을 끊거나 제외. 0 좌표로 채우지 않음 |
| 짧은 gap | 제한된 선형 보간만 허용. 초기 제안: 연속 미관측 시간 0.1초 이하, 창 내 누락률 10% 이하. 실제 timestamp 정의와 함께 확정 |
| 짧은 track | 늘여서 4초로 만들지 않음. 관측 부족으로 기록; pad+mask를 지원한다고 가정하지 않음 |
| 평활화 | 기존 0.3초 SG를 자동 승계하지 않음. 초기 후보는 추가 SG 없이 재표본화; 다운샘플링 시 anti-alias 필터는 별개로 설계 |
| 유효성 | NaN/Inf, 역전·중복 시간, 단위 불일치, 장시간 경계 clamp 등을 검출하고 사유 기록 |

**4초를 바로 확정하면 안 되는 구체적 이유:** 현재 파일의 전체 지속시간만 보면 4초 이상은 bird 22/33, drone 21/21입니다. gap 검사는 적용하기 전이라 실제 적격 수는 더 적을 수 있습니다. 이 상태의 성능은 '4초 이상 관측 가능한 사례'에 한정됩니다.

일정상 길이 후보를 많이 탐색하지는 않습니다. B/C가 먼저 길이별 적격 원본 수를 확인하고, 새의 제외 비중이 수용 불가능하면 **2초/60샘플을 단일 초기 사양으로 선택**합니다. 2초는 행동 전체를 보기 어렵다는 대가가 있습니다. test 성능을 보고 창 길이를 바꾸지 않습니다.

### 5.3 전달 파일

```text
trajectory_dataset_v1_draft/
  train.npz               # X: float32 (N, 4, L), y: (N,)
  validation.npz
  test.npz                # 실제 평가인지 합성 내부 점검인지 명시
  metadata.csv
  dataset_manifest.json
```

`metadata.csv`에는 `sample_id`, `source_group_id`, `parent_track_id`, `label`, `domain(real/real_aug/sim)`, `split`, `window_start_s`, `window_end_s`, `augmentation_parent`, `seed`, `quality_flags`를 기록합니다. 이 항목을 자동으로 모델 특징에 합치면 안 됩니다.

manifest에는 좌표 원천(raw/CMC), 채널 순서, FPS, 창 길이, gap/필터/스케일 규칙, 원본 해시, 분할 기준, 버전, 제외 건수, 클래스별 **독립 그룹 수와 창 수를 따로** 기록합니다. 현재 9열 feature CSV만으로는 이 시계열 입력을 복원할 수 없습니다.

## 6. 데이터 분할과 증강: 모델보다 먼저 지킬 조건

```text
실제 원본 영상·촬영 세션 그룹 확정
                 ↓
그룹 단위 개발 / 최종 평가 분리
                 ↓
개발 내부 train / validation 그룹 분리
                 ↓
train 원본만으로 보정·증강·시뮬레이터 설정 결정
                 ↓
train 창 생성 → MiniRocket fit → scaler fit → Ridge fit
                 ↓
미증강 validation으로 선택 → 설정 고정 → 최종 평가
```

같은 영상의 crop, 재인코딩, 재업로드, 중복 track, 증강 파생본을 모두 같은 그룹에 둡니다. 같은 3D 비행을 여러 카메라로 투영한 합성 파생본도 그룹을 공유합니다. seed가 다르다는 사실만으로 모든 생성 사례가 독립적이지는 않습니다.

그룹 교차검증은 `StratifiedGroupKFold` 등으로 구성할 수 있지만, fold별 양쪽 클래스와 실제 그룹 수를 확인해야 합니다. **각 fold의 train 안에서 증강·보정·MiniRocket fit·표준화를 다시 수행**해야 합니다. 전체 자료로 보정한 시뮬레이터를 먼저 만든 뒤 그룹 CV를 돌려도 보정 누수가 남습니다. [그룹 분할 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

현재 54개는 이미 진단·시각 검토에 사용됐으므로 새로 나누더라도 완전히 미사용인 최종 test라고 부르지 않습니다. 지금은 **개발 그룹 평가**로 사용하고, 이후 확보하는 별도 촬영을 최종 평가로 남기는 것이 가장 명확합니다. 새 자료 확보가 불가능하면 이 한계를 공개합니다.

실제 기반 증강은 train에만 적용합니다. 초기에는 crop과 관측 근거가 있는 소규모 위치 잡음·짧은 dropout 정도로 제한하고, 전혀 새로운 행동을 만든다고 주장하지 않습니다. 무제한 시간 늘이기, 클래스별 주파수 주입, 판별하기 쉽게 만든 잡음은 피합니다. 합성 데이터의 카메라·잡음 조건도 라벨과 불필요하게 결합하지 않게 만듭니다.

## 7. 시간이 부족할 때의 최소 실험

모델 종류를 늘리지 않고 다음 순서만 진행합니다.

| 단계 | 학습 구성 | 판단 질문 |
|---|---|---|
| 기준 R | 9개 특징 + RF를 같은 개발 train 그룹에서 학습 | 동일 실제 평가 조건에서 기존 표현의 기준은 무엇인가? |
| 후보 M | 실제 train + 제한적 증강, MiniRocket + Ridge | 시계열 표현이 실제 영상 그룹 평가에서 도움이 되는가? |
| 후보 M+S | M의 실제 자료에 보정한 합성 자료 추가, 같은 모델 | 시뮬레이터 추가가 실제 일반화를 개선하는가? |

M과 M+S의 실제 train, 평가 그룹, 전처리, 모델 설정, 보류 정책은 동일하게 유지합니다. 합성 추가 전후를 비교할 때 평가군까지 바꾸지 않습니다. 초기 합성 혼합량은 실제 유래 학습 창과 **1:1을 공학적 시작값**으로 제안하되 최적 비율이라는 근거는 없습니다. 그룹별 창 수 상한을 두어 긴 드론 영상이나 많은 증강본이 학습을 지배하지 않게 합니다.

RF와 MiniRocket의 비교에서는 같은 원본·같은 시간 창을 사용해야 합니다. 입력 표현 차이와 모델 차이가 함께 있는 시스템 비교라는 점도 보고합니다. 이번 일정에서 이를 모든 조합으로 분해하지는 않습니다.

**합성-only 6,000/2,000/2,000 학습·검증·테스트는 내부 점검으로 별도 가능하지만, 실제 성능의 근거로 대체할 수 없습니다.** 또한 M+S는 '시뮬레이터만 사용한 연구'가 아니라 실제 및 합성 혼합 학습입니다. 기존 계획과 이 차이를 팀에서 합의해야 합니다.

### 평가 단위와 지표

| 지표 | 보고 방법 |
|---|---|
| 주 지표 | 실제 track 단위 macro-F1, balanced accuracy, bird/drone recall |
| 중복·긴 영상 영향 | 같은 원본 그룹의 track이 많으면 그룹에 동등 총가중치를 주어 보조 집계 |
| 창별 예측 통합 | 같은 track의 창별 decision score 평균 후 분류. 긴 영상에 자동 가산점이 생기지 않게 함 |
| 보류 포함 | 전체 대비 판정률(coverage), 판정한 사례의 정확도, 보류 포함 클래스별 recall을 함께 보고 |
| 불확실성 | 원본 그룹 단위 bootstrap 신뢰구간. 창을 독립 표본처럼 재표집하지 않음 |
| 실패 분석 | 활공, 저이동, 반전, 반복 진동, 짧은 track, 촬영 출처별 표본 수와 오류 |
| 실행 비용 | 변환+분류 추론 지연, 메모리, 첫 실행/JIT 준비 시간과 준비 후 시간을 구분 |

보류 없는 기본 분류 결과와 보류 정책을 적용한 운영 결과를 따로 제시합니다. 짧은 track이나 품질 제외 사례를 분모에서 숨기지 않습니다. 점수 평균과 보류 임계값은 개발 자료에서 고정합니다.

### 채택 판단

1. 누수·입력 계약 검사를 먼저 통과해야 합니다.
2. M이 같은 실제 평가에서 기준 R보다 낫고 한 클래스 recall을 크게 희생하지 않는지 봅니다.
3. M+S가 M보다 좋아지는지 봅니다. 평균 점수만 아니라 그룹별 변화와 불확실성을 함께 봅니다.
4. 특정 촬영원에서만 개선되거나 결과가 불확실하면 '검증 완료'가 아닌 '예비 결과'로 남깁니다.
5. 합성 추가가 악화시키면 합성량을 늘려 해결하려 하지 않고 M을 유지합니다. 모델 후보 자체가 실패하면 범위를 축소하거나 보류를 유지하며 원인을 다시 확인합니다.

정확도 95%나 특정 분포 거리 하나를 통과 기준으로 미리 단정하지 않습니다. 최소 클래스 recall, 허용 보류율, 지연 한계는 실제 사용 목적에 맞게 평가 전에 팀이 정해야 합니다.

## 8. C 구현 참고 코드

아래는 **이미 B에서 전처리·그룹 분할한 배열을 받았다는 전제의 학습 골격**입니다. 실행·성능 검증한 제품 코드가 아니며, 증강·CV·보류·모델 저장 구현은 포함하지 않습니다. aeon 공식 API를 확인해 작성했습니다. 프로젝트 환경에서 호환 버전을 확인·고정한 뒤 적용해야 합니다.

```python
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model():
    return Pipeline([
        ("minirocket", MiniRocket(
            n_kernels=10000,
            max_dilations_per_kernel=32,
            n_jobs=1,
            random_state=42,
        )),
        ("scale", StandardScaler(with_mean=False)),
        ("classifier", RidgeClassifier(alpha=1.0, class_weight="balanced")),
    ])


# X_train: (N_train, 4, L), X_validation: (N_validation, 4, L)
# y_train: bird/drone labels; groups: original acquisition groups.
assert X_train.ndim == 3 and X_train.shape[1] == 4
assert X_validation.shape[1:] == X_train.shape[1:]
assert np.isfinite(X_train).all() and np.isfinite(X_validation).all()
assert set(groups_train).isdisjoint(set(groups_validation))
assert set(np.unique(y_train)) == {"bird", "drone"}

model = make_model()
model.fit(X_train.astype(np.float32), y_train)
predictions = model.predict(X_validation.astype(np.float32))
scores = model.decision_function(X_validation.astype(np.float32))
classes = model.named_steps["classifier"].classes_
# Positive scores correspond to classes[1], not an assumed label order.
```

코드의 `class_weight="balanced"`는 창 개수 기준 클래스 불균형을 완화할 뿐, 같은 영상의 중복 창 문제를 해결하지 않습니다. B의 그룹별 샘플링 제한이 별도로 필요합니다. 실제 파생본을 1,000개 만들어도 실제 독립 사례가 1,000개가 되지 않습니다.

CV에서는 fold마다 새 `make_model()`을 만들어 train에만 fit합니다. 기존 모델의 변환 결과를 전체 자료에서 미리 계산해 CV하는 방식은 사용하지 않습니다. `alpha`를 고른 validation/CV 점수를 최종 독립 성능으로 재사용하지 않습니다.

## 9. B/C 담당 범위와 서비스 연결

| 담당 | 맡을 작업 | 전달 산출물 |
|---|---|---|
| B | 원본 검토·그룹 확정, 시계열 전처리, 품질 검사, 증강, 관측 기반 시뮬레이터 조정 | NPZ, 메타데이터, manifest, 재현 가능한 전처리 함수와 테스트 |
| C | MiniRocket·선형분류기 학습, 그룹 평가, 점수 집계, 보류 설정, 모델 저장 | 변환기+scaler+classifier 묶음, 지표, 오류 목록, 추론 코드 |
| 공동 | 창 길이, 좌표/채널, 낮은 이동량 처리, 학습 구성, 지원 범위, 점수 API | 버전이 명확한 B→C 계약 |
| A | 현재 방식 유지 | TrackSequence 및 현재 제공 메타데이터 |

학습된 MiniRocket도 모델의 일부이므로 Ridge 가중치만 전달하면 안 됩니다. C는 변환기, scaler, classifier, 클래스 순서, 전처리 계약 버전, 집계/보류 정책, 라이브러리 버전을 함께 저장합니다.

기존 C가 FeatureVector만 받는다면 새 시계열 인터페이스가 필요합니다. B의 시계열 생성과 C의 predict 호출을 별도 경로로 검증한 다음 기존 경로 전환을 결정합니다. 처음 선택한 창 길이만큼 관측이 쌓이기 전에는 '관측 수집 중' 상태가 필요하며, 4초 창이면 약 4초 관측 지연에 계산 시간이 추가됩니다.

## 10. 도입 전 검사 목록

- [ ] C가 조류 vs 멀티콥터 범위와 MiniRocket + Ridge 후보에 동의한다.
- [ ] 촬영 그룹을 확인하고 기존 개발 자료와 향후 최종 평가 자료를 구분한다.
- [ ] 4초와 2초의 적격 원본 수를 확인한 뒤 하나를 먼저 확정한다.
- [ ] 4개 채널, 단일 스케일 정규화, `s_floor`, gap 및 필터 규칙을 고정한다.
- [ ] 동일 입력의 학습용/서비스용 전처리 결과 일치 테스트를 통과한다.
- [ ] 서로 다른 FPS·종횡비, 짧은 track, 긴 gap, 정지 근처, NaN 입력을 테스트한다.
- [ ] 이동·단일 배율 변화에 대한 정규화의 의도된 성질을 확인한다. 카메라 각도 불변성까지 주장하지 않는다.
- [ ] 원본/증강/동일 잠재 비행의 split 교차가 0건인지 자동 검사한다.
- [ ] 동일 seed·입력·라이브러리에서 예측 재현 및 모델 저장/재로딩 일치를 확인한다.
- [ ] 기준 R, 후보 M, 후보 M+S를 동일 실제 평가군에서 비교한다.
- [ ] 점수와 확률을 구분하고 저품질·관측 부족·낮은 분리도 보류 사유를 구분한다.

## 11. 팀원에게 보낼 짧은 메시지

> 현재 B의 9개 요약 특징 대신, bbox 없는 2D 궤적 시계열을 사용하는 MiniRocket + RidgeClassifier를 우선 후보로 제안드립니다. MiniRocket이 여러 시간 간격의 국소 패턴을 특징으로 변환하고 선형분류기가 이를 분류하는 구조라, 현재 일정에서 대형 시계열 신경망을 새로 튜닝하는 것보다 먼저 검증할 만하다고 판단했습니다.
>
> B에서는 시간·종횡비를 보존한 정규화 좌표와 변위의 4채널 배열, 원본 그룹 정보, 전처리 manifest를 제공하는 방향입니다. 4초 창은 현재 새 자료 일부를 제외하므로 2초 또는 4초 중 입력 사양을 먼저 합의하고 싶습니다. C에서는 변환기와 분류기의 학습·평가·점수 집계를 담당해 주시면 됩니다.
>
> 성능 개선이 확인된 상태는 아닙니다. 같은 실제 원본 그룹 분할에서 기존 RF와 비교하고, 실제 기반 학습에 보정된 합성 데이터를 추가했을 때 개선되는지도 최소 실험으로 확인하려 합니다. Ridge 점수는 확률이 아니므로 기존 confidence 출력과의 연결도 함께 검토 부탁드립니다. 상세 제안서의 입력 계약과 역할 분담을 확인해 주세요.

## 12. 참고 자료와 근거 구분

- 알고리즘 근거: Dempster, Schmidt, Webb, 2021, [MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification](https://arxiv.org/abs/2012.08791).
- 원 구현: [angus924/minirocket](https://github.com/angus924/minirocket). 해당 방법의 일반 성질 근거이지 우리 데이터의 성능 근거는 아니다.
- 구현 선택: [aeon MiniRocket API](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MiniRocket.html), [공식 예제](https://www.aeon-toolkit.org/en/stable/examples/transformations/minirocket.html).
- 분류기·평가 계약: [RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html), [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [확률 보정](https://scikit-learn.org/stable/modules/calibration.html).
- 프로젝트 관측 근거: [전처리 진단 사용 설명](PREPROCESSING_DIAGNOSTICS.md), [로컬 진단 결과](output/a_preprocessing_audit/report.md), [사례 검토](output/a_preprocessing_audit/case_review.md).
- 4채널 구성, 정규화 식, 창 길이, gap 기준, 혼합 비율은 **프로젝트용 설계 제안**이다. MiniRocket 논문이 우리 문제에 대해 검증한 최적 설정이라고 인용하지 않는다.

결론: **MiniRocket + Ridge는 '정답으로 확정한 모델'이 아니라, 제한된 일정에서 시간 구조를 활용할 수 있는지 가장 먼저 검증할 단일 후보다. 실제 그룹 평가와 합성 추가 효과를 확인한 뒤 도입한다.**
