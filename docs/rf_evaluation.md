# RF 분류기 성능 평가

Part C의 RandomForest bird/drone 분류기에 대한 학습·평가 절차와 확정 지표를 기록한다.

## 실행 방법

```bash
# 학습 (전체 train 데이터로 학습 후 models/rf_classifier.pkl 저장)
python -m ai_server.services.train

# 평가 (train 학습 → test 홀드아웃 평가, reports/ 에 결과 저장)
python -m ai_server.services.evaluate
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--train-path` | `data/train_features.csv` | 학습 CSV 경로 |
| `--test-path` | `data/test_features.csv` | 테스트 CSV 경로 |
| `--report-dir` | `reports/` | 리포트 저장 위치 |
| `--n-estimators` | `100` | 트리 개수 |

산출물은 `reports/metrics.json`(전체 지표)과 `reports/report.png`
(confusion matrix / ROC / feature importance)이다. `reports/`는 생성물이므로 git 추적에서 제외한다.

## 데이터셋

파일명은 `train_features.csv` / `test_features.csv`로 **고정**하고, 버전은 파일명이 아니라
데이터 안의 컬럼으로 추적한다. 학습·평가 시 아래 값이 모델 번들과 리포트에 자동 기록된다.

| 항목 | 값 |
|---|---|
| `simulator_version` | `3.0.0` |
| `feature_version` | `3.0.0` |
| `feature_config_id` | `3b9a6c638c3f3519` |

| 항목 | train | test |
|---|---|---|
| 샘플 수 | 8,000 | 2,000 |
| 라벨 | bird 4,000 / drone 4,000 | bird 1,000 / drone 1,000 |
| `family_id` 수 | 926 | 232 |

**누수 검증** — `family_id` 교집합 0, `sample_id` 교집합 0, 동일 피처 행 0.
패밀리 단위로 분리되어 있어 같은 궤적 계열이 train/test 양쪽에 걸치지 않는다.
이 검사는 `evaluate.py`가 실행할 때마다 수행해 `metrics.json`에 기록한다.

## 피처 (11종)

`v_mean`, `v_std`, `a_mean`, `turn_rate_mean`, `turn_rate_p95`, `heading_change_ratio`,
`straightness`, `stationary_ratio`, `bbox_area_mean`, `bbox_area_cv`, `bbox_scale_rate_std`

목록은 `ai_server/services/train.py`의 `FEATURE_NAMES`가 단일 진실 공급원이며,
`evaluate.py`가 이를 참조하므로 피처가 바뀌어도 평가 스크립트는 수정할 필요가 없다.

## 하이퍼파라미터 선정

테스트셋 성능으로 설정을 고르면 테스트셋에 과적합되므로, **train 내부
`GroupKFold(family_id)` 교차검증**으로 선정한 뒤 테스트셋은 최종 확인에만 사용했다.

| 설정 | CV f1_macro | 모델 크기 |
|---|---|---|
| `n=100` | 0.8053 | 3.5 MB |
| `n=300` | 0.8093 | 10.6 MB |
| `n=300, min_samples_leaf=5` | 0.8059 | 6.7 MB |
| **`n=100, min_samples_leaf=5`** | **0.8033** | **2.2 MB** |
| `n=300, min_samples_leaf=20` | 0.7934 | 2.8 MB |
| `n=300, max_features=None` | 0.8090 | — |
| `n=300, max_depth=10` | — (test 0.7604) | — |

상위 후보들의 CV 차이는 0.006 이내로, 폴드 간 표준편차(±0.011)보다 작아
**통계적으로 구분되지 않는다.** 성능이 동등하다면 저장소 히스토리에 부담이 적은 쪽이
낫다고 판단해 모델 파일이 가장 작은 조합을 택했다.

- `max_features=None`은 테스트셋에서 0.8110으로 가장 높았지만 CV에서는 재현되지 않았고
  분산만 커져(±0.0162) 테스트셋 노이즈로 판단해 채택하지 않았다.
- `max_depth` 제한은 모든 조합에서 성능이 뚜렷하게 떨어져 기본값을 유지한다.
- 압축 없이 저장하면 트리 300개 기준 50MB에 달한다. `joblib.dump(compress=3)`을 적용해
  최종 모델은 2.2MB다.

**확정: `n_estimators=100`, `min_samples_leaf=5`, `random_state=42`, 나머지 sklearn 기본값**

## 확정 지표

### 홀드아웃 테스트셋 (n=2,000)

| 지표 | 값 |
|---|---|
| accuracy | **0.8070** |
| F1 (macro) | **0.8067** |
| ROC-AUC | 0.8902 |
| PR-AUC | 0.8995 |

| 클래스 | precision | recall | f1-score | support |
|---|---|---|---|---|
| bird | 0.7843 | 0.8470 | 0.8144 | 1,000 |
| drone | 0.8337 | 0.7670 | 0.7990 | 1,000 |

drone의 precision(0.8337)이 recall(0.7670)보다 높다. 드론이라 판단하면 대체로 맞지만
실제 드론의 23%를 놓친다는 뜻으로, 탐지 목적상 recall 개선이 우선 과제다.

### Confusion Matrix

|  | pred bird | pred drone |
|---|---|---|
| **true bird** | 847 | 153 |
| **true drone** | 233 | 767 |

### 교차검증 (train, GroupKFold 5-fold)

f1_macro **0.8033 ± 0.0121** — folds: 0.8132 / 0.7998 / 0.8149 / 0.7816 / 0.8071

홀드아웃(0.8067)과 CV(0.8033)가 1 표준편차 내에서 일치해 과적합 징후는 없으며,
train/test 분할이 편향되지 않았음을 뒷받침한다.

### Feature Importance

| 피처 | 중요도 |
|---|---|
| `a_mean` | 0.1460 |
| `bbox_scale_rate_std` | 0.1267 |
| `bbox_area_mean` | 0.1164 |
| `turn_rate_mean` | 0.1136 |
| `v_mean` | 0.1024 |
| `turn_rate_p95` | 0.1020 |
| `straightness` | 0.0921 |
| `bbox_area_cv` | 0.0628 |
| `v_std` | 0.0613 |
| `stationary_ratio` | 0.0385 |
| `heading_change_ratio` | 0.0381 |

11개 피처가 0.038~0.146 범위에 고르게 분포해 특정 피처에 쏠리지 않는다.
신규 도입된 bbox 계열(`bbox_scale_rate_std`, `bbox_area_mean`)이 2·3위를 차지해
피처 확장의 기여가 확인된다.

## 세그먼트별 정확도

### observation_profile — 노이즈 강건성

| 구분 | n | accuracy |
|---|---|---|
| ideal | 1,000 | 0.8390 |
| noisy | 1,000 | 0.7750 |

노이즈 조건에서 **6.4%p 하락**한다. Sim2Real 관점에서 실측 환경 성능의 하한으로 볼 수 있다.

### scenario — 취약 시나리오

| 구분 | n | accuracy |
|---|---|---|
| multi_mode | 500 | 0.8500 |
| baseline | 508 | 0.8209 |
| sharp_turns | 498 | 0.7811 |
| sudden_dash | 494 | 0.7753 |

급격한 방향 전환(`sharp_turns`)과 급가속(`sudden_dash`)이 상대적으로 취약하다.

### behavior_mode

| 구분 | n | accuracy |
|---|---|---|
| flap_jitter | 175 | 0.9486 |
| thermal_circle | 234 | 0.8932 |
| foraging_zigzag | 153 | 0.8627 |
| glide | 212 | 0.8208 |
| cruise | 1,000 | 0.7670 |
| sudden_escape | 226 | 0.7345 |

새 고유의 움직임이 뚜렷한 `flap_jitter`(날갯짓)와 `thermal_circle`(상승기류 선회)은
0.89 이상으로 잘 구분된다. 반면 `sudden_escape`(급회피)는 0.7345로 가장 낮은데,
새의 급기동이 드론의 급기동과 유사해 분리가 어려운 것으로 보인다.
`cruise`는 전체 테스트셋의 절반을 차지하면서 0.7670에 그쳐, **등속 직진 구간의
bird/drone 구분 난이도가 전체 성능의 주된 제약**이다.

### training_length_group

| 구분 | n | accuracy |
|---|---|---|
| short | 193 | 0.8187 |
| standard | 1,807 | 0.8058 |

트랙 길이에 따른 유의미한 성능 차이는 관찰되지 않는다.

## 후속 과제

- `cruise`(0.7670) / `sudden_escape`(0.7345) 구간 개선 — 현재 전체 정확도의 병목
- drone recall 0.7670 개선 — 탐지 목적상 미탐(false negative) 비용이 크다
- noisy 조건 6.4%p 하락 완화
- 실측 트랙 기반 검증 — 현재 지표는 전부 시뮬레이션 데이터 기준이다
