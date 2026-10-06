# C파트: MiniRocket + Ridge 분류기

기존 9개 요약 특징 + RandomForest 를 **2초 궤적 시계열 → MiniRocket → StandardScaler →
RidgeClassifier** 로 대체했다. 근거와 제약은 `research/MINIROCKET_C_PROPOSAL.md`,
`research/REAL_REFERENCE_DATASET_V1.md`, `research/REAL_REFERENCE_COMPARISON_B3.md` 를 따른다.

## 파이프라인

```text
A TrackSequence (CMC 보정, stabilization.applied=true)
        ↓  research.trajectory_sequence.window_track   (B 구현을 그대로 호출)
2초·30Hz·1초 stride 창, float32 (N, 4, 60), 채널 q_x q_y d_x d_y
        ↓  MiniRocket(n_kernels=10000, random_state=20260928)
        ↓  StandardScaler(with_mean=False)
        ↓  RidgeClassifier → 창별 margin (양수 = drone)
창 margin 평균 → label / uncertain(보류 사유)
```

## 파일

| 파일 | 역할 |
| --- | --- |
| `ai_server/services/sequence_model.py` | 패키지 검증 로드, 모델 생성, alpha CV, 집계, 지표 (학습·평가·추론 공통) |
| `ai_server/services/train.py` | 운영 모델 학습 → `models/minirocket_classifier.joblib` |
| `ai_server/services/evaluate.py` | 세 학습 구성 비교 → `reports/minirocket/` (지표 + 카드 이미지), `--publish` 시 `docs/` |
| `ai_server/utils/metrics_plot.py` | 혼동행렬·지표·정밀도/재현율 카드와 추이 그래프 (이슈 기록용) |
| `ai_server/services/classifier.py` | `MiniRocketClassifier`: TrackSequence 추론 |
| `ai_server/services/prediction.py` | 응답(`PredictionResult`) 조립 |
| `ai_server/routers/classify.py` | `POST /classify` |

## 실행

B 패키지(`real_reference_comparison_v1.zip`)를 `research/output/` 아래에 푼다. 이 폴더는
Git 에서 제외된다. 로더가 manifest 의 SHA-256, 건수, 계보, train/validation 그룹 겹침을
검사하고 하나라도 다르면 거부한다.

```bash
python -m ai_server.services.train                                  # 실제-only (기본)
python -m ai_server.services.train --arm real_plus_augmentation --output-path models/aug.joblib
python -m ai_server.services.evaluate                               # 세 구성 비교
python -m ai_server.services.evaluate --margin-threshold 0.2        # 보류 정책 결과 추가
python -m ai_server.services.evaluate --publish                     # docs/ 확정본 갱신
python -m pytest tests/test_minirocket_classifier.py -q
```

## 학습 규칙

| 항목 | 규칙 |
| --- | --- |
| 운영 모델 | **실제-only** (B-3 개발 기준선). 실제+증강은 영상 그룹 성능이 같아 실험 구성으로만 남김 |
| 변환기 fit | 실제 두 구성은 실제 train 에만, 합성-only 는 합성 train 에만. validation 에는 절대 fit 하지 않음 |
| alpha | `{0.1, 1, 10, 100}` 중 train 그룹 `StratifiedGroupKFold(3)` 에서 영상 그룹 macro-F1 최대. fold 마다 MiniRocket·scaler 를 새로 fit. 실제 두 구성은 같은 alpha 공유 |
| 가중치 | CV: 클래스·그룹 균형 가중치. 최종 fit: 패키지의 `sample_weight` (증강 총량 10%) |
| 입력 채널 | `X` 만. `y`, `group_id`, `domain`, 품질 메타데이터는 모델 입력이 아님 |
| 저장 | 변환기 + scaler + classifier + 클래스 순서 + 계약 버전/ID + alpha + 라이브러리 버전 + 패키지 해시를 한 번들로 |

## 응답 (`PredictionResult`)

| 필드 | 의미 |
| --- | --- |
| `label` | `bird` / `drone` / `uncertain` |
| `decision_score` | 창별 Ridge margin 평균. **확률이 아니다.** 양수 = drone |
| `score_type`, `score_is_probability` | 항상 `ridge_margin_mean`, `false` |
| `abstain_reason` | `invalid_input` (비보정·시계 불일치·FPS 미달 등), `insufficient_observation` (2초 창 부족), `low_separation` (\|margin\| < `margin_threshold`) |
| `windows_used`, `window_scores`, `window_rejections` | 판정 근거 창과 제외 사유별 개수 |
| `model_version` | `계약버전/계약ID/학습구성/alpha` |

요청은 `ClassifyRequest{track_sequence, margin_threshold=0, min_windows=1}` 이다.
`margin_threshold` 기본값 0 은 보류 없음이다. 보류 임계값은 test 를 보고 정하지 않고 개발 자료에서 고정한다.
2초 창을 만들려면 관측이 최소 2초 필요하므로 그 전에는 `insufficient_observation` 이 나온다.

## 현재 성능과 한계

`docs/minirocket_evaluation.md` 참고. C 구현이 B-3 의 수치(그룹 macro-F1 실제-only 0.8990,
실제+증강 0.8990, 합성-only 0.5833)를 동일하게 재현했다.

- validation(10개 원본 영상 그룹)은 개발 중 반복 사용됐고 과거 test 는 이미 열람했다.
  **최종 일반화 성능이 아니다.** 새로 확보한 미사용 실제 영상 그룹에서 최종 평가해야 한다.
- 그룹 bootstrap 95% CI 가 넓다 (실제-only macro-F1 약 0.67–1.00).
- 촬영 세션 독립성은 미확인이다.
- MiniRocket 은 numba JIT 를 쓴다. 모델 로드 시 한 번 예열한다.
- numba 0.63 이 NumPy 2.4 를 지원하지 않아 `requirements.txt` 의 NumPy 를 2.3.5 로 고정했다.
