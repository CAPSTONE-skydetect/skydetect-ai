# B-3. 세 학습 구성의 동일 실제 A validation 비교

## 실행 범위

`real_reference_comparison_v1`의 세 train 구성을 동일한 MiniRocket +
StandardScaler + RidgeClassifier로 학습하고, **동일한 실제 A validation**에
평가했다. 2초·30Hz·4채널 `(N,4,60)` 계약은 변경하지 않았다. 평가 코드는
`evaluate_real_reference_comparison.py`이며 상세 프로토콜, 모델, 그룹별
예측은 Git에서 제외된 `research/output/real_reference_evaluation_b3_v2/`에
있다. C 운영 코드에 세 모델을 구현하거나 배포한 작업은 아니다.

| 학습 구성 | 학습 창 / 그룹 | 실제 validation 창 macro-F1 | 궤적 macro-F1 | 원본 영상 그룹 macro-F1 | 새 recall | 드론 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 실제-only | 156 / 29 | 0.8101 | 0.8167 | **0.8990** | 0.8333 | 1.0000 |
| 실제+증강 | 6,156 / 29 | 0.8625 | 0.9060 | **0.8990** | 0.8333 | 1.0000 |
| 합성-only | 587 / 104 | 0.5683 | 0.5299 | **0.5833** | 0.6667 | 0.5000 |

validation은 59개 창, 11개 궤적, **10개 원본 영상 그룹**이다. 실제 두
구성은 같은 새 영상 1개를 드론으로 오분류했고, 합성-only는 영상 4개를
오분류했다. 따라서 이번 증강은 창/궤적 단위 개선은 보였으나 **영상 단위
오류를 줄였다고 볼 수 없다**. 합성-only의 낮은 실제 성능은 현재 합성
데이터만으로 실제 A 궤적을 대체할 근거가 부족함을 보여준다.

## 비교 통제

- 세 구성 모두 MiniRocket 커널 10,000개, 동일 seed와 RidgeClassifier를
  사용했다. 모델 상태는 구성마다 별도로 적합했다.
- 실제-only와 실제+증강의 MiniRocket·scaler는 실제 train에만 적합했다.
  증강 창은 Ridge 학습에만 포함했고 총 sample-weight 질량을 10%로 제한했다.
- 합성-only의 MiniRocket·scaler·Ridge는 모두 합성 train에만 적합했다.
  실제 train 또는 validation으로 변환기를 적합하지 않았다.
- alpha는 train 그룹 3-fold CV에서 선택했다. 두 실제 구성은 같은 실제
  train alpha `0.1`, 합성-only는 합성 train alpha `100`을 사용했다.
  validation은 어떤 적합·하이퍼파라미터 선택에도 사용하지 않았다.
- 실제 파생 창은 원본 영상 그룹, 합성 창은 비행 seed 그룹을 유지한다.
  package 해시·그룹 겹침·증강 부모·입력 계약 검사 후 실행했다.

## 판단과 다음 단계

현재 개발 기준선은 **실제-only**로 유지한다. 실제+증강은 영상 그룹
성능이 같으므로 C에서 별도 실험 구성으로 남기되, 개선이 확인된 방식으로
홍보하지 않는다. 합성-only는 전이 실패를 보여주는 대조군이다. C 담당자가
동일 패키지와 전처리 계약으로 재현한 뒤, 실제+증강의 창 단위 개선이
실서비스 집계에 의미가 있는지 검토해야 한다.

이 validation은 이전 개발 과정에서 반복 사용되었고, 과거 test도 이미
열람했다. 촬영 세션 독립성은 미확인이다. 따라서 이 표는 **개발 비교**이지
새로운 실제 영상에 대한 최종 정확도나 학술적 일반화 성능이 아니다.
최종 평가는 새로 확보한 미사용 실제 A 영상 그룹에서 수행해야 한다.

## 재현

일반 서비스 `venv`와 분리된 환경에서 `requirements-minirocket.txt`를
설치한다. 기존 출력 경로는 덮어쓰지 않으므로 새 이름을 지정한다.

```powershell
.\research\output\.venv-minirocket\Scripts\python.exe -m research.evaluate_real_reference_comparison --package research/output/real_reference_comparison_v1 --output research/output/real_reference_evaluation_b3_v3
.\venv\Scripts\python.exe -m pytest research/tests/test_real_reference_comparison.py research/tests/test_real_reference_package.py -q
```

입력 패키지·모델 해시, 라이브러리 버전, 그룹별 예측 및 혼동행렬은
`protocol.json`, `results.json`, `*_validation_group.csv`에 기록된다.
