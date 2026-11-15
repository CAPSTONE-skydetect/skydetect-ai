# 카메라 거리 50~500m Sim-to-Real 점검

2026-10-05. `sequence-guidance-1.6.0`의 기본 카메라 범위 50~500m를
과거 20~1500m 범위와 비교했다. 이는 **촬영 거리의 설계 제약 점검**이며,
실제 촬영 거리 측정이나 시뮬레이터의 현실성 인증이 아니다.
기존 C 전달 ZIP의 1.4.0 산출물과 동일한 합성 시드로 맞춘 직접 비교는
`SEQUENCE_VERSION_COMPARISON_20261005.md`를 보라. 이 문서의 합성
시드는 C 전달 시드와 다르므로 여기의 점수로 버전 차이를 단정하지 않는다.

## 동일 조건

실제 A 궤적은 기존 `trajectory-sequence-1.0.1`의 2초·30Hz 창이다.
같은 잠재 비행 시드, 행동 명령, 관측 잡음 시드를 두 카메라 범위에 적용했다.
실제 train 영상 그룹을 3-fold로 분리해 fold별로 관측 프로필을 맞춘 뒤
held-out train 그룹과 비교했다. 이후 전체 train으로 프로필을 고정하고
기존 실제 validation을 한 번 비교했다. 실제 test는 열지 않았다.

분포 거리는 새/드론 각각의 투영·형태·시간 지표에 대한 Wasserstein 거리를
실제 train의 IQR 등으로 정규화해 평균한 수치다. **작을수록 가깝지만
0~1로 제한된 점수는 아니며, 분류 정확도도 아니다.** 유효 비행률은
생성 시도 중 2초 창을 하나 이상 만들 수 있었던 비행의 비율이다.

## 결과

| 자료 | 과거 20~1500m 거리 | 현재 50~500m 거리 | 과거 유효 비행률 | 현재 유효 비행률 |
|---|---:|---:|---:|---:|
| train 그룹 CV, 12비행/클래스/fold | 0.9202 | **0.8860** | 94.4% | 94.4% |
| 같은 train의 새 합성 시드 재현, 18비행/클래스/fold | **0.8492** | 0.8513 | 93.5% | **95.4%** |
| 기존 실제 validation, 12비행/클래스 | 1.1449 | **1.0649** | 87.5% (21/24) | **91.7% (22/24)** |

첫 train CV의 세 fold와 validation에서는 거리가 줄었다. 그러나 새 시드의
train 재현에서는 거리 차이가 반대 방향으로 약 0.0021 발생했다.
따라서 **50~500m가 분포 거리를 일관되게 개선한다는 결론은 보류**한다.
유효 비행률 개선은 카메라 조건이 유효 창을 더 자주 만든다는 증거이지,
새·드론 운동 자체를 더 현실적으로 만들었다는 증거가 아니다.

validation의 새 2초 화면 이동 폭 중앙값은 실제 0.171, 과거 합성 0.436,
현재 합성 0.400이었다. 차이는 줄었지만 합성 새가 여전히 훨씬 크게
움직였다. 드론은 실제 0.083, 두 합성 모두 약 0.078이었다.
새의 저이동 비율 지표는 현재 후보에서 오히려 소폭 나빠졌고,
3~10Hz 변위 대역비도 날개짓을 직접 식별하는 근거가 아니다.

validation은 기존 연구에서 여러 차례 사용된 소규모 그룹이므로
독립적인 일반화 검증이 아니다. 이 실험은 **C의 실제 validation
macro-F1 이득이나 합성 데이터 학습 효용을 측정하지 않았다.** 기존 C
전달 ZIP은 변경하지 않았고, 50~500m 합성을 새 기본 학습 데이터로
승인하지 않는다. 실제 test 및 미사용 영상 그룹을 이용한 후속 평가가
필요하다.

## 재현 산출물

- `output/camera_range_v2_audit_20261005/`: train CV, 기존 validation,
  지표별 거리·중앙값, 프로토콜, 결과 JSON.
- `output/camera_range_v2_replication_20261005/`: 다른 합성 시드의
  train-only CV. validation/test 미열람.

```powershell
.\venv\Scripts\python.exe -m research.evaluate_camera_range_v2 --output research/output/camera_range_v2_audit_20261005
.\venv\Scripts\python.exe -m research.evaluate_camera_range_v2 --output research/output/camera_range_v2_replication_20261005 --count 18 --seed-start 361000 --train-only
```

각 명령은 새로운 빈 출력 폴더에 실행해야 한다. 기존 실험 파일을
덮어쓰지 않는다.
