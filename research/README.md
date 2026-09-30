# Research 작업 안내

현재 방향은 **실제 A 궤적 기준 합성 보조·검증 전략**이다. 실제 A 궤적을
평가 기준으로 두고, 합성 궤적은 학습 보조 후보·어려운 사례 생성·대조군에
사용한다. 합성 train/test 점수만으로 실제 성능을 주장하지 않는다.
기존 물리 시뮬레이터와 Sim-to-Real 보정 과정은 유지한다. 새 실험이 과거
실험을 성공으로 바꾸거나 실패 기록을 지우지는 않는다.

## 현재 작업 경로

| 역할 | 코드와 문서 | 현재 상태 |
| --- | --- | --- |
| 실제 궤적 그룹·2초 시계열 입력 | `trajectory_sequence.py`, `build_real_sequence_dataset.py`, `TRAJECTORY_SEQUENCE_V1.md` | 계약 `trajectory-sequence-1.0.1`: 30Hz, float32 `(N, 4, 60)` |
| 합성 비행·A 유사 관측 | `dynamics.py`, `generators.py`, `behavior.py`, `sequence_simulator.py`, `observation.py` | 학습 후보와 어려운 사례 생성에 유지. 실제 자료의 대체재로 승인된 것은 아님 |
| 실제-only / 실제+합성 비교 | `prepare_sequence_handoff.py`, `evaluate_sequence_handoff.py`, `SEQUENCE_HANDOFF_RESULTS.md` | 개발 평가 완료. 전체 합성 혼합은 실제 validation을 악화시킴 |
| 실제 궤적 기반 제한적 증강 | `evaluate_anchor_augmentation.py`, `build_anchor_training_dataset.py`, `ANCHOR_AUGMENTATION_RESULTS.md` | 개발용 후보. 별도 실제 validation 개선은 없었음 |
| 영상·관측 오차 진단 | `ORIGINAL_VIDEO_AUDIT_RESULTS.md`, `SEQUENCE_REFINEMENT_RESULTS.md` | 측정 근거와 기각된 후보를 보존. 후보를 기본 설정으로 간주하지 말 것 |

B-1/B-2의 고정 기준과 세 학습 구성 데이터는
[`REAL_REFERENCE_DATASET_V1.md`](REAL_REFERENCE_DATASET_V1.md)에 정리했다.
로컬 패키지는 `output/real_reference_comparison_v1/`과 전달용 동명 ZIP이며
test를 포함하지 않는다. 세 구성의 B 연구 환경 비교 결과는
`REAL_REFERENCE_COMPARISON_B3.md`에 있고, 실패 사례 분석은
`REAL_REFERENCE_FAILURE_ANALYSIS_B4.md`에 있다. C 운영 모델 연결은 별도 작업이다.

C 비교에는 **같은 2초 입력 계약과 같은 모델 구현**을 쓰되 학습 자료를
구분한다.

1. `real-only`: 실제 A train만 사용한다.
2. `real-plus-augmentation`: 같은 실제 train에 출처가 기록된 특정 증강
   후보를 추가한다. 현재 증강 방식의 성능 향상이 입증된 것은 아니다.
3. `synthetic-only`: 합성 train만 사용하고, 다른 구성과 **동일한 실제
   validation**에서 평가하는 대조군이다. B-3 개발 비교를 실행했다.

세 구성이 별도 C 모듈이라는 뜻은 아니다. 각 학습 실행의 모델 상태는
분리하고, MiniRocket·scaler의 적합 자료를 실험 규칙에 기록한다. 기존
실제+합성 비교는 표현 변환기를 실제 train에서만 적합하고 Ridge 학습
가중치에 합성을 포함했다. 엄밀한 synthetic-only 비교라면 실제 train으로
적합한 변환기를 재사용하면 안 된다. 어느 경우든 실제 validation/test로
변환기나 분류기를 적합하지 않는다.

실제 원본 영상과 그 파생 창·증강 행은 항상 같은 그룹이다. export에는
원본 그룹 ID, 부모 궤적, 합성 donor, 라벨, 시드, 입력 계약 ID와 파일
해시를 남긴다. 실제 validation으로 구성을 선택한 뒤, **새로 확보한
미사용 실제 영상**이 있을 때만 마지막 최종 평가를 한다.

## 현재 근거의 한계

기존 개발 평가의 실제 validation 영상 macro-F1은 real-only `0.8990`,
전체 합성 25% 혼합 `0.6970`이었다. 제한적 실제 궤적 기반 증강은 train
그룹 CV `0.8569 -> 0.9293`이었지만 실제 validation은 `0.8990 ->
0.8990`이었다. 합성 데이터가 실제 분류에 유익하다는 근거는 아직 없다.

현재 실제 자료는 54궤적, 잠정적인 원본 영상 49그룹이다. 서로 다른 영상의
촬영 세션 독립성은 확인되지 않았다. 기존 test도 개발 중 열람한 이력이
있어 새로운 독립 최종 holdout이 아니다. 위 결과 문서에서 집계 단위와
코호트, 실패 사례를 함께 확인한다.

`minirocket_inference.py`는 유효한 2초 창이 없을 때 판정을 보류하는
예시다. 낮은 확신도에 대한 검증된 보류 정책과 B/C 런타임 연결은 아직
구현되지 않았다.

## 과거 코드와 로컬 산출물

- `SIMULATION_V4.md`, `FEATURES_V4.md`, `build_dataset.py`와 물리 모델
  모듈은 이전 합성-only / 특징 CSV 경로를 설명한다. 현재 C 시계열 입력의
  주 경로는 아니지만 시뮬레이터 개발 과정과 합성 대조군 재현에 필요하다.
- `SIM_TO_REAL_SEQUENCE_V1.md`, `STAGE4_MOTION_REFINEMENT.md`와 진단
  스크립트·결과에는 채택/기각된 보정 과정이 남아 있다. 폴더명 `v1`~`v4`는
  실행 라벨이지 현실성 인증이나 물리 모델 버전이 아니다.
- `research/data/`의 실제 영상·궤적과 `research/output/`의 데이터셋·실험
  결과·MiniRocket 별도 환경은 Git에서 제외된다. Git pull로 복원되지
  않으며 서로 다른 실행 결과를 덮어쓰거나 섞지 않는다.
- 현재 개발 전달본은 `output/sequence_handoff_20260928/C_handoff.zip`이다.
  최신 실제 궤적 기반 증강 기록은 `output/anchor_augmentation_v3/`, 개발용
  6,000행 train은 `output/anchor_training_6000_v3/`이다. **C 기본 학습
  자료로 승인한 결과는 아니다.** 보고서에서 참조하는 이전 시뮬레이터
  및 보정 실험 산출물은 유지한다.

재실행 시 새로운 출력 폴더를 지정한다. 증강 평가 CLI는 `--output`을,
대량 train 생성 CLI는 `--evaluation`과 `--output`을 필수로 받는다. 이전
`v1` 폴더를 기본값으로 다시 만들지 않는다. 핵심 경로 테스트:

```powershell
.\venv\Scripts\python.exe -m pytest research/tests/test_trajectory_sequence.py research/tests/test_sequence_simulator.py research/tests/test_anchor_augmentation.py -q
```
