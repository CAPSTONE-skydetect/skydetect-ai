# 시계열 Sim-to-Real 원인 진단과 단일 후보 검증

작성일: 2026-09-28. 대상: B/C 담당자. 현재 상태: 대량 합성 데이터 채택 보류.

## 이번 작업의 결정

실제 train 원본 33개와 저장된 합성 fit 비행을 대조했다. 관측 오차 요소를 하나씩 끈 비교,
원본별 시계열 그림 33개, 실제 A overlay의 두 시점 화면 27개를 생성했다.
그 근거로 클래스 공통 이동량 조건부 잔차 모델 하나를 구현하고 train 내부 3-fold CV로 검증했다.
CV의 일관된 개선 조건을 통과하지 않아 채택하지 않았다. 이번 작업에서 validation/test 좌표를 다시 읽지 않았다.

현재 구현 버전은 `sequence-guidance-1.3.0`이다. 기존 물리 모델은 4.0.0, 입력 계약은 `trajectory-sequence-1.0.1`이다.
잔차 모델의 새 옵션 기본값은 0이며 기존 관측 모델의 동작은 유지한다. 실패한 설정이 자동 적용되지 않는다.

## 원인 분리

같은 합성 비행, 같은 카메라를 사용한다. `optical_same_frames`는 관측과 같은 frame/gap을 유지하고,
그 좌표만 잡음 전 투영 좌표로 치환한다. 다른 ablation은 한 관측 설정씩 바꾸지만 RNG 소비가 달라질 수 있으므로 고립된 인과 효과로 해석하지 않는다.
모든 변형에 공통으로 남은 196개 창, 34개 합성 비행 그룹을 비교했다. 유효 창 조건부 분석이며 전체 생성 가용성 결과는 별도로 본다.

| 진단 변형 | 실제 train과의 거리 |
|---|---:|
| 이전 보정 후보의 관측 | 0.8173 |
| 같은 frame/gap의 잡음 전 투영 | 0.9803 |
| jitter 제거 | 0.9643 |
| drift 제거 | 0.8188 |
| CMC 잔차 제거 | 0.8155 |
| dropout/burst 제거 | 0.8210 |

작은 숫자 차이를 통계적으로 유의한 효과로 주장하지 않는다. 이 코호트에서는 단순한 잡음 제거가 전체 차이를 줄이지 못했다.

- 새의 2초 화면 이동 폭 중앙값은 실제 train 0.2779, 합성 0.1729다. 단위는 화면 너비 비율이다.
- 새의 합성 추진 비행에는 지나치게 매끈한 창이 있고, 작은 투영 이동의 일부 활공 창에는 잡음으로 인한 급방향 변화가 있다.
- 새의 짧은 변위 방향 반전 지표 거리는 관측 2.1922, 잡음 전 0.1246이다. 이 수치를 조종 명령의 U턴 빈도와 동일시하면 안 된다.
- 반대로 잡음 제거 시 자기상관/대역 지표 차이는 커진다. 실제 잔차에는 물체 운동과 A 관측 과정이 함께 있다.
- 드론의 흔들리는 호버와 실제 연속 선회/감속을 동일한 고정 잡음 크기로 설명하기 어렵다.
- 전체 비행과 2초 창의 모습은 다르다. 긴 overlay의 원형 경로가 2초 창에서는 짧은 곡선으로 나타날 수 있다.

0.3초 추세를 양쪽에 동일하게 적용한 진단도 기록했다. raw/coarse는 각자의 train 척도로 정규화하므로 점수를 직접 비교해 개선이라 부르지 않는다.
이 추세 처리는 진단만을 위한 것이며 C에 전달하는 4채널 입력에 추가하지 않았다.

## 추가로 발견한 카메라 한계

저장된 합성 비행 36개 중 8개에서 카메라의 z가 가상 지면 0보다 낮았다.
현재 카메라는 비행 중앙점에서 거리와 관측 각도를 역산하여 배치하므로 실제 지상 설치 위치를 보장하지 않는다.
따라서 현재 투영 범위 보정을 실제 거리·설치 높이 복원으로 해석할 수 없다.
잔차 후보 검증 도중 카메라까지 동시에 바꾸지는 않았다. 지상 촬영 시나리오를 채택하려면 비행 높이와 카메라 높이/시선을 함께 제약하고 새 그룹 검증을 거쳐야 한다.

## 원본 그룹 검토

기존 manifest의 source ID, 좌표 hash, 영상 hash, 파일명 묶음 근거를 다시 검사했다. split을 가로지르는 동일 근거 연결은 0개다.
전체 split은 metadata만 확인했으며 validation/test 파일의 영상·좌표를 열지 않았다.
train 33개 중 6개는 원본 영상 hash가 없다. 나머지 27개는 기존 audit과 연결된 A overlay가 있어 두 시점의 화면을 저장했다.
upload ID와 영상 hash가 다르다는 것만으로 촬영 세션의 독립성을 증명할 수는 없다. 파일명만 보고 촬영 세션을 임의로 확정하지 않았다.
`case_index.csv`에는 추후 촬영 출처 확인용 session_group/session_review 칸이 있다.

## 검증한 수정 하나

`jitter_motion_fraction` 옵션으로 다음 공통 관측식을 추가했다.

```text
sigma_px = sqrt(floor_px^2 + (motion_fraction * optical_step_px)^2)
```

각 fold의 fit 그룹에서만 0.3초 추세 잔차와 화면상 이동량을 측정했다.
원본 그룹·클래스 총가중치를 같게 두고 강건한 비선형 최소제곱으로 계수 두 개를 적합했다.
AR1=0.55의 잔차 필터 감쇠를 반영했으나 실제 운동이 잔차에 포함되므로 순수 추적 오차의 식별은 아니다.
클래스별 계수, 조류 전용 주파수, 특징 벡터의 사후 변형은 사용하지 않았다.
카메라·행동·seed·상관 계수는 대조 후보와 동일하다.

| Fold | 이전 후보 | 이동량 잔차 후보 |
|---|---:|---:|
| 0 | 0.9614 | 0.9513 |
| 1 | 0.8233 | 0.9174 |
| 2 | 1.0355 | 0.9561 |
| 평균 | 0.9401 | 0.9416 |

유효 비행 비율은 양쪽 모두 평균 0.8611이다. full-train 추정값은 floor 0.4213px, motion fraction 0.1176이지만 채택된 운영 설정이 아니다.
평균 차이는 작고 fold별 방향도 다르다. 개선을 입증하지 못했다는 결론이며 통계적으로 악화가 확정됐다는 뜻은 아니다.

평가 전 선언한 gate는 평균 거리/가용성 점수 개선, 모든 fold 거리 개선, 유효 비행 비율 비감소다.
`rejected_by_train_cv`로 판정했고 validation 및 실제 test는 열지 않았다. 후보를 추가 검색하지 않았다.

## 산출물과 재현

| 경로 | 내용 |
|---|---|
| output/sequence_gap_diagnosis_v2/report.md | 관측 요소별 진단 |
| output/sequence_gap_diagnosis_v2/cases/ | 2초 창 비교 33개, A overlay 화면 27개 |
| output/sequence_gap_diagnosis_v2/case_index.csv | 실제 파일과 비교 그림/합성 sample의 연결 |
| output/sequence_gap_diagnosis_v2/camera_geometry.csv | 카메라 높이·거리·FOV 진단 |
| output/motion_residual_v1/report.md | 단일 수정 후보의 CV 결과 |
| output/motion_residual_v1/fold_evidence.json | fold별 원본 그룹과 적합 근거 |
| 각 결과 폴더의 protocol.json/source_snapshot | 입력·코드 해시와 실행 코드 보존 |

```powershell
.\venv\Scripts\python.exe -m research.diagnose_sequence_gap --output research/output/sequence_gap_new_run
.\venv\Scripts\python.exe -m research.refine_motion_residual --output research/output/motion_residual_new_run
.\venv\Scripts\python.exe -m pytest research/tests -q
```

새 출력 폴더를 사용한다. 출력은 기존 gitignore 정책대로 로컬에 남기고 코드·테스트·이 명세를 커밋한다.
C에는 입력 계약 1.0.1을 공유하며 구 계약과 혼합하지 않는다. 실제 test도 최종 평가 전에 같은 고정 계약으로 별도 변환해야 한다.
이번 작업은 C 모델 학습이나 대량 혼합 증강을 포함하지 않는다. 검증되지 않은 합성을 최종 학습 자료로 배포하지 않는다.

## 검증 결과

`python -m pytest research/tests -q`: 185개 통과(422.50초), 노트북 실행 포함.
추가 회귀 테스트는 optical/observed의 동일 frame 유지, sample ID 정렬, test 파일을 읽지 않는 source 근거 검사,
공통 이동량 잔차의 크기 변화, fold별 악화를 숨기지 않는 CV gate, AR1 잔차 필터 감쇠를 검증한다.
커밋 전 기존 HEAD의 관측 코드와도 8개 seed/누락 프레임 조건에서 비교했으며 기본 설정의 관측 history가 정확히 같았다.
테스트 통과는 구현의 확인이며 합성 데이터의 현실성이 입증됐다는 의미는 아니다.
