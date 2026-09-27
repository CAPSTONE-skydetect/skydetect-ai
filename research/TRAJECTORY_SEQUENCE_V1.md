# B -> C 시계열 입력 계약 v1

작성일: 2026-09-27. 사용자 결정: MiniRocket + 선형분류기, **2초 창**.
구현: `trajectory_sequence.py`; 데이터 생성: `build_real_sequence_dataset.py`.
이 문서는 `MINIROCKET_C_PROPOSAL.md`의 미정이었던 창 길이와 입력 전처리를 구체화한다.
이번 범위는 실제 A 자료의 정리와 변환이다. C 모델 학습과 합성/증강은 포함하지 않는다.

## 1. 배열 계약

| 항목 | 값 |
|---|---|
| 계약 버전 | trajectory-sequence-1.0.0 |
| X | float32, `(N, 4, 60)`, 모두 유한값 |
| 채널 순서 | q_x, q_y, d_x, d_y |
| 시간 | timestamp_ms -> 초, 30Hz |
| 창 | 2초, 1초 stride; 원본 첫 관측 시각에 정렬 |
| 샘플 시각 | start + k/30, k=0..59; 마지막 샘플은 1.9667초 |
| 관측 지지 | start+2초까지 실제 원본 시간 범위가 있어야 함; 외삽 없음 |
| 정답 y | 문자열 bird/drone, shape `(N,)` |
| 함께 저장 | sample_id, group_id; 문자열 배열, pickle 없이 로드 |

`np.load(path, allow_pickle=False)`로 읽는다. 정답/그룹/메타데이터를 X에 결합하지 않는다.
2초보다 짧은 track은 시간 늘이기나 zero padding 없이 제외한다. 마스크 채널은 없다.
긴 track은 여러 창이 되지만 같은 원본의 모든 창은 같은 split이다.

## 2. 좌표·정규화

입력은 `stabilization.applied=true`인 실제 A export의 cx/cy이며 overlay raw 좌표를 대신 사용하지 않는다.
각 track의 processed_width/height가 필요하다. 두 값이 없거나 비정상이면 제외한다.

```text
p_i = (cx_i, cy_i * processed_height / processed_width)
c   = componentwise median(p_i over the 60 samples)
r   = quantile_0.9(norm(p_i - c))
s   = max(r, 0.0025)
q_i = (p_i - c) / s
d_0 = (0, 0)
d_i = q_i - q_(i-1), i >= 1
```

가로·세로에 같은 s를 적용해 종횡비와 상대적인 경로 모양을 보존한다. 화면 아래가 y 양수다.
변위는 1/30초당 정규화 이동량이며 m/s가 아니다. 카메라 원근·각도·시간에 따라 변하는 줌은 제거되지 않는다.
한 창의 전체 관측을 사용하므로 스트리밍에서도 2초 창이 완성된 이후 계산해야 한다.

`0.0025`는 영상 너비 단위(너비 1920 환산 시 4.8px)의 **초기 공학적 하한**이다.
실제 잡음에서 최적값을 추정한 것은 아니다. 낮은 이동량에서 잡음을 과도하게 확대하지 않도록 사용한다.
하한이 적용된 창은 `low_spatial_extent`를 표시하되 유지한다. 호버링을 제거하지 않기 위해서다.
하한 변경 시 계약 설정 ID도 바뀐다. 평가 자료로 값을 맞추지 않는다.

## 3. 시간·gap·필터

- frame_index는 증가하는 정수, timestamp_ms는 유한하고 엄격하게 증가하는 값이어야 한다. 정렬·중복 제거로 오류를 숨기지 않는다.
- 기본 프레임 주기 T는 `median(diff(timestamp_seconds)/diff(frame_index))`로 추정한다.
- 구간별 위 비율이 T와 `max(2ms, 10% T)`보다 크게 다르면 비일관 clock으로 제외한다.
- 관측 간격 dt가 1.5T보다 크면 누락 구간으로 보고 누락 지속시간을 dt-T로 계산한다.
- 누락 시간이 0.1초 + timestamp 반올림 허용 1ms를 넘는 구간을 가로지르는 창은 제외한다.
- 창 내부 누락 시간 합 / 2초가 10%보다 크면 제외한다. 누락 구간은 `[이전 관측+T, 다음 관측]`과 창의 교집합으로 합산한다.
- 허용된 짧은 gap에는 선형 보간을 적용한다. 메타데이터에 누락률과 보간 표시를 남긴다.
- 15Hz 미만은 현재 지원 범위 밖이다. 15~30Hz 업샘플은 시간축 정렬일 뿐 정보 복원이 아님을 표시한다.
- 약 30Hz는 추가 SG 없이 선형 재표본화한다. 현재 실제 54개는 모두 약 30Hz다.
- 30.3Hz 초과는 긴 gap으로 끊은 구간마다 원본 주기 등간격화, 4차 Butterworth low-pass(12Hz), 양방향 필터 후 30Hz 재표본화한다. 필터 지지가 부족하면 제외한다.
- 양방향 필터는 관측된 구간 내부의 미래 샘플을 쓴다. 스트리밍의 사용 가능 관측 범위가 달라지면 이를 별도 계약·테스트로 처리해야 한다.
- cx/cy가 0 또는 1인 관측은 `boundary_contact`로 표시한다. 실제 화면 경계와 CMC export clamp를 이것만으로 구별할 수 없으므로 자동 삭제하지 않는다.

기존 9개 특징 전처리 함수와 별개이며 `window_track(track, config)`가 새 입력의 공통 함수다.
향후 서비스에서 동일한 TrackSequence 관측 범위와 설정을 이 함수에 전달한다.

## 4. 원본 그룹과 분할

그룹은 다음 근거를 전이적으로 병합한다: source_video_id, 정확히 같은 시간·중심 좌표와 해상도,
기존 진단에서 원본 JSON 해시가 일치하는 자료에 연결된 영상 바이트 해시, 파일명 파생 관계.
file name의 `(숫자)`, tracksequence 접미사, 같은 영상의 위/아래 새 표기를 정리한 이름은 **추정 근거**다.
영상 해시는 기존 진단 시점의 증거를 재사용하며 JSON 해시가 달라지면 연결하지 않는다.

원본 촬영 세션은 파일만으로 확정할 수 없다. `group_evidence.csv`와 `session_review.csv`에 검토 대상을 남긴다.
session_review.csv의 session_group에 같은 촬영임을 확인한 공통 ID를 적으면 추가 병합할 수 있다.
자동으로 묶인 그룹을 수동 값으로 쪼개지는 않는다. 정답이 다른 완전 동일 궤적은 생성 오류로 중단한다.
단순 동일 궤적 복제본은 한 번만 내보내고 source_inventory.csv에 duplicate 상태를 남긴다.

seed=20260927, 원본 그룹 수 기준 약 60/20/20으로 train/validation/test를 분리한다.
분할은 창 생성·품질 제외 전에 수행한다. 클래스별 그룹 수 기준으로 나누며 혼합 정답 원본도 쪼개지 않는다.
작은 층(5개 미만 그룹)은 train에 둔다. 따라서 소규모 자료에서는 원하는 비율과 정확히 일치하지 않을 수 있다.
train은 원본 그룹당 최대 8개 창을 균등한 인덱스로 선택한다. validation/test는 모든 유효 창을 보존한다.
원본 추가나 그룹 검토 후 재생성하면 분할이 달라질 수 있으므로 배포한 데이터셋의 split은 manifest와 함께 고정한다.

**현재 자료는 모두 기존 진단에 노출된 개발 자료다. test.npz는 개발용 그룹 holdout이며 미사용 최종 평가가 아니다.**
독립적인 촬영 검증과 새 자료 최종 평가는 이후 필요하다. `independent_evaluation_ready=false`를 manifest에 저장한다.
클래스 라벨은 입력 폴더에서 가져온다. drone 기종·행동·추적 정답 검토는 별도이며 아직 완료로 표시하지 않는다.

## 5. 재현 명령과 산출물

```powershell
.\venv\Scripts\python.exe -m pytest research/tests/test_trajectory_sequence.py -q
.\venv\Scripts\python.exe -m research.build_real_sequence_dataset --output research/output/real_sequences_v1
```

```text
real_sequences_v1/
  train.npz / validation.npz / test.npz
  metadata.csv              # 창별 원본, 라벨, 품질, 정규화 값, NPZ 행 번호
  source_inventory.csv      # 제외·중복을 포함한 전체 원본 해시와 split
  group_evidence.csv        # 그룹 병합 근거
  session_review.csv        # 촬영 세션 확인 후 추가 병합 입력 가능
  excluded_windows.csv      # 품질 제외 및 train 그룹 상한 사유
  split_summary.csv
  dataset_manifest.json     # 입력·설정·코드·산출물 해시와 평가 한계
  sequence_examples.png
  report.md
```

생성기는 기존 출력 폴더가 비어 있지 않으면 중단한다. 재생성은 새 폴더에 한다.
검토한 session_review.csv를 적용하려면 `--session-overrides <csv 경로>`를 추가한다.
동일 seed와 입력이면 배열·라벨·ID·분할이 재현된다. 생성 시각 및 ZIP 헤더 등 바이트 차이는 배열 재현과 구별한다.
research/output은 gitignore 대상이며 코드·계약·테스트는 Git 추적 대상이다.

## 6. C 인수 조건

1. `X.shape[1:] == (4, 60)` 및 유한값, 채널 순서를 확인한다.
2. NPZ의 group_id 교차가 없고 metadata의 sample_id/npz_row가 맞는지 확인한다.
3. MiniRocket과 scaler는 train에만 fit한다. 기본 row 단위 CV로 같은 그룹을 재분할하지 않는다.
4. 평가 시 창을 독립 촬영처럼 세지 않고 track/원본 그룹별로 집계한다.
5. 품질 제외·낮은 이동량·경계 접촉별 결과와 전체 분모를 보고한다.
6. 현재 결과는 개발 자료 평가로 보고한다. C가 학습하기 전 그룹 검토의 잠정 상태를 확인한다.

이 전달본에는 실제 기반 증강, 시뮬레이터 샘플, 클래스 균형을 위한 복제, MiniRocket 학습이 없다.
다음 단계에서 같은 입력 계약을 사용해 실제 train 증강 및 합성 train을 추가하고 별도 버전으로 배포한다.
