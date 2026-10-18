# 후속 실험 실행 기록 (2026-10-18)

## 판단

D0를 유지한다. 신규 설정 N4, N4_native, N5 중 위치·관측률 회귀 기준을 함께
통과한 설정은 없다. 모든 후보는 선택 옵션이며 서비스 기본값은 D0다.

| 설정 | 위치 P95의 객체 간 중앙값 | 관측률 평균 | 가속도 P95 특징 상대오차 절댓값 중앙값 |
|---|---:|---:|---:|
| D0 | 3.851px | 99.68% | 59.62% |
| N4: 대비 보정 벡터 시간 결합 | 4.112px | 99.64% | 50.75% |
| N4_native: 위 설정 + 원본 해상도 | 5.266px | 86.18% | 전체 비교 불가 (5/6) |
| N5: 정렬된 차영상 2프레임 결합 | 3.818px | 94.08% | 83.35% |

위치 오차는 모두 폭 1280px 기준이다. 후보마다 관측 프레임이 달라 B matched-GT
통계의 비교 프레임 집합도 달라질 수 있다. 관측률 감소로 어려운 구간을 제외한
후 위치 점수만 높아진 후보를 채택하지 않는다. N4_native는 B가 결손 때문에
bird_2를 거부하므로 5개만의 중앙값을 6개 기준선과 직접 비교하지 않는다.

## 완료한 일

- 대비 보정 시간 결합 N4와 원본 해상도 조합 N4_native를 동일한 6개 객체로 실행했다.
- N5는 이전 차영상을 예상 표적 이동량으로 정렬해 현재 차영상과 0.5씩 결합한다.
  현재 프레임의 증거가 0이면 출력도 0이다. 검색 호출 사이에 프레임 공백이 생기면
  누적 상태를 초기화한다. 이는 E5의 재획득 지연과 구분되는 실제 영상 증거 결합이다.
- CMC replay가 A의 cmc_analysis_width를 무시하던 오류를 수정했다. 기존 E1 및
  N4_native는 추적을 재실행할 필요 없이 B 평가를 다시 수행했다. E1도 채택되지 않았다.
- default 대비 보정의 부동소수점 연산 순서를 보존해 이전 D0 수치가 재현되도록 했다.
- 다음 실행부터 worker별 소스 파일 SHA-256과 작업 설정을 run 디렉터리에 보존한다.

## drone_3 보조 holdout

원본 177프레임에서 CVAT 범위 0~174에 해당하는 앞 175프레임을 FFV1로 복사하고,
원본과 파생 영상의 모든 디코딩 BGR 픽셀 SHA-256이 일치함을 확인했다. 원본이나
XML을 수정하지 않았다. 프레임 오프셋은 0이며 초기 ROI는 첫 GT bbox 중심의 32x32다.

- GT 138프레임 중 95프레임 관측: 68.84%.
- 위치 오차 중앙값 2.787px, P95 21.954px (폭 1280px 기준).
- 후반의 누락과 중심 이탈이 확인됐다. 비교 영상과 6개 프레임 확대 이미지를 생성했다.
- XML은 마지막 구간에 동일 bbox가 반복된다. f174의 영상은 GT 재검수가 필요해 보인다.
  이 구간을 점수에서 사후 삭제하거나 GT를 자동으로 고치지 않았다.
- 이 객체는 원본에서 약 57~86px다. 10~30px 핵심 조건이나 독립적인 새 촬영 출처에
  대한 최종 일반화 검증으로 주장하지 않는다.

## C 실행 및 학습 준비

RF는 각 브랜치의 production 추론 서비스를 실행했다. MiniRocket은 production
MiniRocketClassifier를 실행하고 기존 research.predict_track과 창별 점수·라벨이
일치하는지도 검사했다. D0/N4/N5/holdout의 총 57개 MR 입력에서 모두 일치했다.

| A 궤적 | RF 정답 | MiniRocket 정답 |
|---|---:|---:|
| D0 | 1/6 | 4/6 |
| N4 | 1/6 | 4/6 |
| N5 | 0/6 | 4/6 |
| drone_3 | 1/1 | 0/1 |

이는 서로 다른 학습 자료와 표현을 사용하는 배포 구성 비교다. 알고리즘 자체의
우열을 입증하지 않는다. 서비스의 보류 정책도 RF 확률 0.6, MR margin 0으로 다르다.

새 common_c_training은 다음을 수행한다.

1. 명시적 train/validation/test 그룹 분리, 파일 해시, TrackSequence 스키마,
   real 도메인, 클래스와 human review_status=approved를 검증한다.
2. 동일한 2초 구간에서 MR 시계열과 실제 B v4 RF 특징을 만들고 양쪽 모두 처리할 수
   있는 공통 창만 사용한다. 거부된 창도 기록한다.
3. RF와 MR을 같은 train 그룹에만 fit하고 그룹별 창 수가 많아져 학습 가중치가
   늘지 않도록 원본 그룹당 같은 가중치를 적용한다.
4. validation은 궤적 단위로 창 점수를 집계하고 보류율·클래스별 지표를 산출한다.
   test 성능 평가는 --evaluate-test를 명시한 경우에만 실행한다.

테스트 fixture로 두 모델 학습과 validation 추론까지 실행했다. 실제 자료는 새의
원본 그룹이 1개이며 학습용 원본 패키지도 없어 preflight에서 보류됐다. 실제 추가 GT를
자동으로 만들거나 검수 완료로 바꾸지 않았다.

## 데이터 출처 점검

- 로컬 라벨링 디렉터리: 영상 12파일, 서로 다른 SHA-256 9개, XML 5개.
- 드론/4.mp4와 새/1.mp4는 같은 SHA-256이다. 폴더명을 클래스 라벨로 쓰면 안 된다.
- drone_1/1.mp4와 1.mpg는 해시는 다르지만 같은 프레임 수와 크기를 가지며 네 시점
  축소영상 MAE가 0.47~1.45/255다. 재인코딩 의심 쌍이므로 별도 그룹으로 분할하지
  않도록 검수가 필요하다. 다른 해시는 독립적인 촬영 세션을 보장하지 않는다.
- GitHub 공개 release assets에도 학습 패키지가 없었다. 협업자에게 메시지를 보내거나
  새로운 외부 데이터를 수집하지 않았다.

## 필요한 사람 입력

1. 서로 다른 촬영 출처의 새 영상 최소 2그룹을 CVAT GT로 추가한다. 이는 3분할을
   가능하게 하는 최소 조건이며 안정적인 성능 추정에 충분한 표본 수라는 뜻은 아니다.
2. drone_3 후반 visibility/outside 및 반복 bbox를 원본과 대조해 검수한다.
3. C 담당자에게 아래 자료를 받는다. 기존 고정 train/validation/test를 유지할 수 있도록
   source_group_id와 parent_track_id, GT/A/증강 계보가 필요하다.
4. A 설정을 고정한 뒤 새로운 촬영 출처의 10~30px 표적 영상으로 검증한다.

C 담당자에게 전달할 요청 문안:

> RF와 MiniRocket을 같은 실제 자료로 비교하려고 합니다. 현재 MiniRocket real_only
> 모델에 대응하는 원본 TrackSequence 목록, dataset_manifest.json, metadata.csv,
> train/validation NPZ 및 패키지 manifest와 해시를 부탁드립니다. 원본 영상 또는
> 촬영 세션 기준 source_group_id, parent_track_id, 클래스와 검수 상태, 기존 분할도
> 포함해 주세요. NPZ만으로는 B v4 RF 특징을 같은 관측 구간에서 만들기 어려워
> 원본 TrackSequence 및 창 시작·종료 시각 대응표도 필요합니다. test 좌표는
> 평가 프로토콜을 고정한 뒤 별도 평가에 사용하겠습니다.

## 재실행

저장소 루트: /Users/yuchan/Documents/형주플젝/skydetect-ai-trajectory-experiments

```sh
/Users/yuchan/Desktop/git/skydetect-ai/.venv/bin/python -m research.trajectory_experiments run \
  --manifest artifacts/trajectory_experiments/pilot_manifest.json \
  --output artifacts/trajectory_experiments --variants N4 N5
```

C 실행은 research/requirements-common-c-training.txt 환경을 사용한다. 새 output
디렉터리가 필요하며, preflight 차단 사유는 status.json에 남는다.

```sh
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python \
  -m research.common_c_training \
  --manifest artifacts/trajectory_experiments/pilot-20261017/common_c_dataset/dataset_manifest.json \
  --output artifacts/trajectory_experiments/common_c_next_run
```

## 산출물

- 통합 보고서: artifacts/trajectory_experiments/followup-20261018/index.html
- 기계 판독 결과: 같은 디렉터리의 summary.json
- 중복·GT 상태: local_inventory.json
- 신규 라벨링 후보 및 검수 작업: annotation_jobs.json
- D0/N4/N5/holdout production 추론: production_*/report.md 및 raw_results.json
- holdout 비교 영상: artifacts/trajectory_experiments/holdout-drone3-20261017/evaluation/D0/comparison_videos/
- 새 테스트: 전체 314 passed, 1 skipped (기존 노트북 제외). 이후 작은 수정의 영향 범위는
  추가 집중 테스트로 확인한다. 기존 노트북은 전 세션에서 별도 통과했다.
