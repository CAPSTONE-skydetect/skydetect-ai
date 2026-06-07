# YOLOMG A → B/C 궤적 전달 자료

이 디렉터리는 실제 드론 영상에서 생성한 `TrackSequence`를 B 파트의
특징 추출과 C 파트의 분류 실험에 사용할 수 있도록 정리한 전달 자료입니다.

이번 작업에서는 공용 스키마를 변경하지 않았습니다. 이 디렉터리의 파일은
현재 `ai_server.schemas.TrackSequence` 규약을 따르는 실험 fixture와
재현용 메타데이터입니다.

## 기준 실험

- 영상 ID: `drone_fix_stab_01`
- 정답 라벨: `drone`
- 원본 파일명: `drone:fix:stab:1.mpg`
- 원본 SHA-256:
  `547d06655091ad20ad9265556db66604f6c369759b2a8e7b444f4c4c4e3e443e`
- 원본 정보: 1920x1080, 50fps, 400프레임, 8초
- 평가 구간: 영상 시작부터 5초, 총 250프레임
- 검출기: YOLOMG
- 트래커: `sort_center`
- confidence threshold: `0.20`
- 입력 이미지 크기: `1280`
- motion blur kernel: `11`
- 안정화 방식: `none` (원본이 이미 고정·안정화된 영상)
- YOLOMG 커밋:
  `090a74cd3ece15c66e857cc2e08e01cb103c4550`
- weight SHA-256:
  `fb28a4063f0e935ce419db47cf6ff26c443d12940eb727522a95644adba34bef`

## 제공 파일

- `drone_fix_stab_01/tracks_all.json`
  - `sort_center`가 생성한 4개 track 후보 전체
- `drone_fix_stab_01/selected_track.json`
  - B/C 실험에 우선 사용하도록 선정한 대표 `TrackSequence`
- `drone_fix_stab_01/selection_report.json`
  - 대표 track 선정 기준과 후보별 품질
- `drone_fix_stab_01/summary.csv`
  - 검출기 및 트래커별 실험 요약

대표 track은 ID 1이며 품질은 다음과 같습니다.

- 프레임 2~247 구간에서 246포인트
- track 내부 누락률 `0.0`
- 평균 confidence `0.814`
- 평가한 250프레임 대비 `98.4%` 포인트 확보

대표 track 선정 기준은 다음 우선순위를 사용했습니다.

1. `quality.num_points` 내림차순
2. `quality.missing_ratio` 오름차순
3. `quality.mean_conf` 내림차순

이 기준은 현재 B/C 실험을 시작하기 위한 권장값이며 최종 운영 정책은 아닙니다.
나머지 후보도 downstream 단계의 track 선택 및 저품질 track 거부 정책을 시험할
수 있도록 `tracks_all.json`에 그대로 보존했습니다.

정답 라벨은 `selection_report.json`의 실험 메타데이터에만 기록했습니다.
B/C로 전달되는 `TrackSequence`에는 정답 라벨을 넣지 않아 데이터 누수를
방지했습니다.

## YOLOMG 없이 B/C 테스트하기

B 파트에서는 `selected_track.json`을 바로 `TrackSequence`로 읽을 수 있습니다.

```python
import json
from pathlib import Path

from ai_server.schemas import TrackSequence

path = Path(
    "ai_server/docs/yolomg_handoff/"
    "drone_fix_stab_01/selected_track.json"
)
track = TrackSequence.model_validate(json.loads(path.read_text()))
```

속도와 가속도처럼 시간에 의존하는 특징은 `frame_index`에 고정된 FPS를
임의로 적용하지 말고 `timestamp_ms`를 기준으로 계산해야 합니다.

원본 궤적은 50fps 영상에서 생성됐지만 현재 RF 학습 데이터는 30fps
시뮬레이션에서 만들어졌습니다. 30fps 리샘플링을 실험할 경우 다음 원칙을
권장합니다.

- 원본 `TrackSequence`는 수정하지 않고 보존
- 리샘플링 및 보간 결과는 별도 데이터로 생성
- 보간 방식과 최대 허용 gap을 기록
- 보간을 사용했다면 `feature_status`와 `imputed_fields`에 반영
- 50fps 원본 특징과 30fps 리샘플링 특징을 각각 C에 입력해 결과 비교

## 영상부터 결과 재현하기

로컬에 YOLOMG 저장소와 weight 파일을 준비한 뒤 다음 명령을 실행합니다.

```bash
python scripts/run_detector_eval.py \
  --dataset-dir "/path/to/dataset" \
  --output-root artifacts/experiments/yolomg_handoff_drone_fix_stab_01 \
  --detectors yolomg \
  --trackers sort_center \
  --stabilization-methods none \
  --conf 0.20 \
  --imgsz 1280 \
  --eval-clip-sec 5 \
  --video-id drone_fix_stab_01 \
  --yolomg-repo /path/to/YOLOMG \
  --yolomg-weights /path/to/best.pt
```

생성된 tracker 결과를 B/C 전달 형식으로 내보내려면 다음 명령을 사용합니다.

```bash
python scripts/export_track_handoff.py \
  --input artifacts/experiments/yolomg_handoff_drone_fix_stab_01/tracks/drone_fix_stab_01__none__yolomg__conf020__sort_center.json \
  --output-dir /tmp/drone_fix_stab_01_handoff \
  --ground-truth-label drone
```

## B 파트 확인 요청

1. `selected_track.json`에서 핵심 궤적 특징 5개를 추출해 주세요.
2. 원본 50fps 특징과 `timestamp_ms` 기반 30fps 리샘플링 특징을 비교해 주세요.
3. 사용한 리샘플링, 보간, smoothing 정책을 기록해 주세요.
4. 리샘플링 전후의 `v_mean`, `v_std`, `a_mean`,
   `heading_change_ratio`, `maneuverability_sigma` 변화를 확인해 주세요.
5. 보간이 필요한 경우 `feature_status`와 `imputed_fields` 처리 방식을
   확인해 주세요.

## C 파트 확인 요청

1. 50fps 원본 특징과 30fps 리샘플링 특징을 각각 RuleFilter와 RF에
   입력해 주세요.
2. 각 입력의 최종 label, confidence, reject reason을 기록해 주세요.
3. `tracks_all.json`의 나머지 저품질 후보가 실제 객체처럼 분류되지 않고
   적절히 거부되는지 확인해 주세요.
4. 현재 RF가 시뮬레이션 데이터로 학습됐다는 점을 고려해 실제 영상 궤적과의
   분포 차이를 확인해 주세요.

## 현재 해석 시 주의사항

- 이번 fixture는 A 파트에서 실제 드론 궤적을 B/C 스키마로 전달할 수 있음을
  확인하기 위한 기준 자료입니다.
- 이 단일 영상 결과만으로 YOLOMG, tracker 또는 RF의 전체 성능을 확정할 수
  없습니다.
- 대표 track은 연속성이 높지만, 같은 영상에서 저품질 후보 track도
  생성됐습니다. 운영 파이프라인에는 별도의 후보 선택 또는 거부 정책이
  필요합니다.
- 50fps와 RF 학습 기준 30fps 사이의 시간축 정합성은 B/C 통합 과정에서
  반드시 검증해야 합니다.
