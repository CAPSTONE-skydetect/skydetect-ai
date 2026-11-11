# SkyDetect-AI

Track- and trajectory-based bird/drone analysis repository for the Sky Detect
project.

## Current Scope

The official FastAPI implementation lives under `ai_server/`.

The repository is organized so A/B/C can work in parallel on a shared
contract:

1. A: manual ROI intake, stabilization, tracking, and `TrackSequence` generation
2. B: interpolation and feature extraction from track history
3. C: MiniRocket + Ridge trajectory-sequence classification, training, and evaluation

The shared contract lives in `ai_server/schemas.py`.

## Project Structure

```text
skydetect-ai/
├── ai_server/                            # FastAPI implementation root
│   ├── main.py                           # A: server entrypoint / FastAPI app
│   ├── schemas.py                        # Shared A/B/C contract source
│   ├── routers/
│   │   ├── analyze.py                    # A: analysis endpoint
│   │   └── classify.py                   # C: classify endpoint
│   ├── services/
│   │   ├── detector.py                   # A: unused future detector hook
│   │   ├── manual_roi_tracker.py         # A: manual ROI tracking engine
│   │   ├── online_appearance.py          # A: online appearance matching
│   │   ├── foreground_motion.py          # A: motion recovery and Kalman filter
│   │   ├── tracking_adapter.py           # A: observation-to-contract adapter
│   │   ├── tracking_video_io.py          # A: video and artifact I/O
│   │   ├── stabilization.py              # A: stabilization metadata/helper
│   │   ├── tracker.py                    # A: API pipeline orchestration
│   │   ├── feature_core.py               # B: core motion feature extraction
│   │   ├── feature_signal.py             # B: extended signal/fractal features
│   │   ├── sequence_model.py             # C: MiniRocket/Ridge shared core
│   │   ├── classifier.py                 # C: TrackSequence inference
│   │   ├── prediction.py                 # C: response assembly
│   │   ├── train.py                      # C: training entrypoint
│   │   └── evaluate.py                   # C: three-arm validation comparison
│   ├── utils/
│   │   ├── quality.py                    # A/B: track quality helpers
│   │   ├── interpolate.py                # B: interpolation helpers
│   │   └── fractal.py                    # B: SBFD/LHFD helper functions
│   ├── models/                           # C: trained model artifacts
│   └── docs/                             # Shared docs, examples, contracts
├── research/                             # B: personal experiments/validation
├── tests/                                # unit tests
├── dummy_track.json                      # Sample payload for manual testing
├── requirements.txt
└── README.md
```

## Ownership Guide

- A
  - `ai_server/main.py`
  - `ai_server/routers/analyze.py`
  - `ai_server/services/detector.py`
  - `ai_server/services/stabilization.py`
  - `ai_server/services/tracker.py`
- B
  - `ai_server/services/feature_core.py`
  - `ai_server/services/feature_signal.py`
  - `ai_server/utils/interpolate.py`
  - `ai_server/utils/fractal.py`
- C
  - `ai_server/routers/classify.py`
  - `ai_server/services/sequence_model.py`
  - `ai_server/services/classifier.py`
  - `ai_server/services/prediction.py`
  - `ai_server/services/train.py`
  - `ai_server/services/evaluate.py`
  - `models/minirocket_classifier.joblib`
- Shared
  - `ai_server/schemas.py`
  - `ai_server/utils/quality.py`
  - `ai_server/docs/`

## Working Rules

- Official implementation changes should be made under `ai_server/`.
- `ai_server/schemas.py` is the single source of truth for shared interfaces.
- `research/` is for experimentation and validation, not the shared runtime
  implementation.
- YOLOMG is not part of the active A pipeline. Tracking starts from a user-selected
  bounding box.

## Quick Start

```bash
uvicorn ai_server.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

The `/analyze` endpoint accepts a video path and an initial bounding box in source
video pixel coordinates:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "source_video_id": "sample-001",
    "video_path": "/absolute/path/to/video.mp4",
    "init_frame_index": 0,
    "target_bbox": [100, 80, 32, 32],
    "stabilize": true,
    "write_overlay": true,
    "tuning": {
      "klt_accept_conf": 0.40,
      "recovery_conf": 0.58,
      "update_conf": 0.76,
      "search_radius_multiplier": 2.5,
      "exit_confirmation_frames": 3,
      "online_update_enabled": false
    }
  }'
```

Runtime artifacts are written below `artifacts/manual_tracks/` by default. Set
`SKYDETECT_TRACK_OUTPUT_DIR` to override that location.

Part B computes runtime trajectory features in the canonical
`fhd_width_1920_v1` coordinate space. A may still track a Full HD or 4K source
at 1280px width; B uniformly maps the normalized trajectory back to a
1920px-wide coordinate system before calculating pixel/s and pixel/s² features.

## Part C: MiniRocket Classifier

This branch replaces the former 9-feature Random Forest with
MiniRocket + StandardScaler + RidgeClassifier on 2-second trajectory windows.
See `docs/minirocket_classifier.md` for the contract, commands, and limits, and
`docs/minirocket_evaluation.md` for the current development comparison.

```bash
# unpack real_reference_comparison_v1.zip into research/output/ first
python -m ai_server.services.train      # -> models/minirocket_classifier.joblib
python -m ai_server.services.evaluate   # -> reports/minirocket/
```

## Stack: 새 데이터로 처음부터 학습하기

**기존 학습 영상, `real_reference_comparison_v1.zip`, 기존 모델 가중치는
필요하지 않습니다.** 위의 `ai_server.services.train` 명령은 기존 패키지용입니다.
새로 수집한 TrackSequence로 Stack을 학습할 때는 아래의 `--collection` 경로를
사용합니다. Stack 기능은 `model/minirocket`의 `fe067103`부터 포함됩니다.

협업자에게 전달할 내용:

- 기존 MiniRocket/Scaler/Ridge와 궤적 전처리는 재사용합니다.
- 추가한 방법은 **이동 크기 보존 표현 + 출처 그룹 OOF 기반 점수 결합**입니다.
- 학습 시 두 MiniRocket 구성 모델, 각 Scaler/Ridge, Stack 결합기를 모두 새로
  fit 합니다. 기존 체크포인트를 이어서 학습하는 방식이 아닙니다.
- 새 TrackSequence JSON과 정답 라벨, 원본 해시, 촬영 그룹, 고정된
  train/validation/test 분할을 담은 manifest를 별도로 준비해야 합니다.
- 같은 촬영 세션/원본/객체의 부분 구간을 서로 다른 분할로 나누면 안 됩니다.
  현재 코드는 같은 그룹에 새와 드론이 섞인 자료를 지원하지 않으므로, 그 경우
  그룹을 라벨별로 쪼개지 말고 평가 계약을 먼저 검토합니다.
- 기존 서비스의 기본 모델과 `models/minirocket_classifier.joblib`은 바뀌지
  않습니다. 새 모델은 지정한 출력 폴더의 `stack.joblib`로 저장됩니다.

[설치, manifest 필드, 그룹 제약, 전체 학습/평가 안내](docs/minirocket_stack_training.md)를
먼저 확인하세요. 아래는 환경 설치 및 실제 manifest 준비 후 저장소 루트에서
실행할 명령입니다. `/path/to/new_dataset/manifest.json`은 실제 경로로 바꿉니다.

```bash
# 1. 입력 계약, 해시, 촬영 그룹 및 CV 가능 여부 확인 (학습하지 않음)
python -m research.minirocket_stack_training audit \
  --collection /path/to/new_dataset/manifest.json

# 2. 승인된 새 자료로 처음부터 학습 (출력 폴더는 없어야 함)
python -m research.minirocket_stack_training train \
  --collection /path/to/new_dataset/manifest.json \
  --output research/output/stack-new-01

# 3. 검증 자료 평가
python -m research.minirocket_stack_training evaluate \
  --collection /path/to/new_dataset/manifest.json \
  --model research/output/stack-new-01/stack.joblib \
  --split validation
```

최종 test 평가는 모델/설정을 고정한 뒤에만 실행합니다. 새 데이터와 모델 파일은
코드와 별도로 관리하고, 학습 성능은 같은 분할의 새 Baseline과 비교해야 합니다.
위 명령을 문서에 추가한 것일 뿐, 실제 데이터 학습을 실행한 것은 아닙니다.
