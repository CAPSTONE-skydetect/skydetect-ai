# SkyDetect-AI

Track- and trajectory-based bird/drone analysis repository for the Sky Detect
project.

## 전체 시스템 실행 및 제출

이 레포는 SkyDetect 전체 시스템의 **실행 안내 시작점**입니다. 코드는
각 레포에서 관리하며, Docker Compose는 세 레포를 묶어서 실행합니다.
이 통합 브랜치에서는 AI 모델별 워크트리가 더 이상 필요하지 않습니다.

| 레포 | 담당 | 파이 통합 PR 대상 |
| --- | --- | --- |
| [skydetect-backend](https://github.com/CAPSTONE-skydetect/skydetect-backend) | Spring API, 클립 이력, MySQL | `dev` |
| [skydetect-frontend](https://github.com/CAPSTONE-skydetect/skydetect-frontend) | 라이브 영상, 클립 재생, ROI 지정, 결과 화면 | `main` |
| [skydetect-ai](https://github.com/CAPSTONE-skydetect/skydetect-ai) | AI 서버, Docker Compose, 파이 송출 및 클립 추출 워커 | `dev` |

RF와 MiniRocket은 동일 AI 커밋의 Docker target `rf`, `minirocket`에서
빌드되는 별도 서비스입니다. A 코드는 공유하고 모델·의존성·분류 계약은
분리합니다. 프론트의 기존 모델 선택과 API 경로는 유지합니다.
코드 위치와 학습 협업 방법은 [통합 기록](docs/model_unification.md)을 참고하세요.

- **파이 없이 실행:** [기본 Docker Compose 가이드](docker/README.md).
- **파이 통합 재현:** [호환 커밋, 로컬 설정, 실행 및 검증 가이드](hardware/raspberry-pi/README.md).
  AI 통합 후보는 `codex/ai-model-unification`이며 기존 파이 연동 작업을 포함합니다.
  프론트·백엔드는 기존 파이 연동 버전을 그대로 사용합니다.

### PR 및 최종 제출 순서

1. 레포별 작업 브랜치를 올리고 **백엔드 `dev` → 프론트 `main` →
   AI `dev`** 순서로 PR을 검토합니다. AI의 Pi Compose가 새 Spring 클립
   API를 사용하므로, 백엔드와 프론트 버전을 먼저 맞춥니다.
2. 실제 파이에서 **촬영 → 클립 저장 및 재생 → ROI 지정 → 추적 및
   분류**를 확인합니다. 새 Spring 경로의 저장·재생·재시작 복원은 합성
   녹화로 확인했으며, 실제 파이로 전체 흐름을 검증하는 작업은 남아 있습니다.
3. 통과한 AI·백엔드·프론트·RF·MiniRocket의 정확한 커밋과 검증 결과를
   [파이 가이드](hardware/raspberry-pi/README.md)에 기록한 뒤, 최종 제출에는
   그 문서를 포함하는 AI 레포 버전의 링크를 공유합니다.

## Current Scope

The official FastAPI implementation lives under `ai_server/`.

The repository is organized so A/B/C can work in parallel on a shared
contract:

1. A: manual ROI intake, stabilization, tracking, and `TrackSequence` generation
2. B: interpolation and feature extraction from track history
3. C: classification, model training, and explanation helpers

The shared contract lives in `ai_server/contracts/common.py`. Model contracts
live in `ai_server/classifiers/{rf,minirocket}/contracts.py`.
`ai_server/schemas.py` retains compatibility imports for the selected runtime.

## Project Structure

```text
skydetect-ai/
├── ai_server/                            # FastAPI implementation root
│   ├── main.py                           # A: server entrypoint / FastAPI app
│   ├── schemas.py                        # Compatibility exports (default RF)
│   ├── contracts/common.py               # Shared A/B contracts
│   ├── classifiers/rf/                   # RF contracts, inference, train/evaluate
│   ├── classifiers/minirocket/           # MiniRocket contracts, inference, train/evaluate
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
│   │   ├── rule_filter.py                # C: rule-based pre-filter
│   │   ├── classifier.py                 # C: classifier inference
│   │   └── train.py                      # C: training entrypoint
│   ├── utils/
│   │   ├── quality.py                    # A/B: track quality helpers
│   │   ├── interpolate.py                # B: interpolation helpers
│   │   ├── fractal.py                    # B: SBFD/LHFD helper functions
│   │   └── explain.py                    # C: feature importance helpers
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
  - `ai_server/services/rule_filter.py`
  - `ai_server/services/classifier.py`
  - `ai_server/services/train.py`
  - `ai_server/utils/explain.py`
  - `ai_server/models/`
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
