# SkyDetect-AI

Track- and trajectory-based bird/drone analysis repository for the Sky Detect
project.

## Current Scope

The official FastAPI implementation lives under `ai_server/`.

The repository is organized so A/B/C can work in parallel on a shared
contract:

1. A: manual ROI intake, stabilization, tracking, and `TrackSequence` generation
2. B: interpolation and feature extraction from track history
3. C: classification, model training, and explanation helpers

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
      "online_update_enabled": false
    }
  }'
```

Runtime artifacts are written below `artifacts/manual_tracks/` by default. Set
`SKYDETECT_TRACK_OUTPUT_DIR` to override that location.
