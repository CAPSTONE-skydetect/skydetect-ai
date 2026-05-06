# SkyDetect-AI

Track- and trajectory-based bird/drone analysis repository for the Sky Detect
project.

## Current Scope

The official FastAPI implementation lives under `ai_server/`.

The repository is organized so A/B/C can work in parallel on a shared
contract:

1. A: video intake, stabilization, detection, tracking, and `TrackSequence`
   generation
2. B: interpolation and feature extraction from track history
3. C: classification, model training, and explanation helpers

The shared contract lives in [ai_server/schemas.py](/Users/yuchan/Desktop/git/skydetect-ai/ai_server/schemas.py).

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
│   │   ├── detector.py                   # A: small flying object detection
│   │   ├── stabilization.py              # A: stabilization metadata/helper
│   │   ├── tracker.py                    # A: tracking and TrackSequence creation
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

## Quick Start

```bash
uvicorn ai_server.main:app --reload
```

The `/analyze` endpoint is still a bootstrap stub that returns a valid
`TrackSequence` payload so A/B/C can integrate against the shared contract
before full pipeline logic is connected.
