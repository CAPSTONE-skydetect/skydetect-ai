# SkyDetect-AI

Track and trajectory based bird/drone analysis bootstrap repository.

## Current Scope

The repository is aligned around the A pipeline:

1. video input intake
2. stabilization
3. small flying object detection
4. object tracking
5. `TrackSequence` generation
6. quality checks
7. JSON response/output

Classification is intentionally out of scope for the A server. The shared
contract lives in [ai_server/schemas.py](/Users/yuchan/Desktop/git/skydetect-ai/ai_server/schemas.py).

## Repository Layout

```text
ai_server/
├── main.py
├── routers/
├── services/
├── utils/
├── docs/
└── schemas.py
dummy_track.json
research/
```

## Quick Start

```bash
uvicorn ai_server.main:app --reload
```

The `/analyze` endpoint is currently a bootstrap stub that returns a valid
`TrackSequence` payload so A/B/C can align on the shared interface first.
