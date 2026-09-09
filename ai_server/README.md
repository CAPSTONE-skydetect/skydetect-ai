# AI Server

This folder contains the shared API and the manual ROI A-part tracker for
SkyDetect-AI.

## Current Focus

- keep the shared `TrackSequence` contract stable for B/C
- track a user-selected ROI without YOLOMG
- keep predictions and debug data outside the A-to-B handoff

## Files

- `schemas.py`: shared models for A/B/C handoff
- `tracking_schemas.py`: A-only manual tracking request and tuning models
- `routers/analyze.py`: manual ROI tracking endpoint
- `services/manual_roi_tracker.py`: KLT, appearance, motion, and CMC engine
- `services/tracking_adapter.py`: shared `TrackSequence` contract adapter
- `services/tracker.py`: API pipeline orchestration
- `utils/quality.py`: observed-track quality scoring
- `docs/a_pipeline.md`: A-stage processing order
