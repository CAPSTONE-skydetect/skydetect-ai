# AI Server Bootstrap

This folder contains the A-part bootstrap for SkyDetect-AI.

## Current Focus

- freeze shared schema first
- provide a stable `TrackSequence` contract for B/C
- keep detection, tracking, and stabilization loosely coupled

## Files

- `schemas.py`: shared models for A/B/C handoff
- `routers/analyze.py`: bootstrap FastAPI endpoint
- `services/stabilization.py`: stabilization metadata helper
- `services/detector.py`: detector placeholder
- `services/tracker.py`: track bootstrap generator
- `utils/quality.py`: basic quality scoring helper
- `docs/a_pipeline.md`: A-stage processing order
