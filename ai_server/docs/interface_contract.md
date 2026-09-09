# Shared Interface Contract

## Priority

Freeze the shared schema before detector, tracker, or classifier implementation.
The current contract owner is `ai_server/schemas.py`.

## Core Handoff

- A output: `TrackSequence`
- B input: `TrackSequence`
- B output: `FeatureVector`
- C input: `FeatureVector`
- C output: `PredictionResult`

## Team Boundary

- A: input intake, stabilization, detection, tracking, `TrackSequence` creation, quality summary
- B: interpolation, normalization, trajectory and bbox-based feature extraction
- C: Random Forest classification, importance summary, final response assembly

## Current Proposal

- `TrackSequence.history` contains accepted observations only and is strictly
  ordered by frame without duplicates
- Internal prediction-only frames are not part of `history`; their absence is a
  frame-index gap
- `TrackPoint` coordinates and sizes are normalized to `[0, 1]`
- When stabilization is applied, `cx` and `cy` are camera-motion-compensated
- `TrackPoint.conf` is observation confidence, not detector confidence
- `StabilizationInfo` is included in `TrackSequence` so downstream stages know whether A applied global motion compensation
- `TrackQuality` is included in `TrackSequence` so B/C can filter or inspect track reliability without recomputing basic metadata
- Interpolation is not done by A and should happen in B

## Review Items For Team

- Confirm whether `stabilization` and `quality` should remain optional shared fields
- Confirm whether `track_id` is video-local or globally unique
- Confirm whether C wants a stricter schema than `quality: dict[str, str | int | float]`
