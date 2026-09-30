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
- Manual ROI tracking starts at `init_frame_index`; A does not infer or export
  observations for earlier frames
- `trajectory.csv` contains visible observations only, while
  `debug/observations.csv` retains ACTIVE/LOST/EXITED lifecycle evidence
- Confirmed EXITED frames are diagnostic and are excluded from
  `TrackQuality.missing_ratio`
- `TrackPoint` coordinates and sizes are normalized to `[0, 1]`
- `TrackSequence.processed_width` and `processed_height` are the exact frame
  dimensions used as the normalization denominators, after any A-side resize
- Runtime B converts normalized centers to the canonical
  `fhd_width_1920_v1` coordinate space before feature extraction. The uniform
  scale is `1920 / processed_width`; canonical height preserves the input
  aspect ratio. This keeps speed and acceleration aligned with the existing
  Full HD training data without forcing A to track at Full HD.
- Non-16:9 inputs are currently accepted with
  `aspect_ratio_matches_training=false`; this is a warning-only policy until
  enough real A data exist to justify rejection.
- `timestamp_ms` is the primary B timebase and feature extraction resamples
  accepted segments to 30 Hz. An explicit `fps` is only a fallback when an
  external research record has no timestamps.
- When stabilization is applied, `cx` and `cy` are camera-motion-compensated
- `TrackPoint.conf` is observation confidence, not detector confidence
- `StabilizationInfo` is included in `TrackSequence` so downstream stages know whether A applied global motion compensation
- `TrackQuality` is included in `TrackSequence` so B/C can filter or inspect track reliability without recomputing basic metadata
- Interpolation is not done by A and should happen in B
- B emits `FeatureVector.provenance`. `feature_config_id` identifies only the
  research formula configuration, while `feature_contract_id` also binds the
  coordinate and timebase policies. Runtime and generated datasets must carry
  the same contract id before C compares or trains on them.
- B supports irregular timestamps when genuine timestamps are supplied, but
  the current A timestamp source is average-FPS based. VFR therefore remains an
  A acquisition limitation rather than a capability claimed by this contract.

## Review Items For Team

- Confirm whether `stabilization` and `quality` should remain optional shared fields
- Confirm whether `track_id` is video-local or globally unique
- Confirm whether C wants a stricter schema than `quality: dict[str, str | int | float]`
- Revisit the warning-only non-16:9 policy once real A validation data are
  available
