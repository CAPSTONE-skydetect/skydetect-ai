# A Pipeline Order

## Objective

The A server should not classify bird vs drone directly. Its job is to produce
stable, high-quality `TrackSequence` outputs for downstream feature extraction.

## Processing Order

1. Receive the video path, selected frame, and bounding box.
2. Load frames and assign `frame_index` and `timestamp_ms`.
3. Build foreground and background appearance models from the selected ROI.
4. Follow target features with bidirectional KLT optical flow.
5. Recover weak or lost observations with appearance and residual motion cues.
6. Apply feature-based camera motion compensation when requested.
7. Convert only accepted observations into normalized `TrackSequence.history`.
8. Evaluate track quality and persist the shared JSON plus debug artifacts.

## Notes

- YOLOMG is intentionally not used by this pipeline.
- `prediction` states are written to debug CSV and omitted from `history`.
- Missing frames stay as frame-index gaps for B to interpolate.
- With CMC enabled, `cx` and `cy` use compensated coordinates. Debug overlays
  continue to use source-frame coordinates.
- `conf` is the combined observation confidence from KLT, appearance, motion,
  and camera-motion evidence. It is not YOLO confidence.
- Online appearance updates default to off to reduce background drift.
- Missing value interpolation belongs to B, not A
- Shared schema changes must be coordinated before merge
