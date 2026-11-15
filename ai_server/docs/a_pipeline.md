# A Pipeline Order

## Objective

The A server should not classify bird vs drone directly. Its job is to produce
stable, high-quality `TrackSequence` outputs for downstream feature extraction.

## Processing Order

1. Receive the video path, selected frame, and bounding box.
2. Load frames and assign `frame_index` and `timestamp_ms`.
3. Build foreground and background appearance models from the selected ROI.
4. Follow target features forward from the selected frame with KLT optical flow
   and a forward-backward consistency check.
5. Recover weak or lost observations with appearance and residual motion cues.
6. Apply feature-based camera motion compensation when requested.
7. Convert only accepted observations into normalized `TrackSequence.history`.
8. Store the processed frame dimensions used to normalize the coordinates.
9. Evaluate track quality and persist the shared JSON plus debug artifacts.

## Notes

- YOLOMG is intentionally not used by this pipeline.
- Frames before the selected ROI are `UNTRACKED`: A does not backfill boxes or
  trajectory points before `init_frame_index`.
- Runtime tracking moves through `ACTIVE`, `LOST`, and `EXITED`. An exit is
  confirmed only after the predicted center stays outside the frame for the
  configured number of frames.
- `trajectory.csv` and `TrackSequence.history` contain accepted visible
  observations only. `prediction`/`LOST`/`EXITED` states remain available in
  `debug/observations.csv`.
- Overlay boxes are drawn only for visible observations. LOST and EXITED frames
  can display their status but never a predicted box.
- Missing frames stay as frame-index gaps for B to interpolate.
- `metadata.json` records `tracking_start_frame`, `tracking_end_frame`,
  `exit_confirmed_frame`, `termination_reason`, and lifecycle frame counts.
- ROI-preceding and confirmed-exit frames are excluded from track missing-ratio
  calculations.
- `processed_width` and `processed_height` describe the post-resize frame used
  to normalize `cx`, `cy`, `w`, and `h`; they are not always the source size.
- A records a `timebase` block in `metadata.json`. The current OpenCV path uses
  frame index plus the video's average FPS and integer millisecond timestamps;
  true variable-frame-rate presentation timestamps are not yet supported.
- With CMC enabled, `cx` and `cy` use compensated coordinates. Debug overlays
  continue to use source-frame coordinates.
- `conf` is the combined observation confidence from KLT, appearance, motion,
  and camera-motion evidence. It is not YOLO confidence.
- Online appearance updates default to off to reduce background drift.
- Missing value interpolation belongs to B, not A
- Shared schema changes must be coordinated before merge
