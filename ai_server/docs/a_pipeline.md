# A Pipeline Order

## Objective

The A server should not classify bird vs drone directly. Its job is to produce
stable, high-quality `TrackSequence` outputs for downstream feature extraction.

## Processing Order

1. Receive input metadata and video path.
2. Load frames and assign `frame_index` and `timestamp_ms`.
3. Apply global motion compensation.
4. Detect small flying objects on stabilized frames.
5. Associate detections into tracks.
6. Build normalized `TrackSequence.history`.
7. Evaluate track quality.
8. Return and persist JSON output.

## Notes

- MVP stabilization target: FFmpeg `vidstabdetect` + `vidstabtransform`
- Extension path: OpenCV ECC refinement
- Missing value interpolation belongs to B, not A
- Shared schema changes must be coordinated before merge
