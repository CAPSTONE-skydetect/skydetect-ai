# GT tracking evaluation and Part B feature comparison

Run from the repository root with the project's Python environment:

```bash
python -m research.tracking_evaluation dataset /path/to/samples \
  --output /path/to/results --compare-b --replay-cmc
```

Each sample directory contains `sample.json`:

```json
{
  "sample_id": "clip_1_drone",
  "ground_truth": {"path": "annotations.xml", "track_ids": [0]},
  "prediction": {"trajectory": "run/trajectory.csv", "metadata": "run/metadata.json"},
  "alignment": {"prediction_frame_offset": 0, "source_is_trimmed": false},
  "provenance_note": "Describe original vs reproduced A output here.",
  "comparison_video": "comparison.mp4",
  "b_comparison": {"track_sequence": "run/track_sequence.json"}
}
```

`provenance_note`, `comparison_video`, and `b_comparison` are optional. By default,
the evaluator finds `track_sequence.json` and `debug/observations.csv` beside
`trajectory.csv`. Multiple explicitly selected GT track IDs must describe the
same object without overlapping frames. Never infer object identity from filenames.

## Three evaluation axes

1. Position: center distance, median and P95. Existing source-pixel measurements
   remain available. The human report uses a reference width of 1280 pixels,
   preserving the source aspect ratio; this is not a world-distance unit.
2. Size-relative position: distance divided by the same frame's GT width or bbox
   diagonal. Divide per frame before taking quantiles. Do not divide summary
   medians by one another. GT non-keyframe fraction is disclosed as an annotation
   provenance indicator, not proof that those frames were never reviewed.
3. Motion: `e(t) = A(t) - GT(t)` is a vector. Compare `norm(e(t)-e(t-1))`, its
   timestamp-normalized value in pixels/second, and A/GT path length. Use only
   identical consecutive matched frame pairs in the attempted tracking window.
   Preserve gaps; never join across missing observations. Report pair coverage
   beside the measurements. A stationary GT yields an undefined path ratio.

Motion metrics describe raw image coordinates and include annotation variation.
Time normalization accounts for elapsed time but does not eliminate all sampling
rate effects. Compare matched protocols; do not rank trackers from path ratio alone.

## Actual B runtime

The evaluator calls `ai_server.services.feature_core.build_feature_vector`,
using the unchanged Feature v4 defaults (30 Hz, 0.3-second smoothing, canonical
width 1920 pixels). Its nine feature values are stored with their units, version,
config fingerprint, status, rejection reasons, and runtime quality diagnostics.

Three inputs are exported as valid `TrackSequence` JSON objects:

- `gt_full`: every eligible GT frame in the attempted window.
- `gt_matched`: GT at exactly A's observed frame indices and timestamps.
- `a`: the original saved A history restricted to the same eligible GT window.

The common-frame comparison measures the feature differences from observed
locations at identical sampling. The full-GT comparison includes missingness.
`gt_sampling_only` compares matched GT with full GT. These differences do not
form an additive causal decomposition because feature extraction is nonlinear.
Missing points stay absent from A; the existing B runtime handles interpolation.
Quality is recalculated for the restricted diagnostic inputs, so this experiment
does not test C's quality gating or classification accuracy.

Relative differences are signed: `(A-GT)/abs(GT)*100`. If GT is near zero (absolute
value <= 1e-9), report the relative difference as undefined and retain absolute
values/differences. A large percent is not a measured classifier accuracy loss.

## Camera motion

Raw GT must not be compared with compensated A directly. For `opencv_feature_cmc`,
apply `original_to_processed` and the same full per-frame current-to-reference
matrix to GT. The saved actual A handoff is checked against trajectory CSV,
including timestamps, dimensions, confidence, and its existing clamping policy.

Supply `b_comparison.camera_transforms` or request `--replay-cmc`. Replay decodes
the original video with the tracker resize/gray operations and reruns only its
background camera estimator. Saved debug centers/sizes, including LOST frames,
provide the exclusion masks. GT is never used to estimate camera transforms.
The source's size, frame count, and FPS must match metadata. Cropping/letterbox
replay is rejected; explicitly saved transforms can support those preprocessing
cases. Every replayed debug point must reproduce its saved compensated coordinate
within 0.01 processed pixels, otherwise the B comparison is marked not comparable.

The matrix artifact records source/debug/metadata hashes, tracker hash, OpenCV
version, reference frame, matrices, and residuals. Cached replay is tied to these
inputs and verified against trajectory points again. This verifies numerical
reproduction, not that camera estimation itself is ground truth. A point-wise
camera offset must not substitute for a full transform. Out-of-bounds GT centers
are rejected instead of silently clamped. A clamping count is shown separately.

## Outputs and validation

- `report.html`: position, normalized position, temporal tables and B comparisons.
- `summary.csv`, `summary.json`, `per_video/*.json`: summary and frame/pair records.
- `b_features.csv`: nine features for common GT, full GT, and GT sampling comparisons.
- `b_inputs/<sample_id>/`: exact diagnostic inputs, camera replay, feature results.

Run focused tests:

```bash
python -m pytest research/tests/test_tracking_evaluation.py \
  research/tests/test_tracking_feature_comparison.py tests/test_feature_core.py -q
```

Tests cover vector-vs-distance errors, constant bias, preserved gaps, stationary
paths, actual B runtime equivalence, full CMC transforms, missing/bad transforms,
and GT schema bounds. Small pilot results are descriptive evidence; source videos
are the independent groups, not individual frames or multiple objects in one clip.

## Time-signal diagnostics

```bash
python -m research.tracking_signal_comparison /path/to/results \
  --output /path/to/motion_signals
```

This uses the exported `a` and `gt_matched` inputs, including exact frame gaps,
timestamps, dimensions and verified CMC. `extract_features(return_signals=True)`
exports its own intermediate arrays; the default output and all B formulas remain
unchanged. Recomputed features must match both the actual B runtime and the saved
comparison before an output can be produced. Long gaps separate signal segments;
short gaps are interpolated only inside the existing B computation.

All diagnostic coordinates use B's canonical width 1920, not the spatial report's
1280 reference width. Position times are grid instants; velocity times are interval
midpoints; acceleration times are midpoints between velocity samples. These are
source-video absolute seconds, not seconds since ROI selection. Fractional source
frame positions on the resampled grid are estimates, not new observed frames.

B acceleration is `abs(diff(speed))/dt`, where speed is the norm of the fitted
velocity. It is not the norm of vector acceleration. Signal rows preserve all
samples used for feature aggregation, including filter boundaries. Edge diagnostics
use half the actual SG window on each segment end. The interior-only P95 is an
additional sensitivity check on those same samples, not a replacement feature,
production fix or evidence of an achieved improvement.

Error energy is `sum((A_acc-GT_acc)^2*dt)`. Report its concentration in the largest
10% of absolute-error samples and the share at SG boundaries, alongside boundary
sample fraction. These are descriptive localization metrics, not causal tests.
Peak times can be compared with nearby tracking-source events but the SG support
window spreads effects in time; proximity is not proof of recovery-induced error.
GT annotation noise, interpolation and camera-estimation error remain possible
contributors. Initial ROI offset must not be confused with persistent jitter.

The command exports JSON and CSV for observations, B positions, speed,
acceleration, tracking-source events and the five largest errors for each sample.
Optional `--app-project` writes reviewed query data for the companion local report.
