# ARD100 YOLO Causal Gap Re-acquisition

## Why This Follow-up Exists

The first YOLO pass compared detector-refined points only at frame indices
already present in D0's CSV. D0 represents missing observations as gaps, not
rows, so unchanged observation coverage in that pass was expected. It was a
pointwise detector-fusion test, not a real gap-recovery test.

This follow-up reuses the exact trained weights and confidence threshold from
that pass. It fills short D0 gaps only when a causal constant-velocity prediction
from the last two D0 observations places the search crop, and two adjacent YOLO
detections confirm the candidate within 24 processed pixels. The development GT
is never used to generate a candidate.

## Frozen Setup

- Fixed baseline: `ard100-frozen-20261018-v3`.
- Fixed development videos: `phantom09`, `phantom105`, `phantom111`,
  `phantom117`, `phantom138`, `phantom145`, `phantom150`, `phantom30`,
  `phantom43`, `phantom66`, `phantom75`, `phantom81`.
- The 12 development videos were excluded from fitting and threshold calibration.
- YOLO11n was trained from random initialization on 40 ARD100 train videos and
  confidence-calibrated on 11 separate train videos; 12 development videos were
  held out. The same weights and confidence threshold were reused for this pass.
- Local search crop: 192x192 pixels at the 1280-pixel processing width; YOLO
  input size: 384.
- Constant-velocity prediction is causal, limited to a 10-frame gap. The last
  two D0 observations are used; detector outputs do not feed back into this
  prediction.
- A gap candidate is added only when adjacent detections are both above the
  validation-derived confidence threshold and their centers are within 24 px.

## Result

| Metric | D0 | YOLO local + gap recovery | Change |
|---|---:|---:|---:|
| Macro location success | 27.33% | 31.67% | +4.35 pp |
| Macro observation coverage | 76.99% | 78.18% | +1.19 pp |
| Median of per-video median center error | 111.61 px | 113.52 px | +1.91 px, worse |
| Median of per-video normalized error | 9.45 bbox widths | 9.52 bbox widths | +0.07 width, worse |
| Median temporal residual change | 1.158 px | 1.264 px | +0.106 px, worse |
| Mean temporal-pair coverage | 81.20% | 82.26% | +1.06 pp |

The detector accepted 1,032 points in total. It recovered 43 previously missing
frames: 12 in `phantom105`, 10 in `phantom30`, 11 in `phantom66`, and 10 in
`phantom81`.

Most useful cases:

- `phantom66`: median error 39.18 px to 0.90 px; location success 13.3% to
  44.2%; observation coverage 71.8% to 75.4%; residual change 0.944 px to
  0.732 px. This is strong evidence that a learned local detector can correct
  at least one background-lock failure.
- `phantom105`: 12 gaps recovered; location success 6.0% to 10.0%, observation
  68.4% to 72.4%, median error 253.27 px to 238.03 px, and residual change
  1.356 px to 1.125 px.
- `phantom30`: 10 gaps recovered; location success 77.7% to 81.4%, observation
  78.1% to 81.4%, and median error 2.72 px to 1.04 px. Residual change became
  slightly worse, 0.964 px to 0.995 px.

Counterexamples remain:

- `phantom09` had no gap recovery; YOLO changed 43 observed points. Location
  success moved only 10.3% to 11.0%, while median error worsened 98.04 px to
  101.85 px and residual change worsened 7.79 px to 10.06 px.
- `phantom75` already had nearly perfect D0 tracking. YOLO accepted 301 points
  but did not improve location success; median error and jitter both increased
  slightly. This argues against replacing stable D0 points unconditionally.
- `phantom81` gained 10 observations, but its median error did not change and
  residual change increased 0.939 px to 1.403 px.

The composite development gate **failed**: location success and observation
coverage improved, but median center error and temporal jitter did not both
remain non-worse. Do not replace D0 or enable this model in production.

## Interpretation and Next Experiment

This is a useful partial breakthrough, not a finished tracker. The result
supports a hybrid A: retain D0 for continuity, use the local detector as a
second target hypothesis, and switch only when the detector is temporally
consistent and D0 is lost or independently judged unreliable. The `phantom66`
result shows potential; `phantom09` and `phantom75` show why per-frame
replacement is unsafe.

The next high-value experiment is not another detector architecture. It is a
split-safe fusion gate that preserves D0 coordinates when D0 and YOLO agree or
D0 is stable, but confirms a YOLO switch over multiple frames when evidence
disagrees. Select any thresholds only on source-video validation data, then
run the fixed development set once. Keep the 12-video results as development
evidence, not as a final independent test.

## Artifacts

- Report: `artifacts/trajectory_experiments/ard100-yolo-gap-reacquisition-20261018-v3/index.html`
- Machine-readable summary: `.../summary.json`
- Per-video comparison CSV and overlays: `.../samples/<sample_id>/`
- Reproduction script: `research/ard100_yolo_gap_reacquisition.py`
