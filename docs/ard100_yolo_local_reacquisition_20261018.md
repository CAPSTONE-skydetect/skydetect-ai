# ARD100 YOLO Local Re-acquisition Experiment

## Question

Can a small detector trained on ARD100 training videos verify or refine target
locations around the points D0 already observed?

## Protocol

- Keep frozen D0 and its runtime settings unchanged.
- Train YOLO11n from random initialization; no internet-fetched or external
  pretrained weights are used.
- Exclude the fixed 12-video development subset from both model fitting and
  confidence calibration.
- Split the remaining source videos by video into fit and calibration groups.
- Train on 192x192 processed-resolution crops. Positive crops jitter the GT
  center; hard negatives place the labeled object fully outside the crop.
- Upscale to 384 during YOLO training/inference so small targets occupy more
  detector pixels.
- Calibrate the confidence gate on the separate source-video validation group
  using F0.5 at IoU 0.5, favoring precision to reduce false reacquisition.
- The first pass evaluates only frame indices present in D0's trajectory CSV.
  Since that CSV represents missing frames as gaps (no row), this pass does not
  test actual gap recovery. GT after initialization is used only by the offline
  evaluator.
- Evaluate the same fixed 12 development videos and compare localization,
  observation coverage, bbox-normalized error, temporal jitter, and paired-frame
  coverage against D0.

## Scope and Caveats

This is a pointwise local detector experiment, not a global detector and not a
production tracker replacement. Its search is limited to roughly 96 processed
pixels from D0's observed point. The model starts from random weights and is
trained on a limited set of labeled source videos, so failure would not prove
that detector-assisted tracking is ineffective in general; it would show that
this compact, local, from-scratch candidate is not yet adequate.

The actual short-gap recovery experiment reuses this frozen model and is
documented in `ard100_yolo_gap_reacquisition_20261018.md`.

The fixed development subset is used only once as a gate. Do not tune the
confidence threshold or model settings against it after seeing results. Any
candidate that fails the gate remains experimental and must not replace D0.

## Reproduction

From the experiment repository root, use the configured project Python:

```bash
YOLO_CONFIG_DIR=/tmp/skydetect-ultralytics-config \
/Users/yuchan/Desktop/git/skydetect-ai/.venv/bin/python \
-m research.ard100_yolo_local_reacquisition
```

The default output is ignored under
`artifacts/trajectory_experiments/ard100-yolo-local-reacquisition-20261018-v2/`.
