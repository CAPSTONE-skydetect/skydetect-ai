# MiniRocket focus: completed work and next executable step

Update: the package was received and real-data experiments completed later on
2026-10-18. See `minirocket_training_results_20261018.md`; the missing-package
status below records the earlier state.

Date: 2026-10-18. A is frozen at D0; N10 is a comparison input.

## Completed paired production checks

All three inputs use the same observed frame IDs and timestamps. Labels below
come from the XML, including bird_2 being a drone. Scores are Ridge margins,
not probabilities. These six objects are development material.

| Object | Truth | GT margin | D0 margin | N10 margin | GT drone-vote windows |
| --- | --- | ---: | ---: | ---: | ---: |
| bird_2 | drone | +0.319 | +0.133 | +0.116 | 6/9 |
| drone_1 bird 1 | bird | -0.988 | -0.858 | -0.897 | 0/2 |
| drone_1 bird 2 | bird | -0.901 | -0.769 | -0.724 | 0/1 |
| drone_1 drone | drone | -0.357 | -0.252 | -0.327 | 1/6 |
| drone_5 | drone | -0.378 | -0.256 | -0.428 | 0/8 |
| drone_6 | drone | +0.252 | +0.836 | +0.101 | 8/9 |

Mean and median aggregation both classify 4/6 correctly on all three inputs.
Any-positive aggregation gives 6/6 on D0 but only 5/6 on GT and N10; it is an
exploratory development result, not a justified production change. One positive
window can trigger a false alarm, and independent validation is unavailable.
The two errors are therefore not explained solely by mean aggregation.

Production/research scores match for all 18 inputs. Verified class order,
margin sign, four-channel order, float32 shape, displacement identity, paired
clocks, and model/preprocessing fingerprint `eb9be8154ee0f404`.
The model records real-only training with 156 windows, 29 groups and alpha 0.1.
Original training-array regeneration cannot be verified without its package.

## Concrete new finding: orientation sensitivity

Rotate both coordinate and displacement channels together, preserving shape,
timing, speed magnitudes, and within-window distances. At 90-degree increments,
all six GT objects have at least one prediction flip.

| GT object | Original margin / prediction | 90-degree margin / prediction |
| --- | --- | --- |
| drone_1 bird 1 | -0.988 / bird | +0.663 / drone |
| drone_1 bird 2 | -0.901 / bird | +0.559 / drone |
| drone_1 drone | -0.357 / bird | +0.305 / drone |
| drone_5 | -0.378 / bird | +0.293 / drone |

This demonstrates coordinate-orientation sensitivity, not proof that all errors
have a single training cause. Rotating inputs at inference is not adopted:
it also makes correct birds wrong. The next candidate is train-only rotation
augmentation followed by unchanged original validation and a new holdout.

## Implemented training comparison

`research/minirocket_retrain.py` uses the collaborator's original package
validator, preprocessing, MiniRocket/scaler/Ridge and group-level metrics.
It compares original training against rotations 0/90/180/270 degrees.

- Every rotated child keeps its parent's original source group.
- Four children's weights sum to the original parent's weight.
- Split groups before augmentation. Each fold fits its own transformer/scaler.
- Select alpha on original held-fold windows, never augmented held windows.
- Final validation remains unmodified. No test fitting or test evaluation.
- Save both models and original-window validation predictions; do not deploy automatically.

Nine related tests passed. A separate fixture-only run exercised both complete
training/CV/validation paths with real MiniRocket; it does not measure real-data
classification performance.

Actual retraining is blocked by the missing original package. Preflight records
`fitting_performed=false`; no replacement model has been trained on the six pilot
objects. Existing inference models remain intact.

## Exact package request to C

Please share the complete `real_reference_comparison_v1.zip` used to train the
deployed MiniRocket, including its dataset manifest, four NPZ files,
real_metadata.csv, source_inventory.csv and all other manifest-hashed files.
The deployed model records dataset_manifest SHA-256
`d5dbdd50a28f4113ca751cf12758f92788bc822f62eef5847f2c73ee54f50f8c`.
The source inventory is required to check overlap with our pilot video hashes.
Raw videos are not required for this MiniRocket retraining comparison.

## Run when the package arrives

Run this file directly with the MiniRocket environment, not with `-m`, to keep
the collaborator's runtime imports isolated:

```sh
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python research/minirocket_retrain.py --branch-root /Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime --package /absolute/path/to/real_reference_comparison_v1 --output artifacts/trajectory_experiments/minirocket-retrain-new-run
```

Use a new output directory. After the original validation comparison, evaluate
GT/D0/N10 as development probes and reserve new source groups for final testing.
Do not use pilot correctness to choose the rotation, threshold or aggregation.

Audit artifacts: `artifacts/trajectory_experiments/minirocket-focus-20261018/`.
`tracks.csv`, `windows.csv`, `rotation_sensitivity.csv`, `inference_windows.npz`
and `summary.json` preserve the paired inputs and frozen-model evidence.
