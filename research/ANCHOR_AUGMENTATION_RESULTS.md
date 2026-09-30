# Real-anchor + simulator-residual augmentation

## Decision

The B pipeline can now generate **development-only** large sequence training sets
under the fixed `trajectory-sequence-1.0.1` contract. This does not establish
sim-to-real validity or independent real-world accuracy. No new untouched bird
or drone video was available for the current study, and shooting-session
independence remains unverified.

**Subsequent video review did not approve this dataset for C training or a
realism claim.** See [ORIGINAL_VIDEO_AUDIT_RESULTS.md](ORIGINAL_VIDEO_AUDIT_RESULTS.md).

## Method

1. Read only real A **train** windows (`156` windows from `29` provisional
   source-video groups) and the predeclared A-independent 3D simulator donor
   pool. Never derive from validation or test.
2. Remove the donor window's endpoint trend. Scale the remaining simulated
   motion residual to a small amplitude and add it to a same-class real A
   anchor. This preserves the real trajectory's large-scale motion and A
   observation pattern, while adding bounded simulated local variation.
3. Reject windows that leave the processed frame. Recompute all four channels
   `(q_x, q_y, d_x, d_y)` for 60 samples at 30 Hz. Every derived row inherits
   its anchor's original video group and carries parent/donor provenance.
4. Balance output by class and source-video group. Remove *exact* repeated
   sequence arrays. Synthetic rows receive only 10% of classifier sample-weight
   mass; real train rows retain 90%.

The perturbation is a **development hypothesis**, not a measured bird/drone
flight prior. In particular, this experiment does not prove the donor residual
represents wingbeats or joystick control in the real videos.

## Predeclared selection and observed results

The evaluator tried amplitudes `0.04`/`0.10` and synthetic weight masses
`0.10`/`0.25`. MiniRocket was fitted on real fit groups only within 3-fold
source-video group CV. A candidate had to preserve or improve mean group
macro-F1 and lose no more than 0.05 in any fold before opening validation.
The historically inspected test was not opened by this experiment.

| Comparison | Real-only | Selected real + augmented |
|---|---:|---:|
| Train grouped CV mean macro-F1 | 0.8569 | 0.9293 |
| Minimum fold difference | - | 0.0000 |
| Real validation group macro-F1, 10 groups | 0.8990 | 0.8990 |

Amplitude `0.10` failed the per-fold gate (minimum delta `-0.1139`). The
selected setting is amplitude `0.04`, synthetic mass `0.10`. A full 6,000-row
training smoke run also retained validation macro-F1 `0.8990`; it did **not**
show a validation accuracy improvement. Earlier whole-simulator mixing had
reduced validation group macro-F1 from `0.899` to `0.697`, so the current
result supports only a narrower, real-anchored development approach.

## Generated artifact

`research/output/anchor_training_6000_v3/` contains:

- `augmented_train.npz`: `(6000, 4, 60)` float32 inputs and aligned `y`,
  `group_id`, `sample_id` arrays, with 3,000 bird and 3,000 drone rows.
- `provenance.csv`: original track/group and same-class donor ID for each row.
- `dataset_manifest.json`: contract, source/evaluation hashes, exact counts,
  limitations, and suggested 10% training weight mass.

All 6,000 arrays are exactly distinct. They descend from only 33 real train
tracks in 29 video groups, so they are **not 6,000 independent cases**.
Validation and test sets stay real and unchanged. `research/output/` is
Git-ignored and must be transferred separately if C needs the data.

Reproduce from the existing real-sequence handoff and isolated MiniRocket
environment:

```powershell
.\research\output\minirocket_env\Scripts\python.exe -m research.evaluate_anchor_augmentation --output research/output/anchor_augmentation_v3
.\venv\Scripts\python.exe -m research.build_anchor_training_dataset --evaluation research/output/anchor_augmentation_v3 --output research/output/anchor_training_6000_v3 --target 6000
```

The evaluation and builder refuse a nonempty/existing destination; select a
new versioned output directory when rerunning. Do not use this result as a
paper claim of generalization until a genuinely new, session-separated real
bird/drone holdout is acquired and evaluated once under a frozen pipeline.
