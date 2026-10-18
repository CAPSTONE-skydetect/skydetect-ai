# EXP-CONTINUITY-01 Results

## Outcome

The frozen downstream comparison supports a **promising, narrow ablation result**:
removing classifier windows whose resampling support mixes D0 and YOLO-recovery
observations corrected the phantom66 decision for both frozen classifiers. It did
not reduce object-level decision coverage in this 12-object development cohort.
This is not an operating-model promotion: all 12 targets are drones, and phantom66
is development data.

The paired-data audit found **no unexposed paired GT/A source groups** for a clean
GT-only/A-only/GT+A training comparison. The received MiniRocket package contains
real A-derived training windows, but it does not establish those windows as manual
GT, include source videos/full TrackSequences, or include test arrays.

## Experiment 1: Boundary Segmentation

### Method

- Reused the fixed 12 ARD100 development objects, saved D0/Hybrid TrackSequences,
  accepted-recovery provenance, existing C preprocessing, deployed baseline, and
  frozen Stack candidate. No A tracking was rerun and no model was retrained.
- Kept the existing 30 Hz, 2-second window, 1-second stride, and 60-sample C
  preprocessing contract. Window starts, frame indices, timestamps, and stored
  TrackSequences were not changed.
- Defined source as D0 or an accepted YOLO gap-recovery observation. A window was
  excluded only when its 60-sample interpolation support used both sources. The
  existing anti-alias filter is track-segment-wide; if it were active on a track
  with a source transition, the runner conservatively rejects windows rather
  than assuming the filter has local support.
- Boundary selection used provenance only; labels and GT coordinates were used
  only for evaluation. Each candidate start and C/continuity rejection reason is
  retained in `window_audit.csv`.

### Window Availability

Counts below are unique window starts, not duplicated model-score rows.

| Input | Candidate starts | Accepted by existing C | Rejected by C | Boundary-excluded | Kept after segmentation |
|---|---:|---:|---:|---:|---:|
| D0 | 95 | 65 | 30 | 0 | 65 |
| Hybrid, full | 96 | 67 | 29 | 3 | 64 |

Segmentation removed **3 of 67 otherwise-accepted Hybrid windows (4.48%)**. Those
were phantom30 at 6 s and phantom66 at 6 s and 8 s. The remaining phantom66
7-second window's model score was identical before and after segmentation. The
17 provenance transitions and all candidate-window inclusion/exclusion reasons
are in `source_boundaries.csv` and `window_audit.csv`.

Four windows used the existing C anti-alias filter, all on phantom138. None of
those windows crossed a provenance boundary. The three excluded windows were
excluded because raw interpolation support crossed D0/YOLO origins, not because
of a newly tuned geometric threshold.

### Object-Level Decisions

All 12 objects are drone targets. `Correct / wrong / abstain` is reported over all
12; decided accuracy is over the 10 objects for which each model returned a
class. Margins are raw decision scores, **not probabilities**.

| Frozen classifier | Input | Correct | Wrong | Abstain | Coverage | Accuracy among decided |
|---|---|---:|---:|---:|---:|---:|
| Deployed baseline | D0 | 6 | 4 | 2 | 10/12 (83.3%) | 6/10 (60%) |
| Deployed baseline | Hybrid full | 5 | 5 | 2 | 10/12 (83.3%) | 5/10 (50%) |
| Deployed baseline | Hybrid segmented | 6 | 4 | 2 | 10/12 (83.3%) | 6/10 (60%) |
| Frozen Stack candidate | D0 | 9 | 1 | 2 | 10/12 (83.3%) | 9/10 (90%) |
| Frozen Stack candidate | Hybrid full | 8 | 2 | 2 | 10/12 (83.3%) | 8/10 (80%) |
| Frozen Stack candidate | Hybrid segmented | 9 | 1 | 2 | 10/12 (83.3%) | 9/10 (90%) |

Only **phantom66** changed class between full and segmented Hybrid, for both
models. Coverage stayed 10/12, so the correction was not an object-level abstain
tradeoff. The full track had three accepted windows; segmentation excluded the
two source-mixed windows and retained one clean window:

| Model | Full window margins at 6 s, 7 s, 8 s | Full decision | Segmented retained margin at 7 s | Segmented decision |
|---|---|---|---:|---|
| Deployed baseline | -1.1765, +0.8151, -1.1818 | bird (wrong) | +0.8151 | drone (correct) |
| Frozen Stack | -1.1254, +0.3865, -3.7838 | bird (wrong) | +0.3865 | drone (correct) |

This is evidence that the source-mixed windows affected the aggregate, rather
than just evidence that the object was withheld. It is **not** proof of stable
residual-window consistency: only one window remains, so its zero within-window
score spread is automatic and should not be interpreted as robustness. Phantom30
also lost one boundary window but its drone decision did not change.

The saved D0/full-Hybrid parent results were reproduced for all **48** sample ×
model × input combinations: class, margin (tolerance `1e-6`), and accepted-window
count all matched. The 48-row verification is in `parent_reproduction.csv`.
Same-start D0-versus-Hybrid full/segmented score comparisons are in
`common_window_comparison.csv`.

### Regression Checks

- The six saved pilot objects were passed as matched GT and D0 tracks to both
  frozen models (24 rows). For each model, the class decision on GT-matched and A
  was identical per object; this is a regression check, not an independent test.
- Five saved bird-screening tracks were checked against both models (10 rows).
  B01 (30 Hz) was classified as drone by both models, a known false positive for
  the user-confirmed bird. B02-B04 remained invalid at their actual 14 Hz source
  rate (`uncertain`, zero accepted windows); no timestamps were fabricated.
  B05 (50 Hz) abstained for insufficient contiguous support (zero accepted
  windows). These bird clips have no manual GT and their training exposure is
  unknown, so they are not a held-out bird benchmark.

## Experiment 2: Paired-Source Readiness

### Local GT/A Material

- Found **19 paired TrackSequence records across 17 source-video groups**: 17
  drone-labeled and 2 bird-labeled tracks. All 19 have exact frame/timestamp
  correspondence and compatible processed dimensions/stabilization convention.
- The two bird tracks and one drone track from `drone_1` share one source group;
  counting them as separate videos would leak the same recording across splits.
  Across the 17 groups, birds occur in only **one** group, which also contains a
  drone track.
- All 17 groups were previously exposed to development/evaluation. Thus the
  number of unseen, paired source groups available for training/evaluation is
  **0 bird / 0 drone**. Drone3's paired track is included but its GT tail still
  needs human review and it is not an unseen holdout.
- Four additional local video-inventory groups have no linked GT/A track and
  need source/class review. Exact video hashes found no overlap with the package
  inventory, but that does not rule out re-encoded copies or shared recording
  sessions.

### Received Training Package

| Split inventory | Tracks | Source groups | Bird groups | Drone groups | Source-video hashes missing | Arrays available |
|---|---:|---:|---:|---:|---:|---|
| Train | 33 | 29 | 16 | 13 | 6 tracks | Yes |
| Validation | 11 | 10 | 6 | 4 | 3 tracks | Yes; previously reused |
| Test inventory | 10 | 10 | 6 | 4 | 2 tracks | **No test NPZ** |

The package documents real A-derived normalized 2-second windows, not verified
manual GT. It contains no raw source-video paths or full TrackSequences, and the
test arrays are absent. Its manifest also says recording-session identity is not
verified, validation was reused, and the prior test inventory was inspected.
Therefore it can inform the existing A-window MiniRocket training baseline, but
it cannot by itself supply paired GT-only/A-only/GT+A source-disjoint examples or
an untouched evaluation set.

### Data Request for the Next Training Comparison

Ask the C/data owner for a source manifest and data bundle with, for each track:

1. A stable `source_group_id` for the original recording/session, original video
   SHA-256, class label, and disclosure of prior train/validation/development
   exposure. Include source-video paths or a secure way to retrieve them.
2. For the *same videos*, the manual GT source (CVAT XML plus converted
   TrackSequence) and the A-produced TrackSequence, with matching frame indices,
   timestamps, processed width/height, coordinate convention, and CMC metadata.
3. A source-group split frozen before training: train, validation, and untouched
   test groups must not share a video, re-encode, or recording session. Include
   both bird and drone groups in every partition; more than one group per class
   per partition is preferable for a credible estimate.
4. Keep the three arms paired on identical source groups: train C on GT-only,
   A-only, and GT+A within the training groups; choose settings on validation
   groups; report once on the untouched test groups. Do not move the current
   pilot, ARD100 development cohort, reused validation, or inspected test data
   into the new training split.

## Reproduction and Provenance

From this repository, using the MiniRocket runtime environment:

```bash
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python research/exp_continuity_01.py
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python -m unittest research.tests.test_exp_continuity_01 -v
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python -m py_compile research/exp_continuity_01.py research/tests/test_exp_continuity_01.py
```

The runner refuses to overwrite an existing result directory. It uses the frozen
checkpoints and saved tracks listed below; it does not change production defaults,
model weights, input TrackSequences, or schemas. The complete per-file SHA-256
inventory is in `artifacts/trajectory_experiments/exp-continuity-01/input_model_code_hashes.json`.

| Item | SHA-256 |
|---|---|
| Starting code baseline | `a240387b269f80460785bc305aea5eef6ad6d00c` |
| Protocol-only commit | `daedb85` |
| Deployed MiniRocket checkpoint | `9024731a649b5e0a84ead3fcc45c9dc0a692ce305cea6e5ec8b7d720867a1e5a` |
| Frozen Stack checkpoint | `5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab` |
| `real_reference_comparison_v1.zip` | `445b4d0cac9339a24775841f11efe0aa230719a945e544868ecd9d3adbc299e4` |
| EXP runner at result generation | `38c5794a53da2f33a8948c0f8262142b6ebae9a554dc70617656b537c3faa8d6` |
| Focused tests at result generation | `aae196040c938231aa29cef2d6d6c202f4b00fab9c51376e330c0850d62780bc` |
| Protocol | `9bd040f69e9665de7455e77ffd0add31d3cc56968b656b7a88b452167058255c` |
| Existing C window contract | `0fa02ce7629b8c8fc556c5581123b425582aae41eb9c7cc8fcd68bd4135a678a` |
| Stack adapter | `1177a146234c8c5a685fdfabb2c0d61d0da412b14a0e5e0bf01e8695aff0c8ba` |
| Stack frame-scale helper | `b088fcbeae2c69c1d83f2f52265f0ec305797b882e095f0ed29dee5fd00ac3dc` |
| Stack MiniRocket inference helper | `2b22ee89d673e5e2823b5b10a4e1028e1093ff8f919ae289960c58c610d2f51e` |
| Runtime classifier adapter | `c514bd1af504d15eaffc4cc7fd7be665aef71437056be59493eb809422d96084` |
| Runtime sequence scoring | `d2f76b360048e8321970868ccc71bf5063ab528c0ec7ef47de831ba3d1e9c8ce` |
| Runtime TrackSequence schema | `fcc5fb563ce8fcb0ab8eea0ca9c9c2aae7be6e5d2d45b541aafbbb2ffb3d11f8` |
| Parent v4 summary | `4a8226f0004a8ce609269d0af0ba9d503fbef2a85213f8abb54e1f2e246b6a7b` |
| Parent v4 classifications | `96eeafa592ab385da4de194b05477a38ebf702b6d57b37e3c49209433c1772cb` |

Focused validation: **8 tests passed**, `py_compile` passed, all 48 parent
reproduction checks passed, and `git diff --check` passed. No full A retracking,
paid API, dataset download, or C retraining was performed.

## Outputs

- `artifacts/trajectory_experiments/exp-continuity-01/summary.json`
- `artifacts/trajectory_experiments/exp-continuity-01/object_classification.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/window_audit.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/source_boundaries.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/common_window_comparison.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/prediction_changes.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/parent_reproduction.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/regression_checks.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/paired_source_manifest.csv`
- `artifacts/trajectory_experiments/exp-continuity-01/paired_source_summary.json`
- `artifacts/trajectory_experiments/exp-continuity-01/per_object_results.json`
- `artifacts/trajectory_experiments/exp-continuity-01/input_model_code_hashes.json`

## Limitations and Change Scope

- This is a post-hoc, frozen-model development ablation on 12 drone-only objects;
  the segmentation rule is not validated on an independent bird/drone cohort.
- Phantom66's improved decision relies on one remaining two-second window.
  Collect longer clean post-reacquisition support and evaluate unseen mixed-class
  recordings before considering a runtime gate.
- Exact-hash checks cannot resolve re-encodes/session duplicates in the training
  package; six train, three validation, and two test-inventory tracks have no
  source-video hash.
- Only EXP-CONTINUITY-01's runner, focused tests, result document, and small
  JSON/CSV evidence artifacts are included in its final commit. Pre-existing
  tracked modifications and untracked experiments at start were preserved and
  excluded. The protocol was committed separately before execution. No
  production files or model/video binaries were changed or committed.
