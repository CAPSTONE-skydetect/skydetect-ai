# SkyDetect C-part dataset

The six data packages under `dataset/` contain `minirocket.npz` with arrays
`X` (float32, samples × 4 × 60), `y`, `group_id`, `sample_id`, and `fold`.
The channel order is `qx, qy, dx, dy`. Each package also contains RF features,
augmentation recipes, and an index matching NPZ window IDs to parent tracks.

| Experiment | Train | Validation | Test |
| --- | --- | --- | --- |
| A | `train_real` | `validation_real` | `test_real` |
| B / Version 2 | `train_augmented` | `validation_real` | `test_real` |
| D / Version 1 | `train_augmented` | `validation_augmented` | `test_augmented` |

For grouped cross-validation, the rows in `train_real` with `fold == k`
are the real holdout. A uses the other real rows for fitting. B and D use
`fold_k_train_augmented` for fitting. Fit any transformer and scaler on the
fit rows. Use real validation for model selection and the augmented evaluation
packages for stress testing.

`original_assets.csv` joins the unchanged overlays and TrackSequence JSON
files under `originals/`. It records each available file's SHA-256, size,
and pairing status. There are 519 matched pairs, 11 track-only entries,
and 18 overlay-only entries. Paths in the index, source table, and manifest
are relative to this directory.

Only MP4 files use Git LFS. With LFS smudging skipped, the NPZs and original
JSON files are immediately available. To download the overlays later:

```powershell
git lfs pull --include="data/c_part/final_observation_v1/originals/**/*.mp4" --exclude=""
```

To rebuild from the repository root using the development Python environment:

```powershell
python -m research.build_observation_dataset `
  --input data/c_part/final_observation_v1/originals `
  --label-map data/c_part/final_observation_v1/label_map.csv `
  --audit data/c_part/final_observation_v1/calibration `
  --output research/output/rebuilt_final_observation_v1
```

The calibration inputs are image-derived center proxies from five reviewed
clips. MiniRocket excludes one bird track that has no valid continuous
two-second window. RF retains its features. The dataset manifest records
package-specific counts and preprocessing.
