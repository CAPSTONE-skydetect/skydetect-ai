# MiniRocket real-data experiments, 2026-10-18

The received package exactly matches the deployed model's manifest SHA-256.
All manifest file hashes pass; real train has 156 windows/33 tracks/29 groups,
validation has 59 windows/11 tracks/10 groups. No train/validation group overlap
or exact tensor duplicates were found. Known video hashes do not overlap the
pilot, but 11 inventory rows lack video hashes (6 train, 3 validation, 2 historical
test); re-encoding and session overlap remain unresolved. Test arrays are absent.

## Results

| Candidate | Validation group macro-F1 | GT correct /6 | D0 correct /6 | N10 correct /6 |
| --- | ---: | ---: | ---: | ---: |
| Reproduced baseline | .8990 | 4 | 4 | 4 |
| Train rotation augmentation | .8000 | 3 | 4 | 3 |
| Rotation feature pooling | .8000 | 4 | 4 | 4 |
| Four-second context + short-track fallback | .7917 | 4 | 4 | 4 |
| Image-width motion scale | .8990 | 4 | 4 | 4 |
| Fixed equal dual-scale margins | .8990 | 5 | 4 | 4 |
| Source-group OOF learned dual-scale stack | .8990 | 5 | 5 | 4 |

The development candidate is D0 A + the learned dual-scale stack. It corrects
drone_1_drone; drone_5 remains wrong. Both pilot birds remain correct. bird_2
has a meta margin about +0.0002 and is fragile. No claim of a robust or independent
performance improvement is justified yet. The candidate was explored on these
same development cases, while the old validation was also previously inspected.

Validation group accuracy remains 9/10; drone recall 4/4, bird recall 5/6.
Validation track macro-F1 remains .8167. Neither the test set nor validation/pilot
labels are used in fitting. Candidate selection is still development selection.
The reference model's pilot labels and validation metrics reproduce; window
margins differ by up to 3.29e-5, so reproduction is not bitwise exact. The fitted
MiniRocket parameter arrays match; scaler/Ridge values have small differences.

## What changed

The original preprocessing centers and rescales every two-second window by its
own spatial extent. Constant-velocity paths at different apparent speeds can
therefore collapse to the same input. A new member retains centered image-width
motion extent using the saved normalization_scale. It is not physical world
speed and remains sensitive to distance, zoom, perspective and tracking noise.

The new member fixes drone_1_drone but loses bird_2. A logistic meta head combines
the original and new margins. It is trained on 33 tracks' out-of-fold margins,
with original-video groups kept together and class/group-balanced weights.
Both MiniRocket/scaler pairs are refit inside each fold. Member Ridge alphas are
frozen from the reproduced members; meta C=1 is fixed. OOF predictions train the
meta head and are not advertised as an unbiased performance estimate.
Mean window meta margins equal the meta prediction on mean member margins,
because the meta transform is affine. No pilot threshold/weight search is used.

The exact C4 invariant candidate really eliminates quarter-turn prediction
flips, but does not improve classification. Direction sensitivity alone was
not a sufficient explanation or solution. Four-second inputs are reconstructed
from overlapping real windows by inverting saved normalization. Maximum overlap
disagreement is 1.28e-8 in image-width units; missing support is never filled.
That context candidate also regresses validation and is not selected.

## Use the saved candidate locally

The standard production loader rejects the candidate's new contract on purpose.
It needs the explicit adapter; do not replace the existing model file in place.
This CLI accepts a regular TrackSequence and applies both required preprocessing
paths automatically. Its score is a logistic meta margin, not a probability.

```sh
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python research/minirocket_predict.py --branch-root /Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime --model artifacts/trajectory_experiments/minirocket-stack-20261018/stack.joblib --input artifacts/trajectory_experiments/pilot-20261017/evaluation/D0/b_inputs/drone_1_drone/a_track_sequence.json
```

That saved D0 input returns drone with meta margin 1.71056. No model was deployed
or pushed. Keep A at D0. Next, freeze this candidate and compare it with the
baseline on genuinely unused filming sources, including short and low-contrast
targets. Review drone_5 as a C training/representation failure before resuming
A tuning; its GT is also classified as bird.

Artifacts: `artifacts/trajectory_experiments/minirocket-results-20261018/index.html`
and `summary.json`; raw models, folds and per-window scores remain in the
separate model experiment directories. Source scripts are `minirocket_retrain`,
`minirocket_context`, `minirocket_stack`, `minirocket_focus` and `minirocket_results`.
