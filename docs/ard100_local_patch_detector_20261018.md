# ARD100 local patch verifier experiment

## Question

Can a detector trained on other ARD100 training videos recognize the target near
the frozen D0 position and correct background lock-on without reducing useful
observations?

## Fixed method

- The same 12 development clips selected for AR1–AR4 are held out by video.
- Training uses up to 24 annotated boxes from each other ready ARD100 train
  clip. No frames from a development clip enter model fitting.
- Each positive is a grayscale HOG patch centered on the annotated full-object
  box. Four hard negatives shift that same patch window so the target is
  off-center. A linear SVM is fit once with fixed settings.
- At inference, the verifier sees the current frame, D0 center, and the initial
  box size. It searches a fixed neighborhood and a small set of scales.
- If the best-scale score at D0 is nonnegative, retain D0. Otherwise, accept a
  different local position only when its SVM decision score is nonnegative. If
  neither condition holds, preserve the frame as missing.
- The score is a classifier margin, not a calibrated probability. The zero
  boundary comes from the training fit and is not tuned on the development set.
- D0-missing frames remain missing in this first probe. The experiment tests
  correction of visible but potentially wrong positions, not gap recovery.
- Ground-truth boxes after initialization are only used for offline scoring.

## Decision gate

On the 12 development videos, the candidate must improve macro localization
success (error no greater than half the GT bbox diagonal), keep observation
coverage within 2 percentage points of D0, and avoid increasing the median of
video-level median center error or residual-step jitter. Consecutive-pair
coverage is also reported so abstaining on difficult frames cannot make jitter
look better without an accompanying observation penalty.

A pass would justify a separate confirmation run on the remaining ARD100 train
videos and a MiniRocket comparison. It would not establish production
readiness. A failure means this patch representation and fixed search rule did
not provide useful evidence; it does not rule out learned detectors or
track-conditioned re-identification.

## Result

The experiment completed on all 12 fixed development videos. The verifier was
trained on 51 other ARD100 train videos, using 1,224 positive patches and 4,896
hard negatives. No development video entered training.

| Metric | D0 | Local verifier |
| --- | ---: | ---: |
| Macro localization success | 27.33% | 26.90% |
| Macro observation coverage | 76.99% | 76.99% |
| Median of per-video median center error | 111.61px | 119.22px |
| Median residual-step error | 1.158px | 1.567px |
| Consecutive-pair coverage | 81.20% | 81.20% |

The candidate failed three of four adoption checks: location success did not
increase, center error rose, and jitter rose. It must not replace D0. The
classifier changed positions often but did not reject any D0 observation, so
the fixed zero-margin rule did not provide a useful abstention safeguard.

Some individual difficult clips improved slightly (for example, phantom111
localization rose from 1.7% to 4.0%), while phantom105's median error worsened
from 253.3px to 302.7px and phantom117's localization fell from 99.3% to 91.2%.
The phantom105 overlay shows the local candidate following a strong field
boundary while the GT target is elsewhere. This supports a specific diagnosis:
the cross-video grayscale HOG patch score is not sufficiently target-specific
for these tiny objects and backgrounds. Searching more windows makes the best
score positive even when the best patch is still background.

The result does not rule out a detector trained with a learned image backbone.
It does show that this generic patch verifier is not ready to adopt. A
follow-up should use target-conditioned evidence from the initial ROI and
train-only detector proposals, with a held-out calibration split inside ARD100
train before another development comparison. Do not tune the score threshold
on these same 12 clips.

Detailed results and overlays:
`artifacts/trajectory_experiments/ard100-local-patch-detector-20261018/index.html`.

## Reproduction

```sh
/Users/yuchan/Desktop/git/skydetect-ai/.venv/bin/python -m research.ard100_local_patch_detector
```

The script refuses to overwrite an existing output folder. It records the
model and source hashes, per-video trajectories, evaluation JSON, and comparison
videos for four preselected diagnostic clips. It does not change D0 or the
production service.
