# ARD100 frozen experiment

The user confirmed this source does not overlap the classifier training data.
This is recorded as user-provided provenance, not inferred from filenames.

## Scope fixed before the full run

- Inventory all 100 videos, retaining ARD100's 35 test / 65 train split.
- Evaluate one fixed clip per video: the first valid annotated target through
  10 seconds after initialization, or EOF if earlier. This is not full-video
  long-term tracking evaluation.
- Read the supplied VOC XML. The supplied extraction script starts image names
  at 0001, so XML suffix minus one is the zero-based video frame.
- Preserve the actual XML coordinates. Do not invent subpixel precision or
  interpolate missing/empty annotations. Ambiguous multiple-object annotations
  quarantine the sample rather than choosing an object after seeing results.
- Initialize D0 using the first valid full-object bbox in original pixels.
  No later GT points or boxes are supplied to A. Resize width is 1280, CMC is
  enabled, online template updates are off, and the existing D0 confidence
  thresholds remain fixed. Full-object initialization differs from old pilot
  32px ROIs; cross-dataset score changes cannot isolate that factor.
- Use the existing baseline and learned stacked candidate with their existing
  decision thresholds. No training, parameter sweep or model selection uses
  ARD100 results in this experiment.
- Hash the models, videos, converted XML, protocol and execution source files.

The preparatory v1/v2 folders are retained. The final protocol is v3. Earlier
preflights were too strict about empty annotations and inspected ambiguity
beyond the requested clip. These parser checks were corrected before expanding
the fixed experiment. Empty annotations remain unobserved, not fabricated GT.

## Validity checks and outputs

Initial, intermediate and later frame crops from phantom02, phantom04,
phantom141 and phantom09 were rendered for alignment verification. Visual
inspection confirms that the supplied boxes cover the target in these samples;
this is not exhaustive validation of every label.

The existing scorer measures location error, bbox-width-normalized error,
frame-to-frame error changes, coverage, missing runs and actual B features.
CMC is replayed using saved A centers, with no GT used to estimate camera motion;
its transforms must reproduce A's saved coordinates within 0.01px.

CMC-normalized GT can leave [0,1]. Reject such GT inputs without clamping or
post-hoc trimming. Attempt full-window and A-matched GT exports independently;
keep the failed support's reason and frame numbers. A's actual output remains
eligible for classification even if GT cannot be encoded in the current schema.

Classifier inputs are explicit:

- `D0`: the complete actual A handoff, including its genuine frame gaps.
- `D0_matched`: A points at frames with evaluable GT.
- `GT_matched`: GT at exactly the D0_matched frames and timestamps.
- `GT_full`: all GT within A's actually attempted window, not the full source
  video or necessarily the entire requested clip if A exits early.

All requested videos remain in end-to-end denominators. Separate drone correct,
bird wrong, classifier uncertain and unavailable inputs. Report GT/A comparisons
only where both exist, showing unavailable cases separately. Show the requested
GT coverage so early termination cannot inflate apparent location performance.

This is a drone-only experiment: no bird false-positive rate, balanced accuracy
or complete drone/bird classifier superiority can be established. ARD100 video
files may share filming sessions, even though they are distinct files. Do not
treat overlapping windows or files as independent filming-session observations.

## Commands

Repository root: `/Users/yuchan/Documents/형주플젝/skydetect-ai-trajectory-experiments`.

```sh
/Users/yuchan/Desktop/git/skydetect-ai/.venv/bin/python -m research.ard100_benchmark run --protocol artifacts/trajectory_experiments/ard100-frozen-20261018-v3/protocol.json --output artifacts/trajectory_experiments/ard100-frozen-20261018-v3 --jobs 4
/Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime/.venv/bin/python research/ard100_benchmark.py classify --protocol artifacts/trajectory_experiments/ard100-frozen-20261018-v3/protocol.json --output artifacts/trajectory_experiments/ard100-frozen-20261018-v3 --branch-root /Users/yuchan/Documents/형주플젝/skydetect-ai-minirocket-runtime --candidate artifacts/trajectory_experiments/minirocket-stack-20261018/stack.joblib
/Users/yuchan/Desktop/git/skydetect-ai/.venv/bin/python research/ard100_report.py artifacts/trajectory_experiments/ard100-frozen-20261018-v3
```

The A worker environment and C runtime are separate to preserve each branch's
dependencies. A workers run with one OpenCV/BLAS thread and a per-video timeout.
Existing terminal result files are reused by the batch command; source changes
are refused rather than silently mixing implementations.

Entry points: `research/ard100_benchmark.py`, `research/ard100_report.py`.
Results stay under the experiment artifacts folder. Original ARD100 files and
the production service/model are unchanged.

## Completed results

97 single-object clips were tracked; phantom95 (test), phantom139 and phantom20
(train) were quarantined for multiple-object annotations. No tracking worker
crashed or timed out. There are 800 classifier outcome rows across 100 videos,
four input scopes and two frozen models, including unavailable outcomes.

| Actual A output | Existing MiniRocket | New candidate |
| --- | ---: | ---: |
| ARD100 test, requested 35 | 21/35 (60.0%) | 26/35 (74.3%) |
| ARD100 train, requested 65 | 37/65 (56.9%) | 46/65 (70.8%) |
| All requested 100 | 58/100 | 72/100 |

No ARD100 video was used to fit either model. "Train" here is ARD100's directory
name, not an indication that this experiment trained on those 65 videos.
The candidate's test outcomes are 26 drone, 6 bird, 2 uncertain and 1 quarantined.
It fixes phantom05, phantom113, phantom133, phantom54 and phantom58, with no test
baseline-correct video becoming wrong. These rates measure drone recall, not
balanced classification accuracy or false-positive performance on birds.

Within the test set, matched GT correct increases from 15 to 18 of all 35
requested videos. Both models have 1 uncertain and 6 unavailable GT inputs.
On the 28 test cases where both matched GT/A have binary predictions, the new
candidate yields 17 both correct, 1 GT-only correct, 5 A-only correct and 5 both
wrong. A-only correctness is not evidence that A is more accurate than GT.

This larger dataset exposes a substantial A tracking problem: across the 34
evaluated test videos, the median of video-level median center errors is
51.59px at width 1280; the median video-level median error / GT width is 4.457.
Median requested-GT observation coverage is 95.65%, so high observation coverage
does not imply correct target tracking. phantom133, which the candidate labels
drone correctly, visibly drifts toward a building while GT moves away.
Do not interpret the extra C correct labels as validated faithful trajectories.

CMC bounds block full GT export in 11 tracked clips and matched GT in 10.
Partial exports preserve all requested support or reject that support; no GT
coordinates are clipped. Actual A coordinates were clipped in 10/50 visible
frames of phantom04, as verified against its saved compensated coordinates.
That is an existing A handoff issue retained in this frozen experiment.

All 87 available matched GT/A exports were checked for identical frame indices,
timestamps, processed dimensions and stabilization metadata. The final HTML's
local links were checked, and four comparison videos were converted to H.264
while preserving the original generated MPEG-4 previews. Parser/comparison/C
focused tests passed; the browser's file-URL policy prevented automated page
opening, so rendered browser layout was not verified. Overlay frame images were
visually inspected separately.

The report generator received presentation/interpretation updates after A
completed. `execution_sources.json` retains the exact pre-run hashes; do not
silently rewrite that lock. A rerun in this directory can intentionally refuse
the changed reporting source. Use a new experiment output for a new run.

## Decision from this run

Keep the new C as a candidate: its drone recall improvement transfers beyond the
six pilot objects, including GT inputs, but bird false positives still need
unused-source evaluation. Do not deploy based on drone-only results. Use ARD100's
existing annotations to evaluate target identity / drift, not just whether A
emitted coordinates or C guessed the video class. The next A change should
address background lock-on / target loss, with the frozen benchmark retaining
the current failures; stronger smoothing alone cannot repair a wrong object.

Results page: `artifacts/trajectory_experiments/ard100-frozen-20261018-v3/index.html`.
