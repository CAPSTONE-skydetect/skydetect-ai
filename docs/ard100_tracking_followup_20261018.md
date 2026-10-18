# A background-lock experiments and bird false-positive check

## Decision

No A candidate is adopted. D0 remains the default. The candidate MiniRocket
weights and decision threshold are unchanged. A practical C adapter bug was
fixed: unsupported inputs now return uncertain/invalid_input rather than raising.

## Fixed selection policy

The development subset is 12 ready train videos at a lexicographic stride,
selected without consulting their GT error or C outputs. Each clip, ROI, CMC
setting, frame clock and GT is identical to the existing frozen D0 experiment.
No GT is supplied after initialization. Train contains all 65 requested videos,
including two preflight-quarantined multiple-object cases. Failed or missing
results remain in localization denominators rather than being dropped.

The gate requires increased correct localization, at most a two-percentage-point
drop in raw observation ratio, and non-increased median center error. Correct
localization means error <= half the GT bbox diagonal, divided by all requested
GT frames, then macro-averaged over videos. Center error summaries are medians
of video medians at 1280px width, excluding unmeasurable cases only for that
position statistic. The loss-aware localization score includes failures as zero.

Only a candidate passing the full train gate proceeds to another test run. None
did, so the old ARD test output was not rerun or used for selecting parameters.
Its old failure cases had already informed diagnosis, so any future rerun would
be follow-up validation, not a completely untouched blind test.

## Experiments executed

| Candidate | Change | Dev 12 localization | Full train localization | Full train center median | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| D0 | Existing baseline | 27.33% | 37.53% | 12.76px | Keep |
| AR1 | Target-size KLT windows and pyramid depth | 30.60% | 37.14% | 14.51px | Reject |
| AR2 | AR1 plus contrast recentering OFF | 34.19% | 34.82% | 39.55px | Reject |
| AR3 | Audit accepted KLT with residual motion, anchor appearance and two-frame confirmation | 27.33% | Not run | Not run | No overrides; reject |
| AR4 | AR3 with motion score floor aligned to existing recovery floor 0.38 | 27.24% | Not run | Not run | 33 overrides, no localization gain; reject |

AR1/AR2 were each run on 63 single-target train clips; the two quarantined cases
remain in the requested denominator of 65. AR3/AR4 were stopped after dev12.
Their GT-based full train gates were not fabricated or assumed to pass.

The size-aware KLT option avoids reducing very small targets to approximately
one pixel at coarse pyramid levels, but changing that support alone does not
solve background lock-on. AR2 shows the danger of selecting only on C labels:
new MiniRocket drone correctness rises from 46/65 to 53/65 while localization
gets worse. That increase is not evidence of accurate trajectories.

AR3/AR4 calculate residual evidence even when KLT claims success. They override
only background-like flow, require sufficient anchor appearance and repeated
spatially consistent candidates, and retain the existing recovery radius. The
first setting never activated on dev12. The second activated 33 times but did
not improve localization, so neither is enabled by default.

## Bird check

The user confirms the five local files are birds, but training use is unknown.
They are not claimed as an independent bird test. D0 and AR1 were tested with
both C models. Each configuration has one drone false alarm and four abstentions:
three sources are 14Hz, below the 15Hz input contract, and the fourth lacks
sufficient contiguous two-second tracking support. No timestamps were altered
to make low-rate inputs look supported. AR2/AR3/AR4 bird runs were unnecessary
after their A gates failed and were not performed.

The one decided bird has candidate margin +1.0489743. Raising the threshold above
it would also lose 16 of the 26 D0 test true-drone positives. No threshold was
selected on this example. There is insufficient evidence to declare the new C
safe on birds or recommend deployment.

The stacked C adapter now catches the same invalid-window input exceptions as
the baseline, preserves rejection reasons, and returns explicit abstention.
The existing valid drone score remains exactly 1.7105640242029445. Model weights
remain SHA-256 5ba8fc4ac843f30c8a2a590add3f5854f6cf23093d1b8ed1f3b0731b95ec8aab.
See `docs/bird_false_positive_check_20261018.md` for file-level evidence.

## What is established

Small KLT support changes and contrast toggles are insufficient to recommend a
replacement for D0 on these clips. Adding a residual-motion override does not
yet provide reliable target identity either. Better C label counts can coexist
with worse target localization; the two must not be conflated. The next tracking
engine experiment needs genuinely discriminative target localization / identity
evidence and should be measured by the same loss-aware GT benchmark, rather than
further confidence-only tuning or smoothing background trajectories.

The bird check exposes an actual false alarm and input-rate/support limitations.
A source-reviewed >=15Hz bird set with usable tracking remains necessary before
independent false-alarm claims. No user video, GT, existing baseline result,
production model or default tracking setting was replaced.

## Artifacts

- `artifacts/trajectory_experiments/ard100-followup-20261018/index.html`
- `artifacts/trajectory_experiments/ard100-followup-20261018/summary.json`
- `ard100-AR1-dev12-20261018`, `ard100-AR1-train-20261018`
- `ard100-AR2-dev12-20261018`, `ard100-AR2-train-20261018`
- `ard100-AR3-dev12-20261018`, `ard100-AR4-dev12-20261018`
- `bird-screening-20261018`: inventory, image spot checks, original overlays,
  trajectories, model window scores and explicit abstention reasons.

Each run folder preserves its own protocol, source hashes, per-sample results
and tracking comparison. Do not replace old source locks after later code edits;
use a new experiment output for replay under changed code. In the legacy C
comparison CSV, input="D0" denotes the actual A-output scope; its specific AR1 or
AR2 tuning is declared by that run's protocol and folder, not a claim of D0 tuning.

## Additional SOT candidate probe

OpenCV MIL with CMC was compared on the same 12 development clips. It improved
the typical center-error median and observation coverage, but reduced the
loss-aware localization score and increased normalized error and temporal
jitter. One 5x3px target timed out, while several background-lock cases improved
substantially. It is not a D0 replacement. See
`docs/ard100_mil_sot_probe_20261018.md` and
`artifacts/trajectory_experiments/ard100-mil-cmc-probe-20261018-seeded/index.html`.
