# ARD100 OpenCV MIL SOT probe

Date: 2026-10-18

## Question

Can an online appearance-based single-object tracker reduce D0's background
lock-on on the same ARD100 development clips?

## Protocol

- Reused the fixed 12 ready train clips selected before this experiment.
- Tracked the first 10 seconds at 1280px processing width.
- Initialized both systems from the first full-object GT box only. Later GT was
  used only for evaluation.
- Compared the current D0 outputs against OpenCV `TrackerMIL_create`.
- Enabled CMC for both. D0 uses CMC to transform its motion prediction and
  reported coordinates. The MIL probe warps each frame into the reference view,
  updates MIL there, then maps its center back to source-frame coordinates.
- Fixed OpenCV's RNG seed independently per clip from its sample ID.
- Ran each clip in a child process with a 45-second limit. A timeout counts as
  zero localization and zero observation; it is not removed from the denominator.
- Scored only actual emitted observations. No interpolated positions were added.

## Aggregate result

| Metric | D0 | MIL + CMC | Interpretation |
|---|---:|---:|---|
| Correct localization, error <= half GT bbox diagonal | 27.33% | 25.66% | MIL is lower overall |
| Observation coverage | 76.99% | 91.67% | MIL emits more positions, but these are not always on target |
| Median of per-video median center error | 111.61px | 58.87px | Typical center error improves |
| Median of per-video P95 center error | 164.59px | 163.87px | Similar tail error |
| Median error / GT bbox width | 9.45x | 11.12x | Relative small-object error worsens |
| Median frame-to-frame residual error | 1.16px | 3.30px | Motion signal becomes less stable |

The candidate does not pass the existing gate because localization success falls,
relative error rises, jitter rises, and one clip times out. Its higher observation
rate alone is not a reason to adopt it.
The center-error, normalized-error, and jitter summaries use clips with emitted
observations; the timeout remains a zero in localization and observation
coverage denominators.

## Where it helped and failed

- `phantom105`: location success 5.98% -> 16.28%; median center error
  253.27px -> 19.98px; coverage 68.44% -> 100%.
- `phantom111`: location success 1.66% -> 45.85%; median center error
  237.50px -> 23.88px; coverage remained 100%.
- `phantom81`: location success 4.65% -> 15.61%; median center error
  125.18px -> 45.27px; coverage 11.63% -> 100%.
- `phantom09`: coverage rose 29.24% -> 100%, but median error worsened
  98.04px -> 553.04px. The tracker kept emitting while following the wrong area.
- `phantom117`: the initial target is about 5x3px after resize. MIL exceeded
  the 45-second limit; D0 localized 99.32% of evaluated frames.
- `phantom30`: location success improved 77.74% -> 97.34%, but median error
  rose 2.72px -> 4.62px and jitter increased.

This is mixed evidence: the online appearance model can recover some D0 drift
cases, while continuous MIL updates also follow distractors and add temporal
noise. CMC frame warping does not remove that identity failure.

## Decision

Do not replace D0 with MIL and do not pass MIL trajectories to C. Keep this as a
research candidate for a possible conditional re-detection role. Before trying a
hybrid, test whether observable evidence available at runtime can distinguish
the clips where MIL helps from those where it leaves the target. Any hybrid must
be evaluated with the same all-frame GT denominator and must retain D0 when the
alternate tracker times out or disagrees without reliable identity evidence.

The experiment is limited to 12 development clips and the classical OpenCV MIL
implementation. It does not evaluate a modern learned SOT or a detector trained
on ARD100.

## Artifacts

- `artifacts/trajectory_experiments/ard100-mil-cmc-probe-20261018-seeded/index.html`
- `artifacts/trajectory_experiments/ard100-mil-cmc-probe-20261018-seeded/summary.json`
- `artifacts/trajectory_experiments/ard100-mil-cmc-probe-20261018-seeded/comparison.csv`
- `research/ard100_sot_probe.py`
