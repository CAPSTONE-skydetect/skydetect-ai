# Local bird false-positive check

Five distinct local files were screened. The user confirms they show birds but
does not know whether the MiniRocket training used them. None matches a known
video SHA-256 in the received training inventory. Some training video hashes
are missing, and re-encodes / filming-session overlap cannot be ruled out.
This is an exploratory check, not an independent false-positive benchmark.

Initial ROIs were selected from decoded frames before classifier scores were
inspected. No later GT coordinates were used. B05 has multiple birds; the upper
bird at frame100 was selected because it is visible there. Other clips start at
frame0. Each run uses at most 10 seconds, fixed confidence settings and CMC.

| ID | File | FPS | Existing C / new C with D0 |
| --- | --- | ---: | --- |
| B01 | bird:fix:stab:1.mp4 | 30 | drone / drone (false alarm) |
| B02 | bird:nonstab:far:nobackground:track.avi | 14 | uncertain / uncertain |
| B03 | bird:shake:nonstab:zoom:no backgroundavi.avi | 14 | uncertain / uncertain |
| B04 | bird:shake:zoom:BG:track:long:far.mp4 | 14 | uncertain / uncertain |
| B05 | birds:zoom:shake:track:npBG.avi | 50 | uncertain / uncertain |

The same outcomes occur with the AR1 size-aware KLT option. AR2 is a separate
candidate and is not represented by these bird results.

B02-B04 violate the classifier's existing minimum source-rate contract of
15Hz; they also have substantial tracking loss. Timestamps were not falsified
and no duplicated/interpolated input was used to disguise the source rate.
B05 has insufficient contiguous observation support for an accepted 2s window.

B01 has nine accepted windows and 100% emitted observation coverage. Spot checks
at +2/+5/+8 seconds show the small target near the tracked center. This is not
full-frame GT validation. D0 margins are +0.2967456 for the original C and
+1.0489743 for the candidate: both incorrectly predict drone. These are margins,
not probabilities. One decided false alarm and four abstentions cannot establish
a reliable false-positive rate or safe deployment.

Raising the candidate threshold beyond the B01 margin would reject 16 of the
26 correctly detected ARD100 test drones at the same frozen D0 input. This is a
descriptive tradeoff check, not a threshold selected for deployment. No threshold
or weights were changed to fit this bird example.

## Fix completed during the check

The experimental stacked adapter previously raised ValueError on unsupported
source input while the baseline returned uncertain/invalid_input. It now follows
the same abstention contract and preserves rejection reasons when no windows
survive. The 14Hz cases now return uncertain with a reason, rather than an
unavailable/crashed prediction. The underlying model weights are unchanged.
An existing valid drone input still returns margin 1.7105640242029445 exactly.

Tests execute invalid-rate and insufficient-window cases against the separate
MiniRocket runtime so the RF branch's classifier types are not substituted.
The patched adapter is included in
`artifacts/trajectory_experiments/minirocket-results-20261018/candidate_bundle_input_guard_v2.zip`.
The older bundle is retained.

## Artifacts

- `artifacts/trajectory_experiments/bird-screening-20261018/inventory.json`
- `artifacts/trajectory_experiments/bird-screening-20261018/contact.jpg`
- `artifacts/trajectory_experiments/bird-screening-20261018/tracking_contact.jpg`
- `artifacts/trajectory_experiments/bird-screening-20261018/classification.csv`
- `artifacts/trajectory_experiments/bird-screening-20261018/classification_details.json`
- `artifacts/trajectory_experiments/bird-screening-20261018/summary.json`
- Per-run original overlays, metadata and trajectory CSV under `tracking/`.

Original videos and the production model are unchanged. To measure bird false
alarms independently, source-reviewed >=15Hz bird clips with usable tracking
support are still needed; no new filming-source independence is assumed here.
