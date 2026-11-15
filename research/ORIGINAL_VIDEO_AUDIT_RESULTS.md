# Original-video review of A tracks (2026-09-28)

## Scope and decision

The files in `research/data/original_video/` allow A-to-video alignment and
exploratory image inspection. They do **not** by themselves provide independent
object-center ground truth or recording-session identities. No simulator
calibration or bulk-dataset approval resulted from this review. The existing
6,000-row anchor-derived dataset remains experimental.

## 1. Video-to-track alignment

The audit covered 51 supplied videos (30 bird, 21 drone) and 54 A tracks
(33 bird, 21 drone). The extra bird tracks are multiple objects from a video.
Files marked `(보정)` are treated as the already-prepared inputs supplied to A,
not as CMC outputs. All 54 pairings passed frame-index bounds, elapsed-time,
FPS and processed-vs-video aspect-ratio checks. Of these, 43 also matched the
previously recorded video byte hash **and** an exact A `trajectory.csv` export
containing `raw_x/raw_y`. The other 11 have metadata-consistent name pairings
but no recorded video hash or matching raw-coordinate CSV. They must not be
called byte-verified or used for raw-pixel error measurements.

The three `새36` object-specific video files have different byte hashes but
share one source-video group according to the user's recording identity rule;
they are not independent cases. Different videos may also come from one
shooting session. Session independence remains unverified.

Local audit artifacts: `output/original_video_audit_v1/video_track_matches.csv`
and `output/original_video_audit_v1/audit_summary.json`.

## 2. Image location and center comparison

For five **train** clips, 30 consecutive A-observed frames each were inspected
using the exact matched A `raw_x/raw_y`, scaled from A's processed resolution
to the supplied video resolution. The frame sheets show both unmarked frames
and A-seeded locations. This establishes that the expected object is visible
near the A coordinate in these examples. CMC-compensated `TrackSequence.cx/cy`
were **not** plotted directly onto raw video.

An image-derived dark-silhouette center was calculated inside a small crop
around the A seed. The mask is A-seeded and can follow the wrong object or
background; it is **not independent manual ground truth**. The following are
exploratory A-to-proxy offsets in original-video pixels, not certified tracker
errors:

| Train clip | Proxy found | Median offset | P95 offset | Median selected silhouette area |
|---|---:|---:|---:|---:|
| `새30(1)` | 25/30 | 1.63 px | 2.60 px | 8 px² |
| `새24` | 30/30 | 0.45 px | 0.63 px | 42 px² |
| `새36(1)` | 30/30 | 0.56 px | 1.16 px | 20.5 px² |
| `드론21` | 30/30 | 0.45 px | 0.89 px | 130 px² |
| `드론22` | 30/30 | 0.41 px | 1.00 px | 146 px² |

The two drones are visibly large enough for a useful center check in the
selected frames. `새30(1)` is only a few pixels across; in several frames the
A center appears displaced from the visible dark spot, but a 1–3 pixel
silhouette/annotation ambiguity is itself large relative to the bird.
`새24` visibly changes its wing silhouette while the A location stays close
to the image-derived center, showing that silhouette change need not produce
a large A-center oscillation. `새36(1)` also changes its silhouette while the
tracked location stays close. These observations justify examining target-size-dependent measurement
uncertainty, **not** declaring any frequency component to be wingbeat or
tracking error. Frame-by-frame manual labels, with uncertain frames excluded,
would be needed for a defensible tracker-error distribution.

Review sheets and frame-level proxy CSVs are local under
`output/original_video_audit_v1/contact_sheets/` and
`output/original_video_audit_v1/visual_centers/`.

## 3. Simulator sensitivity, same 2-second contract

The existing `trajectory-sequence-1.0.1` real/synthetic comparison still has
opposing class gaps: median 2-second image span is real/synthetic bird
`0.2779/0.1599` and drone `0.0546/0.1254`. Merely enlarging the camera view
or increasing jitter cannot fix both directions. The current observation
model already increases jitter with simulated target difficulty, but the A ROI
size is not a verified object-size measurement.

A single **exploratory** observation candidate doubled the
`jitter_difficulty_gain` from 1 to 2, leaving flight and camera rules fixed.
It was specified before train-group CV. The lower-is-better group-distance
objective was:

| Fold | Existing observation model | Higher difficulty jitter |
|---|---:|---:|
| 0 | 1.1509 | 1.2786 |
| 1 | 1.8124 | 1.8547 |
| 2 | 1.2313 | 1.2959 |
| Mean | **1.3982** | 1.4764 |

The candidate worsened all three train held-out folds, failed the preregistered
gate, and was rejected. **Validation and test were not opened** for this
candidate; no simulator default was changed. The protocol and per-fold CSV are
in `output/video_noise_hypothesis_v1/`.

## What remains before a credible bulk dataset

1. Manually review short, class-balanced video intervals and mark visible
   object centers independently of the A point; record annotation uncertainty
   and `unresolvable` where the target is too small, blurred or occluded.
2. Preserve exact source-video grouping and obtain recording-session IDs when
   available. Video hashes verify identical bytes, not session independence.
3. Use train groups only to estimate observation error and behavior/camera
   differences separately. Try a bounded, predeclared simulator change;
   reject it if held-out train groups worsen.
4. Freeze the candidate before using real validation. A genuinely new,
   untouched real bird/drone cohort is still required for a final generalization
   claim. The existing test was historically inspected.

The supplied videos have materially improved **auditability** and make manual
reference annotation possible, but they have not yet demonstrated that the
simulator reproduces real A sequences well enough for approved bulk training.
