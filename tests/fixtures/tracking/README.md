# White-road tracking fixture

This is a synthetic regression fixture, not user footage or an accuracy dataset.
It freezes the decoded pixels of the existing white-road test on Linux arm64.
The tracker, tuning, and original assertions are unchanged; the test also checks
that tracking reaches the final frame and verifies every decoded BGR pixel.

## Provenance

- Generator: `_make_white_road_transition_video` in
  `tests/test_manual_roi_tracking.py` at commit `147cdcf` (retained for diagnostics).
- Original encoding: OpenCV mp4v, 320x180, 20 fps, 60 frames.
- Reference runtime: Linux arm64, OpenCV 4.13.0, NumPy 2.4.4,
  `skydetect-unified-check-rf` image ID `79f6aeb6b567`.
- Creation: generate the original MP4, decode all frames with that runtime, and
  write the decoded BGR frames to FFV1/AVI without further lossy encoding.
- File SHA-256:
  `ce22825fc1c794f42c3fb83b7772a3eeac5c172f76bfca980da3311a05bd6a2e`.
- Decoded BGR SHA-256 (all frame bytes concatenated):
  `a65c4660cb3e2389879727b2084e9436945c56fb89d87e6bda8631f1a178561a`.

## Why this input is fixed

PR #49 run 43033185088 failed on both amd64 model jobs. The old test generated
and decoded an MP4 at runtime, so its pixels were not a portable reference.
Even the *same* arm64-generated MP4 decoded to different BGR hashes on amd64.
The FFV1 reference decoded identically on both and retained the original
low-contrast, motion-support, position-error, and coverage assertions.

Diagnostic results before changing any production code:

| Input / runtime | Low-contrast frames | Visible ratio | Last frame | Last x |
| --- | ---: | ---: | ---: | ---: |
| Runtime MP4 / arm64 | 24 | 0.90 | 59 | 214.821 |
| Runtime MP4 / amd64 | 2 | 0.65 | 42 | 127.637 |
| Same arm64 MP4 / amd64 | 4 | 0.5833 | 34 | 130.416 |
| Frozen FFV1 / arm64 | 24 | 0.90 | 59 | 214.821 |
| Frozen FFV1 / amd64 | 24 | 0.90 | 59 | 214.821 |

This makes the regression input reproducible; it does **not** fix or conceal the
tracker's sensitivity to compressed low-contrast footage. The generated lossy
variants exposed real tracking failures and remain an unresolved A robustness
limitation. Do not interpret CI passing as equivalent behavior on arbitrary
MP4s, architectures, or real videos. Neither codec settings in production nor
model weights, thresholds, or tracker implementation are changed by this fix.
