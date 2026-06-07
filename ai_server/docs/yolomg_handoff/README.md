# YOLOMG A-to-B/C Handoff

This directory provides a reproducible real-video `TrackSequence` fixture for
Part B feature extraction and Part C classification experiments.

The shared schema is not modified by this handoff. Files under this directory
are experiment fixtures and metadata.

## Reference Case

- Video ID: `drone_fix_stab_01`
- Ground truth: `drone`
- Source filename: `drone:fix:stab:1.mpg`
- Source SHA-256:
  `547d06655091ad20ad9265556db66604f6c369759b2a8e7b444f4c4c4e3e443e`
- Source metadata: 1920x1080, 50 fps, 400 frames, 8 seconds
- Evaluated clip: first 5 seconds, 250 frames
- Detector: YOLOMG
- Tracker: `sort_center`
- Confidence threshold: `0.20`
- Image size: `1280`
- Motion blur kernel: `11`
- Stabilization: `none` because the source is already stabilized
- YOLOMG commit: `090a74cd3ece15c66e857cc2e08e01cb103c4550`
- Weight SHA-256:
  `fb28a4063f0e935ce419db47cf6ff26c443d12940eb727522a95644adba34bef`

## Fixture Files

- `drone_fix_stab_01/tracks_all.json`: all four `sort_center` candidates
- `drone_fix_stab_01/selected_track.json`: recommended TrackSequence for B/C
- `drone_fix_stab_01/selection_report.json`: selection policy and candidate quality
- `drone_fix_stab_01/summary.csv`: detector/tracker run summary

The selected track is track ID 1:

- 246 points across frames 2-247
- 0.0 internal missing ratio
- 0.814 mean confidence
- 98.4% coverage of the evaluated 250 frames

The other candidates are retained so downstream owners can test their own
track-selection or rejection policy. The `ground_truth_label` appears only in
experiment metadata and is deliberately excluded from `TrackSequence`.

## Use Without YOLOMG

Part B can load `selected_track.json` directly as a `TrackSequence`:

```python
import json
from pathlib import Path

from ai_server.schemas import TrackSequence

path = Path(
    "ai_server/docs/yolomg_handoff/"
    "drone_fix_stab_01/selected_track.json"
)
track = TrackSequence.model_validate(json.loads(path.read_text()))
```

Use `timestamp_ms` when deriving time-dependent features. The source is 50 fps,
while the current RF training data was produced at 30 fps. If B resamples the
track, preserve the raw fixture and report interpolation separately.

## Reproduce From Video

Prepare a local YOLOMG checkout and weight file, then run:

```bash
python scripts/run_detector_eval.py \
  --dataset-dir "/path/to/dataset" \
  --output-root artifacts/experiments/yolomg_handoff_drone_fix_stab_01 \
  --detectors yolomg \
  --trackers sort_center \
  --stabilization-methods none \
  --conf 0.20 \
  --imgsz 1280 \
  --eval-clip-sec 5 \
  --video-id drone_fix_stab_01 \
  --yolomg-repo /path/to/YOLOMG \
  --yolomg-weights /path/to/best.pt
```

Export the tracker output:

```bash
python scripts/export_track_handoff.py \
  --input artifacts/experiments/yolomg_handoff_drone_fix_stab_01/tracks/drone_fix_stab_01__none__yolomg__conf020__sort_center.json \
  --output-dir /tmp/drone_fix_stab_01_handoff \
  --ground-truth-label drone
```

## Requested B/C Checks

1. Extract the five core trajectory features from `selected_track.json`.
2. Compare native 50 fps processing with timestamp-based 30 fps resampling.
3. Record interpolation policy and `feature_status`/`imputed_fields`.
4. Run the C rule filter and RF classifier for both feature variants.
5. Report label, confidence, rejection reason, and feature values.
6. Repeat against `tracks_all.json` to verify that low-quality candidates are
   rejected instead of being treated as additional real objects.
