# A Tracking Evaluation

This evaluator measures the manual-ROI A tracker against rectangle tracks
exported as `CVAT for video 1.1` XML.

## Coordinate contract

- CVAT boxes are interpreted in source-video pixel coordinates.
- `trajectory.csv` is evaluated using `raw_x` and `raw_y`, not CMC-compensated
  coordinates.
- A prediction is converted back to source-video pixels using
  `metadata.json.original_to_processed`.
- Legacy A artifacts without a matrix are accepted only when source and
  processed aspect ratios match. They are treated as direct resize output.
- Trajectories are never min-max normalized or shifted to improve a match.
- Frames are matched by `frame_index`. A non-zero offset must be declared in
  `sample.json`; the evaluator never estimates one silently.

This separation matters when CMC is enabled: CVAT annotates the visible source
frame, while `TrackSequence.cx/cy` may be in the CMC reference coordinate
space. A's localization accuracy must therefore use the raw trajectory output.

## One sample

```bash
python -m research.tracking_evaluation single \
  --gt /path/to/annotations.xml \
  --trajectory /path/to/trajectory.csv \
  --metadata /path/to/metadata.json \
  --sample-id bird_2 \
  --track-id 0 \
  --output /path/to/bird_2_evaluation.json
```

Use `--label drone` instead of `--track-id` only when that label identifies one
track. If one physical object was accidentally split into non-overlapping CVAT
tracks, repeat `--track-id` to merge only those explicit fragments:

```bash
python -m research.tracking_evaluation single \
  ... \
  --track-id 0 --track-id 1 --track-id 2
```

The evaluator rejects overlapping fragments and fragments with different
labels. It never merges every track with the same label automatically.

## Dataset layout

Each sample has one `sample.json`. Referenced paths are relative to that file.

```text
evaluation_dataset/
└── bird_2/
    ├── sample.json
    ├── ground_truth/
    │   └── annotations.xml
    └── prediction/
        ├── trajectory.csv
        ├── metadata.json
        ├── track_sequence.json
        ├── metrics.json
        └── overlay.mp4
```

Example manifest:

```json
{
  "sample_id": "bird_2",
  "ground_truth": {
    "path": "ground_truth/annotations.xml",
    "track_id": 0,
    "label": "drone",
    "include_occluded": true
  },
  "prediction": {
    "trajectory": "prediction/trajectory.csv",
    "metadata": "prediction/metadata.json"
  },
  "alignment": {
    "prediction_frame_offset": 0,
    "source_is_trimmed": false
  }
}
```

Evaluate every manifest below a dataset root:

```bash
python -m research.tracking_evaluation dataset evaluation_dataset \
  --output evaluation_results
```

Outputs:

- `summary.json`: complete machine-readable batch report
- `summary.csv`: one row per video for analysis and plotting
- `report.html`: compact human-readable table
- `per_video/<sample_id>.json`: frame-level errors and failure ranges

## Metrics

- `full_gt_observation_ratio`: GT-visible frames with a visible A observation
- `active_span_observation_ratio`: observations between A's first and last
  visible prediction, separating internal losses from late start/early finish
- `attempted_window_observation_ratio`: observations within the frame range A
  actually processed according to `metadata.json.num_frames_processed`
- observed-frame center error in source pixels: mean, median, p95, maximum
- center error divided by the GT bbox diagonal
- success at 5, 10, and 20 source pixels
- success at 0.25, 0.5, and 1.0 GT bbox diagonals
- missing and extra frame ranges
- longest consecutive missing run
- early termination in frames and milliseconds

Threshold reports contain both an observed-frame ratio and a full-GT ratio.
The latter counts missing predictions as failures and is the end-to-end number
to use when comparing full-length tracker versions. They also include an
`attempted_window_ratio`, which counts misses as failures but limits the
denominator to frames A actually processed. Use this for deliberately partial
runs such as `max_seconds` experiments.

## Trimmed inputs and partial processing

If A receives the same full source but stops early because of `max_seconds`, do
not set an offset. `frame_count` remains the full source length and
`num_frames_processed` defines the attempted evaluation window.

If A receives a verified trimmed copy whose frame zero corresponds to frame 200
of the CVAT source, declare both facts explicitly:

```json
{
  "alignment": {
    "prediction_frame_offset": 200,
    "source_is_trimmed": true
  }
}
```

This is valid only when both videos have the same FPS and no frames were
resampled or removed inside the clip. For variable-rate conversion or edited
cuts, create separate continuous samples or provide a frame mapping instead of
using a single offset.

## Input validation

Evaluation stops instead of guessing when:

- CVAT and metadata source dimensions or frame counts differ;
- a CVAT selector resolves to zero or multiple tracks;
- trajectory rows duplicate a visible frame;
- crop, padding, or an aspect-ratio-changing legacy preprocessing step has no
  explicit coordinate transform;
- required metadata or trajectory columns are absent.
