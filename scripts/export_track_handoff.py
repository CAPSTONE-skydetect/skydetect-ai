#!/usr/bin/env python3
"""Export validated TrackSequence fixtures for downstream B/C experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_server.schemas import TrackSequence


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    tracks = [
        TrackSequence.model_validate(track_payload)
        for track_payload in payload.get("tracks", [])
    ]
    if not tracks:
        raise SystemExit(f"No tracks found in {input_path}")

    ranked_tracks = sorted(tracks, key=_selection_key, reverse=True)
    selected_track = ranked_tracks[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_dir / "tracks_all.json",
        {"tracks": [track.model_dump(mode="json") for track in tracks]},
    )
    _write_json(
        output_dir / "selected_track.json",
        selected_track.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "selection_report.json",
        {
            "selection_policy": [
                "quality.num_points descending",
                "quality.missing_ratio ascending",
                "quality.mean_conf descending",
            ],
            "selected_track_id": selected_track.track_id,
            "ground_truth_label": args.ground_truth_label,
            "note": (
                "ground_truth_label is experiment metadata only and is not part "
                "of the TrackSequence passed to B/C."
            ),
            "candidates": [
                {
                    "track_id": track.track_id,
                    "num_points": track.quality.num_points if track.quality else 0,
                    "missing_ratio": (
                        track.quality.missing_ratio if track.quality else 1.0
                    ),
                    "mean_conf": track.quality.mean_conf if track.quality else 0.0,
                    "track_stability": (
                        track.quality.track_stability if track.quality else "poor"
                    ),
                }
                for track in ranked_tracks
            ],
        },
    )

    print(f"Validated tracks: {len(tracks)}")
    print(f"Selected track: {selected_track.track_id}")
    print(f"Output directory: {output_dir}")
    return 0


def _selection_key(track: TrackSequence) -> tuple[int, float, float]:
    quality = track.quality
    if quality is None:
        return (0, -1.0, 0.0)
    return (
        quality.num_points,
        -quality.missing_ratio,
        quality.mean_conf,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate tracker output and export B/C handoff fixtures.",
    )
    parser.add_argument("--input", required=True, help="Tracker JSON containing tracks.")
    parser.add_argument("--output-dir", required=True, help="Fixture output directory.")
    parser.add_argument(
        "--ground-truth-label",
        choices=["bird", "drone"],
        help="Optional experiment label stored only in selection_report.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
