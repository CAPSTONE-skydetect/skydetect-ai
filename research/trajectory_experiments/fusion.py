"""Fuse saved D0/N7 observations without reading GT during coordinate generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ai_server.schemas import TrackSequence
from ai_server.services.complementary_center import ComplementaryCenter
from ai_server.services.tracking_video_io import transform_point
from ai_server.utils.quality import build_track_quality
from research.io import write_json
from research.tracking_evaluation import evaluate_dataset
from research.trajectory_experiments.manifest import sha256


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: str):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def fuse(root: Path, variant: str = "N10"):
    output = root / "variants" / variant
    output.mkdir(parents=True, exist_ok=True)
    for record in sorted((root / "variants/D0").glob("*/worker_result.json")):
        base = _read(record)
        sample_id = base["sample_id"]
        motion_record = _read(root / "variants/N7" / sample_id / "worker_result.json")
        anchors = _rows(base["artifacts"]["trajectory"])
        motion = {int(row["frame_index"]): row for row in _rows(motion_record["artifacts"]["trajectory"])}
        metadata = _read(Path(base["artifacts"]["metadata"]))
        other_metadata = _read(Path(motion_record["artifacts"]["metadata"]))
        for field in ("source_video_path", "fps", "processed_width", "processed_height", "init_frame_index"):
            if metadata[field] != other_metadata[field]:
                raise ValueError(f"fusion inputs differ in {field}")
        camera_path = root / "evaluation/D0/b_inputs" / sample_id / "camera_transforms.json"
        camera = _read(camera_path)
        matrices = {row["frame_index"]: np.asarray(row["current_to_reference"]) for row in camera["frames"]}
        tracker = ComplementaryCenter()
        width, height = metadata["processed_width"], metadata["processed_height"]
        trace = []
        for row in anchors:
            index = int(row["frame_index"])
            timestamp = int(row["timestamp_ms"])
            proposed = motion.get(index)
            available = (row["visible"].lower() == "true" and proposed is not None
                         and proposed["visible"].lower() == "true")
            if available and int(proposed["timestamp_ms"]) != timestamp:
                raise ValueError("fusion timestamps differ")
            xy = tracker.update(
                [float(row["raw_x"]), float(row["raw_y"])],
                [float(proposed["raw_x"]), float(proposed["raw_y"])] if available else None,
                timestamp,
                max_disagreement=max(float(row["bbox_width"]), float(row["bbox_height"])),
            )
            compensated = transform_point(matrices[index], *xy)
            # Preserve schema bounds with an anchor fallback, never a hidden clamp.
            if not (0 <= compensated[0] <= width and 0 <= compensated[1] <= height):
                xy = np.array([float(row["raw_x"]), float(row["raw_y"])])
                compensated = (float(row["compensated_x"]), float(row["compensated_y"]))
                tracker.bias = None
            row["fusion_anchor_source"] = row["tracking_source"]
            row["fusion_motion_source"] = proposed["tracking_source"] if proposed else "missing"
            row["fusion_mode"] = "complementary" if tracker.bias is not None else "anchor_fallback"
            if tracker.bias is not None:
                row["tracking_source"] = "complementary"
            trace.append({"frame_index": index, "anchor_x": float(row["raw_x"]),
                          "anchor_y": float(row["raw_y"]), "output_x": float(xy[0]),
                          "output_y": float(xy[1]), "motion_available": available})
            row.update(raw_x=float(xy[0]), raw_y=float(xy[1]),
                       compensated_x=compensated[0], compensated_y=compensated[1],
                       bbox_x=float(xy[0]) - float(row["bbox_width"]) / 2,
                       bbox_y=float(xy[1]) - float(row["bbox_height"]) / 2)
        folder = output / sample_id
        folder.mkdir(exist_ok=True)
        trajectory_path = folder / "trajectory.csv"
        with trajectory_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(anchors[0]))
            writer.writeheader()
            writer.writerows(anchors)
        original = TrackSequence.model_validate_json(Path(base["artifacts"]["track_sequence"]).read_text())
        by_frame = {int(row["frame_index"]): row for row in anchors}
        history = []
        for point in original.history:
            row = by_frame[point.frame_index]
            history.append(point.model_copy(update={"cx": float(row["compensated_x"]) / width,
                                                   "cy": float(row["compensated_y"]) / height}))
        quality = build_track_quality(history, attempted_frame_count=metadata["attempted_frame_count"])
        track = original.model_copy(update={"history": history, "quality": quality})
        TrackSequence.model_validate(track.model_dump())
        track_path = folder / "track_sequence.json"
        write_json(track_path, track.model_dump())
        metadata["fusion"] = {
            "method": "D0_position_anchor_N7_motion_complementary", "time_constant_seconds": .25,
            "camera_policy": "D0 channel background transforms",
            "anchor_sha256": sha256(base["artifacts"]["trajectory"]),
            "motion_sha256": sha256(motion_record["artifacts"]["trajectory"]),
            "source_sha256": sha256(Path(__file__)),
            "filter_source_sha256": sha256(Path(__file__).parents[2] / "ai_server/services/complementary_center.py"),
        }
        metadata_path = folder / "metadata.json"
        write_json(metadata_path, metadata)
        write_json(folder / "fusion_trace.json", trace)
        evaluation = _read(root / "evaluation_inputs/D0" / sample_id / "sample.json")
        evaluation.pop("comparison_video", None)
        evaluation["provenance_note"] = "N10 causal dual-channel fusion; fixed tau 0.25s; GT used only for evaluation"
        evaluation["prediction"] = {"trajectory": str(trajectory_path.resolve()), "metadata": str(metadata_path.resolve())}
        evaluation["b_comparison"] = {"track_sequence": str(track_path.resolve()),
                                      "camera_transforms": str(camera_path.resolve())}
        input_path = root / "evaluation_inputs" / variant / sample_id / "sample.json"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(input_path, evaluation)
    reports = evaluate_dataset(root / "evaluation_inputs" / variant, root / "evaluation" / variant, compare_b=True)
    return {"status": "completed", "variant": variant, "samples": len(reports)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fuse(args.root.resolve())))
