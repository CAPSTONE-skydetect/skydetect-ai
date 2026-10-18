import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from research.tracking_evaluation import (
    TrackingEvaluationError,
    evaluate_dataset,
    evaluate_files,
    load_video_geometry,
    write_comparison_video,
    _temporal_metrics,
)


def test_evaluates_cvat_against_resized_raw_trajectory(tmp_path: Path) -> None:
    gt_path, trajectory_path, metadata_path = _write_sample_files(tmp_path)

    report = evaluate_files(
        cvat_xml=gt_path,
        trajectory_csv=trajectory_path,
        metadata_json=metadata_path,
        sample_id="sample-a",
        track_id=0,
    )

    assert report["coordinate_space"] == "source_video_pixels"
    assert report["video"]["transform_source"] == "declared_resize"
    assert report["counts"] == {
        "gt_visible_frames": 3,
        "prediction_visible_frames": 2,
        "matched_visible_frames": 2,
        "missing_gt_frames": 1,
        "prediction_frames_without_gt": 0,
    }
    assert report["coverage"]["full_gt_observation_ratio"] == pytest.approx(2 / 3)
    assert report["coverage"]["active_span_observation_ratio"] == 1.0
    assert report["coverage"]["attempted_window_observation_ratio"] == pytest.approx(
        2 / 3
    )
    assert report["localization"]["observed_frame_error_px"]["median"] == 0.0
    assert report["timeline"]["missing_frame_ranges"] == [[2, 2]]
    assert report["timeline"]["early_termination_frames"] == 1
    assert report["localization"]["pixel_threshold_success"]["within_5_px"][
        "attempted_window_ratio"
    ] == pytest.approx(2 / 3)


def test_explicit_transform_supports_letterbox_coordinates(tmp_path: Path) -> None:
    gt_path, trajectory_path, metadata_path = _write_sample_files(
        tmp_path,
        processed_width=80,
        processed_height=80,
        centers=[(16.0, 36.0), (24.0, 36.0), (32.0, 36.0)],
        original_to_processed=[
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 20.0],
            [0.0, 0.0, 1.0],
        ],
    )

    report = evaluate_files(
        cvat_xml=gt_path,
        trajectory_csv=trajectory_path,
        metadata_json=metadata_path,
        sample_id="letterbox",
        track_id=0,
    )

    assert report["video"]["transform_source"] == "explicit_matrix"
    assert report["localization"]["observed_frame_error_px"]["max"] == 0.0


def test_attempted_window_starts_at_selected_roi_frame(tmp_path: Path) -> None:
    gt_path, trajectory_path, metadata_path = _write_sample_files(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "num_frames_processed": 2,
            "tracking_start_frame": 1,
            "attempted_frame_count": 2,
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = evaluate_files(
        cvat_xml=gt_path,
        trajectory_csv=trajectory_path,
        metadata_json=metadata_path,
        sample_id="mid-video-roi",
        track_id=0,
    )

    assert report["timeline"]["attempted_first_frame"] == 1
    assert report["timeline"]["attempted_last_frame"] == 2
    assert report["coverage"]["attempted_window_observation_ratio"] == 0.5


def test_legacy_aspect_ratio_change_is_rejected_without_transform(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "width": 100,
                "height": 50,
                "processed_width": 80,
                "processed_height": 80,
                "fps": 10.0,
                "frame_count": 3,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrackingEvaluationError, match="explicit transform"):
        load_video_geometry(metadata_path)


def test_explicit_track_ids_merge_non_overlapping_cvat_fragments(tmp_path: Path) -> None:
    gt_path, trajectory_path, metadata_path = _write_sample_files(tmp_path)
    gt_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta><task><size>3</size><original_size><width>100</width><height>50</height></original_size></task></meta>
  <track id="0" label="drone" source="manual">
    <box frame="2" outside="0" occluded="0" keyframe="1" xtl="35" ytl="15" xbr="45" ybr="25" />
  </track>
  <track id="1" label="drone" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="15" ytl="15" xbr="25" ybr="25" />
    <box frame="1" outside="0" occluded="0" keyframe="1" xtl="25" ytl="15" xbr="35" ybr="25" />
  </track>
</annotations>
""",
        encoding="utf-8",
    )

    report = evaluate_files(
        cvat_xml=gt_path,
        trajectory_csv=trajectory_path,
        metadata_json=metadata_path,
        sample_id="fragmented",
        track_ids=(0, 1),
    )

    assert report["target"]["source_track_ids"] == [0, 1]
    assert report["counts"]["gt_visible_frames"] == 3
    assert report["counts"]["matched_visible_frames"] == 2


def test_dataset_manifest_writes_machine_and_human_reports(tmp_path: Path) -> None:
    sample_dir = tmp_path / "dataset" / "sample-a"
    gt_path, trajectory_path, metadata_path = _write_sample_files(sample_dir)
    manifest = {
        "sample_id": "sample-a",
        "ground_truth": {"path": gt_path.name, "track_id": 0},
        "prediction": {
            "trajectory": trajectory_path.name,
            "metadata": metadata_path.name,
        },
        "alignment": {"prediction_frame_offset": 0},
    }
    (sample_dir / "sample.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output_dir = tmp_path / "results"

    reports = evaluate_dataset(tmp_path / "dataset", output_dir)

    assert len(reports) == 1
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "per_video" / "sample-a.json").exists()
    summary = (output_dir / "summary.csv").read_text(encoding="utf-8")
    assert "attempted_window_gt_frames" in summary
    assert "within_0_5_bbox_attempted_ratio" in summary
    summary_json = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary_json["aggregate"]["micro"][
        "attempted_window_observation_ratio"
    ] == pytest.approx(2 / 3)


def test_writes_gt_comparison_video(tmp_path: Path) -> None:
    gt_path, trajectory_path, metadata_path = _write_sample_files(tmp_path)
    overlay_path = tmp_path / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (50, 24)
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((24, 50, 3), dtype=np.uint8))
    writer.release()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["processed_height"] = 24
    metadata["original_to_processed"] = [
        [0.5, 0.0, 0.0],
        [0.0, 0.48, 0.0],
        [0.0, 0.0, 1.0],
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output_path = tmp_path / "comparison.mp4"

    result = write_comparison_video(
        cvat_xml=gt_path,
        trajectory_csv=trajectory_path,
        metadata_json=metadata_path,
        overlay_video=overlay_path,
        output_video=output_path,
        sample_id="sample-a",
        track_id=0,
    )

    assert result["frames_written"] == 3
    assert result["matched_frames"] == 2
    assert result["gt_frames_missing_prediction"] == 1
    assert output_path.exists()
    capture = cv2.VideoCapture(str(output_path))
    ok, rendered = capture.read()
    capture.release()
    assert ok
    assert np.count_nonzero(rendered) > 0


def _temporal_frame(index: int, gt: tuple[float, float], a: tuple[float, float]) -> dict:
    return {"frame_index": index, "timestamp_ms": index * 100,
            "gt_center": list(gt), "prediction_center": list(a)}


def test_constant_center_bias_does_not_change_motion() -> None:
    frames = [_temporal_frame(i, (i * 2, 0), (i * 2 + 10, -4)) for i in range(4)]
    result = _temporal_metrics(frames, list(range(4)))
    assert result["error_change_px"]["max"] == 0
    assert result["velocity_vector_error_px_s"]["max"] == 0
    assert result["path_length_ratio"] == 1


def test_error_change_uses_vectors_and_timestamp_intervals() -> None:
    frames = [_temporal_frame(0, (0, 0), (2, 0)),
              _temporal_frame(1, (0, 0), (-2, 0))]
    result = _temporal_metrics(frames, [0, 1])
    assert result["error_change_px"]["median"] == 4
    assert result["velocity_vector_error_px_s"]["median"] == 40
    assert result["path_length_ratio"] is None


def test_temporal_metrics_do_not_bridge_missing_frames() -> None:
    frames = [_temporal_frame(0, (0, 0), (0, 0)),
              _temporal_frame(2, (4, 0), (100, 0)),
              _temporal_frame(3, (6, 0), (102, 0))]
    result = _temporal_metrics(frames, [0, 1, 2, 3])
    assert result["pair_count"] == 1
    assert result["pair_coverage"] == pytest.approx(1 / 3)
    assert result["prediction_path_length_px"] == 2
    assert result["path_length_ratio"] == 1


def test_temporal_metrics_have_explicit_empty_state() -> None:
    result = _temporal_metrics([_temporal_frame(0, (0, 0), (0, 0))], [0])
    assert result["pair_coverage"] is None
    assert result["error_change_px"]["median"] is None
    assert result["path_length_ratio"] is None


def _write_sample_files(
    root: Path,
    *,
    processed_width: int = 50,
    processed_height: int = 25,
    centers: list[tuple[float, float]] | None = None,
    original_to_processed: list[list[float]] | None = None,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    gt_path = root / "annotations.xml"
    gt_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta><task><size>3</size><original_size><width>100</width><height>50</height></original_size></task></meta>
  <track id="0" label="drone" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="15" ytl="15" xbr="25" ybr="25" />
    <box frame="1" outside="0" occluded="0" keyframe="1" xtl="25" ytl="15" xbr="35" ybr="25" />
    <box frame="2" outside="0" occluded="0" keyframe="1" xtl="35" ytl="15" xbr="45" ybr="25" />
  </track>
</annotations>
""",
        encoding="utf-8",
    )
    metadata = {
        "width": 100,
        "height": 50,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "fps": 10.0,
        "frame_count": 3,
        "preprocessing": {"mode": "resize"},
    }
    if original_to_processed is not None:
        metadata["original_to_processed"] = original_to_processed
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    centers = centers or [(10.0, 10.0), (15.0, 10.0), (20.0, 10.0)]
    trajectory_path = root / "trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame_index",
            "timestamp_ms",
            "raw_x",
            "raw_y",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "confidence",
            "visible",
            "tracking_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index, (cx, cy) in enumerate(centers):
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": frame_index * 100,
                    "raw_x": cx,
                    "raw_y": cy,
                    "bbox_x": cx - 2.5,
                    "bbox_y": cy - 2.5,
                    "bbox_width": 5.0,
                    "bbox_height": 5.0,
                    "confidence": 0.9,
                    "visible": frame_index < 2,
                    "tracking_source": "klt" if frame_index else "manual_roi",
                }
            )
    return gt_path, trajectory_path, metadata_path
