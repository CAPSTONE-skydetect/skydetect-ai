import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ai_server.schemas import StabilizationInfo, TrackPoint, TrackSequence
from ai_server.services.feature_core import build_feature_vector
from research.io import write_json
from research.tracking_evaluation import TrackingEvaluationError, evaluate_files
from research.tracking_feature_comparison import compare_b_features, _verified_matrices


def test_camera_replay_uses_saved_analysis_resolution(tmp_path: Path) -> None:
    import cv2
    from ai_server.services.manual_roi_tracker import process_manual_roi_video
    from ai_server.tracking_schemas import TrackingTuning
    from research.tracking_feature_comparison import replay_camera_transforms

    video = tmp_path / "textured.avi"
    rng = np.random.default_rng(4)
    base = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 25, (640, 360))
    for index in range(10):
        writer.write(np.roll(base, index, axis=1))
    writer.release()
    result = process_manual_roi_video(
        str(video), source_video_id="test", target_bbox=(200, 140, 32, 32),
        init_frame_index=0, max_seconds=None, stabilize=True,
        tuning=TrackingTuning(cmc_analysis_width=320), output_dir=tmp_path / "runs",
        resize_width=None, write_overlay=False,
    )
    replay = replay_camera_transforms(
        Path(result.artifacts["metadata"]), Path(result.artifacts["debug_observations"]),
        tmp_path / "camera.json",
    )
    assert replay["inputs"]["cmc_analysis_width"] == 320
    assert replay["verification"]["max_residual_px"] <= 0.01


def _sample(root: Path, *, cmc: bool = False) -> tuple[dict, TrackSequence]:
    root.mkdir(parents=True)
    width, height, count = 640, 360, 40
    metadata = {"width": width, "height": height, "processed_width": width,
                "processed_height": height, "fps": 30, "frame_count": count,
                "tracking_start_frame": 0, "attempted_frame_count": count,
                "coordinate_mode": "camera_motion_compensated" if cmc else "frame"}
    write_json(root / "metadata.json", metadata)
    boxes, rows, points, matrices = [], [], [], []
    for i in range(count):
        gx, gy = 100 + i * 2, 120
        ax, ay = gx + 10, gy - 4
        # A scale+translation CMC proves that subtracting an A-only camera offset
        # is insufficient to transform a different GT location.
        matrix = np.array([[.9, 0, 3], [0, .9, 5], [0, 0, 1]], dtype=np.float32) if cmc else np.eye(3)
        cx, cy, _ = matrix @ [ax, ay, 1]
        visible = i != 20
        rows.append({"frame_index": i, "timestamp_ms": round(i / 30 * 1000),
                     "raw_x": ax, "raw_y": ay, "compensated_x": cx, "compensated_y": cy,
                     "bbox_width": 10, "bbox_height": 8, "confidence": .9,
                     "visible": visible, "tracking_source": "klt"})
        if visible:
            points.append(TrackPoint(frame_index=i, timestamp_ms=round(i / 30 * 1000),
                                     cx=cx / width, cy=cy / height, w=10 / width, h=8 / height, conf=.9))
        boxes.append(f'<box frame="{i}" outside="0" occluded="0" keyframe="1" xtl="{gx-5}" ytl="{gy-4}" xbr="{gx+5}" ybr="{gy+4}"/>')
        matrices.append({"frame_index": i, "current_to_reference": matrix.tolist()})
    (root / "gt.xml").write_text(
        f'<annotations><meta><task><size>{count}</size><original_size><width>{width}</width><height>{height}</height></original_size></task></meta><track id="0" label="drone">'
        + "".join(boxes) + '</track></annotations>', encoding="utf-8",
    )
    with (root / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    track = TrackSequence(track_id=1, processed_width=width, processed_height=height,
                          history=points, stabilization=StabilizationInfo(applied=cmc, method="opencv_feature_cmc" if cmc else "none"))
    write_json(root / "track_sequence.json", track.model_dump(mode="json"))
    write_json(root / "camera.json", {"coordinate_space": "processed_pixels", "reference_frame": 0, "frames": matrices})
    report = evaluate_files(cvat_xml=root / "gt.xml", trajectory_csv=root / "trajectory.csv",
                            metadata_json=root / "metadata.json", sample_id="test", track_id=0)
    return report, track


def test_calls_actual_b_and_preserves_observation_gaps(tmp_path: Path) -> None:
    report, track = _sample(tmp_path / "sample")
    result = compare_b_features(report, manifest={}, manifest_root=tmp_path,
                                output_dir=tmp_path / "output")
    assert result["status"] == "completed"
    assert result["point_counts"] == {"a": 39, "gt_full": 40, "gt_matched": 39}
    expected = build_feature_vector(track).feature_vector.features.model_dump()
    assert result["results"]["a"]["features"] == expected
    assert max(r["absolute_difference"] for r in result["comparisons"]) < 1e-6
    exported = json.loads((tmp_path / "output/gt_matched_track_sequence.json").read_text())
    assert 20 not in {p["frame_index"] for p in exported["history"]}


def test_full_cmc_matrix_is_applied_to_gt(tmp_path: Path) -> None:
    report, _ = _sample(tmp_path / "sample", cmc=True)
    result = compare_b_features(
        report, manifest={"b_comparison": {"camera_transforms": "sample/camera.json"}},
        manifest_root=tmp_path, output_dir=tmp_path / "output",
    )
    assert result["status"] == "completed"
    gt = json.loads((tmp_path / "output/gt_full_track_sequence.json").read_text())
    assert gt["history"][0]["cx"] == pytest.approx((.9 * 100 + 3) / 640)
    assert gt["history"][0]["cy"] == pytest.approx((.9 * 120 + 5) / 360)
    # Float32 CMC point rounding is amplified by B's acceleration derivative.
    assert max(r["absolute_difference"] for r in result["comparisons"]) < 1e-3


def test_missing_cmc_is_reported_without_guessing(tmp_path: Path) -> None:
    report, _ = _sample(tmp_path / "sample", cmc=True)
    result = compare_b_features(report, manifest={}, manifest_root=tmp_path,
                                output_dir=tmp_path / "output")
    assert result["status"] == "not_comparable"
    assert "camera_transforms" in result["reasons"][0]


def test_invalid_camera_transform_is_rejected() -> None:
    payload = {"coordinate_space": "processed_pixels", "reference_frame": 0,
               "frames": [{"frame_index": 0, "current_to_reference": np.eye(3).tolist()}]}
    rows = [{"frame_index": "0", "raw_x": "10", "raw_y": "10",
             "compensated_x": "20", "compensated_y": "10"}]
    with pytest.raises(TrackingEvaluationError, match="reproduce"):
        _verified_matrices(payload, rows, 0)


def test_gt_outside_schema_is_not_silently_clamped(tmp_path: Path) -> None:
    report, track = _sample(tmp_path / "sample", cmc=True)
    camera_path = tmp_path / "sample/camera.json"
    payload = json.loads(camera_path.read_text())
    for frame in payload["frames"]:
        frame["current_to_reference"][0][2] = -95
    # Keep the A handoff internally consistent while sending GT outside [0, 1].
    rows = list(csv.DictReader((tmp_path / "sample/trajectory.csv").open()))
    for row in rows:
        row["compensated_x"] = str(.9 * float(row["raw_x"]) - 95)
    with (tmp_path / "sample/trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    adjusted = [p.model_copy(update={"cx": max(0, (.9 * (100+p.frame_index*2+10)-95) / 640)}) for p in track.history]
    write_json(tmp_path / "sample/track_sequence.json", track.model_copy(update={"history": adjusted}).model_dump(mode="json"))
    write_json(camera_path, payload)
    result = compare_b_features(report, manifest={"b_comparison": {"camera_transforms": str(camera_path)}},
                                manifest_root=tmp_path, output_dir=tmp_path / "output")
    assert result["status"] == "not_comparable"
    assert "no silent clamp" in result["reasons"][0]
