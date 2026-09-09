import csv
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from ai_server.main import app
from ai_server.schemas import TrackSequence
from ai_server.services.manual_roi_tracker import process_manual_roi_video
from ai_server.tracking_schemas import TrackingTuning


def test_tracks_twenty_pixel_target_and_exports_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "twenty_pixel_target.mp4"
    _make_twenty_pixel_video(video_path)

    result = process_manual_roi_video(
        str(video_path),
        source_video_id="twenty-pixel-target",
        output_dir=tmp_path / "outputs",
        target_bbox=(40.0, 54.0, 32.0, 32.0),
        max_seconds=2.5,
        stabilize=False,
        resize_width=None,
        write_overlay=False,
        tuning=TrackingTuning(online_update_enabled=True),
    )

    assert len(result.track.history) >= 40
    assert result.track.history[-1].cx > result.track.history[0].cx + 0.20
    assert result.track.quality is not None
    assert result.metrics["visible_ratio"] > 0.75
    assert result.metrics["tracking_source_counts"]["klt"] > 20

    track_path = Path(result.artifacts["track_sequence"] or "")
    persisted = TrackSequence.model_validate_json(track_path.read_text(encoding="utf-8"))
    assert persisted == result.track


def test_predictions_stay_out_of_track_history(tmp_path: Path) -> None:
    video_path = tmp_path / "occluded_target.mp4"
    _make_twenty_pixel_video(
        video_path,
        occluded_frames=set(range(18, 23)),
    )

    result = process_manual_roi_video(
        str(video_path),
        source_video_id="occluded-target",
        output_dir=tmp_path / "outputs",
        target_bbox=(40.0, 54.0, 32.0, 32.0),
        max_seconds=2.5,
        stabilize=False,
        resize_width=None,
        write_overlay=False,
        tuning=TrackingTuning(online_update_enabled=True),
    )

    trajectory_path = Path(result.artifacts["trajectory"] or "")
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    prediction_frames = {
        int(row["frame_index"])
        for row in rows
        if row["tracking_source"] == "prediction"
    }
    history_frames = {point.frame_index for point in result.track.history}
    assert prediction_frames
    assert prediction_frames.isdisjoint(history_frames)
    assert result.metrics["tracking_source_counts"]["appearance"] >= 1
    assert result.track.quality is not None
    assert result.track.quality.missing_ratio > 0.0


def test_tracks_white_target_across_white_road(tmp_path: Path) -> None:
    video_path = tmp_path / "white_target_white_road.mp4"
    _make_white_road_transition_video(video_path)

    result = process_manual_roi_video(
        str(video_path),
        source_video_id="white-road-target",
        output_dir=tmp_path / "outputs",
        target_bbox=(36.0, 70.0, 40.0, 36.0),
        max_seconds=3.0,
        stabilize=True,
        resize_width=None,
        write_overlay=False,
        tuning=TrackingTuning(
            klt_accept_conf=0.44,
            recovery_conf=0.55,
            search_radius_multiplier=2.5,
            online_update_enabled=False,
        ),
    )

    final_point = result.track.history[-1]
    expected_x = 56.0 + 59 * 2.6
    assert result.metadata["foreground_mask_segmented"] is True
    assert result.metrics["low_contrast_frames"] >= 4
    assert result.metrics["tracking_source_counts"]["motion"] >= 1
    assert abs(final_point.cx * 320.0 - expected_x) < 12.0
    assert result.metrics["visible_ratio"] > 0.65


def test_analyze_endpoint_returns_manual_roi_track(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "api_target.mp4"
    _make_twenty_pixel_video(video_path, frame_count=20)
    monkeypatch.setenv("SKYDETECT_TRACK_OUTPUT_DIR", str(tmp_path / "api_outputs"))

    response = TestClient(app).post(
        "/analyze",
        json={
            "source_video_id": "api-target",
            "video_path": str(video_path),
            "target_bbox": [40.0, 54.0, 32.0, 32.0],
            "stabilize": False,
            "resize_width": None,
            "write_overlay": False,
            "tuning": {"online_update_enabled": False},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_video_id"] == "api-target"
    assert len(payload["tracks"]) == 1
    track = TrackSequence.model_validate(payload["tracks"][0])
    assert len(track.history) >= 15
    assert all(
        current.frame_index > previous.frame_index
        for previous, current in zip(track.history, track.history[1:])
    )


def _make_twenty_pixel_video(
    path: Path,
    frame_count: int = 48,
    fps: float = 20.0,
    occluded_frames: set[int] | None = None,
) -> None:
    width, height = 320, 200
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()

    for frame_index in range(frame_count):
        frame = np.full((height, width, 3), (205, 214, 222), dtype=np.uint8)
        center_x = int(round(56 + frame_index * 2.0))
        center_y = int(round(70 + np.sin(frame_index / 8.0) * 5.0))
        if not occluded_frames or frame_index not in occluded_frames:
            cv2.line(
                frame,
                (center_x - 10, center_y),
                (center_x + 10, center_y),
                (25, 31, 37),
                3,
            )
            cv2.line(
                frame,
                (center_x, center_y - 6),
                (center_x, center_y + 6),
                (25, 31, 37),
                3,
            )
            cv2.circle(frame, (center_x, center_y), 4, (238, 242, 245), -1)
            cv2.circle(frame, (center_x, center_y), 5, (25, 31, 37), 1)
        writer.write(frame)

    writer.release()


def _make_white_road_transition_video(
    path: Path,
    frame_count: int = 60,
    fps: float = 20.0,
) -> None:
    width, height = 320, 180
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()

    for frame_index in range(frame_count):
        frame = np.full((height, width, 3), (52, 128, 58), dtype=np.uint8)
        for x in range(8, 126, 19):
            for y in range(10, height, 23):
                cv2.circle(frame, (x, y), 1, (44, 109, 48), -1)
        cv2.rectangle(frame, (128, 0), (width, height), (238, 238, 238), -1)
        for x in range(142, width, 31):
            cv2.line(frame, (x, 0), (x, height), (234, 234, 234), 1)

        center_x = int(round(56.0 + frame_index * 2.6))
        center_y = int(round(88.0 + np.sin(frame_index / 9.0) * 2.0))
        body = np.array(
            [
                [center_x - 10, center_y],
                [center_x - 3, center_y - 4],
                [center_x + 10, center_y],
                [center_x - 3, center_y + 4],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, body, (249, 249, 249), lineType=cv2.LINE_AA)
        cv2.line(
            frame,
            (center_x - 2, center_y - 7),
            (center_x + 2, center_y + 7),
            (246, 246, 246),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
