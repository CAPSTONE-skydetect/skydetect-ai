from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from ai_server.schemas import TrackPoint, TrackSequence
from ai_server.services.tracking_adapter import observations_to_track_sequence
from ai_server.tracking_schemas import ManualTrackingRequest


@dataclass
class _Observation:
    frame_index: int
    timestamp_ms: int
    raw_x: float
    raw_y: float
    compensated_x: float
    compensated_y: float
    w: float = 20.0
    h: float = 12.0
    confidence: float = 0.8
    visible: bool = True


def test_adapter_emits_only_observed_points_and_preserves_gaps() -> None:
    track = observations_to_track_sequence(
        [
            _Observation(0, 0, 100.0, 40.0, 90.0, 35.0),
            _Observation(1, 40, 104.0, 42.0, 94.0, 37.0, visible=False),
            _Observation(2, 80, 108.0, 44.0, 98.0, 39.0),
        ],
        track_id=7,
        source_video_id="clip-001",
        frame_width=200,
        frame_height=100,
        stabilize=True,
    )

    assert [point.frame_index for point in track.history] == [0, 2]
    assert track.history[0].cx == pytest.approx(0.45)
    assert track.history[0].cy == pytest.approx(0.35)
    assert track.quality is not None
    assert track.quality.num_points == 2
    assert track.quality.mean_conf == 0.8
    assert track.quality.missing_ratio == pytest.approx(0.333, abs=0.001)
    assert track.stabilization is not None
    assert track.stabilization.method == "opencv_feature_cmc"


def test_adapter_uses_raw_coordinates_without_cmc() -> None:
    track = observations_to_track_sequence(
        [_Observation(0, 0, 100.0, 40.0, 90.0, 35.0)],
        track_id=1,
        source_video_id="clip-002",
        frame_width=200,
        frame_height=100,
        stabilize=False,
    )

    assert track.history[0].cx == pytest.approx(0.5)
    assert track.history[0].cy == pytest.approx(0.4)
    assert track.stabilization is not None
    assert track.stabilization.applied is False


def test_track_sequence_rejects_duplicate_or_unordered_frames() -> None:
    point = TrackPoint(
        frame_index=1,
        timestamp_ms=40,
        cx=0.5,
        cy=0.5,
        w=0.1,
        h=0.1,
        conf=0.9,
    )
    with pytest.raises(ValidationError, match="strictly ordered"):
        TrackSequence(track_id=1, history=[point, point])


def test_manual_tracking_request_validates_bbox_and_tuning() -> None:
    request = ManualTrackingRequest(
        source_video_id="clip-003",
        video_path="/tmp/clip.mp4",
        target_bbox=(10.0, 20.0, 24.0, 18.0),
    )
    assert request.tuning.online_update_enabled is False

    with pytest.raises(ValidationError):
        ManualTrackingRequest(
            source_video_id="clip-003",
            video_path="/tmp/clip.mp4",
            target_bbox=(10.0, 20.0, 0.0, 18.0),
        )
