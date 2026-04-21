import pytest
from pydantic import ValidationError

from ai_server.schemas import TrackPoint, TrackSequence


def _point(
    frame_index: int,
    timestamp_ms: int,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
    w: float = 0.1,
    h: float = 0.1,
    conf: float = 0.9,
) -> TrackPoint:
    return TrackPoint(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        conf=conf,
    )


def test_track_sequence_rejects_unordered_history() -> None:
    with pytest.raises(ValidationError, match="history must be strictly ordered"):
        TrackSequence(
            track_id=1,
            history=[
                _point(1, 40),
                _point(0, 0),
            ],
        )


def test_track_sequence_rejects_duplicate_frame_index() -> None:
    with pytest.raises(ValidationError, match="history must be strictly ordered"):
        TrackSequence(
            track_id=1,
            history=[
                _point(0, 0),
                _point(0, 40),
            ],
        )


def test_track_sequence_rejects_timestamp_regression() -> None:
    with pytest.raises(ValidationError, match="timestamp_ms must not go backwards"):
        TrackSequence(
            track_id=1,
            history=[
                _point(0, 40),
                _point(1, 20),
            ],
        )


def test_track_sequence_allows_frame_gaps_without_interpolation() -> None:
    track = TrackSequence(
        track_id=1,
        history=[
            _point(0, 0),
            _point(2, 80),
            _point(5, 200),
        ],
    )

    assert [point.frame_index for point in track.history] == [0, 2, 5]
    assert len(track.history) == 3


def test_track_point_rejects_out_of_range_normalized_coordinates() -> None:
    with pytest.raises(ValidationError):
        _point(0, 0, cx=1.2)

    with pytest.raises(ValidationError):
        _point(0, 0, w=-0.1)
