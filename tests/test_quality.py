from ai_server.schemas import TrackPoint
from ai_server.utils.quality import build_track_quality


def _point(frame_index: int, timestamp_ms: int, conf: float) -> TrackPoint:
    return TrackPoint(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        cx=0.5,
        cy=0.5,
        w=0.1,
        h=0.1,
        conf=conf,
    )


def test_build_track_quality_matches_history_summary() -> None:
    history = [
        _point(0, 0, 0.91),
        _point(1, 40, 0.90),
        _point(2, 80, 0.89),
        _point(3, 120, 0.88),
    ]

    quality = build_track_quality(history)

    assert quality.num_points == 4
    assert quality.mean_conf == 0.895
    assert quality.missing_ratio == 0.0
    assert quality.track_stability == "good"


def test_build_track_quality_uses_actual_points_without_interpolation() -> None:
    history = [
        _point(0, 0, 0.80),
        _point(2, 80, 0.80),
    ]

    quality = build_track_quality(history)

    assert quality.num_points == 2
    assert quality.mean_conf == 0.8
    assert quality.track_stability == "poor"
