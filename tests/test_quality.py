from ai_server.schemas import TrackPoint
from ai_server.utils.quality import build_track_quality


def _point(
    frame_index: int,
    timestamp_ms: int,
    conf: float,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
) -> TrackPoint:
    return TrackPoint(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        cx=cx,
        cy=cy,
        w=0.1,
        h=0.1,
        conf=conf,
    )


def test_build_track_quality_matches_history_summary() -> None:
    history = [
        _point(0, 0, 0.91, cx=0.10, cy=0.10),
        _point(1, 40, 0.90, cx=0.12, cy=0.12),
        _point(2, 80, 0.89, cx=0.14, cy=0.14),
        _point(3, 120, 0.88, cx=0.16, cy=0.16),
        _point(4, 160, 0.90, cx=0.18, cy=0.18),
        _point(5, 200, 0.91, cx=0.20, cy=0.20),
    ]

    quality = build_track_quality(history)

    assert quality.num_points == 6
    assert quality.mean_conf == 0.898
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
    assert quality.missing_ratio == 0.333
    assert quality.track_stability == "poor"


def test_build_track_quality_penalizes_large_frame_gaps() -> None:
    history = [
        _point(0, 0, 0.95, cx=0.10, cy=0.10),
        _point(3, 120, 0.95, cx=0.13, cy=0.13),
        _point(6, 240, 0.95, cx=0.16, cy=0.16),
        _point(9, 360, 0.95, cx=0.19, cy=0.19),
        _point(12, 480, 0.95, cx=0.22, cy=0.22),
        _point(15, 600, 0.95, cx=0.25, cy=0.25),
    ]

    quality = build_track_quality(history)

    assert quality.missing_ratio == 0.625
    assert quality.track_stability == "poor"


def test_build_track_quality_penalizes_jittery_tracks() -> None:
    history = [
        _point(0, 0, 0.90, cx=0.10, cy=0.10),
        _point(1, 40, 0.90, cx=0.16, cy=0.16),
        _point(2, 80, 0.90, cx=0.11, cy=0.11),
        _point(3, 120, 0.90, cx=0.18, cy=0.18),
        _point(4, 160, 0.90, cx=0.12, cy=0.12),
        _point(5, 200, 0.90, cx=0.20, cy=0.20),
    ]

    quality = build_track_quality(history)

    assert quality.missing_ratio == 0.0
    assert quality.track_stability == "poor"
