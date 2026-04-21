from math import hypot

from ai_server.schemas import TrackPoint, TrackQuality


def build_track_quality(history: list[TrackPoint]) -> TrackQuality:
    num_points = len(history)
    mean_conf = sum(point.conf for point in history) / num_points
    missing_ratio = _compute_missing_ratio(history)
    center_jitter = _compute_center_jitter(history)

    if (
        num_points >= 6
        and mean_conf >= 0.85
        and missing_ratio <= 0.10
        and center_jitter <= 0.02
    ):
        track_stability = "good"
    elif (
        num_points >= 3
        and mean_conf >= 0.65
        and missing_ratio <= 0.40
        and center_jitter <= 0.08
    ):
        track_stability = "fair"
    else:
        track_stability = "poor"

    return TrackQuality(
        num_points=num_points,
        mean_conf=round(mean_conf, 3),
        missing_ratio=round(missing_ratio, 3),
        track_stability=track_stability,
    )


def _compute_missing_ratio(history: list[TrackPoint]) -> float:
    if len(history) <= 1:
        return 0.0

    expected_span = history[-1].frame_index - history[0].frame_index + 1
    if expected_span <= 1:
        return 0.0

    missing_frames = expected_span - len(history)
    return missing_frames / expected_span


def _compute_center_jitter(history: list[TrackPoint]) -> float:
    if len(history) <= 2:
        return 0.0

    step_vectors = [
        (
            current.cx - previous.cx,
            current.cy - previous.cy,
        )
        for previous, current in zip(history, history[1:])
    ]
    acceleration_magnitudes = [
        hypot(
            current_dx - previous_dx,
            current_dy - previous_dy,
        )
        for (previous_dx, previous_dy), (current_dx, current_dy) in zip(
            step_vectors,
            step_vectors[1:],
        )
    ]
    if not acceleration_magnitudes:
        return 0.0

    return sum(acceleration_magnitudes) / len(acceleration_magnitudes)
