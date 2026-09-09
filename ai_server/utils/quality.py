from math import hypot

from ai_server.schemas import TrackPoint, TrackQuality


def build_track_quality(
    history: list[TrackPoint],
    *,
    attempted_frame_count: int | None = None,
) -> TrackQuality:
    if not history:
        raise ValueError("history must contain at least one observed point")

    num_points = len(history)
    mean_conf = sum(point.conf for point in history) / num_points
    expected_frames = attempted_frame_count or _history_span(history)
    if expected_frames < num_points:
        raise ValueError("attempted_frame_count cannot be smaller than observed points")
    missing_ratio = (expected_frames - num_points) / expected_frames
    center_jitter = _compute_center_jitter(history)

    if (
        num_points >= 4
        and mean_conf >= 0.75
        and missing_ratio <= 0.15
        and center_jitter <= 0.02
    ):
        track_stability = "good"
    elif (
        num_points >= 3
        and mean_conf >= 0.40
        and missing_ratio <= 0.50
        and center_jitter <= 0.08
    ):
        track_stability = "fair"
    else:
        track_stability = "poor"

    return TrackQuality(
        num_points=num_points,
        mean_conf=round(mean_conf, 3),
        missing_ratio=missing_ratio,
        track_stability=track_stability,
    )


def _history_span(history: list[TrackPoint]) -> int:
    return max(1, history[-1].frame_index - history[0].frame_index + 1)


def _compute_center_jitter(history: list[TrackPoint]) -> float:
    if len(history) <= 2:
        return 0.0

    velocities: list[tuple[float, float]] = []
    for previous, current in zip(history, history[1:]):
        frame_delta = current.frame_index - previous.frame_index
        if frame_delta <= 0:
            continue
        velocities.append(
            (
                (current.cx - previous.cx) / frame_delta,
                (current.cy - previous.cy) / frame_delta,
            )
        )

    acceleration = [
        hypot(current_dx - previous_dx, current_dy - previous_dy)
        for (previous_dx, previous_dy), (current_dx, current_dy) in zip(
            velocities,
            velocities[1:],
        )
    ]
    return sum(acceleration) / len(acceleration) if acceleration else 0.0
