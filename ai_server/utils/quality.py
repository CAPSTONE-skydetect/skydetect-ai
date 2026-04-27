from ai_server.schemas import TrackPoint, TrackQuality


def build_track_quality(history: list[TrackPoint]) -> TrackQuality:
    num_points = len(history)
    mean_conf = sum(point.conf for point in history) / num_points
    missing_ratio = 0.0

    if num_points >= 4 and mean_conf >= 0.85:
        track_stability = "good"
    elif num_points >= 3 and mean_conf >= 0.70:
        track_stability = "fair"
    else:
        track_stability = "poor"

    return TrackQuality(
        num_points=num_points,
        mean_conf=round(mean_conf, 3),
        missing_ratio=missing_ratio,
        track_stability=track_stability,
    )
