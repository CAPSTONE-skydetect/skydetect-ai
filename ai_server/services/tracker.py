from ai_server.schemas import StabilizationInfo, TrackPoint, TrackSequence
from ai_server.utils.quality import build_track_quality


def build_bootstrap_track(
    track_id: int,
    source_video_id: str,
    stabilization: StabilizationInfo,
) -> TrackSequence:
    history = [
        TrackPoint(
            frame_index=0,
            timestamp_ms=0,
            cx=0.40,
            cy=0.62,
            w=0.03,
            h=0.02,
            conf=0.91,
        ),
        TrackPoint(
            frame_index=1,
            timestamp_ms=40,
            cx=0.42,
            cy=0.60,
            w=0.03,
            h=0.02,
            conf=0.90,
        ),
        TrackPoint(
            frame_index=2,
            timestamp_ms=80,
            cx=0.44,
            cy=0.58,
            w=0.04,
            h=0.03,
            conf=0.89,
        ),
        TrackPoint(
            frame_index=3,
            timestamp_ms=120,
            cx=0.47,
            cy=0.56,
            w=0.04,
            h=0.03,
            conf=0.88,
        ),
    ]
    return TrackSequence(
        track_id=track_id,
        source_video_id=source_video_id,
        stabilization=stabilization,
        history=history,
        quality=build_track_quality(history),
    )
