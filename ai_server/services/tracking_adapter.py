from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ai_server.schemas import StabilizationInfo, TrackPoint, TrackSequence
from ai_server.utils.quality import build_track_quality


class Observation(Protocol):
    frame_index: int
    timestamp_ms: int
    raw_x: float
    raw_y: float
    compensated_x: float
    compensated_y: float
    w: float
    h: float
    confidence: float
    visible: bool


def observations_to_track_sequence(
    observations: Sequence[Observation],
    *,
    track_id: int,
    source_video_id: str,
    frame_width: int,
    frame_height: int,
    stabilize: bool,
) -> TrackSequence:
    if not observations:
        raise ValueError("tracking produced no observations")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero")

    history = [
        _to_track_point(
            observation,
            frame_width=frame_width,
            frame_height=frame_height,
            stabilize=stabilize,
        )
        for observation in observations
        if observation.visible
    ]
    if not history:
        raise ValueError("tracking produced no visible observations")

    attempted_frame_count = len(
        {observation.frame_index for observation in observations}
    )
    return TrackSequence(
        track_id=track_id,
        source_video_id=source_video_id,
        stabilization=StabilizationInfo(
            applied=stabilize,
            method="opencv_feature_cmc" if stabilize else "none",
        ),
        history=history,
        quality=build_track_quality(
            history,
            attempted_frame_count=attempted_frame_count,
        ),
    )


def _to_track_point(
    observation: Observation,
    *,
    frame_width: int,
    frame_height: int,
    stabilize: bool,
) -> TrackPoint:
    x = observation.compensated_x if stabilize else observation.raw_x
    y = observation.compensated_y if stabilize else observation.raw_y
    return TrackPoint(
        frame_index=int(observation.frame_index),
        timestamp_ms=int(observation.timestamp_ms),
        cx=_normalize(x, frame_width),
        cy=_normalize(y, frame_height),
        w=_normalize(observation.w, frame_width),
        h=_normalize(observation.h, frame_height),
        conf=_clamp(float(observation.confidence), 0.0, 1.0),
    )


def _normalize(value: float, denominator: int) -> float:
    return _clamp(float(value) / float(denominator), 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
