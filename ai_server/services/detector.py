from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from ai_server.services.video_io import VideoFrame


@dataclass(frozen=True)
class Detection:
    left: int
    top: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class FrameDetections:
    frame_index: int
    timestamp_ms: int
    detections: list[Detection] = field(default_factory=list)


def detect_flying_objects(frames: Iterable[VideoFrame]) -> Iterator[FrameDetections]:
    """Yield frame-wise detector outputs.

    This placeholder emits one deterministic debug box for the first few frames
    so the A pipeline can produce non-empty TrackSequence outputs during
    end-to-end testing.
    """

    for frame in frames:
        detections: list[Detection] = []
        if frame.frame_index < 12:
            detections = [_build_fallback_detection(frame)]

        yield FrameDetections(
            frame_index=frame.frame_index,
            timestamp_ms=frame.timestamp_ms,
            detections=detections,
        )


def _build_fallback_detection(frame: VideoFrame) -> Detection:
    height, width = frame.frame.shape[:2]
    box_width = max(24, width // 14)
    box_height = max(24, height // 14)
    horizontal_margin = max(1, width - box_width)
    vertical_margin = max(1, height - box_height)

    left = min(
        horizontal_margin,
        int(horizontal_margin * (0.35 + (0.02 * frame.frame_index))),
    )
    top = min(
        vertical_margin,
        int(vertical_margin * (0.30 + (0.015 * frame.frame_index))),
    )

    return Detection(
        left=left,
        top=top,
        width=box_width,
        height=box_height,
        confidence=0.4,
    )
