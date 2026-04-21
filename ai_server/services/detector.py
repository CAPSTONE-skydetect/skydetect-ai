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

    This placeholder keeps the detector contract stable while the actual
    small-object detector is still being wired into the pipeline.
    """

    for frame in frames:
        yield FrameDetections(
            frame_index=frame.frame_index,
            timestamp_ms=frame.timestamp_ms,
            detections=[],
        )
