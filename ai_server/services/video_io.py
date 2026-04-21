from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VideoIOError(ValueError):
    """Raised when video metadata or frames cannot be read."""


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_ms: int
    frame: Any


def load_video_metadata(video_path: str) -> VideoMetadata:
    capture = _open_video_capture(video_path)
    try:
        fps = float(capture.get(_cv2().CAP_PROP_FPS))
        total_frames = int(capture.get(_cv2().CAP_PROP_FRAME_COUNT))
        width = int(capture.get(_cv2().CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(_cv2().CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if fps <= 0:
        raise VideoIOError(f"Could not determine FPS from video: {video_path}")
    if total_frames <= 0:
        raise VideoIOError(f"Could not determine frame count from video: {video_path}")
    if width <= 0 or height <= 0:
        raise VideoIOError(f"Could not determine frame size from video: {video_path}")

    return VideoMetadata(
        path=str(Path(video_path).expanduser()),
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
    )


def iter_video_frames(video_path: str) -> Iterator[VideoFrame]:
    metadata = load_video_metadata(video_path)
    capture = _open_video_capture(video_path)

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            yield VideoFrame(
                frame_index=frame_index,
                timestamp_ms=_frame_timestamp_ms(frame_index, metadata.fps),
                frame=frame,
            )
            frame_index += 1
    finally:
        capture.release()


def _frame_timestamp_ms(frame_index: int, fps: float) -> int:
    return round((frame_index / fps) * 1000)


def _open_video_capture(video_path: str) -> Any:
    path = Path(video_path).expanduser()
    if not path.exists():
        raise VideoIOError(f"Video path does not exist: {path}")
    if not path.is_file():
        raise VideoIOError(f"Video path is not a file: {path}")

    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoIOError(f"Failed to open video file: {path}")

    return capture


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise VideoIOError(
            "OpenCV is required for video loading. Install 'opencv-python-headless' "
            "to enable metadata extraction and frame iteration."
        ) from exc

    return cv2
