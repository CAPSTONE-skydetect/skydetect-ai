from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TrackingVideoError(ValueError):
    """Raised when a tracking video cannot be read or written."""


@dataclass(frozen=True, slots=True)
class TrackingVideoMetadata:
    video_name: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    processed_width: int
    processed_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_name": self.video_name,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_sec": self.duration_sec,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
        }


def read_tracking_video_metadata(
    video_path: str | Path,
    *,
    resize_width: int | None = None,
) -> TrackingVideoMetadata:
    path = Path(video_path).expanduser()
    if not path.exists() or not path.is_file():
        raise TrackingVideoError(f"Video path does not exist or is not a file: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise TrackingVideoError(f"Failed to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise TrackingVideoError(f"Invalid video metadata: {path}")

    processed_width = width
    processed_height = height
    if resize_width and width > resize_width > 0:
        scale = resize_width / width
        processed_width = int(round(width * scale))
        processed_height = int(round(height * scale))

    return TrackingVideoMetadata(
        video_name=path.name,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=frame_count / fps,
        processed_width=processed_width,
        processed_height=processed_height,
    )


def iter_tracking_video_frames(
    video_path: str | Path,
    *,
    resize_width: int | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(Path(video_path).expanduser()))
    if not capture.isOpened():
        raise TrackingVideoError(f"Failed to open video: {video_path}")

    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if resize_width and frame.shape[1] > resize_width > 0:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (
                        int(round(frame.shape[1] * scale)),
                        int(round(frame.shape[0] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def create_tracking_video_writer(
    output_path: str | Path,
    *,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise TrackingVideoError(f"Failed to create overlay video: {path}")
    return writer


def write_tracking_json(payload: Any, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def write_tracking_csv(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return str(path)


def affine_to_homography(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(np.float32)
    result = np.eye(3, dtype=np.float32)
    result[:2, :] = matrix[:2, :]
    return result


def transform_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    homography = affine_to_homography(matrix)
    point = homography @ np.array([x, y, 1.0], dtype=np.float32)
    scale = float(point[2]) if abs(float(point[2])) > 1e-6 else 1.0
    return float(point[0] / scale), float(point[1] / scale)
