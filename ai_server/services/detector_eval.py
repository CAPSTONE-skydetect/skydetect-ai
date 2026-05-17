from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Protocol

import numpy as np

from ai_server.schemas import StabilizationInfo, TrackPoint, TrackSequence
from ai_server.services.detector import Detection, FrameDetections
from ai_server.services.tracker import build_track_sequences
from ai_server.services.video_io import VideoFrame, VideoMetadata, iter_video_frames, load_video_metadata
from ai_server.utils.quality import build_track_quality


LOGGER = logging.getLogger(__name__)


DEFAULT_DATASET_DIR = Path("/Users/yuchan/Desktop/데이터셋")
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "experiments" / "yolomg_eval"
)


@dataclass(frozen=True)
class EvaluationVideoCase:
    video_id: str
    filename: str
    label: str
    condition_tags: tuple[str, ...]
    split: str
    eval_clip_sec: float | None = None

    def path(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.filename


DEFAULT_VIDEO_CASES: tuple[EvaluationVideoCase, ...] = (
    EvaluationVideoCase(
        video_id="drone_fix_stab_01",
        filename="drone:fix:stab:1.mpg",
        label="drone",
        condition_tags=("fix", "stab"),
        split="drone_easy",
    ),
    EvaluationVideoCase(
        video_id="drone_fix_stab_02",
        filename="drone:fix:stab:2.mp4",
        label="drone",
        condition_tags=("fix", "stab"),
        split="drone_easy",
    ),
    EvaluationVideoCase(
        video_id="drone_fix_stab_03",
        filename="drone:fix:stab:3.mp4",
        label="drone",
        condition_tags=("fix", "stab"),
        split="drone_easy",
    ),
    EvaluationVideoCase(
        video_id="drone_fix_stab_04",
        filename="drone:fix:stab:4.mp4",
        label="drone",
        condition_tags=("fix", "stab"),
        split="drone_easy",
    ),
    EvaluationVideoCase(
        video_id="drone_fix_stab_far_01",
        filename="drone:fix:stab4:far.mp4",
        label="drone",
        condition_tags=("fix", "stab", "far"),
        split="drone_hard",
    ),
    EvaluationVideoCase(
        video_id="drone_unfix_unstab_01",
        filename="drone:unfix:unstab.avi",
        label="drone",
        condition_tags=("unfix", "unstab", "unstable_camera"),
        split="drone_hard",
    ),
    EvaluationVideoCase(
        video_id="bird_fix_stab_01",
        filename="bird:fix:stab:1.mp4",
        label="bird",
        condition_tags=("fix", "stab"),
        split="bird",
    ),
    EvaluationVideoCase(
        video_id="bird_nonstab_far_nobg_track_01",
        filename="bird:nonstab:far:nobackground:track.avi",
        label="bird",
        condition_tags=("nonstab", "far", "nobackground", "track"),
        split="bird",
    ),
    EvaluationVideoCase(
        video_id="bird_zoom_shake_track_npbg_01",
        filename="birds:zoom:shake:track:npBG.avi",
        label="bird",
        condition_tags=("zoom", "shake", "track", "partial_bg"),
        split="bird",
    ),
    EvaluationVideoCase(
        video_id="bird_shake_zoom_bg_track_long_far_01",
        filename="bird:shake:zoom:BG:track:long:far.mp4",
        label="bird",
        condition_tags=("shake", "zoom", "bg", "track", "long", "far"),
        split="bird",
    ),
    EvaluationVideoCase(
        video_id="bird_shake_nonstab_zoom_nobg_01",
        filename="bird:shake:nonstab:zoom:no backgroundavi.avi",
        label="bird",
        condition_tags=("shake", "nonstab", "zoom", "nobackground"),
        split="bird",
    ),
)


class DetectorUnavailableError(RuntimeError):
    """Raised when an optional detector backend is not installed or configured."""


class DetectorAdapter(Protocol):
    name: str

    def detect_frames(self, frames: Iterable[VideoFrame]) -> Iterator[FrameDetections]:
        """Yield frame-wise detections for the supplied video frames."""


@dataclass
class UltralyticsYOLOAdapter:
    """Baseline YOLO detector adapter, loaded lazily to keep dependencies optional."""

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    image_size: int = 1280
    name: str = "baseline_yolo"

    def __post_init__(self) -> None:
        self._model: Any | None = None

    def detect_frames(self, frames: Iterable[VideoFrame]) -> Iterator[FrameDetections]:
        model = self._load_model()
        for frame in frames:
            results = model.predict(
                source=frame.frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                verbose=False,
            )
            yield FrameDetections(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                detections=_ultralytics_results_to_detections(results),
            )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise DetectorUnavailableError(
                "ultralytics is not installed. Install it or pass --detectors without "
                "'baseline' to skip the YOLO baseline."
            ) from exc

        self._model = YOLO(self.model_path)
        return self._model


@dataclass
class YOLOMGAdapter:
    """YOLOMG best.pt adapter for YOLOv5-style repositories.

    The adapter expects a local clone of https://github.com/Irisky123/YOLOMG and a
    pretrained best.pt. It uses the common YOLOv5 DetectMultiBackend interface.
    If the local YOLOMG checkout diverges, this adapter fails loudly so the
    experiment summary records a setup blocker instead of silently faking results.
    """

    repo_path: str | None = None
    weights_path: str | None = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 1280
    device: str = ""
    name: str = "yolomg"

    def __post_init__(self) -> None:
        self._backend: Any | None = None
        self._torch: Any | None = None
        self._letterbox: Any | None = None
        self._non_max_suppression: Any | None = None
        self._scale_boxes: Any | None = None

    def detect_frames(self, frames: Iterable[VideoFrame]) -> Iterator[FrameDetections]:
        self._load_backend()
        assert self._backend is not None
        assert self._torch is not None
        assert self._letterbox is not None
        assert self._non_max_suppression is not None
        assert self._scale_boxes is not None

        frame_buffer: list[VideoFrame] = []
        emitted_warmup = False
        last_emitted_frame_index = -1

        for frame in frames:
            frame_buffer.append(frame)
            if len(frame_buffer) < 5:
                continue

            if not emitted_warmup:
                for warmup_frame in frame_buffer[:2]:
                    yield FrameDetections(
                        frame_index=warmup_frame.frame_index,
                        timestamp_ms=warmup_frame.timestamp_ms,
                        detections=[],
                    )
                    last_emitted_frame_index = warmup_frame.frame_index
                emitted_warmup = True

            center_frame = frame_buffer[2]
            detections = self._detect_center_frame(
                older_frame=frame_buffer[0].frame,
                center_frame=center_frame.frame,
                newer_frame=frame_buffer[4].frame,
            )
            yield FrameDetections(
                frame_index=center_frame.frame_index,
                timestamp_ms=center_frame.timestamp_ms,
                detections=detections,
            )
            last_emitted_frame_index = center_frame.frame_index
            frame_buffer.pop(0)

        for tail_frame in frame_buffer:
            if tail_frame.frame_index <= last_emitted_frame_index:
                continue
            yield FrameDetections(
                frame_index=tail_frame.frame_index,
                timestamp_ms=tail_frame.timestamp_ms,
                detections=[],
            )

    def _load_backend(self) -> None:
        if self._backend is not None:
            return

        repo_path = _existing_path(
            self.repo_path or os.environ.get("YOLOMG_REPO"),
            "YOLOMG repo",
        )
        weights_path = _existing_path(
            self.weights_path or os.environ.get("YOLOMG_WEIGHTS"),
            "YOLOMG best.pt",
        )

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise DetectorUnavailableError(
                "torch is not installed. Install the YOLOMG requirements before "
                "running the real YOLOMG detector."
            ) from exc

        sys.path.insert(0, str(repo_path))
        try:
            from models.common import DetectMultiBackend
            from utils.augmentations import letterbox
            from utils.general import check_img_size, non_max_suppression
            try:
                from utils.general import scale_boxes
            except ImportError:
                from utils.general import scale_coords

                def scale_boxes(img1_shape: Any, boxes: Any, img0_shape: Any) -> Any:
                    return scale_coords(img1_shape, boxes, img0_shape)

            from utils.torch_utils import select_device
        except Exception as exc:
            raise DetectorUnavailableError(
                "Could not import YOLOv5-style modules from the YOLOMG repo. "
                f"repo_path={repo_path}"
            ) from exc

        device = select_device(self.device)
        original_torch_load = torch.load

        def trusted_checkpoint_load(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = trusted_checkpoint_load
        try:
            backend = DetectMultiBackend(str(weights_path), device=device)
        finally:
            torch.load = original_torch_load

        backend.imgsz = check_img_size((self.image_size, self.image_size), s=backend.stride)

        self._backend = backend
        self._torch = torch
        self._letterbox = letterbox
        self._non_max_suppression = non_max_suppression
        self._scale_boxes = scale_boxes

    def _detect_center_frame(
        self,
        *,
        older_frame: Any,
        center_frame: Any,
        newer_frame: Any,
    ) -> list[Detection]:
        assert self._backend is not None
        assert self._torch is not None
        assert self._letterbox is not None
        assert self._non_max_suppression is not None
        assert self._scale_boxes is not None

        motion_frame = self._build_motion_frame(
            older_frame=older_frame,
            center_frame=center_frame,
            newer_frame=newer_frame,
        )
        tensor = self._frame_to_tensor(center_frame)
        motion_tensor = self._frame_to_tensor(motion_frame)

        pred = self._backend(tensor, motion_tensor, augment=False, visualize=False)
        pred = self._non_max_suppression(
            pred,
            self.confidence_threshold,
            self.iou_threshold,
            classes=None,
            agnostic=False,
        )
        detections: list[Detection] = []
        for det in pred:
            if len(det) == 0:
                continue
            det[:, :4] = self._scale_boxes(tensor.shape[2:], det[:, :4], center_frame.shape).round()
            for *xyxy, conf, _cls in det.tolist():
                left, top, right, bottom = xyxy
                detections.append(
                    Detection(
                        left=max(0, int(left)),
                        top=max(0, int(top)),
                        width=max(1, int(right - left)),
                        height=max(1, int(bottom - top)),
                        confidence=float(conf),
                    )
                )
        return detections

    def _build_motion_frame(
        self,
        *,
        older_frame: Any,
        center_frame: Any,
        newer_frame: Any,
    ) -> Any:
        try:
            import cv2

            return _build_yolomg_fd5_mask_frame(
                older_frame=older_frame,
                center_frame=center_frame,
                newer_frame=newer_frame,
                cv2=cv2,
            )
        except Exception as exc:
            LOGGER.warning("YOLOMG motion mask build failed: %s", exc)
            return np.zeros_like(center_frame)

    def _frame_to_tensor(self, frame: Any) -> Any:
        assert self._backend is not None
        assert self._torch is not None
        assert self._letterbox is not None

        image = self._letterbox(frame, self._backend.imgsz, stride=self._backend.stride)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = image.copy()
        tensor = self._torch.from_numpy(image).to(self._backend.device)
        tensor = tensor.float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor[None]
        return tensor


@dataclass
class OpenCVMotionDebugAdapter:
    """Local smoke-test detector that needs no model weights.

    It is not part of the default evaluation matrix. Use it to verify the runner,
    JSON writers, tracker conversion, and visualization before YOLOMG is installed.
    """

    min_area: int = 16
    max_detections_per_frame: int = 50
    name: str = "debug_motion"

    def __post_init__(self) -> None:
        self._previous_gray: Any | None = None

    def detect_frames(self, frames: Iterable[VideoFrame]) -> Iterator[FrameDetections]:
        import cv2

        for frame in frames:
            gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)
            detections: list[Detection] = []
            if self._previous_gray is not None:
                diff = cv2.absdiff(gray, self._previous_gray)
                _, thresh = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                boxes: list[tuple[int, int, int, int]] = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w * h >= self.min_area:
                        boxes.append((x, y, w, h))
                boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
                for x, y, w, h in boxes[: self.max_detections_per_frame]:
                    detections.append(
                        Detection(left=x, top=y, width=w, height=h, confidence=0.30)
                    )
            self._previous_gray = gray
            yield FrameDetections(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                detections=detections,
            )


def build_manifest(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    cases: Sequence[EvaluationVideoCase] = DEFAULT_VIDEO_CASES,
) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    for case in cases:
        path = case.path(dataset_dir)
        row: dict[str, Any] = {
            "video_id": case.video_id,
            "filename": case.filename,
            "path": str(path),
            "label": case.label,
            "condition_tags": list(case.condition_tags),
            "split": case.split,
            "eval_clip_sec": case.eval_clip_sec,
            "exists": path.exists(),
        }
        if path.exists():
            metadata = load_video_metadata(str(path))
            row.update(
                {
                    "fps": metadata.fps,
                    "total_frames": metadata.total_frames,
                    "width": metadata.width,
                    "height": metadata.height,
                    "duration_sec": round(metadata.total_frames / metadata.fps, 3),
                }
            )
        videos.append(row)

    return {
        "dataset_dir": str(dataset_dir),
        "videos": videos,
        "default_detectors": ["yolomg", "baseline"],
        "default_trackers": ["nn", "sort"],
        "default_stabilization_methods": ["none", "ffmpeg_vidstab"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_detection_json(
    path: Path,
    *,
    source_video_id: str,
    detector_name: str,
    frame_detections: Sequence[FrameDetections],
) -> None:
    write_json(
        path,
        {
            "source_video_id": source_video_id,
            "detector": detector_name,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "timestamp_ms": frame.timestamp_ms,
                    "detections": [
                        {
                            "left": detection.left,
                            "top": detection.top,
                            "width": detection.width,
                            "height": detection.height,
                            "confidence": detection.confidence,
                        }
                        for detection in frame.detections
                    ],
                }
                for frame in frame_detections
            ],
        },
    )


def read_detection_json(path: Path) -> list[FrameDetections]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        FrameDetections(
            frame_index=int(frame["frame_index"]),
            timestamp_ms=int(frame["timestamp_ms"]),
            detections=[
                Detection(
                    left=int(detection["left"]),
                    top=int(detection["top"]),
                    width=int(detection["width"]),
                    height=int(detection["height"]),
                    confidence=float(detection["confidence"]),
                )
                for detection in frame["detections"]
            ],
        )
        for frame in payload["frames"]
    ]


def write_tracks_json(path: Path, tracks: Sequence[TrackSequence]) -> None:
    write_json(
        path,
        {
            "tracks": [track.model_dump(mode="json") for track in tracks],
        },
    )


def iter_limited_video_frames(
    video_path: str,
    max_frames: int | None,
    start_sec: float = 0.0,
) -> Iterator[VideoFrame]:
    start_frame = _start_frame_for_video(video_path, start_sec)
    yielded_count = 0
    for frame in iter_video_frames(video_path):
        if frame.frame_index < start_frame:
            continue
        if max_frames is not None and yielded_count >= max_frames:
            break
        yielded_count += 1
        yield frame


def _start_frame_for_video(video_path: str, start_sec: float) -> int:
    if start_sec <= 0:
        return 0
    metadata = load_video_metadata(video_path)
    return min(metadata.total_frames - 1, max(0, round(metadata.fps * start_sec)))


def max_frames_for_case(
    metadata: VideoMetadata,
    case: EvaluationVideoCase,
    override_clip_sec: float | None,
) -> int | None:
    clip_sec = override_clip_sec if override_clip_sec is not None else case.eval_clip_sec
    if clip_sec is None:
        return None
    return max(1, min(metadata.total_frames, round(metadata.fps * clip_sec)))


def build_tracks_for_tracker(
    tracker_name: str,
    frame_detections: Sequence[FrameDetections],
    metadata: VideoMetadata,
    source_video_id: str,
    stabilization: StabilizationInfo,
) -> list[TrackSequence]:
    if tracker_name == "nn":
        return build_track_sequences(
            frame_detections=list(frame_detections),
            metadata=metadata,
            source_video_id=source_video_id,
            stabilization=stabilization,
        )
    if tracker_name == "sort":
        return _build_sort_like_track_sequences(
            frame_detections=frame_detections,
            metadata=metadata,
            source_video_id=source_video_id,
            stabilization=stabilization,
        )
    raise ValueError(f"Unknown tracker: {tracker_name}")


@dataclass
class _SortTrackState:
    track_id: int
    history: list[TrackPoint]
    last_bbox: tuple[int, int, int, int]
    last_frame_index: int
    vx: float = 0.0
    vy: float = 0.0

    def predicted_bbox(self, frame_index: int) -> tuple[int, int, int, int]:
        gap = max(1, frame_index - self.last_frame_index)
        left, top, width, height = self.last_bbox
        return (
            round(left + self.vx * gap),
            round(top + self.vy * gap),
            width,
            height,
        )


def _build_sort_like_track_sequences(
    frame_detections: Sequence[FrameDetections],
    metadata: VideoMetadata,
    source_video_id: str,
    stabilization: StabilizationInfo,
    *,
    max_age: int = 8,
    iou_threshold: float = 0.05,
) -> list[TrackSequence]:
    tracks: list[_SortTrackState] = []
    next_track_id = 1

    for frame in frame_detections:
        active_indices = [
            index
            for index, track in enumerate(tracks)
            if frame.frame_index - track.last_frame_index <= max_age
        ]
        detections = list(frame.detections)
        matches = _greedy_iou_matches(
            tracks=tracks,
            active_indices=active_indices,
            detections=detections,
            frame_index=frame.frame_index,
            iou_threshold=iou_threshold,
        )
        matched_detection_indices = {detection_index for _, detection_index in matches}

        for track_index, detection_index in matches:
            track = tracks[track_index]
            detection = detections[detection_index]
            previous_bbox = track.last_bbox
            new_bbox = _bbox_from_detection(detection)
            gap = max(1, frame.frame_index - track.last_frame_index)
            track.vx = (_center_x(new_bbox) - _center_x(previous_bbox)) / gap
            track.vy = (_center_y(new_bbox) - _center_y(previous_bbox)) / gap
            track.last_bbox = new_bbox
            track.last_frame_index = frame.frame_index
            track.history.append(
                _track_point_from_detection(
                    detection=detection,
                    frame_index=frame.frame_index,
                    timestamp_ms=frame.timestamp_ms,
                    metadata=metadata,
                )
            )

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            tracks.append(
                _SortTrackState(
                    track_id=next_track_id,
                    history=[
                        _track_point_from_detection(
                            detection=detection,
                            frame_index=frame.frame_index,
                            timestamp_ms=frame.timestamp_ms,
                            metadata=metadata,
                        )
                    ],
                    last_bbox=_bbox_from_detection(detection),
                    last_frame_index=frame.frame_index,
                )
            )
            next_track_id += 1

    return [
        TrackSequence(
            track_id=track.track_id,
            source_video_id=source_video_id,
            stabilization=stabilization,
            history=track.history,
            quality=build_track_quality(track.history),
        )
        for track in tracks
        if track.history
    ]


def summarize_run(
    *,
    case: EvaluationVideoCase,
    detector_name: str,
    tracker_name: str,
    stabilization: StabilizationInfo,
    frame_detections: Sequence[FrameDetections],
    tracks: Sequence[TrackSequence],
    start_sec: float = 0.0,
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    total_frames = len(frame_detections)
    total_detections = sum(len(frame.detections) for frame in frame_detections)
    detected_frames = sum(1 for frame in frame_detections if frame.detections)
    detection_confidences = [
        detection.confidence
        for frame in frame_detections
        for detection in frame.detections
    ]
    main_track = select_main_track(tracks)
    nontrivial_tracks = [
        track
        for track in tracks
        if track.quality is not None and track.quality.num_points >= 3
    ]
    row: dict[str, Any] = {
        "video_id": case.video_id,
        "label": case.label,
        "split": case.split,
        "condition_tags": "|".join(case.condition_tags),
        "detector": detector_name,
        "tracker": tracker_name,
        "stabilization_method": stabilization.method,
        "stabilization_applied": stabilization.applied,
        "eval_start_sec": start_sec,
        "status": status,
        "error": error,
        "evaluated_frames": total_frames,
        "total_detections": total_detections,
        "detected_frame_ratio": _safe_round(detected_frames / total_frames)
        if total_frames
        else 0.0,
        "mean_detections_per_frame": _safe_round(total_detections / total_frames)
        if total_frames
        else 0.0,
        "detector_mean_conf": _safe_round(
            sum(detection_confidences) / len(detection_confidences)
        )
        if detection_confidences
        else 0.0,
        "track_count": len(tracks),
        "fragmentation_count": max(0, len(nontrivial_tracks) - 1),
        "obvious_false_positive_count_manual": "",
    }
    if main_track is None or main_track.quality is None:
        row.update(
            {
                "main_track_id": "",
                "main_track_num_points": 0,
                "main_track_missing_ratio": 1.0,
                "main_track_mean_conf": 0.0,
                "main_track_stability": "poor",
            }
        )
    else:
        quality = main_track.quality
        row.update(
            {
                "main_track_id": main_track.track_id,
                "main_track_num_points": quality.num_points,
                "main_track_missing_ratio": quality.missing_ratio,
                "main_track_mean_conf": quality.mean_conf,
                "main_track_stability": quality.track_stability,
            }
        )
    row["qualitative_label_auto"] = qualitative_label_for_row(row)
    row["failure_tags_auto"] = "|".join(failure_tags_for_row(row, case.condition_tags))
    row["manual_note"] = ""
    return row


def summarize_failure(
    *,
    case: EvaluationVideoCase,
    detector_name: str,
    tracker_name: str,
    stabilization: StabilizationInfo,
    error: str,
    start_sec: float = 0.0,
) -> dict[str, Any]:
    return summarize_run(
        case=case,
        detector_name=detector_name,
        tracker_name=tracker_name,
        stabilization=stabilization,
        frame_detections=[],
        tracks=[],
        start_sec=start_sec,
        status="error",
        error=error,
    )


def select_main_track(tracks: Sequence[TrackSequence]) -> TrackSequence | None:
    if not tracks:
        return None
    return max(
        tracks,
        key=lambda track: (
            track.quality.num_points if track.quality else 0,
            track.quality.mean_conf if track.quality else 0.0,
        ),
    )


def qualitative_label_for_row(row: dict[str, Any]) -> str:
    if row["status"] != "ok":
        return "fail"
    if (
        row["main_track_num_points"] >= 60
        and row["main_track_missing_ratio"] < 0.10
        and row["main_track_mean_conf"] >= 0.50
        and row["track_count"] <= 2
    ):
        return "pass"
    if (
        row["main_track_num_points"] >= 30
        and row["main_track_missing_ratio"] <= 0.30
        and row["main_track_mean_conf"] >= 0.30
        and row["track_count"] <= 5
    ):
        return "borderline"
    return "fail"


def failure_tags_for_row(row: dict[str, Any], condition_tags: Sequence[str]) -> list[str]:
    tags: list[str] = []
    if row["status"] != "ok":
        tags.append("setup_error")
    if row["total_detections"] == 0:
        tags.append("no_detection")
    if row["fragmentation_count"] > 0 or row["track_count"] > 2:
        tags.append("fragmented")
    tag_set = {tag.lower() for tag in condition_tags}
    if "far" in tag_set:
        tags.append("small_object")
        tags.append("far")
    if "shake" in tag_set:
        tags.append("shake")
    if "zoom" in tag_set:
        tags.append("zoom")
    if "bg" in tag_set or "partial_bg" in tag_set:
        tags.append("complex_bg")
    if {"nonstab", "unstab", "unfix", "unstable_camera"} & tag_set:
        tags.append("unstable_camera")
    return _dedupe(tags)


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_track_visualization(
    *,
    input_video_path: str,
    output_video_path: Path,
    tracks: Sequence[TrackSequence],
    max_frames: int | None = None,
    start_sec: float = 0.0,
) -> None:
    import cv2

    metadata = load_video_metadata(input_video_path)
    start_frame = _start_frame_for_video(input_video_path, start_sec)
    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video for visualization: {input_video_path}")
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    points_by_frame = _points_by_frame(tracks)
    try:
        frame_index = start_frame
        written_frames = 0
        while True:
            if max_frames is not None and written_frames >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            for track_id, point in points_by_frame.get(frame_index, []):
                _draw_track_point(frame, track_id, point, metadata)
            writer.write(frame)
            frame_index += 1
            written_frames += 1
    finally:
        capture.release()
        writer.release()


def sanitize_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "run"


def _ultralytics_results_to_detections(results: Sequence[Any]) -> list[Detection]:
    detections: list[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            left, top, right, bottom = xyxy
            detections.append(
                Detection(
                    left=max(0, int(left)),
                    top=max(0, int(top)),
                    width=max(1, int(right - left)),
                    height=max(1, int(bottom - top)),
                    confidence=conf,
                )
            )
    return detections


def _build_yolomg_fd5_mask_frame(
    *,
    older_frame: Any,
    center_frame: Any,
    newer_frame: Any,
    cv2: Any,
) -> Any:
    """Build a three-channel motion mask similar to YOLOMG's FD5_mask.py.

    YOLOMG trains on an RGB frame plus a second image loaded from the mask32
    directory. That mask image is saved as grayscale but read by cv2 as BGR in
    the original dualdetector path, so we return a BGR image here.
    """

    older_gray = _blurred_gray(older_frame, cv2)
    center_gray = _blurred_gray(center_frame, cv2)
    newer_gray = _blurred_gray(newer_frame, cv2)

    compensated_older = _motion_compensate_to_reference(
        moving_gray=older_gray,
        reference_gray=center_gray,
        cv2=cv2,
    )
    compensated_newer = _motion_compensate_to_reference(
        moving_gray=newer_gray,
        reference_gray=center_gray,
        cv2=cv2,
    )

    diff_older = cv2.absdiff(center_gray, compensated_older)
    diff_newer = cv2.absdiff(center_gray, compensated_newer)
    frame_diff = ((diff_older.astype(np.float32) + diff_newer.astype(np.float32)) / 2).astype(
        np.uint8
    )
    return cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)


def _blurred_gray(frame: Any, cv2: Any) -> Any:
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


def _motion_compensate_to_reference(
    *,
    moving_gray: Any,
    reference_gray: Any,
    cv2: Any,
) -> Any:
    height, width = reference_gray.shape[:2]
    homography = _estimate_grid_homography(
        moving_gray=moving_gray,
        reference_gray=reference_gray,
        cv2=cv2,
    )
    return cv2.warpPerspective(
        moving_gray,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )


def _estimate_grid_homography(
    *,
    moving_gray: Any,
    reference_gray: Any,
    cv2: Any,
) -> Any:
    height, width = reference_gray.shape[:2]
    grid_width = 1920
    grid_height = 1080
    moving_grid = cv2.resize(moving_gray, (grid_width, grid_height), interpolation=cv2.INTER_CUBIC)
    reference_grid = cv2.resize(
        reference_gray,
        (grid_width, grid_height),
        interpolation=cv2.INTER_CUBIC,
    )

    points = _grid_points(grid_width=grid_width, grid_height=grid_height)
    if len(points) < 4:
        return np.eye(3, dtype=np.float32)

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
    )
    tracked_points, status, _err = cv2.calcOpticalFlowPyrLK(
        moving_grid,
        reference_grid,
        points.reshape(-1, 1, 2),
        None,
        **lk_params,
    )
    if tracked_points is None or status is None:
        return np.eye(3, dtype=np.float32)

    old_points = points.reshape(-1, 2)[status.reshape(-1) == 1]
    new_points = tracked_points.reshape(-1, 2)[status.reshape(-1) == 1]
    if len(old_points) < 15 or len(new_points) < 15:
        return np.eye(3, dtype=np.float32)

    distances = np.linalg.norm(new_points - old_points, axis=1)
    keep = distances <= 50
    old_points = old_points[keep]
    new_points = new_points[keep]
    if len(old_points) < 15 or len(new_points) < 15:
        return np.eye(3, dtype=np.float32)

    homography_grid, _status = cv2.findHomography(new_points, old_points, cv2.RANSAC, 3.0)
    if homography_grid is None:
        return np.eye(3, dtype=np.float32)

    scale_to_grid = np.array(
        [
            [grid_width / width, 0, 0],
            [0, grid_height / height, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    scale_to_original = np.linalg.inv(scale_to_grid)
    return scale_to_original @ homography_grid @ scale_to_grid


def _grid_points(*, grid_width: int, grid_height: int) -> Any:
    grid_size_w = 32 * 2
    grid_size_h = 24 * 2
    grid_num_w = int(grid_width / grid_size_w - 1)
    grid_num_h = int(grid_height / grid_size_h - 1)
    points = [
        (np.float32(i * grid_size_w + grid_size_w / 2.0), np.float32(j * grid_size_h + grid_size_h / 2.0))
        for i in range(grid_num_w)
        for j in range(grid_num_h)
    ]
    return np.array(points, dtype=np.float32)


def _existing_path(value: str | None, label: str) -> Path:
    if not value:
        raise DetectorUnavailableError(
            f"{label} path is not configured. Pass the matching CLI flag or set "
            "YOLOMG_REPO/YOLOMG_WEIGHTS."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise DetectorUnavailableError(f"{label} path does not exist: {path}")
    return path


def _greedy_iou_matches(
    *,
    tracks: Sequence[_SortTrackState],
    active_indices: Sequence[int],
    detections: Sequence[Detection],
    frame_index: int,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for track_index in active_indices:
        predicted = tracks[track_index].predicted_bbox(frame_index)
        for detection_index, detection in enumerate(detections):
            iou = _bbox_iou(predicted, _bbox_from_detection(detection))
            if iou >= iou_threshold:
                candidates.append((iou, track_index, detection_index))
    candidates.sort(reverse=True)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for _iou, track_index, detection_index in candidates:
        if track_index in matched_tracks or detection_index in matched_detections:
            continue
        matches.append((track_index, detection_index))
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)
    return matches


def _track_point_from_detection(
    detection: Detection,
    frame_index: int,
    timestamp_ms: int,
    metadata: VideoMetadata,
) -> TrackPoint:
    return TrackPoint(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        cx=(detection.left + (detection.width / 2)) / metadata.width,
        cy=(detection.top + (detection.height / 2)) / metadata.height,
        w=detection.width / metadata.width,
        h=detection.height / metadata.height,
        conf=detection.confidence,
    )


def _bbox_from_detection(detection: Detection) -> tuple[int, int, int, int]:
    return (detection.left, detection.top, detection.width, detection.height)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (aw * ah) + (bw * bh) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _center_x(bbox: tuple[int, int, int, int]) -> float:
    left, _top, width, _height = bbox
    return left + (width / 2)


def _center_y(bbox: tuple[int, int, int, int]) -> float:
    _left, top, _width, height = bbox
    return top + (height / 2)


def _safe_round(value: float) -> float:
    return round(value, 3)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _points_by_frame(
    tracks: Sequence[TrackSequence],
) -> dict[int, list[tuple[int, TrackPoint]]]:
    points: dict[int, list[tuple[int, TrackPoint]]] = {}
    for track in tracks:
        for point in track.history:
            points.setdefault(point.frame_index, []).append((track.track_id, point))
    return points


def _draw_track_point(
    frame: Any,
    track_id: int,
    point: TrackPoint,
    metadata: VideoMetadata,
) -> None:
    import cv2

    left = round((point.cx - point.w / 2) * metadata.width)
    top = round((point.cy - point.h / 2) * metadata.height)
    right = round((point.cx + point.w / 2) * metadata.width)
    bottom = round((point.cy + point.h / 2) * metadata.height)
    color = _track_color(track_id)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(
        frame,
        f"id={track_id} conf={point.conf:.2f}",
        (left, max(12, top - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = (
        (0, 255, 0),
        (255, 180, 0),
        (0, 200, 255),
        (255, 0, 180),
        (180, 255, 0),
    )
    return palette[track_id % len(palette)]
