"""Evaluate manual-ROI tracking output against CVAT video annotations.

The evaluator converts every input to source-video pixel coordinates before
scoring. It never aligns trajectories from their observed min/max values,
because that would hide scale, offset, and early-termination errors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import json
from math import hypot
from pathlib import Path
from typing import Any

from defusedxml import ElementTree
import numpy as np

from .io import write_json


class TrackingEvaluationError(ValueError):
    """Raised when an input cannot be compared without guessing."""


@dataclass(frozen=True, slots=True)
class GroundTruthObservation:
    frame_index: int
    timestamp_ms: int
    cx: float
    cy: float
    width: float
    height: float
    occluded: bool
    keyframe: bool


@dataclass(frozen=True, slots=True)
class PredictionObservation:
    frame_index: int
    timestamp_ms: int
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    tracking_source: str


@dataclass(frozen=True, slots=True)
class CvatTrack:
    track_id: int
    source_track_ids: tuple[int, ...]
    label: str
    source_width: int
    source_height: int
    task_frame_count: int | None
    observations: tuple[GroundTruthObservation, ...]


@dataclass(frozen=True, slots=True)
class VideoGeometry:
    source_width: int
    source_height: int
    processed_width: int
    processed_height: int
    fps: float
    frame_count: int
    num_frames_processed: int
    original_to_processed: np.ndarray
    transform_source: str

    @property
    def processed_to_original(self) -> np.ndarray:
        try:
            return np.linalg.inv(self.original_to_processed)
        except np.linalg.LinAlgError as error:
            raise TrackingEvaluationError(
                "original_to_processed transform is not invertible"
            ) from error


def load_cvat_track(
    path: str | Path,
    *,
    fps: float,
    track_id: int | None = None,
    track_ids: tuple[int, ...] | None = None,
    label: str | None = None,
    include_occluded: bool = True,
) -> CvatTrack:
    """Load one rectangle track from a CVAT for video 1.1 XML export."""
    xml_path = Path(path)
    root = ElementTree.parse(xml_path).getroot()
    size = root.find("./meta/task/original_size")
    if size is None:
        raise TrackingEvaluationError("CVAT XML is missing meta/task/original_size")
    source_width = _positive_int(_required_text(size, "width"), "CVAT width")
    source_height = _positive_int(_required_text(size, "height"), "CVAT height")

    task_size_node = root.find("./meta/task/size")
    task_frame_count = (
        _positive_int(task_size_node.text, "CVAT task size")
        if task_size_node is not None and task_size_node.text
        else None
    )
    if track_id is not None and track_ids is not None:
        raise TrackingEvaluationError("use either track_id or track_ids, not both")
    requested_ids = track_ids or ((track_id,) if track_id is not None else None)
    if requested_ids is not None and len(set(requested_ids)) != len(requested_ids):
        raise TrackingEvaluationError("track_ids must not contain duplicates")

    all_tracks = list(root.findall("./track"))
    candidates = all_tracks
    if requested_ids is not None:
        requested = set(requested_ids)
        candidates = [item for item in candidates if int(item.attrib["id"]) in requested]
        found = {int(item.attrib["id"]) for item in candidates}
        if found != requested:
            raise TrackingEvaluationError(
                f"CVAT tracks not found: {sorted(requested - found)}"
            )
    if label is not None:
        candidates = [item for item in candidates if item.attrib.get("label") == label]
    allow_merge = requested_ids is not None and len(requested_ids) > 1
    if not candidates or (len(candidates) != 1 and not allow_merge):
        available = [
            {"track_id": int(item.attrib["id"]), "label": item.attrib.get("label")}
            for item in all_tracks
        ]
        raise TrackingEvaluationError(
            "CVAT selection must resolve to one track unless explicit track_ids are given; "
            f"matched={len(candidates)}, available={available}"
        )
    labels = {str(item.attrib.get("label") or "unknown") for item in candidates}
    if len(labels) != 1:
        raise TrackingEvaluationError(
            f"explicitly merged CVAT tracks must share one label: {sorted(labels)}"
        )

    observations: list[GroundTruthObservation] = []
    seen_frames: set[int] = set()
    for track in candidates:
        for box in track.findall("./box"):
            if _xml_bool(box.attrib.get("outside", "0")):
                continue
            occluded = _xml_bool(box.attrib.get("occluded", "0"))
            if occluded and not include_occluded:
                continue
            frame_index = int(box.attrib["frame"])
            if frame_index in seen_frames:
                raise TrackingEvaluationError(
                    f"merged CVAT tracks overlap at frame {frame_index}"
                )
            seen_frames.add(frame_index)
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            if xbr <= xtl or ybr <= ytl:
                raise TrackingEvaluationError(f"invalid CVAT box at frame {frame_index}")
            observations.append(
                GroundTruthObservation(
                    frame_index=frame_index,
                    timestamp_ms=_timestamp_ms(frame_index, fps),
                    cx=(xtl + xbr) / 2.0,
                    cy=(ytl + ybr) / 2.0,
                    width=xbr - xtl,
                    height=ybr - ytl,
                    occluded=occluded,
                    keyframe=_xml_bool(box.attrib.get("keyframe", "0")),
                )
            )
    observations.sort(key=lambda item: item.frame_index)
    if not observations:
        raise TrackingEvaluationError("selected CVAT track has no evaluable boxes")
    return CvatTrack(
        track_id=min(int(track.attrib["id"]) for track in candidates),
        source_track_ids=tuple(sorted(int(track.attrib["id"]) for track in candidates)),
        label=next(iter(labels)),
        source_width=source_width,
        source_height=source_height,
        task_frame_count=task_frame_count,
        observations=tuple(observations),
    )


def load_video_geometry(path: str | Path) -> VideoGeometry:
    metadata_path = Path(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_width = _metadata_dimension(metadata, "source_width", "width")
    source_height = _metadata_dimension(metadata, "source_height", "height")
    processed_width = _metadata_dimension(metadata, "processed_width")
    processed_height = _metadata_dimension(metadata, "processed_height")
    fps = float(metadata.get("fps") or 0.0)
    frame_count = int(metadata.get("frame_count") or 0)
    if fps <= 0 or frame_count <= 0:
        raise TrackingEvaluationError("metadata fps and frame_count must be positive")
    num_frames_processed = int(metadata.get("num_frames_processed") or frame_count)
    if num_frames_processed <= 0 or num_frames_processed > frame_count:
        raise TrackingEvaluationError(
            "metadata num_frames_processed must be between 1 and frame_count"
        )

    explicit = metadata.get("original_to_processed")
    if explicit is None and isinstance(metadata.get("coordinate_transform"), dict):
        explicit = metadata["coordinate_transform"].get("original_to_processed")
    if explicit is not None:
        matrix = np.asarray(explicit, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise TrackingEvaluationError(
                "original_to_processed must be a finite 3x3 matrix"
            )
        transform_source = "explicit_matrix"
    else:
        mode = str((metadata.get("preprocessing") or {}).get("mode") or "legacy")
        source_ratio = source_width / source_height
        processed_ratio = processed_width / processed_height
        if mode not in {"legacy", "none", "resize"}:
            raise TrackingEvaluationError(
                f"preprocessing mode {mode!r} requires original_to_processed"
            )
        if mode == "legacy" and not np.isclose(
            source_ratio, processed_ratio, rtol=1e-3, atol=1e-3
        ):
            raise TrackingEvaluationError(
                "legacy metadata changes aspect ratio; an explicit transform is required"
            )
        matrix = np.array(
            [
                [processed_width / source_width, 0.0, 0.0],
                [0.0, processed_height / source_height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        transform_source = "declared_resize" if mode != "legacy" else "legacy_dimensions"
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise TrackingEvaluationError("original_to_processed transform is singular")
    return VideoGeometry(
        source_width=source_width,
        source_height=source_height,
        processed_width=processed_width,
        processed_height=processed_height,
        fps=fps,
        frame_count=frame_count,
        num_frames_processed=num_frames_processed,
        original_to_processed=matrix,
        transform_source=transform_source,
    )


def load_trajectory(
    path: str | Path,
    *,
    geometry: VideoGeometry,
) -> tuple[PredictionObservation, ...]:
    """Load visible raw tracker observations and map them to source pixels."""
    trajectory_path = Path(path)
    with trajectory_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "timestamp_ms", "raw_x", "raw_y", "visible"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise TrackingEvaluationError(
                f"trajectory CSV is missing columns: {sorted(missing)}"
            )
        rows = list(reader)

    inverse = geometry.processed_to_original
    observations: list[PredictionObservation] = []
    seen_frames: set[int] = set()
    for row in rows:
        if not _csv_bool(row["visible"]):
            continue
        frame_index = int(row["frame_index"])
        if frame_index in seen_frames:
            raise TrackingEvaluationError(f"duplicate visible prediction at frame {frame_index}")
        seen_frames.add(frame_index)
        raw_x = float(row["raw_x"])
        raw_y = float(row["raw_y"])
        bbox = _prediction_bbox(row, raw_x=raw_x, raw_y=raw_y)
        cx, cy = _transform_point(inverse, raw_x, raw_y)
        source_box = _transform_box(inverse, bbox)
        observations.append(
            PredictionObservation(
                frame_index=frame_index,
                timestamp_ms=int(round(float(row["timestamp_ms"]))),
                cx=cx,
                cy=cy,
                width=source_box[2],
                height=source_box[3],
                confidence=float(row.get("confidence") or 0.0),
                tracking_source=str(row.get("tracking_source") or "unknown"),
            )
        )
    observations.sort(key=lambda item: item.frame_index)
    if not observations:
        raise TrackingEvaluationError("trajectory has no visible observations")
    return tuple(observations)


def evaluate_track(
    ground_truth: CvatTrack,
    predictions: tuple[PredictionObservation, ...],
    *,
    fps: float,
    sample_id: str,
    prediction_frame_offset: int = 0,
    attempted_frame_count: int | None = None,
    pixel_thresholds: tuple[float, ...] = (5.0, 10.0, 20.0),
    normalized_thresholds: tuple[float, ...] = (0.25, 0.5, 1.0),
) -> dict[str, Any]:
    """Score a prediction without estimating spatial or temporal alignment."""
    gt_by_frame = {item.frame_index: item for item in ground_truth.observations}
    pred_by_frame: dict[int, PredictionObservation] = {}
    for item in predictions:
        aligned_frame = item.frame_index + prediction_frame_offset
        if aligned_frame in pred_by_frame:
            raise TrackingEvaluationError(
                f"frame offset creates duplicate prediction frame {aligned_frame}"
            )
        pred_by_frame[aligned_frame] = item

    gt_frames = set(gt_by_frame)
    prediction_frames = set(pred_by_frame)
    matched_frames = sorted(gt_frames & prediction_frames)
    missing_frames = sorted(gt_frames - prediction_frames)
    extra_frames = sorted(prediction_frames - gt_frames)

    errors_px: list[float] = []
    normalized_errors: list[float] = []
    frame_scores: list[dict[str, Any]] = []
    for frame_index in matched_frames:
        gt = gt_by_frame[frame_index]
        prediction = pred_by_frame[frame_index]
        error_px = hypot(prediction.cx - gt.cx, prediction.cy - gt.cy)
        diagonal = hypot(gt.width, gt.height)
        normalized_error = error_px / diagonal
        errors_px.append(error_px)
        normalized_errors.append(normalized_error)
        frame_scores.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": gt.timestamp_ms,
                "error_px": error_px,
                "error_bbox_diagonal": normalized_error,
                "gt_center": [gt.cx, gt.cy],
                "prediction_center": [prediction.cx, prediction.cy],
                "confidence": prediction.confidence,
                "tracking_source": prediction.tracking_source,
            }
        )

    gt_count = len(gt_frames)
    matched_count = len(matched_frames)
    prediction_count = len(prediction_frames)
    first_prediction = min(prediction_frames) if prediction_frames else None
    last_prediction = max(prediction_frames) if prediction_frames else None
    active_span_gt_frames = (
        {
            frame
            for frame in gt_frames
            if first_prediction is not None
            and last_prediction is not None
            and first_prediction <= frame <= last_prediction
        }
        if prediction_frames
        else set()
    )
    attempted_start = prediction_frame_offset
    attempted_end = (
        attempted_start + attempted_frame_count - 1
        if attempted_frame_count is not None
        else None
    )
    attempted_gt_frames = (
        {
            frame
            for frame in gt_frames
            if attempted_end is not None and attempted_start <= frame <= attempted_end
        }
        if attempted_frame_count is not None
        else set()
    )
    pixel_success = {
        _threshold_key(value, "px"): _success_metrics(errors_px, value, gt_count)
        for value in pixel_thresholds
    }
    normalized_success = {
        _threshold_key(value, "bbox_diagonal"): _success_metrics(
            normalized_errors, value, gt_count
        )
        for value in normalized_thresholds
    }
    first_gt = min(gt_frames)
    last_gt = max(gt_frames)
    late_start_frames = (
        max(0, first_prediction - first_gt) if first_prediction is not None else gt_count
    )
    early_termination_frames = (
        max(0, last_gt - last_prediction) if last_prediction is not None else gt_count
    )
    return {
        "sample_id": sample_id,
        "status": "completed",
        "target": {
            "track_id": ground_truth.track_id,
            "source_track_ids": list(ground_truth.source_track_ids),
            "label": ground_truth.label,
        },
        "coordinate_space": "source_video_pixels",
        "alignment": {
            "mode": "frame_index",
            "prediction_frame_offset": prediction_frame_offset,
            "estimated_alignment_used": False,
        },
        "counts": {
            "gt_visible_frames": gt_count,
            "prediction_visible_frames": prediction_count,
            "matched_visible_frames": matched_count,
            "missing_gt_frames": len(missing_frames),
            "prediction_frames_without_gt": len(extra_frames),
        },
        "coverage": {
            "full_gt_observation_ratio": matched_count / gt_count,
            "missing_ratio": len(missing_frames) / gt_count,
            "active_span_gt_frames": len(active_span_gt_frames),
            "active_span_observation_ratio": (
                len(active_span_gt_frames & prediction_frames) / len(active_span_gt_frames)
                if active_span_gt_frames
                else None
            ),
            "attempted_window_gt_frames": len(attempted_gt_frames),
            "attempted_window_observation_ratio": (
                len(attempted_gt_frames & prediction_frames) / len(attempted_gt_frames)
                if attempted_gt_frames
                else None
            ),
        },
        "timeline": {
            "gt_first_frame": first_gt,
            "gt_last_frame": last_gt,
            "prediction_first_frame": first_prediction,
            "prediction_last_frame": last_prediction,
            "attempted_first_frame": attempted_start,
            "attempted_last_frame": attempted_end,
            "late_start_frames": late_start_frames,
            "late_start_ms": round(late_start_frames / fps * 1000.0),
            "early_termination_frames": early_termination_frames,
            "early_termination_ms": round(early_termination_frames / fps * 1000.0),
            "longest_missing_run_frames": _longest_contiguous_run(missing_frames),
            "missing_frame_ranges": _compact_ranges(missing_frames),
            "extra_frame_ranges": _compact_ranges(extra_frames),
        },
        "localization": {
            "observed_frame_error_px": _distribution(errors_px),
            "observed_frame_error_bbox_diagonal": _distribution(normalized_errors),
            "pixel_threshold_success": pixel_success,
            "normalized_threshold_success": normalized_success,
        },
        "frames": frame_scores,
    }


def evaluate_files(
    *,
    cvat_xml: str | Path,
    trajectory_csv: str | Path,
    metadata_json: str | Path,
    sample_id: str,
    track_id: int | None = None,
    track_ids: tuple[int, ...] | None = None,
    label: str | None = None,
    include_occluded: bool = True,
    prediction_frame_offset: int = 0,
    source_is_trimmed: bool = False,
) -> dict[str, Any]:
    geometry = load_video_geometry(metadata_json)
    ground_truth = load_cvat_track(
        cvat_xml,
        fps=geometry.fps,
        track_id=track_id,
        track_ids=track_ids,
        label=label,
        include_occluded=include_occluded,
    )
    if (
        ground_truth.source_width != geometry.source_width
        or ground_truth.source_height != geometry.source_height
    ):
        raise TrackingEvaluationError(
            "CVAT original size does not match tracker metadata: "
            f"CVAT={ground_truth.source_width}x{ground_truth.source_height}, "
            f"metadata={geometry.source_width}x{geometry.source_height}"
        )
    if ground_truth.task_frame_count is not None:
        frame_counts_match = ground_truth.task_frame_count == geometry.frame_count
        if not frame_counts_match and not source_is_trimmed:
            raise TrackingEvaluationError(
                "CVAT task frame count does not match tracker metadata. "
                "Declare source_is_trimmed and an explicit frame offset only when "
                "the A input is a verified trim of the CVAT source: "
                f"CVAT={ground_truth.task_frame_count}, metadata={geometry.frame_count}"
            )
        mapped_source_end = prediction_frame_offset + geometry.frame_count - 1
        if source_is_trimmed and (
            prediction_frame_offset < 0
            or mapped_source_end >= ground_truth.task_frame_count
        ):
            raise TrackingEvaluationError(
                "trimmed A source falls outside the CVAT task frame range"
            )
    predictions = load_trajectory(trajectory_csv, geometry=geometry)
    report = evaluate_track(
        ground_truth,
        predictions,
        fps=geometry.fps,
        sample_id=sample_id,
        prediction_frame_offset=prediction_frame_offset,
        attempted_frame_count=geometry.num_frames_processed,
    )
    report["inputs"] = {
        "cvat_xml": str(Path(cvat_xml)),
        "trajectory_csv": str(Path(trajectory_csv)),
        "metadata_json": str(Path(metadata_json)),
    }
    report["video"] = {
        "fps": geometry.fps,
        "frame_count": geometry.frame_count,
        "num_frames_processed": geometry.num_frames_processed,
        "source_width": geometry.source_width,
        "source_height": geometry.source_height,
        "processed_width": geometry.processed_width,
        "processed_height": geometry.processed_height,
        "transform_source": geometry.transform_source,
        "original_to_processed": geometry.original_to_processed.tolist(),
    }
    return report


def evaluate_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    ground_truth = manifest.get("ground_truth") or {}
    prediction = manifest.get("prediction") or {}
    alignment = manifest.get("alignment") or {}
    track_ids_value = ground_truth.get("track_ids")
    if track_ids_value is not None and not isinstance(track_ids_value, list):
        raise TrackingEvaluationError("ground_truth.track_ids must be a JSON list")
    return evaluate_files(
        cvat_xml=_relative_path(root, ground_truth.get("path"), "ground_truth.path"),
        trajectory_csv=_relative_path(
            root, prediction.get("trajectory"), "prediction.trajectory"
        ),
        metadata_json=_relative_path(
            root, prediction.get("metadata"), "prediction.metadata"
        ),
        sample_id=str(manifest.get("sample_id") or root.name),
        track_id=(
            int(ground_truth["track_id"])
            if ground_truth.get("track_id") is not None
            else None
        ),
        track_ids=(
            tuple(int(value) for value in track_ids_value)
            if track_ids_value is not None
            else None
        ),
        label=ground_truth.get("label"),
        include_occluded=bool(ground_truth.get("include_occluded", True)),
        prediction_frame_offset=int(alignment.get("prediction_frame_offset", 0)),
        source_is_trimmed=bool(alignment.get("source_is_trimmed", False)),
    )


def evaluate_dataset(root: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    dataset_root = Path(root)
    manifests = sorted(dataset_root.rglob("sample.json"))
    if not manifests:
        raise TrackingEvaluationError(f"no sample.json files found under {dataset_root}")
    output_root = Path(output_dir)
    reports = [evaluate_manifest(path) for path in manifests]
    for report in reports:
        write_json(output_root / "per_video" / f"{report['sample_id']}.json", report)
    _write_summary_csv(output_root / "summary.csv", reports)
    _write_html_report(output_root / "report.html", reports)
    write_json(
        output_root / "summary.json",
        {"status": "completed", "sample_count": len(reports), "samples": reports},
    )
    return reports


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _success_metrics(values: list[float], threshold: float, gt_count: int) -> dict[str, Any]:
    passed = sum(value <= threshold for value in values)
    return {
        "threshold": threshold,
        "passed_frames": passed,
        "observed_frame_ratio": passed / len(values) if values else None,
        "full_gt_ratio": passed / gt_count,
    }


def _prediction_bbox(
    row: dict[str, str], *, raw_x: float, raw_y: float
) -> tuple[float, float, float, float]:
    width = float(row.get("bbox_width") or row.get("w") or 0.0)
    height = float(row.get("bbox_height") or row.get("h") or 0.0)
    if width <= 0 or height <= 0:
        raise TrackingEvaluationError("visible trajectory rows require a positive bbox size")
    x = float(row.get("bbox_x") or raw_x - width / 2.0)
    y = float(row.get("bbox_y") or raw_y - height / 2.0)
    return x, y, width, height


def _transform_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    transformed = matrix @ np.array([x, y, 1.0], dtype=float)
    if abs(float(transformed[2])) < 1e-12:
        raise TrackingEvaluationError("coordinate transform produced a point at infinity")
    return float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2])


def _transform_box(
    matrix: np.ndarray, box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x, y, width, height = box
    points = [
        _transform_point(matrix, px, py)
        for px, py in ((x, y), (x + width, y), (x, y + height), (x + width, y + height))
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _metadata_dimension(metadata: dict[str, Any], *names: str) -> int:
    for name in names:
        if metadata.get(name) is not None:
            return _positive_int(metadata[name], f"metadata {name}")
    raise TrackingEvaluationError(f"metadata is missing one of {names}")


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise TrackingEvaluationError(f"{name} must be positive")
    return result


def _required_text(parent: Any, tag: str) -> str:
    node = parent.find(tag)
    if node is None or node.text is None:
        raise TrackingEvaluationError(f"CVAT XML is missing {tag}")
    return node.text


def _xml_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _csv_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise TrackingEvaluationError(f"invalid CSV boolean: {value!r}")


def _timestamp_ms(frame_index: int, fps: float) -> int:
    if fps <= 0:
        raise TrackingEvaluationError("fps must be positive")
    return int(round(frame_index / fps * 1000.0))


def _threshold_key(value: float, suffix: str) -> str:
    rendered = str(int(value)) if float(value).is_integer() else str(value).replace(".", "_")
    return f"within_{rendered}_{suffix}"


def _longest_contiguous_run(frames: list[int]) -> int:
    longest = current = 0
    previous: int | None = None
    for frame in frames:
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    return longest


def _compact_ranges(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    ranges: list[list[int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            ranges.append([start, previous])
            start = frame
        previous = frame
    ranges.append([start, previous])
    return ranges


def _relative_path(root: Path, value: Any, field: str) -> Path:
    if not value:
        raise TrackingEvaluationError(f"manifest is missing {field}")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _summary_row(report: dict[str, Any]) -> dict[str, Any]:
    error = report["localization"]["observed_frame_error_px"]
    normalized = report["localization"]["observed_frame_error_bbox_diagonal"]
    pixel_success = report["localization"]["pixel_threshold_success"]
    coverage = report["coverage"]
    return {
        "sample_id": report["sample_id"],
        "label": report["target"]["label"],
        "gt_visible_frames": report["counts"]["gt_visible_frames"],
        "matched_visible_frames": report["counts"]["matched_visible_frames"],
        "full_gt_observation_ratio": coverage["full_gt_observation_ratio"],
        "active_span_observation_ratio": coverage["active_span_observation_ratio"],
        "attempted_window_observation_ratio": coverage[
            "attempted_window_observation_ratio"
        ],
        "median_error_px": error["median"],
        "p95_error_px": error["p95"],
        "median_error_bbox_diagonal": normalized["median"],
        "within_5px_full_gt_ratio": pixel_success["within_5_px"]["full_gt_ratio"],
        "within_10px_full_gt_ratio": pixel_success["within_10_px"]["full_gt_ratio"],
        "longest_missing_run_frames": report["timeline"]["longest_missing_run_frames"],
        "early_termination_frames": report["timeline"]["early_termination_frames"],
    }


def _write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    rows = [_summary_row(report) for report in reports]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_html_report(path: Path, reports: list[dict[str, Any]]) -> None:
    rows = [_summary_row(report) for report in reports]
    headers = list(rows[0])
    table_header = "".join(f"<th>{escape(header)}</th>" for header in headers)
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(_render_cell(row[key]))}</td>" for key in headers) + "</tr>"
        for row in rows
    )
    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SkyDetect tracking evaluation</title>
  <style>
    :root {{ color-scheme: light; --ink: #18231d; --paper: #f3f0e5; --line: #b8b3a3; --accent: #bf4f2f; }}
    body {{ margin: 0; padding: 32px; background: linear-gradient(135deg, #f3f0e5, #dce7d7); color: var(--ink); font: 15px/1.5 Georgia, serif; }}
    main {{ max-width: 1400px; margin: auto; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 5vw, 52px); }}
    p {{ margin: 0 0 24px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); background: rgba(255,255,255,.72); box-shadow: 8px 8px 0 rgba(24,35,29,.12); }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: right; }}
    th {{ background: var(--ink); color: var(--paper); font-size: 12px; letter-spacing: .04em; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
  </style>
</head>
<body><main><h1>Tracking Evaluation</h1><p>{len(rows)}개 영상의 GT 대비 A 추적 결과</p>
<div class="table-wrap"><table><thead><tr>{table_header}</tr></thead><tbody>{table_rows}</tbody></table></div>
</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _render_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def _single_command(args: argparse.Namespace) -> None:
    report = evaluate_files(
        cvat_xml=args.gt,
        trajectory_csv=args.trajectory,
        metadata_json=args.metadata,
        sample_id=args.sample_id,
        track_ids=tuple(args.track_id) if args.track_id else None,
        label=args.label,
        include_occluded=not args.ignore_occluded,
        prediction_frame_offset=args.prediction_frame_offset,
        source_is_trimmed=args.source_is_trimmed,
    )
    write_json(args.output, report)
    print(f"completed: {report['sample_id']}")


def _dataset_command(args: argparse.Namespace) -> None:
    reports = evaluate_dataset(args.root, args.output)
    print(f"completed: {len(reports)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    single = subparsers.add_parser("single", help="Evaluate one CVAT/trajectory pair")
    single.add_argument("--gt", type=Path, required=True)
    single.add_argument("--trajectory", type=Path, required=True)
    single.add_argument("--metadata", type=Path, required=True)
    single.add_argument("--sample-id", required=True)
    single.add_argument(
        "--track-id",
        type=int,
        action="append",
        help="CVAT track ID; repeat to merge explicitly selected track fragments",
    )
    single.add_argument("--label")
    single.add_argument("--prediction-frame-offset", type=int, default=0)
    single.add_argument("--source-is-trimmed", action="store_true")
    single.add_argument("--ignore-occluded", action="store_true")
    single.add_argument("--output", type=Path, required=True)
    single.set_defaults(run=_single_command)

    dataset = subparsers.add_parser("dataset", help="Evaluate every sample.json")
    dataset.add_argument("root", type=Path)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.set_defaults(run=_dataset_command)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
