"""Compare GT with the actual Part B runtime under an explicit camera transform."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np

from ai_server.schemas import TrackPoint, TrackSequence
from ai_server.services.feature_core import build_feature_vector
from research.features import FEATURE_COLUMNS, FeatureConfig
from research.io import write_json
from research.tracking_evaluation import (
    TrackingEvaluationError, _csv_bool, _relative_path, _transform_point,
    load_cvat_track, load_video_geometry,
)

FEATURE_UNITS = {
    "speed_median": "canonical_px/s", "speed_cv": "ratio",
    "acceleration_median": "canonical_px/s^2", "acceleration_p95": "canonical_px/s^2",
    "turn_rate_median": "rad/s", "turn_rate_p95": "rad/s",
    "curvature_cv": "ratio", "tortuosity": "ratio", "heading_change_ratio": "ratio",
}
CMC_TOLERANCE_PX = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indices = [int(row["frame_index"]) for row in rows]
    if not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        raise TrackingEvaluationError("camera replay requires ordered, unique observations")
    return rows


def replay_camera_transforms(
    metadata_path: Path, debug_path: Path, output_path: Path,
    *, source_video: Path | None = None,
) -> dict[str, Any]:
    """Replay only background CMC, seeded by saved centers/sizes including LOST rows.

    No GT is used to estimate a transform. Every reconstructed transform must
    reproduce the saved compensated A point before it can be used for GT.
    """
    from ai_server.services.manual_roi_tracker import (
        _estimate_background_affine, _gray, _safe_inverse,
    )
    from ai_server.services.tracking_video_io import (
        affine_to_homography, iter_tracking_video_frames, read_tracking_video_metadata,
        transform_point,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    analysis_width = (metadata.get("tracking_parameters") or {}).get("cmc_analysis_width")
    if not source_video and not metadata.get("source_video_path"):
        raise TrackingEvaluationError("camera replay requires the original source video")
    video = source_video or Path(metadata["source_video_path"])
    geometry = load_video_geometry(metadata_path)
    observed = read_tracking_video_metadata(str(video), resize_width=geometry.processed_width)
    if (observed.width, observed.height, observed.frame_count) != (
        geometry.source_width, geometry.source_height, geometry.frame_count,
    ) or abs(observed.fps - geometry.fps) > 1e-5:
        raise TrackingEvaluationError("camera replay video does not match tracker metadata")
    expected_resize = np.diag([
        geometry.processed_width / geometry.source_width,
        geometry.processed_height / geometry.source_height, 1.0,
    ])
    if not np.allclose(geometry.original_to_processed, expected_resize):
        raise TrackingEvaluationError("camera replay only supports the tracker's declared resize")
    rows = _rows(debug_path)
    if any(int(b["frame_index"]) != int(a["frame_index"]) + 1 for a, b in zip(rows, rows[1:])):
        raise TrackingEvaluationError("camera replay needs all attempted debug rows, including gaps")
    if int(rows[0]["frame_index"]) != geometry.tracking_start_frame:
        raise TrackingEvaluationError("camera replay reference is not the selected ROI frame")
    by_frame = {int(row["frame_index"]): row for row in rows}
    tracker_path = Path(__file__).parents[1] / "ai_server/services/manual_roi_tracker.py"
    inputs = {
        "source_video": str(video), "source_video_sha256": _sha256(video),
        "metadata_sha256": _sha256(metadata_path), "debug_sha256": _sha256(debug_path),
        "tracker_sha256": _sha256(tracker_path), "opencv_version": cv2.__version__,
        "cmc_analysis_width": analysis_width,
    }
    if output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("inputs") == inputs:
            return cached
    cumulative = np.eye(3, dtype=np.float32)
    previous_gray = None
    frames, residuals = [], []
    for index, frame in iter_tracking_video_frames(
        video, resize_width=geometry.processed_width, max_frames=max(by_frame) + 1,
    ):
        if index not in by_frame:
            continue
        if frame.shape[:2] != (geometry.processed_height, geometry.processed_width):
            raise TrackingEvaluationError("camera replay processed dimensions do not match")
        gray = _gray(frame)
        if previous_gray is not None:
            previous = by_frame[index - 1]
            center = np.array([float(previous["raw_x"]), float(previous["raw_y"])], dtype=np.float32)
            affine, _ = _estimate_background_affine(
                previous_gray, gray, target_center=tuple(float(v) for v in center),
                target_size=(float(previous["bbox_width"]), float(previous["bbox_height"])),
                analysis_width=analysis_width,
            )
            cumulative = affine_to_homography(affine) @ cumulative
        matrix = _safe_inverse(cumulative)
        row = by_frame[index]
        compensated = transform_point(matrix, float(row["raw_x"]), float(row["raw_y"]))
        residuals.append(float(np.linalg.norm(np.subtract(
            compensated, [float(row["compensated_x"]), float(row["compensated_y"])],
        ))))
        frames.append({"frame_index": index, "current_to_reference": matrix.tolist()})
        previous_gray = gray
    if len(frames) != len(rows):
        raise TrackingEvaluationError("camera replay could not decode every attempted frame")
    maximum = max(residuals)
    if not np.isfinite(maximum) or maximum > CMC_TOLERANCE_PX:
        raise TrackingEvaluationError(
            f"camera replay differs from saved CMC coordinates by {maximum:.6f}px; "
            f"tolerance={CMC_TOLERANCE_PX}px; GT/B comparison refused"
        )
    result = {
        "method": "verified_background_cmc_replay", "coordinate_space": "processed_pixels",
        "reference_frame": geometry.tracking_start_frame, "inputs": inputs,
        "verification": {"point_count": len(residuals), "max_residual_px": maximum,
                         "tolerance_px": CMC_TOLERANCE_PX},
        "frames": frames,
    }
    write_json(output_path, result)
    return result


def _verified_matrices(
    payload: dict[str, Any], rows: list[dict[str, str]], reference_frame: int,
) -> tuple[dict[int, np.ndarray], float]:
    if payload.get("coordinate_space") != "processed_pixels" or payload.get("reference_frame") != reference_frame:
        raise TrackingEvaluationError("camera transforms have the wrong coordinate space/reference")
    matrices = {}
    for frame in payload["frames"]:
        index = int(frame["frame_index"])
        matrix = np.asarray(frame["current_to_reference"], dtype=np.float32)
        if index in matrices or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise TrackingEvaluationError("invalid/duplicate camera transform")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise TrackingEvaluationError("singular camera transform")
        matrices[index] = matrix
    residuals = []
    from ai_server.services.tracking_video_io import transform_point
    for row in rows:
        index = int(row["frame_index"])
        if index not in matrices:
            raise TrackingEvaluationError(f"camera transform missing at frame {index}")
        xy = transform_point(matrices[index], float(row["raw_x"]), float(row["raw_y"]))
        residuals.append(float(np.linalg.norm(np.subtract(
            xy, [float(row["compensated_x"]), float(row["compensated_y"])],
        ))))
    maximum = max(residuals)
    if not np.isfinite(maximum) or maximum > CMC_TOLERANCE_PX:
        raise TrackingEvaluationError("camera transforms do not reproduce saved A CMC points")
    return matrices, maximum


def _feature_result(track: TrackSequence) -> dict[str, Any]:
    result = build_feature_vector(track)
    return {
        "status": result.feature_vector.feature_status, "reasons": result.reasons,
        "feature_version": result.feature_version, "feature_config_id": result.feature_config_id,
        "coordinate_policy": result.coordinate_policy, "timebase_policy": result.timebase_policy,
        "canonical_width": result.canonical_width, "canonical_height": result.canonical_height,
        "features": result.feature_vector.features.model_dump() if result.feature_vector.features else None,
        "quality": result.raw_quality,
    }


def _comparison(a: dict[str, Any], gt: dict[str, Any]) -> list[dict[str, Any]]:
    if a["features"] is None or gt["features"] is None:
        return []
    rows = []
    for key in FEATURE_COLUMNS:
        av, gv = float(a["features"][key]), float(gt["features"][key])
        difference = av - gv
        rows.append({
            "feature": key, "unit": FEATURE_UNITS[key], "gt": gv, "a": av,
            "signed_difference": difference, "absolute_difference": abs(difference),
            "relative_difference_percent": difference / abs(gv) * 100.0 if abs(gv) > 1e-9 else None,
            "relative_status": "ok" if abs(gv) > 1e-9 else "undefined_gt_near_zero",
        })
    return rows


def compare_b_features(
    report: dict[str, Any], *, manifest: dict[str, Any], manifest_root: Path,
    output_dir: Path, replay_cmc: bool = False,
) -> dict[str, Any]:
    """Full GT vs matched GT isolates missingness from observed-center error."""
    try:
        return _compare_b_features(
            report, manifest=manifest, manifest_root=manifest_root,
            output_dir=output_dir, replay_cmc=replay_cmc,
        )
    except (TrackingEvaluationError, ValueError, OSError, KeyError) as error:
        return {"status": "not_comparable", "reasons": [str(error)], "comparisons": []}


def _compare_b_features(
    report: dict[str, Any], *, manifest: dict[str, Any], manifest_root: Path,
    output_dir: Path, replay_cmc: bool,
) -> dict[str, Any]:
    settings = manifest.get("b_comparison") or {}
    trajectory = Path(report["inputs"]["trajectory_csv"])
    metadata_path = Path(report["inputs"]["metadata_json"])
    geometry = load_video_geometry(metadata_path)
    track_path = (
        _relative_path(manifest_root, settings["track_sequence"], "b_comparison.track_sequence")
        if settings.get("track_sequence") else trajectory.parent / "track_sequence.json"
    )
    original = TrackSequence.model_validate_json(track_path.read_text(encoding="utf-8"))
    if (original.processed_width, original.processed_height) != (geometry.processed_width, geometry.processed_height):
        raise TrackingEvaluationError("TrackSequence dimensions do not match metadata")
    method = original.stabilization.method if original.stabilization else "none"
    if method not in {"none", "opencv_feature_cmc"}:
        raise TrackingEvaluationError(f"B comparison does not support stabilization method {method}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("coordinate_mode") not in {None, "camera_motion_compensated" if method != "none" else "frame"}:
        raise TrackingEvaluationError("TrackSequence/metadata coordinate modes disagree")
    raw_rows = [r for r in _rows(trajectory) if _csv_bool(r["visible"])]
    by_frame = {int(row["frame_index"]): row for row in raw_rows}
    if set(by_frame) != {p.frame_index for p in original.history}:
        raise TrackingEvaluationError("TrackSequence and trajectory visible frames differ")
    clamped_count = 0
    for point in original.history:
        row = by_frame[point.frame_index]
        x = float(row["compensated_x"] if method != "none" else row["raw_x"]) / geometry.processed_width
        y = float(row["compensated_y"] if method != "none" else row["raw_y"]) / geometry.processed_height
        clamped_count += int(not (0 <= x <= 1 and 0 <= y <= 1))
        expected = [np.clip(x, 0, 1), np.clip(y, 0, 1),
                    np.clip(float(row["bbox_width"]) / geometry.processed_width, 0, 1),
                    np.clip(float(row["bbox_height"]) / geometry.processed_height, 0, 1),
                    np.clip(float(row["confidence"]), 0, 1)]
        if point.timestamp_ms != int(round(float(row["timestamp_ms"]))) or not np.allclose(
            [point.cx, point.cy, point.w, point.h, point.conf], expected, rtol=0, atol=1e-8,
        ):
            raise TrackingEvaluationError("TrackSequence does not match the saved trajectory handoff")
    matrices, camera_verification = {}, {"method": "identity", "max_residual_px": 0.0}
    if method == "opencv_feature_cmc":
        if settings.get("camera_transforms"):
            transforms_path = _relative_path(manifest_root, settings["camera_transforms"], "b_comparison.camera_transforms")
            payload = json.loads(transforms_path.read_text(encoding="utf-8"))
        elif replay_cmc:
            debug = (
                _relative_path(manifest_root, settings["debug_observations"], "b_comparison.debug_observations")
                if settings.get("debug_observations") else trajectory.parent / "debug/observations.csv"
            )
            video = _relative_path(manifest_root, settings["source_video"], "b_comparison.source_video") if settings.get("source_video") else None
            payload = replay_camera_transforms(
                metadata_path, debug, output_dir / "camera_transforms.json", source_video=video,
            )
        else:
            raise TrackingEvaluationError("CMC GT comparison needs camera_transforms or --replay-cmc")
        matrices, maximum = _verified_matrices(payload, raw_rows, geometry.tracking_start_frame)
        camera_verification = {"method": payload.get("method", "provided_transforms"),
                               "max_residual_px": maximum, "tolerance_px": CMC_TOLERANCE_PX}
    gt = load_cvat_track(
        report["inputs"]["cvat_xml"], fps=geometry.fps,
        track_ids=tuple(report["target"]["source_track_ids"]),
        include_occluded=bool((manifest.get("ground_truth") or {}).get("include_occluded", True)),
    )
    start, end = report["timeline"]["attempted_first_frame"], report["timeline"]["attempted_last_frame"]
    offset = report["alignment"]["prediction_frame_offset"]
    gt_window = [g for g in gt.observations if start <= g.frame_index <= end]
    gt_local_ids = {g.frame_index - offset for g in gt_window}
    a_history = [p for p in original.history if p.frame_index in gt_local_ids]
    if not a_history:
        raise TrackingEvaluationError("no matched A points for B comparison")
    a_times = {p.frame_index: p.timestamp_ms for p in a_history}
    gt_history = []
    for point in gt_window:
        index = point.frame_index - offset
        x, y = _transform_point(geometry.original_to_processed, point.cx, point.cy)
        if method == "opencv_feature_cmc":
            if index not in matrices:
                raise TrackingEvaluationError(f"GT camera transform missing at frame {index}")
            from ai_server.services.tracking_video_io import transform_point
            x, y = transform_point(matrices[index], x, y)
        if not (0 <= x <= geometry.processed_width and 0 <= y <= geometry.processed_height):
            raise TrackingEvaluationError("GT CMC center lies outside normalized schema bounds; no silent clamp")
        sx = geometry.processed_width / geometry.source_width
        sy = geometry.processed_height / geometry.source_height
        gt_history.append(TrackPoint(
            frame_index=index,
            timestamp_ms=a_times.get(index, int(round(index / geometry.fps * 1000))),
            cx=x / geometry.processed_width, cy=y / geometry.processed_height,
            w=min(1.0, point.width * sx / geometry.processed_width),
            h=min(1.0, point.height * sy / geometry.processed_height), conf=1.0,
        ))
    base = original.model_dump(exclude={"history", "quality"})
    a_track = TrackSequence(**base, history=a_history)
    gt_full = TrackSequence(**base, history=gt_history)
    gt_matched = TrackSequence(**base, history=[p for p in gt_history if p.frame_index in a_times])
    tracks = {"a": a_track, "gt_full": gt_full, "gt_matched": gt_matched}
    results = {name: _feature_result(track) for name, track in tracks.items()}
    comparisons = _comparison(results["a"], results["gt_matched"])
    full_comparisons = _comparison(results["a"], results["gt_full"])
    sampling_comparisons = _comparison(results["gt_matched"], results["gt_full"])
    for name, track in tracks.items():
        write_json(output_dir / f"{name}_track_sequence.json", track.model_dump(mode="json"))
    result = {
        "status": "completed" if all(r["status"] == "ok" for r in results.values()) else "feature_rejected",
        "reasons": [f"{name}: {reason}" for name, r in results.items() for reason in r["reasons"]],
        "coordinate_space": "camera_motion_compensated" if method != "none" else "frame",
        "camera_verification": camera_verification, "a_clamped_point_count": clamped_count,
        "scope": "same_attempted_gt_window; matched_GT_uses_exact_A_frame_indices_and_timestamps",
        "point_counts": {name: len(track.history) for name, track in tracks.items()},
        "config": asdict(FeatureConfig()), "results": results,
        "comparisons": comparisons, "full_gt_comparisons": full_comparisons,
        "sampling_comparisons": sampling_comparisons,
        "provenance": {"track_sequence": str(track_path), "track_sha256": _sha256(track_path),
                       "feature_core_sha256": _sha256(Path(__file__).parents[1] / "ai_server/services/feature_core.py"),
                       "research_features_sha256": _sha256(Path(__file__).with_name("features.py"))},
    }
    write_json(output_dir / "comparison.json", result)
    return result


def write_feature_comparison_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    rows = []
    for report in reports:
        b = report.get("b_features", {})
        for scope, key in [("matched_gt", "comparisons"), ("full_gt", "full_gt_comparisons"),
                           ("gt_sampling_only", "sampling_comparisons")]:
            for comparison in b.get(key, []):
                rows.append({"sample_id": report["sample_id"], "scope": scope, **comparison})
    fields = ["sample_id", "scope", "feature", "unit", "gt", "a", "signed_difference",
              "absolute_difference", "relative_difference_percent", "relative_status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
