from __future__ import annotations

from dataclasses import dataclass
from math import hypot
import os
from pathlib import Path
from statistics import median
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from ai_server.schemas import TrackSequence
from ai_server.services.foreground_motion import (
    ConstantVelocityKalman,
    build_foreground_mask,
    find_residual_motion,
    foreground_contrast_score,
    place_foreground_mask,
)
from ai_server.services.online_appearance import OnlineAppearanceModel
from ai_server.services.tracking_adapter import observations_to_track_sequence
from ai_server.services.tracking_video_io import (
    TrackingVideoError,
    affine_to_homography,
    create_tracking_video_writer,
    iter_tracking_video_frames,
    read_tracking_video_metadata,
    transform_point,
    write_tracking_csv,
    write_tracking_json,
)
from ai_server.tracking_schemas import TrackingTuning


@dataclass(slots=True)
class PointObservation:
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
    visible_points: int
    bg_points: int
    bg_inliers: int
    bg_inlier_ratio: float
    direction: str
    tracking_source: str
    fb_error: float
    template_score: float
    appearance_score: float
    foreground_contrast: float
    motion_score: float
    appearance_model_updated: bool


@dataclass(frozen=True, slots=True)
class ManualTrackingResult:
    track: TrackSequence
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, str | None]


def process_manual_roi_video(
    video_path: str,
    *,
    source_video_id: str,
    target_bbox: tuple[float, float, float, float],
    init_frame_index: int = 0,
    max_seconds: float | None = None,
    stabilize: bool = True,
    track_id: int = 1,
    tuning: TrackingTuning | None = None,
    output_dir: str | Path | None = None,
    resize_width: int | None = 1280,
    write_overlay: bool = True,
) -> ManualTrackingResult:
    tuning = tuning or TrackingTuning()
    metadata_obj = read_tracking_video_metadata(
        video_path,
        resize_width=resize_width,
    )
    metadata = metadata_obj.to_dict()
    fps = metadata_obj.fps

    if init_frame_index < 0:
        raise ValueError("init_frame_index must be zero or greater")
    max_frames = _max_loaded_frames(
        frame_count=metadata_obj.frame_count,
        fps=fps,
        init_frame_index=init_frame_index,
        max_seconds=max_seconds,
    )
    frame_items = list(
        iter_tracking_video_frames(
            video_path,
            resize_width=resize_width,
            max_frames=max_frames,
        )
    )
    if not frame_items:
        raise TrackingVideoError(f"Video contains no readable frames: {video_path}")
    if init_frame_index >= len(frame_items):
        raise ValueError(
            f"init_frame_index {init_frame_index} exceeds the loaded frame range "
            f"0..{len(frame_items) - 1}"
        )

    init_bbox = _scale_initial_bbox(metadata, target_bbox)
    init_frame = frame_items[init_frame_index][1]
    frame_height, frame_width = init_frame.shape[:2]
    init_bbox = _clamp_bbox(init_bbox, width=frame_width, height=frame_height)
    init_point = (
        init_bbox[0] + init_bbox[2] / 2.0,
        init_bbox[1] + init_bbox[3] / 2.0,
    )
    init_point = _clamp_point(init_point, width=frame_width, height=frame_height)

    init_gray = _gray(init_frame)
    foreground = build_foreground_mask(
        init_frame,
        init_point,
        (init_bbox[2], init_bbox[3]),
    )
    init_points = _initialize_roi_points(
        init_gray,
        init_bbox,
        fallback_center=init_point,
        local_mask=foreground.mask,
    )
    init_template = _extract_target_template(
        init_gray,
        init_point,
        (init_bbox[2], init_bbox[3]),
    )

    backward = _track_ordered_frames(
        frame_items=list(reversed(frame_items[: init_frame_index + 1])),
        init_points=init_points,
        init_center=init_point,
        init_bbox=init_bbox,
        init_template=init_template,
        fps=fps,
        stabilize=stabilize,
        direction="backward",
        tuning=tuning,
        foreground_mask=foreground.mask,
    )
    forward = _track_ordered_frames(
        frame_items=frame_items[init_frame_index:],
        init_points=init_points,
        init_center=init_point,
        init_bbox=init_bbox,
        init_template=init_template,
        fps=fps,
        stabilize=stabilize,
        direction="forward",
        tuning=tuning,
        foreground_mask=foreground.mask,
    )
    observations = sorted(
        [*reversed(backward[1:]), *forward],
        key=lambda observation: observation.frame_index,
    )

    track = observations_to_track_sequence(
        observations,
        track_id=track_id,
        source_video_id=source_video_id,
        frame_width=frame_width,
        frame_height=frame_height,
        stabilize=stabilize,
    )
    metrics = _compute_metrics(observations, stabilize=stabilize)
    metrics["method"] = "manual_roi_foreground_klt_motion" + (
        "_cmc" if stabilize else ""
    )
    metrics["tracking_parameters"] = tuning.model_dump()

    run_id = f"{_safe_stem(source_video_id)}_{uuid4().hex[:8]}"
    root_dir = Path(
        output_dir
        or os.environ.get("SKYDETECT_TRACK_OUTPUT_DIR", "artifacts/manual_tracks")
    )
    run_dir = root_dir / run_id
    debug_dir = run_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    foreground_mask_path = debug_dir / "foreground_mask.png"
    if not cv2.imwrite(str(foreground_mask_path), foreground.mask):
        raise TrackingVideoError(f"Failed to write foreground mask: {foreground_mask_path}")

    metadata.update(
        {
            "run_id": run_id,
            "source_video_id": source_video_id,
            "source_video_path": str(Path(video_path).expanduser()),
            "num_frames_processed": len(frame_items),
            "init_frame_index": init_frame_index,
            "init_point_processed": [float(value) for value in init_point],
            "init_bbox_processed": [float(value) for value in init_bbox],
            "coordinate_mode": "camera_motion_compensated" if stabilize else "frame",
            "tracker_method": "manual_roi_klt_appearance_motion",
            "tracking_parameters": tuning.model_dump(),
            "foreground_mask_coverage": foreground.coverage,
            "foreground_mask_segmented": foreground.segmented,
        }
    )

    metadata_path = write_tracking_json(metadata, run_dir / "metadata.json")
    track_path = write_tracking_json(
        track.model_dump(mode="json"),
        run_dir / "track_sequence.json",
    )
    metrics_path = write_tracking_json(metrics, run_dir / "metrics.json")
    debug_csv_path = write_tracking_csv(
        [_debug_row(observation) for observation in observations],
        debug_dir / "observations.csv",
    )
    trajectory_path = write_tracking_csv(
        [_trajectory_row(observation) for observation in observations],
        run_dir / "trajectory.csv",
    )

    overlay_path: str | None = None
    if write_overlay:
        overlay_path = _write_overlay(
            frame_items=frame_items,
            observations=observations,
            output_path=run_dir / "overlay.mp4",
            fps=fps,
        )

    return ManualTrackingResult(
        track=track,
        metadata=metadata,
        metrics=metrics,
        artifacts={
            "run_dir": str(run_dir),
            "track_sequence": track_path,
            "metadata": metadata_path,
            "metrics": metrics_path,
            "trajectory": trajectory_path,
            "overlay": overlay_path,
            "foreground_mask": str(foreground_mask_path),
            "debug_observations": debug_csv_path,
        },
    )


def _track_ordered_frames(
    *,
    frame_items: list[tuple[int, np.ndarray]],
    init_points: np.ndarray,
    init_center: tuple[float, float],
    init_bbox: tuple[float, float, float, float],
    init_template: np.ndarray,
    fps: float,
    stabilize: bool,
    direction: str,
    tuning: TrackingTuning,
    foreground_mask: np.ndarray,
) -> list[PointObservation]:
    first_index, first_frame = frame_items[0]
    height, width = first_frame.shape[:2]
    points = init_points.astype(np.float32)
    center = _clamp_point(init_center, width=width, height=height)
    bbox_w = float(init_bbox[2])
    bbox_h = float(init_bbox[3])
    template = init_template.copy()
    first_gray = _gray(first_frame)
    appearance_model = OnlineAppearanceModel(
        first_gray,
        center,
        (bbox_w, bbox_h),
        foreground_mask=foreground_mask,
    )
    ref_to_current = np.eye(3, dtype=np.float32)
    motion_filter = ConstantVelocityKalman(center)
    initial_foreground_contrast = foreground_contrast_score(
        first_gray,
        center,
        (bbox_w, bbox_h),
        foreground_mask,
    )
    lost_streak = 0
    observations = [
        PointObservation(
            frame_index=int(first_index),
            timestamp_ms=_timestamp_ms(first_index, fps),
            raw_x=float(center[0]),
            raw_y=float(center[1]),
            compensated_x=float(center[0]),
            compensated_y=float(center[1]),
            w=bbox_w,
            h=bbox_h,
            confidence=1.0,
            visible=True,
            visible_points=len(points),
            bg_points=0,
            bg_inliers=0,
            bg_inlier_ratio=1.0,
            direction=direction,
            tracking_source="manual_roi",
            fb_error=0.0,
            template_score=1.0,
            appearance_score=1.0,
            foreground_contrast=initial_foreground_contrast,
            motion_score=0.0,
            appearance_model_updated=False,
        )
    ]

    prev_gray = first_gray
    prev_center = np.array(center, dtype=np.float32)
    for frame_index, frame in frame_items[1:]:
        curr_gray = _gray(frame)
        if stabilize:
            affine, bg_stats = _estimate_background_affine(
                prev_gray,
                curr_gray,
                target_center=tuple(float(value) for value in prev_center),
                target_size=(bbox_w, bbox_h),
            )
            ref_to_current = affine_to_homography(affine) @ ref_to_current
            current_to_reference = _safe_inverse(ref_to_current)
        else:
            affine = _identity_affine()
            current_to_reference = np.eye(3, dtype=np.float32)
            bg_stats = {"points": 0, "inliers": 0, "inlier_ratio": 1.0}

        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            points.reshape(-1, 1, 2),
            None,
            **_lk_params(),
        )
        predicted_compensated = motion_filter.predict()
        predicted_center = np.array(
            transform_point(
                ref_to_current,
                predicted_compensated[0],
                predicted_compensated[1],
            ),
            dtype=np.float32,
        )
        next_points_2d = (
            points.copy() if next_points is None else next_points.reshape(-1, 2)
        )
        good = _good_point_mask(
            next_points_2d,
            status,
            error,
            width=width,
            height=height,
        )
        fb_errors = np.full(len(points), np.inf, dtype=np.float32)
        if next_points is not None and good.any():
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray,
                prev_gray,
                next_points,
                None,
                **_lk_params(),
            )
            if back_points is not None and back_status is not None:
                fb_errors = np.linalg.norm(
                    back_points.reshape(-1, 2) - points,
                    axis=1,
                )
                good &= (back_status.reshape(-1) == 1) & (fb_errors <= 2.2)
        if good.any():
            candidate_center = np.array(
                _median_point(next_points_2d[good]),
                dtype=np.float32,
            )
            distance_gate = max(14.0, max(bbox_w, bbox_h) * 3.8)
            distances = np.linalg.norm(next_points_2d - candidate_center, axis=1)
            good &= distances <= distance_gate

        visible_points = int(good.sum())
        required_points = _minimum_visible_points(len(points))
        tracking_source = "klt"
        template_score = 0.0
        appearance_score = 0.0
        motion_score = 0.0
        fb_error = float(np.mean(fb_errors[good])) if good.any() else 99.0
        if visible_points >= required_points:
            displacement = np.median(next_points_2d[good] - points[good], axis=0)
            new_center = prev_center + displacement.astype(np.float32)
            innovation_distance = float(np.linalg.norm(new_center - predicted_center))
            kinematic_gate = max(8.0, max(bbox_w, bbox_h) * 0.55)
            kinematic_valid = innovation_distance <= kinematic_gate
            refined = _refine_center_by_contrast(
                curr_gray,
                tuple(float(value) for value in new_center),
                radius=max(8, int(max(bbox_w, bbox_h) * 2)),
            )
            if refined is not None and hypot(
                refined[0] - new_center[0],
                refined[1] - new_center[1],
            ) <= max(6.0, max(bbox_w, bbox_h)):
                new_center = 0.85 * new_center + 0.15 * np.array(
                    refined,
                    dtype=np.float32,
                )
            template_score = _template_similarity(
                curr_gray,
                template,
                tuple(float(value) for value in new_center),
                foreground_mask,
            )
            appearance_score = appearance_model.score(
                curr_gray,
                tuple(float(value) for value in new_center),
            )
            visible = kinematic_valid and appearance_score >= tuning.klt_accept_conf
        else:
            new_center = predicted_center
            visible = False

        foreground_contrast = foreground_contrast_score(
            curr_gray,
            tuple(float(value) for value in new_center),
            (bbox_w, bbox_h),
            foreground_mask,
        )
        low_contrast = _is_low_foreground_contrast(
            initial=initial_foreground_contrast,
            current=foreground_contrast,
        )
        base_search_radius = max(
            18,
            int(max(bbox_w, bbox_h) * tuning.search_radius_multiplier),
        )
        search_radius = int(
            base_search_radius * min(3.0, 1.0 + 0.45 * lost_streak)
        )
        recovery_gate = max(8.0, max(init_bbox[2], init_bbox[3]) * 0.45) * min(
            2.5,
            1.0 + 0.20 * lost_streak,
        )

        motion_match = None
        if not visible or low_contrast:
            motion_match = find_residual_motion(
                prev_gray,
                curr_gray,
                affine,
                tuple(float(value) for value in predicted_center),
                target_size=(bbox_w, bbox_h),
                foreground_mask=foreground_mask,
                search_radius=search_radius,
            )
            motion_score = float(motion_match.score) if motion_match else 0.0

        motion_distance = (
            hypot(
                motion_match.center[0] - float(predicted_center[0]),
                motion_match.center[1] - float(predicted_center[1]),
            )
            if motion_match is not None
            else float("inf")
        )
        motion_valid = (
            motion_match is not None
            and motion_match.score >= 0.38
            and motion_distance <= recovery_gate
        )
        if visible and low_contrast:
            support_distance = (
                hypot(
                    motion_match.center[0] - float(new_center[0]),
                    motion_match.center[1] - float(new_center[1]),
                )
                if motion_match is not None
                else float("inf")
            )
            if motion_valid and support_distance <= max(
                8.0,
                max(bbox_w, bbox_h) * 0.80,
            ):
                new_center = 0.62 * new_center + 0.38 * np.array(
                    motion_match.center,
                    dtype=np.float32,
                )
                tracking_source = "motion"
            else:
                visible = False

        if not visible:
            appearance_match = appearance_model.search(
                curr_gray,
                tuple(float(value) for value in predicted_center),
                search_radius=search_radius,
            )
            recovery_distance = (
                hypot(
                    appearance_match.center[0] - float(predicted_center[0]),
                    appearance_match.center[1] - float(predicted_center[1]),
                )
                if appearance_match is not None
                else float("inf")
            )
            appearance_contrast = (
                foreground_contrast_score(
                    curr_gray,
                    appearance_match.center,
                    (bbox_w, bbox_h),
                    foreground_mask,
                )
                if appearance_match is not None
                else 0.0
            )
            appearance_valid = (
                appearance_match is not None
                and appearance_match.score >= tuning.recovery_conf
                and recovery_distance <= recovery_gate
            )
            appearance_low_contrast = _is_low_foreground_contrast(
                initial=initial_foreground_contrast,
                current=appearance_contrast,
            )
            if appearance_valid and not appearance_low_contrast:
                new_center = np.array(appearance_match.center, dtype=np.float32)
                appearance_score = float(appearance_match.score)
                template_score = float(appearance_match.anchor_score)
                tracking_source = "appearance"
                foreground_contrast = appearance_contrast
                visible = True
            elif motion_valid and motion_match is not None:
                new_center = np.array(motion_match.center, dtype=np.float32)
                appearance_score = appearance_model.score(
                    curr_gray,
                    motion_match.center,
                )
                template_score = _template_similarity(
                    curr_gray,
                    template,
                    motion_match.center,
                    foreground_mask,
                )
                tracking_source = "motion"
                foreground_contrast = foreground_contrast_score(
                    curr_gray,
                    motion_match.center,
                    (bbox_w, bbox_h),
                    foreground_mask,
                )
                visible = True
            else:
                new_center = predicted_center
                appearance_score = (
                    float(appearance_match.score) if appearance_match else 0.0
                )
                template_score = (
                    float(appearance_match.anchor_score) if appearance_match else 0.0
                )
                tracking_source = "prediction"
                foreground_contrast = foreground_contrast_score(
                    curr_gray,
                    tuple(float(value) for value in predicted_center),
                    (bbox_w, bbox_h),
                    foreground_mask,
                )
                visible = False

        lost_streak = 0 if visible else lost_streak + 1
        new_center = np.array(
            _clamp_point(
                (float(new_center[0]), float(new_center[1])),
                width=width,
                height=height,
            ),
            dtype=np.float32,
        )
        if tracking_source == "klt" and visible_points >= 4:
            spread_w, spread_h = _point_spread_size(
                next_points_2d[good],
                fallback=(bbox_w, bbox_h),
            )
            bbox_w = _blend_size(bbox_w, spread_w, base=init_bbox[2])
            bbox_h = _blend_size(bbox_h, spread_h, base=init_bbox[3])

        comp_x, comp_y = transform_point(
            current_to_reference,
            float(new_center[0]),
            float(new_center[1]),
        )
        confidence = _confidence(
            visible_points=visible_points,
            total_points=len(points),
            bg_inlier_ratio=float(bg_stats["inlier_ratio"]),
            visible=visible,
            tracking_source=tracking_source,
            fb_error=fb_error,
            template_score=template_score,
            appearance_score=appearance_score,
            motion_score=motion_score,
        )
        appearance_model_updated = False
        if (
            tuning.online_update_enabled
            and tracking_source == "klt"
            and visible
            and confidence >= 0.72
            and appearance_score >= tuning.update_conf
            and fb_error <= 1.5
            and frame_index % 3 == 0
        ):
            appearance_model_updated = appearance_model.update(
                curr_gray,
                (float(new_center[0]), float(new_center[1])),
            )
            candidate_template = _extract_target_template(
                curr_gray,
                (float(new_center[0]), float(new_center[1])),
                (init_bbox[2], init_bbox[3]),
            )
            if candidate_template.shape == template.shape:
                blended_template = cv2.addWeighted(
                    template,
                    0.97,
                    candidate_template,
                    0.03,
                    0.0,
                )
                selected = cv2.resize(
                    foreground_mask,
                    (template.shape[1], template.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
                template[selected] = blended_template[selected]

        observations.append(
            PointObservation(
                frame_index=int(frame_index),
                timestamp_ms=_timestamp_ms(frame_index, fps),
                raw_x=float(new_center[0]),
                raw_y=float(new_center[1]),
                compensated_x=float(comp_x),
                compensated_y=float(comp_y),
                w=float(bbox_w),
                h=float(bbox_h),
                confidence=confidence,
                visible=visible,
                visible_points=visible_points,
                bg_points=int(bg_stats["points"]),
                bg_inliers=int(bg_stats["inliers"]),
                bg_inlier_ratio=float(bg_stats["inlier_ratio"]),
                direction=direction,
                tracking_source=tracking_source,
                fb_error=fb_error,
                template_score=template_score,
                appearance_score=appearance_score,
                foreground_contrast=foreground_contrast,
                motion_score=motion_score,
                appearance_model_updated=appearance_model_updated,
            )
        )

        if visible:
            motion_filter.correct((float(comp_x), float(comp_y)))
        current_bbox = (
            float(new_center[0] - bbox_w / 2.0),
            float(new_center[1] - bbox_h / 2.0),
            float(bbox_w),
            float(bbox_h),
        )
        points = _initialize_roi_points(
            curr_gray,
            _clamp_bbox(current_bbox, width=width, height=height),
            fallback_center=(float(new_center[0]), float(new_center[1])),
            local_mask=foreground_mask,
        )
        prev_center = new_center
        prev_gray = curr_gray

    return observations


def _max_loaded_frames(
    *,
    frame_count: int,
    fps: float,
    init_frame_index: int,
    max_seconds: float | None,
) -> int | None:
    if max_seconds is None:
        return frame_count or None
    duration_frames = max(1, int(round(max_seconds * max(fps, 1e-6))))
    requested = init_frame_index + duration_frames + 1
    return min(frame_count, requested) if frame_count > 0 else requested


def _scale_initial_bbox(
    metadata: dict[str, Any],
    target_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    source_width = max(float(metadata.get("width") or 1.0), 1.0)
    source_height = max(float(metadata.get("height") or 1.0), 1.0)
    processed_width = max(
        float(metadata.get("processed_width") or source_width),
        1.0,
    )
    processed_height = max(
        float(metadata.get("processed_height") or source_height),
        1.0,
    )
    scale_x = processed_width / source_width
    scale_y = processed_height / source_height
    x, y, width, height = target_bbox
    return (
        float(x) * scale_x,
        float(y) * scale_y,
        float(width) * scale_x,
        float(height) * scale_y,
    )


def _estimate_background_affine(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    target_center: tuple[float, float],
    target_size: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = prev_gray.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    margin = max(12.0, max(target_size) * 0.75)
    half_width = target_size[0] / 2.0 + margin
    half_height = target_size[1] / 2.0 + margin
    center_x, center_y = target_center
    x1 = max(0, int(round(center_x - half_width)))
    y1 = max(0, int(round(center_y - half_height)))
    x2 = min(width, int(round(center_x + half_width)))
    y2 = min(height, int(round(center_y + half_height)))
    mask[y1:y2, x1:x2] = 0

    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=420,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        mask=mask,
    )
    if prev_points is None or len(prev_points) < 8:
        return _identity_affine(), {
            "points": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    curr_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        **_lk_params(),
    )
    if curr_points is None or status is None:
        return _identity_affine(), {
            "points": int(len(prev_points)),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    prev_2d = prev_points.reshape(-1, 2)
    curr_2d = curr_points.reshape(-1, 2)
    error_1d = (
        np.zeros(len(prev_2d), dtype=np.float32)
        if error is None
        else error.reshape(-1)
    )
    good = (status.reshape(-1) == 1) & (error_1d <= 35.0)
    good &= (
        (curr_2d[:, 0] >= 0)
        & (curr_2d[:, 0] < width)
        & (curr_2d[:, 1] >= 0)
        & (curr_2d[:, 1] < height)
    )
    if int(good.sum()) < 8:
        return _identity_affine(), {
            "points": int(good.sum()),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    matrix, inliers = cv2.estimateAffinePartial2D(
        prev_2d[good],
        curr_2d[good],
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None or inliers is None:
        return _identity_affine(), {
            "points": int(good.sum()),
            "inliers": 0,
            "inlier_ratio": 0.0,
        }

    inlier_count = int(inliers.reshape(-1).sum())
    ratio = inlier_count / max(int(good.sum()), 1)
    if inlier_count < 8 or ratio < 0.25:
        return _identity_affine(), {
            "points": int(good.sum()),
            "inliers": inlier_count,
            "inlier_ratio": float(ratio),
        }
    return matrix.astype(np.float32), {
        "points": int(good.sum()),
        "inliers": inlier_count,
        "inlier_ratio": float(ratio),
    }


def _compute_metrics(
    observations: list[PointObservation],
    *,
    stabilize: bool,
) -> dict[str, Any]:
    visible = [observation for observation in observations if observation.visible]
    visible_count = len(visible)
    missing_ratio = 1.0 - visible_count / max(len(observations), 1)
    mean_confidence = (
        sum(observation.confidence for observation in visible) / visible_count
        if visible
        else 0.0
    )
    transitions = observations[1:]
    if stabilize and transitions:
        background_ratio: float | None = sum(
            observation.bg_inlier_ratio for observation in transitions
        ) / len(transitions)
        background_valid_ratio: float | None = sum(
            1 for observation in transitions if observation.bg_inliers >= 8
        ) / len(transitions)
    else:
        background_ratio = None
        background_valid_ratio = None

    raw_points = [(observation.raw_x, observation.raw_y) for observation in observations]
    compensated_points = [
        (observation.compensated_x, observation.compensated_y)
        for observation in observations
    ]
    compensated_jumps = _jump_count(compensated_points)
    length_score = min(1.0, len(observations) / 70.0)
    visible_score = 1.0 - missing_ratio
    smooth_score = max(
        0.0,
        1.0 - compensated_jumps / max(len(observations) * 0.08, 1.0),
    )
    if stabilize:
        quality_score = (
            0.34 * length_score
            + 0.28 * visible_score
            + 0.18 * mean_confidence
            + 0.12 * float(background_ratio or 0.0)
            + 0.08 * smooth_score
        )
    else:
        quality_score = (
            0.40 * length_score
            + 0.32 * visible_score
            + 0.20 * mean_confidence
            + 0.08 * smooth_score
        )

    source_counts = {
        source: sum(
            1 for observation in observations if observation.tracking_source == source
        )
        for source in ["manual_roi", "klt", "appearance", "motion", "prediction"]
    }
    return {
        "num_frames_processed": len(observations),
        "visible_frames": visible_count,
        "visible_ratio": visible_count / max(len(observations), 1),
        "missing_ratio": missing_ratio,
        "mean_observation_confidence": mean_confidence,
        "background_inlier_ratio": background_ratio,
        "background_valid_ratio": background_valid_ratio,
        "tracking_source_counts": source_counts,
        "mean_appearance_score": (
            sum(observation.appearance_score for observation in visible) / visible_count
            if visible
            else 0.0
        ),
        "appearance_model_updates": sum(
            1 for observation in observations if observation.appearance_model_updated
        ),
        "low_contrast_frames": sum(
            1 for observation in observations if observation.foreground_contrast < 0.20
        ),
        "mean_foreground_contrast": sum(
            observation.foreground_contrast for observation in observations
        )
        / max(len(observations), 1),
        "mean_motion_score": sum(
            observation.motion_score for observation in observations
        )
        / max(len(observations), 1),
        "raw_jump_count": _jump_count(raw_points),
        "compensated_jump_count": compensated_jumps,
        "raw_motion_extent_px": _motion_extent(raw_points),
        "compensated_motion_extent_px": _motion_extent(compensated_points),
        "quality_score": float(max(0.0, min(1.0, quality_score))),
    }


def _write_overlay(
    *,
    frame_items: list[tuple[int, np.ndarray]],
    observations: list[PointObservation],
    output_path: Path,
    fps: float,
) -> str:
    by_frame = {observation.frame_index: observation for observation in observations}
    first_frame = frame_items[0][1]
    height, width = first_frame.shape[:2]
    writer = create_tracking_video_writer(
        output_path,
        fps=fps,
        width=width,
        height=height,
    )
    trail: list[tuple[int, int]] = []
    try:
        for frame_index, frame in frame_items:
            canvas = frame.copy()
            observation = by_frame.get(frame_index)
            if observation is not None:
                color = {
                    "manual_roi": (0, 0, 255),
                    "klt": (0, 220, 255),
                    "appearance": (220, 160, 40),
                    "motion": (70, 220, 80),
                    "prediction": (0, 140, 255),
                }.get(observation.tracking_source, (0, 220, 255))
                if observation.visible:
                    trail.append(
                        (int(round(observation.raw_x)), int(round(observation.raw_y)))
                    )
                if len(trail) >= 2:
                    cv2.polylines(
                        canvas,
                        [np.array(trail, dtype=np.int32).reshape(-1, 1, 2)],
                        False,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                x1 = int(round(observation.raw_x - observation.w / 2.0))
                y1 = int(round(observation.raw_y - observation.h / 2.0))
                x2 = int(round(observation.raw_x + observation.w / 2.0))
                y2 = int(round(observation.raw_y + observation.h / 2.0))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                label = (
                    f"f{frame_index} {observation.tracking_source} "
                    f"conf={observation.confidence:.2f} "
                    f"app={observation.appearance_score:.2f} "
                    f"motion={observation.motion_score:.2f}"
                )
                cv2.putText(
                    canvas,
                    label,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    label,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
    finally:
        writer.release()
    return str(output_path)


def _initialize_roi_points(
    gray: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    fallback_center: tuple[float, float],
    local_mask: np.ndarray | None = None,
) -> np.ndarray:
    height, width = gray.shape[:2]
    x, y, bbox_width, bbox_height = _clamp_bbox(
        bbox,
        width=width,
        height=height,
    )
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(width, int(np.ceil(x + bbox_width)))
    y2 = min(height, int(np.ceil(y + bbox_height)))
    if local_mask is None:
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
    else:
        mask = place_foreground_mask(
            gray.shape,
            (x, y, bbox_width, bbox_height),
            local_mask,
        )

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=24,
        qualityLevel=0.005,
        minDistance=max(2.0, min(bbox_width, bbox_height) / 8.0),
        blockSize=3,
        mask=mask,
    )
    if corners is not None and len(corners) >= 4:
        return corners.reshape(-1, 2).astype(np.float32)

    center_x, center_y = fallback_center
    offset_x = max(2.0, min(8.0, bbox_width * 0.25))
    offset_y = max(2.0, min(8.0, bbox_height * 0.25))
    offsets = np.array(
        [
            [0.0, 0.0],
            [-offset_x, 0.0],
            [offset_x, 0.0],
            [0.0, -offset_y],
            [0.0, offset_y],
            [-offset_x, -offset_y],
            [offset_x, -offset_y],
            [-offset_x, offset_y],
            [offset_x, offset_y],
        ],
        dtype=np.float32,
    )
    fallback = offsets + np.array([center_x, center_y], dtype=np.float32)
    fallback[:, 0] = np.clip(fallback[:, 0], 0, max(width - 1, 0))
    fallback[:, 1] = np.clip(fallback[:, 1], 0, max(height - 1, 0))
    fallback = _points_inside_mask(fallback, mask)
    if len(fallback) < 4:
        fallback = _sample_mask_points(
            mask,
            fallback_center=fallback_center,
            limit=9,
        )
    if corners is None:
        return fallback
    return np.concatenate(
        [corners.reshape(-1, 2).astype(np.float32), fallback],
        axis=0,
    )


def _points_inside_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32)
    height, width = mask.shape[:2]
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, max(width - 1, 0))
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, max(height - 1, 0))
    return points[mask[y, x] > 0].astype(np.float32)


def _sample_mask_points(
    mask: np.ndarray,
    *,
    fallback_center: tuple[float, float],
    limit: int,
) -> np.ndarray:
    y, x = np.where(mask > 0)
    if len(x) == 0:
        return np.array([fallback_center], dtype=np.float32)
    candidates = np.column_stack([x, y]).astype(np.float32)
    center = np.array(fallback_center, dtype=np.float32)
    distances = np.linalg.norm(candidates - center, axis=1)
    order = np.argsort(distances)
    stride = max(1, len(order) // max(limit, 1))
    selected = candidates[order[::stride][:limit]]
    return selected.astype(np.float32) if len(selected) else center.reshape(1, 2)


def _extract_target_template(
    gray: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
) -> np.ndarray:
    width = max(4, int(round(size[0])))
    height = max(4, int(round(size[1])))
    return cv2.getRectSubPix(gray, (width, height), center)


def _template_similarity(
    gray: np.ndarray,
    template: np.ndarray,
    center: tuple[float, float],
    foreground_mask: np.ndarray,
) -> float:
    patch = _extract_target_template(
        gray,
        center,
        (template.shape[1], template.shape[0]),
    )
    if patch.shape != template.shape:
        return 0.0
    mask = cv2.resize(
        foreground_mask,
        (template.shape[1], template.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    if int(mask.sum()) < 4:
        mask = np.ones(template.shape[:2], dtype=bool)
    template_mean = float(np.mean(template[mask]))
    patch_mean = float(np.mean(patch[mask]))
    template_values = np.where(
        mask,
        template.astype(np.float32) - template_mean,
        0.0,
    )
    patch_values = np.where(
        mask,
        patch.astype(np.float32) - patch_mean,
        0.0,
    )
    denominator = float(np.linalg.norm(template_values) * np.linalg.norm(patch_values))
    if denominator <= 1e-6:
        difference = float(
            np.mean(
                np.abs(
                    template.astype(np.float32)[mask]
                    - patch.astype(np.float32)[mask]
                )
            )
        )
        return float(max(0.0, 1.0 - difference / 255.0))
    correlation = float(np.sum(template_values * patch_values) / denominator)
    return float(max(0.0, min(1.0, correlation)))


def _is_low_foreground_contrast(*, initial: float, current: float) -> bool:
    if initial < 0.24:
        return False
    return current < max(0.14, initial * 0.46)


def _minimum_visible_points(total_points: int) -> int:
    return max(2, min(4, max(total_points, 1) // 3))


def _good_point_mask(
    points: np.ndarray,
    status: np.ndarray | None,
    error: np.ndarray | None,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if status is None:
        return np.zeros(len(points), dtype=bool)
    error_1d = (
        np.zeros(len(points), dtype=np.float32)
        if error is None
        else error.reshape(-1)
    )
    good = (status.reshape(-1) == 1) & (error_1d <= 45.0)
    good &= (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return good


def _point_spread_size(
    points: np.ndarray,
    *,
    fallback: tuple[float, float],
) -> tuple[float, float]:
    if len(points) < 2:
        return fallback
    width = float(np.max(points[:, 0]) - np.min(points[:, 0]) + 4.0)
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]) + 4.0)
    return max(width, 2.0), max(height, 2.0)


def _blend_size(current: float, observed: float, *, base: float) -> float:
    low = max(2.0, base * 0.65)
    high = max(low + 1.0, base * 2.2)
    observed = max(low, min(high, observed))
    return float(0.85 * current + 0.15 * observed)


def _refine_center_by_contrast(
    gray: np.ndarray,
    center: tuple[float, float],
    *,
    radius: int,
) -> tuple[float, float] | None:
    height, width = gray.shape[:2]
    center_x, center_y = center
    x1 = max(0, int(round(center_x - radius)))
    y1 = max(0, int(round(center_y - radius)))
    x2 = min(width, int(round(center_x + radius + 1)))
    y2 = min(height, int(round(center_y + radius + 1)))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    blur_size = 9 if min(crop.shape[:2]) >= 9 else 5
    if blur_size >= min(crop.shape[:2]):
        return None
    background = cv2.GaussianBlur(crop, (blur_size, blur_size), 0)
    response = cv2.absdiff(crop, background)
    _, max_value, _, max_location = cv2.minMaxLoc(response)
    if max_value < 5.0:
        return None
    return float(x1 + max_location[0]), float(y1 + max_location[1])


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x, y, bbox_width, bbox_height = bbox
    bbox_width = max(2.0, min(float(bbox_width), float(width)))
    bbox_height = max(2.0, min(float(bbox_height), float(height)))
    x = max(0.0, min(float(x), max(float(width) - bbox_width, 0.0)))
    y = max(0.0, min(float(y), max(float(height) - bbox_height, 0.0)))
    return x, y, bbox_width, bbox_height


def _clamp_point(
    point: tuple[float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        max(0.0, min(float(point[0]), max(float(width - 1), 0.0))),
        max(0.0, min(float(point[1]), max(float(height - 1), 0.0))),
    )


def _median_point(points: np.ndarray) -> tuple[float, float]:
    return float(np.median(points[:, 0])), float(np.median(points[:, 1]))


def _confidence(
    *,
    visible_points: int,
    total_points: int,
    bg_inlier_ratio: float,
    visible: bool,
    tracking_source: str,
    fb_error: float,
    template_score: float,
    appearance_score: float,
    motion_score: float,
) -> float:
    if not visible:
        return 0.12
    if tracking_source == "appearance":
        return float(max(0.20, min(0.90, 0.14 + 0.76 * appearance_score)))
    if tracking_source == "motion":
        return float(max(0.18, min(0.86, 0.16 + 0.70 * motion_score)))
    target_ratio = visible_points / max(total_points, 1)
    fb_quality = max(0.0, 1.0 - fb_error / 2.2)
    return float(
        max(
            0.15,
            min(
                1.0,
                0.10
                + 0.42 * target_ratio
                + 0.12 * bg_inlier_ratio
                + 0.10 * fb_quality
                + 0.08 * template_score
                + 0.18 * appearance_score,
            ),
        )
    )


def _jump_count(points: list[tuple[float, float]]) -> int:
    if len(points) < 3:
        return 0
    steps = [
        hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(points, points[1:])
    ]
    baseline = max(6.0, float(median(steps)) * 4.0)
    return sum(1 for step in steps if step > baseline)


def _motion_extent(points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return float(
        hypot(max(x_values) - min(x_values), max(y_values) - min(y_values))
    )


def _debug_row(observation: PointObservation) -> dict[str, Any]:
    return {
        **_trajectory_row(observation),
        "visible_points": observation.visible_points,
        "background_points": observation.bg_points,
        "background_inliers": observation.bg_inliers,
        "background_inlier_ratio": observation.bg_inlier_ratio,
        "direction": observation.direction,
        "forward_backward_error": observation.fb_error,
        "template_score": observation.template_score,
    }


def _trajectory_row(observation: PointObservation) -> dict[str, Any]:
    return {
        "frame_index": observation.frame_index,
        "timestamp_ms": observation.timestamp_ms,
        "raw_x": observation.raw_x,
        "raw_y": observation.raw_y,
        "compensated_x": observation.compensated_x,
        "compensated_y": observation.compensated_y,
        "bbox_x": observation.raw_x - observation.w / 2.0,
        "bbox_y": observation.raw_y - observation.h / 2.0,
        "bbox_width": observation.w,
        "bbox_height": observation.h,
        "confidence": observation.confidence,
        "visible": observation.visible,
        "tracking_source": observation.tracking_source,
        "appearance_score": observation.appearance_score,
        "foreground_contrast": observation.foreground_contrast,
        "motion_score": observation.motion_score,
        "appearance_model_updated": observation.appearance_model_updated,
    }


def _gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _identity_affine() -> np.ndarray:
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix).astype(np.float32)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)


def _lk_params() -> dict[str, Any]:
    return {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }


def _timestamp_ms(frame_index: int, fps: float) -> int:
    return int(round(frame_index * 1000.0 / max(fps, 1e-6)))


def _safe_stem(value: str) -> str:
    sanitized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._")
    return sanitized or "video"
