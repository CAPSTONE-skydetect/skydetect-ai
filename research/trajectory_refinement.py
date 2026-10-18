"""Timestamp-aware trajectory refinement without inventing observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ai_server.schemas import TrackPoint, TrackSequence
from ai_server.utils.quality import build_track_quality


@dataclass(frozen=True, slots=True)
class RefinementResult:
    track: TrackSequence
    diagnostics: list[dict[str, Any]]
    mode: str
    segment_count: int


def refine_track(
    track: TrackSequence,
    *,
    mode: Literal["kf", "rts"] = "kf",
    adaptive_measurement: bool = False,
    observation_diagnostics: dict[int, dict[str, Any]] | None = None,
    process_acceleration_std: float = 18.0,
    measurement_std: float = 1.5,
    max_segment_gap_seconds: float = 0.25,
) -> RefinementResult:
    """Refine only observed frames and preserve every frame gap and timestamp."""
    if mode not in {"kf", "rts"}:
        raise ValueError("mode must be 'kf' or 'rts'")
    if process_acceleration_std <= 0 or measurement_std <= 0:
        raise ValueError("noise standard deviations must be positive")
    if max_segment_gap_seconds <= 0:
        raise ValueError("max_segment_gap_seconds must be positive")

    width = float(track.processed_width)
    height = float(track.processed_height)
    rows = observation_diagnostics or {}
    segments: list[list[TrackPoint]] = []
    for point in track.history:
        if not segments:
            segments.append([point])
            continue
        dt = (point.timestamp_ms - segments[-1][-1].timestamp_ms) / 1000.0
        if dt <= 0:
            raise ValueError("track timestamps must be strictly increasing")
        if dt > max_segment_gap_seconds:
            segments.append([point])
        else:
            segments[-1].append(point)

    output: list[TrackPoint] = []
    diagnostics: list[dict[str, Any]] = []
    for segment_id, segment in enumerate(segments):
        points, trace = _refine_segment(
            segment,
            width=width,
            height=height,
            mode=mode,
            adaptive_measurement=adaptive_measurement,
            rows=rows,
            process_acceleration_std=process_acceleration_std,
            measurement_std=measurement_std,
            segment_id=segment_id,
        )
        output.extend(points)
        diagnostics.extend(trace)

    attempted = None
    if track.quality is not None and track.quality.missing_ratio < 1.0:
        attempted = int(round(len(track.history) / max(1e-12, 1.0 - track.quality.missing_ratio)))
    quality = build_track_quality(output, attempted_frame_count=attempted)
    payload = track.model_dump(exclude={"history", "quality"})
    return RefinementResult(
        track=TrackSequence(**payload, history=output, quality=quality),
        diagnostics=diagnostics,
        mode=mode,
        segment_count=len(segments),
    )


def _refine_segment(
    points: list[TrackPoint],
    *,
    width: float,
    height: float,
    mode: str,
    adaptive_measurement: bool,
    rows: dict[int, dict[str, Any]],
    process_acceleration_std: float,
    measurement_std: float,
    segment_id: int,
) -> tuple[list[TrackPoint], list[dict[str, Any]]]:
    observations = np.asarray([[p.cx * width, p.cy * height] for p in points], dtype=float)
    states: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    predicted_covariances: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    sigmas: list[float] = []

    initial_sigma = _measurement_sigma(
        rows.get(points[0].frame_index), measurement_std, adaptive_measurement,
    )
    state = np.array([observations[0, 0], observations[0, 1], 0.0, 0.0], dtype=float)
    covariance = np.diag([initial_sigma**2, initial_sigma**2, 400.0, 400.0])
    states.append(state.copy())
    covariances.append(covariance.copy())
    predicted_states.append(state.copy())
    predicted_covariances.append(covariance.copy())
    transitions.append(np.eye(4))
    sigmas.append(initial_sigma)

    identity = np.eye(4)
    observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    for index in range(1, len(points)):
        dt = (points[index].timestamp_ms - points[index - 1].timestamp_ms) / 1000.0
        transition = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        noise_map = np.array([
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ])
        process_covariance = (
            process_acceleration_std**2 * noise_map @ noise_map.T
            + np.eye(4) * 1e-9
        )
        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + process_covariance
        sigma = _measurement_sigma(
            rows.get(points[index].frame_index), measurement_std, adaptive_measurement,
        )
        measurement_covariance = np.eye(2) * sigma**2
        innovation_covariance = (
            observation_matrix @ predicted_covariance @ observation_matrix.T
            + measurement_covariance
        )
        gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(
            innovation_covariance,
        )
        innovation = observations[index] - observation_matrix @ predicted_state
        state = predicted_state + gain @ innovation
        update = identity - gain @ observation_matrix
        covariance = (
            update @ predicted_covariance @ update.T
            + gain @ measurement_covariance @ gain.T
        )
        covariance = (covariance + covariance.T) / 2.0
        states.append(state.copy())
        covariances.append(covariance.copy())
        predicted_states.append(predicted_state.copy())
        predicted_covariances.append(predicted_covariance.copy())
        transitions.append(transition)
        sigmas.append(sigma)

    selected_states = [item.copy() for item in states]
    selected_covariances = [item.copy() for item in covariances]
    if mode == "rts" and len(points) > 1:
        for index in range(len(points) - 2, -1, -1):
            smoother_gain = (
                covariances[index]
                @ transitions[index + 1].T
                @ np.linalg.inv(predicted_covariances[index + 1])
            )
            selected_states[index] = states[index] + smoother_gain @ (
                selected_states[index + 1] - predicted_states[index + 1]
            )
            selected_covariances[index] = covariances[index] + smoother_gain @ (
                selected_covariances[index + 1] - predicted_covariances[index + 1]
            ) @ smoother_gain.T
            selected_covariances[index] = (
                selected_covariances[index] + selected_covariances[index].T
            ) / 2.0

    refined: list[TrackPoint] = []
    trace: list[dict[str, Any]] = []
    for point, measurement, state_value, covariance_value, sigma in zip(
        points, observations, selected_states, selected_covariances, sigmas,
    ):
        x = float(state_value[0])
        y = float(state_value[1])
        refined.append(point.model_copy(update={
            "cx": max(0.0, min(1.0, x / width)),
            "cy": max(0.0, min(1.0, y / height)),
        }))
        trace.append({
            "frame_index": point.frame_index,
            "timestamp_ms": point.timestamp_ms,
            "segment_id": segment_id,
            "measurement_x": float(measurement[0]),
            "measurement_y": float(measurement[1]),
            "refined_x": x,
            "refined_y": y,
            "velocity_x": float(state_value[2]),
            "velocity_y": float(state_value[3]),
            "measurement_sigma": sigma,
            "position_variance_x": float(covariance_value[0, 0]),
            "position_variance_y": float(covariance_value[1, 1]),
            "clamped": not (0.0 <= x <= width and 0.0 <= y <= height),
        })
    return refined, trace


def _measurement_sigma(
    row: dict[str, Any] | None,
    base: float,
    adaptive: bool,
) -> float:
    if not adaptive or row is None:
        return base
    source = str(row.get("tracking_source", "unknown"))
    source_factor = {
        "manual_roi": 0.8,
        "klt": 1.0,
        "motion": 1.8,
        "appearance": 2.2,
    }.get(source, 2.5)
    try:
        fb_error = max(0.0, float(row.get("forward_backward_error", 0.0)))
    except (TypeError, ValueError):
        fb_error = 0.0
    fb_factor = 1.0 + min(fb_error, 4.0) / 4.0
    try:
        psr = max(0.0, float(row.get("appearance_psr", 0.0)))
    except (TypeError, ValueError):
        psr = 0.0
    ambiguity_factor = 1.0 if source == "klt" else 1.0 + 1.0 / max(psr, 1.0)
    return float(max(0.5, min(8.0, base * source_factor * fb_factor * ambiguity_factor)))
