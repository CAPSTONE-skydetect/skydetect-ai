"""Versioned, label-independent A-track conversion for sequence classifiers."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.signal import butter, sosfiltfilt


CONTRACT_VERSION = "trajectory-sequence-1.0.0"
CHANNELS = ("q_x", "q_y", "d_x", "d_y")


@dataclass(frozen=True)
class SequenceConfig:
    fps: float = 30.0
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    scale_floor: float = 0.0025
    max_missing_seconds: float = 0.1
    max_missing_fraction: float = 0.1
    clock_tolerance_seconds: float = 0.001

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.isfinite(values).all():
            raise ValueError("Non-finite sequence configuration")
        if min(self.fps, self.window_seconds, self.stride_seconds, self.scale_floor) <= 0:
            raise ValueError("Rates, durations and scale floor must be positive")
        if not 0 <= self.max_missing_fraction < 1 or self.max_missing_seconds < 0:
            raise ValueError("Invalid missing-data limits")
        if self.clock_tolerance_seconds < 0:
            raise ValueError("Invalid clock tolerance")
        if self.samples < 9 or not np.isclose(self.fps * self.window_seconds, self.samples):
            raise ValueError("Window must have an integer number of at least 9 samples")

    @property
    def samples(self):
        return int(round(self.fps * self.window_seconds))

    @property
    def fingerprint(self):
        data = dict(version=CONTRACT_VERSION, channels=CHANNELS, config=asdict(self))
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def normalize_window(points, config):
    """One spatial scale for both axes; derivatives keep the original time scale."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (config.samples, 2) or not np.isfinite(points).all():
        raise ValueError("Expected finite (samples, 2) coordinates")
    center = np.median(points, axis=0)
    radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.9))
    scale = max(radius, config.scale_floor)
    q = (points - center) / scale
    d = np.diff(q, axis=0, prepend=q[:1])
    return np.column_stack((q, d)).T.astype(np.float32), dict(
        center_x=float(center[0]), center_y=float(center[1]),
        radius_p90=radius, normalization_scale=scale,
        scale_floor_applied=radius < config.scale_floor,
    )


def _read_observations(track):
    if track.get("stabilization", {}).get("applied") is not True:
        raise ValueError("Expected explicitly stabilized A export")
    width, height = float(track["processed_width"]), float(track["processed_height"])
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("Invalid processed dimensions")
    history = track["history"]
    values = np.asarray([[p["timestamp_ms"], p["frame_index"], p["cx"], p["cy"]]
                         for p in history], dtype=np.float64)
    if len(values) < 2 or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite observations")
    time, frames = values[:, 0] / 1000.0, values[:, 1]
    dt, df = np.diff(time), np.diff(frames)
    if time[0] < 0 or frames[0] < 0 or np.any(dt <= 0) or np.any(df <= 0):
        raise ValueError("Timestamps and frame indices must strictly increase")
    if np.any(frames != np.floor(frames)):
        raise ValueError("Frame indices must be integers")
    if np.any(values[:, 2:] < 0) or np.any(values[:, 2:] > 1):
        raise ValueError("Exported centers must be within [0, 1]")
    period = float(np.median(dt / df))
    # Large inconsistent frame clocks need explicit handling, not silent time warping.
    if np.max(np.abs(dt / df - period)) > max(0.002, 0.1 * period):
        raise ValueError("Inconsistent frame/timestamp clock")
    points = values[:, 2:].copy()
    points[:, 1] *= height / width
    boundary = np.any((values[:, 2:] <= 0) | (values[:, 2:] >= 1), axis=1)
    return time, points, period, boundary


def window_track(track, config=None):
    """Return accepted windows and every rejection, without modifying the track.

    All starts are relative to the first observed time, not to a class or event.
    Two seconds of observation support are required for 60 half-open samples.
    """
    config = config or SequenceConfig()
    time, points, period, boundary = _read_observations(track)
    source_fps = 1.0 / period
    if source_fps < 15.0 - 0.01:
        raise ValueError("Source FPS below supported 15 Hz")
    dt = np.diff(time)
    gap = dt > period * 1.5
    missing = np.where(gap, np.maximum(dt - period, 0), 0)
    long_gap = missing > config.max_missing_seconds + config.clock_tolerance_seconds
    segments = np.split(np.arange(len(time)), np.flatnonzero(long_gap) + 1)
    prepared = []
    for indices in segments:
        ts, ps = time[indices], points[indices]
        filtered = False
        filter_error = False
        if source_fps > config.fps * 1.01 and len(ts) > 1:
            # Filter each uninterrupted segment before lowering the sample rate.
            regular = ts[0] + np.arange(int(np.floor((ts[-1] - ts[0]) / period)) + 1) * period
            dense = np.column_stack([np.interp(regular, ts, ps[:, axis]) for axis in (0, 1)])
            sos = butter(4, 0.4 * config.fps, fs=source_fps, output="sos")
            try:
                dense = sosfiltfilt(sos, dense, axis=0)
            except ValueError:
                filter_error = True
            if not filter_error:
                ts, ps, filtered = regular, dense, True
        prepared.append((ts, ps, filtered, filter_error))

    accepted, rejected = [], []
    duration = float(time[-1] - time[0])
    count = max(0, int(np.floor((duration - config.window_seconds + 1e-9)
                              / config.stride_seconds)) + 1)
    if not count:
        return [], [dict(window_index=-1, reason="short_track", duration_seconds=duration)]
    for i in range(count):
        start = float(time[0] + i * config.stride_seconds)
        end = start + config.window_seconds
        row = dict(window_index=i, window_start_s=start, window_end_s=end,
                   relative_start_s=float(i * config.stride_seconds), source_fps=source_fps)
        overlapping = (time[:-1] < end) & (time[1:] > start)
        if np.any(long_gap & overlapping):
            rejected.append(dict(row, reason="long_gap"))
            continue
        overlap = np.maximum(0, np.minimum(time[1:], end) - np.maximum(time[:-1] + period, start))
        fraction = float(np.sum(np.where(gap, overlap, 0)) / config.window_seconds)
        if fraction > config.max_missing_fraction + 1e-9:
            rejected.append(dict(row, reason="missing_fraction", missing_fraction=fraction))
            continue
        grid = start + np.arange(config.samples) / config.fps
        segment = next((s for s in prepared if s[0][0] <= start + 1e-9
                        and s[0][-1] >= grid[-1] - 1e-9), None)
        if segment is None or segment[3]:
            rejected.append(dict(row, reason="resampling_support"))
            continue
        ts, ps, filtered, _ = segment
        sample = np.column_stack([np.interp(grid, ts, ps[:, axis]) for axis in (0, 1)])
        x, normalization = normalize_window(sample, config)
        selected = (time >= start) & (time < end)
        boundary_fraction = float(np.mean(boundary[selected])) if selected.any() else 0.0
        flags = []
        if fraction > 0:
            flags.append("short_gap_interpolated")
        if normalization["scale_floor_applied"]:
            flags.append("low_spatial_extent")
        if boundary_fraction:
            flags.append("boundary_contact")
        if source_fps < config.fps * 0.99:
            flags.append("upsampled_no_new_information")
        accepted.append(dict(row, X=x, **normalization, missing_fraction=fraction,
                             max_missing_seconds=float(np.max(missing[overlapping], initial=0)),
                             boundary_fraction=boundary_fraction, anti_alias_applied=filtered,
                             quality_flags=";".join(flags)))
    return accepted, rejected
