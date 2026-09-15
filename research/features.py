"""Time-aware, bbox-independent research trajectory features."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.signal import resample_poly, savgol_coeffs, savgol_filter

from . import FEATURE_VERSION

CORE_COLUMNS = ["speed_median", "acceleration_median", "turn_rate_median", "tortuosity"]
VARIABILITY_COLUMNS = ["speed_cv", "acceleration_p95", "turn_rate_p95",
                       "curvature_cv", "heading_change_ratio"]
FEATURE_COLUMNS = ["speed_median", "speed_cv", "acceleration_median", "acceleration_p95",
                   "turn_rate_median", "turn_rate_p95", "curvature_cv", "tortuosity",
                   "heading_change_ratio"]


@dataclass(frozen=True)
class FeatureConfig:
    target_fps: float = 30.
    smoothing_seconds: float = .3
    max_missing_seconds: float = .25
    max_missing_fraction: float = .5
    min_points: int = 5
    min_duration_seconds: float = .2
    minimum_motion_px_s: float = 1.
    turn_threshold_rad_s: float = np.pi / 6
    heading_snr: float = 3.
    tortuosity_cap: float = 100.

    def __post_init__(self):
        if not np.all(np.isfinite(list(asdict(self).values()))):
            raise ValueError("Feature configuration must be finite")
        if self.target_fps <= 0 or self.smoothing_seconds < 0 or self.max_missing_seconds < 0:
            raise ValueError("Invalid time configuration")
        if not 0 <= self.max_missing_fraction <= 1 or self.min_points < 3:
            raise ValueError("Invalid point or missingness threshold")
        if min(self.min_duration_seconds, self.minimum_motion_px_s,
               self.turn_threshold_rad_s, self.heading_snr) < 0 or self.tortuosity_cap < 1:
            raise ValueError("Feature thresholds must be nonnegative")

    @property
    def fingerprint(self):
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _window(length, step, config):
    window = int(round(config.smoothing_seconds / step)) | 1
    return min(window, length if length % 2 else length - 1)


def _smooth(values, step, config):
    window = _window(len(values), step, config)
    if window < 5:
        return values.copy()
    return savgol_filter(values, window, 2, axis=0, mode="interp")


def _resample(values, intervals, source_intervals):
    # Preserve constant/linear motion exactly instead of introducing the small
    # polyphase gain ripple into a large absolute pixel-coordinate offset.
    trend = np.linspace(values[0], values[-1], len(values))
    residual = resample_poly(values-trend, intervals, source_intervals, axis=0, padtype="line")
    return residual[:intervals+1]+np.linspace(values[0], values[-1], intervals+1)


def _weighted_quantile(values, weights, quantile):
    if not len(values):
        return 0.
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(quantile, cumulative / cumulative[-1], values[order]))


def extract_features(history, image_width, image_height, fps=None, config=None):
    """Return an explicit rejection, never replace an invalid feature with zero.

    Timestamp milliseconds take precedence over fps; fps is only an explicit
    fallback when all timestamps are absent. Long gaps separate segments.
    Positions are converted to pixels, but bbox dimensions never enter a feature
    formula. Speeds therefore describe apparent image-plane motion, not physical
    world speed or range-corrected motion.
    """
    cfg = config or FeatureConfig()
    result = dict(feature_version=FEATURE_VERSION, feature_config_id=cfg.fingerprint,
                  feature_status="rejected", features=None, quality={}, reasons=[])
    try:
        if not np.all(np.isfinite([image_width, image_height])) or min(image_width, image_height) <= 0:
            raise ValueError("Image dimensions must be explicit and positive")
        if len(history) < cfg.min_points:
            raise ValueError("insufficient_points")
        frame = np.asarray([p["frame_index"] for p in history], dtype=float)
        values = np.asarray([[p[k] for k in ("cx", "cy", "w", "h", "conf")] for p in history], dtype=float)
        if not np.all(np.isfinite(frame)) or not np.all(np.isfinite(values)):
            raise ValueError("nonfinite_input")
        if np.any(frame < 0) or np.any(frame != np.floor(frame)) or np.any(np.diff(frame) <= 0):
            raise ValueError("frame_indices_must_be_strictly_increasing_integers")
        if np.any(values < 0) or np.any(values > 1) or np.any(values[:, 2:4] <= 0):
            raise ValueError("invalid_normalized_observation")
        has_time = [p.get("timestamp_ms") is not None for p in history]
        if all(has_time):
            time = np.asarray([p["timestamp_ms"] for p in history], dtype=float) / 1000
            clock = "timestamp_ms"
        elif any(has_time):
            raise ValueError("partial_timestamps")
        elif fps is not None and np.isfinite(fps) and fps > 0:
            time = frame / fps
            clock = "frame_index_and_explicit_fps"
        else:
            raise ValueError("timestamps_or_explicit_fps_required")
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0) or time[0] < 0:
            raise ValueError("timestamps_must_be_finite_and_strictly_increasing")
        time -= time[0]
        delta = np.diff(time)
        nominal = float((time[-1]-time[0]) / (frame[-1]-frame[0]))
        missing_fraction = float(1 - len(frame) / (frame[-1] - frame[0] + 1))
        missing_seconds = np.maximum(0., delta - nominal)
        boundaries = np.flatnonzero(missing_seconds > cfg.max_missing_seconds + .001) + 1
        result["quality"] = dict(observed_points=len(frame), duration_seconds=float(time[-1]),
                                 nominal_fps=1 / nominal, clock=clock,
                                 missing_fraction=missing_fraction,
                                 longest_missing_seconds=float(max(missing_seconds, default=0)),
                                 long_gap_count=len(boundaries), mean_conf=float(values[:, 4].mean()),
                                 boundary_fraction=float(np.mean(np.any((values[:, :2] == 0) | (values[:, :2] == 1), axis=1))))
        if missing_fraction > cfg.max_missing_fraction:
            raise ValueError("excessive_missing_fraction")
        if time[-1] < cfg.min_duration_seconds:
            raise ValueError("insufficient_duration")
        if time[-1] * cfg.target_fps > 100000:
            raise ValueError("resampling_limit_exceeded")
        segments = np.split(np.arange(len(frame)), boundaries)
        velocity, accel, turns, curvature = [], [], [], []
        tortuosity, lengths, retained = [], [], 0
        direction_candidates, direction_supported = 0, 0
        direction_duration = 0.
        for indices in segments:
            t, points = time[indices], values[indices]
            if len(t) < cfg.min_points or t[-1] - t[0] < cfg.min_duration_seconds:
                continue
            intervals = max(2, int(round((t[-1] - t[0]) * cfg.target_fps)))
            grid = np.linspace(t[0], t[-1], intervals + 1)
            step = float(grid[1] - grid[0])
            source_intervals = max(intervals, int(round((t[-1]-t[0])/nominal)))
            if source_intervals > 100000:
                raise ValueError("source_resampling_limit_exceeded")
            source_grid = np.linspace(t[0], t[-1], source_intervals+1)
            xy = np.column_stack([np.interp(source_grid, t, points[:, i]) for i in range(2)])
            if source_intervals > intervals:
                # Anti-alias BEFORE reducing sample rate; post-decimation smoothing
                # cannot undo high-frequency jitter aliased into slow motion.
                xy = _resample(xy, intervals, source_intervals)
            raw_xy = xy * [image_width, image_height]
            xy = _smooth(raw_xy, step, cfg)
            pixel_velocity = np.diff(xy, axis=0) / step
            direction_floor = 0.
            window = _window(len(xy), step, cfg)
            if window >= 5:
                # Differentiate the fitted local polynomial directly rather than
                # amplifying residual high frequencies by differencing twice.
                v_at_points = savgol_filter(raw_xy, window, 2, deriv=1, delta=step, axis=0, mode="interp")
                pixel_velocity = (v_at_points[:-1]+v_at_points[1:])/2
                residual = raw_xy-xy
                sigma = np.median(np.abs(residual-np.median(residual, axis=0)), axis=0)/.67448975
                gain = np.linalg.norm(savgol_coeffs(window, 2, deriv=1, delta=step))
                # Heuristic support gate, not a calibrated confidence interval.
                direction_floor = cfg.heading_snr*np.linalg.norm(sigma)*gain
            speed = np.linalg.norm(pixel_velocity, axis=1)
            acceleration = np.abs(np.diff(speed)) / step
            angle = np.arctan2(pixel_velocity[:, 1], pixel_velocity[:, 0])
            angle_change = np.abs((np.diff(angle) + np.pi) % (2 * np.pi) - np.pi)
            angular_rate = angle_change / step
            arc_step = (speed[:-1] + speed[1:]) * step / 2
            speed_floor = max(cfg.minimum_motion_px_s, direction_floor)
            valid_turn = (speed[:-1] >= speed_floor) & (speed[1:] >= speed_floor)
            reliable = np.linalg.norm(pixel_velocity, axis=1) > direction_floor
            valid_turn &= reliable[:-1] & reliable[1:]
            direction_candidates += len(valid_turn)
            direction_supported += int(valid_turn.sum())
            direction_duration += len(angular_rate)*step
            angular_rate = angular_rate[valid_turn]
            discrete_curvature = angle_change[valid_turn] / arc_step[valid_turn]
            velocity.append((speed, np.full(len(speed), step)))
            accel.append((acceleration, np.full(len(acceleration), step)))
            turns.append((angular_rate, np.full(len(angular_rate), step)))
            curvature.append((discrete_curvature, np.full(len(discrete_curvature), step)))
            distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            displacement = float(np.linalg.norm(xy[-1] - xy[0]))
            if distance <= 1e-8:
                ratio = 1.
            else:
                ratio = min(distance / max(displacement, 1e-8), cfg.tortuosity_cap)
            tortuosity.append(ratio)
            lengths.append(float(t[-1] - t[0]))
            retained += len(indices)
        if not velocity:
            raise ValueError("no_usable_contiguous_segment")

        def combine(parts):
            return np.concatenate([v for v, _ in parts]), np.concatenate([w for _, w in parts])

        def mean_std(parts):
            v, w = combine(parts)
            if not len(v):
                return 0., 0.
            mean = float(np.average(v, weights=w))
            return mean, float(np.sqrt(np.average((v - mean) ** 2, weights=w)))

        speeds, speed_weights = combine(velocity)
        accelerations, acceleration_weights = combine(accel)
        rates, weights = combine(turns)
        curvatures, curvature_weights = combine(curvature)
        speed_mean, speed_std = mean_std(velocity)
        curvature_mean, curvature_std = mean_std(curvature)
        curvature_cv = (curvature_std / curvature_mean
                        if curvature_mean > np.sqrt(np.finfo(float).eps) else 0.)
        features = dict(
                        speed_median=_weighted_quantile(speeds, speed_weights, .5),
                        speed_cv=speed_std / max(speed_mean, 1e-12),
                        acceleration_median=_weighted_quantile(accelerations, acceleration_weights, .5),
                        acceleration_p95=_weighted_quantile(accelerations, acceleration_weights, .95),
                        turn_rate_median=_weighted_quantile(rates, weights, .5),
                        turn_rate_p95=_weighted_quantile(rates, weights, .95),
                        curvature_cv=curvature_cv,
                        tortuosity=float(np.average(tortuosity, weights=lengths)),
                        heading_change_ratio=float(weights[rates > cfg.turn_threshold_rad_s].sum()/direction_duration) if direction_duration else 0.,
                        )
        if not np.all(np.isfinite(list(features.values()))):
            raise ValueError("nonfinite_features")
        result.update(features=features, feature_status="accepted")
        result["quality"].update(usable_segments=len(velocity), retained_points=retained,
                                  heading_valid_fraction=direction_supported/max(direction_candidates, 1),
                                  retained_duration_seconds=sum(lengths),
                                  training_length_group="short" if sum(lengths) < 2. else "standard")
    except (ValueError, TypeError, KeyError, IndexError) as error:
        result["reasons"].append(str(error))
    return result
