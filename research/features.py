"""Time-aware research feature contract, deliberately distinct from runtime v1/v2."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.signal import resample_poly, savgol_coeffs, savgol_filter

from . import FEATURE_VERSION

MOTION_COLUMNS = ["v_mean", "v_std", "a_mean", "turn_rate_mean", "turn_rate_p95",
                  "heading_change_ratio", "straightness", "stationary_ratio"]
BBOX_COLUMNS = ["bbox_area_mean", "bbox_area_cv", "bbox_scale_rate_std"]
FEATURE_COLUMNS = MOTION_COLUMNS + BBOX_COLUMNS


@dataclass(frozen=True)
class FeatureConfig:
    target_fps: float = 30.
    smoothing_seconds: float = .3
    max_missing_seconds: float = .25
    max_missing_fraction: float = .5
    min_points: int = 5
    min_duration_seconds: float = .2
    stationary_speed: float = .2
    turn_threshold_rad_s: float = np.pi / 6
    heading_snr: float = 3.

    def __post_init__(self):
        if not np.all(np.isfinite(list(asdict(self).values()))):
            raise ValueError("Feature configuration must be finite")
        if self.target_fps <= 0 or self.smoothing_seconds < 0 or self.max_missing_seconds < 0:
            raise ValueError("Invalid time configuration")
        if not 0 <= self.max_missing_fraction <= 1 or self.min_points < 3:
            raise ValueError("Invalid point or missingness threshold")
        if min(self.min_duration_seconds, self.stationary_speed, self.turn_threshold_rad_s, self.heading_snr) < 0:
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


def extract_features(history, image_width, image_height, fps=None, config=None):
    """Return an explicit rejection, never replace an invalid feature with zero.

    Timestamp milliseconds take precedence over fps; fps is only an explicit
    fallback when all timestamps are absent. Long gaps separate segments.
    Width-normalization uses pixels for BOTH axes, preserving aspect ratio.
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
        velocity, accel, turns, areas, scale_rates = [], [], [], [], []
        straightness, lengths, retained = [], [], 0
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
            log_size = np.column_stack([np.interp(source_grid, t, np.log(points[:, i])) for i in (2, 3)])
            if source_intervals > intervals:
                # Anti-alias BEFORE reducing sample rate; post-decimation smoothing
                # cannot undo high-frequency jitter aliased into slow motion.
                xy = _resample(xy, intervals, source_intervals)
                log_size = _resample(log_size, intervals, source_intervals)
            raw_xy = xy * [image_width, image_height]
            xy = _smooth(raw_xy, step, cfg)
            size = np.exp(_smooth(log_size, step, cfg)) * [image_width, image_height]
            width = size[:, 0]
            pixel_velocity = np.diff(xy, axis=0) / step
            pixel_acceleration = np.diff(pixel_velocity, axis=0) / step
            direction_floor = 0.
            window = _window(len(xy), step, cfg)
            if window >= 5:
                # Differentiate the fitted local polynomial directly rather than
                # amplifying residual high frequencies by differencing twice.
                v_at_points = savgol_filter(raw_xy, window, 2, deriv=1, delta=step, axis=0, mode="interp")
                pixel_velocity = (v_at_points[:-1]+v_at_points[1:])/2
                pixel_acceleration = savgol_filter(raw_xy, window, 2, deriv=2, delta=step, axis=0, mode="interp")[1:-1]
                residual = raw_xy-xy
                sigma = np.median(np.abs(residual-np.median(residual, axis=0)), axis=0)/.67448975
                gain = np.linalg.norm(savgol_coeffs(window, 2, deriv=1, delta=step))
                # Heuristic support gate, not a calibrated confidence interval.
                direction_floor = cfg.heading_snr*np.linalg.norm(sigma)*gain
            speed = np.linalg.norm(pixel_velocity, axis=1) / ((width[:-1] + width[1:]) / 2)
            # Differentiate vector velocity BEFORE dividing by width: zoom alone
            # must not masquerade as physical image acceleration.
            acceleration = np.linalg.norm(pixel_acceleration, axis=1) / width[1:-1]
            angle = np.arctan2(pixel_velocity[:, 1], pixel_velocity[:, 0])
            angular_rate = np.abs((np.diff(angle) + np.pi) % (2 * np.pi) - np.pi) / step
            valid_turn = (speed[:-1] >= cfg.stationary_speed) & (speed[1:] >= cfg.stationary_speed)
            reliable = np.linalg.norm(pixel_velocity, axis=1) > direction_floor
            valid_turn &= reliable[:-1] & reliable[1:]
            direction_candidates += len(valid_turn)
            direction_supported += int(valid_turn.sum())
            direction_duration += len(valid_turn)*step
            angular_rate = angular_rate[valid_turn]
            velocity.append((speed, np.full(len(speed), step)))
            accel.append((acceleration, np.full(len(acceleration), step)))
            turns.append((angular_rate, np.full(len(angular_rate), step)))
            area = size[:, 0] * size[:, 1] / image_width / image_height
            areas.append((area, np.full(len(area), step)))
            scale_rates.append((np.diff(np.log(width)) / step, np.full(intervals, step)))
            distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            straightness.append(float(np.linalg.norm(xy[-1] - xy[0])) / distance if distance > 1e-8 else 0.)
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

        v_mean, v_std = mean_std(velocity)
        area_mean, area_std = mean_std(areas)
        rates, weights = combine(turns)
        speeds, speed_weights = combine(velocity)
        # Weighted quantile avoids giving short segments disproportionate weight.
        order = np.argsort(rates)
        p95 = float(np.interp(.95, np.cumsum(weights[order]) / weights.sum(), rates[order])) if len(rates) else 0.
        features = dict(v_mean=v_mean, v_std=v_std, a_mean=mean_std(accel)[0],
                        turn_rate_mean=mean_std(turns)[0], turn_rate_p95=p95,
                        heading_change_ratio=float(weights[rates > cfg.turn_threshold_rad_s].sum()/direction_duration) if direction_duration else 0.,
                        straightness=float(np.average(straightness, weights=lengths)),
                        stationary_ratio=float(np.average(speeds < cfg.stationary_speed, weights=speed_weights)),
                        bbox_area_mean=area_mean, bbox_area_cv=area_std / max(area_mean, 1e-12),
                        bbox_scale_rate_std=mean_std(scale_rates)[1])
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
