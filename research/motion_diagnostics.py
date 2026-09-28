"""Screen-motion events and residuals; not physical commands or isolated noise."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, detrend, find_peaks, savgol_filter, sosfiltfilt, welch

from .trajectory_sequence import SequenceConfig, _read_observations


MOTION_DIAGNOSTIC_VERSION = "screen-motion-1.0.1"


def duration_group_weights(table):
    duration = table.duration_s.to_numpy(dtype=float)
    totals = table.groupby("group_id").duration_s.transform("sum").to_numpy(dtype=float)
    weights = duration/totals
    return weights/weights.sum()


@dataclass(frozen=True)
class MotionConfig:
    fps: float = 30.
    trend_seconds: float = .3
    minimum_event_seconds: float = .3
    low_speed_floor: float = .0015
    noise_multiplier: float = 2.
    reverse_angle_deg: float = 135.
    context_seconds: float = .3
    turn_rate_deg_s: float = 20.
    minimum_turn_deg: float = 30.
    maximum_pause_bridge_seconds: float = 3.

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Motion settings must be finite")
        if min(self.fps, self.trend_seconds, self.minimum_event_seconds,
               self.context_seconds, self.noise_multiplier) <= 0:
            raise ValueError("Motion time/rate settings must be positive")
        if self.low_speed_floor < 0 or not 90 <= self.reverse_angle_deg <= 180:
            raise ValueError("Invalid motion thresholds")


def resample_segments(track, sequence_config=None):
    cfg = sequence_config or SequenceConfig()
    time, points, period, _ = _read_observations(track)
    if 1/period < 15-.01:
        raise ValueError("Source FPS below supported 15 Hz")
    long_gap = np.diff(time)-period > cfg.max_missing_seconds+cfg.clock_tolerance_seconds
    segments = []
    for indices in np.split(np.arange(len(time)), np.flatnonzero(long_gap)+1):
        ts, ps = time[indices], points[indices]
        if len(ts) < 2 or ts[-1]-ts[0] < .6:
            continue
        native = ts[0]+np.arange(int(np.floor((ts[-1]-ts[0])/period))+1)*period
        dense = np.column_stack([np.interp(native, ts, ps[:, axis]) for axis in (0, 1)])
        if 1/period > cfg.fps*1.01:
            try:
                dense = sosfiltfilt(butter(4, .4*cfg.fps, fs=1/period, output="sos"), dense, axis=0)
            except ValueError:
                continue
        grid = ts[0]+np.arange(int(np.floor((ts[-1]-ts[0])*cfg.fps))+1)/cfg.fps
        sample = np.column_stack([np.interp(grid, native, dense[:, axis]) for axis in (0, 1)])
        nearest = np.searchsorted(ts, grid).clip(1, len(ts)-1)
        distance = np.minimum(np.abs(grid-ts[nearest]), np.abs(grid-ts[nearest-1]))
        segments.append(dict(time=grid, points=sample,
                             interpolated=distance > .55*period,
                             source_period=period))
    return segments


def runs(mask):
    padded = np.r_[False, np.asarray(mask, bool), False].astype(int)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts, ends))


def angle_between(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a@b/denominator, -1, 1)))) if denominator > 1e-12 else 0.


def analyze_segment(time, points, config=None, interpolated=None):
    cfg = config or MotionConfig()
    time, p = np.asarray(time, float), np.asarray(points, float)
    if p.shape != (len(time), 2) or len(time) < 9 or not np.isfinite(p).all():
        raise ValueError("Need at least nine finite screen positions")
    if not np.allclose(np.diff(time), 1/cfg.fps, atol=1e-6):
        raise ValueError("Motion input must be regularly sampled")
    interpolated = np.zeros(len(time), bool) if interpolated is None else np.asarray(interpolated, bool)
    if interpolated.shape != time.shape:
        raise ValueError("Interpolation mask mismatch")
    length = max(5, int(round(cfg.trend_seconds*cfg.fps)) | 1)
    length = min(length, len(time) if len(time) % 2 else len(time)-1)
    trend = savgol_filter(p, length, 2, axis=0)
    residual = p-trend
    mad = np.median(np.abs(residual-np.median(residual, axis=0)), axis=0)*1.4826
    residual_sigma = float(np.linalg.norm(mad))
    threshold = max(cfg.low_speed_floor, cfg.noise_multiplier*residual_sigma/cfg.context_seconds)
    half = max(1, int(round(cfg.context_seconds*cfg.fps/2)))
    indices = np.arange(len(p))
    left, right = np.maximum(0, indices-half), np.minimum(len(p)-1, indices+half)
    velocity = (trend[right]-trend[left])/((right-left)/cfg.fps)[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    edge = (left == 0) | (right == len(p)-1)
    radius = max(half, length//2)
    near_gap = np.convolve(interpolated.astype(int), np.ones(radius*2+1), mode="same") > 0
    unreliable = edge | near_gap
    extent = float(np.linalg.norm(np.ptp(p, axis=0)))
    unresolved = extent <= max(4*residual_sigma, .5/1920)
    low = (speed <= threshold) & ~unreliable
    moving = (speed > threshold*1.5) & ~unreliable
    events = []

    def event(kind, a, b, **extra):
        events.append(dict(kind=kind, start_s=float(time[a]),
                           end_s=float(time[b-1]+1/cfg.fps), duration_s=(b-a)/cfg.fps,
                           includes_interpolation=bool(interpolated[a:b].any()),
                           left_censored=bool(a <= radius+1),
                           right_censored=bool(b >= len(p)-radius-1),
                           interpretation="screen_motion_candidate", **extra))

    minimum = int(np.ceil(cfg.minimum_event_seconds*cfg.fps))
    low_runs = [(a, b) for a, b in runs(low) if b-a >= minimum]
    if not unresolved:
        for a, b in low_runs:
            event("low_motion", a, b)
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    turns = np.zeros(len(p))
    turns[1:] = np.angle(np.exp(1j*np.diff(heading)))
    support = moving & np.r_[False, moving[:-1]]
    turn_rate = turns*cfg.fps
    for sign in (-1, 1):
        active = support & (turn_rate*sign >= np.radians(cfg.turn_rate_deg_s))
        for a, b in runs(active):
            total = float(np.degrees(turns[a:b].sum()))
            if b-a >= minimum and abs(total) >= cfg.minimum_turn_deg:
                event("turn", a, b, signed_turn_deg=total)
    context = max(2, int(round(cfg.context_seconds*cfg.fps)))
    angles = np.zeros(len(p))
    for i in range(context, len(p)-context):
        before, after = velocity[i-context:i], velocity[i:i+context]
        if moving[i-context:i].mean() < .75 or moving[i:i+context].mean() < .75:
            continue
        a, b = before.mean(axis=0), after.mean(axis=0)
        if min(np.linalg.norm(a), np.linalg.norm(b)) <= threshold*1.5:
            continue
        angles[i] = angle_between(a, b)
    peaks, _ = find_peaks(angles, height=cfg.reverse_angle_deg, distance=max(context*2, 1))
    for i in peaks:
        event("reversal", i-context, i+context, change_angle_deg=float(angles[i]),
              via_low_motion=False)
    for a, b in low_runs:
        if (b-a)/cfg.fps > cfg.maximum_pause_bridge_seconds or a < context or b+context > len(p):
            continue
        if moving[a-context:a].mean() < .75 or moving[b:b+context].mean() < .75:
            continue
        angle = angle_between(velocity[a-context:a].mean(axis=0), velocity[b:b+context].mean(axis=0))
        if angle >= cfg.reverse_angle_deg:
            event("reversal", a-context, b+context, change_angle_deg=angle, via_low_motion=True)
    if unresolved:
        events = []
    position_power = float(np.sum(residual**2))
    residual_acf = lambda lag: float(np.sum(residual[:-lag]*residual[lag:])/position_power) if position_power > 1e-18 else 0.
    delta = np.diff(p, axis=0)
    fluctuation = detrend(delta, axis=0)
    frequencies, psd = welch(residual, fs=cfg.fps, nperseg=min(120, len(p)), axis=0)
    power = psd.sum(axis=1)
    velocity_power = float(np.sum(fluctuation**2))
    vel_acf3 = float(np.sum(fluctuation[:-3]*fluctuation[3:])/velocity_power) if velocity_power > 1e-18 else 0.
    summary = dict(duration_s=len(p)/cfg.fps, screen_span=extent,
                   residual_sigma_px=residual_sigma*1920, residual_rms_px=float(np.sqrt(np.mean(residual**2))*1920),
                   residual_acf1=residual_acf(1), residual_acf2=residual_acf(2),
                   residual_acf3=residual_acf(3), displacement_acf3=vel_acf3,
                   residual_band_3_10=float(power[(frequencies >= 3) & (frequencies <= 10)].sum()/max(power[1:].sum(), 1e-24)),
                   low_speed_threshold=threshold, unresolved=bool(unresolved),
                   unreliable_fraction=float(unreliable.mean()),
                   low_motion_fraction=float(low.mean()) if not unresolved else 0.,
                   reversal_per_minute=sum(e["kind"] == "reversal" for e in events)*60/(len(p)/cfg.fps),
                   turn_per_minute=sum(e["kind"] == "turn" for e in events)*60/(len(p)/cfg.fps))
    return dict(summary=summary, events=sorted(events, key=lambda e: e["start_s"]),
                time=time, points=p, trend=trend, residual=residual, speed=speed, threshold=threshold,
                heading=np.where(moving, heading, np.nan), interpolated=interpolated,
                frequencies=frequencies, residual_psd=power)
