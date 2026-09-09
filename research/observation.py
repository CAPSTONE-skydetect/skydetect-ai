"""Stateful observation surrogate; parameter ranges require real-A calibration."""
from dataclasses import asdict, dataclass

import numpy as np

if __package__:
    from .parameters import OBSERVATION_PRIORS
else:
    from parameters import OBSERVATION_PRIORS


@dataclass(frozen=True)
class NoiseConfig:
    jitter_px: float = OBSERVATION_PRIORS["jitter_px"]
    bbox_log_std: float = OBSERVATION_PRIORS["bbox_log_std"]
    dropout_rate: float = OBSERVATION_PRIORS["dropout_rate"]
    drift_probability: float = OBSERVATION_PRIORS["drift_probability"]
    burst_probability: float = OBSERVATION_PRIORS["burst_probability"]
    camera_probability: float = OBSERVATION_PRIORS["camera_probability"]

    def __post_init__(self):
        if not np.all(np.isfinite(list(asdict(self).values()))):
            raise ValueError("Noise parameters must be finite")
        if self.jitter_px < 0 or self.bbox_log_std < 0:
            raise ValueError("Noise magnitudes must be nonnegative")
        for value in (self.dropout_rate, self.drift_probability,
                      self.burst_probability, self.camera_probability):
            if not 0 <= value <= 1:
                raise ValueError("Probabilities must lie in [0, 1]")


def drift_offset(frame, profile):
    if not profile["enabled"] or frame < profile["start"]:
        return np.zeros(2)
    start, peak, end = profile["start"], profile["peak"], profile["end"]
    if frame <= peak:
        u = np.clip((frame - start) / max(peak - start, 1), 0., 1.)
    elif profile["recover"]:
        u = np.clip((end - frame) / max(end - peak, 1), 0., 1.)
    else:
        u = 1.
    return np.asarray(profile["offset_px"]) * (u * u * (3 - 2 * u))


class ObservationModel:
    def __init__(self, camera, rng, enabled=True, config=None):
        self.camera, self.rng, self.enabled = camera, rng, enabled
        self.config = config or NoiseConfig()

    def apply(self, truth, fps):
        if not truth:
            raise ValueError("Observation input is empty")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        c, rng, cfg = self.camera, self.rng, self.config
        n = len(truth)
        initial = next((x for x in truth if x is not None), None)
        initial_size = (np.array([initial["w"] * c.width, initial["h"] * c.height])
                        if initial else np.array([4., 2.]))
        start = int(rng.uniform(.15, .4) * n)
        peak = max(start + 1, int(rng.uniform(.45, .65) * n))
        end = max(peak + 1, int(rng.uniform(.75, .95) * n))
        drift = dict(enabled=bool(self.enabled and rng.random() < cfg.drift_probability),
                     start=start, peak=peak, end=end, recover=bool(rng.random() < .7),
                     offset_px=(rng.normal(size=2) * initial_size[0] * .3).tolist(),
                     duration_seconds=(end - start) / fps)
        drift["persists_to_end"] = bool(drift["enabled"] and not drift["recover"])
        if drift["persists_to_end"]:
            drift["duration_seconds"] = (n - start) / fps
        camera = dict(enabled=bool(self.enabled and rng.random() < cfg.camera_probability),
                      coordinate_space="post_cmc_residual_observation",
                      pan_px=rng.uniform(-5., 5., 2).tolist(),
                      shake_px=rng.uniform(.05, .35, 2).tolist(),
                      shake_hz=rng.uniform(.4, 2.5, 2).tolist(),
                      phase=rng.uniform(0., 2 * np.pi, 2).tolist(),
                      roll_end_rad=float(rng.uniform(-.0015, .0015)),
                      raw_zoom_end=float(rng.uniform(.9, 1.1)),
                      residual_zoom_fraction=float(rng.uniform(.02, .15)))
        burst = None
        if self.enabled and n > 12 and rng.random() < cfg.burst_probability:
            length = int(rng.integers(3, min(20, n // 4) + 1))
            first = int(rng.integers(1, n - length))
            burst = [first, first + length - 1]
        low_start = int(rng.uniform(.1, .65) * n)
        low_end = min(n - 1, low_start + max(1, int(rng.uniform(.1, .25) * n)))
        low_segment = [low_start, low_end] if self.enabled and rng.random() < .5 else None
        # ROI bias is separate from optical dimensions, shared over the whole track.
        roi_scale = float(rng.uniform(*OBSERVATION_PRIORS["roi_scale"])) if self.enabled else 1.
        base_size = np.maximum(1., initial_size * roi_scale) if self.enabled else initial_size
        current_size = base_size.copy()
        jitter = np.zeros(2)
        size_noise = np.zeros(2)
        histories, debug = [], []
        lost = 0
        previous = None
        reacquisition = np.zeros(2)
        for frame, optical in enumerate(truth):
            reason = None
            if optical is None:
                reason = "out_of_view"
            movement = (0. if optical is None or previous is None else
                        float(np.linalg.norm((np.array([optical["cx"], optical["cy"]])
                                              - previous) * [c.width, c.height])))
            size = initial_size if optical is None else np.array([optical["w"] * c.width, optical["h"] * c.height])
            difficulty = float(np.clip(.3 * movement / max(size[0], 1.)
                                      + .3 * max(0., 1. - min(size) / 3.), 0., 1.))
            if optical is not None:
                previous = np.array([optical["cx"], optical["cy"]])
            if self.enabled and reason is None:
                if burst and burst[0] <= frame <= burst[1]:
                    reason = "occlusion"
                elif rng.random() < min(.8, cfg.dropout_rate * (1 + 2 * difficulty)):
                    reason = "dropout"
            if reason:
                lost += 1
                debug.append(dict(frame_index=frame, visible=False, reason=reason,
                                  tracking_source="prediction", conf=.12))
                continue
            offset = drift_offset(frame, drift)
            if lost:
                reacquisition = rng.normal(0, cfg.jitter_px * min(lost, 6), 2) if self.enabled else np.zeros(2)
            reacquisition *= np.exp(-1 / fps / .2)
            center = np.array([optical["cx"] * c.width, optical["cy"] * c.height])
            if camera["enabled"]:
                time = frame / fps
                phase = np.asarray(camera["phase"])
                shake = np.asarray(camera["shake_px"]) * (
                    np.sin(2 * np.pi * np.asarray(camera["shake_hz"]) * time + phase) - np.sin(phase))
                progress = frame / max(n - 1, 1)
                theta = camera["roll_end_rad"] * progress
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                origin = np.array([c.width / 2, c.height / 2])
                raw_scale = 1 + (camera["raw_zoom_end"] - 1) * progress
                scale = 1 + (raw_scale - 1) * camera["residual_zoom_fraction"]
                center = origin + scale * (rotation @ (center - origin))
                center += np.asarray(camera["pan_px"]) * progress + shake
                # A stabilizes centers but retains the raw tracker bbox dimensions.
                size = size * raw_scale
            if self.enabled:
                jitter_rho, size_rho = .55 ** (30 / fps), .8 ** (30 / fps)
                jitter = jitter_rho * jitter + np.sqrt(1 - jitter_rho ** 2) * rng.normal(
                    0., cfg.jitter_px * (1 + difficulty), 2)
                size_noise = size_rho * size_noise + np.sqrt(1 - size_rho ** 2) * rng.normal(0., cfg.bbox_log_std, 2)
                desired_size = size * roi_scale * np.exp(size_noise)
                desired_size = np.clip(desired_size, np.maximum(1., .65 * base_size), 2.2 * base_size)
                alpha = 1 - .85 ** (30 / fps)
                current_size += alpha * (desired_size - current_size)
                center += jitter + offset + reacquisition
            else:
                current_size = size
            state = "manual_roi" if not histories else ("appearance" if lost else "klt")
            if lost and difficulty > .5:
                state = "motion"
            quality = np.clip(1 - .4 * difficulty - .25 * np.linalg.norm(offset) / max(size[0], 1.)
                              - .08 * lost, .1, 1.)
            if low_segment and low_segment[0] <= frame <= low_segment[1]:
                quality *= .65
            conf = 1. if not self.enabled or state == "manual_roi" else float(np.clip(
                quality + rng.normal(0., .025), .15, {"klt":.99, "appearance":.90, "motion":.86}[state]))
            obs = dict(frame_index=int(frame), timestamp_ms=int(round(frame * 1000 / fps)),
                       cx=float(center[0] / c.width), cy=float(center[1] / c.height),
                       w=float(current_size[0] / c.width), h=float(current_size[1] / c.height), conf=conf)
            if not self.enabled:
                obs = dict(optical)
            if not c.visible(obs):
                lost += 1
                debug.append(dict(frame_index=frame, visible=False, reason="observed_out_of_view",
                                  tracking_source="prediction", conf=.12))
                continue
            histories.append(obs)
            debug.append(dict(frame_index=frame, visible=True, reason=None, tracking_source=state,
                              conf=conf, drift_px=float(np.linalg.norm(offset)), difficulty=difficulty))
            lost = 0
        mean_conf = float(np.mean([o["conf"] for o in histories])) if histories else 0.
        missing = 1 - len(histories) / n
        quality = dict(num_points=len(histories), mean_conf=mean_conf, missing_ratio=missing,
                       track_stability="good" if missing < .1 and mean_conf > .8 else
                                       ("fair" if missing < .5 else "poor"))
        return histories, dict(noise_config=asdict(cfg), calibrated=False, enabled=self.enabled,
                               camera_motion=camera, tracking_drift=drift, burst_range=burst,
                               low_conf_range=low_segment, roi_scale=roi_scale,
                               attempted_frame_count=n, quality=quality, states=debug)
