"""Require repeated independent motion before overriding background-like flow."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionLockAudit:
    minimum_motion: float = .60
    previous_candidate: np.ndarray | None = None
    streak: int = 0

    def reset(self):
        self.previous_candidate = None
        self.streak = 0

    def observe(self, *, klt_center, background_center, candidate, motion_score,
                appearance_score, minimum_appearance, affine, size, pixel_scale=1.0):
        if klt_center is None or candidate is None:
            self.reset()
            return None
        klt = np.asarray(klt_center, dtype=np.float32)
        bg = np.asarray(background_center, dtype=np.float32)
        point = np.asarray(candidate, dtype=np.float32)
        background_residual = float(np.linalg.norm(klt - bg))
        independent_motion = float(np.linalg.norm(point - bg))
        displacement = float(np.linalg.norm(point - klt))
        if (not np.isfinite([*klt, *bg, *point, motion_score, appearance_score]).all()
                or motion_score < self.minimum_motion or appearance_score < minimum_appearance
                or background_residual > max(.75 * pixel_scale, .10 * min(size))
                or independent_motion < max(1.5 * pixel_scale, 2 * background_residual)
                or displacement < 1.5 * pixel_scale):
            self.reset()
            return None
        consistent = False
        if self.previous_candidate is not None:
            transported = np.asarray(affine) @ np.r_[self.previous_candidate, 1.]
            consistent = float(np.linalg.norm(point - transported)) <= max(4 * pixel_scale, .5 * max(size))
        self.streak = self.streak + 1 if consistent else 1
        self.previous_candidate = point.copy()
        return tuple(float(v) for v in point) if self.streak >= 2 else None
