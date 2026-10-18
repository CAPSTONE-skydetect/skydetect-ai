"""Causal fusion of a position anchor and an independent motion estimate."""

import numpy as np


class ComplementaryCenter:
    def __init__(self, time_constant_seconds: float = 0.25):
        if time_constant_seconds <= 0:
            raise ValueError("time constant must be positive")
        self.time_constant = time_constant_seconds
        self.bias = None
        self.previous_time = None

    def update(self, anchor, motion, timestamp_ms: int, *, max_disagreement: float):
        anchor = np.asarray(anchor, dtype=float)
        dt = None if self.previous_time is None else (timestamp_ms - self.previous_time) / 1000
        if dt is not None and dt <= 0:
            raise ValueError("timestamps must strictly increase")
        self.previous_time = timestamp_ms
        if motion is None or not np.all(np.isfinite(motion)):
            self.bias = None
            return anchor.copy()
        motion = np.asarray(motion, dtype=float)
        difference = anchor - motion
        if np.linalg.norm(difference) > max_disagreement:
            self.bias = None
            return anchor.copy()
        if self.bias is None or dt is None or dt > 0.25:
            self.bias = difference.copy()
        else:
            alpha = -np.expm1(-dt / self.time_constant)
            self.bias += alpha * (difference - self.bias)
        return motion + self.bias
