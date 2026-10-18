"""Experimental C4-pooled MiniRocket representation and explicit inference adapter."""

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

if __package__:
    from .minirocket_focus import rotate_channels
else:
    from minirocket_focus import rotate_channels


ORBIT_VERSION = "trajectory-sequence-c4pool-1.0.0"


def orbit_features(rocket, x):
    if not len(x):
        raise ValueError("at least one window required")
    joined = np.concatenate([rotate_channels(x, k) for k in range(4)])
    features = rocket.transform(joined).astype(np.float64)
    return features.reshape(4, len(x), -1).mean(axis=0)


def fit_representation(x):
    from ai_server.services.sequence_model import make_rocket, validate_windows
    validate_windows(x)
    rocket = make_rocket()
    rocket.fit(np.concatenate([rotate_channels(x, k) for k in range(4)]))
    scaler = StandardScaler(with_mean=False).fit(orbit_features(rocket, x))
    return rocket, scaler


def transform(rocket, scaler, x):
    return scaler.transform(orbit_features(rocket, x)).astype(np.float64)


def decision_margins(model, x):
    from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig
    from ai_server.services.sequence_model import validate_windows
    if (model["contract_version"] != ORBIT_VERSION
            or model["input_contract_version"] != CONTRACT_VERSION
            or model["input_contract_id"] != SequenceConfig().fingerprint
            or model["classes"] != ["bird", "drone"]
            or model["classifier"].classes_.tolist() != ["bird", "drone"]):
        raise ValueError("Orbit model contract mismatch")
    return model["classifier"].decision_function(transform(model["rocket"], model["scaler"], validate_windows(x)))


class OrbitMiniRocketClassifier:
    """Explicitly experimental: the standard deployment rejects its new contract."""
    def __init__(self, path):
        self.model = joblib.load(Path(path))
        decision_margins(self.model, np.zeros((1, 4, 60), dtype=np.float32))

    def predict(self, track):
        from ai_server.services.classifier import SequencePrediction
        from research.trajectory_sequence import window_track, SequenceConfig
        windows, rejected = window_track(track.model_dump(mode="json"), SequenceConfig())
        if not windows:
            return SequencePrediction("uncertain", None, "insufficient_observation")
        scores = decision_margins(self.model, np.stack([row["X"] for row in windows]))
        score = float(scores.mean())
        return SequencePrediction("drone" if score >= 0 else "bird", score,
                                  window_scores=scores.tolist(),
                                  window_starts_s=[row["relative_start_s"] for row in windows],
                                  window_rejections=dict(Counter(row["reason"] for row in rejected)))
