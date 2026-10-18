"""Fixed equal-margin fusion of extent-normalized and image-width-scale models."""

from collections import Counter

import joblib
import numpy as np

if __package__:
    from .minirocket_framescale import scale_windows, decision_margins as scale_margins
else:
    from minirocket_framescale import scale_windows, decision_margins as scale_margins

VERSION = "trajectory-sequence-dual-scale-1.0.0"


def decision_margins(model, x, scales):
    from research.minirocket_inference import decision_margins as baseline_margins
    if model["contract_version"] != VERSION or model["weights"] != [.5, .5]:
        raise ValueError("dual-scale model contract mismatch")
    baseline = baseline_margins(model["baseline"], x)
    physical = scale_margins(model["framescale"], scale_windows(x, scales))
    return .5 * baseline + .5 * physical


class DualScaleMiniRocketClassifier:
    def __init__(self, path):
        self.model = joblib.load(path)
        decision_margins(self.model, np.zeros((1, 4, 60), dtype=np.float32), [.0025])

    def predict(self, track):
        from ai_server.services.classifier import SequencePrediction
        from research.trajectory_sequence import window_track, SequenceConfig
        windows, rejected = window_track(track.model_dump(mode="json"), SequenceConfig())
        if not windows:
            return SequencePrediction("uncertain", None, "insufficient_observation")
        scores = decision_margins(self.model, np.stack([row["X"] for row in windows]),
                                  [row["normalization_scale"] for row in windows])
        margin = float(scores.mean())
        return SequencePrediction("drone" if margin >= 0 else "bird", margin,
                                  window_scores=scores.tolist(),
                                  window_starts_s=[row["relative_start_s"] for row in windows],
                                  window_rejections=dict(Counter(row["reason"] for row in rejected)))
