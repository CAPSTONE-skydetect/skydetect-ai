"""Preserve apparent image-width motion scale; not world-speed estimation."""

from collections import Counter
import joblib
import numpy as np

VERSION = "trajectory-sequence-frame-scale-1.0.0"
GAIN = 400.0


def scale_windows(x, scales):
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (len(x),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("invalid per-window scales")
    output = x.copy()
    output[:, :2] = (x[:, :2].astype(np.float64) * scales[:, None, None] * GAIN).astype(np.float32)
    output[:, 2:] = np.diff(output[:, :2], axis=2, prepend=output[:, :2, :1])
    return output


def package_data(data, metadata):
    meta = metadata.set_index("sample_id")
    if not meta.index.is_unique:
        raise ValueError("duplicate metadata sample IDs")
    rows = meta.loc[data["sample_id"]]
    if not np.array_equal(rows.label.to_numpy(), data["y"]) or not np.array_equal(rows.source_group_id.to_numpy(), data["group_id"]):
        raise ValueError("metadata label/source mismatch")
    result = {key: value.copy() for key, value in data.items()}
    result["X"] = scale_windows(data["X"], rows.normalization_scale.to_numpy())
    return result


def decision_margins(model, x):
    from research.minirocket_inference import decision_margins as original
    from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig
    if (model["contract_version"] != VERSION or model["scale_gain"] != GAIN
            or model["input_contract_version"] != CONTRACT_VERSION
            or model["input_contract_id"] != SequenceConfig().fingerprint):
        raise ValueError("frame-scale model contract mismatch")
    compatible = dict(model, contract_version=CONTRACT_VERSION, contract_id=SequenceConfig().fingerprint)
    return original(compatible, x)


class FrameScaleMiniRocketClassifier:
    def __init__(self, path):
        self.model = joblib.load(path)
        decision_margins(self.model, np.zeros((1, 4, 60), dtype=np.float32))

    def predict(self, track):
        from ai_server.services.classifier import SequencePrediction
        from research.trajectory_sequence import window_track, SequenceConfig
        windows, rejected = window_track(track.model_dump(mode="json"), SequenceConfig())
        if not windows:
            return SequencePrediction("uncertain", None, "insufficient_observation")
        x = scale_windows(np.stack([row["X"] for row in windows]), [row["normalization_scale"] for row in windows])
        scores = decision_margins(self.model, x)
        score = float(scores.mean())
        return SequencePrediction("drone" if score >= 0 else "bird", score,
                                  window_scores=scores.tolist(),
                                  window_starts_s=[row["relative_start_s"] for row in windows],
                                  window_rejections=dict(Counter(row["reason"] for row in rejected)))
