"""C handoff example: trusted local model + NPZ or one stabilized A TrackSequence.

joblib is executable serialization. Only load a model from a trusted source.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

try:
    from .trajectory_sequence import CONTRACT_VERSION, SequenceConfig, window_track
except ImportError:
    from trajectory_sequence import CONTRACT_VERSION, SequenceConfig, window_track


def decision_margins(model, x):
    if model["contract_version"] != CONTRACT_VERSION or model["contract_id"] != SequenceConfig().fingerprint:
        raise ValueError("Model/preprocessing contract mismatch")
    if model["classes"] != ["bird", "drone"]:
        raise ValueError("Unexpected score sign")
    x = np.asarray(x)
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (4,60) or not np.isfinite(x).all():
        raise ValueError("Expected finite float32 (N,4,60)")
    if not np.allclose(x[:,2:,0],0) or not np.allclose(x[:,2:,1:],np.diff(x[:,:2],axis=2),atol=1e-6):
        raise ValueError("Displacement channels do not match coordinates")
    if not len(x):
        return np.empty(0)
    z = model["scaler"].transform(model["rocket"].transform(x)).astype(np.float64)
    return model["classifier"].decision_function(z)


def predict_track(model, track):
    windows, rejected = window_track(track,SequenceConfig())
    if not windows:
        return dict(status="insufficient_quality_or_duration",prediction=None,windows=0,rejections=rejected)
    margins = decision_margins(model,np.stack([w["X"] for w in windows]))
    average = float(margins.mean())
    return dict(status="predicted",prediction="drone" if average >= 0 else "bird",
                margin=average,windows=len(windows),window_margins=margins.tolist(),
                rejections=rejected,score_is_probability=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    if args.input.suffix == ".npz":
        with np.load(args.input,allow_pickle=False) as data:
            margins = decision_margins(model,data["X"])
        result = dict(window_margins=margins.tolist(),score_is_probability=False)
    else:
        result = predict_track(model,json.loads(args.input.read_text(encoding="utf-8-sig")))
    print(json.dumps(result,ensure_ascii=True,indent=2))
