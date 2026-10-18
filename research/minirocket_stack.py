"""Train a two-margin meta classifier on source-group out-of-fold predictions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

if __package__:
    from .minirocket_framescale import package_data, scale_windows, decision_margins as scale_margins
else:
    from minirocket_framescale import package_data, scale_windows, decision_margins as scale_margins

VERSION = "trajectory-sequence-dual-scale-stack-1.0.0"


def member_margins(model, x, scales):
    from research.minirocket_inference import decision_margins as original
    return np.column_stack([original(model["baseline"], x),
                            scale_margins(model["framescale"], scale_windows(x, scales))])


def decision_margins(model, x, scales):
    if (model["contract_version"] != VERSION or model["classes"] != ["bird", "drone"]
            or model["meta"].classes_.tolist() != ["bird", "drone"]):
        raise ValueError("stack model contract mismatch")
    values = member_margins(model, x, scales)
    return model["meta"].decision_function(model["meta_scaler"].transform(values))


class StackMiniRocketClassifier:
    def __init__(self, path):
        self.model = joblib.load(path)
        decision_margins(self.model, np.zeros((1, 4, 60), dtype=np.float32), [.0025])

    def predict(self, track):
        from ai_server.services.classifier import SequencePrediction
        from research.trajectory_sequence import window_track, SequenceConfig
        try:
            windows, rejected = window_track(track.model_dump(mode="json"), SequenceConfig())
        except (ValueError, KeyError, TypeError) as error:
            return SequencePrediction("uncertain", None, "invalid_input", str(error))
        if not windows:
            return SequencePrediction("uncertain", None, "insufficient_observation",
                                      window_rejections=dict(Counter(row["reason"] for row in rejected)))
        scores = decision_margins(self.model, np.stack([row["X"] for row in windows]),
                                  [row["normalization_scale"] for row in windows])
        mean = float(scores.mean())
        return SequencePrediction("drone" if mean >= 0 else "bird", mean,
                                  window_scores=scores.tolist(),
                                  window_starts_s=[row["relative_start_s"] for row in windows],
                                  window_rejections=dict(Counter(row["reason"] for row in rejected)))


def run(branch, package, output, baseline_path, framescale_path):
    sys.path.insert(0, str(branch))
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from ai_server.services.sequence_model import (load_package, fit_representation, transform,
        balanced_group_weights, aggregate_scores, metrics, RANDOM_STATE, decision_scores)
    output.mkdir(parents=True, exist_ok=False)
    _, cohorts, validation = load_package(package)
    train = cohorts["real_only"]
    metadata = pd.read_csv(package / "real_metadata.csv")
    scaled = package_data(train, metadata)
    baseline, physical = joblib.load(baseline_path), joblib.load(framescale_path)
    received_hash = hashlib.sha256((package / "dataset_manifest.json").read_bytes()).hexdigest()
    if baseline["package_manifest_sha256"] != physical["package_manifest_sha256"] or baseline["package_manifest_sha256"] != received_hash:
        raise ValueError("members use different training packages")
    oof = np.full((len(train["X"]), 2), np.nan)
    folds = np.full(len(train["X"]), -1)
    for fold, (fit, held) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE).split(train["X"], train["y"], train["group_id"])):
        if set(train["group_id"][fit]) & set(train["group_id"][held]):
            raise ValueError("source group crosses OOF fold")
        folds[held] = fold
        weights = balanced_group_weights(train["y"][fit], train["group_id"][fit])
        # Member alphas are frozen from the reproduced models, not selected on pilot labels.
        for column, (values, alpha) in enumerate(((train["X"], baseline["alpha"]), (scaled["X"], physical["alpha"]))):
            rocket, scaler = fit_representation(values[fit])
            ridge = RidgeClassifier(alpha=alpha).fit(transform(rocket, scaler, values[fit]), train["y"][fit], sample_weight=weights)
            oof[held, column] = ridge.decision_function(transform(rocket, scaler, values[held]))
    if not np.isfinite(oof).all() or np.any(folds < 0):
        raise ValueError("incomplete OOF coverage")
    table = pd.DataFrame({"sample_id": train["sample_id"], "track_id": [s.split(":w")[0] for s in train["sample_id"]],
                          "group_id": train["group_id"], "label": train["y"], "fold": folds,
                          "baseline_margin": oof[:, 0], "framescale_margin": oof[:, 1]})
    tracks = table.groupby(["group_id", "track_id", "label", "fold"], as_index=False)[["baseline_margin", "framescale_margin"]].mean()
    values = tracks[["baseline_margin", "framescale_margin"]].to_numpy()
    scaler = StandardScaler().fit(values)
    meta = LogisticRegression(C=1.0, random_state=RANDOM_STATE).fit(
        scaler.transform(values), tracks.label, sample_weight=balanced_group_weights(tracks.label, tracks.group_id))
    model = {key: baseline[key] for key in ("classes", "channels", "alpha", "train_groups", "train_windows", "package_manifest_sha256")}
    model.update(baseline=baseline, framescale=physical, meta=meta, meta_scaler=scaler,
                 contract_version=VERSION, contract_id="dual-scale-stack-v1",
                 input_contract_version=baseline["contract_version"], input_contract_id=baseline["contract_id"],
                 validation_used_for_fit=False, arm="real_only_dual_scale_stack",
                 meta_tracks=len(tracks), member_alpha_policy="frozen reproduced member alphas; OOF used for meta training only",
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    x_base = decision_scores(baseline, validation["X"])
    x_scale = scale_margins(physical, package_data(validation, metadata)["X"])
    scores = meta.decision_function(scaler.transform(np.column_stack([x_base, x_scale])))
    _, val_tracks, val_groups = aggregate_scores(validation, scores)
    joblib.dump(model, output / "stack.joblib")
    table.to_csv(output / "oof_windows.csv", index=False)
    tracks.to_csv(output / "oof_tracks.csv", index=False)
    val_tracks.to_csv(output / "validation_tracks.csv", index=False)
    result = {"validation_track_metrics": metrics(val_tracks), "validation_group_metrics": metrics(val_groups),
              "meta_train_tracks": len(tracks), "meta_train_groups": len(set(tracks.group_id)),
              "test_evaluated": False, "pilot_used_for_fit": False,
              "meta_coefficient": meta.coef_.tolist(), "meta_intercept": meta.intercept_.tolist(),
              "oof_training_metrics_are_not_an_independent_estimate": True}
    (output / "status.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--framescale", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.branch_root.resolve(), a.package.resolve(), a.output.resolve(), a.baseline.resolve(), a.framescale.resolve()), indent=2))
