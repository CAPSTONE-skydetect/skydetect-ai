"""Four-second context reconstructed from overlapping real package windows.

Reconstruction inverts saved per-window normalization, not original raw A
observations. No new source frames are claimed. Missing grid samples reject a
long window; short tracks use the frozen two-second baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


VERSION = "trajectory-sequence-context-4s-1.0.0"


def validate_context(x):
    if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (4, 120) or not np.isfinite(x).all():
        raise ValueError("Expected finite float32 (N,4,120)")
    if not np.allclose(x[:, 2:, 0], 0) or not np.allclose(x[:, 2:, 1:], np.diff(x[:, :2], axis=2), atol=1e-6):
        raise ValueError("context displacement channels disagree")
    return x


def fit_representation(x):
    from ai_server.services.sequence_model import make_rocket
    from sklearn.preprocessing import StandardScaler
    rocket = make_rocket()
    scaler = StandardScaler(with_mean=False).fit(rocket.fit_transform(validate_context(x)))
    return rocket, scaler


def transform(rocket, scaler, x):
    return scaler.transform(rocket.transform(validate_context(x))).astype(np.float64)


def reconstruct(data, metadata):
    from research.trajectory_sequence import normalize_window, SequenceConfig
    config = SequenceConfig(window_seconds=4.0)
    meta = metadata.set_index("sample_id")
    if not meta.index.is_unique:
        raise ValueError("duplicate package metadata sample IDs")
    groups = {}
    for index, sample_id in enumerate(data["sample_id"]):
        row = meta.loc[str(sample_id)]
        if row.label != data["y"][index] or row.source_group_id != data["group_id"][index]:
            raise ValueError("metadata label/group does not match NPZ")
        parent = str(sample_id).split(":w", 1)[0]
        groups.setdefault(parent, []).append((index, row))
    x, y, group_ids, sample_ids = [], [], [], []
    maximum_disagreement = 0.
    for parent, entries in groups.items():
        if len({str(data["y"][i]) for i, _ in entries}) != 1 or len({str(data["group_id"][i]) for i, _ in entries}) != 1:
            raise ValueError("parent has inconsistent label/source group")
        origin = min(float(row.window_start_s) for _, row in entries)
        support_end = max(float(row.window_end_s) for _, row in entries)
        grid = {}
        for index, row in entries:
            offset = (float(row.window_start_s) - origin) * 30
            if abs(offset - round(offset)) > 1e-5:
                raise ValueError("package windows do not share a 30 Hz grid")
            points = data["X"][index, :2].T.astype(np.float64) * float(row.normalization_scale)
            points += [float(row.center_x), float(row.center_y)]
            for j, point in enumerate(points):
                grid.setdefault(round(offset) + j, []).append(point)
        for values in grid.values():
            maximum_disagreement = max(maximum_disagreement, float(np.ptp(values, axis=0).max()))
        if maximum_disagreement > 1e-6:
            raise ValueError("overlapping package windows disagree after inverse normalization")
        dense = {key: np.mean(values, axis=0) for key, values in grid.items()}
        count = max(0, int(np.floor((support_end-origin-4+1e-9)))+1)
        for start in range(count):
            indices = list(range(start * 30, start * 30 + 120))
            if not all(index in dense for index in indices):
                continue
            points = np.stack([dense[index] for index in indices])
            window = normalize_window(points, config)[0]
            x.append(window)
            y.append(str(data["y"][entries[0][0]]))
            group_ids.append(str(data["group_id"][entries[0][0]]))
            sample_ids.append(f"{parent}:w{start:04d}")
    if not x:
        raise ValueError("no fully supported four-second windows")
    return {"X": np.stack(x), "y": np.array(y), "group_id": np.array(group_ids),
            "sample_id": np.array(sample_ids)}, {"max_overlap_disagreement_width_normalized": maximum_disagreement,
                                                "tracks": len(set(s.split(":w")[0] for s in sample_ids)),
                                                "windows": len(x), "input_tracks": len(groups)}


def run(branch, package, output, baseline_path):
    sys.path.insert(0, str(branch))
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import RidgeClassifier
    from ai_server.services.sequence_model import (load_package, ALPHAS, RANDOM_STATE, balanced_group_weights,
        aggregate_scores, metrics, decision_scores)
    output.mkdir(parents=True, exist_ok=False)
    _, cohorts, validation = load_package(package)
    metadata = pd.read_csv(package / "real_metadata.csv")
    train, train_audit = reconstruct(cohorts["real_only"], metadata)
    held, validation_audit = reconstruct(validation, metadata)
    cv = []
    for fold, (fit, test) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE).split(train["X"], train["y"], train["group_id"])):
        assert not set(train["group_id"][fit]) & set(train["group_id"][test])
        if any(set(train["y"][indices]) != {"bird", "drone"} for indices in (fit, test)):
            raise ValueError("context fold missing a class")
        rocket, scaler = fit_representation(train["X"][fit])
        z_fit = transform(rocket, scaler, train["X"][fit])
        z_test = transform(rocket, scaler, train["X"][test])
        weights = balanced_group_weights(train["y"][fit], train["group_id"][fit])
        part = {key: values[test] for key, values in train.items()}
        for alpha in ALPHAS:
            ridge = RidgeClassifier(alpha=alpha).fit(z_fit, train["y"][fit], sample_weight=weights)
            score = metrics(aggregate_scores(part, ridge.decision_function(z_test))[2])["macro_f1"]
            cv.append({"fold": fold, "alpha": alpha, "group_macro_f1": score})
    alpha = float(pd.DataFrame(cv).groupby("alpha").group_macro_f1.mean().idxmax())
    rocket, scaler = fit_representation(train["X"])
    ridge = RidgeClassifier(alpha=alpha).fit(transform(rocket, scaler, train["X"]), train["y"],
                    sample_weight=balanced_group_weights(train["y"], train["group_id"]))
    context_scores = ridge.decision_function(transform(rocket, scaler, held["X"]))
    baseline = joblib.load(baseline_path)
    base_scores = decision_scores(baseline, validation["X"])
    _, base_tracks, _ = aggregate_scores(validation, base_scores)
    _, context_tracks, _ = aggregate_scores(held, context_scores)
    lookup = context_tracks.set_index("track_id").decision.to_dict()
    combined = base_tracks.copy()
    combined["context_used"] = combined.track_id.isin(lookup)
    combined["decision"] = [lookup.get(row.track_id, row.decision) for row in combined.itertuples()]
    grouped = combined.groupby(["group_id", "label"], as_index=False).decision.mean()
    model = {"rocket": rocket, "scaler": scaler, "classifier": ridge, "baseline": baseline,
             "classes": ["bird", "drone"], "contract_version": VERSION,
             "window_seconds": 4., "samples": 120, "alpha": alpha,
             "validation_used_for_fit": False, "alpha_cv": cv,
             "package_manifest_sha256": hashlib.sha256((package / "dataset_manifest.json").read_bytes()).hexdigest(),
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    joblib.dump(model, output / "context4.joblib")
    combined.to_csv(output / "validation_tracks.csv", index=False)
    result = {"train": train_audit, "validation_reconstruction": validation_audit,
              "alpha": alpha, "train_cv": cv,
              "validation_track_metrics": metrics(combined), "validation_group_metrics": metrics(grouped),
              "context_tracks": int(combined.context_used.sum()), "total_tracks": len(combined),
              "short_track_policy": "frozen two-second baseline fallback", "test_evaluated": False}
    (output / "status.json").write_text(json.dumps(result, indent=2))
    return result


def predict(model, track):
    from research.trajectory_sequence import window_track, SequenceConfig
    from research.minirocket_inference import decision_margins
    if model["contract_version"] != VERSION or model["classifier"].classes_.tolist() != ["bird", "drone"]:
        raise ValueError("context model contract mismatch")
    windows, rejected = window_track(track, SequenceConfig(window_seconds=4.))
    used = "context4"
    if windows:
        x = np.stack([row["X"] for row in windows])
        z = transform(model["rocket"], model["scaler"], x)
        scores = model["classifier"].decision_function(z)
    else:
        used = "baseline2_fallback"
        windows, rejected = window_track(track, SequenceConfig())
        if not windows:
            return {"prediction": "uncertain", "margin": None, "mode": "insufficient_support"}
        scores = decision_margins(model["baseline"], np.stack([row["X"] for row in windows]))
    margin = float(scores.mean())
    return {"prediction": "drone" if margin >= 0 else "bird", "margin": margin, "mode": used,
            "window_margins": scores.tolist(), "windows": len(scores), "rejections": rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.branch_root.resolve(), args.package.resolve(), args.output.resolve(), args.baseline.resolve()), indent=2))
