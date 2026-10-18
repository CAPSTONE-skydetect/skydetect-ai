"""Reproduce MiniRocket and compare train-only rotation augmentation.

Uses the collaborator's package validator, model code and group metrics.
Held-out folds and validation always contain original, unaugmented windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .minirocket_focus import rotate_channels
else:
    from minirocket_focus import rotate_channels


def augment_training(data: dict, rotations: bool) -> dict:
    if not rotations:
        return {key: value.copy() for key, value in data.items()}
    result = {}
    for key, values in data.items():
        if key == "X":
            result[key] = np.concatenate([rotate_channels(values, turns) for turns in range(4)])
        elif key == "sample_id":
            result[key] = np.concatenate([np.array([str(value) + f":rotation{90*turns}" for value in values])
                                          for turns in range(4)])
        elif key == "sample_weight":
            result[key] = np.tile(values / 4, 4)
        else:
            result[key] = np.tile(values, 4)
    return result


def run(branch: Path, package: Path, output: Path, candidates=("baseline", "rotation90")) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = package / "dataset_manifest.json"
    if not manifest_path.is_file():
        result = {"status": "blocked", "fitting_performed": False,
                  "reason": "original_C_training_package_missing", "expected_package": str(package),
                  "required_files": ["dataset_manifest.json", "train_real_only.npz", "validation_real.npz",
                                     "train_real_plus_augmented.npz", "train_synthetic_only.npz",
                                     "real_metadata.csv", "source_inventory.csv"],
                  "request": "Provide the complete real_reference_comparison_v1 package, retaining its hashes and source groups."}
        (output / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    sys.path.insert(0, str(branch.resolve()))
    import joblib
    import pandas as pd
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from ai_server.services.sequence_model import (
        ALPHAS, RANDOM_STATE, CV_SPLITS, aggregate_scores, balanced_group_weights,
        decision_scores, fit_arm, fit_representation, load_package, metrics,
        transform, validate_windows,
    )
    import research.trajectory_sequence as sequence_module
    if Path(sequence_module.__file__).resolve() != (branch / "research/trajectory_sequence.py").resolve():
        raise ValueError("Run this file directly with the deployment Python to avoid mixing repository imports")

    manifest, cohorts, validation = load_package(package)
    train = cohorts["real_only"]
    original_validation = validation
    models = {}
    for name in candidates:
        if name not in {"baseline", "rotation90", "orbit", "framescale"}:
            raise ValueError("unknown candidate")
        train, validation = cohorts["real_only"], original_validation
        physical_scale = name == "framescale"
        if physical_scale:
            if __package__:
                from . import minirocket_framescale as framescale
            else:
                import minirocket_framescale as framescale
            metadata = pd.read_csv(package / "real_metadata.csv")
            train = framescale.package_data(train, metadata)
            validation = framescale.package_data(validation, metadata)
        rotation = name == "rotation90"
        pooling = name == "orbit"
        fit_rep, represent = fit_representation, transform
        if pooling:
            if __package__:
                from . import minirocket_orbit as orbit
            else:
                import minirocket_orbit as orbit
            fit_rep, represent = orbit.fit_representation, orbit.transform
        cv = []
        splitter = StratifiedGroupKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (fit, held) in enumerate(splitter.split(train["X"], train["y"], train["group_id"])):
            if set(train["group_id"][fit]) & set(train["group_id"][held]):
                raise ValueError("train group leaked into held fold")
            if any(set(train["y"][indices]) != {"bird", "drone"} for indices in (fit, held)):
                raise ValueError("CV fold missing a class")
            original = {key: values[fit] for key, values in train.items()}
            original["sample_weight"] = balanced_group_weights(original["y"], original["group_id"])
            fitting = augment_training(original, rotation)
            validate_windows(fitting["X"])
            held_data = {key: values[held] for key, values in train.items()}
            rocket, scaler = fit_rep(fitting["X"])
            z_fit = represent(rocket, scaler, fitting["X"])
            z_held = represent(rocket, scaler, held_data["X"])
            for alpha in ALPHAS:
                ridge = RidgeClassifier(alpha=alpha).fit(z_fit, fitting["y"], sample_weight=fitting["sample_weight"])
                groups = aggregate_scores(held_data, ridge.decision_function(z_held))[2]
                cv.append({"fold": fold, "alpha": alpha, "group_macro_f1": metrics(groups)["macro_f1"]})
        table = pd.DataFrame(cv)
        alpha = float(table.groupby("alpha").group_macro_f1.mean().idxmax())
        fitting = augment_training(train, rotation)
        if pooling:
            from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig, CHANNELS
            from ai_server.services.sequence_model import library_versions
            rocket, scaler = fit_rep(fitting["X"])
            ridge = RidgeClassifier(alpha=alpha).fit(represent(rocket, scaler, fitting["X"]),
                                                    fitting["y"], sample_weight=fitting["sample_weight"])
            model = dict(rocket=rocket, scaler=scaler, classifier=ridge, classes=ridge.classes_.tolist(),
                         contract_version=orbit.ORBIT_VERSION, contract_id="c4-pool-v1",
                         input_contract_version=CONTRACT_VERSION, input_contract_id=SequenceConfig().fingerprint,
                         channels=list(CHANNELS), alpha=alpha, train_windows=len(fitting["X"]),
                         train_groups=len(set(fitting["group_id"])), versions=library_versions(),
                         inference_adapter_required="research.minirocket_orbit.OrbitMiniRocketClassifier")
        else:
            model = fit_arm({"real_only": fitting}, "real_only", alpha)
        if physical_scale:
            from research.trajectory_sequence import CONTRACT_VERSION, SequenceConfig
            model.update(contract_version=framescale.VERSION, contract_id="frame-scale-v1",
                         input_contract_version=CONTRACT_VERSION, input_contract_id=SequenceConfig().fingerprint,
                         scale_gain=framescale.GAIN,
                         inference_adapter_required="research.minirocket_framescale.FrameScaleMiniRocketClassifier")
        model.update(arm="real_only_c4pool" if pooling else ("real_only_rotation90" if rotation else "real_only"),
                     package_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                     validation_used_for_fit=False, alpha_selection="3-fold train group CV, original held windows only",
                     alpha_cv=cv, original_train_windows=len(train["X"]),
                     augmentation={"rotation_degrees": [0, 90, 180, 270] if rotation else [0],
                                   "weights_preserve_parent_total": True, "validation_augmented": False})
        if physical_scale:
            model["arm"] = "real_only_frame_scale"
        scores = (framescale.decision_margins(model, validation["X"]) if physical_scale else
                  orbit.decision_margins(model, validation["X"]) if pooling else decision_scores(model, validation["X"]))
        windows, tracks, groups = aggregate_scores(validation, scores)
        joblib.dump(model, output / f"{name}.joblib")
        tracks.to_csv(output / f"{name}_validation_tracks.csv", index=False)
        windows.to_csv(output / f"{name}_validation_windows.csv", index=False)
        models[name] = {"alpha": alpha, "train_windows": len(fitting["X"]),
                        "validation_track_metrics": metrics(tracks),
                        "validation_group_metrics": metrics(groups), "train_cv": cv}
    result = {"status": "completed", "fitting_performed": True, "test_evaluated": False,
              "development_only": True, "validation_used_for_fit": False,
              "selection_policy": "review validation group/track metrics and class recall; independent holdout still required",
              "package_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "models": models}
    (output / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", choices=("baseline", "rotation90", "orbit", "framescale"), default=["baseline", "rotation90"])
    args = parser.parse_args()
    print(json.dumps(run(args.branch_root.resolve(), args.package.resolve(), args.output.resolve(), args.candidates), indent=2))
