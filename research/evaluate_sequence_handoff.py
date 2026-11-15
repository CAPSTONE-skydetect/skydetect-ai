"""Frozen MiniRocket/Ridge ablation and historically inspected real-test replay.

Run in the isolated requirements-minirocket.txt environment after preparation.
No raw video is included in the C handoff package.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .build_real_sequence_dataset import digest, file_hash
from .io import write_json
from .trajectory_sequence import SequenceConfig, CONTRACT_VERSION, CHANNELS, window_track
from .verify_real_track import load_tracks


LABELS = ["bird", "drone"]
ALPHAS = (.1, 1., 10., 100.)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in ("X", "y", "group_id", "sample_id")}
    x, y = data["X"], data["y"]
    if x.dtype != np.float32 or x.shape != (len(y), 4, 60) or not np.isfinite(x).all():
        raise ValueError("Invalid sequence array")
    if set(y) != set(LABELS) or any(len(data[k]) != len(x) for k in data):
        raise ValueError("Both labels and aligned metadata required")
    if len(set(data["sample_id"])) != len(x):
        raise ValueError("Duplicate sample IDs")
    if not np.allclose(x[:, 2:, 1:], np.diff(x[:, :2], axis=2), atol=1e-6) or not np.allclose(x[:,2:,0],0):
        raise ValueError("Displacement channel contract mismatch")
    return data


def balanced_group_weights(y, groups, total):
    """Equal class mass, equal source-video mass within each class."""
    y, groups = np.asarray(y), np.asarray(groups)
    weights = np.zeros(len(y))
    for label in LABELS:
        unique = np.unique(groups[y == label])
        if not len(unique):
            raise ValueError("Missing class")
        for group in unique:
            mask = (y == label) & (groups == group)
            weights[mask] = total / 2 / len(unique) / mask.sum()
    return weights


def aggregate_scores(data, decisions):
    frame = pd.DataFrame(dict(sample_id=data["sample_id"], group_id=data["group_id"],
                              label=data["y"], decision=decisions))
    frame["parent_track_id"] = frame.sample_id.str.rsplit(":w", n=1).str[0]
    tracks = frame.groupby(["group_id", "parent_track_id", "label"], as_index=False).decision.mean()
    if (tracks.groupby("group_id").label.nunique() > 1).any():
        raise ValueError("Video-level binary decision is undefined for a mixed-label video")
    groups = tracks.groupby(["group_id", "label"], as_index=False).decision.mean()
    return frame, tracks, groups


def metrics(table):
    predicted = np.where(table.decision.to_numpy() >= 0, "drone", "bird")
    truth = table.label.to_numpy()
    return dict(n=len(table), accuracy=float(accuracy_score(truth,predicted)),
                balanced_accuracy=float(balanced_accuracy_score(truth,predicted)),
                macro_f1=float(f1_score(truth,predicted,labels=LABELS,average="macro",zero_division=0)),
                recall=dict(zip(LABELS,recall_score(truth,predicted,labels=LABELS,average=None,zero_division=0).tolist())),
                roc_auc=float(roc_auc_score(truth == "drone",table.decision)) if len(set(truth)) == 2 else None,
                confusion_matrix=confusion_matrix(truth,predicted,labels=LABELS).tolist())


def transformer():
    from aeon.transformations.collection.convolution_based import MiniRocket
    return MiniRocket(n_kernels=10000, random_state=20260928, n_jobs=2)


def choose_alpha(train):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20260928)
    rows, folds = [], []
    for fold, (fit, held) in enumerate(splitter.split(train["X"],train["y"],train["group_id"])):
        fit_groups, held_groups = set(train["group_id"][fit]), set(train["group_id"][held])
        if fit_groups & held_groups or set(train["y"][fit]) != set(LABELS) or set(train["y"][held]) != set(LABELS):
            raise ValueError("Invalid grouped model CV")
        rocket = transformer()
        scaler = StandardScaler(with_mean=False)
        z = scaler.fit_transform(rocket.fit_transform(train["X"][fit])).astype(np.float64)
        test_z = scaler.transform(rocket.transform(train["X"][held])).astype(np.float64)
        weight = balanced_group_weights(train["y"][fit],train["group_id"][fit],len(fit))
        part = {k:v[held] for k,v in train.items()}
        for alpha in ALPHAS:
            ridge = RidgeClassifier(alpha=alpha).fit(z,train["y"][fit],sample_weight=weight)
            group = aggregate_scores(part,ridge.decision_function(test_z))[2]
            rows.append(dict(fold=fold,alpha=alpha,group_macro_f1=metrics(group)["macro_f1"]))
        folds.append(dict(fold=fold,fit_groups=sorted(fit_groups),held_groups=sorted(held_groups)))
        print(f"model CV fold {fold} complete",flush=True)
    table = pd.DataFrame(rows)
    best = float(table.groupby("alpha").group_macro_f1.mean().idxmax())
    return best, table, folds


def fit_model(train, synthetic, alpha, fraction):
    if not 0 <= fraction < 1:
        raise ValueError("Synthetic mass must be in [0,1)")
    x, y = train["X"], train["y"]
    weights = balanced_group_weights(y,train["group_id"],len(y)*(1-fraction))
    if fraction:
        x = np.concatenate([x,synthetic["X"]])
        y = np.concatenate([y,synthetic["y"]])
        weights = np.r_[weights,balanced_group_weights(synthetic["y"],synthetic["group_id"],len(train["y"])*fraction)]
    # Shared real-only representation isolates synthetic effects in the classifier.
    # Synthetic rows cannot dominate the random thresholds or feature scaling.
    rocket, scaler = transformer(), StandardScaler(with_mean=False)
    real_z = rocket.fit_transform(train["X"])
    scaler.fit(real_z)
    z = scaler.transform(rocket.transform(x)).astype(np.float64)
    ridge = RidgeClassifier(alpha=alpha).fit(z,y,sample_weight=weights)
    return dict(rocket=rocket,scaler=scaler,classifier=ridge,alpha=alpha,synthetic_mass=fraction,
                contract_version=CONTRACT_VERSION,contract_id=SequenceConfig().fingerprint,
                classes=ridge.classes_.tolist())


def predict(model, x):
    if model["classes"] != LABELS:
        raise ValueError("Unexpected score sign / class order")
    z = model["scaler"].transform(model["rocket"].transform(x)).astype(np.float64)
    return model["classifier"].decision_function(z)


def export_test(manifest, output, freeze_path):
    """Test conversion only after an immutable model-selection checkpoint exists."""
    freeze = read_json(freeze_path)
    if not freeze.get("frozen_before_test") or freeze["contract_id"] != SequenceConfig().fingerprint:
        raise ValueError("Missing frozen selection")
    for name, checksum in freeze["model_hashes"].items():
        if file_hash(Path(freeze_path).parent/name) != checksum:
            raise ValueError("Frozen model changed")
    if Path(output).exists():
        raise ValueError("Do not overwrite test replay")
    cfg, rows, rejected, coverage = SequenceConfig(), [], [], []
    for rec in manifest["sources"]:
        if rec["split"] != "test" or rec["status"] == "duplicate":
            continue
        path = Path(rec["source_path"])
        if file_hash(path) != rec["file_sha256"]:
            raise ValueError("Real test source changed")
        tracks = [t for t in load_tracks(path) if digest(t["history"]) == rec["history_hash"]]
        if len(tracks) != 1:
            raise ValueError("Cannot identify test track")
        windows, rejects = window_track(tracks[0],cfg)
        coverage.append(dict(parent_track_id=rec["parent_track_id"],group_id=rec["source_group_id"],
                             label=rec["label"],accepted_windows=len(windows),rejected_windows=len(rejects)))
        for window in windows:
            rows.append(dict(window, sample_id=f"{rec['parent_track_id']}:w{window['window_index']:04d}",
                             group_id=rec["source_group_id"],parent_track_id=rec["parent_track_id"],label=rec["label"]))
        rejected.extend(dict(r,parent_track_id=rec["parent_track_id"]) for r in rejects)
    if not rows:
        raise ValueError("No valid test windows")
    rows.sort(key=lambda r:r["sample_id"])
    np.savez_compressed(output,X=np.stack([r["X"] for r in rows]),y=np.array([r["label"] for r in rows],dtype="U5"),
                        sample_id=np.array([r["sample_id"] for r in rows]),group_id=np.array([r["group_id"] for r in rows]))
    pd.DataFrame([{k:v for k,v in r.items() if k != "X"} for r in rows]).to_csv(Path(output).with_suffix(".metadata.csv"),index=False)
    return dict(coverage=coverage,rejections=rejected,converted_at=datetime.now(timezone.utc).isoformat(),
                frozen_selection_sha256=file_hash(freeze_path),test_npz_sha256=file_hash(output),
                status="historically_inspected_development_test_replay_not_new_independent_test")


def paired_interval(before, after, draws=2000):
    joined = before.merge(after,on=["group_id","label"],suffixes=("_before","_after"),validate="one_to_one")
    if len(joined) != len(before) or len(joined) != len(after):
        raise ValueError("Unpaired test groups")
    rng = np.random.default_rng(20260928)
    deltas = []
    for _ in range(draws):
        indices = np.concatenate([rng.choice(np.flatnonzero(joined.label == c),sum(joined.label == c),replace=True) for c in LABELS])
        sample = joined.iloc[indices]
        scores = [f1_score(sample.label,np.where(sample[f"decision_{arm}"] >= 0,"drone","bird"),
                           labels=LABELS,average="macro",zero_division=0) for arm in ("before","after")]
        deltas.append(scores[1]-scores[0])
    return dict(group_macro_f1_delta_q025_q50_q975=np.quantile(deltas,[.025,.5,.975]).tolist(),
                draws=draws,method="Paired class-stratified source-video bootstrap; sessions remain unverified")


def evaluate(folder, run_name="model_evaluation", previous_run=None):
    folder = Path(folder)
    if Path(run_name).name != run_name or run_name in (".",".."):
        raise ValueError("Evaluation name must be a folder basename")
    output = folder/run_name
    if output.exists():
        raise ValueError("Model evaluation is immutable; use a new preparation run")
    preparation = read_json(folder/"preparation_results.json")
    manifest = read_json(folder/"real/dataset_manifest.json")
    if preparation["frozen_profile_sha256"] != file_hash(folder/"selected_profile.json"):
        raise ValueError("Calibration changed after validation")
    if manifest["contract_id"] != SequenceConfig().fingerprint or manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError("Contract mismatch")
    train, val, synthetic = [load_arrays(p) for p in (folder/"real/train.npz",folder/"real/validation.npz",folder/"synthetic_train.npz")]
    for name in ("train.npz","validation.npz","metadata.csv"):
        if file_hash(folder/"real"/name) != manifest["artifact_hashes"][name]:
            raise ValueError("Real data changed")
    test_groups = {r["source_group_id"] for r in manifest["sources"] if r["split"] == "test"}
    if set(train["group_id"]) & set(val["group_id"]) or test_groups & (set(train["group_id"]) | set(val["group_id"])):
        raise ValueError("Real split overlap")
    output.mkdir()
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        contract_version=CONTRACT_VERSION,contract_id=SequenceConfig().fingerprint,
        alpha_candidates=list(ALPHAS),alpha_selection="Real train-only grouped 3-fold CV; same alpha for both arms",
        representation="MiniRocket 9996 outputs; fit thresholds and StandardScaler(with_mean=False) on real train only in both arms",
        synthetic_mass=.25,decision_threshold=0.,aggregation="Mean window margin per track, then equal-object mean per source video",
        model_selection="Highest validation source-video macro-F1; exact ties favor real_only",
        scope="Development comparison, no new independent real-test claim; no probability calibration",
        versions={name:importlib.metadata.version(name) for name in ("aeon","numpy","pandas","scikit-learn","scipy","numba","joblib")},
        input_hashes={str(p.relative_to(folder)):file_hash(p) for p in (folder/"real/train.npz",folder/"real/validation.npz",folder/"synthetic_train.npz",folder/"selected_profile.json")},
        evaluator_sha256=file_hash(__file__),test_opened=False,
        solver_precision="float64 to avoid ill-conditioned float32 Ridge solves",
        prior_evaluation=None if previous_run is None else dict(path=str(previous_run),
            results_sha256=file_hash(Path(previous_run)/"results.json"),
            reason="Numerical precision correction only; identical alpha grid, seeds, fractions and selection policy. Test was already replayed."))
    write_json(output/"protocol.json",protocol)
    alpha, cv, folds = choose_alpha(train)
    cv.to_csv(output/"model_cv.csv",index=False)
    write_json(output/"model_cv_groups.json",folds)
    models, results, val_scores = {}, {}, {}
    for arm, fraction in (("real_only",0.),("real_plus_synthetic",.25)):
        model = fit_model(train,synthetic,alpha,fraction)
        joblib.dump(model,output/f"{arm}.joblib",compress=3)
        tables = aggregate_scores(val,predict(model,val["X"]))
        results[arm] = dict(validation={level:metrics(table) for level,table in zip(("window","track","group"),tables)})
        val_scores[arm] = results[arm]["validation"]["group"]["macro_f1"]
        for level,table in zip(("window","track","group"),tables):
            table.to_csv(output/f"{arm}_validation_{level}.csv",index=False)
        models[arm] = model
        print(f"{arm}: validation group F1={val_scores[arm]:.4f}",flush=True)
    chosen = "real_plus_synthetic" if val_scores["real_plus_synthetic"] > val_scores["real_only"] else "real_only"
    freeze = dict(selected_arm=chosen,alpha=alpha,validation_group_macro_f1=val_scores,frozen_before_test=True,
                  frozen_at=datetime.now(timezone.utc).isoformat(),contract_id=SequenceConfig().fingerprint,
                  protocol_sha256=file_hash(output/"protocol.json"),
                  model_hashes={p.name:file_hash(p) for p in output.glob("*.joblib")})
    write_json(output/"model_selection.json",freeze)
    frozen_hash = file_hash(output/"model_selection.json")
    test_protocol = export_test(manifest,output/"test.npz",output/"model_selection.json")
    write_json(output/"test_replay.json",test_protocol)
    test = load_arrays(output/"test.npz")
    test_scores = {}
    for arm, model in models.items():
        tables = aggregate_scores(test,predict(model,test["X"]))
        results[arm]["test"] = {level:metrics(table) for level,table in zip(("window","track","group"),tables)}
        for level,table in zip(("window","track","group"),tables):
            table.to_csv(output/f"{arm}_test_{level}.csv",index=False)
        test_scores[arm] = tables[2]
    if frozen_hash != file_hash(output/"model_selection.json"):
        raise ValueError("Selection changed after test")
    result = dict(selected_arm=chosen,alpha=alpha,arms=results,
        paired_bootstrap=paired_interval(test_scores["real_only"],test_scores["real_plus_synthetic"]),
        independent_evaluation_ready=False,session_independence_verified=False,
        final_test_status=test_protocol["status"],test_counts={c:int(sum(test["y"] == c)) for c in LABELS},
        synthetic_mass=.25,selected_model_sha256=freeze["model_hashes"][chosen+".joblib"])
    write_json(output/"results.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--run-name",default="model_evaluation")
    parser.add_argument("--previous-run",type=Path)
    args = parser.parse_args()
    evaluate(args.folder,args.run_name,args.previous_run)
