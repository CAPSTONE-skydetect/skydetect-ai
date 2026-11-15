"""Train-only support matching for already generated simulator trajectories.

This is a bounded diagnostic. It never reads the historically inspected real test.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import real_partition
from .evaluate_sequence_handoff import aggregate_scores, balanced_group_weights, load_arrays, metrics, predict
from .io import write_json
from .sequence_comparison import compare, group_weights, metric_scales, quantile, score
from .trajectory_sequence import SequenceConfig


SUPPORT_METRICS = ("screen_span", "step_cv", "band_3_10_ratio")
SYNTHETIC_MASSES = (.1, .25)


def support_mask(real, synthetic):
    """Classwise 5-95% real-train envelopes; applied before model fitting."""
    if set(real.label) != {"bird", "drone"} or set(synthetic.label) != {"bird", "drone"}:
        raise ValueError("Both classes required")
    if synthetic.sample_id.duplicated().any():
        raise ValueError("Duplicate synthetic windows")
    result = np.zeros(len(synthetic), dtype=bool)
    limits = {}
    for label in ("bird", "drone"):
        source = real[real.label == label]
        target = synthetic[synthetic.label == label]
        weights = group_weights(source)
        ok = np.ones(len(target), dtype=bool)
        limits[label] = {}
        for name in SUPPORT_METRICS:
            low, high = quantile(source[name].to_numpy(), weights, [.05,.95])
            ok &= target[name].between(low,high,inclusive="both").to_numpy()
            limits[label][name] = dict(q05=float(low),q95=float(high))
        result[np.flatnonzero(synthetic.label == label)] = ok
    return result, limits


def run(folder, output):
    folder, output = Path(folder), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    real_dir = folder/"real"
    manifest = json.loads((real_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Real input contract mismatch")
    train, validation, synthetic = [load_arrays(p) for p in
        (real_dir/"train.npz", real_dir/"validation.npz", folder/"synthetic_train.npz")]
    for name in ("train.npz","validation.npz","metadata.csv"):
        if file_hash(real_dir/name) != manifest["artifact_hashes"][name]:
            raise ValueError("Real data changed")
    _, train_meta, train_metrics = real_partition(real_dir,"train",manifest,cfg)
    synthetic_meta = pd.read_csv(folder/"synthetic_train_metrics.csv",keep_default_na=False)
    if not np.array_equal(synthetic_meta.sample_id,synthetic["sample_id"]) or not np.array_equal(synthetic_meta.group_id,synthetic["group_id"]):
        raise ValueError("Synthetic metadata misaligned")
    if set(train["group_id"]) & set(validation["group_id"]):
        raise ValueError("Real train/validation overlap")
    model_path = folder/"model_evaluation_float64/real_only.joblib"
    frozen = json.loads((folder/"model_evaluation_float64/model_selection.json").read_text(encoding="utf-8"))
    if not frozen["frozen_before_test"] or file_hash(model_path) != frozen["model_hashes"][model_path.name]:
        raise ValueError("Reference model changed")
    reference = joblib.load(model_path)
    if reference["contract_id"] != cfg.fingerprint or reference["classes"] != ["bird","drone"]:
        raise ValueError("Model/input contract mismatch")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
        rule="For each class and each metric: real-train source-group weighted q05 <= synthetic <= q95",
        metrics=list(SUPPORT_METRICS), synthetic_masses=list(SYNTHETIC_MASSES),
        transformation="reuse frozen real-only MiniRocket + scaler; fit only Ridge",
        source_real_manifest_sha256=file_hash(real_dir/"dataset_manifest.json"),
        source_synthetic_npz_sha256=file_hash(folder/"synthetic_train.npz"),
        reference_model_sha256=file_hash(model_path),
        evaluation_policy="Exploratory validation only; test is never loaded or rescored",
        code_sha256=file_hash(__file__), test_opened=False)
    write_json(output/"protocol.json",protocol)
    mask, limits = support_mask(train_metrics,synthetic_meta)
    write_json(output/"support_limits.json",limits)
    synthetic_meta.assign(accepted=mask).to_csv(output/"selection.csv",index=False)
    subset = {k:v[mask] for k,v in synthetic.items()}
    if set(subset["y"]) != {"bird","drone"}:
        raise ValueError("Filtering removed a class")
    np.savez_compressed(output/"synthetic_supported_train.npz",**subset)
    supported_metrics = synthetic_meta.loc[mask].copy()
    full_comparison = compare(train_metrics,synthetic_meta,metric_scales(train_metrics))
    filtered_comparison = compare(train_metrics,supported_metrics,metric_scales(train_metrics))
    pd.concat([full_comparison.assign(variant="all"),filtered_comparison.assign(variant="supported")]).to_csv(output/"distribution_comparison.csv",index=False)
    z_real = reference["scaler"].transform(reference["rocket"].transform(train["X"])).astype(np.float64)
    z_sim = reference["scaler"].transform(reference["rocket"].transform(subset["X"])).astype(np.float64)
    z_val = reference["scaler"].transform(reference["rocket"].transform(validation["X"])).astype(np.float64)
    real_weight = balanced_group_weights(train["y"],train["group_id"],len(train["y"]))
    rows = []
    for mass in SYNTHETIC_MASSES:
        weights = np.r_[real_weight*(1-mass),
            balanced_group_weights(subset["y"],subset["group_id"],len(train["y"])*mass)]
        ridge = RidgeClassifier(alpha=reference["alpha"]).fit(
            np.concatenate([z_real,z_sim]),np.r_[train["y"],subset["y"]],sample_weight=weights)
        decisions = ridge.decision_function(z_val)
        tables = aggregate_scores(validation,decisions)
        for level,table in zip(("window","track","group"),tables):
            table.to_csv(output/f"mass_{mass}_{level}_validation.csv",index=False)
        rows.append(dict(mass=mass, window=metrics(tables[0]),track=metrics(tables[1]),group=metrics(tables[2])))
    baseline_tables = aggregate_scores(validation,predict(reference,validation["X"]))
    summary = dict(accepted_windows=len(subset["X"]),total_windows=len(synthetic["X"]),
        accepted_flights=len(set(subset["group_id"])),
        counts={label:int(sum(subset["y"] == label)) for label in ("bird","drone")},
        train_distance_all=score(full_comparison),train_distance_supported=score(filtered_comparison),
        baseline_validation={level:metrics(table) for level,table in zip(("window","track","group"),baseline_tables)},
        filtered_mixture_validation=rows, test_opened=False,
        status="exploratory_train_support_ablation_not_bulk_generation_approval")
    write_json(output/"results.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--output",type=Path,default=Path("research/output/sequence_selection_v1"))
    args = parser.parse_args()
    run(args.folder,args.output)
