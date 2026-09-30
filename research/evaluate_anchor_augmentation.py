"""Evaluate train-only real anchors with small simulated flight residuals.

Derived windows keep the real source-video group. No new independent case is
created, and the historically inspected real test is never read.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import simulated_windows
from .evaluate_sequence_handoff import (aggregate_scores, balanced_group_weights,
                                        load_arrays, metrics, transformer)
from .io import write_json
from .refine_sequence_calibration import grouped_folds
from .sequence_simulator import AcquisitionProfile, BIRDS, QUADS, latent_flight, observe_flight
from .trajectory_sequence import SequenceConfig, normalize_window


AMPLITUDES = (.04, .10)
MASSES = (.10, .25)
SEED_START = 151000
COPIES_PER_ANCHOR = 3


def seeded_id(*parts):
    payload=json.dumps(parts,sort_keys=True,ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"little")


def donor_pool(count=24):
    """Fixed, A-independent physical donor prior used for all model folds."""
    profile=AcquisitionProfile(name="unfitted_anchor_donor",ground_camera=True,
        span_q10=.04,span_q50=.12,span_q90=.30,span_multiplier=2.,
        jitter_px=.15,dropout_rate=0.,burst_probability=0.,
        drift_probability=0.,camera_probability=0.)
    observed=[]
    for label,subtypes in (("bird",BIRDS),("drone",QUADS)):
        for i in range(count):
            flight=latent_flight(label,subtypes[i%len(subtypes)],SEED_START+i)
            observed.append(observe_flight(flight,profile))
    x,meta,status,_=simulated_windows(observed,SequenceConfig())
    if not len(x) or set(meta.label) != {"bird","drone"}:
        raise ValueError("Donor pool must have both classes")
    return dict(X=x, y=meta.label.to_numpy(dtype=str),
                group_id=meta.group_id.to_numpy(dtype=str),
                sample_id=meta.sample_id.to_numpy(dtype=str)),status,profile


def donor_residual(x):
    q=np.asarray(x[:2].T,dtype=np.float64)
    trend=q[:1]+np.linspace(0.,1.,len(q))[:,None]*(q[-1:]-q[:1])
    residual=q-trend
    residual-=np.median(residual,axis=0)
    radius=float(np.quantile(np.linalg.norm(residual,axis=1),.9))
    return residual/max(radius,.25)


def augment_anchors(data, metadata, source_dimensions, donors, amplitude,
                    copies=COPIES_PER_ANCHOR, copy_start=0):
    if not 0 < amplitude <= .15 or copies < 1 or copy_start < 0:
        raise ValueError("Invalid bounded augmentation")
    if not np.array_equal(data["sample_id"],metadata.sample_id.to_numpy()) or not np.array_equal(data["group_id"],metadata.source_group_id.to_numpy()):
        raise ValueError("Real anchor metadata misaligned")
    donor_indices={label:np.flatnonzero(donors["y"] == label) for label in ("bird","drone")}
    if any(not len(ids) for ids in donor_indices.values()):
        raise ValueError("Donors need both classes")
    records=[]
    rejected=0
    for index,row in enumerate(metadata.itertuples(index=False)):
        label=data["y"][index]
        if row.parent_track_id not in source_dimensions:
            raise ValueError("Missing processed aspect ratio")
        aspect=source_dimensions[row.parent_track_id]
        center=np.array([row.center_x,row.center_y])
        q=np.asarray(data["X"][index,:2].T,dtype=np.float64)
        for copy in range(copy_start, copy_start + copies):
            rng=np.random.default_rng(seeded_id(row.sample_id,copy,amplitude))
            donor=int(rng.choice(donor_indices[label]))
            perturbation=donor_residual(donors["X"][donor])
            sign=-1. if rng.random() < .5 else 1.
            varied=q+sign*amplitude*perturbation
            screen=center+varied*float(row.normalization_scale)
            if (np.any(screen[:,0] < 0) or np.any(screen[:,0] > 1)
                    or np.any(screen[:,1] < 0) or np.any(screen[:,1] > aspect)):
                rejected+=1
                continue
            x,_=normalize_window(varied,SequenceConfig())
            records.append(dict(X=x,y=label,group_id=row.source_group_id,
                sample_id=f"{row.sample_id}:anchor{copy:03d}",
                parent_track_id=row.parent_track_id,
                donor_id=donors["sample_id"][donor]))
    if not records:
        raise ValueError("All augmented windows rejected")
    arrays={key:np.asarray([r[key] for r in records]) for key in ("X","y","group_id","sample_id")}
    if set(arrays["y"]) != {"bird","drone"}:
        raise ValueError("Augmentation removed a class")
    return arrays,pd.DataFrame([{k:r[k] for k in r if k != "X"} for r in records]),rejected


def model_scores(z_fit,y_fit,g_fit,z_aug,y_aug,g_aug,z_held,held,alpha,mass):
    weights=np.r_[balanced_group_weights(y_fit,g_fit,len(y_fit)*(1-mass)),
                  balanced_group_weights(y_aug,g_aug,len(y_fit)*mass)]
    model=RidgeClassifier(alpha=alpha).fit(np.concatenate([z_fit,z_aug]),
                    np.r_[y_fit,y_aug],sample_weight=weights)
    return aggregate_scores(held,model.decision_function(z_held))


def run(folder,output):
    folder,output=Path(folder),Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    real_dir=folder/"real"
    manifest=json.loads((real_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg=SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Input contract mismatch")
    train,val=[load_arrays(real_dir/f"{split}.npz") for split in ("train","validation")]
    metadata=pd.read_csv(real_dir/"metadata.csv",keep_default_na=False)
    metadata=metadata[metadata.split == "train"].sort_values("npz_row").reset_index(drop=True)
    dims={r["parent_track_id"]:r["processed_height"]/r["processed_width"]
          for r in manifest["sources"] if r["split"] == "train"}
    if set(train["group_id"]) & set(val["group_id"]):
        raise ValueError("Train/validation source group overlap")
    for name in ("train.npz","validation.npz","metadata.csv"):
        if file_hash(real_dir/name) != manifest["artifact_hashes"][name]:
            raise ValueError("Real source changed")
    output.mkdir(parents=True)
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=file_hash(real_dir/"dataset_manifest.json"),
        donor="Fixed A-independent force-based 3D simulation, ground camera and no dropout/drift",
        donor_seed_start=SEED_START,donor_count_per_class=24,
        amplitudes=list(AMPLITUDES),synthetic_masses=list(MASSES),copies_per_anchor=COPIES_PER_ANCHOR,
        group_rule="Derived sample inherits source_video_group_id from its real anchor",
        construction="Small signed simulator residual after subtracting donor endpoint trend; renormalize 2-second window",
        model="MiniRocket fit on real fit groups only, Ridge alpha fixed at 0.1",
        gate="Train 3-fold group macro-F1 mean >= real-only and no fold worse by >0.05; validation then must not be worse",
        test_opened=False, code_sha256=file_hash(__file__))
    write_json(output/"protocol.json",protocol)
    donors,donor_status,donor_profile=donor_pool()
    write_json(output/"donor_profile.json",asdict(donor_profile))
    donor_status.to_csv(output/"donor_attempts.csv",index=False)
    np.savez_compressed(output/"donor_pool.npz",**donors)
    records=[]
    for fold,(fit_groups,held_groups) in enumerate(grouped_folds(metadata.rename(columns={"source_group_id":"group_id"}))):
        fit=np.flatnonzero(np.isin(train["group_id"],list(fit_groups)))
        held=np.flatnonzero(np.isin(train["group_id"],list(held_groups)))
        held_data={k:v[held] for k,v in train.items()}
        rocket=transformer()
        scaler=StandardScaler(with_mean=False)
        z_fit=scaler.fit_transform(rocket.fit_transform(train["X"][fit])).astype(np.float64)
        z_held=scaler.transform(rocket.transform(train["X"][held])).astype(np.float64)
        w=balanced_group_weights(train["y"][fit],train["group_id"][fit],len(fit))
        base=RidgeClassifier(alpha=.1).fit(z_fit,train["y"][fit],sample_weight=w)
        baseline=metrics(aggregate_scores(held_data,base.decision_function(z_held))[2])["macro_f1"]
        records.append(dict(fold=fold,variant="real_only",amplitude=0.,mass=0.,group_macro_f1=baseline))
        for amplitude in AMPLITUDES:
            fit_data={k:v[fit] for k,v in train.items()}
            augmentation,_,_ =augment_anchors(fit_data,metadata.iloc[fit].reset_index(drop=True),dims,donors,amplitude)
            z_aug=scaler.transform(rocket.transform(augmentation["X"])).astype(np.float64)
            for mass in MASSES:
                tables=model_scores(z_fit,train["y"][fit],train["group_id"][fit],
                    z_aug,augmentation["y"],augmentation["group_id"],z_held,held_data,.1,mass)
                result=metrics(tables[2])["macro_f1"]
                records.append(dict(fold=fold,variant="real_anchor_sim_residual",amplitude=amplitude,
                    mass=mass,group_macro_f1=result))
        print(f"anchor CV fold {fold} complete",flush=True)
    cv=pd.DataFrame(records)
    cv.to_csv(output/"cv_scores.csv",index=False)
    baseline=cv[cv.variant == "real_only"].set_index("fold").group_macro_f1
    candidates=[]
    for (amplitude,mass),part in cv[cv.variant != "real_only"].groupby(["amplitude","mass"]):
        aligned=part.set_index("fold").group_macro_f1
        delta=aligned-baseline
        candidates.append(dict(amplitude=amplitude,mass=mass,mean_f1=float(aligned.mean()),
            baseline_mean_f1=float(baseline.mean()),minimum_fold_delta=float(delta.min()),
            passed_train=bool(aligned.mean() >= baseline.mean() and delta.min() >= -.05)))
    ranking=sorted(candidates,key=lambda r:(-r["mean_f1"],r["amplitude"],r["mass"]))
    write_json(output/"train_gate.json",dict(candidates=ranking,baseline_mean_f1=float(baseline.mean())))
    result=dict(train_gate=ranking,validation_opened=False,test_opened=False,
        status="rejected_by_train_group_cv",large_generation_ready=False)
    approved=next((r for r in ranking if r["passed_train"]),None)
    if approved:
        amplitude,mass=approved["amplitude"],approved["mass"]
        augmented,augmeta,rejected=augment_anchors(train,metadata,dims,donors,amplitude)
        augmeta.to_csv(output/"selected_train_augmented_metadata.csv",index=False)
        np.savez_compressed(output/"selected_train_augmented.npz",**augmented)
        rocket=transformer()
        scaler=StandardScaler(with_mean=False)
        z_train=scaler.fit_transform(rocket.fit_transform(train["X"])).astype(np.float64)
        z_aug=scaler.transform(rocket.transform(augmented["X"])).astype(np.float64)
        z_val=scaler.transform(rocket.transform(val["X"])).astype(np.float64)
        base=RidgeClassifier(alpha=.1).fit(z_train,train["y"],
            sample_weight=balanced_group_weights(train["y"],train["group_id"],len(train["y"])))
        base_metric=metrics(aggregate_scores(val,base.decision_function(z_val))[2])
        candidate_metric=metrics(model_scores(z_train,train["y"],train["group_id"],
            z_aug,augmented["y"],augmented["group_id"],z_val,val,.1,mass)[2])
        result.update(validation_opened=True,selected=approved,accepted_augmented=len(augmented["X"]),
            rejected_out_of_frame=rejected,validation_real_only=base_metric,
            validation_augmented=candidate_metric,
            status="development_candidate_only" if candidate_metric["macro_f1"] >= base_metric["macro_f1"] else "validation_rejected")
        result["large_generation_ready"]=bool(result["status"] == "development_candidate_only")
        result["independent_real_world_evidence_ready"]=False
    write_json(output/"results.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--output",type=Path,required=True,
                        help="New directory for this experiment; existing results are not overwritten")
    args=parser.parse_args()
    run(args.folder,args.output)
