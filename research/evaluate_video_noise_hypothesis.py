"""Train-group-only check of a bounded small-target jitter hypothesis.

The hypothesis comes from four visually reviewed A/video clips, not from
independent ground-truth labels. It is rejected before validation unless its
grouped distribution distance improves consistently.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from .calibrate_sequence_simulator import estimate_acquisition, real_partition, simulated_windows
from .io import write_json
from .measure_train_motion import train_tracks
from .prepare_sequence_handoff import read_json
from .refine_sequence_calibration import (candidate_profile, fit_motion_priors,
                                          generate_flights, grouped_folds)
from .sequence_comparison import compare, metric_scales, score
from .sequence_simulator import observe_flight
from .trajectory_sequence import SequenceConfig


def evaluate(dataset, output, flights_per_class=18):
    dataset,output=Path(dataset),Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    if flights_per_class < 12:
        raise ValueError("Need at least 12 flights per class")
    manifest=read_json(dataset/"dataset_manifest.json")
    cfg=SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Sequence contract changed")
    _,metadata,real=real_partition(dataset,"train",manifest,cfg)
    original=list(train_tracks(manifest))
    protocol=dict(
        hypothesis="More small-target observation jitter; no physical-flight or camera change",
        evidence="Four video clips with A-seeded silhouette proxies; not independent localization ground truth",
        candidates={"baseline":1.0,"higher_difficulty":2.0},
        gate="Mean grouped CV objective strictly lower and no fold worse by more than 0.05",
        folds=3,seed_start=172000,flights_per_class=flights_per_class,
        contract_id=cfg.fingerprint,validation_opened_only_after_gate=True,test_opened=False)
    output.mkdir(parents=True)
    write_json(output/"protocol.json",protocol)
    records=[]
    for fold,(fit_groups,held_groups) in enumerate(grouped_folds(metadata)):
        fit_real=real[real.group_id.isin(fit_groups)]
        held_real=real[real.group_id.isin(held_groups)]
        fit_manifest=dict(manifest,sources=[r for r in manifest["sources"]
                                           if r["split"]=="train" and r["source_group_id"] in fit_groups])
        base,_,_=estimate_acquisition(fit_real,fit_manifest)
        fit_tracks=[(r,t) for r,t in original if r["source_group_id"] in fit_groups]
        measured,guidance,_,_,_=fit_motion_priors(fit_tracks,base)
        profile=replace(candidate_profile(base,measured,"baseline","old_ar1",2.),
                        ground_camera=True)
        flights=generate_flights(flights_per_class,172000+fold*1000,guidance)
        scales=metric_scales(fit_real)
        for name,gain in protocol["candidates"].items():
            candidate=replace(profile,name=name,jitter_difficulty_gain=gain)
            bundle=simulated_windows([observe_flight(f,candidate) for f in flights],cfg)
            distance=score(compare(held_real,bundle[1],scales))
            usable=float(bundle[2].usable.mean())
            objective=distance+2*(1-usable)
            records.append(dict(fold=fold,candidate=name,distance=distance,
                                usable_fraction=usable,objective=objective))
            print(f"fold {fold} {name}: objective={objective:.4f}",flush=True)
    table=pd.DataFrame(records)
    table.to_csv(output/"train_cv.csv",index=False)
    base=table[table.candidate=="baseline"].set_index("fold").objective
    trial=table[table.candidate=="higher_difficulty"].set_index("fold").objective
    delta=trial-base
    passed=bool(trial.mean()<base.mean() and delta.max()<=.05)
    result=dict(baseline_mean_objective=float(base.mean()),
                candidate_mean_objective=float(trial.mean()),
                fold_deltas=delta.to_dict(),passed_train_gate=passed,
                validation_opened=False,test_opened=False,
                status="train_gate_passed_requires_validation" if passed else "rejected_in_train_cv",
                calibration_approved=False)
    write_json(output/"train_gate.json",result)
    if not passed:
        return result
    # One development validation comparison after a train-only decision.
    base,_,_=estimate_acquisition(real,manifest)
    measured,guidance,_,_,_=fit_motion_priors(original,base)
    profile=replace(candidate_profile(base,measured,"baseline","old_ar1",2.),ground_camera=True)
    _,_,validation=real_partition(dataset,"validation",manifest,cfg)
    flights=generate_flights(flights_per_class,177000,guidance)
    scales=metric_scales(real)
    validation_scores={}
    for name,gain in protocol["candidates"].items():
        candidate=replace(profile,name=name,jitter_difficulty_gain=gain)
        bundle=simulated_windows([observe_flight(f,candidate) for f in flights],cfg)
        validation_scores[name]=float(score(compare(validation,bundle[1],scales)))
    result.update(validation_opened=True,validation_distances=validation_scores,
                  status="development_candidate_only" if validation_scores["higher_difficulty"] < validation_scores["baseline"]
                  else "rejected_in_validation")
    result["calibration_approved"]=False
    write_json(output/"train_gate.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,
        default=Path("research/output/sequence_handoff_20260928/real"))
    parser.add_argument("--output",type=Path,
        default=Path("research/output/video_noise_hypothesis_v1"))
    args=parser.parse_args()
    print(json.dumps(evaluate(args.dataset,args.output),indent=2),flush=True)
