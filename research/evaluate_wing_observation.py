"""One bounded wing-centroid observation hypothesis, tested on train groups first."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import real_partition, simulated_windows
from .io import write_json
from .refine_sequence_calibration import generate_flights, grouped_folds
from .sequence_comparison import compare, metric_scales, score, sequence_mmd
from .sequence_simulator import AcquisitionProfile, GuidanceProfile, observe_flight, SEQUENCE_SIMULATOR_VERSION
from .trajectory_sequence import SequenceConfig


WING_CENTROID_RATIO = .12


def paired_candidate(profile):
    if profile.wing_centroid_ratio:
        raise ValueError("Control profile must have no wing observation shift")
    return replace(profile,name="wing_centroid_0.12",wing_centroid_ratio=WING_CENTROID_RATIO)


def cv_gate(table):
    control = table[table.variant == "control"].set_index("fold")
    candidate = table[table.variant == "wing_centroid_0.12"].set_index("fold")
    if len(control) != 3 or set(control.index) != set(candidate.index):
        raise ValueError("Three aligned train folds required")
    checks = dict(mean_distance_improved=bool(candidate.distance.mean() < control.distance.mean()),
        mean_objective_improved=bool(candidate.objective.mean() < control.objective.mean()),
        all_fold_distances_improved=bool((candidate.distance < control.distance).all()),
        mean_usable_not_decreased=bool(candidate.usable.mean() >= control.usable.mean()))
    return dict(passed=all(checks.values()),checks=checks,
        interpretation="Conservative train-only development gate; not proof of real-world performance")


def plot_cv(output):
    output=Path(output)
    scores=pd.read_csv(output/"cv_scores.csv")
    rows=[]
    for fold in range(3):
        for variant in ("control","wing_centroid_0.12"):
            rows.append(pd.read_csv(output/f"fold{fold}_{variant}.csv").assign(fold=fold,variant=variant))
    metrics=pd.concat(rows,ignore_index=True)
    bird=metrics[metrics.label == "bird"]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    colors={"control":"#777777","wing_centroid_0.12":"#147e92"}
    for variant in colors:
        part=scores[scores.variant == variant]
        axes[0].plot(part.fold,part.distance,"o-",label=variant,color=colors[variant])
    axes[0].set(xticks=[0,1,2],xlabel="Train group fold",ylabel="Distribution distance (lower better)",title="Total domain gap")
    axes[0].legend(fontsize=8)
    for ax,name in ((axes[1],"band_3_10_ratio"),(axes[2],"acf_lag3")):
        part=bird[bird.metric == name]
        for variant in colors:
            values=part[part.variant == variant]
            ax.plot(values.fold,values.sim_median,"o-",label=variant,color=colors[variant])
        ax.plot(part[part.variant == "control"].fold,part[part.variant == "control"].real_median,"k--",label="held real")
        ax.set(xticks=[0,1,2],xlabel="Train group fold",ylabel="Feature median",title=name)
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output/"wing_cv_diagnostics.png",dpi=170)
    plt.close(fig)


def run(dataset, reference, output):
    dataset, reference, output = map(Path,(dataset,reference,output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    manifest = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Sequence contract mismatch")
    fold_data = json.loads((reference/"fold_evidence.json").read_text(encoding="utf-8"))
    selected = json.loads((reference/"selected_profile.json").read_text(encoding="utf-8"))
    real_x, real_meta, real_metrics = real_partition(dataset,"train",manifest,cfg)
    output.mkdir(parents=True)
    protocol = dict(generator=SEQUENCE_SIMULATOR_VERSION,
        contract_id=cfg.fingerprint, real_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
        previous_profile_sha256=file_hash(reference/"selected_profile.json"),
        fold_evidence_sha256=file_hash(reference/"fold_evidence.json"),
        hypothesis="During powered bird flight only, tracker centroid follows sinusoidal wing silhouette by <=12% of optical wingspan; physical COM unchanged",
        ratio=WING_CENTROID_RATIO, flight_count_per_class_per_fold=18,
        fold_seed_start=61000, validation_seed_start=141000,
        gate="Mean distance/objective and each fold distance improve, no mean availability loss",
        validation_policy="Do not read validation unless gate passes; reused development set",
        test_opened=False,code_sha256=file_hash(__file__))
    write_json(output/"protocol.json",protocol)
    rows = []
    for fold,(fit,held) in enumerate(grouped_folds(real_meta)):
        evidence = fold_data[fold]
        if fold != evidence["fold"] or fit != set(evidence["fit_groups"]) or held != set(evidence["held_groups"]):
            raise ValueError("Saved fold membership changed")
        profiles = [AcquisitionProfile(**p) for p in evidence["profiles"]]
        control = next(p for p in profiles if p.name == "ground_span2.0")
        candidate = paired_candidate(control)
        guidance = GuidanceProfile(**evidence["guidance"])
        flights = generate_flights(18,61000+1000*fold,guidance)
        held_real = real_metrics[real_metrics.group_id.isin(held)]
        fit_real = real_metrics[real_metrics.group_id.isin(fit)]
        for profile in (replace(control,name="control"),candidate):
            observed = [observe_flight(f,profile) for f in flights]
            bundle = simulated_windows(observed,cfg)
            comparison = compare(held_real,bundle[1],metric_scales(fit_real))
            comparison.to_csv(output/f"fold{fold}_{profile.name}.csv",index=False)
            distance, usable = score(comparison),float(bundle[2].usable.mean())
            rows.append(dict(fold=fold,variant=profile.name,distance=distance,usable=usable,
                objective=distance+2*(1-usable),windows=len(bundle[0]),
                below_ground=sum(r["metadata"]["camera"]["position"][2] < 0 for r in observed)))
            print(f"fold {fold} {profile.name}: distance={distance:.4f}, usable={usable:.3f}",flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/"cv_scores.csv",index=False)
    plot_cv(output)
    gate = cv_gate(table)
    write_json(output/"cv_gate.json",gate)
    control = AcquisitionProfile(**selected["profile"])
    candidate = paired_candidate(control)
    frozen = dict(profile=asdict(candidate), guidance=selected["guidance"],
        fit_on_train_only=True, cv_gate=gate, protocol_sha256=file_hash(output/"protocol.json"))
    write_json(output/"candidate_profile.json",frozen)
    frozen_hash = file_hash(output/"candidate_profile.json")
    result = dict(cv_gate=gate,validation_opened=False,test_opened=False,
        status="rejected_by_train_cv", bulk_generation_approved=False,
        candidate_profile_sha256=frozen_hash)
    if gate["passed"]:
        val_x,val_meta,val=real_partition(dataset,"validation",manifest,cfg)
        flights=generate_flights(24,141000,GuidanceProfile(**selected["guidance"]))
        variants={}
        for profile in (replace(control,name="control"),candidate):
            observed=[observe_flight(f,profile) for f in flights]
            bundle=simulated_windows(observed,cfg)
            comparison=compare(val,bundle[1],metric_scales(real_metrics))
            comparison.to_csv(output/f"validation_{profile.name}.csv",index=False)
            variants[profile.name]=dict(distance=score(comparison),usable=float(bundle[2].usable.mean()),
                mmd=sequence_mmd(val_x,val_meta,bundle[0],bundle[1],real_x,real_meta))
        result.update(validation_opened=True,validation=variants,
            status="development_candidate_only" if variants[candidate.name]["distance"] < variants["control"]["distance"] else "validation_rejected")
    if file_hash(output/"candidate_profile.json") != frozen_hash:
        raise ValueError("Frozen candidate changed")
    write_json(output/"results.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("research/output/sequence_handoff_20260928/real"))
    parser.add_argument("--reference",type=Path,default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--output",type=Path,default=Path("research/output/wing_observation_v1"))
    args=parser.parse_args()
    run(args.dataset,args.reference,args.output)
