"""Audit source groups and test an above-ground camera using train-only CV.

This prepares a bounded experimental cohort, not an approved bulk dataset.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash, group_records
from .calibrate_sequence_simulator import estimate_acquisition, real_partition, save_cohort, simulated_windows
from .io import write_json
from .measure_train_motion import train_tracks
from .refine_sequence_calibration import candidate_profile, fit_motion_priors, generate_flights, grouped_folds
from .sequence_comparison import compare, metric_scales, score, sequence_mmd
from .sequence_simulator import observe_flight, SEQUENCE_SIMULATOR_VERSION
from .trajectory_sequence import SequenceConfig, CONTRACT_VERSION


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_groups(manifest, uploads):
    """Metadata/byte hashes only: no validation/test coordinates are parsed."""
    records = deepcopy(manifest["sources"])
    checked, recovered = [], []
    for rec in records:
        paths = sorted(Path(uploads).glob(f"{rec['source_video_id']}.*")) if rec["source_video_id"] else []
        hashes = {file_hash(p) for p in paths if p.is_file()}
        if len(hashes) > 1:
            raise ValueError("Ambiguous source video files")
        if hashes:
            actual = next(iter(hashes))
            if rec["video_sha256"] and actual != rec["video_sha256"]:
                raise ValueError("Original video hash changed")
            if not rec["video_sha256"]:
                recovered.append(rec["parent_track_id"])
            rec["video_sha256"] = actual
        checked.append(dict(parent_track_id=rec["parent_track_id"], split=rec["split"],
                            file_name=rec["file_name"], video_present=bool(hashes),
                            video_sha256=rec["video_sha256"]))
    old = {r["parent_track_id"]: r["source_group_id"] for r in records}
    links = group_records(records)
    splits = {}
    for r in records:
        splits.setdefault(r["source_group_id"], set()).add(r["split"])
    cross = {g: sorted(s) for g, s in splits.items() if len(s) > 1}
    changed = [r["parent_track_id"] for r in records if old[r["parent_track_id"]] != r["source_group_id"]]
    return records, dict(evidence_links=links, cross_split_groups=cross, changed_groups=changed,
        recovered_hash_tracks=recovered, video_checks=checked,
        train_without_video_hash=sum(r["split"] == "train" and not r["video_sha256"] for r in records),
        source_video_groups_verified=not cross,
        user_confirmation="Within each filename series, equal numbers identify one source video; parentheses identify different objects, not independent recordings.",
        session_independence_verified=False,
        limitation="Different source videos may share a recording session. Missing videos cannot be byte-verified. All real data were historically inspected development data.")


def fit_profiles(real, manifest, tracks):
    base, _, _ = estimate_acquisition(real, manifest)
    measured, guidance, _, _, _ = fit_motion_priors(tracks, base)
    control = candidate_profile(base, measured, "legacy", "old_ar1", 2.)
    profiles = [control] + [replace(control, name=f"ground_span{span}", ground_camera=True,
                                   span_multiplier=span) for span in (2., 3.)]
    return profiles, guidance


def camera_evidence(records):
    rows = []
    for r in records:
        m = r["metadata"]
        pos = m["camera"]["position"]
        rows.append(dict(label=m["label"], seed=m["seed"], height_m=pos[2],
                         range_m=m["camera_distance_m"], fov_deg=m["camera"]["horizontal_fov_deg"],
                         elevation_deg=m["actual_elevation_deg"]))
    return pd.DataFrame(rows)


def prepare(dataset, output, count=18, training_count=60):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    if min(count, training_count) < 6:
        raise ValueError("Need at least six flights per class")
    manifest = read_json(dataset/"dataset_manifest.json")
    cfg = SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"] or manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError("Wrong preprocessing contract")
    output.mkdir(parents=True)
    records, audit = audit_groups(manifest, Path("storage/uploads"))
    write_json(output/"group_audit.json", audit)
    if audit["cross_split_groups"] or audit["changed_groups"]:
        raise ValueError("New group evidence requires rebuilding fixed splits before fitting")
    manifest["sources"] = records
    real_dir = output/"real"
    real_dir.mkdir()
    for name in ("train.npz", "validation.npz", "metadata.csv"):
        if file_hash(dataset/name) != manifest["artifact_hashes"][name]:
            raise ValueError("Input artifact changed")
        shutil.copy2(dataset/name, real_dir/name)
    manifest.update(previous_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    group_status="source_video_rule_confirmed_sessions_unverified",
                    independent_evaluation_ready=False, release_created_at=datetime.now(timezone.utc).isoformat())
    write_json(real_dir/"dataset_manifest.json", manifest)
    pd.DataFrame(audit["video_checks"]).to_csv(output/"source_review.csv", index=False, encoding="utf-8-sig")
    train_x, train_meta, real = real_partition(real_dir, "train", manifest, cfg)
    tracks = list(train_tracks(manifest))
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, snapshot/p.name)
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(), contract_version=CONTRACT_VERSION,
        contract_id=cfg.fingerprint, generator=SEQUENCE_SIMULATOR_VERSION,
        train_cv_candidates=["legacy", "ground_span2.0", "ground_span3.0"],
        selection="Lowest mean distance + 2*(1-usable) among physical ground-camera candidates; report versus legacy even if worse",
        shared_camera_height_m=1.5, minimum_ground_fov_deg=5.,
        camera_priors="Design constraints, not recovered physical camera intrinsics or distance",
        count_per_class_per_fold=count, training_count_per_class=training_count,
        folds_seed_start=61000, validation_sim_seed_start=121000, training_seed_start=131000,
        test_opened=False, validation_reused=True, bulk_generation_approved=False,
        real_manifest_sha256=file_hash(real_dir/"dataset_manifest.json"),
        code_hashes={p.name:file_hash(p) for p in snapshot.glob("*.py")})
    write_json(output/"protocol.json", protocol)
    rows, evidence = [], []
    for fold, (fit, held) in enumerate(grouped_folds(train_meta)):
        subset = dict(manifest, sources=[r for r in records if r["split"] == "train" and r["source_group_id"] in fit])
        fit_real, held_real = real[real.group_id.isin(fit)], real[real.group_id.isin(held)]
        profiles, guidance = fit_profiles(fit_real, subset, [(r,t) for r,t in tracks if r["source_group_id"] in fit])
        flights = generate_flights(count, 61000+1000*fold, guidance)
        evidence.append(dict(fold=fold, fit_groups=sorted(fit), held_groups=sorted(held),
                             profiles=[asdict(p) for p in profiles], guidance=asdict(guidance)))
        for profile in profiles:
            simulated = [observe_flight(f, profile) for f in flights]
            bundle = simulated_windows(simulated, cfg)
            detail = compare(held_real, bundle[1], metric_scales(fit_real))
            detail.to_csv(output/f"fold{fold}_{profile.name}.csv", index=False)
            cameras = camera_evidence(simulated)
            cameras.to_csv(output/f"cameras_fold{fold}_{profile.name}.csv", index=False)
            distance, usable = score(detail), float(bundle[2].usable.mean())
            rows.append(dict(fold=fold, variant=profile.name, distance=distance, usable=usable,
                             objective=distance+2*(1-usable), below_ground=int((cameras.height_m < 0).sum())))
            print(f"fold {fold} {profile.name}: distance={distance:.4f} usable={usable:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/"camera_cv.csv", index=False)
    write_json(output/"fold_evidence.json", evidence)
    candidates = table[table.variant != "legacy"].groupby("variant").objective.mean()
    chosen = str(candidates.idxmin())
    profiles, guidance = fit_profiles(real, manifest, tracks)
    selected = next(p for p in profiles if p.name == chosen)
    freeze = dict(profile=asdict(selected), guidance=asdict(guidance),
                  selected_before_validation=True, protocol_sha256=file_hash(output/"protocol.json"))
    write_json(output/"selected_profile.json", freeze)
    frozen_hash = file_hash(output/"selected_profile.json")
    val_x, val_meta, val = real_partition(real_dir, "validation", manifest, cfg)
    flights = generate_flights(24, 121000, guidance)
    comparisons, bundles = [], {}
    for profile in (profiles[0], selected):
        simulated = [observe_flight(f, profile) for f in flights]
        bundle = simulated_windows(simulated, cfg)
        save_cohort(output, profile.name+"_validation", simulated, bundle)
        camera_evidence(simulated).to_csv(output/f"cameras_{profile.name}_validation.csv", index=False)
        detail = compare(val, bundle[1], metric_scales(real))
        comparisons.append(detail.assign(variant=profile.name))
        bundles[profile.name] = dict(distance=score(detail), usable=float(bundle[2].usable.mean()),
            sequence_mmd=sequence_mmd(val_x,val_meta,bundle[0],bundle[1],train_x,train_meta))
    pd.concat(comparisons).to_csv(output/"camera_validation.csv", index=False)
    simulated = [observe_flight(f, selected) for f in generate_flights(training_count, 131000, guidance)]
    bundle = simulated_windows(simulated, cfg)
    save_cohort(output, "synthetic_train", simulated, bundle)
    camera_evidence(simulated).to_csv(output/"cameras_synthetic_train.csv", index=False)
    summary = dict(chosen=chosen, cv=table.groupby("variant")[["distance","objective","usable"]].mean().to_dict("index"),
                   validation=bundles, synthetic_windows=len(bundle[0]), synthetic_flights=len(simulated),
                   synthetic_usable_flights=int(bundle[2].usable.sum()),
                   status="experimental_downstream_ablation_only", independent_evaluation_ready=False,
                   bulk_generation_approved=False, test_opened=False, frozen_profile_sha256=frozen_hash)
    if frozen_hash != file_hash(output/"selected_profile.json"):
        raise ValueError("Frozen calibration changed")
    write_json(output/"preparation_results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/real_sequences_clock_v1_1"))
    parser.add_argument("--output", type=Path, default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--training-count", type=int, default=60)
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.count, args.training_count)
