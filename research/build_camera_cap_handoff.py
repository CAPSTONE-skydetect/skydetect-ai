"""Explore bounded camera ranges and package a versioned development-only C handoff."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import real_partition, save_cohort, simulated_windows
from .evaluate_sequence_handoff import (aggregate_scores, balanced_group_weights,
                                        load_arrays, metrics, predict, read_json)
from .io import write_json
from .refine_sequence_calibration import generate_flights
from .sequence_comparison import compare, metric_scales, score
from .sequence_simulator import AcquisitionProfile, GuidanceProfile, observe_flight
from .trajectory_sequence import CHANNELS, CONTRACT_VERSION, SequenceConfig


CAPS_M = (1500., 500., 400., 300., 200.)
SYNTHETIC_MASSES = (.10, .25)


def model_with_synthetic(base_model, real, synthetic, fraction):
    """Use the frozen real-trained MiniRocket/scaler; refit only Ridge."""
    real_z = base_model["scaler"].transform(base_model["rocket"].transform(real["X"])).astype(np.float64)
    sim_z = base_model["scaler"].transform(base_model["rocket"].transform(synthetic["X"])).astype(np.float64)
    n = len(real["y"])
    weights = np.r_[balanced_group_weights(real["y"], real["group_id"], n*(1-fraction)),
                    balanced_group_weights(synthetic["y"], synthetic["group_id"], n*fraction)]
    ridge = RidgeClassifier(alpha=base_model["alpha"]).fit(
        np.concatenate([real_z, sim_z]), np.concatenate([real["y"], synthetic["y"]]),
        sample_weight=weights)
    return dict(base_model, classifier=ridge, synthetic_mass=fraction)


def grouped_f1(model, validation):
    windows, tracks, groups = aggregate_scores(validation, predict(model, validation["X"]))
    return dict(window=metrics(windows), track=metrics(tracks), group=metrics(groups))


def package(output, old, real, validation, synthetic, profile, winner, result, models):
    target = output/"C_handoff_camera_cap_v2"
    target.mkdir()
    n = len(real["y"])
    real_weights = balanced_group_weights(real["y"], real["group_id"], n)
    np.savez_compressed(target/"train_real.npz", **real,
                        sample_weight=real_weights, domain=np.full(n, "real"))
    np.savez_compressed(target/"synthetic_train_experimental.npz", **synthetic)
    combined = {key: np.concatenate([real[key], synthetic[key]]) for key in real}
    weights = np.r_[real_weights*.75,
                    balanced_group_weights(synthetic["y"], synthetic["group_id"], n*.25)]
    np.savez_compressed(target/"train_real_plus_synthetic.npz", **combined,
                        sample_weight=weights,
                        domain=np.r_[np.full(n, "real"), np.full(len(synthetic["y"]), "synthetic")])
    shutil.copy2(old/"validation.npz", target/"validation.npz")
    # Same historically inspected real test; it is not re-evaluated in this run.
    shutil.copy2(old/"test.npz", target/"test_historical_do_not_tune.npz")
    for name in ("real_train_validation_metadata.csv", "real_test_metadata.csv",
                 "source_inventory.csv", "trajectory_sequence.py", "minirocket_inference.py",
                 "requirements-minirocket.txt"):
        shutil.copy2(old/name, target/name)
    shutil.copy2(output/f"synthetic_{int(winner)}m_metrics.csv", target/"synthetic_metadata.csv")
    selected = result["selected_training_arm"]
    model = models[selected]
    joblib.dump(model, target/"reference_model.joblib", compress=3)
    write_json(target/"simulator_profile.json", dict(
        asdict(profile), source="Prior C handoff profile with a bounded camera distance range"))
    write_json(target/"input_contract.json", dict(
        version=CONTRACT_VERSION, fingerprint=SequenceConfig().fingerprint,
        shape="(N,4,60)", dtype="float32", channels=list(CHANNELS), labels=["bird", "drone"],
        score="Ridge margin; not a probability", selected_training_arm=selected,
        synthetic_train_cap_m=winner, synthetic_mass_for_mixed_train=.25,
        real_test_is_historical=True, real_test_re_evaluated=False,
        independent_evaluation_ready=False, sessions_verified=False,
        candidate_selection="Exploratory on previously used real validation; not a final generalization estimate",
        selected_for_C_by="Real validation source-video macro-F1; ties favor real-only",
        train_files_are_alternative_arms=True))
    write_json(target/"evaluation_results.json", result)
    shutil.copy2(output/"camera_cap_scores.csv", target/"camera_cap_scores.csv")
    lines = ["# B to C: camera-range experiment v2", "",
             "This is a **new experimental package**. It does not replace the previous C handoff ZIP.",
             "The real train/validation/test split and 2-second input contract are unchanged.",
             "Camera ranges 50-200, 50-300, 50-400 and 50-500 m were explored;",
             "20-1500 m was retained only as a historical control.", "",
             "## Selection", "",
             f"Packaged synthetic range cap: **{winner:.0f} m**.",
             f"Reference model arm: **{selected}**.",
             f"Real-only validation source-video macro-F1: **{result['real_only']['group']['macro_f1']:.4f}**.",
             f"Best mixed validation source-video macro-F1: **{result['best_mixed']['group']['macro_f1']:.4f}**.",
             "All candidate scores are in camera_cap_scores.csv and evaluation_results.json.",
             "Repeated use of the same small validation set makes this exploratory, not an independent result.",
             "The historical real test is included only for interface compatibility and was NOT re-evaluated.",
             "Do not select settings with that test or call it a new independent holdout.", "",
             "## Files", "",
             "- train_real.npz: unchanged real-only train arm.",
             "- synthetic_train_experimental.npz: new 2-second synthetic windows.",
             "- train_real_plus_synthetic.npz: alternative 75/25 weighted mixed train arm.",
             "- validation.npz: same previously used real development validation.",
             "- test_historical_do_not_tune.npz: same previously inspected real test; no new result.",
             "- reference_model.joblib: selected development model; load trusted files only.",
             "- input_contract.json, simulator_profile.json, synthetic_metadata.csv: version and provenance.",
             "- SHA256SUMS.json: hashes for all packaged files.", "",
             "The input is float32 (N,4,60), channels q_x/q_y/d_x/d_y. No bbox is used.",
             "Source-video groups and all derived windows remain together. Unknown session independence",
             "and missing original-video hashes remain limitations. No new real data were created.", ""]
    (target/"README.md").write_text("\n".join(lines), encoding="utf-8")
    checksums = {str(p.relative_to(target)).replace("\\", "/"): file_hash(p)
                 for p in target.rglob("*") if p.is_file()}
    write_json(target/"SHA256SUMS.json", checksums)
    archive = output/"C_handoff_camera_cap_v2.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                z.write(path, arcname=str(path.relative_to(output)))
    write_json(output/"handoff_artifact.json", dict(zip=archive.name, sha256=file_hash(archive),
               selected_training_arm=selected, synthetic_cap_m=winner,
               independent_evaluation_ready=False, real_test_re_evaluated=False))
    return archive


def build(old_folder, output, count=60, seed_start=331000):
    old_folder, output = Path(old_folder), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    if count < 12 or count % 3:
        raise ValueError("Flight count per class must be a multiple of three and at least 12")
    old = old_folder/"C_handoff"
    old_contract = read_json(old/"input_contract.json")
    if old_contract["fingerprint"] != SequenceConfig().fingerprint:
        raise ValueError("Real sequence contract changed")
    real, validation = load_arrays(old/"train_real.npz"), load_arrays(old/"validation.npz")
    if set(real["group_id"]) & set(validation["group_id"]):
        raise ValueError("Real train/validation source group overlap")
    profile_data = read_json(old_folder/"selected_profile.json")
    profile = AcquisitionProfile.from_saved(profile_data["profile"])
    guidance = GuidanceProfile(**profile_data["guidance"])
    base_model = joblib.load(old/"reference_model.joblib")
    if base_model["synthetic_mass"] != 0 or base_model["contract_id"] != SequenceConfig().fingerprint:
        raise ValueError("Reference model is not the real-only baseline")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(camera_caps_m=list(CAPS_M), synthetic_masses=list(SYNTHETIC_MASSES),
                    flights_per_class=count, seed_start=seed_start,
                    profile_source_sha256=file_hash(old_folder/"selected_profile.json"),
                    baseline_model_sha256=file_hash(old/"reference_model.joblib"),
                    real_train_sha256=file_hash(old/"train_real.npz"),
                    real_validation_sha256=file_hash(old/"validation.npz"),
                    contract_id=SequenceConfig().fingerprint,
                    selection="Among 50-200/300/400/500m: highest reused real validation source-video macro-F1 at fixed .25 synthetic mass; ties favor wider cap",
                    validation_previously_used=True, test_opened=False,
                    no_independent_generalization_claim=True)
    write_json(output/"protocol.json", protocol)
    real_folder = old_folder/"real"
    real_manifest = read_json(real_folder/"dataset_manifest.json")
    real_x, _, real_metrics = real_partition(real_folder, "train", real_manifest, SequenceConfig())
    if not np.array_equal(real_x, real["X"]):
        raise ValueError("Packaged real train differs from verified source")
    scales = metric_scales(real_metrics)
    flights = generate_flights(count, seed_start, guidance)
    baseline_results = grouped_f1(base_model, validation)
    rows, models = [], {"real_only": base_model}
    for cap in CAPS_M:
        selected_profile = replace(profile, min_camera_distance_m=20. if cap == 1500. else 50.,
                                   max_camera_distance_m=cap,
                                   name=f"ground_camera_cap_{int(cap)}m")
        records = [observe_flight(flight, selected_profile) for flight in flights]
        bundle = simulated_windows(records, SequenceConfig())
        if set(bundle[1].label) != {"bird", "drone"}:
            raise ValueError(f"Camera cap {cap} lost a class")
        save_cohort(output, f"synthetic_{int(cap)}m", records, bundle)
        distance = score(compare(real_metrics, bundle[1], scales))
        usable = float(bundle[2].usable.mean())
        synthetic = load_arrays(output/f"synthetic_{int(cap)}m.npz")
        for fraction in SYNTHETIC_MASSES:
            model = model_with_synthetic(base_model, real, synthetic, fraction)
            result = grouped_f1(model, validation)
            name = f"cap_{int(cap)}m_mass_{int(fraction*100)}"
            models[name] = model
            rows.append(dict(cap_m=cap, synthetic_mass=fraction, synthetic_windows=len(synthetic["X"]),
                             usable_flights=int(bundle[2].usable.sum()), attempted_flights=len(records),
                             usable_fraction=usable, train_distribution_distance=distance,
                             validation_group_macro_f1=result["group"]["macro_f1"],
                             validation_track_macro_f1=result["track"]["macro_f1"],
                             validation_window_macro_f1=result["window"]["macro_f1"],
                             bird_recall=result["group"]["recall"]["bird"],
                             drone_recall=result["group"]["recall"]["drone"], model_key=name))
            print(f"cap {cap:.0f} mass {fraction:.2f}: validation group F1={rows[-1]['validation_group_macro_f1']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/"camera_cap_scores.csv", index=False)
    fixed = table[(table.synthetic_mass == .25) & (table.cap_m <= 500.)].sort_values(
        ["validation_group_macro_f1", "cap_m"], ascending=[False, False])
    winner = float(fixed.iloc[0].cap_m)
    best = fixed.iloc[0]
    selected_arm = best.model_key if best.validation_group_macro_f1 > baseline_results["group"]["macro_f1"] else "real_only"
    result = dict(real_only=baseline_results, best_mixed=grouped_f1(models[best.model_key], validation),
                  selected_training_arm=selected_arm, chosen_synthetic_cap_m=winner,
                  chosen_mixed_model_key=best.model_key,
                  improvement_over_real_only=float(best.validation_group_macro_f1-baseline_results["group"]["macro_f1"]),
                  synthetic_use_approved=False, validation_previously_used=True,
                  real_test_re_evaluated=False, independent_evaluation_ready=False)
    write_json(output/"development_results.json", result)
    selected_profile = replace(profile, min_camera_distance_m=20. if winner == 1500. else 50.,
                               max_camera_distance_m=winner,
                               name=f"ground_camera_cap_{int(winner)}m")
    archive = package(output, old, real, validation, load_arrays(output/f"synthetic_{int(winner)}m.npz"),
                      selected_profile, winner, result, models)
    result["archive"] = str(archive)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-folder", type=Path, default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--output", type=Path, default=Path("research/output/camera_cap_handoff_20261005"))
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed-start", type=int, default=331000)
    args = parser.parse_args()
    print(json.dumps(build(args.old_folder, args.output, args.count, args.seed_start), indent=2), flush=True)
