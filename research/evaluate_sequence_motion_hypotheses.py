"""Bounded train-group audit of camera range and physical flap hypotheses."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import real_partition, simulated_windows
from .io import write_json
from .measure_train_motion import train_tracks
from .prepare_sequence_handoff import fit_profiles, read_json
from .refine_sequence_calibration import generate_flights, grouped_folds
from .sequence_comparison import compare, metric_scales, score
from .sequence_simulator import BIRDS, latent_flight, observe_flight, SEQUENCE_SIMULATOR_VERSION
from .trajectory_sequence import SequenceConfig


CANDIDATES = {
    "baseline": (1500., 1.),
    "camera_500m": (500., 1.),
    "camera_400m": (400., 1.),
    "camera_300m": (300., 1.),
    "camera_200m": (200., 1.),
    "flap_force_1p5": (1500., 1.5),
    "camera_300m_flap_1p5": (300., 1.5),
}


def evaluate_candidate(flights, profile, cfg, max_distance, flap_scale, guidance):
    selected = replace(profile, min_camera_distance_m=20., max_camera_distance_m=max_distance)
    records = []
    for flight in flights:
        if flight["label"] == "bird" and flap_scale != 1.:
            flight = latent_flight("bird", flight["subtype"], flight["seed"],
                                   guidance=guidance, bird_flap_force_scale=flap_scale)
        records.append(observe_flight(flight, selected))
    bundle = simulated_windows(records, cfg)
    if set(bundle[1].label) != {"bird", "drone"}:
        raise ValueError("Candidate lost an entire class")
    return records, bundle


def summarize(name, fold, real, fit_real, records, bundle):
    detail = compare(real, bundle[1], metric_scales(fit_real))
    usable = float(bundle[2].usable.mean())
    rows = []
    for label in ("bird", "drone"):
        part = detail[detail.label == label].set_index("metric")
        rows.append(dict(fold=fold, candidate=name, label=label,
                         distance=score(detail),
                         label_distance=float(part.groupby("category").normalized_distance.mean().mean()),
                         objective=score(detail) + 2*(1-usable), usable=usable,
                         screen_span_real=float(part.loc["screen_span", "real_median"]),
                         screen_span_sim=float(part.loc["screen_span", "sim_median"]),
                         band_3_10_real=float(part.loc["band_3_10_ratio", "real_median"]),
                         band_3_10_sim=float(part.loc["band_3_10_ratio", "sim_median"]),
                         low_motion_real=float(part.loc["low_motion_fraction", "real_median"]),
                         low_motion_sim=float(part.loc["low_motion_fraction", "sim_median"]),
                         over_cap=int(sum(r["metadata"]["camera_cap_exceeded_for_ground_geometry"]
                                          for r in records if r["metadata"]["label"] == label))))
    return rows


def select_candidate(table, *, tolerance=.05):
    unique = table.drop_duplicates(["fold", "candidate"])
    folded = unique.pivot(index="fold", columns="candidate", values="objective")
    distances = unique.pivot(index="fold", columns="candidate", values="distance")
    baseline = folded["baseline"]
    admitted = [name for name in folded if name != "baseline"
                and folded[name].mean() < baseline.mean()
                and distances[name].mean() < distances["baseline"].mean()
                and (folded[name]-baseline).max() <= tolerance]
    return min(admitted, key=lambda name: folded[name].mean()) if admitted else None


def evaluate(dataset, output, count=12, seed_start=219000, train_only=False):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    if count < 12:
        raise ValueError("Need at least 12 flights per class and fold")
    manifest = read_json(dataset/"dataset_manifest.json")
    cfg = SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Sequence contract changed")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(generator=SEQUENCE_SIMULATOR_VERSION, contract_id=cfg.fingerprint,
                    candidates={k: dict(max_camera_distance_m=v[0], bird_flap_force_scale=v[1])
                                for k, v in CANDIDATES.items()},
                    train_folds=3, flights_per_class_per_fold=count,
                    seed_start=seed_start, validation_seed_start=229000,
                    gate="Lower mean distance and objective; no fold objective >0.05 worse",
                    objective="Equal-class/category normalized Wasserstein + 2*(1-usable flight fraction)",
                    real_source_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    validation_reused=True, validation_permitted=not train_only, test_opened=False,
                    interpretation="Bounded development hypotheses, not a realism certificate")
    write_json(output/"protocol.json", protocol)
    _, metadata, real = real_partition(dataset, "train", manifest, cfg)
    tracks = list(train_tracks(manifest))
    rows = []
    for fold, (fit, held) in enumerate(grouped_folds(metadata)):
        fit_real, held_real = real[real.group_id.isin(fit)], real[real.group_id.isin(held)]
        subset = dict(manifest, sources=[r for r in manifest["sources"]
                                         if r["split"] == "train" and r["source_group_id"] in fit])
        profiles, guidance = fit_profiles(fit_real, subset,
                                          [(r, t) for r, t in tracks if r["source_group_id"] in fit])
        profile = profiles[1]
        flights = generate_flights(count, protocol["seed_start"] + fold*1000, guidance)
        for name, (distance, flap_scale) in CANDIDATES.items():
            records, bundle = evaluate_candidate(flights, profile, cfg, distance, flap_scale, guidance)
            rows.extend(summarize(name, fold, held_real, fit_real, records, bundle))
            print(f"fold {fold} {name}: objective={rows[-1]['objective']:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/"train_cv.csv", index=False)
    chosen = select_candidate(table)
    result = dict(chosen_train_candidate=chosen, train_means=table.drop_duplicates(
        ["fold", "candidate"]).groupby("candidate")[["distance", "objective", "usable"]].mean().to_dict("index"),
        validation_opened=False, test_opened=False, calibration_approved=False,
        status="rejected_in_train_cv" if chosen is None else "train_gate_passed_not_validated")
    write_json(output/"decision.json", result)
    if chosen is None or train_only:
        return result
    profiles, guidance = fit_profiles(real, manifest, tracks)
    profile = profiles[1]
    selected = dict(name=chosen, profile=asdict(profile), candidate=protocol["candidates"][chosen],
                    selected_before_validation=True, protocol_sha256=file_hash(output/"protocol.json"))
    write_json(output/"selected_profile.json", selected)
    frozen_hash = file_hash(output/"selected_profile.json")
    _, _, validation = real_partition(dataset, "validation", manifest, cfg)
    flights = generate_flights(count, protocol["validation_seed_start"], guidance)
    validation_rows = []
    for name in ("baseline", chosen):
        distance, flap_scale = CANDIDATES[name]
        records, bundle = evaluate_candidate(flights, profile, cfg, distance, flap_scale, guidance)
        validation_rows.extend(summarize(name, "validation", validation, real, records, bundle))
    pd.DataFrame(validation_rows).to_csv(output/"validation.csv", index=False)
    if file_hash(output/"selected_profile.json") != frozen_hash:
        raise ValueError("Selected profile changed after opening validation")
    values = pd.DataFrame(validation_rows).drop_duplicates("candidate").set_index("candidate")
    result.update(validation_opened=True,
                  validation_results=values[["distance", "objective", "usable"]].to_dict("index"),
                  status="development_candidate_only" if (
                      values.loc[chosen, "objective"] < values.loc["baseline", "objective"]
                      and values.loc[chosen, "distance"] < values.loc["baseline", "distance"])
                  else "rejected_in_validation_distribution")
    write_json(output/"decision.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/sequence_handoff_20260928/real"))
    parser.add_argument("--output", type=Path, default=Path("research/output/sequence_motion_hypotheses_v1"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=219000)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.output, args.count, args.seed_start,
                              args.train_only), indent=2), flush=True)
