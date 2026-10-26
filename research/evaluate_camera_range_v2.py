"""Paired real/synthetic audit for historical and current camera distance ranges."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import real_partition, simulated_windows
from .evaluate_sequence_motion_hypotheses import summarize
from .io import write_json
from .measure_train_motion import train_tracks
from .prepare_sequence_handoff import fit_profiles, read_json
from .refine_sequence_calibration import generate_flights, grouped_folds
from .sequence_comparison import compare, metric_scales, score
from .sequence_simulator import SEQUENCE_SIMULATOR_VERSION, observe_flight
from .trajectory_sequence import SequenceConfig


RANGES = {"historical_20_1500m": (20., 1500.), "current_50_500m": (50., 500.)}


def evaluate_pair(flights, profile, cfg, real, fit_real, fold):
    rows, details, flights_status = [], [], []
    for name, (minimum, maximum) in RANGES.items():
        candidate = replace(profile, name=name, min_camera_distance_m=minimum,
                            max_camera_distance_m=maximum)
        records = [observe_flight(flight, candidate) for flight in flights]
        bundle = simulated_windows(records, cfg)
        if set(bundle[1].label) != {"bird", "drone"}:
            raise ValueError(f"Camera range {name} lost a class")
        detail = compare(real, bundle[1], metric_scales(fit_real))
        detail["fold"], detail["camera_range"] = fold, name
        details.append(detail)
        rows.extend(summarize(name, fold, real, fit_real, records, bundle))
        flights_status.append(dict(fold=fold, camera_range=name, attempted=len(records),
                                   usable=int(bundle[2].usable.sum()), windows=len(bundle[1]),
                                   distance=score(detail)))
        print(f"{fold} {name}: distance={score(detail):.4f}, "
              f"usable={bundle[2].usable.mean():.3f}", flush=True)
    return rows, details, flights_status


def evaluate(dataset, output, count=12, seed_start=341000, train_only=False):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    if count < 12 or count % 3:
        raise ValueError("Count per class must be a multiple of three and at least 12")
    manifest = read_json(dataset/"dataset_manifest.json")
    cfg = SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Real sequence contract mismatch")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(simulator_version=SEQUENCE_SIMULATOR_VERSION,
                    camera_ranges_m=RANGES, flights_per_class_per_fold=count,
                    train_seed_start=seed_start, validation_seed_start=seed_start+10000,
                    real_dataset_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    real_contract_id=cfg.fingerprint, train_group_folds=3,
                    validation_previously_used=not train_only, test_opened=False,
                    comparison="Same latent flights, guidance and observer seeds for both camera ranges",
                    interpretation="Descriptive distribution audit; not evidence of real-world equivalence")
    write_json(output/"protocol.json", protocol)
    _, metadata, real = real_partition(dataset, "train", manifest, cfg)
    tracks = list(train_tracks(manifest))
    rows, details, statuses = [], [], []
    for fold, (fit, held) in enumerate(grouped_folds(metadata)):
        fit_real, held_real = real[real.group_id.isin(fit)], real[real.group_id.isin(held)]
        subset = dict(manifest, sources=[r for r in manifest["sources"]
                                         if r["split"] == "train" and r["source_group_id"] in fit])
        profiles, guidance = fit_profiles(fit_real, subset,
                                          [(r, t) for r, t in tracks if r["source_group_id"] in fit])
        flights = generate_flights(count, seed_start+fold*1000, guidance)
        part = evaluate_pair(flights, profiles[1], cfg, held_real, fit_real, fold)
        rows.extend(part[0]); details.extend(part[1]); statuses.extend(part[2])
    pd.DataFrame(rows).to_csv(output/"train_cv_by_class.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(output/"train_cv_metrics.csv", index=False)
    pd.DataFrame(statuses).to_csv(output/"train_cv_flights.csv", index=False)
    train_summary = pd.DataFrame(statuses).groupby("camera_range").apply(
        lambda part: dict(mean_distance=float(part.distance.mean()),
                          usable_fraction=float(part.usable.sum()/part.attempted.sum()),
                          windows=int(part.windows.sum())), include_groups=False).to_dict()
    write_json(output/"train_summary.json", train_summary)
    if train_only:
        return dict(train_cv=train_summary, validation_opened=False, test_opened=False)
    profiles, guidance = fit_profiles(real, manifest, tracks)
    selected = dict(profile=asdict(profiles[1]), guidance=asdict(guidance),
                    ranges=RANGES, train_summary=train_summary,
                    selected_before_validation=True, protocol_sha256=file_hash(output/"protocol.json"))
    write_json(output/"frozen_before_validation.json", selected)
    frozen_hash = file_hash(output/"frozen_before_validation.json")
    _, _, validation = real_partition(dataset, "validation", manifest, cfg)
    flights = generate_flights(count, seed_start+10000, guidance)
    val_rows, val_details, val_status = evaluate_pair(
        flights, profiles[1], cfg, validation, real, "validation")
    if file_hash(output/"frozen_before_validation.json") != frozen_hash:
        raise ValueError("Frozen settings changed after opening validation")
    pd.DataFrame(val_rows).to_csv(output/"validation_by_class.csv", index=False)
    pd.concat(val_details, ignore_index=True).to_csv(output/"validation_metrics.csv", index=False)
    pd.DataFrame(val_status).to_csv(output/"validation_flights.csv", index=False)
    validation_summary = {r["camera_range"]: dict(distance=r["distance"],
                           usable_fraction=r["usable"]/r["attempted"], windows=r["windows"])
                          for r in val_status}
    result = dict(train_cv=train_summary, validation=validation_summary,
                  validation_previously_used=True, test_opened=False,
                  approved_for_C=False, independent_realism_proven=False)
    write_json(output/"result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("research/output/sequence_handoff_20260928/real"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/output/camera_range_v2_audit_20261005"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=341000)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.output, args.count, args.seed_start,
                              args.train_only), indent=2), flush=True)
