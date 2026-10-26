"""Compare archived 1.4 observations with paired 1.6 camera-range observations."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import real_partition, simulated_windows
from .io import write_json
from .sequence_comparison import compare, metric_scales, score, sequence_mmd
from .sequence_simulator import AcquisitionProfile, GuidanceProfile, latent_flight, observe_flight
from .trajectory_sequence import SequenceConfig


def _archive_arrays(path, metadata):
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["sample_id"], metadata.sample_id.to_numpy()):
            raise ValueError("Archived sample IDs do not match metrics")
        return archive["X"].copy()


def _paired_partition(folder, stem, count, profile, guidance, cfg, real_x, real_meta,
                      real_metrics, train_x, train_meta, scales, output):
    old_records_path = folder/f"{stem}_tracks.jsonl"
    old_records = [json.loads(line) for line in old_records_path.read_text(encoding="utf-8").splitlines()]
    old_metrics = pd.read_csv(folder/f"{stem}_metrics.csv", keep_default_na=False)
    old_attempts = pd.read_csv(folder/f"{stem}_attempts.csv")
    if len(old_records) != 2*count or len(old_attempts) != 2*count:
        raise ValueError(f"Archived {stem} flight count changed")
    old_x = _archive_arrays(folder/f"{stem}.npz", old_metrics)
    current_profile = replace(profile, name="paired_50_500m", min_camera_distance_m=50.,
                              max_camera_distance_m=500.)
    current_records = []
    for archived in old_records:
        info = archived["metadata"]
        flight = latent_flight(info["label"], info["subtype"], info["seed"], guidance=guidance)
        if flight["world_truth"] != archived["world_truth"]:
            raise ValueError(f"Latent flight changed at seed {info['seed']}")
        replay = observe_flight(flight, profile)
        if (replay["track"]["history"] != archived["track"]["history"] or
                replay["optical_truth"] != archived["optical_truth"]):
            raise ValueError(f"Archived 1.4 observation cannot be replayed at seed {info['seed']}")
        current_records.append(observe_flight(flight, current_profile))
    current_x, current_metrics, current_attempts, _ = simulated_windows(current_records, cfg)
    if set(current_metrics.label) != {"bird", "drone"}:
        raise ValueError("Current camera range lost a class")
    before = compare(real_metrics, old_metrics, scales)
    after = compare(real_metrics, current_metrics, scales)
    before["version"], after["version"] = "1.4.0", "1.6.0"
    pd.concat([before, after], ignore_index=True).to_csv(output/f"{stem}_metrics_comparison.csv", index=False)
    current_metrics.to_csv(output/f"{stem}_current_window_metrics.csv", index=False)
    current_attempts.to_csv(output/f"{stem}_current_flights.csv", index=False)
    old_mmd = sequence_mmd(real_x, real_meta, old_x, old_metrics,
                           train_x, train_meta)
    new_mmd = sequence_mmd(real_x, real_meta, current_x, current_metrics,
                           train_x, train_meta)
    result = dict(archived_generator="sequence-guidance-1.4.0",
                  current_generator="sequence-guidance-1.6.0",
                  archived_range_m=[20, 1500], current_range_m=[50, 500],
                  same_latent_flights_and_historical_observations_replayed=len(old_records),
                  before=dict(distance=score(before), usable_flights=int(old_attempts.usable.sum()),
                              attempted_flights=len(old_attempts), windows=len(old_metrics), mmd2=old_mmd),
                  after=dict(distance=score(after), usable_flights=int(current_attempts.usable.sum()),
                             attempted_flights=len(current_attempts), windows=len(current_metrics), mmd2=new_mmd))
    write_json(output/f"{stem}_summary.json", result)
    print(f"{stem}: {result['before']['distance']:.4f} -> "
          f"{result['after']['distance']:.4f} distance", flush=True)
    return result


def compare_versions(folder, output):
    folder, output = Path(folder), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    protocol = json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    if protocol["generator"] != "sequence-guidance-1.4.0":
        raise ValueError("Expected the archived 1.4 handoff")
    real_folder = folder/"real"
    manifest = json.loads((real_folder/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    train_x, train_meta, train_metrics = real_partition(real_folder, "train", manifest, cfg)
    scales = metric_scales(train_metrics)
    saved = json.loads((folder/"selected_profile.json").read_text(encoding="utf-8"))
    historical_profile = AcquisitionProfile.from_saved(saved["profile"])
    guidance = GuidanceProfile(**saved["guidance"])
    if (historical_profile.min_camera_distance_m,
            historical_profile.max_camera_distance_m) != (20., 1500.):
        raise ValueError("Historical camera range mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"protocol.json", dict(
        archived_protocol_sha256=file_hash(folder/"protocol.json"),
        archived_profile_sha256=file_hash(folder/"selected_profile.json"),
        real_manifest_sha256=file_hash(real_folder/"dataset_manifest.json"),
        latent_and_historical_observation_replay_required=True,
        same_real_dataset_and_metric_scaling=True, test_opened=False,
        validation_previously_used=True,
        interpretation="Paired camera-range effect, not C classifier performance"))
    training = _paired_partition(folder, "synthetic_train", protocol["training_count_per_class"],
                                 historical_profile, guidance, cfg, train_x, train_meta,
                                 train_metrics, train_x, train_meta, scales, output)
    val_x, val_meta, val_metrics = real_partition(real_folder, "validation", manifest, cfg)
    validation_count = len(pd.read_csv(folder/"ground_span2.0_validation_attempts.csv"))//2
    validation = _paired_partition(folder, "ground_span2.0_validation", validation_count,
                                   historical_profile, guidance, cfg, val_x, val_meta,
                                   val_metrics, train_x, train_meta, scales, output)
    summary = dict(training=training, validation=validation,
                   test_opened=False, C_model_retrained=False, independent_evaluation_ready=False)
    write_json(output/"result.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path,
                        default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/output/paired_1p4_vs_1p6_20261005"))
    args = parser.parse_args()
    print(json.dumps(compare_versions(args.folder, args.output), indent=2), flush=True)
