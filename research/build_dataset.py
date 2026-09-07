"""Build exact-size, class-balanced synthetic feature datasets."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from . import FEATURE_VERSION, SIMULATOR_VERSION
from .features import FEATURE_COLUMNS
from .io import write_json, write_jsonl
from .pipeline import BatchRunner, DRONE_SUBTYPES, SCENARIOS, SPECIES_CONFIG, stable_seed


def assign_train_test(families_by_scenario, seed, test_fraction=.2):
    """Split complete families while preserving every scenario in both sets."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    result = {}
    for scenario, families in sorted(families_by_scenario.items()):
        families = sorted(set(families))
        if len(families) < 2:
            raise ValueError("Each scenario needs at least two families")
        ordered = sorted(families, key=lambda x: stable_seed(seed, scenario, x, "dataset-split"))
        test_count = max(1, min(len(ordered)-1, int(round(len(ordered)*test_fraction))))
        result.update({family: ("test" if i < test_count else "train")
                       for i, family in enumerate(ordered)})
    return result


def _stable_order(table, seed, purpose):
    result = table.copy()
    result["_selection_order"] = [stable_seed(seed, purpose, value) for value in result.sample_id]
    return result.sort_values(["_selection_order", "sample_id"])


def select_balanced(table, count, seed, split):
    """Select exact rows, balanced by class and as evenly as possible by stratum."""
    if count <= 0 or count % 2:
        raise ValueError("Requested row count must be a positive even integer")
    candidates = table[(table.feature_status == "accepted") & (table.split == split)].copy()
    selected = []
    per_label = count // 2
    strata = ["scenario", "subtype", "observation_profile"]
    for label in ("bird", "drone"):
        labeled = candidates[candidates.label == label]
        groups = list(labeled.groupby(strata, observed=True, sort=True))
        if not groups:
            raise ValueError(f"No accepted {label} candidates for {split}")
        base, remainder = divmod(per_label, len(groups))
        used = set()
        for position, (_, group) in enumerate(groups):
            quota = base + int(position < remainder)
            ordered = _stable_order(group, seed, f"{split}:{label}:stratum")
            take = ordered.head(quota)
            selected.append(take)
            used.update(take.sample_id)
        chosen = pd.concat(selected, ignore_index=True)
        chosen_for_label = chosen[chosen.label == label]
        shortage = per_label-len(chosen_for_label)
        if shortage:
            unused = labeled[~labeled.sample_id.isin(used)]
            fill = _stable_order(unused, seed, f"{split}:{label}:fill").head(shortage)
            selected.append(fill)
            shortage -= len(fill)
        if shortage:
            detail = labeled.groupby(strata, observed=True).size().to_dict()
            raise ValueError(f"Insufficient accepted {label} candidates for {split}: need {shortage} more; {detail}")
    result = pd.concat(selected, ignore_index=True)
    # Earlier label selections are re-filtered because selected is cumulative.
    result = result.drop_duplicates("sample_id")
    final = []
    for label in ("bird", "drone"):
        final.append(_stable_order(result[result.label == label], seed, f"{split}:{label}:final").head(per_label))
    result = pd.concat(final, ignore_index=True).drop(columns="_selection_order", errors="ignore")
    if len(result) != count or result.label.value_counts().to_dict() != {"bird": per_label, "drone": per_label}:
        raise AssertionError("Exact balanced selection failed")
    return _stable_order(result, seed, f"{split}:shuffle").drop(columns="_selection_order").reset_index(drop=True)


def _row(sample, split):
    meta, feature = sample["metadata"], sample["feature_result"]
    quality = feature["quality"]
    row = {key: meta[key] for key in (
        "sample_id", "family_id", "label", "subtype", "scenario", "behavior_mode",
        "observation_profile", "frame_count", "simulator_version", "feature_version",
        "requested_depth_mode", "fps", "seed")}
    row.update(split=split,
               attempted_missing_fraction=meta["observation"]["quality"]["missing_ratio"],
               camera_residual_enabled=meta["observation"]["camera_motion"]["enabled"],
               tracking_drift_enabled=meta["observation"]["tracking_drift"]["enabled"],
               feature_status=feature["feature_status"],
               feature_config_id=feature["feature_config_id"],
               rejection_reason=";".join(feature["reasons"]),
               **{name: (feature["features"] or {}).get(name) for name in FEATURE_COLUMNS},
               **quality)
    return row


def build_dataset(output_dir, train_count=8000, test_count=2000, seed=20260906,
                  families_per_scenario=None, paired=True, preview_per_combination=1):
    """Generate candidates and write exact accepted train/test feature CSVs.

    Candidate failures remain in the ledger. Full raw trajectories are omitted;
    a small deterministic preview JSONL is retained for visual inspection.
    """
    if min(train_count, test_count) <= 0 or train_count % 2 or test_count % 2:
        raise ValueError("train_count and test_count must be positive even integers")
    output = Path(output_dir)
    for name in ("dataset_manifest.json", "manifest.json"):
        path = output/name
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))["simulator_version"] != SIMULATOR_VERSION:
            raise ValueError("Refusing to overwrite artifacts from a different simulator version")
    rows_per_family = (2 if paired else 1) * (len(SPECIES_CONFIG)+len(DRONE_SUBTYPES))
    if families_per_scenario is None:
        # 35% headroom covers ordinary visibility/dropout rejection without retries.
        families_per_scenario = math.ceil((train_count+test_count)/len(SCENARIOS)/rows_per_family/.8*1.35)
    if families_per_scenario < 2:
        raise ValueError("families_per_scenario must be at least two")
    runner = BatchRunner(output, seed=seed)
    families = {scenario: [f"{seed}:{scenario}:{i}" for i in range(families_per_scenario)]
                for scenario in SCENARIOS}
    assignments = assign_train_test(families, seed)
    ledger, previews, preview_keys = [], [], set()
    print(f"Generating {len(SCENARIOS)*families_per_scenario*rows_per_family} candidates...", flush=True)
    for scenario in SCENARIOS:
        for i in range(families_per_scenario):
            for label, subtypes in (("bird", tuple(SPECIES_CONFIG)), ("drone", DRONE_SUBTYPES)):
                for subtype in subtypes:
                    for noisy in ((False, True) if paired else (True,)):
                        sample = runner.simulate(scenario, label, subtype, i, noisy)
                        split = assignments[sample["metadata"]["family_id"]]
                        sample["metadata"]["split"] = split
                        ledger.append(_row(sample, split))
                        key = (scenario, subtype, sample["metadata"]["observation_profile"])
                        if preview_per_combination and key not in preview_keys:
                            previews.append(sample)
                            preview_keys.add(key)
            if (i+1) % 50 == 0 or i+1 == families_per_scenario:
                print(f"  {scenario}: {i+1}/{families_per_scenario} families", flush=True)
    ledger = pd.DataFrame(ledger)
    train = select_balanced(ledger, train_count, seed, "train")
    test = select_balanced(ledger, test_count, seed, "test")
    if set(train.family_id) & set(test.family_id):
        raise AssertionError("Family leakage between train and test")
    output.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output/"candidate_ledger_v3.csv", index=False)
    train.to_csv(output/"train_features_v3.csv", index=False)
    test.to_csv(output/"test_features_v3.csv", index=False)
    if previews:
        write_jsonl(output/"preview_trajectories_v3.jsonl", previews)
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    manifest = dict(simulator_version=SIMULATOR_VERSION, feature_version=FEATURE_VERSION,
                    feature_config=asdict(runner.feature_config), feature_config_id=runner.feature_config.fingerprint,
                    noise_config=asdict(runner.noise_config), seed=seed, paired=paired,
                    train_rows=len(train), test_rows=len(test), train_class_counts=train.label.value_counts().to_dict(),
                    test_class_counts=test.label.value_counts().to_dict(), families_per_scenario=families_per_scenario,
                    candidates=len(ledger), accepted_candidates=int((ledger.feature_status == "accepted").sum()),
                    rejected_candidates=int((ledger.feature_status != "accepted").sum()),
                    split_policy="80/20 scenario-stratified family split before exact balanced row selection",
                    selection_policy="Equal bird/drone counts; approximately even scenario/subtype/ideal-noisy strata",
                    family_leakage_count=0, raw_policy="Full raw omitted; one preview per scenario/subtype/profile",
                    calibration_status="uncalibrated_no_real_A", validation_scope="synthetic_internal_only",
                    parameter_manifest_sha256=runner.provenance["sha256"],
                    feature_columns=FEATURE_COLUMNS,
                    runtime_compatibility="NOT compatible with production v1/v2 model or RuleFilter thresholds",
                    source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files})
    write_json(output/"dataset_manifest.json", manifest)
    write_json(output/"parameter_manifest.json", runner.provenance)
    print(f"Wrote train={len(train)}, test={len(test)} to {output.resolve()}", flush=True)
    return train, test, ledger, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"output"/"dataset_10000_sim_v4")
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--test-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--families-per-scenario", type=int)
    parser.add_argument("--no-pairs", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    build_dataset(args.output, args.train_count, args.test_count, args.seed,
                  args.families_per_scenario, not args.no_pairs, 0 if args.no_preview else 1)


if __name__ == "__main__":
    main()
