"""Package three development train arms against one real A validation split.

This is a dataset handoff, not a model evaluation or synthetic-data approval.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash
from .evaluate_sequence_handoff import balanced_group_weights, load_arrays
from .io import write_json
from .trajectory_sequence import CHANNELS, CONTRACT_VERSION, SequenceConfig


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aligned_metadata(path, arrays, *, split=None, group_column="group_id", label_column="label"):
    frame = pd.read_csv(path, keep_default_na=False)
    if split is not None:
        frame = frame.loc[frame.split == split].sort_values("npz_row").reset_index(drop=True)
        if frame.npz_row.tolist() != list(range(len(arrays["X"]))):
            raise ValueError(f"NPZ row mismatch: {path}")
    if len(frame) != len(arrays["X"]):
        raise ValueError(f"Metadata row mismatch: {path}")
    for column, key in (("sample_id", "sample_id"), (group_column, "group_id"), (label_column, "y")):
        if column not in frame or not np.array_equal(frame[column].to_numpy(dtype=str), arrays[key]):
            raise ValueError(f"Metadata {column} mismatch: {path}")
    return frame


def arm_arrays(data, weights, domains):
    if len(weights) != len(data["X"]) or len(domains) != len(weights):
        raise ValueError("Training weights/domains do not align")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Training weights must be positive and finite")
    return dict(data, sample_weight=np.asarray(weights, dtype=np.float64),
                domain=np.asarray(domains, dtype="U24"))


def summary(data):
    return dict(windows=len(data["X"]), groups=len(set(data["group_id"])),
                labels=dict(Counter(data["y"].tolist())),
                domains=dict(Counter(data.get("domain", np.full(len(data["X"]), "real")).tolist())))


def package(handoff, augmentation, evaluation, output):
    handoff, augmentation, evaluation, output = map(Path, (handoff, augmentation, evaluation, output))
    if output.exists():
        raise ValueError("Output exists; select a new directory")
    real_dir = handoff / "real"
    real_manifest = read_json(real_dir / "dataset_manifest.json")
    aug_manifest = read_json(augmentation / "dataset_manifest.json")
    gate = read_json(evaluation / "results.json")
    audit = read_json(handoff / "group_audit.json")
    preparation = read_json(handoff / "preparation_results.json")
    config = SequenceConfig(**real_manifest["config"])
    if (real_manifest["contract_version"] != CONTRACT_VERSION or
            real_manifest["contract_id"] != config.fingerprint or
            real_manifest["channels"] != list(CHANNELS)):
        raise ValueError("Real preprocessing contract mismatch")
    if (aug_manifest["contract_version"] != CONTRACT_VERSION or
            aug_manifest["contract_id"] != config.fingerprint or
            aug_manifest["source_manifest_sha256"] != file_hash(real_dir / "dataset_manifest.json") or
            aug_manifest["evaluation_results_sha256"] != file_hash(evaluation / "results.json") or
            aug_manifest["donor_pool_sha256"] != file_hash(evaluation / "donor_pool.npz")):
        raise ValueError("Augmentation source/evaluation mismatch")
    if (gate["status"] != "development_candidate_only" or not gate["large_generation_ready"] or
            aug_manifest["status"] != "development_training_only_not_independent_cases"):
        raise ValueError("Only the recorded development augmentation candidate is supported")
    if audit["cross_split_groups"] or audit["changed_groups"] or not audit["source_video_groups_verified"]:
        raise ValueError("Source-video grouping audit failed")
    if preparation["status"] != "experimental_downstream_ablation_only":
        raise ValueError("Unexpected synthetic cohort status")
    for name in ("train.npz", "validation.npz", "metadata.csv"):
        if file_hash(real_dir / name) != real_manifest["artifact_hashes"][name]:
            raise ValueError(f"Real source changed: {name}")
    for name, expected in aug_manifest["hashes"].items():
        if file_hash(augmentation / name) != expected:
            raise ValueError(f"Augmented source changed: {name}")

    real = load_arrays(real_dir / "train.npz")
    validation = load_arrays(real_dir / "validation.npz")
    synthetic = load_arrays(handoff / "synthetic_train.npz")
    augmented = load_arrays(augmentation / "augmented_train.npz")
    real_meta = pd.read_csv(real_dir / "metadata.csv", keep_default_na=False)
    train_meta = aligned_metadata(real_dir / "metadata.csv", real, split="train", group_column="source_group_id")
    val_meta = aligned_metadata(real_dir / "metadata.csv", validation, split="validation", group_column="source_group_id")
    sim_meta = aligned_metadata(handoff / "synthetic_train_metrics.csv", synthetic)
    aug_meta = aligned_metadata(augmentation / "provenance.csv", augmented, label_column="y")
    sources = real_manifest["sources"]
    source_groups = {split: {row["source_group_id"] for row in sources if row["split"] == split}
                     for split in ("train", "validation", "test")}
    if any(source_groups[a] & source_groups[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Source video crosses real splits")
    if (set(real["group_id"]) != source_groups["train"] or
            set(validation["group_id"]) != source_groups["validation"] or
            set(augmented["group_id"]) - source_groups["train"] or
            set(synthetic["group_id"]) & set.union(*source_groups.values())):
        raise ValueError("Training/validation group lineage mismatch")
    if not set(aug_meta.parent_track_id).issubset(set(train_meta.parent_track_id)):
        raise ValueError("Augmentation parent is not a real train track")
    if (train_meta.groupby("parent_track_id")[["source_group_id", "label"]].nunique() > 1).any().any():
        raise ValueError("One real track has conflicting lineage")
    parent_lineage = (train_meta.drop_duplicates("parent_track_id").set_index("parent_track_id")
                      [["source_group_id", "label"]].to_dict("index"))
    if any(parent_lineage[row.parent_track_id]["source_group_id"] != row.group_id or
           parent_lineage[row.parent_track_id]["label"] != row.y for row in aug_meta.itertuples()):
        raise ValueError("Augmentation label/group differs from its real parent")
    if not all(str(s).startswith("sim:") for s in aug_meta.donor_id):
        raise ValueError("Augmentation donor ID is not synthetic")
    ids = [set(data["sample_id"]) for data in (real, validation, synthetic, augmented)]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids))):
        raise ValueError("Sample IDs overlap across source cohorts")
    if set(real_meta.split) - {"train", "validation"} or len(real_meta) != len(real["X"]) + len(validation["X"]):
        raise ValueError("Real metadata contains an unexpected split or count")
    if not np.isclose(float(aug_manifest["recommended_classifier_mass"]), .1):
        raise ValueError("Unexpected augmentation weight recommendation")

    n = len(real["X"])
    real_weight = balanced_group_weights(real["y"], real["group_id"], n)
    aug_weight = balanced_group_weights(augmented["y"], augmented["group_id"], n * .1)
    synthetic_weight = balanced_group_weights(synthetic["y"], synthetic["group_id"], n)
    arms = {
        "train_real_only.npz": arm_arrays(real, real_weight, np.full(n, "real")),
        "train_real_plus_augmented.npz": arm_arrays(
            {key: np.concatenate((real[key], augmented[key])) for key in real},
            np.r_[real_weight * .9, aug_weight],
            np.r_[np.full(n, "real"), np.full(len(augmented["X"]), "real_anchor_augmented")]),
        "train_synthetic_only.npz": arm_arrays(
            synthetic, synthetic_weight, np.full(len(synthetic["X"]), "synthetic")),
    }
    if not np.isclose(arms["train_real_plus_augmented.npz"]["sample_weight"].sum(), n):
        raise ValueError("Mixed arm total weight changed")
    if set(arms["train_real_plus_augmented.npz"]["group_id"]) & set(validation["group_id"]):
        raise ValueError("Validation leakage")

    output.mkdir(parents=True)
    for name, arrays in arms.items():
        np.savez_compressed(output / name, **arrays)
    np.savez_compressed(output / "validation_real.npz", **validation)
    real_meta.drop(columns=["file_name", "audit_sample_id"], errors="ignore").to_csv(
        output / "real_metadata.csv", index=False)
    sim_meta.to_csv(output / "synthetic_metadata.csv", index=False)
    aug_meta.to_csv(output / "augmentation_provenance.csv", index=False)
    inventory_fields = ("parent_track_id", "source_group_id", "label", "split",
                        "file_sha256", "history_hash", "video_sha256")
    pd.DataFrame([{key: row.get(key) for key in inventory_fields} for row in sources]).to_csv(
        output / "source_inventory.csv", index=False)
    shutil.copy2(handoff / "selected_profile.json", output / "simulator_profile.json")
    source_hashes = {
        "real_manifest": file_hash(real_dir / "dataset_manifest.json"),
        "real_train": file_hash(real_dir / "train.npz"),
        "real_validation": file_hash(real_dir / "validation.npz"),
        "real_metadata": file_hash(real_dir / "metadata.csv"),
        "synthetic_train": file_hash(handoff / "synthetic_train.npz"),
        "synthetic_metadata": file_hash(handoff / "synthetic_train_metrics.csv"),
        "simulator_profile": file_hash(handoff / "selected_profile.json"),
        "group_audit": file_hash(handoff / "group_audit.json"),
        "augmentation_manifest": file_hash(augmentation / "dataset_manifest.json"),
        "augmentation_train": file_hash(augmentation / "augmented_train.npz"),
        "augmentation_provenance": file_hash(augmentation / "provenance.csv"),
        "augmentation_gate": file_hash(evaluation / "results.json"),
    }
    counts = {name: summary(data) for name, data in arms.items()}
    counts["validation_real.npz"] = summary(validation)
    manifest = dict(
        schema="real-reference-comparison-1", created_at=datetime.now(timezone.utc).isoformat(),
        status="development_comparison_only", contract_version=CONTRACT_VERSION,
        contract_id=config.fingerprint, config=asdict(config), channels=list(CHANNELS),
        dtype="float32", shape=["N", 4, config.samples], labels=["bird", "drone"],
        arms={"real_only": "train_real_only.npz", "real_plus_augmentation": "train_real_plus_augmented.npz",
              "synthetic_only": "train_synthetic_only.npz"},
        common_validation="validation_real.npz", test_included=False,
        real_split_source_groups={key: len(value) for key, value in source_groups.items()},
        real_split_track_counts={key: sum(row["split"] == key for row in sources) for key in source_groups},
        source_group_overlap=0, recording_sessions_verified=False,
        train_without_video_hash=audit["train_without_video_hash"],
        validation_previously_reused=True, prior_test_previously_inspected=True,
        independent_real_world_evaluation_ready=False,
        simulator_cohort="experimental; prior real-plus-whole-simulator mixing worsened validation",
        augmentation_cohort="development-only; prior real validation unchanged",
        augmentation_sample_weight_mass=.1,
        group_policy="All windows and augmentations inherit the parent source-video group; synthetic flight seed is its group",
        fit_policy="Fit transforms/classifier on each arm's training data only; no validation/test fit",
        counts=counts, source_hashes=source_hashes,
        file_hashes={p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
    )
    write_json(output / "dataset_manifest.json", manifest)
    (output / "README.md").write_text(
        "# B to C: real-reference comparison (development only)\n\n"
        "Use one preprocessing/model implementation with three separate training runs. "
        "The three train files are alternative arms, not three splits. All use validation_real.npz.\n\n"
        "- train_real_only.npz: real A train.\n"
        "- train_real_plus_augmented.npz: the same real train plus real-anchor residual variants; "
        "sample_weight gives the variants 10% of total training mass.\n"
        "- train_synthetic_only.npz: simulator flight windows only, for a transfer control.\n\n"
        "Each train NPZ contains X, y, group_id, sample_id, sample_weight, domain. "
        "Validation contains X, y, group_id, sample_id. X is float32 (N,4,60); "
        "do not use y, domain, provenance or quality metadata as model channels. "
        "Fit MiniRocket and scaler using training data allowed by each arm; do not reuse a real-fitted "
        "transformer for the synthetic-only arm. For the real-plus-augmentation arm, the earlier "
        "controlled comparison fit the transformer on real train and added augmented windows only "
        "at Ridge fit. Record the chosen rule. Group CV by source video or synthetic flight, never row.\n\n"
        "The common validation is historically reused development data; the earlier real test was "
        "inspected and is intentionally absent. Sessions across different videos are unverified. "
        "This package does not approve synthetic training or establish real-world generalization. "
        "Freeze a choice before collecting and evaluating new untouched real A tracks.\n\n"
        "Read dataset_manifest.json for contract, counts, checksums and limitations. "
        "real_metadata.csv, synthetic_metadata.csv, augmentation_provenance.csv and "
        "source_inventory.csv provide aligned quality and source evidence without local file paths.\n",
        encoding="utf-8")
    manifest["file_hashes"]["README.md"] = file_hash(output / "README.md")
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--augmentation", type=Path, default=Path("research/output/anchor_training_6000_v3"))
    parser.add_argument("--evaluation", type=Path, default=Path("research/output/anchor_augmentation_v3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.handoff, args.augmentation, args.evaluation, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "counts": result["counts"]}, indent=2))
