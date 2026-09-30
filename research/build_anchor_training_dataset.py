"""Build a balanced development training set from approved real-anchor perturbations.

The requested row count is not an independent-case count. Validation and test
are never augmented or copied into this output.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash
from .evaluate_anchor_augmentation import augment_anchors, seeded_id
from .evaluate_sequence_handoff import load_arrays
from .io import write_json
from .trajectory_sequence import SequenceConfig


def quotas_for_groups(labels, groups, count):
    """Give both classes equal mass, then divide equally among source videos."""
    if count < 2 or count % 2:
        raise ValueError("Target must be an even count of at least two")
    mapping = pd.DataFrame({"label": labels, "group_id": groups})
    if (mapping.groupby("group_id").label.nunique() != 1).any():
        raise ValueError("Mixed-label source group")
    quotas = {}
    for label in ("bird", "drone"):
        group_names = sorted(mapping.loc[mapping.label == label, "group_id"].unique())
        if not group_names:
            raise ValueError("Both classes required")
        base, extra = divmod(count // 2, len(group_names))
        quotas.update({group: base + int(i < extra) for i, group in enumerate(group_names)})
    return quotas


def select_balanced(data, provenance, quotas, target):
    chosen = []
    used_sequences = set()
    for group, quota in sorted(quotas.items()):
        available = np.flatnonzero(data["group_id"] == group)
        rng = np.random.default_rng(seeded_id("bulk", group, target))
        group_chosen = []
        for index in rng.permutation(available):
            digest = hashlib.blake2b(data["X"][index].tobytes(), digest_size=16).digest()
            if digest in used_sequences:
                continue
            used_sequences.add(digest)
            group_chosen.append(int(index))
            if len(group_chosen) == quota:
                break
        if len(group_chosen) != quota:
            raise ValueError(f"Insufficient distinct candidates for {group}: {len(group_chosen)} < {quota}")
        chosen.extend(group_chosen)
    chosen = np.array(chosen, dtype=int)
    if len(chosen) != target or len(set(data["sample_id"][chosen])) != target:
        raise ValueError("Bulk sample count or sample IDs invalid")
    return {key: value[chosen] for key, value in data.items()}, provenance.iloc[chosen].reset_index(drop=True)


def build(folder, evaluation, output, target=6000):
    folder, evaluation, output = Path(folder), Path(evaluation), Path(output)
    if output.exists():
        raise ValueError("Output exists; use a new directory")
    gate = json.loads((evaluation / "results.json").read_text(encoding="utf-8"))
    if gate["status"] != "development_candidate_only" or not gate["large_generation_ready"]:
        raise ValueError("No selected development augmentation candidate")
    protocol = json.loads((evaluation / "protocol.json").read_text(encoding="utf-8"))
    real_dir = folder / "real"
    real_manifest = json.loads((real_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if protocol["source_manifest_sha256"] != file_hash(real_dir / "dataset_manifest.json"):
        raise ValueError("Real manifest changed after evaluation")
    if protocol["code_sha256"] != file_hash(Path(__file__).with_name("evaluate_anchor_augmentation.py")):
        raise ValueError("Augmentation code changed after evaluation")
    if real_manifest["contract_id"] != SequenceConfig(**real_manifest["config"]).fingerprint:
        raise ValueError("Sequence contract mismatch")
    train = load_arrays(real_dir / "train.npz")
    excluded_groups = {row["source_group_id"] for row in real_manifest["sources"]
                       if row["split"] in ("validation", "test")}
    if set(train["group_id"]) & excluded_groups:
        raise ValueError("Real train group overlaps validation or test")
    for name in ("train.npz", "validation.npz", "metadata.csv"):
        if file_hash(real_dir / name) != real_manifest["artifact_hashes"][name]:
            raise ValueError("Real source changed")
    source_metadata = pd.read_csv(real_dir / "metadata.csv", keep_default_na=False)
    source_metadata = source_metadata[source_metadata.split == "train"].sort_values("npz_row").reset_index(drop=True)
    dims = {r["parent_track_id"]: r["processed_height"] / r["processed_width"]
            for r in real_manifest["sources"] if r["split"] == "train"}
    donors = load_arrays(evaluation / "donor_pool.npz")
    quotas = quotas_for_groups(train["y"], train["group_id"], target)
    amplitude = float(gate["selected"]["amplitude"])
    rounds, copy_start, rejections = [], 0, 0
    distinct = {group: set() for group in quotas}
    for copies in (12, 24, 48, 96, 192, 384):
        part, part_metadata, rejected = augment_anchors(
            train, source_metadata, dims, donors, amplitude, copies=copies, copy_start=copy_start)
        rounds.append((part, part_metadata))
        rejections += rejected
        copy_start += copies
        for row, group in zip(part["X"], part["group_id"]):
            distinct[group].add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
        if all(len(distinct[group]) >= quota for group, quota in quotas.items()):
            break
    else:
        raise ValueError("Could not meet every real source-group quota")
    combined = {key: np.concatenate([data[key] for data, _ in rounds]) for key in train}
    provenance = pd.concat([frame for _, frame in rounds], ignore_index=True)
    selected, selected_metadata = select_balanced(combined, provenance, quotas, target)
    if set(selected["group_id"]) - set(train["group_id"]):
        raise ValueError("Unexpected source group")
    if set(selected["sample_id"]) & set(train["sample_id"]):
        raise ValueError("Augmentation collided with original IDs")
    if not np.allclose(selected["X"][:, 2:, 1:], np.diff(selected["X"][:, :2], axis=2), atol=1e-6):
        raise ValueError("Displacement channel mismatch")
    output.mkdir(parents=True)
    np.savez_compressed(output / "augmented_train.npz", **selected)
    selected_metadata.to_csv(output / "provenance.csv", index=False)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        status="development_training_only_not_independent_cases",
        generator_sha256=file_hash(__file__),
        contract_version=real_manifest["contract_version"], contract_id=real_manifest["contract_id"],
        channels=real_manifest["channels"], shape=[target, 4, 60],
        real_train_windows=len(train["X"]), real_train_source_groups=len(set(train["group_id"])),
        real_validation_used_for_gate=True, test_used_for_bulk_selection=False,
        source_manifest_sha256=file_hash(real_dir / "dataset_manifest.json"),
        evaluation_results_sha256=file_hash(evaluation / "results.json"),
        donor_pool_sha256=file_hash(evaluation / "donor_pool.npz"),
        selected_amplitude=amplitude, recommended_classifier_mass=gate["selected"]["mass"],
        synthetic_rows=target, class_counts={label: int(sum(selected["y"] == label)) for label in ("bird", "drone")},
        unique_sequence_count=len({hashlib.blake2b(row.tobytes(), digest_size=16).digest()
                                   for row in selected["X"]}),
        group_counts={group: int(sum(selected["group_id"] == group)) for group in sorted(quotas)},
        attempted_copies_per_anchor=copy_start, rejected_out_of_frame=rejections,
        provenance="Every row has its real anchor track and same-class simulated donor; original source group inherited",
        use="Add to real train with source-group/class-balanced weights at selected mass; validation/test stay real and untouched",
        limitations=["Derived rows are correlated and do not increase independent video count",
                     "Session independence is unverified", "No new unseen bird/drone real holdout exists"],
        hashes={name: file_hash(output / name) for name in ("augmented_train.npz", "provenance.csv")})
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="Frozen anchor augmentation evaluation directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for this development-only dataset")
    parser.add_argument("--target", type=int, default=6000)
    args = parser.parse_args()
    print(json.dumps(build(args.folder, args.evaluation, args.output, args.target), indent=2), flush=True)
