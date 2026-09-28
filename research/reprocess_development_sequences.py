"""Reprocess fixed development groups and saved baseline; never open real test."""
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import digest, file_hash
from .calibrate_sequence_simulator import save_cohort, simulated_windows
from .io import read_jsonl, write_json
from .trajectory_sequence import CONTRACT_VERSION, SequenceConfig, window_track
from .verify_real_track import load_tracks


def rebuild(dataset, reference, output, reference_output):
    dataset, reference, output, reference_output = map(Path, (dataset, reference, output, reference_output))
    for folder in (output, reference_output):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Use new empty output folders")
    original = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**original["config"])
    sources, by_group, rejections = deepcopy(original["sources"]), defaultdict(list), []
    for rec in sources:
        if rec["split"] not in ("train", "validation") or rec["status"] != "accepted":
            continue
        path = Path(rec["source_path"])
        if file_hash(path) != rec["file_sha256"]:
            raise ValueError("Development input changed")
        matches = [t for t in load_tracks(path) if digest(t["history"]) == rec["history_hash"]]
        if len(matches) != 1:
            raise ValueError("Cannot identify development track")
        windows, rejected = window_track(matches[0], cfg)
        identity = {key: rec[key] for key in ("parent_track_id", "source_group_id", "split", "label", "file_name", "audit_sample_id")}
        rec["accepted_windows"] = len(windows)
        rec["status"] = "accepted" if windows else "excluded"
        for w in windows:
            by_group[rec["source_group_id"]].append(dict(w, **identity, domain="real",
                  sample_id=f"{rec['parent_track_id']}:w{w['window_index']:04d}"))
        rejections.extend(dict(identity, **r) for r in rejected)
    selected = []
    cap = original["max_train_windows_per_group"]
    for _, rows in sorted(by_group.items()):
        rows.sort(key=lambda r: (r["parent_track_id"], r["window_index"]))
        if rows[0]["split"] == "train" and len(rows) > cap:
            indices = np.linspace(0, len(rows)-1, cap, dtype=int)
            selected.extend(rows[i] for i in indices)
        else:
            selected.extend(rows)
    output.mkdir(parents=True, exist_ok=True)
    meta, summary = [], []
    for split in ("train", "validation"):
        rows = sorted((r for r in selected if r["split"] == split), key=lambda r: r["sample_id"])
        x = np.stack([r["X"] for r in rows])
        np.savez_compressed(output/f"{split}.npz", X=x,
                            y=np.asarray([r["label"] for r in rows], dtype="U5"),
                            group_id=np.asarray([r["source_group_id"] for r in rows], dtype="U32"),
                            sample_id=np.asarray([r["sample_id"] for r in rows], dtype="U40"))
        meta.extend(dict(npz_row=i, **{k: v for k, v in r.items() if k != "X"}) for i, r in enumerate(rows))
        for label in ("bird", "drone"):
            subset = [r for r in rows if r["label"] == label]
            summary.append(dict(split=split, label=label, windows=len(subset),
                                groups=len({r["source_group_id"] for r in subset})))
    metadata = pd.DataFrame(meta)
    metadata.to_csv(output/"metadata.csv", index=False)
    if set(metadata[metadata.split == "train"].source_group_id) & set(metadata[metadata.split == "validation"].source_group_id):
        raise ValueError("Reprocessing changed group split")
    rebuilt = dict(original, sources=sources, contract_version=CONTRACT_VERSION, contract_id=cfg.fingerprint,
                   created_at=datetime.now(timezone.utc).isoformat(),
                   config=asdict(cfg), summary=summary, development_only=True,
                   test_reprocessed=False, test_npz_present=False,
                   preprocessing_policy="Label-blind clock correction; validation arrays prepared before fitting but not used for selection",
                   previous_manifest_sha256=file_hash(dataset/"dataset_manifest.json"))
    rebuilt["code_hashes"] = {name: file_hash(Path(__file__).with_name(name)) for name in
                              ("reprocess_development_sequences.py", "trajectory_sequence.py")}
    rebuilt["artifact_hashes"] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
    write_json(output/"dataset_manifest.json", rebuilt)
    write_json(output/"rejections.json", rejections)
    reference_output.mkdir(parents=True, exist_ok=True)
    for name in ("before_fit", "before_evaluation"):
        records = list(read_jsonl(reference/f"{name}_tracks.jsonl"))
        save_cohort(reference_output, name, records, simulated_windows(records, cfg))
    past = json.loads((reference/"results.json").read_text(encoding="utf-8"))
    past["old_contract_results_not_directly_comparable"] = True
    past["baseline_reprocessed_from"] = str(reference)
    write_json(reference_output/"results.json", past)
    write_json(reference_output/"reprocess_protocol.json", dict(contract_id=cfg.fingerprint,
               contract_version=CONTRACT_VERSION, test_sources_opened=False,
               source_reference=str(reference), source_real_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
               code_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                            ("reprocess_development_sequences.py", "trajectory_sequence.py")},
               reference_hashes={name: file_hash(reference/name) for name in
                                 ("before_fit_tracks.jsonl", "before_evaluation_tracks.jsonl")}))
    print(json.dumps(dict(contract_id=cfg.fingerprint, summary=summary, test_opened=False), indent=2))
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/real_sequences_v1"))
    parser.add_argument("--reference", type=Path, default=Path("research/output/sim_to_real_v2"))
    parser.add_argument("--output", type=Path, default=Path("research/output/real_sequences_clock_v1_1"))
    parser.add_argument("--reference-output", type=Path, default=Path("research/output/sim_clock_reference"))
    args = parser.parse_args()
    rebuild(args.dataset, args.reference, args.output, args.reference_output)


if __name__ == "__main__":
    main()
