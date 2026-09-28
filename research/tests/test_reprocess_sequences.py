from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from research.build_real_sequence_dataset import digest, file_hash
from research.io import write_json, write_jsonl
from research.reprocess_development_sequences import rebuild
from research.trajectory_sequence import SequenceConfig


def test_rebuild_keeps_groups_and_never_reads_missing_real_test(tmp_path):
    original, reference = tmp_path/"original", tmp_path/"reference"
    original.mkdir()
    reference.mkdir()
    sources, controls = [], []
    for i, (split, label) in enumerate((s, c) for s in ("train", "validation") for c in ("bird", "drone")):
        history = [dict(frame_index=f, timestamp_ms=round(f*1000/30), cx=.2+.001*f, cy=.4) for f in range(121)]
        track = dict(processed_width=1920, processed_height=1080, stabilization={"applied": True}, history=history)
        path = tmp_path/f"source{i}.json"
        write_json(path, track)
        sources.append(dict(split=split, label=label, status="accepted", source_path=str(path),
                            file_sha256=file_hash(path), history_hash=digest(history),
                            parent_track_id=f"track{i}", source_group_id=f"group{i}",
                            file_name=path.name, audit_sample_id=""))
        controls.append(dict(track=track, metadata=dict(label=label, subtype="prior", seed=i,
                             observation={"attempted_frame_count": 121})))
    sources.append(dict(split="test", status="accepted", source_path=str(tmp_path/"never-exists.json")))
    write_json(original/"dataset_manifest.json", dict(sources=sources, config=asdict(SequenceConfig()), max_train_windows_per_group=8))
    for name in ("before_fit", "before_evaluation"):
        write_jsonl(reference/f"{name}_tracks.jsonl", controls)
    write_json(reference/"results.json", {})
    output, rebuilt_reference = tmp_path/"rebuilt", tmp_path/"rebuilt_reference"
    manifest = rebuild(original, reference, output, rebuilt_reference)
    assert manifest["test_reprocessed"] is False and not (output/"test.npz").exists()
    meta = pd.read_csv(output/"metadata.csv")
    assert set(meta[meta.split == "train"].source_group_id) == {"group0", "group1"}
    assert set(meta[meta.split == "validation"].source_group_id) == {"group2", "group3"}
    assert not meta.anti_alias_applied.any()
    assert np.allclose(meta.source_fps, 30, rtol=1e-4)
