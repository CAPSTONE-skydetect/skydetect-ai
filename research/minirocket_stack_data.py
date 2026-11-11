"""Build real-only Stack inputs from immutable TrackSequence manifests."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ai_server.schemas import TrackSequence
from ai_server.services.sequence_model import LABELS, validate_windows
from research.trajectory_sequence import SequenceConfig, window_track

SCHEMA = "stack-tracksequence-collection-1"
SPLITS = ("train", "validation", "test")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_lineage(rows):
    assignments, samples, byte_ids = {}, set(), set()
    for row in rows:
        for key in ("source_sha256", "track_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError(f"Invalid {key}")
        for key in ("source_group", "session_group", "parent_id", "object_id", "segment_id"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Missing {key}")
        if row["source_group"] == "unknown":
            raise ValueError("Provide a conservative source group; session may be unknown")
        if row["label"] not in LABELS or row["split"] not in SPLITS:
            raise ValueError("Unknown label or split")
        sample = (row["source_sha256"], row["object_id"], row["segment_id"])
        if sample in samples or row["track_sha256"] in byte_ids:
            raise ValueError("Duplicate segment or TrackSequence bytes")
        samples.add(sample)
        byte_ids.add(row["track_sha256"])
        for k, value in (("source", row["source_sha256"]), ("group", row["source_group"]),
                         ("session", row["session_group"]), ("parent", row["parent_id"])):
            if k == "session" and value == "unknown":
                continue
            key = (k, value)
            target = (row["split"], row["source_group"])
            if assignments.setdefault(key, target) != target:
                raise ValueError(f"Lineage split/group leak: {key}")


def collection(path, *, include_test=False):
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest["schema"] != SCHEMA or not manifest["records"]:
        raise ValueError("Unexpected or empty collection manifest")
    rows = manifest["records"]
    validate_lineage(rows)
    subsets = {name: [] for name in SPLITS}
    audit, metadata = [], []
    for row in rows:
        item = {k: row[k] for k in ("source_sha256", "source_group", "session_group", "parent_id",
                                    "object_id", "segment_id", "label", "split", "qa_disposition",
                                    "track_path", "track_sha256")}
        audit.append(item)
        raw = (path.parent / row["track_path"]).read_bytes()
        if digest(raw) != row["track_sha256"]:
            raise ValueError(f"TrackSequence hash changed: {row['track_path']}")
        if row["split"] == "test" and not include_test:
            item["status"] = "reserved_test"
            continue
        if row["qa_disposition"] != "accepted":
            item["status"] = "excluded_qa"
            continue
        try:
            track = TrackSequence.model_validate_json(raw)
            windows, rejected = window_track(track.model_dump(mode="json"), SequenceConfig())
        except (ValueError, KeyError, TypeError) as exc:
            item.update(status="rejected_contract", detail=str(exc))
            continue
        scales = [w["normalization_scale"] for w in windows]
        supplied = row.get("normalization_scale")
        if supplied is not None and (not isinstance(supplied, list) or len(supplied) != len(scales)
                                     or not np.allclose(supplied, scales, rtol=0, atol=1e-12)):
            raise ValueError("Supplied normalization_scale disagrees with the sequence contract")
        item.update(status="accepted" if windows else "rejected_windows", rejected=rejected,
                    normalization_scale=scales)
        track_id = "track-" + digest(json.dumps(
            [row["source_sha256"], row["object_id"], row["segment_id"]], ensure_ascii=False).encode())
        for w in windows:
            sample_id = f'{track_id}:w{w["window_index"]:04d}'
            subsets[row["split"]].append((w["X"], row["label"], row["source_group"], sample_id))
            metadata.append(dict(sample_id=sample_id, label=row["label"], source_group_id=row["source_group"],
                                 normalization_scale=w["normalization_scale"]))
    data = {split: dict(X=validate_windows(np.stack([r[0] for r in values])),
                        y=np.array([r[1] for r in values]), group_id=np.array([r[2] for r in values]),
                        sample_id=np.array([r[3] for r in values]))
            for split, values in subsets.items() if values}
    return data, pd.DataFrame(metadata), audit, digest(payload)
