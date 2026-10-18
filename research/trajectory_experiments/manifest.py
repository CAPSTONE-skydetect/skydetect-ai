from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "trajectory-experiment-v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_legacy_manifest(
    legacy_root: str | Path,
    output_path: str | Path,
    *,
    experiment_id: str,
) -> dict[str, Any]:
    samples = []
    for sample_path in sorted(Path(legacy_root).rglob("sample.json")):
        legacy = json.loads(sample_path.read_text(encoding="utf-8"))
        prediction = legacy["prediction"]
        metadata_path = Path(prediction["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_video = Path(metadata["source_video_path"])
        gt_path = Path(legacy["ground_truth"]["path"])
        scale_x = float(metadata["processed_width"]) / float(metadata["width"])
        scale_y = float(metadata["processed_height"]) / float(metadata["height"])
        processed_bbox = [float(value) for value in metadata["init_bbox_processed"]]
        source_bbox = [
            processed_bbox[0] / scale_x,
            processed_bbox[1] / scale_y,
            processed_bbox[2] / scale_x,
            processed_bbox[3] / scale_y,
        ]
        frame_count = int(metadata["num_frames_processed"])
        fps = float(metadata["fps"])
        samples.append({
            "sample_id": legacy["sample_id"],
            "source_group_id": f"video:{sha256(source_video)}",
            "video": str(source_video),
            "video_sha256": sha256(source_video),
            "ground_truth": str(gt_path),
            "ground_truth_sha256": sha256(gt_path),
            "track_ids": list(legacy["ground_truth"].get("track_ids") or []),
            "label": legacy["ground_truth"].get("label"),
            "prediction_frame_offset": int(
                legacy.get("alignment", {}).get("prediction_frame_offset", 0)
            ),
            "source_is_trimmed": bool(
                legacy.get("alignment", {}).get("source_is_trimmed", False)
            ),
            "init_frame_index": int(metadata["init_frame_index"]),
            "target_bbox_source": source_bbox,
            "roi_source": "recovered_from_saved_processed_bbox_and_resize_transform",
            "max_seconds": max(1.0 / fps, (frame_count - 1) / fps),
            "stabilize": metadata.get("coordinate_mode") == "camera_motion_compensated",
            "resize_width": int(metadata["processed_width"]),
            "tracking_tuning": dict(metadata["tracking_parameters"]),
            "attempted_first_frame": int(metadata["tracking_start_frame"]),
            "attempted_last_frame": int(metadata["tracking_start_frame"] + frame_count - 1),
            "legacy_metadata": str(metadata_path),
            "legacy_metadata_sha256": sha256(metadata_path),
            "legacy_trajectory": str(prediction["trajectory"]),
            "legacy_provenance_note": legacy.get("provenance_note", ""),
        })
    if not samples:
        raise ValueError(f"no sample.json files under {legacy_root}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "legacy_root": str(Path(legacy_root)),
        "samples": samples,
        "variants": ["D0", "D0_repeat", "E1", "E2a", "E2b", "E2c", "E12", "E4", "E5"],
        "postprocess_variants": ["E3a", "E3b", "E6"],
        "limits": {"sample_timeout_seconds": 600, "stage_timeout_seconds": 7200},
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_and_validate(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported experiment manifest schema")
    seen = set()
    for sample in payload.get("samples", []):
        sample_id = sample["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        for key, hash_key in [
            ("video", "video_sha256"),
            ("ground_truth", "ground_truth_sha256"),
        ]:
            source = Path(sample[key])
            if not source.is_file():
                raise ValueError(f"missing {key}: {source}")
            if sha256(source) != sample[hash_key]:
                raise ValueError(f"hash mismatch for {key}: {source}")
        if "legacy_metadata" in sample:
            source = Path(sample["legacy_metadata"])
            if not source.is_file():
                raise ValueError(f"missing legacy_metadata: {source}")
            if sha256(source) != sample["legacy_metadata_sha256"]:
                raise ValueError(f"hash mismatch for legacy_metadata: {source}")
        if len(sample["target_bbox_source"]) != 4 or min(sample["target_bbox_source"][2:]) <= 0:
            raise ValueError(f"invalid target bbox for {sample_id}")
    if not seen:
        raise ValueError("experiment manifest has no samples")
    return payload
