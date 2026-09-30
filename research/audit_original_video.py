"""Audit real video/TrackSequence alignment without claiming object localization.

Run with the same processed video files that were supplied to A. A filename
match is only a candidate; hashes, clocks and frame geometry are checked too.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash
from .diagnose_real_preprocessing import artifact_index, read_raw_match, track_identity
from .io import write_json


def video_family(name):
    stem = Path(name).stem.lower().replace(" ", "").replace("_", "")
    stem = re.sub(r"^0922", "", stem)
    stem = re.sub(r"tracksequence$", "", stem)
    stem = re.sub(r"\(보정\)|\(객체두개\)|\(\d+번객체\)", "", stem)
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"(?:위에|아래)새$", "", stem)
    return stem


def object_number(name):
    match = re.search(r"\((\d+)(?:번객체)?\)", Path(name).stem)
    return int(match.group(1)) if match else None


def video_record(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if min(width, height, frames) <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return dict(video_path=str(path.resolve()), video_name=path.name,
                family=video_family(path.name), object_number=object_number(path.name),
                width=width, height=height, fps=fps, frames=frames,
                duration_seconds=frames/fps, sha256=file_hash(path))


def candidates_for_track(source, videos):
    family = video_family(source["file_name"])
    matches = [v for v in videos if v["family"] == family and
               v["video_path"].lower().find(f"{source['label']}\\") >= 0]
    ordinal = object_number(source["file_name"])
    numbered = [v for v in matches if v["object_number"] == ordinal]
    if len(matches) > 1 and len(numbered) == 1:
        return numbered
    return matches


def audit(video_dir, manifest_path, artifact_dir, output):
    video_dir, manifest_path, artifact_dir, output = map(Path,
        (video_dir, manifest_path, artifact_dir, output))
    if output.exists():
        raise ValueError("Output exists; use a new audit directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = [video_record(p) for p in sorted(video_dir.glob("*/*")) if p.is_file()]
    artifacts, artifact_errors = artifact_index(artifact_dir)
    rows = []
    for source in manifest["sources"]:
        track = json.loads(Path(source["source_path"]).read_text(encoding="utf-8-sig"))
        history = track["history"]
        frame_max = max(p["frame_index"] for p in history)
        observed_fps = (history[-1]["frame_index"] - history[0]["frame_index"]) / (
            (history[-1]["timestamp_ms"] - history[0]["timestamp_ms"]) / 1000)
        matches = candidates_for_track(source, videos)
        raw, paths, errors = read_raw_match(track, artifacts.get(track_identity(track), []))
        for candidate in matches or [None]:
            row = dict(parent_track_id=source["parent_track_id"], label=source["label"],
                split=source["split"], source_group_id=source["source_group_id"],
                track_file=source["file_name"], source_video_id=source["source_video_id"],
                candidate_count=len(matches), track_points=len(history),
                first_frame=history[0]["frame_index"], last_frame=frame_max,
                last_timestamp_ms=history[-1]["timestamp_ms"],
                processed_width=track["processed_width"], processed_height=track["processed_height"],
                track_clock_fps=observed_fps, raw_csv_match=raw is not None,
                raw_csv_path=paths[0] + "\\trajectory.csv" if paths else "",
                raw_csv_errors=";".join(errors), video_name="" if candidate is None else candidate["video_name"],
                video_path="" if candidate is None else candidate["video_path"])
            if candidate is None:
                row.update(match_status="no_candidate", hash_match="unknown", frame_ok=False,
                           clock_ok=False, aspect_ok=False, duration_ok=False)
            else:
                frame_ok = frame_max < candidate["frames"]
                clock_ok = abs(observed_fps-candidate["fps"]) <= max(.5, candidate["fps"]*.02)
                aspect_ok = abs(track["processed_width"]/track["processed_height"] -
                                candidate["width"]/candidate["height"]) <= .02
                duration_ok = history[-1]["timestamp_ms"] / 1000 <= candidate["duration_seconds"] + .1
                expected = source.get("video_sha256", "")
                hash_match = "unknown" if not expected else str(expected == candidate["sha256"]).lower()
                row.update(video_width=candidate["width"],video_height=candidate["height"],
                    video_fps=candidate["fps"],video_frames=candidate["frames"],
                    video_duration_seconds=candidate["duration_seconds"],video_sha256=candidate["sha256"],
                    frame_ok=frame_ok, clock_ok=clock_ok, aspect_ok=aspect_ok,
                    duration_ok=duration_ok, hash_match=hash_match,
                    match_status="metadata_consistent" if len(matches) == 1 and all(
                        (frame_ok,clock_ok,aspect_ok,duration_ok)) and hash_match != "false"
                        else "needs_review")
            rows.append(row)
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True)
    frame.to_csv(output / "video_track_matches.csv", index=False, encoding="utf-8-sig")
    summary = dict(created_at=datetime.now(timezone.utc).isoformat(),
        video_count=len(videos), track_count=len(manifest["sources"]),
        candidate_rows=len(frame), match_status_counts=frame.match_status.value_counts().to_dict(),
        raw_csv_matched_tracks=int(frame[frame.raw_csv_match].parent_track_id.nunique()),
        artifact_index_errors=artifact_errors,
        interpretation="Metadata-consistent is not visually verified object identity or session independence",
        source_manifest_sha256=file_hash(manifest_path))
    write_json(output / "audit_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("research/data/original_video"))
    parser.add_argument("--manifest", type=Path,
        default=Path("research/output/sequence_handoff_20260928/real/dataset_manifest.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/manual_tracks"))
    parser.add_argument("--output", type=Path, default=Path("research/output/original_video_audit_v1"))
    args = parser.parse_args()
    print(json.dumps(audit(args.videos,args.manifest,args.artifacts,args.output),
                     ensure_ascii=True,indent=2))
