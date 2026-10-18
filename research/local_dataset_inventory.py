"""Inventory local videos and CVAT annotations without guessing class labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from defusedxml import ElementTree

from research.trajectory_experiments.manifest import sha256


def scan(root: Path, pilot_manifest: Path, output: Path) -> dict:
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    used_hashes = {row["video_sha256"] for row in pilot["samples"]}
    by_hash = defaultdict(list)
    videos = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".mp4", ".avi", ".mkv", ".mpg"} or path.name.lower().startswith("overlay"):
            continue
        digest = sha256(path)
        by_hash[digest].append(str(path))
        cap = cv2.VideoCapture(str(path))
        try:
            videos.append({"path": str(path), "sha256": digest,
                           "source_group_id": f"video:{digest}", "used_in_pilot": digest in used_hashes,
                           "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           "fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                           "class_label": None, "review_status": "needs_review"})
        finally:
            cap.release()
    annotations = []
    for path in sorted(root.rglob("*.xml")):
        xml = ElementTree.parse(path).getroot()
        tracks = []
        for track in xml.findall("track"):
            boxes = [box for box in track.findall("box") if box.get("outside", "0") != "1"]
            repeated_pairs = []
            for previous, current in zip(boxes, boxes[1:]):
                if (int(current.get("frame")) == int(previous.get("frame")) + 1
                        and all(previous.get(key) == current.get(key) for key in ("xtl", "ytl", "xbr", "ybr"))):
                    repeated_pairs.append(int(current.get("frame")))
            tracks.append({"id": int(track.get("id")), "label": track.get("label"),
                           "visible_box_count": len(boxes),
                           "keyframe_count": sum(b.get("keyframe") == "1" for b in boxes),
                           "exactly_repeated_box_frames": repeated_pairs,
                           "gt_review_note": "Repeated boxes can be legitimate; inspect movement before accepting derivative features."})
        annotations.append({"path": str(path), "sha256": sha256(path),
                            "declared_source": xml.findtext("meta/task/source"),
                            "task_size": xml.findtext("meta/task/size"), "tracks": tracks})
    possible_transcodes = []
    representatives = {row["sha256"]: row for row in videos}
    fingerprints = {}
    for digest, row in representatives.items():
        cap = cv2.VideoCapture(row["path"])
        thumbnails = []
        try:
            for fraction in (0.0, 0.25, 0.5, 0.75):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fraction * row["frames"]))
                ok, image = cap.read()
                if ok:
                    thumbnails.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float))
        finally:
            cap.release()
        fingerprints[digest] = thumbnails
    items = list(representatives.items())
    for index, (digest, row) in enumerate(items):
        for other_digest, other in items[index + 1:]:
            if (row["frames"], row["width"], row["height"]) != (other["frames"], other["width"], other["height"]):
                continue
            if abs(row["fps"] - other["fps"]) > 0.01:
                continue
            first, second = fingerprints[digest], fingerprints[other_digest]
            if len(first) != 4 or len(second) != 4:
                continue
            errors = [float(np.mean(np.abs(a - b))) for a, b in zip(first, second)]
            if max(errors) <= 2.0:
                possible_transcodes.append({"paths": [row["path"], other["path"]],
                                            "hashes": [digest, other_digest],
                                            "thumbnail_mae": errors,
                                            "status": "possible_same_source_requires_review"})
    result = {
        "videos": videos, "annotations": annotations,
        "summary": {"video_files": len(videos), "distinct_video_hashes": len(by_hash),
                    "pilot_video_hashes": len(used_hashes),
                    "candidate_nonpilot_hashes": len(set(by_hash) - used_hashes),
                    "annotation_files": len(annotations)},
        "duplicate_video_paths": [paths for paths in by_hash.values() if len(paths) > 1],
        "possible_transcodes": possible_transcodes,
        "limitations": ["Equal hashes establish byte identity; distinct hashes do not establish independent filming sessions.",
                        "Directory names are not class labels. Unannotated videos require reviewed CVAT GT.",
                        "CVAT filenames alone cannot establish frame alignment."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = []
    for digest, row in representatives.items():
        if digest in used_hashes:
            continue
        suspected_pilot = any(digest in pair["hashes"] and set(pair["hashes"]) & used_hashes
                              for pair in possible_transcodes)
        names = {Path(path).name for path in by_hash[digest]}
        jobs.append({
            "video_paths": by_hash[digest], "source_group_id": f"video:{digest}",
            "class_label": None, "review_status": "needs_review",
            "suspected_pilot_transcode": suspected_pilot,
            "possible_cvat_exports": [a["path"] for a in annotations if a["declared_source"] in names],
            "required_actions": ["Verify bird/drone identity from the video, not the folder name.",
                                 "Verify filming-session independence from pilot and C train sources.",
                                 "Label full-object bbox tracks, visibility and outside flags in CVAT.",
                                 "Export CVAT for video XML, preserve the source FPS and original video.",
                                 "Review object size after resizing to 1280px; target regime is 10-30px."],
        })
    output.with_name("annotation_jobs.json").write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(scan(args.root, args.pilot_manifest, args.output)["summary"], indent=2))


if __name__ == "__main__":
    main()
