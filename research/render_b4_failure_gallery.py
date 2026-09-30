"""Render original-video frames with raw A centers for B-4 visual review."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def render(audit_csv, b4_summary, output):
    audit = pd.read_csv(audit_csv, keep_default_na=False)
    summary = json.loads(Path(b4_summary).read_text(encoding="utf-8"))
    group_ids = summary["common_real_error_groups"] + summary["synthetic_additional_error_groups"]
    if len(group_ids) != 4 or len(set(group_ids)) != 4:
        raise ValueError("Expected one shared and three synthetic-only additional failures")
    tiles = []
    for group_id in group_ids:
        matches = audit.loc[audit.source_group_id == group_id]
        if len(matches) != 1 or str(matches.iloc[0].raw_csv_match).lower() != "true":
            raise ValueError(f"Raw A CSV not uniquely matched for {group_id}")
        row = matches.iloc[0]
        video, raw = Path(row.video_path), Path(row.raw_csv_path)
        if not video.is_file() or not raw.is_file():
            raise ValueError(f"Missing video or raw CSV for {group_id}")
        points = pd.read_csv(raw)
        points = points.loc[points.visible.astype(str).str.lower() == "true"].sort_values("frame_index")
        if points.empty:
            raise ValueError(f"No visible A points for {group_id}")
        cap = cv2.VideoCapture(str(video))
        middle = points.iloc[len(points) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(middle.frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot read middle frame for {group_id}")
        height, width = frame.shape[:2]
        xy = np.rint(points[["raw_x", "raw_y"]].to_numpy(dtype=float) *
                      [width / float(row.processed_width), height / float(row.processed_height)]).astype(np.int32)
        center = np.rint([float(middle.raw_x) * width / float(row.processed_width),
                           float(middle.raw_y) * height / float(row.processed_height)]).astype(int)
        overlay = frame.copy()
        cv2.polylines(overlay, [xy.reshape(-1, 1, 2)], False, (255, 200, 0), 3, cv2.LINE_AA)
        cv2.circle(overlay, tuple(center), 12, (0, 0, 255), 3, cv2.LINE_AA)
        radius = 170
        left, top = int(center[0] - radius), int(center[1] - radius)
        padded = cv2.copyMakeBorder(frame, radius, radius, radius, radius,
                                    cv2.BORDER_CONSTANT, value=(40, 40, 40))
        crop = padded[top + radius:top + 3 * radius, left + radius:left + 3 * radius].copy()
        cv2.circle(crop, (radius, radius), 10, (0, 0, 255), 2, cv2.LINE_AA)
        full = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_AREA)
        crop = cv2.resize(crop, (540, 540), interpolation=cv2.INTER_LINEAR)
        tile = np.concatenate((full, crop), axis=1)
        label = f"{group_id}  frame={int(middle.frame_index)}  raw A center"
        cv2.putText(tile, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.concatenate(tiles, axis=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise ValueError("Cannot write visual review sheet")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("research/output/original_video_audit_v1/video_track_matches.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("research/output/real_reference_failure_analysis_b4_v1/summary.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/output/real_reference_failure_analysis_b4_v1/failure_gallery.png"))
    args = parser.parse_args()
    print(render(args.audit, args.summary, args.output))
