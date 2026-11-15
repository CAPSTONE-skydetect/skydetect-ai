"""Create paired raw/marked frame sheets for manual A-center verification."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def contact_sheet(match_csv, track_name, output, start_frame=None, count=30, crop=192):
    table = pd.read_csv(match_csv).fillna("")
    matches = table[(table.track_file == track_name) & (table.raw_csv_match == True)]
    if len(matches) != 1:
        raise ValueError("Need one video and exact A raw-coordinate CSV match")
    item = matches.iloc[0]
    track = json.loads(Path(item.raw_csv_path).with_name("track_sequence.json").read_text(encoding="utf-8-sig"))
    points = pd.read_csv(item.raw_csv_path).set_index("frame_index")
    frames = [p["frame_index"] for p in track["history"]]
    if start_frame is None:
        start_frame = frames[len(frames)//2] - count//2
    chosen = [f for f in frames if start_frame <= f < start_frame+count and f in points.index]
    if len(chosen) < min(count, 8):
        raise ValueError("Too few A observations for requested sheet")
    capture = cv2.VideoCapture(item.video_path)
    if not capture.isOpened():
        raise ValueError("Video cannot be opened")
    scale_x = float(item.video_width) / track["processed_width"]
    scale_y = float(item.video_height) / track["processed_height"]
    tiles = {kind: [] for kind in ("unmarked", "marked")}
    for frame_index in chosen:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Failed to decode frame {frame_index}")
        row = points.loc[frame_index]
        x, y = int(round(row.raw_x*scale_x)), int(round(row.raw_y*scale_y))
        if not 0 <= x < frame.shape[1] or not 0 <= y < frame.shape[0]:
            raise ValueError("A raw location outside video frame")
        pad = crop
        padded = cv2.copyMakeBorder(frame,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(20,20,20))
        raw = padded[y+pad-crop//2:y+pad+crop//2,x+pad-crop//2:x+pad+crop//2]
        raw = cv2.resize(raw,(288,288),interpolation=cv2.INTER_NEAREST)
        marked = raw.copy()
        cv2.circle(marked,(144,144),18,(0,0,255),2)
        cv2.line(marked,(144,115),(144,127),(0,0,255),2)
        cv2.line(marked,(144,161),(144,173),(0,0,255),2)
        for kind,tile in (("unmarked",raw),("marked",marked)):
            cv2.rectangle(tile,(0,0),(288,28),(20,20,20),-1)
            cv2.putText(tile,f"frame {frame_index}  A raw ({x},{y})",(5,19),
                        cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
            tiles[kind].append(tile)
    capture.release()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    paths = {}
    for kind,items in tiles.items():
        blank=np.zeros_like(items[0])
        sheet=cv2.vconcat([cv2.hconcat(items[i:i+5]+[blank]*(5-len(items[i:i+5])))
                           for i in range(0,len(items),5)])
        path=output/f"{Path(track_name).stem}_{kind}.png"
        encoded, buffer = cv2.imencode(".png", sheet)
        if not encoded:
            raise OSError(f"Could not write {path}")
        buffer.tofile(str(path))
        paths[kind]=str(path.resolve())
    return dict(track_name=track_name,video_name=item.video_name,
                start_frame=start_frame,frames=chosen,images=paths)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_name")
    parser.add_argument("--matches",type=Path,
        default=Path("research/output/original_video_audit_v1/video_track_matches.csv"))
    parser.add_argument("--output",type=Path,
        default=Path("research/output/original_video_audit_v1/contact_sheets"))
    parser.add_argument("--start-frame",type=int)
    parser.add_argument("--count",type=int,default=30)
    args=parser.parse_args()
    print(json.dumps(contact_sheet(args.matches,args.track_name,args.output,
                                   args.start_frame,args.count),ensure_ascii=True))
