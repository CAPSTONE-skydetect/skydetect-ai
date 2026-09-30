"""Exploratory silhouette-center proxy near an A raw-center seed.

This is not ground truth. Every accepted segment requires visual review,
especially when the animal is only a few pixels across.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def silhouette_proxy(frame, x, y, radius=32):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    x,y=int(round(x)),int(round(y))
    x0,y0=max(0,x-radius),max(0,y-radius)
    x1,y1=min(gray.shape[1],x+radius+1),min(gray.shape[0],y+radius+1)
    patch=gray[y0:y1,x0:x1]
    if patch.shape != (2*radius+1,2*radius+1):
        return None
    rim=np.r_[patch[0],patch[-1],patch[:,0],patch[:,-1]].astype(np.float32)
    background=float(np.median(rim))
    mad=float(np.median(np.abs(rim-background)))
    threshold=background-max(18.,3.*1.4826*mad)
    binary=np.uint8(patch<threshold)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,components,stats,centers=cv2.connectedComponentsWithStats(binary,8)
    candidates=[]
    for k in range(1,n):
        left,top,width,height,area=map(int,stats[k])
        px,py=centers[k]
        distance=float(np.hypot(px-radius,py-radius))
        if 3 <= area <= 400 and distance <= 10 and min(left,top) > 0 and \
                left+width < binary.shape[1] and top+height < binary.shape[0]:
            candidates.append((distance,-area,k))
    if not candidates:
        return None
    _,_,selected=min(candidates)
    region=components==selected
    cx,cy=centers[selected]
    contrast=background-float(np.median(patch[region]))
    if contrast < 18:
        return None
    left,top,width,height,area=map(int,stats[selected])
    return dict(x=float(x0+cx),y=float(y0+cy),area_px=area,
                width_px=width,height_px=height,contrast=contrast,
                background=background,threshold=threshold,
                mask=region.astype(np.uint8),patch_origin=(x0,y0))


def measure(match_csv, track_name, output, start_frame, count=30):
    matches=pd.read_csv(match_csv).fillna("")
    matches=matches[(matches.track_file==track_name)&(matches.raw_csv_match==True)]
    if len(matches)!=1:
        raise ValueError("Need one video and A raw CSV match")
    item=matches.iloc[0]
    trajectory=pd.read_csv(item.raw_csv_path).set_index("frame_index")
    scale_x=item.video_width/item.processed_width
    scale_y=item.video_height/item.processed_height
    capture=cv2.VideoCapture(item.video_path)
    if not capture.isOpened():
        raise ValueError("Could not open video")
    rows=[]
    tiles=[]
    for frame_index in range(start_frame,start_frame+count):
        if frame_index not in trajectory.index:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
        ok,frame=capture.read()
        if not ok:
            continue
        raw=trajectory.loc[frame_index]
        x,y=float(raw.raw_x*scale_x),float(raw.raw_y*scale_y)
        result=silhouette_proxy(frame,x,y)
        row=dict(track_name=track_name,frame_index=frame_index,raw_x=x,raw_y=y,
                 proxy_status="candidate_requires_review" if result else "unresolved")
        if result:
            row.update(proxy_x=result["x"],proxy_y=result["y"],
                offset_px=float(np.hypot(result["x"]-x,result["y"]-y)),
                area_px=result["area_px"],width_px=result["width_px"],
                height_px=result["height_px"],contrast=result["contrast"])
        rows.append(row)
        crop=64
        padded=cv2.copyMakeBorder(frame,crop,crop,crop,crop,cv2.BORDER_CONSTANT,value=(20,20,20))
        ix,iy=round(x)+crop,round(y)+crop
        tile=padded[iy-crop//2:iy+crop//2,ix-crop//2:ix+crop//2]
        tile=cv2.resize(tile,(256,256),interpolation=cv2.INTER_NEAREST)
        cv2.circle(tile,(128,128),8,(0,0,255),1)
        if result:
            px=int(round(128+(result["x"]-x)*4))
            py=int(round(128+(result["y"]-y)*4))
            cv2.drawMarker(tile,(px,py),(0,255,0),cv2.MARKER_CROSS,12,1)
        cv2.rectangle(tile,(0,0),(256,24),(20,20,20),-1)
        cv2.putText(tile,f"f{frame_index} d={row.get('offset_px',float('nan')):.1f}px",
                    (4,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
    capture.release()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(rows)
    csv=output/f"{Path(track_name).stem}_proxy.csv"
    table.to_csv(csv,index=False)
    blank=np.zeros_like(tiles[0])
    sheet=cv2.vconcat([cv2.hconcat(tiles[i:i+5]+[blank]*(5-len(tiles[i:i+5])))
                       for i in range(0,len(tiles),5)])
    image=output/f"{Path(track_name).stem}_proxy.png"
    encoded,buffer=cv2.imencode(".png",sheet)
    if not encoded:
        raise OSError("Could not encode review sheet")
    buffer.tofile(str(image))
    return dict(track_name=track_name,observed=len(table),candidates=int(table.proxy_x.notna().sum()),
                median_offset_px=float(table.offset_px.median()),csv=str(csv.resolve()),image=str(image.resolve()))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_name")
    parser.add_argument("--start-frame",type=int,required=True)
    parser.add_argument("--count",type=int,default=30)
    parser.add_argument("--matches",type=Path,
        default=Path("research/output/original_video_audit_v1/video_track_matches.csv"))
    parser.add_argument("--output",type=Path,
        default=Path("research/output/original_video_audit_v1/visual_centers"))
    args=parser.parse_args()
    print(json.dumps(measure(args.matches,args.track_name,args.output,
                             args.start_frame,args.count),ensure_ascii=True))
