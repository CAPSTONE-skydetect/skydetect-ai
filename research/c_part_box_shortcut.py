"""상자 지름길 점검: 상자 '움직임 통계'만으로 새/드론을 맞힐 수 있는지 본다 (날갯짓 같은 모양 정보 없이).

track 마다 프레임 간 상자 변화 통계 3개(변화 0 비율, |Δw| 중앙값, |Δw| 90% 값)만 쓰는 로지스틱 회귀를
촬영 그룹 단위 5-fold 로 평가한다. 이 값이 높으면 추적기가 만든 상자 처리 방식 차이가 라벨 단서가 된다는 뜻이다.
val/test 구분 없이 원본 track 통계만 쓰며 학습 모델이나 test 점수는 보지 않는다.

    python -m research.c_part_box_shortcut --data research/output/c_part_cv5_v1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .c_part_compare import source_family


def stats(path):
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    w = np.array([p["w"] for p in t["history"]]) * t["processed_width"]
    h = np.array([p["h"] for p in t["history"]]) * t["processed_height"]
    dw, dh = np.abs(np.diff(w)), np.abs(np.diff(h))
    return dict(zero=((dw < 1e-6) & (dh < 1e-6)).mean(), med=np.median(dw), p90=np.quantile(dw, .9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    src = pd.read_csv(args.data / "sources.csv", encoding="utf-8-sig")
    d = pd.DataFrame([stats(p) for p in src.source_path])
    d["label"], d["group"] = src.label, src.source_group_id
    d["source"] = [source_family(v) for v in src.source_video_id]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    x = d[["zero", "med", "p90"]].to_numpy()
    pred = cross_val_predict(model, x, d.label, groups=d.group, cv=cv)
    print(f"label from box-motion stats: balanced accuracy {balanced_accuracy_score(d.label, pred):.3f}")
    for lab, g in d.groupby("label"):
        part = pred[d.label == lab]
        print(f"  {lab}: recall {np.mean(part == lab):.3f}")
    d["pred"] = pred
    d["correct"] = d.pred == d.label
    print(d.groupby(["label", "source"]).correct.agg(["size", "mean"]).round(3))


if __name__ == "__main__":
    main()
