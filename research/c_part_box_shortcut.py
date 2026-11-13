"""상자 지름길 점검: 상자 값의 단순 통계만으로 새/드론을 맞힐 수 있는지 본다 (궤적·날갯짓 정보 없이).

--features motion (기본, E5): track 마다 프레임 간 상자 변화 통계 3개(변화 0 비율, |Δw| 중앙값, |Δw| 90% 값).
    이 값이 높으면 추적기가 만든 상자 처리 방식 차이가 라벨 단서가 된다는 뜻이다.
--features size (E18): track 의 상자 크기 중앙값 하나(화면 가로 길이 대비, 로그).
    이 값이 높으면 화면에 보이는 크기(촬영 거리) 차이가 라벨 단서가 된다는 뜻이다.
로지스틱 회귀를 촬영 그룹 단위 5-fold 로 평가한다.
val/test 구분 없이 원본 track 통계만 쓰며 학습 모델이나 test 점수는 보지 않는다.

    python -m research.c_part_box_shortcut --data research/output/c_part_cv5_v1 --features size
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

COLUMNS = dict(motion=["zero", "med", "p90"], size=["log_size"])


def stats(path):
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    w = np.array([p["w"] for p in t["history"]]) * t["processed_width"]
    h = np.array([p["h"] for p in t["history"]]) * t["processed_height"]
    dw, dh = np.abs(np.diff(w)), np.abs(np.diff(h))
    size = float(np.median(np.sqrt(np.maximum(w, 1.0) * np.maximum(h, 1.0))))
    return dict(zero=((dw < 1e-6) & (dh < 1e-6)).mean(), med=np.median(dw), p90=np.quantile(dw, .9),
                size_px=size, log_size=np.log(size / t["processed_width"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", choices=list(COLUMNS), default="motion")
    args = parser.parse_args()
    src = pd.read_csv(args.data / "sources.csv", encoding="utf-8-sig")
    d = pd.DataFrame([stats(p) for p in src.source_path])
    d["label"], d["group"] = src.label, src.source_group_id
    d["source"] = [source_family(v) for v in src.source_video_id]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    x = d[COLUMNS[args.features]].to_numpy()
    pred = cross_val_predict(model, x, d.label, groups=d.group, cv=cv)
    print(f"label from box {args.features} stats: balanced accuracy {balanced_accuracy_score(d.label, pred):.3f}")
    for lab, g in d.groupby("label"):
        part = pred[d.label == lab]
        print(f"  {lab}: recall {np.mean(part == lab):.3f}")
    d["pred"] = pred
    d["correct"] = d.pred == d.label
    print(d.groupby(["label", "source"]).agg(n=("correct", "size"), correct=("correct", "mean"),
                                              size_px_median=("size_px", "median")).round(3))


if __name__ == "__main__":
    main()
