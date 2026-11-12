"""E2: 저장된 창 점수(pred/case*_S_validation_real.csv)에서 창→track 집계 방식만 바꿔 비교한다. 재학습 없음.

후보는 미리 3개로 고정한다: mean(현재), median, vote(창별 드론 판정 비율 - 0.5).
track→그룹은 모든 방식에서 track 점수 평균. 임계값은 0 고정.

    python -m research.c_part_aggregation --results research/output/c_part_cv5_E1 --scenario S \
      --output research/output/c_part_cv5_E2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_server.services import sequence_model as seq

AGGREGATORS = {
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "vote": lambda s: (s >= 0).mean() - 0.5,
}


def aggregate(pred: pd.DataFrame, how: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred.assign(track=seq.track_ids(pred.sample_id.to_numpy()))
    tracks = pred.groupby(["group_id", "track", "label"]).decision.agg(AGGREGATORS[how]).reset_index()
    groups = tracks.groupby(["group_id", "label"], as_index=False).decision.mean()
    return tracks, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scenario", default="S")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for k in range(5):
        pred = pd.read_csv(args.results / "pred" / f"case{k}_{args.scenario}_validation_real.csv")
        for how in AGGREGATORS:
            tracks, groups = aggregate(pred, how)
            for level, table in (("track", tracks), ("group", groups)):
                m = seq.metrics(table)
                rows.append(dict(case=k, aggregator=how, level=level, balanced_accuracy=m["balanced_accuracy"],
                                 recall_bird=m["recall"]["bird"], recall_drone=m["recall"]["drone"],
                                 roc_auc=m["roc_auc"], cm=json.dumps(m["confusion_matrix"])))
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "case_results.csv", index=False)
    for (how, level), part in table.groupby(["aggregator", "level"]):
        cm = np.sum([json.loads(x) for x in part.cm], axis=0).tolist()
        print(f"{level:5s} {how:7s} BA {part.balanced_accuracy.mean():.3f}±{part.balanced_accuracy.std():.3f} "
              f"bird {part.recall_bird.mean():.3f} drone {part.recall_drone.mean():.3f} AUC {part.roc_auc.mean():.3f} CM {cm}")


if __name__ == "__main__":
    main()
