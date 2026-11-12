"""배포 조건 점검: track 마다 앞 N초 구간의 창만 남겨 val 성능을 다시 잰다 (재학습 없음).

배포에서는 관리자가 지정한 10초 구간만 분석한다. 창은 2초 길이, 1초 간격이므로 앞 10초 = 창 0~8 (최대 9개).
저장된 val 실촬영 창 점수(pred/case*_<scenario>_validation_real.csv)만 사용한다. test 는 읽지 않는다.

    python -m research.c_part_deploy_window --results research/output/c_part_cv5_E1 --seconds 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ai_server.services import sequence_model as seq


def evaluate(results: Path, seconds: float | None):
    scen = "S" if any((results / "pred").glob("case*_S_validation_real.csv")) else "B"
    rows = []
    for k in range(5):
        p = pd.read_csv(results / "pred" / f"case{k}_{scen}_validation_real.csv")
        idx = p.sample_id.str.extract(r":w(\d+)$")[0].astype(int)
        if seconds is not None:
            p = p[idx <= seconds - 2]
        p = p.assign(track=seq.track_ids(p.sample_id.to_numpy()))
        tracks = p.groupby(["group_id", "track", "label"], as_index=False).decision.mean()
        m = seq.metrics(tracks)
        rows.append(dict(case=k, ba=m["balanced_accuracy"], bird=m["recall"]["bird"], drone=m["recall"]["drone"],
                         windows=len(p)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--seconds", type=float, nargs="+", default=[10, 5, 3])
    args = parser.parse_args()
    print("| 결과 | 구간 | 창 수 | track balanced accuracy | 새 recall | 드론 recall |\n| --- | --- | --- | --- | --- | --- |")
    for r in args.results:
        for sec in [None, *args.seconds]:
            d = evaluate(r, sec)
            label = "전체" if sec is None else f"앞 {sec:g}초"
            print(f"| {r.name} | {label} | {d.windows.sum()} | {d.ba.mean():.3f} ± {d.ba.std():.3f} | "
                  f"{d.bird.mean():.3f} | {d.drone.mean():.3f} |")


if __name__ == "__main__":
    main()
