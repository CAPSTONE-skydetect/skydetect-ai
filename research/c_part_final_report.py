"""5-case 실행 결과 하나를 이슈 기록용 표와 혼동행렬 이미지로 정리한다 (최종 성능 보고용).

    python -m research.c_part_final_report --results research/output/c_part_cv5_FINAL

대표값은 임계값 0, track 단위 balanced accuracy 의 5개 case 평균 ± 표준편차다.
--eval test_real (기본) 은 --final-test 로 만든 결과에만 있다. 개선 단계 결과는 --eval validation_real 로 본다.
이 스크립트는 저장된 결과 파일만 읽으며 모델을 다시 학습하거나 설정을 고르지 않는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .plot_confusion import plot

METRICS = [("balanced_accuracy", "balanced accuracy"), ("recall_bird", "새 recall"),
           ("recall_drone", "드론 recall"), ("roc_auc", "AUC")]
NAMES = {"validation_real": "val", "test_real": "test", "validation_augmented": "val 증강",
         "test_augmented": "test 증강"}


def sel(d: pd.DataFrame, ev: str, level: str) -> pd.DataFrame:
    return d[(d["eval"] == ev) & (d.threshold_mode == "t0") & (d.level == level)].set_index("case").sort_index()


def fmt(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std():.3f}"


def cm_sum(x: pd.DataFrame) -> np.ndarray:
    return np.sum([json.loads(c) for c in x.cm], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--eval", default="test_real", choices=["test_real", "validation_real"])
    parser.add_argument("--name", help="이미지 제목에 쓸 이름 (기본: 결과 폴더 이름)")
    parser.add_argument("--image", type=Path, help="혼동행렬 이미지 경로 (기본: 결과 폴더 안)")
    args = parser.parse_args()
    d = pd.read_csv(args.results / "case_results.csv", keep_default_na=False)
    if "scenario" in d and (d.scenario == "S").any():
        d = d[d.scenario == "S"]
    evals = set(d["eval"])
    if args.eval not in evals:
        raise SystemExit(f"{args.eval} 결과가 없다 (있는 평가: {sorted(evals)})")
    main_name = NAMES[args.eval]
    columns = [args.eval] if args.eval == "validation_real" else ["validation_real", args.eval]
    print(f"### 핵심 결과: {main_name} 실촬영 (임계값 0, 5개 case 평균 ± 표준편차)\n")
    print("| 단위 | 지표 | " + " | ".join(NAMES[c] for c in columns) + " |\n| --- | --- |" + " --- |" * len(columns))
    for level, label in (("track", "track"), ("group", "그룹")):
        for key, metric in METRICS if level == "track" else METRICS[:1]:
            cells = [fmt(sel(d, c, level)[key]) for c in columns]
            print(f"| {label} | {metric} | " + " | ".join(cells) + " |")
    track = sel(d, args.eval, "track")
    print(f"\n#### case별 track balanced accuracy\n")
    print("| 평가 | " + " | ".join(f"case {k}" for k in track.index) + " |\n| --- |" + " --- |" * len(track))
    for c in columns:
        print(f"| {NAMES[c]} | " + " | ".join(f"{v:.3f}" for v in sel(d, c, "track").balanced_accuracy) + " |")
    print(f"\n#### 혼동행렬 ({main_name}, 5개 case 합산, 임계값 0)\n")
    print("| 단위 | 새→새 | 새→드론 | 드론→새 | 드론→드론 |\n| --- | --- | --- | --- | --- |")
    for level, label in (("track", "track"), ("group", "그룹")):
        cm = cm_sum(sel(d, args.eval, level))
        print(f"| {label} | {cm[0][0]} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} |")
    stress = args.eval.replace("_real", "_augmented")
    if stress in evals:
        print(f"\n점검 ({NAMES[stress]} 세트, track balanced accuracy): {fmt(sel(d, stress, 'track').balanced_accuracy)}")
    image = args.image or args.results / f"confusion_S_{main_name}_track.png"
    plot(cm_sum(track), f"{args.name or args.results.name} (track, {main_name}, {len(track)} cases pooled)", image)
    print(f"\n혼동행렬 이미지: {image}")


if __name__ == "__main__":
    main()
