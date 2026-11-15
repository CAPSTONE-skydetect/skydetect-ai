"""최종 test 의 ROC 곡선 이미지를 만든다 (MiniRocket Stack vs MultiRocket Stack, track 단위).

저장된 예측(c_part_cv5_stack --final-test 의 pred/caseK_S_test_real.csv)만 읽는다. 모델을 다시 학습하지 않는다.
창 점수를 track 평균으로 모은 뒤 5개 case 를 합쳐 곡선을 그린다. 범례의 AUC 는 이슈 표와 같은
"case 별 AUC 의 평균 ± 표준편차"이고, 합친 곡선의 AUC 는 따로 적는다. 임계값 0 지점을 표시한다.

    python -m research.c_part_roc_figure --out docs/images/c_part/FINAL_test_roc_track.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ai_server.services import sequence_model as seq

MODELS = [  # 범주 색은 고정 순서 (dataviz 기본 팔레트 1·2번, 색각 이상 검증 통과)
    ("MiniRocket Stack (E1)", "research/output/c_part_cv5_FINAL_minirocket", "#2a78d6"),
    ("MultiRocket Stack (E15, 서비스)", "research/output/c_part_cv5_FINAL_multirocket", "#eb6834"),
]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def track_scores(folder: Path, case: int) -> pd.DataFrame:
    d = pd.read_csv(folder / "pred" / f"case{case}_S_test_real.csv")
    d["track"] = seq.track_ids(d.sample_id)
    t = d.groupby(["track", "group_id", "label"], as_index=False).decision.mean()
    t["case"] = case
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (4, 4)), alpha=0.6, zorder=1)
    ax.text(0.62, 0.55, "무작위 판정", color=INK_2, fontsize=8, rotation=43)

    for name, folder, color in MODELS:
        cases = [track_scores(Path(folder), k) for k in range(5)]
        per_case = [roc_auc_score(c.label == "drone", c.decision) for c in cases]
        pooled = pd.concat(cases)
        y = (pooled.label == "drone").to_numpy()
        fpr, tpr, thr = roc_curve(y, pooled.decision)
        auc_pooled = roc_auc_score(y, pooled.decision)
        label = f"{name}\nAUC {np.mean(per_case):.3f} ± {np.std(per_case, ddof=1):.3f} (case 평균)"
        ax.plot(fpr, tpr, color=color, lw=2, solid_capstyle="round", label=label, zorder=3)
        # 임계값 0 (서비스 판정 기준) 지점: 점수 >= 0 이면 드론
        fp0 = float(np.mean(pooled.decision[~y] >= 0))
        tp0 = float(np.mean(pooled.decision[y] >= 0))
        ax.scatter([fp0], [tp0], s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        print(f"{name}: per-case AUC {np.round(per_case, 3)} mean {np.mean(per_case):.3f}, pooled {auc_pooled:.3f}, "
              f"threshold0 FPR {fp0:.3f} TPR {tp0:.3f}, tracks {len(pooled)}")

    ax.annotate("● 임계값 0 (서비스 판정 지점)", xy=(0.98, 0.03), xycoords="axes fraction", ha="right",
                fontsize=8, color=INK_2)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("새를 드론으로 잘못 판정한 비율 (1 − 새 recall)", color=INK_2, fontsize=9)
    ax.set_ylabel("드론을 드론으로 맞힌 비율 (드론 recall)", color=INK_2, fontsize=9)
    ax.set_title("최종 test ROC 곡선 (track 단위, 5개 case 합산)", color=INK, fontsize=11, loc="left", pad=10)
    ax.grid(color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_aspect("equal")
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=8, frameon=True, labelcolor=INK)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor(SURFACE)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
