"""최종 test 의 case 평균 ROC 곡선 이미지를 만든다 (MultiRocket Stack 단독, track 단위).

c_part_roc_figure.py 는 5개 case 의 track 점수를 합쳐 곡선 하나를 그린다. 이 스크립트는 case 마다 ROC 를 따로 그린 뒤
같은 가로축(1 - 새 recall)에서 보간해 TPR 을 평균한다. 음영은 case 간 ±1 표준편차, 옅은 선은 case 별 곡선이다.
범례 AUC 와 임계값 0 지점의 새/드론 recall 은 case 별 값의 평균이라, 성능 표의 숫자와 같은 기준이다.

저장된 예측(c_part_cv5_stack --final-test 의 pred/caseK_S_<eval>.csv)만 읽는다. 모델을 다시 학습하지 않는다.

    python -m research.c_part_roc_mean_figure --eval test_augmented --out docs/images/c_part/FINAL_test_augmented_roc_track_mean.png
    python -m research.c_part_roc_mean_figure --out docs/images/c_part/FINAL_test_roc_track_mean.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from research.c_part_roc_figure import GRID, INK, INK_2, SURFACE, track_scores

DEFAULT_RESULTS = "research/output/c_part_cv5_FINAL_multirocket"
COLOR = "#eb6834"
N_CASES = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--name", default="MultiRocket Stack")
    parser.add_argument("--eval", default="test_real", choices=["test_real", "test_augmented"],
                        help="test_augmented: 분할 뒤 test 안에서만 만든 증강 포함 세트")
    args = parser.parse_args()
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    grid = np.linspace(0, 1, 501)
    tprs, aucs, fp0s, tp0s = [], [], [], []
    for k in range(N_CASES):
        c = track_scores(args.results, k, args.eval)
        y = (c.label == "drone").to_numpy()
        fpr, tpr, _ = roc_curve(y, c.decision)
        tp = np.interp(grid, fpr, tpr)
        tp[0] = 0.0
        tprs.append(tp)
        aucs.append(roc_auc_score(y, c.decision))
        fp0s.append(np.mean(c.decision[~y] >= 0))  # 임계값 0: 점수 >= 0 이면 드론
        tp0s.append(np.mean(c.decision[y] >= 0))
    tprs = np.array(tprs)
    mean, sd = tprs.mean(0), tprs.std(0, ddof=1)
    bird_recall, drone_recall = 1 - float(np.mean(fp0s)), float(np.mean(tp0s))
    auc_mean, auc_sd = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
    print(f"{args.name}: per-case AUC {np.round(aucs, 3)} mean {auc_mean:.3f} ± {auc_sd:.3f}, "
          f"threshold0 bird recall {bird_recall:.3f} drone recall {drone_recall:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (4, 4)), alpha=0.6, zorder=1)
    ax.text(0.62, 0.55, "무작위 판정", color=INK_2, fontsize=8, rotation=43)
    for t in tprs:
        ax.plot(grid, t, color=COLOR, lw=0.7, alpha=0.25, zorder=2)
    ax.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=COLOR, alpha=0.18, lw=0, zorder=2)
    ax.plot(grid, mean, color=COLOR, lw=2.2, solid_capstyle="round", zorder=3,
            label=f"{args.name} (case 평균 곡선)\nAUC {auc_mean:.3f} ± {auc_sd:.3f}")
    ax.scatter([1 - bird_recall], [drone_recall], s=64, color=COLOR, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.annotate(f"임계값 0 (서비스 판정 지점, case 평균)\n새 재현율 {bird_recall:.3f}\n드론 재현율 {drone_recall:.3f}",
                xy=(1 - bird_recall, drone_recall), xytext=(0.3, 0.6), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))

    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("새를 드론으로 잘못 판정한 비율 (1 − 새 recall)", color=INK_2, fontsize=9)
    ax.set_ylabel("드론을 드론으로 맞힌 비율 (드론 recall)", color=INK_2, fontsize=9)
    what = "test 증강 포함" if args.eval == "test_augmented" else "test"
    ax.set_title(f"최종 {what} ROC 곡선 (track 단위, 5개 case 평균)", color=INK, fontsize=11, loc="left", pad=10)
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
