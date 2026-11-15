"""혼동행렬을 한 색(파랑) 농도 이미지로 그린다. 색의 진하기는 실제 클래스 안의 비율."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot(cm, title: str, path: Path) -> None:
    """cm: [[bird→bird, bird→drone], [drone→bird, drone→drone]] (행=실제, 열=예측)."""
    cm = np.asarray(cm)
    rate = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.9), dpi=150)
    ax.imshow(rate, cmap=plt.get_cmap("Blues"), vmin=0, vmax=1.15, extent=(0, 2, 0, 2))
    for i in range(2):
        for j in range(2):
            color = "white" if rate[i, j] > .5 else "#222"
            ax.text(j + .5, 1.58 - i, str(cm[i, j]), ha="center", va="center",
                    fontsize=26, fontweight="bold", color=color)
            ax.text(j + .5, 1.28 - i, f"{rate[i, j] * 100:.0f}%", ha="center", va="center",
                    fontsize=11, color=color)
    ax.set_xticks([.5, 1.5], ["Bird", "Drone"], fontsize=12)
    ax.set_yticks([.5, 1.5], ["Drone", "Bird"], fontsize=12, rotation=90, va="center")
    ax.set_xlabel("Model Prediction", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, pad=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="results.json of c_part_cv5_experiment")
    parser.add_argument("--scenario", default="B")
    parser.add_argument("--eval", default="validation_real")
    parser.add_argument("--level", default="track")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.results.read_text(encoding="utf-8"))["summary"]
    plot(summary[f"{args.scenario}|{args.eval}|t0|{args.level}"]["confusion_matrix_sum"], args.title, args.output)
