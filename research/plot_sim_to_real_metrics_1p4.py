"""Plot the archived 1.4.0 ground-camera validation diagnostics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .io import write_json
from .sequence_comparison import METRIC_GROUPS, score


LABELS = {
    "screen_span": "screen_span  |  화면 이동 폭",
    "screen_path": "screen_path  |  총 이동 경로",
    "screen_speed_median": "screen_speed_median  |  겉보기 속도",
    "straightness": "straightness  |  직진성",
    "transverse_ratio": "transverse_ratio  |  옆방향 퍼짐",
    "turn_radians_mean": "turn_radians_mean  |  회전 각도",
    "reversal_fraction": "reversal_fraction  |  급격한 반전",
    "step_cv": "step_cv  |  프레임 이동량 변동",
    "low_motion_fraction": "low_motion_fraction  |  저이동 비율",
    "acf_lag3": "acf_lag3  |  0.1초 자기상관",
    "band_3_10_ratio": "band_3_10_ratio  |  3~10Hz 에너지",
}


def plot(source, output):
    source, output = Path(source), Path(output)
    frame = pd.read_csv(source)
    frame = frame[frame.variant == "ground_span2.0"]
    expected = {(label, metric) for label in ("bird", "drone")
                for names in METRIC_GROUPS.values() for metric in names}
    if (set(zip(frame.label, frame.metric)) != expected or len(frame) != len(expected)
            or not np.isclose(score(frame), .903640113771727, atol=1e-9)):
        raise ValueError("Archived 1.4.0 validation metrics changed")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Malgun Gothic" if any(
        "Malgun Gothic" in font.name for font in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 4, 4]}, layout="constrained")
    colors = {"bird": "#147e92", "drone": "#bf5a46"}
    titles = {"projection": "화면상 이동", "shape": "궤적 형상", "timing": "시간 구조"}
    for ax, (category, names) in zip(axes, METRIC_GROUPS.items()):
        y = np.arange(len(names), dtype=float)
        for label, offset in (("bird", -.18), ("drone", .18)):
            rows = frame[frame.label == label].set_index("metric").loc[list(names)]
            values = rows.normalized_distance.to_numpy()
            bars = ax.barh(y+offset, values, height=.32, color=colors[label])
            for bar, value in zip(bars, values):
                ax.text(bar.get_width()+.04, bar.get_y()+bar.get_height()/2,
                        f"{value:.2f}", va="center", fontsize=9)
        ax.set(yticks=y, yticklabels=[LABELS[name] for name in names], xlim=(0, 3.3),
               title=titles[category])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("정규화 Wasserstein 거리 W  |  낮을수록 실제 분포에 가까움")
    axes[0].legend(handles=[Patch(color=colors["bird"], label="새"),
                            Patch(color=colors["drone"], label="드론")],
                   loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Sim-to-Real 진단 지표  |  시뮬레이터 1.4.0 지상 카메라\n"
                 "실제 A validation 대 합성 2초 궤적", fontsize=15)
    image = output/"sim_to_real_11_metrics_1p4.png"
    fig.savefig(image, dpi=180)
    plt.close(fig)
    write_json(output/"figure_manifest.json", dict(
        simulator="sequence-guidance-1.4.0", variant="ground_span2.0",
        real_split="validation", source=str(source), metrics=len(METRIC_GROUPS["projection"])
        +len(METRIC_GROUPS["shape"])+len(METRIC_GROUPS["timing"]),
        score=float(score(frame)), interpretation="Lower is closer, not classification accuracy",
        image=image.name))
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("research/output/sequence_handoff_20260928/camera_validation.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/output/sim_to_real_metrics_1p4"))
    args = parser.parse_args()
    print(plot(args.source, args.output))
