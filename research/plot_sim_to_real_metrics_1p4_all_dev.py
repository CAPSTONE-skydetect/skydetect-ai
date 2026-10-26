"""Compare archived 1.4.0 synthetic and real train+validation windows.

python -m research.plot_sim_to_real_metrics_1p4_all_dev
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .calibrate_sequence_simulator import real_partition
from .io import write_json
from .plot_sim_to_real_metrics_1p4 import LABELS
from .sequence_comparison import METRIC_GROUPS, compare, metric_scales, score
from .trajectory_sequence import SequenceConfig


DESCRIPTIONS = {
    "screen_span": "2초 동안 화면에서 퍼진 범위",
    "screen_path": "프레임 사이 이동거리의 합",
    "screen_speed_median": "프레임별 화면 속도의 중앙값",
    "straightness": "순변위 / 총 이동거리",
    "transverse_ratio": "주 이동 방향 밖으로 퍼진 정도",
    "turn_radians_mean": "연속 이동 방향의 평균 각도 변화",
    "reversal_fraction": "이동 방향이 크게 뒤집힌 비율",
    "step_cv": "프레임 이동거리의 변동계수",
    "low_motion_fraction": "화면 이동이 거의 없는 구간 비율",
    "acf_lag3": "0.1초 간격 이동 변화의 유사성",
    "band_3_10_ratio": "이동 변화의 3~10Hz 에너지 비율",
}


def build(folder, output):
    folder, output = Path(folder), Path(output)
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    if protocol["generator"] != "sequence-guidance-1.4.0":
        raise ValueError("Expected the archived 1.4.0 generator")
    profile = json.loads((folder / "selected_profile.json").read_text(encoding="utf-8"))
    if profile["profile"]["name"] != "ground_span2.0":
        raise ValueError("Expected the archived ground-camera profile")
    real_dir = folder / "real"
    manifest = json.loads((real_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    _, _, train = real_partition(real_dir, "train", manifest, cfg)
    _, _, validation = real_partition(real_dir, "validation", manifest, cfg)
    real = pd.concat([train, validation], ignore_index=True)
    synthetic = pd.concat([
        pd.read_csv(folder / "synthetic_train_metrics.csv"),
        pd.read_csv(folder / "ground_span2.0_validation_metrics.csv"),
    ], ignore_index=True)
    for name, table in (("real", real), ("synthetic", synthetic)):
        if table.sample_id.duplicated().any():
            raise ValueError(f"Duplicate {name} sample ID")
        if set(table.label) != {"bird", "drone"}:
            raise ValueError(f"Both classes required in {name}")
    scales = metric_scales(train)
    result = compare(real, synthetic, scales)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "all_dev_11_metrics.csv", index=False)

    plt.rcParams["font.family"] = "Malgun Gothic" if any(
        "Malgun Gothic" in font.name for font in font_manager.fontManager.ttflist
    ) else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, panels = plt.subplots(3, 2, figsize=(16, 8.7), sharex="col",
                                gridspec_kw={"height_ratios": [3, 4, 4],
                                             "width_ratios": [3, 2]}, layout="constrained")
    colors = {"bird": "#147e92", "drone": "#bf5a46"}
    titles = {"projection": "화면상 이동", "shape": "궤적 형상", "timing": "시간 구조"}
    xmax = max(3.0, float(result.normalized_distance.max()) + .35)
    for (ax, notes), (category, names) in zip(panels, METRIC_GROUPS.items()):
        y = np.arange(len(names), dtype=float)
        for label, offset in (("bird", -.18), ("drone", .18)):
            rows = result[result.label == label].set_index("metric").loc[list(names)]
            values = rows.normalized_distance.to_numpy()
            bars = ax.barh(y + offset, values, height=.32, color=colors[label])
            for bar, value in zip(bars, values):
                ax.text(bar.get_width() + .04, bar.get_y() + bar.get_height() / 2,
                        f"{value:.2f}", va="center", fontsize=9)
        ax.set(yticks=y, yticklabels=[LABELS[name] for name in names], xlim=(0, xmax),
               title=titles[category])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        notes.set(yticks=y, yticklabels=[], xlim=(0, 1), title="지표의 뜻")
        notes.invert_yaxis()
        notes.axis("off")
        for i, name in enumerate(names):
            notes.text(.04, i, DESCRIPTIONS[name], va="center", fontsize=10.5)
    panels[-1, 0].set_xlabel("정규화 Wasserstein 거리 W  |  낮을수록 실제 분포에 가까움")
    panels[0, 0].legend(handles=[Patch(color=colors["bird"], label="새"),
                                Patch(color=colors["drone"], label="드론")],
                        loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Sim-to-Real 진단 지표  |  시뮬레이터 1.4.0 지상 카메라\n"
                 "실제 A train+validation 대 합성 train+validation 2초 궤적", fontsize=15)
    image = output / "sim_to_real_11_metrics_1p4_all_dev_annotated.png"
    fig.savefig(image, dpi=180)
    plt.close(fig)

    summary = dict(simulator="sequence-guidance-1.4.0", profile="ground_span2.0",
                   real_splits=["train", "validation"], synthetic_splits=["train", "validation"],
                   test_included=False, real_windows=len(real), real_groups=int(real.group_id.nunique()),
                   synthetic_windows=len(synthetic),
                   synthetic_flights=int(synthetic.group_id.nunique()),
                   real_by_class={k: int(v) for k, v in real.label.value_counts().items()},
                   synthetic_by_class={k: int(v) for k, v in synthetic.label.value_counts().items()},
                   score=float(score(result)),
                   normalized_using="archived real train only; original metric_scales formula",
                   aggregation="inverse windows per source group/flight, equal class/category weight",
                   interpretation="Descriptive fit to available development data, not independent validation",
                   image=image.name)
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path,
                        default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/output/sim_to_real_metrics_1p4_all_dev"))
    args = parser.parse_args()
    print(json.dumps(build(args.folder, args.output), indent=2, ensure_ascii=False))
