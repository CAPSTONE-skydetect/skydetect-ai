"""Render 1.6.0 trajectory and screen-span figures matching the 1.4 handoff."""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from .calibrate_sequence_simulator import simulated_windows
from .evaluate_sequence_handoff import load_arrays, read_json
from .io import write_json
from .sequence_comparison import group_weights, quantile, sequence_metrics
from .sequence_simulator import AcquisitionProfile, GuidanceProfile, latent_flight, observe_flight
from .trajectory_sequence import SequenceConfig


def _representative(data, meta, label):
    ids = np.flatnonzero(data["y"] == label)
    spans = np.array([sequence_metrics(data["X"][i],
                      float(meta.iloc[i].normalization_scale))["screen_span"] for i in ids])
    return int(ids[np.argsort(spans)[len(ids)//2]])


def _current_window(meta, profile, guidance, sample_id):
    row = meta[meta.sample_id == sample_id]
    if len(row) != 1:
        raise ValueError("Selected synthetic sample is not unique")
    row = row.iloc[0]
    flight = latent_flight(row.label, row.subtype, int(row.seed), guidance=guidance)
    record = observe_flight(flight, profile)
    x, regenerated, _, _ = simulated_windows([record], SequenceConfig())
    index = regenerated.index[regenerated.sample_id == sample_id]
    if len(index) != 1 or not np.isclose(
            regenerated.loc[index[0], "screen_span"], row.screen_span, atol=1e-10):
        raise ValueError("Regenerated example differs from 1.6.0 audit")
    return x[index[0]]


def render(folder, audit, output):
    folder, audit, output = Path(folder), Path(audit), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    real = load_arrays(folder/"real/train.npz")
    real_meta = pd.read_csv(folder/"real/metadata.csv", keep_default_na=False)
    real_meta = real_meta[real_meta.split == "train"].sort_values("npz_row").reset_index(drop=True)
    real_meta["group_id"] = real_meta.source_group_id
    real_meta["screen_span"] = [sequence_metrics(real["X"][i],
                               float(row.normalization_scale))["screen_span"]
                                for i, row in real_meta.iterrows()]
    current_meta = pd.read_csv(audit/"synthetic_train_current_window_metrics.csv", keep_default_na=False)
    selected = read_json(folder/"selected_profile.json")
    profile = replace(AcquisitionProfile.from_saved(selected["profile"]),
                      name="paired_50_500m", min_camera_distance_m=50., max_camera_distance_m=500.)
    guidance = GuidanceProfile(**selected["guidance"])
    plt.rcParams["font.family"] = "Malgun Gothic" if any(
        "Malgun Gothic" in font.name for font in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    examples = []
    for row, label in enumerate(("bird", "drone")):
        real_index = _representative(real, real_meta, label)
        real_xy = real["X"][real_index, :2]
        real_scale = float(real_meta.iloc[real_index].normalization_scale)
        synthetic_ids = np.flatnonzero(current_meta.label.to_numpy() == label)
        synthetic_spans = current_meta.iloc[synthetic_ids].screen_span.to_numpy()
        synthetic_index = int(synthetic_ids[np.argsort(synthetic_spans)[len(synthetic_ids)//2]])
        synthetic_row = current_meta.iloc[synthetic_index]
        synthetic_xy = _current_window(current_meta, profile, guidance, synthetic_row.sample_id)[:2]
        synthetic_scale = float(synthetic_row.normalization_scale)
        examples.extend([dict(label=label, domain="real_train", sample_id=real_meta.iloc[real_index].sample_id,
                              screen_span=float(real_meta.iloc[real_index].normalization_scale*
                                                np.linalg.norm(np.ptp(real_xy, axis=1)))),
                         dict(label=label, domain="synthetic_1.6.0", sample_id=synthetic_row.sample_id,
                              screen_span=float(synthetic_row.screen_span))])
        for col, (xy, scale, name, color) in enumerate((
                (real_xy, real_scale, "Real train", "#147e92"),
                (synthetic_xy, synthetic_scale, "Ground-camera synthetic 1.6.0", "#bf5a46"))):
            for ax, points, limits, suffix in (
                    (axes[row, col*2], xy*scale, (-.5, .5), "screen scale"),
                    (axes[row, col*2+1], xy, (-2, 2), "normalized shape")):
                ax.plot(points[0], points[1], lw=1, color=color)
                ax.scatter(points[0, 0], points[1, 0], s=20, color="black")
                ax.set(xlim=limits, ylim=limits[::-1], aspect="equal",
                       title=f"{label}: {name}\n{suffix}")
                ax.grid(alpha=.2)
    fig.suptitle("Two-second median-span examples; centered for comparison, not original frame position")
    trajectory = output/"trajectory_examples_1p6.png"
    fig.savefig(trajectory, dpi=160)
    plt.close(fig)

    medians = {}
    for label in ("bird", "drone"):
        medians[label] = {}
        for name, meta in (("real_A", real_meta), ("synthetic_1.6.0", current_meta)):
            part = meta[meta.label == label]
            medians[label][name] = float(quantile(part.screen_span.to_numpy(),
                                                  group_weights(part), [.5])[0])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    x = np.arange(2)
    for offset, key, caption, color in ((-.16, "real_A", "실제 A", "#147e92"),
                                        (.16, "synthetic_1.6.0", "합성 1.6.0", "#bf5a46")):
        values = [100*medians[label][key] for label in ("bird", "drone")]
        bars = ax.bar(x+offset, values, width=.28, label=caption, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.45,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set(xticks=x, xticklabels=["새", "드론"], ylim=(0, 32),
           title="화면 이동 폭 중앙값\n2초 창 · 화면 너비 대비 %")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#d2d5d8")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(output/"screen_span_medians_1p6.png", dpi=180)
    plt.close(fig)
    summary = dict(simulator="sequence-guidance-1.6.0", camera_range_m=[50, 500],
                   dataset="synthetic_train seed 131000, real train from 20260928 handoff",
                   example_selection="Median observed 2-second screen-span window per class/domain",
                   screen_span_group_weighted_medians=medians, examples=examples,
                   screen_panels_centered=True, original_frame_position_shown=False)
    write_json(output/"figure_manifest_1p6.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path,
                        default=Path("research/output/sequence_handoff_20260928"))
    parser.add_argument("--audit", type=Path,
                        default=Path("research/output/paired_1p4_vs_1p6_20261005"))
    parser.add_argument("--output", type=Path,
                        default=Path("research/output/paired_1p4_vs_1p6_20261005/figures"))
    args = parser.parse_args()
    render(args.folder, args.audit, args.output)
