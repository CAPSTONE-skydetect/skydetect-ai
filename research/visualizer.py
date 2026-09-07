"""Full-range trajectory and feature plots; no fixed 0..5 feature clipping."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .io import read_jsonl


def plot_sample(sample, path):
    meta = sample["metadata"]
    world = sample["world_truth"]
    observed = sample["track"]["history"]
    optical = [p for p in sample["optical_truth"] if p is not None]
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    ax = fig.add_subplot(221, projection="3d")
    pos = np.array([p["position_m"] for p in world])
    if len(pos):
        ax.plot(*pos.T, color="#00796b")
        ax.scatter(*pos[0], color="#d81b60", s=35)
    ax.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="altitude (m)", title="Latent flight")
    ax = fig.add_subplot(222)
    if optical:
        ax.plot([p["cx"] for p in optical], [p["cy"] for p in optical], label="Optical", color="#00796b")
    if observed:
        ax.scatter([p["cx"] for p in observed], [p["cy"] for p in observed], s=4, label="Observed", color="#d81b60")
    ax.set(xlabel="cx / image width", ylabel="cy / image height", xlim=(0, 1), ylim=(1, 0), title="Fixed camera")
    ax.set_aspect(meta["camera"]["height"]/meta["camera"]["width"])
    ax.legend()
    ax = fig.add_subplot(223)
    if observed:
        time = np.array([p["timestamp_ms"] for p in observed])/1000
        ax.plot(time, [p["w"]*meta["camera"]["width"] for p in observed], label="Tracked width")
    if optical:
        ax.plot(np.array([p["timestamp_ms"] for p in optical])/1000,
                [p["w"]*meta["camera"]["width"] for p in optical], label="Optical width")
    ax.set(xlabel="seconds", ylabel="pixels", title="BBox depth / ROI response")
    ax.legend()
    ax = fig.add_subplot(224)
    if world:
        ax.plot([p["time_seconds"] for p in world], [np.linalg.norm(p["velocity_m_s"]) for p in world], label="Ground speed")
        ax.plot([p["time_seconds"] for p in world], [p["speed_scale"]*meta["individual_parameters"]["cruise_speed_m_s"] for p in world],
                linestyle="--", label="Commanded cruise scale")
    ax.set(xlabel="seconds", ylabel="m/s", title="Motion, independent of observation noise")
    ax.legend()
    fig.suptitle(f'{meta["label"]} / {meta["subtype"]} / {meta["scenario"]} / {meta["behavior_mode"]}\n'
                 f'{meta["observation_profile"]}, accepted points {len(observed)}/{meta["frame_count"]}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_distributions(table, path):
    valid = table[table.feature_status == "accepted"]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for ax, feature in zip(axes.flat, FEATURE_COLUMNS):
        for label, color in (("bird", "#00796b"), ("drone", "#d81b60")):
            values = np.sort(valid.loc[valid.label == label, feature].to_numpy())
            if len(values):
                ax.step(values, np.arange(1, len(values)+1)/len(values), label=label, color=color)
        ax.set(title=feature, ylabel="ECDF")
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Synthetic distributions only; separation does not establish realism")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    table = pd.read_csv(args.dataset_dir / "simulation_features_v3.csv")
    plot_distributions(table, args.dataset_dir / "plots" / "distributions.png")
    for i, sample in enumerate(read_jsonl(args.dataset_dir / "raw_trajectories_v3.jsonl")):
        if i >= args.samples:
            break
        plot_sample(sample, args.dataset_dir / "plots" / f"sample_{i:03d}.png")


if __name__ == "__main__":
    main()
