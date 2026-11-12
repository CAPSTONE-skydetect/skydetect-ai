"""5-case 실험의 val 오판 track 분석: track별 특징표(CSV)와 속도·직진도 산점도 + 대표 궤적 이미지.

    python -m research.c_part_error_analysis --results research/output/c_part_cv5_E0 \
      --data research/output/c_part_cv5_v1 --scenario B --output research/output/c_part_cv5_E0/error_analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def track_table(results: Path, data: Path, scenario: str) -> pd.DataFrame:
    src = pd.read_csv(data / "sources.csv", encoding="utf-8-sig").set_index("parent_track_id")
    rows = []
    for k in range(5):
        p = pd.read_csv(results / "pred" / f"case{k}_{scenario}_validation_real.csv")
        p["track"] = p.sample_id.str.replace(r":\d+:w\d+$", "", regex=True)
        for t, g in p.groupby("track"):
            j = json.loads(Path(src.loc[t].source_path).read_text(encoding="utf-8"))
            h, W, H = j["history"], j["processed_width"], j["processed_height"]
            ms = np.array([q["timestamp_ms"] for q in h])
            x = np.array([q["cx"] for q in h]) * W
            y = np.array([q["cy"] for q in h]) * H
            w = np.array([q.get("w", np.nan) for q in h]) * W
            hh = np.array([q.get("h", np.nan) for q in h]) * H
            step = np.hypot(np.diff(x), np.diff(y))
            rows.append(dict(
                case=k, track=t, label=g.label.iloc[0], group=g.group_id.iloc[0], score=g.decision.mean(),
                n_win=len(g), pos_frac=(g.decision >= 0).mean(), duration_s=(ms[-1] - ms[0]) / 1000,
                box_px=float(np.nanmedian(np.sqrt(w * hh))),
                speed_px_s=float(np.median(step / np.maximum(np.diff(ms) / 1000, 1e-3))),
                straight=float(np.hypot(x[-1] - x[0], y[-1] - y[0]) / max(step.sum(), 1e-6)),
                video=src.loc[t].source_video_id, path=src.loc[t].source_path))
    d = pd.DataFrame(rows)
    d["pred"] = np.where(d.score >= 0, "drone", "bird")
    d["kind"] = d.label + "→" + d.pred
    return d


def draw(d: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(11, 4.6), dpi=140)
    ax = fig.add_subplot(1, 2, 1)
    style = {"bird→bird": ("#e57373", "o", 14, .45, "Bird, correct"),
             "drone→drone": ("#43a047", "o", 14, .35, "Drone, correct"),
             "bird→drone": ("#b71c1c", "X", 60, 1, "Bird → predicted Drone"),
             "drone→bird": ("#1b5e20", "P", 60, 1, "Drone → predicted Bird")}
    for kind, (c, m, s, a, lab) in style.items():
        g = d[d.kind == kind]
        ax.scatter(g.speed_px_s, g.straight, c=c, marker=m, s=s, alpha=a, label=f"{lab} ({len(g)})", edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Median speed (px/s, original frame)")
    ax.set_ylabel("Straightness (net / path length)")
    ax.set_title("Val tracks (5 cases pooled)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.axvline(100, color="#999", lw=.8, ls="--")
    ax.axhline(.85, color="#999", lw=.8, ls="--")
    bird = d[d.label == "bird"]
    slow = (bird.speed_px_s < 100) & (bird.straight < .85)
    ax.text(105, .05, f"slow & curvy birds:\n{(bird[slow].pred == 'drone').mean() * 100:.0f}% misclassified",
            fontsize=8, color="#b71c1c")

    def nearest(kind, speed):
        g = d[d.kind == kind]
        return g.iloc[(g.speed_px_s - speed).abs().argsort().iloc[0]]
    picks = [("Bird → Drone (high score)", d[d.kind == "bird→drone"].sort_values("score").iloc[-2], "#b71c1c"),
             ("Bird, correct (typical)", nearest("bird→bird", bird.speed_px_s.median()), "#e57373"),
             ("Drone, correct (typical)", nearest("drone→drone", d[d.label == "drone"].speed_px_s.median()), "#43a047")]
    for i, (title, row, c) in enumerate(picks):
        j = json.loads(Path(row.path).read_text(encoding="utf-8"))
        W, H = j["processed_width"], j["processed_height"]
        x = np.array([q["cx"] for q in j["history"]]) * W
        y = np.array([q["cy"] for q in j["history"]]) * H
        a = fig.add_subplot(3, 2, 2 * (i + 1))
        a.plot(x - x.mean(), -(y - y.mean()), color=c, lw=1.4)
        a.plot(x[0] - x.mean(), -(y[0] - y.mean()), "o", color=c, ms=4)
        a.set_aspect("equal", adjustable="datalim")
        a.set_title(f"{title}  [{x.max() - x.min():.0f}×{y.max() - y.min():.0f} px, {len(x) / 30:.1f}s]", fontsize=8)
        a.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scenario", default="B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = track_table(args.results, args.data, args.scenario)
    table.drop(columns="path").to_csv(args.output / "val_tracks.csv", index=False, encoding="utf-8-sig")
    draw(table, args.output / "error_analysis.png")
