"""Measure real train only; export event overlays, residuals and sensitivity."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import digest, file_hash
from .io import write_json
from .motion_diagnostics import (MOTION_DIAGNOSTIC_VERSION, MotionConfig, analyze_segment,
                                 duration_group_weights, resample_segments)
from .sequence_comparison import quantile
from .trajectory_sequence import CONTRACT_VERSION, SequenceConfig
from .verify_real_track import load_tracks


def train_tracks(manifest, groups=None):
    for rec in manifest["sources"]:
        if rec["split"] != "train" or rec["status"] != "accepted":
            continue
        if groups is not None and rec["source_group_id"] not in groups:
            continue
        path = Path(rec["source_path"])
        if file_hash(path) != rec["file_sha256"]:
            raise ValueError(f"Train source changed: {path.name}")
        tracks = [t for t in load_tracks(path) if digest(t["history"]) == rec["history_hash"]]
        if len(tracks) != 1:
            raise ValueError("Cannot identify real train track")
        yield rec, tracks[0]


def analyze_tracks(tracks, config=None):
    cfg = config or MotionConfig()
    summaries, events, results = [], [], []
    for rec, track in tracks:
        for index, segment in enumerate(resample_segments(track)):
            result = analyze_segment(segment["time"], segment["points"], cfg, segment["interpolated"])
            identity = dict(label=rec["label"], group_id=rec["source_group_id"],
                            parent_track_id=rec["parent_track_id"], file_name=rec["file_name"], segment=index)
            summaries.append(dict(identity, **result["summary"]))
            events.extend(dict(identity, **e) for e in result["events"])
            results.append((identity, result))
    return pd.DataFrame(summaries), pd.DataFrame(events), results


def plot_track(output, rec, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"low_motion": "#db9a22", "reversal": "#d54e4e", "turn": "#16856e"}
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), layout="constrained")
    for identity, r in results:
        axes[0].plot(r["points"][:, 0]*1920, r["points"][:, 1]*1920, color="#8292a2", lw=.8)
        axes[0].plot(r["trend"][:, 0]*1920, r["trend"][:, 1]*1920, color="#2f6588", lw=1.1)
        axes[1].plot(r["time"], r["speed"]*1920, color="#2f6588")
        axes[1].plot(r["time"], np.full(len(r["time"]), r["threshold"]*1920), color="#db9a22", linestyle="--")
        heading = np.degrees(r["heading"]).copy()
        heading[1:][np.abs(np.diff(heading)) > 180] = np.nan
        axes[2].plot(r["time"], heading, color="#2f6588")
        axes[3].plot(r["time"], r["residual"][:, 0]*1920, color="#2f6588", alpha=.8)
        axes[3].plot(r["time"], r["residual"][:, 1]*1920, color="#b25870", alpha=.8)
        for event in r["events"]:
            chosen = (r["time"] >= event["start_s"]) & (r["time"] < event["end_s"])
            axes[0].plot(r["points"][chosen, 0]*1920, r["points"][chosen, 1]*1920,
                         color=colors[event["kind"]], lw=2)
            for ax in axes[1:]:
                ax.axvspan(event["start_s"], event["end_s"], color=colors[event["kind"]], alpha=.12)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(title=f"{rec['label']} / {rec['parent_track_id']} (train)", ylabel="canonical y (px)", xlabel="canonical x (px)")
    axes[1].set(ylabel="screen speed (px/s)")
    axes[2].set(ylabel="supported heading (deg)", ylim=(-190, 190))
    axes[3].set(ylabel="trend residual (px)", xlabel="A timestamp (s)")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([0], [0], color=color, label=kind) for kind, color in colors.items()], fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/f"{rec['parent_track_id']}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def measure(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    output.mkdir(parents=True, exist_ok=True)
    (output/"tracks").mkdir()
    manifest = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = MotionConfig()
    tracks = list(train_tracks(manifest))
    summary, events, results = analyze_tracks(tracks, cfg)
    summary.to_csv(output/"segments.csv", index=False)
    events.to_csv(output/"events.csv", index=False)
    for rec, _ in tracks:
        matching = [(identity, r) for identity, r in results if identity["parent_track_id"] == rec["parent_track_id"]]
        if matching:
            plot_track(output/"tracks", rec, matching)
    sensitivity = []
    for name, c in (("base", cfg), ("short_context", replace(cfg, trend_seconds=.2, context_seconds=.2, reverse_angle_deg=120)),
                    ("long_context", replace(cfg, trend_seconds=.45, context_seconds=.45, reverse_angle_deg=150))):
        s, es, _ = analyze_tracks(tracks, c)
        for label in ("bird", "drone"):
            part = s[s.label == label]
            weights = duration_group_weights(part)
            sensitivity.append(dict(setting=name, label=label, groups=int(part.group_id.nunique()),
                                    low_motion_fraction=float(np.average(part.low_motion_fraction, weights=weights)),
                                    reversal_per_minute=float(np.average(part.reversal_per_minute, weights=weights)),
                                    turn_per_minute=float(np.average(part.turn_per_minute, weights=weights)),
                                    residual_sigma_px_median=float(quantile(part.residual_sigma_px.to_numpy(), weights, [.5])[0])))
    pd.DataFrame(sensitivity).to_csv(output/"sensitivity.csv", index=False)
    protocol = dict(version=MOTION_DIAGNOSTIC_VERSION, config=asdict(cfg),
                    sequence_contract_version=CONTRACT_VERSION, sequence_contract_id=SequenceConfig().fingerprint,
                    scope="real train only; full eligible track segments, not group-capped two-second windows",
                    real_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    source_groups=summary.group_id.nunique(), source_tracks=len(tracks),
                    test_or_validation_sources_opened=False,
                    sources=[{k: rec[k] for k in ("parent_track_id", "source_group_id", "file_sha256", "history_hash")} for rec, _ in tracks],
                    interpretation="Screen events and trend residuals include true motion, projection, tracking and CMC effects",
                    code_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                                 ("measure_train_motion.py", "motion_diagnostics.py", "trajectory_sequence.py")})
    write_json(output/"protocol.json", protocol)
    lines = ["# 실제 train의 화면상 움직임 진단", "",
             "정지·반전·선회는 화면상의 후보 구간이다. 실제 공중 정지나 조이스틱 명령 정답이 아니다.", "",
             f"train {len(tracks)}개 궤적, {summary.group_id.nunique()}개 원본 그룹만 읽었다. validation/test 원본은 읽지 않았다.", "",
             "## 판정", "",
             "- 0.1초보다 긴 누락을 넘겨 연결하지 않는다. 더 짧은 누락과 분석 구간 양끝은 방향 판정에서 제외한다.",
             "- 0.3초 Savitzky-Golay 추세와 국소 변위로 방향을 구한다. 저이동 기준은 고정 하한과 잔차 강도를 함께 사용한다.",
             "- 135도 이상 방향 변화와 이동 지지가 있는 반전 후보, 방향이 연속해서 변하는 선회 후보를 기록한다.",
             "- 잔차에는 실제 운동도 섞인다. 이를 순수 A 잡음이나 날갯짓으로 해석하지 않는다.",
             "- 2초 입력과 달리 전체 train의 관측 가능한 구간을 진단한다. 짧은 창에서 놓친 행동도 볼 수 있다.", "",
             "## 기준 민감도", "", "| 설정 | 클래스 | 저이동 비율 | 반전/분 | 선회/분 | 잔차 sigma(px) |", "|---|---|---:|---:|---:|---:|"]
    for row in sensitivity:
        lines.append(f"| {row['setting']} | {row['label']} | {row['low_motion_fraction']:.3f} | {row['reversal_per_minute']:.2f} | {row['turn_per_minute']:.2f} | {row['residual_sigma_px_median']:.3f} |")
    lines += ["", "## 검토 파일", "", "- events.csv: 구간 시작·끝·지속시간과 방향 변화각.",
              "- segments.csv: 구간별 이동 범위, 저이동·반전·선회 빈도, 잔차와 자기상관.",
              "- tracks/: 원본별 화면 경로, 속도, 방향, 잔차 그래프. 노랑=저이동, 빨강=반전, 초록=선회.",
              "- 기준에 따라 빈도가 달라지면 명령 분포를 직접 추정하는 데 사용하지 않는다. 원본 overlay 표본 검토가 필요하다.", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(sensitivity).to_string(index=False), flush=True)
    return summary, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/real_sequences_v1"))
    parser.add_argument("--output", type=Path, default=Path("research/output/train_motion_v1"))
    args = parser.parse_args()
    measure(args.dataset, args.output)


if __name__ == "__main__":
    main()
