"""Train-only paired observation ablations and source-evidence review."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .build_real_sequence_dataset import file_hash, group_records
from .calibrate_sequence_simulator import real_partition, simulated_windows
from .io import read_jsonl, write_json
from .measure_train_motion import train_tracks
from .refine_sequence_calibration import load_reference
from .sequence_comparison import compare, metric_scales, score, sequence_metrics
from .sequence_simulator import AcquisitionProfile, observe_flight
from .trajectory_sequence import SequenceConfig, normalize_window


def optical_at_observed_frames(record):
    """Remove position errors, retaining exactly the observed frame/gap pattern."""
    result = deepcopy(record)
    optical = {p["frame_index"]: p for p in record["optical_truth"] if p is not None}
    result["track"]["history"] = [deepcopy(optical[p["frame_index"]])
                                     for p in record["track"]["history"]]
    return result


def temporal_view(x, meta, coarse=False):
    rows, arrays = [], []
    for i, row in meta.reset_index(drop=True).iterrows():
        points = x[i, :2].T.astype(float)*row.normalization_scale
        if coarse:
            points = savgol_filter(points, 9, 2, axis=0)
        normalized, info = normalize_window(points, SequenceConfig())
        arrays.append(normalized)
        rows.append(dict(sample_id=row.sample_id, group_id=row.group_id, label=row.label,
                         normalization_scale=info["normalization_scale"],
                         **sequence_metrics(normalized, info["normalization_scale"])))
    return np.stack(arrays), pd.DataFrame(rows)


def common_windows(bundles):
    ids = set.intersection(*(set(b[1].sample_id) for b in bundles.values()))
    if not ids:
        raise ValueError("No common windows for paired ablation")
    result = {}
    for name, (x, meta, *_) in bundles.items():
        indices = meta.reset_index().set_index("sample_id").loc[sorted(ids), "index"].to_numpy()
        result[name] = (x[indices], meta.iloc[indices].reset_index(drop=True))
    return result


def source_evidence(manifest):
    # Only metadata is inspected across splits; no validation/test files are read.
    records = deepcopy(manifest["sources"])
    links = group_records(records)
    lookup = {r["parent_track_id"]: r for r in records}
    crossing = [link for link in links if lookup[link["left"]]["split"] != lookup[link["right"]]["split"]]
    train = [r for r in records if r["split"] == "train" and r["status"] == "accepted"]
    return dict(metadata_tracks=len(records), train_tracks=len(train),
                cross_split_evidence_links=crossing, evidence_links=links,
                train_without_video_hash=sum(not r.get("video_sha256") for r in train),
                session_independence="unverified; source IDs identify uploads, not recording sessions",
                validation_test_coordinates_opened=False)


def overlay_contact(path, destination):
    import cv2
    import matplotlib.pyplot as plt
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count < 2:
            return False
        frames = []
        for fraction in (.35, .85):
            index = int((count-1)*fraction)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                return False
            frames.append((index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
        for ax, (index, frame) in zip(axes, frames):
            ax.imshow(frame)
            ax.set_title(f"Existing A overlay: frame {index}")
            ax.axis("off")
        fig.savefig(destination, dpi=110)
        plt.close(fig)
        return True
    finally:
        capture.release()


def case_plots(output, tracks, real_x, real_meta, bundles, audit_path):
    import matplotlib.pyplot as plt
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig")) if audit_path.exists() else []
    matches = {(r["file_sha256"], r["history_hash"]): r for r in audit}
    cases = []
    folder = output/"cases"
    folder.mkdir()
    observed_x, observed_meta = bundles["observed"][:2]
    optical_x, optical_meta = bundles["optical_same_frames"][:2]
    for rec, track in tracks:
        available = real_meta.index[real_meta.parent_track_id == rec["parent_track_id"]]
        if not len(available):
            continue
        index = int(available[0])
        sim_indices = observed_meta.index[observed_meta.label == rec["label"]]
        # A fixed cyclic choice avoids selecting examples for visual similarity.
        sim_index = int(sim_indices[len(cases) % len(sim_indices)])
        sid = observed_meta.iloc[sim_index].sample_id
        oi = optical_meta.index[optical_meta.sample_id == sid]
        if not len(oi):
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        for column, (title, x) in enumerate((("Real train", real_x[index]),
                                           ("Same flight: optical", optical_x[int(oi[0])]),
                                           ("Same flight: observed", observed_x[sim_index]))):
            axes[0, column].plot(x[0], x[1], lw=1)
            axes[0, column].scatter(x[0, 0], x[1, 0], s=18)
            axes[0, column].set_aspect("equal", adjustable="box")
            axes[0, column].invert_yaxis()
            axes[0, column].set(title=title, xlabel="q_x", ylabel="q_y")
            axes[1, column].plot(np.arange(60)/30, x[2], label="d_x", lw=1)
            axes[1, column].plot(np.arange(60)/30, x[3], label="d_y", lw=1)
            axes[1, column].set(xlabel="seconds", ylabel="normalized displacement")
            axes[1, column].legend()
        fig.suptitle(f"{rec['label']} / {rec['parent_track_id']} / first exported window")
        fig.savefig(folder/f"{rec['parent_track_id']}.png", dpi=120)
        plt.close(fig)
        old = matches.get((rec["file_sha256"], rec["history_hash"]), {})
        paths = [Path(p)/"overlay.mp4" for p in old.get("matched_run_paths", [])]
        video = next((p for p in paths if p.is_file()), None)
        contact = bool(video and overlay_contact(video, folder/f"{rec['parent_track_id']}_overlay.png"))
        cases.append(dict(parent_track_id=rec["parent_track_id"], label=rec["label"],
                          file_name=rec["file_name"], source_group_id=rec["source_group_id"],
                          real_sample_id=real_meta.iloc[index].sample_id, synthetic_sample_id=sid,
                          overlay_path=str(video) if video else "", contact_created=contact,
                          session_group="", session_review="needs_acquisition_provenance"))
    pd.DataFrame(cases).to_csv(output/"case_index.csv", index=False, encoding="utf-8-sig")
    return cases


def run(dataset, reference, output, audit_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dataset, reference, output = map(Path, (dataset, reference, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    manifest = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    if cfg != SequenceConfig() or manifest["contract_id"] != cfg.fingerprint:
        raise ValueError("Expected current fixed sequence contract")
    real_x, real_meta, _ = real_partition(dataset, "train", manifest, cfg)
    tracks = list(train_tracks(manifest))
    evidence = source_evidence(manifest)
    if evidence["cross_split_evidence_links"]:
        raise ValueError("Source evidence crosses dataset splits")
    selected = json.loads((reference/"selected_profile.json").read_text(encoding="utf-8"))
    if selected["contract_id"] != cfg.fingerprint:
        raise ValueError("Saved simulator input contract mismatch")
    profile = AcquisitionProfile(**selected["profile"])
    flights = list(read_jsonl(reference/"fit_latent_flights.jsonl"))
    records = list(read_jsonl(reference/"after_fit_tracks.jsonl"))
    output.mkdir(parents=True, exist_ok=True)
    (output/"source_snapshot").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output/"source_snapshot"/path.name)
    protocol = dict(contract_id=cfg.fingerprint, scope="real train and saved synthetic fit only",
                    real_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    reference=str(reference), reference_hashes={name: file_hash(reference/name) for name in
                        ("selected_profile.json", "fit_latent_flights.jsonl", "after_fit_tracks.jsonl")},
                    test_opened=False, validation_opened=False,
                    coarse_view="9-sample quadratic SG applied equally to real and synthetic; diagnostic only",
                    ablation_policy="Same physical flights and camera seeds; observation RNG consumption can change",
                    code_hashes={p.name: file_hash(p) for p in (output/"source_snapshot").glob("*.py")})
    write_json(output/"protocol.json", protocol)
    write_json(output/"source_evidence.json", evidence)
    bundles = dict(observed=load_reference(reference, "after_fit"),
                   optical_same_frames=simulated_windows([optical_at_observed_frames(r) for r in records], cfg))
    for name, settings in (("no_jitter", dict(jitter_px=0)),
                           ("no_drift", dict(drift_probability=0)),
                           ("no_cmc", dict(camera_probability=0)),
                           ("no_dropout", dict(dropout_rate=0, burst_probability=0))):
        bundles[name] = simulated_windows([observe_flight(f, replace(profile, **settings)) for f in flights], cfg)
    paired = common_windows(bundles)
    tables, summary = [], []
    for coarse in (False, True):
        _, real = temporal_view(real_x, real_meta, coarse)
        scales = metric_scales(real)
        for name, (x, meta) in paired.items():
            _, sim = temporal_view(x, meta, coarse)
            table = compare(real, sim, scales)
            table["variant"], table["view"] = name, "coarse" if coarse else "raw"
            tables.append(table)
            summary.append(dict(variant=name, view=table.view.iloc[0], distance=score(table),
                                common_windows=len(x), groups=int(meta.group_id.nunique())))
    comparison = pd.concat(tables, ignore_index=True)
    comparison.to_csv(output/"component_comparison.csv", index=False)
    pd.DataFrame(summary).to_csv(output/"component_scores.csv", index=False)
    # Keep every real group visible; a few tails must not hide behind an average.
    by_group = real_meta.copy()
    metrics = pd.DataFrame([sequence_metrics(real_x[i], r.normalization_scale) for i, r in real_meta.iterrows()])
    pd.concat([by_group, metrics], axis=1).to_csv(output/"real_train_windows.csv", index=False)
    sim_meta = bundles["observed"][1].copy()
    template = {f"sim:{r['metadata']['label']}:{r['metadata']['seed']}": ";".join(r["metadata"]["template"]) for r in records}
    sim_meta["template"] = sim_meta.group_id.map(template)
    sim_meta.to_csv(output/"synthetic_fit_windows.csv", index=False)
    geometry = pd.DataFrame([dict(label=r["metadata"]["label"], seed=r["metadata"]["seed"],
                                 camera_height_m=r["metadata"]["camera"]["position"][2],
                                 distance_m=r["metadata"]["camera_distance_m"],
                                 fov_deg=r["metadata"]["camera"]["horizontal_fov_deg"])
                             for r in records])
    geometry.to_csv(output/"camera_geometry.csv", index=False)
    cases = case_plots(output, tracks, real_x, real_meta, bundles, Path(audit_path))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    for ax, label in zip(axes, ("bird", "drone")):
        table = comparison[(comparison.label == label) & (comparison.view == "raw")]
        pivot = table.pivot(index="metric", columns="variant", values="normalized_distance")
        pivot[["observed", "optical_same_frames", "no_jitter"]].plot.bar(ax=ax, rot=30)
        ax.set(title=f"{label}: paired synthetic windows vs real TRAIN", ylabel="W / frozen train scale")
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(output/"observation_components.png", dpi=130)
    plt.close(fig)
    lines = ["# Train-only 원인 분리 진단", "",
             "실제 train만 분석했다. 이번 결과로 validation/test 성능을 주장하지 않는다.", "",
             "같은 물리 비행에서 중심 잡음·drift·CMC·누락을 각각 끈다. 관측 RNG 소비가 달라질 수 있어 고립된 인과 추정은 아니다.",
             "optical_same_frames는 실제 생성된 관측과 정확히 같은 frame/gap을 유지하고 위치 오차만 제거한다.",
             "모든 변형에서 함께 통과한 창만 비교한다. 이 조건부 분석은 전체 비행 가용성을 평가하지 않는다.",
             "coarse는 양쪽에 0.3초 추세 처리를 적용한 진단이며 C 입력을 바꾸지 않는다. raw/coarse 수치는 각 train 척도가 달라 직접 비교하지 않는다.", "",
             "| 관측 변형 | 원래 입력 거리 | 완만한 추세 거리 |", "|---|---:|---:|"]
    scores = pd.DataFrame(summary).pivot(index="variant", columns="view", values="distance")
    for name, row in scores.iterrows():
        lines.append(f"| {name} | {row.raw:.4f} | {row.coarse:.4f} |")
    lines += ["", f"공통 창 {len(next(iter(paired.values()))[0])}개. 실제 원본 {len(cases)}개 비교 그림을 저장했다.",
              f"연결된 기존 overlay의 접촉시트 {sum(r['contact_created'] for r in cases)}개. overlay는 raw 화면 좌표일 수 있으며 B 입력은 안정화 export다.",
              f"metadata상 split을 가로지르는 동일 source/hash/파일군 근거 {len(evidence['cross_split_evidence_links'])}개.",
              f"train 중 원본 영상 해시가 없는 궤적 {evidence['train_without_video_hash']}개. 촬영 세션 독립성은 확정할 수 없다.",
              f"가상 지면 z=0보다 낮은 카메라 {int((geometry.camera_height_m < 0).sum())}/{len(geometry)}개. 현재 카메라가 실제 설치 위치를 복원한 것으로 해석할 수 없다.",
              "case_index.csv의 session_group/session_review는 실제 촬영 출처를 확인할 때 기록할 공간이다.", "",
              "![관측 성분 비교](observation_components.png)", "",
              "각 case 그림은 실제 원본의 첫 export 창과 고정 순서의 합성 창이다. 닮은 예시를 찾아 고르지 않았다."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(scores=summary, real_tracks=len(cases), overlay_contacts=sum(r["contact_created"] for r in cases)), indent=2))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/real_sequences_clock_v1_1"))
    parser.add_argument("--reference", type=Path, default=Path("research/output/sim_to_real_v4"))
    parser.add_argument("--output", type=Path, default=Path("research/output/sequence_gap_diagnosis_v1"))
    parser.add_argument("--audit", type=Path, default=Path("research/output/a_preprocessing_audit/manifest.json"))
    args = parser.parse_args()
    run(args.dataset, args.reference, args.output, args.audit)


if __name__ == "__main__":
    main()
