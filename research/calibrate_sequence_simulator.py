"""Train-only simulator calibration; freeze before opening real validation.

python -m research.calibrate_sequence_simulator --output research/output/sim_to_real_v1
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .build_real_sequence_dataset import file_hash, digest
from .io import write_json, write_jsonl
from .pipeline import BatchRunner, SCENARIOS
from .sequence_comparison import (METRIC_GROUPS, METRICS, bootstrap_delta, compare,
                                  group_weights, metric_scales, quantile, score,
                                  sequence_metrics, sequence_mmd)
from .sequence_simulator import (AcquisitionProfile, BIRDS, QUADS, latent_flight, observe_flight,
                                 SEQUENCE_SIMULATOR_VERSION)
from .trajectory_sequence import SequenceConfig, window_track
from .verify_real_track import load_tracks


def real_partition(folder, split, manifest, cfg):
    if split not in ("train", "validation"):
        raise ValueError("Calibration cannot open real test")
    folder = Path(folder)
    for name in (f"{split}.npz", "metadata.csv"):
        if file_hash(folder/name) != manifest["artifact_hashes"][name]:
            raise ValueError(f"Dataset artifact changed: {name}")
    with np.load(folder/f"{split}.npz", allow_pickle=False) as data:
        x = data["X"].copy()
        ids, labels, groups = data["sample_id"].copy(), data["y"].copy(), data["group_id"].copy()
    meta = pd.read_csv(folder/"metadata.csv", keep_default_na=False)
    meta = meta[meta.split == split].sort_values("npz_row").reset_index(drop=True)
    if not np.array_equal(meta.sample_id, ids) or not np.array_equal(meta.label, labels):
        raise ValueError("Sequence metadata alignment failed")
    if not np.array_equal(meta.source_group_id, groups) or x.shape != (len(meta), 4, cfg.samples):
        raise ValueError("Sequence shape/group alignment failed")
    meta["group_id"] = groups
    if not np.isfinite(x).all():
        raise ValueError("Invalid real input")
    metrics = pd.DataFrame([dict(label=row.label, group_id=row.group_id, sample_id=row.sample_id,
                                **sequence_metrics(x[i], row.normalization_scale))
                            for i, row in meta.iterrows()])
    return x, meta, metrics


def estimate_acquisition(train_metrics, manifest):
    parts = [train_metrics[train_metrics.label == label] for label in ("bird", "drone")]
    pooled = pd.concat(parts, ignore_index=True)
    weights = np.concatenate([group_weights(part)*.5 for part in parts])
    spans = quantile(pooled.screen_span.to_numpy(), weights, [.1, .5, .9])
    spans = np.clip(spans, .01, .65)
    rows, used = [], []
    for rec in manifest["sources"]:
        if rec["split"] != "train" or rec["status"] == "duplicate":
            continue
        path = Path(rec["source_path"])
        if file_hash(path) != rec["file_sha256"]:
            raise ValueError("Real train source changed")
        candidates = [t for t in load_tracks(path) if digest(t["history"]) == rec["history_hash"]]
        if len(candidates) != 1:
            raise ValueError("Cannot identify train track")
        t = candidates[0]
        h = t["history"]
        frames = np.array([p["frame_index"] for p in h])
        xy = np.array([[p["cx"], p["cy"]*t["processed_height"]/t["processed_width"]] for p in h])*1920
        consecutive = (np.diff(frames)[:-1] == 1) & (np.diff(frames)[1:] == 1)
        second = np.diff(xy, n=2, axis=0)[consecutive]
        bound = float(np.median(np.abs(second-np.median(second, axis=0)))*1.4826/np.sqrt(6)) if len(second) else 0.
        missing = 1-len(frames)/(frames[-1]-frames[0]+1)
        rows.append(dict(label=rec["label"], group_id=rec["source_group_id"],
                         dropout=float(missing), burst=float(np.max(np.diff(frames), initial=1) >= 3),
                         residual_bound_px=bound))
        used.append(dict(parent_track_id=rec["parent_track_id"], file_sha256=rec["file_sha256"],
                         source_group_id=rec["source_group_id"]))
    evidence = pd.DataFrame(rows)
    balanced_mean = lambda column: float(np.mean([
        np.average(part[column], weights=group_weights(part))
        for _, part in evidence.groupby("label")]))
    # Second differences also contain true motion; this is only a search bound.
    jitter_bound = float(np.clip(balanced_mean("residual_bound_px"), .15, 2.5))
    profile = AcquisitionProfile(name="train_estimate", span_q10=float(spans[0]),
                                 span_q50=float(spans[1]), span_q90=float(spans[2]),
                                 jitter_px=jitter_bound,
                                 dropout_rate=float(np.clip(balanced_mean("dropout")/1.5, .001, .04)),
                                 burst_probability=float(np.clip(balanced_mean("burst"), 0, .4)))
    return profile, evidence, used


def simulated_windows(records, cfg):
    arrays, metadata, summaries, excluded = [], [], [], []
    for item in records:
        m = item["metadata"]
        group = f"sim:{m['label']}:{m['seed']}"
        try:
            windows, rejects = window_track(item["track"], cfg)
        except (ValueError, KeyError, TypeError) as error:
            windows, rejects = [], [dict(reason="invalid_track", detail=str(error))]
        if m.get("latent_failure"):
            rejects.append(dict(reason="latent_failure", detail=m["latent_failure"]))
            windows = []
        # Same train group cap; eight-second flights otherwise have <=7 windows.
        if len(windows) > 8:
            indices = np.linspace(0, len(windows)-1, 8, dtype=int)
            windows = [windows[i] for i in indices]
        for w in windows:
            sid = group+f":w{w['window_index']}"
            arrays.append(w["X"])
            metadata.append(dict(sample_id=sid, group_id=group, label=m["label"], seed=m["seed"],
                                 normalization_scale=w["normalization_scale"], subtype=m["subtype"],
                                 missing_fraction=w["missing_fraction"], quality_flags=w["quality_flags"],
                                 window_start_s=w["window_start_s"],
                                 **sequence_metrics(w["X"], w["normalization_scale"])))
        observed = len(item["track"]["history"])
        attempted = m["observation"]["attempted_frame_count"]
        summaries.append(dict(group_id=group, label=m["label"], seed=m["seed"], subtype=m["subtype"],
                              usable=bool(windows), windows=len(windows), observed_points=observed,
                              planned_points=attempted, observed_fraction=observed/attempted,
                              latent_failure=m.get("latent_failure"), template=";".join(m.get("template", []))))
        excluded.extend(dict(group_id=group, label=m["label"], **r) for r in rejects)
    x = np.stack(arrays) if arrays else np.empty((0, 4, cfg.samples), np.float32)
    return x, pd.DataFrame(metadata), pd.DataFrame(summaries), excluded


def generate_cohort(count, seed_start):
    baseline, flights = [], []
    runner = BatchRunner(seed=seed_start)
    for label, subtypes in (("bird", BIRDS), ("drone", QUADS)):
        for index in range(count):
            seed = seed_start+index
            subtype = subtypes[index % len(subtypes)]
            old = runner.simulate(SCENARIOS[index % len(SCENARIOS)], label, subtype, index,
                                  noisy=True, frame_count=241)
            old["track"]["stabilization"] = dict(applied=True, method="synthetic_post_cmc_surrogate")
            old["metadata"]["seed"] = seed
            baseline.append(old)
            flights.append(latent_flight(label, subtype, seed))
            if index % 4 == 3 or index == count-1:
                print(f"cohort {seed_start} {label}: {index+1}/{count}", flush=True)
    return baseline, flights


def save_cohort(folder, name, records, bundle):
    x, meta, status, rejects = bundle
    np.savez_compressed(folder/f"{name}.npz", X=x, y=meta.label.to_numpy(dtype="U5"),
                        group_id=meta.group_id.to_numpy(dtype="U40"), sample_id=meta.sample_id.to_numpy(dtype="U50"))
    meta.to_csv(folder/f"{name}_metrics.csv", index=False)
    status.to_csv(folder/f"{name}_attempts.csv", index=False)
    write_json(folder/f"{name}_rejections.json", rejects)
    write_jsonl(folder/f"{name}_tracks.jsonl", records)


def review_acceptance(summary):
    """Conservative development gate; a passing comparison is not real validation."""
    blockers = []
    if summary["validation_after"] >= summary["validation_before"]:
        blockers.append("validation_distribution_distance_not_improved")
    for label in ("bird", "drone"):
        if summary["mmd_after"][label]["mmd2"] > summary["mmd_before"][label]["mmd2"]:
            blockers.append(f"{label}_sequence_mmd_worsened")
    if summary["bootstrap"]["after_minus_before_q025_q50_q975"][2] >= 0:
        blockers.append("bootstrap_does_not_support_consistent_improvement")
    if summary["usable_flight_fraction_after"] < summary["usable_flight_fraction_before"]:
        blockers.append("usable_flight_fraction_decreased")
    return dict(status="not_ready_for_bulk_generation" if blockers else "candidate_for_downstream_ablation",
                bulk_generation_approved=False, blockers=blockers,
                interpretation="Engineering review, not a statistical equivalence test; downstream real evaluation required",
                validation_reused=summary["validation_reused"],
                test_opened=summary["test_opened"])


def calibrate(dataset, output, fit_count=12, evaluation_count=24, bootstrap_draws=200, previous_run=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new empty directory")
    if min(fit_count, evaluation_count) < 6:
        raise ValueError("Need at least six flight seeds per class in each cohort")
    dataset = Path(dataset)
    manifest = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    if cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Real sequence contract mismatch")
    output.mkdir(parents=True)
    train_x, train_meta, train = real_partition(dataset, "train", manifest, cfg)
    train.to_csv(output/"real_train_metrics.csv", index=False)
    scales = metric_scales(train)
    base_profile, evidence, used = estimate_acquisition(train, manifest)
    evidence.to_csv(output/"train_acquisition_evidence.csv", index=False)
    protocol = dict(version=SEQUENCE_SIMULATOR_VERSION, real_dataset_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    real_contract_id=cfg.fingerprint, fit_count_per_class=fit_count,
                    evaluation_count_per_class=evaluation_count, fit_seed_start=41000, evaluation_seed_start=51000,
                    candidate_span_multipliers=[.75, 1.3, 2.], candidate_jitter_bound_fractions=[.35, 1.],
                    objective="equal class/category mean W/train-real scale + 2*(1-usable flight fraction)",
                    metric_groups=METRIC_GROUPS, metric_scales=scales,
                    test_policy="test.npz and test source tracks are never opened",
                    real_train_sources=used, group_status=manifest["group_status"],
                    evaluation_status=manifest["evaluation_status"],
                    base_profile=asdict(base_profile), numpy_version=np.__version__, python=platform.python_version(),
                    previous_run=str(previous_run) if previous_run else None,
                    validation_reused=previous_run is not None,
                    code_hashes={p.name: file_hash(p) for p in [Path(__file__),
                        Path(__file__).with_name("sequence_simulator.py"), Path(__file__).with_name("sequence_comparison.py"),
                        Path(__file__).with_name("dynamics.py"), Path(__file__).with_name("trajectory_sequence.py")]})
    write_json(output/"protocol.json", protocol)
    print("Protocol fixed. Generating before/after training cohort with matching seed IDs.", flush=True)
    before_records, flights = generate_cohort(fit_count, 41000)
    write_jsonl(output/"fit_latent_flights.jsonl", flights)
    before = simulated_windows(before_records, cfg)
    save_cohort(output, "before_fit", before_records, before)
    comparisons, candidates, best = [], [], None
    for span in (.75, 1.3, 2.):
        for noise in (.35, 1.):
            profile = replace(base_profile, name=f"span{span}_noise{noise}", span_multiplier=span,
                              jitter_px=base_profile.jitter_px*noise)
            records = [observe_flight(f, profile) for f in flights]
            bundle = simulated_windows(records, cfg)
            if set(bundle[1].label) != {"bird", "drone"}:
                raise ValueError("Candidate lost a class")
            comparison = compare(train, bundle[1], scales)
            acceptance = float(bundle[2].usable.mean())
            objective = score(comparison)+2*(1-acceptance)
            candidates.append(dict(name=profile.name, objective=objective, distance=score(comparison),
                                   usable_fraction=acceptance, **{k: v for k, v in asdict(profile).items() if k != "name"}))
            comparison["candidate"] = profile.name
            comparisons.append(comparison)
            print(f"candidate {profile.name}: objective={objective:.4f}, usable={acceptance:.3f}", flush=True)
            if best is None or objective < best[0]:
                best = (objective, profile, records, bundle)
    pd.DataFrame(candidates).to_csv(output/"candidate_scores_train_only.csv", index=False)
    pd.concat(comparisons).to_csv(output/"candidate_metrics_train_only.csv", index=False)
    _, selected, after_records, after = best
    frozen = dict(profile=asdict(selected), generator_version=SEQUENCE_SIMULATOR_VERSION,
                  fit_scope="real train only", selected_before_validation=True,
                  contract_id=cfg.fingerprint, protocol_sha256=file_hash(output/"protocol.json"),
                  calibration_evidence="acquisition statistics + train-only distribution search; behavior priors not learned",
                  physical_identifiability="FOV/distance and true movement/tracking error not separately identified")
    write_json(output/"selected_profile.json", frozen)
    frozen_hash = file_hash(output/"selected_profile.json")
    save_cohort(output, "after_fit", after_records, after)
    print(f"Profile frozen ({selected.name}); opening validation once for evaluation.", flush=True)
    val_x, val_meta, val = real_partition(dataset, "validation", manifest, cfg)
    if set(train_meta.group_id) & set(val_meta.group_id):
        raise ValueError("Real source group leakage")
    val.to_csv(output/"real_validation_metrics.csv", index=False)
    before_eval_records, evaluation_flights = generate_cohort(evaluation_count, 51000)
    after_eval_records = [observe_flight(f, selected) for f in evaluation_flights]
    eval_before, eval_after = simulated_windows(before_eval_records, cfg), simulated_windows(after_eval_records, cfg)
    save_cohort(output, "before_evaluation", before_eval_records, eval_before)
    save_cohort(output, "after_evaluation", after_eval_records, eval_after)
    metrics = []
    for split, real, b, a in (("train", train, before, after), ("validation", val, eval_before, eval_after)):
        for phase, bundle in (("before", b), ("after", a)):
            table = compare(real, bundle[1], scales)
            table["split"], table["phase"] = split, phase
            metrics.append(table)
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_csv(output/"comparison.csv", index=False)
    val_before = metrics[(metrics.split == "validation") & (metrics.phase == "before")]
    val_after = metrics[(metrics.split == "validation") & (metrics.phase == "after")]
    summary = dict(train_before=score(metrics[(metrics.split == "train") & (metrics.phase == "before")]),
                   train_after=score(metrics[(metrics.split == "train") & (metrics.phase == "after")]),
                   validation_before=score(val_before), validation_after=score(val_after),
                   validation_coverage_before=float(val_before.coverage.mean()),
                   validation_coverage_after=float(val_after.coverage.mean()),
                   bootstrap=bootstrap_delta(val, eval_before[1], eval_after[1], scales, draws=bootstrap_draws),
                   mmd_before=sequence_mmd(val_x, val_meta, eval_before[0], eval_before[1], train_x, train_meta),
                   mmd_after=sequence_mmd(val_x, val_meta, eval_after[0], eval_after[1], train_x, train_meta),
                   frozen_profile_sha256=frozen_hash, test_opened=False,
                   validation_reused=previous_run is not None,
                   previous_run=str(previous_run) if previous_run else None,
                   real_groups=dict(train=int(train_meta.group_id.nunique()), validation=int(val_meta.group_id.nunique())),
                   candidate_count=len(candidates), simulation_failures_before=int(eval_before[2].latent_failure.notna().sum()),
                   simulation_failures_after=int(eval_after[2].latent_failure.notna().sum()),
                   usable_flight_fraction_before=float(eval_before[2].usable.mean()),
                   usable_flight_fraction_after=float(eval_after[2].usable.mean()))
    if file_hash(output/"selected_profile.json") != frozen_hash:
        raise ValueError("Frozen profile changed during evaluation")
    write_json(output/"results.json", summary)
    make_report(output, summary, metrics, val_x, val_meta, eval_before, eval_after, selected)
    print(json.dumps({k: summary[k] for k in ("train_before", "train_after", "validation_before", "validation_after")}, indent=2), flush=True)
    return summary


def make_report(output, summary, metrics, real_x, real_meta, before, after, profile, acceptance=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    review = review_acceptance(summary) if acceptance is None else acceptance
    write_json(output/"acceptance_review.json", review)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), layout="constrained")
    for ax, label in zip(axes, ("bird", "drone")):
        for j, phase in enumerate(("before", "after")):
            table = metrics[(metrics.label == label) & (metrics.split == "validation") & (metrics.phase == phase)].set_index("metric")
            ax.bar(np.arange(len(METRICS))+(j-.5)*.36, table.loc[list(METRICS), "normalized_distance"], width=.36,
                   color=("#778b9d", "#159581")[j], label=phase)
        ax.set_xticks(np.arange(len(METRICS)), METRICS, rotation=30, ha="right", fontsize=8)
        ax.set(title=f"{label}: real validation vs unseen synthetic seeds", ylabel="Wasserstein / frozen train scale")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output/"validation_distances.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for row, label in enumerate(("bird", "drone")):
        for column, (name, xs, meta) in enumerate((("Real validation", real_x, real_meta),
                                                  ("Before", before[0], before[1]), ("After", after[0], after[1]))):
            ax = axes[row, column]
            # Deterministic first three distinct groups; no visual similarity selection.
            indices = meta[meta.label == label].drop_duplicates("group_id").index[:3]
            for index in indices:
                ax.plot(xs[index, 0], xs[index, 1], alpha=.7, lw=1.)
            ax.set_aspect("equal", adjustable="datalim")
            ax.invert_yaxis()
            ax.set(title=f"{label} / {name}", xlabel="q_x", ylabel="q_y")
            ax.grid(alpha=.2)
    fig.savefig(output/"sequence_comparison.png", dpi=140)
    plt.close(fig)
    lines = ["# 4단계: 실제 A 관측 기반 시뮬레이터 보정", "",
             "이 결과는 개발 자료의 분포 비교다. C 분류 성능, 모든 촬영 조건의 현실성, 물리 파라미터 식별을 입증하지 않는다.", "",
             "## 채택 판단", "",
             f"- 상태: `{review['status']}`. 대량 학습 데이터셋 생성에 승인된 프로파일이 아니다.",
             f"- 검토 사유: {', '.join(review['blockers']) or '분포 비교 이후 C의 실제 자료 평가 필요'}.",
             "- 포함률 상승만으로 성공을 판단하지 않는다. 검토 규칙은 공학적 안전장치이며 현실 동등성의 통계적 증명이 아니다.", "",
             "## 비교 설계", "",
             "- 실제 train만 사용해 공통 카메라/관측 프로파일 후보 6개 중 하나를 선택했다.",
             "- 선택 파일을 저장·해시 고정한 다음 실제 validation을 열었다. test.npz와 test 원본 궤적은 열지 않았다.",
             "- 실제 split과 2초·30Hz·4채널 계약을 유지했다. 양쪽 클래스에 동일한 카메라·잡음 프로파일을 사용했다.",
             "- 보정 전: 기존 BatchRunner v4. 보정 후: 기존 힘 모델 + 명령 전환 + 공통 관측 프로파일.",
             "- 평가 합성 seed는 보정용 seed와 다르다. 클래스·지표 범주별 동일 비중, 각 원본 그룹별 동일 총가중치다.",
             "- 이전 진단에 사용한 자료이며 원본 그룹은 잠정 상태다. validation으로 개선을 확인해도 미사용 최종 평가로 부르지 않는다.", "",
             "## 주요 결과", "", "| 항목 | 보정 전 | 보정 후 |", "|---|---:|---:|"]
    for key, title in (("train", "train 분포 거리"), ("validation", "validation 분포 거리"),
                       ("validation_coverage", "validation 포함률 (지표별 5~95% 평균)")):
        lines.append(f"| {title} | {summary[key+'_before']:.4f} | {summary[key+'_after']:.4f} |")
    lines += [f"| 유효 합성 비행 비율 | {summary['usable_flight_fraction_before']:.3f} | {summary['usable_flight_fraction_after']:.3f} |", "",
              f"- 앞선 실행 이후 validation 반복 사용: {summary['validation_reused']}. 이전 실행: {summary['previous_run']}.",
              f"- 실제 train/validation 원본 그룹: {summary['real_groups']['train']}/{summary['real_groups']['validation']}개.",
              f"- 선택 프로파일: `{profile.name}`. 개선률은 현실 정확도나 분류 정확도가 아니다.",
              f"- 그룹 bootstrap의 보정 후-전 거리 차이 2.5/50/97.5%: {summary['bootstrap']['after_minus_before_q025_q50_q975']}.",
              "- Bootstrap은 고정된 보정값에서 원본 그룹을 재표집한다. 보정값 선택의 불확실성까지 포함하지 않는다.", "",
              "| 클래스 | 시계열 MMD² 전 | 후 |", "|---|---:|---:|"]
    for label in ("bird", "drone"):
        lines.append(f"| {label} | {summary['mmd_before'][label]['mmd2']:.5f} | {summary['mmd_after'][label]['mmd2']:.5f} |")
    lines += ["", "MMD는 4채널 전체 시계열의 RBF 거리이며 낮을수록 유사하다. p-value나 동등성 증명이 아니다.", "",
              "![지표별 validation 비교](validation_distances.png)", "",
              "![시계열 예시](sequence_comparison.png)", "",
              "시계열 그림은 위치·스케일을 정규화한 2초 창(q_x/q_y)이다. 전체 영상의 궤적이나 실제 화면 이동 범위 비교가 아니다.",
              "각 패널은 처음 세 원본 그룹의 첫 창을 표시하며, 잘 맞는 예시를 수작업으로 고르지 않는다. 패널별 축 범위는 다르다.", "",
              "## 개선되지 않은 부분", ""]
    val = metrics[metrics.split == "validation"]
    pivot = val.pivot(index=["label", "metric"], columns="phase", values="normalized_distance")
    worse = pivot[pivot.after > pivot.before].sort_values("after", ascending=False)
    for (label, name), row in worse.iterrows():
        lines.append(f"- {label}/{name}: {row.before:.3f} -> {row.after:.3f}")
    lines += ["", "## 해석과 다음 단계", "",
              "- 새의 추진·활공·선회와 드론의 이동·감속·유지·반전 명령을 한 비행에 연결했다. 실제 GPS에서 학습한 행동 분포는 아니다.",
              "- 순수 활공·추진과 매끄러운 순항, 호버링 템플릿을 남겼다. 중심 좌표에 새 전용 sine 파형을 추가하지 않았다.",
              "- 화면상 크기에 맞춘 유효 투영 비율을 보정했다. 거리와 FOV는 2D 관측만으로 별도로 복원할 수 없다.",
              "- second difference의 강도에는 실제 운동도 섞인다. jitter 검색 상한으로만 사용했고 실제 추적 잡음의 측정값이라고 부르지 않는다.",
              "- drift·CMC 잔차의 발생률은 확인된 정답이 없어 공통 stress prior를 유지했다. 전체 CMC 보정량을 잔차 오차로 취급하지 않았다.",
              "- 유효 창을 조건으로 비교하므로 제외율과 비행별 관측률도 함께 보아야 한다. 전체 지표 평균이 내려가도 모든 지표가 개선된 것은 아니다.",
              "- 5단계 전 C에서 실제 기반 학습 대비 합성 추가의 효과를 동일 실제 평가 그룹으로 확인한다.",
              "- 현재 산출물은 보정 진단 코호트이며 C용 대량 증강 데이터셋이나 학습된 모델이 아니다.", "",
              "## 재현", "",
              "기존 결과를 덮어쓰지 않도록 새 출력 폴더를 지정한다. 원 실행 조건은 protocol.json과 source_snapshot을 확인한다.", "",
              "```powershell",
              ".\\venv\\Scripts\\python.exe -m research.calibrate_sequence_simulator --output research/output/sim_to_real_rerun --previous-run " + output.as_posix(),
              "```", "",
              "위 명령은 기본 12/24개 비행 설정이다. 원 실행에서 개수를 바꿨다면 protocol.json의 개수를 --fit-count/--evaluation-count로 지정한다.", "",
              "설정과 입력 해시는 protocol.json, 후보 점수는 candidate_scores_train_only.csv, 고정 설정은 selected_profile.json에 있다.",
              "before/after_*_tracks.jsonl에는 투영 전 3D와 투영 후 관측 및 명령 이력이 함께 있다.", "",
              "## 참고", "",
              "- [Domain Randomization](https://arxiv.org/abs/1703.06907): 관측 조건 다양화의 근거이며 이 데이터의 전이 성능 근거는 아니다.",
              "- [Gretton et al., A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html): MMD 정의 근거. 여기서는 기술 통계로만 사용한다."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/real_sequences_v1"))
    parser.add_argument("--output", type=Path, default=Path("research/output/sim_to_real_v1"))
    parser.add_argument("--fit-count", type=int, default=12)
    parser.add_argument("--evaluation-count", type=int, default=24)
    parser.add_argument("--previous-run", type=Path, help="Record prior exposure of this validation set")
    args = parser.parse_args()
    calibrate(args.dataset, args.output, args.fit_count, args.evaluation_count, previous_run=args.previous_run)


if __name__ == "__main__":
    main()
