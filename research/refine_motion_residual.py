"""One shared motion-residual proposal; validation is gated by train CV."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_coeffs

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import (estimate_acquisition, real_partition,
                                           review_acceptance, save_cohort, simulated_windows)
from .io import write_json
from .measure_train_motion import analyze_tracks, train_tracks
from .motion_diagnostics import duration_group_weights
from .refine_sequence_calibration import (candidate_profile, fit_motion_priors, generate_flights,
                                          grouped_folds, load_reference)
from .sequence_comparison import bootstrap_delta, compare, metric_scales, score, sequence_mmd
from .sequence_simulator import SEQUENCE_SIMULATOR_VERSION, observe_flight
from .trajectory_sequence import SequenceConfig


def residual_gain(rho=.55):
    h = -savgol_coeffs(9, 2)
    h[4] += 1
    covariance = rho**np.abs(np.arange(9)[:, None]-np.arange(9)[None, :])
    return float(np.sqrt(h@covariance@h))


def fit_residual_profile(tracks, base):
    segments, _, results = analyze_tracks(tracks)
    movement = [float(np.median(r["speed"]))*1920/30 for _, r in results]
    sigma = [r["summary"]["residual_sigma_px"]/np.sqrt(2)/residual_gain() for _, r in results]
    segments["step_px"], segments["target_sigma_px"] = movement, sigma
    weights = np.zeros(len(segments))
    for label in ("bird", "drone"):
        mask = segments.label == label
        weights[mask] = .5*duration_group_weights(segments[mask])
    step, target = np.asarray(movement), np.clip(sigma, .02, 2.5)
    fit = least_squares(lambda p: np.sqrt(weights*len(weights))*(np.hypot(p[0], p[1]*step)-target),
                        x0=[.2, .05], bounds=([.02, 0.], [1.5, .25]), loss="soft_l1", f_scale=.25)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Residual fit did not converge")
    profile = replace(base, name="motion_residual", jitter_px=float(fit.x[0]),
                      jitter_motion_fraction=float(fit.x[1]), jitter_difficulty_gain=0.,
                      jitter_ar1=.55, jitter_ar2=0.)
    evidence = dict(floor_px=profile.jitter_px, motion_fraction=profile.jitter_motion_fraction,
                    residual_filter_gain=residual_gain(), fit_groups=int(segments.group_id.nunique()),
                    formula="sigma_px = hypot(floor_px, motion_fraction * optical_step_px)",
                    interpretation="Shared surrogate; residual includes object motion, not identified tracker error")
    return profile, evidence, segments


def cv_gate(table):
    candidate = table[table.variant == "motion_residual"].set_index("fold")
    control = table[table.variant == "control"].set_index("fold")
    if len(candidate) < 3 or set(candidate.index) != set(control.index):
        raise ValueError("Need aligned results from at least three folds")
    if not candidate.index.is_unique or not control.index.is_unique:
        raise ValueError("Duplicate CV fold result")
    if not np.isfinite(table[["distance", "objective", "usable"]].to_numpy()).all():
        raise ValueError("Non-finite CV result")
    checks = dict(mean_distance_improved=bool(candidate.distance.mean() < control.distance.mean()),
                  mean_objective_improved=bool(candidate.objective.mean() < control.objective.mean()),
                  all_folds_distance_improved=bool((candidate.distance < control.distance).all()),
                  mean_usable_not_decreased=bool(candidate.usable.mean() >= control.usable.mean()))
    return dict(passed=all(checks.values()), checks=checks,
                purpose="Conservative development gate fixed before run, not a significance test")


def evaluate_frozen(dataset, manifest, cfg, real_x, real_meta, real, control, candidate,
                    guidance, baseline, output, count, evaluation_count):
    val_x, val_meta, val = real_partition(dataset, "validation", manifest, cfg)
    if set(val_meta.group_id) & set(real_meta.group_id):
        raise ValueError("Real group overlap")
    bundles, comparisons = {}, []
    for split, n, seed, real_part in (("train", count, 81000, real), ("validation", evaluation_count, 91000, val)):
        flights = generate_flights(n, seed, guidance)
        for profile in (control, candidate):
            records = [observe_flight(f, profile) for f in flights]
            bundle = simulated_windows(records, cfg)
            name = f"{profile.name}_{split}"
            save_cohort(output, name, records, bundle)
            bundles[name] = bundle
            detail = compare(real_part, bundle[1], metric_scales(real))
            comparisons.append(detail.assign(split=split, phase=profile.name))
    pd.concat(comparisons, ignore_index=True).to_csv(output/"comparison.csv", index=False)
    before, after = bundles["control_validation"], bundles["motion_residual_validation"]
    summary = dict(train_before=score(comparisons[0]), train_after=score(comparisons[1]),
                   validation_before=score(comparisons[2]), validation_after=score(comparisons[3]),
                   bootstrap=bootstrap_delta(val, before[1], after[1], metric_scales(real)),
                   mmd_before=sequence_mmd(val_x,val_meta,before[0],before[1],real_x,real_meta),
                   mmd_after=sequence_mmd(val_x,val_meta,after[0],after[1],real_x,real_meta),
                   usable_flight_fraction_before=float(before[2].usable.mean()),
                   usable_flight_fraction_after=float(after[2].usable.mean()),
                   validation_reused=True, test_opened=False)
    original = load_reference(baseline, "before_evaluation")
    original_distance = score(compare(val, original[1], metric_scales(real)))
    review = review_acceptance(summary)
    if summary["validation_after"] >= original_distance:
        review["blockers"].append("original_generator_distance_not_improved")
        review["status"] = "not_ready_for_bulk_generation"
    return dict(summary, validation_opened=True, status=review["status"],
                original_generator_validation_distance=original_distance, acceptance=review)


def run(dataset, reference, baseline, output, count=18, evaluation_count=24):
    dataset, reference, baseline, output = map(Path, (dataset, reference, baseline, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    if min(count, evaluation_count) < 6:
        raise ValueError("Need at least six flights per class")
    manifest = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    if cfg != SequenceConfig() or cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Sequence contract mismatch")
    baseline_protocol = json.loads((baseline/"reprocess_protocol.json").read_text(encoding="utf-8"))
    if baseline_protocol["contract_id"] != cfg.fingerprint:
        raise ValueError("Baseline contract mismatch")
    real_x, real_meta, real = real_partition(dataset, "train", manifest, cfg)
    tracks = list(train_tracks(manifest))
    output.mkdir(parents=True, exist_ok=True)
    (output/"source_snapshot").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output/"source_snapshot"/path.name)
    protocol = dict(generator=SEQUENCE_SIMULATOR_VERSION, contract_id=cfg.fingerprint,
                    manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    prior_selected_profile_sha256=file_hash(reference/"selected_profile.json"),
                    variants=["control", "motion_residual"], fit_count_per_class=count,
                    evaluation_count_per_class=evaluation_count, fold_seed_start=61000,
                    final_fit_seed_start=81000, evaluation_seed_start=91000,
                    control="Previous pipeline refitted within each train fold; old_ar1_span2.0",
                    gate="All folds and mean distance improve, mean objective improves, usable fraction not decreased",
                    code_hashes={p.name:file_hash(p) for p in (output/"source_snapshot").glob("*.py")},
                    development_status="Previously inspected train; validation reused only if CV gate passes",
                    validation_policy="No validation arrays read unless train CV gate passes; actual use recorded in results.json",
                    test_opened=False)
    write_json(output/"protocol.json", protocol)
    rows, fold_evidence = [], []
    for fold, (fit_groups, held_groups) in enumerate(grouped_folds(real_meta)):
        fit_real, held_real = real[real.group_id.isin(fit_groups)], real[real.group_id.isin(held_groups)]
        subset = dict(manifest, sources=[r for r in manifest["sources"] if r["split"] == "train" and r["source_group_id"] in fit_groups])
        base, _, _ = estimate_acquisition(fit_real, subset)
        selected_tracks = [(r,t) for r,t in tracks if r["source_group_id"] in fit_groups]
        measured, guidance, _, _, _ = fit_motion_priors(selected_tracks, base)
        control = candidate_profile(base, measured, "control", "old_ar1", 2.)
        candidate, evidence, _ = fit_residual_profile(selected_tracks, control)
        fold_evidence.append(dict(fold=fold, fit_groups=sorted(fit_groups), held_groups=sorted(held_groups),
                                  evidence=evidence, control=asdict(control), candidate=asdict(candidate)))
        flights = generate_flights(count, 61000+fold*1000, guidance)
        for profile in (control, candidate):
            bundle = simulated_windows([observe_flight(f, profile) for f in flights], cfg)
            comparison = compare(held_real, bundle[1], metric_scales(fit_real))
            comparison.to_csv(output/f"fold{fold}_{profile.name}_metrics.csv", index=False)
            usable, distance = float(bundle[2].usable.mean()), score(comparison)
            rows.append(dict(fold=fold, variant=profile.name, distance=distance,
                             usable=usable, objective=distance+2*(1-usable)))
            print(f"fold {fold} {profile.name}: {distance:.4f} usable={usable:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/"cv_scores.csv", index=False)
    write_json(output/"fold_evidence.json", fold_evidence)
    gate = cv_gate(table)
    write_json(output/"cv_gate.json", gate)
    base, _, _ = estimate_acquisition(real, manifest)
    measured, guidance, _, _, _ = fit_motion_priors(tracks, base)
    control = candidate_profile(base, measured, "control", "old_ar1", 2.)
    candidate, evidence, observations = fit_residual_profile(tracks, control)
    observations.to_csv(output/"fit_observations.csv", index=False)
    write_json(output/"selected_profile.json", dict(profile=asdict(candidate), guidance=asdict(guidance),
               evidence=evidence, contract_id=cfg.fingerprint, selected_before_validation=True,
               protocol_sha256=file_hash(output/"protocol.json"), cv_gate=gate))
    frozen_hash = file_hash(output/"selected_profile.json")
    result = dict(cv_gate=gate, validation_opened=False, test_opened=False,
                  status="rejected_by_train_cv", bulk_generation_approved=False,
                  frozen_profile_sha256=frozen_hash)
    if gate["passed"]:
        print("CV gate passed. Evaluating frozen setting on development validation.", flush=True)
        result.update(evaluate_frozen(dataset, manifest, cfg, real_x, real_meta, real, control,
                                     candidate, guidance, baseline, output, count, evaluation_count))
    if file_hash(output/"selected_profile.json") != frozen_hash:
        raise ValueError("Frozen setting changed")
    write_json(output/"results.json", result)
    lines = ["# 이동량 조건부 잔차 모델: 제한된 검증", "",
             "잔차에는 실제 운동도 포함된다. 이 적합은 순수 추적 오차의 식별이 아니다.",
             "클래스 공통 sigma = hypot(정지 잔차, 비례 계수 × 화면 변위)를 사용한다. 행동·카메라·시간 상관은 이전 후보와 동일하다.", "",
             "| fold | 변형 | 거리 | 가용성 포함 점수 | 유효 비행 비율 |", "|---|---|---:|---:|---:|"]
    lines += [f"| {r.fold} | {r.variant} | {r.distance:.4f} | {r.objective:.4f} | {r.usable:.3f} |" for r in table.itertuples()]
    lines += ["", f"CV gate: {gate['passed']}. 검증 조건: {json.dumps(gate['checks'])}",
              f"최종 상태: {result['status']}. validation 열람: {result['validation_opened']}. 실제 test 열람: False.",
              "설정은 selected_profile.json, fold-fit 그룹과 근거는 fold_evidence.json에 있다.",
              "CV는 이미 조사한 개발 train의 재사용이다. 개선되더라도 독립적인 일반화 증명이 아니다.",
              "C 학습·대량 증강은 수행하지 않았다. 원본 그룹의 촬영 세션 독립성은 여전히 잠정 상태다.", "",
              "```powershell", ".\\venv\\Scripts\\python.exe -m research.refine_motion_residual --output research/output/motion_residual_new_run", "```", ""]
    if result["validation_opened"]:
        lines += [f"이전 후보 validation {result['validation_before']:.4f} → 현재 후보 {result['validation_after']:.4f}; 기존 생성기 {result['original_generator_validation_distance']:.4f}.",
                  "판정 상세는 results.json, 지표별 비교는 comparison.csv를 확인한다.", ""]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/real_sequences_clock_v1_1"))
    parser.add_argument("--reference", type=Path, default=Path("research/output/sim_to_real_v4"))
    parser.add_argument("--baseline", type=Path, default=Path("research/output/sim_clock_reference"))
    parser.add_argument("--output", type=Path, default=Path("research/output/motion_residual_v1"))
    parser.add_argument("--fit-count", type=int, default=18)
    parser.add_argument("--evaluation-count", type=int, default=24)
    args = parser.parse_args()
    run(args.dataset,args.reference,args.baseline,args.output,args.fit_count,args.evaluation_count)


if __name__ == "__main__":
    main()
