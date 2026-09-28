"""Stage-4 refinement: train motion measurements, grouped CV, frozen validation."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .build_real_sequence_dataset import file_hash
from .calibrate_sequence_simulator import (estimate_acquisition, make_report, real_partition,
                                           review_acceptance, save_cohort, simulated_windows)
from .io import write_json, write_jsonl
from .measure_train_motion import analyze_tracks, train_tracks
from .motion_diagnostics import MotionConfig, duration_group_weights
from .sequence_comparison import bootstrap_delta, compare, group_weights, metric_scales, quantile, score, sequence_mmd
from .sequence_simulator import (BIRDS, QUADS, GuidanceProfile,
                                 SEQUENCE_SIMULATOR_VERSION, latent_flight, observe_flight)
from .trajectory_sequence import SequenceConfig


CANDIDATES = tuple((f"{kernel}_span{span}", kernel, span)
                   for kernel in ("old_ar1", "white", "measured_ar2") for span in (1.3, 2.))


def grouped_folds(metadata, splits=3):
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=20260928)
    for fit, held in splitter.split(np.zeros(len(metadata)), metadata.label, metadata.group_id):
        fit_groups, held_groups = set(metadata.iloc[fit].group_id), set(metadata.iloc[held].group_id)
        if fit_groups & held_groups:
            raise ValueError("CV source-group overlap")
        if set(metadata.iloc[fit].label) != {"bird", "drone"} or set(metadata.iloc[held].label) != {"bird", "drone"}:
            raise ValueError("Each CV partition needs both classes")
        yield fit_groups, held_groups


def fit_motion_priors(tracks, base):
    summary, events, _ = analyze_tracks(tracks)
    parts = [summary[summary.label == c] for c in ("bird", "drone")]
    pooled = pd.concat(parts, ignore_index=True)
    weights = np.concatenate([duration_group_weights(p)*.5 for p in parts])
    median = lambda key: float(quantile(pooled[key].to_numpy(), weights, [.5])[0])
    rho1, rho2 = median("residual_acf1"), median("residual_acf2")
    a2 = float(np.clip((rho2-rho1*rho1)/max(1-rho1*rho1, .05), -.8, .8))
    a1 = float(np.clip(rho1*(1-a2), -.95*(1-a2), .95*(1-a2)))
    sigma = float(np.clip(median("residual_sigma_px")/np.sqrt(2), .1, 2.5))
    averages = {}
    for label, part in zip(("bird", "drone"), parts):
        w = duration_group_weights(part)
        averages[label] = {key: float(np.average(part[key], weights=w))
                           for key in ("low_motion_fraction", "reversal_per_minute", "turn_per_minute")}
    drone, bird = averages["drone"], averages["bird"]
    reverse = float(np.clip(1-np.exp(-drone["reversal_per_minute"]*8/60), .02, .15))
    turn = float(np.clip(1-np.exp(-drone["turn_per_minute"]*8/60), .10, .45))
    hold = float(np.clip(drone["low_motion_fraction"], .03, .10))
    bird_transition = float(np.clip(1-np.exp(-bird["turn_per_minute"]*8/60), .08, .25))
    # Screen event rates constrain a proposal; they do not identify joystick commands.
    dwell = 3.
    if not events.empty:
        intervals = []
        selected = events[(events.label == "drone") & events.kind.isin(["turn", "reversal"])]
        for _, e in selected.groupby(["parent_track_id", "segment"]):
            differences = np.diff(np.sort(e.start_s.to_numpy()))
            intervals.extend(differences[differences > .5].tolist())
        if intervals:
            dwell = float(np.clip(np.median(intervals), 2.5, 4.))
    guidance = GuidanceProfile(drone_probabilities=(reverse, turn, 1-reverse-turn-hold, hold),
                               bird_probabilities=(bird_transition*.6, bird_transition*.4, .25, .05, .70-bird_transition),
                               drone_dwell_seconds=(max(1.8, dwell*.75), min(6., dwell*1.5)),
                               bird_dwell_seconds=(3., 6.))
    records = pd.DataFrame([dict(label=rec["label"], group_id=rec["source_group_id"],
                                 four_three=float(abs(t["processed_height"]/t["processed_width"]-.75) < .02))
                            for rec, t in tracks])
    aspect = float(np.mean([np.average(part.four_three, weights=group_weights(part))
                            for _, part in records.groupby("label")]))
    profile = replace(base, jitter_px=sigma, jitter_ar1=a1, jitter_ar2=a2,
                      jitter_difficulty_gain=0., aspect_4_3_probability=aspect)
    evidence = dict(group_balanced_rates=averages, pooled_residual_sigma_px=sigma,
                    pooled_residual_acf1=rho1, pooled_residual_acf2=rho2,
                    ar1_coefficient=a1, ar2_coefficient=a2, drone_dwell_reference_s=dwell,
                    aspect_4_3_probability=aspect,
                    interpretation="Residual includes true motion; probabilities/dwell are constrained proposals, not identified controls")
    return profile, guidance, evidence, summary, events


def candidate_profile(base, measured, name, kernel, span):
    common = replace(measured, name=name, span_multiplier=span)
    if kernel == "old_ar1":
        return replace(common, jitter_px=base.jitter_px, jitter_ar1=.55, jitter_ar2=0., jitter_difficulty_gain=1.)
    if kernel == "white":
        return replace(common, jitter_ar1=0., jitter_ar2=0.)
    return common


def generate_flights(count, seed_start, guidance):
    flights = []
    for label, subtypes in (("bird", BIRDS), ("drone", QUADS)):
        for i in range(count):
            flights.append(latent_flight(label, subtypes[i % 3], seed_start+i, guidance=guidance))
            if i % 6 == 5 or i == count-1:
                print(f"latent {seed_start} {label}: {i+1}/{count}", flush=True)
    return flights


def load_reference(folder, name):
    with np.load(folder/f"{name}.npz", allow_pickle=False) as archive:
        x = archive["X"].copy()
        meta = pd.read_csv(folder/f"{name}_metrics.csv", keep_default_na=False)
        for column in ("group_id", "sample_id"):
            if not np.array_equal(meta[column], archive[column]):
                raise ValueError("Reference array alignment changed")
    return (x, meta, pd.read_csv(folder/f"{name}_attempts.csv"),
            json.loads((folder/f"{name}_rejections.json").read_text(encoding="utf-8")))


def run(dataset, reference, output, fit_count=18, evaluation_count=24):
    dataset, reference, output = Path(dataset), Path(reference), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder")
    if fit_count < 6 or evaluation_count < 6:
        raise ValueError("Need at least six simulation seeds per class")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    cfg = SequenceConfig(**manifest["config"])
    if cfg != SequenceConfig() or cfg.fingerprint != manifest["contract_id"]:
        raise ValueError("Expected fixed 2-second 30-Hz input contract")
    rebuilt_reference = reference/"reprocess_protocol.json"
    if rebuilt_reference.exists():
        reference_protocol = json.loads(rebuilt_reference.read_text(encoding="utf-8"))
        if reference_protocol["contract_id"] != cfg.fingerprint:
            raise ValueError("Baseline preprocessing contract mismatch")
    train_x, train_meta, train = real_partition(dataset, "train", manifest, cfg)
    original_tracks = list(train_tracks(manifest))
    before_fit = load_reference(reference, "before_fit")
    before_evaluation = load_reference(reference, "before_evaluation")
    prior_summary = json.loads((reference/"results.json").read_text(encoding="utf-8"))
    source = Path(__file__).parent
    (output/"source_snapshot").mkdir()
    code_hashes = {}
    for path in sorted(source.glob("*.py")):
        shutil.copy2(path, output/"source_snapshot"/path.name)
        code_hashes[path.name] = file_hash(path)
    protocol = dict(version=SEQUENCE_SIMULATOR_VERSION, real_contract_id=cfg.fingerprint,
                    real_manifest_sha256=file_hash(dataset/"dataset_manifest.json"),
                    candidates=[dict(name=n, kernel=k, span=s) for n, k, s in CANDIDATES],
                    fold_count=3, fold_seed=20260928, fit_count_per_class=fit_count,
                    evaluation_count_per_class=evaluation_count, fold_seed_start=61000, final_fit_seed_start=81000,
                    evaluation_seed_start=91000, motion_config=asdict(MotionConfig()),
                    cv_policy="Fit all priors on fold-fit source groups; score held-out train groups; choose family by mean objective",
                    validation_policy="Freeze chosen family and full-train fit before reading validation; previous development exposure recorded",
                    input_preparation="Validation arrays prepared with label-blind clock correction before CV; never used for prior fitting or candidate selection",
                    test_opened=False, validation_reused=True, reference=str(reference),
                    reference_hashes={p.name: file_hash(p) for p in reference.glob("before_*") if p.is_file()},
                    code_hashes=code_hashes, group_status=manifest["group_status"],
                    real_sources=[{k: r[k] for k in ("parent_track_id", "source_group_id", "file_sha256")} for r, _ in original_tracks])
    write_json(output/"protocol.json", protocol)
    cv_rows, cv_details, fold_records = [], [], []
    for fold, (fit_groups, held_groups) in enumerate(grouped_folds(train_meta)):
        fit_real, held_real = train[train.group_id.isin(fit_groups)], train[train.group_id.isin(held_groups)]
        subset = dict(manifest, sources=[r for r in manifest["sources"] if r["source_group_id"] in fit_groups and r["split"] == "train"])
        base, _, _ = estimate_acquisition(fit_real, subset)
        tracks = [(r, t) for r, t in original_tracks if r["source_group_id"] in fit_groups]
        measured, guidance, evidence, _, _ = fit_motion_priors(tracks, base)
        scales = metric_scales(fit_real)
        fold_records.append(dict(fold=fold, fit_groups=sorted(fit_groups), held_groups=sorted(held_groups),
                                 evidence=evidence, guidance=asdict(guidance), acquisition=asdict(measured)))
        flights = generate_flights(fit_count, 61000+fold*1000, guidance)
        control = score(compare(held_real, before_fit[1], scales))
        for name, kernel, span in CANDIDATES:
            profile = candidate_profile(base, measured, name, kernel, span)
            bundle = simulated_windows([observe_flight(f, profile) for f in flights], cfg)
            distances = compare(held_real, bundle[1], scales)
            usable = float(bundle[2].usable.mean())
            objective = score(distances)+2*(1-usable)
            cv_rows.append(dict(fold=fold, candidate=name, distance=score(distances), objective=objective,
                                usable_fraction=usable, uncalibrated_control_distance=control))
            distances["fold"], distances["candidate"] = fold, name
            cv_details.append(distances)
            print(f"CV {fold} {name}: distance={score(distances):.4f}, objective={objective:.4f}", flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(output/"cv_scores_train_only.csv", index=False)
    pd.concat(cv_details, ignore_index=True).to_csv(output/"cv_metrics_train_only.csv", index=False)
    write_json(output/"cv_fold_priors.json", fold_records)
    ranking = cv.groupby("candidate")[["distance", "objective", "usable_fraction"]].mean().sort_values("objective")
    ranking.to_csv(output/"cv_ranking.csv")
    selected_name = ranking.index[0]
    _, kernel, span = next(c for c in CANDIDATES if c[0] == selected_name)
    base, acquisition_evidence, _ = estimate_acquisition(train, manifest)
    measured, guidance, evidence, motion, events = fit_motion_priors(original_tracks, base)
    profile = candidate_profile(base, measured, selected_name, kernel, span)
    motion.to_csv(output/"train_motion_segments.csv", index=False)
    events.to_csv(output/"train_motion_events.csv", index=False)
    acquisition_evidence.to_csv(output/"train_acquisition_evidence.csv", index=False)
    train.to_csv(output/"real_train_metrics.csv", index=False)
    frozen = dict(profile=asdict(profile), guidance=asdict(guidance), evidence=evidence,
                  generator_version=SEQUENCE_SIMULATOR_VERSION, contract_id=cfg.fingerprint,
                  selected_before_validation=True, protocol_sha256=file_hash(output/"protocol.json"),
                  fit_scope="Grouped CV within real train; full-train refit before validation",
                  cv_candidate=selected_name, cv_objective=float(ranking.loc[selected_name, "objective"]))
    write_json(output/"selected_profile.json", frozen)
    frozen_hash = file_hash(output/"selected_profile.json")
    fit_flights = generate_flights(fit_count, 81000, guidance)
    write_jsonl(output/"fit_latent_flights.jsonl", fit_flights)
    fit_records = [observe_flight(f, profile) for f in fit_flights]
    after_fit = simulated_windows(fit_records, cfg)
    save_cohort(output, "after_fit", fit_records, after_fit)
    print(f"Frozen {selected_name}. Opening real validation for one evaluation.", flush=True)
    val_x, val_meta, val = real_partition(dataset, "validation", manifest, cfg)
    if set(val_meta.group_id) & set(train_meta.group_id):
        raise ValueError("Real group overlap")
    val.to_csv(output/"real_validation_metrics.csv", index=False)
    evaluation_flights = generate_flights(evaluation_count, 91000, guidance)
    records = [observe_flight(f, profile) for f in evaluation_flights]
    after = simulated_windows(records, cfg)
    save_cohort(output, "after_evaluation", records, after)
    comparisons = []
    for split, real, before, new in (("train", train, before_fit, after_fit), ("validation", val, before_evaluation, after)):
        for phase, bundle in (("before", before), ("after", new)):
            table = compare(real, bundle[1], metric_scales(train))
            table["split"], table["phase"] = split, phase
            comparisons.append(table)
    metrics = pd.concat(comparisons, ignore_index=True)
    metrics.to_csv(output/"comparison.csv", index=False)
    before_val, after_val = comparisons[2], comparisons[3]
    summary = dict(train_before=score(comparisons[0]), train_after=score(comparisons[1]),
                   validation_before=score(before_val), validation_after=score(after_val),
                   validation_coverage_before=float(before_val.coverage.mean()),
                   validation_coverage_after=float(after_val.coverage.mean()),
                   bootstrap=bootstrap_delta(val, before_evaluation[1], after[1], metric_scales(train)),
                   mmd_before=sequence_mmd(val_x, val_meta, before_evaluation[0], before_evaluation[1], train_x, train_meta),
                   mmd_after=sequence_mmd(val_x, val_meta, after[0], after[1], train_x, train_meta),
                   usable_flight_fraction_before=float(before_evaluation[2].usable.mean()),
                   usable_flight_fraction_after=float(after[2].usable.mean()),
                   frozen_profile_sha256=frozen_hash, validation_reused=True, previous_run=str(reference), test_opened=False,
                   real_groups=dict(train=int(train_meta.group_id.nunique()), validation=int(val_meta.group_id.nunique())),
                   candidate_count=len(CANDIDATES), previous_refinement=prior_summary,
                   cv_candidate=selected_name, cv_control_distance=float(cv.uncalibrated_control_distance.mean()),
                   cv_candidate_distance=float(ranking.loc[selected_name, "distance"]))
    if file_hash(output/"selected_profile.json") != frozen_hash:
        raise ValueError("Frozen setting changed after validation")
    if set(after_fit[1].group_id) & set(after[1].group_id):
        raise ValueError("Simulation seed overlap")
    for bundle in (after_fit, after):
        x = bundle[0]
        if x.dtype != np.float32 or x.shape[1:] != (4, 60) or not np.isfinite(x).all():
            raise ValueError("Invalid generated sequence")
        if not np.allclose(x[:, 2:, 1:], np.diff(x[:, :2], axis=2), atol=1e-6):
            raise ValueError("Generated displacement contract mismatch")
    write_json(output/"results.json", summary)
    review = review_acceptance(summary)
    if summary["cv_candidate_distance"] >= summary["cv_control_distance"]:
        review["blockers"].append("grouped_cv_distance_not_improved")
        review["status"] = "not_ready_for_bulk_generation"
    make_report(output, summary, metrics, val_x, val_meta, before_evaluation, after, profile,
                acceptance=review)
    write_json(output/"acceptance_review.json", review)
    write_json(output/"artifact_checks.json", dict(status="passed", test_opened=False,
               shape="N,4,60", dtype="float32", frozen_profile_hash_verified=True,
               train_validation_group_overlap=0, fit_evaluation_group_overlap=0,
               fit_windows=len(after_fit[0]), evaluation_windows=len(after[0])))
    report = output/"report.md"
    text = report.read_text(encoding="utf-8")
    text = text.replace("- 실제 train만 사용해 공통 카메라/관측 프로파일 후보 6개 중 하나를 선택했다.",
                        "- 실제 train 원본 그룹의 3-fold CV로 6개 관측 후보를 비교했다. 각 fold의 보정값과 행동 prior는 fold-fit 그룹만으로 계산했다.")
    text = text.replace("- 보정 전: 기존 BatchRunner v4. 보정 후: 기존 힘 모델 + 명령 전환 + 공통 관측 프로파일.",
                        "- 보정 전: 이전 실행에서 보존한 BatchRunner v4 자료. 보정 후: train 행동 통계 기반 명령 prior + 공통 시간 잡음 후보. 서로 다른 합성 seed이며 동일 비행의 짝 비교는 아니다.")
    text = text.replace("candidate_scores_train_only.csv", "cv_scores_train_only.csv")
    text = text.replace("- 선택 파일을 저장·해시 고정한 다음 실제 validation을 열었다. test.npz와 test 원본 궤적은 열지 않았다.",
                        "- 시간 주기 수정에 따른 validation 전처리는 후보 선택 전에 수행했다. 보정값 계산과 후보 선택에는 사용하지 않았으며, 선택 파일의 해시 고정 후 validation 분포를 평가했다. 실제 test 배열과 원본 궤적은 열지 않았다.")
    text = text.replace("위 명령은 기본 12/24개 비행 설정이다. 원 실행에서 개수를 바꿨다면 protocol.json의 개수를 --fit-count/--evaluation-count로 지정한다.",
                        f"이번 실행은 클래스별 fit {fit_count}개, evaluation {evaluation_count}개 비행이다. 재현 시 protocol.json의 개수를 --fit-count/--evaluation-count로 지정한다.")
    text = text.replace("before/after_*_tracks.jsonl에는 투영 전 3D와 투영 후 관측 및 명령 이력이 함께 있다.",
                        f"보정 전 자료는 {reference.as_posix()}/before_*_tracks.jsonl, 보정 후 자료는 이 폴더의 after_*_tracks.jsonl에 있다. 투영 전 3D, 투영 후 관측, 명령 이력을 보존했다.")
    text = text.replace(".\\venv\\Scripts\\python.exe -m research.calibrate_sequence_simulator --output research/output/sim_to_real_rerun --previous-run " + output.as_posix(),
                        ".\\venv\\Scripts\\python.exe -m research.refine_sequence_calibration --output research/output/sim_to_real_refine_rerun")
    cv_text = ["", "## Train 내부 그룹 검증", "", "| 후보 | CV 거리 | 가용성 포함 점수 | 유효 비행 비율 |", "|---|---:|---:|---:|"]
    for name, r in ranking.iterrows():
        cv_text.append(f"| {name} | {r.distance:.4f} | {r.objective:.4f} | {r.usable_fraction:.3f} |")
    cv_text += ["", f"기존 생성기 CV 거리: {summary['cv_control_distance']:.4f}. CV는 후보 선택에 사용했으며 독립 최종 평가가 아니다.",
                "반전 빈도 등을 명령 확률로 바꾸는 식은 화면 관측을 참고한 제한된 제안이다. 실제 조이스틱 명령을 복원한 것이 아니다.",
                "AR2는 추세 잔차의 pooled 자기상관으로 만든 공통 관측 surrogate다. 조류 전용 주파수나 파형을 주입하지 않는다.",
                "기존 실행 결과와 선택값은 보존했다. C 학습과 대량 합성 증강은 이번 실행 범위에 포함하지 않는다.", ""]
    report.write_text(text+"\n".join(cv_text), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("cv_candidate", "cv_control_distance", "cv_candidate_distance", "train_before", "train_after", "validation_before", "validation_after")}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("research/output/real_sequences_clock_v1_1"))
    parser.add_argument("--reference", type=Path, default=Path("research/output/sim_clock_reference"))
    parser.add_argument("--output", type=Path, default=Path("research/output/sim_to_real_v4"))
    parser.add_argument("--fit-count", type=int, default=18)
    parser.add_argument("--evaluation-count", type=int, default=24)
    args = parser.parse_args()
    run(args.dataset, args.reference, args.output, args.fit_count, args.evaluation_count)


if __name__ == "__main__":
    main()
